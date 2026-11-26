"""Processo worker: extração → planejamento → transcrição chunk a chunk, com retomada e escada de fallback.

Roda num processo separado (multiprocessing spawn). Comunica-se com a UI apenas por uma fila de eventos
(kind, payload) e um Event de cancelamento. Todo o estado durável fica no journal SQLite.
"""
from __future__ import annotations

import gc
import json
import logging
import os
import re
import time
import traceback
from pathlib import Path

import numpy as np

from app.pipeline import ffmpeg_tools
from app.pipeline.ffmpeg_tools import AudioError, Cancelled
from app.pipeline.hallucination import clean_segments
from app.pipeline.journal import Heartbeat, Journal
from app.pipeline.planner import SR
from app.pipeline.segmenter import resegment

log = logging.getLogger("whisperdesk.worker")

OOM_PATTERNS = re.compile(
    r"out of memory|CUBLAS_STATUS_ALLOC_FAILED|CUDNN_STATUS_ALLOC_FAILED|cudaErrorMemoryAllocation|"
    r"CUDA_ERROR_OUT_OF_MEMORY|std::bad_alloc", re.I)
GPU_RUNTIME_PATTERNS = re.compile(
    r"cublas|cudnn|cudart|nvrtc|CUDA driver|no CUDA-capable device|Library .* is not found|"
    r"CUDA failed with error|CUDA_ERROR|unsupported compute type|device kernel image is invalid", re.I)

PROGRESS_INTERVAL_S = 0.4
MAX_CONSECUTIVE_CHUNK_FAILURES = 3


class Engine:
    """Mantém exatamente um modelo carregado e desce uma escada determinística sob OOM.

    Degrau = (device, compute_type, batch_size, beam_size); batch_size=0 => decodificação sequencial.
    O modelo nunca muda; só precisão, batch e dispositivo."""

    def __init__(self, model_path: str, model_size: str, cfg: dict, prefer_gpu: bool, emit):
        self.model_path = model_path
        self.model_size = model_size
        self.emit = emit
        self.beam = max(1, int(cfg.get("beam_size") or 5))
        self.vad = bool(cfg.get("vad_filter", True))
        t0 = float(cfg.get("temperature") or 0.0)
        self.temperatures = tuple(sorted({round(t0, 2), *[t for t in (0.2, 0.4, 0.6, 0.8, 1.0) if t > t0]}))
        self.ladder = self._build_ladder(prefer_gpu, cfg.get("compute_type") or "float16")
        self.rung = 0
        self._model = None
        self._batched = None
        self._loaded_key = None

    # ---------- escada ----------
    def _build_ladder(self, prefer_gpu: bool, wanted_compute: str) -> list[tuple]:
        b = self.beam
        cpu = [("cpu", "int8", 0, b)]
        if not prefer_gpu:
            return cpu
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() < 1:
                self.emit("log", "Nenhuma GPU CUDA utilizável detectada; usando CPU.")
                return cpu
            types = set(ctranslate2.get_supported_compute_types("cuda"))
        except Exception as e:  # runtime CUDA ausente/quebrado
            self.emit("log", f"Runtime CUDA indisponível ({e}); usando CPU.")
            return cpu

        rungs: list[tuple] = []
        if "float16" in types and wanted_compute == "float16":
            rungs += [("cuda", "float16", 16, b), ("cuda", "float16", 8, b)]
        if "int8_float16" in types and wanted_compute in ("float16", "int8_float16"):
            rungs += [("cuda", "int8_float16", 4, b), ("cuda", "int8_float16", 0, b), ("cuda", "int8_float16", 0, 1)]
        elif "int8" in types:
            rungs += [("cuda", "int8", 4, b), ("cuda", "int8", 0, b), ("cuda", "int8", 0, 1)]
        elif "float32" in types:
            rungs += [("cuda", "float32", 4, b), ("cuda", "float32", 0, 1)]
        if not self.vad:  # o pipeline batched exige VAD: só degraus sequenciais
            seen, seq = set(), []
            for d, c, _bt, bm in rungs:
                if (d, c, bm) not in seen:
                    seen.add((d, c, bm))
                    seq.append((d, c, 0, bm))
            rungs = seq
        rungs += cpu

        # Pre-flight: pula degraus que claramente não cabem na VRAM livre.
        free = _free_vram_mb()
        if free is not None:
            from app.pipeline.models import estimate_vram_mb

            fitting = [r for r in rungs if r[0] == "cpu" or estimate_vram_mb(self.model_size, r[1], r[2]) <= free * 0.95]
            if fitting and fitting[0] != rungs[0]:
                self.emit("log", f"VRAM livre {free} MB: iniciando em {fitting[0][1]}/batch {fitting[0][2]}.")
            rungs = fitting or cpu
        return rungs

    @property
    def step(self) -> tuple:
        return self.ladder[self.rung]

    def describe(self) -> str:
        d, c, bt, bm = self.step
        mode = f"batch {bt}" if bt else "sequencial"
        return f"{d.upper()} · {c} · {mode} · beam {bm}"

    def _unload(self):
        self._batched = None
        if self._model is not None:
            try:
                self._model.model.unload_model()
            except Exception:
                pass
        self._model = None
        self._loaded_key = None
        gc.collect()

    def close(self):
        self._unload()

    def _ensure_loaded(self):
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        device, compute, batch, _beam = self.step
        key = (device, compute)
        if self._loaded_key == key:
            if batch and self._batched is None:
                self._batched = BatchedInferencePipeline(model=self._model)
            return
        self._unload()  # libera ANTES de alocar o próximo
        self.emit("log", f"Carregando modelo {self.model_size} em {device}/{compute}…")
        t0 = time.monotonic()
        self._model = WhisperModel(
            self.model_path,
            device=device,
            compute_type=compute,
            cpu_threads=max(1, (os.cpu_count() or 4) - 2) if device == "cpu" else 0,
            num_workers=1,
        )
        self._batched = BatchedInferencePipeline(model=self._model) if batch else None
        self._loaded_key = key
        self.emit("engine", {"device": device, "compute_type": compute, "batch": batch,
                             "label": self.describe(), "load_s": round(time.monotonic() - t0, 1)})

    def degrade(self, reason: str) -> bool:
        if self.rung + 1 >= len(self.ladder):
            return False
        self.rung += 1
        self._unload()
        self.emit("degraded", {"reason": reason, "label": self.describe(), "device": self.step[0]})
        return True

    def degrade_to_cpu(self, reason: str) -> bool:
        if self.step[0] == "cpu":
            return False
        while self.step[0] != "cpu" and self.rung + 1 < len(self.ladder):
            self.rung += 1
        self._unload()
        self.emit("degraded", {"reason": reason, "label": self.describe(), "device": self.step[0]})
        return True

    def transcribe(self, audio: np.ndarray, language, task, initial_prompt, on_seg, cancel_event):
        self._ensure_loaded()
        _device, _compute, batch, beam = self.step
        common = dict(
            language=language,
            task=task,
            beam_size=beam,
            word_timestamps=True,
            temperature=self.temperatures,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            initial_prompt=initial_prompt,
        )
        if batch:
            segments, info = self._batched.transcribe(
                audio, batch_size=batch, vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200), **common)
        else:
            segments, info = self._model.transcribe(
                audio,
                condition_on_previous_text=False,
                hallucination_silence_threshold=2.0,
                vad_filter=self.vad,
                vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200) if self.vad else None,
                **common,
            )
        out = []
        for s in segments:  # o gerador é consumido aqui, dentro do try de OOM do chamador
            out.append(s)
            on_seg(s)
            if cancel_event is not None and cancel_event.is_set():
                raise Cancelled()
        return out, info


def _free_vram_mb() -> int | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            return int(pynvml.nvmlDeviceGetMemoryInfo(h).free // (1 << 20))
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None


def owned_segments(raw_segments, start_sample: int, end_sample: int, decode_start: int) -> list[dict]:
    """Converte para ms globais (a partir de amostras inteiras) e aplica a posse por ponto médio."""
    out = []
    for s in raw_segments:
        words = []
        for w in (s.words or []):
            g_start = decode_start + int(round(w.start * SR))
            g_end = decode_start + int(round(w.end * SR))
            if g_end < g_start:
                g_end = g_start
            mid = (g_start + g_end) // 2
            if start_sample <= mid < end_sample:
                words.append({
                    "start_ms": g_start * 1000 // SR,
                    "end_ms": g_end * 1000 // SR,
                    "word": w.word,
                    "prob": float(w.probability),
                })
        if not words:
            continue
        text = "".join(w["word"] for w in words).strip()
        if not text:
            continue
        out.append({
            "start_ms": words[0]["start_ms"],
            "end_ms": max(words[-1]["end_ms"], words[0]["start_ms"] + 1),
            "text": text,
            "words": words,
            "avg_logprob": float(s.avg_logprob),
            "no_speech_prob": float(s.no_speech_prob),
            "compression_ratio": float(s.compression_ratio),
        })
    return out


def _setup_logging(log_dir: Path, job_id: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_dir / f"worker-{job_id}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO)
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


def _fmt_s(samples: int) -> str:
    s = samples // SR
    return f"{s // 3600:d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def run_job(job_id: str, db_path: str, models_dir: str, log_dir: str, events, cancel_event):
    """Ponto de entrada do processo worker."""
    from app import runtime

    runtime.prepare()
    _setup_logging(Path(log_dir), job_id)

    def emit(kind, payload=None):
        if kind == "log":
            log.info("%s", payload)
        try:
            events.put((kind, payload))
        except Exception:
            pass

    j = Journal(Path(db_path))
    pid = os.getpid()
    if not j.claim(job_id, pid):
        emit("error", {"code": "JOB_LOCKED", "message": "Este job já está sendo processado por outra instância."})
        j.close()
        return

    try:
        with ffmpeg_tools.keep_system_awake(), Heartbeat(db_path, job_id, pid):
            _run(j, job_id, Path(models_dir), emit, cancel_event)
    except (Cancelled, InterruptedError):
        j.set_state(job_id, "PAUSED")
        emit("paused")
    except AudioError as e:
        log.exception("AudioError")
        j.set_state(job_id, "FAILED", error=f"{e.code}: {e}")
        emit("error", {"code": e.code, "message": str(e)})
    except Exception as e:
        tb = traceback.format_exc()
        log.error("Falha no job:\n%s", tb)
        j.set_state(job_id, "FAILED", error=tb[-4000:])
        emit("error", {"code": type(e).__name__, "message": str(e)})
    finally:
        j.release(job_id, pid)
        j.close()


def _run(j: Journal, job_id: str, models_dir: Path, emit, cancel_event) -> None:
    import soundfile as sf

    from app.pipeline.models import NO_TRANSLATE, ensure_model
    from app.pipeline.planner import plan_chunks

    job = j.job(job_id)
    cfg = json.loads(job["config_json"])
    workdir = j.workdir(job_id)

    if job["state"] in ("DONE",):
        emit("done", {"failed_chunks": 0})
        return
    if job["state"] == "FAILED":
        j.set_state(job_id, "CREATED")
        job = j.job(job_id)

    # ---- EXTRACTING ----
    wav_ok = bool(job["wav_path"]) and Path(job["wav_path"]).is_file() and bool(job["total_samples"])
    if job["state"] in ("CREATED", "EXTRACTING") or not wav_ok:
        j.set_state(job_id, "EXTRACTING")
        emit("state", "EXTRACTING")
        info = ffmpeg_tools.probe(job["src_path"])
        stream = next((s for s in info.audio_streams if s.index == job["audio_stream"]), None)
        if stream is None:
            raise AudioError("STREAM_MISSING", f"A trilha de áudio #{job['audio_stream']} não existe mais no arquivo.")
        wav = workdir / "audio.wav"
        last = [0.0]

        def _p(frac):
            now = time.monotonic()
            if now - last[0] >= PROGRESS_INTERVAL_S or frac >= 1.0:
                last[0] = now
                emit("extract_progress", frac)

        emit("log", f"Extraindo áudio ({stream.describe()})…")
        t0 = time.monotonic()
        warning = ffmpeg_tools.extract_wav(job["src_path"], wav, stream, info.duration,
                                           cancel_event=cancel_event, on_progress=_p,
                                           log_path=workdir / "ffmpeg.log")
        if warning:
            j.add_warning(job_id, warning)
            emit("warning", warning)
        total = sf.info(str(wav)).frames
        emit("log", f"Áudio extraído em {time.monotonic() - t0:.1f}s ({_fmt_s(total)}).")
        j.set_state(job_id, "PLANNING", wav_path=str(wav), total_samples=total, media_duration=info.duration)
        job = j.job(job_id)

    # ---- PLANNING ----
    if job["state"] == "PLANNING":
        emit("state", "PLANNING")
        plans = plan_chunks(job["wav_path"], job["total_samples"],
                            on_progress=lambda f: emit("plan_progress", f))
        j.insert_plan(job_id, plans)
        forced = sum(1 for p in plans if p.forced)
        emit("log", f"Plano: {len(plans)} trecho(s)" + (f", {forced} corte(s) sem pausa (com sobreposição)." if forced else "."))
        j.set_state(job_id, "TRANSCRIBING")
        job = j.job(job_id)

    # ---- TRANSCRIBING ----
    if j.reset_running(job_id):
        emit("log", "Retomando a partir do último trecho salvo.")
    if job["state"] != "TRANSCRIBING":
        j.set_state(job_id, "TRANSCRIBING")
    emit("state", "TRANSCRIBING")

    pending = j.pending_chunks(job_id)
    total_samples = int(job["total_samples"])
    task = cfg.get("task", "transcribe")
    size = cfg["model_size"]
    if task == "translate" and size in NO_TRANSLATE:
        msg = f"O modelo {size} não suporta tradução; o resultado ficará no idioma original."
        j.add_warning(job_id, msg)
        emit("warning", msg)

    engine = None
    if pending:
        emit("state", "LOADING_MODEL")

        def _dlp(done, total):
            emit("download_progress", {"done": done, "total": total})

        model_path = ensure_model(size, models_dir, on_progress=_dlp, cancel_event=cancel_event)
        engine = Engine(model_path, size, cfg, prefer_gpu=(cfg.get("device") == "cuda"), emit=emit)
        emit("state", "TRANSCRIBING")

    language = cfg.get("language") or job["detected_language"]
    session_audio, session_elapsed = 0.0, 0.0
    consecutive_failures = 0
    try:
        for chunk in pending:
            if cancel_event.is_set():
                raise Cancelled()
            idx = chunk["idx"]
            j.mark_running(job_id, idx)
            decode_start = chunk["start_sample"] - chunk["pad_left"]
            decode_end = chunk["end_sample"] + chunk["pad_right"]
            audio, _ = sf.read(job["wav_path"], start=decode_start, stop=decode_end, dtype="float32",
                               always_2d=False)
            chunk_len = chunk["end_sample"] - chunk["start_sample"]
            base_done = j.done_samples(job_id)

            tail_text, tail_lp = j.previous_tail(job_id, idx)
            prompt = tail_text if (tail_text and tail_lp is not None and tail_lp > -0.5) else None

            t0 = time.monotonic()
            last_emit = [0.0]

            def on_seg(s, _len=len(audio), _base=base_done, _clen=chunk_len, _t0=t0):
                now = time.monotonic()
                if now - last_emit[0] < PROGRESS_INTERVAL_S:
                    return
                last_emit[0] = now
                frac = min(1.0, (s.end * SR) / max(1, _len))
                done = _base + frac * _clen
                el = session_elapsed + (now - _t0)
                emit("progress", {"fraction": done / total_samples,
                                  "rtf": (session_audio + frac * _clen / SR) / el if el > 0 else 0.0})

            try:
                while True:
                    try:
                        raw, info = engine.transcribe(audio, language, task, prompt, on_seg, cancel_event)
                        break
                    except Cancelled:
                        raise
                    except (RuntimeError, MemoryError, OSError, ValueError) as e:
                        msg = str(e)
                        log.warning("Erro no trecho %d em %s: %s", idx, engine.describe(), msg)
                        if isinstance(e, MemoryError) or OOM_PATTERNS.search(msg):
                            if engine.degrade(f"memória insuficiente ({msg[:120]})"):
                                continue
                        elif engine.step[0] == "cuda" and GPU_RUNTIME_PATTERNS.search(msg):
                            if engine.degrade_to_cpu(f"falha do runtime da GPU ({msg[:160]})"):
                                continue
                        raise
            except Cancelled:
                j.mark_pending(job_id, idx)
                raise
            except Exception as e:
                consecutive_failures += 1
                j.mark_failed(job_id, idx, traceback.format_exc())
                warn = (f"Trecho {_fmt_s(chunk['start_sample'])}–{_fmt_s(chunk['end_sample'])} falhou: "
                        f"{type(e).__name__}: {str(e)[:200]}")
                emit("warning", warn)
                if consecutive_failures >= MAX_CONSECUTIVE_CHUNK_FAILURES:
                    raise
                continue
            consecutive_failures = 0

            if not language and info is not None and getattr(info, "language", None):
                if (getattr(info, "language_probability", 0) or 0) >= 0.6:
                    language = info.language
                    j.update_fields(job_id, detected_language=language)
                    emit("language", {"code": language, "prob": round(info.language_probability, 3)})

            segments = resegment(clean_segments(
                owned_segments(raw, chunk["start_sample"], chunk["end_sample"], decode_start), language=language))
            elapsed = time.monotonic() - t0
            last = segments[-1] if segments else None
            device, compute, batch, _ = engine.step
            j.commit_chunk(job_id, idx, segments, {
                "device": device, "compute_type": compute, "batch_size": batch,
                "audio_s": chunk_len / SR,
                "elapsed_s": elapsed,
                "tail_text": last["text"][-200:] if last else None,
                "tail_logprob": last["avg_logprob"] if last else None,
            })
            session_audio += chunk_len / SR
            session_elapsed += elapsed
            done, total = j.progress(job_id)
            emit("chunk_done", {
                "idx": idx, "done": done, "total": total,
                "fraction": j.done_samples(job_id) / total_samples,
                "rtf": session_audio / session_elapsed if session_elapsed > 0 else 0.0,
                "segments": [{**{k: v for k, v in s.items() if k != "words"},
                              "words": [(w["start_ms"], w["end_ms"], w["word"], w["prob"]) for w in s["words"]]}
                             for s in segments],
            })
            del audio, raw, segments
    finally:
        if engine is not None:
            engine.close()

    if cancel_event.is_set():
        raise Cancelled()

    # ---- FINALIZAÇÃO ----
    failed = j.failed_chunks(job_id)
    for c in failed:
        j.add_warning(job_id, f"Trecho {_fmt_s(c['start_sample'])}–{_fmt_s(c['end_sample'])} não foi transcrito.")
    j.set_state(job_id, "DONE")
    stats = j.stats(job_id)
    try:
        wav = Path(job["wav_path"]) if job["wav_path"] else None
        if wav and wav.is_file():
            wav.unlink()  # o resultado vive no journal; o WAV de trabalho pode ter vários GB
    except OSError:
        pass
    emit("done", {"failed_chunks": len(failed), "language": language, **stats})
