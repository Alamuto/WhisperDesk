"""Transcrição ao vivo do microfone, num processo separado (spawn).

A GUI captura o áudio (16 kHz mono), grava o WAV da sessão e envia blocos int16 por uma fila.
Aqui, um detector de voz por energia (com piso de ruído adaptativo) separa falas:
- durante a fala, a cada ~1 s sai um texto provisório (beam 1, sem timestamps);
- ao fim da fala (~0,7 s de silêncio) ou a cada 15 s, o trecho é transcrito com a qualidade
  configurada, com timestamps por palavra, e gravado no journal como um chunk — então a sessão
  aparece no histórico, com player, busca, edição e exportação como qualquer outra transcrição.
"""
from __future__ import annotations

import json
import logging
import queue
import time
import traceback
from pathlib import Path

import numpy as np

from app.pipeline.hallucination import clean_segments
from app.pipeline.journal import Journal
from app.pipeline.planner import SR
from app.pipeline.segmenter import resegment

log = logging.getLogger("whisperdesk.live")

FRAME = 480                 # 30 ms
START_FRAMES = 3            # 90 ms de voz contínua abrem uma fala
END_SILENCE_MS = 700        # silêncio que fecha uma fala com contexto suficiente...
LONG_SILENCE_MS = 1500      # ...ou que fecha qualquer fala
MIN_CONTEXT_S = 2.5         # falas curtas esperam a próxima: menos alucinação, mais contexto
PREROLL_S = 0.35
MAX_UTTER_S = 15.0
PARTIAL_EVERY_S = 1.0
MIN_PARTIAL_S = 0.6
MIN_FINAL_S = 0.35
ABS_MIN_RMS = 0.006


class VoiceGate:
    """Detector de fala por energia com piso de ruído adaptativo. Puro (testável)."""

    def __init__(self):
        self.noise = 0.004
        self.run = 0

    def threshold(self) -> float:
        return max(self.noise * 3.2, ABS_MIN_RMS)

    def frame(self, x: np.ndarray) -> tuple[bool, float]:
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        speech = rms > self.threshold()
        if not speech:
            # o piso desce rápido e sobe devagar (ruído de fundo, não a voz)
            a = 0.10 if rms < self.noise else 0.004
            self.noise = (1 - a) * self.noise + a * rms
        return speech, rms


def live_filter(segments: list[dict], prompt: str | None) -> list[dict]:
    """Filtros extras para trechos curtos e isolados (onde o Whisper mais alucina):
    descarta falas de até 3 palavras com confiança baixa e o eco do texto anterior (prompt)."""
    from app.pipeline.hallucination import _norm

    tail = _norm(prompt or "")[-120:]
    out = []
    for s in segments:
        words = s.get("words") or []
        probs = [w["prob"] for w in words]
        key = _norm(s["text"])
        if len(words) <= 3 and probs and sum(probs) / len(probs) < 0.5:
            continue
        if tail and len(key) >= 6 and key in tail:
            continue
        out.append(s)
    return out


def _setup_logging(log_dir: Path, job_id: str) -> None:
    from logging.handlers import RotatingFileHandler

    log_dir.mkdir(parents=True, exist_ok=True)
    h = RotatingFileHandler(log_dir / f"live-{job_id[:17]}.log", maxBytes=2 << 20, backupCount=1, encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(h)
    root.setLevel(logging.INFO)


def run_live(job_id: str, db_path: str, models_dir: str, log_dir: str, audio_q, events, stop_event):
    """Ponto de entrada do processo. `audio_q` recebe bytes int16 (16 kHz mono) e None ao final."""
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
    try:
        _LiveSession(j, job_id, Path(models_dir), emit, audio_q, stop_event).run()
    except Exception as e:
        log.error("Falha na sessão ao vivo:\n%s", traceback.format_exc())
        try:
            _finish(j, job_id, None)
        except Exception:
            pass
        emit("error", {"code": type(e).__name__, "message": str(e)})
    finally:
        j.close()


def _finish(j: Journal, job_id: str, total_samples: int | None):
    fields = {}
    if total_samples is not None:
        fields = {"total_samples": int(total_samples), "media_duration": total_samples / SR}
    j.set_state(job_id, "DONE", **fields)


class _LiveSession:
    def __init__(self, j: Journal, job_id: str, models_dir: Path, emit, audio_q, stop_event):
        self.j, self.job_id, self.models_dir, self.emit = j, job_id, models_dir, emit
        self.audio_q, self.stop_event = audio_q, stop_event
        self.cfg = json.loads(j.job(job_id)["config_json"])
        self.language = self.cfg.get("language")
        self.task = self.cfg.get("task") or "transcribe"
        self.beam = max(1, int(self.cfg.get("beam_size") or 5))
        self.gate = VoiceGate()
        self.base = 0                       # amostra global do início de self.buf
        self.buf = np.zeros(0, np.float32)
        self.total = 0                      # amostras recebidas
        self.pos = 0                        # próxima amostra global a passar pelo detector
        self.utt_start: int | None = None
        self.silence = 0
        self.rms: list[tuple[int, float]] = []
        self.last_partial = 0.0
        self.partial_every = PARTIAL_EVERY_S
        self.chunk_idx = 0
        self.prompt: str | None = None
        self.engine = None
        self.got_eof = False

    # ------------------------------------------------------------ ciclo
    def run(self):
        from app.pipeline.models import ensure_model
        from app.pipeline.worker import Engine

        self.emit("state", "LOADING_MODEL")

        def _dlp(done, total):
            self.emit("download_progress", {"done": done, "total": total})

        size = self.cfg["model_size"]
        path = ensure_model(size, self.models_dir, on_progress=_dlp, cancel_event=self.stop_event)
        cfg = {**self.cfg, "vad_filter": False}
        self.engine = Engine(path, size, cfg, prefer_gpu=(self.cfg.get("device") == "cuda"), emit=self.emit)
        self._call(lambda m: list(m.transcribe(np.zeros(SR // 2, np.float32), beam_size=1, language="en",
                                               without_timestamps=True)[0]))  # aquece (CUDA/cuDNN)
        self.emit("state", "LIVE")
        try:
            while True:
                self._pull(0.1)
                self._process()
                if self.stop_event.is_set():
                    self._drain_until_eof()
                    self._process()
                    if self.utt_start is not None and self.total - self.utt_start >= MIN_FINAL_S * SR:
                        self._finalize(self.total)
                    break
                if self.utt_start is not None and self.audio_q.empty():
                    self._maybe_partial()
        finally:
            self.engine.close()
        _finish(self.j, self.job_id, self.total)
        self.emit("done", {"language": self.language, **self.j.stats(self.job_id)})

    def _pull(self, timeout: float):
        blocks = []
        try:
            item = self.audio_q.get(timeout=timeout)
            while True:
                if item is None:
                    self.got_eof = True
                    break
                blocks.append(item)
                item = self.audio_q.get_nowait()
        except queue.Empty:
            pass
        if blocks:
            x = np.frombuffer(b"".join(blocks), dtype=np.int16).astype(np.float32) / 32768.0
            self.buf = np.concatenate([self.buf, x])
            self.total += len(x)

    def _drain_until_eof(self):
        deadline = time.monotonic() + 3.0
        while not self.got_eof and time.monotonic() < deadline:
            self._pull(0.2)

    # ------------------------------------------------------------ detecção
    def _process(self):
        while self.pos + FRAME <= self.total:
            a = self.pos - self.base
            speech, rms = self.gate.frame(self.buf[a:a + FRAME])
            self.pos += FRAME
            if self.utt_start is None:
                self.gate.run = self.gate.run + 1 if speech else 0
                if self.gate.run >= START_FRAMES:
                    start = self.pos - START_FRAMES * FRAME - int(PREROLL_S * SR)
                    self.utt_start = max(self.base, start)
                    self.silence = 0
                    self.rms = []
                    self.last_partial = time.monotonic()
                    self.gate.run = 0
                continue
            self.rms.append((self.pos, rms))
            self.silence = 0 if speech else self.silence + 1
            quiet_ms = self.silence * FRAME * 1000 // SR
            long_enough = self.pos - self.utt_start >= MIN_CONTEXT_S * SR
            if quiet_ms >= LONG_SILENCE_MS or (long_enough and quiet_ms >= END_SILENCE_MS):
                end = self.pos - self.silence * FRAME + int(0.2 * SR)
                self._finalize(min(end, self.pos))
            elif self.pos - self.utt_start >= MAX_UTTER_S * SR:
                self._finalize(self._quiet_cut())
        if self.utt_start is None:
            keep = int(PREROLL_S * SR) + FRAME * START_FRAMES
            if len(self.buf) > keep * 4:
                drop = len(self.buf) - keep
                self.buf = self.buf[drop:]
                self.base += drop

    def _quiet_cut(self) -> int:
        """Corte de fala longa: o ponto mais silencioso dos últimos 3 s."""
        window = [(p, r) for p, r in self.rms if p >= self.pos - 3 * SR]
        if not window:
            return self.pos
        return min(window, key=lambda t: t[1])[0]

    # ------------------------------------------------------------ transcrição
    def _call(self, fn):
        """Executa no modelo, descendo para a CPU se o runtime da GPU falhar."""
        from app.pipeline.worker import GPU_RUNTIME_PATTERNS, OOM_PATTERNS

        while True:
            self.engine._ensure_loaded()
            try:
                return fn(self.engine._model)
            except (RuntimeError, MemoryError, OSError) as e:
                msg = str(e)
                if isinstance(e, MemoryError) or OOM_PATTERNS.search(msg):
                    if self.engine.degrade(f"memória insuficiente ({msg[:120]})"):
                        continue
                elif self.engine.step[0] == "cuda" and GPU_RUNTIME_PATTERNS.search(msg):
                    if self.engine.degrade_to_cpu(f"falha do runtime da GPU ({msg[:160]})"):
                        continue
                raise

    def _slice(self, a: int, b: int) -> np.ndarray:
        return self.buf[max(0, a - self.base):max(0, b - self.base)]

    def _maybe_partial(self):
        now = time.monotonic()
        if now - self.last_partial < self.partial_every:
            return
        a, b = self.utt_start, self.pos
        if b - a < MIN_PARTIAL_S * SR:
            return
        audio = self._slice(max(a, b - int(MAX_UTTER_S * SR)), b)
        t0 = time.monotonic()
        segs = self._call(lambda m: list(m.transcribe(
            audio, language=self.language, task=self.task, beam_size=1, temperature=0.0,
            without_timestamps=True, condition_on_previous_text=False,
            vad_filter=False)[0]))
        took = time.monotonic() - t0
        # Em máquinas lentas, os provisórios ficam mais espaçados para não atrasar o texto final.
        self.partial_every = max(PARTIAL_EVERY_S, took * 1.6)
        self.last_partial = time.monotonic()
        text = "".join(s.text for s in segs).strip()
        if text and self.utt_start is not None:
            self.emit("partial", {"text": text, "start_ms": a * 1000 // SR})

    def _finalize(self, end: int):
        from app.pipeline.worker import owned_segments

        a = self.utt_start
        self.utt_start = None
        self.silence = 0
        if end - a < MIN_FINAL_S * SR:
            self.emit("partial", {"text": ""})
            return
        audio = self._slice(a, end)
        t0 = time.monotonic()
        temps = (0.0, 0.2, 0.4, 0.6)
        raw, info = self._call(lambda m: self._full(m, audio, temps))
        elapsed = time.monotonic() - t0
        if not self.language and info is not None and (getattr(info, "language_probability", 0) or 0) >= 0.7:
            self.language = info.language
            self.j.update_fields(self.job_id, detected_language=self.language)
            self.emit("language", {"code": self.language, "prob": round(info.language_probability, 3)})
        segments = live_filter(resegment(clean_segments(owned_segments(raw, a, end, a), language=self.language)),
                               self.prompt)
        # Fala longa cortada no meio: a próxima começa exatamente onde esta terminou.
        if end < self.pos:
            self.utt_start = end
            self.last_partial = time.monotonic()
        self.emit("partial", {"text": ""})
        if not segments:
            return
        idx = self.chunk_idx
        self.chunk_idx += 1
        self.j.add_chunk(self.job_id, idx, a, end)
        last = segments[-1]
        dev, compute, batch, _ = self.engine.step
        self.j.commit_chunk(self.job_id, idx, segments, {
            "device": dev, "compute_type": compute, "batch_size": batch, "audio_s": (end - a) / SR,
            "elapsed_s": elapsed, "tail_text": last["text"][-200:], "tail_logprob": last["avg_logprob"],
        })
        self.prompt = " ".join(s["text"] for s in segments)[-200:]
        self.emit("chunk_done", {
            "idx": idx,
            "segments": [{**{k: v for k, v in s.items() if k != "words"},
                          "key": (idx, n), "words": [(w["start_ms"], w["end_ms"], w["word"], w["prob"]) for w in s["words"]]}
                         for n, s in enumerate(segments)],
        })

    def _full(self, m, audio, temps):
        segs, info = m.transcribe(
            audio, language=self.language, task=self.task, beam_size=self.beam, word_timestamps=True,
            temperature=temps, condition_on_previous_text=False,
            compression_ratio_threshold=2.4, log_prob_threshold=-1.0, no_speech_threshold=0.6,
            vad_filter=False)
        return list(segs), info


def recordings_dir() -> Path:
    from app import paths

    d = paths.local_dir() / "recordings"
    d.mkdir(parents=True, exist_ok=True)
    return d


def new_recording_path(prefix: str = "Recording") -> Path:
    stamp = time.strftime("%Y-%m-%d %H-%M-%S")
    p = recordings_dir() / f"{prefix} {stamp}.wav"
    n = 2
    while p.exists():
        p = recordings_dir() / f"{prefix} {stamp} ({n}).wav"
        n += 1
    return p
