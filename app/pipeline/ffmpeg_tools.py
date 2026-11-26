"""Integração com ffmpeg/ffprobe embutidos: probe, extração do WAV de trabalho e picos para a forma de onda."""
from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000 if os.name == "nt" else 0
SR = 16_000


class AudioError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class Cancelled(Exception):
    pass


def _bin_dirs() -> list[Path]:
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        return [base / "bin"]
    repo = Path(__file__).resolve().parents[2]
    return [repo / "bin", repo / "build_cache" / "bin"]


def tool(name: str) -> str:
    exe = f"{name}.exe" if os.name == "nt" else name
    for d in _bin_dirs():
        if (d / exe).is_file():
            return str(d / exe)
    if not getattr(sys, "frozen", False):
        found = shutil.which(name)  # apenas em desenvolvimento
        if found:
            return found
    raise AudioError("FFMPEG_MISSING", f"{exe} não encontrado (procurado em: {', '.join(map(str, _bin_dirs()))}).")


@dataclass(frozen=True)
class AudioStream:
    index: int            # índice absoluto no container (para -map 0:<index>)
    codec: str
    channels: int
    layout: str
    sample_rate: int
    language: str | None
    title: str | None
    start_time: float
    duration: float | None = None

    def describe(self) -> str:
        parts = [f"#{self.index}", self.codec]
        if self.language:
            parts.append(self.language)
        if self.layout:
            parts.append(self.layout)
        elif self.channels:
            parts.append(f"{self.channels}ch")
        if self.title:
            parts.append(f"“{self.title}”")
        return " · ".join(parts)


@dataclass(frozen=True)
class ProbeInfo:
    path: str
    duration: float
    format_name: str
    size_bytes: int
    audio_streams: list[AudioStream] = field(default_factory=list)
    has_video: bool = False

    @property
    def default_stream(self) -> AudioStream:
        return self.audio_streams[0]


def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=timeout,
        creationflags=CREATE_NO_WINDOW, stdin=subprocess.DEVNULL,
    )


def probe(path: str) -> ProbeInfo:
    if not os.path.isfile(path):
        raise AudioError("FILE_NOT_FOUND", f"Arquivo não encontrado: {path}")
    try:
        r = _run([tool("ffprobe"), "-v", "error", "-print_format", "json",
                  "-show_format", "-show_streams", path], timeout=300)
    except subprocess.TimeoutExpired:
        raise AudioError("PROBE_TIMEOUT", "ffprobe excedeu 5 min (arquivo em rede lenta ou corrompido).") from None
    except OSError as e:
        raise AudioError("FFMPEG_MISSING", f"Falha ao executar ffprobe: {e}") from None
    if r.returncode != 0:
        raise AudioError("PROBE_FAILED", (r.stderr or "").strip()[-1000:] or "ffprobe falhou.")
    try:
        data = json.loads(r.stdout or "{}")
    except json.JSONDecodeError:
        raise AudioError("PROBE_FAILED", "Saída inválida do ffprobe.") from None

    streams: list[AudioStream] = []
    has_video = False
    for s in data.get("streams", []):
        ctype = s.get("codec_type")
        if ctype == "video" and not (s.get("disposition") or {}).get("attached_pic"):
            has_video = True
        if ctype != "audio":
            continue
        tags = {k.lower(): v for k, v in (s.get("tags") or {}).items()}
        try:
            dur = s.get("duration")
            streams.append(AudioStream(
                index=int(s["index"]),
                codec=s.get("codec_name", "?"),
                channels=int(s.get("channels") or 0),
                layout=s.get("channel_layout") or "",
                sample_rate=int(s.get("sample_rate") or 0),
                language=tags.get("language"),
                title=tags.get("title"),
                start_time=float(s.get("start_time") or 0.0),
                duration=float(dur) if dur not in (None, "N/A") else None,
            ))
        except (KeyError, ValueError, TypeError):
            continue
    if not streams:
        raise AudioError("NO_AUDIO_STREAM", "O arquivo não contém trilha de áudio decodificável.")

    fmt = data.get("format", {}) or {}
    duration = 0.0
    for candidate in (fmt.get("duration"), *(s.duration for s in streams)):
        try:
            duration = float(candidate)
            if duration > 0:
                break
        except (TypeError, ValueError):
            continue
    if duration <= 0:
        raise AudioError("NO_DURATION", "Duração indeterminável (arquivo truncado?).")
    try:
        size = int(fmt.get("size") or os.path.getsize(path))
    except (TypeError, ValueError, OSError):
        size = 0
    return ProbeInfo(path=path, duration=duration, format_name=str(fmt.get("format_name", "")),
                     size_bytes=size, audio_streams=streams, has_video=has_video)


def _downmix_filter(stream: AudioStream) -> str:
    """Prioriza o canal central (diálogo) em 5.1/6.1/7.1; downmix padrão nos demais."""
    if stream.channels >= 6 and stream.layout.split("(")[0] in {"5.1", "6.1", "7.1"}:
        return "pan=mono|c0=0.60*FC+0.20*FL+0.20*FR"
    return "aformat=channel_layouts=mono"


_SOXR: bool | None = None


def _has_soxr() -> bool:
    global _SOXR
    if _SOXR is None:
        try:
            r = _run([tool("ffmpeg"), "-hide_banner", "-buildconf"], timeout=30)
            _SOXR = "libsoxr" in (r.stdout + r.stderr)
        except Exception:
            _SOXR = False
    return _SOXR


def _popen(cmd, **kw) -> subprocess.Popen:
    return subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                            creationflags=CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS, **kw)


def _kill(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            pass


def extract_wav(src: str, dst: Path, stream: AudioStream, duration: float, *,
                cancel_event, on_progress: Callable[[float], None], log_path: Path,
                stall_timeout_s: float = 300.0) -> str | None:
    """Extrai a trilha `stream` para WAV mono 16 kHz PCM s16 (RF64 se > 4 GB).

    Retorna um aviso (texto) quando a duração extraída diverge da declarada, ou None."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    need = int(duration * SR * 2 * 1.05) + (64 << 20)
    free = shutil.disk_usage(dst.parent).free
    if free < need:
        raise AudioError("DISK_FULL",
                         f"Espaço insuficiente em {dst.parent}: {need >> 20} MB necessários, {free >> 20} MB livres.")

    tmp = dst.with_name(dst.stem + ".partial.wav")
    tmp.unlink(missing_ok=True)
    resampler = f"aresample={SR}:resampler=soxr" if _has_soxr() else f"aresample={SR}"
    af = ",".join([
        "aresample=async=1:min_hard_comp=0.100:first_pts=0",
        _downmix_filter(stream),
        resampler,
    ])
    cmd = [
        tool("ffmpeg"), "-hide_banner", "-nostdin", "-y", "-loglevel", "warning",
        "-fflags", "+discardcorrupt+genpts", "-err_detect", "ignore_err",
        "-i", src,
        "-map", f"0:{stream.index}", "-vn", "-sn", "-dn",
        "-af", af, "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le",
        "-rf64", "auto", "-f", "wav",
        "-progress", "pipe:1", "-nostats",
        str(tmp),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        proc = _popen(cmd, stdout=subprocess.PIPE, stderr=log, encoding="utf-8", errors="replace", bufsize=1)
        last_activity = [time.monotonic()]
        stalled = [False]
        stop = threading.Event()

        def _watch():
            while not stop.wait(0.5):
                if proc.poll() is not None:
                    return
                if cancel_event is not None and cancel_event.is_set():
                    _kill(proc)
                    return
                if time.monotonic() - last_activity[0] > stall_timeout_s:
                    stalled[0] = True
                    _kill(proc)
                    return

        threading.Thread(target=_watch, name="ffmpeg-watchdog", daemon=True).start()
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                last_activity[0] = time.monotonic()
                if line.startswith("out_time_us=") or line.startswith("out_time_ms="):
                    v = line.split("=", 1)[1].strip()
                    if v.isdigit() and duration > 0:
                        on_progress(min(1.0, int(v) / 1e6 / duration))
            rc = proc.wait()
        finally:
            stop.set()
            _kill(proc)

    if cancel_event is not None and cancel_event.is_set():
        tmp.unlink(missing_ok=True)
        raise Cancelled()
    if stalled[0]:
        tmp.unlink(missing_ok=True)
        raise AudioError("DECODE_STALLED", f"ffmpeg parou de progredir por {stall_timeout_s:.0f}s (mídia/rede travada).")
    if rc != 0:
        tmp.unlink(missing_ok=True)
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
        raise AudioError("DECODE_FAILED", f"ffmpeg saiu com código {rc}.\n{tail}")

    import soundfile as sf

    try:
        frames = sf.info(str(tmp)).frames
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise AudioError("DECODE_FAILED", f"WAV extraído ilegível: {e}") from None
    if frames <= 0:
        tmp.unlink(missing_ok=True)
        raise AudioError("EMPTY_AUDIO", "A trilha de áudio não produziu amostras.")
    expected = stream.duration or duration
    warning = None
    if abs(frames / SR - expected) > max(0.5, expected * 0.005):
        warning = (f"Áudio extraído tem {frames / SR:.1f}s; a trilha declara {expected:.1f}s "
                   f"(trilha truncada ou metadados imprecisos). Detalhes em {log_path}.")
    os.replace(tmp, dst)   # atômico no mesmo volume
    on_progress(1.0)
    return warning


def decode_peaks(src: str, stream_index: int | None, buckets: int, duration: float,
                 cancel_event: threading.Event | None = None, rate: int = 4000) -> np.ndarray:
    """Envelope (máximo absoluto por balde) via ffmpeg em streaming: memória O(buckets).

    Decodifica a `rate` Hz mono (s16) e acumula picos em `buckets` baldes."""
    total = max(1, int(duration * rate))
    per = max(1, total // buckets)
    peaks = np.zeros(buckets, dtype=np.float32)
    cmd = [tool("ffmpeg"), "-hide_banner", "-nostdin", "-loglevel", "error",
           "-fflags", "+discardcorrupt", "-err_detect", "ignore_err", "-i", src]
    cmd += ["-map", f"0:{stream_index}"] if stream_index is not None else ["-map", "0:a:0"]
    cmd += ["-vn", "-sn", "-dn", "-ac", "1", "-ar", str(rate), "-f", "s16le", "-"]
    proc = _popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    pos = 0
    try:
        assert proc.stdout is not None
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise Cancelled()
            buf = proc.stdout.read(rate * 2 * 60)  # 1 min de áudio por leitura
            if not buf:
                break
            x = np.abs(np.frombuffer(buf[: len(buf) // 2 * 2], dtype="<i2").astype(np.float32)) / 32768.0
            idx = np.minimum((pos + np.arange(len(x))) // per, buckets - 1)
            np.maximum.at(peaks, idx, x)
            pos += len(x)
    finally:
        _kill(proc)
    return peaks


@contextmanager
def keep_system_awake():
    """Impede a suspensão automática do Windows enquanto houver job ativo."""
    if os.name != "nt":
        yield
        return
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    k32 = ctypes.windll.kernel32
    k32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        yield
    finally:
        k32.SetThreadExecutionState(ES_CONTINUOUS)
