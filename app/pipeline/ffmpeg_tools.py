"""Mídia via PyAV (bibliotecas do FFmpeg embutidas na wheel `av`): probe, extração do WAV de trabalho
e picos para a forma de onda. Sem executáveis externos: funciona igual em Windows, Linux e macOS.

O nome do módulo é histórico (a v2.1 chamava ffmpeg.exe/ffprobe.exe); a API pública é a mesma."""
from __future__ import annotations

import ctypes
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable

import numpy as np

log = logging.getLogger("whisperdesk.media")

SR = 16_000
OPEN_TIMEOUT_S = 120.0


class AudioError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class Cancelled(Exception):
    pass


def _av():
    try:
        import av
    except Exception as e:  # noqa: BLE001
        raise AudioError("FFMPEG_MISSING", f"Biblioteca de mídia (PyAV) indisponível: {e}") from None
    av.logging.set_level(av.logging.ERROR)
    return av


def backend_version() -> str:
    av = _av()
    try:
        ff = ".".join(str(x) for x in av.library_versions["libavcodec"])
    except Exception:  # noqa: BLE001
        ff = "?"
    return f"PyAV {av.__version__} (libavcodec {ff}, FFmpeg {getattr(av, 'ffmpeg_version_info', '?')})"


@dataclass(frozen=True)
class AudioStream:
    index: int            # índice absoluto no container
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


def _open(path: str, read_timeout: float = 300.0):
    av = _av()
    try:
        return av.open(path, mode="r", timeout=(OPEN_TIMEOUT_S, read_timeout),
                       options={"fflags": "+discardcorrupt+genpts"})
    except FileNotFoundError:
        raise AudioError("FILE_NOT_FOUND", f"Arquivo não encontrado: {path}") from None
    except av.error.ExitError:
        raise AudioError("PROBE_TIMEOUT", "A leitura do arquivo excedeu o tempo limite (rede lenta?).") from None
    except av.error.FFmpegError as e:
        raise AudioError("PROBE_FAILED", f"Formato não reconhecido ou arquivo corrompido: {e}") from None
    except OSError as e:
        raise AudioError("PROBE_FAILED", str(e)) from None


def _seconds(value, time_base) -> float | None:
    if value is None or time_base is None:
        return None
    try:
        v = float(value * time_base)
    except (TypeError, ValueError):
        return None
    return v if v >= 0 else None


def probe(path: str) -> ProbeInfo:
    if not os.path.isfile(path):
        raise AudioError("FILE_NOT_FOUND", f"Arquivo não encontrado: {path}")
    av = _av()
    from av.stream import Disposition

    with _open(path) as c:
        streams: list[AudioStream] = []
        has_video = False
        for s in c.streams:
            if s.type == "video" and not (s.disposition & Disposition.attached_pic):
                has_video = True
            if s.type != "audio":
                continue
            try:
                cc = s.codec_context
                meta = {k.lower(): v for k, v in (s.metadata or {}).items()}
                layout = cc.layout.name if cc.layout is not None else ""
                streams.append(AudioStream(
                    index=int(s.index),
                    codec=cc.name or "?",
                    channels=int(cc.channels or 0),
                    layout="" if layout in ("", "unknown") else layout,
                    sample_rate=int(cc.sample_rate or 0),
                    language=meta.get("language"),
                    title=meta.get("title"),
                    start_time=_seconds(s.start_time, s.time_base) or 0.0,
                    duration=_seconds(s.duration, s.time_base),
                ))
            except (av.error.FFmpegError, AttributeError, ValueError, TypeError):
                continue
        if not streams:
            raise AudioError("NO_AUDIO_STREAM", "O arquivo não contém trilha de áudio decodificável.")
        duration = 0.0
        container_dur = c.duration / av.time_base if c.duration else None
        for candidate in (container_dur, *(s.duration for s in streams)):
            if candidate and candidate > 0:
                duration = float(candidate)
                break
        if duration <= 0:
            duration = _measure_duration(c, streams[0].index)
        if duration <= 0:
            raise AudioError("NO_DURATION", "Duração indeterminável (arquivo truncado?).")
        fmt = c.format.name if c.format is not None else ""
    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    return ProbeInfo(path=path, duration=duration, format_name=str(fmt), size_bytes=size,
                     audio_streams=streams, has_video=has_video)


def _measure_duration(container, index: int) -> float:
    """Para fluxos sem duração no cabeçalho (ex.: ADTS/MP3 cru): percorre os pacotes (sem decodificar)."""
    s = container.streams[index]
    last = 0.0
    try:
        for pkt in container.demux(s):
            if pkt.pts is not None and pkt.time_base is not None:
                end = float((pkt.pts + (pkt.duration or 0)) * pkt.time_base)
                last = max(last, end)
    except Exception:  # noqa: BLE001
        pass
    return last


# ---------------------------------------------------------------------------
# Decodificação + filtros
_PLAIN_DOWNMIX = ("aformat", "channel_layouts=mono")


def _downmix_filters(stream: AudioStream | None) -> list[tuple[str, str]]:
    """Candidatos de downmix, do preferido ao mais simples.

    Em 5.1/6.1/7.1 prioriza o canal central (diálogo): 0,6·FC + 0,2·FL + 0,2·FR ("<" normaliza os
    pesos 3:1:1). A sintaxe do filtro pan aceita varia entre builds do FFmpeg, então há variantes
    equivalentes (nome do canal, índice) e, em último caso, o downmix padrão."""
    if stream and stream.channels >= 6 and stream.layout.split("(")[0] in {"5.1", "6.1", "7.1"}:
        return [("pan", "mono|FC<3*FC+FL+FR"), ("pan", "mono|c0<3*c2+c0+c1"),
                ("pan", "1c|c0=0.6*c2+0.2*c0+0.2*c1"), _PLAIN_DOWNMIX]
    return [_PLAIN_DOWNMIX]


class _Pipeline:
    """Grafo abuffer → compensação de lacunas → downmix → reamostragem → s16 mono.

    Recriado se o formato dos quadros mudar no meio da trilha (ex.: troca de taxa em transmissões)."""

    def __init__(self, stream: AudioStream, rate: int, sample_fmt: str = "s16", async_fill: bool = True):
        self.stream = stream
        self.rate = rate
        self.sample_fmt = sample_fmt
        self.async_fill = async_fill
        self.graph = None
        self.sig = None
        self.first = True

    def _build(self, frame):
        av = _av()
        candidates = _downmix_filters(self.stream)
        for i, downmix in enumerate(candidates):
            try:
                self._build_with(frame, downmix)
                return
            except (av.error.FFmpegError, ValueError):
                if i == len(candidates) - 1:
                    raise
                log.debug("downmix %s rejeitado por este FFmpeg; tentando o próximo", downmix)

    def _build_with(self, frame, downmix):
        av = _av()
        g = av.filter.Graph()
        layout = frame.layout.name if frame.layout is not None else ""
        if not layout or layout == "unknown":
            layout = "mono" if len(frame.layout.channels) == 1 else "stereo"
        src = g.add_abuffer(format=frame.format.name, sample_rate=frame.sample_rate, layout=layout,
                            time_base=frame.time_base or Fraction(1, frame.sample_rate))
        nodes = [src]
        if self.async_fill:
            first = ":first_pts=0" if self.first else ""
            nodes.append(g.add("aresample", f"async=1:min_hard_comp=0.100{first}"))
        nodes.append(g.add(*downmix))
        nodes.append(g.add("aresample", str(self.rate)))
        nodes.append(g.add("aformat", f"sample_fmts={self.sample_fmt}:channel_layouts=mono:sample_rates={self.rate}"))
        nodes.append(g.add("abuffersink"))
        g.link_nodes(*nodes).configure()
        self.graph = g
        self.first = False

    def _drain(self):
        av = _av()
        out = []
        while True:
            try:
                f = self.graph.pull()
            except (BlockingIOError, EOFError):
                break
            out.append(f.to_ndarray().reshape(-1))
        return out

    def push(self, frame) -> list[np.ndarray]:
        sig = (frame.format.name, frame.sample_rate, frame.layout.name, len(frame.layout.channels))
        out: list[np.ndarray] = []
        if self.graph is not None and sig != self.sig:
            out += self.flush()
        if self.graph is None:
            self._build(frame)
            self.sig = sig
        self.graph.push(frame)
        out += self._drain()
        return out

    def flush(self) -> list[np.ndarray]:
        if self.graph is None:
            return []
        try:
            self.graph.push(None)
        except Exception:  # noqa: BLE001
            pass
        out = self._drain()
        self.graph = None
        return out


def _decode_frames(container, stream_index: int, cancel_event, on_error: Callable[[str], None] | None = None):
    """Quadros de áudio da trilha, tolerando pacotes corrompidos (equivalente a -err_detect ignore_err)."""
    av = _av()
    s = container.streams[stream_index]
    s.thread_type = "AUTO"
    next_pts = 0
    for pkt in container.demux(s):
        if cancel_event is not None and cancel_event.is_set():
            raise Cancelled()
        try:
            frames = pkt.decode()
        except av.error.ExitError:
            raise
        except (av.error.InvalidDataError, av.error.FFmpegError, ValueError) as e:
            if on_error:
                on_error(str(e))
            continue
        for f in frames:
            if f.pts is None or f.time_base is None:
                # Sem carimbo de tempo: segue a contagem de amostras (equivale a +genpts).
                f.time_base = Fraction(1, f.sample_rate)
                f.pts = next_pts
            next_pts = int(round(f.pts * f.time_base * f.sample_rate)) + f.samples
            yield f


def extract_wav(src: str, dst: Path, stream: AudioStream, duration: float, *,
                cancel_event, on_progress: Callable[[float], None], log_path: Path,
                stall_timeout_s: float = 300.0) -> str | None:
    """Extrai a trilha `stream` para WAV mono 16 kHz PCM s16 (RF64 se > 4 GB).

    Retorna um aviso (texto) quando a duração extraída diverge da declarada, ou None."""
    import soundfile as sf

    av = _av()
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    need = int(duration * SR * 2 * 1.05) + (64 << 20)
    free = shutil.disk_usage(dst.parent).free
    if free < need:
        raise AudioError("DISK_FULL",
                         f"Espaço insuficiente em {dst.parent}: {need >> 20} MB necessários, {free >> 20} MB livres.")
    tmp = dst.with_name(dst.stem + ".partial.wav")
    tmp.unlink(missing_ok=True)
    big = duration * SR * 2 > 3.8e9
    log_path.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    frames_out = 0
    last_report = 0.0
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        log.write(f"extract {src} stream #{stream.index} ({stream.describe()}) → {tmp} [{backend_version()}]\n")
        try:
            with _open(src, read_timeout=stall_timeout_s) as c, \
                    sf.SoundFile(str(tmp), "w", SR, 1, subtype="PCM_16", format="RF64" if big else "WAV") as out:
                pipe = _Pipeline(stream, SR)

                def _err(msg):
                    if len(errors) < 50:
                        log.write(f"pacote ignorado: {msg}\n")
                    errors.append(msg)

                for f in _decode_frames(c, stream.index, cancel_event, _err):
                    for chunk in pipe.push(f):
                        out.write(chunk)
                        frames_out += len(chunk)
                    now = time.monotonic()
                    if duration > 0 and now - last_report > 0.25:
                        last_report = now
                        on_progress(min(1.0, frames_out / SR / duration))
                for chunk in pipe.flush():
                    out.write(chunk)
                    frames_out += len(chunk)
        except Cancelled:
            tmp.unlink(missing_ok=True)
            raise
        except av.error.ExitError:
            tmp.unlink(missing_ok=True)
            raise AudioError("DECODE_STALLED",
                             f"A leitura parou de progredir por {stall_timeout_s:.0f}s (mídia/rede travada).") from None
        except AudioError:
            tmp.unlink(missing_ok=True)
            raise
        except (av.error.FFmpegError, RuntimeError, ValueError, OSError) as e:
            tmp.unlink(missing_ok=True)
            log.write(f"falha: {e!r}\n")
            raise AudioError("DECODE_FAILED", f"Falha ao decodificar o áudio: {e}") from None
        if errors:
            log.write(f"{len(errors)} pacote(s) corrompido(s) ignorado(s)\n")
    if cancel_event is not None and cancel_event.is_set():
        tmp.unlink(missing_ok=True)
        raise Cancelled()
    if frames_out <= 0:
        tmp.unlink(missing_ok=True)
        raise AudioError("EMPTY_AUDIO", "A trilha de áudio não produziu amostras.")
    expected = stream.duration or duration
    warning = None
    if abs(frames_out / SR - expected) > max(0.5, expected * 0.005):
        warning = (f"Áudio extraído tem {frames_out / SR:.1f}s; a trilha declara {expected:.1f}s "
                   f"(trilha truncada ou metadados imprecisos). Detalhes em {log_path}.")
    os.replace(tmp, dst)   # atômico no mesmo volume
    on_progress(1.0)
    return warning


def decode_peaks(src: str, stream_index: int | None, buckets: int, duration: float,
                 cancel_event: threading.Event | None = None, rate: int = 4000) -> np.ndarray:
    """Envelope (máximo absoluto por balde) decodificando em streaming: memória O(buckets)."""
    total = max(1, int(duration * rate))
    per = max(1, total // buckets)
    peaks = np.zeros(buckets, dtype=np.float32)
    pos = 0
    with _open(src) as c:
        audio = [s for s in c.streams if s.type == "audio"]
        if not audio:
            return peaks
        idx = stream_index if stream_index is not None else audio[0].index
        pipe = _Pipeline(None, rate, sample_fmt="flt")

        def acc(x):
            nonlocal pos
            x = np.abs(x.astype(np.float32, copy=False))
            ids = np.minimum((pos + np.arange(len(x))) // per, buckets - 1)
            np.maximum.at(peaks, ids, x)
            pos += len(x)

        for f in _decode_frames(c, idx, cancel_event):
            for chunk in pipe.push(f):
                acc(chunk)
        for chunk in pipe.flush():
            acc(chunk)
    return peaks


def decode_audio(src: str, stream_index: int | None = None, rate: int = SR) -> np.ndarray:
    """Decodifica a trilha inteira para float32 mono (uso em testes e utilitários)."""
    out = []
    with _open(src) as c:
        audio = [s for s in c.streams if s.type == "audio"]
        idx = stream_index if stream_index is not None else audio[0].index
        pipe = _Pipeline(None, rate, sample_fmt="flt", async_fill=False)
        for f in _decode_frames(c, idx, None):
            out += pipe.push(f)
        out += pipe.flush()
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


# ---------------------------------------------------------------------------
# Impede a suspensão automática enquanto houver job ativo (Windows, macOS e Linux)
@contextmanager
def keep_system_awake():
    if os.name == "nt":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        k32 = ctypes.windll.kernel32
        k32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        try:
            yield
        finally:
            k32.SetThreadExecutionState(ES_CONTINUOUS)
        return
    proc = None
    try:
        if sys.platform == "darwin" and shutil.which("caffeinate"):
            # -i: impede o repouso por inatividade; -w: some sozinho se este processo morrer.
            proc = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())],
                                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform.startswith("linux") and shutil.which("systemd-inhibit"):
            proc = subprocess.Popen(["systemd-inhibit", "--what=idle:sleep", "--who=WhisperDesk",
                                     "--why=Transcrição em andamento", "--mode=block", "sleep", "infinity"],
                                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        proc = None
    try:
        yield
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(5)
            except subprocess.TimeoutExpired:
                proc.kill()
