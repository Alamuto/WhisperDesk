"""Sessão ao vivo: captura do microfone na GUI + transcrição num processo separado.

A GUI é dona do áudio: grava o WAV da sessão (para o player e o histórico) e envia blocos int16
16 kHz mono ao processo `run_live`. Parar = enviar None + sinalizar; o processo fecha a última
fala, marca o job como DONE e responde "done"."""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import time

import numpy as np
from PySide6.QtCore import QByteArray, QObject, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QPainter
from PySide6.QtMultimedia import QAudioDevice, QAudioFormat, QAudioSource, QMediaDevices
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from app import paths
from app.gui import icons, theme
from app.gui.theme import qcolor
from app.gui.widgets import PulseDot, ThemeAware, card
from app.utils.i18n import tr
from app.pipeline import service
from app.pipeline.live_worker import new_recording_path, run_live

log = logging.getLogger(__name__)

SR = 16000
STOP_GRACE_S = 25.0          # tempo para o processo fechar a última fala
NO_SIGNAL_S = 4.0            # sem nenhuma amostra acima disto = microfone mudo/bloqueado
SILENT_PEAK = 0.0008


def input_devices() -> list[QAudioDevice]:
    return list(QMediaDevices.audioInputs())


def device_key(dev: QAudioDevice) -> str:
    return bytes(dev.id()).hex()


def find_device(key: str | None) -> QAudioDevice | None:
    devs = input_devices()
    if key:
        for d in devs:
            if device_key(d) == key:
                return d
    default = QMediaDevices.defaultAudioInput()
    if not default.isNull():
        return default
    return devs[0] if devs else None


def _to_mono_16k(raw: bytes, fmt: QAudioFormat, state: dict) -> np.ndarray:
    """Converte o formato nativo do dispositivo para float32 mono 16 kHz (reamostragem linear)."""
    sf = fmt.sampleFormat()
    if sf == QAudioFormat.SampleFormat.Int16:
        x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    elif sf == QAudioFormat.SampleFormat.Int32:
        x = np.frombuffer(raw, np.int32).astype(np.float32) / 2147483648.0
    elif sf == QAudioFormat.SampleFormat.Float:
        x = np.frombuffer(raw, np.float32).astype(np.float32)
    elif sf == QAudioFormat.SampleFormat.UInt8:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        return np.zeros(0, np.float32)
    ch = max(1, fmt.channelCount())
    if ch > 1:
        n = len(x) // ch
        x = x[:n * ch].reshape(n, ch).mean(axis=1)
    rate = fmt.sampleRate()
    if rate == SR or len(x) == 0:
        return x
    # Reamostragem contínua entre blocos (mantém a fase em `state`).
    pos = state.get("pos", 0.0)
    prev = state.get("prev")
    src = np.concatenate([[prev], x]) if prev is not None else x
    step = rate / SR
    idx = np.arange(pos, len(src) - 1, step)
    out = np.interp(idx, np.arange(len(src)), src).astype(np.float32)
    state["pos"] = (idx[-1] + step - (len(src) - 1)) if len(idx) else pos - (len(src) - 1)
    state["prev"] = src[-1]
    return out


class LiveSession(QObject):
    event = Signal(str, object)           # eventos do processo (state, partial, chunk_done, ...)
    level = Signal(float)                 # nível do microfone 0..1 (para o medidor)
    finished = Signal(str, str, object)   # (job_id, done|failed|cancelled, payload)
    no_signal = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ctx = mp.get_context("spawn")
        self._proc = None
        self._audio_q = None
        self._events = None
        self._stop = None
        self._source: QAudioSource | None = None
        self._io = None
        self._fmt: QAudioFormat | None = None
        self._rs_state: dict = {}
        self._wav = None
        self._wav_path = ""
        self.job_id: str | None = None
        self._samples = 0
        self._peak = 0.0
        self._started = 0.0
        self._stop_at: float | None = None
        self._terminal = None
        self._warned_silent = False
        self._last_flush = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self._pump)

    # ------------------------------------------------------------------ API
    def is_running(self) -> bool:
        return self._proc is not None

    def is_stopping(self) -> bool:
        return self._stop_at is not None

    @property
    def elapsed(self) -> float:
        return self._samples / SR

    @property
    def wav_path(self) -> str:
        return self._wav_path

    def start(self, cfg: dict, device: QAudioDevice, prefix: str) -> str:
        import soundfile as sf

        if self._proc is not None:
            raise RuntimeError("Sessão ao vivo já em andamento.")
        fmt = QAudioFormat()
        fmt.setSampleRate(SR)
        fmt.setChannelCount(1)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        if not device.isFormatSupported(fmt):
            fmt = device.preferredFormat()
        self._fmt = fmt
        self._rs_state = {}
        path = new_recording_path(prefix)
        self._wav_path = str(path)
        self._wav = sf.SoundFile(self._wav_path, "w", SR, 1, "PCM_16")
        self.job_id = service.create_live_job(self._wav_path, cfg)
        self._samples = 0
        self._peak = 0.0
        self._warned_silent = False
        self._terminal = None
        self._stop_at = None
        self._started = time.monotonic()
        self._last_flush = self._started

        self._audio_q = self._ctx.Queue()
        self._events = self._ctx.Queue()
        self._stop = self._ctx.Event()
        self._proc = self._ctx.Process(
            target=run_live,
            args=(self.job_id, str(paths.journal_path()), str(paths.models_dir()), str(paths.logs_dir()),
                  self._audio_q, self._events, self._stop),
            name=f"whisperdesk-live-{self.job_id[5:13]}", daemon=True)
        self._proc.start()

        self._source = QAudioSource(device, fmt, self)
        self._source.setBufferSize(max(4096, fmt.bytesForDuration(200_000)))
        self._io = self._source.start()
        if self._io is None:
            raise RuntimeError("QAudioSource não iniciou")
        self._io.readyRead.connect(self._on_audio)
        self._timer.start()
        log.info("Ao vivo: %s, formato %s Hz × %s canais (%s)", device.description(), fmt.sampleRate(),
                 fmt.channelCount(), fmt.sampleFormat().name)
        return self.job_id

    def stop(self):
        """Para a captura; a transcrição termina a última fala e o job vira DONE."""
        if self._proc is None or self._stop_at is not None:
            return
        self._stop_at = time.monotonic()
        self._close_capture()
        try:
            self._audio_q.put(None)
        except Exception:  # noqa: BLE001
            pass
        self._stop.set()

    def shutdown(self, timeout: float = 8.0):
        """Fechamento da janela: para e aguarda (bloqueante, com limite)."""
        if self._proc is None:
            return
        self.stop()
        self._proc.join(timeout)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(2)
            self._force_done()
        self._drain()
        self._cleanup()

    # ------------------------------------------------------------- áudio
    def _on_audio(self):
        if self._io is None:
            return
        raw = self._io.readAll()
        data = raw.data() if isinstance(raw, QByteArray) else bytes(raw)
        if not data:
            return
        x = _to_mono_16k(data, self._fmt, self._rs_state)
        if not len(x):
            return
        x = np.clip(x, -1.0, 1.0)
        pcm = (x * 32767.0).astype(np.int16)
        try:
            self._wav.write(pcm)
        except Exception:  # noqa: BLE001
            log.exception("Falha ao gravar o WAV ao vivo")
        self._samples += len(pcm)
        try:
            self._audio_q.put(pcm.tobytes())
        except Exception:  # noqa: BLE001
            pass
        peak = float(np.abs(x).max())
        self._peak = max(self._peak, peak)
        rms = float(np.sqrt(np.mean(x * x)))
        # Escala perceptiva: ~-50 dBFS → 0, 0 dBFS → 1.
        db = 20 * np.log10(max(rms, 1e-5))
        self.level.emit(float(min(1.0, max(0.0, (db + 50) / 44))))
        now = time.monotonic()
        if now - self._last_flush > 2.0:
            self._last_flush = now
            try:
                self._wav.flush()
            except Exception:  # noqa: BLE001
                pass
        if (not self._warned_silent and self.elapsed > NO_SIGNAL_S and self._peak < SILENT_PEAK):
            self._warned_silent = True
            self.no_signal.emit()

    def _close_capture(self):
        if self._source is not None:
            try:
                self._on_audio()  # o que ainda estiver no buffer
            except Exception:  # noqa: BLE001
                pass
            try:
                if self._io is not None:
                    self._io.readyRead.disconnect(self._on_audio)
            except (RuntimeError, TypeError):
                pass
            self._source.stop()
            self._source.deleteLater()
            self._source = None
            self._io = None
        if self._wav is not None:
            try:
                self._wav.close()
            except Exception:  # noqa: BLE001
                pass
            self._wav = None

    # ------------------------------------------------------------ eventos
    def _drain(self, limit: int = 300):
        if self._events is None:
            return
        for _ in range(limit):
            try:
                kind, payload = self._events.get_nowait()
            except queue.Empty:
                return
            except (EOFError, OSError):
                return
            if kind in ("done", "error"):
                self._terminal = (kind, payload)
            self.event.emit(kind, payload)

    def _pump(self):
        self._drain()
        proc = self._proc
        if proc is None:
            self._timer.stop()
            return
        if self._stop_at is not None and proc.is_alive() and time.monotonic() - self._stop_at > STOP_GRACE_S:
            log.warning("Processo ao vivo não terminou a tempo; encerrando.")
            proc.terminate()
            proc.join(2)
            self._force_done()
        if proc.is_alive():
            if self._terminal is not None and self._stop_at is None:
                # Erro no processo com captura ainda ativa: encerra a captura.
                self._close_capture()
            return
        proc.join(0.1)
        self._drain()
        jid, term, stopping = self.job_id, self._terminal, self._stop_at is not None
        self._close_capture()
        self._cleanup()
        if term is not None and term[0] == "done":
            self.finished.emit(jid, "done", term[1])
        elif term is not None:
            code = (term[1] or {}).get("code", "")
            self.finished.emit(jid, "cancelled" if stopping and code == "Cancelled" else "failed", term[1])
        else:
            self._force_done(jid)
            self.finished.emit(jid, "done" if stopping else "failed",
                               None if stopping else {"code": "LIVE_CRASH", "message": "processo encerrado"})

    def _force_done(self, jid: str | None = None):
        """O áudio está salvo no WAV: o que já foi transcrito vira um job concluído."""
        jid = jid or self.job_id
        if not jid:
            return
        try:
            j = service.journal()
            row = j.job(jid)
            if row is not None and row["state"] != "DONE":
                j.set_state(jid, "DONE", total_samples=self._samples, media_duration=self._samples / SR)
        except Exception:  # noqa: BLE001
            log.exception("Falha ao concluir a sessão ao vivo")

    def _cleanup(self):
        self._timer.stop()
        for q in (self._audio_q, self._events):
            if q is not None:
                try:
                    q.cancel_join_thread()
                    q.close()
                except Exception:  # noqa: BLE001
                    pass
        self._proc = None
        self._audio_q = None
        self._events = None
        self._stop = None
        self._stop_at = None


def discard_empty(job_id: str, wav_path: str) -> bool:
    """Sessão sem nenhuma fala: remove o job e a gravação."""
    try:
        if service.load_segments(job_id):
            return False
        service.delete_job(job_id)
        if wav_path and os.path.isfile(wav_path):
            os.remove(wav_path)
        return True
    except Exception:  # noqa: BLE001
        log.exception("Falha ao descartar sessão vazia")
        return False


# ============================================================================ painel


class LevelMeter(ThemeAware, QWidget):
    """Barras finas que acompanham o nível do microfone: sobem rápido, descem devagar."""

    BARS = 28

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(22)
        self._hist = [0.0] * self.BARS
        self._target = 0.0
        self._value = 0.0
        self._t = QTimer(self)
        self._t.setInterval(40)
        self._t.timeout.connect(self._tick)
        self._watch_theme()

    def set_level(self, v: float):
        self._target = max(self._target, float(v))

    def start(self):
        self._hist = [0.0] * self.BARS
        self._t.start()

    def stop(self):
        self._t.stop()
        self._hist = [0.0] * self.BARS
        self.update()

    def _tick(self):
        a = 0.55 if self._target > self._value else 0.18
        self._value += (self._target - self._value) * a
        self._target *= 0.6
        self._hist = self._hist[1:] + [self._value]
        self.update()

    def paintEvent(self, _e):
        t = theme.tokens()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        w, h = self.width(), self.height()
        gap = 3.0
        bw = max(2.0, (w - gap * (self.BARS - 1)) / self.BARS)
        for i, v in enumerate(self._hist):
            bh = max(3.0, (h - 2) * min(1.0, v))
            age = (i + 1) / self.BARS
            p.setBrush(qcolor(t["accent"], 0.25 + 0.65 * age) if v > 0.04 else qcolor(t["text3"], 0.25))
            x = i * (bw + gap)
            p.drawRoundedRect(QRectF(x, (h - bh) / 2, bw, bh), bw / 2, bw / 2)


class LivePanel(QWidget):
    """Cartão que substitui o botão de ação durante a gravação ao vivo."""

    stop_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        frame, fl = card("subtleCard", (14, 12, 14, 12), 10)
        top = QHBoxLayout()
        top.setSpacing(9)
        self._dot = PulseDot("danger", 14)
        top.addWidget(self._dot, 0, Qt.AlignmentFlag.AlignVCenter)
        self._title = QLabel(tr("live_recording"))
        self._title.setObjectName("cardTitle")
        top.addWidget(self._title, 1)
        self._clock = QLabel("00:00")
        self._clock.setObjectName("bigNumber")
        self._clock.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        top.addWidget(self._clock)
        fl.addLayout(top)
        self._meter = LevelMeter()
        fl.addWidget(self._meter)
        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        self._status = QLabel("")
        self._status.setObjectName("faint")
        self._status.setWordWrap(True)
        bottom.addWidget(self._status, 1)
        self._stop_btn = QPushButton(tr("live_stop"))
        self._stop_btn.setObjectName("danger")
        self._stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._stop_btn.clicked.connect(self.stop_requested)
        bottom.addWidget(self._stop_btn, 0, Qt.AlignmentFlag.AlignBottom)
        fl.addLayout(bottom)
        layout.addWidget(frame)
        self._elapsed_fn = None
        self._clock_t = QTimer(self)
        self._clock_t.setInterval(250)
        self._clock_t.timeout.connect(self._update_clock)
        theme.manager().changed.connect(self._on_theme)
        self._on_theme()

    def _on_theme(self, *_):
        self._stop_btn.setIcon(icons.icon("stop", theme.tokens()["danger"], 14))

    def retranslate(self):
        self._title.setText(tr("live_recording"))
        self._stop_btn.setText(tr("live_stop"))

    def start(self, elapsed_fn):
        self._elapsed_fn = elapsed_fn
        self._stop_btn.setEnabled(True)
        self._dot.set_color_key("danger")
        self._title.setText(tr("live_recording"))
        self.set_status(tr("live_starting"))
        self._meter.start()
        self._clock_t.start()
        self._update_clock()

    def set_stopping(self):
        self._stop_btn.setEnabled(False)
        self._dot.set_color_key("warning")
        self._title.setText(tr("live_finishing"))
        self._meter.stop()

    def stop(self):
        self._clock_t.stop()
        self._meter.stop()

    def set_status(self, text: str):
        self._status.setText(text)

    def set_level(self, v: float):
        self._meter.set_level(v)

    def _update_clock(self):
        s = int(self._elapsed_fn() if self._elapsed_fn else 0)
        h, m, sec = s // 3600, (s // 60) % 60, s % 60
        self._clock.setText(f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}")
