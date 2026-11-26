"""Pré-visualização de forma de onda: envelope via ffmpeg em streaming (memória O(largura)),
desenhado com QPainter num QImage na thread de trabalho; o QPixmap é criado na thread da GUI."""
from __future__ import annotations

import threading

import numpy as np
from PySide6.QtCore import QPointF, QThread, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen

from app.pipeline import ffmpeg_tools


def render_peaks(peaks: np.ndarray, width: int, height: int, color: str = "#6C63FF") -> QImage:
    img = QImage(max(1, width), max(1, height), QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    if peaks.size == 0:
        return img
    p = peaks / max(float(peaks.max()), 1e-4)
    p = np.sqrt(p)  # realça trechos de fala baixa
    xs = np.linspace(0, width - 1, num=len(p))
    mid = height / 2.0
    amp = mid - 2
    path = QPainterPath(QPointF(0, mid))
    for x, v in zip(xs, p):
        path.lineTo(QPointF(float(x), mid - float(v) * amp))
    for x, v in zip(xs[::-1], p[::-1]):
        path.lineTo(QPointF(float(x), mid + float(v) * amp))
    path.closeSubpath()
    qp = QPainter(img)
    qp.setRenderHint(QPainter.RenderHint.Antialiasing)
    c = QColor(color)
    fill = QColor(c)
    fill.setAlpha(170)
    qp.setPen(QPen(c, 1))
    qp.fillPath(path, fill)
    qp.setPen(QPen(QColor(c.red(), c.green(), c.blue(), 90), 1))
    qp.drawLine(0, int(mid), width, int(mid))
    qp.end()
    return img


class WaveformWorker(QThread):
    waveform_ready = Signal(object)  # QImage (só quando render=True)
    peaks_ready = Signal(object)     # np.ndarray normalizado 0..1
    waveform_error = Signal(str)

    def __init__(self, filepath: str, stream_index: int | None, duration: float,
                 width: int = 560, height: int = 100, parent=None, render: bool = True):
        super().__init__(parent)
        self._render = render
        self._path = filepath
        self._stream = stream_index
        self._duration = max(0.1, float(duration or 0.1))
        self._w, self._h = width, height
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            # Para arquivos muito longos, uma taxa menor mantém o custo de decodificação baixo.
            rate = 4000 if self._duration < 3600 else 1000  # abaixo de ~1 kHz a energia da fala some
            peaks = ffmpeg_tools.decode_peaks(self._path, self._stream, self._w, self._duration,
                                              cancel_event=self._cancel, rate=rate)
            if self._cancel.is_set():
                return
            if peaks.size:
                norm = np.sqrt(peaks / max(float(peaks.max()), 1e-4)).astype(np.float32)
            else:
                norm = peaks.astype(np.float32)
            self.peaks_ready.emit(norm)
            if self._render:
                self.waveform_ready.emit(render_peaks(peaks, self._w, self._h))
        except ffmpeg_tools.Cancelled:
            return
        except Exception as e:  # noqa: BLE001 — mostrado na UI
            if not self._cancel.is_set():
                self.waveform_error.emit(str(e)[:200])
