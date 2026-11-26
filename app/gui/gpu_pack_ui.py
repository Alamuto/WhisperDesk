"""Oferta e download do pacote GPU (cuBLAS/cuDNN) em segundo plano, sem travar a interface."""
from __future__ import annotations

import logging
import threading

from PySide6.QtCore import QObject, QThread, Signal

from app import gpu_pack

log = logging.getLogger("whisperdesk")


class _Downloader(QThread):
    progress = Signal(int, int)
    finished_ok = Signal()
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cancel = threading.Event()

    def run(self):
        try:
            gpu_pack.install(lambda d, t: self.progress.emit(d, t), self.cancel)
        except gpu_pack.Cancelled:
            self.failed.emit("")
            return
        except Exception as e:  # noqa: BLE001 — mostrado ao usuário
            log.exception("Falha ao instalar o pacote GPU")
            self.failed.emit(str(e)[:160] or type(e).__name__)
            return
        self.finished_ok.emit()


class GpuPackManager(QObject):
    """Encapsula o ciclo de vida do download; a janela só reage aos sinais."""
    progress = Signal(int, int)   # MB baixados, MB totais
    ready = Signal()
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: _Downloader | None = None
        self._last_mb = -1

    def is_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def start(self) -> None:
        if self.is_running():
            return
        self._last_mb = -1
        w = _Downloader(self)
        w.progress.connect(self._on_progress)
        w.finished_ok.connect(self._on_ok)
        w.failed.connect(self.failed)
        self._worker = w
        w.start()

    def _on_progress(self, done: int, total: int):
        mb = done >> 20
        if mb != self._last_mb:  # no máximo um sinal por MB
            self._last_mb = mb
            self.progress.emit(mb, total >> 20)

    def _on_ok(self):
        from app.core import gpu_monitor

        if gpu_monitor.refresh_cuda():
            self.ready.emit()
        else:
            self.failed.emit("CUDA")

    def shutdown(self, wait_ms: int = 5000) -> None:
        if self.is_running():
            self._worker.cancel.set()
            self._worker.wait(wait_ms)
