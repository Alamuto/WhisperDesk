"""Executa um job de transcrição num processo separado (spawn) e repassa os eventos à GUI.

Isola a UI de falhas nativas (CUDA/cuDNN, decodificadores): se o processo morrer sem relatar
um estado terminal, o job é reiniciado a partir do checkpoint no journal (até MAX_RESTARTS vezes;
a partir da segunda falha, força CPU)."""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import queue
import time

from PySide6.QtCore import QObject, QTimer, Signal

from app import paths
from app.pipeline import service
from app.pipeline.worker import run_job

log = logging.getLogger(__name__)

MAX_RESTARTS = 3
CANCEL_GRACE_S = 10.0
_TERMINAL = ("done", "paused", "error")


class JobController(QObject):
    event = Signal(str, object)           # (kind, payload) de cada evento do worker
    finished = Signal(str, str, object)   # (job_id, outcome: done|paused|failed, payload)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ctx = mp.get_context("spawn")
        self._proc = None
        self._queue = None
        self._cancel = None
        self._job_id: str | None = None
        self._restarts = 0
        self._terminal: tuple[str, object] | None = None
        self._cancel_at: float | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._pump)

    # ------------------------------------------------------------------ API
    @property
    def job_id(self) -> str | None:
        return self._job_id

    def is_running(self) -> bool:
        return self._proc is not None

    def start(self, job_id: str) -> None:
        if self._proc is not None:
            raise RuntimeError("Já existe um job em execução.")
        self._job_id = job_id
        self._restarts = 0
        self._spawn()

    def cancel(self) -> None:
        """Pausa cooperativa (o progresso fica salvo); força o término após CANCEL_GRACE_S."""
        if self._proc is None or self._cancel is None:
            return
        if self._cancel_at is None:
            self._cancel.set()
            self._cancel_at = time.monotonic()

    def shutdown(self, timeout: float = CANCEL_GRACE_S) -> None:
        """Usado no fechamento da janela: pausa e aguarda (bloqueante, com limite)."""
        if self._proc is None:
            return
        self.cancel()
        self._proc.join(timeout)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(3)
            self._mark_paused()
        self._drain()
        self._cleanup()

    # ------------------------------------------------------------- interno
    def _spawn(self) -> None:
        self._queue = self._ctx.Queue()
        self._cancel = self._ctx.Event()
        self._terminal = None
        self._cancel_at = None
        self._proc = self._ctx.Process(
            target=run_job,
            args=(self._job_id, str(paths.journal_path()), str(paths.models_dir()),
                  str(paths.logs_dir()), self._queue, self._cancel),
            name=f"whisperdesk-job-{self._job_id[:8]}",
            daemon=True,
        )
        self._proc.start()
        self._timer.start()

    def _drain(self, limit: int = 500) -> None:
        if self._queue is None:
            return
        for _ in range(limit):
            try:
                kind, payload = self._queue.get_nowait()
            except queue.Empty:
                return
            except (EOFError, OSError):
                return
            if kind in _TERMINAL:
                self._terminal = (kind, payload)
            self.event.emit(kind, payload)

    def _pump(self) -> None:
        self._drain()
        proc = self._proc
        if proc is None:
            self._timer.stop()
            return
        if self._cancel_at is not None and proc.is_alive() and time.monotonic() - self._cancel_at > CANCEL_GRACE_S:
            log.warning("Worker não respondeu ao cancelamento; encerrando o processo.")
            proc.terminate()
            proc.join(3)
            self._mark_paused()
            self._terminal = ("paused", None)
        if proc.is_alive():
            return
        proc.join(0.1)
        self._drain()  # eventos finais ainda no pipe
        code = proc.exitcode
        jid = self._job_id
        term = self._terminal
        cancelled = self._cancel_at is not None
        self._cleanup()

        if term is not None:
            kind, payload = term
            outcome = {"done": "done", "paused": "paused", "error": "failed"}[kind]
            self.finished.emit(jid, outcome, payload)
            return
        if cancelled:
            self._mark_paused(jid)
            self.finished.emit(jid, "paused", None)
            return
        # Morte sem estado terminal: falha nativa (ex.: driver/cuDNN) ou falta de memória do SO.
        self._restarts += 1
        if self._restarts > MAX_RESTARTS:
            msg = (f"O processo de transcrição terminou inesperadamente {MAX_RESTARTS + 1} vezes "
                   f"(código {code}). O progresso foi salvo; tente novamente ou use CPU.")
            try:
                service.journal().set_state(jid, "FAILED", error=f"WORKER_CRASH exit={code}")
            except Exception:
                pass
            self.finished.emit(jid, "failed", {"code": "WORKER_CRASH", "message": msg})
            return
        note = f"O processo de transcrição terminou inesperadamente (código {code}); retomando do último checkpoint…"
        if self._restarts >= 2:
            self._force_cpu(jid)
            note += " (modo CPU)"
        self.event.emit("warning", note)
        self._job_id = jid
        self._spawn()

    def _force_cpu(self, jid: str) -> None:
        try:
            j = service.journal()
            row = j.job(jid)
            cfg = json.loads(row["config_json"])
            cfg["device"] = "cpu"
            j.update_fields(jid, config_json=json.dumps(cfg))
        except Exception:
            log.exception("Falha ao forçar CPU")

    def _mark_paused(self, jid: str | None = None) -> None:
        jid = jid or self._job_id
        if not jid:
            return
        try:
            j = service.journal()
            row = j.job(jid)
            if row is not None and row["state"] not in ("DONE", "FAILED"):
                j.set_state(jid, "PAUSED")
        except Exception:
            log.exception("Falha ao marcar job como pausado")

    def _cleanup(self) -> None:
        self._timer.stop()
        if self._queue is not None:
            try:
                self._queue.close()
                self._queue.join_thread()
            except Exception:
                pass
        self._proc = None
        self._queue = None
        self._cancel = None
        self._cancel_at = None
