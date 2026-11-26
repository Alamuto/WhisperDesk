"""Cartão de progresso (etapa, %, ETA, velocidade, motor, VRAM, pausar) e painel de log técnico."""
from __future__ import annotations

import time

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication, QTextCursor
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

from app.core.gpu_monitor import get_gpu_info
from app.gui import icons, theme
from app.gui.widgets import IconButton, PulseDot, SmoothProgress, card
from app.utils.i18n import tr

_MAX_LOG_LINES = 2000


class LogPanel(QWidget):
    close_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 10, 12, 12)
        layout.setSpacing(8)
        head = QHBoxLayout()
        head.setSpacing(6)
        self._header = QLabel(tr("tech_log"))
        self._header.setObjectName("cardTitle")
        head.addWidget(self._header)
        self._count = QLabel("")
        self._count.setObjectName("faint")
        head.addWidget(self._count)
        head.addStretch()
        self._copy_btn = IconButton("copy", tr("copy_all"), 28, 15)
        self._copy_btn.clicked.connect(self._copy)
        head.addWidget(self._copy_btn)
        self._close_btn = IconButton("close", tr("close"), 28, 15)
        self._close_btn.clicked.connect(self.close_requested)
        head.addWidget(self._close_btn)
        layout.addLayout(head)
        self._log_text = QPlainTextEdit()
        self._log_text.setObjectName("log")
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumBlockCount(_MAX_LOG_LINES)  # descarte O(1) das linhas antigas
        self._log_text.setPlaceholderText(tr("status_log"))
        self._log_text.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        layout.addWidget(self._log_text, stretch=1)
        self._lines = 0

    def retranslate(self):
        self._header.setText(tr("tech_log"))
        self._copy_btn.setToolTip(tr("copy_all"))
        self._close_btn.setToolTip(tr("close"))
        self._log_text.setPlaceholderText(tr("status_log"))

    def _copy(self):
        QGuiApplication.clipboard().setText(self.text())
        w = self.window()
        if hasattr(w, "show_toast"):
            w.show_toast(tr("log_copied"), kind="success", duration_ms=2500)

    def append(self, message: str):
        sb = self._log_text.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 4
        self._log_text.appendPlainText(f"{time.strftime('%H:%M:%S')}  {message}")
        self._lines = min(_MAX_LOG_LINES, self._lines + 1)
        self._count.setText(str(self._lines))
        if at_bottom:
            self._log_text.moveCursor(QTextCursor.MoveOperation.End)
            sb.setValue(sb.maximum())

    def clear(self):
        self._log_text.clear()
        self._lines = 0
        self._count.setText("")

    def text(self) -> str:
        return self._log_text.toPlainText()


class ProgressArea(QWidget):
    """Cartão compacto que substitui o botão "Transcrever" enquanto um job roda."""

    cancel_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._t0 = 0.0
        self._f0 = 0.0
        self._active = False
        self._speed = ""
        self._eta = ""
        self._setup_ui()
        self._gpu_timer = QTimer(self)
        self._gpu_timer.setInterval(2000)
        self._gpu_timer.timeout.connect(self._update_vram)
        theme.manager().changed.connect(self._on_theme)
        self.reset()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        frame, fl = card("subtleCard", (14, 12, 14, 12), 9)

        top = QHBoxLayout()
        top.setSpacing(9)
        self._dot = PulseDot("accent", 14)
        top.addWidget(self._dot, 0, Qt.AlignmentFlag.AlignVCenter)
        self._stage_label = QLabel()
        self._stage_label.setObjectName("cardTitle")
        top.addWidget(self._stage_label, 1)
        self._pct_label = QLabel("0%")
        self._pct_label.setObjectName("bigNumber")
        self._pct_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        top.addWidget(self._pct_label)
        fl.addLayout(top)

        self._bar = SmoothProgress(6)
        fl.addWidget(self._bar)

        self._eta_label = QLabel()
        self._eta_label.setObjectName("muted")
        fl.addWidget(self._eta_label)

        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        self._engine_label = QLabel()
        self._engine_label.setObjectName("faint")
        self._engine_label.setWordWrap(True)
        bottom.addWidget(self._engine_label, 1)
        self._cancel_btn = QPushButton(tr("pause_job"))
        self._cancel_btn.setObjectName("ghost")
        self._cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._cancel_btn.setToolTip(tr("pause_job_tip"))
        self._cancel_btn.clicked.connect(self._on_cancel)
        bottom.addWidget(self._cancel_btn, 0, Qt.AlignmentFlag.AlignBottom)
        fl.addLayout(bottom)
        layout.addWidget(frame)
        self._on_theme()

    def _on_theme(self, *_):
        self._cancel_btn.setIcon(icons.icon("pause", theme.tokens()["text2"], 14, 2.0))

    def retranslate(self):
        self._cancel_btn.setText(tr("pausing") if not self._cancel_btn.isEnabled() else tr("pause_job"))
        self._cancel_btn.setToolTip(tr("pause_job_tip"))
        if not self._active:
            self._stage_label.setText(tr("idle"))
        self._refresh_eta()

    def _on_cancel(self):
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.setText(tr("pausing"))
        self._dot.set_color_key("warning")
        self.cancel_requested.emit()

    def reset(self):
        self._bar.set_indeterminate(False)
        self._bar.set_value(0, animate=False)
        self._pct_label.setText("0%")
        self._eta = ""
        self._speed = ""
        self._refresh_eta()
        self._engine_label.setText("")
        self._vram = ""
        self._engine = ""
        self._stage_label.setText(tr("idle"))
        self._dot.set_color_key("accent")
        self._t0 = 0.0
        self._f0 = 0.0

    def start(self):
        self.reset()
        self._active = True
        self._bar.set_active(True)
        self._cancel_btn.setText(tr("pause_job"))
        self._cancel_btn.setEnabled(True)
        self._cancel_btn.setVisible(True)
        self._gpu_timer.start()
        self._update_vram()

    def stop(self):
        self._active = False
        self._bar.set_active(False)
        self._bar.set_indeterminate(False)
        self._cancel_btn.setVisible(False)
        self._gpu_timer.stop()

    @property
    def active(self) -> bool:
        return self._active

    def set_stage(self, text: str, indeterminate: bool = False):
        self._stage_label.setText(text)
        self._bar.set_indeterminate(indeterminate)
        if indeterminate:
            self._pct_label.setText("")

    def set_stage_fraction(self, fraction: float):
        """Progresso de uma etapa preparatória (extração/planejamento/download), sem ETA global."""
        self._bar.set_indeterminate(False)
        f = max(0.0, min(1.0, fraction))
        self._bar.set_value(f, animate=f >= self._bar.value())
        self._pct_label.setText(f"{int(f * 100)}%")

    def begin_transcription(self, fraction: float):
        """Marca a origem da estimativa de ETA (retomadas começam com fração > 0)."""
        self._t0 = time.monotonic()
        self._f0 = max(0.0, min(1.0, fraction))
        self._eta = tr("calculating")
        self._bar.set_value(0, animate=False)
        self.update_progress(self._f0)

    def update_progress(self, fraction: float):
        self._bar.set_indeterminate(False)
        f = max(0.0, min(1.0, fraction))
        self._bar.set_value(f)
        self._pct_label.setText(f"{int(f * 100)}%" if f < 0.995 else "100%")
        if self._t0 <= 0:
            self._t0 = time.monotonic()
            self._f0 = f
            return
        elapsed = time.monotonic() - self._t0
        gained = f - self._f0
        if elapsed > 3 and gained > 0.002:
            remaining = (1.0 - f) * elapsed / gained
            self._eta = tr("eta_remaining", time=self.format_duration(remaining))
        self._refresh_eta()

    def update_speed(self, speed_x: float):
        if speed_x and speed_x > 0:
            self._speed = tr("speed_realtime", speed=f"{speed_x:.1f}")
            self._refresh_eta()

    def _refresh_eta(self):
        parts = [p for p in (self._eta, self._speed) if p]
        self._eta_label.setText("  ·  ".join(parts))
        self._eta_label.setVisible(bool(parts))

    def set_engine(self, label: str):
        self._engine = label or ""
        self._refresh_engine()

    def _refresh_engine(self):
        parts = [p for p in (getattr(self, "_engine", ""), getattr(self, "_vram", "")) if p]
        self._engine_label.setText("  ·  ".join(parts))

    def _update_vram(self):
        info = get_gpu_info()
        self._vram = f"VRAM {info.display_str}" if info else ""
        self._refresh_engine()

    @staticmethod
    def format_duration(seconds: float) -> str:
        total = int(max(0, seconds))
        if total < 60:
            return f"{total}s"
        m, s = divmod(total, 60)
        if m < 60:
            return f"{m}m {s:02d}s"
        h, m = divmod(m, 60)
        return f"{h}h {m:02d}m"
