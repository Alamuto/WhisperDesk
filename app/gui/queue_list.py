"""Fila de arquivos da barra lateral: o que será transcrito depois do arquivo atual.

Cada linha entra e sai deslizando (Reveal); a fila é persistida em config ("queue") para
sobreviver a um fechamento do app. A janela principal decide quando avançar."""
from __future__ import annotations

import os

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from app.gui import theme
from app.gui.theme import mix, qcolor
from app.gui.widgets import ElidedLabel, IconButton, IconTile, Reveal, ThemeAware, _anim, section_label
from app.utils import config as app_config
from app.utils.i18n import tr

_VIDEO_EXT = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".wmv", ".m4v", ".flv", ".ts", ".mts", ".3gp"}


class _Row(ThemeAware, QWidget):
    remove_clicked = Signal(str)

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path = path
        self._hover = 0.0
        self._ha = _anim(self, 0, 1, 160, self._set_hover)
        self.setFixedHeight(44)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 4, 4)
        lay.setSpacing(10)
        video = os.path.splitext(path)[1].lower() in _VIDEO_EXT
        self._tile = IconTile("file-video" if video else "file-audio", 30, 16)
        lay.addWidget(self._tile)
        self._name = ElidedLabel(os.path.basename(path))
        self._name.setToolTip(path)
        lay.addWidget(self._name, 1)
        self._badge = QLabel("")
        self._badge.setObjectName("faint")
        lay.addWidget(self._badge)
        self._remove = IconButton("close", tr("queue_remove"), 26, 14)
        self._remove.clicked.connect(lambda: self.remove_clicked.emit(self.path))
        self._remove.setVisible(False)
        lay.addWidget(self._remove)
        self._watch_theme()

    def set_badge(self, text: str):
        self._badge.setText(text)

    def retranslate(self):
        self._remove.setToolTip(tr("queue_remove"))

    def _set_hover(self, v):
        self._hover = float(v)
        self.update()

    def _to(self, v: float):
        self._ha.stop()
        self._ha.setStartValue(self._hover)
        self._ha.setEndValue(v)
        self._ha.start()

    def enterEvent(self, e):
        self._to(1.0)
        self._remove.setVisible(self.isEnabled())
        self._badge.setVisible(False)
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._to(0.0)
        self._remove.setVisible(False)
        self._badge.setVisible(True)
        super().leaveEvent(e)

    def paintEvent(self, _e):
        t = theme.tokens()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        base = qcolor(t["surface2"], 0.0)
        p.setBrush(mix(base, QColor(t["surface2"]), self._hover))
        p.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 10, 10)


class QueueList(QWidget):
    """Lista animada de arquivos pendentes."""

    changed = Signal(int)  # quantidade na fila

    def __init__(self, parent=None):
        super().__init__(parent)
        self._paths: list[str] = []
        self._rows: dict[str, Reveal] = {}
        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 8, 0, 4)
        bl.setSpacing(2)
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 2)
        head.setSpacing(6)
        self._label = section_label(tr("queue_title"))
        head.addWidget(self._label)
        self._count = QLabel("")
        self._count.setObjectName("chip")
        head.addWidget(self._count)
        head.addStretch()
        self._add_btn = IconButton("plus", tr("queue_add"), 26, 15)
        head.addWidget(self._add_btn)
        self._clear_btn = IconButton("trash", tr("queue_clear"), 26, 15)
        self._clear_btn.clicked.connect(self.clear)
        head.addWidget(self._clear_btn)
        bl.addLayout(head)
        self._rows_box = QVBoxLayout()
        self._rows_box.setContentsMargins(0, 0, 0, 0)
        self._rows_box.setSpacing(2)
        bl.addLayout(self._rows_box)
        self._reveal = Reveal(body, shown=False)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._reveal)

    # ----------------------------------------------------------------- API
    @property
    def add_button(self) -> IconButton:
        return self._add_btn

    def items(self) -> list[str]:
        return list(self._paths)

    def __len__(self) -> int:
        return len(self._paths)

    def restore(self):
        """Recarrega a fila salva (arquivos que ainda existem)."""
        saved = app_config.get("queue") or []
        self.add([p for p in saved if isinstance(p, str) and os.path.isfile(p)], animate=False)

    def add(self, paths: list[str], animate: bool = True) -> int:
        added = 0
        for p in paths:
            p = os.path.normpath(p)
            key = os.path.normcase(p)
            if any(os.path.normcase(x) == key for x in self._paths):
                continue
            self._paths.append(p)
            row = _Row(p)
            row.remove_clicked.connect(self.remove)
            rev = Reveal(row, shown=False)
            self._rows[key] = rev
            self._rows_box.addWidget(rev)
            rev.set_shown(True, animate=animate and self._reveal.is_shown())
            added += 1
        if added:
            self._sync(animate)
        return added

    def remove(self, path: str, animate: bool = True):
        key = os.path.normcase(os.path.normpath(path))
        self._paths = [p for p in self._paths if os.path.normcase(p) != key]
        rev = self._rows.pop(key, None)
        if rev is not None and not self._paths and self._reveal.is_shown():
            # Último item: só o contêiner recolhe (efeitos de opacidade não podem se aninhar).
            self._reveal.set_shown(False, animate=animate, finished=rev.deleteLater)
        elif rev is not None:
            if animate and self.isVisible():
                rev.set_shown(False, finished=rev.deleteLater)
            else:
                rev.deleteLater()
        self._sync(animate)

    def pop_next(self) -> str | None:
        while self._paths:
            p = self._paths[0]
            self.remove(p)
            if os.path.isfile(p):
                return p
        return None

    def clear(self):
        for p in list(self._paths):
            self.remove(p)

    def retranslate(self):
        self._label.setText(tr("queue_title").upper())
        self._add_btn.setToolTip(tr("queue_add"))
        self._clear_btn.setToolTip(tr("queue_clear"))
        for rev in self._rows.values():
            rev.findChild(_Row).retranslate()
        self._sync(False, persist=False)

    # ------------------------------------------------------------ interno
    def _sync(self, animate: bool, persist: bool = True):
        n = len(self._paths)
        self._count.setText(str(n))
        for i, p in enumerate(self._paths):
            rev = self._rows.get(os.path.normcase(p))
            row = rev.findChild(_Row) if rev is not None else None
            if row is not None:
                row.set_badge(tr("queue_next") if i == 0 else "")
        if n and not self._reveal.is_shown():
            self._reveal.set_shown(True, animate=animate)
        elif not n and self._reveal.is_shown():
            self._reveal.set_shown(False, animate=animate)
        if persist:  # só mudanças reais gravam (a retradução não pode apagar a fila salva)
            app_config.set_value("queue", list(self._paths))
        self.changed.emit(n)
