"""Histórico: transcrições concluídas, lidas do journal.

HistoryPanel vai na gaveta lateral (busca + lista completa); RecentList é a versão compacta
mostrada na tela inicial. Ambas usam o mesmo delegate pintado à mão."""
from __future__ import annotations

import json
import os
import subprocess
import time

from PySide6.QtCore import QEvent, QModelIndex, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter
from PySide6.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
    QPushButton, QStyle, QStyledItemDelegate, QVBoxLayout, QWidget,
)

from app.gui import icons, theme
from app.gui.theme import qcolor
from app.gui.widgets import IconButton, IconTile, RoundedMenu
from app.pipeline import service
from app.utils.i18n import tr

_VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".wmv", ".flv", ".ts", ".mpg", ".mpeg", ".3gp"}

R_ID = Qt.ItemDataRole.UserRole
R_SRC = Qt.ItemDataRole.UserRole + 1
R_NAME = Qt.ItemDataRole.UserRole + 2
R_META = Qt.ItemDataRole.UserRole + 3
R_EXISTS = Qt.ItemDataRole.UserRole + 4
R_VIDEO = Qt.ItemDataRole.UserRole + 5


def _fmt_dur(seconds) -> str:
    total = int(seconds or 0)
    if total <= 0:
        return "--"
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _fmt_when(ts: float) -> str:
    if not ts:
        return ""
    lt = time.localtime(ts)
    now = time.localtime()
    hm = time.strftime("%H:%M", lt)
    day = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
    today = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, 0, 0, 0, 0, 0, -1))
    diff = round((today - day) / 86400)
    if diff == 0:
        return f"{tr('today')} {hm}"
    if diff == 1:
        return f"{tr('yesterday')} {hm}"
    if lt.tm_year == now.tm_year:
        return time.strftime("%d/%m", lt) + f" {hm}"
    return time.strftime("%d/%m/%Y", lt)


def load_rows(limit: int = 500) -> list[dict]:
    """Linhas do histórico já formatadas para exibição."""
    try:
        rows = service.history(limit=limit)
    except Exception:
        rows = []
    out = []
    for r in rows:
        try:
            cfg = json.loads(r["config_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            cfg = {}
        lang = (cfg.get("language") or r["detected_language"] or "").upper()
        src = r["src_path"] or ""
        parts = [_fmt_when(r["finished_at"] or r["updated_at"] or 0), _fmt_dur(r["media_duration"]),
                 cfg.get("model_size", ""), lang]
        out.append({
            "id": r["id"], "src": src, "name": r["display_name"] or os.path.basename(src),
            "meta": "  ·  ".join(p for p in parts if p),
            "exists": os.path.isfile(src),
            "video": os.path.splitext(src)[1].lower() in _VIDEO_EXT,
        })
    return out


def _make_item(row: dict) -> QListWidgetItem:
    item = QListWidgetItem(row["name"])
    item.setData(R_ID, row["id"])
    item.setData(R_SRC, row["src"])
    item.setData(R_NAME, row["name"])
    item.setData(R_META, row["meta"])
    item.setData(R_EXISTS, row["exists"])
    item.setData(R_VIDEO, row["video"])
    item.setToolTip(row["src"] + ("" if row["exists"] else f"\n{tr('source_missing')}"))
    return item


class HistoryDelegate(QStyledItemDelegate):
    def __init__(self, view: QListWidget, height: int = 58):
        super().__init__(view)
        self._view = view
        self._h = height

    def sizeHint(self, option, index):
        return QSize(self._view.viewport().width(), self._h)

    def paint(self, p: QPainter, option, index: QModelIndex):
        t = theme.tokens()
        dark = theme.is_dark()
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(option.rect).adjusted(4, 2, -4, -2)
        hover = bool(option.state & QStyle.StateFlag.State_MouseOver)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        p.setPen(Qt.PenStyle.NoPen)
        if selected:
            p.setBrush(qcolor(t["accent"], 0.16 if dark else 0.10))
            p.drawRoundedRect(r, 10, 10)
        elif hover:
            p.setBrush(QColor(t["surface3"] if dark else t["surface2"]))
            p.drawRoundedRect(r, 10, 10)

        exists = bool(index.data(R_EXISTS))
        tile = QRectF(r.left() + 8, r.center().y() - 17, 34, 34)
        soft = "accent" if exists else "text3"
        p.setBrush(qcolor(t[soft], (0.16 if dark else 0.10) if exists else 0.12))
        p.drawRoundedRect(tile, 9, 9)
        icon = "file-video" if index.data(R_VIDEO) else "file-audio"
        icons.draw(p, icon, tile.adjusted(8, 8, -8, -8), QColor(t["accent_fg" if exists else "text3"]))

        x = tile.right() + 12
        right_pad = 30
        tw = int(r.right() - x - right_pad)
        f = QFont(option.font)
        f.setPixelSize(13)
        f.setWeight(QFont.Weight.DemiBold)
        fm = QFontMetrics(f)
        p.setFont(f)
        p.setPen(QColor(t["text"]))
        name = fm.elidedText(index.data(R_NAME) or "", Qt.TextElideMode.ElideMiddle, tw)
        p.drawText(QRectF(x, r.top() + 9, tw, 20), int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), name)
        f2 = QFont(option.font)
        f2.setPixelSize(12)
        fm2 = QFontMetrics(f2)
        p.setFont(f2)
        p.setPen(QColor(t["text3"]))
        meta = index.data(R_META) or ""
        if not exists:
            meta = f"{tr('source_missing_short')}  ·  {meta}"
        meta = fm2.elidedText(meta, Qt.TextElideMode.ElideRight, tw)
        p.drawText(QRectF(x, r.bottom() - 27, tw, 18), int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), meta)

        if hover or selected:
            icons.draw(p, "chevron-right", QRectF(r.right() - 24, r.center().y() - 8, 16, 16),
                       QColor(t["text2"]), 1.8)
        elif not exists:
            icons.draw(p, "warning", QRectF(r.right() - 24, r.center().y() - 8, 16, 16), QColor(t["warning"]), 1.6)
        p.restore()


class _List(QListWidget):
    """QListWidget com hover por item (repinta ao mover o mouse) e sem moldura."""

    def __init__(self, row_height: int, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.viewport().setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setUniformItemSizes(True)
        self.setItemDelegate(HistoryDelegate(self, row_height))
        self.viewport().setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        theme.manager().changed.connect(lambda *_: self.viewport().update())

    def viewportEvent(self, e):
        if e.type() in (QEvent.Type.HoverMove, QEvent.Type.HoverLeave):
            self.viewport().update()
        return super().viewportEvent(e)


def _context_menu(owner: QWidget, item: QListWidgetItem, global_pos, on_load, on_deleted):
    jid = item.data(R_ID)
    src = item.data(R_SRC) or ""
    menu = RoundedMenu(parent=owner)
    t = theme.tokens()
    act_load = menu.addAction(icons.icon("play", t["text2"], 16), tr("load"))
    act_folder = menu.addAction(icons.icon("folder", t["text2"], 16), tr("show_in_folder"))
    act_folder.setEnabled(os.path.isfile(src) and os.name == "nt")
    menu.addSeparator()
    act_del = menu.addAction(icons.icon("trash", t["danger"], 16), tr("delete"))
    action = menu.exec(global_pos)
    if action == act_load:
        on_load(jid)
    elif action == act_folder:
        subprocess.Popen(["explorer", "/select,", os.path.normpath(src)])
    elif action == act_del:
        name = item.data(R_NAME) or ""
        if QMessageBox.question(owner, tr("delete"), tr("delete_entry_msg", name=name)) \
                == QMessageBox.StandardButton.Yes:
            service.delete_job(jid)
            on_deleted(jid)


class HistoryPanel(QWidget):
    entry_selected = Signal(str)   # job_id
    entry_deleted = Signal(str)    # job_id
    close_requested = Signal()
    rows_changed = Signal(object)  # list[dict] — para a lista de recentes

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 10, 12)
        layout.setSpacing(10)
        head = QHBoxLayout()
        head.setSpacing(6)
        self._title = QLabel(tr("history"))
        self._title.setObjectName("title")
        head.addWidget(self._title)
        self._count = QLabel("")
        self._count.setObjectName("chip")
        head.addWidget(self._count, 0, Qt.AlignmentFlag.AlignVCenter)
        head.addStretch()
        self._clear_all_btn = QPushButton(tr("clear_all"))
        self._clear_all_btn.setObjectName("ghost")
        self._clear_all_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._clear_all_btn.clicked.connect(self._on_clear_all)
        head.addWidget(self._clear_all_btn)
        self._close_btn = IconButton("close", tr("close"), 30, 16)
        self._close_btn.clicked.connect(self.close_requested)
        head.addWidget(self._close_btn)
        layout.addLayout(head)

        self._filter = QLineEdit()
        self._filter.setObjectName("search")
        self._filter.setPlaceholderText(tr("filter_history"))
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._apply_filter)
        self._search_act = self._filter.addAction(icons.icon("search", theme.tokens()["text3"], 16),
                                                  QLineEdit.ActionPosition.LeadingPosition)
        theme.manager().changed.connect(
            lambda *_: self._search_act.setIcon(icons.icon("search", theme.tokens()["text3"], 16)))
        layout.addWidget(self._filter)

        self._list = _List(58)
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._on_context_menu)
        self._list.itemActivated.connect(self._emit_item)
        self._list.itemClicked.connect(self._emit_item)
        layout.addWidget(self._list, stretch=1)

        self._empty = QWidget()
        el = QVBoxLayout(self._empty)
        el.setContentsMargins(0, 40, 0, 40)
        el.setSpacing(10)
        el.addStretch()
        tile = IconTile("history", 52, 24, "text3", "text3")
        el.addWidget(tile, 0, Qt.AlignmentFlag.AlignHCenter)
        self._empty_label = QLabel(tr("no_transcriptions"))
        self._empty_label.setObjectName("muted")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setWordWrap(True)
        el.addWidget(self._empty_label)
        el.addStretch()
        layout.addWidget(self._empty, stretch=1)
        self._no_match = QLabel(tr("no_matches"))
        self._no_match.setObjectName("hint")
        self._no_match.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._no_match.hide()
        layout.addWidget(self._no_match)

    def retranslate(self):
        self._title.setText(tr("history"))
        self._clear_all_btn.setText(tr("clear_all"))
        self._close_btn.setToolTip(tr("close"))
        self._empty_label.setText(tr("no_transcriptions"))
        self._no_match.setText(tr("no_matches"))
        self._filter.setPlaceholderText(tr("filter_history"))
        self.refresh()

    def focus_filter(self):
        self._filter.setFocus()
        self._filter.selectAll()

    def refresh(self):
        rows = load_rows(500)
        self._list.clear()
        for r in rows:
            self._list.addItem(_make_item(r))
        n = len(rows)
        self._empty.setVisible(n == 0)
        self._list.setVisible(n > 0)
        self._filter.setEnabled(n > 0)
        self._clear_all_btn.setEnabled(n > 0)
        self._count.setText(str(n))
        self._count.setVisible(n > 0)
        self._apply_filter(self._filter.text())
        self.rows_changed.emit(rows)

    def select_job(self, job_id: str | None):
        """Marca o item do job aberto (sem emitir seleção)."""
        self._list.blockSignals(True)
        self._list.clearSelection()
        if job_id:
            for i in range(self._list.count()):
                it = self._list.item(i)
                if it.data(R_ID) == job_id:
                    it.setSelected(True)
                    self._list.scrollToItem(it)
                    break
        self._list.blockSignals(False)

    def _apply_filter(self, text: str):
        q = (text or "").strip().lower()
        visible = 0
        for i in range(self._list.count()):
            it = self._list.item(i)
            hide = bool(q) and q not in (it.data(R_NAME) or "").lower() and q not in (it.data(R_META) or "").lower()
            it.setHidden(hide)
            visible += not hide
        self._no_match.setVisible(bool(q) and visible == 0 and self._list.count() > 0)

    def _emit_item(self, item: QListWidgetItem):
        jid = item.data(R_ID)
        if jid:
            self.entry_selected.emit(jid)

    def _on_deleted(self, jid: str):
        self.entry_deleted.emit(jid)
        self.refresh()

    def _on_context_menu(self, pos):
        item = self._list.itemAt(pos)
        if item is not None:
            _context_menu(self, item, self._list.viewport().mapToGlobal(pos), self.entry_selected.emit, self._on_deleted)

    def _on_clear_all(self):
        if QMessageBox.question(self, tr("clear_history"), tr("clear_history_msg")) == QMessageBox.StandardButton.Yes:
            ids = [self._list.item(i).data(R_ID) for i in range(self._list.count())]
            service.clear_history()
            for jid in ids:
                self.entry_deleted.emit(jid)
            self.refresh()


class RecentList(QWidget):
    """Até N transcrições recentes, para a tela inicial."""

    entry_selected = Signal(str)
    entry_deleted = Signal(str)
    see_all = Signal()

    def __init__(self, limit: int = 4, parent=None):
        super().__init__(parent)
        self._limit = limit
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        head = QHBoxLayout()
        self._title = QLabel(tr("recent").upper())
        self._title.setObjectName("overline")
        head.addWidget(self._title)
        head.addStretch()
        self._all_btn = QPushButton(tr("see_all"))
        self._all_btn.setObjectName("link")
        self._all_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._all_btn.clicked.connect(self.see_all)
        head.addWidget(self._all_btn)
        lay.addLayout(head)
        self._list = _List(56)
        self._list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._on_context_menu)
        self._list.itemClicked.connect(lambda it: self.entry_selected.emit(it.data(R_ID)))
        self._list.itemActivated.connect(lambda it: self.entry_selected.emit(it.data(R_ID)))
        lay.addWidget(self._list)
        self.hide()

    def set_rows(self, rows: list[dict]):
        self._list.clear()
        shown = rows[: self._limit]
        for r in shown:
            self._list.addItem(_make_item(r))
        self._list.setFixedHeight(56 * len(shown) + 4)
        self._all_btn.setVisible(len(rows) > self._limit)
        self.setVisible(bool(shown))

    def retranslate(self):
        self._title.setText(tr("recent").upper())
        self._all_btn.setText(tr("see_all"))

    def _on_context_menu(self, pos):
        item = self._list.itemAt(pos)
        if item is not None:
            _context_menu(self, item, self._list.viewport().mapToGlobal(pos), self.entry_selected.emit,
                          self.entry_deleted.emit)
