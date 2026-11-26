"""Visualização da transcrição: lista virtualizada (QListView + delegate pintado à mão).

Escala para transcrições de 12 h+ (dezenas de milhares de segmentos): só as linhas visíveis são
desenhadas, alturas são calculadas sob demanda em lotes e mantidas em cache. Suporta clique no
tempo/palavra para tocar, destaque karaokê sincronizado com o player, cores por confiança,
busca e exportação a partir do journal."""
from __future__ import annotations

import bisect
import os
import time
from collections import OrderedDict

from PySide6.QtCore import (
    QAbstractListModel, QModelIndex, QPointF, QRect, QRectF, QSize, Qt, Signal,
)
from PySide6.QtGui import (
    QAction, QColor, QFont, QFontMetrics, QGuiApplication, QIcon, QKeySequence, QPainter, QShortcut, QTextCharFormat,
    QTextLayout, QTextOption,
)
from PySide6.QtWidgets import (
    QAbstractItemView, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QListView,
    QMessageBox, QPushButton, QSizePolicy, QStyle, QStyledItemDelegate, QVBoxLayout, QWidget,
)

from app.gui import icons, theme
from app.gui.theme import delegate_colors
from app.gui.widgets import ElidedLabel, IconButton, IconTile, PulseDot, RoundedMenu, Spinner
from app.pipeline import service
from app.pipeline.exporters import EXPORT_FORMATS, fmt_short, to_txt
from app.utils import config as app_config
from app.utils.i18n import tr

PAD_V = 8
PAD_H = 14
GAP = 14
LOW_CONF = 0.50
MID_CONF = 0.75


def _norm_words(words) -> list[tuple[int, int, str, float]]:
    out = []
    for w in words or ():
        if isinstance(w, dict):
            out.append((int(w["start_ms"]), int(w["end_ms"]), w["word"], float(w.get("prob", w.get("probability", 1.0)) or 0.0)))
        else:
            s, e, txt, p = w[0], w[1], w[2], (w[3] if len(w) > 3 else 1.0)
            out.append((int(s), int(e), txt, float(p if p is not None else 1.0)))
    return out


class _Seg:
    """Segmento pronto para exibição: texto + offsets de cada palavra no texto."""
    __slots__ = ("start_ms", "end_ms", "text", "speaker", "words", "spans", "word_starts", "raw")

    def __init__(self, d: dict):
        self.raw = d
        self.start_ms = int(d["start_ms"])
        self.end_ms = int(d["end_ms"])
        self.speaker = d.get("speaker")
        self.words = _norm_words(d.get("words"))
        prefix = f"{self.speaker}: " if self.speaker else ""
        spans: list[tuple[int, int]] = []
        if self.words:
            parts = [prefix]
            pos = len(prefix)
            first = True
            for _s, _e, w, _p in self.words:
                if first:
                    w = w.lstrip()
                    first = False
                parts.append(w)
                lead = len(w) - len(w.lstrip())
                spans.append((pos + lead, pos + len(w.rstrip())))
                pos += len(w)
            self.text = "".join(parts).rstrip()
        else:
            self.text = prefix + (d.get("text") or "").strip()
        self.spans = spans
        self.word_starts = [w[0] for w in self.words]

    def as_dict(self) -> dict:
        return self.raw


class SegmentModel(QAbstractListModel):
    SegRole = Qt.ItemDataRole.UserRole + 1

    def __init__(self, parent=None):
        super().__init__(parent)
        self._segs: list[_Seg] = []
        self._starts: list[int] = []
        self.active_row = -1
        self.active_word = -1
        self.match_rows: set[int] = set()
        self.query = ""

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._segs)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        seg = self._segs[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return seg.text
        if role == self.SegRole:
            return seg
        return None

    def seg(self, row: int) -> _Seg:
        return self._segs[row]

    def segments(self) -> list[dict]:
        return [s.as_dict() for s in self._segs]

    def set_segments(self, segs: list[dict]):
        self.beginResetModel()
        self._segs = [_Seg(d) for d in segs if (d.get("text") or d.get("words"))]
        self._segs.sort(key=lambda s: s.start_ms)
        self._starts = [s.start_ms for s in self._segs]
        self.active_row = -1
        self.active_word = -1
        self.match_rows = set()
        self.endResetModel()

    def append_segments(self, segs: list[dict]):
        new = [_Seg(d) for d in segs if (d.get("text") or d.get("words"))]
        if not new:
            return
        new.sort(key=lambda s: s.start_ms)
        if self._segs and new[0].start_ms < self._segs[-1].start_ms:
            # Fora de ordem (não esperado com chunks sequenciais): recompõe tudo.
            self.set_segments([s.as_dict() for s in self._segs] + [s.as_dict() for s in new])
            return
        first = len(self._segs)
        self.beginInsertRows(QModelIndex(), first, first + len(new) - 1)
        self._segs.extend(new)
        self._starts.extend(s.start_ms for s in new)
        self.endInsertRows()

    def row_at_ms(self, ms: int) -> int:
        i = bisect.bisect_right(self._starts, ms) - 1
        if i < 0:
            return -1
        s = self._segs[i]
        return i if ms <= s.end_ms + 400 else -1

    def set_active(self, row: int, word: int):
        if row == self.active_row and word == self.active_word:
            return
        old = self.active_row
        self.active_row, self.active_word = row, word
        for r in {old, row}:
            if 0 <= r < len(self._segs):
                idx = self.index(r)
                self.dataChanged.emit(idx, idx)

    def word_count(self) -> int:
        return sum(len(s.words) if s.words else len(s.text.split()) for s in self._segs)


class SegmentDelegate(QStyledItemDelegate):
    def __init__(self, view: QListView):
        super().__init__(view)
        self._view = view
        self.colors = delegate_colors(theme.current())
        self.show_confidence = True
        self._cache: OrderedDict[tuple[int, int], tuple[QTextLayout, float]] = OrderedDict()
        self._cache_width = -1
        self._update_fonts(view.font())

    def _update_fonts(self, font: QFont):
        self._font = QFont(font)
        self._ts_font = QFont(font)
        self._ts_font.setFamilies(["Cascadia Mono", "Consolas", "Courier New"])
        self._ts_font.setPointSizeF(max(8.0, font.pointSizeF() - 1))
        fm = QFontMetrics(self._ts_font)
        self.ts_width = fm.horizontalAdvance("00:00:00") + PAD_H
        self._line_h = QFontMetrics(self._font).height()

    def set_theme(self, name: str):
        self.colors = delegate_colors(name)

    def invalidate(self):
        self._cache.clear()

    def _text_width(self, total_width: int) -> int:
        return max(80, total_width - self.ts_width - GAP - 2 * PAD_H)

    def _layout(self, row: int, seg: _Seg, width: int) -> tuple[QTextLayout, float]:
        if width != self._cache_width:
            self._cache.clear()
            self._cache_width = width
        key = (row, seg.start_ms)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        lay = QTextLayout(seg.text, self._font)
        opt = QTextOption()
        opt.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        lay.setTextOption(opt)
        lay.beginLayout()
        y = 0.0
        while True:
            line = lay.createLine()
            if not line.isValid():
                break
            line.setLineWidth(width)
            line.setPosition(QPointF(0, y))
            y += line.height()
        lay.endLayout()
        res = (lay, max(y, float(self._line_h)))
        self._cache[key] = res
        if len(self._cache) > 4000:
            self._cache.popitem(last=False)
        return res

    def sizeHint(self, option, index):
        seg: _Seg = index.data(SegmentModel.SegRole)
        w = self._text_width(self._view.viewport().width())
        _lay, h = self._layout(index.row(), seg, w)
        return QSize(self._view.viewport().width(), int(h) + 2 * PAD_V)

    def paint(self, painter: QPainter, option, index):
        model: SegmentModel = index.model()
        seg: _Seg = index.data(SegmentModel.SegRole)
        row = index.row()
        c = self.colors
        r: QRect = option.rect
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hover = bool(option.state & QStyle.StateFlag.State_MouseOver)
        active = row == model.active_row
        rr = QRectF(r).adjusted(2, 1, -2, -1)
        painter.setPen(Qt.PenStyle.NoPen)
        bg = None
        if selected:
            bg = QColor(c["selected_bg"])
        elif active:
            bg = QColor(c["active_bg"])
        elif row in model.match_rows:
            bg = QColor(c["match"])
        elif hover:
            bg = QColor(c["hover"])
        if bg is not None:
            painter.setBrush(bg)
            painter.drawRoundedRect(rr, 8, 8)
        if active:
            painter.setBrush(QColor(c["accent"]))
            painter.drawRoundedRect(QRectF(rr.left(), rr.top() + 6, 3, rr.height() - 12), 1.5, 1.5)

        # Tempo (clicável)
        painter.setFont(self._ts_font)
        painter.setPen(QColor(c["accent_fg"] if active else c["faint"]))
        ts_rect = QRect(r.left() + PAD_H, r.top() + PAD_V, self.ts_width, self._line_h)
        painter.drawText(ts_rect, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), fmt_short(seg.start_ms))

        # Texto
        tw = self._text_width(r.width())
        lay, _h = self._layout(row, seg, tw)
        formats = []
        if self.show_confidence and seg.words:
            low = QTextCharFormat()
            low.setForeground(QColor(c["low"]))
            mid = QTextCharFormat()
            mid.setForeground(QColor(c["mid"]))
            for (a, b), w in zip(seg.spans, seg.words):
                if w[3] < LOW_CONF:
                    formats.append(_range(a, b - a, low))
                elif w[3] < MID_CONF:
                    formats.append(_range(a, b - a, mid))
        if active and 0 <= model.active_word < len(seg.spans):
            a, b = seg.spans[model.active_word]
            f = QTextCharFormat()
            f.setBackground(QColor(c["word_bg"]))
            f.setForeground(QColor(c["word_fg"]))
            formats.append(_range(a, b - a, f))
        if model.query and row in model.match_rows:
            q = model.query
            low_text = seg.text.lower()
            f = QTextCharFormat()
            f.setFontUnderline(True)
            f.setFontWeight(QFont.Weight.Bold)
            start = low_text.find(q)
            while start >= 0:
                formats.append(_range(start, len(q), f))
                start = low_text.find(q, start + len(q))
        painter.setPen(QColor(c["text"]))
        painter.setFont(self._font)
        lay.draw(painter, QPointF(r.left() + PAD_H + self.ts_width + GAP, r.top() + PAD_V), formats)
        painter.restore()

    def hit_test(self, index, pos) -> tuple[str, int] | None:
        """('ts'|'word', ms) para o ponto clicado."""
        seg: _Seg = index.data(SegmentModel.SegRole)
        r = self._view.visualRect(index)
        x = pos.x() - r.left()
        y = pos.y() - r.top() - PAD_V
        if x < PAD_H + self.ts_width:
            return ("ts", seg.start_ms)
        if not seg.words:
            return ("ts", seg.start_ms)
        lay, _h = self._layout(index.row(), seg, self._text_width(r.width()))
        tx = x - (PAD_H + self.ts_width + GAP)
        for i in range(lay.lineCount()):
            line = lay.lineAt(i)
            if line.y() <= y < line.y() + line.height() or i == lay.lineCount() - 1:
                cpos = line.xToCursor(tx)
                for k, (a, b) in enumerate(seg.spans):
                    if cpos <= b:
                        return ("word", seg.words[k][0])
                return ("word", seg.words[-1][0])
        return ("ts", seg.start_ms)


def _range(start: int, length: int, fmt: QTextCharFormat):
    fr = QTextLayout.FormatRange()
    fr.start = start
    fr.length = length
    fr.format = fmt
    return fr


class _ListView(QListView):
    seek_requested = Signal(int)  # ms

    def __init__(self, parent=None):
        super().__init__(parent)
        self.last_user_scroll = 0.0
        self.setObjectName("resultList")
        self.setUniformItemSizes(False)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setLayoutMode(QListView.LayoutMode.Batched)
        self.setBatchSize(300)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setMouseTracking(True)
        self.setSpacing(0)

    def wheelEvent(self, e):
        self.last_user_scroll = time.monotonic()
        super().wheelEvent(e)

    def mouseMoveEvent(self, e):
        idx = self.indexAt(e.position().toPoint())
        cursor = Qt.CursorShape.ArrowCursor
        if idx.isValid():
            hit = self.itemDelegate().hit_test(idx, e.position().toPoint())
            if hit and hit[0] == "ts":
                cursor = Qt.CursorShape.PointingHandCursor
        self.viewport().setCursor(cursor)
        super().mouseMoveEvent(e)

    def mousePressEvent(self, e):
        idx = self.indexAt(e.position().toPoint())
        if (idx.isValid() and e.button() == Qt.MouseButton.LeftButton
                and not (e.modifiers() & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier))):
            hit = self.itemDelegate().hit_test(idx, e.position().toPoint())
            if hit and hit[0] == "ts":
                self.seek_requested.emit(hit[1])
        super().mousePressEvent(e)

    def mouseDoubleClickEvent(self, e):
        idx = self.indexAt(e.position().toPoint())
        if idx.isValid():
            hit = self.itemDelegate().hit_test(idx, e.position().toPoint())
            if hit:
                self.seek_requested.emit(hit[1])
                return
        super().mouseDoubleClickEvent(e)

    def resizeEvent(self, e):
        width_changed = e.size().width() != e.oldSize().width()
        if width_changed:
            self.itemDelegate().invalidate()
        super().resizeEvent(e)
        if width_changed:
            # QListView (fluxo vertical) não remede as linhas quando só a largura muda
            self.scheduleDelayedItemsLayout()


class ResultView(QWidget):
    timestamp_clicked = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._job_id: str | None = None
        self._live = False
        self._follow_playback = False
        self._match_list: list[int] = []
        self._match_pos = -1
        self._setup_ui()
        self._update_state()

    # ------------------------------------------------------------------ UI
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        head = QHBoxLayout()
        head.setSpacing(6)
        self._title = QLabel(tr("transcript"))
        self._title.setObjectName("cardTitle")
        head.addWidget(self._title)
        self._live_badge = QWidget()
        lb = QHBoxLayout(self._live_badge)
        lb.setContentsMargins(6, 0, 2, 0)
        lb.setSpacing(5)
        self._live_dot = PulseDot("accent", 10)
        lb.addWidget(self._live_dot)
        self._live_label = QLabel(tr("live"))
        self._live_label.setObjectName("chipAccent")
        lb.addWidget(self._live_label)
        self._live_badge.hide()
        head.addWidget(self._live_badge)
        head.addSpacing(6)
        self._stats = ElidedLabel("")
        self._stats.setObjectName("faint")
        self._stats.setMinimumWidth(0)
        self._stats.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        head.addWidget(self._stats, 1)

        self._search = QLineEdit()
        self._search.setObjectName("search")
        self._search.setPlaceholderText(tr("search_placeholder"))
        self._search.setClearButtonEnabled(True)
        self._search.setMinimumWidth(130)
        self._search.setMaximumWidth(240)
        self._search.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._search.textChanged.connect(self._on_search_changed)
        self._search.returnPressed.connect(self.find_next)
        self._search_act = self._search.addAction(QIcon(), QLineEdit.ActionPosition.LeadingPosition)
        head.addWidget(self._search, 3)
        self._match_label = QLabel("")
        self._match_label.setObjectName("hint")
        head.addWidget(self._match_label)
        self._prev_btn = IconButton("chevron-up", tr("prev_match"), 30, 16)
        self._prev_btn.clicked.connect(self.find_prev)
        self._prev_btn.hide()
        head.addWidget(self._prev_btn)
        self._next_btn = IconButton("chevron-down", tr("next_match"), 30, 16)
        self._next_btn.clicked.connect(self.find_next)
        self._next_btn.hide()
        head.addWidget(self._next_btn)
        head.addSpacing(4)
        self._copy_btn = IconButton("copy", tr("copy_all"), 32, 17)
        self._copy_btn.clicked.connect(self.copy_text)
        head.addWidget(self._copy_btn)
        self._export_btn = QPushButton(tr("export"))
        self._export_btn.setObjectName("accentSoft")
        self._export_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._export_menu = RoundedMenu(parent=self)
        self._build_export_menu()
        self._export_btn.setMenu(self._export_menu)
        head.addWidget(self._export_btn)
        self._more_btn = IconButton("more", tr("more_options"), 32, 17)
        self._more_menu = RoundedMenu(parent=self)
        self._more_menu.setToolTipsVisible(True)
        self._conf_act = QAction(tr("show_confidence"), self)
        self._conf_act.setCheckable(True)
        self._conf_act.setChecked(bool(app_config.get("show_confidence", True)))
        self._conf_act.setToolTip(tr("show_confidence_tip"))
        self._conf_act.toggled.connect(self._on_conf_toggled)
        self._more_menu.addAction(self._conf_act)
        self._more_menu.addSeparator()
        self._clear_act = self._more_menu.addAction(tr("clear"))
        self._clear_act.triggered.connect(self.clear)
        self._more_btn.clicked.connect(
            lambda: self._more_menu.exec(self._more_btn.mapToGlobal(self._more_btn.rect().bottomLeft())))
        head.addWidget(self._more_btn)
        layout.addLayout(head)

        self._model = SegmentModel(self)
        self._view = _ListView(self)
        self._view.viewport().setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self._delegate = SegmentDelegate(self._view)
        self._delegate.show_confidence = self._conf_act.isChecked()
        self._view.setItemDelegate(self._delegate)
        self._view.setModel(self._model)
        self._view.seek_requested.connect(lambda ms: self.timestamp_clicked.emit(ms / 1000.0))
        self._view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._view.customContextMenuRequested.connect(self._on_context_menu)
        layout.addWidget(self._view, stretch=1)

        # Estado vazio (antes do primeiro segmento)
        self._placeholder = QWidget()
        pl = QVBoxLayout(self._placeholder)
        pl.setContentsMargins(24, 24, 24, 24)
        pl.setSpacing(8)
        pl.addStretch()
        self._ph_tile = IconTile("sparkle", 56, 26)
        pl.addWidget(self._ph_tile, 0, Qt.AlignmentFlag.AlignHCenter)
        pl.addSpacing(6)
        self._ph_title = QLabel()
        self._ph_title.setObjectName("cardTitle")
        self._ph_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pl.addWidget(self._ph_title)
        self._ph_hint = QLabel()
        self._ph_hint.setObjectName("muted")
        self._ph_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._ph_hint.setWordWrap(True)
        pl.addWidget(self._ph_hint)
        self._ph_spinner = Spinner(20)
        pl.addWidget(self._ph_spinner, 0, Qt.AlignmentFlag.AlignHCenter)
        pl.addStretch(2)
        layout.addWidget(self._placeholder, stretch=1)

        copy_sc = QShortcut(QKeySequence.StandardKey.Copy, self._view)
        copy_sc.setContext(Qt.ShortcutContext.WidgetShortcut)
        copy_sc.activated.connect(self.copy_text)
        find_sc = QShortcut(QKeySequence.StandardKey.Find, self)
        find_sc.setContext(Qt.ShortcutContext.WindowShortcut)
        find_sc.activated.connect(lambda: (self._search.setFocus(), self._search.selectAll()))
        theme.manager().changed.connect(self._on_theme)
        self._on_theme(theme.current())

    def _on_theme(self, name: str):
        self._search_act.setIcon(icons.icon("search", theme.tokens()["text3"], 16))
        self._export_btn.setIcon(icons.icon("download", theme.tokens()["accent_fg"], 16, 2.0))
        self.set_theme(name)

    def _build_export_menu(self):
        self._export_menu.clear()
        self._export_menu.setTitle(tr("export"))
        for key, (label, _ext) in EXPORT_FORMATS.items():
            act = self._export_menu.addAction(tr(f"fmt_{key}") if tr(f"fmt_{key}") != f"fmt_{key}" else label)
            act.triggered.connect(lambda _=False, k=key: self.export(k))

    def retranslate(self):
        self._title.setText(tr("transcript"))
        self._live_label.setText(tr("live"))
        self._conf_act.setText(tr("show_confidence"))
        self._conf_act.setToolTip(tr("show_confidence_tip"))
        self._clear_act.setText(tr("clear"))
        self._copy_btn.setToolTip(tr("copy_all"))
        self._more_btn.setToolTip(tr("more_options"))
        self._prev_btn.setToolTip(tr("prev_match"))
        self._next_btn.setToolTip(tr("next_match"))
        self._export_btn.setText(tr("export"))
        self._search.setPlaceholderText(tr("search_placeholder"))
        self._build_export_menu()
        if self._model.query:
            self._recompute_matches()
        self._update_state()

    def set_theme(self, name: str):
        self._delegate.set_theme(name)
        self._view.viewport().update()

    def _on_conf_toggled(self, on: bool):
        self._delegate.show_confidence = on
        app_config.set_value("show_confidence", on)
        self._view.viewport().update()

    # -------------------------------------------------------------- dados
    @property
    def job_id(self) -> str | None:
        return self._job_id

    def has_content(self) -> bool:
        return self._model.rowCount() > 0

    def begin_job(self, job_id: str, existing: list[dict] | None = None):
        """Início/retomada de um job: mostra o que já está no journal e passa a receber ao vivo."""
        self._job_id = job_id
        self._live = True
        self._model.set_segments(existing or [])
        self._delegate.invalidate()
        self._reset_search()
        self._update_state()
        self._view.scrollToBottom()

    def end_live(self):
        self._live = False
        self._update_state()

    def show_job(self, job_id: str, segments: list[dict], title: str | None = None):
        self._job_id = job_id
        self._live = False
        self._model.set_segments(segments)
        self._delegate.invalidate()
        self._reset_search()
        self._update_state()
        self._view.scrollToTop()

    def append_segments(self, segments: list[dict]):
        sb = self._view.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 8
        self._model.append_segments(segments)
        if self._model.query:
            self._recompute_matches()
        self._update_state()
        if at_bottom:
            self._view.scrollToBottom()

    def clear(self):
        self._job_id = None
        self._live = False
        self._model.set_segments([])
        self._delegate.invalidate()
        self._reset_search()
        self._update_state()

    def segments(self) -> list[dict]:
        return self._model.segments()

    def _update_state(self):
        n = self._model.rowCount()
        self._view.setVisible(n > 0)
        self._placeholder.setVisible(n == 0)
        if n == 0:
            waiting = self._live
            self._ph_title.setText(tr("waiting_title") if waiting else tr("ready_title"))
            self._ph_hint.setText(tr("waiting_hint") if waiting else tr("ready_hint"))
            self._ph_tile.set_icon("wave" if waiting else "sparkle")
            self._ph_spinner.setVisible(waiting)
        self._live_badge.setVisible(self._live)
        has_job = self._job_id is not None and n > 0
        self._export_btn.setEnabled(has_job and not self._live)
        self._copy_btn.setEnabled(n > 0)
        self._clear_act.setEnabled(n > 0 and not self._live)
        self._search.setEnabled(n > 0)
        if n:
            words = self._model.word_count()
            last = self._model.seg(n - 1).end_ms
            self._stats.setText(f"{fmt_short(last)}  ·  {words} {tr('words')}")
            self._stats.setToolTip(f"{n} {tr('segments')}")
        else:
            self._stats.setText("")

    # -------------------------------------------------------- reprodução
    def set_follow_playback(self, on: bool):
        self._follow_playback = on

    def highlight_segment(self, seconds: float):
        ms = int(seconds * 1000)
        row = self._model.row_at_ms(ms)
        word = -1
        if row >= 0:
            seg = self._model.seg(row)
            if seg.word_starts:
                word = max(0, bisect.bisect_right(seg.word_starts, ms) - 1)
        prev = self._model.active_row
        self._model.set_active(row, word)
        if (row >= 0 and row != prev and self._follow_playback
                and time.monotonic() - self._view.last_user_scroll > 3.0):
            self._view.scrollTo(self._model.index(row), QAbstractItemView.ScrollHint.EnsureVisible)

    # -------------------------------------------------------------- busca
    def _reset_search(self):
        self._model.query = ""
        self._model.match_rows = set()
        self._match_list = []
        self._match_pos = -1
        self._search.blockSignals(True)
        self._search.clear()
        self._search.blockSignals(False)
        self._match_label.setText("")
        self._prev_btn.setVisible(False)
        self._next_btn.setVisible(False)

    def _on_search_changed(self, text: str):
        self._model.query = text.strip().lower()
        self._recompute_matches()
        self._match_pos = -1
        if self._match_list:
            self.find_next()
        self._view.viewport().update()

    def _recompute_matches(self):
        q = self._model.query
        if not q:
            self._match_list = []
            self._model.match_rows = set()
            self._match_label.setText("")
            self._prev_btn.setVisible(False)
            self._next_btn.setVisible(False)
            return
        rows = [i for i in range(self._model.rowCount()) if q in self._model.seg(i).text.lower()]
        self._match_list = rows
        self._model.match_rows = set(rows)
        self._match_label.setText(f"{len(rows)} {tr('matches')}" if rows else tr("no_matches"))
        self._prev_btn.setVisible(bool(rows))
        self._next_btn.setVisible(bool(rows))

    def _goto_match(self):
        if not self._match_list:
            return
        row = self._match_list[self._match_pos]
        idx = self._model.index(row)
        self._view.scrollTo(idx, QAbstractItemView.ScrollHint.PositionAtCenter)
        self._view.setCurrentIndex(idx)
        self._match_label.setText(f"{self._match_pos + 1}/{len(self._match_list)}")

    def find_next(self):
        if self._match_list:
            self._match_pos = (self._match_pos + 1) % len(self._match_list)
            self._goto_match()

    def find_prev(self):
        if self._match_list:
            self._match_pos = (self._match_pos - 1) % len(self._match_list)
            self._goto_match()

    # ---------------------------------------------------- copiar/exportar
    def _selected_segments(self) -> list[dict]:
        rows = sorted(i.row() for i in self._view.selectionModel().selectedIndexes())
        return [self._model.seg(r).as_dict() for r in rows]

    def copy_text(self):
        segs = self._selected_segments()
        if len(segs) <= 1 and not self._view.hasFocus():
            segs = self._model.segments()
        if not segs:
            return
        QGuiApplication.clipboard().setText(to_txt(segs))
        w = self.window()
        if hasattr(w, "show_toast"):
            w.show_toast(tr("copied", count=len(segs)), kind="success", duration_ms=2500)

    def _on_context_menu(self, pos):
        idx = self._view.indexAt(pos)
        menu = RoundedMenu(parent=self)
        if idx.isValid():
            seg = self._model.seg(idx.row())
            menu.addAction(tr("play_from_here"), lambda: self.timestamp_clicked.emit(seg.start_ms / 1000.0))
            menu.addAction(tr("copy_selection"), self.copy_text)
            menu.addSeparator()
        menu.addAction(tr("copy_all"), lambda: QGuiApplication.clipboard().setText(to_txt(self._model.segments())))
        if self._job_id and self._model.rowCount() and not self._live:
            menu.addMenu(self._export_menu)
        menu.exec(self._view.viewport().mapToGlobal(pos))

    def export(self, fmt: str):
        if not self._job_id or not self._model.rowCount():
            return
        label, ext = EXPORT_FORMATS[fmt]
        row = service.job(self._job_id)
        base = os.path.splitext(row["display_name"] if row is not None else "transcricao")[0]
        if fmt == "txt_ts":
            base += "_tempos"
        last_dir = app_config.get("last_export_dir") or (os.path.dirname(row["src_path"]) if row is not None else "")
        default = os.path.join(last_dir or os.path.expanduser("~"), base + ext)
        path, _ = QFileDialog.getSaveFileName(self, tr("export"), default, f"{label} (*{ext})")
        if not path:
            return
        if not path.lower().endswith(ext):
            path += ext
        try:
            service.export(self._job_id, fmt, path)
        except Exception as e:  # noqa: BLE001 — mostrado ao usuário
            QMessageBox.critical(self, tr("export_error"), f"{type(e).__name__}: {e}")
            return
        app_config.set_value("last_export_dir", os.path.dirname(path))
        w = self.window()
        if hasattr(w, "show_toast"):
            w.show_toast(tr("exported_to", name=os.path.basename(path)), kind="success")
