"""Janela principal: monta os componentes e orquestra o ciclo de vida dos jobs.

Layout: cabeçalho fino · barra lateral (arquivo → opções → ação) · área de trabalho com dois
estados (tela inicial para soltar um arquivo / player + transcrição) · gavetas animadas de
histórico (direita) e log técnico (rodapé).

A transcrição roda num processo separado (JobController); todo o estado durável está no journal,
então fechar a janela, travar ou faltar energia nunca perde mais que o trecho em andamento."""
from __future__ import annotations

import json
import logging
import os
import time

from PySide6.QtCore import QEasingCurve, QRect, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication, QComboBox, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from app import __version__, paths
from app.core.audio_player import AudioPlayer
from app.core.gpu_monitor import get_gpu_name, is_cuda_available, shutdown as gpu_shutdown
from app.gui import icons, theme
from app.gui.config_panel import ConfigPanel
from app.gui.dropzone import WAVEFORM_BINS, DropZone, is_supported
from app.gui.history import HistoryPanel, RecentList
from app.gui.job_controller import JobController
from app.gui.player_controls import PlayerControls
from app.gui.progress import LogPanel, ProgressArea
from app.gui.result_view import ResultView
from app.gui.widgets import (
    DragOverlay, DropTarget, ElidedLabel, FadeStack, IconButton, IconTile, LogoMark, PulseDot, RoundedMenu,
    ToastHost, animate_property, card, section_label,
)
from app.pipeline import service
from app.pipeline.models import NO_TRANSLATE
from app.utils import config as app_config
from app.utils.i18n import UI_LANGUAGES, get_ui_language, set_ui_language, tr
from app.utils.waveform import WaveformWorker

log = logging.getLogger(__name__)

_STAGES = {
    "EXTRACTING": ("stage_extracting", False),
    "PLANNING": ("stage_planning", False),
    "LOADING_MODEL": ("stage_loading_model", True),
    "TRANSCRIBING": ("stage_transcribing", False),
}

_LEGACY_TOAST_COLORS = {"#4CAF7D": "success", "#C98A2B": "warning", "#6C63FF": "info"}

SIDEBAR_W = 340
HISTORY_W = 372
LOG_H = 230


def _styled(w: QWidget, name: str) -> QWidget:
    w.setObjectName(name)
    w.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    return w


class _Drawer(QWidget):
    """Gaveta que abre animando a largura (lateral) ou a altura (rodapé). O conteúdo mantém o
    tamanho final e é revelado, sem ser espremido durante a animação."""

    def __init__(self, content: QWidget, name: str, horizontal: bool, extent: int, parent=None):
        super().__init__(parent)
        _styled(self, name)
        self._content = content
        self._horizontal = horizontal
        self._extent = extent
        self._open = False
        content.setParent(self)
        if horizontal:
            self.setMaximumWidth(0)
            self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        else:
            self.setMaximumHeight(0)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.hide()

    @property
    def is_open(self) -> bool:
        return self._open

    def sizeHint(self):
        from PySide6.QtCore import QSize
        return QSize(self._extent, 10) if self._horizontal else QSize(10, self._extent)

    def set_open(self, on: bool, animate: bool = True):
        if on == self._open:
            return
        self._open = on
        prop = b"maximumWidth" if self._horizontal else b"maximumHeight"
        cur = self.maximumWidth() if self._horizontal else self.maximumHeight()
        cur = min(cur, self._extent)
        if on:
            self.show()
        end = self._extent if on else 0
        if not animate or not self.window().isVisible():
            self.setProperty(prop.decode(), end)
            if not on:
                self.hide()
            return
        animate_property(self, prop, cur, end, 300 if on else 240,
                         QEasingCurve.Type.OutCubic if on else QEasingCurve.Type.InOutCubic,
                         finished=None if on else self._after_close)

    def _after_close(self):
        if not self._open:
            self.hide()

    def resizeEvent(self, e):
        if self._horizontal:
            self._content.setGeometry(QRect(self.width() - self._extent, 0, self._extent, self.height()))
        else:
            self._content.setGeometry(QRect(0, 0, self.width(), self._extent))
        super().resizeEvent(e)


class _HeroDrop(DropTarget):
    """Grande área de soltar da tela inicial."""

    def __init__(self, parent=None):
        super().__init__(20, parent)
        self.setMinimumHeight(250)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(32, 32, 32, 32)
        lay.setSpacing(8)
        lay.addStretch()
        self.tile = IconTile("upload", 64, 28)
        lay.addWidget(self.tile, 0, Qt.AlignmentFlag.AlignHCenter)
        lay.addSpacing(10)
        self.title = QLabel()
        self.title.setObjectName("heroTitle")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.title)
        self.subtitle = QLabel()
        self.subtitle.setObjectName("muted")
        self.subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle.setWordWrap(True)
        lay.addWidget(self.subtitle)
        lay.addSpacing(6)
        self.formats = QLabel("MP3 · WAV · M4A · FLAC · OGG · MP4 · MKV · MOV · WEBM")
        self.formats.setObjectName("faint")
        self.formats.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.formats)
        lay.addStretch()
        for w in (self.tile, self.title, self.subtitle, self.formats):
            w.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._current_theme: str = "light" if app_config.get("theme", "dark") == "light" else "dark"
        theme.apply(self._current_theme)
        self._current_filepath = ""
        self._current_info = None
        self._job_id: str | None = None
        self._job_started_at = 0.0
        self._pending_resume: dict | None = None
        self._quitting = False
        self._doc_src = ""                 # arquivo exibido na área de trabalho (player)
        self._drop_peaks = ("", None)      # (caminho, picos) vindos da DropZone
        self._hist_wave = None
        self._old_waves: list = []
        self._theme_fade = None

        self._controller = JobController(self)
        self._controller.event.connect(self._on_job_event)
        self._controller.finished.connect(self._on_job_finished)

        self._setup_window()
        self._setup_ui()
        self._wire_signals()
        self._setup_shortcuts()
        self._retranslate_ui()
        self._toasts = ToastHost(self, bottom_margin=52)
        self._overlay = DragOverlay(self)
        theme.style_window_chrome(self)
        self._history_panel.refresh()
        QTimer.singleShot(300, self._startup_tasks)

    # ================================================================ layout
    def _setup_window(self):
        self.setWindowTitle("WhisperDesk")
        self.setMinimumSize(1040, 680)
        self.resize(int(app_config.get("window_width", 1320)), int(app_config.get("window_height", 840)))
        self.setAcceptDrops(True)

    def _setup_ui(self):
        central = _styled(QWidget(), "central")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_sidebar())

        column = QVBoxLayout()
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        self._content = FadeStack(slide=12)
        self._content.setObjectName("content")
        self._empty_page = self._build_empty_page()
        self._work_page = self._build_work_page()
        self._content.addWidget(self._empty_page)
        self._content.addWidget(self._work_page)
        self._content.set_current(self._empty_page, animate=False)
        column.addWidget(self._content, 1)
        self._log_panel = LogPanel()
        self._log_drawer = _Drawer(self._log_panel, "logDrawer", False, LOG_H)
        column.addWidget(self._log_drawer)
        body.addLayout(column, 1)

        self._history_panel = HistoryPanel()
        self._history_drawer = _Drawer(self._history_panel, "drawer", True, HISTORY_W)
        body.addWidget(self._history_drawer)
        root.addLayout(body, 1)
        root.addWidget(self._build_footer())

        # Campos de texto não devem engolir o arrasto de arquivos (a janela inteira é área de soltar).
        for w in central.findChildren(QLineEdit) + central.findChildren(QPlainTextEdit):
            w.setAcceptDrops(False)

    def _build_header(self) -> QWidget:
        header = _styled(QWidget(), "header")
        header.setFixedHeight(54)
        lay = QHBoxLayout(header)
        lay.setContentsMargins(18, 0, 12, 0)
        lay.setSpacing(10)
        lay.addWidget(LogoMark(26))
        word = QLabel("WhisperDesk")
        word.setObjectName("wordmark")
        lay.addWidget(word)
        lay.addStretch()
        self._history_btn = IconButton("history", tr("history"), 36, 19, checkable=True)
        self._history_btn.toggled.connect(self._set_history_open)
        lay.addWidget(self._history_btn)
        self._theme_btn = IconButton(self._theme_icon(), "", 36, 19)
        self._theme_btn.clicked.connect(self._toggle_theme)
        lay.addWidget(self._theme_btn)
        self._more_btn = IconButton("more", tr("more_options"), 36, 19)
        self._more_btn.clicked.connect(self._show_more_menu)
        lay.addWidget(self._more_btn)
        return header

    def _build_sidebar(self) -> QWidget:
        side = _styled(QWidget(), "sidebar")
        side.setFixedWidth(SIDEBAR_W)
        outer = QVBoxLayout(side)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = _styled(QWidget(), "transparent")
        il = QVBoxLayout(inner)
        il.setContentsMargins(18, 18, 18, 12)
        il.setSpacing(10)
        self._sec_file = section_label(tr("section_file"))
        il.addWidget(self._sec_file)
        self._dropzone = DropZone()
        il.addWidget(self._dropzone)
        il.addSpacing(12)
        self._sec_settings = section_label(tr("section_settings"))
        il.addWidget(self._sec_settings)
        self._config_panel = ConfigPanel()
        il.addWidget(self._config_panel)
        il.addStretch()
        scroll.setWidget(inner)
        outer.addWidget(scroll, 1)

        action = _styled(QWidget(), "transparent")
        al = QVBoxLayout(action)
        al.setContentsMargins(18, 10, 18, 18)
        self._action_stack = FadeStack(slide=8)
        self._action_stack.addWidget(self._config_panel.start_button)
        self._progress_area = ProgressArea()
        self._action_stack.addWidget(self._progress_area)
        self._action_stack.set_current(self._config_panel.start_button, animate=False)
        self._action_stack.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        al.addWidget(self._action_stack)
        outer.addWidget(action)
        return side

    def _build_empty_page(self) -> QWidget:
        page = _styled(QWidget(), "page")
        lay = QVBoxLayout(page)
        lay.setContentsMargins(40, 32, 40, 32)
        lay.addStretch(2)
        col = QVBoxLayout()
        col.setSpacing(22)
        self._hero = _HeroDrop()
        self._hero.clicked.connect(lambda: self._dropzone._on_select_file())
        col.addWidget(self._hero)
        self._recent = RecentList(4)
        col.addWidget(self._recent)
        holder = QWidget()
        holder.setMaximumWidth(620)
        holder.setLayout(col)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(holder, 10)
        row.addStretch()
        lay.addLayout(row)
        lay.addStretch(3)
        return page

    def _build_work_page(self) -> QWidget:
        page = _styled(QWidget(), "page")
        lay = QVBoxLayout(page)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(14)

        head = QHBoxLayout()
        head.setSpacing(12)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        self._doc_title = ElidedLabel("")
        self._doc_title.setObjectName("title")
        titles.addWidget(self._doc_title)
        self._doc_meta = QLabel("")
        self._doc_meta.setObjectName("hint")
        titles.addWidget(self._doc_meta)
        head.addLayout(titles, 1)
        lay.addLayout(head)

        self._audio_player = AudioPlayer(self)
        player_card, pl = card("card", (16, 12, 18, 12), 0)
        self._player_controls = PlayerControls(self._audio_player)
        pl.addWidget(self._player_controls)
        lay.addWidget(player_card)

        result_card, rl = card("card", (18, 16, 12, 10), 0)
        self._result_view = ResultView()
        rl.addWidget(self._result_view)
        lay.addWidget(result_card, 1)
        return page

    def _build_footer(self) -> QWidget:
        footer = _styled(QWidget(), "footer")
        footer.setFixedHeight(32)
        lay = QHBoxLayout(footer)
        lay.setContentsMargins(16, 0, 10, 0)
        lay.setSpacing(8)
        self._status_dot = PulseDot("accent", 10)
        self._status_dot.hide()
        lay.addWidget(self._status_dot)
        self._status_label = QLabel(tr("ready"))
        self._status_label.setObjectName("faint")
        lay.addWidget(self._status_label, 1)
        cuda = is_cuda_available()
        self._gpu_chip = QLabel(get_gpu_name() if cuda else tr("cpu_mode_short"))
        self._gpu_chip.setObjectName("chip")
        self._gpu_chip.setToolTip(tr("gpu_cpu_mode") if not cuda else "CUDA")
        lay.addWidget(self._gpu_chip, 0, Qt.AlignmentFlag.AlignVCenter)
        self._log_btn = QPushButton(tr("tech_log"))
        self._log_btn.setObjectName("ghost")
        self._log_btn.setCheckable(True)
        self._log_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._log_btn.toggled.connect(self._set_log_open)
        lay.addWidget(self._log_btn)
        theme.manager().changed.connect(self._refresh_icons)
        self._refresh_icons()
        return footer

    def _refresh_icons(self, *_):
        t = theme.tokens()
        self._log_btn.setIcon(icons.icon("terminal", t["text2"], 14, 1.8))
        self._theme_btn.set_icon(self._theme_icon())

    def _wire_signals(self):
        self._dropzone.file_selected.connect(self._on_file_selected)
        self._dropzone.peaks_changed.connect(self._on_drop_peaks)
        self._dropzone.peaks_failed.connect(lambda m: self._log_panel.append(f"waveform: {m}"))
        self._config_panel.start_clicked.connect(self._on_start_transcription)
        self._progress_area.cancel_requested.connect(self._on_cancel)
        self._history_panel.entry_selected.connect(self._on_history_entry_selected)
        self._history_panel.entry_deleted.connect(self._on_history_entry_deleted)
        self._history_panel.close_requested.connect(lambda: self._history_btn.setChecked(False))
        self._history_panel.rows_changed.connect(self._recent.set_rows)
        self._recent.entry_selected.connect(self._on_history_entry_selected)
        self._recent.entry_deleted.connect(self._on_recent_deleted)
        self._recent.see_all.connect(lambda: self._history_btn.setChecked(True))
        self._log_panel.close_requested.connect(lambda: self._log_btn.setChecked(False))
        self._result_view.timestamp_clicked.connect(self._on_timestamp_clicked)
        self._audio_player.position_changed.connect(self._result_view.highlight_segment)
        self._audio_player.state_changed.connect(lambda s: self._result_view.set_follow_playback(s == "playing"))
        self._audio_player.error_occurred.connect(lambda m: self._log_panel.append(f"{tr('player')}: {m}"))

    def _setup_shortcuts(self):
        def guarded(fn):
            # Espaço/setas não devem roubar a digitação em campos de texto.
            def _run():
                from PySide6.QtWidgets import QAbstractSpinBox, QTextEdit

                fw = QApplication.focusWidget()
                if isinstance(fw, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox, QComboBox)):
                    return
                if self._player_controls.isEnabled() and self._audio_player.filepath:
                    fn()
            return _run

        shortcuts = [
            (QKeySequence.StandardKey.Open, self._dropzone._on_select_file, False),
            ("Ctrl+Return", self._config_panel._on_start, False),
            ("Ctrl+Enter", self._config_panel._on_start, False),
            ("Ctrl+E", self._show_export_menu, False),
            ("Ctrl+H", self._history_btn.toggle, False),
            ("Ctrl+L", self._log_btn.toggle, False),
            ("Space", self._audio_player.toggle_play_pause, True),
            ("Left", lambda: self._audio_player.seek_relative(-5), True),
            ("Right", lambda: self._audio_player.seek_relative(5), True),
        ]
        for key, slot, guard in shortcuts:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(guarded(slot) if guard else slot)
        esc = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        esc.activated.connect(self._on_escape)

    def _show_export_menu(self):
        btn = self._result_view._export_btn
        if btn.isEnabled() and btn.isVisible():
            btn.showMenu()

    def _on_escape(self):
        if self._history_drawer.is_open:
            self._history_btn.setChecked(False)
        elif self._log_drawer.is_open:
            self._log_btn.setChecked(False)

    # ================================================================ gavetas / menus
    def _set_history_open(self, on: bool):
        if on:
            self._history_panel.refresh()
            self._history_panel.select_job(self._result_view.job_id)
        self._history_drawer.set_open(on)
        if on:
            QTimer.singleShot(120, self._history_panel.focus_filter)

    def _set_log_open(self, on: bool):
        self._log_drawer.set_open(on)

    def _show_more_menu(self):
        t = theme.tokens()
        menu = RoundedMenu(parent=self)
        lang_menu = RoundedMenu(tr("ui_language"), menu)
        lang_menu.setIcon(icons.icon("globe", t["text2"], 16))
        group = QActionGroup(lang_menu)
        cur = get_ui_language()
        for code, name in UI_LANGUAGES.items():
            act = QAction(name, lang_menu)
            act.setCheckable(True)
            act.setChecked(code == cur)
            act.triggered.connect(lambda _=False, c=code: self._on_language_changed(c))
            group.addAction(act)
            lang_menu.addAction(act)
        menu.addMenu(lang_menu)
        menu.addAction(icons.icon("folder", t["text2"], 16), tr("data_folder"), self._open_data_folder)
        menu.addAction(icons.icon("terminal", t["text2"], 16), tr("tech_log"), self._log_btn.toggle)
        menu.addSeparator()
        menu.addAction(icons.icon("info", t["text2"], 16), tr("about"), self._show_about)
        menu.exec(self._more_btn.mapToGlobal(self._more_btn.rect().bottomRight()) - self._menu_offset(menu))

    @staticmethod
    def _menu_offset(menu):
        from PySide6.QtCore import QPoint
        return QPoint(menu.sizeHint().width(), -6)

    def _show_about(self):
        QMessageBox.about(self, tr("about"), tr("about_text", version=__version__))

    # ================================================================ tema/idioma
    def _theme_icon(self) -> str:
        return "sun" if self._current_theme == "dark" else "moon"

    def _theme_text(self) -> str:
        return tr("light_mode") if self._current_theme == "dark" else tr("dark_mode")

    def _toggle_theme(self):
        central = self.centralWidget()
        snap = None
        if self.isVisible():
            pix: QPixmap = central.grab()
            snap = QLabel(central)
            snap.setPixmap(pix)
            snap.setGeometry(central.rect())
            snap.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            snap.show()
            snap.raise_()
        self._current_theme = "light" if self._current_theme == "dark" else "dark"
        theme.apply(self._current_theme)
        theme.style_window_chrome(self)
        self._theme_btn.setToolTip(self._theme_text())
        app_config.set_value("theme", self._current_theme)
        if snap is not None:
            eff = QGraphicsOpacityEffect(snap)
            eff.setOpacity(1.0)
            snap.setGraphicsEffect(eff)
            animate_property(eff, b"opacity", 1.0, 0.0, 320, QEasingCurve.Type.InOutQuad,
                             finished=snap.deleteLater)
            self._theme_fade = eff

    def _on_language_changed(self, code: str):
        if code and code != get_ui_language():
            set_ui_language(code)
            self._retranslate_ui()

    def _retranslate_ui(self):
        self._theme_btn.setToolTip(self._theme_text())
        self._history_btn.setToolTip(f"{tr('history')}  (Ctrl+H)")
        self._more_btn.setToolTip(tr("more_options"))
        self._log_btn.setText(tr("tech_log"))
        self._log_btn.setToolTip("Ctrl+L")
        self._sec_file.setText(tr("section_file").upper())
        self._sec_settings.setText(tr("section_settings").upper())
        self._hero.title.setText(tr("drop_title"))
        self._hero.subtitle.setText(tr("drop_subtitle"))
        if not is_cuda_available():
            self._gpu_chip.setText(tr("cpu_mode_short"))
            self._gpu_chip.setToolTip(tr("gpu_cpu_mode"))
        for w in (self._config_panel, self._progress_area, self._result_view, self._history_panel,
                  self._player_controls, self._log_panel, self._dropzone, self._recent):
            w.retranslate()
        if not self._controller.is_running():
            self._status_label.setText(tr("ready"))

    def _open_data_folder(self):
        p = str(paths.local_dir())
        if os.name == "nt":
            os.startfile(p)  # noqa: S606 — pasta local do próprio app

    # ================================================================ arrastar e soltar
    @staticmethod
    def _drag_path(event) -> str:
        md = event.mimeData()
        if not md.hasUrls():
            return ""
        for url in md.urls():
            if url.isLocalFile():
                path = url.toLocalFile()
                if os.path.isfile(path) and is_supported(path):
                    return path
        return ""

    def dragEnterEvent(self, event):
        if self._drag_path(event) and not self._controller.is_running():
            event.acceptProposedAction()
            self._overlay.title = tr("drop_overlay")
            self._overlay.subtitle = tr("drop_overlay_hint")
            self._overlay.appear()
            self._dropzone.set_drag_active(True)
            self._hero.set_drag_active(True)
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self._end_drag()
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        path = self._drag_path(event)
        self._end_drag()
        if path:
            event.acceptProposedAction()
            self._dropzone.load_file(path)

    def _end_drag(self):
        self._overlay.disappear()
        self._dropzone.set_drag_active(False)
        self._hero.set_drag_active(False)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if hasattr(self, "_toasts"):
            self._toasts.relayout()
        if hasattr(self, "_overlay") and self._overlay.isVisible():
            self._overlay.setGeometry(self.rect())

    # ================================================================ inicialização
    def _startup_tasks(self):
        try:
            n = service.import_legacy_history()
            if n:
                self._log_panel.append(tr("legacy_imported", count=n))
                self._history_panel.refresh()
        except Exception:
            log.exception("Falha ao importar histórico antigo")
        try:
            jobs = service.resumable_jobs()
        except Exception:
            log.exception("Falha ao ler jobs retomáveis")
            jobs = []
        if jobs:
            self._offer_resume(jobs[0], extra=len(jobs) - 1)
            return
        last = app_config.get("last_filepath", "")
        if last and os.path.isfile(last) and not self._current_filepath:
            self._dropzone.load_file(last, preferred_stream=app_config.get("last_audio_stream"))

    def _offer_resume(self, row, extra: int = 0):
        pct = service.job_progress(row["id"]) * 100
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(tr("resume_title"))
        text = tr("resume_msg", name=row["display_name"], pct=f"{pct:.0f}")
        if extra > 0:
            text += "\n\n" + tr("resume_more", count=extra)
        box.setText(text)
        b_resume = box.addButton(tr("resume_now"), QMessageBox.ButtonRole.AcceptRole)
        b_discard = box.addButton(tr("discard"), QMessageBox.ButtonRole.DestructiveRole)
        box.addButton(tr("later"), QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(b_resume)
        box.exec()
        clicked = box.clickedButton()
        if clicked is b_resume:
            cfg = json.loads(row["config_json"])
            self._config_panel.apply_config(cfg)
            self._pending_resume = {"job_id": row["id"], "path": os.path.normpath(row["src_path"])}
            self._dropzone.load_file(row["src_path"], preferred_stream=row["audio_stream"])
        elif clicked is b_discard:
            if QMessageBox.question(self, tr("discard"), tr("discard_confirm", name=row["display_name"])) \
                    == QMessageBox.StandardButton.Yes:
                service.discard_job(row["id"])

    # ================================================================ área de trabalho
    def _set_document(self, src: str, title: str, meta: str):
        """Mostra a área de trabalho para um arquivo (player + transcrição)."""
        self._doc_src = os.path.normpath(src) if src else ""
        self._doc_title.setText(title)
        self._doc_meta.setText(meta)
        self._content.set_current(self._work_page)

    def _show_empty(self):
        self._doc_src = ""
        self._audio_player.unload()
        self._player_controls.set_file_loaded(False)
        self._content.set_current(self._empty_page)

    def _on_drop_peaks(self, peaks):
        path = os.path.normpath(self._dropzone.current_filepath or "")
        self._drop_peaks = (path, peaks)
        if path and os.path.normcase(path) == os.path.normcase(self._doc_src):
            self._player_controls.set_peaks(peaks)

    def _load_history_peaks(self, src: str, stream, duration: float):
        """Forma de onda para um item do histórico cujo arquivo não é o da barra lateral."""
        self._retire_wave()
        self._player_controls.set_peaks(None)
        w = WaveformWorker(src, stream, duration, WAVEFORM_BINS, 1, self, render=False)
        w.peaks_ready.connect(lambda p, s=src: self._on_hist_peaks(s, p))
        w.waveform_error.connect(lambda m: self._log_panel.append(f"waveform: {m}"))
        self._hist_wave = w
        w.start()

    def _on_hist_peaks(self, src: str, peaks):
        if os.path.normcase(os.path.normpath(src)) == os.path.normcase(self._doc_src):
            self._player_controls.set_peaks(peaks)

    def _retire_wave(self):
        w = self._hist_wave
        self._hist_wave = None
        if w is None:
            return
        w.cancel()
        for sig in ("peaks_ready", "waveform_error"):
            try:
                getattr(w, sig).disconnect()
            except (RuntimeError, TypeError):
                pass
        if w.isRunning():
            self._old_waves.append(w)
            w.finished.connect(lambda x=w: self._old_waves.remove(x) if x in self._old_waves else None)

    # ================================================================ arquivo
    def _on_file_selected(self, filepath: str, info):
        self._current_filepath = filepath
        self._current_info = info
        self._config_panel.set_file_loaded(True)
        self._status_label.setText(tr("loaded", name=os.path.basename(filepath)))
        app_config.set_value("last_filepath", filepath)
        self._retire_wave()
        self._audio_player.load(filepath)
        self._player_controls.set_file_loaded(True)
        self._player_controls.setEnabled(True)
        self._set_document(filepath, os.path.basename(filepath), self._dropzone._meta_text(filepath, info))
        dp, peaks = self._drop_peaks
        self._player_controls.set_peaks(peaks if os.path.normcase(dp) == os.path.normcase(self._doc_src) else None)

        pending = self._pending_resume
        self._pending_resume = None
        if pending and os.path.normcase(pending["path"]) == os.path.normcase(filepath):
            self._start_job(pending["job_id"], resumed=True)
            return
        self._result_view.clear()

    # ================================================================ transcrição
    def _on_start_transcription(self, cfg: dict):
        if self._controller.is_running():
            return
        if not self._current_filepath or self._current_info is None:
            self.show_toast(tr("no_file_msg"), kind="warning")
            return
        if not os.path.isfile(self._current_filepath):
            QMessageBox.warning(self, tr("file_error"), tr("file_not_found", path=self._current_filepath))
            return
        if cfg.get("task") == "translate" and cfg.get("model_size") in NO_TRANSLATE:
            cfg["task"] = "transcribe"
        if cfg.get("device") == "cuda" and not is_cuda_available():
            cfg["device"] = "cpu"
        stream = self._dropzone.selected_stream
        app_config.set_value("last_audio_stream", stream)
        try:
            jid, state = service.create_or_resume_job(self._current_filepath, cfg, stream, self._current_info.duration)
        except OSError as e:
            QMessageBox.critical(self, tr("file_error"), str(e))
            return
        if state == "DONE":
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Information)
            box.setWindowTitle(tr("already_done_title"))
            box.setText(tr("already_done_msg"))
            b_show = box.addButton(tr("show_result"), QMessageBox.ButtonRole.AcceptRole)
            b_again = box.addButton(tr("transcribe_again"), QMessageBox.ButtonRole.DestructiveRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.setDefaultButton(b_show)
            box.exec()
            if box.clickedButton() is b_show:
                self._show_job(jid)
                return
            if box.clickedButton() is not b_again:
                return
            service.journal().reset_for_rerun(jid)
            state = "CREATED"
        self._start_job(jid, resumed=state != "CREATED")

    def _start_job(self, jid: str, resumed: bool):
        if self._controller.is_running():
            return
        self._job_id = jid
        self._job_started_at = time.monotonic()
        existing = service.load_segments(jid) if resumed else []
        if os.path.normcase(self._doc_src) != os.path.normcase(os.path.normpath(self._current_filepath)):
            self._on_file_selected_view_only()
        self._result_view.begin_job(jid, existing)
        self._log_panel.clear()
        row = service.job(jid)
        cfg = json.loads(row["config_json"])
        self._log_panel.append(tr("job_start", name=row["display_name"], model=cfg.get("model_size"),
                                  device=cfg.get("device")))
        if resumed and existing:
            self._log_panel.append(tr("job_resumed", pct=f"{service.job_progress(jid) * 100:.0f}"))
        self._set_busy(True)
        self._progress_area.start()
        self._progress_area.set_stage(tr("stage_starting"), indeterminate=True)
        self._status_label.setText(tr("transcribing"))
        try:
            self._controller.start(jid)
        except Exception as e:  # noqa: BLE001
            self._set_busy(False)
            self._progress_area.stop()
            QMessageBox.critical(self, tr("transcription_failed"), str(e))

    def _on_file_selected_view_only(self):
        """Volta a área de trabalho para o arquivo da barra lateral (ex.: após ver um item do histórico)."""
        fp, info = self._current_filepath, self._current_info
        self._retire_wave()
        self._audio_player.load(fp)
        self._player_controls.set_file_loaded(True)
        self._set_document(fp, os.path.basename(fp), self._dropzone._meta_text(fp, info))
        dp, peaks = self._drop_peaks
        self._player_controls.set_peaks(peaks if os.path.normcase(dp) == os.path.normcase(self._doc_src) else None)

    def _set_busy(self, busy: bool):
        self._config_panel.set_transcribing(busy)
        self._dropzone.set_busy(busy)
        self._history_panel.setEnabled(not busy)
        self._recent.setEnabled(not busy)
        self._hero.setEnabled(not busy)
        self._status_dot.setVisible(busy)
        self._status_dot.set_color_key("accent")
        if busy:
            self._action_stack.set_current(self._progress_area)
        else:
            QTimer.singleShot(1400, self._restore_action)

    def _restore_action(self):
        if not self._controller.is_running():
            self._action_stack.set_current(self._config_panel.start_button)

    def _on_cancel(self):
        if self._controller.is_running():
            self._controller.cancel()
            self._log_panel.append(tr("pausing_log"))
            self._status_label.setText(tr("pausing"))
            self._status_dot.set_color_key("warning")

    # ---------------------------------------------------------------- eventos
    def _on_job_event(self, kind: str, payload):
        pa = self._progress_area
        if kind == "log":
            self._log_panel.append(str(payload))
        elif kind == "state":
            key, indet = _STAGES.get(payload, (None, False))
            if key:
                pa.set_stage(tr(key), indeterminate=indet)
                self._status_label.setText(tr(key))
            if payload == "TRANSCRIBING" and self._job_id:
                pa.begin_transcription(service.job_progress(self._job_id))
        elif kind in ("extract_progress", "plan_progress"):
            pa.set_stage_fraction(float(payload or 0))
        elif kind == "download_progress":
            done, total = payload.get("done", 0), payload.get("total", 1) or 1
            pa.set_stage(tr("stage_downloading", done=done >> 20, total=total >> 20))
            pa.set_stage_fraction(done / total)
        elif kind == "engine":
            pa.set_engine(payload.get("label", ""))
            self._config_panel.refresh_models()
        elif kind == "degraded":
            pa.set_engine(payload.get("label", ""))
            msg = tr("degraded", label=payload.get("label", ""), reason=payload.get("reason", ""))
            self._log_panel.append(msg)
            self.show_toast(tr("degraded_short", label=payload.get("label", "")), kind="warning")
        elif kind == "progress":
            pa.update_progress(float(payload.get("fraction", 0)))
            pa.update_speed(float(payload.get("rtf", 0)))
        elif kind == "chunk_done":
            pa.update_progress(float(payload.get("fraction", 0)))
            pa.update_speed(float(payload.get("rtf", 0)))
            self._result_view.append_segments(payload.get("segments") or [])
            self._log_panel.append(tr("chunk_done", done=payload.get("done"), total=payload.get("total")))
        elif kind == "language":
            self._log_panel.append(tr("language_detected", code=payload.get("code"),
                                      prob=f"{float(payload.get('prob', 0)) * 100:.0f}"))
        elif kind == "warning":
            self._log_panel.append(f"⚠ {payload}")
            self.show_toast(str(payload)[:110], kind="warning", duration_ms=6000)
        elif kind == "error":
            self._log_panel.append(f"✖ {payload.get('code')}: {payload.get('message')}")

    def _on_job_finished(self, jid: str, outcome: str, payload):
        self._set_busy(False)
        self._progress_area.stop()
        self._result_view.end_live()
        elapsed = time.monotonic() - self._job_started_at
        if outcome == "done":
            self._progress_area.set_stage(tr("stage_done"))
            self._progress_area.update_progress(1.0)
            self._show_job(jid)
            p = payload or {}
            audio_s = float(p.get("audio_s") or 0)
            speed = f" · {audio_s / elapsed:.1f}x" if audio_s and elapsed > 0 else ""
            self._log_panel.append(tr("done_log", elapsed=ProgressArea.format_duration(elapsed),
                                      segments=p.get("segments", "?"), words=p.get("words", "?")) + speed)
            failed = int(p.get("failed_chunks") or 0)
            if failed:
                QMessageBox.warning(self, tr("partial_title"), tr("partial_msg", count=failed))
            self.show_toast(tr("done_toast", elapsed=ProgressArea.format_duration(elapsed)), kind="success")
            self._status_label.setText(tr("transcription_complete"))
            self._history_panel.refresh()
            self._config_panel.refresh_models()
            QApplication.alert(self)
        elif outcome == "paused":
            self._progress_area.set_stage(tr("stage_paused"))
            self._log_panel.append(tr("paused_log"))
            self._status_label.setText(tr("paused_status"))
            if not self._quitting:
                self.show_toast(tr("paused_toast"), kind="info")
        else:
            self._progress_area.set_stage(tr("stage_failed"))
            p = payload or {}
            code, msg = p.get("code", "?"), p.get("message", "")
            self._status_label.setText(tr("transcription_failed"))
            log_path = paths.logs_dir() / f"worker-{jid[:12]}.log"
            QMessageBox.critical(self, tr("transcription_failed"),
                                 tr("failed_msg", code=code, message=msg[:800], log=str(log_path)))

    @staticmethod
    def _job_meta(row) -> str:
        from app.gui.history import _fmt_dur, _fmt_when

        try:
            cfg = json.loads(row["config_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            cfg = {}
        lang = (cfg.get("language") or row["detected_language"] or "").upper()
        parts = [_fmt_when(row["finished_at"] or row["updated_at"] or 0), _fmt_dur(row["media_duration"]),
                 cfg.get("model_size", ""), lang]
        return "  ·  ".join(p for p in parts if p)

    def _show_job(self, jid: str):
        row = service.job(jid)
        if row is None:
            return
        self._job_id = jid
        src = os.path.normpath(row["src_path"] or "") if row["src_path"] else ""
        exists = bool(src) and os.path.isfile(src)
        same = exists and os.path.normcase(src) == os.path.normcase(self._doc_src)
        self._set_document(src, row["display_name"], self._job_meta(row))
        self._result_view.show_job(jid, service.load_segments(jid), row["display_name"])
        if not exists:
            self._retire_wave()
            self._audio_player.unload()
            self._player_controls.set_file_loaded(False)
            self._player_controls.set_message(tr("source_missing_player"))
            return
        self._player_controls.set_file_loaded(True)
        self._player_controls.setEnabled(True)
        if same:
            return
        self._audio_player.load(src)
        dp, peaks = self._drop_peaks
        if os.path.normcase(dp) == os.path.normcase(src) and peaks is not None:
            self._retire_wave()
            self._player_controls.set_peaks(peaks)
        else:
            self._load_history_peaks(src, row["audio_stream"], float(row["media_duration"] or 0))

    # ================================================================ histórico/player
    def _on_history_entry_selected(self, jid: str):
        if self._controller.is_running():
            self.show_toast(tr("busy_cannot_open"), kind="warning")
            return
        row = service.job(jid)
        if row is None:
            QMessageBox.warning(self, tr("not_found"), tr("not_found_msg"))
            self._history_panel.refresh()
            return
        self._show_job(jid)
        self._history_panel.select_job(jid)
        self._status_label.setText(tr("loaded", name=row["display_name"]))
        if not os.path.isfile(row["src_path"] or ""):
            self._log_panel.append(tr("source_missing_log", path=row["src_path"]))

    def _on_recent_deleted(self, jid: str):
        self._on_history_entry_deleted(jid)
        self._history_panel.refresh()

    def _on_history_entry_deleted(self, jid: str):
        if self._result_view.job_id == jid:
            self._result_view.clear()
            if self._current_filepath and os.path.isfile(self._current_filepath):
                self._on_file_selected_view_only()
            else:
                self._show_empty()

    def _on_timestamp_clicked(self, seconds: float):
        if not self._player_controls.isEnabled() or not self._audio_player.filepath:
            return
        self._audio_player.seek(seconds)
        self._audio_player.play()
        self._status_label.setText(tr("playing_from", time=ProgressArea.format_duration(seconds)))

    # ================================================================ toasts
    def show_toast(self, message: str, color: str | None = None, duration_ms: int = 4000, kind: str | None = None):
        if kind is None:
            kind = _LEGACY_TOAST_COLORS.get((color or "").upper(), "info")
        self._toasts.show(message, kind, duration_ms)

    # ================================================================ instância única
    def open_external_file(self, path: str):
        """Arquivo recebido de outra instância (ex.: "Abrir com")."""
        self.showNormal()
        self.raise_()
        self.activateWindow()
        if path and os.path.isfile(path):
            if self._controller.is_running():
                self.show_toast(tr("busy_cannot_open"), kind="warning")
            else:
                self._dropzone.load_file(path)

    # ================================================================ fechamento
    def closeEvent(self, event):
        if self._controller.is_running():
            r = QMessageBox.question(self, tr("quit_title"), tr("quit_running_msg"),
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
            if r != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._quitting = True
            self._status_label.setText(tr("pausing"))
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                self._controller.shutdown()
            finally:
                QApplication.restoreOverrideCursor()
        if not self.isMaximized():
            app_config.set_value("window_width", self.width())
            app_config.set_value("window_height", self.height())
        self._audio_player.stop()
        self._dropzone.shutdown()
        for w in [self._hist_wave, *self._old_waves]:
            if w is not None and w.isRunning():
                w.cancel()
                w.wait(3000)
        gpu_shutdown()
        service.close()
        event.accept()
