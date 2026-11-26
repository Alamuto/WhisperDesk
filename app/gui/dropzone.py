"""DropZone — cartão de arquivo da barra lateral: escolher/arrastar, análise (ffprobe), metadados,
escolha de trilha de áudio e cálculo da forma de onda. ffprobe e a forma de onda rodam fora da
thread da GUI. O arrastar-e-soltar é tratado pela janela inteira (MainWindow), que chama load_file."""
from __future__ import annotations

import os

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import QComboBox, QFileDialog, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from app.gui.widgets import (
    DropTarget, ElidedLabel, FadeStack, IconButton, IconTile, Spinner, card,
)
from app.pipeline import ffmpeg_tools
from app.pipeline.ffmpeg_tools import AudioError, ProbeInfo
from app.utils.i18n import tr
from app.utils.waveform import WaveformWorker

SUPPORTED_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".ogg", ".oga", ".opus", ".m4a", ".m4b", ".aac", ".wma", ".aiff", ".aif",
    ".amr", ".ac3", ".eac3", ".dts", ".mka", ".caf", ".alac", ".ape", ".wv",
    ".mp4", ".m4v", ".mkv", ".webm", ".avi", ".mov", ".wmv", ".flv", ".ts", ".mts", ".m2ts",
    ".mpg", ".mpeg", ".vob", ".3gp", ".3g2", ".ogv", ".mxf", ".asf", ".divx",
}

WAVEFORM_BINS = 1600


def is_supported(path: str) -> bool:
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in SUPPORTED_EXTENSIONS


def format_duration(seconds: float) -> str:
    total = int(round(seconds or 0))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def format_size(n: int) -> str:
    size = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit in ("B", "KB") else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


class _ProbeWorker(QThread):
    ok = Signal(str, object)    # (path, ProbeInfo)
    failed = Signal(str, str)   # (path, mensagem)

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self._path = path

    def run(self):
        try:
            self.ok.emit(self._path, ffmpeg_tools.probe(self._path))
        except AudioError as e:
            self.failed.emit(self._path, str(e))
        except Exception as e:  # noqa: BLE001
            self.failed.emit(self._path, f"{type(e).__name__}: {e}")


class DropZone(QWidget):
    file_selected = Signal(str, object)   # (caminho, ProbeInfo)
    stream_changed = Signal(int)          # índice absoluto da trilha escolhida
    peaks_changed = Signal(object)        # np.ndarray 0..1, ou None enquanto calcula
    peaks_failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_filepath = ""
        self._info: ProbeInfo | None = None
        self._probe: _ProbeWorker | None = None
        self._wave: WaveformWorker | None = None
        self._old_workers: list[QThread] = []
        self._busy = False
        self._preferred_stream = None
        self._setup_ui()

    # ------------------------------------------------------------------ UI
    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self._stack = FadeStack(slide=6)
        outer.addWidget(self._stack)

        # --- vazio: alvo tracejado
        self._empty = DropTarget(radius=14)
        self._empty.setFixedHeight(112)
        self._empty.clicked.connect(self._on_select_file)
        el = QHBoxLayout(self._empty)
        el.setContentsMargins(18, 0, 18, 0)
        el.setSpacing(14)
        self._empty_icon = IconTile("upload", 40, 20)
        el.addWidget(self._empty_icon)
        col = QVBoxLayout()
        col.setSpacing(2)
        col.addStretch()
        self._empty_title = QLabel(tr("choose_file"))
        self._empty_title.setObjectName("cardTitle")
        self._empty_hint = QLabel(tr("or_drag_here"))
        self._empty_hint.setObjectName("hint")
        col.addWidget(self._empty_title)
        col.addWidget(self._empty_hint)
        col.addStretch()
        el.addLayout(col, 1)
        self._stack.addWidget(self._empty)

        # --- analisando
        self._analyzing, al = card("subtleCard", (18, 0, 18, 0), 12, vertical=False)
        self._analyzing.setFixedHeight(112)
        self._spinner = Spinner(20)
        al.addWidget(self._spinner)
        acol = QVBoxLayout()
        acol.setSpacing(2)
        acol.addStretch()
        self._analyzing_title = QLabel(tr("analyzing_file"))
        self._analyzing_title.setObjectName("cardTitle")
        self._analyzing_name = ElidedLabel("")
        self._analyzing_name.setObjectName("hint")
        acol.addWidget(self._analyzing_title)
        acol.addWidget(self._analyzing_name)
        acol.addStretch()
        al.addLayout(acol, 1)
        self._stack.addWidget(self._analyzing)

        # --- carregado
        self._loaded, ll = card("subtleCard", (14, 14, 10, 14), 12)
        row = QHBoxLayout()
        row.setSpacing(12)
        self._file_icon = IconTile("file-audio", 40, 20)
        row.addWidget(self._file_icon, 0, Qt.AlignmentFlag.AlignTop)
        info = QVBoxLayout()
        info.setSpacing(3)
        self._name = ElidedLabel("")
        self._name.setObjectName("cardTitle")
        self._meta = QLabel("")
        self._meta.setObjectName("hint")
        info.addWidget(self._name)
        info.addWidget(self._meta)
        row.addLayout(info, 1)
        self._change_btn = IconButton("swap", tr("change_file"), 30, 16)
        self._change_btn.clicked.connect(self._on_select_file)
        row.addWidget(self._change_btn, 0, Qt.AlignmentFlag.AlignTop)
        ll.addLayout(row)

        self._stream_row = QWidget()
        self._stream_row.setObjectName("transparent")
        sl = QVBoxLayout(self._stream_row)
        sl.setContentsMargins(0, 0, 4, 0)
        sl.setSpacing(6)
        self._stream_label = QLabel(tr("audio_track"))
        self._stream_label.setObjectName("fieldLabel")
        sl.addWidget(self._stream_label)
        self._stream_combo = QComboBox()
        self._stream_combo.currentIndexChanged.connect(self._on_stream_changed)
        sl.addWidget(self._stream_combo)
        self._stream_row.setVisible(False)
        ll.addWidget(self._stream_row)
        self._stack.addWidget(self._loaded)
        self._stack.set_current(self._empty, animate=False)

    def retranslate(self):
        self._empty_title.setText(tr("choose_file"))
        self._empty_hint.setText(tr("or_drag_here"))
        self._analyzing_title.setText(tr("analyzing_file"))
        self._change_btn.setToolTip(tr("change_file"))
        self._stream_label.setText(tr("audio_track"))
        if self._info is not None:
            self._meta.setText(self._meta_text(self._current_filepath, self._info))

    def set_busy(self, busy: bool):
        """Durante uma transcrição não se troca de arquivo nem de trilha."""
        self._busy = busy
        self._change_btn.setEnabled(not busy)
        self._stream_combo.setEnabled(not busy)
        self._empty.setEnabled(not busy)

    def _toast(self, msg: str, kind: str = "error"):
        w = self.window()
        if hasattr(w, "show_toast"):
            w.show_toast(msg, kind=kind, duration_ms=6000)

    # --------------------------------------------------------------- carga
    def _on_select_file(self):
        if self._busy:
            return
        exts = " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTENSIONS))
        start_dir = os.path.dirname(self._current_filepath) if self._current_filepath else ""
        filepath, _ = QFileDialog.getOpenFileName(
            self, tr("select_file_dialog"), start_dir,
            f"{tr('media_files')} ({exts});;{tr('all_files')} (*)")
        if filepath:
            self.load_file(filepath)

    def load_file(self, filepath: str, preferred_stream: int | None = None):
        if self._busy:
            return
        filepath = os.path.normpath(filepath)
        if not os.path.isfile(filepath):
            self._toast(tr("file_not_found", path=filepath))
            return
        self._retire(self._probe)
        self._preferred_stream = preferred_stream
        self._analyzing_name.setText(os.path.basename(filepath))
        self._stack.set_current(self._analyzing)
        self._probe = _ProbeWorker(filepath, self)
        self._probe.ok.connect(self._on_probe_ok)
        self._probe.failed.connect(self._on_probe_failed)
        self._probe.start()

    def _restore_page(self):
        self._stack.set_current(self._loaded if self._info is not None else self._empty)

    def _on_probe_failed(self, path: str, msg: str):
        if self.sender() is not self._probe:
            return
        self._restore_page()
        self._toast(f"{os.path.basename(path)}: {msg}"[:220])

    @staticmethod
    def _meta_text(path: str, info: ProbeInfo) -> str:
        kind = tr("video") if info.has_video else tr("audio")
        fmt = (info.format_name.split(",")[0] or os.path.splitext(path)[1][1:]).upper()
        return f"{format_duration(info.duration)}  ·  {kind} {fmt}  ·  {format_size(info.size_bytes)}"

    def _on_probe_ok(self, path: str, info: ProbeInfo):
        if self.sender() is not self._probe:
            return
        self._current_filepath = path
        self._info = info
        self._name.setText(os.path.basename(path))
        self._meta.setText(self._meta_text(path, info))
        self._file_icon.set_icon("file-video" if info.has_video else "file-audio")

        self._stream_combo.blockSignals(True)
        self._stream_combo.clear()
        for s in info.audio_streams:
            self._stream_combo.addItem(s.describe(), s.index)
        pref = self._preferred_stream
        if pref is not None:
            i = self._stream_combo.findData(pref)
            if i >= 0:
                self._stream_combo.setCurrentIndex(i)
        self._stream_combo.blockSignals(False)
        self._stream_row.setVisible(len(info.audio_streams) > 1)
        self._stack.set_current(self._loaded)

        self._start_waveform()
        self.file_selected.emit(path, info)

    def _on_stream_changed(self, _i: int):
        if self._info is None:
            return
        self._start_waveform()
        self.stream_changed.emit(self.selected_stream)

    # -------------------------------------------------------- forma de onda
    def _start_waveform(self):
        self._retire(self._wave)
        self.peaks_changed.emit(None)
        self._wave = WaveformWorker(self._current_filepath, self.selected_stream, self._info.duration,
                                    WAVEFORM_BINS, 1, self, render=False)
        self._wave.peaks_ready.connect(self._on_peaks)
        self._wave.waveform_error.connect(self._on_waveform_error)
        self._wave.start()

    def _on_peaks(self, peaks):
        if self.sender() is not self._wave:
            return
        self.peaks_changed.emit(peaks)

    def _on_waveform_error(self, msg: str):
        if self.sender() is not self._wave:
            return
        self.peaks_failed.emit(msg)

    def _retire(self, worker):
        """Cancela um worker antigo sem bloquear a GUI; mantém referência até terminar."""
        if worker is None:
            return
        if hasattr(worker, "cancel"):
            worker.cancel()
        for sig in ("ok", "failed", "peaks_ready", "waveform_error", "waveform_ready"):
            try:
                getattr(worker, sig).disconnect()
            except (AttributeError, RuntimeError, TypeError):
                pass
        if worker.isRunning():
            self._old_workers.append(worker)
            worker.finished.connect(lambda w=worker: self._old_workers.remove(w) if w in self._old_workers else None)

    def shutdown(self):
        for w in [self._probe, self._wave, *self._old_workers]:
            if w is not None and w.isRunning():
                if hasattr(w, "cancel"):
                    w.cancel()
                w.wait(3000)

    def set_drag_active(self, on: bool):
        self._empty.set_drag_active(on)

    # ---------------------------------------------------------- acessores
    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def current_filepath(self) -> str:
        return self._current_filepath

    @property
    def current_info(self) -> ProbeInfo | None:
        return self._info

    @property
    def selected_stream(self) -> int:
        d = self._stream_combo.currentData()
        if d is None and self._info is not None:
            return self._info.default_stream.index
        return int(d) if d is not None else -1
