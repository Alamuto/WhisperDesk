"""Painel de configurações da transcrição (barra lateral).

O essencial fica à vista (modelo, idioma, tarefa, dispositivo); o resto vai para "Avançado".
O botão principal (start_button) pertence a este painel, mas é posicionado pela janela no rodapé
fixo da barra lateral, para estar sempre visível."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QSizePolicy, QSpinBox, QVBoxLayout, QWidget

from app import paths
from app.core.gpu_monitor import is_cuda_available
from app.gui.widgets import Collapsible, PrimaryButton, Reveal, SegmentedControl, SwitchRow
from app.pipeline.models import (
    MODEL_DOWNLOAD_MB, MODEL_LIST, MODEL_SPEED_ESTIMATE, NO_TRANSLATE, estimate_vram_mb, is_model_ready,
)
from app.utils import config as app_config
from app.utils.i18n import tr

# Whisper supported languages
WHISPER_LANGUAGES = {
    None: "Auto-detect",
    "en": "English", "zh": "Chinese", "de": "German", "es": "Spanish",
    "ru": "Russian", "ko": "Korean", "fr": "French", "ja": "Japanese",
    "pt": "Portuguese", "tr": "Turkish", "pl": "Polish", "ca": "Catalan",
    "nl": "Dutch", "ar": "Arabic", "sv": "Swedish", "it": "Italian",
    "id": "Indonesian", "hi": "Hindi", "fi": "Finnish", "vi": "Vietnamese",
    "he": "Hebrew", "uk": "Ukrainian", "el": "Greek", "ms": "Malay",
    "cs": "Czech", "ro": "Romanian", "da": "Danish", "hu": "Hungarian",
    "ta": "Tamil", "no": "Norwegian", "th": "Thai", "ur": "Urdu",
    "hr": "Croatian", "bg": "Bulgarian", "lt": "Lithuanian", "la": "Latin",
    "mi": "Maori", "ml": "Malayalam", "cy": "Welsh", "sk": "Slovak",
    "te": "Telugu", "fa": "Persian", "lv": "Latvian", "bn": "Bengali",
    "sr": "Serbian", "az": "Azerbaijani", "sl": "Slovenian", "kn": "Kannada",
    "et": "Estonian", "mk": "Macedonian", "br": "Breton", "eu": "Basque",
    "is": "Icelandic", "hy": "Armenian", "ne": "Nepali", "mn": "Mongolian",
    "bs": "Bosnian", "kk": "Kazakh", "sq": "Albanian", "sw": "Swahili",
    "gl": "Galician", "mr": "Marathi", "pa": "Panjabi", "si": "Sinhala",
    "km": "Khmer", "sn": "Shona", "yo": "Yoruba", "so": "Somali",
    "af": "Afrikaans", "oc": "Occitan", "ka": "Georgian", "be": "Belarusian",
    "tg": "Tajik", "sd": "Sindhi", "gu": "Gujarati", "am": "Amharic",
    "yi": "Yiddish", "lo": "Lao", "uz": "Uzbek", "fo": "Faroese",
    "ht": "Haitian", "ps": "Pashto", "tk": "Turkmen", "nn": "Nynorsk",
    "mt": "Maltese", "sa": "Sanskrit", "lb": "Luxembourgish", "my": "Myanmar",
    "bo": "Tibetan", "tl": "Tagalog", "mg": "Malagasy", "as": "Assamese",
    "tt": "Tatar", "haw": "Hawaiian", "ln": "Lingala", "ha": "Hausa",
    "ba": "Bashkir", "jw": "Javanese", "su": "Sundanese",
}

# Idiomas mais comuns primeiro, o resto em ordem alfabética.
_TOP_LANGS = ["pt", "en", "es", "fr", "de", "it", "ja", "zh", "ru", "ko"]


def language_name(code: str) -> str:
    """Nome do idioma na própria língua (ex.: "Português", "Español"); inglês como reserva."""
    from PySide6.QtCore import QLocale

    loc = QLocale(code)
    if loc.language() not in (QLocale.Language.C, QLocale.Language.AnyLanguage):
        native = loc.nativeLanguageName()
        if native:
            return native[:1].upper() + native[1:]
    return WHISPER_LANGUAGES.get(code, code)


def _repolish(w: QWidget):
    w.style().unpolish(w)
    w.style().polish(w)


def _field(label: QLabel, control: QWidget, hint: QWidget | None = None) -> QVBoxLayout:
    lay = QVBoxLayout()
    lay.setSpacing(7)
    lay.addWidget(label)
    lay.addWidget(control)
    if hint is not None:
        lay.addWidget(hint)
    return lay


def _label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("fieldLabel")
    return lbl


class ConfigPanel(QWidget):
    start_clicked = Signal(dict)  # Emits config dict when start is clicked

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("transparent")
        self._is_transcribing = False
        self._file_loaded = False
        self._cuda = is_cuda_available()
        self._setup_ui()
        self._load_defaults()

    # ------------------------------------------------------------------ UI
    def _setup_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(18)

        # Modelo
        self._model_label = _label(tr("model"))
        self._model_combo = QComboBox()
        self._model_combo.setMaxVisibleItems(12)
        self._populate_models()
        self._model_combo.currentIndexChanged.connect(self._on_model_changed)
        hint_row = QHBoxLayout()
        hint_row.setSpacing(8)
        self._model_chip = QLabel()
        self._model_chip.setObjectName("chipSuccess")
        self._model_chip.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        hint_row.addWidget(self._model_chip, 0, Qt.AlignmentFlag.AlignTop)
        self._model_hint = QLabel()
        self._model_hint.setObjectName("hint")
        self._model_hint.setWordWrap(True)
        hint_row.addWidget(self._model_hint, 1)
        hint_w = QWidget()
        hint_w.setObjectName("transparent")
        hint_w.setLayout(hint_row)
        hint_row.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(_field(self._model_label, self._model_combo, hint_w))

        # Idioma
        self._lang_label = _label(tr("language"))
        self._lang_combo = QComboBox()
        self._lang_combo.setMaxVisibleItems(16)
        self._populate_languages()
        lay.addLayout(_field(self._lang_label, self._lang_combo))

        # Tarefa
        self._task_label = _label(tr("task"))
        self._task_seg = SegmentedControl([("transcribe", tr("transcribe")),
                                           ("translate", tr("task_translate_short"))])
        lay.addLayout(_field(self._task_label, self._task_seg))

        # Dispositivo
        self._device_label = _label(tr("device"))
        self._device_seg = SegmentedControl([("cuda", "GPU"), ("cpu", "CPU")])
        self._device_seg.setToolTip(tr("device_tip"))
        if not self._cuda:
            self._device_seg.set_option_enabled("cuda", False, tr("gpu_cpu_mode"))
            self._device_seg.set_current("cpu", animate=False)
        lay.addLayout(_field(self._device_label, self._device_seg))

        # Falantes (diarização): o seletor de quantidade surge suavemente quando ativado
        spk = QVBoxLayout()
        spk.setSpacing(8)
        self._diar_row = SwitchRow(tr("diarize"), tr("diarize_desc"))
        self._diar_row.setToolTip(tr("diarize_tip"))
        spk.addWidget(self._diar_row)
        count_w = QWidget()
        count_w.setObjectName("transparent")
        count_lay = QVBoxLayout(count_w)
        count_lay.setContentsMargins(0, 2, 0, 0)
        count_lay.setSpacing(6)
        self._spk_label = _label(tr("num_speakers"))
        self._spk_seg = SegmentedControl([("0", tr("auto_short")), ("2", "2"), ("3", "3"), ("4", "4"), ("5", "5")])
        self._spk_seg.setToolTip(tr("num_speakers_tip"))
        count_lay.addWidget(self._spk_label)
        count_lay.addWidget(self._spk_seg)
        self._spk_reveal = Reveal(count_w)
        spk.addWidget(self._spk_reveal)
        self._diar_row.toggled.connect(lambda on: self._spk_reveal.set_shown(on))
        lay.addLayout(spk)

        # Revisão contextual dos trechos de baixa confiança
        self._refine_row = SwitchRow(tr("refine"), tr("refine_desc"))
        self._refine_row.setToolTip(tr("refine_tip"))
        lay.addWidget(self._refine_row)

        # Avançado
        self._advanced = Collapsible(tr("advanced"))
        cl = self._advanced.content_layout
        self._vad_row = SwitchRow(tr("vad_short"), tr("vad_desc"))
        self._vad_row.setToolTip(tr("vad_tip"))
        cl.addWidget(self._vad_row)

        self._compute_label = _label(tr("precision"))
        self._compute_combo = QComboBox()
        self._compute_combo.addItem(tr("prec_fp16"), "float16")
        self._compute_combo.addItem(tr("prec_int8_fp16"), "int8_float16")
        self._compute_combo.addItem(tr("prec_int8"), "int8")
        self._compute_combo.setToolTip(tr("precision_tip"))
        cl.addLayout(_field(self._compute_label, self._compute_combo))

        pair = QHBoxLayout()
        pair.setSpacing(12)
        self._beam_label = _label(tr("beam_size"))
        self._beam_spin = QSpinBox()
        self._beam_spin.setRange(1, 10)
        self._beam_spin.setToolTip(tr("beam_tip"))
        pair.addLayout(_field(self._beam_label, self._beam_spin), 1)
        self._temp_label = _label(tr("temperature"))
        self._temp_spin = QDoubleSpinBox()
        self._temp_spin.setRange(0.0, 1.0)
        self._temp_spin.setSingleStep(0.1)
        self._temp_spin.setDecimals(1)
        self._temp_spin.setToolTip(tr("temperature_tip"))
        pair.addLayout(_field(self._temp_label, self._temp_spin), 1)
        cl.addLayout(pair)
        lay.addWidget(self._advanced)
        lay.addStretch()
        # Combos não impõem largura mínima pelo conteúdo (a barra lateral tem largura fixa)
        for combo in (self._model_combo, self._lang_combo, self._compute_combo):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(8)

        # Botão principal (posicionado pela janela)
        self.start_button = PrimaryButton(tr("transcribe_action"), "sparkle")
        self.start_button.set_hint("Ctrl+Enter")
        self.start_button.setEnabled(False)
        self.start_button.setToolTip(tr("no_file_msg"))
        self.start_button.clicked.connect(self._on_start)

    def _populate_languages(self):
        current = self._lang_combo.currentData() if self._lang_combo.count() else None
        self._lang_combo.blockSignals(True)
        self._lang_combo.clear()
        self._lang_combo.addItem(tr("auto_detect"), None)
        top = [c for c in _TOP_LANGS if c in WHISPER_LANGUAGES]
        names = {c: language_name(c) for c in WHISPER_LANGUAGES if c}
        rest = sorted((c for c in names if c not in top), key=lambda c: names[c].casefold())
        for code in top:
            self._lang_combo.addItem(names[code], code)
            self._lang_combo.setItemData(self._lang_combo.count() - 1, WHISPER_LANGUAGES[code],
                                         Qt.ItemDataRole.ToolTipRole)
        self._lang_combo.insertSeparator(self._lang_combo.count())
        for code in rest:
            self._lang_combo.addItem(names[code], code)
            self._lang_combo.setItemData(self._lang_combo.count() - 1, WHISPER_LANGUAGES[code],
                                         Qt.ItemDataRole.ToolTipRole)
        i = self._lang_combo.findData(current)
        self._lang_combo.setCurrentIndex(max(0, i))
        self._lang_combo.blockSignals(False)

    def set_cuda_available(self, ok: bool):
        """Chamado quando o pacote GPU termina de instalar: libera a opção GPU e a seleciona."""
        self._cuda = ok
        self._device_seg.set_option_enabled("cuda", ok, "" if ok else tr("gpu_cpu_mode"))
        if ok:
            self._device_seg.set_current("cuda")

    def retranslate(self):
        self._model_label.setText(tr("model"))
        self._lang_label.setText(tr("language"))
        self._task_label.setText(tr("task"))
        self._task_seg.set_labels([tr("transcribe"), tr("task_translate_short")])
        self._device_label.setText(tr("device"))
        self._device_seg.setToolTip(tr("device_tip"))
        if not self._cuda:
            self._device_seg.set_option_enabled("cuda", False, tr("gpu_cpu_mode"))
        self._diar_row.set_texts(tr("diarize"), tr("diarize_desc"))
        self._diar_row.setToolTip(tr("diarize_tip"))
        self._refine_row.set_texts(tr("refine"), tr("refine_desc"))
        self._refine_row.setToolTip(tr("refine_tip"))
        self._spk_label.setText(tr("num_speakers"))
        self._spk_seg.set_labels([tr("auto_short"), "2", "3", "4", "5"])
        self._spk_seg.setToolTip(tr("num_speakers_tip"))
        self._advanced.set_title(tr("advanced"))
        self._vad_row.set_texts(tr("vad_short"), tr("vad_desc"))
        self._vad_row.setToolTip(tr("vad_tip"))
        self._compute_label.setText(tr("precision"))
        self._beam_label.setText(tr("beam_size"))
        self._temp_label.setText(tr("temperature"))
        for i, key in enumerate(("prec_fp16", "prec_int8_fp16", "prec_int8")):
            self._compute_combo.setItemText(i, tr(key))
        self._compute_combo.setToolTip(tr("precision_tip"))
        self._beam_spin.setToolTip(tr("beam_tip"))
        self._temp_spin.setToolTip(tr("temperature_tip"))
        self._populate_languages()
        self._populate_models()
        self._on_model_changed()
        self._update_button()

    # ------------------------------------------------------------ dados
    def _load_defaults(self):
        cfg = app_config.load()
        idx = self._model_combo.findData(cfg.get("model_size", "large-v3-turbo"))
        if idx >= 0:
            self._model_combo.setCurrentIndex(idx)
        idx = self._lang_combo.findData(cfg.get("language"))
        if idx >= 0:
            self._lang_combo.setCurrentIndex(idx)
        self._task_seg.set_current("translate" if cfg.get("task") == "translate" else "transcribe", animate=False)
        device = cfg.get("device", "cuda")
        if device == "cuda" and not self._cuda:
            device = "cpu"
        self._device_seg.set_current(device, animate=False)
        idx = self._compute_combo.findData(cfg.get("compute_type", "float16"))
        if idx >= 0:
            self._compute_combo.setCurrentIndex(idx)
        self._vad_row.setChecked(bool(cfg.get("vad_filter", True)))
        self._beam_spin.setValue(int(cfg.get("beam_size", 5)))
        self._temp_spin.setValue(float(cfg.get("temperature", 0.0)))
        self._apply_speakers(cfg, animate=False)
        self._refine_row.setChecked(bool(cfg.get("refine", True)))
        self._on_model_changed()

    def _apply_speakers(self, cfg: dict, animate: bool = True):
        on = bool(cfg.get("diarize", False))
        self._diar_row.setChecked(on)
        n = str(int(cfg.get("num_speakers") or 0))
        self._spk_seg.set_current(n if n in ("0", "2", "3", "4", "5") else "0", animate=animate)
        self._spk_reveal.set_shown(on, animate=animate)

    def get_config_dict(self) -> dict:
        return {
            "model_size": self._model_combo.currentData(),
            "language": self._lang_combo.currentData(),
            "task": self._task_seg.current(),
            "device": self._device_seg.current(),
            "compute_type": self._compute_combo.currentData(),
            "vad_filter": self._vad_row.isChecked(),
            "beam_size": self._beam_spin.value(),
            "temperature": self._temp_spin.value(),
            "diarize": self._diar_row.isChecked(),
            "num_speakers": int(self._spk_seg.current()) if self._diar_row.isChecked() else 0,
            "speaker_label": tr("speaker_n"),
            "refine": self._refine_row.isChecked(),
        }

    def _populate_models(self):
        current = self._model_combo.currentData()
        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        root = paths.models_dir()
        for model in MODEL_LIST:
            vram = estimate_vram_mb(model, "float16", 8) / 1024
            speed = MODEL_SPEED_ESTIMATE[model]
            ready = is_model_ready(root, model)
            label = model if ready else f"{model}   ·   {tr('download_size', size=self._fmt_mb(MODEL_DOWNLOAD_MB[model]))}"
            self._model_combo.addItem(label, model)
            tip = tr("model_tip", vram=f"{vram:.1f}", speed=f"{speed:.0f}")
            if not ready:
                tip += "\n" + tr("model_needs_download", size=f"{MODEL_DOWNLOAD_MB[model]}")
            self._model_combo.setItemData(self._model_combo.count() - 1, tip, Qt.ItemDataRole.ToolTipRole)
        if current is not None:
            i = self._model_combo.findData(current)
            if i >= 0:
                self._model_combo.setCurrentIndex(i)
        self._model_combo.blockSignals(False)

    @staticmethod
    def _fmt_mb(mb: int) -> str:
        return f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{mb} MB"

    def refresh_models(self):
        """Atualiza os marcadores de modelo baixado (após um download)."""
        self._populate_models()
        self._on_model_changed()

    def _on_model_changed(self, *_):
        model = self._model_combo.currentData()
        if model is None:
            return
        no_tr = model in NO_TRANSLATE
        self._task_seg.set_option_enabled("translate", not no_tr,
                                          tr("translate_unsupported", model=model) if no_tr else "")
        if no_tr and self._task_seg.current() == "translate":
            self._task_seg.set_current("transcribe")
        ready = is_model_ready(paths.models_dir(), model)
        self._model_chip.setText(tr("downloaded") if ready else
                                 tr("download_size", size=self._fmt_mb(MODEL_DOWNLOAD_MB[model])))
        self._model_chip.setObjectName("chipSuccess" if ready else "chipAccent")
        _repolish(self._model_chip)
        vram = estimate_vram_mb(model, "float16", 8) / 1024
        self._model_hint.setText(tr("model_hint", vram=f"{vram:.1f}", speed=f"{MODEL_SPEED_ESTIMATE[model]:.0f}"))

    # ------------------------------------------------------------ estado
    def _update_button(self):
        b = self.start_button
        if self._is_transcribing:
            b.setText(tr("transcribing"))
            b.setEnabled(False)
        else:
            q = getattr(self, "_queue_count", 0)
            total = q + (1 if self._file_loaded else 0)
            b.setText(tr("transcribe_n_files", count=total) if total > 1 else tr("transcribe_action"))
            b.setEnabled(total > 0)
            b.setToolTip("" if total else tr("no_file_msg"))

    def set_queue_count(self, n: int):
        self._queue_count = int(n)
        self._update_button()

    def set_file_loaded(self, loaded: bool):
        self._file_loaded = loaded
        self._update_button()

    def set_transcribing(self, active: bool):
        self._is_transcribing = active
        for w in (self._model_combo, self._lang_combo, self._task_seg, self._device_seg,
                  self._compute_combo, self._vad_row, self._beam_spin, self._temp_spin,
                  self._diar_row, self._spk_seg, self._refine_row):
            w.setEnabled(not active)
        self._update_button()

    def apply_config(self, cfg: dict):
        """Mostra a configuração de um job retomado."""
        for combo, key in ((self._model_combo, "model_size"), (self._lang_combo, "language"),
                           (self._compute_combo, "compute_type")):
            i = combo.findData(cfg.get(key))
            if i >= 0:
                combo.setCurrentIndex(i)
        dev = cfg.get("device") or "cuda"
        self._device_seg.set_current("cpu" if dev == "cpu" or not self._cuda else "cuda")
        self._task_seg.set_current("translate" if cfg.get("task") == "translate" else "transcribe")
        self._vad_row.setChecked(bool(cfg.get("vad_filter", True)))
        self._beam_spin.setValue(int(cfg.get("beam_size") or 5))
        self._temp_spin.setValue(float(cfg.get("temperature") or 0.0))
        self._apply_speakers(cfg)
        self._refine_row.setChecked(bool(cfg.get("refine", True)))
        self._on_model_changed()

    def _on_start(self):
        if not self.start_button.isEnabled():
            return
        cfg_dict = self.get_config_dict()
        app_config.save({**app_config.load(), **cfg_dict})
        self.start_clicked.emit(cfg_dict)
