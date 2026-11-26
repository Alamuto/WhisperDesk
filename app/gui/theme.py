"""Sistema de tema: tokens de cor definidos uma única vez por tema, folha de estilos gerada a partir
deles, paleta Qt coerente e um gerenciador que avisa os widgets pintados à mão quando o tema muda.

Nenhum widget deve usar cores literais: tudo vem de `tokens()` (pintura) ou da QSS gerada aqui."""
from __future__ import annotations

import ctypes
import logging
import os
from string import Template

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPainterPath, QPalette, QPen
from PySide6.QtWidgets import QApplication

log = logging.getLogger(__name__)

# ------------------------------------------------------------------ tokens
_DARK = {
    "bg": "#0D0E13",
    "surface": "#15161D",
    "surface2": "#1B1D26",
    "surface3": "#242733",
    "border": "#23252F",
    "border_strong": "#323543",
    "text": "#ECEDF3",
    "text2": "#A0A3B8",
    "text3": "#6C7088",
    "accent": "#7C6CFF",
    "accent_hover": "#8D7FFF",
    "accent_pressed": "#6A5AF2",
    "accent2": "#4E3FE0",
    "accent_text": "#FFFFFF",
    "accent_fg": "#A99FFF",       # accent legível sobre superfície escura
    "success": "#3CCB86",
    "warning": "#F0A93B",
    "danger": "#F2656F",
    "shadow": "#000000",
    # delegate da transcrição
    "low": "#F27A83",
    "mid": "#E8B259",
    "match": "#4A3D14",
    "word_bg": "#7C6CFF",
    "word_fg": "#FFFFFF",
}
_LIGHT = {
    "bg": "#F3F4F8",
    "surface": "#FFFFFF",
    "surface2": "#F5F6FA",
    "surface3": "#ECEEF4",
    "border": "#E2E5EE",
    "border_strong": "#CDD2DF",
    "text": "#161821",
    "text2": "#565C73",
    "text3": "#8C92A8",
    "accent": "#6253F4",
    "accent_hover": "#6F61FF",
    "accent_pressed": "#5243E4",
    "accent2": "#4535D6",
    "accent_text": "#FFFFFF",
    "accent_fg": "#5242E8",
    "success": "#1E9E5E",
    "warning": "#C27C14",
    "danger": "#D8414D",
    "shadow": "#1A1F36",
    "low": "#CF3442",
    "mid": "#B4770F",
    "match": "#FFF0B3",
    "word_bg": "#6253F4",
    "word_fg": "#FFFFFF",
}
THEMES = {"dark": _DARK, "light": _LIGHT}

FONT_UI = '"Segoe UI Variable Text", "Segoe UI", "Inter", sans-serif'
FONT_MONO = '"Cascadia Mono", "Consolas", monospace'


def qcolor(hex_color: str, alpha: float = 1.0) -> QColor:
    c = QColor(hex_color)
    c.setAlphaF(max(0.0, min(1.0, alpha)))
    return c


def mix(a: str | QColor, b: str | QColor, t: float) -> QColor:
    """Interpolação linear entre duas cores (t=0 → a, t=1 → b)."""
    ca, cb = QColor(a), QColor(b)
    t = max(0.0, min(1.0, t))
    return QColor.fromRgbF(ca.redF() + (cb.redF() - ca.redF()) * t, ca.greenF() + (cb.greenF() - ca.greenF()) * t,
                           ca.blueF() + (cb.blueF() - ca.blueF()) * t, ca.alphaF() + (cb.alphaF() - ca.alphaF()) * t)


def _rgba(hex_color: str, a: float) -> str:
    c = QColor(hex_color)
    return f"rgba({c.red()}, {c.green()}, {c.blue()}, {a:.3f})"


# ------------------------------------------------------------------ gerenciador
class _ThemeManager(QObject):
    changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.name = "dark"

    def tokens(self) -> dict:
        return THEMES[self.name]


_manager: _ThemeManager | None = None


def manager() -> _ThemeManager:
    global _manager
    if _manager is None:
        _manager = _ThemeManager()
    return _manager


def tokens() -> dict:
    return manager().tokens()


def current() -> str:
    return manager().name


def is_dark() -> bool:
    return manager().name == "dark"


def apply(name: str) -> None:
    """Aplica o tema na aplicação inteira (QSS + paleta) e notifica os widgets pintados."""
    m = manager()
    m.name = "light" if name == "light" else "dark"
    app = QApplication.instance()
    if app is not None:
        app.setPalette(_palette(m.tokens()))
        app.setStyleSheet(stylesheet(m.name))
    m.changed.emit(m.name)


def _palette(t: dict) -> QPalette:
    p = QPalette()
    roles = {
        QPalette.ColorRole.Window: t["bg"], QPalette.ColorRole.WindowText: t["text"],
        QPalette.ColorRole.Base: t["surface2"], QPalette.ColorRole.AlternateBase: t["surface"],
        QPalette.ColorRole.Text: t["text"], QPalette.ColorRole.Button: t["surface2"],
        QPalette.ColorRole.ButtonText: t["text"], QPalette.ColorRole.Highlight: t["accent"],
        QPalette.ColorRole.HighlightedText: t["accent_text"], QPalette.ColorRole.ToolTipBase: t["surface3"],
        QPalette.ColorRole.ToolTipText: t["text"], QPalette.ColorRole.PlaceholderText: t["text3"],
        QPalette.ColorRole.Link: t["accent_fg"], QPalette.ColorRole.Mid: t["border"],
        QPalette.ColorRole.Midlight: t["border"], QPalette.ColorRole.Dark: t["border_strong"],
        QPalette.ColorRole.Light: t["surface3"], QPalette.ColorRole.Shadow: t["shadow"],
    }
    for role, col in roles.items():
        p.setColor(role, QColor(col))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        p.setColor(QPalette.ColorGroup.Disabled, role, QColor(t["text3"]))
    return p


# ------------------------------------------------------------------ imagens para a QSS
def _asset_dir(name: str):
    from app import paths

    d = paths.local_dir() / "cache" / "ui" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _draw_asset(kind: str, color: str, size: int, path: str) -> None:
    img = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = size / 12.0
    pen = QPen(QColor(color), 1.6 * s, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    path_ = QPainterPath()
    if kind == "down":
        path_.moveTo(QPointF(3 * s, 4.5 * s)); path_.lineTo(QPointF(6 * s, 7.5 * s)); path_.lineTo(QPointF(9 * s, 4.5 * s))
    elif kind == "up":
        path_.moveTo(QPointF(3 * s, 7.5 * s)); path_.lineTo(QPointF(6 * s, 4.5 * s)); path_.lineTo(QPointF(9 * s, 7.5 * s))
    elif kind == "right":
        path_.moveTo(QPointF(4.5 * s, 3 * s)); path_.lineTo(QPointF(7.5 * s, 6 * s)); path_.lineTo(QPointF(4.5 * s, 9 * s))
    elif kind == "check":
        path_.moveTo(QPointF(2.8 * s, 6.2 * s)); path_.lineTo(QPointF(5 * s, 8.4 * s)); path_.lineTo(QPointF(9.2 * s, 3.8 * s))
    elif kind == "dot":
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(color))
        p.drawEllipse(QRectF(3.5 * s, 3.5 * s, 5 * s, 5 * s))
    p.drawPath(path_)
    p.end()
    img.save(path, "PNG")


def _assets(name: str) -> dict:
    t = THEMES[name]
    d = _asset_dir(name)
    out = {}
    specs = {"down": t["text2"], "down_dis": t["text3"], "up": t["text2"], "up_dis": t["text3"],
             "right": t["text2"], "check": t["accent_text"], "dot": t["accent_text"]}
    for key, color in specs.items():
        kind = key.split("_")[0]
        f = d / f"{key}.png"
        for scale, suffix in ((1, ""), (2, "@2x"), (3, "@3x")):
            target = d / f"{key}{suffix}.png"
            try:
                _draw_asset(kind, color, 12 * scale, str(target))
            except Exception:  # noqa: BLE001 — sem a imagem a QSS só perde a seta
                log.exception("Falha ao gerar %s", target)
        out[key] = f.as_posix()
    return out


_ASSET_CACHE: dict[str, dict] = {}


# ------------------------------------------------------------------ QSS
_QSS = Template(r"""
* { outline: none; }
QWidget {
    color: $text;
    font-family: $font_ui;
    font-size: 13px;
}
QMainWindow, QDialog, QMessageBox { background: $bg; }
QWidget#central { background: $bg; }

/* ---------- estrutura ---------- */
QWidget#header { background: $bg; border-bottom: 1px solid $border; }
QWidget#footer { background: $bg; border-top: 1px solid $border; }
QWidget#sidebar { background: $surface; border-right: 1px solid $border; }
QWidget#drawer { background: $surface; border-left: 1px solid $border; }
QWidget#logDrawer { background: $surface; border-top: 1px solid $border; }
QWidget#content, QWidget#page, QWidget#transparent { background: transparent; }
QFrame#card {
    background: $surface;
    border: 1px solid $border;
    border-radius: 14px;
}
QFrame#subtleCard {
    background: $surface2;
    border: 1px solid $border;
    border-radius: 12px;
}
QScrollArea, QScrollArea > QWidget > QWidget, QStackedWidget { background: transparent; border: none; }

/* ---------- tipografia ---------- */
QLabel { background: transparent; border: none; }
QLabel#wordmark { font-size: 15px; font-weight: 600; letter-spacing: 0.2px; }
QLabel#overline { color: $text3; font-size: 11px; font-weight: 600; letter-spacing: 0.8px; }
QLabel#fieldLabel { color: $text2; font-size: 12px; font-weight: 600; }
QLabel#title { font-size: 18px; font-weight: 600; }
QLabel#heroTitle { font-size: 20px; font-weight: 600; }
QLabel#cardTitle { font-size: 14px; font-weight: 600; }
QLabel#muted { color: $text2; }
QLabel#faint { color: $text3; font-size: 12px; }
QLabel#hint { color: $text3; font-size: 12px; }
QLabel#mono { font-family: $font_mono; font-size: 12px; color: $text2; }
QLabel#bigNumber { font-size: 22px; font-weight: 600; }
QLabel#chip {
    color: $text2; background: $surface2; border: 1px solid $border;
    border-radius: 9px; padding: 2px 9px; font-size: 11px; min-height: 14px; max-height: 14px;
}
QLabel#chipAccent {
    color: $accent_fg; background: $accent_soft; border: none;
    border-radius: 9px; padding: 2px 9px; font-size: 11px; min-height: 14px; max-height: 14px; font-weight: 600;
}
QLabel#chipSuccess {
    color: $success; background: $success_soft; border: none;
    border-radius: 9px; padding: 2px 9px; font-size: 11px; min-height: 14px; max-height: 14px; font-weight: 600;
}

/* ---------- botões ---------- */
QPushButton {
    background: $surface2;
    color: $text;
    border: 1px solid $border_strong;
    border-radius: 9px;
    padding: 7px 14px;
    font-weight: 500;
}
QPushButton:hover { background: $surface3; }
QPushButton:pressed { background: $border; }
QPushButton:disabled { color: $text3; background: $surface2; border-color: $border; }
QPushButton::menu-indicator { image: none; width: 0; }
QPushButton#ghost {
    background: transparent; border: 1px solid transparent; color: $text2; padding: 6px 10px;
}
QPushButton#ghost:hover { background: $surface3; color: $text; }
QPushButton#ghost:disabled { color: $text3; }
QPushButton#ghost:checked { background: $accent_soft; color: $accent_fg; }
QPushButton#link {
    background: transparent; border: none; color: $accent_fg; padding: 2px 4px; font-weight: 600;
}
QPushButton#link:hover { color: $accent_hover; text-decoration: underline; }
QPushButton#danger {
    background: transparent; color: $danger; border: 1px solid $danger_border;
}
QPushButton#danger:hover { background: $danger_soft; }
QPushButton#danger:disabled { color: $text3; border-color: $border; }
QPushButton#accentSoft {
    background: $accent_soft; color: $accent_fg; border: 1px solid transparent; font-weight: 600;
}
QPushButton#accentSoft:hover { background: $accent_soft_strong; }

/* ---------- campos ---------- */
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background: $surface2;
    color: $text;
    border: 1px solid $border_strong;
    border-radius: 9px;
    padding: 6px 10px;
    min-height: 20px;
    selection-background-color: $accent;
    selection-color: $accent_text;
}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover { border-color: $text3; }
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus { border-color: $accent; }
QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QLineEdit:disabled {
    color: $text3; border-color: $border;
}
QComboBox { padding-right: 28px; combobox-popup: 0; }
QComboBox::drop-down {
    subcontrol-origin: padding; subcontrol-position: center right;
    width: 26px; border: none; background: transparent;
}
QComboBox::down-arrow { image: url($img_down); width: 12px; height: 12px; }
QComboBox::down-arrow:disabled { image: url($img_down_dis); }
QComboBox QAbstractItemView {
    background: $surface;
    color: $text;
    border: 1px solid $border_strong;
    border-radius: 10px;
    padding: 4px;
    selection-background-color: $accent_soft;
    selection-color: $text;
    outline: none;
}
QComboBox QAbstractItemView::item { min-height: 30px; padding: 0 8px; border-radius: 6px; }
QComboBox QAbstractItemView::item:hover { background: $surface3; }
QComboBox QAbstractItemView::item:selected { background: $accent_soft; }

QSpinBox, QDoubleSpinBox { padding-right: 24px; }
QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border; subcontrol-position: top right; width: 22px;
    border: none; background: transparent; margin-top: 2px; margin-right: 2px;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border; subcontrol-position: bottom right; width: 22px;
    border: none; background: transparent; margin-bottom: 2px; margin-right: 2px;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover { background: $surface3; border-radius: 5px; }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image: url($img_up); width: 10px; height: 10px; }
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image: url($img_down); width: 10px; height: 10px; }
QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled { image: url($img_up_dis); }
QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled { image: url($img_down_dis); }

QLineEdit#search { border-radius: 9px; padding-left: 6px; }

QCheckBox, QRadioButton { spacing: 8px; background: transparent; }
QCheckBox::indicator, QRadioButton::indicator {
    width: 16px; height: 16px; border: 1px solid $border_strong; background: $surface2;
}
QCheckBox::indicator { border-radius: 5px; }
QRadioButton::indicator { border-radius: 9px; }
QCheckBox::indicator:hover, QRadioButton::indicator:hover { border-color: $accent; }
QCheckBox::indicator:checked { background: $accent; border-color: $accent; image: url($img_check); }
QRadioButton::indicator:checked { background: $accent; border-color: $accent; image: url($img_dot); }
QCheckBox:disabled, QRadioButton:disabled { color: $text3; }

/* ---------- listas ---------- */
QListView, QListWidget, QPlainTextEdit, QTextEdit {
    background: transparent;
    border: none;
    color: $text;
    selection-background-color: $accent_soft;
    selection-color: $text;
}
QPlainTextEdit#log {
    font-family: $font_mono; font-size: 12px; color: $text2;
    background: $surface2; border: 1px solid $border; border-radius: 10px; padding: 6px;
}

/* ---------- barras de rolagem ---------- */
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: $scroll; border-radius: 3px; min-height: 36px; margin: 0 2px; }
QScrollBar::handle:vertical:hover { background: $scroll_hover; margin: 0 1px; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px; }
QScrollBar::handle:horizontal { background: $scroll; border-radius: 3px; min-width: 36px; margin: 2px 0; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; border: none; background: none; }
QScrollBar::add-page, QScrollBar::sub-page { background: none; }

/* ---------- menus, dicas ---------- */
QMenu {
    background: $surface;
    border: 1px solid $border_strong;
    border-radius: 10px;
    padding: 5px;
}
QMenu::item { padding: 7px 26px 7px 12px; border-radius: 6px; color: $text; background: transparent; }
QMenu::item:selected { background: $surface3; }
QMenu::item:disabled { color: $text3; }
QMenu::item:checked { color: $accent_fg; font-weight: 600; }
QMenu::indicator { width: 0; }
QMenu::separator { height: 1px; background: $border; margin: 5px 8px; }
QMenu::right-arrow { image: url($img_right); width: 12px; height: 12px; margin-right: 6px; }
QToolTip {
    background: $surface3; color: $text; border: 1px solid $border_strong;
    border-radius: 6px; padding: 5px 8px; font-size: 12px;
}

/* ---------- diálogos ---------- */
QMessageBox QLabel { color: $text; font-size: 13px; }
QMessageBox QPushButton { min-width: 84px; }
QSplitter::handle { background: transparent; }
""")


def stylesheet(name: str) -> str:
    name = "light" if name == "light" else "dark"
    t = THEMES[name]
    if name not in _ASSET_CACHE:
        try:
            _ASSET_CACHE[name] = _assets(name)
        except Exception:  # noqa: BLE001
            log.exception("Falha ao gerar imagens do tema")
            _ASSET_CACHE[name] = {}
    imgs = _ASSET_CACHE[name]
    dark = name == "dark"
    values = dict(t)
    values.update(
        font_ui=FONT_UI, font_mono=FONT_MONO,
        accent_soft=_rgba(t["accent"], 0.16 if dark else 0.10),
        accent_soft_strong=_rgba(t["accent"], 0.26 if dark else 0.17),
        success_soft=_rgba(t["success"], 0.15 if dark else 0.11),
        danger_soft=_rgba(t["danger"], 0.12 if dark else 0.08),
        danger_border=_rgba(t["danger"], 0.45),
        scroll=_rgba(t["text"], 0.16 if dark else 0.18),
        scroll_hover=_rgba(t["text"], 0.30),
        **{f"img_{k}": imgs.get(k, "") for k in ("down", "down_dis", "up", "up_dis", "right", "check", "dot")},
    )
    return _QSS.substitute(values)


def delegate_colors(name: str) -> dict:
    """Cores dos delegates pintados à mão (lista da transcrição, histórico)."""
    t = THEMES["light" if name == "light" else "dark"]
    dark = name != "light"
    return {
        "bg": t["surface"], "alt": t["surface"], "hover": t["surface2"], "text": t["text"],
        "muted": t["text2"], "faint": t["text3"], "accent": t["accent"], "accent_fg": t["accent_fg"],
        "active_bg": qcolor(t["accent"], 0.13 if dark else 0.08).name(QColor.NameFormat.HexArgb),
        "selected_bg": qcolor(t["accent"], 0.22 if dark else 0.14).name(QColor.NameFormat.HexArgb),
        "word_bg": t["word_bg"], "word_fg": t["word_fg"], "low": t["low"], "mid": t["mid"],
        "match": t["match"], "border": t["border"],
    }


# ------------------------------------------------------------------ barra de título do Windows
def style_window_chrome(widget) -> None:
    """Barra de título nativa na cor do cabeçalho (Windows 11) e modo escuro do DWM (Windows 10+)."""
    if os.name != "nt" or QGuiApplication.platformName() != "windows":
        return
    try:
        hwnd = int(widget.winId())
        dwm = ctypes.windll.dwmapi
        t = tokens()
        dark = ctypes.c_int(1 if is_dark() else 0)
        dwm.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(dark), ctypes.sizeof(dark))

        def colorref(hex_color: str) -> ctypes.c_int:
            c = QColor(hex_color)
            return ctypes.c_int(c.red() | (c.green() << 8) | (c.blue() << 16))

        for attr, col in ((35, t["bg"]), (36, t["text"]), (34, t["border"])):
            v = colorref(col)
            dwm.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(v), ctypes.sizeof(v))
    except Exception:  # noqa: BLE001 — puramente cosmético
        log.debug("DWM indisponível", exc_info=True)
