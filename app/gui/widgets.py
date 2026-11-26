"""Componentes visuais reutilizáveis, pintados à mão e animados.

Todas as cores vêm de app.gui.theme.tokens(); cada componente se repinta quando o tema muda.
As animações são curtas (120–300 ms), com curvas de saída suave, e não rodam quando o widget
não está visível."""
from __future__ import annotations

import math

from PySide6.QtCore import (
    Property, QEasingCurve, QEvent, QPoint, QPointF, QPropertyAnimation, QRect, QRectF, QSize, Qt, QTimer,
    QVariantAnimation, Signal,
)
from PySide6.QtGui import (
    QColor, QFont, QFontMetrics, QLinearGradient, QPainter, QPainterPath, QPen,
)
from PySide6.QtWidgets import (
    QAbstractButton, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QMenu, QSizePolicy, QStackedWidget,
    QToolTip, QVBoxLayout, QWidget,
)

from app.gui import icons, theme
from app.gui.theme import mix, qcolor

QWIDGETSIZE_MAX = 16777215


def _anim(owner, start: float, end: float, duration: int, on_value, easing=QEasingCurve.Type.OutCubic,
          finished=None) -> QVariantAnimation:
    a = QVariantAnimation(owner)
    a.setStartValue(float(start))
    a.setEndValue(float(end))
    a.setDuration(duration)
    a.setEasingCurve(easing)
    a.valueChanged.connect(on_value)
    if finished:
        a.finished.connect(finished)
    return a


def animate_property(target, prop: bytes, start, end, duration: int = 240,
                     easing=QEasingCurve.Type.OutCubic, finished=None) -> QPropertyAnimation:
    """Anima uma propriedade Qt mantendo a referência no próprio alvo (uma animação por propriedade)."""
    key = "_anim_" + prop.decode()
    old = getattr(target, key, None)
    if old is not None:
        old.stop()
    a = QPropertyAnimation(target, prop, target)
    a.setStartValue(start)
    a.setEndValue(end)
    a.setDuration(duration)
    a.setEasingCurve(easing)
    if finished:
        a.finished.connect(finished)
    setattr(target, key, a)
    a.start()
    return a


class ThemeAware:
    """Mixin: repinta o widget quando o tema muda."""

    def _watch_theme(self):
        theme.manager().changed.connect(self._on_theme_changed)

    def _on_theme_changed(self, _name: str):
        self.update()


# ======================================================================= botões
class IconButton(ThemeAware, QAbstractButton):
    """Botão só com ícone: fundo aparece suavemente no hover; estado 'checked' em tom de destaque."""

    def __init__(self, icon_name: str, tooltip: str = "", size: int = 34, icon_size: int = 18,
                 checkable: bool = False, parent=None):
        super().__init__(parent)
        self._icon = icon_name
        self._size = size
        self._icon_size = icon_size
        self._hover = 0.0
        self.setCheckable(checkable)
        self.setToolTip(tooltip)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(size, size)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self._kbd_focus = False
        self._ha = _anim(self, 0, 1, 140, self._set_hover)
        self._watch_theme()

    def set_icon(self, name: str):
        self._icon = name
        self.update()

    def focusInEvent(self, e):
        # Anel de foco só para navegação por teclado (equivalente a :focus-visible)
        self._kbd_focus = e.reason() in (Qt.FocusReason.TabFocusReason, Qt.FocusReason.BacktabFocusReason)
        super().focusInEvent(e)

    def focusOutEvent(self, e):
        self._kbd_focus = False
        super().focusOutEvent(e)

    def _set_hover(self, v):
        self._hover = float(v)
        self.update()

    def _hover_to(self, target: float):
        self._ha.stop()
        self._ha.setStartValue(self._hover)
        self._ha.setEndValue(target)
        self._ha.start()

    def enterEvent(self, e):
        self._hover_to(1.0)
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._hover_to(0.0)
        super().leaveEvent(e)

    def sizeHint(self):
        return QSize(self._size, self._size)

    def paintEvent(self, _e):
        t = theme.tokens()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        on = self.isChecked()
        enabled = self.isEnabled()
        if on:
            bg = qcolor(t["accent"], 0.16 + 0.06 * self._hover)
        else:
            bg = qcolor(t["surface3"], self._hover if enabled else 0)
        if self.isDown():
            bg = qcolor(t["border_strong"], 0.9)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(bg)
        p.drawRoundedRect(r, 9, 9)
        if self.hasFocus() and self._kbd_focus:
            p.setPen(QPen(qcolor(t["accent"], 0.7), 1.4))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), 9, 9)
        if not enabled:
            col = QColor(t["text3"])
        elif on:
            col = QColor(t["accent_fg"])
        else:
            col = mix(t["text2"], t["text"], self._hover)
        s = self._icon_size
        icons.draw(p, self._icon, QRectF((self.width() - s) / 2, (self.height() - s) / 2, s, s), col)


class PlayButton(ThemeAware, QAbstractButton):
    """Botão redondo de reprodução com troca animada entre play e pause."""

    def __init__(self, size: int = 42, parent=None):
        super().__init__(parent)
        self._playing = False
        self._morph = 0.0
        self._hover = 0.0
        self.setFixedSize(size, size)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._ma = _anim(self, 0, 1, 180, self._set_morph)
        self._ha = _anim(self, 0, 1, 140, self._set_hover)
        self._watch_theme()

    def _set_morph(self, v):
        self._morph = float(v)
        self.update()

    def _set_hover(self, v):
        self._hover = float(v)
        self.update()

    def set_playing(self, playing: bool):
        if playing == self._playing:
            return
        self._playing = playing
        self._ma.stop()
        self._ma.setStartValue(self._morph)
        self._ma.setEndValue(1.0 if playing else 0.0)
        self._ma.start()

    def enterEvent(self, e):
        self._ha.stop(); self._ha.setStartValue(self._hover); self._ha.setEndValue(1.0); self._ha.start()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._ha.stop(); self._ha.setStartValue(self._hover); self._ha.setEndValue(0.0); self._ha.start()
        super().leaveEvent(e)

    def paintEvent(self, _e):
        t = theme.tokens()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(2, 2, -2, -2)
        if self.isEnabled():
            if self._hover > 0:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(qcolor(t["accent"], 0.22 * self._hover))
                p.drawEllipse(QRectF(self.rect()))
            g = QLinearGradient(r.topLeft(), r.bottomRight())
            g.setColorAt(0, mix(t["accent"], t["accent_hover"], self._hover))
            g.setColorAt(1, t["accent2"])
            p.setBrush(g)
            col = QColor(t["accent_text"])
        else:
            p.setBrush(QColor(t["surface3"]))
            col = QColor(t["text3"])
        p.setPen(Qt.PenStyle.NoPen)
        if self.isDown():
            p.translate(r.center()); p.scale(0.95, 0.95); p.translate(-r.center())
        p.drawEllipse(r)
        s = r.width() * 0.46
        box = QRectF(r.center().x() - s / 2, r.center().y() - s / 2, s, s)
        m = self._morph
        if m < 1.0:
            p.setOpacity(1.0 - m)
            p.save(); p.translate(box.center()); p.scale(1 - 0.3 * m, 1 - 0.3 * m); p.translate(-box.center())
            icons.draw(p, "play", box.translated(s * 0.04, 0), col)
            p.restore()
        if m > 0.0:
            p.setOpacity(m)
            p.save(); p.translate(box.center()); p.scale(0.7 + 0.3 * m, 0.7 + 0.3 * m); p.translate(-box.center())
            icons.draw(p, "pause", box, col)
            p.restore()


class PrimaryButton(ThemeAware, QAbstractButton):
    """Botão principal: gradiente do acento, brilho que acende no hover e leve compressão ao clicar."""

    MARGIN = 5

    def __init__(self, text: str = "", icon_name: str | None = None, height: int = 46, parent=None):
        super().__init__(parent)
        self.setText(text)
        self._icon = icon_name
        self._hint = ""
        self._hover = 0.0
        self._press = 0.0
        self._h = height
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(height + 2 * self.MARGIN)
        self._ha = _anim(self, 0, 1, 180, self._set_hover)
        self._pa = _anim(self, 0, 1, 90, self._set_press)
        self.pressed.connect(lambda: self._press_to(1.0))
        self.released.connect(lambda: self._press_to(0.0))
        self._watch_theme()

    def set_hint(self, hint: str):
        self._hint = hint
        self.update()

    def set_icon(self, name: str | None):
        self._icon = name
        self.update()

    def _set_hover(self, v):
        self._hover = float(v)
        self.update()

    def _set_press(self, v):
        self._press = float(v)
        self.update()

    def _press_to(self, v):
        self._pa.stop(); self._pa.setStartValue(self._press); self._pa.setEndValue(v); self._pa.start()

    def enterEvent(self, e):
        self._ha.stop(); self._ha.setStartValue(self._hover); self._ha.setEndValue(1.0); self._ha.start()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._ha.stop(); self._ha.setStartValue(self._hover); self._ha.setEndValue(0.0); self._ha.start()
        super().leaveEvent(e)

    def sizeHint(self):
        fm = QFontMetrics(self._font())
        return QSize(fm.horizontalAdvance(self.text()) + 80, self._h + 2 * self.MARGIN)

    def _font(self) -> QFont:
        f = QFont(self.font())
        f.setPixelSize(14)
        f.setWeight(QFont.Weight.DemiBold)
        return f

    def paintEvent(self, _e):
        t = theme.tokens()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        m = self.MARGIN
        body = QRectF(self.rect()).adjusted(m, m, -m, -m)
        radius = 12
        enabled = self.isEnabled()
        if self._press > 0:
            s = 1 - 0.018 * self._press
            p.translate(body.center()); p.scale(s, s); p.translate(-body.center())
        if enabled:
            glow = 0.10 + 0.16 * self._hover
            p.setPen(Qt.PenStyle.NoPen)
            for i in range(m, 0, -1):
                p.setBrush(qcolor(t["accent"], glow * (1 - i / (m + 1)) * 0.6))
                p.drawRoundedRect(body.adjusted(-i, -i + 2, i, i + 2), radius + i, radius + i)
            g = QLinearGradient(body.topLeft(), body.bottomRight())
            top = mix(t["accent"], t["accent_hover"], self._hover)
            bottom = mix(t["accent2"], t["accent"], self._hover * 0.35)
            g.setColorAt(0, mix(top, t["accent_pressed"], self._press))
            g.setColorAt(1, mix(bottom, t["accent_pressed"], self._press))
            p.setBrush(g)
            p.drawRoundedRect(body, radius, radius)
            sheen = QLinearGradient(body.topLeft(), body.bottomLeft())
            sheen.setColorAt(0, qcolor("#FFFFFF", 0.16))
            sheen.setColorAt(0.5, qcolor("#FFFFFF", 0.0))
            p.setBrush(sheen)
            p.drawRoundedRect(body, radius, radius)
            fg = QColor(t["accent_text"])
        else:
            p.setPen(QPen(QColor(t["border"]), 1))
            p.setBrush(QColor(t["surface3"]))
            p.drawRoundedRect(body, radius, radius)
            fg = QColor(t["text3"])
        f = self._font()
        p.setFont(f)
        fm = QFontMetrics(f)
        text = self.text()
        icon_w = 18 if self._icon else 0
        gap = 9 if self._icon else 0
        total = icon_w + gap + fm.horizontalAdvance(text)
        x = body.center().x() - total / 2
        if self._icon:
            icons.draw(p, self._icon, QRectF(x, body.center().y() - 9, 18, 18), fg)
        p.setPen(fg)
        p.drawText(QRectF(x + icon_w + gap, body.top(), total, body.height()),
                   int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), text)
        if self._hint and enabled and body.width() > total + 140:
            hf = QFont(self.font())
            hf.setPixelSize(11)
            p.setFont(hf)
            p.setPen(qcolor(t["accent_text"], 0.62))
            p.drawText(body.adjusted(0, 0, -14, 0), int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight),
                       self._hint)


# ======================================================================= seletores
class SegmentedControl(ThemeAware, QWidget):
    """Controle segmentado com a pílula de seleção deslizando entre as opções."""

    changed = Signal(str)

    def __init__(self, options: list[tuple[str, str]], parent=None):
        super().__init__(parent)
        self._keys = [k for k, _ in options]
        self._labels = [lbl for _, lbl in options]
        self._enabled = {k: True for k in self._keys}
        self._tips: dict[str, str] = {}
        self._index = 0
        self._pos = 0.0
        self._hover_i = -1
        self.setMouseTracking(True)
        self.setFixedHeight(36)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self._pa = _anim(self, 0, 0, 240, self._set_pos)
        self._watch_theme()

    # API
    def current(self) -> str:
        return self._keys[self._index]

    def set_current(self, key: str, animate: bool = True):
        if key not in self._keys:
            return
        i = self._keys.index(key)
        changed = i != self._index
        self._index = i
        if animate and self.isVisible():
            self._pa.stop(); self._pa.setStartValue(self._pos); self._pa.setEndValue(float(i)); self._pa.start()
        else:
            self._pos = float(i)
            self.update()
        if changed:
            self.changed.emit(key)

    def set_labels(self, labels: list[str]):
        self._labels = list(labels)
        self.update()

    def set_option_enabled(self, key: str, enabled: bool, tip: str = ""):
        self._enabled[key] = enabled
        self._tips[key] = tip
        self.update()

    def is_option_enabled(self, key: str) -> bool:
        return self._enabled.get(key, False) and self.isEnabled()

    def _set_pos(self, v):
        self._pos = float(v)
        self.update()

    def _seg_w(self) -> float:
        return (self.width() - 6) / max(1, len(self._keys))

    def _index_at(self, x: float) -> int:
        return max(0, min(len(self._keys) - 1, int((x - 3) // self._seg_w())))

    def mouseMoveEvent(self, e):
        i = self._index_at(e.position().x())
        if i != self._hover_i:
            self._hover_i = i
            self.update()

    def leaveEvent(self, e):
        self._hover_i = -1
        self.update()
        super().leaveEvent(e)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self.isEnabled():
            i = self._index_at(e.position().x())
            if self._enabled.get(self._keys[i], True):
                self.set_current(self._keys[i])

    def keyPressEvent(self, e):
        step = {Qt.Key.Key_Left: -1, Qt.Key.Key_Right: 1}.get(e.key())
        if step:
            i = self._index + step
            while 0 <= i < len(self._keys):
                if self._enabled.get(self._keys[i], True):
                    self.set_current(self._keys[i])
                    break
                i += step
        else:
            super().keyPressEvent(e)

    def event(self, e):
        if e.type() == QEvent.Type.ToolTip:
            key = self._keys[self._index_at(e.position().x() if hasattr(e, "position") else e.pos().x())]
            tip = self._tips.get(key) or self.toolTip()
            if tip:
                QToolTip.showText(e.globalPos(), tip, self)
            else:
                QToolTip.hideText()
            return True
        return super().event(e)

    def sizeHint(self):
        fm = QFontMetrics(self.font())
        w = max((fm.horizontalAdvance(lbl) for lbl in self._labels), default=40) + 28
        return QSize(int(w * len(self._labels)) + 6, 36)

    def paintEvent(self, _e):
        t = theme.tokens()
        dark = theme.is_dark()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QPen(QColor(t["border"]), 1))
        p.setBrush(QColor(t["surface2"]))
        p.drawRoundedRect(r, 10, 10)
        sw = self._seg_w()
        if self._hover_i >= 0 and self._hover_i != self._index and self.isEnabled() \
                and self._enabled.get(self._keys[self._hover_i], True):
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(qcolor(t["surface3"], 0.7))
            p.drawRoundedRect(QRectF(3 + self._hover_i * sw, 3, sw, self.height() - 6), 8, 8)
        pill = QRectF(3 + self._pos * sw, 3, sw, self.height() - 6)
        p.setPen(QPen(QColor(t["border_strong"] if dark else t["border"]), 1))
        p.setBrush(QColor(t["surface3"] if dark else t["surface"]))
        if not dark:
            p.save()
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(qcolor(t["shadow"], 0.08))
            p.drawRoundedRect(pill.translated(0, 1), 8, 8)
            p.restore()
        p.drawRoundedRect(pill, 8, 8)
        f = QFont(self.font())
        f.setPixelSize(13)
        for i, label in enumerate(self._labels):
            key = self._keys[i]
            sel = abs(self._pos - i) < 0.5
            f.setWeight(QFont.Weight.DemiBold if sel else QFont.Weight.Medium)
            p.setFont(f)
            if not self.isEnabled() or not self._enabled.get(key, True):
                col = QColor(t["text3"])
            else:
                col = mix(t["text2"], t["text"], max(0.0, 1 - abs(self._pos - i)))
            p.setPen(col)
            rect = QRectF(3 + i * sw, 0, sw, self.height())
            p.drawText(rect, int(Qt.AlignmentFlag.AlignCenter),
                       QFontMetrics(f).elidedText(label, Qt.TextElideMode.ElideRight, int(sw - 10)))


class ToggleSwitch(ThemeAware, QAbstractButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(40, 24)
        self._t = 0.0
        self._a = _anim(self, 0, 1, 170, self._set_t)
        self.toggled.connect(self._on_toggled)
        self._watch_theme()

    def _set_t(self, v):
        self._t = float(v)
        self.update()

    def _on_toggled(self, on: bool):
        if self.isVisible():
            self._a.stop(); self._a.setStartValue(self._t); self._a.setEndValue(1.0 if on else 0.0); self._a.start()
        else:
            self._t = 1.0 if on else 0.0
            self.update()

    def setChecked(self, on: bool):  # noqa: N802 — API Qt
        super().setChecked(on)
        if not self.isVisible():
            self._t = 1.0 if on else 0.0

    def paintEvent(self, _e):
        t = theme.tokens()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(1, 2, self.width() - 2, self.height() - 4)
        on_col = QColor(t["accent"]) if self.isEnabled() else QColor(t["border_strong"])
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(mix(t["border_strong"], on_col, self._t))
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        d = r.height() - 4
        x = r.left() + 2 + (r.width() - d - 4) * self._t
        p.setBrush(qcolor(t["shadow"], 0.18))
        p.drawEllipse(QRectF(x, r.top() + 2.6, d, d))
        p.setBrush(QColor("#FFFFFF") if self.isEnabled() else QColor(t["surface3"]))
        p.drawEllipse(QRectF(x, r.top() + 2, d, d))


class SwitchRow(QWidget):
    """Linha com título, descrição curta e interruptor; clicar em qualquer lugar alterna."""

    toggled = Signal(bool)

    def __init__(self, title: str, desc: str = "", parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(12)
        col = QVBoxLayout()
        col.setSpacing(1)
        self._title = QLabel(title)
        self._title.setObjectName("fieldLabel")
        self._desc = QLabel(desc)
        self._desc.setObjectName("hint")
        self._desc.setWordWrap(True)
        col.addWidget(self._title)
        col.addWidget(self._desc)
        lay.addLayout(col, 1)
        self.switch = ToggleSwitch()
        self.switch.toggled.connect(self.toggled)
        lay.addWidget(self.switch, 0, Qt.AlignmentFlag.AlignVCenter)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_texts(self, title: str, desc: str):
        self._title.setText(title)
        self._desc.setText(desc)
        self._desc.setVisible(bool(desc))

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self.isEnabled():
            self.switch.toggle()

    def isChecked(self) -> bool:  # noqa: N802
        return self.switch.isChecked()

    def setChecked(self, on: bool):  # noqa: N802
        self.switch.setChecked(on)


# ======================================================================= contêineres
class _CollapseHeader(ThemeAware, QAbstractButton):
    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self.setText(text)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(32)
        self._angle = 0.0
        self._hover = 0.0
        self._a = _anim(self, 0, 90, 220, self._set_angle)
        self._ha = _anim(self, 0, 1, 140, self._set_hover)
        self.toggled.connect(self._on_toggled)
        self._watch_theme()

    def _set_angle(self, v):
        self._angle = float(v)
        self.update()

    def _set_hover(self, v):
        self._hover = float(v)
        self.update()

    def _on_toggled(self, on):
        self._a.stop(); self._a.setStartValue(self._angle); self._a.setEndValue(90.0 if on else 0.0); self._a.start()

    def enterEvent(self, e):
        self._ha.stop(); self._ha.setStartValue(self._hover); self._ha.setEndValue(1.0); self._ha.start()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._ha.stop(); self._ha.setStartValue(self._hover); self._ha.setEndValue(0.0); self._ha.start()
        super().leaveEvent(e)

    def paintEvent(self, _e):
        t = theme.tokens()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        col = mix(t["text2"], t["text"], self._hover)
        p.save()
        c = QPointF(9, self.height() / 2)
        p.translate(c); p.rotate(self._angle); p.translate(-c)
        icons.draw(p, "chevron-right", QRectF(1, self.height() / 2 - 8, 16, 16), col, 2.0)
        p.restore()
        f = QFont(self.font())
        f.setPixelSize(12)
        f.setWeight(QFont.Weight.DemiBold)
        p.setFont(f)
        p.setPen(col)
        p.drawText(QRectF(24, 0, self.width() - 24, self.height()),
                   int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), self.text())


class Collapsible(QWidget):
    """Seção recolhível com animação de altura e seta que gira."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.header = _CollapseHeader(title)
        lay.addWidget(self.header)
        self.content = QWidget()
        self.content.setObjectName("transparent")
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 8, 0, 4)
        self.content_layout.setSpacing(14)
        self.content.setMaximumHeight(0)
        self.content.setVisible(False)
        lay.addWidget(self.content)
        self.header.toggled.connect(self._toggle)

    def set_title(self, title: str):
        self.header.setText(title)

    def _toggle(self, on: bool):
        target = self.content.sizeHint().height()
        if on:
            self.content.setVisible(True)
            animate_property(self.content, b"maximumHeight", 0, target, 240,
                             finished=lambda: self.content.setMaximumHeight(QWIDGETSIZE_MAX))
        else:
            start = self.content.height()
            animate_property(self.content, b"maximumHeight", start, 0, 200, QEasingCurve.Type.InOutCubic,
                             finished=lambda: self.content.setVisible(self.header.isChecked()))


class Reveal(QWidget):
    """Envolve um widget e o mostra/oculta com altura + opacidade (entrada e saída suaves).
    O efeito de opacidade só existe durante a animação."""

    def __init__(self, child: QWidget, shown: bool = False, parent=None):
        super().__init__(parent)
        self.setObjectName("transparent")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.child = child
        lay.addWidget(child)
        self._shown = shown
        self.setMaximumHeight(QWIDGETSIZE_MAX if shown else 0)
        self.setVisible(shown)

    def is_shown(self) -> bool:
        return self._shown

    def _effect(self, start: float) -> QGraphicsOpacityEffect:
        fx = self.child.graphicsEffect()
        if not isinstance(fx, QGraphicsOpacityEffect):
            fx = QGraphicsOpacityEffect(self.child)
            fx.setOpacity(start)
            self.child.setGraphicsEffect(fx)
        return fx

    def set_shown(self, on: bool, animate: bool = True, finished=None):
        if on == self._shown:
            if finished:
                finished()
            return
        self._shown = on
        if not animate or not self.window().isVisible():
            self._settle(finished)
            return
        target = max(1, self.child.sizeHint().height())
        if on:
            self.setVisible(True)
            fx = self._effect(0.0)
            animate_property(self, b"maximumHeight", self.height() if self.maximumHeight() else 0, target, 260,
                             finished=lambda: self._settle(finished))
            animate_property(fx, b"opacity", fx.opacity(), 1.0, 320)
        else:
            fx = self._effect(1.0)
            animate_property(self, b"maximumHeight", self.height(), 0, 220, QEasingCurve.Type.InOutCubic,
                             finished=lambda: self._settle(finished))
            animate_property(fx, b"opacity", fx.opacity(), 0.0, 160)

    def _settle(self, finished=None):
        try:
            self.child.setGraphicsEffect(None)
        except RuntimeError:
            return
        self.setMaximumHeight(QWIDGETSIZE_MAX if self._shown else 0)
        self.setVisible(self._shown)
        if finished:
            finished()


class FadeStack(QStackedWidget):
    """QStackedWidget com transição: a nova página surge com fade e um leve deslize vertical."""

    def __init__(self, parent=None, slide: int = 10):
        super().__init__(parent)
        self._slide = slide
        self._fx: list = []

    # O tamanho acompanha a página atual (QStackedWidget usa o maior entre todas)
    def sizeHint(self):
        w = self.currentWidget()
        if w is None:
            return super().sizeHint()
        return w.sizeHint().expandedTo(w.minimumSize()).boundedTo(w.maximumSize())

    def minimumSizeHint(self):
        w = self.currentWidget()
        if w is None:
            return super().minimumSizeHint()
        return w.minimumSizeHint().expandedTo(w.minimumSize()).boundedTo(w.maximumSize())

    def set_current(self, w: QWidget, animate: bool = True):
        if isinstance(w, int):
            w = self.widget(w)
        if w is self.currentWidget():
            return
        self.setCurrentWidget(w)
        self.updateGeometry()
        if not animate or not self.isVisible():
            return
        eff = QGraphicsOpacityEffect(w)
        eff.setOpacity(0.0)
        w.setGraphicsEffect(eff)
        fade = QPropertyAnimation(eff, b"opacity", w)
        fade.setDuration(240)
        fade.setStartValue(0.0)
        fade.setEndValue(1.0)
        fade.setEasingCurve(QEasingCurve.Type.OutCubic)

        def _done():
            try:
                w.setGraphicsEffect(None)
            except RuntimeError:
                pass
        fade.finished.connect(_done)
        fade.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
        if self._slide:
            end = w.pos()
            slide = QPropertyAnimation(w, b"pos", w)
            slide.setDuration(280)
            slide.setStartValue(end + QPoint(0, self._slide))
            slide.setEndValue(end)
            slide.setEasingCurve(QEasingCurve.Type.OutCubic)
            slide.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)


# ======================================================================= progresso/estado
class SmoothProgress(ThemeAware, QWidget):
    """Barra de progresso com valor suavizado, brilho que percorre a parte preenchida e modo indeterminado."""

    def __init__(self, height: int = 8, parent=None):
        super().__init__(parent)
        self.setFixedHeight(height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._value = 0.0
        self._target = 0.0
        self._indeterminate = False
        self._active = False
        self._phase = 0.0
        self._va = _anim(self, 0, 0, 450, self._set_value)
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._watch_theme()

    def _set_value(self, v):
        self._value = float(v)
        self.update()

    def set_value(self, fraction: float, animate: bool = True):
        f = max(0.0, min(1.0, float(fraction)))
        if abs(f - self._target) < 1e-4:
            return
        self._target = f
        if animate and self.isVisible() and f > self._value:
            self._va.stop(); self._va.setStartValue(self._value); self._va.setEndValue(f); self._va.start()
        else:
            self._va.stop()
            self._value = f
            self.update()

    def value(self) -> float:
        return self._target

    def set_indeterminate(self, on: bool):
        self._indeterminate = on
        self._sync_timer()
        self.update()

    def set_active(self, on: bool):
        self._active = on
        self._sync_timer()
        self.update()

    def _sync_timer(self):
        run = (self._active or self._indeterminate) and self.isVisible()
        if run and not self._timer.isActive():
            self._timer.start()
        elif not run:
            self._timer.stop()

    def showEvent(self, e):
        self._sync_timer()
        super().showEvent(e)

    def hideEvent(self, e):
        self._timer.stop()
        super().hideEvent(e)

    def _tick(self):
        self._phase = (self._phase + 0.016 / (1.5 if self._indeterminate else 2.2)) % 1.0
        self.update()

    def paintEvent(self, _e):
        t = theme.tokens()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect())
        rad = r.height() / 2
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(t["surface3"]))
        p.drawRoundedRect(r, rad, rad)
        if self._indeterminate:
            w = r.width() * 0.32
            ph = self._phase
            ease = 0.5 - 0.5 * math.cos(ph * math.pi * 2)
            x = -w + (r.width() + w) * ease if ph < 0.5 else r.width() - (r.width() + w) * ease
            x = (r.width() + w) * ph - w
            fill = QRectF(max(r.left(), x), r.top(), min(w, r.right() - x) - max(0, r.left() - x), r.height())
        else:
            fill = QRectF(r.left(), r.top(), r.width() * self._value, r.height())
        if fill.width() <= 0.5:
            return
        g = QLinearGradient(r.topLeft(), r.topRight())
        g.setColorAt(0, QColor(t["accent2"]))
        g.setColorAt(1, QColor(t["accent_hover"]))
        clip = QPainterPath()
        clip.addRoundedRect(r, rad, rad)
        p.setClipPath(clip)
        p.setBrush(g)
        p.drawRoundedRect(fill, rad, rad)
        if self._active and not self._indeterminate and fill.width() > 20:
            band = max(40.0, fill.width() * 0.25)
            bx = fill.left() - band + (fill.width() + band) * self._phase
            sg = QLinearGradient(QPointF(bx, 0), QPointF(bx + band, 0))
            sg.setColorAt(0, qcolor("#FFFFFF", 0))
            sg.setColorAt(0.5, qcolor("#FFFFFF", 0.32))
            sg.setColorAt(1, qcolor("#FFFFFF", 0))
            inner = QPainterPath()
            inner.addRoundedRect(fill, rad, rad)
            p.setClipPath(inner)
            p.fillRect(QRectF(bx, r.top(), band, r.height()), sg)


class PulseDot(ThemeAware, QWidget):
    """Ponto com anel pulsante (indicador "ao vivo"/ocupado)."""

    def __init__(self, color_key: str = "accent", size: int = 14, parent=None):
        super().__init__(parent)
        self._key = color_key
        self._phase = 0.0
        self.setFixedSize(size, size)
        self._a = _anim(self, 0, 1, 1400, self._set_phase, QEasingCurve.Type.OutCubic)
        self._a.setLoopCount(-1)
        self._watch_theme()

    def set_color_key(self, key: str):
        self._key = key
        self.update()

    def _set_phase(self, v):
        self._phase = float(v)
        self.update()

    def showEvent(self, e):
        self._a.start()
        super().showEvent(e)

    def hideEvent(self, e):
        self._a.stop()
        super().hideEvent(e)

    def paintEvent(self, _e):
        c = QColor(theme.tokens()[self._key])
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        center = QPointF(self.width() / 2, self.height() / 2)
        maxr = self.width() / 2
        p.setPen(Qt.PenStyle.NoPen)
        ring = QColor(c)
        ring.setAlphaF(0.45 * (1 - self._phase))
        p.setBrush(ring)
        p.drawEllipse(center, maxr * (0.35 + 0.65 * self._phase), maxr * (0.35 + 0.65 * self._phase))
        p.setBrush(c)
        p.drawEllipse(center, maxr * 0.36, maxr * 0.36)


class Spinner(ThemeAware, QWidget):
    def __init__(self, size: int = 18, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._angle = 0.0
        self._a = _anim(self, 0, 360, 900, self._set_angle, QEasingCurve.Type.Linear)
        self._a.setLoopCount(-1)
        self._watch_theme()

    def _set_angle(self, v):
        self._angle = float(v)
        self.update()

    def showEvent(self, e):
        self._a.start()
        super().showEvent(e)

    def hideEvent(self, e):
        self._a.stop()
        super().hideEvent(e)

    def paintEvent(self, _e):
        t = theme.tokens()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = max(2.0, self.width() / 9)
        r = QRectF(self.rect()).adjusted(w, w, -w, -w)
        p.setPen(QPen(QColor(t["surface3"]), w))
        p.drawEllipse(r)
        p.setPen(QPen(QColor(t["accent"]), w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawArc(r, int(-self._angle * 16), int(100 * 16))


class LogoMark(ThemeAware, QWidget):
    def __init__(self, size: int = 28, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._watch_theme()

    def paintEvent(self, _e):
        t = theme.tokens()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect())
        g = QLinearGradient(r.topLeft(), r.bottomRight())
        g.setColorAt(0, QColor("#9384FF"))
        g.setColorAt(1, QColor("#4433D6"))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(g)
        p.drawRoundedRect(r, r.width() * 0.3, r.width() * 0.3)
        s = r.width() * 0.62
        icons.draw(p, "wave", QRectF((r.width() - s) / 2, (r.height() - s) / 2, s, s), QColor("#FFFFFF"), 2.6)
        _ = t


class ElidedLabel(QLabel):
    """QLabel que encurta o texto com reticências em vez de forçar a largura do layout."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        self._full = text
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(40)
        super().setText(text)

    def setText(self, text: str):  # noqa: N802
        self._full = text or ""
        self._elide()

    def full_text(self) -> str:
        return self._full

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._elide()

    def _elide(self):
        fm = QFontMetrics(self.font())
        super().setText(fm.elidedText(self._full, Qt.TextElideMode.ElideMiddle, max(10, self.width())))
        self.setToolTip(self._full if fm.horizontalAdvance(self._full) > self.width() else "")


class RoundedMenu(QMenu):
    """QMenu com cantos arredondados de verdade (janela translúcida, sem sombra quadrada do sistema)."""

    def __init__(self, title: str = "", parent=None):
        super().__init__(title, parent)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.FramelessWindowHint
                            | Qt.WindowType.NoDropShadowWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)


class DropTarget(ThemeAware, QWidget):
    """Área clicável com borda tracejada que acende no hover/arrasto."""

    clicked = Signal()

    def __init__(self, radius: int = 16, parent=None):
        super().__init__(parent)
        self._radius = radius
        self._hover = 0.0
        self._active = False
        self._enabled = True
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._ha = _anim(self, 0, 1, 180, self._set_hover)
        self._watch_theme()

    def _set_hover(self, v):
        self._hover = float(v)
        self.update()

    def _to(self, v):
        self._ha.stop(); self._ha.setStartValue(self._hover); self._ha.setEndValue(v); self._ha.start()

    def set_drag_active(self, on: bool):
        self._active = on
        self._to(1.0 if on else 0.0)

    def setEnabled(self, on: bool):  # noqa: N802
        super().setEnabled(on)
        self.setCursor(Qt.CursorShape.PointingHandCursor if on else Qt.CursorShape.ArrowCursor)

    def enterEvent(self, e):
        if self.isEnabled():
            self._to(1.0)
        super().enterEvent(e)

    def leaveEvent(self, e):
        if not self._active:
            self._to(0.0)
        super().leaveEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self.isEnabled() and self.rect().contains(e.position().toPoint()):
            self.clicked.emit()

    def paintEvent(self, _e):
        t = theme.tokens()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        h = self._hover
        p.setBrush(mix(qcolor(t["surface2"], 0.6), qcolor(t["accent"], 0.07 if theme.is_dark() else 0.05), h))
        pen = QPen(mix(t["border_strong"], t["accent"], h), 1.5, Qt.PenStyle.CustomDashLine)
        pen.setDashPattern([5, 4])
        p.setPen(pen)
        p.drawRoundedRect(r, self._radius, self._radius)


# ======================================================================= toasts
class Toast(ThemeAware, QWidget):
    closed = Signal(object)

    _KIND_ICON = {"info": ("info", "accent"), "success": ("success", "success"),
                  "warning": ("warning", "warning"), "error": ("error", "danger")}

    def __init__(self, host: QWidget, message: str, kind: str = "info", duration_ms: int = 4000):
        super().__init__(host)
        self._msg = message
        self._kind = kind if kind in self._KIND_ICON else "info"
        self._opacity = 0.0
        self._duration = duration_ms
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._font = QFont(host.font())
        self._font.setPixelSize(13)
        self._font.setWeight(QFont.Weight.Medium)
        self._layout_size(host.width())
        self._watch_theme()
        self._closing = False

    PAD = 14
    SHADOW = 10

    def _layout_size(self, host_w: int):
        fm = QFontMetrics(self._font)
        max_text = min(460, max(200, int(host_w * 0.6)))
        br = fm.boundingRect(QRect(0, 0, max_text, 1000), int(Qt.TextFlag.TextWordWrap), self._msg)
        self._text_w = min(max_text, br.width() + 2)
        self._text_h = br.height()
        w = self._text_w + 20 + 10 + 2 * self.PAD + 2 * self.SHADOW
        h = max(22, self._text_h) + 2 * 12 + 2 * self.SHADOW
        self.resize(w, h)

    def _get_opacity(self):
        return self._opacity

    def _set_opacity(self, v):
        self._opacity = float(v)
        self.update()

    opacity = Property(float, _get_opacity, _set_opacity)

    def present(self, pos: QPoint):
        self.move(pos + QPoint(0, 16))
        self.show()
        self.raise_()
        animate_property(self, b"opacity", 0.0, 1.0, 220)
        animate_property(self, b"pos", self.pos(), pos, 300)
        QTimer.singleShot(self._duration, self.dismiss)

    def move_to(self, pos: QPoint):
        if self._closing:
            return
        animate_property(self, b"pos", self.pos(), pos, 240)

    def dismiss(self):
        if self._closing:
            return
        self._closing = True
        animate_property(self, b"opacity", self._opacity, 0.0, 180, QEasingCurve.Type.InCubic,
                         finished=self._finish)
        animate_property(self, b"pos", self.pos(), self.pos() + QPoint(0, 8), 180, QEasingCurve.Type.InCubic)

    def _finish(self):
        self.closed.emit(self)
        self.hide()
        self.deleteLater()

    def mouseReleaseEvent(self, _e):
        self.dismiss()

    def paintEvent(self, _e):
        t = theme.tokens()
        dark = theme.is_dark()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setOpacity(self._opacity)
        s = self.SHADOW
        body = QRectF(self.rect()).adjusted(s, s - 3, -s, -s - 3)
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(s, 0, -2):
            p.setBrush(qcolor(t["shadow"], (0.05 if not dark else 0.10) * (1 - i / (s + 1))))
            p.drawRoundedRect(body.adjusted(-i, -i + 4, i, i + 4), 12 + i, 12 + i)
        p.setPen(QPen(QColor(t["border_strong"]), 1))
        p.setBrush(QColor(t["surface3"] if dark else t["surface"]))
        p.drawRoundedRect(body, 12, 12)
        icon_name, key = self._KIND_ICON[self._kind]
        ic = QRectF(body.left() + self.PAD, body.center().y() - 10, 20, 20)
        icons.draw(p, icon_name, ic, QColor(t[key]), 1.9)
        p.setFont(self._font)
        p.setPen(QColor(t["text"]))
        tr_ = QRectF(ic.right() + 10, body.top(), self._text_w, body.height())
        p.drawText(tr_, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap),
                   self._msg)


class ToastHost:
    """Empilha toasts no rodapé da janela; os de cima deslizam quando um some."""

    def __init__(self, host: QWidget, bottom_margin: int = 56):
        self._host = host
        self._bottom = bottom_margin
        self._toasts: list[Toast] = []
        self._queue: list[tuple[str, str, int]] = []

    def show(self, message: str, kind: str = "info", duration_ms: int = 4000):
        if len(self._toasts) >= 3:
            self._queue.append((message, kind, duration_ms))
            return
        toast = Toast(self._host, message, kind, duration_ms)
        toast.closed.connect(self._on_closed)
        self._toasts.append(toast)
        toast.present(self._slot(len(self._toasts) - 1, toast))

    def _slot(self, i: int, toast: Toast) -> QPoint:
        y = self._host.height() - self._bottom - toast.height()
        for other in self._toasts[:i]:
            y -= other.height() - Toast.SHADOW
        return QPoint((self._host.width() - toast.width()) // 2, y)

    def _on_closed(self, toast):
        if toast in self._toasts:
            self._toasts.remove(toast)
        for i, t in enumerate(self._toasts):
            t.move_to(self._slot(i, t))
        if self._queue:
            self.show(*self._queue.pop(0))

    def relayout(self):
        for i, t in enumerate(self._toasts):
            t.move(self._slot(i, t))


# ======================================================================= sobreposições
class DragOverlay(ThemeAware, QWidget):
    """Camada exibida sobre a janela enquanto um arquivo é arrastado."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._opacity = 0.0
        self.title = ""
        self.subtitle = ""
        self.hide()
        self._watch_theme()

    def _get_opacity(self):
        return self._opacity

    def _set_opacity(self, v):
        self._opacity = float(v)
        self.update()

    opacity = Property(float, _get_opacity, _set_opacity)

    def appear(self):
        self.setGeometry(self.parentWidget().rect())
        self.show()
        self.raise_()
        animate_property(self, b"opacity", self._opacity, 1.0, 160)

    def disappear(self):
        animate_property(self, b"opacity", self._opacity, 0.0, 160, finished=self._maybe_hide)

    def _maybe_hide(self):
        if self._opacity <= 0.01:
            self.hide()

    def paintEvent(self, _e):
        t = theme.tokens()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setOpacity(self._opacity)
        p.fillRect(self.rect(), qcolor(t["bg"], 0.86))
        r = QRectF(self.rect()).adjusted(18, 18, -18, -18)
        pen = QPen(QColor(t["accent"]), 2, Qt.PenStyle.CustomDashLine)
        pen.setDashPattern([6, 5])
        p.setPen(pen)
        p.setBrush(qcolor(t["accent"], 0.05))
        p.drawRoundedRect(r, 20, 20)
        c = r.center()
        grow = 0.9 + 0.1 * self._opacity
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qcolor(t["accent"], 0.16))
        p.drawEllipse(QPointF(c.x(), c.y() - 40), 40 * grow, 40 * grow)
        icons.draw(p, "upload", QRectF(c.x() - 18, c.y() - 58, 36, 36), QColor(t["accent_fg"]), 2.0)
        f = QFont(self.font())
        f.setPixelSize(20)
        f.setWeight(QFont.Weight.DemiBold)
        p.setFont(f)
        p.setPen(QColor(t["text"]))
        p.drawText(QRectF(r.left(), c.y() + 12, r.width(), 30), int(Qt.AlignmentFlag.AlignCenter), self.title)
        f.setPixelSize(13)
        f.setWeight(QFont.Weight.Normal)
        p.setFont(f)
        p.setPen(QColor(t["text2"]))
        p.drawText(QRectF(r.left(), c.y() + 44, r.width(), 22), int(Qt.AlignmentFlag.AlignCenter), self.subtitle)


def section_label(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setObjectName("overline")
    return lbl


class IconTile(ThemeAware, QWidget):
    """Ladrilho arredondado com ícone (identifica tipo de arquivo, estados vazios etc.)."""

    def __init__(self, icon_name: str, size: int = 40, icon_size: int = 20, color_key: str = "accent_fg",
                 soft_key: str = "accent", parent=None):
        super().__init__(parent)
        self._icon = icon_name
        self._icon_size = icon_size
        self._color_key = color_key
        self._soft_key = soft_key
        self.setFixedSize(size, size)
        self._watch_theme()

    def set_icon(self, name: str, color_key: str | None = None, soft_key: str | None = None):
        self._icon = name
        if color_key:
            self._color_key = color_key
        if soft_key:
            self._soft_key = soft_key
        self.update()

    def paintEvent(self, _e):
        t = theme.tokens()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect())
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qcolor(t[self._soft_key], 0.16 if theme.is_dark() else 0.10))
        p.drawRoundedRect(r, r.width() * 0.28, r.width() * 0.28)
        s = self._icon_size
        icons.draw(p, self._icon, QRectF((r.width() - s) / 2, (r.height() - s) / 2, s, s),
                   QColor(t[self._color_key]))


def card(object_name: str = "card", margins=(16, 16, 16, 16), spacing: int = 12, vertical: bool = True):
    """QFrame estilizado pela QSS + layout já configurado."""
    from PySide6.QtWidgets import QFrame

    f = QFrame()
    f.setObjectName(object_name)
    lay = QVBoxLayout(f) if vertical else QHBoxLayout(f)
    lay.setContentsMargins(*margins)
    lay.setSpacing(spacing)
    return f, lay
