"""Player: botão play/pause animado, forma de onda clicável (barras com cabeça de leitura suave),
tempo e velocidade de reprodução. Sincronizado nos dois sentidos com um AudioPlayer."""
from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import QEasingCurve, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QColor, QFont, QFontMetrics, QLinearGradient, QPainter, QPainterPath
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QWidget

from app.core.audio_player import AudioPlayer
from app.gui import theme
from app.gui.theme import mix, qcolor
from app.gui.widgets import PlayButton, RoundedMenu, ThemeAware, _anim
from app.utils.i18n import tr

_SPEEDS = (0.75, 1.0, 1.25, 1.5, 2.0)


def fmt_clock(seconds: float) -> str:
    total = int(max(0, seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class WaveformView(ThemeAware, QWidget):
    """Forma de onda em barras. Clique/arraste para buscar; o trecho já tocado fica no tom de destaque."""

    seek_requested = Signal(float)  # fração 0..1

    BAR = 3
    GAP = 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(48)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._peaks: np.ndarray | None = None
        self._bars: np.ndarray | None = None
        self._bars_for = (-1, None)
        self._loading = False
        self._message = ""
        self._duration = 0.0
        self._pos = 0.0                # segundos (último valor real)
        self._pos_t = 0.0              # instante do último valor real
        self._playing = False
        self._rate = 1.0
        self._hover_x = -1.0
        self._hover_a = 0.0
        self._drag = False
        self._drag_frac = 0.0
        self._reveal = 1.0
        self._phase = 0.0
        self._enabled = False
        self._ra = _anim(self, 0, 1, 700, self._set_reveal, QEasingCurve.Type.OutCubic)
        self._ha = _anim(self, 0, 1, 140, self._set_hover_a)
        self._tick = QTimer(self)
        self._tick.setInterval(16)
        self._tick.timeout.connect(self._on_tick)
        self._watch_theme()

    # ----------------------------------------------------------- estado
    def set_peaks(self, peaks):
        """np.ndarray 0..1; None = calculando (mostra brilho de carregamento)."""
        self._message = ""
        if peaks is None:
            self._loading = True
            self._peaks = None
        else:
            self._loading = False
            self._peaks = np.asarray(peaks, dtype=np.float32)
            if self.isVisible():
                self._ra.stop(); self._reveal = 0.0; self._ra.start()
            else:
                self._reveal = 1.0
        self._bars_for = (-1, None)
        self._sync_timer()
        self.update()

    def set_message(self, text: str):
        """Mensagem no lugar da onda (ex.: arquivo de origem ausente)."""
        self._message = text
        self._loading = False
        self._peaks = None
        self._bars_for = (-1, None)
        self._sync_timer()
        self.update()

    def set_interactive(self, on: bool):
        self._enabled = on
        self.setCursor(Qt.CursorShape.PointingHandCursor if on else Qt.CursorShape.ArrowCursor)
        self.update()

    def set_duration(self, d: float):
        self._duration = max(0.0, float(d or 0))
        self.update()

    def set_position(self, seconds: float):
        self._pos = float(seconds)
        self._pos_t = time.monotonic()
        if not self._playing:
            self.update()

    def set_playing(self, on: bool, rate: float = 1.0):
        self._playing = on
        self._rate = rate
        self._pos_t = time.monotonic()
        self._sync_timer()

    def set_rate(self, rate: float):
        self._pos = self._display_pos()
        self._pos_t = time.monotonic()
        self._rate = rate

    def _display_pos(self) -> float:
        if self._playing:
            return min(self._duration, self._pos + (time.monotonic() - self._pos_t) * self._rate)
        return self._pos

    def _sync_timer(self):
        run = self.isVisible() and (self._playing or self._loading)
        if run and not self._tick.isActive():
            self._tick.start()
        elif not run:
            self._tick.stop()

    def _on_tick(self):
        if self._loading:
            self._phase = (self._phase + 0.016 / 1.6) % 1.0
        self.update()

    def showEvent(self, e):
        self._sync_timer()
        super().showEvent(e)

    def hideEvent(self, e):
        self._tick.stop()
        super().hideEvent(e)

    def _set_reveal(self, v):
        self._reveal = float(v)
        self.update()

    def _set_hover_a(self, v):
        self._hover_a = float(v)
        self.update()

    # ----------------------------------------------------------- mouse
    def _frac(self, x: float) -> float:
        return max(0.0, min(1.0, x / max(1, self.width())))

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self._enabled and self._duration > 0:
            self._drag = True
            self._drag_frac = self._frac(e.position().x())
            self.update()

    def mouseMoveEvent(self, e):
        self._hover_x = e.position().x()
        if self._drag:
            self._drag_frac = self._frac(self._hover_x)
        self.update()

    def mouseReleaseEvent(self, e):
        if self._drag and e.button() == Qt.MouseButton.LeftButton:
            self._drag = False
            f = self._frac(e.position().x())
            self._pos = f * self._duration
            self._pos_t = time.monotonic()
            self.seek_requested.emit(f)
            self.update()

    def enterEvent(self, e):
        self._ha.stop(); self._ha.setStartValue(self._hover_a); self._ha.setEndValue(1.0); self._ha.start()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._ha.stop(); self._ha.setStartValue(self._hover_a); self._ha.setEndValue(0.0); self._ha.start()
        super().leaveEvent(e)

    # ----------------------------------------------------------- pintura
    def _bar_values(self, n: int) -> np.ndarray:
        if self._bars_for[0] == n and self._bars_for[1] is self._peaks and self._bars is not None:
            return self._bars
        p = self._peaks
        if p is None or p.size == 0:
            bars = np.zeros(n, dtype=np.float32)
        elif p.size >= n:
            edges = np.linspace(0, p.size, n + 1).astype(int)
            bars = np.maximum.reduceat(p, edges[:-1]) if n > 0 else np.zeros(0)
            # média entre máximo e média dá barras menos "serrilhadas"
            means = np.add.reduceat(p, edges[:-1]) / np.maximum(1, np.diff(edges))
            bars = 0.65 * bars + 0.35 * means
        else:
            bars = np.interp(np.linspace(0, p.size - 1, n), np.arange(p.size), p).astype(np.float32)
        self._bars = bars
        self._bars_for = (n, self._peaks)
        return bars

    def paintEvent(self, _e):
        t = theme.tokens()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        mid = h / 2
        step = self.BAR + self.GAP
        n = max(1, int((w + self.GAP) // step))
        x0 = (w - (n * step - self.GAP)) / 2

        if self._message:
            p.setPen(QColor(t["text3"]))
            f = QFont(self.font()); f.setPixelSize(12); p.setFont(f)
            p.drawText(self.rect(), int(Qt.AlignmentFlag.AlignCenter), self._message)
            return

        dur = self._duration
        frac = (self._drag_frac if self._drag else (self._display_pos() / dur if dur > 0 else 0.0))
        play_x = frac * w
        base = qcolor(t["text3"], 0.45 if theme.is_dark() else 0.38)
        played_a = QColor(t["accent"])
        played_b = QColor(t["accent_hover"])
        hover_col = qcolor(t["text2"], 0.75)
        p.setPen(Qt.PenStyle.NoPen)

        if self._loading or self._peaks is None:
            # Esqueleto: barras baixas com um brilho que percorre a largura.
            for i in range(n):
                x = x0 + i * step
                d = abs((x / max(1, w)) - self._phase)
                glow = max(0.0, 1 - d * 5) if self._loading else 0.0
                bh = 3 + 5 * glow
                p.setBrush(mix(base, qcolor(t["accent"], 0.7), glow * 0.8))
                p.drawRoundedRect(QRectF(x, mid - bh / 2, self.BAR, bh), 1.5, 1.5)
            return

        bars = self._bar_values(n)
        hover = self._hover_x if (self._hover_a > 0 and self._enabled) else -1
        grad = QLinearGradient(QPointF(0, 0), QPointF(max(1.0, play_x), 0))
        grad.setColorAt(0, played_a)
        grad.setColorAt(1, played_b)
        for i in range(n):
            x = x0 + i * step
            v = float(bars[i])
            rev = max(0.0, min(1.0, self._reveal * 1.6 - (i / n) * 0.6))
            bh = max(3.0, (h - 6) * v * rev)
            r = QRectF(x, mid - bh / 2, self.BAR, bh)
            if x + self.BAR <= play_x:
                p.setBrush(grad)
            elif hover >= 0 and x <= hover:
                p.setBrush(mix(base, hover_col, self._hover_a * 0.6))
            else:
                p.setBrush(base)
            p.drawRoundedRect(r, 1.5, 1.5)

        if dur > 0 and (frac > 0 or self._playing):
            p.setBrush(QColor(t["text"]))
            p.drawRoundedRect(QRectF(play_x - 1, 2, 2, h - 4), 1, 1)

        if hover >= 0 and dur > 0:
            label = fmt_clock(self._frac(hover) * dur)
            f = QFont(self.font()); f.setPixelSize(11); f.setWeight(QFont.Weight.DemiBold)
            fm = QFontMetrics(f)
            bw, bh = fm.horizontalAdvance(label) + 12, 18
            bx = max(0.0, min(w - bw, hover - bw / 2))
            p.setOpacity(self._hover_a)
            p.setBrush(QColor(t["surface3"] if theme.is_dark() else t["text"]))
            path = QPainterPath()
            path.addRoundedRect(QRectF(bx, 0, bw, bh), 6, 6)
            p.drawPath(path)
            p.setPen(QColor(t["text"] if theme.is_dark() else t["surface"]))
            p.setFont(f)
            p.drawText(QRectF(bx, 0, bw, bh), int(Qt.AlignmentFlag.AlignCenter), label)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(qcolor(t["text2"], 0.5))
            p.drawRect(QRectF(hover - 0.5, bh + 2, 1, h - bh - 4))


class PlayerControls(QWidget):
    def __init__(self, player: AudioPlayer, parent=None):
        super().__init__(parent)
        self._player = player
        self._rate = 1.0
        self._loaded = False
        self._setup_ui()
        self._wire_signals()
        self.set_file_loaded(False)

    def _setup_ui(self):
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(14)
        self._play_btn = PlayButton(44)
        self._play_btn.setToolTip(tr("play"))
        lay.addWidget(self._play_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        self.wave = WaveformView()
        self.wave.setFixedHeight(56)
        lay.addWidget(self.wave, 1)
        self._time = QLabel("0:00 / 0:00")
        self._time.setObjectName("mono")
        self._time.setMinimumWidth(QFontMetrics(self._time.font()).horizontalAdvance("00:00:00 / 00:00:00") - 10)
        self._time.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(self._time)
        self._speed_btn = QPushButton("1×")
        self._speed_btn.setObjectName("ghost")
        self._speed_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._speed_btn.setFixedWidth(52)
        self._speed_btn.setToolTip(tr("playback_speed"))
        self._speed_menu = RoundedMenu(parent=self)
        group = QActionGroup(self)
        for s in _SPEEDS:
            act = QAction(self._speed_label(s), self, checkable=True)
            act.setChecked(s == 1.0)
            act.triggered.connect(lambda _=False, v=s: self._set_rate(v))
            group.addAction(act)
            self._speed_menu.addAction(act)
        self._speed_btn.setMenu(self._speed_menu)
        lay.addWidget(self._speed_btn)

    @staticmethod
    def _speed_label(s: float) -> str:
        return f"{s:g}×"

    def _wire_signals(self):
        self._play_btn.clicked.connect(self._player.toggle_play_pause)
        self.wave.seek_requested.connect(self._on_seek)
        self._player.position_changed.connect(self._on_player_position)
        self._player.state_changed.connect(self._on_player_state)
        self._player.duration_changed.connect(self._on_player_duration)

    def _set_rate(self, r: float):
        self._rate = r
        self._player.set_rate(r)
        self.wave.set_rate(r)
        self._speed_btn.setText(self._speed_label(r))

    def set_file_loaded(self, loaded: bool):
        self._loaded = loaded
        self._play_btn.setEnabled(loaded)
        self._speed_btn.setEnabled(loaded)
        self.wave.set_interactive(loaded)
        if not loaded:
            self._time.setText("0:00 / 0:00")
            self.wave.set_position(0)
            self.wave.set_playing(False)
            self._play_btn.set_playing(False)

    def set_peaks(self, peaks):
        self.wave.set_peaks(peaks)

    def set_message(self, text: str):
        self.wave.set_message(text)

    def _on_seek(self, frac: float):
        d = self._player.duration
        if d > 0:
            self._player.seek(frac * d)
            self._on_player_position(frac * d)

    def _on_player_position(self, seconds: float):
        self.wave.set_position(seconds)
        self._time.setText(f"{fmt_clock(seconds)} / {fmt_clock(self._player.duration)}")

    def _on_player_state(self, state: str):
        playing = state == "playing"
        self._play_btn.set_playing(playing)
        self._play_btn.setToolTip(tr("pause") if playing else tr("play"))
        self.wave.set_position(self._player.position)
        self.wave.set_playing(playing, self._rate)

    def _on_player_duration(self, duration: float):
        self.wave.set_duration(duration)
        self._time.setText(f"{fmt_clock(self._player.position)} / {fmt_clock(duration)}")

    def retranslate(self):
        self._speed_btn.setToolTip(tr("playback_speed"))
        self._play_btn.setToolTip(tr("pause") if self._player.is_playing else tr("play"))
