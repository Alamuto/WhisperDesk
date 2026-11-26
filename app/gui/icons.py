"""Ícones vetoriais desenhados com QPainter (grade de 24 unidades, traço arredondado).

Sem dependência de fontes de símbolos (que viravam quadrados em algumas máquinas) nem de QtSvg."""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap


def _p(*pts) -> QPainterPath:
    path = QPainterPath(QPointF(*pts[0]))
    for pt in pts[1:]:
        path.lineTo(QPointF(*pt))
    return path


def draw(painter: QPainter, name: str, rect: QRectF, color: QColor | str, stroke: float = 1.8) -> None:
    """Desenha o ícone `name` dentro de `rect` (quadrado)."""
    color = QColor(color)
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    side = min(rect.width(), rect.height())
    painter.translate(rect.center().x() - side / 2, rect.center().y() - side / 2)
    painter.scale(side / 24.0, side / 24.0)
    pen = QPen(color, stroke, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    fn = _ICONS.get(name)
    if fn:
        fn(painter, color)
    painter.restore()


def _fill(painter: QPainter, color: QColor):
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(color)


def _play(p, c):
    _fill(p, c)
    path = QPainterPath()
    path.moveTo(8.2, 5.6)
    path.cubicTo(8.2, 4.8, 9.0, 4.3, 9.7, 4.7)
    path.lineTo(19.0, 10.9)
    path.cubicTo(19.7, 11.4, 19.7, 12.6, 19.0, 13.1)
    path.lineTo(9.7, 19.3)
    path.cubicTo(9.0, 19.7, 8.2, 19.2, 8.2, 18.4)
    path.closeSubpath()
    p.drawPath(path)


def _pause(p, c):
    _fill(p, c)
    p.drawRoundedRect(QRectF(6.5, 5, 3.8, 14), 1.4, 1.4)
    p.drawRoundedRect(QRectF(13.7, 5, 3.8, 14), 1.4, 1.4)


def _stop(p, c):
    _fill(p, c)
    p.drawRoundedRect(QRectF(6.5, 6.5, 11, 11), 2.2, 2.2)


def _upload(p, c):
    p.drawPath(_p((12, 15), (12, 4)))
    p.drawPath(_p((7.5, 8.5), (12, 4), (16.5, 8.5)))
    p.drawPath(_p((4.5, 14), (4.5, 18), (19.5, 18), (19.5, 14)))


def _download(p, c):
    p.drawPath(_p((12, 4), (12, 14.5)))
    p.drawPath(_p((7.5, 10), (12, 14.5), (16.5, 10)))
    p.drawPath(_p((4.5, 15.5), (4.5, 19), (19.5, 19), (19.5, 15.5)))


def _file_audio(p, c):
    body = QPainterPath()
    body.moveTo(14, 3.5)
    body.lineTo(7, 3.5)
    body.quadTo(5.5, 3.5, 5.5, 5)
    body.lineTo(5.5, 19)
    body.quadTo(5.5, 20.5, 7, 20.5)
    body.lineTo(17, 20.5)
    body.quadTo(18.5, 20.5, 18.5, 19)
    body.lineTo(18.5, 8)
    body.closeSubpath()
    p.drawPath(body)
    p.drawPath(_p((14, 3.5), (14, 8), (18.5, 8)))
    for x, h in ((9, 2.0), (11, 4.0), (13, 2.8), (15, 1.4)):
        p.drawLine(QPointF(x, 14 - h), QPointF(x, 14 + h))


def _file_video(p, c):
    p.drawRoundedRect(QRectF(3.5, 6, 12.5, 12), 2.5, 2.5)
    p.drawPath(_p((16, 10.5), (20.5, 7.8), (20.5, 16.2), (16, 13.5)))


def _wave(p, c):
    for x, h in ((4, 2.5), (8, 6.0), (12, 8.5), (16, 5.0), (20, 2.5)):
        p.drawLine(QPointF(x, 12 - h), QPointF(x, 12 + h))


def _history(p, c):
    arc = QPainterPath()
    arc.arcMoveTo(QRectF(4, 4, 16, 16), 160)
    arc.arcTo(QRectF(4, 4, 16, 16), 160, -320)
    p.drawPath(arc)
    p.drawPath(_p((3.6, 7.6), (4.5, 10.6), (7.6, 9.8)))
    p.drawPath(_p((12, 8), (12, 12), (14.8, 13.8)))


def _sun(p, c):
    p.drawEllipse(QPointF(12, 12), 3.8, 3.8)
    for i in range(8):
        a = i * math.pi / 4
        p.drawLine(QPointF(12 + math.cos(a) * 6.6, 12 + math.sin(a) * 6.6),
                   QPointF(12 + math.cos(a) * 8.6, 12 + math.sin(a) * 8.6))


def _moon(p, c):
    outer = QPainterPath()
    outer.addEllipse(QPointF(12, 12), 8, 8)
    cut = QPainterPath()
    cut.addEllipse(QPointF(16.5, 8.2), 6.6, 6.6)
    p.drawPath(outer.subtracted(cut))


def _more(p, c):
    _fill(p, c)
    for x in (6, 12, 18):
        p.drawEllipse(QPointF(x, 12), 1.6, 1.6)


def _folder(p, c):
    path = QPainterPath()
    path.moveTo(3.5, 7)
    path.quadTo(3.5, 5.5, 5, 5.5)
    path.lineTo(9.3, 5.5)
    path.lineTo(11.2, 7.5)
    path.lineTo(19, 7.5)
    path.quadTo(20.5, 7.5, 20.5, 9)
    path.lineTo(20.5, 17.5)
    path.quadTo(20.5, 19, 19, 19)
    path.lineTo(5, 19)
    path.quadTo(3.5, 19, 3.5, 17.5)
    path.closeSubpath()
    p.drawPath(path)


def _globe(p, c):
    p.drawEllipse(QPointF(12, 12), 8.5, 8.5)
    p.drawEllipse(QPointF(12, 12), 3.6, 8.5)
    p.drawLine(QPointF(3.8, 9), QPointF(20.2, 9))
    p.drawLine(QPointF(3.8, 15), QPointF(20.2, 15))


def _copy(p, c):
    p.drawRoundedRect(QRectF(8.5, 8.5, 11, 11), 2.4, 2.4)
    p.drawPath(_p((15.5, 8.5), (15.5, 6.5)))
    path = QPainterPath()
    path.moveTo(15.5, 6.5)
    path.quadTo(15.5, 4.5, 13.5, 4.5)
    path.lineTo(6.5, 4.5)
    path.quadTo(4.5, 4.5, 4.5, 6.5)
    path.lineTo(4.5, 13.5)
    path.quadTo(4.5, 15.5, 6.5, 15.5)
    path.lineTo(8.5, 15.5)
    p.drawPath(path)


def _search(p, c):
    p.drawEllipse(QPointF(10.5, 10.5), 6, 6)
    p.drawLine(QPointF(15, 15), QPointF(19.5, 19.5))


def _chev_down(p, c):
    p.drawPath(_p((7, 9.5), (12, 14.5), (17, 9.5)))


def _chev_up(p, c):
    p.drawPath(_p((7, 14.5), (12, 9.5), (17, 14.5)))


def _chev_right(p, c):
    p.drawPath(_p((9.5, 7), (14.5, 12), (9.5, 17)))


def _close(p, c):
    p.drawLine(QPointF(7, 7), QPointF(17, 17))
    p.drawLine(QPointF(17, 7), QPointF(7, 17))


def _check(p, c):
    p.drawPath(_p((5.5, 12.5), (10, 17), (18.5, 7.5)))


def _terminal(p, c):
    p.drawRoundedRect(QRectF(3.5, 5, 17, 14), 2.6, 2.6)
    p.drawPath(_p((7.5, 9.5), (10.5, 12), (7.5, 14.5)))
    p.drawLine(QPointF(12.5, 15), QPointF(16.5, 15))


def _sparkle(p, c):
    _fill(p, c)
    path = QPainterPath()
    path.moveTo(11, 3)
    path.cubicTo(11.6, 7.8, 13.2, 9.4, 18, 10)
    path.cubicTo(13.2, 10.6, 11.6, 12.2, 11, 17)
    path.cubicTo(10.4, 12.2, 8.8, 10.6, 4, 10)
    path.cubicTo(8.8, 9.4, 10.4, 7.8, 11, 3)
    p.drawPath(path)
    small = QPainterPath()
    small.moveTo(18, 14.5)
    small.cubicTo(18.3, 16.3, 18.9, 16.9, 20.7, 17.2)
    small.cubicTo(18.9, 17.5, 18.3, 18.1, 18, 19.9)
    small.cubicTo(17.7, 18.1, 17.1, 17.5, 15.3, 17.2)
    small.cubicTo(17.1, 16.9, 17.7, 16.3, 18, 14.5)
    p.drawPath(small)


def _swap(p, c):
    p.drawPath(_p((4.5, 9), (18.5, 9)))
    p.drawPath(_p((15, 5.5), (18.5, 9), (15, 12.5)))
    p.drawPath(_p((19.5, 15), (5.5, 15)))
    p.drawPath(_p((9, 11.5), (5.5, 15), (9, 18.5)))


def _info(p, c):
    p.drawEllipse(QPointF(12, 12), 8.5, 8.5)
    p.drawLine(QPointF(12, 11), QPointF(12, 16))
    _fill(p, c)
    p.drawEllipse(QPointF(12, 7.9), 1.1, 1.1)


def _warning(p, c):
    path = QPainterPath()
    path.moveTo(12, 4)
    path.lineTo(20.5, 19)
    path.lineTo(3.5, 19)
    path.closeSubpath()
    p.drawPath(path)
    p.drawLine(QPointF(12, 9.5), QPointF(12, 13.5))
    _fill(p, c)
    p.drawEllipse(QPointF(12, 16.3), 1.05, 1.05)


def _success(p, c):
    p.drawEllipse(QPointF(12, 12), 8.5, 8.5)
    p.drawPath(_p((8.2, 12.3), (10.8, 14.9), (15.9, 9.4)))


def _error(p, c):
    p.drawEllipse(QPointF(12, 12), 8.5, 8.5)
    p.drawLine(QPointF(9.2, 9.2), QPointF(14.8, 14.8))
    p.drawLine(QPointF(14.8, 9.2), QPointF(9.2, 14.8))


def _trash(p, c):
    p.drawLine(QPointF(4.5, 7), QPointF(19.5, 7))
    p.drawPath(_p((9.5, 7), (9.5, 4.5), (14.5, 4.5), (14.5, 7)))
    p.drawPath(_p((6.5, 7), (7.5, 19.5), (16.5, 19.5), (17.5, 7)))


def _chip(p, c):
    p.drawRoundedRect(QRectF(6.5, 6.5, 11, 11), 2, 2)
    for v in (9.5, 14.5):
        p.drawLine(QPointF(v, 3.5), QPointF(v, 6.5))
        p.drawLine(QPointF(v, 17.5), QPointF(v, 20.5))
        p.drawLine(QPointF(3.5, v), QPointF(6.5, v))
        p.drawLine(QPointF(17.5, v), QPointF(20.5, v))


def _sliders(p, c):
    for y, x in ((7, 15), (12, 8.5), (17, 13)):
        p.drawLine(QPointF(4, y), QPointF(20, y))
        _fill(p, c)
        p.drawEllipse(QPointF(x, y), 2.1, 2.1)
        p.setPen(QPen(c, 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.setBrush(Qt.BrushStyle.NoBrush)


def _speed(p, c):
    arc = QPainterPath()
    arc.arcMoveTo(QRectF(3.5, 5, 17, 17), 200)
    arc.arcTo(QRectF(3.5, 5, 17, 17), 200, -220)
    p.drawPath(arc)
    p.drawLine(QPointF(12, 13.5), QPointF(15.5, 9.5))


_ICONS = {
    "play": _play, "pause": _pause, "stop": _stop, "upload": _upload, "download": _download,
    "file-audio": _file_audio, "file-video": _file_video, "wave": _wave, "history": _history,
    "sun": _sun, "moon": _moon, "more": _more, "folder": _folder, "globe": _globe, "copy": _copy,
    "search": _search, "chevron-down": _chev_down, "chevron-up": _chev_up, "chevron-right": _chev_right,
    "close": _close, "check": _check, "terminal": _terminal, "sparkle": _sparkle, "swap": _swap,
    "info": _info, "warning": _warning, "success": _success, "error": _error, "trash": _trash,
    "chip": _chip, "sliders": _sliders, "speed": _speed,
}


def icon(name: str, color: QColor | str, size: int = 18, stroke: float = 1.8) -> QIcon:
    """QIcon nítido em telas de alta densidade (pixmaps 1x, 2x e 3x)."""
    ic = QIcon()
    for scale in (1, 2, 3):
        pm = QPixmap(size * scale, size * scale)
        pm.setDevicePixelRatio(scale)
        pm.fill(Qt.GlobalColor.transparent)
        qp = QPainter(pm)
        draw(qp, name, QRectF(0, 0, size, size), color, stroke)
        qp.end()
        ic.addPixmap(pm)
    return ic
