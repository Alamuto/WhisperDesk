"""Gera assets/icon.ico (16..256 px) e assets/icon.png desenhando o ícone com QPainter (sem arquivos externos)."""
from __future__ import annotations

import math
import struct
import sys
from io import BytesIO
from pathlib import Path

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QImage, QLinearGradient, QPainter, QPainterPath, QPen

ROOT = Path(__file__).resolve().parents[1]
SIZES = (16, 24, 32, 48, 64, 128, 256)


def render(size: int) -> QImage:
    img = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    m = size * 0.04
    rect = QRectF(m, m, size - 2 * m, size - 2 * m)
    g = QLinearGradient(rect.topLeft(), rect.bottomRight())
    g.setColorAt(0.0, QColor("#7C6CFF"))
    g.setColorAt(1.0, QColor("#3B2FC9"))
    path = QPainterPath()
    path.addRoundedRect(rect, size * 0.22, size * 0.22)
    p.fillPath(path, g)
    # barras de forma de onda
    bars = 5 if size >= 32 else 3
    heights = [0.35, 0.7, 1.0, 0.7, 0.35] if bars == 5 else [0.5, 1.0, 0.5]
    inner = rect.adjusted(size * 0.2, size * 0.24, -size * 0.2, -size * 0.24)
    step = inner.width() / bars
    w = max(1.0, step * 0.55)
    pen = QPen(QColor("white"), w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    cy = inner.center().y()
    for i, h in enumerate(heights):
        x = inner.left() + step * (i + 0.5)
        half = max(w * 0.1, inner.height() * h / 2 - w / 2)
        p.drawLine(QPointF(x, cy - half), QPointF(x, cy + half))
    p.end()
    return img


def png_bytes(img: QImage) -> bytes:
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG")
    return bytes(ba)


def write_ico(path: Path, images: list[QImage]) -> None:
    """ICO com entradas PNG (suportado desde o Windows Vista)."""
    blobs = [png_bytes(i) for i in images]
    out = BytesIO()
    out.write(struct.pack("<HHH", 0, 1, len(blobs)))
    offset = 6 + 16 * len(blobs)
    for img, b in zip(images, blobs):
        s = img.width()
        out.write(struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(b), offset))
        offset += len(b)
    for b in blobs:
        out.write(b)
    path.write_bytes(out.getvalue())


def main() -> None:
    app = QGuiApplication.instance() or QGuiApplication(sys.argv)  # noqa: F841
    imgs = [render(s) for s in SIZES]
    write_ico(ROOT / "assets" / "icon.ico", imgs)
    render(512).save(str(ROOT / "assets" / "icon.png"))
    print("ok", ROOT / "assets" / "icon.ico")
    _ = math


if __name__ == "__main__":
    main()
