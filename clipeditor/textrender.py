"""Renders text clips to a transparent full-canvas QImage.

Shared by the live preview and by export (export writes the image to a PNG and overlays it),
so titles look identical in both.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QGuiApplication, QImage, QPainter, QPainterPath, QPen

from .model import Clip


def ensure_qapp():
    """Text rendering needs a Q(Gui)Application; create an offscreen one for CLI export."""
    if QGuiApplication.instance() is None:
        import os
        import sys

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        if os.name == "nt":  # offscreen platform finds no fonts on Windows otherwise
            os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))
        ensure_qapp._app = QGuiApplication(sys.argv[:1])  # keep a reference


def parse_color(s: str, default: str = "#ffffff") -> QColor:
    s = (s or default).strip()
    if len(s) == 9 and s.startswith("#"):  # #rrggbbaa
        c = QColor(s[:7])
        c.setAlpha(int(s[7:9], 16))
        return c
    c = QColor(s)
    return c if c.isValid() else QColor(default)


def render_text_element(clip: Clip, px_scale: float) -> QImage:
    """Tight transparent image of the title (box + outline included), centred on its middle.

    px_scale = output pixels per project pixel (canvas_h / project_h x animated scale)."""
    font = QFont(clip.font_family)
    font.setPixelSize(max(int(clip.font_size * px_scale), 4))
    font.setBold(clip.bold)
    fm = QFontMetricsF(font)
    lines = (clip.text or " ").split("\n")
    line_h = fm.lineSpacing()
    widths = [fm.horizontalAdvance(line) for line in lines]
    block_w = max(max(widths, default=1), 1)
    block_h = line_h * len(lines)
    pad = font.pixelSize() * 0.35
    margin = pad + font.pixelSize() / 10 + 2
    W = int(block_w + 2 * margin) + 2
    H = int(block_h + 2 * margin) + 2
    img = QImage(W, H, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    cx, top = W / 2, (H - block_h) / 2
    if clip.bg_color:
        box = QRectF(cx - block_w / 2 - pad, top - pad * 0.6, block_w + 2 * pad, block_h + pad * 1.2)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(parse_color(clip.bg_color, "#000000"))
        p.drawRoundedRect(box, pad * 0.6, pad * 0.6)
    path = QPainterPath()
    for i, line in enumerate(lines):
        path.addText(QPointF(cx - widths[i] / 2, top + i * line_h + fm.ascent()), font, line)
    if not clip.bg_color:
        p.setPen(QPen(QColor(0, 0, 0, 200), max(font.pixelSize() / 14, 1.5), Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(parse_color(clip.color))
    p.drawPath(path)
    p.end()
    return img


def render_text(clip: Clip, w: int, h: int, project_height: int) -> QImage:
    img = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    if not clip.text:
        return img
    k = (h / max(project_height, 1)) * clip.scale
    font = QFont(clip.font_family)
    font.setPixelSize(max(int(clip.font_size * k), 4))
    font.setBold(clip.bold)
    fm = QFontMetricsF(font)
    lines = clip.text.split("\n")
    line_h = fm.lineSpacing()
    widths = [fm.horizontalAdvance(line) for line in lines]
    block_w = max(widths, default=0)
    block_h = line_h * len(lines)
    cx = w / 2 + clip.pos_x * w
    cy = h / 2 + clip.pos_y * h
    top = cy - block_h / 2

    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    if clip.bg_color:
        pad = font.pixelSize() * 0.35
        box = QRectF(cx - block_w / 2 - pad, top - pad * 0.6, block_w + 2 * pad, block_h + pad * 1.2)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(parse_color(clip.bg_color, "#000000"))
        p.drawRoundedRect(box, pad * 0.6, pad * 0.6)

    path = QPainterPath()
    for i, line in enumerate(lines):
        x = cx - widths[i] / 2
        y = top + i * line_h + fm.ascent()
        path.addText(QPointF(x, y), font, line)
    if not clip.bg_color:  # outline keeps text readable on any footage
        p.setPen(QPen(QColor(0, 0, 0, 200), max(font.pixelSize() / 14, 1.5), Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(parse_color(clip.color))
    p.drawPath(path)
    p.end()
    return img
