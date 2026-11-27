"""Shape clips (rectangles, circles, stars, arrows, bubbles, emoji stickers ...), rendered with Qt.

CONTRACT (used by clipeditor.render.Compositor):
    PRESETS: dict[name, dict]   ready-made shape dicts for the UI gallery
    render_shape(clip, canvas_h: int) -> QImage   tight ARGB image of the shape at scale 1
        (the compositor positions, scales, rotates and fades it like any other layer).
        Sizes in clip.shape are fractions of canvas height so output scales with resolution.

Shape dict keys (all optional except "kind"; lengths are fractions of canvas height):
    kind        rect | rounded_rect | ellipse | circle | triangle | star | polygon | heart |
                arrow | line | bubble | banner | emoji
    w, h        size of the shape body
    fill        "#rrggbb" / "#rrggbbaa" or "" for no fill
    fill2       second gradient colour ("" = solid fill);  angle: gradient direction, degrees
    fill_opacity  0..1, multiplies the fill only (outline stays opaque)
    stroke      outline colour ("" = none);  stroke_w: outline width
    shadow      {} = none, else {"dx", "dy", "blur", "color"}  (dx/dy/blur fractions of canvas h)
    radius      rounded_rect / bubble / line corner radius, fraction of min(w, h) (0..0.5)
    points, inner   star point count and inner radius ratio
    sides       polygon side count
    head, shaft arrow head length (fraction of w) and shaft thickness (fraction of h)
    tail        bubble tail position along the bottom, 0..1
    notch       banner end-notch depth, fraction of h
    char        emoji character (kind "emoji"), drawn with the Segoe UI Emoji colour font
    label, label_color, label_size, label_font   optional centred text (label_size: fraction of h)

Gallery API:
    PRESETS, SHAPE_PRESETS (names), EMOJI_PRESETS (names), preset_category(name)
    new_shape_clip_fields(name) -> dict of Clip kwargs
    thumbnail(name_or_shape, size) -> QImage fitted in a size x size square
"""
from __future__ import annotations

import copy
import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QBrush, QColor, QFont, QImage, QLinearGradient, QPainter, QPainterPath, QPen,
                           QPolygonF)

EMOJI_FONT = "Segoe UI Emoji"
DEFAULT_SHAPE_DURATION = 5.0
DEFAULT_EMOJI_DURATION = 3.0

_SHADOW = {"dx": 0.006, "dy": 0.008, "blur": 0.012, "color": "#00000099"}
_SOFT_SHADOW = {"dx": 0.0, "dy": 0.006, "blur": 0.018, "color": "#00000080"}

PRESETS: dict[str, dict] = {
    # -- basic shapes
    "Rectangle": {"kind": "rect", "w": 0.5, "h": 0.3, "fill": "#ffffff"},
    "Rounded Rectangle": {"kind": "rounded_rect", "w": 0.5, "h": 0.3, "fill": "#3b82f6", "radius": 0.2},
    "Circle": {"kind": "circle", "w": 0.3, "h": 0.3, "fill": "#f59e0b"},
    "Ellipse": {"kind": "ellipse", "w": 0.45, "h": 0.28, "fill": "#10b981"},
    "Triangle": {"kind": "triangle", "w": 0.3, "h": 0.27, "fill": "#8b5cf6"},
    "Star": {"kind": "star", "w": 0.3, "h": 0.3, "fill": "#facc15", "points": 5, "inner": 0.45,
             "stroke": "#b45309", "stroke_w": 0.006},
    "Hexagon": {"kind": "polygon", "w": 0.3, "h": 0.3, "fill": "#06b6d4", "sides": 6},
    "Heart": {"kind": "heart", "w": 0.3, "h": 0.27, "fill": "#ef4444"},
    "Arrow": {"kind": "arrow", "w": 0.4, "h": 0.2, "fill": "#ffffff", "head": 0.4, "shaft": 0.42},
    "Line": {"kind": "line", "w": 0.6, "h": 0.012, "fill": "#ffffff"},
    # -- design elements
    "Lower Third Bar": {"kind": "rect", "w": 1.15, "h": 0.12, "fill": "#1e3a8a", "fill2": "#2563eb",
                        "angle": 0, "fill_opacity": 0.92, "shadow": dict(_SOFT_SHADOW)},
    "Subscribe Button": {"kind": "rounded_rect", "w": 0.42, "h": 0.12, "fill": "#ff0000", "radius": 0.18,
                         "label": "SUBSCRIBE", "label_color": "#ffffff", "label_size": 0.42,
                         "shadow": dict(_SHADOW)},
    "Arrow Callout": {"kind": "arrow", "w": 0.36, "h": 0.16, "fill": "#fde047", "head": 0.42, "shaft": 0.45,
                      "stroke": "#111111", "stroke_w": 0.007, "shadow": dict(_SHADOW)},
    "Highlight Circle": {"kind": "ellipse", "w": 0.36, "h": 0.28, "fill": "", "stroke": "#ef4444",
                         "stroke_w": 0.012},
    "Highlight Box": {"kind": "rounded_rect", "w": 0.45, "h": 0.25, "fill": "#fde047", "fill_opacity": 0.18,
                      "stroke": "#fde047", "stroke_w": 0.009, "radius": 0.08},
    "Star Burst": {"kind": "star", "w": 0.34, "h": 0.34, "fill": "#fde047", "fill2": "#f97316", "angle": 90,
                   "points": 16, "inner": 0.72, "label": "NEW!", "label_color": "#7c2d12", "label_size": 0.22,
                   "shadow": dict(_SHADOW)},
    "Gradient Badge": {"kind": "circle", "w": 0.26, "h": 0.26, "fill": "#8b5cf6", "fill2": "#ec4899",
                       "angle": 45, "stroke": "#ffffff", "stroke_w": 0.008, "label": "#1",
                       "label_color": "#ffffff", "label_size": 0.36, "shadow": dict(_SOFT_SHADOW)},
    "Speech Bubble": {"kind": "bubble", "w": 0.45, "h": 0.3, "fill": "#ffffff", "radius": 0.3, "tail": 0.25,
                      "stroke": "#111111", "stroke_w": 0.006, "shadow": dict(_SOFT_SHADOW)},
    "Banner Ribbon": {"kind": "banner", "w": 0.62, "h": 0.13, "fill": "#dc2626", "fill2": "#991b1b",
                      "angle": 90, "notch": 0.45, "label": "BREAKING", "label_color": "#ffffff",
                      "label_size": 0.5, "shadow": dict(_SHADOW)},
    "Pill Tag": {"kind": "rounded_rect", "w": 0.24, "h": 0.08, "fill": "#22c55e", "radius": 0.5,
                 "label": "LIVE", "label_color": "#ffffff", "label_size": 0.55},
    "Glass Panel": {"kind": "rounded_rect", "w": 0.6, "h": 0.36, "fill": "#ffffff", "fill2": "#93c5fd",
                    "angle": 120, "fill_opacity": 0.28, "stroke": "#ffffffcc", "stroke_w": 0.004,
                    "radius": 0.1},
    "Like Heart": {"kind": "heart", "w": 0.24, "h": 0.22, "fill": "#fb7185", "fill2": "#e11d48", "angle": 90,
                   "stroke": "#ffffff", "stroke_w": 0.007, "shadow": dict(_SHADOW)},
}

_EMOJI = [
    ("Fire", "\U0001F525"), ("Laughing", "\U0001F602"), ("Red Heart", "❤️"),
    ("Thumbs Up", "\U0001F44D"), ("Star Emoji", "⭐"), ("Party Popper", "\U0001F389"),
    ("Heart Eyes", "\U0001F60D"), ("Hundred", "\U0001F4AF"), ("Eyes", "\U0001F440"),
    ("Check Mark", "✅"), ("Cross Mark", "❌"), ("Lightning", "⚡"),
    ("Music Notes", "\U0001F3B5"), ("Pushpin", "\U0001F4CC"), ("Point Right", "\U0001F449"),
    ("Surprised", "\U0001F62E"), ("Raised Hands", "\U0001F64C"), ("Boom", "\U0001F4A5"),
    ("Rocket", "\U0001F680"), ("Mind Blown", "\U0001F92F"), ("Cool", "\U0001F60E"),
    ("Party Face", "\U0001F973"), ("Light Bulb", "\U0001F4A1"), ("Trophy", "\U0001F3C6"),
]

SHAPE_PRESETS: list[str] = list(PRESETS)
EMOJI_PRESETS: list[str] = []
for _name, _ch in _EMOJI:
    PRESETS[_name] = {"kind": "emoji", "char": _ch, "w": 0.22, "h": 0.22}
    EMOJI_PRESETS.append(_name)


def preset_category(name: str) -> str:
    return "Emoji" if name in EMOJI_PRESETS else "Shapes"


def new_shape_clip_fields(preset_name: str) -> dict:
    """Clip kwargs for a new shape clip from a preset (caller adds id/start)."""
    shape = copy.deepcopy(PRESETS[preset_name])
    dur = DEFAULT_EMOJI_DURATION if shape.get("kind") == "emoji" else DEFAULT_SHAPE_DURATION
    return {"asset_id": "", "in_point": 0.0, "out_point": dur, "shape": shape}


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def _poly(pts) -> QPainterPath:
    p = QPainterPath()
    p.addPolygon(QPolygonF([QPointF(x, y) for x, y in pts]))
    p.closeSubpath()
    return p


def _radial(n, w, h, inner=None, rot=-math.pi / 2):
    """Points of a regular polygon (inner=None) or star fitted to a w x h box."""
    pts = []
    cnt = n * 2 if inner is not None else n
    for i in range(cnt):
        a = rot + i * 2 * math.pi / cnt
        r = 1.0 if inner is None or i % 2 == 0 else inner
        pts.append((math.cos(a) * r, math.sin(a) * r))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    sx = w / max(x1 - x0, 1e-6)
    sy = h / max(y1 - y0, 1e-6)
    return [((x - x0) * sx, (y - y0) * sy) for x, y in pts]


def _path(s: dict, kind: str, w: float, h: float) -> QPainterPath:
    p = QPainterPath()
    mn = min(w, h)
    if kind in ("ellipse", "circle"):
        p.addEllipse(QRectF(0, 0, w, h))
    elif kind == "rounded_rect":
        r = max(0.0, min(float(s.get("radius", 0.2)), 0.5)) * mn
        p.addRoundedRect(QRectF(0, 0, w, h), r, r)
    elif kind == "line":
        r = max(0.0, min(float(s.get("radius", 0.5)), 0.5)) * mn
        p.addRoundedRect(QRectF(0, 0, w, h), r, r)
    elif kind == "triangle":
        p = _poly([(w / 2, 0), (w, h), (0, h)])
    elif kind == "star":
        p = _poly(_radial(max(int(s.get("points", 5)), 2), w, h, float(s.get("inner", 0.45))))
    elif kind == "polygon":
        p = _poly(_radial(max(int(s.get("sides", 6)), 3), w, h))
    elif kind == "heart":
        p.moveTo(w * 0.5, h * 0.25)
        p.cubicTo(w * 0.5, h * 0.12, w * 0.38, 0, w * 0.25, 0)
        p.cubicTo(w * 0.05, 0, 0, h * 0.2, 0, h * 0.32)
        p.cubicTo(0, h * 0.58, w * 0.25, h * 0.75, w * 0.5, h)
        p.cubicTo(w * 0.75, h * 0.75, w, h * 0.58, w, h * 0.32)
        p.cubicTo(w, h * 0.2, w * 0.95, 0, w * 0.75, 0)
        p.cubicTo(w * 0.62, 0, w * 0.5, h * 0.12, w * 0.5, h * 0.25)
        p.closeSubpath()
    elif kind == "arrow":
        hl = max(0.05, min(float(s.get("head", 0.4)), 1.0)) * w
        sh = max(0.05, min(float(s.get("shaft", 0.42)), 1.0)) * h
        y0, y1 = (h - sh) / 2, (h + sh) / 2
        p = _poly([(0, y0), (w - hl, y0), (w - hl, 0), (w, h / 2), (w - hl, h), (w - hl, y1), (0, y1)])
    elif kind == "bubble":
        bh = h * 0.78
        r = max(0.0, min(float(s.get("radius", 0.3)), 0.5)) * min(w, bh)
        body = QPainterPath()
        body.addRoundedRect(QRectF(0, 0, w, bh), r, r)
        tx = max(0.1, min(float(s.get("tail", 0.25)), 0.9)) * w
        tw = min(w * 0.14, bh * 0.5)
        tail = _poly([(tx - tw / 2, bh - 2), (tx + tw / 2, bh - 2), (tx - tw * 0.9, h)])
        p = body.united(tail)
    elif kind == "banner":
        n = max(0.0, min(float(s.get("notch", 0.45)), 2.0)) * h
        p = _poly([(0, 0), (w, 0), (w - n, h / 2), (w, h), (0, h), (n, h / 2)])
    else:  # rect
        p.addRect(QRectF(0, 0, w, h))
    return p


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _color(c, default="#ffffff") -> QColor:
    if not c:
        return QColor(default)
    c = str(c)
    if c.startswith("#") and len(c) == 9:  # #rrggbbaa
        q = QColor(c[:7])
        q.setAlpha(int(c[7:9], 16))
        return q
    q = QColor(c)
    return q if q.isValid() else QColor(default)


def _alpha_scaled(c: QColor, k: float) -> QColor:
    c = QColor(c)
    c.setAlphaF(max(0.0, min(1.0, c.alphaF() * k)))
    return c


def _fill_brush(s: dict, rect: QRectF) -> QBrush | None:
    if not s.get("fill"):
        return None
    k = float(s.get("fill_opacity", 1.0))
    c1 = _alpha_scaled(_color(s.get("fill")), k)
    if not s.get("fill2"):
        return QBrush(c1)
    c2 = _alpha_scaled(_color(s.get("fill2")), k)
    a = math.radians(float(s.get("angle", 90)))
    dx, dy = math.cos(a), math.sin(a)
    half = (abs(dx) * rect.width() + abs(dy) * rect.height()) / 2
    cx, cy = rect.center().x(), rect.center().y()
    g = QLinearGradient(QPointF(cx - dx * half, cy - dy * half), QPointF(cx + dx * half, cy + dy * half))
    g.setColorAt(0, c1)
    g.setColorAt(1, c2)
    return QBrush(g)


def _new_image(w: int, h: int) -> QImage:
    img = QImage(max(w, 1), max(h, 1), QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    return img


def _to_array(img: QImage) -> np.ndarray:
    """View of a 32-bit QImage as (h, w, 4) uint8 copy (BGRA byte order on little-endian)."""
    h, w, bpl = img.height(), img.width(), img.bytesPerLine()
    buf = np.frombuffer(img.constBits(), np.uint8, count=bpl * h)
    return buf.reshape(h, bpl)[:, : w * 4].reshape(h, w, 4).copy()


def _from_array(arr: np.ndarray) -> QImage:
    h, w, _ = arr.shape
    arr = np.ascontiguousarray(arr)
    return QImage(arr.data, w, h, w * 4, QImage.Format.Format_ARGB32_Premultiplied).copy()


def _box_blur(a: np.ndarray, r: int, passes: int = 3) -> np.ndarray:
    """Separable box blur (3 passes ~ gaussian) of a float 2-D array."""
    if r < 1:
        return a
    for _ in range(passes):
        for axis in (0, 1):
            pad = [(0, 0), (0, 0)]
            pad[axis] = (r + 1, r)
            c = np.cumsum(np.pad(a, pad, mode="constant"), axis=axis)
            n = a.shape[axis]
            hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
            lo = np.take(c, np.arange(0, n), axis=axis)
            a = (hi - lo) / (2 * r + 1)
    return a


def _shadow_image(body: QImage, color: QColor, blur_px: float) -> QImage:
    arr = _to_array(body)
    alpha = arr[:, :, 3].astype(np.float32) / 255.0
    alpha = _box_blur(alpha, int(round(blur_px / 2)))
    alpha *= color.alphaF()
    out = np.empty(arr.shape, np.uint8)
    out[:, :, 0] = np.clip(alpha * color.blue(), 0, 255)
    out[:, :, 1] = np.clip(alpha * color.green(), 0, 255)
    out[:, :, 2] = np.clip(alpha * color.red(), 0, 255)
    out[:, :, 3] = np.clip(alpha * 255, 0, 255)
    return _from_array(out)


def _draw_label(p: QPainter, s: dict, rect: QRectF, h_px: float) -> None:
    text = s.get("label")
    if not text:
        return
    f = QFont(s.get("label_font", "Segoe UI"))
    f.setBold(True)
    f.setPixelSize(max(int(float(s.get("label_size", 0.45)) * h_px), 1))
    p.setFont(f)
    p.setPen(_color(s.get("label_color"), "#ffffff"))
    p.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), str(text))


def _draw_emoji(p: QPainter, s: dict, rect: QRectF) -> None:
    f = QFont(EMOJI_FONT)
    f.setPixelSize(max(int(min(rect.width(), rect.height()) * 0.8), 1))
    f.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    p.setFont(f)
    p.setPen(QColor("#ffffff"))
    p.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), str(s.get("char", "⭐")))


def render_shape_dict(s: dict, canvas_h: int) -> QImage:
    kind = s.get("kind", "rect")
    H = max(float(canvas_h), 2.0)
    w = max(float(s.get("w", 0.5)) * H, 2.0)
    h = max(float(s.get("h", 0.3)) * H, 2.0)
    stroke_c = s.get("stroke") or ""
    sw = float(s.get("stroke_w", 0.0)) * H if stroke_c else 0.0
    if kind == "emoji":
        sw = 0.0
    sh = s.get("shadow") or {}
    dx = float(sh.get("dx", 0.0)) * H if sh else 0.0
    dy = float(sh.get("dy", 0.0)) * H if sh else 0.0
    blur = max(float(sh.get("blur", 0.0)) * H, 0.0) if sh else 0.0

    # margins: stroke half-width (miter joins can poke further: allow full width) + shadow extent
    m_stroke = sw + 1.0
    spread = blur * 1.6 + 1.0 if sh else 0.0
    ml = math.ceil(max(m_stroke, spread - dx if sh else 0.0))
    mr = math.ceil(max(m_stroke, spread + dx if sh else 0.0))
    mt = math.ceil(max(m_stroke, spread - dy if sh else 0.0))
    mb = math.ceil(max(m_stroke, spread + dy if sh else 0.0))
    W = int(math.ceil(w)) + ml + mr
    Ht = int(math.ceil(h)) + mt + mb

    body = _new_image(W, Ht)
    p = QPainter(body)
    p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing
                     | QPainter.RenderHint.SmoothPixmapTransform)
    p.translate(ml, mt)
    rect = QRectF(0, 0, w, h)
    if kind == "emoji":
        _draw_emoji(p, s, rect)
    else:
        path = _path(s, kind, w, h)
        brush = _fill_brush(s, rect)
        if brush is not None:
            p.fillPath(path, brush)
        if sw > 0:
            pen = QPen(_color(stroke_c), sw)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.strokePath(path, pen)
        label_rect = QRectF(0, 0, w, h * 0.78) if kind == "bubble" else rect
        if kind == "arrow":
            label_rect = QRectF(0, 0, w * (1 - float(s.get("head", 0.4)) * 0.5), h)
        _draw_label(p, s, label_rect, h)
    p.end()

    if not sh or (blur <= 0 and dx == 0 and dy == 0):
        return body
    shadow = _shadow_image(body, _color(sh.get("color"), "#00000099"), blur)
    out = _new_image(W, Ht)
    p = QPainter(out)
    p.drawImage(QPointF(dx, dy), shadow)
    p.drawImage(0, 0, body)
    p.end()
    return out


def render_shape(clip, canvas_h: int) -> QImage:
    return render_shape_dict(clip.shape or {"kind": "rect"}, canvas_h)


def thumbnail(preset, size: int = 96, background: QColor | None = None) -> QImage:
    """Gallery thumbnail: preset name or shape dict, fitted into a size x size square."""
    s = PRESETS[preset] if isinstance(preset, str) else preset
    big = max(float(s.get("w", 0.5)), float(s.get("h", 0.3)), 1e-3)
    img = render_shape_dict(s, max(int(size * 0.8 / big), 2))
    out = _new_image(size, size)
    if background is not None:
        out.fill(background)
    p = QPainter(out)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    k = min(size / img.width(), size / img.height(), 1.0)
    tw, th = img.width() * k, img.height() * k
    p.drawImage(QRectF((size - tw) / 2, (size - th) / 2, tw, th), img)
    p.end()
    return out
