"""Transitions between neighbouring clips on the same track.

Model: a clip B with `transition = d > 0` overlaps the clip A before it on its track by d seconds
(B.start == A.end - d). During the overlap B is drawn on top of A, and the transition kind decides how
B appears (and, for fade-through-black, how A disappears). Audio always crossfades over d.

Preview (Compositor, audio_mix) and export both use these helpers, so they always agree.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from .model import Clip, Track

KINDS: dict[str, str] = {
    "dissolve": "Dissolve",
    "fadeblack": "Fade through black",
    "slideleft": "Slide in from right",
    "slideright": "Slide in from left",
    "slideup": "Slide in from bottom",
    "slidedown": "Slide in from top",
    # ---- render-engine kinds (frame_fx) ----
    "flashwhite": "Flash to white",
    "wipeleft": "Wipe left",
    "wiperight": "Wipe right",
    "wipeup": "Wipe up",
    "wipedown": "Wipe down",
    "wipediag": "Wipe diagonal (from top left)",
    "wipediag2": "Wipe diagonal (from top right)",
    "circleopen": "Iris open",
    "circleclose": "Iris close",
    "diamond": "Diamond iris",
    "heart": "Heart iris",
    "star": "Star iris",
    "clock": "Clock wipe",
    "barnh": "Barn doors (horizontal)",
    "barnv": "Barn doors (vertical)",
    "blinds": "Venetian blinds",
    "checker": "Checkerboard",
    "blocks": "Random blocks",
    "pushleft": "Push left",
    "pushright": "Push right",
    "pushup": "Push up",
    "pushdown": "Push down",
    "zoomin": "Zoom in",
    "zoomout": "Zoom out",
    "zoomthrough": "Zoom through",
    "spin": "Spin",
    "whipleft": "Whip pan left",
    "whipright": "Whip pan right",
    "blurdissolve": "Blur dissolve",
    "glitch": "Glitch",
    "pixelate": "Pixelate",
    "lightleak": "Light leak",
    "ripple": "Ripple",
    "lumafade": "Luma fade",
}
CATEGORIES: dict[str, list[str]] = {
    "Basic": ["dissolve", "fadeblack", "flashwhite"],
    "Wipes": ["wipeleft", "wiperight", "wipeup", "wipedown", "wipediag", "wipediag2"],
    "Shapes": ["circleopen", "circleclose", "diamond", "heart", "star", "clock", "barnh", "barnv",
               "blinds", "checker", "blocks"],
    "Motion": ["slideleft", "slideright", "slideup", "slidedown", "pushleft", "pushright", "pushup",
               "pushdown", "zoomin", "zoomout", "zoomthrough", "spin", "whipleft", "whipright"],
    "Effects": ["blurdissolve", "glitch", "pixelate", "lightleak", "ripple", "lumafade"],
}
ICONS: dict[str, str] = {
    "dissolve": "◐", "fadeblack": "●", "flashwhite": "✦",
    "slideleft": "⇠", "slideright": "⇢", "slideup": "⇡", "slidedown": "⇣",
    "wipeleft": "◧", "wiperight": "◨", "wipeup": "⬓", "wipedown": "⬒", "wipediag": "◩", "wipediag2": "◪",
    "circleopen": "◎", "circleclose": "◉", "diamond": "◆", "heart": "♥", "star": "★", "clock": "◷",
    "barnh": "◫", "barnv": "⊟", "blinds": "☰", "checker": "▦", "blocks": "▩",
    "pushleft": "↞", "pushright": "↠", "pushup": "↟", "pushdown": "↡",
    "zoomin": "⊕", "zoomout": "⊖", "zoomthrough": "⊛", "spin": "↻", "whipleft": "⇇", "whipright": "⇉",
    "blurdissolve": "◌", "glitch": "▚", "pixelate": "▞", "lightleak": "☀", "ripple": "≈", "lumafade": "◑",
}
DEFAULT_KIND = "dissolve"
DEFAULT_DURATION = 0.5
EPS = 1e-3


LEGACY_KINDS = ("dissolve", "fadeblack", "slideleft", "slideright", "slideup", "slidedown")


class FrameFX:
    """Per-frame modifiers a transition applies to one clip (incoming or outgoing).

    scale/rotation/dx/dy/alpha change placement; mask(h, w) -> float32 coverage 0..1 multiplies alpha
    (in the clip's own pixel frame); pixel(rgba) -> rgba changes pixels (blur, glitch, flash...)."""

    def __init__(self, scale=1.0, rotation=0.0, dx=0.0, dy=0.0, alpha=1.0, mask=None, pixel=None):
        self.scale, self.rotation, self.dx, self.dy, self.alpha = scale, rotation, dx, dy, alpha
        self.mask, self.pixel = mask, pixel


def frame_fx(track: Track, clip: Clip, t: float, w: float, h: float) -> "FrameFX | None":
    """Extra effects for the newer (render-engine) transition kinds. None for legacy kinds.

    Pure function of (track layout, t, w, h): preview and render export produce identical frames.
    The incoming clip B (progress p) and the outgoing clip A (progress q) each get a FrameFX; a clip
    that is both (A->B->C with overlaps) gets the two composed."""
    fx_in = fx_out = None
    inc = incoming(track, clip)
    if inc and inc[0] not in LEGACY_KINDS and inc[0] in KINDS:
        kind, d = inc
        p = (t - clip.start) / d
        if p < 1.0:
            fx_in = _incoming_fx(kind, _clamp(p), t, w, h)
    out = outgoing(track, clip)
    if out and out[0] not in LEGACY_KINDS and out[0] in KINDS:
        kind, d = out
        q = (t - (clip.end - d)) / d
        if q > 0.0:
            fx_out = _outgoing_fx(kind, _clamp(q), t, w, h)
    if fx_in is None:
        return fx_out
    if fx_out is None:
        return fx_in
    return _compose(fx_in, fx_out)


def _compose(a: FrameFX, b: FrameFX) -> FrameFX:
    if a.mask and b.mask:
        ma, mb = a.mask, b.mask
        mask = lambda hh, ww: ma(hh, ww) * mb(hh, ww)  # noqa: E731
    else:
        mask = a.mask or b.mask
    if a.pixel and b.pixel:
        pa, pb = a.pixel, b.pixel
        pixel = lambda rgba: pb(pa(rgba))  # noqa: E731
    else:
        pixel = a.pixel or b.pixel
    return FrameFX(a.scale * b.scale, a.rotation + b.rotation, a.dx + b.dx, a.dy + b.dy,
                   a.alpha * b.alpha, mask, pixel)


# ---------------------------------------------------------------- render-engine kinds
def _ease(p: float) -> float:
    """Smootherstep: zero velocity and acceleration at both ends."""
    p = _clamp(p)
    return p * p * p * (p * (6 * p - 15) + 10)


def _bump(p: float) -> float:
    """0 -> 1 -> 0 over the transition, smooth."""
    return math.sin(math.pi * _clamp(p))


@lru_cache(maxsize=16)
def _xy(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre coordinates in 0..1, shapes (1, w) and (h, 1)."""
    x = ((np.arange(w, dtype=np.float32) + 0.5) / w)[None, :]
    y = ((np.arange(h, dtype=np.float32) + 0.5) / h)[:, None]
    return x, y


def _norm(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    lo, hi = float(v.min()), float(v.max())
    v = (v - lo) / max(hi - lo, 1e-6)
    v.setflags(write=False)
    return v


def _polar(h: int, w: int):
    """Aspect-correct centred coordinates (x right, y up), radius and angle."""
    x, y = _xy(h, w)
    ax = (x - 0.5) * (w / max(h, 1))
    ay = 0.5 - y
    ax, ay = np.broadcast_arrays(ax, ay)
    return ax, ay, np.hypot(ax, ay), np.arctan2(ay, ax)


@lru_cache(maxsize=64)
def _vmap(kind: str, h: int, w: int) -> np.ndarray:
    """Static reveal-order map in 0..1: a pixel shows B once the eased progress passes its value."""
    x, y = _xy(h, w)
    x, y = np.broadcast_arrays(x, y)
    if kind == "wipeleft":
        return _norm(1 - x)
    if kind == "wiperight":
        return _norm(x)
    if kind == "wipeup":
        return _norm(1 - y)
    if kind == "wipedown":
        return _norm(y)
    if kind == "wipediag":
        return _norm(x + y)
    if kind == "wipediag2":
        return _norm((1 - x) + y)
    if kind == "barnh":
        return _norm(np.abs(x - 0.5))
    if kind == "barnv":
        return _norm(np.abs(y - 0.5))
    if kind == "blinds":
        n = 10
        return _norm(y * n - np.floor(y * n))
    if kind == "checker":
        n = 8
        cols, rows = np.floor(x * n), np.floor(y * n)
        parity = (cols + rows) % 2
        frac = x * n - cols
        return _norm(parity * 0.5 + frac * 0.5)
    if kind == "blocks":
        ny = 9
        nx = max(int(round(ny * w / max(h, 1))), 1)
        rnd = np.random.default_rng(20240611).random((ny, nx)).astype(np.float32)
        ri = np.minimum((np.arange(h) * ny) // max(h, 1), ny - 1)
        ci = np.minimum((np.arange(w) * nx) // max(w, 1), nx - 1)
        return _norm(rnd[ri[:, None], ci[None, :]])
    ax, ay, r, th = _polar(h, w)
    if kind == "circleopen":
        return _norm(r)
    if kind == "circleclose":
        return _norm(-r)
    if kind == "diamond":
        return _norm(np.abs(ax) + np.abs(ay))
    if kind == "clock":
        ang = (np.pi / 2 - th) % (2 * np.pi)  # clockwise from 12 o'clock
        return _norm(ang)
    if kind == "star":
        k = np.abs(np.cos(2.5 * (th - np.pi / 2)))  # 5 points, one straight up
        return _norm(r / (0.45 + 0.55 * k ** 4))
    if kind == "heart":
        # polar heart about a point a little above centre (point down); +offset keeps it star-shaped
        oy = ay - 0.08
        rr, tt = np.hypot(ax, oy), np.arctan2(oy, ax)
        s, c = np.sin(tt), np.cos(tt)
        f = 2 - 2 * s + s * np.sqrt(np.abs(c)) / (s + 1.4) + 0.6
        return _norm(rr / f)
    raise KeyError(kind)


_EDGE = {"blinds": 0.12, "checker": 0.06, "blocks": 0.03, "clock": 0.02}


def _mask_fx(kind: str, e: float) -> FrameFX:
    edge = _EDGE.get(kind, 0.06)
    thr = e * (1 + edge)
    inv = 1.0 / edge

    def mask(hh, ww):
        v = _vmap(kind, hh, ww)
        return np.clip((thr - v) * inv, 0.0, 1.0)

    return FrameFX(mask=mask)


def _lighten(k: float):
    """Blend toward white by k (0..1), via a lookup table."""
    lut = (np.arange(256, dtype=np.float32) * (1 - k) + 255.0 * k + 0.5).astype(np.uint8)

    def pixel(rgba):
        out = lut[rgba]
        out[..., 3] = rgba[..., 3]
        return out
    return pixel


def _box(a: np.ndarray, r: int, axis: int) -> np.ndarray:
    """Box blur (radius r) along one axis of a uint16 array whose values are <= 255, edges clamped.
    The running sum wraps modulo 2**16, but window sums (<= 255 * (2r + 1)) never do, so the
    difference is exact as long as r < 128."""
    if r < 1:
        return a
    r = min(r, 127)
    n = a.shape[axis]
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    cs = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=np.uint16)
    hi = [slice(None)] * a.ndim
    lo = [slice(None)] * a.ndim
    hi[axis] = slice(2 * r + 1, 2 * r + 1 + n)
    lo[axis] = slice(0, n)
    d = cs[tuple(hi)] - cs[tuple(lo)]
    return d // np.uint16(2 * r + 1)


def _blur(radius_frac: float, horizontal_only: bool = False):
    """Box blur with radius as a fraction of the frame width. The blur runs on a copy downsampled
    by ~r/2 (r/4 for the 1-D whip blur) and is upsampled again with a short horizontal smoothing
    pass, which keeps every radius at a few ms. Radii under 2 px are skipped (invisible)."""
    def pixel(rgba):
        hh, ww = rgba.shape[:2]
        r = int(radius_frac * ww)
        if r < 2:
            return rgba
        f = max(1, min(r // 4, 8)) if horizontal_only else max(2, min(r // 2, 8))
        fy = 1 if horizontal_only else f
        h2, w2 = hh // fy * fy, ww // f * f
        if h2 < 1 or w2 < 1:
            return rgba
        s = rgba[:h2, :w2]
        if f > 1 or fy > 1:
            acc = np.zeros((h2 // fy, w2 // f, 4), np.uint16)
            for oy in range(fy):  # strided adds: much faster than reshape().sum()
                for ox in range(f):
                    acc += s[oy::fy, ox::f]
            s = acc // np.uint16(f * fy)
        else:
            s = s.astype(np.uint16)
        rr = max(r // f, 1)
        s = _box(s, rr, 1)
        if not horizontal_only:
            s = _box(s, rr, 0)
        if fy > 1:
            s = np.repeat(s, fy, axis=0)
        if f > 1:
            s = _box(np.repeat(s, f, axis=1), f // 2, 1)  # hide the upsampling steps
        s = s.astype(np.uint8)
        if h2 == hh and w2 == ww:
            return s
        out = np.empty_like(rgba)
        out[:h2, :w2] = s
        out[h2:, :w2] = s[-1:]
        out[:, w2:] = out[:, w2 - 1:w2]
        return out
    return pixel


def _pixelate(block_frac: float):
    def pixel(rgba):
        hh, ww = rgba.shape[:2]
        b = int(block_frac * ww)
        if b < 2:
            return rgba
        ys = np.minimum((np.arange(hh) // b) * b + b // 2, hh - 1)
        xs = np.minimum((np.arange(ww) // b) * b + b // 2, ww - 1)
        return rgba[ys][:, xs]
    return pixel


def _glitch(p: float, role: int):
    """RGB split plus horizontal slice offsets; changes in 24 steps over the transition."""
    s = _bump(p)
    step = int(p * 24)

    def pixel(rgba):
        if s < 0.02:
            return rgba
        hh, ww = rgba.shape[:2]
        rng = np.random.default_rng(1000 + step * 2 + role)
        out = rgba.copy()
        for _ in range(6):
            y0 = int(rng.integers(0, max(hh - 2, 1)))
            y1 = min(hh, y0 + int(rng.integers(max(hh // 40, 1), max(hh // 8, 2))))
            off = int(rng.integers(-1, 2) * s * ww * 0.08 * rng.random())
            if off:
                out[y0:y1] = np.roll(out[y0:y1], off, axis=1)
        sh = int(s * ww * 0.015) + 1
        res = out.copy()
        res[:, sh:, 0] = out[:, :-sh, 0]
        res[:, :-sh, 2] = out[:, sh:, 2]
        return res
    return pixel


def _glitch_mask(p: float):
    """B arrives band by band (bands reshuffle every few steps)."""
    e = _ease(p)
    step = int(p * 8)

    def mask(hh, ww):
        n = 16
        thr = np.random.default_rng(77 + step).random(n).astype(np.float32) * 0.9
        band = np.minimum((np.arange(hh) * n) // max(hh, 1), n - 1)
        cov = np.clip((e * 1.1 - thr[band]) / 0.1, 0.0, 1.0).astype(np.float32)
        return np.broadcast_to(cov[:, None], (hh, ww))
    return mask


def _ripple(p: float):
    amp = _bump(p) * 0.04
    phase = 2 * math.pi * p * 2

    def pixel(rgba):
        hh, ww = rgba.shape[:2]
        a = amp * ww
        if a < 0.5:
            return rgba
        y = np.arange(hh, dtype=np.float32)
        off = np.rint(a * np.sin(2 * np.pi * 3 * y / hh + phase)).astype(np.int32)
        xs = np.clip(np.arange(ww, dtype=np.int32)[None, :] + off[:, None], 0, ww - 1)
        return rgba[np.arange(hh)[:, None], xs]
    return pixel


def _light_leak(p: float):
    k = _bump(p)
    cx = -0.2 + 1.4 * p
    color = np.array([255.0, 150.0, 60.0], np.float32)

    def pixel(rgba):
        if k < 0.01:
            return rgba
        hh, ww = rgba.shape[:2]
        x, y = _xy(hh, ww)
        gx = np.exp(-((x[0] - cx) / 0.35) ** 2) * (1.2 * k)
        gy = np.exp(-((y[:, 0] - 0.3) / 0.6) ** 2)
        out = rgba.copy()
        for c in range(3):  # separable glow, one outer product per channel
            g = (gy[:, None] * (gx * color[c])[None, :]).astype(np.uint16)
            g += rgba[..., c]
            np.minimum(g, 255, out=g)
            out[..., c] = g
        return out
    return pixel


def _luma_mask(p: float):
    e = _ease(p)
    edge = 0.1

    def pixel(rgba):
        hh, ww = rgba.shape[:2]
        _, y = _xy(hh, ww)
        rgb = rgba[..., :3].astype(np.float32)
        luma = (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114) * (1 / 255.0)
        v = 0.7 * (1 - luma) + 0.3 * y  # brightest areas first, top-down tie-break
        lo, hi = float(v.min()), float(v.max())
        v = (v - lo) / max(hi - lo, 1e-6)
        cov = np.clip((e * (1 + edge) - v) / edge, 0.0, 1.0)
        out = rgba.copy()
        out[..., 3] = (rgba[..., 3] * cov + 0.5).astype(np.uint8)
        return out
    return pixel


_MASK_KINDS = frozenset(("wipeleft", "wiperight", "wipeup", "wipedown", "wipediag", "wipediag2",
                         "circleopen", "circleclose", "diamond", "heart", "star", "clock", "barnh",
                         "barnv", "blinds", "checker", "blocks"))
_PUSH = {"pushleft": (-1, 0), "pushright": (1, 0), "pushup": (0, -1), "pushdown": (0, 1),
         "whipleft": (-1, 0), "whipright": (1, 0)}


def _incoming_fx(kind: str, p: float, t: float, w: float, h: float) -> FrameFX | None:
    e = _ease(p)
    if kind in _MASK_KINDS:
        return _mask_fx(kind, e)
    if kind in _PUSH:
        ux, uy = _PUSH[kind]
        fx = FrameFX(dx=-ux * w * (1 - e), dy=-uy * h * (1 - e))
        if kind.startswith("whip"):
            fx.pixel = _blur(0.12 * _whip_speed(p), horizontal_only=True)
        return fx
    if kind == "flashwhite":
        if p < 0.5:
            return FrameFX(alpha=0.0)
        return FrameFX(pixel=_lighten(1 - _ease(2 * p - 1)))
    if kind == "zoomin":
        return FrameFX(scale=0.25 + 0.75 * e, alpha=_ease(2 * p))
    if kind == "zoomout":
        return FrameFX(scale=2.5 - 1.5 * e, alpha=_ease(1.25 * p))
    if kind == "zoomthrough":
        return FrameFX(scale=1.8 - 0.8 * e, alpha=_ease(1.25 * p))
    if kind == "spin":
        return FrameFX(scale=max(e, 1e-3), rotation=-360.0 * (1 - e))
    if kind == "blurdissolve":
        return FrameFX(alpha=e, pixel=_blur(0.04 * (1 - e)))
    if kind == "glitch":
        return FrameFX(mask=_glitch_mask(p), pixel=_glitch(p, 1))
    if kind == "pixelate":
        return FrameFX(alpha=_ease((p - 0.3) / 0.4), pixel=_pixelate(0.06 * _bump(p)))
    if kind == "lightleak":
        return FrameFX(alpha=e, pixel=_light_leak(p))
    if kind == "ripple":
        return FrameFX(alpha=e, pixel=_ripple(p))
    if kind == "lumafade":
        return FrameFX(pixel=_luma_mask(p))
    return None


def _outgoing_fx(kind: str, q: float, t: float, w: float, h: float) -> FrameFX | None:
    e = _ease(q)
    if kind in _PUSH:
        ux, uy = _PUSH[kind]
        fx = FrameFX(dx=ux * w * e, dy=uy * h * e)
        if kind.startswith("whip"):
            fx.pixel = _blur(0.12 * _whip_speed(q), horizontal_only=True)
        return fx
    if kind == "flashwhite":
        return FrameFX(pixel=_lighten(_ease(2 * q)))
    if kind == "zoomthrough":
        return FrameFX(scale=1 + 1.5 * e, alpha=1 - _ease(1.25 * q - 0.25))
    if kind == "blurdissolve":
        return FrameFX(pixel=_blur(0.04 * e))
    if kind == "glitch":
        return FrameFX(pixel=_glitch(q, 0))
    if kind == "pixelate":
        return FrameFX(pixel=_pixelate(0.06 * _bump(q)))
    if kind == "lightleak":
        return FrameFX(pixel=_light_leak(q))
    if kind == "ripple":
        return FrameFX(pixel=_ripple(q))
    return None


def _whip_speed(p: float) -> float:
    """Normalised speed (0..1) of the smootherstep move: 30p^2(1-p)^2 / 1.875."""
    p = _clamp(p)
    return 16.0 * p * p * (1 - p) * (1 - p)


def needs_render(kind: str) -> bool:
    """True if this transition kind can only be exported by the render engine."""
    return kind not in LEGACY_KINDS


def _clamp(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def smoothstep(p: float) -> float:
    return p * p * (3 - 2 * p)


def prev_clip(track: Track, clip: Clip) -> Clip | None:
    """The clip that `clip` transitions from (its end == clip.start + clip.transition)."""
    joint = clip.start + clip.transition
    for c in track.clips:
        if c is not clip and c.start < clip.start + EPS and abs(c.end - joint) < EPS:
            return c
    return None


def neighbour_before(track: Track, clip: Clip) -> Clip | None:
    """A clip that ends exactly where `clip` starts (a cut a transition could be added to)."""
    for c in track.clips:
        if c is not clip and abs(c.end - clip.start) < EPS:
            return c
    return None


def next_clip(track: Track, clip: Clip) -> Clip | None:
    """The clip that transitions in from `clip`, if any."""
    for c in track.clips:
        if c is not clip and c.transition > 0 and abs(clip.end - (c.start + c.transition)) < EPS \
                and c.start > clip.start:
            return c
    return None


def incoming(track: Track, clip: Clip) -> tuple[str, float] | None:
    if clip.transition <= 0 or prev_clip(track, clip) is None:
        return None
    return clip.transition_kind, clip.transition


def outgoing(track: Track, clip: Clip) -> tuple[str, float] | None:
    n = next_clip(track, clip)
    return (n.transition_kind, n.transition) if n else None


def max_duration(prev: Clip, clip: Clip) -> float:
    return max(min(prev.duration, clip.duration) - 0.05, 0.0)


# ---------------------------------------------------------------- preview / mixing
def video_alpha(track: Track, clip: Clip, t: float) -> float:
    a = clip.fade_gain(t)
    inc = incoming(track, clip)
    if inc:
        kind, d = inc
        p = _clamp((t - clip.start) / d)
        if kind == "dissolve":
            a *= p
        elif kind == "fadeblack":
            a *= _clamp(2 * p - 1)
    out = outgoing(track, clip)
    if out and out[0] == "fadeblack":
        d = out[1]
        q = _clamp((t - (clip.end - d)) / d)
        a *= _clamp(1 - 2 * q)
    return a


def video_offset(track: Track, clip: Clip, t: float, w: float, h: float) -> tuple[float, float]:
    """Pixel offset of the clip at time t (slides)."""
    inc = incoming(track, clip)
    if not inc or not inc[0].startswith("slide"):
        return 0.0, 0.0
    kind, d = inc
    r = 1 - smoothstep(_clamp((t - clip.start) / d))
    return {"slideleft": (w * r, 0.0), "slideright": (-w * r, 0.0),
            "slideup": (0.0, h * r), "slidedown": (0.0, -h * r)}[kind]


def audio_fades(track: Track, clip: Clip) -> tuple[float, float]:
    """Effective (fade_in, fade_out) seconds for audio, including crossfades."""
    fi, fo = clip.fade_in, clip.fade_out
    inc = incoming(track, clip)
    if inc:
        fi = max(fi, inc[1])
    out = outgoing(track, clip)
    if out:
        fo = max(fo, out[1])
    return fi, fo


# ---------------------------------------------------------------- export (FFmpeg)
def ffmpeg_alpha_fades(track: Track, clip: Clip) -> list[str]:
    """`fade` filters (clip-local time, alpha only) for fades + transitions. They multiply."""
    dur = clip.duration
    out = []
    if clip.fade_in > 0:
        out.append(f"fade=t=in:st=0:d={clip.fade_in:.4f}:alpha=1")
    if clip.fade_out > 0:
        out.append(f"fade=t=out:st={max(dur - clip.fade_out, 0):.4f}:d={clip.fade_out:.4f}:alpha=1")
    inc = incoming(track, clip)
    if inc:
        kind, d = inc
        if kind == "dissolve":
            out.append(f"fade=t=in:st=0:d={d:.4f}:alpha=1")
        elif kind == "fadeblack":
            out.append(f"fade=t=in:st={d / 2:.4f}:d={d / 2:.4f}:alpha=1")
    og = outgoing(track, clip)
    if og and og[0] == "fadeblack":
        d = og[1]
        out.append(f"fade=t=out:st={max(dur - d, 0):.4f}:d={d / 2:.4f}:alpha=1")
    return out


def ffmpeg_overlay_xy(track: Track, clip: Clip, x0: float, y0: float, W: int, H: int) -> tuple[str, str]:
    """Overlay x/y expressions (timeline time `t`), animated for slide transitions."""
    inc = incoming(track, clip)
    if not inc or not inc[0].startswith("slide"):
        return f"{x0:.2f}", f"{y0:.2f}"
    kind, d = inc
    p = f"clip((t-{clip.start:.4f})/{d:.4f},0,1)"
    r = f"(1-({p})*({p})*(3-2*({p})))"
    if kind == "slideleft":
        return f"{x0:.2f}+{W}*{r}", f"{y0:.2f}"
    if kind == "slideright":
        return f"{x0:.2f}-{W}*{r}", f"{y0:.2f}"
    if kind == "slideup":
        return f"{x0:.2f}", f"{y0:.2f}+{H}*{r}"
    return f"{x0:.2f}", f"{y0:.2f}-{H}*{r}"
