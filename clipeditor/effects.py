"""Video effects / filters + chroma key, applied to RGBA frames.

CONTRACT (used by clipeditor.render.Compositor for preview AND render-engine export):
    REGISTRY: dict[name, EffectSpec]  (label, category, params with defaults/ranges)
    active(clip) -> bool              # clip has enabled effects or chroma key
    apply(rgba, clip, t_local, canvas_h) -> rgba
        rgba: uint8 (h, w, 4) C-contiguous; may be modified in place or a new array returned.
        t_local: seconds since clip start (for animated effects like shake/glitch).

Effect entries on a clip: {"type": name, "params": {..}, "enabled": bool}.
EffectSpec: {"label", "category", "params": {pname: {"label", "default", "min", "max"}}}
    plus optional "moves_alpha": True for geometric effects (mirror, kaleidoscope, shake, zoom
    pulse, heartbeat, wobble) which transform all four channels so a text/shape clip's alpha
    moves with its pixels. Every other effect changes RGB only and preserves alpha.

Chroma key (clip.chroma = {"color", "similarity", "smooth", "spill"}) runs first and is the
only stage that changes alpha in place.

Pixel sizes (blur radius, pixel blocks, grain, shifts) are fractions of canvas_h, so a look
is the same in a 360p preview and a 1080p export. Time-based effects seed their randomness
from int(t_local * FPS) so the same t always gives the same frame.

UI helpers: new_effect(name), categories(), preview_card(name, rgba_sample), run_effect(...),
new_chroma().  All numpy, vectorised; no scipy / OpenCV.
"""
from __future__ import annotations

import colorsys
import math
from functools import lru_cache

import numpy as np

FPS = 30.0  # time quantum for seeded randomness (frame index = floor(t * FPS))
F32 = np.float32


# =============================================================== registry helpers
def _p(label, default, lo, hi):
    return {"label": label, "default": default, "min": lo, "max": hi}


def _I(default=1.0, hi=1.0, label="Intensity"):
    return _p(label, default, 0.0, hi)


REGISTRY: dict = {}
_IMPL: dict = {}  # name -> (fn, kind) kind: "u8" (rgb uint8 in/out), "f" (rgb float), "rgba"


def _reg(name, label, category, params, kind="f", moves_alpha=False, preview_t=0.4):
    def deco(fn):
        spec = {"label": label, "category": category, "params": params}
        if moves_alpha:
            spec["moves_alpha"] = True
        REGISTRY[name] = spec
        _IMPL[name] = (fn, kind, preview_t)
        return fn
    return deco


# =============================================================== numeric helpers
_LUMA = np.array([0.299, 0.587, 0.114], F32)


def _to_f(rgb_u8):
    return rgb_u8.astype(F32) * F32(1 / 255)


def _to_u8(f):
    f = f * F32(255)
    np.clip(f, 0, 255, out=f)
    f += F32(0.5)
    return f.astype(np.uint8)


def _luma(f):
    return f @ _LUMA


def _mix(a, b, k):
    if k >= 0.999:
        return b
    if k <= 0.001:
        return a
    return a + (b - a) * F32(k)


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / max(e1 - e0, 1e-6), 0, 1)
    return t * t * (3 - 2 * t)


def _rng(t, salt, hold=1):
    frame = int(math.floor(t * FPS + 1e-4)) // max(hold, 1)
    return np.random.default_rng([frame & 0x7FFFFFFF, salt])


def _box1d(a, r, axis):
    r = int(r)
    if r < 1:
        return a
    n = a.shape[axis]
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=F32)
    hi = [slice(None)] * a.ndim
    lo = [slice(None)] * a.ndim
    hi[axis] = slice(2 * r + 1, 2 * r + 1 + n)
    lo[axis] = slice(0, n)
    out = c[tuple(hi)]
    out -= c[tuple(lo)]
    out *= F32(1 / (2 * r + 1))
    return out


def _down(f, s):
    """Area downscale by an integer factor s (crops the remainder)."""
    if s <= 1:
        return f
    h, w = f.shape[:2]
    h2, w2 = max(h // s, 1), max(w // s, 1)
    return _sum_down(f, s, h2, w2) * F32(1 / (s * s))


def _sum_down(f, s, h2, w2):
    """Sum s x s blocks of f (whose size is >= h2*s, w2*s) via strided slice adds."""
    rows = f[0:h2 * s:s, :w2 * s].astype(F32)
    for i in range(1, s):
        rows += f[i:h2 * s:s, :w2 * s]
    out = rows[:, 0::s].copy()
    for j in range(1, s):
        out += rows[:, j::s]
    return out


@lru_cache(maxsize=64)
def _lin_idx(n_src, coords_key):
    xs = np.frombuffer(coords_key, dtype=F32)
    x = np.clip(xs, 0, n_src - 1)
    i0 = np.floor(x).astype(np.intp)
    i1 = np.minimum(i0 + 1, n_src - 1)
    w = (x - i0).astype(F32)
    return i0, i1, w


def _sep_sample(f, ys, xs):
    """Bilinear sample float image f (h, w, c) at row coords ys (H,) and col coords xs (W,)."""
    h, w = f.shape[:2]
    y0, y1, wy = _lin_idx(h, np.ascontiguousarray(ys, F32).tobytes())
    x0, x1, wx = _lin_idx(w, np.ascontiguousarray(xs, F32).tobytes())
    extra = (None,) * (f.ndim - 2)
    if f.dtype != F32:
        f = f.astype(F32)
    a = np.take(f, y0, axis=0)
    d = np.take(f, y1, axis=0)
    d -= a
    d *= wy[(slice(None), None) + extra]
    a += d
    b = np.take(a, x0, axis=1)
    d = np.take(a, x1, axis=1)
    d -= b
    d *= wx[(None, slice(None)) + extra]
    b += d
    return b


def _resize(f, nh, nw):
    h, w = f.shape[:2]
    if (h, w) == (nh, nw):
        return f
    ys = (np.arange(nh, dtype=F32) + 0.5) * F32(h / nh) - 0.5
    xs = (np.arange(nw, dtype=F32) + 0.5) * F32(w / nw) - 0.5
    return _sep_sample(f, ys, xs)


def _blur(f, r, pre=None):
    """Approximate gaussian blur with radius r px (~2 sigma): downscale, 2x box, upscale.

    pre(small) is applied after the downscale (cheap per-pixel ops at low resolution)."""
    if r < 0.5:
        return pre(f) if pre else f
    h, w = f.shape[:2]
    s = max(1, int(r / 2.5))
    s = min(s, max(1, min(h, w) // 4))
    small = _down(f, s)
    if pre is not None:
        small = pre(small)
    rb = max(1, int(round(r / s * 0.6)))
    small = _box1d(_box1d(small, rb, 0), rb, 1)
    if s > 1:  # second pass is cheap at low resolution and makes it gaussian-like
        small = _box1d(_box1d(small, rb, 0), rb, 1)
    return _resize(small, h, w) if s > 1 else small


@lru_cache(maxsize=16)
def _radial(h, w, cx=0.5, cy=0.5):
    """Distance from centre, normalised so the frame corners are ~1 (elliptical)."""
    y = (np.arange(h, dtype=F32) + 0.5) / h - cy
    x = (np.arange(w, dtype=F32) + 0.5) / w - cx
    r = np.sqrt((x[None, :] * 2) ** 2 + (y[:, None] * 2) ** 2) * F32(1 / math.sqrt(2))
    r.flags.writeable = False
    return r


@lru_cache(maxsize=32)
def _vig(h, w, e0, e1):
    """Cached smoothstep(e0, e1, radial distance) map (h, w)."""
    v = _smoothstep(e0, e1, _radial(h, w)).astype(F32)
    v.flags.writeable = False
    return v


@lru_cache(maxsize=32)
def _sat_mat(sat_q):
    P = np.outer(_LUMA, np.ones(3, F32))
    return (P + (sat_q / 1000) * (np.eye(3, dtype=F32) - P)).astype(F32)


def _sat(f, sat):
    """Saturation via one 3x3 matmul (luma preserved)."""
    h, w = f.shape[:2]
    return (np.ascontiguousarray(f).reshape(-1, 3) @ _sat_mat(int(round(sat * 1000)))).reshape(h, w, 3)


def _hue_rgb(hue_deg, s, v):
    return np.array(colorsys.hsv_to_rgb((hue_deg % 360) / 360.0, s, v), F32)


def _px(frac, canvas_h):
    return frac * max(int(canvas_h), 1)


# =============================================================== Filters (colour looks)
_SEPIA = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]], F32).T


@lru_cache(maxsize=64)
def _tone_lut(contrast, bright, gamma, fade, gain):
    x = np.arange(256, dtype=np.float64) / 255
    y = np.clip((x - 0.5) * contrast + 0.5 + bright, 0, 1) ** (1.0 / gamma)
    y = fade + (gain - fade) * y
    return np.clip(y * 255 + 0.5, 0, 255).astype(np.uint8)


@lru_cache(maxsize=64)
def _look_mats(items):
    spec = dict(items)
    luma = _LUMA.astype(np.float64)
    P = np.outer(luma, np.ones(3))  # row-vector convention: g = f @ M
    M = np.eye(3)
    if spec.get("sepia"):
        M = _SEPIA.astype(np.float64)
    elif spec.get("mono"):
        M = P.copy()
    sat = spec.get("sat", 1.0)
    if sat != 1.0 and not spec.get("mono"):
        M = M @ (P + sat * (np.eye(3) - P))
    temp, tint = spec.get("temp", 0.0), spec.get("tint", 0.0)
    if temp or tint:
        M = M @ np.diag([1 + 0.12 * temp, 1 - 0.08 * tint, 1 - 0.12 * temp])
    sh = np.array(spec.get("shadow") or (0, 0, 0), np.float64)
    hi = np.array(spec.get("high") or (0, 0, 0), np.float64)
    # split tone in 0..255 space: g += sh*(1-L)^2 + hi*L^2 = c + a*L255 + b*L255^2
    split = None
    if sh.any() or hi.any():
        split = (np.stack([-2 * sh, (sh + hi) / 255]).astype(F32), (255 * sh).astype(F32),
                 (M @ luma).astype(F32))
    return M.astype(F32), split


def _look(rgb_u8, spec, k):
    M, split = _look_mats(tuple(sorted(spec.items())))
    lut = _tone_lut(spec.get("contrast", 1.0), spec.get("bright", 0.0), spec.get("gamma", 1.0),
                    spec.get("fade", 0.0), spec.get("gain", 1.0))
    h, w = rgb_u8.shape[:2]
    identity = split is None and np.allclose(M, np.eye(3))
    if identity:
        u8 = np.ascontiguousarray(rgb_u8)
    else:
        f = rgb_u8.astype(F32).reshape(-1, 3)
        g = f @ M
        if split is not None:
            B, c, lv = split
            L = f @ lv
            Q = np.empty((L.shape[0], 2), F32)
            Q[:, 0] = L
            np.multiply(L, L, out=Q[:, 1])
            g += Q @ B
            g += c
        g += F32(0.5)
        np.clip(g, 0, 255, out=g)
        u8 = g.astype(np.uint8).reshape(h, w, 3)
    out = lut[u8]
    if k < 0.999:
        o = rgb_u8.astype(F32)
        o += (out - o) * F32(k)
        o += F32(0.5)
        out = o.astype(np.uint8)
    return out


_FILTERS = {
    "vivid": ("Vivid", dict(sat=1.45, contrast=1.12)),
    "cinematic": ("Cinematic (teal-orange)", dict(sat=1.05, contrast=1.1, fade=0.02,
                                                  shadow=(-0.09, 0.02, 0.09), high=(0.09, 0.02, -0.07))),
    "warm": ("Warm", dict(temp=0.6, sat=1.1)),
    "cool": ("Cool", dict(temp=-0.6, contrast=1.03)),
    "vintage": ("Vintage", dict(sat=0.75, fade=0.08, gain=0.95, temp=0.3, contrast=0.9,
                                shadow=(0.02, 0.0, 0.06), high=(0.06, 0.04, -0.04))),
    "sepia": ("Sepia", dict(sepia=True, contrast=1.05)),
    "bw": ("B&W", dict(mono=True, contrast=1.05)),
    "noir": ("Noir (high contrast)", dict(mono=True, contrast=1.6, gamma=0.85, bright=-0.03)),
    "faded": ("Faded film", dict(sat=0.7, fade=0.15, contrast=0.85, gain=0.92)),
    "moody": ("Moody", dict(sat=0.7, contrast=1.15, bright=-0.06, gamma=0.9, temp=-0.2,
                            shadow=(-0.02, 0.0, 0.05))),
    "pastel": ("Pastel", dict(sat=0.6, fade=0.18, bright=0.05, contrast=0.8, high=(0.04, 0.0, 0.05))),
    "cyberpunk": ("Cyberpunk", dict(sat=1.3, contrast=1.15, tint=0.5,
                                    shadow=(0.05, -0.04, 0.18), high=(0.14, -0.04, 0.08))),
    "golden_hour": ("Golden hour", dict(temp=0.9, sat=1.15, bright=0.03, contrast=1.05,
                                        high=(0.10, 0.05, -0.06))),
    "matte": ("Matte", dict(fade=0.12, contrast=0.95, gain=0.9, sat=0.9)),
}


def _make_filter(spec):
    def fn(rgb_u8, p, t, ch):
        return _look(rgb_u8, spec, p["intensity"])
    return fn


for _name, (_label, _spec) in _FILTERS.items():
    _reg(_name, _label, "Filters", {"intensity": _I(1.0)}, kind="u8")(_make_filter(_spec))


# =============================================================== Blur & Sharp
@_reg("blur", "Gaussian blur", "Blur & Sharp", {"intensity": _I(0.3)})
def _fx_blur(f, p, t, ch):
    return _blur(f, _px(0.06 * p["intensity"], ch))


@_reg("motion_blur", "Motion blur", "Blur & Sharp",
      {"intensity": _I(0.5), "angle": _p("Angle", 0.0, -180.0, 180.0)})
def _fx_motion_blur(f, p, t, ch):
    length = _px(0.08 * p["intensity"], ch)
    if length < 1:
        return f
    h, w = f.shape[:2]
    s = max(1, int(length // 8))
    src = _down(f, s)
    L = length / s
    a = math.radians(p["angle"])
    n = 9
    offs = [(round(math.sin(a) * (k / (n - 1) - 0.5) * L), round(math.cos(a) * (k / (n - 1) - 0.5) * L))
            for k in range(n)]
    m = int(math.ceil(L / 2)) + 1
    pad = np.pad(src, ((m, m), (m, m), (0, 0)), mode="edge")
    sh, sw = src.shape[:2]
    acc = np.zeros_like(src)
    for dy, dx in offs:
        acc += pad[m + dy:m + dy + sh, m + dx:m + dx + sw]
    acc *= F32(1 / n)
    return _resize(acc, h, w) if s > 1 else acc


@_reg("zoom_blur", "Radial / zoom blur", "Blur & Sharp", {"intensity": _I(0.4)})
def _fx_zoom_blur(f, p, t, ch):
    k = p["intensity"]
    if k <= 0.001:
        return f
    h, w = f.shape[:2]
    s = max(1, min(4, min(h, w) // 16))
    src = _down(f, s)
    sh, sw = src.shape[:2]
    cy, cx = (sh - 1) / 2, (sw - 1) / 2
    ys0 = np.arange(sh, dtype=F32) - cy
    xs0 = np.arange(sw, dtype=F32) - cx
    n = 8
    acc = np.zeros_like(src)
    for i in range(n):
        sc = 1 - 0.18 * k * i / (n - 1)
        acc += _sep_sample(src, ys0 * sc + cy, xs0 * sc + cx)
    acc *= F32(1 / n)
    blur = _resize(acc, h, w)
    m = np.clip(_radial(h, w) * 2.2, 0, 1)[..., None]
    return f + (blur - f) * m


@_reg("sharpen", "Sharpen", "Blur & Sharp", {"amount": _I(0.6, 2.0, "Amount")})
def _fx_sharpen(f, p, t, ch):
    r = max(1.0, _px(0.004, ch))
    b = _blur(f, r)
    return f + (f - b) * F32(p["amount"] * 1.5)


@_reg("tilt_shift", "Tilt-shift", "Blur & Sharp",
      {"intensity": _I(0.6), "focus": _p("Focus Y", 0.5, 0.0, 1.0), "width": _p("Focus width", 0.25, 0.02, 0.8)})
def _fx_tilt_shift(f, p, t, ch):
    h = f.shape[0]
    b = _blur(f, _px(0.04 * p["intensity"], ch))
    y = (np.arange(h, dtype=F32) + 0.5) / h
    m = np.clip((np.abs(y - p["focus"]) - p["width"] / 2) / 0.2, 0, 1)[:, None, None]
    b -= f
    b *= m
    b += f
    return _sat(b, 1 + 0.3 * p["intensity"])


@_reg("soft_focus", "Soft focus (dreamy)", "Blur & Sharp", {"intensity": _I(0.5)})
def _fx_soft_focus(f, p, t, ch):
    k = p["intensity"]
    b = _blur(f, _px(0.03, ch))
    lighten = np.maximum(f, b)
    return f + (lighten * F32(0.6) + b * F32(0.4) - f) * F32(k * 0.8)


# =============================================================== Stylize
@_reg("vignette", "Vignette", "Stylize", {"intensity": _I(0.6), "size": _p("Size", 0.5, 0.0, 1.0)})
def _fx_vignette(f, p, t, ch):
    h, w = f.shape[:2]
    inner = 0.15 + 0.6 * p["size"]
    inner = round(inner, 3)
    v = _vig(h, w, inner, inner + 0.75)
    return f * (1 - v * F32(p["intensity"]))[..., None]


@_reg("grain", "Film grain", "Stylize", {"intensity": _I(0.4), "size": _p("Grain size", 0.3, 0.0, 1.0)})
def _fx_grain(f, p, t, ch):
    h, w = f.shape[:2]
    g = max(1.0, 1 + p["size"] * _px(0.006, ch))
    nh, nw = max(int(h / g), 1), max(int(w / g), 1)
    noise = _rng(t, 11).random((nh, nw), dtype=F32) - F32(0.5)
    if (nh, nw) != (h, w):
        noise = _resize(noise[..., None], h, w)[..., 0]
    L = _luma(f)
    amp = F32(0.35 * p["intensity"]) * (F32(0.4) + F32(2.4) * L * (1 - L))  # strongest in mid-tones
    return f + (noise * amp)[..., None]


def _block_mean(f, b):
    h, w = f.shape[:2]
    nh, nw = -(-h // b), -(-w // b)
    pad = np.pad(f, ((0, nh * b - h), (0, nw * b - w), (0, 0)), mode="edge")
    return _sum_down(pad, b, nh, nw) * F32(1 / (b * b))


def _block_up(small, b, h, w):
    return np.repeat(np.repeat(small, b, axis=0), b, axis=1)[:h, :w]


@_reg("pixelate", "Pixelate", "Stylize", {"intensity": _I(0.3)})
def _fx_pixelate(f, p, t, ch):
    b = max(2, int(round(_px(0.06 * p["intensity"], ch))))
    h, w = f.shape[:2]
    return _block_up(_block_mean(f, b), b, h, w)


@lru_cache(maxsize=32)
def _poster_lut(levels):
    x = np.arange(256) / 255
    return (np.round(x * (levels - 1)) / (levels - 1) * 255 + 0.5).astype(np.uint8)


@_reg("posterize", "Posterize", "Stylize", {"intensity": _I(0.5)}, kind="u8")
def _fx_posterize(u8, p, t, ch):
    levels = int(round(2 + (1 - p["intensity"]) * 10))
    return _poster_lut(levels)[u8]


def _grad(L):
    Lp = np.pad(L, 1, mode="edge")
    gx = Lp[1:-1, 2:] - Lp[1:-1, :-2]
    gy = Lp[2:, 1:-1] - Lp[:-2, 1:-1]
    return gx, gy


@_reg("sketch", "Sketch / edges", "Stylize", {"intensity": _I(1.0)})
def _fx_sketch(f, p, t, ch):
    L = _luma(f)
    if ch > 500:
        L = _blur(L[..., None], _px(0.0015, ch))[..., 0]
    gx, gy = _grad(L)
    e = np.clip(np.sqrt(gx * gx + gy * gy) * F32(3.0), 0, 1)
    paper = (1 - e) * F32(0.97) * (F32(0.85) + F32(0.15) * L)
    return _mix(f, np.repeat(paper[..., None], 3, axis=2), p["intensity"])


@_reg("emboss", "Emboss", "Stylize", {"intensity": _I(1.0)})
def _fx_emboss(f, p, t, ch):
    L = _luma(f)
    Lp = np.pad(L, 1, mode="edge")
    e = F32(0.5) + (Lp[2:, 2:] - Lp[:-2, :-2]) * F32(2.0)
    e = np.repeat(e[..., None], 3, axis=2)
    return _mix(f, e, p["intensity"])


@lru_cache(maxsize=16)
def _cell_dist(h, w, b):
    yy = (np.arange(h, dtype=F32) % b) - (b - 1) / 2
    xx = (np.arange(w, dtype=F32) % b) - (b - 1) / 2
    d = np.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2)
    d.flags.writeable = False
    return d


@_reg("halftone", "Halftone dots", "Stylize", {"intensity": _I(1.0), "size": _p("Dot size", 0.3, 0.0, 1.0)})
def _fx_halftone(f, p, t, ch):
    h, w = f.shape[:2]
    b = max(3, int(round(_px(0.008 + 0.03 * p["size"], ch))))
    cells = _block_mean(f, b)
    Lc = _luma(cells)
    rad = np.sqrt(np.clip(1 - Lc, 0, 1)) * F32(b * 0.72)
    rad_full = _block_up(rad[..., None], b, h, w)[..., 0]
    dot = np.clip(rad_full - _cell_dist(h, w, b) + F32(0.5), 0, 1)[..., None]
    ink = _block_up(cells, b, h, w) * F32(0.45)
    paper = np.array([0.97, 0.95, 0.89], F32)
    out = paper + (ink - paper) * dot
    return _mix(f, out, p["intensity"])


@_reg("mirror", "Mirror", "Stylize",
      {"amount": _I(1.0, label="Amount"), "mode": _p("Mode (0 L|R, 1 T|B, 2 quad)", 0, 0, 2)},
      kind="rgba", moves_alpha=True)
def _fx_mirror(img, p, t, ch):
    h, w = img.shape[:2]
    mode = int(round(p["mode"]))
    out = img.copy()
    if mode in (0, 2):
        out[:, w - w // 2:] = out[:, :w // 2][:, ::-1]
    if mode in (1, 2):
        out[h - h // 2:] = out[:h // 2][::-1]
    if p["amount"] < 0.999:
        out = _to_u8(_mix(_to_f(img), _to_f(out), p["amount"]))
    return out


@lru_cache(maxsize=16)
def _kaleido_idx(h, w, segments, rot_q):
    rot = rot_q / 1000.0
    y = np.arange(h, dtype=F32) - (h - 1) / 2
    x = np.arange(w, dtype=F32) - (w - 1) / 2
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X * X + Y * Y)
    a = np.arctan2(Y, X) + F32(rot)
    seg = F32(2 * math.pi / segments)
    a = np.mod(a, seg)
    a = np.where(a > seg / 2, seg - a, a) - F32(rot)
    sx = np.clip(np.round(np.cos(a) * r + (w - 1) / 2), 0, w - 1).astype(np.intp)
    sy = np.clip(np.round(np.sin(a) * r + (h - 1) / 2), 0, h - 1).astype(np.intp)
    idx = (sy * w + sx).ravel()
    idx.flags.writeable = False
    return idx


@_reg("kaleidoscope", "Kaleidoscope", "Stylize",
      {"amount": _I(1.0, label="Amount"), "segments": _p("Segments", 6, 3, 16),
       "spin": _p("Spin (rev/s)", 0.0, -1.0, 1.0)}, kind="rgba", moves_alpha=True)
def _fx_kaleidoscope(img, p, t, ch):
    h, w = img.shape[:2]
    rot = (p["spin"] * t * 2 * math.pi) % (2 * math.pi)
    rot_q = int(round(rot * 1000 / 4)) * 4  # quantise for the index cache
    idx = _kaleido_idx(h, w, int(round(p["segments"])), rot_q)
    out = img.reshape(-1, 4)[idx].reshape(h, w, 4)
    if p["amount"] < 0.999:
        out = _to_u8(_mix(_to_f(img), _to_f(out), p["amount"]))
    return out


@_reg("invert", "Invert", "Stylize", {"amount": _I(1.0, label="Amount")})
def _fx_invert(f, p, t, ch):
    return _mix(f, 1 - f, p["amount"])


@_reg("duotone", "Duotone", "Stylize",
      {"intensity": _I(1.0), "hue_dark": _p("Shadow hue", 250.0, 0.0, 360.0),
       "hue_light": _p("Highlight hue", 40.0, 0.0, 360.0)})
def _fx_duotone(f, p, t, ch):
    dark = _hue_rgb(p["hue_dark"], 0.85, 0.22)
    light = _hue_rgb(p["hue_light"], 0.55, 1.0)
    L = _smoothstep(0.03, 0.97, _luma(f))[..., None]
    return _mix(f, dark + (light - dark) * L, p["intensity"])


@_reg("glow", "Glow / bloom", "Stylize", {"intensity": _I(0.6), "threshold": _p("Threshold", 0.5, 0.0, 0.95)})
def _fx_glow(f, p, t, ch):
    thr = p["threshold"]
    def bright(x):
        x = (x - F32(thr)) * F32(1.6 * p["intensity"] / (1 - thr))
        return np.clip(x, 0, 1, out=x)
    halo = _blur(f, _px(0.05, ch), pre=bright)
    np.subtract(1, halo, out=halo)
    np.clip(halo, 0, 1, out=halo)
    halo *= 1 - f
    return np.subtract(1, halo, out=halo)


@lru_cache(maxsize=1)
def _thermal_lut():
    stops = np.array([[0, 0, 0.1], [0.3, 0, 0.6], [0.9, 0.1, 0.3], [1, 0.6, 0], [1, 1, 0.3], [1, 1, 1]], F32)
    x = np.linspace(0, len(stops) - 1, 256)
    i = np.minimum(x.astype(int), len(stops) - 2)
    fr = (x - i)[:, None].astype(F32)
    return stops[i] + (stops[i + 1] - stops[i]) * fr


@_reg("thermal", "Thermal", "Stylize", {"intensity": _I(1.0)})
def _fx_thermal(f, p, t, ch):
    idx = np.clip(_luma(f) * 255 + 0.5, 0, 255).astype(np.uint8)
    return _mix(f, _thermal_lut()[idx], p["intensity"])


# =============================================================== Retro & Glitch
def _shift_x(a, dx):
    """Shift columns right by integer dx with edge clamp."""
    dx = int(dx)
    if dx == 0:
        return a
    out = np.empty_like(a)
    if dx > 0:
        out[:, dx:] = a[:, :-dx]
        out[:, :dx] = a[:, :1]
    else:
        out[:, :dx] = a[:, -dx:]
        out[:, dx:] = a[:, -1:]
    return out


def _shift_y(a, dy):
    dy = int(dy)
    if dy == 0:
        return a
    out = np.empty_like(a)
    if dy > 0:
        out[dy:] = a[:-dy]
        out[:dy] = a[:1]
    else:
        out[:dy] = a[-dy:]
        out[dy:] = a[-1:]
    return out


@_reg("vhs", "VHS", "Retro & Glitch", {"intensity": _I(0.6)})
def _fx_vhs(f, p, t, ch):
    k = p["intensity"]
    h, w = f.shape[:2]
    rng = _rng(t, 21)
    yy = np.arange(h, dtype=F32)
    # per-row horizontal jitter + a rolling tracking band
    jit = rng.standard_normal(h).astype(F32) * F32(_px(0.0025 * k, ch))
    band_y = ((t * 0.35) % 1.3 - 0.15) * h
    band = np.exp(-((yy - F32(band_y)) / F32(max(h * 0.03, 1))) ** 2)
    off = np.round(jit + band * F32(_px(0.03 * k, ch))).astype(np.intp)
    cols = np.clip(np.arange(w)[None, :] - off[:, None], 0, w - 1)
    # luma: jittered, noisy, scanlined, lifted blacks
    Y = np.take(_luma(f), (cols + (np.arange(h) * w)[:, None]).ravel()).reshape(h, w)
    noise = rng.random((h, w), dtype=F32)
    noise -= F32(0.5)
    noise *= (F32(0.08 * k) + band[:, None] * F32(0.5 * k))
    Y += noise
    period = max(2.0, ch / 120)
    sl = (F32(1) - F32(0.12 * k) * (0.5 + 0.5 * np.cos(yy * F32(2 * math.pi / period)))) * F32(1 - 0.08 * k)
    Y *= sl[:, None]
    Y += F32(0.05 * k)
    # chroma: low horizontal resolution (block-averaged), shifted right, same jitter
    q = max(2, int(_px(0.012 * k, ch)))
    wq = -(-w // q)
    fp = np.pad(f, ((0, 0), (0, wq * q - w), (0, 0)), mode="edge")
    Cs = fp[:, 0::q].copy()
    for j in range(1, q):
        Cs += fp[:, j::q]
    Cs *= F32(0.85 / q)
    Cs -= (Cs @ _LUMA)[..., None]
    shift = int(round(_px(0.008 * k, ch)))
    cq = np.clip(cols - shift, 0, w - 1) // q + (np.arange(h) * wq)[:, None]
    C = np.take(Cs.reshape(-1, 3), cq.ravel(), axis=0).reshape(h, w, 3)
    C += Y[..., None]
    return C


@_reg("rgb_split", "RGB split / chromatic aberration", "Retro & Glitch",
      {"amount": _I(0.4, label="Amount"), "angle": _p("Angle", 0.0, -180.0, 180.0)})
def _fx_rgb_split(f, p, t, ch):
    d = _px(0.02 * p["amount"], ch)
    a = math.radians(p["angle"])
    dx, dy = int(round(math.cos(a) * d)), int(round(math.sin(a) * d))
    out = f.copy()
    out[..., 0] = _shift_y(_shift_x(f[..., 0], dx), dy)
    out[..., 2] = _shift_y(_shift_x(f[..., 2], -dx), -dy)
    return out


@_reg("scanlines", "Scanlines", "Retro & Glitch", {"intensity": _I(0.5)})
def _fx_scanlines(f, p, t, ch):
    h = f.shape[0]
    period = max(2.0, ch / 90)
    yy = np.arange(h, dtype=F32)
    wgt = F32(1) - F32(0.6 * p["intensity"]) * (0.5 + 0.5 * np.cos(yy * F32(2 * math.pi / period)))
    return f * wgt[:, None, None] * F32(1 + 0.15 * p["intensity"])


@_reg("glitch", "Glitch blocks", "Retro & Glitch", {"intensity": _I(0.5)})
def _fx_glitch(f, p, t, ch):
    k = p["intensity"]
    h, w = f.shape[:2]
    rng = _rng(t, 31, hold=2)
    out = f.copy()
    n = int(2 + 12 * k)
    for _ in range(n):
        bh = int(rng.integers(max(1, h // 60), max(2, int(h * (0.02 + 0.1 * k)))))
        y0 = int(rng.integers(0, max(1, h - bh)))
        sh = int(rng.normal(0, _px(0.06 * k, ch)))
        band = np.roll(f[y0:y0 + bh], sh, axis=1)
        r = rng.random()
        if r < 0.3:
            band = band[..., [2, 0, 1]]
        elif r < 0.45:
            band = 1 - band
        out[y0:y0 + bh] = band
    # a few displaced rectangles
    for _ in range(int(1 + 4 * k)):
        bw = int(rng.integers(max(2, w // 30), max(3, w // 5)))
        bh = int(rng.integers(max(2, h // 40), max(3, h // 8)))
        x0, y0 = int(rng.integers(0, max(1, w - bw))), int(rng.integers(0, max(1, h - bh)))
        x1 = int(np.clip(x0 + rng.normal(0, w * 0.1 * k), 0, w - bw))
        out[y0:y0 + bh, x1:x1 + bw] = f[y0:y0 + bh, x0:x0 + bw] * np.array([1.2, 0.8, 1.3], F32)
    sp = int(round(_px(0.008 * k, ch)))
    out[..., 0] = _shift_x(out[..., 0], sp)
    return out


@_reg("noise", "Noise", "Retro & Glitch", {"amount": _I(0.3, label="Amount")})
def _fx_noise(f, p, t, ch):
    noise = _rng(t, 41).random(f.shape, dtype=F32) - F32(0.5)
    return f + noise * F32(0.6 * p["amount"])


@_reg("old_film", "Old film (flicker + scratches)", "Retro & Glitch", {"intensity": _I(0.7)})
def _fx_old_film(f, p, t, ch):
    k = p["intensity"]
    h, w = f.shape[:2]
    rng = _rng(t, 51)
    g = _mix(f, f @ _SEPIA, 0.7 * k)
    g = g * F32(1 + (rng.random() - 0.5) * 0.25 * k)  # flicker
    # vertical scratches: light or dark lines
    sw = max(1, int(round(_px(0.003, ch))))
    for _ in range(int(rng.integers(0, 1 + int(4 * k)))):
        x = int(rng.integers(0, max(1, w - sw)))
        y0 = int(rng.integers(0, h // 2))
        y1 = int(rng.integers(h // 2, h))
        val = F32(0.95 if rng.random() < 0.6 else 0.15)
        g[y0:y1, x:x + sw] = g[y0:y1, x:x + sw] * F32(0.3) + val * F32(0.7)
    # dust specks
    ds = max(1, int(round(_px(0.006, ch))))
    for _ in range(int(rng.integers(0, 2 + int(10 * k)))):
        x, y = int(rng.integers(0, max(1, w - ds))), int(rng.integers(0, max(1, h - ds)))
        g[y:y + ds, x:x + ds] = F32(0.05)
    v = _vig(h, w, 0.45, 1.25)
    g = g * (1 - v * F32(0.7 * k))[..., None]
    return g + (rng.random((h, 1), dtype=F32) - F32(0.5))[..., None] * F32(0.03 * k)


@_reg("eight_bit", "8-bit", "Retro & Glitch", {"intensity": _I(0.5)})
def _fx_eight_bit(f, p, t, ch):
    k = p["intensity"]
    b = max(2, int(round(_px(0.008 + 0.03 * k, ch))))
    h, w = f.shape[:2]
    small = _block_mean(f, b)
    levels = max(2, int(round(6 - 3 * k)))
    small = np.round(np.clip(small * F32(1.1), 0, 1) * (levels - 1)) / F32(levels - 1)
    return _block_up(small, b, h, w)


@_reg("crt", "CRT screen", "Retro & Glitch", {"intensity": _I(0.6)})
def _fx_crt(f, p, t, ch):
    k = p["intensity"]
    h, w = f.shape[:2]
    sub = max(1, int(round(ch / 360)))
    phase = (np.arange(w) // sub) % 3
    mask = np.full((w, 3), 1 - 0.5 * k, F32)
    mask[np.arange(w), phase] = 1.0
    g = f * mask[None, :, :] * F32(1 + 0.35 * k)
    period = max(2.0, ch / 120)
    yy = np.arange(h, dtype=F32)
    g *= (F32(1) - F32(0.35 * k) * (0.5 + 0.5 * np.cos(yy * F32(2 * math.pi / period))))[:, None, None]
    v = _vig(h, w, 0.6, 1.2)
    return g * (1 - v * F32(0.6 * k))[..., None]


# =============================================================== Motion (time based)
def _zoom_shift(img, scale, dx=0.0, dy=0.0):
    """Scale about the centre (scale > 1 zooms in) and shift by (dx, dy) px; all 4 channels."""
    if abs(scale - 1) < 1e-5 and abs(dx) < 0.01 and abs(dy) < 0.01:
        return img
    h, w = img.shape[:2]
    cy, cx = (h - 1) / 2, (w - 1) / 2
    ys = (np.arange(h, dtype=np.float64) - (cy + dy)) / scale + cy
    xs = (np.arange(w, dtype=np.float64) - (cx + dx)) / scale + cx
    out = _sep_sample(img, ys.astype(F32), xs.astype(F32))
    out += F32(0.5)
    return out.astype(np.uint8)


def _value_noise(t, salt):
    i = math.floor(t)
    fr = t - i
    a = np.random.default_rng([i & 0x7FFFFFFF, salt]).uniform(-1, 1)
    b = np.random.default_rng([(i + 1) & 0x7FFFFFFF, salt]).uniform(-1, 1)
    s = fr * fr * (3 - 2 * fr)
    return a + (b - a) * s


@_reg("shake", "Camera shake", "Motion",
      {"intensity": _I(0.5), "speed": _p("Speed", 8.0, 0.5, 30.0)}, kind="rgba", moves_alpha=True)
def _fx_shake(img, p, t, ch):
    k = p["intensity"]
    # quantise to frames so preview and export agree exactly
    tq = math.floor(t * FPS + 1e-4) / FPS
    amp = _px(0.025 * k, ch)
    dx = _value_noise(tq * p["speed"], 61) * amp
    dy = _value_noise(tq * p["speed"], 62) * amp
    scale = 1 + 2.2 * amp / max(min(img.shape[:2]), 1)
    return _zoom_shift(img, scale, dx, dy)


@_reg("zoom_pulse", "Zoom pulse", "Motion",
      {"intensity": _I(0.5), "rate": _p("Rate (Hz)", 1.0, 0.1, 8.0)}, kind="rgba", moves_alpha=True,
      preview_t=0.5)
def _fx_zoom_pulse(img, p, t, ch):
    s = 1 + 0.15 * p["intensity"] * (0.5 - 0.5 * math.cos(2 * math.pi * p["rate"] * t))
    return _zoom_shift(img, s)


@_reg("strobe", "Strobe / flash", "Motion",
      {"intensity": _I(0.7), "rate": _p("Rate (Hz)", 4.0, 0.25, 15.0)}, preview_t=0.0)
def _fx_strobe(f, p, t, ch):
    phase = (t * p["rate"]) % 1.0
    env = math.exp(-phase * 7.0) * p["intensity"]
    return f + (1 - f) * F32(env)


def _bump(x, c, wdt):
    return math.exp(-((x - c) / wdt) ** 2)


@_reg("heartbeat", "Heartbeat", "Motion",
      {"intensity": _I(0.6), "bpm": _p("BPM", 72.0, 30.0, 180.0)}, kind="rgba", moves_alpha=True,
      preview_t=0.05)
def _fx_heartbeat(img, p, t, ch):
    ph = (t * p["bpm"] / 60.0) % 1.0
    env = _bump(ph, 0.05, 0.05) + 0.6 * _bump(ph, 0.25, 0.05)
    k = p["intensity"]
    out = _zoom_shift(img, 1 + 0.08 * k * env)
    if env > 0.02:
        f = out[..., :3].astype(F32)
        f *= np.array([1 + 0.25 * k * env, 1 - 0.05 * k * env, 1 - 0.05 * k * env], F32)
        np.clip(f, 0, 255, out=f)
        out = out.copy() if out is img else out
        out[..., :3] = f.astype(np.uint8)
    return out


@_reg("wobble", "Wobble (wavy)", "Motion",
      {"intensity": _I(0.5), "speed": _p("Speed", 1.0, 0.0, 5.0)}, kind="rgba", moves_alpha=True)
def _fx_wobble(img, p, t, ch):
    h, w = img.shape[:2]
    amp = _px(0.02 * p["intensity"], ch)
    yy = np.arange(h, dtype=F32)
    wl = max(h / 2.5, 4)
    off = np.round(np.sin(yy * F32(2 * math.pi / wl) + F32(t * p["speed"] * 2 * math.pi)) * F32(amp))
    cols = np.clip(np.arange(w)[None, :] - off.astype(np.intp)[:, None], 0, w - 1)
    return img[np.arange(h)[:, None], cols]


# =============================================================== shared helpers (library expansion)
@lru_cache(maxsize=16)
def _grid(h, w):
    """Pixel coords relative to the frame centre: Y (h, 1) and X (1, w), float32."""
    y = (np.arange(h, dtype=F32) - F32((h - 1) / 2))[:, None]
    x = (np.arange(w, dtype=F32) - F32((w - 1) / 2))[None, :]
    y.flags.writeable = False
    x.flags.writeable = False
    return y, x


def _reflect(c, n):
    if n <= 1:
        return np.zeros_like(c)
    period = F32(2 * (n - 1))
    c = np.mod(c, period)
    return np.minimum(c, period - c)


def _rgba32(img):
    return np.ascontiguousarray(img, dtype=np.uint8).view(np.uint32).reshape(-1)


_M8 = np.uint32(0x00FF00FF)
_R8 = np.uint32(0x00800080)
_H8 = np.uint32(0xFF00FF00)


def _lerp32(a, b, wq, iwq):
    """Per-byte lerp of packed RGBA uint32 pixels with 8-bit fixed-point weights (SWAR).

    Even and odd bytes are blended in two passes; a lane holds at most 255 * 256 so it never
    overflows into its neighbour. Weights wq + iwq == 256."""
    lo = a & _M8
    lo *= iwq
    t = b & _M8
    t *= wq
    lo += t
    lo += _R8
    lo >>= 8
    lo &= _M8
    hi = a >> 8
    hi &= _M8
    hi *= iwq
    np.right_shift(b, 8, out=t)
    t &= _M8
    t *= wq
    hi += t
    hi += _R8
    hi &= _H8
    lo |= hi
    return lo


def _warp(img, ys, xs, mirror=False):
    """Bilinear sample a uint8 RGBA frame at float pixel coords ys, xs (broadcast to h, w).

    All four channels move together; out-of-frame coords clamp (or mirror) so an opaque
    frame stays opaque. Pixels are gathered as packed uint32 and blended two bytes at a time
    with 8-bit weights, which is several times faster than float RGBA arithmetic."""
    h, w = img.shape[:2]
    ys, xs = np.broadcast_arrays(ys, xs)
    if mirror:
        xs, ys = _reflect(xs, w), _reflect(ys, h)
    x = np.clip(xs, 0, max(w - 1.001, 0))
    y = np.clip(ys, 0, max(h - 1.001, 0))
    xf, yf = np.floor(x), np.floor(y)
    wx = ((x - xf) * F32(256) + F32(0.5)).astype(np.uint32)
    wy = ((y - yf) * F32(256) + F32(0.5)).astype(np.uint32)
    i = yf.astype(np.intp)
    i *= w
    i += xf.astype(np.intp)
    src = _rgba32(img)
    dx, dy = (1 if w > 1 else 0), (w if h > 1 else 0)
    iwx = np.uint32(256) - wx
    top = _lerp32(src[i], src[i + dx], wx, iwx)
    i += dy
    bot = _lerp32(src[i], src[i + dx], wx, iwx)
    out = _lerp32(top, bot, wy, np.uint32(256) - wy)
    return out.view(np.uint8).reshape(out.shape + (4,))


def _nearest(img, ys, xs):
    h, w = img.shape[:2]
    ys, xs = np.broadcast_arrays(ys, xs)
    xi = np.clip(np.rint(xs), 0, w - 1).astype(np.intp)
    yi = np.clip(np.rint(ys), 0, h - 1).astype(np.intp)
    return _rgba32(img)[yi * w + xi].view(np.uint8).reshape(xi.shape + (4,))


def _affine(img, scale=1.0, angle=0.0, dx=0.0, dy=0.0, mirror=False):
    """Rotate by angle (rad), scale (>1 zooms in) about the centre, then shift (px); RGBA."""
    if abs(scale - 1) < 1e-5 and abs(angle) < 1e-5 and abs(dx) < 0.01 and abs(dy) < 0.01:
        return img
    h, w = img.shape[:2]
    Y, X = _grid(h, w)
    c, s = F32(math.cos(angle) / scale), F32(math.sin(angle) / scale)
    u = X - F32(dx)
    v = Y - F32(dy)
    xs = c * u + s * v + F32((w - 1) / 2)
    ys = c * v - s * u + F32((h - 1) / 2)
    return _warp(img, ys, xs, mirror)


def _mix_u8(orig, out, k):
    if k >= 0.999:
        return out
    o = orig.astype(F32)
    o += (out - o) * F32(k)
    o += F32(0.5)
    return o.astype(np.uint8)


def _mix_rgba(img, out, k):
    if k >= 0.999:
        return out
    return _to_u8(_mix(_to_f(img), _to_f(out), k))


def _screen(f, add):
    """Screen blend: 1 - (1 - f) * (1 - add)."""
    return 1 - (1 - f) * (1 - np.clip(add, 0, 1))


def _overlay(base, blend):
    lo = 2 * base * blend
    hi = 1 - 2 * (1 - base) * (1 - blend)
    return np.where(base < 0.5, lo, hi)


def _softlight(base, blend):
    """Pegtop soft light: (1 - 2b) a^2 + 2 b a."""
    return (1 - 2 * blend) * base * base + 2 * blend * base


def _hsv_hue(hue01):
    """Vectorised fully saturated colour for hue in 0..1 (any shape) -> (..., 3)."""
    k = (hue01[..., None] * 6 + np.array([0, 4, 2], F32)) % 6
    return np.clip(np.abs(k - 3) - 1, 0, 1).astype(F32)


def _edges(f, ch, soft=0.0015):
    L = _luma(f)
    if soft > 0:
        L = _blur(L[..., None], max(_px(soft, ch), 0.6))[..., 0]
    gx, gy = _grad(L)
    return np.sqrt(gx * gx + gy * gy)


def _tq(t):
    return math.floor(t * FPS + 1e-4) / FPS


# =============================================================== more Filters
_FILTERS_MORE = {
    "kodak_warm": ("Warm film (Kodak-style)", dict(temp=0.45, sat=1.12, contrast=1.08, fade=0.03,
                                                  gamma=1.03, shadow=(0.03, 0.0, -0.02), high=(0.08, 0.04, -0.05))),
    "fuji_green": ("Green film (Fuji-style)", dict(sat=1.05, contrast=1.06, tint=-0.45, fade=0.03, temp=-0.1,
                                                   shadow=(-0.03, 0.05, 0.03), high=(0.0, 0.03, 0.04))),
    "clarendon": ("Clarendon", dict(contrast=1.22, sat=1.3, bright=0.02,
                                    shadow=(-0.05, 0.01, 0.09), high=(0.05, 0.03, -0.02))),
    "juno": ("Juno", dict(sat=1.28, temp=0.3, contrast=1.1, tint=0.15, high=(0.08, 0.05, -0.04))),
    "lark": ("Lark", dict(bright=0.07, sat=0.85, contrast=0.94, temp=-0.25, gamma=1.1,
                          high=(0.0, 0.04, 0.05))),
    "polaroid": ("Polaroid", dict(fade=0.1, sat=0.8, temp=0.25, contrast=0.92, gain=0.96,
                                  shadow=(0.0, 0.04, 0.07), high=(0.07, 0.05, 0.0))),
    "tropical": ("Tropical", dict(sat=1.55, temp=0.2, tint=-0.35, contrast=1.1,
                                  shadow=(-0.04, 0.04, 0.07), high=(0.08, 0.06, -0.05))),
    "arctic": ("Arctic", dict(temp=-0.95, sat=0.7, bright=0.06, contrast=0.95,
                              shadow=(-0.02, 0.02, 0.09), high=(-0.03, 0.04, 0.10))),
    "desert": ("Desert", dict(temp=0.85, sat=0.85, fade=0.05, contrast=1.05,
                              shadow=(0.07, 0.02, -0.04), high=(0.10, 0.06, -0.05))),
    "neon_night": ("Neon night", dict(sat=1.6, contrast=1.25, bright=-0.06, tint=0.6,
                                      shadow=(0.07, -0.05, 0.18), high=(-0.06, 0.06, 0.12))),
    "seventies": ("70s", dict(sat=0.8, temp=0.55, fade=0.1, contrast=0.9, gain=0.94, tint=-0.1,
                              shadow=(0.07, 0.02, -0.05), high=(0.08, 0.06, -0.07))),
}

for _name, (_label, _spec) in _FILTERS_MORE.items():
    _reg(_name, _label, "Filters", {"intensity": _I(1.0)}, kind="u8")(_make_filter(_spec))


@_reg("bleach_bypass", "Bleach bypass", "Filters", {"intensity": _I(1.0)})
def _fx_bleach_bypass(f, p, t, ch):
    L = _luma(f)[..., None]
    out = _sat(_overlay(f, L), 0.5)
    out -= F32(0.5)
    out *= F32(1.1)
    out += F32(0.5)
    return _mix(f, out, p["intensity"])


@lru_cache(maxsize=1)
def _xpro_lut():
    x = np.arange(256, dtype=np.float64) / 255
    s_curve = lambda a: x - a * np.sin(2 * np.pi * x) / (2 * np.pi)  # noqa: E731
    r = s_curve(0.75) * 0.96 + 0.04
    g = s_curve(0.45) * 0.97 + 0.02
    b = 0.16 + 0.62 * x  # lifted blacks, crushed highs: blue shadows, yellow highlights
    lut = np.stack([r, g, b], axis=1)
    return np.clip(lut * 255 + 0.5, 0, 255).astype(np.uint8)


@_reg("cross_process", "Cross-process", "Filters", {"intensity": _I(1.0)}, kind="u8")
def _fx_cross_process(u8, p, t, ch):
    lut = _xpro_lut()
    out = np.empty_like(u8)
    for c in range(3):
        out[..., c] = lut[:, c][u8[..., c]]
    return _mix_u8(u8, out, p["intensity"])


# =============================================================== more Blur & Sharp
@lru_cache(maxsize=16)
def _disc_taps(rad):
    pts = {(0, 0)}
    for ring, n in ((0.5, 6), (1.0, 12)):
        for i in range(n):
            a = 2 * math.pi * i / n + ring
            pts.add((int(round(math.sin(a) * rad * ring)), int(round(math.cos(a) * rad * ring))))
    return tuple(sorted(pts))


@_reg("lens_blur", "Lens blur (bokeh)", "Blur & Sharp", {"intensity": _I(0.5)})
def _fx_lens_blur(f, p, t, ch):
    r = _px(0.05 * p["intensity"], ch)
    if r < 1:
        return f
    h, w = f.shape[:2]
    s = max(1, int(r / 3))
    src = _down(f, s)
    sh, sw = src.shape[:2]
    hl = np.clip((_luma(src) - F32(0.7)) * F32(3.0), 0, 1)[..., None] * src
    e = np.concatenate([src * src, hl], axis=2)  # gamma-2 average favours highlights
    R = max(1, int(round(r / s)))
    taps = _disc_taps(R)
    pad = np.pad(e, ((R, R), (R, R), (0, 0)), mode="edge")
    acc = np.zeros_like(e)
    for dy, dx in taps:
        acc += pad[R + dy:R + dy + sh, R + dx:R + dx + sw]
    acc *= F32(1 / len(taps))
    rb = max(1, R // 4)
    acc = _box1d(_box1d(acc, rb, 0), rb, 1)
    out = np.sqrt(np.maximum(acc[..., :3], 0)) + acc[..., 3:] * F32(0.8)
    return _resize(out, h, w) if s > 1 else out


@_reg("radial_focus", "Radial focus", "Blur & Sharp",
      {"intensity": _I(0.6), "size": _p("Focus size", 0.35, 0.0, 1.0)})
def _fx_radial_focus(f, p, t, ch):
    h, w = f.shape[:2]
    b = _blur(f, _px(0.04 * p["intensity"], ch))
    e0 = round(0.05 + 0.5 * p["size"], 3)
    m = _vig(h, w, e0, round(e0 + 0.7, 3))[..., None]
    b -= f
    b *= m
    b += f
    return b


@lru_cache(maxsize=16)
def _ellipse(h, w, rx_q, ry_q):
    """Normalised elliptical distance (1 on the edge), radii in fractions of h (x1000)."""
    Y, X = _grid(h, w)
    d = np.sqrt((X / F32(rx_q / 1000 * h)) ** 2 + (Y / F32(ry_q / 1000 * h)) ** 2)
    d.flags.writeable = False
    return d


@_reg("background_blur", "Background blur (centre sharp)", "Blur & Sharp",
      {"intensity": _I(0.7), "size": _p("Subject size", 0.45, 0.05, 1.0)})
def _fx_background_blur(f, p, t, ch):
    h, w = f.shape[:2]
    k = p["intensity"]
    rx = int(round((0.12 + 0.35 * p["size"]) * 1000))
    d = _ellipse(h, w, rx, int(rx * 1.35))
    m = _smoothstep(0.85, 1.15, d)[..., None]
    b = _blur(f, _px(0.06 * k, ch))
    b = _sat(b, 1 - 0.15 * k) * F32(1 - 0.06 * k)
    b -= f
    b *= m
    b += f
    return b


@_reg("smooth_skin", "Surface blur / smooth skin", "Blur & Sharp", {"intensity": _I(0.6)})
def _fx_smooth_skin(f, p, t, ch):
    k = p["intensity"]
    b = _blur(f, _px(0.012, ch))
    d = np.abs(_luma(f) - _luma(b))
    wgt = np.exp(-(d * F32(1 / 0.05)) ** 2) * F32(k)  # edges (big local difference) stay sharp
    out = f + (b - f) * wgt[..., None]
    out += F32(0.02 * k) * (1 - out)  # a little glow
    return out


@_reg("clarity", "Clarity (local contrast)", "Blur & Sharp", {"amount": _I(0.6, 2.0, "Amount")})
def _fx_clarity(f, p, t, ch):
    a = p["amount"]
    L = _luma(f)
    Lb = _blur(L[..., None], _px(0.045, ch))[..., 0]
    d = (L - Lb) * (F32(4) * L * (1 - L)) * F32(1.6 * a)
    return _sat(f + d[..., None], 1 + 0.08 * a)


@_reg("defocus_pulse", "Defocus pulse", "Blur & Sharp",
      {"intensity": _I(0.6), "rate": _p("Rate (Hz)", 1.0, 0.1, 6.0)}, preview_t=0.5)
def _fx_defocus_pulse(f, p, t, ch):
    env = 0.5 - 0.5 * math.cos(2 * math.pi * p["rate"] * t)
    return _blur(f, _px(0.04 * p["intensity"] * env, ch))


# =============================================================== Light & Colour
_LC = "Light & Colour"


def _lowres_axes(h, w, s):
    sh, sw = max(h // s, 2), max(w // s, 2)
    ys = ((np.arange(sh, dtype=F32) + 0.5) / sh)
    xs = ((np.arange(sw, dtype=F32) + 0.5) / sw) * F32(w / h)  # x in units of frame height
    return sh, sw, ys, xs


@_reg("light_leak", "Light leak", _LC, {"intensity": _I(0.7), "speed": _p("Speed", 0.3, 0.0, 2.0)})
def _fx_light_leak(f, p, t, ch):
    h, w = f.shape[:2]
    sh, sw, ys, xs = _lowres_axes(h, w, 8)
    tt = t * p["speed"]
    leak = np.zeros((sh, sw, 3), F32)
    cols = ((1.0, 0.42, 0.08), (1.0, 0.18, 0.22), (1.0, 0.78, 0.3))
    aspect = w / h
    for i, col in enumerate(cols):
        side = 0.0 if i % 2 == 0 else aspect
        cx = side + (0.15 + 0.2 * _value_noise(tt + i * 3.1, 71 + i)) * (1 if side == 0 else -1)
        cy = 0.5 + 0.55 * _value_noise(tt * 0.8 + i * 1.7, 81 + i)
        rad = 0.32 + 0.12 * _value_noise(tt * 0.6 + i * 0.9, 91 + i)
        gx = np.exp(-((xs - F32(cx)) / F32(rad * 0.8)) ** 2)
        gy = np.exp(-((ys - F32(cy)) / F32(rad * 1.6)) ** 2)
        leak += (gy[:, None] * gx[None, :])[..., None] * np.array(col, F32)
    leak *= F32(0.95 * p["intensity"])
    return _screen(f, _resize(leak, h, w))


@_reg("lens_flare", "Lens flare", _LC,
      {"intensity": _I(0.7), "x": _p("Light X", 0.25, 0.0, 1.0), "y": _p("Light Y", 0.22, 0.0, 1.0)})
def _fx_lens_flare(f, p, t, ch):
    h, w = f.shape[:2]
    sh, sw, ys, xs = _lowres_axes(h, w, 4)
    aspect = w / h
    Lx, Ly = p["x"] * aspect, p["y"]
    Cx, Cy = 0.5 * aspect, 0.5
    X = xs[None, :]
    Y = ys[:, None]
    d2 = (X - F32(Lx)) ** 2 + (Y - F32(Ly)) ** 2
    acc = np.zeros((sh, sw, 3), F32)
    acc += np.exp(-d2 * F32(1 / 0.0015))[..., None] * np.array([1.0, 0.97, 0.9], F32) * F32(1.3)
    acc += np.exp(-d2 * F32(1 / 0.03))[..., None] * np.array([1.0, 0.7, 0.4], F32) * F32(0.45)
    streak = np.exp(-((Y - F32(Ly)) / F32(0.008)) ** 2) * np.exp(-((X - F32(Lx)) / F32(0.6)) ** 2)
    acc += streak[..., None] * np.array([0.55, 0.7, 1.0], F32) * F32(0.7)
    ghosts = ((0.55, 0.035, (0.9, 0.5, 0.2), 0.35), (0.8, 0.06, (0.3, 0.9, 0.5), 0.2),
              (1.15, 0.02, (0.6, 0.6, 1.0), 0.45), (1.45, 0.09, (0.5, 0.3, 0.9), 0.15),
              (1.8, 0.05, (1.0, 0.6, 0.3), 0.25))
    for pos, size, col, a in ghosts:
        gx, gy = Lx + (Cx - Lx) * 2 * pos, Ly + (Cy - Ly) * 2 * pos
        dg = np.sqrt((X - F32(gx)) ** 2 + (Y - F32(gy)) ** 2)
        disc = (1 - _smoothstep(size * 0.7, size, dg)) * F32(a)
        acc += disc[..., None] * np.array(col, F32)
    # halo ring around the centre of the frame
    dc = np.sqrt((X - F32(Lx + (Cx - Lx) * 1.3)) ** 2 + (Y - F32(Ly + (Cy - Ly) * 1.3)) ** 2)
    ring = np.exp(-((dc - F32(0.28)) / F32(0.012)) ** 2) * F32(0.12)
    acc += ring[..., None] * np.array([0.6, 0.8, 1.0], F32)
    acc *= F32(p["intensity"])
    return _screen(f, _resize(acc, h, w))


@_reg("god_rays", "God rays", _LC,
      {"intensity": _I(0.7), "x": _p("Source X", 0.5, 0.0, 1.0), "y": _p("Source Y", 0.05, 0.0, 1.0)})
def _fx_god_rays(f, p, t, ch):
    h, w = f.shape[:2]
    s = 4
    src = _down(f, s)
    sh, sw = src.shape[:2]
    bright = np.clip((_luma(src) - F32(0.55)) * F32(2.5), 0, 1)[..., None] * (src * F32(0.5) + F32(0.5))
    Ly, Lx = p["y"] * (sh - 1), p["x"] * (sw - 1)
    ys0 = np.arange(sh, dtype=F32) - F32(Ly)
    xs0 = np.arange(sw, dtype=F32) - F32(Lx)
    n = 14
    acc = np.zeros_like(bright)
    for i in range(n):
        sc = 1 - 0.7 * i / n
        wgt = F32(1 - 0.5 * i / n)
        smp = _sep_sample(bright, ys0 * F32(sc) + F32(Ly), xs0 * F32(sc) + F32(Lx))
        smp *= wgt
        acc += smp
    acc *= F32(1.5 * p["intensity"] / n)
    acc *= np.array([1.0, 0.9, 0.72], F32)
    return _screen(f, _resize(acc, h, w))


@lru_cache(maxsize=16)
def _ramp(h, w, angle_q):
    a = math.radians(angle_q / 10)
    Y, X = _grid(h, w)
    u = X * F32(math.cos(a)) + Y * F32(math.sin(a))
    u = (u - u.min()) / max(float(u.max() - u.min()), 1e-6)
    u = u.astype(F32)
    u.flags.writeable = False
    return u


@_reg("gradient_overlay", "Colour gradient overlay", _LC,
      {"intensity": _I(0.6), "hue_a": _p("Hue A", 320.0, 0.0, 360.0), "hue_b": _p("Hue B", 40.0, 0.0, 360.0),
       "angle": _p("Angle", 30.0, -180.0, 180.0)})
def _fx_gradient_overlay(f, p, t, ch):
    h, w = f.shape[:2]
    u = _ramp(h, w, int(round(p["angle"] * 10)))[..., None]
    a, b = _hue_rgb(p["hue_a"], 0.8, 1.0), _hue_rgb(p["hue_b"], 0.8, 1.0)
    col = a + (b - a) * u
    return _mix(f, _softlight(f, col), p["intensity"])


@lru_cache(maxsize=1)
def _rainbow_lut():
    q = (np.arange(256, dtype=F32) + 0.5) / 256
    env = np.sin(q * F32(math.pi)).astype(F32)
    col = _hsv_hue(F32(0.75) * (1 - q)) * F32(0.75)  # outer red .. inner violet
    env[0] = env[-1] = 0
    return env, col


@_reg("rainbow", "Rainbow", _LC, {"intensity": _I(0.6), "x": _p("Centre X", 0.5, 0.0, 1.0)})
def _fx_rainbow(f, p, t, ch):
    h, w = f.shape[:2]
    rows = max(1, int(0.7 * h) + 1)  # the arc fades out into the ground below this
    Y, X = _grid(h, w)
    Y = Y[:rows]
    cx = (p["x"] - 0.5) * w
    d = np.sqrt((X - F32(cx)) ** 2 + (Y - F32(0.75 * h)) ** 2)
    R, wd = 1.0 * h, 0.14 * h
    q = np.clip((d - F32(R - wd / 2)) * F32(256 / wd), 0, 255).astype(np.uint8)
    env, col = _rainbow_lut()
    fade = np.clip((F32(0.7 * h) - Y) * F32(2.5 / h), 0, 1) * F32(0.5 * p["intensity"])
    a = env[q] * fade
    top = f[:rows]
    top -= top * (a * F32(0.6))[..., None]
    top += col[q] * a[..., None]
    return f


@_reg("prism", "Prism", _LC, {"amount": _I(0.6, label="Amount")})
def _fx_prism(f, p, t, ch):
    h, w = f.shape[:2]
    a = p["amount"]
    d = 0.015 * a
    cy, cx = (h - 1) / 2, (w - 1) / 2
    y0 = np.arange(h, dtype=F32) - F32(cy)
    x0 = np.arange(w, dtype=F32) - F32(cx)
    out = np.empty_like(f)
    for c, sc in ((0, 1 + d), (1, 1.0), (2, 1 - d)):
        if sc == 1.0:
            out[..., c] = f[..., c]
        else:
            out[..., c] = _sep_sample(f[..., c], y0 / F32(sc) + F32(cy), x0 / F32(sc) + F32(cx))
    # a faint spectral ghost image, like light split by a glass prism
    g = _shift_y(_shift_x(f, int(round(_px(0.045 * a, ch)))), -int(round(_px(0.03 * a, ch))))
    ghost = g[..., [2, 0, 1]] * F32(0.3 * a)
    return _screen(out, ghost)


@lru_cache(maxsize=4)
def _bokeh_set(n):
    rng = np.random.default_rng(1234)
    return (rng.random(n), rng.random(n), rng.uniform(0.5, 1.0, n), rng.uniform(-0.03, 0.03, n),
            rng.uniform(0.01, 0.05, n), rng.uniform(0.0, 1.0, n))


@_reg("bokeh", "Bokeh overlay", _LC, {"intensity": _I(0.7), "size": _p("Size", 0.5, 0.0, 1.0)})
def _fx_bokeh(f, p, t, ch):
    h, w = f.shape[:2]
    s = 4
    sh, sw = max(h // s, 2), max(w // s, 2)
    small = _down(f, s) if s > 1 else f
    n = 22
    bx, by, br, vx, vy, hue = _bokeh_set(n)
    acc = np.zeros((sh, sw, 3), F32)
    palette = np.array([[1.0, 0.75, 0.35], [1.0, 0.55, 0.4], [0.9, 0.85, 0.6], [0.6, 0.75, 1.0]], F32)
    for i in range(n):
        rad = (0.03 + 0.07 * p["size"]) * br[i] * sh
        cxp = ((bx[i] + vx[i] * t + 0.02 * math.sin(t * 0.7 + i)) % 1.0) * sw
        cyp = ((by[i] - vy[i] * t) % 1.0) * sh
        x0, x1 = max(int(cxp - rad - 1), 0), min(int(cxp + rad + 2), sw)
        y0, y1 = max(int(cyp - rad - 1), 0), min(int(cyp + rad + 2), sh)
        if x1 <= x0 or y1 <= y0:
            continue
        yy = np.arange(y0, y1, dtype=F32)[:, None] - F32(cyp)
        xx = np.arange(x0, x1, dtype=F32)[None, :] - F32(cxp)
        d = np.sqrt(yy * yy + xx * xx)
        disc = np.clip(F32(rad) - d + F32(0.5), 0, 1) * (F32(0.55) + F32(0.45) * _smoothstep(0.6 * rad, rad, d))
        iy, ix = min(int(cyp), small.shape[0] - 1), min(int(cxp), small.shape[1] - 1)
        col = palette[int(hue[i] * len(palette)) % len(palette)] * F32(0.6) + small[iy, ix] * F32(0.4)
        acc[y0:y1, x0:x1] += disc[..., None] * col * F32(0.35)
    acc *= F32(p["intensity"])
    return _screen(f, _resize(acc, h, w))


@lru_cache(maxsize=8)
def _sunset_rows(h):
    y = (np.arange(h, dtype=F32) + 0.5) / h
    stops = np.array([[1.0, 0.42, 0.18], [1.0, 0.62, 0.42], [0.62, 0.32, 0.55], [0.32, 0.2, 0.45]], F32)
    x = y * (len(stops) - 1)
    i = np.minimum(x.astype(int), len(stops) - 2)
    fr = (x - i)[:, None]
    rows = (stops[i] + (stops[i + 1] - stops[i]) * fr).astype(F32)
    rows.flags.writeable = False
    return rows


@_reg("sunset_tint", "Sunset tint", _LC, {"intensity": _I(0.7)})
def _fx_sunset_tint(f, p, t, ch):
    rows = _sunset_rows(f.shape[0])[:, None, :]
    out = _softlight(f, rows)
    out *= np.array([1.06, 1.0, 0.94], F32)
    return _mix(f, out, p["intensity"])


@_reg("split_tone", "Split-tone", _LC,
      {"intensity": _I(0.8), "hue_shadow": _p("Shadow hue", 200.0, 0.0, 360.0),
       "hue_high": _p("Highlight hue", 35.0, 0.0, 360.0), "balance": _p("Balance", 0.0, -1.0, 1.0)})
def _fx_split_tone(f, p, t, ch):
    def off(hue):
        c = _hue_rgb(hue, 1.0, 1.0)
        return c - F32(float(c @ _LUMA))
    L = _luma(f)
    piv = 0.5 + 0.3 * p["balance"]
    ws = np.clip(1 - L * F32(1 / piv), 0, 1)
    wh = np.clip((L - F32(piv)) * F32(1 / (1 - piv)), 0, 1)
    k = F32(0.35 * p["intensity"])
    return f + (ws[..., None] * off(p["hue_shadow"]) + wh[..., None] * off(p["hue_high"])) * k


_SWAPS = ((2, 0, 1), (1, 2, 0), (2, 1, 0), (0, 2, 1), (1, 0, 2))


@_reg("channel_swap", "Channel swap", _LC,
      {"amount": _I(1.0, label="Amount"), "mode": _p("Mode (BRG, GBR, BGR, RBG, GRB)", 0, 0, 4)}, kind="u8")
def _fx_channel_swap(u8, p, t, ch):
    out = u8[..., list(_SWAPS[int(round(p["mode"]))])]
    return _mix_u8(u8, out, p["amount"])


_YIQ = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])


@lru_cache(maxsize=512)
def _hue_mat(deg_q):
    a = math.radians(deg_q / 4)
    R = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    M = np.linalg.inv(_YIQ) @ R @ _YIQ  # column-vector form; luma preserved
    return M.T.astype(F32)


@_reg("hue_rotate", "Hue rotate (animated)", _LC,
      {"amount": _I(1.0, label="Amount"), "degrees": _p("Offset (deg)", 60.0, 0.0, 360.0),
       "speed": _p("Speed (rev/s)", 0.25, -2.0, 2.0)})
def _fx_hue_rotate(f, p, t, ch):
    deg = (p["degrees"] + 360 * p["speed"] * t) % 360
    h, w = f.shape[:2]
    out = (np.ascontiguousarray(f).reshape(-1, 3) @ _hue_mat(int(round(deg * 4)))).reshape(h, w, 3)
    return _mix(f, out, p["amount"])


@_reg("colour_pop", "Colour pop (keep one hue)", _LC,
      {"intensity": _I(1.0), "hue": _p("Kept hue", 10.0, 0.0, 360.0), "width": _p("Hue width", 35.0, 5.0, 120.0)})
def _fx_colour_pop(f, p, t, ch):
    tgt = _hue_rgb(p["hue"], 1.0, 1.0)
    tb, tr = float(tgt @ _CB), float(tgt @ _CR)
    tn = math.hypot(tb, tr)
    cb = f @ _CB
    cr = f @ _CR
    chroma = np.sqrt(cb * cb + cr * cr)
    cos_a = (cb * F32(tb / tn) + cr * F32(tr / tn)) / (chroma + F32(1e-4))  # cos of the hue angle gap
    wdt = math.radians(p["width"])
    keep = _smoothstep(math.cos(wdt), math.cos(wdt * 0.5), cos_a)
    keep *= np.clip((chroma - F32(0.02)) * F32(20), 0, 1)
    L = _luma(f)[..., None]
    out = L + (f - L) * (keep[..., None] * F32(1.15))
    return _mix(f, out, p["intensity"])


@lru_cache(maxsize=1)
def _sepia_cyan_lut():
    stops = np.array([[0.02, 0.12, 0.16], [0.18, 0.36, 0.4], [0.55, 0.52, 0.47], [0.86, 0.74, 0.56],
                      [1.0, 0.95, 0.85]], F32)
    x = np.linspace(0, len(stops) - 1, 256)
    i = np.minimum(x.astype(int), len(stops) - 2)
    fr = (x - i)[:, None].astype(F32)
    return stops[i] + (stops[i + 1] - stops[i]) * fr


@_reg("sepia_cyan", "Sepia-cyan", _LC, {"intensity": _I(1.0)})
def _fx_sepia_cyan(f, p, t, ch):
    idx = np.clip(_luma(f) * 255 + 0.5, 0, 255).astype(np.uint8)
    return _mix(f, _sepia_cyan_lut()[idx], p["intensity"])


# =============================================================== more Stylize
def _shift2(a, dy, dx):
    return _shift_y(_shift_x(a, dx), dy)


@_reg("oil_paint", "Oil paint", "Stylize", {"intensity": _I(1.0), "size": _p("Brush size", 0.4, 0.0, 1.0)})
def _fx_oil_paint(f, p, t, ch):
    """Kuwahara filter at half resolution: each pixel takes the mean of its calmest quadrant."""
    h, w = f.shape[:2]
    s = 2 if min(h, w) >= 64 else 1
    src = _down(f, s)
    r = max(1, int(round(_px(0.004 + 0.012 * p["size"], ch) / s)))
    L = _luma(src)[..., None]
    stack = np.concatenate([src, L, L * L], axis=2)
    m = _box1d(_box1d(stack, r, 0), r, 1)
    var = m[..., 4] - m[..., 3] ** 2
    best, bv = None, None
    for dy, dx in ((-r, -r), (-r, r), (r, -r), (r, r)):
        mq = _shift2(m[..., :3], -dy, -dx)
        vq = _shift2(var, -dy, -dx)
        if best is None:
            best, bv = mq.copy(), vq.copy()
        else:
            sel = vq < bv
            best[sel] = mq[sel]
            np.minimum(bv, vq, out=bv)
    best = _sat(best, 1.2)
    gx, gy = _grad(_luma(best))
    best += (gx - gy)[..., None] * F32(0.4)  # a hint of brush relief
    out = _resize(best, h, w) if s > 1 else best
    return _mix(f, out, p["intensity"])


@lru_cache(maxsize=8)
def _paper(h, w):
    rng = np.random.default_rng(7)
    n = rng.random((max(h // 2, 1), max(w // 2, 1), 1), dtype=F32)
    n = _box1d(_box1d(n, 2, 0), 2, 1)
    n = _resize(n, h, w)[..., 0]
    n = (n - n.mean()) / max(float(n.std()), 1e-6)
    n = n.astype(F32)
    n.flags.writeable = False
    return n


@_reg("watercolour", "Watercolour", "Stylize", {"intensity": _I(1.0)})
def _fx_watercolour(f, p, t, ch):
    h, w = f.shape[:2]
    s = 2 if min(h, w) >= 64 else 1
    src = _down(f, s)
    r = max(1, int(round(_px(0.006, ch) / s)))
    sm = _box1d(_box1d(src, r, 0), r, 1)
    q = np.round(sm * F32(6)) * F32(1 / 6)
    q = q * F32(0.6) + sm * F32(0.4)
    gx, gy = _grad(_luma(sm))
    e = np.clip(np.sqrt(gx * gx + gy * gy) * F32(5 * s), 0, 1)
    pig = 1 - (1 - _sat(q, 1.15)) * F32(0.8)  # washed, lighter pigment
    pig *= (1 - e * F32(0.45))[..., None]  # pigment pools at the edges
    out = _resize(pig, h, w) if s > 1 else pig
    out = out * (F32(1) + _paper(h, w)[..., None] * F32(0.015))
    return _mix(f, out, p["intensity"])


@_reg("cartoon", "Cartoon", "Stylize", {"intensity": _I(1.0)})
def _fx_cartoon(f, p, t, ch):
    h, w = f.shape[:2]
    s = 2 if min(h, w) >= 64 else 1
    src = _down(f, s)
    r = max(1, int(round(_px(0.006, ch) / s)))
    sm = _box1d(_box1d(src, r, 0), r, 1)
    q = np.round(np.clip(_sat(sm, 1.35), 0, 1) * F32(4)) * F32(1 / 4)
    gx, gy = _grad(_luma(sm))
    edge = np.clip((np.sqrt(gx * gx + gy * gy) * F32(s) - F32(0.035)) * F32(25), 0, 1)
    q *= (1 - edge)[..., None]
    out = _resize(q, h, w) if s > 1 else q
    return _mix(f, out, p["intensity"])


@lru_cache(maxsize=16)
def _dot_screen(h, w, b_q):
    b = b_q / 100
    Y, X = _grid(h, w)
    u = (X + Y) * F32(2 * math.pi / (b * math.sqrt(2)))
    v = (X - Y) * F32(2 * math.pi / (b * math.sqrt(2)))
    P = (F32(0.5) + F32(0.25) * (np.cos(u) + np.cos(v))).astype(F32)
    P.flags.writeable = False
    return P


@_reg("comic_dots", "Comic dots (Ben-Day)", "Stylize",
      {"intensity": _I(1.0), "size": _p("Dot size", 0.4, 0.0, 1.0)})
def _fx_comic_dots(f, p, t, ch):
    h, w = f.shape[:2]
    b = max(3.0, _px(0.01 + 0.02 * p["size"], ch))
    P = _dot_screen(h, w, int(round(b * 100)))[..., None]
    base = _sat(f, 1.6)
    out = np.clip((base - P) * F32(4) + F32(0.5), 0, 1)  # anti-aliased dot screen
    out *= F32(0.88)
    out += F32(0.1)
    edge = _smoothstep(0.06, 0.14, _edges(f, ch, 0.003))
    out *= (1 - edge)[..., None]
    out *= np.array([0.99, 0.97, 0.9], F32)
    return _mix(f, out, p["intensity"])


@lru_cache(maxsize=16)
def _hatch_lines(h, w, per_q):
    per = per_q / 100
    yy = np.arange(h, dtype=F32)[:, None]
    xx = np.arange(w, dtype=F32)[None, :]
    th = max(0.7, per * 0.16)

    def line(c):
        d = np.abs(((c / F32(per)) % 1) - F32(0.5)) * F32(per)  # px from the mid-line
        return np.clip(F32(th) - d + F32(0.5), 0, 1).astype(F32)
    out = (line(xx + yy), line(xx - yy + 0.37 * per), line(yy + 0.25 * per + 0 * xx),
           line(xx + 0.25 * per + 0 * yy))
    for a in out:
        a.flags.writeable = False
    return out


def _ramp_down(L, lo, hi):
    """1 below lo, 0 above hi, linear between."""
    return np.clip((F32(hi) - L) * F32(1 / (hi - lo)), 0, 1)


@_reg("crosshatch", "Crosshatch", "Stylize", {"intensity": _I(1.0)})
def _fx_crosshatch(f, p, t, ch):
    h, w = f.shape[:2]
    per = max(3.0, _px(0.014, ch))
    l1, l2, l3, l4 = _hatch_lines(h, w, int(round(per * 100)))
    L = _blur(_luma(f)[..., None], max(_px(0.003, ch), 0.6))[..., 0]
    ink = l1 * _ramp_down(L, 0.65, 0.8)
    np.maximum(ink, l2 * _ramp_down(L, 0.42, 0.55), out=ink)
    np.maximum(ink, l3 * _ramp_down(L, 0.25, 0.35), out=ink)
    np.maximum(ink, l4 * _ramp_down(L, 0.1, 0.18), out=ink)
    gx, gy = _grad(L)
    np.maximum(ink, np.clip((np.sqrt(gx * gx + gy * gy) - F32(0.08)) * F32(8), 0, 1), out=ink)
    paper = np.array([0.96, 0.94, 0.88], F32)
    pen = np.array([0.12, 0.12, 0.2], F32)
    out = paper + (pen - paper) * ink[..., None]
    return _mix(f, out, p["intensity"])


_GLYPHS = (
    "......|......|......|......|......|......",
    "......|......|......|......|..##..|..##..",
    "......|..##..|..##..|......|..##..|..##..",
    "......|......|######|######|......|......",
    "......|######|......|......|######|......",
    "......|..##..|######|######|..##..|......",
    "#....#|.#..#.|..##..|..##..|.#..#.|#....#",
    ".#..#.|######|.#..#.|.#..#.|######|.#..#.",
    "#.##.#|.####.|######|######|.####.|#.##.#",
    ".####.|##..##|##.###|###.##|##..##|.####.",
    "######|##.###|######|###.##|######|######",
)


@lru_cache(maxsize=1)
def _glyph_atlas():
    g = np.array([[[c == "#" for c in row] for row in s.split("|")] for s in _GLYPHS], F32)
    return g[np.argsort(g.sum(axis=(1, 2)), kind="stable")]


@lru_cache(maxsize=16)
def _glyphs_at(b):
    g = _glyph_atlas()
    idx = np.arange(b) * g.shape[1] // b
    out = g[:, idx][:, :, idx]
    out.flags.writeable = False
    return out


@_reg("ascii_blocks", "ASCII blocks", "Stylize", {"intensity": _I(1.0), "size": _p("Cell size", 0.3, 0.0, 1.0)})
def _fx_ascii_blocks(f, p, t, ch):
    h, w = f.shape[:2]
    b = max(4, int(round(_px(0.014 + 0.03 * p["size"], ch))))
    cells = _block_mean(f, b)
    nh, nw = cells.shape[:2]
    G = _glyphs_at(b)
    n = G.shape[0]
    L = np.clip(_luma(cells) * F32(1.15), 0, 1)
    idx = np.minimum((L * n).astype(np.intp), n - 1)
    mask = G[idx].transpose(0, 2, 1, 3).reshape(nh * b, nw * b)[:h, :w]
    col = np.clip(cells * F32(1.3) + F32(0.1), 0, 1)
    col = _block_up(col, b, h, w)
    bg = np.array([0.02, 0.025, 0.03], F32)
    out = bg + (col - bg) * mask[..., None]
    return _mix(f, out, p["intensity"])


@lru_cache(maxsize=16)
def _tile_maps(h, w, b):
    yy = (np.arange(h) % b).astype(F32)[:, None]
    xx = (np.arange(w) % b).astype(F32)[None, :]
    g = max(1, b // 9)
    grout = ((yy < g) | (xx < g)).astype(F32)
    shade = (F32(1) + F32(0.16) * ((F32(b / 2) - yy) + (F32(b / 2) - xx)) * F32(1 / b)) * (1 - grout)
    rng = np.random.default_rng(5)
    jit = rng.uniform(0.92, 1.08, (-(-h // b), -(-w // b), 1)).astype(F32)
    for a in (grout, shade, jit):
        a.flags.writeable = False
    return grout, shade, jit


@_reg("mosaic_tiles", "Mosaic tiles", "Stylize", {"intensity": _I(1.0), "size": _p("Tile size", 0.4, 0.0, 1.0)})
def _fx_mosaic_tiles(f, p, t, ch):
    h, w = f.shape[:2]
    b = max(4, int(round(_px(0.02 + 0.05 * p["size"], ch))))
    grout, shade, jit = _tile_maps(h, w, b)
    cells = _sat(_block_mean(f, b), 1.15) * jit
    up = _block_up(cells, b, h, w)
    out = up * shade[..., None] + (F32(0.35) + up * F32(0.25)) * grout[..., None]
    return _mix(f, out, p["intensity"])


@lru_cache(maxsize=8)
def _voronoi(h, w, c):
    rng = np.random.default_rng(77)
    ny, nx = h // c + 3, w // c + 3
    sy = ((np.arange(ny)[:, None] - 1) + rng.uniform(0.15, 0.85, (ny, nx))) * c
    sx = ((np.arange(nx)[None, :] - 1) + rng.uniform(0.15, 0.85, (ny, nx))) * c
    yy = np.arange(h, dtype=F32)[:, None]
    xx = np.arange(w, dtype=F32)[None, :]
    cy = (np.arange(h) // c)[:, None] + 1
    cx = (np.arange(w) // c)[None, :] + 1
    d1 = np.full((h, w), 1e9, F32)
    d2 = np.full((h, w), 1e9, F32)
    ids = np.zeros((h, w), np.int32)
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            iy = np.broadcast_to(cy + oy, (h, w))
            ix = np.broadcast_to(cx + ox, (h, w))
            gid = iy * nx + ix
            d = np.sqrt((sy.ravel()[gid] - yy) ** 2 + (sx.ravel()[gid] - xx) ** 2).astype(F32)
            closer = d < d1
            d2 = np.where(closer, d1, np.minimum(d2, d))
            ids = np.where(closer, gid, ids).astype(np.int32)
            d1 = np.where(closer, d, d1)
    seeds_y = np.clip(sy.ravel(), 0, h - 1).astype(F32)
    seeds_x = np.clip(sx.ravel(), 0, w - 1).astype(F32)
    out = (ids, d1, d2 - d1, seeds_y, seeds_x)
    for a in out:
        a.flags.writeable = False
    return out


@_reg("stained_glass", "Stained glass", "Stylize",
      {"intensity": _I(1.0), "size": _p("Cell size", 0.4, 0.0, 1.0)})
def _fx_stained_glass(f, p, t, ch):
    h, w = f.shape[:2]
    c = max(4, int(round(_px(0.04 + 0.1 * p["size"], ch))))
    ids, d1, gap, sy, sx = _voronoi(h, w, c)
    s = 4 if min(h, w) >= 32 else 1
    small = _down(f, s)
    sh, sw = small.shape[:2]
    cols = small[np.minimum((sy / s).astype(np.intp), sh - 1), np.minimum((sx / s).astype(np.intp), sw - 1)]
    cols = np.clip(_sat(cols[None], 1.5)[0] * F32(1.12), 0, 1)
    glass = cols[ids]
    glass *= (F32(1.1) - d1 * F32(0.35 / c))[..., None]  # brighter toward each cell's centre
    lw = max(1.0, _px(0.004, ch))
    lead = np.clip(F32(lw) - gap + F32(0.5), 0, 1)[..., None]
    out = glass + (F32(0.04) - glass) * lead
    return _mix(f, out, p["intensity"])


@_reg("neon_edges", "Neon edges", "Stylize", {"intensity": _I(1.0)})
def _fx_neon_edges(f, p, t, ch):
    e = np.clip(_edges(f, ch, 0.003) * F32(6) - F32(0.1), 0, 1)
    mx = np.maximum(np.maximum(f[..., 0], f[..., 1]), f[..., 2])
    np.maximum(mx, F32(0.15), out=mx)
    e *= F32(0.85) / mx  # full-brightness version of each pixel's colour along the edges
    line = f * e[..., None]
    line += e[..., None] * F32(0.15)
    glow = _blur(line, _px(0.02, ch))
    glow *= F32(2.5)
    glow += line
    return _mix(f, glow, p["intensity"])


@_reg("outline_glow", "Outline glow", "Stylize",
      {"intensity": _I(0.8), "hue": _p("Glow hue", 190.0, 0.0, 360.0)})
def _fx_outline_glow(f, p, t, ch):
    e = np.clip(_edges(f, ch, 0.003) * F32(5) - F32(0.15), 0, 1)[..., None]
    glow = _blur(e, _px(0.015, ch))
    col = _hue_rgb(p["hue"], 0.7, 1.0)
    add = (e * F32(0.9) + glow * F32(2.2)) * col * F32(p["intensity"])
    return _screen(f, add)


@lru_cache(maxsize=8)
def _scatter(h, w):
    rng = np.random.default_rng(13)
    out = rng.uniform(-1, 1, (2, h, w)).astype(F32)
    out.flags.writeable = False
    return out


@_reg("frosted_glass", "Frosted glass", "Stylize", {"amount": _I(0.6, label="Amount")},
      kind="rgba", moves_alpha=True)
def _fx_frosted_glass(img, p, t, ch):
    h, w = img.shape[:2]
    a = p["amount"]
    r = _px(0.012 * a, ch)
    n = _scatter(h, w)
    yy = np.arange(h, dtype=F32)[:, None]
    xx = np.arange(w, dtype=F32)[None, :]
    out = _nearest(img, yy + n[0] * F32(r), xx + n[1] * F32(r)).copy()
    f = _to_f(out[..., :3])
    f += (np.array([0.92, 0.95, 1.0], F32) - f) * F32(0.2 * a)  # frosty, slightly milky
    out[..., :3] = _to_u8(f)
    return out


@lru_cache(maxsize=32)
def _lens_map(h, w, kind, a_q, r_q):
    """Source coords (ys, xs) for static radial distortions (fisheye/bulge/pinch/swirl)."""
    a, R = a_q / 1000, r_q / 1000
    Y, X = _grid(h, w)
    Y = np.broadcast_to(Y, (h, w))
    X = np.broadcast_to(X, (h, w))
    cy, cx = F32((h - 1) / 2), F32((w - 1) / 2)
    if kind == "fisheye":
        rn2 = (X * X + Y * Y) * F32(1 / (cx * cx + cy * cy))
        sc = 1 - F32(a) * (1 - rn2)
        ys, xs = Y * sc + cy, X * sc + cx
    else:
        rad = R * h
        rn = np.sqrt(X * X + Y * Y) * F32(1 / rad)
        fall = np.clip(1 - rn * rn, 0, 1) ** 2
        if kind == "bulge":
            sc = 1 - F32(a) * fall
            ys, xs = Y * sc + cy, X * sc + cx
        elif kind == "pinch":
            sc = 1 + F32(a) * fall
            ys, xs = Y * sc + cy, X * sc + cx
        else:  # swirl
            th = F32(a) * np.clip(1 - rn, 0, 1) ** 2
            c, s = np.cos(th), np.sin(th)
            xs = X * c - Y * s + cx
            ys = X * s + Y * c + cy
    ys, xs = ys.astype(F32), xs.astype(F32)
    ys.flags.writeable = False
    xs.flags.writeable = False
    return ys, xs


def _make_lens(kind, scale_a, has_radius):
    def fn(img, p, t, ch):
        h, w = img.shape[:2]
        a = int(round(p["amount"] * scale_a * 1000))
        r = int(round(p.get("radius", 0.5) * 1000)) if has_radius else 0
        if a == 0:
            return img
        ys, xs = _lens_map(h, w, kind, a, r)
        return _warp(img, ys, xs)
    return fn


_reg("fisheye", "Fisheye lens", "Stylize", {"amount": _I(0.6, label="Amount")},
     kind="rgba", moves_alpha=True)(_make_lens("fisheye", 0.5, False))
_reg("bulge", "Bulge", "Stylize", {"amount": _I(0.6, label="Amount"), "radius": _p("Radius", 0.45, 0.1, 1.0)},
     kind="rgba", moves_alpha=True)(_make_lens("bulge", 0.6, True))
_reg("pinch", "Pinch", "Stylize", {"amount": _I(0.6, label="Amount"), "radius": _p("Radius", 0.45, 0.1, 1.0)},
     kind="rgba", moves_alpha=True)(_make_lens("pinch", 0.8, True))
_reg("swirl", "Swirl", "Stylize", {"amount": _I(0.6, label="Amount"), "radius": _p("Radius", 0.5, 0.1, 1.0)},
     kind="rgba", moves_alpha=True)(_make_lens("swirl", 4.0, True))


@lru_cache(maxsize=8)
def _polar(h, w):
    Y, X = _grid(h, w)
    r = np.sqrt(X * X + Y * Y).astype(F32)
    r.flags.writeable = False
    return r


@_reg("ripple", "Ripple (water)", "Stylize",
      {"amount": _I(0.5, label="Amount"), "speed": _p("Speed", 1.0, 0.0, 5.0)}, kind="rgba", moves_alpha=True)
def _fx_ripple(img, p, t, ch):
    h, w = img.shape[:2]
    Y, X = _grid(h, w)
    r = _polar(h, w)
    lam = max(h * 0.09, 3)
    amp = _px(0.012 * p["amount"], ch)
    ph = (r * F32(1 / lam) - F32(p["speed"] * t * 1.5)) * F32(2 * math.pi)
    disp = np.sin(ph) * np.exp(-r * F32(1 / (0.8 * h))) * F32(amp)
    sc = 1 + disp / (r + F32(1))
    return _warp(img, Y * sc + F32((h - 1) / 2), X * sc + F32((w - 1) / 2))


@_reg("wave", "Wave distortion", "Stylize",
      {"amount": _I(0.5, label="Amount"), "speed": _p("Speed", 1.0, 0.0, 5.0)}, kind="rgba", moves_alpha=True)
def _fx_wave(img, p, t, ch):
    h, w = img.shape[:2]
    amp = _px(0.018 * p["amount"], ch)
    ph = t * p["speed"] * 2 * math.pi
    yy = np.arange(h, dtype=F32)[:, None]
    xx = np.arange(w, dtype=F32)[None, :]
    xs = xx + np.sin(yy * F32(2 * math.pi / max(h * 0.4, 4)) + F32(ph)) * F32(amp)
    ys = yy + np.sin(xx * F32(2 * math.pi / max(h * 0.55, 4)) + F32(ph * 0.8 + 1.0)) * F32(amp)
    return _warp(img, ys, xs)


@_reg("tile_repeat", "Tile / grid repeat", "Stylize",
      {"amount": _I(1.0, label="Amount"), "tiles": _p("Tiles per side", 3, 2, 6)}, kind="rgba", moves_alpha=True)
def _fx_tile_repeat(img, p, t, ch):
    h, w = img.shape[:2]
    n = int(round(p["tiles"]))
    ys = (((np.arange(h, dtype=F32) + F32(0.5)) * F32(n)) % F32(h)) - F32(0.5)
    xs = (((np.arange(w, dtype=F32) + F32(0.5)) * F32(n)) % F32(w)) - F32(0.5)
    out = _sep_sample(img, ys, xs)
    out += F32(0.5)
    return _mix_rgba(img, out.astype(np.uint8), p["amount"])


# =============================================================== more Retro & Glitch
@_reg("datamosh", "Datamosh (block smear)", "Retro & Glitch", {"intensity": _I(0.6)})
def _fx_datamosh(f, p, t, ch):
    k = p["intensity"]
    h, w = f.shape[:2]
    b = max(4, int(round(_px(0.045, ch))))
    rng = _rng(t, 101, hold=3)
    out = f.copy()
    nby, nbx = -(-h // b), -(-w // b)
    mvx, mvy = rng.normal(0, _px(0.04, ch)), rng.normal(0, _px(0.02, ch))
    # a moshed region: a band of rows of macroblocks dragged along one motion vector
    y_lo = int(rng.integers(0, max(1, nby - 2)))
    y_hi = min(nby, y_lo + 2 + int(rng.integers(1, max(2, int(nby * (0.3 + 0.5 * k))))))
    for by in range(y_lo, y_hi):
        for bx in range(nbx):
            if rng.random() > 0.35 + 0.6 * k:
                continue
            y0, x0 = by * b, bx * b
            y1, x1 = min(y0 + b, h), min(x0 + b, w)
            m = 1 + rng.random() * 2
            sy = int(np.clip(y0 - mvy * m, 0, h - (y1 - y0)))
            sx = int(np.clip(x0 - mvx * m, 0, w - (x1 - x0)))
            blk = f[sy:sy + y1 - y0, sx:sx + x1 - x0]
            r = rng.random()
            if r < 0.4:  # pixel bleed: one row/column smeared across the block
                blk = np.broadcast_to(blk[:1], blk.shape) if r < 0.2 else np.broadcast_to(blk[:, :1], blk.shape)
            mean = blk.mean(axis=(0, 1))
            out[y0:y1, x0:x1] = blk * F32(0.7) + mean * F32(0.3)
    # vertical melt streaks
    for _ in range(int(1 + 4 * k)):
        x0 = int(rng.integers(0, max(1, w - 4)))
        x1 = min(w, x0 + int(rng.integers(max(2, b // 2), max(3, b * 3))))
        y0 = int(rng.integers(0, max(1, h - 2)))
        y1 = min(h, y0 + int(rng.integers(b, max(b + 1, h // 2))))
        out[y0:y1, x0:x1] = out[y0:y0 + 1, x0:x1] * F32(0.85) + out[y0:y1, x0:x1] * F32(0.15)
    return out


@_reg("glitch_bars", "Chromatic glitch bars", "Retro & Glitch", {"intensity": _I(0.6)})
def _fx_glitch_bars(f, p, t, ch):
    k = p["intensity"]
    h, w = f.shape[:2]
    rng = _rng(t, 111, hold=2)
    out = f.copy()
    tints = np.array([[1.0, 0.1, 1.0], [0.1, 1.0, 1.0], [1.0, 1.0, 0.1], [1.0, 0.2, 0.2]], F32)
    for _ in range(int(3 + 8 * k)):
        bh = int(rng.integers(max(1, h // 90), max(2, int(h * (0.02 + 0.07 * k)))))
        y0 = int(rng.integers(0, max(1, h - bh)))
        band = np.empty_like(f[y0:y0 + bh])
        for c in range(3):
            band[..., c] = np.roll(f[y0:y0 + bh, :, c], int(rng.normal(0, _px(0.06 * k, ch))), axis=1)
        tint = tints[int(rng.integers(0, len(tints)))]
        band = band * (F32(0.7) + tint * F32(0.45))
        if rng.random() < 0.3:  # digital noise inside the bar
            band += (rng.random(band.shape[:2], dtype=F32)[..., None] - F32(0.5)) * F32(0.4)
        out[y0:y0 + bh] = band
    return out


@_reg("bad_tv", "Bad TV", "Retro & Glitch", {"intensity": _I(0.6)})
def _fx_bad_tv(f, p, t, ch):
    k = p["intensity"]
    h, w = f.shape[:2]
    rng = _rng(t, 121)
    off = int(((t * 0.45 * k) % 1.0) * h)  # vertical hold slipping
    rows = (np.arange(h) + off) % h
    yy = np.arange(h, dtype=F32)
    wob = np.sin(yy * F32(2 * math.pi / max(h * 0.3, 4)) + F32(t * 9)) * F32(_px(0.01 * k, ch))
    wob += rng.standard_normal(h).astype(F32) * F32(_px(0.002 * k, ch))
    cols = np.clip(np.arange(w)[None, :] - np.round(wob).astype(np.intp)[:, None], 0, w - 1)
    g = np.take(f.reshape(-1, 3), (rows[:, None] * w + cols).ravel(), axis=0).reshape(h, w, 3)
    seam = (h - off) % h  # dark band where the picture wraps
    band = np.exp(-((yy - F32(seam)) / F32(max(h * 0.025, 1))) ** 2) * F32(0.85 * k)
    g = g * F32(0.8) + _shift_x(g, int(round(_px(0.035 * k, ch)))) * F32(0.25)  # ghost image
    g = _sat(g, 1 - 0.5 * k)
    snow = rng.random((h, w), dtype=F32)
    g += (snow[..., None] - F32(0.45)) * F32(0.45 * k)
    g *= (1 - band)[:, None, None]
    return g


@_reg("interlace", "Interlace", "Retro & Glitch", {"intensity": _I(0.6)})
def _fx_interlace(f, p, t, ch):
    k = p["intensity"]
    h = f.shape[0]
    lh = max(1, int(round(ch / 360)))
    odd = ((np.arange(h) // lh) % 2) == 1
    d = _px(0.015 * k, ch) * (0.6 + 0.4 * math.sin(_tq(t) * 7.0))
    dx = int(round(d)) or 1
    out = f.copy()
    out[odd] = _shift_x(f, dx)[odd] * F32(1 - 0.25 * k)
    return out


@_reg("lcd_dots", "Dot-matrix LCD", "Retro & Glitch",
      {"intensity": _I(1.0), "size": _p("Dot size", 0.3, 0.0, 1.0)})
def _fx_lcd_dots(f, p, t, ch):
    h, w = f.shape[:2]
    b = max(3, int(round(_px(0.01 + 0.02 * p["size"], ch))))
    cells = np.clip(_block_mean(f, b) * F32(1.2), 0, 1)
    up = _block_up(cells, b, h, w)
    dot = np.clip(F32(b * 0.42) - _cell_dist(h, w, b) + F32(0.5), 0, 1)[..., None]
    bg = np.array([0.03, 0.04, 0.035], F32)
    out = bg + (up - bg) * dot
    return _mix(f, out, p["intensity"])


_GB_PAL = np.array([[15, 56, 15], [48, 98, 48], [139, 172, 15], [155, 188, 15]], F32) / 255
_BAYER4 = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], F32) + 0.5) / 16


@_reg("gameboy", "Game Boy palette", "Retro & Glitch", {"intensity": _I(1.0)})
def _fx_gameboy(f, p, t, ch):
    h, w = f.shape[:2]
    b = max(1, int(round(ch / 144)))
    small = _block_mean(f, b) if b > 1 else f
    nh, nw = small.shape[:2]
    L = np.clip((_luma(small) - F32(0.06)) * F32(1.15), 0, 1)
    th = np.tile(_BAYER4, (-(-nh // 4), -(-nw // 4)))[:nh, :nw]
    idx = np.clip(np.floor(L * F32(3) + th), 0, 3).astype(np.intp)
    out = _GB_PAL[idx]
    if b > 1:
        out = _block_up(out, b, h, w)
    return _mix(f, out, p["intensity"])


@_reg("film_scratch", "Scratched sepia film", "Retro & Glitch", {"intensity": _I(0.8)})
def _fx_film_scratch(f, p, t, ch):
    k = p["intensity"]
    h, w = f.shape[:2]
    rng = _rng(t, 141)
    held = _rng(t, 142, hold=4)
    g = np.clip(f @ _SEPIA, 0, 1)
    g = (g - F32(0.5)) * F32(1.15) + F32(0.47)
    g = _mix(f, g, min(1.0, 0.4 + k))
    g = _shift_y(g, int(round(rng.normal(0, _px(0.004 * k, ch)))))  # gate weave
    colf = np.ones(w, F32)
    sw = max(1, int(round(_px(0.002, ch))))
    for _ in range(int(3 + 10 * k)):  # full-height scratches that persist a few frames
        x = int(held.integers(0, max(1, w - sw)))
        colf[x:x + sw] = F32(1.6 if held.random() < 0.7 else 0.4)
    g = g * colf[None, :, None]
    yy = np.arange(h, dtype=F32)[:, None]
    xx = np.arange(w, dtype=F32)[None, :]
    for _ in range(int(rng.integers(0, 2 + int(3 * k)))):  # dark blotches
        r = _px(rng.uniform(0.006, 0.03), ch)
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        y0, y1 = max(int(cy - 2 * r), 0), min(int(cy + 2 * r) + 1, h)
        x0, x1 = max(int(cx - 2 * r), 0), min(int(cx + 2 * r) + 1, w)
        blob = np.exp(-(((yy[y0:y1] - F32(cy)) ** 2 + (xx[:, x0:x1] - F32(cx)) ** 2) / F32(r * r)))
        g[y0:y1, x0:x1] *= (1 - blob * F32(0.8))[..., None]
    g = g * F32(1 + (rng.random() - 0.5) * 0.3 * k)
    g += (rng.random((h, w), dtype=F32) - F32(0.5))[..., None] * F32(0.12 * k)
    v = _vig(h, w, 0.35, 1.1)
    return g * (1 - v * F32(0.85 * k))[..., None]


@lru_cache(maxsize=16)
def _burn_noise(sh, sw, cycle):
    rng = np.random.default_rng([cycle & 0x7FFFFFFF, 151])
    n = rng.random((sh, sw, 1), dtype=F32)
    n = _box1d(_box1d(n, 2, 0), 2, 1)
    n = _box1d(_box1d(n, 1, 0), 1, 1)[..., 0]
    n = (n - n.mean()) / max(float(n.std()), 1e-6)
    cy, cx = rng.uniform(0.2, 0.8), rng.uniform(0.15, 0.85)
    n = n.astype(F32)
    n.flags.writeable = False
    return n, cy, cx


@_reg("film_burn", "Film burn", "Retro & Glitch",
      {"intensity": _I(0.8), "speed": _p("Speed", 0.5, 0.05, 3.0)}, preview_t=0.8)
def _fx_film_burn(f, p, t, ch):
    k = p["intensity"]
    h, w = f.shape[:2]
    s = 6
    sh, sw = max(h // s, 4), max(w // s, 4)
    pos = t * p["speed"]
    cyc, phase = int(math.floor(pos)), pos - math.floor(pos)
    n, cy, cx = _burn_noise(sh, sw, cyc)
    yy = ((np.arange(sh, dtype=F32) + 0.5) / sh - F32(cy))[:, None]
    xx = ((np.arange(sw, dtype=F32) + 0.5) / sw - F32(cx))[None, :] * F32(w / h)
    d = np.sqrt(yy * yy + xx * xx)
    b = F32(phase * 1.4 * (0.4 + 0.6 * k)) - d + n * F32(0.06)
    hot = _smoothstep(0.0, 0.25, b)[..., None]
    char = (_smoothstep(-0.08, 0.0, b) * (1 - _smoothstep(0.0, 0.08, b)))[..., None]
    glow = hot * np.array([1.0, 0.55, 0.15], F32) + _smoothstep(0.2, 0.45, b)[..., None] * np.array(
        [0.0, 0.42, 0.7], F32)
    glow += _smoothstep(-0.25, 0.0, b)[..., None] * np.array([0.5, 0.12, 0.0], F32) * F32(0.6)
    glow = _resize(glow, h, w)
    char = _resize(char, h, w)
    out = f * (1 - char * F32(0.85))
    return _screen(out, glow * F32(k))


@_reg("polaroid_frame", "Polaroid frame", "Retro & Glitch",
      {"amount": _I(1.0, label="Amount"), "border": _p("Border", 0.06, 0.02, 0.15)}, kind="rgba", moves_alpha=True)
def _fx_polaroid_frame(img, p, t, ch):
    h, w = img.shape[:2]
    bw = max(1, int(round(p["border"] * h)))
    bottom = max(bw + 1, int(round(p["border"] * h * 3.2)))
    y0, y1, x0, x1 = bw, h - bottom, bw, w - bw
    if y1 - y0 < 2 or x1 - x0 < 2:
        return img
    wh, ww = y1 - y0, x1 - x0
    sc = max(wh / h, ww / w)  # cover the window, keep aspect
    ys = (np.arange(wh, dtype=F32) - F32((wh - 1) / 2)) / F32(sc) + F32((h - 1) / 2)
    xs = (np.arange(ww, dtype=F32) - F32((ww - 1) / 2)) / F32(sc) + F32((w - 1) / 2)
    photo = _sep_sample(img, ys, xs)
    photo[..., :3] = photo[..., :3] * F32(0.92) + F32(14)  # slightly faded print
    out = np.empty_like(img)
    out[..., :3] = (247, 244, 236)
    out[..., 3] = 255
    out[y0 - 1:y1 + 1, x0 - 1:x1 + 1, :3] = (200, 196, 188)  # thin inner shadow
    photo += F32(0.5)
    np.clip(photo, 0, 255, out=photo)
    out[y0:y1, x0:x1] = photo.astype(np.uint8)
    return _mix_rgba(img, out, p["amount"])


@_reg("letterbox", "Cinema bars (letterbox)", "Retro & Glitch",
      {"amount": _I(1.0, label="Amount"), "aspect": _p("Aspect ratio", 2.39, 1.0, 3.0)})
def _fx_letterbox(f, p, t, ch):
    h, w = f.shape[:2]
    out = f.copy()
    keep = F32(1 - p["amount"])
    a = p["aspect"]
    if a > w / h:
        bar = int(round((h - w / a) / 2))
        if bar > 0:
            out[:bar] *= keep
            out[h - bar:] *= keep
    else:
        bar = int(round((w - h * a) / 2))
        if bar > 0:
            out[:, :bar] *= keep
            out[:, w - bar:] *= keep
    return out


# =============================================================== more Motion
def _cover(angle, w, h):
    """Zoom needed so a frame rotated by angle still covers the canvas."""
    a = abs(angle)
    return math.cos(a) + max(w / h, h / w) * math.sin(a)


@_reg("sway", "Camera sway", "Motion",
      {"intensity": _I(0.5), "speed": _p("Speed (Hz)", 0.3, 0.05, 2.0)}, kind="rgba", moves_alpha=True)
def _fx_sway(img, p, t, ch):
    k = p["intensity"]
    h, w = img.shape[:2]
    ph = 2 * math.pi * p["speed"] * _tq(t)
    dx = math.sin(ph) * _px(0.025 * k, ch)
    dy = math.sin(ph * 0.7 + 1.0) * _px(0.012 * k, ch)
    ang = math.sin(ph * 0.9 + 0.5) * math.radians(1.5 * k)
    return _affine(img, _cover(ang, w, h) + 0.06 * k, ang, dx, dy)


@_reg("earthquake", "Earthquake", "Motion",
      {"intensity": _I(0.6), "speed": _p("Speed", 18.0, 2.0, 40.0)}, kind="rgba", moves_alpha=True)
def _fx_earthquake(img, p, t, ch):
    k = p["intensity"]
    h, w = img.shape[:2]
    tq = _tq(t) * p["speed"]
    amp = _px(0.045 * k, ch)
    dx = (_value_noise(tq, 131) + 0.4 * _value_noise(tq * 2.3, 132)) * amp
    dy = (_value_noise(tq, 133) + 0.4 * _value_noise(tq * 2.1, 134)) * amp
    ang = _value_noise(tq * 0.8, 135) * math.radians(2.5 * k)
    return _affine(img, _cover(ang, w, h) + 0.12 * k, ang, dx, dy)


@_reg("ken_burns", "Slow zoom (Ken Burns)", "Motion",
      {"intensity": _I(0.6), "duration": _p("Duration (s)", 8.0, 1.0, 30.0),
       "pan_x": _p("Pan X", 0.3, -1.0, 1.0), "pan_y": _p("Pan Y", -0.2, -1.0, 1.0)},
      kind="rgba", moves_alpha=True, preview_t=3.0)
def _fx_ken_burns(img, p, t, ch):
    h, w = img.shape[:2]
    u = min(max(t / p["duration"], 0.0), 1.0)
    sc = 1 + 0.3 * p["intensity"] * u
    room_x, room_y = (sc - 1) * w / 2, (sc - 1) * h / 2  # max shift that keeps the frame covered
    return _zoom_shift(img, sc, -p["pan_x"] * room_x, -p["pan_y"] * room_y)


@_reg("rotate_wobble", "Rotate wobble", "Motion",
      {"intensity": _I(0.5), "speed": _p("Speed (Hz)", 1.0, 0.1, 6.0)}, kind="rgba", moves_alpha=True,
      preview_t=0.25)
def _fx_rotate_wobble(img, p, t, ch):
    h, w = img.shape[:2]
    ang = math.sin(2 * math.pi * p["speed"] * _tq(t)) * math.radians(6 * p["intensity"])
    return _affine(img, _cover(ang, w, h), ang)


@_reg("flash_pulse", "Flash pulse", "Motion",
      {"intensity": _I(0.6), "rate": _p("Rate (Hz)", 1.0, 0.1, 8.0)}, preview_t=0.5)
def _fx_flash_pulse(f, p, t, ch):
    env = (0.5 - 0.5 * math.cos(2 * math.pi * p["rate"] * t)) ** 2 * p["intensity"]
    return f * F32(1 + 0.8 * env) + F32(0.25 * env)


@_reg("beat_bounce", "Beat bounce", "Motion",
      {"intensity": _I(0.6), "bpm": _p("BPM", 120.0, 40.0, 200.0)}, kind="rgba", moves_alpha=True,
      preview_t=0.05)
def _fx_beat_bounce(img, p, t, ch):
    k = p["intensity"]
    ph = (t * p["bpm"] / 60.0) % 1.0
    env = math.exp(-ph * 5.0) if ph > 1 / 6 else math.sin(math.pi / 2 * ph * 6)
    return _zoom_shift(img, 1 + 0.08 * k * env, 0.0, -_px(0.03 * k * env, ch))


@_reg("echo_trail", "Echo trail (ghosting)", "Motion",
      {"intensity": _I(0.6), "copies": _p("Copies", 4, 1, 8), "speed": _p("Speed", 0.5, 0.0, 3.0)},
      kind="rgba", moves_alpha=True)
def _fx_echo_trail(img, p, t, ch):
    k = p["intensity"]
    ph = 2 * math.pi * p["speed"] * _tq(t)
    ang = ph * 0.5 + 0.6
    step = _px(0.04 * k, ch) * (0.6 + 0.4 * math.sin(ph))
    acc = img.astype(F32)
    total = 1.0
    for i in range(1, int(round(p["copies"])) + 1):
        wgt = 0.65 ** i * k
        g = _shift_y(_shift_x(img, -math.cos(ang) * step * i), -math.sin(ang) * step * i)
        acc += g.astype(F32) * F32(wgt)
        total += wgt
    acc *= F32(1 / total)
    acc += F32(0.5)
    return acc.astype(np.uint8)


@_reg("spin", "Spin", "Motion",
      {"intensity": _I(1.0), "speed": _p("Speed (rev/s)", 0.25, -2.0, 2.0), "zoom": _p("Zoom", 1.0, 1.0, 2.5)},
      kind="rgba", moves_alpha=True, preview_t=0.5)
def _fx_spin(img, p, t, ch):
    ang = (2 * math.pi * p["speed"] * p["intensity"] * t) % (2 * math.pi)
    return _affine(img, p["zoom"], ang, mirror=True)


# =============================================================== chroma key
def _hex_rgb(s, default=(0.0, 1.0, 0.0)):
    try:
        s = str(s).lstrip("#")
        if len(s) == 3:
            s = "".join(c * 2 for c in s)
        return tuple(int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except (ValueError, IndexError):
        return default


_CB = np.array([-0.168736, -0.331264, 0.5], F32)
_CR = np.array([0.5, -0.418688, -0.081312], F32)


def new_chroma(color: str = "#00ff00") -> dict:
    return {"color": color, "similarity": 0.3, "smooth": 0.1, "spill": 0.5}


def chroma_key(rgba, ck: dict):
    """Alpha from Cb/Cr distance to the key colour, soft edge, plus spill suppression."""
    key = np.array(_hex_rgb(ck.get("color", "#00ff00")), F32)
    sim = float(ck.get("similarity", 0.3))
    smooth = max(float(ck.get("smooth", 0.1)), 1e-3)
    spill = float(ck.get("spill", 0.5))
    f = _to_f(rgba[..., :3])
    cb = f @ _CB
    cr = f @ _CR
    kcb, kcr = float(key @ _CB), float(key @ _CR)
    dcb, dcr = cb - F32(kcb), cr - F32(kcr)
    d = np.sqrt(dcb * dcb + dcr * dcr)
    a = np.clip((d - F32(sim)) * F32(1 / smooth), 0, 1)
    out = np.empty_like(rgba)
    alpha = rgba[..., 3].astype(F32) * a
    out[..., 3] = (alpha + F32(0.5)).astype(np.uint8)
    if spill > 0:
        kn = math.hypot(kcb, kcr)
        if kn > 1e-4:
            ux, uy = kcb / kn, kcr / kn
            proj = np.maximum(cb * F32(ux) + cr * F32(uy), 0)  # chroma along key direction
            near = np.clip(1 - (d - F32(sim + smooth)) * F32(4.0), 0, 1)
            amt = proj * near * F32(spill)
            # remove that chroma component, keep luma: rgb delta from (dcb, dcr) = -amt*(ux, uy)
            dcb_, dcr_ = -amt * F32(ux), -amt * F32(uy)
            f = f + np.stack([1.402 * dcr_, -0.344136 * dcb_ - 0.714136 * dcr_, 1.772 * dcb_], axis=-1)
        out[..., :3] = _to_u8(f)
    else:
        out[..., :3] = rgba[..., :3]
    return out


# =============================================================== public API
def active(clip) -> bool:
    return bool(clip.chroma) or any(e.get("enabled", True) for e in clip.effects)


def _params(name, given):
    spec = REGISTRY[name]["params"]
    out = {}
    given = given or {}
    for k, ps in spec.items():
        v = given.get(k, ps["default"])
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = float(ps["default"])
        out[k] = min(max(v, ps["min"]), ps["max"])
    return out


def run_effect(name, rgba, params=None, t_local=0.0, canvas_h=None):
    """Apply one registry effect to a uint8 (h, w, 4) frame; raises on unknown names."""
    fn, kind, _ = _IMPL[name]
    p = _params(name, params)
    ch = int(canvas_h or rgba.shape[0])
    if kind == "rgba":
        res = fn(rgba, p, t_local, ch)
        return np.ascontiguousarray(res)
    out = np.empty_like(rgba)
    if kind == "u8":
        out[..., :3] = fn(rgba[..., :3], p, t_local, ch)
    else:
        out[..., :3] = _to_u8(fn(_to_f(rgba[..., :3]), p, t_local, ch))
    out[..., 3] = rgba[..., 3]
    return out


_warned: set = set()


def apply(rgba, clip, t_local: float, canvas_h: int):
    out = rgba
    if clip.chroma:
        try:
            out = chroma_key(out, clip.chroma)
        except Exception as e:  # noqa: BLE001 - never break playback for a bad setting
            if "chroma" not in _warned:
                _warned.add("chroma")
                print(f"[effects] chroma key failed: {e!r}")
    for e in clip.effects:
        if not e.get("enabled", True):
            continue
        name = e.get("type")
        if name not in _IMPL:
            continue
        try:
            out = run_effect(name, out, e.get("params"), t_local, canvas_h)
        except Exception as ex:  # noqa: BLE001
            if name not in _warned:
                _warned.add(name)
                print(f"[effects] {name} failed: {ex!r}")
    return out


def new_effect(name: str) -> dict:
    spec = REGISTRY[name]
    return {"type": name, "params": {k: v["default"] for k, v in spec["params"].items()}, "enabled": True}


CATEGORY_ORDER = ("Filters", "Light & Colour", "Blur & Sharp", "Stylize", "Retro & Glitch", "Motion")


def categories() -> dict:
    out = {c: [] for c in CATEGORY_ORDER}
    for name, spec in REGISTRY.items():
        out.setdefault(spec["category"], []).append(name)
    return out


def preview_card(name: str, rgba_sample):
    """The effect at default params on a sample frame (gallery thumbnail)."""
    _, _, pt = _IMPL[name]
    rgba = np.ascontiguousarray(rgba_sample, dtype=np.uint8)
    return run_effect(name, rgba, None, pt, rgba.shape[0])
