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


CATEGORY_ORDER = ("Filters", "Blur & Sharp", "Stylize", "Retro & Glitch", "Motion")


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
