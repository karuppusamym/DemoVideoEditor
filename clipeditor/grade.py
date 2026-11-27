"""Pro colour grading: wheels, curves, HSL, temperature/tint, exposure, LUT (.cube).

CONTRACT (used by clipeditor.render.Compositor for preview AND render-engine export):
    active(clip) -> bool
    apply(rgba, clip) -> rgba        # uint8 (h, w, 4); changes RGB only, keeps alpha

Grade dict (``Clip.grade``; ``{}`` = off). Every key is optional:
    exposure            stops, -4..4
    temperature, tint   -1..1 (warm / magenta positive)
    contrast            -1..1 (S-curve around mid-grey)
    highlights, shadows, whites, blacks   -1..1
    saturation          -1..1 (global), vibrance -1..1 (weighted toward low-saturation pixels)
    lift, gamma, gain   [r, g, b] (optional 4th = master) offsets, -1..1, default 0
    curves              {"master"|"r"|"g"|"b": [[x, y], ...]} points in 0..1
    hsl                 {band: {"h": deg shift, "s": sat mult delta, "l": lum delta}}
    lut                 path to a .cube file, or a BUILTIN_LUTS name;  lut_mix 0..1

Pipeline (all vectorised numpy, float32 planes):
    1. per-channel stage, baked into a 256-entry LUT per channel and cached by its params:
       exposure + white balance (in ~linear light) -> contrast/tone -> lift/gamma/gain -> curves
    2. HSL bands -> saturation/vibrance -> 3D LUT (trilinear)  (each skipped when neutral)
The neutral grade returns the input untouched.
"""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- UI metadata

WHEEL_KEYS = ("lift", "gamma", "gain")
CURVE_KEYS = ("master", "r", "g", "b")
HSL_BANDS = ("red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta")
HSL_CENTERS = (0.0, 30.0, 60.0, 120.0, 180.0, 240.0, 270.0, 300.0)

BASIC_PARAMS = {
    "exposure":    {"label": "Exposure",    "min": -4.0, "max": 4.0, "default": 0.0},
    "temperature": {"label": "Temperature", "min": -1.0, "max": 1.0, "default": 0.0},
    "tint":        {"label": "Tint",        "min": -1.0, "max": 1.0, "default": 0.0},
    "contrast":    {"label": "Contrast",    "min": -1.0, "max": 1.0, "default": 0.0},
    "highlights":  {"label": "Highlights",  "min": -1.0, "max": 1.0, "default": 0.0},
    "shadows":     {"label": "Shadows",     "min": -1.0, "max": 1.0, "default": 0.0},
    "whites":      {"label": "Whites",      "min": -1.0, "max": 1.0, "default": 0.0},
    "blacks":      {"label": "Blacks",      "min": -1.0, "max": 1.0, "default": 0.0},
    "vibrance":    {"label": "Vibrance",    "min": -1.0, "max": 1.0, "default": 0.0},
    "saturation":  {"label": "Saturation",  "min": -1.0, "max": 1.0, "default": 0.0},
}

LUT_DIR = Path(__file__).resolve().parent / "assets" / "luts"
_BUILTIN_SIZE = 17


def default_grade() -> dict:
    g = {k: spec["default"] for k, spec in BASIC_PARAMS.items()}
    for k in WHEEL_KEYS:
        g[k] = [0.0, 0.0, 0.0]
    g["curves"] = {k: [[0.0, 0.0], [1.0, 1.0]] for k in CURVE_KEYS}
    g["hsl"] = {b: {"h": 0.0, "s": 0.0, "l": 0.0} for b in HSL_BANDS}
    g["lut"] = ""
    g["lut_mix"] = 1.0
    return g


def _num(v, d=0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return d
    return f if math.isfinite(f) else d


def _wheel(v) -> tuple:
    """[r, g, b(, master)] -> (r+m, g+m, b+m)."""
    try:
        vals = [_num(x) for x in list(v)[:4]]
    except TypeError:
        vals = [_num(v)] * 3
    vals += [0.0] * (4 - len(vals))
    m = vals[3]
    return tuple(min(max(x + m, -1.0), 1.0) for x in vals[:3])


def _curve_points(pts) -> tuple:
    out = []
    for p in pts or ():
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            out.append((min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)))
    return tuple(out)


def _curve_is_identity(pts: tuple) -> bool:
    return not pts or (len(pts) >= 2 and all(abs(x - y) < 1e-6 for x, y in pts)
                       and min(x for x, _ in pts) <= 1e-6 and max(x for x, _ in pts) >= 1 - 1e-6)


def _hsl_table(hsl) -> tuple | None:
    """-> (dh[9], ds[9], dl[9]) per band (index 8 wraps to red), or None if neutral."""
    if not isinstance(hsl, dict):
        return None
    dh, ds, dl = np.zeros(9, np.float32), np.zeros(9, np.float32), np.zeros(9, np.float32)
    for i, band in enumerate(HSL_BANDS):
        v = hsl.get(band) or {}
        if not isinstance(v, dict):
            continue
        dh[i] = min(max(_num(v.get("h")), -180.0), 180.0)
        ds[i] = min(max(_num(v.get("s")), -1.0), 1.0)
        dl[i] = min(max(_num(v.get("l")), -1.0), 1.0)
    if not (dh.any() or ds.any() or dl.any()):
        return None
    for a in (dh, ds, dl):
        a[8] = a[0]
    return dh, ds, dl


def _channel_params(g: dict) -> dict:
    p = {k: _num(g.get(k)) for k in ("exposure", "temperature", "tint", "contrast",
                                     "highlights", "shadows", "whites", "blacks")}
    for k in WHEEL_KEYS:
        p[k] = _wheel(g.get(k, (0, 0, 0)))
    curves = g.get("curves") if isinstance(g.get("curves"), dict) else {}
    p["curves"] = {k: _curve_points(curves.get(k)) for k in CURVE_KEYS}
    return p


def _channel_identity(p: dict) -> bool:
    if any(abs(p[k]) > 1e-9 for k in ("exposure", "temperature", "tint", "contrast",
                                      "highlights", "shadows", "whites", "blacks")):
        return False
    if any(abs(x) > 1e-9 for k in WHEEL_KEYS for x in p[k]):
        return False
    return all(_curve_is_identity(v) for v in p["curves"].values())


def _lut_spec(g: dict):
    path = g.get("lut") or ""
    mix = min(max(_num(g.get("lut_mix"), 1.0), 0.0), 1.0)
    if not path or mix <= 0:
        return None
    return str(path), mix


def is_identity(grade) -> bool:
    if not grade or not isinstance(grade, dict):
        return True
    if not _channel_identity(_channel_params(grade)):
        return False
    if abs(_num(grade.get("vibrance"))) > 1e-9 or abs(_num(grade.get("saturation"))) > 1e-9:
        return False
    if _hsl_table(grade.get("hsl")) is not None:
        return False
    return _lut_spec(grade) is None


# ---------------------------------------------------------------- curves

def _fritsch_carlson(xs: np.ndarray, ys: np.ndarray, t: np.ndarray) -> np.ndarray:
    n = len(xs)
    h = np.diff(xs)
    d = np.diff(ys) / h
    m = np.empty(n)
    m[0], m[-1] = d[0], d[-1]
    m[1:-1] = (d[:-1] + d[1:]) / 2
    for k in range(n - 1):
        if d[k] == 0:
            m[k] = m[k + 1] = 0.0
    for k in range(1, n - 1):  # local extrema: flat tangent (no overshoot)
        if d[k - 1] * d[k] < 0:
            m[k] = 0.0
    for k in range(n - 1):
        if d[k] == 0:
            continue
        a, b = m[k] / d[k], m[k + 1] / d[k]
        if a < 0:
            m[k], a = 0.0, 0.0
        if b < 0:
            m[k + 1], b = 0.0, 0.0
        s = a * a + b * b
        if s > 9:
            tau = 3 / math.sqrt(s)
            m[k], m[k + 1] = tau * a * d[k], tau * b * d[k]
    k = np.clip(np.searchsorted(xs, t, side="right") - 1, 0, n - 2)
    hk = h[k]
    u = (t - xs[k]) / hk
    u2, u3 = u * u, u * u * u
    y = ((2 * u3 - 3 * u2 + 1) * ys[k] + (u3 - 2 * u2 + u) * hk * m[k]
         + (-2 * u3 + 3 * u2) * ys[k + 1] + (u3 - u2) * hk * m[k + 1])
    y = np.where(t <= xs[0], ys[0], np.where(t >= xs[-1], ys[-1], y))
    return y


@lru_cache(maxsize=256)
def curve_lut(points: tuple) -> np.ndarray:
    """Monotone cubic (Fritsch-Carlson) through `points` baked into 256 float32 entries.
    Outside the first/last point the curve is held flat at the end values."""
    t = np.linspace(0.0, 1.0, 256)
    pts = sorted(_curve_points(points))
    dedup = {}
    for x, y in pts:
        dedup[round(x, 6)] = y
    xs = np.array(sorted(dedup), np.float64)
    ys = np.array([dedup[x] for x in sorted(dedup)], np.float64)
    if len(xs) == 0:
        y = t
    elif len(xs) == 1:
        y = np.full_like(t, ys[0])
    else:
        y = _fritsch_carlson(xs, ys, t)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def eval_curve(points, x) -> np.ndarray:
    """Evaluate a curve at arbitrary 0..1 values (linear between the 256 baked entries)."""
    lut = curve_lut(_curve_points(points))
    return np.interp(np.asarray(x, np.float64), np.linspace(0, 1, 256), lut).astype(np.float32)


# ---------------------------------------------------------------- per-channel stage

_GAMMA = 2.2


def _tone(x: np.ndarray, p: dict) -> np.ndarray:
    c = p["contrast"]
    if c:
        k = 1.0 + 1.5 * c if c > 0 else 1.0 + c * 0.9   # S-curve slope at mid-grey
        xs = np.clip(x, 0, 1)
        lo, hi = 0.5 * (2 * xs) ** k, 1 - 0.5 * (2 * (1 - xs)) ** k
        x = np.where(xs < 0.5, lo, hi)
    xs = np.clip(x, 0, 1)
    if p["shadows"]:
        x = x + p["shadows"] * 0.6 * xs * (1 - xs) ** 2 * (27 / 16) * 0.55
    if p["highlights"]:
        x = x + p["highlights"] * 0.6 * xs ** 2 * (1 - xs) * (27 / 16) * 0.55
    if p["whites"]:
        x = x + p["whites"] * 0.25 * xs ** 3
    if p["blacks"]:
        x = x + p["blacks"] * 0.2 * (1 - xs) ** 3
    return np.clip(x, 0.0, 1.0)


@lru_cache(maxsize=64)
def _channel_luts_cached(key: str) -> tuple:
    p = json.loads(key)
    x0 = np.arange(256, dtype=np.float64) / 255.0
    temp, tint = p["temperature"], p["tint"]
    # white balance gains in linear light: warm = R up / B down; +tint = magenta (G down)
    wb = (1 + 0.30 * temp + 0.10 * tint, 1 - 0.25 * tint, 1 - 0.30 * temp + 0.10 * tint)
    ev = 2.0 ** p["exposure"]
    curves = {k: tuple(map(tuple, v)) for k, v in p["curves"].items()}
    out = []
    for c in range(3):
        x = x0
        if p["exposure"] or temp or tint:
            x = np.clip((x ** _GAMMA) * ev * wb[c], 0, 1) ** (1 / _GAMMA)
        x = _tone(x, p)
        lift, gam, gain = p["lift"][c], p["gamma"][c], p["gain"][c]
        if lift or gam or gain:
            # ASC-CDL style: out = (in*gain' + lift') ** (1/gamma')
            # lift' = 0.25*lift; gain' = (1+gain)*(1-lift') so lift alone keeps white at 1
            lift_ = 0.25 * lift
            gain_ = (1 + gain) * (1 - lift_)
            gamma_ = 2.0 ** gam
            x = np.clip(x * gain_ + lift_, 0, 1) ** (1 / gamma_)
        grid = np.linspace(0, 1, 256)
        for name in ("master", "rgb"[c]):
            pts = curves.get(name) or ()
            if not _curve_is_identity(pts):
                x = np.interp(x, grid, curve_lut(pts))
        out.append(np.clip(x, 0, 1).astype(np.float32))
    f = np.stack(out)
    u8 = np.clip(np.floor(f * 255.0 + 0.5), 0, 255).astype(np.uint8)
    return f, u8


def channel_luts(grade: dict) -> tuple:
    """(float32 (3,256), uint8 (3,256)) per-channel LUTs for the 1-D part of `grade`."""
    return _channel_luts_cached(json.dumps(_channel_params(grade), sort_keys=True))


# ---------------------------------------------------------------- 3D LUTs (.cube)

class CubeLUT:
    __slots__ = ("size", "table", "flat", "domain_min", "domain_max", "title", "kind")

    def __init__(self, size, table, domain_min, domain_max, title="", kind="3d"):
        self.size = int(size)
        self.table = table                   # 3d: (N,N,N,3) indexed [r,g,b]; 1d: (N,3)
        self.flat = table.reshape(-1, 3)
        self.domain_min = np.asarray(domain_min, np.float32)
        self.domain_max = np.asarray(domain_max, np.float32)
        self.title = title
        self.kind = kind

    def __repr__(self):
        return f"CubeLUT({self.kind}, size={self.size}, title={self.title!r})"


def _parse_cube_text(text: str) -> CubeLUT:
    size3 = size1 = 0
    dmin, dmax = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
    title = ""
    rows = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        head = line.split(None, 1)
        kw = head[0].upper()
        if kw == "TITLE":
            title = head[1].strip().strip('"') if len(head) > 1 else ""
        elif kw == "LUT_3D_SIZE":
            size3 = int(head[1])
        elif kw == "LUT_1D_SIZE":
            size1 = int(head[1])
        elif kw == "DOMAIN_MIN":
            dmin = [float(v) for v in head[1].split()[:3]]
        elif kw == "DOMAIN_MAX":
            dmax = [float(v) for v in head[1].split()[:3]]
        elif kw in ("LUT_3D_INPUT_RANGE", "LUT_1D_INPUT_RANGE"):
            lo, hi = (float(v) for v in head[1].split()[:2])
            dmin, dmax = [lo] * 3, [hi] * 3
        elif kw[0].isdigit() or kw[0] in "-+.":
            vals = line.split()
            if len(vals) >= 3:
                rows.append(vals[:3])
        # unknown keywords are ignored
    if not size3 and not size1:
        raise ValueError("not a .cube LUT: no LUT_3D_SIZE / LUT_1D_SIZE")
    data = np.array(rows, dtype=np.float32)
    if size3:
        if not 2 <= size3 <= 256:
            raise ValueError(f"bad LUT_3D_SIZE {size3}")
        if len(data) != size3 ** 3:
            raise ValueError(f"expected {size3 ** 3} entries, got {len(data)}")
        # file order: R changes fastest -> [b][g][r]; store as [r][g][b]
        table = np.ascontiguousarray(data.reshape(size3, size3, size3, 3).transpose(2, 1, 0, 3))
        return CubeLUT(size3, table, dmin, dmax, title, "3d")
    if len(data) != size1:
        raise ValueError(f"expected {size1} entries, got {len(data)}")
    return CubeLUT(size1, np.ascontiguousarray(data), dmin, dmax, title, "1d")


@lru_cache(maxsize=32)
def _parse_cached(path: str, mtime: float) -> CubeLUT:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return _parse_cube_text(fh.read())


def resolve_lut_path(path: str) -> str:
    if path in BUILTIN_LUTS:
        ensure_builtin_luts()
        return BUILTIN_LUTS[path]
    return os.path.abspath(os.path.expanduser(path))


def parse_cube(path) -> CubeLUT:
    """Parse an Adobe/Resolve .cube file (cached by path + mtime)."""
    p = resolve_lut_path(str(path))
    return _parse_cached(p, os.path.getmtime(p))


def _try_lut(path: str):
    try:
        return parse_cube(path)
    except (OSError, ValueError):
        return None


def apply_lut_planes(r, g, b, lut: CubeLUT):
    """Apply a parsed LUT to float32 planes in 0..1 (trilinear for 3D)."""
    lo, span = lut.domain_min, np.maximum(lut.domain_max - lut.domain_min, 1e-9)
    n = lut.size
    if lut.kind == "1d":
        grid = np.linspace(0, 1, n)
        return tuple(np.interp(np.clip((ch - lo[i]) / span[i], 0, 1), grid, lut.table[:, i])
                     .astype(np.float32) for i, ch in enumerate((r, g, b)))
    idx, frac = [], []
    for i, ch in enumerate((r, g, b)):
        s = np.clip((ch - lo[i]) * ((n - 1) / span[i]), 0, n - 1)
        i0 = np.minimum(s.astype(np.int32), n - 2)
        frac.append((s - i0).astype(np.float32)[..., None])
        idx.append(i0)
    base = (idx[0] * n + idx[1]) * n + idx[2]
    T = lut.flat
    nn, N = n * n, n
    fr, fg, fb = frac

    def lerp(a, c, f):
        return a + (c - a) * f

    c00 = lerp(T[base], T[base + 1], fb)
    c01 = lerp(T[base + N], T[base + N + 1], fb)
    c10 = lerp(T[base + nn], T[base + nn + 1], fb)
    c11 = lerp(T[base + nn + N], T[base + nn + N + 1], fb)
    out = lerp(lerp(c00, c01, fg), lerp(c10, c11, fg), fr)
    return out[..., 0], out[..., 1], out[..., 2]


# ---------------------------------------------------------------- cross-channel stages

_CENTERS9 = np.array(HSL_CENTERS + (360.0,), np.float32)
_WIDTHS = np.diff(_CENTERS9)


def _apply_hsl(r, g, b, table):
    dh_t, ds_t, dl_t = table
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    c = mx - mn
    cs = np.where(c > 1e-6, c, 1.0)
    hue = np.where(mx == r, np.mod((g - b) / cs, 6.0),
                   np.where(mx == g, (b - r) / cs + 2.0, (r - g) / cs + 4.0)) * 60.0
    seg = np.clip(np.searchsorted(_CENTERS9, hue, side="right") - 1, 0, 7)
    t = (hue - _CENTERS9[seg]) / _WIDTHS[seg]
    s = t * t * (3 - 2 * t)
    dh = dh_t[seg] * (1 - s) + dh_t[seg + 1] * s if dh_t.any() else None
    ds = ds_t[seg] * (1 - s) + ds_t[seg + 1] * s if ds_t.any() else None
    dl = dl_t[seg] * (1 - s) + dl_t[seg + 1] * s if dl_t.any() else None
    changed = c > 1e-6
    zero = np.zeros_like(r)
    any_d = np.zeros(r.shape, bool)
    for d in (dh, ds, dl):
        if d is not None:
            any_d |= d != 0
    changed &= any_d
    if not changed.any():
        return r, g, b
    L = (mx + mn) * 0.5
    den = 1.0 - np.abs(2.0 * L - 1.0)
    S = np.where(den > 1e-6, c / np.where(den > 1e-6, den, 1.0), 0.0)
    # near-greys have no meaningful hue: fade the effect in over chroma 0.035..0.2 (the lower
    # bound also keeps greys exact through the baked table's 8-code cells)
    amt = np.clip((c - 0.035) / 0.165, 0.0, 1.0)
    amt = amt * amt * (3 - 2 * amt)
    changed &= amt > 0
    H = hue + (dh * amt if dh is not None else zero)
    if ds is not None:
        S = np.clip(S * (1.0 + ds * amt), 0.0, 1.0)
    if dl is not None:
        L = np.clip(L + dl * 0.5 * amt * np.minimum(S * 2.0, 1.0), 0.0, 1.0)
    a = S * np.minimum(L, 1.0 - L)
    h12 = H / 30.0
    res = []
    for n, old in ((0.0, r), (8.0, g), (4.0, b)):
        k = np.mod(n + h12, 12.0)
        v = L - a * np.clip(np.minimum(k - 3.0, 9.0 - k), -1.0, 1.0)
        res.append(np.where(changed, v, old).astype(np.float32))
    return tuple(res)


def _apply_saturation(r, g, b, sat, vib):
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    f = np.float32(1.0 + sat)
    if vib:
        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        s = (mx - mn) / np.maximum(mx, 1e-4)
        w = (1.0 - s) ** 2 if vib > 0 else (1.0 - s)
        f = f * (1.0 + vib * w)
    return tuple(np.clip(y + (ch - y) * f, 0.0, 1.0) for ch in (r, g, b))


# ---------------------------------------------------------------- apply
#
# The cross-channel stages (HSL, saturation/vibrance, 3D LUT) are baked, per grade, into one
# table over the post-curve RGB codes: 33 nodes on R and G (every 8 codes; the last cell is
# 248..255) x all 256 codes on B. Each entry packs three 16-bit fields (value*4 + 2) into a
# uint64, so a frame needs 4 gathers and two packed integer lerps (exact modulo 2**64 because
# the operations are linear and each final field stays in range); the results land on byte
# boundaries and are read back as bytes 1/3/5. Large frames are split across threads.

_NODES = np.array(list(range(0, 256, 8)) + [255], np.int64)       # 33 node codes
_NN = len(_NODES)
_ROW_G, _ROW_R = 256, 256 * _NN


def _cell_tables():
    v = np.arange(256)
    i = np.minimum(v // 8, _NN - 2)
    f = np.where(i < _NN - 2, v - 8 * i, np.rint((v - 248) * 8 / 7)).astype(np.int64)
    return i, f


_CELL_I, _CELL_F = _cell_tables()


def _eval_cross(r, g, b, grade: dict, lut):
    """HSL -> saturation/vibrance -> 3D LUT on float planes (used to bake the table)."""
    hsl = _hsl_table(grade.get("hsl"))
    sat = min(max(_num(grade.get("saturation")), -1.0), 1.0)
    vib = min(max(_num(grade.get("vibrance")), -1.0), 1.0)
    if hsl is not None:
        r, g, b = _apply_hsl(r, g, b, hsl)
    if sat or vib:
        r, g, b = _apply_saturation(r, g, b, sat, vib)
    if lut is not None:
        mix = np.float32(min(max(_num(grade.get("lut_mix"), 1.0), 0.0), 1.0))
        lr, lg, lb = apply_lut_planes(r, g, b, lut)
        r, g, b = r + (lr - r) * mix, g + (lg - g) * mix, b + (lb - b) * mix
    return r, g, b


@lru_cache(maxsize=16)
def _cross_table_cached(key: str, lut_path: str, lut_mtime: float) -> np.ndarray:
    grade = json.loads(key)
    lut = _parse_cached(lut_path, lut_mtime) if lut_path else None
    t = (_NODES / 255.0).astype(np.float32)
    R, G, B = np.meshgrid(t, t, t, indexing="ij")
    r, g, b = _eval_cross(R, G, B, grade, lut)
    s = np.stack([np.clip(np.asarray(c, np.float64), 0, 1) * 1020.0 for c in (r, g, b)], -1)
    # upsample B from 33 nodes to all 256 codes (exact at the nodes)
    bi = _CELL_I
    bf = ((np.arange(256) - _NODES[bi]) / np.diff(_NODES)[bi])[None, None, :, None]
    s = s[:, :, bi] * (1 - bf) + s[:, :, bi + 1] * bf
    q = np.rint(s).astype(np.uint64) + np.uint64(2)
    packed = q[..., 0] | (q[..., 1] << np.uint64(16)) | (q[..., 2] << np.uint64(32))
    return np.ascontiguousarray(packed.reshape(-1))


@lru_cache(maxsize=64)
def _index_tables(ulut_bytes: bytes) -> tuple:
    u = np.frombuffer(ulut_bytes, np.uint8).reshape(3, 256).astype(np.int64)
    ir = (_CELL_I[u[0]] * _ROW_R).astype(np.intp)
    ig = (_CELL_I[u[1]] * _ROW_G).astype(np.intp)
    ib = u[2].astype(np.intp)
    fr = _CELL_F[u[0]].astype(np.uint64)
    fg = _CELL_F[u[1]].astype(np.uint64)
    return ir, ig, ib, fr, fg


_THREE = np.uint64(3)


def _apply_packed(src, dst, table, tabs):
    ir, ig, ib, fr, fg = tabs
    r, g, b = src[..., 0], src[..., 1], src[..., 2]
    base = ir[r]
    base += ig[g]
    base += ib[b]
    wg = fg[g]

    def glerp(off):  # lerp along G at R-node offset `off` -> fields * 8
        lo = table[off:][base] if off else table[base]
        hi = table[off + _ROW_G:][base]
        hi -= lo
        hi *= wg
        lo <<= _THREE
        lo += hi
        return lo

    lo = glerp(0)
    hi = glerp(_ROW_R)
    hi -= lo
    hi *= fr[r]
    lo <<= _THREE
    lo += hi                                             # fields = (value*4 + 2) * 64
    by = lo.view(np.uint8).reshape(lo.shape + (8,))
    dst[..., 0] = by[..., 1]
    dst[..., 1] = by[..., 3]
    dst[..., 2] = by[..., 5]


_POOL = None
_PAR_MIN = 120_000


def _workers() -> int:
    return max(1, min(4, os.cpu_count() or 1))


def _pool():
    global _POOL
    if _POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _POOL = ThreadPoolExecutor(max_workers=_workers(), thread_name_prefix="grade")
    return _POOL


def _run_rows(fn, src, dst, *args):
    h = src.shape[0]
    n = _workers()
    if h * src.shape[1] < _PAR_MIN or n == 1 or h < 2 * n:
        fn(src, dst, *args)
        return
    cuts = np.linspace(0, h, n + 1).astype(int)
    futs = [_pool().submit(fn, src[a:b], dst[a:b], *args) for a, b in zip(cuts[:-1], cuts[1:])]
    for f in futs:
        f.result()


def _apply_channels(src, dst, ulut):
    for c in range(3):
        dst[..., c] = ulut[c][src[..., c]]


def apply_to_image(rgba: np.ndarray, grade: dict) -> np.ndarray:
    """Grade a uint8 (h, w, 4) image with a grade dict; RGB only, alpha kept. The neutral
    grade returns `rgba` itself (unchanged); otherwise a new array is returned."""
    if not grade or not isinstance(grade, dict) or is_identity(grade):
        return rgba
    _, ulut = channel_luts(grade)
    cross = (_hsl_table(grade.get("hsl")) is not None or abs(_num(grade.get("vibrance"))) > 1e-9
             or abs(_num(grade.get("saturation"))) > 1e-9)
    spec = _lut_spec(grade)
    lut_path, lut_mtime = "", 0.0
    if spec:
        try:
            lut_path = resolve_lut_path(spec[0])
            lut_mtime = os.path.getmtime(lut_path)
            _parse_cached(lut_path, lut_mtime)
            cross = True
        except (OSError, ValueError):   # missing / broken LUT: grade without it
            lut_path, lut_mtime = "", 0.0
    out = np.empty_like(rgba)
    out[..., 3] = rgba[..., 3]
    if not cross:
        _run_rows(_apply_channels, rgba, out, ulut)
        return out
    xkey = {k: grade.get(k) for k in ("hsl", "saturation", "vibrance", "lut_mix")}
    table = _cross_table_cached(json.dumps(xkey, sort_keys=True, default=str),
                                lut_path, lut_mtime)
    _run_rows(_apply_packed, rgba, out, table, _index_tables(ulut.tobytes()))
    return out


def active(clip) -> bool:
    g = getattr(clip, "grade", None)
    return bool(g) and not is_identity(g)


def apply(rgba, clip):
    return apply_to_image(rgba, getattr(clip, "grade", None) or {})


# ---------------------------------------------------------------- built-in LUTs

def _luma(r, g, b):
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _scurve(x, k):
    x = np.clip(x, 0, 1)
    return np.where(x < 0.5, 0.5 * (2 * x) ** k, 1 - 0.5 * (2 * (1 - x)) ** k)


def _sat(rgb, f):
    y = _luma(*rgb)
    return [y + (c - y) * f for c in rgb]


def _split(rgb, shadow, highlight, amount):
    y = _luma(*rgb)
    sh = np.asarray(shadow) - np.mean(shadow)
    hi = np.asarray(highlight) - np.mean(highlight)
    ws, wh = (1 - y) ** 2, y ** 2
    return [c + amount * (ws * sh[i] + wh * hi[i]) for i, c in enumerate(rgb)]


def _fade(rgb, black, white):
    return [black + c * (white - black) for c in rgb]


def _look_teal_orange(r, g, b):
    rgb = _split([r, g, b], (0.0, 0.55, 0.65), (1.0, 0.6, 0.25), 0.35)
    rgb = [_scurve(c, 1.25) for c in rgb]
    return _sat(rgb, 1.15)


def _look_film_warm(r, g, b):
    rgb = _split([r, g, b], (0.55, 0.45, 0.3), (1.0, 0.8, 0.55), 0.22)
    rgb = [_scurve(c, 1.15) for c in rgb]
    return _fade(_sat(rgb, 0.9), 0.035, 0.97)


def _look_film_cool(r, g, b):
    rgb = _split([r, g, b], (0.2, 0.4, 0.6), (0.8, 0.9, 1.0), 0.25)
    rgb = [_scurve(c, 1.15) for c in rgb]
    return _fade(_sat(rgb, 0.85), 0.03, 0.97)


def _look_bleach_bypass(r, g, b):
    rgb = _sat([r, g, b], 0.45)
    rgb = [_scurve(c, 1.7) for c in rgb]
    return _fade(rgb, 0.02, 1.0)


def _look_day_for_night(r, g, b):
    rgb = [c ** 2.2 * 0.3 for c in (r, g, b)]          # ~ -1.7 stops in linear
    rgb = [c ** (1 / 2.2) for c in rgb]
    rgb = _sat(rgb, 0.4)
    gains = (0.7, 0.9, 1.35)
    return [_scurve(c * k, 1.15) for c, k in zip(rgb, gains)]


def _look_vintage_fade(r, g, b):
    rgb = _split([r, g, b], (0.3, 0.45, 0.35), (1.0, 0.85, 0.6), 0.25)
    rgb = _sat(rgb, 0.72)
    return _fade(rgb, 0.12, 0.92)


def _look_high_contrast_bw(r, g, b):
    y = _scurve(0.3 * r + 0.6 * g + 0.1 * b, 1.9)
    return [y, y, y]


def _look_pastel(r, g, b):
    rgb = _sat([r, g, b], 0.6)
    rgb = _split(rgb, (0.5, 0.75, 0.9), (1.0, 0.8, 0.85), 0.15)
    return _fade([_scurve(c, 0.85) for c in rgb], 0.18, 1.0)


_LOOKS = {
    "teal_orange": ("Teal & Orange", _look_teal_orange),
    "film_warm": ("Film Warm", _look_film_warm),
    "film_cool": ("Film Cool", _look_film_cool),
    "bleach_bypass": ("Bleach Bypass", _look_bleach_bypass),
    "day_for_night": ("Day for Night", _look_day_for_night),
    "vintage_fade": ("Vintage Fade", _look_vintage_fade),
    "high_contrast_bw": ("High Contrast B&W", _look_high_contrast_bw),
    "pastel": ("Pastel", _look_pastel),
}
BUILTIN_LUTS = {k: str(LUT_DIR / f"{k}.cube") for k in _LOOKS}
BUILTIN_LUT_LABELS = {k: v[0] for k, v in _LOOKS.items()}


def write_cube(path, fn, size: int = _BUILTIN_SIZE, title: str = "") -> str:
    """Sample fn(r, g, b) -> [r, g, b] (arrays in 0..1) on a size^3 grid into a .cube file."""
    t = np.linspace(0.0, 1.0, size)
    b, g, r = np.meshgrid(t, t, t, indexing="ij")   # r fastest in file order
    out = np.stack([np.clip(np.asarray(c, np.float64), 0, 1) for c in fn(r, g, b)], -1)
    lines = [f'TITLE "{title or Path(path).stem}"', "# generated by clipeditor.grade",
             f"LUT_3D_SIZE {size}", "DOMAIN_MIN 0.0 0.0 0.0", "DOMAIN_MAX 1.0 1.0 1.0"]
    lines += [f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in out.reshape(-1, 3)]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(tmp, path)
    return str(path)


def ensure_builtin_luts(force: bool = False) -> dict:
    """Generate any missing built-in .cube files; returns BUILTIN_LUTS {name: path}."""
    for key, (label, fn) in _LOOKS.items():
        p = BUILTIN_LUTS[key]
        if force or not os.path.exists(p):
            try:
                write_cube(p, fn, title=label)
            except OSError:
                pass
    return dict(BUILTIN_LUTS)
