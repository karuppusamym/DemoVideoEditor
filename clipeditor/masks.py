"""Clip masks (rect, ellipse, linear, mirror; feather; invert).

CONTRACT: active(clip) -> bool; apply(rgba, clip, t_local) -> rgba (multiplies alpha channel).
Mask geometry is in clip-normalised coords (0..1 of the clip's own frame).

Mask dict: {"shape", "x", "y", "w", "h", "angle", "feather", "invert", "round"}
    x, y     centre (0..1 of the frame)
    w, h     size (0..1 of frame width / height); for "mirror" only h (band thickness) is used
    angle    degrees, clockwise on screen, about the centre
    feather  0..1 soft edge; 1 = half the frame's shorter side
    invert   keep the outside instead of the inside
    round    rect corner radius, 0..1 of the half of the shorter side
    enabled  optional, default True
Shapes: rect, ellipse, linear (keeps the half-plane above the line through (x, y) at `angle`;
angle 0 = horizontal line, rotate to wipe from other sides), mirror (a band of thickness h
through (x, y) along `angle`).  Several masks multiply.
"""
from __future__ import annotations

import math

import numpy as np

SHAPES = ("rect", "ellipse", "linear", "mirror")
F32 = np.float32

_DEFAULTS = {
    "rect": {"w": 0.6, "h": 0.6, "round": 0.0},
    "ellipse": {"w": 0.6, "h": 0.7},
    "linear": {"w": 1.0, "h": 1.0},
    "mirror": {"w": 1.0, "h": 0.35},
}


def new_mask(shape: str = "ellipse") -> dict:
    if shape not in SHAPES:
        raise ValueError(f"unknown mask shape {shape!r}")
    m = {"shape": shape, "x": 0.5, "y": 0.5, "w": 0.6, "h": 0.6, "angle": 0.0,
         "feather": 0.05, "invert": False, "round": 0.0, "enabled": True}
    m.update(_DEFAULTS[shape])
    return m


def active(clip) -> bool:
    return any(m.get("enabled", True) for m in (clip.masks or []))


def _num(m, k, d):
    try:
        return float(m.get(k, d))
    except (TypeError, ValueError):
        return float(d)


def mask_alpha(m: dict, h: int, w: int) -> np.ndarray:
    """Coverage 0..1 float32 (h, w) for one mask."""
    shape = m.get("shape", "rect")
    cx, cy = _num(m, "x", 0.5) * w, _num(m, "y", 0.5) * h
    a = math.radians(_num(m, "angle", 0.0))
    ca, sa = math.cos(a), math.sin(a)
    X = (np.arange(w, dtype=F32) + F32(0.5 - cx))[None, :]
    Y = (np.arange(h, dtype=F32) + F32(0.5 - cy))[:, None]
    # rotate into mask space (inverse of a clockwise screen rotation)
    u = X * F32(ca) + Y * F32(sa)
    v = Y * F32(ca) - X * F32(sa)
    hw = max(_num(m, "w", 0.6), 0.0) * w / 2
    hh = max(_num(m, "h", 0.6), 0.0) * h / 2
    if shape == "ellipse":
        ea, eb = max(hw, 0.5), max(hh, 0.5)
        k = np.sqrt((u * F32(1 / ea)) ** 2 + (v * F32(1 / eb)) ** 2)
        sdf = (k - 1) * F32(min(ea, eb))
    elif shape == "linear":
        sdf = v + np.zeros_like(u)
    elif shape == "mirror":
        sdf = np.abs(v) - F32(hh) + np.zeros_like(u)
    else:  # rect (rounded)
        r = min(max(_num(m, "round", 0.0), 0.0), 1.0) * min(hw, hh)
        qx = np.abs(u) - F32(hw - r)
        qy = np.abs(v) - F32(hh - r)
        outside = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2)
        sdf = outside + np.minimum(np.maximum(qx, qy), 0) - F32(r)
    feather = max(_num(m, "feather", 0.0), 0.0) * 0.5 * min(w, h)
    fpx = max(feather, 1.0)
    cov = np.clip(F32(0.5) - sdf * F32(1 / fpx), 0, 1)
    if m.get("invert", False):
        cov = 1 - cov
    return cov


def apply(rgba, clip, t_local: float):
    ms = [m for m in (clip.masks or []) if m.get("enabled", True)]
    if not ms:
        return rgba
    h, w = rgba.shape[:2]
    cov = None
    for m in ms:
        c = mask_alpha(m, h, w)
        cov = c if cov is None else cov * c
    out = rgba if rgba.flags.writeable else rgba.copy()
    a = out[..., 3].astype(F32) * cov
    a += F32(0.5)
    out[..., 3] = a.astype(np.uint8)
    return out
