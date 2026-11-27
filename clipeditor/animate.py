"""Keyframes + In/Out/Loop animation presets -> per-frame clip properties.

CONTRACT (used by clipeditor.render.Compositor and clipeditor.audio_mix):
    resolve(clip, t) -> Props      # t = timeline seconds; combines base values, keyframes, animations
    volume_at(clip, t) -> float    # base volume x volume keyframes (audio); t = timeline seconds
    active(clip) -> bool           # clip needs per-frame evaluation
    ANIMATIONS = {"in": {name: label}, "out": {...}, "loop": {...}}
    PRESET_INFO = {"in": {name: tooltip}, "out": {...}, "loop": {...}}
    KEYFRAME_PROPS, EASINGS
Transitions (clipeditor.transitions) are applied separately by the compositor.

Keyframes: clip.keyframes[prop] = [[t_local, value, easing], ...] with t_local = seconds from the
clip start. The easing stored on a keyframe shapes the segment that LEAVES it (towards the next
keyframe); "hold" keeps its value until the next keyframe. Before the first keyframe the first
value is used, after the last the last value. A prop with keyframes ignores its base value
(except volume, see volume_at).

Composition order per frame:  keyframed/base value  ->  in-anim  ->  out-anim  ->  loop
    scale *= ..., pos_x/pos_y += ... (fractions of canvas W/H), rotation += ... (deg), opacity *= ...
Everything is a pure deterministic function of t, so preview and export match exactly.

UI helpers (t_local = seconds from clip start):
    set_keyframe(clip, prop, t_local, value, easing="ease")
    remove_keyframe(clip, prop, t_local) -> bool
    keyframe_at(clip, prop, t_local) -> [t, value, easing] | None
    all_keyframe_times(clip) -> sorted unique list of t_local
    clear_keyframes(clip, prop=None)
    value_at(clip, prop, t_local) -> float   (keyframed or base value, no presets)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

KEYFRAME_PROPS = ("scale", "pos_x", "pos_y", "rotation", "opacity", "volume")
EASINGS = ("linear", "ease_in", "ease_out", "ease", "hold", "bounce", "elastic")
KF_TOLERANCE = 1 / 60  # keyframes closer than one frame are "the same" keyframe


@dataclass
class Props:
    scale: float
    pos_x: float
    pos_y: float
    rotation: float
    opacity: float


# ---- easing ------------------------------------------------------------------
def _bounce_out(u: float) -> float:
    n1, d1 = 7.5625, 2.75
    if u < 1 / d1:
        return n1 * u * u
    if u < 2 / d1:
        u -= 1.5 / d1
        return n1 * u * u + 0.75
    if u < 2.5 / d1:
        u -= 2.25 / d1
        return n1 * u * u + 0.9375
    u -= 2.625 / d1
    return n1 * u * u + 0.984375


def _elastic_out(u: float) -> float:
    if u <= 0:
        return 0.0
    if u >= 1:
        return 1.0
    return 2 ** (-10 * u) * math.sin((u * 10 - 0.75) * (2 * math.pi / 3)) + 1


def _back_out(u: float, s: float = 1.70158) -> float:
    u -= 1
    return u * u * ((s + 1) * u + s) + 1


def ease(u: float, kind: str = "linear") -> float:
    """Map progress u in [0,1] -> eased progress (bounce/elastic overshoot past 1)."""
    u = 0.0 if u <= 0 else 1.0 if u >= 1 else u
    if kind == "ease_in":
        return u * u * u
    if kind == "ease_out":
        return 1 - (1 - u) ** 3
    if kind == "ease":
        return 4 * u ** 3 if u < 0.5 else 1 - (-2 * u + 2) ** 3 / 2
    if kind == "hold":
        return 1.0 if u >= 1 else 0.0
    if kind == "bounce":
        return _bounce_out(u)
    if kind == "elastic":
        return _elastic_out(u)
    if kind == "back":
        return _back_out(u)
    return u


# ---- keyframes -----------------------------------------------------------------
def _norm(k) -> tuple[float, float, str]:
    t = float(k[0])
    v = float(k[1])
    e = str(k[2]) if len(k) > 2 and k[2] else "linear"
    return t, v, e


def _keys(clip, prop: str) -> list[tuple[float, float, str]]:
    kfs = (getattr(clip, "keyframes", None) or {}).get(prop) or []
    out = []
    for k in kfs:
        try:
            out.append(_norm(k))
        except (TypeError, ValueError, IndexError):
            continue
    out.sort(key=lambda k: k[0])
    return out


def _interp(keys: list[tuple[float, float, str]], t: float) -> float:
    if t <= keys[0][0]:
        return keys[0][1]
    if t >= keys[-1][0]:
        return keys[-1][1]
    # linear scan is fine: keyframe lists are short
    for (t0, v0, e0), (t1, v1, _e1) in zip(keys, keys[1:]):
        if t0 <= t < t1:
            span = t1 - t0
            if span <= 1e-12:
                return v1
            return v0 + (v1 - v0) * ease((t - t0) / span, e0)
    return keys[-1][1]


def value_at(clip, prop: str, t_local: float) -> float:
    """Keyframed value of `prop` at t_local (seconds from clip start), else the clip's base value.
    No in/out/loop presets applied (this is what the inspector shows / edits)."""
    keys = _keys(clip, prop)
    if keys:
        return _interp(keys, t_local)
    return float(getattr(clip, prop))


def set_keyframe(clip, prop: str, t_local: float, value: float, easing: str = "ease") -> list:
    """Add a keyframe, or replace the one within 1/60 s of t_local. Returns the keyframe."""
    if prop not in KEYFRAME_PROPS:
        raise ValueError(f"not a keyframeable property: {prop}")
    if easing not in EASINGS:
        easing = "linear"
    if clip.keyframes is None:
        clip.keyframes = {}
    lst = clip.keyframes.setdefault(prop, [])
    t_local = max(float(t_local), 0.0)
    for k in lst:
        if abs(float(k[0]) - t_local) < KF_TOLERANCE:
            new = [float(k[0]), float(value), easing]
            lst[lst.index(k)] = new
            break
    else:
        new = [t_local, float(value), easing]
        lst.append(new)
    lst.sort(key=lambda k: float(k[0]))
    return new


def remove_keyframe(clip, prop: str, t_local: float) -> bool:
    """Remove the keyframe within 1/60 s of t_local. Returns True if one was removed."""
    lst = (clip.keyframes or {}).get(prop)
    if not lst:
        return False
    for i, k in enumerate(lst):
        if abs(float(k[0]) - t_local) < KF_TOLERANCE:
            del lst[i]
            if not lst:
                del clip.keyframes[prop]
            return True
    return False


def keyframe_at(clip, prop: str, t_local: float):
    """The keyframe [t, value, easing] within 1/60 s of t_local, or None."""
    for k in (clip.keyframes or {}).get(prop) or []:
        if abs(float(k[0]) - t_local) < KF_TOLERANCE:
            return k
    return None


def all_keyframe_times(clip) -> list[float]:
    """Sorted unique keyframe times (t_local) over all props, merged within 1/60 s."""
    ts = sorted(float(k[0]) for lst in (clip.keyframes or {}).values() for k in (lst or []))
    out: list[float] = []
    for t in ts:
        if not out or t - out[-1] >= KF_TOLERANCE:
            out.append(t)
    return out


def clear_keyframes(clip, prop: str | None = None) -> None:
    if not clip.keyframes:
        return
    if prop is None:
        clip.keyframes.clear()
    else:
        clip.keyframes.pop(prop, None)


# ---- presets -----------------------------------------------------------------
# An in-preset is f(u) -> (scale_mul, dx, dy, drot, opacity_mul), u = 0 (start) .. 1 (done);
# f(1) must be the identity (1, 0, 0, 0, 1).
IDENT = (1.0, 0.0, 0.0, 0.0, 1.0)


def _in_fade(u):
    return 1.0, 0.0, 0.0, 0.0, ease(u, "linear")


def _slide(dx0, dy0):
    def f(u):
        r = 1 - ease(u, "ease_out")
        return 1.0, dx0 * r, dy0 * r, 0.0, min(1.0, u * 3)
    return f


def _in_zoom_in(u):  # grows from nothing
    e = ease(u, "ease_out")
    return e, 0.0, 0.0, 0.0, min(1.0, u * 2)


def _in_zoom_out(u):  # shrinks down from big
    e = ease(u, "ease_out")
    return 2.0 - e, 0.0, 0.0, 0.0, min(1.0, u * 2)


def _in_pop(u):
    return max(_back_out(u, 2.5), 0.0), 0.0, 0.0, 0.0, min(1.0, u * 4)


def _in_bounce(u):
    return _bounce_out(u), 0.0, 0.0, 0.0, min(1.0, u * 4)


def _in_elastic(u):
    return max(_elastic_out(u), 0.0), 0.0, 0.0, 0.0, min(1.0, u * 4)


def _in_spin(u):
    e = ease(u, "ease_out")
    return e, 0.0, 0.0, -360.0 * (1 - e), min(1.0, u * 2)


def _in_drop(u):
    return 1.0, 0.0, -1.0 * (1 - _bounce_out(u)), 0.0, min(1.0, u * 5)


def _in_swing(u):
    return 1.0, 0.0, 0.0, -30.0 * (1 - _elastic_out(u)), min(1.0, u * 3)


def _in_rise(u):
    e = ease(u, "ease_out")
    return 1.0, 0.0, 0.12 * (1 - e), 0.0, e


def _in_flip(u):  # 2-D stand-in for a 3-D flip: quarter turn + scale up
    e = ease(u, "ease_out")
    return 0.3 + 0.7 * e, 0.0, 0.0, 90.0 * (1 - e), min(1.0, u * 2)


def _in_roll(u):  # rolls in from the left, rotating
    e = ease(u, "ease_out")
    return 1.0, -1.0 * (1 - e), 0.0, -360.0 * (1 - e), min(1.0, u * 3)


_IN = {
    "fade": ("Fade", _in_fade, "Fade in from transparent"),
    "slide_left": ("Slide from Left", _slide(-1.0, 0.0), "Slide in from the left edge"),
    "slide_right": ("Slide from Right", _slide(1.0, 0.0), "Slide in from the right edge"),
    "slide_top": ("Slide from Top", _slide(0.0, -1.0), "Slide in from the top edge"),
    "slide_bottom": ("Slide from Bottom", _slide(0.0, 1.0), "Slide in from the bottom edge"),
    "zoom_in": ("Zoom In", _in_zoom_in, "Grow from nothing to full size"),
    "zoom_out": ("Zoom Out", _in_zoom_out, "Shrink from double size down to full size"),
    "pop": ("Pop", _in_pop, "Scale up quickly with a small overshoot"),
    "bounce": ("Bounce In", _in_bounce, "Scale up with a bouncy landing"),
    "elastic": ("Elastic", _in_elastic, "Spring into place with an elastic wobble"),
    "spin": ("Spin In", _in_spin, "Spin a full turn while growing in"),
    "drop": ("Drop", _in_drop, "Fall from above and bounce on landing"),
    "swing": ("Swing", _in_swing, "Swing into place like a hanging sign"),
    "rise": ("Rise", _in_rise, "Drift up slightly while fading in"),
    "flip": ("Flip In", _in_flip, "Quarter-turn flip while scaling up"),
    "roll": ("Roll In", _in_roll, "Roll in from the left, rotating"),
}


def _out_from_in(fn):
    """Mirror an in-preset: out progress u (0 -> 1) plays the in-preset backwards."""
    return lambda u: fn(1.0 - u)


def _out_drop(u):  # fall down off screen
    e = ease(u, "ease_in")
    return 1.0, 0.0, 1.2 * e, 8.0 * e, 1.0 - max(0.0, u - 0.7) / 0.3


_OUT = {
    "fade": ("Fade", _out_from_in(_in_fade), "Fade out to transparent"),
    "slide_left": ("Slide to Left", _out_from_in(_slide(-1.0, 0.0)), "Slide out through the left edge"),
    "slide_right": ("Slide to Right", _out_from_in(_slide(1.0, 0.0)), "Slide out through the right edge"),
    "slide_top": ("Slide to Top", _out_from_in(_slide(0.0, -1.0)), "Slide out through the top edge"),
    "slide_bottom": ("Slide to Bottom", _out_from_in(_slide(0.0, 1.0)), "Slide out through the bottom edge"),
    "zoom_in": ("Zoom In", _out_from_in(_in_zoom_out), "Grow to double size while fading out"),
    "zoom_out": ("Zoom Out", _out_from_in(_in_zoom_in), "Shrink away to nothing"),
    "pop": ("Pop", _out_from_in(_in_pop), "Swell slightly, then pop away"),
    "bounce": ("Bounce Out", _out_from_in(_in_bounce), "Bounce, then shrink away"),
    "elastic": ("Elastic", _out_from_in(_in_elastic), "Wobble, then spring away"),
    "spin": ("Spin Out", _out_from_in(_in_spin), "Spin a full turn while shrinking away"),
    "drop": ("Drop", _out_drop, "Fall down off the bottom of the screen"),
    "swing": ("Swing", _out_from_in(_in_swing), "Swing away like a hanging sign"),
    "sink": ("Sink", _out_from_in(_in_rise), "Drift down slightly while fading out"),
    "flip": ("Flip Out", _out_from_in(_in_flip), "Quarter-turn flip while shrinking"),
    "roll": ("Roll Out", _out_from_in(_in_roll), "Roll out to the left, rotating"),
}

_TAU = 2 * math.pi


def _hash01(n: int) -> float:
    """Deterministic pseudo-random in [0,1) from an integer."""
    x = (n * 374761393 + 668265263) & 0xFFFFFFFF
    x = ((x ^ (x >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((x ^ (x >> 16)) & 0xFFFFFF) / float(0x1000000)


# A loop preset is f(tau) -> deltas, tau = seconds from clip start x anim_loop_speed; 1 cycle ~ 1 s.
def _lp_pulse(tau):
    return 1 + 0.06 * math.sin(_TAU * tau), 0.0, 0.0, 0.0, 1.0


def _lp_wiggle(tau):
    return 1.0, 0.004 * math.sin(_TAU * tau * 3), 0.0, 5.0 * math.sin(_TAU * tau * 2), 1.0


def _lp_shake(tau):
    dx = 0.008 * (math.sin(_TAU * tau * 7.0) + 0.5 * math.sin(_TAU * tau * 13.3)) / 1.5
    dy = 0.008 * (math.sin(_TAU * tau * 8.7 + 1.3) + 0.5 * math.sin(_TAU * tau * 11.1)) / 1.5
    return 1.0, dx, dy, 0.0, 1.0


def _lp_float(tau):
    return 1.0, 0.0, 0.02 * math.sin(_TAU * tau * 0.5), 0.0, 1.0


def _lp_spin(tau):
    return 1.0, 0.0, 0.0, (360.0 * tau * 0.5) % 360.0, 1.0


def _lp_heartbeat(tau):
    ph = tau % 1.0
    b = 0.0
    for c in (0.10, 0.30):  # two quick beats per cycle
        b += math.exp(-((ph - c) / 0.05) ** 2)
    return 1 + 0.1 * min(b, 1.0), 0.0, 0.0, 0.0, 1.0


def _lp_swing(tau):
    return 1.0, 0.0, 0.0, 10.0 * math.sin(_TAU * tau * 0.75), 1.0


def _lp_flicker(tau):
    n = int(math.floor(tau * 12))
    r = _hash01(n)
    return 1.0, 0.0, 0.0, 0.0, 0.35 if r < 0.12 else 0.75 + 0.25 * r


def _lp_breathe(tau):
    s = math.sin(_TAU * tau * 0.25)
    return 1 + 0.04 * s, 0.0, 0.0, 0.0, 0.9 + 0.1 * (s + 1) / 2


def _lp_jello(tau):
    ph = tau % 1.0
    return 1.0, 0.0, 0.0, 8.0 * math.exp(-4 * ph) * math.sin(_TAU * ph * 4), 1.0


_LOOP = {
    "pulse": ("Pulse", _lp_pulse, "Gently grow and shrink"),
    "wiggle": ("Wiggle", _lp_wiggle, "Playful side-to-side tilt"),
    "shake": ("Shake", _lp_shake, "Rapid small jitter"),
    "float": ("Float", _lp_float, "Bob slowly up and down"),
    "spin": ("Spin", _lp_spin, "Rotate continuously (one turn every 2 s)"),
    "heartbeat": ("Heartbeat", _lp_heartbeat, "Double-beat pulse, once per second"),
    "swing": ("Swing", _lp_swing, "Pendulum-like rotation"),
    "flicker": ("Flicker", _lp_flicker, "Irregular opacity flicker like a faulty light"),
    "breathe": ("Breathe", _lp_breathe, "Slow scale and opacity swell"),
    "jello": ("Jello", _lp_jello, "Wobbly rotation that settles each second"),
}

_FUNCS = {"in": _IN, "out": _OUT, "loop": _LOOP}
ANIMATIONS: dict[str, dict[str, str]] = {k: {n: v[0] for n, v in d.items()} for k, d in _FUNCS.items()}
PRESET_INFO: dict[str, dict[str, str]] = {k: {n: v[2] for n, v in d.items()} for k, d in _FUNCS.items()}


def preset_deltas(kind: str, name: str, x: float) -> tuple[float, float, float, float, float]:
    """Raw preset output (scale_mul, dx, dy, drot, opacity_mul). x = progress u (in/out) or tau (loop)."""
    entry = _FUNCS.get(kind, {}).get(name)
    if entry is None:
        return IDENT
    return entry[1](x)


# ---- per-frame resolve ---------------------------------------------------------
def resolve(clip, t: float) -> Props:
    tl = t - clip.start
    s = value_at(clip, "scale", tl)
    x = value_at(clip, "pos_x", tl)
    y = value_at(clip, "pos_y", tl)
    r = value_at(clip, "rotation", tl)
    o = value_at(clip, "opacity", tl)

    dur = max(clip.duration, 1e-6)
    layers = []
    if clip.anim_in and clip.anim_in_dur > 0:
        din = min(clip.anim_in_dur, dur)
        layers.append(preset_deltas("in", clip.anim_in, min(max(tl / din, 0.0), 1.0)))
    if clip.anim_out and clip.anim_out_dur > 0:
        dout = min(clip.anim_out_dur, dur)
        u = (tl - (dur - dout)) / dout
        if u > 0:
            layers.append(preset_deltas("out", clip.anim_out, min(u, 1.0)))
    if clip.anim_loop:
        layers.append(preset_deltas("loop", clip.anim_loop, max(tl, 0.0) * (clip.anim_loop_speed or 1.0)))
    for ds, dx, dy, dr, do in layers:
        s *= ds
        x += dx
        y += dy
        r += dr
        o *= do
    vals = [s, x, y, r, o]
    if not all(math.isfinite(v) for v in vals):
        return Props(clip.scale, clip.pos_x, clip.pos_y, clip.rotation, clip.opacity)
    return Props(max(s, 0.0), x, y, r, min(max(o, 0.0), 1.0))


def volume_at(clip, t: float) -> float:
    """Base volume, or the volume keyframes when present (they override it, like other props)."""
    v = value_at(clip, "volume", t - clip.start)
    return max(v, 0.0) if math.isfinite(v) else clip.volume


def active(clip) -> bool:
    """True if this clip needs per-frame animation (forces the render-engine export path)."""
    return bool(clip.keyframes or clip.anim_in or clip.anim_out or clip.anim_loop)
