"""Transitions between neighbouring clips on the same track.

Model: a clip B with `transition = d > 0` overlaps the clip A before it on its track by d seconds
(B.start == A.end - d). During the overlap B is drawn on top of A, and the transition kind decides how
B appears (and, for fade-through-black, how A disappears). Audio always crossfades over d.

Preview (Compositor, audio_mix) and export both use these helpers, so they always agree.
"""
from __future__ import annotations

from .model import Clip, Track

KINDS: dict[str, str] = {
    "dissolve": "Dissolve",
    "fadeblack": "Fade through black",
    "slideleft": "Slide in from right",
    "slideright": "Slide in from left",
    "slideup": "Slide in from bottom",
    "slidedown": "Slide in from top",
}
DEFAULT_KIND = "dissolve"
DEFAULT_DURATION = 0.5
EPS = 1e-3


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
