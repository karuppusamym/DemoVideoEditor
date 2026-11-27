"""FFmpeg-based clip tools. Each processes a source range into a new H.264/AAC .mp4.

Qt-free; run inside a `jobs.Job`. Common signature::

    tool(path, in_point, out_point, progress=None, cancelled=None, status=None, **params) -> str

`run_tool(name, ...)` dispatches by name; `TOOLS` describes them for the UI.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .jobs import Cancelled, check_cancel, ffmpeg_has_filter, ffmpeg_run, output_path

Progress = Callable[[float], None] | None
Cancel = Callable[[], bool] | None
Status = Callable[[str], None] | None

ENC_V = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]
ENC_A = ["-c:a", "aac", "-b:a", "192k"]
CHUNK = 10.0  # reverse: seconds per chunk (bounds memory)


@dataclass
class _Src:
    path: str
    t0: float
    t1: float
    fps: float
    rate: str  # exact rational, e.g. "30000/1001"
    w: int
    h: int
    has_audio: bool

    @property
    def span(self) -> float:
        return self.t1 - self.t0


def _src(path: str, in_point: float, out_point: float | None) -> _Src:
    import av
    with av.open(path) as c:
        if not c.streams.video:
            raise ValueError("This tool needs a video source")
        vs = c.streams.video[0]
        rate = vs.average_rate or vs.guessed_rate or 30
        dur = c.duration / av.time_base if c.duration else 0.0
        has_audio = bool(c.streams.audio)
        w, h = vs.codec_context.width, vs.codec_context.height
    t0 = max(float(in_point or 0.0), 0.0)
    t1 = float(out_point) if out_point and out_point > 0 else dur
    if dur:
        t1 = min(t1, dur)
    if t1 - t0 <= 0.01:
        raise ValueError(f"Empty range {t0:.2f}-{t1:.2f}")
    r = f"{rate.numerator}/{rate.denominator}" if hasattr(rate, "numerator") else str(rate)
    return _Src(os.path.abspath(path), t0, t1, float(rate), r, w, h, has_audio)


def _inp(s: _Src, t0: float | None = None, span: float | None = None) -> list[str]:
    t0 = s.t0 if t0 is None else t0
    span = s.span if span is None else span
    return ["-ss", f"{t0:.6f}", "-t", f"{span:.6f}", "-i", s.path]


def _audio(s: _Src, af: str | None = None) -> list[str]:
    if not s.has_audio:
        return ["-an"]
    return (["-af", af] if af else []) + ENC_A


def _status(status: Status, text: str):
    if status:
        status(text)


# ---- stabilize -------------------------------------------------------------
def stabilize(path, in_point=0.0, out_point=None, progress: Progress = None,
              cancelled: Cancel = None, status: Status = None, strength: float = 0.5) -> str:
    """Two-pass vid.stab when available, else single-pass deshake. strength 0..1."""
    s = _src(path, in_point, out_point)
    k = min(max(float(strength), 0.0), 1.0)
    out = output_path("stabilized", path, ".mp4")
    if ffmpeg_has_filter("vidstabdetect") and ffmpeg_has_filter("vidstabtransform"):
        tmp = tempfile.mkdtemp(prefix="ce_stab_")
        try:
            # relative .trf name + cwd avoids escaping "C:\" inside the filter graph
            trf = "transforms.trf"
            shaky = 3 + int(round(7 * k))
            _status(status, "Analysing motion (pass 1/2)")
            ffmpeg_run([*_inp(s), "-vf", f"vidstabdetect=shakiness={shaky}:accuracy=15:result={trf}",
                        "-an", "-f", "null", "-"], s.span, progress, cancelled, 0.0, 0.45, cwd=tmp)
            smoothing = 5 + int(round(45 * k))
            _status(status, "Stabilizing (pass 2/2)")
            ffmpeg_run([*_inp(s), "-vf",
                        f"vidstabtransform=input={trf}:smoothing={smoothing}:optzoom=1:"
                        f"interpol=bicubic,unsharp=5:5:0.6:3:3:0.0",
                        *ENC_V, *_audio(s), out], s.span, progress, cancelled, 0.45, 1.0, cwd=tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        r = 16 + int(round(48 * k))
        _status(status, "Stabilizing (deshake)")
        ffmpeg_run([*_inp(s), "-vf", f"deshake=rx={r}:ry={r}:edge=mirror",
                    *ENC_V, *_audio(s), out], s.span, progress, cancelled)
    return out


# ---- reverse ---------------------------------------------------------------
def reverse(path, in_point=0.0, out_point=None, progress: Progress = None,
            cancelled: Cancel = None, status: Status = None, chunk: float = CHUNK) -> str:
    """Reverse video + audio. Reverses <= `chunk`-second pieces (the reverse filter buffers
    the whole input in RAM) and concatenates them last-to-first."""
    s = _src(path, in_point, out_point)
    chunk = max(min(float(chunk), CHUNK), 0.5)
    n = max(int(np.ceil(s.span / chunk - 1e-6)), 1)
    out = output_path("reversed", path, ".mp4")
    tmp = tempfile.mkdtemp(prefix="ce_rev_")
    try:
        parts = []
        for i in range(n):
            check_cancel(cancelled)
            a = s.t0 + i * chunk
            span = min(chunk, s.t1 - a)
            part = os.path.join(tmp, f"part{i:04d}.mkv")
            _status(status, f"Reversing chunk {i + 1}/{n}")
            aud = ["-af", "areverse", "-c:a", "pcm_s16le"] if s.has_audio else ["-an"]
            ffmpeg_run([*_inp(s, a, span), "-vf", "reverse", "-c:v", "libx264", "-preset",
                        "veryfast", "-crf", "16", "-pix_fmt", "yuv420p", *aud, part],
                       span, progress, cancelled, 0.9 * i / n, 0.9 * (i + 1) / n)
            parts.append(part)
        lst = os.path.join(tmp, "list.txt")
        with open(lst, "w", encoding="utf-8") as f:
            for p in reversed(parts):
                f.write(f"file '{p.replace(os.sep, '/')}'\n")
        _status(status, "Joining chunks")
        ffmpeg_run(["-f", "concat", "-safe", "0", "-i", lst, "-c:v", "copy",
                    *(ENC_A if s.has_audio else ["-an"]), "-movflags", "+faststart", out],
                   s.span, progress, cancelled, 0.9, 1.0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


# ---- enhance ---------------------------------------------------------------
def enhance(path, in_point=0.0, out_point=None, progress: Progress = None,
            cancelled: Cancel = None, status: Status = None, denoise: float = 0.5,
            sharpen: float = 0.5, color: float = 0.5, upscale: int = 1) -> str:
    """hqdn3d denoise -> optional lanczos upscale (x2) -> unsharp -> mild eq."""
    s = _src(path, in_point, out_point)
    f = []
    if denoise > 0:
        l, c = 4.0 * denoise, 3.0 * denoise
        f.append(f"hqdn3d={l:.2f}:{c:.2f}:{l * 1.5:.2f}:{c * 1.5:.2f}")
    if int(upscale) >= 2:
        f.append(f"scale=iw*{int(upscale)}:ih*{int(upscale)}:flags=lanczos")
    if sharpen > 0:
        f.append(f"unsharp=5:5:{1.2 * sharpen:.2f}:5:5:0.0")
    if color > 0:
        f.append(f"eq=contrast={1 + 0.1 * color:.3f}:saturation={1 + 0.2 * color:.3f}"
                 f":gamma={1 + 0.04 * color:.3f}")
    f.append("format=yuv420p")
    out = output_path("enhanced", path, ".mp4")
    _status(status, "Enhancing")
    ffmpeg_run([*_inp(s), "-vf", ",".join(f), *ENC_V, *_audio(s), out],
               s.span, progress, cancelled)
    return out


# ---- smooth slow motion ----------------------------------------------------
def smooth_slowmo(path, in_point=0.0, out_point=None, progress: Progress = None,
                  cancelled: Cancel = None, status: Status = None, factor: int = 2,
                  quality: str = "mci") -> str:
    """Optical-flow slow motion: minterpolate to factor x fps, then stretch PTS by factor.
    Output plays at the source fps and lasts factor x the range. Audio is time-stretched."""
    s = _src(path, in_point, out_point)
    k = max(int(factor), 2)
    mode = "mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1" if quality == "mci" else "mi_mode=blend"
    vf = f"minterpolate=fps={s.rate}*{k}:{mode},setpts={k}*PTS"
    tempo, kk = [], float(k)
    while kk > 1.0001:  # atempo accepts >= 0.5 per instance
        step = min(kk, 2.0)
        tempo.append(f"atempo={1 / step:.6f}")
        kk /= step
    out = output_path("slowmo", path, ".mp4")
    _status(status, f"Interpolating frames ({k}x)")
    ffmpeg_run([*_inp(s), "-vf", vf, "-r", s.rate, *ENC_V, *_audio(s, ",".join(tempo)), out],
               s.span * k, progress, cancelled)
    return out


# ---- auto reframe ----------------------------------------------------------
ASPECTS = {"9:16": 9 / 16, "1:1": 1.0, "4:5": 4 / 5, "16:9": 16 / 9}


def _box_blur(g: np.ndarray, r: int = 2) -> np.ndarray:
    p = np.pad(g, r, mode="edge")
    c = p.cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    k = 2 * r + 1
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def saliency_track(path: str, t0: float, t1: float, samples_per_s: float = 4.0,
                   width: int = 96, cancelled: Cancel = None, progress: Progress = None):
    """Per-second (time, cx, cy) saliency centre of mass in [0, 1] from motion + contrast."""
    import av
    times, cxs, cys, ws = [], [], [], []
    with av.open(path) as c:
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"
        st = float(vs.start_time * vs.time_base) if vs.start_time is not None else 0.0
        h = max(int(round(width * vs.codec_context.height / max(vs.codec_context.width, 1))), 2)
        c.seek(max(int((t0 + st) / vs.time_base), 0), stream=vs, backward=True)
        prev, next_t = None, t0
        yy, xx = np.mgrid[0:h, 0:width].astype(np.float32)
        xx /= max(width - 1, 1)
        yy /= max(h - 1, 1)
        for fr in c.decode(vs):
            t = (fr.time or 0.0) - st
            if t < next_t - 1e-3:
                continue
            if t >= t1:
                break
            check_cancel(cancelled)
            next_t = t + 1.0 / samples_per_s
            g = fr.reformat(width=width, height=h, format="gray").to_ndarray()[:, :width]
            g = g.astype(np.float32) / 255.0
            contrast = np.abs(g - _box_blur(g, 2))
            motion = np.abs(g - prev) if prev is not None else np.zeros_like(g)
            prev = g
            # blur so coherent regions beat pixel noise, then square to favour the peak
            sal = _box_blur(2.0 * motion + contrast, 3)
            sal = np.maximum(sal - sal.mean() - 0.5 * sal.std(), 0.0) ** 2
            sal *= 0.6 + 0.4 * np.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.18)  # centre bias
            tot = float(sal.sum())
            times.append(t)
            ws.append(tot)
            cxs.append(float((sal * xx).sum() / tot) if tot > 1e-6 else 0.5)
            cys.append(float((sal * yy).sum() / tot) if tot > 1e-6 else 0.5)
            if progress:
                progress(0.3 * min((t - t0) / max(t1 - t0, 1e-6), 1.0))
    if not times:
        return [(0.0, 0.5, 0.5)]
    times, cxs, cys, ws = map(np.asarray, (times, cxs, cys, ws))
    # aggregate per second (weighted), then smooth
    secs = np.floor(times - t0).astype(int)
    out = []
    for sidx in np.unique(secs):
        m = secs == sidx
        wsum = ws[m].sum()
        if wsum > 1e-6:
            cx, cy = (cxs[m] * ws[m]).sum() / wsum, (cys[m] * ws[m]).sum() / wsum
        else:
            cx, cy = 0.5, 0.5
        out.append([float(sidx) + 0.5, cx, cy])
    arr = np.array(out)
    if len(arr) >= 3:  # [1 2 1] smoothing (pans glide instead of jumping)
        for _ in range(1):
            for col in (1, 2):
                v = np.pad(arr[:, col], 1, mode="edge")
                arr[:, col] = (v[:-2] + 2 * v[1:-1] + v[2:]) / 4
    return [tuple(float(v) for v in r) for r in arr]


def _lerp_expr(keys: list[tuple[float, float]]) -> str:
    """Piecewise-linear ffmpeg expression of t through (time, value) keys (clamped ends)."""
    if len(keys) == 1:
        return f"{keys[0][1]:.2f}"
    expr = f"{keys[-1][1]:.2f}"
    for (ta, va), (tb, vb) in reversed(list(zip(keys[:-1], keys[1:]))):
        slope = (vb - va) / max(tb - ta, 1e-6)
        expr = f"if(lt(t,{tb:.3f}),{va:.2f}+({slope:.4f})*(t-{ta:.3f}),{expr})"
    return f"if(lt(t,{keys[0][0]:.3f}),{keys[0][1]:.2f},{expr})"


def auto_reframe(path, in_point=0.0, out_point=None, progress: Progress = None,
                 cancelled: Cancel = None, status: Status = None,
                 target_aspect: str = "9:16") -> str:
    """Crop to `target_aspect` (9:16, 1:1, 4:5) panning smoothly to follow the action."""
    s = _src(path, in_point, out_point)
    ar = ASPECTS.get(target_aspect) if isinstance(target_aspect, str) else float(target_aspect)
    if not ar:
        raise ValueError(f"Unknown aspect {target_aspect!r}; choose from {list(ASPECTS)}")
    _status(status, "Finding the subject")
    track = saliency_track(s.path, s.t0, s.t1, cancelled=cancelled, progress=progress)
    if len(track) > 150:  # keep the expression short for long clips
        idx = np.linspace(0, len(track) - 1, 150).round().astype(int)
        track = [track[i] for i in idx]
    if s.w / s.h > ar:  # too wide: crop width, pan x
        cw, ch = int(s.h * ar) // 2 * 2, s.h // 2 * 2
        keys = [(t, min(max(cx * s.w - cw / 2, 0), s.w - cw)) for t, cx, _ in track]
        xe, ye = _lerp_expr(keys), "0"
    else:  # too tall: crop height, pan y
        cw, ch = s.w // 2 * 2, int(s.w / ar) // 2 * 2
        keys = [(t, min(max(cy * s.h - ch / 2, 0), s.h - ch)) for t, _, cy in track]
        xe, ye = "0", _lerp_expr(keys)
    vf = f"crop={cw}:{ch}:x='{xe}':y='{ye}',format=yuv420p"
    out = output_path("reframed", path, ".mp4")
    _status(status, f"Reframing to {target_aspect}")
    ffmpeg_run([*_inp(s), "-vf", vf, *ENC_V, *_audio(s), out], s.span, progress, cancelled,
               0.3, 1.0)
    return out


# ---- registry --------------------------------------------------------------
TOOLS = {
    "stabilize": {
        "label": "Stabilize",
        "description": "Smooth out camera shake (vid.stab two-pass, or deshake).",
        "params": {"strength": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0,
                                "label": "Strength"}},
        "fn": stabilize, "origin": "stabilized", "duration_factor": 1.0,
    },
    "reverse": {
        "label": "Reverse",
        "description": "Play the clip backwards (video and audio).",
        "params": {},
        "fn": reverse, "origin": "reversed", "duration_factor": 1.0,
    },
    "enhance": {
        "label": "Enhance",
        "description": "Denoise, sharpen and lift colour; optional 2x lanczos upscale.",
        "params": {
            "denoise": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "label": "Denoise"},
            "sharpen": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "label": "Sharpen"},
            "color": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "label": "Colour"},
            "upscale": {"type": "choice", "default": 1, "choices": [1, 2], "label": "Upscale"},
        },
        "fn": enhance, "origin": "enhanced", "duration_factor": 1.0,
    },
    "smooth_slowmo": {
        "label": "Smooth slow motion",
        "description": "Optical-flow frame interpolation, then slow down (slow on CPU).",
        "params": {"factor": {"type": "choice", "default": 2, "choices": [2, 4], "label": "Slow by"},
                   "quality": {"type": "choice", "default": "mci", "choices": ["mci", "blend"],
                               "label": "Method"}},
        "fn": smooth_slowmo, "origin": "slowmo", "duration_factor": None,  # = factor
    },
    "auto_reframe": {
        "label": "Auto reframe",
        "description": "Crop to vertical/square, panning to follow motion and detail.",
        "params": {"target_aspect": {"type": "choice", "default": "9:16",
                                     "choices": ["9:16", "1:1", "4:5"], "label": "Aspect"}},
        "fn": auto_reframe, "origin": "reframed", "duration_factor": 1.0,
    },
}


def run_tool(name: str, path: str, in_point: float, out_point: float | None,
             progress: Progress = None, cancelled: Cancel = None, status: Status = None,
             **params) -> str:
    if name not in TOOLS:
        raise ValueError(f"Unknown tool {name!r}")
    return TOOLS[name]["fn"](path, in_point, out_point, progress=progress, cancelled=cancelled,
                             status=status, **params)


__all__ = ["TOOLS", "run_tool", "stabilize", "reverse", "enhance", "smooth_slowmo",
           "auto_reframe", "saliency_track", "Cancelled"]
