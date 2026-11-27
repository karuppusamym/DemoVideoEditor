"""Render-engine export: draw every frame with the same Compositor as the preview and pipe it to FFmpeg.

Used whenever the project uses features the FFmpeg filter-graph exporter can't express
(keyframes, animations, effects, masks, chroma key, shapes, rotation/flip/crop, blend modes,
audio effects). Audio is the same timeline mix the preview plays (with processed clip audio).

    export_video(project, out_path, progress, cancelled, engine="auto", **opts)
is the single entry point the UI / CLI / API should call.
"""
from __future__ import annotations

import math
import os
import re
import subprocess
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np

from . import animate, audio_fx, effects, masks
from .encoders import codec_args
from .export import export as export_graph
from .export import check_cancelled, export_destination, ffmpeg_exe
from .model import Project

_NOWIN = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def needs_render_engine(project: Project) -> list[str]:
    """Reasons the fast filter-graph exporter can't be used (empty list = graph is fine)."""
    why = set()
    if project.active_timeline.frame_cuts:
        why.add("frame-accurate cuts")
    for tr in project.tracks:
        for c in tr.clips:
            if c.presentation_in or c.presentation_out:
                why.add("frame-accurate cuts")
            if c.speed_ramp:
                why.add("speed ramps")
            elif abs(c.speed - 1.0) > 1e-6:
                why.add("clip speed")
            if animate.active(c):
                why.add("keyframes/animations")
            if effects.active(c):
                why.add("effects")
            if masks.active(c):
                why.add("masks")
            if c.is_shape:
                why.add("shapes")
            if c.rotation or c.flip_h or c.flip_v or any(c.crop or []):
                why.add("rotate/flip/crop")
            if c.blend_mode not in ("", "normal"):
                why.add("blend modes")
            if c.audio_fx:
                why.add("audio effects")
    return sorted(why)


def _timeline_audio(project: Project, sr: int = 48000, cancelled=None) -> np.ndarray:
    from .audio_mix import mix
    from .media import AudioCache

    cache = AudioCache(sr)
    for a in project.timeline_assets():
        check_cancelled(cancelled)
        if a.has_audio:
            cache.load(a)
    D = project.duration
    n_total = int(math.ceil(D * sr))
    out = np.zeros((n_total, 2), np.float32)
    step = sr * 5
    for i0 in range(0, n_total, step):
        check_cancelled(cancelled)
        n = min(step, n_total - i0)
        out[i0:i0 + n] = mix(project, cache, i0 / sr, n, wait=True)
    return out


def export_render(project: Project, out_path: str, progress: Callable[[float], None] | None = None,
                  cancelled: Callable[[], bool] | None = None, width: int | None = None,
                  height: int | None = None, fps: float | None = None, crf: int = 20,
                  preset: str = "medium", codec: str = "h264", gpu: bool = False, ten_bit: bool = False,
                  **_ignored) -> None:
    from .render import Compositor
    from .textrender import ensure_qapp

    ensure_qapp()
    W = (width or project.width) & ~1
    H = (height or project.height) & ~1
    fps = fps or project.fps
    D = project.duration
    if D <= 0:
        raise ValueError("The timeline is empty.")
    n_frames = max(int(round(D * fps)), 1)
    prog = progress or (lambda _v: None)

    with export_destination(project, out_path, cancelled) as pending:
        prog(0.0)
        with tempfile.TemporaryDirectory(prefix="clipeditor_audio_") as work:
            apath = str(Path(work) / "audio.f32")
            _timeline_audio(project, cancelled=cancelled).tofile(apath)
            cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-v", "error",
                   "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
                   "-f", "f32le", "-ar", "48000", "-ac", "2", "-i", apath,
                   "-map", "0:v", "-map", "1:a",
                   *codec_args(codec, gpu, crf, preset, ten_bit),
                   "-c:a", "aac", "-b:a", "192k", "-t", f"{D:.4f}", "-movflags", "+faststart", pending]
            comp = Compositor(use_proxies=False)
            try:
                _write_frames(cmd, comp, project, W, H, fps, n_frames, prog, cancelled)
            finally:
                comp.close()
    prog(1.0)


def _write_frames(cmd, comp, project, width, height, fps, count, progress, cancelled):
    with subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_NOWIN) as proc:
        err: list[bytes] = []
        th = threading.Thread(target=lambda: err.extend(proc.stderr.readlines()), daemon=True)
        th.start()
        try:
            for i in range(count):
                check_cancelled(cancelled)
                img = comp.render(project, i / fps, width, height)
                proc.stdin.write(img.constBits())
                if i % 3 == 0:
                    progress(0.02 + 0.97 * (i + 1) / count)
            proc.stdin.close()
            proc.wait()
        except BrokenPipeError:
            proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            th.join()
        if proc.returncode != 0:
            raise RuntimeError("FFmpeg failed:\n" + b"".join(err[-30:]).decode(errors="replace"))


def export_video(project: Project, out_path: str, progress=None, cancelled=None, engine: str = "auto",
                 **opts) -> str:
    """Export with the fast FFmpeg graph when possible, else the render engine. Returns engine used."""
    use_render = engine == "render" or (engine == "auto" and needs_render_engine(project))
    if use_render:
        export_render(project, out_path, progress, cancelled, **opts)
        return "render"
    export_graph(project, out_path, progress=progress, cancelled=cancelled, **opts)
    return "graph"
