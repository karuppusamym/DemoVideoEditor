"""Render a Project to a video file by building one FFmpeg filter graph."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

from . import transitions
from .encoders import codec_args
from .model import Project


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
        if not exe:
            raise RuntimeError("FFmpeg not found. pip install imageio-ffmpeg")
        return exe


def _f(x: float) -> str:
    return f"{x:.4f}"


def _even(x: float) -> int:
    return max(int(round(x / 2)) * 2, 2)


def _atempo_chain(speed: float) -> list[str]:
    """atempo stages (each limited to 0.5..2) whose product is `speed`."""
    stages: list[str] = []
    s = speed
    while s > 2.0 + 1e-9:
        stages.append("atempo=2.0")
        s /= 2.0
    while s < 0.5 - 1e-9:
        stages.append("atempo=0.5")
        s /= 0.5
    if abs(s - 1.0) > 1e-6:
        stages.append(f"atempo={s:.6f}")
    return stages


def _fit_size(aw: int, ah: int, W: int, H: int, scale: float) -> tuple[int, int]:
    """Fit (aw, ah) inside (W, H) keeping aspect ratio, times `scale`; even dimensions."""
    if aw <= 0 or ah <= 0:
        aw, ah = W, H
    k = min(W / aw, H / ah) * max(scale, 0.01)
    return _even(aw * k), _even(ah * k)


def _prepare_qt_fonts() -> None:
    """The offscreen Qt platform (used for CLI export) finds no fonts on Windows unless
    QT_QPA_FONTDIR points at them; harmless when a real QApplication already exists."""
    if os.name == "nt" and not os.environ.get("QT_QPA_FONTDIR"):
        fonts = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
        if os.path.isdir(fonts):
            os.environ["QT_QPA_FONTDIR"] = fonts


def _has_text(project: Project) -> bool:
    return any(c.is_text for t in project.video_tracks() if not t.hidden for c in t.clips)


def build_command(project: Project, out_path: str, crf: int = 20, preset: str = "medium",
                  width: int | None = None, height: int | None = None,
                  fps: float | None = None, workdir: str | None = None,
                  codec: str = "h264", gpu: bool = False, ten_bit: bool = False) -> tuple[list[str], str]:
    """Returns (ffmpeg argv, filter-graph text). Graph goes in a script file.

    Text clips are rendered to PNGs in `workdir`. If it is None and the project has text
    clips, a temp dir is created that the caller owns (`export()` manages its own).
    """
    if any(c.speed_ramp for track in project.tracks for c in track.clips):
        raise ValueError("Speed ramps require the render engine.")
    W = width or project.width
    H = height or project.height
    W, H = W - W % 2, H - H % 2
    fps = fps or project.fps
    D = project.duration
    if D <= 0:
        raise ValueError("The timeline is empty.")

    args: list[str] = []
    fc: list[str] = [f"color=c=black:s={W}x{H}:r={fps}:d={_f(D)},format=yuva420p[base]"]
    audio: list[str] = []
    cur = "base"
    idx = 0

    def add_audio(i: int, track, clip):
        ms = int(round(clip.start * 1000))
        chain = ["asetpts=PTS-STARTPTS", "aresample=48000",
                 "aformat=sample_fmts=fltp:channel_layouts=stereo"]
        chain += _atempo_chain(clip.speed)
        chain.append(f"volume={clip.volume:.3f}")
        d = clip.duration
        fade_in, fade_out = transitions.audio_fades(track, clip)  # includes crossfades
        if fade_in > 0:
            chain.append(f"afade=t=in:st=0:d={_f(min(fade_in, d))}")
        if fade_out > 0:
            fo = min(fade_out, d)
            chain.append(f"afade=t=out:st={_f(d - fo)}:d={_f(fo)}")
        chain.append(f"adelay={ms}:all=1")
        fc.append(f"[{i}:a]" + ",".join(chain) + f"[a{i}]")
        audio.append(f"[a{i}]")

    def overlay(i: int, track, clip, x0: float, y0: float):
        """Overlay [v{i}] on the running composite. x/y are timeline-time expressions (slides),
        evaluated per frame. Each clip is enabled over its own [start, end], so during a
        transition both the outgoing and the incoming clip are active."""
        nonlocal cur
        x, y = transitions.ffmpeg_overlay_xy(track, clip, x0, y0, W, H)
        fc.append(f"[{cur}][v{i}]overlay=x='{x}':y='{y}':eval=frame:eof_action=pass:"
                  f"enable='between(t,{_f(clip.start)},{_f(clip.end)})'[o{i}]")
        cur = f"o{i}"

    # bottom video track first so upper tracks overlay it
    for track in reversed(project.video_tracks()):
        if track.hidden:
            continue
        # start-time order: the incoming clip of a transition is overlaid on top of the outgoing one
        for clip in track.sorted_clips():
            dur = clip.duration
            # clip-local post-processing: opacity, fades + transition alpha, then move to timeline position
            tail: list[str] = []
            if clip.opacity < 0.999:
                tail.append(f"colorchannelmixer=aa={clip.opacity:.3f}")
            tail += transitions.ffmpeg_alpha_fades(track, clip)
            tail.append(f"setpts=PTS+{_f(clip.start)}/TB")

            if clip.is_text:
                # already full-canvas and positioned by render_text: no scaling
                from .textrender import ensure_qapp, render_text

                if workdir is None:
                    workdir = tempfile.mkdtemp(prefix="clipeditor_text_")
                _prepare_qt_fonts()
                ensure_qapp()
                png = os.path.join(workdir, f"text_{idx}_{clip.id}.png")
                if not render_text(clip, W, H, project.height).save(png, "PNG"):
                    raise RuntimeError(f"Could not write text image {png}")
                args += ["-loop", "1", "-framerate", str(fps), "-t", _f(dur), "-i", png]
                chain = [f"[{idx}:v]setpts=PTS-STARTPTS", f"fps={fps}", "format=yuva420p", *tail]
                fc.append(",".join(chain) + f"[v{idx}]")
                overlay(idx, track, clip, 0.0, 0.0)
                idx += 1
                continue

            asset = project.assets[clip.asset_id]
            if asset.kind == "image":
                args += ["-loop", "1", "-framerate", str(fps), "-t", _f(dur), "-i", asset.path]
                pts = "setpts=PTS-STARTPTS"
            else:
                args += ["-ss", _f(clip.in_point), "-t", _f(clip.out_point - clip.in_point),
                         "-i", asset.path]
                pts = f"setpts=(PTS-STARTPTS)/{clip.speed:.6f}"
            w, h = _fit_size(asset.width, asset.height, W, H, clip.scale)
            x = (W - w) / 2 + clip.pos_x * W
            y = (H - h) / 2 + clip.pos_y * H
            chain = [f"[{idx}:v]{pts}", f"fps={fps}", f"scale={w}:{h}"]
            if (abs(clip.brightness) > 1e-6 or abs(clip.contrast - 1) > 1e-6
                    or abs(clip.saturation - 1) > 1e-6):
                chain.append(f"eq=brightness={clip.brightness:.4f}:contrast={clip.contrast:.4f}"
                             f":saturation={clip.saturation:.4f}")
            chain.append("format=yuva420p")
            chain += tail
            fc.append(",".join(chain) + f"[v{idx}]")
            overlay(idx, track, clip, x, y)
            if asset.has_audio and asset.kind != "image" and clip.volume > 0 and not track.muted:
                add_audio(idx, track, clip)
            idx += 1

    for track in project.audio_tracks():
        if track.muted:
            continue
        for clip in track.sorted_clips():
            if clip.volume <= 0 or clip.is_text or clip.asset_id not in project.assets:
                continue
            asset = project.assets[clip.asset_id]
            if not asset.has_audio:
                continue
            args += ["-ss", _f(clip.in_point), "-t", _f(clip.out_point - clip.in_point),
                     "-i", asset.path]
            add_audio(idx, track, clip)
            idx += 1

    fc.append(f"[{cur}]format=yuv420p[vout]")
    fc.append(f"anullsrc=r=48000:cl=stereo,atrim=0:{_f(D)}[abase]")
    fc.append(f"[abase]{''.join(audio)}amix=inputs={len(audio) + 1}:normalize=0:duration=first[aout]")
    graph = ";\n".join(fc)

    cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-nostats", "-progress", "pipe:1", *args,
           "-/filter_complex", "{GRAPH}",
           "-map", "[vout]", "-map", "[aout]",
           *codec_args(codec, gpu, crf, preset, ten_bit),
           "-r", str(fps), "-c:a", "aac", "-b:a", "192k", "-t", _f(D),
           "-movflags", "+faststart", out_path]
    return cmd, graph


def check_cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise RuntimeError("Export cancelled")


@contextmanager
def export_destination(project: Project, path: str, cancelled: Callable[[], bool] | None):
    check_cancelled(cancelled)
    destination = Path(path).resolve()
    for asset in project.assets.values():
        if any(source and Path(source).resolve() == destination for source in (asset.path, asset.proxy_path)):
            raise ValueError("Choose an export file that is not used as project media.")
    with tempfile.TemporaryDirectory(prefix=".clipeditor-", dir=destination.parent) as work:
        pending = Path(work) / destination.name
        yield str(pending)
        check_cancelled(cancelled)
        os.replace(pending, destination)


def export(project: Project, out_path: str, progress: Callable[[float], None] | None = None,
           cancelled: Callable[[], bool] | None = None, **opts) -> None:
    with export_destination(project, out_path, cancelled) as pending:
        with tempfile.TemporaryDirectory(prefix="clipeditor_export_") as work:
            if opts.get("workdir") is None:
                opts["workdir"] = work
            cmd, graph = build_command(project, pending, **opts)
            graph_path = Path(work) / "graph.ffgraph"
            graph_path.write_text(graph, encoding="utf-8")
            cmd[cmd.index("{GRAPH}")] = str(graph_path)
            _run_graph(cmd, project.duration, progress, cancelled)
    if progress:
        progress(1.0)


def _run_graph(cmd, duration, progress, cancelled):
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                          encoding="utf-8", errors="replace", creationflags=flags) as proc:
        err_tail: list[str] = []

        def drain():
            for line in proc.stderr:
                err_tail.append(line)
                del err_tail[:-40]

        th = threading.Thread(target=drain, daemon=True)
        th.start()
        try:
            for line in proc.stdout:
                check_cancelled(cancelled)
                m = re.match(r"out_time_us=(\d+)", line)
                if m and progress:
                    progress(min(int(m.group(1)) / 1e6 / duration, 0.99))
            proc.wait()
            if proc.returncode != 0:
                th.join()
                raise RuntimeError("FFmpeg failed:\n" + "".join(err_tail))
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            th.join()
