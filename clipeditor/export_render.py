"""Render-engine export: draw every frame with the same Compositor as the preview and pipe it to FFmpeg.

Used whenever the project uses features the FFmpeg filter-graph exporter can't express
(keyframes, animations, effects, masks, chroma key, shapes, rotation/flip/crop, blend modes,
audio effects). Audio is the same timeline mix the preview plays (with processed clip audio).

    export_video(project, out_path, progress, cancelled, engine="auto", **opts)
is the single entry point the UI / CLI / API should call.

Long exports render in parallel: the frame range is split into contiguous chunks, each rendered
by a spawned worker process (clipeditor.render_worker) into its own video-only segment, then the
segments are joined with the FFmpeg concat demuxer (stream copy) and muxed with the timeline
audio. With a GPU encoder the workers render interleaved frame blocks and stream them back to one
FFmpeg instead (GPU encode sessions are limited). CLIPEDITOR_EXPORT_WORKERS=1 forces one process.
"""
from __future__ import annotations

import logging
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
from .encoders import codec_args, gpu_encoder
from .export import export as export_graph
from .export import check_cancelled, export_destination, ffmpeg_exe
from .model import Project

_NOWIN = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
log = logging.getLogger(__name__)

MIN_PARALLEL_FRAMES = 60
MAX_WORKERS = 8
STREAM_BUFFER_BYTES = 512 * 1024 * 1024  # frames buffered across all workers in GPU (stream) mode


def _needs_render_transition(kind: str) -> bool:
    from .transitions import needs_render

    return needs_render(kind)


def needs_render_engine(project: Project) -> list[str]:
    """Reasons the fast filter-graph exporter can't be used (empty list = graph is fine)."""
    why = set()
    if project.active_timeline.frame_cuts:
        why.add("frame-accurate cuts")
    for tr in project.tracks:
        for c in tr.clips:
            if c.sequence_id:
                why.add("nested sequences")
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
            if c.grade:
                from . import grade as _grade

                if _grade.active(c):
                    why.add("colour grading")
            if c.transition > 0 and _needs_render_transition(c.transition_kind):
                why.add("transitions")
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
    from .textrender import ensure_qapp

    ensure_qapp()
    W = (width or project.width) & ~1
    H = (height or project.height) & ~1
    fps = fps or project.fps
    D = project.duration
    if D <= 0:
        raise ValueError("The timeline is empty.")
    n_frames = max(int(round(D * fps)), 1)
    prog = _Monotonic(progress)
    vargs = codec_args(codec, gpu, crf, preset, ten_bit)
    # GPU encoders allow few concurrent sessions: render in parallel but encode in one FFmpeg
    stream = bool(gpu) and gpu_encoder(codec) is not None
    workers = parallel_workers(n_frames)

    with export_destination(project, out_path, cancelled) as pending:
        prog(0.0)
        done = False
        if workers > 1:
            try:
                _export_parallel(project, pending, W, H, fps, D, n_frames, vargs, workers, prog, cancelled,
                                 stream=stream)
                done = True
            except _PoolStartError as e:
                log.warning("Parallel export unavailable (%s); rendering in a single process.", e)
        if not done:
            _export_single(project, pending, W, H, fps, D, n_frames, vargs, prog, cancelled)
    prog(1.0)


def parallel_workers(n_frames: int) -> int:
    """Worker processes to use for an export of n_frames (1 = the single-process path)."""
    if n_frames < MIN_PARALLEL_FRAMES:
        return 1
    n = min((os.cpu_count() or 1) - 1, MAX_WORKERS, n_frames // 30)
    env = os.environ.get("CLIPEDITOR_EXPORT_WORKERS", "").strip()
    if env:
        try:
            n = min(n, int(env))
        except ValueError:
            log.warning("Ignoring invalid CLIPEDITOR_EXPORT_WORKERS=%r", env)
    return max(n, 1)


class _Monotonic:
    """Progress callback wrapper that never reports a value lower than an earlier one."""

    def __init__(self, progress):
        self.progress = progress
        self.value = -1.0

    def __call__(self, v: float) -> None:
        if v > self.value:
            self.value = v
            if self.progress is not None:
                self.progress(v)


def _write_audio(project: Project, path: str, cancelled) -> None:
    # ndarray.tofile() needs a C FILE*, which fails with some Windows CRT builds; write bytes instead
    with open(path, "wb") as f:
        f.write(_timeline_audio(project, cancelled=cancelled).tobytes())


def _mux_args(apath: str, vargs: list[str], D: float, pending: str) -> list[str]:
    """Audio input + mapping + encoder + output arguments shared by every final FFmpeg."""
    return ["-f", "f32le", "-ar", "48000", "-ac", "2", "-i", apath,
            "-map", "0:v", "-map", "1:a",
            *vargs,
            "-c:a", "aac", "-b:a", "192k", "-t", f"{D:.4f}", "-movflags", "+faststart", pending]


def _raw_cmd(W: int, H: int, fps: float, apath: str, vargs: list[str], D: float, pending: str) -> list[str]:
    return [ffmpeg_exe(), "-y", "-hide_banner", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
            *_mux_args(apath, vargs, D, pending)]


def _export_single(project, pending, W, H, fps, D, n_frames, vargs, prog, cancelled) -> None:
    from .render import Compositor

    with tempfile.TemporaryDirectory(prefix="clipeditor_audio_") as work:
        apath = str(Path(work) / "audio.f32")
        _write_audio(project, apath, cancelled)
        cmd = _raw_cmd(W, H, fps, apath, vargs, D, pending)
        comp = Compositor(use_proxies=False)
        try:
            _write_frames(cmd, comp, project, W, H, fps, n_frames, prog, cancelled)
        finally:
            comp.close()


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


# ---- parallel render ------------------------------------------------------------------------
class _PoolStartError(RuntimeError):
    """The worker pool could not start; export_render falls back to the single-process path."""


def _chunks(n_frames: int, n: int) -> list[tuple[int, int]]:
    """n contiguous, near-equal [start, end) frame ranges covering 0..n_frames."""
    b = [round(k * n_frames / n) for k in range(n + 1)]
    return [(b[k], b[k + 1]) for k in range(n) if b[k + 1] > b[k]]


class _Pool:
    """Spawned render workers and their status queue. close() always tears everything down."""

    def __init__(self, jobs: list[dict], stream: bool):
        import multiprocessing as mp

        from . import render_worker

        ctx = mp.get_context("spawn")
        self.status = ctx.Queue()
        self.stop = ctx.Event()
        self.procs: list = []
        self.conns: list = []
        self.ready: set[int] = set()
        self.finished: set[int] = set()
        self.ffmpeg_pids: dict[int, int] = {}
        self.rendered = [0] * len(jobs)
        try:
            for job in jobs:
                recv = send = None
                if stream:
                    recv, send = ctx.Pipe(duplex=False)
                p = ctx.Process(target=render_worker.run, args=(job, self.status, self.stop, send),
                                name=f"clipeditor-render-{job['index']}", daemon=True)
                p.start()
                self.procs.append(p)
                self.conns.append(recv)
                if send is not None:
                    send.close()  # the worker owns the write end now
        except Exception as e:
            self.close()
            raise _PoolStartError(f"{type(e).__name__}: {e}") from e

    def _handle(self, msg) -> None:
        kind, k = msg[0], msg[1]
        if kind == "frames":
            self.rendered[k] += msg[2]
        elif kind == "ready":
            self.ready.add(k)
        elif kind == "ffmpeg":
            self.ffmpeg_pids[k] = msg[2]
        elif kind == "done":
            self.finished.add(k)
        elif kind == "error":
            self._fail(f"Render worker {k} failed:\n{msg[2]}")

    def poll(self, timeout: float = 0.0) -> None:
        """Process queued status messages; raise if a worker failed or died."""
        import queue

        while True:
            try:
                msg = self.status.get(timeout=timeout) if timeout else self.status.get_nowait()
            except queue.Empty:
                break
            timeout = 0.0
            self._handle(msg)
        for k, p in enumerate(self.procs):
            if k in self.finished or p.exitcode is None:
                continue
            try:  # it may have exited right after queueing its last messages
                while k not in self.finished:
                    self._handle(self.status.get(timeout=0.5))
            except queue.Empty:
                self._fail(f"Render worker {k} exited unexpectedly (exit code {p.exitcode}).")

    def _fail(self, text: str):
        if len(self.ready) < len(self.procs) and not any(self.rendered):
            raise _PoolStartError(text)  # never got going: the caller falls back
        raise RuntimeError(text)

    def close(self) -> None:
        import queue
        import signal
        import time

        self.stop.set()
        deadline = time.monotonic() + 3.0
        while any(p.is_alive() for p in self.procs) and time.monotonic() < deadline:
            try:  # keep draining so no worker blocks while flushing its queue
                while True:
                    self._note(self.status.get_nowait())
            except (queue.Empty, OSError, ValueError):
                pass
            for p in self.procs:
                p.join(0.02)
        for p in self.procs:
            if p.is_alive():
                p.terminate()
        for p in self.procs:
            p.join(2.0)
        for k, pid in self.ffmpeg_pids.items():  # a terminated worker can leave its FFmpeg running
            if k not in self.finished:
                try:
                    os.kill(pid, signal.SIGTERM if os.name == "nt" else signal.SIGKILL)
                except OSError:
                    pass
        for c in self.conns:
            if c is not None:
                c.close()
        self.status.close()
        self.status.cancel_join_thread()

    def _note(self, msg) -> None:
        """Record bookkeeping from a message without raising (used during teardown)."""
        if msg[0] == "ffmpeg":
            self.ffmpeg_pids[msg[1]] = msg[2]
        elif msg[0] in ("done", "stopped"):
            self.finished.add(msg[1])


def _rmtree(path: str) -> None:
    import shutil
    import time

    for _ in range(50):  # Windows: a just-killed FFmpeg can hold its segment open for a moment
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError:
            time.sleep(0.1)
    shutil.rmtree(path, ignore_errors=True)


def _export_parallel(project, pending, W, H, fps, D, n_frames, vargs, workers, prog, cancelled,
                     stream: bool = False) -> None:
    work = tempfile.mkdtemp(prefix="clipeditor_render_")
    pool = None
    try:
        apath = str(Path(work) / "audio.f32")
        base = {"project": project.to_dict(), "width": W, "height": H, "fps": fps, "vargs": vargs}
        block = 1
        if stream:
            per_worker = max(STREAM_BUFFER_BYTES // (W * H * 4 * workers), 2)
            block = max(min(per_worker // workers, 30), 1)
            jobs = [{**base, "index": k, "mode": "stream", "buffer": per_worker,
                     "frames": [i for i in range(n_frames) if (i // block) % workers == k]}
                    for k in range(workers)]
        else:
            jobs = [{**base, "index": k, "mode": "segment", "start": a, "end": b,
                     "out": str(Path(work) / f"seg{k:03d}.mp4")}
                    for k, (a, b) in enumerate(_chunks(n_frames, workers))]
        check_cancelled(cancelled)
        pool = _Pool(jobs, stream)

        # the timeline audio is mixed in a thread while the workers render
        audio_stop = threading.Event()
        audio_err: list[BaseException] = []

        def audio():
            try:
                _write_audio(project, apath, audio_stop.is_set)
            except BaseException as e:  # noqa: BLE001
                audio_err.append(e)

        ath = threading.Thread(target=audio, daemon=True)
        ath.start()
        try:
            if stream:
                _wait_audio(pool, ath, audio_err, cancelled)
                cmd = _raw_cmd(W, H, fps, apath, vargs, D, pending)
                _stream_frames(cmd, pool, block, workers, n_frames, prog, cancelled)
            else:
                _wait_segments(pool, len(jobs), n_frames, prog, cancelled, audio_err)
                _wait_audio(pool, ath, audio_err, cancelled)
                prog(0.97)
                listing = Path(work) / "segments.txt"
                listing.write_text("".join(f"file '{Path(j['out']).name}'\n" for j in jobs), encoding="utf-8")
                cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-v", "error",
                       "-f", "concat", "-safe", "0", "-i", str(listing),
                       *_mux_args(apath, ["-c:v", "copy"], D, pending)]
                _run(cmd, cancelled)
                prog(0.99)
        finally:
            audio_stop.set()
            ath.join()
    finally:
        if pool is not None:
            pool.close()
        _rmtree(work)


def _wait_audio(pool, ath, audio_err, cancelled) -> None:
    while ath.is_alive():
        check_cancelled(cancelled)
        pool.poll(0.05)
    if audio_err:
        raise audio_err[0]


def _wait_segments(pool, n_jobs, n_frames, prog, cancelled, audio_err) -> None:
    while len(pool.finished) < n_jobs:
        check_cancelled(cancelled)
        if audio_err:
            raise audio_err[0]
        pool.poll(0.05)
        prog(0.02 + 0.95 * min(sum(pool.rendered), n_frames) / n_frames)


def _stream_frames(cmd, pool, block, workers, n_frames, prog, cancelled) -> None:
    """GPU mode: pull frames from the workers in timeline order and feed one FFmpeg."""
    with subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_NOWIN) as proc:
        err: list[bytes] = []
        th = threading.Thread(target=lambda: err.extend(proc.stderr.readlines()), daemon=True)
        th.start()
        try:
            for i in range(n_frames):
                conn = pool.conns[(i // block) % workers]
                while not conn.poll(0.05):
                    check_cancelled(cancelled)
                    pool.poll()
                try:
                    data = conn.recv_bytes()
                except EOFError:
                    pool.poll(0.5)
                    raise RuntimeError("A render worker stopped sending frames.") from None
                proc.stdin.write(data)
                if i % 3 == 0:
                    check_cancelled(cancelled)
                    prog(0.02 + 0.97 * (i + 1) / n_frames)
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


def _run(cmd: list[str], cancelled) -> None:
    """Run an FFmpeg command to completion, killing it promptly on cancel."""
    with subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stderr=subprocess.PIPE, creationflags=_NOWIN) as proc:
        err: list[bytes] = []
        th = threading.Thread(target=lambda: err.extend(proc.stderr.readlines()), daemon=True)
        th.start()
        try:
            while True:
                try:
                    proc.wait(0.1)
                    break
                except subprocess.TimeoutExpired:
                    check_cancelled(cancelled)
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

    def run(o):
        if use_render:
            export_render(project, out_path, progress, cancelled, **o)
            return "render"
        export_graph(project, out_path, progress=progress, cancelled=cancelled, **o)
        return "graph"

    try:
        return run(opts)
    except Exception as e:  # GPU encoders can fail mid-export (driver/session limits): retry on the CPU
        if not opts.get("gpu") or (cancelled and cancelled()) or "cancel" in str(e).lower():
            raise
        log.warning("GPU export failed (%s); retrying with the software encoder.", str(e).splitlines()[0][:200])
        return run({**opts, "gpu": False})
