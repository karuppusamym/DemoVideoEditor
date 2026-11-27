"""Worker process for the parallel render-engine export (see clipeditor.export_render).

`run(job, status, stop, conn=None)` is the spawn entry point. Each worker builds its own offscreen
QGuiApplication and Compositor from the project dict, then either

  * "segment" mode: renders the contiguous frame range [start, end) and pipes the BGRA frames into
    its own FFmpeg, which writes a video-only segment (it starts on a keyframe because it is a
    separate encode), or
  * "stream" mode: renders the frames it owns (interleaved blocks) and sends the raw BGRA bytes to
    the parent through `conn`, which feeds a single FFmpeg (used for GPU encoders).

Status messages on `status` (a multiprocessing Queue):
    ("ready", k, pid)   worker initialised        ("ffmpeg", k, pid)  its FFmpeg started
    ("frames", k, n)    n more frames rendered    ("done", k)         finished
    ("error", k, text)  failed (traceback text)   ("stopped", k)      stopped on request
This module must stay importable without clipeditor.ui (it runs in a fresh interpreter).
"""
from __future__ import annotations

import os
import queue as _queue
import subprocess
import threading
import traceback

_NOWIN = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

# Every core already runs a worker: keep numpy's BLAS single-threaded to avoid oversubscription.
# (Takes effect when this module is imported before numpy, which is the case in a fresh worker.)
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")


class Stopped(Exception):
    pass


def run(job: dict, status, stop, conn=None) -> None:
    k = job["index"]
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from .model import Project
        from .render import Compositor
        from .textrender import ensure_qapp

        ensure_qapp()
        project = Project.from_dict(job["project"])
        comp = Compositor(use_proxies=False)
        status.put(("ready", k, os.getpid()))
        try:
            if job["mode"] == "segment":
                _segment(job, project, comp, status, stop)
            else:
                _stream(job, project, comp, status, stop, conn)
        finally:
            comp.close()
        status.put(("done", k))
    except Stopped:
        status.put(("stopped", k))
    except BaseException:  # noqa: BLE001 - report everything to the parent
        status.put(("error", k, traceback.format_exc()))
    finally:
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass


def _check(stop) -> None:
    if stop.is_set():
        raise Stopped


def _segment(job, project, comp, status, stop) -> None:
    from .export import ffmpeg_exe

    k, W, H, fps = job["index"], job["width"], job["height"], job["fps"]
    cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           *job["vargs"], "-an", "-f", "mp4", job["out"]]
    _check(stop)
    with subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_NOWIN) as proc:
        status.put(("ffmpeg", k, proc.pid))
        err: list[bytes] = []
        th = threading.Thread(target=lambda: err.extend(proc.stderr.readlines()), daemon=True)
        th.start()
        pending = 0
        try:
            for i in range(job["start"], job["end"]):
                _check(stop)
                img = comp.render(project, i / fps, W, H)
                proc.stdin.write(img.constBits())
                pending += 1
                if pending >= 3:
                    status.put(("frames", k, pending))
                    pending = 0
            proc.stdin.close()
            proc.wait()
        except BrokenPipeError:
            proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            th.join()
        if pending:
            status.put(("frames", k, pending))
        if proc.returncode != 0:
            raise RuntimeError("FFmpeg failed:\n" + b"".join(err[-30:]).decode(errors="replace"))


def _stream(job, project, comp, status, stop, conn) -> None:
    """Render the owned frames; a sender thread pushes them to the parent while we render ahead."""
    W, H, fps = job["width"], job["height"], job["fps"]
    frames = job["frames"]
    buf: _queue.Queue = _queue.Queue(maxsize=max(job.get("buffer", 4), 1))
    failed: list[BaseException] = []

    def sender():
        try:
            while True:
                data = buf.get()
                if data is None:
                    return
                conn.send_bytes(data)
        except BaseException as e:  # noqa: BLE001 - parent went away or stopped reading
            failed.append(e)

    th = threading.Thread(target=sender, daemon=True)
    th.start()
    for i in frames:
        _check(stop)
        img = comp.render(project, i / fps, W, H)
        data = bytes(img.constBits())
        while True:
            if failed:
                raise RuntimeError(f"Sending frames to the exporter failed: {failed[0]!r}")
            _check(stop)
            try:
                buf.put(data, timeout=0.1)
                break
            except _queue.Full:
                pass
    while True:
        _check(stop)
        try:
            buf.put(None, timeout=0.1)
            break
        except _queue.Full:
            if failed:
                break
    while th.is_alive():
        _check(stop)
        th.join(0.1)
    if failed:
        raise RuntimeError(f"Sending frames to the exporter failed: {failed[0]!r}")
