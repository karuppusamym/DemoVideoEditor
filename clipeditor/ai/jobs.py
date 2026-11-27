"""Tiny background-job framework for "process a clip range into a new media file".

Qt-free. A `Job` wraps a tool function that has the common signature::

    fn(progress: Callable[[float], None], cancelled: Callable[[], bool],
       status: Callable[[str], None]) -> str          # returns the output path

Typical use (GUI polls `job.progress` / `job.status` with a QTimer)::

    job = Job("Remove background", lambda progress, cancelled, status:
              remove_background(path, t0, t1, "u2netp", progress, cancelled, status))
    job.start()
    ...
    if job.done: use job.result or show job.error

Every tool works on a source range `(path, in_point, out_point)` in source seconds.
"""
from __future__ import annotations

import os
import re
import subprocess
import threading
import time
import uuid
from typing import Callable

ProgressCB = Callable[[float], None] | None
CancelCB = Callable[[], bool] | None
StatusCB = Callable[[str], None] | None


class Cancelled(Exception):
    """Raised inside a tool when the user cancels."""


class Job:
    def __init__(self, name: str, fn: Callable[..., str]):
        self.name = name
        self.fn = fn
        self.progress = 0.0
        self.status = "Queued"
        self.result: str | None = None
        self.error: str | None = None
        self.started = 0.0
        self.finished = 0.0
        self._cancel = threading.Event()
        self._done = threading.Event()
        self._thread: threading.Thread | None = None

    # -- control -------------------------------------------------------------
    def start(self) -> "Job":
        self.started = time.time()
        self.status = "Starting"
        self._thread = threading.Thread(target=self._run, name=f"job:{self.name}", daemon=True)
        self._thread.start()
        return self

    def run_sync(self) -> str | None:
        """Run in the calling thread (tests / CLI)."""
        self.started = time.time()
        self._run()
        return self.result

    def cancel(self) -> None:
        self._cancel.set()
        if not self.done:
            self.status = "Cancelling"

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)

    # -- state ---------------------------------------------------------------
    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @property
    def ok(self) -> bool:
        return self.done and self.result is not None and self.error is None

    @property
    def elapsed(self) -> float:
        end = self.finished or time.time()
        return end - self.started if self.started else 0.0

    # -- internals -----------------------------------------------------------
    def _set_progress(self, p: float) -> None:
        self.progress = min(max(float(p), 0.0), 1.0)

    def _set_status(self, s: str) -> None:
        self.status = s

    def _run(self) -> None:
        try:
            out = self.fn(progress=self._set_progress, cancelled=self._cancel.is_set,
                          status=self._set_status)
            if self._cancel.is_set():
                raise Cancelled()
            self.result = out
            self.progress = 1.0
            self.status = "Done"
        except Cancelled:
            self.error = "Cancelled"
            self.status = "Cancelled"
        except Exception as e:  # surfaced to the UI
            self.error = f"{type(e).__name__}: {e}"
            self.status = "Failed"
        finally:
            self.finished = time.time()
            self._done.set()


# ---- paths -----------------------------------------------------------------
def output_dir(kind: str) -> str:
    """%APPDATA%/ClipEditor/generated/<kind> (created on demand)."""
    base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), ".config")
    d = os.path.join(base, "ClipEditor", "generated", kind)
    os.makedirs(d, exist_ok=True)
    return d


def output_path(kind: str, src_path: str, ext: str) -> str:
    stem = os.path.splitext(os.path.basename(src_path))[0][:40]
    return os.path.join(output_dir(kind), f"{stem}_{kind}_{uuid.uuid4().hex[:8]}{ext}")


def check_cancel(cancelled: CancelCB) -> None:
    if cancelled is not None and cancelled():
        raise Cancelled()


# ---- ffmpeg ----------------------------------------------------------------
def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


_filters_cache: str | None = None


def ffmpeg_has_filter(name: str) -> bool:
    global _filters_cache
    if _filters_cache is None:
        try:
            _filters_cache = subprocess.run(
                [ffmpeg_exe(), "-hide_banner", "-filters"], capture_output=True, text=True,
                creationflags=_NO_WINDOW, timeout=30).stdout
        except Exception:
            _filters_cache = ""
    return re.search(rf"^\s*\S+\s+{re.escape(name)}\s", _filters_cache, re.M) is not None


_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def ffmpeg_run(args: list[str], duration: float = 0.0, progress_cb: ProgressCB = None,
               cancelled: CancelCB = None, p0: float = 0.0, p1: float = 1.0,
               cwd: str | None = None) -> None:
    """Run ffmpeg with `args` (without the exe), reporting progress in [p0, p1].

    Parses `-progress pipe:1` (out_time_us) against `duration` seconds of output and kills
    the process if `cancelled()` becomes true (raising `Cancelled`). `cwd` lets filters
    take relative file names (avoids escaping "C:" in filter graphs). Raises RuntimeError
    with the stderr tail on failure."""
    cmd = [ffmpeg_exe(), "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
           "-progress", "pipe:1", "-nostats", *args]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            stdin=subprocess.DEVNULL, creationflags=_NO_WINDOW, cwd=cwd)
    err_chunks: list[bytes] = []
    t_err = threading.Thread(target=lambda: err_chunks.append(proc.stderr.read()), daemon=True)
    t_err.start()
    killed = False

    def watchdog():
        nonlocal killed
        while proc.poll() is None:
            if cancelled is not None and cancelled():
                killed = True
                proc.kill()
                return
            time.sleep(0.1)

    t_dog = threading.Thread(target=watchdog, daemon=True)
    t_dog.start()
    try:
        for raw in proc.stdout:
            line = raw.decode("utf-8", "replace").strip()
            if progress_cb is not None and duration > 0 and line.startswith("out_time_us="):
                try:
                    us = int(line.split("=", 1)[1])
                except ValueError:
                    continue
                progress_cb(p0 + (p1 - p0) * min(max(us / 1e6 / duration, 0.0), 1.0))
    finally:
        proc.wait()
        t_err.join(timeout=2)
        t_dog.join(timeout=2)
    if killed:
        raise Cancelled()
    if proc.returncode != 0:
        tail = b"".join(err_chunks).decode("utf-8", "replace").strip()[-800:]
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {tail}")
    if progress_cb is not None:
        progress_cb(p1)
