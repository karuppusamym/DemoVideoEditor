"""AI background removal for video ranges and still images (rembg + onnxruntime, CPU).

Qt-free; run inside a `jobs.Job`. Models download on first use into `~/.u2net`
(or `$U2NET_HOME`).

Pipeline per frame: downscale to ~320 px -> model -> soft mask at low res -> temporal EMA
(less flicker) -> bicubic upscale + feather blur -> RGBA. Video output is QuickTime
Animation (qtrle, argb) in a .mov, which PyAV decodes back with alpha; the source audio
of the range is muxed in (AAC).

    out = remove_background("clip.mp4", 2.0, 6.0, "u2netp", progress=..., cancelled=...)
"""
from __future__ import annotations

import os
import queue
import subprocess
import tempfile
import threading
from typing import Callable

import numpy as np

from .jobs import Cancelled, check_cancel, ffmpeg_exe, output_path, _NO_WINDOW

MODELS = {
    "u2netp": "Fast (u2netp, ~5 MB download)",
    "isnet-general-use": "Quality (ISNet, ~170 MB download, ~5x slower)",
    "u2net_human_seg": "People (u2net human seg, ~170 MB download)",
}
# working resolution (long side) the model sees; the mask is upscaled to full frame
WORK_SIZE = {"u2netp": 320, "isnet-general-use": 512, "u2net_human_seg": 320}
FORMATS = {
    "mov": "QuickTime Animation (qtrle) .mov - lossless, fast to decode",
    "mov_png": "PNG-in-MOV - smaller files, slower to decode",
    "webm": "WebM VP9 + alpha - small; alpha only decodes via libvpx-vp9",
}

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tif", ".tiff"}

_sessions: dict[str, object] = {}
_sess_lock = threading.Lock()


def is_available() -> bool:
    try:
        import rembg  # noqa: F401
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    return True


def get_session(model: str):
    """Cached rembg session (downloads the model on first use)."""
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}; choose from {list(MODELS)}")
    with _sess_lock:
        s = _sessions.get(model)
        if s is None:
            from rembg import new_session
            s = new_session(model)
            _sessions[model] = s
        return s


# ---- mask maths ------------------------------------------------------------
def _small(rgb: np.ndarray, work: int):
    from PIL import Image
    h, w = rgb.shape[:2]
    k = work / max(h, w)
    img = Image.fromarray(rgb)
    if k < 1:
        img = img.resize((max(int(w * k), 1), max(int(h * k), 1)), Image.Resampling.BILINEAR)
    return img


def predict_mask(session, rgb: np.ndarray, work: int = 320) -> np.ndarray:
    """Low-res float32 mask in [0, 1] (shape of the downscaled frame)."""
    small = _small(rgb, work)
    m = session.predict(small)[0]
    return np.asarray(m, dtype=np.float32) / 255.0


def refine_mask(m: np.ndarray, w: int, h: int, feather: float = 1.0) -> np.ndarray:
    """Soft levels + bicubic upscale + feathered edge -> uint8 (h, w) alpha."""
    from PIL import Image, ImageFilter
    m = np.clip((m - 0.08) / 0.84, 0.0, 1.0)  # drop faint haze, keep edges soft
    img = Image.fromarray((m * 255 + 0.5).astype(np.uint8), mode="L")
    img = img.resize((w, h), Image.Resampling.BICUBIC)
    r = feather * max(w, h) / 800.0
    if r >= 0.3:
        img = img.filter(ImageFilter.GaussianBlur(r))
    return np.asarray(img, dtype=np.uint8)


class MaskSmoother:
    """Temporal EMA of low-res masks; resets on a scene cut (big mask change)."""

    def __init__(self, alpha: float = 0.6, cut: float = 0.35):
        self.alpha, self.cut, self.prev = alpha, cut, None

    def __call__(self, m: np.ndarray) -> np.ndarray:
        if self.prev is None or self.prev.shape != m.shape or \
                float(np.abs(m - self.prev).mean()) > self.cut:
            self.prev = m
        else:
            self.prev = self.alpha * m + (1.0 - self.alpha) * self.prev
        return self.prev


# ---- stills ----------------------------------------------------------------
def remove_background_image(path: str, model: str = "u2netp", progress=None, cancelled=None,
                            status=None, feather: float = 1.0) -> str:
    from PIL import Image
    if status:
        status("Loading model")
    sess = get_session(model)
    check_cancel(cancelled)
    if status:
        status("Removing background")
    rgb = np.asarray(Image.open(path).convert("RGB"))
    h, w = rgb.shape[:2]
    work = max(WORK_SIZE.get(model, 320), 512)  # stills: one frame, afford more detail
    a = refine_mask(predict_mask(sess, rgb, work), w, h, feather)
    out = output_path("bg_removed", path, ".png")
    Image.fromarray(np.dstack([rgb, a]), mode="RGBA").save(out)
    if progress:
        progress(1.0)
    return out


# ---- video -----------------------------------------------------------------
def _encode_args(fmt: str) -> tuple[str, list[str], list[str]]:
    if fmt == "mov":
        return ".mov", ["-c:v", "qtrle", "-pix_fmt", "argb"], ["-c:a", "aac", "-b:a", "192k"]
    if fmt == "mov_png":
        return ".mov", ["-c:v", "png", "-pix_fmt", "rgba"], ["-c:a", "aac", "-b:a", "192k"]
    if fmt == "webm":
        return ".webm", ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "30",
                         "-row-mt", "1", "-deadline", "good", "-cpu-used", "4",
                         "-auto-alt-ref", "0"], ["-c:a", "libopus", "-b:a", "128k"]
    raise ValueError(f"Unknown format {fmt!r}; choose from {list(FORMATS)}")


def remove_background(path: str, in_point: float = 0.0, out_point: float | None = None,
                      model: str = "u2netp", progress: Callable[[float], None] | None = None,
                      cancelled: Callable[[], bool] | None = None,
                      status: Callable[[str], None] | None = None, *,
                      fmt: str = "mov_png", smoothing: float = 0.6, feather: float = 1.0) -> str:
    """Remove the background of `path` over [in_point, out_point) -> new file path.

    Stills (by extension) produce a PNG. `smoothing` is the EMA weight of the newest mask
    (1 = no temporal smoothing)."""
    if os.path.splitext(path)[1].lower() in IMAGE_EXT:
        return remove_background_image(path, model, progress, cancelled, status, feather)
    import av

    if status:
        status(f"Loading model {model} (first use downloads it)")
    sess = get_session(model)
    check_cancel(cancelled)
    work = WORK_SIZE.get(model, 320)

    src = av.open(path)
    try:
        vs = src.streams.video[0]
        vs.thread_type = "AUTO"
        has_audio = bool(src.streams.audio)
        rate = vs.average_rate or vs.guessed_rate or 30
        fps = float(rate)
        t0s = float(vs.start_time * vs.time_base) if vs.start_time is not None else 0.0
        dur = src.duration / av.time_base if src.duration else 0.0
        if out_point is None or out_point <= 0 or (dur and out_point > dur):
            out_point = dur or (in_point + 1e9)
        span = max(out_point - in_point, 1.0 / fps)
        n_total = max(int(round(span * fps)), 1)
        w, h = vs.codec_context.width, vs.codec_context.height

        ext, vargs, aargs = _encode_args(fmt)
        out = output_path("bg_removed", path, ext)
        cmd = [ffmpeg_exe(), "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}",
               "-r", f"{rate.numerator}/{rate.denominator}" if hasattr(rate, "numerator") else str(fps),
               "-i", "pipe:0"]
        if has_audio:
            cmd += ["-ss", f"{in_point:.6f}", "-t", f"{span:.6f}", "-i", path,
                    "-map", "0:v:0", "-map", "1:a:0?", *vargs, *aargs]
        else:
            cmd += ["-map", "0:v:0", *vargs]
        cmd += ["-shortest", out] if has_audio else [out]
        errf = tempfile.TemporaryFile()
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                stderr=errf, creationflags=_NO_WINDOW)
        smooth = MaskSmoother(alpha=smoothing)
        n = 0
        # encoder feed runs in its own thread so qtrle/png encoding overlaps inference
        q: queue.Queue = queue.Queue(maxsize=6)
        wr_err: list[BaseException] = []

        def writer():
            try:
                while True:
                    buf = q.get()
                    if buf is None:
                        break
                    proc.stdin.write(buf)
            except BaseException as e:  # broken pipe etc.
                wr_err.append(e)
                while q.get() is not None:  # drain so the producer never blocks
                    pass

        wt = threading.Thread(target=writer, daemon=True)
        wt.start()
        try:
            if status:
                status("Removing background")
            src.seek(max(int((in_point + t0s) / vs.time_base), 0), stream=vs, backward=True)
            half = 0.5 / fps
            for frame in src.decode(vs):
                t = (frame.time or 0.0) - t0s
                if t < in_point - half:
                    continue
                if t >= out_point - half or n >= n_total:
                    break
                check_cancel(cancelled)
                rgb = frame.to_ndarray(format="rgb24")
                m = smooth(predict_mask(sess, rgb, work))
                a = refine_mask(m, w, h, feather)
                if wr_err:
                    break
                q.put(np.ascontiguousarray(np.dstack([rgb, a])).tobytes())
                n += 1
                if progress:
                    progress(0.98 * n / n_total)
            q.put(None)
            wt.join()
            proc.stdin.close()
            proc.wait()
        except BaseException:
            try:
                q.put(None)
                proc.kill()
                proc.wait()
            finally:
                if os.path.exists(out):
                    try:
                        os.remove(out)
                    except OSError:
                        pass
            raise
        if proc.returncode != 0 or n == 0:
            errf.seek(0)
            tail = errf.read().decode("utf-8", "replace")[-800:]
            raise RuntimeError(f"Encoding failed ({proc.returncode}, {n} frames): {tail}")
        errf.close()
    finally:
        src.close()
    if progress:
        progress(1.0)
    return out


__all__ = ["MODELS", "FORMATS", "is_available", "remove_background", "remove_background_image",
           "Cancelled"]
