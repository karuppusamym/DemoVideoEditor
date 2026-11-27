"""Media probing, frame decoding and audio loading via PyAV (bundled FFmpeg)."""
from __future__ import annotations

import os
import threading

import av
import numpy as np

from .model import MediaAsset, new_id

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tif", ".tiff"}
IMAGE_DURATION = 3600.0  # stills can be stretched to any length up to this
DEFAULT_IMAGE_CLIP = 5.0


def _has_alpha(stream) -> bool:
    """True when a video stream's pixel format carries an alpha channel.

    Covers yuva*/rgba/argb/bgra/ya8/gbrap... via the component flags, palette PNGs (pal8,
    which may hold transparency), and VP9/WebM with alpha (stored as BlockAdditional
    side data and flagged by the `alpha_mode` stream tag)."""
    cc = stream.codec_context
    name = getattr(cc, "pix_fmt", None) or ""
    if name:
        if name == "pal8" or "yuva" in name or name.startswith(("rgba", "bgra", "argb", "abgr", "ya")):
            return True
        try:
            if any(getattr(c, "is_alpha", False) for c in av.VideoFormat(name).components):
                return True
        except Exception:
            pass
    meta = {k.lower(): v for k, v in (stream.metadata or {}).items()}
    return str(meta.get("alpha_mode", "")) == "1"


def probe(path: str) -> MediaAsset:
    path = os.path.abspath(path)
    if os.path.splitext(path)[1].lower() in IMAGE_EXT:
        try:
            with av.open(path) as c:
                vs = c.streams.video[0]
                cc = vs.codec_context
                w, h = cc.width, cc.height
                alpha = _has_alpha(vs)
        except Exception:
            w = h = 0
            alpha = False
        return MediaAsset(new_id(), path, "image", IMAGE_DURATION, w, h, 0.0, True, False,
                          alpha=alpha)

    with av.open(path) as c:
        vs = c.streams.video[0] if c.streams.video else None
        as_ = c.streams.audio[0] if c.streams.audio else None
        if vs is None and as_ is None:
            raise ValueError(f"No audio or video streams in {path}")
        dur = 0.0
        if c.duration:
            dur = c.duration / av.time_base
        else:
            s = vs or as_
            if s.duration:
                dur = float(s.duration * s.time_base)
        kind = "video" if vs is not None else "audio"
        return MediaAsset(
            new_id(), path, kind, dur,
            vs.codec_context.width if vs else 0,
            vs.codec_context.height if vs else 0,
            float(vs.average_rate or 30) if vs else 0.0,
            vs is not None, as_ is not None,
            alpha=_has_alpha(vs) if vs is not None else False,
        )


class VideoReader:
    """Sequential-friendly frame reader: decodes forward during playback, seeks on scrub."""

    def __init__(self, path: str):
        self.c = av.open(path)
        self.s = self.c.streams.video[0]
        self.s.thread_type = "AUTO"
        self.fd = 1.0 / float(self.s.average_rate or 30)
        st = self.s.start_time
        self.t0 = float(st * self.s.time_base) if st is not None else 0.0
        self.gen = None
        self.cur = None
        self.cur_t = 0.0
        self.pending = None
        self.pending_t = 0.0
        self._rgb_key = None
        self._rgb = None
        self._rgba_key = None
        self._rgba = None

    def close(self):
        self.c.close()

    def _seek(self, t: float):
        target = int((t + self.t0) / self.s.time_base)
        self.c.seek(max(target, 0), stream=self.s, backward=True, any_frame=False)
        self.gen = self.c.decode(self.s)
        self.cur = self.pending = None

    def _next(self):
        try:
            f = next(self.gen)
        except (StopIteration, av.error.EOFError):
            return None, 0.0
        return f, (f.time or 0.0) - self.t0

    def frame_at(self, t: float):
        half = self.fd * 0.5
        if (self.cur is not None and self.pending is not None
                and self.cur_t - half <= t < self.pending_t - half):
            return self.cur
        if self.gen is None or self.cur is None or t < self.cur_t - half or t > self.cur_t + 2.0:
            self._seek(t)
        while True:
            if self.pending is not None:
                f, ft = self.pending, self.pending_t
                self.pending = None
            else:
                f, ft = self._next()
            if f is None:
                break
            if self.cur is None or ft <= t + half:
                self.cur, self.cur_t = f, ft
            else:
                self.pending, self.pending_t = f, ft
                break
        return self.cur

    def rgb_at(self, t: float, w: int, h: int) -> np.ndarray | None:
        f = self.frame_at(t)
        if f is None:
            return None
        key = (id(f), f.pts, w, h)
        if key != self._rgb_key:
            self._rgb = f.reformat(width=w, height=h, format="rgb24").to_ndarray()
            self._rgb_key = key
        return self._rgb

    def rgba_at(self, t: float, w: int, h: int) -> np.ndarray | None:
        """Like `rgb_at` but returns uint8 (h, w, 4) RGBA (opaque if the source has no alpha)."""
        f = self.frame_at(t)
        if f is None:
            return None
        key = (id(f), f.pts, w, h)
        if key != self._rgba_key:
            arr = f.reformat(width=w, height=h, format="rgba").to_ndarray()
            if arr.ndim == 3 and arr.shape[1] != w:  # strip any line padding
                arr = arr[:, :w]
            self._rgba = np.ascontiguousarray(arr, dtype=np.uint8)
            self._rgba_key = key
        return self._rgba


def load_audio(path: str, sr: int = 48000) -> np.ndarray:
    """Decode the whole audio stream to float32 stereo, shape (n, 2)."""
    chunks = []
    with av.open(path) as c:
        if not c.streams.audio:
            return np.zeros((0, 2), np.float32)
        s = c.streams.audio[0]
        rs = av.AudioResampler(format="flt", layout="stereo", rate=sr)
        for frame in c.decode(s):
            for out in rs.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1, 2))
        for out in rs.resample(None):
            chunks.append(out.to_ndarray().reshape(-1, 2))
    if not chunks:
        return np.zeros((0, 2), np.float32)
    return np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)


def waveform_peaks(samples: np.ndarray, sr: int, per_second: int = 100) -> np.ndarray:
    hop = max(sr // per_second, 1)
    n = len(samples) // hop
    if n == 0:
        return np.zeros(0, np.float32)
    mono = np.abs(samples[: n * hop]).max(axis=1)
    return mono.reshape(n, hop).max(axis=1)


def thumbnails(path: str, duration: float, height: int = 54, step: float = 1.0, limit: int = 120):
    """Yield (time, rgb ndarray) thumbnails roughly every `step` seconds."""
    step = max(step, duration / limit) if duration > 0 else step
    r = VideoReader(path)
    try:
        w = max(int(r.s.codec_context.width * height / max(r.s.codec_context.height, 1)), 1)
        t = 0.0
        while t < max(duration, 0.01):
            arr = r.rgb_at(t, w, height)
            if arr is not None:
                yield t, arr.copy()
            t += step
    finally:
        r.close()


class AudioCache:
    """Thread-safe cache of decoded audio + waveform peaks per asset."""

    def __init__(self, sr: int = 48000):
        self.sr = sr
        self._data: dict[str, np.ndarray] = {}
        self._peaks: dict[str, np.ndarray] = {}
        self._paths: dict[str, str] = {}
        self._lock = threading.Lock()

    def set_asset_path(self, asset: MediaAsset) -> None:
        """Invalidate cached audio when an asset ID is relinked to a different file."""
        with self._lock:
            if self._paths.get(asset.id) != asset.path:
                self._paths[asset.id] = asset.path
                self._data.pop(asset.id, None)
                self._peaks.pop(asset.id, None)

    def load(self, asset: MediaAsset) -> None:
        if not asset.has_audio:
            return
        with self._lock:
            if self._paths.setdefault(asset.id, asset.path) != asset.path or asset.id in self._data:
                return
        data = load_audio(asset.path, self.sr)
        peaks = waveform_peaks(data, self.sr)
        with self._lock:
            if self._paths[asset.id] == asset.path:
                self._data[asset.id] = data
                self._peaks[asset.id] = peaks

    def get(self, asset_id: str) -> np.ndarray | None:
        return self._data.get(asset_id)

    def peaks(self, asset_id: str) -> np.ndarray | None:
        return self._peaks.get(asset_id)
