"""Background loading of thumbnails and audio (keeps the UI responsive)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage

from ..media import AudioCache, thumbnails
from ..model import MediaAsset


def ndarray_to_qimage(arr) -> QImage:
    # PyAV pads rows at some widths, giving a non-contiguous view; QImage needs a plain buffer
    arr = np.ascontiguousarray(arr)
    h, w, _ = arr.shape
    return QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class AssetLoader(QObject):
    thumbs_ready = Signal(str)
    audio_ready = Signal(str)
    failed = Signal(str, str)

    def __init__(self, audio: AudioCache):
        super().__init__()
        self.audio = audio
        self.thumbs: dict[str, list[tuple[float, QImage]]] = {}
        self._pool = ThreadPoolExecutor(max_workers=3)
        self._started: set[str] = set()

    def load(self, asset: MediaAsset):
        if asset.id in self._started:
            return
        self._started.add(asset.id)
        if asset.kind == "image":
            img = QImage(asset.path)
            if not img.isNull():
                self.thumbs[asset.id] = [(0.0, img.scaledToHeight(54))]
            self.thumbs_ready.emit(asset.id)
        elif asset.has_video:
            self._pool.submit(self._thumbs, asset)
        if asset.has_audio:
            self._pool.submit(self._audio, asset)

    def _thumbs(self, asset: MediaAsset):
        try:
            step = 1.0 if asset.duration < 60 else asset.duration / 60
            out = []
            for t, arr in thumbnails(asset.path, asset.duration, step=step):
                out.append((t, ndarray_to_qimage(arr)))
                if len(out) == 1:
                    self.thumbs[asset.id] = list(out)
                    self.thumbs_ready.emit(asset.id)
            self.thumbs[asset.id] = out
            self.thumbs_ready.emit(asset.id)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(asset.id, f"thumbnails: {e}")

    def _audio(self, asset: MediaAsset):
        try:
            self.audio.load(asset)
            self.audio_ready.emit(asset.id)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(asset.id, f"audio: {e}")

    def thumb_at(self, asset_id: str, t: float) -> QImage | None:
        lst = self.thumbs.get(asset_id)
        if not lst:
            return None
        best = lst[0]
        for item in lst:
            if item[0] <= t + 1e-6:
                best = item
            else:
                break
        return best[1]

    def shutdown(self):
        self._pool.shutdown(wait=False, cancel_futures=True)
