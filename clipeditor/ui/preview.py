"""Preview compositor + viewer widget, and the playback clock (audio-driven)."""
from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import QPointF, QObject, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QWidget

from ..audio_mix import mix
from ..media import AudioCache, VideoReader
from ..model import Project
from ..render import Compositor  # noqa: F401  (re-exported for main_window)


class PreviewWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(240, 135)
        self.image: QImage | None = None
        self.aspect = 16 / 9
        self.safe_zones = False

    def frame_size(self) -> tuple[int, int]:
        w, h = self.width(), self.height()
        if w / max(h, 1) > self.aspect:
            w = int(h * self.aspect)
        else:
            h = int(w / self.aspect)
        scale = min(1.0, 960 / max(w, 1))  # cap preview cost
        w, h = int(w * scale * self.devicePixelRatioF()), int(h * scale * self.devicePixelRatioF())
        return max(w - w % 2, 16), max(h - h % 2, 16)

    def set_image(self, img: QImage):
        self.image = img
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(18, 18, 20))
        w, h = self.width(), self.height()
        if w / max(h, 1) > self.aspect:
            dw, dh = h * self.aspect, h
        else:
            dw, dh = w, w / self.aspect
        target = QRectF((w - dw) / 2, (h - dh) / 2, dw, dh)
        if self.image is not None:
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            p.drawImage(target, self.image)
        else:
            p.fillRect(target, Qt.GlobalColor.black)
        if self.safe_zones:
            self._paint_safe(p, target)
        p.end()

    @staticmethod
    def _paint_safe(p: QPainter, r: QRectF):
        """Overlay where social apps put their UI (vertical) or title-safe margins (landscape)."""
        from PySide6.QtGui import QPen

        shade = QColor(255, 60, 60, 55)
        p.setPen(QPen(QColor(255, 255, 255, 140), 1, Qt.PenStyle.DashLine))
        if r.height() > r.width():  # 9:16 / 4:5 -> TikTok / Reels / Shorts
            top = QRectF(r.left(), r.top(), r.width(), r.height() * 0.10)
            bottom = QRectF(r.left(), r.bottom() - r.height() * 0.22, r.width(), r.height() * 0.22)
            right = QRectF(r.right() - r.width() * 0.15, r.top() + r.height() * 0.35, r.width() * 0.15,
                           r.height() * 0.43)
            for z in (top, bottom, right):
                p.fillRect(z, shade)
                p.drawRect(z)
            p.setPen(QColor(255, 255, 255, 200))
            p.drawText(bottom, Qt.AlignmentFlag.AlignCenter, "captions / username")
            p.drawText(right, Qt.AlignmentFlag.AlignCenter, "♥\n💬\n↗")
        else:  # title-safe 90% and action-safe 95%
            for k in (0.05, 0.10):
                z = r.adjusted(r.width() * k / 2, r.height() * k / 2, -r.width() * k / 2, -r.height() * k / 2)
                p.drawRect(z)


class Player(QObject):
    """Playback clock. Audio output drives timing; falls back to wall clock."""

    tick = Signal(float)
    stopped = Signal()

    def __init__(self, get_project, audio: AudioCache):
        super().__init__()
        self.get_project = get_project
        self.audio = audio
        self.playing = False
        self.t_start = 0.0
        self._wall0 = 0.0
        self._samples = 0
        self._stream = None
        self._latency = 0.0
        self.timer = QTimer(self)
        self.timer.setInterval(15)
        self.timer.timeout.connect(self._on_timer)

    def position(self) -> float:
        if self._stream is not None:
            return self.t_start + max(self._samples / self.audio.sr - self._latency, 0.0)
        return self.t_start + (time.perf_counter() - self._wall0)

    def _callback(self, outdata, frames, _time, _status):
        t = self.t_start + self._samples / self.audio.sr
        try:
            outdata[:] = mix(self.get_project(), self.audio, t, frames)
        except Exception:  # noqa: BLE001
            outdata.fill(0)
        self._samples += frames

    def play(self, t: float):
        if self.playing:
            return
        self.t_start = t
        self._samples = 0
        self._wall0 = time.perf_counter()
        try:
            import sounddevice as sd

            self._stream = sd.OutputStream(samplerate=self.audio.sr, channels=2, dtype="float32",
                                           blocksize=1024, callback=self._callback)
            self._stream.start()
            lat = self._stream.latency
            self._latency = float(lat if not isinstance(lat, tuple) else lat[1])
        except Exception:  # noqa: BLE001 - no audio device: silent playback
            self._stream = None
        self.playing = True
        self.timer.start()

    def stop(self):
        if not self.playing:
            return
        pos = self.position()
        self.playing = False
        self.timer.stop()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None
        self.t_start = pos
        self.stopped.emit()

    def _on_timer(self):
        pos = self.position()
        end = self.get_project().duration
        if pos >= end:
            self.stop()
            self.tick.emit(end)
            return
        self.tick.emit(pos)
