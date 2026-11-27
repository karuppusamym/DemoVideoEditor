"""The compositor: draws the timeline at time t into a QImage.

Used by the live preview AND by the render-engine export (clipeditor.export_render), so every
feature implemented here looks identical in both.

Per clip, per frame:
    props   = animate.resolve(clip, t)                  keyframes + in/out/loop animations
    alpha   = props.opacity x transitions.video_alpha   fades + transitions
    offset  = transitions.video_offset                  slide transitions
    source  = video frame | image | text element | shape element
    pixels  = eq colour -> effects.apply (incl. chroma key) -> masks.apply     (RGBA numpy)
    draw    = crop -> flip -> rotate -> scale around centre at (pos_x, pos_y) with blend mode
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter

from . import animate, effects, masks, shapes
from .media import VideoReader
from .model import Project
from .textrender import render_text_element
from .transitions import video_alpha, video_offset

BLEND_MODES = {
    "normal": QPainter.CompositionMode.CompositionMode_SourceOver,
    "multiply": QPainter.CompositionMode.CompositionMode_Multiply,
    "screen": QPainter.CompositionMode.CompositionMode_Screen,
    "overlay": QPainter.CompositionMode.CompositionMode_Overlay,
    "darken": QPainter.CompositionMode.CompositionMode_Darken,
    "lighten": QPainter.CompositionMode.CompositionMode_Lighten,
    "color_dodge": QPainter.CompositionMode.CompositionMode_ColorDodge,
    "color_burn": QPainter.CompositionMode.CompositionMode_ColorBurn,
    "hard_light": QPainter.CompositionMode.CompositionMode_HardLight,
    "soft_light": QPainter.CompositionMode.CompositionMode_SoftLight,
    "difference": QPainter.CompositionMode.CompositionMode_Difference,
    "exclusion": QPainter.CompositionMode.CompositionMode_Exclusion,
    "add": QPainter.CompositionMode.CompositionMode_Plus,
}


def rgb_to_qimage(arr: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(arr)
    h, w, _ = arr.shape
    return QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def rgba_to_qimage(arr: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(arr)
    h, w, _ = arr.shape
    return QImage(arr.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()


def qimage_to_rgba(img: QImage) -> np.ndarray:
    src = img.convertToFormat(QImage.Format.Format_RGBA8888)
    bpl = src.bytesPerLine()
    buf = np.frombuffer(src.constBits(), np.uint8, count=bpl * src.height())
    return buf.reshape(src.height(), bpl)[:, : src.width() * 4].reshape(src.height(), src.width(), 4).copy()


def apply_eq(arr: np.ndarray, brightness: float, contrast: float, saturation: float) -> np.ndarray:
    """FFmpeg `eq` approximation on the RGB channels of an (h, w, 3|4) uint8 array."""
    rgb = arr[..., :3].astype(np.float32)
    rgb *= np.float32(contrast / 255.0)
    rgb += np.float32(0.5 - 0.5 * contrast + brightness)
    if saturation != 1:
        luma = (rgb @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
        rgb -= luma
        rgb *= np.float32(saturation)
        rgb += luma
    np.clip(rgb, 0.0, 1.0, out=rgb)
    rgb *= 255.0
    rgb += 0.5
    out = arr.copy()
    out[..., :3] = rgb.astype(np.uint8)
    return out


class Compositor:
    MAX_DECODE = 2.0  # cap decode size at this multiple of the canvas
    _CACHE_MAX = 48

    def __init__(self, use_proxies: bool = False):
        self.use_proxies = use_proxies
        self.readers: dict = {}
        self.images: dict[str, QImage] = {}
        self._cache: dict[tuple, QImage] = {}
        self.last_error: str | None = None

    # ---- sources ---------------------------------------------------------
    def reader(self, path: str, lane: str = "") -> VideoReader:
        key = (path, lane)
        r = self.readers.get(key)
        if r is None:
            r = self.readers[key] = VideoReader(path)
        return r

    def _cached(self, key, make):
        img = self._cache.get(key)
        if img is None:
            if len(self._cache) >= self._CACHE_MAX:
                self._cache.pop(next(iter(self._cache)))
            img = self._cache[key] = make()
        return img

    def _image(self, path: str) -> QImage | None:
        q = self.images.get(path)
        if q is None:
            q = QImage(path)
            if q.isNull():
                return None
            self.images[path] = q
        return q

    @staticmethod
    def needs_eq(clip) -> bool:
        return clip.brightness != 0 or clip.contrast != 1 or clip.saturation != 1

    apply_eq = staticmethod(apply_eq)

    # ---- render ----------------------------------------------------------
    def render(self, project: Project, t: float, w: int, h: int) -> QImage:
        img = QImage(w, h, QImage.Format.Format_RGB32)
        img.fill(QColor(0, 0, 0))
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        for track in reversed(project.video_tracks()):
            if track.hidden:
                continue
            for lane, clip in enumerate(track.clips_at(t)):
                try:
                    self._draw_clip(p, project, track, clip, lane, t, w, h)
                except Exception as e:  # noqa: BLE001 - one broken layer must not kill the frame
                    self.last_error = f"{type(e).__name__}: {e}"
        p.end()
        return img

    def _draw_clip(self, p: QPainter, project: Project, track, clip, lane: int, t: float, w: int, h: int):
        props = animate.resolve(clip, t)
        alpha = props.opacity * video_alpha(track, clip, t)
        if alpha <= 0.001 or props.scale <= 0:
            return
        dx, dy = video_offset(track, clip, t, w, h)
        t_local = t - clip.start
        k = h / max(project.height, 1)  # output px per project px

        # ---- source pixels ---------------------------------------------
        if clip.is_text:
            if not clip.text:
                return
            q = self._cached(("text", clip.text, clip.font_family, clip.font_size, clip.color, clip.bg_color,
                              clip.bold, round(k * props.scale, 3)),
                             lambda: render_text_element(clip, k * props.scale))
            fw, fh = q.width(), q.height()
            pixel_fx = effects.active(clip) or masks.active(clip)
        elif clip.is_shape:
            q = self._cached(("shape", repr(sorted(clip.shape.items())), round(h * props.scale, 1)),
                             lambda: shapes.render_shape(clip, max(int(h * props.scale), 2)))
            fw, fh = q.width(), q.height()
            pixel_fx = effects.active(clip) or masks.active(clip)
        else:
            asset = project.assets.get(clip.asset_id)
            if asset is None or not (asset.has_video or asset.kind == "image"):
                return
            aw, ah = max(asset.width, 1), max(asset.height, 1)
            s = min(w / aw, h / ah) * props.scale
            fw, fh = aw * s, ah * s
            if fw < 1 or fh < 1:
                return
            cap = min(1.0, self.MAX_DECODE * w / fw, self.MAX_DECODE * h / fh)
            dw, dh = max(int(fw * cap) & ~1, 2), max(int(fh * cap) & ~1, 2)
            pixel_fx = self.needs_eq(clip) or effects.active(clip) or masks.active(clip)
            if asset.kind == "image":
                src = self._image(asset.path)
                if src is None:
                    return
                if not pixel_fx:
                    q = src
                else:
                    q = self._cached(("img", asset.path, dw, dh),
                                     lambda: src.scaled(dw, dh, Qt.AspectRatioMode.IgnoreAspectRatio,
                                                        Qt.TransformationMode.SmoothTransformation))
            else:
                path = asset.proxy_path if (self.use_proxies and asset.proxy_path) else asset.path
                r = self.reader(path, f"{track.id}:{lane}")
                st = clip.source_time(t)
                if asset.alpha and hasattr(r, "rgba_at"):
                    arr = r.rgba_at(st, dw, dh)
                    if arr is None:
                        return
                    q = rgba_to_qimage(arr) if not pixel_fx else None
                    rgba = arr if pixel_fx else None
                else:
                    arr = r.rgb_at(st, dw, dh)
                    if arr is None:
                        return
                    q = rgb_to_qimage(arr) if not pixel_fx else None
                    if pixel_fx:
                        rgba = np.empty(arr.shape[:2] + (4,), np.uint8)
                        rgba[..., :3] = arr
                        rgba[..., 3] = 255
                if pixel_fx:
                    q = self._pixels(rgba, clip, t_local, h)
            if asset.kind == "image" and pixel_fx:
                q = self._pixels(qimage_to_rgba(q), clip, t_local, h)
        if (clip.is_text or clip.is_shape) and pixel_fx:
            q = self._pixels(qimage_to_rgba(q), clip, t_local, h)

        # ---- placement -------------------------------------------------
        cx = w / 2 + props.pos_x * w + dx
        cy = h / 2 + props.pos_y * h + dy
        cl, ct, cr, cb = (list(clip.crop) + [0, 0, 0, 0])[:4]
        cl, ct = min(max(cl, 0), 0.95), min(max(ct, 0), 0.95)
        cr, cb = min(max(cr, 0), 0.95 - cl), min(max(cb, 0), 0.95 - ct)
        iw, ih = q.width(), q.height()
        src_rect = QRectF(cl * iw, ct * ih, iw * (1 - cl - cr), ih * (1 - ct - cb))
        target = QRectF(-fw / 2 + cl * fw, -fh / 2 + ct * fh, fw * (1 - cl - cr), fh * (1 - ct - cb))

        p.save()
        p.translate(cx, cy)
        if props.rotation:
            p.rotate(props.rotation)
        if clip.flip_h or clip.flip_v:
            p.scale(-1 if clip.flip_h else 1, -1 if clip.flip_v else 1)
        p.setOpacity(min(alpha, 1.0))
        mode = BLEND_MODES.get(clip.blend_mode)
        if mode is not None:
            p.setCompositionMode(mode)
        p.drawImage(target, q, src_rect)
        p.restore()

    def _pixels(self, rgba: np.ndarray, clip, t_local: float, canvas_h: int) -> QImage:
        if self.needs_eq(clip):
            rgba = apply_eq(rgba, clip.brightness, clip.contrast, clip.saturation)
        if effects.active(clip):
            rgba = effects.apply(rgba, clip, t_local, canvas_h)
        if masks.active(clip):
            rgba = masks.apply(rgba, clip, t_local)
        return rgba_to_qimage(rgba)

    def close(self):
        for r in self.readers.values():
            try:
                r.close()
            except Exception:  # noqa: BLE001
                pass
        self.readers.clear()
