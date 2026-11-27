"""Multi-track timeline: ruler, clips with thumbnails/waveforms, move/trim/snap, drop."""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QMenu, QScrollBar, QVBoxLayout, QWidget

from ..editor import Editor
from ..model import MIN_CLIP, Track
from ..transitions import KINDS, neighbour_before, prev_clip

ASSET_MIME = "application/x-clipeditor-asset"
RULER_H = 26
HEADER_W = 96
VIDEO_H = 62
AUDIO_H = 46
EDGE = 7
SNAP_PX = 8

C_BG = QColor(30, 31, 35)
C_ROW = QColor(38, 40, 45)
C_ROW2 = QColor(34, 36, 40)
C_HEAD = QColor(46, 48, 54)
C_TEXT = QColor(210, 212, 218)
C_DIM = QColor(130, 134, 142)
C_VIDEO = QColor(52, 104, 168)
C_IMAGE = QColor(118, 86, 170)
C_AUDIO = QColor(46, 132, 92)
C_TEXTCLIP = QColor(196, 120, 40)
C_SEL = QColor(255, 204, 64)
C_PLAY = QColor(240, 70, 70)


def fmt_time(t: float, fps: float = 30.0, frames: bool = False) -> str:
    t = max(t, 0.0)
    m, s = divmod(int(t), 60)
    if frames:
        return f"{m:02d}:{s:02d}:{int((t - int(t)) * fps):02d}"
    return f"{m:02d}:{s:02d}"


class TimelineCanvas(QWidget):
    seek = Signal(float)
    scroll_changed = Signal()
    files_dropped = Signal(list, object, float)  # paths, track_id|None, time
    transition_added = Signal()
    sequence_open_requested = Signal(str)

    def __init__(self, editor: Editor, loader):
        super().__init__()
        self.ed = editor
        self.loader = loader
        self.pps = 60.0  # pixels per second
        self.x_off = 0.0
        self.playhead = 0.0
        self.drag: dict | None = None
        self.snap_x: float | None = None
        self.snapping = True
        self._shape_thumbs: dict = {}
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setMinimumHeight(RULER_H + 2 * VIDEO_H + 2 * AUDIO_H)

    # ---- geometry --------------------------------------------------------
    def t2x(self, t: float) -> float:
        return HEADER_W + t * self.pps - self.x_off

    def x2t(self, x: float) -> float:
        return (x - HEADER_W + self.x_off) / self.pps

    def rows(self):
        y = RULER_H
        for tr in self.ed.project.tracks:
            h = VIDEO_H if tr.kind == "video" else AUDIO_H
            yield tr, QRect(0, y, self.width(), h)
            y += h

    def total_height(self) -> int:
        return RULER_H + sum(VIDEO_H if t.kind == "video" else AUDIO_H for t in self.ed.project.tracks)

    def track_at(self, y: float) -> Track | None:
        for tr, r in self.rows():
            if r.top() <= y < r.bottom():
                return tr
        return None

    def clip_rect(self, row: QRect, clip) -> QRectF:
        x0, x1 = self.t2x(clip.start), self.t2x(clip.end)
        return QRectF(x0, row.top() + 3, max(x1 - x0, 2), row.height() - 6)

    def cut_points(self, tr: Track, row: QRect):
        """(x, incoming clip, has_transition) for every cut between touching/overlapping clips."""
        out = []
        for b in tr.clips:
            if b.transition > 0 and prev_clip(tr, b) is not None:
                out.append((self.t2x(b.start + b.transition / 2), b, True))
            elif neighbour_before(tr, b) is not None:
                out.append((self.t2x(b.start), b, False))
        return out

    def hit(self, pos: QPointF):
        for tr, row in self.rows():
            if row.top() <= pos.y() < row.bottom():
                if pos.y() >= row.bottom() - 22:
                    for x, b, _ in self.cut_points(tr, row):
                        if abs(pos.x() - x) <= 9:
                            return tr, b, "cut"
                for c in reversed(tr.sorted_clips()):  # top-most (incoming) clip wins in overlaps
                    r = self.clip_rect(row, c)
                    if r.left() - 2 <= pos.x() <= r.right() + 2:
                        if pos.y() <= r.top() + 14 and r.width() > 40:
                            fi = self.t2x(c.start + c.fade_in)
                            fo = self.t2x(c.end - c.fade_out)
                            if abs(pos.x() - fi) <= 6 and pos.x() - r.left() > EDGE:
                                return tr, c, "fade_in"
                            if abs(pos.x() - fo) <= 6 and r.right() - pos.x() > EDGE:
                                return tr, c, "fade_out"
                        if pos.x() - r.left() <= EDGE:
                            return tr, c, "trim_l"
                        if r.right() - pos.x() <= EDGE:
                            return tr, c, "trim_r"
                        return tr, c, "move"
                return tr, None, None
        return None, None, None

    def content_width(self) -> float:
        return (self.ed.project.duration + 30) * self.pps

    def set_zoom(self, pps: float, anchor_x: float | None = None):
        anchor_x = anchor_x if anchor_x is not None else self.t2x(self.playhead)
        t = self.x2t(anchor_x)
        self.pps = min(max(pps, 2.0), 800.0)
        self.x_off = max(0.0, t * self.pps - (anchor_x - HEADER_W))
        self.scroll_changed.emit()
        self.update()

    def ensure_visible(self, t: float):
        x = self.t2x(t)
        if x > self.width() - 40:
            self.x_off += x - (self.width() - 40) + (self.width() - HEADER_W) * 0.5
            self.scroll_changed.emit()
        elif x < HEADER_W:
            self.x_off = max(0.0, t * self.pps - 20)
            self.scroll_changed.emit()

    # ---- snapping --------------------------------------------------------
    def snap_points(self, exclude: str | None) -> list[float]:
        pts = [0.0, self.playhead]
        for tr in self.ed.project.tracks:
            for c in tr.clips:
                if c.id != exclude:
                    pts += [c.start, c.end]
        return pts

    def snap(self, t: float, pts: list[float]) -> tuple[float, float | None]:
        if not self.snapping:
            return t, None
        best, dist = None, SNAP_PX / self.pps
        for p in pts:
            if abs(p - t) < dist:
                best, dist = p, abs(p - t)
        return (best, best) if best is not None else (t, None)

    # ---- painting --------------------------------------------------------
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        p.fillRect(self.rect(), C_BG)
        proj = self.ed.project
        font = QFont(self.font())
        font.setPointSizeF(8.5)
        p.setFont(font)

        for i, (tr, row) in enumerate(self.rows()):
            p.fillRect(row, C_ROW if i % 2 == 0 else C_ROW2)
        # clips
        p.save()
        p.setClipRect(QRect(HEADER_W, RULER_H, self.width() - HEADER_W, self.height()))
        for tr, row in self.rows():
            for c in tr.sorted_clips():
                r = self.clip_rect(row, c)
                if r.right() < HEADER_W or r.left() > self.width():
                    continue
                self.paint_clip(p, tr, c, r)
            self.paint_cuts(p, tr, row)
        p.restore()
        # headers
        for tr, row in self.rows():
            h = QRect(0, row.top(), HEADER_W, row.height())
            p.fillRect(h, C_HEAD)
            p.setPen(C_TEXT)
            p.drawText(h.adjusted(8, 4, 0, 0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, tr.name)
            for label, attr, rr in self.header_buttons(tr, row):
                on = getattr(tr, attr)
                p.fillRect(rr, QColor(200, 70, 70) if on else QColor(64, 67, 74))
                p.setPen(C_TEXT)
                p.drawText(rr, Qt.AlignmentFlag.AlignCenter, label)
            p.setPen(QColor(24, 25, 28))
            p.drawLine(0, row.bottom(), self.width(), row.bottom())
        # ruler
        p.fillRect(QRect(0, 0, self.width(), RULER_H), QColor(42, 44, 50))
        p.fillRect(QRect(0, 0, HEADER_W, RULER_H), C_HEAD)
        p.setPen(C_DIM)
        p.drawText(QRect(6, 0, HEADER_W, RULER_H), Qt.AlignmentFlag.AlignVCenter, fmt_time(self.playhead, proj.fps, True))
        step = next(s for s in (0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600) if s * self.pps >= 70)
        t = math.floor(self.x2t(HEADER_W) / step) * step
        while self.t2x(t) < self.width():
            x = self.t2x(t)
            if x >= HEADER_W:
                p.setPen(C_DIM)
                p.drawLine(int(x), RULER_H - 8, int(x), RULER_H)
                label = fmt_time(t) if step >= 1 else f"{t:.2f}s"
                p.drawText(int(x) + 3, RULER_H - 10, label)
                for k in range(1, 5):
                    xs = self.t2x(t + step * k / 5)
                    if xs >= HEADER_W:
                        p.drawLine(int(xs), RULER_H - 4, int(xs), RULER_H)
            t += step
        if not any(t.clips for t in proj.tracks):
            p.setPen(C_DIM)
            hint_font = QFont(self.font())
            hint_font.setPointSizeF(11)
            p.setFont(hint_font)
            p.drawText(QRect(HEADER_W, RULER_H, self.width() - HEADER_W, self.height() - RULER_H),
                       Qt.AlignmentFlag.AlignCenter,
                       "Drag clips here from the Media panel, or drop files from Explorer")
            p.setFont(font)
        # In/Out range: band over the tracks + bracket on the ruler
        rng = self.ed.range()
        for mark in (self.ed.mark_in, self.ed.mark_out):
            if mark is not None and rng is None:
                x = int(self.t2x(mark))
                if x >= HEADER_W:
                    p.setPen(QPen(QColor(90, 170, 255), 2))
                    p.drawLine(x, 0, x, RULER_H)
        if rng is not None:
            x0, x1 = max(self.t2x(rng[0]), HEADER_W), self.t2x(rng[1])
            if x1 > HEADER_W:
                p.fillRect(QRectF(x0, RULER_H, x1 - x0, self.height() - RULER_H), QColor(60, 140, 255, 45))
                p.fillRect(QRectF(x0, 0, x1 - x0, RULER_H), QColor(60, 140, 255, 150))
                p.setPen(QPen(QColor(120, 185, 255), 1))
                p.drawLine(QPointF(x0, 0), QPointF(x0, self.height()))
                p.drawLine(QPointF(x1, 0), QPointF(x1, self.height()))
                p.setPen(QColor(255, 255, 255))
                p.drawText(QRectF(x0 + 4, 0, max(x1 - x0 - 6, 0), RULER_H), Qt.AlignmentFlag.AlignVCenter,
                           f"{rng[1] - rng[0]:.2f}s")
        # snap guide + playhead
        if self.snap_x is not None:
            p.setPen(QPen(QColor(255, 255, 255, 120), 1, Qt.PenStyle.DashLine))
            x = int(self.t2x(self.snap_x))
            p.drawLine(x, RULER_H, x, self.height())
        x = self.t2x(self.playhead)
        if x >= HEADER_W:
            p.setPen(QPen(C_PLAY, 1.5))
            p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
            path = QPainterPath()
            path.moveTo(x - 6, 0)
            path.lineTo(x + 6, 0)
            path.lineTo(x, 9)
            path.closeSubpath()
            p.fillPath(path, C_PLAY)
        p.end()

    def header_buttons(self, tr: Track, row: QRect):
        y = row.bottom() - 20
        btns = [("M", "muted", QRect(8, y, 20, 16))]
        if tr.kind == "video":
            btns.append(("H", "hidden", QRect(32, y, 20, 16)))
        return btns

    def paint_clip(self, p: QPainter, tr: Track, c, r: QRectF):
        asset = self.ed.project.asset_for_clip(c)
        if asset is None and not c.is_generated:
            return
        if c.is_text:
            base = C_TEXTCLIP
        elif c.is_shape:
            base = QColor(176, 70, 140)
        elif c.sequence_id:
            base = QColor(113, 75, 164)
        else:
            base = C_AUDIO if tr.kind == "audio" else (C_IMAGE if asset.kind == "image" else C_VIDEO)
        p.fillRect(r, base.darker(130))
        p.save()
        p.setClipRect(r.intersected(QRectF(HEADER_W, 0, self.width(), self.height())))
        if c.is_shape:
            try:
                from .. import shapes

                key = repr(sorted(c.shape.items()))
                th = self._shape_thumbs.get(key)
                if th is None:
                    th = self._shape_thumbs[key] = shapes.render_shape(c, int(r.height() - 16) * 3)
                hh = r.height() - 16
                tw = th.width() * hh / max(th.height(), 1)
                p.drawImage(QRectF(r.left() + 4, r.top() + 15, min(tw, hh * 3), hh), th)
            except Exception:  # noqa: BLE001
                pass
        elif c.is_text:
            f = QFont(self.font())
            f.setPointSizeF(11)
            f.setBold(True)
            p.setFont(f)
            p.setPen(QColor(255, 236, 210))
            p.drawText(r.adjusted(6, 14, -4, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                       "T   " + c.text.replace("\n", " "))
            f.setPointSizeF(8.5)
            f.setBold(False)
            p.setFont(f)
        elif c.sequence_id:
            p.setPen(QColor(231, 216, 247))
            p.drawText(r.adjusted(6, 14, -4, 0), Qt.AlignmentFlag.AlignVCenter, "Nested sequence")
        elif tr.kind == "video":
            th = self.loader.thumb_at(asset.id, c.in_point)
            if th is not None:
                tw = th.width() * (r.height() - 14) / max(th.height(), 1)
                x = r.left()
                while x < r.right() and tw > 4:
                    if x + tw >= HEADER_W:
                        t = c.in_point if asset.kind == "image" else c.source_time(self.x2t(x))
                        img = self.loader.thumb_at(asset.id, t) or th
                        p.drawImage(QRectF(x, r.top() + 14, tw, r.height() - 14), img)
                    x += tw
        else:
            peaks = self.loader.audio.peaks(asset.id)
            if peaks is not None and len(peaks):
                p.setPen(QColor(160, 230, 190))
                mid = r.center().y() + 5
                amp = (r.height() - 16) / 2
                x0 = int(max(r.left(), HEADER_W))
                x1 = int(min(r.right(), self.width()))
                for x in range(x0, x1):
                    s0 = int(c.source_time(self.x2t(x)) * 100)
                    s1 = max(int(c.source_time(self.x2t(x + 1)) * 100), s0 + 1)
                    if 0 <= s0 < len(peaks):
                        v = float(peaks[s0:s1].max()) * min(c.volume, 2.0) * amp
                        p.drawLine(x, int(mid - v), x, int(mid + v))
        # title strip
        p.fillRect(QRectF(r.left(), r.top(), r.width(), 14), QColor(base.red(), base.green(), base.blue(), 230))
        p.setPen(QColor(240, 240, 245))
        if c.sequence_id:
            label = self.ed.project.timelines[c.sequence_id].name
        else:
            label = "Text" if c.is_text else "Sticker" if c.is_shape else asset.name
        badges = []
        if c.effects:
            badges.append("fx")
        if c.anim_in or c.anim_out or c.anim_loop:
            badges.append("🌀")
        if c.audio_fx:
            badges.append("🎛")
        if c.masks or c.chroma:
            badges.append("◐")
        if badges:
            label += "   " + " ".join(badges)
        if asset is not None and c.volume == 0 and asset.has_audio:
            label += "  · muted"
        if c.speed_ramp:
            label += "  · Speed ramp"
        elif abs(c.speed - 1) > 1e-3:
            label += f"  · {c.speed:g}x"
        p.drawText(QRectF(r.left() + 4, r.top(), r.width() - 6, 14), Qt.AlignmentFlag.AlignVCenter, label)
        # fades: shaded triangles + draggable handles
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        shade = QColor(0, 0, 0, 110)
        if c.fade_in > 0:
            x = self.t2x(c.start + c.fade_in)
            path = QPainterPath()
            path.moveTo(r.left(), r.top() + 14)
            path.lineTo(x, r.top() + 14)
            path.lineTo(r.left(), r.bottom())
            path.closeSubpath()
            p.fillPath(path, shade)
        if c.fade_out > 0:
            x = self.t2x(c.end - c.fade_out)
            path = QPainterPath()
            path.moveTo(r.right(), r.top() + 14)
            path.lineTo(x, r.top() + 14)
            path.lineTo(r.right(), r.bottom())
            path.closeSubpath()
            p.fillPath(path, shade)
        try:
            from ..animate import all_keyframe_times

            p.setBrush(QColor(255, 204, 64))
            p.setPen(QColor(40, 30, 0))
            for kt in all_keyframe_times(c):
                if not 0 <= kt <= c.duration:
                    continue
                x = self.t2x(c.start + kt)
                dm = QPainterPath()
                dm.moveTo(x, r.bottom() - 10)
                dm.lineTo(x + 5, r.bottom() - 5)
                dm.lineTo(x, r.bottom())
                dm.lineTo(x - 5, r.bottom() - 5)
                dm.closeSubpath()
                p.drawPath(dm)
        except Exception:  # noqa: BLE001
            pass
        if c.id in self.ed.selection and r.width() > 40:
            p.setBrush(QColor(255, 255, 255))
            p.setPen(QColor(0, 0, 0))
            for x in (self.t2x(c.start + c.fade_in), self.t2x(c.end - c.fade_out)):
                p.drawEllipse(QPointF(x, r.top() + 7), 4, 4)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        p.restore()
        sel = c.id in self.ed.selection
        p.setPen(QPen(C_SEL if sel else base.lighter(140), 2 if sel else 1))
        p.drawRect(r.adjusted(0.5, 0.5, -0.5, -0.5))

    def paint_cuts(self, p: QPainter, tr: Track, row: QRect):
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for b in tr.clips:  # transition badges over the overlap
            if b.transition > 0 and prev_clip(tr, b) is not None:
                x0, x1 = self.t2x(b.start), self.t2x(b.start + b.transition)
                rr = QRectF(x0, row.top() + 17, max(x1 - x0, 3), row.height() - 22)
                p.fillRect(rr, QColor(255, 255, 255, 60))
                p.setPen(QPen(QColor(255, 255, 255, 170), 1, Qt.PenStyle.DashLine))
                p.drawLine(QPointF(x0, rr.bottom()), QPointF(x1, rr.top()))
                p.drawLine(QPointF(x0, rr.top()), QPointF(x1, rr.bottom()))
        for x, b, has in self.cut_points(tr, row):
            c = QPointF(x, row.bottom() - 11)
            sel = b.id in self.ed.selection and has
            p.setPen(QPen(QColor(20, 20, 24), 1))
            p.setBrush(QColor(255, 204, 64) if sel else QColor(250, 250, 252) if has else QColor(150, 155, 165))
            path = QPainterPath()
            path.moveTo(c.x(), c.y() - 8)
            path.lineTo(c.x() + 8, c.y())
            path.lineTo(c.x(), c.y() + 8)
            path.lineTo(c.x() - 8, c.y())
            path.closeSubpath()
            p.drawPath(path)
            p.setPen(QColor(20, 20, 24))
            p.drawText(QRectF(c.x() - 8, c.y() - 8, 16, 16), Qt.AlignmentFlag.AlignCenter, "⇄" if has else "+")
        p.restore()

    # ---- mouse -----------------------------------------------------------
    def mousePressEvent(self, e):
        pos = e.position()
        if e.button() == Qt.MouseButton.RightButton:
            tr, c, _ = self.hit(pos)
            if c is not None and c.id not in self.ed.selection:
                self.ed.selection = {c.id}
                self.ed.notify("selection")
            self.context_menu(e.globalPosition().toPoint(), tr)
            return
        if e.button() != Qt.MouseButton.LeftButton:
            return
        if pos.x() < HEADER_W and pos.y() >= RULER_H:
            tr = self.track_at(pos.y())
            if tr:
                for _, attr, rr in self.header_buttons(tr, next(r for t, r in self.rows() if t is tr)):
                    if rr.contains(pos.toPoint()):
                        self.ed.toggle_track(tr.id, attr)
                        return
            return
        if pos.y() < RULER_H and e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            t = self.ed.snap_to_frame(max(self.x2t(pos.x()), 0.0))
            self.ed.mark_in = self.ed.mark_out = t
            self.drag = {"mode": "range", "t0": t}
            self.ed.notify("marks")
            return
        if pos.y() < RULER_H:
            self.drag = {"mode": "scrub"}
            self.seek.emit(max(self.x2t(pos.x()), 0.0))
            return
        tr, c, mode = self.hit(pos)
        if mode == "cut":
            self.ed.selection = {c.id}
            if c.transition <= 0:
                self.ed.set_transition(c.id)  # default dissolve, 0.5 s
                self.transition_added.emit()
            else:
                self.ed.notify("selection")
            return
        if c is None:
            if self.ed.selection:
                self.ed.selection.clear()
                self.ed.notify("selection")
            self.drag = {"mode": "scrub"}
            self.seek.emit(max(self.x2t(pos.x()), 0.0))
            return
        ctrl = e.modifiers() & Qt.KeyboardModifier.ControlModifier
        if ctrl:
            self.ed.selection ^= {c.id}
        elif c.id not in self.ed.selection:
            self.ed.selection = {c.id}
        self.ed.notify("selection")
        self.drag = {
            "mode": mode, "clip": c.id, "snap": self.ed.snapshot(), "x0": pos.x(),
            "orig": (c.start, c.in_point, c.out_point), "track": tr.id, "moved": False,
            "keyframes": c.keyframes,
            "pts": self.snap_points(c.id),
        }

    def mouseMoveEvent(self, e):
        pos = e.position()
        d = self.drag
        if d is None:
            _, c, mode = self.hit(pos)
            shape = (Qt.CursorShape.SizeHorCursor if mode in ("trim_l", "trim_r") else
                     Qt.CursorShape.PointingHandCursor if mode in ("fade_in", "fade_out", "cut") else
                     Qt.CursorShape.OpenHandCursor if mode == "move" else Qt.CursorShape.ArrowCursor)
            self.setCursor(QCursor(shape))
            if mode == "cut":
                self.setToolTip("Transition: click to edit" if c.transition > 0 else "Click to add a transition here")
            elif mode in ("fade_in", "fade_out"):
                self.setToolTip("Drag to fade")
            else:
                self.setToolTip("")
            return
        if d["mode"] == "range":
            t = self.ed.snap_to_frame(min(max(self.x2t(pos.x()), 0.0), self.ed.project.duration))
            t, g = self.snap(t, [0.0, self.playhead] + [x for tr in self.ed.project.tracks
                                                         for c in tr.clips for x in (c.start, c.end)])
            self.snap_x = g
            self.ed.mark_in, self.ed.mark_out = min(d["t0"], t), max(d["t0"], t)
            self.ed.notify("marks")
            return
        if d["mode"] == "scrub":
            self.seek.emit(max(self.x2t(pos.x()), 0.0))
            return
        tr, c = self.ed.project.find_clip(d["clip"])
        if c is None:
            return
        dt = (pos.x() - d["x0"]) / self.pps
        if abs(pos.x() - d["x0"]) < 2 and not d["moved"]:
            return
        d["moved"] = True
        start0, in0, out0 = d["orig"]
        asset = self.ed.project.asset_for_clip(c)
        max_src = asset.duration if asset is not None else 3600.0
        if c.sequence_id:
            max_src = max(max_src, out0)
        self.snap_x = None
        if d["mode"] == "move":
            ns = max(0.0, start0 + dt)
            s1, g1 = self.snap(ns, d["pts"])
            s2, g2 = self.snap(ns + c.duration, d["pts"])
            if g1 is not None:
                ns, self.snap_x = s1, g1
            elif g2 is not None:
                ns, self.snap_x = s2 - c.duration, g2
            c.start = max(ns, 0.0)
            target = self.track_at(pos.y())
            if target is not None and target is not tr and target.kind == tr.kind and target.accepts(asset):
                tr.clips.remove(c)
                target.clips.append(c)
        elif d["mode"] == "trim_l":
            from ..animate import shift_keyframes

            ns, g = self.snap(start0 + dt, d["pts"])
            self.snap_x = g
            delta = ns - start0  # timeline seconds
            lo = -start0 if c.is_text else max(c.time_between(in0, 0.0), -start0)
            delta = min(max(delta, lo), c.time_between(in0, out0) - MIN_CLIP)
            c.start, c.in_point = start0 + delta, c.source_after(in0, delta)
            c.keyframes = shift_keyframes(d["keyframes"], delta)
        elif d["mode"] == "trim_r":
            ne, g = self.snap(start0 + c.time_between(in0, out0) + dt, d["pts"])
            self.snap_x = g
            c.out_point = min(c.source_after(in0, max(ne - start0, MIN_CLIP)), max_src)
        elif d["mode"] == "fade_in":
            c.fade_in = round(min(max(self.x2t(pos.x()) - c.start, 0.0), c.duration - c.fade_out), 2)
        elif d["mode"] == "fade_out":
            c.fade_out = round(min(max(c.end - self.x2t(pos.x()), 0.0), c.duration - c.fade_in), 2)
        self.ed.notify("live")

    def mouseReleaseEvent(self, e):
        d, self.drag = self.drag, None
        self.snap_x = None
        if d and d.get("moved"):
            tr, c = self.ed.project.find_clip(d["clip"])
            if c is not None:
                if not self.ed.is_magnetic(tr):  # magnetic main track re-packs instead of overwriting
                    tr.clear_range(c.start, c.end, keep_id=c.id)
            self.ed.checkpoint(d["snap"])
            self.ed.notify("edit")
        self.update()

    def mouseDoubleClickEvent(self, event):
        _, clip, mode = self.hit(event.position())
        if event.button() == Qt.MouseButton.LeftButton and mode == "move" and clip is not None and clip.sequence_id:
            self.drag = None
            self.sequence_open_requested.emit(clip.sequence_id)
        else:
            super().mouseDoubleClickEvent(event)

    def wheelEvent(self, e):
        dy = e.angleDelta().y() or e.angleDelta().x()
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.set_zoom(self.pps * (1.2 if dy > 0 else 1 / 1.2), e.position().x())
        else:
            self.x_off = max(0.0, self.x_off - dy)
            self.scroll_changed.emit()
            self.update()

    def context_menu(self, gpos, tr):
        m = QMenu(self)
        has_sel = bool(self.ed.selection)
        m.addAction("Split at playhead (S)", lambda: self.ed.split(self.playhead))
        a = m.addAction("Delete", self.ed.delete_selected)
        a.setEnabled(has_sel)
        a = m.addAction("Ripple delete", lambda: self.ed.delete_selected(ripple=True))
        a.setEnabled(has_sel)
        if len(self.ed.selection) == 1:
            _tr, sc = self.ed.project.find_clip(next(iter(self.ed.selection)))
            if sc is not None and sc.transition > 0:
                tm = m.addMenu("Transition")
                for k, label in KINDS.items():
                    tm.addAction(label, lambda k=k, cid=sc.id: self.ed.set_transition(cid, k))
                tm.addSeparator()
                tm.addAction("Remove transition", lambda cid=sc.id: self.ed.remove_transition(cid))
            elif sc is not None and _tr is not None and neighbour_before(_tr, sc) is not None:
                tm = m.addMenu("Add transition from previous clip")
                for k, label in KINDS.items():
                    tm.addAction(label, lambda k=k, cid=sc.id: self.ed.set_transition(cid, k))
        a = m.addAction("Duplicate (Ctrl+D)", self.ed.duplicate_selected)
        a.setEnabled(has_sel)
        a = m.addAction("Detach audio (Ctrl+Shift+D)", self.ed.detach_audio)
        a.setEnabled(has_sel)
        m.addAction("Add text at playhead (T)", lambda: self.ed.add_text(start=self.playhead))
        m.addSeparator()
        m.addAction("Add video track", lambda: self.ed.add_track("video"))
        m.addAction("Add audio track", lambda: self.ed.add_track("audio"))
        if tr is not None:
            m.addAction(f"Remove track {tr.name}", lambda: self.ed.remove_track(tr.id))
        m.exec(gpos)

    # ---- drag & drop from media bin -------------------------------------
    def dragEnterEvent(self, e):
        if e.mimeData().hasFormat(ASSET_MIME) or e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        if e.mimeData().hasFormat(ASSET_MIME) or e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        if e.mimeData().hasUrls() and not e.mimeData().hasFormat(ASSET_MIME):
            paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
            tr = self.track_at(e.position().y())
            self.files_dropped.emit(paths, tr.id if tr else None, max(self.x2t(e.position().x()), 0.0))
            e.acceptProposedAction()
            return
        aid = bytes(e.mimeData().data(ASSET_MIME)).decode()
        if aid not in self.ed.project.assets:
            return
        pos = e.position()
        tr = self.track_at(pos.y())
        t, _ = self.snap(max(self.x2t(pos.x()), 0.0), self.snap_points(None))
        self.ed.add_clip(aid, tr.id if tr else None, t)
        e.acceptProposedAction()


class TimelinePanel(QWidget):
    def __init__(self, editor: Editor, loader):
        super().__init__()
        self.canvas = TimelineCanvas(editor, loader)
        self.hbar = QScrollBar(Qt.Orientation.Horizontal)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        # tracks scroll vertically so many tracks (or a short screen) never push the window taller
        from PySide6.QtWidgets import QFrame, QScrollArea
        self.vscroll = QScrollArea()
        self.vscroll.setWidget(self.canvas)
        self.vscroll.setWidgetResizable(True)
        self.vscroll.setFrameShape(QFrame.Shape.NoFrame)
        self.vscroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.vscroll.setMinimumHeight(110)
        lay.addWidget(self.vscroll, 1)
        lay.addWidget(self.hbar)
        self.hbar.valueChanged.connect(self._on_bar)
        self.canvas.scroll_changed.connect(self.sync_bar)

    def _on_bar(self, v):
        self.canvas.x_off = float(v)
        self.canvas.update()

    def sync_bar(self):
        c = self.canvas
        view = max(c.width() - HEADER_W, 1)
        mx = int(max(c.content_width() - view, c.x_off, 0))
        self.hbar.blockSignals(True)
        self.hbar.setRange(0, mx)
        self.hbar.setPageStep(view)
        self.hbar.setValue(int(c.x_off))
        self.hbar.blockSignals(False)

    def refresh(self):
        self.canvas.setMinimumHeight(self.canvas.total_height())
        self.sync_bar()
        self.canvas.update()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.sync_bar()
