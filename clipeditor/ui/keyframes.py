"""Graph and controls for a single clip's keyframe curves."""
from __future__ import annotations

from bisect import bisect_left

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPalette, QPen, QPolygonF
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QPushButton,
                               QVBoxLayout, QWidget)

from .. import animate

PROPERTIES = {
    "scale": ("Scale", 0.05, 5.0, 0.25, "×"),
    "pos_x": ("Horizontal position", -1.5, 1.5, 0.4, ""),
    "pos_y": ("Vertical position", -1.5, 1.5, 0.4, ""),
    "rotation": ("Rotation", -360.0, 360.0, 90.0, "°"),
    "opacity": ("Opacity", 0.0, 1.0, 1.0, ""),
    "volume": ("Volume", 0.0, 2.0, 1.0, "×"),
}


def _bounded_time(keys, index, duration, time):
    low = max(0.0, keys[index - 1][0] + animate.KF_TOLERANCE) if index else 0.0
    high = duration
    if index + 1 < len(keys):
        high = min(high, keys[index + 1][0] - animate.KF_TOLERANCE)
    return min(max(time, low), high) if low <= high else keys[index][0]


class CurvePlot(QWidget):
    selected = Signal(int)
    scrubbed = Signal(float)
    added = Signal(float, float)
    moved = Signal(int, float, float)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(240, 178)
        self.setAccessibleName("Keyframe curve")
        self.setToolTip("Click to scrub. Double-click to add a keyframe. Drag a diamond to move it.")
        self.keys = []
        self.vertices = [(0.0, 0.0), (1.0, 0.0)]
        self.duration = 1.0
        self.base = 0.0
        self.low, self.high = 0.0, 1.0
        self.playhead = 0.0
        self.selection = None
        self._press = None
        self._drag = None

    def set_curve(self, keys, duration, base, span, selection):
        self.keys, self.duration, self.base, self.selection = keys, max(duration, 1e-9), base, selection
        self.vertices = self._curve_vertices(keys)
        values = [v for _, v in self.vertices]
        midpoint = (min(values) + max(values)) / 2
        margin = max((max(values) - min(values)) * 0.65, span / 2)
        self.low, self.high = midpoint - margin, midpoint + margin
        self._press = self._drag = None
        self.update()

    def set_playhead(self, time):
        self.playhead = time
        self.update()

    def _area(self):
        return QRectF(self.rect()).adjusted(43, 10, -12, -24)

    def _position(self, time, value):
        area = self._area()
        return QPointF(area.left() + area.width() * time / self.duration,
                       area.bottom() - area.height() * (value - self.low) / (self.high - self.low))

    def _values(self, point):
        area = self._area()
        x = min(max((point.x() - area.left()) / area.width(), 0.0), 1.0)
        y = min(max((area.bottom() - point.y()) / area.height(), 0.0), 1.0)
        return x * self.duration, self.low + y * (self.high - self.low)

    def _hit(self, point):
        nearest, distance = None, 10.0
        for i, (t, v, _) in enumerate(self.keys):
            delta = (point - self._position(t, v)).manhattanLength()
            if 0 <= t <= self.duration and delta < distance:
                nearest, distance = i, delta
        return nearest

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        palette = self.palette()
        painter.fillRect(self.rect(), palette.color(QPalette.ColorRole.Base))
        area = self._area()
        grid = palette.color(QPalette.ColorRole.Text)
        grid.setAlpha(55)
        for i in range(5):
            fraction = i / 4
            x = area.left() + fraction * area.width()
            y = area.bottom() - fraction * area.height()
            painter.setPen(grid)
            painter.drawLine(QPointF(x, area.top()), QPointF(x, area.bottom()))
            painter.drawLine(QPointF(area.left(), y), QPointF(area.right(), y))
            painter.setPen(palette.color(QPalette.ColorRole.Text))
            if i % 2 == 0:
                value = self.low + fraction * (self.high - self.low)
                painter.drawText(QRectF(1, y - 8, 37, 16), Qt.AlignmentFlag.AlignRight, f"{value:.3g}")
                width = 52.0
                left = min(max(x - width / 2, 1.0), self.width() - width - 1)
                painter.drawText(QRectF(left, area.bottom() + 4, width, 18), Qt.AlignmentFlag.AlignHCenter,
                                 f"{fraction * self.duration:.3g}s")

        keys = self.keys
        drag_index = None
        if self._drag is not None:
            drag_index, t, value = self._drag
            keys = list(keys)
            keys[drag_index] = t, value, keys[drag_index][2]
        painter.save()
        painter.setClipRect(area.adjusted(-0.5, -0.5, 0.5, 0.5))
        if 0 <= self.playhead <= self.duration:
            x = self._position(self.playhead, self.low).x()
            painter.setPen(QPen(QColor("#f37f70"), 1))
            painter.drawLine(QPointF(x, area.top()), QPointF(x, area.bottom()))
        painter.setPen(QPen(QColor("#69c2ff"), 2))
        vertices = self.vertices if drag_index is None else self._curve_vertices(keys)
        path = QPainterPath(self._position(*vertices[0]))
        for t, value in vertices[1:]:
            path.lineTo(self._position(t, value))
        painter.drawPath(path)
        painter.restore()
        for i, (t, value, _) in enumerate(keys):
            if 0 <= t <= self.duration and self.low <= value <= self.high:
                p = self._position(t, value)
                painter.setPen(QPen(palette.color(QPalette.ColorRole.Base), 1))
                painter.setBrush(QColor("#ffd273" if i == self.selection or i == drag_index else "#69c2ff"))
                painter.drawPolygon(QPolygonF([p + QPointF(0, -5), p + QPointF(5, 0),
                                               p + QPointF(0, 5), p + QPointF(-5, 0)]))

    def _curve_vertices(self, keys):
        if not keys:
            return [(0, self.base), (self.duration, self.base)]
        t, value, _ = keys[0]
        vertices = [(min(0, t), value), (t, value)]
        for (t0, v0, ease), (t1, v1, _) in zip(keys, keys[1:]):
            if ease == "hold":
                vertices.append((t1, v0))
            elif ease != "linear":
                for i in range(1, 32):
                    u = i / 32
                    vertices.append((t0 + (t1 - t0) * u, v0 + (v1 - v0) * animate.ease(u, ease)))
            vertices.append((t1, v1))
        t, value, _ = keys[-1]
        vertices.append((max(self.duration, t), value))
        return vertices

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        index = self._hit(event.position())
        if index is not None:
            self.selected.emit(index)
            self._press = index, event.position()
        elif self._area().contains(event.position()):
            self.scrubbed.emit(self._values(event.position())[0])

    def mouseMoveEvent(self, event):
        if self._press is None:
            return
        i, origin = self._press
        if self._drag is None and (event.position() - origin).manhattanLength() < 3:
            return
        t, value = self._values(event.position())
        self._drag = i, _bounded_time(self.keys, i, self.duration, t), value
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            drag, self._drag, self._press = self._drag, None, None
            if drag is not None:
                self.moved.emit(*drag)
                self.update()

    def mouseDoubleClickEvent(self, event):
        if (event.button() == Qt.MouseButton.LeftButton and self._area().contains(event.position())
                and self._hit(event.position()) is None):
            self.added.emit(*self._values(event.position()))


class KeyframeEditor(QWidget):
    def __init__(self, editor, seek):
        super().__init__()
        self.editor, self.seek = editor, seek
        self.clip = None
        self.time = 0.0
        self.selected_time = None
        self.index = None
        self.loading = False
        self.copied = None

        self.property_choice = QComboBox()
        self.plot = CurvePlot()
        self.key_time = QDoubleSpinBox()
        self.key_time.setDecimals(3)
        self.key_time.setSuffix(" s")
        self.value = QDoubleSpinBox()
        self.value.setDecimals(4)
        self.easing = QComboBox()
        self.easing.setToolTip("Interpolation from the selected keyframe to the next one")
        for easing in animate.EASINGS:
            self.easing.addItem(easing.replace("_", " ").capitalize(), easing)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        form = QFormLayout()
        form.addRow("Property", self.property_choice)
        layout.addLayout(form)
        layout.addWidget(self.plot)
        edit = QHBoxLayout()
        self.add_key = QPushButton("Add at playhead")
        self.remove_key = QPushButton("Remove selected")
        for button in (self.add_key, self.remove_key):
            edit.addWidget(button)
        layout.addLayout(edit)
        selected = QFormLayout()
        selected.addRow("Clip time", self.key_time)
        selected.addRow("Value", self.value)
        selected.addRow("Outgoing easing", self.easing)
        layout.addLayout(selected)
        clipboard = QHBoxLayout()
        self.copy_curve = QPushButton("Copy curve")
        self.paste_curve = QPushButton("Paste curve")
        self.paste_curve.setToolTip("Replace this property's curve and scale its times to fit the clip")
        clipboard.addWidget(self.copy_curve)
        clipboard.addWidget(self.paste_curve)
        layout.addLayout(clipboard)
        self.hint = QLabel("Double-click to add a point. Drag a diamond to move it.")
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)
        layout.addStretch(1)

        self.property_choice.currentIndexChanged.connect(self._property_changed)
        self.plot.selected.connect(self._select)
        self.plot.scrubbed.connect(self._scrub)
        self.plot.added.connect(self._add)
        self.plot.moved.connect(self._drag_key)
        self.add_key.clicked.connect(self._add_at_playhead)
        self.remove_key.clicked.connect(self._remove)
        self.key_time.editingFinished.connect(self._edit_time)
        self.value.editingFinished.connect(self._edit_value)
        self.easing.activated.connect(self._edit_easing)
        self.copy_curve.clicked.connect(self._copy)
        self.paste_curve.clicked.connect(self._paste)

    def set_clip(self, clip, properties):
        if self.clip is None or clip is None or clip.id != self.clip.id:
            self.selected_time = None
        self.clip = clip
        if properties != tuple(self.property_choice.itemData(i) for i in range(self.property_choice.count())):
            prop = self.property_choice.currentData()
            self.property_choice.blockSignals(True)
            self.property_choice.clear()
            for name in properties:
                self.property_choice.addItem(PROPERTIES[name][0], name)
            if self.property_choice.findData(prop) >= 0:
                self.property_choice.setCurrentIndex(self.property_choice.findData(prop))
            self.property_choice.blockSignals(False)
        self._refresh()

    def set_playhead(self, time):
        self.time = time
        if self.clip is not None:
            self.plot.set_playhead(time - self.clip.start)

    def _property_changed(self):
        self.selected_time = None
        self._refresh()

    def _refresh(self):
        prop = self.property_choice.currentData()
        if self.clip is None or prop is None:
            return
        self.loading = True
        self.keys = animate.keyframes_for(self.clip, prop)
        self.index = next((i for i, key in enumerate(self.keys) if key[0] == self.selected_time), None)
        _, low, high, span, suffix = PROPERTIES[prop]
        self.plot.set_curve(self.keys, self.clip.duration, float(getattr(self.clip, prop)), span, self.index)
        self.set_playhead(self.time)
        selected = self.index is not None
        for widget in (self.key_time, self.value, self.easing, self.remove_key):
            widget.setEnabled(selected)
        self.copy_curve.setEnabled(bool(self.keys))
        self.paste_curve.setEnabled(self.copied is not None and self.copied[0] == prop)
        self.key_time.setRange(0, self.clip.duration)
        self.value.setRange(low, high)
        self.value.setSuffix(suffix)
        if selected:
            time, value, easing = self.keys[self.index]
            self.key_time.setValue(time)
            self.value.setRange(min(low, value), max(high, value))
            self.value.setValue(value)
            self.easing.setCurrentIndex(max(self.easing.findData(easing), 0))
        else:
            self.key_time.clear()
            self.value.clear()
            self.easing.setCurrentIndex(-1)
        self.loading = False

    def _select(self, index):
        self.selected_time = self.keys[index][0]
        self._refresh()
        self._scrub(self.selected_time)

    def _scrub(self, time):
        self.seek(self.clip.start + time)

    def _replace(self, keys, select=None):
        self.selected_time = select
        self.editor.set_keyframes(self.clip.id, self.property_choice.currentData(), keys)
        self._refresh()

    def _add_at_playhead(self):
        t = min(max(self.time - self.clip.start, 0.0), self.clip.duration)
        self._add(t, animate.value_at(self.clip, self.property_choice.currentData(), t))

    def _add(self, t, value):
        t = self._snap(t)
        value = min(max(value, self.value.minimum()), self.value.maximum())
        near = next((i for i, key in enumerate(self.keys) if abs(key[0] - t) < animate.KF_TOLERANCE), None)
        if near is not None:
            self._select(near)
            return
        keys = list(self.keys)
        keys.insert(bisect_left([key[0] for key in keys], t), (t, value, "ease"))
        self._replace(keys, t)
        self._scrub(t)

    def _snap(self, local):
        if local == self.clip.duration:
            return local
        snapped = self.editor.snap_to_frame(self.clip.start + local) - self.clip.start
        return min(max(snapped, 0.0), self.clip.duration)

    def _remove(self):
        keys = list(self.keys)
        keys.pop(self.index)
        self._replace(keys)

    def _drag_key(self, index, time, value):
        self._edit(index, time=self._snap(time), value=value)
        self._scrub(self.selected_time)

    def _edit_time(self):
        if not self.loading and self.index is not None:
            original = self.keys[self.index][0]
            time = self.key_time.value()
            if abs(time - original) >= 0.0005:
                self._edit(self.index, time=time)
                self._scrub(self.selected_time)

    def _edit_value(self):
        if not self.loading and self.index is not None:
            value = self.value.value()
            if abs(value - self.keys[self.index][1]) >= 0.00005:
                self._edit(self.index, value=value)

    def _edit_easing(self):
        if self.index is not None:
            self._edit(self.index, easing=self.easing.currentData())

    def _edit(self, index, time=None, value=None, easing=None):
        keys = list(self.keys)
        t, v, e = keys[index]
        if time is not None:
            t = _bounded_time(keys, index, self.clip.duration, time)
        if value is not None:
            v = min(max(value, self.value.minimum()), self.value.maximum())
        keys[index] = (t, v, easing or e)
        self._replace(keys, t)

    def _copy(self):
        self.copied = self.property_choice.currentData(), self.clip.duration, self.keys
        self.paste_curve.setEnabled(True)

    def _paste(self):
        _, duration, keys = self.copied
        if duration > 0 and self.clip.duration > 0:
            ratio = self.clip.duration / duration
            self._replace([(t * ratio, value, easing) for t, value, easing in keys])
