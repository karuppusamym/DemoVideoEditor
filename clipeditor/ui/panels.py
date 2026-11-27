"""Media bin and clip inspector."""
from __future__ import annotations

from PySide6.QtCore import QMimeData, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QDrag, QFont, QIcon, QPixmap
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QFontComboBox, QFrame, QGridLayout,
                               QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMenu, QPlainTextEdit,
                               QPushButton, QScrollArea, QSlider, QSpinBox, QToolButton, QVBoxLayout,
                               QWidget)

from ..editor import Editor
from ..transitions import DEFAULT_DURATION, KINDS, max_duration, neighbour_before, prev_clip
from .timeline import ASSET_MIME, fmt_time


# --------------------------------------------------------------------------- media bin
class MediaBin(QWidget):
    import_requested = Signal()
    add_at_playhead = Signal(str)

    def __init__(self, editor: Editor, loader):
        super().__init__()
        self.ed = editor
        self.loader = loader
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        head = QHBoxLayout()
        title = QLabel("Media")
        title.setStyleSheet("font-size: 13pt; font-weight: 600;")
        btn = QPushButton("＋ Import")
        btn.setToolTip("Add videos, music or photos (Ctrl+I)")
        btn.clicked.connect(self.import_requested)
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(btn)
        lay.addLayout(head)
        self.list = _BinList(self)
        lay.addWidget(self.list, 1)
        self.empty = QLabel("No media yet.\n\nClick ＋ Import or drag files\nfrom Explorer into this window.")
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty.setStyleSheet("color: #8a8f99; padding: 20px;")
        lay.addWidget(self.empty)
        tip = QLabel("Tip: drag a clip onto the timeline,\nor double-click to add it at the end.")
        tip.setStyleSheet("color: #7d828c; font-size: 8.5pt;")
        lay.addWidget(tip)
        self.list.itemDoubleClicked.connect(lambda it: self.ed.add_clip(it.data(Qt.ItemDataRole.UserRole)))
        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._menu)

    def _menu(self, pos):
        it = self.list.itemAt(pos)
        if it is None:
            return
        aid = it.data(Qt.ItemDataRole.UserRole)
        m = QMenu(self)
        m.addAction("Add to end of timeline", lambda: self.ed.add_clip(aid))
        m.addAction("Insert at playhead", lambda: self.add_at_playhead.emit(aid))
        m.addSeparator()
        m.addAction("Remove from project", lambda: self.ed.remove_asset(aid))
        m.exec(self.list.mapToGlobal(pos))

    def refresh(self):
        self.list.clear()
        assets = list(self.ed.project.assets.values())
        self.empty.setVisible(not assets)
        self.list.setVisible(bool(assets))
        for a in assets:
            kind = {"video": "🎬", "audio": "🎵", "image": "🖼"}.get(a.kind, "")
            dur = fmt_time(a.duration) if a.kind != "image" else "photo"
            it = QListWidgetItem(f"{a.name}\n{kind} {dur}")
            it.setData(Qt.ItemDataRole.UserRole, a.id)
            it.setToolTip(f"{a.path}\n{a.width}x{a.height}" if a.width else a.path)
            th = self.loader.thumb_at(a.id, 0)
            pm = QPixmap(128, 72)
            pm.fill(QColor(36, 110, 78) if a.kind == "audio" else QColor(50, 52, 58))
            if th is not None:
                scaled = QPixmap.fromImage(th).scaled(128, 72, Qt.AspectRatioMode.KeepAspectRatio,
                                                      Qt.TransformationMode.SmoothTransformation)
                pm.fill(QColor(20, 20, 22))
                from PySide6.QtGui import QPainter

                p = QPainter(pm)
                p.drawPixmap((128 - scaled.width()) // 2, (72 - scaled.height()) // 2, scaled)
                p.end()
            elif a.kind == "audio":
                from PySide6.QtGui import QPainter

                p = QPainter(pm)
                f = QFont()
                f.setPointSize(26)
                p.setFont(f)
                p.setPen(QColor(220, 255, 235))
                p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, "♪")
                p.end()
            it.setIcon(QIcon(pm))
            self.list.addItem(it)


class _BinList(QListWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setIconSize(QSize(128, 72))
        self.setGridSize(QSize(144, 112))
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setMovement(QListWidget.Movement.Static)
        self.setDragEnabled(True)
        self.setWordWrap(True)
        self.setSpacing(4)

    def startDrag(self, _actions):
        it = self.currentItem()
        if it is None:
            return
        md = QMimeData()
        md.setData(ASSET_MIME, it.data(Qt.ItemDataRole.UserRole).encode())
        drag = QDrag(self)
        drag.setMimeData(md)
        drag.setPixmap(it.icon().pixmap(96, 54))
        drag.exec(Qt.DropAction.CopyAction)


# --------------------------------------------------------------------------- inspector widgets
class Section(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("section")
        self.setStyleSheet("#section { background: #2b2d32; border-radius: 8px; }")
        self.lay = QGridLayout(self)
        self.lay.setContentsMargins(10, 8, 10, 10)
        self.lay.setHorizontalSpacing(8)
        self.lay.setVerticalSpacing(6)
        t = QLabel(title)
        t.setStyleSheet("font-weight: 600; color: #e8e8ee;")
        self.lay.addWidget(t, 0, 0, 1, 4)
        self.row = 1

    def add(self, label: str, widget, extra=None):
        lbl = QLabel(label)
        lbl.setStyleSheet("color: #b4b8c0;")
        self.lay.addWidget(lbl, self.row, 0)
        if extra is None:
            self.lay.addWidget(widget, self.row, 1, 1, 3)
        else:
            self.lay.addWidget(widget, self.row, 1, 1, 2)
            self.lay.addWidget(extra, self.row, 3)
        self.row += 1


class SliderRow(QWidget):
    """Slider + value text + reset button bound to one clip attribute."""

    def __init__(self, insp: "Inspector", attr: str, lo: float, hi: float, default: float,
                 scale: float = 100, fmt=lambda v: f"{v:.0%}"):
        super().__init__()
        self.insp, self.attr, self.default, self.scale, self.fmt = insp, attr, default, scale, fmt
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(lo * scale), int(hi * scale))
        self.value = QLabel()
        self.value.setMinimumWidth(46)
        self.value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.reset = QToolButton()
        self.reset.setText("↺")
        self.reset.setToolTip("Reset")
        self.reset.setAutoRaise(True)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.value)
        lay.addWidget(self.reset)
        self.slider.sliderPressed.connect(insp.begin_live)
        self.slider.sliderReleased.connect(insp.end_live)
        self.slider.valueChanged.connect(self._changed)
        self.reset.clicked.connect(lambda: insp.commit(**{attr: default}))

    def _changed(self, v):
        val = v / self.scale
        self.value.setText(self.fmt(val))
        if not self.insp.loading:
            self.insp.apply(self.attr, val, live=self.slider.isSliderDown())

    def load(self, val: float):
        self.slider.setValue(int(round(val * self.scale)))
        self.value.setText(self.fmt(val))


class ColorButton(QPushButton):
    def __init__(self, insp: "Inspector", attr: str, allow_alpha=False):
        super().__init__()
        self.insp, self.attr, self.allow_alpha = insp, attr, allow_alpha
        self.setFixedHeight(24)
        self.clicked.connect(self._pick)

    def _pick(self):
        from ..textrender import parse_color

        _, c = self.insp.current()
        if c is None:
            return
        opts = QColorDialog.ColorDialogOption.ShowAlphaChannel if self.allow_alpha else QColorDialog.ColorDialogOption(0)
        col = QColorDialog.getColor(parse_color(getattr(c, self.attr) or "#000000aa"), self, "Pick a color", opts)
        if col.isValid():
            val = col.name() + (f"{col.alpha():02x}" if self.allow_alpha else "")
            self.insp.commit(**{self.attr: val})

    def load(self, val: str):
        from ..textrender import parse_color

        col = parse_color(val or "#00000000")
        self.setStyleSheet(f"background: {col.name()}; border: 1px solid #666; border-radius: 4px;")


SPEEDS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]


class Inspector(QScrollArea):
    def __init__(self, editor: Editor):
        super().__init__()
        self.ed = editor
        self.clip_id: str | None = None
        self.loading = True
        self._snap = None
        self.setWidgetResizable(True)
        self.setMinimumWidth(310)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        self.setWidget(body)
        lay = QVBoxLayout(body)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        self.title = QLabel()
        self.title.setWordWrap(True)
        self.title.setStyleSheet("font-size: 12pt; font-weight: 600;")
        self.subtitle = QLabel()
        self.subtitle.setStyleSheet("color: #9aa0aa;")
        lay.addWidget(self.title)
        lay.addWidget(self.subtitle)

        self.placeholder = QLabel(
            "Select a clip on the timeline to edit it.\n\n"
            "Quick keys\n"
            "  Space   play / pause\n"
            "  S         split at playhead\n"
            "  T         add text\n"
            "  Del       delete   ·   Shift+Del ripple\n"
            "  Ctrl+D  duplicate\n"
            "  Ctrl+Z  undo\n"
            "  F1       all shortcuts")
        self.placeholder.setStyleSheet("color: #8a8f99; font-family: Consolas, monospace;")
        lay.addWidget(self.placeholder)

        # text
        self.sec_text = Section("✏  Text")
        self.text_edit = QPlainTextEdit()
        self.text_edit.setFixedHeight(64)
        self._text_timer = QTimer(self)
        self._text_timer.setSingleShot(True)
        self._text_timer.setInterval(400)
        self._text_timer.timeout.connect(lambda: self.commit(text=self.text_edit.toPlainText()))
        self.text_edit.textChanged.connect(self._text_changed)
        self.sec_text.add("Text", self.text_edit)
        self.font_combo = QFontComboBox()
        self.font_combo.currentFontChanged.connect(lambda f: self._if_ready(font_family=f.family()))
        self.sec_text.add("Font", self.font_combo)
        self.size_spin = QSpinBox()
        self.size_spin.setRange(8, 400)
        self.size_spin.valueChanged.connect(lambda v: self._if_ready(font_size=v))
        self.bold = QCheckBox("Bold")
        self.bold.toggled.connect(lambda v: self._if_ready(bold=v))
        self.sec_text.add("Size", self.size_spin, self.bold)
        self.color_btn = ColorButton(self, "color")
        self.sec_text.add("Color", self.color_btn)
        self.bg_check = QCheckBox("Box")
        self.bg_check.toggled.connect(lambda v: self._if_ready(bg_color="#000000aa" if v else ""))
        self.bg_btn = ColorButton(self, "bg_color", allow_alpha=True)
        self.sec_text.add("Background", self.bg_btn, self.bg_check)
        style_row = QWidget()
        sl = QHBoxLayout(style_row)
        sl.setContentsMargins(0, 0, 0, 0)
        for name, props in (("Title", dict(font_size=110, bold=True, pos_y=0.0, bg_color="")),
                            ("Subtitle", dict(font_size=54, bold=False, pos_y=0.36, bg_color="#000000aa")),
                            ("Lower third", dict(font_size=60, bold=True, pos_y=0.30, pos_x=-0.25,
                                                 bg_color="#1060f0dd"))):
            b = QPushButton(name)
            b.clicked.connect(lambda _=False, p=props: self.commit(**p))
            sl.addWidget(b)
        self.sec_text.add("Style", style_row)
        lay.addWidget(self.sec_text)

        # transform
        self.sec_xform = Section("⤢  Position & size")
        self.s_scale = SliderRow(self, "scale", 0.1, 4.0, 1.0)
        self.s_x = SliderRow(self, "pos_x", -1.0, 1.0, 0.0, fmt=lambda v: f"{v * 100:+.0f}")
        self.s_y = SliderRow(self, "pos_y", -1.0, 1.0, 0.0, fmt=lambda v: f"{v * 100:+.0f}")
        self.s_op = SliderRow(self, "opacity", 0.0, 1.0, 1.0)
        self.sec_xform.add("Size", self.s_scale)
        self.sec_xform.add("Left/right", self.s_x)
        self.sec_xform.add("Up/down", self.s_y)
        self.sec_xform.add("Opacity", self.s_op)
        lay.addWidget(self.sec_xform)

        # transition into this clip
        self.sec_trans = Section("⇄  Transition from previous clip")
        self.trans_kind = QComboBox()
        self.trans_kind.addItem("None (hard cut)", "")
        for k, label in KINDS.items():
            self.trans_kind.addItem(label, k)
        self.trans_kind.currentIndexChanged.connect(self._trans_kind_changed)
        self.sec_trans.add("Style", self.trans_kind)
        dur_row = QWidget()
        dl = QHBoxLayout(dur_row)
        dl.setContentsMargins(0, 0, 0, 0)
        self.trans_dur = QSlider(Qt.Orientation.Horizontal)
        self.trans_dur.setRange(10, 300)  # 0.1 .. 3.0 s
        self.trans_dur_lbl = QLabel()
        self.trans_dur_lbl.setMinimumWidth(46)
        self.trans_dur_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        dl.addWidget(self.trans_dur, 1)
        dl.addWidget(self.trans_dur_lbl)
        self.trans_dur.valueChanged.connect(self._trans_dur_changed)
        self.trans_dur.sliderReleased.connect(self._trans_dur_commit)
        self.sec_trans.add("Length", dur_row)
        self.trans_all = QPushButton("Apply this to every cut")
        self.trans_all.setToolTip("Use the same transition between all clips on video tracks")
        self.trans_all.clicked.connect(self._trans_all)
        self.sec_trans.add("", self.trans_all)
        hint = QLabel("Tip: click the ＋ diamonds between clips\non the timeline to add transitions.")
        hint.setStyleSheet("color: #8a8f99; font-size: 8.5pt;")
        self.sec_trans.add("", hint)
        lay.addWidget(self.sec_trans)

        # speed + fades
        self.sec_time = Section("⏱  Speed & fades")
        self.speed = QComboBox()
        for s in SPEEDS:
            self.speed.addItem(f"{s:g}x" + ("  (normal)" if s == 1 else "  slow-mo" if s < 1 else "  fast"), s)
        self.speed.currentIndexChanged.connect(lambda i: self._if_ready(speed=SPEEDS[i]))
        self.sec_time.add("Speed", self.speed)
        self.s_fi = SliderRow(self, "fade_in", 0.0, 5.0, 0.0, fmt=lambda v: f"{v:.1f}s")
        self.s_fo = SliderRow(self, "fade_out", 0.0, 5.0, 0.0, fmt=lambda v: f"{v:.1f}s")
        self.sec_time.add("Fade in", self.s_fi)
        self.sec_time.add("Fade out", self.s_fo)
        lay.addWidget(self.sec_time)

        # color
        self.sec_color = Section("🎨  Color")
        self.s_bri = SliderRow(self, "brightness", -0.5, 0.5, 0.0, fmt=lambda v: f"{v * 100:+.0f}")
        self.s_con = SliderRow(self, "contrast", 0.0, 2.0, 1.0)
        self.s_sat = SliderRow(self, "saturation", 0.0, 3.0, 1.0)
        self.sec_color.add("Brightness", self.s_bri)
        self.sec_color.add("Contrast", self.s_con)
        self.sec_color.add("Saturation", self.s_sat)
        presets = QWidget()
        pl = QHBoxLayout(presets)
        pl.setContentsMargins(0, 0, 0, 0)
        for name, props in (("Vivid", dict(saturation=1.5, contrast=1.15, brightness=0.0)),
                            ("B&W", dict(saturation=0.0, contrast=1.2, brightness=0.0)),
                            ("Warm fade", dict(saturation=0.8, contrast=0.85, brightness=0.06)),
                            ("Reset", dict(saturation=1.0, contrast=1.0, brightness=0.0))):
            b = QPushButton(name)
            b.clicked.connect(lambda _=False, p=props: self.commit(**p))
            pl.addWidget(b)
        self.sec_color.add("Looks", presets)
        lay.addWidget(self.sec_color)

        # audio
        self.sec_audio = Section("🔊  Audio")
        self.s_vol = SliderRow(self, "volume", 0.0, 2.0, 1.0)
        self.sec_audio.add("Volume", self.s_vol)
        self.mute_btn = QPushButton("Mute clip")
        self.mute_btn.clicked.connect(lambda: self.commit(volume=0.0))
        self.sec_audio.add("", self.mute_btn)
        lay.addWidget(self.sec_audio)

        lay.addStretch(1)
        self.refresh()

    # ---- binding helpers -------------------------------------------------
    def current(self):
        if self.clip_id is None:
            return None, None
        return self.ed.project.find_clip(self.clip_id)

    def begin_live(self):
        self._snap = self.ed.snapshot()

    def end_live(self):
        if self._snap is not None:
            self.ed.checkpoint(self._snap)
            self._snap = None
            self.ed.notify("props")

    def apply(self, attr, value, live: bool):
        if live:
            _, c = self.current()
            if c is not None:
                setattr(c, attr, value)
                self.ed.notify("live")
        else:
            self.commit(**{attr: value})

    def commit(self, **props):
        if self.clip_id is not None:
            self.ed.set_clip_props(self.clip_id, **props)

    def _if_ready(self, **props):
        if not self.loading:
            self.commit(**props)

    def _text_changed(self):
        if not self.loading:
            self._text_timer.start()

    # ---- transitions -----------------------------------------------------
    def _trans_kind_changed(self, _i):
        if self.loading or self.clip_id is None:
            return
        kind = self.trans_kind.currentData()
        if not kind:
            self.ed.remove_transition(self.clip_id)
        else:
            _, c = self.current()
            dur = c.transition if c is not None and c.transition > 0 else self.trans_dur.value() / 100
            self.ed.set_transition(self.clip_id, kind, dur)

    def _trans_dur_changed(self, v):
        self.trans_dur_lbl.setText(f"{v / 100:.1f}s")
        if not self.loading and not self.trans_dur.isSliderDown():
            self._trans_dur_commit()

    def _trans_dur_commit(self):
        if self.clip_id is None or self.loading:
            return
        _, c = self.current()
        if c is not None and c.transition > 0:
            self.ed.set_transition(self.clip_id, None, self.trans_dur.value() / 100)

    def _trans_all(self):
        kind = self.trans_kind.currentData()
        if kind:
            self.ed.transitions_everywhere(kind, self.trans_dur.value() / 100)

    # ---- refresh ---------------------------------------------------------
    def refresh(self):
        self.loading = True
        tr = c = None
        if len(self.ed.selection) == 1:
            tr, c = self.ed.project.find_clip(next(iter(self.ed.selection)))
        self.clip_id = c.id if c else None
        sections = (self.sec_text, self.sec_xform, self.sec_time, self.sec_color, self.sec_audio, self.sec_trans)
        if c is None:
            for s in sections:
                s.hide()
            self.placeholder.show()
            n = len(self.ed.selection)
            self.title.setText("Properties" if not n else f"{n} clips selected")
            self.subtitle.setText("")
            self.loading = False
            return
        self.placeholder.hide()
        a = self.ed.project.assets.get(c.asset_id)
        fps = self.ed.project.fps
        self.title.setText("Text" if c.is_text else a.name)
        self.subtitle.setText(f"Starts {fmt_time(c.start, fps, True)}  ·  {c.duration:.2f}s long")
        visual = tr.kind == "video"
        self.sec_text.setVisible(c.is_text)
        self.sec_xform.setVisible(visual)
        self.sec_color.setVisible(visual and not c.is_text)
        self.sec_audio.setVisible(a is not None and a.has_audio)
        self.sec_time.show()
        prev = prev_clip(tr, c) if c.transition > 0 else neighbour_before(tr, c)
        self.sec_trans.setVisible(prev is not None)
        if prev is not None:
            has = c.transition > 0
            self.trans_kind.setCurrentIndex(self.trans_kind.findData(c.transition_kind) if has else 0)
            self.trans_dur.setMaximum(max(int(max_duration(prev, c) * 100), 10))
            self.trans_dur.setValue(int(round((c.transition if has else DEFAULT_DURATION) * 100)))
            self.trans_dur_lbl.setText(f"{self.trans_dur.value() / 100:.1f}s")
            self.trans_dur.setEnabled(has)
            self.trans_all.setEnabled(has)
        self.speed.setEnabled(not c.is_text and (a is None or a.kind != "image"))
        if c.is_text:
            if self.text_edit.toPlainText() != c.text and not self.text_edit.hasFocus():
                self.text_edit.setPlainText(c.text)
            self.font_combo.setCurrentFont(QFont(c.font_family))
            self.size_spin.setValue(c.font_size)
            self.bold.setChecked(c.bold)
            self.color_btn.load(c.color)
            self.bg_check.setChecked(bool(c.bg_color))
            self.bg_btn.load(c.bg_color)
            self.bg_btn.setEnabled(bool(c.bg_color))
        for row, val in ((self.s_scale, c.scale), (self.s_x, c.pos_x), (self.s_y, c.pos_y),
                         (self.s_op, c.opacity), (self.s_fi, c.fade_in), (self.s_fo, c.fade_out),
                         (self.s_bri, c.brightness), (self.s_con, c.contrast), (self.s_sat, c.saturation),
                         (self.s_vol, c.volume)):
            row.load(val)
        idx = min(range(len(SPEEDS)), key=lambda i: abs(SPEEDS[i] - c.speed))
        self.speed.setCurrentIndex(idx)
        self.loading = False
