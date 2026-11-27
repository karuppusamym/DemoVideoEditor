"""Right-hand inspector: tabs for Text/Shape, Video, Adjust, Mask & Key, Animation, Speed, Audio.

Every animatable slider has a ◆ button: click it to add/remove a keyframe at the playhead.
Once a property has keyframes, moving its slider edits (or creates) the keyframe under the playhead.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFontComboBox, QFrame, QGridLayout,
                               QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPlainTextEdit,
                               QPushButton, QScrollArea, QSlider, QSpinBox, QTabWidget, QToolButton,
                               QVBoxLayout, QWidget)

from .. import animate, audio_fx
from ..editor import Editor
from ..retime import MAX_RATE, MIN_RATE, curve
from ..transitions import DEFAULT_DURATION, KINDS, max_duration, neighbour_before, prev_clip
from .keyframes import KeyframeEditor
from .timeline import fmt_time

SPEEDS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 8.0]
BLENDS = ["normal", "multiply", "screen", "overlay", "darken", "lighten", "color_dodge", "color_burn",
          "hard_light", "soft_light", "difference", "exclusion", "add"]


class Section(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("section")
        self.setStyleSheet("#section { background: #2b2d32; border-radius: 8px; }")
        self.lay = QGridLayout(self)
        self.lay.setContentsMargins(10, 8, 10, 10)
        self.lay.setHorizontalSpacing(6)
        self.lay.setVerticalSpacing(5)
        t = QLabel(title)
        t.setStyleSheet("font-weight: 600; color: #e8e8ee;")
        self.lay.addWidget(t, 0, 0, 1, 3)
        self.row = 1

    def add(self, label: str, widget):
        lbl = QLabel(label)
        lbl.setStyleSheet("color: #b4b8c0;")
        self.lay.addWidget(lbl, self.row, 0)
        self.lay.addWidget(widget, self.row, 1, 1, 2)
        self.row += 1
        return widget

    def full(self, widget):
        self.lay.addWidget(widget, self.row, 0, 1, 3)
        self.row += 1
        return widget


class Row(QWidget):
    """slider + value + (◆ keyframe) + ↺ reset, bound to a getter/setter on the current clip."""

    def __init__(self, insp: "Inspector", key: str, lo: float, hi: float, default: float, scale: float = 100,
                 fmt=lambda v: f"{v:.0%}", keyable: bool = False, getter=None, setter=None):
        super().__init__()
        self.insp, self.key, self.default, self.scale, self.fmt, self.keyable = insp, key, default, scale, fmt, keyable
        self.getter = getter or (lambda c: animate.value_at(c, key, insp.t_local(c)) if keyable else getattr(c, key))
        self.setter = setter or self._set_prop
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(lo * scale), int(hi * scale))
        self.value = QLabel()
        self.value.setMinimumWidth(44)
        self.value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.value)
        self.kf = None
        if keyable:
            self.kf = QToolButton()
            self.kf.setText("◇")
            self.kf.setToolTip("Add / remove a keyframe at the playhead")
            self.kf.setAutoRaise(True)
            self.kf.clicked.connect(self._toggle_kf)
            lay.addWidget(self.kf)
        reset = QToolButton()
        reset.setText("↺")
        reset.setToolTip("Reset")
        reset.setAutoRaise(True)
        reset.clicked.connect(lambda: insp.mutate(lambda c: self.setter(c, default)))
        lay.addWidget(reset)
        self.slider.sliderPressed.connect(insp.begin_live)
        self.slider.sliderReleased.connect(insp.end_live)
        self.slider.valueChanged.connect(self._changed)

    def _set_prop(self, clip, v):
        if self.keyable and (clip.keyframes or {}).get(self.key):
            animate.set_keyframe(clip, self.key, self.insp.t_local(clip), v)
        else:
            setattr(clip, self.key, v)

    def _changed(self, raw):
        v = raw / self.scale
        self.value.setText(self.fmt(v))
        if self.insp.loading:
            return
        if self.slider.isSliderDown():
            c = self.insp.clip()
            if c is not None:
                self.setter(c, v)
                self.insp.ed.notify("live")
        else:
            self.insp.mutate(lambda c: self.setter(c, v))

    def _toggle_kf(self):
        c = self.insp.clip()
        if c is None:
            return
        tl = self.insp.t_local(c)

        def fn(clip):
            if animate.keyframe_at(clip, self.key, tl):
                animate.remove_keyframe(clip, self.key, tl)
            else:
                animate.set_keyframe(clip, self.key, tl, animate.value_at(clip, self.key, tl))
        self.insp.mutate(fn)

    def load(self, clip):
        v = self.getter(clip)
        self.slider.setValue(int(round(v * self.scale)))
        self.value.setText(self.fmt(v))
        if self.kf is not None:
            has = bool((clip.keyframes or {}).get(self.key))
            on = animate.keyframe_at(clip, self.key, self.insp.t_local(clip)) is not None
            self.kf.setText("◆" if on else "◇")
            self.kf.setStyleSheet("color: #ffcc40;" if on else "color: #ffcc40;" if has else "")


class ColorButton(QPushButton):
    def __init__(self, insp, get, set_, alpha=False):
        super().__init__()
        self.insp, self.get, self.set_, self.alpha = insp, get, set_, alpha
        self.setFixedHeight(24)
        self.clicked.connect(self._pick)

    def _pick(self):
        from ..textrender import parse_color

        c = self.insp.clip()
        if c is None:
            return
        opts = QColorDialog.ColorDialogOption.ShowAlphaChannel if self.alpha else QColorDialog.ColorDialogOption(0)
        col = QColorDialog.getColor(parse_color(self.get(c) or "#000000aa"), self, "Pick a colour", opts)
        if col.isValid():
            val = col.name() + (f"{col.alpha():02x}" if self.alpha else "")
            self.insp.mutate(lambda cl: self.set_(cl, val))

    def load(self, val):
        from ..textrender import parse_color

        self.setStyleSheet(f"background: {parse_color(val or '#00000000').name()}; border: 1px solid #666;"
                           " border-radius: 4px;")


def _page() -> tuple[QScrollArea, QVBoxLayout]:
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    body = QWidget()
    lay = QVBoxLayout(body)
    lay.setContentsMargins(6, 6, 6, 6)
    lay.setSpacing(8)
    sa.setWidget(body)
    return sa, lay


class Inspector(QWidget):
    def __init__(self, editor: Editor, win=None):
        super().__init__()
        self.ed, self.win = editor, win
        self.clip_id: str | None = None
        self.loading = True
        self._snap = None
        self.rows: list[Row] = []
        self.setMinimumWidth(320)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        self.title = QLabel()
        self.title.setStyleSheet("font-size: 12pt; font-weight: 700;")
        self.title.setWordWrap(True)
        self.subtitle = QLabel()
        self.subtitle.setStyleSheet("color: #9aa0aa;")
        outer.addWidget(self.title)
        outer.addWidget(self.subtitle)
        self.placeholder = QLabel(
            "Select a clip on the timeline to edit it.\n\n"
            "Tips\n"
            "•  Click ◇ next to a slider to animate it\n"
            "•  Effects, filters, stickers and AI tools\n    are in the left panel\n"
            "•  Ctrl+K finds any command\n"
            "•  Ask the AI assistant to edit for you")
        self.placeholder.setStyleSheet("color: #8a8f99; padding: 8px;")
        outer.addWidget(self.placeholder)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        outer.addWidget(self.tabs, 1)
        self._build_text()
        self._build_shape()
        self._build_video()
        self._build_adjust()
        self._build_mask()
        self._build_anim()
        self.tab_keys, key_layout = _page()
        self.keyframe_editor = KeyframeEditor(editor, win.seek if win is not None else self.sync_curve_playhead)
        key_layout.addWidget(self.keyframe_editor)
        self._build_speed()
        self._build_audio()
        self.refresh()

    # ---- binding helpers -------------------------------------------------
    def clip(self):
        if self.clip_id is None:
            return None
        return self.ed.project.find_clip(self.clip_id)[1]

    def t_local(self, clip) -> float:
        ph = self.win.playhead if self.win is not None else clip.start
        return min(max(ph - clip.start, 0.0), clip.duration)

    def begin_live(self):
        self._snap = self.ed.snapshot()

    def end_live(self):
        if self._snap is not None:
            self.ed.checkpoint(self._snap)
            self._snap = None
            self.ed.notify("props")

    def mutate(self, fn):
        c = self.clip()
        if c is None or self.loading:
            return
        self.ed.checkpoint()
        fn(c)
        self.ed.notify("props")

    def row(self, *a, **k) -> Row:
        r = Row(self, *a, **k)
        self.rows.append(r)
        return r

    # ---- tabs ------------------------------------------------------------
    def _build_text(self):
        page, lay = _page()
        sec = Section("✏  Text")
        self.text_edit = QPlainTextEdit()
        self.text_edit.setFixedHeight(70)
        self._text_timer = QTimer(self)
        self._text_timer.setSingleShot(True)
        self._text_timer.setInterval(350)
        self._text_timer.timeout.connect(lambda: self.mutate(lambda c: setattr(c, "text", self.text_edit.toPlainText())))
        self.text_edit.textChanged.connect(lambda: None if self.loading else self._text_timer.start())
        sec.full(self.text_edit)
        self.font_combo = QFontComboBox()
        self.font_combo.currentFontChanged.connect(lambda f: self.mutate(lambda c: setattr(c, "font_family", f.family())))
        sec.add("Font", self.font_combo)
        self.size_spin = QSpinBox()
        self.size_spin.setRange(8, 400)
        self.size_spin.valueChanged.connect(lambda v: self.mutate(lambda c: setattr(c, "font_size", v)))
        sec.add("Size", self.size_spin)
        self.bold = QCheckBox("Bold")
        self.bold.toggled.connect(lambda v: self.mutate(lambda c: setattr(c, "bold", v)))
        sec.add("", self.bold)
        self.color_btn = ColorButton(self, lambda c: c.color, lambda c, v: setattr(c, "color", v))
        sec.add("Colour", self.color_btn)
        self.bg_check = QCheckBox("Background box")
        self.bg_check.toggled.connect(lambda v: self.mutate(lambda c: setattr(c, "bg_color", "#000000aa" if v else "")))
        sec.add("", self.bg_check)
        self.bg_btn = ColorButton(self, lambda c: c.bg_color, lambda c, v: setattr(c, "bg_color", v), alpha=True)
        sec.add("Box colour", self.bg_btn)
        lay.addWidget(sec)
        lay.addStretch(1)
        self.tab_text = page

    def _build_shape(self):
        page, lay = _page()
        sec = Section("⭐  Shape")

        def sget(k, d=None):
            return lambda c: (c.shape or {}).get(k, d)

        def sset(k):
            def f(c, v):
                c.shape = {**(c.shape or {}), k: v}
            return f
        self.fill_btn = ColorButton(self, sget("fill", "#ffffff"), sset("fill"), alpha=True)
        sec.add("Fill", self.fill_btn)
        self.fill2_btn = ColorButton(self, sget("fill2", ""), sset("fill2"), alpha=True)
        sec.add("Gradient to", self.fill2_btn)
        self.stroke_btn = ColorButton(self, sget("stroke", "#000000"), sset("stroke"), alpha=True)
        sec.add("Outline", self.stroke_btn)
        self.s_sw = self.row("stroke_w", 0, 0.05, 0, scale=1000, fmt=lambda v: f"{v * 1000:.0f}",
                             getter=lambda c: float((c.shape or {}).get("stroke_w", 0) or 0), setter=sset("stroke_w"))
        sec.add("Outline width", self.s_sw)
        self.s_w = self.row("w", 0.02, 2, 0.5, getter=lambda c: float((c.shape or {}).get("w", 0.5)), setter=sset("w"))
        self.s_h = self.row("h", 0.02, 2, 0.3, getter=lambda c: float((c.shape or {}).get("h", 0.3)), setter=sset("h"))
        sec.add("Width", self.s_w)
        sec.add("Height", self.s_h)
        self.shadow = QCheckBox("Drop shadow")
        self.shadow.toggled.connect(lambda v: self.mutate(lambda c: sset("shadow")(c, v)))
        sec.add("", self.shadow)
        lay.addWidget(sec)
        lay.addStretch(1)
        self.tab_shape = page

    def _build_video(self):
        page, lay = _page()
        sec = Section("⤢  Position, size & rotation")
        self.r_scale = sec.add("Size", self.row("scale", 0.05, 5, 1.0, keyable=True))
        pct = lambda v: f"{v * 100:+.0f}"  # noqa: E731
        sec.add("Left/right", self.row("pos_x", -1.5, 1.5, 0.0, fmt=pct, keyable=True))
        sec.add("Up/down", self.row("pos_y", -1.5, 1.5, 0.0, fmt=pct, keyable=True))
        sec.add("Rotate", self.row("rotation", -360, 360, 0.0, scale=1, fmt=lambda v: f"{v:.0f}°", keyable=True))
        sec.add("Opacity", self.row("opacity", 0, 1, 1.0, keyable=True))
        flips = QWidget()
        fl = QHBoxLayout(flips)
        fl.setContentsMargins(0, 0, 0, 0)
        for text, fn in (("⇋ Mirror", lambda c: setattr(c, "flip_h", not c.flip_h)),
                         ("⇵ Flip", lambda c: setattr(c, "flip_v", not c.flip_v)),
                         ("⟳ 90°", lambda c: setattr(c, "rotation", (c.rotation + 90) % 360)),
                         ("Fit", lambda c: (setattr(c, "scale", 1.0), setattr(c, "pos_x", 0.0), setattr(c, "pos_y", 0.0)))):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, f=fn: self.mutate(f))
            fl.addWidget(b)
        sec.full(flips)
        lay.addWidget(sec)
        crop = Section("✂  Crop")
        for i, name in enumerate(("Left", "Top", "Right", "Bottom")):
            def g(c, i=i):
                return float((list(c.crop) + [0, 0, 0, 0])[i])

            def s(c, v, i=i):
                cr = (list(c.crop) + [0, 0, 0, 0])[:4]
                cr[i] = v
                c.crop = cr
            crop.add(name, self.row(f"crop{i}", 0, 0.9, 0.0, getter=g, setter=s))
        lay.addWidget(crop)
        blend = Section("◑  Blend")
        self.blend = QComboBox()
        for b in BLENDS:
            self.blend.addItem(b.replace("_", " ").title(), b)
        self.blend.currentIndexChanged.connect(lambda _i: self.mutate(lambda c: setattr(c, "blend_mode", self.blend.currentData())))
        blend.add("Mode", self.blend)
        lay.addWidget(blend)
        lay.addStretch(1)
        self.tab_video = page

    def _build_adjust(self):
        page, lay = _page()
        sec = Section("🎨  Colour")
        sec.add("Brightness", self.row("brightness", -0.5, 0.5, 0.0, fmt=lambda v: f"{v * 100:+.0f}"))
        sec.add("Contrast", self.row("contrast", 0, 2, 1.0))
        sec.add("Saturation", self.row("saturation", 0, 3, 1.0))
        lay.addWidget(sec)
        fx = Section("✨  Effects & filter on this clip")
        self.fx_list = QListWidget()
        self.fx_list.setFixedHeight(110)
        self.fx_list.itemChanged.connect(self._fx_toggled)
        self.fx_list.currentRowChanged.connect(self._fx_selected)
        fx.full(self.fx_list)
        btns = QWidget()
        bl = QHBoxLayout(btns)
        bl.setContentsMargins(0, 0, 0, 0)
        for text, fn in (("▲", lambda: self._fx_move(-1)), ("▼", lambda: self._fx_move(1)),
                         ("✕ Remove", self._fx_remove), ("＋ Add…", lambda: self.win and self.win.sidebar.show_panel("effects"))):
            b = QPushButton(text)
            b.clicked.connect(fn)
            bl.addWidget(b)
        fx.full(btns)
        self.fx_params = QWidget()
        self.fx_params_lay = QGridLayout(self.fx_params)
        self.fx_params_lay.setContentsMargins(0, 0, 0, 0)
        fx.full(self.fx_params)
        lay.addWidget(fx)
        lay.addStretch(1)
        self.tab_adjust = page

    def _build_mask(self):
        page, lay = _page()
        key = Section("🟩  Chroma key (green screen)")
        self.key_on = QCheckBox("Remove a colour")
        self.key_on.toggled.connect(lambda v: self.mutate(
            lambda c: setattr(c, "chroma", {"color": "#00ff00", "similarity": 0.3, "smooth": 0.1, "spill": 0.5} if v else {})))
        key.full(self.key_on)
        self.key_color = ColorButton(self, lambda c: (c.chroma or {}).get("color", "#00ff00"),
                                     lambda c, v: setattr(c, "chroma", {**c.chroma, "color": v}))
        key.add("Colour", self.key_color)

        def cg(k, d):
            return lambda c: float((c.chroma or {}).get(k, d))

        def cs(k):
            return lambda c, v: setattr(c, "chroma", {**(c.chroma or {}), k: v}) if c.chroma else None
        key.add("Similarity", self.row("ck_sim", 0, 1, 0.3, getter=cg("similarity", 0.3), setter=cs("similarity")))
        key.add("Smoothness", self.row("ck_smooth", 0, 1, 0.1, getter=cg("smooth", 0.1), setter=cs("smooth")))
        key.add("Spill", self.row("ck_spill", 0, 1, 0.5, getter=cg("spill", 0.5), setter=cs("spill")))
        lay.addWidget(key)
        m = Section("◐  Masks")
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        for shape, label in (("rect", "▭ Rect"), ("ellipse", "◯ Circle"), ("linear", "◧ Split"), ("mirror", "▥ Band")):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, s=shape: self._add_mask(s))
            rl.addWidget(b)
        m.full(row)
        self.mask_info = QLabel()
        self.mask_info.setStyleSheet("color: #9aa0aa;")
        m.full(self.mask_info)

        def mg(k, d):
            return lambda c: float((c.masks[-1] if c.masks else {}).get(k, d))

        def ms(k):
            def f(c, v):
                if c.masks:
                    c.masks[-1] = {**c.masks[-1], k: v}
            return f
        for k, lab, lo, hi, d in (("x", "Centre X", 0, 1, .5), ("y", "Centre Y", 0, 1, .5), ("w", "Width", 0, 1, .5),
                                  ("h", "Height", 0, 1, .5), ("feather", "Feather", 0, 1, .1)):
            m.add(lab, self.row(f"mask_{k}", lo, hi, d, getter=mg(k, d), setter=ms(k)))
        m.add("Angle", self.row("mask_angle", -180, 180, 0, scale=1, fmt=lambda v: f"{v:.0f}°",
                                getter=mg("angle", 0), setter=ms("angle")))
        inv = QWidget()
        il = QHBoxLayout(inv)
        il.setContentsMargins(0, 0, 0, 0)
        b1 = QPushButton("Invert")
        b1.clicked.connect(lambda: self.mutate(lambda c: c.masks and c.masks.__setitem__(-1, {**c.masks[-1], "invert": not c.masks[-1].get("invert")})))
        b2 = QPushButton("✕ Remove mask")
        b2.clicked.connect(lambda: self.mutate(lambda c: c.masks and c.masks.pop()))
        il.addWidget(b1)
        il.addWidget(b2)
        m.full(inv)
        lay.addWidget(m)
        lay.addStretch(1)
        self.tab_mask = page

    def _build_anim(self):
        page, lay = _page()
        sec = Section("🌀  Animation")
        self.anim_combos = {}
        for kind, label in (("in", "In"), ("out", "Out"), ("loop", "Loop")):
            cb = QComboBox()
            cb.addItem("None", "")
            for name, lab in animate.ANIMATIONS.get(kind, {}).items():
                cb.addItem(lab, name)
            cb.currentIndexChanged.connect(lambda _i, k=kind, b=cb: self.mutate(lambda c: setattr(c, f"anim_{k}", b.currentData())))
            self.anim_combos[kind] = cb
            sec.add(label, cb)
        sec.add("In length", self.row("anim_in_dur", 0.1, 3, 0.5, fmt=lambda v: f"{v:.1f}s"))
        sec.add("Out length", self.row("anim_out_dur", 0.1, 3, 0.5, fmt=lambda v: f"{v:.1f}s"))
        sec.add("Loop speed", self.row("anim_loop_speed", 0.1, 4, 1.0, fmt=lambda v: f"{v:.1f}x"))
        lay.addWidget(sec)
        kf = Section("◆  Keyframes")
        self.kf_info = QLabel()
        self.kf_info.setWordWrap(True)
        self.kf_info.setStyleSheet("color: #9aa0aa;")
        kf.full(self.kf_info)
        self.easing = QComboBox()
        for e in animate.EASINGS if hasattr(animate, "EASINGS") else ("linear", "ease"):
            self.easing.addItem(e.replace("_", " "), e)
        self.easing.setToolTip("Easing of keyframes at the playhead")
        self.easing.currentIndexChanged.connect(self._set_easing)
        kf.add("Easing", self.easing)
        clr = QPushButton("Clear all keyframes")
        clr.clicked.connect(lambda: self.mutate(lambda c: animate.clear_keyframes(c)))
        kf.full(clr)
        lay.addWidget(kf)
        lay.addStretch(1)
        self.tab_anim = page

    def _build_speed(self):
        page, lay = _page()
        sec = Section("⏱  Speed")
        self.speed = QComboBox()
        for s in SPEEDS:
            self.speed.addItem(f"{s:g}x" + ("  normal" if s == 1 else "  slow-mo" if s < 1 else "  fast"), s)
        self.speed.currentIndexChanged.connect(
            lambda i: None if self.loading else self.ed.set_clip_props(self.clip_id, speed=SPEEDS[i]))
        sec.add("Speed", self.speed)
        sec.add("Fade in", self.row("fade_in", 0, 5, 0, fmt=lambda v: f"{v:.1f}s"))
        sec.add("Fade out", self.row("fade_out", 0, 5, 0, fmt=lambda v: f"{v:.1f}s"))
        lay.addWidget(sec)
        self.ramp_section = Section("Speed ramp")
        self.ramp_rates = []
        for label in ("Start", "Middle", "End"):
            rate = QDoubleSpinBox()
            rate.setRange(MIN_RATE, MAX_RATE)
            rate.setSingleStep(0.25)
            rate.setSuffix("x")
            self.ramp_section.add(label, rate)
            self.ramp_rates.append(rate)
        self.ramp_status = QLabel()
        self.ramp_section.full(self.ramp_status)
        apply = QPushButton("Apply speed ramp")
        apply.clicked.connect(self._apply_ramp)
        self.ramp_section.full(apply)
        self.clear_ramp = QPushButton("Use constant speed")
        self.clear_ramp.clicked.connect(lambda: self.ed.set_clip_props(self.clip_id, speed_ramp=()))
        self.ramp_section.full(self.clear_ramp)
        lay.addWidget(self.ramp_section)
        tr = Section("⇄  Transition from previous clip")
        self.trans_kind = QComboBox()
        self.trans_kind.addItem("None (hard cut)", "")
        for k, label in KINDS.items():
            self.trans_kind.addItem(label, k)
        self.trans_kind.currentIndexChanged.connect(self._trans_kind_changed)
        tr.add("Style", self.trans_kind)
        self.trans_dur = QSlider(Qt.Orientation.Horizontal)
        self.trans_dur.setRange(10, 300)
        self.trans_dur.sliderReleased.connect(self._trans_dur_commit)
        tr.add("Length", self.trans_dur)
        self.sec_trans = tr
        lay.addWidget(tr)
        lay.addStretch(1)
        self.tab_speed = page

    def _apply_ramp(self):
        _, clip = self.ed.project.find_clip(self.clip_id)
        if clip is not None and clip.out_point > clip.in_point:
            middle = (clip.in_point + clip.out_point) / 2
            points = zip((clip.in_point, middle, clip.out_point), (rate.value() for rate in self.ramp_rates))
            self.ed.set_clip_props(clip.id, speed_ramp=points)

    def _build_audio(self):
        page, lay = _page()
        sec = Section("🔊  Volume")
        sec.add("Volume", self.row("volume", 0, 2, 1.0, keyable=True))
        b = QPushButton("Mute clip")
        b.clicked.connect(lambda: self.mutate(lambda c: setattr(c, "volume", 0.0)))
        sec.full(b)
        lay.addWidget(sec)
        fx = Section("🎛  Voice & sound effects")
        self.audio_effects_section = fx
        self.afx = QListWidget()
        self.afx.setFixedHeight(230)
        self.afx.itemChanged.connect(self._afx_toggled)
        fx.full(self.afx)
        self.afx_status = QLabel()
        self.afx_status.setStyleSheet("color: #9aa0aa;")
        fx.full(self.afx_status)
        try:
            from .. import audio_fx

            effs = getattr(audio_fx, "EFFECTS", {k: {"label": v, "category": ""} for k, v in audio_fx.PRESETS.items()})
            for k, spec in effs.items():
                it = QListWidgetItem(f"{spec.get('label', k)}   ·  {spec.get('category', '')}")
                it.setData(Qt.ItemDataRole.UserRole, k)
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(Qt.CheckState.Unchecked)
                self.afx.addItem(it)
        except Exception:  # noqa: BLE001
            pass
        lay.addWidget(fx)
        lay.addStretch(1)
        self.tab_audio = page

    # ---- effects stack helpers -------------------------------------------
    def _fx_toggled(self, it):
        if self.loading:
            return
        i = self.fx_list.row(it)
        on = it.checkState() == Qt.CheckState.Checked
        self.mutate(lambda c: c.effects.__setitem__(i, {**c.effects[i], "enabled": on}))

    def _fx_move(self, d):
        i = self.fx_list.currentRow()

        def fn(c):
            j = i + d
            if 0 <= i < len(c.effects) and 0 <= j < len(c.effects):
                c.effects[i], c.effects[j] = c.effects[j], c.effects[i]
        self.mutate(fn)

    def _fx_remove(self):
        i = self.fx_list.currentRow()
        self.mutate(lambda c: 0 <= i < len(c.effects) and c.effects.pop(i))

    def _fx_selected(self, i):
        while self.fx_params_lay.count():
            w = self.fx_params_lay.takeAt(0).widget()
            if w:
                w.deleteLater()
        c = self.clip()
        if c is None or not (0 <= i < len(c.effects)):
            return
        from .. import effects

        spec = effects.REGISTRY.get(c.effects[i]["type"], {})
        for r, (pname, p) in enumerate(spec.get("params", {}).items()):
            lo, hi = float(p.get("min", 0)), float(p.get("max", 1))
            scale = 100 if hi - lo <= 10 else 1

            def g(cl, pname=pname, p=p, i=i):
                return float(cl.effects[i].get("params", {}).get(pname, p.get("default", 0)))

            def s(cl, v, pname=pname, i=i):
                cl.effects[i] = {**cl.effects[i], "params": {**cl.effects[i].get("params", {}), pname: v}}
            row = Row(self, f"fx{i}_{pname}", lo, hi, float(p.get("default", 0)), scale=scale,
                      fmt=lambda v: f"{v:.2f}", getter=g, setter=s)
            row.load(c)
            self.fx_params_lay.addWidget(QLabel(p.get("label", pname)), r, 0)
            self.fx_params_lay.addWidget(row, r, 1)

    def _add_mask(self, shape):
        from .. import masks

        m = masks.new_mask(shape) if hasattr(masks, "new_mask") else {"shape": shape, "x": .5, "y": .5, "w": .5,
                                                                       "h": .5, "angle": 0, "feather": .1,
                                                                       "invert": False}
        self.mutate(lambda c: c.masks.append(m))

    def _afx_toggled(self, it):
        if self.loading:
            return
        k = it.data(Qt.ItemDataRole.UserRole)
        on = it.checkState() == Qt.CheckState.Checked

        def fn(c):
            c.audio_fx = [f for f in c.audio_fx if f.get("type") != k]
            if on:
                from .. import audio_fx

                params = audio_fx.default_params(k) if hasattr(audio_fx, "default_params") else {}
                c.audio_fx.append({"type": k, "params": params})
        self.mutate(fn)

    def _set_easing(self, _i):
        if self.loading:
            return
        e = self.easing.currentData()

        def fn(c):
            tl = self.t_local(c)
            for prop in list((c.keyframes or {}).keys()):
                k = animate.keyframe_at(c, prop, tl)
                if k is not None:
                    k[2] = e
        self.mutate(fn)

    # ---- transitions -----------------------------------------------------
    def _trans_kind_changed(self, _i):
        if self.loading or self.clip_id is None:
            return
        kind = self.trans_kind.currentData()
        if not kind:
            self.ed.remove_transition(self.clip_id)
        else:
            c = self.clip()
            self.ed.set_transition(self.clip_id, kind, c.transition if c and c.transition > 0
                                   else self.trans_dur.value() / 100)

    def _trans_dur_commit(self):
        c = self.clip()
        if c is not None and c.transition > 0:
            self.ed.set_transition(self.clip_id, None, self.trans_dur.value() / 100)

    # ---- refresh ---------------------------------------------------------
    def sync_time(self):
        """Playhead moved: update keyframe-driven values."""
        if self.win is not None:
            self.sync_curve_playhead(self.win.playhead)
        c = self.clip()
        if c is None or not c.keyframes:
            return
        self.loading = True
        for r in self.rows:
            if r.keyable:
                r.load(c)
        self._kf_text(c)
        self.loading = False

    def sync_curve_playhead(self, time):
        self.keyframe_editor.set_playhead(time)

    def _kf_text(self, c):
        props = ", ".join(sorted((c.keyframes or {}).keys())) or "none yet"
        n = len(animate.all_keyframe_times(c)) if hasattr(animate, "all_keyframe_times") else 0
        self.kf_info.setText(f"{n} keyframe time(s) · animated: {props}.\n"
                             "Click ◇ beside any slider on the Video/Audio tabs to add one at the playhead.")

    def refresh(self):
        self.loading = True
        tr = c = None
        if len(self.ed.selection) == 1:
            tr, c = self.ed.project.find_clip(next(iter(self.ed.selection)))
        self.clip_id = c.id if c else None
        if c is None:
            self.keyframe_editor.set_clip(None, ())
            self.tabs.hide()
            self.placeholder.show()
            n = len(self.ed.selection)
            self.title.setText("Properties" if not n else f"{n} clips selected")
            self.subtitle.setText("Effects, filters & animations apply to all selected clips" if n else "")
            self.loading = False
            return
        self.placeholder.hide()
        self.tabs.show()
        a = self.ed.project.asset_for_clip(c)
        fps = self.ed.project.fps
        source = self.ed.project.timelines.get(c.sequence_id)
        if source is not None:
            self.title.setText(source.name)
        else:
            self.title.setText("Text" if c.is_text else "Sticker / shape" if c.is_shape else a.name if a else "Clip")
        self.subtitle.setText(f"Starts {fmt_time(c.start, fps, True)}  ·  {c.duration:.2f}s long")
        visual = tr.kind == "video"
        has_audio = a is not None and a.has_audio
        want = []
        if c.is_text:
            want.append((self.tab_text, "Text"))
        if c.is_shape:
            want.append((self.tab_shape, "Shape"))
        if visual:
            want += [(self.tab_video, "Video"), (self.tab_adjust, "Adjust"), (self.tab_mask, "Mask & Key"),
                     (self.tab_anim, "Animation")]
        if visual or has_audio:
            want.append((self.tab_keys, "Keyframes"))
        want.append((self.tab_speed, "Speed"))
        if has_audio:
            want.append((self.tab_audio, "Audio"))
        key_properties = tuple(prop for prop in animate.KEYFRAME_PROPS
                               if (has_audio if prop == "volume" else visual))
        self.keyframe_editor.set_clip(c, key_properties)
        current = self.tabs.currentWidget()
        if [self.tabs.widget(i) for i in range(self.tabs.count())] != [w for w, _ in want]:
            self.tabs.clear()
            for w, label in want:
                self.tabs.addTab(w, label)
            if current is not None and self.tabs.indexOf(current) >= 0:
                self.tabs.setCurrentWidget(current)
        for r in self.rows:
            try:
                r.load(c)
            except Exception:  # noqa: BLE001
                pass
        if c.is_text:
            if self.text_edit.toPlainText() != c.text and not self.text_edit.hasFocus():
                self.text_edit.setPlainText(c.text)
            self.font_combo.setCurrentFont(QFont(c.font_family))
            self.size_spin.setValue(c.font_size)
            self.bold.setChecked(c.bold)
            self.color_btn.load(c.color)
            self.bg_check.setChecked(bool(c.bg_color))
            self.bg_btn.load(c.bg_color)
        if c.is_shape:
            s = c.shape or {}
            self.fill_btn.load(s.get("fill", "#ffffff"))
            self.fill2_btn.load(s.get("fill2", ""))
            self.stroke_btn.load(s.get("stroke", ""))
            self.shadow.setChecked(bool(s.get("shadow")))
        self.blend.setCurrentIndex(max(self.blend.findData(c.blend_mode), 0))
        # effects stack
        from .. import effects

        cur = self.fx_list.currentRow()
        self.fx_list.clear()
        for e in c.effects:
            spec = effects.REGISTRY.get(e.get("type"), {})
            it = QListWidgetItem(f"{spec.get('label', e.get('type'))}   ·  {spec.get('category', '')}")
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if e.get("enabled", True) else Qt.CheckState.Unchecked)
            self.fx_list.addItem(it)
        if c.effects:
            self.fx_list.setCurrentRow(min(max(cur, 0), len(c.effects) - 1))
            self._fx_selected(self.fx_list.currentRow())
        else:
            self._fx_selected(-1)
        # mask & key
        self.key_on.setChecked(bool(c.chroma))
        self.key_color.load((c.chroma or {}).get("color", "#00ff00"))
        self.mask_info.setText(f"{len(c.masks)} mask(s). Sliders edit the newest one." if c.masks
                               else "No masks. Add one to show only part of the clip.")
        # animation
        for kind, cb in self.anim_combos.items():
            cb.setCurrentIndex(max(cb.findData(getattr(c, f"anim_{kind}")), 0))
        self._kf_text(c)
        # speed + transition
        idx = min(range(len(SPEEDS)), key=lambda i: abs(SPEEDS[i] - c.speed))
        self.speed.setCurrentIndex(idx)
        can_retime = not c.is_generated and a is not None and a.kind != "image"
        self.speed.setEnabled(can_retime and not c.speed_ramp)
        self.ramp_section.setEnabled(can_retime)
        self.clear_ramp.setEnabled(bool(c.speed_ramp))
        self.ramp_status.setText("Ramp active · audio pitch follows speed" if c.speed_ramp else "")
        for widget, fraction in zip(self.ramp_rates, (0.0, 0.5, 1.0)):
            source = c.in_point + (c.out_point - c.in_point) * fraction
            widget.setValue(curve(c.speed_ramp).rate_at(source) if c.speed_ramp else c.speed)
        prev = prev_clip(tr, c) if c.transition > 0 else neighbour_before(tr, c)
        self.sec_trans.setVisible(prev is not None)
        if prev is not None:
            self.trans_kind.setCurrentIndex(self.trans_kind.findData(c.transition_kind) if c.transition > 0 else 0)
            self.trans_dur.setMaximum(max(int(max_duration(prev, c) * 100), 10))
            self.trans_dur.setValue(int(round((c.transition or DEFAULT_DURATION) * 100)))
        # audio fx
        self.audio_effects_section.setVisible(not c.sequence_id)
        on = {f.get("type") for f in c.audio_fx}
        for i in range(self.afx.count()):
            it = self.afx.item(i)
            it.setCheckState(Qt.CheckState.Checked if it.data(Qt.ItemDataRole.UserRole) in on else Qt.CheckState.Unchecked)
        if audio_fx.active(c):
            try:
                st = audio_fx.status(c, a)
                self.afx_status.setText({"processing": "⏳ Processing audio…", "ready": "✓ Ready"}.get(st, st))
            except Exception:  # noqa: BLE001
                self.afx_status.setText("")
        else:
            self.afx_status.setText("Tick effects to apply. Export uses exactly what you hear.")
        self.loading = False
