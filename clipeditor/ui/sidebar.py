"""Left side: icon rail + asset panels (Media, Audio, Text, Stickers, Effects, Filters,
Transitions, Animations, AI Magic). Every card shows a real rendered preview."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QLinearGradient, QPainter, QPixmap
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                               QPushButton, QScrollArea, QSlider, QStackedWidget, QTabWidget, QToolButton,
                               QVBoxLayout, QWidget)

CARD = QSize(112, 63)

RAIL = [
    ("media", "🎬", "Media"),
    ("audio", "🎵", "Audio"),
    ("text", "🅣", "Text"),
    ("stickers", "⭐", "Stickers"),
    ("effects", "✨", "Effects"),
    ("filters", "🎨", "Filters"),
    ("transitions", "⇄", "Transitions"),
    ("animations", "🌀", "Animation"),
    ("ai", "🤖", "AI Magic"),
]


def _title(text: str, sub: str = "") -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(4, 2, 4, 6)
    t = QLabel(text)
    t.setStyleSheet("font-size: 13pt; font-weight: 700;")
    lay.addWidget(t)
    if sub:
        s = QLabel(sub)
        s.setWordWrap(True)
        s.setStyleSheet("color: #8d929c; font-size: 8.5pt;")
        lay.addWidget(s)
    return w


class Gallery(QListWidget):
    """Grid of preview cards. Emits `picked(payload)` on click."""
    picked = Signal(object)

    def __init__(self, card: QSize = CARD):
        super().__init__()
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setIconSize(card)
        self.setGridSize(QSize(card.width() + 14, card.height() + 34))
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setMovement(QListWidget.Movement.Static)
        self.setWordWrap(True)
        self.setSpacing(2)
        self.setUniformItemSizes(True)
        self.setMouseTracking(True)
        self.itemClicked.connect(lambda it: self.picked.emit(it.data(Qt.ItemDataRole.UserRole)))

    def add(self, label: str, pixmap: QPixmap, payload, tip: str = ""):
        it = QListWidgetItem(QIcon(pixmap), label)
        it.setData(Qt.ItemDataRole.UserRole, payload)
        if tip:
            it.setToolTip(tip)
        self.addItem(it)


def card_pixmap(img: QImage | None, bg: QColor = QColor(30, 31, 36), size: QSize = CARD) -> QPixmap:
    pm = QPixmap(size)
    pm.fill(bg)
    if img is not None and not img.isNull():
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        sc = img.scaled(size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        p.drawImage((size.width() - sc.width()) // 2, (size.height() - sc.height()) // 2, sc)
        p.end()
    return pm


def icon_card(symbol: str, c1: str, c2: str, size: QSize = CARD) -> QPixmap:
    pm = QPixmap(size)
    p = QPainter(pm)
    g = QLinearGradient(0, 0, size.width(), size.height())
    g.setColorAt(0, QColor(c1))
    g.setColorAt(1, QColor(c2))
    p.fillRect(pm.rect(), g)
    f = p.font()
    f.setPointSize(20)
    p.setFont(f)
    p.setPen(QColor(255, 255, 255))
    p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, symbol)
    p.end()
    return pm


def sample_rgba(win, w: int = 160, h: int = 90) -> np.ndarray:
    """A frame to preview effects on: the current preview frame, or a colourful test card."""
    from ..render import qimage_to_rgba

    img = getattr(win.preview, "image", None)
    if img is not None and not img.isNull() and win.ed.project.duration > 0:
        return qimage_to_rgba(img.scaled(w, h, Qt.AspectRatioMode.IgnoreAspectRatio,
                                         Qt.TransformationMode.SmoothTransformation))
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    out = np.zeros((h, w, 4), np.uint8)
    out[..., 0] = (200 * x / w + 40).astype(np.uint8)
    out[..., 1] = (160 * y / h + 50).astype(np.uint8)
    out[..., 2] = (220 - 120 * x / w).astype(np.uint8)
    out[..., 3] = 255
    cy, cx = h * 0.55, w * 0.5
    face = ((x - cx) ** 2 / (w * 0.16) ** 2 + (y - cy) ** 2 / (h * 0.32) ** 2) < 1
    out[face, :3] = (240, 200, 160)
    return out


# --------------------------------------------------------------------------- panels
class MediaPanel(QWidget):
    def __init__(self, win, media_bin):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(media_bin, 1)
        row = QHBoxLayout()
        rec = QPushButton("Voiceover")
        rec.setToolTip("Record your voice straight onto the timeline")
        rec.clicked.connect(win.record_voiceover)
        tts = QPushButton("Text to speech")
        tts.clicked.connect(win.text_to_speech)
        row.addWidget(rec)
        row.addWidget(tts)
        lay.addLayout(row)


class AudioPanel(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        lay = QVBoxLayout(self)
        lay.addWidget(_title("Sound effects", "Click ▶ to listen, ＋ to drop it at the playhead. All royalty-free."))
        self.cat = QComboBox()
        lay.addWidget(self.cat)
        self.list = QListWidget()
        self.list.setAlternatingRowColors(True)
        lay.addWidget(self.list, 1)
        row = QHBoxLayout()
        play = QPushButton("▶  Listen")
        add = QPushButton("＋  Add at playhead")
        add.setObjectName("primary")
        row.addWidget(play)
        row.addWidget(add)
        lay.addLayout(row)
        music = QPushButton("🎵  Import music…")
        music.clicked.connect(win.import_dialog)
        lay.addWidget(music)
        play.clicked.connect(self._play)
        add.clicked.connect(self._add)
        self.list.itemDoubleClicked.connect(lambda _it: self._add())
        self.cat.currentIndexChanged.connect(self._fill)
        try:
            from .. import sfx

            self.lib = sfx.LIBRARY
        except Exception:  # noqa: BLE001
            self.lib = {}
        cats = sorted({v.get("category", "Other") for v in self.lib.values()})
        self.cat.addItem("All sounds")
        self.cat.addItems(cats)
        self._fill()

    def _fill(self):
        self.list.clear()
        want = self.cat.currentText()
        for name, info in self.lib.items():
            if want != "All sounds" and info.get("category") != want:
                continue
            it = QListWidgetItem(f"🔊  {info.get('label', name)}    ({info.get('duration', 0):.1f}s)")
            it.setData(Qt.ItemDataRole.UserRole, name)
            self.list.addItem(it)
        if self.list.count():
            self.list.setCurrentRow(0)

    def _current(self):
        it = self.list.currentItem()
        return it.data(Qt.ItemDataRole.UserRole) if it else None

    def _play(self):
        name = self._current()
        if not name:
            return
        from .. import sfx
        from ..media import load_audio

        try:
            import sounddevice as sd

            sfx.ensure_library()
            sd.play(load_audio(sfx.path(name), 48000), 48000)
        except Exception as e:  # noqa: BLE001
            self.win.status(f"Can't play sound: {e}")

    def _add(self):
        name = self._current()
        if not name:
            return
        from .. import sfx

        sfx.ensure_library()
        self.win.add_file_at_playhead(sfx.path(name), audio_track=True)


TEXT_STYLES = {
    "Title": dict(text="Your title", font_size=110, bold=True),
    "Subtitle": dict(text="Your subtitle", font_size=54, bold=False, pos_y=0.36, bg_color="#000000aa"),
    "Lower third": dict(text="Name\nRole", font_size=56, bold=True, pos_y=0.30, pos_x=-0.25, bg_color="#1060f0dd"),
    "Big bold": dict(text="WOW!", font_size=150, bold=True, color="#ffe14d"),
    "Caption pop": dict(text="caption", font_size=84, bold=True, pos_y=0.15, anim_in="pop"),
    "Neon": dict(text="NEON", font_size=120, bold=True, color="#39f3ff", bg_color=""),
    "Quote": dict(text="“Your quote here”", font_size=64, bold=False, color="#f4f1e8"),
    "Red label": dict(text="BREAKING", font_size=58, bold=True, bg_color="#e0242bff", pos_y=-0.35),
    "Minimal": dict(text="minimal", font_size=60, bold=False, color="#ffffff", font_family="Segoe UI Light"),
    "Typewriter": dict(text="Once upon a time…", font_size=58, bold=False, font_family="Consolas"),
}


class TextPanel(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        lay = QVBoxLayout(self)
        lay.addWidget(_title("Text", "Click a style to add it at the playhead, then type on the right."))
        g = Gallery()
        lay.addWidget(g, 1)
        from ..model import Clip
        from ..textrender import render_text_element

        for name, style in TEXT_STYLES.items():
            fields = {k: v for k, v in style.items() if k in Clip.__dataclass_fields__}
            c = Clip("x", "", 0, 0, 1, **fields)
            g.add(name, card_pixmap(render_text_element(c, 0.33)), name)
        g.picked.connect(lambda n: win.add_text_style(TEXT_STYLES[n]))
        cap = QPushButton("💬  Auto captions from speech")
        cap.setObjectName("primary")
        cap.clicked.connect(win.auto_captions)
        lay.addWidget(cap)


class StickersPanel(QWidget):
    def __init__(self, win):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.addWidget(_title("Stickers & shapes", "Click to add at the playhead. Resize and rotate on the right."))
        g = Gallery(QSize(80, 64))
        lay.addWidget(g, 1)
        try:
            from .. import shapes
            from ..model import Clip

            for name, shape in shapes.PRESETS.items():
                c = Clip("x", "", 0, 0, 1, shape=dict(shape))
                g.add(name, card_pixmap(shapes.render_shape(c, 64), size=QSize(80, 64)), name)
            g.picked.connect(lambda n: win.add_shape_preset(n))
        except Exception as e:  # noqa: BLE001
            lay.addWidget(QLabel(f"Shapes unavailable: {e}"))


class EffectsPanel(QWidget):
    """Effects (or Filters when filters=True) gallery with live previews on the current frame."""

    def __init__(self, win, filters: bool = False):
        super().__init__()
        self.win, self.filters = win, filters
        lay = QVBoxLayout(self)
        if filters:
            lay.addWidget(_title("Filters", "One-click colour looks. Click again to change; "
                                            "the strength is in the inspector."))
        else:
            lay.addWidget(_title("Effects", "Select a clip, then click an effect. Effects stack."))
        self.cat = QComboBox()
        if not filters:
            lay.addWidget(self.cat)
        self.g = Gallery()
        lay.addWidget(self.g, 1)
        refresh = QPushButton("↻  Preview on current frame")
        refresh.clicked.connect(self.populate)
        lay.addWidget(refresh)
        self.g.picked.connect(self._pick)
        self.cat.currentIndexChanged.connect(self.populate)
        self._cats_loaded = False

    def showEvent(self, e):
        super().showEvent(e)
        if self.g.count() == 0:
            self.populate()

    def populate(self):
        from .. import effects
        from ..render import rgba_to_qimage

        cats = effects.categories() if hasattr(effects, "categories") else {}
        if not self._cats_loaded and not self.filters:
            self.cat.blockSignals(True)
            self.cat.addItems([c for c in cats if c != "Filters"])
            self.cat.blockSignals(False)
            self._cats_loaded = True
        names = cats.get("Filters", []) if self.filters else cats.get(self.cat.currentText(), [])
        self.g.clear()
        sample = sample_rgba(self.win)
        if self.filters:
            self.g.add("None", card_pixmap(rgba_to_qimage(sample)), None, "Remove the filter")
        for n in names:
            spec = effects.REGISTRY.get(n, {})
            try:
                img = rgba_to_qimage(effects.preview_card(n, sample.copy()))
            except Exception:  # noqa: BLE001
                img = rgba_to_qimage(sample)
            self.g.add(spec.get("label", n), card_pixmap(img), n, spec.get("label", n))

    def _pick(self, name):
        if self.filters:
            self.win.set_filter(name)
        else:
            self.win.add_effect(name)


class TransitionsPanel(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        lay = QVBoxLayout(self)
        lay.addWidget(_title("Transitions", "Select the clip AFTER a cut (or nothing: the cut nearest the "
                                            "playhead is used), then click a style."))
        from .. import transitions as T

        icons = {"dissolve": "◐", "fadeblack": "●", "slideleft": "⇠", "slideright": "⇢",
                 "slideup": "⇡", "slidedown": "⇣", **getattr(T, "ICONS", {})}
        cats = getattr(T, "CATEGORIES", None) or {"All": list(T.KINDS)}
        palette = ["#2b6cb0", "#6b46c1", "#c05621", "#2f855a", "#b83280", "#2c7a7b"]
        tabs = QTabWidget()
        galleries = []
        for ci, (cat, kinds) in enumerate(cats.items()):
            g = Gallery()
            for k in kinds:
                if k in T.KINDS:
                    g.add(T.KINDS[k], icon_card(icons.get(k, "⇄"), palette[ci % 6], "#1a202c"), k, T.KINDS[k])
            tabs.addTab(g, cat)
            galleries.append(g)
        lay.addWidget(tabs, 1)
        row = QHBoxLayout()
        row.addWidget(QLabel("Length"))
        self.dur = QSlider(Qt.Orientation.Horizontal)
        self.dur.setRange(10, 200)
        self.dur.setValue(50)
        self.dur_lbl = QLabel("0.5s")
        self.dur.valueChanged.connect(lambda v: self.dur_lbl.setText(f"{v / 100:.1f}s"))
        row.addWidget(self.dur, 1)
        row.addWidget(self.dur_lbl)
        lay.addLayout(row)
        allb = QPushButton("Apply selected style to every cut")
        lay.addWidget(allb)
        self.kind = "dissolve"
        for g in galleries:
            g.picked.connect(self._pick)
        allb.clicked.connect(lambda: win.transitions_everywhere(self.kind, self.dur.value() / 100))

    def _pick(self, kind):
        self.kind = kind
        self.win.apply_transition(kind, self.dur.value() / 100)


class AnimationsPanel(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        lay = QVBoxLayout(self)
        lay.addWidget(_title("Animation", "Select a clip, title or sticker, then pick how it enters, "
                                          "leaves or loops."))
        tabs = QTabWidget()
        lay.addWidget(tabs, 1)
        try:
            from .. import animate

            anims, info = animate.ANIMATIONS, getattr(animate, "PRESET_INFO", {})
        except Exception:  # noqa: BLE001
            anims, info = {"in": {}, "out": {}, "loop": {}}, {}
        colors = {"in": ("#2f855a", "#1c4532"), "out": ("#c53030", "#63171b"), "loop": ("#6b46c1", "#322659")}
        for kind, title in (("in", "In"), ("out", "Out"), ("loop", "Loop")):
            g = Gallery(QSize(96, 54))
            g.add("None", icon_card("∅", "#4a5568", "#1a202c", QSize(96, 54)), "")
            for name, label in anims.get(kind, {}).items():
                g.add(label, icon_card("▶" if kind == "in" else "◀" if kind == "out" else "↻",
                                       *colors[kind], QSize(96, 54)), name,
                      (info.get(kind, {}) or {}).get(name, label))
            g.picked.connect(lambda n, k=kind: self.win.set_animation(k, n, self.dur.value() / 100))
            tabs.addTab(g, title)
        row = QHBoxLayout()
        row.addWidget(QLabel("Length / speed"))
        self.dur = QSlider(Qt.Orientation.Horizontal)
        self.dur.setRange(10, 300)
        self.dur.setValue(60)
        self.lbl = QLabel("0.6s")
        self.dur.valueChanged.connect(lambda v: self.lbl.setText(f"{v / 100:.1f}s"))
        row.addWidget(self.dur, 1)
        row.addWidget(self.lbl)
        lay.addLayout(row)


AI_TOOLS = [
    ("assistant", "💬", "AI Assistant", "Tell ClipEditor what to do in plain words — it edits for you."),
    ("captions", "📝", "Auto captions", "Speech → subtitles, offline (Whisper)."),
    ("silences", "✂", "Remove silences", "Cut out pauses automatically."),
    ("bgremove", "🧍", "Remove background", "Cut out people / objects without a green screen."),
    ("ai_denoise", "🎧", "AI voice cleanup", "Remove background noise from speech (RNNoise)."),
    ("enhance", "🔆", "Enhance / upscale", "Sharpen and clean soft or old footage, up to 2x."),
    ("stabilize", "🎯", "Stabilize", "Steady a shaky shot."),
    ("smooth_slowmo", "🐢", "Smooth slow motion", "Optical-flow frame interpolation."),
    ("auto_reframe", "📱", "Auto reframe 9:16", "Smart crop that follows the action, for Shorts/Reels."),
    ("reverse", "⏪", "Reverse clip", "Play the clip backwards."),
    ("tts", "🗣", "Text to speech", "Turn a script into a voiceover."),
    ("isolate_vocals", "🎤", "Isolate voice", "Keep only the voice — remove music and background."),
    ("remove_vocals", "🎼", "Remove vocals", "Karaoke: keep the music, remove the singing/speech."),
    ("voice_clone", "🧬", "Voice clone / match", "Speak new text in a voice from a recording."),
]


class AIPanel(QWidget):
    def __init__(self, win):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.addWidget(_title("AI Magic", "Everything runs on your computer. Models download once."))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setSpacing(6)
        for key, icon, label, desc in AI_TOOLS:
            b = QPushButton(f"{icon}   {label}\n      {desc}")
            b.setStyleSheet("text-align: left; padding: 8px 10px;")
            b.setMinimumHeight(52)
            b.clicked.connect(lambda _=False, k=key: win.run_ai_tool(k))
            bl.addWidget(b)
        bl.addStretch(1)
        scroll.setWidget(body)
        lay.addWidget(scroll, 1)


class Sidebar(QWidget):
    def __init__(self, win, media_bin):
        super().__init__()
        self.setMinimumWidth(250)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        rail = QWidget()
        rail.setObjectName("rail")
        rail.setStyleSheet("#rail { background: #18191c; }")
        rl = QVBoxLayout(rail)
        rl.setContentsMargins(4, 6, 4, 6)
        rl.setSpacing(2)
        self.stack = QStackedWidget()
        self.group = QButtonGroup(self)
        self.keys = []
        makers = {
            "media": lambda: MediaPanel(win, media_bin), "audio": lambda: AudioPanel(win),
            "text": lambda: TextPanel(win), "stickers": lambda: StickersPanel(win),
            "effects": lambda: EffectsPanel(win), "filters": lambda: EffectsPanel(win, filters=True),
            "transitions": lambda: TransitionsPanel(win), "animations": lambda: AnimationsPanel(win),
            "ai": lambda: AIPanel(win),
        }
        for i, (key, icon, label) in enumerate(RAIL):
            b = QToolButton()
            b.setText(f"{icon}\n{label}")
            b.setCheckable(True)
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            b.setFixedSize(60, 46)
            b.setStyleSheet("QToolButton { border: none; border-radius: 8px; color: #b7bcc6; font-size: 8pt; }"
                            "QToolButton:checked { background: #23395d; color: white; }"
                            "QToolButton:hover { background: #2a2c31; }")
            self.group.addButton(b, i)
            rl.addWidget(b)
            try:
                panel = makers[key]()
            except Exception as e:  # noqa: BLE001 - a broken panel must not kill the app
                panel = QLabel(f"{label} unavailable:\n{e}")
                panel.setWordWrap(True)
            self.stack.addWidget(panel)
            self.keys.append(key)
        rl.addStretch(1)
        # the rail scrolls on short screens instead of forcing the window taller
        from PySide6.QtWidgets import QFrame
        rail_scroll = QScrollArea()
        rail_scroll.setWidget(rail)
        rail_scroll.setWidgetResizable(True)
        rail_scroll.setFrameShape(QFrame.Shape.NoFrame)
        rail_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        rail_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        rail_scroll.setFixedWidth(70)
        rail_scroll.setMinimumHeight(120)
        lay.addWidget(rail_scroll)
        lay.addWidget(self.stack, 1)
        self.group.idClicked.connect(self.stack.setCurrentIndex)
        self.group.button(0).setChecked(True)
        self.stack.setCurrentIndex(0)

    def show_panel(self, key: str):
        i = self.keys.index(key)
        self.group.button(i).setChecked(True)
        self.stack.setCurrentIndex(i)
