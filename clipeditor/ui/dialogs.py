"""Welcome screen, export dialog and shortcut help."""
from __future__ import annotations

import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
                               QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QPushButton, QRadioButton, QTextBrowser, QVBoxLayout, QWidget)

ASPECTS = [
    ("16:9", "YouTube / landscape", 1920, 1080),
    ("9:16", "TikTok · Reels · Shorts", 1080, 1920),
    ("1:1", "Instagram square", 1080, 1080),
    ("4:5", "Instagram portrait", 1080, 1350),
]


class WelcomePage(QWidget):
    import_clicked = Signal()
    open_clicked = Signal()
    open_recent = Signal(str)
    aspect_chosen = Signal(int, int)

    def __init__(self):
        super().__init__()
        outer = QVBoxLayout(self)
        outer.addStretch(1)
        card = QFrame()
        card.setObjectName("drop")
        card.setStyleSheet("#drop { border: 2px dashed #4a4f5a; border-radius: 14px; background: #24262b; }")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(36, 30, 36, 30)
        cl.setSpacing(12)
        h = QLabel("Start your video")
        h.setStyleSheet("font-size: 20pt; font-weight: 700;")
        h.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub = QLabel("Drop videos, music or photos anywhere in this window")
        sub.setStyleSheet("color: #9aa0aa; font-size: 11pt;")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cl.addWidget(h)
        cl.addWidget(sub)

        shape_lbl = QLabel("1. Pick a shape")
        shape_lbl.setStyleSheet("font-weight: 600; margin-top: 8px;")
        cl.addWidget(shape_lbl)
        row = QHBoxLayout()
        self.group = QButtonGroup(self)
        for i, (ratio, desc, w, hgt) in enumerate(ASPECTS):
            b = QRadioButton(f"{ratio}\n{desc}")
            b.setStyleSheet("QRadioButton { padding: 8px; border: 1px solid #3c4048; border-radius: 8px; }"
                            "QRadioButton:checked { border-color: #0568fd; background: #1c2a44; }")
            b.toggled.connect(lambda on, w=w, hgt=hgt: on and self.aspect_chosen.emit(w, hgt))
            self.group.addButton(b, i)
            row.addWidget(b)
        self.group.button(0).setChecked(True)
        cl.addLayout(row)

        add_lbl = QLabel("2. Add your media")
        add_lbl.setStyleSheet("font-weight: 600; margin-top: 8px;")
        cl.addWidget(add_lbl)
        btns = QHBoxLayout()
        imp = QPushButton("＋  Import media")
        imp.setObjectName("primary")
        imp.setMinimumHeight(40)
        imp.clicked.connect(self.import_clicked)
        opn = QPushButton("Open a project…")
        opn.setMinimumHeight(40)
        opn.clicked.connect(self.open_clicked)
        btns.addWidget(imp, 2)
        btns.addWidget(opn, 1)
        cl.addLayout(btns)

        self.recent_lbl = QLabel("Recent projects")
        self.recent_lbl.setStyleSheet("font-weight: 600; margin-top: 8px;")
        self.recent = QListWidget()
        self.recent.setMaximumHeight(110)
        self.recent.itemActivated.connect(lambda it: self.open_recent.emit(it.data(Qt.ItemDataRole.UserRole)))
        self.recent.itemClicked.connect(lambda it: self.open_recent.emit(it.data(Qt.ItemDataRole.UserRole)))
        cl.addWidget(self.recent_lbl)
        cl.addWidget(self.recent)

        wrap = QHBoxLayout()
        wrap.addStretch(1)
        wrap.addWidget(card, 4)
        wrap.addStretch(1)
        outer.addLayout(wrap)
        outer.addStretch(1)

    def set_recent(self, paths):
        self.recent.clear()
        paths = [p for p in paths if os.path.exists(p)]
        for p in paths[:6]:
            it = QListWidgetItem(f"{os.path.basename(p)}    —    {os.path.dirname(p)}")
            it.setData(Qt.ItemDataRole.UserRole, p)
            self.recent.addItem(it)
        self.recent.setVisible(bool(paths))
        self.recent_lbl.setVisible(bool(paths))

    def select_aspect(self, w: int, h: int):
        for i, (_, _, aw, ah) in enumerate(ASPECTS):
            if (aw, ah) == (w, h):
                self.group.blockSignals(True)
                self.group.button(i).setChecked(True)
                self.group.blockSignals(False)


EXPORT_PRESETS = [
    ("Same as project", None),
    ("1080p (Full HD)", 1080),
    ("720p (HD, smaller file)", 720),
    ("480p (quick share)", 480),
    ("4K (2160p)", 2160),
]
QUALITY = [("Best", 17, "slow"), ("Great (recommended)", 20, "medium"), ("Small file", 26, "fast"),
           ("Draft (fastest)", 30, "ultrafast")]


class ExportDialog(QDialog):
    def __init__(self, parent, project, default_path: str):
        super().__init__(parent)
        self.setWindowTitle("Export video")
        self.setMinimumWidth(480)
        self.project = project
        lay = QVBoxLayout(self)
        head = QLabel("Export your video")
        head.setStyleSheet("font-size: 14pt; font-weight: 700;")
        lay.addWidget(head)
        info = QLabel(f"Length {project.duration:.1f}s  ·  canvas {project.width}×{project.height}  ·  "
                      f"{project.fps:g} fps  ·  MP4 (H.264 + AAC)")
        info.setStyleSheet("color: #9aa0aa;")
        lay.addWidget(info)
        form = QFormLayout()
        self.path = QLineEdit(default_path)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.path, 1)
        row.addWidget(browse)
        form.addRow("Save to", row)
        self.res = QComboBox()
        for name, _ in EXPORT_PRESETS:
            self.res.addItem(name)
        self.res.currentIndexChanged.connect(self._update_size)
        form.addRow("Resolution", self.res)
        self.quality = QComboBox()
        for name, *_ in QUALITY:
            self.quality.addItem(name)
        self.quality.setCurrentIndex(1)
        form.addRow("Quality", self.quality)
        from PySide6.QtWidgets import QCheckBox

        from .. import encoders

        self.codec = QComboBox()
        for k, label in encoders.CODECS.items():
            self.codec.addItem(label, k)
        form.addRow("Format", self.codec)
        self.fps = QComboBox()
        self.fps.addItem(f"Same as project ({project.fps:g})", None)
        for f in (24, 25, 30, 50, 60):
            self.fps.addItem(f"{f} fps", float(f))
        form.addRow("Frame rate", self.fps)
        self.gpu = QCheckBox("Use graphics card (much faster)")
        self.ten = QCheckBox("10-bit colour (HDR-ready, HEVC/AV1)")
        form.addRow("", self.gpu)
        form.addRow("", self.ten)
        self.codec.currentIndexChanged.connect(self._update_gpu)
        self._update_gpu()
        self.size_lbl = QLabel()
        self.size_lbl.setStyleSheet("color: #9aa0aa;")
        form.addRow("Output", self.size_lbl)
        lay.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        go = bb.addButton("Export", QDialogButtonBox.ButtonRole.AcceptRole)
        go.setObjectName("primary")
        go.setDefault(True)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._update_size()

    def _browse(self):
        p, _ = QFileDialog.getSaveFileName(self, "Export video", self.path.text(), "MP4 video (*.mp4)")
        if p:
            self.path.setText(p)

    def out_size(self) -> tuple[int, int]:
        target = EXPORT_PRESETS[self.res.currentIndex()][1]
        w, h = self.project.width, self.project.height
        if target:
            short = min(w, h)
            k = target / short
            w, h = int(round(w * k)), int(round(h * k))
        return w - w % 2, h - h % 2

    def _update_size(self):
        w, h = self.out_size()
        self.size_lbl.setText(f"{w} × {h}")

    def _update_gpu(self):
        from .. import encoders

        hw = encoders.gpu_encoder(self.codec.currentData())
        self.gpu.setEnabled(hw is not None)
        self.gpu.setChecked(hw is not None)
        self.gpu.setText(f"Use graphics card — {hw[1]} (much faster)" if hw else "Graphics-card encoding not available")

    def options(self) -> dict:
        w, h = self.out_size()
        _, crf, preset = QUALITY[self.quality.currentIndex()]
        opts = dict(width=w, height=h, crf=crf, preset=preset, codec=self.codec.currentData(),
                    gpu=self.gpu.isChecked(), ten_bit=self.ten.isChecked())
        if self.fps.currentData():
            opts["fps"] = self.fps.currentData()
        return opts

    def out_path(self) -> str:
        p = self.path.text().strip()
        return p if p.lower().endswith(".mp4") else p + ".mp4"


SHORTCUTS = """
<h3>Playback</h3>
<table cellpadding=3>
<tr><td><b>Space</b></td><td>Play / pause</td></tr>
<tr><td><b>← / →</b></td><td>Previous / next frame</td></tr>
<tr><td><b>Shift + ← / →</b></td><td>Jump 1 second</td></tr>
<tr><td><b>Home / End</b></td><td>Go to start / end</td></tr>
</table>
<h3>Editing</h3>
<table cellpadding=3>
<tr><td><b>S</b></td><td>Split at playhead (selected clips, or everything under it)</td></tr>
<tr><td><b>T</b></td><td>Add text at playhead</td></tr>
<tr><td><b>Delete</b></td><td>Delete selected clips</td></tr>
<tr><td><b>Ctrl + Delete</b></td><td>Ripple delete clips (close the gap)</td></tr>
<tr><td><b>Ctrl + D</b></td><td>Duplicate</td></tr>
<tr><td><b>Ctrl + Shift + D</b></td><td>Detach audio from video</td></tr>
<tr><td><b>Ctrl + Z / Ctrl + Y</b></td><td>Undo / redo</td></tr>
<tr><td><b>Ctrl + click</b></td><td>Select several clips</td></tr>
</table>
<h3>Range &amp; frames</h3>
<table cellpadding=3>
<tr><td><b>Shift + drag ruler</b></td><td>Mark a range</td></tr>
<tr><td><b>I / O</b></td><td>Mark range start / end at playhead</td></tr>
<tr><td><b>Shift + Delete</b></td><td>Remove range (close the gap)</td></tr>
<tr><td><b>Shift + Space</b></td><td>Play range</td></tr>
<tr><td><b>Alt + X</b></td><td>Clear range</td></tr>
<tr><td><b>Alt + Delete</b></td><td>Remove the frame under the playhead</td></tr>
<tr><td><b>Alt + D</b></td><td>Duplicate that frame</td></tr>
<tr><td><b>F</b></td><td>Freeze frame for 2 seconds</td></tr>
<tr><td><b>Ctrl + X / C / V</b></td><td>Cut / copy / paste clips</td></tr>
<tr><td><b>Ctrl + Shift + C</b></td><td>Auto captions</td></tr>
</table>
<h3>Timeline</h3>
<table cellpadding=3>
<tr><td><b>Drag clip</b></td><td>Move (also to another track)</td></tr>
<tr><td><b>Drag clip edge</b></td><td>Trim</td></tr>
<tr><td><b>Drag white dots</b></td><td>Fade in / out</td></tr>
<tr><td><b>Click ＋ between clips</b></td><td>Add a transition (⇄ to edit it)</td></tr>
<tr><td><b>Ctrl + wheel</b></td><td>Zoom</td></tr>
<tr><td><b>Shift + Z</b></td><td>Zoom to fit</td></tr>
<tr><td><b>Right-click</b></td><td>More actions</td></tr>
</table>
<h3>Files</h3>
<table cellpadding=3>
<tr><td><b>Ctrl + I</b></td><td>Import media</td></tr>
<tr><td><b>Ctrl + S</b></td><td>Save project (also autosaved every minute)</td></tr>
<tr><td><b>Ctrl + E</b></td><td>Export video</td></tr>
</table>
"""


class ShortcutsDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Keyboard shortcuts")
        self.resize(520, 620)
        lay = QVBoxLayout(self)
        tb = QTextBrowser()
        tb.setHtml(SHORTCUTS)
        lay.addWidget(tb)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
