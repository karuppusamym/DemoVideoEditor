"""Main application window."""
from __future__ import annotations

import os
import threading
import time

from PySide6.QtCore import QObject, QSettings, QStandardPaths, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QKeySequence, QPalette
from PySide6.QtWidgets import (QApplication, QComboBox, QDockWidget, QSlider, QFileDialog, QHBoxLayout, QInputDialog, QLabel, QMainWindow, QMenu,
                               QMessageBox, QProgressDialog, QPushButton, QSizePolicy, QSplitter,
                               QStackedWidget, QToolBar, QToolButton, QVBoxLayout, QWidget)

from ..editor import Editor
from ..media import AudioCache
from ..model import Project
from .dialogs import ExportDialog, ShortcutsDialog, WelcomePage
from .inspector import Inspector
from .panels import MediaBin
from .sidebar import Sidebar
from .preview import Compositor, Player, PreviewWidget
from .timeline import TimelinePanel, fmt_time
from .workers import AssetLoader
from ..transitions import DEFAULT_DURATION, KINDS, neighbour_before

MEDIA_FILTER = ("Media (*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.mp3 *.wav *.aac *.m4a *.flac *.ogg "
                "*.png *.jpg *.jpeg *.bmp *.webp *.gif);;All files (*)")
PROJECT_FILTER = "ClipEditor project (*.clipeditor.json *.json)"
AUTOSAVE_MS = 60_000

TEXT_PRESETS = {
    "Title": dict(text="Your title", font_size=110, bold=True),
    "Subtitle": dict(text="Your subtitle", font_size=54, bold=False, pos_y=0.36, bg_color="#000000aa"),
    "Lower third": dict(text="Name\nRole", font_size=56, bold=True, pos_y=0.30, pos_x=-0.25,
                        bg_color="#1060f0dd"),
}


def app_data_dir() -> str:
    d = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    os.makedirs(d, exist_ok=True)
    return d


class _JobStatus(QObject):
    text = Signal(str)


class _ExportSignals(QObject):
    progress = Signal(float)
    done = Signal(str)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("ClipEditor", "ClipEditor")
        self.ed = Editor()
        self.audio = AudioCache()
        self.loader = AssetLoader(self.audio)
        self.comp = Compositor(use_proxies=True)
        self.player = Player(lambda: self.ed.project, self.audio)
        self.playhead = 0.0
        self._visible_timeline_id = None
        self._timeline_playheads = {}
        self._render_pending = False
        self._aspect_locked = False
        self.autosave_path = os.path.join(app_data_dir(), "autosave.clipeditor.json")

        self.bin = MediaBin(self.ed, self.loader)
        self.preview = PreviewWidget()
        self.inspector = Inspector(self.ed, self)
        self.timeline = TimelinePanel(self.ed, self.loader)
        self.welcome = WelcomePage()

        # transport under the preview
        self.btn_start = self._tbtn("⏮", "Go to start (Home)", lambda: self.seek(0))
        self.btn_back = self._tbtn("◀◀", "Back 1 second (Shift+←)", lambda: self.seek(self.playhead - 1))
        self.btn_play = self._tbtn("▶", "Play / pause (Space)", self.toggle_play)
        self.btn_play.setObjectName("play")
        self.btn_fwd = self._tbtn("▶▶", "Forward 1 second (Shift+→)", lambda: self.seek(self.playhead + 1))
        self.btn_end = self._tbtn("⏭", "Go to end (End)", lambda: self.seek(self.ed.project.duration))
        self.time_lbl = QLabel()
        self.time_lbl.setStyleSheet("font-family: Consolas, monospace; font-size: 11pt; color: #d6d8de;")
        self.ratio = QComboBox()
        for label, w, h in (("16:9", 1920, 1080), ("9:16", 1080, 1920), ("1:1", 1080, 1080), ("4:5", 1080, 1350),
                            ("4:3", 1440, 1080), ("21:9", 2560, 1080), ("4K 16:9", 3840, 2160)):
            self.ratio.addItem(label, (w, h))
        self.ratio.setToolTip("Canvas shape")
        self.ratio.activated.connect(lambda _i: self.set_canvas(*self.ratio.currentData()))
        self.safe_btn = QPushButton("▦ Safe zones")
        self.safe_btn.setCheckable(True)
        self.safe_btn.setToolTip("Show where TikTok / Reels / Shorts buttons and captions cover the video")
        self.safe_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.safe_btn.toggled.connect(self._toggle_safe)
        transport = QHBoxLayout()
        transport.addWidget(self.ratio)
        transport.addWidget(self.safe_btn)
        transport.addStretch(1)
        for b in (self.btn_start, self.btn_back, self.btn_play, self.btn_fwd, self.btn_end):
            transport.addWidget(b)
        transport.addSpacing(16)
        transport.addWidget(self.time_lbl)
        transport.addStretch(1)

        player_page = QWidget()
        pl = QVBoxLayout(player_page)
        pl.setContentsMargins(6, 6, 6, 4)
        pl.addWidget(self.preview, 1)
        pl.addLayout(transport)
        self.center = QStackedWidget()
        self.center.addWidget(self.welcome)
        self.center.addWidget(player_page)

        self.sidebar = Sidebar(self, self.bin)
        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(self.sidebar)
        top.addWidget(self.center)
        top.addWidget(self.inspector)
        top.setStretchFactor(1, 1)
        top.setSizes([400, 720, 340])
        main = QSplitter(Qt.Orientation.Vertical)
        main.addWidget(top)
        tl_box = QWidget()
        tl_lay = QVBoxLayout(tl_box)
        tl_lay.setContentsMargins(0, 0, 0, 0)
        tl_lay.setSpacing(0)
        tl_lay.addWidget(self._build_timeline_selector())
        tl_lay.addWidget(self._build_timeline_bar())
        self.range_bar = self._build_range_bar()
        tl_lay.addWidget(self.range_bar)
        tl_lay.addWidget(self.timeline, 1)
        main.addWidget(tl_box)
        main.setSizes([540, 320])
        self.setCentralWidget(main)
        self.resize(1520, 940)
        self.assistant_dock = None

        self._build_toolbar()
        self._build_menus()
        self.ed.listeners.append(self.on_editor_change)
        self.loader.thumbs_ready.connect(self._on_asset_data)
        self.loader.audio_ready.connect(self._on_asset_data)
        self.loader.failed.connect(lambda aid, msg: self.statusBar().showMessage(f"Couldn't read a file: {msg}", 8000))
        self.player.tick.connect(self._on_tick)
        self.player.stopped.connect(lambda: self.btn_play.setText("▶"))
        self.timeline.canvas.seek.connect(self.seek)
        self.timeline.canvas.files_dropped.connect(self._files_dropped_on_timeline)
        self.timeline.canvas.transition_added.connect(
            lambda: self.statusBar().showMessage("Transition added — pick a style and length on the right", 5000))
        self.bin.import_requested.connect(self.import_dialog)
        self.bin.add_at_playhead.connect(lambda aid: self.ed.add_clip(aid, None, self.playhead))
        self.welcome.import_clicked.connect(self.import_dialog)
        self.welcome.open_clicked.connect(lambda: self.open_project())
        self.welcome.open_recent.connect(self.open_project)
        self.welcome.aspect_chosen.connect(self._choose_aspect)
        self.welcome.set_recent(self.recent_projects())
        self.setAcceptDrops(True)

        self.autosave_timer = QTimer(self)
        self.autosave_timer.timeout.connect(self.autosave)
        self.autosave_timer.start(AUTOSAVE_MS)
        self.statusBar().showMessage("Welcome! Import media to get started  ·  press F1 for shortcuts", 10000)
        self.on_editor_change("load")

    # ---- toolbar / menus -------------------------------------------------
    def _tbtn(self, text, tip, fn):
        b = QPushButton(text)
        b.setToolTip(tip)
        b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        b.setMinimumWidth(44)
        b.clicked.connect(fn)
        return b

    def _build_toolbar(self):
        tb = QToolBar("Header")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.addToolBar(tb)
        logo = QLabel("  ◆ ClipEditor  ")
        logo.setStyleSheet("font-size: 13pt; font-weight: 800; color: #5ea1ff;")
        tb.addWidget(logo)
        self.proj_lbl = QLabel()
        self.proj_lbl.setStyleSheet("color: #aab0ba; padding-left: 8px;")
        tb.addWidget(self.proj_lbl)
        sp = QWidget()
        sp.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(sp)
        search = QPushButton("🔎   Search commands, effects, stickers…        Ctrl+K")
        search.setMinimumWidth(380)
        search.setStyleSheet("text-align: left; color: #9aa0aa; background: #1f2024; padding: 6px 12px;")
        search.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        search.clicked.connect(self.open_palette)
        tb.addWidget(search)
        sp2 = QWidget()
        sp2.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(sp2)
        for text, tip, fn in (("↶", "Undo (Ctrl+Z)", self.ed.undo), ("↷", "Redo (Ctrl+Y)", lambda: self.ed.redo())):
            a = QAction(text, self)
            a.setToolTip(tip)
            a.triggered.connect(fn)
            tb.addAction(a)
        ai = QPushButton("💬  AI Assistant")
        ai.setToolTip("Edit by chatting: “remove the pauses and add captions”")
        ai.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        ai.clicked.connect(self.toggle_assistant)
        tb.addWidget(ai)
        a = QAction("❔", self)
        a.setToolTip("Keyboard shortcuts (F1)")
        a.triggered.connect(self.show_shortcuts)
        tb.addAction(a)
        exp = QPushButton("  ⬆  Export  ")
        exp.setObjectName("primary")
        exp.setToolTip("Save your finished video (Ctrl+E)")
        exp.clicked.connect(self.export_dialog)
        tb.addWidget(exp)

    def _build_timeline_selector(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(6, 4, 6, 2)
        label = QLabel("Timeline")
        self.timeline_choice = QComboBox()
        self.timeline_choice.setAccessibleName("Active timeline")
        self.timeline_choice.setMinimumContentsLength(14)
        self.timeline_choice.setMaximumWidth(320)
        self.timeline_choice.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.timeline_choice.activated.connect(lambda _index: self.ed.switch_timeline(self.timeline_choice.currentData()))
        label.setBuddy(self.timeline_choice)
        layout.addWidget(label)
        layout.addWidget(self.timeline_choice)
        layout.addWidget(self._tbtn("New…", "Create a timeline in this project", self._create_timeline))
        options = QToolButton()
        options.setText("Manage")
        options.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(options)
        menu.addAction("Rename timeline…", self._rename_timeline)
        menu.addAction("Duplicate timeline…", lambda: self._create_timeline(True))
        self.delete_timeline_action = menu.addAction("Delete timeline…", self._delete_timeline)
        options.setMenu(menu)
        layout.addWidget(options)
        layout.addStretch(1)
        return bar

    def _refresh_timeline_selector(self):
        project = self.ed.project
        entries = [(timeline.name, timeline.id) for timeline in project.timelines.values()]
        current = [(self.timeline_choice.itemText(i), self.timeline_choice.itemData(i))
                   for i in range(self.timeline_choice.count())]
        if current != entries:
            self.timeline_choice.clear()
            for name, timeline_id in entries:
                self.timeline_choice.addItem(name, timeline_id)
        self.timeline_choice.setCurrentIndex(self.timeline_choice.findData(project.active_timeline_id))
        self.delete_timeline_action.setEnabled(len(entries) > 1)

    def _create_timeline(self, duplicate=False):
        title = "Duplicate timeline" if duplicate else "New timeline"
        name, accepted = QInputDialog.getText(self, title, "Name", text=self.ed.default_timeline_name(duplicate))
        if accepted:
            try:
                self.ed.create_timeline(name, duplicate)
            except ValueError as error:
                QMessageBox.warning(self, title, str(error))

    def _rename_timeline(self):
        timeline = self.ed.project.active_timeline
        name, accepted = QInputDialog.getText(self, "Rename timeline", "Name", text=timeline.name)
        if accepted:
            try:
                self.ed.rename_timeline(timeline.id, name)
            except ValueError as error:
                QMessageBox.warning(self, "Rename timeline", str(error))

    def _delete_timeline(self):
        timeline = self.ed.project.active_timeline
        answer = QMessageBox.question(self, "Delete timeline", f'Delete "{timeline.name}" and its edits?',
                                      defaultButton=QMessageBox.StandardButton.No)
        if answer == QMessageBox.StandardButton.Yes:
            try:
                self.ed.delete_timeline(timeline.id)
            except ValueError as error:
                QMessageBox.warning(self, "Delete timeline", str(error))

    def _build_timeline_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("tlbar")
        bar.setStyleSheet("#tlbar { background: #26282d; border-top: 1px solid #1b1c1f; }"
                          "#tlbar QToolButton { padding: 4px 8px; border-radius: 6px; color: #dfe1e6; }"
                          "#tlbar QToolButton:hover { background: #3a3d44; }"
                          "#tlbar QToolButton:checked { background: #23395d; }")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(6, 2, 6, 2)
        lay.setSpacing(2)

        def tool(text, tip, fn, checkable=False):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.setCheckable(checkable)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.clicked.connect(fn)
            lay.addWidget(b)
            return b

        tool("✂ Split", "Split at playhead (S)", self.split_at_playhead)
        tool("🗑 Delete", "Delete clips or the marked range (Del)", self.smart_delete)
        tool("⧉ Duplicate", "Duplicate clips or range (Ctrl+D)", self.smart_duplicate)
        tool("❄ Freeze", "Freeze the frame under the playhead for 2 s (F)", lambda: self.freeze_frame(2.0))
        tool("⇋ Mirror", "Flip the selected clip left↔right", lambda: self.ed.update_selected(
            lambda c: setattr(c, "flip_h", not c.flip_h)))
        tool("⟳ Rotate", "Rotate the selected clip 90°", lambda: self.ed.update_selected(
            lambda c: setattr(c, "rotation", (c.rotation + 90) % 360)))
        tool("⏪ Reverse", "Play the selected clip backwards", lambda: self.run_ai_tool("reverse"))
        tool("⇄ Transition", "Add a transition at the nearest cut", self.add_transition_near_playhead)
        tool("🅣 Text", "Add a title at the playhead (T)", lambda: self.add_text("Title"))
        fr = QToolButton()
        fr.setText("🎞 Frame ▾")
        fr.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        fm = QMenu(fr)
        fm.addAction("Delete one frame (hard cut)   (Alt+Del)", self.remove_frame)
        fm.addAction("Duplicate this frame   (Alt+D)", self.duplicate_frame)
        fm.addAction("Freeze this frame for 2 s   (F)", lambda: self.freeze_frame(2.0))
        fm.addAction("Save this frame as picture…", self.save_frame_png)
        fm.addSeparator()
        fm.addAction("Mark range start here   (I)", self.mark_in)
        fm.addAction("Mark range end here   (O)", self.mark_out)
        fr.setMenu(fm)
        lay.addWidget(fr)
        lay.addStretch(1)
        self.snap_btn = tool("🧲", "Snapping on/off", lambda: self._toggle_snap(self.snap_btn.isChecked()), True)
        self.snap_btn.setChecked(True)
        lay.addWidget(QLabel("  🔍"))
        self.zoom = QSlider(Qt.Orientation.Horizontal)
        self.zoom.setRange(2, 400)
        self.zoom.setValue(60)
        self.zoom.setFixedWidth(150)
        self.zoom.setToolTip("Timeline zoom (Ctrl+wheel)")
        self.zoom.valueChanged.connect(lambda v: self.timeline.canvas.set_zoom(float(v)))
        lay.addWidget(self.zoom)
        tool("⤢ Fit", "Zoom to fit (Shift+Z)", self.zoom_fit)
        return bar

    def _act(self, menu, text, fn, key=None):
        a = QAction(text, self)
        if key:
            a.setShortcut(QKeySequence(key))
            a.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        a.triggered.connect(fn)
        menu.addAction(a)
        self.addAction(a)
        return a

    def _build_menus(self):
        mb = self.menuBar()
        f = mb.addMenu("&File")
        self._act(f, "New project", self.new_project, "Ctrl+N")
        self._act(f, "Open project…", lambda: self.open_project(), "Ctrl+O")
        self.recent_menu = f.addMenu("Open recent")
        self.recent_menu.aboutToShow.connect(self._fill_recent_menu)
        self._act(f, "Save", self.save_project, "Ctrl+S")
        self._act(f, "Save as…", lambda: self.save_project(True), "Ctrl+Shift+S")
        f.addSeparator()
        self._act(f, "Import media…", self.import_dialog, "Ctrl+I")
        self._act(f, "Locate missing media…", self.relink_dialog)
        self._act(f, "Export video…", self.export_dialog, "Ctrl+E")
        f.addSeparator()
        canvas = f.addMenu("Canvas shape")
        for label, w, h in (("16:9 landscape (1920×1080)", 1920, 1080), ("9:16 vertical (1080×1920)", 1080, 1920),
                            ("1:1 square (1080×1080)", 1080, 1080), ("4:5 portrait (1080×1350)", 1080, 1350),
                            ("4K 16:9 (3840×2160)", 3840, 2160), ("720p 16:9 (1280×720)", 1280, 720)):
            canvas.addAction(label, lambda w=w, h=h: self.set_canvas(w, h))
        f.addSeparator()
        self._act(f, "Quit", self.close, "Ctrl+Q")
        e = mb.addMenu("&Edit")
        self._act(e, "Undo", self.ed.undo, "Ctrl+Z")
        self._act(e, "Redo", self.ed.redo, "Ctrl+Y")
        e.addSeparator()
        self._act(e, "Split at playhead", self.split_at_playhead, "S")
        self._act(e, "Add text", lambda: self.add_text("Title"), "T")
        self._act(e, "Cut", self.cut, "Ctrl+X")
        self._act(e, "Copy", self.copy, "Ctrl+C")
        self._act(e, "Paste at playhead", self.paste, "Ctrl+V")
        e.addSeparator()
        self._act(e, "Delete (clips or range)", self.smart_delete, "Delete")
        self._act(e, "Ripple delete clips", lambda: self.ed.delete_selected(ripple=True), "Ctrl+Delete")
        self._act(e, "Duplicate (clips or range)", self.smart_duplicate, "Ctrl+D")
        self._act(e, "Detach audio", self.ed.detach_audio, "Ctrl+Shift+D")
        self._act(e, "Select all", self.select_all, "Ctrl+A")
        e.addSeparator()
        self._act(e, "Add video track", lambda: self.ed.add_track("video"))
        self._act(e, "Add audio track", lambda: self.ed.add_track("audio"))
        v = mb.addMenu("&Playback")
        self._act(v, "Play / pause", self.toggle_play, "Space")
        self._act(v, "Previous frame", lambda: self.seek(self.playhead - 1 / self.ed.project.fps), "Left")
        self._act(v, "Next frame", lambda: self.seek(self.playhead + 1 / self.ed.project.fps), "Right")
        self._act(v, "Back 1s", lambda: self.seek(self.playhead - 1), "Shift+Left")
        self._act(v, "Forward 1s", lambda: self.seek(self.playhead + 1), "Shift+Right")
        self._act(v, "Go to start", lambda: self.seek(0), "Home")
        self._act(v, "Go to end", lambda: self.seek(self.ed.project.duration), "End")
        v.addSeparator()
        cv = self.timeline.canvas
        self._act(v, "Zoom in", lambda: cv.set_zoom(cv.pps * 1.25), "Ctrl+=")
        self._act(v, "Zoom out", lambda: cv.set_zoom(cv.pps / 1.25), "Ctrl+-")
        self._act(v, "Zoom to fit", self.zoom_fit, "Shift+Z")
        r = mb.addMenu("&Range && Frame")
        self._act(r, "Mark range start (In)", self.mark_in, "I")
        self._act(r, "Mark range end (Out)", self.mark_out, "O")
        self._act(r, "Select range of selected clip", self.range_from_selection, "Shift+X")
        self._act(r, "Clear range", self.clear_marks, "Alt+X")
        r.addSeparator()
        self._act(r, "Remove range (close gap)", self.remove_range, "Shift+Delete")
        self._act(r, "Duplicate range", self.duplicate_range)
        self._act(r, "Play range", self.play_range, "Shift+Space")
        r.addSeparator()
        self._act(r, "Delete one frame (hard cut)", self.remove_frame, "Alt+Delete")
        self._act(r, "Duplicate frame at playhead", self.duplicate_frame, "Alt+D")
        self._act(r, "Freeze frame (2 s)", lambda: self.freeze_frame(2.0), "F")
        self._act(r, "Save frame as picture…", self.save_frame_png)
        tools = mb.addMenu("&Magic")
        self._act(tools, "Auto captions…", self.auto_captions, "Ctrl+Shift+C")
        self._act(tools, "Remove silences…", self.remove_silences)
        self._act(v, "Command palette", self.open_palette, "Ctrl+K")
        self._act(v, "AI Assistant", lambda: self.toggle_assistant(), "Ctrl+J")
        h = mb.addMenu("&Help")
        self._act(h, "Keyboard shortcuts", self.show_shortcuts, "F1")

    def _fill_recent_menu(self):
        self.recent_menu.clear()
        paths = [p for p in self.recent_projects() if os.path.exists(p)]
        if not paths:
            a = self.recent_menu.addAction("(none yet)")
            a.setEnabled(False)
        for p in paths:
            self.recent_menu.addAction(os.path.basename(p), lambda p=p: self.open_project(p))

    # ---- editor events ---------------------------------------------------
    def on_editor_change(self, reason: str):
        project = self.ed.project
        timeline_id = project.active_timeline_id
        if reason == "load":
            self._timeline_playheads.clear()
            self._visible_timeline_id = None
        switched = timeline_id != self._visible_timeline_id
        if switched:
            self.player.stop()
            if self._visible_timeline_id is not None:
                self._timeline_playheads[self._visible_timeline_id] = self.playhead
            self._visible_timeline_id = timeline_id
            self.playhead = min(self._timeline_playheads.get(timeline_id, 0.0), project.duration)
            self.timeline.canvas.playhead = self.playhead
            self.timeline.canvas.drag = None
            self.timeline.canvas.snap_x = None
            self.timeline.canvas.x_off = 0
            if self.playhead > 0:
                self.timeline.canvas.ensure_visible(self.playhead)
            self._aspect_locked = reason != "load" or bool(self.ed.path)
        if reason in ("load", "timeline", "undo", "redo"):
            self._refresh_timeline_selector()
        self.update_range_bar()
        if reason == "marks":
            self.timeline.canvas.update()
            return
        if reason in ("assets", "load", "undo", "redo"):
            for a in self.ed.project.assets.values():
                self.loader.load(a)
            self.bin.refresh()
        if switched or reason in ("undo", "redo", "settings", "edit"):
            self.preview.aspect = project.width / max(project.height, 1)
        if reason != "live":
            self.inspector.refresh()
        has_content = bool(self.ed.project.assets) or any(t.clips for t in self.ed.project.tracks)
        self.center.setCurrentIndex(1 if has_content else 0)
        if self.playhead > self.ed.project.duration:
            self.playhead = self.ed.project.duration
            self.timeline.canvas.playhead = self.playhead
        self.timeline.refresh()
        self.update_title()
        if switched:
            self.inspector.sync_curve_playhead(self.playhead)
        self.request_render()

    def _on_asset_data(self, _aid):
        self.bin.refresh()
        self.timeline.canvas.update()
        self.request_render()

    def update_title(self):
        name = os.path.basename(self.ed.path) if self.ed.path else "Untitled"
        if hasattr(self, "proj_lbl"):
            pr = self.ed.project
            self.proj_lbl.setText(f"{'● ' if self.ed.dirty else ''}{name}   ·   {pr.width}×{pr.height}  {pr.fps:g} fps")
            idx = self.ratio.findData((pr.width, pr.height))
            if idx >= 0 and self.ratio.currentIndex() != idx:
                self.ratio.setCurrentIndex(idx)
        self.setWindowTitle(f"{'● ' if self.ed.dirty else ''}{name} — ClipEditor")
        d = self.ed.project.duration
        fps = self.ed.project.fps
        self.time_lbl.setText(f"{fmt_time(self.playhead, fps, True)}  /  {fmt_time(d, fps, True)}")

    # ---- playback / preview ---------------------------------------------
    def request_render(self):
        if not self._render_pending:
            self._render_pending = True
            QTimer.singleShot(0, self._render)

    def _render(self):
        self._render_pending = False
        w, h = self.preview.frame_size()
        try:
            time = self.ed.frame_start_at(self.playhead) if self.playhead < self.ed.project.duration else self.playhead
            self.preview.set_image(self.comp.render(self.ed.project, time, w, h))
        except Exception as e:  # noqa: BLE001 - never let a bad frame crash the editor
            self.statusBar().showMessage(f"Preview error: {e}", 5000)

    def seek(self, t: float):
        t = min(max(t, 0.0), max(self.ed.project.duration, 0.0))
        was = self.player.playing
        if was:
            self.player.stop()
        self.playhead = t
        self.timeline.canvas.playhead = t
        self.timeline.canvas.update()
        self.update_title()
        self.request_render()
        self.inspector.sync_time()
        if was:
            self.player.play(t)

    def toggle_play(self):
        self._range_stop = None
        if self.ed.project.duration <= 0:
            self.statusBar().showMessage("Add a clip to the timeline first", 4000)
            return
        if self.player.playing:
            self.player.stop()
            self.btn_play.setText("▶")
        else:
            if self.playhead >= self.ed.project.duration - 1e-3:
                self.playhead = 0.0
            self.player.play(self.playhead)
            self.btn_play.setText("⏸")

    def _on_tick(self, t: float):
        stop = getattr(self, "_range_stop", None)
        if stop is not None and t >= stop:
            self._range_stop = None
            self.player.stop()
            t = stop
        self.playhead = t
        cv = self.timeline.canvas
        cv.playhead = t
        cv.ensure_visible(t)
        cv.update()
        self.update_title()
        self.inspector.sync_curve_playhead(t)
        self._render()

    def zoom_fit(self):
        cv = self.timeline.canvas
        d = max(self.ed.project.duration, 1.0)
        cv.x_off = 0
        cv.set_zoom((cv.width() - 130) / d, 96)

    def _toggle_snap(self, on):
        self.timeline.canvas.snapping = on
        self.statusBar().showMessage("Snapping on" if on else "Snapping off", 2000)

    def select_all(self):
        self.ed.selection = {c.id for t in self.ed.project.tracks for c in t.clips}
        self.ed.notify("selection")

    def add_text(self, preset: str):
        self.ed.add_text(start=self.playhead, **TEXT_PRESETS[preset])
        self.statusBar().showMessage("Text added — type your words in the Text box on the right", 6000)
        self.inspector.text_edit.setFocus()
        self.inspector.text_edit.selectAll()

    def add_transition_near_playhead(self):
        best = None
        for tr in self.ed.project.tracks:
            for b in tr.clips:
                if neighbour_before(tr, b) is None or b.transition > 0:
                    continue
                dist = abs(b.start - self.playhead)
                if best is None or dist < best[0]:
                    best = (dist, b)
        if best is None:
            self.statusBar().showMessage("Put two clips next to each other first — transitions go between them", 6000)
            return
        b = best[1]
        self.ed.selection = {b.id}
        self.ed.set_transition(b.id)
        self.statusBar().showMessage("Transition added — change its style on the right", 5000)

    def _transitions_everywhere(self, kind):
        if kind is None:
            self.ed.checkpoint()
            for tr in self.ed.project.tracks:
                for c in tr.sorted_clips():
                    if c.transition > 0:
                        self.ed.set_transition(c.id, duration=0.0, checkpoint=False)
            self.ed.notify("edit")
            return
        n_before = self.ed.project.duration
        self.ed.transitions_everywhere(kind, DEFAULT_DURATION)
        if self.ed.project.duration == n_before:
            self.statusBar().showMessage("No cuts found — place clips back to back on a video track", 6000)
        else:
            self.statusBar().showMessage(f"{KINDS[kind]} added between all clips", 5000)

    # ---- range & frame tools ---------------------------------------------
    def _build_range_bar(self) -> QWidget:
        bar = QWidget()
        bar.setStyleSheet("background: #1d3352;")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(10, 4, 10, 4)
        self.range_lbl = QLabel()
        self.range_lbl.setStyleSheet("color: #dbe8ff; font-weight: 600;")
        lay.addWidget(self.range_lbl)
        lay.addStretch(1)
        for text, tip, fn in (("▶ Play range", "Shift+Space", self.play_range),
                              ("✂ Remove range", "Cut this part out and close the gap (Shift+Del)", self.remove_range),
                              ("⧉ Duplicate range", "Insert a copy right after it", self.duplicate_range),
                              ("✕", "Clear the range (Alt+X)", self.clear_marks)):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.clicked.connect(fn)
            lay.addWidget(b)
        bar.hide()
        return bar

    def update_range_bar(self):
        rng = self.ed.range()
        if rng is None:
            self.range_bar.hide()
            return
        fps = self.ed.project.fps
        frames = int(round((rng[1] - rng[0]) * fps))
        self.range_lbl.setText(f"Range  {fmt_time(rng[0], fps, True)} → {fmt_time(rng[1], fps, True)}"
                               f"   ·   {rng[1] - rng[0]:.2f}s  ({frames} frames)")
        self.range_bar.show()

    def mark_in(self):
        self.ed.mark_in = self.ed.snap_to_frame(self.playhead)
        if self.ed.mark_out is not None and self.ed.mark_out <= self.ed.mark_in:
            self.ed.mark_out = None
        self.statusBar().showMessage("Range start set — move the playhead and press O for the end", 5000)
        self.ed.notify("marks")

    def mark_out(self):
        self.ed.mark_out = self.ed.snap_to_frame(self.playhead)
        if self.ed.mark_in is None or self.ed.mark_in >= self.ed.mark_out:
            self.ed.mark_in = 0.0 if self.ed.mark_in is None else self.ed.mark_in
        self.ed.notify("marks")

    def clear_marks(self):
        self.ed.mark_in = self.ed.mark_out = None
        self.ed.notify("marks")

    def range_from_selection(self):
        clips = [c for t in self.ed.project.tracks for c in t.clips if c.id in self.ed.selection]
        if clips:
            self.ed.mark_in = min(c.start for c in clips)
            self.ed.mark_out = max(c.end for c in clips)
            self.ed.notify("marks")

    def _need_range(self):
        rng = self.ed.range()
        if rng is None:
            self.statusBar().showMessage(
                "Mark a range first: Shift+drag on the ruler, or press I at the start and O at the end", 7000)
        return rng

    def remove_range(self):
        rng = self._need_range()
        if rng:
            self.ed.remove_range(*rng)
            self.clear_marks()
            self.seek(rng[0])
            self.statusBar().showMessage(f"Removed {rng[1] - rng[0]:.2f}s", 4000)

    def duplicate_range(self):
        rng = self._need_range()
        if rng:
            self.ed.duplicate_range(*rng)
            d = rng[1] - rng[0]
            self.ed.mark_in, self.ed.mark_out = rng[1], rng[1] + d  # highlight the new copy
            self.ed.notify("marks")
            self.statusBar().showMessage(f"Duplicated {d:.2f}s — the copy is highlighted", 4000)

    def play_range(self):
        rng = self._need_range()
        if rng:
            self.seek(rng[0])
            self.player.stop()
            self._range_stop = rng[1]
            self.player.play(rng[0])
            self.btn_play.setText("⏸")

    def smart_delete(self):
        if self.ed.selection:
            self.ed.delete_selected()
        elif self.ed.range():
            self.remove_range()
        else:
            self.statusBar().showMessage("Click a clip (or mark a range) first, then press Delete", 5000)

    def split_at_playhead(self):
        self.ed.split(self.ed.frame_start_at(self.playhead))

    def smart_duplicate(self):
        if self.ed.selection:
            self.ed.duplicate_selected()
        elif self.ed.range():
            self.duplicate_range()
        else:
            self.statusBar().showMessage("Click a clip (or mark a range) first, then press Ctrl+D", 5000)

    def duplicate_frame(self):
        a = self.ed.snap_to_frame(self.playhead)
        self.ed.duplicate_range(a, a + self.ed.frame())

    def remove_frame(self):
        self.player.stop()
        try:
            time = self.ed.frame_start_at(self.playhead)
            removed = self.ed.remove_frame(self.playhead)
        except ValueError as error:
            self.status(str(error), 7000)
            return
        if removed:
            self.clear_marks()
            self.seek(time)
            self.status("Deleted one frame and closed the gap.")
        else:
            self.status("Move the playhead onto the frame you want to delete.")

    def _grab_frame(self, t: float):
        pr = self.ed.project
        k = min(1.0, 1920 / max(pr.width, pr.height))
        return self.comp.render(pr, t, int(pr.width * k) & ~1, int(pr.height * k) & ~1)

    def freeze_frame(self, seconds: float = 2.0):
        if self.ed.project.duration <= 0:
            return
        folder = os.path.join(app_data_dir(), "frames")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"freeze_{int(time.time() * 1000)}.png")
        self._grab_frame(self.playhead).save(path)
        self.ed.insert_still(path, self.playhead, seconds)
        self.statusBar().showMessage(f"Frame frozen for {seconds:g}s — drag its right edge to change the length", 6000)

    def save_frame_png(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save frame", "frame.png", "PNG image (*.png);;JPEG (*.jpg)")
        if path:
            self._grab_frame(self.playhead).save(path)
            self.statusBar().showMessage(f"Saved {path}", 4000)

    def copy(self):
        n = self.ed.copy_selected()
        self.statusBar().showMessage(f"Copied {n} clip(s) — Ctrl+V pastes at the playhead" if n
                                     else "Select clips to copy", 4000)

    def cut(self):
        if self.ed.copy_selected():
            self.ed.delete_selected()

    def paste(self):
        n = self.ed.paste(self.playhead)
        if not n:
            self.statusBar().showMessage("Nothing to paste — copy a clip first (Ctrl+C)", 4000)

    # ---- magic tools -----------------------------------------------------
    def remove_silences(self):
        from ..silence import find_silences

        if self.ed.project.duration <= 0:
            return
        levels = {"Only very quiet parts (-50 dB)": -50.0, "Normal (-40 dB)": -40.0,
                  "Aggressive (-32 dB)": -32.0}
        from PySide6.QtWidgets import QInputDialog

        choice, ok = QInputDialog.getItem(self, "Remove silences",
                                          "Cut out pauses longer than 0.6 s.\nHow quiet counts as silence?",
                                          list(levels), 1, False)
        if not ok:
            return
        missing = [a for a in self.ed.project.timeline_assets() if a.has_audio and self.audio.get(a.id) is None]
        for a in missing:  # make sure audio is decoded
            self.audio.load(a)
        sil = find_silences(self.ed.project, self.audio, threshold_db=levels[choice])
        if not sil:
            QMessageBox.information(self, "Remove silences", "No silent pauses found. 🎉")
            return
        total = sum(b - a for a, b in sil)
        r = QMessageBox.question(self, "Remove silences",
                                 f"Found {len(sil)} pause(s), {total:.1f}s in total.\nRemove them?")
        if r == QMessageBox.StandardButton.Yes:
            self.ed.remove_ranges(sil)
            self.statusBar().showMessage(f"Removed {len(sil)} pauses ({total:.1f}s). Ctrl+Z to undo.", 8000)

    def auto_captions(self):
        from .captions_dialog import run_auto_captions

        run_auto_captions(self)

    # ---- sidebar / inspector actions --------------------------------------
    def status(self, msg: str, ms: int = 6000):
        self.statusBar().showMessage(msg, ms)

    def _need_selection(self, what="this") -> bool:
        if not self.ed.selection:
            self.status(f"Select a clip on the timeline first, then apply {what}.")
            return False
        return True

    def add_file_at_playhead(self, path: str, audio_track: bool = False):
        a = self.import_paths([path])
        if a:
            self.ed.add_clip(a[0].id, None, self.playhead)
            self.status(f"Added {a[0].name} at {fmt_time(self.playhead)}")

    def add_text_style(self, style: dict):
        from ..model import Clip

        fields = {k: v for k, v in style.items() if k in Clip.__dataclass_fields__}
        text = fields.pop("text", "Your text")
        c = self.ed.add_text(text, start=self.playhead, **fields)
        self.status("Text added — type your words in the Text tab on the right")
        self.inspector.text_edit.setFocus()
        self.inspector.text_edit.selectAll()
        return c

    def add_shape_preset(self, name: str):
        from .. import shapes

        f = shapes.new_shape_clip_fields(name) if hasattr(shapes, "new_shape_clip_fields") else {
            "shape": dict(shapes.PRESETS[name]), "out_point": 4.0}
        dur = f.get("out_point", 4.0) - f.get("in_point", 0.0)
        self.ed.add_shape(f["shape"], start=self.playhead, duration=dur)
        self.status(f"{name} added — move, size and animate it on the right")

    def add_effect(self, name: str):
        if not self._need_selection("an effect"):
            return
        from .. import effects

        n = self.ed.update_selected(lambda c: c.effects.append(
            effects.new_effect(name) if hasattr(effects, "new_effect") else {"type": name, "params": {}, "enabled": True}))
        self.status(f"{effects.REGISTRY.get(name, {}).get('label', name)} added to {n} clip(s). "
                    "Tune it in Adjust on the right.")

    def set_filter(self, name):
        """One filter (look) per clip, CapCut-style: replaces the previous filter."""
        if not self._need_selection("a filter"):
            return
        from .. import effects

        filters = set(effects.categories().get("Filters", [])) if hasattr(effects, "categories") else set()

        def fn(c):
            c.effects = [e for e in c.effects if e.get("type") not in filters]
            if name:
                c.effects.insert(0, effects.new_effect(name))
        self.ed.update_selected(fn)
        self.status("Filter removed" if not name else f"Filter: {effects.REGISTRY.get(name, {}).get('label', name)}")

    def apply_transition(self, kind: str, duration: float):
        if len(self.ed.selection) == 1:
            cid = next(iter(self.ed.selection))
            if self.ed.set_transition(cid, kind, duration):
                self.status(f"{KINDS[kind]} added")
                return
        self.add_transition_near_playhead()
        if len(self.ed.selection) == 1:
            self.ed.set_transition(next(iter(self.ed.selection)), kind, duration)

    def transitions_everywhere(self, kind: str, duration: float):
        self.ed.transitions_everywhere(kind, duration)
        self.status(f"{KINDS[kind]} added between all clips")

    def set_animation(self, kind: str, name: str, dur: float):
        if not self._need_selection("an animation"):
            return

        def fn(c):
            setattr(c, f"anim_{kind}", name)
            if kind == "loop":
                c.anim_loop_speed = max(dur, 0.1)
            else:
                setattr(c, f"anim_{kind}_dur", min(dur, c.duration))
        self.ed.update_selected(fn)
        if name:
            # preview the animation right away
            c = self.ed.selected_clips()[0]
            self.seek(c.start if kind != "out" else max(c.end - dur - 0.2, c.start))
            if not self.player.playing:
                self.toggle_play()

    def _toggle_safe(self, on):
        self.preview.safe_zones = on
        self.preview.update()

    # ---- voice ---------------------------------------------------------
    def record_voiceover(self):
        from .. import recorder

        if getattr(self, "_rec", None) is not None:
            path = self._rec.stop()
            self._rec = None
            self._rec_timer.stop()
            if self._rec_timeline_id not in self.ed.project.timelines:
                self.status(f"Voiceover saved to {path}", 10000)
                return
            a = self.import_paths([path])
            if a:
                a[0].origin = "voiceover"
                if self.ed.project.active_timeline_id == self._rec_timeline_id:
                    self.ed.add_clip(a[0].id, None, self._rec_start)
                    self.status("Voiceover added ✓")
                else:
                    self.status("Voiceover saved in the media bin. Open the recording's timeline to use it.")
            return
        try:
            self._rec = recorder.Recorder()
            self._rec.start()
        except Exception as e:  # noqa: BLE001
            self._rec = None
            QMessageBox.warning(self, "Can't record", str(e))
            return
        self._rec_start = self.playhead
        self._rec_timeline_id = self.ed.project.active_timeline_id
        if not self.player.playing and self.ed.project.duration > 0:
            self.toggle_play()  # record while watching
        self._rec_timer = QTimer(self)
        self._rec_timer.timeout.connect(lambda: self.status(
            f"● Recording {self._rec.elapsed():.1f}s   level {'▮' * int(self._rec.level() * 20):<20}   "
            "— click Record voiceover again to stop", 500))
        self._rec_timer.start(100)

    def text_to_speech(self):
        from PySide6.QtWidgets import QInputDialog

        from .. import tts

        text, ok = QInputDialog.getMultiLineText(self, "Text to speech", "Type the script to read out:")
        if not ok or not text.strip():
            return
        vs = tts.voices()
        voice = None
        if len(vs) > 1:
            names = [f"{v['name']} ({v.get('engine', '')})" for v in vs]
            pick, ok = QInputDialog.getItem(self, "Text to speech", "Voice", names, 0, False)
            if not ok:
                return
            voice = vs[names.index(pick)]["id"]
        self.status("Speaking…")
        QApplication.processEvents()
        try:
            path = tts.synthesize(text, voice)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Text to speech", str(e))
            return
        a = self.import_paths([path])
        if a:
            a[0].origin = "tts"
            self.ed.add_clip(a[0].id, None, self.playhead)
            self.status("Voiceover from text added ✓")

    # ---- AI tools --------------------------------------------------------
    def run_ai_tool(self, key: str):
        if key == "assistant":
            return self.toggle_assistant(True)
        if key == "captions":
            return self.auto_captions()
        if key == "silences":
            return self.remove_silences()
        if key == "tts":
            return self.text_to_speech()
        if key == "ai_denoise":
            if not self._need_selection("AI voice cleanup"):
                return
            self.ed.update_selected(lambda c: c.audio_fx.append({"type": "ai_denoise", "params": {"intensity": 1.0}})
                                    if not any(f.get("type") == "ai_denoise" for f in c.audio_fx) else None)
            self.status("AI noise removal on — processing in the background")
            return
        # video jobs on the selected clip -> new file -> swap
        clips = [c for c in self.ed.selected_clips() if not c.is_generated]
        if len(clips) != 1:
            self.status("Select one video clip on the timeline first.")
            return
        clip = clips[0]
        asset = self.ed.project.assets.get(clip.asset_id)
        try:
            if key == "bgremove":
                from ..ai import bgremove

                fn = bgremove.remove_background
                models = getattr(bgremove, "MODELS", {})
                kwargs = {}
                if models:
                    from PySide6.QtWidgets import QInputDialog

                    labels = [f"{k} — {v}" for k, v in models.items()]
                    pick, ok = QInputDialog.getItem(self, "Remove background", "Model", labels, 0, False)
                    if not ok:
                        return
                    kwargs["model"] = list(models)[labels.index(pick)]
                secs = clip.out_point - clip.in_point
                if secs > 20 and asset.kind != "image":
                    r = QMessageBox.question(self, "Remove background",
                                             f"This clip is {secs:.0f}s long. Background removal takes about "
                                             f"{secs * 8 / 60:.0f} minute(s) on this computer. Continue?")
                    if r != QMessageBox.StandardButton.Yes:
                        return
                label = "Removing background"
            else:
                from ..ai import video_tools

                fn = getattr(video_tools, key)
                kwargs = {}
                if key == "auto_reframe":
                    kwargs["target_aspect"] = "9:16"
                label = {"stabilize": "Stabilizing", "enhance": "Enhancing", "reverse": "Reversing",
                         "smooth_slowmo": "Creating smooth slow motion", "auto_reframe": "Reframing"}.get(key, key)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "AI tool unavailable", f"{key}: {e}")
            return
        self._run_job(label, fn, clip, asset, kwargs, key)

    def _run_job(self, label, fn, clip, asset, kwargs, key):
        timeline_id = self.ed.project.active_timeline_id
        prog = QProgressDialog(f"{label}…", "Cancel", 0, 1000, self)
        prog.setWindowTitle("AI Magic")
        prog.setWindowModality(Qt.WindowModality.WindowModal)
        prog.setMinimumDuration(0)
        prog.setMinimumWidth(420)
        sig = _ExportSignals()
        status_sig = _JobStatus()
        cancel = threading.Event()
        prog.canceled.connect(cancel.set)
        sig.progress.connect(lambda v: prog.setValue(int(v * 1000)))
        status_sig.text.connect(prog.setLabelText)
        result = {}

        def finished(err):
            prog.reset()
            if err:
                if not cancel.is_set():
                    QMessageBox.warning(self, "AI Magic failed", err[-2000:])
                return
            from ..media import probe

            try:
                new = probe(result["path"])
            except Exception as e:  # noqa: BLE001
                QMessageBox.warning(self, "AI Magic", f"Result could not be opened: {e}")
                return
            new.origin = f"{key}:{asset.id}"
            if timeline_id not in self.ed.project.timelines:
                self.status(f"{label} done. Output saved to {new.path}", 10000)
                return
            if self.ed.project.find_clip(clip.id)[1] is None:
                self.ed.add_asset(new)
                self.status(f"{label} done. The result is in the media bin.")
                return
            self.ed.swap_asset(clip.id, new, keep_timing=key != "smooth_slowmo")
            self.status(f"{label} done ✓ — Ctrl+Z to go back to the original")

        sig.done.connect(finished)

        def work():
            try:
                result["path"] = fn(asset.path, clip.in_point, clip.out_point, progress=sig.progress.emit,
                                    cancelled=cancel.is_set, status=status_sig.text.emit, **kwargs)
                sig.done.emit("" if not cancel.is_set() else "cancelled")
            except Exception as e:  # noqa: BLE001
                sig.done.emit(f"{type(e).__name__}: {e}")

        self._job_sigs = (sig, status_sig)
        threading.Thread(target=work, daemon=True).start()

    # ---- palette / assistant ----------------------------------------------
    def palette_commands(self):
        cmds = []
        for a in self.actions():
            if a.text():
                cmds.append((a.text().replace("&&", "&"), a.shortcut().toString(), a.trigger))
        try:
            from .. import effects

            for n, spec in effects.REGISTRY.items():
                kind = "Filter" if spec.get("category") == "Filters" else "Effect"
                fn = (lambda n=n: self.set_filter(n)) if kind == "Filter" else (lambda n=n: self.add_effect(n))
                cmds.append((f"{kind}: {spec.get('label', n)}", spec.get("category", ""), fn))
        except Exception:  # noqa: BLE001
            pass
        from .. import animate

        for kind, items in animate.ANIMATIONS.items():
            for n, label in items.items():
                cmds.append((f"Animation {kind}: {label}", "selected clip",
                             lambda k=kind, n=n: self.set_animation(k, n, 0.6)))
        for k, label in KINDS.items():
            cmds.append((f"Transition: {label}", "nearest cut", lambda k=k: self.apply_transition(k, 0.5)))
        try:
            from .. import shapes

            for n in shapes.PRESETS:
                cmds.append((f"Sticker: {n}", "add at playhead", lambda n=n: self.add_shape_preset(n)))
        except Exception:  # noqa: BLE001
            pass
        try:
            from .. import sfx

            for n, info in sfx.LIBRARY.items():
                cmds.append((f"Sound: {info.get('label', n)}", "add at playhead",
                             lambda n=n: (sfx.ensure_library(), self.add_file_at_playhead(sfx.path(n), True))))
        except Exception:  # noqa: BLE001
            pass
        from .sidebar import AI_TOOLS, TEXT_STYLES

        for key, _i, label, desc in AI_TOOLS:
            cmds.append((f"AI: {label}", desc, lambda k=key: self.run_ai_tool(k)))
        for n, st in TEXT_STYLES.items():
            cmds.append((f"Text: {n}", "add at playhead", lambda st=st: self.add_text_style(st)))
        for label, (w, h) in ((self.ratio.itemText(i), self.ratio.itemData(i)) for i in range(self.ratio.count())):
            cmds.append((f"Canvas {label}", f"{w}×{h}", lambda w=w, h=h: self.set_canvas(w, h)))
        return cmds

    def open_palette(self):
        from .palette import CommandPalette

        pal = CommandPalette(self, self.palette_commands())
        g = self.geometry()
        pal.move(g.x() + (g.width() - 560) // 2, g.y() + 70)
        pal.edit.setFocus()
        pal.exec()

    def toggle_assistant(self, show=None):
        if self.assistant_dock is None:
            try:
                from .assistant_panel import AssistantPanel

                panel = AssistantPanel(self)
            except Exception as e:  # noqa: BLE001
                QMessageBox.warning(self, "AI Assistant", f"Assistant unavailable: {e}")
                return
            self.assistant_dock = QDockWidget("AI Assistant", self)
            self.assistant_dock.setWidget(panel)
            self.assistant_dock.setMinimumWidth(340)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.assistant_dock)
            return
        vis = not self.assistant_dock.isVisible() if show is None else bool(show)
        self.assistant_dock.setVisible(vis)

    def show_shortcuts(self):
        ShortcutsDialog(self).exec()

    # ---- canvas ----------------------------------------------------------
    def _choose_aspect(self, w, h):
        self._aspect_locked = (w, h) != (1920, 1080)
        self.set_canvas(w, h, quiet=True)

    def set_canvas(self, w, h, quiet=False):
        pr = self.ed.project
        if (pr.width, pr.height) == (w, h):
            return
        if not quiet:
            self.ed.checkpoint()
        self._aspect_locked = True
        pr.width, pr.height = w, h
        self.welcome.select_aspect(w, h)
        self.ed.notify("settings")

    # ---- files -----------------------------------------------------------
    def import_paths(self, paths) -> list:
        added = []
        empty = self.ed.project.duration == 0
        for p in paths:
            try:
                a = self.ed.import_file(p)
                from .. import proxy

                proxy.ensure(a, lambda _aid: self.status("Proxy ready — 4K preview is now smooth", 4000))
                added.append(a)
                pr = self.ed.project
                if empty and a.kind == "video" and a.width:
                    pr.fps = round(a.fps, 3) or 30.0
                    if not self._aspect_locked:
                        pr.width, pr.height = a.width - a.width % 2, a.height - a.height % 2
                        self.preview.aspect = pr.width / pr.height
                    empty = False
            except Exception as e:  # noqa: BLE001
                QMessageBox.warning(self, "Couldn't import", f"{os.path.basename(p)} could not be opened.\n\n{e}")
        if added:
            self.statusBar().showMessage(
                f"Imported {len(added)} file(s) — drag them onto the timeline below", 6000)
        return added

    def import_dialog(self):
        last = self.settings.value("last_media_dir", "")
        paths, _ = QFileDialog.getOpenFileNames(self, "Import media", last, MEDIA_FILTER)
        if paths:
            self.settings.setValue("last_media_dir", os.path.dirname(paths[0]))
            added = self.import_paths(paths)
            # first import into an empty timeline: put it straight on the timeline
            if added and self.ed.project.duration == 0:
                for a in added:
                    self.ed.add_clip(a.id)
                self.zoom_fit()

    def _files_dropped_on_timeline(self, paths, track_id, t):
        for a in self.import_paths(paths):
            clip = self.ed.add_clip(a.id, track_id, t)
            if clip is not None:
                t = clip.end
        self.zoom_fit() if self.timeline.canvas.pps * self.ed.project.duration < 200 else None

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        added = self.import_paths([u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()])
        if added and self.ed.project.duration == 0:
            for a in added:
                self.ed.add_clip(a.id)
            self.zoom_fit()

    def recent_projects(self) -> list[str]:
        v = self.settings.value("recent", [])
        return [v] if isinstance(v, str) else list(v or [])

    def _remember(self, path):
        path = os.path.abspath(path)
        rec = [p for p in self.recent_projects() if p != path]
        self.settings.setValue("recent", [path] + rec[:9])
        self.welcome.set_recent(self.recent_projects())

    def _confirm_discard(self) -> bool:
        if not self.ed.dirty:
            return True
        r = QMessageBox.question(self, "Unsaved changes", "Save your changes first?",
                                 QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard |
                                 QMessageBox.StandardButton.Cancel)
        if r == QMessageBox.StandardButton.Save:
            return self.save_project()
        return r == QMessageBox.StandardButton.Discard

    def new_project(self):
        if self._confirm_discard():
            self.player.stop()
            self._aspect_locked = False
            self.ed.new_project()
            self.welcome.select_aspect(1920, 1080)
            self.seek(0)

    def open_project(self, path: str | None = None):
        if not self._confirm_discard():
            return
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Open project", "", PROJECT_FILTER)
        if not path:
            return
        self.player.stop()
        try:
            self.ed.open(path)
            self._remember(path)
            missing = self.ed.missing_media()
            if missing:
                answer = QMessageBox.question(self, "Missing media",
                                              f"{len(missing)} media file(s) are missing. Locate them now?")
                if answer == QMessageBox.StandardButton.Yes:
                    self.relink_dialog()
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Couldn't open project", str(e))
        self._aspect_locked = True
        self.seek(0)
        QTimer.singleShot(50, self.zoom_fit)

    def relink_dialog(self):
        missing = self.ed.missing_media()
        if not missing:
            self.statusBar().showMessage("All media files are available.", 4000)
            return
        folder = QFileDialog.getExistingDirectory(self, "Choose the folder containing missing media")
        if not folder:
            return
        count, unresolved = self.ed.relink_media(folder)
        if unresolved:
            names = "\n".join(unresolved[:12])
            more = f"\n…and {len(unresolved) - 12} more" if len(unresolved) > 12 else ""
            QMessageBox.information(self, "Locate missing media",
                                    f"Relinked {count} file(s). Still missing or unmatched:\n\n{names}{more}")
        else:
            self.statusBar().showMessage(f"Relinked {count} file(s).", 4000)

    def save_project(self, save_as: bool = False) -> bool:
        path = self.ed.path
        if save_as or not path:
            path, _ = QFileDialog.getSaveFileName(self, "Save project", "My video.clipeditor.json", PROJECT_FILTER)
            if not path:
                return False
        try:
            self.ed.save(path)
        except (OSError, ValueError, TypeError) as error:
            QMessageBox.warning(self, "Couldn't save project", str(error))
            return False
        self._remember(path)
        self._clear_autosave()
        self.statusBar().showMessage(f"Saved ✓  {path}", 4000)
        return True

    # ---- autosave / recovery --------------------------------------------
    def autosave(self):
        if self.ed.dirty:
            try:
                self.ed.project.save(self.autosave_path)
                self.statusBar().showMessage(f"Autosaved {time.strftime('%H:%M')}", 3000)
            except OSError:
                pass

    def _clear_autosave(self):
        try:
            os.remove(self.autosave_path)
        except OSError:
            pass

    def offer_recovery(self):
        if not os.path.exists(self.autosave_path):
            return
        r = QMessageBox.question(self, "Recover work?",
                                 "ClipEditor closed before your last changes were saved.\n"
                                 "Do you want to recover the autosaved project?")
        if r == QMessageBox.StandardButton.Yes:
            try:
                self.ed.open(self.autosave_path)
                self.ed.path = None
                self.ed.dirty = True
                self.on_editor_change("load")
                self._aspect_locked = True
                QTimer.singleShot(50, self.zoom_fit)
            except Exception as e:  # noqa: BLE001
                QMessageBox.warning(self, "Recovery failed", str(e))
        self._clear_autosave()

    # ---- export ----------------------------------------------------------
    def export_dialog(self):
        if self.ed.project.duration <= 0:
            QMessageBox.information(self, "Nothing to export", "Add some clips to the timeline first.")
            return
        self.player.stop()
        base = os.path.splitext(os.path.splitext(os.path.basename(self.ed.path or "My video"))[0])[0]
        if len(self.ed.project.timelines) > 1:
            name = self.ed.project.active_timeline.name
            safe_name = "".join(ch if ch.isalnum() or ch in " -_" else "_" for ch in name[:64]).strip()
            base += f" - {safe_name or 'Timeline'}"
        folder = self.settings.value("last_export_dir", "") or QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.MoviesLocation)
        dlg = ExportDialog(self, self.ed.project, os.path.join(folder, base + ".mp4"))
        if dlg.exec() != ExportDialog.DialogCode.Accepted:
            return
        path, opts = dlg.out_path(), dlg.options()
        self.settings.setValue("last_export_dir", os.path.dirname(path))
        snapshot = Project.from_dict(self.ed.project.to_dict())
        prog = QProgressDialog("Exporting your video…", "Cancel", 0, 1000, self)
        prog.setWindowTitle("Export")
        prog.setWindowModality(Qt.WindowModality.WindowModal)
        prog.setMinimumDuration(0)
        prog.setMinimumWidth(380)
        sig = _ExportSignals()
        cancel = threading.Event()
        prog.canceled.connect(cancel.set)
        t0 = time.time()

        def on_progress(v):
            prog.setValue(int(v * 1000))
            el = time.time() - t0
            if v > 0.02:
                prog.setLabelText(f"Exporting your video…  {v:.0%}\nabout {max(el / v - el, 0):.0f}s left")

        sig.progress.connect(on_progress)

        def finished(err: str):
            prog.reset()
            if err:
                if not cancel.is_set():
                    QMessageBox.warning(self, "Export failed", err[-3000:])
                return
            self.statusBar().showMessage(f"Exported ✓  {path}", 10000)
            box = QMessageBox(self)
            box.setWindowTitle("Done!")
            box.setText(f"Your video is ready 🎉\n\n{path}")
            play = box.addButton("Play video", QMessageBox.ButtonRole.AcceptRole)
            folder_btn = box.addButton("Show in folder", QMessageBox.ButtonRole.ActionRole)
            box.addButton("Close", QMessageBox.ButtonRole.RejectRole)
            box.exec()
            if box.clickedButton() is play:
                os.startfile(path) if os.name == "nt" else None
            elif box.clickedButton() is folder_btn and os.name == "nt":
                import subprocess

                subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])

        sig.done.connect(finished)

        def run():
            try:
                from ..export_render import export_video

                export_video(snapshot, path, progress=sig.progress.emit, cancelled=cancel.is_set, **opts)
                sig.done.emit("")
            except Exception as e:  # noqa: BLE001
                sig.done.emit(str(e) or repr(e))

        self._export_sig = sig  # keep alive
        threading.Thread(target=run, daemon=True).start()

    def closeEvent(self, e):
        if not self._confirm_discard():
            e.ignore()
            return
        self.player.stop()
        self.loader.shutdown()
        self.comp.close()
        self._clear_autosave()
        e.accept()


STYLE = """
QWidget { font-size: 9.5pt; }
QToolBar { background: #2a2c31; border: none; padding: 4px; spacing: 4px; }
QToolBar QToolButton { padding: 6px 10px; border-radius: 6px; color: #e4e5ea; }
QToolBar QToolButton:hover { background: #3a3d44; }
QToolBar QToolButton:checked { background: #23395d; }
QPushButton { background: #3a3d44; border: 1px solid #474a52; border-radius: 6px; padding: 5px 12px; color: #e4e5ea; }
QPushButton:hover { background: #454952; }
QPushButton:pressed { background: #2f3238; }
QPushButton:disabled { color: #777; }
QPushButton#primary { background: #0568fd; border-color: #0568fd; color: white; font-weight: 600; padding: 7px 16px; }
QPushButton#primary:hover { background: #2a80ff; }
QPushButton#play { font-size: 13pt; min-width: 56px; }
QSlider::groove:horizontal { height: 4px; background: #474a52; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #0568fd; border-radius: 2px; }
QSlider::handle:horizontal { background: #f0f0f4; width: 14px; margin: -6px 0; border-radius: 7px; }
QScrollArea { border: none; }
QSplitter::handle { background: #1b1c1f; }
QListWidget { border: none; background: #1f2024; }
QStatusBar { color: #a8adb7; }
QToolTip { color: #eee; background: #2b2d32; border: 1px solid #555; padding: 4px; }
"""


def apply_dark_theme(app: QApplication):
    app.setStyle("Fusion")
    pal = QPalette()
    base = QColor(36, 37, 41)
    pal.setColor(QPalette.ColorRole.Window, base)
    pal.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 225))
    pal.setColor(QPalette.ColorRole.Base, QColor(28, 29, 32))
    pal.setColor(QPalette.ColorRole.AlternateBase, base)
    pal.setColor(QPalette.ColorRole.Text, QColor(220, 220, 225))
    pal.setColor(QPalette.ColorRole.Button, QColor(50, 52, 58))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 225))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(5, 104, 253))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    pal.setColor(QPalette.ColorRole.ToolTipBase, base)
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(220, 220, 225))
    app.setPalette(pal)
    app.setStyleSheet(STYLE)
