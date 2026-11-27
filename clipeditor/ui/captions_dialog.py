"""UI for AI auto-captions: options dialog + background transcription with progress."""
from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QMessageBox,
                               QProgressDialog, QVBoxLayout)

LANGUAGES = [("Detect automatically", None), ("English", "en"), ("Tamil", "ta"), ("Hindi", "hi"),
             ("Telugu", "te"), ("Malayalam", "ml"), ("Kannada", "kn"), ("Spanish", "es"), ("French", "fr"),
             ("German", "de"), ("Arabic", "ar"), ("Japanese", "ja"), ("Chinese", "zh")]
STYLES = [("Subtitles (bottom, with box)", "subtitle"), ("Big bold words (TikTok style)", "big")]


class _Signals(QObject):
    progress = Signal(float)
    status = Signal(str)
    done = Signal(object, str)  # lines | None, error


class CaptionsDialog(QDialog):
    def __init__(self, parent, models: dict):
        super().__init__(parent)
        self.setWindowTitle("Auto captions")
        self.setMinimumWidth(460)
        lay = QVBoxLayout(self)
        head = QLabel("Turn speech into subtitles")
        head.setStyleSheet("font-size: 14pt; font-weight: 700;")
        lay.addWidget(head)
        info = QLabel("Runs 100% on your computer with the Whisper speech model.\n"
                      "The model downloads once the first time you use it.")
        info.setStyleSheet("color: #9aa0aa;")
        lay.addWidget(info)
        form = QFormLayout()
        self.model = QComboBox()
        for key, label in models.items():
            self.model.addItem(label, key)
        idx = self.model.findData("base")
        self.model.setCurrentIndex(max(idx, 0))
        form.addRow("Accuracy", self.model)
        self.lang = QComboBox()
        for label, code in LANGUAGES:
            self.lang.addItem(label, code)
        form.addRow("Language", self.lang)
        self.style = QComboBox()
        for label, key in STYLES:
            self.style.addItem(label, key)
        form.addRow("Look", self.style)
        lay.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        go = bb.addButton("Create captions", QDialogButtonBox.ButtonRole.AcceptRole)
        go.setObjectName("primary")
        go.setDefault(True)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)


def run_auto_captions(win):
    try:
        from ..ai import captions
    except Exception as e:  # noqa: BLE001
        QMessageBox.warning(win, "Auto captions", f"The captions module could not be loaded:\n{e}")
        return
    if not captions.is_available():
        QMessageBox.information(
            win, "Auto captions",
            "Auto captions need the free Whisper speech model package.\n\n"
            "Install it once by running this in a terminal in the ClipEditor folder:\n\n"
            "    .venv\\Scripts\\pip install faster-whisper")
        return
    ed = win.ed
    if ed.project.duration <= 0:
        QMessageBox.information(win, "Auto captions", "Add a video or audio clip with speech first.")
        return
    dlg = CaptionsDialog(win, captions.MODELS)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return
    model, lang, style = dlg.model.currentData(), dlg.lang.currentData(), dlg.style.currentData()
    win.player.stop()

    # make sure every clip's audio is decoded before mixing
    for a in ed.project.timeline_assets():
        if a.has_audio and win.audio.get(a.id) is None:
            win.audio.load(a)
    from ..model import Project

    snapshot = Project.from_dict(ed.project.to_dict())

    prog = QProgressDialog("Getting ready…", "Cancel", 0, 1000, win)
    prog.setWindowTitle("Auto captions")
    prog.setWindowModality(Qt.WindowModality.WindowModal)
    prog.setMinimumDuration(0)
    prog.setMinimumWidth(400)
    prog.setValue(0)
    sig = _Signals()
    cancel = threading.Event()
    prog.canceled.connect(cancel.set)
    sig.progress.connect(lambda v: prog.setValue(int(v * 1000)))
    sig.status.connect(prog.setLabelText)

    def finished(lines, err):
        prog.reset()
        if err:
            if not cancel.is_set():
                QMessageBox.warning(win, "Auto captions failed", err[-2000:])
            return
        if not lines:
            QMessageBox.information(win, "Auto captions", "No speech was found in this video.")
            return
        if ed.project.active_timeline_id != snapshot.active_timeline_id:
            QMessageBox.information(win, "Auto captions", "The active timeline changed. Run captions on that timeline again.")
            return
        n = captions.add_captions(ed, lines, style=style)
        win.statusBar().showMessage(
            f"Added {n} captions on a new 'Captions' track. Click any caption to fix its words.", 10000)

    sig.done.connect(finished)

    def work():
        try:
            sig.status.emit("Collecting the audio…")
            audio = captions.timeline_audio(snapshot, win.audio)
            segs = captions.transcribe(audio, model_size=model, language=lang,
                                       progress=sig.progress.emit, cancelled=cancel.is_set,
                                       status=sig.status.emit)
            if cancel.is_set():
                sig.done.emit(None, "cancelled")
                return
            sig.done.emit(captions.split_for_screen(segs), "")
        except Exception as e:  # noqa: BLE001
            sig.done.emit(None, f"{type(e).__name__}: {e}")

    win._captions_sig = sig  # keep alive
    threading.Thread(target=work, daemon=True).start()
