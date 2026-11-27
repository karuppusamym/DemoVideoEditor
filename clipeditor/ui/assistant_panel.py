"""AI Assistant dock: chat to edit. Uses Claude when an API key is available, otherwise the
offline LocalAssistant. Tool calls run through clipeditor.api.EditorAPI; every mutation is
marshalled onto the GUI thread (so undo, listeners and repaint stay single-threaded)."""
from __future__ import annotations

import html
import json
import threading

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextBrowser, QVBoxLayout,
                               QWidget)

SUGGESTIONS = ["Remove the silent pauses", "Add captions", "Make it vertical for TikTok",
               "Add dissolve transitions between clips", "Add a title “My Trip” at the start",
               "What's in my project?"]


class _Invoker(QObject):
    """Run a callable on the GUI thread and wait for its result (EditorAPI `invoke`)."""
    run = Signal(object)

    def __init__(self):
        super().__init__()
        self.run.connect(self._do, Qt.ConnectionType.BlockingQueuedConnection)

    def _do(self, box):
        fn, out = box
        try:
            out["r"] = fn()
        except Exception as e:  # noqa: BLE001
            out["e"] = e

    def __call__(self, fn):
        if QThread.currentThread() is self.thread():
            return fn()
        out: dict = {}
        self.run.emit((fn, out))
        if "e" in out:
            raise out["e"]
        return out.get("r")


class _Events(QObject):
    event = Signal(dict)


class AssistantPanel(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.invoker = _Invoker()
        from ..ai.assistant import has_api_key, make_assistant
        from ..api import EditorAPI

        self.api = EditorAPI(win.ed, win.audio, invoke=self.invoker)
        self._make = make_assistant
        self._has_key = has_api_key
        self.session_key: str | None = None
        self.assistant = make_assistant(self.api)
        self.events = _Events()
        self.events.event.connect(self._on_event)
        self.busy = False
        self._streaming = ""

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        self.mode = QLabel()
        self.mode.setWordWrap(True)
        self.mode.setStyleSheet("color: #9aa0aa;")
        lay.addWidget(self.mode)
        key_row = QHBoxLayout()
        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.EchoMode.Password)
        self.key.setPlaceholderText("Optional: Anthropic API key (kept in memory only)")
        use = QPushButton("Use")
        use.clicked.connect(self._set_key)
        key_row.addWidget(self.key, 1)
        key_row.addWidget(use)
        lay.addLayout(key_row)
        self.log = QTextBrowser()
        self.log.setOpenExternalLinks(False)
        lay.addWidget(self.log, 1)
        chips = QWidget()
        cl = QVBoxLayout(chips)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(3)
        for s in SUGGESTIONS:
            b = QPushButton(s)
            b.setStyleSheet("text-align: left; padding: 4px 8px; color: #cfd8ff;")
            b.clicked.connect(lambda _=False, t=s: self.send(t))
            cl.addWidget(b)
        lay.addWidget(chips)
        row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Tell me what to do with your video…")
        self.input.returnPressed.connect(lambda: self.send(self.input.text()))
        self.send_btn = QPushButton("Send")
        self.send_btn.setObjectName("primary")
        self.send_btn.clicked.connect(lambda: self.send(self.input.text()))
        row.addWidget(self.input, 1)
        row.addWidget(self.send_btn)
        lay.addLayout(row)
        self._update_mode()
        self._say("assistant", "Hi! I can edit your video for you. Try one of the suggestions below, "
                               "or type something like “cut the pauses and add captions”.")

    # ---- setup -----------------------------------------------------------
    def _update_mode(self):
        online = getattr(self.assistant, "name", "") == "Claude"
        self.mode.setText("🟢 Claude is connected — ask for anything." if online else
                          "⚪ Offline helper (understands common commands). Add an Anthropic API key "
                          "or set ANTHROPIC_API_KEY to chat freely with Claude.")
        self.key.setVisible(not online)

    def _set_key(self):
        k = self.key.text().strip()
        self.key.clear()
        if not k:
            return
        self.session_key = k
        self.assistant = self._make(self.api, api_key=k)
        self._update_mode()

    # ---- chat ------------------------------------------------------------
    def _say(self, who: str, text: str):
        color = {"you": "#9cc3ff", "assistant": "#e8e8ee", "tool": "#8fd19e", "error": "#ff8a8a"}[who]
        label = {"you": "You", "assistant": "ClipEditor AI", "tool": "🔧", "error": "⚠"}[who]
        self.log.append(f'<p style="margin:4px 0"><b style="color:{color}">{label}</b> '
                        f'<span style="color:{color}">{html.escape(text).replace(chr(10), "<br>")}</span></p>')

    def send(self, text: str):
        text = text.strip()
        if not text or self.busy:
            return
        self.input.clear()
        self.busy = True
        self.send_btn.setEnabled(False)
        self._say("you", text)
        self._streaming = ""

        def work():
            try:
                self.assistant.send(text, self.events.event.emit)
            except Exception as e:  # noqa: BLE001
                self.events.event.emit({"type": "error", "message": str(e)})
                self.events.event.emit({"type": "done", "text": ""})

        threading.Thread(target=work, daemon=True).start()

    def _flush(self):
        if self._streaming.strip():
            self._say("assistant", self._streaming.strip())
        self._streaming = ""

    def _on_event(self, e: dict):
        t = e.get("type")
        if t == "text":
            self._streaming += e.get("text", "")
        elif t == "tool_call":
            self._flush()
            args = json.dumps(e.get("args") or {}, ensure_ascii=False)
            self._say("tool", f"{e.get('name')} {args[:160]}")
        elif t == "tool_result" and e.get("is_error"):
            self._say("error", str(e.get("result", {}).get("error", ""))[:300])
        elif t == "error":
            self._flush()
            self._say("error", e.get("message", "Something went wrong"))
        elif t == "done":
            self._flush()
            self.busy = False
            self.send_btn.setEnabled(True)
            self.win.status("AI Assistant finished — Ctrl+Z undoes its changes")
