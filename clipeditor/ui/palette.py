"""Ctrl+K command palette: type a few letters, press Enter. Finds every menu command,
every effect / filter / animation / transition / sticker / sound, and every AI tool."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QVBoxLayout


def _score(query: str, text: str) -> int:
    q, t = query.lower(), text.lower()
    if not q:
        return 1
    if t.startswith(q):
        return 100
    if q in t:
        return 60 - t.index(q)
    # fuzzy: all letters in order
    i = 0
    for ch in t:
        if i < len(q) and ch == q[i]:
            i += 1
    return 10 if i == len(q) else 0


class CommandPalette(QDialog):
    def __init__(self, win, commands: list[tuple[str, str, object]]):
        """commands: (title, hint, callable)"""
        super().__init__(win)
        self.setWindowFlags(Qt.WindowType.Popup)
        self.setMinimumWidth(560)
        self.commands = commands
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("Type a command, effect, sticker, sound…  (↑↓ Enter)")
        self.edit.setStyleSheet("font-size: 13pt; padding: 8px;")
        lay.addWidget(self.edit)
        self.list = QListWidget()
        self.list.setMinimumHeight(380)
        lay.addWidget(self.list)
        tip = QLabel("Tip: try “vhs”, “captions”, “split”, “9:16”, “pop”, “whoosh”")
        tip.setStyleSheet("color: #8a8f99;")
        lay.addWidget(tip)
        self.edit.textChanged.connect(self._filter)
        self.edit.returnPressed.connect(self._run)
        self.list.itemActivated.connect(lambda _i: self._run())
        self._filter("")

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Down, Qt.Key.Key_Up):
            r = self.list.currentRow() + (1 if e.key() == Qt.Key.Key_Down else -1)
            self.list.setCurrentRow(max(0, min(r, self.list.count() - 1)))
            return
        super().keyPressEvent(e)

    def _filter(self, q):
        self.list.clear()
        scored = sorted(((s, i) for i, (t, h, _f) in enumerate(self.commands)
                         if (s := max(_score(q, t), _score(q, h) // 2)) > 0), key=lambda x: -x[0])
        for _s, i in scored[:60]:
            title, hint, _fn = self.commands[i]
            it = QListWidgetItem(f"{title}      —  {hint}" if hint else title)
            it.setData(Qt.ItemDataRole.UserRole, i)
            self.list.addItem(it)
        if self.list.count():
            self.list.setCurrentRow(0)

    def _run(self):
        it = self.list.currentItem()
        if it is None:
            return
        fn = self.commands[it.data(Qt.ItemDataRole.UserRole)][2]
        self.accept()
        fn()
