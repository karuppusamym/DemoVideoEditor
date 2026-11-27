"""Runtime UI translation: string extraction, language tables, tr() and the widget-tree walker."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
I18N = ROOT / "clipeditor" / "assets" / "i18n"
REQUIRED = ["hi", "ta", "te", "es", "fr", "de", "pt_BR", "ar", "ja", "zh_CN"]


def _keys():
    return list(json.loads((I18N / "strings_en.json").read_text(encoding="utf-8"))["strings"])


def test_extractor_finds_strings():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import extract_strings
    finally:
        sys.path.pop(0)
    strings = extract_strings.extract()
    assert len(strings) > 300
    for key in ("Split", "Export", "Keyboard shortcuts", "AI Magic", "Dissolve", "Title", "File"):
        assert key in strings, key
    assert not any(k.startswith(("✂", "🎬", " ")) for k in strings)


def test_language_files_cover_strings():
    from clipeditor import i18n

    keys = _keys()
    report = []
    for code in i18n.LANGUAGES:
        if code == "en":
            continue
        path = I18N / f"{code}.json"
        if code not in REQUIRED and not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        cov = i18n.coverage(code, keys)
        report.append(f"{code}: {cov:.1%}")
        assert cov >= 0.90, f"{code} coverage {cov:.1%}"
        for k, v in data.items():
            assert i18n.placeholders(k) == i18n.placeholders(v), (code, k, v)
    print("\ncoverage  " + "  ".join(report))
    assert len(report) >= 10


def test_tr_keeps_affixes():
    from clipeditor import i18n

    i18n.set_language("es")
    try:
        table = i18n._table("es")
        split = table["Split"]
        assert i18n.tr("✂ Split") == "✂ " + split
        assert i18n.tr("✂  Split") == "✂  " + split
        frame = table["Frame"]
        assert i18n.tr("🎞 Frame ▾") == "🎞 " + frame + " ▾"
        assert i18n.tr("Open project…") == table["Open project"] + "…"
        assert i18n.tr("Duplicate this frame   (Alt+D)") == table["Duplicate this frame"] + "   (Alt+D)"
        assert i18n.tr("Split at playhead (S)") == table["Split at playhead"] + " (S)"
        assert i18n.tr("&File") == "&" + table["File"]
        # dynamic text and data stay intact
        out = i18n.tr("Copied 3 clip(s) — Ctrl+V pastes at the playhead")
        assert "3" in out and out != "Copied 3 clip(s) — Ctrl+V pastes at the playhead"
        for data in ("00:01:02.15", "1920×1080", "clip_a.mp4", "C:\\Videos\\Split.mp4", "12.5s"):
            assert i18n.tr(data) == data
        assert i18n.tr("Some text nobody translated") == "Some text nobody translated"
        assert i18n.tr("Split", "en") == "Split"
    finally:
        i18n.set_language("en")


def test_split_affixes():
    from clipeditor.i18n import split_affixes

    assert split_affixes("✂  Split") == ("✂  ", "Split", "")
    assert split_affixes("  ⬆  Export  ") == ("  ⬆  ", "Export", "  ")
    assert split_affixes("Mark range start (In)")[1] == "Mark range start (In)"
    assert split_affixes("Freeze this frame for 2 s   (F)") == ("", "Freeze this frame for 2 s", "   (F)")
    assert split_affixes("▶")[1] == ""


def test_main_window_translation():
    """Builds a real MainWindow offscreen in a fresh process (one QApplication per process)."""
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONIOENCODING="utf-8")
    if os.name == "nt":
        env.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))
    r = subprocess.run([sys.executable, "-m", "tests.test_i18n"], cwd=ROOT, env=env, timeout=120,
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "I18N-OK" in r.stdout


def _snapshot(win):
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import QAbstractButton, QLabel, QTabWidget, QWidget

    snap = {}
    for i, w in enumerate([win] + win.findChildren(QWidget)):
        if isinstance(w, (QLabel, QAbstractButton)):
            snap[("text", id(w))] = w.text()
        snap[("tip", id(w))] = w.toolTip()
        if isinstance(w, QTabWidget):
            snap[("tabs", id(w))] = tuple(w.tabText(j) for j in range(w.count()))
    for a in win.findChildren(QAction):
        snap[("action", id(a))] = (a.text(), a.toolTip())
    snap["menus"] = tuple(a.text() for a in win.menuBar().actions())
    return snap


def _exercise_window():
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QAbstractButton, QApplication, QLabel, QWidget

    from clipeditor import i18n
    from clipeditor.ui.main_window import MainWindow, apply_dark_theme

    app = QApplication.instance() or QApplication([])
    apply_dark_theme(app)
    win = MainWindow()
    win.show()
    app.processEvents()
    try:
        en = _snapshot(win)
        menus_en = [a.text() for a in win.menuBar().actions()]
        texts_en = {w.text() for w in win.findChildren(QAbstractButton) + win.findChildren(QLabel)}

        i18n.translate_widget_tree(win, "es")
        es_table = i18n._table("es")
        menus_es = [a.text() for a in win.menuBar().actions()]
        assert menus_es != menus_en
        for en_t, es_t in zip(menus_en, menus_es):
            core = i18n.strip_mnemonic(en_t)
            assert es_t == "&" + es_table[core].replace("&", "&&"), (en_t, es_t)
        texts_es = {w.text() for w in win.findChildren(QAbstractButton) + win.findChildren(QLabel)}
        for en_t in ("✂ Split", "  ⬆  Export  ", "🗑 Delete", "⧉ Duplicate"):
            if en_t in texts_en:
                assert i18n.tr(en_t, "es") in texts_es and en_t not in texts_es, en_t
        rail = [b.text() for b in win.findChildren(QAbstractButton) if "\n" in b.text()]
        if rail:  # sidebar rail buttons "🎬\nMedia"
            assert any(es_table.get("Media", "Media") in t or es_table.get("Audio", "") in t for t in rail)
        changed = sum(1 for k, v in _snapshot(win).items() if en.get(k) != v)
        assert changed > 40, changed
        # timecodes / user content untouched
        assert win.time_lbl.text() == next(v for k, v in en.items() if k == ("text", id(win.time_lbl)))

        # idempotent
        once = _snapshot(win)
        i18n.translate_widget_tree(win, "es")
        assert _snapshot(win) == once

        # other language from a translated state, then back to English exactly
        i18n.translate_widget_tree(win, "ja")
        assert [a.text() for a in win.menuBar().actions()] != menus_es
        i18n.translate_widget_tree(win, "en")
        assert _snapshot(win) == en

        # right-to-left for Arabic, plus install() + a freshly shown widget gets translated
        i18n.set_language("ar", app)
        assert app.layoutDirection() == Qt.LayoutDirection.RightToLeft
        i18n.install(app, win, "ar")
        w = QWidget()
        lbl = QLabel("Keyboard shortcuts", w)
        w.show()
        app.processEvents()
        assert lbl.text() == i18n._table("ar")["Keyboard shortcuts"]
        w.close()
        i18n.set_language("en", app)
        assert app.layoutDirection() == Qt.LayoutDirection.LeftToRight
        i18n.translate_widget_tree(win, "en")
        assert _snapshot(win) == en
    finally:
        try:
            win.loader.shutdown()
            win.comp.close()
        except Exception:  # noqa: BLE001
            pass
        win.hide()
    print("I18N-OK")


if __name__ == "__main__":
    _exercise_window()
