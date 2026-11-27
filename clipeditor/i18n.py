"""Runtime UI translation for ClipEditor.

The UI source stays in plain English. This module translates the *live widget tree*: it walks
widgets, actions, menus, tabs, combo boxes and list items, swaps their English texts for the
current language, and remembers the English original on each object (dynamic properties named
``_i18n_*``) so switching language again, or back to English, is exact.

    from clipeditor import i18n
    i18n.install(app, window)            # loads the saved language, translates, keeps watching
    i18n.switch_language(app, window, "es")   # change at runtime (also saves the choice)

String tables live in ``clipeditor/assets/i18n/<code>.json`` ({english_key: translation}).
Keys are the "core" of a UI string: icon prefixes ("✂  "), menu arrows ("  ▾"), ellipses ("…"),
trailing colons and shortcut hints ("   (Alt+D)") are split off by :func:`split_affixes`,
looked up separately and glued back on. Keys containing ``{name}`` are patterns for dynamic
text ("Copied {n} clip(s)…"). Anything not found stays English.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache

I18N_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "i18n")

LANGUAGES: dict[str, str] = {
    "en": "English",
    "hi": "हिन्दी",
    "ta": "தமிழ்",
    "te": "తెలుగు",
    "bn": "বাংলা",
    "es": "Español",
    "fr": "Français",
    "de": "Deutsch",
    "pt_BR": "Português (Brasil)",
    "ru": "Русский",
    "id": "Bahasa Indonesia",
    "ar": "العربية",
    "ja": "日本語",
    "zh_CN": "简体中文",
}
RTL_LANGUAGES = {"ar"}
SETTINGS_ORG = SETTINGS_APP = "ClipEditor"
SETTINGS_KEY = "language"

_current = "en"

# ---------------------------------------------------------------------------------------------
# string splitting (shared with scripts/extract_strings.py)

_KEY = (r"(?:(?:Ctrl|Shift|Alt|Cmd|Meta)\s*\+\s*)*"
        r"(?:F\d{1,2}|Del|Delete|Space|Home|End|Left|Right|Up|Down|Esc|Enter|Return|Tab|Backspace|"
        r"PgUp|PgDn|wheel|click|drag|[A-Z0-9]|[=+\-])")
_KEYS = rf"{_KEY}(?:\s*/\s*{_KEY})*"
_HINT_PAREN = re.compile(rf"\s+\(\s*{_KEYS}\s*\)$")
_HINT_BARE = re.compile(rf"\s{{2,}}{_KEYS}$")
_LEAD_KEEP = "(\"'“‘«¿¡<[%{"
_TRAIL_KEEP = ".!?)]”\"'’»%}"
_HAS_WORD = re.compile(r"[^\W\d_]", re.UNICODE)          # any letter, any script
_PLACEHOLDER = re.compile(r"%\d+|%[sdif]|\{[A-Za-z_0-9]*\}")
# user content / data that must never be translated
_DATAISH = re.compile(r"^[\s\d:.,;/\\×x*+\-−–—%()\[\]#°'\"]+(?:\s*(?:s|ms|fps|px|dB|Hz|kHz|x|%|p|K))?\s*$")
_TIMECODE = re.compile(r"\d{1,2}:\d{2}(?::\d{2})?(?:[.:;]\d{1,3})?")
_PATHLIKE = re.compile(r"(?:^[A-Za-z]:[\\/])|(?:[\\/][^\s\\/]+[\\/])|(?:\.(?:mp4|mov|mkv|wav|mp3|png|jpe?g|json|webm|avi|srt)\b)",
                       re.IGNORECASE)


def _is_trail_symbol(ch: str) -> bool:
    return not (ch.isalnum() or ch == "_" or ch in _TRAIL_KEEP)


def split_affixes(text: str) -> tuple[str, str, str]:
    """Split "✂  Split   (S)" into ("✂  ", "Split", "   (S)"). Core is "" when there is no word."""
    if not text:
        return "", "", ""
    i = 0
    while i < len(text) and not (text[i].isalnum() or text[i] == "_" or text[i] in _LEAD_KEEP):
        i += 1
    prefix, rest = text[:i], text[i:]
    suffix = ""
    while True:
        before = rest
        m = _HINT_PAREN.search(rest) or _HINT_BARE.search(rest)
        if m and m.start() > 0:
            suffix = rest[m.start():] + suffix
            rest = rest[:m.start()]
        j = len(rest)
        while j > 0 and (_is_trail_symbol(rest[j - 1]) or rest[j - 1].isspace()):
            j -= 1
        if j < len(rest):
            suffix = rest[j:] + suffix
            rest = rest[:j]
        for tail in ("...", "…"):
            if rest.endswith(tail) and len(rest) > len(tail):
                suffix = tail + suffix
                rest = rest[:-len(tail)]
        if rest == before:
            break
    if not _HAS_WORD.search(rest):
        return text, "", ""
    return prefix, rest, suffix


def strip_mnemonic(text: str) -> str:
    return text.replace("&&", "\0").replace("&", "").replace("\0", "&")


def looks_like_data(text: str) -> bool:
    """Digits, timecodes, sizes, file paths: never translated."""
    t = text.strip()
    if not t or not _HAS_WORD.search(t):
        return True
    if _DATAISH.match(t):
        return True
    if _PATHLIKE.search(t):
        return True
    core = _TIMECODE.sub("", t)
    return len(core.strip(" /·→-")) == 0


def placeholders(text: str) -> list[str]:
    return sorted(_PLACEHOLDER.findall(text))


# ---------------------------------------------------------------------------------------------
# tables

@lru_cache(maxsize=None)
def _table(code: str) -> dict[str, str]:
    path = os.path.join(I18N_DIR, f"{code}.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, str) and v and not k.startswith("_")}


@lru_cache(maxsize=None)
def _patterns(code: str) -> list[tuple[re.Pattern, str, list[str]]]:
    out = []
    for key, val in _table(code).items():
        names = re.findall(r"\{([A-Za-z_][A-Za-z_0-9]*)\}", key)
        if not names:
            continue
        rx, pos = "", 0
        for m in re.finditer(r"\{([A-Za-z_][A-Za-z_0-9]*)\}", key):
            rx += re.escape(key[pos:m.start()]) + f"(?P<{m.group(1)}>.+?)"
            pos = m.end()
        rx += re.escape(key[pos:])
        try:
            out.append((re.compile(rf"^{rx}$", re.DOTALL), val, names))
        except re.error:
            continue
    out.sort(key=lambda p: -len(p[0].pattern))
    return out


def available_languages() -> dict[str, str]:
    return {c: n for c, n in LANGUAGES.items() if c == "en" or os.path.exists(os.path.join(I18N_DIR, f"{c}.json"))}


def current() -> str:
    return _current


def set_language(code: str, app=None) -> str:
    """Select the language used by :func:`tr`. Unknown codes fall back to English.
    With ``app`` it also sets the layout direction and loads Qt's own translations."""
    global _current
    code = (code or "en").replace("-", "_")
    if code not in LANGUAGES:
        base = code.split("_")[0]
        code = next((c for c in LANGUAGES if c.split("_")[0] == base), "en")
    _current = code
    _tr_cached.cache_clear()
    if app is not None:
        apply_layout_direction(app)
        _load_qt_translator(app, code)
    return code


def is_rtl(code: str | None = None) -> bool:
    return (code or _current) in RTL_LANGUAGES


def apply_layout_direction(app) -> None:
    from PySide6.QtCore import Qt
    app.setLayoutDirection(Qt.LayoutDirection.RightToLeft if is_rtl() else Qt.LayoutDirection.LeftToRight)


_qt_translator = None


def _load_qt_translator(app, code: str) -> None:
    """Standard buttons (OK / Cancel / Yes) in file and message dialogs come from qtbase_*.qm."""
    global _qt_translator
    try:
        from PySide6.QtCore import QLibraryInfo, QTranslator
        if _qt_translator is not None:
            app.removeTranslator(_qt_translator)
            _qt_translator = None
        if code == "en":
            return
        tr_ = QTranslator(app)
        path = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
        for name in (f"qtbase_{code}", f"qtbase_{code.split('_')[0]}"):
            if tr_.load(name, path):
                app.installTranslator(tr_)
                _qt_translator = tr_
                return
    except Exception:  # noqa: BLE001 - Qt translations are a nicety
        pass


# ---------------------------------------------------------------------------------------------
# translation

def _lookup(core: str, table: dict[str, str]) -> str | None:
    val = table.get(core)
    if val is None:
        return None
    if placeholders(val) != placeholders(core):     # a broken translation must not break formatting
        return None
    return val


def _match_pattern(text: str, code: str) -> str | None:
    for rx, val, names in _patterns(code):
        m = rx.match(text)
        if not m:
            continue
        if any(v not in val for v in (f"{{{n}}}" for n in names)):
            continue
        out = val
        table = _table(code)
        for n in names:
            g = m.group(n)
            gp, gc, gs = split_affixes(g)
            if gc and gc in table:             # "apply {what}" with what = "a filter" -> translated too
                g = gp + table[gc] + gs
            out = out.replace(f"{{{n}}}", g)
        return out
    return None


def _translate(text: str, code: str, depth: int = 0) -> str:
    table = _table(code)
    hit = _lookup(text, table)
    if hit is not None:
        return hit
    if depth > 3:
        return text
    if "&" in text and re.search(r"&[^&\s]", text.replace("&&", "")):
        plain = strip_mnemonic(text)
        out = _translate(plain, code, depth + 1)
        return text if out == plain else "&" + out.replace("&", "&&")
    if "\n" in text:
        return "\n".join(_translate(line, code, depth + 1) for line in text.split("\n"))
    if "<" in text and ">" in text and re.search(r"<\w+[^>]*>", text):
        return _translate_html(text, code)
    prefix, core, suffix = split_affixes(text)
    if not core:
        return text
    if core != text:
        hit = _lookup(core, table)
        if hit is not None:
            return prefix + hit + suffix
    hit = _match_pattern(core, code)
    if hit is not None:
        return prefix + hit + suffix
    if "·" in core:                                    # "Welcome!  ·  press F1 for shortcuts"
        parts = re.split(r"(\s*·\s*)", core)
        out = [p if i % 2 else _translate(p, code, depth + 1) for i, p in enumerate(parts)]
        joined = "".join(out)
        if joined != core:
            return prefix + joined + suffix
    if " — " in core:
        parts = core.split(" — ")
        out = [_translate(p, code, depth + 1) for p in parts]
        if out != parts:
            return prefix + " — ".join(out) + suffix
    return text


def _translate_html(html: str, code: str) -> str:
    def node(m):
        raw = m.group(1)
        if not _HAS_WORD.search(raw):
            return m.group(0)
        lead = raw[:len(raw) - len(raw.lstrip())]
        trail = raw[len(raw.rstrip()):]
        core = raw.strip().replace("&amp;", "&")
        out = _translate(core, code, 4)  # no further html recursion
        if out == core:
            return m.group(0)
        return ">" + lead + out.replace("&", "&amp;") + trail + "<"
    return re.sub(r">([^<>]+)<", node, html)


@lru_cache(maxsize=4096)
def _tr_cached(text: str, code: str) -> str:
    return _translate(text, code)


def tr(text: str, code: str | None = None) -> str:
    """Translate one English UI string into the current (or given) language."""
    code = code or _current
    if not isinstance(text, str) or code == "en" or not text.strip():
        return text
    if looks_like_data(text):
        return text
    return _tr_cached(text, code)


@lru_cache(maxsize=None)
def _reverse(code: str) -> dict[str, str]:
    return {v: k for k, v in _table(code).items() if "{" not in k}


_applied: set[str] = set()      # languages ever applied to widgets in this process


def _untranslate(text: str, codes=None) -> str:
    """Best effort: English source of a text that some other code path already translated (e.g. a
    QToolButton that copied its tooltip from a translated QAction)."""
    if not text:
        return text
    for code in codes or sorted(_applied):
        rev = _reverse(code)
        if text in rev:
            return rev[text]
        if text.startswith("&"):
            inner = text[1:].replace("&&", "&")
            if inner in rev:
                return "&" + rev[inner].replace("&", "&&")
        if "\n" in text:
            lines = text.split("\n")
            parts = [_untranslate(line, [code]) for line in lines]
            if parts != lines:
                return "\n".join(parts)
        prefix, core, suffix = split_affixes(text)
        if core and core in rev:
            return prefix + rev[core] + suffix
    return text


# ---------------------------------------------------------------------------------------------
# widget tree

SKIP_PROP = "i18n_skip"          # setProperty("i18n_skip", True) on any widget to leave it alone
# attributes of MainWindow that hold user content (timeline names, fonts)
_USER_CONTENT_ATTRS = ("timeline_choice", "font", "font_box", "font_combo", "family")


def _slot(obj, name: str, getter, setter, code: str) -> None:
    src_prop, out_prop = f"_i18n_{name}", f"_i18n_{name}_out"
    try:
        cur = getter()
    except RuntimeError:
        return
    if not isinstance(cur, str):
        return
    src, last = obj.property(src_prop), obj.property(out_prop)
    if not isinstance(src, str) or cur != last:
        if not cur and not isinstance(src, str):
            return
        src = _untranslate(cur) if _applied else cur   # first visit, or the app changed the text since
        obj.setProperty(src_prop, src)
    out = src if code == "en" else tr(src, code)
    if out != cur:
        setter(out)
    obj.setProperty(out_prop, out)


def _translate_action(a, code: str) -> None:
    if a.property(SKIP_PROP) or a.isSeparator():
        return
    _slot(a, "text", a.text, a.setText, code)
    tip = a.toolTip()
    derived = strip_mnemonic(a.text()).replace("...", "").replace("…", "")
    src_tip = a.property("_i18n_tip")
    if isinstance(src_tip, str) or (tip and tip != derived and tip != strip_mnemonic(a.text())):
        _slot(a, "tip", a.toolTip, a.setToolTip, code)
    menu = a.menu()
    if menu is not None:
        _translate_menu(menu, code)


def _translate_menu(menu, code: str) -> None:
    _slot(menu, "title", menu.title, menu.setTitle, code)
    for a in menu.actions():
        if a.menu() is not None and a.menu() is not menu:
            _slot(a.menu(), "title", a.menu().title, a.menu().setTitle, code)
        _translate_action(a, code)


def _install_src_accessors(combo) -> None:
    """App code reads combo texts as keys ("All sounds", effect categories). Python calls get the
    English source back; Qt paints the translated text."""
    if getattr(combo, "_i18n_patched", False):
        return
    orig_item_text, orig_current = combo.itemText, combo.currentText
    orig_find = combo.findText

    def item_text(i):
        src = combo.itemData(i, _SRC_ROLE)
        return src if isinstance(src, str) else orig_item_text(i)

    def current_text():
        i = combo.currentIndex()
        if i >= 0 and not combo.isEditable():
            return item_text(i)
        return orig_current()

    def find_text(text, *args):
        for i in range(combo.count()):
            if item_text(i) == text:
                return i
        return orig_find(text, *args)

    try:
        combo.itemText, combo.currentText, combo.findText = item_text, current_text, find_text
        combo._i18n_patched = True
    except (AttributeError, TypeError):
        pass


_SRC_ROLE = 0x0100 + 0x4A7      # Qt.UserRole + n: stores the English item text


def _translate_combo(combo, code: str) -> None:
    from PySide6.QtWidgets import QFontComboBox
    if isinstance(combo, QFontComboBox) or combo.isEditable():
        return
    model = combo.model()
    any_changed = False
    for i in range(combo.count()):
        cur = model.data(model.index(i, 0))
        if not isinstance(cur, str):
            continue
        src, last = combo.itemData(i, _SRC_ROLE), combo.itemData(i, _SRC_ROLE + 1)
        if not isinstance(src, str) or cur != last:
            src = cur
            combo.setItemData(i, src, _SRC_ROLE)
        out = src if code == "en" else tr(src, code)
        if out != cur:
            model.setData(model.index(i, 0), out)
        combo.setItemData(i, out, _SRC_ROLE + 1)
        any_changed = any_changed or out != src
        tip = combo.itemData(i, 3)  # ToolTipRole
        if isinstance(tip, str) and tip:
            tsrc = combo.itemData(i, _SRC_ROLE + 2)
            if not isinstance(tsrc, str) or tip != combo.itemData(i, _SRC_ROLE + 3):
                tsrc = tip
                combo.setItemData(i, tsrc, _SRC_ROLE + 2)
            tout = tsrc if code == "en" else tr(tsrc, code)
            if tout != tip:
                combo.setItemData(i, tout, 3)
            combo.setItemData(i, tout, _SRC_ROLE + 3)
    if any_changed:
        _install_src_accessors(combo)


def _item_slot(item, idx: int, role: int, code: str) -> None:
    cur = item.data(role)
    if not isinstance(cur, str) or not cur:
        if not isinstance(item.data(_SRC_ROLE + idx), str):
            return
    src, last = item.data(_SRC_ROLE + idx), item.data(_SRC_ROLE + idx + 1)
    if not isinstance(src, str) or cur != last:
        src = cur
        item.setData(_SRC_ROLE + idx, src)
    out = src if code == "en" else tr(src, code)
    if out != cur:
        item.setData(role, out)
    item.setData(_SRC_ROLE + idx + 1, out)


def _translate_list(lw, code: str) -> None:
    for i in range(lw.count()):
        it = lw.item(i)
        _item_slot(it, 0, 0, code)   # DisplayRole
        _item_slot(it, 2, 3, code)   # ToolTipRole


def _translate_tree_widget(tw, code: str) -> None:
    hdr = tw.headerItem()
    for col in range(tw.columnCount()):
        cur = hdr.text(col)
        if cur:
            out = tr(cur, code) if code != "en" else cur
            if out != cur:
                hdr.setText(col, out)


def _translate_text_browser(tb, code: str) -> None:
    if tb.property(SKIP_PROP) or not tb.isReadOnly():
        return
    html = tb.toHtml()
    src, last = tb.property("_i18n_html"), tb.property("_i18n_html_out")
    if not isinstance(src, str) or html != last:
        src = html
        tb.setProperty("_i18n_html", src)
    out = src if code == "en" else _translate_html(src, code)
    if out != html:
        tb.setHtml(out)
    tb.setProperty("_i18n_html_out", tb.toHtml())


def translate_widget(w, code: str | None = None) -> None:
    """Translate one widget (not its children)."""
    from PySide6.QtWidgets import (QAbstractButton, QComboBox, QGroupBox, QLabel, QLineEdit, QListWidget,
                                   QMenu, QPlainTextEdit, QTabWidget, QTextBrowser, QTextEdit, QToolBox,
                                   QTreeWidget, QAbstractSpinBox, QDockWidget, QToolBar)
    code = code or _current
    try:
        if w.property(SKIP_PROP):
            return
    except RuntimeError:
        return
    if w.isWindow() or isinstance(w, (QDockWidget, QToolBar)):
        _slot(w, "title", w.windowTitle, w.setWindowTitle, code)
    if w.toolTip():
        _slot(w, "tip", w.toolTip, w.setToolTip, code)
    elif isinstance(w.property("_i18n_tip"), str):
        _slot(w, "tip", w.toolTip, w.setToolTip, code)
    if isinstance(w, QLabel):
        _slot(w, "text", w.text, w.setText, code)
    elif isinstance(w, QAbstractButton):
        _slot(w, "text", w.text, w.setText, code)
    elif isinstance(w, QGroupBox):
        _slot(w, "title", w.title, w.setTitle, code)
    elif isinstance(w, QTextBrowser):
        _translate_text_browser(w, code)
    elif isinstance(w, (QLineEdit, QPlainTextEdit, QTextEdit)):
        _slot(w, "ph", w.placeholderText, w.setPlaceholderText, code)   # never the user's text
    elif isinstance(w, QComboBox):
        _translate_combo(w, code)
    elif isinstance(w, QListWidget):
        _translate_list(w, code)
    elif isinstance(w, QTreeWidget):
        _translate_tree_widget(w, code)
    elif isinstance(w, QTabWidget):
        for i in range(w.count()):
            _tab_slot(w, i, code)
    elif isinstance(w, QToolBox):
        for i in range(w.count()):
            key = f"_i18n_box{i}"
            cur = w.itemText(i)
            src = w.property(key)
            if not isinstance(src, str) or cur != w.property(key + "_out"):
                src = cur
                w.setProperty(key, src)
            out = src if code == "en" else tr(src, code)
            if out != cur:
                w.setItemText(i, out)
            w.setProperty(key + "_out", out)
    elif isinstance(w, QAbstractSpinBox):
        if hasattr(w, "specialValueText") and w.specialValueText():
            _slot(w, "special", w.specialValueText, w.setSpecialValueText, code)
    elif isinstance(w, QMenu):
        _translate_menu(w, code)
    for a in w.actions():
        _translate_action(a, code)


def _tab_slot(tabs, i: int, code: str) -> None:
    for kind, get, put in (("tab", tabs.tabText, tabs.setTabText), ("tabtip", tabs.tabToolTip, tabs.setTabToolTip)):
        key = f"_i18n_{kind}{i}"
        cur = get(i)
        src = tabs.property(key)
        if not cur and not isinstance(src, str):
            continue
        if not isinstance(src, str) or cur != tabs.property(key + "_out"):
            src = cur
            tabs.setProperty(key, src)
        out = src if code == "en" else tr(src, code)
        if out != cur:
            put(i, out)
        tabs.setProperty(key + "_out", out)


def _mark_user_content(root) -> None:
    for attr in _USER_CONTENT_ATTRS:
        w = getattr(root, attr, None)
        if w is not None and hasattr(w, "setProperty"):
            try:
                w.setProperty(SKIP_PROP, True)
            except (RuntimeError, TypeError):
                pass


def translate_widget_tree(root, code: str | None = None) -> None:
    """Translate ``root`` and every child widget, action and menu, in place."""
    from PySide6.QtWidgets import QMainWindow, QWidget
    code = code or _current
    if code != _current and code in LANGUAGES:
        set_language(code)
    if code != "en":
        _applied.add(code)
    _mark_user_content(root)
    widgets = [root] + root.findChildren(QWidget)
    for w in widgets:
        translate_widget(w, code)
    if isinstance(root, QMainWindow):
        mb = root.menuBar()
        for a in mb.actions():
            _translate_action(a, code)
    from PySide6.QtGui import QAction
    for a in root.findChildren(QAction):
        _translate_action(a, code)


# ---------------------------------------------------------------------------------------------
# install / persistence

_filter = None


def _make_filter():
    from PySide6.QtCore import QEvent, QObject
    from PySide6.QtWidgets import QMenu, QWidget

    show, polish = QEvent.Type.Show, QEvent.Type.Polish

    class _Translator(QObject):
        def eventFilter(self, obj, ev):  # noqa: N802 - Qt API
            t = ev.type()
            if (t == show or t == polish) and _current != "en":
                try:
                    if isinstance(obj, QWidget):
                        if isinstance(obj, QMenu):
                            _translate_menu(obj, _current)
                        elif t == show and obj.isWindow() and obj.property("_i18n_win") != _current:
                            obj.setProperty("_i18n_win", _current)
                            translate_widget_tree(obj, _current)
                        else:
                            translate_widget(obj, _current)
                except RuntimeError:
                    pass
            return False

    return _Translator()


_was_translated = False


def install(app, root_window=None, code: str | None = None) -> str:
    """Pick the saved (or given) language, translate ``root_window`` now and watch for new
    dialogs, menus and popups. Returns the active language code."""
    global _filter, _was_translated
    code = set_language(code or load_language(), app)
    if _filter is None:
        _filter = _make_filter()
        app.installEventFilter(_filter)
    if root_window is not None:
        translate_widget_tree(root_window, code)
        _watch_status_bar(root_window)
    _was_translated = _was_translated or code != "en"
    return code


def _watch_status_bar(win) -> None:
    """QStatusBar.showMessage() text is painted, not a widget: retranslate it as it arrives."""
    sb = getattr(win, "statusBar", None)
    if sb is None or getattr(win, "_i18n_status_hooked", False):
        return
    try:
        bar = sb()
    except (TypeError, RuntimeError):
        return

    def on_message(msg):
        if not msg or _current == "en":
            return
        out = tr(msg)
        if out != msg:
            bar.showMessage(out, max(4000, min(10000, 60 * len(msg))))

    bar.messageChanged.connect(on_message)
    win._i18n_status_hooked = True


def switch_language(app, root_window, code: str) -> str:
    """Change language at runtime: retranslate every top-level window and save the choice."""
    global _was_translated
    code = set_language(code, app)
    save_language(code)
    tops = [root_window] if root_window is not None else []
    tops += [w for w in app.topLevelWidgets() if w is not root_window]
    for w in tops:
        try:
            w.setProperty("_i18n_win", code)
            translate_widget_tree(w, code)
        except RuntimeError:
            pass
    _was_translated = True
    return code


def _settings():
    from PySide6.QtCore import QSettings
    return QSettings(SETTINGS_ORG, SETTINGS_APP)


def load_language(default: str = "en") -> str:
    try:
        val = _settings().value(SETTINGS_KEY, default)
    except Exception:  # noqa: BLE001
        return default
    return val if isinstance(val, str) and val in LANGUAGES else default


def save_language(code: str) -> None:
    try:
        s = _settings()
        s.setValue(SETTINGS_KEY, code)
        s.sync()
    except Exception:  # noqa: BLE001
        pass


def coverage(code: str, keys=None) -> float:
    """Share of extracted English keys that have a translation in ``code``."""
    if keys is None:
        with open(os.path.join(I18N_DIR, "strings_en.json"), encoding="utf-8") as f:
            keys = list(json.load(f)["strings"])
    if code == "en":
        return 1.0
    table = _table(code)
    return sum(1 for k in keys if k in table) / max(1, len(keys))
