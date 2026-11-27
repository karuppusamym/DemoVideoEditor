"""Collect every user-visible English UI string into clipeditor/assets/i18n/strings_en.json.

    .venv/Scripts/python scripts/extract_strings.py

Sources:
  * clipeditor/ui/*.py, statically (ast): literals passed to setText / setToolTip / addAction /
    QLabel(...) / QMessageBox.*(...) / _title(...) etc., literals that flow into those calls
    through small wrapper functions (``tool(text, tip, fn)``, ``_act(menu, text, ...)``), loop
    tuples and local variables, f-strings (as ``{name}`` patterns) and module-level label
    tables (RAIL, AI_TOOLS, TEXT_STYLES keys, SHORTCUTS html, ...).
  * registries: effects, transitions, animations, shapes, sound effects, audio effects.

Keys are the *core* of each string (icon prefixes, "…", "  ▾" and shortcut hints are split off
with clipeditor.i18n.split_affixes, exactly as at translation time).
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from clipeditor.i18n import _KEYS, I18N_DIR, looks_like_data, split_affixes, strip_mnemonic  # noqa: E402

UI_DIR = os.path.join(ROOT, "clipeditor", "ui")
OUT = os.path.join(I18N_DIR, "strings_en.json")

TEXT_METHODS = {"setText", "setToolTip", "setPlaceholderText", "setWindowTitle", "setStatusTip", "setTitle",
                "setTabText", "setTabToolTip", "setItemText", "setSpecialValueText", "setWhatsThis",
                "addAction", "addMenu", "addTab", "insertTab", "addItem", "addItems", "insertItem", "addRow",
                "setHeaderLabels", "showMessage", "setLabelText", "setInformativeText", "setDetailedText",
                "addButton", "setItemToolTip", "setToolTipText", "setPrefix", "add"}
CONSTRUCTORS = {"QLabel", "QPushButton", "QCheckBox", "QRadioButton", "QToolButton", "QGroupBox", "QAction",
                "QMenu", "QListWidgetItem", "QTreeWidgetItem", "QToolBar", "QDockWidget", "QCommandLinkButton",
                "_title", "QMessageBox", "Section"}
STATIC_OWNERS = {"QMessageBox", "QInputDialog", "QFileDialog", "QProgressDialog"}
FIRST_STRING_ONLY = {"addItem", "insertItem", "add"}
SKIP_MODULE_TABLES = {"STYLE", "MEDIA_FILTER", "PROJECT_FILTER", "ASSET_MIME", "SPEEDS", "BLENDS"}
HTML_TABLES = {"SHORTCUTS"}
FILTER_RX = re.compile(r"\(\*\.[\w*]+")          # "MP4 video (*.mp4)" file-dialog filters
CSS_RX = re.compile(r"[{}]|\b[a-z-]+\s*:\s*[^;]*;|^\s*#[0-9a-fA-F]{3,8}\b|^[a-z-]+\s*:")


def func_name(node: ast.Call) -> tuple[str | None, str | None]:
    f = node.func
    if isinstance(f, ast.Name):
        return None, f.id
    if isinstance(f, ast.Attribute):
        owner = f.value.id if isinstance(f.value, ast.Name) else None
        return owner, f.attr
    return None, None


def is_target(node: ast.Call) -> bool:
    owner, name = func_name(node)
    if owner in STATIC_OWNERS:
        return True
    return name in TEXT_METHODS or name in CONSTRUCTORS


def ui_text(s: str) -> bool:
    if not isinstance(s, str) or not s.strip():
        return False
    if CSS_RX.search(s) or FILTER_RX.search(s) or "://" in s:
        return False
    if re.fullmatch(r"[a-z0-9_.\-]+", s.strip()) and ("_" in s or "." in s):
        return False                       # identifiers / data keys
    if re.fullmatch(r"[a-z]+", s.strip()) and len(s.strip()) > 1:
        return False                       # lowercase single tokens are data ("slow", "subtitle")
    if len(s.strip()) < 2 or (re.fullmatch(_KEYS, s.strip()) and re.search(r"\+|^F\d+$", s.strip())):
        return False                       # single letters, bare shortcuts ("Shift+Space")
    return not looks_like_data(s)


class Collector:
    def __init__(self):
        self.strings: dict[str, str] = {}      # key -> first source "file:line"

    def add(self, text: str, where: str, template: bool = False):
        if not isinstance(text, str):
            return
        if not template and not ui_text(text):
            return
        text = strip_mnemonic(text) if re.search(r"&[^&\s]", text.replace("&&", "")) else text
        if "<" in text and re.search(r"<\w+[^>]*>", text):
            for chunk in re.findall(r">([^<>]+)<", f">{text}<"):
                self.add(chunk.strip().replace("&amp;", "&"), where, template)
            return
        for line in text.split("\n"):
            self._add_line(line, where, template)

    def _add_line(self, line: str, where: str, template: bool):
        prefix, core, suffix = split_affixes(line)
        if not core:
            return
        if template:
            literal = re.sub(r"\{[A-Za-z_0-9]*\}", "", core)
            if len(re.findall(r"[^\W\d_]", literal)) < 6 or core.count("{") > 3:
                return
            if re.fullmatch(r"\{\w+\}", core.strip()):
                return
        elif not ui_text(core):
            return
        parts = [core]
        if "·" in core:
            parts = [p.strip() for p in re.split(r"\s*·\s*", core) if p.strip()]
        for p in parts:
            pp, pc, ps = split_affixes(p)
            if pc and (template or ui_text(pc)):
                self.strings.setdefault(pc, where)


def template_of(node: ast.JoinedStr) -> str:
    out, used = [], {}
    for v in node.values:
        if isinstance(v, ast.Constant):
            out.append(str(v.value).replace("{", "").replace("}", ""))
        elif isinstance(v, ast.FormattedValue):
            e = v.value
            if isinstance(e, ast.Name):
                name = e.id
            elif isinstance(e, ast.Attribute):
                name = e.attr
            elif isinstance(e, ast.Call) and isinstance(e.func, ast.Name) and e.func.id == "len":
                name = "n"
            else:
                name = "v"
            used[name] = used.get(name, 0) + 1
            if used[name] > 1:
                name = f"{name}{used[name]}"
            out.append("{" + name + "}")
    return "".join(out)


def strings_in(node, local: dict | None = None) -> list[tuple[str, bool]]:
    """String values an expression may evaluate to: (text, is_template)."""
    local = local or {}
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [(node.value, False)]
    if isinstance(node, ast.JoinedStr):
        return [(template_of(node), True)]
    if isinstance(node, ast.IfExp):
        return strings_in(node.body, local) + strings_in(node.orelse, local)
    if isinstance(node, ast.BoolOp):
        return [s for v in node.values for s in strings_in(v, local)]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = strings_in(node.left, local), strings_in(node.right, local)
        if len(left) == 1 and len(right) == 1 and not left[0][1] and not right[0][1]:
            return [(left[0][0] + right[0][0], False)]
        return left + right
    if isinstance(node, (ast.List, ast.Tuple)):
        return [s for v in node.elts for s in strings_in(v, local)]
    if isinstance(node, ast.Name) and node.id in local:
        return local[node.id]
    return []


def call_args(node: ast.Call):
    owner, name = func_name(node)
    args = list(node.args)
    if owner in STATIC_OWNERS and args and isinstance(args[0], ast.Name):
        args = args[1:]                     # parent widget
    if owner == "QFileDialog":
        args = args[:1]                     # caption only (dir and filter are data)
    if owner == "QInputDialog":
        args = args[:2]                     # title, label (then default text / items = data)
    if name in FIRST_STRING_ONLY:
        for a in args:
            if strings_in(a):
                return [a]
        return []
    kws = [k.value for k in node.keywords if k.arg in {"text", "title", "label", "tip", "toolTip", "sub"}]
    return args + kws


class FileScanner(ast.NodeVisitor):
    def __init__(self, path: str, col: Collector, wrappers: dict):
        self.path, self.col, self.wrappers = path, col, wrappers
        self.local_stack: list[dict] = [{}]
        self.rows_stack: list[dict] = [{}]

    def where(self, node):
        return f"{os.path.relpath(self.path, ROOT).replace(os.sep, '/')}:{node.lineno}"

    def visit_FunctionDef(self, node):
        local: dict = {}
        for sub in ast.walk(node):
            if isinstance(sub, ast.Assign) and len(sub.targets) == 1 and isinstance(sub.targets[0], ast.Name):
                vals = strings_in(sub.value)
                if vals:
                    local.setdefault(sub.targets[0].id, []).extend(vals)
        rows: dict = {}       # want = [(w, "Text")]; want += [...]; want.append((w, "Video"))
        for sub in ast.walk(node):
            target, value = None, None
            if isinstance(sub, ast.Assign) and len(sub.targets) == 1 and isinstance(sub.targets[0], ast.Name):
                target, value = sub.targets[0].id, sub.value
            elif isinstance(sub, ast.AugAssign) and isinstance(sub.target, ast.Name):
                target, value = sub.target.id, sub.value
            elif (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "append"
                  and isinstance(sub.func.value, ast.Name) and sub.args):
                target, value = sub.func.value.id, ast.List(elts=[sub.args[0]])
            if target and isinstance(value, (ast.List, ast.Tuple)):
                rows.setdefault(target, []).extend(value.elts)
        self.rows_stack.append(rows)
        self.local_stack.append(local)
        self.generic_visit(node)
        self.local_stack.pop()
        self.rows_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node):
        owner, name = func_name(node)
        local = self.local_stack[-1]
        if is_target(node):
            for a in call_args(node):
                for s, tmpl in strings_in(a, local):
                    self.col.add(s, self.where(node), tmpl)
        if name in self.wrappers:
            for idx in self.wrappers[name]:
                if isinstance(idx, int) and idx < len(node.args):
                    for s, tmpl in strings_in(node.args[idx], local):
                        self.col.add(s, self.where(node), tmpl)
                elif isinstance(idx, str):
                    for k in node.keywords:
                        if k.arg == idx:
                            for s, tmpl in strings_in(k.value, local):
                                self.col.add(s, self.where(node), tmpl)
        self.generic_visit(node)

    def visit_For(self, node):
        # for text, tip, fn in (("▶ Play range", "Shift+Space", ...), ...): collect the loop slots that
        # reach a text call in the body
        tgt = node.target
        names = [t.id if isinstance(t, ast.Name) else None for t in
                 (tgt.elts if isinstance(tgt, ast.Tuple) else [tgt])]
        used = flowing_names(node.body, self.wrappers)
        iter_rows = None
        if isinstance(node.iter, (ast.Tuple, ast.List)):
            iter_rows = node.iter.elts
        elif isinstance(node.iter, ast.Name):
            iter_rows = self.rows_stack[-1].get(node.iter.id)
        if iter_rows:
            for row in iter_rows:
                vals = row.elts if isinstance(row, (ast.Tuple, ast.List)) and isinstance(tgt, ast.Tuple) else [row]
                for n, v in zip(names, vals):
                    if n and n in used:
                        for s, tmpl in strings_in(v):
                            self.col.add(s, self.where(row), tmpl)
        self.generic_visit(node)


def flowing_names(body, wrappers) -> set[str]:
    """Variable names passed (directly or in f-strings / concatenations) to text calls."""
    out: set[str] = set()
    for stmt in body:
        for sub in ast.walk(stmt):
            if not isinstance(sub, ast.Call):
                continue
            _, name = func_name(sub)
            args = []
            if is_target(sub):
                args = list(sub.args) + [k.value for k in sub.keywords]
            elif name in wrappers:
                args = [sub.args[i] for i in wrappers[name] if isinstance(i, int) and i < len(sub.args)]
            for a in args:
                for n in ast.walk(a):
                    if isinstance(n, ast.Name):
                        out.add(n.id)
    return out


def find_wrappers(trees) -> dict[str, list]:
    """def tool(text, tip, fn): b.setText(text); b.setToolTip(tip) -> {"tool": [0, 1]}."""
    wrappers: dict[str, list] = {}
    for _ in range(2):                        # second pass: wrappers of wrappers
        for tree in trees:
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                params = [a.arg for a in node.args.args]
                offset = 1 if params and params[0] in ("self", "cls") else 0
                params = params[offset:]
                flows = flowing_names(node.body, wrappers)
                idx = [i for i, p in enumerate(params) if p in flows and p not in ("parent", "self", "fn", "cb")]
                kw = [p for p in params if p in flows]
                if idx and node.name not in TEXT_METHODS and node.name not in CONSTRUCTORS:
                    wrappers[node.name] = sorted(set(wrappers.get(node.name, []) + idx), key=str) + kw
    return wrappers


def module_tables(tree, path, col: Collector):
    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        elif isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        else:
            continue
        for t in targets:
            if not (isinstance(t, ast.Name) and t.id.isupper()) or t.id in SKIP_MODULE_TABLES:
                continue
            where = f"{os.path.relpath(path, ROOT).replace(os.sep, '/')}:{node.lineno}"
            if t.id in HTML_TABLES and isinstance(value, ast.Constant):
                col.add(value.value, where)
                continue
            if isinstance(value, ast.Dict):
                for k, v in zip(value.keys, value.values):
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        if k.value[:1].isupper() or " " in k.value:
                            col.add(k.value, where)
                    if isinstance(v, ast.Tuple):         # PROPERTIES = {"scale": ("Scale", ...)}
                        for e in v.elts:
                            if isinstance(e, ast.Constant) and isinstance(e.value, str) and e.value[:1].isupper():
                                col.add(e.value, where)
            elif isinstance(value, (ast.List, ast.Tuple)):
                for e in value.elts:
                    items = e.elts if isinstance(e, (ast.Tuple, ast.List)) else [e]
                    for it in items:
                        if isinstance(it, ast.Constant) and isinstance(it.value, str):
                            s = it.value
                            if len(s) > 1 and (" " in s or s[:1].isupper()):
                                col.add(s, where)


def registries(col: Collector):
    def add(s, where):
        if isinstance(s, str):
            col.add(s, where)

    try:
        from clipeditor import effects
        for spec in effects.REGISTRY.values():
            add(spec.get("label"), "effects.REGISTRY")
            add(spec.get("category"), "effects.REGISTRY")
            for p in spec.get("params", {}).values():
                add(p.get("label"), "effects.REGISTRY params")
        for c in getattr(effects, "CATEGORY_ORDER", []):
            add(c, "effects.CATEGORY_ORDER")
    except Exception as e:  # noqa: BLE001
        print("effects:", e)
    try:
        from clipeditor import transitions
        for v in transitions.KINDS.values():
            add(v, "transitions.KINDS")
    except Exception as e:  # noqa: BLE001
        print("transitions:", e)
    try:
        from clipeditor import animate
        for d in animate.ANIMATIONS.values():
            for v in (d.values() if isinstance(d, dict) else d):
                add(v, "animate.ANIMATIONS")
    except Exception as e:  # noqa: BLE001
        print("animate:", e)
    try:
        from clipeditor import shapes
        for k in shapes.PRESETS:
            add(k, "shapes.PRESETS")
    except Exception as e:  # noqa: BLE001
        print("shapes:", e)
    try:
        from clipeditor import sfx
        for v in sfx.LIBRARY.values():
            add(v.get("label"), "sfx.LIBRARY")
            add(v.get("category"), "sfx.LIBRARY")
        for c in getattr(sfx, "CATEGORIES", []):
            add(c, "sfx.CATEGORIES")
    except Exception as e:  # noqa: BLE001
        print("sfx:", e)
    try:
        from clipeditor import audio_fx
        for v in audio_fx.EFFECTS.values():
            add(v.get("label"), "audio_fx.EFFECTS")
            add(v.get("category"), "audio_fx.EFFECTS")
            for p in v.get("params", {}).values():
                add(p.get("label"), "audio_fx.EFFECTS params")
        for c in getattr(audio_fx, "CATEGORIES", []):
            add(c, "audio_fx.CATEGORIES")
    except Exception as e:  # noqa: BLE001
        print("audio_fx:", e)


def extract() -> dict[str, str]:
    col = Collector()
    files = sorted(os.path.join(UI_DIR, f) for f in os.listdir(UI_DIR) if f.endswith(".py"))
    trees = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            trees.append((path, ast.parse(f.read(), path)))
    wrappers = find_wrappers([t for _, t in trees])
    for path, tree in trees:
        module_tables(tree, path, col)
        FileScanner(path, col, wrappers).visit(tree)
    registries(col)
    return dict(sorted(col.strings.items(), key=lambda kv: kv[0].lower()))


def main() -> int:
    strings = extract()
    os.makedirs(I18N_DIR, exist_ok=True)
    data = {"_comment": "Generated by scripts/extract_strings.py - English UI keys and where they come from.",
            "count": len(strings),
            "strings": strings}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"{len(strings)} strings -> {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
