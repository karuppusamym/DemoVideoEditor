"""Font selection shared by the desktop UI and renderers."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase, QRawFont

UI_FAMILIES = ("Segoe UI", "Arial", "Noto Sans", "DejaVu Sans", "Liberation Sans")
MONOSPACE_FAMILIES = ("Consolas", "Cascadia Mono", "DejaVu Sans Mono", "Liberation Mono", "Courier New")
EMOJI_FAMILIES = ("Segoe UI Emoji", "Noto Color Emoji", "Apple Color Emoji", "Segoe UI Symbol", "Noto Emoji")


def prepare_qt_fonts() -> None:
    """Point headless Windows Qt at the installed fonts before creating an application."""
    if os.name == "nt" and os.environ.get("QT_QPA_PLATFORM", "").split(":", 1)[0] == "offscreen":
        directory = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        if directory.is_dir():
            os.environ.setdefault("QT_QPA_FONTDIR", str(directory))


@lru_cache(maxsize=1)
def _installed() -> dict[str, str]:
    return {family.casefold(): family for family in QFontDatabase.families()}


def _first_available(families) -> str | None:
    installed = _installed()
    return next((installed[name.casefold()] for name in families if name.casefold() in installed), None)


@lru_cache(maxsize=2)
def _interface_family(monospace: bool) -> str:
    wanted = MONOSPACE_FAMILIES if monospace else UI_FAMILIES
    system = QFontDatabase.SystemFont.FixedFont if monospace else QFontDatabase.SystemFont.GeneralFont
    fallback = QFontDatabase.systemFont(system).family()
    preferred = _first_available(wanted)
    for family in (preferred, fallback, *_installed().values()):
        if not family or (monospace and not QFontDatabase.isFixedPitch(family)):
            continue
        raw = QRawFont.fromFont(QFont(family, 10))
        if raw.isValid() and all(raw.supportsCharacter(character) for character in "Az09"):
            return family
    return preferred or fallback


def interface_font(*, monospace: bool = False) -> QFont:
    font = QFont(_interface_family(monospace), 10)
    font.setStyleHint(QFont.StyleHint.Monospace if monospace else QFont.StyleHint.SansSerif)
    return font


def title_font(family: str | None) -> QFont:
    installed = _first_available((family,)) if family else None
    return QFont(installed) if installed else interface_font()


def emoji_font() -> QFont:
    family = _first_available(EMOJI_FAMILIES)
    return QFont(family) if family else interface_font()
