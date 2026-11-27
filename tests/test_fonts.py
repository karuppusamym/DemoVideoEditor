import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("mode", ["desktop", "export"])
def test_fonts_resolve_without_a_preconfigured_font_directory(mode):
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    env.pop("QT_QPA_FONTDIR", None)
    subprocess.run([sys.executable, "-m", "tests.test_fonts", mode], env=env, check=True, timeout=30)


def check_fonts(mode):
    from PySide6.QtGui import QFontDatabase, QFontInfo, QGuiApplication, QImage, QRawFont

    from clipeditor.fonts import interface_font, prepare_qt_fonts, title_font
    from clipeditor.model import Clip
    from clipeditor.textrender import ensure_qapp, render_text, render_text_element

    if mode == "desktop":
        from PySide6.QtWidgets import QApplication

        from clipeditor.ui.main_window import MainWindow, apply_dark_theme

        prepare_qt_fonts()
        app = QApplication([])
        apply_dark_theme(app)
        window = MainWindow()
        assert QFontInfo(window.time_lbl.font()).fixedPitch()
    else:
        ensure_qapp()
        app = QGuiApplication.instance()

    ui = QFontInfo(app.font())
    raw = QRawFont.fromFont(app.font())
    assert raw.isValid() and all(raw.supportsCharacter(c) for c in "Play 02:19")
    installed = set(QFontDatabase.families())
    if "Segoe UI" in installed:
        assert ui.family() == "Segoe UI"
    chosen = "Arial" if "Arial" in installed else interface_font().family()
    assert QFontInfo(title_font(chosen)).family() == chosen
    assert title_font("ClipEditor Font That Does Not Exist").family() == ui.family()

    clip = Clip("title", "", 0, 0, 1, text="Cut 02:19", font_family=chosen)
    for image in (render_text_element(clip, 1.0), render_text(clip, 640, 360, 360)):
        pixels = image.convertToFormat(QImage.Format.Format_RGBA8888)
        assert any(pixels.constBits()[3::4])
    clip.font_family = "ClipEditor Font That Does Not Exist"
    missing = render_text_element(clip, 1.0)
    clip.font_family = ui.family()
    assert missing == render_text_element(clip, 1.0)
    if mode == "desktop":
        window.loader.shutdown()
        window.comp.close()


if __name__ == "__main__":
    check_fonts(sys.argv[1])
