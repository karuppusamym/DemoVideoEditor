import os
import subprocess
import sys

import numpy as np
import pytest

from clipeditor import animate
from clipeditor.audio_mix import mix
from clipeditor.editor import Editor
from clipeditor.model import Clip, MediaAsset, Project


def test_audio_volume_curve_applies_within_a_mix_block():
    project = Project.new()
    project.assets["tone"] = MediaAsset("tone", "tone.wav", "audio", 2, has_audio=True)
    clip = Clip("c", "tone", 0.0, 0.0, 2.0, volume=0.0)
    project.audio_tracks()[0].clips.append(clip)
    animate.set_keyframe(clip, "volume", 0, 0, "ease_in")
    animate.set_keyframe(clip, "volume", 1, 1, "hold")
    animate.set_keyframe(clip, "volume", 1.5, 0.25)

    class Cache:
        sr = 1000

        def get(self, asset_id):
            return np.full((2000, 2), 0.5, dtype=np.float32)

    out = mix(project, Cache(), 0.0, 2000)
    expected = [0.5 * animate.volume_at(clip, i / 1000) for i in range(2000)]
    assert out[:, 0] == pytest.approx(expected, abs=1e-6)


def test_split_preserves_curve_across_cut_and_edits_are_independent():
    editor = Editor()
    left = editor.add_text("Text", duration=4.0)
    editor.set_keyframes(left.id, "scale", [(1.0, 1.0, "ease"), (3.0, 2.0, "linear")])
    times = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    expected = [animate.value_at(left, "scale", time) for time in times]

    editor.split(2.0)
    right = editor.project.video_tracks()[0].sorted_clips()[1]
    actual = [animate.value_at(left, "scale", time) if time < 2.0
              else animate.value_at(right, "scale", time - 2.0) for time in times]
    assert actual == pytest.approx(expected)
    editor.set_keyframes(right.id, "scale", [(0.0, 4.0, "hold")])
    assert animate.value_at(left, "scale", 1.5) == pytest.approx(expected[2])


def test_detaching_audio_moves_volume_keyframes():
    editor = Editor()
    editor.project.assets["video"] = MediaAsset("video", "source.mp4", "video", 4.0, has_video=True,
                                                has_audio=True)
    clip = editor.add_clip("video")
    curve = [(0.0, 0.0, "ease"), (2.0, 1.0, "hold")]
    editor.set_keyframes(clip.id, "volume", curve)
    clip.volume = 0.0
    editor.detach_audio()

    detached = editor.project.audio_tracks()[0].clips[0]
    assert animate.keyframes_for(detached, "volume") == curve
    assert animate.volume_at(clip, 2.0) == 0.0


def test_keyframe_editor_gestures():
    # Other render tests may already have a QGuiApplication in the pytest process.
    subprocess.run([sys.executable, "-m", "tests.test_keyframe_editor"], check=True, timeout=30,
                   env={**os.environ, "QT_QPA_PLATFORM": "offscreen"})


def _exercise_graph():
    if os.name == "nt":
        os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ["WINDIR"], "Fonts"))
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from clipeditor.ui.main_window import MainWindow, apply_dark_theme

    app = QApplication([])
    app.setFont(QFont("Segoe UI", 9))
    apply_dark_theme(app)
    window = MainWindow()
    window.resize(1280, 780)
    first = window.ed.add_text("A", duration=4)
    second = window.ed.add_text("B", start=4, duration=2)
    window.ed.selection = {first.id}
    window.ed.notify("selection")
    window.inspector.tabs.setCurrentWidget(window.inspector.tab_keys)
    controls = window.inspector.keyframe_editor
    controls.property_choice.setCurrentIndex(controls.property_choice.findData("opacity"))
    window.show()
    app.processEvents()
    assert controls.plot.isVisible()

    window.seek(0.0)
    controls.add_key.click()
    plot = controls.plot
    point = plot._position(3, 0.8).toPoint()
    QTest.mouseDClick(plot, Qt.MouseButton.LeftButton, pos=point)
    before_drag = animate.keyframes_for(first, "opacity")
    assert len(before_drag) == 2 and before_drag[1][0] == pytest.approx(3.0, abs=0.04)
    old = plot._position(*before_drag[1][:2]).toPoint()
    new = plot._position(2, 0.55).toPoint()
    history = len(window.ed._undo)
    QTest.mousePress(plot, Qt.MouseButton.LeftButton, pos=old)
    QTest.mouseMove(plot, new)
    QTest.mouseRelease(plot, Qt.MouseButton.LeftButton, pos=new)
    after_drag = animate.keyframes_for(first, "opacity")
    assert after_drag[1][:2] == pytest.approx((2, 0.55), abs=0.04)
    assert len(window.ed._undo) == history + 1
    window.ed.undo()
    assert animate.keyframes_for(window.inspector.clip(), "opacity") == before_drag
    window.ed.redo()
    assert animate.keyframes_for(window.inspector.clip(), "opacity") == after_drag

    QTest.mouseClick(plot, Qt.MouseButton.LeftButton, pos=plot._position(*after_drag[1][:2]).toPoint())
    controls.easing.setCurrentIndex(controls.easing.findData("hold"))
    controls.easing.activated.emit(controls.easing.currentIndex())
    assert animate.keyframes_for(window.inspector.clip(), "opacity")[1][2] == "hold"
    controls.copy_curve.click()
    window.ed.selection = {second.id}
    window.ed.notify("selection")
    controls.paste_curve.click()
    keys = animate.keyframes_for(window.inspector.clip(), "opacity")
    assert len(keys) == 2 and keys[1][0] == pytest.approx(after_drag[1][0] / 2)
    assert keys[1][2] == "hold"
    QTest.mouseClick(plot, Qt.MouseButton.LeftButton, pos=plot._position(*keys[1][:2]).toPoint())
    controls.remove_key.click()
    assert len(animate.keyframes_for(window.inspector.clip(), "opacity")) == 1
    window.loader.shutdown()
    window.comp.close()
    window.hide()


if __name__ == "__main__":
    _exercise_graph()
