import os
import subprocess
import sys

import av
import numpy as np
import pytest

from clipeditor.api import EditorAPI
from clipeditor.audio_mix import mix
from clipeditor.editor import Editor
from clipeditor.export_render import export_video
from clipeditor.media import AudioCache
from clipeditor.model import Project
from clipeditor.render import Compositor
from clipeditor.textrender import ensure_qapp

from . import sample_media


def test_nested_sequences_reject_cycles_keep_linked_length_and_support_undo(tmp_path):
    editor = Editor()
    parent = editor.project.active_timeline
    original = editor.add_text("Parent", duration=2)
    source = editor.create_timeline("Source")
    editor.add_text("Child", duration=1)
    editor.switch_timeline(parent.id)
    nested = editor.add_sequence(source.id, start=0.5)
    assert editor.project.find_clip(original.id)[1] is original
    assert nested.sequence_id == source.id and nested.duration == 1
    assert editor.copy_selected() == 1
    editor.switch_timeline(source.id)
    assert editor.paste(0) == 0
    with pytest.raises(ValueError, match="circular"):
        editor.add_sequence(parent.id)
    with pytest.raises(ValueError, match="nested clips"):
        editor.delete_timeline(source.id)
    editor.remove_range(0.5, 1)
    assert source.duration == 0.5
    editor.switch_timeline(parent.id)
    assert editor.project.find_clip(nested.id)[1].duration == 1
    path = tmp_path / "nested.clipeditor.json"
    editor.save(path)
    reopened = Project.load(path)
    assert reopened.find_clip(nested.id)[1].sequence_id == source.id
    editor.undo()
    assert editor.project.timelines[source.id].duration == 1


def test_nested_video_composites_transparently_and_updates_from_source():
    ensure_qapp()
    editor = Editor()
    parent = editor.project.active_timeline
    editor.project.width, editor.project.height = 160, 90
    editor.add_shape({"shape": "rect", "fill": "#0000ff"}, duration=2, scale=4)
    source = editor.create_timeline("Title")
    child = editor.add_shape({"shape": "rect", "fill": "#ff0000"}, duration=1, scale=0.5)
    editor.switch_timeline(parent.id)
    editor.add_sequence(source.id, start=0)
    comp = Compositor()
    try:
        before = comp.render(editor.project, 0.5, 160, 90)
        editor.switch_timeline(source.id)
        editor.set_clip_props(child.id, shape={"shape": "rect", "fill": "#00ff00"})
        editor.switch_timeline(parent.id)
        after = comp.render(editor.project, 0.5, 160, 90)
        assert before.pixelColor(80, 45).red() > 200
        assert after.pixelColor(80, 45).green() > 200
        assert after.pixelColor(2, 2).blue() > 200
    finally:
        comp.close()


def test_nested_audio_speed_and_recursive_export(tmp_path):
    editor = Editor()
    parent = editor.project.active_timeline
    parent.width, parent.height = 160, 90
    middle = editor.create_timeline("Middle")
    source = editor.create_timeline("Source")
    asset = editor.import_file(str(sample_media.SAMPLES / "tone.wav"))
    child = editor.add_clip(asset.id)
    editor.set_clip_props(child.id, out_point=1.0)
    editor.add_text("Nested", duration=1.0, font_size=30)
    editor.switch_timeline(middle.id)
    editor.add_sequence(source.id)
    editor.switch_timeline(parent.id)
    api = EditorAPI(editor)
    result = api.call("add_sequence", {"timeline": middle.id})
    assert result["ok"], result
    nested = result["clip"]["id"]
    editor.set_clip_props(nested, speed=2.0, volume=0.5)
    assert editor.project.duration == 0.5
    assert list(editor.project.timeline_assets()) == [asset]
    cache = AudioCache()
    cache.load(asset)
    original = cache.get(asset.id)
    block = mix(editor.project, cache, 0.0, 5000)
    np.testing.assert_allclose(block, original[:10000:2] * 0.5, atol=1e-6)

    destination = tmp_path / "nested.mp4"
    assert export_video(editor.project, str(destination), preset="ultrafast") == "render"
    with av.open(str(destination)) as output:
        assert output.streams.audio and output.streams.video
        pixels = next(output.decode(video=0)).to_ndarray(format="gray")
        assert pixels.max() > 150
    with av.open(str(destination)) as output:
        audio = next(output.decode(audio=0)).to_ndarray()
        assert np.abs(audio).max() > 0.02


def test_loading_missing_sequence_source_fails_cleanly():
    editor = Editor()
    editor.add_text("Title")
    data = editor.project.to_dict()
    data["timelines"][0]["tracks"][0]["clips"][0]["sequence_id"] = "missing"
    with pytest.raises(ValueError, match="missing source"):
        Project.from_dict(data)


def test_double_click_opens_source_timeline():
    subprocess.run([sys.executable, "-m", "tests.test_nested_sequences"], check=True, timeout=30,
                   env={**os.environ, "QT_QPA_PLATFORM": "offscreen"})


if __name__ == "__main__":
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
    window.show()
    parent = window.ed.project.active_timeline
    child = window.ed.create_timeline("Child")
    window.ed.add_text("Title", duration=2)
    window.ed.switch_timeline(parent.id)
    clip = window.ed.add_sequence(child.id, start=0.0)
    app.processEvents()
    canvas = window.timeline.canvas
    row = next(row for track, row in canvas.rows() if clip in track.clips)
    position = canvas.clip_rect(row, clip).center().toPoint()
    QTest.mouseDClick(canvas, Qt.MouseButton.LeftButton, pos=position)
    assert window.ed.project.active_timeline_id == child.id
    assert window.timeline_choice.currentText() == "Child"
    window.loader.shutdown()
    window.comp.close()
    window.hide()
