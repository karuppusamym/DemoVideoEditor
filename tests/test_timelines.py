import json
import os
import subprocess
import sys
from tempfile import TemporaryDirectory
from pathlib import Path

import av
import pytest

from clipeditor.api import EditorAPI
from clipeditor.editor import Editor
from clipeditor.export_render import export_video
from clipeditor.model import MediaAsset, Project

from . import sample_media


def test_legacy_project_opens_as_main_and_saves_all_timelines(tmp_path):
    original = Editor()
    first = original.add_text("Landscape", duration=3.0)
    legacy = original.project.to_dict()
    legacy.pop("timelines")
    legacy.pop("active_timeline_id")
    legacy["version"] = 1
    path = tmp_path / "edit.clipeditor.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")

    editor = Editor()
    editor.open(str(path))
    main = editor.project.active_timeline
    assert main.name == "Main" and editor.project.find_clip(first.id)[1].text == "Landscape"
    other = editor.create_timeline("Vertical")
    editor.project.width, editor.project.height, editor.project.fps = 720, 1280, 24.0
    editor.add_text("Vertical", duration=1.5)
    editor.save()

    reopened = Project.load(path)
    assert list(reopened.timelines) == [main.id, other.id]
    assert reopened.active_timeline_id == other.id
    assert (reopened.width, reopened.height, reopened.fps, reopened.duration) == (720, 1280, 24.0, 1.5)
    assert reopened.timelines[main.id].duration == 3.0
    assert Project.from_dict(reopened.to_dict()).to_dict() == reopened.to_dict()


def test_duplicate_has_independent_clips_and_history(tmp_path):
    editor = Editor()
    media = editor.import_file(str(sample_media.SAMPLES / "clip_a.mp4"))
    first = editor.add_clip(media.id)
    editor.set_keyframes(first.id, "opacity", [(0.0, 0.5, "hold")])
    main = editor.project.active_timeline.id
    editor.mark_in, editor.mark_out = 1.0, 2.0
    copy = editor.create_timeline(duplicate=True)
    assert copy.name == "Main copy" and editor.range() is None and not editor.selection
    duplicate = editor.project.video_tracks()[-1].clips[0]
    assert duplicate.id != first.id and duplicate.asset_id == first.asset_id
    editor.set_keyframes(duplicate.id, "opacity", [(0.0, 1.0, "linear")])
    assert first.keyframes["opacity"] == [[0.0, 0.5, "hold"]]

    editor.rename_timeline(copy.id, "Vertical cut")
    undo_size = len(editor._undo)
    editor.switch_timeline("main")
    assert len(editor._undo) == undo_size
    editor.undo()
    assert editor.project.active_timeline_id == copy.id and editor.project.active_timeline.name == "Main copy"
    editor.redo()
    editor.switch_timeline("vertical cut")
    editor.save(tmp_path / "edit.clipeditor.json")
    editor.delete_timeline(copy.id)
    assert editor.project.active_timeline_id == main
    with pytest.raises(ValueError, match="at least one"):
        editor.delete_timeline(main)
    editor.undo()
    assert editor.project.active_timeline_id == copy.id and editor.dirty
    assert editor.project.find_clip(duplicate.id)[1] is not None


def test_removing_shared_media_clears_all_timelines_and_undo_restores_it():
    editor = Editor()
    media = editor.import_file(str(sample_media.SAMPLES / "clip_a.mp4"))
    editor.add_clip(media.id)
    editor.create_timeline(duplicate=True)
    editor.remove_asset(media.id)
    assert media.id not in editor.project.assets
    assert all(t.duration == 0 for t in editor.project.timelines.values())
    editor.undo()
    assert media.id in editor.project.assets
    assert all(t.duration == pytest.approx(media.duration) for t in editor.project.timelines.values())


def test_named_timeline_cli_and_api_export_use_independent_canvas(tmp_path):
    editor = Editor()
    editor.project.width, editor.project.height = 320, 180
    editor.add_text("Landscape", duration=1.0)
    api = EditorAPI(editor)
    vertical = api.call("create_timeline", {"name": "Vertical Cut"})["timeline"]
    editor.project.width, editor.project.height, editor.project.fps = 90, 160, 24.0
    editor.add_text("Vertical", duration=0.5)
    editor.switch_timeline("Main")
    assert api.call("get_project_summary")["active_timeline"]["name"] == "Main"
    assert [t["active"] for t in api.call("list_timelines")["timelines"]] == [True, False]
    assert "error" in api.call("create_timeline", {"name": "vertical cut"})
    project_path = tmp_path / "edit.clipeditor.json"
    editor.save(str(project_path))

    api_output = tmp_path / "API.mp4"
    result = api.call("export_video", {"path": str(api_output), "timeline": vertical["id"], "quality": "draft"})
    assert result.get("ok"), result
    cli_output = tmp_path / "CLI.mp4"
    subprocess.run([sys.executable, "-m", "clipeditor", "export", str(project_path), str(cli_output),
                    "--timeline", "Vertical Cut", "--preset", "ultrafast"],
                   check=True, capture_output=True, timeout=30, env={**os.environ, "QT_QPA_PLATFORM": "offscreen"})
    for output in (api_output, cli_output):
        with av.open(str(output)) as video:
            stream = video.streams.video[0]
            assert (stream.width, stream.height, stream.average_rate) == (90, 160, 24)
            assert video.duration / av.time_base == pytest.approx(0.5, abs=0.08)
    assert editor.project.active_timeline.name == Project.load(project_path).active_timeline.name == "Main"


def test_render_export_ignores_offline_media_used_only_in_inactive_timeline(tmp_path):
    editor = Editor()
    editor.project.assets["offline"] = MediaAsset("offline", str(tmp_path / "gone.wav"), "audio", 2.0,
                                                  has_audio=True)
    editor.add_clip("offline")
    editor.create_timeline("Titles")
    editor.project.width, editor.project.height = 160, 90
    editor.add_text("Title", duration=0.25)

    output = tmp_path / "titles.mp4"
    assert export_video(editor.project, str(output), engine="render", preset="ultrafast") == "render"
    assert output.is_file()


def test_timeline_selector_updates_preview_and_autosaves_empty_active_timeline():
    subprocess.run([sys.executable, "-m", "tests.test_timelines"], check=True, timeout=30,
                   env={**os.environ, "QT_QPA_PLATFORM": "offscreen"})


def _exercise_selector():
    if os.name == "nt":
        os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ["WINDIR"], "Fonts"))
    from PySide6.QtWidgets import QApplication

    from clipeditor.ui.main_window import MainWindow

    app = QApplication([])
    window = MainWindow()
    window.show()
    window.ed.add_text("Main", duration=4.0)
    window.seek(2.0)
    main = window.ed.project.active_timeline_id
    vertical = window.ed.create_timeline("Vertical")
    window.set_canvas(720, 1280)
    assert window.timeline_choice.currentText() == "Vertical" and window.playhead == 0.0
    assert window.preview.aspect == pytest.approx(9 / 16)
    with TemporaryDirectory() as tmp:
        window.autosave_path = str(Path(tmp) / "autosave.clipeditor.json")
        window.autosave()
        assert Project.load(window.autosave_path).timelines[main].duration == 4.0
    window.timeline_choice.setCurrentIndex(window.timeline_choice.findData(main))
    window.timeline_choice.activated.emit(window.timeline_choice.currentIndex())
    app.processEvents()
    assert window.ed.project.active_timeline_id == main and window.playhead == pytest.approx(2.0)
    assert window.preview.aspect == pytest.approx(16 / 9)
    window.ed.rename_timeline(vertical.id, "Short")
    assert window.timeline_choice.findText("Short") >= 0
    window.loader.shutdown()
    window.comp.close()
    window.hide()


if __name__ == "__main__":
    _exercise_selector()
