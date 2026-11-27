import json
import shutil
from pathlib import Path

import pytest

from clipeditor.editor import Editor
from clipeditor.export_render import export_video
from clipeditor.model import Clip, Project

from .sample_media import SAMPLES


def test_failed_save_as_preserves_editor_state(tmp_path):
    editor = Editor()
    original = tmp_path / "project.clipeditor.json"
    editor.save(original)
    editor.add_text("Unsaved")
    contents = original.read_bytes()

    with pytest.raises(OSError):
        editor.save(tmp_path / "missing" / "new.clipeditor.json")

    assert editor.path == original and editor.dirty
    assert original.read_bytes() == contents


def test_project_relative_media_survives_moving_folder(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    media = root / "clip_a.mp4"
    shutil.copyfile(SAMPLES / media.name, media)
    editor = Editor()
    portable = editor.import_file(str(media))
    external = editor.import_file(str(SAMPLES / "tone.wav"))
    project = root / "project.clipeditor.json"
    editor.save(project)
    paths = {a["id"]: a["path"] for a in json.loads(project.read_text(encoding="utf-8"))["assets"]}
    assert paths == {portable.id: media.name, external.id: external.path}

    moved = root.rename(tmp_path / "moved")
    editor.open(moved / project.name)
    assert Path(editor.project.assets[portable.id].path) == moved / media.name
    assert editor.missing_media() == []


def test_relink_keeps_clip_and_asset_identity_and_supports_undo(tmp_path):
    old = tmp_path / "old"
    old.mkdir()
    original = old / "clip_a.mp4"
    shutil.copyfile(SAMPLES / original.name, original)
    editor = Editor()
    asset = editor.import_file(str(original))
    clip = editor.add_clip(asset.id)
    editor.set_clip_props(clip.id, in_point=1.0, out_point=4.0)
    moved = old.rename(tmp_path / "new")
    before = editor.project.to_dict()["tracks"]

    assert editor.relink_media(str(tmp_path)) == (0, [original.name])
    assert editor.relink_media(str(moved)) == (1, [])
    assert editor.project.to_dict()["tracks"] == before
    assert Path(editor.project.assets[asset.id].path) == moved / original.name
    editor.undo()
    assert [a.id for a in editor.missing_media()] == [asset.id]
    editor.redo()
    assert editor.missing_media() == []


@pytest.mark.parametrize("engine", ["graph", "render"])
@pytest.mark.parametrize("cancel", [False, True])
def test_export_preserves_existing_file_on_failure(tmp_path, engine, cancel):
    output = tmp_path / "existing.mp4"
    output.write_bytes(b"existing export")
    project = Project.new()
    project.width, project.height = 160, 90
    project.tracks[0].clips.append(Clip("title", "", 0.0, 0.0, 0.3, text="Preview"))
    progress = []

    with pytest.raises(RuntimeError, match="Export cancelled" if cancel else "FFmpeg failed"):
        export_video(project, str(output), engine=engine,
                     preset="ultrafast" if cancel else "invalid-preset",
                     progress=progress.append, cancelled=lambda: cancel and any(v > 0 for v in progress))

    assert output.read_bytes() == b"existing export"
    assert list(tmp_path.iterdir()) == [output]
    assert 1.0 not in progress
