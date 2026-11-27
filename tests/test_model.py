import pytest

from clipeditor.editor import Editor
from clipeditor.model import Clip, MediaAsset, Project, Track


def make():
    ed = Editor()
    a = MediaAsset("a1", "x.mp4", "video", 10.0, 640, 360, 30, True, True)
    ed.project.assets[a.id] = a
    return ed, a


def test_add_append_and_overwrite():
    ed, a = make()
    v1 = ed.project.video_tracks()[1]
    c1 = ed.add_clip(a.id, v1.id, 0)
    c2 = ed.add_clip(a.id, v1.id, None)
    assert c2.start == 10.0
    ed.add_clip(a.id, v1.id, 5.0)  # overwrites 5..15
    spans = sorted((round(c.start, 3), round(c.end, 3)) for c in v1.clips)
    assert spans == [(0, 5), (5, 15), (15, 20)]


def test_split_and_undo_redo():
    ed, a = make()
    v1 = ed.project.video_tracks()[1]
    ed.add_clip(a.id, v1.id, 0)
    ed.selection.clear()
    ed.split(4.0)
    v1 = ed.project.video_tracks()[1]
    l, r = v1.sorted_clips()
    assert (l.in_point, l.out_point, r.start, r.in_point) == (0, 4, 4, 4)
    ed.undo()
    assert len(ed.project.video_tracks()[1].clips) == 1
    ed.redo()
    assert len(ed.project.video_tracks()[1].clips) == 2


def test_ripple_delete():
    ed, a = make()
    v1 = ed.project.video_tracks()[1]
    first = ed.add_clip(a.id, v1.id, 0)
    ed.add_clip(a.id, v1.id, None)
    ed.selection = {first.id}
    ed.delete_selected(ripple=True)
    assert [c.start for c in ed.project.video_tracks()[1].clips] == [0.0]


def test_detach_audio():
    ed, a = make()
    c = ed.add_clip(a.id)
    ed.detach_audio()
    assert c.volume == 0
    assert len(ed.project.audio_tracks()[0].clips) == 1


def test_roundtrip():
    p = Project.new()
    p.tracks[0].clips.append(Clip("c", "a", 1, 2, 3))
    q = Project.from_dict(p.to_dict())
    assert q.to_dict() == p.to_dict()


def test_failed_save_preserves_previous_project(tmp_path):
    path = tmp_path / "edit.clipeditor.json"
    project = Project.new()
    project.save(path)
    original = path.read_bytes()
    project.tracks[0].clips.append(Clip("c", "", 0, 0, 3, text=object()))

    with pytest.raises(TypeError):
        project.save(path)

    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
