from clipeditor.editor import Editor
from clipeditor.model import MediaAsset, Project


def setup():
    ed = Editor()
    a = MediaAsset("a1", "x.mp4", "video", 10.0, 640, 360, 30, True, True)
    ed.project.assets[a.id] = a
    main = ed.main_track()
    clips = [ed.add_clip(a.id, main.id, None) for _ in range(3)]
    return ed, main, clips


def starts(track):
    return [round(c.start, 3) for c in track.sorted_clips()]


def test_delete_closes_gap_when_magnetic():
    ed, main, (c1, c2, c3) = setup()
    ed.set_magnetic(True)
    ed.selection = {c2.id}
    ed.delete_selected()
    assert starts(main) == [0, 10]


def test_off_by_default_leaves_gaps():
    ed, main, (c1, c2, c3) = setup()
    ed.selection = {c2.id}
    ed.delete_selected()
    assert starts(main) == [0, 20]


def test_insert_in_middle_pushes_right():
    ed, main, (c1, c2, c3) = setup()
    ed.set_magnetic(True)
    new = ed.add_clip("a1", main.id, 12.0)  # dropped inside c2 (first half) -> before c2
    assert new.start == 10.0
    assert starts(main) == [0, 10, 20, 30]
    assert ed.project.duration == 40


def test_transition_overlap_kept_and_undo():
    ed, main, (c1, c2, c3) = setup()
    ed.set_magnetic(True)
    ed.set_transition(c2.id, "dissolve", 1.0)
    assert starts(main) == [0, 9, 19]
    ed.undo()
    ed.undo()
    assert ed.project.active_timeline.magnetic is False


def test_roundtrip():
    ed, main, _ = setup()
    ed.set_magnetic(True)
    p = Project.from_dict(ed.project.to_dict())
    assert p.active_timeline.magnetic is True
