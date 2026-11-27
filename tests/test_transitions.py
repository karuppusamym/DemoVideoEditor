import numpy as np

from clipeditor import transitions as T
from clipeditor.audio_mix import mix
from clipeditor.editor import Editor
from clipeditor.model import MediaAsset


class FakeCache:
    sr = 1000

    def __init__(self, n):
        self.data = np.ones((n, 2), np.float32)

    def get(self, _aid):
        return self.data


def make_pair():
    ed = Editor()
    a = MediaAsset("a1", "x.mp4", "video", 10.0, 640, 360, 30, True, True)
    ed.project.assets[a.id] = a
    v1 = ed.project.video_tracks()[-1]
    c1 = ed.add_clip(a.id, v1.id, 0)
    c2 = ed.add_clip(a.id, v1.id, None)
    c3 = ed.add_clip(a.id, v1.id, None)
    return ed, v1, c1, c2, c3


def test_add_transition_overlaps_and_ripples():
    ed, tr, c1, c2, c3 = make_pair()
    assert ed.set_transition(c2.id, "dissolve", 1.0)
    assert c2.start == 9.0 and c3.start == 19.0
    assert T.prev_clip(tr, c2) is c1
    assert T.next_clip(tr, c1) is c2
    assert ed.project.duration == 29.0
    # change length, then remove -> back to hard cut
    ed.set_transition(c2.id, None, 2.0)
    assert c2.start == 8.0 and c3.start == 18.0
    ed.remove_transition(c2.id)
    assert c2.start == 10.0 and c3.start == 20.0 and c2.transition == 0
    ed.undo()
    tr = ed.project.video_tracks()[-1]
    assert sorted(c.start for c in tr.clips) == [0.0, 8.0, 18.0]


def test_transition_survives_overwrite_logic_and_resets_when_moved_apart():
    ed, tr, c1, c2, _ = make_pair()
    ed.set_transition(c2.id, "fadeblack", 1.0)
    tr.clear_range(c2.start, c2.end, keep_id=c2.id)  # what a drag-release does
    assert c1.end == 10.0  # A not trimmed by the allowed overlap
    c2.start += 3  # moved away
    ed.notify("edit")
    assert c2.transition == 0


def test_alpha_and_offsets():
    ed, tr, c1, c2, _ = make_pair()
    ed.set_transition(c2.id, "dissolve", 1.0)
    assert T.video_alpha(tr, c2, 9.0) == 0.0
    assert abs(T.video_alpha(tr, c2, 9.5) - 0.5) < 1e-6
    assert T.video_alpha(tr, c1, 9.5) == 1.0  # A stays under the dissolve
    ed.set_transition(c2.id, "fadeblack", 1.0)
    assert T.video_alpha(tr, c1, 9.5) < 1e-6 and T.video_alpha(tr, c2, 9.5) < 1e-6
    ed.set_transition(c2.id, "slideleft", 1.0)
    dx, dy = T.video_offset(tr, c2, 9.0, 100, 50)
    assert dx == 100 and dy == 0
    assert T.video_offset(tr, c2, 10.0, 100, 50) == (0.0, 0.0)
    x, y = T.ffmpeg_overlay_xy(tr, c2, 0, 0, 1920, 1080)
    assert "clip((t-9.0000)/1.0000,0,1)" in x and y == "0.00"
    fades = T.ffmpeg_alpha_fades(tr, c2)
    assert fades == []  # slides do not fade


def test_split_and_tail_drop_transition():
    ed, tr, c1, c2, _ = make_pair()
    ed.set_transition(c2.id, "dissolve", 1.0)
    ed.selection = {c2.id}
    ed.split(c2.start + 5)
    tr = ed.project.video_tracks()[-1]
    halves = sorted((c for c in tr.clips if c.asset_id == "a1" and 9 <= c.start < 19), key=lambda c: c.start)
    assert halves[0].transition == 1.0 and halves[1].transition == 0.0


def test_audio_crossfade():
    ed, tr, c1, c2, c3 = make_pair()
    ed.set_transition(c2.id, "dissolve", 1.0)
    ed.project.tracks = [tr]
    for c in (c1, c2, c3):
        c.volume = 0.5
    cache = FakeCache(20000)
    out = mix(ed.project, cache, 9.0, 1000)  # the whole overlap, sr=1000
    # equal-gain linear crossfade: sum stays ~constant (0.5) across the overlap
    assert np.allclose(out[:, 0], 0.5, atol=0.01)
    before = mix(ed.project, cache, 8.0, 500)
    assert np.allclose(before[:, 0], 0.5, atol=1e-6)


def test_transitions_everywhere():
    ed, tr, c1, c2, c3 = make_pair()
    ed.transitions_everywhere("slideup", 0.5)
    assert c2.transition == 0.5 and c3.transition == 0.5 and c1.transition == 0
    assert ed.project.duration == 29.0
