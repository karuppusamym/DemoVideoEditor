import numpy as np

from clipeditor.editor import Editor
from clipeditor.model import MediaAsset
from clipeditor.silence import find_silences


def setup():
    ed = Editor()
    a = MediaAsset("a1", "x.mp4", "video", 10.0, 640, 360, 30, True, True)
    m = MediaAsset("m1", "song.wav", "audio", 30.0, has_audio=True)
    ed.project.assets.update({a.id: a, m.id: m})
    v = ed.project.video_tracks()[-1]
    c1 = ed.add_clip(a.id, v.id, 0)
    c2 = ed.add_clip(a.id, v.id, None)
    music = ed.add_clip(m.id, None, 0)
    return ed, v, c1, c2, music


def spans(track):
    return [(round(c.start, 3), round(c.end, 3), round(c.in_point, 3)) for c in track.sorted_clips()]


def test_remove_range_ripples_all_tracks():
    ed, v, c1, c2, music = setup()
    ed.remove_range(4.0, 6.0)
    assert spans(v) == [(0, 4, 0), (4, 8, 6), (8, 18, 0)]
    a1 = ed.project.audio_tracks()[0]
    assert spans(a1) == [(0, 4, 0), (4, 28, 6)]
    assert ed.project.duration == 28
    ed.undo()
    assert ed.project.duration == 30


def test_remove_frame():
    ed, v, *_ = setup()
    ed.remove_frame(5.0)
    assert abs(ed.project.video_tracks()[-1].sorted_clips()[1].in_point - (5 + 1 / 30)) < 1e-6
    assert abs(ed.project.duration - (30 - 1 / 30)) < 1e-6


def test_duplicate_range():
    ed, v, *_ = setup()
    ed.duplicate_range(2.0, 4.0)
    # original 0-4 stays whole, the copy of source 2-4 follows, then the rest of the clip
    assert spans(v) == [(0, 4, 0), (4, 6, 2), (6, 12, 4), (12, 22, 0)]
    assert ed.project.duration == 32
    assert len(ed.selection) == 2  # video copy + music copy


def test_insert_still(tmp_path):
    from clipeditor.textrender import ensure_qapp

    ensure_qapp()
    from PySide6.QtGui import QImage

    img = QImage(64, 36, QImage.Format.Format_RGB32)
    img.fill(0xFF3366AA)
    p = str(tmp_path / "f.png")
    img.save(p)
    ed, v, *_ = setup()
    clip = ed.insert_still(p, 3.0, 2.0)
    assert clip.start == 3.0 and clip.duration == 2.0
    assert spans(v)[:3] == [(0, 3, 0), (3, 5, 0), (5, 12, 3)]
    assert ed.project.duration == 32


def test_copy_paste():
    ed, v, c1, *_ = setup()
    ed.selection = {c1.id}
    assert ed.copy_selected() == 1
    assert ed.paste(25.0) == 1
    assert spans(v)[-1] == (25, 35, 0)


class Cache:
    sr = 1000

    def __init__(self):
        x = np.full((30000, 2), 0.3, np.float32)
        x[5000:8000] = 0  # 3 s of silence at 5..8 s of the source
        self.x = x

    def get(self, _):
        return self.x


def test_find_and_remove_silences():
    ed, v, c1, c2, music = setup()
    ed.project.tracks = [ed.project.audio_tracks()[0]]  # only the music track
    sil = find_silences(ed.project, Cache(), min_silence=0.5, keep_pad=0.1)
    assert len(sil) == 1
    a, b = sil[0]
    assert abs(a - 5.1) < 0.05 and abs(b - 7.9) < 0.05
    removed = ed.remove_ranges(sil)
    assert abs(removed - 2.8) < 0.05
