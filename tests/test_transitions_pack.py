"""The render-engine transition pack: every kind renders, is deterministic and is fast enough."""
import time

import numpy as np
import pytest
from PySide6.QtGui import QColor, QImage

from clipeditor import transitions as T
from clipeditor.editor import Editor
from clipeditor.render import Compositor, qimage_to_rgba

NEW_KINDS = [k for k in T.KINDS if k not in T.LEGACY_KINDS]
FLASH = {"flashwhite"}


def _png(path, color):
    img = QImage(320, 180, QImage.Format.Format_RGB32)
    img.fill(QColor(color))
    assert img.save(str(path))
    return str(path)


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    d = tmp_path_factory.mktemp("tpack")
    ed = Editor()
    ra = ed.import_file(_png(d / "red.png", "#ff0000"))
    ba = ed.import_file(_png(d / "blue.png", "#0000ff"))
    v1 = ed.project.video_tracks()[-1]
    a = ed.add_clip(ra.id, v1.id, 0)
    b = ed.add_clip(ba.id, v1.id, None)
    assert ed.set_transition(b.id, "dissolve", 1.0)
    return ed, v1, a, b


def _frame(comp, ed, t, w=320, h=180):
    return qimage_to_rgba(comp.render(ed.project, t, w, h))[..., :3].astype(np.float32)


def test_kinds_and_categories():
    assert len(T.KINDS) >= 32
    seen = [k for ks in T.CATEGORIES.values() for k in ks]
    assert sorted(seen) == sorted(T.KINDS)  # every kind exactly once
    assert set(T.CATEGORIES) == {"Basic", "Wipes", "Shapes", "Motion", "Effects"}
    for k in T.KINDS:
        assert len(T.ICONS[k]) == 1
        assert T.needs_render(k) == (k not in T.LEGACY_KINDS)
        assert not (k in NEW_KINDS and k.startswith("slide"))  # video_offset must not catch them


@pytest.mark.parametrize("kind", NEW_KINDS)
def test_new_kind_renders(pair, kind):
    ed, tr, a, b = pair
    ed.set_transition(b.id, kind, 1.0)
    assert b.transition == 1.0 and b.start == a.end - 1.0
    comp = Compositor()
    # outside the overlap: only A, then only B, and no extra fx
    before, after = _frame(comp, ed, b.start - 0.2), _frame(comp, ed, a.end + 0.2)
    assert np.allclose(before, [255, 0, 0], atol=2)
    assert np.allclose(after, [0, 0, 255], atol=2)
    assert T.frame_fx(tr, a, b.start - 0.2, 320, 180) is None
    assert T.frame_fx(tr, b, a.end + 0.2, 320, 180) is None
    # video_alpha / video_offset leave new kinds alone (no double application)
    mid = b.start + 0.5
    assert T.video_alpha(tr, b, mid) == 1.0 and T.video_alpha(tr, a, mid) == 1.0
    assert T.video_offset(tr, b, mid, 320, 180) == (0.0, 0.0)
    f1, f5, f9 = (_frame(comp, ed, b.start + p) for p in (0.1, 0.5, 0.9))
    r1, b1 = f1[..., 0].mean(), f1[..., 2].mean()
    r9, b9 = f9[..., 0].mean(), f9[..., 2].mean()
    assert r1 > b1, f"{kind}: A should dominate early ({r1:.0f} vs {b1:.0f})"
    assert b9 > r9, f"{kind}: B should dominate late ({r9:.0f} vs {b9:.0f})"
    if kind in FLASH:
        assert f5.mean() > 200 and f5.mean() > f1.mean() and f5.mean() > f9.mean()
    else:
        assert f5[..., 0].mean() > 30 and f5[..., 2].mean() > 30, f"{kind}: mid frame lacks a colour"
    # determinism (fresh compositor too)
    again = Compositor()
    for p in (0.1, 0.5, 0.9):
        x = comp.render(ed.project, b.start + p, 320, 180)
        y = again.render(ed.project, b.start + p, 320, 180)
        assert qimage_to_rgba(x).tobytes() == qimage_to_rgba(y).tobytes()
    assert comp.last_error is None, comp.last_error


def test_timing_640x360(pair):
    ed, tr, a, b = pair
    comp = Compositor()
    slow = {}
    for kind in NEW_KINDS:
        ed.set_transition(b.id, kind, 1.0)
        for p in (0.3, 0.5):  # warm caches
            comp.render(ed.project, b.start + p, 640, 360)
        best = []
        for p in (0.25, 0.5, 0.75):  # best of 3 per frame: robust against a busy machine
            runs = []
            for _ in range(3):
                t0 = time.perf_counter()
                comp.render(ed.project, b.start + p, 640, 360)
                runs.append(time.perf_counter() - t0)
            best.append(min(runs))
        ms = max(best) * 1000
        if ms > 25:
            slow[kind] = round(ms, 1)
    assert not slow, slow


def test_mask_speed():
    for kind in sorted(T._MASK_KINDS):
        fx = T._mask_fx(kind, 0.5)
        fx.mask(360, 640)
        t0 = time.perf_counter()
        for _ in range(5):
            fx.mask(360, 640)
        assert (time.perf_counter() - t0) / 5 < 0.005, kind


def test_legacy_kinds_untouched(pair):
    ed, tr, a, b = pair
    for kind in T.LEGACY_KINDS:
        ed.set_transition(b.id, kind, 1.0)
        for p in (0.0, 0.25, 0.5, 0.9):
            assert T.frame_fx(tr, b, b.start + p, 320, 180) is None
            assert T.frame_fx(tr, a, b.start + p, 320, 180) is None
    ed.set_transition(b.id, "dissolve", 1.0)
    assert abs(T.video_alpha(tr, b, b.start + 0.5) - 0.5) < 1e-6
    assert T.ffmpeg_alpha_fades(tr, b) == ["fade=t=in:st=0:d=1.0000:alpha=1"]


def test_chain_composes_in_and_out(pair, tmp_path):
    """A clip that is both incoming and outgoing gets both effects."""
    ed, tr, a, b = pair
    c = ed.add_clip(b.asset_id, tr.id, None)
    try:
        ed.set_transition(b.id, "pushleft", 1.0)
        ed.set_transition(c.id, "zoomthrough", 1.0)
        fx = T.frame_fx(tr, b, b.start + 0.5, 320, 180)
        assert fx is not None and fx.dx > 0
        fx = T.frame_fx(tr, b, c.start + 0.5, 320, 180)
        assert fx is not None and fx.scale > 1
    finally:
        ed.remove_transition(c.id)
        tr.clips.remove(c)
