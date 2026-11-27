import math

import numpy as np
import pytest

from clipeditor import animate
from clipeditor.model import Clip, Project, new_id


def clip(dur=4.0, **kw):
    c = Clip(new_id(), "", 0.0, 0.0, dur, text="Hi")
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def finite(p):
    return all(math.isfinite(v) for v in (p.scale, p.pos_x, p.pos_y, p.rotation, p.opacity))


# ---- keyframes -------------------------------------------------------------
def test_linear_midpoint_and_clamping():
    c = clip()
    animate.set_keyframe(c, "pos_x", 1.0, 0.0, "linear")
    animate.set_keyframe(c, "pos_x", 3.0, 1.0, "linear")
    assert animate.value_at(c, "pos_x", 2.0) == pytest.approx(0.5)
    assert animate.value_at(c, "pos_x", 0.0) == 0.0  # before first
    assert animate.value_at(c, "pos_x", 3.9) == 1.0  # after last
    assert animate.resolve(c, 2.0).pos_x == pytest.approx(0.5)


@pytest.mark.parametrize("e", animate.EASINGS)
def test_easing_endpoints(e):
    c = clip()
    animate.set_keyframe(c, "scale", 0.0, 1.0, e)
    animate.set_keyframe(c, "scale", 2.0, 3.0, e)
    assert animate.value_at(c, "scale", 0.0) == pytest.approx(1.0)
    assert animate.value_at(c, "scale", 2.0) == pytest.approx(3.0)
    assert math.isfinite(animate.value_at(c, "scale", 1.0))


def test_ease_curve_shape():
    assert animate.ease(0.5, "ease") == pytest.approx(0.5)
    assert animate.ease(0.25, "ease_in") < 0.25 < animate.ease(0.25, "ease_out")
    assert max(animate.ease(u / 100, "elastic") for u in range(101)) > 1.0  # overshoots


def test_hold():
    c = clip()
    animate.set_keyframe(c, "opacity", 0.0, 0.2, "hold")
    animate.set_keyframe(c, "opacity", 2.0, 0.9, "linear")
    assert animate.value_at(c, "opacity", 1.99) == pytest.approx(0.2)
    assert animate.value_at(c, "opacity", 2.0) == pytest.approx(0.9)


def test_set_replace_remove():
    c = clip()
    animate.set_keyframe(c, "rotation", 1.0, 10)
    animate.set_keyframe(c, "rotation", 1.0 + 1 / 120, 20)  # within one frame -> replace
    assert len(c.keyframes["rotation"]) == 1
    assert animate.keyframe_at(c, "rotation", 1.0)[1] == 20
    animate.set_keyframe(c, "rotation", 2.0, 30)
    animate.set_keyframe(c, "scale", 0.5, 2)
    animate.set_keyframe(c, "scale", 2.0 + 1 / 200, 2)
    assert animate.all_keyframe_times(c) == pytest.approx([0.5, 1.0, 2.0])
    assert animate.keyframe_at(c, "rotation", 1.5) is None
    assert animate.remove_keyframe(c, "rotation", 1.0)
    assert not animate.remove_keyframe(c, "rotation", 1.0)
    assert len(c.keyframes["rotation"]) == 1
    animate.clear_keyframes(c, "scale")
    assert "scale" not in c.keyframes
    animate.clear_keyframes(c)
    assert c.keyframes == {}
    assert animate.value_at(c, "rotation", 1.0) == c.rotation
    with pytest.raises(ValueError):
        animate.set_keyframe(c, "bogus", 0, 1)


def test_keyframes_override_base():
    c = clip(scale=5.0)
    animate.set_keyframe(c, "scale", 1.0, 2.0)
    assert animate.resolve(c, 0.0).scale == pytest.approx(2.0)


def test_volume_keyframes():
    c = clip(volume=1.0)
    c.start = 10.0
    assert animate.volume_at(c, 11.0) == 1.0
    animate.set_keyframe(c, "volume", 0.0, 0.0, "linear")
    animate.set_keyframe(c, "volume", 2.0, 2.0, "linear")
    assert animate.volume_at(c, 11.0) == pytest.approx(1.0)
    assert animate.volume_at(c, 10.0) == pytest.approx(0.0)
    assert animate.volume_at(c, 13.0) == pytest.approx(2.0)


# ---- presets ---------------------------------------------------------------
def test_tables_consistent():
    assert len(animate.ANIMATIONS["in"]) >= 14
    assert len(animate.ANIMATIONS["out"]) >= 14
    assert len(animate.ANIMATIONS["loop"]) >= 10
    for k in ("in", "out", "loop"):
        assert set(animate.ANIMATIONS[k]) == set(animate.PRESET_INFO[k])


ALL = [(k, n) for k in ("in", "out", "loop") for n in animate.ANIMATIONS[k]]


@pytest.mark.parametrize("kind,name", ALL)
def test_preset_sane(kind, name):
    c = clip(dur=4.0)
    setattr(c, "anim_" + kind, name)
    for t in np.linspace(0, 4.0, 81):
        p = animate.resolve(c, float(t))
        assert finite(p)
        assert 0.0 <= p.opacity <= 1.0
        assert 0.0 <= p.scale < 3.0
        assert abs(p.pos_x) <= 1.5 and abs(p.pos_y) <= 1.5
        assert abs(p.rotation) <= 400
    assert animate.active(c)


@pytest.mark.parametrize("name", list(animate.ANIMATIONS["in"]))
def test_in_ends_at_identity(name):
    c = clip(anim_in=name, anim_in_dur=0.5)
    for t in (0.5, 1.0, 2.0):
        p = animate.resolve(c, t)
        assert (p.scale, p.pos_x, p.pos_y, p.rotation, p.opacity) == pytest.approx((1, 0, 0, 0, 1))


@pytest.mark.parametrize("name", list(animate.ANIMATIONS["out"]))
def test_out_starts_at_identity(name):
    c = clip(anim_out=name, anim_out_dur=0.5)
    for t in (0.0, 2.0, 3.5):
        p = animate.resolve(c, t)
        assert (p.scale, p.pos_x, p.pos_y, p.rotation, p.opacity) == pytest.approx((1, 0, 0, 0, 1))


def test_fade_in_out_endpoints():
    c = clip(anim_in="fade", anim_out="fade")
    assert animate.resolve(c, 0.0).opacity == pytest.approx(0.0)
    assert animate.resolve(c, 0.25).opacity == pytest.approx(0.5)
    assert animate.resolve(c, 4.0).opacity == pytest.approx(0.0)


@pytest.mark.parametrize("name", list(animate.ANIMATIONS["loop"]))
def test_loops_bounded(name):
    c = clip(dur=120.0, anim_loop=name, anim_loop_speed=2.0)
    for t in np.linspace(0, 120.0, 2001):
        p = animate.resolve(c, float(t))
        assert finite(p)
        assert 0.8 <= p.scale <= 1.2
        assert abs(p.pos_x) <= 0.05 and abs(p.pos_y) <= 0.05
        assert abs(p.rotation) <= 360
        assert 0.0 <= p.opacity <= 1.0


def test_composition_with_keyframes():
    c = clip(anim_in="slide_left", anim_in_dur=1.0)
    animate.set_keyframe(c, "pos_x", 0.0, 0.25)
    animate.set_keyframe(c, "scale", 0.0, 2.0)
    p0 = animate.resolve(c, 0.0)
    assert p0.pos_x == pytest.approx(0.25 - 1.0)
    assert p0.scale == pytest.approx(2.0)
    c.anim_in = "zoom_in"
    assert animate.resolve(c, 0.5).scale < 2.0


def test_deterministic():
    c = clip(anim_in="pop", anim_out="spin", anim_loop="flicker")
    a = [animate.resolve(c, t / 10) for t in range(41)]
    b = [animate.resolve(c, t / 10) for t in range(41)]
    assert a == b


def test_durations_longer_than_clip():
    c = clip(dur=0.3, anim_in="drop", anim_in_dur=2.0, anim_out="drop", anim_out_dur=2.0)
    for t in np.linspace(0, 0.3, 11):
        assert finite(animate.resolve(c, float(t)))


# ---- render smoke ----------------------------------------------------------
def test_render_smoke():
    from PySide6.QtWidgets import QApplication

    from clipeditor.render import Compositor
    _app = QApplication.instance() or QApplication([])
    proj = Project.new()
    c = Clip(new_id(), "", 0, 0, 4, text="Hi")
    c.anim_in = "pop"
    c.anim_in_dur = 1.0
    animate.set_keyframe(c, "rotation", 0.0, 0.0, "linear")
    animate.set_keyframe(c, "rotation", 3.0, 90.0, "linear")
    proj.video_tracks()[0].clips.append(c)
    comp = Compositor()
    frames = []
    for t in (0.1, 0.5, 1.5, 3.0):
        img = comp.render(proj, t, 320, 180)
        b = img.constBits()
        frames.append(np.frombuffer(b, np.uint8, img.sizeInBytes()).copy())
    assert getattr(comp, "last_error", None) in (None, "")
    for f in frames[1:]:
        assert f.any()  # something drawn
    for a, b in zip(frames, frames[1:]):
        assert not np.array_equal(a, b)
