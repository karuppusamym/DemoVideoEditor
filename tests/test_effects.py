import time

import numpy as np
import pytest

from clipeditor import effects as E
from clipeditor import masks as M
from clipeditor.model import Clip


def frame(seed=0, h=360, w=640):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 4), dtype=np.uint8)


def clip_with(effects=(), chroma=None, masks=()):
    c = Clip.__new__(Clip)  # only the fields effects/masks read
    c.effects = list(effects)
    c.chroma = chroma or {}
    c.masks = list(masks)
    return c


def test_registry_shape():
    assert len(E.REGISTRY) >= 104
    cats = E.categories()
    assert set(cats) >= {"Filters", "Light & Colour", "Blur & Sharp", "Stylize", "Retro & Glitch", "Motion"}
    assert set(cats) == set(E.CATEGORY_ORDER), "every category is listed in CATEGORY_ORDER"
    assert all(len(cats[c]) >= 6 for c in E.CATEGORY_ORDER)
    assert sum(len(v) for v in cats.values()) == len(E.REGISTRY)
    for name, spec in E.REGISTRY.items():
        assert {"label", "category", "params"} <= set(spec)
        assert "intensity" in spec["params"] or "amount" in spec["params"], name
        for p in spec["params"].values():
            assert p["min"] <= p["default"] <= p["max"]
        e = E.new_effect(name)
        assert e["enabled"] and e["type"] == name


@pytest.mark.parametrize("name", list(E.REGISTRY))
def test_effect_runs_and_keeps_alpha(name):
    img = frame(1)
    before = img.copy()
    clip = clip_with([E.new_effect(name)])
    out = E.apply(img, clip, 0.37, 360)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert out.flags.c_contiguous
    assert np.array_equal(img, before), "input frame must not be modified"
    if E.REGISTRY[name].get("moves_alpha"):
        opaque = img.copy()
        opaque[..., 3] = 255
        assert (E.run_effect(name, opaque, None, 0.37, 360)[..., 3] == 255).all()
    else:
        assert np.array_equal(out[..., 3], img[..., 3])
    # direct call raises rather than swallowing errors
    E.run_effect(name, img, None, 0.37, 360)


def test_effects_change_pixels():
    img = frame(2)
    unchanged = [n for n in E.REGISTRY
                 if np.array_equal(E.preview_card(n, img)[..., :3], img[..., :3])]
    assert unchanged == []


@pytest.mark.parametrize("name", list(E.REGISTRY))
def test_effects_deterministic(name):
    img = frame(3)
    a = E.run_effect(name, img, None, 1.2345, 360)
    b = E.run_effect(name, img.copy(), None, 1.2345, 360)
    assert np.array_equal(a, b)


ANIMATED = ["grain", "noise", "glitch", "vhs", "old_film", "shake", "zoom_pulse", "strobe", "heartbeat",
            "wobble", "light_leak", "hue_rotate", "bokeh", "defocus_pulse", "ripple", "wave", "datamosh",
            "glitch_bars", "bad_tv", "film_scratch", "film_burn", "sway", "earthquake", "ken_burns",
            "rotate_wobble", "flash_pulse", "beat_bounce", "echo_trail", "spin"]


@pytest.mark.parametrize("name", ANIMATED)
def test_animated_effects_change_over_time(name):
    img = frame(8)
    a = E.run_effect(name, img, None, 0.4, 360)
    b = E.run_effect(name, img, None, 0.93, 360)
    assert not np.array_equal(a, b)


def _flat(rgb, h=60, w=80):
    img = np.empty((h, w, 4), np.uint8)
    img[..., :3] = rgb
    img[..., 3] = 255
    return img


def test_colour_pop_keeps_only_target_hue():
    img = _flat((200, 30, 30))
    img[:, 40:, :3] = (30, 60, 200)
    out = E.run_effect("colour_pop", img, {"hue": 0.0}, 0.0, 60).astype(int)
    assert out[10, 10, 0] - out[10, 10, 2] > 100  # red kept
    assert np.ptp(out[10, 60, :3]) <= 3  # blue turned grey


def test_letterbox_and_polaroid_frame():
    img = _flat((120, 160, 200), 90, 160)
    out = E.run_effect("letterbox", img, None, 0.0, 90)
    assert out[2, 80, :3].max() == 0 and out[45, 80, :3].tolist() == [120, 160, 200]
    pol = E.run_effect("polaroid_frame", img, None, 0.0, 90)
    assert pol[2, 80, :3].min() > 230 and pol[88, 80, :3].min() > 230  # white border
    assert abs(int(pol[30, 80, 2]) - 198) < 12  # photo inside


def test_channel_swap_and_hue_rotate():
    img = _flat((200, 50, 10))
    out = E.run_effect("channel_swap", img, {"mode": 2}, 0.0, 60)
    assert out[0, 0, :3].tolist() == [10, 50, 200]
    a = E.run_effect("hue_rotate", img, {"degrees": 0.0, "speed": 0.0}, 0.0, 60)
    assert np.abs(a.astype(int) - img.astype(int)).max() <= 1


def test_geometric_effects_keep_centre():
    img = frame(9, 91, 161)
    img[..., 3] = 255
    for name in ("fisheye", "bulge", "pinch", "swirl"):
        out = E.run_effect(name, img, None, 0.0, 91)
        assert np.abs(out[45, 80].astype(int) - img[45, 80].astype(int)).max() <= 1, name
        assert not np.array_equal(out, img), name


def test_disabled_and_unknown_effects_skipped():
    img = frame(4)
    e = E.new_effect("invert")
    e["enabled"] = False
    out = E.apply(img, clip_with([e, {"type": "nope", "params": {}}]), 0.0, 360)
    assert np.array_equal(out, img)


def test_effect_chain_order():
    img = frame(5)
    out = E.apply(img, clip_with([E.new_effect("invert"), E.new_effect("invert")]), 0.0, 360)
    assert np.abs(out.astype(int) - img.astype(int)).max() <= 1


def test_chroma_key_removes_green():
    img = np.zeros((90, 160, 4), np.uint8)
    img[..., 1] = 255
    img[..., 3] = 255
    img[30:60, 50:110] = (200, 60, 60, 255)  # subject
    clip = clip_with(chroma=E.new_chroma("#00ff00"))
    assert E.active(clip)
    out = E.apply(img, clip, 0.0, 90)
    assert out[..., 3][:20].max() == 0  # background gone
    assert (out[35:55, 55:105, 3] == 255).all()  # subject kept
    assert np.abs(out[40, 80, :3].astype(int) - (200, 60, 60)).max() <= 3


def test_chroma_spill_reduces_green_tint():
    img = np.zeros((40, 40, 4), np.uint8)
    img[...] = (150, 200, 140, 255)  # greenish-lit subject, not keyed out
    out = E.chroma_key(img, {"color": "#00ff00", "similarity": 0.2, "smooth": 0.05, "spill": 1.0})
    assert (out[..., 3] == 255).all()
    assert out[0, 0, 1] < 200


def test_ellipse_mask_and_invert():
    img = np.full((100, 200, 4), 255, np.uint8)
    m = M.new_mask("ellipse")
    m.update(x=0.5, y=0.5, w=0.5, h=0.5, feather=0.0)
    clip = clip_with(masks=[m])
    assert M.active(clip)
    out = M.apply(img.copy(), clip, 0.0)
    a = out[..., 3]
    assert a[50, 100] == 255 and a[0, 0] == 0 and a[50, 10] == 0 and a[50, 190] == 0
    assert a[50, 60] == 255 and a[50, 40] == 0 and a[50, 160] == 0  # half width = 50 px
    m["invert"] = True
    inv = M.apply(img.copy(), clip, 0.0)[..., 3]
    assert inv[50, 100] == 0 and inv[0, 0] == 255


def test_mask_shapes_and_multiply():
    img = np.full((100, 200, 4), 255, np.uint8)
    lin = M.new_mask("linear")
    lin.update(feather=0.0)
    a = M.apply(img.copy(), clip_with(masks=[lin]), 0.0)[..., 3]
    assert a[10, 100] == 255 and a[90, 100] == 0
    band = M.new_mask("mirror")
    band.update(h=0.2, feather=0.0)
    b = M.apply(img.copy(), clip_with(masks=[band]), 0.0)[..., 3]
    assert b[50, 5] == 255 and b[10, 5] == 0
    rect = M.new_mask("rect")
    rect.update(w=0.5, h=0.5, round=0.5, feather=0.1, angle=30)
    r = M.apply(img.copy(), clip_with(masks=[rect]), 0.0)[..., 3]
    assert r[50, 100] == 255 and r[0, 0] == 0
    both = M.apply(img.copy(), clip_with(masks=[lin, band]), 0.0)[..., 3]
    assert both[45, 100] == 255 and both[55, 100] == 0 and both[10, 100] == 0
    half = img.copy()
    half[..., 3] = 128
    assert M.apply(half, clip_with(masks=[rect]), 0.0)[50, 100, 3] == 128


def test_preview_card():
    sample = frame(6, 90, 160)
    for name in E.REGISTRY:
        card = E.preview_card(name, sample)
        assert card.shape == sample.shape and card.dtype == np.uint8


def test_effect_timing():
    img = frame(7)
    times = {}
    for name in E.REGISTRY:
        E.run_effect(name, img, None, 0.5, 360)  # warm caches
        best = 1e9
        for i in range(3):
            t0 = time.perf_counter()
            E.run_effect(name, img, None, 0.5 + i / 30, 360)
            best = min(best, time.perf_counter() - t0)
        times[name] = best * 1000
    worst = sorted(times.items(), key=lambda kv: -kv[1])
    print("\nms per 640x360 frame:", ", ".join(f"{n}={ms:.1f}" for n, ms in worst))
    slow = [(n, round(ms, 1)) for n, ms in worst if ms > 60]
    assert not slow, slow
