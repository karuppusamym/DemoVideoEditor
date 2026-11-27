import os

import numpy as np
import pytest
from PySide6.QtGui import QGuiApplication

from clipeditor import sfx, shapes
from clipeditor.media import load_audio
from clipeditor.model import Clip


@pytest.fixture(scope="module", autouse=True)
def app():
    yield QGuiApplication.instance() or QGuiApplication([])


def _clip(shape):
    return Clip(id="s1", start=0.0, **{**shapes.new_shape_clip_fields("Rectangle"), "shape": shape})


def _rgba(img):
    return shapes._to_array(img)


@pytest.mark.parametrize("name", list(shapes.PRESETS))
def test_preset_renders(name):
    img = shapes.render_shape(_clip(shapes.PRESETS[name]), 360)
    assert not img.isNull() and img.width() > 2 and img.height() > 2
    a = _rgba(img)[:, :, 3]
    assert (a > 32).sum() > 50, name


def test_preset_counts_and_fields():
    assert len(shapes.SHAPE_PRESETS) >= 18
    assert len(shapes.EMOJI_PRESETS) >= 24
    f = shapes.new_shape_clip_fields("Fire")
    assert f["asset_id"] == "" and f["out_point"] > f["in_point"] and f["shape"]["kind"] == "emoji"
    f["shape"]["w"] = 9  # deep copy: preset untouched
    assert shapes.PRESETS["Fire"]["w"] != 9
    c = Clip(id="x", start=0.0, **shapes.new_shape_clip_fields("Star"))
    assert c.is_shape and c.is_generated


@pytest.mark.parametrize("name", ["Fire", "Thumbs Up", "Rocket"])
def test_emoji_in_colour(name):
    a = _rgba(shapes.render_shape(_clip(shapes.PRESETS[name]), 400)).astype(int)
    vis = a[a[:, :, 3] > 200][:, :3]
    assert len(vis) > 100
    chroma = vis.max(axis=1) - vis.min(axis=1)
    assert np.percentile(chroma, 75) > 40, "emoji rendered without colour"
    assert len({tuple(c // 32) for c in vis}) > 4


@pytest.mark.parametrize("name", ["Rectangle", "Star Burst", "Speech Bubble", "Fire"])
def test_size_scales_with_canvas(name):
    s = shapes.PRESETS[name]
    small = shapes.render_shape(_clip(s), 360)
    big = shapes.render_shape(_clip(s), 1080)
    assert big.width() == pytest.approx(small.width() * 3, rel=0.08)
    assert big.height() == pytest.approx(small.height() * 3, rel=0.08)
    assert big.width() >= s["w"] * 1080 and big.width() <= s["w"] * 1080 + 0.12 * 1080


def test_style_options():
    base = {"kind": "rect", "w": 0.3, "h": 0.2, "fill": "#ff0000"}
    plain = _rgba(shapes.render_shape(_clip(base), 200))
    grad = _rgba(shapes.render_shape(_clip({**base, "fill2": "#0000ff", "angle": 0}), 200))
    assert grad[grad.shape[0] // 2, -3, 0] > 150  # right edge blue (BGRA)
    half = _rgba(shapes.render_shape(_clip({**base, "fill_opacity": 0.5}), 200))
    assert abs(int(half[half.shape[0] // 2, half.shape[1] // 2, 3]) - 128) < 6
    shadow = shapes.render_shape(_clip({**base, "shadow": {"dx": 0.02, "dy": 0.02, "blur": 0.02}}), 200)
    assert shadow.width() > plain.shape[1]


def test_sfx_library():
    folder = sfx.ensure_library()
    assert os.path.isdir(folder)
    assert len(sfx.LIBRARY) >= 28
    for name, meta in sfx.LIBRARY.items():
        assert meta["category"] in sfx.CATEGORIES and meta["label"]
        p = sfx.path(name)
        assert os.path.exists(p), name
        x = load_audio(p)
        assert x.ndim == 2 and x.shape[1] == 2
        assert abs(len(x) / 48000 - meta["duration"]) < 0.05, name
        peak = float(np.abs(x).max())
        assert peak > 0.1, f"{name} is silent"
        assert peak < 0.999, f"{name} clips"
        assert float(np.sqrt((x ** 2).mean())) > 0.005, name
