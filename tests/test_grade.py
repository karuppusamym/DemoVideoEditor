import time

import numpy as np
import pytest

from clipeditor import grade as G
from clipeditor.model import Clip


def frame(seed=0, h=360, w=640):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 4), dtype=np.uint8)


def solid(rgb, h=8, w=8, a=255):
    img = np.empty((h, w, 4), np.uint8)
    img[..., :3] = rgb
    img[..., 3] = a
    return img


def clip_with(grade):
    c = Clip.__new__(Clip)
    c.grade = grade
    return c


def write_cube(path, size, rows, extra=""):
    lines = ["# test LUT", 'TITLE "t"', f"LUT_3D_SIZE {size}", extra]
    lines += ["%f %f %f" % tuple(r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def grid_rows(size, fn):
    t = np.linspace(0, 1, size)
    rows = []
    for b in t:
        for g in t:
            for r in t:   # R fastest
                rows.append(fn(r, g, b))
    return rows


# ---------------------------------------------------------------- identity / contract

def test_identity_exact():
    img = frame()
    for g in ({}, G.default_grade(), {"exposure": 0.0, "lift": [0, 0, 0], "lut": "", "hsl": {}}):
        assert G.is_identity(g)
        assert np.array_equal(G.apply_to_image(img, g), img)
        assert not G.active(clip_with(g))
        assert np.array_equal(G.apply(img, clip_with(g)), img)
    assert G.active(clip_with({"exposure": 0.5}))
    assert not G.is_identity({"curves": {"master": [[0, 0], [0.5, 0.6], [1, 1]]}})


def test_metadata_for_ui():
    assert G.WHEEL_KEYS == ("lift", "gamma", "gain")
    assert len(G.HSL_BANDS) == 8 and G.HSL_BANDS[0] == "red"
    for k in ("exposure", "temperature", "tint", "highlights", "shadows", "whites", "blacks",
              "vibrance"):
        spec = G.BASIC_PARAMS[k]
        assert spec["min"] < spec["default"] < spec["max"] and spec["label"]
    d = G.default_grade()
    assert set(G.BASIC_PARAMS) <= set(d) and d["lut_mix"] == 1.0


def test_alpha_preserved_and_input_untouched():
    img = frame(1)
    img[..., 3] = np.arange(640, dtype=np.uint16)[None, :] % 256
    before = img.copy()
    g = {"exposure": 1, "hsl": {"red": {"h": 20}}, "vibrance": 0.5, "lut": "teal_orange"}
    out = G.apply_to_image(img, g)
    assert np.array_equal(out[..., 3], img[..., 3])
    assert np.array_equal(img, before)
    assert out.dtype == np.uint8 and out.shape == img.shape


# ---------------------------------------------------------------- basic

def test_exposure_one_stop():
    grey = solid(118)
    out = G.apply_to_image(grey, {"exposure": 1.0})
    lin_in = (118 / 255) ** 2.2
    lin_out = (out[0, 0, :3] / 255.0) ** 2.2
    assert np.all(np.abs(lin_out / lin_in - 2.0) < 0.15)
    dark = G.apply_to_image(grey, {"exposure": -1.0})
    assert np.all(dark[0, 0, :3] < 118)


def test_temperature_and_tint():
    grey = solid(128)
    warm = G.apply_to_image(grey, {"temperature": 0.6})[0, 0]
    assert warm[0] > 128 and warm[2] < 128
    cool = G.apply_to_image(grey, {"temperature": -0.6})[0, 0]
    assert cool[0] < 128 and cool[2] > 128
    magenta = G.apply_to_image(grey, {"tint": 0.6})[0, 0]
    assert magenta[1] < magenta[0] and magenta[1] < magenta[2]


def test_tone_and_vibrance():
    ramp = np.zeros((1, 256, 4), np.uint8)
    ramp[..., :3] = np.arange(256, dtype=np.uint8)[None, :, None]
    sh = G.apply_to_image(ramp, {"shadows": 1.0})[0, :, 0].astype(int)
    assert sh[60] > 60 and abs(sh[250] - 250) <= 3
    hl = G.apply_to_image(ramp, {"highlights": -1.0})[0, :, 0].astype(int)
    assert hl[200] < 200 and abs(hl[10] - 10) <= 2
    muted, vivid = solid((140, 120, 110)), solid((250, 20, 20))
    dm = np.ptp(G.apply_to_image(muted, {"vibrance": 1})[0, 0, :3].astype(int)) - 30
    dv = np.ptp(G.apply_to_image(vivid, {"vibrance": 1})[0, 0, :3].astype(int)) - 230
    assert dm > 0 and dm > dv


def test_wheels():
    grey = solid(100)
    assert G.apply_to_image(grey, {"gain": [0.3, 0, 0]})[0, 0, 0] > 100
    assert G.apply_to_image(grey, {"gamma": [0, 0, 0.5]})[0, 0, 2] > 100
    assert G.apply_to_image(solid(0), {"lift": [0, 0.4, 0]})[0, 0, 1] > 0
    # lift alone keeps white at white
    assert G.apply_to_image(solid(255), {"lift": [0.5, 0.5, 0.5]})[0, 0, 0] == 255


# ---------------------------------------------------------------- curves

@pytest.mark.parametrize("pts", [
    [[0, 0], [0.25, 0.1], [0.5, 0.5], [0.75, 0.95], [1, 1]],
    [[0, 0.1], [0.3, 0.12], [0.4, 0.8], [1, 0.9]],
    [[0, 0], [0.5, 0.5], [0.52, 0.5], [1, 1]],
])
def test_curve_monotone_and_endpoints(pts):
    lut = G.curve_lut(tuple(map(tuple, pts)))
    assert lut.shape == (256,)
    assert np.all(np.diff(lut) >= -1e-6)
    assert lut[0] == pytest.approx(pts[0][1], abs=1e-6)
    assert lut[-1] == pytest.approx(pts[-1][1], abs=1e-6)
    for x, y in pts:
        assert G.eval_curve(pts, [x])[0] == pytest.approx(y, abs=2e-3)


def test_curve_flat_outside_points_and_channel():
    lut = G.curve_lut(((0.2, 0.3), (0.8, 0.7)))
    assert lut[0] == pytest.approx(0.3) and lut[-1] == pytest.approx(0.7)
    out = G.apply_to_image(solid(128), {"curves": {"b": [[0, 0], [0.5, 0.8], [1, 1]]}})[0, 0]
    assert out[0] == 128 and out[1] == 128 and out[2] > 190


# ---------------------------------------------------------------- HSL

def test_hsl_red_band_only():
    img = np.zeros((4, 4, 4), np.uint8)
    img[..., 3] = 255
    img[:2, :, :3] = (30, 60, 200)    # blue
    img[2:, :, :3] = (210, 40, 30)    # red
    g = {"hsl": {"red": {"h": 40, "s": -0.5, "l": 0.2}}}
    out = G.apply_to_image(img, g)
    assert np.array_equal(out[:2], img[:2])
    assert not np.array_equal(out[2:], img[2:])
    sat = G.apply_to_image(img, {"hsl": {"red": {"s": -1.0}}})[3, 0, :3].astype(int)
    assert np.ptp(sat) < np.ptp(img[3, 0, :3].astype(int)) // 3


def test_hsl_greys_untouched():
    ramp = np.zeros((1, 256, 4), np.uint8)
    ramp[..., :3] = np.arange(256, dtype=np.uint8)[None, :, None]
    g = {"hsl": {b: {"h": 30, "s": 0.5, "l": 0.3} for b in G.HSL_BANDS}}
    assert np.array_equal(G.apply_to_image(ramp, g), ramp)


# ---------------------------------------------------------------- .cube LUTs

def test_cube_identity(tmp_path):
    p = write_cube(tmp_path / "id.cube", 9, grid_rows(9, lambda r, g, b: (r, g, b)))
    lut = G.parse_cube(p)
    assert lut.size == 9 and lut.table.shape == (9, 9, 9, 3) and lut.title == "t"
    img = frame(2, 64, 64)
    out = G.apply_to_image(img, {"lut": p})
    assert np.abs(out.astype(int) - img).max() <= 1


def test_cube_invert_2x2x2(tmp_path):
    p = write_cube(tmp_path / "inv.cube", 2, grid_rows(2, lambda r, g, b: (1 - r, 1 - g, 1 - b)))
    img = frame(3, 32, 32)
    out = G.apply_to_image(img, {"lut": p})
    assert np.abs(out[..., :3].astype(int) - (255 - img[..., :3].astype(int))).max() <= 1
    half = G.apply_to_image(img, {"lut": p, "lut_mix": 0.5})
    assert np.abs(half[..., :3].astype(int) - 127.5).max() <= 2
    assert G.parse_cube(p) is G.parse_cube(p)  # cached


def test_cube_domain(tmp_path):
    # identity over domain 0..2: an input of x reads the table at x/2 -> output x/2
    p = write_cube(tmp_path / "dom.cube", 5, grid_rows(5, lambda r, g, b: (r, g, b)),
                   "DOMAIN_MIN 0 0 0\nDOMAIN_MAX 2.0 2.0 2.0")
    lut = G.parse_cube(p)
    assert np.allclose(lut.domain_max, 2.0)
    out = G.apply_to_image(solid((200, 100, 50)), {"lut": p})[0, 0, :3].astype(int)
    assert np.all(np.abs(out - np.array([100, 50, 25])) <= 1)
    r, g, b = G.apply_lut_planes(np.float32([0.5]), np.float32([1.0]), np.float32([0.0]), lut)
    assert r[0] == pytest.approx(0.25, abs=1e-5) and g[0] == pytest.approx(0.5, abs=1e-5)


def test_cube_bad_and_missing(tmp_path):
    bad = tmp_path / "bad.cube"
    bad.write_text("LUT_3D_SIZE 3\n0 0 0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        G.parse_cube(str(bad))
    img = frame(4, 16, 16)
    assert np.array_equal(G.apply_to_image(img, {"lut": str(tmp_path / "nope.cube")}), img)


def test_builtin_luts():
    paths = G.ensure_builtin_luts()
    assert len(paths) >= 8 and paths == G.BUILTIN_LUTS
    img = frame(5, 90, 160)
    for name, path in paths.items():
        lut = G.parse_cube(path)
        assert lut.kind == "3d" and lut.size >= 9
        for ref in (name, path):
            out = G.apply_to_image(img, {"lut": ref})
            assert np.abs(out[..., :3].astype(int) - img[..., :3]).mean() > 3, name
            assert np.array_equal(out[..., 3], img[..., 3])
    bw = G.apply_to_image(img, {"lut": "high_contrast_bw"})
    assert np.ptp(bw[..., :3].astype(int), axis=-1).max() <= 2


# ---------------------------------------------------------------- performance

FULL = {
    "exposure": 0.3, "temperature": 0.2, "tint": -0.1, "contrast": 0.2, "highlights": -0.3,
    "shadows": 0.3, "whites": 0.1, "blacks": -0.1, "vibrance": 0.3, "saturation": 0.1,
    "lift": [0.05, 0, -0.05], "gamma": [0, 0.1, 0], "gain": [0, 0, 0.1],
    "curves": {"master": [[0, 0], [0.25, 0.2], [0.75, 0.8], [1, 1]], "r": [[0, 0.02], [1, 0.98]]},
    "hsl": {"red": {"h": 10, "s": 0.2, "l": 0.1}, "blue": {"h": -10, "s": -0.2, "l": 0}},
    "lut": "teal_orange", "lut_mix": 0.7,
}


def test_full_grade_timing():
    img = frame(6)
    G.apply_to_image(img, FULL)  # bake caches
    n = 10
    t = time.perf_counter()
    for _ in range(n):
        out = G.apply_to_image(img, FULL)
    ms = (time.perf_counter() - t) / n * 1000
    assert not np.array_equal(out, img)
    assert ms < 40, f"full grade {ms:.1f} ms"
