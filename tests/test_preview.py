"""Preview engine: audio mix (speed/fades) and compositor (transform/eq/text)."""
import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.name == "nt":  # Qt's offscreen platform has no font database on Windows without this
    os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))

from clipeditor.textrender import ensure_qapp  # noqa: E402

ensure_qapp()

from clipeditor import audio_mix  # noqa: E402
from clipeditor.media import AudioCache, probe  # noqa: E402
from clipeditor.model import Clip, Project, new_id  # noqa: E402
from clipeditor.ui.preview import Compositor  # noqa: E402

from .sample_media import SAMPLES

CLIP_A = os.path.join(SAMPLES, "clip_a.mp4")
TONE = os.path.join(SAMPLES, "tone.wav")
SCRATCH = os.environ.get("PREVIEW_CHECK_DIR")


@pytest.fixture(scope="module")
def media():
    va, tone = probe(CLIP_A), probe(TONE)
    cache = AudioCache()
    cache.load(va)
    cache.load(tone)
    return va, tone, cache


def _project(*assets):
    p = Project.new()
    p.width, p.height = 1280, 720
    for a in assets:
        p.assets[a.id] = a
    return p


def test_audio_speed2(media):
    _, tone, cache = media
    p = _project(tone)
    a1 = p.audio_tracks()[0]
    a1.clips.append(Clip(new_id(), tone.id, 0.0, 0.0, min(tone.duration, 4.0), speed=2.0))
    data = cache.get(tone.id)
    n = 1024
    t0 = 0.5
    out = audio_mix.mix(p, cache, t0, n)
    assert out.shape == (n, 2)
    s0 = int(round(t0 * 2.0 * cache.sr))
    expect = data[s0 + np.arange(n) * 2]
    assert np.allclose(out, np.clip(expect, -1, 1), atol=1e-5)
    # clip ends at timeline duration/2; mixing past it is silent
    dur = a1.clips[0].duration
    assert np.all(audio_mix.mix(p, cache, dur + 0.01, n) == 0)
    # block straddling the end: samples after the end are zero, none past out_point
    out = audio_mix.mix(p, cache, dur - 0.005, n)
    k = int(round(0.005 * cache.sr))
    assert np.all(out[k + 1:] == 0) and np.any(out[:k] != 0)


def test_audio_fades(media):
    _, tone, cache = media
    p = _project(tone)
    c = Clip(new_id(), tone.id, 1.0, 0.0, 2.0, fade_in=0.5, fade_out=0.5)
    p.audio_tracks()[0].clips.append(c)
    sr, n = cache.sr, 1024
    data = cache.get(tone.id)
    peak = float(np.abs(data[: 2 * sr]).max())
    assert peak > 0.05
    head = audio_mix.mix(p, cache, 1.0, n)
    assert np.abs(head[:16]).max() < peak * 0.01  # ~0 at fade-in start
    tail = audio_mix.mix(p, cache, c.end - n / sr, n)
    assert np.abs(tail[-16:]).max() < peak * 0.01  # ~0 at fade-out end
    mid = audio_mix.mix(p, cache, 2.0, n)
    assert np.abs(mid).max() > peak * 0.9  # full gain in the middle
    # gain ramp matches Clip.fade_gain
    i = 600
    g = c.fade_gain(1.0 + i / sr)
    s = int(round(0.0 * sr)) + i
    assert np.allclose(head[i], data[s] * g, atol=1e-4)
    assert c.fade_gain(c.start) == pytest.approx(0) and c.fade_gain(c.end) == pytest.approx(0)


def test_text_clip_skipped_in_audio(media):
    _, _, cache = media
    p = _project()
    p.audio_tracks()[0].clips.append(Clip(new_id(), "", 0.0, 0.0, 3.0, text="hi"))
    assert np.all(audio_mix.mix(p, cache, 0.0, 512) == 0)


def _scene(va):
    p = _project(va)
    v2, v1 = p.video_tracks()
    v1.clips.append(Clip(new_id(), va.id, 0.0, 0.0, 8.0, brightness=0.1, contrast=1.3, saturation=0.4))
    v2.clips.append(Clip(new_id(), va.id, 0.0, 2.0, 8.0, scale=0.4, pos_x=0.25, pos_y=-0.22,
                         fade_in=1.0))
    extra = p.tracks.insert(0, type(v1)(new_id(), "video", "V3")) or p.tracks[0]
    extra.clips.append(Clip(new_id(), "", 0.0, 0.0, 8.0, text="Preview check\nline two",
                            font_size=64, bg_color="#000000a0", pos_y=0.3))
    # missing-asset clip must not crash
    p.tracks.insert(0, type(v1)(new_id(), "video", "V4", [Clip(new_id(), "nope", 0.0, 0.0, 8.0)]))
    return p


def test_render_and_speed(media):
    va, _, _ = media
    p = _scene(va)
    comp = Compositor()
    try:
        img = comp.render(p, 2.0, 640, 360)
        assert (img.width(), img.height()) == (640, 360)
        if SCRATCH:
            img.save(os.path.join(SCRATCH, "preview_check.png"))
        assert Compositor.apply_eq(np.full((2, 2, 3), 128, np.uint8), 0, 1, 1)[0, 0, 0] == 128
        n = 200
        t0 = time.perf_counter()
        for i in range(n):
            comp.render(p, 1.0 + i / 30, 640, 360)
        ms = (time.perf_counter() - t0) / n * 1000
        print(f"\nrender 640x360: {ms:.2f} ms/frame")
        assert ms < 33
    finally:
        comp.close()


def test_eq_semantics():
    a = np.array([[[0, 128, 255]]], np.uint8)
    out = Compositor.apply_eq(a, 0.0, 1.0, 0.0)  # full desaturation -> grey luma
    assert out[0, 0, 0] == out[0, 0, 1] == out[0, 0, 2]
    out = Compositor.apply_eq(a, 0.2, 1.0, 1.0)
    assert out[0, 0, 0] == 51 and out[0, 0, 2] == 255
    out = Compositor.apply_eq(a, 0.0, 0.0, 1.0)  # zero contrast -> mid grey
    assert np.all(out == 128)
