import os
import time
import wave

import numpy as np
import pytest

from clipeditor import audio_fx, recorder, tts
from clipeditor.media import AudioCache
from clipeditor.model import Clip, MediaAsset

from .sample_media import SAMPLES

SPEECH = os.path.join(SAMPLES, "audio_fx.wav")
TONE = os.path.join(SAMPLES, "tone.wav")
SR = 48000


def _dur(path):
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate()


def _asset(path, aid):
    return MediaAsset(aid, path, "audio", _dur(path), has_audio=True)


def _clip(aid, a, b, speed=1.0, fx=None, start=0.0):
    return Clip("c_" + aid, aid, start, a, b, speed=speed, audio_fx=list(fx or []))


CACHE = AudioCache()
SPEECH_A = _asset(SPEECH, "speech")


@pytest.fixture(scope="module")
def dry():
    # speed 1, no fx is not "active", so render dry through the same path explicitly
    return audio_fx._fit(audio_fx._run_ffmpeg(SPEECH, 1.0, 4.0, 1.0, []), 4 * SR)


@pytest.mark.parametrize("fx_type", list(audio_fx.EFFECTS))
def test_each_effect(fx_type, dry):
    assert fx_type in audio_fx.PRESETS
    clip = _clip("speech", 1.0, 5.0, fx=[{"type": fx_type, "params": audio_fx.default_params(fx_type)}])
    out = audio_fx.clip_audio(clip, SPEECH_A, CACHE, wait=True)
    assert out is not None, audio_fx.status(clip, SPEECH_A)
    assert out.dtype == np.float32 and out.shape == (int(round(clip.duration * SR)), 2)
    assert np.isfinite(out).all()
    assert np.abs(out).max() > 0.01  # not silent
    assert np.abs(out - dry).mean() > 1e-3  # audibly processed
    assert audio_fx.status(clip, SPEECH_A) == "ready"


def _dominant_hz(x):
    m = x.mean(1)
    m = m[len(m) // 4: len(m) // 4 + SR]
    spec = np.abs(np.fft.rfft(m * np.hanning(len(m))))
    return np.argmax(spec) * SR / len(m)


def test_speed_keeps_length_and_pitch():
    tone = _asset(TONE, "tone")
    n_src = min(tone.duration, 6.0)
    clip = _clip("tone", 0.0, n_src, speed=2.0)
    out = audio_fx.clip_audio(clip, tone, CACHE, wait=True)
    assert out.shape == (int(round(clip.duration * SR)), 2)
    assert abs(_dominant_hz(out) - 220.0) < 5.0
    slow = _clip("tone", 0.0, 2.0, speed=0.5)
    out = audio_fx.clip_audio(slow, tone, CACHE, wait=True)
    assert out.shape == (int(round(slow.duration * SR)), 2)
    assert abs(_dominant_hz(out) - 220.0) < 5.0


def test_nowait_then_ready_and_cache_hit():
    clip = _clip("speech", 0.5, 3.5, speed=1.25, fx=[{"type": "echo", "params": {"intensity": 0.31}}])
    assert audio_fx.clip_audio(clip, SPEECH_A, CACHE, wait=False) is None
    assert audio_fx.status(clip, SPEECH_A) in ("processing", "ready")
    deadline = time.time() + 30
    out = None
    while out is None and time.time() < deadline:
        time.sleep(0.05)
        out = audio_fx.clip_audio(clip, SPEECH_A, CACHE, wait=False)
    assert out is not None and out.shape == (int(round(clip.duration * SR)), 2)
    t = time.perf_counter()
    for _ in range(100):
        assert audio_fx.clip_audio(clip, SPEECH_A, CACHE, wait=False) is out
    assert (time.perf_counter() - t) / 100 < 0.002


def test_source_shorter_than_range_is_padded():
    clip = _clip("speech", SPEECH_A.duration - 1.0, SPEECH_A.duration + 1.0, fx=[{"type": "bass_boost"}])
    out = audio_fx.clip_audio(clip, SPEECH_A, CACHE, wait=True)
    assert out.shape == (2 * SR, 2)
    assert np.abs(out[-SR // 2:]).max() < 1e-3


def test_render_preview():
    out = audio_fx.render_preview(SPEECH, [{"type": "robot"}], seconds=2)
    assert out.shape == (2 * SR, 2) and np.isfinite(out).all()


def test_tts_wav():
    if not tts.voices():
        pytest.skip("SAPI unavailable: " + tts.engines()["sapi"])
    path = tts.synthesize('Hello "world" & <friends>; it\'s 100% fine.', rate=1)
    try:
        with wave.open(path) as w:
            assert w.getnframes() / w.getframerate() > 0.5
            assert w.getsampwidth() == 2
    finally:
        os.remove(path)


def test_recorder_devices():
    devs = recorder.input_devices()
    assert isinstance(devs, list)
    for d in devs:
        assert {"index", "name", "channels"} <= set(d)
    r = recorder.Recorder()
    assert r.level() == 0.0
    with pytest.raises(RuntimeError):
        r.stop()
