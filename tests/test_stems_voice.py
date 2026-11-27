"""Stem separation (spectral engine) and voice cloning / voice match fallback.

Synthetic material only: a centre-panned "voice" (harmonic tone with vibrato and formant-like
band energy, 150-3000 Hz) over "music" (wide-panned low bass notes + hi-hat noise bursts).
Run with `-s` to see quality and speed numbers.
"""
import os
import time

import numpy as np
import pytest

from clipeditor.ai import jobs
from clipeditor.ai import stems as ST
from clipeditor.ai import voice_clone as VC

SR = 48000


@pytest.fixture(autouse=True)
def _gen_dir(tmp_path, monkeypatch):
    """Keep generated files out of the real %APPDATA%."""
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))


def _voice(t, f_base=200.0, rng=None, vib=0.02):
    rng = rng or np.random.default_rng(1)
    dur = t[-1] + 1 / SR
    notes = f_base * 2 ** (rng.integers(0, 7, int(dur / 0.8) + 2) / 12)
    f0 = notes[(t / 0.8).astype(int)] * (1 + vib * np.sin(2 * np.pi * 5.5 * t))
    ph = 2 * np.pi * np.cumsum(f0) / SR
    v = np.zeros_like(t)
    for k in range(1, 16):
        fk = f0 * k
        g = (np.exp(-((fk - 700) / 300) ** 2) + 0.6 * np.exp(-((fk - 1200) / 400) ** 2)
             + 0.3 * np.exp(-((fk - 2500) / 500) ** 2))
        v += g * ((fk > 150) & (fk < 3000)) * np.sin(k * ph)
    v *= 0.5 + 0.5 * np.clip(np.sin(np.pi * (t % 0.8) / 0.8) * 1.5, 0, 1)
    return v * (0.25 / np.sqrt(np.mean(v ** 2)))


def _music(t, rng=None):
    rng = rng or np.random.default_rng(2)
    bf = np.array([41.2, 55.0, 49.0, 36.7])[(t / 1.0).astype(int) % 4]
    bph = 2 * np.pi * np.cumsum(bf) / SR
    bass = 0.3 * (np.sin(bph) + 0.3 * np.sin(2 * bph)) * np.exp(-3 * (t % 1.0))
    hats = np.zeros((len(t), 2))
    L = int(0.04 * SR)
    for s in np.arange(0, t[-1], 0.25):
        i = int(s * SR)
        nz = np.diff(rng.standard_normal((L + 1, 2)), axis=0)  # bright, decorrelated L/R
        seg = 0.15 * nz * np.exp(-np.arange(L) / (0.01 * SR))[:, None]
        hats[i:i + L] += seg[: len(hats) - i]
    return np.stack([0.9 * bass, 0.35 * bass], 1) + hats, bass, hats


def _mix(seconds):
    t = np.arange(int(SR * seconds)) / SR
    voice = _voice(t)
    music, bass, hats = _music(t)
    V = np.repeat(voice[:, None], 2, 1)
    return (V + music).astype(np.float32), V, music, bass, hats


def _frac(out, ref):
    """Share of `ref` that ended up in `out` (projection; shares across stems sum to ~1)."""
    return float(np.sum(out * ref) / np.sum(ref * ref))


def _band_energy(x, lo, hi):
    x = np.asarray(x, np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    X = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return float(X[(f >= lo) & (f < hi)].sum())


def test_spectral_split_quality_and_reconstruction():
    mix, V, music, bass, hats = _mix(8.0)
    v, m = ST.separate_array(mix, SR)
    assert v.shape == mix.shape and m.shape == mix.shape
    voice_v, voice_m = _frac(v, V), _frac(m, V)
    mus_v, mus_m = _frac(v, music), _frac(m, music)
    print(f"\n[stems] voice -> vocals {voice_v:.3f} / music {voice_m:.3f};"
          f" music -> vocals {mus_v:.3f} / music {mus_m:.3f}")
    assert voice_v > 0.7 and voice_v > 3 * voice_m
    assert mus_m > 0.7 and mus_m > 3 * mus_v
    # correlation view
    assert np.corrcoef(v.ravel(), V.ravel())[0, 1] > 0.8
    assert np.corrcoef(m.ravel(), music.ravel())[0, 1] > 0.8
    # band energies: bass (<100 Hz) and hats (>8 kHz) mostly in music
    assert _band_energy(m, 20, 100) > 5 * _band_energy(v, 20, 100)
    assert _band_energy(m, 8000, 20000) > 5 * _band_energy(v, 8000, 20000)
    assert _band_energy(v, 300, 3000) > 2 * _band_energy(m, 300, 3000)
    err = mix - (v + m)
    snr = 10 * np.log10(np.sum(mix ** 2) / max(np.sum(err ** 2), 1e-30))
    print(f"[stems] reconstruction SNR {snr:.1f} dB")
    assert snr > 20


def test_spectral_mono_and_short_inputs():
    mix, V, *_ = _mix(2.0)
    mono = mix.mean(axis=1)
    v, m = ST.separate_array(mono, SR)
    assert v.shape == mono.shape
    assert _frac(v, V[:, 0]) > 0.5
    assert np.allclose(v + m, mono, atol=1e-4)
    tiny = np.random.default_rng(0).standard_normal((300, 2)).astype(np.float32) * 0.1
    v, m = ST.separate_array(tiny, SR)
    assert np.allclose(v + m, tiny, atol=1e-4)


def test_separate_file_timing_and_outputs(tmp_path):
    mix, *_ = _mix(30.0)
    src = str(tmp_path / "mix.wav")
    ST.write_wav(src, mix * 0.9, SR)
    prog = []
    t0 = time.time()
    outs = ST.separate(src, 0.0, None, engine="spectral", progress=prog.append)
    el = time.time() - t0
    print(f"\n[speed] spectral stems: {el:.2f} s for 30 s of 48 kHz stereo ({el / 30:.2f}x real time)")
    assert el < 1.5 * 30
    assert set(outs) == {"vocals", "music"}
    a, sr = ST.read_wav(outs["vocals"])
    b, _ = ST.read_wav(outs["music"])
    assert sr == SR and a.shape == b.shape and abs(len(a) - len(mix)) <= SR * 0.01
    assert os.path.dirname(outs["vocals"]) == jobs.output_dir("stems")
    assert prog and prog[-1] == 1.0 and all(x <= y + 1e-9 for x, y in zip(prog, prog[1:]))
    # sub-range + convenience wrappers
    mv = ST.remove_vocals(src, 2.0, 5.0, engine="spectral")
    iv = ST.isolate_vocals(src, 2.0, 5.0, engine="spectral")
    for p in (mv, iv):
        x, _ = ST.read_wav(p)
        assert abs(len(x) / SR - 3.0) < 0.02
    assert ST.best_engine() in ST.ENGINES and ST.engine_available("spectral")


def test_separate_cancel(tmp_path):
    mix, *_ = _mix(6.0)
    src = str(tmp_path / "mix.wav")
    ST.write_wav(src, mix, SR)
    with pytest.raises(jobs.Cancelled):
        ST.separate(src, engine="spectral", cancelled=lambda: True)
    with pytest.raises(jobs.Cancelled):
        ST.separate_array(mix, SR, cancelled=lambda: True)
    job = jobs.Job("stems", lambda progress, cancelled, status:
                   ST.remove_vocals(src, 0, None, progress, cancelled, status, engine="spectral"))
    job.cancel()
    job.run_sync()
    assert job.error == "Cancelled" and job.result is None
    assert not os.listdir(jobs.output_dir("stems"))


@pytest.mark.parametrize("f", [70.0, 110.0, 196.0, 260.0, 440.0])
@pytest.mark.parametrize("sr", [16000, 48000])
def test_f0_estimator_pure_tones(f, sr):
    t = np.arange(int(sr * 1.5)) / sr
    est = VC.estimate_f0(0.3 * np.sin(2 * np.pi * f * t), sr)
    assert abs(est - f) / f < 0.03, (f, est)


def test_f0_estimator_harmonic_and_silence():
    sr = 16000
    t = np.arange(sr * 2) / sr
    x = sum(np.sin(2 * np.pi * 150 * k * t) / k for k in range(1, 8))  # strong harmonics
    assert abs(VC.estimate_f0(x, sr) - 150) / 150 < 0.03
    assert VC.estimate_f0(np.zeros(sr), sr) == 0.0


def test_engines_listing():
    av = VC.available_engines()
    assert av and av[-1] == "match"
    rows = VC.engine_status()
    assert {r["id"] for r in rows} == set(VC.ENGINES)
    assert VC.ENGINES["match"]["label"] == "Voice match (approximate, offline)"
    assert not VC.ENGINES["match"]["real_clone"]
    for e in VC.ENGINES:
        assert isinstance(VC.install_hint(e), str)
    assert "pip install" in VC.install_hint("f5tts") and "GB" in VC.install_hint("xtts")
    with pytest.raises(ValueError):
        VC.install_hint("nope")


def _sapi_ok():
    try:
        from clipeditor import tts
        return tts.engines().get("sapi") == "ok"
    except Exception:
        return False


TEXT = "Hello there, this is a short test of the voice matching feature in the editor."


@pytest.mark.skipif(not _sapi_ok(), reason="Windows SAPI voice unavailable")
def test_voice_match_moves_pitch_toward_reference(tmp_path):
    from clipeditor import tts
    sr = 48000
    t = np.arange(sr * 5) / sr
    f0 = 260 * (1 + 0.02 * np.sin(2 * np.pi * 5 * t))
    ph = 2 * np.pi * np.cumsum(f0) / sr
    ref = sum(np.exp(-((260 * k - 800) / 600) ** 2) * np.sin(k * ph) for k in range(1, 12))
    ref_path = str(tmp_path / "ref.wav")
    ST.write_wav(ref_path, 0.3 * ref / np.abs(ref).max(), sr)
    # the reference may be any media file (here AAC in .m4a)
    m4a = str(tmp_path / "ref.m4a")
    jobs.ffmpeg_run(["-i", ref_path, "-c:a", "aac", "-b:a", "128k", m4a])
    assert abs(VC.voice_profile(m4a)["f0"] - 260) / 260 < 0.03
    plain = tts.synthesize(TEXT, out_path=str(tmp_path / "plain.wav"))
    plain_f0 = VC.voice_profile(plain)["f0"]
    st = []
    t0 = time.time()
    out = VC.clone_and_speak(ref_path, TEXT, out_path=str(tmp_path / "match.wav"),
                             engine="match", status=st.append)
    el = time.time() - t0
    prof = VC.voice_profile(out)
    info = VC.clone_and_speak.last_info
    print(f"\n[voice] plain SAPI F0 {plain_f0:.0f} Hz -> matched {prof['f0']:.0f} Hz "
          f"(ref 260), base {info['base']['f0']:.0f} Hz, ratio {info['pitch_ratio']:.2f}, "
          f"tilt {info['tilt_db']:+.1f} dB, rubberband={info['rubberband']}, {el:.1f} s")
    assert os.path.isfile(out) and prof["seconds"] > 1.0
    assert abs(prof["f0"] - 260) < abs(plain_f0 - 260)
    assert abs(prof["f0"] - 260) / 260 < 0.15
    # asetrate+atempo path
    out2 = VC.clone_and_speak(ref_path, TEXT, out_path=str(tmp_path / "match2.wav"),
                              engine="match", rubberband=False)
    assert abs(VC.voice_profile(out2)["f0"] - 260) < abs(plain_f0 - 260)


def test_voice_clone_cancel_and_errors(tmp_path):
    ref_path = str(tmp_path / "ref.wav")
    t = np.arange(SR) / SR
    ST.write_wav(ref_path, 0.3 * np.sin(2 * np.pi * 200 * t), SR)
    with pytest.raises(jobs.Cancelled):
        VC.clone_and_speak(ref_path, TEXT, engine="match", cancelled=lambda: True)
    job = jobs.Job("clone", lambda progress, cancelled, status:
                   VC.clone_and_speak(ref_path, TEXT, None, "match", progress, cancelled, status))
    job.cancel()
    job.run_sync()
    assert job.error == "Cancelled"
    with pytest.raises(ValueError):
        VC.clone_and_speak(ref_path, "  ", engine="match")
    with pytest.raises(FileNotFoundError):
        VC.clone_and_speak(str(tmp_path / "missing.wav"), TEXT)
    for e in ("f5tts", "xtts", "openvoice"):
        if e not in VC.available_engines():
            with pytest.raises(RuntimeError):
                VC.clone_and_speak(ref_path, TEXT, engine=e)
