"""Royalty-free sound-effect library, synthesised procedurally with numpy.

Every effect is generated from code (no samples), so the output is free to use anywhere.
Files are written as 48 kHz stereo 16-bit WAVs into ``clipeditor/assets/sfx/`` on first
use and cached (regenerated only when ``_VERSION`` changes).

API (for the UI gallery):
    LIBRARY: dict[name, {"label", "category", "duration"}]
    CATEGORIES: list[str]                 display order
    ensure_library() -> str               folder with all WAVs (generates missing ones)
    path(name) -> str                     WAV path of one effect (generated on demand)
"""
from __future__ import annotations

import os
import wave

import numpy as np

SR = 48000
_VERSION = "1"
FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "sfx")
CATEGORIES = ["Transitions", "UI", "Comedy", "Music", "Impacts"]

# ---------------------------------------------------------------------------
# DSP helpers (mono float64 unless noted)
# ---------------------------------------------------------------------------


def _t(dur: float) -> np.ndarray:
    return np.arange(int(round(dur * SR))) / SR


def _rng(name: str) -> np.random.Generator:
    return np.random.default_rng(abs(hash_str(name)) % (2 ** 32))


def hash_str(s: str) -> int:
    h = 2166136261
    for ch in s.encode():
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h


def _phase(freq) -> np.ndarray:
    """Phase (radians) of an oscillator with an instantaneous frequency array."""
    return 2 * np.pi * np.cumsum(freq) / SR


def _sine(freq, n=None):
    if np.isscalar(freq):
        return np.sin(2 * np.pi * freq * np.arange(n) / SR)
    return np.sin(_phase(freq))


def _saw(ph):
    return 2.0 * ((ph / (2 * np.pi)) % 1.0) - 1.0


def _square(ph, duty=0.5):
    return np.where((ph / (2 * np.pi)) % 1.0 < duty, 1.0, -1.0)


def _tri(ph):
    return 2.0 * np.abs(_saw(ph)) - 1.0


def _exp_sweep(f0, f1, n, curve=1.0):
    x = np.linspace(0.0, 1.0, n) ** curve
    return f0 * (f1 / f0) ** x


def _decay(n, tau):
    return np.exp(-np.arange(n) / SR / tau)


def _adsr(n, a=0.005, d=0.05, s=0.7, r=0.1):
    env = np.full(n, s)
    na, nd, nr = int(a * SR), int(d * SR), int(r * SR)
    na = min(na, n)
    env[:na] = np.linspace(0, 1, na, endpoint=False)
    nd = min(nd, n - na)
    env[na:na + nd] = np.linspace(1, s, nd, endpoint=False)
    nr = min(nr, n)
    if nr:
        env[n - nr:] *= np.linspace(1, 0, nr)
    return env


def _fft_filter(x, lo=0.0, hi=None, slope=0.15):
    """Zero-phase band filter via FFT with soft log-frequency skirts."""
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    g = np.ones_like(f)
    lf = np.log2(np.maximum(f, 1.0))
    if lo > 0:
        g *= 1 / (1 + np.exp(-(lf - np.log2(lo)) / slope))
    if hi:
        g *= 1 / (1 + np.exp((lf - np.log2(hi)) / slope))
    return np.fft.irfft(X * g, n)


def _stft_shape(x, gain_fn, nfft=1024, hop=256):
    """Time-varying spectral shaping: gain_fn(freqs[None, :], times[:, None]) -> gains."""
    n = len(x)
    win = np.hanning(nfft)
    pad = np.concatenate([np.zeros(nfft), x, np.zeros(nfft)])
    nfr = (len(pad) - nfft) // hop + 1
    idx = np.arange(nfft)[None, :] + hop * np.arange(nfr)[:, None]
    frames = pad[idx] * win
    X = np.fft.rfft(frames, axis=1)
    freqs = np.fft.rfftfreq(nfft, 1 / SR)
    times = (np.arange(nfr) * hop + nfft / 2 - nfft) / SR
    G = gain_fn(freqs[None, :], np.clip(times, 0, None)[:, None])
    Y = np.fft.irfft(X * G, nfft, axis=1) * win
    out = np.zeros(len(pad))
    norm = np.zeros(len(pad))
    np.add.at(out, idx, Y)
    np.add.at(norm, idx, np.broadcast_to(win ** 2, Y.shape))
    out /= np.maximum(norm, 1e-3)
    return out[nfft:nfft + n]


def _band(freqs, fc, bw_oct):
    return np.exp(-0.5 * (np.log2(np.maximum(freqs, 1.0) / fc) / bw_oct) ** 2)


def _place(buf, sig, t0, gain=1.0):
    i = int(t0 * SR)
    if i >= len(buf):
        return
    m = min(len(sig), len(buf) - i)
    buf[i:i + m] += sig[:m] * gain


def _softclip(x, drive=1.0):
    return np.tanh(x * drive) / np.tanh(drive)


def _pan(mono, pan):
    """pan: scalar or array in -1..1 -> (n, 2) constant-power stereo."""
    p = (np.asarray(pan, dtype=float) + 1) * np.pi / 4
    return np.stack([mono * np.cos(p), mono * np.sin(p)], axis=1)


def _stereo_widen(mono, rng, amount=0.3, delay_ms=0.0):
    """Cheap width: decorrelate R with a tiny delay/side signal."""
    d = int(delay_ms * SR / 1000)
    r = np.concatenate([np.zeros(d), mono[: len(mono) - d]]) if d else mono.copy()
    l = mono
    return np.stack([l * (1 - amount * 0.5) + r * amount * 0.5, r * (1 - amount * 0.5) + l * amount * 0.5], 1)


def _bell_tone(freq, n, tau=0.8, ratio=1.4, index=2.5):
    """Chowning-style FM bell."""
    t = np.arange(n) / SR
    env = np.exp(-t / tau)
    mod = index * np.exp(-t / (tau * 0.6)) * np.sin(2 * np.pi * freq * ratio * t)
    return np.sin(2 * np.pi * freq * t + mod) * env


def _pluck(freq, n, tau=0.3, bright=6.0):
    t = np.arange(n) / SR
    sig = np.zeros(n)
    for k in range(1, 7):
        sig += np.sin(2 * np.pi * freq * k * t) * np.exp(-t * (1 / tau) * (1 + (k - 1) * bright / 6)) / k
    return sig * np.minimum(1, t / 0.002)


def _click_burst(rng, n_ms=8, lo=1500, hi=9000):
    n = int(n_ms * SR / 1000)
    x = _fft_filter(rng.standard_normal(n * 3), lo, hi)[:n]
    return x * _decay(n, n_ms / 1000 / 4)


# ---------------------------------------------------------------------------
# effects: each returns mono (n,) or stereo (n, 2) for the given duration
# ---------------------------------------------------------------------------


def _whoosh(rng, dur, f_lo=250, f_hi=2600, peak=0.55, bw=0.9):
    n = int(dur * SR)
    noise = rng.standard_normal(n)
    pk = dur * peak

    def gain(f, t):
        x = np.clip(t / dur, 0, 1)
        # centre freq rises to peak then falls
        u = np.where(t < pk, t / pk, 1 - (t - pk) / (dur - pk) * 0.7)
        fc = f_lo * (f_hi / f_lo) ** np.clip(u, 0, 1)
        return _band(f, fc, bw) * (1 + 0.0 * x)

    sig = _stft_shape(noise, gain)
    t = _t(dur)
    env = np.where(t < pk, (t / pk) ** 2.2, np.exp(-(t - pk) / (dur * 0.16)))
    sig *= env
    pan = np.clip(np.tanh((t - pk) / (dur * 0.25)) * 0.8, -1, 1)
    return _pan(sig, pan)


def fx_whoosh(rng, dur):
    return _whoosh(rng, dur)


def fx_swoosh_fast(rng, dur):
    return _whoosh(rng, dur, f_lo=700, f_hi=6000, peak=0.4, bw=0.7)


def fx_pop(rng, dur):
    n = int(dur * SR)
    f = _exp_sweep(1100, 180, n, 0.35)
    body = np.sin(_phase(f)) * _decay(n, 0.025)
    click = np.zeros(n)
    c = _click_burst(rng, 4, 2000, 12000)
    click[: len(c)] = c
    return body + 0.35 * click


def fx_bubble_pop(rng, dur):
    n = int(dur * SR)
    f = _exp_sweep(350, 1600, n, 0.5)
    env = np.minimum(1, np.arange(n) / (0.004 * SR)) * _decay(n, 0.05)
    s = np.sin(_phase(f)) * env
    second = np.zeros(n)
    _place(second, np.sin(_phase(_exp_sweep(700, 2400, n // 2, 0.5))) * _decay(n // 2, 0.025), 0.06)
    return s + 0.45 * second


def fx_click(rng, dur):
    n = int(dur * SR)
    out = np.zeros(n)
    _place(out, _click_burst(rng, 6, 1200, 9000), 0)
    _place(out, np.sin(2 * np.pi * 3200 * _t(0.02)) * _decay(int(0.02 * SR), 0.003), 0, 0.6)
    return out


def fx_ding(rng, dur):
    n = int(dur * SR)
    t = _t(dur)
    f = 1760
    sig = (np.sin(2 * np.pi * f * t) * np.exp(-t / 0.5)
           + 0.35 * np.sin(2 * np.pi * f * 2.76 * t) * np.exp(-t / 0.15)
           + 0.2 * np.sin(2 * np.pi * f * 5.40 * t) * np.exp(-t / 0.06))
    sig *= np.minimum(1, t / 0.002)
    return _stereo_widen(sig, rng, 0.4, 7)


def fx_bell(rng, dur):
    n = int(dur * SR)
    sig = _bell_tone(523.25, n, tau=0.9, ratio=1.4, index=3.0) + 0.4 * _bell_tone(1046.5, n, tau=0.5, ratio=3.5, index=1.0)
    return _stereo_widen(sig, rng, 0.5, 11)


def fx_chime(rng, dur):
    n = int(dur * SR)
    out = np.zeros((n, 2))
    notes = [1046.5, 1318.5, 1568.0, 2093.0]
    for i, f in enumerate(notes):
        m = n - int(i * 0.12 * SR)
        tone = _bell_tone(f, m, tau=0.7, ratio=2.0, index=1.2)
        st = _pan(tone, -0.6 + 0.4 * i)
        out[int(i * 0.12 * SR):] += st * 0.6
    return out


def fx_success_jingle(rng, dur):
    n = int(dur * SR)
    out = np.zeros(n)
    notes = [(0.0, 523.25), (0.09, 659.25), (0.18, 783.99), (0.30, 1046.5)]
    for i, (t0, f) in enumerate(notes):
        ln = 0.9 if i == 3 else 0.25
        m = int(ln * SR)
        ph = _phase(np.full(m, f))
        tone = (0.6 * _tri(ph) + 0.4 * np.sin(2 * ph)) * _adsr(m, 0.004, 0.08, 0.5, ln * 0.6)
        tone *= _decay(m, 0.35 if i == 3 else 0.12)
        _place(out, tone, t0)
        _place(out, _pluck(f * 2, m, 0.15), t0, 0.25)
    return _stereo_widen(out, rng, 0.35, 9)


def fx_error_buzz(rng, dur):
    n = int(dur * SR)
    out = np.zeros(n)
    for t0 in (0.0, 0.26):
        m = int(0.22 * SR)
        ph1 = _phase(np.full(m, 110.0))
        ph2 = _phase(np.full(m, 116.5))
        s = _saw(ph1) + _saw(ph2) + 0.5 * _square(ph1 * 2)
        s = _fft_filter(s, 60, 1800)
        s *= _adsr(m, 0.005, 0.03, 0.85, 0.04)
        _place(out, _softclip(s * 0.6, 2.0), t0)
    return out


def fx_notification(rng, dur):
    n = int(dur * SR)
    out = np.zeros(n)
    for t0, f in ((0.0, 880.0), (0.13, 1318.5)):
        m = n - int(t0 * SR)
        t = np.arange(m) / SR
        tone = (np.sin(2 * np.pi * f * t) + 0.25 * np.sin(2 * np.pi * f * 4 * t) * np.exp(-t / 0.03))
        tone *= np.exp(-t / 0.22) * np.minimum(1, t / 0.003)
        _place(out, tone, t0)
    return _stereo_widen(out, rng, 0.4, 6)


def fx_camera_shutter(rng, dur):
    n = int(dur * SR)
    out = np.zeros(n)
    for t0, g, lo, hi in ((0.0, 1.0, 900, 7000), (0.11, 0.8, 600, 5000)):
        m = int(0.05 * SR)
        x = _fft_filter(rng.standard_normal(m), lo, hi) * _decay(m, 0.008)
        _place(out, x, t0, g)
        thump = np.sin(_phase(_exp_sweep(300, 90, m))) * _decay(m, 0.01)
        _place(out, thump, t0, 0.5 * g)
    # spring/whirr
    m = int(0.09 * SR)
    whirr = _fft_filter(rng.standard_normal(m), 2500, 8000) * np.hanning(m) * 0.15
    _place(out, whirr, 0.015)
    return out


def _key(rng, soft=False):
    m = int(0.06 * SR)
    lo, hi = (1500, 6000) if soft else (900, 8000)
    x = _fft_filter(rng.standard_normal(m), lo, hi) * _decay(m, 0.006 if soft else 0.01)
    body = np.sin(_phase(_exp_sweep(rng.uniform(180, 260), 100, m))) * _decay(m, 0.012)
    return x + (0.3 if soft else 0.8) * body


def fx_typewriter(rng, dur):
    n = int(dur * SR)
    out = np.zeros(n)
    t0 = 0.02
    while t0 < dur - 0.45:
        _place(out, _key(rng), t0, rng.uniform(0.6, 1.0))
        t0 += rng.uniform(0.08, 0.16)
    # carriage-return bell
    _place(out, _bell_tone(2637, int(0.4 * SR), tau=0.2, ratio=1.0, index=0.5), dur - 0.42, 0.5)
    return _stereo_widen(out, rng, 0.3, 4)


def fx_keyboard_clicks(rng, dur):
    n = int(dur * SR)
    out = np.zeros((n, 2))
    t0 = 0.01
    while t0 < dur - 0.08:
        k = _key(rng, soft=True)
        st = _pan(k, rng.uniform(-0.5, 0.5))
        i = int(t0 * SR)
        m = min(len(st), n - i)
        out[i:i + m] += st[:m] * rng.uniform(0.5, 1.0)
        t0 += rng.choice([rng.uniform(0.05, 0.11), rng.uniform(0.12, 0.3)], p=[0.85, 0.15])
    return out


def _thump(m, f0=90, f1=40, tau=0.07):
    return np.sin(_phase(_exp_sweep(f0, f1, m, 0.5))) * _decay(m, tau) * np.minimum(1, np.arange(m) / (0.003 * SR))


def fx_heartbeat(rng, dur):
    n = int(dur * SR)
    out = np.zeros(n)
    for beat in (0.05, 0.85):
        m = int(0.3 * SR)
        _place(out, _thump(m, 80, 38, 0.08), beat, 1.0)
        _place(out, _thump(m, 70, 35, 0.07), beat + 0.22, 0.7)
    out = _fft_filter(out, 25, 400)
    return _softclip(out * 1.5, 1.5)


def fx_drum_hit(rng, dur):
    n = int(dur * SR)
    kick = np.sin(_phase(_exp_sweep(160, 45, n, 0.25))) * _decay(n, 0.18)
    m = int(0.03 * SR)
    click = np.zeros(n)
    click[:m] = _fft_filter(rng.standard_normal(m), 1500, 8000) * _decay(m, 0.004)
    return _softclip(kick * 1.2 + 0.3 * click, 1.8)


def fx_bass_drop(rng, dur):
    n = int(dur * SR)
    t = _t(dur)
    f = _exp_sweep(140, 32, n, 0.6)
    ph = _phase(f)
    sub = np.sin(ph) + 0.3 * np.sin(2 * ph)
    env = np.minimum(1, t / 0.01) * np.exp(-t / 1.0)
    sig = _softclip(sub * env * 1.6, 2.2)
    sig = _fft_filter(sig, 25, 2500)
    return _stereo_widen(sig, rng, 0.2, 0)


def fx_riser(rng, dur):
    n = int(dur * SR)
    t = _t(dur)
    noise = rng.standard_normal(n)

    def gain(f, tt):
        fc = 400 * (8000 / 400) ** np.clip(tt / dur, 0, 1) ** 1.5
        return _band(f, fc, 1.0)

    nz = _stft_shape(noise, gain)
    f = _exp_sweep(180, 1800, n, 1.6)
    ph = _phase(f)
    tone = 0.5 * (_saw(ph) + _saw(ph * 1.005)) * 0.5
    tone = _fft_filter(tone, 80, 6000)
    env = (t / dur) ** 2.2
    tail = np.minimum(1, (dur - t) / 0.03)
    out = (nz * 0.8 + tone * 0.35) * env * tail
    # slowly widening stereo
    l = out
    r = np.concatenate([np.zeros(240), out[:-240]])
    w = t / dur
    return np.stack([l * (1 - 0.3 * w) + r * 0.3 * w, r * (1 - 0.3 * w) + l * 0.3 * w], 1)


def fx_impact_boom(rng, dur):
    n = int(dur * SR)
    t = _t(dur)
    boom = np.sin(_phase(_exp_sweep(90, 30, n, 0.4))) * np.exp(-t / 0.6)
    rumble = _fft_filter(rng.standard_normal(n), 20, 220) * np.exp(-t / 0.5)
    rumble /= max(np.abs(rumble).max(), 1e-9)
    m = int(0.08 * SR)
    crack = np.zeros(n)
    crack[:m] = _fft_filter(rng.standard_normal(m), 600, 9000) * _decay(m, 0.012)
    sig = _softclip(boom * 1.1 + rumble * 0.6 + crack * 0.6, 2.0)
    return _stereo_widen(sig, rng, 0.35, 13)


def fx_glitch(rng, dur):
    n = int(dur * SR)
    out = np.zeros(n)
    i = 0
    while i < n:
        m = int(rng.uniform(0.015, 0.07) * SR)
        m = min(m, n - i)
        kind = rng.integers(0, 4)
        if kind == 0:
            seg = _square(_phase(np.full(m, rng.choice([220, 330, 660, 1320, 2640]))))
        elif kind == 1:
            seg = rng.standard_normal(m)
            seg = np.round(seg * 3) / 3  # bitcrush
        elif kind == 2:
            seg = np.sin(_phase(_exp_sweep(rng.uniform(300, 3000), rng.uniform(100, 5000), m)))
        else:
            seg = np.zeros(m)
        hold = int(rng.integers(1, 12))
        seg = np.repeat(seg[::hold], hold)[:m]  # sample-rate reduction
        if len(seg) < m:
            seg = np.pad(seg, (0, m - len(seg)))
        seg = seg * _adsr(m, 0.001, 0.001, 1.0, 0.002) * rng.uniform(0.3, 0.8)
        out[i:i + m] = seg
        i += m
    st = np.stack([out, np.roll(out, int(0.004 * SR))], 1)
    return st


def fx_laser(rng, dur):
    n = int(dur * SR)
    f = _exp_sweep(2400, 180, n, 0.5)
    ph = _phase(f)
    t = _t(dur)
    sig = np.sin(ph + 2.0 * np.exp(-t / 0.1) * np.sin(ph * 0.5)) * np.exp(-t / 0.18)
    sig *= np.minimum(1, t / 0.002)
    return _stereo_widen(sig, rng, 0.5, 9)


def fx_coin(rng, dur):
    n = int(dur * SR)
    out = np.zeros(n)
    m1 = int(0.07 * SR)
    out[:m1] = _square(_phase(np.full(m1, 987.77)), 0.5) * 0.5
    m2 = n - m1
    out[m1:] = _square(_phase(np.full(m2, 1318.5)), 0.5) * 0.5 * _decay(m2, 0.12)
    out = _fft_filter(out, 200, 9000)
    return out * _adsr(n, 0.001, 0.001, 1.0, 0.03)


def fx_magic_sparkle(rng, dur):
    n = int(dur * SR)
    out = np.zeros((n, 2))
    k = 70
    for _ in range(k):
        t0 = (rng.uniform(0, 1) ** 1.4) * (dur - 0.3)
        f = rng.uniform(2200, 7000)
        m = int(0.25 * SR)
        tt = np.arange(m) / SR
        blip = np.sin(2 * np.pi * f * tt) * np.exp(-tt / rng.uniform(0.03, 0.09)) * np.minimum(1, tt / 0.002)
        g = rng.uniform(0.2, 0.6) * (1 - t0 / dur) ** 0.7
        st = _pan(blip, rng.uniform(-0.9, 0.9)) * g
        i = int(t0 * SR)
        mm = min(m, n - i)
        out[i:i + mm] += st[:mm]
    # shimmer bed: rising airy noise
    t = _t(dur)

    def gain(f, tt):
        return _band(f, 6000 + 2000 * np.clip(tt / dur, 0, 1), 0.5)

    bed = _stft_shape(rng.standard_normal(n), gain) * np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 2 * 0.25
    out += _stereo_widen(bed, rng, 0.8, 5)
    # glissando tone
    gl = _bell_tone(1568, n, tau=0.6, ratio=2.0, index=0.8) * 0.3
    out += _pan(gl, 0.0)
    return out


def fx_tick_tock(rng, dur):
    n = int(dur * SR)
    out = np.zeros((n, 2))
    i = 0
    t0 = 0.02
    while t0 < dur - 0.1:
        f = 1900 if i % 2 == 0 else 1300
        m = int(0.08 * SR)
        tt = np.arange(m) / SR
        blk = (np.sin(2 * np.pi * f * tt) + 0.5 * np.sin(2 * np.pi * f * 1.52 * tt)) * np.exp(-tt / 0.012)
        cb = _click_burst(rng, 3, 2000, 10000)
        blk[: len(cb)] += 0.4 * cb
        st = _pan(blk, -0.25 if i % 2 == 0 else 0.25)
        a = int(t0 * SR)
        mm = min(m, n - a)
        out[a:a + mm] += st[:mm]
        t0 += 0.5
        i += 1
    return out


def fx_applause(rng, dur):
    n = int(dur * SR)
    out = np.zeros((n, 2))
    t = _t(dur)
    density = 160  # claps / second at peak
    total = int(density * dur * 0.7)
    for _ in range(total):
        t0 = rng.uniform(0, dur - 0.05)
        m = int(rng.uniform(0.008, 0.02) * SR)
        lo = rng.uniform(700, 1400)
        c = _fft_filter(rng.standard_normal(m * 2), lo, lo * 5)[:m] * _decay(m, m / SR / 3)
        st = _pan(c, rng.uniform(-0.9, 0.9)) * rng.uniform(0.3, 1.0)
        a = int(t0 * SR)
        out[a:a + m] += st
    env = np.minimum(1, t / 0.3) * np.minimum(1, (dur - t) / 0.9) ** 1.5
    return out * env[:, None]


def fx_record_scratch(rng, dur):
    n = int(dur * SR)
    t = _t(dur)
    # source "record": a mid-heavy noise + chord, read at a varying rate
    src_len = n * 3
    src = _fft_filter(rng.standard_normal(src_len), 300, 4000) * 0.6
    st = np.arange(src_len) / SR
    src += 0.4 * (np.sin(2 * np.pi * 220 * st) + np.sin(2 * np.pi * 277 * st) + np.sin(2 * np.pi * 330 * st)) / 3
    rate = np.where(t < 0.18, 3.2 * np.sin(np.pi * t / 0.18),
                    np.where(t < 0.36, -2.5 * np.sin(np.pi * (t - 0.18) / 0.18), 1.8 * np.sin(np.pi * np.clip((t - 0.36) / 0.3, 0, 1))))
    pos = np.cumsum(rate) + src_len / 2
    pos = np.clip(pos, 0, src_len - 2)
    i0 = pos.astype(int)
    fr = pos - i0
    sig = src[i0] * (1 - fr) + src[i0 + 1] * fr
    sig *= np.clip(np.abs(rate) / 1.5, 0, 1) ** 0.7
    sig = _fft_filter(sig, 150, 7000)
    return _stereo_widen(sig, rng, 0.2, 3)


def fx_cartoon_boing(rng, dur):
    n = int(dur * SR)
    t = _t(dur)
    base = 180 * (1 + 0.8 * np.minimum(1, t / 0.08))  # quick upward twang
    vib = 1 + 0.25 * np.sin(2 * np.pi * 14 * t) * np.exp(-t / 0.35)
    f = base * vib
    ph = _phase(f)
    sig = (np.sin(ph) + 0.5 * np.sin(2 * ph + 1.5 * np.sin(ph)) * np.exp(-t / 0.2))
    sig *= np.exp(-t / 0.3) * np.minimum(1, t / 0.004)
    return _stereo_widen(sig, rng, 0.3, 5)


def fx_rewind(rng, dur):
    n = int(dur * SR)
    t = _t(dur)
    u = t / dur
    # tape chirps: stutter rate accelerates, pitch rises
    stutter_rate = 8 + 30 * u ** 1.5
    gate = 0.5 + 0.5 * np.sin(_phase(stutter_rate))
    f = _exp_sweep(300, 2400, n, 1.3) * (1 + 0.15 * np.sin(_phase(stutter_rate * 0.5)))
    ph = _phase(f)
    tone = _fft_filter(_saw(ph) * 0.5 + np.sin(ph * 1.5) * 0.3, 150, 7000)

    def gain(fr, tt):
        return _band(fr, 1200 + 3000 * np.clip(tt / dur, 0, 1), 1.0)

    hiss = _stft_shape(rng.standard_normal(n), gain) * 0.4
    env = np.minimum(1, t / 0.05) * np.minimum(1, (dur - t) / 0.06)
    sig = (tone * gate ** 2 + hiss * (0.5 + 0.5 * gate)) * env
    return _stereo_widen(sig, rng, 0.4, 6)


# name -> (label, category, duration, generator)
_DEFS = {
    "whoosh": ("Whoosh", "Transitions", 1.0, fx_whoosh),
    "swoosh_fast": ("Swoosh (fast)", "Transitions", 0.45, fx_swoosh_fast),
    "riser": ("Riser", "Transitions", 3.0, fx_riser),
    "rewind": ("Rewind", "Transitions", 1.2, fx_rewind),
    "glitch": ("Glitch", "Transitions", 0.8, fx_glitch),
    "magic_sparkle": ("Magic Sparkle", "Transitions", 1.8, fx_magic_sparkle),
    "click": ("Click", "UI", 0.06, fx_click),
    "ding": ("Ding", "UI", 1.5, fx_ding),
    "notification": ("Notification", "UI", 0.8, fx_notification),
    "success_jingle": ("Success Jingle", "UI", 1.2, fx_success_jingle),
    "error_buzz": ("Error Buzz", "UI", 0.55, fx_error_buzz),
    "camera_shutter": ("Camera Shutter", "UI", 0.3, fx_camera_shutter),
    "typewriter": ("Typewriter", "UI", 1.8, fx_typewriter),
    "keyboard_clicks": ("Keyboard Clicks", "UI", 2.0, fx_keyboard_clicks),
    "coin": ("Coin", "UI", 0.5, fx_coin),
    "pop": ("Pop", "Comedy", 0.15, fx_pop),
    "bubble_pop": ("Bubble Pop", "Comedy", 0.25, fx_bubble_pop),
    "cartoon_boing": ("Cartoon Boing", "Comedy", 0.9, fx_cartoon_boing),
    "record_scratch": ("Record Scratch", "Comedy", 0.7, fx_record_scratch),
    "applause": ("Applause", "Comedy", 2.5, fx_applause),
    "laser": ("Laser", "Comedy", 0.5, fx_laser),
    "bell": ("Bell", "Music", 2.5, fx_bell),
    "chime": ("Chime", "Music", 2.0, fx_chime),
    "tick_tock": ("Tick-Tock", "Music", 2.0, fx_tick_tock),
    "heartbeat": ("Heartbeat", "Music", 1.6, fx_heartbeat),
    "drum_hit": ("Drum Hit", "Impacts", 0.6, fx_drum_hit),
    "bass_drop": ("Bass Drop", "Impacts", 2.5, fx_bass_drop),
    "impact_boom": ("Impact Boom", "Impacts", 2.0, fx_impact_boom),
}

LIBRARY: dict[str, dict] = {k: {"label": v[0], "category": v[1], "duration": v[2]} for k, v in _DEFS.items()}

PEAK = 10 ** (-1.0 / 20)  # normalise to -1 dBFS


def synth(name: str) -> np.ndarray:
    """Render one effect to float32 stereo (n, 2), peak-normalised to -1 dBFS."""
    label, cat, dur, gen = _DEFS[name]
    rng = _rng(name)
    x = np.asarray(gen(rng, dur), dtype=np.float64)
    if x.ndim == 1:
        x = np.stack([x, x], 1)
    n = int(round(dur * SR))
    if len(x) < n:
        x = np.pad(x, ((0, n - len(x)), (0, 0)))
    x = x[:n]
    x = np.nan_to_num(x)
    # de-click edges
    e = min(int(0.002 * SR), n // 4)
    if e:
        x[:e] *= np.sqrt(np.linspace(0, 1, e))[:, None]
        x[-e:] *= np.linspace(1, 0, e)[:, None]
    pk = np.abs(x).max()
    if pk > 0:
        x *= PEAK / pk
    return x.astype(np.float32)


def write_wav(fn: str, x: np.ndarray) -> None:
    pcm = np.clip(np.round(x * 32767), -32768, 32767).astype("<i2")
    tmp = fn + ".tmp"
    with wave.open(tmp, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    os.replace(tmp, fn)


def _stamp() -> str:
    return os.path.join(FOLDER, ".version")


def _fresh() -> bool:
    try:
        with open(_stamp(), encoding="utf-8") as f:
            return f.read().strip() == _VERSION
    except OSError:
        return False


def path(name: str) -> str:
    """WAV path of effect `name`, generating it if missing."""
    if name not in _DEFS:
        raise KeyError(f"unknown sound effect: {name}")
    fn = os.path.join(FOLDER, name + ".wav")
    if not os.path.exists(fn) or not _fresh():
        if not _fresh():
            return os.path.join(ensure_library(), name + ".wav")
        os.makedirs(FOLDER, exist_ok=True)
        write_wav(fn, synth(name))
    return fn


def ensure_library() -> str:
    """Generate any missing effects (all of them if the synth version changed); return the folder."""
    os.makedirs(FOLDER, exist_ok=True)
    fresh = _fresh()
    for name in _DEFS:
        fn = os.path.join(FOLDER, name + ".wav")
        if not fresh or not os.path.exists(fn):
            write_wav(fn, synth(name))
    if not fresh:
        with open(_stamp(), "w", encoding="utf-8") as f:
            f.write(_VERSION)
    return FOLDER


def by_category() -> dict[str, list[str]]:
    out = {c: [] for c in CATEGORIES}
    for k, v in LIBRARY.items():
        out[v["category"]].append(k)
    return out
