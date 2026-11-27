"""Stem separation: split a clip range's audio into "vocals" and "music" WAVs.

    separate(path, in_point, out_point, stems=("vocals", "music"), progress, cancelled, status,
             engine=None) -> {stem_name: wav_path}
    remove_vocals(path, in_point, out_point, ...) -> wav path   (the "music" stem)
    isolate_vocals(path, in_point, out_point, ...) -> wav path  (the "vocals" stem)
    ENGINES, engine_available(name), best_engine()

Engines:
  * "demucs"   - Meta's Demucs (htdemucs, two-stems vocals), only if `import demucs` works.
                 Runs `python -m demucs` in a subprocess (cancellable). Downloads its model
                 (~80 MB) on first use.
  * "spectral" - always available, numpy only. STFT (hann 2048 / hop 512) mid/side analysis:
                 harmonic-percussive separation (median filters along time and frequency)
                 x centre-panning weight x vocal-band (≈120 Hz-6 kHz) soft mask. The music
                 stem gets the complementary mask, so vocals + music reconstructs the input
                 (inverse STFT, weighted overlap-add). Processed in blocks to bound memory.

Outputs: 16-bit stereo WAVs in output_dir("stems"). Qt-free; run inside a `jobs.Job`.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
from typing import Callable

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .jobs import Cancelled, check_cancel, ffmpeg_run, output_dir

Progress = Callable[[float], None] | None
Cancel = Callable[[], bool] | None
Status = Callable[[str], None] | None

ENGINES: dict[str, dict] = {
    "demucs": {
        "label": "Demucs htdemucs (high quality, AI)",
        "pip": "pip install demucs",
        "size": "~80 MB model (downloaded on first use) + PyTorch (~200 MB CPU build)",
        "note": "Slow on CPU (roughly 1-3x real time).",
    },
    "spectral": {
        "label": "Spectral split (fast, approximate, offline)",
        "pip": "",
        "size": "built in",
        "note": "Works best on centre-panned vocals; leaves some bleed.",
    },
}

STEM_ALIASES = {"vocals": "vocals", "voice": "vocals", "music": "music",
                "accompaniment": "music", "instrumental": "music", "no_vocals": "music"}

N_FFT = 2048
HOP = 512
KERNEL_T = 17          # harmonic median filter length (frames, ~0.18 s @ 48 kHz)
KERNEL_F = 17          # percussive median filter length (bins)
BLOCK_FRAMES = 1024    # STFT frames per processing block (~11 s @ 48 kHz)
VOCAL_LO, VOCAL_HI = 120.0, 6000.0


def _status(status: Status, text: str) -> None:
    if status:
        status(text)


# ---- engine detection --------------------------------------------------------
_demucs_ok: bool | None = None


def engine_available(name: str) -> bool:
    global _demucs_ok
    if name == "spectral":
        return True
    if name == "demucs":
        if _demucs_ok is None:
            try:
                import importlib.util
                _demucs_ok = importlib.util.find_spec("demucs") is not None
            except Exception:
                _demucs_ok = False
        return bool(_demucs_ok)
    return False


def best_engine() -> str:
    return "demucs" if engine_available("demucs") else "spectral"


# ---- wav io ------------------------------------------------------------------
def read_wav(path: str) -> tuple[np.ndarray, int]:
    """16-bit PCM WAV -> float32 (n, channels), sr."""
    with wave.open(path, "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if sw != 2:
        raise ValueError(f"expected 16-bit PCM WAV, got {8 * sw}-bit")
    x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    return x.reshape(-1, ch), sr


def write_wav(path: str, x: np.ndarray, sr: int) -> str:
    x = np.asarray(x, np.float32)
    if x.ndim == 1:
        x = x[:, None]
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).round().astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(x.shape[1])
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())
    return path


def _probe_rate(path: str) -> int:
    try:
        import av
        with av.open(path) as c:
            if c.streams.audio:
                r = c.streams.audio[0].codec_context.sample_rate or 48000
                return 44100 if r == 44100 else 48000
    except Exception:
        pass
    return 48000


def _range(path: str, in_point, out_point) -> tuple[float, float]:
    t0 = max(float(in_point or 0.0), 0.0)
    dur = 0.0
    try:
        import av
        with av.open(path) as c:
            if not c.streams.audio:
                raise ValueError("This source has no audio")
            dur = c.duration / av.time_base if c.duration else 0.0
    except ValueError:
        raise
    except Exception:
        pass
    t1 = float(out_point) if out_point and float(out_point) > 0 else dur
    if dur:
        t1 = min(t1, dur)
    if t1 - t0 <= 0.01:
        raise ValueError(f"Empty range {t0:.2f}-{t1:.2f}")
    return t0, t1


def extract_audio(path: str, t0: float, span: float, out_wav: str, sr: int, channels: int = 2,
                  progress: Progress = None, cancelled: Cancel = None,
                  p0: float = 0.0, p1: float = 1.0) -> str:
    ffmpeg_run(["-ss", f"{t0:.6f}", "-t", f"{span:.6f}", "-i", os.path.abspath(path), "-vn",
                "-ac", str(channels), "-ar", str(sr), "-c:a", "pcm_s16le", out_wav],
               span, progress, cancelled, p0, p1)
    return out_wav


# ---- spectral engine ---------------------------------------------------------
def _median_filter(a: np.ndarray, k: int, axis: int) -> np.ndarray:
    """Running median of length k (odd) along `axis` with edge padding (strided windows)."""
    h = k // 2
    pad = [(0, 0)] * a.ndim
    pad[axis] = (h, h)
    ap = np.pad(a, pad, mode="edge")
    win = sliding_window_view(ap, k, axis=axis)  # window is the last axis
    return np.median(win, axis=-1).astype(a.dtype, copy=False)


def _band_weight(freqs: np.ndarray) -> np.ndarray:
    """Soft vocal-band weight: ~0 below 70 Hz, 1 between ~150 Hz and ~5 kHz, ~0 above 10 kHz."""
    lf = np.log2(np.maximum(freqs, 1.0))
    lo = 1.0 / (1.0 + np.exp(-(lf - np.log2(VOCAL_LO)) * 6.0))
    hi = 1.0 / (1.0 + np.exp((lf - np.log2(VOCAL_HI)) * 5.0))
    return (lo * hi).astype(np.float32)


def _vocal_mask(L: np.ndarray, R: np.ndarray, band: np.ndarray, strength: float) -> np.ndarray:
    """Soft vocal mask (frames, bins) from the stereo STFT block."""
    M = 0.5 * (L + R)
    S = 0.5 * (L - R)
    pm = (M.real ** 2 + M.imag ** 2).astype(np.float32)
    ps = (S.real ** 2 + S.imag ** 2).astype(np.float32)
    mag = np.sqrt(pm)
    eps = np.float32(1e-10)
    # harmonic/percussive (Wiener-style soft masks, power 2)
    H = _median_filter(mag, KERNEL_T, axis=0)
    P = _median_filter(mag, KERNEL_F, axis=1)
    H2, P2 = H * H, P * P
    harm = H2 / (H2 + P2 + eps)
    # centre weight: 1 for L==R, 0.5 for hard-panned, 0 for anti-phase -> remap to 0..1
    c = pm / (pm + ps + eps)
    centre = np.clip((c - 0.5) * 2.0, 0.0, 1.0) ** 1.5
    # smooth the centre cue a little over time so it doesn't flicker
    mask = harm * centre * band[None, :]
    # strength: >1 tightens (less bleed, thinner vocals), <1 loosens
    if strength != 1.0:
        mask = mask ** (1.0 / max(strength, 1e-3))
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def separate_array(x: np.ndarray, sr: int, progress: Progress = None, cancelled: Cancel = None,
                   strength: float = 1.0, p0: float = 0.0, p1: float = 1.0
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Spectral vocal/music split of float32 audio (n,) or (n, ch). Returns (vocals, music)
    with the input's shape; vocals + music ≈ x (exact up to float rounding)."""
    x = np.asarray(x, np.float32)
    mono_in = x.ndim == 1
    if mono_in:
        x = x[:, None]
    if x.shape[1] == 1:
        st = np.repeat(x, 2, axis=1)
    else:
        st = x[:, :2]
    n = len(st)
    if n == 0:
        z = np.zeros_like(x)
        return z, z.copy()
    win = np.hanning(N_FFT + 1)[:-1].astype(np.float32)  # periodic hann
    half = N_FFT // 2
    # pad so every sample is covered by full overlap; reflect needs n > half
    pad_mode = "reflect" if n > half + 1 else "constant"
    tail = (-(n + 2 * half - N_FFT)) % HOP
    xp = np.pad(st, ((half, half + tail), (0, 0)), mode=pad_mode)
    nfr = 1 + (len(xp) - N_FFT) // HOP
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    band = _band_weight(freqs)
    out_v = np.zeros_like(xp)
    out_m = np.zeros_like(xp)
    ctx = KERNEL_T // 2
    frames_view = [sliding_window_view(xp[:, c], N_FFT)[::HOP] for c in range(2)]
    nblocks = max(int(np.ceil(nfr / BLOCK_FRAMES)), 1)
    for bi in range(nblocks):
        check_cancel(cancelled)
        a, b = bi * BLOCK_FRAMES, min((bi + 1) * BLOCK_FRAMES, nfr)
        ca, cb = max(a - ctx, 0), min(b + ctx, nfr)
        Lf = np.fft.rfft(frames_view[0][ca:cb] * win, axis=1)
        Rf = np.fft.rfft(frames_view[1][ca:cb] * win, axis=1)
        m = _vocal_mask(Lf, Rf, band, strength)[a - ca: b - ca]
        im = 1.0 - m
        for c, X in enumerate((Lf[a - ca: b - ca], Rf[a - ca: b - ca])):
            _ola(out_v[:, c], np.fft.irfft(X * m, N_FFT, axis=1).astype(np.float32) * win, a)
            _ola(out_m[:, c], np.fft.irfft(X * im, N_FFT, axis=1).astype(np.float32) * win, a)
        if progress:
            progress(p0 + (p1 - p0) * b / nfr)
    # inverse STFT normalisation by the summed squared synthesis window
    wsum = np.zeros(len(xp), np.float32)
    _ola(wsum, np.broadcast_to(win * win, (nfr, N_FFT)), 0)
    norm = np.maximum(wsum[half:half + n], 1e-8)[:, None]
    vocals = (out_v[half:half + n] / norm).astype(np.float32)
    music = (out_m[half:half + n] / norm).astype(np.float32)
    if x.shape[1] == 1:
        vocals, music = vocals.mean(axis=1, keepdims=True), music.mean(axis=1, keepdims=True)
    if mono_in:
        vocals, music = vocals[:, 0], music[:, 0]
    return vocals, music


def _ola(out: np.ndarray, frames: np.ndarray, first: int) -> None:
    """Overlap-add `frames` (k, N_FFT) whose first frame index is `first` into `out` (1-D).
    Frames r, r+4, r+8... (N_FFT = 4*HOP) tile without overlapping, so add them as slabs."""
    step = N_FFT // HOP
    k = len(frames)
    for r in range(min(step, k)):
        sel = frames[r::step]
        s = (first + r) * HOP
        out[s:s + sel.size] += sel.reshape(-1)


# ---- demucs engine -----------------------------------------------------------
_HTDEMUCS_SIG = "955717e8"  # htdemucs checkpoint hash prefix in the torch hub cache
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _demucs_model_cached() -> bool:
    try:
        import torch.hub
        d = os.path.join(torch.hub.get_dir(), "checkpoints")
    except Exception:
        d = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")
    try:
        return any(_HTDEMUCS_SIG in f for f in os.listdir(d))
    except OSError:
        return False


def _run_demucs(wav: str, outdir: str, progress: Progress, cancelled: Cancel, status: Status,
                p0: float, p1: float) -> tuple[str, str]:
    if not _demucs_model_cached():
        _status(status, "Downloading model… (Demucs htdemucs, ~80 MB, first use only)")
    else:
        _status(status, "Separating stems (Demucs)")
    cmd = [sys.executable, "-m", "demucs", "-n", "htdemucs", "--two-stems", "vocals",
           "-d", "cpu", "-o", outdir, wav]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, creationflags=_NO_WINDOW)
    killed = False
    log: list[str] = []

    def watchdog():
        nonlocal killed
        while proc.poll() is None:
            if cancelled is not None and cancelled():
                killed = True
                proc.kill()
                return
            time.sleep(0.1)

    t = threading.Thread(target=watchdog, daemon=True)
    t.start()
    buf = b""
    try:
        while True:
            ch = proc.stdout.read(1)
            if not ch:
                break
            if ch in (b"\r", b"\n"):
                line = buf.decode("utf-8", "replace")
                buf = b""
                if line.strip():
                    log.append(line)
                    m = re.search(r"(\d{1,3})%\|", line)
                    if m and progress:
                        progress(p0 + (p1 - p0) * min(int(m.group(1)), 100) / 100.0)
                        if status:
                            status("Separating stems (Demucs)")
            else:
                buf += ch
    finally:
        proc.wait()
        t.join(timeout=2)
    if killed:
        raise Cancelled()
    if proc.returncode != 0:
        raise RuntimeError("Demucs failed: " + " | ".join(log[-4:])[-600:])
    base = os.path.join(outdir, "htdemucs", os.path.splitext(os.path.basename(wav))[0])
    v, m = os.path.join(base, "vocals.wav"), os.path.join(base, "no_vocals.wav")
    if not (os.path.isfile(v) and os.path.isfile(m)):
        raise RuntimeError("Demucs produced no output")
    return v, m


# ---- public API ----------------------------------------------------------------
def _norm_stems(stems) -> list[str]:
    if isinstance(stems, str):
        stems = (stems,)
    out = []
    for s in stems or ("vocals", "music"):
        k = STEM_ALIASES.get(str(s).lower())
        if k is None:
            raise ValueError(f"Unknown stem {s!r} (use 'vocals' or 'music')")
        if k not in out:
            out.append(k)
    return out


def separate(path: str, in_point: float = 0.0, out_point: float | None = None,
             stems=("vocals", "music"), progress: Progress = None, cancelled: Cancel = None,
             status: Status = None, engine: str | None = None, strength: float = 1.0) -> dict:
    """Separate the audio of `path[in_point:out_point]` into stems -> {stem: wav path}."""
    want = _norm_stems(stems)
    engine = engine or best_engine()
    if engine not in ENGINES:
        raise ValueError(f"Unknown stem engine {engine!r}")
    if not engine_available(engine):
        raise RuntimeError(f"{ENGINES[engine]['label']} is not installed "
                           f"({ENGINES[engine]['pip']})")
    t0, t1 = _range(path, in_point, out_point)
    span = t1 - t0
    sr = _probe_rate(path)
    stem = os.path.splitext(os.path.basename(path))[0][:40]
    tag = uuid.uuid4().hex[:8]
    odir = output_dir("stems")
    outs = {k: os.path.join(odir, f"{stem}_{k}_{tag}.wav") for k in want}
    tmp = tempfile.mkdtemp(prefix="ce_stems_")
    try:
        _status(status, "Extracting audio")
        wav = extract_audio(path, t0, span, os.path.join(tmp, f"{stem}_{tag}.wav"), sr,
                            progress=progress, cancelled=cancelled, p0=0.0, p1=0.05)
        check_cancel(cancelled)
        if engine == "demucs":
            v, m = _run_demucs(wav, tmp, progress, cancelled, status, 0.05, 0.97)
            src = {"vocals": v, "music": m}
            for k in want:  # demucs writes 44.1 kHz; resample back to the source rate
                ffmpeg_run(["-i", src[k], "-ar", str(sr), "-c:a", "pcm_s16le", outs[k]],
                           span, None, cancelled)
        else:
            x, sr = read_wav(wav)
            _status(status, "Separating stems (spectral)")
            v, m = separate_array(x, sr, progress, cancelled, strength, 0.05, 0.95)
            check_cancel(cancelled)
            _status(status, "Writing stems")
            if "vocals" in want:
                write_wav(outs["vocals"], v, sr)
            if "music" in want:
                write_wav(outs["music"], m, sr)
    except BaseException:
        for p in outs.values():
            try:
                os.remove(p)
            except OSError:
                pass
        raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if progress:
        progress(1.0)
    return outs


def remove_vocals(path, in_point=0.0, out_point=None, progress: Progress = None,
                  cancelled: Cancel = None, status: Status = None, engine: str | None = None,
                  strength: float = 1.0) -> str:
    """Instrumental / karaoke: returns the "music" stem WAV for the range."""
    return separate(path, in_point, out_point, ("music",), progress, cancelled, status,
                    engine, strength)["music"]


def isolate_vocals(path, in_point=0.0, out_point=None, progress: Progress = None,
                   cancelled: Cancel = None, status: Status = None, engine: str | None = None,
                   strength: float = 1.0) -> str:
    """Dialogue / a cappella: returns the "vocals" stem WAV for the range."""
    return separate(path, in_point, out_point, ("vocals",), progress, cancelled, status,
                    engine, strength)["vocals"]
