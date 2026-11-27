"""Speak text in (an approximation of) a reference speaker's voice.

    ENGINES                      id -> {"label", "package", "pip", "size", "note", "real_clone"}
    available_engines() -> [id]  usable right now, best first; always ends with "match"
    engine_status() -> [{"id", "label", "available", "real_clone", "note"}]   for the UI
    install_hint(engine) -> str  exact pip command + download size
    clone_and_speak(reference_wav, text, out_path=None, engine=None, progress=None,
                    cancelled=None, status=None, **opts) -> wav path
    estimate_f0(x, sr) -> median F0 (Hz) of voiced frames;  voice_profile(path) -> dict

Engines (all optional except "match"; detected at runtime, never installed automatically):
  * "f5tts"     F5-TTS (`f5_tts` package): zero-shot cloning from a few seconds of audio.
  * "xtts"      Coqui XTTS v2 (`TTS` package, e.g. `pip install coqui-tts`). Its model is under
                the Coqui Public Model License (non-commercial); the UI must ask the user to
                accept it and pass `accept_license=True` (sets COQUI_TOS_AGREED=1).
  * "openvoice" OpenVoice v2 tone-colour converter (`openvoice` package + checkpoints_v2 in
                %APPDATA%/ClipEditor/models/openvoice). Base speech comes from the Windows voice.
  * "match"     Voice match (approximate, offline): NOT real cloning. Speaks with the offline
                Windows voice (clipeditor.tts), then shifts its pitch toward the reference
                speaker's median F0 (autocorrelation/YIN pitch tracker) and tilts its spectrum
                toward the reference's spectral centroid (FFmpeg rubberband or
                asetrate+atempo, then shelving EQ).

The reference may be any media file; up to 30 s of its audio is extracted with FFmpeg.
Outputs go to output_dir("voice"). Blocking; run inside a `jobs.Job`.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import tempfile
import uuid
from typing import Callable

import numpy as np

from .jobs import Cancelled, check_cancel, ffmpeg_has_filter, ffmpeg_run, output_dir
from .stems import extract_audio, read_wav

Progress = Callable[[float], None] | None
Cancel = Callable[[], bool] | None
Status = Callable[[str], None] | None

REF_MAX_SECONDS = 30.0
ANALYSIS_SR = 16000
OUT_SR = 48000

ENGINES: dict[str, dict] = {
    "f5tts": {
        "label": "F5-TTS (voice cloning, AI)",
        "package": "f5_tts",
        "pip": "pip install f5-tts",
        "size": "~1.4 GB (F5TTS_v1_Base model ~1.3 GB + Vocos vocoder, downloaded on first use; "
                "plus Whisper for transcribing the reference, ~1.6 GB; needs PyTorch)",
        "note": "Best quality; slow on CPU (often 5-20x real time).",
        "real_clone": True,
    },
    "xtts": {
        "label": "Coqui XTTS v2 (voice cloning, AI)",
        "package": "TTS",
        "pip": "pip install coqui-tts",
        "size": "~1.9 GB (xtts_v2 model, downloaded on first use; needs PyTorch)",
        "note": "Coqui Public Model License: non-commercial use only; you must accept it.",
        "real_clone": True,
    },
    "openvoice": {
        "label": "OpenVoice v2 (tone-colour cloning, AI)",
        "package": "openvoice",
        "pip": "pip install git+https://github.com/myshell-ai/OpenVoice.git",
        "size": "~130 MB converter checkpoint (checkpoints_v2 from the OpenVoice release, unzip "
                "into %APPDATA%/ClipEditor/models/openvoice) + PyTorch",
        "note": "Converts the Windows voice's timbre; rhythm/accent stay the Windows voice's.",
        "real_clone": True,
    },
    "match": {
        "label": "Voice match (approximate, offline)",
        "package": "",
        "pip": "",
        "size": "built in",
        "note": "Not real cloning: the Windows voice with pitch and tone shifted toward the "
                "reference speaker.",
        "real_clone": False,
    },
}

_ORDER = ("f5tts", "xtts", "openvoice", "match")


def _status(status: Status, text: str) -> None:
    if status:
        status(text)


def _has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


def openvoice_ckpt_dir() -> str:
    base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "ClipEditor", "models", "openvoice", "checkpoints_v2", "converter")


def _engine_reason(eng: str) -> str | None:
    """None if usable, else why not."""
    if eng == "match":
        return None
    info = ENGINES.get(eng)
    if info is None:
        return "unknown engine"
    if not _has(info["package"]):
        return f"not installed ({info['pip']})"
    if eng == "openvoice":
        d = openvoice_ckpt_dir()
        if not (os.path.isfile(os.path.join(d, "config.json"))
                and os.path.isfile(os.path.join(d, "checkpoint.pth"))):
            return f"checkpoints missing (put config.json + checkpoint.pth in {d})"
    return None


def available_engines() -> list[str]:
    out = []
    for e in _ORDER:
        try:
            if _engine_reason(e) is None:
                out.append(e)
        except Exception:
            pass
    if "match" not in out:
        out.append("match")
    return out


def engine_status() -> list[dict]:
    rows = []
    for e in _ORDER:
        try:
            why = _engine_reason(e)
        except Exception as ex:  # noqa: BLE001
            why = str(ex)
        rows.append({"id": e, "label": ENGINES[e]["label"], "available": why is None,
                     "real_clone": ENGINES[e]["real_clone"],
                     "note": ENGINES[e]["note"] if why is None else why})
    return rows


def install_hint(engine: str) -> str:
    info = ENGINES.get(engine)
    if info is None:
        raise ValueError(f"Unknown voice engine {engine!r}")
    if not info["pip"]:
        return f"{info['label']}: built in, nothing to install."
    s = f"{info['pip']}   (download: {info['size']})"
    if engine == "xtts":
        s += ". Model licence: Coqui Public Model License (non-commercial)."
    return s


# ---- analysis ----------------------------------------------------------------
def _frames(x: np.ndarray, size: int, hop: int) -> np.ndarray:
    if len(x) < size:
        x = np.pad(x, (0, size - len(x)))
    from numpy.lib.stride_tricks import sliding_window_view
    return sliding_window_view(x, size)[::hop]


def f0_track(x: np.ndarray, sr: int, fmin: float = 60.0, fmax: float = 500.0,
             frame: float = 0.04, hop: float = 0.01, threshold: float = 0.15
             ) -> tuple[np.ndarray, np.ndarray]:
    """YIN-style autocorrelation pitch tracker. Returns (f0 per frame in Hz, voiced mask)."""
    x = np.asarray(x, np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    W = int(round(frame * sr))
    tmax = min(int(np.ceil(sr / fmin)), W - 2)
    tmin = max(int(np.floor(sr / fmax)), 2)
    fr = _frames(x, W + tmax, max(int(round(hop * sr)), 1))
    fr = fr - fr.mean(axis=1, keepdims=True)
    n = fr.shape[1]
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    # difference function d(tau) = sum_{j<W} (x_j - x_{j+tau})^2 via autocorrelation
    F = np.fft.rfft(fr, nfft, axis=1)
    head = np.zeros_like(fr)
    head[:, :W] = fr[:, :W]
    Fh = np.fft.rfft(head, nfft, axis=1)
    cross = np.fft.irfft(np.conj(Fh) * F, nfft, axis=1)[:, :tmax + 1]  # sum x_j x_{j+tau}
    sq = np.concatenate([np.zeros((len(fr), 1)), np.cumsum(fr * fr, axis=1)], axis=1)
    e0 = sq[:, W][:, None]
    taus = np.arange(tmax + 1)
    etau = sq[:, taus + W] - sq[:, taus]
    d = e0 + etau - 2.0 * cross
    d[:, 0] = 0.0
    cum = np.cumsum(d[:, 1:], axis=1) / np.arange(1, tmax + 1)
    cmnd = np.ones_like(d)
    cmnd[:, 1:] = d[:, 1:] / np.maximum(cum, 1e-12)
    rms = np.sqrt(e0[:, 0] / W)
    gate = rms > max(1e-4, 0.05 * np.percentile(rms, 95) if len(rms) else 0.0)
    f0 = np.zeros(len(fr))
    voiced = np.zeros(len(fr), bool)
    seg = cmnd[:, tmin:tmax]
    for i in np.nonzero(gate)[0]:
        row = seg[i]
        below = np.nonzero(row < threshold)[0]
        if len(below):
            j = below[0]
            while j + 1 < len(row) and row[j + 1] < row[j]:
                j += 1
        else:
            j = int(np.argmin(row))
            if row[j] > 0.35:
                continue
        t = j + tmin
        if 1 <= t < tmax:  # parabolic interpolation
            a, b, c = cmnd[i, t - 1], cmnd[i, t], cmnd[i, t + 1]
            den = a - 2 * b + c
            t = t + (0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0)
        f0[i] = sr / t
        voiced[i] = True
    return f0, voiced


def estimate_f0(x: np.ndarray, sr: int, fmin: float = 60.0, fmax: float = 500.0) -> float:
    """Median F0 (Hz) over voiced frames; 0.0 if nothing voiced."""
    f0, v = f0_track(x, sr, fmin, fmax)
    return float(np.median(f0[v])) if v.any() else 0.0


def spectral_centroid(x: np.ndarray, sr: int) -> float:
    """Energy-weighted mean spectral centroid (Hz) of the active frames."""
    x = np.asarray(x, np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    fr = _frames(x, 1024, 512) * np.hanning(1024)
    mag = np.abs(np.fft.rfft(fr, axis=1))
    e = mag.sum(axis=1)
    act = e > 0.05 * (np.percentile(e, 95) if len(e) else 0.0)
    if not act.any():
        return 0.0
    freqs = np.fft.rfftfreq(1024, 1.0 / sr)
    c = (mag[act] * freqs).sum(axis=1) / np.maximum(e[act], 1e-12)
    return float(np.average(c, weights=e[act]))


def _load_mono(path: str, sr: int, max_seconds: float | None, cancelled: Cancel = None) -> np.ndarray:
    tmp = tempfile.mkdtemp(prefix="ce_vc_")
    try:
        wav = os.path.join(tmp, "a.wav")
        span = max_seconds if max_seconds else 36000.0
        extract_audio(path, 0.0, span, wav, sr, channels=1, cancelled=cancelled)
        x, _ = read_wav(wav)
        return x[:, 0]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def voice_profile(path: str, cancelled: Cancel = None) -> dict:
    """{'f0': median Hz, 'centroid': Hz, 'seconds': analysed length} of a media file's audio."""
    x = _load_mono(path, ANALYSIS_SR, REF_MAX_SECONDS, cancelled)
    return {"f0": estimate_f0(x, ANALYSIS_SR), "centroid": spectral_centroid(x, ANALYSIS_SR),
            "seconds": len(x) / ANALYSIS_SR}


# ---- engines -------------------------------------------------------------------
def _default_out(engine: str) -> str:
    return os.path.join(output_dir("voice"), f"voice_{engine}_{uuid.uuid4().hex[:8]}.wav")


def _speak_f5(ref: str, text: str, out: str, status: Status, opts: dict) -> str:
    _status(status, "Loading F5-TTS (downloads ~1.4 GB on first use)…")
    from f5_tts.api import F5TTS  # type: ignore
    try:
        model = F5TTS(device=opts.get("device", "cpu"))
    except TypeError:  # older releases have no device argument
        model = F5TTS()
    _status(status, "Cloning voice (F5-TTS)")
    model.infer(ref_file=ref, ref_text=opts.get("ref_text", ""), gen_text=text, file_wave=out,
                speed=float(opts.get("speed", 1.0)), remove_silence=False)
    return out


def _speak_xtts(ref: str, text: str, out: str, status: Status, opts: dict) -> str:
    if opts.get("accept_license"):
        os.environ["COQUI_TOS_AGREED"] = "1"
    elif os.environ.get("COQUI_TOS_AGREED") != "1":
        raise RuntimeError("XTTS needs the Coqui Public Model License (non-commercial) to be "
                           "accepted first; ask the user, then pass accept_license=True.")
    _status(status, "Loading XTTS v2 (downloads ~1.9 GB on first use)…")
    from TTS.api import TTS  # type: ignore
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    try:
        tts.to("cpu")
    except Exception:
        pass
    _status(status, "Cloning voice (XTTS)")
    tts.tts_to_file(text=text, speaker_wav=ref, language=opts.get("language", "en"), file_path=out)
    return out


def _speak_openvoice(ref: str, text: str, out: str, tmp: str, status: Status, opts: dict) -> str:
    from clipeditor import tts as sapi
    _status(status, "Speaking base voice (Windows)")
    base = sapi.synthesize(text, opts.get("voice_id"), int(opts.get("rate", 0)),
                           os.path.join(tmp, "base.wav"))
    _status(status, "Loading OpenVoice converter")
    from openvoice import se_extractor  # type: ignore
    from openvoice.api import ToneColorConverter  # type: ignore
    d = openvoice_ckpt_dir()
    conv = ToneColorConverter(os.path.join(d, "config.json"), device="cpu")
    conv.load_ckpt(os.path.join(d, "checkpoint.pth"))
    _status(status, "Extracting tone colour")
    src_se, _ = se_extractor.get_se(base, conv, target_dir=tmp, vad=False)
    tgt_se, _ = se_extractor.get_se(ref, conv, target_dir=tmp, vad=True)
    _status(status, "Converting voice (OpenVoice)")
    conv.convert(audio_src_path=base, src_se=src_se, tgt_se=tgt_se, output_path=out)
    return out


def _atempo_chain(factor: float) -> list[str]:
    out, f = [], float(factor)
    while f > 2.0:
        out.append("atempo=2.0")
        f /= 2.0
    while f < 0.5:
        out.append("atempo=0.5")
        f /= 0.5
    out.append(f"atempo={f:.6f}")
    return out


def _pick_sapi_voice(ref_f0: float) -> str | None:
    """A SAPI voice whose gender matches the reference pitch (None = default voice)."""
    try:
        from clipeditor import tts as sapi
        vs = [v for v in sapi.voices() if v.get("engine") == "sapi"]
    except Exception:
        return None
    if not vs or ref_f0 <= 0:
        return None
    want = "female" if ref_f0 >= 165.0 else "male"
    for v in vs:
        if str(v.get("gender", "")).lower() == want and str(v.get("language", "")).lower().startswith("en"):
            return v["id"]
    for v in vs:
        if str(v.get("gender", "")).lower() == want:
            return v["id"]
    return None


def match_filter(ref: dict, base: dict, sr: int, use_rubberband: bool | None = None) -> tuple[str, dict]:
    """FFmpeg -af chain moving `base` (f0/centroid) toward `ref`. Returns (filter, info)."""
    ratio = 1.0
    if ref.get("f0", 0) > 0 and base.get("f0", 0) > 0:
        ratio = float(np.clip(ref["f0"] / base["f0"], 0.5, 2.0))
    if use_rubberband is None:
        use_rubberband = ffmpeg_has_filter("rubberband")
    f: list[str] = []
    if abs(ratio - 1.0) > 0.01:
        if use_rubberband:
            f.append(f"rubberband=pitch={ratio:.5f}:formant=preserved:pitchq=quality")
        else:  # resample-style shift (moves formants too), then restore duration
            f += [f"asetrate={sr * ratio:.3f}", f"aresample={sr}", *_atempo_chain(1.0 / ratio)]
    # spectral tilt toward the reference centroid (after the pitch shift, formants are kept
    # by rubberband, so the centroid change comes mostly from this EQ)
    tilt = 0.0
    bc = base.get("centroid", 0.0)
    if not use_rubberband:
        bc *= ratio  # asetrate moved the whole spectrum
    if ref.get("centroid", 0) > 0 and bc > 0:
        tilt = float(np.clip(8.0 * np.log2(ref["centroid"] / bc), -9.0, 9.0))
    if abs(tilt) > 0.5:
        f += [f"treble=g={tilt / 2:.2f}:f=2500:t=s:w=0.7", f"bass=g={-tilt / 2:.2f}:f=300:t=s:w=0.7"]
    f += ["alimiter=limit=0.95:level=disabled", f"aresample={OUT_SR}"]
    return ",".join(f), {"pitch_ratio": ratio, "tilt_db": tilt, "rubberband": use_rubberband}


def _speak_match(ref_path: str, text: str, out: str, tmp: str, progress: Progress,
                 cancelled: Cancel, status: Status, opts: dict) -> str:
    from clipeditor import tts as sapi
    _status(status, "Analysing reference voice")
    ref = voice_profile(ref_path, cancelled)
    if ref["f0"] <= 0:
        raise RuntimeError("Couldn't find a voiced (pitched) speech in the reference audio.")
    if progress:
        progress(0.2)
    check_cancel(cancelled)
    _status(status, "Speaking (Windows voice)")
    voice_id = opts.get("voice_id") or _pick_sapi_voice(ref["f0"])
    base_wav = sapi.synthesize(text, voice_id, int(opts.get("rate", 0)),
                               os.path.join(tmp, "base.wav"))
    if progress:
        progress(0.6)
    check_cancel(cancelled)
    _status(status, "Matching pitch and tone")
    bx = _load_mono(base_wav, ANALYSIS_SR, None, cancelled)
    base = {"f0": estimate_f0(bx, ANALYSIS_SR), "centroid": spectral_centroid(bx, ANALYSIS_SR)}
    amount = float(np.clip(opts.get("amount", 1.0), 0.0, 1.0))
    tgt = {"f0": base["f0"] * (ref["f0"] / base["f0"]) ** amount if base["f0"] > 0 else 0.0,
           "centroid": base["centroid"] * (ref["centroid"] / base["centroid"]) ** amount
           if base["centroid"] > 0 else 0.0}
    af, info = match_filter(tgt, base, OUT_SR, opts.get("rubberband"))
    try:
        ffmpeg_run(["-i", base_wav, "-af", af, "-ac", "1", "-c:a", "pcm_s16le", out],
                   len(bx) / ANALYSIS_SR, progress, cancelled, 0.65, 1.0)
    except RuntimeError:
        if not info["rubberband"]:
            raise
        af, info = match_filter(tgt, base, OUT_SR, False)
        ffmpeg_run(["-i", base_wav, "-af", af, "-ac", "1", "-c:a", "pcm_s16le", out],
                   len(bx) / ANALYSIS_SR, progress, cancelled, 0.65, 1.0)
    clone_and_speak.last_info = {"engine": "match", "reference": ref, "base": base, **info}
    return out


def clone_and_speak(reference_wav: str, text: str, out_path: str | None = None,
                    engine: str | None = None, progress: Progress = None,
                    cancelled: Cancel = None, status: Status = None, **opts) -> str:
    """Speak `text` in the reference speaker's voice -> WAV path.

    engine: None = best available (see available_engines()); "match" is always possible.
    opts: voice_id / rate (Windows voice), amount (0..1, match strength), language (xtts),
    ref_text (f5tts), accept_license (xtts), rubberband (force True/False for match)."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Nothing to say: the text is empty.")
    if not reference_wav or not os.path.isfile(reference_wav):
        raise FileNotFoundError(f"Reference audio not found: {reference_wav}")
    if engine is None:
        avail = available_engines()
        # XTTS needs an explicit licence acceptance, so don't auto-pick it without one
        if "xtts" in avail and not (opts.get("accept_license")
                                    or os.environ.get("COQUI_TOS_AGREED") == "1"):
            avail.remove("xtts")
        engine = avail[0]
    if engine not in ENGINES:
        raise ValueError(f"Unknown voice engine {engine!r}")
    why = _engine_reason(engine)
    if why:
        raise RuntimeError(f"{ENGINES[engine]['label']}: {why}")
    out = os.path.abspath(out_path or _default_out(engine))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="ce_vclone_")
    try:
        check_cancel(cancelled)
        if engine == "match":
            result = _speak_match(reference_wav, text, out, tmp, progress, cancelled, status, opts)
        else:
            _status(status, "Preparing reference audio")
            ref = os.path.join(tmp, "reference.wav")
            sr = 24000 if engine == "f5tts" else 22050
            extract_audio(reference_wav, 0.0, REF_MAX_SECONDS, ref, sr, channels=1,
                          cancelled=cancelled)
            if progress:
                progress(0.1)
            check_cancel(cancelled)
            # the model call itself can't be interrupted; cancellation is honoured after it
            raw = os.path.join(tmp, "raw.wav")
            try:
                if engine == "f5tts":
                    _speak_f5(ref, text, raw, status, opts)
                elif engine == "xtts":
                    _speak_xtts(ref, text, raw, status, opts)
                else:
                    _speak_openvoice(ref, text, raw, tmp, status, opts)
            except Cancelled:
                raise
            except Exception as e:  # wrap third-party failures with the engine name
                raise RuntimeError(f"{ENGINES[engine]['label']} failed: {type(e).__name__}: {e}") from e
            check_cancel(cancelled)
            ffmpeg_run(["-i", raw, "-ac", "1", "-ar", str(OUT_SR), "-c:a", "pcm_s16le", out],
                       0.0, None, cancelled)
            clone_and_speak.last_info = {"engine": engine}
            result = out
    except BaseException:
        try:
            if os.path.isfile(out) and not out_path:
                os.remove(out)
        except OSError:
            pass
        raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if progress:
        progress(1.0)
    return result


clone_and_speak.last_info = {}  # type: ignore[attr-defined]
