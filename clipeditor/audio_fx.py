"""Per-clip audio processing (voice cleanup, voice changer, music fx) + pitch-correct speed.

CONTRACT (used by clipeditor.audio_mix for preview AND export):
    PRESETS: dict[type, label]
    active(clip) -> bool                       # needs processed audio (audio_fx or speed != 1)
    clip_audio(clip, asset, cache, wait=False) -> np.ndarray | None
        Returns the clip's audio ALREADY in timeline time: float32 stereo at cache.sr,
        sample 0 == clip.start, length == clip.duration * sr (speed applied with pitch kept,
        effects applied). If not ready and wait=False: start a background job and return None
        (the mixer then falls back to raw audio). wait=True blocks until ready (export).

Extras for the UI:
    EFFECTS: {type: {"label", "category", "params": {name: {"label","min","max","default"}}}}
    CATEGORIES: ordered category names
    status(clip, asset=None, sr=48000) -> "ready" | "processing" | "error: ..." | "idle"
    render_preview(path, fx_list, seconds=5, start=0.0) -> float32 (n, 2) @ 48 kHz (blocking)
    app_dir(*sub) -> %APPDATA%/ClipEditor[/sub] (created)

Implementation: FFmpeg subprocess per (asset path, in, out, speed, fx) -> f32le stereo 48k on a
pipe; atempo chain for speed; results in an LRU (~600 MB). Background renders run on a
2-thread pool, in-flight keys are de-duplicated. Everything here is thread-safe.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import urllib.request
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

SR = 48000


# --------------------------------------------------------------------------- paths
def app_dir(*sub: str) -> str:
    """%APPDATA%/ClipEditor (same as QStandardPaths AppDataLocation for org/app ClipEditor)."""
    base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Roaming")
    d = os.path.join(base, "ClipEditor", *sub)
    os.makedirs(d, exist_ok=True)
    return d


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


# --------------------------------------------------------------------------- effect catalogue
def _I(default=0.5, label="Intensity"):
    return {"intensity": {"label": label, "min": 0.0, "max": 1.0, "default": default}}


CATEGORIES = ["Voice cleanup", "Voice changer", "Music"]

EFFECTS: dict[str, dict] = {
    # voice cleanup
    "denoise": {"label": "Reduce noise", "category": "Voice cleanup", "params": _I(0.5, "Strength")},
    "ai_denoise": {"label": "AI noise removal", "category": "Voice cleanup", "params": _I(1.0, "Mix")},
    "enhance_voice": {"label": "Enhance voice", "category": "Voice cleanup", "params": _I(0.5)},
    "normalize": {"label": "Normalize loudness (-14 LUFS)", "category": "Voice cleanup",
                  "params": {"lufs": {"label": "Target LUFS", "min": -30.0, "max": -8.0, "default": -14.0}}},
    "remove_hum": {"label": "Remove hum (50/60 Hz)", "category": "Voice cleanup",
                   "params": {"freq": {"label": "Mains Hz (0 = both)", "min": 0, "max": 60, "default": 0}}},
    # voice changer
    "chipmunk": {"label": "Chipmunk", "category": "Voice changer", "params": _I(0.5)},
    "deep": {"label": "Deep / monster", "category": "Voice changer", "params": _I(0.5)},
    "robot": {"label": "Robot", "category": "Voice changer", "params": _I(0.7)},
    "telephone": {"label": "Telephone", "category": "Voice changer", "params": _I(0.6)},
    "radio": {"label": "Radio", "category": "Voice changer", "params": _I(0.5)},
    "megaphone": {"label": "Megaphone", "category": "Voice changer", "params": _I(0.6)},
    "echo": {"label": "Echo", "category": "Voice changer", "params": _I(0.5)},
    "cave": {"label": "Cave / reverb", "category": "Voice changer", "params": _I(0.5)},
    "alien": {"label": "Alien", "category": "Voice changer", "params": _I(0.5)},
    "whisper": {"label": "Whisper", "category": "Voice changer", "params": _I(0.7)},
    "underwater": {"label": "Underwater", "category": "Voice changer", "params": _I(0.5)},
    # music
    "bass_boost": {"label": "Bass boost", "category": "Music", "params": _I(0.5)},
    "treble_boost": {"label": "Treble boost", "category": "Music", "params": _I(0.5)},
    "8d": {"label": "8D audio (rotating pan)", "category": "Music", "params": _I(0.7)},
    "lofi": {"label": "Lo-fi", "category": "Music", "params": _I(0.5)},
}

PRESETS: dict[str, str] = {k: v["label"] for k, v in EFFECTS.items()}


def default_params(fx_type: str) -> dict:
    return {k: p["default"] for k, p in EFFECTS.get(fx_type, {}).get("params", {}).items()}


# --------------------------------------------------------------------------- filter graph
_RNN_URL = "https://raw.githubusercontent.com/richardpl/arnndn-models/master/std.rnnn"
_RNN_NAME = "std.rnnn"
_rnn_lock = threading.Lock()
_rnn_failed = False
_rubberband: bool | None = None


def rnnoise_model() -> str | None:
    """Path of the RNNoise model (downloaded once into %APPDATA%/ClipEditor/models), or None."""
    global _rnn_failed
    with _rnn_lock:
        path = os.path.join(app_dir("models"), _RNN_NAME)
        if os.path.isfile(path) and os.path.getsize(path) > 1000:
            return path
        if _rnn_failed:
            return None
        try:
            tmp = path + ".part"
            with urllib.request.urlopen(_RNN_URL, timeout=20) as r, open(tmp, "wb") as f:
                f.write(r.read())
            if os.path.getsize(tmp) < 1000:
                raise IOError("model download too small")
            os.replace(tmp, path)
            return path
        except Exception:
            _rnn_failed = True
            return None


def _has_rubberband() -> bool:
    global _rubberband
    if _rubberband is None:
        try:
            out = subprocess.run([ffmpeg_exe(), "-hide_banner", "-filters"], capture_output=True,
                                 text=True, creationflags=_NO_WINDOW, timeout=20).stdout
            _rubberband = " rubberband " in out
        except Exception:
            _rubberband = False
    return _rubberband


def _pitch(ratio: float) -> str:
    """Pitch shift by `ratio` keeping duration."""
    if _has_rubberband():
        return f"rubberband=pitch={ratio:.4f}:formant=shifted"
    return f"asetrate={SR * ratio:.0f},aresample={SR},{_atempo(1.0 / ratio)}"


def _atempo(s: float) -> str:
    st = []
    while s > 2.0:
        st.append(2.0)
        s /= 2.0
    while s < 0.5:
        st.append(0.5)
        s /= 0.5
    st.append(s)
    return ",".join(f"atempo={x:.6f}" for x in st if abs(x - 1.0) > 1e-9) or "anull"


def _num(params: dict, name: str, default: float) -> float:
    try:
        return float(params.get(name, default))
    except (TypeError, ValueError):
        return default


def _fx_filter(fx: dict, uses_model: list) -> str | None:
    t = fx.get("type")
    p = fx.get("params") or {}
    i = max(0.0, min(1.0, _num(p, "intensity", default_params(t).get("intensity", 0.5))))
    if t == "denoise":
        return f"afftdn=nr={6 + 34 * i:.1f}:nf=-40:tn=1"
    if t == "ai_denoise":
        model = rnnoise_model()
        if model:
            uses_model.append(os.path.dirname(model))
            return f"arnndn=m={_RNN_NAME}:mix={max(0.05, i):.2f}"
        return "afftdn=nr=30:nf=-35:tn=1"
    if t == "enhance_voice":
        return (f"highpass=f=80,equalizer=f=250:t=q:w=1:g={-2 * i:.1f},"
                f"equalizer=f=3500:t=q:w=1.2:g={2 + 6 * i:.1f},"
                f"equalizer=f=10000:t=q:w=1:g={2 * i:.1f},"
                f"acompressor=threshold={0.125 - 0.08 * i:.3f}:ratio={2 + 4 * i:.1f}:attack=10:release=150:makeup={1 + 1.5 * i:.2f},"
                "alimiter=limit=0.95")
    if t == "normalize":
        lufs = max(-30.0, min(-8.0, _num(p, "lufs", -14.0)))
        return f"loudnorm=I={lufs:.1f}:TP=-1.5:LRA=11,aresample={SR}"
    if t == "remove_hum":
        f0 = _num(p, "freq", 0)
        bases = [50.0] if 45 <= f0 <= 55 else [60.0] if 55 < f0 <= 65 else [50.0, 60.0]
        notches = [f"bandreject=f={b * h:.0f}:t=q:w=12" for b in bases for h in (1, 2, 3, 4)]
        return ",".join(notches)
    if t == "chipmunk":
        return _pitch(1.3 + 0.7 * i)
    if t == "deep":
        return _pitch(0.85 - 0.3 * i) + f",bass=g={3 + 5 * i:.1f},alimiter=limit=0.95"
    if t == "robot":
        win = 256 if i < 0.34 else 512 if i < 0.67 else 1024  # power of 2 (else afftfilt is ~50x slower)
        return (f"asplit[a][b];[a]afftfilt=real='hypot(re\\,im)':imag='0':win_size={win}:overlap=0.75[r];"
                f"[b]volume={1 - i:.3f}[d];[r]volume={8 * i:.3f}[w];[d][w]amix=inputs=2:normalize=0,"
                "flanger=delay=2:depth=1:regen=60:speed=8,alimiter=limit=0.95")
    if t == "telephone":
        return (f"highpass=f={250 + 150 * i:.0f},lowpass=f={3800 - 800 * i:.0f},"
                f"acompressor=threshold=0.1:ratio=4:makeup=2,acrusher=bits={12 - 4 * i:.0f}:mix={0.3 + 0.4 * i:.2f}:mode=log")
    if t == "radio":
        return (f"highpass=f={300 + 200 * i:.0f},lowpass=f={6000 - 2000 * i:.0f},"
                f"equalizer=f=1500:t=q:w=1:g={3 + 4 * i:.1f},acompressor=threshold=0.08:ratio=6:makeup=2.5,"
                f"aphaser=in_gain=0.9:out_gain=0.9:decay=0.2:speed=0.3,alimiter=limit=0.9")
    if t == "megaphone":
        return (f"highpass=f={450 + 250 * i:.0f},lowpass=f={4200 - 1200 * i:.0f},"
                f"volume={2 + 6 * i:.1f},acrusher=bits={10 - 4 * i:.0f}:mix={0.4 + 0.4 * i:.2f}:mode=lin,"
                "aecho=0.8:0.5:20:0.25,alimiter=limit=0.9")
    if t == "echo":
        d = 200 + 500 * i
        return f"aecho=0.8:0.8:{d:.0f}|{2 * d:.0f}:{0.25 + 0.3 * i:.2f}|{0.1 + 0.2 * i:.2f}"
    if t == "cave":
        dl = "|".join(f"{x:.0f}" for x in (60, 130, 230, 370))
        dc = "|".join(f"{(0.25 + 0.35 * i) * k:.2f}" for k in (1.0, 0.8, 0.6, 0.45))
        return f"aecho=0.8:0.9:{dl}:{dc},aecho=0.8:0.7:{500 + 600 * i:.0f}:{0.15 + 0.25 * i:.2f}"
    if t == "alien":
        return (_pitch(1.15 + 0.25 * i) + f",vibrato=f={5 + 5 * i:.1f}:d={0.3 + 0.5 * i:.2f},"
                f"flanger=delay=5:depth=4:regen=40:speed=1.5")
    if t == "whisper":
        return (f"highpass=f={400 + 400 * i:.0f},asplit[a][b];"
                "[a]afftfilt=real='hypot(re\\,im)*cos((random(0)*2-1)*2*3.14)':"
                "imag='hypot(re\\,im)*sin((random(1)*2-1)*2*3.14)':win_size=128:overlap=0.8[w];"
                f"[b]volume={1 - i:.3f}[d];[w]volume={0.6 + 0.8 * i:.3f}[wv];[d][wv]amix=inputs=2:normalize=0,"
                "alimiter=limit=0.95")
    if t == "underwater":
        return (f"lowpass=f={900 - 600 * i:.0f},lowpass=f={1200 - 700 * i:.0f},"
                f"chorus=0.6:0.9:50|60:0.4|0.32:0.25|0.4:2|1.3,vibrato=f=1.5:d={0.1 + 0.3 * i:.2f},volume=1.6")
    if t == "bass_boost":
        return f"bass=g={4 + 14 * i:.1f}:f=100:w=0.6,alimiter=limit=0.95"
    if t == "treble_boost":
        return f"treble=g={3 + 12 * i:.1f}:f=4000:w=0.6,alimiter=limit=0.95"
    if t == "8d":
        return (f"apulsator=mode=sine:hz={0.08 + 0.12 * i:.3f}:amount={0.5 + 0.5 * i:.2f}:width=1,"
                f"aecho=0.8:0.5:40|80:{0.1 * i:.2f}|{0.05 * i:.2f}")
    if t == "lofi":
        return (f"acrusher=bits={12 - 6 * i:.0f}:samples={1 + 5 * i:.0f}:mix={0.4 + 0.5 * i:.2f}:mode=lin:aa=1,"
                f"lowpass=f={7000 - 4000 * i:.0f},highpass=f=120,"
                f"tremolo=f=0.4:d={0.05 + 0.1 * i:.2f}")
    return None


def build_filter(speed: float, fx_list: list) -> tuple[str, str | None]:
    """-> (filter graph for -filter_complex with [in]/[out] labels, cwd needed by arnndn)."""
    parts = [f"aresample={SR},aformat=sample_fmts=fltp:channel_layouts=stereo"]
    if abs(speed - 1.0) > 1e-6:
        parts.append(_atempo(speed))
    model_dirs: list = []
    for k, fx in enumerate(fx_list or []):
        if not isinstance(fx, dict) or fx.get("enabled", True) is False:
            continue
        f = _fx_filter(fx, model_dirs)
        if f:
            parts.append(re.sub(r"\[(\w+)\]", lambda m: f"[{m.group(1)}_{k}]", f))  # unique labels
    parts.append(f"aformat=sample_fmts=flt:sample_rates={SR}:channel_layouts=stereo")
    return "[0:a]" + ",".join(parts) + "[out]", (model_dirs[0] if model_dirs else None)


def _run_ffmpeg(path: str, start: float, length: float, speed: float, fx_list: list) -> np.ndarray:
    graph, cwd = build_filter(speed, fx_list)
    cmd = [ffmpeg_exe(), "-hide_banner", "-nostdin", "-loglevel", "error"]
    if start > 0:
        cmd += ["-ss", f"{start:.6f}"]
    if length > 0:
        cmd += ["-t", f"{length:.6f}"]
    cmd += ["-i", os.path.abspath(path), "-vn", "-sn", "-filter_complex", graph, "-map", "[out]",
            "-ac", "2", "-ar", str(SR), "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"]
    proc = subprocess.run(cmd, capture_output=True, cwd=cwd, creationflags=_NO_WINDOW)
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError(msg[-1] if msg else f"ffmpeg exited {proc.returncode}")
    a = np.frombuffer(proc.stdout, dtype="<f4")
    a = a[: len(a) // 2 * 2].reshape(-1, 2)
    a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32, copy=False)
    return a


def _fit(a: np.ndarray, n: int) -> np.ndarray:
    if len(a) == n:
        return np.ascontiguousarray(a)
    if len(a) > n:
        return np.ascontiguousarray(a[:n])
    out = np.zeros((n, 2), np.float32)
    out[: len(a)] = a
    return out


def _resample(a: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR or len(a) == 0:
        return a
    n = int(round(len(a) * sr / SR))
    x = np.linspace(0, len(a) - 1, n)
    return np.stack([np.interp(x, np.arange(len(a)), a[:, c]) for c in range(2)], 1).astype(np.float32)


# --------------------------------------------------------------------------- cache / jobs
MAX_CACHE_BYTES = 600 * 1024 * 1024

_lock = threading.Lock()
_cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_cache_bytes = 0
_inflight: dict[tuple, Future] = {}
_errors: dict[tuple, str] = {}
_pool = ThreadPoolExecutor(2, thread_name_prefix="audio_fx")


def active(clip) -> bool:
    return bool(clip.audio_fx) or abs(clip.speed - 1.0) > 1e-6


def _key(clip, asset, sr: int) -> tuple | None:
    if asset is None or not getattr(asset, "path", "") or not getattr(asset, "has_audio", True):
        return None
    try:
        fx = json.dumps(clip.audio_fx or [], sort_keys=True, default=str)
    except Exception:
        fx = repr(clip.audio_fx)
    return (asset.path, round(float(clip.in_point), 6), round(float(clip.out_point), 6),
            round(float(clip.speed or 1.0), 6), fx, sr)


def _store(key: tuple, arr: np.ndarray) -> None:
    global _cache_bytes
    with _lock:
        if key in _cache:
            return
        _cache[key] = arr
        _cache_bytes += arr.nbytes
        while _cache_bytes > MAX_CACHE_BYTES and len(_cache) > 1:
            _, old = _cache.popitem(last=False)
            _cache_bytes -= old.nbytes


def _render(key: tuple) -> np.ndarray:
    path, a, b, speed, fx_json, sr = key
    fx_list = json.loads(fx_json) if fx_json.startswith("[") else []
    speed = speed if speed > 0 else 1.0
    n = int(round((b - a) / speed * sr))
    try:
        arr = _fit(_resample(_run_ffmpeg(path, a, b - a, speed, fx_list), sr), n)
    except Exception as e:  # noqa: BLE001
        with _lock:
            _errors[key] = str(e) or e.__class__.__name__
            _inflight.pop(key, None)
        raise
    _store(key, arr)
    with _lock:
        _errors.pop(key, None)
        _inflight.pop(key, None)
    return arr


def _submit(key: tuple) -> Future:
    with _lock:
        fut = _inflight.get(key)
        if fut is None:
            fut = _pool.submit(_render, key)
            _inflight[key] = fut
        return fut


def clip_audio(clip, asset, cache, wait: bool = False):
    sr = getattr(cache, "sr", SR) if cache is not None else SR
    key = _key(clip, asset, sr)
    if key is None or key[2] <= key[1]:
        return None
    with _lock:
        arr = _cache.get(key)
        if arr is not None:
            _cache.move_to_end(key)
            return arr
        if key in _errors and not wait:
            return None
        fut = _inflight.get(key)
    if not wait:
        if fut is None:
            _submit(key)
        return None
    if fut is None:
        with _lock:
            _errors.pop(key, None)
        fut = _submit(key)
    try:
        return fut.result()
    except Exception:
        return None


def status(clip, asset=None, sr: int = SR) -> str:
    """"ready" | "processing" | "error: ..." | "idle" (not requested yet / nothing to do)."""
    if not active(clip):
        return "ready"
    key = _key(clip, asset, sr) if asset is not None else None
    with _lock:
        if key is None:
            # no asset given: look for any entry for this clip's params
            fx = json.dumps(clip.audio_fx or [], sort_keys=True, default=str)
            match = lambda k: (k[1], k[2], k[3], k[4]) == (round(clip.in_point, 6), round(clip.out_point, 6),
                                                         round(clip.speed or 1.0, 6), fx)
            if any(match(k) for k in _cache):
                return "ready"
            if any(match(k) for k in _inflight):
                return "processing"
            for k, msg in _errors.items():
                if match(k):
                    return "error: " + msg
            return "idle"
        if key in _cache:
            return "ready"
        if key in _inflight:
            return "processing"
        if key in _errors:
            return "error: " + _errors[key]
    return "idle"


def render_preview(path: str, fx_list: list, seconds: float = 5, start: float = 0.0) -> np.ndarray:
    """Blocking: first `seconds` of `path` (from `start`) through fx_list -> float32 (n, 2) @ 48k."""
    arr = _run_ffmpeg(path, max(0.0, start), float(seconds), 1.0, fx_list or [])
    return arr[: int(round(seconds * SR))]


def clear_cache() -> None:
    global _cache_bytes
    with _lock:
        _cache.clear()
        _errors.clear()
        _cache_bytes = 0


def cache_bytes() -> int:
    return _cache_bytes
