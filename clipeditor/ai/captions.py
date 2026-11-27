"""Offline auto-captions with faster-whisper (CPU, int8).

Qt-free: `timeline_audio` and `transcribe` are meant to run in a background thread;
`add_captions` mutates the project and must run on the thread that owns the Editor (GUI).

Typical flow::

    audio = timeline_audio(editor.project, cache)          # worker thread
    segs = transcribe(audio, "base", progress=..., status=..., cancelled=...)
    lines = split_for_screen(segs)
    add_captions(editor, lines)                             # GUI thread
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..model import Clip, Track, new_id

SR = 16000  # Whisper's input rate

MODELS = {
    "tiny": "Tiny (~75 MB download, fastest, rough accuracy)",
    "base": "Base (~145 MB download, fast, good accuracy)",
    "small": "Small (~485 MB download, slower, best accuracy)",
}

STYLES = {
    "subtitle": dict(font_size=54, bold=False, pos_y=0.36, bg_color="#000000aa"),
    "big": dict(font_size=84, bold=True, pos_y=0.15, bg_color=""),
}

MIN_DISPLAY = 0.6


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[tuple[float, float, str]] = field(default_factory=list)


def is_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except Exception:
        return False
    return True


# ---- audio -----------------------------------------------------------------
def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or len(x) == 0:
        return x.astype(np.float32, copy=False)
    n_out = int(round(len(x) * sr_out / sr_in))
    if sr_in % sr_out == 0:  # integer decimation with a boxcar anti-alias filter
        k = sr_in // sr_out
        m = len(x) // k
        y = x[: m * k].reshape(m, k).mean(axis=1)
        if len(y) < n_out:  # pad the trailing partial block
            y = np.concatenate([y, [x[m * k:].mean()]])
        return y.astype(np.float32)
    t = np.arange(n_out, dtype=np.float64) * (sr_in / sr_out)
    return np.interp(t, np.arange(len(x), dtype=np.float64), x).astype(np.float32)


def timeline_audio(project, cache, sr: int = SR, chunk_s: float = 10.0) -> np.ndarray:
    """Mono float32 mix of the whole timeline (what the viewer hears), resampled to `sr`.

    Loads any audio asset that is used on the timeline but not yet in `cache`."""
    from ..audio_mix import mix

    for asset in project.timeline_assets():
        if asset.has_audio and cache.get(asset.id) is None:
            cache.load(asset)
    csr = cache.sr
    total = int(round(project.duration * csr))
    step = max(int(chunk_s * csr), 1)
    parts = []
    for i0 in range(0, total, step):
        n = min(step, total - i0)
        parts.append(mix(project, cache, i0 / csr, n).mean(axis=1))
    mono = np.concatenate(parts) if parts else np.zeros(0, np.float32)
    return _resample(mono, csr, sr)


# ---- transcription -----------------------------------------------------------
_model_cache: dict[str, object] = {}


def _load_model(model_size: str, status):
    m = _model_cache.get(model_size)
    if m is not None:
        return m
    from faster_whisper import WhisperModel

    if status:
        status("Loading model…")
    try:  # already in the Hugging Face cache?
        m = WhisperModel(model_size, device="cpu", compute_type="int8", local_files_only=True)
    except Exception:
        if status:
            status(f"Downloading model ({MODELS.get(model_size, model_size)})…")
        m = WhisperModel(model_size, device="cpu", compute_type="int8")
    _model_cache[model_size] = m
    return m


def transcribe(audio16k: np.ndarray, model_size: str = "base", language: str | None = None,
               progress: Callable[[float], None] | None = None,
               cancelled: Callable[[], bool] | None = None,
               status: Callable[[str], None] | None = None) -> list[Segment]:
    """Speech-to-text with word timestamps. `audio16k` is mono float32 at 16 kHz.

    Returns the segments received so far if `cancelled()` becomes True."""
    audio = np.ascontiguousarray(audio16k, dtype=np.float32)
    total = len(audio) / SR
    if total <= 0:
        return []
    model = _load_model(model_size, status)
    if cancelled and cancelled():
        return []
    if status:
        status("Listening…")
    seg_iter, _info = model.transcribe(audio, language=language, word_timestamps=True,
                                       vad_filter=True, beam_size=5)
    out: list[Segment] = []
    for s in seg_iter:  # generator: decoding happens lazily here
        words = [(float(w.start), float(w.end), w.word.strip()) for w in (s.words or []) if w.word.strip()]
        out.append(Segment(float(s.start), float(s.end), s.text.strip(), words))
        if progress:
            progress(min(float(s.end) / total, 1.0))
        if cancelled and cancelled():
            break
    if progress and not (cancelled and cancelled()):
        progress(1.0)
    return out


# ---- layout ------------------------------------------------------------------
def _wrap(words: list[str], max_chars: int) -> list[str] | None:
    """Lay words out on 1 line, or 2 balanced lines, each <= max_chars; None if impossible."""
    one = " ".join(words)
    if len(one) <= max_chars or len(words) == 1:
        return [one]
    best = None
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        if len(a) <= max_chars and len(b) <= max_chars:
            score = abs(len(a) - len(b))
            if best is None or score < best[0]:
                best = (score, [a, b])
    return best[1] if best else None


def _words_of(seg: Segment) -> list[tuple[float, float, str]]:
    if seg.words:
        return seg.words
    toks = seg.text.split()  # no word timings: spread evenly over the segment
    if not toks:
        return []
    d = (seg.end - seg.start) / len(toks)
    return [(seg.start + i * d, seg.start + (i + 1) * d, w) for i, w in enumerate(toks)]


_SENT_END = re.compile(r"[.!?…]['\"”)]*$")


def split_for_screen(segments: list[Segment], max_chars: int = 42,
                     max_dur: float = 3.5) -> list[tuple[float, float, str]]:
    """Re-chunk word timings into readable captions: <= 2 lines of <= max_chars,
    <= max_dur seconds, >= 0.6 s on screen where possible, never overlapping."""
    words = [w for s in segments for w in _words_of(s)]
    groups: list[list[tuple[float, float, str]]] = []
    cur: list[tuple[float, float, str]] = []
    for w in words:
        if cur:
            gap = w[0] - cur[-1][1]
            too_long = w[1] - cur[0][0] > max_dur
            no_fit = _wrap([x[2] for x in cur] + [w[2]], max_chars) is None
            if gap > 1.0:
                groups.append(cur)
                cur = []
            elif too_long or no_fit:
                # cut after the last comma in the back half, if any, to keep phrases together
                k = next((i for i in range(len(cur) - 1, len(cur) // 2 - 1, -1)
                          if cur[i][2].endswith((",", ";", ":"))), len(cur) - 1)
                groups.append(cur[:k + 1])
                cur = cur[k + 1:]
        cur.append(w)
        # prefer breaking at sentence ends once the caption has some substance
        if _SENT_END.search(w[2]) and len(" ".join(x[2] for x in cur)) >= max_chars // 2:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)

    lines: list[tuple[float, float, str]] = []
    for g in groups:
        text = "\n".join(_wrap([x[2] for x in g], max_chars) or [" ".join(x[2] for x in g)])
        lines.append([g[0][0], g[-1][1], text])
    for i, ln in enumerate(lines):
        nxt = lines[i + 1][0] if i + 1 < len(lines) else None
        if ln[1] - ln[0] < MIN_DISPLAY:
            ln[1] = ln[0] + MIN_DISPLAY
        if nxt is not None and ln[1] > nxt:
            ln[1] = max(nxt, ln[0] + 0.05)
            if ln[1] > nxt:  # words packed tighter than 50 ms: push the next one
                lines[i + 1][0] = ln[1]
                lines[i + 1][1] = max(lines[i + 1][1], ln[1] + 0.05)
    return [(round(a, 3), round(b, 3), t) for a, b, t in lines]


# ---- editor integration ------------------------------------------------------
def add_captions(editor, lines: list[tuple[float, float, str]], style: str = "subtitle") -> int:
    """Add a new top video track "Captions" with one text clip per line (one undo step).

    Must be called on the thread that owns `editor` (it notifies listeners)."""
    if not lines:
        return 0
    st = STYLES.get(style, STYLES["subtitle"])
    editor.checkpoint()
    tr = Track(new_id(), "video", "Captions")
    editor.project.tracks.insert(0, tr)  # index 0 is drawn on top, like Editor.add_track
    for a, b, text in lines:
        dur = max(b - a, 1 / 30)
        tr.clips.append(Clip(new_id(), "", max(a, 0.0), 0.0, dur, text=text, **st))
    editor.notify("edit")
    return len(tr.clips)
