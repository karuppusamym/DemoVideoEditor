"""Real-time audio mixing of the timeline (used by preview playback).

Speed is applied by index stepping (pitch shifts, acceptable for preview); fades are a
vectorised per-sample gain ramp matching `Clip.fade_gain`.
"""
from __future__ import annotations

import numpy as np

from .media import AudioCache
from .model import Project
from . import animate, audio_fx
from .transitions import audio_fades


def _fade_ramp(clip, a: float, length: int, sr: int, fi: float, fo: float) -> np.ndarray | None:
    """Per-sample gain for timeline times a + i/sr, or None if no fade touches the span.

    fi/fo are the effective fade lengths (clip fades combined with transition crossfades)."""
    end = clip.end
    b = a + length / sr
    need_in = fi > 0 and a < clip.start + fi
    need_out = fo > 0 and b > end - fo
    if not (need_in or need_out):
        return None
    tt = a + np.arange(length, dtype=np.float64) / sr
    g = np.ones(length, np.float64)
    if need_in:
        np.minimum(g, (tt - clip.start) / fi, out=g)
    if need_out:
        np.minimum(g, (end - tt) / fo, out=g)
    np.clip(g, 0.0, 1.0, out=g)
    return g.astype(np.float32)


def mix(project: Project, cache: AudioCache, t0: float, n: int, wait: bool = False) -> np.ndarray:
    """wait=True (export) blocks until processed clip audio (audio_fx / speed) is ready."""
    sr = cache.sr
    out = np.zeros((n, 2), np.float32)
    t1 = t0 + n / sr
    for track in project.tracks:
        if track.muted:
            continue
        for clip in list(track.clips):
            if clip.is_text or not clip.asset_id:
                continue
            if clip.end <= t0 or clip.start >= t1:
                continue
            asset = project.assets.get(clip.asset_id)
            vol = animate.volume_at(clip, max(t0, clip.start))
            if vol <= 0:
                continue
            processed = audio_fx.clip_audio(clip, asset, cache, wait=wait) if audio_fx.active(clip) else None
            if processed is not None:
                # already in timeline time: sample 0 == clip.start, speed + effects applied
                a = max(t0, clip.start)
                b = min(t1, clip.end)
                o0 = int(round((a - t0) * sr))
                i0 = int(round((a - clip.start) * sr))
                length = min(int(round((b - a) * sr)), n - o0, len(processed) - i0)
                if length <= 0 or i0 < 0:
                    continue
                seg = processed[i0:i0 + length]
                gain = _fade_ramp(clip, a, len(seg), sr, *audio_fades(track, clip))
                if gain is None:
                    out[o0:o0 + len(seg)] += seg * vol
                else:
                    out[o0:o0 + len(seg)] += seg * (gain * vol)[:, None]
                continue
            data = cache.get(clip.asset_id)
            if data is None or len(data) == 0:
                continue
            a = max(t0, clip.start)
            b = min(t1, clip.end)
            o0 = int(round((a - t0) * sr))
            length = min(int(round((b - a) * sr)), n - o0)
            if length <= 0:
                continue
            speed = clip.speed if clip.speed > 0 else 1.0
            s0 = clip.source_time(a) * sr
            s_end = min(len(data), int(round(clip.out_point * sr)))
            if abs(speed - 1.0) < 1e-9:
                i0 = int(round(s0))
                lo = max(i0, 0)
                hi = min(i0 + length, s_end)
                if hi <= lo:
                    continue
                seg = data[lo:hi]
                off = o0 + (lo - i0)
            else:
                idx = (s0 + np.arange(length, dtype=np.float64) * speed).astype(np.int64)
                valid = (idx >= 0) & (idx < s_end)
                if not valid.any():
                    continue
                first = int(np.argmax(valid))
                last = length - int(np.argmax(valid[::-1]))
                seg = data[idx[first:last]]
                off = o0 + first
            gain = _fade_ramp(clip, a + (off - o0) / sr, len(seg), sr, *audio_fades(track, clip))
            if gain is None:
                out[off:off + len(seg)] += seg * vol
            else:
                out[off:off + len(seg)] += seg * (gain * vol)[:, None]
    np.clip(out, -1.0, 1.0, out=out)
    return out
