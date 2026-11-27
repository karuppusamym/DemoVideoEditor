"""Find silent stretches of the timeline (for one-click "remove silences")."""
from __future__ import annotations

import numpy as np

from .audio_mix import mix
from .media import AudioCache
from .model import Project


def find_silences(project: Project, cache: AudioCache, threshold_db: float = -40.0,
                  min_silence: float = 0.6, keep_pad: float = 0.15, window: float = 0.02) -> list[tuple[float, float]]:
    """Ranges (timeline seconds) quieter than threshold_db for at least min_silence.

    keep_pad seconds of each silence are kept next to the sound so speech isn't clipped."""
    dur = project.duration
    if dur <= 0:
        return []
    sr = cache.sr
    hop = max(int(sr * window), 1)
    levels = []
    t = 0.0
    chunk = hop * 500
    while t < dur:
        n = min(chunk, int(np.ceil((dur - t) * sr)))
        block = mix(project, cache, t, n)
        mono = block.mean(axis=1)
        m = len(mono) // hop
        if m:
            rms = np.sqrt((mono[: m * hop].reshape(m, hop) ** 2).mean(axis=1))
            levels.append(rms)
        t += n / sr
    if not levels:
        return []
    rms = np.concatenate(levels)
    quiet = 20 * np.log10(np.maximum(rms, 1e-9)) < threshold_db
    out = []
    i = 0
    n = len(quiet)
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            a, b = i * window, min(j * window, dur)
            a2 = a + (keep_pad if a > 0 else 0.0)
            b2 = b - (keep_pad if b < dur - 1e-6 else 0.0)
            if b - a >= min_silence and b2 - a2 > 0.05:
                out.append((round(a2, 3), round(b2, 3)))
            i = j
        else:
            i += 1
    return out
