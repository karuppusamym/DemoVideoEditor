"""Convert between source seconds and timeline seconds for a speed ramp."""
from __future__ import annotations

from bisect import bisect_right
from functools import lru_cache
from math import expm1, isfinite, log1p

MIN_RATE = 0.1
MAX_RATE = 8.0


def validate_points(points) -> tuple[tuple[float, float], ...]:
    points = tuple((float(t), float(rate)) for t, rate in points)
    if (len(points) == 1
            or any(not isfinite(t) or not MIN_RATE <= rate <= MAX_RATE for t, rate in points)
            or any(second[0] <= first[0] for first, second in zip(points, points[1:]))):
        raise ValueError("Speed ramps need increasing source positions and speeds from 0.1x to 8x.")
    return points


class Ramp:
    """Speeds interpolate linearly across source time and hold beyond the endpoints."""

    def __init__(self, points: tuple[tuple[float, float], ...]):
        self.positions, self.rates = zip(*points)
        slopes = [(rb - ra) / (b - a) for (a, ra), (b, rb) in zip(points, points[1:])]
        self.slopes = (*slopes, 0.0)
        times = [0.0]
        for i, slope in enumerate(slopes):
            distance = self.positions[i + 1] - self.positions[i]
            times.append(times[-1] + self._elapsed(distance, self.rates[i], slope))
        self.times = tuple(times)

    @staticmethod
    def _elapsed(distance: float, rate: float, slope: float) -> float:
        # Integrating 1 / playback_rate gives the time to traverse a source segment.
        return log1p(slope * distance / rate) / slope if slope else distance / rate

    def timeline_at(self, source: float) -> float:
        i = bisect_right(self.positions, source) - 1
        if i < 0:
            return (source - self.positions[0]) / self.rates[0]
        return self.times[i] + self._elapsed(source - self.positions[i], self.rates[i], self.slopes[i])

    def source_at(self, timeline: float) -> float:
        i = bisect_right(self.times, timeline) - 1
        if i < 0:
            return self.positions[0] + timeline * self.rates[0]
        elapsed = timeline - self.times[i]
        slope = self.slopes[i]
        distance = expm1(slope * elapsed) / slope if slope else elapsed
        return self.positions[i] + self.rates[i] * distance

    def rate_at(self, source: float) -> float:
        i = bisect_right(self.positions, source) - 1
        if i < 0:
            return self.rates[0]
        return self.rates[i] + self.slopes[i] * (source - self.positions[i])

    def source_samples(self, timeline):
        """Map a NumPy array of timeline offsets for audio mixing."""
        import numpy as np

        indices = np.searchsorted(self.times, timeline, side="right") - 1
        segments = np.maximum(indices, 0)
        elapsed = timeline - np.take(self.times, segments)
        slope = np.take(self.slopes, segments)
        slope[indices < 0] = 0.0
        curved = slope != 0.0
        distance = elapsed.copy()
        distance[curved] = np.expm1(slope[curved] * elapsed[curved]) / slope[curved]
        return np.take(self.positions, segments) + np.take(self.rates, segments) * distance


curve = lru_cache(maxsize=128)(Ramp)


def ramp_audio(ramp, data, in_point, out_point, elapsed, count, sr, source_origin=0.0):
    """Read speed-ramped stereo samples in timeline order, including any silent tail."""
    import numpy as np

    out = np.zeros((count, 2), np.float32)
    if not len(data):
        return out
    offsets = ramp.timeline_at(in_point) + elapsed + np.arange(count, dtype=np.float64) / sr
    source = ramp.source_samples(offsets)
    indices = (source - source_origin) * sr
    valid = (indices >= -1e-6) & (indices < len(data)) & (source < out_point)
    indices = np.maximum(indices[valid], 0.0)
    first = indices.astype(np.int64)
    second = np.minimum(first + 1, len(data) - 1)
    fraction = (indices - first).astype(np.float32)[:, None]
    out[valid] = data[first] + (data[second] - data[first]) * fraction
    return out
