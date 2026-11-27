"""Voiceover recording from a microphone (sounddevice.InputStream).

    input_devices() -> [{"index", "name", "channels", "default_samplerate", "default"}]
    r = Recorder(samplerate=48000, device=None)
    r.start()            # RuntimeError with a friendly message if no microphone / device error
    r.level()            # 0..1 peak of the latest block, for a live meter (any thread)
    r.elapsed()          # seconds recorded so far
    path = r.stop()      # 16-bit WAV in %APPDATA%/ClipEditor/voiceovers

The audio callback only appends to a list; start/stop/level are safe from the GUI thread.
"""
from __future__ import annotations

import os
import threading
import time
import wave

import numpy as np

from .audio_fx import app_dir


def _sd():
    try:
        import sounddevice as sd
        return sd
    except Exception as e:  # PortAudio missing etc.
        raise RuntimeError(f"Audio input is not available on this system ({e}).") from e


def input_devices() -> list[dict]:
    """Input-capable devices; [] if none or if the audio system is unavailable."""
    try:
        sd = _sd()
        devs = sd.query_devices()
        try:
            default_in = sd.default.device[0]
        except Exception:
            default_in = -1
    except Exception:
        return []
    out = []
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) > 0:
            out.append({"index": i, "name": str(d.get("name", f"Device {i}")).replace("\n", " ").strip(),
                        "channels": int(d["max_input_channels"]),
                        "default_samplerate": float(d.get("default_samplerate", 48000)),
                        "default": i == default_in})
    return out


class Recorder:
    def __init__(self, samplerate: int = 48000, device: int | str | None = None, channels: int | None = None):
        self.samplerate = int(samplerate)
        self.device = device
        self.channels = channels
        self._blocks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._level = 0.0
        self._stream = None
        self._frames = 0
        self._t0 = 0.0
        self.status_messages: list[str] = []

    @property
    def recording(self) -> bool:
        return self._stream is not None

    def _callback(self, indata, frames, time_info, status):
        if status:
            self.status_messages.append(str(status))
        block = indata.copy()
        with self._lock:
            self._blocks.append(block)
            self._frames += frames
        self._level = float(min(1.0, np.abs(block).max())) if block.size else 0.0

    def start(self) -> None:
        if self._stream is not None:
            return
        sd = _sd()
        try:
            info = sd.query_devices(self.device, kind="input")
        except Exception as e:
            raise RuntimeError("No microphone found. Connect a microphone or check Windows "
                               "privacy settings (Microphone access) and try again.") from e
        if int(info.get("max_input_channels", 0)) < 1:
            raise RuntimeError("The selected audio device has no input channels.")
        ch = self.channels or min(2, int(info["max_input_channels"]))
        with self._lock:
            self._blocks = []
            self._frames = 0
        self._level = 0.0
        try:
            stream = sd.InputStream(samplerate=self.samplerate, device=self.device, channels=ch,
                                    dtype="float32", callback=self._callback)
            stream.start()
        except Exception as e:
            raise RuntimeError(f"Could not open the microphone: {e}") from e
        self.channels = ch
        self._stream = stream
        self._t0 = time.monotonic()

    def level(self) -> float:
        return self._level if self._stream is not None else 0.0

    def elapsed(self) -> float:
        return self._frames / self.samplerate

    def cancel(self) -> None:
        self._close()
        with self._lock:
            self._blocks = []

    def _close(self):
        s, self._stream = self._stream, None
        if s is not None:
            try:
                s.stop()
            finally:
                s.close()
        self._level = 0.0

    def stop(self, out_path: str | None = None) -> str:
        """Stop and write a WAV; returns its path. RuntimeError if nothing was recorded."""
        if self._stream is None and not self._blocks:
            raise RuntimeError("Not recording.")
        self._close()
        with self._lock:
            blocks, self._blocks = self._blocks, []
        if not blocks:
            raise RuntimeError("Nothing was recorded.")
        data = np.concatenate(blocks)
        if out_path is None:
            out_path = os.path.join(app_dir("voiceovers"), time.strftime("voiceover_%Y%m%d_%H%M%S.wav"))
            base, n = out_path[:-4], 1
            while os.path.exists(out_path):
                out_path = f"{base}_{n}.wav"
                n += 1
        write_wav(out_path, data, self.samplerate)
        return out_path


def write_wav(path: str, data: np.ndarray, sr: int) -> None:
    data = np.asarray(data, np.float32)
    if data.ndim == 1:
        data = data[:, None]
    pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())
