"""Video encoder discovery (software + GPU) and FFmpeg argument building."""
from __future__ import annotations

import functools
import os
import subprocess

CODECS = {
    "h264": "H.264 — plays everywhere",
    "hevc": "HEVC / H.265 — half the size",
    "av1": "AV1 — smallest, newest devices",
}
SOFTWARE = {"h264": "libx264", "hevc": "libx265", "av1": "libaom-av1"}
HARDWARE = {
    "h264": [("h264_nvenc", "NVIDIA"), ("h264_qsv", "Intel"), ("h264_amf", "AMD")],
    "hevc": [("hevc_nvenc", "NVIDIA"), ("hevc_qsv", "Intel"), ("hevc_amf", "AMD")],
    "av1": [("av1_nvenc", "NVIDIA"), ("av1_qsv", "Intel"), ("av1_amf", "AMD")],
}
_NOWIN = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def _ffmpeg() -> str:
    from .export import ffmpeg_exe

    return ffmpeg_exe()


@functools.lru_cache(maxsize=1)
def listed() -> frozenset:
    out = subprocess.run([_ffmpeg(), "-hide_banner", "-encoders"], capture_output=True, text=True,
                         creationflags=_NOWIN).stdout
    names = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            names.add(parts[1])
    return frozenset(names)


@functools.lru_cache(maxsize=None)
def works(encoder: str) -> bool:
    """Listed is not enough for GPU encoders (driver, session limits): run a realistic test.

    Pipes 2 s of changing raw BGRA frames in, exactly like the render-engine export does. Some
    Intel QSV drivers pass a tiny lavfi test but fail on piped frames, so a short test is not enough."""
    if encoder not in listed():
        return False
    w, h, n = 320, 180, 60
    extra = {"qsv": ["-global_quality", "22", "-look_ahead", "0", "-bf", "0", "-pix_fmt", "nv12"],
             "nvenc": ["-cq", "22", "-pix_fmt", "yuv420p"],
             "amf": ["-pix_fmt", "yuv420p"]}.get(encoder.rsplit("_", 1)[-1], [])
    try:
        import numpy as np

        frames = bytearray()
        y = np.arange(h, dtype=np.uint16)[:, None]
        x = np.arange(w, dtype=np.uint16)[None, :]
        for i in range(n):
            f = np.empty((h, w, 4), np.uint8)
            f[..., 0] = (x + i * 4) % 256
            f[..., 1] = (y * 2 + i * 3) % 256
            f[..., 2] = ((x + y) // 2 + i) % 256
            f[..., 3] = 255
            frames += f.tobytes()
        r = subprocess.run([_ffmpeg(), "-hide_banner", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgra",
                            "-s", f"{w}x{h}", "-r", "30", "-i", "-", "-c:v", encoder, *extra, "-f", "null", "-"],
                           input=bytes(frames), capture_output=True, timeout=40, creationflags=_NOWIN)
        return r.returncode == 0 and b"Invalid" not in r.stderr and b"failed" not in r.stderr
    except Exception:  # noqa: BLE001
        return False


def gpu_encoder(codec: str) -> tuple[str, str] | None:
    """First working GPU encoder for this codec, as (encoder, vendor)."""
    for enc, vendor in HARDWARE.get(codec, []):
        if works(enc):
            return enc, vendor
    return None


def codec_args(codec: str = "h264", gpu: bool = False, crf: int = 20, preset: str = "medium",
               ten_bit: bool = False) -> list[str]:
    """FFmpeg video-encoder arguments. `crf` is the shared quality knob (lower = better)."""
    hw = gpu_encoder(codec) if gpu else None
    pix = "yuv420p10le" if ten_bit else "yuv420p"
    if hw:
        enc = hw[0]
        if enc.endswith("_nvenc"):
            args = ["-c:v", enc, "-preset", "p5", "-rc", "vbr", "-cq", str(crf), "-b:v", "0"]
            pix = "p010le" if ten_bit else "yuv420p"
        elif enc.endswith("_qsv"):
            # -bf 0: QSV rejects B-frames ("Invalid FrameType:0") with piped raw frames on some Intel drivers
            args = ["-c:v", enc, "-global_quality", str(crf), "-look_ahead", "0", "-bf", "0"]
            pix = "p010le" if ten_bit else "nv12"
        else:  # amf
            args = ["-c:v", enc, "-rc", "cqp", "-qp_i", str(crf), "-qp_p", str(crf)]
            pix = "p010le" if ten_bit else "yuv420p"
    elif codec == "hevc":
        args = ["-c:v", "libx265", "-preset", preset, "-crf", str(crf + 2), "-tag:v", "hvc1",
                "-x265-params", "log-level=error"]
    elif codec == "av1":
        cpu = {"ultrafast": 8, "veryfast": 7, "fast": 6, "medium": 5, "slow": 4}.get(preset, 6)
        args = ["-c:v", "libaom-av1", "-crf", str(crf + 10), "-b:v", "0", "-cpu-used", str(cpu),
                "-row-mt", "1", "-tiles", "2x2"]
    else:
        args = ["-c:v", "libx264", "-preset", preset, "-crf", str(crf)]
        if ten_bit:
            args += ["-profile:v", "high10"]
    return args + ["-pix_fmt", pix]
