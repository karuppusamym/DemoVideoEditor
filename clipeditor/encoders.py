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
    """Listed is not enough for GPU encoders: try encoding a few tiny frames."""
    if encoder not in listed():
        return False
    try:
        r = subprocess.run([_ffmpeg(), "-hide_banner", "-v", "error", "-f", "lavfi", "-i",
                            "color=black:s=256x144:r=30:d=0.2", "-c:v", encoder, "-f", "null", "-"],
                           capture_output=True, timeout=20, creationflags=_NOWIN)
        return r.returncode == 0
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
            args = ["-c:v", enc, "-global_quality", str(crf), "-look_ahead", "0"]
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
