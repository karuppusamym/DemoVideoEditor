import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from clipeditor.export import ffmpeg_exe

from . import sample_media


def pytest_sessionstart(session):
    media = TemporaryDirectory(prefix="clipeditor-tests-")
    session.config.add_cleanup(media.cleanup)
    sample_dir = Path(media.name)
    sample_media.SAMPLES = sample_dir
    ffmpeg = [ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error"]

    for name, seconds, pattern in (
        ("clip_a.mp4", 8, "testsrc2=size=320x180:rate=30"),
        ("clip_b.mp4", 6, "smptebars=size=320x180:rate=30,hue=h=120"),
    ):
        subprocess.run([*ffmpeg, "-f", "lavfi", "-i", pattern,
                        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
                        "-t", str(seconds), "-c:v", "libx264", "-preset", "ultrafast",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", str(sample_dir / name)],
                       check=True)

    for name, signal in (
        ("tone.wav", "0.25*sin(2*PI*220*t)"),
        ("audio_fx.wav", "(0.2+0.1*sin(2*PI*3*t))*(sin(2*PI*190*t)+0.4*sin(2*PI*630*t))"),
    ):
        subprocess.run([*ffmpeg, "-f", "lavfi", "-i", f"aevalsrc={signal}:s=48000:d=10",
                        str(sample_dir / name)], check=True)


def pytest_configure(config):
    config.addinivalue_line("markers", "model: integration test that needs an optional AI model")
