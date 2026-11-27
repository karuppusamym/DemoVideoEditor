"""AI / FFmpeg clip tools: background removal, stabilize, reverse, enhance, slow-mo, reframe.

Background-removal tests skip when rembg is missing or the model can't be downloaded.
Speeds (seconds of processing per output second) are printed; run with `-s` to see them.
"""
import os
import time

import av
import numpy as np
import pytest

from clipeditor.ai import jobs
from clipeditor.ai import video_tools as VT
from clipeditor.media import VideoReader, probe

from .sample_media import SAMPLES

CLIP = os.path.join(SAMPLES, "clip_a.mp4")
TONE = os.path.join(SAMPLES, "tone.wav")
SPEEDS: dict[str, float] = {}


@pytest.fixture(autouse=True)
def _gen_dir(tmp_path, monkeypatch):
    """Keep generated files out of the real %APPDATA%."""
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))


def _speed(name, secs, out_dur):
    SPEEDS[name] = secs / max(out_dur, 1e-6)
    print(f"\n[speed] {name}: {SPEEDS[name]:.2f} s per output second "
          f"({secs:.1f} s for {out_dur:.2f} s)")


def _frame(path, t, w=160, h=90):
    r = VideoReader(path)
    try:
        return r.rgb_at(t, w, h).astype(np.float32)
    finally:
        r.close()


def _synthetic_person(path, seconds=2.0, fps=25, w=320, h=240, moving=True):
    """Busy checker/noise background with a skin/shirt 'person' ellipse (+ speech audio)."""
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:h, 0:w]
    bg = (((xx // 16 + yy // 16) % 2) * 90 + 60).astype(np.uint8)
    bg = np.dstack([bg, bg // 2 + 40, 255 - bg]).astype(np.int16)
    with av.open(path, "w") as c:
        vs = c.add_stream("libx264", rate=fps)
        vs.width, vs.height, vs.pix_fmt = w, h, "yuv420p"
        n = int(seconds * fps)
        for i in range(n):
            cx = w * (0.25 + 0.5 * i / max(n - 1, 1)) if moving else w / 2
            img = np.clip(bg + rng.integers(-25, 25, bg.shape), 0, 255).astype(np.uint8)
            body = ((xx - cx) / (w * 0.13)) ** 2 + ((yy - h * 0.72) / (h * 0.32)) ** 2 < 1
            head = ((xx - cx) / (w * 0.07)) ** 2 + ((yy - h * 0.3) / (h * 0.11)) ** 2 < 1
            img[body] = (200, 40, 40)
            img[head] = (225, 180, 150)
            for p in vs.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(p)
        for p in vs.encode():
            c.mux(p)
    return path


# ---- jobs ------------------------------------------------------------------
def test_job_framework(tmp_path):
    d = jobs.output_dir("test_kind")
    assert d.startswith(os.environ["APPDATA"]) and os.path.isdir(d)
    assert d.replace("\\", "/").endswith("ClipEditor/generated/test_kind")

    def fn(progress, cancelled, status):
        for i in range(5):
            progress(i / 5)
            status(f"step {i}")
        return "ok.mp4"

    j = jobs.Job("demo", fn).start()
    assert j.wait(5) and j.ok and j.result == "ok.mp4" and j.progress == 1.0

    def slow(progress, cancelled, status):
        while True:
            jobs.check_cancel(cancelled)
            time.sleep(0.01)

    j = jobs.Job("slow", slow).start()
    j.cancel()
    assert j.wait(5) and j.error == "Cancelled" and j.result is None

    j = jobs.Job("bad", lambda **kw: 1 / 0).start()
    assert j.wait(5) and "ZeroDivisionError" in j.error


def test_ffmpeg_run_cancel_kills():
    t = time.time()
    with pytest.raises(jobs.Cancelled):
        jobs.ffmpeg_run(["-f", "lavfi", "-i", "testsrc2=s=640x360:r=30", "-t", "600",
                         "-f", "null", "-"], 600, None, lambda: time.time() - t > 0.5)
    assert time.time() - t < 10


def test_ffmpeg_progress():
    seen = []
    jobs.ffmpeg_run(["-f", "lavfi", "-i", "testsrc2=s=320x180:r=30", "-t", "2",
                     "-f", "null", "-"], 2.0, seen.append, None)
    assert seen and seen[-1] == 1.0 and all(0 <= p <= 1 for p in seen)


# ---- video tools -----------------------------------------------------------
@pytest.mark.parametrize("name,params", [
    ("stabilize", {"strength": 0.6}),
    ("reverse", {}),
    ("enhance", {}),
    ("enhance", {"upscale": 2}),
    ("smooth_slowmo", {"factor": 2}),
    ("auto_reframe", {"target_aspect": "9:16"}),
    ("auto_reframe", {"target_aspect": "1:1"}),
])
def test_video_tool(name, params):
    src = probe(CLIP)
    prog = []
    t = time.time()
    out = VT.run_tool(name, CLIP, 2.0, 4.0, progress=prog.append, **params)
    secs = time.time() - t
    a = probe(out)
    want = 2.0 * (params.get("factor", 1) if name == "smooth_slowmo" else 1)
    assert a.has_video and a.has_audio
    assert abs(a.duration - want) < 0.15, (a.duration, want)
    assert prog and prog[-1] == pytest.approx(1.0)
    if name == "auto_reframe":
        ar = 9 / 16 if params["target_aspect"] == "9:16" else 1.0
        assert a.height == src.height // 2 * 2 and abs(a.width / a.height - ar) < 0.02
    elif params.get("upscale") == 2:
        assert (a.width, a.height) == (src.width * 2, src.height * 2)
    else:
        assert (a.width, a.height) == (src.width, src.height)
    # decodable to the end
    r = VideoReader(out)
    try:
        assert r.rgb_at(want - 0.1, 64, 36) is not None
    finally:
        r.close()
    _speed(f"{name}{params or ''}", secs, a.duration)


def test_reverse_really_reverses():
    out = VT.reverse(CLIP, 2.0, 4.0)
    fd = 1.0 / probe(CLIP).fps
    first_out = _frame(out, 0.0)
    last_in = _frame(CLIP, 4.0 - fd)
    first_in = _frame(CLIP, 2.0)
    d_match = np.abs(first_out - last_in).mean()
    d_other = np.abs(first_out - first_in).mean()
    assert d_match < 6.0, d_match
    assert d_match < d_other or d_other < 1.0  # (static clips can't tell ends apart)
    # last frame of the output is the first frame of the range
    assert np.abs(_frame(out, 2.0 - fd) - first_in).mean() < 6.0


def test_reverse_chunks_join_in_order(tmp_path):
    """Chunked path: 3 s in 1 s chunks must equal a single-chunk reverse."""
    out = VT.reverse(CLIP, 1.0, 4.0, chunk=1.0)
    assert abs(probe(out).duration - 3.0) < 0.15
    fd = 1.0 / probe(CLIP).fps
    for t_out in (0.0, 1.0, 2.0, 2.9):
        t_in = 4.0 - fd - t_out
        assert np.abs(_frame(out, t_out) - _frame(CLIP, t_in)).mean() < 8.0, t_out


def test_auto_reframe_follows_subject(tmp_path):
    src = _synthetic_person(str(tmp_path / "walk.mp4"), seconds=4.0, moving=True)
    track = VT.saliency_track(src, 0.0, 4.0)
    xs = [cx for _, cx, _ in track]
    assert xs[-1] - xs[0] > 0.2, xs  # subject walks left -> right
    out = VT.auto_reframe(src, 0.0, 4.0, target_aspect="9:16")
    a = probe(out)
    assert abs(a.width / a.height - 9 / 16) < 0.03


def test_tools_registry():
    for name, t in VT.TOOLS.items():
        assert {"label", "description", "params"} <= set(t)
        assert callable(t["fn"])


def test_tool_cancel():
    t = time.time()
    with pytest.raises(jobs.Cancelled):
        VT.smooth_slowmo(CLIP, 0.0, 8.0, cancelled=lambda: time.time() - t > 0.5)


# ---- background removal ----------------------------------------------------
def _bg_session(model="u2netp"):
    from clipeditor.ai import bgremove as B
    if not B.is_available():
        pytest.skip("rembg/onnxruntime not installed")
    try:
        B.get_session(model)
    except Exception as e:  # offline / download failed
        pytest.skip(f"model {model} unavailable: {e}")
    return B


@pytest.mark.model
def test_bgremove_video_alpha(tmp_path):
    B = _bg_session()
    src = _synthetic_person(str(tmp_path / "moving.mp4"))
    t = time.time()
    out = B.remove_background(src, 0.0, 2.0, "u2netp")
    secs = time.time() - t
    a = probe(out)
    assert a.alpha and a.kind == "video"
    assert abs(a.duration - 2.0) < 0.15
    r = VideoReader(out)
    try:
        rgba = r.rgba_at(1.0, a.width, a.height)
        assert rgba.shape == (a.height, a.width, 4) and rgba.dtype == np.uint8
        assert rgba.flags["C_CONTIGUOUS"]
        al = rgba[..., 3]
        assert al.std() > 10 and al.min() < 30 and al.max() > 220
    finally:
        r.close()
    _speed("bgremove u2netp", secs, a.duration)


@pytest.mark.model
def test_bgremove_synthetic_person(tmp_path):
    B = _bg_session()
    src = _synthetic_person(str(tmp_path / "person.mp4"), seconds=1.0, moving=False)
    # Mux a tone so the audio-copy path is exercised.
    withaud = str(tmp_path / "person_a.mp4")
    jobs.ffmpeg_run(["-i", src, "-i", TONE, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                     "-c:a", "aac", "-shortest", withaud])
    out = B.remove_background(withaud, 0.0, 1.0, "u2netp")
    a = probe(out)
    assert a.alpha and a.has_audio
    r = VideoReader(out)
    try:
        al = r.rgba_at(0.5, a.width, a.height)[..., 3].astype(np.float32)
    finally:
        r.close()
    h, w = al.shape
    inside = al[int(h * 0.6):int(h * 0.85), int(w * 0.45):int(w * 0.55)].mean()
    corner = al[: h // 6, : w // 6].mean()
    print(f"\n[bg] synthetic person alpha inside={inside:.0f} corner={corner:.0f}")
    assert inside > corner + 60


@pytest.mark.model
def test_bgremove_image(tmp_path):
    B = _bg_session()
    from PIL import Image
    src = _synthetic_person(str(tmp_path / "p.mp4"), seconds=0.2, moving=False)
    png = str(tmp_path / "still.png")
    r = VideoReader(src)
    Image.fromarray(r.rgb_at(0.0, 320, 240)).save(png)
    r.close()
    out = B.remove_background(png, 0, 0, "u2netp")
    assert out.endswith(".png")
    a = probe(out)
    assert a.kind == "image" and a.alpha
    im = np.asarray(Image.open(out))
    assert im.shape == (240, 320, 4) and im[..., 3].std() > 10
