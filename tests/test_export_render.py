import os

import av
import numpy as np
import pytest

from clipeditor.editor import Editor
from clipeditor.encoders import codec_args, listed
from clipeditor.export_render import export_video, needs_render_engine

from .sample_media import SAMPLES

A = os.path.join(SAMPLES, "clip_a.mp4")


def project():
    ed = Editor()
    a = ed.import_file(A)
    ed.add_clip(a.id)
    ed.add_text("Render engine", start=0.5, duration=2.0)
    return ed


def test_graph_path_when_simple(tmp_path):
    ed = project()
    assert needs_render_engine(ed.project) == []
    out = str(tmp_path / "g.mp4")
    assert export_video(ed.project, out, width=320, height=180, preset="ultrafast") == "graph"
    assert os.path.getsize(out) > 0


def test_render_engine_matches_duration_and_rotation(tmp_path):
    ed = project()
    clip = ed.project.video_tracks()[-1].clips[0]
    clip.rotation = 90
    clip.flip_h = True
    clip.blend_mode = "screen"
    assert "rotate/flip/crop" in needs_render_engine(ed.project)
    out = str(tmp_path / "r.mp4")
    seen = []
    assert export_video(ed.project, out, progress=seen.append, width=320, height=180,
                        preset="ultrafast", fps=30) == "render"
    assert seen[-1] == 1.0
    with av.open(out) as c:
        assert abs(c.duration / 1e6 - ed.project.duration) < 0.15
        assert c.streams.audio and c.streams.video
        f = next(c.decode(video=0)).to_ndarray(format="rgb24")
    # rotated 90deg 16:9 into a 16:9 frame -> left/right columns are black, centre is not
    assert f[:, :20].mean() < 10 and f[:, 140:180].mean() > 30


@pytest.mark.parametrize("codec", ["hevc", "av1"])
def test_other_codecs(tmp_path, codec):
    enc = codec_args(codec)[1]
    if enc not in listed():
        pytest.skip(f"{enc} not in this ffmpeg")
    ed = project()
    out = str(tmp_path / f"{codec}.mp4")
    export_video(ed.project, out, width=320, height=180, preset="ultrafast", codec=codec, crf=30)
    with av.open(out) as c:
        assert c.streams.video[0].codec_context.name in ("hevc", "av1", "libdav1d", "libaom-av1")


# ---- parallel render-engine export ----------------------------------------------------------
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

from clipeditor import export_render as er  # noqa: E402

B = os.path.join(SAMPLES, "clip_b.mp4")  # 6 s, 320x180 @ 30 fps


def six_second_project():
    ed = Editor()
    b = ed.import_file(B)
    ed.add_clip(b.id)
    ed.add_text("Parallel", start=0.5, duration=2.0)
    ed.project.video_tracks()[-1].clips[0].rotation = 15  # forces the render engine
    assert "rotate/flip/crop" in needs_render_engine(ed.project)
    assert abs(ed.project.duration - 6.0) < 1e-6
    return ed


@pytest.fixture
def parallel(monkeypatch):
    monkeypatch.delenv("CLIPEDITOR_EXPORT_WORKERS", raising=False)
    if er.parallel_workers(180) < 2:
        pytest.skip("needs at least 3 CPU cores")


def spy(monkeypatch, name):
    calls = []
    real = getattr(er, name)

    def wrapper(*a, **k):
        calls.append(name)
        return real(*a, **k)

    monkeypatch.setattr(er, name, wrapper)
    return calls


def frames(path):
    with av.open(path) as c:
        s = c.streams.video[0]
        out = {round(float(f.pts * s.time_base) * 30): f.to_ndarray(format="rgb24").astype(np.int16)
               for f in c.decode(s)}
        return out, c.duration / 1e6, c.streams.audio


def test_parallel_matches_single_process(tmp_path, monkeypatch, parallel):
    ed = six_second_project()
    single_calls = spy(monkeypatch, "_export_single")
    par, seen = str(tmp_path / "par.mp4"), []
    assert export_video(ed.project, par, progress=seen.append, width=320, height=180, fps=30,
                        preset="ultrafast") == "render"
    assert single_calls == []  # really went through the worker pool
    assert seen[0] == 0.0 and seen[-1] == 1.0
    assert all(b >= a for a, b in zip(seen, seen[1:])), "progress must never go backwards"
    assert len(seen) > 10

    monkeypatch.setenv("CLIPEDITOR_EXPORT_WORKERS", "1")
    one = str(tmp_path / "one.mp4")
    export_video(ed.project, one, width=320, height=180, fps=30, preset="ultrafast")
    assert single_calls == ["_export_single"]

    pf, pdur, paudio = frames(par)
    sf, sdur, _ = frames(one)
    assert sorted(pf) == sorted(sf) == list(range(180))  # every frame, evenly timed, no gaps/dupes
    assert abs(pdur - 6.0) < 0.05 and abs(pdur - sdur) < 0.03
    assert paudio
    for i in (0, 29, 30, 31, 89, 90, 150, 179):  # includes segment boundaries (6 chunks of 30)
        assert np.abs(pf[i] - sf[i]).mean() < 2, f"frame {i} differs"
    with av.open(par) as c:  # the joined stream plays straight through
        for pkt in c.demux(video=0):
            if pkt.pts is not None and round(float(pkt.pts * pkt.time_base) * 30) % 30 == 0:
                assert pkt.is_keyframe  # every segment starts on a keyframe


def test_workers_env_var_forces_single_process(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIPEDITOR_EXPORT_WORKERS", "1")
    assert er.parallel_workers(10_000) == 1
    pool_calls = spy(monkeypatch, "_export_parallel")
    out = str(tmp_path / "o.mp4")
    export_video(six_second_project().project, out, width=320, height=180, fps=30, preset="ultrafast")
    assert pool_calls == [] and os.path.getsize(out) > 0


def test_worker_count_rules(monkeypatch):
    monkeypatch.delenv("CLIPEDITOR_EXPORT_WORKERS", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    assert er.parallel_workers(59) == 1
    assert er.parallel_workers(90) == 3
    assert er.parallel_workers(100_000) == 8
    monkeypatch.setenv("CLIPEDITOR_EXPORT_WORKERS", "2")
    assert er.parallel_workers(100_000) == 2
    chunks = er._chunks(181, 6)
    assert chunks[0][0] == 0 and chunks[-1][1] == 181 and len(chunks) == 6
    assert all(a[1] == b[0] for a, b in zip(chunks, chunks[1:]))  # contiguous, no gaps or overlaps
    assert {b - a for a, b in chunks} <= {30, 31}


def test_parallel_cancel_stops_workers_and_cleans_up(tmp_path, monkeypatch, parallel):
    temp = tmp_path / "temp"
    temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    outdir = tmp_path / "out"
    outdir.mkdir()
    flag = threading.Event()
    when = []

    def progress(v):
        if v > 0.3 and not flag.is_set():
            when.append(time.monotonic())
            flag.set()

    with pytest.raises(RuntimeError, match="cancelled"):
        export_video(six_second_project().project, str(outdir / "c.mp4"), progress=progress,
                     cancelled=flag.is_set, width=320, height=180, fps=30, preset="ultrafast")
    assert when and time.monotonic() - when[0] < 10  # workers stop promptly
    assert list(outdir.iterdir()) == []  # no partial output
    assert list(temp.iterdir()) == []  # no segments or audio left behind


def test_gpu_mode_renders_in_parallel_with_one_encoder(tmp_path, monkeypatch, parallel):
    import clipeditor.encoders as enc

    monkeypatch.setattr(enc, "gpu_encoder", lambda codec: None)  # encode with libx264 ...
    monkeypatch.setattr(er, "gpu_encoder", lambda codec: ("h264_test", "Test"))  # ... in stream mode
    single_calls = spy(monkeypatch, "_export_single")
    out, seen = str(tmp_path / "gpu.mp4"), []
    export_video(six_second_project().project, out, progress=seen.append, gpu=True, width=320, height=180,
                 fps=30, preset="ultrafast")
    assert single_calls == []
    assert seen[-1] == 1.0 and all(b >= a for a, b in zip(seen, seen[1:]))
    got, dur, _ = frames(out)
    assert sorted(got) == list(range(180)) and abs(dur - 6.0) < 0.05


def test_pool_start_failure_falls_back(tmp_path, monkeypatch, parallel):
    import multiprocessing.context

    def broken(self):
        raise OSError("no processes today")

    monkeypatch.setattr(multiprocessing.context.SpawnProcess, "start", broken)
    single_calls = spy(monkeypatch, "_export_single")
    out = str(tmp_path / "f.mp4")
    export_video(six_second_project().project, out, width=320, height=180, fps=30, preset="ultrafast")
    assert single_calls == ["_export_single"]
    got, _, _ = frames(out)
    assert len(got) == 180


def test_worker_module_does_not_import_ui():
    code = ("import sys, clipeditor.render_worker, clipeditor.render, clipeditor.export, clipeditor.model; "
            "sys.exit(any(m.startswith('clipeditor.ui') for m in sys.modules))")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert subprocess.run([sys.executable, "-c", code], cwd=root).returncode == 0
