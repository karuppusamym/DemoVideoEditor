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
