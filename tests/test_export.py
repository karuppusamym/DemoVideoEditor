import os

from clipeditor.export import _atempo_chain, _fit_size, build_command, export
from clipeditor.media import probe
from clipeditor.model import Clip, Project, new_id

from .sample_media import SAMPLES



def _probe_streams(path):
    """(duration in s, {stream types}) of a media file."""
    import av

    with av.open(path) as c:
        kinds = {s.type for s in c.streams}
        dur = float(c.duration) / 1e6 if c.duration else 0.0
    return dur, kinds


def make_project() -> Project:
    p = Project.new()
    p.width, p.height = 640, 360
    a = probe(os.path.join(SAMPLES, "clip_a.mp4"))
    b = probe(os.path.join(SAMPLES, "clip_b.mp4"))
    t = probe(os.path.join(SAMPLES, "tone.wav"))
    for asset in (a, b, t):
        p.assets[asset.id] = asset
    v2, v1, a1, _ = p.tracks
    # V1: clip_a at 2x (8s source -> 4s) with fade-in, then clip_b with eq and fade-out
    v1.clips.append(Clip(new_id(), a.id, 0.0, 0.0, 8.0, speed=2.0, fade_in=0.5))
    v1.clips.append(Clip(new_id(), b.id, 4.0, 0.0, 6.0, fade_out=1.0,
                         brightness=0.1, contrast=1.3, saturation=0.2))
    # V2: scaled/offset picture-in-picture with opacity + fades, and a text title
    v2.clips.append(Clip(new_id(), a.id, 1.0, 2.0, 6.0, opacity=0.8, scale=0.4,
                         pos_x=0.25, pos_y=-0.25, fade_in=0.5, fade_out=0.5, volume=0.5))
    v2.clips.append(Clip(new_id(), "", 6.0, 0.0, 3.0, text="Hello\nvedit", font_size=60,
                         bg_color="#00000080", fade_in=0.5, pos_y=0.3))
    # A1: tone slowed to 0.25x (1s source -> 4s) with fades
    a1.clips.append(Clip(new_id(), t.id, 2.0, 0.0, 1.0, speed=0.25, fade_in=0.3, fade_out=0.3))
    return p


def test_helpers():
    def product(chain):
        vals = [float(s.split("=")[1]) for s in chain]
        assert all(0.5 <= v <= 2.0 for v in vals)
        r = 1.0
        for v in vals:
            r *= v
        return r

    assert _atempo_chain(1.0) == []
    for s in (0.25, 0.3, 0.5, 0.75, 1.5, 2.0, 3.0, 4.0):
        assert abs(product(_atempo_chain(s)) - s) < 1e-4
    assert len(_atempo_chain(4.0)) == 2 and len(_atempo_chain(0.25)) == 2
    assert _fit_size(1280, 720, 640, 360, 1.0) == (640, 360)
    w, h = _fit_size(1280, 720, 640, 360, 0.4)
    assert (w, h) == (256, 144) and w % 2 == 0 and h % 2 == 0
    assert _fit_size(720, 1280, 640, 360, 1.0) == (202, 360)


def test_build_command_graph(tmp_path):
    p = make_project()
    cmd, graph = build_command(p, str(tmp_path / "x.mp4"), workdir=str(tmp_path))
    assert "setpts=(PTS-STARTPTS)/2.000000" in graph
    assert "fade=t=in:st=0:d=0.5000:alpha=1" in graph
    assert "eq=brightness=0.1000:contrast=1.3000:saturation=0.2000" in graph
    assert "atempo=0.5,atempo=0.5" in graph
    assert "afade=t=out" in graph
    assert "scale=256:144" in graph
    assert "pad=" not in graph
    pngs = [f for f in os.listdir(tmp_path) if f.endswith(".png")]
    assert len(pngs) == 1


def test_export_runs(tmp_path):
    p = make_project()
    out = str(tmp_path / "out.mp4")
    seen = []
    export(p, out, progress=seen.append, preset="ultrafast")
    assert os.path.getsize(out) > 0
    dur, kinds = _probe_streams(out)
    assert {"video", "audio"} <= kinds
    assert abs(dur - p.duration) < 0.25, (dur, p.duration)
    assert seen and seen[-1] == 1.0


# ---------------------------------------------------------------- transitions
def make_transition_editor(kind: str, d: float = 1.0):
    """clip_a (8 s) then clip_b (6 s) on V1 with a `kind` transition of `d` s, plus a text clip on
    V2 away from the overlap. Returns (editor, clip_a, clip_b)."""
    from clipeditor.editor import Editor

    p = Project.new()
    p.width, p.height = 640, 360
    a = probe(os.path.join(SAMPLES, "clip_a.mp4"))
    b = probe(os.path.join(SAMPLES, "clip_b.mp4"))
    p.assets[a.id] = a
    p.assets[b.id] = b
    v2, v1, _, _ = p.tracks
    ca = Clip(new_id(), a.id, 0.0, 0.0, 8.0)
    cb = Clip(new_id(), b.id, 8.0, 0.0, 6.0)
    v1.clips += [cb, ca]  # deliberately unsorted: export must draw in start-time order
    v2.clips.append(Clip(new_id(), "", 1.0, 0.0, 3.0, text="Title", font_size=48, fade_in=0.3))
    ed = Editor(p)
    assert p.duration == 14.0
    assert ed.set_transition(cb.id, kind, d)
    assert abs(cb.start - (8.0 - d)) < 1e-9 and cb.transition == d and cb.transition_kind == kind
    assert abs(p.duration - (14.0 - d)) < 1e-9
    return ed, ca, cb


def _frame_at(path, t):
    """RGB float array of the first frame at or after t seconds."""
    import av
    import numpy as np

    with av.open(path) as c:
        s = c.streams.video[0]
        c.seek(int(max(t - 1.0, 0) / s.time_base), stream=s)
        for f in c.decode(s):
            if f.time is not None and f.time >= t - 1e-3:
                return f.to_ndarray(format="rgb24").astype(np.float32)
    raise AssertionError(f"no frame at {t} in {path}")


def _src_frame(path, t, w=640, h=360):
    import numpy as np
    import av

    fr = _frame_at(path, t).astype(np.uint8)
    img = av.VideoFrame.from_ndarray(fr, format="rgb24").reformat(width=w, height=h)
    return img.to_ndarray(format="rgb24").astype(np.float32)


def _mad(x, y):
    return float(abs(x - y).mean())


def test_transition_graph_every_kind(tmp_path):
    from clipeditor import transitions

    for kind in transitions.KINDS:
        ed, ca, cb = make_transition_editor(kind)
        tr = ed.project.video_tracks()[1]
        cmd, graph = build_command(ed.project, str(tmp_path / f"{kind}.mp4"), workdir=str(tmp_path))
        assert "-t" in cmd and cmd[cmd.index("-t", len(cmd) - 10) + 1] == "13.0000"
        lines = graph.split(";\n")
        ov = [ln for ln in lines if "overlay=" in ln]
        # base track V1 first, a before b (b on top), then the text clip
        assert len(ov) == 3
        assert "between(t,0.0000,8.0000)" in ov[0]   # outgoing clip keeps its own window
        assert "between(t,7.0000,13.0000)" in ov[1]
        assert "between(t,1.0000,4.0000)" in ov[2]
        assert all("eval=frame" in ln for ln in ov)
        for f in transitions.ffmpeg_alpha_fades(tr, cb) + transitions.ffmpeg_alpha_fades(tr, ca):
            assert f in graph
        x, y = transitions.ffmpeg_overlay_xy(tr, cb, 0.0, 0.0, 640, 360)
        assert f"x='{x}':y='{y}'" in ov[1]
        if kind.startswith("slide"):
            assert "clip((t-7.0000)" in ov[1]
        # audio crossfade over the overlap
        assert "afade=t=out:st=7.0000:d=1.0000" in graph
        assert "afade=t=in:st=0:d=1.0000" in graph
        if kind == "fadeblack":
            assert "fade=t=out:st=7.0000:d=0.5000:alpha=1" in graph
            assert "fade=t=in:st=0.5000:d=0.5000:alpha=1" in graph
        elif kind == "dissolve":
            assert "fade=t=in:st=0:d=1.0000:alpha=1" in graph


def test_transition_export_frames(tmp_path):
    a_path = os.path.join(SAMPLES, "clip_a.mp4")
    b_path = os.path.join(SAMPLES, "clip_b.mp4")
    A = _src_frame(a_path, 7.5)  # clip A at the overlap midpoint (timeline 7.5)
    B = _src_frame(b_path, 0.5)  # clip B 0.5 s in
    assert _mad(A, B) > 10, "sample clips too similar to tell apart"

    results = {}
    for kind in ("dissolve", "slideleft", "fadeblack"):
        ed, _, _ = make_transition_editor(kind)
        out = str(tmp_path / f"{kind}.mp4")
        export(ed.project, out, preset="ultrafast", width=640, height=360)
        dur, kinds = _probe_streams(out)
        assert {"video", "audio"} <= kinds
        assert abs(dur - ed.project.duration) < 0.25, (kind, dur, ed.project.duration)
        results[kind] = _frame_at(out, 7.5)

    # dissolve: about a 50/50 blend of A and B
    mid = results["dissolve"]
    blend = (A + B) / 2
    assert _mad(mid, blend) < 0.5 * min(_mad(mid, A), _mad(mid, B))

    # slideleft: B (smoothstep(0.5) = 0.5) has slid half-way in from the right
    sl = results["slideleft"]
    assert _mad(sl[:, :300], A[:, :300]) < 12          # left half still A
    assert _mad(sl[:, 340:], B[:, 20:320]) < 8         # right half is B's left half, at x=320
    assert _mad(sl[:, 340:], A[:, 340:]) > 2 * _mad(sl[:, 340:], B[:, 20:320])

    # fadeblack: dark at the midpoint
    assert results["fadeblack"].mean() < 12
