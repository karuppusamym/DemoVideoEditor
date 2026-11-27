import av
import numpy as np
import pytest

from clipeditor import audio_fx
from clipeditor.audio_mix import mix
from clipeditor.editor import Editor
from clipeditor.export_render import export_video, needs_render_engine
from clipeditor.media import AudioCache
from clipeditor.model import Project
from clipeditor.retime import curve

from .sample_media import SAMPLES


def test_ramp_mapping_survives_split_and_project_roundtrip(tmp_path):
    editor = Editor()
    asset = editor.import_file(str(SAMPLES / "clip_a.mp4"))
    clip = editor.add_clip(asset.id)
    points = ((0.0, 0.5), (4.0, 3.0), (8.0, 1.0))
    editor.set_clip_props(clip.id, speed_ramp=points)
    assert clip.source_time(clip.end) == pytest.approx(8.0)
    at = np.linspace(0.0, clip.end, num=40)
    expected = [clip.source_time(t) for t in at]
    assert curve(points).source_samples(at) == pytest.approx(expected)
    assert expected == sorted(expected)

    editor.split(clip.duration / 2)
    left, right = editor.project.video_tracks()[-1].sorted_clips()
    assert left.end == pytest.approx(right.start)
    assert [(left if t < right.start else right).source_time(t) for t in at] == pytest.approx(expected)
    destination = tmp_path / "ramp.clipeditor.json"
    editor.save(destination)
    assert Project.load(destination).to_dict() == editor.project.to_dict()
    editor.undo()
    editor.undo()
    restored = editor.project.find_clip(clip.id)[1]
    assert restored.speed_ramp == () and restored.duration == 8.0


def test_ramped_audio_and_export_follow_timeline(tmp_path):
    editor = Editor()
    asset = editor.import_file(str(SAMPLES / "clip_a.mp4"))
    clip = editor.add_clip(asset.id)
    editor.set_clip_props(clip.id, out_point=2.0, speed_ramp=((0, 0.5), (2, 3.0)))
    sr = 48000
    cache = AudioCache(sr)
    cache.load(asset)
    preview = mix(editor.project, cache, 0.0, 2000)
    assert np.abs(preview).max() > 0.01
    index = clip.source_time(1000 / sr) * sr
    lower = int(index)
    original = cache.get(asset.id)
    assert preview[1000] == pytest.approx(original[lower] + (index - lower) *
                                          (original[lower + 1] - original[lower]), abs=1e-5)
    editor.set_clip_props(clip.id, audio_fx=[{"type": "echo"}])
    with_effect = audio_fx.clip_audio(clip, asset, cache, wait=True)
    assert with_effect is not None and len(with_effect) == round(clip.duration * sr)
    assert np.any(with_effect)

    output = tmp_path / "ramp.mp4"
    assert "speed ramps" in needs_render_engine(editor.project)
    assert export_video(editor.project, str(output), width=160, height=90, preset="ultrafast") == "render"
    with av.open(str(output)) as video:
        assert video.streams.video and video.streams.audio
        assert video.duration / av.time_base == pytest.approx(clip.duration, abs=0.08)
