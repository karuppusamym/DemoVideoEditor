import subprocess

import av
import numpy as np
import pytest

from clipeditor.editor import Editor
from clipeditor.export import ffmpeg_exe
from clipeditor.export_render import export_video


def make_video(tmp_path, fps, count):
    path = tmp_path / "numbered-frames.mkv"
    frames = np.broadcast_to(np.linspace(45, 225, count, dtype=np.uint8)[:, None, None, None], (count, 48, 64, 3))
    subprocess.run([ffmpeg_exe(), "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                    "-video_size", "64x48", "-framerate", str(fps), "-i", "pipe:0", "-c:v", "ffv1", str(path)],
                   input=frames.tobytes(), check=True)
    editor = Editor()
    editor.project.width, editor.project.height, editor.project.fps = 64, 48, fps
    asset = editor.import_file(str(path))
    clip = editor.add_clip(asset.id)
    return editor, clip


def frames_at(path):
    with av.open(str(path)) as video:
        return np.stack([frame.to_ndarray(format="rgb24") for frame in video.decode(video=0)])


@pytest.mark.parametrize("fade", [False, True])
def test_remove_one_60fps_frame_keeps_surrounding_frames(tmp_path, fade):
    editor, clip = make_video(tmp_path, 60, 12)
    if fade:
        editor.set_clip_props(clip.id, fade_in=0.085, fade_out=0.065)
    baseline, edited = tmp_path / "before.mp4", tmp_path / "after.mp4"
    export_video(editor.project, str(baseline), engine="render", preset="ultrafast", crf=0)
    before = frames_at(baseline)
    assert len(before) == 12

    assert editor.remove_frame(1.75 / 60)  # between frames, inside frame index 1
    assert editor.project.duration == pytest.approx(11 / 60)
    assert editor.project.video_tracks()[-1].sorted_clips()[0].duration == pytest.approx(1 / 60)
    assert export_video(editor.project, str(edited), preset="ultrafast", crf=0) == "render"
    after = frames_at(edited)
    assert after.shape[0] == 11
    np.testing.assert_allclose(after.astype(int), np.delete(before, 1, axis=0).astype(int), atol=2)
    editor.undo()
    assert editor.project.duration == pytest.approx(12 / 60)
    assert not editor.project.active_timeline.frame_cuts


@pytest.mark.parametrize("speed", [2.0, 3.0])
def test_speed_cut_remove_and_export(tmp_path, speed):
    editor, clip = make_video(tmp_path, 30, 30)
    editor.set_clip_props(clip.id, speed=speed)
    initial = tmp_path / "fast.mp4"
    assert export_video(editor.project, str(initial), preset="ultrafast") == "render"
    fast = frames_at(initial)
    assert len(fast) == round(30 / speed)
    assert fast[0].mean() == pytest.approx(45, abs=2)
    assert fast[-1].mean() > 155  # reaches the end of the source

    editor.split(1 / 30)
    left, right = editor.project.video_tracks()[-1].sorted_clips()
    assert left.duration == pytest.approx(1 / 30)
    assert right.in_point == pytest.approx(speed / 30)
    editor.selection = {left.id}
    editor.delete_selected(ripple=True)
    cut = tmp_path / "fast-cut.mp4"
    export_video(editor.project, str(cut), preset="ultrafast")
    np.testing.assert_allclose(frames_at(cut).astype(int), fast[1:].astype(int), atol=2)


def test_frame_removal_rejects_transition_overlap_and_invalid_time_without_edit(tmp_path):
    editor, clip = make_video(tmp_path, 30, 30)
    other = editor.add_clip(clip.asset_id)
    assert editor.set_transition(other.id, "dissolve", 0.25)
    snapshot = editor.snapshot()
    history = len(editor._undo)
    with pytest.raises(ValueError, match="transition"):
        editor.remove_frame(0.9)
    assert not editor.remove_frame(editor.project.duration)
    assert editor.snapshot() == snapshot and len(editor._undo) == history
