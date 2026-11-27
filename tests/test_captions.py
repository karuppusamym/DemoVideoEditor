import os

import numpy as np
import pytest

from clipeditor.ai import captions as C
from clipeditor.editor import Editor
from clipeditor.media import AudioCache, probe
from clipeditor.model import Clip, Project, new_id

from .sample_media import SAMPLES



def _seg(words, t0=0.0, wdur=0.3, gap=0.05):
    """Segment with synthetic, evenly spaced word timings."""
    out, t = [], t0
    for w in words.split():
        out.append((t, t + wdur, w))
        t += wdur + gap
    return C.Segment(out[0][0], out[-1][1], words, out)


def _check_lines(lines, max_chars=42, max_dur=3.5):
    for i, (a, b, text) in enumerate(lines):
        rows = text.split("\n")
        assert 1 <= len(rows) <= 2
        assert all(len(r) <= max_chars for r in rows), text
        assert b - a >= C.MIN_DISPLAY - 1e-6 or (i + 1 < len(lines) and b == lines[i + 1][0])
        assert b - a <= max_dur + 1e-6
        if i + 1 < len(lines):
            assert b <= lines[i + 1][0] + 1e-9


def test_split_long_segment():
    text = ("the quick brown fox jumps over the lazy dog while the cat watches from the "
            "window and the bird sings a song about summer days and long evenings by the sea")
    seg = _seg(text)
    lines = C.split_for_screen([seg])
    assert len(lines) > 2
    _check_lines(lines)
    # every word survives, in order
    assert " ".join(t.replace("\n", " ") for _, _, t in lines) == text
    assert lines[0][0] == pytest.approx(0.0)


def test_split_two_lines_and_chars():
    seg = _seg("alpha beta gamma delta epsilon zeta eta theta iota kappa", wdur=0.1, gap=0.0)
    lines = C.split_for_screen([seg], max_chars=20, max_dur=10)
    _check_lines(lines, max_chars=20, max_dur=10)
    assert any("\n" in t for _, _, t in lines)


def test_split_sentence_and_gap_breaks():
    s1 = _seg("Hello and welcome to the video editor.")
    s2 = _seg("This is a caption test.", t0=s1.end + 0.1)
    s3 = _seg("After a pause", t0=s2.end + 2.0)
    lines = C.split_for_screen([s1, s2, s3])
    assert [t for _, _, t in lines] == ["Hello and welcome to the video editor.",
                                        "This is a caption test.", "After a pause"]
    _check_lines(lines)


def test_split_min_duration_no_overlap():
    a = C.Segment(0.0, 0.2, "Hi", [(0.0, 0.2, "Hi")])
    b = C.Segment(0.4, 0.6, "there", [(1.5, 1.7, "there")])
    c = C.Segment(1.8, 1.9, "ok", [(1.75, 1.9, "ok")])  # tight after b
    lines = C.split_for_screen([a, b, c])
    assert lines[0] == (0.0, 0.6, "Hi")  # extended to the minimum display time
    for (a0, a1, _), (b0, _, _) in zip(lines, lines[1:]):
        assert a1 <= b0
    assert all(b > a for a, b, _ in lines)


def test_split_without_word_timings():
    seg = C.Segment(0.0, 2.0, "one two three four", [])
    lines = C.split_for_screen([seg])
    assert lines == [(0.0, 2.0, "one two three four")]
    assert C.split_for_screen([]) == []


def test_timeline_audio_length_and_rate():
    p = Project.new()
    t = probe(os.path.join(SAMPLES, "tone.wav"))
    p.assets[t.id] = t
    p.tracks[2].clips.append(Clip(new_id(), t.id, 0.0, 0.0, t.duration))
    cache = AudioCache(48000)
    audio = C.timeline_audio(p, cache)  # loads the asset into the cache itself
    assert audio.dtype == np.float32 and audio.ndim == 1
    assert len(audio) == pytest.approx(p.duration * 16000, abs=2)
    assert abs(p.duration - 10.0) < 0.1
    assert np.abs(audio).max() > 0.05
    # dominant frequency is preserved by the resampler
    ref = cache.get(t.id).mean(axis=1)
    f_ref = np.argmax(np.abs(np.fft.rfft(ref[:48000]))) * 48000 / 48000
    f_out = np.argmax(np.abs(np.fft.rfft(audio[:16000]))) * 16000 / 16000
    assert f_out == pytest.approx(f_ref, abs=2)
    # non-integer ratio path (44.1 kHz cache)
    a2 = C.timeline_audio(p, AudioCache(44100))
    assert len(a2) == pytest.approx(p.duration * 16000, abs=2)


def test_add_captions_one_undo_step():
    ed = Editor()
    n_tracks = len(ed.project.tracks)
    events = []
    ed.listeners.append(events.append)
    lines = [(0.0, 1.5, "Hello"), (1.5, 3.0, "two\nlines")]
    assert C.add_captions(ed, lines) == 2
    top = ed.project.tracks[0]
    assert top.name == "Captions" and top.kind == "video" and len(top.clips) == 2
    c = top.clips[1]
    assert c.is_text and c.asset_id == "" and c.text == "two\nlines"
    assert (c.font_size, c.bold, c.pos_y, c.bg_color) == (54, False, 0.36, "#000000aa")
    assert c.start == 1.5 and c.duration == pytest.approx(1.5)
    assert events == ["edit"]
    assert C.add_captions(ed, lines, style="big") == 2
    b = ed.project.tracks[0].clips[0]
    assert (b.font_size, b.bold, b.pos_y, b.bg_color) == (84, True, 0.15, "")
    ed.undo()
    ed.undo()
    assert len(ed.project.tracks) == n_tracks
    assert C.add_captions(ed, []) == 0


@pytest.mark.model
@pytest.mark.skipif(not C.is_available(), reason="faster-whisper not installed")
def test_transcribe_end_to_end(tmp_path):
    from clipeditor import tts

    voice = next((v for v in tts.voices() if v["language"].lower().startswith("en")), None)
    if voice is None:
        pytest.skip("An English text-to-speech voice is needed for the caption fixture")
    path = tts.synthesize("Hello and welcome to the video editor caption test.",
                          voice_id=voice["id"], out_path=str(tmp_path / "speech.wav"))
    p = Project.new()
    a = probe(path)
    p.assets[a.id] = a
    p.tracks[2].clips.append(Clip(new_id(), a.id, 1.0, 0.0, a.duration))  # starts at 1 s
    audio = C.timeline_audio(p, AudioCache())
    prog, stat = [], []
    try:
        segs = C.transcribe(audio, "tiny", language="en", progress=prog.append, status=stat.append)
    except Exception as e:  # first run needs network for the model download
        pytest.skip(f"model unavailable: {e}")
    text = " ".join(s.text for s in segs).lower()
    for w in ("hello", "welcome", "video", "editor", "caption", "test"):
        assert w in text, text
    assert prog and prog[-1] == 1.0 and "Listening…" in stat
    assert segs[0].words and segs[0].start >= 0.5  # offset by the clip start
    lines = C.split_for_screen(segs)
    _check_lines(lines)
    ed = Editor(p)
    assert C.add_captions(ed, lines) == len(lines)
    # cancellation stops early
    assert C.transcribe(audio, "tiny", cancelled=lambda: True) == []
