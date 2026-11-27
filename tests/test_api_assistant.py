"""EditorAPI tools, LocalAssistant, the Claude tool loop (fake client) and the MCP server.

No network or API key needed."""
import json
import os
from types import SimpleNamespace as NS

import anyio
import pytest

from clipeditor.ai.assistant import MODEL, Assistant, LocalAssistant, make_assistant
from clipeditor.api import EditorAPI, validate
from clipeditor.editor import Editor
from clipeditor.media import AudioCache

from .sample_media import SAMPLES

A = os.path.join(SAMPLES, "clip_a.mp4")
B = os.path.join(SAMPLES, "clip_b.mp4")


@pytest.fixture
def api():
    api = EditorAPI(Editor(), AudioCache())
    assert "error" not in api.call("import_media", {"path": A, "add_to_timeline": True})
    assert "error" not in api.call("import_media", {"path": B, "add_to_timeline": True})
    return api


def main_clips(api):
    s = api.call("get_project_summary")
    return [t for t in s["tracks"] if t["kind"] == "video"][-1]["clips"]


# ---- specs ---------------------------------------------------------------------------
def _check_schema(s, path):
    assert isinstance(s, dict), path
    t = s.get("type")
    assert t in ("object", "array", "string", "number", "integer", "boolean"), (path, t)
    if t == "object":
        props = s.get("properties", {})
        assert isinstance(props, dict)
        for k in s.get("required", []):
            assert k in props, (path, k)
        for k, v in props.items():
            _check_schema(v, f"{path}.{k}")
    if t == "array" and "items" in s:
        _check_schema(s["items"], path + "[]")
    if "enum" in s:
        assert isinstance(s["enum"], list) and s["enum"]


def test_tool_specs_are_valid():
    specs = EditorAPI(Editor()).tool_specs()
    names = [s["name"] for s in specs]
    assert len(names) == len(set(names)) >= 20
    for s in specs:
        assert set(s) == {"name", "description", "input_schema"}
        assert len(s["description"]) > 10
        assert s["input_schema"]["type"] == "object"
        _check_schema(s["input_schema"], s["name"])
        json.dumps(s)
    try:  # full Draft-2020 meta-schema check when jsonschema is installed
        import jsonschema
    except ImportError:
        return
    for s in specs:
        jsonschema.Draft202012Validator.check_schema(s["input_schema"])


def test_validate():
    sch = {"type": "object", "properties": {"t": {"type": "number", "minimum": 0}}, "required": ["t"],
           "additionalProperties": False}
    assert validate(sch, {"t": 1}) == []
    assert validate(sch, {}) and validate(sch, {"t": -1}) and validate(sch, {"t": "x"})
    assert validate(sch, {"t": 1, "z": 2})
    assert validate(sch, {"t": True})


def test_bad_calls_return_errors(api):
    assert "error" in api.call("nope", {})
    assert "error" in api.call("split_at", {})
    assert "error" in api.call("set_clip_properties", {"clip_id": "zzz", "properties": {"volume": 1}})
    assert "error" in api.call("set_canvas", {"preset": "3:2"})
    assert "error" in api.call("import_media", {"path": "does/not/exist.mp4"})


# ---- cheap tools + undo ----------------------------------------------------------------
def test_summary_and_media(api):
    s = api.call("get_project_summary")
    json.dumps(s)
    assert s["duration"] == 14.0 and s["canvas"]["aspect"] == "16:9"
    clips = main_clips(api)
    assert [c["name"] for c in clips] == ["clip_a.mp4", "clip_b.mp4"]
    m = api.call("list_media")["media"]
    assert {x["name"] for x in m} == {"clip_a.mp4", "clip_b.mp4"}
    assert all(x["clips_on_timeline"] == 1 for x in m)
    full = api.call("get_clip", {"clip_id": clips[0]["id"]})
    assert full["properties"]["saturation"] == 1.0


CHEAP = [
    ("split_at", lambda c: {"t": 3.0}),
    ("delete_clips", lambda c: {"clip_ids": [c[0]["id"]], "ripple": True}),
    ("remove_range", lambda c: {"start": 2.0, "end": 4.0}),
    ("duplicate_range", lambda c: {"start": 1.0, "end": 2.0}),
    ("move_clip", lambda c: {"clip_id": c[1]["id"], "start": 20.0}),
    ("move_clip", lambda c: {"clip_id": c[1]["id"], "start": 1.0, "track": "V2"}),
    ("trim_clip", lambda c: {"clip_id": c[0]["id"], "in_point": 1.0, "out_point": 5.0}),
    ("set_clip_properties", lambda c: {"clip_id": c[0]["id"], "properties": {
        "volume": 0.5, "opacity": 0.8, "speed": 2.0, "scale": 1.2, "pos_x": 0.1, "pos_y": -0.1, "rotation": 15,
        "fade_in": 0.5, "fade_out": 0.5, "brightness": 0.1, "contrast": 1.2, "saturation": 0.0,
        "blend_mode": "screen", "effects": [{"type": "blur", "params": {"radius": 3}}],
        "anim_in": "", "anim_out": "", "anim_loop": ""}}),
    ("detach_audio", lambda c: {"clip_ids": [c[0]["id"]]}),
    ("add_track", lambda c: {"kind": "audio"}),
    ("add_text", lambda c: {"text": "Hello", "start": 1.0, "duration": 2.0,
                            "style": {"preset": "title", "color": "#ffcc00"}}),
    ("add_transition", lambda c: {"clip_id": c[1]["id"], "kind": "fadeblack", "duration": 0.75}),
    ("transitions_everywhere", lambda c: {"kind": "slideleft", "duration": 0.5}),
    ("set_canvas", lambda c: {"preset": "9:16"}),
    ("add_clip", lambda c: {"asset_id": c[0]["asset_id"], "track": "V2", "start": 0.0}),
]


@pytest.mark.parametrize("name,make", CHEAP, ids=[f"{i}_{n}" for i, (n, _) in enumerate(CHEAP)])
def test_tool_then_undo_restores(api, name, make):
    if name == "set_clip_properties":  # effects/animation registries may be filled by other modules
        from clipeditor import animate, effects
        args = make(main_clips(api))
        if effects.REGISTRY:
            args["properties"]["effects"] = [{"type": next(iter(effects.REGISTRY))}]
    else:
        args = make(main_clips(api))
    before = api.editor.project.to_dict()
    r = api.call(name, args)
    assert "error" not in r, r
    json.dumps(r)
    after = api.editor.project.to_dict()
    assert after != before
    assert api.call("undo")["ok"]
    assert api.editor.project.to_dict() == before
    assert api.call("redo")["ok"]
    assert api.editor.project.to_dict() == after


def test_set_props_applied(api):
    c = main_clips(api)[0]
    api.call("set_clip_properties", {"clip_id": c["id"], "properties": {"saturation": 0.0, "volume": 0.3}})
    _, clip = api.editor.project.find_clip(c["id"])
    assert clip.saturation == 0.0 and clip.volume == 0.3
    assert len(api.editor._undo) == 3  # two adds + one property edit


def test_remove_silences_on_samples(api):
    r = api.call("remove_silences", {"threshold_db": -40})
    assert "error" not in r and r["removed"] >= 0
    if r["removed"]:
        assert api.call("undo")["ok"] and api.editor.project.duration == 14.0


def test_save_and_undo_redo_noop(api, tmp_path):
    assert "error" in api.call("save_project", {})
    p = str(tmp_path / "x.vedit.json")
    assert api.call("save_project", {"path": p})["path"] == p
    assert os.path.isfile(p)
    ed = Editor()
    a2 = EditorAPI(ed)
    assert a2.call("undo")["ok"] is False and a2.call("redo")["ok"] is False


def test_invoke_hook_routes_calls(api):
    seen = []
    api2 = EditorAPI(api.editor, api.cache, invoke=lambda fn: (seen.append(1), fn())[1])
    api2.call("split_at", {"t": 2.0})
    api2.call("get_project_summary")
    assert len(seen) == 2


# ---- LocalAssistant ----------------------------------------------------------------------
def run_local(api, text):
    events = []
    reply = LocalAssistant(api).send(text, events.append)
    calls = [e["name"] for e in events if e["type"] == "tool_call"]
    assert events[-1]["type"] == "done"
    return reply, calls


@pytest.mark.parametrize("text,tool", [
    ("make it vertical for TikTok", "set_canvas"),
    ("Please undo", "undo"),
    ("add dissolve transitions", "transitions_everywhere"),
    ("add fade to black transitions 1s", "transitions_everywhere"),
    ('add title "Summer Trip" at 2s for 3 seconds', "add_text"),
    ("split at 4.5s", "split_at"),
    ("cut from 2s to 5s", "remove_range"),
    ("duplicate 1 to 2", "duplicate_range"),
    ("speed up 2x", "set_clip_properties"),
    ("make it black and white", "set_clip_properties"),
    ("mute the video", "set_clip_properties"),
    ("remove silences", "remove_silences"),
    ("add captions", "auto_captions"),
    ("square please", "set_canvas"),
    ("fade in and out", "set_clip_properties"),
    ("delete the last clip", "delete_clips"),
])
def test_local_assistant_intents(api, text, tool, monkeypatch):
    from clipeditor.ai import captions
    monkeypatch.setattr(captions, "is_available", lambda: False)  # never download/run Whisper here
    reply, calls = run_local(api, text)
    assert tool in calls, (text, calls, reply)


def test_local_assistant_effects(api):
    run_local(api, "make it vertical")
    assert (api.editor.project.width, api.editor.project.height) == (1080, 1920)
    run_local(api, "undo")
    assert api.editor.project.width == 1920
    run_local(api, 'add title "Hi there" at 2s')
    texts = [c for t in api.editor.project.tracks for c in t.clips if c.text]
    assert texts and texts[0].text == "Hi there" and texts[0].start == 2.0
    run_local(api, "add dissolve transitions")
    assert any(c.transition > 0 and c.transition_kind == "dissolve"
               for t in api.editor.project.tracks for c in t.clips)
    run_local(api, "cut 1s-3s")
    reply, _ = run_local(api, "what's in my project?")
    assert "long" in reply
    reply, calls = run_local(api, "fly me to the moon")
    assert not calls and "didn't understand" in reply


def test_local_export(api, tmp_path):
    out = tmp_path / "o.mp4"
    reply, calls = run_local(api, f"export 480p draft to {out}")
    assert calls == ["export_video"]
    assert out.exists() and out.stat().st_size > 1000, reply


# ---- Claude loop with a fake client --------------------------------------------------------
class FakeStream:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        for b in self.msg.content:
            if b.type == "text":
                for part in (b.text[: len(b.text) // 2], b.text[len(b.text) // 2:]):
                    yield NS(type="content_block_delta", delta=NS(type="text_delta", text=part))
            elif b.type == "thinking" and b.thinking:
                yield NS(type="content_block_delta", delta=NS(type="thinking_delta", thinking=b.thinking))

    def get_final_message(self):
        return self.msg


class FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.beta = NS(messages=NS(stream=self._stream))

    def _stream(self, **kw):
        self.calls.append({**kw, "messages": list(kw["messages"])})
        return FakeStream(self.script.pop(0))


def tool_use(id_, name, args):
    return NS(type="tool_use", id=id_, name=name, input=args)


def text(t):
    return NS(type="text", text=t)


def test_claude_loop_with_fake_client(api):
    script = [
        NS(stop_reason="tool_use", content=[NS(type="thinking", thinking="Checking the timeline.", signature="s"),
                                           tool_use("t1", "get_project_summary", {})]),
        NS(stop_reason="tool_use", content=[text("Splitting now."), tool_use("t2", "split_at", {"t": 3.0}),
                                           tool_use("t3", "set_canvas", {"preset": "9:16"})]),
        NS(stop_reason="end_turn", content=[text("I split the clip at 3s and made the canvas vertical.")]),
    ]
    fake = FakeClient(script)
    asst = Assistant(api, client=fake)
    events = []
    reply = asst.send("split at 3 and make it vertical", events.append)
    assert reply == "I split the clip at 3s and made the canvas vertical."
    kinds = [e["type"] for e in events]
    assert kinds[-1] == "done" and "status" in kinds and "text" in kinds
    assert [e["name"] for e in events if e["type"] == "tool_call"] == ["get_project_summary", "split_at", "set_canvas"]
    assert all(not e["is_error"] for e in events if e["type"] == "tool_result")
    assert api.editor.project.height == 1920 and len(main_clips(api)) == 3
    # request shape
    first = fake.calls[0]
    assert first["model"] == MODEL == "claude-opus-5-5"
    assert [t["name"] for t in first["tools"]] == api.tool_names()
    assert all(t["eager_input_streaming"] for t in first["tools"])
    assert first["system"] and first["cache_control"] == {"type": "ephemeral"}
    assert "temperature" not in first and first["thinking"]["type"] == "adaptive"
    # parallel tool results go back in ONE user message, matched by id
    third = fake.calls[2]["messages"]
    assert third[-1]["role"] == "user" and [r["tool_use_id"] for r in third[-1]["content"]] == ["t2", "t3"]
    assert [m["role"] for m in asst.messages] == ["user", "assistant", "user", "assistant", "user", "assistant"]
    # undo still works for edits made by the assistant
    api.call("undo")
    assert api.editor.project.height == 1080


def test_claude_loop_bad_input_and_cap(api):
    script = [NS(stop_reason="tool_use", content=[tool_use(f"x{i}", "split_at", {"t": "soon"})]) for i in range(3)]
    asst = Assistant(api, client=FakeClient(script), max_iterations=3)
    events = []
    asst.send("split", events.append)
    results = [e for e in events if e["type"] == "tool_result"]
    assert len(results) == 3 and all(r["is_error"] for r in results)
    assert events[-1]["stop_reason"] == "max_iterations"
    assert len(main_clips(api)) == 2  # nothing ran


def test_claude_loop_api_error_rolls_back(api):
    class Boom(FakeClient):
        def _stream(self, **kw):
            raise RuntimeError("network down")

    asst = Assistant(api, client=Boom([]))
    events = []
    assert asst.send("hi", events.append) == ""
    assert events[0]["type"] == "error" and events[-1]["type"] == "done"
    assert asst.messages == []


def test_make_assistant_picks(api, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert isinstance(make_assistant(api), LocalAssistant)
    assert isinstance(make_assistant(api, client=FakeClient([])), Assistant)
    assert isinstance(make_assistant(api, api_key="sk-test-not-real"), Assistant)


# ---- MCP server ---------------------------------------------------------------------------
def test_mcp_server_in_process(tmp_path):
    from mcp import Client

    from clipeditor import mcp_server

    proj = tmp_path / "p.vedit.json"

    async def go():
        server = mcp_server.build_server()
        async with Client(server) as c:
            tools = (await c.list_tools()).tools
            names = [t.name for t in tools]
            assert {"open_project", "new_project", "get_project_summary", "export_video"} <= set(names)
            r = await c.call_tool("import_media", {"path": A, "add_to_timeline": True})
            assert not r.is_error
            r = await c.call_tool("save_project", {"path": str(proj)})
            assert not r.is_error and proj.exists()
            r = await c.call_tool("new_project", {})
            s = json.loads((await c.call_tool("get_project_summary", {})).content[0].text)
            assert s["duration"] == 0
            r = await c.call_tool("open_project", {"path": str(proj)})
            assert not r.is_error
            assert r.structured_content["summary"]["duration"] == 8.0
            r = await c.call_tool("split_at", {"t": -1})
            assert r.is_error

    anyio.run(go)
    assert callable(mcp_server.main)
