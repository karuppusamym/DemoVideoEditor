"""Stable, Qt-free command layer over an `Editor`.

The single source of editing tools for the in-app assistant (clipeditor.ai.assistant), the MCP
server (clipeditor.mcp_server) and scripts::

    api = EditorAPI(editor, AudioCache())
    api.tool_specs()                       # [{name, description, input_schema}, ...]
    api.call("split_at", {"t": 3.0})       # -> JSON-serialisable dict, or {"error": ...}

Every mutating tool goes through Editor methods (or Editor.checkpoint() + Editor.notify()),
so each tool call is one undo step.

Threading: `call()` may run on a worker thread. Pass `invoke=fn` where `fn(callable)` runs the
callable on the thread that owns the Editor (the GUI thread) and returns its result; all reads and
mutations of the editor are routed through it. Slow work (decoding audio, transcribing, FFmpeg
export) runs on the calling thread against a snapshot copy of the project.
"""
from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Callable

from .model import MIN_CLIP, Project

CANVAS_PRESETS: dict[str, tuple[int, int]] = {
    "16:9": (1920, 1080),
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "4:5": (1080, 1350),
}
RESOLUTIONS = {"source": None, "2160p": 2160, "1440p": 1440, "1080p": 1080, "720p": 720, "480p": 480}
QUALITY = {"high": (18, "slow"), "medium": (21, "medium"), "low": (26, "veryfast"), "draft": (30, "ultrafast")}
BLEND_MODES = ["normal", "multiply", "screen", "overlay", "darken", "lighten", "color_dodge", "color_burn",
               "hard_light", "soft_light", "difference", "exclusion", "add"]

# set_clip_properties: name -> JSON schema
_NUM = {"type": "number"}
CLIP_PROPS: dict[str, dict] = {
    "volume": {"type": "number", "minimum": 0, "maximum": 2, "description": "1 = original"},
    "opacity": {"type": "number", "minimum": 0, "maximum": 1},
    "speed": {"type": "number", "minimum": 0.25, "maximum": 4},
    "scale": {"type": "number", "minimum": 0.05, "maximum": 10, "description": "1 = fit to canvas"},
    "pos_x": {"type": "number", "description": "centre offset, fraction of canvas width"},
    "pos_y": {"type": "number", "description": "centre offset, fraction of canvas height"},
    "rotation": {"type": "number", "description": "degrees clockwise"},
    "flip_h": {"type": "boolean"},
    "flip_v": {"type": "boolean"},
    "fade_in": {"type": "number", "minimum": 0},
    "fade_out": {"type": "number", "minimum": 0},
    "brightness": {"type": "number", "minimum": -1, "maximum": 1, "description": "0 = original"},
    "contrast": {"type": "number", "minimum": 0, "maximum": 2, "description": "1 = original"},
    "saturation": {"type": "number", "minimum": 0, "maximum": 3, "description": "1 = original, 0 = greyscale"},
    "blend_mode": {"type": "string", "enum": BLEND_MODES},
    "effects": {"type": "array", "description": "full list, replaces existing: [{type, params, enabled}]",
                "items": {"type": "object", "properties": {
                    "type": {"type": "string"}, "params": {"type": "object"}, "enabled": {"type": "boolean"}},
                    "required": ["type"]}},
    "anim_in": {"type": "string", "description": "in-animation preset name, '' = none"},
    "anim_in_dur": {"type": "number", "minimum": 0},
    "anim_out": {"type": "string", "description": "out-animation preset name, '' = none"},
    "anim_out_dur": {"type": "number", "minimum": 0},
    "anim_loop": {"type": "string", "description": "loop-animation preset name, '' = none"},
    "anim_loop_speed": {"type": "number", "minimum": 0},
    # text clips
    "text": {"type": "string"},
    "font_family": {"type": "string"},
    "font_size": {"type": "integer", "minimum": 8, "maximum": 400},
    "color": {"type": "string", "description": "#rrggbb"},
    "bg_color": {"type": "string", "description": "'' = no box, else #rrggbb or #rrggbbaa"},
    "bold": {"type": "boolean"},
}
TEXT_STYLE_KEYS = ("font_family", "font_size", "color", "bg_color", "bold", "pos_x", "pos_y", "scale",
                   "opacity", "fade_in", "fade_out", "anim_in", "anim_out", "anim_loop")
TEXT_PRESETS = {
    "title": dict(font_size=96, bold=True, pos_y=0.0),
    "subtitle": dict(font_size=54, bold=False, pos_y=0.36, bg_color="#000000aa"),
    "lower_third": dict(font_size=48, bold=True, pos_x=-0.25, pos_y=0.33, bg_color="#000000aa"),
    "caption_big": dict(font_size=84, bold=True, pos_y=0.15),
}


class ToolError(Exception):
    """Raised inside a tool; turned into {"error": message} by call()."""


# ---- tiny JSON-schema validator (the subset our schemas use) -------------------------------
_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool}


def validate(schema: dict, value: Any, path: str = "") -> list[str]:
    """Return a list of problems with `value` against `schema` (empty = valid)."""
    where = path or "input"
    t = schema.get("type")
    errs: list[str] = []
    if t in ("number", "integer"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return [f"{where} must be a {t}"]
        if t == "integer" and not float(value).is_integer():
            return [f"{where} must be an integer"]
        if "minimum" in schema and value < schema["minimum"]:
            errs.append(f"{where} must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errs.append(f"{where} must be <= {schema['maximum']}")
    elif t in _TYPES:
        if not isinstance(value, _TYPES[t]):
            return [f"{where} must be a{'n' if t[0] in 'aeiou' else ''} {t}"]
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{where} must be one of {schema['enum']}")
    if t == "object":
        props = schema.get("properties", {})
        for k in schema.get("required", []):
            if k not in value:
                errs.append(f"{where}.{k} is required" if path else f"'{k}' is required")
        for k, v in value.items():
            if k in props:
                errs += validate(props[k], v, f"{path}.{k}" if path else k)
            elif schema.get("additionalProperties") is False:
                errs.append(f"unknown field '{k}'" + (f" in {path}" if path else ""))
    elif t == "array" and "items" in schema:
        for i, v in enumerate(value):
            errs += validate(schema["items"], v, f"{where}[{i}]")
    return errs


def _obj(props: dict | None = None, required: tuple = ()) -> dict:
    return {"type": "object", "properties": props or {}, "required": list(required), "additionalProperties": False}


def _r(x: float) -> float:
    return round(float(x), 3)


class EditorAPI:
    """Tool surface over one Editor. See module docstring."""

    def __init__(self, editor, audio_cache=None, invoke: Callable[[Callable[[], Any]], Any] | None = None):
        from .media import AudioCache

        self.editor = editor
        self.cache = audio_cache if audio_cache is not None else AudioCache()
        self._invoke = invoke
        self._tools: dict[str, tuple[str, dict, Callable[..., dict]]] = {}
        self._register()

    # ---- public --------------------------------------------------------------
    def tool_specs(self) -> list[dict]:
        """[{name, description, input_schema}] in a fixed order (stable for prompt caching)."""
        return [{"name": n, "description": d, "input_schema": s} for n, (d, s, _) in self._tools.items()]

    def tool_names(self) -> list[str]:
        return list(self._tools)

    def call(self, name: str, args: dict | None = None, progress: Callable[[float], None] | None = None) -> dict:
        """Run tool `name`. Returns a JSON-serialisable dict; failures give {"error": message}."""
        entry = self._tools.get(name)
        if entry is None:
            return {"error": f"Unknown tool '{name}'. Available: {', '.join(self._tools)}"}
        _, schema, fn = entry
        args = dict(args or {})
        problems = validate(schema, args)
        if problems:
            return {"error": "Invalid arguments: " + "; ".join(problems)}
        try:
            if name in ("auto_captions", "export_video", "remove_silences"):
                args["progress"] = progress
            out = fn(**args)
        except ToolError as e:
            return {"error": str(e)}
        except Exception as e:  # keep the assistant / MCP client alive
            return {"error": f"{type(e).__name__}: {e}"}
        return out if isinstance(out, dict) else {"result": out}

    # ---- helpers ---------------------------------------------------------------
    def _run(self, fn: Callable[[], Any]) -> Any:
        """Run on the editor-owning thread (GUI) if an invoke hook was given."""
        return self._invoke(fn) if self._invoke else fn()

    def _snapshot(self) -> Project:
        return Project.from_dict(self._run(lambda: self.editor.project.to_dict()))

    def _timeline_info(self, timeline) -> dict:
        return {"id": timeline.id, "name": timeline.name, "duration": _r(timeline.duration),
                "width": timeline.width, "height": timeline.height, "fps": timeline.fps,
                "active": timeline.id == self.editor.project.active_timeline_id}

    def _require_same_timeline(self, snapshot: Project):
        if self.editor.project.active_timeline_id != snapshot.active_timeline_id:
            raise ToolError("The active timeline changed while processing. Run the tool on that timeline again.")

    def _track(self, ref: str | None):
        if not ref:
            return None
        p = self.editor.project
        tr = p.track(ref) or next((t for t in p.tracks if t.name.lower() == ref.lower()), None)
        if tr is None:
            raise ToolError(f"No track '{ref}'. Tracks: " + ", ".join(f"{t.name} ({t.id})" for t in p.tracks))
        return tr

    def _clip(self, clip_id: str):
        tr, c = self.editor.project.find_clip(clip_id)
        if c is None:
            raise ToolError(f"No clip with id '{clip_id}'. Call get_project_summary for current ids.")
        return tr, c

    def _clip_name(self, c) -> str:
        if c.sequence_id:
            source = self.editor.project.timelines.get(c.sequence_id)
            return source.name if source is not None else "(missing timeline)"
        if c.text is not None:
            return f"Text: {c.text[:40]}"
        if c.shape is not None:
            return "Shape"
        a = self.editor.project.assets.get(c.asset_id)
        return a.name if a else "(missing media)"

    def _clip_info(self, c, full: bool = False) -> dict:
        d = {"id": c.id, "name": self._clip_name(c), "start": _r(c.start), "end": _r(c.end),
             "duration": _r(c.duration)}
        if c.asset_id:
            d.update(asset_id=c.asset_id, in_point=_r(c.in_point), out_point=_r(c.out_point))
        elif c.sequence_id:
            d.update(sequence_id=c.sequence_id, in_point=_r(c.in_point), out_point=_r(c.out_point))
        for k, default in (("speed", 1.0), ("volume", 1.0), ("opacity", 1.0)):
            if getattr(c, k) != default or full:
                d[k] = getattr(c, k)
        if c.transition > 0:
            d["transition"] = {"kind": c.transition_kind, "duration": _r(c.transition)}
        if full:
            skip = {"id", "asset_id", "start", "in_point", "out_point", "transition", "transition_kind"}
            d["properties"] = {k: v for k, v in asdict(c).items() if k not in skip}
        return d

    def _paths(self, path: str) -> str:
        path = os.path.expanduser(path)
        if not os.path.isabs(path):
            base = os.path.dirname(self.editor.path) if self.editor.path else os.getcwd()
            path = os.path.join(base, path)
        return os.path.abspath(path)

    def _load_audio(self, project: Project):
        for a in project.timeline_assets():
            if a.has_audio and self.cache.get(a.id) is None:
                self.cache.load(a)

    def _tool(self, name: str, description: str, schema: dict):
        def deco(fn):
            self._tools[name] = (description, schema, fn)
            return fn
        return deco

    # ---- tool definitions --------------------------------------------------------
    def _register(self):  # noqa: C901 - one flat list of tools reads best
        ed = self.editor
        T = self._tool

        # -- inspection --
        @T("get_project_summary",
           "Current project: active timeline name and ID, canvas size/fps, duration, tracks and clips (ids, names, timeline "
           "start/end in seconds, transitions), media count and undo state. Call this first, and again "
           "after edits whenever you need fresh clip ids.", _obj())
        def get_project_summary():
            def f():
                p = ed.project
                g = _gcd(p.width, p.height)
                return {
                    "path": ed.path, "unsaved_changes": ed.dirty,
                    "active_timeline": self._timeline_info(p.active_timeline),
                    "timeline_count": len(p.timelines),
                    "canvas": {"width": p.width, "height": p.height, "fps": p.fps,
                               "aspect": f"{p.width // g}:{p.height // g}"},
                    "duration": _r(p.duration),
                    "tracks": [{"id": t.id, "name": t.name, "kind": t.kind, "muted": t.muted, "hidden": t.hidden,
                                "clips": [self._clip_info(c) for c in t.sorted_clips()]} for t in p.tracks],
                    "media_count": len(p.assets),
                    "can_undo": bool(ed._undo), "can_redo": bool(ed._redo),
                    "note": "Video tracks are listed top (drawn on top) to bottom.",
                }
            return self._run(f)

        @T("get_clip", "All properties of one clip (transform, color, effects, animation, text style).",
           _obj({"clip_id": {"type": "string"}}, ("clip_id",)))
        def get_clip(clip_id):
            def f():
                tr, c = self._clip(clip_id)
                return {**self._clip_info(c, full=True), "track": {"id": tr.id, "name": tr.name}}
            return self._run(f)

        @T("list_media", "Media in the project bin: asset ids, file names, kind, duration, size, audio.", _obj())
        def list_media():
            def f():
                p = ed.project
                used: dict[str, int] = {}
                for t in p.tracks:
                    for c in t.clips:
                        used[c.asset_id] = used.get(c.asset_id, 0) + 1
                return {"media": [{"asset_id": a.id, "name": a.name, "path": a.path, "kind": a.kind,
                                   "duration": None if a.kind == "image" else _r(a.duration),
                                   "width": a.width, "height": a.height, "has_audio": a.has_audio,
                                   "clips_on_timeline": used.get(a.id, 0)} for a in p.assets.values()]}
            return self._run(f)

        # -- timelines --
        timeline_ref = {"type": "string", "description": "Timeline name or ID from list_timelines"}

        @T("list_timelines", "List the timelines in the current project, including the active timeline.", _obj())
        def list_timelines():
            return self._run(lambda: {"timelines": [self._timeline_info(t) for t in ed.project.timelines.values()]})

        @T("create_timeline", "Create and activate a named timeline. Set duplicate to copy the current edit.",
           _obj({"name": {"type": "string"}, "duplicate": {"type": "boolean"}}))
        def create_timeline(name=None, duplicate=False):
            return self._run(lambda: {"ok": True, "timeline": self._timeline_info(ed.create_timeline(name, duplicate))})

        @T("switch_timeline", "Make a timeline active for subsequent editing and export calls.",
           _obj({"timeline": timeline_ref}, ("timeline",)))
        def switch_timeline(timeline):
            return self._run(lambda: {"ok": True, "timeline": self._timeline_info(ed.switch_timeline(timeline))})

        @T("add_sequence", "Place a linked timeline as a nested clip in the active timeline. Times are seconds.",
           _obj({"timeline": timeline_ref, "track": {"type": "string"},
                 "start": {"type": "number", "minimum": 0}}, ("timeline",)))
        def add_sequence(timeline, track=None, start=None):
            def f():
                target = self._track(track)
                clip = ed.add_sequence(timeline, target.id if target else None, start)
                return {"ok": True, "clip": self._clip_info(clip)}
            return self._run(f)

        @T("rename_timeline", "Rename a timeline in this project. Names must be unique.",
           _obj({"timeline": timeline_ref, "name": {"type": "string"}}, ("timeline", "name")))
        def rename_timeline(timeline, name):
            return self._run(lambda: {"ok": True, "timeline": self._timeline_info(ed.rename_timeline(timeline, name))})

        @T("delete_timeline", "Delete a timeline and its edits. This can be undone. One timeline must remain.",
           _obj({"timeline": timeline_ref}, ("timeline",)))
        def delete_timeline(timeline):
            def f():
                ed.delete_timeline(timeline)
                return {"ok": True, "active_timeline": self._timeline_info(ed.project.active_timeline)}
            return self._run(f)

        # -- media --
        @T("import_media", "Import a video, audio or image file into the media bin. Set add_to_timeline "
           "to also append it to the end of the main track.",
           _obj({"path": {"type": "string", "description": "file path"},
                 "add_to_timeline": {"type": "boolean"}}, ("path",)))
        def import_media(path, add_to_timeline=False):
            path = self._paths(path)
            if not os.path.isfile(path):
                raise ToolError(f"File not found: {path}")
            from .media import probe

            asset = probe(path)  # slow-ish decode, off the GUI thread

            def f():
                ed.project.assets[asset.id] = asset
                ed.dirty = True
                ed.notify("assets")
                out = {"asset_id": asset.id, "name": asset.name, "kind": asset.kind,
                       "duration": _r(asset.duration), "width": asset.width, "height": asset.height,
                       "has_audio": asset.has_audio}
                if add_to_timeline:
                    c = ed.add_clip(asset.id)
                    out["clip"] = self._clip_info(c)
                return out
            return self._run(f)

        @T("add_clip", "Place a media asset on the timeline. Default: appended to the end of the main video "
           "track (or first audio track for audio). Overwrites whatever is under it.",
           _obj({"asset_id": {"type": "string"},
                 "track": {"type": "string", "description": "track id or name (e.g. V1, A1)"},
                 "start": {"type": "number", "minimum": 0, "description": "timeline seconds"}}, ("asset_id",)))
        def add_clip(asset_id, track=None, start=None):
            def f():
                if asset_id not in ed.project.assets:
                    raise ToolError(f"No asset '{asset_id}'. Call list_media.")
                tr = self._track(track)
                c = ed.add_clip(asset_id, tr.id if tr else None, start)
                if c is None:
                    raise ToolError("Could not add the clip.")
                return {"ok": True, "clip": self._clip_info(c)}
            return self._run(f)

        # -- cutting --
        @T("split_at", "Split clips at timeline time t (seconds). Splits only clip_ids if given, otherwise "
           "every clip under t on every track.",
           _obj({"t": {"type": "number", "minimum": 0},
                 "clip_ids": {"type": "array", "items": {"type": "string"}}}, ("t",)))
        def split_at(t, clip_ids=None):
            def f():
                before = {c.id for tr in ed.project.tracks for c in tr.clips}
                saved = set(ed.selection)
                ed.selection = set(clip_ids or ())
                try:
                    ed.split(t)
                finally:
                    ed.selection = saved
                new = [c for tr in ed.project.tracks for c in tr.clips if c.id not in before]
                if not new:
                    raise ToolError(f"No clip to split at {t:.3f}s (too close to a clip edge or nothing there).")
                return {"ok": True, "new_clips": [self._clip_info(c) for c in new]}
            return self._run(f)

        @T("remove_frame", "Delete the single timeline frame containing t and close the gap across all tracks.",
           _obj({"t": {"type": "number", "minimum": 0}}, ("t",)))
        def remove_frame(t):
            def f():
                if not ed.remove_frame(t):
                    raise ToolError("Move the playhead onto an existing timeline frame.")
                return {"ok": True, "duration": _r(ed.project.duration)}
            return self._run(f)

        @T("delete_clips", "Delete clips by id. ripple=true closes the gaps they leave on their tracks.",
           _obj({"clip_ids": {"type": "array", "items": {"type": "string"}},
                 "ripple": {"type": "boolean"}}, ("clip_ids",)))
        def delete_clips(clip_ids, ripple=False):
            def f():
                ids = set(clip_ids)
                missing = [i for i in ids if ed.project.find_clip(i)[1] is None]
                if missing:
                    raise ToolError(f"No clips with ids {missing}.")
                if not ids:
                    raise ToolError("clip_ids is empty.")
                ed.selection = ids
                ed.delete_selected(ripple=ripple)
                return {"ok": True, "deleted": len(ids), "duration": _r(ed.project.duration)}
            return self._run(f)

        rng = _obj({"start": {"type": "number", "minimum": 0}, "end": {"type": "number", "minimum": 0}},
                   ("start", "end"))

        @T("remove_range", "Cut the time range [start, end) out of ALL tracks and close the gap (ripple delete).",
           _obj({**rng["properties"], "ripple": {"type": "boolean", "description": "default true"}},
                ("start", "end")))
        def remove_range(start, end, ripple=True):
            if end - start < MIN_CLIP / 2:
                raise ToolError("end must be after start.")

            def f():
                ed.remove_range(start, end, ripple=ripple)
                return {"ok": True, "removed_seconds": _r(end - start), "duration": _r(ed.project.duration)}
            return self._run(f)

        @T("duplicate_range", "Copy the time range [start, end) of all tracks and insert the copy right after it.",
           rng)
        def duplicate_range(start, end):
            if end - start < MIN_CLIP / 2:
                raise ToolError("end must be after start.")

            def f():
                ed.duplicate_range(start, end)
                return {"ok": True, "duration": _r(ed.project.duration)}
            return self._run(f)

        @T("move_clip", "Move a clip to a new timeline start time, optionally onto another track. Overwrites "
           "clips it lands on.",
           _obj({"clip_id": {"type": "string"}, "start": {"type": "number", "minimum": 0},
                 "track": {"type": "string", "description": "track id or name"}}, ("clip_id", "start")))
        def move_clip(clip_id, start, track=None):
            def f():
                src, c = self._clip(clip_id)
                dst = self._track(track) or src
                asset = ed.project.asset_for_clip(c)
                if not dst.accepts(asset):
                    raise ToolError(f"Track {dst.name} ({dst.kind}) can't hold this clip.")
                ed.checkpoint()
                src.clips.remove(c)
                c.start = start
                c.transition = 0.0
                dst.clear_range(c.start, c.end)
                dst.clips.append(c)
                ed.notify("edit")
                return {"ok": True, "clip": self._clip_info(c), "track": dst.name}
            return self._run(f)

        @T("trim_clip", "Change which part of the source a clip plays: in_point/out_point are SOURCE seconds "
           "(see get_clip). The clip keeps its timeline start.",
           _obj({"clip_id": {"type": "string"}, "in_point": {"type": "number", "minimum": 0},
                 "out_point": {"type": "number", "minimum": 0}}, ("clip_id",)))
        def trim_clip(clip_id, in_point=None, out_point=None):
            def f():
                _, c = self._clip(clip_id)
                a = in_point if in_point is not None else c.in_point
                b = out_point if out_point is not None else c.out_point
                asset = ed.project.asset_for_clip(c)
                limit = max(asset.duration, c.out_point) if asset is not None and c.sequence_id else (
                    asset.duration if asset is not None else None)
                if limit is not None and b > limit + 1e-6:
                    raise ToolError(f"out_point exceeds the source length ({limit:.3f}s).")
                if c.time_between(a, b) < ed.frame() - 1e-9:
                    raise ToolError("in_point must be before out_point.")
                ed.set_clip_props(clip_id, in_point=a, out_point=b)
                return {"ok": True, "clip": self._clip_info(c)}
            return self._run(f)

        # -- properties --
        @T("set_clip_properties", "Set one or more visual/audio/text properties of a clip in one undo step. "
           "Only include the properties you want to change.",
           _obj({"clip_id": {"type": "string"},
                 "properties": {"type": "object", "properties": CLIP_PROPS, "additionalProperties": False}},
                ("clip_id", "properties")))
        def set_clip_properties(clip_id, properties):
            from . import animate, effects

            if not properties:
                raise ToolError("properties is empty.")
            for slot in ("in", "out", "loop"):
                name = properties.get(f"anim_{slot}")
                names = animate.ANIMATIONS.get(slot, {})
                if name and names and name not in names:
                    raise ToolError(f"Unknown {slot} animation '{name}'. Options: {', '.join(names)}")
            if effects.REGISTRY:
                bad = [e["type"] for e in properties.get("effects", []) if e["type"] not in effects.REGISTRY]
                if bad:
                    raise ToolError(f"Unknown effects {bad}. Options: {', '.join(effects.REGISTRY)}")

            def f():
                _, c = self._clip(clip_id)
                text_only = {"text", "font_family", "font_size", "color", "bg_color", "bold"}
                if c.text is None and text_only & set(properties):
                    raise ToolError("Text properties only apply to text clips.")
                props = dict(properties)
                if "effects" in props:
                    props["effects"] = [{"type": e["type"], "params": dict(e.get("params", {})),
                                         "enabled": e.get("enabled", True)} for e in props["effects"]]
                if "font_size" in props:
                    props["font_size"] = int(props["font_size"])
                ed.set_clip_props(clip_id, **props)
                return {"ok": True, "clip": self._clip_info(c)}
            return self._run(f)

        @T("detach_audio", "Move the audio of video clips onto an audio track (the video clip is muted).",
           _obj({"clip_ids": {"type": "array", "items": {"type": "string"}}}, ("clip_ids",)))
        def detach_audio(clip_ids):
            def f():
                before = {c.id for tr in ed.project.tracks for c in tr.clips}
                saved = set(ed.selection)
                ed.selection = set(clip_ids)
                try:
                    ed.detach_audio()
                finally:
                    ed.selection = saved
                new = [c for tr in ed.project.tracks for c in tr.clips if c.id not in before]
                if not new:
                    raise ToolError("Nothing to detach (clips must be video with audio and volume > 0).")
                return {"ok": True, "audio_clips": [self._clip_info(c) for c in new]}
            return self._run(f)

        @T("add_track", "Add an empty video or audio track.",
           _obj({"kind": {"type": "string", "enum": ["video", "audio"]}}, ("kind",)))
        def add_track(kind):
            def f():
                # Editor.add_track(notify=True) checkpoints *after* inserting, so undo would keep
                # the track; checkpoint first ourselves.
                ed.checkpoint()
                tr = ed.add_track(kind, notify=False)
                ed.notify("edit")
                return {"ok": True, "track": {"id": tr.id, "name": tr.name, "kind": tr.kind}}
            return self._run(f)

        # -- text --
        style_schema = {"type": "object", "additionalProperties": False, "properties": {
            "preset": {"type": "string", "enum": list(TEXT_PRESETS)},
            **{k: CLIP_PROPS[k] for k in TEXT_STYLE_KEYS}}}

        @T("add_text", "Add a text title/overlay on the top-most free video track. style.preset is one of "
           f"{', '.join(TEXT_PRESETS)}; explicit style fields override it.",
           _obj({"text": {"type": "string"}, "start": {"type": "number", "minimum": 0},
                 "duration": {"type": "number", "minimum": 0.1}, "style": style_schema}, ("text",)))
        def add_text(text, start=0.0, duration=4.0, style=None):
            st = dict(style or {})
            st = {**TEXT_PRESETS.get(st.pop("preset", ""), {}), **st}
            if "font_size" in st:
                st["font_size"] = int(st["font_size"])

            def f():
                c = ed.add_text(text, start, duration, **st)
                return {"ok": True, "clip": self._clip_info(c)}
            return self._run(f)

        # -- transitions --
        from .transitions import KINDS

        @T("add_transition", "Add/change the transition INTO clip_id from the clip right before it on the same "
           "track (the clips overlap; later clips shift left). duration 0 removes it. Kinds: "
           + ", ".join(f"{k} ({v})" for k, v in KINDS.items()),
           _obj({"clip_id": {"type": "string"}, "kind": {"type": "string", "enum": list(KINDS)},
                 "duration": {"type": "number", "minimum": 0, "maximum": 5}}, ("clip_id",)))
        def add_transition(clip_id, kind="dissolve", duration=0.5):
            def f():
                self._clip(clip_id)
                if not ed.set_transition(clip_id, kind, duration):
                    raise ToolError("This clip has no clip directly before it on its track.")
                _, c = ed.project.find_clip(clip_id)
                return {"ok": True, "clip": self._clip_info(c)}
            return self._run(f)

        @T("transitions_everywhere", "Put the same transition on every cut of every video track.",
           _obj({"kind": {"type": "string", "enum": list(KINDS)},
                 "duration": {"type": "number", "minimum": 0.05, "maximum": 5}}))
        def transitions_everywhere(kind="dissolve", duration=0.5):
            def f():
                ed.transitions_everywhere(kind, duration)
                n = sum(1 for t in ed.project.video_tracks() for c in t.clips if c.transition > 0)
                return {"ok": True, "transitions": n, "duration": _r(ed.project.duration)}
            return self._run(f)

        # -- AI / audio --
        @T("remove_silences", "Detect quiet stretches in the timeline audio and cut them out of all tracks "
           "(one undo step). Lower threshold_db = only remove deeper silence.",
           _obj({"threshold_db": {"type": "number", "minimum": -90, "maximum": -5, "description": "default -40"},
                 "min_silence": {"type": "number", "minimum": 0.1, "description": "seconds, default 0.6"}}))
        def remove_silences(threshold_db=-40.0, min_silence=0.6, progress=None):
            from .silence import find_silences

            snap = self._snapshot()
            if snap.duration <= 0:
                raise ToolError("The timeline is empty.")
            self._load_audio(snap)
            ranges = find_silences(snap, self.cache, threshold_db, min_silence)
            if not ranges:
                return {"ok": True, "removed": 0, "removed_seconds": 0.0, "duration": _r(snap.duration)}

            def f():
                self._require_same_timeline(snap)
                secs = ed.remove_ranges(ranges)
                return {"ok": True, "removed": len(ranges), "removed_seconds": _r(secs),
                        "duration": _r(ed.project.duration)}
            return self._run(f)

        @T("auto_captions", "Transcribe the timeline speech offline (Whisper) and add a Captions track. Slow: "
           "can take a minute or more; the model may download on first use.",
           _obj({"model": {"type": "string", "enum": ["tiny", "base", "small"]},
                 "language": {"type": "string", "description": "ISO code like 'en'; omit to auto-detect"},
                 "style": {"type": "string", "enum": ["subtitle", "big"]}}))
        def auto_captions(model="base", language=None, style="subtitle", progress=None):
            from .ai import captions as C

            if not C.is_available():
                raise ToolError("Captions need faster-whisper: pip install faster-whisper")
            snap = self._snapshot()
            if snap.duration <= 0:
                raise ToolError("The timeline is empty.")
            audio = C.timeline_audio(snap, self.cache)
            segs = C.transcribe(audio, model, language or None, progress=progress)
            lines = C.split_for_screen(segs)
            if not lines:
                return {"ok": True, "captions": 0, "note": "No speech found."}
            def add_captions():
                self._require_same_timeline(snap)
                return C.add_captions(ed, lines, style)

            n = self._run(add_captions)
            return {"ok": True, "captions": n, "preview": [t for _, _, t in lines[:3]]}

        # -- canvas / history / files --
        @T("set_canvas", "Change the canvas aspect: 16:9 (YouTube), 9:16 (TikTok/Reels/Shorts), 1:1, 4:5.",
           _obj({"preset": {"type": "string", "enum": list(CANVAS_PRESETS)}}, ("preset",)))
        def set_canvas(preset):
            w, h = CANVAS_PRESETS[preset]

            def f():
                p = ed.project
                if (p.width, p.height) == (w, h):
                    return {"ok": True, "width": w, "height": h, "note": "Already this size."}
                ed.checkpoint()
                p.width, p.height = w, h
                ed.notify("edit")
                return {"ok": True, "width": w, "height": h}
            return self._run(f)

        @T("undo", "Undo the last edit.", _obj())
        def undo():
            def f():
                if not ed._undo:
                    return {"ok": False, "note": "Nothing to undo."}
                ed.undo()
                return {"ok": True, "duration": _r(ed.project.duration), "can_undo": bool(ed._undo)}
            return self._run(f)

        @T("redo", "Redo the last undone edit.", _obj())
        def redo():
            def f():
                if not ed._redo:
                    return {"ok": False, "note": "Nothing to redo."}
                ed.redo()
                return {"ok": True, "duration": _r(ed.project.duration), "can_redo": bool(ed._redo)}
            return self._run(f)

        @T("save_project", "Save the project file (.vedit.json). path is optional if it was saved before.",
           _obj({"path": {"type": "string"}}))
        def save_project(path=None):
            if not path and not ed.path:
                raise ToolError("This project has never been saved; give a path ending in .vedit.json.")
            full = self._paths(path) if path else ed.path

            def f():
                ed.save(full)
                return {"ok": True, "path": full}
            return self._run(f)

        @T("export_video", "Render the named or active timeline to an MP4 file. Slow (runs FFmpeg). resolution sets the short "
           "side of the output (source = canvas size).",
           _obj({"path": {"type": "string", "description": "output .mp4 path"},
                 "resolution": {"type": "string", "enum": list(RESOLUTIONS)},
                 "timeline": timeline_ref,
                 "quality": {"type": "string", "enum": list(QUALITY)}}, ("path",)))
        def export_video(path, resolution="source", quality="medium", timeline=None, progress=None):
            from .export_render import export_video as render_video

            out = self._paths(path)
            if not out.lower().endswith((".mp4", ".mov", ".mkv")):
                out += ".mp4"
            snap = self._snapshot()
            if timeline is not None:
                selected = snap.find_timeline(timeline)
                if selected is None:
                    raise ToolError(f"Timeline '{timeline}' was not found.")
                snap.active_timeline_id = selected.id
            if snap.duration <= 0:
                raise ToolError("The timeline is empty.")
            short = RESOLUTIONS[resolution]
            w = h = None
            if short:
                k = short / min(snap.width, snap.height)
                w, h = int(round(snap.width * k)), int(round(snap.height * k))
            crf, preset = QUALITY[quality]
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            render_video(snap, out, progress=progress, crf=crf, preset=preset, width=w, height=h)
            return {"ok": True, "path": out, "duration": _r(snap.duration),
                    "size_mb": round(os.path.getsize(out) / 1e6, 2)}


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return max(a, 1)
