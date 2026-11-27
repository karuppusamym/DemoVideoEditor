"""Chat-to-edit assistant over `EditorAPI` tools. Qt-free.

    asst = make_assistant(api)              # Claude if a key is configured, else LocalAssistant
    asst.send("remove the silences and add captions", on_event)

`send()` blocks until the turn is finished (run it on a worker thread). `on_event(dict)` is called
on that same thread with one of:

    {"type": "text",        "text": str}                     # streamed reply text (delta)
    {"type": "status",      "text": str}                     # progress note between tool calls (delta)
    {"type": "tool_call",   "id": str, "name": str, "args": dict}
    {"type": "tool_result", "id": str, "name": str, "result": dict, "is_error": bool}
    {"type": "error",       "message": str}
    {"type": "done",        "text": str, "stop_reason": str}  # always sent last

The API key comes from the ANTHROPIC_API_KEY environment variable or the `api_key` argument; it is
never logged or stored by this module.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

from ..api import validate

MODEL = "claude-opus-5-5"
EFFORT = "medium"
MAX_ITERATIONS = 24
MAX_TOKENS = 64000
BETAS = ["server-side-fallback-2026-07-01", "thinking-display-updates-2026-08-18"]

SYSTEM_PROMPT = """You are the editing assistant inside ClipEditor, a desktop video editor. You are a \
friendly, expert video editor: you turn the user's requests into precise timeline edits using the tools.

How to work:
- Always call get_project_summary first in a conversation (and again after edits that change clip ids \
or timing) so you work from the real tracks, clip ids and times. Never invent ids.
- All times are timeline seconds. Video tracks are listed top (drawn on top) to bottom; the main \
footage is usually on the bottom video track.
- Prefer the fewest tool calls that do the job. Every edit is one undo step, so the user can always \
undo; never try to "clean up" by deleting things the user did not ask about.
- auto_captions and export_video are slow; run them only when the user asks for captions or an \
export. Only save or export to paths the user gave or clearly implied.
- If a request is ambiguous in a way that matters (which clip, which time), ask one short question \
instead of guessing.
- When you are done, explain what you changed in one or two plain sentences (mention it can be \
undone with Ctrl+Z if the change is large). No markdown headings or tables."""

Emit = Callable[[dict], None]


def _noop(_e: dict) -> None:
    pass


def has_api_key(api_key: str | None = None) -> bool:
    return bool(api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


class Assistant:
    """Claude-powered assistant: manual, streamed tool-use loop over api.tool_specs().

    `client` is any object with the Anthropic SDK shape `client.beta.messages.stream(**kw)`
    (tests pass a fake). The conversation persists across send() calls; reset() clears it."""

    name = "Claude"

    def __init__(self, api, client=None, api_key: str | None = None, model: str = MODEL,
                 effort: str = EFFORT, max_iterations: int = MAX_ITERATIONS):
        self.api = api
        if client is None:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort
        self.max_iterations = max_iterations
        self.messages: list[dict] = []
        self._cancel = False
        # Fixed tool list + system prompt = stable cached prefix across turns.
        self._specs = {s["name"]: s for s in api.tool_specs()}
        self._tools = [{**s, "eager_input_streaming": True} for s in api.tool_specs()]

    def reset(self):
        self.messages = []

    def cancel(self):
        """Stop after the current model response / tool call."""
        self._cancel = True

    # ---- one model request -------------------------------------------------------
    def _request(self, emit: Emit):
        kwargs = dict(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=self._tools,
            messages=self.messages,
            thinking={"type": "adaptive", "display": "updates"},
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},
            betas=BETAS,
            fallbacks="default",
        )
        with self.client.beta.messages.stream(**kwargs) as stream:
            for event in stream:
                if getattr(event, "type", None) != "content_block_delta":
                    continue
                d = event.delta
                if d.type == "text_delta" and d.text:
                    emit({"type": "text", "text": d.text})
                elif d.type == "thinking_delta" and d.thinking:
                    emit({"type": "status", "text": d.thinking})
            return stream.get_final_message()

    def _run_tool(self, block, emit: Emit) -> dict:
        args = block.input if isinstance(block.input, dict) else None
        spec = self._specs.get(block.name)
        emit({"type": "tool_call", "id": block.id, "name": block.name, "args": args})
        if args is None:
            result = {"error": json.dumps({"INVALID_JSON": str(block.input)})}
        elif spec is not None and (problems := validate(spec["input_schema"], args)):
            result = {"error": "Invalid arguments: " + "; ".join(problems)}
        else:
            result = self.api.call(block.name, args)
        is_error = "error" in result
        emit({"type": "tool_result", "id": block.id, "name": block.name, "result": result, "is_error": is_error})
        out = {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)}
        if is_error:
            out["is_error"] = True
        return out

    # ---- the loop ----------------------------------------------------------------
    def send(self, user_text: str, on_event: Emit | None = None) -> str:
        emit = on_event or _noop
        self._cancel = False
        start = len(self.messages)
        self.messages.append({"role": "user", "content": user_text})
        final_text, stop = "", "end_turn"
        json_retries = 0
        try:
            for _ in range(self.max_iterations):
                try:
                    response = self._request(emit)
                    json_retries = 0
                except ValueError:
                    # The SDK could not parse a streamed tool input at all: re-issue the turn (bounded).
                    json_retries += 1
                    if json_retries > 2:
                        raise
                    continue
                # Append the assistant turn unchanged (keeps thinking / fallback blocks valid).
                self.messages.append({"role": "assistant", "content": response.content})
                stop = response.stop_reason or "end_turn"
                texts = [b.text for b in response.content if getattr(b, "type", "") == "text"]
                if texts:
                    final_text = "".join(texts)
                if stop == "pause_turn":
                    continue
                if stop == "refusal":
                    if not final_text:
                        final_text = "Sorry, I can't help with that request."
                        emit({"type": "text", "text": final_text})
                    break
                tool_uses = [b for b in response.content if getattr(b, "type", "") == "tool_use"]
                if not tool_uses:
                    break
                if stop == "max_tokens":  # a truncated tool input parses as a valid partial object
                    results = [{"type": "tool_result", "tool_use_id": b.id, "is_error": True,
                                "content": "Tool input was cut off (max_tokens); not run. Retry with smaller input."}
                               for b in tool_uses]
                else:
                    results = []
                    for b in tool_uses:
                        if self._cancel:
                            results.append({"type": "tool_result", "tool_use_id": b.id, "is_error": True,
                                            "content": "Cancelled by the user."})
                        else:
                            results.append(self._run_tool(b, emit))
                self.messages.append({"role": "user", "content": results})
                if self._cancel:
                    stop = "cancelled"
                    break
            else:
                stop = "max_iterations"
                note = f"\n(Stopped after {self.max_iterations} steps; say 'continue' to keep going.)"
                final_text += note
                emit({"type": "text", "text": note})
        except Exception as e:  # API / network failure: drop this turn so history stays valid
            del self.messages[start:]
            msg = _describe_error(e)
            emit({"type": "error", "message": msg})
            emit({"type": "done", "text": "", "stop_reason": "error"})
            return ""
        emit({"type": "done", "text": final_text, "stop_reason": stop})
        return final_text


def _describe_error(e: Exception) -> str:
    try:
        import anthropic
    except Exception:  # pragma: no cover
        anthropic = None
    if anthropic is not None:
        if isinstance(e, anthropic.AuthenticationError):
            return "The Anthropic API key was rejected. Check it in Settings."
        if isinstance(e, anthropic.PermissionDeniedError):
            return "This API key does not have access to the model."
        if isinstance(e, anthropic.RateLimitError):
            return "Rate limited by the Anthropic API. Wait a moment and try again."
        if isinstance(e, anthropic.APIStatusError):
            return f"Anthropic API error ({e.status_code}): {e.message}"
        if isinstance(e, anthropic.APIConnectionError):
            return "Could not reach the Anthropic API. Check your internet connection."
    return f"{type(e).__name__}: {e}"


# =====================================================================================
# Offline fallback
# =====================================================================================
_NUM = r"(\d+(?:\.\d+)?)"
_TIME = r"(\d+:\d{1,2}(?:\.\d+)?|\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds?)?"
_KINDS = [
    (r"fade\s*(?:through|to)?\s*black|fadeblack|black", "fadeblack"),
    (r"slide\s*left|slideleft", "slideleft"),
    (r"slide\s*right|slideright", "slideright"),
    (r"slide\s*up|slideup", "slideup"),
    (r"slide\s*down|slidedown", "slidedown"),
    (r"slide", "slideleft"),
    (r"dissolve|cross\s*fade|crossfade|fade|smooth", "dissolve"),
]
HELP = ("I'm running offline, so I understand simple commands like: remove silences, add captions, "
        "make it vertical / square / 16:9, add dissolve transitions, add title \"Hello\" at 2s, "
        "split at 5s, cut 3s to 6s, duplicate 1s to 2s, speed up 2x, mute, volume 50%, black and white, "
        "fade in/out, undo, redo, save, export 720p. Set an Anthropic API key for full chat editing.")


def _t(s: str) -> float:
    if ":" in s:
        m, sec = s.split(":", 1)
        return int(m) * 60 + float(sec)
    return float(s)


class LocalAssistant:
    """Rule/regex intent parser for common commands, run through the same api tools."""

    name = "Offline"

    def __init__(self, api):
        self.api = api
        self.messages: list[dict] = []

    def reset(self):
        self.messages = []

    def cancel(self):
        pass

    # ---- plumbing ----
    def _call(self, name: str, args: dict, emit: Emit) -> dict:
        cid = f"local_{len(self.messages)}_{name}"
        emit({"type": "tool_call", "id": cid, "name": name, "args": args})
        r = self.api.call(name, args)
        emit({"type": "tool_result", "id": cid, "name": name, "result": r, "is_error": "error" in r})
        self.messages.append({"tool": name, "args": args, "result": r})
        return r

    def _media_clips(self, emit: Emit, main_only: bool = False) -> list[dict]:
        s = self._call("get_project_summary", {}, emit)
        vids = [t for t in s["tracks"] if t["kind"] == "video"]
        tracks = vids[-1:] if main_only else s["tracks"]
        sel = getattr(getattr(self.api, "editor", None), "selection", set()) or set()
        clips = [c for t in tracks for c in t["clips"] if c.get("asset_id")]
        picked = [c for c in clips if c["id"] in sel]
        return picked or clips

    def _props_all(self, props: dict, emit: Emit, main_only: bool = False) -> tuple[int, str | None]:
        clips = self._media_clips(emit, main_only)
        n, err = 0, None
        for c in clips:
            r = self._call("set_clip_properties", {"clip_id": c["id"], "properties": props}, emit)
            if "error" in r:
                err = r["error"]
            else:
                n += 1
        return n, err

    # ---- intents ----
    def _handle(self, text: str, emit: Emit) -> str:  # noqa: C901
        q = text.strip()
        low = q.lower().strip(" .!?")

        def err(r):
            return r.get("error") if isinstance(r, dict) else None

        if not low or re.fullmatch(r"(help|\?|what can you do.*|commands)", low):
            return HELP

        if re.fullmatch(r"(please )?(undo|revert|go back|undo (that|it|the last (edit|change)))( please)?", low):
            r = self._call("undo", {}, emit)
            return "Undone." if r.get("ok") else "There's nothing to undo."
        if re.fullmatch(r"(please )?(redo|redo (that|it))( please)?", low):
            r = self._call("redo", {}, emit)
            return "Redone." if r.get("ok") else "There's nothing to redo."

        if re.search(r"\b(silence|silences|silent|pauses|dead air)\b", low):
            m = re.search(r"(-?\d+(?:\.\d+)?)\s*db", low)
            args = {"threshold_db": -abs(float(m.group(1)))} if m else {}
            r = self._call("remove_silences", args, emit)
            if err(r):
                return f"Couldn't remove silences: {err(r)}"
            if not r["removed"]:
                return "I didn't find any silences long enough to remove."
            return (f"Removed {r['removed']} silent stretch{'es' if r['removed'] != 1 else ''} "
                    f"({r['removed_seconds']:.1f}s). The video is now {r['duration']:.1f}s.")

        if re.search(r"\b(captions?|subtitles?|subs|transcribe|transcript)\b", low):
            args: dict[str, Any] = {}
            if re.search(r"\b(big|large|bold|tiktok style)\b", low):
                args["style"] = "big"
            for model in ("tiny", "small", "base"):
                if re.search(rf"\b{model}\b", low):
                    args["model"] = model
            m = re.search(r"\b(?:in|language)\s+(english|spanish|french|german|italian|portuguese|japanese|hindi|tamil)\b", low)
            if m:
                args["language"] = {"english": "en", "spanish": "es", "french": "fr", "german": "de",
                                    "italian": "it", "portuguese": "pt", "japanese": "ja", "hindi": "hi",
                                    "tamil": "ta"}[m.group(1)]
            r = self._call("auto_captions", args, emit)
            if err(r):
                return f"Couldn't add captions: {err(r)}"
            return f"Added {r['captions']} captions on a new Captions track." if r["captions"] else r.get("note", "No speech found.")

        canvas = None
        if re.search(r"\b(vertical|portrait|tiktok|reels?|shorts|9\s*[:x/]\s*16|story|stories)\b", low):
            canvas = "9:16"
        elif re.search(r"\b(square|1\s*[:x/]\s*1)\b", low):
            canvas = "1:1"
        elif re.search(r"\b4\s*[:x/]\s*5\b|instagram (feed|post)", low):
            canvas = "4:5"
        elif re.search(r"\b(horizontal|landscape|widescreen|youtube|16\s*[:x/]\s*9)\b", low):
            canvas = "16:9"
        if canvas and not re.search(r"\bexport\b", low):
            r = self._call("set_canvas", {"preset": canvas}, emit)
            return err(r) or f"Canvas set to {canvas} ({r['width']}x{r['height']})."

        if re.search(r"\btransitions?\b", low):
            kind = next((k for pat, k in _KINDS if re.search(pat, low)), "dissolve")
            m = re.search(_NUM + r"\s*(?:s|sec|seconds?)\b", low)
            dur = float(m.group(1)) if m else 0.5
            if re.search(r"\b(remove|delete|no|clear)\b", low):
                return "To remove transitions offline, use undo, or select a cut and press Delete on its transition."
            r = self._call("transitions_everywhere", {"kind": kind, "duration": min(max(dur, 0.05), 5)}, emit)
            if err(r):
                return f"Couldn't add transitions: {err(r)}"
            if not r["transitions"]:
                return "There are no cuts between clips to put a transition on yet."
            return f"Added {r['transitions']} {kind} transition{'s' if r['transitions'] != 1 else ''} ({dur:g}s each)."

        m = re.match(r"(?:add|put|insert|create|make)?\s*(?:a\s+)?(?:title|text|caption|heading)\s*(?:that says|saying|reading|:)?\s*(.+)", q, re.I)
        if m and not re.search(r"\b(captions|subtitles)\b", low):
            rest = m.group(1).strip()
            qm = re.search(r"[\"“'‘](.+?)[\"”'’]", rest)
            title = qm.group(1) if qm else re.split(r"\s+(?:at|from|for)\s+\d", rest)[0].strip(" .")
            args = {"text": title, "style": {"preset": "title"}}
            sm = re.search(r"\b(?:at|from)\s+" + _TIME, rest)
            if sm:
                args["start"] = _t(sm.group(1))
            dm = re.search(r"\bfor\s+" + _TIME, rest)
            if dm:
                args["duration"] = max(_t(dm.group(1)), 0.1)
            if not title:
                return "What should the title say? Try: add title \"My trip\"."
            r = self._call("add_text", args, emit)
            if err(r):
                return f"Couldn't add the title: {err(r)}"
            c = r["clip"]
            return f"Added the title \"{title}\" from {c['start']:g}s to {c['end']:g}s."

        m = re.search(r"\bsplit\b.*?\b(?:at|@)?\s*" + _TIME, low)
        if m:
            r = self._call("split_at", {"t": _t(m.group(1))}, emit)
            return err(r) or f"Split at {_t(m.group(1)):g}s."

        m = re.search(r"\b(cut|remove|delete|trim out|drop)\b.*?(?:from\s+)?" + _TIME + r"\s*(?:to|-|–|until|and)\s*" + _TIME, low)
        if m:
            a, b = sorted((_t(m.group(2)), _t(m.group(3))))
            r = self._call("remove_range", {"start": a, "end": b}, emit)
            return err(r) or f"Cut out {a:g}s–{b:g}s. The video is now {r['duration']:.1f}s."

        m = re.search(r"\b(duplicate|repeat|copy)\b.*?(?:from\s+)?" + _TIME + r"\s*(?:to|-|–|until|and)\s*" + _TIME, low)
        if m:
            a, b = sorted((_t(m.group(2)), _t(m.group(3))))
            r = self._call("duplicate_range", {"start": a, "end": b}, emit)
            return err(r) or f"Duplicated {a:g}s–{b:g}s right after itself."

        m = re.search(r"\b(?:delete|remove)\s+(?:the\s+)?(first|last|second|third)\s+clip\b", low)
        if m:
            clips = sorted(self._media_clips(emit, main_only=True), key=lambda c: c["start"])
            idx = {"first": 0, "second": 1, "third": 2, "last": -1}[m.group(1)]
            if not clips or (idx >= 0 and idx >= len(clips)):
                return "There's no such clip on the main track."
            r = self._call("delete_clips", {"clip_ids": [clips[idx]["id"]], "ripple": True}, emit)
            return err(r) or f"Deleted the {m.group(1)} clip ({clips[idx]['name']})."

        sp = None
        m = re.search(r"\b(?:speed|faster|slower|speed up|slow down|slow-mo|slow motion)\b.*?" + _NUM + r"\s*x\b", low) \
            or re.search(_NUM + r"\s*x\s*(?:speed|faster)", low)
        if m:
            sp = float(m.group(1))
            if re.search(r"\bslow", low) and sp > 1:
                sp = 1 / sp
        elif re.search(r"\b(slow motion|slow-mo|slow down|slower)\b", low):
            sp = 0.5
        elif re.search(r"\b(speed up|faster|fast forward)\b", low):
            sp = 2.0
        if sp is not None:
            sp = min(max(sp, 0.25), 4.0)
            n, e = self._props_all({"speed": sp}, emit, main_only=True)
            return e if not n and e else f"Set speed to {sp:g}x on {n} clip{'s' if n != 1 else ''}."

        if re.search(r"\b(unmute)\b", low):
            n, e = self._props_all({"volume": 1.0}, emit)
            return e if not n and e else f"Unmuted {n} clip{'s' if n != 1 else ''}."
        if re.search(r"\b(mute|silence the audio|no sound|remove (the )?audio)\b", low):
            n, e = self._props_all({"volume": 0.0}, emit)
            return e if not n and e else f"Muted {n} clip{'s' if n != 1 else ''}."
        m = re.search(r"\bvolume\b.*?" + _NUM + r"\s*%", low) or re.search(_NUM + r"\s*%\s*volume", low)
        if m:
            v = min(max(float(m.group(1)) / 100, 0.0), 2.0)
            n, e = self._props_all({"volume": v}, emit)
            return e if not n and e else f"Volume set to {v * 100:g}% on {n} clip{'s' if n != 1 else ''}."

        if re.search(r"\b(black and white|black & white|b&w|grayscale|greyscale|monochrome)\b", low):
            n, e = self._props_all({"saturation": 0.0}, emit, main_only=True)
            return e if not n and e else f"Made {n} clip{'s' if n != 1 else ''} black and white."
        if re.search(r"\b(brighter|brighten|darker|darken)\b", low):
            b = 0.15 if re.search(r"bright", low) else -0.15
            n, e = self._props_all({"brightness": b}, emit, main_only=True)
            return e if not n and e else f"{'Brightened' if b > 0 else 'Darkened'} {n} clip{'s' if n != 1 else ''}."

        if re.search(r"\bfade\b", low) and re.search(r"\b(in|out|start|end|both)\b", low):
            clips = sorted(self._media_clips(emit, main_only=True), key=lambda c: c["start"])
            if not clips:
                return "There are no clips to fade."
            m = re.search(_NUM + r"\s*(?:s|sec|seconds?)\b", low)
            d = float(m.group(1)) if m else 1.0
            done = []
            if re.search(r"\b(in|start|both)\b", low):
                r = self._call("set_clip_properties", {"clip_id": clips[0]["id"], "properties": {"fade_in": d}}, emit)
                if not err(r):
                    done.append("in")
            if re.search(r"\b(out|end|both)\b", low):
                r = self._call("set_clip_properties", {"clip_id": clips[-1]["id"], "properties": {"fade_out": d}}, emit)
                if not err(r):
                    done.append("out")
            return f"Added a {d:g}s fade {' and '.join(done)}." if done else "Couldn't add the fade."

        if re.match(r"(please )?(export|render)\b", low):
            res = next((r for r in ("2160p", "1440p", "1080p", "720p", "480p") if r in low), None)
            if re.search(r"\b4k\b", low):
                res = "2160p"
            pm = re.search(r"\b(?:to|as)\s+[\"']?([^\s\"']+\.(?:mp4|mov|mkv))", q, re.I)
            path = pm.group(1) if pm else None
            if path is None:
                ed = getattr(self.api, "editor", None)
                base = (ed.path[:-len(".vedit.json")] if ed and ed.path and ed.path.endswith(".vedit.json")
                        else os.path.splitext(ed.path)[0] if ed and ed.path else os.path.join(os.getcwd(), "export"))
                path = base + ".mp4"
            args = {"path": path}
            if res:
                args["resolution"] = res
            if re.search(r"\b(draft|quick|fast)\b", low):
                args["quality"] = "draft"
            elif re.search(r"\b(high|best)\b", low):
                args["quality"] = "high"
            r = self._call("export_video", args, emit)
            return f"Export failed: {err(r)}" if err(r) else f"Exported {r['path']} ({r['size_mb']} MB)."

        if re.match(r"(please )?save\b", low):
            pm = re.search(r"\b(?:to|as)\s+[\"']?([^\s\"']+)", q, re.I)
            args = {"path": pm.group(1)} if pm else {}
            r = self._call("save_project", args, emit)
            return err(r) or f"Saved to {r['path']}."

        if re.search(r"\b(summary|what'?s in|describe|status|how long|duration)\b", low):
            s = self._call("get_project_summary", {}, emit)
            n = sum(len(t["clips"]) for t in s["tracks"])
            return (f"The project is {s['duration']:.1f}s long at {s['canvas']['width']}x{s['canvas']['height']} "
                    f"({s['canvas']['aspect']}), with {n} clip{'s' if n != 1 else ''} on {len(s['tracks'])} tracks.")

        return "Sorry, I didn't understand that. " + HELP

    def send(self, user_text: str, on_event: Emit | None = None) -> str:
        emit = on_event or _noop
        try:
            reply = self._handle(user_text, emit)
        except Exception as e:
            emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
            emit({"type": "done", "text": "", "stop_reason": "error"})
            return ""
        emit({"type": "text", "text": reply})
        emit({"type": "done", "text": reply, "stop_reason": "end_turn"})
        return reply


def make_assistant(api, api_key: str | None = None, client=None):
    """Claude assistant when an API key (argument or ANTHROPIC_API_KEY) or a client is available,
    otherwise the offline LocalAssistant."""
    if client is not None or has_api_key(api_key):
        try:
            return Assistant(api, client=client, api_key=api_key)
        except Exception:  # SDK missing or misconfigured: stay usable offline
            pass
    return LocalAssistant(api)
