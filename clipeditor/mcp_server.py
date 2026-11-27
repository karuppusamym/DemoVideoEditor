"""MCP stdio server exposing the EditorAPI tools, so external agents (Claude Desktop / Claude Code)
can edit ClipEditor projects.

    python -m clipeditor.mcp_server [project.vedit.json]

Keeps one Editor (+ project path) in memory. Extra tools: open_project(path), new_project().
Edits are not written to disk until save_project is called.
"""
from __future__ import annotations

import json
import os
import sys

import anyio
import mcp_types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .api import EditorAPI, validate
from .editor import Editor
from .media import AudioCache

INSTRUCTIONS = (
    "ClipEditor video editing tools. Start with open_project (or new_project + import_media), then "
    "get_project_summary to see the active timeline, tracks and clip ids. Use list_timelines and "
    "switch_timeline to work across edits. Times are timeline seconds. Every edit is one undo "
    "step (undo/redo tools). Call save_project to write changes; export_video renders an MP4."
)

_EXTRA = [
    {"name": "open_project",
     "description": "Open a ClipEditor project file (.vedit.json). Replaces the project in memory "
                    "(unsaved changes are lost).",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"],
                      "additionalProperties": False}},
    {"name": "new_project",
     "description": "Start a new empty project in memory (unsaved changes are lost).",
     "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
]


class ClipEditorMCP:
    """Holds the Editor and dispatches tool calls (sync; usable without MCP for tests)."""

    def __init__(self, project_path: str | None = None):
        self.editor = Editor()
        self.api = EditorAPI(self.editor, AudioCache())
        if project_path:
            self.editor.open(os.path.abspath(project_path))

    def specs(self) -> list[dict]:
        return _EXTRA + self.api.tool_specs()

    def call(self, name: str, args: dict | None) -> dict:
        args = args or {}
        if name == "open_project":
            problems = validate(_EXTRA[0]["input_schema"], args)
            if problems:
                return {"error": "; ".join(problems)}
            path = os.path.abspath(os.path.expanduser(args["path"]))
            if not os.path.isfile(path):
                return {"error": f"File not found: {path}"}
            try:
                self.editor.open(path)
            except Exception as e:
                return {"error": f"Could not open project: {e}"}
            return {"ok": True, "path": path, "summary": self.api.call("get_project_summary")}
        if name == "new_project":
            self.editor.new_project()
            return {"ok": True}
        return self.api.call(name, args)


def build_server(state: ClipEditorMCP | None = None) -> Server:
    state = state or ClipEditorMCP()
    limiter: list = []  # created lazily inside the running event loop

    async def on_list_tools(ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(tools=[
            types.Tool(name=s["name"], description=s["description"], input_schema=s["input_schema"])
            for s in state.specs()])

    async def on_call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        # Tools are synchronous and may be slow (export, captions): run them off the event loop.
        # A limiter of 1 serialises calls, so the single Editor is never mutated concurrently.
        if not limiter:
            limiter.append(anyio.CapacityLimiter(1))
        result = await anyio.to_thread.run_sync(state.call, params.name, dict(params.arguments or {}),
                                                limiter=limiter[0])
        return types.CallToolResult(content=[types.TextContent(type="text", text=json.dumps(result))],
                                    structured_content=result, is_error="error" in result)

    server = Server("clipeditor", version="1.0", instructions=INSTRUCTIONS,
                    on_list_tools=on_list_tools, on_call_tool=on_call_tool)
    server.state = state  # type: ignore[attr-defined]
    return server


async def _serve(project_path: str | None):
    server = build_server(ClipEditorMCP(project_path))
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    anyio.run(_serve, argv[0] if argv else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
