"""Entry point.

    python -m clipeditor                       # open the editor
    python -m clipeditor clip1.mp4 song.mp3    # open with media imported
    python -m clipeditor project.clipeditor.json    # open a project
    python -m clipeditor export project.clipeditor.json out.mp4 [--crf 20]
"""
from __future__ import annotations

import argparse
import sys


def cli_export(argv):
    from .export_render import export_video
    from .model import Project

    ap = argparse.ArgumentParser(prog="clipeditor export")
    ap.add_argument("project")
    ap.add_argument("output")
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--timeline", help="timeline name or ID (defaults to the saved active timeline)")
    a = ap.parse_args(argv)
    proj = Project.load(a.project)
    if a.timeline is not None:
        timeline = proj.find_timeline(a.timeline)
        if timeline is None:
            ap.error(f"Timeline '{a.timeline}' was not found. Available: "
                     + ", ".join(t.name for t in proj.timelines.values()))
        proj.active_timeline_id = timeline.id

    def prog(p):
        print(f"\rExporting… {p * 100:5.1f}%", end="", flush=True)

    export_video(proj, a.output, progress=prog, crf=a.crf, preset=a.preset)
    print(f"\nWrote {a.output}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "export":
        return cli_export(argv[1:])
    if argv and argv[0] == "mcp":  # MCP server for AI agents (Claude Desktop / Claude Code)
        from .mcp_server import main as mcp_main

        return mcp_main(argv[1:])

    from PySide6.QtWidgets import QApplication

    from .fonts import prepare_qt_fonts
    from .ui.main_window import MainWindow, apply_dark_theme

    prepare_qt_fonts()
    app = QApplication(sys.argv[:1])
    app.setApplicationName("ClipEditor")
    apply_dark_theme(app)
    w = MainWindow()
    projects = [p for p in argv if p.endswith(".json")]
    media = [p for p in argv if not p.endswith(".json")]
    if projects:
        w.open_project(projects[0])
    if media:
        w.import_paths(media)
    w.show()
    if not projects and not media:
        from PySide6.QtCore import QTimer

        QTimer.singleShot(300, w.offer_recovery)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
