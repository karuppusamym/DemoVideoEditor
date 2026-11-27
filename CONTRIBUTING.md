# Contributing to ClipEditor

Bug fixes, documentation improvements, and new editing features are welcome. Open a GitHub issue before starting a large change so maintainers can discuss the scope with you.

1. Fork the repository on GitHub and clone your fork.
2. Follow the Windows setup in the [README](README.md#run), using your fork's clone URL.
3. Create a branch with `git switch -c describe-your-change`.
4. Make the change and run the tests below. If you change the editor UI, exercise the affected workflow in the app.
5. Push your branch to your fork and open a pull request against `main`. Summarize the change, explain how you tested it, and link any related issue.

## Run tests

From the repository root, run these commands in PowerShell.

```powershell
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest -q -m 'not model'
```

Pytest generates the video and audio fixtures in a temporary directory. Run `pytest -m model` with the virtual environment active to try the optional Whisper and background-removal integration tests. These tests may download models. The caption test also needs an English text-to-speech voice, and background removal needs `rembg[cpu]`.

## Build the Windows ZIP

Run `.\.venv\Scripts\python.exe portable\build.py --ref HEAD` on x64 Windows with Python 3.13. The builder packages the selected Git revision with the Python runtime and writes the ZIP and its SHA-256 checksum to `dist/`. Extract the ZIP into a fresh folder and test first launch and export before uploading it to a GitHub release.

## Report a bug

Include your OS and Python version, the commit or branch, steps to reproduce, and the expected and actual behavior. Include a short log or a synthetic media sample if it helps reproduce the problem. Remove credentials, private file paths, and personal media before posting. Report potential security issues privately using the [security policy](SECURITY.md).
