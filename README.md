# ClipEditor

ClipEditor is a desktop video editor built with Python, PySide6, PyAV and FFmpeg and tested on Windows. It combines a multitrack timeline, live preview, effects, local AI tools and optional AI-agent controls.

## Run

On Windows with Python 3.13, clone the repository and install the dependencies in a virtual environment.

```powershell
git clone https://github.com/karuppusamym/DemoVideoEditor.git
cd DemoVideoEditor
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\run.bat
```

Drag a media file into the editor to import it. You can also pass your own media files to the launcher.

```powershell
.\run.bat "C:\path\to\your-video.mp4"
```

Open a saved project by passing its `.clipeditor.json` file. To export from the command line, run the following command.

```powershell
.\run.bat export project.clipeditor.json out.mp4 --crf 20
```

## Workspace

- **Left panel:** Media, Audio, Text, Stickers, Effects, Filters, Transitions, Animation and AI Magic. Effect and filter cards preview on the current frame.
- **Center:** video preview and multitrack timeline.
- **Right inspector:** Text, Shape, Video, Adjust, Mask & Key, Animation, Speed and Audio tabs. Animatable controls have keyframe buttons.
- **Timeline toolbar:** split, delete, duplicate, freeze, mirror, rotate, reverse, transition, text, frame tools and zoom.

The editor also includes undo/redo, range editing, autosave and crash recovery, drag-and-drop import, proxy generation for high-resolution media, and a Ctrl+K command palette.

## Editing and effects

- Edit and arrange clips across multiple tracks; trim, split, move, duplicate, copy/paste, ripple-delete and detach audio.
- Add titles, shapes and emoji stickers. The galleries include 46 shape and emoji presets.
- Use 32 video effects, 14 filter looks and 6 transitions. Masks, chroma key, 13 blend modes, crop, flip and rotation are also available.
- Animate position, scale, rotation, opacity and volume with keyframes and easing. There are 42 in, out and loop animation presets.
- Adjust colour, speed, fades and audio. The built-in library contains 28 sound effects and 20 voice/music effects; voiceover recording and offline text-to-speech are included.
- Enable social safe-zone guides for TikTok/Reels framing.

## AI tools

Local tools include Whisper auto-captions, silence removal, background removal, reverse, stabilization, noise removal, enhancement/upscaling, smooth slow motion and automatic 9:16 reframing. Availability and processing time depend on optional models and local hardware.

The in-app chat assistant uses Claude when `ANTHROPIC_API_KEY` is set or you supply a key in the app. Without a key it runs the built-in offline helper. When using Claude, prompts and editing-tool results are sent to Anthropic. Background removal requires the optional `rembg[cpu]` package and downloads a model on first use. Whisper captions also download a model on first use.

## Export

Export H.264, HEVC or AV1 at up to 4K and 60 fps, with optional 10-bit output. Compatible Intel, NVIDIA or AMD hardware encoders are detected for accelerated export; software encoding remains available.

## AI-agent interface

Run the Model Context Protocol (MCP) server with:

```powershell
.\.venv\Scripts\python.exe -m clipeditor mcp
```

Configure your MCP client to launch that command from the repository directory. Project edits remain in memory until saved.

## Project layout

```text
clipeditor/
  model.py       Project, track, clip and media-asset data
  editor.py      Editing operations and undo/redo
  media.py       Media probing, decoding, waveforms and thumbnails
  render.py      Preview and frame compositor
  export.py      FFmpeg export
  api.py         Shared editing API for the assistant and MCP server
  ui/            PySide6 editor interface
  ai/            Assistant and local AI tools
tests/           pytest suite
```

## Contributing

Fork the repository to make changes and open a pull request. See the [contributing guide](CONTRIBUTING.md) for setup and testing, the [code of conduct](CODE_OF_CONDUCT.md) for community expectations, and the [security policy](SECURITY.md) for private vulnerability reporting.

ClipEditor source code is available under the [MIT License](LICENSE).

## Known gaps

ClipEditor does not yet include speed ramps, a graphical keyframe curve editor, multiple timelines per project, voice cloning, mobile apps or translations. See [COMPARISON.md](COMPARISON.md) for a feature comparison and the current roadmap.
