# ClipEditor vs. other video editors

This is an honest snapshot (v0.5). ✅ means built and tested, 🟡 means partial, and ❌ means not yet built.
The entries for other products come from their public feature lists, so check them before you rely on them.

## Feature by feature, vs. Concat

| Area | Concat | ClipEditor |
|---|---|---|
| Multi-track timeline, split/trim/ripple, snapping | ✅ | ✅ also range editing (I/O), frame delete/duplicate/freeze, copy/paste |
| Magnetic timeline option | ✅ | 🟡 ripple delete/remove-range; no always-on magnetic mode yet |
| Multiple timelines per project | ✅ | ❌ |
| Keyframes + curve editor | ✅ | ✅ keyframes on position/scale/rotation/opacity/volume, 7 easings · ❌ graphical curve editor |
| In / Out / Loop animations | ✅ | ✅ 16 in, 16 out, 10 loop |
| Effects / filters / transitions | 100+ | ✅ 32 effects + 14 filter looks + 6 transitions (preview cards on your own frame) |
| Sound effects | 25 | ✅ 28, synthesised in-app (no licensing questions) |
| Titles & shapes | ✅ | ✅ 10 text styles, 22 shapes, 24 colour emoji stickers, gradients/outline/shadow |
| Blend modes, crop, flip | ✅ | ✅ 13 blend modes, crop, mirror/flip, rotate |
| Masks + chroma key | ✅ | ✅ rect/ellipse/linear/mirror masks with feather + invert; green-screen key with spill |
| Speed, ramps/curves, freeze, reverse | ✅ | ✅ constant speed (pitch-correct), freeze, reverse, optical-flow slow-mo · ❌ speed ramps/curves |
| Auto captions (local) | ✅ | ✅ Whisper, any language, 2 styles |
| Voice isolation | ✅ | 🟡 AI noise removal (RNNoise) + voice enhance; no stem separation yet |
| Text-to-speech + voice cloning | ✅ | 🟡 TTS with Windows voices (Piper optional) · ❌ voice cloning |
| Background removal | ✅ | ✅ rembg (people / general models), video + images |
| Enhance / stabilise | ✅ | ✅ FFmpeg-based enhance/upscale 2x, stabilise · ❌ AI super-resolution |
| Voiceover recording + voice effects | ✅ | ✅ record over playback; 20 voice/music effects (chipmunk, robot, telephone, 8D…) |
| Export H.264/HEVC/AV1, GPU encoders, 10-bit, 4K60 | ✅ | ✅ GPU encoders are auto-detected (Intel QSV found on this PC) |
| Proxies for smooth 4K | ✅ | ✅ automatic 540p proxies |
| MCP server / scripting API | ✅ | ✅ 25-tool API, MCP server, CLI export |
| **Chat-to-edit assistant inside the app** | ❌ | ✅ Claude with an API key, offline helper without one |
| **Command palette (Ctrl+K) over every action/effect/sticker/sound** | ❌ | ✅ |
| **Social safe-zone overlay** | ❌ | ✅ |
| **One-click silence removal** | ❌ | ✅ |
| Platforms | macOS, Windows, Linux, Android, iOS | Windows tested; Python + Qt should run on macOS/Linux (untested) |
| Languages | 27 | English only |
| Licence | AGPL-3.0 | yours (all dependencies are LGPL/MIT/BSD/Apache) |

## Positioning against the wider market

| | ClipEditor | CapCut desktop | DaVinci Resolve (free) | Premiere Pro | Shotcut / Kdenlive / OpenShot | Clipchamp |
|---|---|---|---|---|---|---|
| Price | free / yours | freemium, Pro subscription | free (Studio is paid) | subscription | free (GPL) | freemium |
| Account / upload needed | no | yes | no | Adobe account | no | Microsoft account |
| Watermark-free export | ✅ | some Pro features locked | ✅ | ✅ | ✅ | 4K/premium locked |
| Learning curve | low (CapCut-style layout) | low | high | high | medium | low |
| Local AI captions | ✅ | ✅ (cloud-assisted) | ✅ Studio | ✅ | 🟡 Kdenlive (Whisper) | ✅ (cloud) |
| Chat-to-edit with an LLM | ✅ | ❌ | ❌ | 🟡 (preview features) | ❌ | ❌ |
| Scriptable by AI agents (MCP) | ✅ | ❌ | 🟡 (scripting API) | 🟡 (UXP/ExtendScript) | 🟡 (MLT XML) | ❌ |
| Pro colour, Fusion VFX, Fairlight audio | ❌ | ❌ | ✅ | ✅ | 🟡 | ❌ |
| Performance on long 4K projects | 🟡 Python render engine | ✅ | ✅ GPU | ✅ GPU | ✅ | 🟡 |

**Where ClipEditor is better today:**
- There is no account, no watermark, no upload and no licence lock-in.
- AI is built in: you can edit by chatting, external AI agents can drive it through MCP, and captions, background removal and noise removal all run locally.
- The command palette, the safe-zone overlay and one-click silence removal are things most of these editors don't have.

**Where it is clearly behind:**
- Graphical keyframe curves, speed ramps and multiple timelines are missing.
- Voice cloning and stem separation are missing.
- There are no mobile apps and the UI is English only.
- Export of heavily effected timelines is slower than in GPU-native engines, because the render engine runs in Python on the CPU.

## Next milestones (in order)
1. Speed ramps + a graphical keyframe curve editor
2. Multiple timelines per project, plus nested sequences
3. Stem separation (Demucs) and voice cloning (an open TTS such as XTTS/F5), as optional downloads
4. A GPU/parallel render engine (multiprocess frame rendering, then an OpenGL compositor)
5. A transition pack (wipes, zoom, blur, glitch: 30+) and an effect pack (to 100+)
6. Translations, macOS/Linux packaging, installer
