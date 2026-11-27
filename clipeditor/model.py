"""Project data model: assets, tracks, clips. Pure Python, no Qt."""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .animate import shift_keyframes
from .retime import curve, validate_points

MIN_CLIP = 1 / 30  # shortest clip allowed, seconds


def new_id() -> str:
    return uuid.uuid4().hex[:8]


@dataclass
class MediaAsset:
    id: str
    path: str
    kind: str  # "video" | "audio" | "image"
    duration: float
    width: int = 0
    height: int = 0
    fps: float = 0.0
    has_video: bool = False
    has_audio: bool = False
    alpha: bool = False  # video/image carries transparency (decode with VideoReader.rgba_at)
    proxy_path: str = ""  # low-res copy used for smooth preview (clipeditor.proxy)
    origin: str = ""  # how it was made, e.g. "bg_removed:<asset_id>", "reversed:<id>", "tts", "voiceover"

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


@dataclass
class Clip:
    """A piece of media (or a text title) placed on a track.

    Timeline length depends on the playback speed across the source range.
    A text clip has asset_id == "" and `text` set; its in/out are 0..length.
    """
    id: str
    asset_id: str
    start: float  # position on the timeline (s)
    in_point: float  # source in (s)
    out_point: float  # source out (s)
    volume: float = 1.0  # 0..2
    opacity: float = 1.0  # 0..1
    speed: float = 1.0  # 0.25..4 (video + audio)
    fade_in: float = 0.0  # seconds, opacity for video/text, gain for audio
    fade_out: float = 0.0
    # transform (video/image/text): scale relative to "fit to canvas";
    # pos_x/pos_y offset of the clip centre from canvas centre, as a fraction of canvas W/H
    scale: float = 1.0
    pos_x: float = 0.0
    pos_y: float = 0.0
    # color adjust (FFmpeg `eq` semantics)
    brightness: float = 0.0  # -1..1 (additive)
    contrast: float = 1.0  # 0..2
    saturation: float = 1.0  # 0..3
    # text title
    text: str | None = None
    font_family: str = "Segoe UI"
    font_size: int = 72  # px at project height
    color: str = "#ffffff"
    bg_color: str = ""  # "" = no box, else "#rrggbbaa" or "#rrggbb"
    bold: bool = True
    # transition from the previous clip on the same track: this clip overlaps it by `transition` s
    transition: float = 0.0
    transition_kind: str = "dissolve"

    # ---- v0.5 contract (see clipeditor/pipeline docs in each module) ----
    rotation: float = 0.0  # degrees, clockwise
    flip_h: bool = False
    flip_v: bool = False
    crop: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])  # left, top, right, bottom fractions
    blend_mode: str = "normal"  # normal|multiply|screen|overlay|darken|lighten|color_dodge|color_burn|
    #                              hard_light|soft_light|difference|exclusion|add
    # keyframes: prop -> [[t, value, easing], ...]; t = seconds from clip start (timeline time),
    # props: scale, pos_x, pos_y, rotation, opacity, volume; easing: linear|ease_in|ease_out|ease|hold
    keyframes: dict = field(default_factory=dict)
    anim_in: str = ""  # animation preset name (clipeditor.animate.ANIMATIONS["in"])
    anim_in_dur: float = 0.5
    anim_out: str = ""
    anim_out_dur: float = 0.5
    anim_loop: str = ""
    anim_loop_speed: float = 1.0
    effects: list = field(default_factory=list)  # [{"type": str, "params": {..}, "enabled": bool}]
    chroma: dict = field(default_factory=dict)  # {} = off, else {"color": "#00ff00", "similarity": .3, "smooth": .1, "spill": .5}
    masks: list = field(default_factory=list)  # [{"shape": rect|ellipse|linear|mirror, "x","y","w","h","angle","feather","invert"}]
    audio_fx: list = field(default_factory=list)  # [{"type": str, "params": {..}}] (clipeditor.audio_fx)
    shape: dict | None = None  # shape clip (asset_id ""), see clipeditor.shapes
    speed_ramp: tuple[tuple[float, float], ...] = ()  # (absolute source seconds, playback rate)
    presentation_in: float = 0.0  # elapsed effect time retained after a frame cut
    presentation_out: float = 0.0
    sequence_id: str = ""  # linked source timeline, when this is a nested sequence
    # pro colour grading (clipeditor.grade): {} = off. Keys: lift/gamma/gain [r,g,b] wheels,
    # temperature, tint, exposure, highlights, shadows, vibrance, curves {"master"|"r"|"g"|"b": [[x,y],..]},
    # hsl {hue_name: {"h","s","l"}}, lut (path to .cube), lut_mix
    grade: dict = field(default_factory=dict)

    def __post_init__(self):
        self.speed_ramp = validate_points(self.speed_ramp) if self.speed_ramp else ()

    @property
    def is_text(self) -> bool:
        return self.text is not None

    @property
    def is_shape(self) -> bool:
        return self.shape is not None

    @property
    def is_generated(self) -> bool:
        """Text or shape: rendered by ClipEditor, no media asset."""
        return self.text is not None or self.shape is not None

    @property
    def duration(self) -> float:
        return self.time_between(self.in_point, self.out_point)

    @property
    def end(self) -> float:
        return self.start + self.duration

    @property
    def presentation_start(self) -> float:
        return self.start - self.presentation_in

    @property
    def presentation_end(self) -> float:
        return self.end + self.presentation_out

    def source_time(self, t: float) -> float:
        return self.source_after(self.in_point, t - self.start)

    def time_between(self, first: float, last: float) -> float:
        if not self.speed_ramp:
            return (last - first) / self.speed
        ramp = curve(self.speed_ramp)
        return ramp.timeline_at(last) - ramp.timeline_at(first)

    def source_after(self, start: float, seconds: float) -> float:
        if not self.speed_ramp:
            return start + seconds * self.speed
        ramp = curve(self.speed_ramp)
        return ramp.source_at(ramp.timeline_at(start) + seconds)

    def fade_gain(self, t: float) -> float:
        """0..1 multiplier from fades at timeline time t."""
        g = 1.0
        if self.fade_in > 0:
            g = min(g, max((t - self.presentation_start) / self.fade_in, 0.0))
        if self.fade_out > 0:
            g = min(g, max((self.presentation_end - t) / self.fade_out, 0.0))
        return min(g, 1.0)


@dataclass
class Track:
    id: str
    kind: str  # "video" | "audio"
    name: str
    clips: list[Clip] = field(default_factory=list)
    muted: bool = False
    hidden: bool = False

    def sorted_clips(self) -> list[Clip]:
        return sorted(self.clips, key=lambda c: c.start)

    def clip_at(self, t: float) -> Clip | None:
        for c in self.clips:
            if c.start <= t < c.end:
                return c
        return None

    def clips_at(self, t: float) -> list[Clip]:
        """All clips under t (more than one during a transition), bottom-most first."""
        return sorted((c for c in self.clips if c.start <= t < c.end), key=lambda c: c.start)

    def accepts(self, asset: MediaAsset | None) -> bool:
        if asset is None:  # text clip
            return self.kind == "video"
        if self.kind == "video":
            return asset.has_video or asset.kind == "image"
        return asset.has_audio

    def clear_range(self, start: float, end: float, keep_id: str | None = None,
                    minimum_duration: float = MIN_CLIP, preserve_presentation: bool = False) -> None:
        """Overwrite edit: trim/split/remove other clips overlapping [start, end).

        If `keep_id` has a transition, its overlap with the previous clip is allowed."""
        if keep_id is not None:
            k = next((c for c in self.clips if c.id == keep_id), None)
            if k is not None and k.transition > 0:
                start = min(start + k.transition, end)
        result: list[Clip] = []
        for c in self.clips:
            if c.id == keep_id or c.end <= start + 1e-9 or c.start >= end - 1e-9:
                result.append(c)
                continue
            if c.start < start:  # keep head
                head = Clip(**asdict(c))
                head.out_point = c.source_time(start)
                if preserve_presentation:
                    head.presentation_out += c.end - start
                else:
                    head.fade_out = 0.0
                if head.duration >= minimum_duration - 1e-9:
                    result.append(head)
            if c.end > end:  # keep tail
                tail = Clip(**asdict(c))
                tail.id = new_id() if c.start < start else c.id
                tail.in_point = c.source_time(end)
                tail.keyframes = shift_keyframes(c.keyframes, end - c.start)
                if preserve_presentation:
                    tail.presentation_in += end - c.start
                else:
                    tail.fade_in = 0.0
                tail.transition = 0.0
                tail.start = end
                if tail.duration >= minimum_duration - 1e-9:
                    result.append(tail)
        self.clips = result


def default_tracks() -> list[Track]:
    return [Track(new_id(), kind, name) for kind, name in
            (("video", "V2"), ("video", "V1"), ("audio", "A1"), ("audio", "A2"))]


@dataclass
class Timeline:
    id: str
    name: str
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    tracks: list[Track] = field(default_factory=list)
    frame_cuts: bool = False  # use the preview's frame selection for export
    magnetic: bool = False  # main track closes gaps automatically after edits

    @property
    def duration(self) -> float:
        return max((c.end for track in self.tracks for c in track.clips), default=0.0)

    @classmethod
    def from_dict(cls, data: dict) -> "Timeline":
        known = set(Clip.__dataclass_fields__)
        tracks = [Track(**{k: v for k, v in track.items() if k != "clips"},
                        clips=[Clip(**{k: v for k, v in clip.items() if k in known}) for clip in track["clips"]])
                  for track in data["tracks"]]
        return cls(data["id"], data["name"], data["width"], data["height"], data["fps"], tracks,
                   data.get("frame_cuts", False), data.get("magnetic", False))


class Project:
    """Shared media and named timelines. Editing and rendering use the active timeline."""

    def __init__(self, width=1920, height=1080, fps=30.0, sample_rate=48000, assets=None, tracks=None):
        timeline = Timeline(new_id(), "Main", width, height, fps, [] if tracks is None else tracks)
        self.timelines: dict[str, Timeline] = {timeline.id: timeline}
        self.active_timeline_id = timeline.id
        self.sample_rate = sample_rate
        self.assets: dict[str, MediaAsset] = {} if assets is None else assets

    @classmethod
    def new(cls) -> "Project":
        return cls(tracks=default_tracks())

    @property
    def active_timeline(self) -> Timeline:
        return self.timelines[self.active_timeline_id]

    def find_timeline(self, ref: str) -> Timeline | None:
        return self.timelines.get(ref) or next(
            (timeline for timeline in self.timelines.values() if timeline.name.casefold() == ref.casefold()), None)

    def all_tracks(self) -> Iterator[Track]:
        return (track for timeline in self.timelines.values() for track in timeline.tracks)

    def sequence_dependencies(self, timeline_id: str) -> set[str]:
        return {c.sequence_id for track in self.timelines[timeline_id].tracks for c in track.clips if c.sequence_id}

    def reachable_timelines(self, timeline_id: str) -> set[str]:
        seen = set()
        pending = [timeline_id]
        while pending:
            current = pending.pop()
            if current not in seen and current in self.timelines:
                seen.add(current)
                pending.extend(self.sequence_dependencies(current))
        return seen

    def can_nest(self, sequence_id: str, *, in_timeline: str | None = None) -> bool:
        return (sequence_id in self.timelines
                and (in_timeline or self.active_timeline_id) not in self.reachable_timelines(sequence_id))

    def asset_for_clip(self, clip: Clip) -> MediaAsset | None:
        if clip.sequence_id:
            source = self.timelines.get(clip.sequence_id)
            if source is None:
                return None
            return MediaAsset(source.id, "", "sequence", source.duration, source.width, source.height,
                              source.fps, has_video=True, has_audio=True)
        return self.assets.get(clip.asset_id)

    def timeline_assets(self) -> Iterator[MediaAsset]:
        ids = {clip.asset_id for tid in self.reachable_timelines(self.active_timeline_id)
               for track in self.timelines[tid].tracks for clip in track.clips}
        return (asset for asset in self.assets.values() if asset.id in ids)

    @property
    def width(self) -> int:
        return self.active_timeline.width

    @width.setter
    def width(self, value: int) -> None:
        self.active_timeline.width = value

    @property
    def height(self) -> int:
        return self.active_timeline.height

    @height.setter
    def height(self, value: int) -> None:
        self.active_timeline.height = value

    @property
    def fps(self) -> float:
        return self.active_timeline.fps

    @fps.setter
    def fps(self, value: float) -> None:
        self.active_timeline.fps = value

    @property
    def tracks(self) -> list[Track]:
        return self.active_timeline.tracks

    @tracks.setter
    def tracks(self, value: list[Track]) -> None:
        self.active_timeline.tracks = value

    @property
    def duration(self) -> float:
        return self.active_timeline.duration

    def video_tracks(self) -> list[Track]:
        return [t for t in self.tracks if t.kind == "video"]

    def audio_tracks(self) -> list[Track]:
        return [t for t in self.tracks if t.kind == "audio"]

    def track(self, track_id: str) -> Track | None:
        return next((t for t in self.tracks if t.id == track_id), None)

    def find_clip(self, clip_id: str) -> tuple[Track, Clip] | tuple[None, None]:
        for t in self.tracks:
            for c in t.clips:
                if c.id == clip_id:
                    return t, c
        return None, None

    # ---- serialization -------------------------------------------------
    def to_dict(self) -> dict:
        timelines = {tid: asdict(timeline) for tid, timeline in self.timelines.items()}
        active = timelines[self.active_timeline_id]
        return {
            "version": 2,
            # Keep the active timeline readable by older single-timeline clients.
            **{key: active[key] for key in ("width", "height", "fps", "tracks")},
            "sample_rate": self.sample_rate,
            "assets": [asdict(a) for a in self.assets.values()],
            "active_timeline_id": self.active_timeline_id,
            "timelines": list(timelines.values()),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Project":
        p = cls(sample_rate=d.get("sample_rate", 48000))
        akeys = set(MediaAsset.__dataclass_fields__)
        p.assets = {a["id"]: MediaAsset(**{k: v for k, v in a.items() if k in akeys}) for a in d["assets"]}
        if "timelines" in d:
            timelines = [Timeline.from_dict(entry) for entry in d["timelines"]]
            p.timelines = {timeline.id: timeline for timeline in timelines}
            names = {timeline.name.casefold() for timeline in timelines if timeline.name.strip()}
            if (len(p.timelines) != len(timelines) or len(names) != len(timelines)
                    or d.get("active_timeline_id") not in p.timelines):
                raise ValueError("The project has an invalid timeline list or active timeline.")
            p.active_timeline_id = d["active_timeline_id"]
        else:
            p.timelines[p.active_timeline_id] = Timeline.from_dict({**d, "id": p.active_timeline_id, "name": "Main"})
        for timeline_id in p.timelines:
            if any(not p.can_nest(dependency, in_timeline=timeline_id)
                   for dependency in p.sequence_dependencies(timeline_id)):
                raise ValueError("A nested sequence has a missing source or circular reference.")
        return p

    def save(self, path: str) -> None:
        # Replace the previous save only after the new file is fully written.
        destination = Path(path).resolve()
        data = self.to_dict()
        for asset in data["assets"]:
            media = Path(asset["path"])
            if media.is_absolute() and media.is_relative_to(destination.parent):
                asset["path"] = media.relative_to(destination.parent).as_posix()
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False,
                                             dir=destination.parent, suffix=".tmp") as f:
                temporary = Path(f.name)
                json.dump(data, f, indent=2)
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str) -> "Project":
        source = Path(path).resolve()
        with source.open(encoding="utf-8") as f:
            project = cls.from_dict(json.load(f))
        for asset in project.assets.values():
            if asset.path and not Path(asset.path).is_absolute():
                asset.path = os.path.abspath(source.parent / asset.path)
        return project
