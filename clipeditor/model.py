"""Project data model: assets, tracks, clips. Pure Python, no Qt."""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

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

    Timeline length = (out_point - in_point) / speed.
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
        return (self.out_point - self.in_point) / self.speed

    @property
    def end(self) -> float:
        return self.start + self.duration

    def source_time(self, t: float) -> float:
        return self.in_point + (t - self.start) * self.speed

    def fade_gain(self, t: float) -> float:
        """0..1 multiplier from fades at timeline time t."""
        g = 1.0
        if self.fade_in > 0:
            g = min(g, max((t - self.start) / self.fade_in, 0.0))
        if self.fade_out > 0:
            g = min(g, max((self.end - t) / self.fade_out, 0.0))
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

    def clear_range(self, start: float, end: float, keep_id: str | None = None) -> None:
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
                head.fade_out = 0.0
                if head.duration >= MIN_CLIP:
                    result.append(head)
            if c.end > end:  # keep tail
                tail = Clip(**asdict(c))
                tail.id = new_id() if c.start < start else c.id
                tail.in_point = c.source_time(end)
                tail.fade_in = 0.0
                tail.transition = 0.0
                tail.start = end
                if tail.duration >= MIN_CLIP:
                    result.append(tail)
        self.clips = result


@dataclass
class Project:
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    sample_rate: int = 48000
    assets: dict[str, MediaAsset] = field(default_factory=dict)
    tracks: list[Track] = field(default_factory=list)

    @classmethod
    def new(cls) -> "Project":
        p = cls()
        p.tracks = [
            Track(new_id(), "video", "V2"),
            Track(new_id(), "video", "V1"),
            Track(new_id(), "audio", "A1"),
            Track(new_id(), "audio", "A2"),
        ]
        return p

    @property
    def duration(self) -> float:
        return max((c.end for t in self.tracks for c in t.clips), default=0.0)

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
        return {
            "version": 1,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "sample_rate": self.sample_rate,
            "assets": [asdict(a) for a in self.assets.values()],
            "tracks": [
                {**{k: v for k, v in asdict(t).items() if k != "clips"},
                 "clips": [asdict(c) for c in t.clips]}
                for t in self.tracks
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Project":
        p = cls(d["width"], d["height"], d["fps"], d.get("sample_rate", 48000))
        akeys = set(MediaAsset.__dataclass_fields__)
        p.assets = {a["id"]: MediaAsset(**{k: v for k, v in a.items() if k in akeys}) for a in d["assets"]}
        known = set(Clip.__dataclass_fields__)
        p.tracks = [
            Track(**{**{k: v for k, v in t.items() if k != "clips"},
                     "clips": [Clip(**{k: v for k, v in c.items() if k in known}) for c in t["clips"]]})
            for t in d["tracks"]
        ]
        return p

    def save(self, path: str) -> None:
        # Replace the previous save only after the new file is fully written.
        destination = os.path.realpath(path)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False,
                                             dir=os.path.dirname(destination), suffix=".tmp") as f:
                temporary = Path(f.name)
                json.dump(self.to_dict(), f, indent=2)
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str) -> "Project":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
