"""Editing operations + undo/redo on top of the Project model."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

from . import animate
from .media import DEFAULT_IMAGE_CLIP, probe
from .model import MIN_CLIP, Clip, MediaAsset, Project, Timeline, Track, default_tracks, new_id
from .retime import validate_points


class Editor:
    def __init__(self, project: Project | None = None):
        self.project = project or Project.new()
        self.path: str | None = None
        self.selection: set[str] = set()
        self._undo: list[dict] = []
        self._redo: list[dict] = []
        self.dirty = False
        self.listeners: list = []  # callables(reason)
        self.mark_in: float | None = None  # In/Out range (UI state, not saved)
        self.mark_out: float | None = None
        self.clipboard: list[tuple[int, dict]] = []

    # ---- notifications / history ---------------------------------------
    def notify(self, reason: str = "edit"):
        if reason in ("edit", "props", "undo", "redo", "load"):
            self._normalize_transitions()
        for fn in list(self.listeners):
            fn(reason)

    def checkpoint(self, snap: dict | None = None):
        self._undo.append(snap or self.project.to_dict())
        del self._undo[:-200]
        self._redo.clear()
        self.dirty = True

    def snapshot(self) -> dict:
        return self.project.to_dict()

    def undo(self):
        if not self._undo:
            return
        self._redo.append(self.project.to_dict())
        self._restore_history(self._undo.pop(), "undo")

    def redo(self):
        if not self._redo:
            return
        self._undo.append(self.project.to_dict())
        self._restore_history(self._redo.pop(), "redo")

    def _restore_history(self, snapshot: dict, reason: str):
        old_timeline = self.project.active_timeline_id
        self.project = Project.from_dict(snapshot)
        if self.project.active_timeline_id != old_timeline:
            self._clear_timeline_context()
        self._prune_selection()
        self.dirty = True
        self.notify(reason)

    def _normalize_transitions(self):
        """Drop transitions whose clip no longer overlaps a neighbour (e.g. after a move)."""
        from .transitions import prev_clip

        for tr in self.project.tracks:
            for c in tr.clips:
                if c.transition > 0 and prev_clip(tr, c) is None:
                    c.transition = 0.0

    # ---- transitions -----------------------------------------------------
    def set_transition(self, clip_id: str, kind: str | None = None, duration: float | None = None,
                       checkpoint: bool = True) -> bool:
        """Add/change/remove the transition into `clip_id` from the clip before it.

        Adding overlaps the clips: this clip and everything after it on the track moves left."""
        from .transitions import DEFAULT_DURATION, max_duration, neighbour_before, prev_clip

        tr, c = self.project.find_clip(clip_id)
        if c is None:
            return False
        prev = prev_clip(tr, c) if c.transition > 0 else neighbour_before(tr, c)
        if prev is None:
            return False
        old = c.transition
        new = old if duration is None else duration
        if new <= 0 and old <= 0:
            new = DEFAULT_DURATION
        new = min(max(new, 0.0), max_duration(prev, c))
        if checkpoint:
            self.checkpoint()
        shift = new - old
        joint = c.start
        if abs(shift) > 1e-9:
            for o in tr.clips:
                if o.start >= joint - 1e-9:
                    o.start -= shift
        c.transition = new
        if kind:
            c.transition_kind = kind
        self.notify("props")
        return True

    def remove_transition(self, clip_id: str):
        self.set_transition(clip_id, duration=0.0)

    def transitions_everywhere(self, kind: str, duration: float):
        """Put the same transition on every cut of every video track."""
        from .transitions import neighbour_before

        self.checkpoint()
        for tr in self.project.video_tracks():
            for c in tr.sorted_clips():
                if c.transition > 0 or neighbour_before(tr, c) is not None:
                    self.set_transition(c.id, kind, duration, checkpoint=False)
        self.notify("edit")

    def _prune_selection(self):
        ids = {c.id for t in self.project.tracks for c in t.clips}
        self.selection &= ids

    def _clear_timeline_context(self):
        self.selection.clear()
        self.mark_in = self.mark_out = None

    # ---- project files ---------------------------------------------------
    def new_project(self):
        self.project = Project.new()
        self.path = None
        self._clear_timeline_context()
        self.clipboard.clear()
        self._undo.clear()
        self._redo.clear()
        self.dirty = False
        self.notify("load")

    def open(self, path: str):
        self.project = Project.load(path)
        self.path = path
        self._clear_timeline_context()
        self.clipboard.clear()
        self._undo.clear()
        self._redo.clear()
        self.dirty = False
        self.notify("load")

    def save(self, path: str | None = None):
        destination = path or self.path
        if not destination:
            raise ValueError("Choose a project file before saving.")
        self.project.save(destination)
        self.path = destination
        self.dirty = False
        self.notify("save")

    # ---- timelines -------------------------------------------------------
    def default_timeline_name(self, duplicate: bool = False) -> str:
        base = f"{self.project.active_timeline.name} copy" if duplicate else "Timeline"
        if duplicate and self.project.find_timeline(base) is None:
            return base
        number = 2
        while self.project.find_timeline(f"{base} {number}") is not None:
            number += 1
        return f"{base} {number}"

    def _timeline_name(self, name: str, exclude_id: str = "") -> str:
        name = name.strip()
        if not name:
            raise ValueError("Enter a timeline name.")
        existing = self.project.find_timeline(name)
        if existing is not None and existing.id != exclude_id:
            raise ValueError("A timeline already uses that name.")
        return name

    def require_timeline(self, ref: str) -> Timeline:
        timeline = self.project.find_timeline(ref)
        if timeline is None:
            raise ValueError(f"Timeline '{ref}' was not found.")
        return timeline

    def switch_timeline(self, ref: str) -> Timeline:
        timeline = self.require_timeline(ref)
        if timeline.id != self.project.active_timeline_id:
            self.project.active_timeline_id = timeline.id
            self._clear_timeline_context()
            self.notify("timeline")
        return timeline

    def create_timeline(self, name: str | None = None, duplicate: bool = False) -> Timeline:
        name = self._timeline_name(self.default_timeline_name(duplicate) if name is None else name)
        active = self.project.active_timeline
        tracks = deepcopy(active.tracks) if duplicate else default_tracks()
        if duplicate:
            for track in tracks:
                track.id = new_id()
                for clip in track.clips:
                    clip.id = new_id()
        timeline = Timeline(new_id(), name, active.width, active.height, active.fps, tracks)
        self.checkpoint()
        self.project.timelines[timeline.id] = timeline
        return self.switch_timeline(timeline.id)

    def rename_timeline(self, ref: str, name: str) -> Timeline:
        timeline = self.require_timeline(ref)
        name = self._timeline_name(name, timeline.id)
        if timeline.name != name:
            self.checkpoint()
            timeline.name = name
            self.notify("timeline")
        return timeline

    def delete_timeline(self, ref: str):
        timeline = self.require_timeline(ref)
        ids = list(self.project.timelines)
        if len(ids) == 1:
            raise ValueError("Keep at least one timeline in the project.")
        self.checkpoint()
        del self.project.timelines[timeline.id]
        if timeline.id == self.project.active_timeline_id:
            index = ids.index(timeline.id)
            self.switch_timeline(ids[index - 1] if index else ids[1])
        else:
            self.notify("timeline")

    # ---- media -----------------------------------------------------------
    def import_file(self, path: str):
        asset = probe(path)
        self.project.assets[asset.id] = asset
        self.dirty = True
        self.notify("assets")
        return asset

    def missing_media(self) -> list[MediaAsset]:
        return [asset for asset in self.project.assets.values() if not Path(asset.path).is_file()]

    def relink_media(self, folder: str) -> tuple[int, list[str]]:
        """Relink missing files by filename, keeping all clip edits and timing."""
        missing = self.missing_media()
        filenames = {}
        for asset in missing:
            name = asset.name.casefold()
            filenames[name] = filenames.get(name, 0) + 1
        clips = {}
        for track in self.project.all_tracks():
            for clip in track.clips:
                clips.setdefault(clip.asset_id, []).append(clip)
        replacements = {}
        unresolved = []
        for asset in missing:
            candidate = Path(folder) / asset.name
            if filenames[asset.name.casefold()] > 1 or not candidate.is_file():
                unresolved.append(asset.name)
                continue
            try:
                media = probe(str(candidate))
            except (OSError, ValueError):
                unresolved.append(asset.name)
                continue
            if (media.kind != asset.kind or media.has_audio != asset.has_audio
                    or (media.has_video and not (media.width and media.height))
                    or (asset.kind != "image" and media.duration + MIN_CLIP < max(
                        (clip.out_point for clip in clips.get(asset.id, ())), default=0.0))):
                unresolved.append(asset.name)
                continue
            replacements[asset.id] = replace(media, id=asset.id, origin=asset.origin)
        if replacements:
            self.checkpoint()
            self.project.assets.update(replacements)
            self.notify("assets")
        return len(replacements), unresolved

    # ---- clip ops --------------------------------------------------------
    def add_clip(self, asset_id: str, track_id: str | None = None, start: float | None = None) -> Clip | None:
        p = self.project
        asset = p.assets[asset_id]
        track = p.track(track_id) if track_id else None
        if track is None or not track.accepts(asset):
            want = "video" if (asset.has_video or asset.kind == "image") else "audio"
            # media goes on the main (bottom-most) video track, like CapCut/Premiere
            cands = [t for t in p.tracks if t.kind == want]
            track = (cands[-1] if want == "video" else cands[0]) if cands else None
            if track is None:
                track = self.add_track(want, notify=False)
        if start is None:
            start = max((c.end for c in track.clips), default=0.0)
        dur = DEFAULT_IMAGE_CLIP if asset.kind == "image" else asset.duration
        self.checkpoint()
        clip = Clip(new_id(), asset.id, max(start, 0.0), 0.0, dur)
        track.clear_range(clip.start, clip.end)
        track.clips.append(clip)
        self.selection = {clip.id}
        self.notify("edit")
        return clip

    def add_track(self, kind: str, notify: bool = True) -> Track:
        p = self.project
        if notify:
            self.checkpoint()  # before the change, so undo removes the track
        n = sum(1 for t in p.tracks if t.kind == kind) + 1
        tr = Track(new_id(), kind, f"{'V' if kind == 'video' else 'A'}{n}")
        if kind == "video":
            p.tracks.insert(0, tr)
        else:
            p.tracks.append(tr)
        if notify:
            self.notify("edit")
        return tr

    def remove_track(self, track_id: str):
        p = self.project
        tr = p.track(track_id)
        if tr is None or sum(1 for t in p.tracks if t.kind == tr.kind) <= 1:
            return
        self.checkpoint()
        p.tracks.remove(tr)
        self._prune_selection()
        self.notify("edit")

    def split(self, t: float):
        """Split selected clips at t (or every clip under t if nothing is selected)."""
        p = self.project
        targets = []
        for tr in p.tracks:
            for c in tr.clips:
                if c.start + MIN_CLIP < t < c.end - MIN_CLIP and (not self.selection or c.id in self.selection):
                    targets.append((tr, c))
        if not targets:
            return
        self.checkpoint()
        for tr, c in targets:
            right = replace(c, id=new_id(), start=t, in_point=c.source_time(t), fade_in=0.0, transition=0.0,
                            keyframes=animate.shift_keyframes(c.keyframes, t - c.start))
            c.out_point = c.source_time(t)
            c.fade_out = 0.0
            tr.clips.append(right)
        self.notify("edit")

    # ---- range / frame editing ---------------------------------------------
    def frame(self) -> float:
        return 1.0 / max(self.project.fps, 1.0)

    def snap_to_frame(self, t: float) -> float:
        f = self.frame()
        return round(t / f) * f

    def range(self) -> tuple[float, float] | None:
        if self.mark_in is None or self.mark_out is None:
            return None
        a, b = sorted((self.mark_in, self.mark_out))
        return (a, b) if b - a >= MIN_CLIP / 2 else None

    def _split_all_at(self, t: float):
        for tr in self.project.tracks:
            for c in list(tr.clips):
                if c.start + 1e-6 < t < c.end - 1e-6:
                    right = replace(c, id=new_id(), start=t, in_point=c.source_time(t), fade_in=0.0,
                                    transition=0.0, keyframes=animate.shift_keyframes(c.keyframes, t - c.start))
                    c.out_point = c.source_time(t)
                    c.fade_out = 0.0
                    tr.clips.append(right)

    def _shift_from(self, t: float, delta: float):
        for tr in self.project.tracks:
            for c in tr.clips:
                if c.start >= t - 1e-6:
                    c.start += delta

    def _remove_range(self, a: float, b: float, ripple: bool):
        for tr in self.project.tracks:
            tr.clear_range(a, b)
        if ripple:
            self._shift_from(b, -(b - a))

    def remove_range(self, a: float, b: float, ripple: bool = True):
        """Cut [a, b) out of every track; with ripple the rest of the video moves left."""
        if b - a <= 1e-6:
            return
        self.checkpoint()
        self._remove_range(a, b, ripple)
        self._prune_selection()
        self.notify("edit")

    def remove_ranges(self, ranges: list[tuple[float, float]], ripple: bool = True) -> float:
        """Remove several ranges in one undo step (used by silence removal). Returns seconds removed."""
        ranges = sorted((a, b) for a, b in ranges if b - a > 1e-6)
        if not ranges:
            return 0.0
        self.checkpoint()
        for a, b in reversed(ranges):  # back to front so earlier times stay valid
            self._remove_range(a, b, ripple)
        self._prune_selection()
        self.notify("edit")
        return sum(b - a for a, b in ranges)

    def duplicate_range(self, a: float, b: float):
        """Insert a copy of [a, b) (all tracks) right after it."""
        d = b - a
        if d <= 1e-6:
            return
        self.checkpoint()
        copies = []
        for tr in self.project.tracks:
            for c in tr.clips:
                if c.end <= a + 1e-6 or c.start >= b - 1e-6:
                    continue
                s0, s1 = max(a, c.start), min(b, c.end)
                cp = replace(c, id=new_id(), start=s0 + d, in_point=c.source_time(s0), out_point=c.source_time(s1),
                             keyframes=animate.shift_keyframes(c.keyframes, s0 - c.start))
                if c.start < a - 1e-6:
                    cp.fade_in, cp.transition = 0.0, 0.0
                if c.end > b + 1e-6:
                    cp.fade_out = 0.0
                copies.append((tr, cp))
        self._split_all_at(b)
        self._shift_from(b, d)
        for tr, cp in copies:
            tr.clips.append(cp)
        self.selection = {cp.id for _, cp in copies}
        self.notify("edit")

    def remove_frame(self, t: float):
        f = self.frame()
        a = self.snap_to_frame(t)
        self.remove_range(a, a + f)

    def insert_still(self, image_path: str, t: float, duration: float = 2.0, track_id: str | None = None):
        """Freeze frame: insert a still image at t, pushing everything after it right."""
        from .media import probe

        asset = probe(image_path)
        self.checkpoint()
        self.project.assets[asset.id] = asset
        self._split_all_at(t)
        self._shift_from(t, duration)
        tr = self.project.track(track_id) if track_id else None
        if tr is None or tr.kind != "video":
            tr = self.project.video_tracks()[-1]
        clip = Clip(new_id(), asset.id, t, 0.0, duration)
        tr.clips.append(clip)
        self.selection = {clip.id}
        self.notify("assets")
        self.notify("edit")
        return clip

    def copy_selected(self) -> int:
        idx = {tr.id: i for i, tr in enumerate(self.project.tracks)}
        self.clipboard = []
        for tr in self.project.tracks:
            for c in tr.clips:
                if c.id in self.selection:
                    self.clipboard.append((idx[tr.id], asdict(c)))
        return len(self.clipboard)

    def paste(self, t: float) -> int:
        if not self.clipboard:
            return 0
        t0 = min(d["start"] for _, d in self.clipboard)
        self.checkpoint()
        new_sel = set()
        tracks = self.project.tracks
        for ti, d in self.clipboard:
            c = Clip(**{**d, "id": new_id(), "start": d["start"] - t0 + t, "transition": 0.0})
            asset = self.project.assets.get(c.asset_id)
            if asset is None and not c.is_text:
                continue
            tr = tracks[ti] if ti < len(tracks) and tracks[ti].accepts(asset) else None
            if tr is None:
                tr = next((x for x in tracks if x.accepts(asset)), None)
            if tr is None:
                continue
            tr.clear_range(c.start, c.end)
            tr.clips.append(c)
            new_sel.add(c.id)
        self.selection = new_sel
        self.notify("edit")
        return len(new_sel)

    def delete_selected(self, ripple: bool = False):
        if not self.selection:
            return
        self.checkpoint()
        for tr in self.project.tracks:
            removed = [c for c in tr.clips if c.id in self.selection]
            tr.clips = [c for c in tr.clips if c.id not in self.selection]
            if ripple:
                for r in sorted(removed, key=lambda c: -c.start):
                    for c in tr.clips:
                        if c.start >= r.end - 1e-9:
                            c.start -= r.duration
        self.selection.clear()
        self.notify("edit")

    def detach_audio(self):
        p = self.project
        made = []
        for tr in p.video_tracks():
            for c in tr.clips:
                a = p.assets.get(c.asset_id)
                keys = animate.keyframes_for(c, "volume")
                if (a is not None and c.id in self.selection and a.has_audio and a.kind == "video"
                        and (any(value > 0 for _, value, _ in keys) if keys else c.volume > 0)):
                    made.append(c)
        if not made:
            return
        self.checkpoint()
        for c in made:
            # find an audio track free in this range, else make one
            target = None
            for at in p.audio_tracks():
                if all(o.end <= c.start or o.start >= c.end for o in at.clips):
                    target = at
                    break
            if target is None:
                target = self.add_track("audio", notify=False)
            target.clips.append(Clip(new_id(), c.asset_id, c.start, c.in_point, c.out_point, c.volume,
                                     speed=c.speed, fade_in=c.fade_in, fade_out=c.fade_out,
                                     speed_ramp=c.speed_ramp,
                                     keyframes=animate.shift_keyframes({"volume": (c.keyframes or {}).get("volume")}, 0.0)))
            animate.clear_keyframes(c, "volume")
            c.volume = 0.0
        self.notify("edit")

    def set_clip_props(self, clip_id: str, **props):
        tr, c = self.project.find_clip(clip_id)
        if c is None:
            return
        if "speed" in props:
            props.setdefault("speed_ramp", ())
        if "speed_ramp" in props:
            props["speed_ramp"] = validate_points(props["speed_ramp"])
        if "in_point" in props and c.keyframes:
            offset = c.time_between(c.in_point, props["in_point"])
            props.setdefault("keyframes", animate.shift_keyframes(c.keyframes, offset))
        self.checkpoint()
        for k, v in props.items():
            setattr(c, k, v)
        if props.keys() & {"speed", "speed_ramp", "out_point", "in_point"}:
            asset = self.project.assets.get(c.asset_id)
            if asset is not None:
                c.out_point = min(c.out_point, asset.duration)
            tr.clear_range(c.start, c.end, keep_id=c.id)
        self.notify("props")

    def set_keyframes(self, clip_id: str, prop: str, keys: list[tuple[float, float, str]]) -> bool:
        """Replace one property's curve as a single undoable edit."""
        _, clip = self.project.find_clip(clip_id)
        if clip is None or prop not in animate.KEYFRAME_PROPS:
            return False
        keys = sorted(keys, key=lambda key: key[0])
        if keys == animate.keyframes_for(clip, prop):
            return False
        self.checkpoint()
        if keys:
            if clip.keyframes is None:
                clip.keyframes = {}
            clip.keyframes[prop] = [list(key) for key in keys]
        else:
            clip.keyframes.pop(prop, None)
        self.notify("props")
        return True

    def add_shape(self, shape: dict, start: float | None = None, duration: float = 4.0, **fields) -> Clip:
        """Add a shape/sticker clip on the top-most video track that is free at `start`."""
        p = self.project
        start = 0.0 if start is None else max(start, 0.0)
        track = next((t for t in p.video_tracks()
                      if all(o.end <= start or o.start >= start + duration for o in t.clips)), None)
        self.checkpoint()
        if track is None:
            track = self.add_track("video", notify=False)
        clip = Clip(new_id(), "", start, 0.0, duration, shape=dict(shape), **fields)
        track.clear_range(clip.start, clip.end)
        track.clips.append(clip)
        self.selection = {clip.id}
        self.notify("edit")
        return clip

    def selected_clips(self) -> list[Clip]:
        return [c for t in self.project.tracks for c in t.clips if c.id in self.selection]

    def update_selected(self, fn, reason: str = "props") -> int:
        """Apply fn(clip) to every selected clip as ONE undo step. Returns how many were changed."""
        clips = self.selected_clips()
        if not clips:
            return 0
        self.checkpoint()
        for c in clips:
            fn(c)
        self.notify(reason)
        return len(clips)

    def add_asset(self, asset) -> None:
        self.project.assets[asset.id] = asset
        self.dirty = True
        self.notify("assets")

    def swap_asset(self, clip_id: str, asset, keep_timing: bool = True):
        """Point a clip at a newly generated file (AI result) that covers the clip's old source range."""
        tr, c = self.project.find_clip(clip_id)
        if c is None:
            return
        self.checkpoint()
        self.project.assets[asset.id] = asset
        dur = c.out_point - c.in_point
        c.speed_ramp = tuple((source - c.in_point, rate) for source, rate in c.speed_ramp)
        c.asset_id = asset.id
        c.in_point = 0.0
        c.out_point = min(dur, asset.duration) if keep_timing else asset.duration
        self.notify("assets")
        self.notify("edit")

    def add_text(self, text: str = "Your text", start: float | None = None, duration: float = 4.0,
                 track_id: str | None = None, **style) -> Clip:
        """Add a text title on the top-most video track that is free at `start`."""
        p = self.project
        start = 0.0 if start is None else max(start, 0.0)
        track = p.track(track_id) if track_id else None
        if track is None or track.kind != "video":
            track = next((t for t in p.video_tracks()
                          if all(o.end <= start or o.start >= start + duration for o in t.clips)), None)
        self.checkpoint()
        if track is None:
            track = self.add_track("video", notify=False)
        clip = Clip(new_id(), "", start, 0.0, duration, text=text, **style)
        track.clear_range(clip.start, clip.end)
        track.clips.append(clip)
        self.selection = {clip.id}
        self.notify("edit")
        return clip

    def duplicate_selected(self):
        made = []
        for tr in self.project.tracks:
            for c in list(tr.clips):
                if c.id in self.selection:
                    made.append((tr, c))
        if not made:
            return
        self.checkpoint()
        new_sel = set()
        for tr, c in made:
            d = replace(c, id=new_id(), start=c.end, keyframes=animate.shift_keyframes(c.keyframes, 0.0))
            tr.clear_range(d.start, d.end)
            tr.clips.append(d)
            new_sel.add(d.id)
        self.selection = new_sel
        self.notify("edit")

    def remove_asset(self, asset_id: str):
        if asset_id not in self.project.assets:
            return
        self.checkpoint()
        for tr in self.project.all_tracks():
            tr.clips = [c for c in tr.clips if c.asset_id != asset_id]
        del self.project.assets[asset_id]
        self._prune_selection()
        self.notify("assets")

    def toggle_track(self, track_id: str, attr: str):
        tr = self.project.track(track_id)
        if tr is None:
            return
        self.checkpoint()
        setattr(tr, attr, not getattr(tr, attr))
        self.notify("edit")
