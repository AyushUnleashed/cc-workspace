"""Timeline data model + edit operations.

The timeline is plain data (JSON-serializable). Every change goes through
`apply_ops`, which is the ONLY way the UI or the agent can mutate an edit.
That gives us undo, diffs, validation and provenance in one place.

Time is stored in integer frames at the sequence rate (never float seconds).
"""
from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass, field, asdict
from fractions import Fraction

_ids = itertools.count(1)


def new_id(prefix: str) -> str:
    return f"{prefix}{next(_ids)}"


# ---------------------------------------------------------------- time utils
def sec_to_frame(sec: float, fps: Fraction, mode: str = "round") -> int:
    x = Fraction(sec).limit_denominator(1_000_000) * fps
    return {"round": round, "ceil": math.ceil, "floor": math.floor}[mode](x)


def frame_to_sec(frame: int, fps: Fraction) -> float:
    return float(Fraction(frame) / fps)


def tc(frame: int, fps: Fraction) -> str:
    """Frame -> HH:MM:SS:FF timecode (non-drop, integer fps)."""
    f = int(round(fps))
    s, ff = divmod(frame, f)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}:{ff:02d}"


# ---------------------------------------------------------------- data model
@dataclass
class Item:
    id: str
    kind: str                 # "clip" | "motion" | "music"
    rec_in: int               # position on the timeline (frames)
    duration: int             # length on the timeline (frames)
    media: str | None = None  # media id for clips/music
    src_in: int = 0           # first source frame used
    params: dict = field(default_factory=dict)       # effects / template props
    provenance: dict = field(default_factory=dict)   # who + why

    @property
    def rec_out(self) -> int:
        return self.rec_in + self.duration

    @property
    def src_out(self) -> int:
        return self.src_in + self.duration


@dataclass
class Track:
    id: str
    kind: str                 # "video" | "overlay" | "graphics" | "music"
    items: list[Item] = field(default_factory=list)


@dataclass
class Timeline:
    fps: Fraction
    width: int
    height: int
    tracks: list[Track]

    def track(self, tid: str) -> Track:
        return next(t for t in self.tracks if t.id == tid)

    def find(self, item_id: str) -> tuple[Track, Item]:
        for t in self.tracks:
            for it in t.items:
                if it.id == item_id:
                    return t, it
        raise KeyError(item_id)

    @property
    def duration(self) -> int:
        return max((it.rec_out for t in self.tracks for it in t.items), default=0)

    def to_json(self) -> dict:
        d = asdict(self)
        d["fps"] = f"{self.fps.numerator}/{self.fps.denominator}"
        return d


def empty_timeline(fps=Fraction(30), width=960, height=540) -> Timeline:
    return Timeline(fps, width, height, [
        Track("V1", "video"),      # main story (A-roll), magnetic / rippled
        Track("V2", "overlay"),    # b-roll cutaways over V1 (V1 audio continues)
        Track("G1", "graphics"),   # motion graphics (titles, lower thirds)
        Track("M1", "music"),      # music bed, auto-ducked under dialogue
    ])


# ---------------------------------------------------------------- operations
class OpError(Exception):
    pass


def _ripple(track: Track, from_frame: int, delta: int) -> None:
    for it in track.items:
        if it.rec_in >= from_frame:
            it.rec_in += delta


def _op_insert_clip(tl, media_lib, op):
    tr = tl.track(op["track"])
    m = media_lib[op["media"]]
    src_in, src_out = op["src_in"], op["src_out"]
    if not (0 <= src_in < src_out <= m["frames"]):
        raise OpError(f"source range {src_in}-{src_out} outside media {op['media']} (0-{m['frames']})")
    dur = src_out - src_in
    if tr.kind == "video":                  # magnetic main track: insert + ripple
        at = op.get("at", "end")
        at = max((i.rec_out for i in tr.items), default=0) if at == "end" else at
        _ripple(tr, at, dur)
    else:
        at = op.get("at", 0)
    it = Item(new_id("c"), "clip", at, dur, op["media"], src_in,
              dict(op.get("params", {})), op.get("provenance", {}))
    tr.items.append(it)
    return it.id


def _op_ripple_delete(tl, media_lib, op):
    tr, it = tl.find(op["item"])
    tr.items.remove(it)
    if tr.kind == "video":
        _ripple(tr, it.rec_out, -it.duration)


def _op_trim(tl, media_lib, op):
    """Move one edge of a clip. delta in frames (+ = later)."""
    tr, it = tl.find(op["item"])
    edge, d = op["edge"], op["delta"]
    m = media_lib.get(it.media, {"frames": 10**9})
    if edge == "in":
        new_src_in = it.src_in + d
        if not (0 <= new_src_in < it.src_out):
            raise OpError("trim in-point out of range")
        it.src_in, it.duration = new_src_in, it.duration - d
        if tr.kind == "video":
            _ripple(tr, it.rec_out + d, -d)   # later clips slide
            for child in _children(tl, it.id):  # connected clips stay on the same content
                child.params["anchor"]["offset"] -= d
    else:
        new_dur = it.duration + d
        if not (0 < new_dur and it.src_in + new_dur <= m["frames"]):
            raise OpError("trim out-point out of range")
        old_out = it.rec_out
        it.duration = new_dur
        if tr.kind == "video":
            _ripple_after(tr, it, old_out, d)


def _ripple_after(tr, it, from_frame, d):
    for o in tr.items:
        if o is not it and o.rec_in >= from_frame:
            o.rec_in += d


def _op_move_clip(tl, media_lib, op):
    """Reorder on the main track: remove + reinsert at index."""
    tr, it = tl.find(op["item"])
    if tr.kind != "video":
        it.rec_in = op["to"]
        return
    order = sorted(tr.items, key=lambda i: i.rec_in)
    order.remove(it)
    order.insert(op["index"], it)
    pos = 0
    for o in order:
        o.rec_in, pos = pos, pos + o.duration


def _op_add_motion(tl, media_lib, op):
    tr = tl.track(op.get("track", "G1"))
    params = {"template": op["template"], "props": op.get("props", {})}
    if "anchor" in op:
        params["anchor"] = dict(op["anchor"])
    it = Item(new_id("g"), "motion", op.get("at", 0), op["duration"], None, 0,
              params, op.get("provenance", {}))
    tr.items.append(it)
    return it.id


def _op_add_music(tl, media_lib, op):
    tr = tl.track("M1")
    tr.items.clear()
    it = Item(new_id("m"), "music", 0, 1, op["media"], op.get("src_in", 0),
              {"gain_db": op.get("gain_db", -8), "duck": op.get("duck", True)},
              op.get("provenance", {}))
    tr.items.append(it)
    return it.id


def _op_set_params(tl, media_lib, op):
    _, it = tl.find(op["item"])
    it.params.update(op["params"])


def _op_split(tl, media_lib, op):
    """Blade a main-track clip at a SOURCE frame. Connected clips that sit after the
    blade move to the right-hand piece so they stay on the same content."""
    tr, it = tl.find(op["item"])
    at = op["src_frame"]
    if not (it.src_in < at < it.src_out):
        raise OpError(f"split point {at} not inside {it.id} ({it.src_in}-{it.src_out})")
    left = at - it.src_in
    right = Item(new_id("c"), "clip", it.rec_in + left, it.duration - left, it.media, at,
                 copy.deepcopy(it.params), dict(it.provenance))
    it.duration = left
    tr.items.append(right)
    for child in _children(tl, it.id):
        if child.params["anchor"]["offset"] >= left:
            child.params["anchor"] = {"item": right.id, "offset": child.params["anchor"]["offset"] - left}
    return right.id


def _op_cut_range(tl, media_lib, op):
    """Remove source frames [src_from, src_to) from a main-track clip and ripple."""
    _, it = tl.find(op["item"])
    a, b = op["src_from"], op["src_to"]
    piece = _op_split(tl, media_lib, {"item": it.id, "src_frame": a}) if a > it.src_in else it.id
    _, p = tl.find(piece)
    if b < p.src_out:
        right = _op_split(tl, media_lib, {"item": piece, "src_frame": b})
        for child in _children(tl, piece):   # graphics on the removed bit slide to the join
            child.params["anchor"] = {"item": right, "offset": 0}
    _op_ripple_delete(tl, media_lib, {"item": piece})


def _children(tl, parent_id):
    return [o for t in tl.tracks if t.kind != "video" for o in t.items
            if o.params.get("anchor", {}).get("item") == parent_id]


def _resolve_anchors(tl, report):
    """Connected clips (b-roll, titles) follow their parent clip on V1 wherever it moves."""
    for t in tl.tracks:
        for o in list(t.items):
            a = o.params.get("anchor")
            if not a:
                continue
            try:
                _, parent = tl.find(a["item"])
            except KeyError:
                t.items.remove(o)
                report.append(f"removed {o.id}: its parent clip {a['item']} was deleted")
                continue
            o.rec_in = parent.rec_in + a["offset"]


OPS = {
    "insert_clip": _op_insert_clip,
    "ripple_delete": _op_ripple_delete,
    "trim": _op_trim,
    "move_clip": _op_move_clip,
    "add_motion": _op_add_motion,
    "add_music": _op_add_music,
    "set_params": _op_set_params,
    "split": _op_split,
    "cut_range": _op_cut_range,
}


def validate(tl: Timeline) -> None:
    for t in tl.tracks:
        items = sorted(t.items, key=lambda i: i.rec_in)
        for a, b in zip(items, items[1:]):
            if a.rec_out > b.rec_in:
                raise OpError(f"overlap on {t.id}: {a.id} and {b.id}")
        for it in items:
            if it.rec_in < 0 or it.duration <= 0:
                raise OpError(f"bad timing on {it.id}")


def apply_ops(tl: Timeline, media_lib: dict, ops: list[dict]) -> tuple[Timeline, list[str]]:
    """Transactional: either every op applies and the result validates, or nothing changes."""
    new = copy.deepcopy(tl)
    results = []
    for op in ops:
        fn = OPS.get(op["op"])
        if fn is None:
            raise OpError(f"unknown op {op['op']}")
        results.append(fn(new, media_lib, op))
    notes: list[str] = []
    _resolve_anchors(new, notes)
    # the music bed always spans the story (V1)
    story_len = max((i.rec_out for i in new.track("V1").items), default=1)
    for it in new.track("M1").items:
        it.rec_in, it.duration = 0, story_len
    validate(new)
    return new, results + notes


# ---------------------------------------------------------------- diff
def diff(old: Timeline, new: Timeline) -> list[str]:
    fps = new.fps
    def index(tl):
        return {it.id: (t.id, it) for t in tl.tracks for it in t.items}
    a, b = index(old), index(new)
    out = []
    for iid in b.keys() - a.keys():
        t, it = b[iid]
        why = it.provenance.get("reason", "")
        out.append(f"+ {t}:{iid} {_desc(it)} @ {tc(it.rec_in, fps)}  ({why})")
    for iid in a.keys() - b.keys():
        t, it = a[iid]
        out.append(f"- {t}:{iid} {_desc(it)}")
    for iid in a.keys() & b.keys():
        (_, x), (t, y) = a[iid], b[iid]
        changes = []
        if (x.src_in, x.duration) != (y.src_in, y.duration):
            changes.append(f"src {x.src_in}-{x.src_out} -> {y.src_in}-{y.src_out}")
        if x.rec_in != y.rec_in:
            changes.append(f"moved {tc(x.rec_in, fps)} -> {tc(y.rec_in, fps)}")
        if x.params != y.params:
            changes.append("params changed")
        if changes:
            out.append(f"~ {t}:{iid} {_desc(y)}: " + "; ".join(changes))
    return sorted(out, key=lambda s: s[2:])


def _desc(it: Item) -> str:
    if it.kind == "motion":
        return f"[{it.params['template']}]"
    return f"[{it.media} {it.src_in}-{it.src_out}]"
