"""The agent's tool layer.

The LLM never sees frame numbers it has to invent. It works in *semantic* units
(media ids, word indices, clip ids, template names). These tools compile that
intent into low-level timeline ops, snapping everything to frames with
deterministic rules. The same ops are what the UI's buttons call.

    LLM  --tool call-->  semantic tool  --ops-->  apply_ops  --> new timeline + diff
                                                       |
                                     review() lints the result, the LLM reads the issues
"""
from __future__ import annotations

import re

from .timeline import Timeline, apply_ops, diff, tc

PRE_ROLL = 4    # frames kept before the first word (so the cut doesn't clip the breath)
POST_ROLL = 6   # frames kept after the last word


class Editor:
    """Holds the session state the agent operates on."""

    def __init__(self, tl: Timeline, media_lib: dict, index: dict):
        self.tl, self.media_lib, self.index = tl, media_lib, index
        self.history: list[Timeline] = []   # undo stack
        self.log: list[str] = []

    # ------------------------------------------------------------ read tools
    def search_footage(self, query: str, kind: str | None = None, k: int = 3):
        """Stand-in for embedding search: keyword overlap + a quality penalty."""
        q = set(re.findall(r"\w+", query.lower()))
        hits = []
        for mid, m in self.index["media"].items():
            if kind and m["kind"] != kind:
                continue
            text = (m.get("description", "") + " " + " ".join(w["w"] for w in m.get("words", []))).lower()
            score = len(q & set(re.findall(r"\w+", text))) / max(1, len(q))
            score -= 0.5 * m.get("quality", {}).get("blur", 0)   # never prefer a soft shot
            hits.append((round(score, 2), mid, m.get("description", "")))
        return sorted(hits, reverse=True)[:k]

    def get_transcript(self, media: str) -> str:
        """Words with indices, which the LLM cites instead of timestamps."""
        return " ".join(f"[{i}]{w['w']}" for i, w in enumerate(self.index["media"][media]["words"]))

    # ------------------------------------------------------------ write tools
    def _commit(self, ops, label):
        new, results = apply_ops(self.tl, self.media_lib, ops)   # raises -> nothing changes
        changes = diff(self.tl, new)
        self.history.append(self.tl)
        self.tl = new
        self.log.append(label)
        return results, changes

    def _word_frames(self, media, a, b):
        words = self.index["media"][media]["words"]
        return words[a]["s"], words[b]["e"]

    def insert_soundbite(self, media, from_word, to_word, reason, at="end"):
        s, e = self._word_frames(media, from_word, to_word)
        s = max(0, s - PRE_ROLL)
        e = min(self.media_lib[media]["frames"], e + POST_ROLL)
        (cid,), changes = self._commit([{"op": "insert_clip", "track": "V1", "media": media,
                                         "src_in": s, "src_out": e, "at": at,
                                         "provenance": {"by": "agent", "reason": reason}}],
                                       f"insert_soundbite {media}[{from_word}:{to_word}]")
        return cid, changes

    def insert_broll_segment(self, media, seconds, reason, src_start=0):
        """A b-roll shot as its own story beat on V1 (e.g. an outro)."""
        n = int(seconds * self.tl.fps)
        (cid,), changes = self._commit([{"op": "insert_clip", "track": "V1", "media": media,
                                         "src_in": src_start, "src_out": src_start + n,
                                         "provenance": {"by": "agent", "reason": reason}}],
                                       f"insert_broll_segment {media}")
        return cid, changes

    def cover_with_broll(self, clip, media, from_word, seconds, reason):
        """Cutaway on V2, connected to the V1 clip, starting on a spoken word."""
        _, parent = self.tl.find(clip)
        w = self.index["media"][parent.media]["words"][from_word]
        offset = max(0, w["s"] - parent.src_in)
        n = min(int(seconds * self.tl.fps), parent.duration - offset)
        results, changes = self._commit([{"op": "insert_clip", "track": "V2", "media": media,
                                          "src_in": 0, "src_out": n,
                                          "params": {"anchor": {"item": clip, "offset": offset}},
                                          "provenance": {"by": "agent", "reason": reason}}],
                                        f"cover_with_broll {media} over {clip}")
        return results[0], changes

    def add_lower_third(self, clip, name, title, seconds=3.0):
        n = int(seconds * self.tl.fps)
        results, changes = self._commit([{"op": "add_motion", "template": "lower_third", "duration": n,
                                          "props": {"name": name, "title": title},
                                          "anchor": {"item": clip, "offset": 8},
                                          "provenance": {"by": "agent", "reason": f"identify {name}"}}],
                                        f"lower_third {name}")
        return results[0], changes

    def add_title(self, clip, text, subtitle="", seconds=2.5):
        n = int(seconds * self.tl.fps)
        results, changes = self._commit([{"op": "add_motion", "template": "title_card", "duration": n,
                                          "props": {"text": text, "subtitle": subtitle},
                                          "anchor": {"item": clip, "offset": 0},
                                          "provenance": {"by": "agent", "reason": "branded title"}}],
                                        f"title {text}")
        return results[0], changes

    def add_music(self, media, gain_db=-10):
        return self._commit([{"op": "add_music", "media": media, "gain_db": gain_db,
                              "provenance": {"by": "agent", "reason": "energy bed, ducked under speech"}}],
                            f"music {media}")

    def remove_words(self, clip, from_word, to_word):
        """Cut words out of a clip (e.g. fillers) and close the gap. The cut lands in the
        pauses around the words, keeping a few frames of air so the join sounds natural."""
        _, it = self.tl.find(clip)
        words = self.index["media"][it.media]["words"]
        inside = lambda i: 0 <= i < len(words) and it.src_in <= words[i]["s"] and words[i]["e"] <= it.src_out
        a = words[from_word - 1]["e"] + 3 if inside(from_word - 1) else it.src_in
        b = words[to_word + 1]["s"] - 3 if inside(to_word + 1) else it.src_out
        return self._commit([{"op": "cut_range", "item": clip, "src_from": a, "src_to": b}],
                            f"remove_words {clip}[{from_word}:{to_word}]")

    def update_graphic(self, item, **props):
        _, g = self.tl.find(item)
        return self._commit([{"op": "set_params", "item": item,
                              "params": {"props": {**g.params["props"], **props}}}], f"update {item}")

    def clips(self):
        """What the LLM sees of the main track: ids, order, and the words inside each clip."""
        rows = []
        for it in sorted(self.tl.track("V1").items, key=lambda i: i.rec_in):
            ws = [w["w"] for w in self.index["media"][it.media].get("words", [])
                  if it.src_in <= w["s"] and w["e"] <= it.src_out]
            rows.append({"id": it.id, "media": it.media, "at": tc(it.rec_in, self.tl.fps),
                         "len_s": round(it.duration / float(self.tl.fps), 2),
                         "text": " ".join(ws) or "(no speech)"})
        return rows

    def move_clip(self, clip, index):
        return self._commit([{"op": "move_clip", "item": clip, "index": index}], f"move {clip} -> {index}")

    def undo(self):
        self.tl = self.history.pop()

    # ------------------------------------------------------------ self-review
    def review(self) -> list[str]:
        """Deterministic lint the agent runs after every change. (The product adds a VLM
        that watches the rendered preview for things rules can't catch.)"""
        issues, fps = [], self.tl.fps
        v1 = sorted(self.tl.track("V1").items, key=lambda i: i.rec_in)
        for it in v1:
            m = self.index["media"][it.media]
            for edge, f in (("in", it.src_in), ("out", it.src_out)):
                for w in m.get("words", []):
                    if w["s"] < f < w["e"] - 1:
                        issues.append(f"{it.id} {edge}-point at {tc(it.rec_in, fps)} cuts mid-word '{w['w']}'")
            if it.duration < 12:
                issues.append(f"{it.id} is only {it.duration} frames (flash frame?)")
            for w in m.get("words", []):
                if w.get("filler") and it.src_in <= w["s"] and w["e"] <= it.src_out:
                    issues.append(f"{it.id} still contains filler '{w['w']}'")
        for t in ("V2",):
            for o in self.tl.track(t).items:
                if self.index["media"][o.media].get("quality", {}).get("blur", 0) > 0.5:
                    issues.append(f"{o.id} uses a blurry shot ({o.media})")
        return issues
