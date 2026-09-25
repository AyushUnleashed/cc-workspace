"""End-to-end: brief -> paper edit -> tool calls -> timeline ops -> self-review -> render.
Then a second turn of user feedback -> diff -> re-render.

The tool calls below are exactly what an LLM (with these tools registered) would emit.
They're scripted so the demo runs offline and deterministically. Swapping in a real
model means passing Editor's methods as tools to a tool-use loop.

Run:  python3 demo.py        (outputs in ./out)
"""
import json
from pathlib import Path

from vedit.agent import Editor
from vedit.footage import build
from vedit.render import contact_sheet, render
from vedit.timeline import empty_timeline, tc

HERE = Path(__file__).parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


def call(ed, tool, **args):
    """Print a tool call as the LLM would make it, run it, print what changed."""
    print(f"\n  → {tool}({json.dumps(args)[1:-1]})")
    res = getattr(ed, tool)(**args)
    changes = res[1] if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], list) else None
    if changes:
        for c in changes:
            print(f"      {c}")
    elif res is not None and tool in ("search_footage", "get_transcript", "clips", "review"):
        print("      " + (json.dumps(res, indent=1) if not isinstance(res, str) else res).replace("\n", "\n      "))
    return res


def clip_with(ed, media, text):
    return next(c["id"] for c in ed.clips() if c["media"] == media and text in c["text"])


print("=" * 78 + "\nINGEST: generating synthetic footage + index (ASR words, shot descriptions, quality)")
media_lib, index = build(HERE / "footage")
for mid, m in index["media"].items():
    print(f"  {mid:20s} {m['kind']:9s} {m.get('description', '')[:60]}")

ed = Editor(empty_timeline(), media_lib, index)
BRIEF = ("45-second launch-day recap for LinkedIn. Hook fast, tell the 'almost didn't ship' story, "
         "add customer proof, end on energy. Name people on screen.")
print("\n" + "=" * 78 + f"\nTURN 1 — user brief:\n  \"{BRIEF}\"")

print("\n[agent reads the footage]")
call(ed, "search_footage", query="interview talking customer", kind="interview")
call(ed, "get_transcript", media="maya")
call(ed, "get_transcript", media="raj")

print("""
[agent writes a paper edit — shown to the user before touching the timeline]
  1. HOOK     Maya [0-5]   "Honestly, we almost didn't ship this."          + lower third
  2. STORY    Maya [6-26]  "…demo crashed twice. But the team stayed up…"  + b-roll: team at night
  3. PROOF    Raj  [0-17]  "…price was a no-brainer… signed up during…"     + lower third, b-roll: sign-up
  4. OUTRO    stage b-roll, title "LAUNCH DAY"; music bed under everything""")

print("\n[user approves → agent assembles]")
call(ed, "insert_soundbite", media="maya", from_word=0, to_word=5, reason="hook: stakes in one line")
call(ed, "insert_soundbite", media="maya", from_word=6, to_word=26, reason="story: crash → recovery")
call(ed, "insert_soundbite", media="raj", from_word=0, to_word=17, reason="proof: customer voice")
call(ed, "search_footage", query="team laptops night working", kind="broll")
call(ed, "search_footage", query="crowd cheering stage energetic", kind="broll")
call(ed, "search_footage", query="phone sign up form", kind="broll")
hook, story, proof = (clip_with(ed, "maya", "Honestly"), clip_with(ed, "maya", "crashed"),
                      clip_with(ed, "raj", "no-brainer"))
call(ed, "cover_with_broll", clip=story, media="broll_team_night", from_word=16, seconds=3.0,
     reason="show the team while she says it")
call(ed, "cover_with_broll", clip=proof, media="broll_signup", from_word=12, seconds=2.5,
     reason="visualize 'signed up'")
outro, _ = call(ed, "insert_broll_segment", media="broll_stage", seconds=4.0, reason="end on energy")
call(ed, "add_lower_third", clip=hook, name="Maya Chen", title="Co-founder")
call(ed, "add_lower_third", clip=proof, name="Raj Patel", title="Customer, Acme Corp")
title, _ = call(ed, "add_title", clip=outro, text="LAUNCH DAY", subtitle="Thank you for building with us")
call(ed, "add_music", media="music_upbeat", gain_db=-9)

print("\n[agent self-reviews its draft]")
issues = call(ed, "review")
print("\n[agent fixes what review found: fillers]")
call(ed, "remove_words", clip=clip_with(ed, "maya", "Um,"), from_word=6, to_word=6)
call(ed, "remove_words", clip=clip_with(ed, "raj", "Uh,"), from_word=0, to_word=0)
call(ed, "review")
call(ed, "clips")

render(ed.tl, media_lib, index, OUT / "cut_v1.mp4", OUT / "work")
dur = ed.tl.duration
print(f"\nRENDERED out/cut_v1.mp4  ({dur} frames = {dur / 30:.1f}s)")
contact_sheet(OUT / "cut_v1.mp4", ed.tl, [int(dur * k / 12) + 5 for k in range(12)], OUT / "cut_v1_sheet.png")
(OUT / "cut_v1.timeline.json").write_text(json.dumps(ed.tl.to_json(), indent=1))

print("\n" + "=" * 78 + "\nTURN 2 — user feedback:\n"
      "  \"Open with Raj's no-brainer line, it's a stronger hook. And make the end title say SHIPPED.\"")
call(ed, "clips")
call(ed, "move_clip", clip=clip_with(ed, "raj", "no-brainer"), index=0)
call(ed, "update_graphic", item=title, text="SHIPPED.", subtitle="Launch day 2026")
call(ed, "review")
call(ed, "clips")
render(ed.tl, media_lib, index, OUT / "cut_v2.mp4", OUT / "work")
dur = ed.tl.duration
print(f"\nRENDERED out/cut_v2.mp4  ({dur} frames = {dur / 30:.1f}s)")
contact_sheet(OUT / "cut_v2.mp4", ed.tl, [int(dur * k / 12) + 5 for k in range(12)], OUT / "cut_v2_sheet.png")
(OUT / "cut_v2.timeline.json").write_text(json.dumps(ed.tl.to_json(), indent=1))
print("\nundo stack depth:", len(ed.history), "| every step above is one reversible, logged transaction")
