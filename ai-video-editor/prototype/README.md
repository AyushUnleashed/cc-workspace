# vedit: a tiny agent-driven video editor prototype

Demonstrates the core mechanics from [`../HOW-THE-AGENT-EDITS.md`](../HOW-THE-AGENT-EDITS.md):
brief → paper edit → semantic tool calls → frame-exact timeline ops → self-review → render,
then a round of user feedback → diff → re-render.

```
pip install pillow numpy imageio-ffmpeg
python3 demo.py          # ~1 min; writes out/cut_v1.mp4, out/cut_v2.mp4, contact sheets, timeline JSON
```

| File | What it is |
|---|---|
| `vedit/timeline.py` | Data model (integer frames), ops (`insert_clip`, `split`, `cut_range`, `trim`, `ripple_delete`, `move_clip`, `add_motion`, `add_music`), transactions + validation, connected-clip anchors, diffs |
| `vedit/agent.py` | The LLM-facing tool layer: word-index based tools compiled into ops, plus rule-based `review()` |
| `vedit/motion.py` | Motion graphics: templates → declarative scene spec (keyframes, springs, staggers) → renderer |
| `vedit/render.py` | Compiles a timeline into an FFmpeg filter graph (stand-in for a GPU compositor); derived captions; sidechain ducking |
| `vedit/footage.py` | Synthetic footage (burned-in source timecode) + the index an ingest pipeline would produce |
| `demo.py` | Two scripted agent turns (the exact tool calls an LLM would emit) |

The footage and index are synthetic so it runs offline. The tool calls are scripted; to put a real
model in the loop, register `Editor`'s methods as tools in any tool-use API.
