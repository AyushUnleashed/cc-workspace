# 9:16 talking-head edit — built natively in HyperFrames

HTML/CSS/GSAP compositions rendered by the HyperFrames engine (`hyperframes@0.8.91`).

```
npm install
# put the source take next to index.html as a240a322-VID20260929170440.mp4
#   (or: npx hyperframes init --video <take.mp4> to re-transcode it for seeking)
npm run check       # lint + runtime + layout + motion + contrast
npm run render      # -> renders index.html to MP4
```

| File | Role |
|---|---|
| `index.html` | the edit: 4 jump-cut segments of one take (`data-media-start`), per-cut punch-ins on inner wrappers, 3 B-roll cutaways, vignette/scrim, voice / music / SFX audio buses |
| `compositions/captions.html` | word-by-word caption track (kept inside the title-safe box) |
| `compositions/graphics.html` | badge, keyword title, chips, autonomy + upload HUDs, flash transitions |
| `transcript.json` | corrected word-level transcript (source time) |
| `assets/` | Mixkit free-licence B-roll + music (credits.json), bundled SFX, vendored GSAP. Re-fetch with `../../talking-head-edit/fetch_assets.py` |

Audio: one voice-cleanup chain on the `voiceover` bus (rumble cut, mud cut, compressor, clarity, limiter);
the music bed is carved under the voice with `hyperframes-audio/scripts/carve.mjs`. Final: -14.8 LUFS, -1.6 dBFS peak.
The autonomy HUD (progress + PLAN/CODE/TEST) is an illustrative motion graphic, not real telemetry.
