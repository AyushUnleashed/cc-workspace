# Talking-head → 9:16 social edit

Reproducible pipeline that turned one phone-shot talking-head clip into a ready-to-post 1080×1920 / 30 fps
vertical video (Reels / TikTok / Shorts).

```
pip install imageio-ffmpeg numpy pillow opencv-python-headless scipy
python3 fetch_assets.py                       # crawls Mixkit (free-licence only) for b-roll + music, fetches fonts
python3 build.py --src talking_head.mp4       # -> out/final_9x16.mp4
```

## What the edit does
- **Cuts**: removes 3 dead-air/hesitation gaps (11.7 s → 10.0 s), 6 ms audio fades at each join.
- **Framing**: alternating slow push-ins per segment (jump-cut energy), zoomed about the chin so captions stay clear of the face.
- **B-roll** (Mixkit): code screen (full-bleed), terminal wall + phone typing (floating cards over a blurred talking head).
- **Motion graphics**: LIVE TEST badge, OPUS 5.5 title plate with shine sweep, CLAUDE CODE VM chip, AUTONOMOUS MODE HUD
  (progress + PLAN/CODE/TEST steps — illustrative, not real telemetry), upload progress → ✓ UPLOADED, flash transitions.
- **Captions**: word-by-word, active-word highlight, kept inside social-app safe zones (y≈1430, x within 60–1020).
- **Audio**: rumble cut, denoise, EQ, compression; music (128 BPM) ducked under speech; synthesized whoosh/pop/boom/ding; -14 LUFS.

Set `CLAUDE_CODE = False` in `build.py` to caption the literal ASR text ("cloud code") instead of "Claude Code".

## Credits (see assets/credits.json)
Video/music from Mixkit under the Mixkit Free Licence; fonts Montserrat + JetBrains Mono (SIL OFL).
