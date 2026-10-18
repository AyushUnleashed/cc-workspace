# shorts: long podcast → 9:16 A-roll clips

Cuts a long, multicam podcast into vertical clips under 60 s, with no captions or effects.
It removes fillers and dead air, and the crop follows whoever is talking.

```
pip install faster-whisper mediapipe opencv-python-headless numpy
# put the episode at work/source.mp4 and the speaker-labelled transcript at work/transcript.txt
python3 clipper.py prep            # per clip window: word-level ASR, shot cuts, face + mouth tracks (cached)
python3 clipper.py words 01-ai-mma-coach   # indexed words, pauses, speaker labels -> write "keep" ranges
python3 clipper.py render          # every clip that has "keep" ranges -> out/<id>.mp4 (1080x1920)
python3 clipper.py sheet 01-ai-mma-coach   # contact sheet for a quick visual check
```

## How it works

| Step | What happens |
|---|---|
| Pick | `clips.json` holds a source window, hook and reasoning for each clip. Picks come from reading the full transcript. |
| Cut list | `keep` = word-index ranges, e.g. `[[3, 41], [57, 120, "loose"]]`. The editor cites words, not timestamps (see `../HOW-THE-AGENT-EDITS.md`). `"loose"` keeps pauses intact (laughs, beats). |
| Tighten | Drops pure fillers (um/uh). Cuts pauses over 0.28 s down to about 0.13 s. Snaps every cut onto real silence in the audio, because ASR word times drift. Snaps all cuts to the frame grid so A/V stays frame-exact. |
| Framing | Detects shots (hard cuts) and finds faces with YuNet. FaceLandmarker gives mouth opening per face. A single face gets centred. In a multi-face shot, the transcript's speaker labels decide which face to frame (each face is learned per shot from mouth movement while that person talks), falling back to whichever mouth is moving. Short flickers are absorbed so the frame doesn't jitter. |
| Render | One ffmpeg pass. A crop whose x changes per run, `select` for the kept frames, and `atrim` with 10 ms fades for audio (no clicks). Then a 1080x1920 upscale, H.264 CRF 17, AAC 192k. |
