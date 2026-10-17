# shorts-clipper: podcast → 9:16 speaker-following clips

Raw clips only: no captions, zooms, music or transitions. The only change is the 9:16 crop,
which follows whoever is talking and switches exactly on the source's camera cuts.

```
pip install faster-whisper opencv-python-headless yt-dlp
mkdir -p models && curl -L -o models/face_detection_yunet_2023mar.onnx \
  https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx

python3 transcribe.py work/source.mp4 --out work/          # transcript.txt / transcript.json
# pick moments from transcript.txt → work/clips.json:
#   [{"id": 1, "slug": "fun-beats-grind", "start": 812.4, "end": 861.0, "why": "..."}]
python3 reframe.py work/source.mp4 work/clips.json work/out/   # NN_slug.mp4 + contact sheet + plan
```

| Step | How |
|---|---|
| Camera cuts | HSV histogram correlation drop, or a whole-frame pixel-difference spike (same-room angles share colours) |
| Faces | OpenCV YuNet on every frame at 960px wide; faces under 3% of frame width ignored |
| Who's talking | Per face: motion in the mouth/jaw box minus motion in the upper face (cancels head movement), smoothed over 0.8s |
| Stability | Speaker shots shorter than `--min-hold` (1.2s) are merged into neighbours; each shot has one locked crop, hard switches between |
| Render | One ffmpeg pass, piecewise `crop` x on `t`, scaled to 1080×1920, audio copied as AAC |

`NN_slug.plan.json` lists every crop segment and why it was chosen, and `NN_slug_sheet.png` is a
24-frame contact sheet for checking that a face is in frame throughout.
