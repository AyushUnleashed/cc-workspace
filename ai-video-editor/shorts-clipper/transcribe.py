#!/usr/bin/env python3
"""Transcribe a video with word-level timestamps (faster-whisper, CPU).

Writes <out>/transcript.json (segments + words) and <out>/transcript.txt
("[hh:mm:ss.s] text" per segment), which is what clip selection reads.
"""
import argparse
import json
import os

from faster_whisper import WhisperModel


def ts(sec):
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:04.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="small.en")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        args.video, word_timestamps=True, vad_filter=True, beam_size=5,
        condition_on_previous_text=False,
    )

    data, lines = [], []
    for seg in segments:
        words = [{"w": w.word, "s": round(w.start, 3), "e": round(w.end, 3)} for w in seg.words]
        data.append({"s": round(seg.start, 3), "e": round(seg.end, 3), "text": seg.text.strip(), "words": words})
        lines.append(f"[{ts(seg.start)}] {seg.text.strip()}")
        print(lines[-1], flush=True)

    with open(os.path.join(args.out, "transcript.json"), "w") as f:
        json.dump({"duration": info.duration, "segments": data}, f)
    with open(os.path.join(args.out, "transcript.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
