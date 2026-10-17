#!/usr/bin/env python3
"""Cut 9:16 clips from a 16:9 podcast video, keeping the active speaker in frame.

No edits beyond the crop: no captions, zooms, music or transitions. For each clip:
  1. decode the range at analysis resolution, every frame
  2. detect camera cuts (HSV histogram jumps) so the crop re-targets exactly on them
  3. detect faces (YuNet), link them into per-shot tracks
  4. score each track's "talking" as lower-face motion minus upper-face motion
     (lips/jaw move when speaking; whole-head motion cancels out)
  5. pick the speaker per frame, drop flickers shorter than --min-hold seconds
  6. one locked crop position per (shot, speaker) segment; hard switches between them
  7. render with a single ffmpeg pass (piecewise crop x on t); audio is untouched

Usage: reframe.py SRC CLIPS_JSON OUT_DIR   (CLIPS_JSON: [{"id","start","end",...}])
"""
import argparse
import json
import os
import subprocess

import cv2
import numpy as np

ANALYSIS_W = 960
HERE = os.path.dirname(os.path.abspath(__file__))


def probe(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate", "-of", "json", path])
    s = json.loads(out)["streams"][0]
    num, den = map(int, s["r_frame_rate"].split("/"))
    return s["width"], s["height"], num / den


def read_frames(path, start, dur, w, h):
    ah = int(round(h * ANALYSIS_W / w / 2) * 2)
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", path, "-t", f"{dur:.3f}",
           "-vf", f"scale={ANALYSIS_W}:{ah}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = ANALYSIS_W * ah * 3
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        yield np.frombuffer(buf, np.uint8).reshape(ah, ANALYSIS_W, 3)
    p.wait()


def region_motion(prev, cur, x0, y0, x1, y1):
    h, w = cur.shape
    x0, x1 = max(0, int(x0)), min(w, int(x1))
    y0, y1 = max(0, int(y0)), min(h, int(y1))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return 0.0
    return float(cv2.absdiff(prev[y0:y1, x0:x1], cur[y0:y1, x0:x1]).mean())


def analyze(path, start, dur, w, h, fps, detector_path):
    det = None
    frames_faces, cuts = [], [0]
    prev_hist, prev_gray, prev_tiny, diffs = None, None, None, []
    for i, img in enumerate(read_frames(path, start, dur, w, h)):
        if det is None:
            det = cv2.FaceDetectorYN.create(detector_path, "", (img.shape[1], img.shape[0]), 0.7)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        # same room, same lights: angles can share a colour histogram, so also look for
        # a spike in whole-frame pixel difference relative to the recent baseline
        tiny = cv2.GaussianBlur(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (64, 36)), (3, 3), 0)
        if prev_hist is not None:
            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            diff = float(cv2.absdiff(prev_tiny, tiny).mean())
            base = np.median(diffs[-15:]) if diffs else 0.0
            if corr < 0.7 or diff > max(15.0, 6 * base):
                cuts.append(i)
                prev_gray = None  # motion across a cut is meaningless
            diffs.append(diff)
        prev_hist, prev_tiny = hist, tiny

        gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        _, faces = det.detect(img)
        out = []
        for f in faces if faces is not None else []:
            fx, fy, fw, fh = f[:4]
            if fw < 0.03 * img.shape[1]:
                continue  # background people, posters
            nose_y = f[9]
            mouth_y = (f[11] + f[13]) / 2
            mcx = (f[10] + f[12]) / 2
            mw = max(abs(f[12] - f[10]), 0.25 * fw)
            talk = 0.0
            if prev_gray is not None:
                low = region_motion(prev_gray, gray, mcx - 0.9 * mw, nose_y + 0.2 * (mouth_y - nose_y),
                                    mcx + 0.9 * mw, mouth_y + 1.1 * (mouth_y - nose_y))
                up = region_motion(prev_gray, gray, fx + 0.15 * fw, fy + 0.15 * fh,
                                   fx + 0.85 * fw, fy + 0.45 * fh)
                talk = max(0.0, low - up)
            scale = w / img.shape[1]
            out.append({"cx": (fx + fw / 2) * scale, "cy": (fy + fh / 2) * scale, "fw": fw * scale, "talk": talk})
        frames_faces.append(out)
        prev_gray = gray
    return frames_faces, cuts


def build_tracks(frames_faces, a, b):
    """Greedy nearest-centre linking within one shot (camera is static inside a shot)."""
    tracks = []  # each: {"cx": last cx, "fw", "obs": {frame: face}}
    for i in range(a, b):
        for face in frames_faces[i]:
            best, bd = None, None
            for t in tracks:
                d = abs(t["cx"] - face["cx"])
                if d < 0.6 * max(t["fw"], face["fw"]) and i not in t["obs"] and (bd is None or d < bd):
                    best, bd = t, d
            if best is None:
                best = {"obs": {}}
                tracks.append(best)
            best["cx"], best["fw"] = face["cx"], face["fw"]
            best["obs"][i] = face
    n = b - a
    return [t for t in tracks if len(t["obs"]) >= max(3, 0.3 * n)]


def smooth(x, k):
    if k <= 1 or len(x) == 0:
        return x
    return np.convolve(np.pad(x, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def merge_short_runs(labels, min_len):
    labels = list(labels)
    while True:
        runs, s = [], 0
        for i in range(1, len(labels) + 1):
            if i == len(labels) or labels[i] != labels[s]:
                runs.append((s, i))
                s = i
        short = [r for r in runs if r[1] - r[0] < min_len]
        if not short or len(runs) == 1:
            return labels
        s, e = min(short, key=lambda r: r[1] - r[0])
        k = runs.index((s, e))
        left = runs[k - 1] if k > 0 else None
        right = runs[k + 1] if k + 1 < len(runs) else None
        nb = max([r for r in (left, right) if r], key=lambda r: r[1] - r[0])
        for i in range(s, e):
            labels[i] = labels[nb[0]]


def plan_crops(frames_faces, cuts, w, h, fps, min_hold):
    cw = int(round(h * 9 / 16 / 2) * 2)
    n = len(frames_faces)
    bounds = cuts + [n]
    segments = []  # (start_frame, end_frame, crop_x, note)
    last_x = (w - cw) // 2
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b <= a:
            continue
        tracks = build_tracks(frames_faces, a, b)
        if not tracks:
            segments.append((a, b, last_x, "no face (hold)"))
            continue
        if len(tracks) == 1:
            labels = [0] * (b - a)
        else:
            k = max(1, int(round(0.8 * fps)))
            scores = np.zeros((len(tracks), b - a))
            for j, t in enumerate(tracks):
                raw = np.array([t["obs"][i]["talk"] if i in t["obs"] else 0.0 for i in range(a, b)])
                scores[j] = smooth(raw, k)
            labels = merge_short_runs(np.argmax(scores, axis=0), int(min_hold * fps))
        s = 0
        for i in range(1, len(labels) + 1):
            if i == len(labels) or labels[i] != labels[s]:
                t = tracks[labels[s]]
                xs = [t["obs"][f]["cx"] for f in range(a + s, a + i) if f in t["obs"]] or [t["cx"]]
                x = int(np.clip(np.median(xs) - cw / 2, 0, w - cw)) // 2 * 2
                note = f"speaker@{int(np.median(xs))}px" + (f" of {len(tracks)} faces" if len(tracks) > 1 else "")
                segments.append((a + s, a + i, x, note))
                last_x = x
                s = i
    # merge neighbours with (near) identical crop
    merged = []
    for seg in segments:
        if merged and abs(merged[-1][2] - seg[2]) < 0.02 * w:
            merged[-1] = (merged[-1][0], seg[1], merged[-1][2], merged[-1][3])
        else:
            merged.append(seg)
    return cw, merged


def crop_expr(segments, fps):
    expr = str(segments[-1][2])
    for a, b, x, _ in reversed(segments[:-1]):
        expr = f"if(lt(t\\,{(b - 0.5) / fps:.4f})\\,{x}\\,{expr})"
    return expr


def render(src, start, dur, cw, h, segments, fps, out_path):
    vf = (f"crop=w={cw}:h={h}:x={crop_expr(segments, fps)}:y=0,"
          f"scale=1080:1920:flags=lanczos,setsar=1")
    subprocess.check_call([
        "ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", src, "-t", f"{dur:.3f}",
        "-vf", vf, "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out_path])


def contact_sheet(path, out_png, dur):
    step = max(1.0, dur / 24)
    subprocess.check_call([
        "ffmpeg", "-v", "error", "-y", "-i", path,
        "-vf", f"fps=1/{step:.3f},scale=180:320,tile=8x3:padding=4", "-frames:v", "1", out_png])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("clips")
    ap.add_argument("out")
    ap.add_argument("--detector", default=os.path.join(HERE, "models", "face_detection_yunet_2023mar.onnx"))
    ap.add_argument("--min-hold", type=float, default=1.2, help="shortest speaker shot, seconds")
    ap.add_argument("--only", nargs="*", help="clip ids to (re)render")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    w, h, fps = probe(args.src)
    clips = json.load(open(args.clips))
    for c in clips:
        if args.only and str(c["id"]) not in args.only:
            continue
        dur = c["end"] - c["start"]
        faces, cuts = analyze(args.src, c["start"], dur, w, h, fps, args.detector)
        cw, segs = plan_crops(faces, cuts, w, h, fps, args.min_hold)
        name = f"{int(c['id']):02d}_{c.get('slug', 'clip')}"
        out = os.path.join(args.out, name + ".mp4")
        render(args.src, c["start"], dur, cw, h, segs, fps, out)
        contact_sheet(out, os.path.join(args.out, name + "_sheet.png"), dur)
        with open(os.path.join(args.out, name + ".plan.json"), "w") as f:
            json.dump({"clip": c, "camera_cuts_s": [round(x / fps, 2) for x in cuts],
                       "segments": [{"from_s": round(a / fps, 2), "to_s": round(b / fps, 2), "crop_x": x, "why": n}
                                    for a, b, x, n in segs]}, f, indent=1)
        print(f"{name}: {dur:.1f}s, {len(cuts)} shots, {len(segs)} crop segments", flush=True)


if __name__ == "__main__":
    main()
