"""Long-form podcast -> vertical 9:16 A-roll clips that follow whoever is talking.

No captions, no effects: tight cuts plus a reframe, ready for your own edit.

    python3 clipper.py prep   [ids...]   # ASR words, face/mouth tracks, shot cuts (cached per clip)
    python3 clipper.py words  <id>       # indexed words with pauses + speaker labels, to write a cut list
    python3 clipper.py render [ids...]   # keep-ranges -> tightened segments -> follow-the-speaker crop -> MP4
    python3 clipper.py sheet  <id>       # contact sheet of a render, for a quick visual check

Cut lists in clips.json cite WORD INDICES, not timestamps (see ../HOW-THE-AGENT-EDITS.md):
the editor picks words; code turns them into frame-exact cuts.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import re
import subprocess
import urllib.request
import wave
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
PLAN = HERE / "clips.json"
WORK = HERE / "work"
OUT = HERE / "out"
CACHE = HERE / ".cache"

MODELS = {
    "yunet.onnx": "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
                  "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                            "face_landmarker/float16/1/face_landmarker.task",
}
ASR_MODEL = "small.en"
ASR_PROMPT = "Um, so, uh, I mean, like, you know, yeah. Hmm."  # nudges Whisper to keep disfluencies
FACE_FPS = 8                   # face/mouth sampling rate
ANALYSIS_W, ANALYSIS_H = 1280, 720
OUT_W, OUT_H = 1080, 1920
FILLERS = {"um", "uh", "umm", "uhh", "erm", "er", "ah", "hmm", "mm"}
MAX_GAP = 0.28                 # pauses longer than this inside a keep-range get cut ...
PAD_IN, PAD_OUT = 0.05, 0.08   # ... leaving this much air either side of the words
QUIET_DB = 10                  # "silence" = within this many dB of the window's noise floor
MIN_RUN = 0.7                  # framing runs shorter than this get absorbed (no flicker)


# ---------------------------------------------------------------- plumbing

def load_plan() -> dict:
    plan = json.loads(PLAN.read_text())
    src = HERE / plan["source"]
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate", "-of", "json", str(src)],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = map(int, s["r_frame_rate"].split("/"))
    plan.update(src=str(src), w=s["width"], h=s["height"], fps=num / den)
    return plan


def clip_by_id(plan: dict, cid: str) -> dict:
    return next(c for c in plan["clips"] if c["id"] == cid)


def model(name: str) -> str:
    path = CACHE / name
    if not path.exists():
        CACHE.mkdir(exist_ok=True)
        urllib.request.urlretrieve(MODELS[name], path)
    return str(path)


def norm(word: str) -> str:
    return re.sub(r"[^a-z0-9']", "", word.lower().replace("’", "'"))


def snap(t: float, fps: float) -> float:
    return round(t * fps) / fps


def cached(c: dict, name: str):
    return json.loads((WORK / c["id"] / f"{name}.json").read_text())


# ---------------------------------------------------------------- prep: words, shots, faces

_whisper = None


def transcribe(wav: Path, t0: float) -> list[dict]:
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        _whisper = WhisperModel(ASR_MODEL, device="cpu", compute_type="int8",
                                download_root=str(CACHE / "whisper"))
    segs, _ = _whisper.transcribe(str(wav), language="en", beam_size=5, word_timestamps=True,
                                  condition_on_previous_text=False, initial_prompt=ASR_PROMPT)
    return [{"w": w.word.strip(), "s": round(t0 + w.start, 3), "e": round(t0 + w.end, 3)}
            for s in segs for w in s.words]


def detect_shots(P: dict, w0: float, w1: float) -> list[float]:
    """Hard camera cuts inside the window (multicam podcast edits are hard cuts)."""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-ss", f"{w0}", "-t", f"{w1 - w0}", "-i", P["src"],
                        "-an", "-vf", "scale=320:-2,select='gt(scene,0.3)',showinfo", "-f", "null", "-"],
                       capture_output=True, text=True)
    return [round(w0 + float(t), 3) for t in re.findall(r"pts_time:([\d.]+)", r.stderr)]


def track_faces(P: dict, w0: float, w1: float) -> list[dict]:
    """Per sampled frame: every face's centre/size (source px) and mouth opening.

    YuNet finds faces at any scale (wide shots too); FaceLandmarker on a crop around each
    face gives inner-lip landmarks, so mouth opening = lip gap / face height."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    det = cv2.FaceDetectorYN.create(model("yunet.onnx"), "", (ANALYSIS_W, ANALYSIS_H), 0.7)
    lmk = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model("face_landmarker.task")), num_faces=1))

    def mouth_open(img, x, y, w, h):
        cx, cy, s = x + w / 2, y + h / 2, max(w, h) * 1.6
        x0, y0 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
        x1, y1 = int(min(ANALYSIS_W, cx + s / 2)), int(min(ANALYSIS_H, cy + s / 2))
        crop = np.ascontiguousarray(img[y0:y1, x0:x1, ::-1])
        res = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
        if not res.face_landmarks:
            return None
        L, cw, ch = res.face_landmarks[0], x1 - x0, y1 - y0
        pt = lambda i: (L[i].x * cw, L[i].y * ch)
        return round(math.dist(pt(13), pt(14)) / max(1e-6, math.dist(pt(10), pt(152))), 4)

    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{w0}", "-t", f"{w1 - w0}", "-i", P["src"],
                             "-vf", f"fps={FACE_FPS},scale={ANALYSIS_W}:{ANALYSIS_H}",
                             "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    sx, size, frames, k = P["w"] / ANALYSIS_W, ANALYSIS_W * ANALYSIS_H * 3, [], 0
    while len(buf := proc.stdout.read(size)) == size:
        img = np.frombuffer(buf, np.uint8).reshape(ANALYSIS_H, ANALYSIS_W, 3)
        faces, dets = [], det.detect(img)[1]
        for d in (dets if dets is not None else []):
            x, y, w, h = map(float, d[:4])
            if h < 36:      # background people / posters
                continue
            faces.append({"cx": round((x + w / 2) * sx, 1), "cy": round((y + h / 2) * sx, 1),
                          "h": round(h * sx, 1), "m": mouth_open(img, x, y, w, h)})
        frames.append({"t": round(w0 + k / FACE_FPS, 3), "f": faces})
        k += 1
    proc.wait()
    return frames


def prep(c: dict, P: dict) -> None:
    d = WORK / c["id"]
    d.mkdir(parents=True, exist_ok=True)
    w0, w1 = c["window"]
    wav = d / "win.wav"
    if not wav.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{w0}", "-t", f"{w1 - w0}", "-i", P["src"],
                        "-vn", "-ac", "1", "-ar", "16000", str(wav)], check=True)
    for name, fn in (("words", lambda: transcribe(wav, w0)),
                     ("shots", lambda: detect_shots(P, w0, w1)),
                     ("faces", lambda: track_faces(P, w0, w1))):
        path = d / f"{name}.json"
        if not path.exists():
            path.write_text(json.dumps(fn()))
    print(f"prepped {c['id']}: {len(cached(c, 'words'))} words, {len(cached(c, 'shots'))} cuts")


# ---------------------------------------------------------------- speaker labels (from a labelled transcript)

def speaker_labels(words: list[dict], P: dict) -> list[str]:
    """Align ASR words to the published transcript (speaker-name lines + paragraphs)
    and copy its speaker labels across. Unaligned words inherit from their neighbour."""
    path = HERE / P.get("transcript", "")
    names = P.get("speakers", {})
    if not path.is_file() or not names:
        return ["?"] * len(words)
    toks, who, cur = [], [], None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line in names:
            cur = names[line]
        elif cur:
            for t in filter(None, map(norm, line.split())):
                toks.append(t)
                who.append(cur)
    a = [norm(w["w"]) for w in words]
    grams: dict[tuple, list[int]] = {}
    for j in range(len(toks) - 3):
        grams.setdefault(tuple(toks[j:j + 4]), []).append(j)
    votes = [j - i for i in range(len(a) - 3) for j in grams.get(tuple(a[i:i + 4]), [])]
    if not votes:
        return ["?"] * len(words)
    off = int(np.median(votes))
    lo, hi = max(0, off - 200), min(len(toks), off + len(a) + 200)
    labels: list[str | None] = [None] * len(a)
    for blk in SequenceMatcher(None, a, toks[lo:hi], autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            labels[blk.a + k] = who[lo + blk.b + k]
    last = next((l for l in labels if l), "?")
    for i, l in enumerate(labels):
        labels[i] = last = l or last
    return labels


def show_words(c: dict, P: dict) -> None:
    words = cached(c, "words")
    labels = speaker_labels(words, P)
    out, cur, prev = [], None, None
    for i, w in enumerate(words):
        if labels[i] != cur:
            cur = labels[i]
            out.append(f"\n{cur:>5} {w['s']:8.2f} |")
        if prev and w["s"] - prev["e"] > MAX_GAP:
            out.append(f" ‖{w['s'] - prev['e']:.1f}‖")
        out.append(f" {i}:{w['w']}")
        prev = w
    print("".join(out))


# ---------------------------------------------------------------- render: cuts

def envelope(wav: Path) -> tuple[np.ndarray, float]:
    with wave.open(str(wav)) as f:
        x = np.frombuffer(f.readframes(f.getnframes()), np.int16).astype(np.float32) / 32768
    hop = 160                                            # 10 ms at 16 kHz
    n = len(x) // hop
    db = 20 * np.log10(np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-6)
    return db, float(np.percentile(db, 10))


def keep_segments(c: dict, words: list[dict], P: dict) -> list[list[float]]:
    """Word ranges -> source time segments: drop fillers, cut long pauses, then nudge each
    edge onto real silence in the audio (ASR word times drift by ~0.1 s)."""
    w0, fps = c["window"][0], P["fps"]
    db, floor = envelope(WORK / c["id"] / "win.wav")
    quiet = db < floor + QUIET_DB
    idx = lambda t: int(np.clip((t - w0) * 100, 0, len(db) - 1))

    def snap_in(t):                       # last quiet 10 ms frame at or before the onset
        for i in range(idx(t + 0.05), idx(t - 0.15) - 1, -1):
            if quiet[i]:
                return w0 + (i + 1) / 100
        lo, hi = idx(t - 0.15), idx(t + 0.05) + 1
        return w0 + (lo + int(np.argmin(db[lo:hi]))) / 100

    def snap_out(t):                      # first quiet frame after the word ends
        for i in range(idx(t - 0.05), idx(t + 0.2) + 1):
            if quiet[i]:
                return w0 + i / 100
        lo, hi = idx(t - 0.05), idx(t + 0.2) + 1
        return w0 + (lo + int(np.argmin(db[lo:hi]))) / 100

    runs = []
    for rng in c["keep"]:
        a, b, loose = rng[0], rng[1], "loose" in rng[2:]     # "loose": keep pauses (laughs, beats)
        run = []
        for w in words[a:b + 1]:
            if norm(w["w"]) in FILLERS:
                if run:
                    runs.append(run)
                run = []
                continue
            if run and not loose and w["s"] - run[-1]["e"] > MAX_GAP:
                runs.append(run)
                run = []
            run.append(w)
        if run:
            runs.append(run)
    segs: list[list[float]] = []
    for run in runs:
        s = snap(snap_in(run[0]["s"]) - PAD_IN, fps)
        e = snap(snap_out(run[-1]["e"]) + PAD_OUT, fps)
        if segs and s - segs[-1][1] < 0.06:
            segs[-1][1] = max(segs[-1][1], e)
        else:
            segs.append([s, e])
    return segs


# ---------------------------------------------------------------- render: follow-the-speaker framing

def build_tracks(frames: list[dict], cuts: list[float], W: int) -> dict[int, list[dict]]:
    """Per shot, cluster faces by horizontal position (people don't move much on a podcast set)."""
    shots: dict[int, list[dict]] = {}
    nframes: dict[int, int] = {}
    for f in frames:
        s = bisect.bisect(cuts, f["t"])
        nframes[s] = nframes.get(s, 0) + 1
        tracks = shots.setdefault(s, [])
        for face in f["f"]:
            best = min(tracks, key=lambda tr: abs(tr["cx"] - face["cx"]), default=None)
            if best and abs(best["cx"] - face["cx"]) < 0.08 * W:
                best["pts"].append((f["t"], face["cx"], face["m"]))
                best["cx"] = float(np.median([p[1] for p in best["pts"]]))
            else:
                tracks.append({"cx": face["cx"], "h": face["h"], "pts": [(f["t"], face["cx"], face["m"])]})
    for s, tracks in shots.items():         # drop flickers (false positives, people passing)
        shots[s] = [tr for tr in tracks if len(tr["pts"]) >= 0.3 * nframes[s]]
    return shots


def talk_score(tr: dict, t0: float, t1: float) -> float:
    """Mouth movement in [t0, t1]: mean |change| in lip gap between samples."""
    m = [p[2] for p in tr["pts"] if t0 <= p[0] <= t1 and p[2] is not None]
    return float(np.mean(np.abs(np.diff(m)))) if len(m) >= 3 else 0.0


def frame_plan(c: dict, segs: list[list[float]], P: dict) -> list[list]:
    """For every kept moment, pick the face to frame: the only face in a single, or in a
    multi-face shot the one whose mouth is moving (checked against who the transcript says
    is talking). Returns [t0, t1, crop_x, why] runs covering the kept segments."""
    W, H = P["w"], P["h"]
    cw = int(H * 9 / 16) // 2 * 2
    cuts = cached(c, "shots")
    words = cached(c, "words")
    labels = speaker_labels(words, P)
    shots = build_tracks(cached(c, "faces"), cuts, W)
    starts = [w["s"] for w in words]
    who_at = lambda t: labels[max(0, bisect.bisect(starts, t) - 1)] if words else "?"
    step = 1 / FACE_FPS

    # Who-is-who in multi-face shots: the track whose mouth moves most while label X talks is X.
    identity: dict[tuple[int, int], str] = {}
    for s, tracks in shots.items():
        if len(tracks) < 2:
            continue
        lo = cuts[s - 1] if s > 0 else c["window"][0]
        hi = cuts[s] if s < len(cuts) else c["window"][1]
        spans: dict[str, list[tuple[float, float]]] = {}
        for i, w in enumerate(words):
            if lo <= w["s"] < hi:
                spans.setdefault(labels[i], []).append((w["s"], w["e"]))
        for lab, sp in spans.items():
            if sum(e - b for b, e in sp) < 1.5:
                continue
            scores = [np.mean([talk_score(tr, b - 0.2, e + 0.2) for b, e in sp]) for tr in tracks]
            ti = int(np.argmax(scores))
            if scores[ti] > 1.3 * sorted(scores)[-2] + 1e-3:
                identity[(s, ti)] = lab

    runs: list[list] = []
    for a, b in segs:
        t = a
        while t < b - 1e-6:
            s = bisect.bisect(cuts, t + 1e-3)
            nxt = min(b, cuts[s] if s < len(cuts) else b, t + step)
            tracks = shots.get(s, [])
            if not tracks:
                pick, why = None, "no face"
            elif len(tracks) == 1:
                pick, why = tracks[0], "single"
            else:
                lab = who_at(t)
                named = [tr for i, tr in enumerate(tracks) if identity.get((s, i)) == lab]
                if named:
                    pick, why = named[0], f"label {lab}"
                else:
                    scores = [talk_score(tr, t - 0.6, t + 0.6) for tr in tracks]
                    pick, why = tracks[int(np.argmax(scores))], "mouth"
            x = (runs[-1][2] if runs else (W - cw) // 2) if pick is None else \
                int(np.clip(pick["cx"] - cw / 2, 0, W - cw)) // 2 * 2
            runs.append([t, nxt, x, why, nxt - t, s])
            t = nxt

    # Consecutive runs are adjacent in the OUTPUT (the gaps between them are cut away), so merge
    # by framing, absorb flicker (a run shorter than MIN_RUN of kept time inside one shot takes
    # its predecessor's framing), and merge again.
    def merge(rs):
        out: list[list] = []
        for r in rs:
            if out and out[-1][2] == r[2] and out[-1][5] == r[5]:
                out[-1][1] = r[1]
                out[-1][4] += r[4]
            else:
                out.append(list(r))
        return out

    runs = merge(runs)
    for i in range(1, len(runs)):
        if runs[i][4] < MIN_RUN and runs[i][5] == runs[i - 1][5]:
            runs[i][2] = runs[i - 1][2]
    return [r[:4] for r in merge(runs)]


# ---------------------------------------------------------------- render: ffmpeg

def render(c: dict, P: dict) -> Path:
    fps, H = P["fps"], P["h"]
    cw = int(H * 9 / 16) // 2 * 2
    words = cached(c, "words")
    segs = keep_segments(c, words, P)
    runs = frame_plan(c, segs, P)
    t0 = segs[0][0]
    t1 = segs[-1][1]
    rel = lambda t: t - t0
    half = 0.5 / fps                       # select on frame centres: exactly (b-a)*fps frames per segment

    xexpr = str(runs[-1][2])
    for r in reversed(runs[:-1]):
        xexpr = f"if(lt(t,{rel(r[1]):.4f}),{r[2]},{xexpr})"
    sel = "+".join(f"gte(t,{rel(a) - half:.4f})*lt(t,{rel(b) - half:.4f})" for a, b in segs)
    g = [f"[0:v]setpts=PTS-STARTPTS,crop=w={cw}:h={H}:x='{xexpr}':y=0,select='{sel}',"
         f"setpts=N/FRAME_RATE/TB,scale={OUT_W}:{OUT_H}:flags=lanczos,setsar=1,format=yuv420p[v]",
         f"[0:a]asetpts=PTS-STARTPTS,asplit={len(segs)}" + "".join(f"[s{k}]" for k in range(len(segs)))]
    for k, (a, b) in enumerate(segs):
        g.append(f"[s{k}]atrim=start={rel(a):.6f}:end={rel(b):.6f},asetpts=PTS-STARTPTS,"
                 f"afade=t=in:d=0.01,afade=t=out:st={b - a - 0.01:.6f}:d=0.01[a{k}]")
    g.append("".join(f"[a{k}]" for k in range(len(segs))) + f"concat=n={len(segs)}:v=0:a=1[a]")

    d = WORK / c["id"]
    (d / "graph.txt").write_text(";\n".join(g))
    (d / "edl.json").write_text(json.dumps({"segments": segs, "framing": runs}, indent=1))
    OUT.mkdir(exist_ok=True)
    out = OUT / f"{c['id']}.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t0}", "-t", f"{t1 - t0 + 0.5}", "-i", P["src"],
                    "-filter_complex_script", str(d / "graph.txt"), "-map", "[v]", "-map", "[a]",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart", str(out)], check=True)
    dur = sum(b - a for a, b in segs)
    print(f"rendered {out.name}: {dur:.1f}s from {len(segs)} segments, {len(runs)} framing runs"
          + ("   !! over 60 s" if dur >= 60 else ""))
    return out


def sheet(c: dict) -> Path:
    out = WORK / c["id"] / "sheet.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(OUT / f"{c['id']}.mp4"),
                    "-vf", "fps=1/2,scale=180:320,tile=10x3", "-frames:v", "1", str(out)], check=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prep", "words", "render", "sheet"])
    ap.add_argument("ids", nargs="*")
    args = ap.parse_args()
    P = load_plan()
    clips = [clip_by_id(P, i) for i in args.ids] if args.ids else P["clips"]
    for c in clips:
        if args.cmd == "prep":
            prep(c, P)
        elif args.cmd == "words":
            show_words(c, P)
        elif args.cmd == "render" and c.get("keep"):
            render(c, P)
        elif args.cmd == "sheet":
            print(sheet(c))


if __name__ == "__main__":
    main()
