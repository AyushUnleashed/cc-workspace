"""Synthetic 'raw footage' + the index our ingest pipeline would produce.

In the real product the index comes from ASR (word timestamps + speakers),
shot detection, a VLM describing each shot, embeddings and quality metrics.
Here we fabricate footage whose index we already know, so the demo runs offline.
Every frame burns in its SOURCE timecode so you can verify cuts are frame-exact.
"""
from __future__ import annotations

import json
import math
import subprocess
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .motion import FONT, FONT_BOLD
from .render import FFMPEG
from .timeline import tc

W, H, FPS, SR = 960, 540, 30, 48000

INTERVIEWS = {
    "maya": dict(name="Maya Chen", role="Co-founder", color=((40, 60, 110), (15, 20, 40)),
                 text="Honestly, we almost didn't ship this. Um, the night before launch the demo "
                      "crashed twice. But the team stayed up, fixed it, and the launch went amazing."),
    "raj": dict(name="Raj Patel", role="Customer, Acme Corp", color=((90, 50, 40), (30, 15, 12)),
                text="Uh, the price was a no-brainer for us. One of our leads signed up during the "
                     "keynote itself."),
}
BROLL = {
    "broll_stage": dict(desc="wide shot of the keynote stage, big crowd, speaker on stage, energetic",
                        secs=7, blur=False, scene="stage"),
    "broll_team_night": dict(desc="team working on laptops late at night in the office, tired, focused",
                             secs=6, blur=False, scene="laptops"),
    "broll_crowd_blurry": dict(desc="crowd at the launch event cheering, out of focus",
                               secs=5, blur=True, scene="stage"),
    "broll_signup": dict(desc="close-up of a phone screen showing a sign-up form being completed",
                         secs=5, blur=False, scene="phone"),
}
FILLERS = {"um", "uh"}


def _word_timings(text: str):
    words, t = [], 0.6
    for raw in text.split():
        w = raw.strip(",.")
        d = 0.18 + 0.045 * len(w)
        words.append({"w": raw, "s_sec": t, "e_sec": t + d, "filler": w.lower() in FILLERS})
        t += d + (0.45 if raw[-1] in ".," else 0.07)
    return words, t + 0.8


def _speech_audio(words, secs, pitch):
    t = np.arange(int(secs * SR)) / SR
    env = np.zeros_like(t)
    for w in words:
        a, b = int(w["s_sec"] * SR), int(w["e_sec"] * SR)
        env[a:b] = np.hanning(b - a)
    carrier = sum(np.sin(2 * np.pi * pitch * k * t) / k for k in range(1, 6))
    wobble = 1 + 0.3 * np.sin(2 * np.pi * 5 * t)
    return 0.35 * env * carrier * wobble / 2.3


def _music(secs=90, bpm=120):
    t = np.arange(int(secs * SR)) / SR
    beat = 60 / bpm
    phase = (t % beat) / beat
    kick = np.sin(2 * np.pi * (50 + 80 * np.exp(-phase * 30)) * t) * np.exp(-phase * 12)
    chords = [(220, 277, 330), (196, 247, 294), (175, 220, 262), (196, 247, 294)]
    bar = ((t // (beat * 4)) % 4).astype(int)
    pad = sum(np.sin(2 * np.pi * np.take([c[i] for c in chords], bar) * t) for i in range(3)) / 3
    hat = np.random.default_rng(0).standard_normal(len(t)) * np.exp(-(((t + beat / 2) % beat) / beat) * 60)
    return 0.45 * kick + 0.18 * pad + 0.05 * hat


def _write_wav(path, x):
    x = np.clip(x, -1, 1)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes((x * 32767).astype("<i2").tobytes())


def _draw_scene(kind, f, meta):
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    if kind == "interview":
        c1, c2 = meta["color"]
        for y in range(0, H, 6):
            k = y / H
            d.rectangle([0, y, W, y + 6], fill=tuple(int(c1[i] * (1 - k) + c2[i] * k) for i in range(3)))
        bob = 4 * math.sin(f / 9)
        d.ellipse([W * 0.42, H * 0.22 + bob, W * 0.58, H * 0.5 + bob], fill=(225, 190, 160))
        d.rounded_rectangle([W * 0.34, H * 0.52 + bob, W * 0.66, H * 1.05], radius=60, fill=(60, 60, 70))
        label = f"A-ROLL  ·  {meta['name'].upper()}  ·  CAM A"
    elif meta["scene"] == "stage":
        d.rectangle([0, 0, W, H], fill=(12, 10, 25))
        d.rectangle([W * 0.2, H * 0.25, W * 0.8, H * 0.55], fill=(70, 40, 120))
        d.ellipse([W * 0.47, H * 0.3, W * 0.53, H * 0.4], fill=(240, 220, 200))
        for n in range(90):
            x = (n * 97) % W
            y = H * 0.62 + (n * 53) % int(H * 0.36) + 5 * math.sin(f / 4 + n)
            d.ellipse([x, y, x + 18, y + 18], fill=(30 + n % 40, 30, 50))
        for n in range(3):
            x = W * (0.25 + 0.25 * n) + 60 * math.sin(f / 15 + n)
            d.polygon([(x, 0), (x - 90, H * 0.6), (x + 90, H * 0.6)], fill=(60, 50, 90))
        label = "B-ROLL  ·  " + meta["desc"].split(",")[0].upper()
    elif meta["scene"] == "laptops":
        d.rectangle([0, 0, W, H], fill=(10, 14, 22))
        for n in range(3):
            x = 80 + n * 300
            glow = 150 + int(40 * math.sin(f / 7 + n))
            d.polygon([(x, 330), (x + 220, 330), (x + 200, 200), (x + 20, 200)], fill=(glow, glow, 255))
            d.rectangle([x - 20, 330, x + 240, 345], fill=(80, 80, 90))
        label = "B-ROLL  ·  TEAM, LAPTOPS, NIGHT"
    else:  # phone
        d.rectangle([0, 0, W, H], fill=(230, 225, 215))
        d.rounded_rectangle([W * 0.38, 40, W * 0.62, H - 40], radius=30, fill=(20, 20, 20))
        d.rounded_rectangle([W * 0.395, 70, W * 0.605, H - 70], radius=12, fill=(250, 250, 250))
        filled = min(4, f // 30)
        for n in range(4):
            y = 140 + n * 60
            d.rectangle([W * 0.41, y, W * 0.59, y + 36], outline=(160, 160, 160), width=2,
                        fill=(200, 235, 205) if n < filled else (255, 255, 255))
        label = "B-ROLL  ·  PHONE SIGN-UP CLOSE-UP"
    if meta.get("blur"):
        img = img.filter(ImageFilter.GaussianBlur(9))
        d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 34], fill=(0, 0, 0))
    d.text((12, 7), label, font=ImageFont.truetype(FONT_BOLD, 18), fill=(255, 255, 255))
    d.rectangle([W - 250, H - 34, W, H], fill=(0, 0, 0))
    d.text((W - 240, H - 28), f"SRC {tc(f, FPS)}", font=ImageFont.truetype(FONT, 18), fill=(0, 255, 120))
    return img


def _encode(path, frames_fn, n, wav):
    p = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", str(wav),
                          "-c:v", "libx264", "-preset", "veryfast", "-g", "15", "-pix_fmt", "yuv420p",
                          "-c:a", "aac", "-shortest", str(path)], stdin=subprocess.PIPE)
    for f in range(n):
        p.stdin.write(frames_fn(f).tobytes())
    p.stdin.close()
    p.wait()


def build(dir_: Path) -> tuple[dict, dict]:
    """Returns (media_lib, index)."""
    dir_.mkdir(parents=True, exist_ok=True)
    idx_path = dir_ / "index.json"
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
        return idx["media_lib"], idx
    media_lib, index = {}, {"media": {}}
    for mid, meta in INTERVIEWS.items():
        words, secs = _word_timings(meta["text"])
        n = int(secs * FPS)
        wav = dir_ / f"{mid}.wav"
        _write_wav(wav, _speech_audio(words, n / FPS, 150 if mid == "maya" else 105))
        path = dir_ / f"{mid}.mp4"
        _encode(path, lambda f: _draw_scene("interview", f, meta), n, wav)
        media_lib[mid] = {"path": str(path), "frames": n}
        index["media"][mid] = {
            "kind": "interview", "speaker": meta["name"], "role": meta["role"],
            "description": f"medium shot, {meta['name']} talking to camera, interview",
            "quality": {"blur": 0.05, "shake": 0.02},
            "words": [{"w": w["w"], "s": round(w["s_sec"] * FPS), "e": round(w["e_sec"] * FPS) + 1,
                       "filler": w["filler"]} for w in words],
        }
    for mid, meta in BROLL.items():
        n = meta["secs"] * FPS
        wav = dir_ / f"{mid}.wav"
        _write_wav(wav, 0.02 * np.random.default_rng(1).standard_normal(int(meta["secs"] * SR)))
        path = dir_ / f"{mid}.mp4"
        _encode(path, lambda f, m=meta: _draw_scene("broll", f, m), n, wav)
        media_lib[mid] = {"path": str(path), "frames": n}
        index["media"][mid] = {"kind": "broll", "description": meta["desc"],
                               "quality": {"blur": 0.9 if meta["blur"] else 0.05, "shake": 0.1}}
    music = dir_ / "music_upbeat_120bpm.wav"
    _write_wav(music, _music())
    media_lib["music_upbeat"] = {"path": str(music), "frames": 90 * FPS}
    index["media"]["music_upbeat"] = {"kind": "music", "description": "upbeat electronic, 120 bpm, uplifting",
                                      "bpm": 120}
    index["media_lib"] = media_lib
    idx_path.write_text(json.dumps(index, indent=1))
    return media_lib, index
