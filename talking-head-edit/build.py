#!/usr/bin/env python3
"""Talking-head -> 9:16 social edit. Jump cuts, punch-ins, b-roll cutaways, word-by-word captions,
motion graphics, cleaned voice, ducked music + SFX.  Everything is composited frame by frame in Python
and piped to ffmpeg, so zooms are sub-pixel smooth.

    python3 fetch_assets.py
    python3 build.py --src /path/to/talking_head.mp4            # -> out/final_9x16.mp4
    python3 build.py --src ... --stills 1.0,2.2,5.0 --noencode   # dump preview stills only
"""
import argparse, json, math, os, subprocess, sys, wave
from functools import lru_cache

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
A = lambda *p: os.path.join(HERE, "assets", *p)
FF = imageio_ffmpeg.get_ffmpeg_exe()
W, H, FPS = 1080, 1920, 30
SR = 48000

# ------------------------------------------------------------------ edit decision list
# Source ranges kept (seconds). Gaps removed: 4.45-4.88, 6.92-7.42, 9.50-10.06 (dead air / hesitation).
SEGS = [(0.17, 4.45), (4.88, 6.92), (7.42, 9.50), (10.06, 11.69)]
SEG_FR = [round((b - a) * FPS) for a, b in SEGS]
SEG_OUT = [sum(SEG_FR[:i]) / FPS for i in range(len(SEGS))]            # out-time each segment starts
N = sum(SEG_FR)
DUR = N / FPS
# slow push-in per segment (start zoom, end zoom): alternating tight/wide gives the jump cuts energy
ZOOM = [(1.00, 1.06), (1.10, 1.16), (1.03, 1.08), (1.12, 1.19)]
ANCHOR = (540, 1300)   # zoom about the chin so the face grows upward and stays clear of the captions

# Caption words: (text, src_start, src_end, keyword).  Times hand-corrected against the speech energy
# envelope (Whisper's boundaries drift around pauses).  Set CLAUDE_CODE=False to caption the literal ASR "cloud code".
CLAUDE_CODE = True
WORDS = [
    ("HEY", .30, .58, 0), ("EVERYONE", .58, .92, 0), ("SO", .92, 1.05, 0), ("THIS", 1.05, 1.20, 0), ("IS", 1.20, 1.34, 0),
    ("A", 1.34, 1.42, 0), ("TEST", 1.42, 1.68, 0), ("OF", 1.68, 1.90, 0), ("OPUS", 1.90, 2.25, 1), ("5.5", 2.25, 2.95, 1),
    ("ON", 2.95, 3.08, 0), ("A", 3.08, 3.22, 0),
    ("CLAUDE" if CLAUDE_CODE else "CLOUD", 3.45, 3.74, 1), ("CODE", 3.78, 4.06, 1), ("VM", 4.08, 4.45, 1),
    ("WE", 4.92, 5.08, 0), ("ARE", 5.08, 5.18, 0), ("JUST", 5.18, 5.30, 0), ("CHECKING", 5.30, 5.55, 0), ("OUT", 5.55, 5.78, 0),
    ("IF", 5.78, 5.98, 0), ("IT", 5.98, 6.20, 0), ("CAN", 6.20, 6.42, 0), ("WORK", 6.42, 6.75, 0), ("AND", 6.75, 6.92, 0),
    ("OPERATE", 7.48, 8.10, 0), ("AUTONOMOUSLY", 8.10, 9.20, 1), ("AND", 9.25, 9.50, 0),
    ("I'M", 10.08, 10.24, 0), ("GONNA", 10.24, 10.42, 0), ("UPLOAD", 10.42, 10.66, 0), ("IT", 10.66, 10.84, 0),
    ("FROM", 10.84, 11.04, 0), ("MY", 11.04, 11.22, 0), ("PHONE", 11.22, 11.50, 1),
]
CHUNKS = [2, 3, 3, 2, 2, 3, 3, 2, 3, 2, 1, 1, 3, 2, 3]   # words per caption chunk (sums to len(WORDS))
assert sum(CHUNKS) == len(WORDS)

# b-roll cutaways (out-seconds). mode "full" = full-bleed, "card" = floating card over blurred talking head.
CUTAWAYS = [
    dict(clip="code-screen", t0=3.30, t1=SEG_OUT[1], mode="full", ss=0.0),
    dict(clip="terminal-wall", t0=6.59, t1=SEG_OUT[3], mode="card", ss=1.2),
    dict(clip="phone-typing", t0=8.90, t1=DUR, mode="card", ss=3.5),
]

UPLOAD_DONE = 0.70   # seconds into the phone cutaway when the progress bar completes

# palette
INK = (9, 12, 28); WHITE = (255, 255, 255); YELLOW = (255, 225, 74); CYAN = (0, 229, 255)
VIOLET = (124, 92, 255); GREEN = (52, 232, 140); RED = (255, 69, 84)


def src_to_out(s):
    for k, (a, b) in enumerate(SEGS):
        if a - 1e-6 <= s <= b + 1e-6:
            return SEG_OUT[k] + (s - a)
    raise ValueError(s)


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease_out(u):
    return 1 - (1 - clamp(u)) ** 3


def ease_in_out(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def ease_back(u, s=1.9):
    u = clamp(u) - 1
    return 1 + (s + 1) * u ** 3 + s * u ** 2


# ------------------------------------------------------------------ drawing helpers
@lru_cache(None)
def font(size, wght=800, mono=False):
    f = ImageFont.truetype(A("fonts", "JetBrainsMono.ttf" if mono else "Montserrat.ttf"), int(size))
    try:
        f.set_variation_by_axes([wght])
    except Exception:
        pass
    return f


@lru_cache(None)
def text_sprite(text, size, fill=WHITE, stroke=0, wght=800, shadow=0, mono=False):
    f = font(size, wght, mono)
    asc, desc = f.getmetrics()
    pad = stroke + shadow * 2 + 6
    w = int(f.getlength(text)) + 2 * pad
    h = asc + desc + 2 * pad
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    if shadow:
        sh = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(sh).text((pad, pad + shadow), text, font=f, fill=(0, 0, 0, 200), stroke_width=stroke, stroke_fill=(0, 0, 0, 200))
        img = Image.alpha_composite(img, sh.filter(ImageFilter.GaussianBlur(shadow)))
    ImageDraw.Draw(img).text((pad, pad), text, font=f, fill=fill + (255,), stroke_width=stroke, stroke_fill=(8, 8, 16, 255))
    return np.array(img)


def blit(dst, spr, cx, cy, scale=1.0, alpha=1.0):
    """alpha-composite RGBA sprite (uint8 HxWx4) centred at cx,cy onto dst (uint8 HxWx3), optional scale."""
    if alpha <= 0.003 or scale <= 0.01:
        return
    if abs(scale - 1) > 1e-3:
        h, w = spr.shape[:2]
        spr = cv2.resize(spr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    h, w = spr.shape[:2]
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if X0 >= X1 or Y0 >= Y1:
        return
    s = spr[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    a = (s[..., 3:4].astype(np.float32) / 255.0) * alpha
    d = dst[Y0:Y1, X0:X1].astype(np.float32)
    dst[Y0:Y1, X0:X1] = (d * (1 - a) + s[..., :3].astype(np.float32) * a).astype(np.uint8)


def rrect_sprite(w, h, r, fill, border=None, bw=0):
    S = 3   # supersample for smooth corners
    img = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, w * S - 1, h * S - 1], r * S, fill=fill, outline=border, width=bw * S)
    return np.array(img.resize((w, h), Image.LANCZOS))


@lru_cache(None)
def chip(text, size=40, bg=(9, 12, 28, 205), border=CYAN, fg=WHITE, left=34, right=34, mono_prefix="", wght=800):
    tx = text_sprite(text, size, fg, wght=wght)
    px = text_sprite(mono_prefix, size, border, wght=800, mono=True) if mono_prefix else None
    gap = 14 if px is not None else 0
    inner = tx.shape[1] + (px.shape[1] + gap if px is not None else 0)
    w, h = inner + left + right - 12, int(size * 2.05)
    base = rrect_sprite(w, h, h // 2, bg, border + (255,), 3)
    img = Image.fromarray(base)
    x = left - 6
    if px is not None:
        img.alpha_composite(Image.fromarray(px), (x, (h - px.shape[0]) // 2 + 1)); x += px.shape[1] + gap
    img.alpha_composite(Image.fromarray(tx), (x, (h - tx.shape[0]) // 2 + 1))
    return np.array(img)


@lru_cache(None)
def dot(color, r=12):
    S = 4
    img = Image.new("RGBA", (r * 2 * S, r * 2 * S), (0, 0, 0, 0))
    ImageDraw.Draw(img).ellipse([0, 0, r * 2 * S - 1, r * 2 * S - 1], fill=color + (255,))
    return np.array(img.resize((r * 2, r * 2), Image.LANCZOS))


@lru_cache(None)
def glow_sprite(w, h, r, color, blur=46, alpha=150):
    pad = blur * 3
    img = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle([pad, pad, pad + w, pad + h], r, fill=color + (alpha,))
    return np.array(img.filter(ImageFilter.GaussianBlur(blur)))


def bar_sprite(w, h, p, fill=CYAN, track=(255, 255, 255, 60)):
    S = 3
    img = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, w * S - 1, h * S - 1], h * S // 2, fill=track)
    fw = int((w * S) * clamp(p))
    if fw > h * S // 2:
        d.rounded_rectangle([0, 0, fw, h * S - 1], h * S // 2, fill=fill + (255,))
    return np.array(img.resize((w, h), Image.LANCZOS))


@lru_cache(None)
def check_sprite(color=GREEN, r=54):
    S = 4
    n = r * 2 * S
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([0, 0, n - 1, n - 1], fill=color + (255,))
    d.line([(n * .27, n * .52), (n * .44, n * .68), (n * .74, n * .34)], fill=(255, 255, 255, 255), width=int(n * .11), joint="curve")
    return np.array(img.resize((r * 2, r * 2), Image.LANCZOS))


# ------------------------------------------------------------------ frame-level effects
_LUT = None


def grade(fr):
    """Gentle S-curve, a touch of saturation, lifted mids for the (backlit) face."""
    global _LUT
    if _LUT is None:
        x = np.linspace(0, 1, 256)
        y = x ** 0.93
        y = y + 0.16 * (ease_in_out(np.clip(y, 0, 1)) - y)
        y = y - 0.11 * np.clip((y - 0.68) / 0.32, 0, 1) ** 2       # tame the blown backlit sky
        y = np.clip(y * 0.99 + 0.005, 0, 1)
        _LUT = (y * 255).astype(np.uint8)
    fr = cv2.LUT(fr, _LUT)
    hsv = cv2.cvtColor(fr, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.14, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def zoom(fr, z, anchor=ANCHOR):
    ax, ay = anchor
    M = np.array([[z, 0, ax - z * ax], [0, z, ay - z * ay]], np.float32)
    return cv2.warpAffine(fr, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


_YY, _XX = np.mgrid[0:H, 0:W].astype(np.float32)
_vig = 1 - 0.30 * np.clip((np.hypot((_XX - W / 2) / (W * .75), (_YY - H * .48) / (H * .70)) - .45) / .55, 0, 1) ** 1.6
_grad = 1 - 0.58 * ease_in_out(np.clip((_YY - 1050) / 850, 0, 1))          # darkens the lower third under captions
_top = 1 - 0.30 * np.clip((260 - _YY) / 260, 0, 1)
SHADE = (_vig * _grad * _top)[..., None].astype(np.float32)


def shade(fr):
    return (fr.astype(np.float32) * SHADE).astype(np.uint8)


def flash(fr, a):
    if a > 0.004:
        fr[:] = (fr.astype(np.float32) * (1 - a) + 255 * a).astype(np.uint8)


# ------------------------------------------------------------------ sources
def decode(path, ss, dur, w, h, extra=""):
    vf = f"fps={FPS},scale={w}:{h}:flags=lanczos{extra}"
    cmd = [FF, "-v", "error", "-ss", f"{ss:.3f}", "-t", f"{dur:.3f}", "-i", path, "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = w * h * 3
    try:
        while True:
            b = p.stdout.read(size)
            if len(b) < size:
                break
            yield np.frombuffer(b, np.uint8).reshape(h, w, 3).copy()
    finally:
        p.kill(); p.wait()


class Seq:
    """Frame-by-frame reader that holds the last frame if the decoder runs short."""
    def __init__(self, gen):
        self.g, self.last = gen, None

    def next(self):
        try:
            self.last = next(self.g)
        except StopIteration:
            pass
        return self.last


# ------------------------------------------------------------------ captions
def build_chunks():
    ws = [(t, src_to_out(a), src_to_out(b), k) for t, a, b, k in WORDS]
    out, i = [], 0
    for n in CHUNKS:
        out.append(ws[i:i + n]); i += n
    res = []
    for ci, c in enumerate(out):
        start = c[0][1] - 0.04
        nxt = out[ci + 1][0][1] - 0.04 if ci + 1 < len(out) else DUR
        end = min(nxt, c[-1][2] + 0.30) if ci + 1 < len(out) else DUR
        res.append(dict(words=c, t0=start, t1=end))
    return res


CHUNKS_T = build_chunks()
CAP_Y = 1430


def draw_caption(fr, t):
    for ch in CHUNKS_T:
        if not (ch["t0"] <= t < ch["t1"]):
            continue
        words = ch["words"]
        size = 104
        gap = lambda s: int(s * 0.34)
        while True:
            widths = [font(size).getlength(w[0]) + 30 for w in words]
            total = sum(widths) + gap(size) * (len(words) - 1)
            if total <= 860 or size <= 64:
                break
            size -= 4
        u = t - ch["t0"]
        chunk_scale = 0.72 + 0.28 * ease_back(u / 0.16)
        a_in = clamp(u / 0.07)
        a_out = clamp((ch["t1"] - t) / 0.06) if ch["t1"] >= DUR - 1e-6 else 1.0   # only the last chunk fades; others hand off directly
        x = 540 - total / 2
        # last-started word is the active one
        started = [i for i, w in enumerate(words) if t >= w[1] - 0.02]
        active = started[-1] if started else -1
        for i, (txt, ws, we, key) in enumerate(words):
            wpx = widths[i]
            cx = x + wpx / 2
            x += wpx + gap(size)
            is_act = i == active
            col = YELLOW if (is_act or (key and t >= ws)) else WHITE
            spr = text_sprite(txt, size, col, stroke=int(size * 0.11), shadow=10)
            pop = 1.0
            if is_act:
                pop = 1.05 + 0.11 * (1 - ease_out((t - ws) / 0.22))
            if key and t >= ws:
                pop *= 1.03
            # position relative to chunk centre so the whole chunk scales together
            dx = (cx - 540) * chunk_scale
            blit(fr, spr, 540 + dx, CAP_Y + (1 - ease_out(u / 0.18)) * 26, chunk_scale * pop, a_in * a_out)
        return


# ------------------------------------------------------------------ motion graphics
def draw_badge(fr, t):
    """'LIVE TEST' pill (top-left) with a pulsing red dot."""
    t0, t1 = 0.25, 1.55
    if not (t0 <= t < t1):
        return
    sp = chip("LIVE TEST", 36, border=(255, 255, 255), left=66, right=30)
    u_in, u_out = (t - t0) / 0.45, (t1 - t) / 0.2
    sx = 56 + sp.shape[1] / 2 - (1 - ease_back(u_in)) * 420
    a = clamp(u_out)
    blit(fr, sp, sx, 238, 1.0, a)
    pulse = 0.55 + 0.45 * math.sin((t - t0) * 9.0)
    blit(fr, dot(RED, 12), sx - sp.shape[1] / 2 + 38, 238, 1.0 + 0.25 * pulse, a)


def draw_title_opus(fr, t):
    """Big 'OPUS 5.5' title plate that lands on the spoken word."""
    t0, t1 = src_to_out(1.90) - 0.02, 3.20
    if not (t0 <= t < t1):
        return
    u = t - t0
    txt = text_sprite("OPUS 5.5", 118, WHITE, stroke=0, wght=900, shadow=8)
    pw, ph = txt.shape[1] + 90, 168
    plate = Image.fromarray(rrect_sprite(pw, ph, 42, (255, 255, 255, 255)))
    grad = np.zeros((ph, pw, 4), np.uint8)
    gx = np.linspace(0, 1, pw)[None, :, None]
    c1, c2 = np.array(VIOLET), np.array(CYAN)
    grad[..., :3] = (c1 * (1 - gx) + c2 * gx).astype(np.uint8)
    grad[..., 3] = np.array(plate)[..., 3]
    # diagonal shine sweep
    sweep = clamp(u / 0.55)
    sx = -pw * .3 + sweep * pw * 1.6
    yy, xx = np.mgrid[0:ph, 0:pw]
    band = np.exp(-(((xx - sx) - (yy - ph / 2) * .6) / 34.0) ** 2)[..., None]
    grad[..., :3] = np.clip(grad[..., :3] + band * 90, 0, 255).astype(np.uint8)
    img = Image.fromarray(grad)
    img.alpha_composite(Image.fromarray(txt), ((pw - txt.shape[1]) // 2, (ph - txt.shape[0]) // 2 + 2))
    spr = np.array(img)
    sc = 0.35 + 0.65 * ease_back(u / 0.32)
    a = clamp((t1 - t) / 0.18) * clamp(u / 0.05)
    cy = 268 - (1 - clamp((t1 - t) / 0.25)) * 40
    blit(fr, glow_sprite(pw, ph, 42, VIOLET, 40, 120), 540, cy, sc, a * 0.9)
    blit(fr, spr, 540, cy, sc, a)


def draw_chip_top(fr, t, t0, t1, text, prefix="", color=CYAN, y=280, size=44):
    if not (t0 <= t < t1):
        return
    sp = chip(text, size, border=color, mono_prefix=prefix)
    u = (t - t0) / 0.35
    a = clamp((t1 - t) / 0.12)
    blit(fr, sp, 540, y - (1 - ease_out(u)) * 50, 0.8 + 0.2 * ease_back(u), a * clamp(u * 4))


def cutaway_frame(cw, b, head, lt, fr_len):
    """Base layer for a cutaway: full-bleed b-roll with a punch-in, or a blurred/dimmed talking head backdrop."""
    if cw["mode"] == "full":
        z = 1.0 + 0.18 * (1 - ease_out(lt / 0.40)) + 0.05 * (lt / fr_len)
        return grade(zoom(b, z, (540, 960)))
    small = cv2.resize(head, (216, 384), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.GaussianBlur(small, (0, 0), 5), (W, H), interpolation=cv2.INTER_CUBIC)
    return (bg.astype(np.float32) * np.array([0.40, 0.44, 0.62], np.float32)).astype(np.uint8)


CARD_W, CARD_H, CARD_CY = 1000, 562, 780


def draw_card(fr, cw, b, lt, fr_len):
    # slow push-in; the terminal wall is framed tighter (anchored high) to crop out the monitor's brand logo
    base, ay = (1.40, CARD_H * 0.20) if cw["clip"] == "terminal-wall" else (1.0, CARD_H / 2)
    zk = base + 0.07 * (lt / fr_len)
    b = cv2.warpAffine(
        b, np.array([[zk, 0, CARD_W / 2 - zk * CARD_W / 2], [0, zk, ay - zk * ay]], np.float32), (CARD_W, CARD_H),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    b = grade(b)
    mask = rrect_sprite(CARD_W, CARD_H, 44, (255, 255, 255, 255))[..., 3]
    spr = np.dstack([b, mask])
    frame_border = rrect_sprite(CARD_W, CARD_H, 44, (0, 0, 0, 0), (255, 255, 255, 90), 3)
    spr = np.array(Image.alpha_composite(Image.fromarray(spr), Image.fromarray(frame_border)))
    sc = 0.80 + 0.20 * ease_back(lt / 0.34)
    a = clamp(lt / 0.10) * clamp((fr_len - lt) / 0.08)
    gcol = CYAN if cw["clip"] == "terminal-wall" else VIOLET
    blit(fr, glow_sprite(CARD_W, CARD_H, 44, gcol, 50, 130), 540, CARD_CY, sc, a)
    blit(fr, spr, 540, CARD_CY, sc, a)


def draw_hud_autonomous(fr, t, lt, fr_len):
    """Chip + progress bar + PLAN/CODE/TEST steps. Illustrative motion graphic (not real telemetry)."""
    a_out = clamp((fr_len - lt) / 0.10)
    u = lt / 0.35
    sp = chip("AUTONOMOUS MODE", 40, border=CYAN, left=70, right=32)
    y = 420 - (1 - ease_out(u)) * 40
    blit(fr, sp, 540, y, 0.85 + 0.15 * ease_back(u), a_out * clamp(u * 4))
    blit(fr, dot(GREEN, 11), 540 - sp.shape[1] / 2 + 42, y, 1.0 + 0.3 * (0.5 + 0.5 * math.sin(lt * 10)), a_out * clamp(u * 4))
    # progress
    p = ease_in_out(clamp((lt - 0.15) / (fr_len - 0.35)))
    bw = 860
    blit(fr, bar_sprite(bw, 22, p, CYAN), 540, 1150, 1.0, a_out * clamp(lt / 0.25))
    pct = text_sprite(f"{int(p * 100):d}%", 46, WHITE, wght=800, shadow=4)
    blit(fr, pct, 540 + bw / 2 - pct.shape[1] / 2, 1105, 1.0, a_out * clamp(lt / 0.25))
    lab = text_sprite("RUNNING TASK", 30, (190, 210, 255), wght=700, shadow=4)
    blit(fr, lab, 540 - bw / 2 + lab.shape[1] / 2, 1105, 1.0, a_out * clamp(lt / 0.25))
    for k, name in enumerate(["PLAN", "CODE", "TEST"]):
        t_k = 0.30 + k * 0.42
        if lt >= t_k:
            s = chip(name, 32, border=GREEN, left=52, right=26, bg=(9, 22, 20, 210))
            uu = (lt - t_k) / 0.25
            cx = 540 + (k - 1) * 290
            blit(fr, s, cx, 1232, 0.6 + 0.4 * ease_back(uu), a_out * clamp(uu * 5))
            blit(fr, dot(GREEN, 8), cx - s.shape[1] / 2 + 32, 1232, 0.6 + 0.4 * ease_back(uu), a_out * clamp(uu * 5))


def draw_hud_upload(fr, t, lt, fr_len):
    a_out = clamp((fr_len - lt) / 0.06)
    u = lt / 0.30
    sp = chip("UPLOADING FROM PHONE", 38, border=VIOLET, left=70, right=32)
    y = 420 - (1 - ease_out(u)) * 40
    blit(fr, sp, 540, y, 0.85 + 0.15 * ease_back(u), a_out * clamp(u * 4))
    blit(fr, dot(VIOLET, 11), 540 - sp.shape[1] / 2 + 42, y, 1.0 + 0.3 * (0.5 + 0.5 * math.sin(lt * 12)), a_out * clamp(u * 4))
    p = ease_in_out(clamp((lt - 0.05) / (fr_len - 0.45)))
    bw = 860
    blit(fr, bar_sprite(bw, 22, p, VIOLET), 540, 1150, 1.0, a_out * clamp(lt / 0.2))
    done = p >= 0.999
    label = "UPLOADED" if done else f"{int(p * 100):d}%"
    pct = text_sprite(label, 46, GREEN if done else WHITE, wght=800, shadow=4)
    blit(fr, pct, 540 + bw / 2 - pct.shape[1] / 2, 1105, 1.0, a_out * clamp(lt / 0.2))
    if done:
        uu = (lt - (fr_len - 0.45)) / 0.3
        blit(fr, check_sprite(GREEN, 54), 540, 1235, 0.4 + 0.6 * ease_back(uu), a_out * clamp(uu * 6))


# ------------------------------------------------------------------ audio
def read_pcm(path, ch=1, ss=None, dur=None):
    cmd = [FF, "-v", "error"] + (["-ss", str(ss)] if ss else []) + (["-t", str(dur)] if dur else []) + ["-i", path, "-ac", str(ch), "-ar", str(SR), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.float32).reshape(-1, ch)


def write_wav(path, x):
    x = np.clip(x, -1, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(x.shape[1]); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((x * 32767).astype("<i2").tobytes())


def synth_sfx():
    rng = np.random.default_rng(7)

    def whoosh(d=0.45, f0=500, f1=5000):
        n = int(d * SR); t = np.linspace(0, 1, n)
        noise = rng.standard_normal(n)
        # swept band-pass via cumulative-phase filtered noise (simple one-pole sweep)
        fc = f0 * (f1 / f0) ** t
        a = np.exp(-2 * np.pi * fc / SR)
        y = np.zeros(n); lp = 0.0; lp2 = 0.0
        for i in range(n):
            lp = (1 - a[i]) * noise[i] + a[i] * lp
            lp2 = (1 - a[i]) * lp + a[i] * lp2
            y[i] = noise[i] * 0.0 + (lp - lp2) * 6
        env = np.sin(np.pi * t) ** 1.6
        return (y * env * 0.55).astype(np.float32)

    def pop(f0=1100, f1=380, d=0.10):
        n = int(d * SR); t = np.arange(n) / SR
        ph = 2 * np.pi * np.cumsum(f1 + (f0 - f1) * np.exp(-t * 45)) / SR
        return (np.sin(ph) * np.exp(-t * 38) * 0.6).astype(np.float32)

    def boom(d=0.5):
        n = int(d * SR); t = np.arange(n) / SR
        return (np.sin(2 * np.pi * (46 + 60 * np.exp(-t * 14)) * t) * np.exp(-t * 7) * 0.9).astype(np.float32)

    def ding(d=0.7):
        n = int(d * SR); t = np.arange(n) / SR
        return ((np.sin(2 * np.pi * 1318 * t) + 0.5 * np.sin(2 * np.pi * 1975 * t) + 0.25 * np.sin(2 * np.pi * 2637 * t)) * np.exp(-t * 6) * 0.28).astype(np.float32)

    raw = dict(whoosh=whoosh(), whoosh_s=whoosh(0.28, 900, 6500), pop=pop(), boom=boom(), ding=ding())
    # level each effect against the (loud, limited) voice: noisy sweeps by RMS, transients by peak
    target = dict(whoosh=("rms", -24), whoosh_s=("rms", -25), pop=("peak", -14), boom=("peak", -10), ding=("rms", -25))
    out = {}
    for k, x in raw.items():
        mode, db = target[k]
        cur = np.sqrt((x.astype(np.float64) ** 2).mean()) if mode == "rms" else np.abs(x).max()
        out[k] = (x * (10 ** (db / 20) / cur)).astype(np.float32)
    return out


def build_audio(src, path_out):
    voice = read_pcm(src, 1)[:, 0]
    fade = int(0.006 * SR)
    parts = []
    for a, b in SEGS:
        s = voice[int(a * SR):int(b * SR)].copy()
        s[:fade] *= np.linspace(0, 1, fade); s[-fade:] *= np.linspace(1, 0, fade)
        parts.append(s)
    v = np.concatenate(parts)
    tmp = os.path.join(HERE, "work"); os.makedirs(tmp, exist_ok=True)
    write_wav(os.path.join(tmp, "voice_cut.wav"), np.stack([v, v], 1))
    # clean + polish the voice: rumble cut, broadband denoise, warmth + presence, compress, limit
    chain = ("highpass=f=85,afftdn=nr=16:nf=-30:tn=1,equalizer=f=170:t=q:w=1:g=2.0,equalizer=f=3200:t=q:w=1.1:g=3.5,"
             "equalizer=f=9000:t=h:w=2000:g=-2,acompressor=threshold=-22dB:ratio=3.2:attack=6:release=90:makeup=5,alimiter=limit=0.9")
    subprocess.run([FF, "-y", "-v", "error", "-i", os.path.join(tmp, "voice_cut.wav"), "-af", chain, "-ar", str(SR), os.path.join(tmp, "voice_proc.wav")], check=True)
    v = read_pcm(os.path.join(tmp, "voice_proc.wav"), 1)[:, 0]
    total = int(DUR * SR)
    v = np.pad(v, (0, max(0, total - len(v))))[:total]
    # music: loop-free, start at 0, ducked under voice
    m = read_pcm(A("music", "minimal-techno-01.mp3"), 2)[:total]
    win = int(0.02 * SR)
    rms = np.sqrt(np.convolve(v ** 2, np.ones(win) / win, "same"))
    active = (rms > 0.03).astype(np.float64)
    att, rel = int(0.03 * SR), int(0.28 * SR)
    env = np.zeros_like(active); cur = 0.0
    for i in range(0, len(active), 48):     # coarse steps are plenty for a smoothing envelope
        tgt = active[i]
        k = 1 / att if tgt > cur else 1 / rel
        cur += (tgt - cur) * min(1.0, k * 48)
        env[i:i + 48] = cur
    duck = 1 - 0.50 * env
    tt = np.arange(total) / SR
    mgain = 0.12 * duck * np.clip(tt / 0.15, 0, 1) * np.clip((DUR - tt) / 0.9, 0, 1)
    mix = np.stack([v, v], 1) + m * mgain[:, None]
    # sfx
    sfx = synth_sfx()
    cues = [(0.30, "pop", .55), (src_to_out(1.90) - 0.02, "boom", .7), (src_to_out(1.90) - 0.02, "pop", .5)]
    for cw in CUTAWAYS:
        cues.append((cw["t0"] - 0.18, "whoosh", .8))
    cues += [(SEG_OUT[1] - 0.02, "whoosh_s", .5), (SEG_OUT[3] - 0.02, "whoosh_s", .5), (CUTAWAYS[1]["t0"] + 0.30, "pop", .45)]
    cues += [(CUTAWAYS[1]["t0"] + 0.30 + 0.42 * k + 0.0, "pop", .35) for k in range(1, 3)]
    cues.append((CUTAWAYS[2]["t0"] + UPLOAD_DONE, "ding", .9))
    for tc, name, g in cues:
        s = sfx[name] * g
        i0 = max(0, int(tc * SR)); seg = s[:max(0, total - i0)]
        mix[i0:i0 + len(seg)] += seg[:, None]
    mix *= 0.6   # headroom: loudnorm below brings it back up to -14 LUFS without clipping
    write_wav(os.path.join(tmp, "mix_pre.wav"), mix)
    # loudness to social-feed standard (-14 LUFS, -1.5 dBTP)
    subprocess.run([FF, "-y", "-v", "error", "-i", os.path.join(tmp, "mix_pre.wav"), "-af", "loudnorm=I=-14:TP=-1.5:LRA=9", "-ar", str(SR), path_out], check=True)


# ------------------------------------------------------------------ main render
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "final_9x16.mp4"))
    ap.add_argument("--stills", default="")
    ap.add_argument("--noencode", action="store_true")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    work = os.path.join(HERE, "work"); os.makedirs(work, exist_ok=True)
    audio = os.path.join(work, "mix.wav")
    if not (args.noencode and os.path.exists(audio)):
        build_audio(args.src, audio)
    stills = {round(float(x) * FPS) for x in args.stills.split(",") if x}

    # b-roll readers
    cws = []
    for cw in CUTAWAYS:
        path = A("broll", cw["clip"] + ".mp4")
        n = round((cw["t1"] - cw["t0"]) * FPS) + 2
        if cw["mode"] == "full":
            gen = decode(path, cw["ss"], n / FPS + .1, W, H)
        else:
            gen = decode(path, cw["ss"], n / FPS + .1, CARD_W, CARD_H)
        cws.append(dict(cw, seq=Seq(gen), f0=round(cw["t0"] * FPS), f1=round(cw["t1"] * FPS)))

    enc = None
    if not args.noencode:
        cmd = [FF, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", audio,
               "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
               "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-profile:v", "high", "-level", "4.2", "-r", str(FPS),
               "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
               "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest", "-movflags", "+faststart", args.out]
        enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    seg_of_frame = np.repeat(np.arange(len(SEGS)), SEG_FR)
    heads = {}
    cut_frames = [round(s * FPS) for s in SEG_OUT[1:]]
    for i in range(N):
        t = i / FPS
        k = int(seg_of_frame[i])
        if k not in heads:
            a, b = SEGS[k]
            heads[k] = Seq(decode(args.src, a, (b - a) + 0.15, W, H))
        raw = heads[k].next()
        j = i - round(SEG_OUT[k] * FPS)
        z0, z1 = ZOOM[k]
        z = z0 + (z1 - z0) * ease_in_out(j / max(1, SEG_FR[k] - 1))
        head = grade(zoom(raw, z))

        frame = head
        active_cw = next((c for c in cws if c["f0"] <= i < c["f1"]), None)
        if active_cw:
            lt = (i - active_cw["f0"]) / FPS
            fr_len = (active_cw["f1"] - active_cw["f0"]) / FPS
            b = active_cw["seq"].next()
            frame = cutaway_frame(active_cw, b, head, lt, fr_len)
            if active_cw["mode"] == "card":
                draw_card(frame, active_cw, b, lt, fr_len)
        frame = shade(frame)

        # graphics
        if active_cw is None or active_cw["mode"] == "full":
            draw_badge(frame, t)
            draw_title_opus(frame, t)
        if active_cw:
            lt = (i - active_cw["f0"]) / FPS
            fr_len = (active_cw["f1"] - active_cw["f0"]) / FPS
            if active_cw["clip"] == "code-screen":
                draw_chip_top(frame, t, active_cw["t0"] + 0.05, active_cw["t1"], "CLAUDE CODE VM" if CLAUDE_CODE else "CLOUD CODE VM", "›_", CYAN, 290, 46)
            elif active_cw["clip"] == "terminal-wall":
                draw_hud_autonomous(frame, t, lt, fr_len)
            else:
                draw_hud_upload(frame, t, lt, fr_len)
        draw_caption(frame, t)

        # transition flashes (cutaway in/out + jump cuts)
        fa = 0.0
        for c in cws:
            for edge, amp in ((c["f0"], 0.55), (c["f1"], 0.25)):
                d = i - edge
                if 0 <= d < 5:
                    fa = max(fa, amp * (1 - d / 5) ** 2)
        for cf in cut_frames:
            d = i - cf
            if 0 <= d < 4:
                fa = max(fa, 0.16 * (1 - d / 4) ** 2)
        flash(frame, fa)

        if i in stills:
            Image.fromarray(frame).save(os.path.join(work, f"still_{i / FPS:05.2f}.png"))
        if enc:
            enc.stdin.write(frame.tobytes())
        if i % 30 == 0:
            print(f"frame {i}/{N}", flush=True)
    if enc:
        enc.stdin.close(); enc.wait()
        print("wrote", args.out)


if __name__ == "__main__":
    main()
