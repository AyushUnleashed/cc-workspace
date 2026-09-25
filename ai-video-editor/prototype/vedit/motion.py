"""Motion graphics = data, rendered by our own engine.

Three layers:
  1. TEMPLATES   (designer-made, versioned): props -> scene spec
  2. SCENE SPEC  (JSON): layers + animated properties (keyframes, easing, springs)
  3. RENDERER    (ours): scene spec + frame number -> RGBA image

The agent almost never writes keyframes by hand. It picks a template and fills
props ("lower_third", name="Maya Chen", title="Co-founder"). A designer's
template guarantees taste; the agent supplies content and timing.

Pillow stands in for the real GPU renderer (Skia / Vello / wgpu) here.
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


# ------------------------------------------------------------------ easing
def ease(name: str, p: float) -> float:
    p = min(max(p, 0.0), 1.0)
    if name == "linear":
        return p
    if name == "out_cubic":
        return 1 - (1 - p) ** 3
    if name == "in_cubic":
        return p ** 3
    if name == "spring":  # damped spring settling at 1 (slight overshoot)
        zeta, omega = 0.45, 12.0
        wd = omega * math.sqrt(1 - zeta ** 2)
        return 1 - math.exp(-zeta * omega * p) * (math.cos(wd * p) + zeta * omega / wd * math.sin(wd * p))
    raise ValueError(name)


def sample(track, t: int):
    """track: constant or [{"t": frame, "v": value, "ease": name}, ...]"""
    if not isinstance(track, list):
        return track
    if t <= track[0]["t"]:
        return track[0]["v"]
    for a, b in zip(track, track[1:]):
        if a["t"] <= t <= b["t"]:
            p = (t - a["t"]) / max(1, b["t"] - a["t"])
            e = ease(b.get("ease", "linear"), p)
            return a["v"] + (b["v"] - a["v"]) * e
    return track[-1]["v"]


# ------------------------------------------------------------------ templates
def lower_third(props, w, h, dur):
    name, title = props["name"], props.get("title", "")
    accent = props.get("accent", "#FF5A36")
    out_t = dur - 12
    x0 = int(w * 0.06)
    y0 = int(h * 0.72)
    slide = [{"t": 0, "v": -w * 0.5}, {"t": 18, "v": 0, "ease": "spring"},
             {"t": out_t, "v": 0}, {"t": dur, "v": -w * 0.5, "ease": "in_cubic"}]
    fade = [{"t": 0, "v": 0}, {"t": 8, "v": 1}, {"t": out_t + 4, "v": 1}, {"t": dur, "v": 0}]
    return {"layers": [
        {"type": "rect", "x": x0, "y": y0, "w": int(w * 0.36), "h": int(h * 0.14),
         "fill": "#111111", "opacity": fade, "dx": slide, "radius": 10},
        {"type": "rect", "x": x0, "y": y0, "w": 10, "h": int(h * 0.14),
         "fill": accent, "opacity": fade, "dx": slide},
        {"type": "text", "x": x0 + 28, "y": y0 + 12, "text": name, "size": int(h * 0.05),
         "font": FONT_BOLD, "fill": "#FFFFFF", "opacity": fade,
         "dx": [{"t": 0, "v": -w * 0.5}, {"t": 22, "v": 0, "ease": "spring"},
                {"t": out_t, "v": 0}, {"t": dur, "v": -w * 0.5, "ease": "in_cubic"}]},
        {"type": "text", "x": x0 + 28, "y": y0 + 12 + int(h * 0.06), "text": title,
         "size": int(h * 0.032), "font": FONT, "fill": "#BBBBBB", "opacity": fade,
         "dx": [{"t": 0, "v": -w * 0.5}, {"t": 26, "v": 0, "ease": "spring"},
                {"t": out_t, "v": 0}, {"t": dur, "v": -w * 0.5, "ease": "in_cubic"}]},
    ]}


def title_card(props, w, h, dur):
    text = props["text"]
    sub = props.get("subtitle", "")
    fade_bg = [{"t": 0, "v": 0}, {"t": 6, "v": 0.85}, {"t": dur - 8, "v": 0.85}, {"t": dur, "v": 0}]
    return {"layers": [
        {"type": "rect", "x": 0, "y": 0, "w": w, "h": h, "fill": "#000000", "opacity": fade_bg},
        {"type": "text", "x": w // 2, "y": int(h * 0.42), "anchor": "mm", "text": text,
         "size": int(h * 0.1), "font": FONT_BOLD, "fill": "#FFFFFF",
         "scale": [{"t": 0, "v": 0.6}, {"t": 16, "v": 1.0, "ease": "spring"}],
         "opacity": [{"t": 0, "v": 0}, {"t": 8, "v": 1}, {"t": dur - 8, "v": 1}, {"t": dur, "v": 0}]},
        {"type": "text", "x": w // 2, "y": int(h * 0.56), "anchor": "mm", "text": sub,
         "size": int(h * 0.04), "font": FONT, "fill": props.get("accent", "#FF5A36"),
         "dy": [{"t": 6, "v": 30}, {"t": 22, "v": 0, "ease": "out_cubic"}],
         "opacity": [{"t": 6, "v": 0}, {"t": 18, "v": 1}, {"t": dur - 8, "v": 1}, {"t": dur, "v": 0}]},
    ]}


def captions(props, w, h, dur):
    """Word-by-word captions. props.words = [{"w": str, "s": frame, "e": frame}] on the
    RECORD timeline, so they're recomputed whenever the edit changes."""
    return {"layers": [{"type": "captions", "words": props["words"], "size": int(h * 0.06),
                        "y": int(h * 0.86), "highlight": props.get("accent", "#FFD400")}]}


TEMPLATES = {"lower_third": lower_third, "title_card": title_card, "captions": captions}


# ------------------------------------------------------------------ renderer
_font_cache: dict = {}


def _font(path, size):
    key = (path, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(path, size)
    return _font_cache[key]


def _hex(c, alpha):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4)) + (int(255 * alpha),)


def render_frame(scene, t, w, h) -> Image.Image:
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    for L in scene["layers"]:
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        op = sample(L.get("opacity", 1), t)
        if op <= 0.001 and L["type"] != "captions":
            continue
        dx, dy = sample(L.get("dx", 0), t), sample(L.get("dy", 0), t)
        if L["type"] == "rect":
            x, y = L["x"] + dx, L["y"] + dy
            d.rounded_rectangle([x, y, x + L["w"], y + L["h"]], radius=L.get("radius", 0),
                                fill=_hex(L["fill"], op))
        elif L["type"] == "text":
            size = max(1, int(L["size"] * sample(L.get("scale", 1), t)))
            d.text((L["x"] + dx, L["y"] + dy), L["text"], font=_font(L["font"], size),
                   fill=_hex(L["fill"], op), anchor=L.get("anchor", "la"))
        elif L["type"] == "captions":
            _draw_captions(d, L, t, w)
        img = Image.alpha_composite(img, layer)
    return img


def _draw_captions(d, L, t, w, window=4):
    words = L["words"]
    cur = next((i for i, x in enumerate(words) if x["s"] <= t < x["e"]), None)
    if cur is None:
        return
    start = (cur // window) * window          # show words in chunks of `window`
    chunk = words[start:start + window]
    font = _font(FONT_BOLD, L["size"])
    texts = [x["w"].upper() for x in chunk]
    widths = [d.textlength(s + " ", font=font) for s in texts]
    x = (w - sum(widths)) / 2
    for i, (s, wd) in enumerate(zip(texts, widths)):
        active = start + i == cur
        pop = 1.0 + 0.12 * (1 - ease("out_cubic", (t - chunk[i]["s"]) / 5)) if active else 1.0
        f = _font(FONT_BOLD, int(L["size"] * pop))
        fill = _hex(L["highlight"], 1) if active else (255, 255, 255, 255)
        d.text((x, L["y"]), s, font=f, fill=fill, anchor="ls", stroke_width=4, stroke_fill=(0, 0, 0, 255))
        x += wd


def render_sequence(template: str, props: dict, w: int, h: int, dur: int, out_dir: Path) -> Path:
    scene = TEMPLATES[template](props, w, h, dur)
    out_dir.mkdir(parents=True, exist_ok=True)
    for t in range(dur):
        render_frame(scene, t, w, h).save(out_dir / f"{t:05d}.png", compress_level=1)
    return out_dir
