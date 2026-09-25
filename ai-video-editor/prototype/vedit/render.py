"""Compile a Timeline into pixels.

Real product: a GPU compositor (Rust + wgpu) pulling decoded frames.
Prototype: compile the timeline into an FFmpeg filter graph. Same idea:
the timeline is a program and the renderer is its compiler.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

from .motion import FONT_BOLD, render_sequence
from .timeline import Timeline, frame_to_sec, tc

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def caption_words(tl: Timeline, index: dict) -> list[dict]:
    """Derive captions from the edit: map each transcript word that survives in a V1
    clip from SOURCE time to RECORD time. Re-cut the edit -> captions follow."""
    out = []
    for it in sorted(tl.track("V1").items, key=lambda i: i.rec_in):
        for w in index["media"][it.media].get("words", []):
            if it.src_in <= w["s"] and w["e"] <= it.src_out:
                out.append({"w": w["w"], "s": it.rec_in + w["s"] - it.src_in,
                            "e": it.rec_in + w["e"] - it.src_in})
    return out


def render(tl: Timeline, media_lib: dict, index: dict, out: Path, work: Path) -> Path:
    fps = tl.fps
    W, H = tl.width, tl.height
    sec = lambda f: f"{frame_to_sec(f, fps):.6f}"
    work.mkdir(parents=True, exist_ok=True)
    inputs: list[list[str]] = []
    g: list[str] = []

    def add_input(*args):
        inputs.append(list(args))
        return len(inputs) - 1

    norm = f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps}"

    # --- V1: the story track. Trim each source range, concat video+audio.
    v1 = sorted(tl.track("V1").items, key=lambda i: i.rec_in)
    for k, it in enumerate(v1):
        i = add_input("-i", media_lib[it.media]["path"])
        g.append(f"[{i}:v]trim=start_frame={it.src_in}:end_frame={it.src_out},setpts=PTS-STARTPTS,{norm}[v{k}]")
        g.append(f"[{i}:a]atrim=start={sec(it.src_in)}:end={sec(it.src_out)},asetpts=PTS-STARTPTS[a{k}]")
    g.append("".join(f"[v{k}][a{k}]" for k in range(len(v1))) + f"concat=n={len(v1)}:v=1:a=1[base][dlg]")
    cur = "base"

    # --- V2: b-roll cutaways laid over V1 (dialogue keeps playing underneath).
    for k, it in enumerate(sorted(tl.track("V2").items, key=lambda i: i.rec_in)):
        i = add_input("-i", media_lib[it.media]["path"])
        g.append(f"[{i}:v]trim=start_frame={it.src_in}:end_frame={it.src_out},setpts=PTS-STARTPTS+{sec(it.rec_in)}/TB,{norm}[b{k}]")
        g.append(f"[{cur}][b{k}]overlay=eof_action=pass:enable='between(t,{sec(it.rec_in)},{sec(it.rec_out - 1)})'[o{k}]")
        cur = f"o{k}"

    # --- G1: motion graphics, rendered by our motion engine to RGBA frames.
    gfx = list(tl.track("G1").items)
    caps = caption_words(tl, index)
    if caps:  # captions are a derived graphics layer spanning the whole edit
        from .timeline import Item
        gfx.append(Item("captions", "motion", 0, tl.track("V1").items and max(i.rec_out for i in v1),
                        params={"template": "captions", "props": {"words": caps}}))
    for k, it in enumerate(gfx):
        d = work / f"gfx_{it.id}"
        shutil.rmtree(d, ignore_errors=True)
        render_sequence(it.params["template"], it.params["props"], W, H, it.duration, d)
        i = add_input("-framerate", str(fps), "-i", str(d / "%05d.png"))
        g.append(f"[{i}:v]format=rgba,setpts=PTS-STARTPTS+{sec(it.rec_in)}/TB[gf{k}]")
        g.append(f"[{cur}][gf{k}]overlay=eof_action=pass:format=auto[go{k}]")
        cur = f"go{k}"

    # --- Audio: dialogue + music bed ducked by a sidechain compressor keyed on dialogue.
    music = tl.track("M1").items
    if music:
        m = music[0]
        i = add_input("-i", media_lib[m.media]["path"])
        g.append(f"[{i}:a]atrim=start={sec(m.src_in)}:duration={sec(m.duration)},asetpts=PTS-STARTPTS,"
                 f"volume={m.params.get('gain_db', -8)}dB,afade=t=out:st={frame_to_sec(m.duration, fps) - 1.5:.3f}:d=1.5[mus]")
        g.append("[dlg]asplit=2[dlg1][sc]")
        g.append("[mus][sc]sidechaincompress=threshold=0.02:ratio=10:attack=15:release=350[duck]")
        g.append("[dlg1][duck]amix=inputs=2:duration=first:normalize=0,loudnorm=I=-16:TP=-1.5[aout]")
    else:
        g.append("[dlg]loudnorm=I=-16:TP=-1.5[aout]")
    g.append(f"[{cur}]format=yuv420p[vout]")

    script = work / "graph.txt"
    script.write_text(";\n".join(g))
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error"]
    for a in inputs:
        cmd += a
    cmd += ["-filter_complex_script", str(script), "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "160k",
            "-r", str(fps), str(out)]
    subprocess.run(cmd, check=True)
    return out


def contact_sheet(video: Path, tl: Timeline, frames: list[int], out: Path, cols=4) -> Path:
    """A strip of stills (with record timecode) so a human, or a VLM, can review the cut."""
    thumbs = []
    for f in frames:
        p = out.parent / f"_thumb_{f}.png"
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(video), "-vf",
                        f"select=eq(n\\,{f}),scale=320:-1", "-frames:v", "1", str(p)], check=True)
        im = Image.open(p).convert("RGB")
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 130, 22], fill=(0, 0, 0))
        d.text((6, 3), tc(f, tl.fps), font=ImageFont.truetype(FONT_BOLD, 14), fill=(255, 255, 255))
        thumbs.append(im)
        p.unlink()
    w, h = thumbs[0].size
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w + (cols + 1) * 6, rows * h + (rows + 1) * 6), (30, 30, 30))
    for n, im in enumerate(thumbs):
        sheet.paste(im, (6 + (n % cols) * (w + 6), 6 + (n // cols) * (h + 6)))
    sheet.save(out)
    return out
