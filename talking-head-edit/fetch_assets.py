#!/usr/bin/env python3
"""Crawl Mixkit for free-licence b-roll + music and download the picks used by build.py.

    python3 fetch_assets.py            # crawl categories (candidates.json) then download the picks
    python3 fetch_assets.py --crawl    # crawl only

Only items marked "Free License" are kept. Items marked "Restricted License" are skipped.
Credits/licence info for everything downloaded lands in assets/credits.json.
"""
import json, os, re, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
UA = {"User-Agent": "Mozilla/5.0"}

# script beat -> Mixkit categories worth crawling
BEATS = {
    "claude-code-vm": ["code", "programming"],
    "autonomous":     ["hacker", "data", "technology"],
    "phone-upload":   ["phone", "typing-on-laptop"],
}
MUSIC_CATEGORY = "tech-house"

# what build.py uses (chosen by eye from the crawl)
PICKS_VIDEO = {"41654": "code-screen", "50748": "terminal-wall", "4915": "phone-typing"}
PICKS_MUSIC = {"162": "minimal-techno-01"}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


def page(url):
    return get(url).decode("utf-8", "ignore")


def crawl_video(category, limit=12):
    html = page(f"https://mixkit.co/free-stock-video/{category}/")
    items, seen = [], set()
    for _, slug, vid in re.findall(r'href="(/free-stock-video/([a-z0-9\-]+?)-(\d+)/)"', html):
        if vid in seen or slug == "new":
            continue
        seen.add(vid)
        items.append((vid, slug))
    out = []
    for vid, slug in items[:limit]:
        h = page(f"https://mixkit.co/free-stock-video/{slug}-{vid}/")
        lic = "Free License" if "Stock Video Free License" in h else "Restricted License"
        mp4 = re.findall(rf"https://assets\.mixkit\.co/videos/{vid}/{vid}-720\.mp4", h)
        if lic == "Free License" and mp4:
            out.append({"id": vid, "slug": slug, "license": lic, "url": mp4[0],
                        "page": f"https://mixkit.co/free-stock-video/{slug}-{vid}/"})
    return out


def crawl_music(category):
    html = page(f"https://mixkit.co/free-stock-music/{category}/")
    out = []
    for m in re.finditer(r'"name":"([^"]+)","genre":"[^"]*","byArtist":"([^"]+)","duration":"[^"]*","url":"(https://assets\.mixkit\.co/music/(\d+)/\d+\.mp3)".*?"copyrightNotice":"([^"]+)"', html):
        out.append({"id": m.group(4), "title": m.group(1), "artist": m.group(2), "url": m.group(3), "license": m.group(5)})
    return out


def crawl():
    cand = {"video": {}, "music": crawl_music(MUSIC_CATEGORY)}
    for beat, cats in BEATS.items():
        cand["video"][beat] = [c for cat in cats for c in crawl_video(cat)]
        print(f"{beat}: {len(cand['video'][beat])} free-licence candidates")
    os.makedirs(ASSETS, exist_ok=True)
    json.dump(cand, open(os.path.join(ASSETS, "candidates.json"), "w"), indent=1)
    return cand


def download(cand):
    credits = []
    os.makedirs(os.path.join(ASSETS, "broll"), exist_ok=True)
    os.makedirs(os.path.join(ASSETS, "music"), exist_ok=True)
    os.makedirs(os.path.join(ASSETS, "fonts"), exist_ok=True)
    vids = {c["id"]: c for beat in cand["video"].values() for c in beat}
    for vid, name in PICKS_VIDEO.items():
        c = vids[vid]
        dst = os.path.join(ASSETS, "broll", f"{name}.mp4")
        if not os.path.exists(dst):
            open(dst, "wb").write(get(c["url"]))
        credits.append({"file": f"broll/{name}.mp4", "source": c["page"], "license": c["license"]})
    tracks = {t["id"]: t for t in cand["music"]}
    for mid, name in PICKS_MUSIC.items():
        t = tracks[mid]
        dst = os.path.join(ASSETS, "music", f"{name}.mp3")
        if not os.path.exists(dst):
            open(dst, "wb").write(get(t["url"]))
        credits.append({"file": f"music/{name}.mp3", "title": t["title"], "artist": t["artist"],
                        "source": f"https://mixkit.co/free-stock-music/{MUSIC_CATEGORY}/", "license": t["license"]})
    fonts = {
        "Montserrat.ttf": "https://cdn.jsdelivr.net/gh/google/fonts@main/ofl/montserrat/Montserrat%5Bwght%5D.ttf",
        "JetBrainsMono.ttf": "https://cdn.jsdelivr.net/gh/google/fonts@main/ofl/jetbrainsmono/JetBrainsMono%5Bwght%5D.ttf",
    }
    for fn, url in fonts.items():
        dst = os.path.join(ASSETS, "fonts", fn)
        if not os.path.exists(dst):
            open(dst, "wb").write(get(url))
        credits.append({"file": f"fonts/{fn}", "source": "Google Fonts (OFL)", "license": "SIL Open Font License"})
    json.dump(credits, open(os.path.join(ASSETS, "credits.json"), "w"), indent=1)
    print("downloaded", len(credits), "assets ->", ASSETS)


if __name__ == "__main__":
    cand = crawl()
    if "--crawl" not in sys.argv:
        download(cand)
