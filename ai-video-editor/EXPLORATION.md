# Building an AI Video Editor (as of September 2026): An Open Exploration

> **Problem statement.** You bring footage: phone clips, camera rolls, screen recordings, interviews, drone shots, anything. The editor gets you to a **finished edit faster**. It is **our own product end to end**: not a Premiere/Resolve plugin and not dependent on anyone's proprietary editor. We own the timeline, the renderer, the AI and the customer.

---

## 0. TL;DR: what I would build

**Product:** a **local-first, AI-native non-linear editor (NLE)**. It *watches and indexes all your footage*, proposes a **story-first rough cut** that you shape in plain language, and then hands you a **real timeline** for the last 20% of taste work. The AI never paints pixels on the main path. It **writes and edits a structured timeline (an edit decision list) through typed tools**, the way a coding agent edits code.

**One-line thesis:** *Most editing time isn't cutting. It's **watching, logging, selecting and assembling**. So an AI that has "watched everything" and turns intent into an editable timeline takes away the worst 60–80% of the job, while the human keeps the taste.*

**Wedge (first customers):** people with **lots of raw footage and no dedicated editor**, who produce talk-led or event-led video: founders and marketing teams (product demos, testimonials, event recaps), YouTubers and vloggers, agencies, educators, podcasters. Expand later toward pro editors (multicam, docs) and toward "any footage" (travel, sports, weddings).

**Key architectural bets:**
1. **Own the timeline data model**: an open JSON schema, OpenTimelineIO-compatible. That's the heart of the product and the agent's "API".
2. **Own the render engine**: a GPU compositor (Rust + wgpu, or WebCodecs/WebGL/WebGPU in the browser), decoding and encoding through FFmpeg (LGPL) and hardware codecs.
3. **Indexing pipeline = moat**: shot detection, transcripts, visual captions, embeddings, faces, quality scores, beats. It's all stored as a queryable "footage database".
4. **Agent = planner LLM + tools** (`search_footage`, `view_frames`, `apply_ops`, `render_preview`, `critique`), with **every AI change shown as a reviewable diff** on the timeline.
5. **Export XML/EDL/OTIO** to Premiere/Resolve/FCP as an *escape hatch*, not a dependency. It lowers adoption fear, and we don't rely on those tools.

---

## 1. What does an editor actually do? (First principles)

Before designing an AI editor, break "editing" into jobs. Here's a rough time split for a typical 10-minute YouTube video from 3 hours of footage (a common editor anecdote. Treat the percentages as illustrative, not measured):

| Phase | What happens | Rough share of time | How automatable in 2026 |
|---|---|---|---|
| **Ingest & organize** | copy, proxies, bins, sync audio/multicam | 5–10% | 🟢 Very (mechanical) |
| **Logging / watching** | watch everything, note good moments, transcribe | 20–30% | 🟢 Very (ASR + VLM) |
| **Selects & story (paper edit)** | choose what to use, in what order | 15–20% | 🟡 Good for dialogue, weaker for pure visuals |
| **Assembly / rough cut** | put selects on the timeline | 10–15% | 🟢 Very (given selects) |
| **Fine cut** | pacing, J/L cuts, trims, b-roll coverage, rhythm | 15–25% | 🟡 Partial: taste-heavy |
| **Finishing** | captions, audio mix, color, titles, graphics | 10–20% | 🟢 captions/loudness, 🟡 color, 🟡 graphics |
| **Versions** | 16:9 → 9:16, 15s/30s cutdowns, languages | 5–15% | 🟢 Very |

**Gut-level takeaway:** the rows marked 🟢 are *search, bookkeeping and following rules*. LLMs plus perception models are great at that. The 🟡 rows are *taste*, and there the AI should **propose options** while the human decides.

### Alternative lens: an edit is a *program*
An edit is a small program:
```
timeline = [
  clip(src="A003.mov", in=00:01:12.400, out=00:01:19.050, track=V1),
  clip(src="broll_12.mp4", in=3.2, out=6.0, track=V2, over=V1@5.0s),
  caption(words=..., style="bold-pop"),
  audio(src="music.wav", duck_under=dialogue, -18 LUFS),
]
```
Rendering is "compiling" that program into pixels. **The AI's job is to write and refactor this program**, which LLMs already do well (Claude Code, Cursor). That's why the right primitive is **structured timeline ops**, *not* end-to-end video generation. End-to-end generation would re-synthesize the user's own footage and hallucinate faces, logos and words.

### Alternative lens: an editor is a *search engine + storyteller*
- **Search:** "Find the moment where the customer says the price was a no-brainer" / "a wide shot of the venue at sunset with no people".
- **Storyteller:** order the found moments into hook → context → tension → payoff.

If the search is bad, no storytelling can save you. **Indexing quality is the ceiling on edit quality.**

---

## 2. The competitive landscape (September 2026)

| Product | What it is | Strength | Gap we can attack |
|---|---|---|---|
| **[Cardboard](https://www.ycombinator.com/companies/cardboard)** (YC W26) | Browser-based agentic editor. Custom WebCodecs + WebGL2 renderer, fully client-side. Text-to-edit, semantic footage search, multi-aspect reformat, beat sync, captions, voice cloning, XML export. $60/mo unlimited. | Highest-upvoted HN launch of W26. Fast shipping (13 releases in ~3 months). No credits. | Browser limits (no Firefox; 10GB upload cap at launch), aimed at marketers/founders. Heavy pro footage (4K ProRes, hours of multicam) is hard in a tab. |
| **[Descript](https://www.descript.com/underlord)** + Underlord | Transcript-first editor. The agentic co-editor chains ops ("remove filler, captions, 3 vertical clips"). | Huge brand. The "edit video like a doc" paradigm. | Built around *speech*. Weak when footage is mostly visual (travel, events, sports). |
| **[Mosaic](https://mosaic.so/)** (YC W25, $3.8M seed) | A canvas of agentic editing workflows. A/B variants from the same raw footage, then a timeline. | Enterprise/agency scale (TubeScience, News Corp). | Workflow and ads focus, less a "general footage → story" editor. |
| **[Eddie AI](https://www.heyeddie.ai/)** | Assistant editor for pros: logging, multicam sync, interview rough cuts. Exports to Premiere/Resolve/FCP/Avid, and can run inside Claude/ChatGPT. Credits. | Loved by pros for interview-driven work. | *Depends on the NLE for finishing.* It's an assistant, not the editor. That's the thing we refuse to be. |
| **Opus Clip / Captions / Submagic** | Long → short clipping, captions, viral hooks. | Nail one high-frequency job. | Narrow: they don't do "raw footage → full edit". |
| **[Adobe Premiere AI Assistant](https://news.adobe.com/news/2026/06/adobe-unveils-major-expansion)** (public beta June 2026) + Generative Media in timeline (Sept 2026) | Organizes bins, renames, flags interview questions, assembles a rough opening cut. | Distribution, pro trust. | Bolted onto a 20-year-old UI. Pricing and complexity. Incumbent's dilemma. |
| **CapCut** | Consumer and prosumer editor with many AI effects. | Massive distribution, templates. | Template-first, not intent/story-first. ByteDance trust issues for businesses. |
| **Runway / Veo / Sora class** | Generative video. | Creating shots that don't exist. | Not *editing your footage*. We use them as a **tool** (b-roll fill, extend a shot), not as the core. |

**Enabling tech that landed recently:** Google shipped **[agentic video understanding in Gemini](https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-agentic-video-in-gemini/)** (Sept 1, 2026). The model *chooses what to watch*: it reads the transcript first, then loads only relevant windows at a chosen frame rate, and uses up to ~88% fewer tokens on long videos. It can find precise editing boundaries. That's exactly the "watch everything cheaply" primitive an editor needs, and it means **raw model access is commoditizing**. Our value has to sit *above* it: the index, the timeline, the UX and the loop.

### Where's the gap?
Plot the players on two axes:

```
                     General footage (any kind)
                              ▲
                              │         ★ (us: story-first, any footage,
                              │            own editor, handles big files)
          Adobe AI Asst ●     │
                              │   ● Cardboard
     Eddie ● (needs NLE)      │
─────────────────────────────┼──────────────────────────────► Full editor, own timeline
  Assistant / plugin          │                                (you finish inside)
                              │   ● Descript
            Opus Clip ●       │   ● CapCut
                              │   ● Mosaic
                              │
                     Speech-only / narrow job
```

The top-right quadrant (general footage + a full independent editor + agent-native) has only Cardboard and the incumbents in it. Cardboard is browser-bound and marketer-focused. Adobe is burdened by legacy. **That's the space to take.**

---

## 3. Scope: what the editor can do

### 3.1 MVP (months 0–4): "From a folder of footage to a watchable rough cut"
1. **Ingest anything**: drag a folder (MP4/MOV/HEVC/ProRes/screen recordings/audio). Proxies are generated automatically. Timecode and metadata are preserved.
2. **Auto-index** (background, with progress):
   - Shot/scene detection
   - Transcripts with **word-level timestamps + speaker diarization**
   - Per-shot visual description ("medium shot, woman at a laptop, warm window light, smiling")
   - Embeddings for semantic search (text ↔ frame, text ↔ speech)
   - Quality flags: blur, shake, under/over-exposure, clipped audio, wind noise
3. **Semantic footage search**: "all shots of the product on a table", "when Raj talks about pricing".
4. **Story-first rough cut:** the user writes a brief ("3-min recap of our launch event: energy, the keynote highlights, 2 customer reactions, end on the team photo"). The agent writes a **paper edit** (an outline with the selected moments), and the user approves or reorders it. Then the agent **assembles the timeline**.
5. **Text-based editing** of dialogue (Descript-style): delete words → ripple cuts. Filler/silence removal.
6. **Chat edits with diffs:** "tighten the intro to 10 seconds", "swap the b-roll at 0:42 for something with people". Each change shows up as a highlighted diff you can accept or reject.
7. **Captions** (styled, word-by-word), **loudness normalization**, **music bed with auto-ducking**.
8. **Reformat** to 9:16 / 1:1 with subject-tracking reframe.
9. **Export**: MP4 (H.264/HEVC), plus **FCPXML / OTIO / EDL** export as the escape hatch.

### 3.2 v1 (months 4–9): "Your AI assistant editor, for real"
- **Multicam** sync (audio waveform cross-correlation) + AI camera switching (cut to whoever is speaking, with reaction shots).
- **B-roll auto-coverage:** when the talking head says "our warehouse", find warehouse shots and lay them on V2 with J/L cuts.
- **Beat-synced montage cuts** for music-driven sections.
- **Style memory / "house style"**: learn from the user's past edits (pacing, caption style, LUT, lower-thirds). Brand kit.
- **Variants:** "give me 3 hooks for the first 5 seconds". A/B cutdowns (60/30/15s).
- **Self-review loop:** the agent *renders a preview and watches it* with a VLM, catching jump cuts, black frames, mid-word cuts, captions over faces, and b-roll that contradicts the voiceover.
- **Collaboration:** share a link, and comments are timecoded. "Apply all comments" becomes an agent action.
- **Audio cleanup:** voice isolation/denoise, EQ presets.
- **Basic color:** auto white balance/exposure match across cameras, LUTs, "match shot A to shot B".

### 3.3 Later (9–18 months)
- **Generative assists**: fill a gap with generated b-roll, extend a shot by 1 second, remove an object, fix an eye-line, AI voiceover/dub with lip-sync. They're always **labeled** and on their own track.
- **Motion graphics templates** driven by data (lower-thirds, animated charts, kinetic type) using a code-based motion layer.
- **Team/enterprise:** asset library across projects ("find every shot of our CEO from the last 2 years"), permissions, brand compliance checks, API.
- **Long-form & doc workflows:** hours of interviews → story structure with themes, "string-outs" per topic.
- **Local models** for privacy-sensitive customers (on-device ASR + small VLM).

### 3.4 Explicitly *out of scope* (at least initially)
- High-end color grading (Resolve's color page), VFX compositing (Nuke), professional audio post (Pro Tools). We do "good enough" and export for the rest.
- Pure text-to-video generation products (we're an *editor*).
- Film/TV feature workflows (conform, deliverables specs, 8K RAW).

---

## 4. System architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                               CLIENT (desktop app)                        │
│  ┌───────────┐  ┌─────────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Chat/Brief│  │ Paper edit view │  │  Timeline UI │  │ Media browser│  │
│  │  panel    │  │ (script/outline)│  │ (tracks,diff)│  │ + sem. search│  │
│  └─────┬─────┘  └───────┬─────────┘  └──────┬───────┘  └──────┬───────┘  │
│        └────────────────┴── Timeline Store (JSON, CRDT, undo) ─┘          │
│                                   │                                       │
│   ┌───────────────────────────────▼──────────────────────────────────┐    │
│   │ Media Engine (Rust): decode (FFmpeg/HW) → GPU compositor (wgpu)   │    │
│   │ → real-time preview; export encode (HW encoders/FFmpeg)           │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│   Local: originals stay on disk · proxies · frame cache · local index      │
└───────────────┬───────────────────────────────────────────────────────────┘
                │  (upload only proxies/audio/sampled frames, not 4K originals)
┌───────────────▼────────────────────────────────────────────────────────────┐
│                               CLOUD                                         │
│  Indexing workers: shot detect · ASR+diarization · VLM captions ·           │
│                    embeddings · faces · OCR · quality · beats                │
│  Footage DB: Postgres + pgvector (or LanceDB) → "shots" table                │
│  Agent service: planner LLM + tools; eval & telemetry                        │
│  Optional: cloud render farm, generative models, collaboration sync          │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 4.1 Desktop vs browser: the first big decision

| | **Browser (Cardboard's bet)** | **Desktop local-first (my bet for "any footage")** |
|---|---|---|
| Onboarding | Instant, a link | Download/install friction |
| Big files (100GB+ shoots) | Painful: File System Access API is Chromium-only, memory limits | Native disk I/O, no upload of originals |
| Codec coverage (ProRes, HEVC 10-bit, log) | WebCodecs varies by browser/OS | FFmpeg + OS HW decoders handle it all |
| GPU perf | WebGL2/WebGPU: good, but a sandbox | Full wgpu/Metal/Vulkan |
| Collaboration | Natural | Needs sync layer |
| Iteration speed | Deploy anytime | Auto-updates (fine) |

**Recommendation:** a **desktop app (Tauri shell + web UI + Rust media engine)** that's local-first, and a **web viewer/reviewer** for sharing and comments. Write the compositor in Rust/wgpu so it can **also compile to WASM + WebGPU** later for a lighter browser editor. This covers *"any footage"*, which is the user's stated goal. It's also the most defensible difference from Cardboard.

*Counter-view (steel-man):* if the target is marketers with phone clips and screen recordings, **browser wins on adoption**, and Cardboard proves you can build a real renderer on WebCodecs. If your first ICP is that segment, go browser. The ICP decides this, not tech taste.

### 4.2 The timeline data model (the most important file in the codebase)

```jsonc
{
  "fps": {"num": 30000, "den": 1001},        // rational time, never floats!
  "resolution": [1920, 1080],
  "tracks": [
    {"id": "V1", "kind": "video", "items": [
      {"id": "c1", "type": "clip", "media": "m_A003",
       "src_in": 43210, "src_out": 43410,      // in frames of the source
       "rec_in": 0,                           // position on timeline
       "effects": [{"type": "transform", "scale": 1.1, "keyframes": [...]}],
       "provenance": {"by": "agent", "reason": "hook: strongest laugh", "op_id": "op_17"}}
    ]},
    {"id": "A1", "kind": "audio", "items": [...]},
    {"id": "CAP", "kind": "caption", "items": [...]}
  ],
  "markers": [...],
  "story": {"beats": [{"id": "hook", "items": ["c1", "c2"]}, ...]}   // links timeline ↔ paper edit
}
```

Design rules:
- **Rational time** (frames/timebase), not float seconds. That avoids off-by-one-frame drift. *Concrete example:* at 29.97 fps a float `12.345s` isn't an exact frame, so repeated trims accumulate half-frame errors and you get flash frames.
- **Every mutation is an op** (`insert`, `trim`, `ripple_delete`, `move`, `split`, `set_effect`, `add_caption`…). The UI and the agent call the *same* ops. That gives you undo/redo, diffs, collaboration (CRDT/OT) and an audit trail for free.
- **Provenance** on every item: who made it (user/agent) and *why*. The UI can then explain "why is this shot here?".
- **Import/export via OpenTimelineIO** for interop.

### 4.3 The media engine
- **Decode:** FFmpeg (libav*) **dynamically linked, LGPL build** plus platform HW decoders (VideoToolbox, NVDEC, VAAPI/D3D11). Keep a frame cache and GOP-aware seeking. Generate **proxies** (e.g. 720p, all-intra or short-GOP) for scrubbing.
- **Composite:** GPU graph (wgpu): transforms, crops, opacity, blend, LUTs, text/caption rendering (we use our own text engine or Skia), transitions. Work in linear light and a 16-bit float pipeline, so log/HDR footage doesn't band.
- **Audio:** a separate audio graph (mixing, ducking, EBU R128 loudness, denoise).
- **Encode:** HW encoders (VideoToolbox/NVENC/QSV) as the default. Software paths for ProRes/DNxHR.
- **Licensing gotchas (important for "sell the whole thing"):**
  - FFmpeg: stay LGPL. **Don't** statically link GPL parts like x264/x265. Use OS hardware encoders, or license x264 commercially.
  - H.264/HEVC patent pools: using OS-provided codecs usually shifts royalties to the OS vendor. AV1/VP9 are royalty-free. Get a lawyer's read before launch.
  - **Remotion** needs a paid company license above a small team size. Fine for prototypes, but don't make it the core renderer if you want full ownership.
  - Model licenses: check commercial terms (e.g. Parakeet is CC-BY-4.0, which is fine. Some VLM weights are research-only).

### 4.4 The indexing pipeline ("the AI has watched everything")

For each media file, produce **shot records**:

| Signal | Tooling options (2026) | Why the editor needs it |
|---|---|---|
| Shot boundaries | PySceneDetect / TransNetV2 | unit of retrieval and cutting |
| Transcript + word timestamps + speakers | ElevenLabs Scribe v2 (managed, very accurate), WhisperX or NVIDIA Parakeet (self-host) | text editing, soundbite selection, captions |
| Visual description per shot | Gemini (agentic video understanding), Claude/GPT vision on sampled frames, TwelveLabs Pegasus | "what's in this shot", story reasoning |
| Embeddings | TwelveLabs Marengo, SigLIP-class open models | semantic search |
| Faces / people clustering | face embeddings + clustering, user names them once | "shots of Priya" |
| OCR | on-screen text, slides, signage | screen recordings, events |
| Quality | blur (Laplacian variance), shake (optical flow), exposure histogram, audio clipping/SNR | never pick a blurry take |
| Audio events | laughter, applause, music, silence | the "energy" of moments |
| Music analysis | beat/downbeat tracking, sections | montage sync |
| Camera motion | optical flow classification (pan/tilt/static/handheld) | cutting on motion, matching |

**Concrete cost example (back-of-envelope, check current prices):** a shoot has **3 hours** of footage and ~**1,500 shots**.
- ASR: 180 min × ~$0.4/hr-ish managed ≈ **~$1–2**. (Self-hosted Parakeet on a GPU is cents.)
- VLM captions: 1,500 shots × 3 frames × ~300 tokens ≈ 1.35M input tokens. On a Flash-class model that's roughly **~$0.5–2**.
- Embeddings via TwelveLabs indexing: 180 min × $0.042 ≈ **$7.56**, plus $0.27/month to keep it indexed. (Self-hosted SigLIP-class embeddings are near-free.)
- **Total ≈ $3–12 per 3-hour shoot.** That's affordable under a $30–60/mo subscription for typical users, and ruinous for someone uploading 50 hours a week. **Hence tiered indexing:** cheap signals for everything (ASR, shots, embeddings), and expensive VLM detail only on demand, or on shots the agent is actually considering. This is exactly the "agentic watching" pattern Gemini just productized.

### 4.5 The agent

**Loop:** *Brief → Plan (paper edit) → Assemble → Self-review → Human feedback → Refine.*

**Tools the agent gets (typed, deterministic, testable):**
```
search_footage(query, filters) -> [shot refs + scores]
get_transcript(media, t0, t1) -> words w/ timestamps + speaker
view_frames(media, t0, t1, fps) -> images   # expensive: used sparingly
get_shot_info(shot_id) -> description, quality, faces, motion
get_timeline() / apply_ops([op...]) -> new timeline + diff
render_preview(range, res=360p) -> video file
critique(preview) -> list of issues (VLM watches its own output)
music_search(mood, bpm, duration) / beats(track)
ask_user(question, options)   # when intent is truly ambiguous
```

**Why a paper edit first?** It's cheap to iterate on text. It's also *legible to the user* ("Hook: Maya laughing 'we almost didn't ship this' → Context: keynote 0:12–0:40 → …"). Only after approval do we spend compute on assembly and rendering. This mirrors how pro editors work, and it cuts wasted LLM calls.

**Precision trick:** LLMs are bad at frame-exact numbers. So the agent picks *semantic* boundaries ("start at the word 'honestly', end after 'shipped'"), and **deterministic code snaps** them to frames. Snap targets: word boundaries from ASR, plus a padding rule (~4–6 frames of pre-roll), plus avoiding cuts mid-motion. *Concrete example:* the agent says "cut after the word 'launch'". The word ends at 00:12.483. Snap to the next frame at 29.97 fps (frame 375 = 12.5125s), then add 5 frames of tail so the breath doesn't feel clipped → out-point is frame 380.

**Model choice:** a frontier model (Claude Opus/Sonnet class, or Gemini) for planning and story reasoning, and Flash-class models for bulk captioning. Keep it **model-agnostic behind an interface**, because prices and quality leapfrog every quarter.

### 4.6 UX principles
1. **Never a black box.** Every AI edit is a diff with a "why". Accept/reject per change.
2. **Three synchronized views**: *Brief/Chat* ↔ *Paper edit (script)* ↔ *Timeline*. Editing any one updates the others.
3. **Point + talk:** select a range, then say "make this punchier". Deixis (pointing) plus language beats language alone.
4. **Always a real timeline underneath**, so pros never hit a wall where "the AI can't do it and I can't either".
5. **Fast feedback:** a proxy preview in seconds, and final renders in the background.

---

## 5. Limitations (honest ones)

| Limitation | Why it's hard | Mitigation |
|---|---|---|
| **Taste & pacing** | "Good" is subjective and genre-specific. Rhythm is felt, not described. | Offer variants, learn house style from accept/reject data, keep the human in the loop for the fine cut. |
| **Non-dialogue footage** (travel, sports, weddings, music videos) | No transcript to anchor story. It needs visual storytelling and emotion reading. | Beats + motion + VLM "moment" scoring. Treat as v1+. Start with templates (montage structures). |
| **Frame-exact timing from models** | VLMs sample frames sparsely and hallucinate timestamps. | Semantic boundaries + deterministic snapping (ASR words, shot cuts, optical flow). |
| **Hallucinated footage content** | "There's a shot of the CEO on stage" when there isn't. | Every claim must reference a shot ID. The self-review watches the render. Retrieval, not memory. |
| **Cost at scale** | Hours of 4K footage × VLM tokens. | Tiered indexing, proxies, local models, agentic "watch only what's needed". |
| **Upload/bandwidth** | 200GB shoots can't go to the cloud quickly. | Local-first: only proxies/audio/frames leave the machine. |
| **Codec zoo & pro formats** | RAW, log, VFR phone footage, broken timecode. | FFmpeg + HW decode, VFR → CFR conform on ingest. Start with common formats. |
| **Audio mixing & color** | Pro-level work is an art and a whole product. | Good-enough auto tools + export to pro tools. |
| **Rights & ethics** | Music licensing, generated faces/voices, consent. | Licensed music library, labeled generative content, voice-cloning consent flow. |
| **Trust / control** | Editors hate losing control, and creators hate "AI slop". | Diffs, provenance, a real timeline, and never auto-publish. |
| **Commoditization** | Adobe/CapCut/Descript all add agents. Models get cheaper and smarter every quarter. | The moat lives in data, workflow and the index (see §7), not in "we call an LLM". |

---

## 6. Build plan

### Team (lean, first 6 months)
- 1 **media engine** engineer (Rust, FFmpeg, GPU): the rarest hire
- 1–2 **full-stack/UI** engineers (timeline UI is deceptively hard: virtualization, drag physics, waveform/thumbnail strips)
- 1 **AI/agent** engineer (indexing pipeline, agent tools, evals)
- Founder on product and design, living with users

### Milestones
| Month | Milestone | "Done" means |
|---|---|---|
| 0–1 | **Engine spike**: decode → composite → encode, scrub 4K proxy smoothly, JSON timeline → MP4 | Render a 3-track timeline exactly as the JSON describes |
| 1–2 | **Index v0**: shots + ASR + captions + embeddings; semantic search UI | "find X" returns the right shot in top-5 ≥ 80% on our test set |
| 2–3 | **Agent v0**: brief → paper edit → assembly with ops + diff UI | 10 real users get a watchable rough cut from their own footage |
| 3–4 | **MVP polish**: captions, reframe, music + ducking, export (MP4 + FCPXML/OTIO) | 5 users publish something made with it |
| 4–6 | Self-review loop, b-roll coverage, multicam, style memory | Measured time-to-first-publish drops ≥ 50% vs their old workflow |

### How to know if the AI is good (evals)
Build an **eval set early**: raw footage + the *human-made final cut* (get these from friendly creators/agencies with permission).
- **Selection recall:** what % of the moments the human editor used did the AI also pick?
- **Keep-rate:** what % of AI-proposed edits did users accept without changes? (The north-star product metric.)
- **Edit distance to final:** how many ops did the user apply after the AI draft? Fewer is better.
- **Defect rate from self-review:** jump cuts, mid-word cuts, black frames, caption collisions per minute.
- **Time-to-first-export** in real usage.

*Concrete example:* a human editor used 12 soundbites. Our AI picked 15, and 9 of them overlap with the human's → recall = 9/12 = **75%**, precision = 9/15 = **60%**. If the user then deleted 4 AI clips and added 2 of their own, that's 6 corrective ops on a 20-op edit → **~70% of the draft survived**.

### Pricing thoughts
- Cardboard: $60/mo unlimited. Eddie: credits. Descript: seats + AI credits.
- **Suggestion:** a subscription with generous **indexed hours/month** (the true cost driver), unlimited editing and agent chat, and generative features as add-on credits. Say "hours of footage" (a unit users understand), never "tokens".

---

## 7. Moats: what makes this defensible when models are commodities?

1. **The footage index as a system of record.** Once a team's whole archive is indexed and searchable ("every shot of our product, 2024–2026"), switching costs are high.
2. **Edit-decision data flywheel.** Every accept/reject/trim is a training signal about taste, per user and per genre. Nobody else gets *your users'* corrections.
3. **Owning the engine.** Plugins live and die at the host's mercy (Adobe can ship your feature natively tomorrow). Owning timeline + render means we can do things plugins can't, like agent-native diffs and a self-review render loop.
4. **Workflow depth in a vertical** (e.g. event recaps for agencies, or customer-testimonial pipelines for B2B marketing) beats horizontal breadth early.

---

## 8. Open questions I'd resolve with customer interviews
1. Which segment feels the "too much footage, not enough editing time" pain most *and* pays? (Agencies? B2B marketing? YouTubers?)
2. Do they want a **finished video** or a **great rough cut** they finish themselves? The answer changes the whole UX.
3. How big are typical projects (GB, hours)? That decides browser vs desktop.
4. Would they accept cloud processing of proxies? (Enterprise/legal/medical say no → on-device models.)
5. Is the export escape hatch a *requirement* to try us? (Probably yes for pros, irrelevant for creators.)

---

## 9. Check your understanding (exercises)

1. **Cost reasoning:** a wedding videographer uploads 12 hours of footage per wedding, 4 weddings a month. Using the §4.4 numbers, estimate monthly indexing cost with *full* VLM captioning vs *tiered* (VLM only on the ~15% of shots the agent considers). What subscription price keeps ≥ 70% gross margin?
2. **Frame math:** a clip is 23.976 fps. The ASR says the word "amazing" ends at 00:00:07.310. Which frame number do you snap the out-point to (round *up*), and what's the out-point if you add 4 frames of tail? Why would rounding *down* be worse here?
3. **Design choice:** why is it dangerous to let the LLM output the whole timeline JSON directly each turn, instead of emitting ops like `trim(c7, out=-12f)`? List at least 3 concrete failure modes.
4. **Metrics:** in a test, the AI selected 20 moments and the human's final cut used 10. 8 of the AI's are in the human's. Compute precision and recall. Which matters more for a *rough-cut* product, and why?
5. **Strategy:** make the strongest case *for* building in the browser (like Cardboard) instead of desktop. Then say what evidence from customer interviews would flip your decision either way.
6. **Hard case:** a user uploads 2 hours of GoPro mountain-biking footage with no speech and asks for "a 90-second hype edit". Which signals from the §4.4 table would the agent lean on? Sketch the paper edit structure it should produce.

---

### Sources
- [Cardboard (YC)](https://www.ycombinator.com/companies/cardboard) · [Launch HN: Cardboard](https://news.ycombinator.com/item?id=47170174) · [StartupHub on Cardboard](https://www.startuphub.ai/ai-news/claude's-corner/2026/claudes-corner-cardboard-yc-w2026)
- [Descript Underlord](https://www.descript.com/underlord)
- [Mosaic](https://mosaic.so/) · [Mosaic seed announcement](https://mosaic.so/blog/mosaic-seed-round-announcement)
- [Eddie AI](https://www.heyeddie.ai/)
- [Adobe creative agent expansion (June 2026)](https://news.adobe.com/news/2026/06/adobe-unveils-major-expansion) · [Adobe generative media in timeline (Sept 2026)](https://blog.adobe.com/en/publish/2026/09/08/generate-create-directly-in-your-timeline-with-new-ai-powered-innovations-in-premiere-after-effects)
- [Gemini agentic video understanding](https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-agentic-video-in-gemini/) · [Gemini video understanding docs](https://ai.google.dev/gemini-api/docs/video-understanding)
- [TwelveLabs pricing](https://www.twelvelabs.io/pricing)
- [ElevenLabs Speech to Text](https://elevenlabs.io/speech-to-text) · [WhisperX guide](https://localaimaster.com/blog/whisperx-guide)
- [Open-source video editor SDKs roundup](https://img.ly/blog/best-open-source-video-editor-sdks-2025-roundup/) · [OpenReel (WebCodecs editor)](https://github.com/Augani/openreel-video)
- [AI editing tools overview 2026](https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/opus-clip-descript-submagic-captions-ai-video-editor-tools-2026)

*Prices and product details are as reported in the sources above around September 2026. Verify them before making financial decisions.*
