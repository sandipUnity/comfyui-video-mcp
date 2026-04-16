# ComfyUI Video MCP — Skill-Powered Automated Video Pipeline

A **Model Context Protocol (MCP) server** that turns raw notes into fully generated videos through ComfyUI, powered by 15 cinematic skill frameworks adapted from [higgsfield-seedance2-jineng](https://github.com/beshuaxian/higgsfield-seedance2-jineng).

---

## What This Does

Write a sentence. Get a cinema-quality video.

```
"rainy tokyo street, cyberpunk neon, midnight chase"
        ↓  skill auto-detection → anime
        ↓  Claude / Ollama generates 5 cinematic ideas
        ↓  pick one → 4 scenes with Kelvin-precise prompts
        ↓  ComfyUI generates each scene
        ↓  FFmpeg compiles montage with xfade transitions
        ↓  final MP4 ready
```

No API key required. Works fully offline with Ollama or template mode.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Claude Code (MCP Client)                    │
│                  (you talk here — natural language)             │
└────────────────────────┬────────────────────────────────────────┘
                         │  MCP protocol (stdio)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    server.py  (FastMCP)                         │
│                                                                 │
│  15 MCP Tools:                                                  │
│  list_skills · detect_skill_for_notes · generate_ideas          │
│  list_ideas · select_idea · regenerate_ideas                    │
│  generate_video · check_status · list_videos                    │
│  compile_montage · list_montages · session_status               │
│  ping_comfyui · get_available_models · configure_pipeline       │
└──────┬───────────────────┬────────────────────┬─────────────────┘
       │                   │                    │
       ▼                   ▼                    ▼
┌─────────────┐   ┌─────────────────┐   ┌──────────────────┐
│skills_engine│   │  idea_generator │   │ comfyui_client   │
│             │   │                 │   │                  │
│ 15 Skills:  │──▶│ LLM Providers:  │   │ REST API:        │
│ • cinematic │   │ • Claude API    │   │ /prompt          │
│ • anime     │   │ • Ollama local  │   │ /history         │
│ • 3d_cgi    │   │ • offline tpl   │   │ /queue           │
│ • cartoon   │   │                 │   │ /view            │
│ • fight     │   │ Skill-enhanced  │   │                  │
│ • food      │   │ system prompts  │   │ WebSocket:       │
│ • fashion   │   │ injected into   │   │ /ws (progress)   │
│ • ...+9more │   │ every LLM call  │   │                  │
└──────┬──────┘   └────────┬────────┘   └────────┬─────────┘
       │                   │                     │
       │  auto-detects     │  generates          │  queues workflow
       │  from notes       │  cinema-quality     │  JSON, polls
       │  keywords         │  prompts            │  completion
       │                   │                     │
       └───────────────────▼─────────────────────▼
                    ┌──────────────────┐
                    │   session.py     │
                    │  (in-memory)     │
                    │                  │
                    │ ideas[]          │
                    │ scenes[]         │
                    │ generation_jobs{}│
                    │ montage_jobs[]   │
                    └──────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  ComfyUI  :8188        │
              │                        │
              │  Models:               │
              │  • AnimateDiff         │
              │  • Stable Video Diff.  │
              │  • Wan2.1              │
              │  • CogVideoX           │
              └────────────┬───────────┘
                           │  output MP4/GIF
                           ▼
              ┌────────────────────────┐
              │  montage_compiler.py   │
              │  (FFmpeg)              │
              │                        │
              │  • xfade transitions   │
              │  • music overlay       │
              │  • title cards         │
              │  • resolution scale    │
              └────────────┬───────────┘
                           │
                           ▼
              output/montages/final.mp4
```

---

## Component Map

| File | Role | Key Responsibility |
|------|------|--------------------|
| [`server.py`](server.py) | MCP server | 15 tools exposed to Claude Code |
| [`skills_engine.py`](skills_engine.py) | Skill library | 15 cinematic frameworks, auto-detect, prompt builder |
| [`idea_generator.py`](idea_generator.py) | LLM bridge | Calls Claude/Ollama with skill-injected system prompts |
| [`comfyui_client.py`](comfyui_client.py) | ComfyUI API | Queue, poll, download via REST + WebSocket |
| [`montage_compiler.py`](montage_compiler.py) | FFmpeg wrapper | xfade transitions, music, title cards |
| [`session.py`](session.py) | State manager | Ideas, scenes, jobs — in-memory per session |
| [`config.yaml`](config.yaml) | Configuration | All defaults — LLM provider, resolutions, models |
| [`workflows/`](workflows/) | ComfyUI JSON | AnimateDiff, SVD, Wan2.1 workflow templates |

---

## Skill Framework Integration

The 15 skills from `higgsfield-seedance2-jineng` are embedded in `skills_engine.py`. Each skill carries:

```python
SkillSpec(
    id         = "anime",
    keywords   = ["anime", "manga", "shonen", ...],   # auto-detection
    quality_boosters = ["masterpiece", "cel shading", "anime style", ...],
    negative_tags    = ["western cartoon", "3d render", "realistic", ...],
    camera_vocabulary = [
        "dramatic pull-back from extreme close-up to full scene reveal",
        "impact frame freeze: held 0.3s with speed lines radiating outward",
        ...
    ],
    lighting_vocabulary = [
        "shonen style: 150% saturated warm amber sunlight from 45° above",
        "cyberpunk anime: dark base with neon cyan 16000K and magenta 7000K",
        ...
    ],
    hook_patterns = [
        "speed lines converging to center then exploding outward",
        ...
    ],
    technical_specs = {"fps": 24, "width": 768, "height": 432, "steps": 22, "cfg": 8.0},
    prompt_template = "You are an anime director..."   # LLM system prompt
)
```

When `generate_ideas()` is called, the skill's `prompt_template` becomes the LLM system prompt, forcing cinema-precise language. Every scene prompt is then post-processed to inject missing `quality_boosters` before being sent to ComfyUI.

### Auto-Detection Logic

```
notes = "tokyo street rain cyberpunk neon midnight"
         ↓
keywords matched:
  anime        → ["anime", "cyberpunk anime"] = 1
  cinematic    → ["cinematic", "realistic"]   = 0
  fight_scenes → ["fight", "combat"]          = 0
         ↓
highest score wins → anime selected
         ↓
ComfyUI overrides applied: 768×432, 24fps, CFG 8.0
```

---

## Prompt Quality: Before vs After Skills

### Without skills (plain LLM prompt):
```
masterpiece, best quality, tokyo street at night, neon lights,
rain, cyberpunk style
```

### With Anime skill injected:
```
Two figures clash on rain-soaked neon-lit street, speed lines
radiating from impact point, smear frame at 270° spinning heel
kick moment, shonen style: 150% saturated warm amber sunlight
from 45° above mixed with neon cyan 16000K left and magenta 7000K
right, extreme close-up impact frame held 0.3s, rain droplets
frozen mid-air in slow-motion 0.25x, torn jacket edges indicating
damage level, masterpiece, best quality, detailed anime art,
cel shading, anime shading, clean line art, key visual quality,
official art style, vibrant colors, expressive eyes, anime style,
manga style, Japanese animation
```

That second prompt is what actually goes into ComfyUI.

---

## Setup

### Prerequisites

| Tool | Purpose | Required |
|------|---------|----------|
| Python 3.11+ | Run MCP server | Yes |
| ComfyUI | Video generation | Yes |
| FFmpeg | Montage compilation | Yes (for montage) |
| Ollama | Local LLM (no API key) | Optional |
| ANTHROPIC_API_KEY | Claude API (best quality) | Optional |

### Install

```bat
cd comfyui-video-mcp
setup.bat
```

This creates a venv and installs all Python dependencies.

### Choose Your LLM Provider

Edit `config.yaml`:

```yaml
idea_generation:
  provider: "auto"   # auto → claude if key exists, else ollama, else offline
```

**Option A — Local Ollama (no API key):**
```bat
# Install from https://ollama.ai
ollama pull llama3.2
# Set provider: "ollama" in config.yaml
```

**Option B — Claude API (best prompt quality):**
```bat
# Copy .env.example to .env
copy .env.example .env
# Edit .env:  ANTHROPIC_API_KEY=sk-ant-...
# Set provider: "claude" in config.yaml
```

**Option C — Offline (no LLM):**
```yaml
# config.yaml
idea_generation:
  provider: "offline"
```

### Install ComfyUI Nodes

```bat
install_comfyui_nodes.bat
```

Installs:
- `ComfyUI-AnimateDiff-Evolved` — text-to-video animation
- `ComfyUI-VideoHelperSuite` — `VHS_VideoCombine` output node
- `ComfyUI-WanVideoWrapper` — Wan2.1 model support

### Download Models

Place in your ComfyUI `models/` folder:

```
ComfyUI/models/checkpoints/
  v1-5-pruned-emaonly.safetensors    ← SD 1.5 (AnimateDiff base)

ComfyUI/models/animatediff_models/
  mm_sd_v15_v2.ckpt                  ← AnimateDiff motion module
                                       (from HuggingFace: guoyww/animatediff)

ComfyUI/models/diffusion_models/
  Wan2.1-T2V-1.3B/                   ← Wan2.1 (optional, better quality)
                                       (from HuggingFace: Wan-AI/Wan2.1-T2V-1.3B)
```

### Register with Claude Code

```bat
claude mcp add comfyui-video -- python "C:\Users\Sandip\Documents\Claude\comfyui-video-mcp\server.py"
```

Verify:
```bat
claude mcp list
```

---

## Full Workflow Walkthrough

### Example: Cyberpunk Short Film

```
You: generate_ideas("rain-soaked tokyo alley, samurai vs drone, neon reflections")
```

**Server detects:** `anime` skill (keywords: cyberpunk anime, neon)  
**LLM called with:** Anime director system prompt (impact frames, speed lines, genre-specific lighting)

```
Skill detected: Anime & Japanese Animation [anime]
Provider: ollama

Generated 5 ideas:

[1] Blade and Circuit: Steel Meets Silicon
     A lone samurai faces a swarm of government drones in a flooded neon alley.
     Shonen-style explosive combat with cyan/magenta lighting and speed-line impacts.
     Style: Japanese animation | Mood: intense
     Tags: shonen, cyberpunk, combat, neon, rain

[2] Ghost Protocol: Last Stand
     ...

[3] Circuit Breaker
     ...
```

```
You: select_idea(1)
```

**Server generates 4 scenes** using Anime skill's cinema vocabulary:

```
Selected: [1] Blade and Circuit: Steel Meets Silicon
Skill: Anime & Japanese Animation [anime]
ComfyUI: 768×432 @ 24fps, 22 steps, CFG 8.0

Scene 1: Opening — drone swarm descends through neon rain
  Prompt: Rain-soaked neon alley, speed lines converging from edges as drone
          swarm descends through cyan 16000K light shafts, samurai silhouette
          at bottom of frame, impact frame opening held 0.3s with radiating speed
          lines, shonen style: 150% saturated, masterpiece, cel shading...
  Negative: western cartoon, 3d render, realistic, photographic...
  Duration: 3.0s  |  Hook: speed lines exploding outward

Scene 2: Confrontation — samurai draws blade
  ...

Scene 3: Combat exchange — blade deflects laser bolt
  ...

Scene 4: Resolution — drone falls, samurai sheathes blade in rain
  ...
```

```
You: generate_video()
```

```
Queued 4 generation job(s):
  Scene 1: queued (job: a3f2b1c0)
  Scene 2: queued (job: d4e5f6a7)
  Scene 3: queued (job: b8c9d0e1)
  Scene 4: queued (job: f2a3b4c5)
```

```
You: check_status(wait=True)
```

```
ComfyUI Queue Status:
Running: 1 | Pending: 3
...
Complete! Downloaded 4 file(s):
  output/videos/scene_a3f2b1c0_video.mp4
  output/videos/scene_d4e5f6a7_video.mp4
  ...
```

```
You: compile_montage(title="Blade and Circuit", transition="dissolve")
```

```
Montage compiled successfully!

Title:      Blade and Circuit
Clips:      4
Output:     output/montages/Blade_and_Circuit_1713276543.mp4
Size:       18.4 MB
Transition: dissolve
```

---

## All MCP Tools Reference

| Tool | Parameters | What It Does |
|------|-----------|--------------|
| `list_skills()` | — | Show all 15 skill frameworks |
| `detect_skill_for_notes(notes)` | `notes` | Preview skill auto-detection |
| `generate_ideas(notes, count, skill_id)` | notes required | Generate cinematic ideas |
| `list_ideas()` | — | Show current session ideas |
| `regenerate_ideas(feedback, count, skill_id)` | feedback required | Regen with changes |
| `select_idea(idea_id)` | `idea_id` | Pick idea → generate scenes |
| `configure_pipeline(model, width, height, ...)` | all optional | Adjust settings |
| `generate_video(scene_id, all_scenes, model, ...)` | all optional | Queue in ComfyUI |
| `check_status(job_id, wait)` | all optional | Monitor + download |
| `list_videos()` | — | Browse generated clips |
| `compile_montage(title, transition, ...)` | title optional | Build final MP4 |
| `list_montages()` | — | Browse finished montages |
| `ping_comfyui()` | — | Check ComfyUI connection |
| `get_available_models()` | — | List loaded ComfyUI models |
| `session_status()` | — | Full pipeline state |

---

## Configuration Reference

`config.yaml` — all settings with defaults:

```yaml
comfyui:
  host: "127.0.0.1"
  port: 8188
  timeout: 300          # seconds before generation times out

pipeline:
  default_model: "animatediff"   # animatediff | svd | wan21
  default_frames: 24
  default_fps: 8
  default_width: 512
  default_height: 512

idea_generation:
  provider: "auto"      # auto | claude | ollama | offline
  claude_model: "claude-opus-4-6"
  ollama_model: "llama3.2"
  ideas_per_request: 5
  scenes_per_idea: 4

skills:
  default_skill: "auto" # auto-detect or force a skill id
  inject_quality_boosters: true
  inject_style_tags: true

montage:
  default_transition: "fade"    # fade | dissolve | wipe | slide | zoom | none
  transition_duration: 0.5
  default_resolution: "1280x720"
  default_fps: 24
  music_volume: 0.3
```

---

## Output Structure

```
comfyui-video-mcp/
└── output/
    ├── videos/       ← generated scene clips from ComfyUI
    │   ├── scene_a3f2b1c0_video.mp4
    │   └── ...
    └── montages/     ← compiled final videos
        ├── Blade_and_Circuit_1713276543.mp4
        └── ...
```

---

## Supported Video Models

| Model | Workflow File | Best For | Quality |
|-------|--------------|---------|---------|
| AnimateDiff | `animatediff_api.json` | Stylized animation, anime | Good |
| Stable Video Diffusion | `svd_api.json` | Realistic motion from image | High |
| Wan2.1 T2V | `wan21_api.json` | Pure text-to-video, general | Very High |

Switch model:
```
configure_pipeline(model="wan21")
```

---

## Troubleshooting

**ComfyUI not reachable:**
```
ping_comfyui()
→ Make sure ComfyUI is running with: python main.py --listen
```

**No models in ComfyUI:**
```
get_available_models()
→ Download checkpoint files into ComfyUI/models/checkpoints/
```

**Ollama not responding:**
```bash
ollama serve          # start Ollama
ollama pull llama3.2  # download model
```

**FFmpeg not found (montage fails):**
- Download from https://ffmpeg.org/download.html
- Add `ffmpeg/bin` to Windows PATH environment variable
- Restart terminal

**VHS_VideoCombine node missing:**
```bat
install_comfyui_nodes.bat
# then restart ComfyUI
```

---

## Dependencies

```
mcp[cli]          MCP server framework
anthropic         Claude API client
aiohttp           Async HTTP (ComfyUI + Ollama)
aiofiles          Async file I/O
ffmpeg-python     FFmpeg Python bindings
websockets        ComfyUI WebSocket progress
pyyaml            Config file parsing
python-dotenv     .env loading
pillow            Image utilities
rich              Terminal output
```

---

## License

MIT — skill frameworks adapted from [higgsfield-seedance2-jineng](https://github.com/beshuaxian/higgsfield-seedance2-jineng) (MIT).
