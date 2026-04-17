# ComfyUI Video MCP

## Purpose

ComfyUI Video MCP is a Model Context Protocol (MCP) server that turns a single sentence of raw notes into a finished multi-scene video with minimal effort. You describe a vibe ("tokyo cyberpunk street, neon rain, anime style"), and the pipeline auto-detects the most appropriate cinematic skill framework, uses an LLM to generate five creative video concepts, lets you pick one, expands it into four scenes with cinematically-precise prompts (locking a protagonist anchor and following a HOOK → BUILD → CLIMAX → RESOLUTION narrative arc), queues each scene to a remote ComfyUI instance for actual video generation, and finally compiles all scenes into a montage with transitions and optional music overlay. The goal is an automated OpenMontage pipeline: raw idea in, finished MP4 out.

---

## Architecture

```
User notes (one sentence)
        │
        ▼
┌───────────────────┐
│   skills_engine   │  detect_skill() — matches 15 cinematic frameworks
│   (skills_engine  │  build_comfyui_positive/negative() — injects domain vocab
│    .py)           │  get_workflow_overrides() — sets model/resolution/fps
└────────┬──────────┘
         │  SkillSpec (camera moves, lighting K, quality tags, hook patterns)
         ▼
┌───────────────────┐
│  idea_generator   │  IdeaGenerator — calls Claude / Ollama / offline templates
│  (idea_generator  │  Generates 5 ideas → user picks one
│   .py)            │  Expands to 4 scenes with scene continuity enforcement
└────────┬──────────┘
         │  4 scene prompts (positive + negative, skill-injected)
         ▼
┌───────────────────┐
│  comfyui_client   │  queue_prompt() → ComfyUI REST API at 192.168.1.196:8188
│  (comfyui_client  │  get_queue_status(), get_history()
│   .py)            │  Workflow: wan22_lightx2v_api.json
└────────┬──────────┘
         │  completed video files (output/)
         ▼
┌───────────────────┐
│ montage_compiler  │  FFmpeg xfade transitions, music overlay
│ (montage_compiler │  → final MP4
│  .py)             │
└───────────────────┘
         │
         ▼
    Finished montage
```

Session state (ideas, scenes, generation jobs, montage jobs) is held in memory by `session.py`. All 15 MCP tools are exposed through `server.py` via FastMCP.

---

## Models

### Active Model: Wan2.2 + LightX2V (recommended)

| Property | Value |
|---|---|
| Config key | `wan22_lightx2v` |
| Architecture | Two-stage cascade (high_noise + low_noise UNET) |
| Workflow file | `workflows/wan22_lightx2v_api.json` |
| Steps | 4 (LightX2V 4-step LoRA — do not change) |
| CFG | 1.0 (LightX2V requirement — do not change) |
| Shift | 5.0 (ModelSamplingSD3) |
| Resolution | 640 × 640 |
| Frames | 81 (≈ 5 seconds at 16 fps) |
| FPS | 16 |
| Time per scene | ~108 seconds on RTX 4090D |
| ComfyUI host | 192.168.1.196:8188 |

**Workflow internals:** Two `UNETLoader` nodes load the high-noise and low-noise checkpoints, each receives its matching LoRA, each is wrapped in `ModelSamplingSD3(shift=5.0)`, and both feed into two `KSamplerAdvanced` nodes chained at steps 0→2 and 2→4. Output goes through `VAEDecode → CreateVideo → SaveVideo`.

**Workflow placeholders** (substituted at queue time):
`{{POSITIVE_PROMPT}}`, `{{NEGATIVE_PROMPT}}`, `{{WIDTH}}`, `{{HEIGHT}}`, `{{FRAMES}}`, `{{FPS}}`, `{{SEED}}`, `{{OUTPUT_PREFIX}}`

#### Model files (on remote machine at 192.168.1.196)

| Role | Path |
|---|---|
| High-noise UNET | `diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors` |
| Low-noise UNET | `diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors` |
| High-noise LoRA | `loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors` |
| Low-noise LoRA | `loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors` |
| Text encoder | `text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors` |
| VAE | `vae/wan_2.1_vae.safetensors` |

### Other Available Models

| Config key | Description | Status |
|---|---|---|
| `wan22_lightx2v` | Wan2.2 14B + LightX2V 4-step LoRA | **Active (default)** |
| `wan22` | Wan2.2 T2V 14B high-noise only | Installed, ~600s/scene |
| `wan22_calm` | Wan2.2 T2V 14B low-noise only | Installed |
| `ltxvideo` | LTX-Video 2 19B | Installed |
| `ltxvideo_fast` | LTX-Video 19B + distilled LoRA | Installed |
| `ltxvideo_camera` | LTX-Video 19B + camera control LoRA | Installed (partial LoRAs) |
| `animatediff` | SD1.5 + AnimateDiff | Not installed |
| `svd` | Stable Video Diffusion XT | Not installed |

---

## Quick Start

```bash
# 1. Clone and enter the project
cd comfyui-video-mcp

# 2. Activate the virtual environment
./venv/Scripts/python.exe -m pip install -r requirements.txt   # first time only

# 3. (Optional) Set your Anthropic API key for best idea quality
export ANTHROPIC_API_KEY=sk-ant-...

# 4. Register the MCP server with Claude Code
claude mcp add comfyui-video -- C:/path/to/comfyui-video-mcp/venv/Scripts/python.exe C:/path/to/comfyui-video-mcp/server.py

# 5. Verify connectivity
# In Claude Code: ping_comfyui()
```

No API key is required. The system falls back automatically:
- `ANTHROPIC_API_KEY` set → Claude API (best quality)
- Key not set, Ollama running → Ollama with llama3.2
- Neither available → offline template mode

---

## Full Pipeline Walkthrough

The complete flow from idea to finished montage:

```
1. generate_ideas("tokyo cyberpunk anime virtual reality")
   → auto-detects: anime skill
   → calls LLM with skill-injected system prompt
   → returns 5 video concepts, e.g.:
      1. Neon Ghost Protocol
      2. Akihabara Overflow
      3. Ghost Signal Uprising
      4. Synthetic Cherry Blossom
      5. Digital Dreamscape

2. select_idea(5)
   → selects "Digital Dreamscape"
   → expands to 4 scenes with continuity:
      Scene 1 (HOOK)       — protagonist introduced in neon rain alley
      Scene 2 (BUILD)      — protagonist enters virtual data-scape
      Scene 3 (CLIMAX)     — reality collapses, VR bleeds through
      Scene 4 (RESOLUTION) — protagonist stands in merged world
   → PROTAGONIST anchor locked across all scenes
   → post-processing enforcement re-injects protagonist if LLM drifted

3. generate_video()
   → reads wan22_lightx2v_api.json
   → substitutes {{POSITIVE_PROMPT}}, {{NEGATIVE_PROMPT}}, {{SEED}}, etc.
   → queues 4 prompts to ComfyUI at 192.168.1.196:8188
   → returns job IDs

4. check_status(wait=True)
   → polls ComfyUI /history and /queue endpoints
   → downloads completed .mp4 files to output/
   → ~108s per scene, ~7-8 minutes total

5. compile_montage(title="Digital Dreamscape", transition="dissolve")
   → FFmpeg xfade dissolve between scenes
   → optional music overlay (music_volume: 0.3)
   → outputs output/Digital_Dreamscape_montage.mp4
```

### Standalone Script (no MCP)

```bash
# Queue 4 scenes directly, bypassing the MCP server
./venv/Scripts/python.exe run_generate.py "tokyo cyberpunk anime virtual reality"
```

`run_generate.py` uses `detect_skill()` + `build_comfyui_positive/negative()` directly and queues to ComfyUI without starting the MCP server.

---

## MCP Tools Reference

All 15 tools are exposed through `server.py` via FastMCP and available in Claude Code after registration.

| Tool | Description |
|---|---|
| `list_skills` | List all 15 cinematic skill frameworks with their detection keywords |
| `detect_skill_for_notes` | Auto-detect which skill best matches a set of user notes |
| `generate_ideas` | Generate 5 video ideas from notes using the detected skill + LLM |
| `list_ideas` | Show the current session's generated ideas |
| `select_idea` | Pick one idea by number; expands it into 4 scene prompts |
| `regenerate_ideas` | Discard current ideas and generate a fresh set |
| `configure_pipeline` | Override model, resolution, frames, fps, or CFG for this session |
| `generate_video` | Queue all 4 scenes to ComfyUI; returns job IDs |
| `check_status` | Poll generation status; `wait=True` blocks until all scenes complete |
| `list_videos` | List completed video files in the output directory |
| `compile_montage` | Run FFmpeg montage compilation with transitions and optional music |
| `list_montages` | List completed montage files |
| `ping_comfyui` | Health-check the ComfyUI endpoint (useful for verifying connectivity) |
| `get_available_models` | List models configured in config.yaml and their install status |
| `session_status` | Dump current in-memory session state (ideas, jobs, files) |

---

## Skills Framework

`skills_engine.py` defines 15 `SkillSpec` dataclasses. Each skill encodes domain-specific production knowledge that is injected directly into every ComfyUI prompt.

### What each SkillSpec contains

| Field | Example (anime skill) |
|---|---|
| Camera moves | `"slow tracking shot at 2ft/s"`, `"handheld push-in"` |
| Lighting | `"neon underlighting 4200K"`, `"rim light 6500K"` |
| Quality tags | `"anime cel shading"`, `"sakuga animation quality"` |
| Negative tags | `"western cartoon"`, `"pixar style"`, `"3d render"` |
| Hook patterns | `"city wide establishing shot → tight face reveal"` |
| Generation params | width, height, frames, fps, steps, cfg overrides |

### The 15 Skills

| Skill key | Auto-detection keywords |
|---|---|
| `cinematic` | film, movie, cinematic, dramatic, epic, shot |
| `anime` | anime, manga, japanese, shonen, seinen, cyberpunk, neon city, neon lights, tokyo, akihabara, shibuya |
| `3d_cgi` | 3d, cgi, render, blender, octane, unreal, photorealistic |
| `cartoon` | cartoon, animated, pixar, disney, toon |
| `fight_scenes` | fight, action, battle, martial arts, combat, kickboxing |
| `motion_design` | motion, graphics, kinetic, typography, logo, brand |
| `ecommerce` | product, ecommerce, shop, buy, retail, showcase |
| `social_hook` | viral, hook, tiktok, reel, short, social |
| `music_video` | music, song, beat, rhythm, concert, festival |
| `brand_story` | brand, story, narrative, company, founder, origin |
| `fashion` | fashion, model, runway, lookbook, style, outfit |
| `food` | food, recipe, restaurant, chef, cooking, beverage |
| `real_estate` | real estate, property, house, interior, architecture |
| *(+ 2 more)* | — |

### Scene Continuity System

When an idea is expanded into 4 scenes:

1. A **PROTAGONIST anchor** is extracted and locked (name, appearance, costume, distinguishing features).
2. Scenes follow the narrative arc: **HOOK → BUILD → CLIMAX → RESOLUTION**.
3. After LLM generation, a **post-processing enforcement pass** checks every scene prompt for the protagonist anchor and re-injects it if the LLM drifted.

This keeps characters visually consistent across all four generated clips.

---

## Configuration Reference

Full path: `config.yaml`

### ComfyUI connection

```yaml
comfyui:
  host: "192.168.1.196"   # remote machine on LAN
  port: 8188
  timeout: 600            # seconds — large 14B models need this
  output_dir: "output"
```

### Pipeline defaults

```yaml
pipeline:
  default_model: "wan22_lightx2v"   # Wan2.2 + LightX2V 4-step LoRA
  default_frames: 25                # used by LTX models
  default_fps: 24
  default_width: 768
  default_height: 512
  default_steps: 20
  default_cfg: 3.0
```

Note: `wan22_lightx2v` overrides these with its own values (640x640, 81 frames, 16fps, 4 steps, CFG 1.0).

### Active model (wan22_lightx2v)

```yaml
models:
  wan22_lightx2v:
    unet_high:  "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
    unet_low:   "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
    lora_high:  "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors"
    lora_low:   "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors"
    text_encoder: "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    vae:        "wan_2.1_vae.safetensors"
    workflow:   "workflows/wan22_lightx2v_api.json"
    steps_override: 4       # do NOT change
    cfg_override: 1.0       # do NOT change
    shift: 5.0
    default_width: 640
    default_height: 640
    default_frames: 81      # 81 / 16fps = ~5 seconds
    default_fps: 16
```

### Montage

```yaml
montage:
  default_transition: "dissolve"   # fade | dissolve | wipe | slide | zoom | none
  transition_duration: 0.5
  default_resolution: "1280x720"
  default_fps: 24
  default_music: ""
  music_volume: 0.3
```

### Idea generation

```yaml
idea_generation:
  provider: "auto"              # claude → ollama → offline
  claude_model: "claude-opus-4-6"
  ollama_model: "llama3.2"
  ollama_host: "http://localhost:11434"
  ideas_per_request: 5
  scenes_per_idea: 4
```

### Skills injection

```yaml
skills:
  default_skill: "auto"
  inject_quality_boosters: true
  inject_style_tags: true
```

---

## Setup Instructions

### Prerequisites

- Python 3.10+ with `./venv` already created
- ComfyUI running on the remote machine (default: `192.168.1.196:8188`) with the Wan2.2 + LightX2V models installed
- FFmpeg in PATH (required for `compile_montage`)
- Claude Code CLI installed

### 1. Install dependencies

```bash
cd C:/Users/Sandip/Documents/Claude/comfyui-video-mcp
./venv/Scripts/python.exe -m pip install -r requirements.txt
```

### 2. Configure the ComfyUI host

Edit `config.yaml` if your ComfyUI instance is on a different address:

```yaml
comfyui:
  host: "192.168.1.196"
  port: 8188
```

### 3. Set LLM provider (optional)

For best results, set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Without it, the system automatically tries Ollama (`localhost:11434`), then falls back to offline template mode. All three modes produce valid video prompts.

### 4. Register with Claude Code

```bash
claude mcp add comfyui-video -- \
  C:/Users/Sandip/Documents/Claude/comfyui-video-mcp/venv/Scripts/python.exe \
  C:/Users/Sandip/Documents/Claude/comfyui-video-mcp/server.py
```

Verify registration:

```bash
claude mcp list
```

### 5. Test connectivity

In Claude Code, call `ping_comfyui()`. A successful response confirms the ComfyUI API is reachable and the MCP server is running.

### File structure

```
comfyui-video-mcp/
├── server.py               FastMCP server, 15 MCP tools
├── skills_engine.py        15 SkillSpec dataclasses + detect/build functions
├── idea_generator.py       IdeaGenerator (Claude/Ollama/offline) + scene continuity
├── comfyui_client.py       ComfyUI REST API client
├── montage_compiler.py     FFmpeg wrapper
├── session.py              In-memory session state
├── run_generate.py         Standalone script (no MCP)
├── config.yaml             All configuration
├── output/                 Generated videos and montages
└── workflows/
    ├── wan22_lightx2v_api.json     Active workflow (Wan2.2 + LightX2V)
    ├── wan22_t2v_api.json          Wan2.2 standalone
    ├── ltxvideo_api.json           LTX-Video 2
    └── ltxvideo_camera_api.json    LTX-Video + camera LoRA
```

---

## Troubleshooting

### ping_comfyui() fails / connection refused

- Confirm ComfyUI is running on the remote machine: open `http://192.168.1.196:8188` in a browser.
- Check that `comfyui.host` and `comfyui.port` in `config.yaml` match your setup.
- On Windows, confirm the firewall on the remote machine allows port 8188 inbound.

### Generation queued but never completes

- The `timeout` in `config.yaml` is 600 seconds. Wan2.2 14B at 81 frames takes ~108s on an RTX 4090D; slower GPUs may exceed this. Increase `timeout` if needed.
- Check ComfyUI's own console output on the remote machine for OOM or missing model errors.

### "Model not found" error in ComfyUI

- Verify the model files are in the correct subfolders on the remote machine (`diffusion_models/`, `loras/`, `text_encoders/`, `vae/`).
- File names are case-sensitive. Compare exactly against the names in `config.yaml`.

### Ideas are generic / ignoring the skill

- Check that `inject_quality_boosters` and `inject_style_tags` are both `true` in `config.yaml`.
- Run `detect_skill_for_notes("your notes")` to confirm the correct skill is being selected before generating ideas.
- If using offline mode (no API key, no Ollama), prompts are templated rather than LLM-generated. Set up Ollama or add an API key for better results.

### Montage compilation fails

- Confirm FFmpeg is installed and on your PATH: `ffmpeg -version`.
- Ensure all 4 scene videos exist in `output/` before calling `compile_montage`. Use `list_videos()` to verify.
- If scenes have mismatched resolutions, the xfade filter may fail. Use `configure_pipeline()` to set a consistent resolution before generating.

### Wrong protagonist across scenes

- The scene continuity system re-injects the protagonist anchor after generation. If drift persists, the PROTAGONIST anchor extracted from the idea title may be too vague. Choose an idea (`select_idea`) with a clearly named or described central character.

### Steps or CFG were changed and quality dropped

For `wan22_lightx2v`, do not override `steps_override` (4) or `cfg_override` (1.0). The LightX2V LoRA was trained specifically for these values. Use `configure_pipeline()` only to change resolution or frames, not sampler settings.

---

## License

MIT
