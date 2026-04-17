# AGENT_CONTEXT.md — ComfyUI Video MCP

**Machine-readable specification for autonomous AI agents.**
**Last updated:** 2026-04-16
**Purpose:** Any agentic AI (Claude, GPT, Gemini, local LLMs) can read this document and operate the project without human guidance.

---

## TABLE OF CONTENTS

1. [Project Overview](#1-project-overview)
2. [File Inventory](#2-file-inventory)
3. [Environment and Execution](#3-environment-and-execution)
4. [ComfyUI API Format](#4-comfyui-api-format)
5. [Active Workflow: wan22_lightx2v_api.json](#5-active-workflow-wan22_lightx2v_apijson)
6. [Skills Engine](#6-skills-engine)
7. [Idea Generator](#7-idea-generator)
8. [MCP Server Tools (15 tools)](#8-mcp-server-tools-15-tools)
9. [Session State](#9-session-state)
10. [ComfyUI Client](#10-comfyui-client)
11. [Montage Compiler](#11-montage-compiler)
12. [Config.yaml Reference](#12-configyaml-reference)
13. [Known Constraints and Gotchas](#13-known-constraints-and-gotchas)
14. [Typical Agent Workflow (Step-by-Step)](#14-typical-agent-workflow-step-by-step)
15. [Alternative: Standalone Script](#15-alternative-standalone-script)
16. [Workflow Node Graphs (All Workflows)](#16-workflow-node-graphs-all-workflows)
17. [Model Directory Mapping](#17-model-directory-mapping)
18. [Error Recovery Procedures](#18-error-recovery-procedures)

---

## 1. PROJECT OVERVIEW

**Name:** ComfyUI Video MCP
**Type:** MCP (Model Context Protocol) server
**Transport:** stdio
**Framework:** FastMCP

### Pipeline Summary

```
User writes notes
      ↓
detect_skill() — auto-selects 1 of 15 SkillSpec presets
      ↓
generate_ideas() — LLM generates N creative video concepts
      ↓
User selects one idea
      ↓
generate_scenes() — LLM generates 4 cinematically-precise scene prompts
      ↓
generate_video() — scenes queued to ComfyUI via REST API
      ↓
ComfyUI renders videos (~108s/scene on RTX 4090D)
      ↓
compile_montage() — FFmpeg assembles scenes into final video
```

### Key Design Decisions

- Skills engine is deterministic (keyword scoring), no LLM needed for skill detection.
- LLM is only used for idea generation and scene prompt writing.
- ComfyUI runs on a remote LAN machine; all communication is HTTP/WebSocket.
- Session state is in-memory (no database). Restarting the server clears state.
- MCP server exposes exactly 15 tools, all async.

---

## 2. FILE INVENTORY

All paths are relative to the project root (`comfyui-video-mcp/`).

| File | Role |
|------|------|
| `server.py` | FastMCP server; 15 tools exposed via stdio MCP protocol |
| `skills_engine.py` | 15 SkillSpec dataclasses + detect/build/override functions |
| `idea_generator.py` | IdeaGenerator class; LLM calls; scene continuity system |
| `comfyui_client.py` | Async ComfyUIClient; queue/status/history/download |
| `montage_compiler.py` | FFmpeg wrapper; compile_montage() |
| `session.py` | SessionState dataclass; in-memory pipeline state |
| `config.yaml` | All configuration (YAML) |
| `run_generate.py` | Standalone script; bypasses MCP; queues 4 scenes directly |
| `workflows/wan22_lightx2v_api.json` | **WORKING** — Wan2.2 + LightX2V 4-step LoRA |
| `workflows/ltxvideo_api.json` | LTX-Video workflow (model may not be installed) |
| `workflows/ltxvideo_camera_api.json` | LTX-Video + camera LoRA workflow |
| `workflows/wan22_t2v_api.json` | Wan2.2 plain (no LoRA) workflow |
| `comfyUI_workflow/video_wan2_2_14B_t2v.json` | Original ComfyUI GUI format (reference only, not for API) |
| `modelScaffold/structure.txt` | Snapshot of remote ComfyUI model directory tree |

### Output Directories (auto-created)

```
output/
  videos/       ← individual scene videos downloaded from ComfyUI
  montages/     ← compiled montage files from FFmpeg
```

---

## 3. ENVIRONMENT AND EXECUTION

### Python Environment

```bash
# ALWAYS use the project venv — system Python lacks dependencies
./venv/Scripts/python.exe       # Windows
./venv/Scripts/pip.exe install  # Windows
```

Do NOT use `py.exe`, `python`, or `python3` — these may resolve to the system interpreter.

### Starting the MCP Server

```bash
./venv/Scripts/python.exe server.py
```

MCP transport: stdio. Claude Desktop or any MCP client connects via process stdin/stdout.

### Running the Standalone Generator

```bash
./venv/Scripts/python.exe run_generate.py
```

This bypasses MCP entirely and queues scenes directly. Useful for testing or batch runs outside an agent session.

### Environment Variables

| Variable | Effect |
|----------|--------|
| `ANTHROPIC_API_KEY` | If set, IdeaGenerator uses Claude API (claude-opus-4-6) |
| *(absent)* | Falls back to Ollama (localhost:11434), then offline templates |

### Remote ComfyUI Machine

- **Host:** `192.168.1.196`
- **Port:** `8188`
- **Connection type:** LAN HTTP/WebSocket
- **GPU:** RTX 4090D
- **ETA per scene:** ~108 seconds

---

## 4. COMFYUI API FORMAT

**Critical:** The GUI workflow JSON (`comfyUI_workflow/` files) uses a `"nodes"` array format. The API format is a **flat dict keyed by node ID string**. These are NOT interchangeable.

### API Format Structure

```json
{
  "NODE_ID_STRING": {
    "class_type": "NodeClassName",
    "inputs": {
      "param_name": "value",
      "link_param": ["source_node_id_string", output_slot_integer]
    }
  }
}
```

### Link Format

Node-to-node connections use a 2-element array: `["source_node_id", output_slot]`

```json
"model": ["75", 0]    ← connect to node "75", first output (slot 0)
"clip":  ["71", 0]    ← connect to node "71", first output (slot 0)
```

### Submission Endpoint

```
POST http://192.168.1.196:8188/prompt
Content-Type: application/json

{
  "prompt": { ...workflow_dict... },
  "client_id": "UUID-string"
}
```

Response:
```json
{ "prompt_id": "uuid-string" }
```

### Other ComfyUI Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/queue` | Queue status (running + pending) |
| `GET` | `/history/{prompt_id}` | Job result + output file paths |
| `GET` | `/view?filename=X&subfolder=Y&type=output` | Download output file |
| `GET` | `/object_info` | Available models and node types |
| `WS` | `/ws?client_id=X` | Real-time progress events |

---

## 5. ACTIVE WORKFLOW: wan22_lightx2v_api.json

This is the **primary working workflow**. Use this unless explicitly overriding.

### Template Placeholders

All placeholders are double-brace format: `{{PLACEHOLDER_NAME}}`

| Placeholder | Type | Default | Notes |
|-------------|------|---------|-------|
| `{{POSITIVE_PROMPT}}` | str | — | Required. From skills engine. |
| `{{NEGATIVE_PROMPT}}` | str | — | Required. From skills engine. |
| `{{WIDTH}}` | int | 640 | Pixel width |
| `{{HEIGHT}}` | int | 640 | Pixel height |
| `{{FRAMES}}` | int | 81 | Total frames |
| `{{FPS}}` | int | 16 | Frames per second (24 for anime) |
| `{{SEED}}` | int | random | Noise seed for KSampler |
| `{{OUTPUT_PREFIX}}` | str | — | Filename prefix for saved video |

### Node Graph — Complete

```
Node "75": UNETLoader
  inputs:
    unet_name: "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
    weight_dtype: "default"

Node "76": UNETLoader
  inputs:
    unet_name: "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
    weight_dtype: "default"

Node "83": LoraLoaderModelOnly
  inputs:
    model: ["75", 0]
    lora_name: "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors"
    strength_model: 1.0

Node "85": LoraLoaderModelOnly
  inputs:
    model: ["76", 0]
    lora_name: "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors"
    strength_model: 1.0

Node "82": ModelSamplingSD3
  inputs:
    model: ["83", 0]
    shift: 5.0

Node "86": ModelSamplingSD3
  inputs:
    model: ["85", 0]
    shift: 5.0

Node "71": CLIPLoader
  inputs:
    clip_name: "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    type: "wan"

Node "73": VAELoader
  inputs:
    vae_name: "wan_2.1_vae.safetensors"

Node "89": CLIPTextEncode  ← POSITIVE
  inputs:
    clip: ["71", 0]
    text: "{{POSITIVE_PROMPT}}"

Node "72": CLIPTextEncode  ← NEGATIVE
  inputs:
    clip: ["71", 0]
    text: "{{NEGATIVE_PROMPT}}"

Node "74": EmptyHunyuanLatentVideo
  inputs:
    width: {{WIDTH}}
    height: {{HEIGHT}}
    length: {{FRAMES}}
    batch_size: 1

Node "81": KSamplerAdvanced  ← STAGE 1 (noise steps 0-2)
  inputs:
    model: ["82", 0]
    positive: ["89", 0]
    negative: ["72", 0]
    latent_image: ["74", 0]
    add_noise: "enable"
    noise_seed: {{SEED}}
    steps: 4
    cfg: 1.0
    sampler_name: "euler"
    scheduler: "simple"
    start_at_step: 0
    end_at_step: 2
    return_with_leftover_noise: "enable"

Node "78": KSamplerAdvanced  ← STAGE 2 (refine steps 2-4)
  inputs:
    model: ["86", 0]
    positive: ["89", 0]
    negative: ["72", 0]
    latent_image: ["81", 0]
    add_noise: "disable"
    noise_seed: 0
    steps: 4
    cfg: 1.0
    sampler_name: "euler"
    scheduler: "simple"
    start_at_step: 2
    end_at_step: 4
    return_with_leftover_noise: "disable"

Node "87": VAEDecode
  inputs:
    samples: ["78", 0]
    vae: ["73", 0]

Node "88": CreateVideo
  inputs:
    images: ["87", 0]
    fps: {{FPS}}

Node "80": SaveVideo
  inputs:
    video: ["88", 0]
    filename_prefix: "{{OUTPUT_PREFIX}}"
    format: "auto"
    codec: "auto"
```

### Two-Stage Sampling Architecture

LightX2V uses a dual-UNet two-pass approach:

```
High-noise UNet (75) → LoRA (83) → ModelSamplingSD3 shift=5.0 (82)
                                              ↓
                                     KSamplerAdvanced Stage 1
                                     (adds noise, steps 0→2)
                                              ↓
Low-noise UNet (76) → LoRA (85) → ModelSamplingSD3 shift=5.0 (86)
                                              ↓
                                     KSamplerAdvanced Stage 2
                                     (refines, steps 2→4)
                                              ↓
                                         VAEDecode → CreateVideo → SaveVideo
```

**NEVER change steps from 4 or cfg from 1.0.** The LoRA schedule is calibrated to exactly these values.

---

## 6. SKILLS ENGINE

### File: `skills_engine.py`

### SkillSpec Dataclass Fields

```python
@dataclass
class SkillSpec:
    id: str                      # unique identifier, e.g. "anime"
    name: str                    # human name, e.g. "Anime & Manga"
    description: str             # one-line summary
    keywords: list[str]          # scored against user notes for auto-detection
    quality_boosters: list[str]  # appended to positive prompt
    negative_tags: list[str]     # appended to negative prompt
    camera_vocabulary: list[str] # precise camera moves with velocity specs
    lighting_vocabulary: list[str] # Kelvin-precise lighting descriptions
    hook_patterns: list[str]     # 2-second opening hooks for scenes
    style_tags: list[str]        # style identifiers appended to positive
    technical_specs: dict        # {fps, width, height, steps, cfg}
    prompt_template: str         # LLM system prompt persona for this skill
    comfyui_notes: str           # ComfyUI-specific guidance
```

### 15 Skill IDs

```
cinematic    anime        3d_cgi       cartoon      fight_scenes
motion_design ecommerce   social_hook  music_video  brand_story
fashion      food         real_estate  (+ 2 more)
```

### Key Functions

#### `detect_skill(notes: str, override: Optional[str] = None) → SkillSpec`

- Scores each skill's `keywords` list against `notes` (case-insensitive substring matching)
- Returns the highest-scoring SkillSpec
- Defaults to `"cinematic"` if no keywords match
- If `override` is provided (a skill ID string), returns that skill directly

**Anime skill keywords** (relevant — it handles cyberpunk/tokyo/neon content):
```
anime, manga, japanese, shonen, seinen, magical girl, mecha, kawaii, otaku,
sakura, ninja, samurai, slice of life, cyberpunk anime, cyberpunk, neon city,
neon lights, tokyo, akihabara, shibuya
```

#### `build_comfyui_positive(base_prompt: str, skill: SkillSpec, include_quality_boosters=True) → str`

Returns:
```
"base_prompt, booster1, booster2, ..., style_tag1, style_tag2, ..."
```

#### `build_comfyui_negative(skill: SkillSpec, custom_negative: str = "") → str`

Returns:
```
"custom_negative, ugly, deformed, noisy, blurry, ..., skill.negative_tags..."
```

#### `get_workflow_overrides(skill: SkillSpec) → dict`

Returns:
```python
{
    "fps": skill.technical_specs["fps"],
    "width": skill.technical_specs["width"],
    "height": skill.technical_specs["height"],
    "steps": skill.technical_specs["steps"],
    "cfg": skill.technical_specs["cfg"]
}
```

#### `list_skills() → list[dict]`

Returns:
```python
[{"id": "...", "name": "...", "description": "..."}, ...]
```

---

## 7. IDEA GENERATOR

### File: `idea_generator.py`
### Class: `IdeaGenerator(config: dict)`

### Constructor Config Keys

```python
config = {
    "provider": "auto",          # "auto" | "claude" | "ollama" | "offline"
    "claude_model": "claude-opus-4-6",
    "ollama_model": "llama3.2",
    "ollama_host": "http://localhost:11434"
}
```

### Provider Resolution (`_effective_provider()`)

```
provider == "auto"
      ↓
ANTHROPIC_API_KEY set? → use "claude"
      ↓ (no)
Ollama reachable at localhost:11434? → use "ollama"
      ↓ (no)
→ use "offline"  (hardcoded template fallback)
```

### Methods

#### `generate_ideas(notes, count=5, skill_id=None) → tuple[list[dict], SkillSpec]`

Returns a tuple:
- `ideas`: list of idea dicts
- `skill`: the detected SkillSpec

**Idea dict schema:**
```python
{
    "title": str,        # short evocative title
    "description": str,  # 2-3 sentence concept description
    "style": str,        # visual style descriptor
    "mood": str,         # emotional tone
    "tags": list[str]    # searchable tags
}
```

#### `regenerate_ideas(notes, feedback, count=5, skill_id=None) → tuple[list[dict], SkillSpec]`

Same return schema as `generate_ideas`. The `feedback` string is appended to the LLM prompt to steer regeneration.

#### `generate_scenes(idea, scene_count=4, skill=None) → list[dict]`

**Scene dict schema:**
```python
{
    "scene_number": int,          # 1-indexed
    "act": str,                   # "HOOK" | "BUILD" | "CLIMAX" | "RESOLUTION"
    "protagonist": str,           # character/subject anchor string
    "description": str,           # human-readable scene summary
    "visual_prompt": str,         # ComfyUI positive prompt (full, enriched)
    "negative_prompt": str,       # ComfyUI negative prompt
    "duration": float,            # seconds (typically 3.0-5.0)
    "transition_to_next": str     # e.g. "dissolve", "cut", "match_cut"
}
```

**Scene generation pipeline:**
```
LLM call (with skill persona)
      ↓
_enforce_continuity(scenes, skill)
      ↓
_enrich_scene(scene, skill)  ← per scene
      ↓
Return enriched scene list
```

### Scene Continuity System

#### `_ACT_LABELS`
```python
["HOOK", "BUILD", "CLIMAX", "RESOLUTION"]
```

#### `_extract_protagonist_anchor(scenes) → str`

Pulls from `scenes[0]["protagonist"]` or first 2-3 descriptive clauses of `scenes[0]["visual_prompt"]`. Used as the anchor for enforcing continuity.

#### `_enforce_continuity(scenes, skill) → list[dict]`

- Re-numbers scenes (1-indexed)
- Assigns act labels from `_ACT_LABELS` (truncated/padded to actual scene count)
- For scenes 2+: if `anchor` string not found in `visual_prompt`, prepends `"same protagonist — [anchor], "` to the prompt

#### `_build_protagonist(idea, skill) → str`

Used only in offline mode. Generates a style-aware character description from the idea dict and skill's `style_tags`.

### Ollama API Note

**Do NOT** use `"format": "json"` in the Ollama API request body. This forces single-object output and breaks JSON array responses.

Use instead:
```python
{
    "num_predict": 8192,
    "num_ctx": 8192
}
```

---

## 8. MCP SERVER TOOLS (15 tools)

### File: `server.py`
### Transport: stdio via FastMCP
### All tools are async

---

### Tool 1: `list_skills()`

**Returns:** Formatted string listing all 15 skills with id, name, description.

---

### Tool 2: `detect_skill_for_notes(notes: str)`

**Returns:** String showing which skill auto-detects from the notes, plus its workflow overrides (fps, width, height, steps, cfg).

---

### Tool 3: `generate_ideas(notes: str, count: int = 5, skill_id: str = None)`

**Behavior:**
1. Calls `IdeaGenerator.generate_ideas(notes, count, skill_id)`
2. Stores ideas and detected skill in session
3. Returns formatted string with numbered idea list

**Session side-effects:** `session.notes`, `session.ideas`, `session.skill` updated.

---

### Tool 4: `list_ideas()`

**Returns:** Formatted list of ideas currently in session (from last `generate_ideas` call).

---

### Tool 5: `regenerate_ideas(feedback: str, count: int = 5, skill_id: str = None)`

**Behavior:**
1. Uses `session.notes` from previous call
2. Calls `IdeaGenerator.regenerate_ideas(notes, feedback, count, skill_id)`
3. Replaces `session.ideas` with new results

**Requires:** Prior `generate_ideas` call (needs `session.notes`).

---

### Tool 6: `select_idea(idea_id: int)`

**Behavior:**
1. Sets `session.selected_idea = session.ideas[idea_id - 1]` (1-indexed)
2. Calls `IdeaGenerator.generate_scenes(idea, 4, session.skill)`
3. Stores result in `session.scenes`
4. Returns scene summaries

**Session side-effects:** `session.selected_idea`, `session.scenes` updated.

---

### Tool 7: `configure_pipeline(model=None, width=None, height=None, frames=None, fps=None, steps=None, cfg=None, llm_provider=None)`

**Behavior:** Updates `session.pipeline_config` with provided non-None values. All params optional.

**Valid model IDs:** `wan22_lightx2v`, `wan22`, `ltxvideo`, `ltxvideo_fast`, `ltxvideo_camera`

---

### Tool 8: `generate_video(scene_id=None, all_scenes=True, model=None, width=None, height=None, frames=None, fps=None, steps=None, cfg=None)`

**Behavior:**
1. Loads workflow JSON for selected model
2. Substitutes template placeholders from scene prompts + config
3. POSTs to ComfyUI `/prompt`
4. Stores returned job IDs in `session.generation_jobs` keyed by `scene_number`
5. Returns job IDs and ETA

**If `all_scenes=True`:** Queues all scenes in `session.scenes`.
**If `scene_id` specified:** Queues that single scene only.

**Session side-effects:** `session.generation_jobs` updated.

---

### Tool 9: `check_status(job_id=None, wait=False)`

**Behavior:**
- If `job_id` specified: checks that specific job
- If no `job_id`: checks all jobs in `session.generation_jobs`
- If `wait=True`: polls until completion, then downloads videos to `output/videos/`
- Returns status string (queued / running / complete / error)

---

### Tool 10: `list_videos()`

**Returns:** Formatted list of files in `output/videos/`.

---

### Tool 11: `compile_montage(title=None, transition="dissolve", music_path=None, resolution=None, fps=24, add_title_card=False, video_ids=None, use_session_videos=True)`

**Behavior:**
1. Selects videos: if `use_session_videos=True`, uses videos from current session jobs; else uses `video_ids` list
2. Calls `montage_compiler.compile_montage()`
3. Saves to `output/montages/`
4. Appends job record to `session.montage_jobs`

**Transition options:** `dissolve`, `cut`, `fade`, `wipe`

---

### Tool 12: `list_montages()`

**Returns:** Formatted list of files in `output/montages/`.

---

### Tool 13: `ping_comfyui()`

**Returns:** Connectivity status for `192.168.1.196:8188`, current queue depth (running + pending counts).

---

### Tool 14: `get_available_models()`

**Behavior:** GETs `/object_info` from ComfyUI, extracts model lists for UNETLoader, CLIPLoader, VAELoader, LoraLoaderModelOnly, CheckpointLoaderSimple.

**Returns:** Formatted string of available model filenames per loader type.

---

### Tool 15: `session_status()`

**Returns:** Full dump of current `SessionState`:
- Notes summary
- Detected skill
- Ideas count + titles
- Selected idea
- Scenes (with truncated prompts)
- Generation jobs (scene → job_id → status)
- Montage jobs
- Pipeline config

---

## 9. SESSION STATE

### File: `session.py`
### Class: `SessionState` (dataclass, in-memory)

```python
@dataclass
class SessionState:
    notes: str = ""
    ideas: list[dict] = field(default_factory=list)
    selected_idea: Optional[dict] = None
    scenes: list[dict] = field(default_factory=list)
    generation_jobs: dict[int, str] = field(default_factory=dict)
    # key: scene_number (int), value: comfyui_job_id (UUID string)
    montage_jobs: list[dict] = field(default_factory=list)
    skill: Optional[SkillSpec] = None
    pipeline_config: dict = field(default_factory=dict)
    # pipeline_config keys: model, width, height, frames, fps, steps, cfg, llm_provider
```

**Important:** State is lost on server restart. There is no persistence layer.

---

## 10. COMFYUI CLIENT

### File: `comfyui_client.py`
### Class: `ComfyUIClient(host: str, port: int)`

All methods are async (`await` required).

```python
client = ComfyUIClient("192.168.1.196", 8188)

# Queue a workflow
job_id: str = await client.queue_prompt(workflow_dict)

# Check queue
status: dict = await client.get_queue_status()
# Returns: {"queue_running": [...], "queue_pending": [...]}

# Get job result
history: dict = await client.get_history(prompt_id)
# Returns job history dict including output file paths

# Check availability
alive: bool = await client.is_available()

# Download video
# Uses GET /view?filename=X&subfolder=Y&type=output
await client.download_video(filename, subfolder, dest_path)
```

### Job ID Lifecycle

```
POST /prompt → {"prompt_id": "uuid"}
                    ↓
GET /queue → appears in queue_pending
                    ↓
GET /queue → moves to queue_running
                    ↓
GET /history/{uuid} → contains output file info
```

---

## 11. MONTAGE COMPILER

### File: `montage_compiler.py`
### Function: `compile_montage(video_paths, output_path, transition, fps, resolution, title, music_path, add_title_card)`

Wraps FFmpeg. Requires FFmpeg in system PATH.

**Output format:** MP4 (H.264)
**Default resolution:** `1280x720`
**Default transition:** `dissolve` (0.5s)
**Music volume:** 0.3 (30% relative to video audio)

---

## 12. CONFIG.YAML REFERENCE

```yaml
comfyui:
  host: "192.168.1.196"
  port: 8188
  timeout: 600          # seconds before job assumed failed
  output_dir: "output"

pipeline:
  default_model: "wan22_lightx2v"
  default_frames: 25
  default_fps: 24
  default_width: 768
  default_height: 512
  default_steps: 20
  default_cfg: 3.0

models:
  wan22_lightx2v:       # ← PRIMARY MODEL
    unet_high: "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
    unet_low: "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
    lora_high: "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors"
    lora_low: "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors"
    text_encoder: "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    vae: "wan_2.1_vae.safetensors"
    workflow: "workflows/wan22_lightx2v_api.json"
    steps_override: 4       # FIXED — do not change
    cfg_override: 1.0       # FIXED — do not change
    shift: 5.0              # ModelSamplingSD3 shift — do not change
    default_width: 640
    default_height: 640
    default_frames: 81
    default_fps: 16

  wan22:
    checkpoint: "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
    text_encoder: "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    vae: "wan_2.1_vae.safetensors"
    workflow: "workflows/wan22_t2v_api.json"

  ltxvideo:
    # Stored in models/checkpoints/ — use CheckpointLoaderSimple
    # workflow: "workflows/ltxvideo_api.json"

  ltxvideo_fast:
    # workflow: "workflows/ltxvideo_api.json" with reduced steps

  ltxvideo_camera:
    # workflow: "workflows/ltxvideo_camera_api.json"

idea_generation:
  provider: "auto"
  claude_model: "claude-opus-4-6"
  ollama_model: "llama3.2"
  ollama_host: "http://localhost:11434"

skills:
  default_skill: "auto"
  inject_quality_boosters: true
  inject_style_tags: true

montage:
  default_transition: "dissolve"
  transition_duration: 0.5
  default_resolution: "1280x720"
  default_fps: 24
  music_volume: 0.3
```

---

## 13. KNOWN CONSTRAINTS AND GOTCHAS

### CRITICAL: Model Loader Types

```
UNETLoader       → scans ComfyUI/models/diffusion_models/ ONLY
CheckpointLoader → scans ComfyUI/models/checkpoints/ ONLY
```

- Wan2.2 models are in `diffusion_models/` → use `UNETLoader`
- LTX models are in `checkpoints/` → use `CheckpointLoaderSimple`
- **Never use UNETLoader for LTX models or vice versa**

### CRITICAL: LightX2V Fixed Parameters

```
steps = 4       ← FIXED. The 4-step LoRA is calibrated to exactly 4 steps.
cfg = 1.0       ← FIXED. Flow matching models use cfg=1.0.
shift = 5.0     ← FIXED. ModelSamplingSD3 shift for LightX2V.
```

Changing any of these breaks the LoRA schedule and produces garbage output.

### CRITICAL: KSamplerAdvanced Stage Settings

```
Stage 1 (node "81"):
  add_noise: "enable"
  return_with_leftover_noise: "enable"
  start_at_step: 0
  end_at_step: 2

Stage 2 (node "78"):
  add_noise: "disable"
  return_with_leftover_noise: "disable"
  start_at_step: 2
  end_at_step: 4
```

Swapping these settings breaks the two-stage diffusion process.

### Output Node: Use CreateVideo + SaveVideo

```python
# CORRECT:
Node "88": CreateVideo
Node "80": SaveVideo

# WRONG (may fail in newer ComfyUI builds):
# VHS_VideoCombine requires pingpong and save_output params
```

### Ollama API: No JSON Format Flag

```python
# WRONG — breaks array responses:
{"format": "json", ...}

# CORRECT:
{"num_predict": 8192, "num_ctx": 8192}
```

### Python Environment

```bash
# WRONG — may use system Python missing dependencies:
python server.py
py server.py

# CORRECT:
./venv/Scripts/python.exe server.py
```

### Scene Continuity

Scene 1 must define a clear protagonist. `_enforce_continuity()` will inject continuity into scenes 2+ if the anchor drifts, but the best practice is to set it explicitly in the LLM prompt for Scene 1.

### ComfyUI Job IDs

Job IDs are UUIDs returned in the `prompt_id` field of the POST response, not from the queue status endpoint. Store these from the initial queue call.

### Fashion Skill Keyword Note

The `"style"` keyword was previously in the fashion skill keywords list and caused false-positive matches on unrelated content. It has been removed. Fashion skill now requires more specific keywords.

---

## 14. TYPICAL AGENT WORKFLOW (STEP-BY-STEP)

### Step 1: Verify Connectivity

```
ping_comfyui()
```

Expected response: ComfyUI reachable at 192.168.1.196:8188, queue depth N.
If unreachable: check LAN, confirm remote machine is running.

### Step 2: Generate Ideas

```
generate_ideas("your creative brief or notes here")
```

Optional — force a specific skill:
```
generate_ideas("cyberpunk tokyo street scene", skill_id="anime")
```

### Step 3: Review Ideas

```
list_ideas()
```

Review the 5 generated concepts. If unsatisfactory:
```
regenerate_ideas("make it darker and more action-focused")
```

### Step 4: Select Idea and Generate Scenes

```
select_idea(3)   ← 1-indexed
```

This automatically generates 4 scenes with act structure: HOOK → BUILD → CLIMAX → RESOLUTION.

### Step 5: Review Pipeline (Optional)

```
session_status()
```

Inspect the 4 scenes, their visual prompts, and current pipeline config.

Adjust if needed:
```
configure_pipeline(model="wan22_lightx2v", width=640, height=640, frames=81, fps=16)
```

### Step 6: Generate Videos

```
generate_video()
```

This queues all 4 scenes. Each returns a `prompt_id` (UUID) stored in `session.generation_jobs`.

Wait for completion:
```
check_status(wait=True)
```

ETA: ~108 seconds per scene on RTX 4090D → ~7 minutes for 4 scenes.

### Step 7: Compile Montage

```
compile_montage(title="My Film Title", transition="dissolve")
```

View result:
```
list_montages()
```

---

## 15. ALTERNATIVE: STANDALONE SCRIPT

Use `run_generate.py` when the MCP server is not running, for testing, or for batch operations.

```python
# run_generate.py — key imports and pattern
from skills_engine import detect_skill, build_comfyui_positive, build_comfyui_negative

# Auto-detect skill from creative brief
SKILL = detect_skill("your creative brief here")

# Build prompts
visual_prompt = build_comfyui_positive(
    "base scene description",
    SKILL,
    include_quality_boosters=True
)

negative_prompt = build_comfyui_negative(
    SKILL,
    custom_negative="static, frozen, text, watermark"
)

# Prompts are now ready for ComfyUI workflow substitution
```

Execute:
```bash
./venv/Scripts/python.exe run_generate.py
```

The script queues 4 scenes directly to ComfyUI using the active workflow template.

---

## 16. WORKFLOW NODE GRAPHS (ALL WORKFLOWS)

### workflows/wan22_t2v_api.json (Plain Wan2.2, no LoRA)

This is a single-stage workflow. Uses `KSampler` (not `KSamplerAdvanced`).

Key differences from `wan22_lightx2v_api.json`:
- Single UNETLoader (high-noise model only)
- No LoraLoaderModelOnly nodes
- No ModelSamplingSD3 node
- Single KSampler (not two-stage)
- Steps: 20, CFG: 3.0 (standard values)

### workflows/ltxvideo_api.json (LTX-Video)

Uses `CheckpointLoaderSimple`, NOT `UNETLoader`. LTX model lives in `models/checkpoints/`.

**Warning:** LTX model may not be installed on the remote machine. Run `get_available_models()` to verify before use.

### workflows/ltxvideo_camera_api.json (LTX-Video + Camera LoRA)

Extends `ltxvideo_api.json` with a LoRA for camera movement control.

---

## 17. MODEL DIRECTORY MAPPING

Based on `modelScaffold/structure.txt` (snapshot of remote ComfyUI installation).

```
ComfyUI/models/
  diffusion_models/           ← UNETLoader scans here
    wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors
    wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors

  checkpoints/                ← CheckpointLoaderSimple scans here
    [LTX model files if installed]

  loras/                      ← LoraLoader scans here
    wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors
    wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors

  clip/                       ← CLIPLoader scans here
    umt5_xxl_fp8_e4m3fn_scaled.safetensors

  vae/                        ← VAELoader scans here
    wan_2.1_vae.safetensors
```

To verify current state of remote model directories:
```
get_available_models()   ← queries ComfyUI /object_info live
```

---

## 18. ERROR RECOVERY PROCEDURES

### ComfyUI Unreachable

```
Symptom: ping_comfyui() returns error or timeout
Cause: Remote machine offline, ComfyUI not started, LAN issue
Fix: 
  1. Verify network connectivity to 192.168.1.196
  2. Confirm ComfyUI is running on remote machine
  3. Check port 8188 is not firewalled
```

### Job Stuck in Queue

```
Symptom: check_status() shows job pending for >5 minutes
Cause: ComfyUI queue backed up, or previous job errored
Fix:
  1. Check ComfyUI web UI at http://192.168.1.196:8188
  2. Clear queue if needed via ComfyUI UI
  3. Re-queue with generate_video()
```

### Bad Video Quality

```
Symptom: Generated video is noisy, artifacts, or wrong motion
Common causes and fixes:

1. Steps or CFG changed from LightX2V defaults
   Fix: configure_pipeline(steps=4, cfg=1.0) — ALWAYS for wan22_lightx2v

2. ModelSamplingSD3 shift not set correctly
   Fix: Verify shift=5.0 in workflow JSON nodes "82" and "86"

3. KSamplerAdvanced stage settings swapped
   Fix: Verify Stage 1 has add_noise="enable", Stage 2 has add_noise="disable"

4. Wrong model used with wrong loader
   Fix: Wan2.2 → UNETLoader; LTX → CheckpointLoaderSimple
```

### LLM Provider Fallback Chain

```
Symptom: Ideas generated seem generic or templated
Cause: Running in offline mode (no API key, no Ollama)
Fix:
  1. Set ANTHROPIC_API_KEY environment variable, OR
  2. Start Ollama: ollama serve (ensure llama3.2 pulled)
  3. Verify: detect_skill_for_notes("test") — should show provider in use
```

### Protagonist Drift in Scenes

```
Symptom: Scene 2/3/4 prompts describe a different character than Scene 1
Cause: LLM changed subject; _enforce_continuity() may not have caught it
Fix:
  1. Review scenes via session_status()
  2. Use regenerate_ideas() with feedback: "maintain consistent protagonist"
  3. After select_idea(), manually inspect scene visual_prompts
  4. Re-run select_idea() if drift is severe (this re-runs generate_scenes)
```

### Ollama Returning Malformed JSON

```
Symptom: idea_generator.py raises JSON parse error
Cause: Ollama called with format:"json" or context too short
Fix: Verify Ollama API call uses num_predict=8192, num_ctx=8192
     Do NOT use format:"json" parameter
```

### Session State Lost

```
Symptom: list_ideas() returns empty after server restart
Cause: SessionState is in-memory only
Fix: Re-run the full pipeline from generate_ideas()
     There is no session persistence — this is by design
```

---

*End of AGENT_CONTEXT.md*
*This document is the authoritative specification for autonomous operation of ComfyUI Video MCP.*
*When in doubt, ping_comfyui() first, then follow the 7-step workflow in Section 14.*
