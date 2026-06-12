# AI-First Video Generation Pipeline — Implementation Plan

**Project:** comfyui-video-mcp  
**Last updated:** 2026-04-18  
**Status:** Sprint 1 — COMPLETE ✅ | Sprint 2 — COMPLETE ✅ | Sprint 3 — COMPLETE ✅ | Sprint 4 — COMPLETE ✅

---

## Table of Contents

1. [Project Philosophy](#1-project-philosophy)
2. [Architecture Overview](#2-architecture-overview)
3. [Technology Stack](#3-technology-stack)
4. [Data Models](#4-data-models)
5. [Workflow Node Maps](#5-workflow-node-maps)
6. [File Structure](#6-file-structure)
7. [Sprint Plan](#7-sprint-plan)
8. [Milestone Tracker](#8-milestone-tracker)
9. [API Reference](#9-api-reference)
10. [Known Constraints](#10-known-constraints)

---

## 1. Project Philosophy

| Role | Responsibility |
|---|---|
| **User** | Provide intent, approve or nudge at key checkpoints |
| **AI (Claude + Skills Engine)** | Infer style, generate story, build scenes, write prompts |
| **ComfyUI** | Execute image and video generation |
| **OpenMontage** | Compile final video from scene clips |

**Three guiding principles:**
- Expose decisions, hide complexity
- Minimize user input — AI fills every gap it can
- Every checkpoint is optional; the user can skip forward or loop back

---

## 2. Architecture Overview

### Pipeline Flow

```
User types one idea
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 1  IDEA INPUT                                     │
│  concept (required) + duration + mood (optional)        │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 2  AI STYLE INFERENCE                             │
│  detect_skill() → StyleDNA                             │
│  User: accept / regenerate / override                   │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 3  STORY OPTIONS                                  │
│  Claude → 3-5 narrative treatments                      │
│  (summary + emotional curve + pacing + reasoning)       │
│  User: select / regenerate / mix                        │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 3.5  CHARACTER LOCKING                            │
│  AI generates protagonist description from story        │
│  User: approve / edit / regenerate                      │
│  Result: Character (description + seed) locked          │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 4  SCENE BREAKDOWN                                │
│  scene_count = duration ÷ 5s                           │
│  Each scene: act, description, camera, lighting, prompt │
│  User: edit / reorder / add / remove                    │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 5  STORYBOARD GENERATION  [T2I — Flux Schnell]   │
│  1-5 images per scene (slider, default 1)               │
│  Character description injected into every prompt       │
│  Same seed base + scene offset for consistency          │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 5.5  IMAGE REVIEW                                 │
│  Filmstrip: all scenes visible at once (continuity)     │
│  Per scene: approve / regenerate with new prompt        │
│  Approved image locked to scene                         │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 6  SCENE FINALIZATION                             │
│  Scene description → video motion prompt                │
│  User can edit each video prompt                        │
│  SceneState = single source of truth                    │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 6.5  CONTINUITY CHECK                             │
│  Filmstrip of all approved images                       │
│  User confirms visual consistency                       │
│  Fix: go back to step 5.5   Ignore: proceed             │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 7  TECHNICAL CONFIGURATION                        │
│  Resolution, fps, frames — auto from skill              │
│  User can override                                      │
│  Model availability check — show missing model warnings │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 8  QUEUE TO COMFYUI  [I2V — LTX-Video 2.3]      │
│  For each scene:                                        │
│    upload_image(approved_image) → server_filename       │
│    fill_workflow(prompt + image + settings)             │
│    queue_prompt() → job_id                              │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 9  PROGRESS MONITOR                               │
│  Live queue status (running / pending per scene)        │
│  Cancel individual scene                                │
│  Auto-retry on failure                                  │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 10  PLAYBACK + EXPORT                             │
│  Per-scene video player                                 │
│  Compile montage → single final video                   │
│  Download button                                        │
└─────────────────────────────────────────────────────────┘
```

### Component Map

```
┌──────────────────────────────────────────────────────────────┐
│  Streamlit UI  (app.py — rebuilt)                           │
│  10-step wizard with session state persistence              │
└──────┬───────────────────┬──────────────────────────────────┘
       │                   │
       ▼                   ▼
┌─────────────┐   ┌─────────────────────────────────────────┐
│ skills_     │   │  pipeline/                              │
│ engine.py   │   │    scene_state.py    (SceneState)       │
│             │   │    project_state.py  (ProjectState)     │
│ detect_     │   │    t2i_engine.py     (Flux Schnell)     │
│ skill()     │   │    i2v_engine.py     (LTX 2.3)         │
│             │   │    style_inference.py (StyleDNA)        │
│ build_      │   └──────────────┬──────────────────────────┘
│ comfyui_    │                  │
│ positive()  │                  ▼
│ negative()  │   ┌─────────────────────────────────────────┐
└─────────────┘   │  comfyui_client.py                      │
                  │    queue_prompt()                        │
                  │    upload_image()          ← NEW         │
                  │    get_output_images()     ← NEW         │
                  │    wait_for_completion()                 │
                  │    download_outputs()                    │
                  └──────────────┬──────────────────────────┘
                                 │
                                 ▼
                  ┌─────────────────────────────────────────┐
                  │  ComfyUI  192.168.1.196:8188            │
                  │                                         │
                  │  T2I: Flux Schnell fp8                  │
                  │    flux1-schnell-fp8.safetensors        │
                  │                                         │
                  │  I2V: LTX-Video 2.3 22B                 │
                  │    ltx-2.3-22b-dev-fp8.safetensors     │
                  │    gemma_3_12B_it_fp4_mixed.safetensors │
                  │    ltx-2.3-22b-distilled-lora-384       │
                  │    ltx-2.3-spatial-upscaler-x2-1.1     │
                  └─────────────────────────────────────────┘
```

---

## 3. Technology Stack

### Video Generation — LTX-Video 2.3 I2V

| Property | Value |
|---|---|
| Model | ltx-2.3-22b-dev-fp8.safetensors |
| Text encoder | gemma_3_12B_it_fp4_mixed.safetensors (Gemma 3 12B) |
| LoRA | ltx-2.3-22b-distilled-lora-384.safetensors (strength 0.5) |
| Upscaler | ltx-2.3-spatial-upscaler-x2-1.1.safetensors |
| Mode | I2V (Image-to-Video) with T2V bypass toggle |
| Audio | Built-in audio generation (LTXVAudioVAE) |
| Pipeline | Two-stage: low-res → latent upscale → high-res refine |
| Default resolution | 1280×720 |
| Default frames | 121 (~5s at 25fps) |
| Default FPS | 25 |
| Workflow file | workflows/ltx23_i2v_api.json |

### Image Generation — Flux Schnell

| Property | Value |
|---|---|
| Model | flux1-schnell-fp8.safetensors (CheckpointLoaderSimple) |
| Steps | 4 |
| CFG | 1.0 |
| Sampler | euler / simple scheduler |
| Default resolution | 1024×1024 |
| Workflow file | workflows/flux_schnell_t2i_api.json |

### Character Consistency Strategy

| Phase | Approach |
|---|---|
| v1 | Detailed character description injected into every scene prompt + same seed base + scene offset |
| v2 | IPAdapter / Flux Redux for reference-image-based consistency |

### Key Decision: T2I → Review → I2V

Images are generated first (T2I), reviewed by the user, and approved images are uploaded to ComfyUI and fed as the reference frame into the I2V workflow. The video prompt describes motion only — the image defines what it looks like.

---

## 4. Data Models

### StyleDNA

```python
@dataclass
class StyleDNA:
    skill_id: str           # e.g. "anime", "cinematic", "3d_cgi"
    skill_name: str
    visual_style: str
    color_palette: list[str]
    lighting_style: str     # from skill.lighting_vocabulary
    camera_language: list[str]  # from skill.camera_vocabulary
    motion_style: str
    quality_boosters: list[str]
    negative_tags: list[str]
    fps: int
    recommended_width: int
    recommended_height: int
```

### Character

```python
@dataclass
class Character:
    id: str                         # uuid
    description: str                # full visual description locked at step 3.5
    reference_image_path: str | None  # local path to reference image (v2: IPAdapter)
    base_seed: int                  # locked seed for consistency
```

### SceneState

```python
@dataclass
class SceneState:
    # Identity
    scene_id: str           # "scene_01", "scene_02", ...
    scene_number: int
    act: str                # HOOK | BUILD | CLIMAX | RESOLUTION | etc.
    description: str        # one-line human-readable summary

    # Composition
    environment: str
    camera_index: int       # index into skill.camera_vocabulary
    lighting_index: int     # index into skill.lighting_vocabulary
    camera: str             # resolved camera description
    lighting: str           # resolved lighting description

    # Prompts
    base_prompt: str        # user-facing scene prompt (with {visual_anchor})
    visual_prompt: str      # full positive prompt after skill injection
    negative_prompt: str    # full negative prompt after skill injection
    video_prompt: str       # motion-focused prompt for I2V (editable at step 6)

    # Seed
    seed: int               # hash(global_seed + scene_number)

    # Storyboard phase
    storyboard_images: list[str]    # local file paths to generated images
    approved_image_path: str | None # local path to approved image
    server_image_filename: str | None  # filename on ComfyUI server after upload

    # Video phase
    video_job_id: str | None    # ComfyUI prompt_id
    video_filename: str | None  # output filename on ComfyUI server
    video_local_path: str | None  # downloaded to local output/

    # Status
    status: str  # pending | generating_image | reviewing | approved |
                 # uploading | generating_video | done | failed
    error: str | None
```

### ProjectState

```python
@dataclass
class ProjectState:
    # Meta
    project_id: str         # uuid
    project_name: str
    created_at: str         # ISO timestamp
    updated_at: str

    # Step 1 — Idea
    idea: str
    duration_seconds: int   # 15 | 30 | 60
    mood: str | None

    # Step 2 — Style
    style_dna: StyleDNA | None

    # Step 3 — Story
    story_options: list[dict]       # raw LLM output: [{summary, arc, reasoning}]
    selected_story_index: int | None

    # Step 3.5 — Character
    character: Character | None
    global_seed: int            # random at project creation, used for all seeds

    # Step 4-6 — Scenes
    scenes: list[SceneState]

    # Step 7 — Technical
    workflow_t2i: str           # "workflows/flux_schnell_t2i_api.json"
    workflow_i2v: str           # "workflows/ltx23_i2v_api.json"
    width: int
    height: int
    frames: int
    fps: int
    images_per_scene: int       # 1-5, default 1

    # Progress
    current_step: int           # 1-10
    output_dir: str             # "output/{project_id}/"
```

**Persistence:** ProjectState serialises to JSON at `projects/{project_name}.json` after every step. UI loads from this file on refresh — no lost work.

---

## 5. Workflow Node Maps

### T2I — Flux Schnell (`workflows/flux_schnell_t2i_api.json`)

Source file: `comfyUI_workflow/flux_schnell.json` (already API format)

| Placeholder | Node | Field | Type |
|---|---|---|---|
| `{{POSITIVE_PROMPT}}` | `"6"` | `inputs.text` | string |
| `{{NEGATIVE_PROMPT}}` | `"33"` | `inputs.text` | string |
| `{{WIDTH}}` | `"27"` | `inputs.width` | int |
| `{{HEIGHT}}` | `"27"` | `inputs.height` | int |
| `{{SEED}}` | `"31"` | `inputs.seed` | int |
| `{{OUTPUT_PREFIX}}` | `"9"` | `inputs.filename_prefix` | string |

**Injection method:** parse JSON → walk dict → replace placeholder strings/ints directly.  
Numeric fields use int injection. String fields use direct Python string assignment.  
No JSON string escaping ever touches prompt text.

### I2V — LTX-Video 2.3 (`workflows/ltx23_i2v_api.json`)

Source file: `comfyUI_workflow/video_ltx2_3_i2v_API.json`

| Placeholder | Node | Field | Type |
|---|---|---|---|
| `{{POSITIVE_PROMPT}}` | `"267:266"` | `inputs.value` | string |
| `{{NEGATIVE_PROMPT}}` | `"267:247"` | `inputs.text` | string |
| `{{IMAGE_FILENAME}}` | `"269"` | `inputs.image` | string (server filename) |
| `{{WIDTH}}` | `"267:257"` | `inputs.value` | int |
| `{{HEIGHT}}` | `"267:258"` | `inputs.value` | int |
| `{{FRAMES}}` | `"267:225"` | `inputs.value` | int |
| `{{FPS}}` | `"267:260"` | `inputs.value` | int |
| `{{SEED}}` | `"267:237"` | `inputs.noise_seed` | int |
| `{{OUTPUT_PREFIX}}` | `"75"` | `inputs.filename_prefix` | string |
| `{{DISABLE_I2V}}` | `"267:201"` | `inputs.value` | bool (false=I2V, true=T2V) |

**Critical:** `{{IMAGE_FILENAME}}` must be a filename that exists in ComfyUI's `input/` folder. Image must be uploaded via `POST /upload/image` before calling `queue_prompt`.

### fill_workflow() Contract (both workflows)

```python
def fill_workflow(wf_template: str, params: dict) -> dict:
    """
    Step 1: Replace NUMERIC/BOOL placeholders via string replace (safe).
    Step 2: json.loads() — string placeholders remain as valid JSON strings.
    Step 3: Walk parsed dict, swap placeholder strings → real Python values.
    Returns ready-to-queue workflow dict.
    """
```

---

## 6. File Structure

```
comfyui-video-mcp/
│
├── app.py                          Streamlit UI (full rebuild in Sprint 3)
├── comfyui_client.py               ComfyUI REST + WebSocket client
│                                     + upload_image()       ← Sprint 1
│                                     + get_output_images()  ← Sprint 1
├── skills_engine.py                13 SkillSpec definitions, detect_skill()
├── run_project.py                  Legacy CLI runner (T2V, keep as-is)
├── run_generate.py                 Legacy Digital Dreamscape (keep as-is)
├── run_ai_rise.py                  Legacy AI Rise (keep as-is)
├── config.yaml                     Server + model config (update Sprint 1)
├── start_ui.bat                    Windows launcher
├── Workflow_Implementation.md      THIS FILE
│
├── workflows/                      API-format workflow templates
│   ├── flux_schnell_t2i_api.json   T2I workflow with {{PLACEHOLDERS}}  ← Sprint 1
│   ├── ltx23_i2v_api.json          I2V workflow with {{PLACEHOLDERS}}  ← Sprint 1
│   ├── wan22_lightx2v_api.json     Legacy T2V (keep)
│   └── (other legacy workflows)
│
├── comfyUI_workflow/               Source workflow files (UI format, reference only)
│   ├── flux_schnell.json           Flux Schnell source (already API format)
│   ├── video_ltx2_3_i2v.json       LTX 2.3 UI format
│   └── video_ltx2_3_i2v_API.json  LTX 2.3 API format (our source)
│
├── pipeline/                       New pipeline data + logic layer ← Sprint 2
│   ├── __init__.py
│   ├── scene_state.py              SceneState dataclass
│   ├── project_state.py            ProjectState dataclass + JSON persistence
│   ├── t2i_engine.py               Flux Schnell generation logic
│   ├── i2v_engine.py               LTX 2.3 generation logic
│   └── style_inference.py          detect_skill() → StyleDNA builder
│
├── projects/                       Saved project YAML/JSON files
│   ├── ai_rise.yaml                Example project
│   ├── template_anime.yaml
│   └── template_cinematic.yaml
│
├── output/                         Downloaded generated files
│   └── {project_id}/
│       ├── storyboard/             T2I generated images
│       └── video/                  I2V generated videos
│
└── tests/
    ├── test_sprint1.py             End-to-end: upload → T2I → I2V ← Sprint 1
    └── test_sprint2.py             Data model + persistence tests  ← Sprint 2
```

---

## 7. Sprint Plan

### Sprint 1 — Foundation
**Goal:** End-to-end pipeline works in Python. No UI. Upload image → generate T2I → feed into I2V → get video out.

**Deliverables:**

#### 1.1 — Create `workflows/flux_schnell_t2i_api.json`
Copy `comfyUI_workflow/flux_schnell.json` and replace hardcoded values with `{{PLACEHOLDERS}}`.

```json
{
  "6":  { "inputs": { "text": "{{POSITIVE_PROMPT}}", ... }},
  "33": { "inputs": { "text": "{{NEGATIVE_PROMPT}}", ... }},
  "27": { "inputs": { "width": "{{WIDTH}}", "height": "{{HEIGHT}}", ... }},
  "31": { "inputs": { "seed": "{{SEED}}", ... }},
  "9":  { "inputs": { "filename_prefix": "{{OUTPUT_PREFIX}}", ... }}
}
```

#### 1.2 — Create `workflows/ltx23_i2v_api.json`
Copy `comfyUI_workflow/video_ltx2_3_i2v_API.json` and replace values per the node map in section 5.

#### 1.3 — Update `comfyui_client.py` — add two methods

**`upload_image(image_bytes, filename) → str`**
```
POST /upload/image
Body: multipart form data
  image: <bytes>
  type: "input"
  overwrite: "true"
Returns: server filename (may differ from input filename)
```

**`get_output_images(history) → list[dict]`**
```
Parse history dict from get_history()
Return list of {filename, subfolder, type} for all images in outputs
```

#### 1.4 — Update `config.yaml`
Add model entries for `flux_schnell` and `ltx23_i2v`:
```yaml
models:
  flux_schnell:
    checkpoint: "flux1-schnell-fp8.safetensors"
    workflow: "workflows/flux_schnell_t2i_api.json"
    default_width: 1024
    default_height: 1024
    steps: 4
    cfg: 1.0

  ltx23_i2v:
    checkpoint: "ltx-2.3-22b-dev-fp8.safetensors"
    text_encoder: "gemma_3_12B_it_fp4_mixed.safetensors"
    lora: "ltx-2.3-22b-distilled-lora-384.safetensors"
    upscaler: "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
    workflow: "workflows/ltx23_i2v_api.json"
    default_width: 1280
    default_height: 720
    default_frames: 121
    default_fps: 25
```

#### 1.5 — Write `tests/test_sprint1.py`
```
1. Ping ComfyUI — assert online
2. Generate T2I image via Flux Schnell with test prompt
3. Wait for completion, download image
4. Upload image to ComfyUI via upload_image()
5. Queue I2V job with that image + test motion prompt
6. Wait for completion, download video
7. Assert video file exists and size > 0
Print: all job IDs, timings, file sizes
```

**Sprint 1 complete when:** `test_sprint1.py` runs end-to-end with no errors and produces a real video file.

---

### Sprint 2 — Data Layer
**Goal:** Project state persists. Pipeline can be resumed. All data objects are structured.

**Deliverables:**

#### 2.1 — `pipeline/__init__.py`
Empty init, exports all public classes.

#### 2.2 — `pipeline/scene_state.py`
SceneState dataclass with `to_dict()` / `from_dict()` for JSON serialisation.

#### 2.3 — `pipeline/project_state.py`
```python
class ProjectState:
    def save(self, path: str)       # write to projects/{name}.json
    def load(cls, path: str)        # classmethod, restore from JSON
    def add_scene(self, scene)
    def get_scene(self, scene_id)
    def update_scene(self, scene_id, **kwargs)
    def next_step(self)
```

#### 2.4 — `pipeline/style_inference.py`
Wraps `detect_skill()` → returns StyleDNA dataclass.  
Maps SkillSpec fields to StyleDNA fields.

#### 2.5 — `pipeline/t2i_engine.py`
```python
async def generate_image(
    client: ComfyUIClient,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    seed: int,
    output_prefix: str,
    output_dir: Path,
) -> Path:
    """Generate one T2I image via Flux Schnell. Returns local file path."""
```

#### 2.6 — `pipeline/i2v_engine.py`
```python
async def generate_video(
    client: ComfyUIClient,
    prompt: str,
    negative_prompt: str,
    image_path: Path,
    width: int,
    height: int,
    frames: int,
    fps: int,
    seed: int,
    output_prefix: str,
    output_dir: Path,
) -> tuple[str, Path]:
    """Upload image, queue I2V job. Returns (job_id, local_video_path)."""
```

#### 2.7 — `tests/test_sprint2.py`
Test all data model serialise/deserialise round-trips.  
Test t2i_engine and i2v_engine using real ComfyUI calls.

**Sprint 2 complete when:** Can create a ProjectState, generate a scene image, generate a scene video, and reload the entire project from disk with full fidelity.

---

### Sprint 3 — Storyboard UI
**Goal:** Full steps 1-6.5 work in the browser. User can go from idea to approved storyboard.

**Deliverables:**

#### 3.1 — Rebuild `app.py` as multi-step wizard

Streamlit session state holds the active `ProjectState`.  
Navigation: numbered step tabs + Back / Next buttons.  
Auto-saves ProjectState after every step completion.

**Step 1 — IDEA INPUT**
- Text area: concept (required)
- Select: duration (15s / 30s / 60s), default 30s
- Text input: mood (optional)
- Button: Generate →

**Step 2 — STYLE INFERENCE**
- Display: inferred skill name, visual style, color palette
- Show camera vocabulary (first 3) and lighting vocabulary (first 3)
- Buttons: Accept / Regenerate / Override (dropdown to pick skill manually)

**Step 3 — STORY OPTIONS**
- Display: 3-5 story treatments (summary + arc + reasoning per card)
- Buttons per card: Select this
- Button: Regenerate all
- Future: Mix elements (v2)

**Step 3.5 — CHARACTER**
- Display: AI-generated protagonist description
- Text area: user can edit
- Show: locked seed value
- Buttons: Accept / Regenerate

**Step 4 — SCENE BREAKDOWN**
- Editable table: act, description, camera dropdown, lighting dropdown, prompt text area
- Buttons: Add scene / Remove last / Reorder (up/down arrows)
- Shows: total duration based on scene count × 5s

**Step 5 — STORYBOARD GENERATION**
- Slider: images per scene (1-5, default 1)
- Button: Generate all storyboard images
- Progress bar per scene
- Shows generated images in a grid as they complete

**Step 5.5 — IMAGE REVIEW**
- Filmstrip: all scenes in a row (continuity view)
- Per scene panel:
  - Image preview
  - Buttons: Approve / Regenerate
  - If Regenerate: text input for additional/replacement prompt
  - Approved scenes show green checkmark
- Button: Continue → (only active when all scenes approved)

**Step 6 — SCENE FINALIZATION**
- For each scene: show approved image + editable video prompt text area
- Default video prompt = scene description
- Note: "Describe motion and action. The image defines what it looks like."
- Button: Finalise all →

**Step 6.5 — CONTINUITY CHECK**
- Filmstrip: all approved images side by side
- Warning if any scene has no approved image
- Buttons: Fix issues (→ back to 5.5) / Looks good →

**Sprint 3 complete when:** User can complete steps 1-6.5 in the browser, and all SceneState objects have approved images and finalized video prompts saved to disk.

---

### Sprint 4 — Video Generation + Playback
**Goal:** Full end-to-end. Queue I2V, monitor, play result, export.

**Deliverables:**

#### 4.1 — Step 7: Technical Configuration

- Show auto-configured values: resolution, fps, frames per scene, estimated duration
- Model availability check: scan ComfyUI `/object_info` endpoint
  - ✅ Model installed → show green
  - ❌ Model missing → show download instructions (HuggingFace URL + wget command)
- Override panel: width, height, frames, fps sliders
- Button: Confirm settings →

#### 4.2 — Step 8: Queue to ComfyUI

- Summary: N scenes, estimated time, server address
- Button: Queue all scenes
- For each scene: upload_image → fill_workflow → queue_prompt
- Shows each job_id as it queues
- Link: Open ComfyUI monitor

#### 4.3 — Step 9: Progress Monitor

- Auto-refreshing status table (per scene):
  - Scene #, act, status (queued / running / done / failed), job_id
- Overall: N/total done progress bar
- Button per scene: Cancel
- On failure: retry options (same settings / reduced resolution)
- Button: Download all completed →

#### 4.4 — Step 10: Playback + Export

- Per scene: inline HTML5 video player, file size, duration
- Button: Compile montage (calls OpenMontage)
  - Transition selector: dissolve / fade / cut
  - Music: none / upload file
- Final video: inline player + Download button

#### 4.5 — Model Availability Feature (reusable)

```python
async def check_model_availability(client: ComfyUIClient) -> dict:
    """
    Query ComfyUI /object_info.
    Return {model_key: {installed: bool, filename: str, download_url: str}}
    """
```

Used in Step 7 UI to show ✅ / ❌ per required model.

**Sprint 4 complete when:** Full 13-step pipeline runs in browser from blank idea to compiled 30-second video. ✅ COMPLETE

---

## 8. Milestone Tracker

| Sprint | Milestone | Status | Completed |
|---|---|---|---|
| 1 | `flux_schnell_t2i_api.json` created with placeholders | ✅ Done | 2026-04-17 |
| 1 | `ltx23_i2v_api.json` created with placeholders | ✅ Done | 2026-04-17 |
| 1 | `comfyui_client.upload_image()` implemented | ✅ Done | 2026-04-17 |
| 1 | `comfyui_client.get_output_images()` implemented | ✅ Done | 2026-04-17 |
| 1 | `config.yaml` updated with flux_schnell + ltx23_i2v | ✅ Done | 2026-04-17 |
| 1 | `tests/test_sprint1.py` passes (24/24 unit tests) | ✅ Done | 2026-04-17 |
| 2 | `pipeline/scene_state.py` with serialisation | ✅ Done | 2026-04-17 |
| 2 | `pipeline/project_state.py` with save/load | ✅ Done | 2026-04-17 |
| 2 | `pipeline/style_inference.py` → StyleDNA | ✅ Done | 2026-04-17 |
| 2 | `pipeline/t2i_engine.py` working | ✅ Done | 2026-04-17 |
| 2 | `pipeline/i2v_engine.py` working | ✅ Done | 2026-04-17 |
| 2 | `tests/test_sprint2.py` passes (55/55 unit tests) | ✅ Done | 2026-04-17 |
| 3 | Step 1-2 UI: Idea + Style inference | ✅ Done | 2026-04-17 |
| 3 | Step 3-3.5 UI: Story options + Character | ✅ Done | 2026-04-17 |
| 3 | Step 4 UI: Scene breakdown editable | ✅ Done | 2026-04-17 |
| 3 | Step 5 UI: Storyboard generation | ✅ Done | 2026-04-17 |
| 3 | Step 5.5 UI: Image review filmstrip | ✅ Done | 2026-04-17 |
| 3 | Step 6-6.5 UI: Finalisation + continuity | ✅ Done | 2026-04-17 |
| 4 | `pipeline/model_checker.py` — REQUIRED_MODELS + check_model_availability() | ✅ Done | 2026-04-18 |
| 4 | `pipeline/video_queue.py` — queue_video_job / get_all_statuses / download | ✅ Done | 2026-04-18 |
| 4 | `pipeline/montage.py` — compile_montage (moviepy + ffmpeg backends) | ✅ Done | 2026-04-18 |
| 4 | `pipeline/__init__.py` updated with all Sprint 4 exports | ✅ Done | 2026-04-18 |
| 4 | Step 10 UI: Technical config + model check (app.py step_10) | ✅ Done | 2026-04-18 |
| 4 | Step 11 UI: Queue all scenes to ComfyUI (app.py step_11) | ✅ Done | 2026-04-18 |
| 4 | Step 12 UI: Progress monitor + per-clip download (app.py step_12) | ✅ Done | 2026-04-18 |
| 4 | Step 13 UI: Playback + montage export (app.py step_13) | ✅ Done | 2026-04-18 |
| 4 | `tests/test_sprint4.py` passes (51/51 unit tests) | ✅ Done | 2026-04-18 |
| 4 | Full suite (tests/): 130/130 unit tests pass, zero regressions | ✅ Done | 2026-04-18 |

**Status key:** ⬜ Pending | 🔄 In Progress | ✅ Done | ❌ Blocked

---

## 9. API Reference

### ComfyUI Endpoints Used

| Endpoint | Method | Purpose |
|---|---|---|
| `/system_stats` | GET | Health check — is server online? |
| `/prompt` | POST | Queue a workflow job → returns prompt_id |
| `/queue` | GET | Queue status → running + pending counts |
| `/history/{prompt_id}` | GET | Job result + output filenames |
| `/upload/image` | POST | Upload image to input/ folder → returns server filename |
| `/view` | GET | Download output file by filename + subfolder |
| `/interrupt` | POST | Cancel current job |
| `/ws?clientId={id}` | WebSocket | Live progress events |

### `/upload/image` Request Format

```
POST http://192.168.1.196:8188/upload/image
Content-Type: multipart/form-data

Fields:
  image     bytes    — image file data
  type      string   — "input"
  overwrite string   — "true"

Response 200:
{
  "name": "scene_01_abc123.png",
  "subfolder": "",
  "type": "input"
}
```

The returned `name` is the filename to put in `node "269" inputs.image`.

---

## 10. Known Constraints

| Constraint | Impact | Mitigation |
|---|---|---|
| ComfyUI image input must be uploaded first | Cannot pass image bytes directly to workflow | upload_image() before queue_prompt() |
| LTX 2.3 text encoder is Gemma 3 12B (large, ~7GB) | Slow to load on first use | Pre-warm: ping model before queueing |
| T2I and I2V share the same GPU | Cannot run in parallel | Sequential: all T2I done → then I2V queue starts |
| LTX 2.3 two-stage pipeline | ~45-90s per scene on RTX 4090D | Show per-scene ETA in monitor |
| Character consistency via prompt only (v1) | Character may drift across scenes | Filmstrip review at step 6.5 is the safeguard |
| Streamlit re-runs entire script on interaction | State management critical | All state in st.session_state via ProjectState object |
| Windows path separators | Workflow dropdown picks wrong file | Use Path.as_posix() consistently — already fixed |

---

*This document is updated at the end of each sprint. Check the Milestone Tracker for current progress.*
