# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

An automated "raw idea → finished video" pipeline that drives a **remote ComfyUI instance** (default `192.168.1.196:8188` on the LAN) to generate multi-scene videos. A keyword-scored *skills engine* injects cinematic domain vocabulary into prompts, an LLM (or Claude Code itself) writes the story/scenes, ComfyUI renders, and FFmpeg/moviepy compiles a montage.

There is **no local ComfyUI** — all generation happens over HTTP to the remote machine. ComfyUI must be reachable for any real generation; most logic is testable offline.

## Two coexisting systems (this is the key thing to understand)

The repo contains two generations of the product. Both are live; know which one you're touching.

1. **Legacy text-to-video MCP server** — `server.py` (FastMCP, 15 tools), `session.py` (in-memory `SessionState`), `idea_generator.py`, `comfyui_client.py`, `montage_compiler.py`. Pipeline: notes → 5 ideas → 4 scenes → Wan2.2 T2V → montage. Documented in `README.md` and `AGENT_CONTEXT.md`. Run standalone via `run_generate.py` / `run_ai_rise.py` / `run_project.py`.

2. **New Streamlit T2I→I2V pipeline** — `app.py` (13-step wizard UI) + the `pipeline/` package. Pipeline: idea → style → story → character → scenes → **Flux Schnell storyboard images** → user approves → **LTX-Video 2.3 image-to-video** → montage. Project state persists to `projects/{name}.json` (`ProjectState`/`SceneState` dataclasses). Documented in `Workflow_Implementation.md`.

3. **Pipeline job-queue MCP server** — `pipeline/mcp_server.py` + `pipeline/ai_bridge.py`. A second MCP server (registered in `.mcp.json`, auto-starts in Claude Code) that lets the Streamlit UI hand AI-generation jobs to *this Claude Code session* instead of an API key. The UI drops JSON job files in `ai_jobs/pending/`; Claude Code reads each job's `prompt_for_human`, generates the result, and calls `pipeline_submit_result`, which writes `ai_jobs/done/`. Activate by telling Claude Code **"start pipeline worker"**. Documented in `AI_Generation_Modes.md`.

`skills_engine.py` and `comfyui_client.py` are shared by all three.

## Commands

Always use the project venv interpreter — system Python lacks the dependencies:

```powershell
.\venv\Scripts\python.exe              # run scripts / modules
.\venv\Scripts\python.exe -m pip install -r requirements.txt   # first-time setup
```

Do **not** use bare `python`, `py`, or `python3`.

```powershell
# Streamlit pipeline UI (the new system)  → http://localhost:8501
.\start_ui.bat
# or: .\venv\Scripts\streamlit run app.py --server.port 8501

# Legacy MCP server (stdio)
.\venv\Scripts\python.exe server.py

# Pipeline job-queue MCP server (normally auto-started by .mcp.json)
.\venv\Scripts\python.exe pipeline\mcp_server.py

# Tests — exclude live tests that need a running ComfyUI
.\venv\Scripts\python.exe -m pytest -m "not live"

# A single test file / single test
.\venv\Scripts\python.exe -m pytest tests\test_prompt_builder.py
.\venv\Scripts\python.exe -m pytest tests\test_response_parser.py::test_name
```

`test_sprint1.py`, `test_sprint2.py`, and `test_sprint4.py` contain **`@pytest.mark.live`** tests that hit the real ComfyUI at `192.168.1.196:8188`. Use `-m "not live"` for offline/CI runs (see `pytest.ini`).

## How a workflow gets queued (critical pattern)

ComfyUI's **API workflow format** is a flat dict keyed by node-ID string (`{"75": {"class_type": ..., "inputs": {...}}}`), with node links as `["source_id", slot_int]`. This is **not** the same as the GUI `"nodes"`-array format in `comfyUI_workflow/*.json` — those are reference sources only. The API templates live in `workflows/*.json` and contain `{{PLACEHOLDER}}` tokens.

`pipeline/utils.py::fill_workflow()` is the canonical injector and the pattern to copy whenever you add a workflow:

1. String-replace **numeric/bool** placeholders (`{{WIDTH}}`, `{{SEED}}`, `{{FRAMES}}`, `{{FPS}}`) while the template is still text — they appear unquoted in the JSON.
2. `json.loads()`.
3. Walk the parsed dict and swap **string** placeholders (`{{POSITIVE_PROMPT}}`, `{{NEGATIVE_PROMPT}}`, `{{OUTPUT_PREFIX}}`, `{{INPUT_IMAGE}}`) for real Python values in place.

Never `json.dumps()`-escape prompt text into the template — that breaks on em-dashes, curly quotes, backslashes, and unicode. Inject the raw string into the already-parsed dict instead. (The legacy `server.py` path had this bug historically; the parse-then-inject order is the fix.)

For **I2V (LTX 2.3)**, the reference image must be uploaded to ComfyUI's `input/` folder via `POST /upload/image` (`comfyui_client.upload_image()`) **before** queuing — the workflow references it by server-side filename, not bytes.

`pipeline/workflow_catalog.py` scans `workflows/*.json` and classifies each template (`t2i`/`i2v`/`t2v`) by its placeholders; templates using tokens `fill_workflow()` can't fill (`{{CFG}}`, `{{STEPS}}`, `{{CHECKPOINT}}`, …) are marked incompatible. The Streamlit UI offers workflow pickers (storyboard step for T2I, technical-config step for video) persisted in `ProjectState.workflow_t2i`/`workflow_i2v`; engines and `video_queue.queue_video_job` honour them. Paths are stored posix-style — Windows backslash paths broke selectbox matching before (commit 07f0fce).

`workflow_catalog.auto_template_workflow()` converts a raw ComfyUI **API-format export** into a template: prompts are found by following the sampler's positive/negative links (with a CLIPTextEncode fallback), dimensions/frames/fps also via `PrimitiveInt` `_meta` titles, seeds/prefix/input-image by field name. The UI exposes this as an upload box. `pipeline/model_catalog.py` detects model-loader slots in a template (generic: any string input with a model-file extension) and lists installed server models per slot from `/object_info` (`comfyui_client.get_object_info()`; supports both classic `[[options], cfg]` and v3 `["COMBO", {options}]` specs). Per-slot overrides persist in `ProjectState.model_overrides_t2i/_i2v` and are applied by `apply_model_overrides()` after `fill_workflow()`.

## Hard constraints — do not "tune" these

- **Wan2.2 + LightX2V** (`wan22_lightx2v`): `steps=4`, `cfg=1.0`, `ModelSamplingSD3 shift=5.0` are calibrated to the 4-step LoRA. Changing any produces garbage. The two `KSamplerAdvanced` stages must keep their split (stage 1: `add_noise=enable`, steps 0→2, leftover noise on; stage 2: `add_noise=disable`, steps 2→4, leftover off).
- **Loader ↔ directory coupling**: `UNETLoader` only scans `models/diffusion_models/` (Wan2.2); `CheckpointLoaderSimple` only scans `models/checkpoints/` (LTX, Flux). Using the wrong loader fails to find the model.
- **Flux Schnell** T2I: `steps=4`, `cfg=1.0`.
- **Ollama** calls must use `num_predict`/`num_ctx`, never `format: "json"` — the JSON-format flag forces a single object and breaks array responses.

## Skills engine

`skills_engine.py` defines ~13–15 `SkillSpec` dataclasses (camera vocab with velocities, Kelvin-precise lighting, quality boosters, negative tags, hook patterns, per-skill resolution/fps/steps/cfg). Keyword scoring in `detect_skill(notes, override=)` picks one (default `cinematic`). The skill's vocabulary is both used as the LLM persona **and** directly appended to the final ComfyUI prompt via `build_comfyui_positive()` / `build_comfyui_negative()` / `get_workflow_overrides()`. `SKILL_USAGE.md` documents each skill and `pipeline/style_inference.py` maps a `SkillSpec` into the `StyleDNA` dataclass the new pipeline uses.

## AI generation modes (new pipeline)

The same prompt text (built by `pipeline/prompt_builder.py`) feeds four interchangeable backends, parsed back by `pipeline/response_parser.py`. Each scene carries a `character_presence` field (`featured`/`background`/`none`, assigned in `story_generator._assign_character_presence`, user-editable in the scene breakdown UI) that gates whether the locked character description is injected — character *consistency* applies only when the protagonist is on screen; `none` scenes are pure establishing/insert shots. Priority: **MCP worker** (this Claude Code session, free) → **Copy-Paste** (manual, any chatbot) → **Mechanical** (offline templates) → **API key** (`ANTHROPIC_API_KEY`, paid, lowest priority by design — the user is on a Max plan). Stages that use AI: story options, character description, scene prompts. All other steps are deterministic.

## Configuration

`config.yaml` holds ComfyUI host/port/timeout, per-model definitions (`wan22_lightx2v`, `wan22`, `ltxvideo*`, `flux_schnell`, `ltx23_i2v`), montage defaults, and idea-generation provider settings. `default_model` is `wan22_lightx2v`, but each model's own `*_override` / `default_*` keys take precedence over the generic `pipeline:` defaults. `.env` (see `.env.example`) supplies `ANTHROPIC_API_KEY` / `OLLAMA_HOST`.

State note: legacy `SessionState` is in-memory (lost on restart); new `ProjectState` persists to `projects/{name}.json` after every step.
