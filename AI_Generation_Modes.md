# AI Generation Modes — Implementation Plan

**Project:** comfyui-video-mcp  
**Last updated:** 2026-04-18  
**Status:** Mode 1 (Copy-Paste) — Pending | Mode 2 (MCP) — Pending

---

## Table of Contents

1. [Philosophy](#1-philosophy)
2. [The Four Modes](#2-the-four-modes)
3. [Priority Cascade](#3-priority-cascade)
4. [Stages That Use AI](#4-stages-that-use-ai)
5. [Mode 1 — Copy-Paste Specification](#5-mode-1--copy-paste-specification)
6. [Mode 2 — MCP Server Specification](#6-mode-2--mcp-server-specification)
7. [UI Changes Required](#7-ui-changes-required)
8. [File Structure Changes](#8-file-structure-changes)
9. [Prompt Templates — Exact Format](#9-prompt-templates--exact-format)
10. [Sprint Plan](#10-sprint-plan)
11. [Milestone Tracker](#11-milestone-tracker)

---

## 1. Philosophy

The user already pays for Claude Code (Max plan). They should not need to pay a second bill for the Anthropic API. The system must work at full AI quality using only what the user already has.

**Four modes, one priority cascade:**

| Priority | Mode | What drives it | Cost |
|---|---|---|---|
| 1 | 🤖 MCP Auto | Claude Code session (Option C) | Already paid |
| 2 | 📋 Copy-Paste | Any chatbot manually | Already paid (Claude.ai) |
| 3 | ⚙️ Mechanical | Python templates | Free |
| 4 | 🔑 API Key | `ANTHROPIC_API_KEY` env var | Separate charge |

**Mode 1 (Copy-Paste) is permanent.** It does not get removed when Mode 2 is built. It serves a different purpose — it allows iteration, image sharing, and conversation-style refinement that an automated pipeline cannot do.

---

## 2. The Four Modes

### Mode 4 — API Key (legacy, lowest priority)
- Existing behaviour: `_claude_options()`, `_claude_character()`, `_claude_visual_prompts_batch()`, `_claude_video_prompts_batch()`
- Active when: `ANTHROPIC_API_KEY` environment variable is set
- Runs silently inside `generate_scenes_from_story()` with no UI interaction
- **Kept for compatibility. Not the preferred path.**

### Mode 3 — Mechanical (always available)
- Existing behaviour: `_offline_options()`, template character, `build_comfyui_positive()`, `build_comfyui_video_prompt()`
- Active when: user explicitly chooses it, or all AI modes are unavailable
- No API calls, no internet, fully offline
- **Permanent. Never removed.**

### Mode 1 — Copy-Paste (new, permanent)
- UI generates a self-contained prompt with all project context
- User copies it, pastes into any chatbot (Claude.ai, ChatGPT, Gemini, this Claude Code session)
- Chatbot responds with strict JSON
- User copies the JSON response, pastes back into a UI text area
- UI parses it and continues the pipeline
- **Supports iteration** — user can paste multiple times, ask for changes, share character images
- **Works with any AI, any device, any chatbot subscription**
- **Permanent. Never removed.**

### Mode 2 — MCP Auto (new, Phase 2)
- Pipeline exposes an MCP server (`pipeline/mcp_server.py`)
- Claude Code connects to it at startup (configured in `.mcp.json`)
- User activates monitoring once per session: *"start pipeline worker"*
- From that point: UI drops jobs, Claude Code processes them automatically, results appear in UI
- Uses Claude Code's intelligence — no API key, no extra cost
- **Requires active Claude Code session**

---

## 3. Priority Cascade

### Detection logic (checked in this order)

```python
def detect_ai_mode(project_state) -> str:
    """
    Returns: "mcp" | "copy_paste" | "api_key" | "mechanical"
    """
    if mcp_worker_active():          # MCP server is up and Claude Code is watching
        return "mcp"
    # copy_paste is a user choice, not auto-detected
    # It is presented as an option whenever MCP is not active
    if os.getenv("ANTHROPIC_API_KEY"):
        return "api_key"
    return "mechanical"
```

### UI mode selector (shown at each AI stage)

When MCP is NOT active, the user sees a mode picker at the top of the relevant step:

```
How should this be generated?

  [ 🤖 Auto (MCP) ]  ← greyed out with "Start worker first" tooltip if inactive
  [ 📋 Copy & Paste ]  ← always active
  [ ⚙️  Mechanical ]   ← always active
  [ 🔑 API Key ]       ← only shown if ANTHROPIC_API_KEY is set
```

The selected mode is stored in `st.session_state.ai_mode` and persists for the session.

### Mode is remembered per-project

`ProjectState` gets a new field: `ai_mode_used: str` — records which mode was used so
regenerations use the same mode by default.

---

## 4. Stages That Use AI

Only three stages need AI generation. All others are deterministic.

| Stage | Step in UI | What AI generates | Iteration? |
|---|---|---|---|
| Stage 3 | Step 3 — Story | 3 narrative treatment options | Yes — regenerate all |
| Stage 4 | Step 4 — Character | Protagonist visual description | Yes — multiple rounds, image support |
| Stage 5 | Step 5 — Scenes | N visual prompts + N video prompts | Yes — per-scene or all |

---

## 5. Mode 1 — Copy-Paste Specification

### 5.1 User flow per stage

```
User arrives at Stage 3
    ↓
UI shows: "Generate with AI — Copy & Paste mode"
    ↓
UI shows: [generated prompt in a read-only text area]
UI shows: [📋 Copy prompt] button
    ↓
User pastes into any chatbot → gets JSON response
    ↓
UI shows: [Paste AI response here] text area
UI shows: [Parse & Apply] button
    ↓
UI validates JSON, shows result (3 story cards / character / scene prompts)
    ↓
User selects / iterates / proceeds
```

### 5.2 Prompt generation

New module: `pipeline/prompt_builder.py`

```python
def build_story_prompt(idea, n_scenes, mood, style_dna) -> str
def build_character_prompt(idea, story, mood, style_dna) -> str
def build_scene_prompts_prompt(idea, story, character, style_dna, n_scenes) -> str
```

Each function returns a **complete, self-contained string** that:
- Includes all project context
- Specifies the response format precisely
- Works in any chatbot without prior context
- Is safe to copy as plain text (no special encoding needed)

### 5.3 Response parsing

New module: `pipeline/response_parser.py`

```python
def parse_story_response(pasted_text: str) -> list[dict] | None
def parse_character_response(pasted_text: str) -> str | None
def parse_scene_prompts_response(pasted_text: str) -> dict | None
    # returns {"visual_prompts": [...], "video_prompts": [...]}
```

Parser strategy:
1. Search for `{"stage":` in the pasted text
2. Find matching `}` — extract the JSON block
3. Validate required fields
4. Return parsed data or `None` on failure
5. Show clear error message if parsing fails (wrong format, truncated, etc.)

### 5.4 Iteration support

**Story (Stage 3):**
- User can paste the prompt again after asking the AI to "regenerate all"
- UI shows all 3 options; user selects one

**Character (Stage 4):**
- User can paste multiple times — each paste replaces the current description
- User can tell the chatbot: *"make her younger"* or *"she has a scar on her left eye"*
- User can share a character image with the chatbot and paste the improved description back
- The UI shows the current description as editable text — user can also just type directly

**Scene prompts (Stage 5):**
- User can regenerate specific scenes: UI shows per-scene "Regenerate this scene" buttons
- Clicking one generates a single-scene version of the prompt
- User can also paste a full-batch regeneration

### 5.5 Prompt design principles

Every prompt must follow this structure:

```
╔══════════════════════════════════════════════════════════╗
║  AI VIDEO PIPELINE — [STAGE NAME]                        ║
╚══════════════════════════════════════════════════════════╝

[Role / persona for the AI]

PROJECT CONTEXT:
[All relevant project data]

YOUR TASK:
[Specific task with exact output count]

RULES:
[Numbered list of constraints]

RESPOND WITH ONLY THIS JSON — no text before or after:
[exact JSON schema with comments]
```

The `RESPOND WITH ONLY THIS JSON` instruction ensures any chatbot (ChatGPT, Gemini,
Claude.ai, Claude Code) produces parseable output. The JSON schema with placeholder
values shows exactly what shape is expected.

---

## 6. Mode 2 — MCP Server Specification

### 6.1 Architecture

```
┌─────────────────────┐         ┌──────────────────────────┐
│   Streamlit UI      │         │   Claude Code session    │
│   (app.py)          │         │   (this conversation)    │
└────────┬────────────┘         └───────────┬──────────────┘
         │                                   │
         │  MCP Tools                        │  MCP Client
         │  (HTTP / stdio)                   │  (built into Claude Code CLI)
         ▼                                   ▼
┌─────────────────────────────────────────────────────────┐
│              pipeline/mcp_server.py                     │
│                                                         │
│  Tools exposed:                                         │
│    pipeline_get_pending_job()                           │
│    pipeline_submit_result(job_id, stage, result)        │
│    pipeline_get_project_context(project_name)           │
│    pipeline_list_pending_jobs()                         │
│    pipeline_job_status(job_id)                          │
└─────────────────────────────────────────────────────────┘
         │                                   
         │  Job storage: ai_jobs/ folder     
         │  (JSON files, human-readable)     
         ▼                                   
┌─────────────────────┐
│   ai_jobs/          │
│     pending/        │  ← UI writes here
│     done/           │  ← Claude Code writes here
│     failed/         │  ← errors land here
└─────────────────────┘
```

### 6.2 Job file format

**Pending job** (`ai_jobs/pending/{job_id}.json`):

```json
{
  "job_id": "uuid4",
  "stage": "story_options",
  "created_at": "2026-04-18T10:00:00",
  "project_name": "my_film",
  "payload": {
    "idea": "An Egyptian queen leads robot soldiers through a desert",
    "n_scenes": 6,
    "mood": "epic",
    "style_dna": { ... }
  },
  "prompt_for_human": "╔══ AI VIDEO PIPELINE ... ╗\n...(full copy-paste prompt)..."
}
```

The `prompt_for_human` field is the same prompt Mode 1 would show — so the MCP server and 
copy-paste mode share the same prompt builder. Mode 2 just automates the submission.

**Done result** (`ai_jobs/done/{job_id}.json`):

```json
{
  "job_id": "uuid4",
  "stage": "story_options",
  "completed_at": "2026-04-18T10:00:15",
  "result": [ ...3 story options... ]
}
```

### 6.3 MCP tools specification

```python
@mcp_tool
def pipeline_get_pending_job() -> dict | None:
    """
    Returns the oldest pending job as a dict, or null if none waiting.
    The dict includes the full prompt_for_human so Claude Code can
    process it without any additional context.
    """

@mcp_tool
def pipeline_submit_result(job_id: str, stage: str, result: any) -> bool:
    """
    Write the completed result for job_id to ai_jobs/done/.
    Returns True on success, False if job_id not found.
    """

@mcp_tool
def pipeline_get_project_context(project_name: str) -> dict:
    """
    Returns the full ProjectState for a named project.
    Used when Claude Code needs deeper context than the job payload contains.
    """

@mcp_tool
def pipeline_list_pending_jobs() -> list[dict]:
    """
    Returns all pending jobs with summary (job_id, stage, created_at).
    Used by Claude Code to decide processing order.
    """

@mcp_tool
def pipeline_job_status(job_id: str) -> str:
    """
    Returns: "pending" | "processing" | "done" | "failed" | "not_found"
    """
```

### 6.4 Claude Code activation

User types once per session:

```
start pipeline worker
```

Claude Code enters a monitoring loop:
1. Call `pipeline_list_pending_jobs()`
2. For each pending job: call `pipeline_get_pending_job()`, process, call `pipeline_submit_result()`
3. Wait 10 seconds
4. Repeat

Processing a job means: Claude Code reads the `prompt_for_human` field, generates the
content using its own intelligence (same quality as Claude.ai, no API key needed),
formats the result as the required JSON, and submits it.

### 6.5 MCP server configuration

**`.mcp.json`** (project root — already the config file Claude Code reads):

```json
{
  "mcpServers": {
    "pipeline": {
      "command": "venv/Scripts/python.exe",
      "args": ["pipeline/mcp_server.py"],
      "description": "AI Video Pipeline job queue"
    }
  }
}
```

This means the pipeline MCP server starts automatically when Claude Code starts,
and the tools are available immediately without any setup.

### 6.6 UI polling for results

When Mode 2 is active and a job is submitted, the UI:
1. Shows a spinner: *"Waiting for Claude Code to process…"*
2. Polls `ai_jobs/done/{job_id}.json` every 3 seconds
3. When result file appears: parses and applies it
4. Shows a success toast: *"Generated by Claude Code ✅"*
5. If no result after 120 seconds: falls back to showing the copy-paste prompt

---

## 7. UI Changes Required

### 7.1 New session state variables

```python
_ss("ai_mode", "auto")              # "mcp" | "copy_paste" | "api_key" | "mechanical"
_ss("copy_paste_prompt", None)      # str | None — current generated prompt
_ss("copy_paste_stage", None)       # "story" | "character" | "scenes"
_ss("copy_paste_pending_job", None) # job_id waiting for result
_ss("mcp_worker_active", False)     # bool — is Claude Code watching?
```

### 7.2 Mode selector widget

New helper function `render_mode_selector(stage_name) -> str`:
- Returns selected mode: `"mcp"` | `"copy_paste"` | `"mechanical"` | `"api_key"`
- Shows a compact horizontal radio group at the top of each AI step
- Remembers selection in session state

### 7.3 Step 3 changes (Story)

Replace the existing `with st.spinner("Generating story treatments…")` block with:

```
if mode == "mcp":         → submit job, poll for result, show cards
if mode == "copy_paste":  → show prompt area + paste area + parse button + cards
if mode == "api_key":     → existing Claude API call
if mode == "mechanical":  → existing _offline_options() call
```

### 7.4 Step 4 changes (Character)

Same pattern. Additional UI for copy-paste mode:
- "You can share a character image with the chatbot before pasting the prompt"
- Iteration hint: "Paste the response, then update the prompt to refine further"
- Previous description shown for comparison when a new paste arrives

### 7.5 Step 5 changes (Scenes)

Same pattern. Additional feature for copy-paste mode:
- Per-scene "Regenerate" button generates a single-scene prompt
- Full-batch "Regenerate all" generates one prompt for all N scenes
- Individual scene prompts can be updated without regenerating the whole set

### 7.6 Sidebar additions

New section in sidebar:

```
── AI Mode ──────────────────
  Current: 📋 Copy & Paste
  
  [ 🤖 Start MCP Worker ]   ← activates Mode 2
  or "MCP Worker active ✅"
  
  Mode priority:
  1. 🤖 MCP (inactive)
  2. 📋 Copy & Paste  ← current
  3. ⚙️  Mechanical
  4. 🔑 API Key (not set)
```

---

## 8. File Structure Changes

```
comfyui-video-mcp/
│
├── pipeline/
│   ├── prompt_builder.py      ← NEW: builds copy-paste prompts for all stages
│   ├── response_parser.py     ← NEW: parses chatbot JSON responses
│   ├── mcp_server.py          ← NEW (Phase 2): MCP server + job queue tools
│   └── ai_bridge.py           ← NEW (Phase 2): job file read/write helpers
│
├── ai_jobs/                   ← NEW (Phase 2): job queue directory
│   ├── pending/
│   ├── done/
│   └── failed/
│
├── .mcp.json                  ← NEW (Phase 2): MCP server config for Claude Code
│
├── tests/
│   ├── test_prompt_builder.py ← NEW: tests for all prompt templates
│   └── test_response_parser.py← NEW: tests for JSON parsing + edge cases
│
└── app.py                     ← MODIFIED: mode selector, copy-paste UI per step
```

---

## 9. Prompt Templates — Exact Format

### 9.1 Stage 3 — Story Options

```
╔══════════════════════════════════════════════════════════════════════╗
║  AI VIDEO PIPELINE — STAGE 3: STORY OPTIONS                          ║
╚══════════════════════════════════════════════════════════════════════╝

You are a professional screenwriter and video director specialising in
short-form cinematic storytelling.

PROJECT CONTEXT:
  Idea:     "{idea}"
  Duration: {duration}s ({n_scenes} scenes × 5 seconds each)
  Style:    {style_name}
  Mood:     {mood or "not specified"}

STYLE NOTES:
  Visual: {style_dna.visual_style}
  Motion: {style_dna.motion_style}
  Palette: {", ".join(style_dna.color_palette[:3])}

YOUR TASK:
Generate exactly 3 distinct story treatment options for this video.
Each treatment must suggest exactly {n_scenes} scenes.

RULES:
1. Each treatment must have a genuinely different narrative structure
2. Act labels must be UPPERCASE (e.g. HOOK, BUILD, CLIMAX, RESOLUTION)
3. Scene descriptions must be one sentence each — visual and specific
4. Summary must be exactly 2 sentences
5. Arc must be exactly 5 emotional beats separated by →
6. Reasoning must be one sentence explaining why this structure fits the idea
7. Do not add any explanation or text outside the JSON

RESPOND WITH ONLY THIS JSON — no text before or after the JSON block:

{
  "stage": "story_options",
  "result": [
    {
      "title": "short 2-4 word title",
      "summary": "sentence one. sentence two.",
      "arc": "beat1 → beat2 → beat3 → beat4 → beat5",
      "pacing": "one sentence about rhythm and tension",
      "reasoning": "one sentence on why this structure fits",
      "act_labels": ["HOOK", "BUILD", "...", "RESOLUTION"],
      "scene_descriptions": ["scene 1 description", "...", "scene N description"]
    },
    { ...option 2... },
    { ...option 3... }
  ]
}
```

### 9.2 Stage 4 — Character Description

```
╔══════════════════════════════════════════════════════════════════════╗
║  AI VIDEO PIPELINE — STAGE 4: CHARACTER DESCRIPTION                  ║
╚══════════════════════════════════════════════════════════════════════╝

You are a professional casting director and visual development artist.
Your descriptions are used directly as image generation prompts — they
must be purely visual, hyper-specific, and camera-ready.

PROJECT CONTEXT:
  Idea:     "{idea}"
  Style:    {style_name}
  Mood:     {mood or "not specified"}

SELECTED STORY:
  Title:    "{story.title}"
  Summary:  "{story.summary}"
  Arc:      "{story.arc}"

{IF character image provided:}
CHARACTER REFERENCE IMAGE:
  [image attached]
  Use this image as the primary visual reference. The description must
  match what you see in the image as closely as possible.

YOUR TASK:
Write a vivid 2-3 sentence visual description of the protagonist.

RULES:
1. Describe ONLY what a camera would see — no backstory, no emotions, no personality
2. Include: approximate age, build, distinctive facial features, hair, clothing
3. Include one distinctive detail that will make this character recognisable across all scenes
4. Be specific enough that two different image models would produce similar-looking results
5. Do not use vague words like "beautiful", "stunning", "interesting"
6. Do not add any text outside the JSON

RESPOND WITH ONLY THIS JSON — no text before or after:

{
  "stage": "character_description",
  "result": "full character description as a single string"
}
```

### 9.3 Stage 5 — Scene Prompts (Visual + Video)

```
╔══════════════════════════════════════════════════════════════════════╗
║  AI VIDEO PIPELINE — STAGE 5: SCENE PROMPTS                          ║
╚══════════════════════════════════════════════════════════════════════╝

You are working as two experts simultaneously:
  VISUAL EXPERT: A professional cinematographer writing ComfyUI image prompts
  MOTION EXPERT: An I2V director writing motion-only video prompts

{skill.prompt_template}

PROJECT CONTEXT:
  Idea:       "{idea}"
  Style:      {style_name}
  Character:  "{character.description}"
  Motion style for this project: "{style_dna.motion_style}"

SCENES TO PROMPT ({n_scenes} total):
{for each scene:
  Scene N | Act: {act}
  Description: "{description}"
  Camera: "{camera}"
  Lighting: "{lighting}"
}

YOUR TASK — TWO OUTPUTS REQUIRED:

VISUAL PROMPTS ({n_scenes} prompts):
  - Start with the scene action/setting — NOT the character description
  - Include the exact camera move and lighting as specified
  - Weave the character description in after the scene content
  - Append quality boosters: {", ".join(skill.quality_boosters[:6])}
  - Each prompt under 150 words, single line, no line breaks
  - Every prompt must start differently — no two prompts with the same first 10 words

VIDEO PROMPTS ({n_scenes} prompts):
  - Describe ONLY motion and action — never appearance
  - Include the camera move with precise speed/direction
  - Include environmental motion (wind, fabric, water, crowd)
  - Maximum 60 words each
  - End with a pacing word: [slow] [medium] [fast] [explosive]

RESPOND WITH ONLY THIS JSON — no text before or after:

{
  "stage": "scene_prompts",
  "result": {
    "visual_prompts": [
      "scene 1 visual prompt",
      "...",
      "scene N visual prompt"
    ],
    "video_prompts": [
      "scene 1 motion prompt",
      "...",
      "scene N motion prompt"
    ]
  }
}
```

---

## 10. Sprint Plan

### Sprint 5A — Mode 1: Copy-Paste (Permanent)

**Goal:** Any user without an API key can get full AI quality by copy-pasting prompts.

**Deliverables:**

#### 5A.1 — `pipeline/prompt_builder.py`
- `build_story_prompt(idea, n_scenes, mood, style_dna) → str`
- `build_character_prompt(idea, story, mood, style_dna, has_image_hint=False) → str`
- `build_scene_prompts_prompt(idea, story, character, style_dna, scenes) → str`
- `build_single_scene_prompt(idea, scene, character, style_dna) → str` (for per-scene regen)
- All prompts follow the exact templates in Section 9

#### 5A.2 — `pipeline/response_parser.py`
- `parse_story_response(text) → list[dict] | None`
- `parse_character_response(text) → str | None`
- `parse_scene_prompts_response(text) → dict | None`
- `ParseError` exception with human-readable message
- Each parser: find JSON block → validate schema → return typed result

#### 5A.3 — Mode selector widget
- `render_mode_selector(key_suffix) → str` in `app.py`
- Stores selection in `st.session_state.ai_mode`
- Shows MCP as greyed-out with tooltip until Phase 2

#### 5A.4 — Step 3 UI update
- Mode selector at top
- Copy-paste path: show prompt → copy button → paste area → parse button → story cards
- Iteration: "Ask AI to regenerate all" re-shows the same prompt with a hint appended
- All existing paths (mechanical, API key) unchanged

#### 5A.5 — Step 4 UI update
- Mode selector at top
- Copy-paste path: show prompt → copy → paste → parse → description textarea
- Image hint: info box *"You can share a character image with the chatbot before pasting"*
- Iteration: each paste replaces the description; previous shown for comparison

#### 5A.6 — Step 5 UI update
- Mode selector at top
- Copy-paste path: full batch prompt → copy → paste → parse → all scenes updated
- Per-scene regeneration: individual prompt per scene
- Preserves manually edited prompts — only replaces scenes included in the paste

#### 5A.7 — Tests: `tests/test_prompt_builder.py`
- Prompt contains all context fields (idea, style, mood, etc.)
- Prompt contains response format instructions
- Prompt contains `"stage": "story_options"` schema marker
- Prompt works for all 13 skills
- Single-scene prompt contains only that scene's data

#### 5A.8 — Tests: `tests/test_response_parser.py`
- Valid JSON parses correctly for all three stages
- JSON embedded in chatbot preamble text still parses
- Wrong `stage` field returns None
- Missing required fields returns None
- Truncated JSON returns None with clear error
- Empty string returns None
- Test with real example responses from Claude, ChatGPT format

**Sprint 5A complete when:**
- User with no API key can reach Step 3, copy a prompt, paste a response, see 3 story cards
- Same for Steps 4 and 5
- All prompts work verbatim in Claude.ai, ChatGPT, and Gemini (manually verified)

---

### Sprint 5B — Mode 2: MCP Auto (Phase 2)

**Goal:** When Claude Code is active, jobs process automatically with no user copy-pasting.

**Deliverables:**

#### 5B.1 — `pipeline/ai_bridge.py`
- `write_pending_job(stage, payload, prompt_for_human) → str` (returns job_id)
- `read_result(job_id, timeout_seconds=120) → dict | None`
- `list_pending_jobs() → list[dict]`
- `mark_job_failed(job_id, error) → None`
- Job storage: `ai_jobs/pending/`, `ai_jobs/done/`, `ai_jobs/failed/`
- Job format: Section 6.2

#### 5B.2 — `pipeline/mcp_server.py`
- MCP server using `mcp` Python library
- Five tools: Section 6.3
- Handles concurrent access (UI writing + Claude Code reading)
- Cleans up done jobs older than 24 hours

#### 5B.3 — `.mcp.json`
- Registers the pipeline MCP server
- Points to `venv/Scripts/python.exe pipeline/mcp_server.py`

#### 5B.4 — UI polling in Steps 3, 4, 5
- When mode is "mcp": write job → show spinner → poll every 3s for result
- Timeout after 120s → fall back to copy-paste mode with a message
- Show `"Generated by Claude Code ✅"` toast on success

#### 5B.5 — Sidebar MCP status widget
- Shows: active / inactive
- "Start worker" button (instructions for user)
- Job queue stats: N pending, N completed today

#### 5B.6 — Tests: `tests/test_mcp_server.py`
- `write_pending_job` creates file in correct location
- `read_result` returns None when file absent
- `read_result` returns parsed dict when file present
- `pipeline_get_pending_job` MCP tool returns oldest pending job
- `pipeline_submit_result` writes done file and removes pending file
- Concurrent write + read does not corrupt job file

**Sprint 5B complete when:**
- User types "start pipeline worker" in Claude Code
- User clicks through Steps 3-5 without touching copy-paste
- All AI generation completes automatically within 30 seconds per stage

---

## 11. Milestone Tracker

| Sprint | Milestone | Status | Completed |
|---|---|---|---|
| 5A | `pipeline/prompt_builder.py` — all 4 prompt functions | ⬜ Pending | — |
| 5A | `pipeline/response_parser.py` — all 3 parsers + error handling | ⬜ Pending | — |
| 5A | Mode selector widget in app.py | ⬜ Pending | — |
| 5A | Step 3 UI — copy-paste path fully working | ⬜ Pending | — |
| 5A | Step 4 UI — copy-paste path + iteration + image hint | ⬜ Pending | — |
| 5A | Step 5 UI — copy-paste path + per-scene regen | ⬜ Pending | — |
| 5A | `tests/test_prompt_builder.py` passes | ⬜ Pending | — |
| 5A | `tests/test_response_parser.py` passes | ⬜ Pending | — |
| 5A | Manual verify: prompts work in Claude.ai + ChatGPT | ⬜ Pending | — |
| 5B | `pipeline/ai_bridge.py` — job file read/write | ⬜ Pending | — |
| 5B | `pipeline/mcp_server.py` — MCP server + 5 tools | ⬜ Pending | — |
| 5B | `.mcp.json` — Claude Code auto-connects to pipeline | ⬜ Pending | — |
| 5B | UI polling in Steps 3, 4, 5 for MCP results | ⬜ Pending | — |
| 5B | Sidebar MCP status widget | ⬜ Pending | — |
| 5B | `tests/test_mcp_server.py` passes | ⬜ Pending | — |
| 5B | End-to-end: Claude Code worker processes all 3 stages auto | ⬜ Pending | — |

**Status key:** ⬜ Pending | 🔄 In Progress | ✅ Done | ❌ Blocked

---

## Appendix A — Mode Detection Flowchart

```
User reaches Step 3 / 4 / 5
          │
          ▼
   MCP worker active?
   (mcp_worker_active in session state)
      │         │
     YES        NO
      │          │
      ▼          ▼
  Submit     Show mode selector:
  job to     [ Copy & Paste ]
  MCP.       [ Mechanical   ]
  Poll for   [ API Key      ]  ← only if ANTHROPIC_API_KEY set
  result.         │
                  ▼
           User picks mode
                  │
        ┌─────────┼─────────┐
        ▼         ▼         ▼
   Copy/Paste  Mechanical  API Key
   Show prompt  Run        Run Claude
   + paste box  templates  API call
```

## Appendix B — JSON Response Examples

### Story options response
```json
{
  "stage": "story_options",
  "result": [
    {
      "title": "The Discovery",
      "summary": "A robot encounters human art for the first time. The experience rewires everything it thought it understood about the world.",
      "arc": "isolation → curiosity → wonder → confusion → transformation",
      "pacing": "Slow observational open, building to an overwhelmed climax, resolving in quiet stillness.",
      "reasoning": "Discovery arcs let the audience experience revelation alongside the subject — ideal for a first-contact premise.",
      "act_labels": ["HOOK", "ENCOUNTER", "WONDER", "OVERWHELM", "RESOLUTION"],
      "scene_descriptions": [
        "A robot moves through an empty white corridor, mechanical and purposeful, alone.",
        "It stops at an open gallery doorway, one foot raised mid-step, circuits processing.",
        "The robot stands before a large oil painting — its camera eye dilates, scanning every brushstroke.",
        "It reaches toward the canvas, then freezes — something in its processing loop has changed.",
        "The robot sits cross-legged on the gallery floor, surrounded by art, perfectly still."
      ]
    }
  ]
}
```

### Character response
```json
{
  "stage": "character_description",
  "result": "A lean humanoid robot standing 1.8 metres tall, chassis brushed titanium-grey with amber oxidation at the joints. Its single camera eye is large and amber-lit, mounted in a smooth ovoid head with no mouth. The torso carries a faded manufacturer's serial number stencilled in black across the chest plate."
}
```

### Scene prompts response
```json
{
  "stage": "scene_prompts",
  "result": {
    "visual_prompts": [
      "Empty white gallery corridor, harsh overhead 5500K lighting casting long shadows, robot stands mid-step in a doorway, lean titanium-grey chassis with amber joints, single amber camera eye scanning forward, 3 ft/s dolly forward tracking subject, masterpiece, best quality, 8k uhd, cinematic",
      "..."
    ],
    "video_prompts": [
      "Robot advances three paces, stops abruptly at doorway. Camera dollies in 2 ft over 4 seconds. Ambient air conditioning causes slight vibration in ceiling fixtures. [slow]",
      "..."
    ]
  }
}
```

---

*This document covers Sprint 5A (Mode 1) and Sprint 5B (Mode 2). Both modes are permanent features — Mode 1 is never deprecated when Mode 2 is added. Mode 1 always provides the copy-paste path for iteration, image sharing, and offline use.*
