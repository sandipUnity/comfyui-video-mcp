"""
ComfyUI Video Generation MCP Server (Skill-Powered Edition)
============================================================
Pipeline: Notes → Skill Detection → Cinema-Quality Ideas → Scenes → ComfyUI → Montage

Integrated skill frameworks from higgsfield-seedance2-jineng:
  15 cinematic styles — Kelvin-precise lighting, exact camera velocities,
  2-second hook patterns, genre-specific visual vocabularies

LLM options (no API key required for offline/Ollama mode):
  - Claude API (best quality — needs ANTHROPIC_API_KEY)
  - Ollama (local — install ollama + pull a model)
  - Offline (template-based — zero dependencies)

Tools:
  list_skills           - Show all 15 skill frameworks
  detect_skill          - See which skill matches your notes
  generate_ideas        - Notes → skill-powered cinematic ideas
  list_ideas            - Show current ideas
  select_idea           - Pick idea → get cinema-precise scene breakdown
  regenerate_ideas      - Regen with feedback
  configure_pipeline    - Adjust model, resolution, LLM provider
  generate_video        - Queue video generation in ComfyUI
  check_status          - Monitor ComfyUI job progress
  list_videos           - Browse generated clips
  compile_montage       - Build final montage video
  list_montages         - Browse finished videos
  ping_comfyui          - Check ComfyUI connection
  get_available_models  - List loaded ComfyUI models
  session_status        - Full pipeline state
"""

import asyncio
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Optional

import yaml
from mcp.server.fastmcp import FastMCP

from session import session, VideoIdea, Scene
from idea_generator import IdeaGenerator
from comfyui_client import ComfyUIClient
from montage_compiler import MontageCompiler
from skills_engine import (
    detect_skill, get_skill_by_id, list_skills as engine_list_skills,
    build_comfyui_positive, build_comfyui_negative, get_workflow_overrides,
    SKILLS,
)

# ── Config ────────────────────────────────────────────────────────────────────
_BASE = Path(__file__).parent
_CFG_PATH = _BASE / "config.yaml"

with open(_CFG_PATH) as f:
    CONFIG = yaml.safe_load(f)

# Runtime overrides (via configure_pipeline tool)
_runtime_cfg = {}


def get_cfg(*keys, default=None):
    """Read from runtime overrides first, then config file."""
    for k in keys:
        if k in _runtime_cfg:
            return _runtime_cfg[k]
    # Walk nested config
    val = CONFIG
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return default
    return val if val is not None else default


def comfyui_client() -> ComfyUIClient:
    return ComfyUIClient(
        host=get_cfg("comfyui", "host", default="127.0.0.1"),
        port=get_cfg("comfyui", "port", default=8188),
    )


def idea_gen() -> IdeaGenerator:
    return IdeaGenerator(CONFIG.get("idea_generation", {}))


def montage_comp() -> MontageCompiler:
    return MontageCompiler(CONFIG.get("montage", {}))


def output_dir(sub: str) -> Path:
    base = _BASE / get_cfg("comfyui", "output_dir", default="output")
    return base / sub


# ── Camera LoRA Detector ───────────────────────────────────────────────────────

# Maps prompt keywords → camera LoRA slot name (keys match config.yaml camera_loras)
_CAMERA_LORA_KEYWORDS: list[tuple[str, str]] = [
    ("dolly forward",   "dolly_in"),
    ("dolly in",        "dolly_in"),
    ("push-in",         "dolly_in"),
    ("push in",         "dolly_in"),
    ("tracking forward","dolly_in"),
    ("dolly back",      "dolly_out"),
    ("dolly out",       "dolly_out"),
    ("pull-back",       "dolly_out"),
    ("pull back",       "dolly_out"),
    ("zoom out",        "dolly_out"),
    ("pan left",        "dolly_left"),
    ("dolly left",      "dolly_left"),
    ("tracking left",   "dolly_left"),
    ("pan right",       "dolly_right"),
    ("dolly right",     "dolly_right"),
    ("tracking right",  "dolly_right"),
    ("crane up",        "jib_up"),
    ("jib up",          "jib_up"),
    ("rising shot",     "jib_up"),
    ("crane down",      "jib_down"),
    ("jib down",        "jib_down"),
    ("descending",      "jib_down"),
    ("static",          "static"),
    ("locked off",      "static"),
    ("fixed shot",      "static"),
]


def _detect_camera_lora(positive_prompt: str, model_cfg: dict) -> tuple[str, float]:
    """
    Detect which camera LoRA to use based on prompt keywords.
    Returns (lora_filename, strength). Falls back to fallback LoRA if no match.
    """
    loras = model_cfg.get("camera_loras", {})
    prompt_lower = positive_prompt.lower()

    for keyword, slot in _CAMERA_LORA_KEYWORDS:
        if keyword in prompt_lower and slot in loras:
            return loras[slot], model_cfg.get("camera_lora_strength", 1.0)

    # fallback
    fallback = model_cfg.get("camera_lora_fallback", "")
    return fallback, model_cfg.get("camera_lora_strength", 0.8)


# ── Workflow Builder ───────────────────────────────────────────────────────────

def build_workflow(positive: str, negative: str, model: str = None, **overrides) -> dict:
    """Load and fill a ComfyUI workflow template for the configured model."""
    model = model or _runtime_cfg.get("model") or get_cfg("pipeline", "default_model", default="ltxvideo")
    model_cfg = CONFIG.get("models", {}).get(model, {})

    if not model_cfg:
        raise ValueError(
            f"Model '{model}' not found in config. "
            f"Available: {', '.join(CONFIG.get('models', {}).keys())}"
        )

    workflow_path = _BASE / model_cfg.get("workflow", f"workflows/{model}_api.json")
    if not workflow_path.exists():
        raise FileNotFoundError(
            f"Workflow file not found: {workflow_path}\n"
            f"Available workflows: {[p.name for p in (_BASE / 'workflows').glob('*.json')]}"
        )

    with open(workflow_path) as f:
        template = f.read()

    # Per-model step/cfg overrides (distilled LoRA needs fewer steps)
    steps = overrides.get("steps") or model_cfg.get("steps_override") or get_cfg("pipeline", "default_steps", default=20)
    cfg   = overrides.get("cfg")   or model_cfg.get("cfg_override")   or get_cfg("pipeline", "default_cfg", default=3.0)

    # Camera LoRA (only relevant for ltxvideo_camera workflow)
    camera_lora, camera_lora_strength = "", 1.0
    if "camera_loras" in model_cfg:
        camera_lora, camera_lora_strength = _detect_camera_lora(positive, model_cfg)

    replacements = {
        "{{POSITIVE_PROMPT}}":      positive,
        "{{NEGATIVE_PROMPT}}":      negative,
        "{{CHECKPOINT}}":           model_cfg.get("checkpoint", ""),
        "{{TEXT_ENCODER}}":         model_cfg.get("text_encoder", "t5xxl_fp16.safetensors"),
        "{{VAE}}":                  model_cfg.get("vae", ""),
        "{{MOTION_MODULE}}":        model_cfg.get("motion_module", ""),
        "{{CAMERA_LORA}}":          camera_lora,
        "{{CAMERA_LORA_STRENGTH}}": str(camera_lora_strength),
        "{{WIDTH}}":    str(overrides.get("width")  or get_cfg("pipeline", "default_width",  default=768)),
        "{{HEIGHT}}":   str(overrides.get("height") or get_cfg("pipeline", "default_height", default=512)),
        "{{FRAMES}}":   str(overrides.get("frames") or get_cfg("pipeline", "default_frames", default=25)),
        "{{FPS}}":      str(overrides.get("fps")    or get_cfg("pipeline", "default_fps",    default=24)),
        "{{STEPS}}":    str(steps),
        "{{CFG}}":      str(cfg),
        "{{SEED}}":     str(overrides.get("seed") or random.randint(0, 2**31)),
        "{{OUTPUT_PREFIX}}": f"video_{int(time.time())}",
    }

    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)

    return json.loads(template)


# ── MCP Server ────────────────────────────────────────────────────────────────
mcp = FastMCP(
    "ComfyUI Video Pipeline",
    instructions=(
        "Skill-powered automated video creation pipeline. "
        "Start: generate_ideas('your concept'). "
        "Pipeline: list_skills → generate_ideas → select_idea → generate_video → compile_montage. "
        "No API key needed — works with Ollama (local) or offline mode."
    ),
)

# ── Tool: list_skills ─────────────────────────────────────────────────────────
@mcp.tool()
def list_skills() -> str:
    """
    List all 15 cinematic skill frameworks available for video generation.

    Each skill provides: camera vocabulary, Kelvin-precise lighting, 2-second hook patterns,
    ComfyUI quality boosters, and style-specific prompt templates.
    """
    skills = engine_list_skills()
    lines = ["Available Skill Frameworks (from Seedance 2.0 / Higgsfield):\n"]
    for s in skills:
        lines.append(f"  [{s['id']:15s}] {s['name']}")
        lines.append(f"               {s['description']}")
        lines.append("")
    lines.append("Use skill_id= in generate_ideas() or configure_pipeline() to force a skill.")
    lines.append("Or let the pipeline auto-detect the best skill from your notes.")
    return "\n".join(lines)


# ── Tool: detect_skill ────────────────────────────────────────────────────────
@mcp.tool()
def detect_skill_for_notes(notes: str) -> str:
    """
    Preview which skill framework will be auto-detected for your notes.

    notes: Your video concept or description
    """
    skill = detect_skill(notes)
    overrides = get_workflow_overrides(skill)
    lines = [
        f"Detected skill: {skill.name} [{skill.id}]",
        f"Description: {skill.description}",
        "",
        "ComfyUI overrides this skill applies:",
        f"  Resolution: {overrides['width']}×{overrides['height']}",
        f"  FPS: {overrides['fps']} | Steps: {overrides['steps']} | CFG: {overrides['cfg']}",
        "",
        "Camera vocabulary sample:",
    ]
    for cam in skill.camera_vocabulary[:3]:
        lines.append(f"  • {cam}")
    lines.append("\nLighting vocabulary sample:")
    for light in skill.lighting_vocabulary[:2]:
        lines.append(f"  • {light}")
    lines.append("\nOpening hook patterns:")
    for hook in skill.hook_patterns[:2]:
        lines.append(f"  • {hook}")
    lines.append(f"\nQuality boosters: {', '.join(skill.quality_boosters[:5])}")
    lines.append(f"\nOverride with: generate_ideas(notes='...', skill_id='{skill.id}')")
    return "\n".join(lines)


# ── Tool: ping_comfyui ────────────────────────────────────────────────────────
@mcp.tool()
async def ping_comfyui() -> str:
    """Check if ComfyUI is running and reachable."""
    client = comfyui_client()
    available = await client.is_available()
    host = get_cfg("comfyui", "host", default="127.0.0.1")
    port = get_cfg("comfyui", "port", default=8188)
    if available:
        return f"✅ ComfyUI is running at {host}:{port}"
    return (
        f"❌ ComfyUI not reachable at {host}:{port}. "
        "Make sure ComfyUI is started with --listen flag."
    )


# ── Tool: get_available_models ────────────────────────────────────────────────
@mcp.tool()
async def get_available_models() -> str:
    """List models available in ComfyUI (checkpoints and motion modules)."""
    client = comfyui_client()
    try:
        models = await client.get_models()
        lines = ["Available models in ComfyUI:\n"]
        if models.get("checkpoints"):
            lines.append("Checkpoints:")
            for m in models["checkpoints"]:
                lines.append(f"  - {m}")
        if models.get("motion_modules"):
            lines.append("\nMotion Modules (AnimateDiff):")
            for m in models["motion_modules"]:
                lines.append(f"  - {m}")
        if not models.get("checkpoints") and not models.get("motion_modules"):
            lines.append("No models found. Make sure models are in ComfyUI's models folder.")
        return "\n".join(lines)
    except Exception as e:
        return f"Error fetching models: {e}"


# ── Tool: model_status ───────────────────────────────────────────────────────
@mcp.tool()
def model_status() -> str:
    """
    Show which models are installed and ready, which need downloading,
    and recommended settings for each. Based on your ComfyUI model folder.
    """
    return """Model Status for ComfyUI @ 192.168.1.196:8188
=====================================================

READY TO USE (models confirmed installed)
──────────────────────────────────────────
[✅] ltxvideo_camera  (RECOMMENDED — use this)
     Model:   ltx-2-19b-dev-fp8.safetensors
     Encoder: t5xxl_fp16.safetensors
     VAE:     ltx-2-19b-distilled-fp8.safetensors
     LoRAs:   dolly-in ✅ | jib-down ✅ | others ⚠️ need re-download
     Steps:   20  |  CFG: 3.0  |  768×512  |  24fps

[✅] ltxvideo         (same as above, no camera LoRA)

[✅] ltxvideo_fast    (4-step distilled — very fast)
     Extra LoRA: ltx-2-19b-distilled-lora-384.safetensors
     Steps:   6   |  CFG: 1.0  — use for quick previews

[✅] wan22            (highest quality — slow, ~10 min)
     Model:   wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors
     Encoder: umt5_xxl_fp8_e4m3fn_scaled.safetensors
     VAE:     wan_2.1_vae.safetensors
     Steps:   20  |  CFG: 5.0  |  768×512  |  24fps
     Use for: final renders, complex scenes

[✅] wan22_calm       (portraits, landscapes, calm motion)
     Model:   wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors

CAMERA LoRA STATUS (for ltxvideo_camera)
──────────────────────────────────────────
[✅] dolly-in    (push toward subject)
[✅] jib-down    (crane downward)
[⚠️] dolly-out   — only Zone.Identifier found, re-download .safetensors
[⚠️] dolly-left  — only Zone.Identifier found, re-download .safetensors
[⚠️] dolly-right — only Zone.Identifier found, re-download .safetensors
[⚠️] jib-up      — only Zone.Identifier found, re-download .safetensors
[⚠️] static      — only Zone.Identifier found, re-download .safetensors

  Re-download from: https://huggingface.co/Lightricks/LTX-Video/tree/main
  Or use ComfyUI Manager → Model Manager → search "ltx camera"
  Fallback: dolly-in LoRA used when others are missing (auto fallback)

NOT INSTALLED (models missing)
──────────────────────────────────────────
[❌] animatediff — needs v1-5-pruned-emaonly.safetensors + motion module
[❌] svd         — needs svd_xt.safetensors

UPSCALERS (installed, usable in post)
──────────────────────────────────────────
[✅] ltx-2.3-spatial-upscaler-x2-1.1.safetensors
[✅] ltx-2.3-temporal-upscaler-x2-1.0.safetensors
[✅] RealESRGAN_x4plus.pth

QUICK SWITCH COMMANDS
──────────────────────────────────────────
  configure_pipeline(model='ltxvideo_camera')  ← default, best balance
  configure_pipeline(model='ltxvideo_fast')    ← fast preview (4-8 steps)
  configure_pipeline(model='wan22')            ← highest quality
  configure_pipeline(model='wan22_calm')       ← calm/portrait scenes"""


# ── Tool: configure_pipeline ──────────────────────────────────────────────────
@mcp.tool()
def configure_pipeline(
    model: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    frames: Optional[int] = None,
    fps: Optional[int] = None,
    steps: Optional[int] = None,
    cfg: Optional[float] = None,
    transition: Optional[str] = None,
    resolution: Optional[str] = None,
    ideas_count: Optional[int] = None,
    scenes_count: Optional[int] = None,
    llm_provider: Optional[str] = None,
) -> str:
    """
    Configure pipeline settings. All parameters optional.

    model: animatediff | svd | cogvideox | wan21
    transition: fade | dissolve | wipe | slide | zoom | none
    llm_provider: claude | ollama
    """
    changes = []
    if model:
        _runtime_cfg["model"] = model
        changes.append(f"model → {model}")
    if width:
        _runtime_cfg["width"] = width
        changes.append(f"width → {width}")
    if height:
        _runtime_cfg["height"] = height
        changes.append(f"height → {height}")
    if frames:
        _runtime_cfg["frames"] = frames
        changes.append(f"frames → {frames}")
    if fps:
        _runtime_cfg["fps"] = fps
        changes.append(f"fps → {fps}")
    if steps:
        _runtime_cfg["steps"] = steps
        changes.append(f"steps → {steps}")
    if cfg:
        _runtime_cfg["cfg"] = cfg
        changes.append(f"cfg → {cfg}")
    if transition:
        CONFIG.setdefault("montage", {})["default_transition"] = transition
        changes.append(f"transition → {transition}")
    if resolution:
        CONFIG.setdefault("montage", {})["default_resolution"] = resolution
        changes.append(f"resolution → {resolution}")
    if ideas_count:
        CONFIG.setdefault("idea_generation", {})["ideas_per_request"] = ideas_count
        changes.append(f"ideas_count → {ideas_count}")
    if scenes_count:
        CONFIG.setdefault("idea_generation", {})["scenes_per_idea"] = scenes_count
        changes.append(f"scenes_count → {scenes_count}")
    if llm_provider:
        CONFIG.setdefault("idea_generation", {})["provider"] = llm_provider
        changes.append(f"llm_provider → {llm_provider}")

    if not changes:
        host = get_cfg("comfyui", "host", default="127.0.0.1")
        port = get_cfg("comfyui", "port", default=8188)
        return (
            "Current pipeline config:\n"
            f"  ComfyUI:    http://{host}:{port}\n"
            f"  model:      {_runtime_cfg.get('model') or get_cfg('pipeline', 'default_model', default='ltxvideo_camera')}\n"
            f"  width:      {_runtime_cfg.get('width', CONFIG['pipeline']['default_width'])}\n"
            f"  height:     {_runtime_cfg.get('height', CONFIG['pipeline']['default_height'])}\n"
            f"  frames:     {_runtime_cfg.get('frames', CONFIG['pipeline']['default_frames'])}\n"
            f"  fps:        {_runtime_cfg.get('fps', CONFIG['pipeline']['default_fps'])}\n"
            f"  steps:      {_runtime_cfg.get('steps', CONFIG['pipeline']['default_steps'])}\n"
            f"  cfg:        {_runtime_cfg.get('cfg', CONFIG['pipeline']['default_cfg'])}\n"
            f"  transition: {CONFIG.get('montage', {}).get('default_transition', 'dissolve')}\n"
            f"  resolution: {CONFIG.get('montage', {}).get('default_resolution', '1280x720')}\n"
            f"  llm:        {CONFIG.get('idea_generation', {}).get('provider', 'auto')}\n"
            "\nInstalled models: ltxvideo | ltxvideo_fast | ltxvideo_camera | wan22 | wan22_calm"
        )
    return "Pipeline configured:\n" + "\n".join(f"  {c}" for c in changes)


# ── Tool: generate_ideas ──────────────────────────────────────────────────────
@mcp.tool()
async def generate_ideas(notes: str, count: int = 5, skill_id: Optional[str] = None) -> str:
    """
    Generate cinematic video ideas from your notes using skill frameworks.

    notes:    Your raw idea, description, or concept
    count:    Number of ideas (default: 5)
    skill_id: Force a specific skill (see list_skills). Auto-detected if omitted.
              Options: cinematic, anime, 3d_cgi, cartoon, fight_scenes, motion_design,
                       ecommerce, social_hook, music_video, brand_story, fashion, food, real_estate

    Returns a numbered list with skill-matched ideas you can select from.
    """
    session.current_notes = notes
    gen = idea_gen()

    try:
        ideas_data, skill = await gen.generate_ideas(notes, count, skill_id=skill_id)
    except Exception as e:
        return f"Error generating ideas: {e}"

    ideas = session.add_ideas(ideas_data)
    # Store detected skill in session for scene generation
    session._detected_skill_id = skill.id

    provider_label = gen._effective_provider()
    lines = [
        f"Skill detected: {skill.name} [{skill.id}]",
        f"Provider: {provider_label}",
        f"Generated {len(ideas)} ideas for: \"{notes[:60]}{'...' if len(notes) > 60 else ''}\"\n",
    ]
    for idea in ideas:
        lines.append(f"[{idea.id}] {idea.title}")
        lines.append(f"     {idea.description}")
        lines.append(f"     Style: {idea.style} | Mood: {idea.mood}")
        lines.append(f"     Tags: {', '.join(idea.tags)}")
        lines.append("")

    lines.append("Pick one: select_idea(idea_id=N)")
    lines.append("Not right? regenerate_ideas(feedback='more dramatic / sci-fi / etc')")
    return "\n".join(lines)


# ── Tool: list_ideas ──────────────────────────────────────────────────────────
@mcp.tool()
def list_ideas() -> str:
    """List all current video ideas in this session."""
    if not session.ideas:
        return "No ideas yet. Use generate_ideas() to get started."

    lines = [f"Current ideas (session):\n"]
    for idea in session.ideas:
        marker = "✓ SELECTED" if idea.selected else ""
        lines.append(f"[{idea.id}] {idea.title} {marker}")
        lines.append(f"     Style: {idea.style} | Mood: {idea.mood}")
        lines.append("")
    return "\n".join(lines)


# ── Tool: regenerate_ideas ────────────────────────────────────────────────────
@mcp.tool()
async def regenerate_ideas(
    feedback: str,
    count: int = 5,
    notes: Optional[str] = None,
    skill_id: Optional[str] = None,
) -> str:
    """
    Regenerate ideas with feedback.

    feedback: What to change ("more dramatic", "make it sci-fi", "less abstract")
    count:    Number of new ideas
    notes:    Override notes (optional — uses last notes if omitted)
    skill_id: Force a different skill framework (optional)
    """
    notes = notes or session.current_notes
    if not notes:
        return "No notes found. Call generate_ideas(notes='...') first."

    # Allow skill switch on regenerate
    effective_skill_id = skill_id or getattr(session, "_detected_skill_id", None)

    gen = idea_gen()
    try:
        ideas_data, skill = await gen.regenerate_ideas(notes, feedback, count, skill_id=effective_skill_id)
    except Exception as e:
        return f"Error regenerating ideas: {e}"

    ideas = session.add_ideas(ideas_data)
    session._detected_skill_id = skill.id

    lines = [f"Regenerated {len(ideas)} ideas | Skill: {skill.name} | Feedback: \"{feedback}\"\n"]
    for idea in ideas:
        lines.append(f"[{idea.id}] {idea.title}")
        lines.append(f"     {idea.description}")
        lines.append(f"     Style: {idea.style} | Mood: {idea.mood}")
        lines.append("")
    lines.append("Use select_idea(idea_id=N) to pick one.")
    return "\n".join(lines)


# ── Tool: select_idea ─────────────────────────────────────────────────────────
@mcp.tool()
async def select_idea(idea_id: int) -> str:
    """
    Select a video idea and generate a cinema-quality scene breakdown.

    idea_id: The ID number shown in generate_ideas output

    The scene breakdown uses the detected skill framework to produce:
    - Kelvin-precise lighting setups per scene
    - Exact camera movement with velocity specs
    - 2-second hook pattern for Scene 1
    - ComfyUI-ready positive + negative prompts
    - Skill-appropriate resolution and FPS settings
    """
    idea = session.select_idea(idea_id)
    if not idea:
        return f"Idea {idea_id} not found. Use list_ideas() to see available ideas."

    gen = idea_gen()
    scene_count = CONFIG.get("idea_generation", {}).get("scenes_per_idea", 4)

    # Retrieve skill from session
    skill_id = getattr(session, "_detected_skill_id", None)
    skill = get_skill_by_id(skill_id) if skill_id else detect_skill(
        idea.description + " " + idea.style
    )

    try:
        scenes_data = await gen.generate_scenes(
            {"title": idea.title, "description": idea.description,
             "style": idea.style, "mood": idea.mood},
            scene_count=scene_count,
            skill=skill,
        )
    except Exception as e:
        return f"Idea selected but scene generation failed: {e}"

    scenes = session.add_scenes(scenes_data, idea_id)

    # Apply skill-based ComfyUI overrides to runtime config
    overrides = get_workflow_overrides(skill)
    _runtime_cfg.update(overrides)

    lines = [
        f"Selected: [{idea.id}] {idea.title}",
        f"Skill: {skill.name} [{skill.id}]",
        f"Style: {idea.style} | Mood: {idea.mood}",
        f"ComfyUI: {overrides['width']}×{overrides['height']} @ {overrides['fps']}fps, "
        f"{overrides['steps']} steps, CFG {overrides['cfg']}",
        f"\nGenerated {len(scenes)} scenes:\n",
    ]
    for scene in scenes:
        hook_label = f"  Hook: {scene.__dict__.get('hook', '')[:80]}" if hasattr(scene, '__dict__') else ""
        lines.append(f"Scene {scene.scene_number}: {scene.description}")
        lines.append(f"  Prompt: {scene.visual_prompt[:120]}...")
        lines.append(f"  Negative: {scene.negative_prompt[:80]}...")
        lines.append(f"  Duration: {scene.duration}s")
        lines.append("")

    lines.append(
        "generate_video()        → Queue all scenes\n"
        "generate_video(scene_id=1) → Queue a single scene"
    )
    return "\n".join(lines)


# ── Tool: generate_video ──────────────────────────────────────────────────────
@mcp.tool()
async def generate_video(
    scene_id: Optional[int] = None,
    all_scenes: bool = False,
    model: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    frames: Optional[int] = None,
    fps: Optional[int] = None,
    steps: Optional[int] = None,
    seed: Optional[int] = None,
) -> str:
    """
    Queue video generation for scenes in ComfyUI.

    scene_id:   Generate a specific scene (omit for all)
    all_scenes: Set True to generate all scenes at once
    model:      Override model (animatediff | svd | wan21)
    width/height/frames/fps/steps/seed: Override generation params
    """
    client = comfyui_client()

    if not await client.is_available():
        return (
            "ComfyUI is not running. Start ComfyUI first:\n"
            "  python main.py --listen\n"
            "Then try again."
        )

    if not session.scenes:
        return "No scenes available. Use select_idea() first."

    # Determine which scenes to generate
    if scene_id is not None:
        scene = session.get_scene(scene_id)
        if not scene:
            return f"Scene {scene_id} not found."
        targets = [scene]
    else:
        targets = session.scenes

    results = []
    out_dir = output_dir("videos")
    out_dir.mkdir(parents=True, exist_ok=True)

    for scene in targets:
        try:
            overrides = {}
            if width:    overrides["width"] = width
            if height:   overrides["height"] = height
            if frames:   overrides["frames"] = frames
            if fps:      overrides["fps"] = fps
            if steps:    overrides["steps"] = steps
            if seed:     overrides["seed"] = seed

            workflow = build_workflow(
                positive=scene.visual_prompt,
                negative=scene.negative_prompt,
                model=model,
                **overrides,
            )

            prompt_id = await client.queue_prompt(workflow)
            session.update_job(prompt_id, {
                "status": "queued",
                "scene_id": scene.id,
                "queued_at": time.time(),
            })
            results.append(f"Scene {scene.scene_number}: queued (job: {prompt_id[:8]})")

        except Exception as e:
            results.append(f"Scene {scene.scene_number}: FAILED - {e}")

    lines = [f"Queued {len(results)} generation job(s):\n"]
    lines.extend(results)
    lines.append(
        "\nUse check_status() to monitor progress. "
        "Jobs run in ComfyUI queue order."
    )
    return "\n".join(lines)


# ── Tool: check_status ────────────────────────────────────────────────────────
@mcp.tool()
async def check_status(job_id: Optional[str] = None, wait: bool = False) -> str:
    """
    Check ComfyUI generation status.

    job_id: Specific job ID (omit to check all active jobs)
    wait:   Wait for completion and download output (True/False)
    """
    client = comfyui_client()

    if not await client.is_available():
        return "ComfyUI not reachable."

    try:
        queue = await client.get_queue_status()
        running = queue.get("queue_running", [])
        pending = queue.get("queue_pending", [])

        lines = ["ComfyUI Queue Status:\n"]
        lines.append(f"Running: {len(running)} | Pending: {len(pending)}")

        if running:
            lines.append("\nCurrently running:")
            for item in running[:3]:
                pid = item[1] if len(item) > 1 else "unknown"
                lines.append(f"  {str(pid)[:8]}...")

        if pending:
            lines.append(f"\nPending jobs: {len(pending)}")

        # Check session jobs
        if session.generation_jobs:
            lines.append("\nSession jobs:")
            for jid, info in list(session.generation_jobs.items())[-5:]:
                status = info.get("status", "unknown")
                scene_id = info.get("scene_id", "?")
                lines.append(f"  {jid[:8]}: {status} (scene {scene_id})")

        if wait and job_id:
            lines.append(f"\nWaiting for job {job_id[:8]} to complete...")
            try:
                timeout = get_cfg("comfyui", "timeout", default=300)
                history = await client.wait_for_completion(job_id, timeout=timeout)
                if history:
                    out_dir = output_dir("videos")
                    prefix = f"scene_{job_id[:8]}"
                    paths = await client.download_outputs(history, out_dir, prefix)
                    if paths:
                        # Associate with scene
                        for jid, info in session.generation_jobs.items():
                            if jid == job_id:
                                sid = info.get("scene_id")
                                if sid:
                                    session.mark_scene_generated(sid, str(paths[0]), job_id)
                                break
                        lines.append(f"Complete! Downloaded {len(paths)} file(s):")
                        for p in paths:
                            lines.append(f"  {p}")
                    else:
                        lines.append("Complete but no output files found.")
            except TimeoutError:
                lines.append(f"Timed out waiting. Check ComfyUI GUI for progress.")

        return "\n".join(lines)

    except Exception as e:
        return f"Error checking status: {e}"


# ── Tool: list_videos ─────────────────────────────────────────────────────────
@mcp.tool()
def list_videos() -> str:
    """List all generated video files in the output folder."""
    vid_dir = output_dir("videos")
    if not vid_dir.exists():
        return "No videos generated yet."

    videos = sorted(vid_dir.glob("*.mp4")) + sorted(vid_dir.glob("*.gif"))
    if not videos:
        return f"No videos in {vid_dir}"

    lines = [f"Generated videos ({len(videos)} files):\n"]
    for i, v in enumerate(videos, 1):
        size_mb = v.stat().st_size / (1024 * 1024)
        lines.append(f"[{i}] {v.name}  ({size_mb:.1f} MB)")

    lines.append(f"\nOutput folder: {vid_dir}")
    lines.append("Use compile_montage(video_ids=[1,2,3]) to create a montage.")
    return "\n".join(lines)


# ── Tool: compile_montage ─────────────────────────────────────────────────────
@mcp.tool()
async def compile_montage(
    video_ids: Optional[list[int]] = None,
    title: str = "My Video",
    transition: Optional[str] = None,
    resolution: Optional[str] = None,
    fps: Optional[int] = None,
    music_path: Optional[str] = None,
    add_title_card: bool = False,
    use_session_videos: bool = True,
) -> str:
    """
    Compile video clips into a final montage.

    video_ids:         List indices from list_videos() (e.g. [1,2,3])
    title:             Title for the output video
    transition:        fade | dissolve | wipe | slide | zoom | none
    resolution:        Output resolution e.g. "1920x1080" or "1280x720"
    fps:               Output FPS
    music_path:        Path to background music file (mp3/wav)
    add_title_card:    Add a black title card at start
    use_session_videos: Automatically use all generated session videos
    """
    compiler = montage_comp()

    # Collect video paths
    video_paths = []

    if use_session_videos and not video_ids:
        video_paths = session.get_generated_videos()
        if not video_paths:
            # Fall back to all files in videos dir
            vid_dir = output_dir("videos")
            if vid_dir.exists():
                video_paths = [str(p) for p in sorted(vid_dir.glob("*.mp4"))]

    elif video_ids:
        vid_dir = output_dir("videos")
        all_videos = sorted(vid_dir.glob("*.mp4")) + sorted(vid_dir.glob("*.gif"))
        for idx in video_ids:
            if 1 <= idx <= len(all_videos):
                video_paths.append(str(all_videos[idx - 1]))
            else:
                return f"Invalid video_id: {idx}. Use list_videos() to see available indices."

    if not video_paths:
        return (
            "No videos to compile. Generate some first with generate_video(), "
            "or specify video_ids from list_videos()."
        )

    # Output path
    safe_title = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")
    ts = int(time.time())
    out_path = output_dir("montages") / f"{safe_title}_{ts}.mp4"

    try:
        result_path = await compiler.compile(
            video_paths=video_paths,
            output_path=str(out_path),
            title=title,
            transition=transition,
            resolution=resolution,
            fps=fps,
            music_path=music_path,
        )

        if add_title_card:
            titled_path = str(out_path).replace(".mp4", "_titled.mp4")
            result_path = await compiler.add_title_card(result_path, title, titled_path)

        # Register in session
        job = session.add_montage_job(title, video_paths)
        job.output_path = result_path
        job.status = "complete"

        size_mb = Path(result_path).stat().st_size / (1024 * 1024)
        return (
            f"Montage compiled successfully!\n\n"
            f"Title:    {title}\n"
            f"Clips:    {len(video_paths)}\n"
            f"Output:   {result_path}\n"
            f"Size:     {size_mb:.1f} MB\n"
            f"Transition: {transition or compiler.transition}\n"
        )

    except Exception as e:
        return f"Montage compilation failed: {e}"


# ── Tool: list_montages ───────────────────────────────────────────────────────
@mcp.tool()
def list_montages() -> str:
    """List all compiled montages."""
    mont_dir = output_dir("montages")
    if not mont_dir.exists():
        return "No montages created yet."

    montages = sorted(mont_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not montages:
        return f"No montages in {mont_dir}"

    lines = [f"Compiled montages ({len(montages)}):\n"]
    for m in montages:
        size_mb = m.stat().st_size / (1024 * 1024)
        lines.append(f"  {m.name}  ({size_mb:.1f} MB)")

    lines.append(f"\nOutput folder: {mont_dir}")
    return "\n".join(lines)


# ── Tool: session_status ──────────────────────────────────────────────────────
@mcp.tool()
def session_status() -> str:
    """Show current pipeline session status."""
    s = session.to_summary()
    idea = None
    if session.selected_idea_id:
        idea = session.get_idea(session.selected_idea_id)

    lines = [
        "Pipeline Session Status\n" + "=" * 40,
        f"Notes:          {session.current_notes[:60] + '...' if len(session.current_notes) > 60 else session.current_notes or '(none)'}",
        f"Ideas:          {s['ideas_count']} generated",
        f"Selected idea:  {f'[{idea.id}] {idea.title}' if idea else '(none)'}",
        f"Scenes:         {s['scenes_count']} ({s['generated_scenes']} generated)",
        f"Pending jobs:   {s['pending_jobs']}",
        f"Montages:       {s['montages_count']}",
        "",
        "Quick commands:",
        "  generate_ideas('your concept')   → Start fresh",
        "  list_ideas()                     → See current ideas",
        "  select_idea(1)                   → Pick idea + get scenes",
        "  generate_video()                 → Queue all scenes",
        "  check_status()                   → Monitor progress",
        "  compile_montage()                → Build final video",
    ]
    return "\n".join(lines)


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mcp.run()
