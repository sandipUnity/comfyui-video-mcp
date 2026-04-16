"""
Skill-powered idea and prompt generation.
Uses Seedance 2.0 skill frameworks (adapted from higgsfield-seedance2-jineng)
to generate cinematically precise ComfyUI prompts with full scene continuity.

Supports:
  - Claude API (ANTHROPIC_API_KEY)
  - Local Ollama (no API key needed)
  - Pure skill-based generation (no LLM needed — offline mode)
"""

import json
import os
import re
import random
import aiohttp
from typing import Optional

from skills_engine import (
    SkillSpec, detect_skill, get_skill_by_id, list_skills,
    build_comfyui_positive, build_comfyui_negative, get_workflow_overrides,
)

# ── Prompt Templates ───────────────────────────────────────────────────────────

IDEA_USER_TEMPLATE = """Based on these notes: "{notes}"
Skill framework: {skill_name} ({skill_description})

Generate {count} distinct video ideas that fit this skill's visual style.

For each idea:
- title: Short catchy title (max 8 words)
- description: What the video shows (2-3 sentences, be visual and specific)
- style: Visual style matching the skill framework
- mood: Emotional tone
- tags: 3-5 descriptive tags

Respond with JSON array ONLY — no markdown, no explanation:
[{{"title":"...","description":"...","style":"...","mood":"...","tags":["..."]}}]"""


SCENE_USER_TEMPLATE = """You are writing a {scene_count}-scene video script with STRICT visual continuity.
Every scene must feel like the same film — same character, same world, same color logic.

═══════════════════════════════════════════════════
 STORY BRIEF
═══════════════════════════════════════════════════
Title:        "{title}"
Logline:      {description}
Visual Style: {style}
Mood Arc:     {mood} — building across all {scene_count} scenes
Framework:    {skill_name}

═══════════════════════════════════════════════════
 STEP 1 — DEFINE YOUR PROTAGONIST (do this mentally before writing)
═══════════════════════════════════════════════════
Invent ONE protagonist with 4-6 specific physical details:
  • Hair color + style  • Clothing (specific garment + color)
  • One distinctive accessory  • Skin tone or notable feature
Example: "young woman, cropped silver hair, worn burgundy leather jacket,
          holographic stud earrings, pale skin with scattered freckles"

This EXACT description must appear in every scene's visual_prompt.
Do NOT vary it. If clothing gets damaged (torn jacket, muddy boots), KEEP mentioning it.

═══════════════════════════════════════════════════
 STEP 2 — NARRATIVE ARC (one scene per act)
═══════════════════════════════════════════════════
Scene 1 [HOOK — 10% tension]:
  • Establish protagonist + world in one striking image
  • Camera: wide establishing → slow push-in to close-up revelation
  • Emotional tone: calm / curious / unaware of what's coming

Scene 2 [BUILD — 40% tension]:
  • Protagonist moves deeper, discovers the conflict
  • Camera: tracking shot following motion, or dolly-in during realization
  • Emotional tone: unease / wonder / first signs of danger

Scene 3 [CLIMAX — 90% tension]:
  • Maximum conflict. Highest stakes. Unstable camera.
  • Camera: handheld shake, Dutch tilt, crash zoom, or whip-pan
  • Emotional tone: fear / awe / fight-or-flight

Scene 4 [RESOLUTION — 60% tension, falling]:
  • Aftermath. Protagonist changed. World feels different.
  • Camera: slow pull-back to wide, or static contemplative hold
  • Emotional tone: exhaustion / revelation / fragile peace

═══════════════════════════════════════════════════
 CAMERA VOCABULARY (choose from these exactly)
═══════════════════════════════════════════════════
{camera_examples}

═══════════════════════════════════════════════════
 LIGHTING VOCABULARY (choose from these exactly)
═══════════════════════════════════════════════════
{lighting_examples}

═══════════════════════════════════════════════════
 QUALITY TAGS (end EVERY visual_prompt with these)
═══════════════════════════════════════════════════
{quality_tags}

═══════════════════════════════════════════════════
 NEGATIVE PROMPT (use for ALL scenes)
═══════════════════════════════════════════════════
{negative_tags}

═══════════════════════════════════════════════════
 OUTPUT FORMAT — JSON array only, no other text
═══════════════════════════════════════════════════
[
  {{
    "scene_number": 1,
    "act": "HOOK",
    "protagonist": "EXACT character description you invented — paste same string in every scene",
    "description": "one sentence: what happens narratively in this scene",
    "visual_prompt": "PROTAGONIST_DESC, action verb + what they do, camera_move, lighting_spec, quality_tags",
    "negative_prompt": "ugly, deformed, bad anatomy, blurry, watermark, static, frozen",
    "duration": 5.0,
    "transition_to_next": "how this shot ends / what visual leads into the next scene"
  }}
]

═══════════════════════════════════════════════════
 CONTINUITY EXAMPLE (notice protagonist is IDENTICAL across all 4)
═══════════════════════════════════════════════════
Scene 1 visual_prompt: "young woman, cropped silver hair, worn burgundy leather jacket, holographic earrings — standing at rain-slicked intersection gazing upward, wide establishing shot pulling back from close-up, cool 5600K overhead mixed with neon magenta bounce from left, masterpiece, best quality"
Scene 2 visual_prompt: "young woman, cropped silver hair, worn burgundy leather jacket, holographic earrings — walking forward through parting crowd, tracking shot at hip height following her stride, 4200K neon cyan flooding from right casting hard shadows, speed lines at frame edges, masterpiece, best quality"
Scene 3 visual_prompt: "young woman, cropped silver hair, worn burgundy leather jacket now torn at shoulder, holographic earrings catching light — facing towering mechanical figure, handheld 15° Dutch tilt, creature silhouetted against 8000K cold backlight with orange rim from explosion below, masterpiece, best quality"
Scene 4 visual_prompt: "young woman, cropped silver hair, worn burgundy leather jacket with torn shoulder — kneeling on wet pavement, slow pull-back from extreme close-up of face to wide city panorama, warm 2700K single streetlamp above creating intimate pool of light, holographic earrings faintly glowing, masterpiece, best quality"

Now write {scene_count} scenes for "{title}". Respond with the JSON array ONLY."""


REGEN_USER_TEMPLATE = """Notes: "{notes}" | Skill: {skill_name}
Feedback: "{feedback}"

Generate {count} NEW video ideas different from before.
Apply {skill_name} framework: {skill_description}
Use its vocabulary: camera moves, lighting specs, visual hooks.

JSON array ONLY:
[{{"title":"...","description":"...","style":"...","mood":"...","tags":["..."]}}]"""


# ── Narrative Arc Definitions ──────────────────────────────────────────────────

_ACT_LABELS = ["HOOK", "BUILD", "CLIMAX", "RESOLUTION"]
_ACT_TENSION = [0.1, 0.4, 0.9, 0.6]
_ACT_CAMERA_HINTS = [
    "wide establishing shot pulling back from extreme close-up",
    "tracking shot following subject's motion",
    "handheld shake with Dutch tilt 15-25 degrees",
    "slow static hold or gentle pull-back to wide",
]
_ACT_EMOTION = ["curious/unaware", "unease/discovery", "fear/awe/climax", "exhaustion/revelation"]


# ── Offline Fallback Templates (no LLM) ───────────────────────────────────────

def _build_protagonist(idea: dict, skill: SkillSpec) -> str:
    """Build a consistent protagonist description from the idea and skill."""
    style = idea.get("style", "").lower()
    # Style-aware protagonist hints
    if any(k in style for k in ("anime", "shonen", "seinen", "manga")):
        templates = [
            "young woman, short dark hair with teal highlights, worn white school jacket, silver chain necklace",
            "young man, spiky auburn hair, tactical vest over dark shirt, fingerless gloves",
            "young woman, long violet hair in loose braid, flowing oversized coat, glowing blue earrings",
        ]
    elif any(k in style for k in ("cyber", "neon", "sci-fi", "futurist")):
        templates = [
            "young woman, cropped silver hair, worn burgundy leather jacket, holographic stud earrings",
            "young man, shaved head with neural implant scar, reflective visor, black tactical jacket",
        ]
    elif any(k in style for k in ("fantasy", "magic", "dragon", "mystical")):
        templates = [
            "young woman, flame-red hair braided with silver threads, worn emerald cloak, runic amulet",
            "young man, silver-streaked dark hair, tattered battle robe, glowing amber eyes",
        ]
    else:
        templates = [
            "young woman, cropped dark hair, worn leather jacket, small gold hoop earrings",
            "young man, tousled brown hair, charcoal coat over grey shirt, weathered boots",
        ]
    return random.choice(templates)


def _offline_ideas(notes: str, skill: SkillSpec, count: int) -> list[dict]:
    """Generate template-based ideas when no LLM is available."""
    hook = random.choice(skill.hook_patterns)
    camera = random.choice(skill.camera_vocabulary)
    mood_options = ["epic", "melancholic", "intense", "mysterious", "joyful", "dramatic"]
    return [
        {
            "title": f"{skill.name} — {notes[:30].strip()}",
            "description": (
                f"A {skill.name.lower()} video exploring: {notes.strip()}. "
                f"Opening with: {hook}. "
                f"Camera technique: {camera}."
            ),
            "style": skill.style_tags[0] if skill.style_tags else "cinematic",
            "mood": random.choice(mood_options),
            "tags": skill.style_tags[:3] + [skill.id],
        }
        for _ in range(min(count, 3))
    ]


def _offline_scenes(idea: dict, skill: SkillSpec, scene_count: int) -> list[dict]:
    """Generate template-based scenes with built-in continuity when no LLM is available."""
    protagonist = _build_protagonist(idea, skill)
    scenes = []

    for i in range(scene_count):
        act_idx = min(i, len(_ACT_LABELS) - 1)
        act = _ACT_LABELS[act_idx]
        cam_hint = _ACT_CAMERA_HINTS[act_idx]
        emotion = _ACT_EMOTION[act_idx]

        # Pull skill vocabulary in round-robin
        camera = skill.camera_vocabulary[i % len(skill.camera_vocabulary)]
        lighting = skill.lighting_vocabulary[i % len(skill.lighting_vocabulary)]

        # Scene-specific action that builds the narrative
        base_desc = idea.get("description", "").strip()
        if act == "HOOK":
            action = f"standing in the world of {base_desc[:40]}, taking in the surroundings"
        elif act == "BUILD":
            action = f"moving deeper into the scene, discovering the central conflict"
        elif act == "CLIMAX":
            action = f"facing the peak danger, maximum tension and motion"
        else:
            action = f"in the aftermath, changed by the experience"

        positive = build_comfyui_positive(
            f"{protagonist}, {action}, {camera}, {lighting}",
            skill,
        )
        negative = build_comfyui_negative(skill)

        # Transition hint
        next_act = _ACT_LABELS[min(i + 1, len(_ACT_LABELS) - 1)]
        transition = f"shot ends with protagonist facing the next challenge — leads into {next_act}"

        scenes.append({
            "scene_number": i + 1,
            "act": act,
            "protagonist": protagonist,
            "description": f"[{act}] {action[:80]}",
            "visual_prompt": positive,
            "negative_prompt": negative,
            "duration": 5.0,
            "transition_to_next": transition,
        })

    return scenes


# ── Continuity Helpers ─────────────────────────────────────────────────────────

def _extract_protagonist_anchor(scenes: list[dict]) -> str:
    """
    Pull the protagonist description from Scene 1's 'protagonist' field,
    falling back to extracting the first clause of its visual_prompt.
    """
    if not scenes:
        return ""

    s1 = scenes[0]

    # Prefer the dedicated protagonist field
    anchor = s1.get("protagonist", "").strip()
    if len(anchor) > 20:
        return anchor

    # Fall back: first comma-separated clause of visual_prompt
    vp = s1.get("visual_prompt", "")
    if vp:
        clauses = [c.strip() for c in vp.split(",")]
        # Grab first 3-5 clauses that describe a person
        person_clauses = []
        for c in clauses[:8]:
            cl = c.lower()
            if any(w in cl for w in ("woman", "man", "girl", "boy", "person",
                                      "hair", "jacket", "coat", "shirt", "earring",
                                      "young", "old", "face", "eye")):
                person_clauses.append(c)
            if len(person_clauses) >= 5:
                break
        if person_clauses:
            return ", ".join(person_clauses)

    return ""


def _enforce_continuity(scenes: list[dict], skill: SkillSpec) -> list[dict]:
    """
    Post-processing pass that ensures every scene beyond Scene 1:
      1. Has the same protagonist anchor at the start of visual_prompt
      2. Has a valid act label
      3. Has a non-trivial negative_prompt
    Also re-numbers scenes to be sequential.
    """
    if not scenes:
        return scenes

    anchor = _extract_protagonist_anchor(scenes)

    for i, scene in enumerate(scenes):
        # Ensure sequential scene numbers
        scene["scene_number"] = i + 1

        # Assign act label if missing or wrong type
        if not isinstance(scene.get("act"), str) or not scene["act"].strip():
            scene["act"] = _ACT_LABELS[min(i, len(_ACT_LABELS) - 1)]

        # Inject protagonist anchor into visual_prompt for scenes 2+
        if anchor and i > 0:
            vp = scene.get("visual_prompt", "")
            # Only inject if anchor isn't already there (check first 200 chars)
            anchor_check = anchor[:30].lower()
            if anchor_check not in vp[:200].lower():
                # Prepend the anchor + continuity marker
                same_label = "same protagonist — " if i == 1 else "same protagonist, now "
                scene["visual_prompt"] = f"{same_label}{anchor}, {vp}"

        # Ensure protagonist field is set
        if anchor and not scene.get("protagonist"):
            scene["protagonist"] = anchor

        # Rebuild negative if too short
        np = scene.get("negative_prompt", "")
        if len(np) < 30:
            scene["negative_prompt"] = build_comfyui_negative(skill)

        # Ensure transition_to_next exists
        if not scene.get("transition_to_next"):
            next_act = _ACT_LABELS[min(i + 1, len(_ACT_LABELS) - 1)]
            scene["transition_to_next"] = f"cut to {next_act} scene"

    return scenes


# ── Main Generator ─────────────────────────────────────────────────────────────

class IdeaGenerator:
    """Generate video ideas and scenes using Seedance skills + LLM."""

    def __init__(self, config: dict):
        self.config = config
        self.provider = config.get("provider", "auto")
        self.claude_model = config.get("claude_model", "claude-opus-4-6")
        self.ollama_model = config.get("ollama_model", "llama3.2")
        self.ollama_host = config.get("ollama_host", "http://localhost:11434")

    def _effective_provider(self) -> str:
        """Determine which LLM to use based on available credentials."""
        if self.provider not in ("auto", "claude", "ollama", "offline"):
            return "offline"

        if self.provider == "claude":
            if os.environ.get("ANTHROPIC_API_KEY"):
                return "claude"
            return "ollama"  # fallback

        if self.provider == "ollama":
            return "ollama"

        if self.provider == "offline":
            return "offline"

        # auto: try claude → ollama → offline
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "claude"
        return "ollama"

    async def generate_ideas(
        self, notes: str, count: int = 5, skill_id: Optional[str] = None
    ) -> tuple[list[dict], SkillSpec]:
        """Generate video ideas. Returns (ideas, detected_skill)."""
        skill = detect_skill(notes, override=skill_id)
        provider = self._effective_provider()

        if provider == "offline":
            return _offline_ideas(notes, skill, count), skill

        system = self._build_system(skill)
        user = IDEA_USER_TEMPLATE.format(
            notes=notes,
            skill_name=skill.name,
            skill_description=skill.description,
            count=count,
        )

        try:
            text = await self._call_llm(provider, system, user)
            ideas = self._parse_json_list(text)
            ideas = [i for i in ideas if isinstance(i, dict) and i.get("title")]
            if not ideas:
                raise ValueError("LLM returned no valid idea objects")
            return ideas, skill
        except Exception as e:
            import sys
            print(f"[idea_generator] LLM error ({provider}): {e}", file=sys.stderr)
            return _offline_ideas(notes, skill, count), skill

    async def regenerate_ideas(
        self, notes: str, feedback: str, count: int = 5, skill_id: Optional[str] = None
    ) -> tuple[list[dict], SkillSpec]:
        skill = detect_skill(notes, override=skill_id)
        provider = self._effective_provider()

        if provider == "offline":
            return _offline_ideas(notes, skill, count), skill

        system = self._build_system(skill)
        user = REGEN_USER_TEMPLATE.format(
            notes=notes,
            feedback=feedback,
            count=count,
            skill_name=skill.name,
            skill_description=skill.description,
        )

        try:
            text = await self._call_llm(provider, system, user)
            ideas = self._parse_json_list(text)
            return ideas, skill
        except Exception:
            return _offline_ideas(notes, skill, count), skill

    async def generate_scenes(
        self, idea: dict, scene_count: int = 4, skill: Optional[SkillSpec] = None
    ) -> list[dict]:
        """
        Generate a scene breakdown with full visual continuity.

        Pipeline:
          1. LLM generates all scenes with protagonist anchor + act structure
          2. _enforce_continuity() injects anchor into any scene that drifted
          3. _enrich_scene() adds quality boosters and validates negative prompts
        """
        if skill is None:
            skill = detect_skill(idea.get("description", "") + " " + idea.get("style", ""))

        provider = self._effective_provider()
        if provider == "offline":
            return _offline_scenes(idea, skill, scene_count)

        system = self._build_system_scenes(skill)
        user = SCENE_USER_TEMPLATE.format(
            scene_count=scene_count,
            title=idea.get("title", ""),
            description=idea.get("description", ""),
            skill_name=skill.name,
            style=idea.get("style", skill.style_tags[0] if skill.style_tags else ""),
            mood=idea.get("mood", "dramatic"),
            camera_examples="\n  ".join(skill.camera_vocabulary[:4]),
            lighting_examples="\n  ".join(skill.lighting_vocabulary[:3]),
            quality_tags=", ".join(skill.quality_boosters[:5]),
            negative_tags=build_comfyui_negative(skill),
        )

        try:
            text = await self._call_llm(provider, system, user,
                                        max_tokens_claude=5000)
            scenes = self._parse_json_list(text)
            if not scenes:
                raise ValueError("LLM returned empty scene list")

            # ── Continuity pass ────────────────────────────────────────────────
            scenes = _enforce_continuity(scenes, skill)

            # ── Quality booster pass ───────────────────────────────────────────
            scenes = [self._enrich_scene(s, skill) for s in scenes]

            return scenes

        except Exception as e:
            import sys
            print(f"[idea_generator] Scene generation error ({provider}): {e}", file=sys.stderr)
            return _offline_scenes(idea, skill, scene_count)

    def _enrich_scene(self, scene: dict, skill: SkillSpec) -> dict:
        """Post-process a scene — add quality boosters and validate fields."""
        vp = scene.get("visual_prompt", "")

        # Add skill quality boosters if not present
        for b in skill.quality_boosters[:4]:
            if b.lower() not in vp.lower():
                vp = vp + ", " + b
        scene["visual_prompt"] = vp

        # Rebuild negative if too short
        np = scene.get("negative_prompt", "")
        if len(np) < 30:
            scene["negative_prompt"] = build_comfyui_negative(skill, np)

        # Ensure duration is a float
        if not isinstance(scene.get("duration"), (int, float)):
            scene["duration"] = 5.0

        # Ensure act label
        sn = scene.get("scene_number", 1)
        if not scene.get("act"):
            scene["act"] = _ACT_LABELS[min(int(sn) - 1, len(_ACT_LABELS) - 1)]

        return scene

    def _build_system(self, skill: SkillSpec) -> str:
        """System prompt for idea generation."""
        return (
            f"You are an expert AI video concept developer specialising in {skill.name}.\n"
            f"{skill.prompt_template}\n\n"
            f"ComfyUI integration notes: {skill.comfyui_notes}\n\n"
            "CRITICAL: Respond with valid JSON only. No markdown, no explanations, no code fences.\n"
            "Specificity rule: never use vague terms. Always use precise technical language:\n"
            "  ✓ '3000K key light at 45° camera-left, 100% intensity'\n"
            "  ✓ '3 ft/s dolly forward tracking subject'\n"
            "  ✗ 'warm lighting'  ✗ 'fast camera'  ✗ 'beautiful'"
        )

    def _build_system_scenes(self, skill: SkillSpec) -> str:
        """System prompt for scene generation — emphasises continuity."""
        return (
            f"You are an expert AI video script writer and cinematographer specialising in {skill.name}.\n"
            f"{skill.prompt_template}\n\n"
            f"ComfyUI integration notes: {skill.comfyui_notes}\n\n"
            "CRITICAL RULES:\n"
            "1. Respond with valid JSON only — no markdown, no explanations, no code fences.\n"
            "2. VISUAL CONTINUITY: The same protagonist must appear with IDENTICAL physical description "
            "in every scene's visual_prompt. This is non-negotiable.\n"
            "3. NARRATIVE ARC: Each scene must escalate tension following HOOK → BUILD → CLIMAX → RESOLUTION.\n"
            "4. SPECIFICITY: Never use vague terms:\n"
            "   ✓ '4200K neon cyan from camera-left, 80% intensity, hard shadows'\n"
            "   ✓ 'handheld 2 ft/s tracking shot at hip height'\n"
            "   ✗ 'colorful lighting'  ✗ 'dynamic camera'  ✗ 'beautiful scene'"
        )

    async def _call_llm(
        self,
        provider: str,
        system: str,
        user: str,
        max_tokens_claude: int = 3000,
    ) -> str:
        if provider == "claude":
            return await self._call_claude(system, user, max_tokens_claude)
        return await self._call_ollama(system, user)

    async def _call_claude(self, system: str, user: str, max_tokens: int = 3000) -> str:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=self.claude_model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text

    async def _call_ollama(self, system: str, user: str) -> str:
        # Do NOT use format:"json" — Ollama forces object output which breaks JSON arrays.
        # Instead rely on the prompt's explicit instruction to return a JSON array.
        payload = {
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "num_predict": 8192,
                "temperature": 0.7,
                "num_ctx": 8192,
            },
        }
        async with aiohttp.ClientSession() as http:
            async with http.post(
                f"{self.ollama_host}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Ollama returned {resp.status}")
                data = await resp.json()
                return data["message"]["content"]

    def _parse_json_list(self, text: str) -> list[dict]:
        # Strip markdown fences
        text = re.sub(r"```(?:json)?\s*", "", text).strip()
        text = re.sub(r"```\s*$", "", text).strip()

        # Try full parse first
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                for key in ("scenes", "ideas", "results", "items", "data"):
                    if key in result and isinstance(result[key], list):
                        return result[key]
                return [result]
        except json.JSONDecodeError:
            pass

        # Try extracting the array portion only
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(0))
                return result if isinstance(result, list) else [result]
            except json.JSONDecodeError:
                pass

        # Recovery: extract all complete JSON objects from a truncated array
        objects = []
        for m in re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}', text, re.DOTALL):
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict) and any(k in obj for k in ("title", "scene_number", "description")):
                    objects.append(obj)
            except json.JSONDecodeError:
                continue
        if objects:
            return objects

        raise ValueError(f"Could not parse LLM response as JSON.\nResponse: {text[:500]}")
