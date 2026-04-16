"""
Skill-powered idea and prompt generation.
Uses Seedance 2.0 skill frameworks (adapted from higgsfield-seedance2-jineng)
to generate cinematically precise ComfyUI prompts.

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

SCENE_USER_TEMPLATE = """Create a {scene_count}-scene breakdown for:
Title: {title}
Description: {description}
Skill: {skill_name}
Style: {style} | Mood: {mood}

SKILL FRAMEWORK RULES — apply these exactly:
{skill_prompt_template}

For each scene provide a ComfyUI-ready prompt following the skill's technical language:
- scene_number: sequential number
- description: what happens (1-2 sentences)
- visual_prompt: DETAILED ComfyUI positive prompt using skill vocabulary:
    * Camera: {camera_examples}
    * Lighting: {lighting_examples}
    * Include quality boosters: {quality_tags}
- negative_prompt: {negative_tags}
- duration: seconds (2-5)
- hook: opening hook technique for scene 1 only: {hook_examples}

Respond with JSON array ONLY:
[{{"scene_number":1,"description":"...","visual_prompt":"...","negative_prompt":"...","duration":3.0,"hook":"..."}}]"""

REGEN_USER_TEMPLATE = """Notes: "{notes}" | Skill: {skill_name}
Feedback: "{feedback}"

Generate {count} NEW video ideas different from before.
Apply {skill_name} framework: {skill_description}
Use its vocabulary: camera moves, lighting specs, visual hooks.

JSON array ONLY:
[{{"title":"...","description":"...","style":"...","mood":"...","tags":["..."]}}]"""


# ── Offline Fallback Templates (no LLM) ───────────────────────────────────────

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
    """Generate template-based scenes when no LLM is available."""
    scenes = []
    for i in range(scene_count):
        camera = skill.camera_vocabulary[i % len(skill.camera_vocabulary)]
        lighting = skill.lighting_vocabulary[i % len(skill.lighting_vocabulary)]
        hook = skill.hook_patterns[0] if i == 0 else ""
        base = f"{idea.get('description', '')} — scene {i+1}"
        positive = build_comfyui_positive(
            f"{base}, {camera}, {lighting}",
            skill,
        )
        negative = build_comfyui_negative(skill)
        scenes.append({
            "scene_number": i + 1,
            "description": f"Scene {i+1}: {base[:80]}",
            "visual_prompt": positive,
            "negative_prompt": negative,
            "duration": 3.0,
            "hook": hook,
        })
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
            return ideas, skill
        except Exception:
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
        """Generate detailed scene breakdown with skill-quality prompts."""
        if skill is None:
            skill = detect_skill(idea.get("description", "") + " " + idea.get("style", ""))

        provider = self._effective_provider()
        if provider == "offline":
            return _offline_scenes(idea, skill, scene_count)

        system = self._build_system(skill)
        user = SCENE_USER_TEMPLATE.format(
            scene_count=scene_count,
            title=idea.get("title", ""),
            description=idea.get("description", ""),
            skill_name=skill.name,
            skill_prompt_template=skill.prompt_template,
            style=idea.get("style", skill.style_tags[0] if skill.style_tags else ""),
            mood=idea.get("mood", "dramatic"),
            camera_examples="\n    ".join(skill.camera_vocabulary[:3]),
            lighting_examples="\n    ".join(skill.lighting_vocabulary[:2]),
            quality_tags=", ".join(skill.quality_boosters[:5]),
            negative_tags=build_comfyui_negative(skill),
            hook_examples="\n    ".join(skill.hook_patterns[:2]),
        )

        try:
            text = await self._call_llm(provider, system, user)
            scenes = self._parse_json_list(text)
            # Post-process: inject skill boosters into each scene's prompt
            return [self._enrich_scene(s, skill) for s in scenes]
        except Exception:
            return _offline_scenes(idea, skill, scene_count)

    def _enrich_scene(self, scene: dict, skill: SkillSpec) -> dict:
        """Post-process a scene — ensure quality boosters and proper negative prompt."""
        vp = scene.get("visual_prompt", "")
        # Add quality boosters if not already present
        boosters = skill.quality_boosters[:4]
        for b in boosters:
            if b.lower() not in vp.lower():
                vp = vp + ", " + b
        scene["visual_prompt"] = vp

        # Rebuild negative if too short
        np = scene.get("negative_prompt", "")
        if len(np) < 30:
            scene["negative_prompt"] = build_comfyui_negative(skill, np)

        return scene

    def _build_system(self, skill: SkillSpec) -> str:
        return (
            f"You are an expert AI video prompt engineer specializing in {skill.name}.\n"
            f"{skill.prompt_template}\n\n"
            f"ComfyUI integration notes: {skill.comfyui_notes}\n\n"
            "CRITICAL: Respond with valid JSON only. No markdown, no explanations, no code fences.\n"
            "Specificity rule: never use vague terms. Always use precise technical language:\n"
            "  ✓ '3000K key light at 45° camera-left, 100% intensity'\n"
            "  ✓ '3 ft/s dolly forward tracking subject'\n"
            "  ✗ 'warm lighting'  ✗ 'fast camera'  ✗ 'beautiful'"
        )

    async def _call_llm(self, provider: str, system: str, user: str) -> str:
        if provider == "claude":
            return await self._call_claude(system, user)
        return await self._call_ollama(system, user)

    async def _call_claude(self, system: str, user: str) -> str:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=self.claude_model,
            max_tokens=3000,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text

    async def _call_ollama(self, system: str, user: str) -> str:
        payload = {
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": "json",
            "options": {"num_predict": 3000, "temperature": 0.7},
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.ollama_host}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=180),
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Ollama returned {resp.status}")
                data = await resp.json()
                return data["message"]["content"]

    def _parse_json_list(self, text: str) -> list[dict]:
        text = re.sub(r"```(?:json)?\s*", "", text).strip()
        text = re.sub(r"```\s*$", "", text).strip()
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            text = match.group(0)
        try:
            result = json.loads(text)
            return result if isinstance(result, list) else [result]
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON parse error: {e}\nResponse: {text[:400]}")
