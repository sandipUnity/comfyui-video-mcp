"""
Tests for pipeline/prompt_builder.py  (Sprint 5A)

Covers:
  - build_story_prompt        — Stage 3
  - build_character_prompt    — Stage 4
  - build_scene_prompts_prompt — Stage 5 (batch)
  - build_single_scene_prompt  — Stage 5 (single re-prompt)

Run:
  pytest tests/test_prompt_builder.py -v
"""

from __future__ import annotations

import sys
import types

import pytest

sys.path.insert(0, ".")

from pipeline.prompt_builder import (
    build_story_prompt,
    build_character_prompt,
    build_scene_prompts_prompt,
    build_single_scene_prompt,
    _header,
    _style_block,
    _story_block,
    _scenes_block,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_style_dna(skill_id="cinematic"):
    """Return a minimal StyleDNA-like object."""
    dna = types.SimpleNamespace(
        skill_id=skill_id,
        skill_name="Cinematic Realism",
        visual_style="photorealistic, film grain",
        motion_style="slow push-in",
        lighting_style="golden-hour side-lighting",
        color_palette=["#D4A96A", "#2C3E50", "#ECF0F1"],
        quality_boosters=["masterpiece", "best quality", "8k uhd", "sharp focus"],
    )
    return dna


def _make_scene(n=1, act="HOOK", desc="A warrior stands on a cliff edge",
                camera="Extreme wide shot", lighting="Golden-hour side-lighting"):
    return types.SimpleNamespace(
        scene_number=n,
        scene_id=f"scene_{n:02d}",
        act=act,
        description=desc,
        camera=camera,
        lighting=lighting,
        visual_prompt="",
        video_prompt="",
    )


def _make_character(desc="30-year-old woman, athletic build, long dark hair"):
    return types.SimpleNamespace(description=desc)


# ── _header ───────────────────────────────────────────────────────────────────

class TestHeader:
    def test_returns_string(self):
        result = _header("TEST")
        assert isinstance(result, str)

    def test_contains_title(self):
        result = _header("MY TITLE")
        assert "MY TITLE" in result

    def test_box_characters_present(self):
        result = _header("TEST")
        assert "╔" in result
        assert "╗" in result
        assert "╚" in result
        assert "╝" in result
        assert "║" in result

    def test_three_lines(self):
        result = _header("TEST")
        lines = result.strip().split("\n")
        assert len(lines) == 3

    def test_top_bottom_lines_same_width(self):
        result = _header("TEST")
        lines = result.strip().split("\n")
        assert len(lines[0]) == len(lines[2])


# ── _style_block ──────────────────────────────────────────────────────────────

class TestStyleBlock:
    def test_none_returns_empty(self):
        assert _style_block(None) == ""

    def test_contains_skill_name(self):
        dna = _make_style_dna()
        result = _style_block(dna)
        assert dna.skill_name in result

    def test_contains_visual_style(self):
        dna = _make_style_dna()
        result = _style_block(dna)
        assert dna.visual_style in result

    def test_contains_palette(self):
        dna = _make_style_dna()
        result = _style_block(dna)
        # first colour of palette
        assert dna.color_palette[0] in result

    def test_empty_palette_shows_dash(self):
        dna = _make_style_dna()
        dna.color_palette = []
        result = _style_block(dna)
        assert "—" in result


# ── _story_block ──────────────────────────────────────────────────────────────

class TestStoryBlock:
    def test_contains_title(self):
        story = {"title": "Desert Clash", "summary": "Two armies meet.", "arc": "rise → fall"}
        result = _story_block(story)
        assert "Desert Clash" in result

    def test_contains_summary(self):
        story = {"title": "X", "summary": "Epic battle unfolds.", "arc": "a → b"}
        result = _story_block(story)
        assert "Epic battle unfolds" in result

    def test_missing_keys_show_dash(self):
        result = _story_block({})
        assert "—" in result


# ── _scenes_block ─────────────────────────────────────────────────────────────

class TestScenesBlock:
    def test_shows_scene_count(self):
        scenes = [_make_scene(n) for n in range(1, 5)]
        result = _scenes_block(scenes)
        assert "4 total" in result

    def test_shows_each_scene_number(self):
        scenes = [_make_scene(n) for n in range(1, 4)]
        result = _scenes_block(scenes)
        for n in range(1, 4):
            assert f"Scene {n}" in result

    def test_shows_act_labels(self):
        scenes = [_make_scene(1, act="HOOK"), _make_scene(2, act="CLIMAX")]
        result = _scenes_block(scenes)
        assert "HOOK" in result
        assert "CLIMAX" in result

    def test_shows_camera_and_lighting(self):
        scene = _make_scene(camera="Drone flyover", lighting="Moonlit shadows")
        result = _scenes_block([scene])
        assert "Drone flyover" in result
        assert "Moonlit shadows" in result


# ── build_story_prompt ────────────────────────────────────────────────────────

class TestBuildStoryPrompt:
    def test_returns_non_empty_string(self):
        dna = _make_style_dna()
        result = build_story_prompt("A lone samurai fights robots", 6, "epic", dna)
        assert isinstance(result, str) and len(result) > 100

    def test_contains_stage_header(self):
        result = build_story_prompt("test idea", 4, None, None)
        assert "STAGE 3" in result

    def test_contains_idea(self):
        idea = "An octopus learns ballet"
        result = build_story_prompt(idea, 4, None, None)
        assert idea in result

    def test_contains_n_scenes(self):
        result = build_story_prompt("idea", 8, None, None)
        assert "8" in result

    def test_contains_duration_calculation(self):
        # 6 scenes × 5 s = 30 s
        result = build_story_prompt("idea", 6, None, None)
        assert "30s" in result or "30 s" in result.lower() or "30" in result

    def test_contains_mood_when_provided(self):
        result = build_story_prompt("idea", 4, "melancholic", None)
        assert "melancholic" in result

    def test_mood_line_fallback_when_none(self):
        result = build_story_prompt("idea", 4, None, None)
        assert "not specified" in result.lower()

    def test_contains_style_when_provided(self):
        dna = _make_style_dna()
        result = build_story_prompt("idea", 4, None, dna)
        assert dna.skill_name in result

    def test_json_schema_stage_tag_present(self):
        result = build_story_prompt("idea", 4, None, None)
        assert '"stage": "story_options"' in result

    def test_respond_with_only_json_instruction(self):
        result = build_story_prompt("idea", 4, None, None)
        assert "RESPOND WITH ONLY THIS JSON" in result

    def test_scene_count_in_rules(self):
        # Rule 7 references n_scenes
        result = build_story_prompt("idea", 5, None, None)
        assert "5" in result

    @pytest.mark.parametrize("n_scenes", [4, 6, 8])
    def test_duration_matches_scenes(self, n_scenes):
        result = build_story_prompt("idea", n_scenes, None, None)
        expected_dur = str(n_scenes * 5)
        assert expected_dur in result


# ── build_character_prompt ────────────────────────────────────────────────────

class TestBuildCharacterPrompt:
    def test_returns_string(self):
        result = build_character_prompt("idea", None, None, None)
        assert isinstance(result, str) and len(result) > 100

    def test_contains_stage_header(self):
        result = build_character_prompt("idea", None, None, None)
        assert "STAGE 4" in result

    def test_contains_idea(self):
        idea = "A quantum physicist discovers time travel"
        result = build_character_prompt(idea, None, None, None)
        assert idea in result

    def test_contains_mood_when_provided(self):
        result = build_character_prompt("idea", None, "eerie", None)
        assert "eerie" in result

    def test_story_section_present_when_story_given(self):
        story = {"title": "Rise", "summary": "A hero rises.", "arc": "up → down"}
        result = build_character_prompt("idea", story, None, None)
        assert "Rise" in result

    def test_story_section_absent_when_none(self):
        result = build_character_prompt("idea", None, None, None)
        assert "SELECTED STORY" not in result

    def test_image_hint_section_when_enabled(self):
        result = build_character_prompt("idea", None, None, None, has_image_hint=True)
        assert "reference image" in result.lower() or "CHARACTER REFERENCE IMAGE" in result

    def test_no_image_hint_when_disabled(self):
        result = build_character_prompt("idea", None, None, None, has_image_hint=False)
        assert "CHARACTER REFERENCE IMAGE" not in result

    def test_json_schema_stage_tag(self):
        result = build_character_prompt("idea", None, None, None)
        assert '"stage": "character_description"' in result

    def test_respond_only_json_instruction(self):
        result = build_character_prompt("idea", None, None, None)
        assert "RESPOND WITH ONLY THIS JSON" in result

    def test_style_injected_when_provided(self):
        dna = _make_style_dna()
        result = build_character_prompt("idea", None, None, dna)
        assert dna.skill_name in result


# ── build_scene_prompts_prompt ────────────────────────────────────────────────

class TestBuildScenePromptsPrompt:
    def _make_scenes(self, n=3):
        acts = ["HOOK", "BUILD", "CLIMAX", "RESOLUTION", "CODA"]
        return [
            _make_scene(i + 1, act=acts[i % len(acts)],
                        desc=f"Unique description for scene {i+1}")
            for i in range(n)
        ]

    def test_returns_string(self):
        dna = _make_style_dna()
        scenes = self._make_scenes(3)
        result = build_scene_prompts_prompt("idea", None, None, dna, scenes)
        assert isinstance(result, str) and len(result) > 200

    def test_contains_stage_header(self):
        dna = _make_style_dna()
        result = build_scene_prompts_prompt("idea", None, None, dna, self._make_scenes(2))
        assert "STAGE 5" in result

    def test_schema_has_correct_prompt_count(self):
        dna = _make_style_dna()
        scenes = self._make_scenes(4)
        result = build_scene_prompts_prompt("idea", None, None, dna, scenes)
        # Schema should list Scene 1 through Scene 4 placeholders
        assert '"Scene 1 visual prompt"' in result
        assert '"Scene 4 visual prompt"' in result

    def test_schema_stage_tag(self):
        dna = _make_style_dna()
        result = build_scene_prompts_prompt("idea", None, None, dna, self._make_scenes(2))
        assert '"stage": "scene_prompts"' in result

    def test_contains_visual_and_video_arrays(self):
        dna = _make_style_dna()
        result = build_scene_prompts_prompt("idea", None, None, dna, self._make_scenes(2))
        assert "visual_prompts" in result
        assert "video_prompts" in result

    def test_character_description_included(self):
        dna = _make_style_dna()
        char = _make_character("tall woman with red braids")
        result = build_scene_prompts_prompt("idea", None, char, dna, self._make_scenes(2))
        assert "tall woman with red braids" in result

    def test_no_character_section_when_none(self):
        dna = _make_style_dna()
        result = build_scene_prompts_prompt("idea", None, None, dna, self._make_scenes(2))
        assert "Character:" not in result or 'Character: ""' in result

    def test_all_scene_descriptions_included(self):
        dna = _make_style_dna()
        scenes = self._make_scenes(3)
        result = build_scene_prompts_prompt("idea", None, None, dna, scenes)
        for s in scenes:
            assert s.description in result

    def test_motion_style_included(self):
        dna = _make_style_dna()
        result = build_scene_prompts_prompt("idea", None, None, dna, self._make_scenes(2))
        assert dna.motion_style in result

    def test_respond_only_json_instruction(self):
        dna = _make_style_dna()
        result = build_scene_prompts_prompt("idea", None, None, dna, self._make_scenes(2))
        assert "RESPOND WITH ONLY THIS JSON" in result

    @pytest.mark.parametrize("n", [2, 4, 6])
    def test_schema_count_matches_scenes(self, n):
        dna = _make_style_dna()
        scenes = self._make_scenes(n)
        result = build_scene_prompts_prompt("idea", None, None, dna, scenes)
        # n visual + n video placeholders
        assert result.count("visual prompt") >= n
        assert result.count("motion prompt") >= n


# ── build_single_scene_prompt ─────────────────────────────────────────────────

class TestBuildSingleScenePrompt:
    def test_returns_string(self):
        scene = _make_scene(3, act="CLIMAX")
        dna   = _make_style_dna()
        result = build_single_scene_prompt("idea", scene, None, dna)
        assert isinstance(result, str) and len(result) > 100

    def test_contains_stage_header(self):
        scene = _make_scene(2)
        result = build_single_scene_prompt("idea", scene, None, None)
        assert "STAGE 5" in result

    def test_contains_scene_number(self):
        scene = _make_scene(7, act="RESOLUTION")
        result = build_single_scene_prompt("idea", scene, None, None)
        assert "7" in result

    def test_contains_scene_description(self):
        scene = _make_scene(1, desc="The dragon swoops over burning ruins")
        result = build_single_scene_prompt("idea", scene, None, None)
        assert "The dragon swoops over burning ruins" in result

    def test_contains_act_label(self):
        scene = _make_scene(2, act="CLIMAX")
        result = build_single_scene_prompt("idea", scene, None, None)
        assert "CLIMAX" in result

    def test_contains_camera(self):
        scene = _make_scene(1, camera="Low-angle tracking shot")
        result = build_single_scene_prompt("idea", scene, None, None)
        assert "Low-angle tracking shot" in result

    def test_contains_lighting(self):
        scene = _make_scene(1, lighting="Neon-lit underbelly")
        result = build_single_scene_prompt("idea", scene, None, None)
        assert "Neon-lit underbelly" in result

    def test_schema_single_array_entries(self):
        scene = _make_scene(3)
        result = build_single_scene_prompt("idea", scene, None, None)
        # Schema should reference scene 3 only
        assert f"scene {scene.scene_number}" in result.lower()

    def test_schema_stage_tag(self):
        scene = _make_scene(1)
        result = build_single_scene_prompt("idea", scene, None, None)
        assert '"stage": "scene_prompts"' in result

    def test_character_injected(self):
        scene = _make_scene(1)
        char  = _make_character("old man with a silver beard and cobalt robes")
        result = build_single_scene_prompt("idea", scene, char, None)
        assert "old man with a silver beard" in result

    def test_respond_only_json_instruction(self):
        scene = _make_scene(1)
        result = build_single_scene_prompt("idea", scene, None, None)
        assert "RESPOND WITH ONLY THIS JSON" in result

    def test_video_prompt_rules_present(self):
        scene = _make_scene(1)
        result = build_single_scene_prompt("idea", scene, None, None)
        assert "motion" in result.lower() or "VIDEO PROMPT" in result

    def test_pacing_words_mentioned(self):
        scene = _make_scene(1)
        result = build_single_scene_prompt("idea", scene, None, None)
        pacing = ["slow", "medium", "fast", "explosive"]
        found = any(p in result.lower() for p in pacing)
        assert found, "At least one pacing word should appear in the video prompt rules"
