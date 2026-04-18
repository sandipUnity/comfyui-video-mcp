"""
Tests for the AI-driven prompt generation improvements.

Covers:
  - skills_engine.build_comfyui_video_prompt()
  - story_generator._claude_visual_prompts_batch()  (mocked)
  - story_generator._claude_video_prompts_batch()   (mocked)
  - story_generator.generate_scenes_from_story() — Claude path + fallback

Unit tests (no ComfyUI, no API key needed):
    python -m pytest tests/test_prompt_generation.py -v -m "not live"
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_skill():
    from skills_engine import SKILLS
    return SKILLS["cinematic"]


def _make_style_dna(skill_id="cinematic"):
    from pipeline import infer_style_from_skill_id
    return infer_style_from_skill_id(skill_id)


def _make_character():
    from pipeline.style_inference import Character
    return Character.new(
        "Tall woman in gold armour, mid-30s, hawk-like eyes, moves with precision"
    )


def _make_story(n=3):
    from pipeline.story_generator import _offline_options
    return _offline_options("a robot discovers art", n, None)[0]


def _make_project(tmp_path):
    from pipeline.project_state import ProjectState
    p = ProjectState.new("test", "a robot discovers art", 15, None)
    p.output_dir = str(tmp_path)
    return p


# ══════════════════════════════════════════════════════════════════════════════
# build_comfyui_video_prompt
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildComfyuiVideoPrompt:

    def test_includes_base_prompt(self):
        from skills_engine import build_comfyui_video_prompt
        skill = _make_skill()
        result = build_comfyui_video_prompt("robot walks forward", skill)
        assert "robot walks forward" in result

    def test_includes_camera_move(self):
        from skills_engine import build_comfyui_video_prompt
        skill = _make_skill()
        result = build_comfyui_video_prompt(
            "robot walks forward", skill, cam="3 ft/s dolly forward tracking subject"
        )
        assert "3 ft/s dolly forward tracking subject" in result

    def test_includes_motion_style(self):
        from skills_engine import build_comfyui_video_prompt
        skill = _make_skill()
        result = build_comfyui_video_prompt(
            "subject stands still", skill,
            motion_style="smooth, intentional camera work with dramatic reveals"
        )
        assert "smooth, intentional camera work" in result

    def test_includes_style_tags(self):
        from skills_engine import build_comfyui_video_prompt
        skill = _make_skill()
        result = build_comfyui_video_prompt("subject moves", skill)
        # cinematic skill has "cinematic" and "photorealistic" as first style_tags
        assert skill.style_tags[0] in result

    def test_empty_cam_not_added(self):
        from skills_engine import build_comfyui_video_prompt
        skill = _make_skill()
        result = build_comfyui_video_prompt("subject moves", skill, cam="")
        parts = [p.strip() for p in result.split(",")]
        # No empty parts
        assert all(p for p in parts)

    def test_returns_string(self):
        from skills_engine import build_comfyui_video_prompt
        skill = _make_skill()
        result = build_comfyui_video_prompt("anything", skill)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_all_skills_work(self):
        from skills_engine import build_comfyui_video_prompt, SKILLS
        for skill in SKILLS.values():
            result = build_comfyui_video_prompt("test scene", skill, cam="pan left", motion_style="slow")
            assert isinstance(result, str)
            assert "test scene" in result

    def test_video_prompt_shorter_than_visual(self):
        """Video prompt should not include the long quality-booster list."""
        from skills_engine import build_comfyui_positive, build_comfyui_video_prompt
        skill = _make_skill()
        base = "robot stands in gallery"
        visual = build_comfyui_positive(base, skill)
        video  = build_comfyui_video_prompt(base, skill)
        assert len(video) < len(visual)


# ══════════════════════════════════════════════════════════════════════════════
# _claude_visual_prompts_batch
# ══════════════════════════════════════════════════════════════════════════════

class TestClaudeVisualPromptsBatch:

    def _scene_inputs(self, n=3):
        skill = _make_skill()
        return [
            {
                "act": "HOOK",
                "description": f"Scene {i+1}: robot explores art gallery",
                "camera": skill.camera_vocabulary[i % len(skill.camera_vocabulary)],
                "lighting": skill.lighting_vocabulary[i % len(skill.lighting_vocabulary)],
            }
            for i in range(n)
        ]

    def test_returns_none_without_api_key(self):
        from pipeline.story_generator import _claude_visual_prompts_batch
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            result = _claude_visual_prompts_batch(
                self._scene_inputs(), _make_skill(), "a character", "an idea"
            )
        assert result is None

    def test_returns_list_of_strings_when_claude_succeeds(self):
        from pipeline.story_generator import _claude_visual_prompts_batch

        fake_prompts = [
            "Robot stands amid paintings, dolly forward, golden hour light, cinematic",
            "Robot reaches toward canvas, extreme close-up, rim lighting, cinematic",
            "Robot and curator face each other, wide shot, overcast soft box, cinematic",
        ]
        fake_response = MagicMock()
        fake_response.content = [MagicMock(text=json.dumps(fake_prompts))]

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value.messages.create.return_value = fake_response

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}), \
             patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            result = _claude_visual_prompts_batch(
                self._scene_inputs(3), _make_skill(), "a robot", "robot discovers art"
            )

        assert result == fake_prompts
        assert len(result) == 3

    def test_uses_skill_prompt_template_as_system(self):
        """Verify skill.prompt_template is passed as the system argument."""
        from pipeline.story_generator import _claude_visual_prompts_batch
        skill = _make_skill()

        fake_response = MagicMock()
        fake_response.content = [MagicMock(text=json.dumps(["prompt"] * 2))]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = fake_response
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}), \
             patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            _claude_visual_prompts_batch(
                self._scene_inputs(2), skill, "character", "idea"
            )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == skill.prompt_template

    def test_returns_none_on_wrong_count(self):
        """If Claude returns wrong number of prompts, fall back gracefully."""
        from pipeline.story_generator import _claude_visual_prompts_batch

        # Returns 2 prompts for 3 scenes
        fake_response = MagicMock()
        fake_response.content = [MagicMock(text=json.dumps(["p1", "p2"]))]
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value.messages.create.return_value = fake_response

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}), \
             patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            result = _claude_visual_prompts_batch(
                self._scene_inputs(3), _make_skill(), "char", "idea"
            )

        assert result is None

    def test_returns_none_on_exception(self):
        from pipeline.story_generator import _claude_visual_prompts_batch
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.side_effect = RuntimeError("API error")

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}), \
             patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            result = _claude_visual_prompts_batch(
                self._scene_inputs(2), _make_skill(), "char", "idea"
            )

        assert result is None

    def test_returns_none_on_invalid_json(self):
        from pipeline.story_generator import _claude_visual_prompts_batch

        fake_response = MagicMock()
        fake_response.content = [MagicMock(text="not json at all")]
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value.messages.create.return_value = fake_response

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}), \
             patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            result = _claude_visual_prompts_batch(
                self._scene_inputs(2), _make_skill(), "char", "idea"
            )

        assert result is None


# ══════════════════════════════════════════════════════════════════════════════
# _claude_video_prompts_batch
# ══════════════════════════════════════════════════════════════════════════════

class TestClaudeVideoPromptsBatch:

    def _scene_inputs(self, n=3):
        skill = _make_skill()
        return [
            {
                "act": "HOOK",
                "description": f"Scene {i+1}: action",
                "camera": skill.camera_vocabulary[i % len(skill.camera_vocabulary)],
            }
            for i in range(n)
        ]

    def test_returns_none_without_api_key(self):
        from pipeline.story_generator import _claude_video_prompts_batch
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            result = _claude_video_prompts_batch(
                self._scene_inputs(), _make_skill(), "", "smooth motion", "idea"
            )
        assert result is None

    def test_returns_list_when_claude_succeeds(self):
        from pipeline.story_generator import _claude_video_prompts_batch

        fake_prompts = [
            "Robot walks three paces forward, camera dollies in 2ft. [medium]",
            "Subject raises arm slowly, camera tilts up 15 degrees. [slow]",
            "Both figures turn toward each other, camera pans right. [medium]",
        ]
        fake_response = MagicMock()
        fake_response.content = [MagicMock(text=json.dumps(fake_prompts))]
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value.messages.create.return_value = fake_response

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}), \
             patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            result = _claude_video_prompts_batch(
                self._scene_inputs(3), _make_skill(),
                "a robot", "smooth motion", "robot discovers art"
            )

        assert result == fake_prompts

    def test_uses_video_system_prompt(self):
        """_VIDEO_PROMPT_SYSTEM (not skill.prompt_template) used as system."""
        from pipeline.story_generator import _claude_video_prompts_batch, _VIDEO_PROMPT_SYSTEM

        fake_response = MagicMock()
        fake_response.content = [MagicMock(text=json.dumps(["motion"] * 2))]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = fake_response
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}), \
             patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            _claude_video_prompts_batch(
                self._scene_inputs(2), _make_skill(), "char", "smooth motion", "idea"
            )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == _VIDEO_PROMPT_SYSTEM

    def test_returns_none_on_wrong_count(self):
        from pipeline.story_generator import _claude_video_prompts_batch

        fake_response = MagicMock()
        fake_response.content = [MagicMock(text=json.dumps(["only one"]))]
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value.messages.create.return_value = fake_response

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}), \
             patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            result = _claude_video_prompts_batch(
                self._scene_inputs(3), _make_skill(), "", "motion", "idea"
            )

        assert result is None


# ══════════════════════════════════════════════════════════════════════════════
# generate_scenes_from_story — full integration
# ══════════════════════════════════════════════════════════════════════════════

class TestGenerateScenesPromptIntegration:

    def _run(self, mock_visual=None, mock_video=None, n=3):
        """Run generate_scenes_from_story with optional Claude mock returns."""
        from pipeline.story_generator import generate_scenes_from_story

        story    = _make_story(n)
        char     = _make_character()
        dna      = _make_style_dna("cinematic")

        with patch("pipeline.story_generator._claude_visual_prompts_batch",
                   return_value=mock_visual), \
             patch("pipeline.story_generator._claude_video_prompts_batch",
                   return_value=mock_video):
            return generate_scenes_from_story(
                "a robot discovers art", story, char, dna,
                global_seed=42, duration_seconds=n * 5,
            )

    # ── Mechanical fallback (Claude returns None) ──────────────────────────────

    def test_fallback_visual_includes_skill_boosters(self):
        scenes = self._run(mock_visual=None, mock_video=None)
        skill = _make_skill()
        for scene in scenes:
            # At least one quality booster should appear
            assert any(b in scene.visual_prompt for b in skill.quality_boosters[:3])

    def test_fallback_visual_includes_character(self):
        scenes = self._run(mock_visual=None, mock_video=None)
        for scene in scenes:
            assert "gold armour" in scene.visual_prompt

    def test_fallback_visual_scene_desc_comes_before_character(self):
        """Scene description must appear before the character prefix so the
        text-area shows something unique at the top for every scene."""
        scenes = self._run(mock_visual=None, mock_video=None)
        char_desc = _make_character().description
        for scene in scenes:
            desc_pos = scene.visual_prompt.find(scene.description[:20])
            char_pos = scene.visual_prompt.find(char_desc[:20])
            assert desc_pos != -1, "scene description missing from visual_prompt"
            assert char_pos  != -1, "character description missing from visual_prompt"
            assert desc_pos < char_pos, (
                f"Scene desc should come before character in visual_prompt.\n"
                f"  desc pos={desc_pos}, char pos={char_pos}\n"
                f"  prompt={scene.visual_prompt[:120]}"
            )

    def test_all_scene_visual_prompts_start_differently(self):
        """The first 30 chars of each visual_prompt must differ across scenes."""
        scenes = self._run(mock_visual=None, mock_video=None, n=4)
        openings = [s.visual_prompt[:30] for s in scenes]
        # All openings should be unique
        assert len(set(openings)) == len(openings), (
            f"Some scenes share the same opening in their visual_prompt:\n"
            + "\n".join(f"  S{i+1}: {o}" for i, o in enumerate(openings))
        )

    def test_fallback_video_includes_camera_move(self):
        scenes = self._run(mock_visual=None, mock_video=None)
        skill = _make_skill()
        for scene in scenes:
            # Camera move from vocabulary should be in the video prompt
            assert any(cam in scene.video_prompt for cam in skill.camera_vocabulary)

    def test_fallback_video_includes_motion_style(self):
        scenes = self._run(mock_visual=None, mock_video=None)
        dna = _make_style_dna("cinematic")
        for scene in scenes:
            assert dna.motion_style in scene.video_prompt

    def test_fallback_video_different_from_just_description(self):
        """Mechanical video prompt must contain more than the bare description."""
        scenes = self._run(mock_visual=None, mock_video=None)
        for scene in scenes:
            # Before the fix, video_prompt == description; now it's richer
            assert scene.video_prompt != scene.description

    # ── Claude path ────────────────────────────────────────────────────────────

    def test_claude_visual_replaces_mechanical(self):
        claude_prompts = [f"Claude visual {i}" for i in range(3)]
        scenes = self._run(mock_visual=claude_prompts, mock_video=None)
        for scene, expected in zip(scenes, claude_prompts):
            assert scene.visual_prompt == expected

    def test_claude_video_replaces_mechanical(self):
        claude_prompts = [f"Claude motion {i}" for i in range(3)]
        scenes = self._run(mock_visual=None, mock_video=claude_prompts)
        for scene, expected in zip(scenes, claude_prompts):
            assert scene.video_prompt == expected

    def test_both_claude_paths_together(self):
        vis = [f"v{i}" for i in range(3)]
        vid = [f"m{i}" for i in range(3)]
        scenes = self._run(mock_visual=vis, mock_video=vid)
        for scene, v, m in zip(scenes, vis, vid):
            assert scene.visual_prompt == v
            assert scene.video_prompt  == m

    def test_partial_claude_failure_uses_fallback_for_that_part(self):
        """Visual Claude fails → mechanical visual; video Claude works → used."""
        vid = [f"motion {i}" for i in range(3)]
        scenes = self._run(mock_visual=None, mock_video=vid)
        skill = _make_skill()
        for scene, m in zip(scenes, vid):
            # Video: Claude version
            assert scene.video_prompt == m
            # Visual: mechanical fallback (contains quality boosters)
            assert any(b in scene.visual_prompt for b in skill.quality_boosters[:3])

    # ── Structural invariants ──────────────────────────────────────────────────

    def test_correct_scene_count(self):
        scenes = self._run()
        assert len(scenes) == 3

    def test_scene_ids_sequential(self):
        scenes = self._run(n=4)
        assert [s.scene_id for s in scenes] == ["scene_01", "scene_02", "scene_03", "scene_04"]

    def test_seeds_deterministic(self):
        a = self._run()
        b = self._run()
        assert [s.seed for s in a] == [s.seed for s in b]

    def test_negative_prompt_always_populated(self):
        scenes = self._run()
        for scene in scenes:
            assert scene.negative_prompt
            assert "ugly" in scene.negative_prompt

    def test_visual_and_video_prompts_are_different(self):
        """They serve different purposes and should never be identical."""
        scenes = self._run()
        for scene in scenes:
            assert scene.visual_prompt != scene.video_prompt
