"""
Tests for the per-scene FOCUS system — the fix for character-centric video output.

Covers the deterministic, fully-offline mechanical engine in story_generator.py
(no API key, no MCP worker) plus SceneState serialization of the new fields.

Run:
  pytest tests/test_focus.py -v -m "not live"
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, ".")

from pipeline.scene_state import SceneState, VALID_FOCUS
from pipeline.story_generator import (
    generate_scenes_from_story, _offline_options,
    _assign_focus, _enforce_focus_variety, _derive_focus_subject,
    _derive_presence_from_focus, _build_visual_prompt_with_framing,
)
from pipeline.style_inference import Character, infer_style_from_skill_id


CHAR_DESC = "a weathered ranger in a grey cloak, mid-40s, distinctive jagged scar"


def _gen(idea, n_scenes, template="discovery", with_char=True):
    """Generate scenes via the FULLY OFFLINE mechanical path (Claude mocked out)."""
    story = _offline_options(idea, n_scenes, None)[
        {"discovery": 0, "struggle": 1, "change": 2}[template]
    ]
    char = Character.new(CHAR_DESC) if with_char else None
    dna  = infer_style_from_skill_id("cinematic")
    with patch("pipeline.story_generator._claude_visual_prompts_batch", return_value=None), \
         patch("pipeline.story_generator._claude_video_prompts_batch", return_value=None):
        return generate_scenes_from_story(idea, story, char, dna, 42,
                                          duration_seconds=n_scenes * 5)


# ── Focus assignment + variety guarantees ─────────────────────────────────────

class TestFocusAssignment:
    def test_every_scene_has_valid_focus(self):
        for s in _gen("a lone astronaut repairs a derelict space station", 7):
            assert s.focus in VALID_FOCUS

    def test_non_subject_scenes_have_a_subject_noun(self):
        for s in _gen("a man discovers a golden waterfall", 7):
            if s.focus != "subject":
                assert s.focus_subject, f"scene {s.scene_number} ({s.focus}) has no focus_subject"

    @pytest.mark.parametrize("n", [5, 6, 8, 12])
    def test_subject_cap_enforced(self, n):
        """The user's bug: NOT all scenes can be character-focused."""
        scenes = _gen("a robot discovers art in an abandoned gallery", n)
        n_subj = sum(1 for s in scenes if s.focus == "subject")
        cap = max(1, round(n * 0.45))
        assert n_subj <= cap, f"{n_subj}/{n} subject scenes exceeds cap {cap}"

    @pytest.mark.parametrize("n", [6, 8, 12])
    def test_focus_diversity_floor(self, n):
        scenes = _gen("a storm rolls over a fishing village", n)
        assert len({s.focus for s in scenes}) >= 3

    def test_not_all_character_focused(self):
        """Direct guard on the reported complaint."""
        scenes = _gen("Cristiano Ronaldo lifting the world cup trophy", 8)
        assert any(s.focus != "subject" for s in scenes)

    @pytest.mark.parametrize("n", [5, 6, 8, 12])
    def test_subject_floor_for_character_story(self, n):
        """A character story must keep a MINORITY of character scenes — but never
        zero (the opposite-extreme bug)."""
        scenes = _gen("a man discovers a golden waterfall", n, with_char=True)
        n_subj = sum(1 for s in scenes if s.focus == "subject")
        floor = max(1, round(n * 0.30))
        cap   = max(1, round(n * 0.45))
        assert floor <= n_subj <= cap, f"{n_subj}/{n} subject scenes outside [{floor},{cap}]"

    def test_no_character_story_can_be_subject_free(self):
        """With no locked character there is no subject floor."""
        scenes = _gen("a storm over an empty desert", 8, with_char=False)
        # allowed to be entirely non-subject; must still be valid + diverse
        assert all(s.focus in VALID_FOCUS for s in scenes)
        assert len({s.focus for s in scenes}) >= 2

    def test_proper_name_not_used_as_subject_noun(self):
        """A protagonist's proper name must not become a place/object noun."""
        scenes = _gen("Cristiano Ronaldo lifting the world cup trophy", 8)
        for s in scenes:
            assert "cristiano" not in s.focus_subject.lower()
            assert "ronaldo" not in s.focus_subject.lower()

    def test_person_words_not_in_non_subject_subjects(self):
        scenes = _gen("a man discovers a golden waterfall", 10)
        for s in scenes:
            if s.focus != "subject":
                assert not any(w in s.focus_subject.lower().split()
                               for w in ("man", "person", "figure", "woman"))

    def test_deterministic(self):
        a = _gen("a dragon sails through a violent sea", 8)
        b = _gen("a dragon sails through a violent sea", 8)
        assert [s.focus for s in a] == [s.focus for s in b]
        assert [s.focus_subject for s in a] == [s.focus_subject for s in b]


# ── presence is a projection of focus ─────────────────────────────────────────

class TestPresenceProjection:
    def test_subject_focus_is_featured(self):
        assert _derive_presence_from_focus("subject", "MEDIUM SHOT") == "featured"

    def test_environment_focus_is_none(self):
        for f in ("establishing", "object", "detail", "phenomenon"):
            assert _derive_presence_from_focus(f, "CLOSE-UP") == "none"

    def test_secondary_focus_presence_depends_on_shot(self):
        assert _derive_presence_from_focus("secondary", "CLOSE-UP") == "featured"
        assert _derive_presence_from_focus("secondary", "WIDE SHOT") == "background"

    def test_generated_scenes_presence_matches_focus(self):
        for s in _gen("a man discovers a golden waterfall", 8):
            if s.focus == "subject":
                assert s.character_presence == "featured"
            elif s.focus in ("establishing", "object", "detail", "phenomenon"):
                assert s.character_presence == "none"


# ── no character leak into non-subject scenes ─────────────────────────────────

class TestNoCharacterLeak:
    def test_character_absent_from_environment_scenes(self):
        scenes = _gen("a lone astronaut repairs a derelict space station", 10)
        env_scenes = [s for s in scenes
                      if s.focus in ("establishing", "object", "detail", "phenomenon")]
        assert env_scenes, "expected at least one environment-type scene"
        for s in env_scenes:
            assert CHAR_DESC[:20] not in s.visual_prompt, (
                f"character leaked into {s.focus} scene {s.scene_number}: {s.visual_prompt[:160]}"
            )
            assert "no people in frame" in s.visual_prompt

    def test_subject_scenes_still_carry_character(self):
        scenes = _gen("a detective confronts the truth at the climax", 8)
        subj = [s for s in scenes if s.focus == "subject"]
        for s in subj:
            assert CHAR_DESC[:20] in s.visual_prompt

    def test_video_prompt_leads_with_focus_subject(self):
        scenes = _gen("a waterfall crashes through an ancient jungle", 8)
        for s in scenes:
            if s.focus != "subject" and s.focus_subject:
                # the focus subject (or its head noun) should appear in the motion prompt
                head = s.focus_subject.split()[0]
                assert head.lower() in s.video_prompt.lower()


# ── robustness ─────────────────────────────────────────────────────────────────

class TestRobustness:
    def test_short_idea_still_gets_subjects(self):
        scenes = _gen("x", 7)
        for s in scenes:
            if s.focus != "subject":
                assert s.focus_subject  # never empty even with a degenerate idea

    def test_no_leftover_placeholders_in_prompts(self):
        import re
        for s in _gen("a robot discovers art", 8):
            assert not re.search(r"\{[a-z_]+\}", s.visual_prompt), s.visual_prompt
            assert not re.search(r"\{[a-z_]+\}", s.video_prompt), s.video_prompt

    def test_works_with_no_character(self):
        scenes = _gen("neon city at night", 6, with_char=False)
        assert all(s.focus in VALID_FOCUS for s in scenes)


# ── SceneState serialization ──────────────────────────────────────────────────

class TestSerialization:
    def test_focus_survives_roundtrip(self):
        s = SceneState(scene_id="scene_01", scene_number=1, act="HOOK",
                       description="x", focus="object", focus_subject="a brass key")
        d = s.to_dict()
        assert d["focus"] == "object" and d["focus_subject"] == "a brass key"
        r = SceneState.from_dict(d)
        assert r.focus == "object" and r.focus_subject == "a brass key"

    def test_legacy_dict_defaults_focus(self):
        """Projects saved before the focus field load with sane defaults."""
        legacy = {"scene_id": "scene_01", "scene_number": 1, "act": "HOOK",
                  "description": "x"}  # no focus / focus_subject keys
        r = SceneState.from_dict(legacy)
        assert r.focus == "subject"
        assert r.focus_subject == ""

    def test_invalid_focus_coerced_not_raised(self):
        s = SceneState(scene_id="scene_01", scene_number=1, act="HOOK",
                       description="x", focus="garbage")
        assert s.focus == "subject"   # coerced, no exception
