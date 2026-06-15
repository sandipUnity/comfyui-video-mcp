"""
Tests for the narrative_role / shot_intent / hero_moment storytelling enrichment
(patterns borrowed from OpenMontage scene_plan — concepts only, no code copied)
and the edit_decisions / compile_from_edit_decisions compositing layer.

Run:
  pytest tests/test_narrative_role.py -v -m "not live"
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, ".")

from pipeline.scene_state import SceneState, VALID_NARRATIVE_ROLE
from pipeline.story_generator import (
    generate_scenes_from_story, _offline_options, _assign_narrative_role,
    _HERO_ACTS,
)
from pipeline.style_inference import Character, infer_style_from_skill_id
from pipeline.montage import build_edit_decisions


def _gen(idea, n_scenes=8, template="discovery", with_char=True):
    story = _offline_options(idea, n_scenes, None)[
        {"discovery": 0, "struggle": 1, "change": 2}[template]
    ]
    char = Character.new("an explorer in a worn leather jacket") if with_char else None
    dna  = infer_style_from_skill_id("cinematic")
    with patch("pipeline.story_generator._claude_visual_prompts_batch", return_value=None), \
         patch("pipeline.story_generator._claude_video_prompts_batch", return_value=None):
        return generate_scenes_from_story(idea, story, char, dna, 42,
                                          duration_seconds=n_scenes * 5)


# ── narrative_role assignment ─────────────────────────────────────────────────

class TestNarrativeRole:
    def test_known_acts_map_to_valid_roles(self):
        for act in ("HOOK", "BUILD", "REVELATION", "RESOLUTION", "CLIMAX", "ORDINARY"):
            r = _assign_narrative_role(act)
            assert r in VALID_NARRATIVE_ROLE, f"{act} -> {r!r} not a valid role"

    def test_unknown_act_returns_empty(self):
        assert _assign_narrative_role("NOT_A_REAL_ACT") == ""

    def test_generated_scenes_get_narrative_role(self):
        for s in _gen("a man discovers a golden waterfall"):
            assert s.narrative_role in VALID_NARRATIVE_ROLE or s.narrative_role == ""

    def test_known_act_scenes_have_a_role(self):
        scenes = _gen("a robot discovers art", 6)
        # the discovery story uses only documented acts, so every scene gets a role
        assert all(s.narrative_role for s in scenes)


# ── hero_moment marker ────────────────────────────────────────────────────────

class TestHeroMoment:
    def test_revelation_scene_is_hero(self):
        for s in _gen("a robot discovers art", 6):
            if s.act.upper().strip() in _HERO_ACTS:
                assert s.hero_moment, f"{s.act} should be a hero moment"

    def test_non_hero_acts_are_not_hero(self):
        for s in _gen("a robot discovers art", 6):
            if s.act.upper().strip() not in _HERO_ACTS:
                assert not s.hero_moment

    def test_struggle_template_has_at_least_one_hero(self):
        scenes = _gen("a fighter trains for the final bout", 7, template="struggle")
        assert any(s.hero_moment for s in scenes)


# ── shot_intent default text ──────────────────────────────────────────────────

class TestShotIntent:
    def test_every_scene_has_shot_intent(self):
        for s in _gen("a storm rolls over a fishing village", 8):
            assert s.shot_intent and isinstance(s.shot_intent, str)

    def test_intent_mentions_focus_subject_for_object_scenes(self):
        scenes = _gen("a lone astronaut repairs a derelict space station", 10)
        for s in scenes:
            if s.focus == "object" and s.focus_subject:
                head = s.focus_subject.split()[0].lower()
                assert head in s.shot_intent.lower() or "object" in s.shot_intent.lower()


# ── SceneState serialization round-trip ───────────────────────────────────────

class TestSerialization:
    def test_all_new_fields_survive_roundtrip(self):
        s = SceneState(scene_id="scene_01", scene_number=1, act="REVELATION",
                       description="x", narrative_role="deliver_payload",
                       shot_intent="Reveal the artifact.", hero_moment=True)
        d = s.to_dict()
        assert d["narrative_role"] == "deliver_payload"
        assert d["shot_intent"] == "Reveal the artifact."
        assert d["hero_moment"] is True
        r = SceneState.from_dict(d)
        assert r.narrative_role == "deliver_payload"
        assert r.shot_intent == "Reveal the artifact."
        assert r.hero_moment is True

    def test_legacy_dict_defaults(self):
        legacy = {"scene_id": "scene_01", "scene_number": 1, "act": "HOOK",
                  "description": "x"}
        r = SceneState.from_dict(legacy)
        assert r.narrative_role == ""
        assert r.shot_intent == ""
        assert r.hero_moment is False

    def test_invalid_narrative_role_coerced(self):
        s = SceneState(scene_id="scene_01", scene_number=1, act="HOOK",
                       description="x", narrative_role="garbage_value")
        assert s.narrative_role == ""   # coerced to empty, not raised


# ── prompt_builder surfaces the storytelling fields ───────────────────────────

class TestPromptBuilderSurface:
    def test_role_and_intent_appear_in_scenes_block(self):
        from pipeline.prompt_builder import build_scene_prompts_prompt
        from pipeline import infer_style_from_skill_id as _isfid
        dna = _isfid("cinematic")
        scenes = _gen("a robot discovers art", 4)
        result = build_scene_prompts_prompt("a robot discovers art", None, None, dna, scenes)
        assert "Role:" in result
        assert "Intent:" in result
        # At least one HERO MOMENT marker for the discovery story (REVELATION beat)
        assert "HERO MOMENT" in result


# ── edit_decisions ────────────────────────────────────────────────────────────

class TestEditDecisions:
    def test_hero_scenes_get_extra_hold(self):
        scenes = _gen("a robot discovers art", 6)
        ed = build_edit_decisions(scenes, base_transition="dissolve", hero_hold_extra=0.6)
        by_id = {d["scene_id"]: d for d in ed["scenes"]}
        for s in scenes:
            if s.hero_moment:
                assert by_id[s.scene_id]["extra_hold_seconds"] == 0.6
            else:
                assert by_id[s.scene_id]["extra_hold_seconds"] == 0.0

    def test_payload_and_hero_use_cut_transition(self):
        scenes = _gen("a robot discovers art", 6)
        ed = build_edit_decisions(scenes, base_transition="dissolve")
        # Find a hero or payload scene that's not the first; its transition_in must be a cut
        hero_or_payload = [i for i, s in enumerate(scenes)
                           if (s.hero_moment or s.narrative_role == "deliver_payload")
                           and i > 0]
        assert hero_or_payload, "expected at least one hero/payload scene after scene 1"
        for i in hero_or_payload:
            assert ed["scenes"][i]["transition_in"] == "cut"

    def test_subject_scenes_duck_music(self):
        scenes = _gen("a man discovers a waterfall", 8)
        ed = build_edit_decisions(scenes, base_transition="dissolve")
        for s, d in zip(scenes, ed["scenes"]):
            assert d["duck_music"] is (s.focus == "subject")

    def test_no_music_block_when_no_music(self):
        scenes = _gen("a man discovers a waterfall", 6)
        ed = build_edit_decisions(scenes)
        assert ed["music"] is None

    def test_music_block_present_when_path_given(self):
        scenes = _gen("a man discovers a waterfall", 6)
        ed = build_edit_decisions(scenes, music_path="/tmp/song.mp3", music_duck_db=-10.0)
        assert ed["music"] == {"path": "/tmp/song.mp3", "duck_db": -10.0}

    def test_scene_count_matches_input(self):
        scenes = _gen("a robot discovers art", 7)
        ed = build_edit_decisions(scenes)
        assert len(ed["scenes"]) == 7
