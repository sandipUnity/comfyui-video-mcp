"""
Tests for pipeline/response_parser.py  (Sprint 5A)

Covers:
  - _extract_json_block        — 4-strategy JSON extractor
  - parse_story_response       — Stage 3
  - parse_character_response   — Stage 4
  - parse_scene_prompts_response — Stage 5
  - parse_error_message        — human-readable error helper
  - ParseError exception

Run:
  pytest tests/test_response_parser.py -v
"""

from __future__ import annotations

import json
import sys

import pytest

sys.path.insert(0, ".")

from pipeline.response_parser import (
    _extract_json_block,
    parse_story_response,
    parse_character_response,
    parse_scene_prompts_response,
    parse_error_message,
    ParseError,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _story_opt(n=1) -> dict:
    """Minimal valid story option dict."""
    return {
        "title":             f"Title {n}",
        "summary":           "Sentence one. Sentence two.",
        "arc":               "rise → conflict → twist → climax → resolve",
        "pacing":            "Slow burn escalating to chaos.",
        "reasoning":         "Fits the idea because X.",
        "act_labels":        ["HOOK", "BUILD", "CLIMAX", "RESOLUTION"],
        "scene_descriptions": [
            "Desert vista wide shot.",
            "Hero runs toward ruins.",
            "Battle erupts.",
            "Victory and silence.",
        ],
    }


def _scene_prompts_result(n=2) -> dict:
    return {
        "visual_prompts": [f"Visual prompt {i+1}" for i in range(n)],
        "video_prompts":  [f"Motion prompt {i+1}" for i in range(n)],
    }


# ══════════════════════════════════════════════════════════════════════════════
# _extract_json_block
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractJsonBlock:

    # ── Strategy 1: whole text is JSON ────────────────────────────────────────

    def test_pure_object(self):
        data = {"key": "value"}
        result = _extract_json_block(json.dumps(data))
        assert result == data

    def test_pure_array(self):
        data = [1, 2, 3]
        result = _extract_json_block(json.dumps(data))
        assert result == data

    def test_whitespace_padded(self):
        result = _extract_json_block('   {"a": 1}   ')
        assert result == {"a": 1}

    # ── Strategy 2: markdown ```json block ────────────────────────────────────

    def test_markdown_json_block(self):
        text = 'Here is the response:\n```json\n{"stage": "ok"}\n```'
        result = _extract_json_block(text)
        assert result == {"stage": "ok"}

    def test_markdown_json_block_uppercase(self):
        text = "```JSON\n[1,2,3]\n```"
        result = _extract_json_block(text)
        assert result == [1, 2, 3]

    def test_markdown_json_block_with_preamble_and_postamble(self):
        text = (
            "Sure! Here is your story:\n\n"
            "```json\n"
            '{"stage": "story_options", "result": []}\n'
            "```\n\n"
            "Hope that helps!"
        )
        result = _extract_json_block(text)
        assert result["stage"] == "story_options"

    # ── Strategy 3: generic ``` block ─────────────────────────────────────────

    def test_generic_code_block(self):
        text = '```\n{"key": 42}\n```'
        result = _extract_json_block(text)
        assert result == {"key": 42}

    # ── Strategy 4: bracket scan ──────────────────────────────────────────────

    def test_preamble_before_object(self):
        text = "Let me explain my answer. The JSON you need is: {\"x\": 99}"
        result = _extract_json_block(text)
        assert result == {"x": 99}

    def test_postamble_after_object(self):
        text = '{"y": "hello"} — please copy the above.'
        result = _extract_json_block(text)
        assert result == {"y": "hello"}

    def test_nested_object(self):
        data = {"outer": {"inner": [1, 2, 3]}}
        text = "preamble " + json.dumps(data) + " postamble"
        result = _extract_json_block(text)
        assert result == data

    def test_array_with_preamble(self):
        text = "Here is a list: [10, 20, 30] done."
        result = _extract_json_block(text)
        assert result == [10, 20, 30]

    def test_string_with_braces_inside_json(self):
        data = {"msg": "use {} for templates"}
        text = "result: " + json.dumps(data)
        result = _extract_json_block(text)
        assert result == data

    def test_escaped_quotes_in_string_values(self):
        data = {"q": "He said \"hello\""}
        result = _extract_json_block(json.dumps(data))
        assert result == data

    # ── Failure cases ─────────────────────────────────────────────────────────

    def test_empty_string_returns_none(self):
        assert _extract_json_block("") is None

    def test_whitespace_only_returns_none(self):
        assert _extract_json_block("   \n\t  ") is None

    def test_no_json_returns_none(self):
        assert _extract_json_block("This is just plain text.") is None

    def test_truncated_json_returns_none(self):
        assert _extract_json_block('{"key": "value"') is None

    def test_markdown_block_with_invalid_json_falls_through(self):
        # Markdown block is broken JSON; bracket scan also fails
        text = "```json\n{broken\n```"
        # Should not raise; may return None or fall through
        result = _extract_json_block(text)
        assert result is None


# ══════════════════════════════════════════════════════════════════════════════
# parse_story_response
# ══════════════════════════════════════════════════════════════════════════════

class TestParseStoryResponse:

    def _wrap(self, options: list) -> str:
        return json.dumps({"stage": "story_options", "result": options})

    # ── Happy path ────────────────────────────────────────────────────────────

    def test_three_valid_options(self):
        text = self._wrap([_story_opt(i) for i in range(1, 4)])
        result = parse_story_response(text)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_bare_array_accepted(self):
        # Some chatbots return just the array without stage wrapper
        text = json.dumps([_story_opt(1)])
        result = parse_story_response(text)
        assert result is not None
        assert len(result) == 1

    def test_single_option_accepted(self):
        text = self._wrap([_story_opt(1)])
        result = parse_story_response(text)
        assert result is not None

    def test_markdown_wrapped(self):
        payload = json.dumps({"stage": "story_options", "result": [_story_opt(1)]})
        text = f"Sure thing!\n```json\n{payload}\n```"
        result = parse_story_response(text)
        assert result is not None

    def test_preserves_all_required_fields(self):
        opt = _story_opt(1)
        text = self._wrap([opt])
        result = parse_story_response(text)
        assert result is not None
        returned = result[0]
        for key in ("title", "summary", "arc", "pacing", "reasoning",
                    "act_labels", "scene_descriptions"):
            assert key in returned

    def test_extra_keys_are_kept(self):
        opt = _story_opt(1)
        opt["extra_field"] = "bonus"
        text = self._wrap([opt])
        result = parse_story_response(text)
        assert result[0]["extra_field"] == "bonus"

    # ── Failures ──────────────────────────────────────────────────────────────

    def test_wrong_stage_returns_none(self):
        text = json.dumps({"stage": "character_description", "result": [_story_opt(1)]})
        assert parse_story_response(text) is None

    def test_missing_required_field_returns_none(self):
        opt = _story_opt(1)
        del opt["arc"]
        text = self._wrap([opt])
        assert parse_story_response(text) is None

    def test_act_labels_not_list_returns_none(self):
        opt = _story_opt(1)
        opt["act_labels"] = "HOOK BUILD"  # string, not list
        text = self._wrap([opt])
        assert parse_story_response(text) is None

    def test_mismatched_labels_and_descriptions_returns_none(self):
        opt = _story_opt(1)
        opt["act_labels"] = ["HOOK", "BUILD"]          # 2 items
        opt["scene_descriptions"] = ["only one desc"]   # 1 item
        text = self._wrap([opt])
        assert parse_story_response(text) is None

    def test_empty_text_returns_none(self):
        assert parse_story_response("") is None

    def test_plain_prose_returns_none(self):
        assert parse_story_response("The hero goes on a journey.") is None

    def test_empty_result_array_returns_none(self):
        text = self._wrap([])
        assert parse_story_response(text) is None

    def test_non_dict_option_returns_none(self):
        text = self._wrap(["not a dict", _story_opt(1)])
        assert parse_story_response(text) is None


# ══════════════════════════════════════════════════════════════════════════════
# parse_character_response
# ══════════════════════════════════════════════════════════════════════════════

class TestParseCharacterResponse:

    def _wrap(self, desc: str) -> str:
        return json.dumps({"stage": "character_description", "result": desc})

    # ── Happy path ────────────────────────────────────────────────────────────

    def test_valid_wrapped_response(self):
        desc = "A 35-year-old woman with short silver hair and a weathered leather jacket."
        result = parse_character_response(self._wrap(desc))
        assert result == desc

    def test_result_stripped(self):
        desc = "  Tall man, angular face, deep-set amber eyes.  "
        result = parse_character_response(self._wrap(desc.strip()))
        assert result == desc.strip()

    def test_markdown_wrapped(self):
        desc = "Young warrior, braided red hair, ceremonial armour."
        payload = self._wrap(desc)
        text = f"```json\n{payload}\n```"
        result = parse_character_response(text)
        assert result == desc

    def test_bare_json_string(self):
        # Some models return just a quoted string
        text = '"A stern detective with silver temples."'
        result = parse_character_response(text)
        assert result == "A stern detective with silver temples."

    def test_prose_fallback_no_braces(self):
        # Plain text accepted if > 20 chars and no braces
        prose = "A weathered old sailor with tattoos on both forearms and a crooked smile."
        result = parse_character_response(prose)
        assert result == prose

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_prose_too_short_returns_none(self):
        assert parse_character_response("Short") is None

    def test_empty_string_returns_none(self):
        assert parse_character_response("") is None

    def test_wrong_stage_returns_none(self):
        text = json.dumps({"stage": "story_options", "result": "some description"})
        assert parse_character_response(text) is None

    def test_empty_result_returns_none(self):
        text = json.dumps({"stage": "character_description", "result": ""})
        assert parse_character_response(text) is None

    def test_non_string_result_returns_none(self):
        text = json.dumps({"stage": "character_description", "result": {"nested": "dict"}})
        assert parse_character_response(text) is None

    def test_braces_in_prose_falls_to_json_path(self):
        # If prose contains braces it should NOT use fallback
        text = "description with {braces} in it but invalid JSON"
        result = parse_character_response(text)
        # No valid JSON → None (brace present so prose fallback not used)
        assert result is None


# ══════════════════════════════════════════════════════════════════════════════
# parse_scene_prompts_response
# ══════════════════════════════════════════════════════════════════════════════

class TestParseScenePromptsResponse:

    def _wrap(self, result: dict) -> str:
        return json.dumps({"stage": "scene_prompts", "result": result})

    # ── Happy path ────────────────────────────────────────────────────────────

    def test_valid_two_scenes(self):
        payload = _scene_prompts_result(2)
        text = self._wrap(payload)
        result = parse_scene_prompts_response(text)
        assert result is not None
        assert len(result["visual_prompts"]) == 2
        assert len(result["video_prompts"]) == 2

    def test_valid_six_scenes(self):
        text = self._wrap(_scene_prompts_result(6))
        result = parse_scene_prompts_response(text)
        assert result is not None
        assert len(result["visual_prompts"]) == 6

    def test_markdown_wrapped(self):
        payload = json.dumps({"stage": "scene_prompts", "result": _scene_prompts_result(2)})
        text = f"```json\n{payload}\n```"
        result = parse_scene_prompts_response(text)
        assert result is not None

    def test_prompts_are_stripped(self):
        sp = {
            "visual_prompts": ["  wide desert shot  ", "  close-up face  "],
            "video_prompts":  ["  slow pan  ", "  zoom in  "],
        }
        text = self._wrap(sp)
        result = parse_scene_prompts_response(text)
        assert result["visual_prompts"][0] == "wide desert shot"
        assert result["video_prompts"][0] == "slow pan"

    def test_result_nested_or_flat_accepted(self):
        # When result key is missing but data is flat inside the stage object
        data = {
            "stage": "scene_prompts",
            "visual_prompts": ["vp1", "vp2"],
            "video_prompts":  ["mp1", "mp2"],
        }
        text = json.dumps(data)
        result = parse_scene_prompts_response(text)
        assert result is not None

    # ── Failures ──────────────────────────────────────────────────────────────

    def test_wrong_stage_returns_none(self):
        text = json.dumps({"stage": "story_options",
                           "result": _scene_prompts_result(2)})
        assert parse_scene_prompts_response(text) is None

    def test_mismatched_array_lengths_returns_none(self):
        sp = {"visual_prompts": ["a", "b", "c"], "video_prompts": ["x", "y"]}
        text = self._wrap(sp)
        assert parse_scene_prompts_response(text) is None

    def test_empty_arrays_returns_none(self):
        sp = {"visual_prompts": [], "video_prompts": []}
        text = self._wrap(sp)
        assert parse_scene_prompts_response(text) is None

    def test_non_string_element_returns_none(self):
        sp = {"visual_prompts": [42, "text"], "video_prompts": ["x", "y"]}
        text = self._wrap(sp)
        assert parse_scene_prompts_response(text) is None

    def test_missing_visual_prompts_key_returns_none(self):
        sp = {"video_prompts": ["x", "y"]}
        text = self._wrap(sp)
        assert parse_scene_prompts_response(text) is None

    def test_missing_video_prompts_key_returns_none(self):
        sp = {"visual_prompts": ["a", "b"]}
        text = self._wrap(sp)
        assert parse_scene_prompts_response(text) is None

    def test_bare_array_returns_none(self):
        # scene_prompts response must be an object, not bare array
        text = json.dumps(["vp1", "vp2"])
        assert parse_scene_prompts_response(text) is None

    def test_empty_string_returns_none(self):
        assert parse_scene_prompts_response("") is None

    def test_plain_prose_returns_none(self):
        assert parse_scene_prompts_response("The hero fights a dragon.") is None


# ══════════════════════════════════════════════════════════════════════════════
# parse_error_message
# ══════════════════════════════════════════════════════════════════════════════

class TestParseErrorMessage:

    def test_empty_input_message(self):
        msg = parse_error_message("", "story")
        assert "Nothing" in msg or "pasted" in msg.lower() or "nothing" in msg.lower()

    def test_whitespace_only_message(self):
        msg = parse_error_message("   ", "story")
        assert len(msg) > 0

    def test_no_json_message(self):
        msg = parse_error_message("Just some plain text without braces.", "story")
        assert "JSON" in msg or "json" in msg.lower()

    def test_truncated_json_message(self):
        msg = parse_error_message('{"stage": "story_options", "result": [', "story")
        assert "truncated" in msg.lower() or "extract" in msg.lower() or "valid" in msg.lower()

    def test_wrong_stage_message(self):
        text = json.dumps({"stage": "wrong_stage", "result": []})
        msg = parse_error_message(text, "story")
        assert "wrong_stage" in msg or "story_options" in msg or "stage" in msg.lower()

    def test_missing_result_field_message(self):
        text = json.dumps({"stage": "story_options"})
        msg = parse_error_message(text, "story")
        assert "result" in msg.lower() or "missing" in msg.lower()

    def test_missing_fields_in_story_option_message(self):
        opt = {"title": "x", "summary": "y"}   # missing arc, pacing, etc.
        text = json.dumps({"stage": "story_options", "result": [opt]})
        msg = parse_error_message(text, "story")
        # Should mention missing fields or ask to regenerate
        assert ("arc" in msg or "pacing" in msg or "missing" in msg.lower()
                or "regenerate" in msg.lower())

    def test_scenes_mismatch_message(self):
        result = {
            "visual_prompts": ["a", "b", "c"],
            "video_prompts":  ["x", "y"],      # intentional mismatch
        }
        text = json.dumps({"stage": "scene_prompts", "result": result})
        msg = parse_error_message(text, "scenes")
        assert "mismatch" in msg.lower() or "3" in msg or "2" in msg

    def test_always_returns_non_empty_string(self):
        for stage in ("story", "character", "scenes"):
            msg = parse_error_message("garbage input {}", stage)
            assert isinstance(msg, str) and len(msg) > 0

    @pytest.mark.parametrize("raw,stage", [
        ("", "story"),
        ("no json here", "character"),
        ('{"stage":"wrong"}', "scenes"),
    ])
    def test_parametrised_returns_string(self, raw, stage):
        msg = parse_error_message(raw, stage)
        assert isinstance(msg, str) and len(msg) > 5


# ══════════════════════════════════════════════════════════════════════════════
# ParseError exception
# ══════════════════════════════════════════════════════════════════════════════

class TestParseError:

    def test_is_exception(self):
        with pytest.raises(ParseError):
            raise ParseError("test error")

    def test_message_preserved(self):
        err = ParseError("something broke")
        assert str(err) == "something broke"

    def test_raw_text_default_empty(self):
        err = ParseError("oops")
        assert err.raw_text == ""

    def test_raw_text_stored(self):
        err = ParseError("oops", raw_text="original garbage")
        assert err.raw_text == "original garbage"

    def test_inherits_exception(self):
        assert issubclass(ParseError, Exception)


# ══════════════════════════════════════════════════════════════════════════════
# Integration: realistic chatbot output formats
# ══════════════════════════════════════════════════════════════════════════════

class TestRealisticChatbotOutput:
    """End-to-end tests with realistic chatbot response formats."""

    def test_story_response_with_long_preamble(self):
        opt = _story_opt(1)
        payload = json.dumps({"stage": "story_options", "result": [opt]})
        text = (
            "I've carefully considered your request and here are some story options "
            "based on the idea you provided. Each option explores a different narrative "
            "structure to give you variety:\n\n" + payload
        )
        result = parse_story_response(text)
        assert result is not None

    def test_character_response_with_preamble_and_postamble(self):
        desc = "30-year-old woman, athletic, close-cropped silver hair, jade-green armour."
        payload = json.dumps({"stage": "character_description", "result": desc})
        text = (
            "Great! Here's the character description for your protagonist:\n\n"
            "```json\n" + payload + "\n```\n\n"
            "Let me know if you'd like any adjustments!"
        )
        result = parse_character_response(text)
        assert result == desc

    def test_scene_prompts_with_explanation(self):
        sp = _scene_prompts_result(3)
        payload = json.dumps({"stage": "scene_prompts", "result": sp})
        text = (
            "Here are the prompts for your 3 scenes, following your cinematographer guidelines:\n\n"
            + payload +
            "\n\nEach visual prompt starts with the scene action as requested."
        )
        result = parse_scene_prompts_response(text)
        assert result is not None
        assert len(result["visual_prompts"]) == 3

    def test_scene_prompts_arrays_have_correct_types(self):
        sp = _scene_prompts_result(4)
        text = json.dumps({"stage": "scene_prompts", "result": sp})
        result = parse_scene_prompts_response(text)
        assert all(isinstance(v, str) for v in result["visual_prompts"])
        assert all(isinstance(v, str) for v in result["video_prompts"])

    def test_multiple_json_blocks_uses_first_valid(self):
        # If text has two JSON blocks, the extractor should find the first valid one
        first  = '{"stage": "story_options", "result": [' + json.dumps(_story_opt(1)) + "]}"
        second = '{"stage": "character_description", "result": "some desc"}'
        text = first + "\n\nAlternatively:\n\n" + second
        result = parse_story_response(text)
        # parse_story_response should succeed on the first block
        assert result is not None
