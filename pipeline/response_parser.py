"""
Response Parser — extracts and validates JSON from chatbot responses.

Handles all real-world chatbot output formats:
  - Raw JSON only (ideal)
  - JSON preceded by chatbot preamble text
  - JSON inside a markdown code block (```json ... ```)
  - JSON followed by explanation text
  - Nested JSON with extra keys (lenient — only required keys checked)

Public API:
    parse_story_response(text)        → list[dict] | None
    parse_character_response(text)    → str | None
    parse_scene_prompts_response(text)→ dict | None
        returns {"visual_prompts": [...], "video_prompts": [...]}

    parse_error_message(text, stage)  → str
        Human-readable error for display in the UI when parsing fails.
"""

from __future__ import annotations

import json
import re


# ── Exceptions ────────────────────────────────────────────────────────────────

class ParseError(Exception):
    """Raised when a chatbot response cannot be parsed into the expected schema."""
    def __init__(self, message: str, raw_text: str = ""):
        super().__init__(message)
        self.raw_text = raw_text


# ── Core JSON extractor ───────────────────────────────────────────────────────

def _extract_json_block(text: str) -> dict | list | None:
    """Find and parse the first valid JSON object or array in arbitrary text.

    Strategy (tried in order):
    1. Strip and parse the whole text as JSON
    2. Extract from ```json ... ``` markdown block
    3. Extract from ``` ... ``` markdown block
    4. Find first { or [ and match its closing bracket
    """
    if not text or not text.strip():
        return None

    text = text.strip()

    # 1. Try whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Markdown ```json block
    md_match = re.search(r"```json\s*([\s\S]*?)```", text, re.IGNORECASE)
    if md_match:
        try:
            return json.loads(md_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. Any ``` block
    md_match = re.search(r"```\s*([\s\S]*?)```", text)
    if md_match:
        try:
            return json.loads(md_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 4. Find first { or [ and scan for matching close bracket
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape_next = False
        for i, ch in enumerate(text[start:], start=start):
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    candidate = text[start:i+1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    return None


def _check_stage(data: dict, expected: str) -> bool:
    """Return True if data["stage"] matches expected (case-insensitive)."""
    return isinstance(data, dict) and \
           data.get("stage", "").lower() == expected.lower()


# ── Stage 3 — Story Options ───────────────────────────────────────────────────

_STORY_REQUIRED = {"title", "summary", "arc", "pacing", "reasoning",
                   "act_labels", "scene_descriptions"}


def parse_story_response(text: str) -> list[dict] | None:
    """Parse a chatbot response into a list of 3 story option dicts.

    Returns None (not raises) on any failure.
    """
    data = _extract_json_block(text)
    if data is None:
        return None

    # Accept both {"stage": "story_options", "result": [...]} and bare [...]
    if isinstance(data, dict):
        if not _check_stage(data, "story_options"):
            return None
        options = data.get("result")
    elif isinstance(data, list):
        options = data
    else:
        return None

    if not isinstance(options, list) or len(options) < 1:
        return None

    validated = []
    for opt in options:
        if not isinstance(opt, dict):
            return None
        missing = _STORY_REQUIRED - set(opt.keys())
        if missing:
            return None
        if not isinstance(opt["act_labels"], list):
            return None
        if not isinstance(opt["scene_descriptions"], list):
            return None
        if len(opt["act_labels"]) != len(opt["scene_descriptions"]):
            return None
        validated.append(opt)

    return validated if validated else None


# ── Stage 4 — Character Description ──────────────────────────────────────────

def parse_character_response(text: str) -> str | None:
    """Parse a chatbot response into a character description string.

    Returns None on failure.
    """
    data = _extract_json_block(text)

    if data is None:
        # Last resort: if the text looks like a plain prose description
        # (no JSON at all), accept it directly as long as it's > 20 chars
        stripped = text.strip()
        if stripped and len(stripped) > 20 and '{' not in stripped:
            return stripped
        return None

    if isinstance(data, dict):
        if not _check_stage(data, "character_description"):
            return None
        result = data.get("result")
        if isinstance(result, str) and result.strip():
            return result.strip()
        return None

    # Accept bare string JSON: "description text"
    if isinstance(data, str) and data.strip():
        return data.strip()

    return None


# ── Stage 5 — Scene Prompts ───────────────────────────────────────────────────

def parse_scene_prompts_response(text: str) -> dict | None:
    """Parse a chatbot response into visual + video prompt arrays.

    Returns {"visual_prompts": [...], "video_prompts": [...]} or None.
    The two arrays must have the same length (≥ 1).
    """
    data = _extract_json_block(text)
    if data is None:
        return None

    if isinstance(data, dict):
        if not _check_stage(data, "scene_prompts"):
            return None
        result = data.get("result", data)   # accept result nested or flat
    else:
        return None

    if not isinstance(result, dict):
        return None

    visual = result.get("visual_prompts")
    video  = result.get("video_prompts")

    if not isinstance(visual, list) or not isinstance(video, list):
        return None
    if len(visual) == 0 or len(visual) != len(video):
        return None
    if not all(isinstance(p, str) for p in visual + video):
        return None

    return {
        "visual_prompts": [p.strip() for p in visual],
        "video_prompts":  [p.strip() for p in video],
    }


# ── Human-readable error messages ────────────────────────────────────────────

def parse_error_message(text: str, stage: str) -> str:
    """Return a helpful error message when parsing fails."""
    if not text or not text.strip():
        return "Nothing was pasted. Please copy the AI response and paste it here."

    if '{' not in text and '[' not in text:
        return (
            "The response doesn't contain JSON. Make sure you copied the "
            "AI's complete response, including the JSON block at the end."
        )

    data = _extract_json_block(text)
    if data is None:
        return (
            "Could not extract valid JSON from the response. "
            "The JSON may be truncated — try asking the AI to resend "
            "its response, or paste a larger portion of the text."
        )

    stage_map = {
        "story":     "story_options",
        "character": "character_description",
        "scenes":    "scene_prompts",
    }
    expected_stage = stage_map.get(stage, stage)

    if isinstance(data, dict):
        got_stage = data.get("stage", "(missing)")
        if got_stage != expected_stage:
            return (
                f"Wrong stage in response. Expected \"stage\": \"{expected_stage}\" "
                f"but got \"{got_stage}\". "
                "Make sure you're pasting the response for the correct step."
            )

        result = data.get("result")
        if result is None:
            return (
                "The JSON is valid but missing the \"result\" field. "
                "Ask the AI to regenerate its response."
            )

        if stage == "story" and isinstance(result, list):
            for i, opt in enumerate(result):
                if isinstance(opt, dict):
                    missing = _STORY_REQUIRED - set(opt.keys())
                    if missing:
                        return (
                            f"Story option {i+1} is missing fields: "
                            f"{', '.join(sorted(missing))}. "
                            "Ask the AI to regenerate."
                        )

        if stage == "scenes" and isinstance(result, dict):
            vp = result.get("visual_prompts", [])
            mp = result.get("video_prompts", [])
            if len(vp) != len(mp):
                return (
                    f"Mismatch: {len(vp)} visual prompts but {len(mp)} video prompts. "
                    "They must be the same count. Ask the AI to regenerate."
                )

    return (
        "The response could not be parsed. Try copying the AI's full response "
        "again — make sure nothing is cut off at the beginning or end."
    )
