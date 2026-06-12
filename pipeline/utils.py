"""
Shared workflow injection utility.

fill_workflow() — the definitive parse-then-inject pattern.
  Step 1: String-replace bare numeric/bool placeholders (safe — no special chars possible)
  Step 2: json.loads()  — string placeholders remain as valid JSON string values
  Step 3: Walk parsed dict, swap placeholder strings → real Python values in-place

This pattern is immune to:
  - Em-dashes, curly quotes, backslashes in prompts
  - Unicode characters
  - Arbitrarily long prompt text
  - YAML trailing newlines loaded into the template

Never touches prompts with json.dumps() escaping — prompts go straight into the dict.
"""

from __future__ import annotations

import json
from pathlib import Path


def fill_workflow(
    template_path: str | Path,
    *,
    positive_prompt: str,
    negative_prompt: str = "",
    width: int,
    height: int,
    seed: int,
    output_prefix: str,
    # I2V-only params (omit for T2I)
    input_image: str | None = None,
    frames: int | None = None,
    fps: int | None = None,
) -> dict:
    """Load a workflow template and inject all parameters.

    Args:
        template_path:    Path to a ``workflows/*.json`` template file.
        positive_prompt:  Full positive prompt string (any characters allowed).
        negative_prompt:  Full negative prompt string.
        width:            Output width in pixels (int).
        height:           Output height in pixels (int).
        seed:             Random seed (int).
        output_prefix:    Filename prefix for saved output (string).
        input_image:      Server-side filename from upload_image() — I2V only.
        frames:           Number of frames — I2V only.
        fps:              Frames per second — I2V only.

    Returns:
        Filled workflow dict ready to pass to ``ComfyUIClient.queue_prompt()``.
    """
    template = Path(template_path).read_text(encoding="utf-8")

    # ── Step 1: Replace NUMERIC / BOOL placeholders via string replace ─────────
    # These placeholders appear as bare values in the JSON (no quotes around them)
    # so they must be replaced before json.loads().
    numeric_map: dict[str, str] = {
        "{{WIDTH}}":  str(width),
        "{{HEIGHT}}": str(height),
        "{{SEED}}":   str(seed),
    }
    if frames is not None:
        numeric_map["{{FRAMES}}"] = str(frames)
    if fps is not None:
        numeric_map["{{FPS}}"] = str(fps)

    for placeholder, value in numeric_map.items():
        template = template.replace(placeholder, value)

    # ── Step 2: Parse ──────────────────────────────────────────────────────────
    wf = json.loads(template)

    # ── Step 3: Walk dict and inject string values ─────────────────────────────
    string_map: dict[str, str] = {
        "{{POSITIVE_PROMPT}}": positive_prompt,
        "{{NEGATIVE_PROMPT}}": negative_prompt,
        "{{OUTPUT_PREFIX}}":   output_prefix,
    }
    if input_image is not None:
        string_map["{{INPUT_IMAGE}}"] = input_image

    def _inject(node: dict) -> None:
        for key, val in node.items():
            if isinstance(val, dict):
                _inject(val)
            elif isinstance(val, str) and val in string_map:
                node[key] = string_map[val]

    _inject(wf)
    return wf
