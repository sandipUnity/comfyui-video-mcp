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
import re
from pathlib import Path

_LEFTOVER_TOKEN_RE = re.compile(r"\{\{[A-Z_]+\}\}")


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
    # Additional placeholder defaults, e.g. {"CFG": 5.0, "CHECKPOINT": "x.safetensors"}
    extra: dict | None = None,
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
        extra:            Extra placeholder values keyed by bare token name
                          (no braces), typically from
                          ``workflow_catalog.template_defaults()`` — sampler
                          settings ({{STEPS}}, {{CFG}}) and model files
                          ({{CHECKPOINT}}, {{TEXT_ENCODER}}, {{VAE}}, …) that
                          legacy templates expect. Numeric values are replaced
                          as bare tokens; strings are injected after parsing.

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

    string_extras: dict[str, str] = {}
    for token, value in (extra or {}).items():
        if isinstance(value, bool):
            numeric_map["{{" + token + "}}"] = "true" if value else "false"
        elif isinstance(value, (int, float)):
            numeric_map["{{" + token + "}}"] = str(value)
        elif isinstance(value, str):
            string_extras["{{" + token + "}}"] = value

    for placeholder, value in numeric_map.items():
        template = template.replace(placeholder, value)

    # ── Step 2: Parse ──────────────────────────────────────────────────────────
    wf = json.loads(template)

    # ── Step 3: Walk dict and inject string values ─────────────────────────────
    string_map: dict[str, str] = {
        "{{POSITIVE_PROMPT}}": positive_prompt,
        "{{NEGATIVE_PROMPT}}": negative_prompt,
        "{{OUTPUT_PREFIX}}":   output_prefix,
        **string_extras,
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

    # Guard: a template selected/persisted before it was validated may still
    # carry placeholders the pipeline couldn't fill. Fail loudly here rather
    # than queue a literal "{{CHECKPOINT}}" string to ComfyUI (which produces
    # a confusing server-side error).
    leftover = sorted(set(_LEFTOVER_TOKEN_RE.findall(json.dumps(wf))))
    if leftover:
        raise ValueError(
            "Workflow template still contains unfilled placeholders after fill: "
            + ", ".join(leftover)
            + ". Add the missing values to the matching models: entry in "
            "config.yaml, or choose a different workflow."
        )
    return wf
