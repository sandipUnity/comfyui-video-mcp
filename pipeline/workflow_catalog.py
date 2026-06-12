"""
Workflow catalog — discover and classify local ComfyUI workflow templates.

Scans ``workflows/*.json`` so the UI can offer a picker instead of hardcoding
Flux Schnell / LTX 2.3. Each template is classified by the placeholders it
contains:

    kind = "i2v"  — has {{INPUT_IMAGE}}            (image-to-video)
    kind = "t2v"  — has {{FRAMES}} but no image    (text-to-video)
    kind = "t2i"  — neither                        (text-to-image)

A template is *compatible* with the new pipeline only when every placeholder
it uses is one ``pipeline.utils.fill_workflow()`` knows how to fill. Legacy
templates that expect the old server.py injector ({{CFG}}, {{STEPS}},
{{CHECKPOINT}}, …) are listed but marked incompatible with a reason, so the
UI can grey them out instead of failing at json.loads() on queue.

Paths are stored as forward-slash strings relative to the project root
(e.g. "workflows/flux_schnell_t2i_api.json") on ALL platforms — Windows
backslash paths broke selectbox matching before (see commit 07f0fce).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT  = Path(__file__).parent.parent
WORKFLOWS_DIR = PROJECT_ROOT / "workflows"

# Everything fill_workflow() can inject. A template using anything outside
# this set would survive injection with raw {{TOKENS}} left in place and
# fail to parse / fail on the ComfyUI server.
SUPPORTED_PLACEHOLDERS = frozenset({
    "POSITIVE_PROMPT", "NEGATIVE_PROMPT", "OUTPUT_PREFIX", "INPUT_IMAGE",
    "WIDTH", "HEIGHT", "SEED", "FRAMES", "FPS",
})

_PLACEHOLDER_RE = re.compile(r"\{\{([A-Z_]+)\}\}")


@dataclass
class WorkflowInfo:
    path: str                 # posix-style, relative to project root
    name: str                 # filename stem, e.g. "flux_schnell_t2i_api"
    kind: str                 # "t2i" | "i2v" | "t2v" | "unknown"
    placeholders: list[str] = field(default_factory=list)
    compatible: bool = False
    reason: str = ""          # human-readable reason when not compatible

    @property
    def label(self) -> str:
        """Display label for UI selectboxes."""
        return self.name


def _classify(placeholders: set[str]) -> str:
    if "INPUT_IMAGE" in placeholders:
        return "i2v"
    if "FRAMES" in placeholders:
        return "t2v"
    return "t2i"


def _check_compatibility(raw_text: str, placeholders: set[str]) -> tuple[bool, str]:
    """Return (compatible, reason). Mirrors what fill_workflow() will do."""
    if "POSITIVE_PROMPT" not in placeholders:
        return False, "no {{POSITIVE_PROMPT}} placeholder — prompts cannot be injected"

    unsupported = sorted(placeholders - SUPPORTED_PLACEHOLDERS)
    if unsupported:
        tokens = ", ".join("{{" + p + "}}" for p in unsupported)
        return False, f"uses placeholders the pipeline cannot fill: {tokens}"

    # Dry-run the same numeric substitution fill_workflow() performs, then
    # confirm the result is valid JSON. String placeholders are quoted in the
    # template, so they parse as-is.
    text = raw_text
    for token in ("WIDTH", "HEIGHT", "SEED", "FRAMES", "FPS"):
        text = text.replace("{{" + token + "}}", "1")
    try:
        json.loads(text)
    except json.JSONDecodeError as e:
        return False, f"template is not valid JSON after placeholder fill: {e}"

    return True, ""


def discover_workflows(workflows_dir: str | Path | None = None) -> list[WorkflowInfo]:
    """Scan the workflows directory and return info for every *.json template.

    Results are sorted by name. Invalid/unreadable files are included with
    compatible=False so the UI can surface them rather than hide them.
    """
    wf_dir = Path(workflows_dir) if workflows_dir else WORKFLOWS_DIR
    results: list[WorkflowInfo] = []

    for fp in sorted(wf_dir.glob("*.json")):
        # Relative posix path when under the project root; absolute posix otherwise
        try:
            rel = fp.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            rel = fp.as_posix()

        try:
            raw = fp.read_text(encoding="utf-8")
        except OSError as e:
            results.append(WorkflowInfo(
                path=rel, name=fp.stem, kind="unknown",
                compatible=False, reason=f"unreadable: {e}",
            ))
            continue

        placeholders = set(_PLACEHOLDER_RE.findall(raw))
        kind = _classify(placeholders)
        compatible, reason = _check_compatibility(raw, placeholders)

        results.append(WorkflowInfo(
            path=rel, name=fp.stem, kind=kind,
            placeholders=sorted(placeholders),
            compatible=compatible, reason=reason,
        ))

    return results


def workflows_for(kinds: str | tuple[str, ...],
                  workflows_dir: str | Path | None = None,
                  include_incompatible: bool = False) -> list[WorkflowInfo]:
    """Return workflows of the given kind(s), compatible ones only by default."""
    if isinstance(kinds, str):
        kinds = (kinds,)
    return [
        wf for wf in discover_workflows(workflows_dir)
        if wf.kind in kinds and (include_incompatible or wf.compatible)
    ]


def resolve_workflow_path(path: str | Path) -> Path:
    """Resolve a stored workflow path (usually project-relative posix) to absolute."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
