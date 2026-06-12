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


# ── Auto-templating of raw ComfyUI API exports ────────────────────────────────
#
# Lets the user bring ANY workflow from their ComfyUI: export it with
# "Save (API Format)", upload it in the UI, and this turns the hardcoded
# values into {{PLACEHOLDERS}} so the pipeline can drive it.

# Fields that may hold prompt text on a conditioning/primitive node
_PROMPT_FIELDS = ("text", "value", "string", "prompt")
# Integer fields treated as frame count on latent/video nodes
_FRAMES_FIELDS = ("length", "frames", "num_frames", "video_frames")
# Primitive-node _meta titles → numeric tokens (subgraph workflows route
# dimensions through PrimitiveInt nodes titled "Width", "Frame Rate", …)
_TITLE_TOKEN_MAP = {
    "width": "WIDTH", "height": "HEIGHT",
    "length": "FRAMES", "frames": "FRAMES", "frame count": "FRAMES",
    "frame rate": "FPS", "fps": "FPS",
}


def auto_template_workflow(raw_text: str) -> tuple[str, list[str]]:
    """Insert {{PLACEHOLDER}} tokens into a raw API-format workflow export.

    Returns (templated_json_text, notes). Notes describe every substitution
    made and anything that could NOT be located (so the user can hand-edit).

    Raises ValueError for invalid JSON or GUI-format ("nodes" array) files.
    """
    try:
        wf = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Not valid JSON: {e}") from e

    if not isinstance(wf, dict):
        raise ValueError("Unexpected JSON structure — expected an object keyed by node id.")
    if isinstance(wf.get("nodes"), list):
        raise ValueError(
            "This is a GUI-format workflow (nodes array). In ComfyUI use "
            "Workflow → Export (API) / 'Save (API Format)' and upload that file instead."
        )

    notes: list[str] = []

    def _follow_to_text(link, depth: int = 0):
        """Follow a ["node_id", slot] link to a node holding editable prompt text."""
        if depth > 4 or not (isinstance(link, list) and len(link) == 2):
            return None
        node = wf.get(str(link[0]))
        if not isinstance(node, dict):
            return None
        inputs = node.get("inputs", {})
        for fieldname in _PROMPT_FIELDS:
            val = inputs.get(fieldname)
            if isinstance(val, str):
                return (str(link[0]), fieldname)
            if isinstance(val, list):
                deeper = _follow_to_text(val, depth + 1)
                if deeper:
                    return deeper
        return None

    # ── 1. Prompts — via sampler positive/negative conditioning links ─────────
    pos_done = neg_done = False
    for node in wf.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        pos_link, neg_link = inputs.get("positive"), inputs.get("negative")
        if isinstance(pos_link, list) and not pos_done:
            hit = _follow_to_text(pos_link)
            if hit:
                wf[hit[0]]["inputs"][hit[1]] = "{{POSITIVE_PROMPT}}"
                notes.append(f"✅ positive prompt → node {hit[0]}.{hit[1]}")
                pos_done = True
        if isinstance(neg_link, list) and not neg_done:
            hit = _follow_to_text(neg_link)
            if hit:
                wf[hit[0]]["inputs"][hit[1]] = "{{NEGATIVE_PROMPT}}"
                notes.append(f"✅ negative prompt → node {hit[0]}.{hit[1]}")
                neg_done = True

    # Fallback: CLIPTextEncode nodes in id order — first = positive, second = negative
    if not pos_done:
        encoders = [
            (nid, n) for nid, n in sorted(wf.items())
            if isinstance(n, dict) and n.get("class_type", "").startswith("CLIPTextEncode")
            and isinstance(n.get("inputs", {}).get("text"), str)
        ]
        if encoders:
            nid, n = encoders[0]
            n["inputs"]["text"] = "{{POSITIVE_PROMPT}}"
            notes.append(f"✅ positive prompt → node {nid}.text (fallback)")
            pos_done = True
            if len(encoders) > 1 and not neg_done:
                nid2, n2 = encoders[1]
                n2["inputs"]["text"] = "{{NEGATIVE_PROMPT}}"
                notes.append(f"✅ negative prompt → node {nid2}.text (fallback)")
                neg_done = True
    if not pos_done:
        notes.append("❌ positive prompt NOT found — edit the JSON manually before use")
    if not neg_done:
        notes.append("⚠️ negative prompt not found (some workflows don't use one)")

    # ── 2-6. Seeds, dimensions, frames, fps, output prefix, input image ───────
    for nid, node in wf.items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type", "")
        inputs     = node.get("inputs", {})

        # Primitive value nodes — recognised by their _meta title
        if class_type.startswith("Primitive") and isinstance(inputs.get("value"), (int, float)) \
                and not isinstance(inputs.get("value"), bool):
            title = (node.get("_meta", {}).get("title") or "").strip().lower()
            token = _TITLE_TOKEN_MAP.get(title)
            if token:
                inputs["value"] = "{{" + token + "}}"
                notes.append(f"✅ {token.lower()} → node {nid}.value (titled '{title}')")
                continue

        for fieldname, val in list(inputs.items()):
            if isinstance(val, bool) or isinstance(val, list):
                continue
            if fieldname in ("seed", "noise_seed") and isinstance(val, int):
                inputs[fieldname] = "{{SEED}}"
                notes.append(f"✅ seed → node {nid}.{fieldname}")
            elif fieldname in ("width", "height") and isinstance(val, int) \
                    and ("Latent" in class_type or class_type.startswith("Empty")):
                inputs[fieldname] = "{{" + fieldname.upper() + "}}"
                notes.append(f"✅ {fieldname} → node {nid}.{fieldname}")
            elif fieldname in _FRAMES_FIELDS and isinstance(val, int) \
                    and ("Latent" in class_type or "Video" in class_type or class_type.startswith("Empty")):
                inputs[fieldname] = "{{FRAMES}}"
                notes.append(f"✅ frames → node {nid}.{fieldname}")
            elif fieldname == "fps" and isinstance(val, (int, float)):
                inputs[fieldname] = "{{FPS}}"
                notes.append(f"✅ fps → node {nid}.fps")
            elif fieldname == "filename_prefix" and isinstance(val, str):
                inputs[fieldname] = "{{OUTPUT_PREFIX}}"
                notes.append(f"✅ output prefix → node {nid}.filename_prefix")
            elif fieldname == "image" and isinstance(val, str) and class_type == "LoadImage":
                inputs[fieldname] = "{{INPUT_IMAGE}}"
                notes.append(f"✅ input image → node {nid}.image")

    # ── Serialise; numeric tokens must be UNQUOTED in the final template ──────
    text = json.dumps(wf, indent=2)
    for token in ("WIDTH", "HEIGHT", "SEED", "FRAMES", "FPS"):
        text = text.replace('"{{' + token + '}}"', "{{" + token + "}}")

    return text, notes


def save_uploaded_workflow(filename: str, raw_text: str,
                           workflows_dir: str | Path | None = None) -> tuple[str, list[str]]:
    """Template (if needed) and save an uploaded workflow into workflows/.

    If the file already contains {{PLACEHOLDERS}} it is saved as-is; otherwise
    auto_template_workflow() runs first. Returns (posix_rel_path, notes).
    """
    wf_dir = Path(workflows_dir) if workflows_dir else WORKFLOWS_DIR
    wf_dir.mkdir(parents=True, exist_ok=True)

    if _PLACEHOLDER_RE.search(raw_text):
        text, notes = raw_text, ["already templated — saved as-is"]
    else:
        text, notes = auto_template_workflow(raw_text)

    # Safe, unique filename
    stem = re.sub(r"[^A-Za-z0-9_\-]", "_", Path(filename).stem) or "workflow"
    dest = wf_dir / f"{stem}.json"
    counter = 2
    while dest.exists():
        dest = wf_dir / f"{stem}_{counter}.json"
        counter += 1
    dest.write_text(text, encoding="utf-8")

    try:
        rel = dest.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        rel = dest.as_posix()
    return rel, notes
