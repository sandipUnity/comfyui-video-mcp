"""
Model catalog — detect model slots in workflow templates and list the models
actually installed on the remote ComfyUI server, so the UI can offer per-slot
model dropdowns instead of hardcoding filenames.

How it fits together:
  detect_model_slots(template)        → which loader nodes/fields hold model files
  ComfyUIClient.get_object_info()     → what the server has installed (per class+field)
  options_for_slot(object_info, slot) → valid choices for one slot
  apply_model_overrides(wf, overrides)→ swap models into the filled workflow dict

Overrides are stored per project as {"<node_id>:<field>": "<model filename>"}
(ProjectState.model_overrides_t2i / model_overrides_i2v) and applied AFTER
fill_workflow(), right before queuing.

Detection is generic on purpose: any *string* input whose value looks like a
model file (by extension) counts as a slot. That covers custom loaders such as
LTXAVTextEncoderLoader / LatentUpscaleModelLoader without maintaining a class
allowlist.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pipeline.workflow_catalog import resolve_workflow_path

# File extensions that identify a model-file input value
MODEL_FILE_EXTS = (".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx")

# Node classes whose file inputs are NOT models (handled elsewhere in the pipeline)
_SKIP_CLASSES = frozenset({"LoadImage", "LoadImageMask", "SaveImage", "SaveVideo"})


@dataclass
class ModelSlot:
    node_id: str        # workflow node id, e.g. "75" or "267:243"
    class_type: str     # e.g. "UNETLoader"
    field: str          # e.g. "unet_name"
    current: str        # model filename currently in the template

    @property
    def key(self) -> str:
        """Stable override key: '<node_id>:<field>'."""
        return f"{self.node_id}:{self.field}"

    @property
    def label(self) -> str:
        """Human-readable label for UI dropdowns."""
        return f"{self.class_type} · {self.field}"


def _parse_template(template_path: str | Path) -> dict:
    """Load a workflow template, dummy-filling numeric placeholders so it parses."""
    text = resolve_workflow_path(template_path).read_text(encoding="utf-8")
    for token in ("WIDTH", "HEIGHT", "SEED", "FRAMES", "FPS"):
        text = text.replace("{{" + token + "}}", "1")
    return json.loads(text)


def detect_model_slots(template_path: str | Path) -> list[ModelSlot]:
    """Return every model-file input found in a workflow template.

    Slots are ordered by node id for a stable UI layout.
    """
    wf = _parse_template(template_path)
    slots: list[ModelSlot] = []

    for node_id in sorted(wf.keys()):
        node = wf[node_id]
        class_type = node.get("class_type", "")
        if class_type in _SKIP_CLASSES:
            continue
        for fieldname, value in node.get("inputs", {}).items():
            if isinstance(value, str) and value.lower().endswith(MODEL_FILE_EXTS):
                slots.append(ModelSlot(
                    node_id=node_id, class_type=class_type,
                    field=fieldname, current=value,
                ))

    return slots


def options_for_slot(object_info: dict, slot: ModelSlot) -> list[str]:
    """Return the server's installed models valid for this slot.

    Looks up the slot's node class in /object_info and reads the option list
    for the slot's field. Empty list means the server doesn't expose options
    (e.g. custom node missing) — the UI should fall back to free entry.
    """
    node_spec = object_info.get(slot.class_type)
    if not node_spec:
        return []
    inputs = node_spec.get("input", {})
    for section in ("required", "optional"):
        spec = (inputs.get(section) or {}).get(slot.field)
        if not (isinstance(spec, list) and spec):
            continue
        # Classic format: [[...options], config?]
        if isinstance(spec[0], list):
            return [str(x) for x in spec[0]]
        # Newer format (ComfyUI v3 nodes): ["COMBO", {"options": [...]}]
        if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
            opts = spec[1].get("options")
            if isinstance(opts, list):
                return [str(x) for x in opts]
    return []


def apply_model_overrides(wf: dict, overrides: dict | None) -> dict:
    """Swap overridden model filenames into a filled workflow dict, in place.

    Keys are '<node_id>:<field>' as produced by ModelSlot.key. Unknown keys
    (e.g. the user switched to a workflow without that node) are skipped
    silently so stale overrides never break queuing.
    """
    for key, value in (overrides or {}).items():
        node_id, _, fieldname = key.partition(":")
        # Node ids may themselves contain ':' (subgraphs like "267:243"):
        # partition splits at the FIRST colon, so re-join correctly by trying
        # the longest node-id match present in the workflow.
        if node_id not in wf and ":" in fieldname:
            # try progressively longer node ids: "267" -> "267:243"
            parts = key.split(":")
            for cut in range(len(parts) - 1, 0, -1):
                cand_id = ":".join(parts[:cut])
                cand_field = ":".join(parts[cut:])
                if cand_id in wf:
                    node_id, fieldname = cand_id, cand_field
                    break
        node = wf.get(node_id)
        if node and fieldname in node.get("inputs", {}):
            node["inputs"][fieldname] = value
    return wf


def slot_status(slot: ModelSlot, options: list[str], override: str | None = None) -> tuple[str, bool]:
    """Return (effective_model, installed) for display.

    effective_model = override if set, else the template default.
    installed       = whether that file appears in the server's option list
                      (False also when the option list is empty/unknown).
    """
    effective = override or slot.current
    return effective, bool(options) and effective in options
