"""
Model availability checker — queries ComfyUI /models/{folder} and /object_info
to verify that every required model file is installed on the server.

Usage:
    from pipeline.model_checker import check_model_availability, REQUIRED_MODELS

    results = await check_model_availability(client)
    for key, info in results.items():
        print(info["status"], info["filename"])

Result dict per model key:
    {
        "display_name":    "LTX-Video 2.3 Main Checkpoint",
        "filename":        "ltx-2.3-22b-dev-fp8.safetensors",
        "node_class":      "CheckpointLoaderSimple",
        "node_available":  True,   # False if custom node package not installed
        "installed":       True,   # False if model file not found
        "status":          "ok",   # "ok" | "missing_file" | "missing_node" | "unknown"
        "download_url":    "https://...",
        "size_gb":         22.0,
        "required_for":    "I2V generation",
    }

Detection strategy (in order):
    1. GET /models/{folder}  — direct file listing per model subfolder (primary)
    2. GET /object_info      — parse node input options (fallback)
"""

from __future__ import annotations

import aiohttp

# ── Required model specifications ─────────────────────────────────────────────

REQUIRED_MODELS: dict[str, dict] = {
    # ── T2I — Flux Schnell ────────────────────────────────────────────────────
    "flux_schnell": {
        "display_name": "Flux Schnell fp8  (T2I storyboard)",
        "filename":     "flux1-schnell-fp8.safetensors",
        "node_class":   "CheckpointLoaderSimple",
        "field":        "ckpt_name",
        "folder":       "checkpoints",
        "download_url": "https://huggingface.co/black-forest-labs/FLUX.1-schnell",
        "wget_cmd":     "wget -P ComfyUI/models/checkpoints/ https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors",
        "size_gb":      8.0,
        "required_for": "Storyboard image generation (T2I)",
        "pipeline":     "t2i",
    },
    # ── I2V — LTX-Video 2.3 ──────────────────────────────────────────────────
    "ltx_main": {
        "display_name": "LTX-Video 2.3 22B fp8  (main checkpoint)",
        "filename":     "ltx-2.3-22b-dev-fp8.safetensors",
        "node_class":   "CheckpointLoaderSimple",
        "field":        "ckpt_name",
        "folder":       "checkpoints",
        "download_url": "https://huggingface.co/Lightricks/LTX-Video",
        "wget_cmd":     "wget -P ComfyUI/models/checkpoints/ https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltx-video-2b-v0.9.5.safetensors",
        "size_gb":      22.0,
        "required_for": "I2V video generation",
        "pipeline":     "i2v",
    },
    "ltx_lora": {
        "display_name": "LTX 2.3 Distilled LoRA  (speed)",
        "filename":     "ltx-2.3-22b-distilled-lora-384.safetensors",
        "node_class":   "LoraLoaderModelOnly",
        "field":        "lora_name",
        "folder":       "loras",
        "download_url": "https://huggingface.co/Lightricks/LTX-Video",
        "wget_cmd":     "wget -P ComfyUI/models/loras/ <URL>",
        "size_gb":      1.5,
        "required_for": "I2V generation — distilled inference (fast mode)",
        "pipeline":     "i2v",
    },
    "ltx_upscaler": {
        "display_name": "LTX Spatial Upscaler x2  (high-res refinement)",
        "filename":     "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "node_class":   "UpscaleModelLoader",
        "field":        "model_name",
        "folder":       "upscale_models",
        "download_url": "https://huggingface.co/Lightricks/LTX-Video",
        "wget_cmd":     "wget -P ComfyUI/models/upscale_models/ <URL>",
        "size_gb":      0.5,
        "required_for": "I2V generation — high-res refinement stage",
        "pipeline":     "i2v",
    },
    "gemma_text_encoder": {
        "display_name": "Gemma 3 12B fp4 mixed  (text encoder)",
        "filename":     "gemma_3_12B_it_fp4_mixed.safetensors",
        "node_class":   "LTXAVTextEncoderLoader",
        "field":        "text_encoder",
        "field_aliases": ["text_encoder_name", "model_name", "ckpt_name",
                          "encoder", "text_encoder_path"],
        "folder":       "text_encoders",
        "download_url": "https://huggingface.co/Lightricks/LTX-Video",
        "wget_cmd":     "wget -P ComfyUI/models/text_encoders/ <URL>",
        "size_gb":      7.0,
        "required_for": "I2V generation — prompt encoding",
        "pipeline":     "i2v",
    },
}


# ── Main check function ────────────────────────────────────────────────────────

async def check_model_availability(client) -> dict[str, dict]:
    """Query ComfyUI and check all required model files.

    Primary:  GET /models/{folder}  — direct file listing per subfolder
    Fallback: GET /object_info      — parse node input option lists

    Returns a dict keyed by model key (same keys as REQUIRED_MODELS).
    Each value has all REQUIRED_MODELS fields plus:
        status: "ok" | "missing_file" | "missing_node" | "unknown"
    """
    # Fetch both data sources concurrently
    import asyncio
    object_info_task = asyncio.create_task(_fetch_object_info(client))

    # Collect unique folders needed
    folders_needed = {spec["folder"] for spec in REQUIRED_MODELS.values() if "folder" in spec}
    folder_tasks = {
        folder: asyncio.create_task(_fetch_model_folder(client, folder))
        for folder in folders_needed
    }

    object_info = await object_info_task
    folder_files: dict[str, list[str]] = {}
    for folder, task in folder_tasks.items():
        result = await task
        if result is not None:
            folder_files[folder] = result

    results: dict[str, dict] = {}
    for key, spec in REQUIRED_MODELS.items():
        node_class = spec["node_class"]
        filename   = spec["filename"]
        folder     = spec.get("folder")

        # ── Node availability (from object_info) ──────────────────────────────
        if object_info is None:
            node_available = False
        else:
            node_available = node_class in object_info

        # ── File detection ────────────────────────────────────────────────────
        installed = False

        # Strategy 1: direct /models/{folder} listing (primary — most reliable)
        if folder and folder in folder_files:
            installed = _file_in_list(filename, folder_files[folder])

        # Strategy 2: object_info field parsing (fallback)
        if not installed and node_available and object_info:
            field   = spec["field"]
            aliases = [field] + list(spec.get("field_aliases", []))
            installed = _check_field(object_info[node_class], aliases, filename)

        # ── Status ────────────────────────────────────────────────────────────
        if not node_available:
            status = "missing_node"
        elif not installed:
            status = "missing_file"
        else:
            status = "ok"

        results[key] = {
            **spec,
            "node_available": node_available,
            "installed":      installed,
            "status":         status,
        }

    return results


def all_i2v_models_ok(results: dict[str, dict]) -> bool:
    """Return True only if every I2V-required model is installed."""
    return all(
        v["status"] == "ok"
        for v in results.values()
        if v.get("pipeline") == "i2v"
    )


def all_t2i_models_ok(results: dict[str, dict]) -> bool:
    """Return True only if every T2I-required model is installed."""
    return all(
        v["status"] == "ok"
        for v in results.values()
        if v.get("pipeline") == "t2i"
    )


# ── API helpers ───────────────────────────────────────────────────────────────

async def _fetch_object_info(client) -> dict | None:
    """GET /object_info — full node schema."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{client.base_url}/object_info",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    return None
                return await resp.json()
    except Exception:
        return None


async def _fetch_model_folder(client, folder_name: str) -> list[str] | None:
    """GET /models/{folder_name} — list of filenames in that model subfolder.

    ComfyUI returns a JSON list of filename strings, e.g.:
        ["model_a.safetensors", "subdir/model_b.safetensors"]

    Returns None if the endpoint is unavailable or returns an error.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{client.base_url}/models/{folder_name}",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                if isinstance(data, list):
                    return [str(f) for f in data]
                return None
    except Exception:
        return None


# ── File-matching helpers ─────────────────────────────────────────────────────

def _file_in_list(filename: str, file_list: list[str]) -> bool:
    """Return True if *filename* matches any entry in *file_list*.

    Matching is case-insensitive and ignores leading path components so that
    entries like "subdir/model.safetensors" match the bare "model.safetensors".
    Also tries stripping the .safetensors extension in case ComfyUI omits it.
    """
    target       = filename.lower()
    target_stem  = target.rsplit(".", 1)[0]   # without extension

    for entry in file_list:
        entry_lower = entry.lower()
        entry_base  = entry_lower.replace("\\", "/").rsplit("/", 1)[-1]  # basename only
        entry_stem  = entry_base.rsplit(".", 1)[0]

        if entry_lower == target:        # exact match (full path)
            return True
        if entry_base  == target:        # basename exact
            return True
        if entry_stem  == target_stem:   # basename without extension
            return True

    return False


def _check_field(node_def: dict, fields: list[str], target: str) -> bool:
    """Return True if *target* appears in node_def's input options.

    Search strategy (stops as soon as a match is found):
    1. Try each field name in *fields* for an exact match.
    2. Try each field name in *fields* case-insensitively.
    3. Search ALL input fields of the node for the filename (catches renamed fields).
    4. Search ALL fields case-insensitively (last resort).
    """
    target_lower = target.lower()
    try:
        inputs = node_def.get("input", {})
        all_sections = {
            **inputs.get("required", {}),
            **inputs.get("optional", {}),
        }

        def _options(field_spec) -> list:
            if field_spec and isinstance(field_spec, (list, tuple)):
                opts = field_spec[0]
                if isinstance(opts, list):
                    return opts
            return []

        # Pass 1 & 2 — named fields (exact then case-insensitive)
        for fname in fields:
            opts = _options(all_sections.get(fname))
            if target in opts:
                return True
            if any(o.lower() == target_lower for o in opts):
                return True

        # Pass 3 & 4 — scan every field in the node
        for opts in [_options(v) for v in all_sections.values()]:
            if target in opts:
                return True
            if any(o.lower() == target_lower for o in opts):
                return True

    except Exception:
        pass
    return False
