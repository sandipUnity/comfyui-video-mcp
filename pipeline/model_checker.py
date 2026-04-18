"""
Model availability checker — queries ComfyUI /object_info to verify
that every required model file is installed on the server.

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
        "installed":       True,   # False if model file not found in options
        "status":          "ok",   # "ok" | "missing_file" | "missing_node" | "unknown"
        "download_url":    "https://...",
        "size_gb":         22.0,
        "required_for":    "I2V generation",
    }
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
        "download_url": "https://huggingface.co/Lightricks/LTX-Video",
        "wget_cmd":     "wget -P ComfyUI/models/loras/ <URL>",
        "size_gb":      1.5,
        "required_for": "I2V generation — distilled inference (fast mode)",
        "pipeline":     "i2v",
    },
    "ltx_upscaler": {
        "display_name": "LTX Spatial Upscaler x2  (high-res refinement)",
        "filename":     "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "node_class":   "LatentUpscaleModelLoader",
        "field":        "model_name",
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
        "download_url": "https://huggingface.co/Lightricks/LTX-Video",
        "wget_cmd":     "wget -P ComfyUI/models/text_encoders/ <URL>",
        "size_gb":      7.0,
        "required_for": "I2V generation — prompt encoding",
        "pipeline":     "i2v",
    },
}


# ── Main check function ────────────────────────────────────────────────────────

async def check_model_availability(client) -> dict[str, dict]:
    """Query ComfyUI /object_info and check all required model files.

    Returns a dict keyed by model key (same keys as REQUIRED_MODELS).
    Each value has all REQUIRED_MODELS fields plus:
        status: "ok" | "missing_file" | "missing_node" | "unknown"
    """
    object_info = await _fetch_object_info(client)
    if object_info is None:
        return {k: {**v, "node_available": False,
                    "installed": False, "status": "unknown"}
                for k, v in REQUIRED_MODELS.items()}

    results: dict[str, dict] = {}
    for key, spec in REQUIRED_MODELS.items():
        node_class    = spec["node_class"]
        field         = spec["field"]
        filename      = spec["filename"]
        node_available = node_class in object_info
        installed      = False

        if node_available:
            # ComfyUI input schema: object_info[NodeClass]["input"]["required"|"optional"][field][0]
            # where [0] is the list of available option strings.
            installed = _check_field(object_info[node_class], field, filename)

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


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _fetch_object_info(client) -> dict | None:
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


def _check_field(node_def: dict, field: str, target: str) -> bool:
    """Return True if *target* appears in node_def's input options for *field*."""
    try:
        for section in ("required", "optional"):
            field_spec = node_def.get("input", {}).get(section, {}).get(field)
            if field_spec and isinstance(field_spec, (list, tuple)):
                options = field_spec[0]
                if isinstance(options, list) and target in options:
                    return True
    except Exception:
        pass
    return False
