"""
T2I Engine — Flux Schnell image generation.

generate_image() is the single entry point.
It handles the full lifecycle:
  1. Fill the Flux Schnell workflow template with all parameters
  2. Queue the job via ComfyUI API
  3. Wait for completion (WebSocket with polling fallback)
  4. Download the output image to output_dir/storyboard/
  5. Return the local file path

Usage:
    from pipeline.t2i_engine import generate_image

    path = await generate_image(
        client=comfy_client,
        prompt="a red rose on a marble table, photorealistic",
        negative_prompt="blurry, ugly",
        width=1024,
        height=1024,
        seed=42,
        output_prefix="scene_01",
        output_dir=Path("output/my_project/storyboard"),
    )
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

from comfyui_client import ComfyUIClient
from pipeline.utils import fill_workflow

# Path to the T2I workflow template — resolved relative to project root
_T2I_TEMPLATE = Path(__file__).parent.parent / "workflows" / "flux_schnell_t2i_api.json"


async def generate_image(
    client: ComfyUIClient,
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    seed: int | None = None,
    output_prefix: str = "t2i",
    output_dir: Path | None = None,
    timeout: int = 180,
    progress_callback: Optional[Callable] = None,
) -> Path:
    """Generate a single image via Flux Schnell. Returns the local file path.

    Args:
        client:            A ComfyUIClient connected to the ComfyUI server.
        prompt:            Full positive prompt string.
        negative_prompt:   Negative prompt string (default: empty — Flux ignores it).
        width:             Output width. Flux Schnell works best at 1024×1024.
        height:            Output height.
        seed:              Random seed. If None, ComfyUI picks one.
        output_prefix:     Filename prefix for the saved image.
        output_dir:        Local directory to save the image to. Defaults to
                           output/storyboard/
        timeout:           Max seconds to wait for job completion.
        progress_callback: Optional async callable(value, max) for progress updates.

    Returns:
        Path to the downloaded image file.

    Raises:
        RuntimeError:  If ComfyUI returns an error or no images are produced.
        TimeoutError:  If the job doesn't complete within *timeout* seconds.
    """
    import random as _random
    if seed is None:
        seed = _random.randint(0, 2**31 - 1)
    if output_dir is None:
        output_dir = Path("output") / "storyboard"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a unique prefix so multiple generations don't collide
    timed_prefix = f"{output_prefix}_{int(time.time())}"

    # Fill the workflow template
    wf = fill_workflow(
        _T2I_TEMPLATE,
        positive_prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        seed=seed,
        output_prefix=timed_prefix,
    )

    # Queue
    prompt_id = await client.queue_prompt(wf)

    # Wait
    history = await client.wait_for_completion(
        prompt_id,
        timeout=timeout,
        progress_callback=progress_callback,
    )
    if not history:
        raise RuntimeError(f"T2I job {prompt_id} completed but history is empty.")

    # Extract image info
    images = await client.get_output_images(history)
    if not images:
        raise RuntimeError(
            f"T2I job {prompt_id} finished but produced no images. "
            f"Check ComfyUI logs."
        )

    # Download first (and typically only) image
    img_info = images[0]
    img_bytes = await client.get_image(
        img_info["filename"],
        img_info.get("subfolder", ""),
        img_info.get("type", "output"),
    )

    # Save locally
    local_path = output_dir / img_info["filename"]
    local_path.write_bytes(img_bytes)

    return local_path


async def generate_images(
    client: ComfyUIClient,
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    base_seed: int | None = None,
    count: int = 1,
    output_prefix: str = "t2i",
    output_dir: Path | None = None,
    timeout: int = 180,
    progress_callback: Optional[Callable] = None,
) -> list[Path]:
    """Generate *count* images (1-5) for a single scene. Each uses seed + i.

    Returns a list of local file paths, one per generated image.
    """
    import random as _random
    if base_seed is None:
        base_seed = _random.randint(0, 2**31 - 1)

    paths: list[Path] = []
    for i in range(count):
        path = await generate_image(
            client=client,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            seed=(base_seed + i) % (2**31),
            output_prefix=f"{output_prefix}_v{i+1}",
            output_dir=output_dir,
            timeout=timeout,
            progress_callback=progress_callback,
        )
        paths.append(path)
    return paths
