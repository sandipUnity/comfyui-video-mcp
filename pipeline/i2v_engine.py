"""
I2V Engine — LTX-Video 2.3 image-to-video generation.

generate_video() is the single entry point.
It handles the full lifecycle:
  1. Read the approved image from disk
  2. Upload it to ComfyUI via POST /upload/image → get server filename
  3. Fill the LTX 2.3 I2V workflow template with all parameters
  4. Queue the job via ComfyUI API → get job_id (prompt_id)
  5. Wait for completion (WebSocket with polling fallback)
  6. Download the output video to output_dir/
  7. Return (job_id, local_video_path)

Usage:
    from pipeline.i2v_engine import generate_video

    job_id, video_path = await generate_video(
        client=comfy_client,
        prompt="the queen walks forward, sand swirling around her feet",
        negative_prompt="cartoon, video game",
        image_path=Path("output/scene_01.png"),
        width=1280,
        height=720,
        frames=121,
        fps=25,
        seed=42,
        output_prefix="video/scene_01",
        output_dir=Path("output/my_project/video"),
    )
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

from comfyui_client import ComfyUIClient
from pipeline.utils import fill_workflow

# Path to the I2V workflow template
_I2V_TEMPLATE = Path(__file__).parent.parent / "workflows" / "ltx23_i2v_api.json"

# Default negative prompt for LTX 2.3
_DEFAULT_NEGATIVE = "pc game, console game, video game, cartoon, childish, ugly, deformed, blurry"


async def generate_video(
    client: ComfyUIClient,
    prompt: str,
    image_path: Path,
    negative_prompt: str = _DEFAULT_NEGATIVE,
    width: int = 1280,
    height: int = 720,
    frames: int = 121,
    fps: int = 25,
    seed: int | None = None,
    output_prefix: str = "i2v",
    output_dir: Path | None = None,
    timeout: int = 600,
    progress_callback: Optional[Callable] = None,
) -> tuple[str, Path]:
    """Upload image and generate video via LTX-Video 2.3 I2V.

    Args:
        client:            A ComfyUIClient connected to the ComfyUI server.
        prompt:            Motion-focused positive prompt. Describe what moves and how.
                           The image defines what it looks like — the prompt drives motion.
        image_path:        Local path to the approved storyboard image (PNG/JPG).
        negative_prompt:   Negative prompt. Default targets game/cartoon artifacts.
        width:             Output width. Default 1280 (the PrimitiveInt node "267:257").
        height:            Output height. Default 720 (the PrimitiveInt node "267:258").
        frames:            Number of frames. Default 121 ≈ 5s at 25fps.
        fps:               Frames per second. Default 25.
        seed:              Random seed. If None, a random one is chosen.
        output_prefix:     Filename prefix for the saved video (may include subdir like
                           "video/scene_01").
        output_dir:        Local directory to save the video to. Defaults to output/video/
        timeout:           Max seconds to wait for job completion. Default 600 (10 min).
        progress_callback: Optional async callable(value, max) for progress updates.

    Returns:
        Tuple of (prompt_id, local_video_path).

    Raises:
        FileNotFoundError: If image_path does not exist.
        RuntimeError:      If upload, queue, or download fails.
        TimeoutError:      If the job doesn't complete within *timeout* seconds.
    """
    import random as _random
    if seed is None:
        seed = _random.randint(0, 2**31 - 1)
    if output_dir is None:
        output_dir = Path("output") / "video"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # ── Step 1: Upload image to ComfyUI ──────────────────────────────────────
    img_bytes = image_path.read_bytes()
    server_filename = await client.upload_image(img_bytes, image_path.name)

    # ── Step 2: Fill workflow ─────────────────────────────────────────────────
    timed_prefix = f"{output_prefix}_{int(time.time())}"
    wf = fill_workflow(
        _I2V_TEMPLATE,
        positive_prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        seed=seed,
        output_prefix=timed_prefix,
        input_image=server_filename,
        frames=frames,
        fps=fps,
    )

    # ── Step 3: Queue ─────────────────────────────────────────────────────────
    prompt_id = await client.queue_prompt(wf)

    # ── Step 4: Wait ──────────────────────────────────────────────────────────
    history = await client.wait_for_completion(
        prompt_id,
        timeout=timeout,
        progress_callback=progress_callback,
    )
    if not history:
        raise RuntimeError(f"I2V job {prompt_id} completed but history is empty.")

    # ── Step 5: Download ──────────────────────────────────────────────────────
    saved_paths = await client.download_outputs(history, output_dir, prefix=output_prefix)
    if not saved_paths:
        raise RuntimeError(
            f"I2V job {prompt_id} finished but produced no output files. "
            f"Check ComfyUI logs for node 75 (SaveVideo)."
        )

    return prompt_id, saved_paths[0]
