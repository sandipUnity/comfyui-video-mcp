"""
Video generation queue helpers for Sprint 4.

Splits the I2V lifecycle into two separable phases:

Phase 1 — queue_video_job()       (Step 11: Queue to ComfyUI)
    upload approved image → fill workflow → queue_prompt() → return job_id
    Fast: a few seconds per scene.

Phase 2 — check_and_download()    (Step 12: Monitor)
    poll /history for job_id → if done, download video → return local Path
    Called repeatedly until all jobs finish.

Helper — get_all_statuses()
    Single-call snapshot of every scene's current job status.
    Uses /queue (running + pending) and /history (done/failed).
"""

from __future__ import annotations

import time
from pathlib import Path

from comfyui_client import ComfyUIClient
from pipeline.utils import fill_workflow

# Path to I2V workflow template
_I2V_TEMPLATE = Path(__file__).parent.parent / "workflows" / "ltx23_i2v_api.json"

# Default negative for LTX 2.3
_DEFAULT_NEGATIVE = "pc game, console game, video game, cartoon, childish, ugly, deformed, blurry"


# ── Phase 1: Queue ────────────────────────────────────────────────────────────

async def queue_video_job(
    client: ComfyUIClient,
    scene,           # SceneState
    project,         # ProjectState
) -> str:
    """Upload the approved image and queue the I2V job.

    Args:
        client:  ComfyUIClient connected to the server.
        scene:   SceneState with approved_image_path set.
        project: ProjectState for resolution/fps/frames settings.

    Returns:
        ComfyUI prompt_id (job_id).

    Raises:
        FileNotFoundError: approved_image_path doesn't exist.
        RuntimeError:      Upload or queue fails.
    """
    from pathlib import Path as _P
    img_path = _P(scene.approved_image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Approved image not found: {img_path}")

    # Upload image
    img_bytes = img_path.read_bytes()
    server_filename = await client.upload_image(img_bytes, img_path.name)

    # Fill workflow
    output_prefix = f"video/{scene.scene_id}_{int(time.time())}"
    wf = fill_workflow(
        _I2V_TEMPLATE,
        positive_prompt  = scene.video_prompt,
        negative_prompt  = scene.negative_prompt or _DEFAULT_NEGATIVE,
        width            = project.width,
        height           = project.height,
        seed             = scene.seed,
        output_prefix    = output_prefix,
        input_image      = server_filename,
        frames           = project.frames,
        fps              = project.fps,
    )

    # Queue
    prompt_id = await client.queue_prompt(wf)
    return prompt_id


# ── Phase 2: Status + Download ────────────────────────────────────────────────

async def get_job_status(client: ComfyUIClient, job_id: str) -> str:
    """Return the current status of a job.

    Returns one of: "queued" | "running" | "done" | "failed" | "unknown"
    """
    try:
        # Check active queue first
        queue_data = await client.get_queue_status()

        for item in queue_data.get("queue_running", []):
            # ComfyUI queue item: [exec_number, prompt_id, prompt, extra_data, outputs_to_execute]
            if len(item) >= 2 and item[1] == job_id:
                return "running"

        for item in queue_data.get("queue_pending", []):
            if len(item) >= 2 and item[1] == job_id:
                return "queued"

        # Not in queue → check history
        history = await client.get_history(job_id)
        if not history:
            return "unknown"

        status_info = history.get("status", {})
        if status_info.get("status_str") == "error" or not status_info.get("completed", True):
            return "failed"
        return "done"

    except Exception:
        return "unknown"


async def get_all_statuses(client: ComfyUIClient, project) -> dict[str, str]:
    """Return {scene_id: status_str} for all scenes that have a video_job_id.

    Status strings: "not_queued" | "queued" | "running" | "done" | "failed" | "unknown"
    """
    # Fetch queue once and check all job_ids
    try:
        queue_data = await client.get_queue_status()
        running_ids = {
            item[1] for item in queue_data.get("queue_running", [])
            if len(item) >= 2
        }
        pending_ids = {
            item[1] for item in queue_data.get("queue_pending", [])
            if len(item) >= 2
        }
    except Exception:
        running_ids, pending_ids = set(), set()

    statuses: dict[str, str] = {}
    for scene in project.scenes:
        if not scene.video_job_id:
            statuses[scene.scene_id] = "not_queued"
            continue

        jid = scene.video_job_id
        if jid in running_ids:
            statuses[scene.scene_id] = "running"
        elif jid in pending_ids:
            statuses[scene.scene_id] = "queued"
        else:
            # Check history
            try:
                history = await client.get_history(jid)
                if not history:
                    statuses[scene.scene_id] = "unknown"
                else:
                    status_info = history.get("status", {})
                    if status_info.get("status_str") == "error":
                        statuses[scene.scene_id] = "failed"
                    else:
                        statuses[scene.scene_id] = "done"
            except Exception:
                statuses[scene.scene_id] = "unknown"

    return statuses


async def download_completed_video(
    client: ComfyUIClient,
    scene,    # SceneState
    project,  # ProjectState
) -> Path | None:
    """Download the completed video for a scene. Returns local Path or None.

    Queries /history for scene.video_job_id, downloads first video output,
    saves to output_dir/video/ and returns the path.
    Does nothing (returns None) if the job isn't done yet.
    """
    if not scene.video_job_id:
        return None

    history = await client.get_history(scene.video_job_id)
    if not history:
        return None

    status_info = history.get("status", {})
    if status_info.get("status_str") == "error":
        return None

    output_dir = Path(project.output_dir) / "video"
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = await client.download_outputs(history, output_dir, prefix=scene.scene_id)
    if not saved:
        return None
    return saved[0]
