"""
MCP Server — exposes the AI pipeline job queue to Claude Code.

Claude Code connects to this server automatically via .mcp.json.
When the user types "start pipeline worker", Claude Code enters a loop:
  1. Call pipeline_list_pending_jobs()
  2. For each job: call pipeline_get_pending_job()
  3. Read prompt_for_human, generate the result with Claude's intelligence
  4. Call pipeline_submit_result(job_id, stage, result)
  5. Wait, repeat

Start manually (for testing):
    venv/Scripts/python.exe pipeline/mcp_server.py

Auto-start via .mcp.json (Claude Code reads this at startup):
    {"mcpServers": {"pipeline": {"command": "venv/Scripts/python.exe",
                                  "args": ["pipeline/mcp_server.py"]}}}

Tools:
    pipeline_get_pending_job      → oldest pending job dict, or null
    pipeline_submit_result        → mark job done, write result file
    pipeline_list_pending_jobs    → summary list of all pending jobs
    pipeline_job_status           → status string for one job
    pipeline_get_project_context  → full project JSON for deeper context
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Ensure project root is on the path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.fastmcp import FastMCP

from pipeline.ai_bridge import (
    get_oldest_pending_job,
    write_done_job,
    list_pending_jobs,
    job_status as _job_status,
    cleanup_old_jobs,
)

mcp = FastMCP(
    "pipeline-worker",
    instructions=(
        "You are connected to the AI Video Pipeline job queue. "
        "When asked to 'start pipeline worker' or 'process pending jobs', enter a loop:\n"
        "1. Call pipeline_list_pending_jobs() to see what's waiting.\n"
        "2. For each job, call pipeline_get_pending_job() to get the full job.\n"
        "3. Read the job's 'prompt_for_human' field. Generate the result described in "
        "   that prompt using your own intelligence — no external API needed.\n"
        "4. Call pipeline_submit_result(job_id, stage, result) with the parsed result "
        "   value (NOT the raw JSON string — the actual Python object).\n"
        "5. Wait ~10 seconds, then repeat from step 1.\n\n"
        "Stage result formats:\n"
        "  story_options: list of story option dicts\n"
        "  character_description: plain string description\n"
        "  scene_prompts: {\"visual_prompts\": [...], \"video_prompts\": [...]}\n"
    ),
)


@mcp.tool()
def pipeline_get_pending_job() -> dict | None:
    """Return the oldest pending pipeline job, or null if none are waiting.

    The returned dict contains:
    - job_id           : pass this to pipeline_submit_result()
    - stage            : "story_options" | "character_description" | "scene_prompts"
    - payload          : all project data as a dict
    - prompt_for_human : the complete prompt — read this and generate the JSON result

    HOW TO PROCESS:
    1. Read prompt_for_human carefully.
    2. Generate the content it asks for (story options, character desc, or scene prompts).
    3. Extract ONLY the result value (not the full JSON wrapper).
       - story_options → the list of 3 dicts
       - character_description → the string
       - scene_prompts → the {"visual_prompts": [...], "video_prompts": [...]} dict
    4. Call pipeline_submit_result(job_id, stage, that_value).

    The job is atomically marked "processing" to prevent double pick-up.
    """
    return get_oldest_pending_job()


@mcp.tool()
def pipeline_submit_result(job_id: str, stage: str, result: Any) -> dict:
    """Submit the completed result for a pipeline job.

    Args:
        job_id  : the job_id from pipeline_get_pending_job()
        stage   : must match the job's stage field exactly
        result  : the parsed result value (NOT a JSON string):
                  - story_options        → Python list of story-option dicts
                  - character_description → Python string
                  - scene_prompts        → Python dict with visual_prompts + video_prompts

    Returns {"success": true} or {"success": false, "error": "..."}.

    After this call the Streamlit UI will detect the done file within its
    3-second polling interval and apply the result automatically.
    """
    try:
        success = write_done_job(job_id, stage, result)
        # Opportunistic cleanup of old completed jobs
        cleanup_old_jobs(max_age_hours=24)
        return {"success": success, "job_id": job_id, "stage": stage}
    except Exception as exc:
        return {"success": False, "error": str(exc), "job_id": job_id}


@mcp.tool()
def pipeline_list_pending_jobs() -> list[dict]:
    """Return a summary of all pending jobs, oldest first.

    Each entry: job_id, stage, created_at, project_name, status.

    Call pipeline_get_pending_job() to retrieve the full job (including the prompt).
    This tool is useful for deciding processing order or checking the queue length.
    """
    return list_pending_jobs()


@mcp.tool()
def pipeline_job_status(job_id: str) -> str:
    """Return the current status of a specific job.

    Returns one of:
        "pending"    — waiting in queue
        "processing" — picked up but result not yet written
        "done"       — result written, UI will apply it
        "failed"     — moved to failed folder with error
        "not_found"  — no file found for this job_id
    """
    return _job_status(job_id)


@mcp.tool()
def pipeline_get_project_context(project_name: str) -> dict:
    """Return the full saved ProjectState for a named project.

    Useful when the job payload doesn't contain enough context, or when you
    need to understand the full storyboard / approved scenes for a project.

    Returns the raw project JSON dict, or {"error": "..."} if not found.
    """
    projects_dir = Path(__file__).parent.parent / "projects"
    project_file = projects_dir / f"{project_name}.json"
    if not project_file.exists():
        return {"error": f"Project '{project_name}' not found in projects/"}
    try:
        return json.loads(project_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return {"error": str(exc)}


if __name__ == "__main__":
    mcp.run()
