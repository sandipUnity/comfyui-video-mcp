"""
AI Bridge — job file persistence layer for the MCP pipeline queue.

The UI (Streamlit) writes pending jobs; Claude Code reads them via the MCP
server, processes them, and writes done results. The UI polls for results.

Directory layout:
    ai_jobs/pending/   UI writes here; MCP server picks up jobs from here
    ai_jobs/done/      MCP server writes completed results here; UI reads here
    ai_jobs/failed/    timed-out or errored jobs land here

Public API:
    write_pending_job(stage, payload, prompt_for_human, project_name) → str
    read_result(job_id) → dict | None
    list_pending_jobs() → list[dict]
    get_oldest_pending_job() → dict | None
    write_done_job(job_id, stage, result) → bool
    mark_job_failed(job_id, error) → None
    delete_done_job(job_id) → None
    job_status(job_id) → str
    cleanup_old_jobs(max_age_hours=24) → int
    pending_job_count() → int
    done_job_count_today() → int
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

# Job directory — resolved relative to the project root (two levels up from
# this file), so it works regardless of where Python is invoked from.
_PROJECT_ROOT = Path(__file__).parent.parent
AI_JOBS_DIR  = _PROJECT_ROOT / "ai_jobs"
PENDING_DIR  = AI_JOBS_DIR / "pending"
DONE_DIR     = AI_JOBS_DIR / "done"
FAILED_DIR   = AI_JOBS_DIR / "failed"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _ensure_dirs() -> None:
    """Create the ai_jobs directory tree if it doesn't exist."""
    for d in (PENDING_DIR, DONE_DIR, FAILED_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Write a new pending job ───────────────────────────────────────────────────

def write_pending_job(
    stage: str,
    payload: dict,
    prompt_for_human: str,
    project_name: str = "",
) -> str:
    """Write a pending job file. Returns the job_id (UUID4 string).

    Args:
        stage:            "story_options" | "character_description" | "scene_prompts"
        payload:          dict of all data Claude Code needs to generate the result
        prompt_for_human: the copy-paste prompt — Claude Code processes this directly
        project_name:     project name for reference / get_project_context tool
    """
    _ensure_dirs()
    job_id = str(uuid.uuid4())
    job = {
        "job_id":           job_id,
        "stage":            stage,
        "created_at":       _now_iso(),
        "project_name":     project_name,
        "status":           "pending",
        "payload":          payload,
        "prompt_for_human": prompt_for_human,
    }
    _write_json(PENDING_DIR / f"{job_id}.json", job)
    return job_id


# ── Read a completed result ───────────────────────────────────────────────────

def read_result(job_id: str) -> dict | None:
    """Return the parsed done-file for job_id, or None if not yet complete.

    Non-blocking — returns immediately. The UI is responsible for polling.
    """
    return _read_json(DONE_DIR / f"{job_id}.json")


# ── List / retrieve pending jobs ──────────────────────────────────────────────

def list_pending_jobs() -> list[dict]:
    """Return summary dicts for all pending jobs, oldest first.

    Each entry: job_id, stage, created_at, project_name, status.
    """
    _ensure_dirs()
    jobs: list[dict] = []
    for path in sorted(PENDING_DIR.glob("*.json")):
        data = _read_json(path)
        if data:
            jobs.append({
                "job_id":       data.get("job_id",       path.stem),
                "stage":        data.get("stage",        "unknown"),
                "created_at":   data.get("created_at",   ""),
                "project_name": data.get("project_name", ""),
                "status":       data.get("status",       "pending"),
            })
    jobs.sort(key=lambda j: j["created_at"])
    return jobs


def get_oldest_pending_job() -> dict | None:
    """Return the full job dict for the oldest pending (not yet processing) job.

    "Oldest" is determined by the created_at timestamp in the job file, so the
    queue is properly FIFO regardless of filename ordering.

    Atomically marks the job as "processing" to prevent double pick-up.
    Returns None if no jobs are waiting.
    """
    _ensure_dirs()
    # Read all pending jobs and sort by created_at (oldest first)
    candidates: list[tuple[str, Path, dict]] = []
    for path in PENDING_DIR.glob("*.json"):
        data = _read_json(path)
        if data and data.get("status") == "pending":
            candidates.append((data.get("created_at", ""), path, data))

    candidates.sort(key=lambda x: x[0])   # lexicographic ISO sort → chronological

    for _, path, data in candidates:
        data["status"] = "processing"
        try:
            _write_json(path, data)
        except OSError:
            continue
        return data
    return None


# ── Write a completed result ──────────────────────────────────────────────────

def write_done_job(job_id: str, stage: str, result: Any) -> bool:
    """Write a completed result and remove the pending file.

    Returns True on success, False if the pending job was not found.
    """
    _ensure_dirs()
    pending_path = PENDING_DIR / f"{job_id}.json"
    done_path    = DONE_DIR    / f"{job_id}.json"

    if not pending_path.exists():
        # Already moved or never existed — still write the done file
        # so the UI can pick it up if it was moved manually.
        done = {
            "job_id":       job_id,
            "stage":        stage,
            "completed_at": _now_iso(),
            "result":       result,
        }
        _write_json(done_path, done)
        return False

    done = {
        "job_id":       job_id,
        "stage":        stage,
        "completed_at": _now_iso(),
        "result":       result,
    }
    _write_json(done_path, done)
    try:
        pending_path.unlink()
    except OSError:
        pass
    return True


# ── Mark a job as failed ──────────────────────────────────────────────────────

def mark_job_failed(job_id: str, error: str) -> None:
    """Move a job from pending to failed, recording the error message."""
    _ensure_dirs()
    pending_path = PENDING_DIR / f"{job_id}.json"
    failed_path  = FAILED_DIR  / f"{job_id}.json"

    data = _read_json(pending_path) or {"job_id": job_id}
    data["status"]    = "failed"
    data["error"]     = error
    data["failed_at"] = _now_iso()
    _write_json(failed_path, data)
    try:
        pending_path.unlink()
    except OSError:
        pass


# ── Delete a done job ─────────────────────────────────────────────────────────

def delete_done_job(job_id: str) -> None:
    """Remove a done job file (e.g. after the UI has consumed the result)."""
    path = DONE_DIR / f"{job_id}.json"
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


# ── Status query ──────────────────────────────────────────────────────────────

def job_status(job_id: str) -> str:
    """Return the current status string for job_id.

    Returns: "pending" | "processing" | "done" | "failed" | "not_found"
    """
    if (DONE_DIR    / f"{job_id}.json").exists():
        return "done"
    if (FAILED_DIR  / f"{job_id}.json").exists():
        return "failed"
    pending_path = PENDING_DIR / f"{job_id}.json"
    if pending_path.exists():
        data = _read_json(pending_path)
        if data:
            return data.get("status", "pending")
        return "pending"
    return "not_found"


# ── Housekeeping ──────────────────────────────────────────────────────────────

def cleanup_old_jobs(max_age_hours: int = 24) -> int:
    """Delete done/failed job files older than max_age_hours.

    Returns the number of files deleted.
    """
    _ensure_dirs()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    deleted = 0
    for directory in (DONE_DIR, FAILED_DIR):
        for path in directory.glob("*.json"):
            data = _read_json(path)
            if data is None:
                continue
            ts_str = data.get("completed_at") or data.get("failed_at") or ""
            try:
                if ts_str and datetime.fromisoformat(ts_str) < cutoff:
                    path.unlink(missing_ok=True)
                    deleted += 1
            except (ValueError, OSError):
                pass
    return deleted


# ── Stats helpers (used by sidebar widget) ────────────────────────────────────

def pending_job_count() -> int:
    """Return the number of jobs currently in the pending folder."""
    _ensure_dirs()
    return len(list(PENDING_DIR.glob("*.json")))


def done_job_count_today() -> int:
    """Return the number of jobs completed today (UTC)."""
    _ensure_dirs()
    today = datetime.now(timezone.utc).date()
    count = 0
    for path in DONE_DIR.glob("*.json"):
        data = _read_json(path)
        if data:
            ts_str = data.get("completed_at", "")
            try:
                if ts_str and datetime.fromisoformat(ts_str).date() == today:
                    count += 1
            except ValueError:
                pass
    return count
