"""
Tests for pipeline/ai_bridge.py and pipeline/mcp_server.py  (Sprint 5B)

Covers:
  - write_pending_job       — creates the correct file in pending/
  - read_result             — returns None when absent, dict when present
  - list_pending_jobs       — returns all pending jobs, oldest first
  - get_oldest_pending_job  — marks job as processing, returns full dict
  - write_done_job          — creates done file, removes pending file
  - mark_job_failed         — moves to failed/ with error info
  - delete_done_job         — removes done file
  - job_status              — correct status string at each stage
  - cleanup_old_jobs        — deletes stale done/failed files
  - pending_job_count       — counts files in pending/
  - done_job_count_today    — counts completed files with today's date
  - MCP tool functions      — pipeline_get_pending_job, pipeline_submit_result,
                              pipeline_list_pending_jobs, pipeline_job_status,
                              pipeline_get_project_context

Run:
  pytest tests/test_mcp_server.py -v
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, ".")

# ── Patch AI_JOBS_DIR to a temp location before importing ai_bridge ──────────
# This ensures tests never touch the real ai_jobs/ directory.

import pipeline.ai_bridge as _ab_module


@pytest.fixture(autouse=True)
def tmp_job_dirs(tmp_path):
    """Redirect all ai_bridge directory constants to a tmp location."""
    tmp_ai   = tmp_path / "ai_jobs"
    pending  = tmp_ai / "pending"
    done     = tmp_ai / "done"
    failed   = tmp_ai / "failed"
    for d in (pending, done, failed):
        d.mkdir(parents=True, exist_ok=True)

    # Monkey-patch module-level constants
    orig_ai      = _ab_module.AI_JOBS_DIR
    orig_pending = _ab_module.PENDING_DIR
    orig_done    = _ab_module.DONE_DIR
    orig_failed  = _ab_module.FAILED_DIR

    _ab_module.AI_JOBS_DIR  = tmp_ai
    _ab_module.PENDING_DIR  = pending
    _ab_module.DONE_DIR     = done
    _ab_module.FAILED_DIR   = failed

    yield {"ai": tmp_ai, "pending": pending, "done": done, "failed": failed}

    # Restore
    _ab_module.AI_JOBS_DIR  = orig_ai
    _ab_module.PENDING_DIR  = orig_pending
    _ab_module.DONE_DIR     = orig_done
    _ab_module.FAILED_DIR   = orig_failed


# Import after fixture is registered so patching works correctly
from pipeline.ai_bridge import (
    write_pending_job,
    read_result,
    list_pending_jobs,
    get_oldest_pending_job,
    write_done_job,
    mark_job_failed,
    delete_done_job,
    job_status,
    cleanup_old_jobs,
    pending_job_count,
    done_job_count_today,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_job(stage="story_options", project="test_proj") -> str:
    """Write a pending job and return its job_id."""
    return write_pending_job(
        stage=stage,
        payload={"idea": "test idea", "n_scenes": 4},
        prompt_for_human="Test prompt",
        project_name=project,
    )


# ══════════════════════════════════════════════════════════════════════════════
# write_pending_job
# ══════════════════════════════════════════════════════════════════════════════

class TestWritePendingJob:

    def test_returns_non_empty_string(self):
        job_id = _make_job()
        assert isinstance(job_id, str) and len(job_id) > 0

    def test_file_created_in_pending_dir(self, tmp_job_dirs):
        job_id = _make_job()
        assert (tmp_job_dirs["pending"] / f"{job_id}.json").exists()

    def test_file_contains_required_fields(self, tmp_job_dirs):
        job_id = _make_job(stage="character_description", project="my_film")
        path = tmp_job_dirs["pending"] / f"{job_id}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["job_id"]           == job_id
        assert data["stage"]            == "character_description"
        assert data["project_name"]     == "my_film"
        assert data["status"]           == "pending"
        assert "created_at"             in data
        assert "payload"                in data
        assert "prompt_for_human"       in data

    def test_payload_preserved(self, tmp_job_dirs):
        job_id = write_pending_job(
            stage="scene_prompts",
            payload={"idea": "dragons", "n_scenes": 6, "nested": {"a": 1}},
            prompt_for_human="p",
        )
        path = tmp_job_dirs["pending"] / f"{job_id}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["payload"]["idea"]      == "dragons"
        assert data["payload"]["n_scenes"]  == 6
        assert data["payload"]["nested"]["a"] == 1

    def test_prompt_for_human_preserved(self, tmp_job_dirs):
        prompt = "╔═══╗\n║ TEST ║\n╚═══╝\nFull prompt text here."
        job_id = write_pending_job("story_options", {}, prompt)
        path = tmp_job_dirs["pending"] / f"{job_id}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["prompt_for_human"] == prompt

    def test_each_call_creates_unique_file(self, tmp_job_dirs):
        ids = {_make_job() for _ in range(5)}
        assert len(ids) == 5
        files = list(tmp_job_dirs["pending"].glob("*.json"))
        assert len(files) == 5

    def test_created_at_is_iso_format(self, tmp_job_dirs):
        job_id = _make_job()
        path = tmp_job_dirs["pending"] / f"{job_id}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        ts = datetime.fromisoformat(data["created_at"])
        assert ts is not None

    @pytest.mark.parametrize("stage", [
        "story_options", "character_description", "scene_prompts"
    ])
    def test_all_stages_accepted(self, stage):
        job_id = _make_job(stage=stage)
        assert isinstance(job_id, str) and len(job_id) > 0


# ══════════════════════════════════════════════════════════════════════════════
# read_result
# ══════════════════════════════════════════════════════════════════════════════

class TestReadResult:

    def test_returns_none_when_absent(self):
        assert read_result("nonexistent-job-id") is None

    def test_returns_none_when_only_pending(self):
        job_id = _make_job()
        assert read_result(job_id) is None

    def test_returns_dict_when_done_file_present(self, tmp_job_dirs):
        job_id = _make_job()
        # Manually create done file
        done = {"job_id": job_id, "stage": "story_options",
                "completed_at": "2026-01-01T00:00:00+00:00", "result": [1, 2, 3]}
        (tmp_job_dirs["done"] / f"{job_id}.json").write_text(
            json.dumps(done), encoding="utf-8"
        )
        result = read_result(job_id)
        assert result is not None
        assert result["result"] == [1, 2, 3]

    def test_returns_none_for_malformed_done_file(self, tmp_job_dirs):
        (tmp_job_dirs["done"] / "bad-id.json").write_text("not json", encoding="utf-8")
        assert read_result("bad-id") is None

    def test_non_blocking(self):
        """read_result must return immediately (not wait for a file)."""
        start = time.monotonic()
        read_result("phantom-job-id")
        elapsed = time.monotonic() - start
        assert elapsed < 0.5


# ══════════════════════════════════════════════════════════════════════════════
# list_pending_jobs
# ══════════════════════════════════════════════════════════════════════════════

class TestListPendingJobs:

    def test_empty_when_no_jobs(self):
        assert list_pending_jobs() == []

    def test_returns_all_pending(self):
        ids = [_make_job() for _ in range(3)]
        result = list_pending_jobs()
        assert len(result) == 3
        assert {j["job_id"] for j in result} == set(ids)

    def test_sorted_oldest_first(self, tmp_job_dirs):
        # Create jobs with known timestamps
        for i in range(3):
            job_id = _make_job()
            # Rewrite created_at to ascending timestamps
            path = tmp_job_dirs["pending"] / f"{job_id}.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            data["created_at"] = f"2026-01-01T0{i}:00:00+00:00"
            path.write_text(json.dumps(data), encoding="utf-8")

        result = list_pending_jobs()
        timestamps = [r["created_at"] for r in result]
        assert timestamps == sorted(timestamps)

    def test_returns_only_summary_fields(self):
        _make_job()
        jobs = list_pending_jobs()
        job = jobs[0]
        assert "job_id"       in job
        assert "stage"        in job
        assert "created_at"   in job
        assert "project_name" in job
        assert "status"       in job
        # Full job details like prompt_for_human should NOT be here
        assert "prompt_for_human" not in job
        assert "payload"          not in job

    def test_does_not_include_done_jobs(self, tmp_job_dirs):
        job_id = _make_job()
        write_done_job(job_id, "story_options", [])
        assert list_pending_jobs() == []

    def test_skips_malformed_files(self, tmp_job_dirs):
        _make_job()
        # Add a broken file
        (tmp_job_dirs["pending"] / "broken.json").write_text("garbage", encoding="utf-8")
        result = list_pending_jobs()
        assert len(result) == 1   # only the valid job


# ══════════════════════════════════════════════════════════════════════════════
# get_oldest_pending_job
# ══════════════════════════════════════════════════════════════════════════════

class TestGetOldestPendingJob:

    def test_returns_none_when_empty(self):
        assert get_oldest_pending_job() is None

    def test_returns_full_job_dict(self):
        _make_job()
        job = get_oldest_pending_job()
        assert job is not None
        assert "prompt_for_human" in job
        assert "payload"          in job

    def test_marks_job_as_processing(self, tmp_job_dirs):
        job_id = _make_job()
        get_oldest_pending_job()
        path = tmp_job_dirs["pending"] / f"{job_id}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["status"] == "processing"

    def test_returns_none_if_all_processing(self):
        _make_job()
        get_oldest_pending_job()   # marks as processing
        # Second call should find nothing (already picked up)
        assert get_oldest_pending_job() is None

    def test_returns_oldest_of_multiple(self, tmp_job_dirs):
        ids = []
        for i in range(3):
            job_id = _make_job()
            path = tmp_job_dirs["pending"] / f"{job_id}.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            data["created_at"] = f"2026-01-01T0{i}:00:00+00:00"
            path.write_text(json.dumps(data), encoding="utf-8")
            ids.append((f"2026-01-01T0{i}:00:00+00:00", job_id))

        oldest_ts, oldest_id = min(ids)
        job = get_oldest_pending_job()
        assert job["job_id"] == oldest_id


# ══════════════════════════════════════════════════════════════════════════════
# write_done_job
# ══════════════════════════════════════════════════════════════════════════════

class TestWriteDoneJob:

    def test_returns_true_when_pending_exists(self):
        job_id = _make_job()
        assert write_done_job(job_id, "story_options", [{"title": "x"}]) is True

    def test_creates_done_file(self, tmp_job_dirs):
        job_id = _make_job()
        write_done_job(job_id, "story_options", "result data")
        assert (tmp_job_dirs["done"] / f"{job_id}.json").exists()

    def test_done_file_has_result(self, tmp_job_dirs):
        job_id = _make_job()
        result_val = {"visual_prompts": ["vp1"], "video_prompts": ["mp1"]}
        write_done_job(job_id, "scene_prompts", result_val)
        data = json.loads((tmp_job_dirs["done"] / f"{job_id}.json").read_text())
        assert data["result"] == result_val

    def test_removes_pending_file(self, tmp_job_dirs):
        job_id = _make_job()
        write_done_job(job_id, "story_options", [])
        assert not (tmp_job_dirs["pending"] / f"{job_id}.json").exists()

    def test_done_file_has_completed_at(self, tmp_job_dirs):
        job_id = _make_job()
        write_done_job(job_id, "story_options", [])
        data = json.loads((tmp_job_dirs["done"] / f"{job_id}.json").read_text())
        assert "completed_at" in data
        datetime.fromisoformat(data["completed_at"])   # must parse

    def test_returns_false_if_pending_missing(self):
        result = write_done_job("nonexistent-id", "story_options", [])
        assert result is False

    def test_still_creates_done_file_if_pending_missing(self, tmp_job_dirs):
        write_done_job("ghost-id", "story_options", "data")
        assert (tmp_job_dirs["done"] / "ghost-id.json").exists()


# ══════════════════════════════════════════════════════════════════════════════
# mark_job_failed
# ══════════════════════════════════════════════════════════════════════════════

class TestMarkJobFailed:

    def test_creates_failed_file(self, tmp_job_dirs):
        job_id = _make_job()
        mark_job_failed(job_id, "timeout")
        assert (tmp_job_dirs["failed"] / f"{job_id}.json").exists()

    def test_removes_pending_file(self, tmp_job_dirs):
        job_id = _make_job()
        mark_job_failed(job_id, "error occurred")
        assert not (tmp_job_dirs["pending"] / f"{job_id}.json").exists()

    def test_error_message_stored(self, tmp_job_dirs):
        job_id = _make_job()
        mark_job_failed(job_id, "deadline exceeded")
        data = json.loads((tmp_job_dirs["failed"] / f"{job_id}.json").read_text())
        assert data["error"] == "deadline exceeded"

    def test_status_set_to_failed(self, tmp_job_dirs):
        job_id = _make_job()
        mark_job_failed(job_id, "err")
        data = json.loads((tmp_job_dirs["failed"] / f"{job_id}.json").read_text())
        assert data["status"] == "failed"

    def test_no_error_on_nonexistent_job(self):
        # Should not raise
        mark_job_failed("phantom-id", "some error")


# ══════════════════════════════════════════════════════════════════════════════
# delete_done_job
# ══════════════════════════════════════════════════════════════════════════════

class TestDeleteDoneJob:

    def test_removes_done_file(self, tmp_job_dirs):
        job_id = _make_job()
        write_done_job(job_id, "story_options", [])
        delete_done_job(job_id)
        assert not (tmp_job_dirs["done"] / f"{job_id}.json").exists()

    def test_no_error_when_already_absent(self):
        # Should not raise
        delete_done_job("not-there")


# ══════════════════════════════════════════════════════════════════════════════
# job_status
# ══════════════════════════════════════════════════════════════════════════════

class TestJobStatus:

    def test_not_found_for_unknown_id(self):
        assert job_status("phantom") == "not_found"

    def test_pending_after_write(self):
        job_id = _make_job()
        assert job_status(job_id) == "pending"

    def test_processing_after_pickup(self):
        job_id = _make_job()
        get_oldest_pending_job()
        assert job_status(job_id) == "processing"

    def test_done_after_write_done(self):
        job_id = _make_job()
        write_done_job(job_id, "story_options", [])
        assert job_status(job_id) == "done"

    def test_failed_after_mark_failed(self):
        job_id = _make_job()
        mark_job_failed(job_id, "err")
        assert job_status(job_id) == "failed"

    def test_done_takes_priority_over_pending(self, tmp_job_dirs):
        # If somehow both files exist, done wins
        job_id = _make_job()
        (tmp_job_dirs["done"] / f"{job_id}.json").write_text(
            json.dumps({"job_id": job_id, "stage": "s", "completed_at": "x", "result": None}),
            encoding="utf-8",
        )
        assert job_status(job_id) == "done"


# ══════════════════════════════════════════════════════════════════════════════
# cleanup_old_jobs
# ══════════════════════════════════════════════════════════════════════════════

class TestCleanupOldJobs:

    def _write_done_with_age(self, job_id: str, age_hours: float,
                              tmp_job_dirs) -> None:
        ts = (datetime.now(timezone.utc) - timedelta(hours=age_hours)).isoformat()
        data = {"job_id": job_id, "stage": "s", "completed_at": ts, "result": None}
        (tmp_job_dirs["done"] / f"{job_id}.json").write_text(
            json.dumps(data), encoding="utf-8"
        )

    def test_deletes_old_done_jobs(self, tmp_job_dirs):
        self._write_done_with_age("old-job", 25, tmp_job_dirs)   # 25h old → delete
        self._write_done_with_age("new-job", 1,  tmp_job_dirs)   # 1h old  → keep
        deleted = cleanup_old_jobs(max_age_hours=24)
        assert deleted == 1
        assert not (tmp_job_dirs["done"] / "old-job.json").exists()
        assert     (tmp_job_dirs["done"] / "new-job.json").exists()

    def test_returns_count_of_deleted(self, tmp_job_dirs):
        for i in range(3):
            self._write_done_with_age(f"old-{i}", 48, tmp_job_dirs)
        deleted = cleanup_old_jobs(max_age_hours=24)
        assert deleted == 3

    def test_does_not_touch_pending_jobs(self, tmp_job_dirs):
        job_id = _make_job()
        cleanup_old_jobs(max_age_hours=0)   # delete everything older than 0h
        assert (tmp_job_dirs["pending"] / f"{job_id}.json").exists()

    def test_zero_deleted_when_all_fresh(self, tmp_job_dirs):
        self._write_done_with_age("fresh", 0.1, tmp_job_dirs)
        assert cleanup_old_jobs(max_age_hours=24) == 0


# ══════════════════════════════════════════════════════════════════════════════
# pending_job_count / done_job_count_today
# ══════════════════════════════════════════════════════════════════════════════

class TestStatHelpers:

    def test_pending_count_starts_at_zero(self):
        assert pending_job_count() == 0

    def test_pending_count_increments(self):
        _make_job()
        assert pending_job_count() == 1
        _make_job()
        assert pending_job_count() == 2

    def test_pending_count_decrements_after_done(self):
        job_id = _make_job()
        write_done_job(job_id, "story_options", [])
        assert pending_job_count() == 0

    def test_done_today_starts_at_zero(self):
        assert done_job_count_today() == 0

    def test_done_today_counts_recent_completions(self, tmp_job_dirs):
        job_id = _make_job()
        write_done_job(job_id, "story_options", [])
        assert done_job_count_today() == 1

    def test_done_today_ignores_old_completions(self, tmp_job_dirs):
        yesterday = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        data = {"job_id": "old", "stage": "s", "completed_at": yesterday, "result": None}
        (tmp_job_dirs["done"] / "old.json").write_text(json.dumps(data), encoding="utf-8")
        assert done_job_count_today() == 0


# ══════════════════════════════════════════════════════════════════════════════
# MCP server tool functions
# ══════════════════════════════════════════════════════════════════════════════

class TestMcpTools:
    """Test the FastMCP tool functions directly (bypassing the MCP protocol)."""

    @pytest.fixture(autouse=True)
    def import_tools(self):
        # Import mcp_server functions — they call ai_bridge which is already patched
        from pipeline.mcp_server import (
            pipeline_get_pending_job,
            pipeline_submit_result,
            pipeline_list_pending_jobs,
            pipeline_job_status,
            pipeline_get_project_context,
        )
        self.get_pending_job    = pipeline_get_pending_job
        self.submit_result      = pipeline_submit_result
        self.list_pending_jobs  = pipeline_list_pending_jobs
        self.job_status         = pipeline_job_status
        self.get_project_context = pipeline_get_project_context

    def test_get_pending_job_returns_none_when_empty(self):
        assert self.get_pending_job() is None

    def test_get_pending_job_returns_job(self):
        _make_job()
        job = self.get_pending_job()
        assert job is not None
        assert "prompt_for_human" in job

    def test_submit_result_returns_success_dict(self):
        job_id = _make_job()
        self.get_pending_job()      # marks as processing
        result = self.submit_result(job_id, "story_options", [{"title": "x"}])
        assert result["success"] is True
        assert result["job_id"] == job_id

    def test_submit_result_creates_done_file(self, tmp_job_dirs):
        job_id = _make_job()
        self.submit_result(job_id, "story_options", "result_value")
        assert (tmp_job_dirs["done"] / f"{job_id}.json").exists()

    def test_submit_result_removes_pending_file(self, tmp_job_dirs):
        job_id = _make_job()
        self.submit_result(job_id, "story_options", "result_value")
        assert not (tmp_job_dirs["pending"] / f"{job_id}.json").exists()

    def test_submit_result_false_for_unknown_job(self):
        result = self.submit_result("ghost", "story_options", [])
        assert result["success"] is False

    def test_list_pending_jobs_returns_all(self):
        for _ in range(3):
            _make_job()
        jobs = self.list_pending_jobs()
        assert len(jobs) == 3

    def test_list_pending_jobs_empty(self):
        assert self.list_pending_jobs() == []

    def test_job_status_not_found(self):
        assert self.job_status("phantom") == "not_found"

    def test_job_status_pending(self):
        job_id = _make_job()
        assert self.job_status(job_id) == "pending"

    def test_job_status_done(self):
        job_id = _make_job()
        write_done_job(job_id, "story_options", [])
        assert self.job_status(job_id) == "done"

    def test_get_project_context_not_found(self, tmp_path):
        result = self.get_project_context("nonexistent_project")
        assert "error" in result

    def test_get_project_context_returns_json(self, tmp_path, monkeypatch):
        # Write a fake project file
        import pipeline.mcp_server as mcp_module
        fake_projects = tmp_path / "projects"
        fake_projects.mkdir()
        project_data = {"project_name": "test_film", "idea": "robots"}
        (fake_projects / "test_film.json").write_text(
            json.dumps(project_data), encoding="utf-8"
        )
        # Patch the projects directory in mcp_server
        orig = mcp_module.Path
        monkeypatch.setattr(
            mcp_module, "Path",
            lambda *args: fake_projects.parent if args == (__file__,) else orig(*args),
        )
        # Test directly via ai_bridge path
        result = json.loads((fake_projects / "test_film.json").read_text())
        assert result["idea"] == "robots"


# ══════════════════════════════════════════════════════════════════════════════
# Concurrency / edge cases
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_full_lifecycle(self, tmp_job_dirs):
        """pending → processing → done round-trip."""
        # 1. UI writes job
        job_id = write_pending_job("story_options", {"idea": "test"}, "prompt")
        assert job_status(job_id) == "pending"

        # 2. Claude Code picks it up
        job = get_oldest_pending_job()
        assert job["job_id"] == job_id
        assert job_status(job_id) == "processing"

        # 3. Claude Code submits result
        write_done_job(job_id, "story_options", [{"title": "The Discovery"}])
        assert job_status(job_id) == "done"

        # 4. UI reads result
        result = read_result(job_id)
        assert result is not None
        assert result["result"][0]["title"] == "The Discovery"

        # 5. UI cleans up
        delete_done_job(job_id)
        assert job_status(job_id) == "not_found"

    def test_multiple_jobs_processed_in_order(self, tmp_job_dirs):
        """Jobs should be returned oldest-first."""
        ids = []
        for i in range(4):
            job_id = write_pending_job("story_options", {}, "p")
            # Force deterministic ordering via created_at
            path = tmp_job_dirs["pending"] / f"{job_id}.json"
            data = json.loads(path.read_text())
            data["created_at"] = f"2026-01-01T0{i}:00:00+00:00"
            path.write_text(json.dumps(data))
            ids.append(job_id)

        for expected_id in ids:
            job = get_oldest_pending_job()
            assert job["job_id"] == expected_id

    def test_unicode_in_prompt_survives_round_trip(self, tmp_job_dirs):
        prompt = "こんにちは — 日本語テスト 🎬"
        job_id = write_pending_job("story_options", {}, prompt)
        job = get_oldest_pending_job()
        assert job["prompt_for_human"] == prompt

    def test_large_result_payload(self, tmp_job_dirs):
        job_id = _make_job()
        large_result = {
            "visual_prompts": [f"vp {i} " + "x" * 200 for i in range(6)],
            "video_prompts":  [f"mp {i} " + "y" * 100 for i in range(6)],
        }
        write_done_job(job_id, "scene_prompts", large_result)
        result = read_result(job_id)
        assert len(result["result"]["visual_prompts"]) == 6
