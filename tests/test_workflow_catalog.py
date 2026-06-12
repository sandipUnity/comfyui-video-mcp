"""
Tests for pipeline/workflow_catalog.py — local workflow discovery + classification.

Run:
  pytest tests/test_workflow_catalog.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, ".")

from pipeline.workflow_catalog import (
    discover_workflows,
    workflows_for,
    resolve_workflow_path,
    WorkflowInfo,
    PROJECT_ROOT,
    SUPPORTED_PLACEHOLDERS,
)


def _by_name(workflows, name):
    return next((wf for wf in workflows if wf.name == name), None)


# ── Discovery over the real workflows/ folder ─────────────────────────────────

class TestDiscoverRealWorkflows:
    def test_returns_list_of_workflowinfo(self):
        result = discover_workflows()
        assert result, "workflows/ should contain templates"
        assert all(isinstance(wf, WorkflowInfo) for wf in result)

    def test_paths_are_posix_relative(self):
        """Windows backslash paths broke selectbox matching before (07f0fce)."""
        for wf in discover_workflows():
            assert "\\" not in wf.path, f"non-posix path: {wf.path}"
            assert wf.path.startswith("workflows/")

    def test_flux_schnell_is_compatible_t2i(self):
        wf = _by_name(discover_workflows(), "flux_schnell_t2i_api")
        assert wf is not None
        assert wf.kind == "t2i"
        assert wf.compatible, wf.reason

    def test_ltx23_is_compatible_i2v(self):
        wf = _by_name(discover_workflows(), "ltx23_i2v_api")
        assert wf is not None
        assert wf.kind == "i2v"
        assert wf.compatible, wf.reason

    def test_wan22_lightx2v_is_compatible_t2v(self):
        wf = _by_name(discover_workflows(), "wan22_lightx2v_api")
        assert wf is not None
        assert wf.kind == "t2v"
        assert wf.compatible, wf.reason

    def test_legacy_templates_marked_incompatible(self):
        """Templates with {{CFG}}/{{STEPS}}/{{CHECKPOINT}} need the legacy
        injector — they must be excluded from the new pipeline's pickers."""
        workflows = discover_workflows()
        for name in ("animatediff_api", "wan22_t2v_api", "ltxvideo_api", "svd_api"):
            wf = _by_name(workflows, name)
            assert wf is not None, f"{name} not discovered"
            assert not wf.compatible, f"{name} should be incompatible"
            assert wf.reason, f"{name} must carry a human-readable reason"

    def test_workflows_for_filters_kind_and_compatibility(self):
        t2i = workflows_for("t2i")
        assert t2i and all(wf.kind == "t2i" and wf.compatible for wf in t2i)

        video = workflows_for(("i2v", "t2v"))
        assert video and all(wf.kind in ("i2v", "t2v") and wf.compatible for wf in video)
        names = {wf.name for wf in video}
        assert "ltx23_i2v_api" in names
        assert "svd_api" not in names           # incompatible — excluded

    def test_include_incompatible_flag(self):
        all_video = workflows_for(("i2v", "t2v"), include_incompatible=True)
        assert any(not wf.compatible for wf in all_video)


# ── Synthetic templates in a temp dir ─────────────────────────────────────────

class TestDiscoverSynthetic:
    def _write(self, tmp_path, name, content):
        (tmp_path / name).write_text(content, encoding="utf-8")

    def test_no_positive_prompt_is_incompatible(self, tmp_path):
        self._write(tmp_path, "raw_export.json", '{"1": {"inputs": {"text": "hardcoded"}}}')
        wf = discover_workflows(tmp_path)[0]
        assert not wf.compatible
        assert "POSITIVE_PROMPT" in wf.reason

    def test_valid_t2i_template(self, tmp_path):
        self._write(tmp_path, "my_t2i.json",
                    '{"1": {"inputs": {"text": "{{POSITIVE_PROMPT}}", '
                    '"w": {{WIDTH}}, "h": {{HEIGHT}}, "seed": {{SEED}}}}}')
        wf = discover_workflows(tmp_path)[0]
        assert wf.kind == "t2i"
        assert wf.compatible, wf.reason

    def test_input_image_classifies_i2v(self, tmp_path):
        self._write(tmp_path, "my_i2v.json",
                    '{"1": {"inputs": {"text": "{{POSITIVE_PROMPT}}", '
                    '"image": "{{INPUT_IMAGE}}", "frames": {{FRAMES}}}}}')
        wf = discover_workflows(tmp_path)[0]
        assert wf.kind == "i2v"
        assert wf.compatible, wf.reason

    def test_frames_without_image_classifies_t2v(self, tmp_path):
        self._write(tmp_path, "my_t2v.json",
                    '{"1": {"inputs": {"text": "{{POSITIVE_PROMPT}}", "frames": {{FRAMES}}}}}')
        wf = discover_workflows(tmp_path)[0]
        assert wf.kind == "t2v"

    def test_broken_json_is_incompatible(self, tmp_path):
        self._write(tmp_path, "broken.json",
                    '{"1": {"inputs": {"text": "{{POSITIVE_PROMPT}}", }}}')  # trailing comma
        wf = discover_workflows(tmp_path)[0]
        assert not wf.compatible
        assert "not valid JSON" in wf.reason

    def test_unsupported_placeholder_is_incompatible(self, tmp_path):
        self._write(tmp_path, "needs_cfg.json",
                    '{"1": {"inputs": {"text": "{{POSITIVE_PROMPT}}", "cfg": {{CFG}}}}}')
        wf = discover_workflows(tmp_path)[0]
        assert not wf.compatible
        assert "{{CFG}}" in wf.reason


# ── resolve_workflow_path ─────────────────────────────────────────────────────

class TestResolveWorkflowPath:
    def test_relative_resolves_under_project_root(self):
        p = resolve_workflow_path("workflows/flux_schnell_t2i_api.json")
        assert p.is_absolute()
        assert p == PROJECT_ROOT / "workflows" / "flux_schnell_t2i_api.json"
        assert p.exists()

    def test_absolute_passthrough(self, tmp_path):
        f = tmp_path / "x.json"
        assert resolve_workflow_path(f) == f

    def test_supported_set_matches_fill_workflow(self):
        """Guard: if fill_workflow learns new placeholders, update the catalog."""
        import inspect
        from pipeline import utils
        src = inspect.getsource(utils.fill_workflow)
        for token in SUPPORTED_PLACEHOLDERS:
            assert "{{" + token + "}}" in src, (
                f"{token} in SUPPORTED_PLACEHOLDERS but not handled by fill_workflow"
            )
