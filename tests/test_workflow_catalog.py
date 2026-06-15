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
    template_defaults,
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

    def test_wan22_i2v_lightx2v_is_compatible_i2v(self):
        """New Wan 2.2 I2V LightX2V cascade — classified I2V, no extra
        placeholders (calibrated values baked in like wan22_lightx2v)."""
        wf = _by_name(discover_workflows(), "wan22_i2v_lightx2v_api")
        assert wf is not None
        assert wf.kind == "i2v", "must be I2V (has {{INPUT_IMAGE}})"
        assert wf.compatible, wf.reason
        # calibrated do-not-tune values are baked in, not injected
        assert template_defaults("workflows/wan22_i2v_lightx2v_api.json") == {}

    def test_wan22_lightx2v_is_compatible_t2v(self):
        wf = _by_name(discover_workflows(), "wan22_lightx2v_api")
        assert wf is not None
        assert wf.kind == "t2v"
        assert wf.compatible, wf.reason

    def test_legacy_templates_compatible_via_config_defaults(self):
        """Templates with {{CFG}}/{{STEPS}}/{{CHECKPOINT}} become compatible
        when a config.yaml models: entry supplies the values."""
        workflows = discover_workflows()
        for name in ("animatediff_api", "wan22_t2v_api", "ltxvideo_api", "ltxvideo_camera_api"):
            wf = _by_name(workflows, name)
            assert wf is not None, f"{name} not discovered"
            assert wf.compatible, f"{name} should be compatible via defaults: {wf.reason}"
            assert "config.yaml" in wf.note

    def test_templates_without_config_entry_stay_incompatible(self):
        workflows = discover_workflows()
        wan21 = _by_name(workflows, "wan21_api")     # no models: entry
        svd   = _by_name(workflows, "svd_api")       # no positive prompt
        assert wan21 is not None and not wan21.compatible
        assert "config.yaml" in wan21.reason         # reason tells the user the fix
        assert svd is not None and not svd.compatible
        assert "POSITIVE_PROMPT" in svd.reason

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
        # {{CHECKPOINT}} has no config entry for this template and no pipeline
        # fallback — must stay incompatible with an actionable reason
        self._write(tmp_path, "needs_ckpt.json",
                    '{"1": {"inputs": {"text": "{{POSITIVE_PROMPT}}", "ckpt_name": "{{CHECKPOINT}}"}}}')
        wf = discover_workflows(tmp_path)[0]
        assert not wf.compatible
        assert "{{CHECKPOINT}}" in wf.reason

    def test_cfg_only_template_compatible_via_pipeline_fallback(self, tmp_path):
        # CFG/STEPS are generic sampler knobs — pipeline defaults cover them
        self._write(tmp_path, "needs_cfg.json",
                    '{"1": {"inputs": {"text": "{{POSITIVE_PROMPT}}", "cfg": {{CFG}}}}}')
        wf = discover_workflows(tmp_path)[0]
        assert wf.compatible, wf.reason


# ── template_defaults ─────────────────────────────────────────────────────────

class TestTemplateDefaults:
    def test_wan22_defaults_from_model_entry(self):
        d = template_defaults("workflows/wan22_t2v_api.json")
        assert d["CHECKPOINT"].startswith("wan2.2_t2v_high")
        assert d["CFG"] == 5.0
        assert d["STEPS"] == 20

    def test_first_entry_owns_defaults_not_variants(self):
        """ltxvideo_fast shares the workflow but its cfg=1.0/steps=6 are
        calibrated to a distilled LoRA the base template doesn't load —
        the base ltxvideo entry (pipeline fallbacks 3.0/20) must win."""
        d = template_defaults("workflows/ltxvideo_api.json")
        assert d["CFG"] == 3.0
        assert d["STEPS"] == 20

    def test_supported_template_has_no_defaults(self):
        assert template_defaults("workflows/flux_schnell_t2i_api.json") == {}

    def test_fill_workflow_with_defaults_leaves_no_tokens(self):
        import json as _json
        import re as _re
        from pipeline.utils import fill_workflow
        tpl = "workflows/wan22_t2v_api.json"
        wf = fill_workflow(tpl, positive_prompt="p — “q”", negative_prompt="n",
                           width=64, height=64, seed=1, output_prefix="x",
                           frames=9, fps=8, extra=template_defaults(tpl))
        assert not _re.findall(r"\{\{[A-Z_]+\}\}", _json.dumps(wf))

    def test_calibrated_model_has_no_injected_defaults(self):
        """wan22_lightx2v's steps/cfg/shift are baked into the template JSON and
        must NEVER be injected — template_defaults must stay empty so the
        do-not-tune values can't be overwritten via config."""
        assert template_defaults("workflows/wan22_lightx2v_api.json") == {}

    def test_ltxvideo_camera_fills_completely(self):
        """CAMERA_LORA (string) + CAMERA_LORA_STRENGTH (bare float) must both
        fill — exercises the mixed string/numeric extra path end-to-end."""
        import json as _json
        import re as _re
        from pipeline.utils import fill_workflow
        tpl = "workflows/ltxvideo_camera_api.json"
        d = template_defaults(tpl)
        assert "CAMERA_LORA" in d and "CAMERA_LORA_STRENGTH" in d
        wf = fill_workflow(tpl, positive_prompt="p", negative_prompt="n",
                           width=64, height=64, seed=1, output_prefix="x",
                           frames=9, fps=8, extra=d)
        assert not _re.findall(r"\{\{[A-Z_]+\}\}", _json.dumps(wf))


# ── fill_workflow leftover-token guard ────────────────────────────────────────

class TestFillWorkflowLeftoverGuard:
    def _write(self, tmp_path, content):
        f = tmp_path / "wf.json"
        f.write_text(content, encoding="utf-8")
        return f

    def test_raises_on_unfilled_placeholder(self, tmp_path):
        from pipeline.utils import fill_workflow
        f = self._write(tmp_path,
            '{"1": {"inputs": {"text": "{{POSITIVE_PROMPT}}", "ckpt": "{{CHECKPOINT}}"}}}')
        with pytest.raises(ValueError, match="unfilled placeholders"):
            fill_workflow(f, positive_prompt="p", negative_prompt="n",
                          width=1, height=1, seed=1, output_prefix="x")

    def test_no_raise_when_extra_supplies_value(self, tmp_path):
        from pipeline.utils import fill_workflow
        f = self._write(tmp_path,
            '{"1": {"inputs": {"text": "{{POSITIVE_PROMPT}}", "ckpt": "{{CHECKPOINT}}"}}}')
        wf = fill_workflow(f, positive_prompt="p", negative_prompt="n",
                           width=1, height=1, seed=1, output_prefix="x",
                           extra={"CHECKPOINT": r"has\backslash.safetensors"})
        assert wf["1"]["inputs"]["ckpt"] == r"has\backslash.safetensors"


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
