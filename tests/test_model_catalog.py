"""
Tests for pipeline/model_catalog.py — model slot detection, server options,
override application — and workflow_catalog auto-templating of raw exports.

Run:
  pytest tests/test_model_catalog.py -v
"""

from __future__ import annotations

import json
import sys

import pytest

sys.path.insert(0, ".")

from pipeline.model_catalog import (
    detect_model_slots,
    options_for_slot,
    apply_model_overrides,
    slot_status,
    required_node_types,
    missing_node_types,
    ModelSlot,
)
from pipeline.workflow_catalog import auto_template_workflow, save_uploaded_workflow


# ── detect_model_slots on real templates ──────────────────────────────────────

class TestDetectModelSlots:
    def test_flux_has_checkpoint_slot(self):
        slots = detect_model_slots("workflows/flux_schnell_t2i_api.json")
        assert any(s.class_type == "CheckpointLoaderSimple" and s.field == "ckpt_name"
                   for s in slots)

    def test_wan22_lightx2v_has_all_loader_slots(self):
        slots = detect_model_slots("workflows/wan22_lightx2v_api.json")
        classes = {s.class_type for s in slots}
        # dual UNETs, CLIP, VAE, dual LoRAs
        assert {"UNETLoader", "CLIPLoader", "VAELoader", "LoraLoaderModelOnly"} <= classes
        assert len([s for s in slots if s.class_type == "UNETLoader"]) == 2

    def test_ltx23_detects_custom_loader_classes(self):
        """Generic extension-based detection must catch LTX custom loaders."""
        slots = detect_model_slots("workflows/ltx23_i2v_api.json")
        classes = {s.class_type for s in slots}
        assert "LTXAVTextEncoderLoader" in classes
        assert "LatentUpscaleModelLoader" in classes

    def test_input_image_not_a_model_slot(self):
        slots = detect_model_slots("workflows/ltx23_i2v_api.json")
        assert not any(s.class_type == "LoadImage" for s in slots)

    def test_slot_key_format(self):
        slots = detect_model_slots("workflows/flux_schnell_t2i_api.json")
        for s in slots:
            assert s.key == f"{s.node_id}:{s.field}"


# ── options_for_slot ──────────────────────────────────────────────────────────

class TestOptionsForSlot:
    _INFO = {
        "CheckpointLoaderSimple": {
            "input": {"required": {"ckpt_name": [["a.safetensors", "b.safetensors"], {}]}}
        },
        "UNETLoader": {
            "input": {"required": {
                "unet_name": [["wan_high.safetensors"], {}],
                "weight_dtype": [["default", "fp8_e4m3fn"], {}],
            }}
        },
        # Newer ComfyUI v3 node spec format (used by LTX custom loaders)
        "LTXAVTextEncoderLoader": {
            "input": {"required": {
                "text_encoder": ["COMBO", {"options": ["gemma_a.safetensors", "gemma_b.safetensors"]}],
            }}
        },
    }

    def _slot(self, cls="CheckpointLoaderSimple", field="ckpt_name"):
        return ModelSlot(node_id="1", class_type=cls, field=field, current="x.safetensors")

    def test_returns_server_models(self):
        opts = options_for_slot(self._INFO, self._slot())
        assert opts == ["a.safetensors", "b.safetensors"]

    def test_combo_spec_format_supported(self):
        opts = options_for_slot(self._INFO, self._slot(cls="LTXAVTextEncoderLoader",
                                                       field="text_encoder"))
        assert opts == ["gemma_a.safetensors", "gemma_b.safetensors"]

    def test_unknown_class_returns_empty(self):
        assert options_for_slot(self._INFO, self._slot(cls="NoSuchLoader")) == []

    def test_unknown_field_returns_empty(self):
        assert options_for_slot(self._INFO, self._slot(field="nope")) == []

    def test_slot_status_installed_flag(self):
        slot = self._slot()
        eff, installed = slot_status(slot, ["x.safetensors", "y.safetensors"])
        assert eff == "x.safetensors" and installed
        eff, installed = slot_status(slot, ["y.safetensors"])
        assert eff == "x.safetensors" and not installed
        eff, installed = slot_status(slot, ["y.safetensors"], override="y.safetensors")
        assert eff == "y.safetensors" and installed


# ── apply_model_overrides ─────────────────────────────────────────────────────

class TestApplyModelOverrides:
    def test_simple_override(self):
        wf = {"75": {"class_type": "UNETLoader",
                     "inputs": {"unet_name": "old.safetensors"}}}
        apply_model_overrides(wf, {"75:unet_name": "new.safetensors"})
        assert wf["75"]["inputs"]["unet_name"] == "new.safetensors"

    def test_subgraph_node_id_with_colons(self):
        """LTX subgraph ids like '267:243' contain ':' — key parsing must cope."""
        wf = {"267:243": {"class_type": "LTXAVTextEncoderLoader",
                          "inputs": {"text_encoder": "old.safetensors"}}}
        apply_model_overrides(wf, {"267:243:text_encoder": "new.safetensors"})
        assert wf["267:243"]["inputs"]["text_encoder"] == "new.safetensors"

    def test_stale_override_skipped_silently(self):
        wf = {"1": {"class_type": "X", "inputs": {"a": 1}}}
        apply_model_overrides(wf, {"99:gone": "x.safetensors", "1:missing": "y"})
        assert wf == {"1": {"class_type": "X", "inputs": {"a": 1}}}

    def test_none_overrides_noop(self):
        wf = {"1": {"inputs": {"m": "a"}}}
        assert apply_model_overrides(wf, None) == wf

    def test_user_override_beats_config_default_end_to_end(self):
        """The user's per-slot pick must win over the config.yaml default —
        apply_model_overrides runs AFTER fill_workflow(extra=defaults)."""
        from pipeline.utils import fill_workflow
        from pipeline.workflow_catalog import template_defaults
        tpl = "workflows/wan22_t2v_api.json"
        wf = fill_workflow(tpl, positive_prompt="p", negative_prompt="n",
                           width=64, height=64, seed=1, output_prefix="x",
                           frames=9, fps=8, extra=template_defaults(tpl))
        unet = next(s for s in detect_model_slots(tpl) if s.field == "unet_name")
        # config default is the high-noise UNET; override to low-noise
        new_model = "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
        assert wf[unet.node_id]["inputs"]["unet_name"] != new_model   # default first
        apply_model_overrides(wf, {unet.key: new_model})
        assert wf[unet.node_id]["inputs"]["unet_name"] == new_model    # override wins


# ── node-type availability ────────────────────────────────────────────────────

class TestNodeAvailability:
    def test_required_node_types_real_template(self):
        nodes = required_node_types("workflows/flux_schnell_t2i_api.json")
        assert "CheckpointLoaderSimple" in nodes
        assert "CLIPTextEncode" in nodes

    def test_wan22_t2v_uses_empty_wan_latent(self):
        """The node that this server actually lacks (proven by a live 400)."""
        nodes = required_node_types("workflows/wan22_t2v_api.json")
        assert "EmptyWanLatentVideo" in nodes

    def test_missing_node_types_flags_absent_class(self):
        # Server exposes everything EXCEPT EmptyWanLatentVideo
        info = {n: {} for n in required_node_types("workflows/wan22_t2v_api.json")}
        del info["EmptyWanLatentVideo"]
        assert missing_node_types("workflows/wan22_t2v_api.json", info) == ["EmptyWanLatentVideo"]

    def test_missing_node_types_empty_when_all_present(self):
        tpl = "workflows/flux_schnell_t2i_api.json"
        info = {n: {} for n in required_node_types(tpl)}
        assert missing_node_types(tpl, info) == []

    def test_missing_node_types_none_when_server_unreachable(self):
        assert missing_node_types("workflows/wan22_t2v_api.json", None) == []


# ── _parse_template robustness (model-filename special characters) ────────────

class TestParseTemplateRobustness:
    def test_backslash_string_default_does_not_break_json(self, tmp_path, monkeypatch):
        """Windows model filenames contain backslashes (e.g. a sharded Gemma
        checkpoint 'subdir\\model-00001.safetensors'). String defaults must be
        injected AFTER json.loads, never via a raw pre-parse text-replace."""
        import pipeline.model_catalog as mc
        wf_file = tmp_path / "wf.json"
        wf_file.write_text(
            '{"1": {"class_type": "CheckpointLoaderSimple", '
            '"inputs": {"ckpt_name": "{{CHECKPOINT}}"}}}',
            encoding="utf-8",
        )
        monkeypatch.setattr(mc, "template_defaults",
                            lambda p: {"CHECKPOINT": r"subdir\model-00001.safetensors"})
        slots = mc.detect_model_slots(wf_file)   # must not raise JSONDecodeError
        assert len(slots) == 1
        assert slots[0].current == r"subdir\model-00001.safetensors"


# ── auto_template_workflow ────────────────────────────────────────────────────

class TestAutoTemplateWorkflow:
    def test_real_flux_export_fully_templated(self):
        raw = open("comfyUI_workflow/flux_schnell.json", encoding="utf-8").read()
        text, notes = auto_template_workflow(raw)
        for token in ("{{POSITIVE_PROMPT}}", "{{NEGATIVE_PROMPT}}", "{{WIDTH}}",
                      "{{HEIGHT}}", "{{SEED}}", "{{OUTPUT_PREFIX}}"):
            assert token in text, f"{token} missing"
        # Numeric tokens must be UNQUOTED
        assert '"{{SEED}}"' not in text
        assert '"{{WIDTH}}"' not in text

    def test_real_ltx_export_fully_templated(self):
        """Subgraph ids, primitive-node titles, prompt chains — the hard case."""
        raw = open("comfyUI_workflow/video_ltx2_3_i2v_API.json", encoding="utf-8").read()
        text, _ = auto_template_workflow(raw)
        for token in ("{{POSITIVE_PROMPT}}", "{{NEGATIVE_PROMPT}}", "{{INPUT_IMAGE}}",
                      "{{WIDTH}}", "{{HEIGHT}}", "{{FRAMES}}", "{{FPS}}",
                      "{{SEED}}", "{{OUTPUT_PREFIX}}"):
            assert token in text, f"{token} missing"

    def test_gui_format_rejected(self):
        with pytest.raises(ValueError, match="GUI-format"):
            auto_template_workflow('{"nodes": [], "links": []}')

    def test_invalid_json_rejected(self):
        with pytest.raises(ValueError, match="JSON"):
            auto_template_workflow("not json at all {")

    def test_result_is_valid_after_dummy_fill(self):
        raw = open("comfyUI_workflow/flux_schnell.json", encoding="utf-8").read()
        text, _ = auto_template_workflow(raw)
        for token in ("WIDTH", "HEIGHT", "SEED", "FRAMES", "FPS"):
            text = text.replace("{{" + token + "}}", "1")
        json.loads(text)   # must not raise


# ── save_uploaded_workflow ────────────────────────────────────────────────────

class TestSaveUploadedWorkflow:
    def test_saves_and_templates(self, tmp_path):
        raw = open("comfyUI_workflow/flux_schnell.json", encoding="utf-8").read()
        rel, notes = save_uploaded_workflow("My Custom Flow!.json", raw, workflows_dir=tmp_path)
        saved = list(tmp_path.glob("*.json"))
        assert len(saved) == 1
        assert "{{POSITIVE_PROMPT}}" in saved[0].read_text(encoding="utf-8")
        # Filename sanitised
        assert saved[0].name == "My_Custom_Flow_.json"

    def test_already_templated_saved_verbatim(self, tmp_path):
        raw = '{"1": {"inputs": {"text": "{{POSITIVE_PROMPT}}", "seed": {{SEED}}}}}'
        rel, notes = save_uploaded_workflow("tpl.json", raw, workflows_dir=tmp_path)
        assert (tmp_path / "tpl.json").read_text(encoding="utf-8") == raw
        assert any("as-is" in n for n in notes)

    def test_collision_gets_unique_name(self, tmp_path):
        raw = '{"1": {"inputs": {"text": "{{POSITIVE_PROMPT}}"}}}'
        save_uploaded_workflow("dup.json", raw, workflows_dir=tmp_path)
        save_uploaded_workflow("dup.json", raw, workflows_dir=tmp_path)
        names = sorted(p.name for p in tmp_path.glob("*.json"))
        assert names == ["dup.json", "dup_2.json"]
