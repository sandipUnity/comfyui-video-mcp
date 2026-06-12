"""
Sprint 4 tests — Model checker, video queue helpers, montage compiler.

Unit tests (no ComfyUI needed):
    python -m pytest tests/test_sprint4.py -v -m "not live"

Live integration tests (ComfyUI must be running):
    python -m pytest tests/test_sprint4.py -v -m live
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures & helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_scene(scene_id="scene_01", scene_number=1, **kwargs):
    from pipeline.scene_state import SceneState
    defaults = dict(
        scene_id=scene_id,
        scene_number=scene_number,
        act="HOOK",
        description="Test scene",
        visual_prompt="a test image",
        negative_prompt="ugly",
        video_prompt="camera pans slowly",
        seed=42,
    )
    defaults.update(kwargs)
    return SceneState(**defaults)


def _make_project(tmp_path, **kwargs):
    from pipeline.project_state import ProjectState
    p = ProjectState.new(
        project_name="test_s4",
        idea="A robot discovers art",
        duration_seconds=15,
        mood="contemplative",
    )
    p.output_dir = str(tmp_path)
    for k, v in kwargs.items():
        setattr(p, k, v)
    return p


# ══════════════════════════════════════════════════════════════════════════════
# TestModelChecker
# ══════════════════════════════════════════════════════════════════════════════

class TestModelChecker:
    """Tests for pipeline/model_checker.py"""

    def test_required_models_has_five_entries(self):
        from pipeline.model_checker import REQUIRED_MODELS
        assert len(REQUIRED_MODELS) == 5

    def test_required_models_keys(self):
        from pipeline.model_checker import REQUIRED_MODELS
        expected = {"flux_schnell", "ltx_main", "ltx_lora", "ltx_upscaler", "gemma_text_encoder"}
        assert set(REQUIRED_MODELS.keys()) == expected

    def test_each_model_has_required_fields(self):
        from pipeline.model_checker import REQUIRED_MODELS
        required_fields = {
            "display_name", "filename", "node_class", "field",
            "download_url", "size_gb", "required_for", "pipeline",
        }
        for key, spec in REQUIRED_MODELS.items():
            missing = required_fields - set(spec.keys())
            assert not missing, f"Model '{key}' missing fields: {missing}"

    def test_pipeline_tags_are_valid(self):
        from pipeline.model_checker import REQUIRED_MODELS
        for key, spec in REQUIRED_MODELS.items():
            assert spec["pipeline"] in ("t2i", "i2v"), \
                f"Model '{key}' has unexpected pipeline tag: {spec['pipeline']}"

    def test_flux_schnell_is_t2i(self):
        from pipeline.model_checker import REQUIRED_MODELS
        assert REQUIRED_MODELS["flux_schnell"]["pipeline"] == "t2i"

    def test_ltx_models_are_i2v(self):
        from pipeline.model_checker import REQUIRED_MODELS
        for key in ("ltx_main", "ltx_lora", "ltx_upscaler", "gemma_text_encoder"):
            assert REQUIRED_MODELS[key]["pipeline"] == "i2v"

    def test_check_field_found(self):
        from pipeline.model_checker import _check_field
        node_def = {
            "input": {
                "required": {
                    "ckpt_name": [["modelA.safetensors", "modelB.safetensors"], {}]
                }
            }
        }
        assert _check_field(node_def, "ckpt_name", "modelA.safetensors") is True

    def test_check_field_not_found(self):
        from pipeline.model_checker import _check_field
        node_def = {
            "input": {
                "required": {
                    "ckpt_name": [["modelA.safetensors"], {}]
                }
            }
        }
        assert _check_field(node_def, "ckpt_name", "missing.safetensors") is False

    def test_check_field_optional_section(self):
        from pipeline.model_checker import _check_field
        node_def = {
            "input": {
                "optional": {
                    "lora_name": [["my_lora.safetensors"], {}]
                }
            }
        }
        assert _check_field(node_def, "lora_name", "my_lora.safetensors") is True

    def test_check_field_empty_node_def(self):
        from pipeline.model_checker import _check_field
        assert _check_field({}, "ckpt_name", "anything.safetensors") is False

    def test_all_i2v_models_ok(self):
        from pipeline.model_checker import all_i2v_models_ok, REQUIRED_MODELS
        # Build fake results where all i2v models are "ok"
        results = {k: {**v, "status": "ok"} for k, v in REQUIRED_MODELS.items()}
        assert all_i2v_models_ok(results) is True

    def test_all_i2v_models_ok_with_missing(self):
        from pipeline.model_checker import all_i2v_models_ok, REQUIRED_MODELS
        results = {k: {**v, "status": "ok"} for k, v in REQUIRED_MODELS.items()}
        results["ltx_main"]["status"] = "missing_file"
        assert all_i2v_models_ok(results) is False

    def test_all_t2i_models_ok(self):
        from pipeline.model_checker import all_t2i_models_ok, REQUIRED_MODELS
        results = {k: {**v, "status": "ok"} for k, v in REQUIRED_MODELS.items()}
        assert all_t2i_models_ok(results) is True

    def test_all_t2i_models_ok_with_missing(self):
        from pipeline.model_checker import all_t2i_models_ok, REQUIRED_MODELS
        results = {k: {**v, "status": "ok"} for k, v in REQUIRED_MODELS.items()}
        results["flux_schnell"]["status"] = "missing_node"
        assert all_t2i_models_ok(results) is False

    def test_check_availability_returns_all_keys_on_failure(self):
        """When ComfyUI is unreachable, all models should get status='unknown'."""
        from pipeline.model_checker import check_model_availability, REQUIRED_MODELS

        mock_client = MagicMock()
        mock_client.base_url = "http://nowhere:9999"

        results = asyncio.run(check_model_availability(mock_client))

        assert set(results.keys()) == set(REQUIRED_MODELS.keys())
        for key, info in results.items():
            assert info["status"] == "unknown"
            assert info["installed"] is False
            assert info["node_available"] is False

    def test_check_availability_detects_ok_model(self):
        """With a mocked object_info response, correctly-present model gets status='ok'."""
        from pipeline.model_checker import check_model_availability, REQUIRED_MODELS

        spec = REQUIRED_MODELS["flux_schnell"]
        fake_object_info = {
            spec["node_class"]: {
                "input": {
                    "required": {
                        spec["field"]: [[spec["filename"]], {}]
                    }
                }
            }
        }

        mock_client = MagicMock()
        mock_client.base_url = "http://localhost:8188"

        with patch("pipeline.model_checker._fetch_object_info",
                   new=AsyncMock(return_value=fake_object_info)):
            results = asyncio.run(check_model_availability(mock_client))

        assert results["flux_schnell"]["status"] == "ok"
        assert results["flux_schnell"]["installed"] is True
        assert results["flux_schnell"]["node_available"] is True

    def test_check_availability_detects_missing_file(self):
        """Node present but file not in options → status='missing_file'."""
        from pipeline.model_checker import check_model_availability, REQUIRED_MODELS

        spec = REQUIRED_MODELS["flux_schnell"]
        fake_object_info = {
            spec["node_class"]: {
                "input": {
                    "required": {
                        spec["field"]: [["some_other_model.safetensors"], {}]
                    }
                }
            }
        }

        mock_client = MagicMock()
        mock_client.base_url = "http://localhost:8188"

        with patch("pipeline.model_checker._fetch_object_info",
                   new=AsyncMock(return_value=fake_object_info)):
            results = asyncio.run(check_model_availability(mock_client))

        assert results["flux_schnell"]["status"] == "missing_file"
        assert results["flux_schnell"]["node_available"] is True
        assert results["flux_schnell"]["installed"] is False

    def test_check_availability_detects_missing_node(self):
        """Node class absent from object_info → status='missing_node'."""
        from pipeline.model_checker import check_model_availability

        fake_object_info = {}   # no nodes at all

        mock_client = MagicMock()
        mock_client.base_url = "http://localhost:8188"

        with patch("pipeline.model_checker._fetch_object_info",
                   new=AsyncMock(return_value=fake_object_info)):
            results = asyncio.run(check_model_availability(mock_client))

        for key, info in results.items():
            assert info["status"] == "missing_node"
            assert info["node_available"] is False


# ══════════════════════════════════════════════════════════════════════════════
# TestVideoQueue
# ══════════════════════════════════════════════════════════════════════════════

class TestVideoQueue:
    """Tests for pipeline/video_queue.py"""

    def test_queue_video_job_raises_if_no_image(self, tmp_path):
        from pipeline.video_queue import queue_video_job

        scene = _make_scene()
        scene.approved_image_path = str(tmp_path / "nonexistent.png")

        project = _make_project(tmp_path)

        mock_client = MagicMock()

        with pytest.raises(FileNotFoundError, match="Approved image not found"):
            asyncio.run(queue_video_job(mock_client, scene, project))

    def test_queue_video_job_uploads_and_queues(self, tmp_path):
        from pipeline.video_queue import queue_video_job

        # Create a fake approved image
        img_path = tmp_path / "scene_01.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        scene = _make_scene()
        scene.approved_image_path = str(img_path)
        scene.video_prompt = "camera pans left"
        scene.negative_prompt = "blurry"
        scene.seed = 12345

        project = _make_project(tmp_path)

        mock_client = AsyncMock()
        mock_client.upload_image = AsyncMock(return_value="uploaded_scene_01.png")
        mock_client.queue_prompt = AsyncMock(return_value="abc123-job-id")

        job_id = asyncio.run(queue_video_job(mock_client, scene, project))

        assert job_id == "abc123-job-id"
        mock_client.upload_image.assert_called_once()
        mock_client.queue_prompt.assert_called_once()

        # Check that the workflow dict was passed (not raw template)
        call_args = mock_client.queue_prompt.call_args
        workflow = call_args[0][0]
        assert isinstance(workflow, dict), "queue_prompt should receive a dict workflow"

    def test_get_job_status_running(self):
        from pipeline.video_queue import get_job_status

        mock_client = AsyncMock()
        mock_client.get_queue_status = AsyncMock(return_value={
            "queue_running": [[1, "target-job-id", {}, {}, []]],
            "queue_pending": [],
        })

        status = asyncio.run(get_job_status(mock_client, "target-job-id"))
        assert status == "running"

    def test_get_job_status_queued(self):
        from pipeline.video_queue import get_job_status

        mock_client = AsyncMock()
        mock_client.get_queue_status = AsyncMock(return_value={
            "queue_running": [],
            "queue_pending": [[2, "target-job-id", {}, {}, []]],
        })

        status = asyncio.run(get_job_status(mock_client, "target-job-id"))
        assert status == "queued"

    def test_get_job_status_done(self):
        from pipeline.video_queue import get_job_status

        mock_client = AsyncMock()
        mock_client.get_queue_status = AsyncMock(return_value={
            "queue_running": [],
            "queue_pending": [],
        })
        mock_client.get_history = AsyncMock(return_value={
            "status": {"status_str": "success", "completed": True}
        })

        status = asyncio.run(get_job_status(mock_client, "target-job-id"))
        assert status == "done"

    def test_get_job_status_failed(self):
        from pipeline.video_queue import get_job_status

        mock_client = AsyncMock()
        mock_client.get_queue_status = AsyncMock(return_value={
            "queue_running": [],
            "queue_pending": [],
        })
        mock_client.get_history = AsyncMock(return_value={
            "status": {"status_str": "error", "completed": False}
        })

        status = asyncio.run(get_job_status(mock_client, "target-job-id"))
        assert status == "failed"

    def test_get_job_status_unknown(self):
        from pipeline.video_queue import get_job_status

        mock_client = AsyncMock()
        mock_client.get_queue_status = AsyncMock(return_value={
            "queue_running": [],
            "queue_pending": [],
        })
        mock_client.get_history = AsyncMock(return_value={})  # empty history

        status = asyncio.run(get_job_status(mock_client, "missing-job-id"))
        assert status == "unknown"

    def test_get_job_status_on_exception(self):
        from pipeline.video_queue import get_job_status

        mock_client = AsyncMock()
        mock_client.get_queue_status = AsyncMock(side_effect=ConnectionError("refused"))

        status = asyncio.run(get_job_status(mock_client, "any-job-id"))
        assert status == "unknown"

    def test_get_all_statuses_not_queued(self, tmp_path):
        from pipeline.video_queue import get_all_statuses

        project = _make_project(tmp_path)
        scene = _make_scene()
        project.scenes = [scene]   # no video_job_id set

        mock_client = AsyncMock()
        mock_client.get_queue_status = AsyncMock(return_value={
            "queue_running": [], "queue_pending": [],
        })

        statuses = asyncio.run(get_all_statuses(mock_client, project))
        assert statuses["scene_01"] == "not_queued"

    def test_get_all_statuses_running(self, tmp_path):
        from pipeline.video_queue import get_all_statuses

        project = _make_project(tmp_path)
        scene = _make_scene()
        scene.video_job_id = "job-abc"
        project.scenes = [scene]

        mock_client = AsyncMock()
        mock_client.get_queue_status = AsyncMock(return_value={
            "queue_running": [[1, "job-abc", {}, {}, []]],
            "queue_pending": [],
        })

        statuses = asyncio.run(get_all_statuses(mock_client, project))
        assert statuses["scene_01"] == "running"

    def test_download_completed_video_returns_none_if_no_job(self, tmp_path):
        from pipeline.video_queue import download_completed_video

        scene = _make_scene()
        project = _make_project(tmp_path)

        mock_client = AsyncMock()
        result = asyncio.run(download_completed_video(mock_client, scene, project))
        assert result is None
        mock_client.get_history.assert_not_called()

    def test_download_completed_video_returns_none_on_error_status(self, tmp_path):
        from pipeline.video_queue import download_completed_video

        scene = _make_scene()
        scene.video_job_id = "job-fail"
        project = _make_project(tmp_path)

        mock_client = AsyncMock()
        mock_client.get_history = AsyncMock(return_value={
            "status": {"status_str": "error"}
        })

        result = asyncio.run(download_completed_video(mock_client, scene, project))
        assert result is None

    def test_download_completed_video_saves_to_output_dir(self, tmp_path):
        from pipeline.video_queue import download_completed_video

        scene = _make_scene()
        scene.video_job_id = "job-done"
        project = _make_project(tmp_path)

        fake_video = tmp_path / "video" / "scene_01_output.mp4"
        fake_video.parent.mkdir(parents=True)
        fake_video.write_bytes(b"fake video data")

        mock_client = AsyncMock()
        mock_client.get_history = AsyncMock(return_value={
            "status": {"status_str": "success", "completed": True}
        })
        mock_client.download_outputs = AsyncMock(return_value=[fake_video])

        result = asyncio.run(download_completed_video(mock_client, scene, project))

        assert result == fake_video
        mock_client.download_outputs.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
# TestMontage
# ══════════════════════════════════════════════════════════════════════════════

class TestMontage:
    """Tests for pipeline/montage.py"""

    def test_has_montage_support_returns_bool(self):
        from pipeline.montage import has_montage_support
        result = has_montage_support()
        assert isinstance(result, bool)

    def test_available_backend_returns_valid_string(self):
        from pipeline.montage import available_backend
        result = available_backend()
        assert result in ("moviepy", "ffmpeg", "none")

    def test_available_backend_consistent_with_has_montage_support(self):
        from pipeline.montage import available_backend, has_montage_support
        backend = available_backend()
        has_support = has_montage_support()
        if backend == "none":
            assert has_support is False
        else:
            assert has_support is True

    def test_compile_montage_raises_on_no_valid_files(self, tmp_path):
        from pipeline.montage import compile_montage

        with pytest.raises(RuntimeError, match="No valid video files"):
            compile_montage(
                video_paths=[tmp_path / "nonexistent.mp4"],
                output_path=tmp_path / "out.mp4",
            )

    def test_compile_montage_single_clip_copies(self, tmp_path):
        from pipeline.montage import compile_montage

        src = tmp_path / "clip.mp4"
        src.write_bytes(b"fake mp4 data")
        out = tmp_path / "output" / "final.mp4"

        result = compile_montage(
            video_paths=[src],
            output_path=out,
        )

        assert result == out
        assert out.exists()
        assert out.read_bytes() == b"fake mp4 data"

    def test_compile_montage_creates_parent_dirs(self, tmp_path):
        from pipeline.montage import compile_montage

        src = tmp_path / "clip.mp4"
        src.write_bytes(b"x")
        # Output in a deeply nested directory that doesn't exist yet
        out = tmp_path / "a" / "b" / "c" / "final.mp4"

        compile_montage(video_paths=[src], output_path=out)
        assert out.parent.exists()

    def test_compile_montage_raises_on_no_backend(self, tmp_path):
        """When no backend is available, should raise RuntimeError."""
        from pipeline import montage as montage_mod

        # Create two real (dummy) video files
        v1 = tmp_path / "v1.mp4"
        v2 = tmp_path / "v2.mp4"
        v1.write_bytes(b"fake1")
        v2.write_bytes(b"fake2")
        out = tmp_path / "out.mp4"

        with patch.object(montage_mod, "_has_moviepy", return_value=False), \
             patch.object(montage_mod, "_has_ffmpeg", return_value=False):
            with pytest.raises(RuntimeError, match="No video compilation backend"):
                montage_mod.compile_montage(
                    video_paths=[v1, v2],
                    output_path=out,
                )

    def test_ffmpeg_concat_simple_called_on_cut_transition(self, tmp_path):
        """FFmpeg backend with 'cut' transition calls simple concat, not xfade."""
        from pipeline import montage as montage_mod

        v1 = tmp_path / "v1.mp4"
        v2 = tmp_path / "v2.mp4"
        v1.write_bytes(b"fake1")
        v2.write_bytes(b"fake2")
        out = tmp_path / "out.mp4"

        with patch.object(montage_mod, "_has_moviepy", return_value=False), \
             patch.object(montage_mod, "_has_ffmpeg", return_value=True), \
             patch.object(montage_mod, "_ffmpeg_concat_simple") as mock_simple, \
             patch.object(montage_mod, "_ffmpeg_concat_xfade") as mock_xfade:
            mock_simple.return_value = None   # don't actually run ffmpeg
            montage_mod.compile_montage(
                video_paths=[v1, v2],
                output_path=out,
                transition="cut",
            )

        mock_simple.assert_called_once()
        mock_xfade.assert_not_called()

    def test_ffmpeg_concat_xfade_called_on_dissolve(self, tmp_path):
        """FFmpeg backend with 'dissolve' transition calls xfade concat."""
        from pipeline import montage as montage_mod

        v1 = tmp_path / "v1.mp4"
        v2 = tmp_path / "v2.mp4"
        v1.write_bytes(b"fake1")
        v2.write_bytes(b"fake2")
        out = tmp_path / "out.mp4"
        out.write_bytes(b"")  # xfade mock won't create it

        with patch.object(montage_mod, "_has_moviepy", return_value=False), \
             patch.object(montage_mod, "_has_ffmpeg", return_value=True), \
             patch.object(montage_mod, "_ffmpeg_concat_xfade") as mock_xfade:
            mock_xfade.return_value = None
            montage_mod.compile_montage(
                video_paths=[v1, v2],
                output_path=out,
                transition="dissolve",
            )

        mock_xfade.assert_called_once()

    def test_ffmpeg_concat_xfade_falls_back_on_error(self, tmp_path):
        """If xfade raises, falls back to simple concat."""
        from pipeline import montage as montage_mod

        v1 = tmp_path / "v1.mp4"
        v2 = tmp_path / "v2.mp4"
        v1.write_bytes(b"fake1")
        v2.write_bytes(b"fake2")
        out = tmp_path / "out.mp4"

        with patch.object(montage_mod, "_has_moviepy", return_value=False), \
             patch.object(montage_mod, "_has_ffmpeg", return_value=True), \
             patch.object(montage_mod, "_ffmpeg_concat_xfade",
                          side_effect=RuntimeError("xfade unsupported")), \
             patch.object(montage_mod, "_ffmpeg_concat_simple") as mock_simple:
            mock_simple.return_value = None
            montage_mod.compile_montage(
                video_paths=[v1, v2],
                output_path=out,
                transition="dissolve",
            )

        mock_simple.assert_called_once()

    def test_get_duration_fallback_on_error(self, tmp_path):
        """_get_duration returns 5.0 when ffprobe fails."""
        from pipeline.montage import _get_duration

        result = _get_duration(tmp_path / "nonexistent.mp4")
        assert result == 5.0

    @pytest.mark.skipif(
        not shutil.which("ffprobe"),
        reason="ffprobe not on PATH",
    )
    def test_get_duration_with_real_file(self, tmp_path):
        """_get_duration returns a float for any file ffprobe accepts."""
        from pipeline.montage import _get_duration

        # Create a minimal 1-frame silent mp4 using ffmpeg
        out = tmp_path / "test.mp4"
        result = subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1",
             "-c:v", "libx264", str(out)],
            capture_output=True
        )
        if result.returncode == 0 and out.exists():
            dur = _get_duration(out)
            assert isinstance(dur, float)
            assert dur > 0


# ══════════════════════════════════════════════════════════════════════════════
# TestPipelineInit (Sprint 4 exports)
# ══════════════════════════════════════════════════════════════════════════════

class TestPipelineInitSprint4:
    """Verify all Sprint 4 symbols are exported from pipeline/__init__.py"""

    def test_check_model_availability_importable(self):
        from pipeline import check_model_availability
        assert callable(check_model_availability)

    def test_required_models_importable(self):
        from pipeline import REQUIRED_MODELS
        assert isinstance(REQUIRED_MODELS, dict)

    def test_queue_video_job_importable(self):
        from pipeline import queue_video_job
        assert callable(queue_video_job)

    def test_get_all_statuses_importable(self):
        from pipeline import get_all_statuses
        assert callable(get_all_statuses)

    def test_download_completed_video_importable(self):
        from pipeline import download_completed_video
        assert callable(download_completed_video)

    def test_compile_montage_importable(self):
        from pipeline import compile_montage
        assert callable(compile_montage)

    def test_has_montage_support_importable(self):
        from pipeline import has_montage_support
        assert callable(has_montage_support)

    def test_available_backend_importable(self):
        from pipeline import available_backend
        assert callable(available_backend)


# ══════════════════════════════════════════════════════════════════════════════
# Live integration tests (require running ComfyUI)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.live
class TestLiveModelCheck:
    """Live model check against a real ComfyUI instance."""

    def _client(self):
        import yaml
        from comfyui_client import ComfyUIClient
        cfg = yaml.safe_load(open(ROOT / "config.yaml"))
        return ComfyUIClient(cfg["comfyui"]["host"], cfg["comfyui"]["port"])

    def test_check_model_availability_returns_all_keys(self):
        from pipeline.model_checker import check_model_availability, REQUIRED_MODELS
        results = asyncio.run(check_model_availability(self._client()))
        assert set(results.keys()) == set(REQUIRED_MODELS.keys())

    def test_each_result_has_status(self):
        from pipeline.model_checker import check_model_availability
        results = asyncio.run(check_model_availability(self._client()))
        for key, info in results.items():
            assert info["status"] in ("ok", "missing_file", "missing_node", "unknown"), \
                f"Unexpected status for '{key}': {info['status']}"

    def test_results_are_json_serialisable(self):
        from pipeline.model_checker import check_model_availability
        results = asyncio.run(check_model_availability(self._client()))
        # Should not raise
        json.dumps(results)
