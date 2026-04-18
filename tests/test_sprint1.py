"""
Sprint 1 end-to-end tests.

Tests covered:
  1. Workflow template validation   — both JSON files parse cleanly with placeholders
  2. fill_t2i_workflow()            — injects all params, result is valid JSON
  3. fill_i2v_workflow()            — injects all params including image filename
  4. ComfyUIClient.upload_image()   — uploads a PNG to ComfyUI, gets filename back
  5. ComfyUIClient.get_output_images() — extracts image list from mock history
  6. Full T2I pipeline              — T2I queue → wait → download images  (live, skipped if ComfyUI offline)
  7. Full I2V pipeline              — upload image → I2V queue → wait → download video (live, skipped if ComfyUI offline)

Run all unit tests (no ComfyUI needed):
    python -m pytest tests/test_sprint1.py -v -m "not live"

Run live integration tests (ComfyUI must be running):
    python -m pytest tests/test_sprint1.py -v -m live
"""

import io
import json
import random
import time
import asyncio
from pathlib import Path

import pytest

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
T2I_TEMPLATE  = ROOT / "workflows" / "flux_schnell_t2i_api.json"
I2V_TEMPLATE  = ROOT / "workflows" / "ltx23_i2v_api.json"

# ── Shared fill helpers (same parse-then-inject logic used in app.py) ──────────

def fill_t2i_workflow(
    positive_prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    seed: int | None = None,
    output_prefix: str = "t2i_test",
) -> dict:
    """Return a filled T2I workflow dict ready to queue."""
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    template = T2I_TEMPLATE.read_text(encoding="utf-8")
    # Step 1 — replace bare numeric placeholders
    for ph, val in {
        "{{WIDTH}}":  str(width),
        "{{HEIGHT}}": str(height),
        "{{SEED}}":   str(seed),
    }.items():
        template = template.replace(ph, val)
    # Step 2 — parse
    wf = json.loads(template)
    # Step 3 — inject strings
    str_map = {
        "{{POSITIVE_PROMPT}}": positive_prompt,
        "{{NEGATIVE_PROMPT}}": negative_prompt,
        "{{OUTPUT_PREFIX}}":   output_prefix,
    }
    _inject_strings(wf, str_map)
    return wf


def fill_i2v_workflow(
    positive_prompt: str,
    input_image: str,               # server-side filename from upload_image()
    negative_prompt: str = "pc game, console game, video game, cartoon, childish, ugly",
    width: int = 1280,
    height: int = 720,
    frames: int = 121,
    fps: int = 25,
    seed: int | None = None,
    output_prefix: str = "i2v_test",
) -> dict:
    """Return a filled I2V workflow dict ready to queue."""
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    template = I2V_TEMPLATE.read_text(encoding="utf-8")
    # Step 1 — replace bare numeric placeholders
    for ph, val in {
        "{{WIDTH}}":  str(width),
        "{{HEIGHT}}": str(height),
        "{{FRAMES}}": str(frames),
        "{{FPS}}":    str(fps),
        "{{SEED}}":   str(seed),
    }.items():
        template = template.replace(ph, val)
    # Step 2 — parse
    wf = json.loads(template)
    # Step 3 — inject strings
    str_map = {
        "{{POSITIVE_PROMPT}}": positive_prompt,
        "{{NEGATIVE_PROMPT}}": negative_prompt,
        "{{INPUT_IMAGE}}":     input_image,
        "{{OUTPUT_PREFIX}}":   output_prefix,
    }
    _inject_strings(wf, str_map)
    return wf


def _inject_strings(d: dict, str_map: dict) -> None:
    """Recursively replace string placeholder values in-place."""
    for k, v in d.items():
        if isinstance(v, dict):
            _inject_strings(v, str_map)
        elif isinstance(v, str) and v in str_map:
            d[k] = str_map[v]


# ── Unit tests ─────────────────────────────────────────────────────────────────

class TestWorkflowTemplates:
    """Validate that the raw template files exist and are valid after placeholder fill."""

    def test_t2i_template_file_exists(self):
        assert T2I_TEMPLATE.exists(), f"Missing: {T2I_TEMPLATE}"

    def test_i2v_template_file_exists(self):
        assert I2V_TEMPLATE.exists(), f"Missing: {I2V_TEMPLATE}"

    def test_t2i_template_fills_cleanly(self):
        wf = fill_t2i_workflow(
            positive_prompt="a red rose on a marble table, photorealistic",
            width=1024, height=1024, seed=42,
        )
        assert isinstance(wf, dict), "Workflow should be a dict"
        assert "6" in wf,  "Node 6 (positive CLIPTextEncode) must exist"
        assert "33" in wf, "Node 33 (negative CLIPTextEncode) must exist"
        assert "27" in wf, "Node 27 (EmptySD3LatentImage) must exist"
        assert "31" in wf, "Node 31 (KSampler) must exist"
        assert "9" in wf,  "Node 9 (SaveImage) must exist"

    def test_t2i_positive_prompt_injected(self):
        prompt = "A futuristic city with flying cars, cinematic lighting"
        wf = fill_t2i_workflow(positive_prompt=prompt, seed=1)
        assert wf["6"]["inputs"]["text"] == prompt

    def test_t2i_negative_prompt_injected(self):
        neg = "blurry, ugly, watermark"
        wf = fill_t2i_workflow(positive_prompt="x", negative_prompt=neg, seed=1)
        assert wf["33"]["inputs"]["text"] == neg

    def test_t2i_dimensions_injected(self):
        wf = fill_t2i_workflow(positive_prompt="x", width=768, height=512, seed=1)
        assert wf["27"]["inputs"]["width"]  == 768
        assert wf["27"]["inputs"]["height"] == 512

    def test_t2i_seed_injected_as_int(self):
        wf = fill_t2i_workflow(positive_prompt="x", seed=999)
        assert wf["31"]["inputs"]["seed"] == 999
        assert isinstance(wf["31"]["inputs"]["seed"], int)

    def test_t2i_output_prefix_injected(self):
        wf = fill_t2i_workflow(positive_prompt="x", seed=1, output_prefix="scene_01")
        assert wf["9"]["inputs"]["filename_prefix"] == "scene_01"

    def test_t2i_no_remaining_placeholders(self):
        wf = fill_t2i_workflow(positive_prompt="hello world", seed=1)
        wf_str = json.dumps(wf)
        assert "{{" not in wf_str, f"Unfilled placeholders in T2I workflow: {wf_str}"

    def test_t2i_special_chars_in_prompt(self):
        """Prompts with em-dashes, quotes, backslashes must not break parsing."""
        tricky = 'She says: "Walk forward" — then stops. Path: C:\\Users\\test'
        wf = fill_t2i_workflow(positive_prompt=tricky, seed=1)
        assert wf["6"]["inputs"]["text"] == tricky

    def test_i2v_template_fills_cleanly(self):
        wf = fill_i2v_workflow(
            positive_prompt="Egyptian queen walks through desert",
            input_image="scene_01.png",
            seed=42,
        )
        assert isinstance(wf, dict)
        assert "267:266" in wf, "Node 267:266 (Prompt) must exist"
        assert "267:247" in wf, "Node 267:247 (negative CLIPTextEncode) must exist"
        assert "267:257" in wf, "Node 267:257 (Width PrimitiveInt) must exist"
        assert "267:258" in wf, "Node 267:258 (Height PrimitiveInt) must exist"
        assert "267:225" in wf, "Node 267:225 (Length PrimitiveInt) must exist"
        assert "267:260" in wf, "Node 267:260 (FPS PrimitiveInt) must exist"
        assert "269"     in wf, "Node 269 (LoadImage) must exist"
        assert "75"      in wf, "Node 75 (SaveVideo) must exist"

    def test_i2v_positive_prompt_injected(self):
        prompt = "The queen walks forward, sand swirling around her feet"
        wf = fill_i2v_workflow(positive_prompt=prompt, input_image="img.png", seed=1)
        assert wf["267:266"]["inputs"]["value"] == prompt

    def test_i2v_negative_prompt_injected(self):
        neg = "cartoon, video game, ugly"
        wf = fill_i2v_workflow(positive_prompt="x", input_image="img.png", negative_prompt=neg, seed=1)
        assert wf["267:247"]["inputs"]["text"] == neg

    def test_i2v_image_filename_injected(self):
        wf = fill_i2v_workflow(positive_prompt="x", input_image="uploaded_scene_02.png", seed=1)
        assert wf["269"]["inputs"]["image"] == "uploaded_scene_02.png"

    def test_i2v_dimensions_injected(self):
        wf = fill_i2v_workflow(positive_prompt="x", input_image="img.png", width=1280, height=720, seed=1)
        assert wf["267:257"]["inputs"]["value"] == 1280
        assert wf["267:258"]["inputs"]["value"] == 720

    def test_i2v_frames_fps_injected(self):
        wf = fill_i2v_workflow(positive_prompt="x", input_image="img.png", frames=97, fps=24, seed=1)
        assert wf["267:225"]["inputs"]["value"] == 97
        assert wf["267:260"]["inputs"]["value"] == 24

    def test_i2v_seeds_injected_as_int(self):
        wf = fill_i2v_workflow(positive_prompt="x", input_image="img.png", seed=12345)
        assert wf["267:216"]["inputs"]["noise_seed"] == 12345
        assert wf["267:237"]["inputs"]["noise_seed"] == 12345
        assert isinstance(wf["267:216"]["inputs"]["noise_seed"], int)

    def test_i2v_output_prefix_injected(self):
        wf = fill_i2v_workflow(positive_prompt="x", input_image="img.png", seed=1, output_prefix="video/scene_01")
        assert wf["75"]["inputs"]["filename_prefix"] == "video/scene_01"

    def test_i2v_no_remaining_placeholders(self):
        wf = fill_i2v_workflow(
            positive_prompt="cinematic shot of a warrior",
            input_image="warrior.png",
            seed=99,
        )
        wf_str = json.dumps(wf)
        assert "{{" not in wf_str, f"Unfilled placeholders in I2V workflow: {wf_str}"

    def test_i2v_i2v_mode_default(self):
        """I2V mode should default to false (Image-to-Video, not Text-to-Video)."""
        wf = fill_i2v_workflow(positive_prompt="x", input_image="img.png", seed=1)
        assert wf["267:201"]["inputs"]["value"] is False


class TestGetOutputImages:
    """Unit tests for ComfyUIClient.get_output_images() — no network needed."""

    @pytest.fixture
    def client(self):
        from comfyui_client import ComfyUIClient
        return ComfyUIClient()

    def test_extracts_single_image(self, client):
        history = {
            "outputs": {
                "9": {
                    "images": [
                        {"filename": "ComfyUI_00001_.png", "subfolder": "", "type": "output"}
                    ]
                }
            }
        }
        result = asyncio.run(client.get_output_images(history))
        assert len(result) == 1
        assert result[0]["filename"] == "ComfyUI_00001_.png"
        assert result[0]["node_id"] == "9"

    def test_extracts_multiple_images(self, client):
        history = {
            "outputs": {
                "9": {
                    "images": [
                        {"filename": "img_1.png", "subfolder": "", "type": "output"},
                        {"filename": "img_2.png", "subfolder": "", "type": "output"},
                    ]
                }
            }
        }
        result = asyncio.run(client.get_output_images(history))
        assert len(result) == 2

    def test_returns_empty_for_no_images(self, client):
        history = {"outputs": {"75": {"videos": [{"filename": "vid.mp4"}]}}}
        result = asyncio.run(client.get_output_images(history))
        assert result == []

    def test_returns_empty_for_empty_history(self, client):
        result = asyncio.run(client.get_output_images({}))
        assert result == []


# ── Live integration tests (require ComfyUI at config host:port) ───────────────

@pytest.mark.live
class TestLiveT2I:
    """Full T2I pipeline: generate image → download it."""

    @pytest.fixture
    def client(self):
        import yaml
        from comfyui_client import ComfyUIClient
        cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
        return ComfyUIClient(
            host=cfg["comfyui"]["host"],
            port=cfg["comfyui"]["port"],
        )

    def test_comfyui_reachable(self, client):
        available = asyncio.run(client.is_available())
        assert available, "ComfyUI is not reachable — check host/port in config.yaml"

    def test_t2i_queue_and_download(self, client, tmp_path):
        wf = fill_t2i_workflow(
            positive_prompt="a single red apple on a white background, photorealistic",
            width=512, height=512,
            seed=42,
            output_prefix="sprint1_test",
        )

        async def run():
            prompt_id = await client.queue_prompt(wf)
            assert prompt_id, "Expected a non-empty prompt_id"
            print(f"\n  Queued T2I job: {prompt_id}")

            history = await client.wait_for_completion(prompt_id, timeout=180)
            assert history, "History should not be empty after completion"

            images = await client.get_output_images(history)
            assert len(images) > 0, "Expected at least one output image"

            saved = await client.download_outputs(history, tmp_path, prefix="t2i_sprint1")
            assert len(saved) > 0, "Expected downloaded files"
            for p in saved:
                assert p.exists()
                assert p.stat().st_size > 0
                print(f"  Downloaded: {p}")

        asyncio.run(run())


@pytest.mark.live
class TestLiveI2V:
    """Full I2V pipeline: upload image → generate video → download it."""

    @pytest.fixture
    def client(self):
        import yaml
        from comfyui_client import ComfyUIClient
        cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
        return ComfyUIClient(
            host=cfg["comfyui"]["host"],
            port=cfg["comfyui"]["port"],
        )

    def _make_test_png(self) -> bytes:
        """Create a minimal 64×64 solid-colour PNG in-memory (no Pillow needed)."""
        import struct, zlib
        def make_png(width, height, rgb=(180, 100, 60)):
            def chunk(name, data):
                c = name + data
                return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
            signature = b"\x89PNG\r\n\x1a\n"
            ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
            ihdr = chunk(b"IHDR", ihdr_data)
            raw = b""
            for _ in range(height):
                raw += b"\x00" + bytes(rgb) * width
            idat = chunk(b"IDAT", zlib.compress(raw))
            iend = chunk(b"IEND", b"")
            return signature + ihdr + idat + iend
        return make_png(64, 64)

    def test_upload_then_i2v(self, client, tmp_path):
        png_bytes = self._make_test_png()

        async def run():
            # 1. Upload image
            filename = await client.upload_image(png_bytes, "sprint1_test_input.png")
            assert filename, "upload_image() should return a non-empty filename"
            print(f"\n  Uploaded image as: {filename}")

            # 2. Build I2V workflow
            wf = fill_i2v_workflow(
                positive_prompt="a terracotta pot sits on a wooden table, gentle breeze, leaves sway",
                input_image=filename,
                width=640, height=360,
                frames=25,
                fps=25,
                seed=42,
                output_prefix="sprint1_i2v_test",
            )

            # 3. Queue and wait
            prompt_id = await client.queue_prompt(wf)
            assert prompt_id
            print(f"  Queued I2V job: {prompt_id}")

            history = await client.wait_for_completion(prompt_id, timeout=600)
            assert history

            # 4. Download outputs
            saved = await client.download_outputs(history, tmp_path, prefix="i2v_sprint1")
            assert len(saved) > 0, "Expected at least one video output"
            for p in saved:
                assert p.exists()
                assert p.stat().st_size > 0
                print(f"  Downloaded: {p}")

        asyncio.run(run())
