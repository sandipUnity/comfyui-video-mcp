"""
Sprint 2 tests — Data layer: serialisation, persistence, style inference, engines.

Unit tests (no ComfyUI needed):
    python -m pytest tests/test_sprint2.py -v -m "not live"

Live integration tests (ComfyUI must be running):
    python -m pytest tests/test_sprint2.py -v -m live
"""

import asyncio
import json
import random
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent

# ── SceneState ─────────────────────────────────────────────────────────────────

class TestSceneState:
    def _make(self, **kwargs):
        from pipeline.scene_state import SceneState
        defaults = dict(scene_id="scene_01", scene_number=1, act="HOOK",
                        description="Opening shot")
        defaults.update(kwargs)
        return SceneState(**defaults)

    def test_create_minimal(self):
        s = self._make()
        assert s.scene_id == "scene_01"
        assert s.status == "pending"
        assert s.is_approved is False
        assert s.is_done is False

    def test_invalid_status_raises(self):
        from pipeline.scene_state import SceneState
        with pytest.raises(ValueError):
            SceneState(scene_id="s", scene_number=1, act="HOOK",
                       description="x", status="invalid_status")

    def test_set_status(self):
        s = self._make()
        s.set_status("approved")
        assert s.status == "approved"

    def test_set_status_invalid_raises(self):
        s = self._make()
        with pytest.raises(ValueError):
            s.set_status("not_a_real_status")

    def test_is_approved_property(self):
        s = self._make(approved_image_path="/tmp/img.png")
        assert s.is_approved is True

    def test_is_done_property(self):
        s = self._make(status="done")
        assert s.is_done is True

    def test_has_video_property(self):
        s = self._make(video_local_path="/tmp/video.mp4")
        assert s.has_video is True

    def test_round_trip_minimal(self):
        from pipeline.scene_state import SceneState
        s = self._make()
        restored = SceneState.from_dict(s.to_dict())
        assert restored.scene_id == s.scene_id
        assert restored.act == s.act
        assert restored.status == s.status

    def test_round_trip_full(self):
        from pipeline.scene_state import SceneState
        s = self._make(
            environment="desert",
            camera="3 ft/s dolly forward",
            lighting="3000K key light at 45°",
            base_prompt="Egyptian queen walks forward",
            visual_prompt="Egyptian queen walks forward, golden light, masterpiece",
            negative_prompt="cartoon, ugly",
            video_prompt="She walks steadily forward, sand swirling",
            seed=42,
            storyboard_images=["/tmp/img_v1.png", "/tmp/img_v2.png"],
            approved_image_path="/tmp/img_v1.png",
            server_image_filename="scene_01_uploaded.png",
            video_job_id="abc123",
            video_filename="LTX_scene_01.mp4",
            video_local_path="/tmp/LTX_scene_01.mp4",
            status="done",
        )
        d = s.to_dict()
        restored = SceneState.from_dict(d)
        assert restored.environment == "desert"
        assert restored.camera == "3 ft/s dolly forward"
        assert restored.storyboard_images == ["/tmp/img_v1.png", "/tmp/img_v2.png"]
        assert restored.approved_image_path == "/tmp/img_v1.png"
        assert restored.server_image_filename == "scene_01_uploaded.png"
        assert restored.video_job_id == "abc123"
        assert restored.video_local_path == "/tmp/LTX_scene_01.mp4"
        assert restored.status == "done"
        assert restored.seed == 42

    def test_to_dict_is_json_serialisable(self):
        s = self._make(storyboard_images=["/tmp/a.png"])
        json.dumps(s.to_dict())  # must not raise

    def test_scene_seed_deterministic(self):
        from pipeline.scene_state import _scene_seed
        s1 = _scene_seed(global_seed=999, scene_number=1)
        s2 = _scene_seed(global_seed=999, scene_number=1)
        s3 = _scene_seed(global_seed=999, scene_number=2)
        assert s1 == s2          # same inputs → same seed
        assert s1 != s3          # different scene_number → different seed

    def test_repr(self):
        s = self._make()
        r = repr(s)
        assert "HOOK" in r
        assert "pending" in r


# ── ProjectState ───────────────────────────────────────────────────────────────

class TestProjectState:
    def _make(self):
        from pipeline.project_state import ProjectState
        return ProjectState.new("test_project", "A robot discovers music")

    def _make_scene(self, number: int = 1):
        from pipeline.scene_state import SceneState
        return SceneState(
            scene_id=f"scene_{number:02d}",
            scene_number=number,
            act="HOOK",
            description=f"Scene {number}",
        )

    def test_new_creates_uuid(self):
        p = self._make()
        assert len(p.project_id) == 36  # UUID format
        assert p.project_name == "test_project"
        assert p.current_step == 1

    def test_new_sets_timestamps(self):
        p = self._make()
        assert p.created_at.endswith("Z")
        assert p.updated_at.endswith("Z")

    def test_expected_scene_count(self):
        from pipeline.project_state import ProjectState
        p = ProjectState.new("x", "y", duration_seconds=30)
        assert p.expected_scene_count == 6

        p2 = ProjectState.new("x", "y", duration_seconds=15)
        assert p2.expected_scene_count == 3

    def test_add_scene(self):
        p = self._make()
        p.add_scene(self._make_scene(1))
        assert p.scene_count == 1

    def test_add_duplicate_scene_raises(self):
        p = self._make()
        p.add_scene(self._make_scene(1))
        with pytest.raises(ValueError):
            p.add_scene(self._make_scene(1))  # same scene_id

    def test_get_scene(self):
        p = self._make()
        p.add_scene(self._make_scene(1))
        s = p.get_scene("scene_01")
        assert s.scene_number == 1

    def test_get_scene_missing_raises(self):
        p = self._make()
        with pytest.raises(KeyError):
            p.get_scene("scene_99")

    def test_get_scene_by_number(self):
        p = self._make()
        p.add_scene(self._make_scene(3))
        s = p.get_scene_by_number(3)
        assert s.scene_id == "scene_03"

    def test_update_scene(self):
        p = self._make()
        p.add_scene(self._make_scene(1))
        p.update_scene("scene_01", status="approved",
                       approved_image_path="/tmp/img.png")
        s = p.get_scene("scene_01")
        assert s.status == "approved"
        assert s.approved_image_path == "/tmp/img.png"

    def test_update_invalid_field_raises(self):
        p = self._make()
        p.add_scene(self._make_scene(1))
        with pytest.raises(AttributeError):
            p.update_scene("scene_01", nonexistent_field="x")

    def test_remove_scene(self):
        p = self._make()
        p.add_scene(self._make_scene(1))
        p.add_scene(self._make_scene(2))
        p.remove_scene("scene_01")
        assert p.scene_count == 1
        assert p.scenes[0].scene_id == "scene_02"

    def test_all_approved_false_when_no_scenes(self):
        p = self._make()
        assert p.all_approved is False

    def test_all_approved_true(self):
        p = self._make()
        s = self._make_scene(1)
        s.approved_image_path = "/tmp/img.png"
        p.add_scene(s)
        assert p.all_approved is True

    def test_next_step(self):
        p = self._make()
        assert p.current_step == 1
        p.next_step()
        assert p.current_step == 2
        for _ in range(20):   # beyond max
            p.next_step()
        assert p.current_step == 10  # capped at 10

    def test_goto_step(self):
        p = self._make()
        p.goto_step(7)
        assert p.current_step == 7

    def test_goto_step_invalid_raises(self):
        p = self._make()
        with pytest.raises(ValueError):
            p.goto_step(11)

    def test_selected_story_none_when_unset(self):
        p = self._make()
        assert p.selected_story is None

    def test_selected_story_returns_correct_option(self):
        p = self._make()
        p.story_options = [{"summary": "Option A"}, {"summary": "Option B"}]
        p.selected_story_index = 1
        assert p.selected_story["summary"] == "Option B"

    def test_round_trip_empty(self):
        from pipeline.project_state import ProjectState
        p = self._make()
        restored = ProjectState.from_dict(p.to_dict())
        assert restored.project_id == p.project_id
        assert restored.idea == p.idea
        assert restored.current_step == 1

    def test_round_trip_with_scenes(self):
        from pipeline.project_state import ProjectState
        p = self._make()
        s = self._make_scene(1)
        s.visual_prompt = "a robot holding a violin"
        s.status = "approved"
        s.approved_image_path = "/tmp/img.png"
        p.add_scene(s)
        restored = ProjectState.from_dict(p.to_dict())
        assert restored.scene_count == 1
        assert restored.scenes[0].visual_prompt == "a robot holding a violin"
        assert restored.scenes[0].status == "approved"

    def test_to_dict_json_serialisable(self):
        p = self._make()
        p.add_scene(self._make_scene(1))
        json.dumps(p.to_dict())  # must not raise

    def test_save_and_load(self, tmp_path):
        from pipeline.project_state import ProjectState
        p = self._make()
        p.add_scene(self._make_scene(1))
        save_path = tmp_path / "test_proj.json"
        p.save(save_path)
        assert save_path.exists()
        loaded = ProjectState.load(save_path)
        assert loaded.project_id == p.project_id
        assert loaded.idea == p.idea
        assert loaded.scene_count == 1

    def test_save_updates_updated_at(self, tmp_path):
        from pipeline.project_state import ProjectState
        import time
        p = self._make()
        t1 = p.updated_at
        time.sleep(1.05)  # ensure timestamp changes
        p.save(tmp_path / "proj.json")
        assert p.updated_at != t1

    def test_repr(self):
        p = self._make()
        r = repr(p)
        assert "test_project" in r
        assert "step=1" in r


# ── StyleDNA & infer_style ─────────────────────────────────────────────────────

class TestStyleDNA:
    def test_infer_style_cinematic(self):
        from pipeline.style_inference import infer_style
        dna = infer_style("a dramatic film scene with noir lighting")
        assert dna.skill_id == "cinematic"
        assert len(dna.quality_boosters) > 0
        assert len(dna.negative_tags) > 0
        assert len(dna.camera_language) == 3

    def test_infer_style_anime(self):
        from pipeline.style_inference import infer_style
        # Use multiple anime keywords so the score is unambiguous
        dna = infer_style("an anime manga shonen Japanese cyberpunk style")
        assert dna.skill_id == "anime"
        assert dna.fps in (24, 30)

    def test_infer_style_override(self):
        from pipeline.style_inference import infer_style
        dna = infer_style("some generic idea", override_skill_id="food")
        assert dna.skill_id == "food"

    def test_infer_style_unknown_falls_back_to_cinematic(self):
        from pipeline.style_inference import infer_style
        dna = infer_style("xyzzy completely unknown gibberish")
        assert dna.skill_id == "cinematic"  # default fallback

    def test_infer_style_from_skill_id(self):
        from pipeline.style_inference import infer_style_from_skill_id
        dna = infer_style_from_skill_id("3d_cgi")
        assert dna.skill_id == "3d_cgi"
        assert dna.recommended_width == 768

    def test_infer_style_invalid_skill_id(self):
        from pipeline.style_inference import infer_style_from_skill_id
        with pytest.raises(ValueError):
            infer_style_from_skill_id("not_a_real_skill")

    def test_style_dna_round_trip(self):
        from pipeline.style_inference import StyleDNA, infer_style
        dna = infer_style("a cinematic film")
        restored = StyleDNA.from_dict(dna.to_dict())
        assert restored.skill_id == dna.skill_id
        assert restored.quality_boosters == dna.quality_boosters
        assert restored.camera_language == dna.camera_language
        assert restored.fps == dna.fps
        assert restored.recommended_width == dna.recommended_width

    def test_style_dna_json_serialisable(self):
        from pipeline.style_inference import infer_style
        dna = infer_style("a product advertisement")
        json.dumps(dna.to_dict())  # must not raise

    def test_lighting_style_not_empty(self):
        from pipeline.style_inference import infer_style
        for idea in ["film", "anime", "food", "product"]:
            dna = infer_style(idea)
            assert dna.lighting_style != "", f"lighting_style empty for idea={idea!r}"

    def test_repr(self):
        from pipeline.style_inference import infer_style
        dna = infer_style("a cinematic film")
        assert "cinematic" in repr(dna)


class TestCharacter:
    def test_new_creates_uuid(self):
        from pipeline.style_inference import Character
        c = Character.new("Egyptian queen in blue and gold headdress", base_seed=42)
        assert len(c.id) == 36
        assert c.base_seed == 42

    def test_round_trip(self):
        from pipeline.style_inference import Character
        c = Character.new("A lone warrior in black armour", base_seed=100)
        restored = Character.from_dict(c.to_dict())
        assert restored.id == c.id
        assert restored.description == c.description
        assert restored.base_seed == c.base_seed

    def test_round_trip_with_reference_image(self):
        from pipeline.style_inference import Character
        c = Character.new("A chef in white uniform", base_seed=7)
        c.reference_image_path = "/tmp/chef.png"
        restored = Character.from_dict(c.to_dict())
        assert restored.reference_image_path == "/tmp/chef.png"


# ── ProjectState with StyleDNA + Character ────────────────────────────────────

class TestProjectStateWithStyle:
    def test_save_load_with_style_dna(self, tmp_path):
        from pipeline.project_state import ProjectState
        from pipeline.style_inference import infer_style
        p = ProjectState.new("styled_project", "A cinematic epic")
        p.style_dna = infer_style("dramatic film scene")
        p.save(tmp_path / "styled.json")
        loaded = ProjectState.load(tmp_path / "styled.json")
        assert loaded.style_dna is not None
        assert loaded.style_dna.skill_id == "cinematic"

    def test_save_load_with_character(self, tmp_path):
        from pipeline.project_state import ProjectState
        from pipeline.style_inference import Character
        p = ProjectState.new("char_project", "A warrior's journey")
        p.character = Character.new("Tall warrior with silver armour", base_seed=999)
        p.save(tmp_path / "char.json")
        loaded = ProjectState.load(tmp_path / "char.json")
        assert loaded.character is not None
        assert "silver armour" in loaded.character.description
        assert loaded.character.base_seed == 999


# ── fill_workflow (utils) ─────────────────────────────────────────────────────

class TestFillWorkflow:
    def test_t2i_fill(self):
        from pipeline.utils import fill_workflow
        wf = fill_workflow(
            ROOT / "workflows" / "flux_schnell_t2i_api.json",
            positive_prompt="a rose",
            negative_prompt="ugly",
            width=512, height=512, seed=1,
            output_prefix="test",
        )
        assert wf["6"]["inputs"]["text"] == "a rose"
        assert wf["33"]["inputs"]["text"] == "ugly"
        assert wf["27"]["inputs"]["width"] == 512
        assert wf["31"]["inputs"]["seed"] == 1
        assert wf["9"]["inputs"]["filename_prefix"] == "test"

    def test_i2v_fill(self):
        from pipeline.utils import fill_workflow
        wf = fill_workflow(
            ROOT / "workflows" / "ltx23_i2v_api.json",
            positive_prompt="she walks forward",
            negative_prompt="cartoon",
            width=1280, height=720,
            seed=42,
            output_prefix="video/test",
            input_image="uploaded.png",
            frames=121, fps=25,
        )
        assert wf["267:266"]["inputs"]["value"] == "she walks forward"
        assert wf["267:247"]["inputs"]["text"] == "cartoon"
        assert wf["269"]["inputs"]["image"] == "uploaded.png"
        assert wf["267:257"]["inputs"]["value"] == 1280
        assert wf["267:225"]["inputs"]["value"] == 121

    def test_special_chars_in_prompt(self):
        from pipeline.utils import fill_workflow
        tricky = 'She says: "Walk!" — then stops. C:\\Users\\path'
        wf = fill_workflow(
            ROOT / "workflows" / "flux_schnell_t2i_api.json",
            positive_prompt=tricky,
            width=512, height=512, seed=1,
            output_prefix="x",
        )
        assert wf["6"]["inputs"]["text"] == tricky

    def test_no_remaining_placeholders(self):
        from pipeline.utils import fill_workflow
        wf = fill_workflow(
            ROOT / "workflows" / "ltx23_i2v_api.json",
            positive_prompt="test",
            width=640, height=360, seed=99,
            output_prefix="p",
            input_image="img.png",
            frames=25, fps=25,
        )
        assert "{{" not in json.dumps(wf)


# ── Live integration tests ─────────────────────────────────────────────────────

@pytest.mark.live
class TestLiveT2IEngine:
    """Full T2I engine test against real ComfyUI."""

    @pytest.fixture
    def client(self):
        import yaml
        from comfyui_client import ComfyUIClient
        cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
        return ComfyUIClient(host=cfg["comfyui"]["host"], port=cfg["comfyui"]["port"])

    def test_generate_image(self, client, tmp_path):
        from pipeline.t2i_engine import generate_image

        async def run():
            path = await generate_image(
                client=client,
                prompt="a single red apple on a white marble table, photorealistic, studio lighting",
                negative_prompt="blurry, ugly",
                width=512, height=512,
                seed=42,
                output_prefix="sprint2_t2i",
                output_dir=tmp_path,
                timeout=180,
            )
            print(f"\n  Generated image: {path} ({path.stat().st_size} bytes)")
            assert path.exists()
            assert path.stat().st_size > 0
            return path

        asyncio.run(run())

    def test_generate_images_multi(self, client, tmp_path):
        from pipeline.t2i_engine import generate_images

        async def run():
            paths = await generate_images(
                client=client,
                prompt="a candle on a wooden table, soft warm light",
                width=512, height=512,
                base_seed=100,
                count=2,
                output_prefix="sprint2_multi",
                output_dir=tmp_path,
                timeout=180,
            )
            print(f"\n  Generated {len(paths)} images")
            assert len(paths) == 2
            for p in paths:
                assert p.exists()
                assert p.stat().st_size > 0

        asyncio.run(run())


@pytest.mark.live
class TestLiveI2VEngine:
    """Full I2V engine test against real ComfyUI."""

    @pytest.fixture
    def client(self):
        import yaml
        from comfyui_client import ComfyUIClient
        cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
        return ComfyUIClient(host=cfg["comfyui"]["host"], port=cfg["comfyui"]["port"])

    def _make_test_png(self, tmp_path: Path) -> Path:
        """Create a minimal valid PNG file (64×64 solid colour)."""
        import struct, zlib
        def make_png(w, h, rgb=(120, 80, 50)):
            def chunk(name, data):
                c = name + data
                return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
            raw = b""
            for _ in range(h):
                raw += b"\x00" + bytes(rgb) * w
            return (b"\x89PNG\r\n\x1a\n"
                    + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                    + chunk(b"IDAT", zlib.compress(raw))
                    + chunk(b"IEND", b""))
        img_path = tmp_path / "test_input.png"
        img_path.write_bytes(make_png(64, 64))
        return img_path

    def test_generate_video(self, client, tmp_path):
        from pipeline.i2v_engine import generate_video
        img_path = self._make_test_png(tmp_path)

        async def run():
            job_id, video_path = await generate_video(
                client=client,
                prompt="a terracotta pot on a wooden table, gentle breeze, leaves sway",
                image_path=img_path,
                width=640, height=360,
                frames=25, fps=25,
                seed=42,
                output_prefix="sprint2_i2v",
                output_dir=tmp_path / "video",
                timeout=600,
            )
            print(f"\n  job_id:     {job_id}")
            print(f"  video_path: {video_path} ({video_path.stat().st_size} bytes)")
            assert job_id
            assert video_path.exists()
            assert video_path.stat().st_size > 0

        asyncio.run(run())


@pytest.mark.live
class TestLiveFullPipeline:
    """T2I → approve → I2V full pipeline using ProjectState."""

    @pytest.fixture
    def client(self):
        import yaml
        from comfyui_client import ComfyUIClient
        cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
        return ComfyUIClient(host=cfg["comfyui"]["host"], port=cfg["comfyui"]["port"])

    def test_full_scene_lifecycle(self, client, tmp_path):
        from pipeline.project_state import ProjectState
        from pipeline.scene_state import SceneState
        from pipeline.style_inference import infer_style
        from pipeline.t2i_engine import generate_image
        from pipeline.i2v_engine import generate_video

        async def run():
            # Create project
            proj = ProjectState.new("sprint2_live", "A lone candle burns in a dark room")
            proj.style_dna = infer_style(proj.idea)
            proj.output_dir = str(tmp_path) + "/"

            # Create scene
            scene = SceneState(
                scene_id="scene_01", scene_number=1, act="HOOK",
                description="Close-up of a candle flame flickering in darkness",
                visual_prompt=(
                    "close-up of a single white candle with a bright flame, "
                    "deep darkness behind, warm 2800K light, photorealistic, masterpiece"
                ),
                negative_prompt="blurry, ugly, watermark",
                video_prompt=(
                    "the candle flame flickers gently, small smoke wisps rise, "
                    "wax melts slowly at the base"
                ),
                seed=42,
            )
            proj.add_scene(scene)

            # Step 5: Generate storyboard image
            scene.set_status("generating_image")
            proj.save(tmp_path / "proj.json")

            img_path = await generate_image(
                client=client,
                prompt=scene.visual_prompt,
                negative_prompt=scene.negative_prompt,
                width=512, height=512,
                seed=scene.seed,
                output_prefix=f"storyboard_{scene.scene_id}",
                output_dir=tmp_path / "storyboard",
                timeout=180,
            )

            scene.storyboard_images = [str(img_path)]
            scene.set_status("reviewing")
            proj.save(tmp_path / "proj.json")
            print(f"\n  Storyboard: {img_path}")

            # Step 5.5: Approve image (simulated)
            scene.approved_image_path = str(img_path)
            scene.set_status("approved")
            proj.save(tmp_path / "proj.json")

            # Step 8: Generate video
            scene.set_status("generating_video")
            job_id, video_path = await generate_video(
                client=client,
                prompt=scene.video_prompt,
                image_path=img_path,
                width=640, height=360,
                frames=25, fps=25,
                seed=scene.seed,
                output_prefix=f"video_{scene.scene_id}",
                output_dir=tmp_path / "video",
                timeout=600,
            )

            scene.video_job_id = job_id
            scene.video_local_path = str(video_path)
            scene.set_status("done")
            proj.save(tmp_path / "proj.json")
            print(f"  Video:      {video_path} ({video_path.stat().st_size} bytes)")

            # Reload and verify full fidelity
            from pipeline.project_state import ProjectState as PS
            reloaded = PS.load(tmp_path / "proj.json")
            rs = reloaded.get_scene("scene_01")
            assert rs.status == "done"
            assert rs.video_job_id == job_id
            assert Path(rs.video_local_path).exists()
            print("  Full pipeline lifecycle: PASSED")

        asyncio.run(run())
