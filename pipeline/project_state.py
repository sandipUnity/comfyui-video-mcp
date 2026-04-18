"""
ProjectState — full project data model with JSON persistence.

One ProjectState object lives in Streamlit session_state and is saved to
  projects/{project_name}.json
after every step so the UI can be refreshed or closed and resumed with zero loss.

Usage:
    # Create new
    proj = ProjectState.new("my_film", "A lonely robot discovers music")

    # Add scenes
    proj.add_scene(scene)

    # Update a scene
    proj.update_scene("scene_01", status="approved", approved_image_path="/path/to/img.png")

    # Save / load
    proj.save("projects/my_film.json")
    proj = ProjectState.load("projects/my_film.json")

    # Advance step
    proj.next_step()
"""

from __future__ import annotations

import json
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pipeline.scene_state import SceneState
from pipeline.style_inference import StyleDNA, Character


# ── Default workflow paths ─────────────────────────────────────────────────────
DEFAULT_T2I_WORKFLOW = "workflows/flux_schnell_t2i_api.json"
DEFAULT_I2V_WORKFLOW = "workflows/ltx23_i2v_api.json"


@dataclass
class ProjectState:
    # ── Meta ────────────────────────────────────────────────────────────────────
    project_id:   str
    project_name: str
    created_at:   str   # ISO 8601 UTC
    updated_at:   str   # ISO 8601 UTC

    # ── Step 1 — Idea ───────────────────────────────────────────────────────────
    idea:             str
    duration_seconds: int = 30          # 15 | 30 | 60
    mood:             Optional[str] = None

    # ── Step 2 — Style ──────────────────────────────────────────────────────────
    style_dna: Optional[StyleDNA] = None

    # ── Step 3 — Story ──────────────────────────────────────────────────────────
    story_options:         list[dict] = field(default_factory=list)
    selected_story_index:  Optional[int] = None

    # ── Step 3.5 — Character ────────────────────────────────────────────────────
    character:   Optional[Character] = None
    global_seed: int = field(default_factory=lambda: random.randint(0, 2**31 - 1))

    # ── Step 4-6 — Scenes ───────────────────────────────────────────────────────
    scenes: list[SceneState] = field(default_factory=list)

    # ── Step 7 — Technical config ────────────────────────────────────────────────
    workflow_t2i:    str = DEFAULT_T2I_WORKFLOW
    workflow_i2v:    str = DEFAULT_I2V_WORKFLOW
    width:           int = 1280
    height:          int = 720
    frames:          int = 121
    fps:             int = 25
    t2i_width:       int = 1024    # T2I uses square by default
    t2i_height:      int = 1024
    images_per_scene: int = 1      # 1-5

    # ── Navigation ──────────────────────────────────────────────────────────────
    current_step: int = 1           # 1-10

    # ── Output ──────────────────────────────────────────────────────────────────
    output_dir: str = ""            # set at creation: "output/{project_id}/"

    # ── Factory ─────────────────────────────────────────────────────────────────

    @classmethod
    def new(cls, project_name: str, idea: str, duration_seconds: int = 30,
            mood: str | None = None) -> "ProjectState":
        """Create a fresh ProjectState with a new UUID and timestamps."""
        now = _now()
        pid = str(uuid.uuid4())
        return cls(
            project_id=pid,
            project_name=project_name,
            created_at=now,
            updated_at=now,
            idea=idea,
            duration_seconds=duration_seconds,
            mood=mood,
            output_dir=f"output/{pid}/",
        )

    # ── Persistence ─────────────────────────────────────────────────────────────

    def save(self, path: str | Path | None = None) -> Path:
        """Serialise to JSON. If path is None, writes to projects/{project_name}.json."""
        if path is None:
            out = Path("projects") / f"{self.project_name}.json"
        else:
            out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = _now()
        out.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str | Path) -> "ProjectState":
        """Restore a ProjectState from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    # ── Scene helpers ────────────────────────────────────────────────────────────

    def add_scene(self, scene: SceneState) -> None:
        """Append a scene. Raises if scene_id already exists."""
        if any(s.scene_id == scene.scene_id for s in self.scenes):
            raise ValueError(f"Scene '{scene.scene_id}' already exists.")
        self.scenes.append(scene)

    def get_scene(self, scene_id: str) -> SceneState:
        """Return the scene with the given scene_id. Raises KeyError if not found."""
        for s in self.scenes:
            if s.scene_id == scene_id:
                return s
        raise KeyError(f"Scene '{scene_id}' not found.")

    def get_scene_by_number(self, number: int) -> SceneState:
        """Return scene by 1-based scene_number."""
        for s in self.scenes:
            if s.scene_number == number:
                return s
        raise KeyError(f"Scene number {number} not found.")

    def update_scene(self, scene_id: str, **kwargs) -> SceneState:
        """Update one or more fields on a scene in-place. Returns the updated scene."""
        scene = self.get_scene(scene_id)
        for key, value in kwargs.items():
            if not hasattr(scene, key):
                raise AttributeError(f"SceneState has no attribute '{key}'")
            setattr(scene, key, value)
        return scene

    def remove_scene(self, scene_id: str) -> None:
        self.scenes = [s for s in self.scenes if s.scene_id != scene_id]

    def clear_scenes(self) -> None:
        self.scenes.clear()

    # ── Navigation ───────────────────────────────────────────────────────────────

    def next_step(self) -> int:
        """Advance current_step by 1 (max 10). Returns new step number."""
        self.current_step = min(self.current_step + 1, 10)
        return self.current_step

    def goto_step(self, step: int) -> None:
        if not 1 <= step <= 10:
            raise ValueError(f"Step must be 1-10, got {step}")
        self.current_step = step

    # ── Derived properties ────────────────────────────────────────────────────────

    @property
    def scene_count(self) -> int:
        return len(self.scenes)

    @property
    def expected_scene_count(self) -> int:
        """Scenes needed to fill duration_seconds at ~5 s/scene."""
        return max(1, self.duration_seconds // 5)

    @property
    def approved_scene_count(self) -> int:
        return sum(1 for s in self.scenes if s.is_approved)

    @property
    def all_approved(self) -> bool:
        return self.scene_count > 0 and self.approved_scene_count == self.scene_count

    @property
    def done_scene_count(self) -> int:
        return sum(1 for s in self.scenes if s.is_done)

    @property
    def all_done(self) -> bool:
        return self.scene_count > 0 and self.done_scene_count == self.scene_count

    @property
    def selected_story(self) -> dict | None:
        if self.selected_story_index is not None and self.story_options:
            idx = self.selected_story_index
            if 0 <= idx < len(self.story_options):
                return self.story_options[idx]
        return None

    # ── Serialisation ─────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "project_id":           self.project_id,
            "project_name":         self.project_name,
            "created_at":           self.created_at,
            "updated_at":           self.updated_at,
            "idea":                 self.idea,
            "duration_seconds":     self.duration_seconds,
            "mood":                 self.mood,
            "style_dna":            self.style_dna.to_dict() if self.style_dna else None,
            "story_options":        list(self.story_options),
            "selected_story_index": self.selected_story_index,
            "character":            self.character.to_dict() if self.character else None,
            "global_seed":          self.global_seed,
            "scenes":               [s.to_dict() for s in self.scenes],
            "workflow_t2i":         self.workflow_t2i,
            "workflow_i2v":         self.workflow_i2v,
            "width":                self.width,
            "height":               self.height,
            "frames":               self.frames,
            "fps":                  self.fps,
            "t2i_width":            self.t2i_width,
            "t2i_height":           self.t2i_height,
            "images_per_scene":     self.images_per_scene,
            "current_step":         self.current_step,
            "output_dir":           self.output_dir,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProjectState":
        from pipeline.style_inference import StyleDNA, Character
        style_dna = StyleDNA.from_dict(data["style_dna"]) if data.get("style_dna") else None
        character = Character.from_dict(data["character"]) if data.get("character") else None
        scenes    = [SceneState.from_dict(s) for s in data.get("scenes", [])]
        return cls(
            project_id           = data["project_id"],
            project_name         = data["project_name"],
            created_at           = data["created_at"],
            updated_at           = data["updated_at"],
            idea                 = data["idea"],
            duration_seconds     = data.get("duration_seconds", 30),
            mood                 = data.get("mood"),
            style_dna            = style_dna,
            story_options        = list(data.get("story_options", [])),
            selected_story_index = data.get("selected_story_index"),
            character            = character,
            global_seed          = data.get("global_seed", random.randint(0, 2**31 - 1)),
            scenes               = scenes,
            workflow_t2i         = data.get("workflow_t2i", DEFAULT_T2I_WORKFLOW),
            workflow_i2v         = data.get("workflow_i2v", DEFAULT_I2V_WORKFLOW),
            width                = data.get("width",  1280),
            height               = data.get("height", 720),
            frames               = data.get("frames", 121),
            fps                  = data.get("fps",    25),
            t2i_width            = data.get("t2i_width",  1024),
            t2i_height           = data.get("t2i_height", 1024),
            images_per_scene     = data.get("images_per_scene", 1),
            current_step         = data.get("current_step", 1),
            output_dir           = data.get("output_dir", ""),
        )

    def __repr__(self) -> str:
        return (
            f"ProjectState({self.project_name!r} step={self.current_step} "
            f"scenes={self.scene_count}/{self.expected_scene_count} "
            f"approved={self.approved_scene_count})"
        )


# ── Helper ────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
