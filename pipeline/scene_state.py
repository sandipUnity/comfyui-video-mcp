"""
SceneState — single source of truth for one scene in the pipeline.

Lifecycle of a scene:
  pending
    → generating_image   (T2I job queued)
    → reviewing          (images returned, waiting for user approval)
    → approved           (user approved an image)
    → uploading          (approved image being uploaded to ComfyUI)
    → generating_video   (I2V job queued)
    → done               (video downloaded locally)
    → failed             (any step failed — see .error for details)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional


# All valid status values — kept as a frozenset for validation
VALID_STATUSES = frozenset({
    "pending",
    "generating_image",
    "reviewing",
    "approved",
    "uploading",
    "generating_video",
    "done",
    "failed",
})


def _scene_seed(global_seed: int, scene_number: int) -> int:
    """Derive a deterministic per-scene seed from the project global seed."""
    raw = f"{global_seed}:{scene_number}".encode()
    return int(hashlib.sha256(raw).hexdigest(), 16) % (2**31)


@dataclass
class SceneState:
    # ── Identity ────────────────────────────────────────────────────────────────
    scene_id:     str   # "scene_01", "scene_02", ...
    scene_number: int   # 1-based

    # ── Story structure ─────────────────────────────────────────────────────────
    act:         str    # HOOK | BUILD | CLIMAX | RESOLUTION | CODA | etc.
    description: str    # one-line human-readable summary of this scene

    # ── Composition ─────────────────────────────────────────────────────────────
    environment:    str = ""
    camera_index:   int = 0   # index into skill.camera_vocabulary
    lighting_index: int = 0   # index into skill.lighting_vocabulary
    camera:         str = ""  # resolved camera description
    lighting:       str = ""  # resolved lighting description

    # ── Prompts ─────────────────────────────────────────────────────────────────
    base_prompt:     str = ""   # user-facing scene prompt (may contain {visual_anchor})
    visual_prompt:   str = ""   # full positive prompt after skill injection
    negative_prompt: str = ""   # full negative prompt after skill injection
    video_prompt:    str = ""   # motion-focused prompt for I2V (editable at step 6)

    # ── Seed ────────────────────────────────────────────────────────────────────
    seed: int = 0  # call scene_seed() to fill this at creation time

    # ── Storyboard phase ────────────────────────────────────────────────────────
    storyboard_images:    list[str] = field(default_factory=list)  # local file paths
    approved_image_path:  Optional[str] = None  # local path to approved image
    server_image_filename: Optional[str] = None  # filename on ComfyUI server after upload

    # ── Video phase ─────────────────────────────────────────────────────────────
    video_job_id:    Optional[str] = None  # ComfyUI prompt_id
    video_filename:  Optional[str] = None  # output filename on ComfyUI server
    video_local_path: Optional[str] = None  # downloaded to local output/

    # ── Status ──────────────────────────────────────────────────────────────────
    status: str = "pending"
    error:  Optional[str] = None

    # ── Validation ──────────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{self.status}'. Must be one of {VALID_STATUSES}")

    def set_status(self, status: str, error: str | None = None) -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{status}'")
        self.status = status
        self.error = error

    # ── Convenience predicates ──────────────────────────────────────────────────

    @property
    def is_approved(self) -> bool:
        return self.approved_image_path is not None

    @property
    def is_done(self) -> bool:
        return self.status == "done"

    @property
    def is_failed(self) -> bool:
        return self.status == "failed"

    @property
    def has_video(self) -> bool:
        return self.video_local_path is not None

    # ── Serialisation ────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict."""
        return {
            "scene_id":             self.scene_id,
            "scene_number":         self.scene_number,
            "act":                  self.act,
            "description":          self.description,
            "environment":          self.environment,
            "camera_index":         self.camera_index,
            "lighting_index":       self.lighting_index,
            "camera":               self.camera,
            "lighting":             self.lighting,
            "base_prompt":          self.base_prompt,
            "visual_prompt":        self.visual_prompt,
            "negative_prompt":      self.negative_prompt,
            "video_prompt":         self.video_prompt,
            "seed":                 self.seed,
            "storyboard_images":    list(self.storyboard_images),
            "approved_image_path":  self.approved_image_path,
            "server_image_filename": self.server_image_filename,
            "video_job_id":         self.video_job_id,
            "video_filename":       self.video_filename,
            "video_local_path":     self.video_local_path,
            "status":               self.status,
            "error":                self.error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SceneState":
        """Restore from a previously serialised dict."""
        return cls(
            scene_id             = data["scene_id"],
            scene_number         = data["scene_number"],
            act                  = data["act"],
            description          = data["description"],
            environment          = data.get("environment", ""),
            camera_index         = data.get("camera_index", 0),
            lighting_index       = data.get("lighting_index", 0),
            camera               = data.get("camera", ""),
            lighting             = data.get("lighting", ""),
            base_prompt          = data.get("base_prompt", ""),
            visual_prompt        = data.get("visual_prompt", ""),
            negative_prompt      = data.get("negative_prompt", ""),
            video_prompt         = data.get("video_prompt", ""),
            seed                 = data.get("seed", 0),
            storyboard_images    = list(data.get("storyboard_images", [])),
            approved_image_path  = data.get("approved_image_path"),
            server_image_filename= data.get("server_image_filename"),
            video_job_id         = data.get("video_job_id"),
            video_filename       = data.get("video_filename"),
            video_local_path     = data.get("video_local_path"),
            status               = data.get("status", "pending"),
            error                = data.get("error"),
        )

    def __repr__(self) -> str:
        return (
            f"SceneState(#{self.scene_number} {self.act!r} "
            f"status={self.status!r} approved={self.is_approved})"
        )
