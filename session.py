"""Session state management for the video pipeline."""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path


@dataclass
class VideoIdea:
    id: int
    title: str
    description: str
    style: str
    mood: str
    tags: list[str]
    created_at: float = field(default_factory=time.time)
    selected: bool = False


@dataclass
class Scene:
    id: int
    idea_id: int
    scene_number: int
    description: str
    visual_prompt: str
    negative_prompt: str
    duration: float = 3.0  # seconds
    generated: bool = False
    video_path: Optional[str] = None
    job_id: Optional[str] = None


@dataclass
class MontageJob:
    id: str
    title: str
    video_paths: list[str]
    output_path: Optional[str] = None
    status: str = "pending"  # pending | processing | complete | failed
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


class PipelineSession:
    """In-memory session state for the video pipeline."""

    def __init__(self):
        self.ideas: list[VideoIdea] = []
        self.scenes: list[Scene] = []
        self.generation_jobs: dict[str, dict] = {}  # job_id -> status
        self.montage_jobs: list[MontageJob] = []
        self.current_notes: str = ""
        self.selected_idea_id: Optional[int] = None
        self._idea_counter = 0
        self._scene_counter = 0

    def add_ideas(self, ideas: list[dict]) -> list[VideoIdea]:
        """Add new ideas to session, replacing old ones."""
        self.ideas = []
        self._idea_counter = 0
        result = []
        for idea_data in ideas:
            self._idea_counter += 1
            idea = VideoIdea(
                id=self._idea_counter,
                title=idea_data.get("title", f"Idea {self._idea_counter}"),
                description=idea_data.get("description", ""),
                style=idea_data.get("style", "cinematic"),
                mood=idea_data.get("mood", "neutral"),
                tags=idea_data.get("tags", []),
            )
            self.ideas.append(idea)
            result.append(idea)
        return result

    def get_idea(self, idea_id: int) -> Optional[VideoIdea]:
        return next((i for i in self.ideas if i.id == idea_id), None)

    def select_idea(self, idea_id: int) -> Optional[VideoIdea]:
        idea = self.get_idea(idea_id)
        if idea:
            for i in self.ideas:
                i.selected = False
            idea.selected = True
            self.selected_idea_id = idea_id
            self.scenes = []  # Clear old scenes
        return idea

    def add_scenes(self, scenes: list[dict], idea_id: int) -> list[Scene]:
        """Add scenes for a selected idea."""
        self.scenes = []
        self._scene_counter = 0
        result = []
        for s in scenes:
            self._scene_counter += 1
            scene = Scene(
                id=self._scene_counter,
                idea_id=idea_id,
                scene_number=s.get("scene_number", self._scene_counter),
                description=s.get("description", ""),
                visual_prompt=s.get("visual_prompt", ""),
                negative_prompt=s.get("negative_prompt", "ugly, blurry, low quality"),
                duration=s.get("duration", 3.0),
            )
            self.scenes.append(scene)
            result.append(scene)
        return result

    def get_scene(self, scene_id: int) -> Optional[Scene]:
        return next((s for s in self.scenes if s.id == scene_id), None)

    def update_job(self, job_id: str, status: dict):
        self.generation_jobs[job_id] = status

    def mark_scene_generated(self, scene_id: int, video_path: str, job_id: str):
        scene = self.get_scene(scene_id)
        if scene:
            scene.generated = True
            scene.video_path = video_path
            scene.job_id = job_id

    def get_generated_videos(self) -> list[str]:
        return [s.video_path for s in self.scenes if s.generated and s.video_path]

    def add_montage_job(self, title: str, video_paths: list[str]) -> MontageJob:
        import uuid
        job = MontageJob(
            id=str(uuid.uuid4())[:8],
            title=title,
            video_paths=video_paths,
        )
        self.montage_jobs.append(job)
        return job

    def to_summary(self) -> dict:
        return {
            "ideas_count": len(self.ideas),
            "selected_idea": self.selected_idea_id,
            "scenes_count": len(self.scenes),
            "generated_scenes": len([s for s in self.scenes if s.generated]),
            "montages_count": len(self.montage_jobs),
            "pending_jobs": len([j for j in self.generation_jobs.values()
                                 if j.get("status") in ("pending", "processing")]),
        }


# Global session instance
session = PipelineSession()
