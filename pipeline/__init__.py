"""
pipeline/ — Data layer and generation engines for the AI video pipeline.

Public API:
    from pipeline import SceneState, ProjectState, StyleDNA, Character
    from pipeline import infer_style
    from pipeline import generate_image, generate_video
    from pipeline.utils import fill_workflow
"""

from pipeline.scene_state    import SceneState
from pipeline.project_state  import ProjectState
from pipeline.style_inference import StyleDNA, Character, infer_style, infer_style_from_skill_id
from pipeline.t2i_engine     import generate_image, generate_images
from pipeline.i2v_engine     import generate_video
from pipeline.utils          import fill_workflow
from pipeline.model_checker  import check_model_availability, REQUIRED_MODELS
from pipeline.video_queue    import queue_video_job, get_all_statuses, download_completed_video
from pipeline.montage        import compile_montage, has_montage_support, available_backend

__all__ = [
    "SceneState",
    "ProjectState",
    "StyleDNA",
    "Character",
    "infer_style",
    "infer_style_from_skill_id",
    "generate_image",
    "generate_images",
    "generate_video",
    "fill_workflow",
    "check_model_availability",
    "REQUIRED_MODELS",
    "queue_video_job",
    "get_all_statuses",
    "download_completed_video",
    "compile_montage",
    "has_montage_support",
    "available_backend",
]
