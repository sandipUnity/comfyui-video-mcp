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
from pipeline.workflow_catalog import (
    discover_workflows, workflows_for, resolve_workflow_path, WorkflowInfo,
    auto_template_workflow, save_uploaded_workflow, template_defaults,
)
from pipeline.model_catalog import (
    detect_model_slots, options_for_slot, apply_model_overrides, slot_status,
    required_node_types, missing_node_types, ModelSlot,
)
from pipeline.prompt_builder import (
    build_story_prompt, build_character_prompt,
    build_scene_prompts_prompt, build_single_scene_prompt,
)
from pipeline.response_parser import (
    parse_story_response, parse_character_response, parse_scene_prompts_response,
    parse_error_message, ParseError,
)
from pipeline.ai_bridge import (
    write_pending_job, read_result, list_pending_jobs,
    get_oldest_pending_job, write_done_job, mark_job_failed,
    job_status, pending_job_count, done_job_count_today,
)

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
    "discover_workflows",
    "workflows_for",
    "resolve_workflow_path",
    "WorkflowInfo",
    "auto_template_workflow",
    "save_uploaded_workflow",
    "template_defaults",
    "detect_model_slots",
    "options_for_slot",
    "apply_model_overrides",
    "slot_status",
    "required_node_types",
    "missing_node_types",
    "ModelSlot",
]
