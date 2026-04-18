"""
ComfyUI AI Video Pipeline — 13-Step Wizard UI
Sprint 3: Steps 1-9  (Idea → Storyboard → Approved images + finalized video prompts)
Sprint 4: Steps 10-13 (Tech config → Queue → Monitor → Playback/Export)

Run:  venv\\Scripts\\streamlit run app.py
      then open  http://localhost:8501
"""

from __future__ import annotations

import asyncio
import os
import random
import sys
import time
from pathlib import Path

import streamlit as st
import yaml

sys.path.insert(0, ".")

from comfyui_client import ComfyUIClient
from pipeline import ProjectState, SceneState, infer_style, infer_style_from_skill_id, generate_image
from pipeline import StyleDNA
from pipeline import (
    check_model_availability, REQUIRED_MODELS,
    queue_video_job, get_all_statuses, download_completed_video,
    compile_montage, has_montage_support, available_backend,
)
from pipeline.style_inference import Character
from pipeline.story_generator import (
    generate_story_options,
    generate_scenes_from_story,
    generate_character_description,
)
from skills_engine import SKILLS, build_comfyui_positive, build_comfyui_negative

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = yaml.safe_load(open("config.yaml", encoding="utf-8"))
PROJECTS_DIR = Path("projects")
PROJECTS_DIR.mkdir(exist_ok=True)

# ── Step metadata ─────────────────────────────────────────────────────────────
STEPS = {
    1:  ("💡", "Idea",          "Enter your concept"),
    2:  ("🎨", "Style",         "Confirm visual style"),
    3:  ("📖", "Story",         "Pick a narrative"),
    4:  ("🧑", "Character",     "Lock the protagonist"),
    5:  ("📋", "Scenes",        "Edit scene breakdown"),
    6:  ("🖼️", "Storyboard",    "Generate images"),
    7:  ("✅", "Review",        "Approve images"),
    8:  ("📝", "Video Prompts", "Write motion prompts"),
    9:  ("🎞️", "Continuity",   "Final check"),
    10: ("⚙️", "Tech Config",   "Resolution & model check"),
    11: ("📤", "Queue",         "Send scenes to ComfyUI"),
    12: ("📡", "Monitor",       "Track progress & download"),
    13: ("🎬", "Playback",      "Watch & export montage"),
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Pipeline",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state init ────────────────────────────────────────────────────────
def _ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

_ss("project", None)            # ProjectState | None
_ss("gen_scene_idx", None)      # int | None — which scene is currently generating (storyboard)
_ss("story_options", None)      # list[dict] | None — cached story options
_ss("regen_scene_id", None)     # scene_id being regenerated in review step
_ss("model_check_results", None) # dict | None — cached model availability check
_ss("queue_scene_idx", None)    # int | None — which scene is currently being queued (Step 11)
_ss("montage_path", None)       # str | None — path to compiled montage (Step 13)

# ── Shortcuts ─────────────────────────────────────────────────────────────────
def proj() -> ProjectState | None:
    return st.session_state.project

def set_proj(p: ProjectState):
    st.session_state.project = p

def save():
    p = proj()
    if p:
        p.save()

def client() -> ComfyUIClient:
    return ComfyUIClient(CONFIG["comfyui"]["host"], CONFIG["comfyui"]["port"])

def goto(step: int):
    p = proj()
    if p:
        p.goto_step(step)
        save()
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🎬 AI Video Pipeline")
    st.divider()

    # ── Step progress ─────────────────────────────────────────────────────────
    p = proj()
    current = p.current_step if p else 1
    for step_num, (icon, label, hint) in STEPS.items():
        if step_num == current:
            st.markdown(f"**→ {icon} {step_num}. {label}**")
        elif p and step_num < current:
            # Completed steps are clickable for back-navigation
            if st.button(f"✓ {icon} {step_num}. {label}", key=f"nav_{step_num}",
                         use_container_width=True):
                goto(step_num)
        else:
            st.markdown(f"<span style='color:#666'>{icon} {step_num}. {label}</span>",
                        unsafe_allow_html=True)

    st.divider()

    # ── ComfyUI status ────────────────────────────────────────────────────────
    host = CONFIG["comfyui"]["host"]
    port = CONFIG["comfyui"]["port"]
    if st.button("⟳ Queue status", use_container_width=True):
        try:
            q = asyncio.run(client().get_queue_status())
            r = len(q.get("queue_running", []))
            pend = len(q.get("queue_pending", []))
            st.success(f"Online — {r} running, {pend} pending")
        except Exception as e:
            st.error(f"Offline: {e}")
    st.caption(f"Server: http://{host}:{port}")
    st.link_button("Open ComfyUI", f"http://{host}:{port}", use_container_width=True)

    st.divider()

    # ── Project management ────────────────────────────────────────────────────
    st.subheader("Projects")
    json_files = sorted(PROJECTS_DIR.glob("*.json"))
    json_names = [f.stem for f in json_files]

    if json_names:
        sel = st.selectbox("Load saved project", ["— new —"] + json_names)
        if sel != "— new —" and st.button("Load", use_container_width=True):
            loaded = ProjectState.load(PROJECTS_DIR / f"{sel}.json")
            set_proj(loaded)
            st.session_state.story_options = None
            st.session_state.gen_scene_idx = None
            st.rerun()
    else:
        st.info("No saved projects yet.")

    if p and st.button("💾 Save project", use_container_width=True):
        path = save()
        st.success(f"Saved → projects/{p.project_name}.json")

    if st.button("New project", use_container_width=True):
        set_proj(None)
        st.session_state.story_options = None
        st.session_state.gen_scene_idx = None
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — IDEA INPUT
# ══════════════════════════════════════════════════════════════════════════════

def step_1():
    st.header("💡 Step 1 — Your Idea")
    st.caption("Describe what you want to make. One sentence is enough.")

    with st.form("idea_form"):
        idea = st.text_area(
            "Concept *",
            placeholder="e.g. An Egyptian queen leads robot soldiers through a desert",
            height=100,
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            duration = st.select_slider(
                "Duration", options=[15, 30, 60], value=30,
                help="Number of 5-second scenes = duration ÷ 5",
            )
        with col2:
            project_name = st.text_input("Project name", value="my_film")
        with col3:
            mood = st.text_input("Mood (optional)", placeholder="e.g. epic, melancholic, playful")

        submitted = st.form_submit_button("Generate →", type="primary", use_container_width=True)

    if submitted:
        if not idea.strip():
            st.error("Please enter a concept — it can be as short as one sentence.")
            return

        p = ProjectState.new(
            project_name=project_name.strip() or "my_film",
            idea=idea.strip(),
            duration_seconds=duration,
            mood=mood.strip() or None,
        )
        set_proj(p)
        st.session_state.story_options = None
        p.next_step()       # → step 2
        save()
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — STYLE INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def step_2():
    p = proj()
    st.header("🎨 Step 2 — Visual Style")
    st.caption(f"Idea: *{p.idea}*")

    # Auto-infer on first visit
    if p.style_dna is None:
        with st.spinner("Detecting visual style…"):
            p.style_dna = infer_style(p.idea)
        save()

    dna: StyleDNA = p.style_dna

    # ── Display current style ─────────────────────────────────────────────────
    col_info, col_override = st.columns([3, 2])

    with col_info:
        st.subheader(f"{dna.skill_name}")
        st.write(f"**Visual style:** {dna.visual_style}")

        st.write("**Color palette:**")
        cols = st.columns(min(4, len(dna.color_palette)))
        for i, color in enumerate(dna.color_palette[:4]):
            cols[i].markdown(f"`{color}`")

        st.write("**Camera vocabulary** (first 3):")
        for cam in dna.camera_language[:3]:
            st.markdown(f"- {cam}")

        st.write("**Lighting style:**")
        st.markdown(f"- {dna.lighting_style}")

        st.write("**Recommended resolution:**",
                 f"{dna.recommended_width}×{dna.recommended_height} @ {dna.fps}fps")

    with col_override:
        st.subheader("Override style")
        skill_ids = list(SKILLS.keys())
        skill_opts = [f"{sid}  —  {SKILLS[sid].name}" for sid in skill_ids]
        cur_idx = skill_ids.index(dna.skill_id) if dna.skill_id in skill_ids else 0

        chosen = st.selectbox("Pick skill manually", ["(keep detected)"] + skill_opts)
        if st.button("Apply override", use_container_width=True):
            if chosen != "(keep detected)":
                override_id = chosen.split("  —  ")[0].strip()
                p.style_dna = infer_style_from_skill_id(override_id)
                save()
                st.rerun()

        if st.button("🔄 Re-detect from idea", use_container_width=True):
            p.style_dna = infer_style(p.idea)
            save()
            st.rerun()

    st.divider()
    col_back, col_next = st.columns([1, 4])
    with col_back:
        if st.button("← Back", use_container_width=True):
            goto(1)
    with col_next:
        if st.button("Accept this style →", type="primary", use_container_width=True):
            # Copy recommended resolution into project settings
            p.width    = dna.recommended_width
            p.height   = dna.recommended_height
            p.fps      = dna.fps
            p.next_step()   # → step 3
            save()
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — STORY OPTIONS
# ══════════════════════════════════════════════════════════════════════════════

def step_3():
    p = proj()
    st.header("📖 Step 3 — Story Options")
    st.caption(f"Idea: *{p.idea}*  ·  {p.expected_scene_count} scenes  ·  {p.duration_seconds}s")

    # Generate options if not yet done
    if st.session_state.story_options is None:
        with st.spinner("Generating story treatments…"):
            st.session_state.story_options = generate_story_options(
                p.idea, p.duration_seconds, p.mood
            )

    options: list[dict] = st.session_state.story_options

    col_regen, _ = st.columns([1, 4])
    with col_regen:
        if st.button("🔄 Regenerate all", use_container_width=True):
            st.session_state.story_options = None
            st.rerun()

    st.divider()

    # ── Display 3 story cards ─────────────────────────────────────────────────
    for i, opt in enumerate(options):
        with st.container(border=True):
            col_text, col_btn = st.columns([5, 1])
            with col_text:
                st.subheader(f"{i+1}. {opt.get('title', f'Option {i+1}')}")
                st.write(opt.get("summary", ""))
                st.markdown(f"**Arc:** {opt.get('arc', '')}")
                st.markdown(f"**Pacing:** *{opt.get('pacing', '')}*")
                st.caption(f"Why it works: {opt.get('reasoning', '')}")

                # Show act labels preview
                acts = opt.get("act_labels", [])
                st.markdown("**Scenes:** " + "  →  ".join(f"`{a}`" for a in acts))

            with col_btn:
                st.write("")  # vertical spacing
                st.write("")
                if st.button(f"Select →", key=f"story_{i}", type="primary",
                             use_container_width=True):
                    p.story_options = options
                    p.selected_story_index = i
                    p.next_step()   # → step 4 (character)
                    save()
                    st.rerun()

    st.divider()
    if st.button("← Back", use_container_width=True):
        goto(2)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — CHARACTER (UI step 3.5)
# ══════════════════════════════════════════════════════════════════════════════

def step_4():
    p = proj()
    st.header("🧑 Step 3.5 — Character")
    st.caption("Lock the protagonist's visual description. It will be injected into every scene prompt.")

    story = p.selected_story

    # Generate on first visit
    if p.character is None:
        with st.spinner("Generating protagonist description…"):
            desc = generate_character_description(p.idea, story or {}, p.mood)
        p.character = Character.new(desc, base_seed=p.global_seed)
        save()

    char = p.character

    st.info(
        "**Why this matters:** This exact description will prefix every image generation prompt. "
        "The more specific and visual it is, the more consistent your protagonist will look across scenes."
    )

    new_desc = st.text_area(
        "Protagonist description",
        value=char.description,
        height=150,
        help="Be specific: age, build, clothing, distinctive features, how they move.",
    )
    char.description = new_desc

    col_seed, col_regen = st.columns([2, 2])
    with col_seed:
        st.metric("Locked seed", char.base_seed,
                  help="This seed is used for all scene image generations for consistency.")
        new_seed = st.number_input("Override seed (0 = keep)", value=0, step=1)
        if new_seed != 0:
            char.base_seed = int(new_seed)
            p.global_seed  = int(new_seed)
    with col_regen:
        if st.button("🔄 Regenerate description", use_container_width=True):
            with st.spinner("Regenerating…"):
                desc = generate_character_description(p.idea, story or {}, p.mood)
            char.description = desc
            save()
            st.rerun()

    st.divider()
    col_back, col_next = st.columns([1, 4])
    with col_back:
        if st.button("← Back", use_container_width=True):
            goto(3)
    with col_next:
        if st.button("Accept character →", type="primary", use_container_width=True):
            p.character = char
            p.next_step()   # → step 5 (scenes)
            save()
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — SCENE BREAKDOWN (UI step 4)
# ══════════════════════════════════════════════════════════════════════════════

def step_5():
    p = proj()
    st.header("📋 Step 4 — Scene Breakdown")

    dna      = p.style_dna
    skill    = SKILLS.get(dna.skill_id, SKILLS["cinematic"])
    cam_opts = [f"{i}  {v[:60]}" for i, v in enumerate(skill.camera_vocabulary)]
    lit_opts = [f"{i}  {v[:60]}" for i, v in enumerate(skill.lighting_vocabulary)]

    # Generate scenes from story on first visit
    if not p.scenes:
        story = p.selected_story
        if story:
            with st.spinner("Building scene breakdown…"):
                scenes = generate_scenes_from_story(
                    p.idea, story, p.character, dna, p.global_seed,
                    p.duration_seconds, p.t2i_width, p.t2i_height,
                )
            p.scenes = scenes
            save()

    # ── Header controls ───────────────────────────────────────────────────────
    col_add, col_remove, col_reset, col_dur = st.columns([1, 1, 1, 3])
    with col_add:
        if st.button("➕ Add scene", use_container_width=True):
            n = len(p.scenes) + 1
            from pipeline.scene_state import _scene_seed as ss
            p.add_scene(SceneState(
                scene_id=f"scene_{n:02d}", scene_number=n,
                act="BUILD", description="New scene",
                camera_index=0, lighting_index=0,
                camera=skill.camera_vocabulary[0], lighting=skill.lighting_vocabulary[0],
                visual_prompt="", negative_prompt="",
                video_prompt="New scene", seed=ss(p.global_seed, n),
            ))
            save()
            st.rerun()

    with col_remove:
        if st.button("➖ Remove last", use_container_width=True) and len(p.scenes) > 1:
            last = p.scenes[-1]
            p.remove_scene(last.scene_id)
            save()
            st.rerun()

    with col_reset:
        if st.button("🔄 Regenerate all", use_container_width=True):
            story = p.selected_story
            if story:
                with st.spinner("Rebuilding…"):
                    p.clear_scenes()
                    scenes = generate_scenes_from_story(
                        p.idea, story, p.character, dna, p.global_seed,
                        p.duration_seconds, p.t2i_width, p.t2i_height,
                    )
                    p.scenes = scenes
                save()
                st.rerun()

    with col_dur:
        total_s = len(p.scenes) * 5
        st.metric("Total duration", f"{total_s}s", delta=f"{len(p.scenes)} scenes × 5s")

    st.divider()

    # ── Per-scene editor ──────────────────────────────────────────────────────
    for i, scene in enumerate(p.scenes):
        with st.expander(f"Scene {scene.scene_number}  ·  `{scene.act}`  ·  {scene.description[:60]}",
                         expanded=(i < 3)):

            r1c1, r1c2, r1c3 = st.columns([1, 1, 4])
            new_act = r1c1.text_input("Act label", value=scene.act, key=f"act_{i}")
            # Move up/down
            if r1c2.button("↑", key=f"up_{i}", disabled=(i == 0)):
                p.scenes[i], p.scenes[i-1] = p.scenes[i-1], p.scenes[i]
                # Re-number
                for j, s in enumerate(p.scenes):
                    s.scene_number = j + 1
                    s.scene_id = f"scene_{j+1:02d}"
                save()
                st.rerun()
            if r1c2.button("↓", key=f"dn_{i}", disabled=(i == len(p.scenes) - 1)):
                p.scenes[i], p.scenes[i+1] = p.scenes[i+1], p.scenes[i]
                for j, s in enumerate(p.scenes):
                    s.scene_number = j + 1
                    s.scene_id = f"scene_{j+1:02d}"
                save()
                st.rerun()
            new_desc = r1c3.text_input("Description", value=scene.description, key=f"desc_{i}")

            r2c1, r2c2 = st.columns(2)
            cam_cur = min(scene.camera_index, len(cam_opts) - 1)
            lit_cur = min(scene.lighting_index, len(lit_opts) - 1)
            chosen_cam = r2c1.selectbox("Camera", cam_opts, index=cam_cur, key=f"cam_{i}")
            chosen_lit = r2c2.selectbox("Lighting", lit_opts, index=lit_cur, key=f"lit_{i}")

            new_prompt = st.text_area("Image prompt (visual_prompt)", value=scene.visual_prompt,
                                      height=80, key=f"vp_{i}")

            # Apply edits on any change
            cam_idx = int(chosen_cam.split("  ")[0])
            lit_idx = int(chosen_lit.split("  ")[0])
            scene.act          = new_act
            scene.description  = new_desc
            scene.camera_index = cam_idx
            scene.lighting_index = lit_idx
            scene.camera       = skill.camera_vocabulary[cam_idx]
            scene.lighting     = skill.lighting_vocabulary[lit_idx]
            scene.visual_prompt = new_prompt
            if not scene.video_prompt or scene.video_prompt == scene.description:
                scene.video_prompt = new_desc  # keep in sync until step 6

    save()

    st.divider()
    col_back, col_next = st.columns([1, 4])
    with col_back:
        if st.button("← Back", use_container_width=True):
            goto(4)
    with col_next:
        if st.button("Confirm scenes →", type="primary", use_container_width=True):
            p.next_step()   # → step 6 (storyboard)
            save()
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — STORYBOARD GENERATION (UI step 5)
# ══════════════════════════════════════════════════════════════════════════════

def step_6():
    p = proj()
    st.header("🖼️ Step 5 — Storyboard Generation")
    st.caption("Generate reference images for each scene using Flux Schnell.")

    dna = p.style_dna
    col_cfg, col_go = st.columns([3, 2])

    with col_cfg:
        p.images_per_scene = st.slider(
            "Images per scene", 1, 5, value=p.images_per_scene,
            help="More images = more choice at the review step, but takes longer",
        )
        c1, c2 = st.columns(2)
        p.t2i_width  = c1.number_input("Image width",  value=p.t2i_width,  step=64, min_value=256)
        p.t2i_height = c2.number_input("Image height", value=p.t2i_height, step=64, min_value=256)

    with col_go:
        n_pending = sum(1 for s in p.scenes if not s.storyboard_images)
        n_done    = len(p.scenes) - n_pending

        st.metric("Progress", f"{n_done}/{len(p.scenes)} scenes",
                  delta="complete" if n_pending == 0 else f"{n_pending} pending")

        if n_pending > 0:
            if st.button("🖼️ Generate storyboard images", type="primary",
                         use_container_width=True):
                # Find first scene without images
                for i, s in enumerate(p.scenes):
                    if not s.storyboard_images:
                        st.session_state.gen_scene_idx = i
                        break
                save()
                st.rerun()

        if n_pending == 0:
            st.success("All scenes have images!")

    # ── Active generation (state machine: one scene per rerun) ────────────────
    if st.session_state.gen_scene_idx is not None:
        idx   = st.session_state.gen_scene_idx
        scene = p.scenes[idx]
        outdir = Path(p.output_dir) / "storyboard"

        progress_bar = st.progress(n_done / len(p.scenes),
                                   text=f"Generating scene {idx+1} of {len(p.scenes)}…")

        scene.set_status("generating_image")
        save()

        try:
            c = client()
            paths: list[Path] = []
            for v in range(p.images_per_scene):
                with st.spinner(f"Scene {idx+1} / {len(p.scenes)}  —  "
                                f"image {v+1} of {p.images_per_scene}…"):
                    path = asyncio.run(generate_image(
                        client=c,
                        prompt=scene.visual_prompt,
                        negative_prompt=scene.negative_prompt,
                        width=p.t2i_width,
                        height=p.t2i_height,
                        seed=(scene.seed + v) % (2**31),
                        output_prefix=f"{scene.scene_id}_v{v+1}",
                        output_dir=outdir,
                        timeout=180,
                    ))
                paths.append(path)

            scene.storyboard_images = [str(p2) for p2 in paths]
            scene.set_status("reviewing")

        except Exception as e:
            scene.set_status("failed", error=str(e))
            st.error(f"Scene {idx+1} failed: {e}")
            st.session_state.gen_scene_idx = None
            save()
            st.stop()

        # Find next pending scene
        next_idx = None
        for j in range(idx + 1, len(p.scenes)):
            if not p.scenes[j].storyboard_images:
                next_idx = j
                break

        st.session_state.gen_scene_idx = next_idx
        save()
        st.rerun()

    # ── Show current storyboard ────────────────────────────────────────────────
    st.divider()
    if any(s.storyboard_images for s in p.scenes):
        st.subheader("Current storyboard")
        cols = st.columns(min(len(p.scenes), 4))
        for i, scene in enumerate(p.scenes):
            col = cols[i % 4]
            with col:
                if scene.storyboard_images:
                    img_path = scene.storyboard_images[0]
                    if Path(img_path).exists():
                        st.image(img_path, caption=f"S{scene.scene_number} {scene.act}",
                                 use_container_width=True)
                else:
                    st.markdown(f"*S{scene.scene_number} — pending*")

    st.divider()
    col_back, col_next = st.columns([1, 4])
    with col_back:
        if st.button("← Back", use_container_width=True):
            goto(5)
    with col_next:
        all_have_images = all(s.storyboard_images for s in p.scenes)
        if st.button("Review images →", type="primary", use_container_width=True,
                     disabled=not all_have_images):
            p.next_step()   # → step 7 (review)
            save()
            st.rerun()
        if not all_have_images:
            st.caption("⚠ Generate images for all scenes before continuing.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — IMAGE REVIEW (UI step 5.5)
# ══════════════════════════════════════════════════════════════════════════════

def step_7():
    p = proj()
    st.header("✅ Step 5.5 — Image Review")
    st.caption("Approve one image per scene. All scenes must be approved to continue.")

    approved = p.approved_scene_count
    total    = p.scene_count
    st.progress(approved / max(total, 1), text=f"{approved}/{total} scenes approved")

    dna   = p.style_dna
    skill = SKILLS.get(dna.skill_id, SKILLS["cinematic"])

    # ── Regeneration in progress ──────────────────────────────────────────────
    regen_id = st.session_state.regen_scene_id
    if regen_id:
        scene = p.get_scene(regen_id)
        outdir = Path(p.output_dir) / "storyboard"
        scene.set_status("generating_image")
        save()

        try:
            c = client()
            paths: list[Path] = []
            for v in range(p.images_per_scene):
                with st.spinner(f"Regenerating {scene.scene_id} — image {v+1}/{p.images_per_scene}…"):
                    # Rebuild prompt if user changed it
                    new_seed = random.randint(0, 2**31 - 1)
                    path = asyncio.run(generate_image(
                        client=c,
                        prompt=scene.visual_prompt,
                        negative_prompt=scene.negative_prompt,
                        width=p.t2i_width,
                        height=p.t2i_height,
                        seed=new_seed + v,
                        output_prefix=f"{scene.scene_id}_regen_v{v+1}",
                        output_dir=outdir,
                        timeout=180,
                    ))
                paths.append(path)

            scene.storyboard_images = [str(p2) for p2 in paths]
            scene.approved_image_path = None
            scene.set_status("reviewing")
        except Exception as e:
            scene.set_status("failed", error=str(e))
            st.error(f"Regeneration failed: {e}")

        st.session_state.regen_scene_id = None
        save()
        st.rerun()

    # ── Per-scene review panels ───────────────────────────────────────────────
    for scene in p.scenes:
        is_approved = scene.is_approved
        border_style = "border: 2px solid #28a745;" if is_approved else ""

        with st.container(border=True):
            st.subheader(
                f"{'✅ ' if is_approved else ''}Scene {scene.scene_number}  ·  `{scene.act}`"
            )
            st.caption(scene.description)

            if scene.storyboard_images:
                # Show versions as tabs if multiple
                imgs = [img for img in scene.storyboard_images if Path(img).exists()]
                if not imgs:
                    st.warning("Image file not found on disk.")
                elif len(imgs) == 1:
                    col_img, col_actions = st.columns([3, 1])
                    with col_img:
                        st.image(imgs[0], use_container_width=True)
                else:
                    col_img, col_actions = st.columns([3, 1])
                    with col_img:
                        tab_labels = [f"Version {j+1}" for j in range(len(imgs))]
                        tabs = st.tabs(tab_labels)
                        for j, (tab, img_path) in enumerate(zip(tabs, imgs)):
                            with tab:
                                st.image(img_path, use_container_width=True)
                                if st.button(f"✅ Approve version {j+1}",
                                             key=f"approve_v{j}_{scene.scene_id}",
                                             type="primary"):
                                    p.update_scene(scene.scene_id,
                                                   approved_image_path=img_path,
                                                   status="approved")
                                    save()
                                    st.rerun()

                # Single-image actions (if not using version tabs)
                if len(imgs) == 1:
                    with col_actions:
                        st.write("")
                        if is_approved:
                            st.success("✅ Approved")
                            if st.button("Unapprove", key=f"unapprove_{scene.scene_id}"):
                                p.update_scene(scene.scene_id,
                                               approved_image_path=None,
                                               status="reviewing")
                                save()
                                st.rerun()
                        else:
                            if st.button("✅ Approve", key=f"approve_{scene.scene_id}",
                                         type="primary", use_container_width=True):
                                p.update_scene(scene.scene_id,
                                               approved_image_path=imgs[0],
                                               status="approved")
                                save()
                                st.rerun()

                        # Regenerate panel
                        with st.expander("🔄 Regenerate"):
                            new_vp = st.text_area(
                                "Update prompt (optional)",
                                value=scene.visual_prompt,
                                height=80,
                                key=f"regen_prompt_{scene.scene_id}",
                            )
                            if st.button("Regenerate now",
                                         key=f"regen_btn_{scene.scene_id}",
                                         use_container_width=True):
                                # Save updated prompt and queue regeneration
                                p.update_scene(scene.scene_id, visual_prompt=new_vp)
                                st.session_state.regen_scene_id = scene.scene_id
                                save()
                                st.rerun()
            else:
                st.warning("No images generated for this scene yet.")
                if st.button("Generate now", key=f"gen_now_{scene.scene_id}"):
                    # Find this scene's index and trigger generation
                    idx = next((i for i, s in enumerate(p.scenes)
                                if s.scene_id == scene.scene_id), None)
                    if idx is not None:
                        st.session_state.gen_scene_idx = idx
                    goto(6)

    st.divider()
    col_back, col_next = st.columns([1, 4])
    with col_back:
        if st.button("← Back to Storyboard", use_container_width=True):
            goto(6)
    with col_next:
        all_ok = p.all_approved
        if st.button("Finalise video prompts →", type="primary", use_container_width=True,
                     disabled=not all_ok):
            p.next_step()   # → step 8 (finalisation)
            save()
            st.rerun()
        if not all_ok:
            st.caption(f"⚠ {total - approved} scene(s) still need approval.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — SCENE FINALISATION (UI step 6)
# ══════════════════════════════════════════════════════════════════════════════

def step_8():
    p = proj()
    st.header("📝 Step 6 — Video Prompts")
    st.info(
        "**Describe motion and action.** The approved image defines what everything looks like — "
        "the video prompt only needs to say *what moves* and *how*. "
        "Keep it under 2 sentences."
    )

    for scene in p.scenes:
        with st.container(border=True):
            col_img, col_prompt = st.columns([1, 2])
            with col_img:
                if scene.approved_image_path and Path(scene.approved_image_path).exists():
                    st.image(scene.approved_image_path, use_container_width=True)
                st.caption(f"S{scene.scene_number} · `{scene.act}`")

            with col_prompt:
                st.write(f"**{scene.description}**")
                new_vp = st.text_area(
                    "Video prompt",
                    value=scene.video_prompt,
                    height=100,
                    key=f"vp_final_{scene.scene_id}",
                    help="Describe motion only. The image handles appearance.",
                    placeholder=(
                        "e.g. She walks steadily forward, sand swirling around her feet. "
                        "Robot soldiers march in formation on either side."
                    ),
                )
                scene.video_prompt = new_vp

    save()

    st.divider()
    col_back, col_next = st.columns([1, 4])
    with col_back:
        if st.button("← Back", use_container_width=True):
            goto(7)
    with col_next:
        if st.button("Continuity check →", type="primary", use_container_width=True):
            p.next_step()   # → step 9 (continuity)
            save()
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — CONTINUITY CHECK (UI step 6.5)
# ══════════════════════════════════════════════════════════════════════════════

def step_9():
    p = proj()
    st.header("🎞️ Step 6.5 — Continuity Check")
    st.caption("Review all approved images side by side. Do they feel like one coherent film?")

    # Check for any issues
    missing = [s for s in p.scenes if not s.approved_image_path
               or not Path(s.approved_image_path).exists()]
    missing_prompts = [s for s in p.scenes if not s.video_prompt.strip()]

    if missing:
        st.warning(
            f"⚠ {len(missing)} scene(s) have no approved image: "
            + ", ".join(f"S{s.scene_number}" for s in missing)
        )
    if missing_prompts:
        st.warning(
            f"⚠ {len(missing_prompts)} scene(s) have empty video prompts: "
            + ", ".join(f"S{s.scene_number}" for s in missing_prompts)
        )
    if not missing and not missing_prompts:
        st.success(f"✅ All {p.scene_count} scenes are approved with video prompts.")

    # ── Filmstrip ─────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Filmstrip — full continuity view")

    n_cols = min(len(p.scenes), 6)
    cols = st.columns(n_cols)
    for i, scene in enumerate(p.scenes):
        col = cols[i % n_cols]
        with col:
            if scene.approved_image_path and Path(scene.approved_image_path).exists():
                st.image(scene.approved_image_path, use_container_width=True)
            else:
                st.markdown("*no image*")
            st.caption(
                f"**S{scene.scene_number}** `{scene.act}`\n\n"
                f"{scene.description[:40]}…" if len(scene.description) > 40
                else f"**S{scene.scene_number}** `{scene.act}`\n\n{scene.description}"
            )

    # ── Video prompts summary ─────────────────────────────────────────────────
    st.divider()
    st.subheader("Video prompt summary")
    for scene in p.scenes:
        st.markdown(
            f"**S{scene.scene_number} `{scene.act}`** — {scene.video_prompt}"
        )

    st.divider()
    col_back, col_fix, col_next = st.columns([1, 1, 3])
    with col_back:
        if st.button("← Edit prompts", use_container_width=True):
            goto(8)
    with col_fix:
        if st.button("Fix images", use_container_width=True):
            goto(7)
    with col_next:
        if st.button("✅ Looks good — proceed to video generation →",
                     type="primary", use_container_width=True, disabled=bool(missing)):
            p.next_step()   # → step 10 (Sprint 4)
            save()
            st.rerun()
        if missing:
            st.caption("⚠ Resolve missing images before proceeding.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 — TECHNICAL CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

def step_10():
    p = proj()
    st.header("⚙️ Step 7 — Technical Configuration")
    st.caption("Set output resolution, verify required models, then confirm to queue videos.")

    # ── Resolution / frame settings ───────────────────────────────────────────
    st.subheader("Output settings")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        new_w = st.number_input("Width (px)", value=p.width, step=64, min_value=256, max_value=2048)
    with col2:
        new_h = st.number_input("Height (px)", value=p.height, step=64, min_value=256, max_value=2048)
    with col3:
        new_frames = st.number_input("Frames", value=p.frames, step=1, min_value=9, max_value=257)
    with col4:
        new_fps = st.number_input("FPS", value=p.fps, step=1, min_value=8, max_value=60)

    # Apply any changes live (doesn't persist until Confirm)
    p.width   = int(new_w)
    p.height  = int(new_h)
    p.frames  = int(new_frames)
    p.fps     = int(new_fps)

    dur_secs = new_frames / new_fps
    st.caption(
        f"Video duration per clip: **{dur_secs:.1f}s**  ·  "
        f"Total: ~**{dur_secs * p.scene_count:.0f}s** for {p.scene_count} scenes"
    )

    st.divider()

    # ── Model availability check ───────────────────────────────────────────────
    st.subheader("Model availability")

    col_check, col_clear = st.columns([2, 1])
    with col_check:
        if st.button("🔍 Check models on server", use_container_width=True):
            with st.spinner("Querying ComfyUI /object_info…"):
                try:
                    st.session_state.model_check_results = asyncio.run(
                        check_model_availability(client())
                    )
                except Exception as e:
                    st.error(f"Could not reach ComfyUI: {e}")
                    st.session_state.model_check_results = None
    with col_clear:
        if st.button("Clear results", use_container_width=True):
            st.session_state.model_check_results = None
            st.rerun()

    results = st.session_state.model_check_results
    if results is None:
        st.info("Click **Check models** to verify your ComfyUI installation before queuing.")
    else:
        # Tally
        ok_count      = sum(1 for v in results.values() if v["status"] == "ok")
        missing_count = len(results) - ok_count

        if missing_count == 0:
            st.success(f"✅ All {ok_count} required models are installed.")
        else:
            st.warning(f"⚠ {missing_count} model(s) missing — video generation may fail.")

        # Table of all models
        for key, info in results.items():
            status  = info["status"]
            icon    = "✅" if status == "ok" else ("❌" if status == "missing_file" else "⚠️")
            pipeline_tag = f"`{info.get('pipeline', '?').upper()}`"

            with st.expander(
                f"{icon} {info['display_name']}  ·  {pipeline_tag}  ·  {info['size_gb']}GB",
                expanded=(status != "ok"),
            ):
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    st.write(f"**File:** `{info['filename']}`")
                    st.write(f"**Node class:** `{info['node_class']}`")
                    st.write(f"**Required for:** {info['required_for']}")
                with col_b:
                    st.write(f"**Node available:** {'Yes' if info.get('node_available') else '❌ No'}")
                    st.write(f"**File found:** {'Yes' if info.get('installed') else '❌ No'}")
                    if status != "ok":
                        st.markdown(f"**Download:** {info.get('download_url', '—')}")
                        wget = info.get("wget_cmd", "")
                        if wget and "<URL>" not in wget:
                            st.code(wget, language="bash")

    st.divider()
    col_back, col_next = st.columns([1, 4])
    with col_back:
        if st.button("← Back", use_container_width=True):
            goto(9)
    with col_next:
        if st.button("Confirm settings → Queue videos", type="primary", use_container_width=True):
            save()
            p.next_step()   # → step 11 (queue)
            save()
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 11 — QUEUE TO COMFYUI
# ══════════════════════════════════════════════════════════════════════════════

def step_11():
    p = proj()
    st.header("📤 Step 8 — Queue Videos")
    st.caption(
        f"Upload approved images and queue I2V jobs for all {p.scene_count} scenes. "
        "Each job takes ~2-10 minutes depending on your GPU."
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    already_queued = [s for s in p.scenes if s.video_job_id]
    not_queued     = [s for s in p.scenes if not s.video_job_id]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total scenes", p.scene_count)
    col2.metric("Already queued", len(already_queued))
    col3.metric("Pending queue", len(not_queued))

    est_minutes = len(not_queued) * 5   # rough 5-min estimate per scene
    if not_queued:
        st.info(
            f"**{len(not_queued)} scene(s) to queue.**  "
            f"Estimated total wait time: ~{est_minutes} min  ·  "
            f"Running sequentially on your server."
        )
    else:
        st.success("✅ All scenes are queued!")

    # ── Queue state machine (one scene per rerun) ─────────────────────────────
    if st.session_state.queue_scene_idx is not None:
        idx   = st.session_state.queue_scene_idx
        scene = p.scenes[idx]

        progress_val = idx / len(p.scenes)
        st.progress(progress_val, text=f"Queuing scene {idx + 1} of {len(p.scenes)}…")

        with st.spinner(f"Uploading image and queueing scene {idx + 1} — {scene.scene_id}…"):
            try:
                job_id = asyncio.run(queue_video_job(client(), scene, p))
                scene.video_job_id = job_id
                scene.set_status("generating_video")
                st.toast(f"✅ Scene {idx + 1} queued — job `{job_id[:8]}…`")
            except Exception as e:
                scene.set_status("failed", error=str(e))
                st.error(f"Scene {idx + 1} queue failed: {e}")
                st.session_state.queue_scene_idx = None
                save()
                st.stop()

        # Advance to next un-queued scene
        next_idx = None
        for j in range(idx + 1, len(p.scenes)):
            if not p.scenes[j].video_job_id:
                next_idx = j
                break

        st.session_state.queue_scene_idx = next_idx
        save()
        st.rerun()

    # ── Queue job table ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("Job queue")

    for scene in p.scenes:
        col_num, col_act, col_status, col_job = st.columns([1, 2, 2, 4])
        col_num.write(f"**S{scene.scene_number}**")
        col_act.write(f"`{scene.act}`")
        if scene.video_job_id:
            col_status.success("queued")
            col_job.code(scene.video_job_id, language=None)
        elif scene.status == "failed":
            col_status.error("failed")
            col_job.write(scene.error_message or "—")
        else:
            col_status.write("pending")
            col_job.write("—")

    # ── Controls ──────────────────────────────────────────────────────────────
    st.divider()
    col_back, col_queue, col_requeue, col_next = st.columns([1, 2, 2, 2])

    with col_back:
        if st.button("← Back", use_container_width=True):
            goto(10)

    with col_queue:
        if not_queued and st.session_state.queue_scene_idx is None:
            if st.button("▶ Queue all scenes", type="primary", use_container_width=True):
                # Find first un-queued scene
                first_idx = next(
                    (i for i, s in enumerate(p.scenes) if not s.video_job_id), None
                )
                st.session_state.queue_scene_idx = first_idx
                st.rerun()

    with col_requeue:
        failed = [s for s in p.scenes if s.status == "failed"]
        if failed and st.button(f"↺ Retry {len(failed)} failed", use_container_width=True):
            # Clear job_ids on failed scenes so they get re-queued
            for s in failed:
                s.video_job_id = None
                s.set_status("approved")
            save()
            st.rerun()

    with col_next:
        all_queued = all(s.video_job_id for s in p.scenes)
        if st.button("Monitor progress →", type="primary", use_container_width=True,
                     disabled=not all_queued):
            p.next_step()   # → step 12 (monitor)
            save()
            st.rerun()
        if not all_queued:
            st.caption("⚠ Queue all scenes before continuing.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 12 — PROGRESS MONITOR
# ══════════════════════════════════════════════════════════════════════════════

def step_12():
    p = proj()
    st.header("📡 Step 9 — Progress Monitor")
    st.caption("Track video generation. Download clips as they complete.")

    # ── Status fetch ──────────────────────────────────────────────────────────
    auto_refresh = st.checkbox("Auto-refresh every 30s", value=False)

    col_refresh, col_dl = st.columns([2, 3])
    with col_refresh:
        manual_refresh = st.button("🔄 Refresh now", use_container_width=True)

    statuses: dict[str, str] = {}
    if manual_refresh or auto_refresh:
        with st.spinner("Fetching job statuses…"):
            try:
                statuses = asyncio.run(get_all_statuses(client(), p))
            except Exception as e:
                st.warning(f"Could not reach ComfyUI: {e}")

    # ── Overall progress ──────────────────────────────────────────────────────
    done_scenes  = [s for s in p.scenes if s.status == "done" and s.video_path]
    total        = p.scene_count
    n_done       = len(done_scenes)

    st.progress(n_done / max(total, 1),
                text=f"{n_done}/{total} scenes complete")

    if n_done == total:
        st.success("🎉 All scenes generated! Proceed to playback.")

    # ── Per-scene status table ────────────────────────────────────────────────
    st.divider()

    STATUS_ICONS = {
        "not_queued":       "⬜ not queued",
        "queued":           "🟡 queued",
        "running":          "🔵 running",
        "done":             "✅ done",
        "failed":           "🔴 failed",
        "unknown":          "⚪ unknown",
        "generating_video": "🔵 generating",
    }

    for scene in p.scenes:
        live_status = statuses.get(scene.scene_id, scene.status)
        icon_label  = STATUS_ICONS.get(live_status, f"⚪ {live_status}")

        col_num, col_act, col_status, col_job, col_actions = st.columns([1, 2, 2, 3, 2])
        col_num.write(f"**S{scene.scene_number}**")
        col_act.write(f"`{scene.act}`")
        col_status.write(icon_label)
        col_job.code(scene.video_job_id[:12] + "…" if scene.video_job_id else "—",
                     language=None)

        with col_actions:
            # Download if done and not yet saved locally
            if live_status == "done" and not scene.video_path:
                if st.button("⬇ Download", key=f"dl_{scene.scene_id}",
                             use_container_width=True):
                    with st.spinner(f"Downloading S{scene.scene_number}…"):
                        try:
                            vid_path = asyncio.run(
                                download_completed_video(client(), scene, p)
                            )
                            if vid_path:
                                scene.video_path = str(vid_path)
                                scene.set_status("done")
                                save()
                                st.rerun()
                            else:
                                st.warning("Download returned no file.")
                        except Exception as e:
                            st.error(f"Download failed: {e}")

            elif scene.video_path and Path(scene.video_path).exists():
                st.write("✅ saved")

            # Retry failed scenes
            if live_status == "failed":
                if st.button("↺ Retry", key=f"retry_{scene.scene_id}",
                             use_container_width=True):
                    scene.video_job_id = None
                    scene.set_status("approved")
                    save()
                    goto(11)

    # ── Bulk download ─────────────────────────────────────────────────────────
    with col_dl:
        downloadable = [
            s for s in p.scenes
            if statuses.get(s.scene_id, s.status) == "done" and not s.video_path
        ]
        if downloadable:
            if st.button(f"⬇ Download all {len(downloadable)} ready clips",
                         use_container_width=True):
                prog = st.progress(0.0, text="Downloading…")
                for i, scene in enumerate(downloadable):
                    prog.progress((i + 1) / len(downloadable),
                                  text=f"Downloading S{scene.scene_number}…")
                    try:
                        vid_path = asyncio.run(
                            download_completed_video(client(), scene, p)
                        )
                        if vid_path:
                            scene.video_path = str(vid_path)
                            scene.set_status("done")
                    except Exception:
                        pass
                save()
                st.rerun()

    # ── Auto-refresh (polling) ────────────────────────────────────────────────
    if auto_refresh and n_done < total:
        time.sleep(30)
        st.rerun()

    # ── Navigation ────────────────────────────────────────────────────────────
    st.divider()
    col_back, col_next = st.columns([1, 4])
    with col_back:
        if st.button("← Back to Queue", use_container_width=True):
            goto(11)
    with col_next:
        any_done = any(s.video_path and Path(s.video_path).exists() for s in p.scenes)
        if st.button("Playback & Export →", type="primary", use_container_width=True,
                     disabled=not any_done):
            p.next_step()   # → step 13 (playback)
            save()
            st.rerun()
        if not any_done:
            st.caption("⚠ Download at least one clip before continuing.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 13 — PLAYBACK + MONTAGE EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def step_13():
    p = proj()
    st.header("🎬 Step 10 — Playback & Export")
    st.caption("Watch your generated clips and compile a final montage.")

    # ── Per-scene video players ───────────────────────────────────────────────
    st.subheader("Scene clips")

    ready_scenes    = [s for s in p.scenes if s.video_path and Path(s.video_path).exists()]
    missing_scenes  = [s for s in p.scenes if not s.video_path or not Path(s.video_path).exists()]

    if missing_scenes:
        st.warning(
            f"⚠ {len(missing_scenes)} scene(s) have no downloaded video: "
            + ", ".join(f"S{s.scene_number}" for s in missing_scenes)
            + "  ·  Go back to Monitor to download them."
        )

    if ready_scenes:
        n_cols = min(len(ready_scenes), 3)
        rows = [ready_scenes[i:i+n_cols] for i in range(0, len(ready_scenes), n_cols)]
        for row in rows:
            cols = st.columns(n_cols)
            for col, scene in zip(cols, row):
                with col:
                    st.video(scene.video_path)
                    st.caption(
                        f"**S{scene.scene_number}** `{scene.act}`  \n"
                        f"{scene.description[:60]}"
                        + ("…" if len(scene.description) > 60 else "")
                    )
    else:
        st.info("No clips downloaded yet. Return to Monitor to download completed videos.")

    # ── Montage compilation ───────────────────────────────────────────────────
    st.divider()
    st.subheader("Compile montage")

    backend = available_backend()
    if backend == "none":
        st.error(
            "No compilation backend available.\n\n"
            "Install one of:\n"
            "- `pip install moviepy`\n"
            "- Download [ffmpeg](https://ffmpeg.org/download.html) and add to PATH"
        )
    else:
        st.caption(f"Backend: **{backend}**")

        col_opts1, col_opts2, col_opts3 = st.columns(3)
        with col_opts1:
            transition = st.selectbox(
                "Transition",
                options=["dissolve", "fade", "cut"],
                index=0,
                help="dissolve = crossfade · fade = fade through black · cut = hard cut",
            )
        with col_opts2:
            t_dur = st.slider(
                "Transition duration (s)", 0.0, 2.0, value=0.5, step=0.1,
                disabled=(transition == "cut"),
            )
        with col_opts3:
            out_fps = st.number_input("Output FPS", value=p.fps, step=1, min_value=8)

        music_path = None
        music_vol  = 0.3
        with st.expander("🎵 Background music (optional)"):
            uploaded = st.file_uploader(
                "Upload audio file (MP3 / WAV / OGG)",
                type=["mp3", "wav", "ogg", "m4a"],
            )
            if uploaded:
                music_dir = Path(p.output_dir) / "music"
                music_dir.mkdir(parents=True, exist_ok=True)
                music_path = music_dir / uploaded.name
                music_path.write_bytes(uploaded.read())
                music_vol = st.slider("Music volume", 0.0, 1.0, value=0.3, step=0.05)
                st.success(f"Audio loaded: {uploaded.name}")

        montage_output = Path(p.output_dir) / "montage" / f"{p.project_name}_final.mp4"

        if st.button("🎬 Compile montage", type="primary",
                     use_container_width=True, disabled=len(ready_scenes) < 2):
            video_paths = [Path(s.video_path) for s in ready_scenes]
            with st.spinner(
                f"Compiling {len(video_paths)} clips with {transition} transitions…  "
                "(this may take a minute)"
            ):
                try:
                    result = compile_montage(
                        video_paths=video_paths,
                        output_path=montage_output,
                        transition=transition,
                        transition_duration=t_dur,
                        music_path=music_path,
                        music_volume=music_vol,
                        fps=int(out_fps),
                    )
                    st.session_state.montage_path = str(result)
                    save()
                    st.rerun()
                except Exception as e:
                    st.error(f"Montage compilation failed: {e}")

        if len(ready_scenes) < 2:
            st.caption("⚠ Need at least 2 clips to compile a montage.")

    # ── Final video player + download ─────────────────────────────────────────
    montage = st.session_state.montage_path
    if montage and Path(montage).exists():
        st.divider()
        st.subheader("🎥 Final montage")
        st.video(montage)

        with open(montage, "rb") as f:
            st.download_button(
                label="⬇ Download final video",
                data=f,
                file_name=Path(montage).name,
                mime="video/mp4",
                use_container_width=True,
                type="primary",
            )
        st.caption(f"Saved at: `{montage}`")

    # ── Project complete metrics ──────────────────────────────────────────────
    st.divider()
    st.subheader("Project complete 🎉")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total scenes", p.scene_count)
    col2.metric("Clips ready", len(ready_scenes))
    col3.metric("Style", p.style_dna.skill_name if p.style_dna else "—")
    col4.metric("Output dir", str(Path(p.output_dir).name))

    st.divider()
    if st.button("← Back to Monitor", use_container_width=True):
        goto(12)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ══════════════════════════════════════════════════════════════════════════════

p = proj()
step = p.current_step if p else 1

STEP_FN = {
    1:  step_1,
    2:  step_2,
    3:  step_3,
    4:  step_4,
    5:  step_5,
    6:  step_6,
    7:  step_7,
    8:  step_8,
    9:  step_9,
    10: step_10,
    11: step_11,
    12: step_12,
    13: step_13,
}

fn = STEP_FN.get(step, step_1)
fn()
