"""
ComfyUI AI Video Pipeline — 13-Step Wizard UI
Sprint 3: Steps 1-9  (Idea → Storyboard → Approved images + finalized video prompts)
Sprint 4: Steps 10-13 (Tech config → Queue → Monitor → Playback/Export)

Run:  venv\\Scripts\\streamlit run app.py
      then open  http://localhost:8501
"""

from __future__ import annotations

import asyncio
import html
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable

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
    discover_workflows, save_uploaded_workflow,
    detect_model_slots, options_for_slot, slot_status, missing_node_types,
)
from pipeline.style_inference import Character
from pipeline.story_generator import (
    generate_story_options,
    generate_scenes_from_story,
    generate_character_description,
)
from pipeline.ui_tokens import (
    FOCUS_META, ROLE_META, ACT_COLOR,
    focus_chip, role_chip, act_pill, hero_badge,
    focus_label, role_label, focus_color,
    OFFLINE_BADGE_HTML,
)
from skills_engine import SKILLS, build_comfyui_positive, build_comfyui_negative
from pipeline import (
    build_story_prompt, build_character_prompt,
    build_scene_prompts_prompt, build_single_scene_prompt,
    parse_story_response, parse_character_response,
    parse_scene_prompts_response, parse_error_message,
)
from pipeline import (
    write_pending_job, read_result,
    pending_job_count, done_job_count_today,
)

# ── Config ────────────────────────────────────────────────────────────────────
# Anchor all paths to the directory that contains app.py so the UI works
# regardless of which directory the user launches Streamlit from.
_APP_DIR = Path(__file__).parent

CONFIG = yaml.safe_load(open(_APP_DIR / "config.yaml", encoding="utf-8"))
PROJECTS_DIR = _APP_DIR / "projects"
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
_ss("ai_mode", "copy_paste")   # str — active AI generation mode
_ss("mcp_worker_active", False) # bool — user signals that Claude Code worker is running
# Per-stage MCP job tracking
_ss("mcp_story_job",    None)   # str | None — pending job_id for Step 3
_ss("mcp_story_start",  None)   # float | None — time.time() when submitted
_ss("mcp_char_job",     None)   # str | None — pending job_id for Step 4
_ss("mcp_char_start",   None)
_ss("mcp_scenes_job",   None)   # str | None — pending job_id for Step 5
_ss("mcp_scenes_start", None)

# ── Shortcuts ─────────────────────────────────────────────────────────────────
def proj() -> ProjectState | None:
    return st.session_state.project

def set_proj(p: ProjectState):
    st.session_state.project = p

def save():
    p = proj()
    if p:
        # Use absolute path so saves land in the right place regardless of CWD
        p.save(PROJECTS_DIR / f"{p.project_name}.json")


def _clear_all_mcp_state():
    """Clear every MCP job key and per-step session cache.

    Called when loading or creating a project so stale job IDs from a
    previous session never trigger the loading overlay on the new project.
    """
    for key in ("mcp_story_job", "mcp_story_start",
                "mcp_char_job",  "mcp_char_start",
                "mcp_scenes_job","mcp_scenes_start"):
        st.session_state[key] = None
    st.session_state.story_options  = None
    st.session_state.gen_scene_idx  = None

def client() -> ComfyUIClient:
    return ComfyUIClient(CONFIG["comfyui"]["host"], CONFIG["comfyui"]["port"])

def goto(step: int):
    p = proj()
    if p:
        p.goto_step(step)
        save()
    st.rerun()


def _scenes_sig(scenes) -> tuple:
    """Cheap stable signature of a scene list — used as a cache/rebuild key."""
    return tuple(
        (s.scene_id, getattr(s, "focus", "subject"),
         getattr(s, "narrative_role", ""), bool(getattr(s, "hero_moment", False)),
         s.act, (s.description or "")[:32])
        for s in scenes
    )


def _story_overview(p) -> None:
    """At-a-glance story arc panel — colour-coded focus strip, role checklist,
    hero placement, and rule-based health warnings. Fully offline (no AI)."""
    if not p.scenes:
        return
    n        = len(p.scenes)
    n_subj   = sum(1 for s in p.scenes if getattr(s, "focus", "subject") == "subject")
    n_hero   = sum(1 for s in p.scenes if getattr(s, "hero_moment", False))
    n_role   = sum(1 for s in p.scenes if getattr(s, "narrative_role", ""))
    pct_subj = round((n_subj / n) * 100) if n else 0

    expanded_default = n <= 8
    with st.expander(
        f"🎬 Story overview — {n} scenes · {n_hero} hero · {pct_subj}% character-led",
        expanded=expanded_default,
    ):
        cm, cc, cr = st.columns([1.4, 3, 2])

        # ── Block A: metrics ────────────────────────────────────────────────
        with cm:
            st.metric("Total scenes", f"{n}", f"≈ {n * 5}s of video")
            if n_hero == 0:
                st.metric("Hero moment", "—", "none set ⚠", delta_color="inverse")
            elif n_hero == 1:
                hero_scene = next(s for s in p.scenes if getattr(s, "hero_moment", False))
                st.metric("Hero moment", f"S{hero_scene.scene_number}", hero_scene.act)
            else:
                st.metric("Hero moment", f"{n_hero} scenes", "consider thinning",
                          delta_color="inverse")
            balance = ("well balanced" if 25 <= pct_subj <= 60
                       else ("character-heavy" if pct_subj > 60 else "world-heavy"))
            st.metric("Character-led", f"{pct_subj}%", balance,
                      delta_color="off" if 25 <= pct_subj <= 60 else "inverse")

        # ── Block B: coloured focus strip with hero stars ───────────────────
        with cc:
            st.markdown("**Focus across the timeline**")
            try:
                import altair as alt
                import pandas as pd
                df = pd.DataFrame([
                    {"scene": s.scene_number,
                     "focus": getattr(s, "focus", "subject"),
                     "focus_label": focus_label(getattr(s, "focus", "subject")),
                     "hero": bool(getattr(s, "hero_moment", False))}
                    for s in p.scenes
                ])
                domain = list(FOCUS_META.keys())
                rng    = [FOCUS_META[k][2] for k in domain]
                bars = (alt.Chart(df)
                    .mark_bar(size=30, cornerRadius=4)
                    .encode(
                        x=alt.X("scene:O", title=None, axis=alt.Axis(labelAngle=0)),
                        y=alt.value(50),
                        color=alt.Color("focus:N",
                            scale=alt.Scale(domain=domain, range=rng),
                            legend=alt.Legend(orient="bottom", title="Shot focus",
                                              labelExpr=("datum.label == 'subject' ? '🧑 Protagonist' :"
                                                         "datum.label == 'establishing' ? '🌅 Place' :"
                                                         "datum.label == 'object' ? '📦 Object' :"
                                                         "datum.label == 'detail' ? '🔍 Detail' :"
                                                         "datum.label == 'phenomenon' ? '🌪 Action' :"
                                                         "datum.label == 'secondary' ? '👥 Figure' :"
                                                         "datum.label == 'reaction' ? '👁 Reaction' : datum.label"))),
                        tooltip=["scene:O", "focus_label:N"],
                    ).properties(height=80))
                hero_df = df[df.hero]
                if len(hero_df):
                    bars = bars + alt.Chart(hero_df).mark_text(
                        text="★", size=22, dy=-30, color="#fbbf24",
                    ).encode(x="scene:O")
                st.altair_chart(bars, use_container_width=True)
            except Exception:
                # Plain-HTML fallback if altair isn't available for some reason
                cells = "".join(
                    f"<div title='S{s.scene_number}: {focus_label(getattr(s, 'focus', 'subject'))}'"
                    f"style='flex:1;height:32px;border-radius:4px;"
                    f"background:{focus_color(getattr(s, 'focus', 'subject'))};"
                    f"margin:0 1px;display:flex;align-items:center;justify-content:center;"
                    f"color:#000;font-weight:700'>"
                    f"{'★' if getattr(s, 'hero_moment', False) else ''}</div>"
                    for s in p.scenes
                )
                st.markdown(
                    f"<div style='display:flex'>{cells}</div>",
                    unsafe_allow_html=True,
                )

        # ── Block C: role checklist ─────────────────────────────────────────
        with cr:
            st.markdown("**Role coverage**")
            present = {getattr(s, "narrative_role", "") for s in p.scenes
                       if getattr(s, "narrative_role", "")}
            recommended = ["establish_context", "introduce_subject",
                           "build_tension", "deliver_payload", "resolution"]
            for r in recommended:
                emo, lbl, _ = ROLE_META[r]
                hit = r in present
                mark = "✅" if hit else "·"
                color = "#22c55e" if hit else "#6b7280"
                st.markdown(
                    f"<span style='color:{color};font-size:.9rem'>{mark} {emo} {lbl}</span>",
                    unsafe_allow_html=True,
                )

        # ── Health checks ───────────────────────────────────────────────────
        warnings: list[str] = []
        if n_hero == 0:
            warnings.append(
                "⚠ **No hero moment marked.** Your final video won't have an extended "
                "hold beat — pick your visual peak and toggle ★ in the "
                "**✨ Make it special** tab of any scene."
            )
        if n_hero > 4:
            warnings.append(
                "⚠ **Many hero scenes** — 1–2 is ideal so the peak reads. "
                "Untoggle ★ on the less essential ones."
            )
        if n_subj == n and n > 1:
            warnings.append(
                "⚠ **Every shot is character-led.** Your video may feel claustrophobic. "
                "Set Focus = 🌅 *About the place* on scene 1, and consider 📦/🔍/🌪 "
                "for at least one mid-scene."
            )
        if n_subj == 0 and p.character:
            warnings.append(
                "💡 **Character locked but zero character-led scenes.** "
                "Set Focus = 🧑 *About the protagonist* on at least one beat "
                "(usually the payoff)."
            )
        if not present:
            warnings.append(
                "💡 **No narrative roles assigned.** Pick one per scene under "
                "**✨ Make it special** to unlock storytelling-grade compose at Step 10."
            )
        for w in warnings:
            st.warning(w)
        if not warnings:
            st.success("✅ Solid coverage: hero set, varied focus, story roles in use.")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🎬 AI Video Pipeline")
    st.markdown(OFFLINE_BADGE_HTML, unsafe_allow_html=True)
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

    # ── MCP Worker status ─────────────────────────────────────────────────────
    st.subheader("🤖 MCP Worker")

    worker_active = st.session_state.get("mcp_worker_active", False)
    worker_toggle = st.toggle(
        "Worker is running",
        value=worker_active,
        key="mcp_worker_toggle",
        help=(
            "Check this after starting the pipeline worker in Claude Code. "
            "Enables the 🤖 MCP Auto mode button in Steps 3, 4, and 5."
        ),
    )
    if worker_toggle != worker_active:
        st.session_state.mcp_worker_active = worker_toggle
        # If worker just deactivated while in MCP mode, fall back to copy_paste
        if not worker_toggle and st.session_state.get("ai_mode") == "mcp":
            st.session_state.ai_mode = "copy_paste"
        st.rerun()

    if worker_active:
        n_pending = pending_job_count()
        n_done    = done_job_count_today()
        st.success("✅ Worker active")
        col_a, col_b = st.columns(2)
        col_a.metric("Pending jobs", n_pending)
        col_b.metric("Done today",   n_done)
    else:
        with st.expander("How to start the worker"):
            st.markdown(
                "1. Open a **Claude Code** session in this project folder\n"
                "2. Type: **`start pipeline worker`**\n"
                "3. Claude Code will begin monitoring and auto-processing jobs\n"
                "4. Come back here and check **Worker is running** above\n\n"
                "The `.mcp.json` file in this project registers the pipeline "
                "MCP server so Claude Code connects automatically on startup."
            )

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
            _clear_all_mcp_state()
            st.rerun()
    else:
        st.info("No saved projects yet.")

    if p and st.button("💾 Save project", use_container_width=True):
        path = save()
        st.success(f"Saved → projects/{p.project_name}.json")

    if st.button("New project", use_container_width=True):
        set_proj(None)
        _clear_all_mcp_state()
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# AI MODE SELECTOR — shared widget used in Steps 3, 4, 5
# ══════════════════════════════════════════════════════════════════════════════

def _render_mode_selector(on_change: "Callable[[str], None] | None" = None) -> str:
    """Render the AI generation mode selector. Returns the current mode string.

    Args:
        on_change: optional callback called with the *new* mode string whenever
                   the user switches to a different mode.  Use it to clear
                   stale per-step content so the new mode's UI is shown fresh.

    Modes
    -----
    copy_paste  — user copies a prompt, pastes it into any chatbot, pastes
                  the JSON response back into the UI
    mechanical  — rule-based, no AI  (always available)
    mcp         — MCP Auto via Claude Code session
    api_key     — Anthropic API key  (coming later)
    """
    mode        = st.session_state.get("ai_mode", "copy_paste")
    mcp_active  = st.session_state.get("mcp_worker_active", False)

    def _switch(new_mode: str) -> None:
        """Switch mode, fire on_change callback if mode actually changed."""
        if new_mode != st.session_state.get("ai_mode"):
            if on_change is not None:
                on_change(new_mode)
        st.session_state.ai_mode = new_mode
        st.rerun()

    with st.container(border=True):
        st.caption(
            "AI generation mode — choose how story / character / scene content is produced"
        )
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            mcp_label = "🤖 MCP Auto" + (" ◀" if mode == "mcp" else "")
            if mcp_active:
                if st.button(
                    mcp_label,
                    type="primary" if mode == "mcp" else "secondary",
                    use_container_width=True,
                    key="mode_mcp",
                    help="Claude Code processes jobs automatically — worker is active",
                ):
                    _switch("mcp")
            else:
                st.button(
                    "🤖 MCP Auto",
                    disabled=True,
                    use_container_width=True,
                    key="mode_mcp",
                    help="Enable in sidebar: mark MCP Worker as active first",
                )
        with c2:
            if st.button(
                "✂️ Copy-Paste" + (" ◀" if mode == "copy_paste" else ""),
                type="primary" if mode == "copy_paste" else "secondary",
                use_container_width=True,
                key="mode_cp",
                help="Generate a prompt here → paste into any chatbot → paste response back",
            ):
                _switch("copy_paste")
        with c3:
            if st.button(
                "⚙️ Mechanical" + (" ◀" if mode == "mechanical" else ""),
                type="primary" if mode == "mechanical" else "secondary",
                use_container_width=True,
                key="mode_mech",
                help="Rule-based generation — no AI, no chatbot needed",
            ):
                _switch("mechanical")
        with c4:
            st.button(
                "🔑 API Key",
                disabled=True,
                use_container_width=True,
                key="mode_apikey",
                help="Anthropic API key integration — coming later",
            )
    return mode


# ══════════════════════════════════════════════════════════════════════════════
# MCP LOADING SCREEN
# ══════════════════════════════════════════════════════════════════════════════

# Tips rotate every 8 seconds — keyed by stage
_MCP_TIPS: dict[str, list[tuple[str, str]]] = {
    "story_options": [
        ("💡", "Add a mood like *'melancholic'*, *'tense'* or *'euphoric'* on Step 1 — it shapes the entire narrative structure Claude picks."),
        ("🎬", "Each of the 3 story options has a genuinely different arc — compare them before choosing. You can't go back without regenerating."),
        ("📐", "Longer duration = more scenes = richer cinematic development. 60-second videos support full 5-act structures."),
        ("🎭", "The *arc* field uses 5 emotional beats separated by → — these guide the visual language of every scene downstream."),
        ("✂️", "In Copy-Paste mode you can use any chatbot — Claude.ai, ChatGPT, Gemini. The JSON schema works everywhere."),
        ("🎯", "Scene descriptions are intentionally one sentence. Brevity forces visual specificity — which models respond to better."),
        ("🧠", "Claude Code has full knowledge of your chosen visual style and colour palette — it bakes them into every story beat."),
        ("🔄", "Not happy with the options? Hit **Regenerate** — each run produces genuinely different narrative structures."),
    ],
    "character_description": [
        ("📸", "In Copy-Paste mode you can attach a **reference photo** to your chatbot message — the AI will match the description to what it sees."),
        ("👁️", "The character description is injected into every scene where the character is *on screen* — establishing and insert shots stay character-free, like real cinema."),
        ("🎭", "Be specific about clothing: *'worn canvas duster, open at the collar'* generates far more consistent results than just *'jacket'*."),
        ("🔍", "Add one unmistakable physical detail — a scar, unusual eye colour, or specific accessory — as a recognition anchor across scenes."),
        ("💡", "Avoid vague adjectives like *'beautiful'* or *'strong'*. Describe exactly what a camera lens would capture."),
        ("🎬", "Age range matters more than exact age: *'mid-forties'* is more stable than *'44 years old'* across image models."),
        ("🧵", "Fabric and texture descriptions help: *'sun-bleached linen'* tells the model about light behaviour, not just colour."),
        ("🌗", "Skin tone described in lighting terms (*'warm olive under harsh sunlight'*) integrates better with your scene lighting presets."),
    ],
    "scene_prompts": [
        ("🎬", "Visual prompts describe what the **storyboard image** looks like. Video prompts describe only **motion** — never duplicate appearance."),
        ("⚡", "End each video prompt with exactly one pacing word: `[slow]` `[medium]` `[fast]` `[explosive]` — this controls I2V inference speed."),
        ("📷", "Camera moves get precise: *'dolly forward 3 ft/s'* gives the model a measurable instruction. *'epic camera move'* does nothing."),
        ("💡", "Lighting in Kelvin (*3000K warm key*) + ratios (*100% key / 33% fill / 60% back rim*) produces more consistent light matching across scenes."),
        ("🔄", "You can **reprompt individual scenes** after generation — use the ↩️ reprompt button on any scene card without redoing the full batch."),
        ("🎨", "Quality boosters are appended automatically — they tell ComfyUI which rendering style to prioritise. Don't repeat them manually."),
        ("🌊", "Environmental motion details in video prompts (*wind in fabric, sand shifting, crowd swaying*) dramatically improve I2V realism."),
        ("🏗️", "Depth of field notation (*f/1.4 for shallow, f/11 for deep*) helps image models render the correct background blur for each scene."),
    ],
}

# Generic tips shown when stage has no specific tips or for variety
_MCP_TIPS_GENERIC: list[tuple[str, str]] = [
    ("🚀", "This pipeline uses **LTX-Video 2.3** (22-billion parameters) for image-to-video generation — one of the most capable open I2V models available."),
    ("🧠", "Claude Code processes your prompts directly in-session — no external API calls, no rate limits, no cost per generation."),
    ("⏱️", "Each 5-second video scene typically takes **2–10 minutes** on a modern GPU, depending on resolution and sampling steps."),
    ("🎞️", "The final montage stitches all scene videos in narrative order with optional fade transitions and audio normalisation."),
    ("💾", "Projects are auto-saved after every step — you can close the browser and resume exactly where you left off."),
    ("🖼️", "You can generate **1–5 storyboard images per scene** and choose the best one before committing to video generation."),
    ("🎨", "The **Cinematic** style uses 3000K warm key light at 45° camera-left with a 100/33/60 key-fill-back intensity ratio."),
    ("📡", "The Monitor step streams live ComfyUI progress via WebSocket — you see frame-by-frame updates as each video renders."),
]


def _render_mcp_loading_screen(stage: str, elapsed: float, job_id: str,
                               job_key: str, start_key: str) -> None:
    """Render a full-page MCP waiting screen using only Streamlit-native widgets.

    CSS is scoped tightly to class names used only here — no global selectors
    that could bleed into the rest of the app between Streamlit reruns.
    The router skips fn() while this is shown, so the page contains ONLY
    this loading content + the cancel button.
    """
    # Scroll to top so the loading screen is always fully visible
    _scroll_to_top()

    # ── Stage display config ──────────────────────────────────────────────────
    stage_meta = {
        "story_options":         ("📖", "Story Options",         "Crafting 3 cinematic narrative treatments…"),
        "character_description": ("🧑", "Character Description", "Defining your protagonist's visual identity…"),
        "scene_prompts":         ("🎬", "Scene Prompts",         "Writing cinematographer & director prompts for every scene…"),
    }
    icon, label, subtitle = stage_meta.get(stage, ("🤖", stage.replace("_", " ").title(), "Processing…"))

    mins, secs = divmod(int(elapsed), 60)
    elapsed_str = f"{mins}m {secs}s" if mins else f"{secs}s"

    pool = _MCP_TIPS.get(stage, []) + _MCP_TIPS_GENERIC
    tip_icon, tip_text = pool[int(elapsed / 8) % len(pool)]

    dot_states = ["●  ○  ○", "○  ●  ○", "○  ○  ●", "○  ●  ○"]
    dots = dot_states[int(elapsed / 0.75) % len(dot_states)]

    bar_pct = int((elapsed % 20) / 20 * 100)

    # animation-delay keeps the spinner at the correct rotation angle after each
    # rerun — avoids the visual "jump back to 0°" blink every 2 seconds.
    spin_delay = -(elapsed % 1.0)

    # ── Scoped CSS — uses unique class prefix, no global element selectors ────
    st.markdown(f"""
<style>
@keyframes mcp-spin {{
    from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }}
}}
.mcp-wrap {{
    display: flex; flex-direction: column; align-items: center;
    padding: 60px 16px 40px; min-height: 80vh; justify-content: center;
}}
.mcp-ring {{
    width: 64px; height: 64px;
    border: 4px solid rgba(79,142,247,.15);
    border-top-color: #4f8ef7;
    border-radius: 50%;
    animation: mcp-spin 1s linear {spin_delay:.3f}s infinite;
    margin-bottom: 32px;
}}
.mcp-card {{
    width: 100%; max-width: 660px;
    background: rgba(22,26,46,.7);
    border: 1px solid rgba(79,142,247,.22);
    border-radius: 14px;
    padding: 34px 40px 28px;
    text-align: center;
}}
.mcp-icon   {{ font-size: 2.4rem; margin-bottom: 12px; }}
.mcp-title  {{ font-size: 1.4rem; font-weight: 700; color: #e8eaf6; margin: 0 0 6px; }}
.mcp-sub    {{ font-size: 0.9rem;  color: #8892b0; margin: 0 0 20px; }}
.mcp-dots   {{ font-size: 1.2rem; letter-spacing: 6px; color: #4f8ef7;
              margin-bottom: 18px; font-family: monospace; }}
.mcp-bar-track {{
    width: 100%; height: 3px; background: rgba(79,142,247,.12);
    border-radius: 2px; overflow: hidden; margin-bottom: 24px;
}}
.mcp-tip {{
    background: rgba(255,255,255,.04);
    border: 1px solid rgba(255,255,255,.07);
    border-left: 3px solid #f7a84f;
    border-radius: 8px; padding: 12px 16px;
    margin-bottom: 20px; text-align: left;
}}
.mcp-tip-lbl {{
    font-size: 0.67rem; font-weight: 700; letter-spacing: .1em;
    text-transform: uppercase; color: #f7a84f; margin-bottom: 4px;
}}
.mcp-tip-txt {{ font-size: 0.87rem; color: #c0c8e0; line-height: 1.5; margin: 0; }}
.mcp-footer {{
    font-size: 0.77rem; color: #4a5270;
    display: flex; gap: 18px; justify-content: center; flex-wrap: wrap;
}}
.mcp-elapsed {{ color: #6b7db0; font-weight: 600; }}
</style>
""", unsafe_allow_html=True)

    st.markdown(f"""
<div class="mcp-wrap">
  <div class="mcp-ring"></div>
  <div class="mcp-card">
    <div class="mcp-icon">{icon}</div>
    <p class="mcp-title">Claude Code is generating {label}</p>
    <p class="mcp-sub">{subtitle}</p>
    <div class="mcp-dots">{dots}</div>
    <div class="mcp-bar-track">
      <div style="height:100%;width:{bar_pct}%;background:linear-gradient(90deg,#4f8ef7,#7eb3ff);border-radius:2px;transition:width .1s"></div>
    </div>
    <div class="mcp-tip">
      <div class="mcp-tip-lbl">{tip_icon} &nbsp; Did you know?</div>
      <p class="mcp-tip-txt">{tip_text}</p>
    </div>
    <div class="mcp-footer">
      <span class="mcp-elapsed">⏱ {elapsed_str}</span>
      <span>Job <code style="color:#5570a8;font-size:.75rem">{job_id[:8]}…</code></span>
      <span>Polling every 2s</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.divider()
    _, col_btn, _ = st.columns([1, 2, 1])
    with col_btn:
        if st.button("✕  Cancel — switch to Copy-Paste mode",
                     key=f"mcp_cancel_{job_key}", use_container_width=True):
            st.session_state[job_key]   = None
            st.session_state[start_key] = None
            st.session_state.ai_mode    = "copy_paste"
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MCP POLLING HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _mcp_submit_and_poll(
    job_key: str,
    start_key: str,
    stage: str,
    build_job: "Callable[[], tuple[dict, str]]",
) -> dict | None:
    """Submit a job if none is pending; poll until result arrives.

    Args:
        job_key   : session_state key for the job_id (e.g. "mcp_story_job")
        start_key : session_state key for submission timestamp
        stage     : pipeline stage string
        build_job : zero-arg callable → (payload_dict, prompt_for_human_str)

    Returns the raw done-file dict on success, None while waiting.
    Polls every 3 s indefinitely — no hard timeout.
    The Cancel button lets the user abort and switch to Copy-Paste mode.
    """
    job_id = st.session_state.get(job_key)

    if job_id is None:
        # First call — build and submit
        payload, prompt = build_job()
        p = proj()
        job_id = write_pending_job(
            stage=stage,
            payload=payload,
            prompt_for_human=prompt,
            project_name=p.project_name if p else "",
        )
        st.session_state[job_key]   = job_id
        st.session_state[start_key] = time.time()
        st.rerun()
        return None

    # Job already submitted — poll for result
    done = read_result(job_id)

    if done is not None:
        # Result ready — clear state and return
        st.session_state[job_key]   = None
        st.session_state[start_key] = None
        st.toast("✅ Generated by Claude Code!")
        return done

    # Still waiting — return None.
    # The main router renders the full-screen overlay and triggers the next poll.
    return None


# ── Workflow + model selection panel ──────────────────────────────────────────

def _get_server_object_info() -> dict | None:
    """Fetch /object_info once per session (it's ~MBs). Cached in session state."""
    if "server_object_info" not in st.session_state:
        st.session_state.server_object_info = None
    if st.session_state.server_object_info is None:
        try:
            st.session_state.server_object_info = asyncio.run(client().get_object_info())
        except Exception:
            st.session_state.server_object_info = None
    return st.session_state.server_object_info


def _workflow_picker(kinds: tuple, current: str, key: str, label: str) -> str:
    """Selectbox over local workflow templates of the given kind(s).

    Returns the selected workflow path (posix, project-relative) — falls back
    to *current* when nothing matches.
    """
    all_wf      = [wf for wf in discover_workflows() if wf.kind in kinds]
    usable      = [wf for wf in all_wf if wf.compatible]
    unusable    = [wf for wf in all_wf if not wf.compatible]

    if not usable:
        st.warning(
            f"No compatible workflow templates found in `workflows/` for {label}. "
            f"Using the project default."
        )
        return current

    kind_tag = {"t2i": "🖼 T2I", "i2v": "🎬 I2V", "t2v": "📝 T2V — ignores approved image!"}
    options  = [wf.path for wf in usable]
    labels   = {wf.path: f"{wf.name}   ·   {kind_tag.get(wf.kind, wf.kind)}" for wf in usable}

    # Normalise the stored value to posix so matching works on Windows
    cur = Path(current).as_posix() if current else ""
    idx = options.index(cur) if cur in options else 0

    chosen = st.selectbox(
        label, options, index=idx, key=key,
        format_func=lambda pth: labels.get(pth, pth),
        help="Templates are scanned from the local workflows/ folder. "
             "Use “Add a workflow” below to import one from your ComfyUI.",
    )

    chosen_info = next((wf for wf in usable if wf.path == chosen), None)
    if chosen_info and chosen_info.note:
        st.caption(f"⚙️ {chosen_info.note}")

    # Pre-flight node-class check: a template can be fillable yet need a custom
    # node this server doesn't have (→ cryptic 400 at queue time). Warn early.
    info = _get_server_object_info()
    if info:
        missing = missing_node_types(chosen, info)
        if missing:
            st.error(
                "⛔ This workflow needs ComfyUI node(s) your server doesn't have: "
                + ", ".join(f"`{m}`" for m in missing)
                + ". It will fail at queue time — install the custom node on the "
                "server or pick another workflow."
            )

    if unusable:
        with st.expander(f"ℹ️ {len(unusable)} other template(s) found but not selectable"):
            for wf in unusable:
                st.caption(f"`{wf.path}` — {wf.reason}")

    return chosen


def _model_slots_editor(workflow_path: str, overrides: dict, key_prefix: str) -> dict:
    """Per-slot model dropdowns fed by the server's installed models.

    Shows every model file the selected workflow loads, with an ✅/❌ installed
    badge, and lets the user swap each one for any model of the same loader
    type found on the ComfyUI server. Returns the updated overrides dict.
    """
    try:
        slots = detect_model_slots(workflow_path)
    except Exception as e:
        st.caption(f"Could not inspect models in this workflow: {e}")
        return overrides

    if not slots:
        st.caption("This workflow loads no model files directly.")
        return overrides

    info = _get_server_object_info()

    head_l, head_r = st.columns([4, 1])
    head_l.markdown(f"**Models used by this workflow** ({len(slots)})")
    with head_r:
        if st.button("🔄 Refresh", key=f"{key_prefix}_refresh",
                     help="Re-query the ComfyUI server's installed models"):
            st.session_state.server_object_info = None
            st.rerun()

    if info is None:
        st.warning("ComfyUI not reachable — showing template defaults without server validation.")

    new_overrides = dict(overrides)
    for slot in slots:
        opts = options_for_slot(info, slot) if info else []
        effective, installed = slot_status(slot, opts, overrides.get(slot.key))

        col_label, col_pick = st.columns([2, 3])
        with col_label:
            badge = "✅" if installed else ("❌" if info else "❓")
            st.markdown(f"{badge} `{slot.label}`")
            if not installed and info:
                st.caption("not found on server — pick an installed model →")
        with col_pick:
            if opts:
                # Effective value first if it's a valid option, else option 0
                idx = opts.index(effective) if effective in opts else 0
                choice = st.selectbox(
                    slot.label, opts, index=idx,
                    key=f"{key_prefix}_{slot.key}",
                    label_visibility="collapsed",
                )
            else:
                choice = st.text_input(
                    slot.label, value=effective,
                    key=f"{key_prefix}_{slot.key}",
                    label_visibility="collapsed",
                    help="Server options unavailable — enter the exact filename",
                )
            if choice and choice != slot.current:
                new_overrides[slot.key] = choice
            else:
                new_overrides.pop(slot.key, None)   # back to template default

    changed = {k: v for k, v in new_overrides.items() if v}
    if changed:
        names = " · ".join(f"`{Path(v).name}`" for v in changed.values())
        st.caption(f"🔧 {names} will replace the template defaults at queue time.")
    return new_overrides


def _workflow_upload_box(key: str) -> None:
    """Import a workflow exported from ComfyUI ('Save (API Format)').

    Auto-inserts {{PLACEHOLDER}} tokens so the file becomes selectable.
    """
    with st.expander("➕ Add a workflow from your ComfyUI"):
        st.markdown(
            "In ComfyUI: **Workflow → Export (API)** (or *Save (API Format)*), "
            "then drop the file here. Prompts, seed, size, frames and output "
            "nodes are detected and templated automatically."
        )
        up = st.file_uploader(
            "Workflow JSON (API format)", type=["json"],
            key=f"{key}_uploader", label_visibility="collapsed",
        )
        if up is not None and st.button("Import workflow", key=f"{key}_import", type="primary"):
            try:
                raw = up.getvalue().decode("utf-8")
                rel, notes = save_uploaded_workflow(up.name, raw)
                st.success(f"Imported as `{rel}` — it's now in the workflow list.")
                for n in notes:
                    st.caption(n)
                st.rerun()
            except ValueError as e:
                st.error(str(e))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — IDEA INPUT
# ══════════════════════════════════════════════════════════════════════════════

def step_1():
    st.header("💡 Step 1 — Your Idea")
    st.caption("Describe what you want to make. One sentence is enough.")

    # ── Reference media upload (before the form so files persist on rerun) ────
    st.markdown("##### 📎 Reference images *(optional)*")
    st.caption(
        "Upload mood boards, style references, or inspiration images. "
        "**MCP Auto:** the AI worker will view them directly. "
        "**Copy-Paste:** images are shown with the prompt so you can attach them to your chatbot."
    )
    uploaded_media = st.file_uploader(
        "Drop images here or click to browse",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="step1_idea_media",
        label_visibility="collapsed",
    )
    if uploaded_media:
        cols = st.columns(min(5, len(uploaded_media)))
        for i, uf in enumerate(uploaded_media):
            with cols[i % 5]:
                st.image(uf, caption=uf.name, use_container_width=True)

    st.divider()

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

        # Save any uploaded reference images to disk
        _files = st.session_state.get("step1_idea_media") or []
        if _files:
            media_dir = _APP_DIR / "media" / p.project_id / "idea"
            media_dir.mkdir(parents=True, exist_ok=True)
            for uf in _files:
                dest = media_dir / uf.name
                dest.write_bytes(uf.getvalue())
                p.idea_media_paths.append(str(dest))

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
    # Guard: if an MCP story job is in flight and not yet complete, render
    # nothing — the router loading-screen intercept handles the overlay.
    _sj = st.session_state.get("mcp_story_job")
    if _sj is not None and read_result(_sj) is None:
        return
    _scroll_to_top()
    p = proj()
    st.header("📖 Step 3 — Story Options")
    st.caption(f"Idea: *{p.idea}*  ·  {p.expected_scene_count} scenes  ·  {p.duration_seconds}s")

    # Restore story options when navigating back (session_state cleared by reload / back-nav)
    if st.session_state.story_options is None and p.story_options:
        st.session_state.story_options = p.story_options

    def _clear_story_state(_new_mode: str) -> None:
        """Wipe stale story options + any in-flight MCP job when mode switches."""
        st.session_state.story_options  = None
        st.session_state.mcp_story_job  = None
        st.session_state.mcp_story_start = None

    mode = _render_mode_selector(on_change=_clear_story_state)
    st.divider()

    options: list[dict] | None = st.session_state.story_options

    # ── Copy-paste path: show prompt + paste area ─────────────────────────────
    if mode == "copy_paste" and options is None:
        st.subheader("Step 1 — Copy this prompt into any chatbot")
        _idea_media = [mp for mp in (p.idea_media_paths or []) if Path(mp).exists()]
        prompt_text = build_story_prompt(
            p.idea, p.expected_scene_count, p.mood, p.style_dna,
            media_paths=_idea_media if _idea_media else None,
        )
        if _idea_media:
            st.info(
                "📎 **Reference images detected** — copy the prompt below, then attach "
                "the images shown underneath to your chatbot message before sending."
            )
        else:
            st.info(
                "📋 Copy the prompt below → paste it into **Claude.ai**, **ChatGPT**, "
                "**Gemini**, or any chatbot → copy the entire JSON response "
                "→ paste it in **Step 2** below."
            )
        st.code(prompt_text, language=None)   # built-in copy button

        # Show reference images the user should attach to the chatbot
        if _idea_media:
            st.markdown("**📎 Attach these images to your chatbot message:**")
            img_cols = st.columns(min(4, len(_idea_media)))
            for i, mp in enumerate(_idea_media):
                with img_cols[i % 4]:
                    st.image(mp, use_container_width=True, caption=Path(mp).name)

        st.subheader("Step 2 — Paste the AI response")
        raw = st.text_area(
            "Paste the full AI response here",
            height=220,
            key="cp_story_paste",
            placeholder=(
                'Paste the chatbot\'s full response here.\n'
                'It should contain a JSON block like:\n'
                '{"stage": "story_options", "result": [...]}'
            ),
        )
        col_parse, col_clear = st.columns([4, 1])
        with col_parse:
            if st.button("✅ Parse response →", type="primary", use_container_width=True,
                         key="cp_story_parse_btn"):
                raw_val = st.session_state.get("cp_story_paste", "").strip()
                if not raw_val:
                    st.error("Nothing pasted. Copy the AI response and paste it above.")
                else:
                    result = parse_story_response(raw_val)
                    if result:
                        st.session_state.story_options = result
                        st.success(
                            f"✅ Parsed {len(result)} story option(s)! "
                            "Scroll down to select one."
                        )
                        st.rerun()
                    else:
                        st.error(parse_error_message(raw_val, "story"))
        with col_clear:
            if st.button("Clear", use_container_width=True, key="cp_story_clear_btn"):
                st.session_state["cp_story_paste"] = ""
                st.rerun()

        st.divider()
        if st.button("← Back", use_container_width=True, key="step3_back_cp"):
            goto(2)
        return   # Don't render story cards until a response is parsed

    # ── MCP Auto path: submit job + poll ─────────────────────────────────────
    if mode == "mcp" and options is None:
        import dataclasses

        def _build_story_job():
            _media = [mp for mp in (p.idea_media_paths or []) if Path(mp).exists()]
            prompt  = build_story_prompt(
                p.idea, p.expected_scene_count, p.mood, p.style_dna,
                media_paths=_media if _media else None,
            )
            payload = {
                "idea":        p.idea,
                "n_scenes":    p.expected_scene_count,
                "mood":        p.mood,
                "style_dna":   dataclasses.asdict(p.style_dna) if p.style_dna else None,
                "media_paths": _media,
            }
            return payload, prompt

        # Only submit if the user has explicitly clicked Generate — never auto-fire
        if st.session_state.get("mcp_story_job") is None:
            st.info(
                "🤖 **MCP Auto mode** — Click **Generate** and Claude Code will "
                f"create 3 distinct cinematic story treatments "
                f"({p.expected_scene_count} scenes each) for your project."
            )
            col_gen, col_back = st.columns([3, 1])
            with col_gen:
                if st.button("🤖 Generate Story Options", type="primary",
                             use_container_width=True, key="step3_mcp_generate"):
                    _mcp_submit_and_poll("mcp_story_job", "mcp_story_start",
                                        "story_options", _build_story_job)
            with col_back:
                if st.button("← Back", use_container_width=True, key="step3_back_mcp_pre"):
                    goto(2)
            return

        done = _mcp_submit_and_poll("mcp_story_job", "mcp_story_start",
                                    "story_options", _build_story_job)
        if done is None:
            # Still waiting (router shows full-screen overlay); stop rendering this step
            if st.button("← Back", use_container_width=True, key="step3_back_mcp"):
                goto(2)
            return
        # Got result — parse and store
        raw_json = json.dumps({"stage": "story_options", "result": done["result"]})
        parsed = parse_story_response(raw_json)
        if parsed:
            st.session_state.story_options = parsed
            options = parsed
        else:
            st.error("Claude Code returned an unrecognised result format. Try Copy-Paste mode.")
            st.session_state.ai_mode = "copy_paste"
            st.rerun()
            return

    # ── Mechanical path: auto-generate on first visit ─────────────────────────
    if options is None:
        with st.spinner("Generating story treatments…"):
            st.session_state.story_options = generate_story_options(
                p.idea, p.duration_seconds, p.mood
            )
        options = st.session_state.story_options

    # ── Regenerate / new-prompt button ────────────────────────────────────────
    col_regen, _ = st.columns([1, 4])
    with col_regen:
        regen_label = (
            "✂️ New AI prompt"   if mode == "copy_paste" else
            "🤖 Re-generate"    if mode == "mcp"         else
            "🔄 Regenerate all"
        )
        if st.button(regen_label, use_container_width=True, key="step3_regen"):
            st.session_state.story_options = None
            st.rerun()

    st.divider()

    # ── Display story cards ───────────────────────────────────────────────────
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
    if st.button("← Back", use_container_width=True, key="step3_back"):
        goto(2)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — CHARACTER (UI step 3.5)
# ══════════════════════════════════════════════════════════════════════════════

def _scroll_to_top() -> None:
    """Inject JavaScript to scroll the Streamlit page to the top."""
    try:
        import streamlit.components.v1 as _cv1
        _cv1.html(
            "<script>"
            "window.parent.document.querySelector"
            "('[data-testid=\"stAppViewContainer\"]')?.scrollTo(0,0);"
            "window.parent.scrollTo(0,0);"
            "</script>",
            height=0,
        )
    except Exception:
        pass


def step_4():
    # Guard: if an MCP character job is in flight and not yet complete, render
    # nothing — the router loading-screen intercept handles the overlay.
    _cj = st.session_state.get("mcp_char_job")
    if _cj is not None and read_result(_cj) is None:
        return
    _scroll_to_top()
    p = proj()
    st.header("🧑 Step 3.5 — Character")
    st.caption("Lock the protagonist's visual description. It will be injected into every scene prompt.")

    def _clear_char_state(_new_mode: str) -> None:
        """Cancel any in-flight character MCP job when mode switches."""
        st.session_state.mcp_char_job   = None
        st.session_state.mcp_char_start = None
        # Also clear the character from the project so the new mode's form shows
        _p = proj()
        if _p:
            _p.character = None
            save()

    mode = _render_mode_selector(on_change=_clear_char_state)

    # ── Character reference media (Copy-Paste and MCP modes only) ─────────────
    if mode in ("copy_paste", "mcp"):
        st.markdown("##### 📎 Character reference images *(optional)*")
        st.caption(
            "Upload photos or concept art of your protagonist. "
            "**MCP Auto:** worker reads images directly. "
            "**Copy-Paste:** displayed below the prompt so you can attach them to your chatbot."
        )
        char_uploaded = st.file_uploader(
            "Drop images here or click to browse",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key="step4_char_media",
            label_visibility="collapsed",
        )
        if char_uploaded:
            # Save files to disk and update project
            media_dir = _APP_DIR / "media" / p.project_id / "character"
            media_dir.mkdir(parents=True, exist_ok=True)
            new_paths: list[str] = []
            for uf in char_uploaded:
                dest = media_dir / uf.name
                dest.write_bytes(uf.getvalue())
                new_paths.append(str(dest))
            if sorted(new_paths) != sorted(p.character_media_paths or []):
                p.character_media_paths = new_paths
                save()
            # Thumbnail strip
            img_cols = st.columns(min(5, len(char_uploaded)))
            for i, uf in enumerate(char_uploaded):
                with img_cols[i % 5]:
                    st.image(uf, caption=uf.name, use_container_width=True)

    st.divider()

    story = p.selected_story

    # ── Copy-paste path: show prompt + paste area ─────────────────────────────
    if mode == "copy_paste" and p.character is None:
        st.subheader("Step 1 — Copy this prompt into any chatbot")
        _char_media = [mp for mp in (p.character_media_paths or []) if Path(mp).exists()]
        prompt_text = build_character_prompt(
            p.idea, story, p.mood, p.style_dna,
            has_image_hint=not bool(_char_media),   # generic hint only if no real files
            media_paths=_char_media if _char_media else None,
        )
        if _char_media:
            st.info(
                "📎 **Character reference images detected** — copy the prompt below, "
                "then attach the images shown underneath to your chatbot message."
            )
        else:
            st.info(
                "💡 **Tip:** You can attach a character reference image to your chatbot "
                "message — the AI will describe what it sees.\n\n"
                "Copy the prompt → paste into any chatbot (optionally with an image) "
                "→ paste the JSON response in Step 2 below."
            )
        st.code(prompt_text, language=None)

        # Show reference images to attach
        if _char_media:
            st.markdown("**📎 Attach these images to your chatbot message:**")
            img_cols = st.columns(min(4, len(_char_media)))
            for i, mp in enumerate(_char_media):
                with img_cols[i % 4]:
                    st.image(mp, use_container_width=True, caption=Path(mp).name)

        st.subheader("Step 2 — Paste the AI response")
        raw = st.text_area(
            "Paste AI response here",
            height=150,
            key="cp_char_paste",
            placeholder='{"stage": "character_description", "result": "30-year-old woman…"}',
        )
        col_parse, col_skip = st.columns([3, 2])
        with col_parse:
            if st.button("✅ Parse & Apply →", type="primary", use_container_width=True,
                         key="cp_char_parse_btn"):
                raw_val = st.session_state.get("cp_char_paste", "").strip()
                if not raw_val:
                    st.error("Nothing pasted.")
                else:
                    result = parse_character_response(raw_val)
                    if result:
                        p.character = Character.new(result, base_seed=p.global_seed)
                        save()
                        st.success("✅ Character description applied!")
                        st.rerun()
                    else:
                        st.error(parse_error_message(raw_val, "character"))
        with col_skip:
            if st.button("⚙️ Use mechanical description instead",
                         use_container_width=True, key="cp_char_skip_btn"):
                with st.spinner("Generating protagonist description…"):
                    desc = generate_character_description(p.idea, story or {}, p.mood)
                p.character = Character.new(desc, base_seed=p.global_seed)
                save()
                st.rerun()

        st.divider()
        if st.button("← Back", use_container_width=True, key="step4_back_cp"):
            goto(3)
        return   # Wait until response parsed before showing edit UI

    # ── MCP Auto path: submit job + poll ─────────────────────────────────────
    if mode == "mcp" and p.character is None:
        import dataclasses

        def _build_char_job():
            _media = [mp for mp in (p.character_media_paths or []) if Path(mp).exists()]
            prompt  = build_character_prompt(
                p.idea, story, p.mood, p.style_dna,
                has_image_hint=False,
                media_paths=_media if _media else None,
            )
            payload = {
                "idea":        p.idea,
                "mood":        p.mood,
                "story":       story or {},
                "style_dna":   dataclasses.asdict(p.style_dna) if p.style_dna else None,
                "media_paths": _media,
            }
            return payload, prompt

        # Only submit if the user has explicitly clicked Generate — never auto-fire
        if st.session_state.get("mcp_char_job") is None:
            st.info(
                "🤖 **MCP Auto mode** — Click **Generate** and Claude Code will "
                "write a precise visual description of your protagonist that will "
                "be injected into every scene prompt."
            )
            col_gen, col_back = st.columns([3, 1])
            with col_gen:
                if st.button("🤖 Generate Character Description", type="primary",
                             use_container_width=True, key="step4_mcp_generate"):
                    _mcp_submit_and_poll("mcp_char_job", "mcp_char_start",
                                        "character_description", _build_char_job)
            with col_back:
                if st.button("← Back", use_container_width=True, key="step4_back_mcp_pre"):
                    goto(3)
            return

        done = _mcp_submit_and_poll("mcp_char_job", "mcp_char_start",
                                    "character_description", _build_char_job)
        if done is None:
            if st.button("← Back", use_container_width=True, key="step4_back_mcp"):
                goto(3)
            return
        raw_json = json.dumps({"stage": "character_description", "result": done["result"]})
        desc = parse_character_response(raw_json)
        if desc:
            p.character = Character.new(desc, base_seed=p.global_seed)
            save()
        else:
            st.error("Claude Code returned an unrecognised character format. Try Copy-Paste mode.")
            st.session_state.ai_mode = "copy_paste"
            st.rerun()
            return

    # ── Mechanical path: auto-generate on first visit ─────────────────────────
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
        if mode in ("copy_paste", "mcp"):
            label = "✂️ Re-prompt with AI" if mode == "copy_paste" else "🤖 Re-generate"
            if st.button(label, use_container_width=True, key="char_reprompt_btn"):
                p.character = None
                # Clear any pending MCP job so a fresh one is submitted
                st.session_state.mcp_char_job   = None
                st.session_state.mcp_char_start = None
                save()
                st.rerun()
        else:
            if st.button("🔄 Regenerate description", use_container_width=True):
                with st.spinner("Regenerating…"):
                    desc = generate_character_description(p.idea, story or {}, p.mood)
                char.description = desc
                save()
                st.rerun()

    st.divider()
    col_back, col_next = st.columns([1, 4])
    with col_back:
        if st.button("← Back", use_container_width=True, key="step4_back"):
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
    # Guard: if an MCP scenes job is in flight and not yet complete, render
    # nothing — the router loading-screen intercept handles the overlay.
    _scj = st.session_state.get("mcp_scenes_job")
    if _scj is not None and read_result(_scj) is None:
        return
    p = proj()
    st.header("📋 Step 4 — Scene Breakdown")

    def _clear_scene_state(_new_mode: str) -> None:
        """Cancel any in-flight scene-prompts MCP job when mode switches."""
        st.session_state.mcp_scenes_job   = None
        st.session_state.mcp_scenes_start = None

    mode     = _render_mode_selector(on_change=_clear_scene_state)
    dna      = p.style_dna
    skill    = SKILLS.get(dna.skill_id, SKILLS["cinematic"])
    cam_opts = [f"{i}  {v[:60]}" for i, v in enumerate(skill.camera_vocabulary)]
    lit_opts = [f"{i}  {v[:60]}" for i, v in enumerate(skill.lighting_vocabulary)]

    # Generate scenes from story on first visit (mechanical always runs for structure)
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
        if st.button("🔄 Regenerate all", use_container_width=True, key="step5_regen_all"):
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
        if p.scenes:
            n_subj = sum(1 for s in p.scenes if getattr(s, "focus", "subject") == "subject")
            st.caption(f"🎯 Focus coverage: **{n_subj}/{len(p.scenes)}** character-focused "
                       f"· {len(p.scenes) - n_subj} subject/environment shots")

    # ── MCP Auto: batch prompt enhancement ───────────────────────────────────
    if mode == "mcp" and p.scenes:
        import dataclasses

        with st.expander("🤖 Enhance all prompts via MCP Auto", expanded=True):
            st.info(
                "Click **Enhance** to send all scenes to the MCP worker. "
                "Claude Code will return optimised visual and video prompts automatically."
            )
            if st.button("🤖 Enhance all scenes →", type="primary",
                         use_container_width=True, key="mcp_scenes_enhance"):
                def _build_scenes_job():
                    prompt  = build_scene_prompts_prompt(
                        p.idea, p.selected_story, p.character, p.style_dna, p.scenes
                    )
                    import dataclasses as _dc
                    payload = {
                        "idea":      p.idea,
                        "n_scenes":  len(p.scenes),
                        "style_dna": _dc.asdict(p.style_dna) if p.style_dna else None,
                        "character": p.character.description if p.character else "",
                        "scenes": [
                            {
                                "scene_number":       s.scene_number,
                                "act":                s.act,
                                "description":        s.description,
                                "camera":             s.camera,
                                "lighting":           s.lighting,
                                "shot_size":          s.shot_size,
                                "character_presence": getattr(s, "character_presence", "featured"),
                                "focus":              getattr(s, "focus", "subject"),
                                "focus_subject":      getattr(s, "focus_subject", ""),
                                "narrative_role":     getattr(s, "narrative_role", ""),
                                "shot_intent":        getattr(s, "shot_intent", ""),
                                "hero_moment":        bool(getattr(s, "hero_moment", False)),
                            }
                            for s in p.scenes
                        ],
                    }
                    return payload, prompt

                done = _mcp_submit_and_poll("mcp_scenes_job", "mcp_scenes_start",
                                            "scene_prompts", _build_scenes_job)
                if done is not None:
                    raw_json = json.dumps({"stage": "scene_prompts", "result": done["result"]})
                    result = parse_scene_prompts_response(raw_json)
                    if result and len(result["visual_prompts"]) == len(p.scenes):
                        for scene, vp, mp in zip(p.scenes,
                                                  result["visual_prompts"],
                                                  result["video_prompts"]):
                            scene.visual_prompt = vp
                            scene.video_prompt  = mp
                        save()
                        st.success(f"✅ Updated prompts for all {len(p.scenes)} scenes!")
                        st.rerun()
                    else:
                        st.error("Result count mismatch or parse error. Try again.")

            # Show if a job is currently in flight (button pressed previously)
            if st.session_state.get("mcp_scenes_job"):
                job_id  = st.session_state.mcp_scenes_job
                elapsed = time.time() - (st.session_state.get("mcp_scenes_start") or time.time())
                done = read_result(job_id)
                if done is not None:
                    raw_json = json.dumps({"stage": "scene_prompts", "result": done["result"]})
                    result = parse_scene_prompts_response(raw_json)
                    st.session_state.mcp_scenes_job   = None
                    st.session_state.mcp_scenes_start = None
                    if result and len(result["visual_prompts"]) == len(p.scenes):
                        for scene, vp, mp in zip(p.scenes,
                                                  result["visual_prompts"],
                                                  result["video_prompts"]):
                            scene.visual_prompt = vp
                            scene.video_prompt  = mp
                        save()
                        st.success(f"✅ Updated prompts for all {len(p.scenes)} scenes!")
                        st.rerun()
                    else:
                        st.error("Result count mismatch or parse error.")
                elif elapsed > _MCP_TIMEOUT_SECONDS:
                    st.warning("⏰ Timed out. Try again or switch to Copy-Paste mode.")
                    st.session_state.mcp_scenes_job   = None
                    st.session_state.mcp_scenes_start = None
                else:
                    st.info(f"⏳ Waiting for Claude Code… ({elapsed:.0f}s)")
                    time.sleep(3)
                    st.rerun()

    # ── Copy-paste: batch prompt enhancement ─────────────────────────────────
    if mode == "copy_paste" and p.scenes:
        st.divider()
        with st.expander("✂️ Enhance all prompts with AI (copy-paste batch)", expanded=False):
            st.info(
                "The scene structure (act labels, camera, lighting) is already set above. "
                "This generates **better visual and video prompts** by sending all scenes "
                "to an AI in one go."
            )
            batch_prompt = build_scene_prompts_prompt(
                p.idea, p.selected_story, p.character, p.style_dna, p.scenes
            )
            st.subheader("Copy this batch prompt")
            st.code(batch_prompt, language=None)

            st.subheader("Paste AI response")
            raw_batch = st.text_area(
                "Paste the full AI response here",
                height=200,
                key="cp_scenes_batch_paste",
                placeholder=(
                    '{"stage": "scene_prompts", "result": '
                    '{"visual_prompts": [...], "video_prompts": [...]}}'
                ),
            )
            col_pbatch, col_cbatch = st.columns([4, 1])
            with col_pbatch:
                if st.button("✅ Apply to all scenes →", type="primary",
                             use_container_width=True, key="cp_scenes_batch_parse"):
                    raw_val = st.session_state.get("cp_scenes_batch_paste", "").strip()
                    if not raw_val:
                        st.error("Nothing pasted.")
                    else:
                        result = parse_scene_prompts_response(raw_val)
                        if result:
                            vp = result["visual_prompts"]
                            mp = result["video_prompts"]
                            if len(vp) != len(p.scenes):
                                st.error(
                                    f"Response has {len(vp)} prompts but project has "
                                    f"{len(p.scenes)} scenes. Ask the AI to regenerate "
                                    "with the correct scene count."
                                )
                            else:
                                for scene, v, m in zip(p.scenes, vp, mp):
                                    scene.visual_prompt = v
                                    scene.video_prompt  = m
                                save()
                                st.success(
                                    f"✅ Updated prompts for all {len(p.scenes)} scenes!"
                                )
                                st.rerun()
                        else:
                            st.error(parse_error_message(raw_val, "scenes"))
            with col_cbatch:
                if st.button("Clear", use_container_width=True, key="cp_scenes_batch_clear"):
                    st.session_state["cp_scenes_batch_paste"] = ""
                    st.rerun()

    st.divider()

    # ══ Story Overview ═══════════════════════════════════════════════════════
    # At-a-glance summary of the whole arc — colour-coded focus strip, role
    # checklist, health-check warnings. Fully offline (no AI, no API).
    _story_overview(p)

    # CSS injected once for hero gradient + card spacing
    st.html("""
    <style>
      .scene-row{padding:6px 4px;margin:-6px -4px 4px;border-radius:6px;}
      .scene-hero{background:linear-gradient(90deg,#fef3c733 0%,transparent 60%);
                  border-left:4px solid #f59e0b;padding-left:8px;}
    </style>
    """)

    # ── Per-scene cards: bordered container + summary + tabs ─────────────────
    from pipeline.story_generator import _derive_presence_from_focus
    from pipeline.scene_state import VALID_NARRATIVE_ROLE
    _focus_opts = list(FOCUS_META.keys())   # ordered: subject first, etc.
    _role_opts  = [""] + sorted(VALID_NARRATIVE_ROLE)
    _presence_opts = ["featured", "background", "none"]

    for i, scene in enumerate(p.scenes):
        is_hero = bool(getattr(scene, "hero_moment", False))
        with st.container(border=True):
            # Subtle gold background strip for hero scenes (paints the row)
            wrap_class = "scene-row scene-hero" if is_hero else "scene-row"
            st.html(f"<div class='{wrap_class}'></div>")

            # ── Summary row (always visible) ────────────────────────────────
            sc_num, sc_chips, sc_actions = st.columns([0.5, 5.5, 1.5])
            with sc_num:
                star = "★" if is_hero else "·"
                star_color = "#f59e0b" if is_hero else "#9ca3af"
                act_color = ACT_COLOR.get((scene.act or "").upper().strip(), "#9ca3af")
                st.markdown(
                    f"<div style='font-size:1.4rem;font-weight:700;line-height:1.1'>"
                    f"<span style='color:{star_color}'>{star}</span> "
                    f"<span style='color:{act_color}'>#{scene.scene_number:02d}</span></div>",
                    unsafe_allow_html=True,
                )
            with sc_chips:
                st.markdown(
                    act_pill(scene.act)
                    + focus_chip(getattr(scene, "focus", "subject"))
                    + role_chip(getattr(scene, "narrative_role", ""))
                    + hero_badge(is_hero)
                    + f"<div style='color:#9ca3af;font-size:.85rem;margin-top:3px'>"
                      f"{html.escape((scene.description or '')[:120])}</div>",
                    unsafe_allow_html=True,
                )
            with sc_actions:
                ca, cb = st.columns(2)
                if ca.button("↑", key=f"up_{i}", disabled=(i == 0), help="Move up"):
                    p.scenes[i], p.scenes[i-1] = p.scenes[i-1], p.scenes[i]
                    for j, s in enumerate(p.scenes):
                        s.scene_number = j + 1
                        s.scene_id = f"scene_{j+1:02d}"
                    save(); st.rerun()
                if cb.button("↓", key=f"dn_{i}", disabled=(i == len(p.scenes) - 1),
                             help="Move down"):
                    p.scenes[i], p.scenes[i+1] = p.scenes[i+1], p.scenes[i]
                    for j, s in enumerate(p.scenes):
                        s.scene_number = j + 1
                        s.scene_id = f"scene_{j+1:02d}"
                    save(); st.rerun()

            # ── Internal tabs ───────────────────────────────────────────────
            tab_story, tab_shot, tab_special, tab_prompt = st.tabs(
                ["📖 Story", "🎥 Shot", "✨ Make it special", "✍️ Prompt"]
            )

            # Tab 1: Story (act + description + intent) ─────────────────────
            with tab_story:
                ts_c1, ts_c2 = st.columns([1, 3])
                new_act = ts_c1.text_input(
                    "Act label", value=scene.act, key=f"act_{i}",
                    help="Story beat tag — HOOK, BUILD, CLIMAX, etc.",
                )
                new_desc = ts_c2.text_input(
                    "Scene description (what we see)",
                    value=scene.description, key=f"desc_{i}",
                )
                new_intent = st.text_input(
                    "Why this shot exists (optional)",
                    value=getattr(scene, "shot_intent", "") or "",
                    key=f"intent_{i}",
                    placeholder="e.g. 'Isolate the cracked valve so it reads as significant'",
                    help="Free-text guidance the AI uses (and your future self reads).",
                )

            # Tab 2: Shot (focus + camera + lighting + presence-override) ────
            with tab_shot:
                st.markdown("**What is this shot ABOUT?**")
                _foc_cur = getattr(scene, "focus", "subject")
                chosen_focus = st.radio(
                    "Focus", _focus_opts,
                    index=_focus_opts.index(_foc_cur) if _foc_cur in _focus_opts else 0,
                    format_func=focus_label,
                    horizontal=True,
                    key=f"focus_{i}",
                    label_visibility="collapsed",
                )
                _, _, _, focus_help = FOCUS_META[chosen_focus]
                st.caption(f"💡 {focus_help}")

                if chosen_focus != "subject":
                    new_focsub = st.text_input(
                        "What fills the frame?",
                        value=getattr(scene, "focus_subject", "") or "",
                        key=f"focsub_{i}",
                        placeholder="a cracked brass valve · wind across the dunes · "
                                    "the cathedral spire at dusk",
                    )
                else:
                    new_focsub = ""  # subject focus uses the character description directly

                st.divider()
                col_cam, col_lit = st.columns(2)
                cam_cur = min(scene.camera_index, len(cam_opts) - 1)
                lit_cur = min(scene.lighting_index, len(lit_opts) - 1)
                chosen_cam = col_cam.selectbox(
                    "Camera move", cam_opts, index=cam_cur, key=f"cam_{i}",
                )
                chosen_lit = col_lit.selectbox(
                    "Lighting", lit_opts, index=lit_cur, key=f"lit_{i}",
                )

                _focus_default_pres = _derive_presence_from_focus(chosen_focus, scene.shot_size)
                cur_pres = getattr(scene, "character_presence", _focus_default_pres)
                advanced_default = cur_pres != _focus_default_pres
                if st.toggle("Override character-in-shot", value=advanced_default,
                             key=f"pres_adv_{i}",
                             help="By default, derived from Focus "
                                  f"(here: **{_focus_default_pres}**). "
                                  "Toggle to manually pick."):
                    idx = _presence_opts.index(cur_pres) if cur_pres in _presence_opts \
                          else _presence_opts.index(_focus_default_pres)
                    chosen_pres = st.radio(
                        "Character in shot", _presence_opts,
                        index=idx, horizontal=True, key=f"pres_{i}",
                    )
                else:
                    chosen_pres = _focus_default_pres

            # Tab 3: Make it special (hero + narrative role) ────────────────
            with tab_special:
                chosen_hero = st.toggle(
                    "★ Mark as the hero moment of this video",
                    value=is_hero, key=f"hero_{i}",
                    help="The visual peak. Gets extra hold time in the storytelling "
                         "montage and extra craft attention in AI prompts. "
                         "1–2 per project is ideal.",
                )
                st.caption("✨ Hero scenes hold +0.6s longer · transitions cut "
                           "around them · music ducks on the way in.")

                st.divider()
                cur_role = getattr(scene, "narrative_role", "") or ""
                chosen_role = st.selectbox(
                    "Role in the story",
                    _role_opts,
                    index=_role_opts.index(cur_role) if cur_role in _role_opts else 0,
                    format_func=lambda r: role_label(r) if r else "— none (auto from act) —",
                    key=f"nrole_{i}",
                    help="What JOB this scene does in the arc. Drives how the "
                         "storytelling montage cuts and holds.",
                )

            # Tab 4: Prompt (image prompt + offline rebuild + AI re-prompt) ──
            with tab_prompt:
                rebuild_clicked = st.button(
                    "🔧 Rebuild prompts offline — no AI, no API",
                    key=f"rebuild_{i}", type="primary", use_container_width=True,
                    help="Rewrite image & motion prompts from your focus / camera / "
                         "lighting using the built-in engine. Pure local — no Claude, "
                         "no key required.",
                )
                st.caption("🔒 Works offline · deterministic · same inputs → same prompt")

                new_prompt = st.text_area(
                    "Image prompt (visual_prompt)",
                    value=scene.visual_prompt, height=80, key=f"vp_{i}",
                )

                if mode == "copy_paste":
                    with st.expander(f"✂️ Re-prompt with AI (copy-paste)",
                                     expanded=False):
                        single_prompt = build_single_scene_prompt(
                            p.idea, scene, p.character, p.style_dna
                        )
                        st.code(single_prompt, language=None)
                        raw_single = st.text_area(
                            "Paste AI response", height=120,
                            key=f"cp_scene_paste_{i}",
                            placeholder='{"stage": "scene_prompts", "result": {…}}',
                        )
                        if st.button("✅ Apply to this scene", key=f"cp_scene_apply_{i}",
                                     type="primary", use_container_width=True):
                            raw_val = st.session_state.get(f"cp_scene_paste_{i}", "").strip()
                            if not raw_val:
                                st.error("Nothing pasted.")
                            else:
                                result = parse_scene_prompts_response(raw_val)
                                if result and result["visual_prompts"]:
                                    scene.visual_prompt = result["visual_prompts"][0]
                                    scene.video_prompt  = result["video_prompts"][0]
                                    save(); st.success("✅ Scene prompts updated!")
                                    st.rerun()
                                else:
                                    st.error(parse_error_message(raw_val, "scenes"))

            # ── Write-back (runs every rerun) ───────────────────────────────
            cam_idx = int(chosen_cam.split("  ")[0])
            lit_idx = int(chosen_lit.split("  ")[0])
            scene.act              = new_act
            scene.description      = new_desc
            scene.camera_index     = cam_idx
            scene.lighting_index   = lit_idx
            scene.camera           = skill.camera_vocabulary[cam_idx]
            scene.lighting         = skill.lighting_vocabulary[lit_idx]
            scene.focus            = chosen_focus
            scene.focus_subject    = (new_focsub or "").strip()
            scene.character_presence = chosen_pres
            scene.visual_prompt    = new_prompt
            scene.narrative_role   = chosen_role
            scene.shot_intent      = new_intent
            scene.hero_moment      = chosen_hero
            if not scene.video_prompt or scene.video_prompt == scene.description:
                scene.video_prompt = new_desc   # keep in sync until step 6

            # Rebuild click runs AFTER write-back so it uses the latest field values
            if rebuild_clicked:
                from pipeline.story_generator import (
                    _build_visual_prompt_with_framing, build_comfyui_video_prompt,
                    build_comfyui_negative,
                )
                cdesc = p.character.description if p.character else ""
                scene.visual_prompt = _build_visual_prompt_with_framing(
                    scene.shot_size or "MEDIUM SHOT", scene.description, scene.camera,
                    scene.lighting, cdesc, skill,
                    focus=scene.focus, focus_subject=scene.focus_subject,
                    character_presence=scene.character_presence,
                )
                scene.negative_prompt = build_comfyui_negative(skill)
                if scene.focus == "subject" and cdesc:
                    vbase = f"{scene.description}, {cdesc}"
                elif scene.focus_subject:
                    vbase = f"{scene.focus_subject}, {scene.description}"
                else:
                    vbase = scene.description
                scene.video_prompt = build_comfyui_video_prompt(
                    vbase, skill, scene.camera, p.style_dna.motion_style,
                )
                save(); st.rerun()

    save()

    st.divider()
    col_back, col_next = st.columns([1, 4])
    with col_back:
        if st.button("← Back", use_container_width=True, key="step5_back"):
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
    st.caption("Generate reference images for each scene using the selected T2I workflow.")

    dna = p.style_dna
    col_cfg, col_go = st.columns([3, 2])

    with col_cfg:
        p.workflow_t2i = _workflow_picker(
            kinds=("t2i",), current=p.workflow_t2i,
            key="wf_t2i_pick", label="Image workflow (T2I)",
        )
        with st.expander("🧩 Models for this workflow"):
            p.model_overrides_t2i = _model_slots_editor(
                p.workflow_t2i, p.model_overrides_t2i, key_prefix="t2i_models",
            )
        _workflow_upload_box("t2i_wf")

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
                        workflow=p.workflow_t2i,
                        model_overrides=p.model_overrides_t2i,
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
                        st.image(img_path, use_container_width=True)
                        st.markdown(
                            f"<div style='margin-top:-8px;line-height:1.6'>"
                            f"<b>S{scene.scene_number}</b> "
                            + act_pill(scene.act)
                            + focus_chip(getattr(scene, "focus", "subject"))
                            + hero_badge(getattr(scene, "hero_moment", False))
                            + "</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        f"*S{scene.scene_number} {focus_label(getattr(scene, 'focus', 'subject'))}"
                        + (" ★" if getattr(scene, "hero_moment", False) else "")
                        + " — pending*"
                    )

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

    # ── Bulk approve / unapprove ──────────────────────────────────────────────
    _approvable = [
        s for s in p.scenes
        if not s.is_approved
        and s.storyboard_images
        and any(Path(img).exists() for img in s.storyboard_images)
    ]
    col_approve_all, col_unapprove_all, _ = st.columns([2, 2, 5])
    with col_approve_all:
        if st.button(
            f"✅ Approve all ({len(_approvable)} remaining)",
            type="primary",
            use_container_width=True,
            disabled=len(_approvable) == 0,
            key="approve_all_btn",
        ):
            for s in _approvable:
                # Pick the first image that exists on disk
                first_img = next(img for img in s.storyboard_images if Path(img).exists())
                p.update_scene(s.scene_id, approved_image_path=first_img, status="approved")
            save()
            st.rerun()
    with col_unapprove_all:
        if st.button(
            "↩ Unapprove all",
            use_container_width=True,
            disabled=p.approved_scene_count == 0,
            key="unapprove_all_btn",
        ):
            for s in p.scenes:
                if s.is_approved:
                    p.update_scene(s.scene_id, approved_image_path=None, status="reviewing")
            save()
            st.rerun()

    st.divider()

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
                        workflow=p.workflow_t2i,
                        model_overrides=p.model_overrides_t2i,
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
            st.markdown(
                f"### {'✅ ' if is_approved else ''}Scene {scene.scene_number} "
                + hero_badge(getattr(scene, "hero_moment", False)),
                unsafe_allow_html=True,
            )
            st.markdown(
                act_pill(scene.act)
                + focus_chip(getattr(scene, "focus", "subject"))
                + role_chip(getattr(scene, "narrative_role", "")),
                unsafe_allow_html=True,
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
                st.markdown(
                    f"<b>S{scene.scene_number}</b> "
                    + act_pill(scene.act)
                    + focus_chip(getattr(scene, "focus", "subject"))
                    + hero_badge(getattr(scene, "hero_moment", False)),
                    unsafe_allow_html=True,
                )

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
            short = (scene.description[:40] + "…") if len(scene.description) > 40 \
                    else scene.description
            star = " ★" if getattr(scene, "hero_moment", False) else ""
            st.markdown(
                f"**S{scene.scene_number}{star}** {focus_label(getattr(scene, 'focus', 'subject'))}"
                f"  ·  `{scene.act}`<br><span style='color:#9ca3af'>{html.escape(short)}</span>",
                unsafe_allow_html=True,
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
    st.caption("Choose the video workflow, set output resolution, verify required models, then confirm.")

    # ── Video workflow selection ──────────────────────────────────────────────
    st.subheader("Video workflow")
    p.workflow_i2v = _workflow_picker(
        kinds=("i2v", "t2v"), current=p.workflow_i2v,
        key="wf_i2v_pick", label="Video workflow",
    )
    _wf_kinds = {wf.path: wf.kind for wf in discover_workflows()}
    if _wf_kinds.get(Path(p.workflow_i2v).as_posix()) == "t2v":
        st.warning(
            "⚠️ The selected workflow is **text-to-video** — it will generate "
            "directly from the video prompt and **ignore your approved storyboard images**."
        )
    with st.expander("🧩 Models for this workflow"):
        p.model_overrides_i2v = _model_slots_editor(
            p.workflow_i2v, p.model_overrides_i2v, key_prefix="i2v_models",
        )
    _workflow_upload_box("i2v_wf")

    st.divider()

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
            with st.spinner("Querying ComfyUI model folders…"):
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
                    st.write(f"**File found:** {'✅ Yes' if info.get('installed') else '❌ No'}")
                    if info.get("folder"):
                        st.caption(f"Folder checked: `models/{info['folder']}/`")
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
        col_num, col_meta, col_status, col_job = st.columns([1, 3, 2, 3])
        star = " ★" if getattr(scene, "hero_moment", False) else ""
        col_num.markdown(f"**S{scene.scene_number}**{star}")
        col_meta.markdown(
            act_pill(scene.act)
            + focus_chip(getattr(scene, "focus", "subject")),
            unsafe_allow_html=True,
        )
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
    done_scenes  = [s for s in p.scenes if s.status == "done" and s.video_local_path]
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

        col_num, col_meta, col_status, col_job, col_actions = st.columns([1, 3, 2, 2, 2])
        star = " ★" if getattr(scene, "hero_moment", False) else ""
        col_num.markdown(f"**S{scene.scene_number}**{star}")
        col_meta.markdown(
            act_pill(scene.act)
            + focus_chip(getattr(scene, "focus", "subject")),
            unsafe_allow_html=True,
        )
        col_status.write(icon_label)
        col_job.code(scene.video_job_id[:12] + "…" if scene.video_job_id else "—",
                     language=None)

        with col_actions:
            # Download if done and not yet saved locally
            if live_status == "done" and not scene.video_local_path:
                if st.button("⬇ Download", key=f"dl_{scene.scene_id}",
                             use_container_width=True):
                    with st.spinner(f"Downloading S{scene.scene_number}…"):
                        try:
                            vid_path = asyncio.run(
                                download_completed_video(client(), scene, p)
                            )
                            if vid_path:
                                scene.video_local_path = str(vid_path)
                                scene.set_status("done")
                                save()
                                st.rerun()
                            else:
                                st.warning("Download returned no file.")
                        except Exception as e:
                            st.error(f"Download failed: {e}")

            elif scene.video_local_path and Path(scene.video_local_path).exists():
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
            if statuses.get(s.scene_id, s.status) == "done" and not s.video_local_path
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
                            scene.video_local_path = str(vid_path)
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
        any_done = any(s.video_local_path and Path(s.video_local_path).exists() for s in p.scenes)
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

    ready_scenes    = [s for s in p.scenes if s.video_local_path and Path(s.video_local_path).exists()]
    missing_scenes  = [s for s in p.scenes if not s.video_local_path or not Path(s.video_local_path).exists()]

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
                    st.video(scene.video_local_path)
                    short = (scene.description[:60] + "…") if len(scene.description) > 60 \
                            else scene.description
                    st.markdown(
                        f"<b>S{scene.scene_number}</b> "
                        + act_pill(scene.act)
                        + focus_chip(getattr(scene, "focus", "subject"))
                        + hero_badge(getattr(scene, "hero_moment", False))
                        + f"<div style='color:#9ca3af;font-size:.85rem'>"
                          f"{html.escape(short)}</div>",
                        unsafe_allow_html=True,
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
        from pipeline.montage import (
            build_edit_decisions, compile_from_edit_decisions, _has_ffmpeg,
        )

        st.caption(f"Backend: **{backend}**  ·  🔒 Runs 100% locally — no API, no upload.")

        has_ffmpeg = _has_ffmpeg()
        story_default = any(getattr(s, "hero_moment", False)
                            or getattr(s, "narrative_role", "") for s in p.scenes)
        mode_choice = st.radio(
            "Compose mode",
            options=["story", "simple"] if (story_default and has_ffmpeg) else ["simple", "story"],
            format_func=lambda m: {
                "simple": "🎞 Simple — uniform transitions for every cut",
                "story":  "🎬 Storytelling compose — hero holds, cuts on payoffs · 🔒 fully offline",
            }[m],
            horizontal=False,
            key="step13_compose_mode",
        )
        if mode_choice == "story" and not has_ffmpeg:
            st.warning("⚠ Storytelling compose needs **ffmpeg** on PATH. "
                       "Install from https://ffmpeg.org/ — falling back to Simple below.")
            mode_choice = "simple"

        montage_output = Path(p.output_dir) / "montage" / f"{p.project_name}_final.mp4"

        if mode_choice == "simple":
            # ── Simple montage (existing, untouched feature) ────────────────
            col_opts1, col_opts2, col_opts3 = st.columns(3)
            with col_opts1:
                transition = st.selectbox(
                    "Transition", ["dissolve", "fade", "cut"], index=0,
                    help="dissolve = crossfade · fade = fade through black · cut = hard cut",
                )
            with col_opts2:
                t_dur = st.slider("Transition duration (s)", 0.0, 2.0, 0.5, 0.1,
                                  disabled=(transition == "cut"))
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

            if st.button("🎬 Compile simple montage", type="primary",
                         use_container_width=True, disabled=len(ready_scenes) < 2):
                video_paths = [Path(s.video_local_path) for s in ready_scenes]
                with st.spinner(
                    f"Compiling {len(video_paths)} clips with {transition} transitions…"
                ):
                    try:
                        result = compile_montage(
                            video_paths=video_paths, output_path=montage_output,
                            transition=transition, transition_duration=t_dur,
                            music_path=music_path, music_volume=music_vol,
                            fps=int(out_fps),
                        )
                        st.session_state.montage_path = str(result); save(); st.rerun()
                    except Exception as e:
                        st.error(f"Montage compilation failed: {e}")
            if len(ready_scenes) < 2:
                st.caption("⚠ Need at least 2 clips to compile a montage.")

        else:  # mode_choice == "story"
            st.info(
                "🛠 **Local engine** — reads each scene's **hero ★** flag and "
                "**narrative role** (Step 4) and plans the edit deterministically. "
                "Hero scenes hold longer · transitions cut around your reveal beats · "
                "music ducks on character moments."
            )

            # Knobs
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                base_trans = st.selectbox("Base transition",
                                          ["dissolve", "fade", "cut"], index=0,
                                          key="story_base_trans")
            with col_b:
                hero_hold_extra = st.slider(
                    "Hero hold (extra s)", 0.0, 2.0, 0.6, 0.1,
                    key="story_hero_hold",
                    help="Extra seconds the hero scene's last frame holds.",
                )
            with col_c:
                trans_dur = st.slider("Transition duration (s)", 0.0, 2.0, 0.5, 0.1,
                                      key="story_trans_dur")
            with col_d:
                out_fps = st.number_input("Output FPS", value=p.fps, step=1,
                                          min_value=8, key="story_fps")

            # Music
            music_path = None
            with st.expander("🎵 Background music (optional)"):
                uploaded = st.file_uploader(
                    "Upload audio file", type=["mp3", "wav", "ogg", "m4a"],
                    key="story_music_uploader",
                )
                if uploaded:
                    music_dir = Path(p.output_dir) / "music"
                    music_dir.mkdir(parents=True, exist_ok=True)
                    music_path = music_dir / uploaded.name
                    music_path.write_bytes(uploaded.read())
                    st.success(f"Audio loaded: {uploaded.name}")
                    st.caption("💡 Music ducks on character-led scenes when every "
                               "transition is a cut. Pick 'cut' as base transition for "
                               "the full ducking effect.")

            # Build plan from the latest scene + UI choices
            plan = build_edit_decisions(
                p.scenes,
                base_transition=base_trans,
                hero_hold_extra=float(hero_hold_extra),
                transition_duration=float(trans_dur),
                music_path=str(music_path) if music_path else None,
            )

            # Editable plan preview
            st.markdown("**Edit decisions plan**")
            st.caption(
                "💡 Computed from your hero / role / focus tags — no AI. "
                "Override any row before compiling; edits flow into the final cut."
            )
            try:
                import pandas as pd
                plan_rows = [
                    {
                        "Scene":         f"S{s.scene_number}",
                        "★":             "★" if getattr(s, "hero_moment", False) else "",
                        "Focus":         focus_label(getattr(s, "focus", "subject")),
                        "Role":          role_label(getattr(s, "narrative_role", ""))
                                         if getattr(s, "narrative_role", "") else "—",
                        "Transition in": d["transition_in"],
                        "Hold +sec":     float(d["extra_hold_seconds"]),
                        "Duck music":    bool(d["duck_music"]),
                    }
                    for s, d in zip(p.scenes, plan["scenes"])
                ]
                editor_key = (
                    f"story_plan_editor_{len(p.scenes)}_{base_trans}_"
                    f"{hero_hold_extra}_{trans_dur}"
                )
                edited = st.data_editor(
                    pd.DataFrame(plan_rows),
                    hide_index=True, use_container_width=True, num_rows="fixed",
                    column_config={
                        "Scene":         st.column_config.TextColumn(disabled=True, width="small"),
                        "★":             st.column_config.TextColumn(disabled=True, width="small"),
                        "Focus":         st.column_config.TextColumn(disabled=True),
                        "Role":          st.column_config.TextColumn(disabled=True),
                        "Transition in": st.column_config.SelectboxColumn(
                            options=["cut", "dissolve", "fade"]),
                        "Hold +sec":     st.column_config.NumberColumn(
                            min_value=0.0, max_value=3.0, step=0.1),
                        "Duck music":    st.column_config.CheckboxColumn(),
                    },
                    key=editor_key,
                )
            except Exception as e:
                st.warning(f"Couldn't render plan table: {e}")
                edited = None

            with st.expander("🔎 Show raw edit-decisions JSON"):
                st.json(plan)

            compile_disabled = len(ready_scenes) < 2
            if st.button("🎬 Compose storytelling montage", type="primary",
                         use_container_width=True, disabled=compile_disabled):
                # Apply user edits back into the plan
                if edited is not None:
                    sid_to_dec = {s.scene_id: d for s, d in zip(p.scenes, plan["scenes"])}
                    for row in edited.to_dict(orient="records"):
                        try:
                            num = int(row["Scene"][1:])
                        except ValueError:
                            continue
                        target = next(
                            (s for s in p.scenes if s.scene_number == num), None
                        )
                        if target is None:
                            continue
                        d = sid_to_dec[target.scene_id]
                        d["transition_in"]      = row["Transition in"]
                        d["extra_hold_seconds"] = float(row["Hold +sec"])
                        d["duck_music"]         = bool(row["Duck music"])

                # CRITICAL: filter plan to ready scenes only — montage.py:404
                # raises if lengths mismatch.
                ready_ids = {s.scene_id for s in ready_scenes}
                matched_plan = {
                    **plan,
                    "scenes": [d for d in plan["scenes"] if d["scene_id"] in ready_ids],
                }
                video_paths = [Path(s.video_local_path) for s in ready_scenes]
                with st.spinner(
                    f"Composing storytelling cut from {len(video_paths)} clips…"
                ):
                    try:
                        result = compile_from_edit_decisions(
                            video_paths=video_paths,
                            edit_decisions=matched_plan,
                            output_path=montage_output,
                            fps=int(out_fps),
                        )
                        # Persist the plan alongside the video for reproducibility
                        try:
                            (montage_output.with_suffix(".plan.json")).write_text(
                                json.dumps(matched_plan, indent=2), encoding="utf-8"
                            )
                        except OSError:
                            pass   # plan persistence is best-effort
                        st.session_state.montage_path = str(result); save(); st.rerun()
                    except Exception as e:
                        st.error(
                            f"Storytelling compose failed: {e}\n\n"
                            "Try switching to **Simple** mode above as a fallback."
                        )
            if compile_disabled:
                st.caption("⚠ Need at least 2 clips to compose a montage.")
            st.caption("🔒 Runs entirely on your machine via FFmpeg. "
                       "No models loaded, no API calls, no data leaves the box.")

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

# ── MCP loading intercept ──────────────────────────────────────────────────
# Check for any active MCP job BEFORE rendering the step.  If a job is in
# flight and the result isn't ready yet, render the full-screen overlay and
# re-poll every 3 s.  When the result arrives, fall through to the normal
# step render so _mcp_submit_and_poll() can consume and clear the result.
_MCP_JOB_SLOTS = [
    ("mcp_story_job",  "mcp_story_start",  "story_options"),
    ("mcp_char_job",   "mcp_char_start",   "character_description"),
    ("mcp_scenes_job", "mcp_scenes_start", "scene_prompts"),
]
_active_slot = None
for _jk, _sk, _stage in _MCP_JOB_SLOTS:
    if st.session_state.get(_jk):
        _active_slot = (_jk, _sk, _stage)
        break

if _active_slot:
    _jk, _sk, _stage = _active_slot
    _job_id = st.session_state[_jk]
    _done   = read_result(_job_id)

    if _done is None:
        # Still waiting — render full-screen overlay, sleep, rerun
        _elapsed = time.time() - (st.session_state.get(_sk) or time.time())
        _render_mcp_loading_screen(_stage, _elapsed, _job_id, _jk, _sk)
        time.sleep(2)
        st.rerun()
    else:
        # Result ready — let the step function consume it normally
        fn = STEP_FN.get(step, step_1)
        fn()
else:
    fn = STEP_FN.get(step, step_1)
    fn()
