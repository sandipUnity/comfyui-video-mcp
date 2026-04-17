"""ComfyUI Video Pipeline — Streamlit UI

Run:  ./venv/Scripts/streamlit run app.py
      then open  http://localhost:8501
"""
import asyncio, json, random, sys, time, copy
from pathlib import Path

import streamlit as st
import yaml

sys.path.insert(0, ".")
from skills_engine import SKILLS, detect_skill, build_comfyui_positive, build_comfyui_negative
from comfyui_client import ComfyUIClient

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG       = yaml.safe_load(open("config.yaml", encoding="utf-8"))
PROJECTS_DIR = Path("projects")
PROJECTS_DIR.mkdir(exist_ok=True)
WORKFLOWS    = sorted(Path("workflows").glob("*.json"))

SKILL_IDS    = ["auto"] + list(SKILLS.keys())
SKILL_NAMES  = {"auto": "Auto-detect from description"} | {k: v.name for k, v in SKILLS.items()}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ComfyUI Video Pipeline",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ────────────────────────────────────────────────────
def _default_project() -> dict:
    return {
        "project":        {"name": "My New Video", "output_prefix": "video"},
        "skill":          {"id": "auto", "detect_from": "cinematic film dramatic"},
        "generation":     {"width": 640, "height": 640, "frames": 81, "fps": 16,
                           "workflow": "workflows/wan22_lightx2v_api.json"},
        "visual_anchor":  "describe your protagonist or recurring visual motif here",
        "extra_negative": "static, motionless, frozen, text, subtitles, watermark",
        "scenes": [
            {"act": "HOOK",       "description": "Opening scene",
             "camera": 0, "lighting": 0,
             "prompt": "{visual_anchor}, opening action here"},
            {"act": "BUILD",      "description": "Tension rises",
             "camera": 1, "lighting": 1,
             "prompt": "same protagonist — {visual_anchor}, tension escalating"},
            {"act": "CLIMAX",     "description": "Peak moment",
             "camera": 2, "lighting": 2,
             "prompt": "same protagonist — {visual_anchor}, climactic confrontation"},
            {"act": "RESOLUTION", "description": "Aftermath",
             "camera": 3, "lighting": 0,
             "prompt": "same protagonist — {visual_anchor}, resolution and calm"},
        ]
    }

if "proj" not in st.session_state:
    st.session_state.proj = _default_project()
if "queue_result" not in st.session_state:
    st.session_state.queue_result = None
if "dry_result" not in st.session_state:
    st.session_state.dry_result = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def resolve_skill(proj: dict):
    sid = proj["skill"]["id"]
    if sid == "auto":
        return detect_skill(proj["skill"].get("detect_from", "cinematic"))
    return SKILLS.get(sid, SKILLS["cinematic"])


def build_scenes(proj: dict, skill) -> list[dict]:
    anchor    = proj.get("visual_anchor", "").strip()
    extra_neg = proj.get("extra_negative", "")
    scenes    = []
    for raw in proj["scenes"]:
        base = raw["prompt"].strip().replace("{visual_anchor}", anchor)
        cam_idx  = raw.get("camera",   0)
        lite_idx = raw.get("lighting", 0)
        cam_vocab  = skill.camera_vocabulary
        lite_vocab = skill.lighting_vocabulary
        cam_idx  = min(cam_idx,  len(cam_vocab)  - 1)
        lite_idx = min(lite_idx, len(lite_vocab) - 1)
        full_base = f"{base}, {cam_vocab[cam_idx]}, {lite_vocab[lite_idx]}"
        scenes.append({
            "scene_number":    len(scenes) + 1,
            "act":             raw.get("act", f"S{len(scenes)+1}"),
            "description":     raw.get("description", ""),
            "visual_prompt":   build_comfyui_positive(full_base, skill),
            "negative_prompt": build_comfyui_negative(skill, custom_negative=extra_neg),
            "gen":             proj.get("generation", {}),
        })
    return scenes


def fill_workflow(scene: dict, wf_template: str, output_prefix: str) -> dict:
    gen    = scene["gen"]
    seed   = random.randint(0, 2**31 - 1)
    prefix = f"{output_prefix}_s{scene['scene_number']}_{int(time.time())}"

    # Step 1: replace only NUMERIC placeholders via string replace (safe — no special chars).
    t = wf_template
    for ph, val in {
        "{{WIDTH}}":  str(gen.get("width",  640)),
        "{{HEIGHT}}": str(gen.get("height", 640)),
        "{{FRAMES}}": str(gen.get("frames", 81)),
        "{{FPS}}":    str(gen.get("fps",    16)),
        "{{SEED}}":   str(seed),
    }.items():
        t = t.replace(ph, val)

    # Step 2: parse JSON — string placeholders like "{{POSITIVE_PROMPT}}" are valid JSON strings.
    wf = json.loads(t)

    # Step 3: walk the dict and swap placeholder strings → real values.
    # No JSON escaping ever touches the prompt text — immune to em-dashes, quotes, newlines, etc.
    str_map = {
        "{{POSITIVE_PROMPT}}": scene["visual_prompt"],
        "{{NEGATIVE_PROMPT}}": scene["negative_prompt"],
        "{{OUTPUT_PREFIX}}":   prefix,
    }

    def _inject(d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                _inject(v)
            elif isinstance(v, str) and v in str_map:
                d[k] = str_map[v]

    _inject(wf)
    return wf


async def _get_queue_status():
    host, port = CONFIG["comfyui"]["host"], CONFIG["comfyui"]["port"]
    client = ComfyUIClient(host, port)
    return await client.get_queue_status()


async def _queue_all(proj: dict, scenes: list, skill) -> list[dict]:
    host, port = CONFIG["comfyui"]["host"], CONFIG["comfyui"]["port"]
    client = ComfyUIClient(host, port)
    wf_path    = proj["generation"].get("workflow", "workflows/wan22_lightx2v_api.json")
    wf_template = open(wf_path).read()
    output_prefix = proj["project"].get("output_prefix", "video")
    results = []
    for sc in scenes:
        try:
            wf  = fill_workflow(sc, wf_template, output_prefix)
            pid = await client.queue_prompt(wf)
            results.append({"scene": sc["scene_number"], "act": sc["act"],
                            "description": sc["description"], "job_id": pid, "status": "queued"})
        except Exception as e:
            results.append({"scene": sc["scene_number"], "act": sc["act"],
                            "description": sc["description"], "job_id": None,
                            "status": f"FAILED: {e}"})
        await asyncio.sleep(0.3)
    return results


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎬 Video Pipeline")
    st.caption("ComfyUI · Wan2.2 · LightX2V")
    st.divider()

    # ComfyUI status
    host = CONFIG["comfyui"]["host"]
    port = CONFIG["comfyui"]["port"]
    if st.button("⟳  Refresh queue status", use_container_width=True):
        try:
            q = asyncio.run(_get_queue_status())
            r = len(q.get("queue_running", []))
            p = len(q.get("queue_pending", []))
            st.success(f"Online — {r} running, {p} pending")
        except Exception as e:
            st.error(f"Offline: {e}")

    st.caption(f"Server: http://{host}:{port}")
    st.link_button("Open ComfyUI Monitor", f"http://{host}:{port}", use_container_width=True)

    st.divider()

    # Project file management
    st.subheader("Projects")
    yaml_files = sorted(PROJECTS_DIR.glob("*.yaml"))
    yaml_names = [f.stem for f in yaml_files]

    if yaml_names:
        selected = st.selectbox("Load project", ["— new —"] + yaml_names)
        if selected != "— new —" and st.button("Load", use_container_width=True):
            with open(PROJECTS_DIR / f"{selected}.yaml", encoding="utf-8") as f:
                st.session_state.proj = yaml.safe_load(f)
            st.session_state.queue_result = None
            st.session_state.dry_result   = None
            st.rerun()
    else:
        st.info("No saved projects yet.")

    if st.button("New blank project", use_container_width=True):
        st.session_state.proj = _default_project()
        st.session_state.queue_result = None
        st.session_state.dry_result   = None
        st.rerun()

    st.divider()

    # Save current project
    save_name = st.text_input("Save as", value=st.session_state.proj["project"].get("output_prefix", "video"))
    if st.button("💾  Save YAML", use_container_width=True):
        dest = PROJECTS_DIR / f"{save_name}.yaml"
        with open(dest, "w", encoding="utf-8") as f:
            yaml.dump(st.session_state.proj, f, allow_unicode=True, sort_keys=False)
        st.success(f"Saved → projects/{save_name}.yaml")

    # Download YAML
    yaml_str = yaml.dump(st.session_state.proj, allow_unicode=True, sort_keys=False)
    st.download_button("⬇  Download YAML", yaml_str,
                       file_name=f"{save_name}.yaml", mime="text/yaml",
                       use_container_width=True)


# ── Main area ─────────────────────────────────────────────────────────────────
proj  = st.session_state.proj
skill = resolve_skill(proj)

st.title(proj["project"].get("name", "Untitled Project"))
st.caption(f"Skill: **{skill.name}** (id=`{skill.id}`)  ·  "
           f"{len(proj.get('scenes', []))} scenes  ·  "
           f"~{len(proj.get('scenes', [])) * round(proj['generation']['frames'] / proj['generation']['fps'], 1):.0f}s total")

tab_settings, tab_scenes, tab_preview, tab_queue = st.tabs(
    ["⚙️ Settings", "🎬 Scenes", "👁 Preview Prompts", "🚀 Queue to ComfyUI"]
)


# ─── Tab 1: Settings ──────────────────────────────────────────────────────────
with tab_settings:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Project")
        proj["project"]["name"] = st.text_input(
            "Project name", value=proj["project"].get("name", ""))
        proj["project"]["output_prefix"] = st.text_input(
            "Output file prefix  (e.g. airise → airise_s1_…)",
            value=proj["project"].get("output_prefix", "video"))

        st.subheader("Skill")
        skill_labels = [f"{sid}  —  {SKILL_NAMES[sid]}" for sid in SKILL_IDS]
        cur_sid = proj["skill"].get("id", "auto")
        cur_idx = SKILL_IDS.index(cur_sid) if cur_sid in SKILL_IDS else 0
        chosen_label = st.selectbox("Skill", skill_labels, index=cur_idx)
        proj["skill"]["id"] = chosen_label.split("  —  ")[0].strip()

        if proj["skill"]["id"] == "auto":
            proj["skill"]["detect_from"] = st.text_input(
                "Auto-detect from text",
                value=proj["skill"].get("detect_from", "cinematic film dramatic"),
                help="The engine scores keywords to pick the best-matching skill")
            skill = resolve_skill(proj)
            st.info(f"Auto-detected: **{skill.name}** (id=`{skill.id}`)")

    with col2:
        st.subheader("Generation")
        gen = proj["generation"]
        c1, c2 = st.columns(2)
        gen["width"]  = c1.number_input("Width",  value=gen.get("width",  640), step=64, min_value=256, max_value=1920)
        gen["height"] = c2.number_input("Height", value=gen.get("height", 640), step=64, min_value=256, max_value=1080)
        gen["frames"] = st.slider("Frames per scene", 17, 161, value=gen.get("frames", 81), step=8,
                                  help="81 frames / 16 fps ≈ 5 s  |  161 frames ≈ 10 s")
        gen["fps"]    = st.select_slider("FPS", options=[8, 12, 16, 24, 30], value=gen.get("fps", 16))

        wf_names = [str(w) for w in WORKFLOWS]
        cur_wf   = gen.get("workflow", "workflows/wan22_lightx2v_api.json")
        wf_idx   = wf_names.index(cur_wf) if cur_wf in wf_names else 0
        gen["workflow"] = st.selectbox("Workflow", wf_names, index=wf_idx)

        secs_per = round(gen["frames"] / gen["fps"], 1)
        total    = secs_per * len(proj.get("scenes", []))
        st.metric("Duration", f"{secs_per}s / scene",
                  delta=f"{total:.0f}s total ({len(proj.get('scenes',[]))} scenes)")

        st.subheader("Visual Anchor")
        proj["visual_anchor"] = st.text_area(
            "Character or recurring visual motif — use {visual_anchor} in scene prompts",
            value=proj.get("visual_anchor", ""),
            height=100)

        st.subheader("Extra Negative Tags")
        proj["extra_negative"] = st.text_input(
            "Added on top of skill's built-in negatives",
            value=proj.get("extra_negative", "static, motionless, frozen, text, subtitles"))


# ─── Tab 2: Scenes ────────────────────────────────────────────────────────────
with tab_scenes:
    cam_vocab  = skill.camera_vocabulary
    lite_vocab = skill.lighting_vocabulary
    cam_options  = [f"{i}  {v}" for i, v in enumerate(cam_vocab)]
    lite_options = [f"{i}  {v}" for i, v in enumerate(lite_vocab)]

    # Add / remove scene buttons
    col_add, col_remove, _ = st.columns([1, 1, 4])
    if col_add.button("➕ Add scene"):
        proj["scenes"].append({
            "act": f"SCENE{len(proj['scenes'])+1}",
            "description": "New scene",
            "camera": 0, "lighting": 0,
            "prompt": "{visual_anchor}, describe what happens here"
        })
    if col_remove.button("➖ Remove last") and len(proj["scenes"]) > 1:
        proj["scenes"].pop()

    st.divider()

    for i, scene in enumerate(proj["scenes"]):
        with st.expander(f"Scene {i+1}  ·  {scene.get('act','?')}  ·  {scene.get('description','')}", expanded=True):
            row1, row2 = st.columns([1, 3])
            scene["act"]         = row1.text_input(f"Act label##{i}",         value=scene.get("act", ""), key=f"act_{i}")
            scene["description"] = row2.text_input(f"One-line description##{i}", value=scene.get("description", ""), key=f"desc_{i}")

            cam_idx  = min(scene.get("camera",   0), len(cam_options)  - 1)
            lite_idx = min(scene.get("lighting", 0), len(lite_options) - 1)
            c1, c2 = st.columns(2)
            chosen_cam  = c1.selectbox(f"Camera movement##{i}",  cam_options,  index=cam_idx,  key=f"cam_{i}")
            chosen_lite = c2.selectbox(f"Lighting##{i}",         lite_options, index=lite_idx, key=f"lite_{i}")
            scene["camera"]   = int(chosen_cam.split("  ")[0])
            scene["lighting"] = int(chosen_lite.split("  ")[0])

            scene["prompt"] = st.text_area(
                f"Scene prompt##{i}",
                value=scene.get("prompt", ""),
                height=100, key=f"prompt_{i}",
                help="Use {visual_anchor} to insert the visual anchor text automatically")


# ─── Tab 3: Preview Prompts ───────────────────────────────────────────────────
with tab_preview:
    st.info("Prompts are built by combining your scene text + camera/lighting + skill quality boosters and style tags. Nothing is sent to ComfyUI.")

    if st.button("🔍  Build preview", type="primary", use_container_width=True):
        skill = resolve_skill(proj)
        scenes = build_scenes(proj, skill)
        st.session_state.dry_result = scenes

    if st.session_state.dry_result:
        scenes = st.session_state.dry_result
        st.success(f"Built {len(scenes)} scenes — skill: **{skill.name}**")

        # Skill injections summary
        with st.expander("Skill injections (added automatically to every scene)"):
            st.write("**Quality boosters:**", ", ".join(skill.quality_boosters))
            st.write("**Style tags:**", ", ".join(skill.style_tags))
            st.write("**Negative tags:**", ", ".join(skill.negative_tags))

        for sc in scenes:
            st.subheader(f"Scene {sc['scene_number']} — {sc['act']} — {sc['description']}")
            col_pos, col_neg = st.columns(2)
            with col_pos:
                st.caption("POSITIVE PROMPT")
                st.text_area(f"pos_{sc['scene_number']}", sc["visual_prompt"],
                             height=160, label_visibility="collapsed", disabled=True)
            with col_neg:
                st.caption("NEGATIVE PROMPT")
                st.text_area(f"neg_{sc['scene_number']}", sc["negative_prompt"],
                             height=160, label_visibility="collapsed", disabled=True)


# ─── Tab 4: Queue to ComfyUI ─────────────────────────────────────────────────
with tab_queue:
    gen    = proj["generation"]
    fps    = gen.get("fps", 16)
    frames = gen.get("frames", 81)
    n      = len(proj.get("scenes", []))
    secs   = round(frames / fps, 1)
    eta    = round(108 * n / 60)

    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.info(
            f"**{n} scenes** · {secs}s each · **{secs*n:.0f}s total video**  |  "
            f"Model: Wan2.2 + LightX2V  |  "
            f"Size: {gen.get('width',640)}×{gen.get('height',640)}  |  "
            f"ETA: ~{eta} min on RTX 4090D"
        )

    with col_btn:
        queue_clicked = st.button("🚀  Queue all scenes", type="primary",
                                  use_container_width=True, key="queue_btn")

    if queue_clicked:
        skill  = resolve_skill(proj)
        scenes = build_scenes(proj, skill)
        with st.spinner(f"Queueing {len(scenes)} scenes to ComfyUI…"):
            try:
                results = asyncio.run(_queue_all(proj, scenes, skill))
                st.session_state.queue_result = results
            except Exception as e:
                st.error(f"Connection failed: {e}")

    if st.session_state.queue_result:
        results = st.session_state.queue_result
        ok  = [r for r in results if r["status"] == "queued"]
        bad = [r for r in results if r["status"] != "queued"]

        if ok:
            st.success(f"✅  Queued {len(ok)}/{len(results)} scenes successfully")
        if bad:
            st.error(f"❌  {len(bad)} scene(s) failed")

        st.subheader("Job IDs")
        for r in results:
            icon = "✅" if r["status"] == "queued" else "❌"
            col_a, col_b = st.columns([2, 3])
            col_a.write(f"{icon} Scene {r['scene']} · **{r['act']}** · {r['description']}")
            if r["job_id"]:
                col_b.code(r["job_id"])
            else:
                col_b.write(r["status"])

        st.divider()
        col_mon, col_out = st.columns(2)
        col_mon.link_button("📺  Open ComfyUI Monitor",
                            f"http://{host}:{port}", use_container_width=True)
        col_out.info(f"Output files: `ComfyUI/output/`  prefix: `{proj['project'].get('output_prefix','video')}_s*`")

        if st.button("Clear results"):
            st.session_state.queue_result = None
            st.rerun()
