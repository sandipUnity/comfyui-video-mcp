"""Universal video project runner.

Usage
-----
  python run_project.py projects/ai_rise.yaml          # queue video to ComfyUI
  python run_project.py projects/ai_rise.yaml --dry    # print prompts, do NOT queue
  python run_project.py --list-skills                  # show all available skill IDs
  python run_project.py --list-skill 3d_cgi            # show camera/lighting vocab for a skill
  python run_project.py --new my_story                 # copy blank template → projects/my_story.yaml

Edit a YAML file in projects/ and re-run — no Claude needed.
"""
import asyncio, argparse, json, random, sys, time, io, shutil, yaml
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

from comfyui_client import ComfyUIClient
from skills_engine import detect_skill, build_comfyui_positive, build_comfyui_negative, SKILLS

CONFIG = yaml.safe_load(open("config.yaml", encoding="utf-8"))


# ── Helpers ──────────────────────────────────────────────────────────────────

def list_skills():
    print("\nAvailable skill IDs\n" + "=" * 40)
    for sid, spec in SKILLS.items():
        print(f"  {sid:<30} {spec.name}")
    print()
    print("Use any of these as  skill.id  in your YAML project file.")
    print("Set  skill.id: auto  to auto-detect from  skill.detect_from  text.")
    print()


def show_skill(sid: str):
    if sid not in SKILLS:
        print(f"[ERROR] Unknown skill '{sid}'. Run --list-skills to see options.")
        sys.exit(1)
    s = SKILLS[sid]
    print(f"\nSkill: {s.name}  (id={s.id})\n" + "=" * 60)
    print(f"\nCamera vocabulary (use index as  camera:  in scene)")
    for i, c in enumerate(s.camera_vocabulary):
        print(f"  {i}  {c}")
    print(f"\nLighting vocabulary (use index as  lighting:  in scene)")
    for i, c in enumerate(s.lighting_vocabulary):
        print(f"  {i}  {c}")
    print(f"\nQuality boosters injected automatically:")
    print(f"  {', '.join(s.quality_boosters)}")
    print(f"\nStyle tags injected automatically:")
    print(f"  {', '.join(s.style_tags)}")
    print(f"\nNegative tags injected automatically:")
    print(f"  {', '.join(s.negative_tags[:6])}, ...")
    print(f"\nTechnical specs: {s.technical_specs}")
    print()


def create_new(name: str):
    dest = Path(f"projects/{name}.yaml")
    if dest.exists():
        print(f"[ERROR] {dest} already exists. Choose a different name.")
        sys.exit(1)
    src = Path("projects/template_anime.yaml")
    if not src.exists():
        src = next(Path("projects").glob("template_*.yaml"), None)
    if src is None:
        print("[ERROR] No template found in projects/. Cannot create new project.")
        sys.exit(1)
    shutil.copy(src, dest)
    print(f"Created: {dest}")
    print(f"Edit it and run:  python run_project.py {dest}")
    print()


def load_project(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)
    with open(p, encoding="utf-8") as f:
        proj = yaml.safe_load(f)
    return proj


def resolve_skill(proj: dict):
    skill_cfg = proj.get("skill", {})
    sid = skill_cfg.get("id", "auto")
    if sid == "auto":
        detect_from = skill_cfg.get("detect_from", "cinematic")
        skill = detect_skill(detect_from)
    else:
        if sid not in SKILLS:
            print(f"[ERROR] Unknown skill id '{sid}'. Run --list-skills to see options.")
            sys.exit(1)
        skill = SKILLS[sid]
    return skill


def build_scenes(proj: dict, skill) -> list[dict]:
    gen    = proj.get("generation", {})
    anchor = proj.get("visual_anchor", "").strip()
    extra_neg = proj.get("extra_negative", "")

    scenes = []
    for raw in proj["scenes"]:
        base_prompt = raw["prompt"].strip()

        # Replace {visual_anchor} placeholder
        if "{visual_anchor}" in base_prompt:
            base_prompt = base_prompt.replace("{visual_anchor}", anchor)

        # Append camera and lighting from skill vocabulary using index
        cam_idx  = raw.get("camera",   0)
        lite_idx = raw.get("lighting", 0)

        cam_vocab  = skill.camera_vocabulary
        lite_vocab = skill.lighting_vocabulary

        if cam_idx >= len(cam_vocab):
            print(f"[WARN] camera index {cam_idx} out of range for skill '{skill.id}' "
                  f"(max {len(cam_vocab)-1}), using 0")
            cam_idx = 0
        if lite_idx >= len(lite_vocab):
            print(f"[WARN] lighting index {lite_idx} out of range for skill '{skill.id}' "
                  f"(max {len(lite_vocab)-1}), using 0")
            lite_idx = 0

        full_base = f"{base_prompt}, {cam_vocab[cam_idx]}, {lite_vocab[lite_idx]}"

        scenes.append({
            "scene_number":    len(scenes) + 1,
            "act":             raw.get("act", f"SCENE{len(scenes)+1}"),
            "description":     raw.get("description", ""),
            "visual_prompt":   build_comfyui_positive(full_base, skill),
            "negative_prompt": build_comfyui_negative(skill, custom_negative=extra_neg),
            "gen":             gen,
        })
    return scenes


def fill_workflow(scene: dict, wf_template: str, output_prefix: str) -> dict:
    gen    = scene["gen"]
    seed   = random.randint(0, 2**31 - 1)
    prefix = f"{output_prefix}_s{scene['scene_number']}_{int(time.time())}"

    # Step 1: replace only NUMERIC placeholders via string replace (safe).
    t = wf_template
    for ph, val in {
        "{{WIDTH}}":  str(gen.get("width",  640)),
        "{{HEIGHT}}": str(gen.get("height", 640)),
        "{{FRAMES}}": str(gen.get("frames", 81)),
        "{{FPS}}":    str(gen.get("fps",    16)),
        "{{SEED}}":   str(seed),
    }.items():
        t = t.replace(ph, val)

    # Step 2: parse JSON — string placeholders are valid JSON strings at this point.
    wf = json.loads(t)

    # Step 3: inject string values directly into the parsed dict — no escaping needed.
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


def print_dry_run(proj: dict, scenes: list, skill):
    gen = proj.get("generation", {})
    print()
    print("=" * 72)
    print(f"  DRY RUN — {proj['project']['name']}")
    print("=" * 72)
    print(f"  Skill    : {skill.name}  (id={skill.id})")
    print(f"  Size     : {gen.get('width',640)}x{gen.get('height',640)}  "
          f"{gen.get('frames',81)} frames  {gen.get('fps',16)} fps")
    print(f"  Workflow : {gen.get('workflow','workflows/wan22_lightx2v_api.json')}")
    print()
    for sc in scenes:
        print(f"  [{sc['act']:<12}] Scene {sc['scene_number']} — {sc['description']}")
        print(f"  POSITIVE:\n    {sc['visual_prompt'][:200]}...")
        print(f"  NEGATIVE:\n    {sc['negative_prompt'][:100]}...")
        print()


# ── Async queue ───────────────────────────────────────────────────────────────

async def queue_scenes(proj: dict, scenes: list, skill):
    host = CONFIG["comfyui"]["host"]
    port = CONFIG["comfyui"]["port"]
    client = ComfyUIClient(host, port)

    gen            = proj.get("generation", {})
    output_prefix  = proj["project"].get("output_prefix", "video")
    wf_path        = gen.get("workflow", "workflows/wan22_lightx2v_api.json")
    wf_template    = open(wf_path).read()

    name    = proj["project"]["name"]
    n       = len(scenes)
    fps     = gen.get("fps", 16)
    frames  = gen.get("frames", 81)
    secs    = round(frames / fps, 1)
    total_s = secs * n

    print()
    print("=" * 72)
    print(f"  generate_video() — {name}  [{n} scenes, {total_s:.0f}s total]")
    print("=" * 72)
    print(f"  Skill   : {skill.name}  (id={skill.id})")
    print(f"  Quality : {', '.join(skill.quality_boosters[:4])}")
    print(f"  Size    : {gen.get('width',640)}x{gen.get('height',640)}  "
          f"{frames} frames  {fps} fps (~{secs}s/scene)")
    print(f"  Workflow: {wf_path}")
    print(f"  ETA     : ~108s/scene on RTX 4090D  "
          f"(~{round(108*n/60)} min total)")
    print(f"  Server  : http://{host}:{port}")
    print("=" * 72)
    print()

    try:
        q = await client.get_queue_status()
        r = len(q.get("queue_running", []))
        p = len(q.get("queue_pending", []))
        print(f"  ComfyUI online — queue: {r} running, {p} pending")
        print()
    except Exception as e:
        print(f"  [ERROR] Cannot reach ComfyUI at {host}:{port}: {e}")
        return

    job_ids = {}
    for sc in scenes:
        try:
            wf  = fill_workflow(sc, wf_template, output_prefix)
            pid = await client.queue_prompt(wf)
            job_ids[sc["scene_number"]] = pid
            print(f"  [{sc['act']:<12}] Scene {sc['scene_number']} — {sc['description']}")
            print(f"                job_id : {pid}")
            print()
        except Exception as e:
            print(f"  [FAILED] Scene {sc['scene_number']} — {e}")
        await asyncio.sleep(0.4)

    q2 = await client.get_queue_status()
    print("=" * 72)
    print(f"  Queued {len(job_ids)}/{n}  |  "
          f"{len(q2.get('queue_running',[]))} running  "
          f"{len(q2.get('queue_pending',[]))} pending")
    print("=" * 72)
    print(f"\n  Monitor : http://{host}:{port}")
    print(f"  Output  : ComfyUI/output/  (prefix: {output_prefix}_s*)\n")
    for sn, pid in job_ids.items():
        act = scenes[sn-1]["act"]
        print(f"    Scene {sn} [{act:<12}]  ->  {pid}")


# ── CLI entry-point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Queue a YAML video project to ComfyUI — no Claude needed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_project.py projects/ai_rise.yaml
  python run_project.py projects/ai_rise.yaml --dry
  python run_project.py --list-skills
  python run_project.py --list-skill 3d_cgi
  python run_project.py --new my_zombie_film
        """
    )
    parser.add_argument("project",        nargs="?",       help="Path to .yaml project file")
    parser.add_argument("--dry",          action="store_true", help="Print prompts but do NOT queue to ComfyUI")
    parser.add_argument("--list-skills",  action="store_true", help="Show all available skill IDs and exit")
    parser.add_argument("--list-skill",   metavar="SKILL_ID",  help="Show camera/lighting vocab for a skill")
    parser.add_argument("--new",          metavar="NAME",       help="Create a new project YAML from template")

    args = parser.parse_args()

    if args.list_skills:
        list_skills()
        return

    if args.list_skill:
        show_skill(args.list_skill)
        return

    if args.new:
        create_new(args.new)
        return

    if not args.project:
        parser.print_help()
        return

    proj  = load_project(args.project)
    skill = resolve_skill(proj)
    scenes = build_scenes(proj, skill)

    if args.dry:
        print_dry_run(proj, scenes, skill)
    else:
        asyncio.run(queue_scenes(proj, scenes, skill))


if __name__ == "__main__":
    main()
