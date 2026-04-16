"""Queue all 4 Digital Dreamscape scenes to ComfyUI.

Story:     A woman discovers her reality is a simulation and must break free.
Arc:       HOOK → BUILD → CLIMAX → RESOLUTION

Skill:     anime  (auto-detected from "tokyo cyberpunk anime")
           Pulls camera vocabulary, lighting vocabulary, quality_boosters,
           negative_tags, and generation overrides directly from the skill spec.

Uses: Wan2.2 T2V 14B + LightX2V 4-step LoRA (two-stage cascade)
"""
import asyncio, yaml, sys, json, random, time, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

from comfyui_client import ComfyUIClient
from skills_engine import detect_skill, build_comfyui_positive, build_comfyui_negative

CONFIG = yaml.safe_load(open("config.yaml", encoding="utf-8"))

# ── Detect skill from the project's creative brief ────────────────────────────
SKILL = detect_skill("tokyo cyberpunk anime virtual reality")
# → resolves to SkillSpec(id="anime", ...)

# ── Workflow resolution/fps from skill's technical_specs ─────────────────────
# Wan2.2 works best at 640×640; skill fps used for the video encoder
WIDTH  = 640    # Wan2.2 LightX2V sweet-spot
HEIGHT = 640
FRAMES = 81     # 81 ÷ 16 fps ≈ 5 s of video
FPS    = SKILL.technical_specs.get("fps", 16)   # anime: 24 fps

# ── Protagonist anchor — one description, locked across ALL 4 scenes ──────────
PROTAGONIST = (
    "young woman, sharp black bob haircut with violet streak, "
    "worn white oversized jacket with circuit-board pattern, "
    "cracked holographic visor pushed up on forehead, "
    "pale skin, dark circles under eyes"
)

# ── Scene base prompts ─────────────────────────────────────────────────────────
# Each base = PROTAGONIST + action + camera (from skill) + lighting (from skill)
# build_comfyui_positive() then appends the skill's quality_boosters + style_tags
# build_comfyui_negative() appends the skill's negative_tags on top of base negatives
#
# skill.camera_vocabulary (anime):
#   [0] "dramatic pull-back from extreme close-up to full scene reveal"
#   [1] "impact frame freeze: single key frame held 0.3s with speed lines radiating outward"
#   [2] "slow-motion 0.2x speed during emotional climax"
#   [3] "dynamic Dutch tilt 25° during intense battle"
#   [4] "smear frame transition between action positions"
#   [5] "static establishing shot with single moving element (hair, leaves)"
#   [6] "spiral camera orbit at power-up moment"
#   [7] "eye-level intimate shot for emotional dialogue"
#
# skill.lighting_vocabulary (anime):
#   [0] "shonen style: 150% saturated warm amber sunlight from 45° above"
#   [1] "seinen style: desaturated cool 8000K with 60% contrast harsh shadows"
#   [2] "magical girl: soft pink 3200K omnidirectional sparkle glow"
#   [3] "mecha: cold industrial 6500K with neon cyan/magenta accent lights"
#   [4] "slice of life: golden hour 2800K dappled through window, dust motes"
#   [5] "cyberpunk anime: dark base with 100% neon cyan 16000K and magenta 7000K"
#   [6] "dramatic power-up: internal white glow expanding outward, 6 second ramp"

cam  = SKILL.camera_vocabulary      # shorthand
lite = SKILL.lighting_vocabulary

_SCENE_BASES = [
    # ── Scene 1: HOOK (10% tension) ──────────────────────────────────────────
    # Camera [0]: dramatic pull-back — wide establish
    # Lighting [1]: seinen desaturated — clinical, eerie calm
    (
        f"{PROTAGONIST}, "
        "eyes snapping open from black, slow blink in pristine white VR pod, "
        f"{cam[0]}, "
        f"{lite[1]}, "
        "subtle cyan rim light from pod edges"
    ),

    # ── Scene 2: BUILD (40% tension) ─────────────────────────────────────────
    # Camera [5]: static shot with one moving element — world stutters
    # Lighting [5]: cyberpunk neon — world looks wrong, too vivid
    (
        f"same protagonist — {PROTAGONIST}, "
        "striding down rain-slicked street, pausing as holographic buildings flicker and tear, "
        f"{cam[5]}, hair and neon signs moving while everything else freezes, "
        f"{lite[5]}, "
        "speed lines at frame edges indicating motion"
    ),

    # ── Scene 3: CLIMAX (90% tension) ────────────────────────────────────────
    # Camera [3]: Dutch tilt 25° — maximum instability
    # Lighting [3]: mecha industrial + explosion rim — threat feels mechanical, cold
    (
        f"same protagonist — {PROTAGONIST}, jacket shoulder now torn, visor cracked further, "
        "staring upward at colossal mechanical entity descending through shattered skyline, "
        f"{cam[3]}, "
        f"{lite[3]}, orange explosion rim light 2400K from below, aura pulses radiating outward, "
        "shrapnel sparks filling frame"
    ),

    # ── Scene 4: RESOLUTION (60% tension, falling) ───────────────────────────
    # Camera [2]: slow-motion during emotional climax — contemplative
    # Lighting [6]: power-up glow → reality dissolving into light
    (
        f"same protagonist — {PROTAGONIST}, jacket torn, visor glowing faintly, "
        "kneeling on fragmented digital ground as code particles float upward, "
        f"{cam[2]}, "
        f"{lite[6]}, "
        "violet bioluminescent particles 380nm wavelength, flowing hair lifting against gravity"
    ),
]

_ACT_LABELS    = ["HOOK", "BUILD", "CLIMAX", "RESOLUTION"]
_DESCRIPTIONS  = [
    "Yuki awakens in a VR pod — reality feels perfect and safe",
    "Yuki walks neon streets and notices the world glitching around her",
    "Yuki faces the simulation's guardian — a towering mechanical entity",
    "Yuki kneels in collapsing simulation space as reality dissolves into light",
]

# ── Assemble scenes using skill builder functions ─────────────────────────────
SCENES = []
for i, base in enumerate(_SCENE_BASES):
    SCENES.append({
        "scene_number":  i + 1,
        "act":           _ACT_LABELS[i],
        "description":   _DESCRIPTIONS[i],
        # build_comfyui_positive appends SKILL.quality_boosters[:6] + SKILL.style_tags[:3]
        "visual_prompt": build_comfyui_positive(base, SKILL),
        # build_comfyui_negative prepends base negatives then appends SKILL.negative_tags
        "negative_prompt": build_comfyui_negative(
            SKILL,
            custom_negative="static, motionless, frozen, no movement, text, subtitles"
        ),
    })

# ── Workflow template ─────────────────────────────────────────────────────────
WF_TEMPLATE = open("workflows/wan22_lightx2v_api.json").read()


def fill_workflow(scene: dict) -> dict:
    seed   = random.randint(0, 2**31 - 1)
    prefix = f"dreamscape_s{scene['scene_number']}_{int(time.time())}"
    t = WF_TEMPLATE
    for ph, val in {
        "{{POSITIVE_PROMPT}}": scene["visual_prompt"],
        "{{NEGATIVE_PROMPT}}": scene["negative_prompt"],
        "{{WIDTH}}":           str(WIDTH),
        "{{HEIGHT}}":          str(HEIGHT),
        "{{FRAMES}}":          str(FRAMES),
        "{{FPS}}":             str(FPS),
        "{{SEED}}":            str(seed),
        "{{OUTPUT_PREFIX}}":   prefix,
    }.items():
        t = t.replace(ph, val)
    return json.loads(t)


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    host = CONFIG["comfyui"]["host"]
    port = CONFIG["comfyui"]["port"]
    client = ComfyUIClient(host, port)

    print("=" * 68)
    print("  generate_video() — Digital Dreamscape [4 scenes]")
    print("=" * 68)
    print(f"  Skill   : {SKILL.name}  (id={SKILL.id})")
    print(f"  Quality : {', '.join(SKILL.quality_boosters[:4])}")
    print(f"  Story   : Woman discovers her reality is a simulation")
    print(f"  Arc     : HOOK → BUILD → CLIMAX → RESOLUTION")
    print(f"  Model   : Wan2.2 T2V 14B + LightX2V 4-step LoRA")
    print(f"  Size    : {WIDTH}x{HEIGHT}   {FRAMES} frames  {FPS} fps (~5s/scene)")
    print(f"  Steps   : 4 (2 high-noise → 2 low-noise)   CFG: 1.0")
    print(f"  ETA     : ~108s/scene on RTX 4090D  (~7 min total)")
    print(f"  Server  : http://{host}:{port}")
    print("=" * 68)
    print()

    # Show what the skill injected
    print(f"  [SKILL BOOSTERS INJECTED]")
    print(f"    Quality : {', '.join(SKILL.quality_boosters[:6])}")
    print(f"    Style   : {', '.join(SKILL.style_tags[:3])}")
    print(f"    Negative: {', '.join(SKILL.negative_tags[:4])}, ...")
    print()

    try:
        q_before = await client.get_queue_status()
        r = len(q_before.get("queue_running", []))
        p = len(q_before.get("queue_pending", []))
        print(f"  ComfyUI online — queue: {r} running, {p} pending")
        print()
    except Exception as e:
        print(f"  [ERROR] Cannot reach ComfyUI at {host}:{port}: {e}")
        return

    job_ids = {}
    for scene in SCENES:
        try:
            wf  = fill_workflow(scene)
            pid = await client.queue_prompt(wf)
            job_ids[scene["scene_number"]] = pid
            print(f"  [{scene['act']:<10}] Scene {scene['scene_number']} — {scene['description']}")
            print(f"              job_id : {pid}")
            print()
        except Exception as e:
            print(f"  [FAILED] Scene {scene['scene_number']} — {e}")
        await asyncio.sleep(0.4)

    q = await client.get_queue_status()
    print("=" * 68)
    print(f"  Queued {len(job_ids)}/4  |  {len(q.get('queue_running',[]))} running  "
          f"{len(q.get('queue_pending',[]))} pending")
    print("=" * 68)
    print(f"\n  Monitor : http://{host}:{port}")
    print(f"  Output  : ComfyUI/output/  (prefix: dreamscape_s*)\n")
    for sn, pid in job_ids.items():
        print(f"    Scene {sn} [{_ACT_LABELS[sn-1]:<10}]  →  {pid}")


asyncio.run(main())
