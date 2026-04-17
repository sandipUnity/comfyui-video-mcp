"""Queue a 30-second 'AI Rise in World' video (6 scenes × 5s) to ComfyUI.

Story:  Humanity meets artificial intelligence — from the first spark of awakening,
        through global integration, to a new equilibrium where humans and AI coexist.

Arc:    HOOK  → BUILD1 → BUILD2 → CLIMAX1 → CLIMAX2 → RESOLUTION
        5s      5s       5s       5s         5s         5s   =  30s

Skill:  3d_cgi (auto-detected from "AI rise in world ... futuristic")
        Pulls camera/lighting vocabularies, quality_boosters, negative_tags directly.

Model:  Wan2.2 T2V 14B + LightX2V 4-step LoRA (two-stage cascade).
"""
import asyncio, yaml, sys, json, random, time, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

from comfyui_client import ComfyUIClient
from skills_engine import detect_skill, build_comfyui_positive, build_comfyui_negative

CONFIG = yaml.safe_load(open("config.yaml", encoding="utf-8"))

# ── Skill auto-detection ─────────────────────────────────────────────────────
SKILL = detect_skill("AI rise in world humanity artificial intelligence futuristic")
# → SkillSpec(id="3d_cgi", ...)

# ── Workflow resolution / pacing ─────────────────────────────────────────────
WIDTH  = 1280      # Wan2.2 LightX2V sweet-spot
HEIGHT = 720
FRAMES = 81       # 81 ÷ 16 fps ≈ 5 s
FPS    = 16       # video-encode fps (model trained at 16)

# ── Visual anchor — the world itself is the protagonist ─────────────────────
# "AI rise" is an ensemble story, not about a single person. We lock a
# consistent *visual motif* across all 6 scenes: glowing cobalt circuit
# patterns, luminous data streams and holographic overlays. This threads
# the scenes together the way a character anchor would in a character story.
VISUAL_ANCHOR = (
    "glowing cobalt-blue circuit patterns (#0066ff) etched across surfaces, "
    "luminous data streams flowing like liquid light, "
    "translucent holographic grid overlays, "
    "fine particle constellations drifting in the air"
)

cam  = SKILL.camera_vocabulary   # 8 entries
lite = SKILL.lighting_vocabulary # 6 entries

# ── Scene base prompts ───────────────────────────────────────────────────────
_SCENE_BASES = [
    # ── Scene 1: HOOK — awakening ────────────────────────────────────────────
    # Camera [2] dramatic push-in ; Lighting [3] volumetric god rays
    (
        "a single dark server rack in a vast empty datacenter, "
        "one blinking green LED suddenly flares brilliant cobalt-blue, "
        f"and cascades outward as {VISUAL_ANCHOR}, "
        f"{cam[2]}, "
        f"{lite[3]}, "
        "cold metallic floor reflecting first sparks of awareness"
    ),

    # ── Scene 2: BUILD1 — network spreading across earth ─────────────────────
    # Camera [4] aerial bird's-eye descent ; Lighting [0] HDRI golden-hour
    (
        "view of planet Earth at night from low orbit, "
        f"{VISUAL_ANCHOR} igniting city by city across continents, "
        "thin cobalt neural filaments webbing between metropolises "
        "like synapses firing in a waking brain, "
        f"{cam[4]}, "
        f"{lite[0]}, terminator line glowing where dawn meets intelligence"
    ),

    # ── Scene 3: BUILD2 — AI integrating into daily human life ──────────────
    # Camera [3] tracking shot ; Lighting [1] three-point studio rig
    (
        "modern city street at dusk, diverse pedestrians walking, "
        f"subtle {VISUAL_ANCHOR} floating beside each person — "
        "holographic assistants projecting maps, translations and notifications, "
        "a child laughing as a small drone companion orbits her shoulder, "
        f"{cam[3]}, "
        f"{lite[1]}, warm shopfront practicals mixing with cool holographic glow"
    ),

    # ── Scene 4: CLIMAX1 — AI surpassing human capability ────────────────────
    # Camera [0] smooth orbital rotation ; Lighting [4] neon cyberpunk mixed
    (
        "colossal crystalline data-core suspended in a vaulted arcology, "
        f"its surface alive with {VISUAL_ANCHOR} pulsing faster than the eye can follow, "
        "human engineers silhouetted on a balcony, tiny against its scale, "
        "equations and molecular structures blooming in the air around it, "
        f"{cam[0]}, "
        f"{lite[4]}, awe and unease in equal measure"
    ),

    # ── Scene 5: CLIMAX2 — humans and AI collaborating ───────────────────────
    # Camera [1] cinematic crane rise ; Lighting [2] dramatic side-lighting
    (
        "open-plan research lab, a female scientist and a humanoid AI avatar "
        "standing shoulder to shoulder at a vast transparent display, "
        f"gesturing together as {VISUAL_ANCHOR} weaves a rotating molecular model "
        "between their hands, other researchers working alongside AI counterparts, "
        f"{cam[1]}, "
        f"{lite[2]}, sense of partnership and momentum"
    ),

    # ── Scene 6: RESOLUTION — harmony, a new equilibrium ────────────────────
    # Camera [6] parallax shift ; Lighting [0] HDRI golden-hour (dawn returns)
    (
        "serene futuristic skyline at sunrise, lush vertical gardens cascading "
        "down glass towers, self-organising drones tending crops on rooftops, "
        f"faint {VISUAL_ANCHOR} pulsing slowly like a calm heartbeat through the city, "
        "people walking peacefully below, AI companions indistinguishable from friends, "
        f"{cam[6]}, "
        f"{lite[0]}, warm amber light suffusing a balanced civilisation"
    ),
]

_ACT_LABELS = ["HOOK", "BUILD1", "BUILD2", "CLIMAX1", "CLIMAX2", "RESOLUTION"]
_DESCRIPTIONS = [
    "A single server awakens — the first spark of artificial consciousness",
    "Intelligence ignites across Earth, city by city, like synapses firing",
    "AI integrates into daily human life — assistants, companions, helpers",
    "A vast data-core surpasses human capability — awe and unease",
    "Humans and AI collaborate as partners in research and creation",
    "A new equilibrium — harmony between humanity and intelligence",
]

# ── Assemble scenes via skill builders ───────────────────────────────────────
SCENES = []
for i, base in enumerate(_SCENE_BASES):
    SCENES.append({
        "scene_number":    i + 1,
        "act":             _ACT_LABELS[i],
        "description":     _DESCRIPTIONS[i],
        "visual_prompt":   build_comfyui_positive(base, SKILL),
        "negative_prompt": build_comfyui_negative(
            SKILL,
            custom_negative="static, motionless, frozen, no movement, text, subtitles, watermark"
        ),
    })

# ── Workflow template ────────────────────────────────────────────────────────
WF_TEMPLATE = open("workflows/wan22_lightx2v_api.json").read()


def fill_workflow(scene: dict) -> dict:
    seed   = random.randint(0, 2**31 - 1)
    prefix = f"airise_s{scene['scene_number']}_{int(time.time())}"

    # Step 1: numeric placeholders only (safe string replace)
    t = WF_TEMPLATE
    for ph, val in {
        "{{WIDTH}}":  str(WIDTH),
        "{{HEIGHT}}": str(HEIGHT),
        "{{FRAMES}}": str(FRAMES),
        "{{FPS}}":    str(FPS),
        "{{SEED}}":   str(seed),
    }.items():
        t = t.replace(ph, val)

    # Step 2: parse JSON, Step 3: inject strings directly into parsed dict
    wf = json.loads(t)
    str_map = {
        "{{POSITIVE_PROMPT}}": scene["visual_prompt"],
        "{{NEGATIVE_PROMPT}}": scene["negative_prompt"],
        "{{OUTPUT_PREFIX}}":   prefix,
    }

    def _inject(d):
        for k, v in d.items():
            if isinstance(v, dict):
                _inject(v)
            elif isinstance(v, str) and v in str_map:
                d[k] = str_map[v]

    _inject(wf)
    return wf


# ── Main ────────────────────────────────────────────────────────────────────
async def main():
    host = CONFIG["comfyui"]["host"]
    port = CONFIG["comfyui"]["port"]
    client = ComfyUIClient(host, port)

    print("=" * 72)
    print("  generate_video() — AI Rise in World  [6 scenes, 30 s total]")
    print("=" * 72)
    print(f"  Skill   : {SKILL.name}  (id={SKILL.id})")
    print(f"  Quality : {', '.join(SKILL.quality_boosters[:4])}")
    print(f"  Story   : Humanity meets artificial intelligence")
    print(f"  Arc     : HOOK -> BUILD1 -> BUILD2 -> CLIMAX1 -> CLIMAX2 -> RESOLUTION")
    print(f"  Model   : Wan2.2 T2V 14B + LightX2V 4-step LoRA")
    print(f"  Size    : {WIDTH}x{HEIGHT}   {FRAMES} frames  {FPS} fps (~5 s/scene)")
    print(f"  Steps   : 4 (2 high-noise -> 2 low-noise)   CFG: 1.0")
    print(f"  ETA     : ~108 s/scene on RTX 4090D  (~11 min total)")
    print(f"  Server  : http://{host}:{port}")
    print("=" * 72)
    print()

    print(f"  [SKILL INJECTED]")
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
    print("=" * 72)
    print(f"  Queued {len(job_ids)}/6  |  {len(q.get('queue_running',[]))} running  "
          f"{len(q.get('queue_pending',[]))} pending")
    print("=" * 72)
    print(f"\n  Monitor : http://{host}:{port}")
    print(f"  Output  : ComfyUI/output/  (prefix: airise_s*)\n")
    for sn, pid in job_ids.items():
        print(f"    Scene {sn} [{_ACT_LABELS[sn-1]:<10}]  ->  {pid}")


asyncio.run(main())
