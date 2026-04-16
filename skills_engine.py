"""
Seedance 2.0 Skill Engine — adapted for ComfyUI prompt generation.
Embeds the cinematic expertise from higgsfield-seedance2-jineng into
structured prompt builders that produce precise, professional ComfyUI prompts.

Principle: specificity over vagueness.
  ❌ "warm lighting"     → ✅ "3000K key light at 45° camera-left, 100% intensity"
  ❌ "fast movement"     → ✅ "3 ft/s dolly forward tracking subject"
  ❌ "they fight"        → ✅ "attacker launches spinning heel kick, defender flows
                               backward in three precise steps raising forearms to parry"
"""

from dataclasses import dataclass, field
from typing import Optional
import re


# ── Skill Definitions ─────────────────────────────────────────────────────────

@dataclass
class SkillSpec:
    id: str
    name: str
    description: str
    keywords: list[str]                 # detect from user notes
    quality_boosters: list[str]         # ComfyUI positive suffix tags
    negative_tags: list[str]            # ComfyUI negative prompt additions
    camera_vocabulary: list[str]        # precise camera moves
    lighting_vocabulary: list[str]      # Kelvin-precise lighting setups
    hook_patterns: list[str]            # 2-second opening hooks
    style_tags: list[str]               # style identifier tags
    technical_specs: dict               # fps, resolution defaults
    prompt_template: str                # system prompt for LLM
    comfyui_notes: str                  # ComfyUI-specific guidance


SKILLS: dict[str, SkillSpec] = {

    # ── 01 CINEMATIC ──────────────────────────────────────────────────────────
    "cinematic": SkillSpec(
        id="cinematic",
        name="Cinematic Film-Style",
        description="Live-action film with pro cinematography, precise lighting, dramatic camera work",
        keywords=["film", "cinematic", "movie", "dramatic", "live action", "realistic", "documentary",
                  "noir", "thriller", "drama", "portrait", "landscape", "scene"],
        quality_boosters=[
            "masterpiece", "best quality", "8k uhd", "film grain", "cinematic color grading",
            "anamorphic lens flare", "shallow depth of field", "RAW photo", "DSLR", "35mm film",
            "professional cinematography", "golden ratio composition", "bokeh"
        ],
        negative_tags=[
            "cartoon", "anime", "3d render", "cgi", "painting", "illustration",
            "low quality", "blurry", "overexposed", "flat lighting", "stock photo", "watermark"
        ],
        camera_vocabulary=[
            "3 ft/s dolly forward tracking subject",
            "slow 0.5 ft/s push-in on face",
            "sweeping crane shot rising 15 ft in 4 seconds",
            "handheld follow-cam at shoulder height",
            "whip-pan left to right in 0.3 seconds",
            "orbital camera circling subject at 2 ft/s",
            "low-angle Dutch tilt 15°",
            "rack focus from foreground to background in 1.5 seconds",
            "extreme close-up macro lens 50mm f/1.4",
            "wide establishing shot 24mm f/11"
        ],
        lighting_vocabulary=[
            "3000K key light at 45° camera-left, 100% intensity, 33% fill, 60% back rim",
            "golden hour sunlight 2700K from camera-right, natural shadows",
            "harsh overhead noon sun 5500K creating dramatic under-eye shadows",
            "single practical bulb 2400K creating intimate amber atmosphere",
            "neon sign bounce 4200K mixed with tungsten 3200K",
            "overcast soft box 6500K, shadowless, clinical",
            "motivated window light 5600K with dust particles visible",
            "three-point studio setup: 5600K key 45°, 3200K fill 135°, 6000K back 270°"
        ],
        hook_patterns=[
            "extreme close-up of eyes snapping open in first frame",
            "black screen to blinding white flash in 0.5s then subject reveal",
            "reverse-motion sequence playing backward then snapping forward",
            "unexpected scale contrast: tiny object filling full frame then pulled back to reveal enormous environment",
            "subject emerging from total darkness into single spotlight"
        ],
        style_tags=["cinematic", "photorealistic", "film photography", "dramatic lighting"],
        technical_specs={"fps": 24, "width": 768, "height": 512, "steps": 25, "cfg": 7.0},
        prompt_template="""You are a professional cinematographer writing ComfyUI prompts.
Use precise technical language:
- Camera moves: specify velocity in ft/s, exact angle in degrees
- Lighting: always include Kelvin temperature, intensity ratios (key/fill/back)
- Depth of field: f-number (f/1.4 for shallow, f/11 for deep)
- Film language: dolly, crane, rack focus, Dutch tilt, lens choice
- Composition: rule of thirds, leading lines, frame within frame
Never use vague terms like "beautiful" or "nice". Be technical and specific.""",
        comfyui_notes="Works best with realistic checkpoints (Realistic Vision, Deliberate, epiCRealism). Add 'film grain' for texture.",
    ),

    # ── 02 3D CGI ─────────────────────────────────────────────────────────────
    "3d_cgi": SkillSpec(
        id="3d_cgi",
        name="3D CGI Render",
        description="Photorealistic or stylized 3D computer-generated imagery with PBR materials",
        keywords=["3d", "cgi", "render", "blender", "octane", "unreal", "product", "visualization",
                  "abstract", "geometric", "architecture", "interior", "futuristic", "sci-fi"],
        quality_boosters=[
            "masterpiece", "octane render", "unreal engine 5", "photorealistic 3D render",
            "PBR materials", "subsurface scattering", "global illumination", "ray tracing",
            "4K texture detail", "HDRI lighting", "volumetric atmosphere", "motion blur"
        ],
        negative_tags=[
            "hand-drawn", "2d", "flat", "cartoon outline", "sketch", "watercolor",
            "low poly pixelated", "texture artifacts", "z-fighting", "clipping"
        ],
        camera_vocabulary=[
            "smooth orbital rotation at 45°/s around subject",
            "cinematic crane shot rising from ground to 20ft elevation",
            "dramatic push-in from 15ft to 2ft in 3 seconds",
            "tracking shot following subject at constant 3ft distance",
            "aerial bird's-eye descent from 100ft to 30ft",
            "first-person fly-through at 8 ft/s",
            "parallax shift left at 0.5 ft/s for depth illusion",
            "360° turntable rotation at 30°/s, camera fixed at 15° elevation"
        ],
        lighting_vocabulary=[
            "HDRI golden-hour environment map 2700K, sun disc at 15° elevation",
            "three-point studio rig: 5600K key area light 2m, 3200K fill 4m, 6000K rim 1m",
            "dramatic side-lighting 4000K with hard shadow at 85° angle",
            "volumetric god rays 5500K penetrating through fog at 30° angle",
            "neon cyberpunk mixed: cyan 16000K left, magenta 7000K right",
            "product turntable: front-key 5600K soft box, fill 45° opposite, rim 180° back"
        ],
        hook_patterns=[
            "camera fly-through solid surface materializing around it",
            "particles assembling from dust cloud into final 3D form",
            "extreme macro reveal pulling back to full product/environment",
            "photorealistic object in surreal impossible environment",
            "morph from wireframe mesh to full photorealistic render"
        ],
        style_tags=["3D render", "CGI", "photorealistic", "Octane", "Unreal Engine"],
        technical_specs={"fps": 30, "width": 768, "height": 432, "steps": 30, "cfg": 7.5},
        prompt_template="""You are a 3D rendering artist writing ComfyUI prompts.
Use render engine terminology:
- Materials: PBR (metalness %, roughness %), SSS depth in mm, IOR values
- Lighting: HDRI environment maps, area lights with size/power, directional angle
- Camera: focal length in mm, aperture in f-stops, motion blur shutter angle
- Effects: volumetric scatter density, particle count, motion blur intensity
Always specify the "render engine look": Octane, Cycles, V-Ray, Unreal Engine 5.
Include specific material descriptions: "brushed aluminum 0.8 roughness 0.95 metalness".""",
        comfyui_notes="Use DreamShaper or Juggernaut for CGI look. AnimateDiff smooth motion module recommended.",
    ),

    # ── 03 CARTOON ────────────────────────────────────────────────────────────
    "cartoon": SkillSpec(
        id="cartoon",
        name="Cartoon & Animation",
        description="Western cartoon animation styles from classic Disney to modern CGI-animated",
        keywords=["cartoon", "animation", "animated", "disney", "pixar", "dreamworks",
                  "kids", "fun", "colorful", "character", "whimsical", "cute", "rubber hose"],
        quality_boosters=[
            "cartoon style", "clean line art", "bold outlines", "flat cel shading",
            "vibrant saturated colors", "squash and stretch", "smooth animation",
            "expressive character", "classic animation principles", "masterpiece quality"
        ],
        negative_tags=[
            "realistic", "photographic", "3d render", "anime", "sketch", "rough",
            "dark themes", "gore", "hyperrealistic textures"
        ],
        camera_vocabulary=[
            "smash-zoom punch-in 5x in 0.2 seconds",
            "smooth pan following bouncing character at 2ft/s",
            "comedy timing: freeze-frame on shocked expression 0.5 seconds",
            "anticipation pull-back before speed rush forward",
            "circular wipe transition at 1 second",
            "static wide shot for comedic punchline",
            "low-angle hero shot for dramatic character moment"
        ],
        lighting_vocabulary=[
            "flat even cel-shading illumination, no directional shadows",
            "stylized single-shadow drop shadow at 45°, 50% opacity",
            "bright saturated rim light outlining character shape",
            "warm 3000K toon-shaded key with blue complementary fill",
            "comic book speedline burst from center outward"
        ],
        hook_patterns=[
            "character smashes through frame border in 0.3 seconds",
            "rubber hose limb stretching to 5x normal length then snapping back",
            "color explosion burst from black-and-white to full saturation",
            "character running so fast they leave frame leaving trail of dust",
            "fourth-wall break: character notices and turns to camera"
        ],
        style_tags=["cartoon", "animation", "cel shading", "flat color", "clean lineart"],
        technical_specs={"fps": 24, "width": 768, "height": 432, "steps": 20, "cfg": 8.0},
        prompt_template="""You are a professional animation director writing ComfyUI prompts for cartoon content.
Apply the 12 classic animation principles through prompt language:
- Squash and stretch: describe deformation percentages
- Anticipation: describe pre-action tells
- Exaggeration: describe physical over-expression
- Timing: specify frame counts for comedic beats
Use style anchors: "Disney Renaissance style", "Looney Tunes physics", "Pixar 3D look", "retro rubber hose 1930s".
Colors should be bold and described by hex or Pantone when possible.""",
        comfyui_notes="Use cartoon/animation LoRAs. CartoonStyle, ToonCrafter workflows work well.",
    ),

    # ── 05 FIGHT SCENES ───────────────────────────────────────────────────────
    "fight_scenes": SkillSpec(
        id="fight_scenes",
        name="Fight Scene & Action",
        description="Choreographed combat and high-intensity action sequences",
        keywords=["fight", "action", "combat", "battle", "martial arts", "sword", "punch",
                  "kick", "warrior", "ninja", "soldier", "explosion", "chase", "clash"],
        quality_boosters=[
            "masterpiece", "dynamic action shot", "motion blur on moving parts",
            "impact frames", "speed lines", "dust particles", "spark effects",
            "cinematic action photography", "freeze frame at impact", "dramatic lighting"
        ],
        negative_tags=[
            "static", "still", "calm", "peaceful", "blurry faces", "out of focus subject",
            "awkward poses", "incorrect anatomy", "floating limbs"
        ],
        camera_vocabulary=[
            "whip-pan tracking punch impact at 0.1 second shutter",
            "camera pushes forward at 5 ft/s on aggressive strike",
            "camera pulls back 3ft as defender staggers",
            "low Dutch tilt 20° during intense exchange",
            "360° orbital at 6 ft/s during spinning kick",
            "handheld shake 2-pixel jitter during grapple",
            "extreme slow-motion 240fps on point-of-impact freeze",
            "tracking shot from behind attacker running at 8 ft/s"
        ],
        lighting_vocabulary=[
            "harsh side-key 6000K at 90°, deep shadow opposite, strong rim 180°",
            "red-orange fire light 1800K flickering ±20% on left",
            "cool blue moonlight 8000K top-down 70°, warm torch fill right",
            "dust-diffused sunlight 5500K creating volumetric shafts",
            "neon-lit warehouse 4000K cyan overhead with practical floor lighting"
        ],
        hook_patterns=[
            "clash of weapons with spark shower in first 0.3 seconds, sound of steel impact",
            "mid-spin freeze-frame: combatant frozen in 270° spinning heel kick",
            "camera stares down fist rushing straight toward lens",
            "dust cloud explosion reveals two fighters in combat stance",
            "impact shockwave ripples fabric, dirt, and hair outward from strike point"
        ],
        style_tags=["action", "combat", "dynamic", "motion blur", "impact frames"],
        technical_specs={"fps": 24, "width": 768, "height": 432, "steps": 25, "cfg": 7.5},
        prompt_template="""You are an action choreographer writing ComfyUI prompts.
CRITICAL: Be choreographically specific, never vague.
  ✓ "attacker launches right spinning heel kick, foot connects at jaw height,
     defender flows backward in 3 precise steps raising crossed forearms to parry"
  ✗ "they fight intensely"

Always specify:
- Attacker/defender positioning (compass directions, distances in feet)
- Momentum direction and speed (advancing/retreating, ft/s)
- Environmental destruction that occurs (dust, debris, sparks, fabric ripple)
- Camera position relative to combatants during each beat""",
        comfyui_notes="Use AnimateDiff with action LoRAs. High CFG 7.5-8.5 for coherent motion.",
    ),

    # ── 08 ANIME ACTION ───────────────────────────────────────────────────────
    "anime": SkillSpec(
        id="anime",
        name="Anime & Japanese Animation",
        description="Japanese animation aesthetic with cel shading, speed lines, and genre-specific visual language",
        keywords=["anime", "manga", "japanese", "shonen", "seinen", "magical girl", "mecha",
                  "kawaii", "otaku", "sakura", "ninja", "samurai", "slice of life", "cyberpunk anime"],
        quality_boosters=[
            "anime style", "masterpiece", "best quality", "detailed anime art",
            "cel shading", "anime shading", "clean line art", "key visual quality",
            "official art style", "vibrant colors", "expressive eyes"
        ],
        negative_tags=[
            "western cartoon", "3d render", "realistic", "photographic", "sketch",
            "rough lines", "inconsistent style", "western comic"
        ],
        camera_vocabulary=[
            "dramatic pull-back from extreme close-up to full scene reveal",
            "impact frame freeze: single key frame held 0.3s with speed lines radiating outward",
            "slow-motion 0.2x speed during emotional climax",
            "dynamic Dutch tilt 25° during intense battle",
            "smear frame transition between action positions",
            "static establishing shot with single moving element (hair, leaves)",
            "spiral camera orbit at power-up moment",
            "eye-level intimate shot for emotional dialogue"
        ],
        lighting_vocabulary=[
            "shonen style: 150% saturated warm amber sunlight from 45° above",
            "seinen style: desaturated cool 8000K with 60% contrast harsh shadows",
            "magical girl: soft pink 3200K omnidirectional sparkle glow",
            "mecha: cold industrial 6500K with neon cyan/magenta accent lights",
            "slice of life: golden hour 2800K dappled through window, dust motes",
            "cyberpunk anime: dark base with 100% neon cyan 16000K and magenta 7000K",
            "dramatic power-up: internal white glow expanding outward, 6 second ramp"
        ],
        hook_patterns=[
            "speed lines converging to center reveal then explode outward",
            "impact frame: frozen moment at point of contact with speed lines + motion blur edges",
            "sudden dramatic zoom from wide to extreme close-up on eyes in 0.4 seconds",
            "black screen → single held musical note → full saturated scene explosion",
            "anime opening sequence style: character pose with name overlay, dramatic wind"
        ],
        style_tags=["anime", "manga style", "cel shading", "Japanese animation", "2D animation"],
        technical_specs={"fps": 24, "width": 768, "height": 432, "steps": 22, "cfg": 8.0},
        prompt_template="""You are an anime director writing ComfyUI prompts.
Apply genre-specific visual language:
- SHONEN (action): explosive poses, 130%+ saturation, speed lines, impact frames,
  power-up glows, sweat/blood drops, torn clothing indicating damage level
- SEINEN (mature): desaturated 60-70%, cool tones, introspective close-ups, subtle expressions
- MAGICAL GIRL: pastels, sparkle effects, transformation sequences, flowing hair physics
- MECHA: mechanical joint detail, exhaust heat distortion, neon HUD overlays on cockpit
- SLICE OF LIFE: warm golden hour, bokeh background flowers, subtle character micro-expressions

Never describe anime as "big eyes and colorful" — use genre-precise visual vocabulary.
Include specific animation effects: speed lines, smear frames, impact frames, aura pulses.""",
        comfyui_notes="Use anime-specific checkpoints (Anything V5, Counterfeit, ReV Animated). AnimateDiff ToonCrafter for smooth motion.",
    ),

    # ── 06 MOTION DESIGN AD ───────────────────────────────────────────────────
    "motion_design": SkillSpec(
        id="motion_design",
        name="Motion Design & Tech Ad",
        description="Modern SaaS/tech product advertisement with UI animations and data visualizations",
        keywords=["app", "software", "saas", "tech", "startup", "product", "ui", "ux", "interface",
                  "dashboard", "data", "analytics", "code", "digital", "advertisement", "ad"],
        quality_boosters=[
            "professional motion graphics", "clean modern design", "smooth animation",
            "premium brand aesthetic", "crisp UI elements", "4K product mockup",
            "glassmorphism", "neumorphism", "minimalist design", "high-end commercial"
        ],
        negative_tags=[
            "stock footage", "amateur", "generic", "cluttered", "dated design",
            "pixelated text", "inconsistent brand colors"
        ],
        camera_vocabulary=[
            "device float-in from below at 0.5 ft/s decelerating to rest",
            "smooth screen zoom into UI at 2x per second",
            "multi-device carousel rotating at 30°/s",
            "parallax scroll: background 0.3x speed vs foreground 1x",
            "glowing circle spotlight expanding outward from product",
            "cinematic push-in to hero feature in 2 seconds",
            "split-screen: before/after product adoption side by side"
        ],
        lighting_vocabulary=[
            "dark premium: navy #0a0e27 background, bright electric blue #0066ff accent glow",
            "neon cyber: pure black background, cyan #00ffff + pink #ff00ff accent strips",
            "clean white: 6500K soft diffuse light, 95% white background, subtle product shadow",
            "gradient mesh: purple-to-blue background 4000K ambient with product rim light",
            "floating product: dark gradient, 360° edge glow at 5600K, zero-shadow float"
        ],
        hook_patterns=[
            "data visualization explodes from single point to full infographic in 0.5s",
            "UI elements materialize from particles assembling on dark background",
            "device appearing from fog/smoke revealing glowing screen",
            "metric counter rapidly incrementing from 0 to impressive number",
            "code lines appearing and instantly morphing into finished product"
        ],
        style_tags=["motion graphics", "UI design", "tech commercial", "product showcase", "modern"],
        technical_specs={"fps": 60, "width": 1280, "height": 720, "steps": 25, "cfg": 7.0},
        prompt_template="""You are a motion designer writing ComfyUI prompts for tech/SaaS advertisements.
Device framing rule: product UI must fill 40-60% of frame for legibility.
Visual styles to specify: dark premium, neon cyber, clean white, gradient mesh.
Animation principles: ease-in-out curves, 50ms audio sync, particle materialization.
Include: brand color hex codes, font weight descriptors, UI element sizes.
Platform specs: 60fps for fluid animation, crisp edges on all text elements.
CTA placement: bottom third, high contrast, 5-8 word maximum.""",
        comfyui_notes="Use DreamShaper or SDXL for clean commercial look. ComfyUI-Impact for particle effects.",
    ),

    # ── 07 ECOMMERCE AD ───────────────────────────────────────────────────────
    "ecommerce": SkillSpec(
        id="ecommerce",
        name="E-Commerce Product Ad",
        description="Product advertising for online retail with lifestyle integration and purchase intent",
        keywords=["product", "shop", "buy", "store", "retail", "brand", "unboxing",
                  "fashion", "beauty", "cosmetic", "jewelry", "food product", "packaging", "commercial"],
        quality_boosters=[
            "commercial photography quality", "product hero shot", "lifestyle photography",
            "high-end retail aesthetic", "crisp product details", "professional lighting",
            "brand campaign quality", "magazine editorial", "clean background"
        ],
        negative_tags=[
            "amateur photography", "harsh flash", "cluttered background", "incorrect product color",
            "out-of-focus product", "stock photo feel", "inconsistent lighting"
        ],
        camera_vocabulary=[
            "smooth 360° product turntable at 30°/s, camera at 15° elevation",
            "dramatic slow reveal from packaging to product in 2 seconds",
            "lifestyle tracking shot: product in natural use context",
            "macro close-up texture shot: 1:1 magnification on material",
            "before/after split screen with simultaneous zoom",
            "unboxing overhead flat-lay with elements sliding into frame",
            "hand model interaction: reaching for product at 0.5 ft/s"
        ],
        lighting_vocabulary=[
            "white seamless product: 5600K overhead key 80%, bounce fill below 40%, rim 60%",
            "lifestyle beauty: 3200K warm key, golden reflector fill, silver rim highlight",
            "jewelry macro: 6000K directional hard light for sparkle, dark graduated background",
            "food: 2800K warm side-key 45°, natural reflector opposite, steam from top",
            "fashion: window light 5500K from camera-left, white foam-board fill right"
        ],
        hook_patterns=[
            "product emerges dramatically from silk fabric uncovering in slow motion",
            "extreme close-up texture detail pulling back to full product reveal",
            "before/after transformation showing product results in 1.5 seconds",
            "unboxing reveal with layered tissue paper floating away",
            "product hero shot: single item in perfect lighting against gradient"
        ],
        style_tags=["commercial photography", "product shot", "lifestyle", "retail", "brand"],
        technical_specs={"fps": 30, "width": 1080, "height": 1080, "steps": 25, "cfg": 7.0},
        prompt_template="""You are a commercial product photographer writing ComfyUI prompts.
Product visibility rule: subject must be sharp, well-lit, filling 50-70% of frame.
Include: product material texture description, color accuracy notes, reflection/specularity.
Lifestyle context: describe environment that enhances product's perceived value.
Emotional arc: problem → transformation → desire → purchase intent.
Text overlay guidance: 5-8 word benefits, high contrast placement, brand colors.""",
        comfyui_notes="Use Realistic Vision or epiCRealism for product shots. IPAdapter for product consistency across frames.",
    ),

    # ── 11 SOCIAL HOOK ────────────────────────────────────────────────────────
    "social_hook": SkillSpec(
        id="social_hook",
        name="Viral Social Hook",
        description="Scroll-stopping social media content engineered for maximum retention and virality",
        keywords=["social media", "tiktok", "instagram", "reels", "viral", "trending", "hook",
                  "short video", "content", "scroll", "attention", "engaging", "meme"],
        quality_boosters=[
            "vertical video 9:16", "high contrast colors", "bold on-screen text",
            "trending aesthetic", "satisfying motion", "eye-catching", "scroll-stopping",
            "authentic feel", "high energy", "dynamic cuts"
        ],
        negative_tags=[
            "boring opening", "slow start", "static shot", "no movement", "muted colors",
            "out-of-focus", "shaky unintentional", "silent without purpose"
        ],
        camera_vocabulary=[
            "sudden unexpected 2x smash-zoom in 0.2 seconds — pattern interrupt",
            "handheld natural movement 1-pixel jitter for authenticity",
            "rapid cuts every 0.5-1 second maintaining visual momentum",
            "direct eye-contact to lens creating personal connection",
            "split-screen reveal: left before / right after simultaneous",
            "360° spin with speed blur at 3 revolutions/second",
            "negative space pause then sudden fill — curiosity gap technique"
        ],
        lighting_vocabulary=[
            "natural window 5600K for authentic UGC aesthetic",
            "ring light 5500K front-facing for beauty/talking head",
            "golden hour 2700K backlit for aspirational lifestyle",
            "neon accent 7000K in background for trending aesthetic",
            "high-key white overexposed by +1 stop for surreal pop effect"
        ],
        hook_patterns=[
            "0.0-0.3s: unexpected visual shock or sound surprise to stop scroll",
            "0.3-0.8s: incomplete information reveal creating curiosity gap",
            "0.8-1.5s: momentum confirmation showing hook is delivering",
            "1.5-2.0s: commitment moment — viewer decides to watch through",
            "impossible physics: object defying gravity in first frame",
            "direct address: character locks eyes with camera and speaks to viewer",
            "rapid satisfaction loop: problem shown → solved in 1 second"
        ],
        style_tags=["social media", "vertical video", "UGC", "trending", "high retention"],
        technical_specs={"fps": 30, "width": 576, "height": 1024, "steps": 20, "cfg": 7.0},
        prompt_template="""You are a viral content creator writing ComfyUI prompts for social media.
The 2-second hook framework is NON-NEGOTIABLE:
- 0.0-0.3s: visual or audio shock that stops the scroll
- 0.3-0.8s: incomplete reveal building curiosity
- 0.8-1.5s: momentum — confirm the hook delivers
- 1.5-2.0s: commitment moment

Sound = 50% of impact. Describe audio elements explicitly.
Vertical framing (9:16) is mandatory. Text overlay placement: top or bottom third.
Avoid: false promises, static openings, no audio, shaky unintentional camera.""",
        comfyui_notes="Set width=576, height=1024 for 9:16. AnimateDiff smooth + lora for trending aesthetics.",
    ),

    # ── 10 MUSIC VIDEO ────────────────────────────────────────────────────────
    "music_video": SkillSpec(
        id="music_video",
        name="Music Video",
        description="Beat-synchronized visual content with genre-specific aesthetics and rhythm-driven editing",
        keywords=["music", "song", "beat", "rhythm", "artist", "band", "concert", "performance",
                  "hip hop", "pop", "electronic", "edm", "lo-fi", "rock", "singer", "dance"],
        quality_boosters=[
            "music video quality", "beat-synced editing", "concert lighting", "stage production",
            "choreographed movement", "visual rhythm", "dynamic cuts on beat", "professional production"
        ],
        negative_tags=[
            "static shot", "off-beat timing", "mismatched mood", "inconsistent style",
            "low production value", "random motion unrelated to rhythm"
        ],
        camera_vocabulary=[
            "beat-cut transition: hard cut synchronized to snare hit",
            "hi-hat rapid cuts: 0.25s per shot on 16th notes",
            "bass drop: camera crashes forward 2ft/s on beat 1",
            "slow-motion 0.25x during pre-chorus tension build",
            "concert orbital 360° completing one revolution per bar",
            "push-in on artist face during emotional lyric",
            "pull-back reveal for chorus energy expansion"
        ],
        lighting_vocabulary=[
            "hip-hop: hard neon accents 4200K in urban environment, practical lights only",
            "pop: bright saturated RGB concert rig, synchronized strobes on beat",
            "electronic/EDM: pure dark with laser grid 16000K cyan cuts and particle rain",
            "lo-fi: warm 2800K golden, soft window light, vintage film grain, analog warmth",
            "rock: dramatic side-key 4000K, smoke machine backlit 6000K",
            "R&B: sensual warm 2600K, rim light 5600K, jewel-tone background"
        ],
        hook_patterns=[
            "bass drop visual: screen white flash then color explosion on beat 1",
            "hi-hat synced rapid-cut sequence: 8 shots in 1 second",
            "slow-motion pre-drop tension hold then explosive beat release",
            "artist emerging from fog/smoke into spotlight on first note",
            "waveform visualization morphing into performance environment"
        ],
        style_tags=["music video", "concert", "performance", "beat-synced", "visual rhythm"],
        technical_specs={"fps": 24, "width": 1280, "height": 720, "steps": 25, "cfg": 7.5},
        prompt_template="""You are a music video director writing ComfyUI prompts.
Genre-visual mapping — reference these specific aesthetics:
- Hip-Hop: urban environments (warehouses/parking lots), hard shadows, neon practical
- Pop: bright saturated choreography, fast cuts, synchronized performance
- EDM: neon purple/cyan abstract landscape, particles, laser grid, dark base
- Lo-Fi: warm desaturated, nostalgic grain, minimal movement, cozy intimacy
- Rock: gritty texture, dramatic side-lighting, smoke and stage haze
Always specify timing references: "at 0:15 when snare hits" or "on beat drop".
Describe beat synchronization explicitly in the prompt.""",
        comfyui_notes="AnimateDiff with temporal consistency for performance shots. Use ControlNet OpenPose for dancer consistency.",
    ),

    # ── 12 BRAND STORY ────────────────────────────────────────────────────────
    "brand_story": SkillSpec(
        id="brand_story",
        name="Brand Storytelling",
        description="Emotional brand narrative videos that build trust and communicate values",
        keywords=["brand", "story", "founder", "company", "startup", "mission", "values",
                  "inspiration", "journey", "authenticity", "documentary style", "corporate"],
        quality_boosters=[
            "documentary style", "authentic", "emotional", "cinematic brand film",
            "professional corporate video", "premium production", "story-driven",
            "natural light", "real environments", "candid moments"
        ],
        negative_tags=[
            "stock footage aesthetic", "overly polished fake", "generic corporate",
            "talking-head only", "static slides", "inconsistent style"
        ],
        camera_vocabulary=[
            "intimate handheld 1-pixel jitter following subject naturally",
            "fly-on-the-wall observational tracking at 1 ft/s",
            "reveal pull-back: tight on detail expanding to full context",
            "candid over-shoulder during authentic work moment",
            "slow push-in 0.3 ft/s during emotional speaking moment",
            "wide establishing showing real environment and people",
            "close-up on worn hands/tools communicating authentic work"
        ],
        lighting_vocabulary=[
            "golden hour natural 2700K window light for warmth and authenticity",
            "overcast outdoor 6000K soft diffuse — honest and unfiltered",
            "practical office lighting 4000K mixed with screen glow 7000K",
            "workshop/studio: natural skylights 5500K, tool-area practical pendants 3000K",
            "evening warm 2600K creating intimate community gathering feel"
        ],
        hook_patterns=[
            "hands at work in close-up: crafting, typing, creating — no faces yet",
            "provocative question as text overlay on quiet authentic scene",
            "single sensory detail that encapsulates brand values",
            "founder in natural habitat before they're aware of camera",
            "customer transformation: before state establishing shot"
        ],
        style_tags=["brand film", "documentary", "authentic", "emotional", "corporate story"],
        technical_specs={"fps": 24, "width": 1280, "height": 720, "steps": 22, "cfg": 7.0},
        prompt_template="""You are a brand filmmaker writing ComfyUI prompts.
Show don't tell principle is absolute:
  ✓ "worn leather tool belt with sawdust on the carpenter's hands"
  ✗ "hardworking craftsman"

Narrative structures: origin story, transformation, day-in-the-life, vision manifesto.
Authenticity signals: natural imperfect lighting, real environments, candid moments.
Emotional register options: inspirational, warm/intimate, rebellious, calm/trustworthy.
Avoid: polish that feels fake, voiceover description, generic stock aesthetics.""",
        comfyui_notes="Use Realistic Vision for authentic look. ControlNet reference for consistent characters.",
    ),

    # ── 13 FASHION LOOKBOOK ───────────────────────────────────────────────────
    "fashion": SkillSpec(
        id="fashion",
        name="Fashion Lookbook",
        description="High-fashion editorial and lookbook video content with model direction and styling",
        keywords=["fashion", "style", "clothing", "outfit", "model", "lookbook", "editorial",
                  "luxury", "streetwear", "designer", "runway", "clothing brand", "wardrobe"],
        quality_boosters=[
            "fashion editorial quality", "Vogue aesthetic", "high fashion",
            "professional model", "editorial lighting", "magazine quality",
            "luxury brand aesthetic", "designer clothing detail", "fashion photography"
        ],
        negative_tags=[
            "casual snapshot", "unflattering pose", "bad styling", "incorrect fit",
            "overexposed", "distracting background", "amateur model direction"
        ],
        camera_vocabulary=[
            "power walk tracking shot: camera at hip height moving backward 2 ft/s",
            "360° model rotation: camera orbits at 1 ft/s, subject rotates opposite",
            "fabric detail macro pull-back: 1:1 texture to full outfit reveal",
            "editorial overhead flat-lay: camera drops vertically 2ft in 1 second",
            "runway-style: long focal 200mm tracking walk from 30ft distance",
            "wind machine: shoot into fabric movement at 1/1000s freeze"
        ],
        lighting_vocabulary=[
            "golden hour 2700K backlit creating rim on fabric edges, 40% front fill",
            "studio strobe 5600K beauty dish 45° with large 4x6ft silver reflector",
            "urban neon 4000K mixed practical: signage providing colored accent fills",
            "natural overcast 6000K: clean color accuracy, zero harsh shadows",
            "dramatic fashion backlight 5600K 180°, model silhouette with details: 20% front fill"
        ],
        hook_patterns=[
            "dramatic outfit reveal: black cloth drops away revealing full look",
            "power walk entrance: model enters frame at full stride into hero position",
            "rapid outfit change: 3 looks in 1.5 seconds through cut/dissolve",
            "macro texture reveal: fabric thread detail expanding to full outfit",
            "editorial freeze: model freezes in dynamic pose, environment continues moving"
        ],
        style_tags=["fashion editorial", "lookbook", "Vogue", "high fashion", "model photography"],
        technical_specs={"fps": 24, "width": 1080, "height": 1350, "steps": 25, "cfg": 7.5},
        prompt_template="""You are a fashion director writing ComfyUI prompts for editorial content.
Outfit description formula: [garment type] in [material] [color] with [construction detail] [fit description].
  Example: "oversized blazer in wool-mohair blend deep navy with peak lapels, structured shoulders, draped fit"
Model direction: specify gait type, posture attitude, expression, hand placement.
  "purposeful heel-to-toe stride, chin parallel to floor, direct neutral gaze, hands at sides"
Fabric physics: describe how the material should move (drape, billow, cling, stiffen).
Location: describe how environment communicates brand positioning.""",
        comfyui_notes="Use DreamShaper or realistic SDXL. ControlNet pose for consistent model positioning.",
    ),

    # ── 14 FOOD & BEVERAGE ────────────────────────────────────────────────────
    "food": SkillSpec(
        id="food",
        name="Food & Beverage",
        description="Hunger-inducing food content with texture-first cinematography and sensory audio design",
        keywords=["food", "cooking", "recipe", "restaurant", "chef", "meal", "dish", "drink",
                  "coffee", "cocktail", "baking", "grill", "cuisine", "beverage", "dessert"],
        quality_boosters=[
            "food photography", "appetizing", "fresh ingredients", "steam wisps",
            "natural textures visible", "professional food styling", "restaurant quality",
            "warm inviting light", "crispy/juicy/glossy texture detail"
        ],
        negative_tags=[
            "unappetizing", "cold food appearance", "artificial colors", "flat lighting",
            "blurry texture", "messy plating", "cold white light on food"
        ],
        camera_vocabulary=[
            "overhead 90° flat-lay with fork scraping revealing texture in slow motion",
            "macro lens 1:1 on surface bubbling/sizzling/melting",
            "45° angle classic food shot: plate fills bottom third, background bokeh",
            "cheese pull extreme slow-motion 240fps tracking stretch",
            "steam follow: rack focus through rising steam to dish",
            "sauce cascade: overhead pour in 0.25x slow motion",
            "bite reveal: fork cuts through showing interior cross-section"
        ],
        lighting_vocabulary=[
            "warm 2800K key light camera-left 45°, white foam-board fill right, natural shadows",
            "backlit steam: 5500K hard back-key 180° makes steam glow, warm front fill 2800K",
            "3200K window light creating natural shadow and texture depth",
            "golden hour outdoor dining: 2700K ambient with 3200K practical candle fill",
            "restaurant: 2600K pendant practical overhead, 3000K accent strip, dark atmosphere"
        ],
        hook_patterns=[
            "money shot first: cheese pull or sauce cascade in opening 0.5 seconds",
            "sizzle sound + steam cloud + slow-reveal of cooking surface simultaneously",
            "knife slice cross-section reveal showing interior layers and textures",
            "pour in slow-motion: liquid hitting surface with crown splash effect",
            "from-above ingredient assembly: each element entering frame in sequence"
        ],
        style_tags=["food photography", "culinary", "appetizing", "restaurant", "food styling"],
        technical_specs={"fps": 24, "width": 1080, "height": 1080, "steps": 25, "cfg": 7.5},
        prompt_template="""You are a food cinematographer writing ComfyUI prompts.
Texture is taste: describe visible surface qualities that suggest flavor.
  "glistening caramelized crust with visible sugar crystallization, steam venting from soft interior"
Sound design is 50% of food content — describe audio explicitly:
  "sizzle of butter in pan, crisp crunch of crust breaking, bubbling reduction"
Lighting for food: warm 2700-3200K always, backlit steam creates appetite cues.
The money shot: identify 1 climactic moment (cheese stretch, sauce cascade, chocolate pour) and build the entire prompt around maximizing that 2-second sequence.""",
        comfyui_notes="Use photorealistic checkpoints. AnimateDiff for subtle steam/sizzle motion.",
    ),

    # ── 15 REAL ESTATE ────────────────────────────────────────────────────────
    "real_estate": SkillSpec(
        id="real_estate",
        name="Real Estate & Architecture",
        description="Property showcase with purposeful camera movement, strategic lighting, and aspirational atmosphere",
        keywords=["house", "home", "property", "real estate", "apartment", "interior", "exterior",
                  "architecture", "building", "luxury home", "commercial space", "office", "renovation"],
        quality_boosters=[
            "architectural photography quality", "real estate video", "luxury property showcase",
            "professional drone footage", "interior design photography", "golden hour exterior",
            "aspirational lifestyle", "spacious well-lit", "natural light filled"
        ],
        negative_tags=[
            "cluttered", "dark", "unflattering angle", "fish-eye distortion",
            "incorrect perspective", "empty cold feel", "overexposed windows"
        ],
        camera_vocabulary=[
            "steadicam walk-through at 1 ft/s, camera at 5ft height following room flow",
            "drone approach from 200ft banking to face facade",
            "reveal pull-back from detail to full room in 3 seconds",
            "window POV: interior looking out through window to view",
            "architectural tilt-shift: camera tilt down at 30° over property",
            "twilight time-lapse: 4 second compression of golden to blue hour",
            "high-angle corner shot maximizing perceived room size"
        ],
        lighting_vocabulary=[
            "golden hour 2700K exterior: warm glow on facade, long shadows, romance",
            "bright midday 5500K: showing maximum space, clean lines, natural light",
            "twilight: exterior 2800K warm interior lights through windows, blue sky",
            "interior natural: window 5600K, supplemental 3200K warm fill, no ceiling harshness",
            "luxury night: pool/garden 3000K, interior 2800K warm, facade uplighting 4000K"
        ],
        hook_patterns=[
            "dramatic exterior reveal: drone rises over trees to reveal full property",
            "golden hour: facade in perfect warm light creating immediate desire",
            "luxury feature close-up: marble texture, custom millwork, water feature",
            "before/after renovation: split-screen original vs transformed space",
            "window view reveal: camera approaches window to reveal stunning view beyond"
        ],
        style_tags=["architectural photography", "real estate", "interior design", "property", "luxury"],
        technical_specs={"fps": 24, "width": 1920, "height": 1080, "steps": 25, "cfg": 7.0},
        prompt_template="""You are an architectural photographer writing ComfyUI prompts.
Light is the primary storyteller:
- Golden hour exterior = romance, desire, lifestyle aspiration
- Bright midday = honest, spacious, transparency
- Twilight = luxury, warmth, domestic bliss
Camera movement principle: always purposeful, never arbitrary.
  "steadicam advancing 1 ft/s through archway reveals open plan living beyond" — purpose: reveal
  NOT "camera moving through room" — no purpose stated
Properties should feel aspirational AND lived-in simultaneously.
Include: time of day, weather, season, lifestyle occupant hints.""",
        comfyui_notes="Use architecture/interior LoRAs. Wide-angle renders need Realistic Vision or Deliberate.",
    ),
}

# ── Skill Auto-Detector ───────────────────────────────────────────────────────

def detect_skill(notes: str, override: Optional[str] = None) -> SkillSpec:
    """Auto-detect the best skill from user notes. Returns the SkillSpec."""
    if override and override in SKILLS:
        return SKILLS[override]

    notes_lower = notes.lower()
    scores: dict[str, int] = {sid: 0 for sid in SKILLS}

    for skill_id, skill in SKILLS.items():
        for keyword in skill.keywords:
            if keyword in notes_lower:
                scores[skill_id] += 1

    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0:
        return SKILLS["cinematic"]  # Default
    return SKILLS[best]


def get_skill_by_id(skill_id: str) -> Optional[SkillSpec]:
    return SKILLS.get(skill_id)


def list_skills() -> list[dict]:
    return [{"id": s.id, "name": s.name, "description": s.description} for s in SKILLS.values()]


# ── Prompt Builder ────────────────────────────────────────────────────────────

def build_comfyui_positive(
    base_prompt: str,
    skill: SkillSpec,
    include_quality_boosters: bool = True,
) -> str:
    """Build a complete ComfyUI positive prompt from a base prompt + skill."""
    parts = [base_prompt.strip()]
    if include_quality_boosters:
        parts.extend(skill.quality_boosters[:6])  # top 6 boosters
        parts.extend(skill.style_tags[:3])
    return ", ".join(parts)


def build_comfyui_negative(skill: SkillSpec, custom_negative: str = "") -> str:
    """Build a complete ComfyUI negative prompt from skill defaults."""
    base_negatives = [
        "ugly", "deformed", "disfigured", "bad anatomy", "blurry", "jpeg artifacts",
        "low quality", "worst quality", "lowres", "normal quality", "watermark", "signature",
    ]
    combined = list(dict.fromkeys(base_negatives + skill.negative_tags))  # deduplicate
    if custom_negative:
        combined.insert(0, custom_negative)
    return ", ".join(combined)


def get_workflow_overrides(skill: SkillSpec) -> dict:
    """Get ComfyUI generation parameter overrides for a skill."""
    return {
        "fps":    skill.technical_specs.get("fps", 24),
        "width":  skill.technical_specs.get("width", 512),
        "height": skill.technical_specs.get("height", 512),
        "steps":  skill.technical_specs.get("steps", 20),
        "cfg":    skill.technical_specs.get("cfg", 7.5),
    }
