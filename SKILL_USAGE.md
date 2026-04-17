# Skill Usage Guide — ComfyUI Video MCP

How to use the 15 Seedance 2.0 skill frameworks to generate cinema-quality videos via ComfyUI.

---

## The Core Idea

Skills are **prompt engineering frameworks** embedded in the MCP server. Each skill knows:

- What **camera moves** exist for that style (with exact velocity specs)
- What **lighting** looks right (with Kelvin temperature and intensity ratios)
- What **2-second hook** to open with (the universal engagement principle)
- What **quality tags** to inject into ComfyUI prompts
- What **resolution/FPS/CFG** settings ComfyUI should use

When you call `generate_ideas()`, the matching skill's framework becomes the LLM's system prompt — forcing cinema-precise language instead of generic descriptions.

---

## How Skills Actually Flow Into ComfyUI

The skill data doesn't just guide the LLM — it is **directly injected** into the final prompt string. Here's the complete data flow:

```
1. detect_skill("notes")
       │  keyword scoring across 15 SkillSpec objects
       ▼
2. skill.prompt_template
       │  becomes the LLM's system persona
       ▼
3. skill.camera_vocabulary + skill.lighting_vocabulary
       │  passed as examples inside SCENE_USER_TEMPLATE
       ▼
4. LLM writes scene prompt
       │
       ▼
5. build_comfyui_positive(base, skill)
       │  appends skill.quality_boosters[:6] + skill.style_tags[:3] to the prompt
       ▼
6. build_comfyui_negative(skill, custom)
       │  prepends custom + appends skill.negative_tags on top of base negatives
       ▼
7. get_workflow_overrides(skill)
       │  returns {fps, width, height, steps, cfg} for ComfyUI
       ▼
8. _enrich_scene(scene, skill)
       │  post-LLM pass: inject any missing skill.quality_boosters[:4]
       ▼
Final prompt reaches ComfyUI with skill vocabulary baked in
```

Direct use (from `run_generate.py`):

```python
from skills_engine import detect_skill, build_comfyui_positive, build_comfyui_negative

SKILL = detect_skill("tokyo cyberpunk anime virtual reality")
# → resolves to anime SkillSpec

# Pick from skill vocabulary by scene act
cam  = SKILL.camera_vocabulary
lite = SKILL.lighting_vocabulary

base = f"{PROTAGONIST}, {cam[0]}, {lite[1]}"            # HOOK scene
visual_prompt   = build_comfyui_positive(base, SKILL)   # → adds anime quality tags
negative_prompt = build_comfyui_negative(
    SKILL,
    custom_negative="static, frozen, no movement"
)                                                        # → adds anime negatives
```

---

## Scene Continuity System

Skills alone don't guarantee the same character appears across all 4 scenes. The continuity system enforces that.

### The problem skills cannot solve

Without continuity, an LLM generates each scene independently:
- Scene 1: "a woman in a VR pod"
- Scene 2: "a girl walking the street"
- Scene 3: "female protagonist facing the creature"
- Scene 4: "a figure kneels"

Technically the same person, but ComfyUI generates 4 different-looking characters.

### The fix: protagonist anchor + narrative arc

`idea_generator.py` adds three mechanisms:

**1. Protagonist anchor** — one description, locked across all scenes

```
young woman, sharp black bob haircut with violet streak, worn white oversized
jacket with circuit-board pattern, cracked holographic visor pushed up on
forehead, pale skin, dark circles under eyes
```

This exact string appears verbatim in Scene 1's `visual_prompt` and at the start of every subsequent scene's prompt. If clothing gets damaged (torn jacket, cracked visor), the damage is kept mentioned in later scenes.

**2. Narrative arc — HOOK → BUILD → CLIMAX → RESOLUTION**

| Scene | Act | Tension | Camera language | Lighting shift |
|---|---|---|---|---|
| 1 | HOOK | 10% | Wide establish pulling back from close-up | Neutral / calm |
| 2 | BUILD | 40% | Tracking motion | Warmer / more saturated |
| 3 | CLIMAX | 90% | Handheld Dutch tilt 15-25° | Harshest contrast |
| 4 | RESOLUTION | 60%↓ | Slow pull-back to wide | Softer fall-off |

**3. `_enforce_continuity()` post-processing pass**

After the LLM returns, this function:
- Extracts the protagonist anchor from Scene 1's dedicated `protagonist` field (or first descriptive clauses of the prompt if not set)
- Checks every subsequent scene's first 200 characters
- If anchor is missing, prepends `"same protagonist — [anchor], "`
- Re-numbers scene indices, assigns act labels, validates negative prompt length

Result: even if the LLM drifts, the final ComfyUI prompts have consistent character description.

---

## Quick Start: 3 Commands to a Video

```
generate_ideas("golden hour beach, surfer, slow motion sunrise")
select_idea(1)
generate_video()
```

That's it. The skill is auto-detected, prompts are generated, scenes are queued.

---

## All 15 Skills

### Overview Table

| Skill ID | Name | Best For | Auto-Detect Keywords | ComfyUI Res |
|----------|------|---------|---------------------|-------------|
| `cinematic` | Cinematic Film | Drama, documentary, realistic scenes | film, movie, dramatic, portrait | 768×512 |
| `anime` | Anime & Japanese Animation | Anime, manga, shonen, cyberpunk | anime, manga, shonen, kawaii, otaku | 768×432 |
| `3d_cgi` | 3D CGI Render | Product viz, architecture, abstract | 3d, cgi, render, blender, unreal | 768×432 |
| `cartoon` | Cartoon & Animation | Disney/Pixar style, kids content | cartoon, animation, disney, pixar | 768×432 |
| `fight_scenes` | Fight Scene & Action | Combat, martial arts, action | fight, combat, battle, ninja, explosion | 768×432 |
| `motion_design` | Motion Design & Tech Ad | SaaS, app demos, UI animation | app, software, saas, ui, dashboard | 1280×720 |
| `ecommerce` | E-Commerce Product Ad | Product shots, unboxing, retail | product, shop, retail, brand, unboxing | 1080×1080 |
| `social_hook` | Viral Social Hook | TikTok/Reels, scroll-stopping | tiktok, viral, trending, social, reels | 576×1024 |
| `music_video` | Music Video | Beat-synced content, performance | music, song, beat, concert, hip hop | 1280×720 |
| `brand_story` | Brand Storytelling | Brand films, founder stories | brand, story, founder, company, mission | 1280×720 |
| `fashion` | Fashion Lookbook | Editorial fashion, lookbooks | fashion, style, model, lookbook, runway | 1080×1350 |
| `food` | Food & Beverage | Cooking, restaurant, drinks | food, cooking, recipe, chef, coffee | 1080×1080 |
| `real_estate` | Real Estate & Architecture | Property tours, interiors | house, property, interior, architecture | 1920×1080 |

---

## Skill Deep Dives

### cinematic — Film-Style

**Use when:** You want realistic, dramatic, documentary-style footage.

```
detect_skill_for_notes("old man sitting alone in diner, night rain on window")
→ Detected: cinematic
```

**What the skill injects into prompts:**
- Camera: `"3 ft/s dolly forward"`, `"rack focus from foreground to background in 1.5 seconds"`, `"extreme close-up macro lens 50mm f/1.4"`
- Lighting: `"3000K key light at 45° camera-left, 100% intensity, 33% fill, 60% back rim"`, `"golden hour sunlight 2700K from camera-right"`
- Quality tags: `masterpiece, best quality, 8k uhd, film grain, cinematic color grading, anamorphic lens flare, shallow depth of field`
- ComfyUI settings: 768×512, 24fps, 25 steps, CFG 7.0

**Example session:**
```
generate_ideas("elderly man in empty diner late at night, rainy window, 1950s america")

[1] Last Customer
     An old man nurses cold coffee in a window booth as rain streaks the glass.
     Warm tungsten light contrasts with cold blue exterior. 35mm cinematic.
     Style: cinematic | Mood: melancholic

select_idea(1)

Scene 1: Exterior establishing shot
  Prompt: "Rain-soaked diner exterior, 1950s neon sign reflected in puddles,
           warm tungsten 3200K glow from interior windows, cold blue moonlight
           8000K exterior, dolly push-in at 0.5 ft/s toward window, 35mm
           anamorphic lens, slight film grain, masterpiece, best quality, 8k uhd,
           cinematic color grading, anamorphic lens flare..."
```

---

### anime — Japanese Animation

**Use when:** You want anime aesthetics — shonen action, slice of life, cyberpunk anime, mecha.

```
generate_ideas("high school girl discovers she can control lightning", skill_id="anime")
```

**What the skill injects:**

Genre-specific lighting vocabulary:
- Shonen: `"150% saturated warm amber sunlight from 45° above"`
- Cyberpunk: `"dark base with neon cyan 16000K and magenta 7000K"`
- Magical girl: `"soft pink 3200K omnidirectional sparkle glow"`

Camera techniques:
- `"impact frame freeze: single key frame held 0.3s with speed lines radiating outward"`
- `"smear frame transition between action positions"`
- `"spiral camera orbit at power-up moment"`

**Example session:**
```
generate_ideas("girl awakens lightning powers at train station, shonen style", skill_id="anime")

[1] Volt Rising: Platform Zero
     Lightning crackles from a student's fingertips as her school bag drops.
     Speed lines explode outward. 150% saturated shonen lighting.
     Style: Japanese animation | Mood: epic

select_idea(1)

Scene 1 prompt (generated):
  "High school girl at train platform, lightning arcs from raised hand,
   impact frame held 0.3s with speed lines radiating outward, shonen style:
   150% saturated warm amber sunlight from 45° above mixed with electric blue
   16000K power discharge, smear frame at peak power moment, torn jacket
   collar indicating energy level, speed lines converging from screen edges,
   masterpiece, best quality, detailed anime art, cel shading, anime shading,
   clean line art, key visual quality, official art style..."
```

---

### fight_scenes — Action & Combat

**Use when:** You need choreographed combat, martial arts, or high-intensity action.

**Critical rule this skill enforces:** *Specific choreography, never vague description.*

```
✓ "attacker launches right spinning heel kick, foot connects at jaw height,
   defender flows backward in 3 steps raising crossed forearms to parry"
✗ "they fight intensely"
```

**Example session:**
```
generate_ideas("underground boxing club, desperate fighter, rain seeping through ceiling")

[1] The Last Round
     A battered boxer faces a giant opponent in an abandoned warehouse,
     water dripping from exposed rebar. Handheld urgency and harsh shadows.
     Style: action | Mood: intense

select_idea(1)

Scene 2 prompt:
  "Desperate boxer launches left jab, opponent rolls right shoulder absorbing
   impact, cement dust falls from ceiling impact, camera pushes forward 5 ft/s
   on aggressive strike then pulls back 3ft as defender staggers, Dutch tilt
   20°, harsh side-key 6000K at 90° creating deep shadow opposite, strong
   rim light 180°, sweat spray frozen in 240fps slow-motion at point of contact,
   dynamic action shot, motion blur on moving parts, impact frames, masterpiece..."
```

---

### anime vs fight_scenes — Choosing for Combat

Both skills work for combat content. Here's when to use which:

| Situation | Use |
|-----------|-----|
| Anime-style characters fighting | `anime` |
| Realistic hand-to-hand combat | `fight_scenes` |
| Samurai / ninja with anime aesthetic | `anime` |
| MMA / boxing / military combat | `fight_scenes` |
| Superhero anime powers | `anime` |
| Street brawl / grounded realism | `fight_scenes` |

```
# Force a specific skill
generate_ideas("samurai vs ronin in cherry blossom field", skill_id="anime")
generate_ideas("boxer vs muay thai fighter in rooftop fight", skill_id="fight_scenes")
```

---

### social_hook — Viral Content

**Use when:** You want TikTok/Instagram Reels content — scroll-stopping, high retention.

This skill enforces the **2-second hook framework** on every single scene:

```
0.0 – 0.3s  → visual/audio shock  (stop the scroll)
0.3 – 0.8s  → incomplete reveal   (build curiosity)
0.8 – 1.5s  → momentum            (confirm the hook)
1.5 – 2.0s  → commitment moment   (viewer decides to stay)
```

Auto-sets resolution to **576×1024 (9:16 vertical)** for TikTok/Reels.

**Example session:**
```
generate_ideas("showing how to make satisfying pasta from scratch", skill_id="social_hook")

[1] 60-Second Pasta Magic
     Flour poured from impossible height, dough pulled to impossible stretch.
     Satisfying textures, rhythm cuts, no talking — pure visual ASMR.
     Style: social media | Mood: satisfying

select_idea(1)

Scene 1 prompt:
  "0–0.3s: flour falling from 3ft above dark surface — unexpected scale
   contrast stops scroll, vertical 9:16 framing, 0.3–0.8s: dough stretching
   to 3x length — incomplete reveal, 0.8–1.5s: hands folding dough rhythmically
   — momentum builds, direct overhead camera dropping vertically 2ft in 1 second,
   warm window 5600K, high contrast colors, bold visual rhythm, masterpiece..."
```

---

### music_video — Beat-Synced Content

**Use when:** Creating videos that sync to music, performance content, or genre-specific visual aesthetics.

Genre-visual mapping the skill knows:

| Genre | Visual Style |
|-------|-------------|
| Hip-Hop | Urban (warehouses, parking lots), hard shadows, neon practical |
| Pop | Bright saturated choreography, fast cuts on beat |
| EDM | Dark base, neon purple/cyan, laser grid, particles |
| Lo-Fi | Warm 2800K desaturated, vintage grain, minimal movement |
| Rock | Gritty, dramatic side-lighting, stage haze |

```
generate_ideas("lo-fi study session, late night rain, cozy desk", skill_id="music_video")

[1] 3AM Study Session
     Warm desk lamp glow, rain on window, slow camera drift over open books.
     Lo-fi: warm 2800K, vintage grain, nostalgic intimacy.
     Style: music video | Mood: melancholic

Scene 1 prompt:
  "Late-night desk with open textbooks, pencil resting across notebook,
   warm 2800K desk lamp creating golden circle on wooden surface, vintage
   16mm film grain overlaid at 40% opacity, rain streaking window glass
   in background, slow push-in 0.3 ft/s toward lamp, dust particles
   visible in beam, analog warmth, slightly desaturated, nostalgic tone,
   music video quality, beat-synced editing, visual rhythm..."
```

---

### ecommerce — Product Advertising

**Use when:** Showcasing products for sale — fashion, beauty, electronics, food products.

Auto-sets to **1080×1080** (square format for Instagram/Amazon).

**Example session:**
```
generate_ideas("luxury leather wallet, gift for him, premium unboxing", skill_id="ecommerce")

[1] The Artisan's Gift
     Slow silk reveal unfolds to expose hand-stitched tan leather wallet.
     Macro texture shots. Warm light for luxury positioning.
     Style: commercial photography | Mood: premium

Scene 1 prompt:
  "Silk cloth slowly pulling back from tan leather wallet, macro lens 1:1
   on hand-stitched seam detail, warm 2800K key light camera-left 45°,
   white foam-board fill opposite, warm reflections in buffed leather surface,
   smooth 360° turntable at 30°/s camera at 15° elevation, product fills
   60% of 1080×1080 frame, leather grain and thread detail crisp,
   commercial photography quality, product hero shot, lifestyle photography..."
```

---

### food — Food & Beverage

**Use when:** Restaurants, cooking channels, recipe videos, any food content.

**The money shot principle:** Every scene is built around 1 climactic 2-second moment — cheese pull, sauce cascade, chocolate pour, steam cloud.

**Example session:**
```
generate_ideas("homemade ramen, slow cooked chashu, midnight bowl")

[1] Midnight Bowl
     Steaming ramen bowl with chashu falling apart at the touch, golden
     tare drizzled in extreme slow-motion, backlit steam column.
     Style: food photography | Mood: warm

Scene 1 prompt (money shot):
  "Extreme close-up: tare sauce pouring from ceramic spoon in 0.25x
   slow-motion, golden liquid hitting ramen surface creating crown splash,
   backlit steam: 5500K hard back-key 180° makes steam column glow,
   warm 2800K front fill, pork chashu surface glistening with visible
   fat marbling, chopsticks breaking tender meat to reveal layered interior,
   steam venting with volumetric rays, macro lens sizzle sounds implied,
   food photography, appetizing, fresh ingredients, steam wisps, masterpiece..."
```

---

### real_estate — Property Showcase

**Use when:** House tours, architecture, interior design, property listings.

Auto-sets to **1920×1080** for maximum quality. Light as the primary storyteller:

| Time of Day | Lighting Spec | Communicates |
|-------------|--------------|--------------|
| Golden hour | 2700K, long shadows | Romance, desire |
| Bright midday | 5500K clean | Space, honesty |
| Twilight | Interior 2800K + blue sky | Luxury, warmth |
| Night | Pool/garden 3000K | Exclusivity |

**Example session:**
```
generate_ideas("modern minimalist home, ocean view, open plan living")

[1] Horizon House
     Steadicam glide through open-plan living toward floor-to-ceiling ocean view.
     Golden hour exterior, warm interior, aspirational but lived-in.

Scene 1 prompt:
  "Steadicam advancing 1 ft/s through archway reveals open-plan living toward
   floor-to-ceiling glass wall, Pacific ocean at golden hour 2700K beyond,
   warm interior 3200K pendant lights, dust motes visible in shaft of sunlight,
   camera at 5ft height following natural room flow, window frames ocean at
   golden hour with long shadows across polished concrete floor,
   architectural photography quality, real estate video, luxury property showcase..."
```

---

## Forcing vs Auto-Detecting Skills

### Auto-detect (default — recommended):
```
generate_ideas("rainy city alley, neon reflections, street photography")
→ server reads keywords → detects "cinematic"
```

### Force a specific skill:
```
generate_ideas("rainy city alley, neon reflections", skill_id="anime")
generate_ideas("rainy city alley, neon reflections", skill_id="social_hook")
```

Same notes, completely different cinematic language:

| Same Notes | `cinematic` Output | `anime` Output |
|------------|-------------------|----------------|
| Lighting | `"3000K key at 45°, 33% fill"` | `"neon cyan 16000K + magenta 7000K"` |
| Camera | `"rack focus from foreground"` | `"impact frame held 0.3s, speed lines"` |
| Quality tags | `"35mm film, anamorphic"` | `"cel shading, key visual quality"` |
| Resolution | 768×512 | 768×432 |

### Preview before generating:
```
detect_skill_for_notes("product launch video for smartphone app")
→ Detected: motion_design [motion_design]
→ Resolution: 1280×720
→ Camera: device float-in from below at 0.5 ft/s decelerating to rest
→ Lighting: dark premium: navy #0a0e27, electric blue #0066ff accent glow
```

### Switch skill on regenerate:
```
generate_ideas("street dancer in subway station")
→ auto: cinematic

regenerate_ideas("make it more viral, TikTok style", skill_id="social_hook")
→ switches to social_hook → vertical 576×1024, 2-second hook framework
```

---

## Configuring Pipeline Settings

### Switch video model:
```
configure_pipeline(model="wan21")     # best text-to-video quality
configure_pipeline(model="animatediff") # faster, good for animation
configure_pipeline(model="svd")         # image-to-video
```

### Override resolution (skill overrides applied per-scene automatically):
```
configure_pipeline(width=1024, height=576)   # 16:9 widescreen
configure_pipeline(width=576, height=1024)   # 9:16 vertical
configure_pipeline(width=1080, height=1080)  # square
```

### Switch LLM at runtime:
```
configure_pipeline(llm_provider="ollama")
configure_pipeline(llm_provider="claude")
configure_pipeline(llm_provider="offline")
```

### Adjust generation quality:
```
configure_pipeline(steps=30, cfg=8.0)   # more detailed, slower
configure_pipeline(steps=15, cfg=6.5)   # faster, less precise
configure_pipeline(frames=32, fps=8)    # longer clips
```

---

## Montage Options

After generating scenes:

```
# Simple compilation with default settings
compile_montage(title="My Video")

# With xfade transition
compile_montage(title="My Video", transition="dissolve")

# With background music
compile_montage(title="My Video", transition="wipe", music_path="C:/music/bgm.mp3")

# Specific resolution
compile_montage(title="My Video", resolution="1920x1080", fps=30)

# With title card at start
compile_montage(title="My Video", add_title_card=True)

# Pick specific clips by index (from list_videos)
compile_montage(title="My Video", video_ids=[1, 3, 5], use_session_videos=False)
```

### Transition Options

| Transition | Effect | Best For |
|------------|--------|---------|
| `fade` | Fade in/out to black | Dramatic, cinematic |
| `dissolve` | Cross-dissolve blend | Smooth, natural |
| `wipe` | Wipe left to right | Action, dynamic |
| `slide` | Slide left | Modern, tech |
| `zoom` | Zoom-in transition | Energetic |
| `none` | Hard cut | Music video, fast pacing |

---

## Full Pipeline Examples

### Example 1: Anime Short Film (no API key)

```yaml
# config.yaml
idea_generation:
  provider: "ollama"
  ollama_model: "llama3.2"
```

```
list_skills()

generate_ideas("fox spirit girl lost in modern tokyo searching for shrine", skill_id="anime")

[1] Spirit Finder
[2] Digital Kitsune
[3] ...

regenerate_ideas("make idea 2 more melancholic, seinen style", skill_id="anime")

[1] Neon Ghost: A Fox's Lament
     Desaturated cyberpunk tokyo, 9-tailed fox navigating between worlds...

select_idea(1)

configure_pipeline(model="animatediff", steps=25)

generate_video()

check_status(wait=True)

compile_montage(title="Neon Ghost", transition="dissolve")
```

---

### Example 2: Product Ad (TikTok vertical)

```
detect_skill_for_notes("lip gloss reveal, satisfying texture, beauty")
→ ecommerce detected, 1080×1080

# But we want TikTok vertical:
generate_ideas("luxury lip gloss texture reveal for TikTok", skill_id="social_hook")

select_idea(2)

# social_hook auto-sets 576×1024 (9:16)
generate_video()

compile_montage(title="Gloss Drop", transition="none", fps=30)
```

---

### Example 3: Real Estate Tour

```
configure_pipeline(model="wan21")  # best quality for architecture

generate_ideas("modern farmhouse kitchen reveal, white marble, golden hour", skill_id="real_estate")

select_idea(1)

generate_video(steps=30)  # high quality for property content

compile_montage(
    title="Farmhouse Kitchen Tour",
    transition="dissolve",
    resolution="1920x1080",
    fps=24
)
```

---

### Example 4: Lo-Fi Music Video

```
generate_ideas("late night study session, cozy desk lamp, rain on glass", skill_id="music_video")

select_idea(3)

configure_pipeline(fps=24, frames=48)  # longer clips for music content

generate_video()

compile_montage(
    title="Study Session",
    transition="fade",
    music_path="C:/music/lofi_beat.mp3",
    fps=24
)
```

---

## Checking What Happened

At any point:

```
session_status()
→ Shows:
   Notes:          "fox spirit girl in tokyo..."
   Ideas:          5 generated
   Selected idea:  [1] Neon Ghost: A Fox's Lament
   Scenes:         4 (2 generated)
   Pending jobs:   2
   Montages:       0

list_videos()
→ Generated videos (2 files):
   [1] scene_a3f2b1_video.mp4   (4.2 MB)
   [2] scene_d4e5f6_video.mp4   (3.8 MB)

list_montages()
→ Compiled montages (1):
   Neon_Ghost_1713276543.mp4   (18.4 MB)
```

---

## Prompt Quality Reference

What the skill framework adds to every ComfyUI prompt:

**Generic (no skill):**
```
tokyo street, anime style, night, neon lights, rain
```

**With `anime` skill:**
```
Rain-soaked neon alley, two figures in shadow, speed lines converging
from edges, impact frame opening held 0.3s with radiating speed lines,
shonen style: 150% saturated warm amber sunlight from 45° above mixed
with neon cyan 16000K left and magenta 7000K right, smear frame
transition between positions, dramatic pull-back from extreme close-up
to full scene, rain droplets frozen mid-air in slow-motion 0.25x,
torn jacket indicating damage, masterpiece, best quality, detailed
anime art, cel shading, anime shading, clean line art, key visual
quality, official art style, vibrant colors, expressive eyes, anime
style, manga style, Japanese animation, 2D animation
```

**Negative prompt (auto-built from skill):**
```
western cartoon, 3d render, realistic, photographic, sketch, rough
lines, inconsistent style, western comic, ugly, deformed, disfigured,
bad anatomy, blurry, jpeg artifacts, low quality, worst quality,
lowres, normal quality, watermark, signature
```

The specificity is what makes ComfyUI produce professional results.
