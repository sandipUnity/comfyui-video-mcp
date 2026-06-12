"""
Story and scene generation.

Two modes — whichever is available:
  1. Claude API  — if ANTHROPIC_API_KEY is set  (richer, more creative)
  2. Offline     — template-based               (always works)

Public API:
    generate_story_options(idea, duration_seconds, mood)
        → list[dict]  (3 story treatment options)

    generate_scenes_from_story(idea, story, character, style_dna, global_seed, duration_seconds,
                               t2i_width, t2i_height)
        → list[SceneState]

    generate_character_description(idea, story, mood)
        → str

Story option dict format:
    {
        "title":             "The Discovery",
        "summary":           "two-sentence description",
        "arc":               "stillness → awakening → revelation → transformation",
        "pacing":            "one sentence about rhythm and tension",
        "reasoning":         "why this fits the idea",
        "act_labels":        ["HOOK", "BUILD", "REVELATION", "RESOLUTION"],
        "scene_descriptions":["...", "...", "...", "..."],
    }
"""

from __future__ import annotations

import json
import os
import random
from typing import Optional

from pipeline.scene_state import SceneState, _scene_seed
from skills_engine import SKILLS, build_comfyui_positive, build_comfyui_negative, build_comfyui_video_prompt


# ── Shot-size assignment ──────────────────────────────────────────────────────
#
# Maps act labels → cinematographic shot size.
# Wide shots for world-building / resolution; close-ups for emotional peak.
# Paired with an explicit character-placement note so image models don't
# default to centred medium portraits for every scene.

_SHOT_SIZE_BY_ACT: dict[str, str] = {
    # Establishing / world acts → very wide
    "HOOK":           "EXTREME WIDE SHOT",
    "BEFORE":         "EXTREME WIDE SHOT",
    "ORDINARY":       "WIDE SHOT",
    "ORDINARY1":      "WIDE SHOT",
    "ORDINARY2":      "WIDE SHOT",
    "AFTER":          "WIDE SHOT",
    "CODA":           "WIDE SHOT",
    "RESOLUTION":     "WIDE SHOT",
    "REBORN":         "WIDE SHOT",
    "SETUP":          "WIDE SHOT",
    "DEPARTURE":      "WIDE SHOT",
    "JOURNEY":        "WIDE SHOT",
    # Transitional / momentum acts → medium wide / medium
    "INCITING":       "MEDIUM WIDE SHOT",
    "CATALYST":       "MEDIUM WIDE SHOT",
    "REGROUPING":     "MEDIUM WIDE SHOT",
    "BUILD":          "MEDIUM SHOT",
    "BUILD1":         "MEDIUM SHOT",
    "BUILD2":         "MEDIUM SHOT",
    "BUILD3":         "MEDIUM SHOT",
    "MIDPOINT":       "MEDIUM SHOT",
    "CHANGE":         "MEDIUM SHOT",
    "CHANGE1":        "MEDIUM SHOT",
    "CHANGE2":        "MEDIUM SHOT",
    "TEST":           "MEDIUM SHOT",
    "TEST1":          "MEDIUM SHOT",
    "TEST2":          "MEDIUM SHOT",
    "SETBACK":        "MEDIUM SHOT",
    "CHALLENGE":      "MEDIUM SHOT",
    # Emotional peak acts → close
    "CONFRONTATION":  "MEDIUM CLOSE-UP",
    "WONDER":         "WIDE SHOT",          # wonder = environment dominant
    "FIRST LIGHT":    "WIDE SHOT",          # light reveal = go wide
    "FIRST SIGHT":    "WIDE SHOT",
    "REVELATION":     "MEDIUM CLOSE-UP",
    "DISCOVERY":      "MEDIUM CLOSE-UP",
    "INSCRIPTION":    "MEDIUM CLOSE-UP",
    "TWIST":          "MEDIUM CLOSE-UP",
    "DOUBT":          "CLOSE-UP",
    "DECISION":       "MEDIUM CLOSE-UP",
    "RECKONING":      "MEDIUM CLOSE-UP",
    "CRISIS":         "CLOSE-UP",
    "SACRIFICE":      "MEDIUM SHOT",
    "CLIMAX":         "CLOSE-UP",
    "VICTORY":        "WIDE SHOT",          # triumphant wide
}

# Character placement note per shot size — appended when the character is in shot
_PLACEMENT_BY_SHOT: dict[str, str] = {
    "EXTREME WIDE SHOT":   "subject is a tiny figure in the landscape, occupying less than 10% of frame height, positioned at lower-center or off-center",
    "WIDE SHOT":           "full body visible, character occupies lower third of frame, environment dominates upper two-thirds",
    "MEDIUM WIDE SHOT":    "character from knees up, positioned at rule-of-thirds left or right, background clearly visible",
    "MEDIUM SHOT":         "waist-up framing, character slightly off-center, background in soft focus",
    "MEDIUM CLOSE-UP":     "chest-and-face framing, face occupies upper half of frame, shallow depth of field f/2.0",
    "CLOSE-UP":            "face fills the frame, extreme shallow depth of field f/1.4, background fully bokeh",
}

# Composition note when the shot contains NO character — pure environment/insert
_ENV_PLACEMENT_BY_SHOT: dict[str, str] = {
    "EXTREME WIDE SHOT":   "environment-only composition, no people in frame, layered depth with foreground-midground-background separation",
    "WIDE SHOT":           "environment-only composition, no people in frame, strong leading lines drawing the eye through the space",
    "MEDIUM WIDE SHOT":    "environment detail composition, no people in frame, architectural or natural forms as the subject",
    "MEDIUM SHOT":         "detail shot of the environment, no people in frame, textures and materials as the subject",
    "MEDIUM CLOSE-UP":     "insert shot of a significant object or detail, no people in frame, shallow depth of field f/2.0",
    "CLOSE-UP":            "extreme detail insert shot of an object or texture, no people in frame, macro-level sharpness on the subject",
}


# ── Character presence assignment ─────────────────────────────────────────────
#
# Cinematic coverage means the protagonist does NOT appear in every shot.
# Character consistency = the character looks identical *whenever on screen*,
# not that every frame is a character shot.

def _assign_character_presence(act: str, shot_size: str, scene_idx: int, total: int) -> str:
    """Return "featured" | "background" | "none" for this scene.

    Defaults (user-overridable per scene in the UI):
      - Opening extreme-wide establishing shot → pure environment ("none")
      - Other extreme-wide / wide shots        → distant figure ("background")
      - Medium and closer                      → character is the subject ("featured")
    """
    if shot_size == "EXTREME WIDE SHOT":
        return "none" if scene_idx == 0 else "background"
    if shot_size == "WIDE SHOT":
        return "background"
    return "featured"


def _distant_character_clause(character_desc: str) -> str:
    """Reduce a full character description to a silhouette-level distant cue.

    Used for "background" presence: facial detail is invisible at that distance,
    so only the first identifying clause (build / clothing) is kept.
    """
    if not character_desc:
        return ""
    first = character_desc.split(",")[0].split(".")[0].strip()
    return f"{first} visible only as a small distant figure, silhouette and clothing readable, no facial detail"


def _assign_shot_size(act: str, scene_idx: int, total: int) -> str:
    """Return the appropriate shot size for this scene.

    Priority:
      1. Act-label lookup (narrative intent)
      2. Position fallback: first and last scenes → wide; middle → vary by thirds
    """
    act_key = act.upper().strip()
    if act_key in _SHOT_SIZE_BY_ACT:
        shot = _SHOT_SIZE_BY_ACT[act_key]
    else:
        # Position-based fallback
        progress = scene_idx / max(total - 1, 1)
        if progress < 0.15 or progress > 0.85:
            shot = "WIDE SHOT"
        elif progress < 0.35 or progress > 0.65:
            shot = "MEDIUM WIDE SHOT"
        elif 0.45 < progress < 0.55:
            shot = "CLOSE-UP"
        else:
            shot = "MEDIUM SHOT"

    # Force variety: every 4th scene that would be MEDIUM SHOT → WIDE SHOT
    # so we never get 4 consecutive medium portraits
    if shot == "MEDIUM SHOT" and scene_idx % 4 == 3:
        shot = "WIDE SHOT"

    return shot


def _build_visual_prompt_with_framing(
    shot_size: str,
    scene_desc: str,
    camera: str,
    lighting: str,
    character_desc: str,
    skill,
    character_presence: str = "featured",
) -> str:
    """Build a mechanical visual prompt that leads with shot size + placement.

    Structure:  [SHOT SIZE]. [Placement note]. [Scene content]. [Camera]. [Lighting].
                [Character — after environment, only if in shot]. [Quality boosters].

    character_presence gates how the character appears:
      "featured"   → full description after the environment
      "background" → silhouette-level distant clause only
      "none"       → no character text; environment composition note instead
    """
    char_clean = character_desc.rstrip(", ")

    if character_presence == "none" or not char_clean:
        placement = _ENV_PLACEMENT_BY_SHOT.get(shot_size, "") if character_presence == "none" \
                    else _PLACEMENT_BY_SHOT.get(shot_size, "")
        core = f"{scene_desc}, {camera}, {lighting}"
    elif character_presence == "background":
        placement = _PLACEMENT_BY_SHOT.get(shot_size, "")
        core = f"{scene_desc}, {camera}, {lighting}, {_distant_character_clause(char_clean)}"
    else:  # featured
        placement = _PLACEMENT_BY_SHOT.get(shot_size, "")
        core = f"{scene_desc}, {camera}, {lighting}, {char_clean}"

    placement_clause = f" {placement}." if placement else ""
    base = f"{shot_size}.{placement_clause} {core}"
    return build_comfyui_positive(base, skill)


# ── Act label sets per template per scene count ───────────────────────────────

_ACTS: dict[str, dict[int, list[str]]] = {
    "discovery": {
        3:  ["HOOK",    "REVELATION",   "RESOLUTION"],
        4:  ["HOOK",    "BUILD",        "REVELATION",   "RESOLUTION"],
        5:  ["HOOK",    "SETUP",        "BUILD",        "REVELATION",   "RESOLUTION"],
        6:  ["HOOK",    "ORDINARY",     "INCITING",     "BUILD",        "REVELATION",   "RESOLUTION"],
        7:  ["HOOK",    "ORDINARY",     "INCITING",     "BUILD",        "CRISIS",       "REVELATION",   "RESOLUTION"],
        12: ["HOOK",    "ORDINARY",     "INCITING",     "BUILD1",       "BUILD2",       "MIDPOINT",
             "BUILD3",  "CRISIS",       "REVELATION",   "TWIST",        "CLIMAX",       "RESOLUTION"],
    },
    "struggle": {
        3:  ["HOOK",    "CONFRONTATION","VICTORY"],
        4:  ["HOOK",    "CHALLENGE",    "CONFRONTATION","VICTORY"],
        5:  ["HOOK",    "SETUP",        "CHALLENGE",    "CONFRONTATION","VICTORY"],
        6:  ["HOOK",    "ORDINARY",     "INCITING",     "CHALLENGE",    "CONFRONTATION","VICTORY"],
        7:  ["HOOK",    "ORDINARY",     "INCITING",     "BUILD",        "CONFRONTATION","CRISIS",    "VICTORY"],
        12: ["HOOK",    "ORDINARY",     "INCITING",     "BUILD",        "CHALLENGE",    "MIDPOINT",
             "SETBACK", "REGROUPING",   "CONFRONTATION","CRISIS",       "CLIMAX",       "VICTORY"],
    },
    "change": {
        3:  ["BEFORE",  "CATALYST",     "AFTER"],
        4:  ["BEFORE",  "CATALYST",     "CHANGE",       "AFTER"],
        5:  ["BEFORE",  "CATALYST",     "CHANGE",       "TEST",         "AFTER"],
        6:  ["BEFORE",  "ORDINARY",     "CATALYST",     "CHANGE",       "TEST",         "AFTER"],
        7:  ["BEFORE",  "ORDINARY",     "CATALYST",     "CHANGE",       "TEST",         "DOUBT",    "AFTER"],
        12: ["BEFORE",  "ORDINARY1",    "ORDINARY2",    "CATALYST",     "CHANGE1",      "CHANGE2",
             "TEST1",   "TEST2",        "DOUBT",        "DECISION",     "CLIMAX",       "AFTER"],
    },
}


def _closest_acts(template: str, n: int) -> list[str]:
    """Return the closest act list for a given n, padding/trimming as needed."""
    tpl = _ACTS[template]
    if n in tpl:
        return tpl[n]
    # Find closest key
    keys = sorted(tpl.keys())
    best = min(keys, key=lambda k: abs(k - n))
    acts = list(tpl[best])
    # Trim or pad to exactly n
    while len(acts) < n:
        acts.insert(-1, "BUILD")        # pad before last
    return acts[:n]


# ── Scene description templates ───────────────────────────────────────────────

def _scene_desc(act: str, idea: str, idx: int, total: int, mood: Optional[str] = None) -> str:
    """Generate a one-line cinematic scene description for an act label.

    Templates name a concrete subject, an action, and one striking detail —
    written as something a camera can frame, not an abstract summary.
    """
    act_up = act.upper()
    m = f", {mood} undertone" if mood else ""

    templates = {
        "HOOK":         f"A vast establishing frame of the world of {idea} — held perfectly still until one small detail moves and breaks the silence{m}",
        "ORDINARY":     f"Extreme close textures of the daily rhythm of {idea}: hands, surfaces, repeated motions, dust hanging in a shaft of light",
        "BEFORE":       f"The world of {idea} at rest — long shadows, slow drifting atmosphere, a single light source that will matter later",
        "INCITING":     f"One wrong element enters the frame of {idea} — small at first, reflected in a surface before it is seen directly",
        "CATALYST":     f"The exact moment {idea} tips: an object falls, a light changes colour, a line is crossed — shot tight on the point of contact",
        "SETUP":        f"Pieces of {idea} click into position one by one, each cut tighter than the last, the empty space in frame shrinking",
        "BUILD":        f"Motion accelerates through the world of {idea} — shadows lengthen, sound and movement stack until the frame can barely hold it",
        "BUILD1":       f"First pressure on {idea}: a hairline crack appears in something that looked permanent, almost too small to notice",
        "BUILD2":       f"The stakes of {idea} turn physical — wind rises, surfaces tremble, the comfortable distance between safety and danger halves",
        "BUILD3":       f"Everything in the world of {idea} converges on one point — converging lines, gathering crowd, narrowing corridor of light",
        "MIDPOINT":     f"A reveal flips the scale of {idea}: pull back or push in until what we thought we understood becomes something else entirely",
        "CHALLENGE":    f"The largest obstacle in {idea} fills the frame, dwarfing everything — shot from below, edges disappearing out of frame",
        "CONFRONTATION":f"Two forces of {idea} face each other across a charged gap — dust or rain hangs between them, nothing moves yet",
        "CRISIS":       f"The darkest frame of {idea}: a single failing light source, debris of what was built, stillness that reads as defeat",
        "SETBACK":      f"What was gained in {idea} slips away in one continuous motion — the camera holds as it goes, refusing to cut",
        "REGROUPING":   f"In the quiet wreckage of {idea}, one deliberate gesture begins the rebuild — small, precise, defiant",
        "REVELATION":   f"The hidden truth of {idea} is uncovered in hard light — a slow reveal that recontextualises the opening image",
        "DISCOVERY":    f"First full sight of the heart of {idea}: the frame opens wide, scale lands, dust and light pour through",
        "TWIST":        f"The frame of {idea} turns literal somersault — what was background becomes subject, an earlier detail returns meaning something new",
        "TEST":         f"The new strength of {idea} is struck hard, once — impact frozen at the moment of contact, outcome withheld a beat",
        "TEST1":        f"A first trial for {idea}: deliberate, watched, one chance — the surrounding world holds its breath",
        "TEST2":        f"The limit of {idea} is found and pushed past — material strain made visible: bending, glowing, fraying",
        "CHANGE":       f"Transformation made visible on {idea}: old surface giving way to new in one continuous visual metamorphosis",
        "CHANGE1":      f"The first piece of the old world of {idea} falls away — caught mid-air, weightless, beautiful",
        "CHANGE2":      f"Deep in transformation, {idea} is barely recognisable — mirrored against what it was in the opening frame",
        "DOUBT":        f"A long still frame inside {idea}: reflections, halved light, the visual language of a decision not yet made",
        "DECISION":     f"One decisive physical act commits {idea} forever — a door, a switch, a step over a visible line, no cut away",
        "CLIMAX":       f"Everything {idea} has built collides at maximum intensity — the motif from the opening returns at the centre of the frame",
        "VICTORY":      f"Wide triumphant frame of {idea} remade — the threat's geometry now broken on the ground, light fully returned",
        "RESOLUTION":   f"The world of {idea} settles into its new shape — same angle as the opening frame, everything changed inside it",
        "AFTER":        f"Long aftermath frame of {idea}: repaired, quieter, the planted motif resting where the story leaves it",
        "CODA":         f"A final held image of {idea} from a distance — small against the horizon, the question of frame one answered",
    }
    return templates.get(act_up, f"Scene {idx+1}: {idea}, {act.lower()} moment rendered as one concrete image")


# ── Offline story generation ──────────────────────────────────────────────────

def _offline_options(idea: str, n_scenes: int, mood: Optional[str]) -> list[dict]:
    mood_text = f" ({mood} tone)" if mood else ""

    return [
        {
            "title":       "First Light",
            "summary":     (
                f"A perfectly still world hides something extraordinary inside {idea} — and one small "
                f"moving detail gives it away{mood_text}. What is uncovered rewrites the meaning of the "
                f"opening frame."
            ),
            "arc":         "stillness → suspicion → pursuit → revelation → transformation",
            "pacing":      "A held-breath open, tightening cuts toward an explosive reveal, then one long exhale of an ending.",
            "reasoning":   "A discovery engine lets the audience uncover the truth at the exact moment the subject does — the strongest form of investment.",
            "act_labels":  _closest_acts("discovery", n_scenes),
            "scene_descriptions": [
                _scene_desc(act, idea, i, n_scenes, mood)
                for i, act in enumerate(_closest_acts("discovery", n_scenes))
            ],
        },
        {
            "title":       "Breaking Point",
            "summary":     (
                f"An opposing force presses on {idea} until something visibly cracks{mood_text}. "
                f"Victory is taken at the moment all the visual geometry says defeat."
            ),
            "arc":         "strength → pressure → collapse → defiance → triumph",
            "pacing":      "Relentless escalation, a near-silent rock-bottom beat, then a sharp cathartic release.",
            "reasoning":   "A duel engine creates physical, frameable stakes — every scene shows force against resistance.",
            "act_labels":  _closest_acts("struggle", n_scenes),
            "scene_descriptions": [
                _scene_desc(act, idea, i, n_scenes, mood)
                for i, act in enumerate(_closest_acts("struggle", n_scenes))
            ],
        },
        {
            "title":       "Point of No Return",
            "summary":     (
                f"One irreversible act sets {idea} transforming, piece by visible piece{mood_text}. "
                f"The final frame mirrors the first — same angle, everything inside it changed."
            ),
            "arc":         "ordinary → rupture → metamorphosis → trial → rebirth",
            "pacing":      "A quiet ordinary world ruptured early, a long mesmerising transformation, one hard test, a still new reality.",
            "reasoning":   "A transformation engine delivers the strongest before/after contrast — the bookend framing makes change visible.",
            "act_labels":  _closest_acts("change", n_scenes),
            "scene_descriptions": [
                _scene_desc(act, idea, i, n_scenes, mood)
                for i, act in enumerate(_closest_acts("change", n_scenes))
            ],
        },
    ]


# ── Claude-powered story generation ──────────────────────────────────────────

def _claude_options(idea: str, n_scenes: int, mood: Optional[str]) -> list[dict] | None:
    """Call Claude API. Returns None on any failure (caller falls back to offline)."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic   # optional dependency
        client = anthropic.Anthropic(api_key=api_key)
        mood_clause = f" The mood/tone should be: {mood}." if mood else ""
        prompt = (
            f"You are an award-winning director. Generate exactly 3 distinct narrative treatments "
            f"for a {n_scenes * 5}-second video about: \"{idea}\".{mood_clause}\n\n"
            f"Each treatment should suggest exactly {n_scenes} scenes.\n\n"
            f"CRAFT REQUIREMENTS:\n"
            f"- Scene 1 is a striking visual hook that raises a question — never someone simply "
            f"standing, walking, or waking up\n"
            f"- Plant one concrete visual motif early and pay it off with new meaning in the final scene\n"
            f"- Every scene escalates; around the midpoint something flips (reveal/reversal/scale change)\n"
            f"- Adjacent scenes collide: vast vs intimate, still vs violent, dark vs blinding\n"
            f"- The final scene is one indelible image answering the question scene 1 asked\n"
            f"- Every scene description: concrete subject + concrete action + ONE striking visual detail\n"
            f"- Banned: 'we see', 'the camera shows', 'a sense of', 'begins to'; banned generic titles "
            f"containing 'Journey', 'Discovery', 'Story', 'Tale'\n\n"
            f"Return a JSON array of 3 objects, each with these exact keys:\n"
            f'  "title":             evocative name (2-4 words)\n'
            f'  "summary":           2-sentence description containing the central visual hook\n'
            f'  "arc":               emotional journey as 5 beats separated by →\n'
            f'  "pacing":            one sentence about rhythm and tension\n'
            f'  "reasoning":         one sentence on why this structure fits the idea\n'
            f'  "act_labels":        list of exactly {n_scenes} act names in UPPERCASE (e.g. HOOK, BUILD, CLIMAX, RESOLUTION)\n'
            f'  "scene_descriptions": list of exactly {n_scenes} one-sentence scene descriptions\n\n'
            f"Write coverage like a film director: mix establishing shots, pure environment beats, "
            f"and detail/insert shots with character moments — the protagonist must NOT appear in "
            f"every scene description.\n\n"
            f"Respond with ONLY the JSON array. No markdown, no explanation."
        )
        msg = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()
        # Extract JSON array
        start = text.find("[")
        end   = text.rfind("]") + 1
        if start < 0 or end <= start:
            return None
        options = json.loads(text[start:end])
        # Validate structure
        for opt in options:
            for key in ("title", "summary", "arc", "pacing", "reasoning", "act_labels", "scene_descriptions"):
                if key not in opt:
                    return None
        return options
    except Exception:
        return None


# ── Character description generation ─────────────────────────────────────────

_CHARACTER_TEMPLATES = [
    "{subject}, athletic build, determined expression, wearing practical dark clothing, "
    "mid-30s, distinctive scar above left eyebrow, moves with purpose and economy of motion",

    "{subject}, lean and weathered, eyes that have seen too much, silver-streaked hair "
    "kept close, a quiet intensity that fills every room they enter",

    "{subject}, young but world-weary, bright sharp eyes contrasting worn features, "
    "always in motion, hands rarely still, a nervous energy barely contained",

    "{subject}, imposing presence, broad shoulders, unhurried movements, "
    "deep-set eyes that miss nothing, a stillness that makes others uneasy",
]


def generate_character_description(idea: str, story: dict, mood: Optional[str] = None) -> str:
    """Generate a protagonist description. Uses Claude if available, else template."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        desc = _claude_character(idea, story, mood)
        if desc:
            return desc

    # Extract a subject hint from the idea
    words = idea.split()
    subject = " ".join(words[:4]) if len(words) > 4 else idea
    template = random.choice(_CHARACTER_TEMPLATES)
    return template.format(subject=f"The protagonist of '{subject}'")


def _claude_character(idea: str, story: dict, mood: Optional[str]) -> str | None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        prompt = (
            f"Write a vivid, specific 2-3 sentence visual description of the protagonist "
            f"for a video about: \"{idea}\"\n"
            f"Story treatment: {story.get('title', '')}: {story.get('summary', '')}\n"
            f"{'Mood: ' + mood if mood else ''}\n\n"
            f"Focus on: appearance, age, clothing, distinctive physical traits, how they move. "
            f"Be specific and visual — this will be injected into image generation prompts. "
            f"No backstory, no emotions — only what a camera would see. "
            f"Write the description directly, no intro or label."
        )
        msg = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception:
        return None


# ── Claude visual-prompt batch ────────────────────────────────────────────────

# System prompt injected from skill.prompt_template at call time.
# User message asks for N ComfyUI image prompts in one round-trip.

def _claude_visual_prompts_batch(
    scenes: list[dict],          # [{act, description, camera, lighting}, ...]
    skill,                       # SkillSpec
    character_desc: str,
    idea: str,
) -> list[str] | None:
    """One Claude call → list[visual_prompt] for all scenes, or None on failure.

    Uses skill.prompt_template as the system prompt so Claude behaves like a
    professional cinematographer for the correct genre/style.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        n = len(scenes)

        def _scene_lines(i: int, s: dict) -> str:
            shot     = s.get("shot_size", "MEDIUM SHOT")
            presence = s.get("character_presence", "featured").upper()
            placement = (
                _ENV_PLACEMENT_BY_SHOT.get(shot, "") if presence == "NONE"
                else _PLACEMENT_BY_SHOT.get(shot, "")
            )
            return (
                f"{i+1}. Act: {s['act']} | Shot: {shot} | Character: {presence}\n"
                f"   Placement: {placement}\n"
                f"   Scene: {s['description']}\n"
                f"   Camera: {s['camera']}\n"
                f"   Lighting: {s['lighting']}"
            )

        scenes_text = "\n".join(_scene_lines(i, s) for i, s in enumerate(scenes))
        char_line = f'Protagonist (use ONLY in scenes marked Character: FEATURED or BACKGROUND): "{character_desc}"' if character_desc else ""

        user_msg = (
            f"Generate exactly {n} ComfyUI positive image-generation prompts "
            f"for a CINEMATIC video project.\n\n"
            f'Project idea: "{idea}"\n'
            f"{char_line}\n\n"
            "CHARACTER PRESENCE — this is how real films are shot. Character consistency means "
            "the protagonist looks IDENTICAL whenever they are on screen — NOT that they appear "
            "in every shot. Each scene is marked:\n"
            "  Character: FEATURED   → weave the FULL protagonist description in AFTER the environment\n"
            "  Character: BACKGROUND → protagonist is a small distant figure; mention only silhouette,\n"
            "                          build and clothing colour — NO facial detail\n"
            "  Character: NONE       → pure environment / establishing / insert shot. The protagonist\n"
            "                          must NOT appear and must NOT be mentioned at all\n\n"
            "CRITICAL RULES — read every line:\n"
            "1. BEGIN each prompt with the SHOT SIZE (e.g. 'EXTREME WIDE SHOT.', 'CLOSE-UP.') — exactly as specified\n"
            "2. IMMEDIATELY follow with the PLACEMENT note for that shot\n"
            "3. Then describe the ENVIRONMENT and SCENE ACTION\n"
            "4. Apply the CHARACTER PRESENCE marking for that scene — full description only when FEATURED\n"
            "5. Include the exact camera move and lighting as specified\n"
            "6. Append quality-boosters and style tags at the very end\n"
            "7. Each prompt: under 150 words, single paragraph, no line breaks\n"
            "8. Every prompt MUST open with a DIFFERENT shot size — variety is mandatory\n"
            "9. NEVER produce a medium portrait of a centred character — use the placement note\n"
            "10. Output ONLY a JSON array of strings — no markdown, no labels\n\n"
            f"Scenes:\n{scenes_text}\n\n"
            f"Return a JSON array of exactly {n} strings."
        )

        msg = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=4096,
            system=skill.prompt_template,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = msg.content[0].text.strip()
        start, end = text.find("["), text.rfind("]") + 1
        if start < 0 or end <= start:
            return None
        prompts = json.loads(text[start:end])
        if len(prompts) != n:
            return None
        return [str(p).strip() for p in prompts]
    except Exception:
        return None


# ── Claude video-prompt batch ─────────────────────────────────────────────────

_VIDEO_PROMPT_SYSTEM = """\
You are an expert at writing motion prompts for AI image-to-video generation (I2V).

Your sole job is to describe MOVEMENT and ACTION — never appearance.
The reference image already defines exactly how everything looks.

Motion prompt rules:
1. Describe what moves and how: speed, direction, arc, distance
2. Include the camera move precisely (dolly speed in ft/s, pan degrees, etc.)
3. Include environmental motion: wind, water, fabric, smoke, crowd
4. Maximum 60 words per prompt — tight and specific beats vague and long
5. NEVER mention: colors, clothing details, facial features, or physical appearance
6. End with a single pacing word: [slow] [medium] [fast] [explosive]
"""


def _claude_video_prompts_batch(
    scenes: list[dict],          # [{act, description, camera}, ...]
    skill,                       # SkillSpec
    character_desc: str,
    motion_style: str,
    idea: str,
) -> list[str] | None:
    """One Claude call → list[video_prompt] for all scenes, or None on failure."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        n = len(scenes)

        cam_examples = "\n".join(
            f"  • {c}" for c in skill.camera_vocabulary[:4]
        )
        scenes_text = "\n".join(
            f"{i+1}. Act: {s['act']} | Character: {s.get('character_presence', 'featured').upper()} — {s['description']}\n"
            f"   Assigned camera: {s['camera']}"
            for i, s in enumerate(scenes)
        )
        subject_hint = character_desc[:80] if character_desc else idea

        user_msg = (
            f"Write {n} I2V motion prompts for these scenes.\n\n"
            f'Subject: "{subject_hint}"\n'
            f"Project motion style: {motion_style}\n\n"
            f"Camera vocabulary reference:\n{cam_examples}\n\n"
            f"Scenes:\n{scenes_text}\n\n"
            "Rules: motion only, under 60 words each, precise language.\n"
            "Character presence per scene:\n"
            "  FEATURED   → describe the subject's movement plus camera + environment motion\n"
            "  BACKGROUND → the subject is a distant figure; describe their broad movement only\n"
            "  NONE       → no person in this shot; describe ONLY camera + environmental motion\n"
            f"Return ONLY a JSON array of {n} strings."
        )

        msg = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=2048,
            system=_VIDEO_PROMPT_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = msg.content[0].text.strip()
        start, end = text.find("["), text.rfind("]") + 1
        if start < 0 or end <= start:
            return None
        prompts = json.loads(text[start:end])
        if len(prompts) != n:
            return None
        return [str(p).strip() for p in prompts]
    except Exception:
        return None


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_story_options(
    idea: str,
    duration_seconds: int = 30,
    mood: Optional[str] = None,
) -> list[dict]:
    """Return 3 story treatment options for the given idea.

    Tries Claude API first (if ANTHROPIC_API_KEY is set), falls back to offline templates.
    """
    n_scenes = max(3, duration_seconds // 5)
    options = _claude_options(idea, n_scenes, mood)
    if options and len(options) >= 3:
        return options[:3]
    return _offline_options(idea, n_scenes, mood)


def generate_scenes_from_story(
    idea: str,
    story: dict,
    character,                      # Character | None
    style_dna,                      # StyleDNA
    global_seed: int,
    duration_seconds: int = 30,
    t2i_width: int = 1024,
    t2i_height: int = 1024,
) -> list[SceneState]:
    """Build SceneState objects from a selected story treatment.

    Prompt generation strategy (two-tier):
      1. Claude API — if ANTHROPIC_API_KEY is set.
         Visual prompts: one batch call using skill.prompt_template as system
                         prompt so Claude writes like a genre expert.
         Video prompts:  one batch call with _VIDEO_PROMPT_SYSTEM so Claude
                         describes only motion/action (not appearance).
      2. Mechanical fallback — always available, no API key required.
         Visual: character + scene desc + camera + lighting + quality boosters
         Video:  scene desc + camera move + project motion style

    Args:
        idea:             Original idea string.
        story:            One story option dict from generate_story_options().
        character:        Locked Character (or None if not yet set).
        style_dna:        Inferred StyleDNA for this project.
        global_seed:      Project-level random seed.
        duration_seconds: Total target duration.
        t2i_width:        T2I image width.
        t2i_height:       T2I image height.

    Returns:
        List of SceneState, one per scene.
    """
    skill = SKILLS.get(style_dna.skill_id, SKILLS["cinematic"])
    act_labels  = list(story.get("act_labels", []))
    scene_descs = list(story.get("scene_descriptions", []))
    n = max(len(act_labels), len(scene_descs), max(3, duration_seconds // 5))

    # Pad if mismatched
    while len(act_labels)  < n:
        act_labels.append("BUILD")
    while len(scene_descs) < n:
        scene_descs.append(f"Scene {len(scene_descs)+1}: {idea}")

    character_desc = character.description if character else ""

    # ── Step 1: build per-scene structure + mechanical prompts (always works) ──
    scene_inputs: list[dict] = []
    scenes: list[SceneState] = []

    for i in range(n):
        act      = act_labels[i]
        desc     = scene_descs[i]
        cam_idx  = i % len(skill.camera_vocabulary)
        lite_idx = i % len(skill.lighting_vocabulary)
        cam      = skill.camera_vocabulary[cam_idx]
        lite     = skill.lighting_vocabulary[lite_idx]

        # Assign narrative-aware shot size (wide/medium/close based on act + position)
        shot_size = _assign_shot_size(act, i, n)

        # Decide whether the character is in this shot at all — cinematic
        # coverage mixes establishing/insert shots with character shots.
        presence = _assign_character_presence(act, shot_size, i, n)

        # Mechanical visual prompt — leads with shot size + placement directive
        # so the model frames correctly even without Claude.
        visual_prompt   = _build_visual_prompt_with_framing(
            shot_size, desc, cam, lite, character_desc, skill,
            character_presence=presence,
        )
        negative_prompt = build_comfyui_negative(skill)
        video_prompt    = build_comfyui_video_prompt(
            f"{desc}, {character_desc}" if (character_desc and presence == "featured") else desc,
            skill, cam, style_dna.motion_style
        )

        scene_inputs.append({
            "act": act, "description": desc, "camera": cam, "lighting": lite,
            "shot_size": shot_size, "character_presence": presence,
        })
        scenes.append(SceneState(
            scene_id        = f"scene_{i+1:02d}",
            scene_number    = i + 1,
            act             = act,
            description     = desc,
            environment     = "",
            camera_index    = cam_idx,
            lighting_index  = lite_idx,
            camera          = cam,
            lighting        = lite,
            base_prompt     = desc,
            visual_prompt   = visual_prompt,
            negative_prompt = negative_prompt,
            video_prompt    = video_prompt,
            shot_size       = shot_size,
            character_presence = presence,
            seed            = _scene_seed(global_seed, i + 1),
            status          = "pending",
        ))

    # ── Step 2: try Claude for visual prompts (one batch call) ────────────────
    claude_visual = _claude_visual_prompts_batch(
        scene_inputs, skill, character_desc, idea
    )
    if claude_visual:
        for scene, prompt in zip(scenes, claude_visual):
            scene.visual_prompt = prompt

    # ── Step 3: try Claude for video prompts (one batch call) ─────────────────
    claude_video = _claude_video_prompts_batch(
        scene_inputs, skill, character_desc, style_dna.motion_style, idea
    )
    if claude_video:
        for scene, prompt in zip(scenes, claude_video):
            scene.video_prompt = prompt

    return scenes
