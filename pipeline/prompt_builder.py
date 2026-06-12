"""
Prompt Builder — generates self-contained copy-paste prompts for every AI stage.

These prompts work verbatim in any chatbot (Claude.ai, ChatGPT, Gemini,
Claude Code). They include all project context and specify the exact JSON
response format. The user copies the prompt, pastes it into any chatbot,
copies the JSON response, pastes it back into the UI.

Public API:
    build_story_prompt(idea, n_scenes, mood, style_dna)          → str
    build_character_prompt(idea, story, mood, style_dna,
                           has_image_hint=False)                  → str
    build_scene_prompts_prompt(idea, story, character,
                               style_dna, scenes)                 → str
    build_single_scene_prompt(idea, scene, character, style_dna) → str
"""

from __future__ import annotations

from typing import Optional, List


# ── Visual helpers ────────────────────────────────────────────────────────────

def _header(title: str) -> str:
    width = 72
    inner = f"  AI VIDEO PIPELINE — {title}"
    pad   = width - len(inner) - 2   # 2 for ║ chars
    return (
        f"╔{'═' * width}╗\n"
        f"║{inner}{' ' * max(pad, 1)}║\n"
        f"╚{'═' * width}╝"
    )


def _style_block(style_dna) -> str:
    if style_dna is None:
        return ""
    palette_str = ", ".join(style_dna.color_palette[:3]) if style_dna.color_palette else "—"
    return (
        f"VISUAL STYLE:\n"
        f"  Name:    {style_dna.skill_name}\n"
        f"  Visual:  {style_dna.visual_style}\n"
        f"  Motion:  {style_dna.motion_style}\n"
        f"  Palette: {palette_str}\n"
        f"  Lighting: {style_dna.lighting_style}"
    )


def _story_block(story: dict) -> str:
    return (
        f"SELECTED STORY:\n"
        f"  Title:   {story.get('title', '—')}\n"
        f"  Summary: {story.get('summary', '—')}\n"
        f"  Arc:     {story.get('arc', '—')}"
    )


def _scenes_block(scenes: list, include_presence: bool = False) -> str:
    lines = [f"SCENES ({len(scenes)} total):"]
    for s in scenes:
        shot = getattr(s, "shot_size", "") or "MEDIUM SHOT"
        presence_part = ""
        if include_presence:
            presence = (getattr(s, "character_presence", "") or "featured").upper()
            presence_part = f" | Character: {presence}"
        lines.append(
            f"\n  Scene {s.scene_number} | Act: {s.act} | Shot: {shot}{presence_part}\n"
            f"  Description: \"{s.description}\"\n"
            f"  Camera:      \"{s.camera}\"\n"
            f"  Lighting:    \"{s.lighting}\""
        )
    return "\n".join(lines)


# Explains the per-scene "Character:" marking to the AI. Only included when the
# project actually has a locked character.
_PRESENCE_RULES = """\
CHARACTER PRESENCE — shoot like a real film. Character consistency means the
protagonist looks IDENTICAL whenever they are on screen — it does NOT mean they
appear in every shot. Each scene is marked with one of:
  Character: FEATURED   → weave the FULL character description in after the environment
  Character: BACKGROUND → protagonist is a small distant figure; mention only
                          silhouette, build and clothing colour — no facial detail
  Character: NONE       → pure environment / establishing / insert shot — the
                          protagonist must NOT appear and must NOT be mentioned"""


# ── Stage 3 — Story Options ───────────────────────────────────────────────────

def build_story_prompt(
    idea: str,
    n_scenes: int,
    mood: Optional[str],
    style_dna,
    media_paths: Optional[list] = None,
) -> str:
    """Generate the copy-paste prompt for Stage 3 — Story Options.

    Returns a complete, self-contained string that any chatbot can process.
    The response must match the JSON schema at the bottom of the prompt.
    """
    mood_line = f"  Mood:     {mood}" if mood else "  Mood:     (not specified)"

    schema = '''{
  "stage": "story_options",
  "result": [
    {
      "title": "short 2-4 word title",
      "summary": "sentence one. sentence two.",
      "arc": "beat 1 → beat 2 → beat 3 → beat 4 → beat 5",
      "pacing": "one sentence about rhythm and tension",
      "reasoning": "one sentence on why this structure fits the idea",
      "act_labels": ["HOOK", "BUILD", "CLIMAX", "RESOLUTION"],
      "scene_descriptions": ["scene 1 one-sentence description", "..."]
    },
    { "...": "option 2 same structure" },
    { "...": "option 3 same structure" }
  ]
}'''

    media_block = _media_section(media_paths)

    return f"""{_header("STAGE 3: STORY OPTIONS")}

You are a professional screenwriter and video director specialising in
short-form cinematic storytelling.

PROJECT CONTEXT:
  Idea:     "{idea}"
  Duration: {n_scenes * 5}s ({n_scenes} scenes × 5 seconds each)
{mood_line}

{_style_block(style_dna)}
{media_block}

YOUR TASK:
Generate exactly 3 distinct story treatment options for this video project.
Each treatment must cover exactly {n_scenes} scenes.

RULES:
1. Each of the 3 treatments must have a genuinely different narrative structure
2. Act labels must be UPPERCASE (e.g. HOOK, BUILD, CLIMAX, RESOLUTION)
3. Scene descriptions must be exactly one sentence each — visual and specific
4. Write coverage like a film director: mix establishing shots, pure environment
   beats, and detail/insert shots with character moments — the protagonist must
   NOT appear in every scene description (aim for 1-2 scenes with no character)
5. "summary" must be exactly 2 sentences
6. "arc" must be exactly 5 emotional beats separated by →
7. "reasoning" must be one sentence explaining why this structure fits the idea
8. "act_labels" and "scene_descriptions" must each have exactly {n_scenes} items
9. If reference images are attached, let them directly inform the visual style,
   setting, and aesthetic choices in your scene descriptions
10. Do not add any explanation, preamble, or text outside the JSON block

RESPOND WITH ONLY THIS JSON — no text before or after:

{schema}"""


# ── Stage 4 — Character Description ──────────────────────────────────────────

def build_character_prompt(
    idea: str,
    story: Optional[dict],
    mood: Optional[str],
    style_dna,
    has_image_hint: bool = False,
    media_paths: Optional[list] = None,
) -> str:
    """Generate the copy-paste prompt for Stage 4 — Character Description.

    Args:
        has_image_hint: If True, adds a note that the user may attach a
                        character reference image to the chatbot message.
        media_paths:    Absolute paths to character reference images on disk.
                        When provided they override has_image_hint (paths take
                        precedence — the AI gets the full REFERENCE MEDIA block).
    """
    mood_line  = f"  Mood:    {mood}" if mood else "  Mood:    (not specified)"
    story_sect = f"\n{_story_block(story)}\n" if story else ""

    # Media section — specific paths take priority over the generic hint
    if media_paths:
        image_section = _media_section(media_paths) + "\n"
        image_section += (
            "  Use the image(s) above as the PRIMARY visual reference. "
            "Your description must match what is visible as closely as possible.\n"
        )
    elif has_image_hint:
        image_section = """
CHARACTER REFERENCE IMAGE:
  A reference image has been shared with this message.
  Use it as the PRIMARY visual reference — your description must match
  what is visible in the image as closely as possible.
  If there is no image attached, describe a character that fits the project.
"""
    else:
        image_section = ""

    schema = '''{
  "stage": "character_description",
  "result": "Complete 2-3 sentence visual description as a single string."
}'''

    return f"""{_header("STAGE 4: CHARACTER DESCRIPTION")}

You are a professional casting director and visual development artist.
Your descriptions are used directly as image generation prompts — they
must be purely visual, hyper-specific, and camera-ready.

PROJECT CONTEXT:
  Idea:    "{idea}"
{mood_line}
{_style_block(style_dna)}
{story_sect}{image_section}
YOUR TASK:
Write a vivid 2-3 sentence visual description of the protagonist.

RULES:
1. Describe ONLY what a camera would see — no backstory, no emotions, no personality traits
2. Include: approximate age, build, distinctive facial features, hair, clothing/armour/costume
3. Include one uniquely distinctive detail that will make this character recognisable
   across all {("" if not story else str(len(story.get("act_labels", []))) + " ")}scenes
4. Be specific enough that two different image models produce similar-looking results
5. Never use vague words like "beautiful", "stunning", "interesting", "unique"
6. If reference images are attached, prioritise what you see over generic defaults
7. Do not add any text outside the JSON block

RESPOND WITH ONLY THIS JSON — no text before or after:

{schema}"""


# ── Stage 5 — Scene Prompts (Full Batch) ─────────────────────────────────────

def build_scene_prompts_prompt(
    idea: str,
    story: Optional[dict],
    character,              # Character | None
    style_dna,
    scenes: list,           # list[SceneState]
) -> str:
    """Generate the copy-paste prompt for Stage 5 — all visual + video prompts.

    One call covers all N scenes, returning two arrays:
    visual_prompts[N] and video_prompts[N].
    """
    char_line   = f'  Character: "{character.description}"' if character else ""
    n           = len(scenes)
    skill_notes = _get_skill_notes(style_dna)
    has_char    = character is not None
    presence_block = f"\n{_PRESENCE_RULES}\n" if has_char else ""

    visual_schema_items  = "\n    ".join(f'"Scene {i+1} visual prompt",' for i in range(n))
    video_schema_items   = "\n    ".join(f'"Scene {i+1} motion prompt",' for i in range(n))

    schema = f'''{{"stage": "scene_prompts",
  "result": {{
    "visual_prompts": [
    {visual_schema_items}
    ],
    "video_prompts": [
    {video_schema_items}
    ]
  }}
}}'''

    return f"""{_header("STAGE 5: SCENE PROMPTS")}

You are working as two experts simultaneously:
  VISUAL EXPERT — a professional cinematographer writing ComfyUI image prompts
  MOTION EXPERT — an I2V director writing motion-only video prompts

CINEMATOGRAPHER GUIDELINES:
{skill_notes}

PROJECT CONTEXT:
  Idea:    "{idea}"
{char_line}
  Motion style: "{style_dna.motion_style if style_dna else ""}"

{_style_block(style_dna)}
{presence_block}
{_scenes_block(scenes, include_presence=has_char)}

━━━ VISUAL PROMPTS — {n} required ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL COMPOSITION RULES — follow every one:
1. BEGIN each prompt with the SHOT SIZE exactly as listed (e.g. "EXTREME WIDE SHOT.", "CLOSE-UP.")
2. If the character IS in the shot (FEATURED or BACKGROUND), follow with their POSITION IN FRAME:
   - EXTREME WIDE SHOT  → "subject is a tiny figure, less than 10% of frame height, lower-center"
   - WIDE SHOT          → "full body visible in lower third, environment fills upper two-thirds"
   - MEDIUM WIDE SHOT   → "knees-up, positioned at rule-of-thirds left or right"
   - MEDIUM SHOT        → "waist-up, slightly off-center, background in soft focus"
   - MEDIUM CLOSE-UP    → "chest-and-face, face in upper half, f/2.0 shallow DOF"
   - CLOSE-UP           → "face fills frame, extreme shallow DOF f/1.4, background fully bokeh"
   If Character: NONE, instead open with an environment composition note
   (leading lines, layered depth, or a detail/insert subject) — NO people in frame
3. Then describe the ENVIRONMENT and SCENE ACTION
4. Apply the scene's Character marking: full description only when FEATURED (always AFTER
   the environment — never first); silhouette-level cues when BACKGROUND; nothing when NONE
5. Include the exact camera move and lighting as specified for each scene
6. Append these quality boosters at the end of every prompt:
   {_quality_boosters(style_dna)}
7. Each prompt: under 150 words, single line, no line breaks
8. Every prompt MUST use a DIFFERENT shot size — no two consecutive prompts share the same framing

━━━ VIDEO PROMPTS — {n} required ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rules:
1. Describe ONLY motion and action — NEVER appearance (the image handles that)
2. Include the camera move with precise speed/direction (ft/s, degrees)
3. Include environmental motion: wind in fabric, sand shifting, crowd movement, etc.
4. Match the scene's Character marking: NONE → camera + environmental motion only;
   BACKGROUND → distant figure's broad movement; FEATURED → subject action in detail
5. Maximum 60 words each
6. End each prompt with exactly one pacing word: [slow] [medium] [fast] [explosive]

RESPOND WITH ONLY THIS JSON — no text before or after:

{schema}"""


# ── Stage 5 — Single Scene Re-prompt ─────────────────────────────────────────

def build_single_scene_prompt(
    idea: str,
    scene,              # SceneState
    character,          # Character | None
    style_dna,
) -> str:
    """Generate a copy-paste prompt to regenerate one specific scene's prompts."""
    char_line   = f'  Character: "{character.description}"' if character else ""
    skill_notes = _get_skill_notes(style_dna)
    presence    = (getattr(scene, "character_presence", "") or "featured").upper()
    presence_block = f"\n{_PRESENCE_RULES}\n" if character else ""
    presence_part  = f" | Character: {presence}" if character else ""

    schema = f'''{{"stage": "scene_prompts",
  "result": {{
    "visual_prompts": ["visual prompt for scene {scene.scene_number}"],
    "video_prompts":  ["motion prompt for scene {scene.scene_number}"]
  }}
}}'''

    return f"""{_header(f"STAGE 5: SINGLE SCENE REPROMPT — SCENE {scene.scene_number}")}

You are both a cinematographer (visual prompt) and an I2V director (motion prompt).

CINEMATOGRAPHER GUIDELINES:
{skill_notes}

PROJECT CONTEXT:
  Idea:    "{idea}"
{char_line}
  Motion style: "{style_dna.motion_style if style_dna else ""}"

{presence_block}
SCENE TO REPROMPT:
  Scene {scene.scene_number} | Act: {scene.act} | Shot: {getattr(scene, 'shot_size', '') or 'MEDIUM SHOT'}{presence_part}
  Description: "{scene.description}"
  Camera:      "{scene.camera}"
  Lighting:    "{scene.lighting}"

VISUAL PROMPT RULES:
1. BEGIN with the SHOT SIZE exactly as listed above (e.g. "WIDE SHOT.", "CLOSE-UP.")
2. If the character is in the shot (FEATURED/BACKGROUND), follow with their position in
   frame; if Character: NONE, open with an environment composition note — no people
3. Describe the ENVIRONMENT and scene action next
4. Apply the Character marking: full description AFTER the environment only when FEATURED;
   silhouette-level cues when BACKGROUND; no character at all when NONE
5. Include the exact camera move and lighting
6. Append: {_quality_boosters(style_dna)}
7. Under 150 words, single line

VIDEO PROMPT RULES:
1. Motion and action only — no appearance
2. Camera move with speed/direction
3. Environmental motion
4. Respect the Character marking: NONE → camera + environment motion only
5. Max 60 words
6. End with: [slow] [medium] [fast] [explosive]

RESPOND WITH ONLY THIS JSON — no text before or after:

{schema}"""


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_skill_notes(style_dna) -> str:
    """Return the skill's prompt_template.

    Tries skills_engine first (full OpenMontage brief with technical camera/
    lighting language).  Falls back to reconstructing the key rules from the
    style_dna fields so the LLM still gets meaningful direction even when
    running from a detached worker context where skills_engine isn't importable.
    """
    if style_dna is None:
        return "Use professional cinematographic language."
    try:
        import sys, pathlib
        # Ensure project root is on path so skills_engine is importable from
        # any working directory (MCP server, CLI worker, tests, etc.)
        _root = str(pathlib.Path(__file__).parent.parent)
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from skills_engine import SKILLS
        skill = SKILLS.get(style_dna.skill_id)
        if skill and skill.prompt_template:
            return skill.prompt_template.strip()
    except Exception:
        pass
    # Fallback: reconstruct a useful brief from style_dna fields
    cam_examples = "\n".join(f"  - {c}" for c in style_dna.camera_language[:3]) if style_dna.camera_language else ""
    return (
        f"Visual style: {style_dna.visual_style}\n"
        f"Motion style: {style_dna.motion_style}\n"
        f"Lighting: {style_dna.lighting_style}\n"
        + (f"Camera vocabulary examples:\n{cam_examples}" if cam_examples else "")
    )


def _media_section(media_paths: list | None) -> str:
    """Return a REFERENCE MEDIA block for the prompt when paths are provided.

    In MCP Auto mode the worker uses its Read tool to view the images.
    In Copy-Paste mode the UI shows the images so the user can attach them
    directly to the chatbot message — this section tells the AI to expect them.
    """
    if not media_paths:
        return ""
    lines = [
        "",
        "REFERENCE MEDIA:",
        "  The following reference images are attached to this message.",
        "  Study them carefully before responding — they take priority over any",
        "  default assumptions you might make about style, character, or mood.",
    ]
    for p in media_paths:
        # Show only the filename so the prompt stays readable in both contexts
        from pathlib import Path as _Path
        lines.append(f"  - {_Path(p).name}  [path: {p}]")
    return "\n".join(lines)


def _quality_boosters(style_dna) -> str:
    """Return comma-joined quality booster tags for the skill."""
    if style_dna is None:
        return "masterpiece, best quality, 8k uhd"
    try:
        import sys, pathlib
        _root = str(pathlib.Path(__file__).parent.parent)
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from skills_engine import SKILLS
        skill = SKILLS.get(style_dna.skill_id)
        if skill:
            boosters = skill.quality_boosters[:6] + skill.style_tags[:2]
            return ", ".join(boosters)
    except Exception:
        pass
    return ", ".join(style_dna.quality_boosters[:6])
