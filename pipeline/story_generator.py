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


# ── Per-scene FOCUS (area of focus) — the fix for character-centric output ─────
#
# Each scene gets an explicit FOCUS: what the shot is *about*. Only "subject"
# foregrounds the protagonist; the rest make the shot about a place, figure,
# prop, texture or event. character_presence is then a *projection* of focus,
# so all the existing presence-gated prompt code keeps working unchanged.

# Act → narrative-intent focus (what each story beat is "about")
_FOCUS_BY_ACT: dict[str, str] = {
    # establishing / world
    "HOOK": "establishing", "BEFORE": "establishing", "ORDINARY": "establishing",
    "ORDINARY1": "establishing", "ORDINARY2": "establishing", "SETUP": "establishing",
    "DEPARTURE": "establishing", "JOURNEY": "establishing",
    "RESOLUTION": "establishing", "AFTER": "establishing", "CODA": "establishing",
    "VICTORY": "establishing", "REBORN": "establishing",
    "WONDER": "establishing", "FIRST LIGHT": "establishing", "FIRST SIGHT": "establishing",
    # object beats ("one wrong element enters", "an object falls")
    "INCITING": "object", "CATALYST": "object", "CHALLENGE": "object",
    # motion / force / transformation
    "BUILD": "phenomenon", "BUILD1": "phenomenon", "BUILD2": "phenomenon", "BUILD3": "phenomenon",
    "MIDPOINT": "phenomenon", "TEST": "phenomenon", "TEST1": "phenomenon", "TEST2": "phenomenon",
    "CHANGE": "phenomenon", "CHANGE1": "phenomenon", "CHANGE2": "phenomenon", "SETBACK": "phenomenon",
    # slow reveal of a thing
    "DISCOVERY": "detail", "REVELATION": "detail", "INSCRIPTION": "detail", "TWIST": "detail",
    # opposing force / figure
    "CONFRONTATION": "secondary", "REGROUPING": "secondary",
    # protagonist's decisive emotional beats — the only character-led acts
    "DOUBT": "subject", "DECISION": "subject", "CRISIS": "subject", "CLIMAX": "subject",
    # close reaction beats
    "SACRIFICE": "reaction", "RECKONING": "reaction",
}

# Fallback when act is unknown — derive focus from shot size
_FOCUS_BY_SHOT: dict[str, str] = {
    "EXTREME WIDE SHOT": "establishing",
    "WIDE SHOT":         "establishing",
    "MEDIUM WIDE SHOT":  "secondary",
    "MEDIUM SHOT":       "subject",
    "MEDIUM CLOSE-UP":   "reaction",
    "CLOSE-UP":          "detail",
}

# Ordered non-character fallback used by the diversity sweep
_FOCUS_FALLBACK_ORDER = ["establishing", "object", "detail", "phenomenon"]

# Acts whose "subject" focus is narratively essential — never demoted
_PROTECTED_SUBJECT_ACTS = {"CLIMAX", "DECISION", "CRISIS"}

# Act → narrative_role (the JOB this beat does in the arc). Vocabulary mirrors
# OpenMontage's scene_plan schema (concepts only, no code copied).
_NARRATIVE_ROLE_BY_ACT: dict[str, str] = {
    "HOOK": "establish_context", "BEFORE": "establish_context",
    "ORDINARY": "establish_context", "ORDINARY1": "establish_context",
    "ORDINARY2": "establish_context", "SETUP": "establish_context",
    "DEPARTURE": "establish_context", "JOURNEY": "establish_context",
    "INCITING": "introduce_subject", "CATALYST": "introduce_subject",
    "BUILD": "build_tension", "BUILD1": "build_tension",
    "BUILD2": "build_tension", "BUILD3": "build_tension",
    "MIDPOINT": "build_tension", "TEST": "build_tension",
    "TEST1": "build_tension", "TEST2": "build_tension",
    "CHALLENGE": "build_tension", "DOUBT": "build_tension",
    "CHANGE": "build_tension", "CHANGE1": "build_tension",
    "CHANGE2": "build_tension", "SETBACK": "build_tension",
    "REGROUPING": "transition",
    "DISCOVERY": "deliver_payload", "REVELATION": "deliver_payload",
    "INSCRIPTION": "deliver_payload", "TWIST": "deliver_payload",
    "FIRST LIGHT": "deliver_payload", "FIRST SIGHT": "deliver_payload",
    "WONDER": "emotional_beat", "RECKONING": "emotional_beat",
    "SACRIFICE": "emotional_beat",
    "CONFRONTATION": "build_tension",
    "CRISIS": "emotional_beat",
    "DECISION": "deliver_payload",
    "CLIMAX": "deliver_payload",
    "VICTORY": "resolution", "RESOLUTION": "resolution",
    "AFTER": "resolution", "REBORN": "resolution", "CODA": "resolution",
}

# Acts that are visual peaks — get hero_moment=True so the compositor holds
# them longer and the AI prompts get extra craft attention.
_HERO_ACTS = frozenset({
    "REVELATION", "DISCOVERY", "CLIMAX", "FIRST SIGHT", "FIRST LIGHT",
    "VICTORY", "TWIST", "WONDER", "REBORN",
})


def _assign_narrative_role(act: str) -> str:
    """Map an act label to OpenMontage-style narrative_role; "" if unknown."""
    return _NARRATIVE_ROLE_BY_ACT.get(act.upper().strip(), "")


def _assign_focus(act: str, shot_size: str, scene_idx: int, total: int) -> str:
    """Raw per-scene focus from the act label (or shot size when act unknown)."""
    act_key = act.upper().strip()
    if act_key in _FOCUS_BY_ACT:
        return _FOCUS_BY_ACT[act_key]
    return _FOCUS_BY_SHOT.get(shot_size, "establishing")


# Acts where featuring the protagonist reads naturally — used to pick which
# scenes to promote to "subject" when a character-led story under-features them.
_CHARACTER_FRIENDLY_ACTS = (
    "CLIMAX", "DECISION", "CRISIS", "CONFRONTATION", "DOUBT", "RECKONING",
    "SACRIFICE", "REVELATION", "DISCOVERY", "VICTORY", "MIDPOINT",
)
_CHARACTER_FRIENDLY_SHOTS = ("MEDIUM SHOT", "MEDIUM CLOSE-UP", "CLOSE-UP", "MEDIUM WIDE SHOT")


def _enforce_focus_variety(focuses: list[str], shots: list[str], acts: list[str],
                           has_character: bool = True) -> list[str]:
    """Balance 'subject' (character) scenes into a healthy minority and ensure
    focus diversity. Deterministic and index-ordered (no RNG) for reproducibility.

    Guarantees, for a story with a locked character:
      - subject scenes are a MINORITY:  count <= round(n*0.45)
      - but never ZERO when it's a character story: count >= max(1, round(n*0.30))
      - >= 3 distinct focus types when n >= 5
    With no character, the subject floor is 0 (pure subject-driven coverage).
    """
    out = list(focuses)
    n = len(out)
    if n == 0:
        return out

    cap   = max(1, round(n * 0.45))
    floor = max(1, round(n * 0.30)) if has_character else 0

    # 1. Cap surplus subject scenes (demote latest, unprotected acts first)
    subject_idxs = [i for i, f in enumerate(out) if f == "subject"]
    if len(subject_idxs) > cap:
        demotable = [i for i in subject_idxs
                     if acts[i].upper().strip() not in _PROTECTED_SUBJECT_ACTS]
        n_demote = len(subject_idxs) - cap
        for i in reversed(demotable):
            if n_demote <= 0:
                break
            alt = _FOCUS_BY_SHOT.get(shots[i], "establishing")
            if alt in ("subject", "reaction"):     # never demote to a character type
                alt = "establishing"
            out[i] = alt
            n_demote -= 1

    # 2. Subject FLOOR — promote the most character-appropriate scenes so a
    #    character story is never reduced to zero character shots.
    n_subject = sum(1 for f in out if f == "subject")
    if n_subject < floor:
        # rank non-subject scenes by how naturally they feature the protagonist
        def _promo_rank(i: int) -> tuple:
            act_ok  = acts[i].upper().strip() in _CHARACTER_FRIENDLY_ACTS
            shot_ok = shots[i] in _CHARACTER_FRIENDLY_SHOTS
            return (not act_ok, not shot_ok, i)   # True sorts last → prefer act_ok then shot_ok
        candidates = sorted((i for i, f in enumerate(out) if f != "subject"), key=_promo_rank)
        for i in candidates:
            if n_subject >= floor:
                break
            out[i] = "subject"
            n_subject += 1

    # 3. Diversity floor: >= 3 distinct focus types when total >= 5
    if n >= 5 and len(set(out)) < 3:
        needed = [f for f in _FOCUS_FALLBACK_ORDER if f not in set(out)]
        slots = [i for i in range(n)
                 if acts[i].upper().strip() not in _PROTECTED_SUBJECT_ACTS
                 and out[i] != "subject"]
        if slots and needed:
            step = max(1, len(slots) // max(len(needed), 1))
            for k, f in enumerate(needed):
                out[slots[min(k * step, len(slots) - 1)]] = f

    return out


# Idea-string tokens that make focus character-centric or are noise
_STOPWORDS = frozenset({
    "a", "an", "the", "of", "in", "on", "with", "and", "to", "that", "who", "as", "at",
    "by", "for", "from", "into", "lone", "single", "his", "her", "its", "their", "they",
    "discovers", "finds", "repairs", "walks", "runs", "becomes", "learns", "fights",
    "builds", "awakens", "explores", "video", "story", "about", "is", "are", "was", "while",
    "lifting", "doing", "through", "over", "across",
})

# Words denoting a PERSON — excluded from non-subject focus subjects so an
# environment/object/detail shot never names "the man" as its subject.
_PROTAGONIST_WORDS = frozenset({
    "man", "woman", "men", "women", "person", "people", "boy", "girl", "guy",
    "hero", "heroine", "ranger", "explorer", "astronaut", "detective", "warrior",
    "king", "queen", "soldier", "child", "kid", "figure", "protagonist", "rider",
    "dancer", "player", "worker", "miner", "scientist", "pilot", "knight", "monk",
})

# Domain keyword → bucket; first match in the idea wins
_DOMAIN_NOUNS: dict[str, str] = {
    "robot": "machine", "android": "machine", "machine": "machine", "drone": "machine",
    "engine": "machine", "mining": "machine", "factory": "machine",
    "ocean": "water", "sea": "water", "wave": "water", "river": "water",
    "rain": "water", "waterfall": "water", "storm": "water",
    "city": "urban", "street": "urban", "neon": "urban", "skyline": "urban",
    "forest": "nature", "desert": "nature", "mountain": "nature", "field": "nature", "cave": "nature",
    "gallery": "art", "painting": "art", "art": "art", "sculpture": "art",
    "temple": "ruins", "ruins": "ruins", "ancient": "ruins", "shiva": "ruins",
    "dragon": "fantasy", "magic": "fantasy",
}

# Per-focus noun-phrase banks. "{idea_noun}", "{place}", "{thing}" are filled.
_SUBJECT_BANKS: dict[str, dict[str, list[str]]] = {
    "establishing": {
        "generic":  ["the wide landscape of the {place}",
                     "the empty space around the {place} before anything moves",
                     "the {place} at rest, seen in full"],
        "urban":    ["the rain-slick city street where {idea_noun} unfolds",
                     "the neon skyline above the {place}"],
        "nature":   ["the vast {place} stretching to the horizon",
                     "the wild expanse surrounding {idea_noun}"],
        "ruins":    ["the silent ancient {place}", "the weathered ruins where {idea_noun} unfolds"],
        "water":    ["the open {place} meeting the sky", "the churning expanse of {place}"],
    },
    "object": {
        "generic":  ["a single significant object within the {place}",
                     "a lone artifact resting in the {place}"],
        "machine":  ["the brushed-steel chassis at the heart of the {place}",
                     "a single intricate mechanism, isolated in frame"],
        "art":      ["a lone canvas central to the scene", "a single framed artwork on the wall"],
        "ruins":    ["a weathered carved relic from the {place}", "an ancient stone artifact half-buried"],
    },
    "detail": {
        "generic":  ["an extreme macro texture from the world of {idea_noun}",
                     "the worn surface of the {thing}"],
        "water":    ["droplets beading on a cold surface", "the rippling skin of the {place}"],
        "machine":  ["oil glinting on machined metal", "the fine grain of brushed steel"],
        "ruins":    ["chiselled detail in ancient stone", "moss creeping across carved {place}"],
        "nature":   ["dew on a single leaf", "grains of {place} shifting in the wind"],
    },
    "phenomenon": {
        "generic":  ["light and dust moving through the {place}",
                     "a surge of motion sweeping across the {place}"],
        "water":    ["a cresting wave breaking across the frame", "water surging over the {place}"],
        "machine":  ["sparks and steam venting from the {thing}", "machinery roaring into motion"],
        "nature":   ["wind tearing across the {place}", "a dust storm rolling over the {place}"],
        "fantasy":  ["arcs of energy crackling across the frame", "fire sweeping through the {place}"],
    },
    "secondary": {
        "generic":  ["a second figure inside the {place}, face turned away",
                     "the crowd moving through the {place}"],
    },
    "reaction": {
        "generic":  ["hands acting decisively within the scene",
                     "eyes catching the change in the {place}"],
    },
    "subject": {"generic": [""]},   # subject focus => the character IS the subject
}


def _idea_tokens(idea: str) -> list[str]:
    import re
    toks = re.findall(r"[a-z0-9]+", (idea or "").lower())
    return [t for t in toks if t not in _STOPWORDS and len(t) >= 3]


def _domain_key(tokens: list[str]) -> str:
    for t in tokens:
        if t in _DOMAIN_NOUNS:
            return _DOMAIN_NOUNS[t]
    return "generic"


def _derive_focus_subject(focus: str, idea: str, scene_idx: int, seed: int) -> str:
    """Deterministically pick a concrete frameable subject noun (no LLM).

    Person words are excluded so a non-character shot never names the protagonist
    as its subject; domain (place/thing) nouns are preferred for readability.
    """
    if focus == "subject":
        return ""
    import re
    tokens  = _idea_tokens(idea)
    # Proper nouns (capitalised in the original idea) are likely names/brands —
    # avoid using them as a generic place/thing ("the cristiano"). Domain nouns
    # still win below, so "dragon"/"waterfall" are unaffected.
    proper  = {w.lower() for w in re.findall(r"\b([A-Z][a-zA-Z]+)", idea or "")}
    # Common-noun setting tokens: drop people AND proper nouns
    setting = [t for t in tokens if t not in _PROTAGONIST_WORDS and t not in proper]
    dkey    = _domain_key(tokens)
    banks   = _SUBJECT_BANKS.get(focus, _SUBJECT_BANKS["establishing"])
    bank    = banks.get(dkey) or banks["generic"]
    # Deterministic selection reusing _scene_seed + a STABLE focus offset.
    # NOTE: builtin hash() on str is randomized per-process (PYTHONHASHSEED), so
    # it must NOT be used here — use a hashlib digest so the same project always
    # regenerates identical subjects across separate runs.
    import hashlib
    foff = int(hashlib.sha256(focus.encode()).hexdigest(), 16) & 0xFFFF
    pick = (_scene_seed(seed, scene_idx + 1) + foff) % len(bank)
    tmpl = bank[pick]

    domain_tok = next((t for t in tokens if t in _DOMAIN_NOUNS and t not in proper), None)
    place = domain_tok or (setting[0] if setting else "surrounding environment")
    thing = domain_tok or (setting[0] if setting else "central object")
    idea_noun = " ".join(setting[:3]) if setting else "the scene"

    phrase = " ".join(tmpl.format(idea_noun=idea_noun, place=place, thing=thing).split()).strip(" .,")
    return phrase or "the surrounding environment"


def _derive_presence_from_focus(focus: str, shot_size: str) -> str:
    """Project focus → character_presence so existing gated code keeps working."""
    if focus == "subject":
        return "featured"
    if focus in ("secondary", "reaction"):
        return "featured" if shot_size in ("MEDIUM SHOT", "MEDIUM CLOSE-UP", "CLOSE-UP") else "background"
    return "none"   # establishing / object / detail / phenomenon


# Focus directives that lead the visual prompt for non-subject shots
_FOCUS_DIRECTIVE_BY_TYPE: dict[str, str] = {
    "establishing": "the subject of this shot is {subject}; no people in frame",
    "object":       "the subject of this shot is {subject}, isolated and emphasised, "
                    "occupying the compositional center, shallow depth of field, no people in frame",
    "detail":       "macro insert: the subject of this shot is {subject}, frame-filling "
                    "surface detail, razor-sharp focus on texture, no people in frame",
    "phenomenon":   "the subject of this shot is {subject} — motion and energy are the focus, "
                    "captured mid-movement, no people in frame",
    "secondary":    "the subject of this shot is {subject}",
    "reaction":     "the subject of this shot is {subject}; tight framing on the gesture, "
                    "identity secondary to the moment",
}


# ── Character presence assignment ─────────────────────────────────────────────
#
# Cinematic coverage means the protagonist does NOT appear in every shot.
# Character consistency = the character looks identical *whenever on screen*,
# not that every frame is a character shot.
#
# NOTE: _assign_character_presence is KEPT for backward-compat / external callers
# but the main path now derives presence from focus via _derive_presence_from_focus.

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
    focus: str = "subject",
    focus_subject: str = "",
    character_presence: str = "featured",
) -> str:
    """Build a mechanical visual prompt that leads with the scene's FOCUS subject.

    Structure: [SHOT SIZE]. [Placement]. [FOCUS DIRECTIVE naming the subject].
               [Scene content]. [Camera]. [Lighting]. [Character — only if in shot].
               [Quality boosters].

    For focus=="subject" the focus directive is empty and the character is the
    subject, so the prompt is byte-identical to the prior featured/character path.
    character_presence (a projection of focus, or a manual UI override) gates the
    character clause.
    """
    char_clean = character_desc.rstrip(", ")

    # Focus directive only for non-subject shots that have a concrete subject
    directive = ""
    if focus != "subject" and focus_subject:
        directive = _FOCUS_DIRECTIVE_BY_TYPE.get(focus, "").format(subject=focus_subject)

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
    directive_clause = f" {directive}." if directive else ""
    base = f"{shot_size}.{placement_clause}{directive_clause} {core}"
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
            focus    = s.get("focus", "subject").upper()
            focus_sub = s.get("focus_subject", "")
            return (
                f"{i+1}. Act: {s['act']} | Shot: {shot} | Focus: {focus} | Character: {presence}\n"
                f"   Focus subject: \"{focus_sub}\"\n"
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
            "SCENE FOCUS — every shot has ONE subject, given as 'Focus' + 'Focus subject':\n"
            "  SUBJECT      → the protagonist is the subject (apply the full character description)\n"
            "  ESTABLISHING → the location/world is the subject; no single person dominates\n"
            "  SECONDARY    → another figure/crowd is the subject\n"
            "  OBJECT       → a specific prop/artifact is the subject, isolated and centered\n"
            "  DETAIL       → a macro texture/surface insert is the subject\n"
            "  PHENOMENON   → an action/force/atmosphere (fire, water, light, dust) is the subject\n"
            "  REACTION     → a close gesture/eyes beat; the moment is the subject\n"
            "Only when Focus is SUBJECT may the protagonist be the main subject. For all other\n"
            "focus types, BUILD THE SHOT AROUND THE FOCUS SUBJECT, not the character.\n\n"
            "CRITICAL RULES — read every line:\n"
            "1. BEGIN each prompt with the SHOT SIZE (e.g. 'EXTREME WIDE SHOT.', 'CLOSE-UP.') — exactly as specified\n"
            "2. IMMEDIATELY follow with the PLACEMENT note for that shot\n"
            "3. Then NAME AND FRAME the scene's FOCUS SUBJECT as the dominant element\n"
            "4. Apply the CHARACTER PRESENCE marking — full description only when FEATURED; the\n"
            "   protagonist must NOT be the subject unless Focus is SUBJECT\n"
            "5. Include the exact camera move and lighting as specified\n"
            "6. Append quality-boosters and style tags at the very end\n"
            "7. Each prompt: under 150 words, single paragraph, no line breaks\n"
            "8. Every prompt MUST open with a DIFFERENT shot size — variety is mandatory\n"
            "9. Do NOT default to a centred person — only SUBJECT-focus scenes center the protagonist\n"
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
            f"{i+1}. Act: {s['act']} | Focus: {s.get('focus', 'subject').upper()} | "
            f"Character: {s.get('character_presence', 'featured').upper()} — {s['description']}\n"
            f"   Focus subject: \"{s.get('focus_subject', '')}\"\n"
            f"   Assigned camera: {s['camera']}"
            for i, s in enumerate(scenes)
        )
        subject_hint = character_desc[:80] if character_desc else idea

        user_msg = (
            f"Write {n} I2V motion prompts for these scenes.\n\n"
            f'Protagonist (only relevant on SUBJECT-focus scenes): "{subject_hint}"\n'
            f"Project motion style: {motion_style}\n\n"
            f"Camera vocabulary reference:\n{cam_examples}\n\n"
            f"Scenes:\n{scenes_text}\n\n"
            "Rules: motion only, under 60 words each, precise language.\n"
            "The thing that MOVES is the scene's Focus subject:\n"
            "  SUBJECT                → describe the protagonist's movement + camera + environment motion\n"
            "  ESTABLISHING/OBJECT/DETAIL/PHENOMENON → no person; describe the focus subject's motion\n"
            "                           (or the place/object/event) + camera + environmental motion\n"
            "  SECONDARY/REACTION     → describe that figure's/gesture's movement + camera\n"
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

    # ── Pass 1: shot sizes + raw focus, then the sequence-level variety guard ──
    # The variety guard caps "subject" (character) scenes at ~45% and guarantees
    # focus diversity, so the output can never be all character-focused.
    shots   = [_assign_shot_size(act_labels[i], i, n) for i in range(n)]
    raw_foc = [_assign_focus(act_labels[i], shots[i], i, n) for i in range(n)]
    focuses = _enforce_focus_variety(raw_foc, shots, act_labels,
                                     has_character=bool(character_desc))

    # ── Pass 2: build per-scene structure + mechanical prompts (always works) ──
    scene_inputs: list[dict] = []
    scenes: list[SceneState] = []

    for i in range(n):
        act      = act_labels[i]
        desc     = scene_descs[i]
        cam_idx  = i % len(skill.camera_vocabulary)
        lite_idx = i % len(skill.lighting_vocabulary)
        cam      = skill.camera_vocabulary[cam_idx]
        lite     = skill.lighting_vocabulary[lite_idx]
        shot_size = shots[i]
        focus     = focuses[i]
        focus_subject = _derive_focus_subject(focus, idea, i, global_seed)
        # Presence is a projection of focus (UI can override later).
        presence  = _derive_presence_from_focus(focus, shot_size)

        # Mechanical visual prompt — leads with the FOCUS subject so each scene
        # is about its own area of focus, not always the character.
        visual_prompt   = _build_visual_prompt_with_framing(
            shot_size, desc, cam, lite, character_desc, skill,
            focus=focus, focus_subject=focus_subject, character_presence=presence,
        )
        negative_prompt = build_comfyui_negative(skill)
        # Video motion leads with the focus subject for non-character shots.
        if focus == "subject" and character_desc:
            video_base = f"{desc}, {character_desc}"
        elif focus_subject:
            video_base = f"{focus_subject}, {desc}"
        else:
            video_base = desc
        video_prompt    = build_comfyui_video_prompt(
            video_base, skill, cam, style_dna.motion_style
        )

        # Narrative role + hero marker — mechanically derived from the act.
        nrole       = _assign_narrative_role(act)
        is_hero     = act.upper().strip() in _HERO_ACTS
        # shot_intent: a one-liner explaining WHY this beat exists, used by AI
        # backends as creative guidance.
        if is_hero:
            shot_intent = f"Visual peak: {desc.lower().split(':')[-1].strip()[:80] or 'deliver the central moment'}."
        elif focus == "subject":
            shot_intent = "Foreground the protagonist's action and emotional state."
        elif focus == "establishing":
            shot_intent = f"Establish {focus_subject or 'the world'}; orient the viewer in space."
        elif focus == "object":
            shot_intent = f"Isolate {focus_subject or 'the central object'} so it reads as significant."
        elif focus == "detail":
            shot_intent = f"Macro reveal of {focus_subject or 'a key surface'} — texture as evidence."
        elif focus == "phenomenon":
            shot_intent = f"Capture {focus_subject or 'the unfolding motion'} at the moment of force."
        elif focus == "secondary":
            shot_intent = f"Introduce {focus_subject or 'a second figure'} for contrast or conflict."
        else:
            shot_intent = "Hold on the moment; let the gesture do the work."

        scene_inputs.append({
            "act": act, "description": desc, "camera": cam, "lighting": lite,
            "shot_size": shot_size, "character_presence": presence,
            "focus": focus, "focus_subject": focus_subject,
            "narrative_role": nrole, "shot_intent": shot_intent, "hero_moment": is_hero,
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
            focus           = focus,
            focus_subject   = focus_subject,
            narrative_role  = nrole,
            shot_intent     = shot_intent,
            hero_moment     = is_hero,
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
