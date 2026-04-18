"""
StyleDNA and Character data classes, plus infer_style() which wraps detect_skill().

infer_style(idea, override_skill_id=None) → StyleDNA
  Detects the best skill for the given idea string, then maps SkillSpec fields
  to StyleDNA fields. Returns a fully populated StyleDNA.

The StyleDNA is used throughout the pipeline to:
  - Set default resolution, fps, and quality boosters
  - Drive per-scene prompt generation (camera/lighting vocabulary)
  - Build the negative prompt
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

from skills_engine import detect_skill, SkillSpec, SKILLS


# ── Color palette extraction ──────────────────────────────────────────────────
# Each skill has characteristic color keywords. We maintain a lightweight
# mapping so StyleDNA.color_palette is always descriptive and useful.

_SKILL_PALETTES: dict[str, list[str]] = {
    "cinematic":    ["deep black", "warm amber", "desaturated steel blue", "film grain ivory"],
    "3d_cgi":       ["metallic silver", "HDRI white", "volumetric fog blue", "neon highlight"],
    "cartoon":      ["primary red", "sunshine yellow", "sky blue", "clean white"],
    "anime":        ["vibrant cyan", "warm gold", "sakura pink", "midnight navy"],
    "fight_scenes": ["blood orange", "deep shadow black", "electric blue", "dust grey"],
    "motion_design":["electric blue #0066ff", "deep navy #0a0e27", "cyan #00ffff", "white #ffffff"],
    "ecommerce":    ["product white", "lifestyle warm beige", "brand accent", "soft shadow grey"],
    "social_hook":  ["high-contrast white", "neon accent", "black background", "bold colour pop"],
    "music_video":  ["stage black", "concert RGB", "neon cyan", "warm amber spotlight"],
    "brand_story":  ["natural ochre", "warm cream", "honest grey", "golden hour amber"],
    "fashion":      ["editorial white", "luxury black", "accent colour", "natural skin"],
    "food":         ["warm amber 2800K", "natural green", "rich brown", "steam white"],
    "real_estate":  ["golden hour amber", "crisp white", "sky blue", "warm interior glow"],
}

# Default motion style descriptor per skill
_SKILL_MOTION: dict[str, str] = {
    "cinematic":    "smooth, intentional camera work with dramatic reveals",
    "3d_cgi":       "orbital and fly-through motion with precise mechanical precision",
    "cartoon":      "exaggerated squash-and-stretch with comedic timing",
    "anime":        "impact frames, speed lines, and emotionally-driven timing",
    "fight_scenes": "fast, impact-focused with precise choreographic beats",
    "motion_design":"eased UI animations with particle effects and smooth transitions",
    "ecommerce":    "hero product orbit and lifestyle tracking shots",
    "social_hook":  "rapid cuts, smash zooms, and scroll-stopping pattern interrupts",
    "music_video":  "beat-synchronized cuts with genre-specific energy",
    "brand_story":  "intimate, observational, slow-push handheld",
    "fashion":      "editorial power walk and fabric-motion reveals",
    "food":         "macro slow-motion with steam and texture reveals",
    "real_estate":  "purposeful steadicam walk-through with light reveals",
}


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class StyleDNA:
    """Inferred style descriptor that drives all prompt generation."""

    skill_id:          str
    skill_name:        str
    visual_style:      str               # prose description of the visual aesthetic
    color_palette:     list[str]         # characteristic color descriptors
    lighting_style:    str               # first entry from skill.lighting_vocabulary
    camera_language:   list[str]         # first 3 entries from skill.camera_vocabulary
    motion_style:      str               # brief descriptor of how things move
    quality_boosters:  list[str]         # from skill.quality_boosters
    negative_tags:     list[str]         # from skill.negative_tags
    fps:               int
    recommended_width: int
    recommended_height:int

    def to_dict(self) -> dict:
        return {
            "skill_id":           self.skill_id,
            "skill_name":         self.skill_name,
            "visual_style":       self.visual_style,
            "color_palette":      list(self.color_palette),
            "lighting_style":     self.lighting_style,
            "camera_language":    list(self.camera_language),
            "motion_style":       self.motion_style,
            "quality_boosters":   list(self.quality_boosters),
            "negative_tags":      list(self.negative_tags),
            "fps":                self.fps,
            "recommended_width":  self.recommended_width,
            "recommended_height": self.recommended_height,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StyleDNA":
        return cls(
            skill_id          = data["skill_id"],
            skill_name        = data["skill_name"],
            visual_style      = data["visual_style"],
            color_palette     = list(data.get("color_palette", [])),
            lighting_style    = data.get("lighting_style", ""),
            camera_language   = list(data.get("camera_language", [])),
            motion_style      = data.get("motion_style", ""),
            quality_boosters  = list(data.get("quality_boosters", [])),
            negative_tags     = list(data.get("negative_tags", [])),
            fps               = data.get("fps", 24),
            recommended_width = data.get("recommended_width", 768),
            recommended_height= data.get("recommended_height", 512),
        )

    def __repr__(self) -> str:
        return f"StyleDNA(skill={self.skill_id!r} {self.recommended_width}×{self.recommended_height}@{self.fps}fps)"


@dataclass
class Character:
    """Protagonist / subject locked at step 3.5 for visual consistency."""

    id:                    str         # uuid
    description:           str         # full visual description used in every scene prompt
    reference_image_path:  Optional[str] = None   # local path (v2: IPAdapter)
    base_seed:             int = 0

    @classmethod
    def new(cls, description: str, base_seed: int = 0) -> "Character":
        return cls(id=str(uuid.uuid4()), description=description, base_seed=base_seed)

    def to_dict(self) -> dict:
        return {
            "id":                   self.id,
            "description":          self.description,
            "reference_image_path": self.reference_image_path,
            "base_seed":            self.base_seed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Character":
        return cls(
            id                   = data["id"],
            description          = data["description"],
            reference_image_path = data.get("reference_image_path"),
            base_seed            = data.get("base_seed", 0),
        )

    def __repr__(self) -> str:
        snippet = self.description[:60] + "…" if len(self.description) > 60 else self.description
        return f"Character({snippet!r})"


# ── infer_style() ─────────────────────────────────────────────────────────────

def infer_style(idea: str, override_skill_id: str | None = None) -> StyleDNA:
    """Detect the best skill for *idea* and build a StyleDNA from it.

    Args:
        idea:              The user's idea / concept string.
        override_skill_id: If provided, bypass auto-detection and use this skill.

    Returns:
        A fully populated StyleDNA.
    """
    skill: SkillSpec = detect_skill(idea, override=override_skill_id)
    return _skill_to_dna(skill)


def infer_style_from_skill_id(skill_id: str) -> StyleDNA:
    """Build a StyleDNA directly from a known skill ID."""
    if skill_id not in SKILLS:
        raise ValueError(f"Unknown skill_id '{skill_id}'. Available: {list(SKILLS.keys())}")
    return _skill_to_dna(SKILLS[skill_id])


def _skill_to_dna(skill: SkillSpec) -> StyleDNA:
    """Map a SkillSpec to a StyleDNA."""
    specs = skill.technical_specs
    return StyleDNA(
        skill_id          = skill.id,
        skill_name        = skill.name,
        visual_style      = skill.description,
        color_palette     = _SKILL_PALETTES.get(skill.id, skill.style_tags[:4]),
        lighting_style    = skill.lighting_vocabulary[0] if skill.lighting_vocabulary else "",
        camera_language   = list(skill.camera_vocabulary[:3]),
        motion_style      = _SKILL_MOTION.get(skill.id, skill.description),
        quality_boosters  = list(skill.quality_boosters),
        negative_tags     = list(skill.negative_tags),
        fps               = specs.get("fps", 24),
        recommended_width = specs.get("width", 768),
        recommended_height= specs.get("height", 512),
    )
