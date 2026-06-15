"""Single source of truth for the UI visual language across wizard steps.

Provides:
  - FOCUS_META / ROLE_META / ACT_COLOR  (emoji + plain-English label + color)
  - focus_chip / role_chip / hero_badge / focus_label / role_label  (HTML helpers)

Every Streamlit step that names a scene reuses these helpers — keeps emoji and
colour usage consistent so users learn the visual vocabulary once.

No Streamlit / pandas imports here — these helpers run anywhere, including in
tests, and return plain HTML strings the caller renders with
``st.markdown(..., unsafe_allow_html=True)`` (or ``st.html``).
"""

from __future__ import annotations

import html


# ── Focus types ───────────────────────────────────────────────────────────────
# Keys match pipeline.scene_state.VALID_FOCUS.
# Tuple: (emoji, plain-English label, hex color, one-line help/tooltip)
FOCUS_META: dict[str, tuple[str, str, str, str]] = {
    "subject":      ("🧑", "About the protagonist", "#ff8a65",
                     "Character description is injected. The hero of the shot."),
    "establishing": ("🌅", "About the place",       "#4fc3f7",
                     "World/location — no people dominate the frame."),
    "secondary":    ("👥", "Another figure",         "#ba68c8",
                     "Non-protagonist person or crowd is the subject."),
    "object":       ("📦", "An object",              "#ffb74d",
                     "Prop/product/vehicle/structure is isolated."),
    "detail":       ("🔍", "Close-up detail",        "#aed581",
                     "Macro insert — surface, texture, material."),
    "phenomenon":   ("🌪", "Action or atmosphere",   "#7986cb",
                     "Fire, water, wind, light — motion is the subject."),
    "reaction":     ("👁", "Reaction beat",          "#f06292",
                     "Hands, eyes, silhouette — the moment, not identity."),
}


# ── Narrative roles ───────────────────────────────────────────────────────────
# Keys match pipeline.scene_state.VALID_NARRATIVE_ROLE.
# Tuple: (emoji, plain-English label, hex color)
ROLE_META: dict[str, tuple[str, str, str]] = {
    "establish_context": ("📍", "Set the scene",       "#64748b"),
    "introduce_subject": ("🚪", "Meet the hero",       "#0ea5e9"),
    "build_tension":     ("⏳", "Build tension",       "#f59e0b"),
    "deliver_payload":   ("🎯", "The payoff",          "#dc2626"),
    "transition":        ("🔗", "Connect beats",       "#a3a3a3"),
    "emotional_beat":    ("💗", "Emotional beat",      "#ec4899"),
    "evidence":          ("🔬", "Show evidence",       "#10b981"),
    "comparison":        ("⚖",  "Compare / contrast",  "#8b5cf6"),
    "resolution":        ("🕊", "Wrap up",             "#22c55e"),
    "call_to_action":    ("📣", "Call to action",      "#f97316"),
}


# Color hint per well-known act label (drives the # number color + act pill).
ACT_COLOR: dict[str, str] = {
    "HOOK":       "#f59e0b",
    "BEFORE":     "#64748b",
    "ORDINARY":   "#94a3b8",
    "INCITING":   "#f97316",
    "SETUP":      "#0ea5e9",
    "BUILD":      "#3b82f6",
    "BUILD1":     "#3b82f6",
    "BUILD2":     "#3b82f6",
    "BUILD3":     "#3b82f6",
    "MIDPOINT":   "#8b5cf6",
    "CHALLENGE":  "#a855f7",
    "CRISIS":     "#b91c1c",
    "CONFRONTATION": "#dc2626",
    "REVELATION": "#fbbf24",
    "DISCOVERY":  "#fbbf24",
    "WONDER":     "#fbbf24",
    "DECISION":   "#dc2626",
    "TWIST":      "#a78bfa",
    "CLIMAX":     "#ef4444",
    "VICTORY":    "#22c55e",
    "RESOLUTION": "#22c55e",
    "AFTER":      "#22c55e",
    "REBORN":     "#22c55e",
    "CODA":       "#a78bfa",
    "SACRIFICE":  "#9333ea",
    "RECKONING":  "#7c3aed",
}


# ── HTML helpers (all return safe strings; user content is escaped) ───────────

HERO_BADGE_HTML = (
    "<span style='background:linear-gradient(90deg,#fbbf24,#f59e0b);"
    "color:#422006;font-weight:700;font-size:.68rem;padding:2px 8px;"
    "border-radius:10px;letter-spacing:.06em;margin-left:6px;"
    "box-shadow:0 0 6px rgba(251,191,36,.45)'>★ HERO</span>"
)

OFFLINE_BADGE_HTML = (
    "<div style='background:#10b9811a;border:1px solid #10b98155;"
    "color:#34d399;font-size:.7rem;font-weight:700;letter-spacing:.05em;"
    "padding:3px 10px;border-radius:11px;text-align:center;margin-bottom:8px'>"
    "🔒 100% LOCAL · NO API REQUIRED</div>"
)


def focus_chip(focus: str) -> str:
    """Return an inline-styled HTML chip for a focus type."""
    emo, label, color, helptext = FOCUS_META.get(focus, FOCUS_META["subject"])
    return (
        f"<span title='{html.escape(helptext)}' "
        f"style='background:{color}22;border:1px solid {color}66;"
        f"color:{color};font-size:.75rem;padding:2px 9px;border-radius:11px;"
        f"margin-right:4px;display:inline-block'>{emo} {html.escape(label)}</span>"
    )


def role_chip(role: str) -> str:
    """Return an inline-styled HTML chip for a narrative role (or empty)."""
    meta = ROLE_META.get(role)
    if not meta:
        return ""
    emo, label, color = meta
    return (
        f"<span style='background:{color}22;border:1px solid {color}66;"
        f"color:{color};font-size:.75rem;padding:2px 9px;border-radius:11px;"
        f"margin-right:4px;display:inline-block'>{emo} {html.escape(label)}</span>"
    )


def act_pill(act: str) -> str:
    """Small monospace pill for the act label, coloured by ACT_COLOR."""
    key = (act or "").upper().strip()
    color = ACT_COLOR.get(key, "#9ca3af")
    return (
        f"<span style='background:{color}22;color:{color};"
        f"border:1px solid {color}66;font-size:.7rem;font-family:monospace;"
        f"padding:1px 7px;border-radius:5px;font-weight:600;"
        f"margin-right:6px;display:inline-block'>{html.escape(act or '')}</span>"
    )


def hero_badge(is_hero: bool) -> str:
    return HERO_BADGE_HTML if is_hero else ""


def focus_label(focus: str) -> str:
    """Plain-text 'emoji label' for selectbox format_func and dataframes."""
    emo, label, _, _ = FOCUS_META.get(focus, FOCUS_META["subject"])
    return f"{emo} {label}"


def role_label(role: str) -> str:
    """Plain-text 'emoji label' for narrative role (or '— none —')."""
    meta = ROLE_META.get(role)
    return f"{meta[0]} {meta[1]}" if meta else "— none —"


def focus_color(focus: str) -> str:
    """Hex color for a focus type — used by Altair / inline borders."""
    return FOCUS_META.get(focus, FOCUS_META["subject"])[2]
