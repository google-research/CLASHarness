# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single source of truth (SSoT) for VLM platform visual design archetypes.

Historically the visual-style vocabulary and per-style token presets were
duplicated across several modules and had begun to drift:

  * ``llm_client.py``            -- the ``visual_style`` prose enum + a per-style
                                    prose design guide inside ``compose_vlm_html``.
  * ``vlm_materializer.py``      -- a hardcoded ``suggested_visual_styles`` list
                                    used for MD5-seeded style selection.
  * ``scripts/generate_style_gallery.py`` -- a second, independent list of the
                                    same styles with their token presets.
  * ``templates/chart_template.html``     -- a JS ``switch`` over style names
                                    (no-LLM fallback render path).

This module is the canonical definition consumed by all of them. A "design
archetype" is a *whole, coherent design language* (a complete ``style_details``
token bundle + palette + domain fit + render guidance), not an axis of
independent style knobs. Sampling (see ``vlm_materializer``) picks an archetype
and then varies *within* the archetype's declared ranges, so a platform always
looks like one real product rather than a random mix of unrelated choices.

Design principle (VD option "b"): the LLM composes one coherent per-platform
design language from this library; it is *not* free per-axis combination and
*not* static template expansion.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Token-axis vocabularies (SSoT for the ``style_details`` enums).
# Keep these in sync with the ``compose_vlm_html`` / ``generate_vlm_layout_spec``
# contracts, which are generated from this module (see llm_client.py).
# ---------------------------------------------------------------------------

FONT_FAMILIES: tuple[str, ...] = (
    "monospace",
    "serif",
    "sans-serif-retro",
    "sans-serif-modern",
    "pixel",
)
FONT_SIZE_SCALES: tuple[str, ...] = ("compact", "coarse", "default")
BORDER_STYLES: tuple[str, ...] = (
    "beveled_outset",
    "beveled_inset",
    "solid_thick",
    "solid_thin",
    "none",
)
BORDER_RADII: tuple[str, ...] = ("none", "rounded-sm", "rounded-lg", "rounded-full")
SHADOW_TYPES: tuple[str, ...] = ("none", "hard-retro", "soft-modern", "neon-glow")
HEADER_STYLES: tuple[str, ...] = ("solid", "gradient", "classic-title-bar", "glossy")
DISPLAY_EFFECTS: tuple[str, ...] = ("none", "crt_scanlines", "noise_overlay")
DIVIDER_STYLES: tuple[str, ...] = ("dashed", "dotted", "solid", "double", "ascii")
BACKGROUND_TEXTURES: tuple[str, ...] = ("solid", "grid_lines", "dots")

# Ordered map of every ``style_details`` enum axis -> its allowed values.
# ``background_css_declaration`` (free string) and ``custom_css_rules`` (list)
# are intentionally not enumerable axes and are excluded here.
STYLE_DETAIL_AXES: dict[str, tuple[str, ...]] = {
    "font_family": FONT_FAMILIES,
    "font_size_scale": FONT_SIZE_SCALES,
    "border_style": BORDER_STYLES,
    "border_radius": BORDER_RADII,
    "shadow_type": SHADOW_TYPES,
    "header_style": HEADER_STYLES,
    "display_effect": DISPLAY_EFFECTS,
    "divider_style": DIVIDER_STYLES,
    "background_texture": BACKGROUND_TEXTURES,
}

# Platform-level vocabularies (SSoT for the layout-spec contract + sampling).
COLOR_THEMES: tuple[str, ...] = ("light", "dark")
PRIMARY_COLORS: tuple[str, ...] = (
    "blue",
    "emerald",
    "amber",
    "rose",
    "purple",
    "indigo",
    "cyan",
    "teal",
    "orange",
    "fuchsia",
    "pink",
    "violet",
)
UI_SHELL_TYPES: tuple[str, ...] = (
    "monitoring_dashboard",
    "ide",
    "cloud_console",
    "chat_app",
    "database_manager",
    "email_client",
    "calendar_portal",
    "general_web_page",
)


@dataclass(frozen=True)
class DesignArchetype:
    """A complete, coherent per-platform design language.

    Attributes:
      name: The canonical ``visual_style`` identifier.
      default_theme: Preferred ``color_theme`` when unconstrained.
      default_primary_color: Preferred ``primary_color`` when unconstrained.
      palette_hints: Primary colors that stay coherent with this archetype.
      domain_hints: ``ui_shell_type`` values this archetype realistically fits.
        Empty means "universal" (fits any shell).
      legacy: True for anachronistic/vintage looks that must only be used when a
        platform explicitly signals legacy/degraded hardware (see the realism
        conventions in ``generate_vlm_layout_spec``).
      token_preset: The default ``style_details`` enum-token bundle.
      custom_css_rules: Illustrative micro-styling directions (the LLM may
        elaborate; these seed the intent).
      aesthetic_descriptor: A compact 1-3 sentence summary of the look.
      render_guidance: The detailed rendering guidance injected into
        ``compose_vlm_html`` so the renderer reproduces the look faithfully.
      variation_ranges: Per-axis coherent alternatives that may be sampled
        *within* this archetype without breaking its identity. Axes absent here
        are considered signature axes and stay fixed to ``token_preset``.
      theme_hints: ``color_theme`` values that stay coherent with this
        archetype. Empty means "only the ``default_theme`` is coherent" (see
        ``allowed_themes``). Used by ``sample_platform_design`` so a dark-only
        look (e.g. ``terminal_tui``) is never sampled in a light theme.
    """

    name: str
    default_theme: str
    default_primary_color: str
    palette_hints: tuple[str, ...]
    domain_hints: tuple[str, ...]
    legacy: bool
    token_preset: dict[str, str]
    custom_css_rules: tuple[str, ...]
    aesthetic_descriptor: str
    render_guidance: str
    variation_ranges: dict[str, tuple[str, ...]] = field(default_factory=dict)
    theme_hints: tuple[str, ...] = ()

    def allowed_themes(self) -> tuple[str, ...]:
        """Color themes coherent with this archetype.

        Falls back to ``(default_theme,)`` when ``theme_hints`` is empty so
        every archetype always yields at least one coherent theme.
        """
        return self.theme_hints or (self.default_theme,)

    def default_style_details(self) -> dict[str, Any]:
        """Return a fresh ``style_details`` dict (token preset + css rule seeds).

        A copy is returned so callers can mutate it without corrupting the
        shared library record.
        """
        details: dict[str, Any] = dict(self.token_preset)
        details["custom_css_rules"] = list(self.custom_css_rules)
        return details

    def variable_axes(self) -> dict[str, tuple[str, ...]]:
        """Axes that may vary within-archetype, each with its allowed values."""
        return dict(self.variation_ranges)


# ---------------------------------------------------------------------------
# The archetype library. Order is significant: it matches the historical
# ``suggested_visual_styles`` order so pre-existing MD5-seeded selections remain
# reproducible until seeded sampling (WD2) supersedes them. New archetypes (WD4)
# are *appended* after the historical block -- never inserted -- so the canonical
# order stays stable.
# ---------------------------------------------------------------------------

_ARCHETYPES: tuple[DesignArchetype, ...] = (
    DesignArchetype(
        name="modern_clean",
        default_theme="light",
        default_primary_color="blue",
        palette_hints=("blue", "emerald", "indigo", "violet", "cyan", "teal", "purple"),
        domain_hints=(
            "monitoring_dashboard",
            "ide",
            "cloud_console",
            "chat_app",
            "database_manager",
            "email_client",
            "calendar_portal",
            "general_web_page",
        ),
        legacy=False,
        token_preset={
            "font_family": "sans-serif-modern",
            "font_size_scale": "default",
            "border_style": "solid_thin",
            "border_radius": "rounded-lg",
            "shadow_type": "soft-modern",
            "header_style": "glossy",
            "display_effect": "none",
            "divider_style": "solid",
            "background_texture": "solid",
        },
        custom_css_rules=("Use soft shadows on hover", "Accent borders using primary slate"),
        aesthetic_descriptor=(
            "Soft modern SaaS dashboard: rounded cards, very soft ambient shadows, "
            "thin light-gray borders, clean sans-serif type, spacious layouts."
        ),
        render_guidance=(
            "Soft modern SaaS dashboard. Rounded cards (rounded-lg or rounded-xl), very soft "
            "ambient shadows (shadow-sm or shadow-md), thin light-gray border lines (border-slate-100 "
            "or border-slate-800), clean modern sans-serif typography, and spacious layouts. DO NOT "
            "use scanlines or pixel fonts."
        ),
        variation_ranges={
            "font_size_scale": ("default", "compact"),
            "border_radius": ("rounded-lg", "rounded-sm", "rounded-full"),
            "shadow_type": ("soft-modern", "none"),
            "header_style": ("glossy", "solid", "gradient"),
            "divider_style": ("solid", "dashed"),
            "background_texture": ("solid", "dots", "grid_lines"),
        },
        theme_hints=("light", "dark"),
    ),
    DesignArchetype(
        name="retro_90s",
        default_theme="light",
        default_primary_color="teal",
        palette_hints=("teal", "blue", "emerald", "purple", "amber"),
        domain_hints=("general_web_page", "database_manager"),
        legacy=True,
        token_preset={
            "font_family": "monospace",
            "font_size_scale": "default",
            "border_style": "beveled_outset",
            "border_radius": "none",
            "shadow_type": "none",
            "header_style": "classic-title-bar",
            "display_effect": "none",
            "divider_style": "double",
            "background_texture": "solid",
        },
        custom_css_rules=("Beveled outset borders for window", "Chicago retro headers"),
        aesthetic_descriptor=(
            "1990s retro desktop window (Windows 95): light gray panels with beveled outset "
            "borders and retro system typography."
        ),
        render_guidance=(
            "1990s retro desktop window look (e.g. Windows 95). Typically light gray panel canvases "
            "(bg-gray-300) with beveled outset borders (white top/left border, gray-800 bottom/right "
            "border). If color_theme is dark, render as dark gray panels (#2d3748) with matching dark "
            "beveled outset borders. Use retro system-ui typography."
        ),
        variation_ranges={
            "divider_style": ("double", "solid"),
            "display_effect": ("none", "noise_overlay"),
        },
        theme_hints=("light", "dark"),
    ),
    DesignArchetype(
        name="flat_minimalist",
        default_theme="light",
        default_primary_color="orange",
        palette_hints=("orange", "rose", "blue", "emerald", "indigo", "amber"),
        domain_hints=("chat_app", "general_web_page", "database_manager", "email_client", "calendar_portal"),
        legacy=False,
        token_preset={
            "font_family": "sans-serif-retro",
            "font_size_scale": "coarse",
            "border_style": "solid_thick",
            "border_radius": "none",
            "shadow_type": "hard-retro",
            "header_style": "solid",
            "display_effect": "none",
            "divider_style": "solid",
            "background_texture": "grid_lines",
        },
        custom_css_rules=("Thick borders 2px", "Blueprint grid lines in back"),
        aesthetic_descriptor=(
            "Flat wireframe look: crisp thin solid borders, extra-light backgrounds, no "
            "rounded corners, zero shadows, clean sans-serif/monospace type."
        ),
        render_guidance=(
            "Flat wireframe look. Crisp thin 1px solid black or dark-gray borders, solid white or "
            "extra-light gray backgrounds, no rounded corners (rounded-none), zero drop shadows "
            "(shadow-none), and clean sans-serif/monospace typography. DO NOT use scanlines."
        ),
        variation_ranges={
            "font_size_scale": ("coarse", "default"),
            "background_texture": ("grid_lines", "solid", "dots"),
            "border_style": ("solid_thick", "solid_thin"),
            "divider_style": ("solid", "dashed", "dotted"),
        },
        theme_hints=("light", "dark"),
    ),
    DesignArchetype(
        name="terminal_tui",
        default_theme="dark",
        default_primary_color="emerald",
        palette_hints=("emerald", "cyan", "amber", "blue"),
        domain_hints=("ide", "monitoring_dashboard"),
        legacy=False,
        token_preset={
            "font_family": "pixel",
            "font_size_scale": "default",
            "border_style": "solid_thin",
            "border_radius": "none",
            "shadow_type": "neon-glow",
            "header_style": "solid",
            "display_effect": "crt_scanlines",
            "divider_style": "ascii",
            "background_texture": "dots",
        },
        custom_css_rules=("Green phosphor scanline glow", "Dotted grid backdrop"),
        aesthetic_descriptor=(
            "Retro terminal / sci-fi HUD console: deep black background, monospace/pixel type, "
            "ASCII borders, CRT scanlines, neon text matching the accent color."
        ),
        render_guidance=(
            "Retro terminal or Sci-Fi HUD console. Deep pure black background, monospaced or pixel "
            "typography, technical borders (ASCII border bars like +-----+ or clean thin panels), CRT "
            "scanline overlays, neon text/graphics (emerald-green, cyan, amber, etc. matching the "
            "primary_color), and technical indicators (e.g. [SECURE CONNECTION], [SYSTEM READINGS])."
        ),
        variation_ranges={
            "font_family": ("pixel", "monospace"),
            "divider_style": ("ascii", "solid"),
            "background_texture": ("dots", "grid_lines"),
        },
        theme_hints=("dark",),
    ),
    DesignArchetype(
        name="vercel_minimalist",
        default_theme="dark",
        default_primary_color="blue",
        palette_hints=("blue", "violet", "indigo", "cyan", "emerald"),
        domain_hints=("cloud_console", "general_web_page", "database_manager"),
        legacy=False,
        token_preset={
            "font_family": "monospace",
            "font_size_scale": "default",
            "border_style": "solid_thin",
            "border_radius": "none",
            "shadow_type": "none",
            "header_style": "solid",
            "display_effect": "none",
            "divider_style": "solid",
            "background_texture": "solid",
        },
        custom_css_rules=("Vercel pure black canvas", "No roundness, thin borders"),
        aesthetic_descriptor=(
            "Vercel-like flat portal: pure black/white canvas, no rounded corners, no shadows, "
            "1px crisp borders, monospace font, centered top-tab navigation."
        ),
        render_guidance=(
            "Vercel-like flat portal. Absolute solid black (#000000) or white (#ffffff) canvas, no "
            "rounded corners, no shadows, 1px thin crisp borders, and clean monospace font. CRITICAL "
            "LAYOUT RULE: Remove the left sidebar entirely. Add a horizontal top tab sub-navigation bar "
            "below the header (Overview, Deployments, Settings) and center the main panels with a "
            "maximum width container (max-w-6xl mx-auto)."
        ),
        variation_ranges={
            "font_family": ("monospace", "sans-serif-modern"),
        },
        theme_hints=("dark", "light"),
    ),
    DesignArchetype(
        name="datadog_dashboard",
        default_theme="light",
        default_primary_color="purple",
        palette_hints=("purple", "indigo", "violet", "blue", "cyan"),
        domain_hints=("monitoring_dashboard", "cloud_console"),
        legacy=False,
        token_preset={
            "font_family": "sans-serif-modern",
            "font_size_scale": "default",
            "border_style": "solid_thin",
            "border_radius": "rounded-sm",
            "shadow_type": "none",
            "header_style": "solid",
            "display_effect": "none",
            "divider_style": "solid",
            "background_texture": "solid",
        },
        custom_css_rules=("Datadog Purple top bar #632ca6", "Compact widget frames"),
        aesthetic_descriptor=(
            "Datadog-style monitoring grid: dark-slate metrics widgets, compact indicators, "
            "deep purple branding accents, full-width dashboard (no left sidebar)."
        ),
        render_guidance=(
            "Datadog look. Dark-slate monitoring metrics grid, compact indicators, and deep purple "
            "branding accents (#632ca6). CRITICAL LAYOUT RULE: Remove the left sidebar entirely to make "
            "the dashboard fullscreen layout."
        ),
        variation_ranges={
            "font_size_scale": ("default", "compact"),
            "border_radius": ("rounded-sm", "none"),
            "header_style": ("solid", "gradient"),
            "background_texture": ("solid", "grid_lines"),
        },
        theme_hints=("light", "dark"),
    ),
    DesignArchetype(
        name="apple_developer",
        default_theme="light",
        default_primary_color="indigo",
        palette_hints=("indigo", "blue", "violet", "purple", "teal"),
        domain_hints=("email_client", "general_web_page", "cloud_console", "database_manager", "calendar_portal"),
        legacy=False,
        token_preset={
            "font_family": "sans-serif-modern",
            "font_size_scale": "default",
            "border_style": "solid_thin",
            "border_radius": "rounded-lg",
            "shadow_type": "soft-modern",
            "header_style": "glossy",
            "display_effect": "none",
            "divider_style": "solid",
            "background_texture": "solid",
        },
        custom_css_rules=("Frosted glass navigation panel", "macOS Traffic Light window controls"),
        aesthetic_descriptor=(
            "Apple macOS/Developer portal: extra-rounded cards, SF-Pro-like type, frosted glass "
            "header backing, macOS traffic-light window controls."
        ),
        render_guidance=(
            "Apple macOS/Developer portal look. Extra-rounded card panels (rounded-2xl or rounded-3xl), "
            "clean SF Pro sans-serif typography, frosted glass header backings (backdrop-blur-md "
            "bg-white/70 in light, bg-slate-900/70 in dark), and left-aligned red/yellow/green macOS "
            "window controls."
        ),
        variation_ranges={
            "border_radius": ("rounded-lg", "rounded-full"),
            "header_style": ("glossy", "solid"),
            "shadow_type": ("soft-modern", "none"),
            "background_texture": ("solid", "dots"),
        },
        theme_hints=("light", "dark"),
    ),
    DesignArchetype(
        name="neo_brutalism",
        default_theme="light",
        default_primary_color="rose",
        palette_hints=("rose", "amber", "orange", "fuchsia", "pink", "emerald"),
        domain_hints=("general_web_page", "chat_app"),
        legacy=False,
        token_preset={
            "font_family": "sans-serif-retro",
            "font_size_scale": "coarse",
            "border_style": "solid_thick",
            "border_radius": "none",
            "shadow_type": "hard-retro",
            "header_style": "solid",
            "display_effect": "none",
            "divider_style": "solid",
            "background_texture": "solid",
        },
        custom_css_rules=(
            "Thick 4px borders",
            "Hot yellow background",
            "Hard offset zero-blur shadows",
        ),
        aesthetic_descriptor=(
            "Neo-Brutalism: thick solid black borders, highly saturated flat fills, flat zero-blur "
            "offset drop shadows, bold raw typography."
        ),
        render_guidance=(
            "Neo-Brutalism style. Thick solid black borders (border-4 border-black), highly saturated "
            "flat background fills (neon-yellow, hot-pink, intense orange), completely flat drop shadows "
            "with zero blur offset by 6px (shadow-[6px_6px_0px_0px_rgba(0,0,0,1)]), and bold raw "
            "typography."
        ),
        variation_ranges={
            "font_size_scale": ("coarse", "default"),
            "border_radius": ("none", "rounded-sm"),
            "divider_style": ("solid", "dashed", "double"),
            "background_texture": ("solid", "dots", "grid_lines"),
        },
        theme_hints=("light",),
    ),
    DesignArchetype(
        name="glassmorphism_3d",
        default_theme="light",
        default_primary_color="purple",
        palette_hints=("purple", "violet", "indigo", "blue", "cyan", "fuchsia"),
        domain_hints=("general_web_page", "cloud_console"),
        legacy=False,
        token_preset={
            "font_family": "sans-serif-modern",
            "font_size_scale": "default",
            "border_style": "solid_thin",
            "border_radius": "rounded-lg",
            "shadow_type": "soft-modern",
            "header_style": "glossy",
            "display_effect": "none",
            "divider_style": "solid",
            "background_texture": "solid",
        },
        custom_css_rules=("Frosted glass panel backdrop-blur", "Colorful mesh background"),
        aesthetic_descriptor=(
            "3D Glassmorphism: vibrant translucent radial-gradient mesh background with "
            "frosted, slate-tinted translucent panels and soft ambient depth shadows."
        ),
        render_guidance=(
            "3D Glassmorphism style. Strictly implement the background on the body using a fixed CSS "
            "gradient blending 2-3 translucent radial-gradient colors (one of which must match the "
            "selected primary_color accent) over a neutral background (#f8fafc for light, #0b0f19 for "
            "dark) to simulate a vibrant mesh gradient. Use only translucent, slightly slate-tinted "
            "white classes (e.g., bg-slate-50/40 backdrop-blur-xl border border-white/45 for light "
            "theme, bg-slate-950/40 backdrop-blur-xl border border-white/10 for dark theme) on "
            "cards/panels, and apply a dark, ambient soft shadow (soft-modern) for clear depth "
            "separation. DO NOT use solid panel backgrounds."
        ),
        variation_ranges={
            "border_radius": ("rounded-lg", "rounded-full"),
            "header_style": ("glossy", "solid"),
            "background_texture": ("solid", "dots"),
        },
        theme_hints=("light", "dark"),
    ),
    DesignArchetype(
        name="classic_mac_80s",
        default_theme="light",
        default_primary_color="blue",
        palette_hints=("blue", "teal", "emerald"),
        domain_hints=("general_web_page",),
        legacy=True,
        token_preset={
            "font_family": "monospace",
            "font_size_scale": "default",
            "border_style": "solid_thin",
            "border_radius": "none",
            "shadow_type": "none",
            "header_style": "solid",
            "display_effect": "none",
            "divider_style": "double",
            "background_texture": "solid",
        },
        custom_css_rules=(
            "Monochromatic black and white",
            "Checkerboard grey background dither",
            "Double ruled borders",
        ),
        aesthetic_descriptor=(
            "1980s Macintosh (System 1/6): monochrome black-and-white dithered pixels, "
            "checkerboard grey background, double-ruled window frames."
        ),
        render_guidance=(
            "1980s retro computed Macintosh look (System 1/6). Typically monochromatic black-and-white "
            "dithered pixels with checkerboard grey background patterns and double-ruled window frames. "
            "If color_theme is dark, render as a dark monochrome green/amber screen to simulate a CRT "
            "computer display of that era."
        ),
        variation_ranges={
            "display_effect": ("none", "crt_scanlines"),
        },
        theme_hints=("light", "dark"),
    ),
    # -----------------------------------------------------------------------
    # WD4: expanded modern archetype library. These are *appended* after the
    # historical 10 so the canonical order -- and any pre-existing MD5-seeded
    # selection that indexes into it -- stays reproducible. All are non-legacy
    # modern product design languages with strong, coherent in-archetype
    # variation. They render via the template's generic token path
    # (chart_template.html sections 1-7); a novel ``visual_style`` name simply
    # skips the optional name-keyed signature flourishes and still receives full
    # token treatment (fonts, borders, shadows, header, textures, dividers).
    # -----------------------------------------------------------------------
    DesignArchetype(
        name="material_design",
        default_theme="light",
        default_primary_color="indigo",
        palette_hints=("indigo", "blue", "purple", "teal", "cyan", "emerald"),
        domain_hints=(
            "monitoring_dashboard",
            "cloud_console",
            "chat_app",
            "email_client",
            "calendar_portal",
            "general_web_page",
        ),
        legacy=False,
        token_preset={
            "font_family": "sans-serif-modern",
            "font_size_scale": "default",
            "border_style": "solid_thin",
            "border_radius": "rounded-lg",
            "shadow_type": "soft-modern",
            "header_style": "solid",
            "display_effect": "none",
            "divider_style": "solid",
            "background_texture": "solid",
        },
        custom_css_rules=(
            "Material elevation shadows (dp levels)",
            "Bold filled primary-color app bar",
            "Pill-shaped FAB and chips",
        ),
        aesthetic_descriptor=(
            "Google Material Design 3: elevation-based soft shadows, a bold filled "
            "primary-color app bar, rounded cards and pill-shaped chips/buttons, and clean "
            "Roboto-like sans-serif type."
        ),
        render_guidance=(
            "Google Material Design 3 (Material You) look. Use elevation-based soft ambient "
            "shadows to separate surfaces, a bold solid primary-color top app bar with white "
            "text, generously rounded cards (rounded-lg/rounded-xl), pill-shaped buttons and "
            "chips (rounded-full), and clean Roboto-like sans-serif typography. Tint surfaces "
            "subtly with the primary color. DO NOT use scanlines, pixel fonts, or beveled borders."
        ),
        variation_ranges={
            "font_size_scale": ("default", "coarse"),
            "border_radius": ("rounded-lg", "rounded-full"),
            "shadow_type": ("soft-modern", "none"),
            "header_style": ("solid", "gradient"),
            "background_texture": ("solid", "dots"),
        },
        theme_hints=("light", "dark"),
    ),
    DesignArchetype(
        name="notion_docs",
        default_theme="light",
        default_primary_color="blue",
        palette_hints=("blue", "indigo", "amber", "emerald", "rose", "purple"),
        domain_hints=(
            "general_web_page",
            "database_manager",
            "email_client",
            "calendar_portal",
        ),
        legacy=False,
        token_preset={
            "font_family": "sans-serif-modern",
            "font_size_scale": "default",
            "border_style": "solid_thin",
            "border_radius": "rounded-sm",
            "shadow_type": "none",
            "header_style": "solid",
            "display_effect": "none",
            "divider_style": "solid",
            "background_texture": "solid",
        },
        custom_css_rules=(
            "Generous whitespace on a calm neutral canvas",
            "Hairline gray dividers between blocks",
            "Subtle hover-gray on rows",
        ),
        aesthetic_descriptor=(
            "Notion-style minimalist document workspace: calm neutral canvas, generous "
            "whitespace, hairline gray dividers, a restrained accent color, and an optional serif body."
        ),
        render_guidance=(
            "Notion-style minimalist document/workspace look. Calm near-white (or near-black in "
            "dark) canvas, generous whitespace, hairline light-gray dividers between content "
            "blocks, small rounded corners, no drop shadows, and a very restrained use of the "
            "accent color (mostly for links and small highlights). Typography is clean and "
            "readable; a serif body font is acceptable. DO NOT use heavy borders, scanlines, or bold fills."
        ),
        variation_ranges={
            "font_family": ("sans-serif-modern", "serif"),
            "font_size_scale": ("default", "coarse"),
            "border_radius": ("rounded-sm", "rounded-lg"),
            "divider_style": ("solid", "dotted"),
        },
        theme_hints=("light", "dark"),
    ),
    DesignArchetype(
        name="linear_dark",
        default_theme="dark",
        default_primary_color="violet",
        palette_hints=("violet", "indigo", "purple", "blue", "fuchsia"),
        domain_hints=(
            "ide",
            "monitoring_dashboard",
            "cloud_console",
            "general_web_page",
        ),
        legacy=False,
        token_preset={
            "font_family": "sans-serif-modern",
            "font_size_scale": "default",
            "border_style": "solid_thin",
            "border_radius": "rounded-lg",
            "shadow_type": "soft-modern",
            "header_style": "gradient",
            "display_effect": "none",
            "divider_style": "solid",
            "background_texture": "solid",
        },
        custom_css_rules=(
            "Deep charcoal canvas with a subtle accent gradient wash",
            "Crisp 1px low-contrast borders",
            "Tight, precise spacing",
        ),
        aesthetic_descriptor=(
            "Linear-style modern product UI: deep charcoal canvas with a subtle accent gradient "
            "wash, crisp low-contrast 1px borders, rounded cards, tight precise spacing, and refined sans-serif type."
        ),
        render_guidance=(
            "Linear-app-style modern product UI. Deep charcoal/near-black canvas (in dark) or a "
            "clean off-white (in light) with a subtle accent-colored gradient wash in the header "
            "or background, crisp low-contrast 1px borders, rounded-lg cards, soft ambient shadows, "
            "tight precise spacing, and a refined modern sans-serif. Keep it sleek and understated; "
            "DO NOT use scanlines, pixel fonts, or thick brutalist borders."
        ),
        variation_ranges={
            "border_radius": ("rounded-lg", "rounded-sm"),
            "shadow_type": ("soft-modern", "none"),
            "header_style": ("gradient", "solid"),
            "background_texture": ("solid", "dots"),
        },
        theme_hints=("dark", "light"),
    ),
    DesignArchetype(
        name="stripe_dashboard",
        default_theme="light",
        default_primary_color="indigo",
        palette_hints=("indigo", "violet", "purple", "blue", "cyan"),
        domain_hints=(
            "cloud_console",
            "monitoring_dashboard",
            "database_manager",
            "general_web_page",
        ),
        legacy=False,
        token_preset={
            "font_family": "sans-serif-modern",
            "font_size_scale": "default",
            "border_style": "solid_thin",
            "border_radius": "rounded-lg",
            "shadow_type": "soft-modern",
            "header_style": "gradient",
            "display_effect": "none",
            "divider_style": "solid",
            "background_texture": "solid",
        },
        custom_css_rules=(
            "Signature angled multi-hue gradient header band",
            "Refined rounded cards with soft shadows",
            "Crisp data tables with hairline rules",
        ),
        aesthetic_descriptor=(
            "Stripe-style fintech dashboard: a signature angled multi-hue gradient header band, "
            "refined rounded cards with soft shadows, crisp data tables, and polished sans-serif type."
        ),
        render_guidance=(
            "Stripe-style fintech dashboard. Feature a polished angled/diagonal multi-hue gradient "
            "header band built from the primary color (e.g. indigo -> violet -> cyan), refined rounded "
            "cards (rounded-lg) with soft ambient shadows, crisp data tables with hairline row rules, "
            "and clean polished sans-serif typography. Feels premium and financial. DO NOT use "
            "scanlines, pixel fonts, or beveled borders."
        ),
        variation_ranges={
            "header_style": ("gradient", "glossy", "solid"),
            "border_radius": ("rounded-lg", "rounded-sm"),
            "shadow_type": ("soft-modern", "none"),
            "font_size_scale": ("default", "compact"),
        },
        theme_hints=("light", "dark"),
    ),
    DesignArchetype(
        name="github_dark",
        default_theme="dark",
        default_primary_color="blue",
        palette_hints=("blue", "indigo", "emerald", "purple", "cyan"),
        domain_hints=(
            "ide",
            "cloud_console",
            "database_manager",
            "general_web_page",
        ),
        legacy=False,
        token_preset={
            "font_family": "sans-serif-modern",
            "font_size_scale": "compact",
            "border_style": "solid_thin",
            "border_radius": "rounded-sm",
            "shadow_type": "none",
            "header_style": "solid",
            "display_effect": "none",
            "divider_style": "solid",
            "background_texture": "solid",
        },
        custom_css_rules=(
            "Dark slate canvas with muted 1px borders",
            "Monospace for code and identifiers",
            "Compact, information-dense rows",
        ),
        aesthetic_descriptor=(
            "GitHub-style developer platform: dark slate canvas, muted 1px borders, compact "
            "information-dense rows, small rounded corners, and monospace for code/identifiers."
        ),
        render_guidance=(
            "GitHub-style developer platform look. Dark slate canvas (#0d1117-like) or a clean light "
            "gray, muted low-contrast 1px borders on boxed sections, compact information-dense rows, "
            "small rounded corners (rounded-sm/rounded-md), no drop shadows, a solid header bar, and "
            "monospace typography for code, hashes, and identifiers mixed with a clean sans-serif for "
            "prose. DO NOT use scanlines, neon glow, or thick brutalist borders."
        ),
        variation_ranges={
            "font_size_scale": ("compact", "default"),
            "border_radius": ("rounded-sm", "rounded-lg"),
            "font_family": ("sans-serif-modern", "monospace"),
            "divider_style": ("solid", "dashed"),
        },
        theme_hints=("dark", "light"),
    ),
    DesignArchetype(
        name="bloomberg_terminal",
        default_theme="dark",
        default_primary_color="amber",
        palette_hints=("amber", "orange", "emerald", "cyan", "rose"),
        domain_hints=("monitoring_dashboard", "database_manager"),
        legacy=False,
        token_preset={
            "font_family": "monospace",
            "font_size_scale": "compact",
            "border_style": "solid_thin",
            "border_radius": "none",
            "shadow_type": "none",
            "header_style": "solid",
            "display_effect": "none",
            "divider_style": "solid",
            "background_texture": "grid_lines",
        },
        custom_css_rules=(
            "Ultra-dense monospace data grid",
            "Amber/green tickers on black",
            "Thin ruled column separators",
        ),
        aesthetic_descriptor=(
            "Bloomberg-style financial data terminal: ultra-dense monospace data grids on a black "
            "canvas, amber/green ticker text, thin ruled column separators, and zero rounding."
        ),
        render_guidance=(
            "Bloomberg-style modern financial data terminal. Ultra-dense monospace data grids on a "
            "deep black canvas, high-contrast amber/green/cyan ticker-style numeric text keyed to the "
            "primary color, thin ruled column and row separators (subtle grid lines), square corners "
            "(no rounding), and no drop shadows. Pack information tightly like a trading desk. This is "
            "a MODERN high-density terminal, not a retro CRT: DO NOT add scanlines or pixel fonts."
        ),
        variation_ranges={
            "font_size_scale": ("compact", "default"),
            "background_texture": ("grid_lines", "solid", "dots"),
            "divider_style": ("solid", "dotted", "dashed"),
            "border_radius": ("none", "rounded-sm"),
        },
        theme_hints=("dark",),
    ),
    DesignArchetype(
        name="spotify_media",
        default_theme="dark",
        default_primary_color="emerald",
        palette_hints=("emerald", "fuchsia", "pink", "violet", "purple"),
        domain_hints=("chat_app", "calendar_portal", "general_web_page"),
        legacy=False,
        token_preset={
            "font_family": "sans-serif-modern",
            "font_size_scale": "default",
            "border_style": "none",
            "border_radius": "rounded-lg",
            "shadow_type": "soft-modern",
            "header_style": "gradient",
            "display_effect": "none",
            "divider_style": "solid",
            "background_texture": "solid",
        },
        custom_css_rules=(
            "Near-black canvas with a vivid accent-gradient hero",
            "Borderless rounded media cards and pill controls",
            "Bold heavy display headings",
        ),
        aesthetic_descriptor=(
            "Spotify-style media console: near-black canvas, a vivid accent-gradient hero band, "
            "borderless rounded media cards, pill-shaped controls, and bold heavy display headings."
        ),
        render_guidance=(
            "Spotify-style media/streaming console. Near-black canvas, a vivid accent-color gradient "
            "hero/header band fading into the background, borderless rounded media cards (rounded-lg) "
            "with soft shadows, pill-shaped controls and chips (rounded-full), and bold heavy display "
            "headings with a clean sans-serif. Rely on fills and gradients instead of borders. DO NOT "
            "use scanlines, pixel fonts, or hard brutalist borders."
        ),
        variation_ranges={
            "border_radius": ("rounded-lg", "rounded-full"),
            "header_style": ("gradient", "solid"),
            "shadow_type": ("soft-modern", "none"),
            "background_texture": ("solid", "dots"),
        },
        theme_hints=("dark",),
    ),
    DesignArchetype(
        name="discord_chat",
        default_theme="dark",
        default_primary_color="indigo",
        palette_hints=("indigo", "violet", "blue", "purple", "fuchsia"),
        domain_hints=("chat_app", "general_web_page"),
        legacy=False,
        token_preset={
            "font_family": "sans-serif-modern",
            "font_size_scale": "default",
            "border_style": "none",
            "border_radius": "rounded-lg",
            "shadow_type": "none",
            "header_style": "solid",
            "display_effect": "none",
            "divider_style": "solid",
            "background_texture": "solid",
        },
        custom_css_rules=(
            "Blurple accents on dark slate channels",
            "Borderless rounded message groups and pills",
            "Rounded avatar chips",
        ),
        aesthetic_descriptor=(
            "Discord-style community chat: dark slate channel columns, blurple accents, borderless "
            "rounded message groups, rounded avatar chips, and a friendly modern sans-serif."
        ),
        render_guidance=(
            "Discord-style community chat app. Dark slate multi-column layout (channel list + message "
            "pane), 'blurple' indigo/violet accents keyed to the primary color, borderless rounded "
            "message groups and rounded-lg panels, rounded avatar chips, hover-highlighted rows, and a "
            "friendly modern sans-serif. Rely on fills rather than borders; keep it playful but clean. "
            "DO NOT use scanlines, pixel fonts, or brutalist borders."
        ),
        variation_ranges={
            "border_radius": ("rounded-lg", "rounded-full"),
            "font_size_scale": ("default", "compact"),
            "background_texture": ("solid", "dots"),
        },
        theme_hints=("dark", "light"),
    ),
)

# Fast lookup by name.
_BY_NAME: dict[str, DesignArchetype] = {a.name: a for a in _ARCHETYPES}

# Canonical ordered list of ``visual_style`` names (historical order preserved).
VISUAL_STYLE_NAMES: tuple[str, ...] = tuple(a.name for a in _ARCHETYPES)


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


def all_archetypes() -> tuple[DesignArchetype, ...]:
    """Return the full archetype library (canonical order)."""
    return _ARCHETYPES


def archetype_names() -> list[str]:
    """Return every canonical ``visual_style`` name (canonical order)."""
    return list(VISUAL_STYLE_NAMES)


def has_archetype(name: str) -> bool:
    """Return whether ``name`` is a known archetype."""
    return name in _BY_NAME


def get_archetype(name: str, default: str = "modern_clean") -> DesignArchetype:
    """Return the archetype for ``name``, falling back to ``default``.

    ``default`` must itself be a valid archetype name (defaults to the
    universal ``modern_clean``).
    """
    archetype = _BY_NAME.get(name)
    if archetype is not None:
        return archetype
    return _BY_NAME[default]


def default_style_details(name: str) -> dict[str, Any]:
    """Return a fresh default ``style_details`` dict for archetype ``name``."""
    return get_archetype(name).default_style_details()


def archetypes_for_shell(
    shell: str | None,
    *,
    include_legacy: bool = False,
) -> list[DesignArchetype]:
    """Return archetypes that realistically fit ``shell``.

    An archetype matches when its ``domain_hints`` is empty (universal) or
    contains ``shell``. Legacy archetypes are excluded unless ``include_legacy``
    is set. If nothing matches, all non-legacy archetypes are returned so
    callers always have a usable pool.
    """
    pool = [a for a in _ARCHETYPES if include_legacy or not a.legacy]
    if not shell:
        return pool
    matched = [a for a in pool if not a.domain_hints or shell in a.domain_hints]
    return matched or pool


# ---------------------------------------------------------------------------
# Deterministic seeded sampling (WD2)
# ---------------------------------------------------------------------------


def sample_platform_design(
    seed: Any,
    *,
    shell_hint: str | None = None,
    allow_legacy: bool = False,
) -> dict[str, Any]:
    """Deterministically compose one coherent per-platform design language.

    Given a stable ``seed`` (any hashable; strings are recommended and are
    reproducible across runs/interpreters via CPython's version-2 seeding), this
    samples a *whole* coherent design -- a UI shell, a design archetype that
    realistically fits that shell, a coherent theme/accent, and per-token
    micro-styling varied only within the archetype's declared
    ``variation_ranges``. The same seed always yields the same design.

    This is the SSoT sampler used by the materializer in place of the historical
    MD5-index-into-flat-lists selection, which could pair incoherent
    style/theme/shell combinations (e.g. a dark-only terminal look forced into a
    light email client). Sampling *within* an archetype keeps every platform
    looking like one real product while still varying across scenarios/seeds.

    Args:
      seed: Stable seed value (typically ``f"{title}_{platform_id}_{style_seed}"``).
      shell_hint: If given and valid, forces the ``shell_type`` instead of
        sampling one (the archetype is still sampled among those that fit it).
      allow_legacy: When True, anachronistic/vintage archetypes (retro_90s,
        classic_mac_80s) become eligible. Off by default so modern platforms
        never accidentally get a vintage look.

    Returns:
      A dict with keys ``shell_type``, ``visual_style``, ``color_theme``,
      ``primary_color`` and ``style_details`` (a fresh, mutable token bundle).
    """
    rng = random.Random(seed)

    if shell_hint and shell_hint in UI_SHELL_TYPES:
        shell = shell_hint
    else:
        shell = rng.choice(UI_SHELL_TYPES)

    # ``archetypes_for_shell`` returns a list in canonical ``_ARCHETYPES`` order,
    # so the choice below is deterministic for a given seed.
    pool = archetypes_for_shell(shell, include_legacy=allow_legacy)
    archetype = rng.choice(pool)

    theme = rng.choice(archetype.allowed_themes())
    palette = archetype.palette_hints or PRIMARY_COLORS
    primary_color = rng.choice(palette)

    style_details = archetype.default_style_details()
    # Sort axes for a stable RNG-consumption order independent of dict authoring.
    for axis, choices in sorted(archetype.variation_ranges.items()):
        if choices:
            style_details[axis] = rng.choice(list(choices))

    return {
        "shell_type": shell,
        "visual_style": archetype.name,
        "color_theme": theme,
        "primary_color": primary_color,
        "style_details": style_details,
    }
