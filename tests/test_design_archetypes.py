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

"""Tests for the design-archetype SSoT library (VD / WD1).

These lock the single-source-of-truth guarantees:
  * the canonical style vocabulary and its historical order (reproducibility),
  * every archetype's token preset is valid against the declared axis vocab,
  * the default presets byte-match the historical hand-maintained values
    (so wiring the gallery / materializer to the library changed nothing), and
  * the accessors (fallback, per-shell filtering, copy-on-read) behave.
"""

from __future__ import annotations

import copy

from clasharness.scenario import design_archetypes as da


# Historical ordered style list (was hardcoded in vlm_materializer's
# suggested_visual_styles). Order is load-bearing: the MD5-seeded selection
# indexes into it, so any reorder silently changes existing style_seeds.
_HISTORICAL_STYLE_ORDER = [
    "modern_clean",
    "retro_90s",
    "flat_minimalist",
    "terminal_tui",
    "vercel_minimalist",
    "datadog_dashboard",
    "apple_developer",
    "neo_brutalism",
    "glassmorphism_3d",
    "classic_mac_80s",
]

# WD4 grew the modern (non-legacy) library. New archetypes are *appended* after
# the historical block so the canonical order (and any pre-existing MD5-seeded
# selection that indexes into it) stays reproducible.
_WD4_STYLE_ORDER = [
    "material_design",
    "notion_docs",
    "linear_dark",
    "stripe_dashboard",
    "github_dark",
    "bloomberg_terminal",
    "spotify_media",
    "discord_chat",
]

# The full canonical order = historical block followed by the WD4 additions.
_ALL_STYLE_ORDER = _HISTORICAL_STYLE_ORDER + _WD4_STYLE_ORDER

# Historical gallery presets (name -> theme, primary_color, full style_details).
_HISTORICAL_PRESETS: dict[str, tuple[str, str, dict]] = {
    "modern_clean": ("light", "blue", {
        "font_family": "sans-serif-modern", "font_size_scale": "default",
        "border_style": "solid_thin", "border_radius": "rounded-lg",
        "shadow_type": "soft-modern", "header_style": "glossy",
        "display_effect": "none", "divider_style": "solid",
        "background_texture": "solid",
        "custom_css_rules": ["Use soft shadows on hover", "Accent borders using primary slate"],
    }),
    "retro_90s": ("light", "teal", {
        "font_family": "monospace", "font_size_scale": "default",
        "border_style": "beveled_outset", "border_radius": "none",
        "shadow_type": "none", "header_style": "classic-title-bar",
        "display_effect": "none", "divider_style": "double",
        "background_texture": "solid",
        "custom_css_rules": ["Beveled outset borders for window", "Chicago retro headers"],
    }),
    "flat_minimalist": ("light", "orange", {
        "font_family": "sans-serif-retro", "font_size_scale": "coarse",
        "border_style": "solid_thick", "border_radius": "none",
        "shadow_type": "hard-retro", "header_style": "solid",
        "display_effect": "none", "divider_style": "solid",
        "background_texture": "grid_lines",
        "custom_css_rules": ["Thick borders 2px", "Blueprint grid lines in back"],
    }),
    "terminal_tui": ("dark", "emerald", {
        "font_family": "pixel", "font_size_scale": "default",
        "border_style": "solid_thin", "border_radius": "none",
        "shadow_type": "neon-glow", "header_style": "solid",
        "display_effect": "crt_scanlines", "divider_style": "ascii",
        "background_texture": "dots",
        "custom_css_rules": ["Green phosphor scanline glow", "Dotted grid backdrop"],
    }),
    "vercel_minimalist": ("dark", "blue", {
        "font_family": "monospace", "font_size_scale": "default",
        "border_style": "solid_thin", "border_radius": "none",
        "shadow_type": "none", "header_style": "solid",
        "display_effect": "none", "divider_style": "solid",
        "background_texture": "solid",
        "custom_css_rules": ["Vercel pure black canvas", "No roundness, thin borders"],
    }),
    "datadog_dashboard": ("light", "purple", {
        "font_family": "sans-serif-modern", "font_size_scale": "default",
        "border_style": "solid_thin", "border_radius": "rounded-sm",
        "shadow_type": "none", "header_style": "solid",
        "display_effect": "none", "divider_style": "solid",
        "background_texture": "solid",
        "custom_css_rules": ["Datadog Purple top bar #632ca6", "Compact widget frames"],
    }),
    "apple_developer": ("light", "indigo", {
        "font_family": "sans-serif-modern", "font_size_scale": "default",
        "border_style": "solid_thin", "border_radius": "rounded-lg",
        "shadow_type": "soft-modern", "header_style": "glossy",
        "display_effect": "none", "divider_style": "solid",
        "background_texture": "solid",
        "custom_css_rules": ["Frosted glass navigation panel", "macOS Traffic Light window controls"],
    }),
    "neo_brutalism": ("light", "rose", {
        "font_family": "sans-serif-retro", "font_size_scale": "coarse",
        "border_style": "solid_thick", "border_radius": "none",
        "shadow_type": "hard-retro", "header_style": "solid",
        "display_effect": "none", "divider_style": "solid",
        "background_texture": "solid",
        "custom_css_rules": ["Thick 4px borders", "Hot yellow background", "Hard offset zero-blur shadows"],
    }),
    "glassmorphism_3d": ("light", "purple", {
        "font_family": "sans-serif-modern", "font_size_scale": "default",
        "border_style": "solid_thin", "border_radius": "rounded-lg",
        "shadow_type": "soft-modern", "header_style": "glossy",
        "display_effect": "none", "divider_style": "solid",
        "background_texture": "solid",
        "custom_css_rules": ["Frosted glass panel backdrop-blur", "Colorful mesh background"],
    }),
    "classic_mac_80s": ("light", "blue", {
        "font_family": "monospace", "font_size_scale": "default",
        "border_style": "solid_thin", "border_radius": "none",
        "shadow_type": "none", "header_style": "solid",
        "display_effect": "none", "divider_style": "double",
        "background_texture": "solid",
        "custom_css_rules": ["Monochromatic black and white", "Checkerboard grey background dither", "Double ruled borders"],
    }),
}


def test_canonical_style_order_is_stable() -> None:
    # UT-ARCH-01: order is load-bearing for MD5-seeded selection. The historical
    # block must stay first and in order; WD4 additions are appended after it.
    assert list(da.VISUAL_STYLE_NAMES) == _ALL_STYLE_ORDER
    assert da.archetype_names() == _ALL_STYLE_ORDER
    assert da.archetype_names()[: len(_HISTORICAL_STYLE_ORDER)] == _HISTORICAL_STYLE_ORDER


def test_all_style_names_resolve() -> None:
    # UT-ARCH-02: every canonical name resolves to a real archetype.
    for name in _ALL_STYLE_ORDER:
        assert da.has_archetype(name)
        assert da.get_archetype(name).name == name


def test_token_presets_are_valid_against_axes() -> None:
    # UT-ARCH-03: every preset token is within its declared axis vocab.
    for a in da.all_archetypes():
        for axis, allowed in da.STYLE_DETAIL_AXES.items():
            assert axis in a.token_preset, f"{a.name} missing axis {axis}"
            assert a.token_preset[axis] in allowed, f"{a.name}.{axis}={a.token_preset[axis]}"


def test_variation_ranges_are_valid_and_include_preset() -> None:
    # UT-ARCH-04: within-archetype variation stays inside the axis vocab and
    # always includes the archetype's own preset value.
    for a in da.all_archetypes():
        for axis, vals in a.variation_ranges.items():
            assert axis in da.STYLE_DETAIL_AXES, f"{a.name} varies unknown axis {axis}"
            for v in vals:
                assert v in da.STYLE_DETAIL_AXES[axis], f"{a.name}.{axis} bad var {v}"
            assert a.token_preset[axis] in vals, f"{a.name}.{axis} preset not in variation range"


def test_palette_and_theme_defaults_are_coherent() -> None:
    # UT-ARCH-05: default color is within the archetype's palette; theme valid.
    for a in da.all_archetypes():
        assert a.default_primary_color in a.palette_hints
        for c in a.palette_hints:
            assert c in da.PRIMARY_COLORS, f"{a.name} palette color {c} invalid"
        assert a.default_theme in da.COLOR_THEMES


def test_presets_byte_match_historical_values() -> None:
    # UT-ARCH-06: wiring the gallery/materializer to the SSoT changed nothing.
    for name, (theme, color, details) in _HISTORICAL_PRESETS.items():
        a = da.get_archetype(name)
        assert a.default_theme == theme, name
        assert a.default_primary_color == color, name
        assert a.default_style_details() == details, name


def test_default_style_details_returns_a_copy() -> None:
    # UT-ARCH-07: mutating the returned dict must not corrupt the library.
    a = da.get_archetype("modern_clean")
    snapshot = copy.deepcopy(a.token_preset)
    d = a.default_style_details()
    d["font_family"] = "pixel"
    d["custom_css_rules"].append("mutation")
    assert a.token_preset == snapshot
    assert a.default_style_details()["font_family"] == snapshot["font_family"]


def test_get_archetype_falls_back_to_modern_clean() -> None:
    # UT-ARCH-08: unknown names fall back to the universal default.
    assert da.get_archetype("nonexistent_style").name == "modern_clean"
    assert not da.has_archetype("nonexistent_style")


def test_archetypes_for_shell_excludes_legacy_by_default() -> None:
    # UT-ARCH-09: legacy looks are opt-in; universal/matching pools returned.
    ide_pool = da.archetypes_for_shell("ide")
    assert all(not a.legacy for a in ide_pool)
    assert "terminal_tui" in [a.name for a in ide_pool]

    # Legacy archetypes appear only when explicitly included.
    legacy_names = {a.name for a in da.all_archetypes() if a.legacy}
    assert legacy_names == {"retro_90s", "classic_mac_80s"}
    incl = {a.name for a in da.archetypes_for_shell("general_web_page", include_legacy=True)}
    assert "classic_mac_80s" in incl

    # Unknown shell -> full non-legacy pool (never empty).
    assert da.archetypes_for_shell("no_such_shell")
    assert da.archetypes_for_shell(None)


# ---------------------------------------------------------------------------
# Seeded sampling (WD2)
# ---------------------------------------------------------------------------


def test_allowed_themes_fall_back_to_default_theme() -> None:
    # UT-ARCH-10: theme_hints, when empty, resolves to just the default theme;
    # every hinted theme is a valid COLOR_THEMES value.
    for a in da.all_archetypes():
        themes = a.allowed_themes()
        assert themes, f"{a.name} has no allowed themes"
        for t in themes:
            assert t in da.COLOR_THEMES, f"{a.name} theme {t} invalid"
        assert a.default_theme in themes, f"{a.name} default theme not allowed"
        if not a.theme_hints:
            assert themes == (a.default_theme,)


def test_every_shell_has_a_coherent_non_legacy_archetype() -> None:
    # UT-ARCH-11: every UI shell must be realistically covered by at least one
    # non-legacy archetype, so the sampler never has to fall back to an
    # incoherent full-pool pairing (e.g. a Vercel look for a calendar portal).
    for shell in da.UI_SHELL_TYPES:
        fit = [
            a
            for a in da.archetypes_for_shell(shell)
            if not a.domain_hints or shell in a.domain_hints
        ]
        assert fit, f"shell {shell} has no coherent non-legacy archetype"


def test_sample_platform_design_is_deterministic() -> None:
    # UT-ARCH-12: same seed -> byte-identical design (forward reproducibility).
    for seed in ("alpha_p1_0", "beta_default_7", "gamma_platform_x_3"):
        first = da.sample_platform_design(seed)
        second = da.sample_platform_design(seed)
        assert first == second, seed
        # A fresh, mutable style_details each call (no shared library state).
        first["style_details"]["font_family"] = "pixel"
        assert da.sample_platform_design(seed)["style_details"]["font_family"] != "pixel"


def test_sample_platform_design_is_coherent_and_legal() -> None:
    # UT-ARCH-13: sampled designs stay within their archetype's identity across a
    # large seed sweep -- coherent theme/palette/shell and only signature-safe
    # token variation.
    axes = da.STYLE_DETAIL_AXES
    for i in range(300):
        d = da.sample_platform_design(f"scenario_{i}_platform_{i % 4}")
        arch = da.get_archetype(d["visual_style"])
        assert not arch.legacy, d["visual_style"]
        assert d["color_theme"] in arch.allowed_themes()
        assert d["primary_color"] in (arch.palette_hints or da.PRIMARY_COLORS)
        assert d["shell_type"] in da.UI_SHELL_TYPES
        assert (not arch.domain_hints) or d["shell_type"] in arch.domain_hints
        sd = d["style_details"]
        assert isinstance(sd.get("custom_css_rules"), list)
        for axis, vocab in axes.items():
            assert sd[axis] in vocab, (axis, sd[axis])
            if axis in arch.variation_ranges:
                assert sd[axis] in arch.variation_ranges[axis]
            else:
                # Signature axes never drift from the preset.
                assert sd[axis] == arch.token_preset[axis], (d["visual_style"], axis)


def test_sample_excludes_legacy_unless_opted_in() -> None:
    # UT-ARCH-14: vintage looks never surface by default, but can be opted in.
    legacy = {a.name for a in da.all_archetypes() if a.legacy}
    default_styles = {
        da.sample_platform_design(f"nolegacy_{i}")["visual_style"] for i in range(400)
    }
    assert not (default_styles & legacy)
    opted_in = {
        da.sample_platform_design(f"legacy_{i}", allow_legacy=True)["visual_style"]
        for i in range(600)
    }
    assert opted_in & legacy


def test_sample_respects_shell_hint() -> None:
    # UT-ARCH-15: a valid shell_hint pins the shell and the archetype fits it.
    for i in range(60):
        d = da.sample_platform_design(f"hint_{i}", shell_hint="monitoring_dashboard")
        assert d["shell_type"] == "monitoring_dashboard"
        arch = da.get_archetype(d["visual_style"])
        assert not arch.domain_hints or "monitoring_dashboard" in arch.domain_hints
    # An invalid hint is ignored (falls back to sampling a real shell).
    d = da.sample_platform_design("badhint", shell_hint="not_a_shell")
    assert d["shell_type"] in da.UI_SHELL_TYPES


def test_sample_produces_diverse_designs() -> None:
    # UT-ARCH-16: across seeds the sampler exercises the whole non-legacy
    # library and both themes (the point of WD2 -- real cross-scenario variety).
    styles = {da.sample_platform_design(f"div_{i}")["visual_style"] for i in range(500)}
    themes = {da.sample_platform_design(f"div_{i}")["color_theme"] for i in range(500)}
    non_legacy = {a.name for a in da.all_archetypes() if not a.legacy}
    assert styles == non_legacy
    assert themes == set(da.COLOR_THEMES)


# ---------------------------------------------------------------------------
# Expanded modern library (WD4)
# ---------------------------------------------------------------------------


def test_wd4_library_is_expanded_and_fully_reachable() -> None:
    # UT-ARCH-17: WD4 grew the library into a broad, coherent modern set while
    # keeping the historical order stable and the legacy set unchanged. Every new
    # archetype must be a coherent, non-legacy, reachable design language with
    # strong in-archetype variation (not a dead entry behind an unreachable pool).
    names = da.archetype_names()
    assert names[: len(_HISTORICAL_STYLE_ORDER)] == _HISTORICAL_STYLE_ORDER
    assert names[len(_HISTORICAL_STYLE_ORDER):] == _WD4_STYLE_ORDER

    non_legacy = {a.name for a in da.all_archetypes() if not a.legacy}
    assert len(non_legacy) >= 15, non_legacy
    # WD4 only added modern looks; the vintage/legacy set is unchanged.
    assert {a.name for a in da.all_archetypes() if a.legacy} == {
        "retro_90s",
        "classic_mac_80s",
    }

    for name in _WD4_STYLE_ORDER:
        a = da.get_archetype(name)
        assert a.name == name
        assert not a.legacy, name
        # Strong in-archetype variation: at least a couple of coherent knobs.
        assert len(a.variation_ranges) >= 2, name
        # Fits at least one shell coherently (universal or explicit domain fit).
        assert a.domain_hints, name

    # Deterministic default sampling must actually reach every new archetype.
    sampled = {da.sample_platform_design(f"div_{i}")["visual_style"] for i in range(500)}
    for name in _WD4_STYLE_ORDER:
        assert name in sampled, f"{name} unreachable in 500 seeds"


def test_wd4_every_shell_gained_more_coherent_variety() -> None:
    # UT-ARCH-18: the expansion widened per-shell choice so platforms of the same
    # kind don't collapse onto one look. Every shell now has multiple coherent
    # non-legacy archetypes to sample from.
    for shell in da.UI_SHELL_TYPES:
        fit = [
            a
            for a in da.archetypes_for_shell(shell)
            if not a.domain_hints or shell in a.domain_hints
        ]
        assert len(fit) >= 3, (shell, [a.name for a in fit])
