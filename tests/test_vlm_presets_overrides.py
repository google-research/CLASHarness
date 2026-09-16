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

from __future__ import annotations

import pytest
from typing import Any
from clasharness.scenario.chart_presets import CHART_PRESETS, CAPTION_TEMPLATES, ChartPreset, CaptionTemplate
from clasharness.scenario.vlm_spec import VLMAssetSpec, VLMScenarioSpec, VLMInputPolicy, VLMGenerationPolicy
from clasharness.scenario.vlm_materializer import (
    _chart_spec_for_asset,
    _generate_telemetry_noise,
    _process_caption_and_template,
    _template_styling_for_asset,
)
from clasharness.scenario import design_archetypes as da


def test_presets_integrity() -> None:
    # UT-PRESETS-01
    assert "preset_multi_line_chart" in CHART_PRESETS
    preset = CHART_PRESETS["preset_multi_line_chart"]
    assert isinstance(preset, ChartPreset)
    assert preset.chart_type == "multi_line_chart"
    assert preset.series_names == ["External Ingress", "Internal Healthchecks"]
    assert preset.default_styling == {"theme": "cool", "show_grid": True}

    assert "operational_scene_update" in CAPTION_TEMPLATES
    caption = CAPTION_TEMPLATES["operational_scene_update"]
    assert isinstance(caption, CaptionTemplate)
    assert "required_checklist" in dir(caption)
    assert len(caption.required_checklist) == 3


def test_chart_spec_for_asset_merges_presets_and_overrides() -> None:
    # UT-MERGE-01
    dummy_spec = VLMScenarioSpec(
        series="pro_vlm_v1",
        scenario_slug="test_slug",
        title="Default Spec Title",
        domain="test",
        core_conflict="conflict",
        modalities=[],
        input_policy=VLMInputPolicy(),
        generation_policy=VLMGenerationPolicy(),
        allowed_tool_families=[],
        assets=[],
        rounds=[],
        evals=[],
    )

    asset = VLMAssetSpec(
        id="test_asset_01",
        kind="chart",
        generator="template_chart",
        path="charts/test.png",
        mime_type="image/png",
        available_from_round=1,
        anchor_ids=["EVIDENCE:1"],
        preset_id="preset_multi_line_chart",
        chart_rows=[{"label": "10:00", "values": [1200, 20]}],
        custom_overrides={
            "title": "Overridden Chart Title",
            "y_label": "Overridden Y-Label",
            "styling": {"theme": "warm", "custom_key": "custom_val"}
        }
    )

    spec = _chart_spec_for_asset(asset, dummy_spec)

    # Overridden values
    assert spec["title"] == "Overridden Chart Title"
    assert spec["y_label"] == "Overridden Y-Label"
    # Merged styling: theme overridden, show_grid preserved, custom_key added
    assert spec["styling"] == {"theme": "warm", "show_grid": True, "custom_key": "custom_val"}
    # Preset default preserved values
    assert spec["chart_type"] == "multi_line_chart"
    assert spec["series_names"] == ["External Ingress", "Internal Healthchecks"]
    assert spec["rows"] == [{"label": "10:00", "values": [1200, 20]}]


def test_chart_spec_raises_on_unknown_preset() -> None:
    # UT-VALIDATE-01
    dummy_spec = VLMScenarioSpec(
        series="pro_vlm_v1",
        scenario_slug="test_slug",
        title="Default Spec Title",
        domain="test",
        core_conflict="conflict",
        modalities=[],
        input_policy=VLMInputPolicy(),
        generation_policy=VLMGenerationPolicy(),
        allowed_tool_families=[],
        assets=[],
        rounds=[],
        evals=[],
    )
    asset = VLMAssetSpec(
        id="test_asset_01",
        kind="chart",
        generator="template_chart",
        path="charts/test.png",
        mime_type="image/png",
        available_from_round=1,
        anchor_ids=["EVIDENCE:1"],
        preset_id="invalid_preset_id_foo",
    )

    with pytest.raises(ValueError, match="Unknown chart preset ID: invalid_preset_id_foo"):
        _chart_spec_for_asset(asset, dummy_spec)


def test_process_caption_and_template_formats_correctly() -> None:
    # UT-CAPTION-01
    caption_data = {
        "service_name": "Ingress API",
        "visual_anomaly_details": "a major traffic drop to zero at 10:05",
        "anchors": "EVIDENCE:INGRESS_DOWN"
    }
    custom_overrides = {"caption_template_id": "operational_scene_update"}
    anchor_ids = ["EVIDENCE:INGRESS_DOWN"]

    caption = _process_caption_and_template(caption_data, anchor_ids, custom_overrides)

    # This caption template is intentionally modality-neutral: it is shared by
    # charts, physical natural_image photos, and schematic diagrams, so it must
    # NOT presuppose a "status board" / "the display shows" screen framing.
    expected = (
        "Operational evidence for Ingress API: "
        "a major traffic drop to zero at 10:05. "
        "Visual anchors EVIDENCE:INGRESS_DOWN confirm the current state."
    )
    assert caption == expected


def test_process_caption_and_template_raises_on_missing_key() -> None:
    # UT-CHECKLIST-01
    caption_data = {
        "service_name": "Ingress API",
        # Missing visual_anomaly_details
        "anchors": "EVIDENCE:INGRESS_DOWN"
    }
    custom_overrides = {"caption_template_id": "operational_scene_update"}
    anchor_ids = ["EVIDENCE:INGRESS_DOWN"]

    with pytest.raises(ValueError, match="Missing caption template key:"):
        _process_caption_and_template(caption_data, anchor_ids, custom_overrides)


# ---------------------------------------------------------------------------
# WD9 -- no-LLM fallback styling reaches chart_template.html (custom_overrides.styling)
# ---------------------------------------------------------------------------


def _mk_spec(title: str = "Prod Outage") -> VLMScenarioSpec:
    return VLMScenarioSpec(
        series="pro_vlm_v1", scenario_slug="s", title=title, domain="d",
        core_conflict="c", modalities=[], input_policy=VLMInputPolicy(),
        generation_policy=VLMGenerationPolicy(), allowed_tool_families=[],
        assets=[], rounds=[], evals=[],
    )


def _mk_asset(**kw: Any) -> VLMAssetSpec:
    base: dict[str, Any] = dict(
        id="a1", kind="chart", generator="template_chart", path="charts/x.png",
        mime_type="image/png", available_from_round=1, anchor_ids=["EVIDENCE:1"],
    )
    base.update(kw)
    return VLMAssetSpec(**base)


def _assert_template_valid(styling: dict) -> None:
    assert styling["theme"] in da.COLOR_THEMES
    assert styling["primary_color"] in da.PRIMARY_COLORS
    assert da.has_archetype(styling["visual_style"])
    assert isinstance(styling["style_details"], dict) and styling["style_details"]
    # Signature (non-varying) axes must match the resolved archetype's preset,
    # i.e. the tokens are coherent with the named visual_style.
    arch = da.get_archetype(styling["visual_style"])
    for axis in da.STYLE_DETAIL_AXES:
        if axis not in arch.variation_ranges:
            assert styling["style_details"][axis] == arch.token_preset[axis]


def test_wd9_preset_asset_gets_coherent_template_styling() -> None:
    # UT-WD9-01: the no-LLM preset path attaches styling where chart_template.html
    # actually reads it (custom_overrides.styling), sourced from the SSoT library.
    sp = _mk_spec()
    asset = _mk_asset(
        preset_id="preset_multi_line_chart", platform_id="api-gw",
        chart_rows=[{"label": "10:00", "values": [1, 2]}],
    )
    cs = _chart_spec_for_asset(asset, sp)
    assert "custom_overrides" in cs and "styling" in cs["custom_overrides"]
    _assert_template_valid(cs["custom_overrides"]["styling"])
    # The preset's matplotlib-fallback styling stays at the top level, untouched.
    assert cs["styling"] == {"theme": "cool", "show_grid": True}


def test_wd9_non_preset_fallback_also_attaches_styling() -> None:
    # UT-WD9-02: the no-preset fallback branch attaches styling too (previously
    # it emitted no styling at all).
    sp = _mk_spec()
    cs = _chart_spec_for_asset(_mk_asset(platform_id="p5", kind="diagram"), sp)
    assert "custom_overrides" in cs and "styling" in cs["custom_overrides"]
    _assert_template_valid(cs["custom_overrides"]["styling"])


def test_wd9_matplotlib_theme_hint_does_not_leak() -> None:
    # UT-WD9-03: preset 'theme' ('cool'/'warm') is a matplotlib palette hint and
    # must never become the template's light/dark theme.
    sp = _mk_spec()
    asset = _mk_asset(
        preset_id="preset_multi_line_chart", platform_id="api-gw",
        chart_rows=[{"label": "x", "values": [1]}],
        custom_overrides={"styling": {"theme": "warm", "show_grid": True}},
    )
    styling = _chart_spec_for_asset(asset, sp)["custom_overrides"]["styling"]
    assert styling["theme"] in da.COLOR_THEMES
    assert styling["theme"] != "warm"


def test_wd9_styling_is_per_platform_deterministic() -> None:
    # UT-WD9-04: same platform -> identical styling (one look per product);
    # deterministic across calls.
    sp = _mk_spec()
    s1 = _template_styling_for_asset(_mk_asset(id="b1", platform_id="db-01"), sp)
    s2 = _template_styling_for_asset(_mk_asset(id="b2", platform_id="db-01"), sp)
    assert s1 == s2
    # A repeat call reproduces exactly.
    assert _template_styling_for_asset(_mk_asset(id="b3", platform_id="db-01"), sp) == s1


def test_wd9_explicit_visual_style_anchors_coherent_tokens() -> None:
    # UT-WD9-05: an explicitly named, valid visual_style keeps its own coherent
    # token preset -- never mixed with another archetype's tokens.
    sp = _mk_spec()
    asset = _mk_asset(
        platform_id="p", custom_overrides={"styling": {"visual_style": "terminal_tui"}},
    )
    styling = _template_styling_for_asset(asset, sp)
    assert styling["visual_style"] == "terminal_tui"
    tt = da.get_archetype("terminal_tui")
    assert styling["style_details"]["font_family"] == tt.token_preset["font_family"]
    assert styling["theme"] == tt.default_theme
    _assert_template_valid(styling)


def test_wd9_full_archetype_styling_is_honored() -> None:
    # UT-WD9-06: a complete, template-valid styling (e.g. from the LLM design
    # path) is passed through unchanged.
    sp = _mk_spec()
    explicit = {
        "theme": "dark",
        "primary_color": "cyan",
        "visual_style": "datadog_dashboard",
        "style_details": da.get_archetype("datadog_dashboard").default_style_details(),
    }
    asset = _mk_asset(platform_id="p", custom_overrides={"styling": dict(explicit)})
    styling = _template_styling_for_asset(asset, sp)
    assert styling["theme"] == "dark"
    assert styling["primary_color"] == "cyan"
    assert styling["visual_style"] == "datadog_dashboard"
    assert styling["style_details"] == explicit["style_details"]


# ---------------------------------------------------------------------------
# WD5 -- reproducibility hardening: telemetry noise + styling seeded by style_seed
# ---------------------------------------------------------------------------


def test_wd5_telemetry_noise_is_deterministic_for_pinned_seed() -> None:
    # UT-WD5-01: a pinned (title, asset_id, style_seed) yields byte-identical
    # noise across calls. Historically _generate_telemetry_noise used the
    # unseeded module-global RNG, so a pinned seed still jittered.
    sp = _mk_spec(title="Prod Outage")
    n1 = _generate_telemetry_noise(sp, "chart_cpu", style_seed=42)
    n2 = _generate_telemetry_noise(sp, "chart_cpu", style_seed=42)
    assert n1 == n2
    # The randomized metrics are actually populated (not a constant stub).
    assert n1["metrics"] and all("sparkline" in m for m in n1["metrics"])


def test_wd5_telemetry_noise_varies_with_seed_and_asset() -> None:
    # UT-WD5-02: the seed and asset_id are real knobs -- different seeds (and
    # different assets) produce different noise, so pinning is meaningful.
    sp = _mk_spec(title="Prod Outage")
    variants = {
        tuple(m["value"] for m in _generate_telemetry_noise(sp, "chart_cpu", style_seed=s)["metrics"])
        for s in range(8)
    }
    assert len(variants) > 1
    # Distinct assets under the same seed also diverge.
    a = _generate_telemetry_noise(sp, "chart_a", style_seed=0)
    b = _generate_telemetry_noise(sp, "chart_b", style_seed=0)
    assert a != b


def test_wd5_telemetry_noise_is_cross_process_stable() -> None:
    # UT-WD5-03: string-seeded RNG (version-2 seeding) is PYTHONHASHSEED-
    # independent, so a pinned seed reproduces byte-identically in a fresh
    # interpreter -- the WD5 exit-gate guarantee.
    import json
    import subprocess
    import sys

    in_proc = _generate_telemetry_noise(_mk_spec(title="Prod Outage"), "chart_cpu", style_seed=7)
    prog = (
        "import json;"
        "from clasharness.scenario.vlm_materializer import _generate_telemetry_noise;"
        "from tests.test_vlm_presets_overrides import _mk_spec;"
        "print(json.dumps(_generate_telemetry_noise(_mk_spec(title='Prod Outage'), 'chart_cpu', style_seed=7)))"
    )
    out = subprocess.check_output(
        [sys.executable, "-c", prog], env={**__import__("os").environ, "PYTHONHASHSEED": "1"}
    )
    assert json.loads(out) == in_proc


def test_wd5_template_styling_honors_style_seed() -> None:
    # UT-WD5-04: _template_styling_for_asset is keyed on style_seed and matches
    # the exact key the LLM design path uses (title, platform_id, style_seed),
    # so the no-LLM fallback and the design path agree for a pinned seed.
    sp = _mk_spec(title="T")
    got = _template_styling_for_asset(_mk_asset(platform_id="pid-x"), sp, style_seed=7)
    want = da.sample_platform_design("T_pid-x_7")
    assert got["theme"] == want["color_theme"]
    assert got["primary_color"] == want["primary_color"]
    assert got["visual_style"] == want["visual_style"]
    assert got["style_details"] == want["style_details"]
    # Same seed reproduces; the seed is a real knob (>=2 looks across seeds).
    assert _template_styling_for_asset(_mk_asset(platform_id="pid-x"), sp, style_seed=7) == got
    looks = {
        _template_styling_for_asset(_mk_asset(platform_id="pid-x"), sp, style_seed=s)["visual_style"]
        + _template_styling_for_asset(_mk_asset(platform_id="pid-x"), sp, style_seed=s)["theme"]
        for s in range(12)
    }
    assert len(looks) > 1


def test_wd5_chart_spec_threads_seed_to_noise_and_styling() -> None:
    # UT-WD5-05: _chart_spec_for_asset forwards style_seed to BOTH the telemetry
    # noise and the template styling, so a pinned seed reproduces the whole chart.
    sp = _mk_spec(title="Prod Outage")

    def _cs(seed: int) -> dict:
        asset = _mk_asset(
            platform_id="mon-1",
            provider_metadata={"layout": "devops_dashboard"},
        )
        return _chart_spec_for_asset(asset, sp, style_seed=seed)

    cs_a1 = _cs(3)
    cs_a2 = _cs(3)
    assert cs_a1["dashboard_data"] == cs_a2["dashboard_data"]
    assert cs_a1["custom_overrides"]["styling"] == cs_a2["custom_overrides"]["styling"]
    _assert_template_valid(cs_a1["custom_overrides"]["styling"])
    # The styling matches what the seeded helper would produce directly.
    assert cs_a1["custom_overrides"]["styling"] == _template_styling_for_asset(
        _mk_asset(platform_id="mon-1"), sp, style_seed=3
    )
