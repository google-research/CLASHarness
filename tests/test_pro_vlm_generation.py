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

import base64
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

# Make the checkout importable when run as a plain script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# When pytest is unavailable (running this module as a plain script), fall back
# to a minimal shim providing `pytest.raises`. IMPORTANT: never overwrite a real,
# installed pytest in sys.modules — doing so corrupts global state for every
# other test collected afterwards (e.g. `pytest.mark` would disappear).
try:
  import pytest  # noqa: F401
except ImportError:
  dummy_pytest = ModuleType("pytest")

  class PytestRaisesContext:
    def __init__(self, expected_exception, match=None):
      self.expected_exception = expected_exception
      self.match = match
    def __enter__(self):
      return self
    def __exit__(self, exc_type, exc_val, exc_tb):
      if exc_type is None:
        raise AssertionError(f"{self.expected_exception.__name__} not raised")
      if not issubclass(exc_type, self.expected_exception):
        return False
      if self.match:
        import re
        if not re.search(self.match, str(exc_val)):
          raise AssertionError(f"Pattern '{self.match}' not found in '{exc_val}'")
      return True

  def raises(expected_exception, match=None):
    return PytestRaisesContext(expected_exception, match=match)

  dummy_pytest.raises = raises
  sys.modules["pytest"] = dummy_pytest

from clasharness.core.task import Task, load_task, load_task_ids
from clasharness.scenario.spec import ScenarioSpec
from clasharness.scenario.strict_validator import strict_validate_scenario_package
from clasharness.scenario.vlm_asset_generators import CodeChartGenerator, MockImageGenerator, _extract_gemini_inline_image
from clasharness.scenario import vlm_materializer
from clasharness.scenario.vlm_materializer import build_vlm_scenario_spec, materialize_vlm_scenario
from clasharness.scenario.vlm_spec import VLMScenarioSpec, validate_vlm_scenario_spec

original_load_vlm_seed_file = vlm_materializer.load_vlm_seed_file
def patched_load_vlm_seed_file(path, *args, **kwargs):
  path_str = str(path)
  if "clasharness/data" in path_str:
    resolved_path = Path(__file__).parents[1] / path_str
    if resolved_path.exists():
      path = resolved_path
  return original_load_vlm_seed_file(path, *args, **kwargs)

vlm_materializer.load_vlm_seed_file = patched_load_vlm_seed_file
load_vlm_seed_file = patched_load_vlm_seed_file


def _repo_path(rel_path: str) -> str:
  base_path = Path(__file__).parents[1]
  p = base_path / rel_path
  if p.exists():
    return str(p)
  p_py = base_path / f"{rel_path}.py"
  if p_py.exists():
    return str(p_py)
  p_script = base_path / "scripts" / f"{rel_path}.py"
  if p_script.exists():
    return str(p_script)
  p_example = base_path / "examples" / f"{rel_path}.py"
  if p_example.exists():
    return str(p_example)
  return str(p)



class FakeVLMTextClient:
    def __init__(self) -> None:
        self.layers: list[str] = []

    def generate_layer(self, layer: str, payload: dict) -> dict:
        self.layers.append(layer)
        if layer == "vlm_asset_detail":
            slot = payload["slot"]
            anchors = " ".join(slot["anchor_ids"])
            return {
                "description": f"LLM refined {slot['asset_id']}",
                "caption": f"caption with {anchors}",
                "ocr_text": f"ocr with {anchors}",
                "prompt": "realistic operations photo with visible status board",
                "chart_rows": [{"label": "LLM row", "value": "blocked", "anchor": slot["anchor_ids"][0]}],
            }
        if layer == "vlm_round_question":
            return {"question": "Use the visible asset evidence to complete the required action without leaking EVIDENCE:BAD."}
        raise AssertionError(layer)


def test_vlm_spec_uses_text_llm_for_asset_and_round_details() -> None:
    seed = load_vlm_seed_file(Path("clasharness/data/scenario_seeds/pro_vlm_v1/visual_release_evidence_reconciliation.json"))
    client = FakeVLMTextClient()
    spec = build_vlm_scenario_spec(
        seed,
        provider="openai",
        image_provider="mock",
        chart_provider="template",
        google_model="gemini-3.1-flash-image",
        llm_client=client,
    )
    assert "vlm_asset_detail" in client.layers
    assert "vlm_round_question" in client.layers
    assert spec.assets[0].description.startswith("LLM refined")
    assert "EVIDENCE:BAD" not in spec.rounds[0].question
    assert "the relevant visible evidence" in spec.rounds[0].question

def test_vlm_spec_round_trip_and_validation(tmp_path: Path) -> None:
    seed = load_vlm_seed_file(Path("clasharness/data/scenario_seeds/pro_vlm_v1/visual_release_evidence_reconciliation.json"))
    spec = materialize_vlm_scenario(
        seed,
        scenario_series_dir=tmp_path / "scenarios",
        output_data_dir=tmp_path / "tasks",
    )
    vlm_spec_path = tmp_path / "scenarios" / "pro_vlm_workflow" / seed.slug / "vlm_spec.json"
    vlm_spec = VLMScenarioSpec.read_json(vlm_spec_path)

    assert vlm_spec.series == "pro_vlm_v1"
    assert vlm_spec.assets
    assert not validate_vlm_scenario_spec(vlm_spec)
    round_trip = VLMScenarioSpec.from_dict(vlm_spec.to_dict())
    assert round_trip.scenario_slug == seed.slug
    assert round_trip.assets[0].path.endswith(".png")

    # FS-DENSE-DATA-01 assertions:
    chart_assets = [a for a in vlm_spec.assets if a.kind == "chart"]
    assert chart_assets, "At least one chart asset must exist in vlm_spec.json"
    for chart in chart_assets:
        metadata = chart.provider_metadata or {}
        rows = metadata.get("chart_rows", [])
        assert len(rows) >= 5, f"Chart asset {chart.id} should have >= 5 chart rows, got {len(rows)}"
        assert chart.caption, f"Chart asset {chart.id} must have a non-empty caption"
        assert len(chart.caption) > 20, f"Chart asset {chart.id} caption is too short: '{chart.caption}'"


def test_manifest_entry_drops_non_dict_chart_rows() -> None:
    """Regression: an LLM may emit chart rows as positional arrays instead of
    keyed objects (e.g. ['03:00', 1210, 0.05, 'Operational']). Manifest
    serialization calls dict(row), which raised "dictionary update sequence
    element #0 has length 5; 2 is required" and aborted a whole scenario.
    manifest_entry must drop non-dict rows (matching from_dict) and never crash.
    """
    from clasharness.scenario.vlm_spec import VLMAssetSpec, VLMAssetManifest

    bad_rows = [["03:00", 1210, 0.05, "Operational"], ["04:00", 1230, 0.04, "Operational"]]
    good_rows = [{"label": "x", "value": 1.0, "anchor": ""}]
    asset = VLMAssetSpec(
        id="merchant_portal_devops_dashboard",
        kind="chart",
        generator="template",
        path="p.png",
        mime_type="image/png",
        available_from_round=1,
        anchor_ids=[],
        chart_rows=bad_rows + good_rows,
        provider_metadata={"chart_rows": bad_rows + good_rows},
    )

    entry = asset.manifest_entry(include_generation_metadata=True)
    assert entry["chart_rows"] == good_rows, "non-dict rows must be dropped"

    manifest = VLMAssetManifest([asset])
    # Both serializers must succeed without raising.
    assert len(manifest.to_package_dict()["assets"]) == 1
    assert len(manifest.to_runtime_dict()["assets"]) == 1

    # Serialize -> deserialize must be symmetric (from_dict also drops non-dicts).
    assert VLMAssetSpec.from_dict(entry).chart_rows == good_rows


def test_runtime_manifest_hides_answer_bearing_metadata() -> None:
    """A1 (difficulty-collapse fix): the RUNTIME manifest handed to the solving
    agent must NOT carry answer-bearing fields (description/caption/ocr_text/
    regions/chart_rows/custom_overrides/anchor_ids/prompt). Those survive only in
    the build-time PACKAGE manifest that feeds the renderer/checker. The rendered
    PNG bakes the values into pixels, so the agent loses no information -- it just
    has to LOOK instead of reading a leaked text side-channel.
    """
    from clasharness.scenario.vlm_spec import (
        VLMAssetSpec,
        VLMAssetManifest,
        VLMVisualRegion,
    )

    asset = VLMAssetSpec(
        id="release_board",
        kind="chart",
        generator="template",
        path="board.png",
        mime_type="image/png",
        available_from_round=1,
        anchor_ids=["EVIDENCE:DECISIVE"],
        description="The decisive answer is roll back to v4.2",
        caption="Release board caption mentioning v4.2",
        ocr_text="OCR: v4.2 FAILED",
        prompt="render a board showing v4.2 failing",
        regions=[
            VLMVisualRegion(
                id="r1",
                label="verdict: v4.2",
                bbox=[0.0, 0.0, 1.0, 1.0],
                anchor_ids=["EVIDENCE:DECISIVE"],
            )
        ],
        chart_rows=[{"label": "v4.2", "value": 1.0, "anchor": "EVIDENCE:DECISIVE"}],
        custom_overrides={"secret": "v4.2"},
    )

    answer_bearing = (
        "description",
        "caption",
        "ocr_text",
        "prompt",
        "regions",
        "chart_rows",
        "custom_overrides",
        "anchor_ids",
    )

    runtime = asset.manifest_entry(include_generation_metadata=False)
    for key in answer_bearing:
        assert key not in runtime, f"runtime manifest must not leak '{key}'"
    # Structural fields the agent legitimately needs are still present at runtime.
    assert runtime["id"] == "release_board"
    assert runtime["path"] == "board.png"
    assert runtime["kind"] == "chart"

    package = asset.manifest_entry(include_generation_metadata=True)
    for key in answer_bearing:
        assert key in package, f"package manifest must retain '{key}' for build/render"
    assert package["description"] == "The decisive answer is roll back to v4.2"
    assert package["chart_rows"] == [
        {"label": "v4.2", "value": 1.0, "anchor": "EVIDENCE:DECISIVE"}
    ]

    # The manifest-level serializers mirror the per-asset split.
    manifest = VLMAssetManifest([asset])
    runtime_asset = manifest.to_runtime_dict()["assets"][0]
    package_asset = manifest.to_package_dict()["assets"][0]
    for key in answer_bearing:
        assert key not in runtime_asset
        assert key in package_asset


def test_asset_generators_mock_chart_code_and_google_response_parse() -> None:
    chart_gen = CodeChartGenerator()
    chart = chart_gen.render_chart_spec(
        {
            "title": "Chart",
            "subtitle": "Evidence",
            "rows": [{"label": "Gate", "value": "blocked", "anchor": "EVIDENCE:X"}],
        }
    )
    assert chart.mime_type == "image/png"
    assert chart.data.startswith(b"\x89PNG")

    # WD6 render floor: >=16x16, >1 distinct color, >=0.2% ink. A solid fill of
    # ANY size reads as "blank", so this fixture needs a background + a block.
    rendered = chart_gen.render_python_code(
        """
from PIL import Image
img = Image.new('RGB', (64, 64), color=(255, 255, 255))
img.paste((255, 0, 0), (8, 8, 40, 40))
img.save(OUTPUT_PATH)
"""
    )
    assert rendered.data.startswith(b"\x89PNG")

    mock = MockImageGenerator().generate(prompt="scene", caption="caption", anchors=["UPDATE:X"])
    assert mock.data.startswith(b"\x89PNG")

    raw = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": base64.b64encode(b"image-bytes").decode("ascii"),
                            }
                        }
                    ]
                }
            }
        ]
    }
    data, mime = _extract_gemini_inline_image(raw)
    assert data == b"image-bytes"
    assert mime == "image/png"


def test_asset_generators_security_and_timeout() -> None:
    import pytest
    chart_gen = CodeChartGenerator()

    # Verify blocked call: open()
    with pytest.raises(ValueError, match="render code call is not allowed: open"):
        chart_gen.render_python_code(
            "open('test.txt', 'w')\n"
            "import matplotlib.pyplot as plt\n"
            "plt.figure().savefig(OUTPUT_PATH)\n"
        )

    # Verify blocked attribute call: Path.write_text()
    with pytest.raises(ValueError, match="render code call is not allowed: .write_text"):
        chart_gen.render_python_code(
            "from pathlib import Path\n"
            "Path('test.txt').write_text('hacked')\n"
            "import matplotlib.pyplot as plt\n"
            "plt.figure().savefig(OUTPUT_PATH)\n"
        )

    # Verify blocked import: os
    with pytest.raises(ValueError, match="render code import is not allowed: os"):
        chart_gen.render_python_code(
            "import os\n"
            "import matplotlib.pyplot as plt\n"
            "plt.figure().savefig(OUTPUT_PATH)\n"
        )

    # Verify timeout handling
    with pytest.raises(RuntimeError, match="chart render timed out"):
        chart_gen.render_python_code(
            "while True:\n"
            "    pass\n",
            timeout_seconds=1
        )


def test_new_chart_types_rendering() -> None:
    chart_gen = CodeChartGenerator()
    
    # 1. Pie Chart
    pie_chart = chart_gen.render_chart_spec({
        "title": "Pie Chart Test",
        "chart_type": "pie_chart",
        "data": [
            {"label": "A", "value": 30},
            {"label": "B", "value": 70}
        ]
    })
    assert pie_chart.mime_type == "image/png"
    assert pie_chart.data.startswith(b"\x89PNG")
    
    # 2. Stacked Bar Chart
    stacked_chart = chart_gen.render_chart_spec({
        "title": "Stacked Bar Test",
        "chart_type": "stacked_bar_chart",
        "series_names": ["S1", "S2"],
        "data": [
            {"label": "Day 1", "values": [10, 20]},
            {"label": "Day 2", "values": [15, 25]}
        ]
    })
    assert stacked_chart.mime_type == "image/png"
    assert stacked_chart.data.startswith(b"\x89PNG")

    # 3. Scatter Plot
    scatter_plot = chart_gen.render_chart_spec({
        "title": "Scatter Plot Test",
        "chart_type": "scatter_plot",
        "x_label": "Hours",
        "y_label": "Errors",
        "data": [
            {"x": 1.0, "y": 5.0, "label": "P1"},
            {"x": 2.0, "y": 8.0, "label": "P2"}
        ]
    })
    assert scatter_plot.mime_type == "image/png"
    assert scatter_plot.data.startswith(b"\x89PNG")

    # 4. Dashboard
    dashboard = chart_gen.render_chart_spec({
        "title": "System Dashboard",
        "chart_type": "dashboard",
        "left_chart": {
            "title": "CPUs",
            "chart_type": "bar_chart",
            "data": [{"label": "CPU0", "value": 85}, {"label": "CPU1", "value": 90}]
        },
        "right_chart": {
            "title": "Network",
            "chart_type": "line_chart",
            "data": [{"label": "10h", "value": 50}, {"label": "11h", "value": 80}]
        }
    })
    assert dashboard.mime_type == "image/png"
    assert dashboard.data.startswith(b"\x89PNG")

def test_pro_vlm_materializer_writes_package_and_tasks(tmp_path: Path) -> None:
    seed = load_vlm_seed_file(Path("clasharness/data/scenario_seeds/pro_vlm_v1/visual_release_evidence_reconciliation.json"))
    spec = materialize_vlm_scenario(
        seed,
        scenario_series_dir=tmp_path / "scenarios",
        output_data_dir=tmp_path / "tasks",
    )
    scenario_dir = tmp_path / "scenarios" / "pro_vlm_workflow" / seed.slug

    assert (scenario_dir / "assets").is_dir()
    assert (scenario_dir / "asset_manifest.json").exists()
    assert (scenario_dir / "vlm_spec.json").exists()
    strict_validate_scenario_package(spec, scenario_dir).raise_for_errors()

    task_ids = load_task_ids("total", data_dir=tmp_path / "tasks")
    assert len(task_ids) == 24
    q01 = load_task(task_ids[0], data_dir=tmp_path / "tasks")
    overrides = q01.data.private["initial_state_overrides"]
    assert "asset_manifest.json" in overrides
    assert "assets/scene_update.png" not in overrides
    assert any(key.startswith("assets/") and isinstance(value, dict) and "__bytes_b64__" in value for key, value in overrides.items())
    assert "asset" in q01.domains
    assert q01.ground_truth is not None
    assert any(cmd.startswith("asset read-text") for cmd in q01.ground_truth.solution_commands)
    Task.from_dict(q01.to_dict())


def test_generate_vlm_assets_resume_cache_hit_rehydrates_content_b64(tmp_path: Path) -> None:
    # Fix #2 regression: vlm_spec.json is checkpointed with
    # include_asset_content=False, so content_b64 is stripped to keep it small. On
    # resume, a cache-hit render MUST re-hydrate the rendered bytes from disk;
    # otherwise the resumed asset carries an empty content_b64 and the task's
    # data.private.initial_state_overrides embeds an empty asset payload.
    seed = load_vlm_seed_file(Path("clasharness/data/scenario_seeds/pro_vlm_v1/visual_release_evidence_reconciliation.json"))
    materialize_vlm_scenario(
        seed,
        scenario_series_dir=tmp_path / "scenarios",
        output_data_dir=tmp_path / "tasks",
    )
    scenario_dir = tmp_path / "scenarios" / "pro_vlm_workflow" / seed.slug

    # Simulate the resume load: read the checkpoint that omits content_b64.
    resumed = VLMScenarioSpec.read_json(scenario_dir / "vlm_spec.json")
    assert resumed.assets, "expected assets in resumed spec"
    assert all(a.content_b64 == "" for a in resumed.assets), (
        "precondition: vlm_spec.json checkpoint must strip content_b64"
    )

    # Re-run asset generation against the already-populated scenario_dir. Every
    # asset should hit the on-disk cache instead of re-rendering.
    msgs: list[str] = []
    rehydrated = vlm_materializer.generate_vlm_assets(
        resumed,
        scenario_dir=scenario_dir,
        image_provider="mock",
        chart_provider="template",
        google_model="gemini-3.1-flash-image",
        google_api_key_env="GOOGLE_API_KEY",
        google_base_url="https://example.invalid",
        raw_response_dir=None,
        progress=msgs.append,
    )

    assert any("[CACHE HIT]" in m for m in msgs), (
        "expected the resume to take the on-disk cache-hit path"
    )

    restored = 0
    for asset in rehydrated.assets:
        disk = scenario_dir / "assets" / asset.path
        if not disk.is_file():
            continue
        restored += 1
        assert asset.content_b64, f"content_b64 not re-hydrated for {asset.id}"
        assert base64.b64decode(asset.content_b64) == disk.read_bytes(), (
            f"re-hydrated bytes differ from disk for {asset.id}"
        )
        assert asset.mime_type, f"mime_type missing for {asset.id}"
        assert asset.dimensions.get("width") and asset.dimensions.get("height"), (
            f"dimensions missing for {asset.id}"
        )
    assert restored > 0, "expected at least one cache-hit asset with a file on disk"


def test_generate_vlm_assets_cache_hit_richtext_google_image_uses_code_chart(tmp_path: Path) -> None:
    # Fix #3 regression: a chart/rich_text asset whose generator is "google_image"
    # is rendered through the chart path (its kind promotes the effective generator
    # to *_chart), so it lands on disk with a "code_chart" provider. The cache-check
    # must expect "code_chart" for it - not the image provider - otherwise it never
    # cache-hits and is needlessly re-rendered (new bytes, LLM cost) on every resume.
    import json as _json
    from clasharness.scenario.vlm_spec import (
        VLMAssetSpec,
        VLMInputPolicy,
        VLMGenerationPolicy,
    )

    # A valid 1x1 PNG so _asset_dimensions (PIL) also succeeds.
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    assets_dir = tmp_path / "assets" / "status_page"
    assets_dir.mkdir(parents=True)
    (assets_dir / "alert.png").write_bytes(png)
    # Metadata written by the original chart render: provider == "code_chart".
    (tmp_path / "assets" / "status_alert.metadata.json").write_text(
        _json.dumps({
            "id": "status_alert",
            "generator": "google_image",
            "provider_metadata": {"provider": "code_chart"},
            "caption": "cached caption",
            "prompt": "cached prompt",
        }),
        encoding="utf-8",
    )

    asset = VLMAssetSpec(
        id="status_alert",
        kind="rich_text",
        generator="google_image",
        path="status_page/alert.png",
        mime_type="image/png",
        available_from_round=1,
        anchor_ids=[],
    )
    spec = VLMScenarioSpec(
        series="pro_vlm_v1",
        scenario_slug="fix3_probe",
        title="Fix3 Probe",
        domain="ops",
        core_conflict="x",
        modalities=["vision"],
        input_policy=VLMInputPolicy.from_dict(None),
        generation_policy=VLMGenerationPolicy.from_dict(None),
        allowed_tool_families=[],
        assets=[asset],
        rounds=[],
        evals=[],
    )

    msgs: list[str] = []
    out = vlm_materializer.generate_vlm_assets(
        spec,
        scenario_dir=tmp_path,
        image_provider="vertex",   # would be used to build a Vertex image client,
        chart_provider="template", # but a correct cache-hit must never call it.
        google_model="gemini-3.1-flash-image",
        google_api_key_env="GOOGLE_API_KEY",
        google_base_url="https://example.invalid",
        raw_response_dir=None,
        llm_client=None,
        progress=msgs.append,
    )

    assert any("[CACHE HIT] Asset status_alert" in m for m in msgs), (
        f"rich_text+google_image asset should cache-hit as code_chart, got: {msgs}"
    )
    assert not any("Rendering visual asset" in m for m in msgs), (
        f"asset must not be re-rendered on cache-hit, got: {msgs}"
    )
    (rehydrated,) = out.assets
    assert base64.b64decode(rehydrated.content_b64) == png


def test_pro_vlm_binary_assets_build_direct_image_parts(tmp_path: Path) -> None:
    from examples import train_and_eval

    seed = load_vlm_seed_file(Path("clasharness/data/scenario_seeds/pro_vlm_v1/visual_release_evidence_reconciliation.json"))
    materialize_vlm_scenario(
        seed,
        scenario_series_dir=tmp_path / "scenarios",
        output_data_dir=tmp_path / "tasks",
    )
    task = load_task("pro_vlm_visual_release_evidence_reconciliation_q01", data_dir=tmp_path / "tasks")
    parts = train_and_eval._multimodal_image_parts_from_task(task, mode="direct")
    assert parts
    assert parts[0]["image_url"]["url"].startswith("data:image/png;base64,")

def test_pro_vlm_mock_cli_compile_existing_and_expert_smoke(tmp_path: Path) -> None:
    scenario_dir = tmp_path / "scenarios"
    task_dir = tmp_path / "tasks"

    bin_path = _repo_path("generate_pro_vlm_scenarios")
    cmd1 = [bin_path]
    if bin_path.endswith(".py"):
      cmd1 = [sys.executable, bin_path]

    res = subprocess.run(
        cmd1 + [
            "--provider",
            "mock",
            "--image-provider",
            "mock",
            "--chart-provider",
            "template",
            "--seed-dir",
            _repo_path("clasharness/data/scenario_seeds/pro_vlm_v1"),
            "--output-scenario-dir",
            str(scenario_dir),
            "--output-data-dir",
            str(task_dir),
            "--target-rounds",
            "24",
        ],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
      raise AssertionError(f"generate_pro_vlm_scenarios failed with code {res.returncode}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
    spec = ScenarioSpec.read_json(
        scenario_dir
        / "pro_vlm_workflow"
        / "visual_release_evidence_reconciliation"
        / "scenario.json"
    )
    assert len(spec.rounds) == 24

    compiled_dir = tmp_path / "compiled"

    cmd2 = [bin_path]
    if bin_path.endswith(".py"):
      cmd2 = [sys.executable, bin_path]

    subprocess.run(
        cmd2 + [
            "--compile-existing-scenarios",
            "--output-scenario-dir",
            str(scenario_dir),
            "--output-data-dir",
            str(compiled_dir),
        ],
        check=True,
    )

    eval_bin = _repo_path("train_and_eval")
    cmd3 = [eval_bin]
    if eval_bin.endswith(".py"):
      cmd3 = [sys.executable, eval_bin]

    result = subprocess.run(
        cmd3 + [
            "--agent",
            "expert",
            "--task-data-dir",
            str(compiled_dir),
            "--split",
            "total",
            "--mode",
            "martin",
            "--max-steps",
            "25",
            "--verbose",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if "Tasks : 24/24 passed" not in result.stdout:
      raise AssertionError(f"Expert agent evaluation failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")


def test_render_chart_matplotlib_fallback():
    """FS-FALLBACK-01: Mock shutil.which to return None (Chrome unavailable). Verify fallback to Matplotlib."""
    import os
    import shutil
    import tempfile
    from unittest import mock
    from clasharness.scenario.vlm_asset_generators import render_chart_spec

    spec = {
        "title": "Test Fallback Chart",
        "chart_type": "line",
        "data": {
            "labels": ["A", "B", "C"],
            "datasets": [{"label": "Metric", "data": [10, 20, 15]}],
        },
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "chart.png")
        with mock.patch("shutil.which", return_value=None):
            render_chart_spec(spec, out_path)
        assert os.path.exists(out_path)
        with open(out_path, "rb") as f:
            header = f.read(4)
        assert header == b"\x89PNG"


def test_render_chart_race_animation_disabled():
    """FS-RACE-01: Verify animation: false or duration: 0 is in the Chart.js config to prevent blank screenshots."""
    from pathlib import Path
    template_path = (
        Path(__file__).parent.parent
        / "clasharness"
        / "scenario"
        / "templates"
        / "chart_template.html"
    )
    assert template_path.exists(), f"Template not found at: {template_path}"
    content = template_path.read_text()
    assert "animation: false" in content or "duration: 0" in content


def test_render_chart_cleanup():
    """FS-CLEANUP-01: Run rendering and verify no leaked temp HTML files or dirs."""
    import os
    import tempfile
    import glob
    from clasharness.scenario.vlm_asset_generators import render_chart_spec

    spec = {
        "title": "Cleanup Test",
        "chart_type": "line",
        "data": {
            "labels": ["A", "B"],
            "datasets": [{"label": "Metric", "data": [1, 2]}],
        },
    }
    temp_dir = tempfile.gettempdir()
    before_files = set(glob.glob(os.path.join(temp_dir, "*")))
    
    with tempfile.TemporaryDirectory() as out_dir:
        out_path = os.path.join(out_dir, "chart.png")
        render_chart_spec(spec, out_path)
        
    after_files = set(glob.glob(os.path.join(temp_dir, "*")))
    leaked = after_files - before_files
    for path in leaked:
        assert "chart.html" not in path


def test_render_chart_chrome_failure_fallback():
    """FS-CHROME-ERR-01: Simulate Chrome crash or timeout. Verify Matplotlib fallback succeeds."""
    import os
    import shutil
    import subprocess
    import tempfile
    from unittest import mock
    from clasharness.scenario.vlm_asset_generators import render_chart_spec

    spec = {
        "title": "Chrome Error Fallback",
        "chart_type": "line",
        "data": {
            "labels": ["A", "B"],
            "datasets": [{"label": "Metric", "data": [1, 2]}],
        },
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "chart.png")
        with mock.patch("shutil.which", return_value="/usr/bin/google-chrome"), \
             mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=[], timeout=15)):
            render_chart_spec(spec, out_path)
            
        assert os.path.exists(out_path)
        with open(out_path, "rb") as f:
            header = f.read(4)
        assert header == b"\x89PNG"


def test_render_chart_edge_cases():
    """FS-EDGE-01: Verify generator handles empty rows, negative values, and unsupported types without crashing."""
    import os
    import tempfile
    from clasharness.scenario.vlm_asset_generators import render_chart_spec

    with tempfile.TemporaryDirectory() as tmpdir:
        # Empty data rows
        spec_empty = {
            "title": "Empty Data",
            "chart_type": "line",
            "data": {"labels": [], "datasets": []}
        }
        out_empty = os.path.join(tmpdir, "empty.png")
        render_chart_spec(spec_empty, out_empty)
        assert os.path.exists(out_empty)

        # Negative values
        spec_neg = {
            "title": "Negatives",
            "chart_type": "bar",
            "data": {
                "labels": ["A", "B"],
                "datasets": [{"label": "Metric", "data": [-10, -20]}],
            }
        }
        out_neg = os.path.join(tmpdir, "neg.png")
        render_chart_spec(spec_neg, out_neg)
        assert os.path.exists(out_neg)

        # Unsupported chart type spec
        spec_unsupported = {
            "title": "Unsupported",
            "chart_type": "unsupported_type",
            "data": {
                "labels": ["A", "B"],
                "datasets": [{"label": "Metric", "data": [1, 2]}],
            }
        }
        out_unsupported = os.path.join(tmpdir, "unsupported.png")
        render_chart_spec(spec_unsupported, out_unsupported)
        assert os.path.exists(out_unsupported)


def test_llm_dynamic_html_template_rendering(tmp_path: Path) -> None:
    from unittest.mock import MagicMock, patch
    from clasharness.scenario.vlm_materializer import materialize_vlm_scenario
    from clasharness.scenario.vlm_asset_generators import CodeChartGenerator, GeneratedAsset

    seed = load_vlm_seed_file(Path("clasharness/data/scenario_seeds/pro_vlm_v1/visual_release_evidence_reconciliation.json"))
    
    # Create mock LLM Client
    mock_llm = MagicMock()
    def side_effect(layer, payload=None):
        if layer == "generate_vlm_layout_spec":
            return {
                "ui_shell_type": "ide",
                "color_theme": "dark",
                "grid_layout": "sidebar_and_editor",
                "common_header_html": "<header>Mock Header</header>",
                "common_sidebar_html": "<aside>Mock Sidebar</aside>",
                "common_footer_html": "<footer>Mock Footer</footer>",
                "common_styles_css": "body { margin: 0; }"
            }
        elif layer == "generate_vlm_redundant_noise":
            return {
                "sidebar_content": "file_tree",
                "terminal_logs": "mock_git_status",
                "alerts_and_metrics": "no_alerts"
            }
        elif layer == "compose_vlm_html":
            return {
                "html_content": (
                    "<!DOCTYPE html><html><body>"
                    "<h1>Custom AWS Console Leak Demo</h1>"
                    "<p>Secret token anchor: ANCHOR_LEAK_XY1</p>"
                    "</body></html>"
                )
            }
        elif layer == "vlm_asset_detail":
            return {
                "description": "LLM refined description",
                "caption": "LLM refined caption",
                "chart_rows": [
                    {"label": "A", "value": 10},
                    {"label": "B", "value": 20},
                    {"label": "C", "value": 30},
                    {"label": "D", "value": 40},
                    {"label": "E", "value": 50}
                ]
            }
        return {}
    mock_llm.generate_layer.side_effect = side_effect

    with patch.object(CodeChartGenerator, "render_raw_html") as mock_render:
        mock_render.return_value = GeneratedAsset(
            data=b"\x89PNG dummy bytes",
            mime_type="image/png",
            extension="png",
            metadata={"renderer": "mocked_raw"}
        )
        spec = materialize_vlm_scenario(
            seed,
            scenario_series_dir=tmp_path / "scenarios",
            output_data_dir=tmp_path / "tasks",
            llm_client=mock_llm,
        )

    # Verify that mock_llm.generate_layer was called for the three multi-step layers
    called_layers = [call[0][0] for call in mock_llm.generate_layer.call_args_list]
    assert "generate_vlm_layout_spec" in called_layers
    assert "generate_vlm_redundant_noise" in called_layers
    assert "compose_vlm_html" in called_layers

    # Verify that the generated image file exists and has a PNG header
    vlm_spec_path = tmp_path / "scenarios" / "pro_vlm_workflow" / seed.slug / "vlm_spec.json"
    vlm_spec = VLMScenarioSpec.read_json(vlm_spec_path)
    
    # Check that the first template_chart asset is rendered and is a valid PNG
    rendered_image_path = tmp_path / "scenarios" / "pro_vlm_workflow" / seed.slug / "assets" / vlm_spec.assets[0].path
    assert rendered_image_path.exists()
    assert rendered_image_path.read_bytes().startswith(b"\x89PNG")


def test_safe_asset_relpath_sanitizes_unsafe_asset_ids():
    from clasharness.scenario.vlm_materializer import _safe_asset_relpath, _SAFE_PATH_RE

    # A ``:`` in a model-authored asset id must not hard-fail generation.
    out = _safe_asset_relpath("workspace:database_connections_chart.render.html")
    assert out == "workspace_database_connections_chart.render.html"
    assert _SAFE_PATH_RE.match(Path(out).name)

    # Safe names are left unchanged.
    assert _safe_asset_relpath("normal_asset.render.py") == "normal_asset.render.py"

    # Sub-paths keep their structure; path traversal segments are dropped and
    # every component ends up write-safe.
    out = _safe_asset_relpath("../evil/bad:name.html")
    assert ".." not in out.split("/")
    assert all(_SAFE_PATH_RE.match(part) for part in out.split("/"))


def test_write_generated_asset_contains_escaping_paths(tmp_path: Path) -> None:
    """A model-authored asset path must never escape ``<scenario>/assets/``.

    ``_write_generated_asset`` stripped only a literal ``assets/`` / ``/assets/``
    prefix, so any *other* absolute path (e.g. ``/lab_portal/charts/x.png``) fell
    through to ``scenario_dir / "assets" / rel_path`` -- and ``Path.__truediv__``
    *discards* the left operand when the right one is absolute. The write then
    landed wherever the model pointed, or died with ``PermissionError`` and took
    all 10 tasks of the scenario-version down with it (no retry). ``..`` segments
    escaped the same way, since ``Path`` does not normalize them.

    ``chart_spec_path`` is the quieter half of the same bug: it goes through
    ``_write_json``, which -- unlike ``_write_text`` -- has no ``_SAFE_PATH_RE``
    guard, so it escaped *silently* and left ``visualize.html`` pointing at a
    chart spec that is not in the package.

    ``render_source_path`` already called ``_safe_asset_relpath``; these two
    siblings must behave the same way.
    """
    from clasharness.scenario.vlm_asset_generators import GeneratedAsset
    from clasharness.scenario.vlm_materializer import _write_generated_asset
    from clasharness.scenario.vlm_spec import VLMAssetSpec

    scenario_dir = tmp_path / "scenario"
    scenario_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    assets_root = (scenario_dir / "assets").resolve()

    def _asset(path: str, **kwargs: object) -> VLMAssetSpec:
        return VLMAssetSpec(
            id="chart_a",
            kind="chart",
            generator="code_chart",
            path=path,
            mime_type="image/png",
            available_from_round=1,
            anchor_ids=[],
            **kwargs,
        )

    generated = GeneratedAsset(
        data=b"\x89PNG\r\n\x1a\n-not-a-real-png",
        mime_type="image/png",
        extension="png",
    )

    # 1. An absolute asset path is contained instead of written where it points.
    escaped = outside / "charts" / "escaped.png"
    updated = _write_generated_asset(scenario_dir, _asset(str(escaped)), generated)
    assert not escaped.exists(), f"asset escaped to {escaped}"
    assert not updated.path.startswith("/")
    written = (assets_root / updated.path).resolve()
    assert written.is_relative_to(assets_root), f"{written} escaped {assets_root}"
    assert written.read_bytes() == generated.data

    # 2. ``..`` traversal is dropped rather than resolved against the parent.
    updated = _write_generated_asset(scenario_dir, _asset("../../pwned.png"), generated)
    assert not (tmp_path / "pwned.png").exists(), "asset escaped via .. traversal"
    written = (assets_root / updated.path).resolve()
    assert written.is_relative_to(assets_root), f"{written} escaped {assets_root}"
    assert written.read_bytes() == generated.data

    # 3. ``chart_spec_path`` gets the same containment (it has no writer-side
    #    guard of its own, so an escape here is silent).
    spec_escape = outside / "specs" / "escaped.chart.json"
    generated_with_spec = GeneratedAsset(
        data=generated.data,
        mime_type="image/png",
        extension="png",
        chart_spec={"chart_type": "bar", "rows": []},
    )
    updated = _write_generated_asset(
        scenario_dir,
        _asset("charts/ok.png", chart_spec_path=str(spec_escape)),
        generated_with_spec,
    )
    assert not spec_escape.exists(), f"chart spec escaped to {spec_escape}"
    assert not updated.chart_spec_path.startswith("/")
    spec_written = (assets_root / updated.chart_spec_path).resolve()
    assert spec_written.is_relative_to(assets_root), f"{spec_written} escaped {assets_root}"
    assert json.loads(spec_written.read_text())["chart_type"] == "bar"

    # 4. Nothing at all landed outside the scenario package.
    assert sorted(q.name for q in outside.rglob("*")) == []


def test_diagram_sketch_type_model_split() -> None:
    # Redesign: ``diagram`` is now a DIGITAL schematic rendered as an in-UI HTML
    # module (like chart/rich_text); ``sketch`` is the new hand-drawn image kind.
    from clasharness.scenario import vlm_spec as V

    assert V.IMAGE_KINDS == frozenset({"natural_image", "sketch"})
    assert "diagram" not in V.IMAGE_KINDS
    assert V.IMAGE_GENERATORS == frozenset({"google_image", "sketch_image"})
    assert "diagram_image" not in V.IMAGE_GENERATORS
    assert V.default_generator_for_kind("natural_image") == "google_image"
    assert V.default_generator_for_kind("sketch") == "sketch_image"
    assert V.default_generator_for_kind("diagram") == "template_chart"
    assert V.default_generator_for_kind("chart") == ""
    # An explicit generator is always respected over the backfill.
    assert V.default_generator_for_kind("sketch", "custom_gen") == "custom_gen"


def test_from_dict_backfills_sketch_diagram_natural_generators() -> None:
    from clasharness.scenario.vlm_spec import VLMAssetSpec

    def mk(kind: str) -> VLMAssetSpec:
        return VLMAssetSpec.from_dict({
            "id": f"p1.{kind}", "kind": kind, "path": f"assets/{kind}.png",
            "mime_type": "image/png", "available_from_round": 2,
            "anchor_ids": ["EVIDENCE:x"],
        })

    # A bare asset (empty generator) must never fall through to the mock renderer.
    assert mk("sketch").generator == "sketch_image"
    assert mk("diagram").generator == "template_chart"
    assert mk("natural_image").generator == "google_image"


def test_validate_sketch_needs_prompt_diagram_validated_as_chart() -> None:
    from clasharness.scenario.vlm_spec import (
        VLMAssetSpec,
        VLMScenarioSpec,
        VLMInputPolicy,
        VLMGenerationPolicy,
        validate_vlm_scenario_spec,
    )

    def errs(**kw):
        kind = kw.pop("kind")
        asset = VLMAssetSpec.from_dict({
            "id": "p1.a", "kind": kind, "path": "assets/a.png",
            "mime_type": "image/png", "available_from_round": 2,
            "anchor_ids": ["EVIDENCE:x"], **kw,
        })
        spec = VLMScenarioSpec(
            series="conflict_vlm_v1", scenario_slug="s1", title="t", domain="ops",
            core_conflict="x", modalities=["vision"],
            input_policy=VLMInputPolicy.from_dict(None),
            generation_policy=VLMGenerationPolicy.from_dict(None),
            allowed_tool_families=[], assets=[asset], rounds=[], evals=[],
        )
        return [e for e in validate_vlm_scenario_spec(spec) if "p1.a" in e]

    # sketch is an image kind -> requires prompt + caption.
    assert any("requires prompt and caption" in e for e in errs(kind="sketch", caption="c"))
    assert not any(
        "requires prompt and caption" in e
        for e in errs(kind="sketch", prompt="a hand-drawn whiteboard sketch", caption="c")
    )
    # diagram is now a chart-like HTML kind: it must NOT require an image prompt,
    # but (like chart/rich_text) needs a chart_spec_path or render_source_path.
    diagram_errs = errs(kind="diagram", caption="c")
    assert not any("requires prompt and caption" in e for e in diagram_errs)
    assert any("chart asset requires" in e for e in diagram_errs)
    assert errs(kind="diagram", caption="c", chart_spec_path="assets/a.chart.json") == []


def test_generate_vlm_assets_routes_sketch_to_image_diagram_to_chart(tmp_path: Path) -> None:
    # sketch -> hand-drawn image model; diagram -> in-UI HTML/chart renderer.
    from unittest.mock import patch
    from clasharness.scenario.vlm_asset_generators import GeneratedAsset
    from clasharness.scenario.vlm_spec import (
        VLMAssetSpec,
        VLMScenarioSpec,
        VLMInputPolicy,
        VLMGenerationPolicy,
    )

    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    image_calls: list[str] = []
    chart_calls: list[str] = []

    def _gen(provider: str) -> GeneratedAsset:
        return GeneratedAsset(data=png, mime_type="image/png", extension="png",
                              metadata={"provider": provider})

    class RecImage:
        def generate(self, prompt, caption, anchors, **kw):
            image_calls.append(prompt)
            return _gen("mock_image")

    class RecChart:
        def render_chart_spec(self, spec):
            chart_calls.append("chart_spec")
            return _gen("code_chart")

        def render_text_spec(self, spec):
            chart_calls.append("text_spec")
            return _gen("code_chart")

        def render_python_code(self, code):
            chart_calls.append("python")
            return _gen("code_chart")

        def render_raw_html(self, html):
            chart_calls.append("raw_html")
            return _gen("code_chart")

    sketch = VLMAssetSpec(
        id="p1.sk", kind="sketch", generator="sketch_image", path="p1/sk.png",
        mime_type="image/png", available_from_round=1, anchor_ids=["EVIDENCE:s"],
        prompt="a topology schematic", caption="cap s",
    )
    diagram = VLMAssetSpec(
        id="p1.dg", kind="diagram", generator="template_chart", path="p1/dg.png",
        mime_type="image/png", available_from_round=1, anchor_ids=["EVIDENCE:d"],
        caption="cap d", chart_rows=[{"label": "node A", "value": "ok"}],
    )
    spec = VLMScenarioSpec(
        series="conflict_vlm_v1", scenario_slug="route_probe", title="t", domain="ops",
        core_conflict="x", modalities=["vision"],
        input_policy=VLMInputPolicy.from_dict(None),
        generation_policy=VLMGenerationPolicy.from_dict(None),
        allowed_tool_families=[], assets=[sketch, diagram], rounds=[], evals=[],
    )

    with patch.object(vlm_materializer, "MockImageGenerator", lambda: RecImage()), \
         patch.object(vlm_materializer, "CodeChartGenerator", lambda: RecChart()):
        vlm_materializer.generate_vlm_assets(
            spec,
            scenario_dir=tmp_path,
            image_provider="mock",
            chart_provider="template",
            google_model="m",
            google_api_key_env="GOOGLE_API_KEY",
            google_base_url="https://example.invalid",
            raw_response_dir=None,
            llm_client=None,
            progress=None,
        )

    # sketch went through the image model with a hand-drawn prompt suffix.
    assert len(image_calls) == 1, image_calls
    sk = image_calls[0].lower()
    assert "hand-drawn" in sk
    assert ("whiteboard" in sk or "paper" in sk)
    # diagram went through the chart/HTML renderer, never the image model.
    assert chart_calls, "diagram should render via the chart/HTML path"
    assert all("EVIDENCE:d" not in p for p in image_calls)


def test_compose_vlm_html_and_plan_prompts_cover_diagram_and_sketch() -> None:
    from clasharness.scenario.llm_client import _layer_contract

    compose = _layer_contract("compose_vlm_html")["instruction"]
    assert "inline SVG" in compose
    assert "diagram" in compose.lower()

    plan = _layer_contract("conflict_layer0c_assets_plan")["instruction"]
    assert "sketch" in plan.lower()
    assert "diagram" in plan.lower()


# ---------------------------------------------------------------------------
# S7: the asset refiner must not silently delete a round's decisive reading.
#
# ROOT CAUSE this pins down. ``ConflictLLMPipeline`` already GUARANTEES that a
# round's ``resolved_value`` appears verbatim in the asset named by its
# ``resolved_value_locus`` -- ``_validate_round_resolution`` rejects a resolution
# that is not grounded there, and the retry-exhausted fallback degrades to an
# empty value rather than an ungrounded one. But ``_generate_fresh_spec`` then
# calls ``llm_refine_vlm_assets``, which REWRITES ``caption``/``ocr_text``, and
# nothing re-checks grounding afterwards. The pipeline's guarantee is therefore
# undone underneath it: in a 523-round corpus, 84 rounds (16.1%) had a decisive
# value that appeared in NO asset and NO workspace file.
#
# The fix is two-part and deliberately NOT a rule-based patch-up:
#   Part A -- tell the refiner up front which strings must survive verbatim.
#   Part B -- if a rewrite drops one anyway, retry once with feedback; if the
#             retry also drops it, roll the two carrier surfaces back to the
#             pipeline-blessed text (a transaction abort, not a text splice).
# ---------------------------------------------------------------------------


class _ScriptedAssetClient:
  """Fake text client returning scripted ``vlm_asset_detail`` payloads.

  Responses are consumed in order; the final response repeats for any further
  call, so a test that scripts a single always-dropping response exercises both
  the first attempt and the retry.
  """

  def __init__(self, responses):
    self.responses = [dict(r) for r in responses]
    self.payloads = []

  def generate_layer(self, layer, payload):
    assert layer == "vlm_asset_detail", layer
    self.payloads.append(payload)
    idx = min(len(self.payloads) - 1, len(self.responses) - 1)
    return dict(self.responses[idx])

  @property
  def slots(self):
    return [p["slot"] for p in self.payloads]


def _s7_seed():
  return load_vlm_seed_file(
      Path("clasharness/data/scenario_seeds/pro_vlm_v1/visual_release_evidence_reconciliation.json")
  )


def _s7_asset(**overrides):
  """A carrier asset whose ORIGINAL text already holds the decisive reading.

  ``anchor_ids`` is empty on purpose so ``_ensure_anchors`` is a no-op and the
  assertions isolate S7's behaviour from the pre-existing anchor backfill.
  """
  from clasharness.scenario.vlm_spec import VLMAssetSpec

  base = dict(
      id="ops_chart",
      kind="chart",
      generator="template",
      path="charts/ops.png",
      mime_type="image/png",
      available_from_round=1,
      anchor_ids=[],
      description="original description",
      caption="Queue depth 402 rows; ticket TICKET-4471 pending; 12 retries.",
      ocr_text="TICKET-4471 | 402 | 12",
      prompt="original prompt",
  )
  base.update(overrides)
  return VLMAssetSpec(**base)


def _s7_refine(client, asset, carriers, **kwargs):
  return vlm_materializer.llm_refine_vlm_assets(
      _s7_seed(),
      assets=[asset],
      client=client,
      progress=None,
      num_generation_workers=1,
      decisive_values_by_asset=carriers,
      **kwargs,
  )[0]


def test_s7_refiner_is_told_which_readings_must_survive() -> None:
  """Part A: the refiner is INSTRUCTED to keep the reading, not just audited."""
  client = _ScriptedAssetClient([
      {"description": "d", "caption": "cap TICKET-4471 ok", "ocr_text": "TICKET-4471"}
  ])
  _s7_refine(client, _s7_asset(), {"ops_chart": ["TICKET-4471"]})

  assert len(client.slots) == 1, "a compliant rewrite must not trigger a retry"
  slot = client.slots[0]
  assert slot.get("must_carry_verbatim") == ["TICKET-4471"], slot
  blob = " ".join(slot["requirements"]).lower()
  assert "verbatim" in blob
  assert "TICKET-4471" in " ".join(slot["requirements"])


def test_s7_refiner_accepts_a_rewrite_that_keeps_the_reading() -> None:
  """A compliant rewrite is kept in full -- S7 must not cost us the refinement."""
  client = _ScriptedAssetClient([{
      "description": "refined description",
      "caption": "Throughput fell to 402 rows while TICKET-4471 stayed open.",
      "ocr_text": "rows 402 | ticket TICKET-4471 | retries 12",
      "prompt": "refined prompt",
  }])
  out = _s7_refine(client, _s7_asset(), {"ops_chart": ["TICKET-4471"]})

  assert out.description == "refined description"
  assert out.caption.startswith("Throughput fell")
  assert "TICKET-4471" in out.ocr_text


def test_s7_refiner_retries_when_the_decisive_reading_is_dropped() -> None:
  """Part B, first move: repair by asking the LLM again, with feedback."""
  client = _ScriptedAssetClient([
      {"description": "d1", "caption": "Throughput dipped overnight.", "ocr_text": "402 | 12"},
      {"description": "d2", "caption": "Throughput dipped; TICKET-4471 open.", "ocr_text": "402 | 12"},
  ])
  out = _s7_refine(client, _s7_asset(), {"ops_chart": ["TICKET-4471"]})

  assert len(client.slots) == 2, "a dropped reading must trigger exactly one retry"
  retry_slot = client.slots[1]
  feedback = str(retry_slot.get("retry_feedback") or "")
  assert "TICKET-4471" in feedback, retry_slot
  # The repaired rewrite is accepted, not rolled back.
  assert out.caption == "Throughput dipped; TICKET-4471 open."
  assert out.description == "d2"


def test_s7_refiner_rolls_back_carrier_text_when_the_retry_also_drops_it() -> None:
  """Part B, fallback: abort the rewrite of the two carrier surfaces.

  Rolling back to the pipeline-blessed caption/ocr_text restores exactly the
  state ``_validate_round_resolution`` accepted. We deliberately do NOT splice
  the value back into the new text: an appended "value: X" tail would be an
  un-buried giveaway and would defeat the difficulty the burial requirement
  exists to protect.
  """
  asset = _s7_asset()
  client = _ScriptedAssetClient([
      {"description": "refined description",
       "caption": "Throughput dipped overnight.",
       "ocr_text": "402 | 12",
       "prompt": "refined prompt"}
  ])
  out = _s7_refine(client, asset, {"ops_chart": ["TICKET-4471"]})

  assert len(client.slots) == 2, "rollback happens only after the retry is spent"
  assert out.caption == asset.caption
  assert out.ocr_text == asset.ocr_text
  # Everything that cannot destroy the reading keeps the refinement.
  assert out.description == "refined description"
  assert out.prompt == "refined prompt"


def test_s7_refiner_leaves_non_carrier_assets_untouched() -> None:
  """Assets that carry no decisive value are refined exactly as before.

  Only the locus asset is protected. Other assets are free to lose the value --
  measured difficulty RISES as the number of carriers falls (1 carrier scores
  0.63 vs 0.71-0.75 for 2-4), so protecting every asset would be a regression.
  """
  client = _ScriptedAssetClient([
      {"description": "d", "caption": "no reading here", "ocr_text": "nothing"}
  ])
  out = _s7_refine(client, _s7_asset(), {"some_other_asset": ["TICKET-4471"]})

  assert len(client.slots) == 1, "a non-carrier must never trigger a retry"
  assert "must_carry_verbatim" not in client.slots[0]
  assert out.caption == "no reading here"


def test_s7_survival_check_runs_after_the_1200_char_truncation() -> None:
  """``_compact_text`` hard-truncates at 1200 chars; carrier text runs to 1596.

  A reading pushed past the cut is destroyed just as surely as one the model
  reworded, so the survival check must sit AFTER compaction, not before it.
  """
  long_prefix = "filler reading " * 120  # ~1800 chars, well past the 1200 cap
  client = _ScriptedAssetClient([{
      "description": "d",
      "caption": "TICKET-4471 present in caption",
      "ocr_text": long_prefix + " TICKET-4471",
  }])
  asset = _s7_asset(caption="cap without it", ocr_text="TICKET-4471 | 402")
  out = _s7_refine(client, asset, {"ops_chart": ["TICKET-4471"]})

  # The caption still holds it, so nothing is rolled back...
  assert len(client.slots) == 1
  # ...but the truncated ocr_text must not be presented as if it carried it.
  assert len(out.ocr_text) <= 1200

  # Now the same truncation with a caption that does NOT hold the reading:
  client2 = _ScriptedAssetClient([{
      "description": "d",
      "caption": "cap with no reading",
      "ocr_text": long_prefix + " TICKET-4471",
  }])
  out2 = _s7_refine(client2, asset, {"ops_chart": ["TICKET-4471"]})
  assert len(client2.slots) == 2, "truncation-induced loss must be detected"
  assert out2.ocr_text == asset.ocr_text, "and must fall back to blessed text"


def test_s7_survival_check_is_whitespace_and_case_insensitive() -> None:
  """A rewrite that only re-wraps or re-cases the reading is NOT a loss.

  The check folds exactly like the deterministic grader, so it accepts every
  rewrite the grader would still score as a hit -- no false rollbacks.
  """
  client = _ScriptedAssetClient([{
      "description": "d",
      "caption": "the queue shows\n  ticket-4471   open",
      "ocr_text": "402",
  }])
  out = _s7_refine(client, _s7_asset(), {"ops_chart": ["TICKET-4471"]})

  assert len(client.slots) == 1, "a fold-equal rewrite must not trigger a retry"
  assert out.caption.startswith("the queue shows")


def test_s7_fold_matches_the_pipeline_and_file_checker_semantics() -> None:
  """Drift guard.

  Three folds must stay identical: the grader's (``file_checker._fold``), the
  pipeline grounding gate's (``_fold_for_grounding``) and the refiner's. If they
  diverge, the refiner can bless text the grader will later score as a miss --
  which is precisely the failure S7 exists to close.
  """
  from clasharness.evaluation.checkers.file_checker import _fold as checker_fold
  from clasharness.scenario.conflict_llm_pipeline import _fold_for_grounding
  from clasharness.scenario.vlm_materializer import _fold_for_match

  samples = [
      "TICKET-4471",
      "  Ticket-4471\tOpen\n",
      "a\n\n b   c",
      "MiXeD CaSe 402 Rows",
      "",
  ]
  for sample in samples:
    assert _fold_for_match(sample) == checker_fold(sample), sample
    assert _fold_for_match(sample) == _fold_for_grounding(sample), sample


def test_s7_refiner_no_longer_takes_the_dead_anchors_parameter() -> None:
  """``anchors=`` was accepted and never read -- one caller even passed None.

  Keeping a dead knob next to the new, live ``decisive_values_by_asset`` knob
  invites passing the wrong one.
  """
  import inspect

  params = inspect.signature(vlm_materializer.llm_refine_vlm_assets).parameters
  assert "anchors" not in params
  assert "decisive_values_by_asset" in params


def test_s7_asset_detail_contract_mentions_the_verbatim_obligation() -> None:
  from clasharness.scenario.llm_client import _layer_contract

  instruction = _layer_contract("vlm_asset_detail")["instruction"]
  assert "must_carry_verbatim" in instruction


import unittest

class TestProVLMGeneration(unittest.TestCase):
  pass


# Dynamically bind all plain functions starting with test_ to the TestCase class
for name, value in list(globals().items()):
  if name.startswith("test_") and callable(value):
    import inspect
    sig = inspect.signature(value)
    if "tmp_path" in sig.parameters:
      def make_wrapper(func=value):
        def wrapper(self):
          import tempfile
          from pathlib import Path
          with tempfile.TemporaryDirectory() as tmpdir:
            return func(Path(tmpdir))
        return wrapper
      setattr(TestProVLMGeneration, name, make_wrapper())
    else:
      def make_method(func=value):
        return lambda self: func()
      setattr(TestProVLMGeneration, name, make_method())


if __name__ == "__main__":
  unittest.main()
