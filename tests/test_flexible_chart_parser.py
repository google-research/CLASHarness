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

"""Unit tests for the flexible key-normalizing chart rendering template."""

import os
import tempfile
from examples.train_and_eval import LLMAgent, _normalize_model_command
from clasharness.scenario.pattern_generator import generate_pattern_data
from clasharness.scenario.vlm_asset_generators import CodeChartGenerator
from clasharness.scenario.vlm_asset_generators import render_chart_spec


def test_flexible_parsing_single_line():
  """S1: Verify parsing of single-line chart specs containing custom keys."""
  spec = {
      "title": "Application Errors",
      "chart_type": "line_chart",
      "rows": [
          {"time": "10:10", "errors_per_minute": 0},
          {"time": "10:15", "errors_per_minute": 5},
          {"time": "10:20", "errors_per_minute": 35},
      ],
  }
  with tempfile.TemporaryDirectory() as tmpdir:
    out_path = os.path.join(tmpdir, "chart.png")
    render_chart_spec(spec, out_path)
    assert os.path.exists(out_path)
    with open(out_path, "rb") as f:
      header = f.read(4)
    assert header == b"\x89PNG"


def test_flexible_parsing_multi_line():
  """S2: Verify parsing of multi-line chart specs with spacing/casing variations in keys."""
  spec = {
      "title": "API Gateway Traffic",
      "chart_type": "multi_line_chart",
      "series_names": [
          "External Ingress (RPS)",
          "Internal Healthchecks (RPS)",
      ],
      "rows": [
          {
              "Timestamp": "10:00",
              "External Ingress (RPS)": 120,
              "Internal Health Checks (RPS)": 10,
          },
          {
              "Timestamp": "10:05",
              "External Ingress (RPS)": 125,
              "Internal Health Checks (RPS)": 10,
          },
      ],
  }
  with tempfile.TemporaryDirectory() as tmpdir:
    out_path = os.path.join(tmpdir, "chart.png")
    render_chart_spec(spec, out_path)
    assert os.path.exists(out_path)
    with open(out_path, "rb") as f:
      header = f.read(4)
    assert header == b"\x89PNG"


def test_flexible_parsing_fallback():
  """S3: Verify fallback behavior in chart_template.html when no label or numeric keys match."""
  spec = {
      "title": "Fallback Empty Keys Chart",
      "chart_type": "bar_chart",
      "rows": [
          {"foo": "A", "bar": "B"},
          {"foo": "C", "bar": "D"},
      ],
  }
  with tempfile.TemporaryDirectory() as tmpdir:
    out_path = os.path.join(tmpdir, "chart.png")
    render_chart_spec(spec, out_path)
    assert os.path.exists(out_path)
    with open(out_path, "rb") as f:
      header = f.read(4)
    assert header == b"\x89PNG"


def test_mathematical_curve_presets():
  """S4: Verify mathematical curve preset generation (flatline, spike, decay, periodic)."""

  # Flatline
  data = generate_pattern_data(
      "flatline", {"baseline": 50.0, "noise_std": 0.0}, length=5
  )
  assert data == [50.0, 50.0, 50.0, 50.0, 50.0]

  # Spike / Drop
  data = generate_pattern_data(
      "drop",
      {
          "baseline": 100.0,
          "spike_value": 0.0,
          "spike_index": 2,
          "noise_std": 0.0,
      },
      length=4,
  )
  assert data == [100.0, 100.0, 0.0, 100.0]

  # Decay
  data = generate_pattern_data(
      "decay",
      {"start_value": 10.0, "end_value": 0.0, "noise_std": 0.0},
      length=3,
  )
  assert data == [10.0, 5.0, 0.0]

  # Periodic (Sine wave)
  data = generate_pattern_data(
      "periodic",
      {"baseline": 10.0, "amplitude": 5.0, "frequency": 1.0, "noise_std": 0.0},
      length=5,
  )
  assert len(data) == 5
  assert all(v >= 0.0 for v in data)


def test_gantt_chart_flexible_keys():
  """S5: Verify Gantt rendering maps custom timestamps and duration keys correctly."""
  spec = {
      "title": "Script Executions",
      "chart_type": "gantt_chart",
      "rows": [
          {
              "Phase": "Init",
              "Start_Time": "14:00",
              "End_Time": "14:05",
              "Duration_Minutes": 5,
          },
          {
              "Phase": "Reload",
              "Start_Time": "14:05",
              "End_Time": "14:06",
              "Duration_Minutes": 1,
          },
      ],
  }
  with tempfile.TemporaryDirectory() as tmpdir:
    out_path = os.path.join(tmpdir, "gantt_chart.png")
    render_chart_spec(spec, out_path)
    assert os.path.exists(out_path)
    with open(out_path, "rb") as f:
      header = f.read(4)
    assert header == b"\x89PNG"


def test_text_block_card_rendering():
  """S6: Verify rendering of a code/text document as a monospace card using Chrome."""
  spec = {
      "title": "edge_proxy.yaml",
      "type": "YAML",
      "content": "proxy:\n  listen_port: 443\n  backend: 'null_route'",
  }
  generator = CodeChartGenerator()
  asset = generator.render_text_spec(spec)
  assert asset.mime_type == "image/png"
  assert len(asset.data) > 0
  assert asset.data[:4] == b"\x89PNG"


def test_llm_agent_gemini_init():
  """S7: Verify LLMAgent can be initialized with gemini provider."""
  # pylint: disable=protected-access
  agent = LLMAgent(
      provider="gemini",
      model="gemini-2.5-flash",
  )
  assert agent._provider == "gemini"
  assert agent._model == "gemini-2.5-flash"
  assert agent._base_url == "https://generativelanguage.googleapis.com/v1beta"
  assert agent._api_key_name() == "GEMINI_API_KEY"


def test_normalize_model_command_chatty():
  """S8: Verify robust extraction of commands from chatty LLM responses."""
  # Case 1: Chatty commentary before inline code block command
  raw_1 = "I think we need to read the index file:\n`file read --path /index.md`"
  assert _normalize_model_command(raw_1) == "file read --path /index.md"

  # Case 2: Chatty commentary before fenced code block command
  raw_2 = "Sure, let's list the directory:\n```bash\nfile list --path /\n```"
  assert _normalize_model_command(raw_2) == "file list --path /"

  # Case 3: Just the command directly
  raw_3 = "email read --id pro_s01"
  assert _normalize_model_command(raw_3) == "email read --id pro_s01"

  # Case 4: No valid prefix match, fall back to first line
  raw_4 = "Invalid commentary without command lines"
  assert _normalize_model_command(raw_4) == "Invalid commentary without command lines"




