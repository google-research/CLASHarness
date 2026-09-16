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

import unittest
from clasharness.scenario.vlm_spec import VLMGenerationPolicy, VLMAssetSpec, VLMScenarioSpec, VLMInputPolicy
from clasharness.scenario.vlm_materializer import _chart_spec_for_asset, _clean_prompt_text


class TestDashboardTemplate(unittest.TestCase):

  def test_generation_policy_loads_layouts(self):
    data = {
        "chart_provider": "template",
        "chart_layouts": {
            "asset_board_early": "devops_dashboard",
            "asset_table_conflict": "single"
        }
    }
    policy = VLMGenerationPolicy.from_dict(data)
    self.assertEqual(policy.chart_provider, "template")
    self.assertEqual(policy.chart_layouts.get("asset_board_early"), "devops_dashboard")
    self.assertEqual(policy.chart_layouts.get("asset_table_conflict"), "single")
    self.assertEqual(policy.chart_layouts.get("unknown_asset"), None)

  def test_chart_spec_populates_dashboard_data(self):
    # Setup mock asset with devops_dashboard layout in metadata
    asset = VLMAssetSpec(
        id="asset_board_early",
        kind="chart",
        generator="template_chart",
        path="board.png",
        mime_type="image/png",
        available_from_round=1,
        anchor_ids=["test_anchor"],
        description="Core Traffic Chart",
        provider_metadata={
            "layout": "devops_dashboard",
            "chart_rows": [
                {"label": "10:00", "value": 100},
                {"label": "10:05", "value": 0}
            ]
        }
    )

    # Setup dummy spec
    spec = VLMScenarioSpec(
        series="test",
        scenario_slug="test_slug",
        title="Test System Outage Incident",
        domain="network",
        core_conflict="API Gateway is drop to zero traffic.",
        modalities=["chart"],
        input_policy=VLMInputPolicy(),
        generation_policy=VLMGenerationPolicy(),
        allowed_tool_families=["asset"],
        assets=[asset],
        rounds=[],
        evals=[]
    )

    # Build spec
    chart_spec = _chart_spec_for_asset(asset, spec)

    # Assert layout and telemetry
    self.assertEqual(chart_spec.get("layout"), "devops_dashboard")
    self.assertIn("dashboard_data", chart_spec)
    
    dash_data = chart_spec["dashboard_data"]
    self.assertEqual(dash_data.get("system_name"), "Production Control Cluster")
    self.assertEqual(len(dash_data.get("services", [])), 5)
    
    # Assert microservices list includes sus service marked as warning
    sus_service = next((s for s in dash_data["services"] if "gateway" in s["name"]), None)
    self.assertIsNotNone(sus_service)
    self.assertEqual(sus_service["status"], "warning")
    
    # Assert other services are healthy
    auth_service = next((s for s in dash_data["services"] if "auth" in s["name"]), None)
    self.assertIsNotNone(auth_service)
    self.assertEqual(auth_service["status"], "healthy")

    # Assert metrics are present
    self.assertEqual(len(dash_data.get("metrics", [])), 3)
    cpu_metric = next((m for m in dash_data["metrics"] if m["name"] == "CPU Usage"), None)
    self.assertIsNotNone(cpu_metric)
    self.assertEqual(len(cpu_metric["sparkline"]), 7)

  def test_clean_prompt_text_strips_metadata(self):
    prompt1 = "Show the traffic drop. Evidence anchors: EXTERNAL_INGRESS, ANOTHER_ONE."
    self.assertEqual(_clean_prompt_text(prompt1), "Show the traffic drop.")

    prompt2 = "Create a whiteboard diagram for EVIDENCE:ARCH_DIAGRAM showing Client -> DB."
    self.assertEqual(_clean_prompt_text(prompt2), "Create a whiteboard diagram for showing Client -> DB.")

    # Natural english text with "anchor" should be preserved (no uppercase anchors follow)
    prompt3 = "Diagram showing the network. The database pool utilization is our main anchor for this comparison."
    self.assertEqual(_clean_prompt_text(prompt3), "Diagram showing the network. The database pool utilization is our main anchor for this comparison.")

    # Uppercase anchors following label should be stripped
    prompt4 = "A clean dashboard. Caption: Traffic plunge. EVIDENCE:GATEWAY_CRASH."
    self.assertEqual(_clean_prompt_text(prompt4), "A clean dashboard. Caption: Traffic plunge.")


if __name__ == "__main__":
  unittest.main()
