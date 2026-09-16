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

import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from clasharness.scenario.llm_auditor import audit_scenario_with_llm
from clasharness.scenario.spec import ScenarioRound, ScenarioSpec


class MockAuditRepairLLMClient:
    """Mock client that simulates a failed audit on iteration 1, and success on iteration 2."""

    def __init__(self):
        self.calls = []

    def generate_layer(self, layer: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((layer, payload))

        # First Audit attempt: Report that instruction is contradicting, and an anchor is missing.
        if layer == "audit_instruction_checker":
            # Check if instruction has already been updated to the corrected version
            if "REPAIRED INSTRUCTION" in payload.get("instruction", ""):
                return {"has_contradiction": False}
            return {
                "has_contradiction": True,
                "remediation": {
                    "target_file": "workspace/config.yaml",
                    "required_content": "allow_untrusted_credentials: true",
                },
            }

        elif layer == "audit_anchor_discoverability":
            # Check if overrides file content has already been edited to include anchor
            content = payload.get("visible_files_content", "")
            if "evidence_anchor_101" in content:
                return {"missing_anchors": []}
            return {
                "missing_anchors": [
                    {
                        "anchor_id": "evidence_anchor_101",
                        "target_file": "workspace/config.yaml",
                        "injection_note": "evidence_anchor_101: verified",
                    }
                ]
            }

        elif layer == "audit_telemetry_alignment":
            return {"has_mismatch": False}

        # Repair prompts
        elif layer == "repair_instruction_contradiction":
            return {
                "updated_instruction": "REPAIRED INSTRUCTION: Read config.",
                "subagent_effects_to_add": ["file append --path workspace/config.yaml --content 'allow_untrusted_credentials: true'"],
            }

        elif layer == "repair_missing_anchors":
            original_content = payload.get("file_content", "")
            return {
                "edited_content": original_content.rstrip() + "\n\nevidence_anchor_101: verified\n"
            }

        return {}


def test_iterative_self_repair_loop() -> None:
    # 1. Create a dummy scenario spec with a contradiction
    task_spec = {
        "instruction": "Read config. Do not run terminal commands.",
        "ground_truth": {
            "evaluation_checks": [
                {
                    "check_id": "c01",
                    "kind": "file_contains",
                    "path": "workspace/config.yaml",
                    "expected": "allow_untrusted_credentials: true",
                }
            ],
            "answer": {
                "expected_anchor_ids": ["evidence_anchor_101"]
            }
        },
        "data": {
            "private": {
                "expected_anchor_ids": ["evidence_anchor_101"],
                "mock_subagents": [
                    {
                        "agent_role": "assistant",
                        "mock_response": "Hello",
                        "state_effects": []
                    }
                ],
                "initial_state_overrides": {
                    "workspace/config.yaml": "allow_untrusted_credentials: false\n"
                }
            }
        }
    }

    round_item = ScenarioRound(
        round_id="r01",
        task_id="test_task_01",
        instruction="Read config. Do not run terminal commands.",
        task_spec=task_spec,
    )

    spec = ScenarioSpec(
        scenario_id="test_repair_spec",
        generator_id="test_gen",
        scenario_slug="test_slug",
        title="Test Repair Spec",
        description="Test repair spec",
        version="1.0",
        narrative={},
        workspace={"files": []},
        sessions=[],
        updates=[],
        rounds=[round_item],
        eval_items=[],
    )

    client = MockAuditRepairLLMClient()

    # 2. Run the audit
    repaired_spec = audit_scenario_with_llm(spec, client, max_iterations=3)

    # 3. Assertions
    # First mock subagent state effects should contain the file write effect
    private_data = repaired_spec.rounds[0].task_spec["data"]["private"]
    effects = private_data["mock_subagents"][0]["state_effects"]
    assert any("allow_untrusted_credentials: true" in eff for eff in effects)

    # The file config.yaml content should contain the integrated anchor ID
    config_content = private_data["initial_state_overrides"]["workspace/config.yaml"]
    assert "evidence_anchor_101: verified" in config_content

    # The instruction should be updated to the repaired version
    assert repaired_spec.rounds[0].task_spec["instruction"] == "REPAIRED INSTRUCTION: Read config."

    # Verify generate_layer was called twice for each auditor (first failed, second passed)
    layer_calls = [c[0] for c in client.calls]
    assert layer_calls.count("audit_instruction_checker") == 2
    assert layer_calls.count("audit_anchor_discoverability") == 2
    assert layer_calls.count("repair_instruction_contradiction") == 1
    assert layer_calls.count("repair_missing_anchors") == 1


def test_audit_skips_binary_missing_anchors() -> None:
    # 1. Create a task spec with a missing anchor targeting a binary/image file
    task_spec = {
        "instruction": "Read status.",
        "ground_truth": {
            "evaluation_checks": [],
            "answer": {
                "expected_anchor_ids": ["evidence_anchor_image"]
            }
        },
        "data": {
            "private": {
                "expected_anchor_ids": ["evidence_anchor_image"],
                "mock_subagents": [],
                "initial_state_overrides": {
                    "assets/workspace/workspace_flight_schedule.png": "binary_content_placeholder"
                }
            }
        }
    }

    round_item = ScenarioRound(
        round_id="r01",
        task_id="test_task_01",
        instruction="Read status.",
        task_spec=task_spec,
    )

    spec = ScenarioSpec(
        scenario_id="test_repair_spec",
        generator_id="test_gen",
        scenario_slug="test_slug",
        title="Test Binary Skip Spec",
        description="Test binary skip spec",
        version="1.0",
        narrative={},
        workspace={"files": []},
        sessions=[],
        updates=[],
        rounds=[round_item],
        eval_items=[],
    )

    # Mock client that reports a missing anchor in a png file
    class MockBinaryAuditClient(MockAuditRepairLLMClient):
        def generate_layer(self, layer: str, payload: dict[str, Any]) -> dict[str, Any]:
            if layer == "audit_anchor_discoverability":
                return {
                    "missing_anchors": [
                        {
                            "anchor_id": "evidence_anchor_image",
                            "target_file": "assets/workspace/workspace_flight_schedule.png",
                        }
                    ]
                }
            return super().generate_layer(layer, payload)

    client = MockBinaryAuditClient()

    # 2. Run the audit
    repaired_spec = audit_scenario_with_llm(spec, client, max_iterations=1)

    # 3. Assertions: repair_missing_anchors should NOT have been called for the PNG file!
    layer_calls = [c[0] for c in client.calls]
    assert "repair_missing_anchors" not in layer_calls


def test_audit_passes_if_anchor_is_in_visible_visual_assets() -> None:
    # 1. Create task spec where expected anchor matches a visible visual asset in visible_refs
    task_spec = {
        "instruction": "Read status.",
        "ground_truth": {
            "evaluation_checks": [],
            "answer": {
                "expected_anchor_ids": ["ops_dashboard_flight_schedule"]
            }
        },
        "data": {
            "private": {
                "expected_anchor_ids": ["ops_dashboard_flight_schedule"],
                "mock_subagents": [],
                "initial_state_overrides": {}
            }
        }
    }

    round_item = ScenarioRound(
        round_id="r01",
        task_id="test_task_01",
        instruction="Read status.",
        task_spec=task_spec,
        visible_refs=["ops_dashboard_flight_schedule"]
    )

    spec = ScenarioSpec(
        scenario_id="test_repair_spec",
        generator_id="test_gen",
        scenario_slug="test_slug",
        title="Test Visual Discoverability Spec",
        description="Test visual discoverability spec",
        version="1.0",
        narrative={},
        workspace={"files": []},
        sessions=[],
        updates=[],
        rounds=[round_item],
        eval_items=[],
    )

    # Mock client that behaves normally. It will receive payload with 'visible_visual_assets'
    class MockVisualDiscoverabilityClient(MockAuditRepairLLMClient):
        def generate_layer(self, layer: str, payload: dict[str, Any]) -> dict[str, Any]:
            if layer == "audit_anchor_discoverability":
                assert "visible_visual_assets" in payload
                assets_list = json.loads(payload["visible_visual_assets"])
                assert "ops_dashboard_flight_schedule" in assets_list
                # Since it is present in visual assets, return no missing anchors!
                return {"missing_anchors": []}
            return super().generate_layer(layer, payload)

    client = MockVisualDiscoverabilityClient()

    # 2. Run the audit
    repaired_spec = audit_scenario_with_llm(spec, client, max_iterations=1)

    # 3. Assertions
    layer_calls = [c[0] for c in client.calls]
    assert "repair_missing_anchors" not in layer_calls


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__]))
