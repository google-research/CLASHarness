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

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from clasharness.scenario.llm_client import MockScenarioLLMClient, ScenarioLLMError
from clasharness.scenario.llm_pipeline import LLMScenarioPipeline, ScenarioSeed


class BadClient:
    def generate_layer(self, layer, payload):
        return {}


class UnderscoreUpdateClient(MockScenarioLLMClient):
    def generate_layer(self, layer, payload):
        if layer == "layer4_update_outline":
            return {
                "updates": [
                    {
                        "update_id": "upd_01",
                        "before_round": "q08",
                        "purpose": "late owner note after early rounds",
                        "workspace_path": "late_owner_note_1.md",
                        "session_id": "s01",
                    }
                ]
            }
        return super().generate_layer(layer, payload)


class LeakyQuestionClient(MockScenarioLLMClient):
    def generate_layer(self, layer, payload):
        if layer == "layer3_eval_round":
            context = payload.get("context", {})
            narrative = context.get("narrative", {})
            decision_id = narrative.get("canonical_decision_id", "DECISION:LEAKED")
            return {
                "question": f"Does {decision_id} apply here? Write the report.",
                "rationale_summary": "private rationale",
            }
        return super().generate_layer(layer, payload)


class JsonArtifactWorkspaceClient(MockScenarioLLMClient):
    def generate_layer(self, layer, payload):
        if layer == "layer1_workspace_outline":
            return {
                "files": [
                    {
                        "id": "f01",
                        "path": "index.json",
                        "purpose": "structured workspace index",
                    },
                    {
                        "id": "f02",
                        "path": "owner_notes.md",
                        "purpose": "authoritative owner notes",
                    },
                    {
                        "id": "f03",
                        "path": "evidence_matrix.md",
                        "purpose": "evidence matrix",
                    },
                ]
            }
        if layer == "layer1_workspace_file":
            slot = payload.get("slot", {})
            if slot.get("path") == "index.json":
                return {
                    "id": "f01",
                    "path": "index.json",
                    "artifact_type": "workspace_index",
                    "files": [
                        {
                            "id": "f02",
                            "path": "owner_notes.md",
                            "status": "current",
                        }
                    ],
                }
        return super().generate_layer(layer, payload)


def test_pipeline_generates_all_layers() -> None:
    seed = ScenarioSeed(
        slug="mock_case",
        domain="ops",
        title="Mock Case",
        target_rounds=24,
    )
    layers = LLMScenarioPipeline(MockScenarioLLMClient()).generate(seed)
    assert set(layers) == {
        "layer0_narrative",
        "layer1_workspace",
        "layer2_sessions",
        "layer3_eval",
        "layer4_dynamic",
    }
    assert len(layers["layer3_eval"]["rounds"]) == 24
    narrative = layers["layer0_narrative"]
    assert narrative["canonical_decision_id"].startswith("DECISION:")
    assert narrative["canonical_rule_id"].startswith("RULE:")
    assert narrative["status_timeline"]
    first_round = layers["layer3_eval"]["rounds"][0]
    assert first_round["required_evidence_ids"]
    assert first_round["required_evidence_ids"][0].startswith("DECISION:")


def test_pipeline_uses_real_update_ids_for_expected_anchors() -> None:
    seed = ScenarioSeed(slug="mock_case", domain="ops", title="Mock Case", target_rounds=8)
    layers = LLMScenarioPipeline(UnderscoreUpdateClient()).generate(seed)
    round_q08 = layers["layer3_eval"]["rounds"][7]
    assert "UPDATE:UPD_01_APPLIED" in round_q08["required_evidence_ids"]
    assert "UPDATE:U01_APPLIED" not in round_q08["required_evidence_ids"]


def test_pipeline_sanitizes_anchor_ids_from_questions() -> None:
    seed = ScenarioSeed(slug="mock_case", domain="ops", title="Mock Case", target_rounds=1)
    layers = LLMScenarioPipeline(LeakyQuestionClient()).generate(seed)
    question = layers["layer3_eval"]["rounds"][0]["question"]
    assert "DECISION:" not in question
    assert "the current scenario decision" in question


def test_pipeline_accepts_structured_workspace_file_without_content() -> None:
    seed = ScenarioSeed(slug="mock_case", domain="ops", title="Mock Case", target_rounds=1)
    layers = LLMScenarioPipeline(JsonArtifactWorkspaceClient()).generate(seed)
    content = layers["layer1_workspace"]["files"][0]["content"]
    assert '"artifact_type": "workspace_index"' in content
    assert '"owner_notes.md"' in content


def test_pipeline_rejects_malformed_layer() -> None:
    seed = ScenarioSeed(slug="mock_case", domain="ops", title="Mock Case")
    with pytest.raises(ScenarioLLMError):
        LLMScenarioPipeline(BadClient()).generate(seed)


def test_seed_rejects_bad_slug() -> None:
    with pytest.raises(ValueError):
        ScenarioSeed.from_dict({"slug": "Bad Slug", "domain": "ops", "title": "Bad"})


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__]))
