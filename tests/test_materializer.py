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

from pathlib import Path

from clasharness.core.task import Task, load_task_ids
from clasharness.scenario.llm_client import MockScenarioLLMClient
from clasharness.scenario.llm_pipeline import LLMScenarioPipeline, ScenarioSeed
from clasharness.scenario.materializer import materialize_generated_scenario
from clasharness.scenario.strict_validator import strict_validate_scenario_package


def test_materializer_writes_package_and_tasks(tmp_path: Path) -> None:
    seed = ScenarioSeed(slug="mock_case", domain="ops", title="Mock Case", target_rounds=24)
    layers = LLMScenarioPipeline(MockScenarioLLMClient()).generate(seed)
    spec = materialize_generated_scenario(
        seed,
        layers,
        scenario_series_dir=tmp_path / "scenarios",
        output_data_dir=tmp_path / "data",
    )
    scenario_dir = tmp_path / "scenarios" / "pro_scenario_workflow" / "mock_case"
    assert (scenario_dir / "scenario.json").exists()
    assert (scenario_dir / "workspace" / "workspace_index.md").exists()
    canonical = scenario_dir / "internal" / "canonical_evidence.md"
    assert canonical.exists()
    canonical_text = canonical.read_text()
    assert "DECISION:" in canonical_text
    assert "RULE:" in canonical_text
    assert not (scenario_dir / "workspace" / "canonical_evidence.md").exists()
    assert (scenario_dir / "sessions" / "s01.jsonl").exists()
    assert (scenario_dir / "updates" / "u01" / "workspace" / "late_owner_note_1.md").exists()
    strict_validate_scenario_package(spec, scenario_dir).raise_for_errors()
    task_ids = load_task_ids("total", data_dir=tmp_path / "data")
    assert len(task_ids) == 24
    task = Task.from_dict(spec.rounds[0].task_spec)
    assert Task.from_dict(task.to_dict()).task_id == task.task_id
    assert task.ground_truth
    assert task.ground_truth.answer["expected_anchor_ids"]
    assert "Truth anchor" not in task.instruction
    assert task.ground_truth.answer["truth_anchor"] not in task.instruction
    assert task.ground_truth.answer["expected_anchor_ids"][0] not in task.instruction
    overrides = task.data.private["initial_state_overrides"]
    assert "workspace/canonical_evidence.md" not in overrides
    assert "emails.json" in overrides
    assert "tasks.json" in overrides
    assert "calendar_events.json" in overrides
    assert task.data.public["round_type"] == "observation"
    assert all(not anchor.startswith("DECISION:") for anchor in task.ground_truth.answer["expected_anchor_ids"])
    assert "workspace/owner_decision.md" not in overrides
    assert "workspace/evidence_matrix.md" not in overrides
    assert "workspace/late_owner_note_1.md" not in overrides
    q01_runtime = str(overrides).upper()
    for forbidden in (
        "DECISION:",
        "RULE:",
        "UPDATE:",
        "DEPRECATION NOTICE",
        "APPENDED ROUND",
        "ROUND 24",
        "SUPERSEDED",
        "CANONICAL DECISION",
        "GOVERNING RULE",
        "AUTHORITATIVE SOURCES",
        "NEWER OWNER",
        "DEPRECATED",
        "GATED LAUNCH",
        "CORRECT LAUNCH STATUS",
        "FIRST FORMAL OVERRIDE",
        "DO NOT ACT",
        "BEFORE COMPLIANCE FLAGS",
        "RETAINED FOR AUDIT",
        "AUDIT AND RECONCILIATION",
        "EVIDENCE_LOG_CASEY_R11_R16",
        "Q22",
        "R11",
    ):
        assert forbidden not in q01_runtime
    assert not any(
        anchor.startswith("EVIDENCE:CAL_") or anchor.startswith("EVIDENCE:TASK_")
        for anchor in task.ground_truth.answer["expected_anchor_ids"]
    )
    assert "EVIDENCE:EARLY_VISIBLE_STATE" not in task.ground_truth.answer["expected_anchor_ids"]
    assert "EVIDENCE:EARLY_VISIBLE_STATE" not in q01_runtime
    assert all(
        anchor in q01_runtime
        for anchor in task.ground_truth.answer["expected_anchor_ids"]
    )
    assert "/reports/" not in task.instruction
    assert any(command.startswith("tasks add --title") for command in task.ground_truth.solution_commands)
    assert not any("/reports/" in command for command in task.ground_truth.solution_commands)
    assert any(
        check.get("effect_type") == "tasks_created"
        for check in task.ground_truth.evaluation_checks
        if isinstance(check, dict)
    )
    assert any(
        check.get("effect_type") == "tasks_created"
        and check.get("condition") == "exists"
        for check in task.ground_truth.evaluation_checks
        if isinstance(check, dict)
    )
    assert not any(
        check.get("effect_type") == "tasks_created"
        and check.get("condition") == "field_contains"
        and check.get("expected", {}).get("value") == "PRO"
        for check in task.ground_truth.evaluation_checks
        if isinstance(check, dict)
    )
    assert {"email", "tasks", "calendar"}.issubset(task.domains)
    assert any(command.startswith("email read --id") for command in task.ground_truth.solution_commands)
    assert "tasks list --status all" in task.ground_truth.solution_commands
    assert "calendar list" in task.ground_truth.solution_commands
    q08 = Task.from_dict(spec.rounds[7].task_spec)
    assert q08.ground_truth
    assert q08.data.public["round_type"] == "reconciliation"
    assert "/reports/" not in q08.instruction
    assert any(command.startswith("email send ") for command in q08.ground_truth.solution_commands)
    assert any(
        check.get("effect_type") == "emails_sent"
        for check in q08.ground_truth.evaluation_checks
        if isinstance(check, dict)
    )
    assert any(
        command == "file read --path /late_owner_note_1.md"
        for command in q08.ground_truth.solution_commands
    )
    q15 = Task.from_dict(spec.rounds[14].task_spec)
    assert q15.ground_truth
    assert q15.data.public["round_type"] == "update_aware"
    assert "/actions/mock_case_q15_update.md" in q15.instruction
    assert "/reports/" not in q15.instruction
    assert any(
        command.startswith("file create --path /actions/mock_case_q15_update.md")
        for command in q15.ground_truth.solution_commands
    )
    q21 = Task.from_dict(spec.rounds[20].task_spec)
    assert q21.ground_truth
    assert q21.data.public["round_type"] == "action"
    assert "/reports/mock_case_q21.md" in q21.instruction
    assert any(
        command.startswith("file create --path /reports/mock_case_q21.md")
        for command in q21.ground_truth.solution_commands
    )
    assert any(
        check.get("effect_type") == "files_created"
        and check.get("condition") == "field_contains"
        and check.get("expected", {}).get("field") == "content"
        and check.get("expected", {}).get("value") == "Evidence anchors"
        for check in q21.ground_truth.evaluation_checks
        if isinstance(check, dict)
    )


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__]))
