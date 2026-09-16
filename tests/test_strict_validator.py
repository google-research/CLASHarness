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

from clasharness.scenario.llm_client import MockScenarioLLMClient
from clasharness.scenario.llm_pipeline import LLMScenarioPipeline, ScenarioSeed
from clasharness.scenario.materializer import materialize_generated_scenario
from clasharness.scenario.strict_validator import strict_validate_scenario_package
from clasharness.scenario.strict_validator import (
    _visible_text_references_future_round as _strict_future_round,
)
from clasharness.scenario.conflict_strict_validator import (
    _visible_text_references_future_round as _conflict_future_round,
)


def _make_package(tmp_path: Path):
    seed = ScenarioSeed(slug="mock_case", domain="ops", title="Mock Case", target_rounds=24)
    layers = LLMScenarioPipeline(MockScenarioLLMClient()).generate(seed)
    spec = materialize_generated_scenario(
        seed,
        layers,
        scenario_series_dir=tmp_path / "scenarios",
        output_data_dir=tmp_path / "data",
    )
    return spec, tmp_path / "scenarios" / "pro_scenario_workflow" / "mock_case"


def test_strict_validator_rejects_missing_workspace_file(tmp_path: Path) -> None:
    spec, scenario_dir = _make_package(tmp_path)
    (scenario_dir / "workspace" / "workspace_index.md").unlink()
    result = strict_validate_scenario_package(spec, scenario_dir)
    assert not result.passed
    assert any("missing workspace file" in err for err in result.errors)


def test_strict_validator_rejects_untraceable_evidence_id(tmp_path: Path) -> None:
    spec, scenario_dir = _make_package(tmp_path)
    spec.eval_items[0].metadata["required_evidence_ids"] = ["DECISION:NOT_IN_FILES"]
    result = strict_validate_scenario_package(spec, scenario_dir)
    assert not result.passed
    assert any("not traceable" in err for err in result.errors)


def test_strict_validator_allows_long_narrative_metadata_when_ids_trace(tmp_path: Path) -> None:
    spec, scenario_dir = _make_package(tmp_path)
    spec.eval_items[0].metadata["required_evidence"] = ["LONG-NARRATIVE-NOT-IN-FILES"]
    result = strict_validate_scenario_package(spec, scenario_dir)
    assert result.passed


def test_strict_validator_rejects_answer_leakage(tmp_path: Path) -> None:
    spec, scenario_dir = _make_package(tmp_path)
    spec.rounds[0].instruction += f" {spec.eval_items[0].metadata['required_evidence_ids'][0]}"
    result = strict_validate_scenario_package(spec, scenario_dir)
    assert not result.passed
    assert any("instruction leaks expected evidence id" in err for err in result.errors)


def test_strict_validator_rejects_early_future_marker_leakage(tmp_path: Path) -> None:
    spec, scenario_dir = _make_package(tmp_path)
    overrides = spec.rounds[0].task_spec["data"]["private"]["initial_state_overrides"]
    overrides["workspace/future_notice.md"] = (
        "DEPRECATION NOTICE appended Round 12: use the Round 24 final approval note."
    )
    result = strict_validate_scenario_package(spec, scenario_dir)
    assert not result.passed
    assert any("early runtime state leaks future marker" in err for err in result.errors)


def test_strict_validator_rejects_early_semantic_answer_leakage(tmp_path: Path) -> None:
    spec, scenario_dir = _make_package(tmp_path)
    overrides = spec.rounds[0].task_spec["data"]["private"]["initial_state_overrides"]
    overrides["workspace/posthoc_rule.md"] = (
        "Governing Rule and Canonical Decision: newer owner notes supersede this document. "
        "Correct launch status: gated launch required."
    )
    result = strict_validate_scenario_package(spec, scenario_dir)
    assert not result.passed
    assert any("early runtime state leaks future marker" in err for err in result.errors)


def test_strict_validator_rejects_early_plain_future_file_refs(tmp_path: Path) -> None:
    spec, scenario_dir = _make_package(tmp_path)
    overrides = spec.rounds[0].task_spec["data"]["private"]["initial_state_overrides"]
    overrides["workspace/leaky_future_ref.md"] = (
        "Do NOT act on this. See evidence_log_casey_r11_r16.jsonl and q22 owner notes later."
    )
    result = strict_validate_scenario_package(spec, scenario_dir)
    assert not result.passed
    assert any(
        "future marker" in err
        or "compact future round reference" in err
        or "future file reference" in err
        for err in result.errors
    )


def test_strict_validator_allows_visible_past_round_file_token(tmp_path: Path) -> None:
    spec, scenario_dir = _make_package(tmp_path)
    round_item = spec.rounds[14]
    overrides = round_item.task_spec["data"]["private"]["initial_state_overrides"]
    overrides["workspace/role_reassignment_email_r11.eml"] = (
        "This already-visible provenance file was issued in r11."
    )
    result = strict_validate_scenario_package(spec, scenario_dir)
    assert result.passed


def test_strict_validator_rejects_unavailable_future_round_file_token(tmp_path: Path) -> None:
    spec, scenario_dir = _make_package(tmp_path)
    round_item = spec.rounds[14]
    overrides = round_item.task_spec["data"]["private"]["initial_state_overrides"]
    overrides["workspace/visible_note.md"] = (
        "Customer deadline is documented in CUSTOMER_DEADLINE_R18.ics."
    )
    result = strict_validate_scenario_package(spec, scenario_dir)
    assert not result.passed
    assert any("future file reference" in err or "invisible file" in err for err in result.errors)


def test_strict_validator_rejects_early_invisible_doc_refs(tmp_path: Path) -> None:
    spec, scenario_dir = _make_package(tmp_path)
    overrides = spec.rounds[0].task_spec["data"]["private"]["initial_state_overrides"]
    overrides["workspace/visible_note.md"] = (
        "Gate 3 is governed by delegation/DELEGATION_LOG_v3_corrected.xlsx."
    )
    result = strict_validate_scenario_package(spec, scenario_dir)
    assert not result.passed
    assert any("invisible file" in err for err in result.errors)


def test_strict_validator_rejects_early_canonical_anchor_leakage(tmp_path: Path) -> None:
    spec, scenario_dir = _make_package(tmp_path)
    overrides = spec.rounds[0].task_spec["data"]["private"]["initial_state_overrides"]
    overrides["workspace/final_answer.md"] = "DECISION:GATED_LAUNCH_REQUIRED"
    result = strict_validate_scenario_package(spec, scenario_dir)
    assert not result.passed
    assert any("early runtime state leaks DECISION:" in err for err in result.errors)


def test_strict_validator_rejects_weak_status_anchor(tmp_path: Path) -> None:
    spec, scenario_dir = _make_package(tmp_path)
    spec.eval_items[0].metadata["required_evidence_ids"] = ["DECISION:STATUS_07"]
    result = strict_validate_scenario_package(spec, scenario_dir)
    assert not result.passed
    assert any("weak evidence id" in err for err in result.errors)


def test_strict_validator_rejects_bad_update_round_ref(tmp_path: Path) -> None:
    spec, scenario_dir = _make_package(tmp_path)
    spec.updates[0].before_round = "missing_round"
    result = strict_validate_scenario_package(spec, scenario_dir)
    assert not result.passed
    assert any("unknown before_round" in err for err in result.errors)


# --- C: future-round-leak heuristic precision (both copies kept in sync) ------


def test_future_round_heuristic_ignores_domain_noise() -> None:
    # A bare R<N>/Q<N> WITHOUT a round-context cue is domain noise (a release
    # tag, a rack id, a fiscal quarter), not a future-round reference. Flagging
    # it is a false positive that aborted otherwise-valid generation
    # (the it_deployment_gate_v01 failure mode).
    for fn in (_conflict_future_round, _strict_future_round):
        assert fn("roll back to release R2 today", round_index=1) is False
        assert fn("power-cycle rack R12 in the datacenter", round_index=1) is False
        assert fn("Q3 revenue dashboard shows a dip", round_index=1) is False
        assert fn("compare Q3 vs Q4 fiscal projections", round_index=1) is False
        # 'phase' is deliberately NOT a cue word, so a version tag near it stays
        # unflagged.
        assert fn("deploy R2 during phase 3 window", round_index=1) is False


def test_future_round_heuristic_flags_genuine_leaks() -> None:
    # Explicit round refs are always decisive; bare tokens are flagged when a
    # round-context cue (round/next/later/...) is nearby.
    for fn in (_conflict_future_round, _strict_future_round):
        assert fn("You will handle this in round 5.", round_index=1) is True
        assert fn("ROUND_03 covers the rollback.", round_index=1) is True
        assert fn("Defer to the next round, R5.", round_index=1) is True
        assert fn("This is covered later; see R7.", round_index=1) is True


def test_future_round_heuristic_current_and_past_rounds_ok() -> None:
    for fn in (_conflict_future_round, _strict_future_round):
        assert fn("This is round r01 instruction.", round_index=1) is False
        assert fn("Recap of round 2 done in this round.", round_index=3) is False


def test_future_round_heuristic_matches_legacy_positive() -> None:
    # The pre-existing post-generation positive case must still be flagged.
    for fn in (_conflict_future_round, _strict_future_round):
        assert fn("Wait until round r02 to complete.", round_index=1) is True


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__]))
