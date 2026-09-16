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

"""Tests for examples/train_and_eval.py CLI mode handling and preflight."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from clasharness.evaluation.llm_retry import (
    anthropic_kwargs_carry_temperature as _has_temp,
)

import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import examples.train_and_eval as train_and_eval


def _make_dummy_summary(calls: dict) -> type:
    """Return a minimal run-summary stub whose ``print_report`` records into ``calls``.

    Shared by the CLI-dispatch tests that only need to assert the report was printed.
    """

    class _DummySummary:
        def print_report(self) -> None:
            calls["printed"] = True

    return _DummySummary


def test_preflight_requires_openclaw_for_real_and_hybrid(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(train_and_eval.shutil, "which", lambda _: None)

    for mode in ("real", "hybrid"):
        with pytest.raises(RuntimeError, match="openclaw"):
            train_and_eval._preflight_mode(mode)


def test_preflight_skips_openclaw_check_for_mock_and_multi(monkeypatch: pytest.MonkeyPatch):
    def _unexpected(_: str):
        raise AssertionError("shutil.which should not be called for mock/multi/martin")

    monkeypatch.setattr(train_and_eval.shutil, "which", _unexpected)

    train_and_eval._preflight_mode("mock")
    train_and_eval._preflight_mode("multi")
    train_and_eval._preflight_mode("martin")


def test_missing_required_report_stop_reason_detects_report_protocol_failure() -> None:
    task = SimpleNamespace(
        ground_truth=SimpleNamespace(
            evaluation_checks=[
                {
                    "type": "effect",
                    "expected": {"field": "path", "value": "/reports/q01.md"},
                }
            ]
        ),
        instruction="create /reports/q01.md",
    )

    assert (
        train_and_eval._missing_required_report_stop_reason(task, ["file list --path /"])
        == "missing_required_report: /reports/q01.md"
    )
    assert (
        train_and_eval._missing_required_report_stop_reason(
            task,
            ["file create --path /reports/q01.md --content ok"],
        )
        == ""
    )


def test_required_submission_stop_reason_distinguishes_side_effect_and_anchor_failures() -> None:
    task = SimpleNamespace(
        ground_truth=SimpleNamespace(
            evaluation_checks=[
                {
                    "type": "effect",
                    "effect_type": "tasks_created",
                    "expected": {"field": "title", "value": "PRO"},
                },
                {
                    "type": "output",
                    "name": "anchor cited: EVIDENCE:X",
                    "expected": "EVIDENCE:X",
                },
            ]
        ),
        instruction="record the current blocker state with tasks add",
    )

    assert (
        train_and_eval._required_submission_stop_reason(task, [])
        == "missing_required_side_effect: tasks_created"
    )
    assert (
        train_and_eval._required_submission_stop_reason(
            task,
            ["tasks add --title PRO --priority high"],
            SimpleNamespace(
                success=False,
                details=[SimpleNamespace(passed=False, name="anchor cited: EVIDENCE:X")],
            ),
        )
        == "failed_required_anchor_check"
    )
    assert (
        train_and_eval._required_submission_stop_reason(
            task,
            ["tasks add --title PRO --priority high"],
            SimpleNamespace(
                success=False,
                details=[SimpleNamespace(passed=False, name="task board entry created")],
            ),
        )
        == "incomplete_required_submission"
    )


def test_required_submission_stop_reason_detects_email_and_action_log() -> None:
    email_task = SimpleNamespace(
        ground_truth=SimpleNamespace(
            evaluation_checks=[
                {
                    "type": "effect",
                    "effect_type": "emails_sent",
                    "expected": {"field": "body", "value": "EVIDENCE:X"},
                }
            ]
        )
    )
    action_task = SimpleNamespace(
        ground_truth=SimpleNamespace(
            evaluation_checks=[
                {
                    "type": "effect",
                    "effect_type": "files_created",
                    "expected": {"field": "path", "value": "/actions/q15.md"},
                }
            ]
        )
    )

    assert (
        train_and_eval._required_submission_stop_reason(email_task, ["email read --id pro_s01"])
        == "missing_required_side_effect: emails_sent"
    )
    assert (
        train_and_eval._required_submission_stop_reason(action_task, ["file list --path /"])
        == "missing_required_action_log: /actions/q15.md"
    )


def test_missing_required_side_effect_ignores_non_required_roles() -> None:
    # Regression: ``assets_accessed`` checks are tagged ``role="diagnostic"`` by
    # the conflict-VLM materializer and gate neither success nor score. They must
    # never be reported as a missing REQUIRED side effect. Before the fix, an
    # unsatisfied diagnostic asset check (listed first) hijacked the stop_reason
    # of tasks whose real blocker was a downstream required content check.
    task = SimpleNamespace(
        ground_truth=SimpleNamespace(
            evaluation_checks=[
                {
                    "type": "effect",
                    "effect_type": "assets_accessed",
                    "role": "diagnostic",
                    "expected": {"field": "id", "value": "onboarding_policy"},
                },
                {
                    "type": "effect",
                    "effect_type": "files_created",
                    "expected": {"field": "path", "value": "/reports/q01.md"},
                },
            ]
        ),
        instruction="inspect the chart and write /reports/q01.md",
    )

    # Agent created the required report but never opened the diagnostic asset:
    # the diagnostic miss is ignored, so nothing required is missing.
    assert (
        train_and_eval._missing_required_side_effect_stop_reason(
            task, ["file create --path /reports/q01.md --content ok"]
        )
        == ""
    )
    # A genuinely-missing REQUIRED effect is still reported.
    assert (
        train_and_eval._missing_required_side_effect_stop_reason(task, [])
        == "missing_required_report: /reports/q01.md"
    )

    # "scored" checks (partial credit, non-gating) are also not "required".
    scored_task = SimpleNamespace(
        ground_truth=SimpleNamespace(
            evaluation_checks=[
                {
                    "type": "effect",
                    "effect_type": "tasks_created",
                    "role": "scored",
                    "expected": {"field": "title", "value": "PRO"},
                }
            ]
        ),
        instruction="optionally record via tasks add",
    )
    assert (
        train_and_eval._missing_required_side_effect_stop_reason(scored_task, [])
        == ""
    )


def test_resume_successes_only_skips_only_successful_records(tmp_path: Path) -> None:
    out_file = tmp_path / "llm_total.jsonl"
    out_file.write_text(
        "\n".join(
            [
                json.dumps({"task_id": "task_ok", "success": True}),
                json.dumps({"task_id": "task_failed", "success": False}),
                json.dumps({"task_id": "task_provider", "success": False, "provider_impacted": True}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert train_and_eval._load_completed_trajectory_task_ids(out_file) == {
        "task_ok",
        "task_failed",
        "task_provider",
    }
    assert train_and_eval._load_completed_trajectory_task_ids(out_file, successes_only=True) == {
        "task_ok",
    }


def test_retry_provider_failures_only_uses_latest_record(tmp_path: Path) -> None:
    out_file = tmp_path / "llm_total.jsonl"
    out_file.write_text(
        "\n".join(
            [
                json.dumps({"task_id": "task_ok", "success": True}),
                json.dumps({"task_id": "task_logic_failed", "success": False, "error_type": "logical_failure"}),
                json.dumps({"task_id": "task_provider", "success": False, "provider_impacted": True}),
                json.dumps({"task_id": "task_429_text", "success": False, "error": "HTTP 429: too many tokens"}),
                json.dumps({"task_id": "task_recovered", "success": False, "provider_impacted": True}),
                json.dumps({"task_id": "task_recovered", "success": True}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert train_and_eval._load_provider_failed_trajectory_task_ids(out_file) == {
        "task_provider",
        "task_429_text",
    }


def test_retry_provider_failures_reads_matching_scenario_files(tmp_path: Path) -> None:
    (tmp_path / "llm_total.jsonl").write_text(
        json.dumps({"task_id": "task_a", "success": False, "provider_impacted": True}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "llm_total_privacy_deletion_request_hold.jsonl").write_text(
        json.dumps({"task_id": "task_b", "success": False, "error_type": "provider_failure"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "expert_total.jsonl").write_text(
        json.dumps({"task_id": "task_c", "success": False, "provider_impacted": True}) + "\n",
        encoding="utf-8",
    )

    provider_failed = train_and_eval._load_provider_failed_trajectory_task_ids_for_run(
        tmp_path,
        tmp_path / "llm_total.jsonl",
        "llm",
        "total",
        None,
    )

    assert provider_failed == {"task_a", "task_b"}


def test_select_retry_provider_failure_task_ids_reruns_failed_and_unrun() -> None:
    run_task_ids = [
        "task_ok",
        "task_logic_failed",
        "task_provider",
        "task_never_run",
    ]
    recorded_task_ids = {"task_ok", "task_logic_failed", "task_provider"}
    provider_failed_task_ids = {"task_provider"}

    selected = train_and_eval._select_retry_provider_failure_task_ids(
        run_task_ids, recorded_task_ids, provider_failed_task_ids
    )

    # Provider-failed + never-run are rerun (order preserved); successes and
    # genuine (non-provider) failures are skipped.
    assert selected == ["task_provider", "task_never_run"]


def test_select_retry_provider_failure_task_ids_end_to_end(tmp_path: Path) -> None:
    out_file = tmp_path / "llm_total.jsonl"
    out_file.write_text(
        "\n".join(
            [
                json.dumps({"task_id": "task_ok", "success": True}),
                json.dumps(
                    {
                        "task_id": "task_logic_failed",
                        "success": False,
                        "error_type": "logical_failure",
                    }
                ),
                json.dumps(
                    {
                        "task_id": "task_provider",
                        "success": False,
                        "provider_impacted": True,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    run_task_ids = [
        "task_ok",
        "task_logic_failed",
        "task_provider",
        "task_never_run",
    ]

    recorded = train_and_eval._load_completed_trajectory_task_ids_for_run(
        tmp_path,
        out_file,
        "llm",
        "total",
        None,
        successes_only=False,
        run_task_ids=run_task_ids,
    )
    provider_failed = (
        train_and_eval._load_provider_failed_trajectory_task_ids_for_run(
            tmp_path,
            out_file,
            "llm",
            "total",
            None,
            run_task_ids=run_task_ids,
        )
    )

    selected = train_and_eval._select_retry_provider_failure_task_ids(
        run_task_ids, recorded, provider_failed
    )

    assert selected == ["task_provider", "task_never_run"]


def test_preflight_reports_status_failures(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(train_and_eval.shutil, "which", lambda _: "/usr/bin/openclaw")
    monkeypatch.setattr(
        train_and_eval.subprocess,
        "run",
        lambda *_, **__: SimpleNamespace(returncode=1, stderr="boom", stdout=""),
    )

    with pytest.raises(RuntimeError, match="status --json"):
        train_and_eval._preflight_mode("real")


def test_cli_accepts_real_mode(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}

    _DummySummary = _make_dummy_summary(calls)

    monkeypatch.setattr(train_and_eval, "_preflight_mode", lambda mode: calls.setdefault("mode", mode))
    monkeypatch.setattr(train_and_eval, "_build_agent", lambda *_, **__: object())
    monkeypatch.setattr(
        train_and_eval,
        "run_evaluation",
        lambda *_, **kwargs: (calls.setdefault("eval_mode", kwargs["mode"]), _DummySummary())[1],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["train_and_eval.py", "--agent", "expert", "--split", "dev", "--mode", "real", "--limit", "1"],
    )

    train_and_eval.main()

    assert calls.get("mode") == "real"
    assert calls.get("eval_mode") == "real"
    assert calls.get("printed") is True


def test_cli_accepts_martin_mode(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}

    _DummySummary = _make_dummy_summary(calls)

    monkeypatch.setattr(train_and_eval, "_preflight_mode", lambda mode: calls.setdefault("mode", mode))
    monkeypatch.setattr(train_and_eval, "_build_agent", lambda *_, **__: object())
    monkeypatch.setattr(
        train_and_eval,
        "run_evaluation",
        lambda *_, **kwargs: (calls.setdefault("eval_mode", kwargs["mode"]), _DummySummary())[1],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["train_and_eval.py", "--agent", "expert", "--split", "dev", "--mode", "martin", "--limit", "1"],
    )

    train_and_eval.main()

    assert calls.get("mode") == "martin"
    assert calls.get("eval_mode") == "martin"
    assert calls.get("printed") is True


def test_cli_accepts_online_flags(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}

    _DummySummary = _make_dummy_summary(calls)

    monkeypatch.setattr(train_and_eval, "_preflight_mode", lambda mode: calls.setdefault("mode", mode))
    monkeypatch.setattr(train_and_eval, "_build_agent", lambda *_, **__: object())
    monkeypatch.setattr(
        train_and_eval,
        "run_evaluation",
        lambda *_, **kwargs: (
            calls.setdefault("online_clean", kwargs["online_clean"]),
            calls.setdefault("skip_incompatible_openclaw", kwargs["skip_incompatible_openclaw"]),
            calls.setdefault("online_openclaw_only", kwargs["online_openclaw_only"]),
            _DummySummary(),
        )[3],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent",
            "expert",
            "--split",
            "dev",
            "--mode",
            "hybrid",
            "--online-clean",
            "--online-openclaw-only",
            "--skip-incompatible-openclaw",
            "--limit",
            "1",
        ],
    )

    train_and_eval.main()

    assert calls.get("mode") == "hybrid"
    assert calls.get("online_clean") is True
    assert calls.get("skip_incompatible_openclaw") is True
    assert calls.get("online_openclaw_only") is True
    assert calls.get("printed") is True


def test_cli_defaults_max_steps_to_fifteen(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}

    _DummySummary = _make_dummy_summary(calls)

    monkeypatch.setattr(train_and_eval, "_preflight_mode", lambda mode: None)
    monkeypatch.setattr(
        train_and_eval,
        "_build_agent",
        lambda *_, **kwargs: (calls.setdefault("max_steps_hint", kwargs["max_steps_hint"]), object())[1],
    )
    monkeypatch.setattr(
        train_and_eval,
        "run_evaluation",
        lambda *_, **kwargs: (calls.setdefault("max_steps", kwargs["max_steps"]), _DummySummary())[1],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["train_and_eval.py", "--agent", "expert", "--split", "dev", "--limit", "1"],
    )

    train_and_eval.main()

    assert calls.get("max_steps") == 15
    assert calls.get("max_steps_hint") == 15
    assert calls.get("printed") is True


def test_cli_total_split_keeps_default_max_steps_to_fifteen(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}

    _DummySummary = _make_dummy_summary(calls)

    monkeypatch.setattr(train_and_eval, "_preflight_mode", lambda mode: None)
    monkeypatch.setattr(
        train_and_eval,
        "_build_agent",
        lambda *_, **kwargs: (calls.setdefault("max_steps_hint", kwargs["max_steps_hint"]), object())[1],
    )
    monkeypatch.setattr(
        train_and_eval,
        "run_evaluation",
        lambda *_, **kwargs: (calls.setdefault("max_steps", kwargs["max_steps"]), _DummySummary())[1],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["train_and_eval.py", "--agent", "expert", "--split", "total", "--limit", "1"],
    )

    train_and_eval.main()

    assert calls.get("max_steps") == 15
    assert calls.get("max_steps_hint") == 15
    assert calls.get("printed") is True


def test_cli_accepts_task_data_dir(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}

    _DummySummary = _make_dummy_summary(calls)

    monkeypatch.setattr(train_and_eval, "_preflight_mode", lambda mode: calls.setdefault("mode", mode))
    monkeypatch.setattr(train_and_eval, "_build_agent", lambda *_, **__: object())
    monkeypatch.setattr(
        train_and_eval,
        "run_evaluation",
        lambda *_, **kwargs: (
            calls.setdefault("task_data_dir", kwargs["task_data_dir"]),
            _DummySummary(),
        )[1],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent",
            "expert",
            "--split",
            "dev",
            "--mode",
            "hybrid",
            "--task-data-dir",
            "/tmp/profile_local",
            "--limit",
            "1",
        ],
    )

    train_and_eval.main()

    assert calls.get("mode") == "hybrid"
    assert calls.get("task_data_dir") == "/tmp/profile_local"
    assert calls.get("printed") is True


def test_cli_accepts_task_prefix(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}

    _DummySummary = _make_dummy_summary(calls)

    monkeypatch.setattr(train_and_eval, "_preflight_mode", lambda mode: calls.setdefault("mode", mode))
    monkeypatch.setattr(train_and_eval, "_build_agent", lambda *_, **__: object())
    monkeypatch.setattr(
        train_and_eval,
        "run_evaluation",
        lambda *_, **kwargs: (
            calls.setdefault("task_prefix", kwargs["task_prefix"]),
            _DummySummary(),
        )[1],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent",
            "expert",
            "--split",
            "dev",
            "--task-prefix",
            "hard_decision_workflow_",
            "--limit",
            "1",
        ],
    )

    train_and_eval.main()

    assert calls.get("mode") == "multi"
    assert calls.get("task_prefix") == "hard_decision_workflow_"
    assert calls.get("printed") is True


def test_cli_accepts_scenario_slug(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}

    _DummySummary = _make_dummy_summary(calls)

    monkeypatch.setattr(train_and_eval, "_preflight_mode", lambda mode: calls.setdefault("mode", mode))
    monkeypatch.setattr(train_and_eval, "_build_agent", lambda *_, **__: object())
    monkeypatch.setattr(
        train_and_eval,
        "run_evaluation",
        lambda *_, **kwargs: (
            calls.setdefault("scenario_slugs", kwargs["scenario_slugs"]),
            _DummySummary(),
        )[1],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent",
            "expert",
            "--split",
            "total",
            "--scenario-slug",
            "privacy_deletion_request_hold",
        ],
    )

    train_and_eval.main()

    assert calls.get("mode") == "multi"
    assert calls.get("scenario_slugs") == ["privacy_deletion_request_hold"]
    assert calls.get("printed") is True


def test_cli_accepts_custom_dataset_split_names(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}

    _DummySummary = _make_dummy_summary(calls)

    monkeypatch.setattr(train_and_eval, "_preflight_mode", lambda mode: calls.setdefault("mode", mode))
    monkeypatch.setattr(train_and_eval, "_build_agent", lambda *_, **__: object())
    monkeypatch.setattr(
        train_and_eval,
        "run_evaluation",
        lambda *_, **kwargs: (
            calls.setdefault("split", kwargs["split"]),
            _DummySummary(),
        )[1],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent",
            "expert",
            "--split",
            "duplicate_avoidance_followthrough",
            "--limit",
            "1",
        ],
    )

    train_and_eval.main()

    assert calls.get("mode") == "multi"
    assert calls.get("split") == "duplicate_avoidance_followthrough"
    assert calls.get("printed") is True


def test_filter_task_ids_by_scenario_slugs_uses_public_metadata(monkeypatch: pytest.MonkeyPatch):
    tasks = {
        "task_a": SimpleNamespace(data=SimpleNamespace(public={"scenario_slug": "alpha"})),
        "task_b": SimpleNamespace(data=SimpleNamespace(public={"scenario_slug": "beta"})),
        "task_c": SimpleNamespace(data=SimpleNamespace(public={"hard_decision_scenario": "alpha"})),
    }
    monkeypatch.setattr(
        train_and_eval,
        "load_task",
        lambda task_id, data_dir=None: tasks[task_id],
    )

    assert train_and_eval._filter_task_ids_by_scenario_slugs(
        ["task_a", "task_b", "task_c"],
        ["alpha"],
    ) == ["task_a", "task_c"]


def test_filter_task_ids_by_scenario_slugs_matches_base_slug_across_versions(
    monkeypatch: pytest.MonkeyPatch,
):
    # conflict_vlm scenario slugs are versioned ("foo_v01"); a base slug should
    # select every version, without leaking into a different scenario whose name
    # merely starts with the same prefix ("foobar").
    tasks = {
        "t_v01": SimpleNamespace(data=SimpleNamespace(public={"scenario_slug": "foo_v01"})),
        "t_v02": SimpleNamespace(data=SimpleNamespace(public={"scenario_slug": "foo_v02"})),
        "t_other": SimpleNamespace(data=SimpleNamespace(public={"scenario_slug": "foobar_v01"})),
    }
    monkeypatch.setattr(
        train_and_eval,
        "load_task",
        lambda task_id, data_dir=None: tasks[task_id],
    )
    ids = ["t_v01", "t_v02", "t_other"]

    # Base slug selects all versions (and does not match "foobar_v01").
    assert train_and_eval._filter_task_ids_by_scenario_slugs(ids, ["foo"]) == [
        "t_v01",
        "t_v02",
    ]
    # A versioned slug still selects only that one version (backward compatible).
    assert train_and_eval._filter_task_ids_by_scenario_slugs(
        ids, ["foo_v01"]
    ) == ["t_v01"]


def test_trajectory_output_file_uses_scenario_slug(tmp_path: Path):
    assert train_and_eval._trajectory_output_file(
        tmp_path,
        "llm",
        "total",
        ["privacy_deletion_request_hold"],
    ) == tmp_path / "llm_total_privacy_deletion_request_hold.jsonl"


def test_resume_without_scenario_reads_all_matching_trajectory_files(tmp_path: Path):
    (tmp_path / "llm_total.jsonl").write_text(
        json.dumps({"task_id": "task_a"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "llm_total_privacy_deletion_request_hold.jsonl").write_text(
        json.dumps({"task_id": "task_b"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "expert_total.jsonl").write_text(
        json.dumps({"task_id": "task_c"}) + "\n",
        encoding="utf-8",
    )

    completed = train_and_eval._load_completed_trajectory_task_ids_for_run(
        tmp_path,
        tmp_path / "llm_total.jsonl",
        "llm",
        "total",
        None,
    )

    assert completed == {"task_a", "task_b"}


def test_episode_result_record_roundtrips_through_to_dict() -> None:
    res = train_and_eval.EpisodeResult(
        task_id="conflict_vlm_scenario_a_r01",
        agent_name="llm",
        success=True,
        score=0.5,
        steps=3,
        duration_s=1.25,
        trajectory=[{"action": "look"}],
        final_response="ok",
        error="",
        error_type="",
        provider_impacted=False,
        stop_reason="agent requested stop",
        date_anchor_mismatch=False,
        mode="multi",
        metadata={"k": "v"},
    )

    assert train_and_eval._episode_result_from_record(res.to_dict()) == res


def test_aggregate_summary_over_out_dir_uses_latest_and_ignores_other_agents(
    tmp_path: Path,
) -> None:
    # Parallel naming ({slug}.jsonl), and the latest record per task wins:
    # r01 recovered from a provider failure to success on its second attempt.
    (tmp_path / "scenario_a.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"task_id": "conflict_vlm_scenario_a_r01", "agent": "llm", "success": False, "score": 0.2, "error_type": "provider_failure", "provider_impacted": True}),
                json.dumps({"task_id": "conflict_vlm_scenario_a_r01", "agent": "llm", "success": True, "score": 1.0}),
                json.dumps({"task_id": "conflict_vlm_scenario_a_r02", "agent": "llm", "success": False, "score": 0.4, "error_type": "logical_failure"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    # A different agent's file in the same dir must be excluded from the aggregate.
    (tmp_path / "scenario_b.jsonl").write_text(
        json.dumps({"task_id": "conflict_vlm_scenario_b_r01", "agent": "expert", "success": True, "score": 1.0}) + "\n",
        encoding="utf-8",
    )

    template = train_and_eval.EvalSummary(
        agent_name="llm", split="dev", total=0, passed=0, total_score=0.0
    )
    full = train_and_eval._aggregate_summary_over_out_dir(
        template, tmp_path, "llm", None
    )

    assert full.total == 2  # only the two llm tasks, not the expert one
    assert full.passed == 1  # r01's latest record is a success
    assert full.provider_failures == 0  # r01 recovered; latest is not a failure
    assert full.provider_impacted_tasks == 0


def test_cli_verbose_log_tees_stdout_to_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    class _DummySummary:
        def print_report(self) -> None:
            print("REPORT LINE")

    monkeypatch.setattr(train_and_eval, "_preflight_mode", lambda mode: None)
    monkeypatch.setattr(train_and_eval, "_build_agent", lambda *_, **__: object())
    monkeypatch.setattr(train_and_eval, "run_evaluation", lambda *_, **__: _DummySummary())

    log_path = tmp_path / "verbose.log"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent",
            "expert",
            "--split",
            "dev",
            "--limit",
            "1",
            "--verbose-log",
            str(log_path),
        ],
    )

    train_and_eval.main()

    captured = capsys.readouterr()
    assert "REPORT LINE" in captured.out
    assert log_path.exists()
    assert "REPORT LINE" in log_path.read_text(encoding="utf-8")




def test_cli_accepts_inter_task_sleep(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}

    _DummySummary = _make_dummy_summary(calls)

    monkeypatch.setattr(train_and_eval, "_preflight_mode", lambda mode: calls.setdefault("mode", mode))
    monkeypatch.setattr(train_and_eval, "_build_agent", lambda *_, **__: object())
    monkeypatch.setattr(
        train_and_eval,
        "run_evaluation",
        lambda *_, **kwargs: (
            calls.setdefault("inter_task_sleep_s", kwargs["inter_task_sleep_s"]),
            _DummySummary(),
        )[1],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent",
            "expert",
            "--split",
            "dev",
            "--inter-task-sleep",
            "2.5",
            "--limit",
            "1",
        ],
    )

    train_and_eval.main()

    assert calls.get("mode") == "multi"
    assert calls.get("inter_task_sleep_s") == 2.5
    assert calls.get("printed") is True


def test_run_evaluation_sleeps_between_tasks(monkeypatch: pytest.MonkeyPatch):
    task_ids = ["agent_create_1", "tasks_add_1", "file_create_1"]
    sleeps: list[float] = []

    monkeypatch.setattr(train_and_eval, "load_task_ids", lambda *_, **__: list(task_ids))
    monkeypatch.setattr(
        train_and_eval,
        "run_episode",
        lambda task_id, *_, **__: train_and_eval.EpisodeResult(
            task_id=task_id,
            agent_name="expert",
            success=True,
            score=1.0,
            steps=1,
            duration_s=0.0,
        ),
    )
    monkeypatch.setattr(train_and_eval.time, "sleep", lambda seconds: sleeps.append(seconds))

    summary = train_and_eval.run_evaluation(
        agent=train_and_eval.ExpertAgent(),
        split="dev",
        inter_task_sleep_s=1.25,
        verbose=False,
    )

    assert summary.total == 3
    assert sleeps == [1.25, 1.25]

def test_cli_accepts_openai_llm_options(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}

    _DummySummary = _make_dummy_summary(calls)

    def _build_agent(*args, **kwargs):
        calls["build_agent_args"] = args
        calls["build_agent_kwargs"] = kwargs
        return object()

    monkeypatch.setattr(train_and_eval, "_preflight_mode", lambda mode: calls.setdefault("mode", mode))
    monkeypatch.setattr(train_and_eval, "_build_agent", _build_agent)
    monkeypatch.setattr(train_and_eval, "run_evaluation", lambda *_, **__: _DummySummary())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent",
            "llm",
            "--model",
            "gpt-5.2",
            "--llm-provider",
            "openai",
            "--llm-base-url",
            "https://example.test/v1",
            "--llm-api-key-env",
            "CUSTOM_API_KEY",
            "--llm-history-mode",
            "summary",
            "--llm-temperature",
            "0.3",
            "--llm-max-tokens",
            "384",
            "--llm-timeout-s",
            "17",
            "--llm-request-retries",
            "2",
            "--llm-retry-backoff-s",
            "2.5",
            "--limit",
            "1",
        ],
    )

    train_and_eval.main()

    assert calls.get("mode") == "multi"
    assert calls["build_agent_args"] == ("llm", "gpt-5.2")
    assert calls["build_agent_kwargs"] == {
        "llm_provider": "openai",
        "llm_base_url": "https://example.test/v1",
        "llm_api_key_env": "CUSTOM_API_KEY",
        "llm_history_mode": "summary",
        "llm_temperature": 0.3,
        "llm_max_tokens": 384,
        "llm_timeout_s": 17,
        "llm_request_retries": 2,
        "llm_empty_reply_retries": 5,
        "llm_retry_backoff_s": 2.5,
        "llm_transient_max_backoff_s": 12.0,
        "llm_deadline_retry_backoff_s": 60.0,
        "llm_deadline_max_backoff_s": 60.0,
        "multimodal_input_mode": "both",
        "max_steps_hint": 15,
        "verbose": False,
        "vertex_project": None,
        "vertex_location": None,
        "anthropic_max_image_edge": "auto",
        "gemini_max_image_edge": None,
        # Unset by default: Gemini keeps its own server-side per-image budget
        # (1120 visual tokens, == "high"). Pinned so raising it is deliberate.
        "gemini_media_resolution": None,
        "openai_max_image_edge": 1920,
        "capture_thinking": False,
        "capture_interactions": False,
        "llm_allow_compact_fallback": True,
        "llm_tool_mode": "native",
        "llm_tool_schema": "native",
        "llm_native_reasoning": train_and_eval.DEFAULT_LLM_NATIVE_REASONING,
        # Unset by default: each vendor keeps its OWN default effort (Anthropic
        # high, gpt-5.5 medium), which is what every existing result set was
        # produced with. Pinned here so flipping the default is a deliberate act.
        "llm_reasoning_effort": None,
        # Both unset by default: these reach reasoning control on the Chat
        # Completions path only (open-weight models), and every hosted-vendor
        # result set was produced without them. Pinned so enabling either is
        # a deliberate act.
        "llm_chat_template_kwargs": None,
        "llm_replay_chat_reasoning": False,
    }
    assert calls.get("printed") is True


def _run_main_capturing_build_agent_kwargs(
    monkeypatch: pytest.MonkeyPatch, extra_argv: list[str]
) -> dict[str, object]:
    """Run ``main()`` with ``_build_agent`` stubbed, returning the captured kwargs.

    The base argv selects the llm agent against an openai model; ``extra_argv``
    appends flags under test. No network is touched because ``_build_agent`` and
    ``run_evaluation`` are both replaced with capturing stubs.
    """
    calls: dict[str, object] = {}
    _DummySummary = _make_dummy_summary(calls)

    def _build_agent(*args, **kwargs):
        calls["build_agent_args"] = args
        calls["build_agent_kwargs"] = kwargs
        return object()

    monkeypatch.setattr(
        train_and_eval, "_preflight_mode", lambda mode: calls.setdefault("mode", mode)
    )
    monkeypatch.setattr(train_and_eval, "_build_agent", _build_agent)
    monkeypatch.setattr(train_and_eval, "run_evaluation", lambda *_, **__: _DummySummary())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent",
            "llm",
            "--model",
            "gpt-5.2",
            "--llm-provider",
            "openai",
            "--limit",
            "1",
        ]
        + extra_argv,
    )

    train_and_eval.main()
    return calls


def test_cli_llm_empty_reply_retries_defaults_to_five(monkeypatch: pytest.MonkeyPatch):
    calls = _run_main_capturing_build_agent_kwargs(monkeypatch, [])

    kwargs = calls["build_agent_kwargs"]
    assert kwargs["llm_empty_reply_retries"] == 5


def test_cli_llm_empty_reply_retries_is_threaded(monkeypatch: pytest.MonkeyPatch):
    calls = _run_main_capturing_build_agent_kwargs(
        monkeypatch, ["--llm-empty-reply-retries", "3"]
    )

    kwargs = calls["build_agent_kwargs"]
    assert kwargs["llm_empty_reply_retries"] == 3


def test_cli_llm_tool_mode_defaults_to_native(monkeypatch: pytest.MonkeyPatch):
    calls = _run_main_capturing_build_agent_kwargs(monkeypatch, [])

    kwargs = calls["build_agent_kwargs"]
    assert kwargs["llm_tool_mode"] == "native"


def test_cli_llm_tool_mode_text_is_threaded(monkeypatch: pytest.MonkeyPatch):
    calls = _run_main_capturing_build_agent_kwargs(
        monkeypatch, ["--llm-tool-mode", "text"]
    )

    kwargs = calls["build_agent_kwargs"]
    assert kwargs["llm_tool_mode"] == "text"


def test_cli_llm_tool_mode_rejects_unknown_choice(monkeypatch: pytest.MonkeyPatch):
    # argparse rejects values outside the {native, text} choice set with SystemExit.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent",
            "llm",
            "--model",
            "gpt-5.2",
            "--llm-provider",
            "openai",
            "--llm-tool-mode",
            "bogus",
            "--limit",
            "1",
        ],
    )
    with pytest.raises(SystemExit):
        train_and_eval.main()


def test_build_agent_threads_tool_mode_into_llm_agent():
    # _build_agent constructs an LLMAgent lazily (no network at construction), so
    # we can assert the tool_mode plumbing end-to-end.
    native_agent = train_and_eval._build_agent(
        "llm", "gpt-5.2", llm_provider="openai", llm_tool_mode="native"
    )
    assert native_agent._tool_mode == "native"

    text_agent = train_and_eval._build_agent(
        "llm", "gpt-5.2", llm_provider="openai", llm_tool_mode="text"
    )
    assert text_agent._tool_mode == "text"


def test_build_agent_tool_mode_defaults_to_native():
    agent = train_and_eval._build_agent("llm", "gpt-5.2", llm_provider="openai")
    assert agent._tool_mode == "native"


def test_cli_llm_native_reasoning_defaults(monkeypatch: pytest.MonkeyPatch):
    calls = _run_main_capturing_build_agent_kwargs(monkeypatch, [])

    kwargs = calls["build_agent_kwargs"]
    assert (
        kwargs["llm_native_reasoning"]
        == train_and_eval.DEFAULT_LLM_NATIVE_REASONING
    )


def test_cli_no_llm_native_reasoning_is_threaded(monkeypatch: pytest.MonkeyPatch):
    calls = _run_main_capturing_build_agent_kwargs(
        monkeypatch, ["--no-llm-native-reasoning"]
    )

    kwargs = calls["build_agent_kwargs"]
    assert kwargs["llm_native_reasoning"] is False


def test_build_agent_threads_native_reasoning_into_llm_agent():
    agent = train_and_eval._build_agent(
        "llm", "gpt-5.2", llm_provider="openai", llm_native_reasoning=False
    )
    assert agent._native_reasoning is False


def test_cli_llm_tool_schema_defaults_to_native(monkeypatch: pytest.MonkeyPatch):
    calls = _run_main_capturing_build_agent_kwargs(monkeypatch, [])

    kwargs = calls["build_agent_kwargs"]
    assert kwargs["llm_tool_schema"] == "native"


def test_cli_llm_tool_schema_unified_is_threaded(monkeypatch: pytest.MonkeyPatch):
    calls = _run_main_capturing_build_agent_kwargs(
        monkeypatch, ["--llm-tool-schema", "unified"]
    )

    kwargs = calls["build_agent_kwargs"]
    assert kwargs["llm_tool_schema"] == "unified"


def test_cli_llm_tool_schema_rejects_unknown_choice(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent",
            "llm",
            "--model",
            "gpt-5.2",
            "--llm-provider",
            "openai",
            "--llm-tool-schema",
            "bogus",
            "--limit",
            "1",
        ],
    )
    with pytest.raises(SystemExit):
        train_and_eval.main()


def test_build_agent_threads_tool_schema_into_llm_agent():
    native = train_and_eval._build_agent(
        "llm", "gpt-5.2", llm_provider="openai", llm_tool_schema="native"
    )
    assert native._tool_schema_mode == "native"

    unified = train_and_eval._build_agent(
        "llm", "gpt-5.2", llm_provider="openai", llm_tool_schema="unified"
    )
    assert unified._tool_schema_mode == "unified"


def test_build_agent_tool_schema_defaults_to_native():
    agent = train_and_eval._build_agent("llm", "gpt-5.2", llm_provider="openai")
    assert agent._tool_schema_mode == "native"


def test_llm_agent_full_history_mode_keeps_messages_for_openai():
    agent = train_and_eval.LLMAgent(provider="openai", model="gpt-5.2", history_mode="full")
    agent._history = [
        {"action": "tasks list --status pending", "stdout": "Tasks...", "stderr": "", "exit_code": 0},
    ]
    messages = [
        {"role": "user", "content": "Task: first"},
        {"role": "assistant", "content": "cmd-1"},
        {"role": "user", "content": "obs-1"},
        {"role": "assistant", "content": "cmd-2"},
        {"role": "user", "content": "obs-2"},
    ]

    assert agent._request_messages(messages) == messages


def test_llm_agent_summary_history_mode_trims_messages_for_anthropic():
    agent = train_and_eval.LLMAgent(provider="anthropic", model="claude-haiku-4-5-20251001", history_mode="summary")
    agent._history = [
        {"action": "weather forecast --location Berlin --days 1", "stdout": "Rain tomorrow", "stderr": "", "exit_code": 0},
        {"action": "calendar today --timezone Europe/Berlin", "stdout": "Review at 14:00", "stderr": "", "exit_code": 0},
        {"action": "tasks list --status pending", "stdout": "1. Berlin follow-up", "stderr": "", "exit_code": 0},
    ]
    messages = [
        {"role": "user", "content": "Task: first"},
        {"role": "assistant", "content": "cmd-1"},
        {"role": "user", "content": "obs-1"},
        {"role": "assistant", "content": "cmd-2"},
        {"role": "user", "content": "obs-2"},
        {"role": "assistant", "content": "cmd-3"},
        {"role": "user", "content": "obs-3"},
        {"role": "assistant", "content": "cmd-4"},
        {"role": "user", "content": "obs-4"},
        {"role": "assistant", "content": "cmd-5"},
        {"role": "user", "content": "obs-5"},
    ]

    trimmed = agent._request_messages(messages)

    assert trimmed[0] == messages[0]
    assert trimmed[1]["role"] == "user"
    assert trimmed[1]["content"].startswith("Known context:\n")
    assert len(trimmed) < len(messages)


def test_llm_agent_auto_history_mode_is_provider_agnostic():
    history = [
        {"action": "weather forecast --location Berlin --days 1", "stdout": "Rain tomorrow", "stderr": "", "exit_code": 0},
        {"action": "calendar today --timezone Europe/Berlin", "stdout": "Review at 14:00", "stderr": "", "exit_code": 0},
        {"action": "tasks list --status pending", "stdout": "1. Berlin follow-up", "stderr": "", "exit_code": 0},
    ]
    messages = [
        {"role": "user", "content": "Task: first"},
        {"role": "assistant", "content": "cmd-1"},
        {"role": "user", "content": "obs-1"},
        {"role": "assistant", "content": "cmd-2"},
        {"role": "user", "content": "obs-2"},
        {"role": "assistant", "content": "cmd-3"},
        {"role": "user", "content": "obs-3"},
        {"role": "assistant", "content": "cmd-4"},
        {"role": "user", "content": "obs-4"},
        {"role": "assistant", "content": "cmd-5"},
        {"role": "user", "content": "obs-5"},
    ]
    openai_agent = train_and_eval.LLMAgent(provider="openai", model="gpt-5.2", history_mode="auto")
    anthropic_agent = train_and_eval.LLMAgent(provider="anthropic", model="claude-haiku-4-5-20251001", history_mode="auto")
    openai_agent._history = list(history)
    anthropic_agent._history = list(history)

    assert openai_agent._request_messages(messages) == anthropic_agent._request_messages(messages)


def test_cli_accepts_fallback_openclaw_network_to_mock_flag(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}

    _DummySummary = _make_dummy_summary(calls)

    monkeypatch.setattr(train_and_eval, "_preflight_mode", lambda mode: calls.setdefault("mode", mode))
    monkeypatch.setattr(train_and_eval, "_build_agent", lambda *_, **__: object())
    monkeypatch.setattr(
        train_and_eval,
        "run_evaluation",
        lambda *_, **kwargs: (
            calls.setdefault(
                "fallback_openclaw_network_to_mock",
                kwargs["fallback_openclaw_network_to_mock"],
            ),
            _DummySummary(),
        )[1],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent",
            "expert",
            "--split",
            "dev",
            "--mode",
            "hybrid",
            "--fallback-openclaw-network-to-mock",
            "--limit",
            "1",
        ],
    )

    train_and_eval.main()

    assert calls.get("mode") == "hybrid"
    assert calls.get("fallback_openclaw_network_to_mock") is True
    assert calls.get("printed") is True


def test_cli_accepts_strict_online_data_flag(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}

    _DummySummary = _make_dummy_summary(calls)

    monkeypatch.setattr(train_and_eval, "_preflight_mode", lambda mode: calls.setdefault("mode", mode))
    monkeypatch.setattr(train_and_eval, "_build_agent", lambda *_, **__: object())
    monkeypatch.setattr(
        train_and_eval,
        "run_evaluation",
        lambda *_, **kwargs: (
            calls.setdefault("strict_online_data", kwargs["strict_online_data"]),
            _DummySummary(),
        )[1],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent",
            "expert",
            "--split",
            "dev",
            "--mode",
            "hybrid",
            "--strict-online-data",
            "--limit",
            "1",
        ],
    )

    train_and_eval.main()

    assert calls.get("mode") == "hybrid"
    assert calls.get("strict_online_data") is True
    assert calls.get("printed") is True


def test_run_episode_records_extended_trajectory_fields():
    result = train_and_eval.run_episode(
        task_id="agent_create_1",
        agent=train_and_eval.ExpertAgent(),
        mode="mock",
        max_steps=3,
        record_trajectory=True,
        online_clean=True,
        verbose=False,
    )

    assert result.trajectory


def test_openai_llm_agent_calls_chat_completions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    captured: dict[str, object] = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {"choices": [{"message": {"content": "weather get --location 'Paris'"}}]}
            ).encode("utf-8")

    def _urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(train_and_eval.urllib.request, "urlopen", _urlopen)

    agent = train_and_eval.LLMAgent(
        provider="openai",
        model="legacy-openai-chat-model",
        base_url="https://example.test/v1",
    )
    agent.reset("Check the weather in Paris.")
    observation = SimpleNamespace(
        step_number=0,
        command_output="",
        error_output="",
        exit_code=0,
    )

    action = agent.act(observation)

    assert action == "weather get --location 'Paris'"
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["method"] == "POST"
    assert captured["body"]["model"] == "legacy-openai-chat-model"
    assert captured["body"]["max_tokens"] == 256
    assert "max_completion_tokens" not in captured["body"]
    assert captured["body"]["messages"][0]["role"] == "system"
    assert captured["body"]["messages"][1]["role"] == "user"
    assert captured["headers"]["Authorization"] == "Bearer test-key"


def test_openai_gpt_models_use_responses_api(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    captured: dict[str, object] = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "object": "response",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": "weather get --location 'Paris'"}
                            ],
                        }
                    ],
                }
            ).encode("utf-8")

    def _urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _Response()

    monkeypatch.setattr(train_and_eval.urllib.request, "urlopen", _urlopen)

    agent = train_and_eval.LLMAgent(
        provider="openai",
        model="gpt-5.2",
        base_url="https://example.test/v1",
    )
    agent.reset("Check the weather in Paris.")

    action = agent.act(SimpleNamespace(step_number=0, command_output="", error_output="", exit_code=0))

    assert action == "weather get --location 'Paris'"
    assert captured["url"] == "https://example.test/v1/responses"
    assert captured["body"]["model"] == "gpt-5.2"
    assert captured["body"]["max_output_tokens"] == 256
    assert captured["body"]["instructions"] == agent._system_prompt
    assert captured["body"]["input"][0]["role"] == "user"


def test_to_responses_input_converts_multimodal_parts():
    agent = train_and_eval.LLMAgent(
        provider="openai",
        model="gpt-5",
        base_url="https://example.test/v1",
    )
    data_url = "data:image/png;base64,ABC123"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the chart."},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
        {"role": "assistant", "content": "chart render --id 1"},
    ]

    converted = agent._to_responses_input(messages)

    # User text/image parts are rewritten to the Responses API schema.
    assert converted[0]["role"] == "user"
    assert converted[0]["content"] == [
        {"type": "input_text", "text": "Describe the chart."},
        {"type": "input_image", "image_url": data_url},
    ]
    # Assistant messages with plain string content are passed through untouched.
    assert converted[1] == {"role": "assistant", "content": "chart render --id 1"}


def test_openai_responses_api_sends_input_image_and_omits_temperature_for_gpt5(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    captured: dict[str, object] = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "object": "response",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": "chart render --id 1"}
                            ],
                        }
                    ],
                }
            ).encode("utf-8")

    def _urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _Response()

    monkeypatch.setattr(train_and_eval.urllib.request, "urlopen", _urlopen)

    agent = train_and_eval.LLMAgent(
        provider="openai",
        model="gpt-5",
        base_url="https://example.test/v1",
    )
    data_url = "data:image/png;base64,ABC123"
    # Mirror the user message reset() builds when a task carries image assets.
    agent._messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the chart."},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]

    assert agent._openai_response_text() == "chart render --id 1"
    assert captured["url"] == "https://example.test/v1/responses"
    body = captured["body"]
    # Multimodal parts must reach the wire in Responses format so gpt-5 sees the
    # image instead of the endpoint rejecting the chat-shaped parts with a 400.
    assert body["input"][0]["content"] == [
        {"type": "input_text", "text": "Describe the chart."},
        {"type": "input_image", "image_url": data_url},
    ]
    # gpt-5 only accepts its default temperature, so it must be omitted entirely.
    assert "temperature" not in body


def test_openai_responses_api_deduplicates_adjacent_command_parts(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "object": "response",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": "email search --query budget"},
                                {"type": "output_text", "text": "email search --query budget"},
                            ],
                        }
                    ],
                }
            ).encode("utf-8")

    monkeypatch.setattr(
        train_and_eval.urllib.request,
        "urlopen",
        lambda req, timeout=0: _Response(),
    )

    agent = train_and_eval.LLMAgent(
        provider="openai",
        model="gpt-5.4",
        base_url="https://example.test/v1",
    )
    task = train_and_eval.load_task(
        "hard_decision_workflow_1",
        data_dir=Path(train_and_eval.DEFAULT_TASK_DATA_DIR),
    )
    agent.reset(task.instruction, task=task)

    assert agent._openai_response_text() == "email search --query budget"
    assert any("deduped 1 adjacent duplicate" in item for item in agent.debug_reply_attempts())


def test_openai_responses_api_preserves_split_command_chunks(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "object": "response",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": "email search --query "},
                                {"type": "output_text", "text": "budget"},
                            ],
                        }
                    ],
                }
            ).encode("utf-8")

    monkeypatch.setattr(
        train_and_eval.urllib.request,
        "urlopen",
        lambda req, timeout=0: _Response(),
    )

    agent = train_and_eval.LLMAgent(
        provider="openai",
        model="gpt-5.4",
        base_url="https://example.test/v1",
    )
    task = train_and_eval.load_task(
        "hard_decision_workflow_1",
        data_dir=Path(train_and_eval.DEFAULT_TASK_DATA_DIR),
    )
    agent.reset(task.instruction, task=task)

    assert agent._openai_response_text() == "email search --query budget"



def test_openai_request_falls_back_to_responses_when_chat_is_unsupported(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls: list[str] = []

    class _Body:
        def __init__(self, payload: str) -> None:
            self._payload = payload.encode("utf-8")

        def read(self) -> bytes:
            return self._payload

        def close(self) -> None:
            return None

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "object": "response",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": "tasks list --status pending"}
                            ],
                        }
                    ],
                }
            ).encode("utf-8")

    def _urlopen(req, timeout=0):
        calls.append(req.full_url)
        if req.full_url.endswith("/chat/completions"):
            raise train_and_eval.urllib.error.HTTPError(
                req.full_url,
                400,
                "Bad Request",
                hdrs=None,
                fp=_Body('{"error":{"message":"The requested operation is unsupported."}}'),
            )
        return _Response()

    monkeypatch.setattr(train_and_eval.urllib.request, "urlopen", _urlopen)

    agent = train_and_eval.LLMAgent(
        provider="openai",
        model="legacy-openai-chat-model",
        base_url="https://example.test/v1",
        request_retries=0,
    )
    agent.reset("List pending tasks.")

    assert agent._openai_response_text() == "tasks list --status pending"
    assert calls == [
        "https://example.test/v1/chat/completions",
        "https://example.test/v1/responses",
    ]
    assert any("unsupported; retrying with responses API" in item for item in agent.debug_reply_attempts())


def test_openai_response_text_returns_empty_for_null_content(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {"choices": [{"message": {"content": None, "role": "assistant"}}]}
            ).encode("utf-8")

    monkeypatch.setattr(
        train_and_eval.urllib.request,
        "urlopen",
        lambda req, timeout=0: _Response(),
    )

    agent = train_and_eval.LLMAgent(
        provider="openai",
        model="legacy-openai-chat-model",
        base_url="https://example.test/v1",
    )
    agent.reset("Check the weather in Paris.")

    # A contentless (HTTP 200) reply is no longer a hard error: the provider layer
    # exhausts the small empty-reply budget and then degrades to "" so the caller
    # records a generation stall (kept a provider_failure), mirroring Anthropic/Gemini.
    assert agent._openai_response_text() == ""
    assert any("api-empty:" in line for line in agent.debug_reply_attempts())


def test_openai_request_backs_off_before_retry(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls = {"count": 0}
    sleeps: list[float] = []

    class _Body:
        def __init__(self, payload: str) -> None:
            self._payload = payload.encode("utf-8")

        def read(self) -> bytes:
            return self._payload

        def close(self) -> None:
            return None

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {"choices": [{"message": {"content": "weather get --location 'Paris'", "role": "assistant"}}]}
            ).encode("utf-8")

    def _urlopen(req, timeout=0):
        calls["count"] += 1
        if calls["count"] == 1:
            raise train_and_eval.urllib.error.HTTPError(
                req.full_url,
                429,
                "Too Many Requests",
                hdrs=None,
                fp=_Body('{"error":{"message":"queue is full"}}'),
            )
        return _Response()

    monkeypatch.setattr(train_and_eval.urllib.request, "urlopen", _urlopen)
    monkeypatch.setattr(train_and_eval.time, "sleep", lambda seconds: sleeps.append(seconds))

    agent = train_and_eval.LLMAgent(
        provider="openai",
        model="legacy-openai-chat-model",
        base_url="https://example.test/v1",
        request_retries=1,
        retry_backoff_s=2.0,
    )
    agent.reset("Check the weather in Paris.")

    assert agent._openai_response_text() == "weather get --location 'Paris'"
    # Equal-jitter backoff: a single retry sleeps once for a value in
    # [base/2, base] rather than exactly `base`.
    assert len(sleeps) == 1
    assert 1.0 <= sleeps[0] <= 2.0
    assert any("api-backoff" in attempt for attempt in agent.debug_reply_attempts())


class _HttpErrorBody:
    def __init__(self, payload: str) -> None:
        self._payload = payload.encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def close(self) -> None:
        return None


class _ChatResponse:
    def __init__(self, content: str = "weather get --location 'Paris'") -> None:
        self._content = content

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return json.dumps(
            {"choices": [{"message": {"content": self._content, "role": "assistant"}}]}
        ).encode("utf-8")


def test_openai_request_retries_transient_http_400(monkeypatch: pytest.MonkeyPatch):
    # A 400 whose body is a wrapped server-side failure ("Internal server
    # error") should be retried, not treated as fatal.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls = {"count": 0}
    sleeps: list[float] = []

    def _urlopen(req, timeout=0):
        calls["count"] += 1
        if calls["count"] == 1:
            raise train_and_eval.urllib.error.HTTPError(
                req.full_url,
                400,
                "Bad Request",
                hdrs=None,
                fp=_HttpErrorBody(
                    '{"error":{"code":"validation_error","message":"Internal'
                    ' server error","param":null,"type":"invalid_request_error"}}'
                ),
            )
        return _ChatResponse()

    monkeypatch.setattr(train_and_eval.urllib.request, "urlopen", _urlopen)
    monkeypatch.setattr(train_and_eval.time, "sleep", lambda seconds: sleeps.append(seconds))

    agent = train_and_eval.LLMAgent(
        provider="openai",
        model="legacy-openai-chat-model",
        base_url="https://example.test/v1",
        request_retries=1,
        retry_backoff_s=2.0,
    )
    agent.reset("Check the weather in Paris.")

    assert agent._openai_response_text() == "weather get --location 'Paris'"
    assert calls["count"] == 2
    assert len(sleeps) == 1


def test_openai_request_does_not_retry_fatal_http_400(monkeypatch: pytest.MonkeyPatch):
    # A genuine validation 400 (no transient marker in the body) stays fatal and
    # is not retried.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls = {"count": 0}

    def _urlopen(req, timeout=0):
        calls["count"] += 1
        raise train_and_eval.urllib.error.HTTPError(
            req.full_url,
            400,
            "Bad Request",
            hdrs=None,
            fp=_HttpErrorBody(
                '{"error":{"code":"validation_error","message":"invalid'
                ' argument","param":"foo","type":"invalid_request_error"}}'
            ),
        )

    monkeypatch.setattr(train_and_eval.urllib.request, "urlopen", _urlopen)

    agent = train_and_eval.LLMAgent(
        provider="openai",
        model="legacy-openai-chat-model",
        base_url="https://example.test/v1",
        request_retries=1,
    )
    agent.reset("Check the weather in Paris.")

    with pytest.raises(RuntimeError, match="HTTP 400"):
        agent._openai_response_text()
    assert calls["count"] == 1


def test_is_content_filter_llm_error_matches_invalid_prompt() -> None:
    flagged = RuntimeError(
        "OpenAI-compatible request failed: HTTP 400: "
        '{"error":{"code":"invalid_prompt","message":"Your prompt was flagged'
        ' as potentially violating our usage policy.",'
        '"type":"invalid_request_error"}}'
    )
    assert train_and_eval._is_content_filter_llm_error(flagged)
    # An unrelated 400 is not misclassified as a content-filter error.
    assert not train_and_eval._is_content_filter_llm_error(
        RuntimeError("HTTP 400: invalid argument")
    )


def test_openai_request_falls_back_to_compact_on_invalid_prompt(monkeypatch: pytest.MonkeyPatch):
    # An invalid_prompt 400 routes to the compact-prompt fallback (which then
    # succeeds), rather than failing the task.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls = {"count": 0}

    def _urlopen(req, timeout=0):
        calls["count"] += 1
        if calls["count"] == 1:
            raise train_and_eval.urllib.error.HTTPError(
                req.full_url,
                400,
                "Bad Request",
                hdrs=None,
                fp=_HttpErrorBody(
                    '{"error":{"code":"invalid_prompt","message":"Your prompt'
                    ' was flagged as potentially violating our usage policy.",'
                    '"type":"invalid_request_error"}}'
                ),
            )
        return _ChatResponse(content="DONE")

    monkeypatch.setattr(train_and_eval.urllib.request, "urlopen", _urlopen)

    agent = train_and_eval.LLMAgent(
        provider="openai",
        model="legacy-openai-chat-model",
        base_url="https://example.test/v1",
        request_retries=0,
    )
    agent.reset("Do something.")

    assert agent._openai_response_text() == "DONE"
    assert calls["count"] == 2
    assert any(
        "content filter triggered" in item for item in agent.debug_reply_attempts()
    )


def test_openai_request_disable_compact_fallback_raises(monkeypatch: pytest.MonkeyPatch):
    # With the fallback disabled (--no-compact-fallback), an invalid_prompt 400
    # is fatal: the request raises (so the task is recorded as a
    # provider_failure) rather than silently degrading to a compact text-only
    # retry. Exactly one HTTP attempt is made -- no compact retry.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls = {"count": 0}

    def _urlopen(req, timeout=0):
        calls["count"] += 1
        raise train_and_eval.urllib.error.HTTPError(
            req.full_url,
            400,
            "Bad Request",
            hdrs=None,
            fp=_HttpErrorBody(
                '{"error":{"code":"invalid_prompt","message":"Your prompt was'
                ' flagged as potentially violating our usage policy.",'
                '"type":"invalid_request_error"}}'
            ),
        )

    monkeypatch.setattr(train_and_eval.urllib.request, "urlopen", _urlopen)

    agent = train_and_eval.LLMAgent(
        provider="openai",
        model="legacy-openai-chat-model",
        base_url="https://example.test/v1",
        request_retries=0,
        allow_compact_fallback=False,
    )
    agent.reset("Do something.")

    with pytest.raises(RuntimeError, match="HTTP 400"):
        agent._openai_response_text()
    assert calls["count"] == 1
    assert not any(
        "content filter triggered" in item for item in agent.debug_reply_attempts()
    )


# Realistic body of the OpenAI-compatible gateway's INPUT safety pre-check
# rejection: an HTTP 400 whose message points at the safety-checks cybersecurity
# guide. This fires FLAKILY on benign benchmark prompts that merely sound
# security-adjacent.
_SAFETY_400_BODY = (
    '{"error":{"code":"validation_error","message":"This request was flagged'
    ' for potentially high-risk cyber activity. See'
    ' https://platform.openai.com/docs/guides/safety-checks/cybersecurity",'
    '"type":"invalid_request_error"}}'
)


def test_is_flaky_safety_llm_error_matches_safety_checks() -> None:
    # The provider input-safety pre-check rejection is recognized ...
    flagged = RuntimeError(
        "OpenAI-compatible request failed: HTTP 400: " + _SAFETY_400_BODY
    )
    assert train_and_eval._is_flaky_safety_llm_error(flagged)
    # ... but an ordinary validation 400 ("invalid argument") stays fatal, so it
    # is NOT treated as a flaky safety false-positive.
    assert not train_and_eval._is_flaky_safety_llm_error(
        RuntimeError("HTTP 400: invalid argument")
    )
    # A content-policy (invalid_prompt) 400 is a DIFFERENT class handled by the
    # compact fallback -- not the safety-checks retry path.
    assert not train_and_eval._is_flaky_safety_llm_error(
        RuntimeError(
            'HTTP 400: {"error":{"code":"invalid_prompt","message":"Your prompt'
            ' was flagged as potentially violating our usage policy."}}'
        )
    )
    # A genuinely transient wrapped 5xx is handled by the general retry path.
    assert not train_and_eval._is_flaky_safety_llm_error(
        RuntimeError("HTTP 400: Internal server error")
    )


def test_openai_request_retries_flaky_safety_400_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
):
    # A flaky provider input-safety 400 gets a small bounded verbatim retry; the
    # very next identical attempt succeeds (the classifier is probabilistic).
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls = {"count": 0}
    sleeps: list[float] = []

    def _urlopen(req, timeout=0):
        calls["count"] += 1
        if calls["count"] == 1:
            raise train_and_eval.urllib.error.HTTPError(
                req.full_url,
                400,
                "Bad Request",
                hdrs=None,
                fp=_HttpErrorBody(_SAFETY_400_BODY),
            )
        return _ChatResponse()

    monkeypatch.setattr(train_and_eval.urllib.request, "urlopen", _urlopen)
    monkeypatch.setattr(
        train_and_eval.time, "sleep", lambda seconds: sleeps.append(seconds)
    )

    agent = train_and_eval.LLMAgent(
        provider="openai",
        model="legacy-openai-chat-model",
        base_url="https://example.test/v1",
        request_retries=20,
        safety_request_retries=2,
        retry_backoff_s=2.0,
    )
    agent.reset("Check the weather in Paris.")

    assert agent._openai_response_text() == "weather get --location 'Paris'"
    assert calls["count"] == 2
    assert len(sleeps) == 1
    assert any(
        "bounded verbatim" in item for item in agent.debug_reply_attempts()
    )


def test_openai_request_flaky_safety_400_gives_up_after_safety_cap(
    monkeypatch: pytest.MonkeyPatch,
):
    # A persistently-failing safety 400 is retried only up to the small
    # ``safety_request_retries`` cap (NOT the full ``request_retries`` budget),
    # then raises so the task is recorded as a provider_failure.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls = {"count": 0}
    sleeps: list[float] = []

    def _urlopen(req, timeout=0):
        calls["count"] += 1
        raise train_and_eval.urllib.error.HTTPError(
            req.full_url,
            400,
            "Bad Request",
            hdrs=None,
            fp=_HttpErrorBody(_SAFETY_400_BODY),
        )

    monkeypatch.setattr(train_and_eval.urllib.request, "urlopen", _urlopen)
    monkeypatch.setattr(
        train_and_eval.time, "sleep", lambda seconds: sleeps.append(seconds)
    )

    agent = train_and_eval.LLMAgent(
        provider="openai",
        model="legacy-openai-chat-model",
        base_url="https://example.test/v1",
        request_retries=20,
        safety_request_retries=2,
        retry_backoff_s=2.0,
    )
    agent.reset("Check the weather in Paris.")

    with pytest.raises(RuntimeError, match="HTTP 400"):
        agent._openai_response_text()
    # 1 initial attempt + exactly 2 bounded safety retries == 3 (never 21).
    assert calls["count"] == 3
    assert len(sleeps) == 2
    assert any(
        "giving up (provider_failure)" in item
        for item in agent.debug_reply_attempts()
    )


def test_safety_request_retries_never_exceeds_request_retries() -> None:
    # The safety budget is clamped to the general request-retry budget so a
    # smaller ``--llm-request-retries`` can never be silently exceeded.
    agent = train_and_eval.LLMAgent(
        provider="openai",
        model="legacy-openai-chat-model",
        base_url="https://example.test/v1",
        request_retries=1,
        safety_request_retries=5,
    )
    assert agent._safety_request_retries == 1


def _compact_fallback_episode(
    task_id: str,
    *,
    success: bool = False,
    provider_impacted: bool = False,
    compact_fallback: bool = False,
) -> "train_and_eval.EpisodeResult":
    return train_and_eval.EpisodeResult(
        task_id=task_id,
        agent_name="llm",
        success=success,
        score=1.0 if success else 0.0,
        steps=1,
        duration_s=0.1,
        trajectory=[],
        final_response="",
        error="",
        error_type="",
        provider_impacted=provider_impacted,
        compact_fallback=compact_fallback,
        stop_reason="",
        date_anchor_mismatch=False,
        mode="multi",
        metadata={},
    )


def test_episode_result_roundtrips_compact_fallback() -> None:
    res = _compact_fallback_episode(
        "conflict_vlm_scenario_a_r01",
        success=True,
        provider_impacted=True,
        compact_fallback=True,
    )
    record = res.to_dict()
    assert record["compact_fallback"] is True
    assert train_and_eval._episode_result_from_record(record) == res
    # Legacy records written before the field existed default to False.
    legacy = {k: v for k, v in record.items() if k != "compact_fallback"}
    assert (
        train_and_eval._episode_result_from_record(legacy).compact_fallback is False
    )


def test_update_summary_tracks_compact_fallback_and_clean_accuracy() -> None:
    summary = train_and_eval.EvalSummary(
        agent_name="llm", split="dev", total=0, passed=0, total_score=0.0
    )
    # A degraded (compact-fallback) PASS and a clean FAIL.
    degraded_pass = _compact_fallback_episode(
        "t1", success=True, provider_impacted=True, compact_fallback=True
    )
    clean_fail = _compact_fallback_episode("t2", success=False)
    train_and_eval._update_summary(summary, degraded_pass, "t1", None)
    train_and_eval._update_summary(summary, clean_fail, "t2", None)

    assert summary.compact_fallback_tasks == 1
    assert summary.provider_impacted_tasks == 1
    # Raw accuracy counts the degraded pass (1/2) ...
    assert summary.passed == 1
    assert summary.tgc == 0.5
    # ... but clean accuracy excludes it: the only pass was degraded, so 0/1.
    assert summary.clean_completed_tasks == 1
    assert summary.clean_passed == 0
    assert summary.clean_tgc == 0.0


def test_is_retryable_llm_error_uses_numeric_status_codes() -> None:
    class _E(Exception):
        def __init__(self, msg, code=None):
            super().__init__(msg)
            if code is not None:
                self.code = code

    # A numeric status code alone (no text marker in the message) is enough:
    # gemini/Vertex raise "Quota exceeded"-style messages that need not match
    # any string marker, so string matching alone would mis-classify them.
    assert train_and_eval._is_retryable_llm_error(_E("boom", code=429))
    assert train_and_eval._is_retryable_llm_error(_E("boom", code=500))
    assert train_and_eval._is_retryable_llm_error(_E("boom", code=503))
    assert train_and_eval._is_retryable_llm_error(_E("boom", code=599))
    # Non-retryable client errors must not trigger the retry loop.
    assert not train_and_eval._is_retryable_llm_error(_E("bad", code=400))
    assert not train_and_eval._is_retryable_llm_error(_E("missing", code=404))
    # No code and no marker -> not retryable.
    assert not train_and_eval._is_retryable_llm_error(_E("weird"))
    # bool is an int subclass; a stray True/False code must be ignored.
    assert not train_and_eval._is_retryable_llm_error(_E("weird", code=True))

    # The ``status_code`` attribute name is also honored.
    class _E2(Exception):
        def __init__(self):
            super().__init__("boom")
            self.status_code = 429

    assert train_and_eval._is_retryable_llm_error(_E2())
    # Text markers still work for backends that only surface a string.
    assert train_and_eval._is_retryable_llm_error(Exception("Quota exceeded"))
    assert train_and_eval._is_retryable_llm_error(Exception("RESOURCE_EXHAUSTED"))


def test_retry_backoff_delay_applies_equal_jitter() -> None:
    # Equal jitter: the returned delay always lies in [capped/2, capped].
    for attempt in range(5):
        capped = min(12.0, 2.0 * (2**attempt))
        for _ in range(50):
            delay = train_and_eval._retry_backoff_delay(2.0, attempt)
            assert capped * 0.5 <= delay <= capped
    # Non-positive base delay disables backoff.
    assert train_and_eval._retry_backoff_delay(0.0, 3) == 0.0
    assert train_and_eval._retry_backoff_delay(-1.0, 3) == 0.0
    # The 12s ceiling bounds even large attempt indices.
    assert train_and_eval._retry_backoff_delay(2.0, 20) <= 12.0
    # Jitter actually varies the delay rather than returning a constant.
    values = {train_and_eval._retry_backoff_delay(2.0, 2) for _ in range(50)}
    assert len(values) > 1


def test_retry_backoff_delay_respects_custom_cap() -> None:
    # A larger cap lifts the ceiling: with base == cap == 60 every attempt lands
    # in the flat [30, 60] band used for the deadline/timeout class.
    for attempt in range(5):
        for _ in range(50):
            delay = train_and_eval._retry_backoff_delay(60.0, attempt, 60.0)
            assert 30.0 <= delay <= 60.0
    # The cap bounds even a growing exponential term.
    assert train_and_eval._retry_backoff_delay(15.0, 10, 60.0) <= 60.0
    # The default cap stays 12s so ordinary transient retries are unchanged.
    assert train_and_eval._retry_backoff_delay(2.0, 20) <= 12.0


def test_is_deadline_error_classifies_deadline_and_timeout() -> None:
    class _E(Exception):
        def __init__(self, msg, code=None, status_code=None):
            super().__init__(msg)
            if code is not None:
                self.code = code
            if status_code is not None:
                self.status_code = status_code

    # Server-side 504, however the provider surfaces it.
    assert train_and_eval._is_deadline_error(_E("boom", code=504))
    assert train_and_eval._is_deadline_error(_E("boom", status_code=504))
    assert train_and_eval._is_deadline_error(
        Exception("504 DEADLINE_EXCEEDED. Deadline expired before operation ...")
    )
    assert train_and_eval._is_deadline_error(Exception("504 Gateway Timeout"))
    assert train_and_eval._is_deadline_error(
        RuntimeError("OpenAI-compatible request failed: HTTP 504: upstream")
    )
    # Client-side timeout (our --llm-timeout-s firing before the backend's 504).
    assert train_and_eval._is_deadline_error(
        Exception("The read operation timed out")
    )
    assert train_and_eval._is_deadline_error(TimeoutError("request timeout"))
    # Throttling/unavailability is NOT the deadline class -> keeps short backoff.
    assert not train_and_eval._is_deadline_error(_E("slow down", code=429))
    assert not train_and_eval._is_deadline_error(_E("unavailable", code=503))
    assert not train_and_eval._is_deadline_error(_E("boom", code=500))
    assert not train_and_eval._is_deadline_error(_E("bad", code=400))
    # bool is an int subclass; a stray True/False code must be ignored.
    assert not train_and_eval._is_deadline_error(_E("weird", code=True))


def test_gemini_deadline_error_uses_long_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A 504 must route to the long deadline backoff (flat 30-60s), not the short
    # transient backoff, so an overloaded backend can drain before we re-send.
    _install_fake_genai(monkeypatch)
    sleeps: list[float] = []
    monkeypatch.setattr(train_and_eval.time, "sleep", lambda s: sleeps.append(s))

    class _Resp:
        text = "file list "

    class _DeadlineError(Exception):
        def __init__(self):
            super().__init__(
                "504 DEADLINE_EXCEEDED. Deadline expired before operation could"
                " complete."
            )
            self.code = 504

    calls = {"n": 0}

    class _Models:
        def generate_content(self, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _DeadlineError()
            return _Resp()

    class _Client:
        models = _Models()

    agent = train_and_eval.LLMAgent(
        provider="gemini",
        model="gemini-3.5-flash",
        vertex_project="p",
        vertex_location="us-central1",
        request_retries=3,
        retry_backoff_s=2.0,
        deadline_retry_backoff_s=60.0,
        deadline_max_backoff_s=60.0,
    )
    agent._client = _Client()
    agent._messages = [{"role": "user", "content": "hi"}]

    out = agent._gemini_response_text()
    assert out == "file list"
    assert calls["n"] == 2  # exactly one retry after the 504
    # Flat 30-60s band from base == cap == 60 with equal jitter -- NOT the 1-2s
    # the transient backoff (retry_backoff_s=2.0) would have produced.
    assert len(sleeps) == 1
    assert 30.0 <= sleeps[0] <= 60.0


def test_cli_accepts_deadline_backoff_flags(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}

    _DummySummary = _make_dummy_summary(calls)

    def _build_agent(*args, **kwargs):
        calls["build_agent_kwargs"] = kwargs
        return object()

    monkeypatch.setattr(train_and_eval, "_preflight_mode", lambda mode: None)
    monkeypatch.setattr(train_and_eval, "_build_agent", _build_agent)
    monkeypatch.setattr(
        train_and_eval, "run_evaluation", lambda *_, **__: _DummySummary()
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent",
            "llm",
            "--model",
            "gemini-3.5-flash",
            "--llm-provider",
            "gemini",
            "--llm-deadline-retry-backoff-s",
            "45",
            "--llm-deadline-max-backoff-s",
            "90",
            "--limit",
            "1",
        ],
    )

    train_and_eval.main()

    kwargs = calls["build_agent_kwargs"]
    assert kwargs["llm_deadline_retry_backoff_s"] == 45.0
    assert kwargs["llm_deadline_max_backoff_s"] == 90.0


def _install_fake_genai(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Inject a minimal fake ``google.genai`` so gemini paths need no SDK.

    Returns a dict capturing the kwargs passed to ``genai.Client(...)``.
    """
    import types as _types

    captured: dict[str, object] = {}

    class _FakeHttpOptions:
        def __init__(self, timeout=None):
            self.timeout = timeout

    class _FakePart:
        @staticmethod
        def from_text(text=None):
            return {"text": text}

        @staticmethod
        def from_bytes(data=None, mime_type=None):
            return {"bytes": data, "mime_type": mime_type}

    class _FakeContent:
        def __init__(self, role=None, parts=None):
            self.role = role
            self.parts = parts

    class _FakeGenConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_types = _types.ModuleType("google.genai.types")
    fake_types.HttpOptions = _FakeHttpOptions
    fake_types.Part = _FakePart
    fake_types.Content = _FakeContent
    fake_types.GenerateContentConfig = _FakeGenConfig

    fake_genai = _types.ModuleType("google.genai")
    fake_genai.Client = _FakeClient
    fake_genai.types = fake_types

    fake_google = _types.ModuleType("google")
    fake_google.genai = fake_genai

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)
    return captured


def test_gemini_get_client_passes_http_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_fake_genai(monkeypatch)
    # Vertex is the fallback, not the default, so the key must be absent or
    # this test would pass or fail depending on the developer's shell.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    agent = train_and_eval.LLMAgent(
        provider="gemini",
        model="gemini-3.5-flash",
        vertex_project="proj-x",
        vertex_location="us-central1",
        request_timeout_s=180,
    )
    agent._get_gemini_client()

    # --llm-timeout-s is seconds; HttpOptions.timeout is milliseconds. Without
    # this the gemini SDK can hang far past the intended cap (latency fat-tail).
    http_options = captured["http_options"]
    assert http_options.timeout == 180_000
    assert captured["project"] == "proj-x"
    assert captured["location"] == "us-central1"
    assert captured["vertexai"] is True


def test_gemini_get_client_prefers_the_developer_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An API key selects the Gemini Developer API over Vertex AI.

    The published runs used Vertex, but reproducing them only needs a key, so
    the key wins even when a project is also configured.
    """
    captured = _install_fake_genai(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "key-abc")

    agent = train_and_eval.LLMAgent(
        provider="gemini",
        model="gemini-3.5-flash",
        vertex_project="proj-x",
        vertex_location="us-central1",
        request_timeout_s=180,
    )
    agent._get_gemini_client()

    assert captured["api_key"] == "key-abc"
    assert "vertexai" not in captured
    assert "project" not in captured
    # The timeout must be threaded through on this path too.
    assert captured["http_options"].timeout == 180_000


def test_gemini_get_client_errors_when_neither_key_nor_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Naming both ways out beats the SDK's own opaque credentials error."""
    _install_fake_genai(monkeypatch)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    agent = train_and_eval.LLMAgent(provider="gemini", model="gemini-3.5-flash")
    with pytest.raises(RuntimeError) as excinfo:
        agent._get_gemini_client()

    message = str(excinfo.value)
    assert "GEMINI_API_KEY" in message
    assert "--vertex-project" in message


def test_gemini_non_retryable_error_raises_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_genai(monkeypatch)
    sleeps: list[float] = []
    monkeypatch.setattr(train_and_eval.time, "sleep", lambda s: sleeps.append(s))

    calls = {"n": 0}

    class _Models:
        def generate_content(self, **kwargs):
            calls["n"] += 1
            # 400 has no retry marker and its code is not retryable.
            raise ValueError("Error code: 400 - invalid argument")

    class _Client:
        models = _Models()

    agent = train_and_eval.LLMAgent(
        provider="gemini",
        model="gemini-3.5-flash",
        vertex_project="p",
        vertex_location="us-central1",
        request_retries=5,
        retry_backoff_s=2.0,
    )
    agent._client = _Client()  # bypass real client construction
    agent._messages = [{"role": "user", "content": "hi"}]

    with pytest.raises(ValueError):
        agent._gemini_response_text()
    # The gate stops the loop after the first attempt (no wasted retries/sleeps).
    assert calls["n"] == 1
    assert sleeps == []


def test_gemini_retryable_error_retries_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_genai(monkeypatch)
    sleeps: list[float] = []
    monkeypatch.setattr(train_and_eval.time, "sleep", lambda s: sleeps.append(s))

    class _Resp:
        text = "file list "

    class _TransientError(Exception):
        def __init__(self):
            # No text marker on purpose: retryability comes solely from .code,
            # exercising the numeric-status-code path added to the classifier.
            super().__init__("transient upstream hiccup")
            self.code = 503

    calls = {"n": 0}

    class _Models:
        def generate_content(self, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _TransientError()
            return _Resp()

    class _Client:
        models = _Models()

    agent = train_and_eval.LLMAgent(
        provider="gemini",
        model="gemini-3.5-flash",
        vertex_project="p",
        vertex_location="us-central1",
        request_retries=3,
        retry_backoff_s=2.0,
    )
    agent._client = _Client()
    agent._messages = [{"role": "user", "content": "hi"}]

    out = agent._gemini_response_text()
    assert out == "file list"  # stripped, returned from the second attempt
    assert calls["n"] == 2  # exactly one retry after the retryable 503
    # Equal-jitter backoff: exactly one sleep, within [base/2, base].
    assert len(sleeps) == 1
    assert 1.0 <= sleeps[0] <= 2.0


def test_gemini_native_replays_thought_signature_on_history() -> None:
    # Regression guard for the Gemini 2.5+ requirement that the opaque
    # thought_signature accompanying a functionCall part be replayed on the
    # reconstructed history part; without it multi-step native turns fail with
    # 400 INVALID_ARGUMENT ("Function call is missing a thought_signature").
    # Uses the real google-genai types so Part actually carries the attribute.
    pytest.importorskip("google.genai")
    from train_eval import tool_calling as tc

    agent = train_and_eval.LLMAgent(
        provider="gemini",
        model="gemini-2.5-flash",
        vertex_project="p",
        vertex_location="us-central1",
    )
    agent._native_tools_gemini = [
        {"functionDeclarations": [{"name": "asset_list"}, {"name": "done"}]}
    ]

    sig = b"\x00\x01SIGbytes\xff"
    prior = tc.ToolCall(
        name="asset_list",
        arguments={},
        id=None,
        thought_signature=train_and_eval.base64.b64encode(sig).decode("ascii"),
    )
    agent._messages = [
        {"role": "user", "content": "go"},
        tc.assistant_tool_turn(prior),
        tc.tool_result_turn(prior, "ok"),
    ]

    captured: dict[str, object] = {}

    class _Models:
        def generate_content(self, **kwargs):
            captured["contents"] = kwargs["contents"]
            # A fresh (unsigned) call ends the turn.
            return SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            parts=[
                                SimpleNamespace(
                                    function_call=SimpleNamespace(name="done", args={}),
                                    thought_signature=None,
                                )
                            ]
                        )
                    )
                ]
            )

    class _Client:
        models = _Models()

    agent._client = _Client()

    call = agent._gemini_tool_call()
    assert call is not None and call.name == "done"
    assert call.thought_signature is None  # response part had no signature

    contents = captured["contents"]
    # user turn -> replayed model function_call -> tool result.
    model_content = contents[1]
    assert model_content.role == "model"
    replayed_part = model_content.parts[0]
    assert replayed_part.function_call.name == "asset_list"
    # The base64 signature must be decoded back to the exact original bytes.
    assert replayed_part.thought_signature == sig


def test_gemini_sends_original_size_images_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression guard: an earlier "fix" pre-shrank gemini images to 2576px, but
    # LANCZOS re-encoding these flat UI PNGs INFLATED the payload (+45%) - the
    # interpolation invents colors that defeat PNG/DEFLATE - and pushed accumulated
    # full-history requests over the Vertex deadline (504 DEADLINE_EXCEEDED).
    # Original-size renders compress ~57:1 and ran with zero provider failures, so
    # the gemini path must forward image bytes untouched.
    pytest.importorskip("PIL")
    _install_fake_genai(monkeypatch)

    big = _png_b64(3840, 2160)
    original_bytes = train_and_eval.base64.b64decode(big)

    captured: dict[str, object] = {}

    class _Resp:
        text = "asset list "

    class _Models:
        def generate_content(self, **kwargs):
            captured["contents"] = kwargs["contents"]
            return _Resp()

    class _Client:
        models = _Models()

    agent = train_and_eval.LLMAgent(
        provider="gemini",
        model="gemini-3.5-flash",
        vertex_project="p",
        vertex_location="us-central1",
    )
    agent._client = _Client()  # bypass real client construction
    agent._messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "see assets"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{big}"},
                },
            ],
        }
    ]

    out = agent._gemini_response_text()
    assert out == "asset list"

    # The bytes handed to the SDK must be byte-identical to the source image: no
    # downscale, no re-encode.
    contents = captured["contents"]
    image_parts = [
        p
        for c in contents
        for p in c.parts
        if isinstance(p, dict) and "bytes" in p
    ]
    assert len(image_parts) == 1
    assert image_parts[0]["bytes"] == original_bytes
    w, h = _png_dims_from_b64(
        train_and_eval.base64.b64encode(image_parts[0]["bytes"]).decode("ascii")
    )
    assert (w, h) == (3840, 2160)  # original dimensions preserved


def test_classify_error_type_maps_vertex_deadline_to_provider_failure() -> None:
    classify = train_and_eval._classify_error_type
    # Success -> no error type; empty error on failure -> logical (model) failure.
    assert classify("", success=True) == ""
    assert classify("", success=False) == "logical_failure"
    # The real Vertex/gemini 504 string has no "http 504" prefix; it must still be
    # attributed to the provider (not the runner) so provider-adjusted accuracy is
    # computed correctly.
    vertex_504 = (
        "504 DEADLINE_EXCEEDED. {'error': {'code': 504, 'message': 'Deadline "
        "expired before operation could complete.', 'status': 'DEADLINE_EXCEEDED'}}"
    )
    assert classify(vertex_504, success=False) == "provider_failure"
    assert (
        classify("The read operation timed out", success=False)
        == "provider_failure"
    )
    assert classify("429 RESOURCE_EXHAUSTED", success=False) == "provider_failure"
    assert classify("503 UNAVAILABLE", success=False) == "provider_failure"
    # A non-provider runtime error is still a runner failure.
    assert (
        classify("unexpected KeyError in runner", success=False) == "runner_failure"
    )


def test_classify_error_type_maps_vertex_request_size_to_provider_failure() -> None:
    """Bug 2: the 30MB total-payload rejection is a provider failure, not the model's.

    Vertex rejects an Anthropic request whose serialized payload exceeds 30MB with
    a 400 "The request size (N bytes) exceeds 30.000MB limit. FAILED_PRECONDITION".
    Conflict-VLM rounds accumulate every round-visible image (r10 ~= 25 images,
    several 4K), so the full-history payload trips this through no fault of the
    solver. It must classify as ``provider_failure`` so it is excluded from
    ``completed_tasks`` and does not depress provider-adjusted accuracy.
    """
    classify = train_and_eval._classify_error_type
    vertex_30mb = (
        "Error code: 400 - {'error': {'code': 400, 'message': 'The request size "
        "(40912345 bytes) exceeds 30.000MB limit.', 'status': 'FAILED_PRECONDITION'}}"
    )
    assert classify(vertex_30mb, success=False) == "provider_failure"
    # Either marker alone suffices (message phrasing varies across SDK versions).
    assert (
        classify("the request size (31000000 bytes) exceeds", success=False)
        == "provider_failure"
    )
    assert classify("400 FAILED_PRECONDITION", success=False) == "provider_failure"
    # A genuine model/runtime mistake is still a runner failure (no false positives).
    assert (
        classify("assertion failed: wrong final answer", success=False)
        == "runner_failure"
    )


def test_generation_stall_is_a_provider_failure_not_a_logical_failure() -> None:
    """An empty-reply stall is the provider declining to answer, not a bad answer.

    A terminal ``generation_stall_empty_reply`` carries NO error string: the HTTP
    call succeeded and the body was empty, repeatedly, until the invalid-reply
    retry budget was exhausted. Classified on ``error`` alone it falls through to
    ``logical_failure``, charging a provider non-response to the model's
    reasoning.
    """
    classify = train_and_eval._classify_error_type
    stall = "generation_stall_empty_reply"

    assert classify("", success=False, stop_reason=stall) == "provider_failure"
    # Positional form (how the runner calls it) behaves identically.
    assert classify("", False, stall) == "provider_failure"
    # Success always wins: a stall that still passed is not an error at all.
    assert classify("", success=True, stop_reason=stall) == ""
    # No regression for the pre-existing 2-arg contract.
    assert classify("", success=False) == "logical_failure"
    # Other stop_reasons are untouched -- only the empty-reply stall reclassifies.
    assert (
        classify("", success=False, stop_reason="invalid_command_exhausted")
        == "logical_failure"
    )
    assert (
        classify("", success=False, stop_reason="reached max_steps=20")
        == "logical_failure"
    )


def test_generation_stall_provider_attribution_is_vendor_symmetric() -> None:
    """The SAME refusal must be attributed identically across vendors.

    Regression test for a measured asymmetry on
    ``conflict_vlm_payment_chargeback_dispute_v03_r08``: the OpenAI-compatible
    gateway refuses with an explicit HTTP 400 ("flagged for potentially high-risk
    cyber activity") -> provider_failure -> eligible for
    ``--retry-provider-failures-only``, while Anthropic returned an empty body ->
    logical_failure -> never retried. Both shapes are the provider failing to
    answer and must land in the same bucket.
    """
    classify = train_and_eval._classify_error_type
    is_provider_failed = train_and_eval._trajectory_record_is_provider_failed
    stall = "generation_stall_empty_reply"

    openai_refusal = (
        'OpenAI-compatible request failed: HTTP 400: {"error":{"code":'
        '"validation_error","message":"This request has been flagged for '
        'potentially high-risk cyber activity"}}'
    )
    # OpenAI shape: explicit error string, no stall.
    assert classify(openai_refusal, success=False) == "provider_failure"
    # Anthropic/Gemini shape: silent empty body, surfaced only via stop_reason.
    assert classify("", success=False, stop_reason=stall) == "provider_failure"

    # ...and both are retry-eligible at the trajectory-record level.
    assert is_provider_failed(
        {"success": False, "error": openai_refusal, "error_type": "provider_failure"}
    )
    assert is_provider_failed(
        {"success": False, "error": "", "error_type": "provider_failure",
         "stop_reason": stall}
    )


def test_stalled_records_written_before_reclassification_are_retry_eligible() -> None:
    """Records already on disk carry the OLD ``error_type`` and must still requeue.

    ``--retry-provider-failures-only`` reads previously written trajectory jsonl.
    Those rows were classified before the fix and carry
    ``error_type="logical_failure"``, so eligibility cannot rely on ``error_type``
    alone -- it is matched on ``stop_reason`` too. Without this, re-running the
    affected tasks would require discarding and regenerating the whole run.
    """
    is_provider_failed = train_and_eval._trajectory_record_is_provider_failed
    on_disk = {
        "task_id": "conflict_vlm_payment_chargeback_dispute_v03_r08",
        "success": False,
        "error": "",
        "error_type": "logical_failure",  # stale label written by the old code
        "provider_impacted": False,
        "stop_reason": "generation_stall_empty_reply",
        "final_response": "",
    }
    assert is_provider_failed(on_disk)

    # A genuine reasoning failure with no stall is NOT requeued (no over-firing).
    ordinary_miss = dict(on_disk, stop_reason="incomplete_required_submission")
    assert not is_provider_failed(ordinary_miss)
    # A stalled-but-successful record is never a provider failure.
    assert not is_provider_failed(dict(on_disk, success=True))


def test_has_provider_noise_sees_the_stall_breadcrumb_for_every_vendor() -> None:
    """``provider_impacted`` must not depend on which vendor produced the stall.

    Only the OpenAI-compatible path RAISES on an empty body, leaving an
    ``api-error: ... empty message content`` line in the debug trail; Anthropic
    and Gemini return "" silently. The vendor-neutral ``stall: ...`` breadcrumb is
    emitted by provider-independent code, so keying off it restores symmetry.
    """
    has_noise = train_and_eval._has_provider_noise

    anthropic_trail = [
        "api: content=''",
        "parsed: raw='' normalized=''",
        "stall: generation_stall_empty_reply",
    ]
    gemini_trail = [
        "api: content='' finish_reason=STOP",
        "stall: generation_stall_empty_reply",
    ]
    openai_trail = [
        "api-error: OpenAI-compatible response contained empty message content: {}",
        "stall: generation_stall_empty_reply",
    ]
    assert has_noise(anthropic_trail)
    assert has_noise(gemini_trail)
    assert has_noise(openai_trail)

    # A recovered empty reply (no stall line) is not provider noise on the silent
    # vendors -- it cost a retry but the episode continued normally.
    assert not has_noise(["api: content=''", "parsed: raw='DONE' normalized='DONE'"])
    # An ordinary invalid-command exhaustion is the model's fault, not the
    # provider's.
    assert not has_noise(
        ["api: content='blah'", "stall: invalid_command_exhausted"]
    )


def test_openai_response_text_extracts_command_from_reasoning_content(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "reasoning_content": (
                                    "I should check the inbox first.\n"
                                    "Next command: email search --query budget"
                                ),
                                "role": "assistant",
                            }
                        }
                    ]
                }
            ).encode("utf-8")

    monkeypatch.setattr(
        train_and_eval.urllib.request,
        "urlopen",
        lambda req, timeout=0: _Response(),
    )

    agent = train_and_eval.LLMAgent(
        provider="openai",
        model="legacy-openai-chat-model",
        base_url="https://example.test/v1",
    )
    task = train_and_eval.load_task(
        "hard_decision_workflow_1",
        data_dir=Path(train_and_eval.DEFAULT_TASK_DATA_DIR),
    )
    agent.reset("Review budget-related email.", task=task)

    assert agent._openai_response_text() == "email search --query budget"


def test_openai_response_text_reprompts_when_reasoning_has_no_command(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    captured_bodies: list[dict[str, object]] = []

    class _Response:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(self._payload).encode("utf-8")

    responses = iter(
        [
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "reasoning_content": "I should inspect the calendar context before deciding.",
                            "role": "assistant",
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "content": "calendar today --timezone America/Chicago",
                            "role": "assistant",
                        }
                    }
                ]
            },
        ]
    )

    def _urlopen(req, timeout=0):
        captured_bodies.append(json.loads(req.data.decode("utf-8")))
        return _Response(next(responses))

    monkeypatch.setattr(train_and_eval.urllib.request, "urlopen", _urlopen)

    agent = train_and_eval.LLMAgent(
        provider="openai",
        model="Kimi-K2.5",
        base_url="https://example.test/v1",
    )
    task = train_and_eval.load_task(
        "hard_decision_workflow_1",
        data_dir=Path(train_and_eval.DEFAULT_TASK_DATA_DIR),
    )
    agent.reset("Set up a daily ops check for Austin.", task=task)

    assert agent._openai_response_text() == "calendar today --timezone America/Chicago"
    assert len(captured_bodies) == 2
    assert captured_bodies[0]["max_tokens"] == 256
    assert captured_bodies[1]["max_tokens"] == 64
    assert "previous reply included reasoning but no final command" in captured_bodies[1]["messages"][-1]["content"]


def test_openai_request_uses_max_tokens_for_non_gpt5_models(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    captured: dict[str, object] = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {"choices": [{"message": {"content": "weather get --location 'Paris'"}}]}
            ).encode("utf-8")

    def _urlopen(req, timeout=0):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _Response()

    monkeypatch.setattr(train_and_eval.urllib.request, "urlopen", _urlopen)

    agent = train_and_eval.LLMAgent(
        provider="openai",
        model="Kimi-K2.5",
        base_url="https://example.test/v1",
    )
    agent.reset("Check the weather in Paris.")
    observation = SimpleNamespace(
        step_number=0,
        command_output="",
        error_output="",
        exit_code=0,
    )

    assert agent.act(observation) == "weather get --location 'Paris'"
    assert captured["body"]["max_tokens"] == 256
    assert "max_completion_tokens" not in captured["body"]


def test_openai_response_text_returns_empty_without_text_parts(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": [{"type": "tool_use", "id": "abc"}],
                                "role": "assistant",
                            }
                        }
                    ]
                }
            ).encode("utf-8")

    monkeypatch.setattr(
        train_and_eval.urllib.request,
        "urlopen",
        lambda req, timeout=0: _Response(),
    )

    agent = train_and_eval.LLMAgent(
        provider="openai",
        model="legacy-openai-chat-model",
        base_url="https://example.test/v1",
    )
    agent.reset("Check the weather in Paris.")

    # Message content that carries no text parts (only a tool_use block) is treated
    # as an empty reply: retried under the empty-reply budget, then degraded to "".
    assert agent._openai_response_text() == ""
    assert any("api-empty:" in line for line in agent.debug_reply_attempts())


def test_command_validator_rejects_unknown_cli_shape():
    task = SimpleNamespace(domains=["email", "tasks", "calendar"])

    ok, reason = train_and_eval._validate_command("email read --path email_seed_3", task)

    assert ok is False
    assert "invalid command" in reason
    assert "email read" in reason


def test_command_validator_rejects_unclosed_quotes():
    task = SimpleNamespace(domains=["tasks"])

    ok, reason = train_and_eval._validate_command('tasks add --title "', task)

    assert ok is False
    assert "shell parse error" in reason


def test_command_validator_rejects_template_placeholder_values():
    data_dir = Path(train_and_eval.DEFAULT_TASK_DATA_DIR)
    task = train_and_eval.load_task("hard_decision_workflow_1", data_dir=data_dir)

    ok, reason = train_and_eval._validate_command(
        "calendar today --timezone TIMEZONE",
        task,
    )

    assert ok is False
    assert "template placeholder" in reason


def test_command_validator_allows_brackets_inside_quoted_free_text():
    task = SimpleNamespace(domains=["email"])

    ok, reason = train_and_eval._validate_command(
        'email send --to finance@example.com --subject "Toronto Daily Ops Setup Confirmed" '
        '--body "Confirmed: cron job [cron_existing_ops] is already scheduled."',
        task,
    )

    assert ok is True
    assert reason == ""


def test_command_validator_rejects_incomplete_openclaw_side_effect_commands():
    task = SimpleNamespace(domains=["cron_webhook", "messaging", "channel_mgmt", "setup_config"])

    ok, reason = train_and_eval._validate_command("openclaw cron add", task)
    assert ok is False
    assert "--name" in reason or "missing required" in reason.lower()

    ok, reason = train_and_eval._validate_command("openclaw message send --target @alice --message hi", task)
    assert ok is False
    assert "--channel" in reason

    ok, reason = train_and_eval._validate_command("openclaw channels login", task)
    assert ok is False
    assert "--channel" in reason


def test_llm_agent_retries_after_invalid_command(monkeypatch: pytest.MonkeyPatch):
    responses = iter(["email read --path email_seed_3", "email search --query budget"])
    agent = train_and_eval.LLMAgent(provider="openai", model="gpt-5.2")
    agent.reset(
        "Review budget-related email.",
        task=SimpleNamespace(domains=["email", "tasks", "calendar"]),
    )
    monkeypatch.setattr(agent, "_openai_response_text", lambda: next(responses))

    action = agent.act(
        SimpleNamespace(
            step_number=0,
            command_output="",
            error_output="",
            exit_code=0,
        )
    )

    assert action == "email search --query budget"
    assert any(
        msg["role"] == "user" and "invalid command" in msg["content"]
        for msg in agent._messages
    )
    attempts = agent.debug_reply_attempts()
    assert any("parsed: raw='email read --path email_seed_3'" in attempt for attempt in attempts)
    assert any("validator: invalid command" in attempt for attempt in attempts)
    assert any("parsed: raw='email search --query budget'" in attempt for attempt in attempts)


def test_llm_agent_exhausted_invalid_replies_stops_instead_of_tasks_list(
    monkeypatch: pytest.MonkeyPatch,
):
    agent = train_and_eval.LLMAgent(provider="openai", model="gpt-5.2")
    agent.reset(
        "Check the forecast in New York and reschedule the event if needed.",
        task=train_and_eval.load_task(
            "calendar_weather_check_19",
            data_dir=Path(train_and_eval.DEFAULT_TASK_DATA_DIR),
        ),
    )
    monkeypatch.setattr(agent, "_openai_response_text", lambda: "tasks list")

    action = agent.act(
        SimpleNamespace(
            step_number=0,
            command_output="",
            error_output="",
            exit_code=0,
        )
    )

    assert action == "DONE"
    assert agent.should_stop() is True


def test_llm_agent_warns_on_repeated_no_progress(monkeypatch: pytest.MonkeyPatch):
    agent = train_and_eval.LLMAgent(provider="openai", model="gpt-5.2")
    agent.reset(
        "Review budget-related email.",
        task=SimpleNamespace(domains=["email", "tasks", "calendar"]),
    )
    agent._pending_action = "tasks list"
    agent._history = [
        {"action": "tasks list", "stdout": "Tasks:\n...", "stderr": "", "exit_code": 0},
        {"action": "tasks list", "stdout": "Tasks:\n...", "stderr": "", "exit_code": 0},
    ]
    monkeypatch.setattr(agent, "_openai_response_text", lambda: "email search --query budget")

    action = agent.act(
        SimpleNamespace(
            step_number=3,
            command_output="Tasks:\n...",
            error_output="",
            exit_code=0,
        )
    )

    assert action == "email search --query budget"
    assert any(
        msg["role"] == "user" and "Pick a different command from the catalog." in msg["content"]
        for msg in agent._messages
    )


def test_normalize_model_command_strips_code_fence_and_prompt_marker():
    raw = "```bash\n$ tasks add --title 'Write report' --due 2026-03-05\n```"
    assert train_and_eval._normalize_model_command(raw) == "tasks add --title 'Write report' --due 2026-03-05"


def test_normalize_model_command_strips_trailing_explanation():
    raw = "calendar today --timezone TIMEZONE - for reviewing calendar context"
    assert train_and_eval._normalize_model_command(raw) == "calendar today --timezone TIMEZONE"


def test_normalize_model_command_strips_catalog_optional_suffix():
    raw = "openclaw status [--json]"
    assert train_and_eval._normalize_model_command(raw) == "openclaw status"


def test_normalize_model_command_preserves_quoted_dash_content():
    raw = (
        'email send --to alice@example.com --subject "Budget Report Follow-up - Berlin" '
        '--body "Follow-up on the budget report for Berlin."'
    )
    assert train_and_eval._normalize_model_command(raw) == raw


def test_normalize_model_command_strips_provider_json_suffix_noise():
    raw = "tasks list】【。json"
    assert train_and_eval._normalize_model_command(raw) == "tasks list"


def test_normalize_model_command_strips_provider_analysis_suffix_noise():
    raw = "tasks list】【。analysis to=final code  omitted"
    assert train_and_eval._normalize_model_command(raw) == "tasks list"


def test_normalize_model_command_preserves_valid_json_path_argument():
    raw = 'file create --path notes.json --content "hello"'
    assert train_and_eval._normalize_model_command(raw) == raw


def test_normalize_model_command_preserves_multiline_content_body():
    # Regression: the normalizer used to keep only the first physical line, which
    # orphaned the opening quote of a multi-line ``--content`` report body and
    # made shlex.split raise "No closing quotation" (the dominant Gemini failure
    # that scored those episodes ~0). It must now preserve the whole command.
    raw = (
        'file create --path /reports/decision.md --content "# Access Decision\n'
        '\n## Summary\nApprove the request.\n\n## Rationale\n- Scope is minimal\n"'
    )
    normalized = train_and_eval._normalize_model_command(raw)
    assert "\n" in normalized
    assert normalized.rstrip().endswith('"')
    # The full body survived (not truncated to the first line).
    assert "## Rationale" in normalized
    # And it is now shell-parseable.
    import shlex

    shlex.split(normalized)


def test_normalize_model_command_multiline_does_not_strip_trailing_bracket():
    # The single-line hygiene pass strips a trailing ``[...]``; it must be skipped
    # for multi-line report bodies so a trailing citation is not truncated.
    raw = (
        'file create --path /r.md --content "# Title\n\nSee prior review '
        '[cl/12345]"'
    )
    normalized = train_and_eval._normalize_model_command(raw)
    assert normalized.rstrip().endswith('[cl/12345]"')


def test_command_validator_accepts_multiline_content():
    task = SimpleNamespace(domains=["file"])
    cmd = (
        'file create --path /reports/decision.md --content "# Access Decision\n'
        '\n## Summary\nApprove.\n"'
    )
    ok, reason = train_and_eval._validate_command(cmd, task)
    assert ok is True, reason
    assert reason == ""


def test_command_validator_accepts_content_with_unescaped_inner_quotes():
    # Inner unescaped quotes/apostrophes in a ``--content`` body would make
    # shlex.split raise; the validator peels the content before tokenizing so the
    # structurally-valid command reaches the (robust) executor instead of being
    # rejected and retried into a stall.
    task = SimpleNamespace(domains=["file"])
    cmd = (
        'file create --path /r.md --content "The reviewer said "no" and it\'s '
        'blocked until scope is cut"'
    )
    ok, reason = train_and_eval._validate_command(cmd, task)
    assert ok is True, reason


def test_command_catalog_is_scoped_by_relevant_domains_with_gt_backfill():
    data_dir = Path(train_and_eval.DEFAULT_TASK_DATA_DIR)
    calendar_task = train_and_eval.load_task("calendar_weather_check_19", data_dir=data_dir)
    hard_task = train_and_eval.load_task("hard_decision_workflow_1", data_dir=data_dir)
    import clasharness.tasks.generators  # noqa: F401  (auto-registers all generators)
    from clasharness.tasks.generation_options import set_generation_options
    from clasharness.tasks.registry import generate_all_tasks

    set_generation_options(
        message_dry_run=False,
        plugin_install_mode="mixed",
        command_profile="local_skill",
        complex_task_pack="standard",
        complex_scenario_profile="life_work",
        complex_min_steps=3,
        complex_max_steps=5,
        hard_decision_variants_per_scenario=16,
        include_branch_sensitive=True,
        branch_sensitive_variants_per_scenario=12,
    )
    branch_task = next(
        task
        for task in generate_all_tasks(generator_ids=["branch_sensitive_workflow"])
        if task.task_id == "branch_sensitive_workflow_27"
    )

    calendar_catalog = train_and_eval._command_catalog_text(calendar_task)
    hard_catalog = train_and_eval._command_catalog_text(hard_task)
    branch_catalog = train_and_eval._command_catalog_text(branch_task)

    assert "weather get --location LOCATION" in calendar_catalog
    assert "calendar add-event --title TITLE --start DATETIME" in calendar_catalog
    assert "tasks list" not in calendar_catalog
    assert "system cron list" not in calendar_catalog

    assert "email read --id ID" in hard_catalog
    assert "email send --to EMAIL --subject SUBJECT --body BODY" in hard_catalog
    assert "tasks add --title TITLE" in hard_catalog
    assert "calendar today --timezone TIMEZONE" in hard_catalog
    assert "system config get PATH" in hard_catalog
    assert "channel list [--json]" in hard_catalog
    assert "channel login --channel NAME" in hard_catalog
    assert "message send --channel NAME --target TARGET --message TEXT" in hard_catalog

    assert "system config get PATH" in branch_catalog
    assert "system models set MODEL" in branch_catalog


def test_run_evaluation_uses_task_data_dir_for_task_loading(monkeypatch: pytest.MonkeyPatch, tmp_path):
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        train_and_eval,
        "load_task_ids",
        lambda split, data_dir=None, difficulty=None, domain=None: (
            calls.setdefault("ids_data_dir", data_dir),
            ["agent_create_1"],
        )[1],
    )
    monkeypatch.setattr(
        train_and_eval,
        "load_task",
        lambda task_id, data_dir=None: (
            calls.setdefault("task_data_dir", data_dir),
            SimpleNamespace(domains=["agent_mgmt"], difficulty=1),
        )[1],
    )
    monkeypatch.setattr(
        train_and_eval,
        "run_episode",
        lambda *_, **kwargs: train_and_eval.EpisodeResult(
            task_id="agent_create_1",
            agent_name="expert",
            success=True,
            score=1.0,
            steps=1,
            duration_s=0.01,
        ),
    )

    summary = train_and_eval.run_evaluation(
        agent=SimpleNamespace(name="expert"),
        split="dev",
        task_data_dir=tmp_path,
        limit=1,
    )

    assert summary.total == 1
    assert calls.get("ids_data_dir") == tmp_path
    assert calls.get("task_data_dir") == tmp_path


def test_run_evaluation_filters_task_ids_by_prefix(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        train_and_eval,
        "load_task_ids",
        lambda split, data_dir=None, difficulty=None, domain=None: [
            "hard_decision_workflow_1",
            "calendar_add_event_1",
            "hard_decision_workflow_2",
        ],
    )
    monkeypatch.setattr(
        train_and_eval,
        "load_task",
        lambda task_id, data_dir=None: SimpleNamespace(domains=["calendar"], difficulty=3),
    )
    monkeypatch.setattr(
        train_and_eval,
        "run_episode",
        lambda task_id, *_, **__: train_and_eval.EpisodeResult(
            task_id=task_id,
            agent_name="expert",
            success=True,
            score=1.0,
            steps=1,
            duration_s=0.01,
        ),
    )

    summary = train_and_eval.run_evaluation(
        agent=SimpleNamespace(name="expert"),
        split="dev",
        task_prefix="hard_decision_workflow_",
    )

    assert summary.total == 2
    assert [r.task_id for r in summary.results] == [
        "hard_decision_workflow_1",
        "hard_decision_workflow_2",
    ]


def test_run_evaluation_tracks_provider_failures(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        train_and_eval,
        "load_task_ids",
        lambda split, data_dir=None, difficulty=None, domain=None: [
            "hard_decision_workflow_1",
            "hard_decision_workflow_2",
        ],
    )
    monkeypatch.setattr(
        train_and_eval,
        "load_task",
        lambda task_id, data_dir=None: SimpleNamespace(domains=["calendar"], difficulty=3),
    )

    def _run_episode(task_id, *_, **__):
        if task_id.endswith("_1"):
            return train_and_eval.EpisodeResult(
                task_id=task_id,
                agent_name="llm",
                success=False,
                score=0.0,
                steps=0,
                duration_s=0.01,
                error="HTTP 429: queue is full",
                error_type="provider_failure",
            )
        return train_and_eval.EpisodeResult(
            task_id=task_id,
            agent_name="llm",
            success=True,
            score=1.0,
            steps=1,
            duration_s=0.01,
        )

    monkeypatch.setattr(train_and_eval, "run_episode", _run_episode)

    summary = train_and_eval.run_evaluation(
        agent=SimpleNamespace(name="llm"),
        split="dev",
    )

    assert summary.total == 2
    assert summary.passed == 1
    assert summary.provider_failures == 1
    assert summary.completed_tasks == 1
    assert summary.provider_adjusted_tgc == 1.0


def test_run_evaluation_tracks_provider_impacted_tasks(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        train_and_eval,
        "load_task_ids",
        lambda split, data_dir=None, difficulty=None, domain=None: [
            "hard_decision_workflow_1",
            "hard_decision_workflow_2",
        ],
    )
    monkeypatch.setattr(
        train_and_eval,
        "load_task",
        lambda task_id, data_dir=None: SimpleNamespace(domains=["calendar"], difficulty=3),
    )

    def _run_episode(task_id, *_, **__):
        return train_and_eval.EpisodeResult(
            task_id=task_id,
            agent_name="llm",
            success=task_id.endswith("_2"),
            score=1.0 if task_id.endswith("_2") else 0.5,
            steps=1,
            duration_s=0.01,
            provider_impacted=task_id.endswith("_1"),
        )

    monkeypatch.setattr(train_and_eval, "run_episode", _run_episode)

    summary = train_and_eval.run_evaluation(
        agent=SimpleNamespace(name="llm"),
        split="dev",
    )

    assert summary.total == 2
    assert summary.provider_failures == 0
    assert summary.provider_impacted_tasks == 1
    assert summary.clean_completed_tasks == 1


def test_run_episode_stops_early_on_success_by_default():
    class _StickyAgent(train_and_eval.BaseAgent):
        name = "sticky"

        def __init__(self):
            self._step = 0

        def reset(self, task_instruction: str, task=None) -> None:
            self._step = 0

        def act(self, observation):
            self._step += 1
            if self._step == 1:
                return "openclaw agents add researcher --model anthropic/claude-opus-4-6"
            return "openclaw agents list"

    result = train_and_eval.run_episode(
        task_id="agent_create_1",
        agent=_StickyAgent(),
        mode="mock",
        max_steps=10,
        record_trajectory=True,
        verbose=False,
    )

    assert result.success is True
    assert result.steps == 1
    assert len(result.trajectory) == 1


def test_run_episode_does_not_execute_done_command():
    calls = {"step": 0}

    class _DoneAgent(train_and_eval.BaseAgent):
        name = "done-agent"

        def reset(self, task_instruction: str, task=None) -> None:
            return None

        def act(self, observation):
            return "DONE"

    class _FakeEnv:
        def __init__(self):
            self.task = SimpleNamespace(domains=["tasks"], ground_truth=None)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def reset(self):
            return SimpleNamespace(
                task_instruction="Add a task.",
                command_output="",
                error_output="",
                exit_code=0,
                step_number=0,
            )

        def step(self, action):
            calls["step"] += 1
            raise AssertionError(f"env.step should not be called for {action!r}")

        def evaluate(self):
            return train_and_eval.EvaluationResult(success=False, score=0.0, details=[])

    original_make_env = train_and_eval.make_env
    try:
        train_and_eval.make_env = lambda *_, **__: _FakeEnv()
        result = train_and_eval.run_episode(
            task_id="tasks_add_1",
            agent=_DoneAgent(),
            mode="mock",
            max_steps=3,
            verbose=False,
        )
    finally:
        train_and_eval.make_env = original_make_env

    assert calls["step"] == 0
    assert result.steps == 0


def test_run_episode_stops_after_repeated_stagnant_transitions():
    class _LoopAgent(train_and_eval.BaseAgent):
        name = "loop-agent"

        def reset(self, task_instruction: str, task=None) -> None:
            return None

        def act(self, observation):
            return "tasks list"

    class _FakeEnv:
        def __init__(self):
            self.task = SimpleNamespace(domains=["tasks"], ground_truth=None)
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def reset(self):
            return SimpleNamespace(
                task_instruction="List tasks.",
                command_output="",
                error_output="",
                exit_code=0,
                step_number=0,
            )

        def step(self, action):
            self.calls += 1
            return (
                SimpleNamespace(
                    task_instruction="List tasks.",
                    command_output="Tasks:\n[item_1] Demo",
                    error_output="",
                    exit_code=0,
                    step_number=self.calls,
                ),
                0.0,
                False,
                {},
            )

        def evaluate(self):
            return train_and_eval.EvaluationResult(success=False, score=0.0, details=[])

    original_make_env = train_and_eval.make_env
    fake_env = _FakeEnv()
    try:
        train_and_eval.make_env = lambda *_, **__: fake_env
        result = train_and_eval.run_episode(
            task_id="tasks_add_1",
            agent=_LoopAgent(),
            mode="mock",
            max_steps=10,
            max_stagnant_steps=3,
            record_trajectory=True,
            verbose=False,
        )
    finally:
        train_and_eval.make_env = original_make_env

    assert fake_env.calls == 3
    assert result.steps == 3
    assert len(result.trajectory) == 3


def test_run_episode_step_cap_reports_max_steps_over_missing_side_effect():
    # H2/H3: an episode that exhausts its step budget must be reported as
    # "reached max_steps", even when a required side effect is also missing.
    # Otherwise step_capped_tasks reads 0 and done_early_tasks is inflated.
    class _BusyAgent(train_and_eval.BaseAgent):
        name = "busy-agent"

        def reset(self, task_instruction: str, task=None) -> None:
            self._i = 0

        def act(self, observation):
            self._i += 1
            return f"tasks list --tag t{self._i}"  # distinct each step (no stagnation)

    ground_truth = SimpleNamespace(
        solution_commands=[],
        evaluation_checks=[
            {
                "type": "effect",
                "effect_type": "files_created",
                "expected": {"field": "path", "value": "/reports/x.md"},
            }
        ],
    )

    class _FakeEnv:
        def __init__(self):
            self.task = SimpleNamespace(
                domains=["tasks"],
                ground_truth=ground_truth,
                instruction="Write a report to /reports/x.md",
            )
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def reset(self):
            return SimpleNamespace(
                task_instruction="Write a report to /reports/x.md",
                command_output="",
                error_output="",
                exit_code=0,
                step_number=0,
            )

        def step(self, action):
            self.calls += 1
            return (
                SimpleNamespace(
                    task_instruction="Write a report to /reports/x.md",
                    command_output=f"out{self.calls}",
                    error_output="",
                    exit_code=0,
                    step_number=self.calls,
                ),
                0.0,
                False,
                {},
            )

        def evaluate(self):
            return train_and_eval.EvaluationResult(success=False, score=0.0, details=[])

    fake_env = _FakeEnv()
    original_make_env = train_and_eval.make_env
    try:
        train_and_eval.make_env = lambda *_, **__: fake_env
        result = train_and_eval.run_episode(
            task_id="tasks_add_1",
            agent=_BusyAgent(),
            mode="mock",
            max_steps=4,
            max_stagnant_steps=0,
            verbose=False,
        )
    finally:
        train_and_eval.make_env = original_make_env

    assert result.steps == 4
    assert result.stop_reason.startswith("reached max_steps=")
    assert "missing_required_side_effect" not in result.stop_reason


def test_eval_summary_step_capped_and_done_early_are_disjoint():
    # H2/H3: step-capped and done-early must be mutually exclusive buckets.
    R = train_and_eval.EpisodeResult
    results = [
        R(task_id="a", agent_name="x", success=False, score=0.0, steps=20,
          duration_s=0.0, stop_reason="reached max_steps=20"),
        R(task_id="b", agent_name="x", success=False, score=0.0, steps=4,
          duration_s=0.0, stop_reason="missing_required_side_effect: files_created"),
        R(task_id="c", agent_name="x", success=False, score=0.0, steps=3,
          duration_s=0.0,
          stop_reason="agent returned DONE before completing required action"),
        R(task_id="d", agent_name="x", success=True, score=1.0, steps=5,
          duration_s=0.0, stop_reason=""),
    ]
    summary = train_and_eval.EvalSummary(
        agent_name="x", split="dev", total=4, passed=1, total_score=1.0,
        results=results,
    )
    assert summary.step_capped_tasks == 1  # only "a"
    assert summary.done_early_tasks == 2  # "b" and "c"
    capped_ids = {
        r.task_id for r in results if r.stop_reason.startswith("reached max_steps=")
    }
    done_early_ids = {
        r.task_id
        for r in results
        if r.stop_reason.startswith("missing_required_side_effect:")
        or "DONE before completing required action" in r.stop_reason
    }
    assert capped_ids.isdisjoint(done_early_ids)


def test_run_episode_verbose_failure_prints_reason_and_failed_checks(capsys):
    class _DoneAgent(train_and_eval.BaseAgent):
        name = "done-agent"

        def reset(self, task_instruction: str, task=None) -> None:
            return None

        def act(self, observation):
            return "DONE"

    result = train_and_eval.run_episode(
        task_id="tasks_add_1",
        agent=_DoneAgent(),
        mode="mock",
        max_steps=3,
        verbose=True,
    )

    out = capsys.readouterr().out
    assert result.success is False
    assert "reason:" in out
    assert "failed checks:" in out


def test_run_episode_marks_provider_impacted_from_reply_debug():
    class _NoisyAgent(train_and_eval.BaseAgent):
        name = "noisy-agent"

        def reset(self, task_instruction: str, task=None) -> None:
            self._calls = 0

        def act(self, observation):
            self._calls += 1
            if self._calls == 1:
                return "tasks list"
            return "DONE"

        def debug_last_reply(self):
            return None

        def debug_reply_attempts(self):
            if self._calls == 1:
                return [
                    "api-fallback: content filter triggered; retrying with compact prompt",
                    "api: content='tasks list'",
                ]
            return ["api: content='DONE'"]

    class _FakeEnv:
        def __init__(self):
            self.task = SimpleNamespace(domains=["tasks"], ground_truth=None)
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def reset(self):
            return SimpleNamespace(
                task_instruction="List tasks.",
                command_output="",
                error_output="",
                exit_code=0,
                step_number=0,
            )

        def step(self, action):
            self.calls += 1
            return (
                SimpleNamespace(
                    task_instruction="List tasks.",
                    command_output="Tasks:\n[item_1] Demo",
                    error_output="",
                    exit_code=0,
                    step_number=self.calls,
                ),
                0.0,
                False,
                {},
            )

        def evaluate(self):
            return train_and_eval.EvaluationResult(success=False, score=0.0, details=[])

    original_make_env = train_and_eval.make_env
    try:
        train_and_eval.make_env = lambda *_, **__: _FakeEnv()
        result = train_and_eval.run_episode(
            task_id="tasks_add_1",
            agent=_NoisyAgent(),
            mode="mock",
            max_steps=2,
            record_trajectory=True,
            verbose=False,
        )
    finally:
        train_and_eval.make_env = original_make_env

    assert result.provider_impacted is True
    assert result.trajectory
    assert any(
        "api-fallback: content filter triggered" in item
        for item in result.trajectory[0]["model_debug"]
    )


def test_run_episode_verbose_prints_last_model_reply_on_done(capsys):
    agent = train_and_eval.LLMAgent(provider="openai", model="gpt-5.2")
    task = train_and_eval.load_task(
        "hard_decision_workflow_1",
        data_dir=Path(train_and_eval.DEFAULT_TASK_DATA_DIR),
    )
    agent.reset("Review budget-related email.", task=task)
    agent._openai_response_text = lambda: "DONE"

    result = train_and_eval.run_episode(
        task_id="hard_decision_workflow_1",
        agent=agent,
        mode="mock",
        max_steps=3,
        verbose=True,
    )

    out = capsys.readouterr().out
    assert result.success is False
    assert "model: parsed: raw='DONE' normalized='DONE'" in out


def test_run_episode_marks_date_anchor_mismatch_for_host_date_leak(monkeypatch: pytest.MonkeyPatch):
    class _DateLeakAgent(train_and_eval.BaseAgent):
        name = "date-leak"

        def reset(self, task_instruction: str, task=None) -> None:
            self._calls = 0

        def act(self, observation):
            self._calls += 1
            if self._calls == 1:
                return "calendar list --from 2026-04-24 --to 2026-04-25"
            return "DONE"

    class _FakeEnv:
        def __init__(self):
            self.task = train_and_eval.load_task(
                "hard_decision_workflow_1",
                data_dir=Path(train_and_eval.DEFAULT_TASK_DATA_DIR),
            )
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def reset(self):
            return SimpleNamespace(
                task_instruction=self.task.instruction,
                command_output="",
                error_output="",
                exit_code=0,
                step_number=0,
            )

        def step(self, action):
            self.calls += 1
            return (
                SimpleNamespace(
                    task_instruction=self.task.instruction,
                    command_output="No events found.",
                    error_output="",
                    exit_code=0,
                    step_number=self.calls,
                ),
                0.0,
                False,
                {},
            )

        def evaluate(self):
            return train_and_eval.EvaluationResult(success=False, score=0.5, details=[])

    class _FixedDate(train_and_eval.date):
        @classmethod
        def today(cls):
            return cls(2026, 4, 25)

    original_make_env = train_and_eval.make_env
    try:
        monkeypatch.setattr(train_and_eval, "date", _FixedDate)
        train_and_eval.make_env = lambda *_, **__: _FakeEnv()
        result = train_and_eval.run_episode(
            task_id="hard_decision_workflow_1",
            agent=_DateLeakAgent(),
            mode="mock",
            max_steps=2,
            verbose=False,
        )
    finally:
        train_and_eval.make_env = original_make_env

    assert result.date_anchor_mismatch is True


def test_llm_agent_rejects_done_after_only_read_only_steps(monkeypatch: pytest.MonkeyPatch):
    responses = iter(["DONE", "tasks add --title 'Berlin budget follow-up' --priority high"])
    agent = train_and_eval.LLMAgent(provider="openai", model="gpt-5.2")
    task = train_and_eval.load_task(
        "hard_decision_workflow_1",
        data_dir=Path(train_and_eval.DEFAULT_TASK_DATA_DIR),
    )
    agent.reset("Review budget-related email, turn it into a follow-up for Berlin.", task=task)
    agent._history = [
        {"action": "email search --query budget", "stdout": "Emails matching...", "stderr": "", "exit_code": 0},
        {"action": "email read --id email_seed_3", "stdout": "Email [email_seed_3]", "stderr": "", "exit_code": 0},
    ]
    monkeypatch.setattr(agent, "_openai_response_text", lambda: next(responses))

    action = agent.act(
        SimpleNamespace(
            step_number=2,
            command_output="Email [email_seed_3]",
            error_output="",
            exit_code=0,
        )
    )

    assert action == "tasks add --title 'Berlin budget follow-up' --priority high"
    attempts = agent.debug_reply_attempts()
    assert any(
        "validator: the task likely still requires a state-changing command before DONE" in attempt
        for attempt in attempts
    )


def test_llm_agent_retries_empty_openai_content_then_recovers(monkeypatch: pytest.MonkeyPatch):
    # A contentless (HTTP 200) OpenAI reply is now re-sent under the dedicated
    # empty-reply budget INSIDE ``_openai_response_text`` (not re-prompted by the
    # invalid-retry loop). A transient empty followed by a valid reply therefore
    # recovers in a single ``act`` turn with no stall.
    valid = (
        "email send --to alice@example.com --subject 'Budget update'"
        " --body 'Following up on Berlin.'"
    )
    payloads = iter(
        [
            {"choices": [{"message": {"content": None}}]},  # empty -> retry
            {"choices": [{"message": {"content": valid}}]},  # recovered
        ]
    )
    agent = train_and_eval.LLMAgent(provider="openai", model="gpt-5.2")
    task = train_and_eval.load_task(
        "hard_decision_workflow_1",
        data_dir=Path(train_and_eval.DEFAULT_TASK_DATA_DIR),
    )
    agent.reset("Review budget-related email and send Alice a short update.", task=task)

    monkeypatch.setattr(agent, "_openai_request", lambda *a, **k: next(payloads))

    action = agent.act(
        SimpleNamespace(
            step_number=2,
            command_output="Email [email_seed_3]",
            error_output="",
            exit_code=0,
        )
    )

    assert action == valid
    assert agent.debug_stall_reason() is None  # recovered by retry, not a stall
    attempts = agent.debug_reply_attempts()
    assert any("api-empty:" in attempt for attempt in attempts)


def test_openai_terminal_empty_reply_sets_generation_stall(monkeypatch: pytest.MonkeyPatch):
    # A persistently empty OpenAI reply exhausts the dedicated empty-reply budget
    # and short-circuits straight to a generation stall (kept a provider_failure)
    # instead of burning the invalid-retry re-prompts. The exact request count
    # proves BOTH the empty-reply cap AND that the invalid-retry loop does not
    # re-run the whole provider round (no nested retries).
    agent = train_and_eval.LLMAgent(provider="openai", model="gpt-5.2")
    agent.reset("investigate the issue", None)
    calls = {"n": 0}

    def _req(*a, **k):
        calls["n"] += 1
        return {"choices": [{"message": {"content": None}}]}

    monkeypatch.setattr(agent, "_openai_request", _req)

    action = agent.act(train_and_eval.Observation(step_number=0))

    assert action == "DONE"
    assert agent.debug_stall_reason() == "generation_stall_empty_reply"
    # 1 initial call + _empty_reply_retries (min(5, request_retries=1) == 1) == 2,
    # NOT (invalid_retry_limit + 1) x 2 which nesting would produce.
    assert calls["n"] == 2


def test_empty_reply_retries_never_exceeds_request_retries() -> None:
    # The empty-reply budget is clamped to the general request-retry budget so a
    # smaller ``--llm-request-retries`` can never be silently exceeded.
    agent = train_and_eval.LLMAgent(
        provider="openai",
        model="legacy-openai-chat-model",
        base_url="https://example.test/v1",
        request_retries=1,
        empty_reply_retries=5,
    )
    assert agent._empty_reply_retries == 1


def test_empty_reply_retry_backs_off_between_attempts(monkeypatch: pytest.MonkeyPatch):
    # Each empty-reply retry sleeps with the ordinary (non-deadline) backoff,
    # exactly like an exception retry, so an overloaded backend gets breathing
    # room before the verbatim re-send.
    sleeps: list[float] = []
    monkeypatch.setattr(train_and_eval.time, "sleep", lambda s: sleeps.append(s))
    valid = "weather get --location 'Paris'"
    payloads = iter(
        [
            {"choices": [{"message": {"content": None}}]},  # empty -> retry
            {"choices": [{"message": {"content": valid}}]},  # recovered
        ]
    )
    agent = train_and_eval.LLMAgent(
        provider="openai",
        model="legacy-openai-chat-model",
        base_url="https://example.test/v1",
        request_retries=1,
        retry_backoff_s=2.0,
    )
    agent.reset("Check the weather in Paris.")
    monkeypatch.setattr(agent, "_openai_request", lambda *a, **k: next(payloads))

    assert agent._openai_response_text() == valid
    # One empty retry -> exactly one equal-jitter sleep in [base/2, base].
    assert len(sleeps) == 1
    assert 1.0 <= sleeps[0] <= 2.0
    assert any("api-backoff" in a for a in agent.debug_reply_attempts())


def test_non_empty_invalid_reply_still_uses_invalid_retry_loop(monkeypatch: pytest.MonkeyPatch):
    # Regression: the empty-reply short-circuit must NOT fire on a non-empty but
    # invalid reply. Such a reply still earns the catalog re-prompt, so a later
    # valid command recovers the turn (no stall).
    replies = iter(["not a real command", "weather get --location 'Paris'"])
    agent = train_and_eval.LLMAgent(provider="openai", model="gpt-5.2")
    agent.reset("Check the weather in Paris.")
    monkeypatch.setattr(agent, "_openai_response_text", lambda: next(replies))

    action = agent.act(train_and_eval.Observation(step_number=0))

    assert action == "weather get --location 'Paris'"
    assert agent.debug_stall_reason() is None
    # The invalid (non-empty) reply produced a catalog nudge, not a stall.
    assert any("did not match the command catalog" in m["content"]
               for m in agent._messages if m["role"] == "user")


def test_act_native_empty_reply_short_circuits_to_stall(monkeypatch: pytest.MonkeyPatch):
    # Native counterpart of the empty-reply short-circuit. When the provider
    # layer already exhausted its empty-reply retries (returns no tool call AND
    # no prose text), ``_act_native`` must stall immediately rather than nudge
    # ``invalid_retry_limit + 1`` times (each re-running a full provider round).
    agent = train_and_eval.LLMAgent(
        provider="openai", model="gpt-5.6-sol", tool_mode="native"
    )
    agent.reset("investigate the issue", None)
    calls = {"n": 0}

    def _empty_native_call():
        calls["n"] += 1
        agent._last_no_call_text = ""  # no tool call AND no prose == empty
        return None

    monkeypatch.setattr(agent, "_native_tool_call", _empty_native_call)

    action = agent.act(train_and_eval.Observation(step_number=0))

    assert action == "DONE"
    assert agent.debug_stall_reason() == "generation_stall_empty_reply"
    assert calls["n"] == 1  # short-circuit: no invalid-retry re-prompts


def test_act_native_prose_without_call_still_nudges(monkeypatch: pytest.MonkeyPatch):
    # Guard: a prose-but-no-call turn (the model reasoned without acting) is NOT
    # an empty reply -- the empty-reply short-circuit must NOT fire when prose is
    # present, so it still earns the nudge across the full invalid-retry budget.
    agent = train_and_eval.LLMAgent(
        provider="openai", model="gpt-5.6-sol", tool_mode="native"
    )
    agent.reset("investigate the issue", None)
    calls = {"n": 0}

    def _prose_native_call():
        calls["n"] += 1
        agent._last_no_call_text = "I will think about it."  # prose, no call
        return None

    monkeypatch.setattr(agent, "_native_tool_call", _prose_native_call)

    agent.act(train_and_eval.Observation(step_number=0))

    # No short-circuit: prose is re-prompted across the whole invalid-retry
    # budget (invalid_retry_limit + 1 calls), unlike an empty reply (1 call).
    assert calls["n"] == agent._invalid_retry_limit + 1
    assert any("did not call a tool" in m.get("content", "")
               for m in agent._messages if m.get("role") == "user")


def test_llm_agent_builds_memory_aware_openai_request_messages():
    agent = train_and_eval.LLMAgent(provider="openai", model="gpt-5.2")
    messages = [
        {"role": "user", "content": "Task: first"},
        {"role": "assistant", "content": "cmd-1"},
        {"role": "user", "content": "obs-1"},
        {"role": "assistant", "content": "cmd-2"},
        {"role": "user", "content": "obs-2"},
        {"role": "assistant", "content": "cmd-3"},
        {"role": "user", "content": "obs-3"},
        {"role": "assistant", "content": "cmd-4"},
        {"role": "user", "content": "obs-4"},
    ]
    agent._history = [
        {"action": "weather forecast --location Berlin --days 1", "stdout": "Rain tomorrow", "stderr": "", "exit_code": 0},
        {"action": "calendar today --timezone Europe/Berlin", "stdout": "Review at 14:00", "stderr": "", "exit_code": 0},
        {"action": "tasks list --status pending", "stdout": "1. Berlin follow-up", "stderr": "", "exit_code": 0},
        {"action": "openclaw cron list", "stdout": "existing-daily-check", "stderr": "", "exit_code": 0},
        {"action": "openclaw config get agent.model", "stdout": "agent.model = openai/gpt-4o", "stderr": "", "exit_code": 0},
        {"action": "weather forecast --location Berlin --days 1", "stdout": "Clear later", "stderr": "", "exit_code": 0},
        {"action": "tasks add --title 'Berlin follow-up'", "stdout": "Created task", "stderr": "", "exit_code": 0},
        {"action": "calendar add-event --title 'Berlin sync' --start 2026-03-27T14:00", "stdout": "", "stderr": "slot conflict", "exit_code": 1},
    ]

    trimmed = agent._trim_request_messages(messages)

    assert trimmed[0] == messages[0]
    assert trimmed[1]["role"] == "user"
    assert trimmed[1]["content"].startswith("Known context:\n")
    assert "weather forecast --location Berlin --days 1 => stdout=Clear later" in trimmed[1]["content"]
    assert "calendar today --timezone Europe/Berlin => stdout=Review at 14:00" in trimmed[1]["content"]
    assert "tasks list --status pending => stdout=1. Berlin follow-up" in trimmed[1]["content"]
    assert "openclaw cron list => stdout=existing-daily-check" in trimmed[1]["content"]
    assert "openclaw config get agent.model => stdout=agent.model = openai/gpt-4o" in trimmed[1]["content"]
    assert "Rain tomorrow" not in trimmed[1]["content"]
    assert "last state-changing action: tasks add --title 'Berlin follow-up'" in trimmed[1]["content"]
    assert "recent state-changing action: calendar add-event --title 'Berlin sync' --start 2026-03-27T14:00" in trimmed[1]["content"]
    assert "last failed action: calendar add-event --title 'Berlin sync' --start 2026-03-27T14:00" in trimmed[1]["content"]
    assert trimmed[2:] == messages[-8:]


def test_llm_agent_skips_memory_summary_for_short_history():
    agent = train_and_eval.LLMAgent(provider="openai", model="gpt-5.2")
    messages = [
        {"role": "user", "content": "Task: first"},
        {"role": "assistant", "content": "cmd-1"},
        {"role": "user", "content": "obs-1"},
    ]

    trimmed = agent._trim_request_messages(messages)

    assert trimmed == messages


def test_llm_agent_uses_litellm_proxy_defaults_for_claude_openai_models(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("BEDROCK_LITELLM_BASE_URL", raising=False)
    monkeypatch.setenv("LITELLM_PROXY_KEY", "proxy-key")
    monkeypatch.setenv("LITELLM_PROXY_BASE_URL", "http://localhost:4000/v1")

    agent = train_and_eval.LLMAgent(provider="openai", model="claude-sonnet-4.6")

    assert agent._api_key_name() == "LITELLM_PROXY_KEY"
    assert agent._api_key() == "proxy-key"
    assert agent._openai_base_url() == "http://localhost:4000/v1"


def test_llm_agent_hard_prompt_uses_raw_task_instruction_without_benchmark_notes():
    agent = train_and_eval.LLMAgent(provider="openai", model="gpt-5.2")
    task = train_and_eval.load_task(
        "hard_decision_workflow_1",
        data_dir=Path(train_and_eval.DEFAULT_TASK_DATA_DIR),
    )

    agent.reset(task.instruction, task=task)

    assert "Benchmark time anchor" not in agent._system_prompt
    assert "board' means the task board managed with `tasks *` commands" not in agent._system_prompt
    assert "calendar today --timezone Europe/Berlin" not in agent._system_prompt
    assert agent._messages[0]["content"].startswith(f"Task: {task.instruction}")
    assert "Complete the task using the available CLI commands." in agent._messages[0]["content"]


def test_llm_agent_explicit_openai_base_url_and_key_env_override_litellm_defaults(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("LITELLM_PROXY_KEY", "proxy-key")
    monkeypatch.setenv("CUSTOM_KEY_ENV", "custom-key")

    agent = train_and_eval.LLMAgent(
        provider="openai",
        model="claude-sonnet-4.6",
        base_url="https://example.test/v1",
        api_key_env="CUSTOM_KEY_ENV",
    )

    assert agent._api_key_name() == "CUSTOM_KEY_ENV"
    assert agent._api_key() == "custom-key"
    assert agent._openai_base_url() == "https://example.test/v1"


def test_openai_content_filter_retries_with_compact_prompt(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    captured_bodies: list[dict[str, object]] = []

    class _Body:
        def __init__(self, text: str) -> None:
            self._text = text

        def read(self) -> bytes:
            return self._text.encode("utf-8")

        def close(self) -> None:
            return None

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {"choices": [{"message": {"content": "tasks add --title 'Berlin follow-up'", "role": "assistant"}}]}
            ).encode("utf-8")

    calls = {"count": 0}

    def _urlopen(req, timeout=0):
        captured_bodies.append(json.loads(req.data.decode("utf-8")))
        calls["count"] += 1
        if calls["count"] == 1:
            raise train_and_eval.urllib.error.HTTPError(
                req.full_url,
                400,
                "Bad Request",
                hdrs=None,
                fp=_Body(
                    '{"error":{"message":"The response was filtered due to the prompt triggering policy.","param":"prompt","code":"content_filter","innererror":{"code":"ResponsibleAIPolicyViolation","content_filter_result":{"jailbreak":{"filtered":true,"detected":true}}}}}'
                ),
            )
        return _Response()

    monkeypatch.setattr(train_and_eval.urllib.request, "urlopen", _urlopen)

    agent = train_and_eval.LLMAgent(
        provider="openai",
        model="gpt-5.2",
        base_url="https://example.test/v1",
    )
    task = train_and_eval.load_task(
        "hard_decision_workflow_1",
        data_dir=Path(train_and_eval.DEFAULT_TASK_DATA_DIR),
    )
    agent.reset(task.instruction, task=task)
    agent._history = [
        {"action": "openclaw config get agent.model", "stdout": "agent.model = openai/gpt-4o", "stderr": "", "exit_code": 0},
        {"action": "calendar list", "stdout": "Events: 1", "stderr": "", "exit_code": 0},
        {"action": "tasks list --status pending", "stdout": "Tasks: 1", "stderr": "", "exit_code": 0},
    ]

    text = agent._openai_response_text()

    assert text == "tasks add --title 'Berlin follow-up'"
    assert len(captured_bodies) == 2
    assert "content filter triggered" in " ".join(agent.debug_reply_attempts())
    first_payload = captured_bodies[0]
    second_payload = captured_bodies[1]
    assert first_payload["instructions"] == agent._system_prompt
    assert len(first_payload["input"]) >= 1
    assert second_payload["instructions"] == agent._compact_system_prompt()
    compact_user = second_payload["input"][0]["content"]
    assert "Known context:" in compact_user
    assert "calendar list" in compact_user
    assert "tasks list --status pending" in compact_user
    assert "openclaw config get agent.model" in compact_user
    assert "last state-changing action" not in compact_user


def test_eval_summary_prints_hard_scenario_and_ability_sections(capsys: pytest.CaptureFixture[str]) -> None:
    summary = train_and_eval.EvalSummary(
        agent_name="llm",
        split="dev",
        total=4,
        passed=2,
        total_score=2.5,
        exec_mode="multi",
        llm_provider="openai",
        llm_model="claude-sonnet-4.6",
        llm_history_mode="full",
        by_hard_scenario={
            "existing_state_followthrough": train_and_eval.DomainMetrics(passed=1, total=2, total_score=1.1),
        },
        by_hard_ability={
            "gap_completion": train_and_eval.DomainMetrics(passed=1, total=2, total_score=1.1),
        },
        by_hard_ability_tag={
            "duplicate_avoidance": train_and_eval.DomainMetrics(passed=1, total=3, total_score=2.0),
        },
        results=[
            train_and_eval.EpisodeResult(
                task_id="hard_decision_workflow_1",
                agent_name="llm",
                success=False,
                score=0.8,
                steps=8,
                duration_s=0.1,
                stop_reason="agent returned DONE before executing another command",
                date_anchor_mismatch=True,
            ),
            train_and_eval.EpisodeResult(
                task_id="hard_decision_workflow_2",
                agent_name="llm",
                success=False,
                score=0.2,
                steps=30,
                duration_s=0.1,
                stop_reason="reached max_steps=30",
            ),
        ],
    )

    summary.print_report()
    out = capsys.readouterr().out
    assert "Mode : multi" in out
    assert "claude-sonnet-4.6" in out
    assert "History : full" in out
    assert "pass@budget=50.0%" in out
    assert "near_miss_rate=25.0%" in out
    assert "done_early_tasks=1" in out
    assert "step_capped_tasks=1" in out
    assert "date_anchor_mismatch_tasks=1" in out
    assert "Compare avg_score first and raw_accuracy second" in out
    assert "By hard scenario:" in out
    assert "existing_state_followthrough" in out
    assert "By hard primary ability:" in out
    assert "gap_completion" in out
    assert "By hard ability tag (overlapping):" in out
    assert "duplicate_avoidance" in out


def test_cli_save_report_includes_llm_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _DummySummary:
        agent_name = "llm"
        split = "test"
        exec_mode = "multi"
        llm_provider = "openai"
        llm_model = "claude-sonnet-4.6"
        llm_history_mode = "summary"
        tgc = 0.5
        pass_at_budget = 0.5
        avg_score = 0.75
        near_miss_rate = 0.25
        near_miss_tasks = 1
        done_early_tasks = 1
        step_capped_tasks = 0
        date_anchor_mismatch_tasks = 1
        score_rank_note = "Compare avg_score first and raw_accuracy second."
        passed = 1
        total = 2
        completed_tasks = 2
        clean_completed_tasks = 2
        provider_failures = 0
        provider_impacted_tasks = 0
        compact_fallback_tasks = 0
        provider_adjusted_tgc = 0.5
        clean_tgc = 0.5
        by_domain: dict[str, object] = {}
        by_difficulty: dict[str, object] = {}
        by_hard_scenario: dict[str, object] = {}
        by_hard_ability: dict[str, object] = {}
        by_hard_ability_tag: dict[str, object] = {}
        results: list[object] = []

        def print_report(self) -> None:
            return None

    monkeypatch.setattr(train_and_eval, "_preflight_mode", lambda mode: None)
    monkeypatch.setattr(train_and_eval, "_build_agent", lambda *_, **__: object())
    monkeypatch.setattr(train_and_eval, "run_evaluation", lambda *_, **__: _DummySummary())

    report_path = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent",
            "llm",
            "--split",
            "test",
            "--save-report",
            str(report_path),
        ],
    )

    train_and_eval.main()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["mode"] == "multi"
    assert report["max_steps_budget"] == 15
    assert report["llm_provider"] == "openai"
    assert report["llm_model"] == "claude-sonnet-4.6"
    assert report["llm_history_mode"] == "summary"
    assert report["pass_at_budget"] == 0.5
    assert report["near_miss_rate"] == 0.25
    assert report["done_early_tasks"] == 1
    assert report["date_anchor_mismatch_tasks"] == 1
    assert "avg_score first" in report["score_rank_note"]


def test_run_episode_records_step_eval_only_when_enabled() -> None:
    default_result = train_and_eval.run_episode(
        "tasks_add_1",
        train_and_eval.ExpertAgent(),
        mode="martin",
        max_steps=3,
        record_trajectory=True,
        record_step_eval=False,
    )
    dense_result = train_and_eval.run_episode(
        "tasks_add_1",
        train_and_eval.ExpertAgent(),
        mode="martin",
        max_steps=3,
        record_trajectory=True,
        record_step_eval=True,
    )

    assert default_result.success is True
    assert dense_result.success is True
    assert default_result.trajectory
    assert dense_result.trajectory
    assert "step_evaluation" not in default_result.trajectory[0]
    assert dense_result.trajectory[0]["step_evaluation"]["current_score"] == 1.0
    assert dense_result.trajectory[0]["current_score"] == 1.0
    assert dense_result.trajectory[0]["score_delta"] == 1.0
    assert dense_result.trajectory[0]["passed_checks"] == ["task created"]
    assert dense_result.metadata["mode"] == "martin"


def test_multimodal_image_parts_use_round_visible_assets(tmp_path: Path) -> None:
    from clasharness.core.task import GroundTruth, Task, TaskData

    task = Task(
        task_id="vlm_demo_q01",
        instruction="inspect visible asset",
        domains=["asset"],
        difficulty=3,
        initial_state="default",
        data=TaskData(
            public={},
            private={
                "initial_state_overrides": {
                    "asset_manifest.json": {
                        "assets": [
                            {
                                "id": "asset_a",
                                "kind": "image",
                                "path": "a.svg",
                                "mime_type": "image/svg+xml",
                            }
                        ]
                    },
                    "assets/a.svg": "<svg><text>visible</text></svg>",
                }
            },
        ),
        ground_truth=GroundTruth(solution_commands=["DONE"], evaluation_checks=[]),
    )

    parts = train_and_eval._multimodal_image_parts_from_task(task, mode="direct")
    assert len(parts) == 1
    assert parts[0]["type"] == "image_url"
    assert parts[0]["image_url"]["url"].startswith("data:image/svg+xml;base64,")
    assert train_and_eval._effective_multimodal_input_mode(
        requested="direct",
        provider="openai",
        task=task,
    ) == "direct"
    assert train_and_eval._effective_multimodal_input_mode(
        requested="direct",
        provider="anthropic",
        task=task,
    ) == "path"


def test_effect_satisfied_by_actions_assets_accessed() -> None:
    expected = {"field": "id", "value": "api_gateway_chart"}
    
    # Matches info
    assert train_and_eval._effect_satisfied_by_actions(
        "assets_accessed", expected, ["asset info --id api_gateway_chart"]
    ) is True
    # Matches read-text
    assert train_and_eval._effect_satisfied_by_actions(
        "assets_accessed", expected, ["asset read-text --id api_gateway_chart"]
    ) is True
    # Matches open
    assert train_and_eval._effect_satisfied_by_actions(
        "assets_accessed", expected, ["asset open --id api_gateway_chart"]
    ) is True
    # Matches describe
    assert train_and_eval._effect_satisfied_by_actions(
        "assets_accessed", expected, ["asset describe --id api_gateway_chart"]
    ) is True
    # Mismatch ID fails
    assert train_and_eval._effect_satisfied_by_actions(
        "assets_accessed", expected, ["asset info --id app_logs"]
    ) is False
    # Reading the matching asset's data file directly counts in this lenient
    # stop-detection heuristic (the authoritative grader still requires an actual
    # `asset` command via EffectChecker).
    assert train_and_eval._effect_satisfied_by_actions(
        "assets_accessed", expected, ["file read --path /assets/api_gateway_chart"]
    ) is True
    # An unrelated file read does NOT count.
    assert train_and_eval._effect_satisfied_by_actions(
        "assets_accessed", expected, ["file read --path /config/unrelated.yaml"]
    ) is False


def test_effect_satisfied_by_actions_messages_sent_relaxed() -> None:
    expected = {"field": "message", "value": "verify logs"}
    
    # Exact match
    assert train_and_eval._effect_satisfied_by_actions(
        "messages_sent", expected, ["agent delegate --agent Dev --instruction 'verify logs'"]
    ) is True
    # Relaxed match with gaps and pronouns ignored
    assert train_and_eval._effect_satisfied_by_actions(
        "messages_sent", expected, ["agent delegate --agent Dev --instruction 'please verify their logs now'"]
    ) is True
    # Out of order fails
    assert train_and_eval._effect_satisfied_by_actions(
        "messages_sent", expected, ["agent delegate --agent Dev --instruction 'logs should be verified'"]
    ) is False


# ---------------------------------------------------------------------------
# Claude-on-Vertex ('vertex-anthropic') provider
# ---------------------------------------------------------------------------


def test_vertex_anthropic_is_accepted_provider() -> None:
    agent = train_and_eval.LLMAgent(
        provider="vertex-anthropic",
        model="claude-sonnet-4@20250514",
        vertex_project="proj-x",
        vertex_location="us-east5",
    )
    assert agent._provider == "vertex-anthropic"


def test_unknown_provider_still_rejected() -> None:
    with pytest.raises(ValueError):
        train_and_eval.LLMAgent(provider="bedrock", model="claude-x")


def test_to_anthropic_messages_converts_data_uri_image() -> None:
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look at this"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,QUJD"},
                },
            ],
        },
        {"role": "assistant", "content": "ok"},
    ]
    out = train_and_eval._to_anthropic_messages(msgs)
    assert out[0]["content"][0] == {"type": "text", "text": "look at this"}
    assert out[0]["content"][1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"},
    }
    # String content passes through untouched.
    assert out[1] == {"role": "assistant", "content": "ok"}


def test_to_anthropic_messages_handles_remote_url_and_existing_block() -> None:
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "ZZ",
                    },
                },
                {"type": "unknown_kind", "junk": 1},
            ],
        },
    ]
    out = train_and_eval._to_anthropic_messages(msgs)
    parts = out[0]["content"]
    # Remote URL -> anthropic url source.
    assert parts[0] == {
        "type": "image",
        "source": {"type": "url", "url": "https://x/y.png"},
    }
    # Already-anthropic image block is preserved.
    assert parts[1]["source"]["media_type"] == "image/jpeg"
    # Unknown part types are dropped.
    assert len(parts) == 2


def test_to_anthropic_messages_preserves_tool_use_and_tool_result_blocks() -> None:
    # Regression guard for Bug 2: the byte-budget converter kept only text/image
    # parts, silently dropping native tool_use/tool_result blocks and leaving
    # ``content: []``. Vertex/Anthropic then rejected the request with
    # "messages.N: <role> messages must have non-empty content".
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "start"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,QUJD"},
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "call_1", "name": "asset_list",
                 "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_1",
                 "content": "ok"},
            ],
        },
    ]
    out = train_and_eval._to_anthropic_messages(msgs)
    # No turn is emptied (the core regression).
    assert all(m["content"] for m in out)
    # The tool turns keep their blocks verbatim.
    assert out[1]["content"] == [
        {"type": "tool_use", "id": "call_1", "name": "asset_list", "input": {}}
    ]
    assert out[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "call_1", "content": "ok"}
    ]
    # Ordinary image parts are still rewritten into Anthropic image blocks.
    assert out[0]["content"][1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"},
    }


def test_effective_multimodal_mode_enables_images_for_vertex_anthropic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force non-empty image parts so only the provider gate is under test.
    monkeypatch.setattr(
        train_and_eval.agents,
        "_multimodal_image_parts_from_task",
        lambda task, mode: [{"type": "image_url", "image_url": {"url": "x"}}],
    )
    assert (
        train_and_eval._effective_multimodal_input_mode(
            requested="direct", provider="vertex-anthropic", task=None
        )
        == "direct"
    )
    # Plain 'anthropic' stays path-only (unchanged behavior).
    assert (
        train_and_eval._effective_multimodal_input_mode(
            requested="direct", provider="anthropic", task=None
        )
        == "path"
    )


def test_vertex_anthropic_get_client_uses_adc_no_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import types as _types

    captured: dict[str, object] = {}

    class _FakeVertex:
        def __init__(self, project_id=None, region=None):
            captured["project_id"] = project_id
            captured["region"] = region

    fake_mod = _types.ModuleType("anthropic")
    fake_mod.AnthropicVertex = _FakeVertex
    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)
    # No ANTHROPIC_API_KEY: the Vertex path must not require one (ADC only).
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    agent = train_and_eval.LLMAgent(
        provider="vertex-anthropic",
        model="claude-sonnet-4@20250514",
        vertex_project="proj-x",
        vertex_location="us-east5",
    )
    client = agent._get_client()
    assert isinstance(client, _FakeVertex)
    assert captured == {"project_id": "proj-x", "region": "us-east5"}


def test_vertex_anthropic_response_text_converts_images_and_returns_text() -> None:
    captured: dict[str, object] = {}

    class _Txt:
        def __init__(self, t):
            self.text = t

    class _Resp:
        def __init__(self, t):
            self.content = [_Txt(t)]

    class _Msgs:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _Resp("file list ")

    class _Client:
        messages = _Msgs()

    agent = train_and_eval.LLMAgent(
        provider="vertex-anthropic",
        model="claude-sonnet-4@20250514",
        vertex_project="p",
        vertex_location="us-east5",
    )
    # Inject a fake client so no network/SDK is needed.
    agent._client = _Client()
    agent._messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "see chart"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,QUJD"},
                },
            ],
        }
    ]
    out = agent._anthropic_response_text()
    assert out == "file list"  # stripped
    assert captured["model"] == "claude-sonnet-4@20250514"
    # An explicit timeout must be passed so the SDK does not require streaming
    # for large max_tokens; it should equal the configured request timeout.
    assert captured["timeout"] == agent._request_timeout_s
    sent = captured["messages"][0]["content"]
    assert {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"},
    } in sent


def test_anthropic_native_tool_call_preserves_tool_turn_content() -> None:
    # End-to-end Bug 2 guard: a multi-step native Anthropic turn (an assistant
    # tool_use plus a user tool_result already in history) must reach the SDK
    # with non-empty content on every message. Before the fix the byte-budget
    # pass (_anthropic_create -> _fit_anthropic_messages_to_byte_budget ->
    # _to_anthropic_messages) stripped the tool blocks, so Vertex/Anthropic
    # rejected the request with "messages.N ... must have non-empty content".
    from train_eval import tool_calling as tc

    captured: dict[str, object] = {}

    class _Msgs:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                content=[
                    {"type": "tool_use", "id": "call_2", "name": "done",
                     "input": {}}
                ]
            )

    class _Client:
        messages = _Msgs()

    agent = train_and_eval.LLMAgent(
        provider="vertex-anthropic",
        model="claude-sonnet-4@20250514",
        vertex_project="p",
        vertex_location="us-east5",
    )
    agent._client = _Client()
    agent._native_tools_anthropic = [{"name": "asset_list"}, {"name": "done"}]

    prior = tc.ToolCall(name="asset_list", arguments={}, id="call_1")
    agent._messages = [
        {"role": "user", "content": "inspect the workspace"},
        tc.assistant_tool_turn(prior),
        tc.tool_result_turn(prior, "asset list output"),
    ]

    call = agent._anthropic_tool_call()
    assert call is not None and call.name == "done"

    sent = captured["messages"]
    # Every message reaches the SDK with non-empty content (the regression).
    assert all(m["content"] for m in sent)
    # The native tool_use / tool_result blocks survived the conversion.
    kinds = [
        part.get("type")
        for m in sent
        for part in (m["content"] if isinstance(m["content"], list) else [])
    ]
    assert "tool_use" in kinds
    assert "tool_result" in kinds


def test_act_routes_vertex_anthropic_to_anthropic_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = train_and_eval.LLMAgent(
        provider="vertex-anthropic",
        model="claude-sonnet-4@20250514",
        vertex_project="p",
        vertex_location="us-east5",
    )
    calls: list[str] = []

    def _anthropic():
        calls.append("anthropic")
        return "file list"

    def _openai():
        calls.append("openai")
        return "x"

    def _gemini():
        calls.append("gemini")
        return "x"

    monkeypatch.setattr(agent, "_anthropic_response_text", _anthropic)
    monkeypatch.setattr(agent, "_openai_response_text", _openai)
    monkeypatch.setattr(agent, "_gemini_response_text", _gemini)
    monkeypatch.setattr(
        train_and_eval.agents, "_validate_command", lambda action, task: (True, "")
    )

    task = train_and_eval.load_task(
        "hard_decision_workflow_1",
        data_dir=Path(train_and_eval.DEFAULT_TASK_DATA_DIR),
    )
    agent.reset("inspect the workspace", task=task)
    obs = train_and_eval.Observation(
        command_output="",
        error_output="",
        exit_code=0,
        current_config={},
        gateway_status="",
        available_commands=[],
        task_instruction="inspect the workspace",
        step_number=0,
    )
    result = agent.act(obs)
    assert calls == ["anthropic"]
    assert result == "file list"


# ---------------------------------------------------------------------------
# Grok-on-Vertex ('vertex-xai') provider
# ---------------------------------------------------------------------------


class _FakeAdcCreds:
    """Minimal stand-in for google.auth credentials used by vertex-xai tests."""

    def __init__(self, token: str, valid: bool) -> None:
        self.token = token
        self.valid = valid
        self.refresh_calls = 0

    def refresh(self, request: object) -> None:
        self.refresh_calls += 1
        self.valid = True
        self.token = "fresh"


def _install_fake_google_auth(
    monkeypatch: pytest.MonkeyPatch,
    creds: _FakeAdcCreds,
    captured: dict,
) -> None:
    """Inject fake google.auth modules so no real ADC lookup happens."""
    import types as _types
    import google  # real namespace package (google.genai is a dependency)

    fake_auth = _types.ModuleType("google.auth")
    fake_transport = _types.ModuleType("google.auth.transport")
    fake_requests = _types.ModuleType("google.auth.transport.requests")

    def _fake_default(scopes=None):
        captured["default_calls"] = captured.get("default_calls", 0) + 1
        captured["scopes"] = list(scopes) if scopes else scopes
        return creds, "proj-from-adc"

    fake_auth.default = _fake_default
    fake_auth.transport = fake_transport
    fake_transport.requests = fake_requests

    class _FakeRequest:
        pass

    fake_requests.Request = _FakeRequest

    monkeypatch.setitem(sys.modules, "google.auth", fake_auth)
    monkeypatch.setitem(sys.modules, "google.auth.transport", fake_transport)
    monkeypatch.setitem(sys.modules, "google.auth.transport.requests", fake_requests)
    monkeypatch.setattr(google, "auth", fake_auth, raising=False)


def test_vertex_xai_is_accepted_provider() -> None:
    agent = train_and_eval.LLMAgent(
        provider="vertex-xai",
        model="grok-4.20-reasoning",
        vertex_project="proj-x",
        vertex_location="global",
    )
    assert agent._provider == "vertex-xai"


def test_vertex_xai_prefixes_model_id_with_xai() -> None:
    # Bare model id gets the 'xai/' publisher prefix automatically.
    agent = train_and_eval.LLMAgent(
        provider="vertex-xai", model="grok-4.20-reasoning"
    )
    assert agent._model == "xai/grok-4.20-reasoning"
    # Already-prefixed ids are left untouched (no double prefix).
    agent2 = train_and_eval.LLMAgent(
        provider="vertex-xai", model="xai/grok-4.20-reasoning"
    )
    assert agent2._model == "xai/grok-4.20-reasoning"
    # Other providers never get the prefix.
    agent3 = train_and_eval.LLMAgent(
        provider="openai", model="grok-4.20-reasoning"
    )
    assert agent3._model == "grok-4.20-reasoning"


def test_vertex_xai_base_url_defaults_to_global() -> None:
    # 'us' (the default), an explicit 'global', and empty all map to the global
    # OpenAI-compatible endpoint.
    for loc in ("us", "global", ""):
        agent = train_and_eval.LLMAgent(
            provider="vertex-xai",
            model="grok-4.20-reasoning",
            vertex_project="proj-x",
            vertex_location=loc,
        )
        assert agent._openai_base_url() == (
            "https://aiplatform.googleapis.com/v1/projects/proj-x/"
            "locations/global/endpoints/openapi"
        )


def test_vertex_xai_base_url_uses_region_when_hyphenated() -> None:
    agent = train_and_eval.LLMAgent(
        provider="vertex-xai",
        model="grok-4.20-reasoning",
        vertex_project="proj-x",
        vertex_location="us-central1",
    )
    assert agent._openai_base_url() == (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/proj-x/"
        "locations/us-central1/endpoints/openapi"
    )


def test_vertex_xai_explicit_base_url_overrides_vertex_endpoint() -> None:
    agent = train_and_eval.LLMAgent(
        provider="vertex-xai",
        model="grok-4.20-reasoning",
        base_url="https://proxy.example/v1",
        vertex_project="proj-x",
        vertex_location="global",
    )
    assert agent._openai_base_url() == "https://proxy.example/v1"


def test_bedrock_gpt_is_accepted_provider() -> None:
    agent = train_and_eval.LLMAgent(
        provider="bedrock-gpt", model="openai.gpt-5.5"
    )
    assert agent._provider == "bedrock-gpt"
    # The 'openai.' publisher prefix is left untouched (unlike vertex-xai).
    assert agent._model == "openai.gpt-5.5"


def test_bedrock_gpt_base_url_defaults_to_mantle() -> None:
    agent = train_and_eval.LLMAgent(
        provider="bedrock-gpt", model="openai.gpt-5.5"
    )
    assert agent._openai_base_url() == (
        "https://bedrock-mantle.us-east-2.api.aws/openai/v1"
    )


def test_bedrock_gpt_explicit_base_url_overrides_mantle() -> None:
    agent = train_and_eval.LLMAgent(
        provider="bedrock-gpt",
        model="openai.gpt-5.5",
        base_url="https://bedrock-mantle.eu-west-1.api.aws/openai/v1",
    )
    assert agent._openai_base_url() == (
        "https://bedrock-mantle.eu-west-1.api.aws/openai/v1"
    )


def test_bedrock_gpt_key_env_defaults_to_bedrock_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = train_and_eval.LLMAgent(
        provider="bedrock-gpt", model="openai.gpt-5.5"
    )
    assert agent._api_key_name() == "BEDROCK_OPENAI_API_KEY"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("BEDROCK_OPENAI_API_KEY", "bedrock-secret")
    assert agent._api_key() == "bedrock-secret"
    # Explicit --llm-api-key-env still wins.
    agent2 = train_and_eval.LLMAgent(
        provider="bedrock-gpt",
        model="openai.gpt-5.5",
        api_key_env="CUSTOM_BEDROCK_KEY",
    )
    assert agent2._api_key_name() == "CUSTOM_BEDROCK_KEY"


def test_bedrock_gpt_forces_responses_api_and_omits_temperature() -> None:
    # The Bedrock model id ('openai.gpt-5.5') would not match the 'gpt-'/'gpt-5'
    # heuristics, so both must be forced on by provider instead.
    agent = train_and_eval.LLMAgent(
        provider="bedrock-gpt", model="openai.gpt-5.5"
    )
    assert agent._openai_prefers_responses_api() is True
    assert agent._openai_omits_temperature() is True


def test_bedrock_gpt_applies_openai_max_image_edge() -> None:
    # The --openai-max-image-edge cap is applied at reset() for bedrock-gpt too.
    pytest.importorskip("PIL")
    task = _png_image_task(3840, 2160)
    agent = train_and_eval.LLMAgent(
        provider="bedrock-gpt",
        model="openai.gpt-5.5",
        multimodal_input_mode="direct",
        openai_max_image_edge=1920,
    )
    agent.reset("inspect visible asset", task)
    assert _image_dims_from_parts(agent._messages[0]["content"]) == (1920, 1080)


def test_effective_multimodal_mode_enables_images_for_vertex_xai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force non-empty image parts so only the provider gate is under test.
    monkeypatch.setattr(
        train_and_eval.agents,
        "_multimodal_image_parts_from_task",
        lambda task, mode: [{"type": "image_url", "image_url": {"url": "x"}}],
    )
    assert (
        train_and_eval._effective_multimodal_input_mode(
            requested="both", provider="vertex-xai", task=None
        )
        == "both"
    )


def test_vertex_xai_uses_adc_bearer_token_no_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    creds = _FakeAdcCreds(token="tok-initial", valid=True)
    captured: dict = {}
    _install_fake_google_auth(monkeypatch, creds, captured)
    # No OPENAI_API_KEY: the vertex-xai path authenticates via ADC only.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    agent = train_and_eval.LLMAgent(
        provider="vertex-xai",
        model="grok-4.20-reasoning",
        vertex_project="p",
        vertex_location="global",
    )
    assert agent._bearer_token() == "tok-initial"
    assert captured["scopes"] == [
        "https://www.googleapis.com/auth/cloud-platform"
    ]
    assert creds.refresh_calls == 0
    # Credentials are cached: a second call does not re-run google.auth.default().
    assert agent._bearer_token() == "tok-initial"
    assert captured["default_calls"] == 1


def test_vertex_xai_refreshes_expired_adc_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    creds = _FakeAdcCreds(token="stale", valid=False)
    captured: dict = {}
    _install_fake_google_auth(monkeypatch, creds, captured)

    agent = train_and_eval.LLMAgent(
        provider="vertex-xai",
        model="grok-4.20-reasoning",
        vertex_project="p",
        vertex_location="global",
    )
    # An expired token is refreshed before use.
    assert agent._vertex_access_token() == "fresh"
    assert creds.refresh_calls == 1


def test_act_routes_vertex_xai_to_openai_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = train_and_eval.LLMAgent(
        provider="vertex-xai",
        model="grok-4.20-reasoning",
        vertex_project="p",
        vertex_location="global",
    )
    calls: list[str] = []

    def _anthropic():
        calls.append("anthropic")
        return "x"

    def _openai():
        calls.append("openai")
        return "file list"

    def _gemini():
        calls.append("gemini")
        return "x"

    monkeypatch.setattr(agent, "_anthropic_response_text", _anthropic)
    monkeypatch.setattr(agent, "_openai_response_text", _openai)
    monkeypatch.setattr(agent, "_gemini_response_text", _gemini)
    monkeypatch.setattr(
        train_and_eval.agents, "_validate_command", lambda action, task: (True, "")
    )

    task = train_and_eval.load_task(
        "hard_decision_workflow_1",
        data_dir=Path(train_and_eval.DEFAULT_TASK_DATA_DIR),
    )
    agent.reset("inspect the workspace", task=task)
    obs = train_and_eval.Observation(
        command_output="",
        error_output="",
        exit_code=0,
        current_config={},
        gateway_status="",
        available_commands=[],
        task_instruction="inspect the workspace",
        step_number=0,
    )
    result = agent.act(obs)
    assert calls == ["openai"]
    assert result == "file list"


def test_is_anthropic_temperature_unsupported_detects_deprecation() -> None:
    class _Err(Exception):
        pass

    assert train_and_eval._is_anthropic_temperature_unsupported(
        _Err("Error code: 400 - `temperature` is deprecated for this model.")
    )
    # Message carried on a `.message` attribute is also inspected.
    e = _Err("boom")
    e.message = "temperature is not supported"  # type: ignore[attr-defined]
    assert train_and_eval._is_anthropic_temperature_unsupported(e)
    # Unrelated errors are not misclassified.
    assert not train_and_eval._is_anthropic_temperature_unsupported(
        _Err("model not found")
    )
    assert not train_and_eval._is_anthropic_temperature_unsupported(
        _Err("temperature must be between 0 and 1")
    )


def test_vertex_anthropic_drops_temperature_on_deprecation() -> None:
    model = "claude-omit-temp@20990101"
    # Ensure a clean slate for the process-wide cache.
    train_and_eval._ANTHROPIC_MODELS_OMIT_TEMPERATURE.discard(model)

    attempts: list[dict[str, object]] = []

    class _Txt:
        def __init__(self, t):
            self.text = t

    class _Resp:
        def __init__(self, t):
            self.content = [_Txt(t)]

    class _Msgs:
        def create(self, **kwargs):
            attempts.append(kwargs)
            if _has_temp(kwargs):
                raise ValueError(
                    "Error code: 400 - `temperature` is deprecated for this model."
                )
            return _Resp("file list ")

    class _Client:
        messages = _Msgs()

    agent = train_and_eval.LLMAgent(
        provider="vertex-anthropic",
        model=model,
        vertex_project="p",
        vertex_location="us-east5",
    )
    agent._client = _Client()
    agent._messages = [{"role": "user", "content": "hi"}]

    out = agent._anthropic_response_text()
    assert out == "file list"
    # First attempt included temperature and failed; second omitted it.
    assert len(attempts) == 2
    assert _has_temp(attempts[0])
    assert not _has_temp(attempts[1])
    # The model is remembered so later calls skip temperature immediately.
    assert model in train_and_eval._ANTHROPIC_MODELS_OMIT_TEMPERATURE
    attempts.clear()
    assert agent._anthropic_response_text() == "file list"
    assert len(attempts) == 1
    assert not _has_temp(attempts[0])

    # Cleanup so the cache does not leak into other tests.
    train_and_eval._ANTHROPIC_MODELS_OMIT_TEMPERATURE.discard(model)


def test_anthropic_text_from_response_skips_thinking_blocks() -> None:
    class _Block:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    # Reasoning models put a ThinkingBlock (no `.text`) before the TextBlock.
    class _Thinking:
        type = "thinking"
        thinking = "let me reason..."

    resp = type(
        "R",
        (),
        {"content": [_Thinking(), _Block(type="text", text="ls -la")]},
    )()
    assert train_and_eval._anthropic_text_from_response(resp) == "ls -la"

    # Multiple text blocks are concatenated; non-text blocks ignored.
    resp2 = type(
        "R",
        (),
        {
            "content": [
                _Block(type="text", text="foo "),
                _Block(type="tool_use", name="x"),
                _Block(type="text", text="bar"),
            ]
        },
    )()
    assert train_and_eval._anthropic_text_from_response(resp2) == "foo bar"

    # Dict-shaped blocks are also supported.
    resp3 = {"content": [{"type": "thinking"}, {"type": "text", "text": " hi "}]}
    assert train_and_eval._anthropic_text_from_response(resp3) == "hi"

    # A fake block exposing only `.text` (no `.type`) is treated as text.
    resp4 = type("R", (), {"content": [_Block(text="plain")]})()
    assert train_and_eval._anthropic_text_from_response(resp4) == "plain"

    # No text blocks -> empty string (graceful, not a crash).
    resp5 = type("R", (), {"content": [_Thinking()]})()
    assert train_and_eval._anthropic_text_from_response(resp5) == ""


# ---------------------------------------------------------------------------
# Anthropic adaptive image-downscaling (auto max-fidelity + learned cap)
# ---------------------------------------------------------------------------


def _png_b64(width: int, height: int) -> str:
    """Return base64 PNG bytes of a solid-color image of the given size."""
    import base64 as _b64
    import io as _io

    from PIL import Image

    img = Image.new("RGB", (width, height), (10, 20, 30))
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    return _b64.b64encode(buf.getvalue()).decode("ascii")


def _png_dims_from_b64(data: str) -> tuple[int, int]:
    import base64 as _b64
    import io as _io

    from PIL import Image

    with Image.open(_io.BytesIO(_b64.b64decode(data))) as im:
        return im.size


def test_normalize_anthropic_max_image_edge_handles_auto_and_ints() -> None:
    norm = train_and_eval._normalize_anthropic_max_image_edge
    # 'auto'/'max'/empty/None/unparseable/non-positive all mean auto.
    assert norm(None) == "auto"
    assert norm("auto") == "auto"
    assert norm("  AUTO ") == "auto"
    assert norm("max") == "auto"
    assert norm("") == "auto"
    assert norm("abc") == "auto"
    assert norm("0") == "auto"
    assert norm("-5") == "auto"
    assert norm(0) == "auto"
    assert norm(-3) == "auto"
    assert norm(True) == "auto"  # bool guarded (subclass of int)
    # Positive integers (as string or int) force a fixed cap.
    assert norm("1568") == 1568
    assert norm("  2000 ") == 2000
    assert norm(1568) == 1568


def test_is_anthropic_image_dimension_error_detects_oversized() -> None:
    detect = train_and_eval._is_anthropic_image_dimension_error

    class _Err(Exception):
        pass

    assert detect(
        _Err(
            "Error code: 400 - image dimensions 4000 pixels exceed the maximum"
            " allowed size of 1568 pixels for many-image requests."
        )
    )
    # Message on a `.message` attribute is inspected too.
    e = _Err("boom")
    e.message = "image exceeds max allowed size of 2576 pixels"  # type: ignore[attr-defined]
    assert detect(e)
    # Unrelated errors are not misclassified.
    assert not detect(_Err("`temperature` is deprecated for this model"))
    assert not detect(_Err("image not found"))  # no 'pixel'
    assert not detect(_Err("request exceeds 500 pixels"))  # no 'image'


def test_parse_anthropic_image_pixel_limit_picks_min_reasonable() -> None:
    parse = train_and_eval._parse_anthropic_image_pixel_limit

    class _Err(Exception):
        pass

    # Both the offending dimension and the cap are named; the cap is smaller.
    assert (
        parse(
            _Err(
                "image dimensions 4000 pixels exceed maximum 1568 pixels"
            )
        )
        == 1568
    )
    assert parse(_Err("max allowed size 2576 pixels")) == 2576
    # No parseable pixel count -> None (caller falls back).
    assert parse(_Err("image too large")) is None
    # Absurdly small numbers are ignored (below the reasonable floor).
    assert parse(_Err("uses 5 pixels")) is None


def test_current_anthropic_image_edge_combines_fixed_and_learned() -> None:
    model = "claude-edge-test@20990101"
    train_and_eval._ANTHROPIC_MODEL_IMAGE_EDGE.pop(model, None)
    try:
        agent = train_and_eval.LLMAgent(
            provider="vertex-anthropic",
            model=model,
            vertex_project="p",
            vertex_location="us-east5",
        )
        # auto + nothing learned -> uncapped (max fidelity).
        assert agent._anthropic_max_image_edge == "auto"
        assert agent._current_anthropic_image_edge() is None
        # auto + learned cap -> learned cap.
        train_and_eval._ANTHROPIC_MODEL_IMAGE_EDGE[model] = 1568
        assert agent._current_anthropic_image_edge() == 1568
        # Fixed cap takes the min with any learned cap.
        agent._anthropic_max_image_edge = 2000
        assert agent._current_anthropic_image_edge() == 1568
        agent._anthropic_max_image_edge = 1000
        assert agent._current_anthropic_image_edge() == 1000
        # Fixed cap with nothing learned -> the fixed value.
        train_and_eval._ANTHROPIC_MODEL_IMAGE_EDGE.pop(model, None)
        assert agent._current_anthropic_image_edge() == 1000
    finally:
        train_and_eval._ANTHROPIC_MODEL_IMAGE_EDGE.pop(model, None)


def test_downscale_image_b64_shrinks_memoizes_and_is_safe() -> None:
    pytest.importorskip("PIL")
    train_and_eval._DOWNSCALED_IMAGE_CACHE.clear()

    big = _png_b64(4000, 3000)
    out = train_and_eval._downscale_image_b64(big, "image/png", 2000)
    w, h = _png_dims_from_b64(out)
    assert max(w, h) == 2000
    assert (w, h) == (2000, 1500)  # aspect ratio preserved

    # Memoized: same inputs return the identical cached string.
    out2 = train_and_eval._downscale_image_b64(big, "image/png", 2000)
    assert out2 == out
    key = (
        train_and_eval.hashlib.sha1(big.encode("utf-8")).hexdigest(),
        2000,
    )
    assert key in train_and_eval._DOWNSCALED_IMAGE_CACHE

    # An image already within the cap is returned unchanged.
    small = _png_b64(100, 80)
    assert train_and_eval._downscale_image_b64(small, "image/png", 2000) == small

    # Undecodable data never raises - it is passed through untouched.
    assert (
        train_and_eval._downscale_image_b64("not-base64!!!", "image/png", 100)
        == "not-base64!!!"
    )


def test_to_anthropic_messages_downscales_only_when_edge_set() -> None:
    pytest.importorskip("PIL")
    big = _png_b64(3000, 3000)
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "see chart"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{big}"},
                },
            ],
        }
    ]

    # Without a cap, the original base64 is passed through unchanged.
    passthrough = train_and_eval._to_anthropic_messages(msgs)
    img = [p for p in passthrough[0]["content"] if p["type"] == "image"][0]
    assert img["source"]["data"] == big

    # With a cap, the inline image is pre-shrunk to <= max_image_edge.
    shrunk = train_and_eval._to_anthropic_messages(msgs, max_image_edge=512)
    img2 = [p for p in shrunk[0]["content"] if p["type"] == "image"][0]
    w, h = _png_dims_from_b64(img2["source"]["data"])
    assert max(w, h) <= 512


def _png_b64_noise(width: int, height: int) -> str:
    """Base64 PNG of RANDOM noise (does not compress away), so downscaling it
    meaningfully reduces the encoded byte count -- unlike a solid color."""
    import base64 as _b64
    import io as _io

    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(width * 100003 + height)
    arr = rng.integers(0, 256, size=(height, width, 3), dtype="uint8")
    buf = _io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return _b64.b64encode(buf.getvalue()).decode("ascii")


def test_estimate_anthropic_request_bytes_sums_payload() -> None:
    estimate = train_and_eval._estimate_anthropic_request_bytes
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "abcde"},  # 5
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png",
                               "data": "x" * 1000},  # 1000
                },
                {
                    "type": "image",
                    "source": {"type": "url", "url": "y" * 40},  # 40
                },
            ],
        },
        {"role": "assistant", "content": "hello"},  # 5 (string content)
    ]
    assert estimate(msgs) == 5 + 1000 + 40 + 5


def test_estimate_anthropic_request_bytes_counts_tool_result_text() -> None:
    # Bug 2 follow-on: native tool-result observations carry their payload in
    # ``content`` (a string, or occasionally a list of text blocks). The
    # estimator must count it so tool-heavy native histories are not understated
    # against the byte budget (which would let an oversized request through).
    estimate = train_and_eval._estimate_anthropic_request_bytes
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "c1",
                 "content": "x" * 30},  # 30
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "c2",
                    "content": [
                        {"type": "text", "text": "y" * 12},  # 12
                        {"type": "text", "text": "z" * 8},  # 8
                    ],
                },
            ],
        },
    ]
    assert estimate(msgs) == 30 + 12 + 8


def test_fit_anthropic_messages_to_byte_budget_shrinks_many_image_round() -> None:
    """Bug 2 prevention: a heavy multi-image request is proactively downscaled
    under the byte budget BEFORE it is sent (the 30MB rejection never fires)."""
    pytest.importorskip("PIL")
    pytest.importorskip("numpy")
    train_and_eval._DOWNSCALED_IMAGE_CACHE.clear()

    fit = train_and_eval._fit_anthropic_messages_to_byte_budget
    estimate = train_and_eval._estimate_anthropic_request_bytes

    # A conflict-VLM-like round: several large 4K-ish noise images accumulated.
    parts = [{"type": "text", "text": "reconcile the panels " * 20}]
    for _ in range(6):
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{_png_b64_noise(2600, 2000)}"},
        })
    base_messages = [{"role": "user", "content": parts}]
    ladder = train_and_eval._ANTHROPIC_BYTE_BUDGET_EDGE_LADDER

    # Full-fidelity (originals) baseline, and a budget that forces shrinking.
    full = train_and_eval._to_anthropic_messages(base_messages, max_image_edge=None)
    full_bytes = estimate(full)
    budget = int(full_bytes * 0.4)

    fitted, edge_used = fit(base_messages, None, budget, ladder)
    # It actually fits now, and it climbed DOWN the ladder to do so.
    assert estimate(fitted) <= budget
    assert edge_used in ladder
    assert edge_used < 2600

    # A generous budget keeps maximum fidelity (originals, no cap learned).
    fitted_big, edge_big = fit(base_messages, None, full_bytes + 10, ladder)
    assert edge_big is None
    assert estimate(fitted_big) <= full_bytes + 10


def test_fit_anthropic_messages_respects_existing_edge_cap() -> None:
    """When a per-model cap is already learned, the byte-budget fit starts there
    (never UP-samples above it) and only descends further if still too big."""
    pytest.importorskip("PIL")
    pytest.importorskip("numpy")
    train_and_eval._DOWNSCALED_IMAGE_CACHE.clear()

    fit = train_and_eval._fit_anthropic_messages_to_byte_budget
    estimate = train_and_eval._estimate_anthropic_request_bytes

    parts = []
    for _ in range(4):
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{_png_b64_noise(3000, 3000)}"},
        })
    base_messages = [{"role": "user", "content": parts}]
    ladder = (2000, 1568, 1280, 1024, 768)

    # Huge budget but an existing 1024 cap -> stays at 1024 (does not jump to 2000).
    fitted, edge_used = fit(base_messages, 1024, 10**9, ladder)
    assert edge_used == 1024
    for part in fitted[0]["content"]:
        if part["type"] == "image":
            w, h = _png_dims_from_b64(part["source"]["data"])
            assert max(w, h) <= 1024


def test_vertex_anthropic_learns_image_cap_on_dimension_error() -> None:
    pytest.importorskip("PIL")
    model = "claude-img-cap@20990101"
    train_and_eval._ANTHROPIC_MODEL_IMAGE_EDGE.pop(model, None)
    train_and_eval._DOWNSCALED_IMAGE_CACHE.clear()

    attempts: list[int] = []

    class _Txt:
        def __init__(self, t):
            self.text = t

    class _Resp:
        def __init__(self, t):
            self.content = [_Txt(t)]

    class _Msgs:
        def create(self, **kwargs):
            content = kwargs["messages"][0]["content"]
            longest = 0
            for part in content:
                if part.get("type") == "image":
                    w, h = _png_dims_from_b64(part["source"]["data"])
                    longest = max(longest, w, h)
            attempts.append(longest)
            if longest > 1568:
                raise ValueError(
                    "Error code: 400 - image dimensions 4000 pixels exceed the"
                    " maximum allowed size of 1568 pixels for many-image requests."
                )
            return _Resp("file list ")

    class _Client:
        messages = _Msgs()

    try:
        agent = train_and_eval.LLMAgent(
            provider="vertex-anthropic",
            model=model,
            vertex_project="p",
            vertex_location="us-east5",
        )
        agent._client = _Client()
        big = _png_b64(4000, 3000)
        agent._messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "see chart"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{big}"},
                    },
                ],
            }
        ]

        out = agent._anthropic_response_text()
        assert out == "file list"
        # First attempt sent the original (4000px) and was rejected; the second
        # was pre-shrunk to the learned 1568px cap and succeeded.
        assert len(attempts) == 2
        assert attempts[0] == 4000
        assert attempts[1] <= 1568
        assert train_and_eval._ANTHROPIC_MODEL_IMAGE_EDGE[model] == 1568

        # A later response pre-shrinks from the start (single attempt).
        attempts.clear()
        assert agent._anthropic_response_text() == "file list"
        assert len(attempts) == 1
        assert attempts[0] <= 1568
    finally:
        train_and_eval._ANTHROPIC_MODEL_IMAGE_EDGE.pop(model, None)
        train_and_eval._DOWNSCALED_IMAGE_CACHE.clear()


def test_cli_accepts_anthropic_max_image_edge(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}

    _DummySummary = _make_dummy_summary(calls)

    def _fake_build(*_, **kwargs):
        calls["edge"] = kwargs.get("anthropic_max_image_edge")
        return object()

    monkeypatch.setattr(
        train_and_eval, "_preflight_mode", lambda mode: calls.setdefault("mode", mode)
    )
    monkeypatch.setattr(train_and_eval, "_build_agent", _fake_build)
    monkeypatch.setattr(
        train_and_eval,
        "run_evaluation",
        lambda *_, **__: _DummySummary(),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent",
            "expert",
            "--split",
            "dev",
            "--anthropic-max-image-edge",
            "1568",
            "--limit",
            "1",
        ],
    )
    train_and_eval.main()
    assert calls.get("edge") == "1568"
    assert calls.get("printed") is True


def test_cli_anthropic_max_image_edge_defaults_to_auto(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: dict[str, object] = {}

    _DummySummary = _make_dummy_summary(calls)

    def _fake_build(*_, **kwargs):
        calls["edge"] = kwargs.get("anthropic_max_image_edge")
        return object()

    monkeypatch.setattr(
        train_and_eval, "_preflight_mode", lambda mode: calls.setdefault("mode", mode)
    )
    monkeypatch.setattr(train_and_eval, "_build_agent", _fake_build)
    monkeypatch.setattr(
        train_and_eval,
        "run_evaluation",
        lambda *_, **__: _DummySummary(),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent",
            "expert",
            "--split",
            "dev",
            "--limit",
            "1",
        ],
    )
    train_and_eval.main()
    assert calls.get("edge") == "auto"


def test_cli_accepts_gemini_max_image_edge(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}

    _DummySummary = _make_dummy_summary(calls)

    def _fake_build(*_, **kwargs):
        calls["edge"] = kwargs.get("gemini_max_image_edge")
        return object()

    monkeypatch.setattr(
        train_and_eval, "_preflight_mode", lambda mode: calls.setdefault("mode", mode)
    )
    monkeypatch.setattr(train_and_eval, "_build_agent", _fake_build)
    monkeypatch.setattr(
        train_and_eval,
        "run_evaluation",
        lambda *_, **__: _DummySummary(),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent",
            "expert",
            "--split",
            "dev",
            "--gemini-max-image-edge",
            "1568",
            "--limit",
            "1",
        ],
    )
    train_and_eval.main()
    # --gemini-max-image-edge is type=int, so it parses to an int cap.
    assert calls.get("edge") == 1568
    assert calls.get("printed") is True


def test_cli_gemini_max_image_edge_defaults_to_off(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: dict[str, object] = {}

    _DummySummary = _make_dummy_summary(calls)

    def _fake_build(*_, **kwargs):
        calls["edge"] = kwargs.get("gemini_max_image_edge")
        calls["edge_present"] = "gemini_max_image_edge" in kwargs
        return object()

    monkeypatch.setattr(
        train_and_eval, "_preflight_mode", lambda mode: calls.setdefault("mode", mode)
    )
    monkeypatch.setattr(train_and_eval, "_build_agent", _fake_build)
    monkeypatch.setattr(
        train_and_eval,
        "run_evaluation",
        lambda *_, **__: _DummySummary(),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent",
            "expert",
            "--split",
            "dev",
            "--limit",
            "1",
        ],
    )
    train_and_eval.main()
    # Off by default: the flag is threaded through but resolves to None.
    assert calls.get("edge_present") is True
    assert calls.get("edge") is None


def test_cli_accepts_gemini_media_resolution(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}

    _DummySummary = _make_dummy_summary(calls)

    def _fake_build(*_, **kwargs):
        calls["res"] = kwargs.get("gemini_media_resolution")
        calls["res_present"] = "gemini_media_resolution" in kwargs
        return object()

    monkeypatch.setattr(
        train_and_eval, "_preflight_mode", lambda mode: calls.setdefault("mode", mode)
    )
    monkeypatch.setattr(train_and_eval, "_build_agent", _fake_build)
    monkeypatch.setattr(
        train_and_eval,
        "run_evaluation",
        lambda *_, **__: _DummySummary(),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent",
            "expert",
            "--split",
            "dev",
            "--gemini-media-resolution",
            "ultra_high",
            "--limit",
            "1",
        ],
    )
    train_and_eval.main()
    assert calls.get("res") == "ultra_high"
    assert calls.get("printed") is True


def test_cli_gemini_media_resolution_defaults_to_unset(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: dict[str, object] = {}

    _DummySummary = _make_dummy_summary(calls)

    def _fake_build(*_, **kwargs):
        calls["res"] = kwargs.get("gemini_media_resolution")
        calls["res_present"] = "gemini_media_resolution" in kwargs
        return object()

    monkeypatch.setattr(
        train_and_eval, "_preflight_mode", lambda mode: calls.setdefault("mode", mode)
    )
    monkeypatch.setattr(train_and_eval, "_build_agent", _fake_build)
    monkeypatch.setattr(
        train_and_eval,
        "run_evaluation",
        lambda *_, **__: _DummySummary(),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent",
            "expert",
            "--split",
            "dev",
            "--limit",
            "1",
        ],
    )
    train_and_eval.main()
    # Unset by default so Gemini keeps its own server-side budget; the flag is
    # still threaded through so the wiring cannot silently rot.
    assert calls.get("res_present") is True
    assert calls.get("res") is None


def test_cli_capture_log_enables_capture_and_defaults_log_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # --capture-log is the single master switch: it turns on BOTH thought
    # capture and the interaction sidecar, and the sidecar defaults to
    # '<out-dir>/log'.
    calls: dict[str, object] = {}
    _DummySummary = _make_dummy_summary(calls)

    def _fake_build(*_, **kwargs):
        calls["build_agent_kwargs"] = kwargs
        return object()

    def _fake_run_eval(*_, **kwargs):
        calls["interaction_log"] = kwargs.get("interaction_log")
        return _DummySummary()

    monkeypatch.setattr(
        train_and_eval, "_preflight_mode", lambda mode: calls.setdefault("mode", mode)
    )
    monkeypatch.setattr(train_and_eval, "_build_agent", _fake_build)
    monkeypatch.setattr(train_and_eval, "run_evaluation", _fake_run_eval)

    out_dir = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent", "llm",
            "--model", "gemini-3.5-flash",
            "--llm-provider", "gemini",
            "--split", "dev",
            "--out-dir", str(out_dir),
            "--capture-log",
            "--limit", "1",
        ],
    )
    train_and_eval.main()

    kwargs = calls["build_agent_kwargs"]
    assert kwargs["capture_thinking"] is True
    assert kwargs["capture_interactions"] is True
    # Default log directory is '<out-dir>/log'.
    assert calls["interaction_log"] == str(out_dir / "log")


def test_cli_capture_log_off_by_default_disables_interaction_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # Without --capture-log the run stays byte-for-byte unchanged: no capture,
    # and interaction_log is threaded through as None.
    calls: dict[str, object] = {}
    _DummySummary = _make_dummy_summary(calls)

    def _fake_build(*_, **kwargs):
        calls["build_agent_kwargs"] = kwargs
        return object()

    def _fake_run_eval(*_, **kwargs):
        calls["interaction_log"] = kwargs.get("interaction_log")
        calls["interaction_log_present"] = "interaction_log" in kwargs
        return _DummySummary()

    monkeypatch.setattr(
        train_and_eval, "_preflight_mode", lambda mode: calls.setdefault("mode", mode)
    )
    monkeypatch.setattr(train_and_eval, "_build_agent", _fake_build)
    monkeypatch.setattr(train_and_eval, "run_evaluation", _fake_run_eval)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent", "llm",
            "--model", "gemini-3.5-flash",
            "--llm-provider", "gemini",
            "--split", "dev",
            "--out-dir", str(tmp_path / "results"),
            "--limit", "1",
        ],
    )
    train_and_eval.main()

    kwargs = calls["build_agent_kwargs"]
    assert kwargs["capture_thinking"] is False
    assert kwargs["capture_interactions"] is False
    assert calls["interaction_log_present"] is True
    assert calls["interaction_log"] is None


def test_cli_interaction_log_overrides_default_log_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # --interaction-log overrides the default '<out-dir>/log' directory.
    calls: dict[str, object] = {}
    _DummySummary = _make_dummy_summary(calls)

    def _fake_run_eval(*_, **kwargs):
        calls["interaction_log"] = kwargs.get("interaction_log")
        return _DummySummary()

    monkeypatch.setattr(
        train_and_eval, "_preflight_mode", lambda mode: calls.setdefault("mode", mode)
    )
    monkeypatch.setattr(train_and_eval, "_build_agent", lambda *_, **__: object())
    monkeypatch.setattr(train_and_eval, "run_evaluation", _fake_run_eval)

    custom = tmp_path / "my_logs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent", "llm",
            "--model", "gemini-3.5-flash",
            "--llm-provider", "gemini",
            "--split", "dev",
            "--out-dir", str(tmp_path / "results"),
            "--capture-log",
            "--interaction-log", str(custom),
            "--limit", "1",
        ],
    )
    train_and_eval.main()
    assert calls["interaction_log"] == str(custom)


def test_cli_interaction_log_without_capture_log_warns_and_disables(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
):
    # --interaction-log without --capture-log is a no-op: warn and disable.
    calls: dict[str, object] = {}
    _DummySummary = _make_dummy_summary(calls)

    def _fake_run_eval(*_, **kwargs):
        calls["interaction_log"] = kwargs.get("interaction_log")
        return _DummySummary()

    monkeypatch.setattr(
        train_and_eval, "_preflight_mode", lambda mode: calls.setdefault("mode", mode)
    )
    monkeypatch.setattr(train_and_eval, "_build_agent", lambda *_, **__: object())
    monkeypatch.setattr(train_and_eval, "run_evaluation", _fake_run_eval)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent", "llm",
            "--model", "gemini-3.5-flash",
            "--llm-provider", "gemini",
            "--split", "dev",
            "--out-dir", str(tmp_path / "results"),
            "--interaction-log", str(tmp_path / "my_logs"),
            "--limit", "1",
        ],
    )
    train_and_eval.main()
    assert calls["interaction_log"] is None
    out = capsys.readouterr().out
    assert "no effect without --capture-log" in out


def test_cli_collect_data_capture_log_passes_default_log_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # collect-data mode threads the resolved '<out-dir>/log' into
    # collect_training_data (this is the path the real flash command uses).
    calls: dict[str, object] = {}

    def _fake_collect(*_, **kwargs):
        calls["interaction_log"] = kwargs.get("interaction_log")
        calls["parallel_workers"] = kwargs.get("parallel_workers")
        return tmp_path / "results" / "llm_dev.jsonl"

    monkeypatch.setattr(
        train_and_eval, "_preflight_mode", lambda mode: calls.setdefault("mode", mode)
    )
    monkeypatch.setattr(train_and_eval, "_build_agent", lambda *_, **__: object())
    monkeypatch.setattr(train_and_eval, "collect_training_data", _fake_collect)

    out_dir = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent", "llm",
            "--model", "gemini-3.5-flash",
            "--llm-provider", "gemini",
            "--split", "dev",
            "--collect-data",
            "--parallel-workers", "10",
            "--out-dir", str(out_dir),
            "--capture-log",
            "--limit", "1",
        ],
    )
    train_and_eval.main()
    assert calls["interaction_log"] == str(out_dir / "log")
    assert calls["parallel_workers"] == 10


def test_llm_agent_gemini_max_image_edge_normalizes_to_none_or_positive_int() -> None:
    # None / non-positive / bool -> disabled (None); positive int -> fixed cap.
    off = train_and_eval.LLMAgent(
        provider="gemini", model="gemini-3.5-flash", gemini_max_image_edge=None
    )
    assert off._gemini_max_image_edge is None
    zero = train_and_eval.LLMAgent(
        provider="gemini", model="gemini-3.5-flash", gemini_max_image_edge=0
    )
    assert zero._gemini_max_image_edge is None
    capped = train_and_eval.LLMAgent(
        provider="gemini", model="gemini-3.5-flash", gemini_max_image_edge=1568
    )
    assert capped._gemini_max_image_edge == 1568


# --- Gemini media_resolution (per-image visual token budget) -----------------
#
# This is the only knob that changes how much detail Gemini actually sees:
# Gemini resamples every image to a fixed server-side token budget, so
# --gemini-max-image-edge is a no-op for model-visible fidelity (measured on
# eval_results/data_3_5_flash: 8.29 MP and 1.08 MP both cost ~1100 tokens).

_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAM"
    "BAQDJ/pLvAAAAAElFTkSuQmCC"
)


def _gemini_image_message() -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "look at this"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_TINY_PNG_B64}"},
            },
        ],
    }


def test_normalize_gemini_media_resolution_accepts_levels_and_blank() -> None:
    norm = train_and_eval.agents._normalize_gemini_media_resolution
    # Unset spellings all mean "leave the request untouched".
    assert norm(None) is None
    assert norm("") is None
    assert norm("  ") is None
    assert norm("unspecified") is None
    # Case/whitespace tolerant.
    assert norm("ULTRA_HIGH") == "ultra_high"
    assert norm(" high ") == "high"
    for level in ("low", "medium", "high", "ultra_high"):
        assert norm(level) == level


def test_normalize_gemini_media_resolution_rejects_unknown_level() -> None:
    # A typo must fail loudly: silently falling back to the default budget is
    # exactly the silent-misconfiguration this flag exists to prevent.
    with pytest.raises(ValueError, match="Unsupported gemini media resolution"):
        train_and_eval.agents._normalize_gemini_media_resolution("ultrahigh")
    with pytest.raises(ValueError):
        train_and_eval.agents._normalize_gemini_media_resolution("2240")


def test_gemini_image_part_carries_ultra_high_media_resolution() -> None:
    from google.genai import types

    agent = train_and_eval.LLMAgent(
        provider="gemini",
        model="gemini-3.5-flash",
        gemini_media_resolution="ultra_high",
    )
    assert agent._gemini_media_resolution == "ultra_high"

    content = agent._gemini_content_from_message(types, _gemini_image_message())
    text_part, image_part = content.parts
    # Only the media part is annotated; text parts stay untouched.
    assert text_part.media_resolution is None
    level = image_part.media_resolution.level
    assert getattr(level, "name", level) == "MEDIA_RESOLUTION_ULTRA_HIGH"


def test_gemini_image_part_unset_media_resolution_leaves_request_unchanged() -> None:
    from google.genai import types

    agent = train_and_eval.LLMAgent(
        provider="gemini", model="gemini-3.5-flash"
    )
    assert agent._gemini_media_resolution is None

    content = agent._gemini_content_from_message(types, _gemini_image_message())
    _, image_part = content.parts
    assert image_part.media_resolution is None


def test_run_config_records_image_settings_per_provider() -> None:
    """A finished run must state the image size it used.

    The three caps have asymmetric defaults (anthropic "auto"=uncapped, gemini
    None=off, openai 1920), so an unset flag does NOT mean "unchanged" -- which
    is how an OpenAI run silently ended up at 1920x1080 (2.07 MP) against
    Claude's 2576x1449 (3.73 MP). Recording the effective value makes that
    visible in the trajectory sidecar instead of leaving it to be
    reverse-engineered from token counts.
    """
    openai_agent = train_and_eval.LLMAgent(
        provider="openai", model="gpt-5.6-sol", openai_max_image_edge=2576
    )
    cfg = openai_agent.debug_run_config()
    assert cfg["max_image_edge"] == "2576"
    # Gemini-only key stays blank for other providers rather than absent, so
    # downstream readers can treat the schema as fixed.
    assert cfg["gemini_media_resolution"] == ""
    assert cfg["native_tool_choice"] == "auto"
    assert all(isinstance(v, str) for v in cfg.values())

    gemini_agent = train_and_eval.LLMAgent(
        provider="gemini",
        model="gemini-3.5-flash",
        gemini_max_image_edge=2576,
        gemini_media_resolution="ultra_high",
    )
    cfg = gemini_agent.debug_run_config()
    assert cfg["max_image_edge"] == "2576"
    assert cfg["gemini_media_resolution"] == "ultra_high"

    # Gemini with the cap off reports "" (originals sent), not "0"/"None".
    cfg = train_and_eval.LLMAgent(
        provider="gemini", model="gemini-3.5-flash"
    ).debug_run_config()
    assert cfg["max_image_edge"] == ""


def test_run_config_records_anthropic_auto_vs_fixed_image_edge() -> None:
    # Unique model name so a learned cap cached by another test cannot leak in.
    model = "claude-run-config-probe"
    auto_agent = train_and_eval.LLMAgent(
        provider="vertex-anthropic", model=model, anthropic_max_image_edge="auto"
    )
    # "auto" means originals were sent and the SERVER applied its own cap
    # (2576 long edge / 4784 visual tokens on Claude 4.7+), which is not a
    # client-side number we can report -- so say "auto", not "".
    assert auto_agent.debug_run_config()["max_image_edge"] == "auto"

    fixed_agent = train_and_eval.LLMAgent(
        provider="vertex-anthropic", model=model, anthropic_max_image_edge=2576
    )
    assert fixed_agent.debug_run_config()["max_image_edge"] == "2576"


def test_run_config_records_forced_tool_choice() -> None:
    agent = train_and_eval.LLMAgent(
        provider="openai", model="gpt-5.6-sol", tool_mode="native"
    )
    assert agent.debug_run_config()["native_tool_choice"] == "auto"
    agent._native_tool_choice = "forced"
    assert agent.debug_run_config()["native_tool_choice"] == "forced"


def test_gemini_part_media_resolution_degrades_on_old_sdk() -> None:
    # An SDK without the per-part enum must fall back to the server default
    # rather than crash a long run mid-flight.
    build = train_and_eval.agents._gemini_part_media_resolution
    assert build(SimpleNamespace(), "ultra_high") is None
    assert build(SimpleNamespace(PartMediaResolution=object), "ultra_high") is None
    # Unset short-circuits before touching the SDK at all.
    assert build(SimpleNamespace(), None) is None


def _png_image_task(width: int, height: int):
    """A minimal Task whose single round-visible asset is a PNG of the given size."""
    from clasharness.core.task import GroundTruth, Task, TaskData

    return Task(
        task_id="vlm_openai_edge_q01",
        instruction="inspect visible asset",
        domains=["asset"],
        difficulty=3,
        initial_state="default",
        data=TaskData(
            public={},
            private={
                "initial_state_overrides": {
                    "asset_manifest.json": {
                        "assets": [
                            {
                                "id": "dash",
                                "kind": "image",
                                "path": "dash.png",
                                "mime_type": "image/png",
                            }
                        ]
                    },
                    "assets/dash.png": {"__bytes_b64__": _png_b64(width, height)},
                }
            },
        ),
        ground_truth=GroundTruth(solution_commands=["DONE"], evaluation_checks=[]),
    )


def _image_dims_from_parts(parts: list) -> tuple[int, int]:
    """Decode the single image_url data URL in ``parts`` and return its dimensions."""
    urls = [
        p["image_url"]["url"]
        for p in parts
        if isinstance(p, dict) and p.get("type") == "image_url"
    ]
    assert len(urls) == 1
    b64 = urls[0].split(",", 1)[1]
    return _png_dims_from_b64(b64)


def test_openai_max_image_edge_downscales_parts_and_none_preserves() -> None:
    # gpt-5.x bills image tokens ~linearly with pixel area, so oversized inline
    # images are downscaled to the long-edge cap; None sends originals unchanged.
    pytest.importorskip("PIL")
    task = _png_image_task(3840, 2160)

    capped = train_and_eval._multimodal_image_parts_from_task(
        task, mode="direct", max_edge=1920
    )
    w, h = _image_dims_from_parts(capped)
    assert max(w, h) == 1920  # long edge capped to 1080p
    assert (w, h) == (1920, 1080)  # aspect ratio preserved

    original = train_and_eval._multimodal_image_parts_from_task(
        task, mode="direct", max_edge=None
    )
    assert _image_dims_from_parts(original) == (3840, 2160)  # untouched

    # Images already within the cap pass through unchanged (no upscaling).
    small = train_and_eval._multimodal_image_parts_from_task(
        _png_image_task(1408, 768), mode="direct", max_edge=1920
    )
    assert _image_dims_from_parts(small) == (1408, 768)


def test_llm_agent_openai_max_image_edge_normalizes_to_none_or_positive_int() -> None:
    # None / non-positive / bool -> disabled (None); positive int -> fixed cap.
    off = train_and_eval.LLMAgent(
        provider="openai", model="gpt-5.6-sol", openai_max_image_edge=None
    )
    assert off._openai_max_image_edge is None
    zero = train_and_eval.LLMAgent(
        provider="openai", model="gpt-5.6-sol", openai_max_image_edge=0
    )
    assert zero._openai_max_image_edge is None
    capped = train_and_eval.LLMAgent(
        provider="openai", model="gpt-5.6-sol", openai_max_image_edge=1920
    )
    assert capped._openai_max_image_edge == 1920


def test_openai_max_image_edge_applied_only_for_openai_providers() -> None:
    # The cap is applied at reset() for OpenAI-compatible providers only; other
    # providers (gemini) keep their own edge handling and are left untouched here.
    pytest.importorskip("PIL")
    task = _png_image_task(3840, 2160)

    openai_agent = train_and_eval.LLMAgent(
        provider="openai",
        model="gpt-5.6-sol",
        multimodal_input_mode="direct",
        openai_max_image_edge=1920,
    )
    openai_agent.reset("inspect visible asset", task)
    assert _image_dims_from_parts(openai_agent._messages[0]["content"]) == (1920, 1080)

    gemini_agent = train_and_eval.LLMAgent(
        provider="gemini",
        model="gemini-3.5-flash",
        multimodal_input_mode="direct",
        openai_max_image_edge=512,
    )
    gemini_agent.reset("inspect visible asset", task)
    # The openai cap must NOT touch a gemini episode's stored images.
    assert _image_dims_from_parts(gemini_agent._messages[0]["content"]) == (3840, 2160)


def test_cli_accepts_openai_max_image_edge(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}
    _DummySummary = _make_dummy_summary(calls)

    def _fake_build(*_, **kwargs):
        calls["edge"] = kwargs.get("openai_max_image_edge")
        return object()

    monkeypatch.setattr(
        train_and_eval, "_preflight_mode", lambda mode: calls.setdefault("mode", mode)
    )
    monkeypatch.setattr(train_and_eval, "_build_agent", _fake_build)
    monkeypatch.setattr(
        train_and_eval, "run_evaluation", lambda *_, **__: _DummySummary()
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent",
            "expert",
            "--split",
            "dev",
            "--openai-max-image-edge",
            "1568",
            "--limit",
            "1",
        ],
    )
    train_and_eval.main()
    assert calls.get("edge") == 1568
    assert calls.get("printed") is True


def test_cli_openai_max_image_edge_defaults_to_1080p(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: dict[str, object] = {}
    _DummySummary = _make_dummy_summary(calls)

    def _fake_build(*_, **kwargs):
        calls["edge"] = kwargs.get("openai_max_image_edge")
        calls["edge_present"] = "openai_max_image_edge" in kwargs
        return object()

    monkeypatch.setattr(
        train_and_eval, "_preflight_mode", lambda mode: calls.setdefault("mode", mode)
    )
    monkeypatch.setattr(train_and_eval, "_build_agent", _fake_build)
    monkeypatch.setattr(
        train_and_eval, "run_evaluation", lambda *_, **__: _DummySummary()
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_and_eval.py", "--agent", "expert", "--split", "dev", "--limit", "1"],
    )
    train_and_eval.main()
    # Default is 1080p (1920) so runs downscale 4K assets unless overridden.
    assert calls.get("edge_present") is True
    assert calls.get("edge") == 1920


def test_cli_bedrock_gpt_builds_with_explicit_model(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: dict[str, object] = {}
    _DummySummary = _make_dummy_summary(calls)

    def _fake_build(*args, **kwargs):
        calls["provider"] = kwargs.get("llm_provider")
        calls["model"] = args[1] if len(args) > 1 else kwargs.get("model")
        return object()

    monkeypatch.setattr(
        train_and_eval, "_preflight_mode", lambda mode: calls.setdefault("mode", mode)
    )
    monkeypatch.setattr(train_and_eval, "_build_agent", _fake_build)
    monkeypatch.setattr(
        train_and_eval, "run_evaluation", lambda *_, **__: _DummySummary()
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent",
            "llm",
            "--llm-provider",
            "bedrock-gpt",
            "--model",
            "openai.gpt-5.5",
            "--split",
            "dev",
            "--limit",
            "1",
        ],
    )
    train_and_eval.main()
    assert calls.get("provider") == "bedrock-gpt"
    assert calls.get("model") == "openai.gpt-5.5"


def test_cli_bedrock_gpt_requires_explicit_model(
    monkeypatch: pytest.MonkeyPatch,
):
    # Without an explicit --model, the claude default is nonsensical for
    # bedrock-gpt, so the run must fail fast with a clear error.
    monkeypatch.setattr(train_and_eval, "_preflight_mode", lambda mode: None)
    monkeypatch.setattr(train_and_eval, "_build_agent", lambda *_, **__: object())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent",
            "llm",
            "--llm-provider",
            "bedrock-gpt",
            "--split",
            "dev",
            "--limit",
            "1",
        ],
    )
    with pytest.raises(SystemExit):
        train_and_eval.main()


def _populate_out_dir(out_dir: Path) -> None:
    out_dir.mkdir()
    (out_dir / "llm_total.jsonl").write_text("stale\n", encoding="utf-8")
    (out_dir / "parallel_run.summary").write_text("old\n", encoding="utf-8")
    nested = out_dir / "dashboard_assets"
    nested.mkdir()
    (nested / "img.png").write_text("x", encoding="utf-8")


def test_clean_output_dir_removes_contents_but_keeps_dir(tmp_path: Path):
    out_dir = tmp_path / "eval_results"
    _populate_out_dir(out_dir)

    train_and_eval._clean_output_dir(out_dir)

    assert out_dir.exists()
    assert out_dir.is_dir()
    assert list(out_dir.iterdir()) == []


def test_clean_output_dir_takes_no_force_argument():
    """The confirmation flag is gone -- callers must not resurrect it.

    ``--clean-out-dir`` IS the request to delete; a second flag that every
    invocation ends up carrying is friction, not a guard. The guard that
    actually prevents the expensive failure lives in the resume mutual
    exclusion below.
    """
    import inspect

    params = inspect.signature(train_and_eval._clean_output_dir).parameters
    assert list(params) == ["out_path"]


def test_clean_output_dir_reports_what_it_deletes(tmp_path: Path, capsys):
    """Eval results are not in version control -- the wipe must leave a record.

    It no longer refuses, but it must still name the path and the size, so a
    mistyped ``--out-dir`` is answerable after the fact from the run log
    instead of being an unlogged, unrecoverable loss.
    """
    out_dir = tmp_path / "eval_results"
    _populate_out_dir(out_dir)

    train_and_eval._clean_output_dir(out_dir)

    message = capsys.readouterr().out
    assert str(out_dir) in message
    assert "3 file(s)" in message
    assert "MB" in message
    assert list(out_dir.iterdir()) == []


def test_clean_output_dir_stays_quiet_on_empty_dir(tmp_path: Path, capsys):
    """Nothing was lost, so nothing to account for -- and no spurious warning."""
    out_dir = tmp_path / "eval_results"
    out_dir.mkdir()

    train_and_eval._clean_output_dir(out_dir)

    assert out_dir.exists()
    assert list(out_dir.iterdir()) == []
    assert "removing" not in capsys.readouterr().out


def test_force_clean_out_dir_flag_no_longer_exists(monkeypatch, tmp_path: Path):
    """A leftover --force-clean-out-dir must fail loudly, not be silently ignored."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--collect-data",
            "--clean-out-dir",
            "--force-clean-out-dir",
            "--out-dir",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(SystemExit):
        train_and_eval.main()


@pytest.mark.parametrize(
    "resume_kwarg,flag_name",
    [
        ("retry_provider_failures_only", "--retry-provider-failures-only"),
        ("resume_trajectories", "--resume-trajectories"),
        ("resume_successes_only", "--resume-successes-only"),
    ],
)
def test_clean_out_dir_is_rejected_with_each_resume_flag(
    tmp_path: Path, resume_kwarg: str, flag_name: str
):
    """Combining them silently re-runs the whole split AND destroys the results.

    ``_clean_output_dir()`` wipes ``--out-dir`` before the resume logic reads it,
    so resume finds no prior records, concludes every task "never ran", and
    re-runs the entire split at full API cost -- against a directory whose
    contents it was supposed to resume from and has just deleted. The failure is
    invisible: the run looks like a normal fresh run.
    """
    out_dir = tmp_path / "eval_results"
    _populate_out_dir(out_dir)

    with pytest.raises(SystemExit) as excinfo:
        train_and_eval.collect_training_data(
            agent=SimpleNamespace(name="stub"),
            out_dir=out_dir,
            clean_out_dir=True,
            **{resume_kwarg: True},
        )

    message = str(excinfo.value)
    assert "--clean-out-dir cannot be combined with" in message
    assert flag_name in message
    # Must fail BEFORE deleting anything.
    assert (out_dir / "llm_total.jsonl").read_text(encoding="utf-8") == "stale\n"
    assert (out_dir / "dashboard_assets" / "img.png").exists()


def test_clean_out_dir_rejection_names_every_conflicting_flag(tmp_path: Path):
    """All offending flags are listed, not just the first one found."""
    out_dir = tmp_path / "eval_results"
    _populate_out_dir(out_dir)

    with pytest.raises(SystemExit) as excinfo:
        train_and_eval.collect_training_data(
            agent=SimpleNamespace(name="stub"),
            out_dir=out_dir,
            clean_out_dir=True,
            retry_provider_failures_only=True,
            resume_trajectories=True,
            resume_successes_only=True,
        )

    message = str(excinfo.value)
    for flag in (
        "--retry-provider-failures-only",
        "--resume-trajectories",
        "--resume-successes-only",
    ):
        assert flag in message


def test_clean_output_dir_is_noop_when_missing(tmp_path: Path):
    missing = tmp_path / "does_not_exist"

    # Should not raise and should not create the directory.
    train_and_eval._clean_output_dir(missing)

    assert not missing.exists()


def test_clean_output_dir_refuses_filesystem_root():
    with pytest.raises(ValueError, match="unsafe out-dir"):
        train_and_eval._clean_output_dir(Path("/"))


def test_clean_output_dir_rejects_non_directory_target(tmp_path: Path):
    target = tmp_path / "a_file"
    target.write_text("not a dir", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        train_and_eval._clean_output_dir(target)


def test_cli_collect_data_passes_clean_out_dir_flag(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}

    monkeypatch.setattr(train_and_eval, "_preflight_mode", lambda mode: None)
    monkeypatch.setattr(train_and_eval, "_build_agent", lambda *_, **__: object())
    monkeypatch.setattr(
        train_and_eval,
        "collect_training_data",
        lambda *_, **kwargs: calls.setdefault("clean_out_dir", kwargs["clean_out_dir"]),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent",
            "expert",
            "--split",
            "dev",
            "--collect-data",
            "--clean-out-dir",
            "--out-dir",
            "/tmp/clasharness-clean-test",
            "--limit",
            "1",
        ],
    )

    train_and_eval.main()

    assert calls.get("clean_out_dir") is True


def test_cli_collect_data_defaults_clean_out_dir_false(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}

    monkeypatch.setattr(train_and_eval, "_preflight_mode", lambda mode: None)
    monkeypatch.setattr(train_and_eval, "_build_agent", lambda *_, **__: object())
    monkeypatch.setattr(
        train_and_eval,
        "collect_training_data",
        lambda *_, **kwargs: calls.setdefault("clean_out_dir", kwargs["clean_out_dir"]),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_and_eval.py",
            "--agent",
            "expert",
            "--split",
            "dev",
            "--collect-data",
            "--limit",
            "1",
        ],
    )

    train_and_eval.main()

    assert calls.get("clean_out_dir") is False


# ---------------------------------------------------------------------------
# Empty-reply de-conflation, generation-stall bucketing, done-early split, and
# the conservative early-DONE gate (FP-1) fixes.
# ---------------------------------------------------------------------------


def _make_gemini_agent(monkeypatch: pytest.MonkeyPatch, client):
    _install_fake_genai(monkeypatch)
    agent = train_and_eval.LLMAgent(
        provider="gemini",
        model="gemini-3.5-flash",
        vertex_project="p",
        vertex_location="us-central1",
    )
    agent._client = client
    return agent


def _models_returning(*responses):
    """A fake gemini client whose generate_content yields responses in order.

    The final response repeats once the list is exhausted.
    """
    seq = list(responses)
    state = {"n": 0}

    class _Models:
        def generate_content(self, **kwargs):
            idx = min(state["n"], len(seq) - 1)
            state["n"] += 1
            return seq[idx]

    return SimpleNamespace(models=_Models())


def test_gemini_response_diagnostics_records_finish_reason_and_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resp = SimpleNamespace(
        text="file list --path /",
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))],
        usage_metadata=SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=5,
            total_token_count=105,
        ),
    )
    agent = _make_gemini_agent(monkeypatch, _models_returning(resp))
    agent._messages = [{"role": "user", "content": "hi"}]

    out = agent._gemini_response_text()

    assert out == "file list --path /"
    api_lines = [a for a in agent.debug_reply_attempts() if a.startswith("api:")]
    assert api_lines
    assert "finish_reason=STOP" in api_lines[-1]
    assert "usage=(prompt=100,candidates=5,total=105)" in api_lines[-1]


def test_gemini_extract_text_survives_blocked_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ``response.text`` raises when the sole candidate was safety-blocked; that
    # must degrade to an empty reply (handled by the retry loop), not a crash.
    class _Blocked:
        candidates = [SimpleNamespace(finish_reason=SimpleNamespace(name="SAFETY"))]
        usage_metadata = None

        @property
        def text(self):
            raise ValueError("blocked candidate has no text")

    agent = _make_gemini_agent(monkeypatch, _models_returning(_Blocked()))
    agent._messages = [{"role": "user", "content": "hi"}]

    out = agent._gemini_response_text()

    assert out == ""
    api_lines = [a for a in agent.debug_reply_attempts() if a.startswith("api:")]
    assert api_lines and "finish_reason=SAFETY" in api_lines[-1]


def test_gemini_terminal_empty_reply_sets_generation_stall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty = SimpleNamespace(
        text="",
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))],
        usage_metadata=None,
    )
    agent = _make_gemini_agent(monkeypatch, _models_returning(empty))
    agent.reset("investigate the issue", None)

    action = agent.act(train_and_eval.Observation(step_number=0))

    assert action == "DONE"
    assert agent.debug_stall_reason() == "generation_stall_empty_reply"


def test_gemini_empty_reply_recovered_is_not_a_stall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty = SimpleNamespace(text="", candidates=[], usage_metadata=None)
    recovered = SimpleNamespace(text="DONE", candidates=[], usage_metadata=None)
    agent = _make_gemini_agent(monkeypatch, _models_returning(empty, recovered))
    agent.reset("investigate the issue", None)

    action = agent.act(train_and_eval.Observation(step_number=0))

    assert action == "DONE"
    assert agent.debug_stall_reason() is None  # recovered by retry, not a stall
    assert any(
        a.startswith("api: content=''") for a in agent.debug_reply_attempts()
    )


def _done_client():
    return _models_returning(
        SimpleNamespace(text="DONE", candidates=[], usage_metadata=None)
    )


def test_early_done_gate_skipped_without_required_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # FP-1: a task whose instruction merely *mentions* a side-effect word
    # ("create"/"report") but has no required report/effect must NOT have its
    # DONE blocked just because the agent only inspected so far.
    task = SimpleNamespace(
        ground_truth=SimpleNamespace(evaluation_checks=[]),
        instruction="please create a short summary and report back",
    )
    agent = _make_gemini_agent(monkeypatch, _done_client())
    agent.reset(task.instruction, task)
    agent._history = [
        {"action": "file list --path /", "stdout": "", "stderr": "", "exit_code": 0}
    ]
    agent._pending_action = ""

    action = agent.act(train_and_eval.Observation(step_number=1))

    assert action == "DONE"
    assert agent.debug_stall_reason() is None


def test_early_done_gate_still_fires_with_required_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Guard: FP-1 must not neuter the gate when a real required report exists.
    # A persistent DONE gets rejected until the retry budget is exhausted, which
    # surfaces as an invalid-command stall (NOT a clean agent-chosen DONE).
    task = SimpleNamespace(
        ground_truth=SimpleNamespace(
            evaluation_checks=[
                {
                    "type": "effect",
                    "expected": {"field": "path", "value": "/reports/q01.md"},
                }
            ]
        ),
        instruction="create /reports/q01.md with the summary",
    )
    agent = _make_gemini_agent(monkeypatch, _done_client())
    agent.reset(task.instruction, task)
    agent._history = [
        {"action": "file list --path /", "stdout": "", "stderr": "", "exit_code": 0}
    ]
    agent._pending_action = ""

    action = agent.act(train_and_eval.Observation(step_number=1))

    assert action == "DONE"
    assert agent.debug_stall_reason() == "invalid_command_exhausted"


def _episode(stop_reason, trajectory=None, success=False):
    return train_and_eval.EpisodeResult(
        task_id="t",
        agent_name="a",
        success=success,
        score=0.0,
        steps=len(trajectory or []),
        duration_s=0.0,
        stop_reason=stop_reason,
        trajectory=trajectory or [],
    )


def _summary(results):
    return train_and_eval.EvalSummary(
        agent_name="a",
        split="s",
        total=len(results),
        passed=sum(1 for r in results if r.success),
        total_score=0.0,
        results=results,
    )


def test_done_early_split_separates_gave_up_and_acted_but_wrong() -> None:
    gave_up = _episode(
        "missing_required_side_effect: files_created",
        [{"executed_action": "file list --path /"}],
    )
    acted = _episode(
        "missing_required_report: /reports/q01.md",
        [{"executed_action": "file create --path /wrong/place.md --content x"}],
    )
    summary = _summary([gave_up, acted])

    assert summary.done_early_tasks == 2
    assert summary.gave_up_no_action_tasks == 1
    assert summary.acted_but_wrong_tasks == 1


def test_generation_stall_is_not_bucketed_as_done_early() -> None:
    stall = _episode(
        "generation_stall_empty_reply",
        [{"executed_action": "file list --path /"}],
    )
    summary = _summary([stall])

    assert summary.generation_stall_tasks == 1
    assert summary.done_early_tasks == 0
    assert summary.gave_up_no_action_tasks == 0
    assert summary.acted_but_wrong_tasks == 0


def test_recovered_empty_replies_counted_only_for_non_stalled_episodes() -> None:
    recovered = _episode(
        "missing_required_side_effect: files_created",
        [
            {
                "executed_action": "file list --path /",
                "model_debug": [
                    "api: content='' finish_reason=STOP",
                    "parsed: raw='' normalized=''",
                ],
            },
            {
                "executed_action": "file list --path /a",
                "model_debug": ["api: content='file list --path /a'"],
            },
        ],
    )
    stalled = _episode(
        "generation_stall_empty_reply",
        [
            {
                "executed_action": "file list --path /",
                "model_debug": ["api: content=''", "api: content=''"],
            }
        ],
    )
    summary = _summary([recovered, stalled])

    # The stalled episode's empties are the failure (counted by
    # generation_stall_tasks), not recovered telemetry.
    assert summary.recovered_empty_reply_events == 1
    assert summary.recovered_empty_reply_tasks == 1
    assert summary.generation_stall_tasks == 1


# ===========================================================================
# --llm-judge-provider flag + Vertex backfill + judge-config validation
# ===========================================================================


_JUDGE_ENV_VARS = (
    "LLM_JUDGE_PROVIDER",
    "LLM_JUDGE_MODEL",
    "LLM_JUDGE_REQUEST_RETRIES",
    "LLM_JUDGE_RETRY_BACKOFF_S",
    "VERTEX_PROJECT",
    "VERTEX_LOCATION",
    "GOOGLE_CLOUD_PROJECT",
    "GEMINI_PROJECT",
    "GOOGLE_CLOUD_REGION",
    "GEMINI_LOCATION",
    "ANTHROPIC_API_KEY",
    # gemini judge: resolve_provider auto-detects from GEMINI_API_KEY, so a real
    # key in the developer's shell would otherwise decide these tests.
    "GEMINI_API_KEY",
    # bedrock-gpt judge: resolve_provider auto-detects from BEDROCK_OPENAI_API_KEY
    # and the endpoint honours BEDROCK_OPENAI_BASE_URL, so both must be cleared to
    # keep the "unconfigured" tests hermetic against a real Bedrock key in the env.
    "BEDROCK_OPENAI_API_KEY",
    "BEDROCK_OPENAI_BASE_URL",
    # openai judge: same story, and OPENAI_API_KEY is exported in a great many
    # developer shells, so leaving it would silently resolve the "unconfigured"
    # cases to the openai provider.
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
)


def _clear_judge_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear (and register for restore) every env var the judge wiring reads."""
    for var in _JUDGE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _stub_main_run(monkeypatch: pytest.MonkeyPatch, calls: dict) -> None:
    """Neutralize the heavy parts of main() so only arg/env wiring runs."""
    _DummySummary = _make_dummy_summary(calls)
    monkeypatch.setattr(
        train_and_eval, "_preflight_mode", lambda mode: calls.setdefault("mode", mode)
    )
    monkeypatch.setattr(train_and_eval, "_build_agent", lambda *_, **__: object())
    monkeypatch.setattr(
        train_and_eval,
        "run_evaluation",
        lambda *_, **kwargs: _DummySummary(),
    )


def _judge_argv(*extra: str) -> list[str]:
    return [
        "train_and_eval.py",
        "--agent",
        "llm",
        "--split",
        "dev",
        "--mode",
        "mock",
        "--limit",
        "1",
        *extra,
    ]


def test_llm_judge_provider_flag_writes_env_and_backfills_project(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: dict[str, object] = {}
    _clear_judge_env(monkeypatch)
    _stub_main_run(monkeypatch, calls)
    monkeypatch.setattr(
        sys,
        "argv",
        _judge_argv(
            "--llm-provider",
            "vertex-anthropic",
            "--llm-judge-provider",
            "vertex-anthropic",
            "--llm-judge-model",
            "claude-sonnet-4@20250514",
            "--vertex-project",
            "proj-x",
            "--vertex-location",
            "us-east5",
        ),
    )

    train_and_eval.main()

    assert os.environ["LLM_JUDGE_PROVIDER"] == "vertex-anthropic"
    assert os.environ["LLM_JUDGE_MODEL"] == "claude-sonnet-4@20250514"
    assert os.environ["VERTEX_PROJECT"] == "proj-x"
    assert calls.get("printed") is True


def test_judge_gemini_backfills_vertex_project_and_location(
    monkeypatch: pytest.MonkeyPatch,
):
    # No --vertex-project given: the gemini judge must still pick the project
    # up from the shared fallback chain, so it never silently drops out of its
    # provider branch onto a different project than the agent.
    calls: dict[str, object] = {}
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "gcp-proj")
    _stub_main_run(monkeypatch, calls)
    monkeypatch.setattr(
        sys, "argv", _judge_argv("--llm-judge-provider", "gemini")
    )

    train_and_eval.main()

    assert os.environ["VERTEX_PROJECT"] == "gcp-proj"  # chain, not a guess
    assert os.environ["VERTEX_LOCATION"] == "us"  # gemini default


def test_judge_gemini_with_api_key_needs_no_vertex_project(
    monkeypatch: pytest.MonkeyPatch,
):
    """An API key is enough for the gemini judge; no project is invented.

    Setting VERTEX_PROJECT to "" would make the variable *present* and empty,
    which reads as "configured" downstream and would send the judge into the
    Vertex branch it has no credentials for.
    """
    calls: dict[str, object] = {}
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "key-abc")
    _stub_main_run(monkeypatch, calls)
    monkeypatch.setattr(
        sys, "argv", _judge_argv("--llm-judge-provider", "gemini")
    )

    train_and_eval.main()

    assert not os.environ.get("VERTEX_PROJECT")


def test_judge_gemini_errors_without_key_or_project(
    monkeypatch: pytest.MonkeyPatch,
):
    """Neither backend configured is a loud startup error, not silent ok=False."""
    calls: dict[str, object] = {}
    _clear_judge_env(monkeypatch)
    _stub_main_run(monkeypatch, calls)
    monkeypatch.setattr(
        sys, "argv", _judge_argv("--llm-judge-provider", "gemini")
    )

    with pytest.raises(SystemExit) as excinfo:
        train_and_eval.main()

    message = str(excinfo.value)
    assert "GEMINI_API_KEY" in message
    assert "VERTEX_PROJECT" in message


def test_judge_vertex_anthropic_missing_model_hard_errors(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: dict[str, object] = {}
    _clear_judge_env(monkeypatch)
    _stub_main_run(monkeypatch, calls)
    monkeypatch.setattr(
        sys,
        "argv",
        _judge_argv(
            "--llm-judge-provider",
            "vertex-anthropic",
            "--vertex-project",
            "proj-x",
        ),
    )

    with pytest.raises(SystemExit):
        train_and_eval.main()


def test_judge_anthropic_missing_api_key_hard_errors(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: dict[str, object] = {}
    _clear_judge_env(monkeypatch)
    _stub_main_run(monkeypatch, calls)
    monkeypatch.setattr(
        sys, "argv", _judge_argv("--llm-judge-provider", "anthropic")
    )

    with pytest.raises(SystemExit):
        train_and_eval.main()


def test_judge_openai_missing_api_key_hard_errors(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: dict[str, object] = {}
    _clear_judge_env(monkeypatch)
    _stub_main_run(monkeypatch, calls)
    monkeypatch.setattr(
        sys,
        "argv",
        _judge_argv(
            "--llm-judge-provider", "openai", "--llm-judge-model", "gpt-5.5"
        ),
    )

    with pytest.raises(SystemExit) as excinfo:
        train_and_eval.main()
    assert "OPENAI_API_KEY" in str(excinfo.value)


def test_judge_openai_missing_model_hard_errors(monkeypatch: pytest.MonkeyPatch):
    # Like bedrock-gpt, the openai judge has no built-in default model, so an
    # unspecified id would resolve to "" and 400 on every semantic check.
    calls: dict[str, object] = {}
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    _stub_main_run(monkeypatch, calls)
    monkeypatch.setattr(
        sys, "argv", _judge_argv("--llm-judge-provider", "openai")
    )

    with pytest.raises(SystemExit) as excinfo:
        train_and_eval.main()
    assert "--llm-judge-model" in str(excinfo.value)


def test_judge_openai_accepts_a_bare_gpt_id(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    _stub_main_run(monkeypatch, calls)
    monkeypatch.setattr(
        sys,
        "argv",
        _judge_argv(
            "--llm-judge-provider", "openai", "--llm-judge-model", "gpt-5.5"
        ),
    )

    train_and_eval.main()

    assert os.environ["LLM_JUDGE_PROVIDER"] == "openai"
    assert os.environ["LLM_JUDGE_MODEL"] == "gpt-5.5"
    # An official-API judge must not drag a Vertex project in behind it.
    assert not os.environ.get("VERTEX_PROJECT")
    assert calls.get("printed") is True


def test_judge_openai_warns_on_bedrock_publisher_prefix(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    # 'openai.gpt-5.5' is Bedrock's spelling. The judge strips the prefix rather
    # than 404ing, so this is a warning and the run proceeds.
    calls: dict[str, object] = {}
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    _stub_main_run(monkeypatch, calls)
    monkeypatch.setattr(
        sys,
        "argv",
        _judge_argv(
            "--llm-judge-provider",
            "openai",
            "--llm-judge-model",
            "openai.gpt-5.5",
        ),
    )

    train_and_eval.main()

    err = capsys.readouterr().err
    assert "openai." in err and "gpt-5.5" in err
    assert calls.get("printed") is True


def test_judge_vertex_anthropic_dash_model_is_accepted(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    # A plain dash id is now passed through verbatim (matching the agent), so it
    # must NOT trigger the old '@'/404 warning and must be pinned as the judge
    # model.
    calls: dict[str, object] = {}
    _clear_judge_env(monkeypatch)
    _stub_main_run(monkeypatch, calls)
    monkeypatch.setattr(
        sys,
        "argv",
        _judge_argv(
            "--llm-judge-provider",
            "vertex-anthropic",
            "--llm-judge-model",
            "claude-opus-4-8",  # dash form -> passed through, no warning
            "--vertex-project",
            "proj-x",
            "--vertex-location",
            "us-east5",
        ),
    )

    train_and_eval.main()  # neither warns about '@' nor exits

    err = capsys.readouterr().err
    assert "404" not in err
    assert "'@'" not in err
    assert os.environ["LLM_JUDGE_PROVIDER"] == "vertex-anthropic"
    assert os.environ["LLM_JUDGE_MODEL"] == "claude-opus-4-8"


def test_judge_vertex_anthropic_us_location_warns(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    calls: dict[str, object] = {}
    _clear_judge_env(monkeypatch)
    _stub_main_run(monkeypatch, calls)
    monkeypatch.setattr(
        sys,
        "argv",
        _judge_argv(
            "--llm-judge-provider",
            "vertex-anthropic",
            "--llm-judge-model",
            "claude-sonnet-4@20250514",
            "--vertex-project",
            "proj-x",
            "--vertex-location",
            "us",  # does not serve Claude
        ),
    )

    train_and_eval.main()  # warns, does not exit

    err = capsys.readouterr().err
    assert "us-east5" in err


def test_no_judge_provider_is_non_fatal_warning(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    # With nothing configured, evaluation still runs; the judge just warns that
    # semantic checks will record ok=False.
    calls: dict[str, object] = {}
    _clear_judge_env(monkeypatch)
    _stub_main_run(monkeypatch, calls)
    monkeypatch.setattr(sys, "argv", _judge_argv())

    train_and_eval.main()

    assert calls.get("printed") is True
    assert "no LLM judge configured" in capsys.readouterr().err


def test_judge_retry_env_seeded_from_llm_request_retries(
    monkeypatch: pytest.MonkeyPatch,
):
    # The judge reuses the agent's --llm-request-retries / --llm-retry-backoff-s
    # via env (LLM_JUDGE_REQUEST_RETRIES / LLM_JUDGE_RETRY_BACKOFF_S) so a
    # transient judge blip gets the same resilience as the agent path, without a
    # new user-facing knob. Seeding is unconditional (before judge validation),
    # so it applies even when no provider is configured.
    calls: dict[str, object] = {}
    _clear_judge_env(monkeypatch)
    _stub_main_run(monkeypatch, calls)
    monkeypatch.setattr(
        sys,
        "argv",
        _judge_argv(
            "--llm-request-retries",
            "7",
            "--llm-retry-backoff-s",
            "2.5",
        ),
    )

    train_and_eval.main()

    assert os.environ["LLM_JUDGE_REQUEST_RETRIES"] == "7"
    assert os.environ["LLM_JUDGE_RETRY_BACKOFF_S"] == "2.5"


def test_resolve_vertex_helpers_fallback_chain(monkeypatch: pytest.MonkeyPatch):
    _clear_judge_env(monkeypatch)
    # Explicit wins.
    assert train_and_eval._resolve_vertex_project("explicit-p") == "explicit-p"
    assert train_and_eval._resolve_vertex_location("explicit-l") == "explicit-l"
    # Env chain.
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "gcp-proj")
    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "us-west1")
    assert train_and_eval._resolve_vertex_project(None) == "gcp-proj"
    assert train_and_eval._resolve_vertex_location(None) == "us-west1"
    # Nothing set: no project is invented (a guessed one would only fail as a
    # permission error against somebody else's project), but the location keeps
    # a default because it is not account-specific.
    _clear_judge_env(monkeypatch)
    assert train_and_eval._resolve_vertex_project(None) == ""
    assert train_and_eval._resolve_vertex_location(None) == "us"


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__]))
