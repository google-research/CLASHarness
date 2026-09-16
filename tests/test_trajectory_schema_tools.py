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
import subprocess
import sys
from pathlib import Path

from clasharness.utils.trajectory_schema import (
    iter_normalized_steps,
    normalize_episode,
    summarize_records,
)

# No `.resolve()`: under the build system `__file__` is a symlink into the CAS,
# and resolving it walks out of the runfiles tree that holds `scripts/`.
_REPO_ROOT = Path(__file__).parents[1]


def _write_fixture(path: Path) -> None:
    records = [
        {
            "task_id": "pro_llm_alpha_q01",
            "agent": "llm",
            "mode": "martin",
            "success": True,
            "score": 1.0,
            "steps": 2,
            "duration_s": 1.2,
            "metadata": {"scenario_slug": "alpha", "round_id": "q01", "round_type": "observation"},
            "trajectory": [
                {
                    "step": 0,
                    "action": "tasks add --title alpha",
                    "stdout": "created",
                    "stderr": "",
                    "exit_code": 0,
                    "reward": 0.0,
                    "done": False,
                    "current_score": 0.5,
                    "score_delta": 0.5,
                    "passed_checks": ["task created"],
                    "failed_checks": ["anchor"],
                },
                {
                    "step": 1,
                    "action": "tasks list",
                    "stdout": "alpha",
                    "stderr": "",
                    "exit_code": 0,
                    "reward": 1.0,
                    "done": True,
                    "current_score": 1.0,
                    "score_delta": 0.5,
                    "passed_checks": ["task created", "anchor"],
                    "failed_checks": [],
                },
            ],
        },
        {
            "task_id": "pro_llm_alpha_q08",
            "agent": "llm",
            "mode": "martin",
            "success": False,
            "score": 0.2,
            "steps": 1,
            "duration_s": 0.5,
            "stop_reason": "failed_required_anchor_check",
            "metadata": {"scenario_slug": "alpha", "round_id": "q08", "round_type": "reconciliation"},
            "trajectory": [
                {"step": 0, "action": "email read --id x", "stdout": "", "stderr": "", "exit_code": 0, "reward": 0.0, "done": False}
            ],
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def test_normalize_episode_and_steps() -> None:
    record = {
        "task_id": "pro_llm_demo_case_q15",
        "agent": "expert",
        "success": False,
        "score": 0.4,
        "steps": 1,
        "stop_reason": "missing_required_action_log: /actions/q15.md",
        "trajectory": [{"step": 0, "action": "file list --path /", "current_score": 0.4}],
    }

    episode = normalize_episode(record)
    steps = list(iter_normalized_steps(record))

    assert episode["scenario_slug"] == "demo_case"
    assert episode["round_id"] == "q15"
    assert episode["round_type"] == "update_aware"
    assert episode["failure_mode"] == "missing_required_action_log: /actions/q15.md"
    assert steps[0]["final_score"] == 0.4
    assert steps[0]["current_score"] == 0.4


def test_summarize_records_by_stage_and_scenario(tmp_path: Path) -> None:
    path = tmp_path / "traj.jsonl"
    _write_fixture(path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    summary = summarize_records(records)

    assert summary["total"] == 2
    assert summary["successful"] == 1
    assert summary["by_round_type"]["observation"]["passed"] == 1
    assert summary["by_round_type"]["reconciliation"]["total"] == 1
    assert summary["by_scenario"]["alpha"]["total"] == 2
    assert summary["failure_modes"]["failed_required_anchor_check"] == 1


def test_analyze_trajectories_cli_outputs_summary(tmp_path: Path) -> None:
    path = tmp_path / "traj.jsonl"
    out = tmp_path / "summary.json"
    _write_fixture(path)

    proc = subprocess.run(
        [sys.executable, "scripts/analyze_trajectories.py", str(path), "--json-out", str(out)],
        cwd=_REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Success: 1/2" in proc.stdout
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["by_round_type"]["observation"]["passed"] == 1


def test_export_trajectory_dataset_cli_writes_three_jsonl_files(tmp_path: Path) -> None:
    path = tmp_path / "traj.jsonl"
    out_dir = tmp_path / "export"
    _write_fixture(path)

    proc = subprocess.run(
        [sys.executable, "scripts/export_trajectory_dataset.py", str(path), "--output-dir", str(out_dir)],
        cwd=_REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Wrote 2 episodes" in proc.stdout
    episodes = out_dir / "episodes.jsonl"
    steps = out_dir / "steps.jsonl"
    sft = out_dir / "sft_commands.jsonl"
    assert episodes.exists()
    assert steps.exists()
    assert sft.exists()
    assert len(episodes.read_text(encoding="utf-8").splitlines()) == 2
    assert len(steps.read_text(encoding="utf-8").splitlines()) == 3
    assert len(sft.read_text(encoding="utf-8").splitlines()) == 2
