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

import re
import subprocess
import sys
import shutil
from pathlib import Path

from clasharness.core.task import load_task, load_task_ids


def test_generate_pro_scenarios_mock_cli(tmp_path: Path) -> None:
    scenario_dir = tmp_path / "scenarios"
    data_dir = tmp_path / "data"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_pro_scenarios.py",
            "--provider",
            "mock",
            "--seed-dir",
            "tests/fixtures/pro_seeds",
            "--output-scenario-dir",
            str(scenario_dir),
            "--output-data-dir",
            str(data_dir),
            "--target-rounds",
            "24",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Generated 1 scenario" in result.stdout
    assert (
        scenario_dir
        / "pro_scenario_workflow"
        / "grant_launch_evidence_reconciliation"
        / "scenario.json"
    ).exists()
    task_ids = load_task_ids("total", data_dir=data_dir)
    assert len(task_ids) == 24
    tasks = [load_task(task_id, data_dir=data_dir) for task_id in task_ids]
    non_report = [
        task
        for task in tasks
        if task.data.public["round_type"] in {"observation", "reconciliation", "update_aware"}
    ]
    report = [task for task in tasks if task.data.public["round_type"] == "action"]
    assert len(non_report) == 20
    assert len(report) == 4
    assert all("/reports/" not in task.instruction for task in non_report)
    assert all("/reports/" in task.instruction for task in report)
    assert any(
        any(check.get("effect_type") == "tasks_created" for check in task.ground_truth.evaluation_checks)
        for task in non_report
    )
    assert any(
        any(check.get("effect_type") == "emails_sent" for check in task.ground_truth.evaluation_checks)
        for task in non_report
    )
    required_effects_by_block: dict[str, set[str]] = {
        "q08-q14": set(),
        "q15-q20": set(),
    }
    for task in tasks:
        round_id = str(task.data.public["round_id"])
        round_index = int(round_id[1:])
        block = "q08-q14" if 8 <= round_index <= 14 else "q15-q20" if 15 <= round_index <= 20 else ""
        if not block:
            continue
        for check in task.ground_truth.evaluation_checks:
            if isinstance(check, dict) and check.get("type") == "effect":
                required_effects_by_block[block].add(str(check.get("effect_type")))
    assert {
        "tasks_created",
        "emails_sent",
        "calendar_events_created",
        "files_created",
    }.issubset(required_effects_by_block["q08-q14"])
    assert {
        "tasks_created",
        "emails_sent",
        "files_created",
    }.issubset(required_effects_by_block["q15-q20"])
    future_leaks: list[str] = []
    for task in tasks:
        round_index = int(str(task.data.public["round_id"])[1:])
        overrides = task.data.private.get("initial_state_overrides", {})
        runtime_text = "\n".join(str(value) for value in overrides.values())
        for match in re.finditer(
            r"\bq(\d{1,2})\b|\bRound[-_ ]?(\d{1,2})\b|\br(\d{1,2})\b",
            runtime_text,
            flags=re.IGNORECASE,
        ):
            value = int(next(group for group in match.groups() if group))
            snippet = runtime_text[max(0, match.start() - 50) : match.end() + 80]
            if value > round_index and value <= 24 and "Total Rounds" not in snippet and "Target Rounds" not in snippet:
                future_leaks.append(f"{task.task_id}: {snippet}")
                break
    assert future_leaks == []


def test_generate_pro_scenarios_compile_existing_cli(tmp_path: Path) -> None:
    scenario_dir = tmp_path / "scenarios"
    data_dir = tmp_path / "data"
    compile_dir = tmp_path / "compiled"
    base_cmd = [
        sys.executable,
        "scripts/generate_pro_scenarios.py",
        "--provider",
        "mock",
        "--seed-dir",
        "tests/fixtures/pro_seeds",
        "--output-scenario-dir",
        str(scenario_dir),
        "--output-data-dir",
        str(data_dir),
        "--target-rounds",
        "24",
    ]
    subprocess.run(base_cmd, check=True, capture_output=True, text=True)
    shutil.rmtree(data_dir)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_pro_scenarios.py",
            "--compile-existing-scenarios",
            "--seed-dir",
            "tests/fixtures/pro_seeds",
            "--output-scenario-dir",
            str(scenario_dir),
            "--output-data-dir",
            str(compile_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Compiled 1 existing scenario" in result.stdout
    task_ids = load_task_ids("total", data_dir=compile_dir)
    assert len(task_ids) == 24
