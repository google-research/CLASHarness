#!/usr/bin/env python3

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

"""Export one representative sample per realistic workflow scenario."""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = ROOT / "clasharness" / "data" / "tasks"
DEFAULT_OUTPUT = ROOT / "clasharness" / "data" / "datasets" / "realistic_workflow_samples.json"
FAMILIES = ("complex_composed_workflow", "hard_decision_workflow")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _load_spec(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for spec_path in sorted(TASKS_DIR.glob("*/specs.json")):
        spec = _load_spec(spec_path)
        if spec.get("generator_id") in FAMILIES:
            spec["_spec_path"] = str(spec_path)
            specs.append(spec)
    return specs


def _display_actions(spec: dict[str, Any]) -> list[str]:
    commands = list(spec.get("ground_truth", {}).get("solution_commands", []))
    if spec.get("generator_id") != "hard_decision_workflow":
        return commands

    hidden_types = {str(item.get("type", "")) for item in spec.get("hidden_constraints", [])}
    decisions = set(spec.get("decision_requirements", []))
    display: list[str] = []
    for command in commands:
        updated = command
        if "infer_schedule" in decisions:
            updated = re.sub(r"--due\s+\S+", "--due <agent-chosen due date>", updated)
            updated = re.sub(r"--start\s+\S+", "--start <agent-chosen start time>", updated)
            updated = re.sub(r"--cron\s+'[^']+'", "--cron '<agent-chosen cron schedule>'", updated)
        if "title" in hidden_types:
            updated = re.sub(r"--title\s+'[^']+'", "--title '<agent-chosen title>'", updated)
        if "subject" in hidden_types:
            updated = re.sub(r"--subject\s+'[^']+'", "--subject '<agent-chosen subject>'", updated)
        if "body" in hidden_types:
            updated = re.sub(r"--body\s+'[^']+'", "--body '<agent-chosen short update>'", updated)
        if "path" in hidden_types:
            updated = re.sub(r"--path\s+'[^']+'", "--path '<agent-chosen notes path>'", updated)
        if "priority" in hidden_types and "infer_priority" in decisions:
            updated = re.sub(r"--priority\s+\S+", "--priority <agent-chosen priority>", updated)
        display.append(updated)
    return display


def _scenario_slug(spec: dict[str, Any]) -> str:
    return str(
        spec.get("complex_scenario_slug")
        or spec.get("hard_decision_scenario")
        or spec.get("data", {}).get("public", {}).get("complex_scenario_slug")
        or spec.get("data", {}).get("public", {}).get("hard_decision_scenario")
        or spec.get("template_id")
        or spec.get("task_id")
    )


def _sample_entry(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "family": spec.get("generator_id"),
        "scenario_slug": _scenario_slug(spec),
        "task_id": spec.get("task_id"),
        "sample_question": spec.get("instruction"),
        "sample_actions": _display_actions(spec),
        "reference_solution_commands": list(spec.get("ground_truth", {}).get("solution_commands", [])),
        "step_count": len(spec.get("ground_truth", {}).get("solution_commands", [])),
        "domains": list(spec.get("domains", [])),
        "decision_requirements": list(spec.get("decision_requirements", [])),
    }


def main() -> None:
    args = parse_args()
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for spec in _iter_specs():
        key = (str(spec.get("generator_id")), _scenario_slug(spec))
        by_key.setdefault(key, spec)

    samples = [_sample_entry(spec) for _, spec in sorted(by_key.items())]
    payload = {
        "generated_at": date.today().isoformat(),
        "note": (
            "One representative sample per realistic workflow scenario. "
            "For hard_decision_workflow, sample_actions are display-oriented and may replace "
            "hidden execution details with <agent-chosen ...> placeholders; "
            "reference_solution_commands preserves the canonical reference commands."
        ),
        "total_samples": len(samples),
        "samples": samples,
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
