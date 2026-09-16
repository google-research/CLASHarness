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

"""Lightweight validation for PRO scenario specs."""

from __future__ import annotations

from dataclasses import dataclass, field

from clasharness.scenario.spec import ScenarioSpec


@dataclass
class ScenarioValidationResult:
    """Validation outcome for a scenario spec."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            joined = "\n".join(f"- {error}" for error in self.errors)
            raise ValueError(f"Scenario validation failed:\n{joined}")


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def validate_scenario(spec: ScenarioSpec) -> ScenarioValidationResult:
    """Validate v1 scenario constraints.

    This intentionally enforces a light compatibility profile. Future strict
    mode can add full scenario-forge round-count and script-file constraints.
    """

    result = ScenarioValidationResult()

    if not spec.scenario_id:
        result.errors.append("scenario_id is required")
    if not spec.generator_id:
        result.errors.append("generator_id is required")
    if not spec.scenario_slug:
        result.errors.append("scenario_slug is required")
    if not spec.title:
        result.errors.append("title is required")
    if not spec.rounds:
        result.errors.append("at least one round is required")

    round_ids = [round_item.round_id for round_item in spec.rounds]
    task_ids = [round_item.task_id for round_item in spec.rounds]
    update_ids = [update.update_id for update in spec.updates]
    session_ids = [session.session_id for session in spec.sessions]

    for duplicate in _duplicates(round_ids):
        result.errors.append(f"duplicate round_id: {duplicate}")
    for duplicate in _duplicates(update_ids):
        result.errors.append(f"duplicate update_id: {duplicate}")
    for duplicate in _duplicates(session_ids):
        result.errors.append(f"duplicate session_id: {duplicate}")

    round_id_set = set(round_ids)
    update_id_set = set(update_ids)
    session_id_set = set(session_ids)
    workspace_file_ids = {
        str(item.get("id") or item.get("path") or "")
        for item in spec.workspace.files
        if item.get("id") or item.get("path")
    }

    for round_item in spec.rounds:
        prefix = f"round {round_item.round_id or '<missing>'}"
        if not round_item.round_id:
            result.errors.append("round_id is required for every round")
        if not round_item.task_id:
            result.errors.append(f"{prefix}: task_id is required")
        if not round_item.instruction:
            result.errors.append(f"{prefix}: instruction is required")
        if not isinstance(round_item.task_spec, dict) or not round_item.task_spec:
            result.errors.append(f"{prefix}: task_spec is required")
            continue
        for required in ("task_id", "instruction", "initial_state", "ground_truth", "data"):
            if required not in round_item.task_spec:
                result.errors.append(f"{prefix}: task_spec.{required} is required")
        ground_truth = round_item.task_spec.get("ground_truth") or {}
        if "evaluation_checks" not in ground_truth:
            result.errors.append(f"{prefix}: task_spec.ground_truth.evaluation_checks is required")
        if "solution_commands" not in ground_truth:
            result.errors.append(f"{prefix}: task_spec.ground_truth.solution_commands is required")
        for update_id in round_item.update_ids:
            if update_id not in update_id_set:
                result.errors.append(f"{prefix}: unknown update reference {update_id}")
        for session_id in round_item.session_refs:
            if session_id not in session_id_set:
                result.errors.append(f"{prefix}: unknown session reference {session_id}")
        for workspace_ref in round_item.workspace_refs:
            if workspace_ref not in workspace_file_ids:
                result.errors.append(f"{prefix}: unknown workspace reference {workspace_ref}")

    for eval_item in spec.eval_items:
        if not eval_item.eval_id:
            result.errors.append("eval_id is required for every eval item")
        if eval_item.round_id not in round_id_set:
            result.errors.append(
                f"eval item {eval_item.eval_id or '<missing>'}: unknown round reference {eval_item.round_id}"
            )
        if eval_item.kind == "task_checks" and not eval_item.checks:
            result.warnings.append(
                f"eval item {eval_item.eval_id or '<missing>'}: task_checks item has no checks"
            )

    for update in spec.updates:
        if update.before_round and update.before_round not in round_id_set:
            result.errors.append(
                f"update {update.update_id or '<missing>'}: unknown before_round {update.before_round}"
            )

    if spec.manifest:
        missing = sorted(set(spec.manifest.compiled_task_ids) - set(task_ids))
        if missing:
            result.errors.append(
                "manifest.compiled_task_ids contains ids not present in rounds: "
                + ", ".join(missing[:10])
            )
        if spec.manifest.source_generator and spec.manifest.source_generator != spec.generator_id:
            result.warnings.append(
                "manifest.source_generator differs from scenario generator_id"
            )

    return result
