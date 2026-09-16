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

import pytest

from clasharness.core.task import GroundTruth, Task, TaskData
from clasharness.scenario.spec import (
    ScenarioEvalItem,
    ScenarioManifest,
    ScenarioRound,
    ScenarioSpec,
)
from clasharness.scenario.validator import validate_scenario


def _task(task_id: str = "demo_task") -> Task:
    return Task(
        task_id=task_id,
        instruction="Finish the demo workflow.",
        domains=["messaging"],
        difficulty=2,
        initial_state="default",
        ground_truth=GroundTruth(
            solution_commands=["openclaw message send --channel ops --text done"],
            evaluation_checks=[{"type": "state", "key": "ops.sent", "condition": "exists"}],
        ),
        data=TaskData(public={"scenario": "demo"}, private={}),
        generator_id="demo_generator",
        template_id="demo_generator",
    )


def _scenario() -> ScenarioSpec:
    task = _task()
    round_item = ScenarioRound(
        round_id="round_001",
        task_id=task.task_id,
        instruction=task.instruction,
        task_spec=task.to_dict(),
        expert_trajectory=list(task.ground_truth.solution_commands),
    )
    return ScenarioSpec(
        scenario_id="demo_generator:demo",
        generator_id="demo_generator",
        scenario_slug="demo",
        title="Demo",
        rounds=[round_item],
        eval_items=[
            ScenarioEvalItem(
                eval_id="eval_round_001",
                round_id="round_001",
                checks=list(task.ground_truth.evaluation_checks),
            )
        ],
        manifest=ScenarioManifest(
            source_generator="demo_generator",
            compiled_task_ids=[task.task_id],
        ),
    )


def test_scenario_spec_serializes_and_deserializes_minimal_scenario():
    spec = _scenario()

    loaded = ScenarioSpec.from_json(spec.to_json())

    assert loaded.scenario_id == spec.scenario_id
    assert loaded.rounds[0].task_id == "demo_task"
    assert loaded.eval_items[0].round_id == "round_001"
    assert validate_scenario(loaded).passed


def test_scenario_validator_rejects_missing_ids():
    spec = _scenario()
    spec.scenario_id = ""
    spec.rounds[0].round_id = ""

    result = validate_scenario(spec)

    assert not result.passed
    assert "scenario_id is required" in result.errors
    assert "round_id is required for every round" in result.errors


def test_scenario_validator_rejects_duplicate_round_ids():
    spec = _scenario()
    spec.rounds.append(spec.rounds[0])

    result = validate_scenario(spec)

    assert not result.passed
    assert "duplicate round_id: round_001" in result.errors


def test_scenario_validator_rejects_invalid_update_references():
    spec = _scenario()
    spec.rounds[0].update_ids = ["upd_missing"]

    result = validate_scenario(spec)

    assert not result.passed
    assert "round round_001: unknown update reference upd_missing" in result.errors


def test_scenario_validation_result_raises_for_errors():
    spec = _scenario()
    spec.eval_items[0].round_id = "missing"

    with pytest.raises(ValueError, match="unknown round reference"):
        validate_scenario(spec).raise_for_errors()
