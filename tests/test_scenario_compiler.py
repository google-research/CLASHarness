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

from clasharness.core.task import GroundTruth, Task, TaskData
from clasharness.scenario.compiler import compile_scenario_to_tasks
from clasharness.scenario.spec import ScenarioEvalItem, ScenarioRound, ScenarioSpec


def test_compile_scenario_round_into_current_task_and_round_trip():
    task = Task(
        task_id="compile_demo",
        instruction="Resolve the compile demo.",
        domains=["tasks"],
        difficulty=3,
        initial_state="default",
        ground_truth=GroundTruth(
            solution_commands=["openclaw tasks complete --id task_0001"],
            evaluation_checks=[{"type": "state", "key": "task_0001.status", "condition": "equals", "expected": "done"}],
        ),
        data=TaskData(public={"decision_requirements": ["infer_target"]}, private={}),
        template_id="demo_template",
        generator_id="demo_generator",
        decision_requirements=["infer_target"],
        realism_tags=["multi_step", "challenge"],
    )
    spec = ScenarioSpec(
        scenario_id="demo_generator:compile",
        generator_id="demo_generator",
        scenario_slug="compile",
        title="Compile",
        rounds=[
            ScenarioRound(
                round_id="round_001",
                task_id=task.task_id,
                instruction=task.instruction,
                task_spec=task.to_dict(),
                expert_trajectory=list(task.ground_truth.solution_commands),
            )
        ],
        eval_items=[
            ScenarioEvalItem(
                eval_id="eval_round_001",
                round_id="round_001",
                checks=list(task.ground_truth.evaluation_checks),
            )
        ],
    )

    compiled = compile_scenario_to_tasks(spec)

    assert len(compiled) == 1
    compiled_task = compiled[0]
    assert compiled_task.task_id == task.task_id
    assert compiled_task.ground_truth.evaluation_checks == task.ground_truth.evaluation_checks
    assert compiled_task.decision_requirements == ["infer_target"]
    assert compiled_task.realism_tags == ["multi_step", "challenge"]
    assert Task.from_dict(compiled_task.to_dict()).to_dict() == compiled_task.to_dict()
