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

import clasharness.tasks.generators  # noqa: F401  (auto-registers all generators)
from clasharness.scenario.compiler import compile_scenario_to_tasks
from clasharness.scenario.export import complex_tasks_to_scenario_specs
from clasharness.scenario.validator import validate_scenario
from clasharness.tasks.generation_options import set_generation_options
from clasharness.tasks.registry import generate_all_tasks


def test_complex_composed_workflow_exports_valid_scenarios_and_compiles_back():
    set_generation_options(
        complex_task_pack="standard",
        complex_scenario_profile="life_work",
        complex_min_steps=3,
        complex_max_steps=5,
    )
    tasks = generate_all_tasks(generator_ids=["complex_composed_workflow"])

    specs = complex_tasks_to_scenario_specs(tasks)

    assert specs
    spec = specs[0]
    assert spec.generator_id == "complex_composed_workflow"
    assert validate_scenario(spec).passed
    assert spec.rounds[0].expert_trajectory
    assert "complex_scenario_slug" in spec.metadata

    compiled = compile_scenario_to_tasks(spec)
    assert {task.task_id for task in compiled} == {round_item.task_id for round_item in spec.rounds}
    assert compiled[0].to_dict()["ground_truth"]["evaluation_checks"]


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__]))
