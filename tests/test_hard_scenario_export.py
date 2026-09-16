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
from clasharness.scenario.export import hard_tasks_to_scenario_specs
from clasharness.scenario.validator import validate_scenario
from clasharness.tasks.generation_options import set_generation_options
from clasharness.tasks.registry import generate_all_tasks


def test_hard_decision_tasks_group_by_scenario_and_compile_to_same_ids():
    set_generation_options(
        hard_decision_variants_per_scenario=2,
        hard_decision_scenario_counts={},
    )
    tasks = generate_all_tasks(generator_ids=["hard_decision_workflow"])

    specs = hard_tasks_to_scenario_specs(tasks)

    assert len(specs) == 17
    assert all(validate_scenario(spec).passed for spec in specs)

    original_ids = {task.task_id for task in tasks}
    compiled_ids = {
        task.task_id
        for spec in specs
        for task in compile_scenario_to_tasks(spec)
    }
    assert compiled_ids == original_ids

    existing_state = next(
        spec for spec in specs if spec.scenario_slug == "existing_state_followthrough"
    )
    assert existing_state.metadata["hard_decision_scenario"] == "existing_state_followthrough"
    assert existing_state.rounds[0].task_spec["data"]["public"]["hard_decision_scenario"] == "existing_state_followthrough"


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__]))
