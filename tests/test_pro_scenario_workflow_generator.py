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

from collections import Counter

import clasharness.tasks.generators  # noqa: F401  (auto-registers all generators)
from clasharness.scenario.compiler import compile_scenario_to_tasks
from clasharness.scenario.export import pro_tasks_to_scenario_specs
from clasharness.scenario.validator import validate_scenario
from clasharness.tasks.generation_options import set_generation_options
from clasharness.tasks.registry import generate_all_tasks


def test_pro_scenario_workflow_is_disabled_by_default():
    set_generation_options()

    tasks = generate_all_tasks(generator_ids=["pro_scenario_workflow"])

    assert tasks == []


def test_pro_scenario_workflow_generates_forge_style_scenarios():
    set_generation_options(include_pro_scenarios=True, pro_rounds_per_scenario=24)

    tasks = generate_all_tasks(generator_ids=["pro_scenario_workflow"])
    specs = pro_tasks_to_scenario_specs(tasks)

    assert len(tasks) == 48
    assert len(specs) == 2
    assert Counter(task.data.public["pro_round_type"] for task in tasks) == {
        "multi_choice": 14,
        "exec_check": 34,
    }

    for spec in specs:
        result = validate_scenario(spec)
        assert result.passed, result.errors
        assert spec.metadata["scenario_series"] == "pro_strict_v1"
        assert spec.metadata["session_count"] >= 5
        assert spec.metadata["workspace_file_count"] >= 7
        assert spec.metadata["update_count"] == 3
        assert len(spec.rounds) == 24
        assert len(spec.eval_items) == 24
        assert len(compile_scenario_to_tasks(spec)) == 24
        assert {item.kind for item in spec.eval_items} == {"multi_choice", "exec_check"}


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__]))
