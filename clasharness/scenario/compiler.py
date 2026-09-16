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

"""Compile PRO scenario specs back into current runnable Task specs."""

from __future__ import annotations

from clasharness.core.task import Task
from clasharness.scenario.spec import ScenarioSpec
from clasharness.scenario.validator import validate_scenario


def compile_scenario_to_tasks(spec: ScenarioSpec, *, validate: bool = True) -> list[Task]:
    """Compile all scenario rounds into current `Task` objects."""

    from clasharness.scenario.consistency_layer import audit_and_repair_scenario
    spec = audit_and_repair_scenario(spec)

    if validate:
        validate_scenario(spec).raise_for_errors()

    return [Task.from_dict(round_item.task_spec) for round_item in spec.rounds]
