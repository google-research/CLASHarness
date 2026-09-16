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

"""Scenario-first generation support for CLASHarness PRO."""

from clasharness.scenario.compiler import compile_scenario_to_tasks
from clasharness.scenario.export import (
    complex_tasks_to_scenario_specs,
    hard_tasks_to_scenario_specs,
    pro_tasks_to_scenario_specs,
    scenario_specs_for_tasks,
    write_scenario_specs,
)
from clasharness.scenario.llm_client import (
    MockScenarioLLMClient,
    OpenAICompatibleScenarioLLMClient,
    ScenarioLLMError,
)
from clasharness.scenario.llm_pipeline import LLMScenarioPipeline, ScenarioSeed
from clasharness.scenario.spec import (
    ScenarioEvalItem,
    ScenarioManifest,
    ScenarioRound,
    ScenarioSession,
    ScenarioSpec,
    ScenarioUpdate,
    ScenarioWorkspace,
)
from clasharness.scenario.strict_validator import strict_validate_scenario_package
from clasharness.scenario.validator import ScenarioValidationResult, validate_scenario

__all__ = [
    "LLMScenarioPipeline",
    "MockScenarioLLMClient",
    "OpenAICompatibleScenarioLLMClient",
    "ScenarioEvalItem",
    "ScenarioLLMError",
    "ScenarioManifest",
    "ScenarioRound",
    "ScenarioSeed",
    "ScenarioSession",
    "ScenarioSpec",
    "ScenarioUpdate",
    "ScenarioWorkspace",
    "ScenarioValidationResult",
    "compile_scenario_to_tasks",
    "complex_tasks_to_scenario_specs",
    "hard_tasks_to_scenario_specs",
    "pro_tasks_to_scenario_specs",
    "scenario_specs_for_tasks",
    "strict_validate_scenario_package",
    "validate_scenario",
    "write_scenario_specs",
]
