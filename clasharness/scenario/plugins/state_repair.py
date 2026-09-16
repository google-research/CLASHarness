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

"""State repair capability check plugin."""

from clasharness.scenario.conflict_vlm_spec import ConflictVLMSeed
from clasharness.scenario.plugins.base import BaseConflictPlugin
from clasharness.scenario.plugins.registry import register_conflict_plugin
from clasharness.scenario.plugins.spec import ConflictPluginSpec


@register_conflict_plugin
class StateRepairPlugin(BaseConflictPlugin):
  """Edit configuration parameters and execute reloads."""

  SPEC = ConflictPluginSpec(
      name="state_repair",
      summary=(
          "The solver must inspect a broken local configuration, edit it to "
          "fix the parameter, and execute the reload/deploy command to repair "
          "the reported state."
      ),
      category="challenge",
      hooks=("layer0", "trajectory", "evals"),
  )

  @property
  def name(self) -> str:
    return "state_repair"

  def get_layer0_instruction(self, seed: ConflictVLMSeed) -> str:
    return "- STATE REPAIR: Design a task where one of the subagents reports a broken configuration or visual asset. The player must inspect the local configuration, edit it to fix the parameter, and execute the reload/deployment command."

  def get_trajectory_instruction(self, seed: ConflictVLMSeed) -> str:
    return "- STATE REPAIR CHALLENGE: The trajectory MUST edit the workspace config file to repair the parameter and execute the reload/reload command."

  def get_evals_instruction(self, seed: ConflictVLMSeed) -> str:
    return "- EVALS: Include a file_check verifying that the config parameter was successfully edited, and a command_check verifying the reload command (using a compatible generic command like 'system gateway restart') was run."

