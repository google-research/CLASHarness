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

"""Workflow completion capability check plugin."""

from clasharness.scenario.conflict_vlm_spec import ConflictVLMSeed
from clasharness.scenario.plugins.base import BaseConflictPlugin
from clasharness.scenario.plugins.registry import register_conflict_plugin
from clasharness.scenario.plugins.spec import ConflictPluginSpec


@register_conflict_plugin
class WorkflowCompletionPlugin(BaseConflictPlugin):
  """Verify the fix success at the end of the debugging loop."""

  SPEC = ConflictPluginSpec(
      name="workflow_completion",
      summary=(
          "Status only turns green on the next round after the reload runs, "
          "forcing the solver to perform a final verification step to confirm "
          "the fix actually succeeded."
      ),
      category="challenge",
      hooks=("layer0", "trajectory", "evals"),
  )

  @property
  def name(self) -> str:
    return "workflow_completion"

  def get_layer0_instruction(self, seed: ConflictVLMSeed) -> str:
    return "- WORKFLOW COMPLETION: Ensure that after the fix is applied, the status is not updated on the dashboard until the next round. The dashboard will show a green status only in the next round after the reload script runs."

  def get_trajectory_instruction(self, seed: ConflictVLMSeed) -> str:
    return "- WORKFLOW COMPLETION CHALLENGE: The trajectory MUST perform a final verification step (e.g., inspect the updated visual dashboard in the next round) to confirm the fix succeeded."

  def get_evals_instruction(self, seed: ConflictVLMSeed) -> str:
    return "- EVALS: Add a check verifying that the final status check action was executed."
