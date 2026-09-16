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

"""Information transfer capability check plugin."""

from clasharness.scenario.conflict_vlm_spec import ConflictVLMSeed
from clasharness.scenario.plugins.base import BaseConflictPlugin
from clasharness.scenario.plugins.registry import register_conflict_plugin
from clasharness.scenario.plugins.spec import ConflictPluginSpec


@register_conflict_plugin
class InformationTransferPlugin(BaseConflictPlugin):
  """Extract dynamic tokens from visual assets and pass them in messages."""

  SPEC = ConflictPluginSpec(
      name="information_transfer",
      summary=(
          "A dynamic passcode is embedded in a visual dashboard; the solver "
          "must extract it and relay the exact string to a subagent via Slack "
          "to authorize an action."
      ),
      category="challenge",
      hooks=("layer0", "trajectory", "evals"),
  )

  @property
  def name(self) -> str:
    return "information_transfer"

  def get_layer0_instruction(self, seed: ConflictVLMSeed) -> str:
    return "- INFORMATION TRANSFER: Embed a dynamic security passcode directly inside a visual telemetry dashboard chart. The passcode is required by one of the subagents to authorize reload/deploy actions."

  def get_trajectory_instruction(self, seed: ConflictVLMSeed) -> str:
    return "- INFORMATION TRANSFER CHALLENGE: The trajectory MUST inspect the visual telemetry dashboard chart to find the security passcode, and send a Slack message to the subagent containing the exact passcode string."

  def get_evals_instruction(self, seed: ConflictVLMSeed) -> str:
    return "- EVALS: Enforce an evaluation assertion verifying that the Slack message sent to the subagent contains the passcode extracted from the VLM asset."
