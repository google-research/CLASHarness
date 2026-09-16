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

"""Gap completion capability check plugin."""

from clasharness.scenario.conflict_vlm_spec import ConflictVLMSeed
from clasharness.scenario.plugins.base import BaseConflictPlugin
from clasharness.scenario.plugins.registry import register_conflict_plugin
from clasharness.scenario.plugins.spec import ConflictPluginSpec


@register_conflict_plugin
class GapCompletionPlugin(BaseConflictPlugin):
  """Resolve conceptual names to physical resources via directory mappings."""

  SPEC = ConflictPluginSpec(
      name="gap_completion",
      summary=(
          "The task omits the resource path and owner; the solver must read a "
          "visual chart to find the anomalous resource and query a directory "
          "subagent to resolve its owner before delegating."
      ),
      category="challenge",
      hooks=("layer0", "trajectory"),
  )

  @property
  def name(self) -> str:
    return "gap_completion"

  def get_layer0_instruction(self, seed: ConflictVLMSeed) -> str:
    return "- GAP COMPLETION: The starting task instruction must ask the player to identify the target resource with anomalous metrics from a visual comparison chart and delegate restart to its owner subagent. The starting instruction must NOT state the resource path or the owner name directly. Design one subagent to serve as a directory catalog who maps resources to owners in chat conversation."

  def get_trajectory_instruction(self, seed: ConflictVLMSeed) -> str:
    return "- GAP COMPLETION CHALLENGE: The trajectory MUST: (1) inspect the visual metric chart to identify the worst resource, (2) send a Slack message to the directory subagent to retrieve the owner name, and (3) delegate the repair action to the owner subagent."
