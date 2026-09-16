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

"""Multi-source reasoning capability check plugin."""

from clasharness.scenario.conflict_vlm_spec import ConflictVLMSeed
from clasharness.scenario.plugins.base import BaseConflictPlugin
from clasharness.scenario.plugins.registry import register_conflict_plugin
from clasharness.scenario.plugins.spec import ConflictPluginSpec


@register_conflict_plugin
class MultiSourceReasoningPlugin(BaseConflictPlugin):
  """Cross-reference visual charts and text messages to reconcile claim conflicts."""

  SPEC = ConflictPluginSpec(
      name="multi_source_reasoning",
      summary=(
          "Two subagents make conflicting claims; the solver must "
          "cross-reference a visual metric chart's timeline against the chat "
          "log to determine which claim the evidence supports."
      ),
      category="challenge",
      hooks=("layer0", "trajectory"),
  )

  @property
  def name(self) -> str:
    return "multi_source_reasoning"

  def get_layer0_instruction(self, seed: ConflictVLMSeed) -> str:
    return "- MULTI-SOURCE REASONING: Create a conflict of claims between two subagents in the chat log (e.g., one claiming their fix resolved the incident, another claiming the leak is still active). Generate a visual database/traffic metric chart whose trend (e.g. latency dropping back to normal at a specific timestamp) resolves this conflict."

  def get_trajectory_instruction(self, seed: ConflictVLMSeed) -> str:
    return "- MULTI-SOURCE REASONING CHALLENGE: The trajectory MUST open the visual metric chart, compare the timestamps with the claims of the subagents in the chat logs, and send a Slack message to the subagent reconciling the conflict using the visual evidence."
