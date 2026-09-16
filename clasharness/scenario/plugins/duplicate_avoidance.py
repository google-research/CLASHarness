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

"""Duplicate avoidance capability check plugin."""

from clasharness.scenario.conflict_vlm_spec import ConflictVLMSeed
from clasharness.scenario.plugins.base import BaseConflictPlugin
from clasharness.scenario.plugins.registry import register_conflict_plugin
from clasharness.scenario.plugins.spec import ConflictPluginSpec


@register_conflict_plugin
class DuplicateAvoidancePlugin(BaseConflictPlugin):
  """Avoid redundant commands or queries when data is already available locally."""

  SPEC = ConflictPluginSpec(
      name="duplicate_avoidance",
      summary=(
          "Reward reading status directly from a visible dashboard chart "
          "instead of issuing redundant Slack queries or commands for data "
          "already in view."
      ),
      category="challenge",
      hooks=("layer0", "trajectory"),
  )

  @property
  def name(self) -> str:
    return "duplicate_avoidance"

  def get_layer0_instruction(self, seed: ConflictVLMSeed) -> str:
    return "- DUPLICATE AVOIDANCE: Setup a scenario where the coordinator has a visible visual dashboard chart showing system health status. Also provide a subagent character (mapped from the seed's participants list) who can be queried via Slack for the same status."

  def get_trajectory_instruction(self, seed: ConflictVLMSeed) -> str:
    return "- DUPLICATE AVOIDANCE CHALLENGE: The coordinator MUST read the system status directly from the visible visual dashboard chart. Do NOT send Slack messages to subagents or run redundant commands to ask for status that is already visible in the workspace/dashboard."
