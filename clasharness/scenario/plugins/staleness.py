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

"""Staleness and Developer out-of-sync context conflict plugin."""

from clasharness.scenario.conflict_vlm_spec import ConflictVLMSeed
from clasharness.scenario.plugins.base import BaseConflictPlugin
from clasharness.scenario.plugins.registry import register_conflict_plugin
from clasharness.scenario.plugins.spec import ConflictPluginSpec


@register_conflict_plugin
class TemporalStalenessPlugin(BaseConflictPlugin):
  """Plugin modeling temporal staleness and out-of-sync subagent context."""

  SPEC = ConflictPluginSpec(
      name="temporal_staleness",
      summary=(
          "Realign an out-of-sync subagent that relies on a stale cache or "
          "hallucinated status until it is corrected by the specific trigger "
          "evidence found in the workspace."
      ),
      category="inconsistency",
      hooks=("layer0", "layer1", "layer1b"),
  )

  @property
  def name(self) -> str:
    return "temporal_staleness"

  def get_layer0_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "Plan phases covering: 1) Initial latency anomaly detection, and "
        "2) Aligning the out-of-sync subagent (relying on stale cache or hallucination) with a critical trigger evidence token."
    )

  def get_layer1_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "Model the stale context or hallucination behavior of one of the participating subagents: "
        "they must present a stale/healthy status or misread logs until aligned/corrected with the specific trigger evidence found in the workspace."
    )

  def get_layer1b_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "Embed subtle clues in workspace files and chat sessions about "
        "the active subagent relying on outdated context, and the trigger evidence "
        "needed to synchronize/update their state."
    )

