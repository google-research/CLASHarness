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

"""Evidence reconciliation and mismatching source data conflict plugin."""

from clasharness.scenario.conflict_vlm_spec import ConflictVLMSeed
from clasharness.scenario.plugins.base import BaseConflictPlugin
from clasharness.scenario.plugins.registry import register_conflict_plugin
from clasharness.scenario.plugins.spec import ConflictPluginSpec


@register_conflict_plugin
class EvidenceMismatchPlugin(BaseConflictPlugin):
  """Plugin modeling factual mismatch or contradicting evidence across files."""

  SPEC = ConflictPluginSpec(
      name="evidence_mismatch",
      summary=(
          "Contradicting evidence across sources (logs vs charts, invoice vs "
          "receipt) that must be reconciled by presenting the correct source to "
          "the deciding subagent."
      ),
      category="inconsistency",
      hooks=("layer0", "layer1", "layer1b"),
  )

  @property
  def name(self) -> str:
    return "evidence_mismatch"

  def get_layer0_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "Plan phases covering: 1) Discovery of conflicting evidence across different sources (e.g. logs vs charts, invoices vs receipts), "
        "and 2) Reconciling the discrepancy by presenting the correct source to the relevant agent."
    )

  def get_layer1_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "Model a subagent making decisions or claims based on one source of information, "
        "and refusing to correct their action until the solver presents contradicting evidence from another source."
    )

  def get_layer1b_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "Create conflicting data files in the workspace (e.g., a transaction record listing one amount, "
        "and a visual invoice image listing a different amount, or a config file mismatched with log files)."
    )
