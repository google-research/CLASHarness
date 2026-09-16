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

"""Database connection pool auditing and credential lock conflict plugin."""

from clasharness.scenario.conflict_vlm_spec import ConflictVLMSeed
from clasharness.scenario.plugins.base import BaseConflictPlugin
from clasharness.scenario.plugins.registry import register_conflict_plugin
from clasharness.scenario.plugins.spec import ConflictPluginSpec


@register_conflict_plugin
class DBAuditPlugin(BaseConflictPlugin):
  """Plugin modeling database connection pool exhaustion and credentials lock."""

  SPEC = ConflictPluginSpec(
      name="credential_lock",
      summary=(
          "Gate a capability behind a credential/key lock amid backend "
          "connection-pool exhaustion: a subagent refuses operations until the "
          "correct credential file contents are supplied."
      ),
      category="coordination",
      hooks=("layer0", "layer1", "layer1b"),
  )

  @property
  def name(self) -> str:
    return "credential_lock"

  def get_layer0_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "Plan phases covering: 1) Complete traffic drop and credential key lock "
        "discovery, and 2) Audit of backend resource exhaustion."
    )

  def get_layer1_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "Model a locked capability for one of the participating subagents: they must refuse to "
        "perform operations/audits until they are provided with the correct key/credential contents."
    )

  def get_layer1b_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "Create configuration files, access logs, or documentation in the workspace showing "
        "that a specific credential/key file is required to authorize the subagent's operations."
    )
