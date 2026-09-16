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

"""Distributed lock for a shared resource conflict plugin (#13).

Steers the generator to build a scenario around a single shared resource that
multiple actors want to edit, where the coordinator must acquire an exclusive
``lock`` before editing and release it afterwards so no concurrent write is
overwritten. The trajectory/evals hooks push the generator to emit ``lock
acquire``/``lock release`` steps, which the checker scaffolder turns into
``lock_acquired``/``lock_released`` effect checks.
"""

from clasharness.scenario.conflict_vlm_spec import ConflictVLMSeed
from clasharness.scenario.plugins.base import BaseConflictPlugin
from clasharness.scenario.plugins.registry import register_conflict_plugin
from clasharness.scenario.plugins.spec import ConflictPluginSpec


@register_conflict_plugin
class DistributedLockPlugin(BaseConflictPlugin):
  """Plugin modeling exclusive access to a shared resource via acquire/release."""

  SPEC = ConflictPluginSpec(
      name="distributed_lock",
      summary=(
          "Serialize concurrent writers on one shared resource by acquiring an "
          "exclusive lock before editing and releasing it after, preventing a "
          "stale-cache lost update."
      ),
      category="coordination",
      skill_families=("lock",),
      emits_effects=("lock_acquired", "lock_released"),
      hooks=("layer0", "layer1", "layer1b", "trajectory", "evals"),
  )

  @property
  def name(self) -> str:
    return "distributed_lock"

  def get_layer0_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "Plan phases covering: 1) Discovery that a single shared resource (e.g. a "
        "shared config, ledger, inventory record, or firmware slot) is being "
        "edited concurrently by more than one actor, with a stale-cache write that "
        "risks silently overwriting a peer's change (a lost update); and 2) "
        "Serializing access by acquiring an exclusive lock on that resource before "
        "editing and releasing it afterwards, so the two writes no longer collide."
    )

  def get_layer1_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "Model a subagent (or background event) that also intends to edit the same "
        "shared resource from a stale local cache. It must hold or contend for the "
        "resource until the coordinator has taken the lock, so that editing without "
        "first acquiring the lock would clobber the peer's in-flight change."
    )

  def get_layer1b_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "Create workspace files showing the shared resource and its access state: "
        "the resource file itself, a lock/ownership registry or status file, and "
        "log/diagram evidence of a prior lost-update (an overwrite that dropped a "
        "concurrent edit) to justify why acquire-before-edit is required."
    )

  def get_trajectory_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "- DISTRIBUTED LOCK CHALLENGE: Before editing the shared resource the "
        "trajectory MUST acquire the exclusive lock with `lock acquire --resource "
        "<resource>` (check `lock status --resource <resource>` first), perform the "
        "edit only while holding it, and then run `lock release --resource "
        "<resource>` once the edit is written. The coordinator MUST NOT edit the "
        "resource while another holder owns the lock."
    )

  def get_evals_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "- EVALS: Add an effect check asserting the exclusive lock was acquired "
        "(`lock_acquired`) before the shared-resource edit and released "
        "(`lock_released`) afterwards, plus a check that the coordinator's edit "
        "preserved (did not overwrite) the concurrent holder's change."
    )
