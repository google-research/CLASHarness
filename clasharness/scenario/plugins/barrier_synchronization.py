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

"""Barrier synchronization for deployment cutover conflict plugin (#15).

Steers the generator to build a deployment guarded by several readiness gates
that turn ready at different phases (driven by background events / subagents).
The cutover is a BARRIER: it is only valid once every gate is ready at the same
time, so a premature cutover is the failure mode. The trajectory/evals hooks
push the generator to emit ``deploy gate-set``/``deploy cutover`` steps, which
the checker scaffolder turns into a ``deployment_cutover`` effect check.
"""

from clasharness.scenario.conflict_vlm_spec import ConflictVLMSeed
from clasharness.scenario.plugins.base import BaseConflictPlugin
from clasharness.scenario.plugins.registry import register_conflict_plugin
from clasharness.scenario.plugins.spec import ConflictPluginSpec


@register_conflict_plugin
class BarrierSynchronizationPlugin(BaseConflictPlugin):
  """Plugin modeling an all-gates-ready barrier before deployment cutover."""

  SPEC = ConflictPluginSpec(
      name="barrier_synchronization",
      summary=(
          "Deployment cutover is a barrier: valid only once every independent "
          "readiness gate is simultaneously ready; a premature cutover while any "
          "gate is pending is the failure mode."
      ),
      category="coordination",
      skill_families=("deploy",),
      emits_effects=("deployment_cutover",),
      hooks=("layer0", "layer1", "layer1b", "trajectory", "evals"),
  )

  @property
  def name(self) -> str:
    return "barrier_synchronization"

  def get_layer0_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "Plan phases covering: 1) Discovery of a deployment gated by several "
        "independent readiness gates (e.g. canary health, data backfill, "
        "dependency version, on-call sign-off) that reach the ready state at "
        "DIFFERENT phases, with at least one gate lagging behind; and 2) "
        "Synchronizing on the barrier so the cutover happens only after every gate "
        "is simultaneously ready, never while any gate is still pending."
    )

  def get_layer1_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "Model subagents (or background events) that flip individual gates to "
        "ready at staggered times, and a stakeholder pressing for an early "
        "cutover. One gate must remain pending until a late phase, so cutting over "
        "before it clears is the tempting-but-wrong action the scenario tests."
    )

  def get_layer1b_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "Create workspace files showing the gate board and deployment record: a "
        "readiness-gate status dashboard/config with gates in mixed states (some "
        "ready, some pending), the change/deployment request, and release-runbook "
        "notes stating cutover requires ALL gates green at once."
    )

  def get_trajectory_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "- BARRIER SYNCHRONIZATION CHALLENGE: As each gate clears, the trajectory "
        "records it with `deploy gate-set --gate <gate> --state ready`, verifies "
        "the full board with `deploy status`, and only once EVERY gate is ready "
        "runs `deploy cutover --change <change>`. The coordinator MUST NOT call "
        "`deploy cutover` while any gate is still pending (a premature cutover is "
        "blocked and is the failure the round grades against)."
    )

  def get_evals_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "- EVALS: Add an effect check asserting the cutover happened "
        "(`deployment_cutover`) and that it occurred only after all readiness gates "
        "were set ready, i.e. the barrier held and no cutover was attempted while a "
        "gate was pending."
    )
