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

"""Circular deadlock approval/dependency conflict plugin.

Mechanized on the ``approval`` family. The conflict is "two approval legs each
wait on the other, so the grant can never clear on the normal path", and the
approval skill already models exactly that shape and both of its escape
hatches: ``approval escalate --resource <r> --to <authority>`` kicks a stuck
leg upward, ``approval provisional --resource <r> --approver <leg>`` records a
deadlock-breaking stand-in approval (it emits the same ``approval_recorded``
effect a real approval does, so it genuinely satisfies the precondition chain),
and ``approval grant --resource <r>`` is refused until every required leg has
cleared one way or the other.

Binding the plugin to that family turns the deadlock from narrative colour into
something the effect stream can be graded on: the terminal ``approval_granted``
effect exists only if the solver actually broke the cycle, because the skill
itself blocks the grant while a leg is still outstanding.

Only seeds that expose the ``approval`` family pick the mechanic up; for the
others (``_mechanic_command_sets`` intersects with ``allowed_tool_families``)
the declaration is inert and the plugin stays narrative-only.
"""

from clasharness.scenario.conflict_vlm_spec import ConflictVLMSeed
from clasharness.scenario.plugins.base import BaseConflictPlugin
from clasharness.scenario.plugins.registry import register_conflict_plugin
from clasharness.scenario.plugins.spec import ConflictPluginSpec


@register_conflict_plugin
class DeadlockDependencyPlugin(BaseConflictPlugin):
  """Plugin modeling circular deadlock approval dependency chains."""

  SPEC = ConflictPluginSpec(
      name="deadlock_dependency",
      summary=(
          "A circular approval dependency (A waits on B while B waits on A) "
          "that the solver must break via escalation, provisional bypass, or "
          "manual override."
      ),
      category="coordination",
      skill_families=("approval",),
      emits_effects=("approval_granted",),
      hooks=("layer0", "layer1", "layer1b", "trajectory", "evals"),
  )

  @property
  def name(self) -> str:
    return "deadlock_dependency"

  def get_layer0_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "Plan phases covering: 1) Detection of circular approval deadlock between subagents, "
        "and 2) Breaking the circular block via escalation, provisional bypass, or manual override."
    )

  def get_layer1_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "Model wait/dependency behavior for multiple participating subagents: each must refuse to proceed, "
        "citing that they are waiting for another participant agent's approval or review, creating a circular loop."
    )

  def get_layer1b_instruction(self, seed: ConflictVLMSeed) -> str:
    base = (
        "Embed circular reference links, pending ticket statuses, or emails in the workspace showing "
        "Agent A waiting for Agent B, while Agent B is waiting for Agent A."
    )
    if not self.mechanic_available(seed):
      # See DependencyOrderingPlugin.get_layer1b_instruction: narrative-only
      # seeds stay byte-identical to the pre-mechanic baseline.
      return base
    return base + (
        " The artifacts must name each approval leg explicitly and state which authority a stuck "
        "leg escalates to, so the cycle and its exit are readable from the workspace rather than "
        "guessable."
    )

  def get_trajectory_instruction(self, seed: ConflictVLMSeed) -> str:
    if not self.mechanic_available(seed):
      return ""
    return (
        "- DEADLOCK CHALLENGE: The trajectory opens the request with `approval "
        "request --resource <target> --approvers <legA>,<legB>`, records the "
        "leg that CAN clear normally with `approval approve --resource "
        "<target> --approver <legA>`, and then breaks the cycle on the stuck "
        "leg -- either `approval escalate --resource <target> --to "
        "<authority>` followed by the authority's approval, or `approval "
        "provisional --resource <target> --approver <legB>` -- before running "
        "`approval grant --resource <target>`. The coordinator MUST NOT run "
        "`approval grant` while the deadlocked leg is still outstanding: the "
        "skill refuses it and no `approval_granted` effect is emitted, which "
        "is precisely the un-broken-deadlock failure this round grades "
        "against."
    )

  def get_evals_instruction(self, seed: ConflictVLMSeed) -> str:
    if not self.mechanic_available(seed):
      return ""
    return (
        "- EVALS: Add an effect check asserting the grant completed "
        "(`approval_granted`) on the correct target, and that the deadlocked "
        "leg was actually resolved -- escalated or covered by a provisional "
        "approval -- rather than the cycle being narrated and then ignored."
    )
