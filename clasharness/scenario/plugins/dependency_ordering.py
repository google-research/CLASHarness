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

"""Dependency ordering and prerequisite verification conflict plugin.

Mechanized on the ``deploy`` family. The conflict is "a downstream action is
blocked until every prerequisite has cleared", and the deploy skill already
models exactly that shape: ``deploy gate-set --gate <prerequisite> --state
ready`` records a prerequisite clearing, ``deploy status`` reads the board, and
``deploy cutover`` is refused while ANY gate is still pending. Binding the
plugin to that family turns the ordering constraint from narrative colour into
something the effect stream can be graded on -- the terminal
``deployment_cutover`` effect exists only if the ordering was respected,
because the skill itself blocks the premature path.

Only seeds that expose the ``deploy`` family pick the mechanic up; for the
others (``_mechanic_command_sets`` intersects with ``allowed_tool_families``)
the declaration is inert and the plugin stays narrative-only.
"""

from clasharness.scenario.conflict_vlm_spec import ConflictVLMSeed
from clasharness.scenario.plugins.base import BaseConflictPlugin
from clasharness.scenario.plugins.registry import register_conflict_plugin
from clasharness.scenario.plugins.spec import ConflictPluginSpec


@register_conflict_plugin
class DependencyOrderingPlugin(BaseConflictPlugin):
  """Plugin modeling sequential dependency and blocked downstream execution."""

  SPEC = ConflictPluginSpec(
      name="dependency_ordering",
      summary=(
          "A downstream subagent blocks its task until shown proof that a "
          "skipped prerequisite step (approval, risk assessment) was completed, "
          "enforcing correct step ordering."
      ),
      category="order",
      skill_families=("deploy",),
      emits_effects=("deployment_cutover",),
      hooks=("layer0", "layer1", "layer1b", "trajectory", "evals"),
  )

  @property
  def name(self) -> str:
    return "dependency_ordering"

  def get_layer0_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "Plan phases covering: 1) Identification of blocked downstream action due to missing prerequisite, "
        "and 2) Fulfilling/verifying the skipped prerequisite step to unblock the workflow."
    )

  def get_layer1_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "Model a blocking behavior for one of the downstream subagents: they must refuse to execute "
        "their task (e.g. granting access, deploying code) and block the solver until presented with proof "
        "that the prerequisite step (e.g. manager approval, risk assessment) has been completed."
    )

  def get_layer1b_instruction(self, seed: ConflictVLMSeed) -> str:
    base = (
        "Create checklist files, onboarding logs, or gate status configurations in the workspace "
        "indicating that earlier workflow steps are pending, incomplete, or bypassed."
    )
    if not self.mechanic_available(seed):
      # Narrative-only seeds must read byte-identically to the pre-mechanic
      # baseline: several of them are deliberately-preserved control arms, and
      # a prompt edit there would contaminate the comparison this mechanic is
      # supposed to be measured against.
      return base
    return base + (
        " The board must name every prerequisite explicitly and show which one is still "
        "outstanding, so the ordering constraint is readable rather than guessable."
    )

  def get_trajectory_instruction(self, seed: ConflictVLMSeed) -> str:
    if not self.mechanic_available(seed):
      return ""
    return (
        "- DEPENDENCY ORDERING CHALLENGE: The trajectory opens the downstream "
        "action with `deploy request --change <action>`, records each "
        "prerequisite the moment it clears with `deploy gate-set --gate "
        "<prerequisite> --state ready` (one gate per prerequisite on the "
        "checklist), re-reads the board with `deploy status`, and only once "
        "EVERY prerequisite gate is ready runs `deploy cutover --change "
        "<action>`. The coordinator MUST NOT run `deploy cutover` while any "
        "prerequisite is still pending: the skill refuses it and no "
        "`deployment_cutover` effect is emitted, which is precisely the "
        "out-of-order failure this round grades against."
    )

  def get_evals_instruction(self, seed: ConflictVLMSeed) -> str:
    if not self.mechanic_available(seed):
      return ""
    return (
        "- EVALS: Add an effect check asserting the downstream action completed "
        "(`deployment_cutover`) on the correct target, and that it landed only "
        "after every prerequisite gate had been recorded ready -- i.e. the "
        "ordering constraint held rather than the blocked step being forced "
        "early."
    )
