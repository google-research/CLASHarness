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

"""Coordinated multi-point containment conflict plugin (M4, #4).

Steers the generator to build a security-incident response that is only truly
resolved once **every one of several independent control points** has been
actioned. A compromise is not contained by a single dramatic action: the
account's live sessions must be revoked, its forwarding rules disabled, the
compromised device isolated, and the leaked token blocked. Skipping any one of
them leaves an open door, and the tempting-but-wrong move is to declare victory
after the most visible action while a quieter channel is still live.

The mechanic rides the ``containment`` family: each of the four verbs
(``revoke-sessions``, ``disable-forwarding``, ``isolate-device``,
``block-token``) actions one control point and emits a single
``containment_action`` effect the first time it lands. Each point is
idempotency-guarded, so re-running a verb is a no-op that emits nothing — the
effect count tops out at four, one per distinct point. The checker scaffolder
grades this with a ``count_gte(4)`` assertion on ``containment_action``: full
credit requires all four points contained, and an agent cannot fake it by
spamming a single verb.
"""

from clasharness.scenario.conflict_vlm_spec import ConflictVLMSeed
from clasharness.scenario.plugins.base import BaseConflictPlugin
from clasharness.scenario.plugins.registry import register_conflict_plugin
from clasharness.scenario.plugins.spec import ConflictPluginSpec


@register_conflict_plugin
class CoordinatedContainmentPlugin(BaseConflictPlugin):
  """Plugin modeling an incident contained only when all control points are actioned."""

  SPEC = ConflictPluginSpec(
      name="coordinated_containment",
      summary=(
          "A security compromise is contained only when every one of four "
          "independent control points is actioned (revoke sessions, disable "
          "forwarding, isolate device, block token); stopping after the most "
          "visible action while another channel stays live is the failure mode."
      ),
      category="subagent_collaboration",
      skill_families=("containment",),
      emits_effects=("containment_action",),
      hooks=("layer0", "layer1", "layer1b", "trajectory", "evals"),
  )

  @property
  def name(self) -> str:
    return "coordinated_containment"

  def get_layer0_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "Plan phases covering: 1) Discovery that a single compromise exposes "
        "MULTIPLE independent control points that must each be shut down "
        "(active sessions, forwarding rules, a device, and an access token); "
        "and 2) Coordinating a full containment that actions EVERY point, "
        "rather than stopping after the most visible one while another channel "
        "stays open."
    )

  def get_layer1_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "Model subagents/stakeholders who each surface a DIFFERENT exposed "
        "control point (one reports live sessions, another a forwarding rule, "
        "another a still-connected device, another a leaked token) and who "
        "arrive at different times, so a coordinator that acts on the first "
        "one or two and declares the incident closed misses the rest. A "
        "stakeholder should press for a quick 'all clear' after the loudest "
        "action, tempting a premature close."
    )

  def get_layer1b_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "Create workspace files that make the full control-point checklist "
        "legible: an incident record enumerating all four exposed points, "
        "per-point evidence (session list, forwarding config, device "
        "inventory, token audit), and a runbook stating that the incident may "
        "be marked contained ONLY once all four points have been actioned."
    )

  def get_trajectory_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "- COORDINATED CONTAINMENT CHALLENGE: The trajectory actions EVERY one "
        "of the four control points exactly once: `containment revoke-sessions "
        "--account <a>`, `containment disable-forwarding --account <a>`, "
        "`containment isolate-device --device <d>`, and `containment "
        "block-token --token <t>`. Each first action emits one "
        "`containment_action`; re-running a verb is a no-op that emits nothing, "
        "so the agent MUST cover all four distinct points rather than repeating "
        "one. The incident is contained only when all four have landed."
    )

  def get_evals_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "- EVALS: Add an effect check asserting that containment actions were "
        "taken across all four control points, i.e. a `count_gte`-style "
        "assertion that at least four distinct `containment_action` effects "
        "were emitted (one per point). A run that actioned only some points — "
        "or spammed a single verb — must NOT pass."
    )
