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

"""Reversible-action hold with a confirmation deadline conflict plugin (M3, #8).

Steers the generator to build a scarce resource that must first be *held*
(a reversible, tentative action) and then *confirmed* before a short time window
closes. The hold auto-expires: if the coordinator waits past the TTL and only
then tries to confirm, the confirmation fails and the tentative hold silently
reverses. This is the ``reversible_action_hold`` mechanic — it pairs with the
``time_window_expiration`` framing on the same seed (the expiry is *why* the
reversible hold matters).

The trajectory/evals hooks push the generator to emit ``booking hold`` →
(watch the clock) → ``booking confirm`` steps before the window closes, which
the checker scaffolder turns into a ``booking_confirmed`` effect check. The
tempting-but-wrong action is confirming (or waiting) after the window has
already elapsed.
"""

from clasharness.scenario.conflict_vlm_spec import ConflictVLMSeed
from clasharness.scenario.plugins.base import BaseConflictPlugin
from clasharness.scenario.plugins.registry import register_conflict_plugin
from clasharness.scenario.plugins.spec import ConflictPluginSpec


@register_conflict_plugin
class ReversibleActionHoldPlugin(BaseConflictPlugin):
  """Plugin modeling a reversible hold that must be confirmed before it expires."""

  SPEC = ConflictPluginSpec(
      name="reversible_action_hold",
      summary=(
          "A scarce resource is taken with a reversible, tentative hold that "
          "must be confirmed before a short time window closes; an unconfirmed "
          "hold auto-expires (reverses), so confirming after the window has "
          "elapsed is the failure mode."
      ),
      category="timeout",
      skill_families=("booking",),
      emits_effects=("booking_held", "booking_confirmed", "booking_released"),
      hooks=("layer0", "layer1", "layer1b", "trajectory", "evals"),
  )

  @property
  def name(self) -> str:
    return "reversible_action_hold"

  def get_layer0_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "Plan phases covering: 1) Discovery of a scarce resource that can only "
        "be taken with a reversible, tentative HOLD (not an immediate commit), "
        "where the hold is valid only for a short window and auto-expires if "
        "left unconfirmed; and 2) Confirming (or deliberately releasing) the "
        "hold before that window closes, so the tentative action is made "
        "permanent in time rather than silently reversing."
    )

  def get_layer1_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "Model a subagent or background event that owns the expiration clock: "
        "it grants the hold with an explicit TTL, warns as the window nears its "
        "close, and refuses to honor a confirmation once the hold has expired. "
        "A stakeholder should press the coordinator to attend to other work "
        "first, tempting it to let the window lapse before confirming."
    )

  def get_layer1b_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "Create workspace files showing the tentative hold and its deadline: a "
        "reservation/hold record with an explicit expiration (e.g. valid for "
        "only 300 seconds from when it was placed), and runbook notes stating "
        "that a hold is only provisional until confirmed and that an expired "
        "hold is automatically released."
    )

  def get_trajectory_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "- REVERSIBLE-HOLD DEADLINE CHALLENGE: The trajectory takes the resource "
        "with `booking hold --resource <r> --ttl <n>`, tracks the remaining "
        "window (advancing virtual time only with `wait <seconds>` when it must), "
        "and runs `booking confirm --resource <r>` BEFORE the TTL elapses. The "
        "coordinator MUST NOT let the window lapse and then confirm: a confirm "
        "after expiry fails and emits no `booking_confirmed` effect (the hold has "
        "reversed). If the plan changes, it releases the hold with "
        "`booking release --resource <r>` rather than abandoning it."
    )

  def get_evals_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "- EVALS: Add an effect check asserting the hold was confirmed within its "
        "window (`booking_confirmed`) and that the confirmation followed a "
        "`booking_held` on the same resource, i.e. the reversible hold was made "
        "permanent before the time window closed rather than being confirmed "
        "after it had already expired."
    )
