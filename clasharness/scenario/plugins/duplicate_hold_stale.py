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

"""Duplicate-hold avoidance under a stale multi-tab view plugin (M4, #6).

Steers the generator to build a scarce-slot booking watched across SEVERAL tabs
or views that refresh at different times, so what one tab shows as free may
already be taken. The coordinator must secure exactly ONE slot for the need —
placing a single hold, confirming it against the CURRENT state, and not leaving
redundant duplicate holds fanned out across tabs. The two tempting-but-wrong
moves are (1) holding several slots at once "to be safe" (a duplicate booking
that strands capacity), and (2) acting on a STALE tab that still shows a slot as
free when a peer has since grabbed it — so the hold conflicts or the confirm
lands on a slot that is no longer really available.

The mechanic rides the ``booking`` family: ``booking search`` / ``booking
status`` expose the CURRENT availability (the fresh view), ``booking hold
--resource <r>`` takes one slot (a live hold by another holder is refused, which
is exactly what a stale view walks into), and ``booking confirm --resource <r>``
commits the single chosen slot, emitting ``booking_confirmed``. Full credit
requires re-reading current availability before committing and confirming a
single, still-available slot rather than duplicating holds or trusting a stale
snapshot. This pairs with the ``temporal_staleness`` framing on the same seed
(the stale tab is *why* the duplicate hold is tempting).
"""

from clasharness.scenario.conflict_vlm_spec import ConflictVLMSeed
from clasharness.scenario.plugins.base import BaseConflictPlugin
from clasharness.scenario.plugins.registry import register_conflict_plugin
from clasharness.scenario.plugins.spec import ConflictPluginSpec


@register_conflict_plugin
class DuplicateHoldStalePlugin(BaseConflictPlugin):
  """Plugin modeling securing one slot without duplicate holds or a stale view."""

  SPEC = ConflictPluginSpec(
      name="duplicate_hold_stale",
      summary=(
          "A scarce slot is monitored across several tabs/views that refresh at "
          "different times; the coordinator must secure exactly one slot — a "
          "single confirmed hold against current availability — rather than "
          "fanning out duplicate holds or acting on a stale tab that still "
          "shows a slot free after a peer has taken it."
      ),
      category="subagent_collaboration",
      skill_families=("booking",),
      emits_effects=("booking_confirmed",),
      hooks=("layer0", "layer1", "layer1b", "trajectory", "evals"),
  )

  @property
  def name(self) -> str:
    return "duplicate_hold_stale"

  def get_layer0_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "Plan phases covering: 1) Discovery that a single scarce slot is being "
        "watched across MULTIPLE tabs/views that refresh at different times, so "
        "a slot shown as free in one tab may already be taken; and 2) Securing "
        "exactly ONE slot for the need — checking current availability, placing "
        "a single hold, and confirming it — rather than fanning out duplicate "
        "holds across tabs or committing based on a stale, out-of-date view."
    )

  def get_layer1_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "Model peers/subagents who are competing for the same scarce slots and "
        "who take slots between the coordinator's refreshes, so an older tab "
        "goes stale: a slot it still shows as free has actually been grabbed. A "
        "stakeholder should push the coordinator to 'just grab a few slots so we "
        "don't miss out' (tempting duplicate holds) or to book straight off the "
        "slot the visible-but-stale tab shows as open (tempting a stale-view "
        "commit that conflicts or strands a hold)."
    )

  def get_layer1b_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "Create workspace files that expose the multi-tab, time-stamped nature "
        "of the availability: per-tab snapshots each carrying their own refresh "
        "timestamp (so it is clear which view is newest), and a booking policy "
        "stating that exactly one slot should be secured for the need, that "
        "duplicate holds must be released, and that availability must be "
        "re-checked against the current state before committing."
    )

  def get_trajectory_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "- DUPLICATE-HOLD / STALE-VIEW CHALLENGE: The trajectory re-reads "
        "current availability (`booking search` / `booking status`) before "
        "committing, places a SINGLE hold on the chosen slot (`booking hold "
        "--resource <r>`), and confirms that one slot (`booking confirm "
        "--resource <r>`), emitting one `booking_confirmed`. The coordinator "
        "MUST NOT hold several slots at once (a duplicate hold strands "
        "capacity) and MUST NOT confirm off a stale tab — a slot a peer has "
        "already taken will refuse the hold or fail to confirm. If it ever "
        "placed a redundant hold, it releases it with `booking release "
        "--resource <r>` so exactly one slot is secured."
    )

  def get_evals_instruction(self, seed: ConflictVLMSeed) -> str:
    return (
        "- EVALS: Add an effect check asserting exactly one slot was secured "
        "against the current state, i.e. an `exists` assertion on a single "
        "`booking_confirmed` effect for the correct still-available slot. A run "
        "that confirmed a slot taken from a stale view, or that left duplicate "
        "holds fanned out across slots, must NOT pass."
    )
