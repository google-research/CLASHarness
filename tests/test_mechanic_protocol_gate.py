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

"""Tests for the conflict-mechanic protocol gate.

The gate exists because of one measured gap: across the 300 corpus rounds of the
10 seeds that route to a mechanic plugin, 92.3% touch a mechanic command family
but only 29.7% actually run the terminal command that emits the effect the
plugin declares it grades. The other 62.6% only *read* the protocol, so the
plugin's assertion has nothing to assert against.

Two tiers, and the split is the whole design:

  ADVANCE  -- most selected rounds must run *some* registered protocol command.
              Now satisfiable by a prerequisite step (``approval request``,
              ``deploy gate-set``, ``pairing enter-code``) because those emit
              their own effects, which lifts the reachable rate from 29.7% to
              42.0% on the existing corpus.
  COMPLETE -- the scenario must run a terminal command at least ONCE. Enforced
              as a scenario-level invariant tracked across rounds, not pinned to
              a fixed round: 7 of the 10 mechanic seeds have exactly one
              distinct terminal command line across all 30 of their rounds, so
              demanding completion every selected round would force absurd
              repetition or fail generation outright.

The properties below are the ones that, if broken, make the gate either
worthless (it stops requiring the protocol) or destructive (it discards
scenarios it cannot possibly get).
"""

import json
from pathlib import Path
import unittest

from clasharness.scenario.conflict_llm_pipeline import MECHANIC_ROUND_RATIO
from clasharness.scenario.conflict_llm_pipeline import MechanicCommands
from clasharness.scenario.conflict_llm_pipeline import _commands_include
from clasharness.scenario.conflict_llm_pipeline import _final_mechanic_round
from clasharness.scenario.conflict_llm_pipeline import _mechanic_command_sets
from clasharness.scenario.conflict_llm_pipeline import _round_requires_mechanic
from clasharness.scenario.plugins import get_plugins_for_taxonomy

_SEEDS_DIR = (
    # No `.resolve()`: under the build system `__file__` is a symlink into the
    # CAS, and resolving it walks out of the runfiles tree.
    Path(__file__).parents[1]
    / "clasharness"
    / "data"
    / "scenario_seeds"
    / "conflict_vlm_v1"
)

# Deliberately unhandled control arms: these must keep routing to no mechanic
# plugin so the gate is provably a no-op for them and they stay comparable
# against the pre-change corpus.
_CONTROL_SEEDS = frozenset({
    "clinical_trial_dosage_conflict",
    "payment_chargeback_dispute",
    "sla_support_response",
})


def _load_seeds():
  out = {}
  for path in sorted(_SEEDS_DIR.glob("*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    out[data.get("slug") or path.stem] = data
  return out


def _mech_for(seed: dict) -> MechanicCommands:
  active = (seed.get("conflict_taxonomy") or []) + (
      seed.get("difficulty_challenges") or []
  )
  return _mechanic_command_sets(
      get_plugins_for_taxonomy(active), seed.get("allowed_tool_families") or []
  )


class RoundSelectionTest(unittest.TestCase):
  """The round allocation must be deterministic and evenly spread."""

  def test_ten_rounds_selects_the_documented_set(self):
    selected = [i for i in range(1, 11) if _round_requires_mechanic(i)]
    self.assertEqual([2, 3, 5, 6, 8, 9], selected)

  def test_selection_is_deterministic(self):
    first = [_round_requires_mechanic(i) for i in range(1, 21)]
    second = [_round_requires_mechanic(i) for i in range(1, 21)]
    self.assertEqual(first, second)

  def test_selection_hits_the_target_ratio(self):
    selected = sum(1 for i in range(1, 31) if _round_requires_mechanic(i))
    self.assertEqual(int(30 * MECHANIC_ROUND_RATIO), selected)

  def test_no_three_consecutive_rounds_selected(self):
    """Investigation rounds must survive; the protocol must not clump."""
    flags = [_round_requires_mechanic(i) for i in range(1, 31)]
    for i in range(len(flags) - 2):
      self.assertFalse(
          flags[i] and flags[i + 1] and flags[i + 2],
          msg=f"three consecutive selected rounds at {i + 1}",
      )

  def test_degenerate_inputs_select_nothing_or_everything(self):
    self.assertFalse(_round_requires_mechanic(0))
    self.assertFalse(_round_requires_mechanic(-1))
    self.assertFalse(_round_requires_mechanic(5, ratio=0.0))
    self.assertTrue(_round_requires_mechanic(5, ratio=1.0))

  def test_final_mechanic_round_is_the_last_selected_round(self):
    self.assertEqual(9, _final_mechanic_round(10))
    for target in range(2, 25):
      final = _final_mechanic_round(target)
      self.assertIsNotNone(final)
      self.assertTrue(_round_requires_mechanic(final))
      self.assertLessEqual(final, target)
      # Nothing after it is selected.
      for later in range(final + 1, target + 1):
        self.assertFalse(_round_requires_mechanic(later))

  def test_final_mechanic_round_is_none_when_nothing_is_selected(self):
    self.assertIsNone(_final_mechanic_round(0))
    self.assertIsNone(_final_mechanic_round(10, ratio=0.0))


class CommandsIncludeTest(unittest.TestCase):
  """Prefix matching over authored command lines."""

  def _cmds(self, *lines):
    return [{"command": line} for line in lines]

  def test_matches_command_with_arguments(self):
    self.assertTrue(
        _commands_include(
            self._cmds("approval grant --request REQ-1"), ("approval grant",)
        )
    )

  def test_is_case_insensitive_and_ignores_surrounding_space(self):
    self.assertTrue(
        _commands_include(self._cmds("  APPROVAL GRANT x"), ("approval grant",))
    )

  def test_status_only_trajectory_does_not_satisfy_the_gate(self):
    """The entire point: reading the protocol is not performing it."""
    self.assertFalse(
        _commands_include(
            self._cmds("approval status", "approval list", "file read a.txt"),
            ("approval grant",),
        )
    )

  def test_empty_requirement_is_vacuously_satisfied(self):
    """No mechanic => no requirement => never blocks generation."""
    self.assertTrue(_commands_include(self._cmds("anything"), ()))
    self.assertTrue(_commands_include([], ()))

  def test_empty_commands_fail_a_real_requirement(self):
    self.assertFalse(_commands_include([], ("approval grant",)))

  def test_malformed_entries_do_not_raise(self):
    self.assertFalse(
        _commands_include([None, {}, {"command": None}], ("approval grant",))
    )


class MechanicCommandSetsTest(unittest.TestCase):
  """The vocabulary must be derived from the SSOT, never hand-listed."""

  @classmethod
  def setUpClass(cls):
    cls.seeds = _load_seeds()
    cls.mech = {slug: _mech_for(s) for slug, s in cls.seeds.items()}

  def test_no_plugins_yields_an_empty_falsey_result(self):
    empty = _mechanic_command_sets([], ["approval"])
    self.assertFalse(empty)
    self.assertEqual((), empty.terminals)
    self.assertEqual((), empty.protocol)

  def test_control_seeds_route_to_no_mechanic(self):
    """These are the untouched comparison arms; the gate must not reach them."""
    for slug in _CONTROL_SEEDS:
      self.assertIn(slug, self.mech)
      self.assertFalse(
          self.mech[slug],
          msg=f"{slug} must stay a no-op control arm",
      )

  def test_exactly_the_expected_seeds_have_a_mechanic(self):
    with_mech = {s for s, m in self.mech.items() if m}
    self.assertEqual(
        {
            "access_request_approval",
            "barrier_sync_deployment",
            "collaborative_contract_review",
            "coordinated_containment",
            # Mechanized when the three narrative plugins were bound to the
            # skill families their seeds already expose: deadlock_dependency ->
            # approval (escalate/provisional/grant) and dependency_ordering ->
            # deploy (gate-set/status/cutover). Both matched the seed's own
            # ``core_conflict`` wording; time_window_expiration -> booking /
            # pairing added hooks but no new seed, since those families were
            # already reachable via other plugins.
            "deadlock_multi_approval",
            "distributed_lock_shared_resource",
            "employee_onboarding_workflow",
            "expiring_resource_hold",
            "it_deployment_gate",
            "multi_device_pairing",
            "multi_party_access_permit",
            "multi_tab_slot_monitoring",
        },
        with_mech,
    )

  def test_terminals_are_a_subset_of_protocol(self):
    """Completing must always also count as advancing, or the gates conflict."""
    for slug, m in self.mech.items():
      self.assertTrue(
          set(m.terminals).issubset(set(m.protocol)),
          msg=f"{slug}: terminals leaked outside protocol",
      )

  def test_prerequisite_steps_widen_protocol_beyond_terminals(self):
    """The supply fix: single-terminal seeds must gain advanceable commands.

    Without this the advance gate collapses back onto the terminal gate and
    every selected round would demand a repeat of the one terminal command.
    """
    for slug in (
        "access_request_approval",
        "barrier_sync_deployment",
        "collaborative_contract_review",
        "it_deployment_gate",
        "multi_device_pairing",
        "multi_party_access_permit",
    ):
      m = self.mech[slug]
      self.assertEqual(1, len(m.terminals), msg=f"{slug} precondition changed")
      self.assertGreater(
          len(m.protocol), len(m.terminals), msg=f"{slug} gained no prerequisites"
      )

  def test_read_only_commands_are_never_in_the_protocol_set(self):
    """A status command must not be able to satisfy the gate."""
    for slug, m in self.mech.items():
      for name in m.protocol:
        verb = name.split(" ", 1)[-1]
        self.assertNotIn(
            verb,
            {"status", "list", "show", "search", "monitor"},
            msg=f"{slug}: read-only '{name}' would satisfy the gate",
        )

  def test_protocol_commands_stay_inside_allowed_families(self):
    """A requirement the scenario cannot express would be unsatisfiable."""
    for slug, m in self.mech.items():
      allowed = set(self.seeds[slug].get("allowed_tool_families") or [])
      if not allowed:
        continue
      for name in m.protocol:
        self.assertIn(
            name.split(" ", 1)[0],
            allowed,
            msg=f"{slug}: '{name}' is outside allowed_tool_families",
        )

  def test_declared_effects_are_reported(self):
    for slug, m in self.mech.items():
      if m:
        self.assertTrue(m.effects, msg=f"{slug} has terminals but no effects")


if __name__ == "__main__":
  unittest.main()
