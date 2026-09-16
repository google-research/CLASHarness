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

"""Unit tests for the deterministic Protocol Prerequisite Resolver (C4).

Two layers are covered:

1. *Construction* — ``_inject_prerequisites_for_effects`` /
   ``_build_prerequisite_commands`` build the correct, machine-derived
   prerequisite command lines for each shipped multi-step protocol (booking,
   pairing, deploy execute/cutover), driven solely by the skill SSOT
   (``CommandSpec.requires``); chain-head effects (lock) and identifier-less
   terminals construct nothing.

2. *Orchestration* — ``_resolve_protocol_prerequisites`` only rewrites a round
   when replay shows a failing ``effect`` check AND the rewrite strictly reduces
   effect failures without regressing any other check (commit gate). A round with
   no failing effect check is left byte-identical (diversity/no-regression
   invariant); a rewrite that does not improve, or that regresses another check,
   is reverted and left to the F1 LLM backstop.

Replay is mocked (the real end-to-end effect-check behavior is exercised by the
generation pipeline itself); these tests pin the deterministic construction and
the safety gate.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from clasharness.scenario.conflict_vlm_materializer import (
    _INJECTED_PREREQ_TTL,
    _build_prerequisite_commands,
    _calibrate_checks_against_state,
    _calibrate_files_created_contains,
    _failing_effect_types,
    _failing_indices,
    _first_command_index,
    _gate_names_before,
    _inject_prerequisites_for_effects,
    _is_files_created_contains,
    _protocol_contract_for_failed_checks,
    _rederive_contains_value,
    _resolve_protocol_prerequisites,
)
from clasharness.scenario.conflict_llm_pipeline import command_spec_for_effect


_RESOLVE_MOD = "clasharness.scenario.conflict_vlm_materializer"


class TestPrerequisiteConstruction(unittest.TestCase):
  """The prerequisite command lines are derived correctly from the SSOT."""

  def test_booking_missing_hold_inserts_matching_hold(self):
    traj = ["booking confirm --resource R1 --holder alice", "DONE"]
    out = _inject_prerequisites_for_effects(traj, {"booking_confirmed"})
    self.assertEqual(
        out,
        [
            f"booking hold --resource R1 --holder alice --ttl {_INJECTED_PREREQ_TTL}",
            "booking confirm --resource R1 --holder alice",
            "DONE",
        ],
    )

  def test_booking_hold_copies_only_provided_args(self):
    # No --holder on confirm -> injected hold carries none either.
    traj = ["booking confirm --resource R1", "DONE"]
    out = _inject_prerequisites_for_effects(traj, {"booking_confirmed"})
    self.assertEqual(
        out[0],
        f"booking hold --resource R1 --ttl {_INJECTED_PREREQ_TTL}",
    )

  def test_pairing_inserts_ordered_chain_same_device(self):
    traj = ["pairing confirm --device phone-1", "DONE"]
    out = _inject_prerequisites_for_effects(traj, {"pairing_completed"})
    self.assertEqual(
        out,
        [
            f"pairing start --device phone-1 --ttl {_INJECTED_PREREQ_TTL}",
            "pairing show-code --device phone-1",
            "pairing enter-code --device phone-1",
            "pairing confirm --device phone-1",
            "DONE",
        ],
    )
    # enter-code must NOT carry a --code (an omitted code is always accepted;
    # a guessed one would risk a mismatch against the derived session code).
    self.assertNotIn("--code", out[2])

  def test_deploy_execute_inserts_request(self):
    traj = ["deploy execute --change c9", "DONE"]
    out = _inject_prerequisites_for_effects(traj, {"deployment_executed"})
    self.assertEqual(
        out, ["deploy request --change c9", "deploy execute --change c9", "DONE"]
    )

  def test_deploy_cutover_flips_existing_gate_ready(self):
    traj = [
        "deploy gate-set --gate g1 --state blocked",
        "deploy cutover --change c9",
        "DONE",
    ]
    out = _inject_prerequisites_for_effects(traj, {"deployment_cutover"})
    # request + a ready flip for the seen gate are inserted before cutover.
    self.assertIn("deploy request --change c9", out)
    self.assertIn("deploy gate-set --gate g1 --state ready", out)
    self.assertLess(
        out.index("deploy gate-set --gate g1 --state ready"),
        out.index("deploy cutover --change c9"),
    )

  def test_deploy_cutover_seeds_a_gate_when_none(self):
    traj = ["deploy cutover --change c9", "DONE"]
    out = _inject_prerequisites_for_effects(traj, {"deployment_cutover"})
    self.assertIn("deploy request --change c9", out)
    self.assertTrue(
        any(c.startswith("deploy gate-set") and c.endswith("--state ready")
            for c in out),
        out,
    )

  def test_chain_head_effect_constructs_nothing(self):
    # lock.acquire is a chain head (no requires) -> nothing to inject.
    traj = ["lock acquire --resource db --holder a", "DONE"]
    out = _inject_prerequisites_for_effects(traj, {"lock_acquired"})
    self.assertEqual(out, traj)

  def test_missing_identifier_constructs_nothing(self):
    traj = ["booking confirm", "DONE"]
    out = _inject_prerequisites_for_effects(traj, {"booking_confirmed"})
    self.assertEqual(out, traj)

  def test_unknown_effect_constructs_nothing(self):
    traj = ["booking confirm --resource R1", "DONE"]
    out = _inject_prerequisites_for_effects(traj, {"not_a_real_effect"})
    self.assertEqual(out, traj)

  def test_build_prerequisite_commands_direct(self):
    spec = command_spec_for_effect("booking_confirmed")
    out = _build_prerequisite_commands(
        "booking confirm --resource R2 --holder bob", spec, []
    )
    self.assertEqual(
        out, [f"booking hold --resource R2 --holder bob --ttl {_INJECTED_PREREQ_TTL}"]
    )


class TestSmallHelpers(unittest.TestCase):

  def test_first_command_index(self):
    cmds = ["file read --path x", "booking confirm --resource R", "DONE"]
    self.assertEqual(_first_command_index(cmds, "booking confirm"), 1)
    self.assertIsNone(_first_command_index(cmds, "pairing confirm"))

  def test_gate_names_before(self):
    cmds = [
        "deploy gate-set --gate a --state ready",
        "deploy gate-set --gate b --state blocked",
        "deploy cutover --change c",
    ]
    self.assertEqual(
        _gate_names_before(cmds, 2, "deploy gate-set", "--gate"), ["a", "b"]
    )
    # A gate-set after the index is ignored.
    self.assertEqual(
        _gate_names_before(cmds, 1, "deploy gate-set", "--gate"), ["a"]
    )

  def test_failing_helpers(self):
    checks = [
        {"type": "effect", "effect_type": "booking_confirmed"},
        {"type": "file_check", "path": "x"},
        {"type": "effect", "effect_type": "deployment_executed"},
    ]
    details = [
        SimpleNamespace(passed=False),
        SimpleNamespace(passed=True),
        SimpleNamespace(passed=False),
    ]
    idx = _failing_indices(checks, details)
    self.assertEqual(idx, {0, 2})
    self.assertEqual(
        _failing_effect_types(checks, idx),
        ["booking_confirmed", "deployment_executed"],
    )


def _round(task_id, checks, trajectory):
  """A minimal round_spec whose task_spec carries a real ground_truth."""
  return SimpleNamespace(
      task_id=task_id,
      expert_trajectory=list(trajectory),
      task_spec={
          "task_id": task_id,
          "instruction": "x",
          "domains": ["composite"],
          "difficulty": 2,
          "initial_state": "base",
          "ground_truth": {
              "solution_commands": list(trajectory),
              "evaluation_checks": checks,
          },
      },
  )


def _details(*passes):
  return SimpleNamespace(details=[SimpleNamespace(passed=p) for p in passes])


class TestResolverOrchestration(unittest.TestCase):
  """Commit gate, byte-identical guarantee, and revert behavior."""

  def test_repairs_and_commits_when_effect_check_flips_green(self):
    checks = [{"type": "effect", "effect_type": "booking_confirmed",
               "condition": "exists"}]
    traj = ["booking confirm --resource R1 --holder alice", "DONE"]
    round_spec = _round("r1", checks, traj)
    spec = SimpleNamespace(rounds=[round_spec])
    # 1st replay: effect fails. 2nd replay (after injection): passes.
    with patch(f"{_RESOLVE_MOD}._replay_trajectory",
               side_effect=[(_details(False), "", {}), (_details(True), "", {})]), \
         patch(f"{_RESOLVE_MOD}.save_task") as save_mock:
      n = _resolve_protocol_prerequisites(spec, Path("/tmp/x"), lambda _m: None)
    self.assertEqual(n, 1)
    self.assertEqual(
        round_spec.expert_trajectory,
        [
            f"booking hold --resource R1 --holder alice --ttl {_INJECTED_PREREQ_TTL}",
            "booking confirm --resource R1 --holder alice",
            "DONE",
        ],
    )
    # ground_truth.solution_commands and task_spec are updated in lockstep.
    self.assertEqual(
        round_spec.task_spec["ground_truth"]["solution_commands"],
        round_spec.expert_trajectory,
    )
    save_mock.assert_called_once()

  def test_passing_round_left_byte_identical_and_not_saved(self):
    checks = [{"type": "effect", "effect_type": "booking_confirmed",
               "condition": "exists"}]
    traj = ["booking hold --resource R1", "booking confirm --resource R1", "DONE"]
    round_spec = _round("r2", checks, traj)
    spec = SimpleNamespace(rounds=[round_spec])
    original = list(traj)
    # Effect check already passes -> resolver must not re-replay or save.
    with patch(f"{_RESOLVE_MOD}._replay_trajectory",
               side_effect=[(_details(True), "", {})]) as replay_mock, \
         patch(f"{_RESOLVE_MOD}.save_task") as save_mock:
      n = _resolve_protocol_prerequisites(spec, Path("/tmp/x"), lambda _m: None)
    self.assertEqual(n, 0)
    self.assertEqual(round_spec.expert_trajectory, original)  # byte-identical
    self.assertEqual(replay_mock.call_count, 1)  # no re-replay
    save_mock.assert_not_called()

  def test_reverts_when_injection_does_not_improve(self):
    checks = [{"type": "effect", "effect_type": "booking_confirmed",
               "condition": "exists"}]
    traj = ["booking confirm --resource R1", "DONE"]
    round_spec = _round("r3", checks, traj)
    spec = SimpleNamespace(rounds=[round_spec])
    original = list(traj)
    # Both replays fail -> no improvement -> revert, no save.
    with patch(f"{_RESOLVE_MOD}._replay_trajectory",
               side_effect=[(_details(False), "", {}), (_details(False), "", {})]), \
         patch(f"{_RESOLVE_MOD}.save_task") as save_mock:
      n = _resolve_protocol_prerequisites(spec, Path("/tmp/x"), lambda _m: None)
    self.assertEqual(n, 0)
    self.assertEqual(round_spec.expert_trajectory, original)
    save_mock.assert_not_called()

  def test_reverts_when_injection_regresses_another_check(self):
    # Effect fixed but a previously-passing file_check now fails -> after-failures
    # are NOT a subset of before-failures -> revert.
    checks = [
        {"type": "effect", "effect_type": "booking_confirmed", "condition": "exists"},
        {"type": "file_check", "path": "reports/out.md"},
    ]
    traj = ["booking confirm --resource R1", "DONE"]
    round_spec = _round("r4", checks, traj)
    spec = SimpleNamespace(rounds=[round_spec])
    original = list(traj)
    with patch(f"{_RESOLVE_MOD}._replay_trajectory",
               side_effect=[
                   (_details(False, True), "", {}),   # effect fails, file passes
                   (_details(True, False), "", {}),   # effect passes, file regressed
               ]), \
         patch(f"{_RESOLVE_MOD}.save_task") as save_mock:
      n = _resolve_protocol_prerequisites(spec, Path("/tmp/x"), lambda _m: None)
    self.assertEqual(n, 0)
    self.assertEqual(round_spec.expert_trajectory, original)
    save_mock.assert_not_called()

  def test_non_effect_failures_are_ignored_by_resolver(self):
    # Only a file_check fails (no effect check) -> resolver does nothing (that is
    # C2/F1 territory), leaving the round byte-identical.
    checks = [{"type": "file_check", "path": "reports/out.md"}]
    traj = ["booking confirm --resource R1", "DONE"]
    round_spec = _round("r5", checks, traj)
    spec = SimpleNamespace(rounds=[round_spec])
    original = list(traj)
    with patch(f"{_RESOLVE_MOD}._replay_trajectory",
               side_effect=[(_details(False), "", {})]) as replay_mock, \
         patch(f"{_RESOLVE_MOD}.save_task") as save_mock:
      n = _resolve_protocol_prerequisites(spec, Path("/tmp/x"), lambda _m: None)
    self.assertEqual(n, 0)
    self.assertEqual(round_spec.expert_trajectory, original)
    self.assertEqual(replay_mock.call_count, 1)
    save_mock.assert_not_called()


class TestRepairContractForFailedChecks(unittest.TestCase):
  """B3: the F1 repair payload carries a targeted, SSOT-derived contract."""

  def test_contract_built_for_failing_effect_checks(self):
    failed = [
        {"check": {"type": "effect", "effect_type": "booking_confirmed"}},
        {"check": {"type": "effect", "effect_type": "deployment_cutover"}},
    ]
    out = _protocol_contract_for_failed_checks(failed)
    by_effect = {e["effect_type"]: e for e in out}
    self.assertEqual(
        by_effect["booking_confirmed"]["terminal_command"], "booking confirm"
    )
    self.assertEqual(
        by_effect["booking_confirmed"]["preconditions"][0]["ordered_prerequisites"],
        ["booking hold"],
    )
    self.assertEqual(
        by_effect["booking_confirmed"]["preconditions"][0]["matching_arg"],
        "--resource",
    )
    # cutover carries the all-ready quantifier for gates.
    self.assertEqual(
        by_effect["deployment_cutover"]["preconditions"][0]["all_ready"],
        {"command": "deploy gate-set", "arg": "--state", "value": "ready"},
    )

  def test_pairing_contract_is_full_ordered_chain(self):
    out = _protocol_contract_for_failed_checks(
        [{"check": {"type": "effect", "effect_type": "pairing_completed"}}]
    )
    self.assertEqual(
        out[0]["preconditions"][0]["ordered_prerequisites"],
        ["pairing start", "pairing show-code", "pairing enter-code"],
    )

  def test_chain_head_effect_yields_no_contract(self):
    # lock.acquire has no requires -> nothing to tell the repair LLM.
    out = _protocol_contract_for_failed_checks(
        [{"check": {"type": "effect", "effect_type": "lock_acquired"}}]
    )
    self.assertEqual(out, [])

  def test_non_effect_failures_yield_no_contract(self):
    out = _protocol_contract_for_failed_checks(
        [{"check": {"type": "file_check", "path": "reports/out.md"}}]
    )
    self.assertEqual(out, [])

  def test_duplicate_effects_are_deduped(self):
    failed = [
        {"check": {"type": "effect", "effect_type": "booking_confirmed"}},
        {"check": {"type": "effect", "effect_type": "booking_confirmed"}},
    ]
    out = _protocol_contract_for_failed_checks(failed)
    self.assertEqual(len(out), 1)


class TestRederiveContainsValue(unittest.TestCase):
  """C2 content-check re-derivation tolerates only insignificant separator drift."""

  def _assert_runtime_match(self, value, content):
    # Mirror EffectChecker's run-time files_created field_contains grading.
    self.assertIn(str(value).lower(), str(content).lower())

  def test_json_quote_drift_is_rederived(self):
    # The real barrier_sync_deployment_v01_r07 case.
    out = _rederive_contains_value('"Status": GO', '{\n  "Status": "GO"\n}')
    self.assertEqual(out, '"Status": "GO"')
    self._assert_runtime_match(out, '{\n  "Status": "GO"\n}')

  def test_missing_space_after_colon_is_rederived(self):
    out = _rederive_contains_value("Status: GO", '{"Status":"GO"}')
    self.assertIsNotNone(out)
    self._assert_runtime_match(out, '{"Status":"GO"}')

  def test_collapsed_whitespace_is_rederived(self):
    out = _rederive_contains_value("deploy   cutover", "ran deploy cutover now")
    self.assertEqual(out, "deploy cutover")

  def test_markdown_emphasis_drift_is_rederived(self):
    out = _rederive_contains_value("**Decision:** APPROVE", "Decision: APPROVE now")
    self.assertEqual(out, "Decision: APPROVE")

  def test_case_insensitive(self):
    out = _rederive_contains_value("APPROVE", "final status: approve")
    self.assertEqual(out, "approve")

  def test_genuinely_absent_value_returns_none(self):
    self.assertIsNone(_rederive_contains_value("REJECT", "the request was approved"))

  def test_empty_needle_returns_none(self):
    self.assertIsNone(_rederive_contains_value("", "anything"))
    self.assertIsNone(_rederive_contains_value('"" ', "anything"))

  def test_does_not_cross_non_separator_characters(self):
    # Whitespace/quotes/markdown may be skipped; letters/digits/other punctuation
    # may NOT - so a coincidental cross-token match cannot weaken grading.
    self.assertIsNone(_rederive_contains_value("go live", "we go DIE then live"))
    self.assertIsNone(_rederive_contains_value('"result": "PASS"', "result = PASS"))

  def test_rederived_value_always_satisfies_runtime_match(self):
    for needle, content in [
        ('"x": 1', '{"x": 1}'),
        ("A B", "A     B"),
        ("`code`", "the code path"),
    ]:
      out = _rederive_contains_value(needle, content)
      if out is not None:
        self._assert_runtime_match(out, content)


class TestIsFilesCreatedContains(unittest.TestCase):

  def test_true_for_files_created_field_contains(self):
    self.assertTrue(_is_files_created_contains({
        "type": "effect", "effect_type": "files_created",
        "condition": "field_contains"}))

  def test_false_for_other_effects_or_conditions(self):
    self.assertFalse(_is_files_created_contains({
        "type": "effect", "effect_type": "messages_sent",
        "condition": "field_contains"}))
    self.assertFalse(_is_files_created_contains({
        "type": "effect", "effect_type": "files_created",
        "condition": "exists"}))
    self.assertFalse(_is_files_created_contains(
        {"type": "file_check", "path": "x"}))
    self.assertFalse(_is_files_created_contains("not-a-dict"))


def _files_created_state(path, content):
  return {"effects": {"files_created": [{"path": path, "content": content}]}}


class TestCalibrateFilesCreatedContains(unittest.TestCase):

  def _check(self, value, **extra):
    chk = {"type": "effect", "effect_type": "files_created",
           "condition": "field_contains",
           "expected": {"field": "content", "value": value}}
    chk.update(extra)
    return chk

  def test_rederives_and_preserves_identity_without_mutating_original(self):
    chk = self._check('"Status": GO', name="cutover_report", weight=2.0)
    state = _files_created_state("reports/x.md", '{"Status": "GO"}')
    out = _calibrate_files_created_contains(chk, state, lambda _m: None, "t")
    self.assertIsNotNone(out)
    self.assertEqual(out["expected"]["value"], '"Status": "GO"')
    self.assertEqual(out["condition"], "field_contains")  # not degraded
    self.assertEqual(out["name"], "cutover_report")
    self.assertEqual(out["weight"], 2.0)
    # Original check untouched.
    self.assertEqual(chk["expected"]["value"], '"Status": GO')

  def test_already_matching_returns_none(self):
    # Nothing to fix (raw value already present) -> leave untouched.
    chk = self._check('"Status": "GO"')
    state = _files_created_state("reports/x.md", '{"Status": "GO"}')
    self.assertIsNone(
        _calibrate_files_created_contains(chk, state, lambda _m: None, "t"))

  def test_absent_value_degrades_to_exists_when_file_created(self):
    chk = self._check("REJECT", name="n")
    state = _files_created_state("reports/x.md", '{"Status": "GO"}')
    out = _calibrate_files_created_contains(chk, state, lambda _m: None, "t")
    self.assertEqual(out["condition"], "exists")
    self.assertEqual(out["name"], "n")

  def test_absent_value_and_no_file_returns_none(self):
    chk = self._check("REJECT")
    state = {"effects": {"files_created": []}}
    self.assertIsNone(
        _calibrate_files_created_contains(chk, state, lambda _m: None, "t"))

  def test_empty_needle_returns_none(self):
    chk = self._check("")
    state = _files_created_state("reports/x.md", "anything")
    self.assertIsNone(
        _calibrate_files_created_contains(chk, state, lambda _m: None, "t"))

  def test_non_content_field_is_honored(self):
    # field="path" must be read from the effect's ``path``, not ``content``; the
    # value needs drift correction (space after slash) so it isn't short-circuited
    # by the already-present guard.
    chk = self._check("reports/ GO.md")
    chk["expected"]["field"] = "path"
    state = {"effects": {"files_created": [
        {"path": "reports/GO.md", "content": "no match here"}]}}
    out = _calibrate_files_created_contains(chk, state, lambda _m: None, "t")
    self.assertIsNotNone(out)
    self.assertEqual(out["expected"]["value"], "reports/GO.md")


class TestCalibrateChecksAgainstStateEffects(unittest.TestCase):
  """The C2 pass (via mocked replay) re-derives failing effect content checks,
  leaves passing checks byte-identical, and still degrades doomed file_checks."""

  def _effect_check(self, value):
    return {"type": "effect", "effect_type": "files_created",
            "condition": "field_contains",
            "expected": {"field": "content", "value": value}}

  def test_rederives_failing_effect_content_check_and_saves(self):
    checks = [self._effect_check('"Status": GO')]
    round_spec = _round("rc1", checks, ["file create --path reports/x.md", "DONE"])
    spec = SimpleNamespace(rounds=[round_spec])
    state = _files_created_state("reports/x.md", '{"Status": "GO"}')
    with patch(f"{_RESOLVE_MOD}._replay_trajectory",
               return_value=(_details(False), "", state)), \
         patch(f"{_RESOLVE_MOD}.save_task") as save_mock:
      n = _calibrate_checks_against_state(spec, Path("/tmp/x"), lambda _m: None)
    self.assertEqual(n, 1)
    new = round_spec.task_spec["ground_truth"]["evaluation_checks"][0]
    self.assertEqual(new["expected"]["value"], '"Status": "GO"')
    save_mock.assert_called_once()

  def test_passing_effect_check_left_untouched_and_not_saved(self):
    checks = [self._effect_check('"Status": "GO"')]
    round_spec = _round("rc2", checks, ["file create --path reports/x.md", "DONE"])
    spec = SimpleNamespace(rounds=[round_spec])
    state = _files_created_state("reports/x.md", '{"Status": "GO"}')
    with patch(f"{_RESOLVE_MOD}._replay_trajectory",
               return_value=(_details(True), "", state)), \
         patch(f"{_RESOLVE_MOD}.save_task") as save_mock:
      n = _calibrate_checks_against_state(spec, Path("/tmp/x"), lambda _m: None)
    self.assertEqual(n, 0)
    self.assertEqual(
        round_spec.task_spec["ground_truth"]["evaluation_checks"], checks)
    save_mock.assert_not_called()

  def test_file_check_degrade_path_still_works(self):
    # Regression for the refactored loop: a failing content file_check whose file
    # exists in the resolved state degrades to an existence check.
    checks = [{"type": "file_check", "path": "reports/out.md",
               "condition": "contains", "expected": "GO", "weight": 3.0}]
    round_spec = _round("rc3", checks, ["file create --path reports/out.md", "DONE"])
    spec = SimpleNamespace(rounds=[round_spec])
    # FileChecker normalizes paths to a leading slash; mirror that in the state.
    state = {"files": {"/reports/out.md": "some other body"}}
    with patch(f"{_RESOLVE_MOD}._replay_trajectory",
               return_value=(_details(False), "", state)), \
         patch(f"{_RESOLVE_MOD}.save_task") as save_mock:
      n = _calibrate_checks_against_state(spec, Path("/tmp/x"), lambda _m: None)
    self.assertEqual(n, 1)
    new = round_spec.task_spec["ground_truth"]["evaluation_checks"][0]
    self.assertEqual(new.get("condition", "exists"), "exists")
    self.assertNotIn("expected", new)
    self.assertEqual(new["weight"], 3.0)
    save_mock.assert_called_once()


if __name__ == "__main__":
  unittest.main()
