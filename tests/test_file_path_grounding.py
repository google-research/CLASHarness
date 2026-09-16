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

"""Tests for file-path grounding of the reference (gold) trajectory.

WHY THIS EXISTS
---------------
In ``clasharness.backend.file_backend`` every file verb except ``create``
hard-fails on a path that is not already in the workspace: ``read``, ``append``,
``delete`` and ``move --src`` return ``File not found: <p>`` with
``exit_code=1`` and -- the part that actually breaks grading -- NO
``state_changes`` at all. ``create`` is the only verb that materializes a path.

So a gold step that appends to an invented path emits an EMPTY effect stream,
and every check the scaffolder derived from that step (``effect:files_created``
above all) becomes unsatisfiable by the reference solution itself. Confirmed by
replaying ``employee_onboarding_workflow_v01_r05``, whose gold appends to
``/logs/telemetry_verification.log`` -- a file that update ``u_r06`` only adds
one round LATER: exit 1, 0 files_created, 0 files_appended.

Measured over the 600 rounds of each corpus: v3 had 29 rounds (4.8%, reads only)
containing such a command, v4 had 61 (10.2%) -- 9 appends and 67 reads -- and
``effect:files_created`` gold-replay failures rose from 1 to 20 in step.

The three-stage response, and the property each stage must keep:

  ``_ensure_file_writes_materialize``  append -> create when the target does not
      exist. The single correct deterministic repair (no other verb can create
      the file) and it costs no retry.
  ``_validate_file_path_grounding``    hard gate feeding the steps retry loop
      while the budget lasts, so the author re-writes a coherent trajectory.
  ``_reground_missing_file_reads``     final-attempt escape only. Exhausting the
      3-attempt budget raises ``ScenarioLLMError`` and discards the whole
      SCENARIO, and 9.5% of v4 rounds across 14 of 60 seeds would enter the gate,
      so a pure-reject policy would trade a grading defect for a far larger loss
      of corpus diversity.

The tests below pin the properties that, if broken, make the chain either
worthless (paths stop being checked) or destructive (valid trajectories are
rejected, or an append silently clobbers a file it should have extended).
"""

from __future__ import annotations

import unittest

from clasharness.backend.file_backend import FileSystemBackend
from clasharness.scenario.conflict_llm_pipeline import _bind_orphan_updates_to_round
from clasharness.scenario.conflict_llm_pipeline import _collect_delayed_arrivals
from clasharness.scenario.conflict_llm_pipeline import _collect_mock_created_paths
from clasharness.scenario.conflict_llm_pipeline import _ensure_file_writes_materialize
from clasharness.scenario.conflict_llm_pipeline import _ensure_wait_before_delayed_reads
from clasharness.scenario.conflict_llm_pipeline import _file_subcommand
from clasharness.scenario.conflict_llm_pipeline import _format_delayed_arrival_directive
from clasharness.scenario.conflict_llm_pipeline import _reground_missing_file_reads
from clasharness.scenario.conflict_llm_pipeline import _validate_file_path_grounding
from clasharness.scenario.conflict_llm_pipeline import _wait_seconds


def _cmds(*lines: str) -> list[dict]:
  return [{"step_number": i, "command": c} for i, c in enumerate(lines, 1)]


def _lines(commands: list[dict]) -> list[str]:
  return [c["command"] for c in commands]


class TestBackendPathContract(unittest.TestCase):
  """The backend behaviour the whole gate is derived from.

  If someone ever makes ``file append`` create-on-missing, the gate stops being
  necessary and starts being a pure cost; if ``file create`` ever starts
  requiring an existing parent, the repair starts producing broken trajectories.
  Either way this file must be revisited, so pin both here rather than trusting
  a comment.
  """

  def _backend(self, tmp: str) -> FileSystemBackend:
    backend = FileSystemBackend()
    backend.initialize(tmp, {"OPENCLAW_STATE_DIR": tmp})
    return backend

  def test_append_to_missing_path_fails_and_emits_no_effect(self) -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
      backend = self._backend(tmp)
      res = backend.execute_cli(
          "file append --path /logs/nope.log --content 'x'"
      )
      self.assertEqual(res.exit_code, 1)
      self.assertIn("File not found", res.stderr)
      # The decisive property: no effect at all, so a files_created check
      # derived from this step can never pass.
      self.assertFalse(res.state_changes or {})

  def test_read_and_delete_to_missing_path_fail(self) -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
      backend = self._backend(tmp)
      for cmd in (
          "file read --path /logs/nope.log",
          "file delete --path /logs/nope.log",
      ):
        with self.subTest(cmd=cmd):
          res = backend.execute_cli(cmd)
          self.assertEqual(res.exit_code, 1)
          self.assertIn("File not found", res.stderr)

  def test_create_materializes_any_path_and_emits_files_created(self) -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
      backend = self._backend(tmp)
      res = backend.execute_cli(
          "file create --path /logs/deep/nested/new.log --content 'body'"
      )
      self.assertEqual(res.exit_code, 0)
      created = (res.state_changes or {}).get("files_created") or []
      self.assertEqual(len(created), 1)
      self.assertEqual(created[0]["content"], "body")
      # And it is readable afterwards -- which is what makes the append ->
      # create repair safe for a later read of the same path.
      self.assertEqual(
          backend.execute_cli(
              "file read --path /logs/deep/nested/new.log"
          ).exit_code,
          0,
      )


class TestFileSubcommandParsing(unittest.TestCase):

  def test_recognizes_file_commands_only(self) -> None:
    self.assertEqual(_file_subcommand("file read --path 'a.md'"), "read")
    self.assertEqual(_file_subcommand("  file   append --path a.md"), "append")
    self.assertEqual(_file_subcommand("message send --channel '#x'"), "")
    self.assertEqual(_file_subcommand("DONE"), "")
    self.assertEqual(_file_subcommand(""), "")

  def test_unbalanced_quotes_do_not_raise(self) -> None:
    # A malformed command must fall through to the tool-spec validator with its
    # clearer error, never blow up the grounding walk.
    self.assertEqual(_file_subcommand("file read --path 'unclosed"), "")


class TestEnsureFileWritesMaterialize(unittest.TestCase):

  def test_append_to_missing_path_becomes_create(self) -> None:
    cmds = _cmds(
        "file append --path 'reports/workpad.md' --content 'finding: 4.2%'",
        "DONE",
    )
    self.assertEqual(_ensure_file_writes_materialize(cmds, {"docs/policy.md"}), 1)
    self.assertEqual(
        _lines(cmds)[0],
        "file create --path 'reports/workpad.md' --content 'finding: 4.2%'",
    )

  def test_append_to_existing_path_is_left_alone(self) -> None:
    """Rewriting this would CLOBBER the file instead of extending it."""
    cmds = _cmds("file append --path 'logs/audit.log' --content 'more'", "DONE")
    self.assertEqual(_ensure_file_writes_materialize(cmds, {"logs/audit.log"}), 0)
    self.assertEqual(
        _lines(cmds)[0], "file append --path 'logs/audit.log' --content 'more'"
    )

  def test_append_after_own_create_is_left_alone(self) -> None:
    cmds = _cmds(
        "file create --path 'reports/wp.md' --content 'head'",
        "file append --path 'reports/wp.md' --content 'tail'",
        "DONE",
    )
    self.assertEqual(_ensure_file_writes_materialize(cmds, set()), 0)

  def test_leading_slash_is_normalized(self) -> None:
    """The corpus mixes '/logs/x.log' and 'logs/x.log' for the same file."""
    cmds = _cmds("file append --path '/logs/audit.log' --content 'more'", "DONE")
    self.assertEqual(_ensure_file_writes_materialize(cmds, {"logs/audit.log"}), 0)

  def test_content_containing_the_word_append_is_not_corrupted(self) -> None:
    cmds = _cmds(
        "file append --path 'r/w.md' --content 'we append the audit trail'",
        "DONE",
    )
    _ensure_file_writes_materialize(cmds, set())
    self.assertEqual(
        _lines(cmds)[0],
        "file create --path 'r/w.md' --content 'we append the audit trail'",
    )

  def test_repair_is_idempotent(self) -> None:
    cmds = _cmds("file append --path 'r/w.md' --content 'x'", "DONE")
    _ensure_file_writes_materialize(cmds, set())
    self.assertEqual(_ensure_file_writes_materialize(cmds, set()), 0)


class TestValidateFilePathGrounding(unittest.TestCase):

  def test_accepts_paths_present_in_the_workspace(self) -> None:
    _validate_file_path_grounding(
        _cmds("file read --path 'docs/policy.md'", "DONE"), {"docs/policy.md"}
    )

  def test_accepts_a_path_the_trajectory_created_earlier(self) -> None:
    _validate_file_path_grounding(
        _cmds(
            "file create --path 'reports/out.md' --content 'v'",
            "file read --path 'reports/out.md'",
            "DONE",
        ),
        set(),
    )

  def test_rejects_read_of_an_invented_path(self) -> None:
    with self.assertRaises(ValueError) as ctx:
      _validate_file_path_grounding(
          _cmds("file read --path 'logs/telemetry_verification.log'", "DONE"),
          {"logs/onboarding_audit.log"},
      )
    msg = str(ctx.exception)
    # The retry feedback has to be actionable: name the step, name the offending
    # path, and list what does exist.
    self.assertIn("step 1", msg)
    self.assertIn("logs/telemetry_verification.log", msg)
    self.assertIn("logs/onboarding_audit.log", msg)

  def test_rejects_append_delete_and_move_src_of_an_invented_path(self) -> None:
    for cmd in (
        "file append --path 'x/nope.md' --content 'c'",
        "file delete --path 'x/nope.md'",
        "file move --src 'x/nope.md' --dst 'x/ok.md'",
    ):
      with self.subTest(cmd=cmd):
        with self.assertRaises(ValueError):
          _validate_file_path_grounding(_cmds(cmd, "DONE"), {"x/real.md"})

  def test_read_after_own_delete_is_rejected(self) -> None:
    """The walk must apply each command's own effect, not just creations."""
    with self.assertRaises(ValueError):
      _validate_file_path_grounding(
          _cmds(
              "file delete --path 'docs/policy.md'",
              "file read --path 'docs/policy.md'",
              "DONE",
          ),
          {"docs/policy.md"},
      )

  def test_move_makes_the_destination_readable(self) -> None:
    _validate_file_path_grounding(
        _cmds(
            "file move --src 'a.md' --dst 'b.md'",
            "file read --path 'b.md'",
            "DONE",
        ),
        {"a.md"},
    )

  def test_non_file_commands_are_ignored(self) -> None:
    _validate_file_path_grounding(
        _cmds(
            "message send --channel '#ops' --message 'see /logs/whatever.log'",
            "asset open --id 'dash_01'",
            "file list",
            "wait 600",
            "DONE",
        ),
        set(),
    )


class TestRegroundMissingFileReads(unittest.TestCase):

  def test_no_op_on_a_clean_trajectory(self) -> None:
    cmds = _cmds("file read --path 'docs/policy.md'", "DONE")
    before = _lines(cmds)
    self.assertEqual(_reground_missing_file_reads(cmds, ["docs/policy.md"]), 0)
    self.assertEqual(_lines(cmds), before)

  def test_invented_read_is_repointed_at_the_last_written_file(self) -> None:
    cmds = _cmds(
        "file create --path 'reports/out.md' --content 'v'",
        "file read --path 'reports/invented.md'",
        "DONE",
    )
    self.assertEqual(_reground_missing_file_reads(cmds, ["docs/policy.md"]), 1)
    self.assertEqual(_lines(cmds)[1], "file read --path reports/out.md")

  def test_falls_back_to_the_newest_workspace_file(self) -> None:
    cmds = _cmds("file read --path 'reports/invented.md'", "DONE")
    _reground_missing_file_reads(cmds, ["docs/policy.md", "logs/newest.log"])
    self.assertEqual(_lines(cmds)[0], "file read --path logs/newest.log")

  def test_unrepairable_delete_is_dropped_and_steps_renumbered(self) -> None:
    cmds = _cmds(
        "file read --path 'docs/policy.md'",
        "file delete --path 'x/nope.md'",
        "DONE",
    )
    self.assertEqual(_reground_missing_file_reads(cmds, ["docs/policy.md"]), 1)
    self.assertEqual(len(cmds), 2)
    self.assertEqual([c["step_number"] for c in cmds], [1, 2])

  def test_terminal_done_survives_regrounding(self) -> None:
    cmds = _cmds(
        "file create --path 'r/o.md' --content 'v'",
        "file read --path 'r/invented.md'",
        "DONE",
    )
    _reground_missing_file_reads(cmds, ["docs/policy.md"])
    self.assertEqual(_lines(cmds)[-1], "DONE")

  def test_full_chain_always_converges(self) -> None:
    """The escape must leave NOTHING for the gate: it is the last line."""
    cmds = _cmds(
        "file append --path 'r/wp.md' --content 'a'",
        "file read --path 'r/wp.md'",
        "file read --path 'totally/invented.md'",
        "file delete --path 'also/invented.md'",
        "file move --src 'still/invented.md' --dst 'z.md'",
        "DONE",
    )
    ordered = ["docs/policy.md"]
    _ensure_file_writes_materialize(cmds, set(ordered))
    _reground_missing_file_reads(cmds, ordered)
    _validate_file_path_grounding(cmds, set(ordered))  # must not raise
    self.assertEqual(_lines(cmds)[-1], "DONE")


class TestCollectMockCreatedPaths(unittest.TestCase):
  """A gold may read a file its own delegated sub-agent wrote.

  Mock ``state_effects`` execute for real, so counting them keeps the gate from
  rejecting that trajectory. On v3/v4 this set never rescues a violation (0 of
  29 / 0 of 61), so it is false-rejection insurance, not a loophole.
  """

  def test_collects_create_and_move_destinations(self) -> None:
    narrative = {
        "rounds": [
            {
                "mock_subagents": [
                    {
                        "state_effects": [
                            "file create --path '/logs/subagent.log' --content 'x'",
                            "file move --src 'a.md' --dst 'b.md'",
                            "file read --path 'docs/policy.md'",
                            "message send --channel '#ops' --message 'hi'",
                        ]
                    }
                ]
            }
        ]
    }
    self.assertEqual(
        _collect_mock_created_paths(narrative), {"logs/subagent.log", "b.md"}
    )

  def test_includes_prior_rounds(self) -> None:
    prior = [
        {
            "mock_subagents": [
                {"state_effects": ["file create --path 'old/from_r01.log'"]}
            ]
        }
    ]
    self.assertIn(
        "old/from_r01.log",
        _collect_mock_created_paths({"rounds": []}, prior_rounds=prior),
    )

  def test_tolerates_malformed_shapes(self) -> None:
    self.assertEqual(_collect_mock_created_paths({}), set())
    self.assertEqual(
        _collect_mock_created_paths({"rounds": [{"mock_subagents": ["junk"]}]}),
        set(),
    )



# ---------------------------------------------------------------------------
# Mid-round arrivals (the timing arm).
#
# A round's own ``workspace_adds`` never reach the task's
# ``initial_state_overrides``. ``conflict_vlm_materializer`` (:2915) compiles
# them into ``background_events`` of ``type: file_update`` carrying the update's
# ``trigger_virtual_time``, and ``OpenClawEnvironment`` fires an event only once
# ``trigger_time <= virtual_time`` (:896) on a clock that advances SOLELY inside
# ``wait`` (:887). So the file is real, is listed in ``valid_paths``, and still
# hard-fails a step-1 read exactly like an invented path.
#
# This is the dominant grounding defect and it predates every fix in this file:
# of the unreadable gold reads, 25 of 29 in v3, 60 of 76 in v4 and 41 of 51 in
# v5 were this, not a hallucinated path. The flat "these paths exist" set could
# not express it, which is why the whole class survived the v4/v5 work.
# ---------------------------------------------------------------------------


class TestWaitSeconds(unittest.TestCase):
  """The clock parser. Everything below is only as correct as this is."""

  def test_parses_a_wait(self) -> None:
    self.assertEqual(_wait_seconds("wait 600"), 600)

  def test_parses_a_float_valued_wait(self) -> None:
    """The corpus contains ``wait 3600.0``; truncating beats returning 0."""
    self.assertEqual(_wait_seconds("wait 3600.0"), 3600)

  def test_non_wait_commands_do_not_move_the_clock(self) -> None:
    for cmd in (
        "file read --path 'docs/policy.md'",
        "message send --channel '#ops' --message 'wait 600'",
        "DONE",
        "",
    ):
      with self.subTest(cmd=cmd):
        self.assertEqual(_wait_seconds(cmd), 0)

  def test_malformed_waits_are_zero_not_a_crash(self) -> None:
    for cmd in ("wait", "wait abc", "wait -5", "wait 'unbalanced"):
      with self.subTest(cmd=cmd):
        self.assertEqual(_wait_seconds(cmd), 0)


def _upd(uid: str, trigger: int, *paths: str) -> dict:
  return {
      "update_id": uid,
      "trigger_virtual_time": trigger,
      "workspace_adds": [{"path": p} for p in paths],
  }


def _payload(*updates: dict, declared: list[str] | None = None) -> dict:
  """A round-scoped narrative payload declaring its updates, as layer 2 does."""
  ids = (
      declared
      if declared is not None
      else [str(u.get("update_id")) for u in updates]
  )
  return {"rounds": [{"round_id": "r01", "update_ids": ids}], "updates": list(updates)}


class TestCollectDelayedArrivals(unittest.TestCase):

  def test_extracts_path_and_trigger(self) -> None:
    self.assertEqual(
        _collect_delayed_arrivals(
            _payload(_upd("u_r01_01", 15, "logs/panel_gate_check.log")), set()
        ),
        {"logs/panel_gate_check.log": 15},
    )

  def test_a_path_readable_from_step_one_is_not_delayed(self) -> None:
    """A round rewriting an existing file must not be told to wait for it.

    The earlier content is in ``initial_state_overrides`` and opens fine at
    step 1; only the REFRESH is deferred. Treating it as delayed would inject a
    pointless wait and make the gate reject correct trajectories.
    """
    self.assertEqual(
        _collect_delayed_arrivals(
            _payload(_upd("u_r01_01", 30, "logs/a.log")), {"logs/a.log"}
        ),
        {},
    )

  def test_earliest_trigger_wins_when_two_updates_write_one_path(self) -> None:
    """Readable as soon as the FIRST event fires; a later one cannot un-create."""
    self.assertEqual(
        _collect_delayed_arrivals(
            _payload(_upd("u_r01_01", 90, "x.md"), _upd("u_r01_02", 20, "x.md")),
            set(),
        ),
        {"x.md": 20},
    )

  def test_missing_trigger_is_treated_as_time_zero(self) -> None:
    payload = _payload(
        {"update_id": "u_r01_01", "workspace_adds": [{"path": "x.md"}]}
    )
    self.assertEqual(_collect_delayed_arrivals(payload, set()), {"x.md": 0})

  def test_leading_slash_is_normalized(self) -> None:
    """The corpus mixes '/logs/x.log' and 'logs/x.log' for one file."""
    payload = _payload(_upd("u_r01_01", 5, "/logs/x.log"))
    self.assertEqual(_collect_delayed_arrivals(payload, set()), {"logs/x.log": 5})
    self.assertEqual(_collect_delayed_arrivals(payload, {"logs/x.log"}), {})

  # -- orphan scoping ------------------------------------------------------
  #
  # 131 of v5's 600 rounds declare NO update_ids and 175 of 644 updates are
  # bound to no round at all. The materializer walks ``round_spec.update_ids``,
  # so an undeclared update is generated, written into the spec, and then never
  # compiled into a background event: its files never arrive, at any time.

  def test_an_undeclared_update_is_not_a_delayed_arrival(self) -> None:
    """It never arrives, so a wait would buy nothing and the gate must keep
    treating a read of it as a plain violation."""
    self.assertEqual(
        _collect_delayed_arrivals(
            _payload(_upd("u_r01_01", 15, "logs/orphan.log"), declared=[]), set()
        ),
        {},
    )

  def test_a_round_declaring_only_its_first_update_orphans_the_second(self) -> None:
    """The measured shape: layer 2 is prompted for exactly ``u_<round>_01`` and
    layer 3 sometimes emits ``_02`` as well."""
    self.assertEqual(
        _collect_delayed_arrivals(
            _payload(
                _upd("u_r01_01", 15, "logs/shipped.log"),
                _upd("u_r01_02", 20, "logs/orphan.log"),
                declared=["u_r01_01"],
            ),
            set(),
        ),
        {"logs/shipped.log": 15},
    )

  def test_a_cross_round_declaration_is_honoured(self) -> None:
    """8 of v5's 469 bindings name another round's id; the id is not the
    predicate, the declaration is."""
    self.assertEqual(
        _collect_delayed_arrivals(
            _payload(_upd("u_r07_01", 15, "logs/x.log"), declared=["u_r07_01"]),
            set(),
        ),
        {"logs/x.log": 15},
    )

  def test_an_update_without_an_id_is_never_admitted(self) -> None:
    payload = {
        "rounds": [{"round_id": "r01", "update_ids": ["u_r01_01"]}],
        "updates": [{"trigger_virtual_time": 15, "workspace_adds": [{"path": "x.md"}]}],
    }
    self.assertEqual(_collect_delayed_arrivals(payload, set()), {})

  def test_tolerates_malformed_shapes(self) -> None:
    for payload in (
        {},
        {"updates": None},
        {"updates": ["junk"]},
        {"rounds": ["junk"], "updates": [_upd("u_r01_01", 5, "x.md")]},
        {"rounds": None, "updates": [_upd("u_r01_01", 5, "x.md")]},
        _payload({"update_id": "u_r01_01", "workspace_adds": None}),
        _payload({"update_id": "u_r01_01", "workspace_adds": ["junk"]}),
        _payload({"update_id": "u_r01_01", "workspace_adds": [{"path": ""}]}),
        _payload(
            {
                "update_id": "u_r01_01",
                "trigger_virtual_time": "soon",
                "workspace_adds": [{}],
            }
        ),
    ):
      with self.subTest(payload=payload):
        self.assertEqual(_collect_delayed_arrivals(payload, set()), {})


class TestFormatDelayedArrivalDirective(unittest.TestCase):

  def test_empty_when_nothing_is_delayed(self) -> None:
    """Must add no text at all, or every round pays for a rare case."""
    self.assertEqual(_format_delayed_arrival_directive({}), "")

  def test_names_each_path_its_time_and_the_wait_verb(self) -> None:
    out = _format_delayed_arrival_directive({"logs/b.log": 30, "logs/a.log": 5})
    self.assertIn("logs/a.log", out)
    self.assertIn("logs/b.log", out)
    self.assertIn("30", out)
    self.assertIn("wait", out)
    # Ordered by arrival time so the author reads it as a timeline.
    self.assertLess(out.index("logs/a.log"), out.index("logs/b.log"))


class TestValidateFilePathGroundingTiming(unittest.TestCase):

  ARRIVALS = {"logs/panel_gate_check.log": 15}
  EXISTING = {"docs/policy.md", "logs/panel_gate_check.log"}

  def _read(self, *lines: str) -> None:
    _validate_file_path_grounding(_cmds(*lines), self.EXISTING, self.ARRIVALS)

  def test_reading_a_delayed_file_at_step_one_is_rejected(self) -> None:
    """The v5 defect, verbatim: multi_tab_slot_monitoring_v01_r02."""
    with self.assertRaises(ValueError) as ctx:
      self._read("file read --path 'logs/panel_gate_check.log'", "DONE")
    msg = str(ctx.exception)
    self.assertIn("timing", msg)
    self.assertIn("logs/panel_gate_check.log", msg)
    self.assertIn("15", msg)
    self.assertIn("wait", msg)

  def test_reading_it_after_a_sufficient_wait_is_accepted(self) -> None:
    self._read(
        "wait 3600", "file read --path 'logs/panel_gate_check.log'", "DONE"
    )

  def test_an_insufficient_wait_is_still_rejected(self) -> None:
    with self.assertRaises(ValueError):
      self._read("wait 5", "file read --path 'logs/panel_gate_check.log'", "DONE")

  def test_waits_accumulate_across_steps(self) -> None:
    """``environment.py`` adds to ``_virtual_time``; it does not set it."""
    self._read(
        "wait 8",
        "asset open --id 'dash_01'",
        "wait 7",
        "file read --path 'logs/panel_gate_check.log'",
        "DONE",
    )

  def test_a_wait_exactly_equal_to_the_trigger_fires_the_event(self) -> None:
    """``due = trigger_time <= virtual_time`` -- inclusive. Off-by-one here
    would reject a correct trajectory."""
    self._read("wait 15", "file read --path 'logs/panel_gate_check.log'", "DONE")

  def test_creating_the_path_first_removes_the_wait_requirement(self) -> None:
    """``create`` materializes it locally; the later event is then a refresh."""
    self._read(
        "file create --path 'logs/panel_gate_check.log' --content 'x'",
        "file read --path 'logs/panel_gate_check.log'",
        "DONE",
    )

  def test_appending_to_a_not_yet_arrived_file_is_rejected_too(self) -> None:
    """``append`` hard-fails on a missing path exactly like ``read``."""
    with self.assertRaises(ValueError):
      self._read(
          "file append --path 'logs/panel_gate_check.log' --content 'c'", "DONE"
      )

  def test_an_immediate_path_is_unaffected_by_a_pending_arrival(self) -> None:
    self._read("file read --path 'docs/policy.md'", "DONE")

  def test_a_delayed_path_is_not_offered_as_an_available_substitute(self) -> None:
    """The retry feedback lists what to use instead; a file that has not landed
    yet is not a valid substitute, so it must not appear there."""
    with self.assertRaises(ValueError) as ctx:
      self._read("file read --path 'totally/invented.md'", "DONE")
    msg = str(ctx.exception)
    self.assertIn("docs/policy.md", msg)
    self.assertNotIn("logs/panel_gate_check.log", msg)

  def test_behaviour_is_unchanged_when_nothing_is_delayed(self) -> None:
    """Backwards compatibility: the parameter defaults to no arrivals."""
    _validate_file_path_grounding(
        _cmds("file read --path 'logs/panel_gate_check.log'", "DONE"),
        self.EXISTING,
    )
    _validate_file_path_grounding(
        _cmds("file read --path 'logs/panel_gate_check.log'", "DONE"),
        self.EXISTING,
        {},
    )

  def test_a_move_of_a_delayed_source_degrades_to_a_plain_violation(self) -> None:
    """Documented gap, not an oversight: the timing walk reads ``--path`` only,
    so ``file move --src`` on a pending file raises the generic error and is
    repointed by the last-attempt escape rather than fixed with a wait. Adding
    a branch for it would be untestable dead weight -- ``file move`` occurs 0
    times across the 600 gold trajectories of v5."""
    with self.assertRaises(ValueError) as ctx:
      self._read(
          "file move --src 'logs/panel_gate_check.log' --dst 'z.md'", "DONE"
      )
    self.assertNotIn("timing", str(ctx.exception))


class TestEnsureWaitBeforeDelayedReads(unittest.TestCase):

  def test_no_op_without_arrivals(self) -> None:
    cmds = _cmds("file read --path 'docs/policy.md'", "DONE")
    before = _lines(cmds)
    self.assertEqual(_ensure_wait_before_delayed_reads(cmds, {}), 0)
    self.assertEqual(_lines(cmds), before)

  def test_no_op_when_the_trajectory_already_waits_enough(self) -> None:
    """The author writing its own wait is the desired outcome; this repair must
    stay invisible then, or every round grows a machine-inserted duplicate."""
    cmds = _cmds("wait 3600", "file read --path 'a.log'", "DONE")
    before = _lines(cmds)
    self.assertEqual(_ensure_wait_before_delayed_reads(cmds, {"a.log": 15}), 0)
    self.assertEqual(_lines(cmds), before)

  def test_inserts_the_missing_wait_immediately_before_the_reader(self) -> None:
    cmds = _cmds(
        "file read --path 'docs/policy.md'", "file read --path 'a.log'", "DONE"
    )
    self.assertEqual(_ensure_wait_before_delayed_reads(cmds, {"a.log": 15}), 1)
    self.assertEqual(
        _lines(cmds),
        [
            "file read --path 'docs/policy.md'",
            "wait 15",
            "file read --path 'a.log'",
            "DONE",
        ],
    )

  def test_an_existing_partial_wait_only_tops_up_the_remainder(self) -> None:
    cmds = _cmds("wait 10", "file read --path 'a.log'", "DONE")
    self.assertEqual(_ensure_wait_before_delayed_reads(cmds, {"a.log": 15}), 1)
    self.assertEqual(_lines(cmds)[1], "wait 5")

  def test_steps_are_renumbered_after_insertion(self) -> None:
    """Check derivation is 1-based and positional; a duplicate step_number
    silently misattributes every check derived after the insert."""
    cmds = _cmds("file read --path 'a.log'", "file read --path 'b.log'", "DONE")
    _ensure_wait_before_delayed_reads(cmds, {"a.log": 15, "b.log": 40})
    self.assertEqual([c["step_number"] for c in cmds], [1, 2, 3, 4, 5])

  def test_one_wait_can_land_several_files(self) -> None:
    """Waiting to 40 also delivers everything due earlier."""
    cmds = _cmds("file read --path 'b.log'", "file read --path 'a.log'", "DONE")
    self.assertEqual(
        _ensure_wait_before_delayed_reads(cmds, {"a.log": 15, "b.log": 40}), 1
    )
    self.assertEqual(_lines(cmds)[0], "wait 40")

  def test_own_create_preempts_the_wait(self) -> None:
    cmds = _cmds(
        "file create --path 'a.log' --content 'x'",
        "file read --path 'a.log'",
        "DONE",
    )
    self.assertEqual(_ensure_wait_before_delayed_reads(cmds, {"a.log": 15}), 0)

  def test_done_stays_terminal(self) -> None:
    cmds = _cmds("file read --path 'a.log'", "DONE")
    _ensure_wait_before_delayed_reads(cmds, {"a.log": 15})
    self.assertEqual(_lines(cmds)[-1], "DONE")

  def test_repair_is_idempotent(self) -> None:
    cmds = _cmds("file read --path 'a.log'", "DONE")
    _ensure_wait_before_delayed_reads(cmds, {"a.log": 15})
    self.assertEqual(_ensure_wait_before_delayed_reads(cmds, {"a.log": 15}), 0)

  def test_full_chain_converges_on_a_mixed_trajectory(self) -> None:
    """The pipeline order, end to end: the gate must have nothing left to say."""
    cmds = _cmds(
        "file append --path 'reports/wp.md' --content 'a'",
        "file read --path 'logs/late.log'",
        "file read --path 'totally/invented.md'",
        "DONE",
    )
    ordered = ["docs/policy.md", "logs/late.log"]
    arrivals = {"logs/late.log": 15}
    _ensure_file_writes_materialize(cmds, set(ordered))
    _ensure_wait_before_delayed_reads(cmds, arrivals)
    _reground_missing_file_reads(cmds, ordered)
    _validate_file_path_grounding(cmds, set(ordered), arrivals)  # must not raise
    self.assertEqual(_lines(cmds)[-1], "DONE")


class TestDelayedArrivalsDoNotResurrectTheAppendRegression(unittest.TestCase):
  """Diversity lock for the append verb.

  Telling the author "materialize a new file with ``file create`` FIRST" wiped
  out ``file append`` corpus-wide: 223 occurrences in v4 against 13 in v5, a 94%
  kill that traded 214 valid trajectories for the 9 genuinely broken ones. The
  timing work must not re-open that hole from the other side -- a mid-round
  arrival is an EXISTING file, so appending to it is correct and must survive
  both the deterministic repairs untouched.
  """

  def test_append_to_a_delayed_file_is_not_rewritten_to_create(self) -> None:
    """``create`` would overwrite the very content the event delivered."""
    cmds = _cmds("file append --path 'logs/late.log' --content 'note'", "DONE")
    # The caller passes the FULL path list, delayed entries included.
    self.assertEqual(
        _ensure_file_writes_materialize(cmds, {"logs/late.log"}), 0
    )
    self.assertEqual(
        _lines(cmds)[0], "file append --path 'logs/late.log' --content 'note'"
    )

  def test_append_to_a_delayed_file_is_fixed_with_a_wait_not_a_repoint(
      self,
  ) -> None:
    cmds = _cmds("file append --path 'logs/late.log' --content 'note'", "DONE")
    arrivals = {"logs/late.log": 15}
    self.assertEqual(_ensure_wait_before_delayed_reads(cmds, arrivals), 1)
    self.assertEqual(_lines(cmds)[0], "wait 15")
    self.assertEqual(
        _lines(cmds)[1], "file append --path 'logs/late.log' --content 'note'"
    )
    _validate_file_path_grounding(cmds, {"logs/late.log"}, arrivals)

  def test_append_to_an_immediate_file_never_becomes_a_wait(self) -> None:
    cmds = _cmds("file append --path 'logs/audit.log' --content 'more'", "DONE")
    before = _lines(cmds)
    self.assertEqual(_ensure_wait_before_delayed_reads(cmds, {}), 0)
    self.assertEqual(_lines(cmds), before)


class TestBindOrphanUpdatesToRound(unittest.TestCase):
  """``_bind_orphan_updates_to_round`` -- adopt undeclared updates.

  Everything downstream keys off ``round_spec.update_ids``: the materializer
  compiles background events from it, and ``_collect_delayed_arrivals`` (above)
  derives its wait directives from it. An update the round authored but did not
  restate there is written into the spec and then delivered to nobody -- 175 of
  644 v5 updates (27.2%), carrying 413 ``session_appends``.

  The binding must stay DETERMINISTIC. The ``u_<round_id>_NN`` id form is
  mandated by the generation prompt, so the update names its own round and no
  content is inspected. The tests below pin that it adopts on that prefix and
  nothing else -- a prefix rule that leaked across rounds would fabricate
  traffic the round's own narrative never sets up.
  """

  def test_orphan_sharing_the_round_prefix_is_adopted(self) -> None:
    updates = [{"update_id": "u_r03_01"}, {"update_id": "u_r03_02"}]
    self.assertEqual(
        _bind_orphan_updates_to_round("r03", [], updates),
        ["u_r03_01", "u_r03_02"],
    )

  def test_another_rounds_updates_are_never_adopted(self) -> None:
    updates = [
        {"update_id": "u_r03_01"},
        {"update_id": "u_r04_01"},
        {"update_id": "u_r02_09"},
    ]
    self.assertEqual(
        _bind_orphan_updates_to_round("r03", [], updates), ["u_r03_01"]
    )

  def test_a_shorter_round_id_is_not_a_prefix_of_a_longer_one(self) -> None:
    # "u_r1" must not swallow r10's updates: the trailing underscore in the
    # prefix is the only thing keeping r1 and r10 apart.
    updates = [{"update_id": "u_r10_01"}, {"update_id": "u_r1_01"}]
    self.assertEqual(
        _bind_orphan_updates_to_round("r1", [], updates), ["u_r1_01"]
    )
    self.assertEqual(
        _bind_orphan_updates_to_round("r10", [], updates), ["u_r10_01"]
    )

  def test_already_declared_ids_are_kept_once_and_stay_first(self) -> None:
    updates = [{"update_id": "u_r03_01"}, {"update_id": "u_r03_02"}]
    self.assertEqual(
        _bind_orphan_updates_to_round("r03", ["u_r03_02"], updates),
        ["u_r03_02", "u_r03_01"],
    )

  def test_a_declaration_naming_a_foreign_update_is_preserved(self) -> None:
    # The round explicitly asked for another round's update. That is a choice
    # the author made; adoption is additive and must not second-guess it.
    self.assertEqual(
        _bind_orphan_updates_to_round(
            "r03", ["u_r01_04"], [{"update_id": "u_r03_01"}]
        ),
        ["u_r01_04", "u_r03_01"],
    )

  def test_malformed_entries_are_skipped_not_raised(self) -> None:
    updates = [
        "not-a-dict",
        {"no_id": 1},
        {"update_id": ""},
        {"update_id": "u_r03_01"},
    ]
    self.assertEqual(
        _bind_orphan_updates_to_round("r03", None, updates), ["u_r03_01"]
    )

  def test_no_updates_leaves_the_declaration_untouched(self) -> None:
    self.assertEqual(
        _bind_orphan_updates_to_round("r03", ["u_r03_01"], []), ["u_r03_01"]
    )

  def test_adopted_updates_become_visible_to_collect_delayed_arrivals(
      self,
  ) -> None:
    """The point of the repair: a bound update now reaches the timing gate.

    Undeclared, the file it adds is invisible to ``_collect_delayed_arrivals``,
    so no ``wait`` is demanded and a gold step reading it hard-fails.
    """
    payload = {
        "rounds": [{"round_id": "r03", "update_ids": []}],
        "updates": [{
            "update_id": "u_r03_01",
            "trigger_virtual_time": 30,
            "workspace_adds": [{"path": "logs/late.log", "content": "x"}],
        }],
    }
    self.assertEqual(_collect_delayed_arrivals(payload, set()), {})

    payload["rounds"][0]["update_ids"] = _bind_orphan_updates_to_round(
        "r03", [], payload["updates"]
    )
    self.assertEqual(
        _collect_delayed_arrivals(payload, set()), {"logs/late.log": 30}
    )


if __name__ == "__main__":
  unittest.main()
