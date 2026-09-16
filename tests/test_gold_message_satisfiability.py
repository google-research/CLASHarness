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

"""Tests for the gold-satisfiability gate on ``messages_sent`` criteria.

WHY THIS EXISTS
---------------
A check whose criterion the round's OWN gold trajectory cannot satisfy is dead
weight twice over: no solver can ever pass it, so it contributes zero variance
and zero discrimination, while still taxing every model's score. Replaying the
v3 corpus put a number on it -- 110 of 1682 checks (6.5%) failed against their
own reference solution, and 109 of those were a single cell:

    effect        | condition             | checks | gold fails
    messages_sent | field_semantic_match  |    524 |  109 (20.8%)
    files_created | field_semantic_match  |    459 |    1 ( 0.2%)
    everything else                       |    699 |    0

It survived to ship because the materializer's calibration pass replays gold
with ``_stub_semantic_file_checks`` installed -- an always-PASS stub -- so a
semantic check is, from calibration's point of view, incapable of failing.

The gate closes that hole at GENERATION time, inside the retry loop, by
reconstructing the transcript the grader will build and asking the real judge
whether gold satisfies the criterion. The direction of the repair matters and is
pinned below: the criterion is the design intent, so a failure sends the AUTHOR
back to extend the trajectory (harder, not easier). Nothing here rewrites,
relaxes or downgrades a criterion.

The reconstruction was validated against 94 real replay transcripts recovered by
stubbing the judge: identical SET of lines 94/94 (100%), byte-identical text
81/94, and on the 13 that differed only in the ordering of ambient sub-agent
chatter the real judge returned the same verdict 13/13.

The safety property is structural, and the tests enforce each half of it: the
gate can only ever (a) raise, which triggers a retry, or (b) on the last attempt
drop the offending check. It has no path that makes a check HARDER or invents
one, so it cannot manufacture an unwinnable check. Symmetrically, an unavailable
or erroring judge must leave everything untouched -- dropping checks on an
infrastructure blip would silently thin the corpus.
"""

from __future__ import annotations

import re
import unittest
from unittest import mock

from clasharness.evaluation.checkers.effect_checker import _format_transcript_line
from clasharness.evaluation.llm_judge import JudgeResult
from clasharness.scenario.conflict_llm_pipeline import (
    _enforce_gold_satisfiable_message_criteria,
)
from clasharness.scenario.conflict_llm_pipeline import (
    _parse_outgoing_message_effects,
)
from clasharness.scenario.conflict_llm_pipeline import (
    _reconstruct_gold_messages_sent,
)
from clasharness.scenario.conflict_llm_pipeline import (
    _render_messages_sent_transcript,
)
from clasharness.scenario.conflict_llm_pipeline import (
    _canonicalize_gold_channels,
)
from clasharness.scenario.conflict_llm_pipeline import (
    _channel_vocabulary,
)
from clasharness.scenario.conflict_llm_pipeline import (
    _project_round_subagents_for_gate,
)
from clasharness.scenario.conflict_llm_pipeline import (
    _criterion_channel_tokens,
)
from clasharness.scenario.conflict_llm_pipeline import (
    _normalize_session_channels,
)
from clasharness.scenario.conflict_llm_pipeline import (
    _unknown_criterion_channels,
)
from clasharness.scenario.conflict_vlm_materializer import (
    _session_display_channel,
)
from clasharness.scenario.spec import ScenarioSession, session_channel_value

JUDGE = "clasharness.evaluation.llm_judge.judge_majority"

# A session in the shape the GENERATOR holds it: a raw layer1b dict whose
# 'channel' is the literal default, so the display name '#ops-triage' -- the
# name the grading criteria and the runtime both use -- is derived from the
# storage key and exists nowhere the round's author can see it.
_SESSION = {"session_id": "channel_ops_triage", "channel": "default"}


def _cmds(*lines: str) -> list[dict[str, str]]:
  return [{"command": line} for line in lines]


_GOLD = _cmds(
    "message send --channel '#ops-triage' --target 'Casey' --message 'audit?'"
)


def _scaffold(criterion: str, effect: str = "messages_sent") -> dict:
  return {
      "command": "message send --channel '#ops' --target 'X' --message 'hi'",
      "checker": {
          "effect_type": effect,
          "condition": "field_semantic_match",
          "expected": {"field": "message", "criterion": criterion},
      },
  }


class TestParseOutgoingMessageEffects(unittest.TestCase):
  """Faithfulness to the skills, including where they produce NOTHING.

  The gate's whole value is predicting an empty or incomplete transcript before
  it ships. If this parser is more generous than ``message_skill`` -- crediting
  gold for a message the runtime never records -- the gate goes blind exactly
  where the defect lives.
  """

  def test_send_with_all_three_required_flags(self) -> None:
    self.assertEqual(
        _parse_outgoing_message_effects(
            "message send --channel '#ops' --target 'Priya' --message 'ready'"
        ),
        [{"target": "Priya", "message": "ready", "channel": "#ops"}],
    )

  def test_send_missing_target_records_nothing(self) -> None:
    # message_skill._action_send exits 1 without emitting any state change.
    self.assertEqual(
        _parse_outgoing_message_effects(
            "message send --channel '#ops' --message 'ready'"
        ),
        [],
    )

  def test_send_missing_channel_records_nothing(self) -> None:
    self.assertEqual(
        _parse_outgoing_message_effects(
            "message send --target 'Priya' --message 'ready'"
        ),
        [],
    )

  def test_send_missing_message_records_nothing(self) -> None:
    self.assertEqual(
        _parse_outgoing_message_effects(
            "message send --channel '#ops' --target 'Priya'"
        ),
        [],
    )

  def test_broadcast_ignores_target_and_falls_back_to_the_channel(self) -> None:
    """The property most likely to be "fixed" into a bug.

    ``_action_broadcast`` never reads ``--target``; it reads ``--targets``, which
    the agent-facing tool spec does not expose. So a broadcast addressed at a
    person is recorded against the CHANNEL, and a criterion demanding that the
    person be messaged is not satisfied by it.
    """
    self.assertEqual(
        _parse_outgoing_message_effects(
            "message broadcast --channel '#ops' --target 'Priya'"
            " --message 'all hands'"
        ),
        [{"target": "#ops", "message": "all hands", "channel": "#ops"}],
    )

  def test_broadcast_fans_out_over_comma_separated_targets(self) -> None:
    got = _parse_outgoing_message_effects(
        "message broadcast --channel '#ops' --targets 'A,B' --message 'x'"
    )
    self.assertEqual([e["target"] for e in got], ["A", "B"])
    self.assertTrue(all(e["channel"] == "#ops" for e in got))

  def test_broadcast_fans_out_over_space_separated_targets(self) -> None:
    got = _parse_outgoing_message_effects(
        "message broadcast --channel '#ops' --targets A B --message 'x'"
    )
    self.assertEqual([e["target"] for e in got], ["A", "B"])

  def test_broadcast_without_channel_or_targets_records_nothing(self) -> None:
    self.assertEqual(
        _parse_outgoing_message_effects("message broadcast --message 'x'"), []
    )

  def test_poll_is_invisible_to_the_judge(self) -> None:
    # A poll records question/options but no ``message`` key, so
    # _format_transcript_line drops it and no criterion can be met by it.
    self.assertEqual(
        _parse_outgoing_message_effects(
            "message poll --channel '#ops' --question 'ship it?'"
            " --options 'yes,no'"
        ),
        [],
    )

  def test_react_is_invisible_to_the_judge(self) -> None:
    self.assertEqual(
        _parse_outgoing_message_effects(
            "message react --message-id 'm1' --emoji ':+1:'"
        ),
        [],
    )

  def test_delegate_renders_as_an_addressed_message_with_no_channel(
      self,
  ) -> None:
    self.assertEqual(
        _parse_outgoing_message_effects(
            "agent delegate --agent 'Casey' --instruction 'audit the cohort'"
        ),
        [{"target": "Casey", "message": "audit the cohort", "channel": ""}],
    )

  def test_delegate_without_an_instruction_records_nothing(self) -> None:
    self.assertEqual(
        _parse_outgoing_message_effects("agent delegate --agent 'Casey'"), []
    )

  def test_equals_form_flags_are_understood(self) -> None:
    self.assertEqual(
        _parse_outgoing_message_effects(
            "message send --channel=#ops --target=Priya --message='ready'"
        ),
        [{"target": "Priya", "message": "ready", "channel": "#ops"}],
    )

  def test_non_message_commands_contribute_nothing(self) -> None:
    for cmd in (
        "file read --path '/a.md'",
        "agent status",
        "agent list",
        "message search --query 'x'",
        "wait 30",
        "",
        "message",
    ):
      with self.subTest(cmd=cmd):
        self.assertEqual(_parse_outgoing_message_effects(cmd), [])

  def test_unbalanced_quotes_do_not_raise(self) -> None:
    self.assertEqual(
        _parse_outgoing_message_effects("message send --message 'oops"), []
    )


class TestReconstructGoldMessagesSent(unittest.TestCase):
  """Both effect sources, and the environment's idempotency rule."""

  def test_solver_messages_appear_in_order(self) -> None:
    got = _reconstruct_gold_messages_sent(
        _cmds(
            "message send --channel '#ops' --target 'A' --message 'first'",
            "message send --channel '#ops' --target 'B' --message 'second'",
        ),
        [],
    )
    self.assertEqual([e["message"] for e in got], ["first", "second"])

  def test_a_sub_agent_replies_right_after_being_addressed(self) -> None:
    got = _reconstruct_gold_messages_sent(
        _cmds(
            "message send --channel '#ops' --target 'Casey' --message 'audit?'",
            "message send --channel '#ops' --target 'B' --message 'fyi'",
        ),
        [{
            "agent_role": "Casey",
            "mock_response": "audit complete",
            "reply_channel": "#ops",
        }],
    )
    self.assertEqual(
        [e["message"] for e in got], ["audit?", "audit complete", "fyi"]
    )
    self.assertEqual(got[1]["sender"], "Casey")

  def test_state_effect_messages_are_additional_not_duplicates(self) -> None:
    """``mock_response`` and a ``state_effects`` send are two utterances.

    ``Environment._record_subagent_message`` records both; their text differs,
    so collapsing them would under-count what the judge actually sees.
    """
    got = _reconstruct_gold_messages_sent(
        _cmds("message send --channel '#ops' --target 'Casey' --message 'go'"),
        [{
            "agent_role": "Casey",
            "mock_response": "starting",
            "reply_channel": "#ops",
            "state_effects": [
                "message send --channel '#ops' --target 'A' --message 'done'"
            ],
        }],
    )
    self.assertEqual(
        [e["message"] for e in got], ["go", "starting", "done"]
    )

  def test_identical_sender_text_channel_is_recorded_once(self) -> None:
    # Mirrors the environment's idempotency key exactly.
    got = _reconstruct_gold_messages_sent(
        _cmds("message send --channel '#ops' --target 'Casey' --message 'go'"),
        [{
            "agent_role": "Casey",
            "mock_response": "ack",
            "reply_channel": "#ops",
            "state_effects": [
                "message send --channel '#ops' --target 'A' --message 'ack'"
            ],
        }],
    )
    self.assertEqual([e["message"] for e in got], ["go", "ack"])

  def test_a_sub_agent_addressed_twice_still_speaks_once(self) -> None:
    got = _reconstruct_gold_messages_sent(
        _cmds(
            "message send --channel '#ops' --target 'Casey' --message 'a'",
            "message send --channel '#ops' --target 'Casey' --message 'b'",
        ),
        [{"agent_role": "Casey", "mock_response": "ack", "reply_channel": "#ops"}],
    )
    self.assertEqual([e["message"] for e in got], ["a", "ack", "b"])

  def test_never_addressed_sub_agents_still_drain_at_the_end(self) -> None:
    """Ambient chatter reaches the judge whether or not gold talks to them.

    In the measured corpus 48% of transcript lines were sub-agent messages, so
    omitting the unaddressed ones would make the gate hallucinate failures.
    """
    got = _reconstruct_gold_messages_sent(
        _cmds("file read --path '/a.md'"),
        [{"agent_role": "Dana", "mock_response": "heads up", "reply_channel": "#ops"}],
    )
    self.assertEqual([e["message"] for e in got], ["heads up"])

  def test_empty_mock_response_contributes_nothing(self) -> None:
    self.assertEqual(
        _reconstruct_gold_messages_sent(
            [], [{"agent_role": "Dana", "mock_response": "   "}]
        ),
        [],
    )

  def test_the_trajectory_stops_at_the_terminator(self) -> None:
    got = _reconstruct_gold_messages_sent(
        _cmds(
            "message send --channel '#ops' --target 'A' --message 'kept'",
            "DONE",
            "message send --channel '#ops' --target 'B' --message 'unreachable'",
        ),
        [],
    )
    self.assertEqual([e["message"] for e in got], ["kept"])

  def test_malformed_inputs_do_not_raise(self) -> None:
    self.assertEqual(_reconstruct_gold_messages_sent([], []), [])
    self.assertEqual(
        _reconstruct_gold_messages_sent(["nope"], ["nope"]),  # type: ignore[list-item]
        [],
    )


class TestRenderMessagesSentTranscript(unittest.TestCase):
  """The renderer must BE the grader's, not merely resemble it."""

  def test_it_delegates_to_the_graders_own_formatter(self) -> None:
    effects = [
        {"target": "Priya", "message": "ready", "channel": "#ops"},
        {"sender": "Casey", "target": "#ops", "message": "ack", "channel": "#ops"},
    ]
    expected = "\n".join(
        _format_transcript_line(i, e, "message")
        for i, e in enumerate(effects, 1)
    )
    self.assertEqual(_render_messages_sent_transcript(effects), expected)

  def test_a_channel_only_message_renders_as_in_channel(self) -> None:
    line = _render_messages_sent_transcript(
        [{"target": "#ops", "message": "all hands", "channel": "#ops"}]
    )
    self.assertEqual(line, "[1] In #ops: all hands")

  def test_an_empty_message_is_dropped_entirely(self) -> None:
    self.assertEqual(
        _render_messages_sent_transcript(
            [{"target": "A", "message": "", "channel": "#ops"}]
        ),
        "",
    )

  def test_no_effects_render_to_the_empty_string(self) -> None:
    self.assertEqual(_render_messages_sent_transcript([]), "")


class TestEnforceGoldSatisfiableMessageCriteria(unittest.TestCase):
  """Sensitivity, and -- more important -- the four ways it must do nothing."""

  def setUp(self) -> None:
    self.commands = _cmds(
        "message send --channel '#ops' --target 'Priya' --message 'ready'"
    )

  def test_a_satisfied_criterion_is_left_alone(self) -> None:
    scaffolds = [_scaffold("Notifies Priya that it is ready.")]
    with mock.patch(
        JUDGE, return_value=JudgeResult(passed=True, score=1.0, reason="ok")
    ) as judge:
      _enforce_gold_satisfiable_message_criteria(
          scaffolds, self.commands, [], last_attempt=False
      )
    judge.assert_called_once()
    self.assertIsNotNone(scaffolds[0]["checker"])

  def test_an_unsatisfiable_criterion_raises_while_retries_remain(self) -> None:
    scaffolds = [_scaffold("Also messages Tara with the ticket id.")]
    with mock.patch(
        JUDGE,
        return_value=JudgeResult(passed=False, score=0.0, reason="no Tara msg"),
    ):
      with self.assertRaises(ValueError) as ctx:
        _enforce_gold_satisfiable_message_criteria(
            scaffolds, self.commands, [], last_attempt=False
        )
    msg = str(ctx.exception)
    # The retry feedback must push toward a LONGER trajectory, never a weaker
    # criterion -- that is the entire difficulty argument for this gate.
    self.assertIn("EXTEND the gold trajectory", msg)
    self.assertIn("Do NOT weaken", msg)
    self.assertIn("no Tara msg", msg)  # the judge's own reason is fed back
    self.assertIn("ready", msg)  # and the transcript it saw
    self.assertIsNotNone(scaffolds[0]["checker"])  # nothing mutated on raise

  def test_the_last_attempt_drops_only_the_offending_check(self) -> None:
    bad = _scaffold("Also messages Tara with the ticket id.")
    good = _scaffold("Notifies Priya that it is ready.")

    def _verdict(_transcript: str, criterion: str) -> JudgeResult:
      passed = "Priya" in criterion
      return JudgeResult(passed=passed, score=1.0 if passed else 0.0, reason="r")

    with mock.patch(JUDGE, side_effect=_verdict):
      _enforce_gold_satisfiable_message_criteria(
          [bad, good], self.commands, [], last_attempt=True
      )
    self.assertIsNone(bad["checker"])
    self.assertIsNotNone(good["checker"])
    # The criterion text itself is never rewritten or relaxed.
    self.assertEqual(
        good["checker"]["expected"]["criterion"],
        "Notifies Priya that it is ready.",
    )

  def test_an_empty_transcript_fails_without_spending_a_judge_call(self) -> None:
    scaffolds = [_scaffold("Notifies Priya.")]
    with mock.patch(JUDGE) as judge:
      with self.assertRaises(ValueError) as ctx:
        _enforce_gold_satisfiable_message_criteria(
            scaffolds, _cmds("file read --path '/a.md'"), [], last_attempt=False
        )
    judge.assert_not_called()
    self.assertIn("sends no message at all", str(ctx.exception))

  def test_a_judge_outage_leaves_every_check_intact(self) -> None:
    scaffolds = [_scaffold("Notifies Priya.")]
    with mock.patch(JUDGE, side_effect=RuntimeError("503")):
      _enforce_gold_satisfiable_message_criteria(
          scaffolds, self.commands, [], last_attempt=True
      )
    self.assertIsNotNone(scaffolds[0]["checker"])

  def test_a_judge_error_result_leaves_every_check_intact(self) -> None:
    scaffolds = [_scaffold("Notifies Priya.")]
    err = JudgeResult(passed=False, score=0.0, reason="", ok=False, error="quota")
    with mock.patch(JUDGE, return_value=err):
      _enforce_gold_satisfiable_message_criteria(
          scaffolds, self.commands, [], last_attempt=True
      )
    self.assertIsNotNone(scaffolds[0]["checker"])

  def test_other_effect_types_and_conditions_are_out_of_scope(self) -> None:
    scaffolds = [
        # Out of scope by effect_type: files_created semantic checks failed
        # gold 1/459 times, so they are not worth a judge call per round.
        _scaffold("x", effect="files_created"),
        {
            # Out of scope by condition. ``field_contains`` is graded by literal
            # substring, never by the judge, so a judge verdict about it would
            # be meaningless -- and acting on one could drop a perfectly
            # satisfiable check. The stray ``criterion`` key is deliberate: it
            # makes the CONDITION filter the thing under test rather than the
            # incidental absence of a criterion.
            "checker": {
                "effect_type": "messages_sent",
                "condition": "field_contains",
                "expected": {"value": "SEC-8842", "criterion": "mentions it"},
            }
        },
        {"checker": None},
        {"checker": "not-a-dict"},
    ]
    with mock.patch(JUDGE) as judge:
      _enforce_gold_satisfiable_message_criteria(
          scaffolds, self.commands, [], last_attempt=True
      )
    judge.assert_not_called()
    self.assertIsNotNone(scaffolds[0]["checker"])

  def test_a_blank_criterion_is_not_sent_to_the_judge(self) -> None:
    scaffolds = [_scaffold("   ")]
    with mock.patch(JUDGE) as judge:
      _enforce_gold_satisfiable_message_criteria(
          scaffolds, self.commands, [], last_attempt=True
      )
    judge.assert_not_called()

  def test_sub_agent_chatter_can_be_what_satisfies_a_criterion(self) -> None:
    """The judge must be shown the sub-agent lines, not just the solver's.

    Criteria are routinely written about a sub-agent's acknowledgement, and
    sub-agent messages are 48% of transcript lines in the measured corpus.
    """
    seen: dict[str, str] = {}

    def _capture(transcript: str, _criterion: str) -> JudgeResult:
      seen["t"] = transcript
      return JudgeResult(passed=True, score=1.0, reason="ok")

    with mock.patch(JUDGE, side_effect=_capture):
      _enforce_gold_satisfiable_message_criteria(
          [_scaffold("Casey confirms the audit.")],
          self.commands,
          [{
              "agent_role": "Casey",
              "mock_response": "audit confirmed",
              "reply_channel": "#ops",
          }],
          last_attempt=False,
      )
    self.assertIn("From Casey", seen["t"])
    self.assertIn("audit confirmed", seen["t"])


class TestChannelVocabulary(unittest.TestCase):
  """What the round's author is allowed to say, and what can be repaired."""

  def test_it_reports_names_to_show_and_aliases_to_repair(self) -> None:
    names, aliases = _channel_vocabulary([_SESSION])
    self.assertEqual(names, ["#ops-triage"])
    self.assertEqual(aliases["channel_ops_triage"], "#ops-triage")

  def test_no_sessions_means_no_vocabulary(self) -> None:
    # Callers MUST read this as "enforce nothing". A scenario really can ship
    # no sessions, and enforcing against an empty vocabulary would leave the
    # author no valid answer, burn every retry and discard the scenario.
    self.assertEqual(_channel_vocabulary([]), ([], {}))


class TestCanonicalizeGoldChannels(unittest.TestCase):
  """The generation-time half of the fix: one name for one conversation."""

  def _cmd_entries(self, *lines: str) -> list[dict[str, str]]:
    return [{"step_number": i, "command": c} for i, c in enumerate(lines, 1)]

  def test_a_provable_alias_is_repaired_in_place(self) -> None:
    cmds = self._cmd_entries(
        "message send --channel 'channel_ops_triage' --target 'A' --message 'x'"
    )
    left = _canonicalize_gold_channels(cmds, [_SESSION])
    self.assertEqual(left, [])
    self.assertIn("--channel '#ops-triage'", cmds[0]["command"])

  def test_an_invented_name_is_reported_not_guessed(self) -> None:
    original = "message send --channel '#risk-fraud-ops' --target 'A' --message 'x'"
    cmds = self._cmd_entries(original)
    left = _canonicalize_gold_channels(cmds, [_SESSION])
    self.assertEqual(left, ["#risk-fraud-ops"])
    self.assertEqual(cmds[0]["command"], original)  # untouched, for the author

  def test_the_last_attempt_binds_rather_than_discards_the_scenario(self) -> None:
    cmds = self._cmd_entries(
        "message send --channel '#risk-fraud-ops' --target 'A' --message 'x'"
    )
    logged: list[str] = []
    left = _canonicalize_gold_channels(
        cmds, [_SESSION], last_attempt=True, log=logged.append
    )
    self.assertEqual(left, [])
    self.assertIn("--channel '#ops-triage'", cmds[0]["command"])
    # Loud, because it is a guess: the corpus audit has to be able to find it.
    self.assertTrue(any("ChannelVocab" in m for m in logged), logged)

  def test_it_enforces_nothing_when_the_scenario_ships_no_sessions(self) -> None:
    original = "message send --channel '#anything' --target 'A' --message 'x'"
    cmds = self._cmd_entries(original)
    self.assertEqual(_canonicalize_gold_channels(cmds, []), [])
    self.assertEqual(cmds[0]["command"], original)

  def test_subagent_effects_are_repaired_too(self) -> None:
    """A reply posted to a channel nobody can read is as lost as a gold one."""
    subs = [{
        "agent_role": "Casey",
        "state_effects": [
            "message send --channel 'ops-triage' --target 'A' --message 'x'"
        ],
    }]
    left = _canonicalize_gold_channels(
        self._cmd_entries("file list"), [_SESSION], mock_subagents=subs
    )
    self.assertEqual(left, [])
    self.assertIn("--channel '#ops-triage'", subs[0]["state_effects"][0])

  def test_an_invented_subagent_channel_is_reported(self) -> None:
    subs = [{
        "agent_role": "Casey",
        "state_effects": [
            "message send --channel '#risk-fraud-ops' --target 'A'"
            " --message 'x'"
        ],
    }]
    left = _canonicalize_gold_channels(
        self._cmd_entries("file list"), [_SESSION], mock_subagents=subs
    )
    self.assertEqual(left, ["#risk-fraud-ops"])

  def test_non_string_effects_survive_at_their_index(self) -> None:
    """Dict effects carry no '--channel' and must not be reordered or dropped."""
    subs = [{
        "agent_role": "Casey",
        "state_effects": [
            {"type": "file_create", "path": "/a.md", "content": "z"},
            "message send --channel 'ops-triage' --target 'A' --message 'x'",
        ],
    }]
    _canonicalize_gold_channels(
        self._cmd_entries("file list"), [_SESSION], mock_subagents=subs
    )
    self.assertEqual(subs[0]["state_effects"][0]["path"], "/a.md")
    self.assertIn("--channel '#ops-triage'", subs[0]["state_effects"][1])

  def test_the_last_attempt_binds_commands_and_effects_alike(self) -> None:
    """One mapping for the whole round, or one conversation becomes two.

    ``assign_unresolved_channels`` binds by order of first appearance, so
    repairing the commands and the effects in separate batches could send the
    gold's question to one real channel and the sub-agent's answer to another.
    """
    cmds = self._cmd_entries(
        "message send --channel '#risk-fraud-ops' --target 'A' --message 'q'"
    )
    subs = [{
        "agent_role": "Casey",
        "state_effects": [
            "message send --channel '#risk-fraud-ops' --target 'A'"
            " --message 'a'"
        ],
    }]
    two = [_SESSION, {"session_id": "channel_audit_log", "channel": "default"}]
    left = _canonicalize_gold_channels(
        cmds, two, mock_subagents=subs, last_attempt=True
    )
    self.assertEqual(left, [])
    self.assertIn("--channel '#ops-triage'", cmds[0]["command"])
    self.assertIn("--channel '#ops-triage'", subs[0]["state_effects"][0])

  def test_distinct_invented_names_stay_distinct_across_the_round(self) -> None:
    cmds = self._cmd_entries(
        "message send --channel '#risk-fraud-ops' --target 'A' --message 'q'"
    )
    subs = [{
        "agent_role": "Casey",
        "state_effects": [
            "message send --channel '#support-verification' --target 'A'"
            " --message 'a'"
        ],
    }]
    two = [_SESSION, {"session_id": "channel_audit_log", "channel": "default"}]
    _canonicalize_gold_channels(
        cmds, two, mock_subagents=subs, last_attempt=True
    )
    self.assertIn("--channel '#ops-triage'", cmds[0]["command"])
    self.assertIn("--channel '#audit-log'", subs[0]["state_effects"][0])


class TestChannelVocabularyFallback(unittest.TestCase):
  """A scenario whose sessions declare nothing still has a vocabulary."""

  DEGENERATE = [{"session_id": "", "channel": "default"}]
  UPDATES = [{
      "session_appends": [
          {"channel_or_thread": "#incident-war-room"},
          {"channel_or_thread": "secops-endpoint-coordination"},
          {"channel_or_thread": "#incident-war-room"},
      ]
  }]

  def test_updates_supply_the_names_when_sessions_do_not(self) -> None:
    names, aliases = _channel_vocabulary(self.DEGENERATE, self.UPDATES)
    self.assertEqual(
        names, ["#incident-war-room", "#secops-endpoint-coordination"]
    )
    self.assertEqual(aliases["secops-endpoint-coordination"],
                     "#secops-endpoint-coordination")

  def test_sessions_win_whenever_they_have_anything_to_say(self) -> None:
    """Never a union: an update's own invented name must not become canonical."""
    names, _ = _channel_vocabulary([_SESSION], self.UPDATES)
    self.assertEqual(names, ["#ops-triage"])

  def test_the_gate_enforces_the_fallback_vocabulary(self) -> None:
    original = (
        "message send --channel '#made-up' --target 'A' --message 'x'"
    )
    cmds = [{"step_number": 1, "command": original}]
    left = _canonicalize_gold_channels(
        cmds, self.DEGENERATE, updates=self.UPDATES
    )
    self.assertEqual(left, ["#made-up"])
    self.assertEqual(cmds[0]["command"], original)

  def test_still_a_no_op_when_neither_source_names_a_channel(self) -> None:
    original = "message send --channel '#made-up' --target 'A' --message 'x'"
    cmds = [{"step_number": 1, "command": original}]
    self.assertEqual(
        _canonicalize_gold_channels(cmds, self.DEGENERATE, updates=[]), []
    )
    self.assertEqual(cmds[0]["command"], original)


class TestProjectRoundSubagentsForGate(unittest.TestCase):
  """The gate must predict the transcript the RUNTIME builds, not a fiction."""

  def test_an_unaddressed_authored_mock_alone_loses_the_channel(self) -> None:
    """Characterizes the defect, so a regression cannot pass silently.

    Every pre-existing test in this file hands the reconstructor a mock with
    ``reply_channel`` already filled in -- a shape the generation-time gate
    never actually holds, which is why the defect survived them. An authored
    mock has no such key: the materializer adds it later.

    Deliberately a sub-agent the gold never messages, because that is where the
    projection is the ONLY possible source of a channel. One the gold DOES
    address picks a channel up from the eliciting command as well (Route 2,
    pinned below), which would mask the defect this test exists to hold down.
    """
    authored = {"agent_role": "Dana", "mock_response": "policy reviewed"}
    rendered = _render_messages_sent_transcript(
        _reconstruct_gold_messages_sent(_GOLD, [authored])
    )
    self.assertIn("From Dana", rendered)
    self.assertNotIn("From Dana in #ops-triage", rendered)

  def test_projection_restores_it(self) -> None:
    authored = {"agent_role": "Casey", "mock_response": "audit confirmed"}
    rendered = _render_messages_sent_transcript(
        _reconstruct_gold_messages_sent(
            _GOLD,
            _project_round_subagents_for_gate(
                [authored], [_SESSION], [c["command"] for c in _GOLD]
            ),
        )
    )
    self.assertIn("From Casey in #ops-triage", rendered)

  def test_the_gate_judges_the_projected_transcript(self) -> None:
    # End to end: what reaches the judge carries the channel.
    seen: dict[str, str] = {}

    def _capture(transcript: str, _criterion: str) -> JudgeResult:
      seen["t"] = transcript
      return JudgeResult(passed=True, score=1.0, reason="ok")

    with mock.patch(JUDGE, side_effect=_capture):
      _enforce_gold_satisfiable_message_criteria(
          [_scaffold("Casey confirms the audit in the triage channel.")],
          _GOLD,
          _project_round_subagents_for_gate(
              [{"agent_role": "Casey", "mock_response": "audit confirmed"}],
              [_SESSION],
              [c["command"] for c in _GOLD],
          ),
          last_attempt=False,
      )
    self.assertIn("From Casey in #ops-triage", seen["t"])


def _bare_lines(rendered: str) -> list[str]:
  """Transcript lines without the renderer's ``[N] `` index prefix."""
  return [re.sub(r"^\[\d+\]\s*", "", ln) for ln in rendered.splitlines() if ln]


class TestElicitedSubagentReplies(unittest.TestCase):
  """Route 2: a sub-agent the gold addresses answers where it was addressed.

  ``Environment._apply_message_triggers`` replies on ``reply_channel or the
  channel the solver used``, while the baked ``subagent_completion`` (Route 1)
  replies on ``reply_channel`` alone. The gate has to model both or it misplaces
  -- or loses -- an utterance the judge will see.
  """

  def test_reply_lands_in_the_channel_the_gold_used(self) -> None:
    authored = {"agent_role": "Casey", "mock_response": "audit confirmed"}
    lines = _bare_lines(
        _render_messages_sent_transcript(
            _reconstruct_gold_messages_sent(_GOLD, [authored])
        )
    )
    self.assertIn("From Casey in #ops-triage: audit confirmed", lines)

  def test_both_routes_are_emitted_when_they_disagree(self) -> None:
    """The runtime dedups on (sender, text, channel) -- not on the sub-agent.

    With no ``reply_channel`` the two routes carry different channels, so the
    same utterance is recorded twice. Suppressing the second would hand the
    judge a transcript the replay does not produce; the shipped
    ``coordinated_containment_v02`` r10 replay contains exactly this pair.
    """
    authored = {"agent_role": "Casey", "mock_response": "audit confirmed"}
    lines = _bare_lines(
        _render_messages_sent_transcript(
            _reconstruct_gold_messages_sent(_GOLD, [authored])
        )
    )
    self.assertIn("From Casey in #ops-triage: audit confirmed", lines)
    self.assertIn("From Casey: audit confirmed", lines)

  def test_a_known_reply_channel_collapses_the_pair(self) -> None:
    """The common case: both routes agree, so the dedup leaves ONE line."""
    projected = {
        "agent_role": "Casey",
        "mock_response": "audit confirmed",
        "reply_channel": "#ops-triage",
    }
    lines = _bare_lines(
        _render_messages_sent_transcript(
            _reconstruct_gold_messages_sent(_GOLD, [projected])
        )
    )
    self.assertEqual(
        1, lines.count("From Casey in #ops-triage: audit confirmed")
    )
    self.assertNotIn("From Casey: audit confirmed", lines)

  def test_the_match_is_a_substring_like_the_runtime(self) -> None:
    """``sub_name.lower() in target.lower()`` -- a role need not equal a target."""
    gold = _cmds(
        "message send --channel '#ops-triage' --target 'Casey_Auditor'"
        " --message 'audit?'"
    )
    lines = _bare_lines(
        _render_messages_sent_transcript(
            _reconstruct_gold_messages_sent(
                gold, [{"agent_role": "Casey", "mock_response": "confirmed"}]
            )
        )
    )
    self.assertIn("From Casey in #ops-triage: confirmed", lines)

  def test_an_async_only_subagent_is_not_elicited(self) -> None:
    """It must surface only when ``wait`` advances the clock.

    The runtime skips ``async_only`` sub-agents on Route 2 so a no-wait solver
    cannot pull the decisive value out at virtual_time 0. Eliciting it here
    would have the gate judge a transcript strictly richer than the replay's.
    """
    authored = {
        "agent_role": "Casey",
        "mock_response": "audit confirmed",
        "async_only": True,
    }
    lines = _bare_lines(
        _render_messages_sent_transcript(
            _reconstruct_gold_messages_sent(_GOLD, [authored])
        )
    )
    self.assertNotIn("From Casey in #ops-triage: audit confirmed", lines)
    self.assertIn("From Casey: audit confirmed", lines)

  def test_an_effect_send_keeps_its_own_channel(self) -> None:
    """An explicit ``--channel`` in a state_effect outranks the elicitor's."""
    authored = {
        "agent_role": "Casey",
        "mock_response": "on it",
        "state_effects": [
            "message send --channel '#audit-log' --target 'Dana'"
            " --message 'logged'"
        ],
    }
    lines = _bare_lines(
        _render_messages_sent_transcript(
            _reconstruct_gold_messages_sent(_GOLD, [authored])
        )
    )
    self.assertIn("From Casey to Dana in #audit-log: logged", lines)
    # And only once: both routes agree on it, so the dedup collapses them.
    self.assertEqual(1, lines.count("From Casey to Dana in #audit-log: logged"))


# The v7 regression, verbatim. This session declares its channel under 'name',
# which the generator's gate could not see; it therefore derived
# '#01-partnership-review' from the session_id, told the gold author that was
# the only legal channel, and rewrote the real name away -- while
# materialization went on shipping '#partnership-contract-review'. Every
# assertion below is the property whose absence produced 433 phantom gold
# channels and 320 phantom criterion mentions.
_ALIAS_SESSION = {
    "session_id": "session_01_partnership_review",
    "name": "#partnership-contract-review",
}


class TestSessionChannelValue(unittest.TestCase):
  """One answer to 'which key holds the channel name', for every reader."""

  def test_every_accepted_spelling_is_read(self) -> None:
    for key in ("channel", "channel_name", "name"):
      with self.subTest(key=key):
        self.assertEqual(session_channel_value({key: "#ops"}), "#ops")

  def test_an_undeclared_channel_is_the_default_sentinel(self) -> None:
    # Not the empty string: callers branch on the literal 'default' to decide
    # whether to derive a display name from the session_id.
    self.assertEqual(session_channel_value({"session_id": "x"}), "default")

  def test_precedence_is_channel_then_channel_name_then_name(self) -> None:
    self.assertEqual(
        session_channel_value(
            {"channel": "#a", "channel_name": "#b", "name": "#c"}
        ),
        "#a",
    )
    self.assertEqual(
        session_channel_value({"channel_name": "#b", "name": "#c"}), "#b"
    )

  def test_the_raw_and_parsed_views_agree_on_every_spelling(self) -> None:
    """THE root-cause regression: two readers, one answer.

    The generator reads sessions as raw dicts and materialization reads them as
    parsed ``ScenarioSession`` objects. While those two disagreed, the gate
    enforced a vocabulary the shipped scenario did not have.
    """
    cases = [
        {"session_id": "channel_ops_triage", "channel": "#ops-triage"},
        {"session_id": "channel_ops_triage", "channel_name": "#ops-triage"},
        {"session_id": "channel_ops_triage", "name": "#ops-triage"},
        {"session_id": "channel_ops_triage"},
        {"session_id": "channel_ops_triage", "channel": "default"},
        {"session_id": "channel_ops_triage", "name": "ops-triage"},
        _ALIAS_SESSION,
    ]
    for raw in cases:
      with self.subTest(raw=raw):
        self.assertEqual(
            _session_display_channel(raw),
            _session_display_channel(ScenarioSession.from_dict(raw)),
        )

  def test_a_name_declared_channel_is_not_overridden_by_its_session_id(
      self,
  ) -> None:
    self.assertEqual(
        _session_display_channel(_ALIAS_SESSION), "#partnership-contract-review"
    )


class TestNormalizeSessionChannels(unittest.TestCase):
  """Collapsing the two views before any round reads the sessions."""

  def test_it_moves_a_declared_name_onto_the_channel_key(self) -> None:
    sessions = [dict(_ALIAS_SESSION)]
    self.assertEqual(_normalize_session_channels(sessions), 1)
    self.assertEqual(sessions[0]["channel"], "#partnership-contract-review")

  def test_it_is_idempotent(self) -> None:
    sessions = [dict(_ALIAS_SESSION)]
    _normalize_session_channels(sessions)
    # Second call must be a no-op, because it runs on the checkpoint-resume
    # path over a payload the fresh path may already have normalized.
    self.assertEqual(_normalize_session_channels(sessions), 0)

  def test_it_writes_only_the_channel_key(self) -> None:
    sessions = [{
        "session_id": "s1",
        "name": "#ops",
        "kind": "text",
        "messages": [{"speaker": "A", "text": "hi"}],
        "metadata": {"k": "v"},
    }]
    _normalize_session_channels(sessions)
    self.assertEqual(sessions[0]["name"], "#ops")  # LLM's own key survives
    self.assertEqual(sessions[0]["kind"], "text")
    self.assertEqual(sessions[0]["metadata"], {"k": "v"})
    self.assertEqual(len(sessions[0]["messages"]), 1)

  def test_an_undeclared_session_is_left_at_the_default_sentinel(self) -> None:
    sessions = [{"session_id": "channel_ops_triage"}]
    self.assertEqual(_normalize_session_channels(sessions), 1)
    self.assertEqual(sessions[0]["channel"], "default")
    # And the derived display name still works, so nothing is lost.
    self.assertEqual(_session_display_channel(sessions[0]), "#ops-triage")

  def test_non_dict_entries_and_empty_input_are_survived(self) -> None:
    sessions: list = ["junk", None, 7]
    self.assertEqual(_normalize_session_channels(sessions), 0)
    self.assertEqual(_normalize_session_channels([]), 0)
    self.assertEqual(_normalize_session_channels(None), 0)

  def test_the_enforced_vocabulary_becomes_the_shipped_vocabulary(self) -> None:
    """The fairness invariant, stated directly.

    What the gate tells the author is legal must be what materialization
    ships. Before normalization these two lists diverge, which is the whole
    defect; after it they must be equal.
    """
    sessions = [dict(_ALIAS_SESSION), {"session_id": "channel_audit_log"}]
    shipped = [
        _session_display_channel(ScenarioSession.from_dict(s)) for s in sessions
    ]
    _normalize_session_channels(sessions)
    enforced, _ = _channel_vocabulary(sessions)
    self.assertEqual(enforced, shipped)
    self.assertEqual(enforced, ["#partnership-contract-review", "#audit-log"])

  def test_gold_written_in_the_real_name_is_no_longer_rewritten(self) -> None:
    """End to end: the exact v7 failure, on the exact v7 shape."""
    sessions = [dict(_ALIAS_SESSION)]
    _normalize_session_channels(sessions)
    cmds = [{
        "step_number": 1,
        "command": (
            "message send --channel '#partnership-contract-review'"
            " --target 'A' --message 'x'"
        ),
    }]
    self.assertEqual(_canonicalize_gold_channels(cmds, sessions), [])
    self.assertIn("--channel '#partnership-contract-review'", cmds[0]["command"])


class TestCriterionChannelTokens(unittest.TestCase):
  """Which '#tokens' in a criterion are channel claims worth checking."""

  def test_it_finds_a_channel_reference(self) -> None:
    self.assertEqual(
        _criterion_channel_tokens("posts a summary in #bay4-ops"), ["#bay4-ops"]
    )

  def test_a_bare_ticket_number_is_not_a_channel(self) -> None:
    # The only safe shape-based exclusion: no channel is all digits.
    for tok in ("#802", "#4829"):
      with self.subTest(tok=tok):
        self.assertEqual(_criterion_channel_tokens(f"cites {tok} in full"), [])

  def test_an_all_caps_token_is_still_a_channel_claim(self) -> None:
    """The exclusion that used to live here was blind to the actual defect.

    '#SESS-GOV-02' is not an identifier -- it is the session_id-derived name
    the broken gate forced onto the author of deadlock_multi_approval_v01,
    overwriting the real '#gov-approval-escalations'. Nineteen of the 25
    mentions an all-caps exclusion hid on v7 were of exactly this kind, so the
    rule was blind to the one thing the gate exists to catch. Shape cannot
    separate these from '#GR-8841'; only membership can, and the alias table
    already decides that.
    """
    for tok in ("#SESS-GOV-02", "#SESS-MIGRATION-OPS-01", "#GR-8841"):
      with self.subTest(tok=tok):
        self.assertEqual(
            _criterion_channel_tokens(f"posts in {tok} promptly"), [tok]
        )

  def test_trailing_sentence_punctuation_is_stripped(self) -> None:
    # '.' and '-' are legal inside a channel name, so the regex cannot tell a
    # full stop from part of the name; the lookup has to.
    self.assertEqual(
        _criterion_channel_tokens("posted to #incident-response."),
        ["#incident-response"],
    )

  def test_empty_and_bare_hash_yield_nothing(self) -> None:
    self.assertEqual(_criterion_channel_tokens(""), [])
    self.assertEqual(_criterion_channel_tokens("a # b"), [])


class TestUnknownCriterionChannels(unittest.TestCase):
  """The criterion half of the gate: is the name it prints reachable?"""

  def setUp(self) -> None:
    _, self.aliases = _channel_vocabulary([_SESSION])

  def test_a_phantom_channel_is_reported(self) -> None:
    self.assertEqual(
        _unknown_criterion_channels(["notifies Casey in #bay4-ops"], self.aliases),
        ["#bay4-ops"],
    )

  def test_a_real_channel_and_a_provable_alias_both_pass(self) -> None:
    self.assertEqual(
        _unknown_criterion_channels(
            ["posts in #ops-triage", "posts in channel_ops_triage"],
            self.aliases,
        ),
        [],
    )

  def test_no_vocabulary_means_enforce_nothing(self) -> None:
    # With no vocabulary there is no valid answer, so enforcing would burn
    # every retry on the scenario least able to afford it.
    self.assertEqual(_unknown_criterion_channels(["#anything"], {}), [])

  def test_a_repeated_phantom_is_reported_once(self) -> None:
    self.assertEqual(
        _unknown_criterion_channels(
            ["posts in #bay4-ops", "and again in #bay4-ops"], self.aliases
        ),
        ["#bay4-ops"],
    )

  def test_a_bare_ticket_number_does_not_bounce_a_criterion(self) -> None:
    self.assertEqual(
        _unknown_criterion_channels(
            ["cites ticket #802 in #ops-triage"], self.aliases
        ),
        [],
    )

  def test_a_session_id_shaped_phantom_is_reported(self) -> None:
    """The v7 regression the earlier all-caps exclusion silently allowed."""
    self.assertEqual(
        _unknown_criterion_channels(
            ["escalates in #SESS-GOV-02 before the deadline"], self.aliases
        ),
        ["#SESS-GOV-02"],
    )


class TestBlindBindAccounting(unittest.TestCase):
  """Counting the guesses, without changing how the guess is made.

  The last-attempt bind stays exactly as it was -- it is what keeps a stubborn
  round from costing the whole scenario. What was missing is that the run had
  no idea how often it fired, so a corpus could ship full of bound channels and
  look clean.
  """

  def _cmd(self, line: str) -> list[dict]:
    return [{"step_number": 1, "command": line}]

  def test_a_blind_bind_is_counted(self) -> None:
    stats: dict[str, int] = {}
    cmds = self._cmd(
        "message send --channel '#invented-name' --target 'A' --message 'x'"
    )
    self.assertEqual(
        _canonicalize_gold_channels(
            cmds, [_SESSION], last_attempt=True, stats=stats
        ),
        [],
    )
    self.assertEqual(stats.get("blind_bound"), 1)

  def test_nothing_is_counted_when_nothing_was_guessed(self) -> None:
    stats: dict[str, int] = {}
    cmds = self._cmd(
        "message send --channel '#ops-triage' --target 'A' --message 'x'"
    )
    _canonicalize_gold_channels(
        cmds, [_SESSION], last_attempt=True, stats=stats
    )
    self.assertEqual(stats.get("blind_bound", 0), 0)

  def test_counting_is_optional_and_the_repair_is_unchanged(self) -> None:
    cmds = self._cmd(
        "message send --channel '#invented-name' --target 'A' --message 'x'"
    )
    self.assertEqual(
        _canonicalize_gold_channels(cmds, [_SESSION], last_attempt=True), []
    )
    self.assertIn("--channel '#ops-triage'", cmds[0]["command"])


if __name__ == "__main__":
  unittest.main()
