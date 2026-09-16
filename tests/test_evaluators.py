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

"""Unit tests for the evaluation checkers."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from types import SimpleNamespace

from clasharness.core.observation import CheckResult, EvaluationResult
from clasharness.evaluation.checkers.config_checker import ConfigChecker
from clasharness.evaluation.checkers.effect_checker import EffectChecker
from clasharness.evaluation.checkers.file_checker import FileChecker
from clasharness.evaluation.checkers.output_checker import OutputChecker
from clasharness.evaluation.checkers.state_checker import StateChecker
from clasharness.evaluation.evaluator import EvaluatorComb, build_evaluator
from examples.train_and_eval import ExpertAgent


# ---- StateChecker ----

class TestStateChecker:
    def make_state(self):
        return {
            "agents": {
                "alice": {"model": "gpt-4o", "emoji": "🤖"},
            },
            "channels": {
                "telegram": {"status": "connected"},
            },
            "gateway_status": {"running": True},
        }

    def test_exists_pass(self):
        checker = StateChecker(field="agents.alice", condition="exists", expected=None)
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is True

    def test_exists_fail(self):
        checker = StateChecker(field="agents.bob", condition="exists", expected=None)
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is False

    def test_not_exists(self):
        checker = StateChecker(field="agents.unknown", condition="not_exists", expected=None)
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is True

    def test_equals_pass(self):
        checker = StateChecker(
            field="agents.alice.model", condition="equals", expected="gpt-4o"
        )
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is True

    def test_equals_fail(self):
        checker = StateChecker(
            field="agents.alice.model", condition="equals", expected="claude"
        )
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is False

    def test_count_gte_pass(self):
        state = {"cron_jobs": [{"id": "1"}, {"id": "2"}]}
        checker = StateChecker(field="cron_jobs", condition="count_gte", expected=2)
        result = checker.evaluate(state, {})
        assert result.passed is True

    def test_count_gte_fail(self):
        state = {"cron_jobs": [{"id": "1"}]}
        checker = StateChecker(field="cron_jobs", condition="count_gte", expected=3)
        result = checker.evaluate(state, {})
        assert result.passed is False

    def test_gateway_running(self):
        checker = StateChecker(
            field="gateway_status.running", condition="equals", expected=True
        )
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is True


# ---- OutputChecker ----

class TestOutputChecker:
    def make_state(self, stdout="Hello World"):
        return {"last_stdout": stdout, "last_stderr": ""}

    def test_contains_pass(self):
        checker = OutputChecker(condition="contains", expected="Hello")
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is True

    def test_contains_fail(self):
        checker = OutputChecker(condition="contains", expected="Goodbye")
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is False

    def test_contains_case_insensitive(self):
        checker = OutputChecker(condition="contains", expected="hello", ignore_case=True)
        result = checker.evaluate(self.make_state("Hello World"), {})
        assert result.passed is True

    def test_exact_pass(self):
        checker = OutputChecker(condition="exact", expected="Hello World")
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is True

    def test_exact_fail(self):
        checker = OutputChecker(condition="exact", expected="Hello")
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is False

    def test_regex_pass(self):
        checker = OutputChecker(condition="regex", expected=r"Hello\s+\w+")
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is True

    def test_regex_fail(self):
        checker = OutputChecker(condition="regex", expected=r"^\d+$")
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is False

    def test_not_contains_pass(self):
        checker = OutputChecker(condition="not_contains", expected="Error")
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is True

    def test_not_contains_fail(self):
        checker = OutputChecker(condition="not_contains", expected="World")
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is False

    # ---- not_contains is universally quantified over command_history ----
    #
    # ``last_stdout`` holds only the most recent command, so the checker also
    # consults ``command_history``. A negative assertion must hold for EVERY
    # entry; folding it into the positive any-match rescue let a single clean
    # earlier command mask a real violation.

    def _state_with_history(self, stdout, history_stdouts):
        return {
            "last_stdout": stdout,
            "last_stderr": "",
            "command_history": [{"stdout": h} for h in history_stdouts],
        }

    def test_not_contains_not_rescued_by_clean_history_entry(self):
        # last_stdout genuinely contains the forbidden text; an unrelated clean
        # earlier command must not rescue it.
        checker = OutputChecker(condition="not_contains", expected="Error")
        state = self._state_with_history(
            "Error: disk full", ["total 0", "Error: disk full"]
        )
        assert checker.evaluate(state, {}).passed is False

    def test_not_contains_fails_on_violation_in_earlier_command(self):
        # Clean last_stdout, but the forbidden text appeared earlier in the run.
        checker = OutputChecker(condition="not_contains", expected="Error")
        state = self._state_with_history("done", ["Error: disk full", "done"])
        assert checker.evaluate(state, {}).passed is False

    def test_not_contains_passes_when_absent_everywhere(self):
        checker = OutputChecker(condition="not_contains", expected="Error")
        state = self._state_with_history("done", ["total 0", "ok", "done"])
        assert checker.evaluate(state, {}).passed is True

    def test_not_contains_history_scan_respects_ignore_case(self):
        checker = OutputChecker(
            condition="not_contains", expected="error", ignore_case=True
        )
        state = self._state_with_history("done", ["ERROR: nope", "done"])
        assert checker.evaluate(state, {}).passed is False

    def test_contains_still_rescued_by_any_history_entry(self):
        # Positive assertions stay existentially quantified.
        checker = OutputChecker(condition="contains", expected="gateway")
        state = self._state_with_history("done", ["gateway running", "done"])
        assert checker.evaluate(state, {}).passed is True

    def test_regex_still_rescued_by_any_history_entry(self):
        checker = OutputChecker(condition="regex", expected=r"gate\w+")
        state = self._state_with_history("done", ["gateway running", "done"])
        assert checker.evaluate(state, {}).passed is True

    def test_stderr_field(self):
        checker = OutputChecker(
            condition="contains", expected="error", output_field="last_stderr"
        )
        result = checker.evaluate({"last_stdout": "ok", "last_stderr": "some error"}, {})
        assert result.passed is True

    def test_from_spec_condition_alias(self):
        checker = OutputChecker.from_spec({
            "condition": "contains",
            "expected": "hello",
        })
        result = checker.evaluate(self.make_state("Hello World"), {})
        assert result.passed is True

    def test_from_spec_accepts_match_type_alias(self):
        """``match_type`` is the spelling persisted in the shipped task specs.

        It used to be unread, so the condition silently defaulted to
        ``contains`` and a regex alternation was searched for as a literal
        substring -- impossible to satisfy, and it failed even the expert
        solution (``check_system_status_1``).
        """
        checker = OutputChecker.from_spec({
            "match_type": "regex",
            "expected": r"(Gateway|gateway|running|stopped)",
        })
        assert checker.condition == "regex"
        result = checker.evaluate(
            self.make_state("Gateway: stopped (port 18789)\nChannels: 0"), {}
        )
        assert result.passed is True

    def test_from_spec_condition_wins_over_match_type(self):
        checker = OutputChecker.from_spec({
            "condition": "contains",
            "match_type": "regex",
            "expected": "hello",
        })
        assert checker.condition == "contains"

    def test_from_spec_match_type_literal_still_literal(self):
        """``match_type: contains`` must stay a literal match, not become regex."""
        checker = OutputChecker.from_spec({
            "match_type": "contains",
            "expected": "(a|b)",
        })
        assert checker.condition == "contains"
        assert checker.evaluate(self.make_state("a"), {}).passed is False
        assert checker.evaluate(self.make_state("x (a|b) y"), {}).passed is True

    def test_exit_code_zero_with_null_expected(self):
        checker = OutputChecker.from_spec({
            "condition": "exit_code_zero",
            "expected": None,
        })
        ok = checker.evaluate({"last_stdout": "", "last_exit_code": 0}, {})
        bad = checker.evaluate({"last_stdout": "", "last_exit_code": 1}, {})
        assert ok.passed is True
        assert bad.passed is False


# ---- FileChecker ----

def _never_call_judge(*args, **kwargs):
    raise AssertionError("judge must not be called on a fail-closed path")


class TestFileChecker:
    def state(self, path, content):
        return {"files": {path: content}}

    def test_exists_and_not_exists(self):
        st = self.state("/report.md", "hello")
        assert FileChecker("/report.md", "exists", None).evaluate(st, {}).passed is True
        assert FileChecker("/missing.md", "exists", None).evaluate(st, {}).passed is False
        assert (
            FileChecker("/missing.md", "not_exists", None).evaluate(st, {}).passed
            is True
        )

    def test_contains_folds_case_and_whitespace(self):
        # Cosmetic differences (case, indentation, newlines, repeated spaces)
        # must not defeat a substring check on generated content.
        st = self.state("/notes.txt", "The   Quick\nBROWN   fox\tjumps")
        checker = FileChecker("/notes.txt", "contains", "quick brown fox")
        assert checker.evaluate(st, {}).passed is True

    def test_contains_still_fails_on_absent_substring(self):
        st = self.state("/notes.txt", "the quick brown fox")
        checker = FileChecker("/notes.txt", "contains", "lazy dog")
        assert checker.evaluate(st, {}).passed is False

    def test_contains_missing_file_fails(self):
        checker = FileChecker("/notes.txt", "contains", "anything")
        assert checker.evaluate({"files": {}}, {}).passed is False

    def test_equals_folds_case_and_whitespace(self):
        st = self.state("/status.txt", "  READY\n")
        checker = FileChecker("/status.txt", "equals", "ready")
        assert checker.evaluate(st, {}).passed is True

    def test_equals_still_fails_on_different_content(self):
        st = self.state("/status.txt", "READY")
        checker = FileChecker("/status.txt", "equals", "PENDING")
        assert checker.evaluate(st, {}).passed is False

    # ---- field_semantic_match: file-scoped LLM grading of ambiguous content ----

    def test_semantic_match_missing_file_fails_closed_without_judge(self, monkeypatch):
        # A positive semantic assertion cannot be satisfied by a missing file, and
        # must fail WITHOUT spending a model call.
        from clasharness.evaluation import llm_judge

        def _boom(*args, **kwargs):
            raise AssertionError("judge must not run when the file is missing")

        monkeypatch.setattr(llm_judge, "judge_majority", _boom)
        checker = FileChecker(
            "/report.md",
            "field_semantic_match",
            {"criterion": "states the rollback verdict"},
        )
        result = checker.evaluate({"files": {}}, {})
        assert result.passed is False
        assert result.score == 0.0

    def test_semantic_match_empty_criterion_fails_without_judge(self, monkeypatch):
        from clasharness.evaluation import llm_judge

        def _boom(*args, **kwargs):
            raise AssertionError("judge must not run with an empty criterion")

        monkeypatch.setattr(llm_judge, "judge_majority", _boom)
        checker = FileChecker(
            "/report.md", "field_semantic_match", {"criterion": ""}
        )
        result = checker.evaluate(self.state("/report.md", "some content"), {})
        assert result.passed is False

    def test_semantic_match_grades_file_content_via_judge(self, monkeypatch):
        from clasharness.evaluation import llm_judge
        from clasharness.evaluation.llm_judge import JudgeResult

        seen = {}

        def _fake(content, criterion, *, model=None, k=3, temperature=0.5):
            seen["content"] = content
            seen["criterion"] = criterion
            seen["model"] = model
            return JudgeResult(passed=True, score=0.75, reason="matches")

        monkeypatch.setattr(llm_judge, "judge_majority", _fake)
        checker = FileChecker(
            "/report.md",
            "field_semantic_match",
            {"criterion": "states the rollback verdict", "model": "m1"},
        )
        result = checker.evaluate(
            self.state("/report.md", "We rolled back to v4.2."), {}
        )
        assert result.passed is True
        # Judge's continuous score is propagated, not coerced to 1.0.
        assert result.score == 0.75
        assert "rolled back" in seen["content"]
        assert seen["criterion"] == "states the rollback verdict"
        assert seen["model"] == "m1"

    def test_semantic_match_judge_failure_fails_check(self, monkeypatch):
        from clasharness.evaluation import llm_judge
        from clasharness.evaluation.llm_judge import JudgeResult

        def _fake(content, criterion, *, model=None, k=3, temperature=0.5):
            return JudgeResult(passed=False, score=0.0, reason="no verdict stated")

        monkeypatch.setattr(llm_judge, "judge_majority", _fake)
        checker = FileChecker(
            "/report.md",
            "field_semantic_match",
            {"criterion": "states the rollback verdict"},
        )
        result = checker.evaluate(self.state("/report.md", "raw data only"), {})
        assert result.passed is False

    # ---- judge_invoked: "never asked" vs "asked and it was wrong" ----
    #
    # Both fail closed with passed=False, but they are not the same measurement:
    # one says the solver produced nothing to grade, the other says it produced
    # the wrong thing. The report layer used to tell them apart by substring-
    # matching the human-facing message, which silently breaks on a reword.

    def test_semantic_match_missing_file_records_judge_not_invoked(self, monkeypatch):
        from clasharness.evaluation import llm_judge

        monkeypatch.setattr(llm_judge, "judge_majority", _never_call_judge)
        checker = FileChecker(
            "/report.md", "field_semantic_match", {"criterion": "states a verdict"}
        )
        result = checker.evaluate({"files": {}}, {})
        assert result.judge_invoked is False
        # It is a real verdict, not an infra failure -- ``ok`` stays True.
        assert result.ok is True

    def test_semantic_match_empty_criterion_records_judge_not_invoked(
        self, monkeypatch
    ):
        from clasharness.evaluation import llm_judge

        monkeypatch.setattr(llm_judge, "judge_majority", _never_call_judge)
        checker = FileChecker("/report.md", "field_semantic_match", {"criterion": ""})
        result = checker.evaluate(self.state("/report.md", "content"), {})
        assert result.judge_invoked is False

    def test_semantic_match_that_reached_the_judge_records_judge_invoked(
        self, monkeypatch
    ):
        from clasharness.evaluation import llm_judge
        from clasharness.evaluation.llm_judge import JudgeResult

        monkeypatch.setattr(
            llm_judge,
            "judge_majority",
            lambda *a, **k: JudgeResult(passed=False, score=0.0, reason="wrong"),
        )
        checker = FileChecker(
            "/report.md", "field_semantic_match", {"criterion": "states a verdict"}
        )
        result = checker.evaluate(self.state("/report.md", "raw data only"), {})
        assert result.passed is False
        assert result.judge_invoked is True

    def test_deterministic_file_checks_record_judge_invoked(self):
        """Non-judge branches must not masquerade as skipped judge calls."""
        st = self.state("/status.txt", "READY")
        for condition, expected in (
            ("exists", None),
            ("not_exists", None),
            ("contains", "READY"),
            ("equals", "READY"),
        ):
            checker = FileChecker("/status.txt", condition, expected)
            assert checker.evaluate(st, {}).judge_invoked is True, condition

    def test_semantic_match_from_spec_normalizes_top_level_criterion(self):
        spec = {
            "type": "file_check",
            "path": "/r.md",
            "condition": "field_semantic_match",
            "criterion": "states X",
            "model": "m",
        }
        checker = FileChecker.from_spec(spec)
        assert checker.condition == "field_semantic_match"
        assert checker.expected == {"criterion": "states X", "model": "m"}

    def test_semantic_match_from_spec_reads_nested_criterion(self):
        spec = {
            "type": "file_check",
            "path": "/r.md",
            "condition": "field_semantic_match",
            "expected": {"criterion": "states Y"},
        }
        checker = FileChecker.from_spec(spec)
        assert checker.expected["criterion"] == "states Y"


def test_expert_agent_should_stop_after_commands():
    agent = ExpertAgent()
    task = SimpleNamespace(
        ground_truth=SimpleNamespace(solution_commands=["tasks list"])
    )
    agent.reset("dummy", task=task)
    assert agent.should_stop() is False
    _ = agent.act(SimpleNamespace())
    assert agent.should_stop() is True


# ---- EffectChecker ----

class TestEffectChecker:
    def make_state(self, effects=None):
        return {
            "effects": effects or {},
        }

    def test_exists_pass(self):
        checker = EffectChecker(
            effect_type="messages_sent", condition="exists", expected=None
        )
        state = self.make_state({"messages_sent": [{"target": "@alice"}]})
        result = checker.evaluate(state, {})
        assert result.passed is True

    def test_exists_fail(self):
        checker = EffectChecker(
            effect_type="messages_sent", condition="exists", expected=None
        )
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is False

    def test_not_exists(self):
        checker = EffectChecker(
            effect_type="messages_sent", condition="not_exists", expected=None
        )
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is True

    def test_count(self):
        checker = EffectChecker(
            effect_type="messages_sent", condition="count", expected=2
        )
        state = self.make_state({
            "messages_sent": [{"id": "1"}, {"id": "2"}]
        })
        result = checker.evaluate(state, {})
        assert result.passed is True

    def test_count_gte(self):
        checker = EffectChecker(
            effect_type="cron_jobs_created", condition="count_gte", expected=1
        )
        state = self.make_state({"cron_jobs_created": [{"id": "abc"}]})
        result = checker.evaluate(state, {})
        assert result.passed is True

    def test_count_lte(self):
        checker = EffectChecker(
            effect_type="tasks_created", condition="count_lte", expected=0
        )
        result = checker.evaluate(self.make_state({"tasks_created": []}), {})
        assert result.passed is True

    def test_field_equals_pass(self):
        checker = EffectChecker(
            effect_type="messages_sent",
            condition="field_equals",
            expected={"field": "target", "value": "@alice"},
        )
        state = self.make_state({
            "messages_sent": [{"target": "@alice", "message": "hi"}]
        })
        result = checker.evaluate(state, {})
        assert result.passed is True

    def test_field_equals_fail(self):
        checker = EffectChecker(
            effect_type="messages_sent",
            condition="field_equals",
            expected={"field": "target", "value": "@bob"},
        )
        state = self.make_state({
            "messages_sent": [{"target": "@alice", "message": "hi"}]
        })
        result = checker.evaluate(state, {})
        assert result.passed is False

    def test_field_contains_pass(self):
        checker = EffectChecker(
            effect_type="messages_sent",
            condition="field_contains",
            expected={"field": "message", "value": "hello"},
        )
        state = self.make_state({
            "messages_sent": [{"target": "@x", "message": "say hello world"}]
        })
        result = checker.evaluate(state, {})
        assert result.passed is True

    def test_field_contains_messages_sent_ignores_articles_and_pronouns(self):
        checker = EffectChecker(
            effect_type="messages_sent",
            condition="field_contains",
            expected={"field": "message", "value": "verify the gateway metrics in their logs"},
        )
        state = self.make_state({
            "messages_sent": [{"target": "@x", "message": "verify gateway metrics in your logs"}]
        })
        result = checker.evaluate(state, {})
        assert result.passed is True

    def test_field_contains_messages_sent_allows_gaps_but_enforces_order(self):
        checker = EffectChecker(
            effect_type="messages_sent",
            condition="field_contains",
            expected={"field": "message", "value": "cross-reference gateway timeouts"},
        )
        state = self.make_state({
            "messages_sent": [{"target": "@x", "message": "please cross-reference the logs with gateway timeouts"}]
        })
        result = checker.evaluate(state, {})
        assert result.passed is True

    def test_field_contains_messages_sent_out_of_order_fails(self):
        checker = EffectChecker(
            effect_type="messages_sent",
            condition="field_contains",
            expected={"field": "message", "value": "cross-reference gateway timeouts"},
        )
        state = self.make_state({
            "messages_sent": [{"target": "@x", "message": "gateway timeouts should be cross-referenced"}]
        })
        result = checker.evaluate(state, {})
        assert result.passed is False

    def test_field_contains_messages_sent_ignores_plurals(self):
        checker = EffectChecker(
            effect_type="messages_sent",
            condition="field_contains",
            expected={"field": "message", "value": "log timeout transition"},
        )
        state = self.make_state({
            "messages_sent": [{"target": "@x", "message": "analyze sudden logs timeout transition"}]
        })
        result = checker.evaluate(state, {})
        assert result.passed is True

    def test_field_contains_is_case_insensitive(self):
        checker = EffectChecker(
            effect_type="tasks_created",
            condition="field_contains",
            expected={"field": "title", "value": "Outage"},
        )
        state = self.make_state({
            "tasks_created": [{"title": "outage follow-up"}]
        })
        result = checker.evaluate(state, {})
        assert result.passed is True

    def test_plugins_installed(self):
        checker = EffectChecker(
            effect_type="plugins_installed",
            condition="field_equals",
            expected={"field": "name", "value": "slack-integration"},
        )
        state = self.make_state({
            "plugins_installed": [{"name": "slack-integration", "version": "1.0.0"}]
        })
        result = checker.evaluate(state, {})
        assert result.passed is True

    # ---- field_semantic_match: empty short-circuit + k-way majority vote ----

    def test_field_semantic_match_empty_transcript_short_circuits(self, monkeypatch):
        # No messages sent -> the check must fail WITHOUT invoking the judge.
        from clasharness.evaluation import llm_judge

        def _boom(*args, **kwargs):
            raise AssertionError("judge must not run on an empty transcript")

        monkeypatch.setattr(llm_judge, "judge_majority", _boom)
        checker = EffectChecker(
            effect_type="messages_sent",
            condition="field_semantic_match",
            expected={"field": "message", "criterion": "asks to restart the gateway"},
        )
        result = checker.evaluate(self.make_state({"messages_sent": []}), {})
        assert result.passed is False
        assert "empty transcript" in result.message

    def test_field_semantic_match_field_missing_short_circuits(self, monkeypatch):
        # Effects exist but none carry the graded field -> still empty transcript.
        from clasharness.evaluation import llm_judge

        def _boom(*args, **kwargs):
            raise AssertionError("judge must not run when field content is absent")

        monkeypatch.setattr(llm_judge, "judge_majority", _boom)
        checker = EffectChecker(
            effect_type="messages_sent",
            condition="field_semantic_match",
            expected={"field": "message", "criterion": "asks to restart the gateway"},
        )
        state = self.make_state({"messages_sent": [{"target": "@ops", "message": ""}]})
        assert checker.evaluate(state, {}).passed is False

    def test_empty_transcript_is_marked_as_judge_not_invoked(self, monkeypatch):
        """"Sent nothing" and "sent the wrong thing" must be distinguishable.

        Both land as a failed required semantic check, but they mean opposite
        things about the solver. Measured on the conflict-VLM set the mix varies
        by an order of magnitude across models (39.4% of 3.5-flash's semantic
        checks never reached the judge versus 1.4% for terra), so a report that
        cannot separate them reads a non-acting model as a semantically wrong
        one. ``judge_invoked`` carries that split structurally instead of forcing
        consumers to substring-match the message text.
        """
        from clasharness.evaluation import llm_judge

        def _boom(*args, **kwargs):
            raise AssertionError("judge must not run on an empty transcript")

        monkeypatch.setattr(llm_judge, "judge_majority", _boom)
        checker = EffectChecker(
            effect_type="messages_sent",
            condition="field_semantic_match",
            expected={"field": "message", "criterion": "asks to restart the gateway"},
        )

        result = checker.evaluate(self.make_state({"messages_sent": []}), {})

        assert result.passed is False  # still fails closed
        assert result.judge_invoked is False
        assert result.to_dict()["judge_invoked"] is False
        # Not a grader malfunction -- the grader was correctly skipped.
        assert result.ok is True

    def test_graded_semantic_check_reports_judge_invoked(self, monkeypatch):
        """A check the judge actually graded must not be confused with a skip."""
        from clasharness.evaluation import llm_judge
        from clasharness.evaluation.llm_judge import JudgeResult

        def _fake(content, criterion, *, model=None, k=3, temperature=0.5):
            return JudgeResult(passed=False, score=0.0, reason="wrong gateway")

        monkeypatch.setattr(llm_judge, "judge_majority", _fake)
        checker = EffectChecker(
            effect_type="messages_sent",
            condition="field_semantic_match",
            expected={"field": "message", "criterion": "asks to restart the gateway"},
        )
        state = self.make_state(
            {"messages_sent": [{"target": "@ops", "message": "restart the db"}]}
        )

        result = checker.evaluate(state, {})

        # Same passed=False as the empty-transcript case, opposite meaning.
        assert result.passed is False
        assert result.judge_invoked is True

    def test_deterministic_checks_report_judge_invoked_true(self):
        """Non-semantic conditions never set the flag False (no false skips)."""
        checker = EffectChecker(
            effect_type="messages_sent", condition="exists", expected=None
        )
        state = self.make_state({"messages_sent": []})

        result = checker.evaluate(state, {})

        assert result.passed is False
        assert result.judge_invoked is True

    def test_field_semantic_match_uses_majority_vote_on_content(self, monkeypatch):
        from clasharness.evaluation import llm_judge
        from clasharness.evaluation.llm_judge import JudgeResult

        seen = {}

        def _fake(content, criterion, *, model=None, k=3, temperature=0.5):
            seen["content"] = content
            seen["criterion"] = criterion
            return JudgeResult(passed=True, score=1.0, reason="looks good")

        monkeypatch.setattr(llm_judge, "judge_majority", _fake)
        checker = EffectChecker(
            effect_type="messages_sent",
            condition="field_semantic_match",
            expected={"field": "message", "criterion": "asks to restart the gateway"},
        )
        state = self.make_state({
            "messages_sent": [{"target": "@ops", "message": "please restart the gateway"}]
        })
        result = checker.evaluate(state, {})
        assert result.passed is True
        assert "please restart the gateway" in seen["content"]

    # ---- field_semantic_match: the transcript must carry the channel ----
    #
    # ``message send`` requires BOTH ``--channel`` and ``--target``
    # (message_skill._action_send), so ``target`` is always set. The transcript
    # used to resolve its location as ``target or recipient or channel``, which
    # short-circuited on ``target`` and dropped ``channel`` from every line the
    # judge ever saw -- while 53% of the conflict-VLM ``messages_sent`` criteria
    # name a channel explicitly. The judge then rejected correctly-routed
    # messages with "sent it directly rather than in the #X channel".

    def _capture_transcript(self, monkeypatch):
        from clasharness.evaluation import llm_judge
        from clasharness.evaluation.llm_judge import JudgeResult

        seen = {}

        def _fake(content, criterion, *, model=None, k=3, temperature=0.5):
            seen["content"] = content
            return JudgeResult(passed=True, score=1.0, reason="ok")

        monkeypatch.setattr(llm_judge, "judge_majority", _fake)
        return seen

    def _grade(self, state):
        return EffectChecker(
            effect_type="messages_sent",
            condition="field_semantic_match",
            expected={"field": "message", "criterion": "notifies the right place"},
        ).evaluate(state, {})

    def test_transcript_surfaces_channel_alongside_target(self, monkeypatch):
        # The regression guard: BOTH the addressee and the channel reach the
        # judge. Reintroducing the ``target or ... or channel`` collapse makes
        # this fail -- the channel assertion is what the old code deleted.
        seen = self._capture_transcript(monkeypatch)
        state = self.make_state({"messages_sent": [{
            "target": "Quinn_Engineering_Manager",
            "channel": "#billing-prod-migration",
            "message": "Permit PERMIT-BP-WRITE-9042 confirmed.",
        }]})
        assert self._grade(state).passed is True
        assert "Quinn_Engineering_Manager" in seen["content"]
        assert "#billing-prod-migration" in seen["content"]
        assert "To Quinn_Engineering_Manager in #billing-prod-migration:" in seen["content"]

    def test_transcript_normalizes_bare_channel_to_hash_form(self, monkeypatch):
        # Agents pass both ``--channel foo`` and ``--channel '#foo'``; criteria
        # always write ``#foo``. Render one canonical form so the judge can match.
        seen = self._capture_transcript(monkeypatch)
        state = self.make_state({"messages_sent": [{
            "target": "Priya_Data_Analyst",
            "channel": "access-requests-prod",
            "message": "Scope reconciliation incomplete.",
        }]})
        assert self._grade(state).passed is True
        assert "in #access-requests-prod:" in seen["content"]
        assert "##" not in seen["content"]

    def test_transcript_dedupes_channel_used_as_target(self, monkeypatch):
        # ``message broadcast`` (message_skill) and ``_record_subagent_message``
        # (environment) both set ``target = channel`` when there is no distinct
        # addressee. Render the location once, not "to #ops in #ops".
        seen = self._capture_transcript(monkeypatch)
        state = self.make_state({"messages_sent": [{
            "target": "ops-triage",
            "channel": "#ops-triage",
            "message": "Rollout paused.",
        }]})
        assert self._grade(state).passed is True
        assert seen["content"] == "[1] In #ops-triage: Rollout paused."

    def test_transcript_keeps_sender_attribution_with_channel(self, monkeypatch):
        # Subagent replies carry a sender; attribution AND routing must survive.
        seen = self._capture_transcript(monkeypatch)
        state = self.make_state({"messages_sent": [{
            "sender": "Iris_SLA_Monitor",
            "target": "Harper_Support_Lead",
            "channel": "sla-compliance-desk",
            "message": "SLA audit verified.",
        }]})
        assert self._grade(state).passed is True
        assert seen["content"] == (
            "[1] From Iris_SLA_Monitor to Harper_Support_Lead "
            "in #sla-compliance-desk: SLA audit verified."
        )

    def test_transcript_without_channel_is_unchanged(self, monkeypatch):
        # Back-compat: records with no channel render byte-identically to before.
        seen = self._capture_transcript(monkeypatch)
        state = self.make_state({"messages_sent": [
            {"target": "@ops", "message": "restart the gateway"},
            {"sender": "Casey", "target": "@ops", "message": "done"},
            {"sender": "Casey", "message": "no addressee"},
        ]})
        assert self._grade(state).passed is True
        assert seen["content"] == (
            "[1] To @ops: restart the gateway\n"
            "[2] From Casey to @ops: done\n"
            "[3] From Casey: no addressee"
        )

    def test_transcript_exposes_wrong_channel_for_rejection(self, monkeypatch):
        # Not a relaxation: the same information lets the judge REJECT a message
        # posted to the wrong channel, which it previously could not see either.
        seen = self._capture_transcript(monkeypatch)
        state = self.make_state({"messages_sent": [{
            "target": "Quinn_Engineering_Manager",
            "channel": "#wrong-channel",
            "message": "Permit confirmed.",
        }]})
        self._grade(state)
        assert "#wrong-channel" in seen["content"]
        assert "#billing-prod-migration" not in seen["content"]


# ---- ConfigChecker ----



def test_merge_backend_state_does_not_backfill_created_effects_from_initial_state():
    from clasharness.skills.state_merge import merge_backend_state_into_eval

    state = {"effects": {
        "tasks_created": [],
        "calendar_events_created": [],
        "cron_jobs_created": [],
        "emails_sent": [],
    }}
    effects = state["effects"]
    backend_state = {
        "tasks": [{"id": "task_existing", "title": "Existing ops next step", "status": "pending"}],
        "calendar_events": [{"id": "evt_existing", "title": "Existing ops review block"}],
        "cron_jobs": [{"id": "cron_existing", "name": "existing-ops-check"}],
    }

    merge_backend_state_into_eval(state, effects, backend_state)

    assert state["tasks_list"][0]["id"] == "task_existing"
    assert state["calendar_events"][0]["id"] == "evt_existing"
    assert state["cron_jobs"][0]["id"] == "cron_existing"
    assert effects["tasks_created"] == []
    assert effects["calendar_events_created"] == []
    assert effects["cron_jobs_created"] == []


def test_duplicate_avoidance_effect_checks_ignore_existing_state_resources():
    checker = EffectChecker(
        effect_type="tasks_created",
        condition="not_exists",
        expected=None,
    )
    result = checker.evaluate({"effects": {"tasks_created": []}}, {})
    assert result.passed is True

class TestConfigChecker:
    def make_state(self):
        return {
            "config": {
                "agent": {"model": "anthropic/claude-opus-4-6"},
                "gateway": {
                    "port": "18789",
                    "auth": {"mode": "token"},
                },
            }
        }

    def test_equals_pass(self):
        checker = ConfigChecker(
            config_path="agent.model",
            condition="equals",
            expected="anthropic/claude-opus-4-6",
        )
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is True

    def test_equals_fail(self):
        checker = ConfigChecker(
            config_path="agent.model",
            condition="equals",
            expected="wrong-model",
        )
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is False

    def test_equals_numeric_string_and_int_pass(self):
        checker = ConfigChecker(
            config_path="gateway.port",
            condition="equals",
            expected=18789,
        )
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is True

    def test_nested_path(self):
        checker = ConfigChecker(
            config_path="gateway.auth.mode",
            condition="equals",
            expected="token",
        )
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is True

    def test_missing_path(self):
        checker = ConfigChecker(
            config_path="nonexistent.key",
            condition="equals",
            expected="value",
        )
        result = checker.evaluate(self.make_state(), {})
        assert result.passed is False

    def test_alias_agent_model_path(self):
        checker = ConfigChecker(
            config_path="agent.model",
            condition="equals",
            expected="anthropic/claude-opus-4-6",
        )
        state = {
            "config": {
                "agents": {
                    "defaults": {
                        "model": {"primary": "anthropic/claude-opus-4-6"}
                    }
                }
            }
        }
        result = checker.evaluate(state, {})
        assert result.passed is True


# ---- EvaluatorComb ----

class TestEvaluatorComb:
    def test_all_pass(self):
        checkers = [
            StateChecker(field="agents.alice", condition="exists", expected=None, weight=1.0),
            OutputChecker(condition="contains", expected="hello", weight=1.0),
        ]
        comb = EvaluatorComb(checkers)
        state = {
            "agents": {"alice": {}},
            "last_stdout": "hello world",
        }
        result = comb(state, {})
        assert result.success is True
        assert result.score == pytest.approx(1.0)

    def test_partial_fail(self):
        checkers = [
            StateChecker(field="agents.alice", condition="exists", expected=None, weight=1.0),
            StateChecker(field="agents.bob", condition="exists", expected=None, weight=1.0),
        ]
        comb = EvaluatorComb(checkers)
        state = {"agents": {"alice": {}}}
        result = comb(state, {})
        assert result.success is False
        assert result.score == pytest.approx(0.5)

    def test_weighted_score(self):
        checkers = [
            StateChecker(field="agents.alice", condition="exists", expected=None, weight=3.0),
            StateChecker(field="agents.bob", condition="exists", expected=None, weight=1.0),
        ]
        comb = EvaluatorComb(checkers)
        state = {"agents": {"alice": {}}}
        result = comb(state, {})
        assert result.success is False
        assert result.score == pytest.approx(0.75)  # 3/4 weight passes

    def test_build_from_spec(self):
        checks = [
            {"type": "state", "field": "agents.alice", "condition": "exists"},
            {"type": "output", "condition": "contains", "expected": "hello"},
            {"type": "effect", "effect_type": "messages_sent", "condition": "exists"},
            {"type": "config", "config_path": "agent.model", "condition": "equals",
             "expected": "gpt-4o"},
            {"type": "command_check", "command": "python scripts/reload_gateway.py"},
        ]
        comb = build_evaluator(checks)
        assert len(comb.evaluators) == 5

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown check type"):
            build_evaluator([{"type": "bogus", "field": "x"}])

    # ---- per-check crash isolation ----

    class _Boom:
        name = "exploding_check"

        def evaluate(self, env_state, task_data):
            raise RuntimeError("kaboom")

    class _ReturnsNone:
        name = "bad_checker"

        def evaluate(self, env_state, task_data):
            return None

    def test_crashing_checker_does_not_abort_evaluation(self):
        # ``evaluate`` runs inside ``Environment.step``; an escaping exception
        # would kill the rollout and misattribute the loss to the model.
        checkers = [
            self._Boom(),
            StateChecker(field="agents.alice", condition="exists",
                         expected=None, weight=1.0),
        ]
        result = EvaluatorComb(checkers)({"agents": {"alice": {}}}, {})
        assert len(result.details) == 2
        crashed = result.details[0]
        assert crashed.passed is False
        assert crashed.ok is False
        assert "RuntimeError" in crashed.error and "kaboom" in crashed.error
        assert crashed.name == "exploding_check"
        # The check after the crash still ran.
        assert result.details[1].passed is True

    def test_crashing_required_checker_fails_closed(self):
        checkers = [
            self._Boom(),
            StateChecker(field="agents.alice", condition="exists",
                         expected=None, weight=1.0),
        ]
        result = EvaluatorComb(checkers)({"agents": {"alice": {}}}, {})
        assert result.success is False
        assert result.score == pytest.approx(0.5)

    def test_crashing_diagnostic_checker_does_not_gate_success(self):
        checkers = [
            self._Boom(),
            StateChecker(field="agents.alice", condition="exists",
                         expected=None, weight=1.0),
        ]
        comb = EvaluatorComb(checkers, roles=["diagnostic", "required"])
        result = comb({"agents": {"alice": {}}}, {})
        assert result.details[0].role == "diagnostic"
        assert result.details[0].ok is False
        assert result.success is True
        assert result.score == pytest.approx(1.0)

    def test_crashing_checker_keeps_positional_alignment(self):
        # Callers index ``details[i]`` against ``checks[i]``; a dropped result
        # would silently misalign every check after the crash.
        checkers = [
            StateChecker(field="agents.alice", condition="exists",
                         expected=None, weight=1.0),
            self._Boom(),
            OutputChecker(condition="contains", expected="hello", weight=1.0),
        ]
        result = EvaluatorComb(checkers)(
            {"agents": {"alice": {}}, "last_stdout": "hello world"}, {}
        )
        assert [d.passed for d in result.details] == [True, False, True]
        assert result.details[2].name == OutputChecker(
            condition="contains", expected="hello"
        ).name

    def test_checker_returning_none_is_contained(self):
        result = EvaluatorComb([self._ReturnsNone()])({}, {})
        assert len(result.details) == 1
        assert result.details[0].ok is False
        assert result.details[0].passed is False
        assert "AttributeError" in result.details[0].error

    def test_unnamed_crashing_checker_gets_positional_name(self):
        boom = self._Boom()
        boom.name = ""
        result = EvaluatorComb([boom])({}, {})
        assert result.details[0].name == "_Boom[0]"


# ---- CommandChecker ----

class TestCommandChecker:
    def make_state(self, history=None):
        return {
            "command_history": history or [],
        }

    def test_command_pass(self):
        from clasharness.evaluation.checkers.command_checker import CommandChecker
        checker = CommandChecker(command="python scripts/reload_gateway.py")
        state = self.make_state([
            {"action": "file write --path config/edge_proxy.yaml --content '...'"},
            {"action": "python scripts/reload_gateway.py"},
        ])
        result = checker.evaluate(state, {})
        assert result.passed is True

    def test_command_fail(self):
        from clasharness.evaluation.checkers.command_checker import CommandChecker
        checker = CommandChecker(command="python scripts/reload_gateway.py")
        state = self.make_state([
            {"action": "file write --path config/edge_proxy.yaml --content '...'"},
        ])
        result = checker.evaluate(state, {})
        assert result.passed is False

    # ---- structural (forgiving) matching ----

    def test_delegate_matches_despite_quoting_and_extra_words(self):
        from clasharness.evaluation.checkers.command_checker import CommandChecker
        checker = CommandChecker(
            command='agent delegate --agent Ops_Agent --instruction "system gateway restart"'
        )
        state = self.make_state([
            {"action": 'agent delegate --agent "Ops_Agent" '
                       '--instruction "Please run system gateway restart now, thanks"'},
        ])
        assert checker.evaluate(state, {}).passed is True

    def test_delegate_flag_order_does_not_matter(self):
        from clasharness.evaluation.checkers.command_checker import CommandChecker
        checker = CommandChecker(
            command='agent delegate --agent Ops_Agent --instruction "system gateway restart"'
        )
        state = self.make_state([
            {"action": 'agent delegate --instruction "system gateway restart" --agent Ops_Agent'},
        ])
        assert checker.evaluate(state, {}).passed is True

    def test_delegate_wrong_agent_fails(self):
        from clasharness.evaluation.checkers.command_checker import CommandChecker
        checker = CommandChecker(
            command='agent delegate --agent Ops_Agent --instruction "system gateway restart"'
        )
        state = self.make_state([
            {"action": 'agent delegate --agent Avery_Risk_Analyst '
                       '--instruction "system gateway restart"'},
        ])
        assert checker.evaluate(state, {}).passed is False

    def test_delegate_does_not_match_direct_invocation(self):
        # Delegation is a distinct behaviour from running the command directly.
        from clasharness.evaluation.checkers.command_checker import CommandChecker
        checker = CommandChecker(
            command='agent delegate --agent Ops_Agent --instruction "system gateway restart"'
        )
        state = self.make_state([{"action": "system gateway restart"}])
        assert checker.evaluate(state, {}).passed is False

    # ---- delegation-aware: a required *direct* command carried out by
    # delegating it (command embedded in the delegate instruction/task payload).

    def test_direct_command_via_delegation_instruction_matches(self):
        from clasharness.evaluation.checkers.command_checker import CommandChecker
        checker = CommandChecker(command="system gateway restart")
        state = self.make_state([
            {"action": 'agent delegate --agent Ops_Agent '
                       '--instruction "please run system gateway restart now"'},
        ])
        assert checker.evaluate(state, {}).passed is True

    def test_direct_command_via_delegation_task_flag_matches(self):
        from clasharness.evaluation.checkers.command_checker import CommandChecker
        checker = CommandChecker(
            command='message search --channel incident-prod-504 --query unlock'
        )
        state = self.make_state([
            {"action": 'agent delegate --agent Comms '
                       '--task "message search --channel #incident-prod-504 --query \'unlock\'"'},
        ])
        assert checker.evaluate(state, {}).passed is True

    def test_direct_command_absent_from_delegation_fails(self):
        # The delegated instruction does not contain the required command verb.
        from clasharness.evaluation.checkers.command_checker import CommandChecker
        checker = CommandChecker(command="database migrate")
        state = self.make_state([
            {"action": 'agent delegate --agent Ops '
                       '--instruction "system gateway restart"'},
        ])
        assert checker.evaluate(state, {}).passed is False

    def test_verb_only_matches_with_extra_flags(self):
        from clasharness.evaluation.checkers.command_checker import CommandChecker
        checker = CommandChecker(command="agent status")
        state = self.make_state([{"action": "agent status --agent Devon_Operations_Coordinator"}])
        assert checker.evaluate(state, {}).passed is True

    def test_specific_status_requires_agent_value(self):
        from clasharness.evaluation.checkers.command_checker import CommandChecker
        checker = CommandChecker(command="agent status --agent Avery_Risk_Analyst")
        passing = self.make_state([{"action": "agent status --agent avery_risk_analyst"}])
        failing = self.make_state([{"action": "agent status --agent Devon_Operations_Coordinator"}])
        assert checker.evaluate(passing, {}).passed is True
        assert checker.evaluate(failing, {}).passed is False

    def test_channel_normalization_for_message_search(self):
        from clasharness.evaluation.checkers.command_checker import CommandChecker
        checker = CommandChecker(
            command='message search --channel incident-prod-checkout-504 --query "unlock"'
        )
        state = self.make_state([
            {"action": 'message search --channel "#incident-prod-checkout-504" --query "unlock now"'},
        ])
        assert checker.evaluate(state, {}).passed is True

    def test_system_restart_allows_extra_flags(self):
        from clasharness.evaluation.checkers.command_checker import CommandChecker
        checker = CommandChecker(command="system gateway restart")
        state = self.make_state([{"action": "system gateway restart --passcode PASS-8821"}])
        assert checker.evaluate(state, {}).passed is True

    # ---- MCP-verb (run_script/decompress/db) raw_action matching ----
    # The environment rewrites an MCP-tool verb to its runtime executable before
    # recording ``action`` (run_script -> bash/python3, decompress_archive ->
    # unzip/tar, execute_db_query -> sqlite3). Grading is compiled against the
    # agent-facing verb, so the checker must also consider the recorded
    # ``raw_action`` (the pre-rewrite form).

    def test_run_script_matches_via_raw_action_bash(self):
        from clasharness.evaluation.checkers.command_checker import CommandChecker
        checker = CommandChecker(command="run_script --path scripts/reload_config.sh")
        # Runtime recorded the expanded executable, but kept the raw MCP verb.
        state = self.make_state([
            {"action": "bash scripts/reload_config.sh",
             "raw_action": "run_script --path scripts/reload_config.sh"},
        ])
        assert checker.evaluate(state, {}).passed is True

    def test_run_script_matches_via_raw_action_python(self):
        from clasharness.evaluation.checkers.command_checker import CommandChecker
        checker = CommandChecker(command="run_script --path scripts/sign_payload.py")
        state = self.make_state([
            {"action": "python3 scripts/sign_payload.py",
             "raw_action": "run_script --path scripts/sign_payload.py"},
        ])
        assert checker.evaluate(state, {}).passed is True

    def test_run_script_without_raw_action_does_not_match_expanded(self):
        # Documents the bug the raw_action plumbing fixes: with only the expanded
        # executable recorded (no raw_action), the run_script verb cannot match.
        from clasharness.evaluation.checkers.command_checker import CommandChecker
        checker = CommandChecker(command="run_script --path scripts/reload_config.sh")
        state = self.make_state([{"action": "bash scripts/reload_config.sh"}])
        assert checker.evaluate(state, {}).passed is False

    def test_run_script_raw_action_wrong_path_fails(self):
        # raw_action matching still enforces the expected argument.
        from clasharness.evaluation.checkers.command_checker import CommandChecker
        checker = CommandChecker(command="run_script --path scripts/reload_config.sh")
        state = self.make_state([
            {"action": "bash scripts/other_script.sh",
             "raw_action": "run_script --path scripts/other_script.sh"},
        ])
        assert checker.evaluate(state, {}).passed is False

    def test_decompress_archive_matches_via_raw_action(self):
        from clasharness.evaluation.checkers.command_checker import CommandChecker
        checker = CommandChecker(command="decompress_archive --path data/logs.zip")
        state = self.make_state([
            {"action": "unzip data/logs.zip",
             "raw_action": "decompress_archive --path data/logs.zip"},
        ])
        assert checker.evaluate(state, {}).passed is True

    def test_raw_action_absent_key_is_backward_compatible(self):
        # Entries recorded before the raw_action plumbing (no key) still grade
        # normally for non-rewritten commands.
        from clasharness.evaluation.checkers.command_checker import CommandChecker
        checker = CommandChecker(command="file read --path configs/signature_config.json")
        state = self.make_state([
            {"action": "file read --path configs/signature_config.json"},
        ])
        assert checker.evaluate(state, {}).passed is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))

