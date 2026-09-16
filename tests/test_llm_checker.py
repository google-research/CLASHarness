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

"""Tests for LLMChecker (fuzzy semantic evaluation)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import clasharness.evaluation.llm_judge as llm_judge
from clasharness.evaluation.checkers.llm_checker import (
    LLMChecker,
    _build_evidence,
    _normalise,
    _parse_response,
)
from clasharness.evaluation.evaluator import build_evaluator


@pytest.fixture(autouse=True)
def _clear_judge_cache():
    """LLMChecker now routes through the shared, cached llm_judge."""
    llm_judge.clear_cache()
    yield
    llm_judge.clear_cache()


# ------------------------------------------------------------------ #
# _normalise                                                            #
# ------------------------------------------------------------------ #

class TestNormalise:
    def test_basic(self):
        d = _normalise({"passed": True, "score": 0.9, "reason": "ok"})
        assert d["passed"] is True
        assert d["score"] == pytest.approx(0.9)

    def test_clamps_score_above_1(self):
        d = _normalise({"passed": True, "score": 1.5, "reason": ""})
        assert d["score"] == pytest.approx(1.0)

    def test_clamps_score_below_0(self):
        d = _normalise({"passed": False, "score": -0.5, "reason": ""})
        assert d["score"] == pytest.approx(0.0)

    def test_defaults_for_missing_keys(self):
        d = _normalise({})
        assert d["passed"] is False
        assert d["score"] == pytest.approx(0.0)
        assert d["reason"] == ""


# ------------------------------------------------------------------ #
# _parse_response                                                       #
# ------------------------------------------------------------------ #

class TestParseResponse:
    def test_valid_json(self):
        r = _parse_response('{"passed": true, "score": 1.0, "reason": "done"}')
        assert r["passed"] is True
        assert r["score"] == pytest.approx(1.0)

    def test_valid_json_false(self):
        r = _parse_response('{"passed": false, "score": 0.0, "reason": "nope"}')
        assert r["passed"] is False

    def test_partial_score(self):
        r = _parse_response('{"passed": true, "score": 0.6, "reason": "partial"}')
        assert r["score"] == pytest.approx(0.6)

    def test_json_embedded_in_text(self):
        text = 'My evaluation: {"passed": true, "score": 0.8, "reason": "looks good"}'
        r = _parse_response(text)
        assert r["passed"] is True
        assert r["score"] == pytest.approx(0.8)

    def test_fallback_true_keyword(self):
        r = _parse_response("The criterion is satisfied. true overall.")
        assert r["passed"] is True

    def test_fallback_false_keyword(self):
        r = _parse_response("This is false. The task was not completed.")
        assert r["passed"] is False

    def test_malformed_json_fallback(self):
        r = _parse_response("{broken json true}")
        # Should not raise
        assert isinstance(r["passed"], bool)

    def test_empty_string_fallback(self):
        r = _parse_response("")
        assert isinstance(r["passed"], bool)


# ------------------------------------------------------------------ #
# _build_evidence (criterion + JSON format are now owned by llm_judge)  #
# ------------------------------------------------------------------ #

class TestBuildEvidence:
    def _make_state(self, **overrides):
        base = {
            "task_instruction": "Send an email to alice@example.com",
            "last_stdout": "Email sent successfully",
            "last_exit_code": 0,
            "effects": {"emails_sent": [{"to": "alice@example.com"}]},
            "command_history": [
                {"action": "email send --to alice@example.com", "stdout": "Email sent", "exit_code": 0}
            ],
        }
        base.update(overrides)
        return base

    def test_evidence_contains_task_instruction(self):
        evidence = _build_evidence(self._make_state(), {})
        assert "alice@example.com" in evidence

    def test_evidence_contains_effects(self):
        evidence = _build_evidence(self._make_state(), {})
        assert "emails_sent" in evidence

    def test_evidence_contains_command_history(self):
        evidence = _build_evidence(self._make_state(), {})
        assert "email send" in evidence

    def test_evidence_with_no_history(self):
        state = self._make_state(command_history=[])
        evidence = _build_evidence(state, {})
        assert "no commands" in evidence

    def test_evidence_falls_back_to_task_data_instruction(self):
        state = self._make_state(task_instruction="")
        task_data = {"public": {"instruction": "From task_data"}}
        evidence = _build_evidence(state, task_data)
        assert "From task_data" in evidence

    def test_evidence_long_history_truncated(self):
        history = [
            {"action": f"cmd {i}", "stdout": "ok", "exit_code": 0}
            for i in range(20)
        ]
        state = self._make_state(command_history=history)
        evidence = _build_evidence(state, {})
        # Should only include last 8
        assert "cmd 19" in evidence
        assert "cmd 0" not in evidence


# ------------------------------------------------------------------ #
# LLMChecker.from_spec                                                  #
# ------------------------------------------------------------------ #

class TestLLMCheckerFromSpec:
    def test_basic(self):
        c = LLMChecker.from_spec({"criterion": "agent completed the task"})
        assert c.criterion == "agent completed the task"
        assert c.model == LLMChecker.DEFAULT_MODEL
        assert c.weight == pytest.approx(1.0)

    def test_custom_fields(self):
        c = LLMChecker.from_spec({
            "criterion": "email found",
            "name": "my check",
            "weight": 2.5,
            "model": "claude-haiku-4-5-20251001",
            "temperature": 0.2,
        })
        assert c.name == "my check"
        assert c.weight == pytest.approx(2.5)
        assert c.temperature == pytest.approx(0.2)

    def test_default_name_derived_from_criterion(self):
        c = LLMChecker.from_spec({"criterion": "the agent sent a message"})
        assert "the agent sent a message" in c.name


# ------------------------------------------------------------------ #
# LLMChecker.evaluate — graceful failure modes                          #
# ------------------------------------------------------------------ #

class TestLLMCheckerGracefulFailure:
    _state = {
        "task_instruction": "Do X",
        "last_stdout": "done",
        "last_exit_code": 0,
        "effects": {},
        "command_history": [],
    }

    def test_no_anthropic_package(self):
        c = LLMChecker(criterion="task done")
        with patch.dict("os.environ", {"LLM_JUDGE_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "sk-test"}):
            with patch.dict("sys.modules", {"anthropic": None}):
                result = c.evaluate(self._state, {})
        assert result.passed is False
        assert "anthropic" in result.message.lower()

    def test_no_api_key(self):
        c = LLMChecker(criterion="task done")
        mock_pkg = MagicMock()
        with patch.dict("sys.modules", {"anthropic": mock_pkg}):
            with patch.dict("os.environ", {}, clear=True):
                # ANTHROPIC_API_KEY absent
                import os
                os.environ.pop("ANTHROPIC_API_KEY", None)
                result = c.evaluate(self._state, {})
        assert result.passed is False
        assert "ANTHROPIC_API_KEY" in result.message

    def test_api_exception(self):
        c = LLMChecker(criterion="task done")
        mock_pkg = MagicMock()
        mock_pkg.Anthropic.return_value.messages.create.side_effect = RuntimeError("network error")
        with patch.dict("sys.modules", {"anthropic": mock_pkg}):
            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test", "LLM_JUDGE_PROVIDER": "anthropic"}):
                result = c.evaluate(self._state, {})
        assert result.passed is False
        assert "network error" in result.message


# ------------------------------------------------------------------ #
# LLMChecker.evaluate — successful call (mocked Anthropic)             #
# ------------------------------------------------------------------ #

class TestLLMCheckerSuccess:
    def _mock_anthropic(self, response_text: str):
        mock_pkg = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=response_text)]
        mock_pkg.Anthropic.return_value.messages.create.return_value = mock_response
        return mock_pkg

    _state = {
        "task_instruction": "Add 'Team standup' to my calendar.",
        "last_stdout": "Event created: Team standup at 2026-03-10T09:00",
        "last_exit_code": 0,
        "effects": {"calendar_events_created": [{"title": "Team standup"}]},
        "command_history": [
            {"action": "calendar add-event --title 'Team standup' --start 2026-03-10T09:00",
             "stdout": "Event created", "exit_code": 0}
        ],
    }

    def test_passed_true(self):
        c = LLMChecker(criterion="agent added an event titled 'Team standup'")
        mock = self._mock_anthropic('{"passed": true, "score": 1.0, "reason": "event created"}')
        with patch.dict("sys.modules", {"anthropic": mock}):
            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test", "LLM_JUDGE_PROVIDER": "anthropic"}):
                result = c.evaluate(self._state, {})
        assert result.passed is True
        assert result.score == pytest.approx(1.0)

    def test_passed_false(self):
        c = LLMChecker(criterion="agent sent an email")
        mock = self._mock_anthropic('{"passed": false, "score": 0.0, "reason": "no email sent"}')
        with patch.dict("sys.modules", {"anthropic": mock}):
            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test", "LLM_JUDGE_PROVIDER": "anthropic"}):
                result = c.evaluate(self._state, {})
        assert result.passed is False
        assert result.score == pytest.approx(0.0)

    def test_partial_score(self):
        c = LLMChecker(criterion="agent added the event with correct attendees", weight=2.0)
        mock = self._mock_anthropic('{"passed": true, "score": 0.5, "reason": "title ok, no attendees"}')
        with patch.dict("sys.modules", {"anthropic": mock}):
            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test", "LLM_JUDGE_PROVIDER": "anthropic"}):
                result = c.evaluate(self._state, {})
        assert result.score == pytest.approx(0.5)
        assert result.weight == pytest.approx(2.0)

    def test_reason_in_message(self):
        c = LLMChecker(criterion="agent added an event")
        mock = self._mock_anthropic('{"passed": true, "score": 1.0, "reason": "calendar event found"}')
        with patch.dict("sys.modules", {"anthropic": mock}):
            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test", "LLM_JUDGE_PROVIDER": "anthropic"}):
                result = c.evaluate(self._state, {})
        assert "calendar event found" in result.message

    def test_malformed_response_fallback(self):
        c = LLMChecker(criterion="agent did something")
        mock = self._mock_anthropic("The agent definitely passed true here")
        with patch.dict("sys.modules", {"anthropic": mock}):
            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test", "LLM_JUDGE_PROVIDER": "anthropic"}):
                result = c.evaluate(self._state, {})
        assert isinstance(result.passed, bool)


# ------------------------------------------------------------------ #
# ok/error threading + shared "LLM semantic match:" prefix              #
# ------------------------------------------------------------------ #

class TestLLMCheckerThreadsOkError:
    _state = {
        "task_instruction": "Do X",
        "last_stdout": "done",
        "last_exit_code": 0,
        "effects": {},
        "command_history": [],
    }

    def test_infra_failure_threads_ok_false(self):
        """No judge configured -> ok=False (infra failure), not a content 0."""
        c = LLMChecker(criterion="task done")
        with patch.dict("os.environ", {}, clear=True):
            result = c.evaluate(self._state, {})
        assert result.ok is False
        assert result.passed is False
        assert result.error  # diagnostic reason retained
        # Shared prefix so the judge-health report can identify the check.
        assert result.message.startswith("LLM semantic match:")
        d = result.to_dict()
        assert d["ok"] is False
        assert d["error"]

    def test_success_path_is_ok_true_with_prefix(self):
        c = LLMChecker(criterion="agent added an event")
        mock_pkg = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text='{"passed": true, "score": 1.0, "reason": "found it"}')
        ]
        mock_pkg.Anthropic.return_value.messages.create.return_value = mock_response
        with patch.dict("sys.modules", {"anthropic": mock_pkg}):
            with patch.dict(
                "os.environ",
                {"ANTHROPIC_API_KEY": "sk-test", "LLM_JUDGE_PROVIDER": "anthropic"},
            ):
                result = c.evaluate(self._state, {})
        assert result.ok is True
        assert result.passed is True
        assert result.message.startswith("LLM semantic match:")
        assert "found it" in result.message


# ------------------------------------------------------------------ #
# build_evaluator integration                                           #
# ------------------------------------------------------------------ #

class TestBuildEvaluatorLLM:
    def test_llm_type_registered(self):
        comb = build_evaluator([
            {
                "type": "llm",
                "criterion": "agent completed the task",
                "name": "semantic check",
            }
        ])
        assert len(comb.evaluators) == 1
        assert isinstance(comb.evaluators[0], LLMChecker)

    def test_llm_mixed_with_other_checkers(self):
        from clasharness.evaluation.checkers.effect_checker import EffectChecker
        comb = build_evaluator([
            {"type": "effect", "effect_type": "emails_sent", "condition": "exists"},
            {"type": "llm", "criterion": "email content was professional"},
        ])
        assert len(comb.evaluators) == 2
        assert isinstance(comb.evaluators[1], LLMChecker)

    def test_unknown_type_still_raises(self):
        with pytest.raises(ValueError, match="Unknown check type"):
            build_evaluator([{"type": "bogus"}])


# ------------------------------------------------------------------ #
# env_state fields added by environment                                 #
# ------------------------------------------------------------------ #

class TestEnvironmentStateFields:
    """Verify task_instruction and command_history appear in env_state."""

    def test_task_instruction_in_state(self):
        from clasharness import make_env
        with make_env("calendar_add_event_1", mode="multi") as env:
            env.reset()
            env.step("calendar list")
            # Access env_state directly via evaluate path
            state = env._get_evaluation_state()
        assert state["task_instruction"] != ""
        assert "calendar" in state["task_instruction"].lower() or "block" in state["task_instruction"].lower()

    def test_command_history_populated(self):
        from clasharness import make_env
        with make_env("calendar_add_event_1", mode="multi") as env:
            env.reset()
            env.step("calendar list")
            env.step("calendar add-event --title 'X' --start 2026-03-10T09:00")
            state = env._get_evaluation_state()
        assert len(state["command_history"]) == 2
        assert state["command_history"][0]["action"] == "calendar list"
        assert state["command_history"][1]["action"] == "calendar add-event --title 'X' --start 2026-03-10T09:00"

    def test_command_history_cleared_on_reset(self):
        from clasharness import make_env
        with make_env("calendar_add_event_1", mode="multi") as env:
            env.reset()
            env.step("calendar list")
            env.reset()  # second reset
            state = env._get_evaluation_state()
        assert state["command_history"] == []

    def test_command_history_capped_at_50(self):
        from clasharness import make_env
        with make_env("calendar_add_event_1", mode="multi", max_steps=200) as env:
            env.reset()
            for _ in range(60):
                env.step("calendar list")
            state = env._get_evaluation_state()
        assert len(state["command_history"]) == 50


# ------------------------------------------------------------------ #
# evaluate() outer exception handler: infra failure, not a content 0    #
# ------------------------------------------------------------------ #
#
# _call_llm's own error paths already thread ok/error, but anything raising
# OUTSIDE them (evidence assembly, an unhandled judge fault) landed in the
# blanket ``except`` in evaluate(), which recorded ok=True and dropped the
# shared prefix -- so a melting-down judge was scored as a model that produced
# genuinely wrong content, AND was invisible to the judge-health report.


class TestLLMCheckerOuterExceptionIsInfraFailure:
    _state = {
        "task_instruction": "Do X",
        "last_stdout": "done",
        "last_exit_code": 0,
        "effects": {},
        "command_history": [],
    }

    def _exploding(self, monkeypatch, exc=RuntimeError("evidence blew up")):
        c = LLMChecker(criterion="task done")

        def _boom(*args, **kwargs):
            raise exc

        monkeypatch.setattr(c, "_call_llm", _boom)
        return c.evaluate(self._state, {})

    def test_marks_ok_false(self):
        import pytest as _pytest
        mp = _pytest.MonkeyPatch()
        try:
            result = self._exploding(mp)
        finally:
            mp.undo()
        assert result.ok is False
        assert result.passed is False

    def test_retains_the_exception_text_in_error(self):
        import pytest as _pytest
        mp = _pytest.MonkeyPatch()
        try:
            result = self._exploding(mp)
        finally:
            mp.undo()
        assert "evidence blew up" in result.error

    def test_keeps_the_shared_judge_prefix(self):
        """Without the prefix the check vanishes from the judge-health report."""
        import pytest as _pytest
        mp = _pytest.MonkeyPatch()
        try:
            result = self._exploding(mp)
        finally:
            mp.undo()
        assert result.message.startswith("LLM semantic match:")
        assert "evidence blew up" in result.message

    def test_is_counted_as_an_infra_failure_by_the_report(self):
        """End-to-end: the report must bucket it as infra, not a content miss."""
        from train_eval.eval_report import EpisodeResult, _compute_judge_health

        import pytest as _pytest
        mp = _pytest.MonkeyPatch()
        try:
            result = self._exploding(mp)
        finally:
            mp.undo()
        episode = EpisodeResult(
            task_id="t", agent_name="a", success=False, score=0.0, steps=1,
            duration_s=0.0, metadata={"check_results": [result.to_dict()]},
        )
        health = _compute_judge_health([episode])
        assert health.judge_checks == 1
        assert health.infra_failures == 1
        assert health.single_shot == 0
        assert health.fail_closed_empty == 0
