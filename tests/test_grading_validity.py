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

"""Phase 3: grading-validity tests (no vacuous pass, de-hardcoded checker, unified judge)."""

from __future__ import annotations

import inspect

import clasharness.evaluation.llm_judge as lj
from clasharness.evaluation.checkers.state_checker import StateChecker
from clasharness.evaluation.evaluator import EvaluatorComb, build_evaluator


# --------------------------------------------------------------------------- #
# B1: empty / zero-weight check sets must not pass vacuously                   #
# --------------------------------------------------------------------------- #


def test_empty_evaluator_is_not_a_vacuous_pass():
    result = EvaluatorComb([])({}, {})
    assert result.success is False
    assert result.score == 0.0
    assert result.metadata.get("no_effective_checks") is True


def test_zero_weight_checks_do_not_pass_vacuously():
    # A passing check that carries zero weight contributes no signal.
    comb = EvaluatorComb(
        [StateChecker(field="x", condition="exists", expected=None, weight=0.0)]
    )
    result = comb({"x": 1}, {})
    assert result.success is False
    assert result.metadata.get("no_effective_checks") is True


def test_build_evaluator_with_no_checks_is_not_a_pass():
    result = build_evaluator([])({}, {})
    assert result.success is False


def test_real_checks_still_pass():
    comb = EvaluatorComb(
        [StateChecker(field="x", condition="exists", expected=None, weight=1.0)]
    )
    result = comb({"x": 1}, {})
    assert result.success is True
    assert result.score == 1.0


# --------------------------------------------------------------------------- #
# ①: check roles decouple success from score                                  #
# --------------------------------------------------------------------------- #


def _passing():
    return StateChecker(field="x", condition="exists", expected=None, weight=1.0)


def _failing():
    return StateChecker(field="missing", condition="exists", expected=None, weight=1.0)


def test_diagnostic_check_does_not_gate_success_or_score():
    # A failing diagnostic must neither fail the round nor lower the score.
    comb = EvaluatorComb([_passing(), _failing()], roles=["required", "diagnostic"])
    result = comb({"x": 1}, {})
    assert result.success is True
    assert result.score == 1.0
    # The diagnostic result is still recorded (observability).
    roles = {d.role for d in result.details}
    assert roles == {"required", "diagnostic"}


def test_scored_check_lowers_score_but_does_not_fail_success():
    # required passes (gates success); scored fails (partial credit only).
    comb = EvaluatorComb([_passing(), _failing()], roles=["required", "scored"])
    result = comb({"x": 1}, {})
    assert result.success is True  # only the required check gates success
    assert result.score == 0.5  # scored failure halves the weighted score


def test_required_failure_fails_success():
    comb = EvaluatorComb([_passing(), _failing()], roles=["required", "required"])
    result = comb({"x": 1}, {})
    assert result.success is False
    assert result.score == 0.5


def test_diagnostic_only_check_set_is_not_a_vacuous_pass():
    comb = EvaluatorComb([_passing()], roles=["diagnostic"])
    result = comb({"x": 1}, {})
    assert result.success is False
    assert result.metadata.get("no_effective_checks") is True


def test_scored_only_check_set_cannot_certify_success():
    # Scored checks contribute a score but cannot gate success on their own.
    comb = EvaluatorComb([_passing()], roles=["scored"])
    result = comb({"x": 1}, {})
    assert result.success is False
    assert result.score == 1.0
    assert result.metadata.get("no_required_checks") is True


def test_build_evaluator_threads_and_validates_roles():
    comb = build_evaluator(
        [
            {"type": "state", "field": "x", "condition": "exists", "role": "required"},
            {"type": "state", "field": "y", "condition": "exists", "role": "diagnostic"},
        ]
    )
    assert comb.roles == ["required", "diagnostic"]
    # Default role is required (backward compatible).
    default = build_evaluator([{"type": "state", "field": "x", "condition": "exists"}])
    assert default.roles == ["required"]


def test_build_evaluator_rejects_unknown_role():
    import pytest

    with pytest.raises(ValueError, match="Unknown check role"):
        build_evaluator(
            [{"type": "state", "field": "x", "condition": "exists", "role": "bogus"}]
        )


def test_check_result_role_is_serialized():
    comb = EvaluatorComb([_passing()], roles=["required"])
    result = comb({"x": 1}, {})
    assert result.details[0].to_dict()["role"] == "required"


# --------------------------------------------------------------------------- #
# B2: no scenario-specific hardcodes remain in the generic effect checker      #
# --------------------------------------------------------------------------- #


def test_effect_checker_has_no_scenario_hardcodes():
    import clasharness.evaluation.checkers.effect_checker as ec

    src = inspect.getsource(ec)
    assert "proxy_key_written" not in src
    assert "decrypt_key_2026" not in src
    assert "datacenter_monitor_db_pool_utilization_20260330" not in src
    assert '"r05"' not in src and "'r05'" not in src


# --------------------------------------------------------------------------- #
# B3: unified LLM judge — provider selection, fail-closed, caching            #
# --------------------------------------------------------------------------- #


def _clear_judge_env(monkeypatch):
    # EVERY variable `resolve_provider` auto-detects from. Leaving any of them
    # set makes "no provider configured" mean "whatever this developer happens
    # to have exported", so the test would pass or fail depending on the machine
    # -- OPENAI_API_KEY especially, which is exported in a great many shells.
    # The API-key variables are named via the module constants rather than
    # spelled out, so they cannot drift if a variable is renamed.
    for var in ("LLM_JUDGE_PROVIDER", "LLM_JUDGE_MODEL",
                "GEMINI_API_KEY", "VERTEX_PROJECT", "ANTHROPIC_API_KEY",
                lj._BEDROCK_GPT_API_KEY_ENV, lj._OPENAI_API_KEY_ENV):
        monkeypatch.delenv(var, raising=False)


def test_judge_fail_closed_when_no_provider(monkeypatch):
    _clear_judge_env(monkeypatch)
    lj.clear_cache()
    result = lj.judge("some content", "some criterion")
    assert result.ok is False
    assert result.passed is False
    assert result.error  # diagnostic message present


def test_resolve_provider_precedence(monkeypatch):
    _clear_judge_env(monkeypatch)
    assert lj.resolve_provider() == ""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert lj.resolve_provider() == "anthropic"
    monkeypatch.setenv("VERTEX_PROJECT", "p")
    assert lj.resolve_provider() == "gemini"  # VERTEX wins over anthropic
    monkeypatch.setenv("LLM_JUDGE_PROVIDER", "anthropic")
    assert lj.resolve_provider() == "anthropic"  # explicit override wins


def test_pick_model_respects_provider_family(monkeypatch):
    _clear_judge_env(monkeypatch)
    # A claude model requested under a gemini provider falls back to gemini default.
    assert lj._pick_model("gemini", "claude-haiku-4-5") == lj._GEMINI_DEFAULT
    assert lj._pick_model("gemini", "gemini-2.5-flash") == "gemini-2.5-flash"
    assert lj._pick_model("anthropic", "claude-haiku-4-5") == "claude-haiku-4-5"
    assert lj._pick_model("anthropic", "gemini-2.5-flash") == lj._ANTHROPIC_DEFAULT


def test_judge_caches_identical_calls(monkeypatch):
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("VERTEX_PROJECT", "dummy")
    lj.clear_cache()

    calls = {"n": 0}

    def fake_gemini(content, criterion, model, temperature=0.0):
        calls["n"] += 1
        return lj.JudgeResult(passed=True, score=1.0, reason="ok")

    monkeypatch.setattr(lj, "_judge_gemini", fake_gemini)

    r1 = lj.judge("content", "criterion")
    r2 = lj.judge("content", "criterion")
    assert r1.passed and r2.passed
    assert calls["n"] == 1, "identical judgements must be served from cache"

    # Different content bypasses the cache.
    lj.judge("other content", "criterion")
    assert calls["n"] == 2


def test_parse_judge_text_variants():
    good = lj._parse_judge_text('{"passed": true, "score": 0.5, "reason": "x"}')
    assert good.passed is True and good.score == 0.5

    embedded = lj._parse_judge_text('prefix {"passed": false, "reason": "no"} suffix')
    assert embedded.passed is False

    # A bare JSON boolean is off-contract but unambiguous -- still honoured.
    bare_true = lj._parse_judge_text("true")
    assert bare_true.ok is True and bare_true.passed is True
    bare_false = lj._parse_judge_text("false")
    assert bare_false.ok is True and bare_false.passed is False

    # Prose must NOT be mined for a verdict. This previously returned
    # passed=True (the reply merely contains the word "true"), and because it
    # carried the default ok=True the fabricated pass was cached and reused.
    prose = lj._parse_judge_text("this looks true to me")
    assert prose.ok is False, "an unparseable reply is a grader failure"
    assert prose.passed is False, "a check must never pass on a guessed verdict"
    assert "malformed judge reply" in prose.error

    # The dangerous shape: prose that would have been mined as a pass while
    # describing a FAILURE.
    misleading = lj._parse_judge_text(
        "It is not true that the agent notified anyone."
    )
    assert misleading.ok is False and misleading.passed is False


def test_malformed_judge_reply_is_not_cached(monkeypatch):
    """A fabricated verdict must not be persisted for every identical input.

    ``judge()`` only writes ``result.ok`` results to the shared cache, so making
    a malformed reply ``ok=False`` also stops one bad parse from contaminating
    every later check with the same (content, criterion).
    """
    lj.clear_cache()
    monkeypatch.setenv("VERTEX_PROJECT", "p")
    monkeypatch.delenv("LLM_JUDGE_PROVIDER", raising=False)
    calls = {"n": 0}

    def _bad(content, criterion, model, temperature=0.0):
        calls["n"] += 1
        return lj._parse_judge_text("this looks true to me")

    monkeypatch.setattr(lj, "_judge_gemini", _bad)

    first = lj.judge("content", "criterion")
    second = lj.judge("content", "criterion")

    assert first.ok is False and first.passed is False
    assert second.ok is False
    assert calls["n"] == 2, "a failed parse must not be served from cache"
