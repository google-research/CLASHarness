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

"""Tests for the judge-health aggregation + report block in eval_report."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train_eval.eval_report import (  # noqa: E402
    EvalSummary,
    EpisodeResult,
    JudgeHealth,
    _compute_judge_health,
    _judge_was_invoked,
)


def _episode(check_results: list[dict]) -> EpisodeResult:
    return EpisodeResult(
        task_id="t",
        agent_name="a",
        success=True,
        score=1.0,
        steps=1,
        duration_s=0.0,
        metadata={"check_results": check_results},
    )


def _judge_check(message: str, ok: bool = True) -> dict:
    return {"name": "field_semantic_match", "passed": True, "message": message, "ok": ok}


# ------------------------------------------------------------------ #
# _compute_judge_health                                                 #
# ------------------------------------------------------------------ #


def test_compute_judge_health_categorises_every_check():
    results = [
        _episode([
            # Deterministic (non-judge) checks are ignored.
            {"name": "x", "passed": True, "message": "emails_sent occurred: True"},
            _judge_check("LLM semantic match: ok [self-consistency 3/3 passed]"),
            _judge_check("LLM semantic match: ok [self-consistency 2/3 passed]"),
            _judge_check("LLM semantic match: No LLM judge configured.", ok=False),
            _judge_check(
                "LLM semantic match: no 'body' content in messages_sent to "
                "evaluate (empty transcript)"
            ),
            _judge_check("LLM semantic match: single shot verdict"),
        ]),
    ]
    h = _compute_judge_health(results)
    assert h.judge_checks == 5  # deterministic check excluded
    assert h.infra_failures == 1
    assert h.fail_closed_empty == 1
    assert h.voted == 2
    assert h.split_votes == 1
    assert h.unanimous_votes == 1
    assert h.single_shot == 1


def test_compute_judge_health_missing_ok_defaults_true():
    # Records predating the ``ok`` field must not be miscounted as failures.
    results = [_episode([{"name": "s", "passed": True,
                          "message": "LLM semantic match: fine"}])]
    h = _compute_judge_health(results)
    assert h.judge_checks == 1
    assert h.infra_failures == 0
    assert h.single_shot == 1


def test_compute_judge_health_empty_when_no_judge_checks():
    results = [_episode([{"name": "x", "passed": True, "message": "count == 2: True"}])]
    h = _compute_judge_health(results)
    assert h.judge_checks == 0
    assert h.degraded is False


# ------------------------------------------------------------------ #
# JudgeHealth.degraded threshold                                        #
# ------------------------------------------------------------------ #


def test_degraded_above_one_percent():
    assert JudgeHealth(judge_checks=100, infra_failures=2).degraded is True


def test_not_degraded_at_or_below_one_percent():
    # Exactly 1% is not > 1%.
    assert JudgeHealth(judge_checks=100, infra_failures=1).degraded is False


# ------------------------------------------------------------------ #
# print_report rendering                                                #
# ------------------------------------------------------------------ #


def _summary_with(check_results_per_episode: list[list[dict]], **kw) -> EvalSummary:
    s = EvalSummary(
        agent_name="agent", split="test", total=0, passed=0, total_score=0.0, **kw
    )
    for crs in check_results_per_episode:
        s.results.append(_episode(crs))
        s.total += 1
        s.passed += 1
        s.total_score += 1.0
    return s


def test_print_report_shows_judge_line_and_health(capsys):
    s = _summary_with(
        [[_judge_check("LLM semantic match: ok [self-consistency 3/3 passed]")]],
        judge_provider="gemini",
        judge_model="gemini-3.1-flash-lite",
    )
    s.print_report()
    out = capsys.readouterr().out
    assert "Judge : gemini / gemini-3.1-flash-lite" in out
    assert "Judge health :" in out
    assert "voted=1" in out
    assert "takes no temperature" not in out


@pytest.mark.parametrize("provider, model, noted", [
    ("vertex-anthropic", "claude-sonnet-5-5", True),
    ("anthropic", "claude-opus-4-8", True),
    ("vertex-anthropic", "claude-sonnet-4@20250514", False),
    # Same model name under a non-Claude provider: the provider decides.
    ("openai", "claude-sonnet-5-5", False),
])
def test_print_report_notes_a_judge_without_temperature(
    capsys, provider, model, noted
):
    # The default 3 votes are greedy only where temperature 0 can be sent.
    s = _summary_with(
        [[_judge_check("LLM semantic match: ok [self-consistency 3/3 passed]")]],
        judge_provider=provider,
        judge_model=model,
    )
    s.print_report()
    out = capsys.readouterr().out
    assert f"Judge : {provider} / {model}" in out
    assert ("takes no temperature" in out) is noted


def test_print_report_warns_when_degraded(capsys):
    # 2 infra failures out of 2 judge checks = 100% > 1% -> loud warning.
    s = _summary_with([
        [_judge_check("LLM semantic match: boom", ok=False)],
        [_judge_check("LLM semantic match: boom", ok=False)],
    ])
    s.print_report()
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "unreliable" in captured.out
    # Also emitted to stderr for fail-loud visibility.
    assert "WARNING" in captured.err


def test_print_report_no_warning_when_healthy(capsys):
    s = _summary_with([
        [_judge_check("LLM semantic match: ok [self-consistency 3/3 passed]")],
    ])
    s.print_report()
    captured = capsys.readouterr()
    assert "WARNING" not in captured.out
    assert "Judge health :" in captured.out


# ------------------------------------------------------------------ #
# _judge_was_invoked: structured flag first, prose only as legacy fallback
# ------------------------------------------------------------------ #
#
# "The judge was never called" and "the judge was called and said no" are
# different measurements that both land as passed=False. Deriving which one
# happened by substring-matching the human-facing message is a latent bug: the
# strings are prose, so rewording one silently reclassifies every fail-closed
# check as a real semantic failure. The checkers now emit ``judge_invoked``;
# the text match survives only for result files written before it existed.


def test_structured_flag_false_marks_fail_closed():
    assert _judge_was_invoked({"judge_invoked": False}, "any prose at all") is False


def test_structured_flag_true_wins_over_legacy_prose():
    """A reworded message must not override the field the checker actually set."""
    check = {"judge_invoked": True}
    assert _judge_was_invoked(check, "LLM semantic match: no file /x to evaluate")


def test_structured_flag_survives_a_message_reword():
    """The whole point of the field: classification stops depending on wording."""
    reworded = "LLM semantic match: nothing was produced for /report.md"
    assert _judge_was_invoked({"judge_invoked": False}, reworded) is False


def test_legacy_record_falls_back_to_empty_transcript_marker():
    msg = "LLM semantic match: no 'body' content in messages_sent to evaluate (empty transcript)"
    assert _judge_was_invoked({}, msg) is False


def test_legacy_record_falls_back_to_missing_file_marker():
    assert _judge_was_invoked({}, "LLM semantic match: no file /r.md to evaluate") is False


def test_legacy_record_falls_back_to_empty_criterion_marker():
    assert _judge_was_invoked({}, "LLM semantic match: empty criterion for /r.md") is False


def test_legacy_record_with_an_ordinary_message_counts_as_invoked():
    assert _judge_was_invoked({}, "LLM semantic match: states the verdict") is True


def test_explicit_none_is_treated_as_a_legacy_record():
    """An absent value must not be read as a falsy 'not invoked'."""
    assert _judge_was_invoked({"judge_invoked": None}, "LLM semantic match: fine") is True


# ------------------------------------------------------------------ #
# _compute_judge_health consumes the structured flag
# ------------------------------------------------------------------ #


def test_health_uses_structured_flag_over_message_text():
    check = _judge_check("LLM semantic match: solver produced nothing")
    check["judge_invoked"] = False
    h = _compute_judge_health([_episode([check])])
    assert h.fail_closed_empty == 1
    assert h.single_shot == 0 and h.voted == 0


def test_health_does_not_mistake_prose_for_a_skipped_judge():
    """A judged check whose reason merely mentions a missing file is not skipped."""
    check = _judge_check("LLM semantic match: report says no file was archived")
    check["judge_invoked"] = True
    h = _compute_judge_health([_episode([check])])
    assert h.fail_closed_empty == 0
    assert h.single_shot == 1


def test_health_still_classifies_legacy_records_without_the_flag():
    h = _compute_judge_health([_episode([
        _judge_check(
            "LLM semantic match: no 'body' content in messages_sent to "
            "evaluate (empty transcript)"
        ),
    ])])
    assert h.fail_closed_empty == 1


def test_infra_failure_outranks_the_structured_flag():
    """``ok=False`` means the grader broke; that is not a fail-closed verdict."""
    check = _judge_check("LLM semantic match: boom", ok=False)
    check["judge_invoked"] = False
    h = _compute_judge_health([_episode([check])])
    assert h.infra_failures == 1 and h.fail_closed_empty == 0
