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

"""Tests for the failure-clustering aggregation + report block in eval_report."""

from __future__ import annotations

import os
import random
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train_eval.eval_report import (  # noqa: E402
    EvalSummary,
    EpisodeResult,
    FailureClustering,
    _compute_failure_clustering,
)


def _episode(check_results: list[dict]) -> EpisodeResult:
    return EpisodeResult(
        task_id="t",
        agent_name="a",
        success=all(
            c.get("passed", False) for c in check_results if isinstance(c, dict)
        ),
        score=1.0,
        steps=1,
        duration_s=0.0,
        metadata={"check_results": check_results},
    )


def _check(passed: bool, role: str = "required") -> dict:
    return {"name": "c", "passed": passed, "score": 1.0, "weight": 1.0,
            "message": "", "role": role, "ok": True}


def _task(k: int, failures: int = 0) -> EpisodeResult:
    """One episode with ``k`` required checks, ``failures`` of them failed."""
    return _episode([_check(i >= failures) for i in range(k)])


# ------------------------------------------------------------------ #
# _compute_failure_clustering: what gets counted                       #
# ------------------------------------------------------------------ #


def test_only_required_checks_are_counted():
    # ``scored`` and ``diagnostic`` never gate success, so a failure in either
    # must not register as a gating failure nor spoil the task's all-pass.
    fc = _compute_failure_clustering([
        _episode([
            _check(True),
            _check(False, role="scored"),
            _check(False, role="diagnostic"),
        ])
    ])
    assert fc.gating_checks == 1
    assert fc.failed_checks == 0
    assert fc.observed_all_pass == 1
    assert fc.tasks == 1


def test_missing_role_defaults_to_required():
    # Records predating the ``role`` field carried the historical gating
    # semantics (success = all(passed)); they must stay inside the metric.
    fc = _compute_failure_clustering([_episode([{"name": "c", "passed": False}])])
    assert fc.gating_checks == 1
    assert fc.failed_checks == 1
    assert fc.observed_all_pass == 0


def test_task_without_required_checks_is_skipped_entirely():
    # p ** 0 == 1 would credit such a task as a guaranteed pass and drag the
    # baseline up; it cannot pass at all, so it must not be in the universe.
    fc = _compute_failure_clustering([
        _episode([_check(True, role="diagnostic")]),
        _episode([]),
    ])
    assert fc.tasks == 0
    assert fc.gating_checks == 0
    assert fc.available is False


def test_malformed_records_are_ignored():
    fc = _compute_failure_clustering([
        _episode(["not-a-dict", _check(True)]),  # type: ignore[list-item]
        EpisodeResult(task_id="t", agent_name="a", success=True, score=1.0,
                      steps=1, duration_s=0.0,
                      metadata={"check_results": "not-a-list"}),
        EpisodeResult(task_id="t", agent_name="a", success=True, score=1.0,
                      steps=1, duration_s=0.0, metadata={}),
    ])
    assert fc.tasks == 1
    assert fc.gating_checks == 1


def test_sufficient_statistics_track_the_k_histogram():
    fc = _compute_failure_clustering([_task(3), _task(3, failures=2), _task(5)])
    assert fc.k_histogram == {3: 2, 5: 1}
    assert fc.tasks == 3
    assert fc.gating_checks == 11
    assert fc.failed_checks == 2
    assert fc.sum_f_sq == 4
    assert fc.observed_all_pass == 2


# ------------------------------------------------------------------ #
# The independence baseline                                            #
# ------------------------------------------------------------------ #


def test_independent_rate_matches_a_permutation_baseline():
    """The analytic baseline must agree with shuffling failures across checks.

    This is the load-bearing claim: ``mean_i(p ** k_i)`` is used *instead of* a
    sampled permutation baseline, so it has to reproduce one.
    """
    rng = random.Random(0)
    ks = [rng.choice([2, 3, 4, 5, 7]) for _ in range(600)]
    episodes, flat = [], []
    for k in ks:
        outcomes = [rng.random() < 0.72 for _ in range(k)]
        flat += outcomes
        episodes.append(_episode([_check(o) for o in outcomes]))
    fc = _compute_failure_clustering(episodes)

    total = 0.0
    trials = 400
    for _ in range(trials):
        rng.shuffle(flat)
        i = all_pass = 0
        for k in ks:
            if all(flat[i:i + k]):
                all_pass += 1
            i += k
        total += all_pass / len(ks)
    permuted = total / trials

    assert abs(fc.independent_rate - permuted) < 0.005


def test_independent_rate_averages_over_k_and_is_not_p_to_the_mean_k():
    """Jensen guard: ``p ** mean(k)`` is a different, smaller number.

    ``x -> p ** x`` is convex, so collapsing the histogram to its mean
    understates the baseline and inflates ``excess``. Mixed check counts must
    expose the difference.
    """
    fc = _compute_failure_clustering([_task(2), _task(8, failures=4)])
    p = fc.per_check_pass_rate
    collapsed = p ** fc.mean_checks_per_task
    assert fc.independent_rate > collapsed
    assert fc.independent_rate == (p**2 + p**8) / 2


def test_uniform_k_makes_the_two_baselines_coincide():
    # With no spread in k there is no Jensen gap -- a sanity bound on the above.
    fc = _compute_failure_clustering([_task(4), _task(4, failures=1)])
    p = fc.per_check_pass_rate
    assert abs(fc.independent_rate - p ** fc.mean_checks_per_task) < 1e-12


# ------------------------------------------------------------------ #
# excess / overdispersion behaviour                                    #
# ------------------------------------------------------------------ #


def test_independent_failures_give_zero_excess_and_unit_overdispersion():
    rng = random.Random(7)
    episodes = []
    for _ in range(800):
        k = rng.choice([2, 3, 4, 5, 7])
        episodes.append(_episode([_check(rng.random() < 0.72) for _ in range(k)]))
    fc = _compute_failure_clustering(episodes)
    assert abs(fc.excess) < 0.03
    assert 0.9 < fc.overdispersion < 1.1


def test_clustered_failures_give_large_positive_excess_and_overdispersion():
    # Same per-check pass rate, but every failure is concentrated into tasks
    # that fail wholesale -- the regime the metric exists to name.
    rng = random.Random(7)
    episodes = []
    for _ in range(800):
        k = rng.choice([2, 3, 4, 5, 7])
        doomed = rng.random() < 0.30
        episodes.append(_episode([_check(not doomed) for _ in range(k)]))
    fc = _compute_failure_clustering(episodes)
    assert fc.excess > 0.30
    assert fc.overdispersion > 2.0


def test_overdispersion_denominator_includes_the_k_heterogeneity_term():
    """Var_indep = mean(k) p (1-p) + (1-p)**2 Var(k).

    Dropping the second term shrinks the denominator and manufactures
    overdispersion out of nothing but variation in task size, so pin the exact
    value against a hand-computed expectation on a spread-out k histogram.
    """
    fc = _compute_failure_clustering([_task(2), _task(10, failures=3)])
    p = fc.per_check_pass_rate
    assert p == 9 / 12
    mean_k, var_k = fc.mean_checks_per_task, fc.var_checks_per_task
    assert mean_k == 6.0
    assert var_k == 16.0
    var_independent = mean_k * p * (1 - p) + (1 - p) ** 2 * var_k
    mean_f = 3 / 2
    var_observed = (0**2 + 3**2) / 2 - mean_f**2
    assert abs(fc.overdispersion - var_observed / var_independent) < 1e-12
    # And the k-free denominator would have given a materially different answer.
    assert abs(fc.overdispersion - var_observed / (mean_k * p * (1 - p))) > 0.1


# ------------------------------------------------------------------ #
# Degenerate inputs                                                    #
# ------------------------------------------------------------------ #


def test_empty_run_is_all_zeros_and_unavailable():
    fc = FailureClustering()
    assert fc.tasks == 0
    assert fc.gating_checks == 0
    assert fc.per_check_pass_rate == 0.0
    assert fc.observed_rate == 0.0
    assert fc.independent_rate == 0.0
    assert fc.excess == 0.0
    assert fc.overdispersion == 0.0
    assert fc.available is False


def test_flawless_run_has_no_baseline_variance_so_overdispersion_is_zero():
    # p == 1 makes Var_indep vanish; the ratio is undefined, not infinite.
    fc = _compute_failure_clustering([_task(3), _task(4)])
    assert fc.per_check_pass_rate == 1.0
    assert fc.observed_rate == 1.0
    assert fc.independent_rate == 1.0
    assert fc.excess == 0.0
    assert fc.overdispersion == 0.0


def test_single_task_is_not_available():
    fc = _compute_failure_clustering([_task(3, failures=1)])
    assert fc.tasks == 1
    assert fc.available is False


# ------------------------------------------------------------------ #
# print_report rendering                                               #
# ------------------------------------------------------------------ #


def _summary_with(episodes: list[EpisodeResult], **kw) -> EvalSummary:
    s = EvalSummary(
        agent_name="solver", split="test", total=0, passed=0, total_score=0.0, **kw
    )
    for ep in episodes:
        s.results.append(ep)
        s.total += 1
        s.passed += 1 if ep.success else 0
        s.total_score += 1.0 if ep.success else 0.3
    return s


def test_print_report_shows_failure_clustering_line(capsys):
    s = _summary_with([_task(3) for _ in range(6)] + [_task(3, failures=2)] * 4)
    s.print_report()
    out = capsys.readouterr().out
    assert "Failure clustering :" in out
    assert "per_check_pass=" in out
    assert "all_pass_if_independent=" in out
    assert "overdispersion=" in out
    assert "gating_checks=30 over 10 tasks" in out


def test_print_report_omits_line_when_no_gating_checks(capsys):
    s = _summary_with([_episode([_check(True, role="diagnostic")]) for _ in range(4)])
    s.print_report()
    assert "Failure clustering :" not in capsys.readouterr().out


def test_print_report_omits_line_for_a_single_task(capsys):
    s = _summary_with([_task(3, failures=1)])
    s.print_report()
    assert "Failure clustering :" not in capsys.readouterr().out


def test_failure_clustering_line_does_not_disturb_the_parsed_lines(capsys):
    """Regression guard for the two line-oriented regex readers.

    ``eval_report._generate_evaluation_visualizer`` and
    ``scripts/generate_eval_visualizer.parse_summary_file`` both scrape this
    report with anchored regexes. A new line is safe; editing the Tasks /
    Completed / Diagnostics lines is not, and fails silently.
    """
    s = _summary_with(
        [_task(3) for _ in range(6)] + [_task(3, failures=2)] * 4,
        exec_mode="react",
        llm_provider="vertex-anthropic",
        llm_model="claude-opus-5",
    )
    s.print_report()
    out = capsys.readouterr().out
    assert "Failure clustering :" in out

    assert re.search(
        r"Agent\s*:\s*([^\s]+)\s+Split\s*:\s*([^\s]+)\s+Mode\s*:\s*([^\s]+)", out
    )
    assert re.search(r"LLM\s*:\s*([^\n]+)", out).group(1).strip() == (
        "vertex-anthropic / claude-opus-5"
    )
    m_tasks = re.search(
        r"Tasks\s*:\s*(\d+)/(\d+)\s+passed\s+pass@budget=([^\s]+)"
        r"\s+raw_accuracy=([^\s]+)\s+avg_score=([^\s]+)",
        out,
    )
    assert m_tasks and m_tasks.group(1) == "6" and m_tasks.group(2) == "10"
    assert re.search(
        r"Completed\s*:\s*(\d+)/(\d+)\s+provider_failures=(\d+)"
        r"\s+provider_impacted_tasks=(\d+)\s+clean_completed_tasks=(\d+)"
        r"\s+provider_adjusted_accuracy=([^\s]+)",
        out,
    )
    assert re.search(
        r"Diagnostics\s*:\s*near_miss_rate=([^\s]+)\s+near_miss_tasks=(\d+)", out
    )
