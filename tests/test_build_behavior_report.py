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

"""Tests for the check-outcome accounting in build_behavior_report.

The report's ``semantic_match`` bucket used to count only ``passed``, which
pooled three outcomes that mean completely different things:

  * the judge read the content and said no  -> a real semantic failure
  * the judge could not run (``ok=False``)  -> an infrastructure failure
  * the judge was never called              -> the solver produced nothing

All three arrive as ``passed=False``. Pooling them ranks models on how often
they acted rather than on how well they reasoned: measured on r01, 39.4% of
3.5-flash's semantic checks never reached the judge versus 1.4% for terra.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_TOOL = (
    # No `.resolve()`: under a sandboxed test runner `__file__` can be a
    # symlink into a staging tree, and resolving it walks out of that tree.
    Path(__file__).parent.parent
    / "docs/analysis/analysis_v8/tools/build_behavior_report.py"
)


def _load_tool():
    spec = importlib.util.spec_from_file_location("_bbr_under_test", _TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bbr = _load_tool()


# ------------------------------------------------------------------ #
# _judge_reached                                                        #
# ------------------------------------------------------------------ #


def test_structured_flag_false_means_not_reached():
    assert bbr._judge_reached({"judge_invoked": False, "message": "x"}) is False


def test_structured_flag_true_beats_legacy_prose():
    check = {"judge_invoked": True, "message": "no file /r.md to evaluate"}
    assert bbr._judge_reached(check) is True


def test_legacy_record_uses_the_prose_markers():
    for marker in ("(empty transcript)", "no file /r.md", "empty criterion for /r"):
        assert bbr._judge_reached({"message": f"LLM semantic match: {marker}"}) is False


def test_legacy_record_with_a_plain_message_counts_as_reached():
    assert bbr._judge_reached({"message": "LLM semantic match: states it"}) is True


def test_missing_message_and_flag_counts_as_reached():
    assert bbr._judge_reached({}) is True


# ------------------------------------------------------------------ #
# Drift pin: this tool mirrors eval_report rather than importing it     #
# ------------------------------------------------------------------ #


def test_matches_the_canonical_implementation_in_eval_report():
    """The tool is stdlib-only by design, so the copy is pinned by test.

    Scoped to judge-authored records on purpose. ``_judge_was_invoked`` is only
    ever called from ``_compute_judge_health`` *after* that function has already
    filtered on ``_JUDGE_MSG_PREFIX``, so its return value on a non-prefixed
    record is unreachable in production and pinning against it would pin
    undefined behaviour. The two implementations are compared on non-prefixed
    records at the call-site level instead -- see
    ``test_call_site_agrees_with_eval_report_on_non_judge_records``.
    """
    from train_eval.eval_report import _judge_was_invoked

    records = [
        {"judge_invoked": False, "message": "anything"},
        {"judge_invoked": True, "message": "no file /r.md to evaluate"},
        {"judge_invoked": None, "message": "LLM semantic match: (empty transcript)"},
        {"message": "LLM semantic match: no file /r.md to evaluate"},
        {"message": "LLM semantic match: empty criterion for /r.md"},
        {"message": "LLM semantic match: the verdict is stated"},
    ]
    for rec in records:
        mirrored = bbr._judge_reached(rec)
        canonical = _judge_was_invoked(rec, str(rec.get("message", "")))
        assert mirrored == canonical, rec


def test_a_deterministic_legacy_check_is_not_read_as_judge_attrition():
    """The markers are ordinary English and must not be matched unscoped.

    ``FileChecker``'s deterministic ``exists`` branch emits "file ... exists",
    and an effect check's prose can contain "no file " verbatim. Neither ever
    involved a judge, so neither can be "the judge was never reached".
    """
    for message in (
        "no file /report.md was produced",
        "field_contains: found (empty criterion) in body",
        "assets_accessed: (empty transcript) marker in prose",
    ):
        assert bbr._judge_reached({"message": message}) is True


def test_the_prefix_guard_only_scopes_the_legacy_path():
    """An explicit ``judge_invoked=False`` is honoured whatever the prose says."""
    assert bbr._judge_reached(
        {"judge_invoked": False, "message": "no prefix here"}
    ) is False


# ------------------------------------------------------------------ #
# End-to-end bucket accounting                                          #
# ------------------------------------------------------------------ #


_LEGACY = object()  # sentinel: omit ``judge_invoked`` entirely (pre-field record)


def _check(
    name,
    passed,
    *,
    ok=True,
    judge_invoked=True,
    role="required",
    message="LLM semantic match: whatever",
):
    check = {
        "name": name,
        "passed": passed,
        "ok": ok,
        "role": role,
        "message": message,
    }
    if judge_invoked is not _LEGACY:
        check["judge_invoked"] = judge_invoked
    return check


def _record(checks, task_id="t1"):
    return {
        "task_id": task_id,
        "success": False,
        "score": 0.0,
        "stop_reason": "done",
        "trajectory": [],
        "metadata": {"check_results": checks},
    }


def _semantic(recs):
    return bbr.analyze(recs)["check_split"]["semantic_match"]


def test_judged_failure_is_not_counted_as_infra_or_skipped():
    split = _semantic([_record([_check("field_semantic_match m", False)])])
    assert split["total"] == 1
    assert split["passed"] == 0
    assert split["infra_failures"] == 0
    assert split["not_invoked"] == 0
    assert split["graded"] == 1
    assert split["graded_pass_pct"] == 0.0


def test_infra_failure_is_split_out_and_excluded_from_graded():
    split = _semantic([_record([_check("field_semantic_match m", False, ok=False)])])
    assert split["infra_failures"] == 1
    assert split["graded"] == 0
    assert split["graded_pass_pct"] is None


def test_never_invoked_is_split_out_and_excluded_from_graded():
    split = _semantic(
        [_record([_check("field_semantic_match m", False, judge_invoked=False)])]
    )
    assert split["not_invoked"] == 1
    assert split["graded"] == 0
    assert split["graded_pass_pct"] is None


def test_graded_pass_pct_ignores_the_model_simply_not_acting():
    """The bug in one assertion: a non-acting model must not look wrong."""
    checks = [
        _check("field_semantic_match a", True),
        _check("field_semantic_match b", False, judge_invoked=False),
        _check("field_semantic_match c", False, judge_invoked=False),
        _check("field_semantic_match d", False, judge_invoked=False),
    ]
    split = _semantic([_record(checks)])
    # Pooled rate says 25% -- as if the model reasoned badly three times.
    assert split["pass_pct"] == 25.0
    # It actually never produced anything to grade three times; of the one
    # check a judge did read, it passed.
    assert split["not_invoked"] == 3
    assert split["graded"] == 1
    assert split["graded_pass_pct"] == 100.0


def test_infra_takes_precedence_over_not_invoked():
    """Same precedence as eval_report, so the two reports cannot disagree."""
    check = _check("field_semantic_match m", False, ok=False, judge_invoked=False)
    split = _semantic([_record([check])])
    assert split["infra_failures"] == 1
    assert split["not_invoked"] == 0


def test_legacy_records_without_the_fields_still_tally():
    legacy = {
        "name": "field_semantic_match m",
        "passed": False,
        "role": "required",
        "message": (
            "LLM semantic match: no 'body' content in messages_sent to "
            "evaluate (empty transcript)"
        ),
    }
    split = _semantic([_record([legacy])])
    assert split["not_invoked"] == 1
    assert split["infra_failures"] == 0


def test_deterministic_buckets_report_zero_judge_attrition():
    checks = [
        _check("field_contains id", True),
        _check("field_contains id", False),
        _check("exists thing", True),
    ]
    split_all = bbr.analyze([_record(checks)])["check_split"]
    for name in ("field_contains", "other"):
        assert split_all[name]["infra_failures"] == 0
        assert split_all[name]["not_invoked"] == 0
        assert split_all[name]["graded"] == split_all[name]["total"]


# ------------------------------------------------------------------ #
# Call-site accounting: the helper is correct only where it is CALLED   #
# ------------------------------------------------------------------ #
#
# Every end-to-end case above routes through ``_check``, whose default message
# carries the judge prefix, so none of them exercise the legacy fallback on a
# deterministic record. That gap is exactly where the unscoped marker match hid:
# the helper-level tests passed while ``analyze`` misfiled real checks.


def test_a_passing_deterministic_check_is_not_demoted_by_marker_prose():
    """A graded ``field_contains`` whose prose happens to contain a marker.

    Regression: the markers were matched against every legacy record, so this
    check -- which no judge ever touched and which PASSED -- was booked as
    "the judge was never invoked", pulling it out of ``graded`` and halving the
    bucket's ``pass_pct``.
    """
    checks = [
        _check("field_contains a", True, judge_invoked=_LEGACY,
               message="field_contains 'x': True"),
        _check("field_contains b", True, judge_invoked=_LEGACY,
               message="no file /old.md in workspace, matched inline instead"),
    ]
    split = bbr.analyze([_record(checks)])["check_split"]["field_contains"]
    assert split["total"] == 2
    assert split["passed"] == 2
    assert split["not_invoked"] == 0
    assert split["graded"] == 2
    assert split["pass_pct"] == 100.0
    assert split["graded_pass_pct"] == 100.0


def test_a_legacy_judge_record_is_still_booked_as_not_invoked():
    """The fallback must keep working where it is actually meant to apply."""
    checks = [
        _check("field_semantic_match a", False, judge_invoked=_LEGACY,
               message="LLM semantic match: no file /r.md to evaluate"),
    ]
    split = _semantic([_record(checks)])
    assert split["not_invoked"] == 1
    assert split["graded"] == 0


def _health(checks):
    from types import SimpleNamespace

    from train_eval.eval_report import _compute_judge_health

    return _compute_judge_health(
        [SimpleNamespace(metadata={"check_results": checks})]
    )


def test_call_site_agrees_with_eval_report_on_judge_records():
    """Both reports must book the same judge-authored records identically."""
    checks = [
        _check("field_semantic_match a", True,
               message="LLM semantic match: the verdict is stated"),
        _check("field_semantic_match b", False, ok=False,
               message="LLM semantic match: backend error"),
        _check("field_semantic_match c", False, judge_invoked=False,
               message="LLM semantic match: no file /r.md to evaluate"),
        _check("field_semantic_match d", False, judge_invoked=_LEGACY,
               message="LLM semantic match: empty criterion for /r.md"),
    ]
    split = _semantic([_record(checks)])
    health = _health(checks)
    assert health.judge_checks == split["total"] == 4
    assert health.infra_failures == split["infra_failures"] == 1
    assert health.fail_closed_empty == split["not_invoked"] == 2


def test_call_site_agrees_with_eval_report_on_non_judge_records():
    """Non-judge records contribute zero judge attrition on BOTH sides.

    ``eval_report`` reaches this by skipping them outright (they never enter
    ``judge_checks``); the tool reaches it by answering ``_judge_reached`` True.
    Different mechanisms, same conclusion -- which is the property that matters:
    neither report may claim a judge went missing on a check that never had one.
    """
    checks = [
        _check("field_contains a", True, judge_invoked=_LEGACY,
               message="no file /old.md in workspace, matched inline instead"),
        _check("field_contains b", False, judge_invoked=_LEGACY,
               message="field_contains 'y': False (empty criterion)"),
    ]
    split = bbr.analyze([_record(checks)])["check_split"]["field_contains"]
    health = _health(checks)
    assert health.judge_checks == 0
    assert health.fail_closed_empty == 0
    assert split["not_invoked"] == 0
    assert split["graded"] == split["total"] == 2


def test_original_pass_pct_fields_are_unchanged():
    """stats.json consumers predating the split must keep reading the same numbers."""
    checks = [
        _check("field_semantic_match a", True),
        _check("field_semantic_match b", False),
        _check("field_semantic_match c", False, ok=False),
        _check("field_semantic_match d", False, judge_invoked=False),
    ]
    split = _semantic([_record(checks)])
    assert split["total"] == 4
    assert split["passed"] == 1
    assert split["pass_pct"] == 25.0


# ------------------------------------------------------------------ #
# failure_clustering: the independence-baseline histogram              #
# ------------------------------------------------------------------ #
#
# The clustering section used to render only the OBSERVED failure-count
# histogram, captioned "a clustered model is bimodal (mass at both ends)".
# That caption cannot be true of that chart: the observed histogram is
# located by the model's ABILITY (a strong model piles up at 0 lost checks,
# a weak one at 3-4), so on the real corpus all ten models come out
# unimodal and the promised two-humped shape appears in none of them.
#
# The shape is only visible as observed MINUS the model's own independence
# baseline, so the baseline is now computed per bucket and the difference is
# what gets drawn. These tests pin the identities that let a reader check
# the chart against the summary table, and the sign pattern itself.

import random as _random


def _vectors(n, seed, per_task_p=None, p=0.7, kmin=2, kmax=7):
    """n tasks of random size; `per_task_p` makes difficulty vary BY TASK."""
    rnd = _random.Random(seed)
    out = []
    for _ in range(n):
        k = rnd.randint(kmin, kmax)
        rate = per_task_p(rnd) if per_task_p else p
        out.append([rnd.random() < rate for _ in range(k)])
    return out


def test_bucket_zero_is_the_observed_pass_rate():
    """A task passes exactly when it loses zero required checks."""
    c = bbr.failure_clustering(_vectors(800, seed=1))
    assert c["fail_hist_pct"][0] == c["obs_pass_pct"]


def test_bucket_zero_of_the_baseline_is_the_independent_pass_rate():
    """mean_i(C(k,0) q^0 p^k) is mean_i(p^k), which is indep_pass_pct."""
    c = bbr.failure_clustering(_vectors(800, seed=2))
    assert c["indep_hist_pct"][0] == c["indep_pass_pct"]


def test_leftmost_delta_is_exactly_the_excess_column():
    """The anchor the caption tells the reader to rely on.

    Differencing the two ROUNDED histograms would be off by 0.1 often enough
    to matter, so the deltas are taken from the unrounded vectors. This is
    the test that fails if that is ever "simplified".
    """
    for seed in range(1, 12):
        c = bbr.failure_clustering(_vectors(700, seed=seed))
        assert c["delta_hist_pp"][0] == c["excess_pp"], seed


def test_both_histograms_are_distributions():
    """Both vectors must be exhaustive: every task lands in exactly one bucket.

    Tolerance is 0.25 because each of the five buckets is independently rounded
    to one decimal, so the sum can legitimately drift by 5 * 0.05.
    """
    for seed in (3, 11, 12, 13):
        c = bbr.failure_clustering(_vectors(900, seed=seed))
        assert abs(sum(c["fail_hist_pct"]) - 100.0) <= 0.25, c["fail_hist_pct"]
        assert abs(sum(c["indep_hist_pct"]) - 100.0) <= 0.25, c["indep_hist_pct"]


def test_independent_data_produces_a_flat_difference():
    """The null must actually be null: no shape when there is no clustering."""
    c = bbr.failure_clustering(_vectors(6000, seed=4, p=0.7))
    assert abs(c["excess_pp"]) < 1.5
    assert max(abs(d) for d in c["delta_hist_pp"]) < 1.5


def test_clustered_data_produces_the_two_humped_difference():
    """Mass out of the middle, into both ends -- the signature the prose claims."""
    c = bbr.failure_clustering(
        _vectors(6000, seed=5,
                 per_task_p=lambda r: r.choice([0.97, 0.95, 0.30, 0.25])))
    d = c["delta_hist_pp"]
    assert d[0] > 5, d          # more clean tasks than independence allows
    assert d[1] < -5, d         # drained out of "lost exactly one"
    assert d[2] < 0, d
    assert d[4] > 5, d          # and more total collapses
    assert c["lift"] > 1.2


def test_the_observed_histogram_alone_does_not_show_the_shape():
    """Why the difference is drawn instead of the raw histogram.

    Clustered data whose ability is high enough still yields a MONOTONE
    observed histogram -- no second hump -- even though the same data is
    strongly clustered against its own baseline. Reading bimodality off the
    raw bars is therefore not possible, which is the bug this replaced.
    """
    c = bbr.failure_clustering(
        _vectors(6000, seed=6,
                 per_task_p=lambda r: r.choice([0.99, 0.98, 0.97, 0.55])))
    obs = c["fail_hist_pct"]
    assert obs == sorted(obs, reverse=True), obs      # unimodal, no right hump
    assert c["delta_hist_pp"][0] > 3                  # yet clearly clustered
    assert c["delta_hist_pp"][4] > 0


def test_baseline_uses_the_real_k_distribution_not_the_mean():
    """Collapsing to mean(k) biases every bucket, as it does for indep_pass_pct."""
    vecs = [[True] * 2 for _ in range(50)] + [[True] * 8 for _ in range(50)]
    vecs[0][0] = False
    for i in range(25):
        vecs[50 + i][0] = False
    c = bbr.failure_clustering(vecs)
    p = c["per_check_pct"] / 100
    naive = 100 * p ** c["mean_k"]
    assert abs(c["indep_hist_pct"][0] - naive) > 0.5
    assert c["indep_hist_pct"][0] == c["indep_pass_pct"]


def test_no_negative_zero_in_the_difference_vector():
    for seed in range(20):
        for d in bbr.failure_clustering(_vectors(400, seed=seed))["delta_hist_pp"]:
            assert not (d == 0 and str(d).startswith("-")), d


# ------------------------------------------------------------------ #
# Steps vs. gold: censoring, the pass/fail split, and the shape chart  #
# ------------------------------------------------------------------ #
#
# The band table used to print, per cell, "median actual steps" pooled over every
# run and "overshoot" = that median divided by the BAND's gold median. Three
# things were wrong with it and each has a test below.
#
#   * The divisor is one constant per row, so overshoot was an exact rescale of
#     the steps column -- no information, two numbers.
#   * 23.5% of runs stop exactly at the 20-step cap. For a model past 50% capped
#     the printed median IS the cap, so the cell reduced to cap / gold_median: a
#     fact about the band. It also made such a model look like it improved on
#     long tasks (2.50x -> 1.54x) because only the denominator moved.
#   * Passes and failures were pooled, which puts the summary between the modes
#     for any model that either finishes on budget or runs out the clock.

import re as _re

import pytest as _pytest


# The fixtures below need a couple of model keys and nothing about which ones.
# They are spelled literally for readability, so pin them against the tool's own
# roster: a key that is not in MODELS otherwise fails as a KeyError several
# frames deep inside gold_summary, which reads as a logic bug rather than as the
# stale fixture it is.
_FIXTURE_MODEL_KEYS = ("opus5", "terra", "sonnet", "flash")


def test_fixture_model_keys_exist_in_the_tools_roster():
    roster = {key for _name, _slug, key in bbr.MODELS}
    assert set(_FIXTURE_MODEL_KEYS) <= roster, sorted(roster)


def _gold_stats(runs_by_key):
    """Minimal stats shape gold_summary() reads: key -> [(task, steps, ok)]."""
    return {k: {"by_task": {t: {"steps": s, "success": ok}
                            for t, s, ok in runs_by_key.get(k, [])}}
            for _n, _s, k in bbr.MODELS}


def _overall(out, key="opus5"):
    return next(r for r in out["by_model"][key] if r["band"] == bbr.OVERALL_BAND)


def _flat(html):
    return _re.sub(r"<[^>]+>", "", html)


def test_capped_runs_are_kept_out_of_the_effort_median():
    """Half the runs pinned at the cap must not drag the effort figure."""
    gold = {f"t{i}": 9 for i in range(10)}
    runs = ([(f"t{i}", 9, True) for i in range(5)]
            + [(f"t{i}", bbr.STEP_CAP, False) for i in range(5, 10)])
    row = _overall(bbr.gold_summary(_gold_stats({"opus5": runs}), gold))
    assert row["med_steps_pass"] == 9        # pooled over all runs it was 14.5
    assert row["n_pass_uncapped"] == 5
    assert row["capped_pct"] == 50.0


def test_a_pass_that_hit_the_cap_is_still_censored():
    """Reaching the cap censors the step count even when the task passed."""
    gold = {"a": 9, "b": 9}
    runs = [("a", 8, True), ("b", bbr.STEP_CAP, True)]
    row = _overall(bbr.gold_summary(_gold_stats({"opus5": runs}), gold))
    assert row["med_steps_pass"] == 8        # not (8 + 20) / 2
    assert row["n_pass_uncapped"] == 1
    assert row["pass_pct"] == 100.0          # the run still counts as a pass
    assert row["capped_pct"] == 50.0


def test_overshoot_divides_by_each_tasks_own_gold_not_the_band_median():
    """Two 12-step runs mean different things on a 6-step and a 12-step task."""
    gold = {"a": 6, "b": 12}
    runs = [("a", 12, True), ("b", 12, True)]
    row = _overall(bbr.gold_summary(_gold_stats({"opus5": runs}), gold))
    assert row["ratio_pass"] == 1.5          # median of 2.00 and 1.00
    # ratio-of-medians against the pooled gold median (9) would have said 1.33.
    assert row["ratio_pass"] != 1.33


def test_a_cell_resting_on_few_passes_is_flagged_thin():
    gold = {f"t{i}": 9 for i in range(60)}

    def runs(npass):
        return [(f"t{i}", 9, i < npass) for i in range(60)]

    out = bbr.gold_summary(_gold_stats({
        "opus5": runs(bbr.THIN_PASSES - 1),
        "terra": runs(bbr.THIN_PASSES + 1)}), gold)
    assert _overall(out, "opus5")["thin"] is True
    assert _overall(out, "terra")["thin"] is False


def test_a_model_that_always_hits_the_cap_reports_no_measured_effort():
    """The gemini-3.5-flash case: 83% capped, so the old median WAS the cap."""
    gold = {f"t{i}": 9 for i in range(20)}
    runs = [(f"t{i}", bbr.STEP_CAP, False) for i in range(20)]
    row = _overall(bbr.gold_summary(_gold_stats({"opus5": runs}), gold))
    assert row["med_steps_pass"] is None
    assert row["ratio_pass"] is None
    assert row["capped_pct"] == 100.0


def test_a_run_above_the_cap_is_refused():
    """A harness change must break the analysis loudly, not silently."""
    gold = {"a": 9}
    stats = _gold_stats({"opus5": [("a", bbr.STEP_CAP + 1, True)]})
    with _pytest.raises(AssertionError):
        bbr.gold_summary(stats, gold)


def test_the_overshoot_histogram_partitions_every_run():
    gold = {f"t{i}": 9 for i in range(40)}
    runs = [(f"t{i}", 5 + (i % 16), i % 3 == 0) for i in range(40)]
    out = bbr.gold_summary(_gold_stats({"opus5": runs}), gold)
    h = out["over_hist"]["opus5"]
    assert h["n"] == 40
    assert abs(sum(h["pct"]) - 100.0) < 0.2
    assert len(h["pct"]) == len(bbr.OVER_BUCKETS) + 1


def test_a_capped_run_is_not_binned_as_if_the_cap_were_its_length():
    """Binning a censored run at 20 asserts a measurement never taken."""
    gold = {"a": 6}
    out = bbr.gold_summary(_gold_stats({"opus5": [("a", bbr.STEP_CAP, False)]}),
                           gold)
    pct = out["over_hist"]["opus5"]["pct"]
    assert pct[-1] == 100.0                  # the terminal cap bucket
    assert sum(pct[:-1]) == 0.0              # nothing placed on the finite axis


def test_a_bimodal_model_is_visible_in_the_histogram_but_not_in_a_median():
    """The defect this section had: one median per cell, landing in the trough.

    gemini-3.1-pro either finishes on budget or runs out the clock. Its pooled
    median was 17 steps -- a range holding 7.5% of its runs. The synthetic model
    here is the same shape sharpened: the pooled median falls where the model has
    NO runs at all, while the difference histogram shows both modes.
    """
    gold = {f"t{i}": 9 for i in range(100)}
    runs = ([(f"t{i}", 9, True) for i in range(30)]                    # on budget
            + [(f"t{i}", 12 + (i % 6), True) for i in range(30, 50)]   # 12..17
            + [(f"t{i}", bbr.STEP_CAP, False) for i in range(50, 100)])
    steps = sorted(s for _t, s, _ok in runs)
    pooled = (steps[49] + steps[50]) / 2
    assert pooled == 18.5
    assert not any(s == pooled for s in steps)   # the summary sits in a hole

    out = bbr.gold_summary(_gold_stats({"opus5": runs}), gold)
    pct = out["over_hist"]["opus5"]["pct"]
    assert pct[0] == 30.0                        # at or under gold
    assert pct[-1] == 50.0                       # stopped by the cap
    assert pct[0] > pct[1] and pct[-1] > pct[-2]  # mass at both ends
    # The effort figure now sits ON a mode instead of between them: pooled over
    # every run the summary was 18.5 steps, a value the model never produces;
    # over its uncapped passes it is 9, which is the on-budget mode itself.
    assert _overall(out, "opus5")["med_steps_pass"] == 9
    assert _overall(out, "opus5")["ratio_pass"] == 1.0


# ------------------------- rendering ------------------------------- #


def _row(**kw):
    base = {"band": bbr.OVERALL_BAND, "n": 100, "pass_pct": 30.0,
            "capped_pct": 20.0, "n_pass_uncapped": 25, "med_steps_pass": 11,
            "ratio_pass": 1.14, "thin": False}
    base.update(kw)
    return base


def test_the_gold_cell_prints_every_number_as_visible_text():
    """No number may live only in a title= attribute: print and copy lose it."""
    html = bbr._gold_cell(_row())
    text = _flat(html)
    for want in ("30%", "11 st", "1.14", "20% capped"):
        assert want in text, (want, text)
    assert "title=" not in html


def test_a_thin_gold_cell_is_greyed_and_carries_its_count():
    html = bbr._gold_cell(_row(n_pass_uncapped=6, thin=True))
    assert "thin" in html
    assert "n6" in _flat(html)


def test_a_cell_with_no_uncapped_pass_says_so_instead_of_printing_a_dash():
    html = bbr._gold_cell(_row(med_steps_pass=None, ratio_pass=None,
                               n_pass_uncapped=0, thin=True, capped_pct=100.0))
    assert "no uncapped pass" in _flat(html)
    assert "100% capped" in _flat(html)


def _hist_summary(caps):
    """caps maps key -> capped %; the rest of the mass goes to bucket 0."""
    return {"over_hist": {k: {"n": 100, "pct": [100.0 - caps.get(k, 0.0)]
                              + [0.0] * (len(bbr.OVER_BUCKETS) - 1)
                              + [caps.get(k, 0.0)]}
                          for _n, _s, k in bbr.MODELS}}


def test_the_distribution_orders_models_by_how_often_they_run_out_the_clock():
    caps = {k: float(i * 7) for i, (_n, _s, k) in enumerate(bbr.MODELS)}
    html = bbr.render_gold_distribution(_hist_summary(caps))
    names = [n for n, _s, k in sorted(bbr.MODELS, key=lambda m: caps[m[2]])]
    seen = [n for n in names if n in html]
    assert [html.index(n) for n in seen] == sorted(html.index(n) for n in seen)


def test_the_distribution_gives_the_cap_its_own_terminal_column():
    html = bbr.render_gold_distribution(_hist_summary({"flash": 83.0}))
    assert "hit the cap" in html
    assert 'class="ohn cap"' in html          # styled apart from the finite bins
    assert "83.0" in _flat(html)
    assert "title=" not in html


def test_the_distribution_is_empty_without_data():
    assert bbr.render_gold_distribution({}) == ""


# ------------------------------------------------------------------ #
# clustering_exemplar: the named-task illustration                      #
# ------------------------------------------------------------------ #
#
# A cross-model "how concentrated are the misses" contrast does not work on this
# corpus: lost-checks-per-spoiled-task tracks ability almost exactly (3.5-flash
# 2.96 down to gpt-5.5 1.79), so it restates the per-check column -- the same
# trap the raw failure histogram fell into. The exemplar therefore holds ability
# fixed by construction and only compares scenarios where both models lost the
# SAME number of required checks. These tests pin both constraints.


def _ex_stats(spec, pc, lift):
    """spec: key -> {task_id: (req_k, req_f)}; pc/lift: key -> float."""
    out = {}
    for _n, _s, k in bbr.MODELS:
        tasks = spec.get(k)
        if tasks is None:
            out[k] = {"by_task": {}}
            continue
        out[k] = {
            "by_task": {t: {"req_k": kk, "req_f": ff, "success": ff == 0}
                        for t, (kk, ff) in tasks.items()},
            "clustering": {"per_check_pct": pc[k], "lift": lift[k]},
        }
    return out


def _family(prefix, vecs, k=4):
    return {f"{prefix}_r{i + 1:02d}": (k, f) for i, f in enumerate(vecs)}


def _two_model_spec(a_vec, b_vec, prefix="scen_v01", k=4):
    return {"opus5": _family(prefix, a_vec, k), "terra": _family(prefix, b_vec, k)}


def test_the_exemplar_pairs_models_of_near_identical_per_check_accuracy():
    """Ability must be held fixed or the contrast just restates ability."""
    spec = _two_model_spec([0, 0, 3, 1], [1, 1, 1, 1])
    spec["sonnet"] = _family("scen_v01", [0, 0, 3, 1])
    pc = {"opus5": 72.0, "terra": 71.0, "sonnet": 40.0}
    lift = {"opus5": 1.60, "terra": 1.25, "sonnet": 9.99}
    ex = bbr.clustering_exemplar(_ex_stats(spec, pc, lift))
    # sonnet has by far the biggest lift gap but is 32pp away on ability
    assert {ex["a"], ex["b"]} == {"opus5", "terra"}
    assert abs(ex["a_pc"] - ex["b_pc"]) <= bbr.EXEMPLAR_PC_TOL


def test_the_exemplar_only_uses_a_scenario_with_an_equal_error_budget():
    """Unequal totals would let ability leak back into the pass-count gap."""
    spec = {
        "opus5": {**_family("equal_v01", [0, 0, 4, 0]),
                  **_family("uneven_v01", [0, 0, 0, 0])},
        "terra": {**_family("equal_v01", [1, 1, 1, 1]),
                  **_family("uneven_v01", [1, 1, 1, 1])},
    }
    pc = {"opus5": 72.0, "terra": 71.0}
    lift = {"opus5": 1.60, "terra": 1.25}
    ex = bbr.clustering_exemplar(_ex_stats(spec, pc, lift))
    # "uneven" has a far bigger pass gap (4 vs 0) but the budgets differ 0 vs 4
    assert ex["scenario"] == "equal_v01"
    assert sum(r["a"] for r in ex["rounds"]) == sum(r["b"] for r in ex["rounds"])
    assert ex["a_passed"] == 3 and ex["b_passed"] == 0


def test_the_exemplar_puts_the_model_that_converts_its_errors_first():
    spec = _two_model_spec([1, 1, 1, 1], [0, 0, 4, 0])   # b is the clustered one
    pc = {"opus5": 72.0, "terra": 71.0}
    lift = {"opus5": 1.25, "terra": 1.60}
    ex = bbr.clustering_exemplar(_ex_stats(spec, pc, lift))
    assert ex["a"] == "terra"
    assert ex["a_passed"] > ex["b_passed"]


def test_the_exemplar_reports_the_round_where_extra_misses_were_free():
    """The mechanism: misses spent on an already-lost round change no outcome."""
    spec = _two_model_spec([0, 0, 4, 1], [1, 1, 2, 1])
    pc = {"opus5": 72.0, "terra": 71.0}
    lift = {"opus5": 1.60, "terra": 1.25}
    ex = bbr.clustering_exemplar(_ex_stats(spec, pc, lift))
    assert ex["free"]["round"] == "r03"          # 4 vs 2, and both lose it
    assert ex["free"]["a"] and ex["free"]["b"]


def test_the_exemplar_is_absent_rather_than_forced():
    spec = _two_model_spec([0, 1, 2, 3], [0, 1, 2, 3])   # identical: no gap
    pc = {"opus5": 72.0, "terra": 71.0}
    lift = {"opus5": 1.60, "terra": 1.25}
    assert bbr.clustering_exemplar(_ex_stats(spec, pc, lift)) is None
    assert bbr.render_clustering_exemplar(_ex_stats(spec, pc, lift)) == ""


def test_the_exemplar_prose_is_computed_from_the_chosen_scenario():
    """Numbers in the sentence must move with the data, not be hardcoded."""
    pc = {"opus5": 72.0, "terra": 71.0}
    lift = {"opus5": 1.60, "terra": 1.25}
    html = bbr.render_clustering_exemplar(
        _ex_stats(_two_model_spec([0, 0, 4, 0], [1, 1, 1, 1]), pc, lift))
    text = _flat(html)
    assert "3 rounds passed versus 0" in text
    assert "4 checks each" in text
    other = bbr.render_clustering_exemplar(
        _ex_stats(_two_model_spec([0, 0, 0, 3], [1, 1, 1, 0]), pc, lift))
    assert "3 rounds passed versus 1" in _flat(other)
    assert "3 checks each" in _flat(other)
    assert "title=" not in html
