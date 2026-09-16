#!/usr/bin/env python3

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

"""Emit the generated tables of the thirteen `<model>-solver-failures.md` deep-dives.

Every table in those documents is *generated* from `failure_taxonomy.json` and
`cross-model-behavioral-comparison.stats.json` and injected between HTML-comment
sentinels, so no per-model number is ever typed by hand. Prose lives outside the
sentinels and survives re-runs; `tools/audit_report.py` (pass E) re-parses the
emitted rows back out of the markdown and recomputes each cell.

On first run the tool scaffolds a document containing the tables plus
`<!-- PROSE: ... -->` placeholders. The audit treats a surviving placeholder as
a failure, so a half-written deep-dive cannot be mistaken for a finished one.

Usage:
    python tools/emit_solver_failures.py            # inject (in place)
    python tools/emit_solver_failures.py --check    # verify all 13 are current
    python tools/emit_solver_failures.py --model opus5 --stdout
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

TAXONOMY = os.path.join(ROOT, "failure_taxonomy.json")
STATS = os.path.join(ROOT, "cross-model-behavioral-comparison.stats.json")

# The scenario seeds, which are the only record of the benchmark's DESIGNED
# difficulty (section 4.3). Derived from ROOT like the two files above rather
# than exposed as a flag: unlike an eval-results tree, the seeds are always in
# this same source checkout at a fixed relative depth.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(ROOT)))
SEED_DIR = os.path.join(REPO, "clasharness", "data", "scenario_seeds",
                        "conflict_vlm_v1")

# key -> (display name, file slug, provider, served model id)
#
# `provider` and `model id` are the run_config values, verbatim, read from the
# episode headers (one distinct value across all 60 log files per model). They are
# NOT cosmetic:
#   * the three open-source models all report provider `openai` -- that is the
#     OpenAI-COMPATIBLE client adapter, not the vendor. Only the model id names the
#     weights, which is why both are rendered.
#   * `pro` is served as `gemini-3.1-pro-preview`, a preview build. The display name
#     drops the suffix; the provenance line must not.
#   * `glm` is served as `zai-org/GLM-5.2-FP8`, i.e. FP8-quantized, which no other
#     model in the table declares.
MODELS = {
    "flash": ("gemini-3.5-flash", "gemini-3-5-flash", "gemini",
              "gemini-3.5-flash"),
    "flash36": ("gemini-3.6-flash", "gemini-3-6-flash", "gemini",
                "gemini-3.6-flash"),
    "flash37": ("gemini-3.7-flash", "gemini-3-7-flash", "gemini",
                "gemini-3.7-flash"),
    "pro": ("gemini-3.1-pro", "gemini-3-1-pro", "gemini",
            "gemini-3.1-pro-preview"),
    "opus": ("claude-opus-4.8", "claude-opus-4-8", "vertex-anthropic",
             "claude-opus-4-8"),
    "opus5": ("claude-opus-5", "claude-opus-5", "vertex-anthropic",
              "claude-opus-5"),
    "sonnet": ("claude-sonnet-5", "claude-sonnet-5", "vertex-anthropic",
               "claude-sonnet-5"),
    "sol": ("gpt-5.6-sol", "gpt-5-6-sol", "bedrock-gpt", "openai.gpt-5.6-sol"),
    "luna": ("gpt-5.6-luna", "gpt-5-6-luna", "bedrock-gpt",
             "openai.gpt-5.6-luna"),
    "terra": ("gpt-5.6-terra", "gpt-5-6-terra", "bedrock-gpt",
              "openai.gpt-5.6-terra"),
    "dsv4": ("deepseek-v4-pro", "deepseek-v4-pro", "openai",
             "deepseek-ai/DeepSeek-V4-Pro"),
    "kimi": ("kimi-k3", "kimi-k3", "openai", "moonshotai/Kimi-K3"),
    "glm": ("glm-5.2", "glm-5-2", "openai", "zai-org/GLM-5.2-FP8"),
}
K = list(MODELS)
# Field size, used by every "rank of N" header. Never write the literal.
NK = len(K)

# Models that were run once, at high, instead of being swept over low/medium/high.
# They are absent from the `low` and `medium` cells of failure_taxonomy.json and
# from stats.json's `effort.cells` / `effort.mcnemar` -- see sf4_effort, which
# renders a note for them instead of an effort table rather than crashing.
NO_SWEEP = ("dsv4", "kimi", "glm")

BUCKETS = [("F1", "content_msg", "message content"),
           ("F2", "content_file", "file content"),
           ("F3", "grounding", "asset grounding"),
           ("F4", "anchor", "literal anchor"),
           ("F5", "side_effect", "terminal action"),
           ("F6", "window", "time window")]

EFFORTS = ("low", "medium", "high")
# high, not medium: it is the only effort at which all thirteen models exist in one
# coordinate system, because the NO_SWEEP three have no other setting. Must stay in
# step with failure_taxonomy.PRIMARY_EFFORT and build_behavior_report.PRIMARY_EFFORT.
PRIMARY = "high"

BLOCKS = ["SF1", "SF2", "SF3", "SF4", "SF5", "SF6", "SF7", "SF8", "SF9",
          "SF10", "SF11"]

# Hand-written prose slots. Each lives between its own sentinel pair so that
# (a) `audit_report.py` can prove a slot is still empty, and (b) adding a new
# generated block later can rebuild the scaffold without losing prose.
PROSE = ["signature", "termination", "buckets", "effort", "actions", "waits",
         "clustering", "stops", "scenarios", "design_axes", "interpretation"]

# --- designed difficulty axes (section 4.3) ----------------------------------
# The conflict-plugin registry, hand-copied from the `ConflictPluginSpec` of
# each module under clasharness/scenario/plugins/. The duplication is
# deliberate: SF11 is a fourth independent derivation of these buckets --
# design_axis_slice.py imports the live registry, build_behavior_report.py
# carries its own copy, audit pass D13 parses the plugin sources textually, and
# this one reads failure_taxonomy.json. Pass D14 asserts they agree on every
# integer, which is only evidence because they share no code path; pass D13
# pins this copy to the sources.
#
# `archetype` was the grouping here through v7 and is deliberately NOT an axis
# now. It is declared on 15 of the 20 seeds but the generator never resolves it,
# so it grouped tasks by a field that changed nothing about them -- see the
# footnote closing section 5.3 of the capability analysis.
PLUGIN_CATEGORY = {
    # inconsistency
    "evidence_mismatch": "inconsistency",
    "temporal_staleness": "inconsistency",
    # coordination
    "barrier_synchronization": "coordination",
    "credential_lock": "coordination",
    "deadlock_dependency": "coordination",
    "distributed_lock": "coordination",
    # order
    "controlled_deployment_gate": "order",
    "dependency_ordering": "order",
    # timeout
    "multi_device_pairing": "timeout",
    "reversible_action_hold": "timeout",
    "time_window_expiration": "timeout",
    # subagent_collaboration
    "coordinated_containment": "subagent_collaboration",
    "duplicate_hold_stale": "subagent_collaboration",
    "multi_party_synchronization": "subagent_collaboration",
    "non_overlapping_assignment": "subagent_collaboration",
    # main_supervised
    "scope_reconciliation": "main_supervised",
    # challenge -- the difficulty_challenges half of the token list
    "duplicate_avoidance": "challenge",
    "gap_completion": "challenge",
    "information_transfer": "challenge",
    "multi_source_reasoning": "challenge",
    "state_repair": "challenge",
    "vision_decoy": "challenge",
    "workflow_completion": "challenge",
}

CHALLENGE_TOKENS = frozenset(n for n, c in PLUGIN_CATEGORY.items()
                             if c == "challenge")

# A dropped or mistyped entry above would quietly shrink section 4.3 by one row
# in all thirteen documents, so the shape is asserted at import rather than left to
# the audit: this module is imported by write_prose.py and audit_report.py too,
# and it should not be possible to load a half-registry anywhere.
if len(PLUGIN_CATEGORY) != 23 or len(CHALLENGE_TOKENS) != 7:
    raise SystemExit(f"PLUGIN_CATEGORY has {len(PLUGIN_CATEGORY)} plugins and "
                     f"{len(CHALLENGE_TOKENS)} challenge tokens, expected 23/7")

# (bucket, axis, is_lever). is_lever=False is a CONTROL arm -- the complement of
# a knob, shown because it is what makes the knob's delta readable, but excluded
# from "N of M levers" counts because it carries the same fact with the sign
# flipped. Only the knobs partition the corpus; plugin rows overlap.
KNOB_ROWS = [("async_required", "async", True),
             ("no_async", "async", False),
             ("vision_graded", "vision", True),
             ("not_vision_graded", "vision", False)]


_SEEDS = {}


def seeds(seed_dir=SEED_DIR):
    """base seed slug -> seed dict, loaded once per directory.

    Lazy on purpose. `write_prose.py` and `audit_report.py` import this module
    for its sentinel helpers, and neither should be made to touch the seed tree
    -- or to die on a missing one -- merely by importing it.
    """
    if seed_dir not in _SEEDS:
        found = {}
        for f in sorted(glob.glob(os.path.join(seed_dir, "*.json"))):
            with open(f, encoding="utf-8") as fh:
                found[os.path.basename(f)[:-len(".json")]] = json.load(fh)
        if not found:
            raise SystemExit(f"no scenario seeds under {seed_dir}")
        _SEEDS[seed_dir] = found
    return _SEEDS[seed_dir]


def base_of(slug):
    """scenario slug -> base seed name (strip the trailing _v<NN> variant)."""
    return re.sub(r"_v\d+$", "", slug or "")


def seed_plugins(seed):
    """Every registered conflict plugin one seed routes to.

    The generator concatenates `conflict_taxonomy` and `difficulty_challenges`
    and resolves the whole list against the plugin registry
    (`conflict_llm_pipeline.py:1582`), so which of the two fields a token came
    from does not affect what the scenario contains, and a token that is not a
    plugin name is silently dropped. Deduplicated because a plugin named by both
    fields is still one plugin.
    """
    tokens = ((seed.get("conflict_taxonomy") or [])
              + (seed.get("difficulty_challenges") or []))
    return sorted({t for t in tokens if t in PLUGIN_CATEGORY})


def seed_buckets(seed):
    """One seed -> every (axis, bucket) pair it belongs to."""
    return ([("vision", "vision_graded" if seed.get("force_vision_locus")
              else "not_vision_graded"),
             ("async", "async_required" if seed.get("force_async_wait")
              else "no_async")]
            + [("plugin", p) for p in seed_plugins(seed)])


def seed_counts(sd):
    """bucket -> how many of the 20 seeds it covers."""
    n = Counter()
    for seed in sd.values():
        for _axis, bucket in seed_buckets(seed):
            n[bucket] += 1
    return n


def design_rows(sd):
    """(bucket, axis, is_lever) in table order: knob arms, then all 23 plugins.

    Plugins are ordered by seed coverage descending, then by name -- the same
    rule section 5.3 uses, recomputed here rather than copied so the two orders
    cannot drift apart while both claim to be sorted by coverage. Sorting by
    coverage also collects the single-seed rows, the ones whose delta cannot be
    attributed to the lever at all, together at the bottom.
    """
    n = seed_counts(sd)
    return KNOB_ROWS + [(p, "plugin", True) for p in
                        sorted(PLUGIN_CATEGORY, key=lambda p: (-n[p], p))]


# -------------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------------

def rank_of(key, values, best="min"):
    """1-based rank of `key` in `values` ({model: number}); ties share a rank."""
    ordered = sorted(values.values(), reverse=(best == "max"))
    return ordered.index(values[key]) + 1


def oxford(items):
    """'a', 'a and b', 'a, b, and c'."""
    if len(items) < 3:
        return " and ".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]


def req_rate(cell, bucket):
    a = cell["failed_checks"][bucket]["required"]
    b = cell["total_checks"][bucket]["required"]
    return 100.0 * a / b


def med(values):
    return statistics.median(values)


def signed(v, dp=3):
    """A signed number with a typographic minus.

    Scoped to the number itself on purpose: applying `.replace('-', '\u2212')`
    to a whole formatted row also rewrites any hyphen in a scenario slug or a
    model name that happens to share the row.
    """
    return f"{v:+.{dp}f}".replace("-", "\u2212")


def arrow(v, ref, lower_is_better=True):
    """A compact 'vs field median' marker."""
    if abs(v - ref) < 1e-9:
        return "="
    better = (v < ref) if lower_is_better else (v > ref)
    return "better" if better else "worse"


# -------------------------------------------------------------------------
# block builders -- each returns a markdown fragment
# -------------------------------------------------------------------------

def sf1_snapshot(k, FT, S):
    """Capability snapshot: this model's row plus its rank in the field."""
    M = FT["cells"][PRIMARY]["models"]
    pas = {m: M[m]["pass_pct"] for m in K}
    avg = {m: M[m]["avg_score"] for m in K}
    pc = {m: S[m]["clustering"]["per_check_pct"] for m in K}
    ft = {m: M[m]["failed_total"] for m in K}
    rows = [
        ("pass@budget", f"{M[k]['pass_pct']}% ({M[k]['passed']}/600)",
         rank_of(k, pas, "max"), f"{med(list(pas.values()))}%"),
        ("avg_score", f"{M[k]['avg_score']:.4f}",
         rank_of(k, avg, "max"), f"{med(list(avg.values())):.4f}"),
        ("per-check accuracy", f"{S[k]['clustering']['per_check_pct']}%",
         rank_of(k, pc, "max"), f"{med(list(pc.values())):.1f}%"),
        ("failed required+diagnostic units", f"{M[k]['failed_total']:,}",
         rank_of(k, ft, "min"), f"{med(list(ft.values())):,.0f}"),
    ]
    out = [f"| metric | value | rank of {NK} | field median |",
           "|---|---:|---:|---:|"]
    for name, val, rk, fm in rows:
        out.append(f"| {name} | **{val}** | {rk} | {fm} |")
    return "\n".join(out)


def sf2_termination(k, FT, S):
    M = FT["cells"][PRIMARY]["models"]
    fields = [("near_miss", "near-miss tasks", "max"),
              ("zero_score", "zero-score tasks", "min"),
              ("capped", "step-capped tasks", "min"),
              ("loopy_capped", "of those, single-verb loops", "min"),
              ("zero_open_tasks", "tasks with zero asset opens", "min"),
              ("zero_write_tasks", "tasks with zero writes", "min"),
              ("step_p50", "median steps", "min"),
              ("step_mean", "mean steps", "min")]
    out = [f"| signal | value | rank of {NK} | field median |",
           "|---|---:|---:|---:|"]
    for f, label, best in fields:
        vals = {m: M[m][f] for m in K}
        v = M[k][f]
        fm = med(list(vals.values()))
        fm_s = f"{fm:.2f}" if isinstance(v, float) else f"{fm:g}"
        v_s = f"{v:.2f}" if isinstance(v, float) else f"{v}"
        out.append(f"| {label} | **{v_s}** | {rank_of(k, vals, best)} | {fm_s} |")
    sem = M[k]["semantic"]
    out.append(f"| fail-closed semantic checks | **{sem['fail_closed']}** of "
               f"{sem['n']:,} ({sem['fail_closed_pct']}%) | "
               f"{rank_of(k, {m: M[m]['semantic']['fail_closed_pct'] for m in K})}"
               f" | {med([M[m]['semantic']['fail_closed_pct'] for m in K]):.1f}% |")
    return "\n".join(out)


def sf3_buckets(k, FT, S):
    M = FT["cells"][PRIMARY]["models"]
    out = ["| bucket | what it asserts | failed (req/diag) | required units | "
           f"failure rate | rank of {NK} | field median rate |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for fid, b, what in BUCKETS:
        fc = M[k]["failed_checks"][b]
        tot = M[k]["total_checks"][b]["required"]
        rates = {m: req_rate(M[m], b) for m in K}
        out.append(f"| **{fid}** {b} | {what} | {fc['required']}/{fc['diagnostic']} "
                   f"| {tot:,} | **{rates[k]:.1f}%** | {rank_of(k, rates)} | "
                   f"{med(list(rates.values())):.1f}% |")
    tot_req = sum(M[k]["failed_checks"][b]["required"] for _, b, _ in BUCKETS)
    tot_diag = sum(M[k]["failed_checks"][b]["diagnostic"] for _, b, _ in BUCKETS)
    out.append(f"| **total** | | **{tot_req:,}/{tot_diag}** | "
               f"**{sum(M[k]['total_checks'][b]['required'] for _, b, _ in BUCKETS):,}**"
               f" | | | |")
    return "\n".join(out)


def sf4_effort(k, FT, S):
    cells = S["effort"]["cells"]
    # A NO_SWEEP model has exactly one cell, so there is no effort curve to draw
    # and no low->high McNemar to run. Say that, rather than rendering a one-row
    # table that looks like a curve or a 0.0 pp delta that looks like a result.
    if k in NO_SWEEP:
        m = FT["cells"][PRIMARY]["models"][k]
        buck = ", ".join(f"{fid} {m['failed_checks'][b]['required']}"
                         for fid, b, _ in BUCKETS)
        note = [
            f"This model was **not swept**. It was run once, at "
            f"`reasoning_effort=high`, so there is no low/medium/high curve and "
            f"no low\u2192high McNemar test for it; every other figure in this "
            f"document is that single run. Its required-check failures at that "
            f"setting are {buck}.",
            "",
            "The ten swept models each have a three-point curve in the "
            "corresponding section of their own deep-dive, and section 6 of "
            "[cross-model-behavioral-comparison.md]"
            "(./cross-model-behavioral-comparison.md) compares all thirteen at "
            "high, which is the one setting they share.",
        ]
        miss = m["missing_check_results"]
        if miss:
            note += ["", f"Tasks whose episode produced no check results at all, "
                         f"counted as failures in the denominator without being "
                         f"scored: {miss}. See section 4.1."]
        return "\n".join(note)
    out = ["| effort | pass@budget | avg_score | capped | mean steps | "
           + " | ".join(f"{fid} failed" for fid, _, _ in BUCKETS) + " |",
           "|---|---:|---:|---:|---:|" + "---:|" * len(BUCKETS)]
    for eff in EFFORTS:
        c = cells[f"{k}@{eff}"]
        m = FT["cells"][eff]["models"][k]
        buck = " | ".join(str(m["failed_checks"][b]["required"])
                          for _, b, _ in BUCKETS)
        out.append(f"| {eff} | {c['pass_pct']}% ({c['passed']}/600) | "
                   f"{c['avg_score']:.4f} | {c['capped']} | {c['mean_steps']} | "
                   f"{buck} |")
    mc = S["effort"]["mcnemar"][k]
    out.append("")
    out.append(f"low \u2192 high McNemar: **{signed(mc['delta_pp'], 1)} pp** "
               f"({mc['gained']} gained, {mc['lost']} lost, "
               f"p = {mc['p']:.2g}).")
    # A task whose episode yielded no check results is graded fail-closed: it
    # is in the 600-task denominator but scored nothing. That is the honest
    # treatment, but it is invisible in the table above, so say it out loud.
    miss = {eff: FT["cells"][eff]["models"][k]["missing_check_results"]
            for eff in EFFORTS}
    if any(miss.values()):
        where = ", ".join(f"{n} at {eff}" for eff, n in miss.items() if n)
        out.append("")
        out.append(f"Tasks whose episode produced no check results at all, "
                   f"counted as failures in the denominator above without "
                   f"being scored: {where}. See section 4.1.")
    return "\n".join(out)


def sf5_actions(k, FT, S):
    mix = S[k]["catmix"]
    total = S[k]["catmix_total"]
    n = FT["cells"][PRIMARY]["models"][k]["rows"]
    field = {c: med([S[m]["catmix"].get(c, 0) / FT["cells"][PRIMARY]["models"][m]["rows"]
                     for m in K]) for c in mix}
    out = ["| action category | count | per task | field median per task | share |",
           "|---|---:|---:|---:|---:|"]
    for c, v in sorted(mix.items(), key=lambda kv: -kv[1]):
        out.append(f"| `{c}` | {v:,} | {v / n:.2f} | {field[c]:.2f} | "
                   f"{100.0 * v / total:.1f}% |")
    out.append(f"| **total** | **{total:,}** | **{total / n:.2f}** | | |")
    return "\n".join(out)


def sf6_waits(k, FT, S):
    M = FT["cells"][PRIMARY]["models"]
    w = M[k]["waits"]
    vals = {m: M[m]["waits"] for m in K}
    # This is the one table whose rank column changes direction between rows:
    # volume is better low, per-wait duration is better high. Derive the note
    # from the same spec that drives the rows so the two cannot disagree.
    spec = (("n", "wait actions", "min"),
            ("sub_30", "waits shorter than 30 s", "min"),
            ("mean", "mean wait (s)", "max"),
            ("median", "median wait (s)", "max"),
            ("max", "longest wait (s)", "max"),
            ("total", "total simulated wait (s)", "min"))
    out = [f"| wait statistic | value | rank of {NK} | field median |",
           "|---|---:|---:|---:|"]
    for f, label, best in spec:
        vv = {m: vals[m][f] for m in K}
        out.append(f"| {label} | **{w[f]:g}** | {rank_of(k, vv, best)} | "
                   f"{med(list(vv.values())):g} |")
    lo = [lb for _, lb, b in spec if b == "min"]
    hi = [lb for _, lb, b in spec if b == "max"]
    out.append("")
    out.append(f"Rank 1 is the lowest value for {oxford(lo)}, and the highest "
               f"value for {oxford(hi)}.")
    hist = ", ".join(f"`{name}` x{cnt}"
                     for name, cnt in sorted(w["hist"].items(),
                                             key=lambda kv: -kv[1]))
    shown = sum(w["hist"].values())
    out.append("")
    out.append(f"Most-used durations (top {len(w['hist'])} of the histogram, "
               f"{shown} of {w['n']} waits): {hist or '*(none)*'}.")
    return "\n".join(out)


def sf7_clustering(k, FT, S):
    c = S[k]["clustering"]
    vals = {m: S[m]["clustering"] for m in K}
    out = [f"| clustering statistic | value | rank of {NK} | field median |",
           "|---|---:|---:|---:|"]
    for f, label, best, unit in (
            ("per_check_pct", "per-check pass rate", "max", "%"),
            ("obs_pass_pct", "observed task pass rate", "max", "%"),
            ("indep_pass_pct", "pass rate if checks were independent",
             "max", "%"),
            ("excess_pp", "excess (observed \u2212 independent)", "max", " pp"),
            ("lift", "lift", "max", "\u00d7"),
            ("overdispersion", "overdispersion", "max", "\u00d7")):
        vv = {m: vals[m][f] for m in K}
        out.append(f"| {label} | **{c[f]}{unit}** | {rank_of(k, vv, best)} | "
                   f"{med(list(vv.values())):g}{unit} |")
    lo, hi = c["lift_ci"]
    out.append(f"| lift 95% CI | **[{lo}, {hi}]** | | |")
    out.append(f"| checks / tasks scored | **{c['n_checks']:,} / {c['n_tasks']}** "
               f"| | |")
    return "\n".join(out)


def sf8_stops(k, FT, S):
    M = FT["cells"][PRIMARY]["models"]
    stops = M[k]["stops"]
    n = M[k]["rows"]
    allnames = sorted({s for m in K for s in M[m]["stops"]})
    out = ["| stop reason | tasks | share | field median |", "|---|---:|---:|---:|"]
    for s in sorted(allnames, key=lambda s: -stops.get(s, 0)):
        fm = med([M[m]["stops"].get(s, 0) for m in K])
        out.append(f"| `{s}` | {stops.get(s, 0)} | "
                   f"{100.0 * stops.get(s, 0) / n:.1f}% | {fm:g} |")
    g = M[k]["gating"]
    out.append("")
    out.append(f"Of {g['n_fail']} failed tasks, **{g['content_involved']}** "
               f"involve a content check (F1/F2) and **{g['grounding_only']}** "
               f"fail on grounding alone; "
               f"{g['no_required_check_failed']} fail with no required check "
               f"failing at all.")
    # Without this, the `none` row reads as inconsistent with the pass count in
    # section 1, and the "no required check failed" residual above looks
    # impossible. Both are the same unscored tasks, so name them.
    miss = M[k]["missing_check_results"]
    if miss:
        ids = ", ".join(f"`{t}`" for t in M[k]["missing_check_result_ids"])
        out.append("")
        out.append(f"That residual is {miss} task{'s' if miss != 1 else ''} "
                   f"whose episode produced no check results at all ({ids}). "
                   f"Nothing failed, so the stop reason is `none`; nothing was "
                   f"scored either, so such a task neither passes nor enters "
                   f"the {S[k]['clustering']['n_tasks']} tasks scored in "
                   f"section 3.3. That accounts for the entire difference "
                   f"between the `none` row above and the pass count in "
                   f"section 1.")
    return "\n".join(out)


def sf9_scenarios(k, FT, S):
    M = FT["cells"][PRIMARY]["models"]
    slugs = sorted(M[k]["scenarios"])
    field = {s: sum(M[m]["scenarios"][s]["mean_score"] for m in K) / len(K)
             for s in slugs}
    gap = {s: M[k]["scenarios"][s]["mean_score"] - field[s] for s in slugs}
    zero = sorted((s for s in slugs if M[k]["scenarios"][s]["passed"] == 0),
                  key=lambda s: field[s])

    def tbl(items, title):
        o = [f"**{title}**", "",
             "| scenario | this model | field mean | gap | passes |",
             "|---|---:|---:|---:|---:|"]
        for s in items:
            o.append(f"| `{s}` | {M[k]['scenarios'][s]['mean_score']:.3f} | "
                     f"{field[s]:.3f} | {signed(gap[s])} | "
                     f"{M[k]['scenarios'][s]['passed']}/10 |")
        return "\n".join(o)

    best = sorted(slugs, key=lambda s: -gap[s])[:5]
    worst = sorted(slugs, key=lambda s: gap[s])[:5]
    out = [tbl(best, "Largest positive gaps against the field"), "",
           tbl(worst, "Largest negative gaps against the field"), "",
           f"**Zero-pass scenarios: {len(zero)} of 60.** "
           + (", ".join(f"`{s}`" for s in zero) if zero else "*(none)*") + "."]
    return "\n".join(out)


def pick_example(items):
    """Deterministic example choice: first `required` failure in scan order.

    Not cherry-picked -- `failure_taxonomy.py` collects examples in task order,
    and this takes the earliest one that actually gates a task. Falls back to
    the first example of any role if a bucket somehow has no required failure.
    """
    for e in items:
        if e["role"] == "required":
            return e
    return items[0] if items else None


def sf10_evidence(k, FT, S):
    """One verbatim failure message per bucket, quoted from the check records.

    Emitted rather than hand-copied: a paraphrased evaluator message is a
    fabricated quote, and the exact wording (especially the fail-closed
    'empty transcript' form) is the evidence for the mechanism claims.
    """
    ex = FT["cells"][PRIMARY]["models"][k]["examples"]
    out = ["Earliest *required* failure of each kind, quoted verbatim from the "
           "evaluator record (scan order, not selected for effect):", ""]
    for fid, b, _what in BUCKETS:
        e = pick_example(ex.get(b, []))
        if e is None:
            out.append(f"- **{fid}** `{b}` — *(no failure of this kind)*")
            continue
        # Messages are single-line today; collapse defensively so a future
        # multi-line message cannot silently break the blockquote.
        msg = " ".join(e["message"].split()).replace("|", "\\|")
        out.append(f"- **{fid}** `{b}` — `{e['task_id']}` ({e['role']}):")
        out.append(f"  > {msg}")
    return "\n".join(out)


def _axis_tally(scen, sd):
    """One model's per-scenario record -> {(axis, bucket): [n, passed, score]}.

    Also returns the same triple for the whole corpus and the set of scenario
    slugs that matched no seed.
    """
    agg = defaultdict(lambda: [0, 0, 0.0])
    base = [0, 0, 0.0]
    unmapped = set()
    for slug, sc in scen.items():
        seed = sd.get(base_of(slug))
        if seed is None:
            unmapped.add(base_of(slug))
            continue
        n, p, tot = sc["n"], sc["passed"], sc["mean_score"] * sc["n"]
        base[0] += n
        base[1] += p
        base[2] += tot
        for pair in seed_buckets(seed):
            agg[pair][0] += n
            agg[pair][1] += p
            agg[pair][2] += tot
    return agg, base, unmapped


def sf11_design_axes(k, FT, S):
    """Pass rate per DESIGNED difficulty lever, for this model.

    Derived from `FT[...]["scenarios"]` and the scenario seeds, and from
    NOTHING ELSE -- `S` is deliberately unused. This is the fourth of four
    independent derivations of the same buckets; the others are
    tools/design_axis_slice.py (raw jsonl), the `design_axes` block of
    cross-model-behavioral-comparison.stats.json (the report builder's own task
    table) and audit pass D13 (a textual parse of the plugin sources). Pass D14
    asserts they agree on every integer, and that is only evidence because they
    share no source.

    Rank is taken on the integer pass COUNT, never on a rate or a mean: within a
    bucket every model has the same denominator, so the two orderings are
    identical, and the integer one cannot produce a spurious tie-break at the
    fourth decimal.
    """
    del S  # see docstring
    sd = seeds()
    nseeds = seed_counts(sd)
    tal = {m: _axis_tally(FT["cells"][PRIMARY]["models"][m]["scenarios"], sd)
           for m in K}
    agg, base, unmapped = tal[k]
    base_pct = 100.0 * base[1] / base[0] if base[0] else 0.0

    def pct(triple):
        return 100.0 * triple[1] / triple[0] if triple[0] else 0.0

    out = ["| designed lever | category | seeds | tasks | passed | pass@budget "
           f"| \u0394 pp vs own baseline | avg_score | rank of {NK} by passes | "
           "field median pass |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    base_passes = {m: tal[m][1][1] for m in K}
    base_pcts = [pct(tal[m][1]) for m in K]
    out.append(f"| *all {base[0]} tasks (baseline)* | \u2014 | {len(sd)} | "
               f"{base[0]} | {base[1]} | **{base_pct:.1f}%** | \u2014 | "
               f"{base[2] / base[0]:.4f} | "
               f"{rank_of(k, base_passes, 'max')} | {med(base_pcts):.1f}% |")

    rows = design_rows(sd)
    below = []
    for bucket, axis, is_lever in rows:
        cell = agg.get((axis, bucket))
        if not cell:
            continue
        passes = {m: tal[m][0][(axis, bucket)][1] for m in K}
        pcts = [pct(tal[m][0][(axis, bucket)]) for m in K]
        d = pct(cell) - base_pct
        if is_lever and d < 0:
            below.append((d, bucket, cell[0], axis))
        out.append(f"| `{bucket}` | {PLUGIN_CATEGORY.get(bucket, 'knob')} | "
                   f"{nseeds[bucket]} | {cell[0]} | {cell[1]} | "
                   f"**{pct(cell):.1f}%** | {signed(d, 1)} | "
                   f"{cell[2] / cell[0]:.4f} | "
                   f"{rank_of(k, passes, 'max')} | {med(pcts):.1f}% |")

    n_levers = sum(1 for _b, _a, lv in rows if lv)
    out.append("")
    if below:
        # Ties break on the bucket name, so the sentence is a function of the
        # data alone and does not inherit the table's ordering.
        below.sort(key=lambda r: (r[0], r[1]))
        d, bucket, n, axis = below[0]
        passes = {m: tal[m][0][(axis, bucket)][1] for m in K}
        # 25 levers is too many to list, and the tail of the list is its least
        # informative part: naming the sharpest few and counting the rest says
        # more than an oxford comma with sixteen items in it. The cut is
        # extended through any tie it lands inside -- most of these buckets are
        # 30 tasks, so equal deltas are common, and showing two members of a
        # four-way tie would invent a distinction the data does not make.
        top = below[:5]
        while len(top) < len(below) and (signed(below[len(top)][0], 1)
                                         == signed(top[-1][0], 1)):
            top.append(below[len(top)])
        lead = ("The only one is " if len(below) == 1 else "The sharpest are ")
        thin = [b for _d, b, _n, _a in below if nseeds[b] <= 3]
        out.append(
            f"**{len(below)} of {n_levers} designed levers** sit below this "
            f"model's own {base_pct:.1f}% baseline, and **{len(thin)} of those "
            f"{len(below)} are carried by 3 seeds or fewer**. " + lead
            + oxford([f"`{b}` at {signed(_d, 1)} pp"
                      for _d, b, _n, _a in top])
            + f". `{bucket}` is measured on {n} tasks from "
            f"{nseeds[bucket]} seed{'' if nseeds[bucket] == 1 else 's'}, where "
            f"this model ranks {rank_of(k, passes, 'max')} of {NK} by passes.")
    else:
        out.append(f"**No designed lever** sits below this model's own "
                   f"{base_pct:.1f}% baseline.")
    if unmapped:
        out.append("")
        out.append(f"Scenarios matching no seed and therefore excluded from "
                   f"every row above: {oxford(sorted('`%s`' % u for u in unmapped))}.")
    return "\n".join(out)


BUILDERS = {"SF1": sf1_snapshot, "SF2": sf2_termination, "SF3": sf3_buckets,
            "SF4": sf4_effort, "SF5": sf5_actions, "SF6": sf6_waits,
            "SF7": sf7_clustering, "SF8": sf8_stops, "SF9": sf9_scenarios,
            "SF10": sf10_evidence, "SF11": sf11_design_axes}


# -------------------------------------------------------------------------
# document scaffold
# -------------------------------------------------------------------------

def sentinel(tag, body):
    return f"<!-- {tag}-BEGIN -->\n\n{body}\n\n<!-- {tag}-END -->"


def scaffold(k):
    disp, slug, provider, model_id = MODELS[k]
    sweep_note = ""
    if k in NO_SWEEP:
        sweep_note = (" That is the only setting this model was run at — it was "
                      "not swept, so it has no effort curve and section 4 says so "
                      "instead of tabulating one.")
    served = f"`{provider}` as `{model_id}`"
    if provider == "openai" and "/" in model_id:
        served = (f"`{provider}` — an OpenAI-*compatible* endpoint, not OpenAI the "
                  f"vendor — as `{model_id}`")

    def p(slot):
        return sentinel(f"PROSE-{slot}-{k}", "")

    return f"""# {disp} — solver failure deep-dive (v8 corpus)

> [!NOTE]
> **Scope.** One model, one corpus. Every figure below is
> `reasoning_effort=high` on the v8 conflict corpus (600 tasks = 60 scenarios
> × 10 rounds) unless a row says otherwise.{sweep_note} Cross-model comparisons live in
> [cross-model-behavioral-comparison.md](./cross-model-behavioral-comparison.md);
> the measurement-validity argument, the F1–F6 definitions and the corpus-wide
> difficulty analysis live in
> [eval-results-capability-analysis.md](./eval-results-capability-analysis.md).
> This document does not restate them.

**Provenance.** Served through {served}. Every table here is generated by
`tools/emit_solver_failures.py` from `failure_taxonomy.json` and
`cross-model-behavioral-comparison.stats.json` — plus, for section 4.3 only, the
scenario seeds under `clasharness/data/scenario_seeds/conflict_vlm_v1/`, which
supply the bucket each task belongs to but none of its numbers. Every table is
re-verified cell by cell by `tools/audit_report.py` (pass E). No number in this
file was typed by hand.

## 1. Capability snapshot and behavioral signature

{sentinel(f'SF1-{k}', '')}

{p('signature')}

### Termination profile

{sentinel(f'SF2-{k}', '')}

{p('termination')}

## 2. Failure-mode taxonomy

{sentinel(f'SF3-{k}', '')}

### One failure of each kind, verbatim

{sentinel(f'SF10-{k}', '')}

{p('buckets')}

### Stability across reasoning effort

{sentinel(f'SF4-{k}', '')}

{p('effort')}

## 3. Behavioral signature in the action stream

### 3.1 Action mix

{sentinel(f'SF5-{k}', '')}

{p('actions')}

### 3.2 Waiting

{sentinel(f'SF6-{k}', '')}

{p('waits')}

### 3.3 Failure clustering

{sentinel(f'SF7-{k}', '')}

{p('clustering')}

## 4. Difficulty structure for this model

### 4.1 How runs end

{sentinel(f'SF8-{k}', '')}

{p('stops')}

### 4.2 Scenario extremes

{sentinel(f'SF9-{k}', '')}

{p('scenarios')}

### 4.3 Designed difficulty levers

Sections 4.1 and 4.2 measure difficulty as it turned out. This one measures it
as it was *designed*: each task is joined to the scenario seed it was generated
from, and grouped by what that seed declares. The two boolean knobs give
non-overlapping arms; the conflict plugins do not. The generator resolves each
seed's `conflict_taxonomy` and `difficulty_challenges` tokens against the plugin
registry and one seed routes to several plugins at once, so those rows overlap
and do not sum to the corpus.

Read the `seeds` column first. Some plugins are carried by a single seed, and a
delta on one of those describes that one scenario rather than the lever, however
large it looks. The corpus-wide version of this table, with all thirteen models side
by side and the levers ranked, is section 5.3 of the capability analysis.

{sentinel(f'SF11-{k}', '')}

{p('design_axes')}

## 5. Interpretation

{p('interpretation')}

### Cross-references

- [cross-model-behavioral-comparison.md](./cross-model-behavioral-comparison.md)
- [eval-results-capability-analysis.md](./eval-results-capability-analysis.md)
"""


def splice(text, tag, body):
    b, e = f"<!-- {tag}-BEGIN -->", f"<!-- {tag}-END -->"
    if b not in text or e not in text:
        raise SystemExit(f"sentinels for {tag} not found")
    head, rest = text.split(b, 1)
    _stale, tail = rest.split(e, 1)
    return f"{head}{b}\n\n{body}\n\n{e}{tail}"


def extract(text, tag):
    """Body between a sentinel pair, or None if the pair is absent."""
    b, e = f"<!-- {tag}-BEGIN -->", f"<!-- {tag}-END -->"
    if b not in text or e not in text:
        return None
    return text.split(b, 1)[1].split(e, 1)[0].strip()


def missing_sentinels(text, k):
    want = [f"{t}-{k}" for t in BLOCKS] + [f"PROSE-{s}-{k}" for s in PROSE]
    return [t for t in want
            if f"<!-- {t}-BEGIN -->" not in text or f"<!-- {t}-END -->" not in text]


def render(k, FT, S, existing=None, rescaffold=False):
    """Inject the generated blocks, leaving hand-written prose untouched.

    When the scaffold gains a block the existing files do not have, splicing in
    place is impossible; `rescaffold` rebuilds the document from the current
    scaffold and carries every prose body forward by tag, so restructuring
    never silently drops prose.
    """
    if existing is None:
        text = scaffold(k)
    elif rescaffold:
        text = scaffold(k)
        for slot in PROSE:
            body = extract(existing, f"PROSE-{slot}-{k}")
            if body:
                text = splice(text, f"PROSE-{slot}-{k}", body)
    else:
        text = existing
        gone = missing_sentinels(text, k)
        if gone:
            raise SystemExit(
                f"{MODELS[k][1]}-solver-failures.md is missing "
                f"{len(gone)} sentinel pair(s): {', '.join(gone[:4])}"
                f"{' ...' if len(gone) > 4 else ''}\n"
                f"       The scaffold changed. Re-run with --rescaffold to "
                f"rebuild it (prose is carried forward by tag).")
    for tag in BLOCKS:
        text = splice(text, f"{tag}-{k}", BUILDERS[tag](k, FT, S))
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--stdout", action="store_true")
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--rescaffold", action="store_true",
                    help="rebuild from the current scaffold, carrying prose "
                         "forward by tag (use after adding a block)")
    args = ap.parse_args()

    with open(TAXONOMY) as fh:
        FT = json.load(fh)
    with open(STATS) as fh:
        S = json.load(fh)

    keys = [args.model] if args.model else K
    stale, wrote = [], []
    for k in keys:
        path = os.path.join(ROOT, f"{MODELS[k][1]}-solver-failures.md")
        old = None
        if os.path.exists(path):
            with open(path) as fh:
                old = fh.read()
        new = render(k, FT, S, old, rescaffold=args.rescaffold)
        if args.stdout:
            print(new)
            continue
        if new == old:
            continue
        if args.check:
            stale.append(path)
            continue
        with open(path, "w") as fh:
            fh.write(new)
        wrote.append(path)

    if args.stdout:
        return 0
    if args.check:
        if stale:
            print(f"STALE: {len(stale)} deep-dive(s) do not match the data.")
            for p in stale:
                print("  -", os.path.basename(p))
            print("       Run: python tools/emit_solver_failures.py")
            return 1
        print(f"OK: all {len(keys)} deep-dive tables are current.")
        return 0
    if not wrote:
        print(f"unchanged: all {len(keys)} deep-dive(s) already current.")
        return 0
    for p in wrote:
        print("wrote", os.path.basename(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
