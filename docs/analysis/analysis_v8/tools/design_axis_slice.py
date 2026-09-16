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

"""Slice solver eval results by the benchmark's *designed* difficulty axes.

The per-task eval metadata only varies on `round_type` and `scenario_slug`; the
deliberately-designed difficulty structure lives in the 20 scenario *seeds*
(`clasharness/data/scenario_seeds/conflict_vlm_v1/*.json`). This script joins
each eval task to its seed (by `scenario_slug` with the trailing `_v<NN>` variant
suffix stripped) and reports pass@budget / avg_score grouped by the designed axes:

  * conflict plugin     -- the 23 registered conflict plugins, bucketed by the
                           pipeline's own routing rule. *Overlapping*: a seed
                           activates 4-7 plugins, so the buckets do not sum.
  * conflict category   -- ConflictPluginSpec.category of the plugins a seed
                           routes to, excluding the 7 `challenge` plugins (all
                           20 seeds carry those, so they cannot discriminate).
                           Also *overlapping*.
  * difficulty knobs    -- force_vision_locus (vision actually graded) and
                           force_async_wait (temporal async required).
  * difficulty_challenges -- the 7 VLM/reasoning challenge tokens. These are
                           *overlapping* (a seed carries several), so the buckets
                           do not sum to the task count.

The plugin and category axes are derived from the *same* call the generator
makes -- `get_plugins_for_taxonomy(conflict_taxonomy + difficulty_challenges)`
(conflict_llm_pipeline.py:1582) -- so this module cannot drift from the routing
that actually shaped the scenarios. `archetype` does not participate in routing
at all and is therefore no longer a grouping key here; the legacy
ARCHETYPE_CATEGORY map is retained only for the footnote that contrasts the two.

Every bucket is also reported as a delta against *that model's own* 600-task
baseline. Raw bucket rates are not comparable across models here -- the field
spans 3.2% to 25.3% pass@budget -- so "which lever hurts" is only readable
relative to each model's own level.

Every bucket row carries a `seeds` count, and it is load-bearing: 10 of the 23
plugins are carried by a single seed (30 tasks per model). In those buckets the
plugin's effect is not separable from that one scenario's difficulty, so a large
delta is a statement about a scenario, not about a designed lever. Read the
`seeds` column before reading the rate.

It is deliberately dependency-free (stdlib only) and read-only. Numbers feed
section 5.3 of eval-results-capability-analysis.md and the SF11 block of the
per-model deep-dives, and are re-verified by tools/audit_report.py (passes D13,
D14 and E14).

V8 corpus: fourteen models, r01. Ten were run as a low/medium/high effort sweep
under eval_results/data_3_6_flash_v8/aligned_setting_*/; the other four were run
once each and are pinned to a fixed subtree by FIXED_TREE, which means they
IGNORE --effort. Three of those four are the open-source rows; the fourth,
deepseek-v4-flash, is an API model that was staged separately under
aligned_setting_high_ocr/ and so is NOT open-weight -- do not read FIXED_TREE
membership as a provider grouping. The default slice is --effort high, chosen
because it is the only setting on which all fourteen models are comparable.
`OpenAI/5_5` is excluded -- it exists in the tree but every one of its 480 v8
records scores zero with no check_results, so it is not a measurement.
The 60 scenarios are the 20 seeds x 3 variants, so every scenario_slug maps to a
seed and the unmapped-bases warning should stay silent.

TWO ROOTS ARE REQUIRED. The v8 eval corpus and the scenario seeds do not live
under a common directory: `data_3_6_flash_v8` exists only in the data checkout
(`--root`), and the seeds exist only in the source tree (`--seeds`, defaulted
off this file's own location). Passing a single root will silently find zero
seeds, which the seed-count warning below catches.

Usage:
    python3 docs/analysis/analysis_v8/tools/design_axis_slice.py
    python3 docs/analysis/analysis_v8/tools/design_axis_slice.py --model opus5
    python3 docs/analysis/analysis_v8/tools/design_axis_slice.py --markdown
    python3 docs/analysis/analysis_v8/tools/design_axis_slice.py --json
    python3 docs/analysis/analysis_v8/tools/design_axis_slice.py --root DATA --seeds DIR
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import re
import sys
from collections import defaultdict

# (display name, eval_results vendor path, short key) -- mirrors
# build_behavior_report.py MODELS. gpt-5.5 is absent by design; see the module
# docstring.
MODELS = [
    ("gemini-3.5-flash", "Gemini/3_5_flash", "flash"),
    ("gemini-3.6-flash", "Gemini/3_6_flash", "flash36"),
    ("gemini-3.7-flash", "Gemini/3_7_flash", "flash37"),
    ("gemini-3.1-pro", "Gemini/3_1_pro", "pro"),
    ("claude-opus-4.8", "Claude/opus_4_8", "opus"),
    ("claude-opus-5", "Claude/opus_5", "opus5"),
    ("claude-sonnet-5", "Claude/sonnet_5", "sonnet"),
    ("gpt-5.6-sol", "OpenAI/5_6_sol", "sol"),
    ("gpt-5.6-luna", "OpenAI/5_6_luna", "luna"),
    ("gpt-5.6-terra", "OpenAI/5_6_terra", "terra"),
    ("deepseek-v4-pro", "DeepSeek/v4_pro", "dsv4"),
    ("deepseek-v4-flash", "DeepSeek/dsv4_flash_v8_high_ocr", "dsv4f"),
    ("kimi-k3", "Moonshot/kimi_k3", "kimi"),
    ("glm-5.2", "ZhipuAI/glm_5_2", "glm"),
]

# Mirrors build_behavior_report.py EFFORT_TREE / DEFAULT_RUN.
EFFORT_TREE = "data_3_6_flash_v8/aligned_setting_{eff}"
EFFORTS = ("low", "medium", "high")
# high, not medium: it is the only effort at which all fourteen models sit in one
# coordinate system, because the four FIXED_TREE models have no other setting.
PRIMARY_EFFORT = "high"
RUN_SUBDIR = "r01"

# Models pinned to ONE subtree, ignoring --effort, because they were not run as an
# effort sweep. Mirrors build_behavior_report.FIXED_TREE, but the paths here are
# written WITHOUT the leading `eval_results/`: load_model() inserts that segment,
# exactly as it does for EFFORT_TREE. (scan_difficulty_axes.py keeps the same dict
# WITH the prefix because its loader joins differently -- do not copy one into the
# other verbatim.)
FIXED_TREE = {
    "dsv4": "data_3_6_flash_v8/open_source_high/DeepSeek/v4_pro",
    "dsv4f": ("data_3_6_flash_v8/aligned_setting_high_ocr/DeepSeek/"
              "dsv4_flash_v8_high_ocr"),
    "kimi": "data_3_6_flash_v8/open_source_high/Moonshot/kimi_k3",
    "glm": "data_3_6_flash_v8/open_source_high/ZhipuAI/glm_5_2",
}

# Compact-but-versioned headers for the --markdown cross-model tables (mirrors
# build_behavior_report.py MODEL_LABEL).
MODEL_LABEL = {"flash": "3.5 flash", "flash36": "3.6 flash",
               "flash37": "3.7 flash", "pro": "3.1 pro",
               "opus": "opus 4.8", "opus5": "opus 5", "sonnet": "sonnet 5",
               "sol": "5.6-sol", "luna": "5.6-luna", "terra": "5.6-terra",
               "dsv4": "v4-pro", "dsv4f": "v4-flash",
               "kimi": "kimi-k3", "glm": "glm-5.2"}

# Seeds live in the source tree, not the data checkout. parents[4] walks
# tools -> analysis_v8 -> analysis -> docs -> repo root.
SEED_DIR = os.path.join("clasharness", "data", "scenario_seeds", "conflict_vlm_v1")
DEFAULT_SEEDS = str(pathlib.Path(__file__).resolve().parents[4] / SEED_DIR)
# Corpus root: the data checkout holding eval_results/. Mirrors
# analysis/corpus.py -- CLASHARNESS_DATA_ROOT, else <repo>/data.
DEFAULT_ROOT = os.environ.get(
    "CLASHARNESS_DATA_ROOT",
    str(pathlib.Path(__file__).resolve().parents[4] / "data"))
DEFAULT_OUT = str(pathlib.Path(__file__).resolve().parents[1] / "design_axes.json")

REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[4])

# The plugin registry is imported, not hand-copied, so the plugin and category
# axes are grouped by exactly what the generator routed on. tools/audit_report.py
# pass D13 re-derives the same registry by *parsing* the plugin modules, so the
# two paths stay independent and a bad import here cannot audit itself clean.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
try:
    from clasharness.scenario.plugins import (  # noqa: E402
        _ALL_PLUGINS, get_plugins_for_taxonomy)
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        f"design_axis_slice: cannot import the conflict plugin registry from "
        f"{REPO_ROOT!r} ({exc}). This tool groups by the generator's own routing "
        f"and has no fallback; run it from inside the source tree.") from exc

# name -> ConflictPluginSpec.category, and the challenge subset. Tokens are 1:1
# with plugin names across all 23 plugins (no plugin declares taxonomy_tokens),
# so routing is just "keep the known names" and is order-independent.
PLUGIN_SPECS = [p.SPEC for p in _ALL_PLUGINS if p.SPEC is not None]
PLUGIN_CATEGORY = {s.name: s.category for s in PLUGIN_SPECS}
CHALLENGE_PLUGINS = {n for n, c in PLUGIN_CATEGORY.items() if c == "challenge"}

# Plugins that gate real command execution (they declare emits_effects) versus
# those that only inject prompt text. Not an axis -- section 5.3 reports the
# 1-vs-2 contrast as prose -- but the split is needed for that sentence.
MECHANIC_PLUGINS = {s.name for s in PLUGIN_SPECS
                    if tuple(getattr(s, "emits_effects", ()) or ())}

# Legacy archetype -> category map, retained only for the section 5.3 footnote
# that contrasts it with ConflictPluginSpec.category. It disagrees with the
# plugin specs on 5 of its 15 entries and has no `coordination` bucket at all,
# and `archetype` never reaches the router, so it no longer groups anything.
# Kept byte-identical to clasharness/scenario/conflict_vlm_spec.py (audit D13).
ARCHETYPE_CATEGORY = {
    # 1. Main Agent Supervised
    "evidence_mismatch": "main_supervised",
    "scope_reconciliation": "main_supervised",
    # 2. SubAgent Collaboration
    "multi_party_synchronization": "subagent_collaboration",
    "coordinated_containment": "subagent_collaboration",
    "non_overlapping_assignment": "subagent_collaboration",
    "duplicate_hold_stale": "subagent_collaboration",
    # 3. Timeout
    "time_window_expiration": "timeout",
    "reversible_action_hold": "timeout",
    "multi_device_pairing": "timeout",
    # 4. Order
    "dependency_ordering": "order",
    "controlled_deployment_gate": "order",
    # 5. Inconsistency
    "inconsistency_expense": "inconsistency",
    "distributed_lock": "inconsistency",
    "deadlock_dependency": "inconsistency",
    "barrier_synchronization": "inconsistency",
}

CHALLENGE_TOKENS = [
    "workflow_completion",
    "information_transfer",
    "multi_source_reasoning",
    "state_repair",
    "gap_completion",
    "duplicate_avoidance",
    "vision_decoy",
]

# Every non-challenge category that at least one plugin declares, most-used
# first. `challenge` is excluded on purpose: all 20 seeds carry challenge
# plugins, so that bucket would be the whole corpus and discriminate nothing.
CAT_ORDER = ["inconsistency", "coordination", "order", "timeout",
             "subagent_collaboration", "main_supervised"]
VISION_ORDER = ["vision_graded", "not_vision_graded"]
ASYNC_ORDER = ["async_required", "no_async"]

# The plugin axis has 23 buckets whose useful ordering (widest coverage first)
# is a property of the seed corpus, not a constant, so it is resolved at render
# time by `axis_order` against the per-bucket seed counts.
PLUGIN_ORDER = None

# (markdown heading, axis key, bucket order). The heading text is what section
# 5.3 of the capability doc carries, so audit pass D13 keys off it.
AXIS_META = [
    ("conflict plugin (overlapping; buckets do not sum)", "plugin", PLUGIN_ORDER),
    ("conflict category (overlapping; buckets do not sum)", "category", CAT_ORDER),
    ("difficulty knob: force_vision_locus (vision graded)", "vision", VISION_ORDER),
    ("difficulty knob: force_async_wait (temporal async)", "async", ASYNC_ORDER),
    ("difficulty_challenge (overlapping tokens; buckets do not sum)", "challenge",
     CHALLENGE_TOKENS),
]

# Rows of the relative-to-baseline table: the two knob axes, the challenge
# tokens, then every conflict plugin (appended at render time by `delta_order`,
# since the plugin order is data-derived). The plugin buckets belong here
# because the "below baseline" column is the only place the cross-model
# *unanimity* of a lever is readable, and unanimity is what survives the
# small-bucket noise. The category axis stays out: a category is a grouping of
# levers, not a lever, so a category delta has no single design decision behind
# it.
DELTA_HEAD = ASYNC_ORDER + VISION_ORDER + CHALLENGE_TOKENS


def load_seeds(seed_dir):
    """base seed slug -> seed dict."""
    seeds = {}
    for f in sorted(glob.glob(os.path.join(seed_dir, "*.json"))):
        with open(f, encoding="utf-8") as fh:
            seeds[os.path.basename(f)[:-len(".json")]] = json.load(fh)
    return seeds


def load_model(root, vendor, eff, key=None):
    """Records for one model, deduplicated by task_id (last wins).

    sol and terra have 604 raw lines for 600 tasks -- a few re-attempts are
    appended -- so a raw read would inflate their denominators and disagree with
    build_behavior_report / parallel_run.summary. Last-occurrence-wins matches both.

    A FIXED_TREE model ignores `eff` and `vendor`: its path is fully specified.
    """
    if key in FIXED_TREE:
        d = os.path.join(root, "eval_results", FIXED_TREE[key], RUN_SUBDIR)
    else:
        d = os.path.join(root, "eval_results", EFFORT_TREE.format(eff=eff), vendor,
                         RUN_SUBDIR)
    by_id = {}
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    by_id[r.get("task_id")] = r
    return list(by_id.values())


def base_of(slug):
    """conflict scenario_slug -> base seed name (strip trailing _v<NN> variant)."""
    return re.sub(r"_v\d+$", "", slug or "")


class Agg:
    """pass/avg/n accumulator keyed by an arbitrary bucket label."""

    def __init__(self):
        self.n = defaultdict(int)
        self.passed = defaultdict(int)
        self.score = defaultdict(float)

    def add(self, key, success, score):
        self.n[key] += 1
        self.passed[key] += 1 if success else 0
        self.score[key] += score

    def rows(self, order=None):
        # Name breaks the count tie so the fallback ordering is deterministic;
        # design_axes.json is diffed byte-for-byte across reruns.
        keys = order or sorted(self.n, key=lambda k: (-self.n[k], k))
        for k in keys:
            if k not in self.n:
                continue
            n = self.n[k]
            yield k, self.passed[k], n, 100 * self.passed[k] / n, self.score[k] / n


def seed_plugins(seed):
    """The conflict plugins one seed activates, as the generator routes them.

    Mirrors conflict_llm_pipeline.py:1582 exactly: `conflict_taxonomy` and
    `difficulty_challenges` are concatenated into one token list and resolved
    through the same registry index, with unknown tokens dropped. `archetype`
    is deliberately absent -- it never reaches the router.

    Note the registry's own edge case: `get_plugins_for_taxonomy([])` returns
    *every* plugin. No seed in this corpus hits it (all 20 carry 4-7 tokens),
    but a seed that ever did would silently join all 23 buckets, so guard it.
    """
    tokens = ((seed.get("conflict_taxonomy") or [])
              + (seed.get("difficulty_challenges") or []))
    if not tokens:
        return []
    return [p.SPEC.name for p in get_plugins_for_taxonomy(tokens)
            if p.SPEC is not None]


def seed_buckets(seed):
    """One seed -> the (axis, bucket) pairs it belongs to, for every axis."""
    plugins = seed_plugins(seed)
    pairs = [("plugin", p) for p in plugins]
    pairs += [("category", c) for c in sorted(
        {PLUGIN_CATEGORY[p] for p in plugins if p not in CHALLENGE_PLUGINS})]
    pairs.append(("vision", "vision_graded" if seed.get("force_vision_locus")
                  else "not_vision_graded"))
    pairs.append(("async", "async_required" if seed.get("force_async_wait")
                  else "no_async"))
    pairs += [("challenge", t) for t in (seed.get("difficulty_challenges") or [])]
    return pairs


AXES = ("plugin", "category", "vision", "async", "challenge")


def bucket_seed_counts(seeds):
    """(axis, bucket) -> number of *seeds* in it. Model-independent.

    Reported next to every rate because task counts alone hide that 10 of the
    23 plugin buckets rest on a single scenario.
    """
    counts = defaultdict(set)
    for slug, sd in seeds.items():
        for pair in seed_buckets(sd):
            counts[pair].add(slug)
    return {k: len(v) for k, v in counts.items()}


def slice_model(recs, seeds):
    """Bucket one model's records onto every designed axis.

    Returns (aggs_by_axis, baseline, unmapped). `baseline` is the model's own
    all-tasks rate, which every delta in this module is taken against.
    """
    aggs = {a: Agg() for a in AXES}
    unmapped = set()
    n = passed = 0
    total = 0.0
    for r in recs:
        base = base_of((r.get("metadata") or {}).get("scenario_slug"))
        s = seeds.get(base)
        succ = bool(r.get("success"))
        sc = float(r.get("score") or 0)
        if s is None:
            unmapped.add(base)
            continue
        n += 1
        passed += 1 if succ else 0
        total += sc
        for axis, bucket in seed_buckets(s):
            aggs[axis].add(bucket, succ, sc)
    baseline = {"n": n, "passed": passed,
                "pass_pct": 100 * passed / n if n else 0.0,
                "avg": total / n if n else 0.0}
    return aggs, baseline, unmapped


def collect(root, seeds, eff):
    """model key -> {"baseline": .., "axes": {axis: {bucket: rec}}, "unmapped": ..}.

    A model with no records on disk is omitted entirely rather than reported as
    zero, so a missing tree cannot masquerade as a failing model.
    """
    out = {}
    nseeds = bucket_seed_counts(seeds)
    for _, vendor, key in MODELS:
        recs = load_model(root, vendor, eff, key)
        if not recs:
            continue
        aggs, baseline, unmapped = slice_model(recs, seeds)
        axes = {}
        for axis, agg in aggs.items():
            axes[axis] = {
                b: {"passed": p, "n": bn, "pass_pct": pct, "avg": avg,
                    "seeds": nseeds.get((axis, b), 0),
                    "d_pass_pct": pct - baseline["pass_pct"],
                    "d_avg": avg - baseline["avg"]}
                for b, p, bn, pct, avg in agg.rows()
            }
        out[key] = {"baseline": baseline, "axes": axes, "unmapped": sorted(unmapped)}
    return out


def axis_order(axis, order, data, keys):
    """Resolve an AXIS_META bucket order against the data actually present.

    A static order is filtered to the buckets that exist; a `None` order (the
    plugin axis) is derived widest-coverage-first, with the bucket name breaking
    ties so the rendering is reproducible.
    """
    present = {b for k in keys for b in data[k]["axes"].get(axis, {})}
    if order is not None:
        return [b for b in order if b in present]

    def rank(b):
        rec = next((data[k]["axes"][axis][b] for k in keys
                    if b in data[k]["axes"][axis]), None)
        return (-(rec or {}).get("seeds", 0), b)

    return sorted(present, key=rank)


def delta_order(data, keys):
    """Rows of the relative-to-baseline table: fixed head, then every plugin.

    The 7 challenge plugins are already in DELTA_HEAD as challenge tokens (the
    token and the plugin are the same bucket over the same tasks), so they are
    dropped from the tail rather than rendered as a second identical row.
    """
    tail = [b for b in axis_order("plugin", PLUGIN_ORDER, data, keys)
            if b not in set(DELTA_HEAD)]
    return DELTA_HEAD + tail


# ---------------------------------------------------------------- rendering


def cell(rec):
    return "\u2014" if not rec else f"{rec['pass_pct']:.1f}% ({rec['avg']:.2f})"


def bold_max(recs, cells):
    """Bold the row-maximum cell(s) by pass rate.

    Emphasis is generated, never hand-applied: audit pass D13 recomputes which
    cells should be bold, so a stale bold marker is a hard failure rather than a
    cosmetic drift.
    """
    vals = [r["pass_pct"] for r in recs if r]
    if not vals:
        return cells
    top = max(vals)
    return [f"**{c}**" if r and r["pass_pct"] == top else c
            for r, c in zip(recs, cells)]


def markdown(data, eff):
    keys = [k for _, _, k in MODELS if k in data]
    hdr = " | ".join(MODEL_LABEL[k] for k in keys)
    rule = "|---|---:|---:|" + "---:|" * len(keys)
    out = []
    for title, axis, order in AXIS_META:
        # Only the plugin table carries a category column: it is what keeps the
        # 23-row view from being a bare superset of the challenge table, and it
        # is the join back to the category table below it.
        catcol = axis == "plugin"
        out.append(f"\n#### by {title}\n")
        out.append(f"| bucket | {'category | ' if catcol else ''}seeds | tasks "
                   f"| {hdr} |")
        out.append(("|---|---|---:|---:|" if catcol else "|---|---:|---:|")
                   + "---:|" * len(keys))
        for b in axis_order(axis, order, data, keys):
            recs = [data[k]["axes"][axis].get(b) for k in keys]
            n = next((r["n"] for r in recs if r), 0)
            sd = next((r["seeds"] for r in recs if r), 0)
            cells = bold_max(recs, [cell(r) for r in recs])
            pre = f"{PLUGIN_CATEGORY.get(b, '?')} | " if catcol else ""
            out.append(f"| {b} | {pre}{sd} | {n} | " + " | ".join(cells) + " |")

    out.append("\n#### relative to each model's own baseline "
               "(pass-point delta)\n")
    out.append(f"| lever | seeds | tasks | {hdr} | below baseline |")
    out.append(rule + "---:|")
    axis_of = {b: a for _, a, order in AXIS_META if order for b in order}
    for b in delta_order(data, keys):
        axis = axis_of.get(b, "plugin")
        recs = [data[k]["axes"][axis].get(b) for k in keys]
        if not any(recs):
            continue
        n = next((r["n"] for r in recs if r), 0)
        sd = next((r["seeds"] for r in recs if r), 0)
        cells = [("\u2014" if not r else f"{r['d_pass_pct']:+.1f}") for r in recs]
        below = sum(1 for r in recs if r and r["d_pass_pct"] < 0)
        out.append(f"| {b} | {sd} | {n} | " + " | ".join(cells)
                   + f" | {below}/{len(keys)} |")

    base_cells = " | ".join(
        f"{data[k]['baseline']['pass_pct']:.1f}% ({data[k]['baseline']['avg']:.2f})"
        for k in keys)
    out.append(f"\nBaseline (all {data[keys[0]]['baseline']['n']} tasks, "
               f"`reasoning_effort={eff}`): {base_cells}.")
    return "\n".join(out)


def print_tables(name, key, entry):
    b = entry["baseline"]
    print(f"\n########## {name} ({key}) ##########")
    print(f"  overall: {b['passed']}/{b['n']} = {b['pass_pct']:.1f}%   "
          f"avg {b['avg']:.3f}")
    if entry["unmapped"]:
        print(f"  WARNING unmapped scenario bases: {entry['unmapped']}")
    for title, axis, order in AXIS_META:
        print(f"\n--- by {title} ---")
        print(f"  {'bucket':32s} {'sd':>3s} {'pass':>10s}  {'pass%':>6s}  "
              f"{'avg':>6s}  {'dpp':>6s}")
        buckets = order if order is not None else sorted(
            entry["axes"][axis],
            key=lambda b: (-entry["axes"][axis][b]["seeds"], b))
        for bk in buckets:
            r = entry["axes"][axis].get(bk)
            if not r:
                continue
            print(f"  {bk:32s} {r['seeds']:3d} {r['passed']:4d}/{r['n']:<5d} "
                  f"{r['pass_pct']:6.1f}  {r['avg']:6.3f}  {r['d_pass_pct']:+6.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help="data checkout containing eval_results/")
    ap.add_argument("--seeds", default=DEFAULT_SEEDS,
                    help="scenario-seed directory (source tree, not --root)")
    ap.add_argument("--effort", default=PRIMARY_EFFORT, choices=list(EFFORTS))
    ap.add_argument("--model", default=None, help="short key: flash|flash36|"
                    "flash37|pro|opus|opus5|sonnet|sol|luna|terra|dsv4|kimi|glm")
    ap.add_argument("--markdown", action="store_true",
                    help="emit the cross-model tables for capability doc 5.3")
    ap.add_argument("--json", nargs="?", const=DEFAULT_OUT, default=None,
                    metavar="PATH", help=f"write the dump (default {DEFAULT_OUT})")
    args = ap.parse_args()

    seeds = load_seeds(args.seeds)
    if len(seeds) != 20:
        print(f"WARNING: expected 20 seeds, found {len(seeds)} in {args.seeds}")
    data = collect(args.root, seeds, args.effort)
    if not data:
        raise SystemExit(f"no eval records under {args.root}/eval_results/"
                         f"{EFFORT_TREE.format(eff=args.effort)}")
    stray = {k: v["unmapped"] for k, v in data.items() if v["unmapped"]}
    if stray:
        print(f"WARNING unmapped scenario bases: {stray}")

    if args.json:
        mkeys = [k for _, _, k in MODELS if k in data]
        payload = {
            "corpus": EFFORT_TREE.format(eff=args.effort),
            "effort": args.effort,
            # The FIXED_TREE models are NOT in `corpus`/`effort` above: they came
            # from their own subtree at one setting. Ship the mapping so a reader
            # can tell which rows the top-level effort actually describes.
            "fixed_tree": FIXED_TREE,
            "run": RUN_SUBDIR,
            "n_seeds": len(seeds),
            "models": mkeys,
            # Resolved, not declared: the plugin axis has no static order, and a
            # consumer must not have to re-derive it to match this file's tables.
            "axis_order": {a: axis_order(a, order, data, mkeys)
                           for _, a, order in AXIS_META},
            "delta_order": delta_order(data, mkeys),
            "plugin_category": PLUGIN_CATEGORY,
            "mechanic_plugins": sorted(MECHANIC_PLUGINS),
            "by_model": data,
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"wrote {args.json}")
        return

    if args.markdown:
        print(markdown(data, args.effort))
        return

    for name, _, key in MODELS:
        if key not in data or (args.model and key != args.model):
            continue
        print_tables(name, key, data[key])


if __name__ == "__main__":
    main()
