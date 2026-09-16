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

"""Bucket every failed evaluator check in the v8 corpus into a failure taxonomy.

Ported from `analysis_think_history/tools/failure_taxonomy.py`. The per-model
deep-dives (`<model>-solver-failures.md`) and the capability analysis quote
failed-check counts per failure mode; deriving them mechanically is what lets
`audit_report.py` recompute rather than merely spot the numbers in prose.

Four things differ from the think-history original and every one of them is a
correctness issue, not a cosmetic one:

  * **The corpus has no `exists` checks.** v8 emits 28 distinct check *names*
    built from exactly four conditions -- `field_semantic_match`,
    `field_contains`, `any_field_equals`, `within_window`. The old bucketer
    keyed terminal side-effects off ``name.endswith(" exists")``, so all 24
    `effect:<x> any_field_equals` names would have silently landed in `other`.
    `bucket_of` below keys off `any_field_equals` instead, and `other` is now a
    guard bucket that MUST stay empty -- `main` exits non-zero if it does not.

  * **Three efforts, but not for every model.** v8 pins `reasoning_effort` to
    low / medium / high across TEN swept models; the three open-source models
    (FIXED_TREE) were run once each and appear at `high` only. That is 10x3 + 3
    = **33 cells**, 600 tasks each, 19,800 episodes. Everything here is computed
    per cell; `high` is the primary effort the hub report leads with, because it
    is the only setting on which all thirteen models are comparable.

  * **gpt-5.5 does not exist in v8.** `eval_results/.../OpenAI/5_5/` is present
    under medium and high but its logs are empty, so the model is absent from
    `MODELS`, which holds **thirteen** models.

  * **No duplicate result rows.** The think-history corpus had 604 rows for 600
    tasks on two models, which is why records are de-duplicated last-wins by
    `task_id`. All 33 v8 cells verify 600 rows / 600 tasks, so the dedup is a
    no-op here -- it is retained as a guard and `dupe_rows` reports it.

Two invariants carried over unchanged:

  * **Role decides impact.** Only `role == "required"` gates `success`;
    `diagnostic` checks affect neither pass nor score. Every count is reported
    split by role, never merged. `role` defaults to `"required"` to match
    `CheckResult`'s own default (clasharness/evaluation/results.py). In v8 the
    only roles observed are `required` (108,333) and `diagnostic` (15,702), and
    the diagnostic set is entirely `effect:assets_accessed field_contains`.
    A complete cell carries 3,284 required and 476 diagnostic checks; 30 of the
    33 cells are complete, and the three that are not are exactly the three with
    `missing_check_results` (medium/3.5-flash 3,278+476, medium/3.1-pro
    3,278+475, high/3.1-pro 3,257+471). Those two facts reconcile to the totals
    above -- if they stop reconciling, a cell lost records.

  * **Only records carrying `metadata` + `score` + `task_id` are results.** The
    result `.jsonl` files interleave task results with per-call generation
    records and a run-config row. `log/` subdirectories are skipped outright.

Seven tasks corpus-wide carry `metadata` with no `check_results` (1 under
medium/3.5-flash, 1 under medium/3.1-pro, 5 under high/3.1-pro). They still
count toward `n`, `passed` and `stops`; they contribute no checks. The count is
reported as `missing_check_results` so downstream denominators can be explained
rather than quietly differing by one.

Usage:
    python3 tools/failure_taxonomy.py                      # all cells, text + JSON
    python3 tools/failure_taxonomy.py --effort medium
    python3 tools/failure_taxonomy.py --model opus5 --examples 8
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import re
import statistics
import sys

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
    ("kimi-k3", "Moonshot/kimi_k3", "kimi"),
    ("glm-5.2", "ZhipuAI/glm_5_2", "glm"),
]
MODEL_KEYS = [k for *_, k in MODELS]
EFFORTS = ["low", "medium", "high"]
PRIMARY_EFFORT = "high"
BASE = "eval_results/data_3_6_flash_v8/aligned_setting_{effort}"

# Per-model run-subdirectory override. MUST MIRROR build_behavior_report's
# DEFAULT_RUN / RUN_SUBDIR / run_for, because both tools are read as describing
# the same episodes and the paper cross-checks this file against the corpus
# row-for-row.
#
# This used to be a bare `RUN = "r01"` with no override, which meant gemini-3.7-
# flash was bucketed here from r01 while every other artefact -- the behavioural
# report, its stats.json, and every table in the paper -- reports it from r02.
# The two runs are both complete and clean, but they are not the same draw:
#     r01  18.67% pass (112)  246 capped
#     r02  20.33% pass (122)  244 capped
# so the mismatch was worth 1.7 points of pass rate on one row, and it was
# invisible here because nothing in this file compares itself to the report.
# The paper's src/make_body_figures.taxonomy() does compare, and that is what
# caught it. Keep the two rosters' run selection in lockstep.
DEFAULT_RUN = "r01"
RUN_SUBDIR = {"flash37": "r02"}

# Models pinned to ONE subtree because they were not run as an effort sweep.
# Complete eval_results-relative paths, mirroring build_behavior_report.FIXED_TREE.
#
# THEY APPEAR IN THE `high` CELL ONLY, and that is deliberate. Loading them under
# low and medium would succeed -- the path ignores `effort` -- and would silently
# copy the same run into three cells, making it look as though they were swept and
# happened to score identically at every setting. They were not swept. `main` skips
# them outside PRIMARY_EFFORT so an absent key means "not measured here", which is
# the truth. Consumers that iterate all three efforts per model must tolerate that
# absence (see emit_solver_failures.sf4_effort).
FIXED_TREE = {
    "dsv4": "eval_results/data_3_6_flash_v8/open_source_high/DeepSeek/v4_pro",
    "kimi": "eval_results/data_3_6_flash_v8/open_source_high/Moonshot/kimi_k3",
    "glm": "eval_results/data_3_6_flash_v8/open_source_high/ZhipuAI/glm_5_2",
}

# (bucket key, human label, doc id). Order is the order the docs present them in.
BUCKETS = [
    ("content_msg", "Content wrong / insufficient -- message", "F1"),
    ("content_file", "Content wrong / insufficient -- written file", "F2"),
    ("grounding", "Grounding gap -- required asset never opened", "F3"),
    ("anchor", "Missing evidence anchor (literal substring in a file)", "F4"),
    ("side_effect", "Required terminal action absent", "F5"),
    ("window", "Acted outside the required time window", "F6"),
    ("other", "UNBUCKETED (guard -- must stay empty)", "--"),
]
BUCKET_LABEL = {k: lab for k, lab, _ in BUCKETS}
BUCKET_FID = {k: fid for k, _, fid in BUCKETS}
CONTENT_BUCKETS = ("content_msg", "content_file", "anchor")

# Metrics ranked across the nine models. True => larger is better.
RANKED = [
    ("pass_pct", True), ("avg_score", True), ("failed_total", False),
    ("loopy_capped", False), ("step_p50", False), ("step_mean", False),
    ("capped", False), ("zero_write_tasks", False), ("zero_open_tasks", False),
    ("zero_score", False), ("near_miss", False),
]


def bucket_of(name: str) -> str:
    """Map an evaluator check name onto a taxonomy bucket.

    Matching is on name *shape*, not an enumerated allowlist, so a scenario that
    introduces another `effect:<x> any_field_equals` check lands in `side_effect`
    rather than silently vanishing. Anything matching no shape lands in `other`,
    which `main` treats as a hard error.
    """
    if name.startswith("effect:assets_accessed "):
        return "grounding"
    if name.startswith("effect:messages_sent "):
        if name.endswith(" within_window"):
            return "window"
        if name.endswith(" field_semantic_match"):
            return "content_msg"
        return "other"
    if name.startswith("effect:files_created "):
        if name.endswith(" field_contains"):
            return "anchor"
        if name.endswith(" field_semantic_match"):
            return "content_file"
        return "other"
    if name.startswith("effect:") and name.endswith(" any_field_equals"):
        return "side_effect"
    return "other"


def run_for(key: str | None, effort: str) -> str:
    """Run subdirectory for one (model, effort) cell.

    Overrides apply at PRIMARY_EFFORT only, because that is the only effort at
    which a second run exists: r02 is present for 3_7_flash under
    aligned_setting_high and nowhere else in the tree. Returning "r02" for a
    low/medium cell would resolve to a directory that does not exist and fail
    far from the cause. FIXED_TREE models reach this too and are never in
    RUN_SUBDIR, so they keep DEFAULT_RUN.
    """
    if key is not None and effort == PRIMARY_EFFORT:
        return RUN_SUBDIR.get(key, DEFAULT_RUN)
    return DEFAULT_RUN


def load_tasks(root: str, sub: str, effort: str, key: str | None = None):
    """Return ({task_id: result_record}, n_rows_seen), de-duplicated last-wins.

    A FIXED_TREE model ignores `effort` and `sub`: its path is fully specified.
    """
    seen: dict = {}
    rows = 0
    run = run_for(key, effort)
    if key in FIXED_TREE:
        d = pathlib.Path(root) / FIXED_TREE[key] / run
    else:
        d = pathlib.Path(root) / BASE.format(effort=effort) / sub / run
    if not d.is_dir():
        raise SystemExit(f"missing results dir: {d}")
    for f in sorted(d.rglob("*.jsonl")):
        if f.parent.name == "log":
            continue
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if "metadata" in r and "score" in r and "task_id" in r:
                    rows += 1
                    seen[r["task_id"]] = r
    return seen, rows


def action_str(step) -> str:
    a = step.get("action")
    return a if isinstance(a, str) else json.dumps(a)


def action_key(action: str) -> str:
    """`<head> <sub>` sub-command key -- mirrors build_behavior_report.categorize."""
    t = action.strip().split()
    if not t:
        return "(empty)"
    return (t[0] + " " + (t[1] if len(t) > 1 else "")).strip()


def is_single_verb_loop(traj, thresh=5):
    """Canonical loop test: a step-capped run counts as a loop when its most
    frequent *sub-command* occurs >= 5 times.

    Deliberately NOT "3 identical consecutive actions": the environment already
    has its own guard for that (`stopped after 3 identical no-progress
    transitions`, which fires only 15 times in 16,200 v8 episodes). Real loops
    vary their arguments -- `message search --query` with degrading queries,
    identifier-guessing on `pairing show-code` -- so a consecutive-identical
    test scores almost every looping run as clean.
    """
    c = collections.Counter(action_key(action_str(s)) for s in traj)
    return bool(c) and max(c.values()) >= thresh


def top_loop_verb(traj):
    c = collections.Counter(action_key(action_str(s)) for s in traj)
    return c.most_common(1)[0] if c else ("", 0)


_WAIT = re.compile(r"^wait\s+(\d+(?:\.\d+)?)\b")


def wait_profile(tasks):
    """Duration histogram for `wait <seconds>` actions.

    The duration is the action's own argument, so this is exact rather than
    inferred from wall-clock. `sub_30` counts polls shorter than 30s -- the
    think-history corpus used it to separate "schedules" from "polls".
    """
    secs = []
    hist = collections.Counter()
    for r in tasks.values():
        for s in (r.get("trajectory") or []):
            m = _WAIT.match(action_str(s).strip())
            if m:
                secs.append(float(m.group(1)))
                hist[f"wait {m.group(1)}"] += 1
    if not secs:
        return {"n": 0, "sub_30": 0, "mean": 0.0, "median": 0.0,
                "max": 0.0, "total": 0.0, "hist": {}}
    return {
        "n": len(secs),
        "sub_30": sum(1 for v in secs if v < 30),
        "mean": round(statistics.fmean(secs), 1),
        "median": round(statistics.median(secs), 1),
        "max": max(secs),
        "total": round(sum(secs), 1),
        "hist": dict(hist.most_common(8)),
    }


def scenario_of(task_id: str) -> str:
    """`conflict_vlm_<scenario>_r<NN>` -> `<scenario>`."""
    s = re.sub(r"^conflict_vlm_", "", task_id)
    return re.sub(r"_r\d+$", "", s)


def analyze(tasks, rows, n_examples=4):
    n = len(tasks)
    passed = sum(1 for r in tasks.values() if r["success"])
    scores = [r["score"] for r in tasks.values()]
    steps = sorted(r["steps"] for r in tasks.values())

    stops = collections.Counter()
    for r in tasks.values():
        s = r.get("stop_reason") or "none"
        # `missing_required_side_effect: <field>` -- keep the family, count the
        # fields separately so the doc can name the top offenders.
        stops[s.split(":")[0].strip() if s.startswith("missing_required") else s] += 1
    side_fields = collections.Counter(
        r["stop_reason"].split(":", 1)[1].strip()
        for r in tasks.values()
        if (r.get("stop_reason") or "").startswith("missing_required_side_effect:")
    )

    # ---- failed checks, split by role -------------------------------------
    failed = {k: {"required": 0, "diagnostic": 0} for k, *_ in BUCKETS}
    total = {k: {"required": 0, "diagnostic": 0} for k, *_ in BUCKETS}
    examples = collections.defaultdict(list)
    unbucketed_names = collections.Counter()
    # Full check-name inventory per bucket. The capability-analysis document
    # states the rubric's shape ("28 distinct names, 23 of them
    # any_field_equals"); emitting the inventory here is what lets the audit
    # recompute that claim instead of trusting a constant.
    bucket_names = collections.defaultdict(collections.Counter)
    task_req_fail = {}
    missing_cr = []
    n_sem = n_sem_failclosed = n_ok_false = 0
    for tid, r in sorted(tasks.items()):
        cr = r["metadata"].get("check_results")
        if cr is None:
            missing_cr.append(tid)
            cr = []
        buckets_failed = set()
        for c in cr:
            name = c["name"]
            b = bucket_of(name)
            if b == "other":
                unbucketed_names[name] += 1
            bucket_names[b][name] += 1
            role = c.get("role") or "required"
            role = role if role in ("required", "diagnostic") else "required"
            total[b][role] += 1
            if c.get("ok") is False:
                n_ok_false += 1
            if name.endswith(" field_semantic_match"):
                n_sem += 1
                if c.get("judge_invoked") is False:
                    n_sem_failclosed += 1
            if not c.get("passed"):
                failed[b][role] += 1
                if role == "required":
                    buckets_failed.add(b)
                if len(examples[b]) < n_examples and c.get("message"):
                    examples[b].append(
                        {"task_id": tid, "role": role,
                         "message": re.sub(r"\s+", " ", c["message"]).strip()[:260]}
                    )
        task_req_fail[tid] = buckets_failed

    # ---- gating structure --------------------------------------------------
    # Of the tasks that fail, how many fail on grounding ALONE (i.e. the
    # deliverable was otherwise acceptable)? This is the figure that decides
    # whether a pass-rate adjustment on grounding would be defensible.
    fails = [t for t, r in sorted(tasks.items()) if not r["success"]]
    grounding_only = [t for t in fails if task_req_fail[t] == {"grounding"}]
    content_involved = [t for t in fails if task_req_fail[t] & set(CONTENT_BUCKETS)]
    no_req_fail = [t for t in fails if not task_req_fail[t]]

    # ---- participation / loops --------------------------------------------
    zero_open = zero_write = capped = loopy = 0
    loop_verbs = collections.Counter()
    loop_examples = []
    for tid, r in sorted(tasks.items()):
        traj = r.get("trajectory") or []
        acts = [action_str(s) for s in traj]
        if not any(a.startswith("asset open") for a in acts):
            zero_open += 1
        if not any(a.startswith(("file create", "file write", "file append")) for a in acts):
            zero_write += 1
        if (r.get("stop_reason") or "").startswith("reached max_steps"):
            capped += 1
            if is_single_verb_loop(traj):
                loopy += 1
                verb, cnt = top_loop_verb(traj)
                loop_verbs[verb] += 1
                if len(loop_examples) < n_examples:
                    loop_examples.append({"task_id": tid, "verb": verb,
                                          "count": cnt, "score": r["score"]})

    # ---- per-scenario roll-up ---------------------------------------------
    per_scen = collections.defaultdict(list)
    for tid, r in tasks.items():
        per_scen[scenario_of(tid)].append(r)
    scenarios = {
        s: {"n": len(v),
            "mean_score": round(sum(x["score"] for x in v) / len(v), 4),
            "passed": sum(1 for x in v if x["success"])}
        for s, v in sorted(per_scen.items())
    }

    return {
        "n": n,
        "rows": rows,
        "dupe_rows": rows - n,
        "missing_check_results": len(missing_cr),
        "missing_check_result_ids": missing_cr,
        "passed": passed,
        "pass_pct": round(100.0 * passed / n, 1),
        "avg_score": round(sum(scores) / n, 4),
        "near_miss": sum(1 for r in tasks.values() if not r["success"] and r["score"] >= 0.5),
        "zero_score": sum(1 for r in tasks.values() if r["score"] == 0.0),
        "step_p50": steps[n // 2],
        "step_mean": round(sum(steps) / n, 2),
        "capped": capped,
        "loopy_capped": loopy,
        "loop_verbs": dict(loop_verbs.most_common(6)),
        "loop_examples": loop_examples,
        "zero_open_tasks": zero_open,
        "zero_write_tasks": zero_write,
        "stops": dict(stops.most_common()),
        "side_effect_fields": dict(side_fields.most_common()),
        "failed_checks": {k: failed[k] for k, *_ in BUCKETS},
        "total_checks": {k: total[k] for k, *_ in BUCKETS},
        "failed_total": sum(v["required"] + v["diagnostic"] for v in failed.values()),
        "checks_total": sum(v["required"] + v["diagnostic"] for v in total.values()),
        "unbucketed_names": dict(unbucketed_names),
        "bucket_names": {k: dict(sorted(bucket_names[k].items()))
                         for k, *_ in BUCKETS},
        "semantic": {
            "n": n_sem, "fail_closed": n_sem_failclosed,
            "fail_closed_pct": round(100.0 * n_sem_failclosed / n_sem, 1) if n_sem else 0.0,
        },
        "ok_false": n_ok_false,
        "waits": wait_profile(tasks),
        "gating": {
            "n_fail": len(fails),
            "grounding_only": len(grounding_only),
            "grounding_only_ids": grounding_only[:12],
            "content_involved": len(content_involved),
            "no_required_check_failed": len(no_req_fail),
            "no_required_check_failed_ids": no_req_fail[:12],
        },
        "scenarios": scenarios,
        "examples": {k: examples.get(k, []) for k, *_ in BUCKETS},
    }


def head_to_head(cell):
    """Per-scenario win/loss/tie between every ordered pair of models.

    Compared on the scenario *mean score* over its 10 rounds, not on pass
    counts: pass is a threshold on the same quantity and throws away the
    margin, and with only 10 rounds per scenario, ties on pass count are common
    enough to swamp the table.
    """
    keys = [k for k in MODEL_KEYS if k in cell]
    out = {}
    for a in keys:
        row = {}
        for b in keys:
            if a == b:
                continue
            w = l = t = 0
            for s, va in cell[a]["scenarios"].items():
                vb = cell[b]["scenarios"].get(s)
                if vb is None:
                    continue
                if va["mean_score"] > vb["mean_score"]:
                    w += 1
                elif va["mean_score"] < vb["mean_score"]:
                    l += 1
                else:
                    t += 1
            row[b] = {"win": w, "loss": l, "tie": t}
        out[a] = row
    return out


def rank_block(cell):
    """Rank each model 1..N on each RANKED metric; rank 1 is the better end.

    Competition ranking: equal values share the better rank, so a two-way tie
    for best yields 1, 1, 3.
    """
    keys = [k for k in MODEL_KEYS if k in cell]
    out = {k: {} for k in keys}
    for metric, higher_better in RANKED:
        vals = sorted(((cell[k][metric], k) for k in keys),
                      key=lambda kv: (-kv[0] if higher_better else kv[0], kv[1]))
        prev, prev_rank = None, 0
        for i, (v, k) in enumerate(vals, start=1):
            rank = prev_rank if v == prev else i
            out[k][metric] = rank
            prev, prev_rank = v, rank
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get(
        "CLASHARNESS_DATA_ROOT",
        str(pathlib.Path(__file__).resolve().parents[4] / "data")),
        help="corpus root holding eval_results/")
    ap.add_argument("--effort", action="append", choices=EFFORTS, help="repeatable")
    ap.add_argument("--model", action="append", help="short key; repeatable")
    ap.add_argument("--examples", type=int, default=4)
    ap.add_argument("--json", default=str(pathlib.Path(__file__).resolve().parent.parent
                                          / "failure_taxonomy.json"))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    efforts = args.effort or EFFORTS
    wanted = set(args.model) if args.model else set(MODEL_KEYS)
    out = {
        "efforts": efforts,
        "models": {k: d for d, _, k in MODELS},
        # `models` above is the full roster; it is NOT the roster of any one cell.
        # The FIXED_TREE models appear under `primary_effort` only -- check for the
        # key rather than assuming every model is in every cell.
        "fixed_tree": FIXED_TREE,
        "primary_effort": PRIMARY_EFFORT,
        "buckets": [{"key": k, "label": lab, "fid": fid} for k, lab, fid in BUCKETS],
        "cells": {},
    }
    problems = []

    for eff in efforts:
        cell = {}
        for disp, sub, key in MODELS:
            if key not in wanted:
                continue
            # Not swept: emit once, in the primary cell. See FIXED_TREE.
            if key in FIXED_TREE and eff != PRIMARY_EFFORT:
                continue
            tasks, rows = load_tasks(args.root, sub, eff, key)
            a = analyze(tasks, rows, args.examples)
            a["display"] = disp
            cell[key] = a
            if a["unbucketed_names"]:
                problems.append(f"{eff}/{key}: unbucketed check names "
                                f"{sorted(a['unbucketed_names'])}")
            if a["dupe_rows"]:
                problems.append(f"{eff}/{key}: {a['rows']} rows for {a['n']} tasks")
            if a["ok_false"]:
                problems.append(f"{eff}/{key}: {a['ok_false']} checks with ok=False")
        ranks = rank_block(cell)
        for k in cell:
            cell[k]["rank"] = ranks[k]
        out["cells"][eff] = {"models": cell, "head_to_head": head_to_head(cell)}

        if args.quiet:
            continue
        for key in [k for k in MODEL_KEYS if k in cell]:
            a = cell[key]
            print(f"\n===== [{eff}] {a['display']}  ({key}) =====")
            print(f"  pass {a['pass_pct']}% ({a['passed']}/{a['n']})  avg_score {a['avg_score']}  "
                  f"near_miss {a['near_miss']}  zero_score {a['zero_score']}")
            print(f"  steps p50 {a['step_p50']} mean {a['step_mean']}  capped {a['capped']} "
                  f"({a['loopy_capped']} loopy)  zero-open {a['zero_open_tasks']}  "
                  f"zero-write {a['zero_write_tasks']}")
            w = a["waits"]
            print(f"  waits {w['n']} calls, {w['sub_30']} sub-30s, mean {w['mean']}s, "
                  f"median {w['median']}s, max {w['max']}s")
            print(f"  semantic {a['semantic']['n']} of which fail-closed "
                  f"{a['semantic']['fail_closed']} ({a['semantic']['fail_closed_pct']}%)")
            print(f"  failed checks total {a['failed_total']} of {a['checks_total']}")
            for k, label, fid in BUCKETS:
                f, t = a["failed_checks"][k], a["total_checks"][k]
                tt = t["required"] + t["diagnostic"]
                if not tt:
                    continue
                print(f"    {fid:3s} {label:52s} {f['required'] + f['diagnostic']:5d} failed "
                      f"({f['required']} req / {f['diagnostic']} diag) of {tt}")
            g = a["gating"]
            print(f"  of {g['n_fail']} failed tasks: grounding-only {g['grounding_only']}, "
                  f"content-involved {g['content_involved']}, no required check failed "
                  f"{g['no_required_check_failed']}")

    with open(args.json, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(f"\nwrote {args.json}")

    if problems:
        print("\n!! STRUCTURAL PROBLEMS", file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
