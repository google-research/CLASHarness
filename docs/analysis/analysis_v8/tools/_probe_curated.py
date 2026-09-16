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

"""Rank all 60 scenarios by how behaviorally revealing they are, for CURATED_SCENARIOS.

A scenario is "revealing" when the nine models DISAGREE on it: the metric is the
mean, over that scenario's ten rounds, of (max score - min score) across the nine
models in that round. A round where everyone scores the same teaches nothing; a
round that splits the field is what the paired-timeline walkthrough is for.

Rounds missing any model are skipped (there are none in v8, but the guard keeps the
ranking apples-to-apples if a cell is ever incomplete). The printed shortlist caps
each scenario FAMILY at one variant, so eight slots buy eight different stories
rather than the same conflict told twice.

Usage: python3 tools/_probe_curated.py [--root REPO] [--effort medium]
"""
import argparse, glob, json, os, re, sys, importlib.util
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "b", os.path.join(HERE, "build_behavior_report.py"))
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.environ.get(
    "CLASHARNESS_DATA_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(HERE)))), "data")),
    help="corpus root holding eval_results/")
ap.add_argument("--effort", default=b.PRIMARY_EFFORT)
ap.add_argument("--top", type=int, default=8)
a = ap.parse_args()

# task_id -> model key -> score
sc = defaultdict(dict)
for name, sub, key in b.MODELS:
    d = os.path.join(a.root, "eval_results", b.tree(sub, a.effort), "r01")
    for fp in glob.glob(os.path.join(d, "*.jsonl")):
        if fp.endswith(".interactions.jsonl"):
            continue
        for line in open(fp):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tid = r.get("task_id")
            if tid:
                sc[tid][key] = r.get("score") or 0.0

keys = [k for _, _, k in b.MODELS]
spreads = defaultdict(list)
for tid, row in sc.items():
    if any(k not in row for k in keys):
        continue
    v = [row[k] for k in keys]
    spreads[b.scenario_of(tid)].append(max(v) - min(v))

rank = sorted(((sum(v) / len(v), s, len(v)) for s, v in spreads.items()),
              reverse=True)
print(f"effort={a.effort}  scenarios={len(rank)}  models={len(keys)}\n")
print(f'{"#":>3} {"mean spread":>11} {"rounds":>6}  scenario')
for i, (sp, s, n) in enumerate(rank, 1):
    print(f"{i:3d} {sp:11.3f} {n:6d}  {s}")

fam = lambda s: re.sub(r"_v\d+$", "", s)
seen, pick = set(), []
for sp, s, n in rank:
    if fam(s) in seen:
        continue
    seen.add(fam(s)); pick.append((sp, s))
    if len(pick) == a.top:
        break
print(f"\n--- CURATED_SCENARIOS (top {a.top}, one variant per family) ---")
print("CURATED_SCENARIOS = [")
for sp, s in pick:
    print(f'    "{s}",{" " * max(1, 42 - len(s))}# spread {sp:.3f}')
print("]")
