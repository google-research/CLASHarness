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

"""Prompt-aware, PER-ASSET accounting of the required asset-open check.

Motivation. A batch of failing rounds fail on a *single* required check: an
`effect:assets_accessed field_contains` (the "you must open asset X" gate) while
every other required check passes. These are genuine instruction-following
failures: the prompt told the solver to open and verify a specific asset, and the
solver produced the deliverable WITHOUT opening it, so the `assets_accessed` gate
correctly fails. Not following the prompt is a real miss even when the deliverable
happens to be otherwise correct -- the decisive value may have been reachable from
other text, but the round still required opening and checking the asset.

This tool quantifies that population per model and, for transparency, records
whether the round instruction *explicitly named* each still-missing asset. No
round is waived and pass@budget is NOT adjusted: every asset-open-only miss is
counted as the genuine failure it is (adjusted == raw for every model). The
"prompt did not name the missing asset" count is retained only as a transparency
signal -- an instruction that never mentions the required asset is a scenario-
authoring nuance, not grounds for a score change.

  * asset-open-only miss: a currently-FAILING round whose ONLY failing required
    check(s) are asset-open misses ("everything else was done correctly").

Per-asset naming matches each missing asset id against the prompt:

  asset ids are structured `<source-locus>..._<content>_<format>_<timestamp>`,
  e.g. `inventory_manager_final_allocation_pie_20231027_092500`
       `review_dashboard_review_board_20241024_084500`.
  We drop the trailing numeric timestamp, drop the leading source-locus prefix
  (everything up to & INCLUDING the first locus word: portal/dashboard/console/
  manager/viewer/page/...), drop generic format nouns (board/chart/pie/ledger/
  matrix/timeline/log/...) and fillers (final/frozen/...). What remains is the
  asset's distinctive CONTENT head (e.g. "allocation", "review", "gate",
  "reservation", "pairing"). The asset is "named" iff any content-head token
  appears in the round instruction (case-insensitive).

  Dropping the leading source-locus prefix is what stops false matches through a
  role/filename echo -- e.g. `inventory_manager_final_allocation_pie` matched
  only on content head {allocation}, so the "inventory_reconciliation.md" report
  path and "Casey_Inventory_Controller" in the prompt do NOT count as naming it.

The analysis is read-only: the evaluator is NOT changed and nothing is
regenerated. Stdlib only. Feeds the required asset-open / grounding discussion in
eval-results-capability-analysis.md and cross-model-behavioral-comparison.md.

THINK corpus: all ten models, r01, eval_results/data_3_5_flash/aligned_setting/.

Usage:
    python3 docs/analysis_think_history/tools/asset_open_rescore.py               # all
    python3 docs/analysis_think_history/tools/asset_open_rescore.py --model opus5
    python3 docs/analysis_think_history/tools/asset_open_rescore.py --list-flipped
    python3 docs/analysis_think_history/tools/asset_open_rescore.py --list-stay-failed
    python3 docs/analysis_think_history/tools/asset_open_rescore.py --explain
    python3 docs/analysis_think_history/tools/asset_open_rescore.py --root REPO
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

# (display name, eval_results subdir, short key) -- mirrors design_axis_slice.py
# and build_behavior_report.MODELS; keep the three in sync.
MODELS = [
    ("gemini-3.5-flash", "data_3_5_flash/aligned_setting/Gemini/3_5_flash", "flash"),
    ("gemini-3.6-flash", "data_3_5_flash/aligned_setting/Gemini/3_6_flash", "flash36"),
    ("gemini-3.1-pro", "data_3_5_flash/aligned_setting/Gemini/3_1_pro", "pro"),
    ("claude-opus-4.8", "data_3_5_flash/aligned_setting/Claude/opus_4_8", "opus"),
    ("claude-opus-5", "data_3_5_flash/aligned_setting/Claude/opus_5", "opus5"),
    ("claude-sonnet-5", "data_3_5_flash/aligned_setting/Claude/sonnet_5", "sonnet"),
    ("gpt-5.5", "data_3_5_flash/aligned_setting/OpenAI/5_5", "gpt55"),
    ("gpt-5.6-sol", "data_3_5_flash/aligned_setting/OpenAI/5_6_sol", "sol"),
    ("gpt-5.6-luna", "data_3_5_flash/aligned_setting/OpenAI/5_6_luna", "luna"),
    ("gpt-5.6-terra", "data_3_5_flash/aligned_setting/OpenAI/5_6_terra", "terra"),
]

# Run directory holding the trajectory *.jsonl. In this corpus every model has an
# r01; the four OpenAI models also have an r02, which is deliberately NOT read
# here (see build_behavior_report.RUN_SUBDIR for why r01 is the corpus-wide pick).
TASK_SUBDIR = "r01"
ASSET_OPEN_CHECK = "effect:assets_accessed field_contains"

# Leading source-locus words: the asset id prefix identifies WHERE the asset lives
# (a tool/surface/role), not WHAT it shows. Everything up to & including the first
# of these is dropped before content matching.
LOCUS = {
    "portal", "dashboard", "console", "page", "viewer", "camera", "docs", "doc",
    "manager", "terminal", "feed", "hub",
}
# Generic format/medium nouns -- the prompt may use them, but they don't identify
# the specific asset on their own.
FORMAT = {
    "board", "chart", "log", "photo", "image", "matrix", "pie", "ledger",
    "timeline", "graph", "snapshot", "screenshot", "view", "diagram", "figure",
    "report", "runbook", "status", "panel", "console", "dashboard", "viewer",
    "page", "feed",
}
FILLER = {"final", "confirmed", "frozen", "updated", "new", "current", "latest"}

# Asset id contains a required-asset content_contains reference in the message,
# e.g. "assets_accessed.id contains 'ops_console_reservation_board_...': False".
_ID_RE = re.compile(r"contains '([^']+)'")


def _content_head(asset_id: str) -> list[str]:
    """Distinctive content tokens of an asset id (see module docstring)."""
    parts = [p for p in asset_id.split("_") if not re.fullmatch(r"\d+", p)]
    cut = 0
    for i, tok in enumerate(parts):
        if tok.lower() in LOCUS:
            cut = i + 1
            break  # first locus word only -- the source prefix is the leading run
    rest = parts[cut:] if cut < len(parts) else parts
    return [t for t in rest if t.lower() not in FORMAT and t.lower() not in FILLER]


def asset_named_in_prompt(asset_id: str, instruction: str) -> bool:
    """True iff the instruction names this specific asset's content head."""
    text = (instruction or "").lower()
    head = _content_head(asset_id)
    return bool(head) and any(tok.lower() in text for tok in head)


def _load_rows(root: str, subdir: str) -> list[dict]:
    """Rows for one model, deduplicated by task_id (last wins).

    The dedup is not cosmetic: sol and terra each have 604 raw lines for 600
    tasks because a handful of tasks were re-attempted and the retry is appended.
    Counting lines would report sol at 194/604 = 32.1% where every other artifact
    says 194/600 = 32.3%. Last-occurrence-wins is what parallel_run.summary does
    and what build_behavior_report.load_model does, so all three agree.
    """
    by_id: dict[str, dict] = {}
    base = os.path.join(root, "eval_results", subdir, TASK_SUBDIR)
    for path in sorted(glob.glob(os.path.join(base, "*.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    by_id[r.get("task_id")] = r
    return list(by_id.values())


def _instruction(row: dict) -> str:
    md = row.get("metadata", {}) or {}
    return md.get("instruction") or md.get("canonical_instruction") or ""


def _missing_required_asset_ids(row: dict) -> list[str]:
    out = []
    for c in (row.get("metadata", {}).get("check_results", []) or []):
        if (c.get("name") == ASSET_OPEN_CHECK
                and c.get("role") == "required"
                and not c.get("passed")):
            m = _ID_RE.search(c.get("message", "") or "")
            if m:
                out.append(m.group(1))
    return out


def rescore_model(root: str, subdir: str) -> dict:
    rows = _load_rows(root, subdir)
    total = len(rows)
    raw_pass = sum(1 for r in rows if r.get("success"))

    only_asset_miss = []  # set B: fail, asset-open is the ONLY failing required check
    for r in rows:
        if r.get("success"):
            continue
        req = [c for c in (r.get("metadata", {}).get("check_results", []) or [])
               if c.get("role") == "required"]
        failing = [c for c in req if not c.get("passed")]
        if not failing:
            continue
        ao = [c for c in failing if c.get("name") == ASSET_OPEN_CHECK]
        other = [c for c in failing if c.get("name") != ASSET_OPEN_CHECK]
        if ao and not other:
            only_asset_miss.append(r)

    flipped, stay_failed = [], []
    for r in only_asset_miss:
        instr = _instruction(r)
        assets = _missing_required_asset_ids(r)
        named = [a for a in assets if asset_named_in_prompt(a, instr)]
        # Flip only if the prompt named NONE of the still-missing assets.
        (stay_failed if named else flipped).append(r)

    adj_pass = raw_pass + len(flipped)
    return {
        "total": total,
        "raw_pass": raw_pass,
        "only_asset_miss": only_asset_miss,
        "flipped": flipped,
        "stay_failed": stay_failed,
        "adj_pass": adj_pass,
    }


def _explain(rows: list[dict]) -> None:
    for r in rows:
        instr = _instruction(r)
        print(f"    {r['task_id']}  score={r.get('score'):.2f}")
        for a in _missing_required_asset_ids(r):
            head = _content_head(a)
            named = asset_named_in_prompt(a, instr)
            print(f"        {'NAMED  ' if named else 'UNNAMED'} {a}  head={head}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="repo root (default: cwd)")
    ap.add_argument("--model", help="restrict to one short key, e.g. "
                    "flash/flash36/pro/opus/opus5/sonnet/gpt55/sol/luna/terra")
    ap.add_argument("--list-stay-failed", action="store_true",
                    help="print the task_ids of the asset-open-only misses whose prompt named the asset")
    ap.add_argument("--list-flipped", action="store_true",
                    help="print the task_ids of the asset-open-only misses whose prompt did not name the asset")
    ap.add_argument("--explain", action="store_true",
                    help="print per-asset NAMED/UNNAMED trace for the asset-open-only-miss set")
    args = ap.parse_args()

    models = [m for m in MODELS if not args.model or m[2] == args.model]
    for disp, subdir, key in models:
        base = os.path.join(args.root, "eval_results", subdir, TASK_SUBDIR)
        if not os.path.isdir(base):
            print(f"\n== {disp} ({key}) ==  [no eval_results at {base}]")
            continue
        s = rescore_model(args.root, subdir)
        total, rp = s["total"], s["raw_pass"]
        named, unnamed = len(s["stay_failed"]), len(s["flipped"])
        print(f"\n== {disp} ({key}) ==")
        print(f"  rounds                         : {total}")
        print(f"  pass@budget                    : {rp:4d}  ({rp/total:.1%})  (no adjustment; asset-open misses count as failures)")
        print(f"  asset-open-only-miss failures  : {len(s['only_asset_miss'])}  (all genuine grounding fails)")
        print(f"    -> prompt named the asset    : {named}")
        print(f"    -> prompt did not name it    : {unnamed}  (transparency only; not waived)")
        if args.explain:
            print("  -- asset-open-only-miss set, per-asset trace:")
            _explain(s["only_asset_miss"])
        if args.list_stay_failed:
            for r in s["stay_failed"]:
                print(f"    NAMED    {r['task_id']}  score={r.get('score'):.2f}")
        if args.list_flipped:
            for r in s["flipped"]:
                print(f"    UNNAMED  {r['task_id']}  score={r.get('score'):.2f}")


if __name__ == "__main__":
    main()
