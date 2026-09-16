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

"""Gold-replay audit: replay each task's own gold trajectory and report taint.

A task is "tainted" (unfair) if its own ground-truth ``solution_commands`` fail
its own ``evaluation_checks`` when replayed through ``CLASHarnessEnv`` with the same
backend used by the materializer (``backend="multi"``). No agent can ever pass a
tainted task, so taint measures the unfairness we are trying to eliminate.

This is a read-only diagnostic: it never mutates task data. Run it before and
after the checker/generation fixes and diff the JSON reports to measure the
tainted-count delta.

Usage (remember to `source` the venv and gemini_env.sh so the LLM judge that
backs semantic effect checks is configured):

  python scripts/gold_replay_audit.py \
      --data-dir /tmp/clasharness-conflict-vlm-data \
      --split total \
      --out /tmp/gold_replay_report.json \
      [--limit N] [--task-id SINGLE_ID]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

# Make the repo root importable when run as a plain script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clasharness.core.environment import CLASHarnessEnv  # noqa: E402
from clasharness.core.task import load_task, load_task_ids  # noqa: E402

# Trajectory terminators recognized by the materializer's own replay loop.
STOP_TOKENS = {"done", "exit", "quit"}


def list_task_ids(data_dir: Path, split: str | None) -> list[str]:
    """Return task IDs for the split, falling back to the tasks/ directory."""
    if split:
        ids = load_task_ids(split, data_dir=data_dir)
        if ids:
            return ids
    tasks_dir = data_dir / "tasks"
    if not tasks_dir.is_dir():
        return []
    return sorted(
        p.name for p in tasks_dir.iterdir() if (p / "specs.json").exists()
    )


def replay_task(task_id: str, data_dir: Path) -> dict:
    """Replay a single task's gold trajectory and return a result record."""
    rec: dict = {
        "task_id": task_id,
        "status": None,  # "pass" | "fail" | "error"
        "n_commands": 0,
        "n_checks": 0,
        "failed_checks": [],
        "error": None,
    }
    try:
        task = load_task(task_id, data_dir=data_dir)
        gt = task.ground_truth
        sol = list(gt.solution_commands or []) if gt else []
        rec["n_commands"] = len(sol)
        rec["n_checks"] = len(gt.evaluation_checks or []) if gt else 0
        if not sol:
            rec["status"] = "error"
            rec["error"] = "no solution_commands in ground_truth"
            return rec

        # Give replay enough head-room so a long gold trajectory is never
        # truncated by the default max_steps.
        max_steps = max(50, len(sol) + 10)
        with CLASHarnessEnv(
            task_id=task_id,
            task_data_dir=data_dir,
            backend="multi",
            max_steps=max_steps,
        ) as env:
            env.reset()
            for cmd in sol:
                if cmd.strip().lower() in STOP_TOKENS:
                    break
                env.step(cmd)
            result = env.evaluate()

        if result.success:
            rec["status"] = "pass"
        else:
            rec["status"] = "fail"
            rec["failed_checks"] = [
                {
                    "name": c.name,
                    "weight": c.weight,
                    "message": (c.message or "")[:240],
                }
                for c in result.details
                if not c.passed
            ]
            # An all-checks-zero-weight / no-effective-checks evaluator also
            # reports success=False; surface that so it is not mistaken for a
            # real check failure.
            if not result.details:
                rec["error"] = "evaluate() returned success=False with no check details"
                rec["metadata"] = dict(result.metadata or {})
    except Exception as exc:  # noqa: BLE001 - audit must survive any single task
        rec["status"] = "error"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()[-2000:]
    return rec


def failure_category(check_name: str) -> str:
    """Coarse-grain a check name for aggregate reporting.

    Check names look like ``effect:messages_sent``, ``file:contains``,
    ``command`` etc. We keep the leading token(s) and drop any per-instance
    suffix in brackets/parens so counts group sensibly.
    """
    name = (check_name or "unknown").strip()
    for sep in ("[", "(", "#"):
        idx = name.find(sep)
        if idx != -1:
            name = name[:idx]
    return name.strip() or "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="/tmp/clasharness-conflict-vlm-data")
    parser.add_argument("--split", default="total")
    parser.add_argument("--limit", type=int, default=0, help="cap #tasks (smoke test)")
    parser.add_argument("--task-id", default=None, help="audit a single task id")
    parser.add_argument("--out", default="/tmp/gold_replay_report.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.task_id:
        task_ids = [args.task_id]
    else:
        task_ids = list_task_ids(data_dir, args.split)
        if args.limit and args.limit > 0:
            task_ids = task_ids[: args.limit]

    if not task_ids:
        print(f"No tasks found under {data_dir} (split={args.split}).")
        return 1

    print(f"Auditing {len(task_ids)} task(s) from {data_dir} (split={args.split})\n")
    t0 = time.time()
    records: list[dict] = []
    cat_counter: Counter[str] = Counter()
    name_counter: Counter[str] = Counter()
    n_pass = n_fail = n_error = 0

    for i, tid in enumerate(task_ids, 1):
        rec = replay_task(tid, data_dir)
        records.append(rec)
        status = rec["status"]
        if status == "pass":
            n_pass += 1
        elif status == "fail":
            n_fail += 1
            for fc in rec["failed_checks"]:
                name_counter[fc["name"]] += 1
                cat_counter[failure_category(fc["name"])] += 1
        else:
            n_error += 1

        flag = {"pass": "PASS", "fail": "FAIL", "error": "ERR "}[status]
        detail = ""
        if status == "fail":
            detail = "  -> " + ", ".join(fc["name"] for fc in rec["failed_checks"])
        elif status == "error":
            detail = f"  -> {rec['error']}"
        print(f"[{i}/{len(task_ids)}] {flag} {tid}{detail}")

    tainted = [r for r in records if r["status"] in ("fail", "error")]
    summary = {
        "data_dir": str(data_dir),
        "split": args.split,
        "n_tasks": len(task_ids),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_error": n_error,
        "n_tainted": len(tainted),
        "pass_rate": round(n_pass / len(task_ids), 4),
        "failure_categories": dict(cat_counter.most_common()),
        "failed_check_names": dict(name_counter.most_common()),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    report = {
        "summary": summary,
        "tainted_task_ids": [r["task_id"] for r in tainted],
        "records": records,
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(report, indent=2))

    print("\n===== GOLD-REPLAY AUDIT SUMMARY =====")
    for key, val in summary.items():
        print(f"  {key}: {val}")
    print(f"  report -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
