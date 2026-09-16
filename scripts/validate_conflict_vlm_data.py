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

"""Reference-solution replay validator for Conflict-VLM task data (F1 audit).

Independently re-checks the well-posedness invariant the F1 admission gate is
supposed to enforce: for every shipped task, replaying its own reference (expert)
trajectory must satisfy the task's checks, and a null (no-op) trajectory must
NOT. Any task that fails is "flagged" (self-inconsistent) or "non-discriminating"
(passes with no work) - both are fairness defects.

Usage:
  python3 scripts/validate_conflict_vlm_data.py \
      --task-data-dir /tmp/clasharness-conflict-vlm-data --split dev

Exit code is non-zero if any task is flagged, so it can gate CI / a run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the package importable when run directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from clasharness.core.environment import CLASHarnessEnv
from clasharness.core.task import load_task


def _replay_eval(task_id: str, data_dir: Path, replay: list[str] | None):
  with CLASHarnessEnv(task_id=task_id, task_data_dir=data_dir, backend="multi") as env:
    env.reset()
    for cmd in replay or []:
      if cmd.strip().lower() in {"done", "exit", "quit"}:
        break
      env.step(cmd)
    return env.evaluate()


def _load_task_ids(data_dir: Path, split: str) -> list[str]:
  ds = data_dir / "datasets" / f"{split}.txt"
  if not ds.is_file():
    raise SystemExit(f"Dataset index not found: {ds}")
  return [l.strip() for l in ds.read_text("utf-8").splitlines() if l.strip()]


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--task-data-dir", default="/tmp/clasharness-conflict-vlm-data")
  ap.add_argument("--split", default="dev")
  ap.add_argument("--min-grounded-checks", type=int, default=2)
  ap.add_argument("--limit", type=int, default=0, help="Validate only the first N (0=all).")
  ap.add_argument("--report", default="", help="Optional path to write a JSON report.")
  args = ap.parse_args()

  data_dir = Path(args.task_data_dir)
  task_ids = _load_task_ids(data_dir, args.split)
  if args.limit:
    task_ids = task_ids[: args.limit]

  total = len(task_ids)
  flagged: list[dict] = []      # gold solution fails its own checks
  non_disc: list[dict] = []     # null solution passes (vacuous)
  under_k: list[dict] = []      # < K grounded discriminating checks
  errored: list[dict] = []      # env / evaluator raised
  clean = 0

  print(f"Validating {total} task(s) in split '{args.split}' from {data_dir}\n")
  for idx, tid in enumerate(task_ids, 1):
    try:
      task = load_task(tid, data_dir=data_dir)
      gt = task.ground_truth
      checks = list(gt.evaluation_checks) if gt else []
      sol = list(gt.solution_commands) if gt else []
      res_e = _replay_eval(tid, data_dir, sol)
      res_n = _replay_eval(tid, data_dir, [])
      ed, nd = list(res_e.details), list(res_n.details)
    except Exception as e:  # noqa: BLE001
      errored.append({"task_id": tid, "error": str(e)})
      print(f"[{idx}/{total}] ERROR {tid}: {e}")
      continue

    gold_fail = [c.name for c in ed if not c.passed]
    if gold_fail:
      flagged.append({"task_id": tid, "failed_checks": gold_fail})
      print(f"[{idx}/{total}] FLAGGED {tid}: gold fails {gold_fail}")
      continue

    # gold passes everything; classify discrimination.
    if len(ed) == len(checks) == len(nd):
      grounded = sum(1 for i in range(len(checks)) if ed[i].passed and not nd[i].passed)
    else:
      grounded = len(checks) if not res_n.success else 0

    if res_n.success:
      non_disc.append({"task_id": tid, "note": "null solution passes"})
      print(f"[{idx}/{total}] NON-DISCRIMINATING {tid}: null passes")
    elif grounded < args.min_grounded_checks:
      under_k.append({"task_id": tid, "grounded": grounded})
      print(f"[{idx}/{total}] UNDER-K {tid}: {grounded} grounded < {args.min_grounded_checks}")
    else:
      clean += 1

  print("\n" + "=" * 60)
  print("  CONFLICT-VLM REFERENCE-REPLAY VALIDATION REPORT")
  print("=" * 60)
  print(f"  Total tasks:                 {total}")
  print(f"  Clean (gold pass, null fail, >=K grounded): {clean}")
  print(f"  FLAGGED (gold fails own checks):            {len(flagged)}")
  print(f"  NON-DISCRIMINATING (null passes):           {len(non_disc)}")
  print(f"  UNDER-K (<{args.min_grounded_checks} grounded checks):               {len(under_k)}")
  print(f"  ERRORED (env/evaluator raised):             {len(errored)}")
  print("=" * 60)

  if args.report:
    Path(args.report).write_text(
        json.dumps(
            {
                "total": total,
                "clean": clean,
                "flagged": flagged,
                "non_discriminating": non_disc,
                "under_k": under_k,
                "errored": errored,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote report to {args.report}")

  # Non-zero exit if any hard well-posedness defect remains.
  if flagged or non_disc or errored:
    sys.exit(1)


if __name__ == "__main__":
  main()
