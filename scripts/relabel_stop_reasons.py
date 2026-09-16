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

r"""Offline re-label of eval ``stop_reason`` values using the fixed logic.

Recomputes ``run_episode``'s ``stop_reason`` for an *existing* eval run WITHOUT
re-running the model, so we can see the corrected failure taxonomy after the
``_required_effect_checks`` role-filter fix.

Background: ``assets_accessed`` checks are tagged ``role="diagnostic"`` by the
conflict-VLM materializer (observability only -- they never gate ``success``).
The old ``_required_effect_checks`` ignored ``role`` and so
``_missing_required_side_effect_stop_reason`` reported unopened diagnostic assets
as ``missing_required_side_effect: assets_accessed``, inflating it into the
apparent #1 failure bucket and masking the genuine required-check failures.

This script re-derives the label for each recorded episode exactly the way
``run_episode`` does, but through the *fixed* helpers, and prints a before/after
taxonomy. It is safe by construction: any record whose reason the fix cannot
possibly touch (success runs, ``reached max_steps=*``, and any reason that is not
``missing_required_side_effect: assets_accessed``) is asserted to recompute to
its original value -- if that invariant ever breaks, the script fails loudly
instead of silently rewriting history.

Dry-run by default (report only). Pass ``--write`` to persist the corrected
``stop_reason`` back into the ``.jsonl`` files.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# ``examples`` is an implicit namespace package (no __init__.py); make the
# package root importable, mirroring tests/test_train_and_eval_cli.py.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import examples.train_and_eval as te  # pylint: disable=g-import-not-at-top,wrong-import-position
from clasharness.core.task import load_task  # pylint: disable=g-import-not-at-top,wrong-import-position

_ASSET_REASON_PREFIX = "missing_required_side_effect: assets_accessed"
_DONE_FALLBACK = "agent returned DONE before completing required action"


def _bucket(stop_reason: str) -> str:
  """Collapse parametrized reasons into stable taxonomy buckets."""
  if not stop_reason:
    return "(empty / success)"
  if stop_reason.startswith("reached max_steps"):
    return "reached max_steps"
  if stop_reason.startswith("missing_required_side_effect:"):
    return "missing_required_side_effect: " + stop_reason.split(":", 1)[1].strip()
  if stop_reason.startswith("missing_required_report:"):
    return "missing_required_report"
  if stop_reason.startswith("missing_required_action_log:"):
    return "missing_required_action_log"
  return stop_reason


def _actions_from_record(rec: dict) -> list[str]:
  """Reproduce ``run_episode``'s ``executed_actions`` list.

  ``run_episode`` appends the raw ``action`` string (not ``executed_action``)
  for every executed step and never records the terminal ``DONE`` step, so the
  ordered ``action`` fields of the trajectory reconstruct it exactly.
  """
  return [str(step.get("action", "")) for step in rec.get("trajectory", []) or []]


def _eval_result_from_record(rec: dict) -> SimpleNamespace:
  """Rebuild the ``EvaluationResult`` view that ``run_episode`` passes along.

  ``metadata.check_results`` is a 1:1 serialization of ``eval_result.details``
  (name/passed/score/weight/message/role, including diagnostic checks), and
  ``_required_submission_stop_reason`` only reads ``detail.passed`` and
  ``detail.name``.
  """
  details = [
      SimpleNamespace(
          name=str(c.get("name", "")),
          passed=bool(c.get("passed")),
          message=str(c.get("message", "") or ""),
      )
      for c in (rec.get("metadata", {}) or {}).get("check_results", []) or []
  ]
  return SimpleNamespace(
      success=bool(rec.get("success")),
      score=float(rec.get("score") or 0.0),
      details=details,
  )


def relabel_record(rec: dict, task) -> tuple[str, str, bool]:
  """Return ``(orig, new, consistent)`` for a single episode record.

  ``consistent`` is only meaningful for records the fix must not change; for
  those it is ``True`` iff the recomputation matches the original (i.e. the fix
  provably left them alone).
  """
  orig = rec.get("stop_reason") or ""

  # Success runs never carry a stop_reason.
  if rec.get("success"):
    return orig, orig, True
  # max_steps wins in run_episode's precedence and is untouched by the fix.
  if orig.startswith("reached max_steps"):
    return orig, orig, True

  actions = _actions_from_record(rec)
  eval_result = _eval_result_from_record(rec)
  recomputed = te._required_submission_stop_reason(task, actions, eval_result)
  new = recomputed or _DONE_FALLBACK

  if not orig.startswith(_ASSET_REASON_PREFIX):
    # The fix can only drop diagnostic assets_accessed reasons; every other
    # failure reason must recompute to itself. Verify, then keep verbatim.
    return orig, orig, (new == orig)
  return orig, new, True


def _load_records(results_dir: Path) -> list[tuple[Path, list[dict]]]:
  out = []
  for name in sorted(os.listdir(results_dir)):
    if not name.endswith(".jsonl"):
      continue
    path = results_dir / name
    records = []
    with open(path) as f:
      for line in f:
        line = line.strip()
        if line:
          records.append(json.loads(line))
    out.append((path, records))
  return out


def _print_taxonomy(title: str, counter: collections.Counter, total: int) -> None:
  print(f"\n=== {title} ===")
  for key, count in counter.most_common():
    print(f"  {count:4d}  {100 * count / total:5.1f}%  {key}")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--results_dir",
      required=True,
      help="Directory containing the per-scenario *.jsonl eval outputs.",
  )
  parser.add_argument(
      "--task_data_dir",
      required=True,
      help="Task data dir (contains tasks/<id>/specs.json) for load_task.",
  )
  parser.add_argument(
      "--write",
      action="store_true",
      help="Persist corrected stop_reason back into the .jsonl files.",
  )
  args = parser.parse_args()

  results_dir = Path(args.results_dir)
  data_dir = Path(args.task_data_dir)
  files = _load_records(results_dir)

  before = collections.Counter()
  after = collections.Counter()
  total = 0
  changed = 0
  inconsistencies: list[tuple[str, str, str]] = []
  transitions = collections.Counter()
  load_failures: list[str] = []
  task_cache: dict[str, object] = {}
  fallback_hits = 0

  for path, records in files:
    dirty = False
    for rec in records:
      total += 1
      task_id = rec.get("task_id", "")
      if task_id not in task_cache:
        try:
          task_cache[task_id] = load_task(task_id, data_dir=data_dir)
        except Exception as exc:  # pylint: disable=broad-except
          task_cache[task_id] = None
          load_failures.append(f"{task_id}: {exc}")
      task = task_cache[task_id]

      orig = rec.get("stop_reason") or ""
      before[_bucket(orig)] += 1

      if task is None:
        after[_bucket(orig)] += 1
        continue

      orig2, new, consistent = relabel_record(rec, task)
      after[_bucket(new)] += 1
      if not consistent:
        inconsistencies.append((task_id, orig2, new))
      if new != orig2:
        changed += 1
        transitions[(_bucket(orig2), _bucket(new))] += 1
        if new == _DONE_FALLBACK:
          fallback_hits += 1
        if args.write:
          rec["stop_reason"] = new
          dirty = True

    if args.write and dirty:
      with open(path, "w") as f:
        for rec in records:
          f.write(json.dumps(rec, ensure_ascii=False) + "\n")

  print(f"Scanned {total} records across {len(files)} files.")
  if load_failures:
    print(f"\n!! {len(load_failures)} task load failures (kept original label):")
    for msg in load_failures[:10]:
      print(f"   - {msg}")

  _print_taxonomy("BEFORE (recorded stop_reason)", before, total)
  _print_taxonomy("AFTER  (fixed role-aware logic)", after, total)

  print(f"\n=== {changed} records re-labeled ===")
  for (src, dst), count in transitions.most_common():
    print(f"  {count:4d}  {src!r}  ->  {dst!r}")
  if fallback_hits:
    print(f"  ({fallback_hits} fell back to the DONE literal)")

  if inconsistencies:
    print(
        f"\n!! CORRECTNESS GATE FAILED: {len(inconsistencies)} records the fix"
        " should not touch changed under recomputation:"
    )
    for task_id, orig, new in inconsistencies[:20]:
      print(f"   - {task_id}: {orig!r} -> {new!r}")
    raise SystemExit(1)
  print(
      "\nCorrectness gate passed: every non-assets_accessed failure recomputed"
      " to its original label (the fix moved only assets_accessed records)."
  )
  if args.write:
    print("\nWrote corrected stop_reason values back to the .jsonl files.")
  else:
    print("\nDry run (report only). Re-run with --write to persist changes.")


if __name__ == "__main__":
  main()
