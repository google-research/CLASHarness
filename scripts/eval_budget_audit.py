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

"""Recover, cross-check and gate the *measurement conditions* of an eval cell.

Why this exists
---------------
A leaderboard row is only meaningful if every arm was measured under the same
step budget. The budget is chosen at launch (``--max-steps``) and is **not**
recorded anywhere inside the cell's own artifacts:

  * ``parallel_run.summary``          -- reports ``step_capped_tasks`` (how many
                                        episodes hit the cap) but never the cap.
  * ``log/*.interactions.jsonl``      -- its leading ``run_config`` record has
                                        ``reasoning_effort``/``_effective`` but
                                        no ``max_steps``.
  * episode rows                      -- ``stop_reason`` spells the cap out, but
                                        *only* for episodes that actually hit it.

The directory name is the usual stand-in (``r01`` == 20 steps, ``r02`` == 30 by
local convention), and it is exactly the wrong thing to trust: it is a label a
human typed, it survives a re-run with different flags, and a single mislabelled
cell silently turns a leaderboard into a comparison of two different tests.

So the budget is *reconstructed from evidence* and, when the evidence does not
determine it, this module refuses to guess. A board that cannot prove its arms
are comparable should fail loudly, not print a plausible table.

Evidence, in precedence order
-----------------------------
``stop_reason``
    ``"reached max_steps=30"`` -> budget 30. Exact and co-located with the data,
    but only observable if at least one episode was capped. A strong model can
    finish every episode early (``opus-5`` at medium: 0/135 capped), which is
    precisely when this source goes silent.
``launcher``
    ``run_effort_panel.sh`` writes the exact command line it ran to
    ``$LOG_DIR/launchers/<tag>_<level>_<slug>.sh``; it carries ``--max-steps``
    and ``--llm-reasoning-effort`` and is joined back to the cell by its own
    ``--out-dir``. Authoritative for panel-launched cells, absent for anything
    started by hand.

When both speak they must agree; disagreement is a hard error rather than a
precedence win, because it means one of the two artifacts is lying about the run
and there is no way to tell which.

Effort is handled separately: ``reasoning_effort_effective`` in ``run_config``
is the ground truth (it reflects any vendor-side clamp), while the launcher only
records what was *requested*. A gap between the two is a clamp -- reported, not
fatal, since the measurement is still internally consistent.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

_STOP_CAP_RE = re.compile(r"max_steps=(\d+)")
_LAUNCHER_FLAG_RE = {
    "budget": re.compile(r"--max-steps\s+(\S+)"),
    "effort": re.compile(r"--llm-reasoning-effort\s+(\S+)"),
    "out_dir": re.compile(r"--out-dir\s+(\S+)"),
}


class BudgetAuditError(RuntimeError):
  """Raised when measurement conditions are unknown or not comparable."""


@dataclasses.dataclass
class CellFacts:
  """Everything we could establish about how one cell was measured."""

  cell: str
  n_episodes: int = 0
  n_capped: int = 0
  max_steps_seen: int = 0
  budget: int | None = None
  budget_source: str = "unresolved"
  effort: str | None = None
  effort_requested: str | None = None
  effort_source: str = "unresolved"
  problems: list[str] = dataclasses.field(default_factory=list)

  @property
  def ok(self) -> bool:
    return not self.problems and self.budget is not None

  def label(self) -> str:
    b = self.budget if self.budget is not None else "?"
    e = self.effort or "?"
    return f"steps={b} effort={e}"


def _episode_rows(cell_dir: Path) -> Iterable[dict[str, Any]]:
  """Yield episode records from a cell.

  Only the cell's own top-level ``*.jsonl`` files are read. ``log/`` holds
  step-level agent transcripts -- thousands of lines with no episode outcome --
  and is deliberately not walked here; rows without a ``metadata.scenario_slug``
  are dropped for the same reason.
  """
  for path in sorted(cell_dir.glob("*.jsonl")):
    with path.open(errors="replace") as fh:
      for line in fh:
        line = line.strip()
        if not line:
          continue
        try:
          row = json.loads(line)
        except json.JSONDecodeError:
          continue
        if not (row.get("metadata") or {}).get("scenario_slug"):
          continue
        yield row


def _budget_from_stop_reasons(rows: Iterable[dict[str, Any]]) -> tuple[
    set[int], int, int, int]:
  """Return ``(budgets_seen, n_episodes, n_capped, max_steps_seen)``.

  ``budgets_seen`` holding more than one value means episodes inside a single
  directory were produced by runs with different caps -- an append-mode re-run
  with changed flags, say. That cell is not one measurement and must not be
  reported as one.
  """
  budgets: set[int] = set()
  n = capped = high_water = 0
  for row in rows:
    n += 1
    steps = row.get("steps")
    if isinstance(steps, int):
      high_water = max(high_water, steps)
    match = _STOP_CAP_RE.search(row.get("stop_reason") or "")
    if match:
      capped += 1
      budgets.add(int(match.group(1)))
  return budgets, n, capped, high_water


def _effort_from_run_configs(cell_dir: Path) -> tuple[set[str], set[str]]:
  """Return ``(effective_efforts, requested_efforts)`` from interaction logs.

  The leading record of every ``log/*.interactions.jsonl`` is a ``run_config``.
  Every episode in a cell must have been run with the same effort; a set with
  more than one member is a mixed cell.
  """
  effective: set[str] = set()
  requested: set[str] = set()
  for path in sorted((cell_dir / "log").glob("*.interactions.jsonl")):
    try:
      with path.open(errors="replace") as fh:
        head = fh.readline().strip()
      if not head:
        continue
      cfg = (json.loads(head) or {}).get("run_config") or {}
    except (json.JSONDecodeError, OSError):
      continue
    if cfg.get("reasoning_effort_effective"):
      effective.add(str(cfg["reasoning_effort_effective"]))
    if cfg.get("reasoning_effort"):
      requested.add(str(cfg["reasoning_effort"]))
  return effective, requested


def index_launchers(launcher_dirs: Iterable[Path]) -> dict[str, dict[str, str]]:
  """Map ``realpath(--out-dir) -> {budget, effort, launcher}``.

  Joining on the launcher's own ``--out-dir`` rather than on a filename
  convention is what makes this safe: a launcher can only ever claim the cell it
  actually wrote to.
  """
  index: dict[str, dict[str, str]] = {}
  for d in launcher_dirs:
    d = Path(os.path.expanduser(str(d)))
    if not d.is_dir():
      continue
    for path in sorted(d.glob("*.sh")):
      try:
        text = path.read_text(errors="replace")
      except OSError:
        continue
      out = _LAUNCHER_FLAG_RE["out_dir"].search(text)
      if not out:
        continue
      key = os.path.realpath(os.path.expanduser(out.group(1)))
      budget = _LAUNCHER_FLAG_RE["budget"].search(text)
      effort = _LAUNCHER_FLAG_RE["effort"].search(text)
      index[key] = {
          "budget": budget.group(1) if budget else "",
          "effort": effort.group(1) if effort else "",
          "launcher": str(path),
      }
  return index


def audit_cell(cell_dir: Path,
               launcher_index: dict[str, dict[str, str]] | None = None,
               ) -> CellFacts:
  """Reconstruct one cell's measurement conditions from evidence."""
  launcher_index = launcher_index or {}
  facts = CellFacts(cell=str(cell_dir))

  budgets, n, capped, high_water = _budget_from_stop_reasons(
      _episode_rows(cell_dir))
  facts.n_episodes, facts.n_capped, facts.max_steps_seen = n, capped, high_water

  if n == 0:
    facts.problems.append("no episode rows")

  if len(budgets) > 1:
    facts.problems.append(
        f"episodes report {len(budgets)} different caps {sorted(budgets)} -- "
        "this directory mixes runs and is not a single measurement")

  from_stop = next(iter(budgets)) if len(budgets) == 1 else None

  hit = launcher_index.get(os.path.realpath(str(cell_dir)))
  from_launcher = None
  if hit and hit["budget"].isdigit():
    from_launcher = int(hit["budget"])

  if from_stop is not None and from_launcher is not None:
    if from_stop != from_launcher:
      facts.problems.append(
          f"budget disagreement: stop_reason says {from_stop}, launcher "
          f"{hit['launcher']} says {from_launcher}")
    facts.budget, facts.budget_source = from_stop, "stop_reason+launcher"
  elif from_stop is not None:
    facts.budget, facts.budget_source = from_stop, "stop_reason"
  elif from_launcher is not None:
    facts.budget, facts.budget_source = from_launcher, "launcher"
  else:
    facts.problems.append(
        "budget unresolvable: no episode hit the cap "
        f"(max steps seen = {high_water}) and no launcher records --out-dir "
        "for this cell; the directory name is not evidence")

  # A step count above the cap means the cap was not what we think it was.
  if facts.budget is not None and high_water > facts.budget:
    facts.problems.append(
        f"an episode ran {high_water} steps under a claimed cap of "
        f"{facts.budget}")

  effective, requested = _effort_from_run_configs(cell_dir)
  if len(effective) > 1:
    facts.problems.append(
        f"cell mixes reasoning efforts {sorted(effective)}")
  elif effective:
    facts.effort, facts.effort_source = next(iter(effective)), "run_config"
  elif hit and hit["effort"]:
    facts.effort, facts.effort_source = hit["effort"], "launcher(requested)"
  if requested:
    facts.effort_requested = "/".join(sorted(requested))

  return facts


def find_cells(root: Path) -> list[Path]:
  """Every directory under ``root`` that holds episode ``*.jsonl`` output."""
  found = []
  for dirpath, _dirnames, filenames in os.walk(root):
    if os.path.basename(dirpath) == "log":
      continue
    if any(f.endswith(".jsonl") for f in filenames):
      found.append(Path(dirpath))
  return sorted(found)


def assert_comparable(facts: list[CellFacts], *, axis: str = "effort") -> None:
  """Raise unless these cells form one honest comparison.

  ``axis`` names the dimension the comparison is *about* and is therefore
  allowed to vary; everything else must be constant. Comparing models at a fixed
  budget uses ``axis="model"``; an effort sweep uses ``axis="effort"``. The step
  budget is never a free axis -- an arm given more steps is running a different
  test.
  """
  broken = [f for f in facts if not f.ok]
  if broken:
    detail = "\n".join(f"  {f.cell}\n    - " + "\n    - ".join(f.problems)
                       for f in broken)
    raise BudgetAuditError(
        f"{len(broken)} cell(s) have unknown or inconsistent measurement "
        f"conditions:\n{detail}")

  budgets = {f.budget for f in facts}
  if len(budgets) > 1:
    rows = "\n".join(f"  steps={f.budget:<4} {f.cell}"
                     for f in sorted(facts, key=lambda x: (x.budget or 0)))
    raise BudgetAuditError(
        f"mixed step budgets {sorted(budgets)} in one comparison -- these arms "
        f"were not given the same test:\n{rows}")

  if axis != "effort":
    efforts = {f.effort for f in facts}
    if len(efforts) > 1:
      rows = "\n".join(f"  effort={f.effort:<8} {f.cell}" for f in facts)
      raise BudgetAuditError(
          f"mixed reasoning efforts {sorted(efforts)} in one comparison:\n"
          f"{rows}")


def main() -> int:
  ap = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("roots", nargs="+", type=Path,
                  help="Eval tree(s) to audit. Every directory holding episode "
                       "*.jsonl output is treated as one cell.")
  ap.add_argument("--launchers", action="append", type=Path,
                  default=[Path("~/eval_logs/launchers")],
                  help="Directory of launcher scripts written by "
                       "run_effort_panel.sh. Repeatable.")
  ap.add_argument("--gate", action="store_true",
                  help="Exit non-zero unless every cell resolved cleanly AND "
                       "all cells share one step budget.")
  ap.add_argument("--axis", default="effort", choices=["effort", "model"],
                  help="Dimension the comparison is about (may vary). Only "
                       "meaningful with --gate.")
  ap.add_argument("--json", type=Path, default=None,
                  help="Write the audit as JSON here.")
  args = ap.parse_args()

  launcher_index = index_launchers(args.launchers)
  print(f"launcher index: {len(launcher_index)} cell(s) with a recorded "
        f"command line\n")

  facts: list[CellFacts] = []
  for root in args.roots:
    root = Path(os.path.expanduser(str(root)))
    for cell in find_cells(root):
      facts.append(audit_cell(cell, launcher_index))

  width = max((len(os.path.relpath(f.cell)) for f in facts), default=20)
  width = min(width, 78)
  print(f"{'cell':{width}} {'n':>5} {'capped':>6} {'steps':>6} "
        f"{'budget':>7} {'source':>19} {'effort':>8} {'src':>18}")
  for f in facts:
    name = os.path.relpath(f.cell)
    if len(name) > width:
      name = "..." + name[-(width - 3):]
    print(f"{name:{width}} {f.n_episodes:5d} {f.n_capped:6d} "
          f"{f.max_steps_seen:6d} {str(f.budget):>7} {f.budget_source:>19} "
          f"{str(f.effort):>8} {f.effort_source:>18}")

  flagged = [f for f in facts if f.problems]
  if flagged:
    print(f"\n{len(flagged)} cell(s) need attention:")
    for f in flagged:
      print(f"  {os.path.relpath(f.cell)}")
      for p in f.problems:
        print(f"    - {p}")

  clamped = [f for f in facts
             if f.effort and f.effort_requested
             and f.effort_requested != f.effort]
  if clamped:
    print("\nvendor clamped the requested effort (not fatal, but the arm is "
          "not running what was asked for):")
    for f in clamped:
      print(f"  {os.path.relpath(f.cell)}: requested "
            f"{f.effort_requested} -> effective {f.effort}")

  if args.json:
    out = Path(os.path.expanduser(str(args.json)))
    out.write_text(json.dumps(
        [dataclasses.asdict(f) for f in facts], indent=2) + "\n")
    print(f"\nwrote {out}")

  if args.gate:
    try:
      assert_comparable(facts, axis=args.axis)
    except BudgetAuditError as exc:
      print(f"\nGATE FAILED\n{exc}", file=sys.stderr)
      return 1
    print(f"\nGATE PASSED: {len(facts)} cell(s), one step budget "
          f"({facts[0].budget}), varying axis = {args.axis}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
