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

"""Which rounds were PROVABLY UNWINNABLE when the shipped eval runs were scored?

Background. A visibility defect in the materializer collapsed unrelated assets
into one base key and pushed answer-bearing assets past the round that needed
them, so some rounds graded on a value that appeared in nothing they showed the
solver. Nine solvers went 0/162 on those rounds while scoring 19.1% on the rest
of the same corpus. The pipeline is fixed and the two shipped corpora have been
repaired in place -- but every eval result collected BEFORE the repair was
scored against the broken corpus, and those rows measure the defect, not the
model.

This tool answers "which (scenario, round) pairs were unwinnable at the time
those runs were scored", so the affected rows can be dropped from a re-score.

HOW IT AVOIDS GUESSING
  * The verdict comes from ``fairness_unwinnable_rows`` in the generator --
    the SAME function the shipping gate uses. A second implementation of
    "was this round winnable" would be free to drift, and the two would then
    disagree about which eval rows to drop, silently.
  * The pre-repair state is reconstructed from the repair's own revert
    manifest (round -> asset ids ADDED), applied to the parsed scenario in
    memory. The repair was purely additive, so removing those ids reproduces
    the exact bytes the runs were scored against. Nothing is retyped from a
    previous session's console output.
  * The CURRENT corpus is audited in the same pass. The pre-repair list is only
    meaningful if the post-repair corpus is actually clean, and asserting that
    at the same moment, with the same code, is the only way to know.

Read-only. It never writes into the corpus or the eval results.

  python3 scripts/dead_round_exclusions.py \
      --corpus conflict_vlm_v8_3.6_flash=~/Code/.../conflict_vlm_v8_3.6_flash \
      --revert_manifest ~/backups/.../revert_manifest.json \
      --out /tmp/dead_rounds.json \
      --eval_root conflict_vlm_v8_3.6_flash=~/Code/.../data_3_6_flash_v8
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _load_generator():
  """Import the generator module by path.

  ``scripts/`` is not a package and the module name has no import path, so the
  reuse this tool depends on has to be spelled out. Importing it (rather than
  copying two functions out of it) is the entire point: see the module
  docstring.
  """
  spec = importlib.util.spec_from_file_location(
      "_clasharness_generator", _HERE / "generate_conflict_vlm_scenarios.py")
  if spec is None or spec.loader is None:  # pragma: no cover - defensive
    raise SystemExit("could not import generate_conflict_vlm_scenarios.py")
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def _named_path(text):
  """``name=path`` -> ``(name, Path)``."""
  if "=" not in text:
    raise argparse.ArgumentTypeError(
        f"expected NAME=PATH, got {text!r}")
  name, _, raw = text.partition("=")
  return name, Path(raw).expanduser().resolve()


def _scenario_dirs(corpus_dir):
  """Finished scenario packages under a corpus, newest-first is irrelevant.

  ``.checkpoints`` is a sibling of the scenarios and holds half-written specs;
  auditing it would mix an abandoned corpus into the answer.
  """
  return sorted(
      d for d in corpus_dir.iterdir()
      if d.is_dir() and not d.name.startswith(".")
      and (d / "scenario.json").is_file())


def _revert_visible_refs(scen, added_by_round):
  """Undo the repair for one scenario, in memory. Returns rounds touched.

  The repair only ever APPENDED ids, so set-subtraction is its exact inverse.
  That is asserted rather than assumed: if an id the manifest claims to have
  added is not present now, the manifest and the corpus have diverged and every
  number downstream is suspect.
  """
  touched = 0
  for rnd in scen.get("rounds") or []:
    added = added_by_round.get(rnd.get("round_id"))
    if not added:
      continue
    refs = list(rnd.get("visible_refs") or [])
    missing = [a for a in added if a not in refs]
    if missing:
      raise SystemExit(
          f"revert manifest disagrees with the corpus: round "
          f"{rnd.get('round_id')} does not contain {missing}; refusing to "
          f"produce an exclusion list from a stale manifest")
    drop = set(added)
    rnd["visible_refs"] = [r for r in refs if r not in drop]
    touched += 1
  return touched


def scan_corpus(gen, corpus_name, corpus_dir, manifest, verbose=True):
  """Audit every scenario twice: as it ships today, and as it shipped before."""
  dirs = _scenario_dirs(corpus_dir)
  out = {
      "corpus_dir": str(corpus_dir),
      "scenarios_scanned": len(dirs),
      "scenarios_reverted": 0,
      "rounds_reverted": 0,
      "pre_repair_unwinnable": [],
      "still_unwinnable": [],
  }
  for i, d in enumerate(dirs, 1):
    spec_path = d / f"{d.name}_spec.json"
    manifest_path = d / "asset_manifest.json"
    if not (spec_path.is_file() and manifest_path.is_file()):
      raise SystemExit(
          f"{d} is missing {spec_path.name} or asset_manifest.json; a finished "
          f"package always carries both, so this corpus is not auditable")
    spec = json.loads(spec_path.read_text())
    _by_round, exact = gen._fairness_exact_arm(spec)
    if not exact:
      continue
    assets = {a["id"]: a
              for a in (json.loads(manifest_path.read_text()).get("assets")
                        or []) if a.get("id")}
    scen = json.loads((d / "scenario.json").read_text())

    # Today's corpus first, before anything is mutated.
    for rid, value, tier, holders in gen.fairness_unwinnable_rows(
        scen, assets, exact, json):
      out["still_unwinnable"].append({
          "scenario_slug": d.name, "round_id": rid, "value": value,
          "tier": tier, "holders": holders})

    added_by_round = manifest.get(str(d)) or manifest.get(str(d.resolve())) or {}
    if added_by_round:
      out["scenarios_reverted"] += 1
      out["rounds_reverted"] += _revert_visible_refs(scen, added_by_round)
      for rid, value, tier, holders in gen.fairness_unwinnable_rows(
          scen, assets, exact, json):
        out["pre_repair_unwinnable"].append({
            "scenario_slug": d.name, "round_id": rid, "value": value,
            "tier": tier, "holders": holders})
    if verbose and i % 10 == 0:
      print(f"  [{corpus_name}] {i}/{len(dirs)} scenarios", flush=True)
  return out


def _iter_eval_rows(eval_root, skipped):
  """Yield ``(solver, row)`` for every EPISODE under an eval tree.

  ``solver`` is the whole directory path below the root (setting / provider /
  model / repeat), so a re-score can never pool two settings, or two repeats of
  the same model, by accident.

  Rows carrying no ``metadata.scenario_slug`` are skipped. An eval tree also
  contains ``<model>/r01/log/*.jsonl`` -- step-level agent logs, thousands of
  lines per model, no episode outcome and no join key. Counting them would bury
  600 real episodes under 8,000 rows of 0% and make every rate meaningless.

  ``skipped`` (a caller-supplied counter dict) records how many rows were
  dropped that way. If the eval writer ever renames that field, EVERY row
  becomes a skip, every arm reports n=0, and the table would otherwise render
  that catastrophe as a tidy set of zeroes.
  """
  for path in sorted(eval_root.rglob("*.jsonl")):
    rel = path.relative_to(eval_root).parts
    if len(rel) < 2:
      continue
    solver = "/".join(rel[:-1])
    with path.open() as fh:
      for line in fh:
        line = line.strip()
        if not line:
          continue
        try:
          row = json.loads(line)
        except json.JSONDecodeError:
          skipped["unparseable"] = skipped.get("unparseable", 0) + 1
          continue
        if not (row.get("metadata") or {}).get("scenario_slug"):
          skipped["no_slug"] = skipped.get("no_slug", 0) + 1
          continue
        yield solver, row


def rescore(eval_root, excluded):
  """Per-solver success rate with and without the dead rounds.

  Reported per solver and never pooled. The dropped set is identical for every
  solver by construction (it is a property of the corpus, not of the model), so
  the two columns are a matched comparison -- and that identity is checked
  below rather than trusted.

  Returns ``(rows, skipped)``; see ``_iter_eval_rows`` for why the skip counts
  are reported rather than discarded.
  """
  per = {}
  skipped = {}
  for solver, row in _iter_eval_rows(eval_root, skipped):
    md = row.get("metadata") or {}
    key = (md.get("scenario_slug"), md.get("round_id"))
    bucket = per.setdefault(solver, {"n": 0, "ok": 0, "n_keep": 0,
                                     "ok_keep": 0, "dropped": set()})
    bucket["n"] += 1
    bucket["ok"] += 1 if row.get("success") else 0
    if key in excluded:
      bucket["dropped"].add((key, row.get("task_id")))
      continue
    bucket["n_keep"] += 1
    bucket["ok_keep"] += 1 if row.get("success") else 0
  rows = []
  for solver, b in sorted(per.items()):
    rows.append({
        "solver": solver,
        "n": b["n"],
        "rate": 100.0 * b["ok"] / b["n"] if b["n"] else 0.0,
        "n_excluded": len(b["dropped"]),
        "n_kept": b["n_keep"],
        "rate_kept": (100.0 * b["ok_keep"] / b["n_keep"]
                      if b["n_keep"] else 0.0),
    })
  return rows, skipped


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--corpus", action="append", type=_named_path, default=[],
                  metavar="NAME=PATH", required=True,
                  help="Corpus to audit. NAME labels it in the output.")
  ap.add_argument("--revert_manifest", type=Path, required=True,
                  help="The repair's revert manifest (scenario dir -> round -> "
                       "asset ids added).")
  ap.add_argument("--out", type=Path, default=None,
                  help="Write the exclusion list here as JSON.")
  ap.add_argument("--eval_root", action="append", type=_named_path, default=[],
                  metavar="NAME=PATH",
                  help="Re-score this eval tree against the corpus called NAME. "
                       "Repeatable.")
  args = ap.parse_args()

  gen = _load_generator()
  manifest = json.loads(args.revert_manifest.expanduser().read_text())
  print(f"revert manifest: {len(manifest)} scenario(s)\n")

  t0 = time.time()
  report = {
      "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
      "revert_manifest": str(args.revert_manifest),
      "what": "Rounds that were provably unwinnable BEFORE the visible_refs "
              "repair. Eval rows collected before the repair should be dropped "
              "for these (scenario_slug, round_id) pairs; they measure a "
              "corpus defect, not the model.",
      "corpora": {},
      "exclude": [],
  }
  for name, path in args.corpus:
    print(f"scanning {name} ...", flush=True)
    res = scan_corpus(gen, name, path, manifest)
    report["corpora"][name] = res
    for row in res["pre_repair_unwinnable"]:
      report["exclude"].append([name, row["scenario_slug"], row["round_id"]])
    print(f"  {res['scenarios_scanned']} scenario(s), "
          f"{res['scenarios_reverted']} reverted "
          f"({res['rounds_reverted']} round(s))")
    print(f"  PRE-REPAIR unwinnable : {len(res['pre_repair_unwinnable'])}")
    for row in res["pre_repair_unwinnable"]:
      print(f"    {row['scenario_slug']} {row['round_id']} "
            f"[{row['tier']}]: {row['value']!r}")
    # Printed unconditionally, including when zero: an absent line is not the
    # same reassurance as an explicit 0, and this is the number that says
    # whether the repair actually held.
    print(f"  STILL unwinnable today: {len(res['still_unwinnable'])}")
    for row in res["still_unwinnable"]:
      print(f"    !! {row['scenario_slug']} {row['round_id']} "
            f"[{row['tier']}]: {row['value']!r}")
    print()

  print(f"scan took {time.time() - t0:.0f}s")
  total = len(report["exclude"])
  still = sum(len(c["still_unwinnable"]) for c in report["corpora"].values())
  print(f"TOTAL dead rounds to exclude: {total}")
  print(f"TOTAL still unwinnable today: {still}"
        + ("   <-- the repair did NOT hold; fix before using these numbers"
           if still else ""))

  if args.out:
    args.out.expanduser().write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {args.out}")

  for name, path in args.eval_root:
    if name not in report["corpora"]:
      raise SystemExit(f"--eval_root {name} has no matching --corpus {name}")
    excluded = {(s, r) for n, s, r in report["exclude"] if n == name}
    print(f"\n{'=' * 78}\nre-score {name}: {path}")
    print(f"excluding {len(excluded)} (scenario, round) pair(s)\n")
    rows, skipped = rescore(path, excluded)
    print(f"{'solver':<44}{'n':>6}{'rate':>8}{'drop':>6}{'kept rate':>11}"
          f"{'delta':>8}")
    for r in rows:
      print(f"{r['solver']:<44}{r['n']:>6}{r['rate']:>7.1f}%"
            f"{r['n_excluded']:>6}{r['rate_kept']:>10.1f}%"
            f"{r['rate_kept'] - r['rate']:>+8.1f}")
    if not rows:
      # Zero arms out of a tree that plainly contains jsonl files means the
      # join key moved, not that nothing ran. Say so instead of printing an
      # empty table under a confident header.
      raise SystemExit(
          f"no episode rows found under {path}: every row was skipped "
          f"({skipped}). metadata.scenario_slug is the join key -- if the eval "
          f"writer renamed it, this tool is silently scoring nothing.")
    print(f"\n  rows skipped (agent logs, no join key): "
          f"{skipped.get('no_slug', 0)}"
          + (f", unparseable: {skipped['unparseable']}"
             if skipped.get("unparseable") else ""))
    drops = {r["n_excluded"] for r in rows if r["n"]}
    if len(drops) > 1:
      # The dead rounds are a property of the corpus, so every solver that ran
      # the full corpus must lose the same rows. Unequal drops mean the arms
      # did not run the same task set, and the two columns are then not a
      # matched comparison at all.
      print(f"\n  ! solvers dropped DIFFERENT row counts {sorted(drops)}: the"
            f" arms did not all run the same tasks, so compare within an arm"
            f" only")


if __name__ == "__main__":
  sys.exit(main())
