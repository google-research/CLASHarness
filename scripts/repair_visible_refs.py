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

"""Re-derive ``visible_refs`` on already-shipped corpora after the A1b fix.

WHY THIS EXISTS
---------------
``_effective_available_from_round`` used to defer EVERY bare (non-timestamped)
asset that shared a base key with a later timestamped one. ``_asset_base_key``
strips both ``_YYYYMMDD_HHMMSS`` and ``_r<N>``, so unrelated visuals collapsed
into one group and an asset could be pushed past the very round whose answer it
held. Gold is authored from ``resolved_value``, so gold still wrote the answer
and the gold-satisfiability gate stayed silent; only the solver was blocked.

Across the v8 and sonnet-5 corpora that produced 27 rounds where the graded
value was readable from no surface the round exposed -- confirmed by 0/162
solver successes on them against 19.1% on every other round.

The generator is fixed, but the corpora already on disk carry the bad
``visible_refs`` baked into ``scenario.json``. Every asset involved is already
in the manifest and already rendered, so nothing needs regenerating: the repair
is a pure re-derivation of a list of ids.

SAFETY
------
* Dry-run by default; ``--apply`` is required to write.
* ``_effective_available_from_round`` is IMPORTED, never reimplemented, so this
  tool cannot drift from what the generator would now produce.
* ADDITIVE ONLY, and deliberately so -- see below.
* Writes are atomic (tmp + rename) and reuse the pipeline's exact serialization
  (``indent=2, ensure_ascii=False`` plus trailing newline).

WHY ADDITIVE ONLY (a deliberate divergence from the generator)
--------------------------------------------------------------
The old rule returned ``min(sibling_rounds)`` unconditionally, which could also
pull an asset EARLIER than its declared round -- e.g. an asset declared for r9
was visible from r03. The fixed rule returns the declared round, so a faithful
re-derivation would both add and REMOVE ids.

Removing is right for fresh generation: ``_reconcile_asset_checks`` runs
afterwards and drops any check pointing at a now-invisible asset, so the round
stays consistent. It is NOT safe here. In a shipped corpus the checks are
already baked into ``scenario.json``; silently withdrawing an asset that an
existing check references would manufacture exactly the unwinnable round this
repair exists to remove. One real instance was found while building this tool:
``satellite_ground_station_handover_v01`` r01 references
``ops_console_asset_bg_01`` (declared r2) in its own checks.

So the repair takes the union with the existing list. The corpus ends up
slightly more permissive than a fresh generation would be -- some assets stay
visible a round or two early -- which costs nothing in fairness (a solver seeing
more can still solve the round) whereas removal can cost everything. The
``pipeline would also drop`` counter reports that divergence rather than hiding
it; regenerate the corpus if you need exact parity.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clasharness.scenario.conflict_vlm_materializer import (  # noqa: E402
    _effective_available_from_round,
)

# Mirrors the manifest payload fields the fix inspects. Wrapped as attributes
# because the pipeline works on dataclasses while the manifest holds dicts.
_ASSET_FIELDS = (
    "id", "available_from_round", "ocr_text", "caption", "description",
    "chart_rows", "render_code", "regions", "prompt", "content_text",
)


def _as_obj(asset: dict) -> SimpleNamespace:
  return SimpleNamespace(**{f: asset.get(f) for f in _ASSET_FIELDS})


def _round_index(round_id: str) -> int:
  try:
    return int("".join(ch for ch in str(round_id) if ch.isdigit()) or 1)
  except ValueError:
    return 1


def repair_scenario(scenario_dir: Path, apply: bool) -> dict:
  """Re-derive visible_refs for one scenario. Returns a summary dict."""
  scen_path = scenario_dir / "scenario.json"
  man_path = scenario_dir / "asset_manifest.json"
  out = {"name": scenario_dir.name, "status": "skipped", "rounds": 0,
         "added": 0, "would_drop": 0, "detail": []}
  if not (scen_path.is_file() and man_path.is_file()):
    return out

  assets = [a for a in (json.loads(man_path.read_text()).get("assets") or [])
            if a.get("id")]
  objs = [_as_obj(a) for a in assets]
  scen = json.loads(scen_path.read_text())

  changed = False
  for rnd in scen.get("rounds") or []:
    idx = _round_index(rnd.get("round_id"))
    want = [o.id for o in objs
            if _effective_available_from_round(o, objs) <= idx]
    have = list(rnd.get("visible_refs") or [])
    gained = [i for i in want if i not in set(have)]
    # Reported, never acted on: ids a fresh generation would no longer show.
    out["would_drop"] += len(set(have) - set(want))
    if gained:
      out["rounds"] += 1
      out["added"] += len(gained)
      out["detail"].append((rnd.get("round_id"), "added", sorted(gained)))
      rnd["visible_refs"] = have + gained
      changed = True

  if not changed:
    out["status"] = "clean"
    return out

  out["status"] = "repaired" if apply else "would-repair"
  if apply:
    tmp = scenario_dir / "scenario.json.repair.tmp"
    tmp.write_text(json.dumps(scen, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    tmp.replace(scen_path)
  return out


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("corpora", nargs="+", help="corpus directories to repair")
  ap.add_argument("--apply", action="store_true",
                  help="write changes (default: dry run)")
  ap.add_argument("--verbose", action="store_true",
                  help="list every added asset id")
  ap.add_argument("--revert-manifest", type=Path, default=None,
                  help="with --apply, write a JSON record of every added id. "
                       "Because the repair only ever adds, removing exactly "
                       "these ids restores the corpus byte-for-byte, which is "
                       "far cheaper than copying ~13GB of scenario.json.")
  args = ap.parse_args()

  grand: dict[str, int] = {}
  total_added = total_rounds = total_drop = 0
  revert: dict[str, dict[str, list[str]]] = {}
  for corpus in args.corpora:
    root = Path(corpus).expanduser()
    if not root.is_dir():
      print(f"!! not a directory: {root}")
      continue
    print(f"\n=== {root.name}  ({'APPLY' if args.apply else 'DRY RUN'})")
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
      r = repair_scenario(d, args.apply)
      grand[r["status"]] = grand.get(r["status"], 0) + 1
      total_added += r["added"]
      total_rounds += r["rounds"]
      total_drop += r["would_drop"]
      if r["rounds"]:
        print(f"  {r['status']:<13} {d.name}: "
              f"{r['rounds']} round(s), +{r['added']} asset ref(s)")
        revert[str(d)] = {rid: ids for rid, _kind, ids in r["detail"]}
        if args.verbose:
          for rid, kind, ids in r["detail"]:
            print(f"       {rid} {kind}: {', '.join(ids)}")

  print("\n--- summary ---")
  for k, v in sorted(grand.items()):
    if v:
      print(f"  {k:<20} {v}")
  print(f"  rounds changed       {total_rounds}")
  print(f"  asset refs added     {total_added}")
  print(f"  pipeline would also drop {total_drop} ref(s) -- NOT applied, see "
        f"module docstring")
  if args.revert_manifest and args.apply:
    args.revert_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.revert_manifest.write_text(
        json.dumps(revert, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"  revert manifest      {args.revert_manifest}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
