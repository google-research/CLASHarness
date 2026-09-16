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

"""Drop thin (<k) ensembles from a judge's verdict file so they are re-judged.

The sweep runs at 80-150 concurrent workers, well above the concurrency of the
original evaluation, so provider throttling occasionally kills one or two of a
pair's three votes. ``judge_majority`` still returns a verdict from the
survivors, and phase1 faithfully reproduces that, but a 1- or 2-vote verdict is
thinner than the 3-vote verdict it is being compared against. That thinness is
an artefact of how this ablation was executed, not a property of the judge
under test, and on a partial sweep it was worth up to 4.7pp of leaderboard
movement -- far too wide to simply disclose.

Deleting those lines is enough to fix it: phase1 resumes from the verdict file,
so anything removed here is re-judged on the next pass. Run this once the other
sweeps have finished and there is no contention left to lose votes to.

Usage:
    python3 repair_thin.py --judge sonnet5 [--min-votes 3] [--apply]
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import replay_core as rc  # noqa: E402

VERDICT_DIR = os.path.join(rc.WORK, "rejudge", "verdicts")


def restore_missing(path: str, backup: str) -> int:
    """Put back any verdict that the re-judge failed to refill.

    Dropping a thin verdict is only an improvement if a fuller one replaces it.
    Where the re-judge came back empty the pair would otherwise go from a
    2-vote verdict to NO verdict, which is strictly worse: phase 2 would fall
    back to the incumbent's answer and quietly bias the comparison. Restoring
    the old line keeps the thin verdict, which the ensemble-health bound
    already accounts for.
    """
    have = set()
    for line in io.open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                have.add(json.loads(line)["pair_id"])
            except Exception:  # noqa: BLE001
                pass
    restored = []
    for line in io.open(backup, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            pid = json.loads(line)["pair_id"]
        except Exception:  # noqa: BLE001
            continue
        if pid not in have:
            restored.append(line)
            have.add(pid)
    if restored:
        with io.open(path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(restored) + "\n")
    print(f"restored {len(restored):,} verdicts the re-judge did not refill")
    return len(restored)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True)
    ap.add_argument("--min-votes", type=int, default=3)
    ap.add_argument("--apply", action="store_true",
                    help="without this the script only reports")
    ap.add_argument("--restore-from", default="",
                    help="backup file to re-merge after a repair sweep")
    args = ap.parse_args()

    path = os.path.join(VERDICT_DIR, f"{args.judge}.jsonl")
    if not os.path.exists(path):
        print(f"no verdict file at {path}")
        return 1

    if args.restore_from:
        return 0 if restore_missing(path, args.restore_from) >= 0 else 1

    keep, drop, torn = [], 0, 0
    flipped_risk = 0
    for line in io.open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            torn += 1
            continue
        if r.get("ok") and int(r.get("n_ok", 0)) < args.min_votes:
            drop += 1
            # A 1-vote verdict is the most fragile: nothing corroborates it.
            if int(r.get("n_ok", 0)) <= 1:
                flipped_risk += 1
        else:
            keep.append(line)

    total = len(keep) + drop + torn
    print(f"{args.judge}: {total:,} lines  keep {len(keep):,}  "
          f"thin(<{args.min_votes}) {drop:,}  torn {torn:,}  "
          f"single-vote {flipped_risk:,}")
    if not args.apply:
        print("dry run; pass --apply to rewrite, then re-run the sweep")
        return 0
    if not drop and not torn:
        print("nothing to do")
        return 0

    stamp = time.strftime("%Y%m%d-%H%M%S")
    shutil.copy2(path, f"{path}.bak-thin-{stamp}")
    io.open(path, "w", encoding="utf-8").write(
        "\n".join(keep) + ("\n" if keep else ""))
    print(f"rewrote {path} (backup .bak-thin-{stamp}); "
          f"re-run the sweep to refill {drop + torn:,} pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
