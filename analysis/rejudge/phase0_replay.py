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

"""Phase 0: replay every high-effort round, gate fidelity, dump judge prompts.

Costs zero provider calls. Produces three things per model slug:

  rejudge/pairs/<slug>.jsonl   one line per judge prompt the FINAL evaluation
                               would have issued: the (content, criterion) pair
                               plus the identity of the check it belongs to.
  rejudge/rounds/<slug>.jsonl  one line per round: the recorded check list with
                               role/weight/passed, plus the pair id attached to
                               each judge-originated check. Phase 2 recomputes
                               ``success`` from this without replaying again.
  rejudge/fidelity/<slug>.json fidelity of the replay against the recorded run,
                               measured ONLY on deterministic checks.

Why the gate is deterministic-only: semantic verdicts are stubbed here by
construction, so they say nothing about whether the replay reproduced the
episode. Everything else -- grounding, anchors, side effects, time windows,
the asset gate -- is a pure function of the terminal environment state, so a
mismatch there is direct evidence the replay diverged.

Usage:
    python3 phase0_replay.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import replay_core as rc  # noqa: E402

OUT = os.path.join(rc.WORK, "rejudge")


def _out(kind: str, slug: str, ext: str) -> str:
    d = os.path.join(OUT, kind)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{slug}.{ext}")


def run_model(slug: str, limit: int = 0) -> dict:
    capture = rc.Capture()
    capture.install()

    stats = Counter()
    tok = []  # content lengths in characters
    errors: list[dict] = []
    mismatches: list[dict] = []

    fp = open(_out("pairs", slug, "jsonl"), "w", encoding="utf-8")
    fr = open(_out("rounds", slug, "jsonl"), "w", encoding="utf-8")
    pair_seq = 0
    t0 = time.time()

    for pkg, idx, record in rc.iter_records(slug):
        if limit and stats["rounds"] >= limit:
            break
        stats["rounds"] += 1
        recorded = record.get("metadata", {}).get("check_results") or []
        out = rc.replay_round(record, capture)

        if out["status"] != "ok":
            stats["replay_error"] += 1
            errors.append({"task_id": out["task_id"], "error": out["error"]})
            continue

        details = out["details"]
        if len(details) != len(recorded):
            # Without a 1:1 alignment we cannot attribute a verdict to a check,
            # so the round is unusable rather than silently mis-scored.
            stats["arity_mismatch"] += 1
            errors.append({
                "task_id": out["task_id"],
                "error": f"arity {len(details)} replay vs {len(recorded)} recorded",
            })
            continue
        stats["aligned"] += 1

        by_index = {p["check_index"]: p for p in out["pairs"]}
        if -1 in by_index:
            # A pair captured outside any wrapped sub-evaluator means the index
            # binding failed; refuse to guess.
            stats["unbound_pair"] += 1
            errors.append({"task_id": out["task_id"], "error": "unbound judge pair"})
            continue

        checks = []
        for i, (rec, rep) in enumerate(zip(recorded, details)):
            semantic = rc.is_semantic(rec.get("message", ""))
            entry = {
                "i": i,
                "name": rec.get("name", ""),
                "role": rec.get("role", "required"),
                "weight": float(rec.get("weight", 1.0)),
                "recorded_passed": bool(rec.get("passed")),
                "recorded_score": float(rec.get("score", 0.0) or 0.0),
                "semantic": semantic,
                "judge_invoked": bool(rec.get("judge_invoked")) and semantic,
                "pair_id": None,
            }
            if semantic:
                stats["semantic"] += 1
                if rec.get("name") != rep.get("name"):
                    stats["name_drift"] += 1
                pair = by_index.get(i)
                if pair is not None:
                    stats["pairs"] += 1
                    tok.append(len(pair["content"]))
                    pid = f"{slug}:{pair_seq}"
                    pair_seq += 1
                    entry["pair_id"] = pid
                    fp.write(json.dumps({
                        "pair_id": pid, "slug": slug, "task_id": out["task_id"],
                        "check_index": i, "name": entry["name"],
                        "criterion": pair["criterion"], "content": pair["content"],
                        "spec_model": pair["model"],
                        "recorded_passed": entry["recorded_passed"],
                    }, ensure_ascii=False) + "\n")
                else:
                    # Fail-closed semantic check: the harness skipped the judge
                    # (empty transcript), so there is nothing to re-judge and the
                    # recorded verdict carries over unchanged into Phase 2.
                    stats["fail_closed"] += 1
            else:
                stats["deterministic"] += 1
                if bool(rec.get("passed")) == bool(rep.get("passed")):
                    stats["det_match"] += 1
                else:
                    stats["det_mismatch"] += 1
                    if len(mismatches) < 40:
                        mismatches.append({
                            "task_id": out["task_id"], "check": entry["name"],
                            "recorded": bool(rec.get("passed")),
                            "replay": bool(rep.get("passed")),
                            "recorded_msg": (rec.get("message") or "")[:200],
                            "replay_msg": (rep.get("message") or "")[:200],
                        })
            checks.append(entry)

        fr.write(json.dumps({
            "slug": slug, "task_id": out["task_id"], "package": pkg,
            "recorded_success": bool(record.get("success")),
            "recorded_round_score": float(
                (record.get("metadata", {}) or {}).get("score",
                                                       record.get("score", 0.0))
                or 0.0),
            "checks": checks,
        }, ensure_ascii=False) + "\n")

    fp.close()
    fr.close()

    det = stats["det_match"] + stats["det_mismatch"]
    summary = {
        "slug": slug,
        "elapsed_s": round(time.time() - t0, 1),
        "counts": dict(stats),
        "det_total": det,
        "det_fidelity": (stats["det_match"] / det) if det else None,
        "content_chars_mean": (sum(tok) / len(tok)) if tok else 0,
        "content_chars_p50": sorted(tok)[len(tok) // 2] if tok else 0,
        "content_chars_p95": sorted(tok)[int(len(tok) * 0.95)] if tok else 0,
        "content_chars_total": sum(tok),
        "criterion_pairs": stats["pairs"],
        "errors": errors[:40],
        "n_errors": len(errors),
        "mismatches": mismatches,
    }
    with open(_out("fidelity", slug, "json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    return summary


def _job(args):
    slug, limit = args
    try:
        return run_model(slug, limit)
    except Exception as exc:  # noqa: BLE001
        import traceback
        return {"slug": slug, "fatal": f"{type(exc).__name__}: {exc}",
                "tb": traceback.format_exc()[-1500:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="rounds per model")
    ap.add_argument("--models", default="", help="comma-separated slugs")
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()

    # ALL_CELLS = the 14-model panel plus auxiliary cells (currently only
    # gemini-3.7-flash r01, the discarded twin of the published r02). The aux
    # cells are replayed and re-judged but never enter the leaderboard.
    slugs = [s for s, *_ in rc.ALL_CELLS]
    if args.models:
        want = {x.strip() for x in args.models.split(",") if x.strip()}
        slugs = [s for s in slugs if s in want]

    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_job, (s, args.limit)): s for s in slugs}
        for fut in as_completed(futs):
            r = fut.result()
            results[r["slug"]] = r
            if "fatal" in r:
                print(f"[FATAL] {r['slug']}: {r['fatal']}")
            else:
                f = r["det_fidelity"]
                print(f"[done] {r['slug']:11s} {r['elapsed_s']:6.1f}s  "
                      f"rounds={r['counts'].get('rounds',0):4d}  "
                      f"det={r['det_total']:5d} fid={f if f is None else round(f*100,3)}  "
                      f"pairs={r['criterion_pairs']:5d}  err={r['n_errors']}")

    print(f"\n=== PHASE 0 SUMMARY ({time.time()-t0:.0f}s) ===")
    tot = Counter()
    worst = 1.0
    for s in slugs:
        r = results.get(s, {})
        if "fatal" in r or not r:
            print(f"{s:11s} FATAL"); worst = 0.0; continue
        for k, v in r["counts"].items():
            tot[k] += v
        tot["det_total"] += r["det_total"]
        tot["content_chars_total"] += r["content_chars_total"]
        if r["det_fidelity"] is not None:
            worst = min(worst, r["det_fidelity"])
    fid = tot["det_match"] / tot["det_total"] if tot["det_total"] else 0
    print(f"rounds            {tot['rounds']:,}")
    print(f"aligned           {tot['aligned']:,}")
    print(f"replay errors     {tot['replay_error']:,}  arity {tot['arity_mismatch']:,}")
    print(f"deterministic     {tot['det_total']:,}  match {tot['det_match']:,}  "
          f"mismatch {tot['det_mismatch']:,}")
    print(f"DET FIDELITY      {fid*100:.4f}%   (worst model {worst*100:.4f}%)")
    print(f"semantic checks   {tot['semantic']:,}")
    print(f"judge pairs       {tot['pairs']:,}  fail-closed {tot['fail_closed']:,}")
    print(f"content chars     {tot['content_chars_total']:,}")
    aux = [s_ for s_, *_ in rc.AUX if s_ in results and "fatal" not in results[s_]]
    if aux:
        a_rounds = sum(results[s_]["counts"].get("rounds", 0) for s_ in aux)
        a_pairs = sum(results[s_]["criterion_pairs"] for s_ in aux)
        print(f"  of which AUX    {a_rounds:,} rounds / {a_pairs:,} pairs "
              f"({', '.join(aux)}) -- excluded from the 14-model leaderboard")
    est_in = tot["content_chars_total"] / 4.0
    print(f"est input tokens  {est_in/1e6:.2f}M per single pass over the pairs")
    with open(os.path.join(OUT, "phase0_summary.json"), "w") as fh:
        json.dump({"per_model": results, "totals": dict(tot),
                   "det_fidelity": fid, "worst_model_fidelity": worst}, fh, indent=1)
    print(f"\nGATE: {'PASS' if fid >= 0.99 else 'FAIL'} (threshold 99%)")
    return 0 if fid >= 0.99 else 1


if __name__ == "__main__":
    raise SystemExit(main())
