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

"""Phase 1: re-judge the Phase 0 prompt dump under an alternate judge.

One process per judge, because the judge backend is selected through
process-global environment variables. Results append to
``rejudge/verdicts/<judge>.jsonl`` and the run resumes from whatever is already
there, so an interrupted sweep never re-pays for work it has done.

The voting protocol is a faithful reimplementation of
``llm_judge.judge_majority`` (k=3, temperature 0.0, per-vote ``cache_salt``,
strict majority over the votes that came back ``ok``). It is reimplemented
rather than called so the INDIVIDUAL votes can be recorded: the split rate is
the judge-health signal the paper reports, and ``judge_majority`` only returns
the aggregate.

One asymmetry is inherent and is recorded per row rather than hidden: newer
Claude models reject an explicit ``temperature``, so ``llm_judge`` omits it for
them (``llm_retry.anthropic_rejects_sampling_params``). The Gemini judge is
therefore sampled at temperature 0 and the Claude judges at their service
default. That is exactly what the harness would have done had it been pointed at
those judges, so the comparison is between realisable configurations.

Usage:
    python3 phase1_judge.py \
        --judge sonnet5 [--limit N] [--workers 16]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import threading
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import replay_core as rc  # noqa: E402

OUT = os.path.join(rc.WORK, "rejudge")
PAIRS_DIR = os.path.join(OUT, "pairs")
VERDICT_DIR = os.path.join(OUT, "verdicts")

# The default Vertex project/location for this workstation. Overridable so the
# script is not welded to one cloud project.
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT") or os.environ.get(
    "GOOGLE_CLOUD_PROJECT", "cloud-ai-agentic-coding"
)

# name -> (provider, model, vertex_location)
#
# ``gemini36`` is not an alternate judge: it is the CONTROL. Re-running the
# incumbent judge over the same prompts measures how much of any observed
# disagreement is simply provider nondeterminism (greedy decoding on hosted
# backends is not bit-deterministic) rather than a property of the judge. A
# Claude-vs-Gemini delta is only interpretable against that floor.
JUDGES: dict[str, tuple[str, str, str]] = {
    "gemini36": ("gemini", "gemini-3.6-flash", "global"),
    "sonnet5": ("vertex-anthropic", "claude-sonnet-5", "global"),
    "opus5": ("vertex-anthropic", "claude-opus-5", "global"),
}

K_VOTES = 3


def configure(judge: str) -> tuple[str, str]:
    provider, model, location = JUDGES[judge]
    os.environ["LLM_JUDGE_PROVIDER"] = provider
    os.environ["LLM_JUDGE_MODEL"] = model
    os.environ["VERTEX_PROJECT"] = VERTEX_PROJECT
    os.environ["VERTEX_LOCATION"] = location
    # Transient 429/5xx are retried inside llm_judge with the same classifier the
    # agent path uses; without this the default is a single shot and a quota blip
    # would be recorded as a content failure.
    os.environ.setdefault("LLM_JUDGE_REQUEST_RETRIES", "6")
    os.environ.setdefault("LLM_JUDGE_RETRY_BACKOFF_S", "2.0")
    return provider, model


def load_pairs(limit: int = 0, models: str = "") -> list[dict]:
    want = {x.strip() for x in models.split(",") if x.strip()} if models else None
    pairs: list[dict] = []
    for path in sorted(glob.glob(os.path.join(PAIRS_DIR, "*.jsonl"))):
        slug = os.path.basename(path)[:-6]
        if want and slug not in want:
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))
    pairs.sort(key=lambda p: p["pair_id"])
    if limit:
        pairs = pairs[:limit]
    return pairs


def done_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    seen = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(json.loads(line)["pair_id"])
            except Exception:  # noqa: BLE001 - a torn final line is just redone
                continue
    return seen


def vote(pair: dict) -> dict:
    """Run the k=3 majority protocol on one pair and return all votes."""
    from clasharness.evaluation import llm_judge

    votes = []
    for i in range(K_VOTES):
        r = llm_judge.judge(
            pair["content"], pair["criterion"],
            model=None,          # spec model is always None; env decides
            temperature=0.0,
            cache_salt=f"vote:{i}",
        )
        votes.append({"passed": bool(r.passed), "score": float(r.score),
                      "ok": bool(r.ok), "error": (r.error or "")[:200]})

    ok = [v for v in votes if v["ok"]]
    if not ok:
        return {"pair_id": pair["pair_id"], "slug": pair["slug"],
                "task_id": pair["task_id"], "check_index": pair["check_index"],
                "ok": False, "passed": None, "score": 0.0,
                "n_ok": 0, "n_passed": 0, "votes": votes,
                "error": votes[-1]["error"]}

    n_passed = sum(1 for v in ok if v["passed"])
    passed = n_passed * 2 > len(ok)          # strict majority, as in judge_majority
    return {"pair_id": pair["pair_id"], "slug": pair["slug"],
            "task_id": pair["task_id"], "check_index": pair["check_index"],
            "ok": True, "passed": passed,
            "score": sum(v["score"] for v in ok) / len(ok),
            "n_ok": len(ok), "n_passed": n_passed, "votes": votes, "error": ""}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True, choices=sorted(JUDGES))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--models", default="")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    provider, model = configure(args.judge)
    os.makedirs(VERDICT_DIR, exist_ok=True)
    out_path = os.path.join(
        VERDICT_DIR, f"{args.judge}{args.out_suffix}.jsonl")

    pairs = load_pairs(args.limit, args.models)
    already = done_ids(out_path)
    todo = [p for p in pairs if p["pair_id"] not in already]
    print(f"judge={args.judge} provider={provider} model={model} "
          f"location={os.environ['VERTEX_LOCATION']}")
    print(f"pairs={len(pairs):,}  done={len(already):,}  todo={len(todo):,}  "
          f"calls={len(todo)*K_VOTES:,}")
    if not todo:
        print("nothing to do")
        return 0

    lock = threading.Lock()
    stats = Counter()
    t0 = time.time()
    fh = open(out_path, "a", encoding="utf-8")
    ffh = open(out_path.replace(".jsonl", ".failed.jsonl"), "a", encoding="utf-8")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(vote, p) for p in todo]
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                rec = fut.result()
            except Exception as exc:  # noqa: BLE001
                stats["crash"] += 1
                continue
            with lock:
                if rec["ok"]:
                    # Only successful verdicts go to the resume file. A pair that
                    # failed every vote (quota, transport) must stay un-done, or
                    # a transient outage would be frozen into the dataset as a
                    # permanent no-verdict on the next resume.
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    stats["done"] += 1
                    if rec["n_ok"] and rec["n_passed"] not in (0, rec["n_ok"]):
                        stats["split"] += 1
                    if rec["n_ok"] < K_VOTES:
                        stats["partial_votes"] += 1
                else:
                    ffh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    stats["judge_error"] += 1
                if n % 500 == 0:
                    fh.flush()
                    el = time.time() - t0
                    rate = n / el
                    eta = (len(todo) - n) / rate / 60 if rate else 0
                    print(f"  {n:,}/{len(todo):,}  {rate*K_VOTES:.1f} calls/s  "
                          f"eta {eta:.0f}m  split={stats['split']}  "
                          f"err={stats['judge_error']}  partial={stats['partial_votes']}  "
                          f"crash={stats['crash']}",
                          flush=True)
    fh.close()
    ffh.close()
    el = time.time() - t0
    print(f"\ndone {stats['done']:,} pairs in {el/60:.1f}m  "
          f"split={stats['split']:,}  judge_error={stats['judge_error']:,}  "
          f"partial_votes={stats['partial_votes']:,}  crash={stats['crash']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
