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

"""Phase 2: recompute the leaderboard under each judge and compare.

Success is recomputed exactly the way the harness computes it
(``evaluation/evaluator.py``): a round succeeds iff it has at least one
``required`` check and every one of them passed. Only the verdicts of
judge-originated checks are substituted; deterministic checks, fail-closed
semantic checks (the judge was never called) and rounds the harness never
scored at all (provider failures) carry their recorded outcome through
unchanged.

The pipeline validates itself before it reports anything: replaying the
recorded verdicts through the same recomputation must reproduce each model's
recorded ``success`` flag round for round. If that identity does not hold, the
substitution is not measuring the judge, and the run aborts.

Usage:
    python3 phase2_recompute.py [--judges gemini36,sonnet5,opus5]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import replay_core as rc  # noqa: E402

OUT = os.path.join(rc.WORK, "rejudge")

# Display names, matching tab:main.
PRETTY = {
    "opus_5": "claude-opus-5", "5_6_sol": "gpt-5.6-sol",
    "opus_4_8": "claude-opus-4.8", "kimi_k3": "kimi-k3",
    "sonnet_5": "claude-sonnet-5", "3_7_flash": "gemini-3.7-flash",
    "5_6_terra": "gpt-5.6-terra", "5_6_luna": "gpt-5.6-luna",
    "v4_pro": "deepseek-v4-pro", "3_6_flash": "gemini-3.6-flash",
    "glm_5_2": "glm-5.2-fp8", "3_1_pro": "gemini-3.1-pro",
    "dsv4_flash": "deepseek-v4-flash", "3_5_flash": "gemini-3.5-flash",
}
FAMILY = {
    "opus_5": "Claude", "opus_4_8": "Claude", "sonnet_5": "Claude",
    "3_1_pro": "Gemini", "3_5_flash": "Gemini", "3_6_flash": "Gemini",
    "3_7_flash": "Gemini",
    "5_6_sol": "OpenAI", "5_6_terra": "OpenAI", "5_6_luna": "OpenAI",
    "kimi_k3": "Moonshot", "v4_pro": "DeepSeek", "dsv4_flash": "DeepSeek",
    "glm_5_2": "ZhipuAI",
}
ROUNDS_PER_MODEL = 600

# The judge that produced the recorded corpus. Every comparison is against this.
INCUMBENT = "gemini36"


def load_rounds() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for path in sorted(glob.glob(os.path.join(OUT, "rounds", "*.jsonl"))):
        slug = os.path.basename(path)[:-6]
        rows = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        out[slug] = rows
    return out


def load_verdicts(judge: str) -> dict[str, bool]:
    return {k: m["passed"] for k, m in load_verdicts_full(judge).items()}


def load_verdicts_full(judge: str) -> dict[str, dict]:
    """pair_id -> {passed, n_ok, n_passed}; n_ok < 3 means a thin ensemble."""
    path = os.path.join(OUT, "verdicts", f"{judge}.jsonl")
    v: dict[str, dict] = {}
    if not os.path.exists(path):
        return v
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if r.get("ok"):
                v[r["pair_id"]] = {"passed": bool(r["passed"]),
                                   "score": float(r.get("score", 0.0) or 0.0),
                                   "n_ok": int(r.get("n_ok", 0)),
                                   "n_passed": int(r.get("n_passed", 0))}
    return v


def ensemble_health(slugs, rounds, judges, rate, pretty) -> dict:
    """Split rate, thin-ensemble rate, and an adversarial bound on thin ones."""
    print("\n=== ensemble health of the substituted verdicts ===")
    out: dict[str, object] = {}
    for j in judges:
        full = load_verdicts_full(j)
        used = {p: m for p, m in full.items()
                if p.split(":")[0] in set(slugs) or True}
        if not used:
            continue
        n = len(used)
        thin = {p for p, m in used.items() if m["n_ok"] < 3}
        split = sum(1 for m in used.values()
                    if 0 < m["n_passed"] < m["n_ok"])
        # Adversarial: force every thin verdict to pass, then to fail.
        v_hi = {p: (True if p in thin else m["passed"]) for p, m in used.items()}
        v_lo = {p: (False if p in thin else m["passed"]) for p, m in used.items()}
        worst = 0.0
        for slug in slugs:
            rows = rounds.get(slug, [])
            if not rows:
                continue
            hi = 100.0 * sum(1 for r in rows
                             if round_success(r["checks"], v_hi)) / ROUNDS_PER_MODEL
            lo = 100.0 * sum(1 for r in rows
                             if round_success(r["checks"], v_lo)) / ROUNDS_PER_MODEL
            worst = max(worst, hi - lo)
        print(f"  {j:9s} n={n:,}  non-unanimous {split:,} ({100.0*split/n:.2f}%)  "
              f"thin (<3 votes) {len(thin):,} ({100.0*len(thin)/n:.2f}%)  "
              f"widest model swing if every thin verdict flips: {worst:.2f}pp")
        out[j] = {"n": n, "split": split, "thin": len(thin),
                  "thin_flip_max_swing_pp": worst}
    return out


def round_score(checks: list[dict], verdicts: dict[str, bool] | None) -> float:
    """Weighted mean over non-diagnostic checks (evaluator.py:118).

    Substituted checks are scored 1.0/0.0 rather than with the judge's graded
    score, because effect_checker.py:455 does exactly that -- all 28,009
    judge-invoked checks here are ``field_semantic_match`` and all 28,009
    recorded scores are binary. Carrying the graded score through instead
    inflates every model uniformly and measures the substitution, not the judge.
    """
    scoring = [c for c in checks if c["role"] != "diagnostic"]
    total_w = sum(c["weight"] for c in scoring)
    if not scoring or total_w <= 0:
        return 0.0
    acc = 0.0
    for c in scoring:
        pid = c.get("pair_id")
        if verdicts is not None and pid and pid in verdicts:
            sc = 1.0 if verdicts[pid] else 0.0
        else:
            sc = c["recorded_score"]
        acc += sc * c["weight"]
    return acc / total_w


def round_success(checks: list[dict], verdicts: dict[str, bool] | None) -> bool:
    """Apply the harness success rule to one round's check list."""
    required = [c for c in checks if c["role"] == "required"]
    if not required:
        return False
    for c in required:
        pid = c.get("pair_id")
        if verdicts is not None and pid and pid in verdicts:
            passed = verdicts[pid]
        else:
            passed = c["recorded_passed"]
        if not passed:
            return False
    return True


def spearman(a: list[float], b: list[float]) -> float:
    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((x - mb) ** 2 for x in rb))
    return num / (da * db) if da and db else float("nan")


def kappa(pairs: list[tuple[bool, bool]]) -> tuple[float, float]:
    """Cohen's kappa and raw agreement for a list of (a, b) binary verdicts."""
    n = len(pairs)
    if not n:
        return float("nan"), float("nan")
    agree = sum(1 for a, b in pairs if a == b) / n
    pa = sum(1 for a, _ in pairs if a) / n
    pb = sum(1 for _, b in pairs if b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    k = (agree - pe) / (1 - pe) if pe < 1 else float("nan")
    return k, agree


def round_success_forced(checks, verdicts, forced: bool) -> bool:
    """As round_success, but pairs MISSING from ``verdicts`` are forced.

    Used to bound the unscored-pair fallback rather than trusting it.
    """
    required = [c for c in checks if c["role"] == "required"]
    if not required:
        return False
    for c in required:
        pid = c.get("pair_id")
        if pid:
            passed = verdicts[pid] if pid in verdicts else forced
        else:
            passed = c["recorded_passed"]
        if not passed:
            return False
    return True


def bound_missing(rounds, slugs, verdicts, ready, rate, pretty) -> dict:
    """Report how far each model's rate can move on unscored pairs alone."""
    print("\n=== adversarial bound on pairs the substitute judge never scored ===")
    print("    (missing pairs forced all-pass / all-fail; the reported number "
          "uses the recorded verdict, which sits between them)")
    print("    NOTE: on the common subset this fallback set is IDENTICAL for "
          "every judge, so it cancels in any")
    print("    cross-judge difference -- it bounds each column's absolute "
          "level, not the judge effect itself.")
    out: dict[str, object] = {}
    for j in ready:
        v = verdicts[j]
        miss = sum(1 for slug in slugs for r in rounds.get(slug, [])
                   for c in r["checks"]
                   if c.get("pair_id") and c["pair_id"] not in v)
        if miss == 0:
            print(f"  {j:9s} complete: 0 unscored pairs, bound is exact")
            out[j] = {"missing": 0, "max_swing_pp": 0.0}
            continue
        worst = 0.0
        detail = {}
        for slug in slugs:
            rows = rounds.get(slug, [])
            hi = 100.0 * sum(1 for r in rows
                             if round_success_forced(r["checks"], v, True)) / ROUNDS_PER_MODEL
            lo = 100.0 * sum(1 for r in rows
                             if round_success_forced(r["checks"], v, False)) / ROUNDS_PER_MODEL
            detail[slug] = {"lo": lo, "hi": hi, "reported": rate[j][slug]}
            worst = max(worst, hi - lo)
        print(f"  {j:9s} {miss:,} unscored pairs; widest model interval "
              f"{worst:.2f}pp")
        for slug in slugs:
            d = detail[slug]
            if d["hi"] - d["lo"] >= 0.5:
                print(f"      {pretty[slug]:20s} [{d['lo']:.1f}, {d['hi']:.1f}] "
                      f"reported {d['reported']:.1f}")
        out[j] = {"missing": miss, "max_swing_pp": worst, "per_model": detail}
    return out


def mcnemar(pairs: list[tuple[bool, bool]]) -> tuple[int, int, float]:
    """Exact two-sided McNemar on paired binary outcomes -> (b, c, p)."""
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if y and not x)
    n = b + c
    if n == 0:
        return 0, 0, 1.0
    # Exact binomial tail at q=0.5, doubled and clipped.
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return b, c, min(1.0, 2.0 * tail)


# The published gemini-3.7-flash cell (r02) and its discarded twin (r01).
PUB_37, AUX_37 = "3_7_flash", "3_7_flash_r01"


def decompose_37(rounds, verdicts, ready) -> dict:
    """Split the gemini-3.7-flash r01 -> r02 delta into judge and rollout terms.

    analysis/corpus.py:131-140 records this cell as the corpus's one
    irreducible confound: r01 and r02 differ in BOTH the policy rollout and the
    grader (r01 was scored by gemini-3.6-flash, r02 by gemini-3.7-flash -- the
    model under test), so the +1.67pp gap "cannot be attributed to either".
    Re-judging both rollouts under one common judge removes the grader term and
    makes the attribution identifiable for the first time.

    With R(run, judge) the pass rate, the observed gap telescopes exactly:

        R(r02, g37_rec) - R(r01, g36_rec)
          = [R(r02, g37_rec) - R(r02, J)]   judge      (rollout held at r02)
          + [R(r02, J)      - R(r01, J)]    rollout    (judge held at J)
          + [R(r01, J)      - R(r01, g36_rec)]  incumbent self-replication

    The third term is the noise floor: for J = gemini36 it is the SAME judge
    model re-run on the SAME rollout, so whatever it is, it is sampling noise
    rather than a provenance effect, and it bounds how much of the other two
    terms we are entitled to believe.
    """
    pub, aux = rounds.get(PUB_37, []), rounds.get(AUX_37, [])
    if not pub or not aux:
        return {}

    def rate(rows, v):
        return 100.0 * sum(1 for r in rows if round_success(r["checks"], v)) / len(rows)

    def by_task(rows, v):
        return {r["task_id"]: round_success(r["checks"], v) for r in rows}

    r01_rec, r02_rec = rate(aux, None), rate(pub, None)
    observed = r02_rec - r01_rec

    print("\n=== gemini-3.7-flash: decomposing the r01 -> r02 confound ===")
    print(f"  recorded  r01 (judge gemini-3.6-flash) {r01_rec:6.2f}%   n={len(aux)}")
    print(f"  recorded  r02 (judge gemini-3.7-flash, SELF) {r02_rec:6.2f}%   n={len(pub)}")
    print(f"  observed gap {observed:+.2f}pp  <- confounded in the published table")

    out: dict[str, object] = {"r01_recorded": r01_rec, "r02_recorded": r02_rec,
                              "observed_gap": observed, "under": {}}
    for j in ready:
        v = verdicts[j]
        # Only report if this judge covered BOTH rollouts essentially fully.
        need = sum(1 for rows in (pub, aux) for r in rows
                   for c in r["checks"] if c.get("pair_id"))
        have = sum(1 for rows in (pub, aux) for r in rows for c in r["checks"]
                   if c.get("pair_id") and c["pair_id"] in v)
        if need and have / need < 0.999:
            print(f"  [{j}] skipped: covers {have}/{need} of this cell's pairs")
            continue
        r01_j, r02_j = rate(aux, v), rate(pub, v)
        judge_term = r02_rec - r02_j
        rollout_term = r02_j - r01_j
        noise_term = r01_j - r01_rec
        a, b = by_task(aux, v), by_task(pub, v)
        common = [(a[t], b[t]) for t in a if t in b]
        nb, nc, pv = mcnemar(common)
        print(f"  [common judge = {j}]")
        print(f"     r01 {r01_j:6.2f}%   r02 {r02_j:6.2f}%")
        print(f"     judge term   {judge_term:+6.2f}pp  (r02 rollout, "
              f"gemini-3.7-flash self-grade minus {j})")
        print(f"     rollout term {rollout_term:+6.2f}pp  (judge held at {j}; "
              f"paired McNemar b={nb} c={nc} p={pv:.3f}, n={len(common)})")
        print(f"     self-repl.   {noise_term:+6.2f}pp  (r01 rollout, {j} minus "
              f"the recorded verdicts)")
        print(f"     sum          {judge_term + rollout_term + noise_term:+6.2f}pp "
              f"(identity check vs observed {observed:+.2f}pp)")
        out["under"][j] = {  # type: ignore[index]
            "r01": r01_j, "r02": r02_j, "judge_term": judge_term,
            "rollout_term": rollout_term, "self_replication": noise_term,
            "mcnemar_b": nb, "mcnemar_c": nc, "mcnemar_p": pv, "n_paired": len(common),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", default="gemini36,sonnet5,opus5")
    ap.add_argument("--allow-partial", action="store_true",
                    help="report even if a judge has not finished every pair")
    ap.add_argument("--no-common", action="store_true",
                    help="compare on each judge's own coverage instead of the "
                         "intersection (confounds opinion with coverage)")
    args = ap.parse_args()
    judges = [j.strip() for j in args.judges.split(",") if j.strip()]

    rounds = load_rounds()
    slugs = [s for s, *_ in rc.PANEL]

    # ---- self-validation: recorded verdicts must reproduce recorded success --
    bad = 0
    total = 0
    for slug in slugs:
        for r in rounds.get(slug, []):
            total += 1
            if round_success(r["checks"], None) != r["recorded_success"]:
                bad += 1
    badsc = 0
    for slug in slugs:
        for r in rounds.get(slug, []):
            if abs(round_score(r["checks"], None)
                   - r.get("recorded_round_score", -1)) > 1e-6:
                badsc += 1
    print(f"self-check: recomputed success == recorded success on "
          f"{total - bad:,}/{total:,} rounds")
    print(f"self-check: recomputed score   == recorded score   on "
          f"{total - badsc:,}/{total:,} rounds")
    if bad or badsc:
        print(f"ABORT: {bad} success / {badsc} score rounds do not reproduce; "
              f"substitution is unsound.")
        return 1

    # ---- coverage ---------------------------------------------------------
    need = Counter()
    for slug in slugs:
        for r in rounds.get(slug, []):
            for c in r["checks"]:
                if c.get("pair_id"):
                    need[slug] += 1
    total_pairs = sum(need.values())

    verdicts = {j: load_verdicts(j) for j in judges}
    print("\njudge coverage:")
    ready = []
    panel_coverage: dict[str, dict[str, int]] = {}
    for j in judges:
        have = sum(1 for slug in slugs for r in rounds.get(slug, [])
                   for c in r["checks"]
                   if c.get("pair_id") and c["pair_id"] in verdicts[j])
        pct = 100.0 * have / total_pairs if total_pairs else 0
        refused = total_pairs - have
        panel_coverage[j] = {"scored": have, "total": total_pairs,
                             "declined": refused}
        print(f"  {j:9s} {have:,}/{total_pairs:,}  ({pct:.2f}%)"
              + (f"   {refused:,} unscored (provider refusals)" if refused else ""))
        if pct >= 80.0 or args.allow_partial:
            ready.append(j)
    if not ready:
        print("\nno judge is complete yet; nothing to report")
        return 0

    verdicts_full = {j: dict(verdicts[j]) for j in ready}
    if len(ready) > 1 and not args.no_common:
        common = set(verdicts[ready[0]])
        for j in ready[1:]:
            common &= set(verdicts[j])
        panel_pairs = {c["pair_id"] for slug in slugs
                       for r in rounds.get(slug, []) for c in r["checks"]
                       if c.get("pair_id")}
        kept = len(common & panel_pairs)
        print(f"\ncommon subset: {len(common):,} pairs scored by all "
              f"{len(ready)} judges ({kept:,}/{len(panel_pairs):,} panel pairs, "
              f"{100.0*kept/len(panel_pairs):.1f}%)")
        print("  all judges substitute on these and fall back to the recorded "
              "verdict elsewhere, so the columns differ only in opinion")
        for j in ready:
            verdicts[j] = {p: v for p, v in verdicts[j].items() if p in common}

    # ---- leaderboards -----------------------------------------------------
    #  "recorded" is the incumbent's ORIGINAL verdicts, i.e. the published table.
    settings = ["recorded"] + ready
    rate: dict[str, dict[str, float]] = {}
    passed_n: dict[str, dict[str, int]] = {}
    for setting in settings:
        v = None if setting == "recorded" else verdicts[setting]
        rate[setting] = {}
        passed_n[setting] = {}
        for slug in slugs:
            n = sum(1 for r in rounds.get(slug, []) if round_success(r["checks"], v))
            passed_n[setting][slug] = n
            rate[setting][slug] = 100.0 * n / ROUNDS_PER_MODEL

    score_of: dict[str, dict[str, float]] = {}
    for setting in settings:
        sm = None if setting == "recorded" else verdicts[setting]
        score_of[setting] = {
            slug: (sum(round_score(r["checks"], sm) for r in rounds.get(slug, []))
                   / max(1, len(rounds.get(slug, []))))
            for slug in slugs}

    order = sorted(slugs, key=lambda s: -rate["recorded"][s])
    print("\n=== pass rate (%) by judge; rounds fixed at 600 ===")
    head = f"{'model':20s} {'family':9s}" + "".join(f"{s:>12s}" for s in settings)
    print(head)
    for slug in order:
        row = f"{PRETTY[slug]:20s} {FAMILY[slug]:9s}"
        for s in settings:
            row += f"{rate[s][slug]:11.1f} "
        print(row)

    print("\n=== mean score by judge ===")
    print(f"{'model':20s}" + "".join(f"{s:>12s}" for s in settings))
    for slug in order:
        print(f"{PRETTY[slug]:20s}" +
              "".join(f"{score_of[s][slug]:11.4f} " for s in settings))
    score_summary: dict[str, dict[str, float]] = {}
    for s in settings[1:]:
        rho_s = spearman([score_of["recorded"][x] for x in slugs],
                         [score_of[s][x] for x in slugs])
        d = max(abs(score_of[s][x] - score_of["recorded"][x]) for x in slugs)
        score_summary[s] = {"rho": rho_s, "largest_shift": d}
        print(f"  {s:9s} score rho={rho_s:.4f}  largest absolute shift {d:.4f}")

    def rank_of(setting: str) -> dict[str, int]:
        o = sorted(slugs, key=lambda s: (-rate[setting][s], s))
        return {s: i + 1 for i, s in enumerate(o)}

    print("\n=== rank (1 = best) ===")
    ranks = {s: rank_of(s) for s in settings}
    print(f"{'model':20s}" + "".join(f"{s:>12s}" for s in settings))
    for slug in order:
        print(f"{PRETTY[slug]:20s}" +
              "".join(f"{ranks[s][slug]:11d} " for s in settings))

    print("\n=== rank correlation vs recorded ===")
    base = [rate["recorded"][s] for s in slugs]
    summary: dict[str, object] = {}
    # gemini-3.7-flash is the one row the corpus-standard judge did not grade,
    # so its movement is a correction rather than jitter; hold it out to get a
    # clean noise floor.
    held = [x for x in slugs if x != PUB_37]
    for s in settings[1:]:
        cur = [rate[s][x] for x in slugs]
        rho = spearman(base, cur)
        disp = max(abs(ranks["recorded"][x] - ranks[s][x]) for x in slugs)
        moved = sum(1 for x in slugs if ranks["recorded"][x] != ranks[s][x])
        rho_h = spearman([rate["recorded"][x] for x in held],
                         [rate[s][x] for x in held])
        biggest = max(slugs, key=lambda x: abs(rate[s][x] - rate["recorded"][x]))
        print(f"  {s:9s} rho={rho:.4f}  max displacement={disp}  "
              f"models moved={moved}/14  "
              f"gemini-3.6-flash rank {ranks['recorded']['3_6_flash']} -> "
              f"{ranks[s]['3_6_flash']}")
        print(f"            rho={rho_h:.4f} excluding the self-graded "
              f"gemini-3.7-flash row; largest single move "
              f"{PRETTY[biggest]} {rate[s][biggest] - rate['recorded'][biggest]:+.1f}pp")
        summary[s] = {"rho": rho, "rho_ex_self_graded": rho_h,
                      "max_disp": disp, "moved": moved,
                      "largest_move_model": biggest,
                      "largest_move_pp": rate[s][biggest] - rate["recorded"][biggest]}

    # ---- per-check agreement ---------------------------------------------
    print("\n=== per-check agreement with the recorded verdicts ===")
    rec_by_pair: dict[str, bool] = {}
    slug_by_pair: dict[str, str] = {}
    for slug in slugs:
        for r in rounds.get(slug, []):
            for c in r["checks"]:
                if c.get("pair_id"):
                    rec_by_pair[c["pair_id"]] = c["recorded_passed"]
                    slug_by_pair[c["pair_id"]] = slug
    agreement: dict[str, dict] = {}
    for s in ready:
        common = [(rec_by_pair[p], verdicts[s][p])
                  for p in verdicts[s] if p in rec_by_pair]
        k, agr = kappa(common)
        flips_up = sum(1 for a, b in common if not a and b)
        flips_dn = sum(1 for a, b in common if a and not b)
        print(f"  {s:9s} n={len(common):,}  agreement={agr*100:.2f}%  "
              f"kappa={k:.4f}  F->T {flips_up:,}  T->F {flips_dn:,}")
        agreement[s] = {"n": len(common), "agreement": agr, "kappa": k,
                        "false_to_true": flips_up, "true_to_false": flips_dn}

    # ---- family-level difference in differences ---------------------------
    print("\n=== family mean pass rate (pp change vs recorded) ===")
    family_delta: dict[str, dict[str, float]] = {}
    fams = sorted(set(FAMILY.values()))
    print(f"{'family':10s} {'n':>3s} {'recorded':>10s}" +
          "".join(f"{s:>12s}" for s in ready))
    for f in fams:
        members = [s for s in slugs if FAMILY[s] == f]
        b = sum(rate["recorded"][s] for s in members) / len(members)
        row = f"{f:10s} {len(members):3d} {b:10.2f}"
        family_delta[f] = {"n": len(members), "recorded": b}
        for s in ready:
            cur = sum(rate[s][x] for x in members) / len(members)
            family_delta[f][s] = cur - b
            row += f"{cur - b:+11.2f} "
        print(row)

    bounds = bound_missing(rounds, slugs, verdicts, ready, rate, PRETTY)
    health = ensemble_health(slugs, rounds, ready, rate, PRETTY)
    # Single-judge analysis: use each judge's own full coverage, not the
    # three-way intersection, which opus-5's refusals would otherwise starve.
    dec = decompose_37(rounds, verdicts_full, ready)

    with open(os.path.join(OUT, "phase2_summary.json"), "w") as fh:
        json.dump({"rate": rate, "passed_n": passed_n, "score": score_of,
                   "ranks": {k: v for k, v in ranks.items()},
                   "summary": summary, "settings": settings,
                   "gemini_3_7_decomposition": dec,
                   "missing_pair_bounds": bounds,
                   "ensemble_health": health,
                   "agreement": agreement,
                   "panel_coverage": panel_coverage,
                   "score_summary": score_summary,
                   "family_delta": family_delta,
                   "coverage": {j: len(verdicts[j]) for j in ready}},
                  fh, indent=1)
    print(f"\nwrote {os.path.join(OUT, 'phase2_summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
