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

"""Whole-corpus results: pass rate, mean score and cap-hit across three efforts.

NO LONGER THE BODY TABLE. Sec 6.1 leads with difficulty_bands.py instead: the
effort axis this script tabulates was already printed in full by appendix
Table~\\ref{tab:rep-19}, so keeping it in the body spent a page restating an
appendix table, and the difficulty axis had nowhere to go. What survives here
and is load-bearing is cell() -- difficulty_bands.py imports it for the
whole-corpus column and its bootstrap interval, precisely so that the paper's
headline pass rates and CIs have ONE derivation rather than two that agree to
three digits and disagree on the fourth. The LaTeX bodies below are still
emitted and still correct; nothing in main.tex consumes them today.

Every number comes from analysis/corpus.py, so the population (600 rounds,
error rows retained) and the dead-round guards are not restated here.

The three columns are deliberately shown together because the gap between them
is the paper's first claim: the strongest model satisfies most of the scored
units and still closes only a quarter of the tasks. A pass-rate-only table
would make that invisible.

  pass %      fraction of rounds where every required check passed
  score       mean per-round score = mean fraction of that round's required
              checks that passed. Verified against the data: all weights are
              1.0 and |score - unweighted required pass fraction| is 0.000000
              on every round, so this column IS the partial-credit rate and a
              separate "scored units" column would be the same number twice.
  cap %       fraction of rounds that ended by exhausting the 20-step budget

NOT here, on purpose: the POOLED per-check rate (every required check in the
cell over one denominator) is a different statistic -- 63.00% vs a per-round
mean of 64.19% for opus-5 at high effort, because rounds carry unequal numbers
of checks. It belongs to T2, where it is the baseline the clustering analysis
is taken against, and repeating it here would invite the two to be read as the
same quantity.

Prints a LaTeX body as well as the console view; both come from one dict so
they cannot drift.

Two LaTeX bodies are emitted because the paper has two page budgets. The WIDE
body is what the arXiv version prints: all three efforts, a bootstrap interval
on the primary column, and the two cost columns. The COMPACT body is the
high-effort-only fallback held in reserve for a 9-page venue. Both are
generated from the same dict, so choosing one cannot silently change a number.

The interval is a percentile bootstrap over rounds on the primary-effort pass
rate. It is not decoration: this section's claim is that the frontier is a
cluster rather than a ladder, which is a claim about overlap, and an interval
is the only way a reader can check it against the table instead of taking the
McNemar sentence on faith.

Deliberately NOT bolded: the appendix's verbatim report tables bold the best
cell per column, and copying that here would use typography to assert a winner
that the significance testing does not support (11 of the 15 top-six pairs are
indistinguishable). The ordering is already carried by the row blocking.
"""
import json
import sys

import corpus


def tokens_per_task():
    """Reasoning tokens per task, keyed by display name.

    The report's JSON is keyed by the unqualified short form ("opus-5"), so the
    join goes through corpus.REPORT_SHORT rather than a second name table.
    """
    with open(corpus.REASONING_TOKENS) as fh:
        raw = json.load(fh)
    out = {}
    for short, disp in corpus.REPORT_SHORT.items():
        assert short in raw, f"{corpus.REASONING_TOKENS} has no key {short!r}"
        out[disp] = {lvl: raw[short][lvl]["per_task"] for lvl in corpus.LEVELS}
    return out


def cell(eps):
    v = list(eps.values())
    n = len(v)
    # Guard the claim the docstring makes, on every cell, every run.
    for e in v:
        req = [c for c in (e["checks"] or []) if c.get("role") == "required"]
        if not req:
            continue
        frac = sum(1 for c in req if c.get("passed")) / len(req)
        assert abs(frac - e["score"]) < 1e-9, (
            f"{e['task_id']}: score {e['score']} is not the required-check pass "
            f"fraction {frac}; the score column no longer means what T1 says")
    # Seed from the cell's own data, so the interval is reproducible and each
    # cell still draws its own stream. stable_seed rather than hash(): see its
    # docstring for the reproducibility bug that motivated it.
    n_passed = sum(bool(e["success"]) for e in v)
    # ONE resample stream, two functionals. tab:main prints a +- on the pass
    # rate AND on the mean score, and both have to come off the same draws --
    # see corpus.boot_vals. The tuple grew a third slot for the score while the
    # pass functional still reads slot 1, so the resample indices are exactly
    # the ones this function has always drawn and every interval it has printed
    # is unchanged by the score arriving.
    pv, sv = corpus.boot_vals(
        [(e["task_id"], e["success"], e["score"]) for e in v],
        (lambda ps: 100.0 * sum(t[1] for t in ps) / len(ps),
         lambda ps: sum(t[2] for t in ps) / len(ps)),
        seed=corpus.stable_seed("t1-pass", n_passed, n))
    lo, hi = corpus.pct_ci(pv)
    return {
        "n": n,
        "pass": 100.0 * sum(e["success"] for e in v) / n,
        "score": sum(e["score"] for e in v) / n,
        "cap": 100.0 * sum(e["cap_hit"] for e in v) / n,
        "steps": sum(e["steps"] for e in v) / n,
        "err": sum(bool(e["is_error"]) for e in v),
        "passed": n_passed,
        "ci": (lo, hi),
        # Bootstrap standard errors off the same stream as `ci`, so a reader
        # who checks pass +- 1.96*pass_sd against the bracket gets the bracket
        # back to within the asymmetry of the percentile method rather than to
        # within the difference between two runs of the sampler.
        "pass_sd": corpus.boot_sd(pv),
        "score_sd": corpus.boot_sd(sv),
    }


def main():
    sweep = {lvl: corpus.load_eval(lvl, keep_checks=True)[0]
             for lvl in corpus.LEVELS}
    order = corpus.model_order(sweep[corpus.PRIMARY_EFFORT])
    tab = {m: {lvl: cell(sweep[lvl][m]) for lvl in corpus.LEVELS} for m in order}

    blocks = corpus.provider_blocks(sweep[corpus.PRIMARY_EFFORT])
    print(f"T1  main results   population={corpus.PRIMARY_POPULATION} "
          f"(n={corpus.N_RAW})   blocked by provider, each block and the "
          f"blocks themselves ordered by pass rate at "
          f"{corpus.PRIMARY_EFFORT} effort")
    hdr = (f"{'model':<{corpus.NAME_W}} | {'pass % h':>8} {'m':>6} {'l':>6} | "
           f"{'score h':>7} {'m':>6} {'l':>6} | {'cap % h':>7} {'m':>6} {'l':>6}")
    print(hdr)
    print("-" * len(hdr))
    for m in order:
        r = tab[m]
        print(f"{m:<{corpus.NAME_W}} | "
              + " ".join(f"{r[l]['pass']:>{8 if l=='high' else 6}.2f}" for l in corpus.LEVELS)
              + " | "
              + " ".join(f"{r[l]['score']:>{7 if l=='high' else 6}.3f}" for l in corpus.LEVELS)
              + " | "
              + " ".join(f"{r[l]['cap']:>{7 if l=='high' else 6}.1f}" for l in corpus.LEVELS))

    hi = corpus.PRIMARY_EFFORT
    best = max(order, key=lambda m: tab[m][hi]["pass"])
    r = tab[best][hi]
    print(f"\nthe headline gap: {best} banks a mean score of {r['score']:.3f} at "
          f"{hi} effort -- {100*r['score']:.1f}% of the scored units in an average "
          f"round -- and closes {r['pass']:.1f}% of rounds")

    # Scale of the sweep, for the caption. Asserted rather than asserted-in-
    # prose: the caption claims every cell is exactly N_RAW, so check it.
    allcells = [tab[m][l] for m in order for l in corpus.LEVELS]
    bad = [(m, l) for m in order for l in corpus.LEVELS
           if tab[m][l]["n"] != corpus.N_RAW]
    assert not bad, f"cells not at n={corpus.N_RAW}: {bad}"
    episodes = sum(c["n"] for c in allcells)
    errors = sum(c["err"] for c in allcells)
    print(f"scale: {len(order)} models x {len(corpus.LEVELS)} efforts x "
          f"{corpus.N_RAW} rounds = {episodes:,} episodes; "
          f"{errors:,} provider-error episodes retained and scored as failures "
          f"({100.0*errors/episodes:.2f}%)")

    # Overlap of the primary-effort intervals -- the cluster claim, measured.
    top = order[0]
    tlo, thi = tab[top][hi]["ci"]
    overlap = [m for m in order[1:]
               if tab[m][hi]["ci"][1] >= tlo and tab[m][hi]["ci"][0] <= thi]
    print(f"cluster: {len(overlap)} of the other {len(order)-1} models have a "
          f"95% interval overlapping {top}'s [{tlo:.1f}, {thi:.1f}]: "
          + ", ".join(overlap))

    # Two LaTeX bodies from one dict, so the choice of venue cannot change a
    # number. WIDE is what the arXiv body prints; COMPACT is the fallback held
    # in reserve for a hard page limit.
    def emit(title, rowfn, spec):
        print(f"\n% ---- LaTeX body for {title} (paste into the tabular) ----")
        print(f"%      column spec: {spec}")
        for i, (_prov, ms) in enumerate(blocks):
            if i:
                print(r"    \midrule")
            for m in ms:
                print(f"    \\texttt{{{m}}}" + " " * (18 - len(m))
                      + " & " + " & ".join(rowfn(m)) + r" \\")

    print("\n% ---- caption clause forced by the provider blocking ----")
    print("% " + corpus.blocking_note(sweep[corpus.PRIMARY_EFFORT]))

    tok = tokens_per_task()

    def wide(m):
        r = tab[m]
        lo, up = r[hi]["ci"]
        return ([f"{r[hi]['pass']:.1f}", f"[{lo:.1f},\\,{up:.1f}]"]
                + [f"{r[l]['pass']:.1f}" for l in corpus.LEVELS if l != hi]
                + [f"{r[l]['score']:.3f}" for l in corpus.LEVELS]
                + [f"{r[l]['cap']:.1f}" for l in corpus.LEVELS]
                + [f"{tok[m][hi]/1000.0:.1f}"])

    def compact(m):
        r = tab[m][hi]
        return [f"{r['pass']:.1f}", f"{r['score']:.3f}", f"{r['cap']:.1f}"]

    # Mean steps is computed and printed in the console view above but is
    # deliberately NOT a column: step behaviour gets its own figure, where the
    # budget ceiling and the shape of the distribution are visible. A single
    # mean next to a cap-hit percentage says less than either.
    emit("T1 WIDE body table (arXiv): pass h/CI/m/l, score h/m/l, cap h/m/l, "
         "ktok", wide, "l r c rr rrr rrr r")
    emit(f"T1 COMPACT fallback, {hi} effort only", compact, "l rrr")
    return 0


if __name__ == "__main__":
    sys.exit(main())
