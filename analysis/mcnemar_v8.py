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

"""T6 -- paired exact McNemar tests across the three reasoning-effort settings.

Every number comes from analysis/corpus.py, so the population (600 rounds,
error rows retained) and the dead-round guards are not restated here. The
earlier version of this script carried its own loader, its own dead-round
filter and its own copy of the exact test; it also dropped provider-error rows,
which silently changed the paired denominator from one model to the next. All
of that now lives in corpus.py and is shared with T1, so the two tables cannot
disagree about who passed what.

Why paired and exact. The three effort settings are run on the *same* 600
rounds, so the unpaired comparison throws away the pairing that makes small
differences measurable. And the discordant counts here are small (b + c is
often under 60), which is exactly where the chi-squared approximation to
McNemar is least trustworthy -- so the test is the exact binomial one, summing
the tail rather than approximating it.

Multiplicity. The study makes 9 models x 3 effort pairs = 27 comparisons, and
the threshold is Bonferroni-corrected over all 27 whether or not a given table
shows all 27. Correcting only over the rows the paper prints would be picking
the correction after seeing the data.

Pairing is asserted to be complete: with error rows retained, every model has
the same 600 round keys at every effort, so n_pair is 600 everywhere. If that
assertion ever fires, the paired denominators have started to differ between
models and the table is no longer comparable row to row.

T7, added second, is the MODEL-vs-MODEL grid at the primary effort, which is
what section 6.1 cites for the claim that the frontier is a cluster. It was
computed by hand once and quoted from notes; that is exactly how a number
drifts, so it lives here now and is regenerated with everything else.

There is ONE inferential family here: all pairs over the whole panel, corrected
over C(n, 2). It is not selected on the outcome, and an earlier version of this
file already warned that the alternative -- taking all pairs among "the top
six" and correcting over 15 -- is optimistic, because "the top six" is itself
chosen after seeing the pass rates. That warning ended with "the paper should
only lean on a conclusion that holds under the stricter one", so the selected
family has now been removed rather than printed alongside and quoted anyway.

What replaces it is a READOUT of the strict family, not a second family:
corpus.leading_group() names the models that this same all-pairs test does not
separate from the best one, and the separations are then decomposed into
leader-vs-leader, leader-vs-trailer and trailer-vs-trailer. No p-value is
re-corrected against a smaller denominator anywhere in that decomposition.

Doing it this way also retired a genuinely misleading line. The old code
derived its "leading group" as the set of models appearing on the winning side
of a surviving TOP-SIX separation -- which on this data is two models -- and
then reported "no pair inside the leading group separates". True, but about a
pair of models, while the prose around it was read as a statement about the
frontier. The frontier now has one definition, in corpus.leading_group, and
this file consumes it like everyone else.
"""
import itertools
import sys

import corpus

PAIRS = [("high", "low"), ("high", "medium"), ("medium", "low")]


def contingency(a_eps, b_eps):
    """b = solved under a but not b; c = the reverse; over the common keys."""
    common = sorted(set(a_eps) & set(b_eps))
    b = sum(1 for k in common if a_eps[k]["success"] and not b_eps[k]["success"])
    c = sum(1 for k in common if not a_eps[k]["success"] and b_eps[k]["success"])
    return len(common), b, c


def verdict(a, b, c, p, alpha):
    if p >= alpha:
        return "n.s."
    return f"{a} better" if b > c else f"{a} worse"


def main():
    sweep = {lvl: corpus.load_eval(lvl)[0] for lvl in corpus.LEVELS}
    order = corpus.model_order(sweep[corpus.PRIMARY_EFFORT])
    alpha = corpus.bonferroni(len(order) * len(PAIRS))

    rows = {}
    for m in order:
        for a, b_lvl in PAIRS:
            n, b, c = contingency(sweep[a][m], sweep[b_lvl][m])
            assert n == corpus.N_RAW, (
                f"{m} {a} vs {b_lvl}: paired on {n} rounds, not {corpus.N_RAW}; "
                "the effort arms no longer cover the same rounds")
            p = corpus.exact_mcnemar(b, c)
            rows[(m, a, b_lvl)] = (n, b, c, p, verdict(a, b, c, p, alpha))

    print(f"T6  paired exact McNemar   population={corpus.PRIMARY_POPULATION} "
          f"(n={corpus.N_RAW}, complete pairing)   "
          f"Bonferroni alpha = 0.05/{len(order) * len(PAIRS)} = {alpha:.5f}")

    for a, b_lvl in PAIRS:
        hdr = (f"\n{a} vs {b_lvl}\n"
               f"{'model':<{corpus.NAME_W}} {'b':>4} {'c':>4} {'b-c':>5} {'p':>11}  verdict")
        print(hdr)
        print("-" * (len(hdr.splitlines()[-1])))
        for m in order:
            n, b, c, p, v = rows[(m, a, b_lvl)]
            print(f"{m:<{corpus.NAME_W}} {b:>4} {c:>4} {b - c:>5} {p:>11.3g}  {v}")

    sig = [k for k, v in rows.items() if v[4] != "n.s."]
    print(f"\n{len(sig)} of {len(rows)} comparisons clear the corrected "
          f"threshold:")
    for m, a, b_lvl in sorted(sig):
        print(f"  {m:<{corpus.NAME_W}} {a} vs {b_lvl}: {rows[(m, a, b_lvl)][4]}")

    print("\n% ---- LaTeX body for T2, high vs low (paste into the tabular) ----")
    hi_lo = sorted(order, key=lambda m: (rows[(m, 'high', 'low')][3],
                                         -rows[(m, 'high', 'low')][1]))
    for m in hi_lo:
        n, b, c, p, v = rows[(m, "high", "low")]
        # Scientific notation below 0.01, not below 0.001: sol's p is 0.00164
        # against a threshold of 0.00167, and "$0.002$" would read as a value
        # that fails the threshold it actually clears.
        mant, exp = f"{p:.1e}".split("e")
        ptex = (f"${mant}\\times10^{{{int(exp)}}}$" if p < 1e-2
                else f"${p:.3f}$")
        vtex = v if v == "n.s." else (r"\textbf{" + v + "}" if "worse" in v
                                      else v)
        print(f"    \\texttt{{{m}}} & {b} & {c} & {ptex} & {vtex} \\\\")

    model_grid(sweep[corpus.PRIMARY_EFFORT])
    return 0


def model_grid(eps):
    """T7 -- every model against every other, at the primary effort."""
    ranked = corpus.rank_order(eps)
    grid = {}
    for a, b_ in itertools.combinations(ranked, 2):
        n, b, c = contingency(eps[a], eps[b_])
        assert n == corpus.N_RAW, (
            f"{a} vs {b_}: paired on {n} rounds, not {corpus.N_RAW}")
        grid[(a, b_)] = (b, c, corpus.exact_mcnemar(b, c))

    alpha = corpus.bonferroni(len(grid))
    lead = corpus.leading_group(eps)
    trail = [m for m in ranked if m not in lead]
    pp = {m: 100.0 * sum(e["success"] for e in eps[m].values()) / len(eps[m])
          for m in ranked}

    print(f"\n\nT7  paired exact McNemar, model vs model at "
          f"{corpus.PRIMARY_EFFORT} effort   population="
          f"{corpus.PRIMARY_POPULATION} (n={corpus.N_RAW})")
    print(f"    the frontier-is-a-cluster claim of sec.6.1.")

    sig = sorted((grid[k][2], k) for k in grid if grid[k][2] < alpha)
    print(f"\n  all {len(ranked)} models -> {len(grid)} pairs, Bonferroni "
          f"alpha = 0.05/{len(grid)} = {alpha:.5f}: {len(sig)} separate")
    for p, (a, b_) in sig:
        b, c, _ = grid[(a, b_)]
        print(f"    {a:<{corpus.NAME_W}} vs {b_:<{corpus.NAME_W}} "
              f"b={b:>3} c={c:>3}  p={p:.5f}")

    # The frontier cut, and the evidence that it is a gap rather than a graze.
    # Both boundary p-values are printed so a reader can see the jump without
    # taking "robust" on trust.
    print(f"\n  leading group (not separated from {ranked[0]} at the same "
          f"alpha): {len(lead)} of {len(ranked)}")
    print(f"    {', '.join(f'{m} ({pp[m]:.1f})' for m in lead)}")
    if trail:
        b, c, p_out = grid[tuple(sorted((ranked[0], trail[0]), key=ranked.index))]
        _, _, p_in = grid[tuple(sorted((ranked[0], lead[-1]), key=ranked.index))]
        print(f"    boundary: {lead[-1]} in at p={p_in:.5f}, {trail[0]} out at "
              f"p={p_out:.5f} ({p_in / p_out:.0f}x), across a "
              f"{pp[lead[-1]] - pp[trail[0]]:.1f}-point gap in pass rate")

    # THE SHAPE OF THE RESULT, NOT JUST THE COUNT. sec.6.1 reads the frontier as
    # a cluster, which is a claim about the leader-vs-leader cell of this
    # decomposition specifically -- not about the total, which is dominated by
    # leader-vs-trailer pairs that separate trivially. The three cells are
    # printed separately and must add up, so a future roster change cannot
    # quietly move a separation from one cell to another.
    cells = {"leader vs leader": [k for k in grid
                                  if k[0] in lead and k[1] in lead],
             "leader vs trailer": [k for k in grid
                                   if (k[0] in lead) != (k[1] in lead)],
             "trailer vs trailer": [k for k in grid
                                    if k[0] in trail and k[1] in trail]}
    assert sum(len(v) for v in cells.values()) == len(grid)
    print(f"\n  decomposition of the {len(sig)} separations")
    tally = 0
    for name, keys in cells.items():
        hit = sorted((grid[k][2], k) for k in keys if grid[k][2] < alpha)
        tally += len(hit)
        print(f"    {name:<20} {len(hit):>2} of {len(keys):>2} pairs separate")
    assert tally == len(sig), (tally, len(sig))

    inside = sorted((grid[k][2], k) for k in cells["leader vs leader"]
                    if grid[k][2] < alpha)
    print(f"\n  no pair inside the leading group separates: "
          f"{'CONFIRMED' if not inside else 'FALSE'}")
    for p, (a, b_) in inside:
        b, c, _ = grid[(a, b_)]
        same = a.split("-")[0] == b_.split("-")[0]
        print(f"    {a:<{corpus.NAME_W}} vs {b_:<{corpus.NAME_W}} "
              f"b={b:>3} c={c:>3}  p={p:.5f}"
              f"{'  (same provider)' if same else ''}")
    closest = min((grid[k][2], k) for k in cells["leader vs leader"]
                  if grid[k][2] >= alpha)
    print(f"  nearest miss inside the leading group: {closest[1][0]} vs "
          f"{closest[1][1]} at p={closest[0]:.5f} against alpha={alpha:.5f}")

    # The union of models appearing in the separating pairs is the wrong
    # summary -- it grows with the count and says nothing. The INTERSECTION is
    # the claim worth making: if one model is party to every internal
    # separation, the leading group is a cluster with one outlier rather than a
    # group that has come apart. Reported only when the intersection is a
    # single model, because "all involve one of these three" is not a finding.
    common = set.intersection(*[set(k) for _, k in inside]) if inside else set()
    same_prov = all(a.split("-")[0] == b_.split("-")[0] for _, (a, b_) in inside)
    print(f"\n  caption fact: at alpha=0.05/{len(grid)}={alpha:.5f} the "
          f"{len(lead)} leading models are separated by {len(inside)} of their "
          f"{len(cells['leader vs leader'])} pairs"
          + (f", both involving {common.pop()}" if len(common) == 1 and
             len(inside) == 2 else
             f", every one involving {common.pop()}" if len(common) == 1 else "")
          + (" and every one within a single provider" if inside and same_prov
             else "")
          + f"; every one of the {len(cells['trailer vs trailer'])} pairs among "
          f"the {len(trail)} trailing models separates")


if __name__ == "__main__":
    sys.exit(main())
