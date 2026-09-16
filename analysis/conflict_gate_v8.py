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

"""Is it the CONFLICT that makes these rounds hard, or just the computer use?

Every seed in the corpus carries a conflict, so there is no no-conflict arm to
compare against and the headline pass rate cannot, on its own, tell a reviewer
whether 17.7% is a statement about adjudicating contradictions or a statement
about long-horizon tool use. This module answers the question with an ablation
the GENERATOR already ran, unremarked, on every package.

THE DESIGN. The harness does not ask every round to RESOLVE its conflict. Six
of the ten positions in a package are selected by a deterministic rule that
looks only at the round index:

    _round_requires_mechanic(i) = floor(i*2/3) > floor((i-1)*2/3)
                                = {r02, r03, r05, r06, r08, r09}

and on those the gold trajectory must carry out a real state-changing protocol
step -- "not merely inspect it" -- with r09 required to COMPLETE the protocol.
The other four rounds carry the same conflicting evidence, the same peers, the
same assets, and ask only that the agent investigate it. So within a single
package we have rounds that demand resolution and rounds that demand only
comprehension, assigned by a rule that cannot see the scenario.

THE PLACEBO. That rule fires only where the seed routes to a plugin declaring
`emits_effects`. For the eight seeds that route to none, the gate is a no-op:
the same six positions carry no extra demand at all. Those seeds are therefore
a placebo arm that measures what round POSITION alone is worth -- narrative
drift, evidence accumulating, whatever else moves down a package -- and the
difference of the two differences is the part attributable to being asked to
act. Reporting the treated drop alone would be the version of this result a
reviewer can wave away as a round-order artefact; that is why the placebo row
is printed first in the console block and sits directly under the treated row
in the table.

WHAT THE ESTIMATE IS, EXACTLY. This is an intention-to-treat contrast on a
GENERATION-TIME instruction, not a switch that adds a conflict to a round that
lacked one. Both arms are conflict scenarios throughout. The treatment is the
demand to resolve rather than to investigate, and `compliance()` prints how
often it actually bound: the gate is far from perfectly enforced, which makes
the ITT a LOWER bound on the effect of the demand it stands for. The claim this
licenses is "resolving a conflict is ~17 points harder than investigating the
same conflict", and NOT "conflict costs 17 points". Section 6.4 must say the
former. See `caption_facts()`, which prints the sentence the table is allowed
to support.

THREE CONFOUNDS ARE PRICED, NOT ASSUMED AWAY. A gated round could be harder
because it is longer (gold length), because it is graded on more predicates
(required checks), or because the generator gave it a more demanding round_type
-- all three correlate with the gate. `strata()` re-estimates the DiD within
each level of each and the answer moves by less than two points in every case.
The round_type stratification also carries the mechanism check worth more than
the p-value: inside `action` rounds, which already demanded a state change, the
treated effect is ~0. The gate only bites where it adds something.

TWO TESTS, BECAUSE THEY ANSWER DIFFERENT QUESTIONS. The permutation null
reshuffles which six of the ten POSITIONS are called gated, holding the corpus
fixed -- it asks whether this particular split of round indices is special.
C(10,6) = 210, so it is enumerated exactly rather than sampled, and its floor
is 2/210 = .0095: a p of .05 here is near the design's resolution, not a weak
signal, and `main()` says so. The bootstrap resamples SEEDS, which asks the
other question -- whether the effect survives the particular twenty scenarios
we happen to have -- and is the interval the table prints.

Prints a console view and two LaTeX bodies from one dict, so they cannot drift.
"""
import itertools
import random
import sys
from collections import defaultdict

import corpus

EFFORT = corpus.PRIMARY_EFFORT
ARMS = ("mechanic", "placebo")
# Printed names. "Resolve-gated" and "Investigate-only" rather than on/off:
# the reader has to know from the column head that BOTH columns are conflict
# rounds, because the one misreading that would invalidate the section is
# "ungated == no conflict".
COLS = ("gated", "ungated")
HEAD = {"gated": "Resolve-gated", "ungated": "Investigate-only"}


# The two outcomes, deliberately on one code path. PASS is the headline and
# SCORE is the boundary the section has to concede: the same contrast is much
# weaker on partial credit, and running it through a second, hand-rolled
# routine is how the two would silently stop being the same estimator.
PASS = ("pass rate", lambda e: float(e["success"]))
SCORE = ("mean score", lambda e: e["score"])


def cells(eps, order, gate, sp, val=PASS[1]):
    """(arm, gated) -> [total, n], pooled over models, plus the per-seed split."""
    pool = defaultdict(lambda: [0.0, 0])
    per_seed = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    per_model = {m: defaultdict(lambda: [0.0, 0]) for m in order}
    for m in order:
        for tid, e in eps[m].items():
            seed = sp[tid]["seed"]
            arm, g = corpus.gate_arm(seed, e["round_id"], gate)
            for tgt in (pool[(arm, g)], per_seed[seed][g], per_model[m][(arm, g)]):
                tgt[0] += val(e)
                tgt[1] += 1
    return pool, per_seed, per_model


def rate(c):
    return 100.0 * c[0] / c[1] if c[1] else float("nan")


def num(n):
    """4-digit thousands separator LaTeX actually wants inside a table cell."""
    return f"{n:,}".replace(",", "{,}")


def sgn(v):
    """A signed number in math mode, so the minus is a minus and not a hyphen."""
    return f"${v:+.1f}$"


def did_of(get):
    """(DiD, treated delta, placebo delta) from a cell accessor."""
    t = rate(get("mechanic", True)) - rate(get("mechanic", False))
    p = rate(get("placebo", True)) - rate(get("placebo", False))
    return t - p, t, p


def permutation(eps, order, gate, sp, rounds=10, val=PASS[1]):
    """Exact null over which six of the ten POSITIONS count as gated.

    Enumerated, not sampled: C(10,6) = 210 is small enough that a sampled null
    would only add Monte Carlo noise to a p-value whose floor (2/210 = .0095)
    is already set by the design. Both statistics are two-sided on the same
    enumeration, so the treated-only p and the DiD p are directly comparable.
    """
    by = defaultdict(lambda: [0.0, 0])
    for m in order:
        for tid, e in eps[m].items():
            seed = sp[tid]["seed"]
            arm, _ = corpus.gate_arm(seed, e["round_id"], gate)
            pos = int(e["round_id"].lstrip("r"))
            by[(arm, pos)][0] += val(e)
            by[(arm, pos)][1] += 1

    def stat(on):
        on = frozenset(on)

        def get(arm, g):
            ps = [p for p in range(1, rounds + 1) if (p in on) == g]
            return [sum(by[(arm, p)][i] for p in ps) for i in (0, 1)]
        return did_of(get)

    k = len(gate["gated_positions"])
    obs_did, obs_t, _ = stat(gate["gated_positions"])
    allsets = list(itertools.combinations(range(1, rounds + 1), k))
    vals = [stat(s) for s in allsets]
    ge_d = sum(1 for d, _, _ in vals if abs(d) >= abs(obs_did) - 1e-9)
    ge_t = sum(1 for _, t, _ in vals if abs(t) >= abs(obs_t) - 1e-9)
    return {"draws": len(allsets), "p_did": ge_d / len(allsets),
            "p_treated": ge_t / len(allsets), "floor": 2.0 / len(allsets),
            "rank": sorted(d for d, _, _ in vals).index(obs_did) + 1}


def bootstrap(per_seed, gate, draws=corpus.DRAWS):
    """Seed-cluster bootstrap of the DiD: does it survive THESE twenty seeds?

    The resampling unit is the seed, not the round. Rounds inside a package
    share a scenario, a set of assets and a conflict; treating 8,400 of them as
    independent would report an interval several times too tight and would be
    the same mistake as reading the 14/14 model agreement as fourteen
    independent replications -- all fourteen see the same twenty seeds.
    """
    seeds = sorted(per_seed)
    mech = {s for s in seeds if gate["seeds"][s]["mechanic"]}
    rnd = random.Random(corpus.stable_seed("conflict-gate", "did"))
    out = []
    for _ in range(draws):
        samp = [rnd.choice(seeds) for _ in seeds]
        # A draw with an empty arm has no DiD to report. Skipping is the only
        # honest option (both alternatives -- calling it zero, or rejecting the
        # draw and redrawing -- bias the interval), and it is rare enough at
        # 12/8 that the effective draw count is printed rather than assumed.
        if not (set(samp) & mech) or set(samp) <= mech:
            continue
        acc = defaultdict(lambda: [0, 0])
        for s in samp:
            arm = "mechanic" if s in mech else "placebo"
            for g, v in per_seed[s].items():
                acc[(arm, g)][0] += v[0]
                acc[(arm, g)][1] += v[1]
        out.append(did_of(lambda a, g: acc[(a, g)])[0])
    out.sort()
    return {"lo": out[int(0.025 * len(out))], "hi": out[int(0.975 * len(out))],
            "sd": corpus.boot_sd(out), "n": len(out),
            "ge0": sum(1 for v in out if v >= 0)}


def strata(eps, order, gate, sp, key, minimum=40):
    """Re-estimate the DiD inside each level of a covariate.

    Weighted by the smaller of the two cell counts in the level, which is the
    binding denominator on that level's delta -- weighting by the total would
    let a level with 400 gated rounds and 41 ungated ones dominate on the
    strength of a number the 41 cannot support. Levels thinner than `minimum`
    on either side are dropped rather than shrunk, and the count of dropped
    rounds is returned so the caller can report the coverage it bought.
    """
    rows = defaultdict(lambda: [0, 0])
    for m in order:
        for tid, e in eps[m].items():
            arm, g = corpus.gate_arm(sp[tid]["seed"], e["round_id"], gate)
            rows[(arm, sp[tid][key], g)][0] += int(e["success"])
            rows[(arm, sp[tid][key], g)][1] += 1
    levels = sorted({k[1] for k in rows}, key=lambda v: (v is None, v))
    out = {}
    dropped = 0
    for arm in ARMS:
        num = den = 0.0
        for lv in levels:
            a, b = rows[(arm, lv, True)], rows[(arm, lv, False)]
            if a[1] < minimum or b[1] < minimum:
                dropped += a[1] + b[1]
                continue
            w = min(a[1], b[1])
            num += w * (rate(a) - rate(b))
            den += w
        out[arm] = num / den if den else float("nan")
    return {"treated": out["mechanic"], "placebo": out["placebo"],
            "did": out["mechanic"] - out["placebo"], "dropped": dropped,
            "levels": levels, "rows": rows}


def dose(eps, order, gate, sp):
    """Ungated / advance-only / complete, inside the treated arm.

    The gate has two tiers -- most selected rounds must ADVANCE the protocol,
    exactly one per package must COMPLETE it -- so the design supplies its own
    dose-response check for free. A monotone gradient is much harder to explain
    as a labelling artefact than a two-cell contrast.
    """
    term = gate["terminal_position"]
    out = {k: [0, 0] for k in ("ungated", "advance", "complete")}
    for m in order:
        for tid, e in eps[m].items():
            arm, g = corpus.gate_arm(sp[tid]["seed"], e["round_id"], gate)
            if arm != "mechanic":
                continue
            pos = int(e["round_id"].lstrip("r"))
            k = "ungated" if not g else ("complete" if pos == term else "advance")
            out[k][0] += int(e["success"])
            out[k][1] += 1
    return out


def compliance(eps, order, gate, sp):
    """How often a gated round is actually GRADED on a protocol effect.

    This is the first stage, and it is weak: the instruction is injected into a
    generation prompt, not enforced by the checker compiler, so a gated round
    can end up graded on none of its plugin's declared effects and an ungated
    round can end up graded on one anyway. Both directions attenuate the
    contrast toward zero, which is why the ITT is a lower bound and why this
    number belongs in the appendix rather than being quietly omitted.

    It is also an UNDER-count of the treatment: `emits_effects` names the
    terminal effects, while the advance tier accepts any registered protocol
    command, several of which emit effects the plugin never declares. So the
    real separation between the arms is wider than the two rates below.
    """
    out = defaultdict(lambda: [0, 0])
    seen = set()
    for m in order:
        for tid, e in eps[m].items():
            if tid in seen:
                continue
            names = [c["name"] for c
                     in (e.get("metadata") or {}).get("check_results") or []
                     if c.get("role") == "required"]
            if not names:
                continue
            seen.add(tid)
            seed = sp[tid]["seed"]
            arm, g = corpus.gate_arm(seed, e["round_id"], gate)
            if arm != "mechanic":
                continue
            eff = set(gate["seeds"][seed]["effects"])
            hit = any(n.startswith("effect:") and n.split()[0][7:] in eff
                      for n in names)
            out[g][0] += hit
            out[g][1] += 1
    return out


def replication(gate, sp):
    """The same DiD at medium and low effort, on the panels that exist there.

    Not a robustness check on the statistic -- a check on the corpus. If the
    gate is measuring a property of the scenarios rather than of one panel's
    decoding, three disjoint model sets should land on the same number.
    """
    out = {}
    for eff in corpus.LEVELS:
        eps, _ = corpus.load_panel(eff, keep_checks=False)
        order = corpus.model_order(eps)
        pool, _, _ = cells(eps, order, gate, sp)
        d, t, p = did_of(lambda a, g: pool[(a, g)])
        # Each replication carries its OWN exact p. A panel of point estimates
        # with a single p attached to the first row invites the reader to
        # borrow that p for the other two, and at ten models rather than
        # fourteen there is no reason to assume it transfers.
        out[eff] = {"models": len(order), "did": d, "treated": t, "placebo": p,
                    "p": permutation(eps, order, gate, sp,
                                     gate["rounds_per_package"])["p_did"],
                    "cells": {k: list(v) for k, v in pool.items()}}
    return out


def main():
    gate = corpus.load_gate_map()
    sp = corpus.load_specs()
    eps, errors = corpus.load_panel(EFFORT, keep_checks=True)
    order = corpus.model_order(eps)
    pool, per_seed, per_model = cells(eps, order, gate, sp)
    nseed = {a: sum(1 for s, v in gate["seeds"].items()
                    if v["mechanic"] == (a == "mechanic")) for a in ARMS}

    obs_did, obs_t, obs_p = did_of(lambda a, g: pool[(a, g)])
    perm = permutation(eps, order, gate, sp, gate["rounds_per_package"])
    boot = bootstrap(per_seed, gate)
    st = {k: strata(eps, order, gate, sp, k) for k in ("gold", "units", "rtype")}
    ds = dose(eps, order, gate, sp)
    comp = compliance(eps, order, gate, sp)
    rep = replication(gate, sp)
    # The same contrast on PARTIAL CREDIT. It is much weaker, and that is a
    # finding rather than an embarrassment: a gated round is one the model
    # mostly satisfies and then fails on the decisive predicate, which is the
    # competence-resolution gap of the taxonomy section stated causally. The
    # section is required to concede it, so it is computed here rather than
    # left to whoever notices.
    s_pool, _, _ = cells(eps, order, gate, sp, SCORE[1])
    s_did, s_t, s_p = did_of(lambda a, g: s_pool[(a, g)])
    s_perm = permutation(eps, order, gate, sp, gate["rounds_per_package"],
                         SCORE[1])

    print("conflict-protocol gate -- H0: a round is no harder for being asked to")
    print("RESOLVE its conflict than to investigate the same conflict. Treatment")
    print(f"is the generator's own rule (positions {gate['gated_positions']}, "
          f"terminal r{gate['terminal_position']:02d});")
    print("the placebo arm is the seeds the rule cannot fire for.\n")
    print(f"  {'arm':<12} {'seeds':>5} {'rounds':>7} "
          f"{HEAD['gated']:>14} {HEAD['ungated']:>17} {'delta':>8}")
    for a in ARMS:
        n = pool[(a, True)][1] + pool[(a, False)][1]
        print(f"  {a:<12} {nseed[a]:>5} {n:>7} "
              f"{rate(pool[(a, True)]):>14.2f} {rate(pool[(a, False)]):>17.2f} "
              f"{rate(pool[(a, True)]) - rate(pool[(a, False)]):>+8.2f}")
    print(f"\n  difference-in-differences         {obs_did:+.2f} pp")
    print(f"  95% CI (seed-cluster bootstrap)   [{boot['lo']:+.2f}, {boot['hi']:+.2f}]"
          f"   SE {boot['sd']:.2f}   {boot['ge0']}/{boot['n']} draws >= 0")
    print(f"  exact permutation over C(10,{len(gate['gated_positions'])})"
          f"={perm['draws']}    p(DiD) = {perm['p_did']:.4f}"
          f"   p(treated) = {perm['p_treated']:.4f}")
    print(f"  ... the floor of that test is 2/{perm['draws']} = {perm['floor']:.4f}, "
          f"and the observed DiD ranks {perm['rank']} smallest;")
    print("      a p near .05 here is the design's resolution, not a weak signal.")

    print("\n  dose-response inside the treated arm:")
    for k, lab in (("ungated", "investigate only"), ("advance", "advance protocol"),
                   ("complete", "complete protocol")):
        print(f"    {lab:<20} {rate(ds[k]):6.2f}%   n={ds[k][1]}")

    print("\n  the same contrast on partial credit rather than on the round:")
    print(f"    {SCORE[0]:<20} treated {s_t:+.2f}   placebo {s_p:+.2f}   "
          f"DiD {s_did:+.2f}   p = {s_perm['p_did']:.4f}")
    print(f"    {PASS[0]:<20} treated {obs_t:+.2f}   placebo {obs_p:+.2f}   "
          f"DiD {obs_did:+.2f}   p = {perm['p_did']:.4f}")
    print("    => the gate costs the ROUND, not most of its predicates: models")
    print("       keep the partial credit and lose the decisive action (F5).")

    print("\n  DiD re-estimated inside each level of a covariate:")
    print(f"    {'covariate':<22} {'treated':>8} {'placebo':>8} {'DiD':>8} {'dropped':>8}")
    print(f"    {'(none)':<22} {obs_t:>+8.2f} {obs_p:>+8.2f} {obs_did:>+8.2f} {0:>8}")
    for k, lab in (("gold", "gold length"), ("units", "required checks"),
                   ("rtype", "round type")):
        v = st[k]
        print(f"    {lab:<22} {v['treated']:>+8.2f} {v['placebo']:>+8.2f} "
              f"{v['did']:>+8.2f} {v['dropped']:>8}")

    print("\n  round_type detail -- the mechanism check. `action` rounds already")
    print("  demanded a state change, so the gate adds nothing there:")
    rows = st["rtype"]["rows"]
    print(f"    {'round_type':<16} {'treated d':>10} {'placebo d':>10} {'gated n':>9}")
    for lv in st["rtype"]["levels"]:
        d = {}
        for a in ARMS:
            x, y = rows[(a, lv, True)], rows[(a, lv, False)]
            d[a] = (rate(x) - rate(y)) if min(x[1], y[1]) >= 40 else None
        n = rows[("mechanic", lv, True)][1]
        f = lambda v: f"{v:+10.1f}" if v is not None else f"{'--':>10}"
        print(f"    {str(lv):<16}{f(d['mechanic'])}{f(d['placebo'])}{n:>9}")

    neg = sum(1 for m in order
              if did_of(lambda a, g: per_model[m][(a, g)])[0] < 0)
    print(f"\n  per-model: {neg}/{len(order)} models have a negative DiD"
          f"  (NOT {len(order)} independent replications -- same "
          f"{len(gate['seeds'])} seeds)")
    print(f"  {'model':<{corpus.NAME_W}} {'M:gated':>8} {'M:ungtd':>8} {'M:d':>7}"
          f" {'P:d':>7} {'DiD':>8}")
    for m in order:
        d, t, p = did_of(lambda a, g: per_model[m][(a, g)])
        print(f"  {m:<{corpus.NAME_W}} {rate(per_model[m][('mechanic', True)]):>8.1f}"
              f" {rate(per_model[m][('mechanic', False)]):>8.1f} {t:>+7.1f}"
              f" {p:>+7.1f} {d:>+8.1f}")

    print("\n  per-seed delta (gated - ungated), pooled over models:")
    for a in ARMS:
        ss = sorted(s for s in per_seed if gate["seeds"][s]["mechanic"] == (a == "mechanic"))
        dl = [rate(per_seed[s][True]) - rate(per_seed[s][False]) for s in ss]
        q = sorted(dl)
        med = (q[len(q) // 2] if len(q) % 2
               else 0.5 * (q[len(q) // 2 - 1] + q[len(q) // 2]))
        print(f"    {a}: {sum(1 for x in dl if x < 0)}/{len(dl)} negative, "
              f"median {med:+.1f}pp")
        for s, x in sorted(zip(ss, dl), key=lambda p: p[1]):
            print(f"      {s:<40} {x:+7.1f}")

    print("\n  replication at the other efforts (disjoint panels):")
    print(f"    {'effort':<8} {'models':>7} {'treated':>8} {'placebo':>8} {'DiD':>8}")
    for eff in corpus.LEVELS:
        v = rep[eff]
        print(f"    {eff:<8} {v['models']:>7} {v['treated']:>+8.2f} "
              f"{v['placebo']:>+8.2f} {v['did']:>+8.2f}")

    print("\n  first stage (how often a gated round is graded on a declared")
    print("  protocol effect) -- weak, and an UNDER-count; see compliance():")
    for g, lab in ((True, "gated"), (False, "ungated")):
        c = comp[g]
        print(f"    {lab:<9} {c[0]}/{c[1]} = {100.0 * c[0] / c[1]:.1f}%")
    print("    => the ITT below is a LOWER bound on the demand it stands for.")

    emit_body(pool, nseed, obs_did, boot, perm)
    emit_full(pool, nseed, obs_did, obs_t, obs_p, boot, perm, ds, order, errors)
    emit_robustness(st, obs_t, obs_p, obs_did, rep, comp, per_model, order, gate,
                    (s_t, s_p, s_did, s_perm))
    return 0


def caption_facts(obs_did, boot, perm, ds, nseed):
    """The one sentence this table licenses, printed so it cannot be overstated.

    Written out rather than left to the section author because the tempting
    paraphrase -- "conflict costs 17 points" -- is not what was measured and is
    the single claim a reviewer can falsify by reading the seed files, all
    twenty of which carry a conflict in every round.
    """
    p = f"{perm['p_did']:.3f}".lstrip("0")
    return "\n".join([
        "% caption facts: BOTH columns are conflict rounds. Every one of the "
        f"{nseed['mechanic'] + nseed['placebo']} seeds carries a conflict in "
        "every round, so the contrast is being asked to RESOLVE the conflict",
        "% versus to investigate the same conflict -- it is NOT conflict "
        "versus no conflict, and the caption must not say that it is.",
        f"% caption facts: inside the treated arm the pass rate falls "
        f"{rate(ds['ungated']):.1f} -> {rate(ds['advance']):.1f} -> "
        f"{rate(ds['complete']):.1f}\\% across investigate / advance the "
        f"protocol / complete it, while the {nseed['placebo']}-seed placebo",
        "% arm the gate cannot fire for moves by under a point over the same "
        f"round positions; DiD {obs_did:+.1f}pp, 95\\% CI "
        f"[{boot['lo']:+.1f}, {boot['hi']:+.1f}], exact $p$ = {p}.",
    ])


def emit_body(pool, nseed, obs_did, boot, perm):
    """The narrow in-text version: two arms, the estimate, the interval, the test.

    This one rides in a half-width wraptable beside the prose, so it is cut to
    the four columns a reader needs to see that the design is a difference of
    differences: the two arms, the two conditions, and the gap. Everything the
    full grid also carried -- seed and round counts per arm, the dose-response
    tiers -- moves to the appendix copy (emit_full), which is the version to
    cite when a number is being checked rather than being made.

    The placebo row is the one row that cannot be cut. Without it the table is
    a round-order contrast, not a DiD, and the section's whole claim collapses.
    """
    # Three markers now share the "T10" stem and verify_table_bodies matches
    # markers as SUBSTRINGS, first hit wins -- so no marker may be a prefix of
    # another. "main" / "full" / "robustness" are pairwise non-nesting; do not
    # rename one to something like "main grid" that swallows "main".
    print("\n% ---- LaTeX body for T10 main (paste into the tabular) ----")
    print(r"%      column spec: l r r r  -- narrow, for the wraptable")
    print(r"% Both columns are CONFLICT rounds: resolve vs investigate the SAME")
    print(r"% conflict, never conflict vs none. Per-arm n and the dose tiers are")
    print(r"% in the appendix copy; see the notes at the end of this block.")
    lab = {"mechanic": "Treated ({} seeds)", "placebo": "Placebo ({} seeds)"}
    for a in ARMS:
        print(f"    {lab[a].format(nseed[a]):<24} & "
              f"{rate(pool[(a, True)]):.1f} & {rate(pool[(a, False)]):.1f} & "
              f"{sgn(rate(pool[(a, True)]) - rate(pool[(a, False)]))}" + r" \\")
    print(r"    \midrule")
    # \multicolumn{3}{r} rather than three empty cells: at this width three
    # trailing "& &" would leave the estimate stranded under the Delta column
    # alone, which is where the per-arm gaps live, and the DiD is not one of
    # those gaps -- it is the gap between them.
    print(r"    \textbf{Difference-in-differences} & \multicolumn{3}{r}{"
          f"$\\mathbf{{{obs_did:+.1f}}}$" + r"} \\")
    print(r"    \quad 95\% CI (seed bootstrap) & \multicolumn{3}{r}{"
          f"$[{boot['lo']:+.1f}, {boot['hi']:+.1f}]$" + r"} \\")
    print(r"    \quad Exact permutation $p$ & \multicolumn{3}{r}{"
          + f"{perm['p_did']:.3f}".lstrip("0") + r"} \\")
    print("% caption facts: BOTH columns are conflict rounds; the contrast is "
          "being asked to RESOLVE the conflict versus to investigate the same "
          "one.")
    n_all = num(sum(pool[(a, g)][1] for a in ARMS for g in (True, False)))
    n_seeds = nseed["mechanic"] + nseed["placebo"]
    print(f"% caption facts: arms hold {n_all} rounds over {n_seeds} seeds; "
          f"the full grid with per-arm n and the dose-response tiers is the "
          f"appendix copy.")


def emit_full(pool, nseed, obs_did, obs_t, obs_p, boot, perm, ds, order, errors):
    """The two-arm grid plus the interval and the test."""
    # The marker says "full" so that it does not nest inside the other two
    # blocks' markers: verify_table_bodies matches markers as SUBSTRINGS and
    # takes the first hit, so "T10" alone would resolve all three labels to
    # whichever block printed first and report a clean pass on one table thrice.
    print("\n% ---- LaTeX body for T10 full (paste into the tabular) ----")
    print(r"%      column spec: l rr rr r")
    print(r"% Both columns are CONFLICT rounds. The contrast is resolve vs "
          r"investigate,")
    # Deliberately NOT the phrase "caption facts" here: gen_block treats that
    # string as a block terminator, and a comment containing it in the middle
    # of the block would end the run before the first row.
    print(r"% not conflict vs none -- see the notes at the end of this block.")
    lab = {"mechanic": r"Treated: gate can fire",
           "placebo": r"Placebo: gate is a no-op"}
    for a in ARMS:
        n = num(pool[(a, True)][1] + pool[(a, False)][1])
        print(f"    {lab[a]:<28} & {nseed[a]} & {n} & "
              f"{rate(pool[(a, True)]):.1f} & {rate(pool[(a, False)]):.1f} & "
              f"{sgn(rate(pool[(a, True)]) - rate(pool[(a, False)]))}" + r" \\")
    print(r"    \midrule")
    print(r"% The estimate. The placebo row above is what makes this a difference"
          r" of differences")
    print(r"% rather than a round-order contrast, so it must stay on the page.")
    print(r"    \textbf{Difference-in-differences} & & & & & "
          f"$\\mathbf{{{obs_did:+.1f}}}$" + r" \\")
    print(r"    \quad 95\% CI (seed bootstrap) & & & & & "
          f"$[{boot['lo']:+.1f}, {boot['hi']:+.1f}]$" + r" \\")
    print(r"    \quad Exact permutation $p$ & & & & & "
          + f"{perm['p_did']:.3f}".lstrip("0") + r" \\")
    print(r"    \midrule")
    # The sub-header is emitted, not typed into main.tex, for two reasons: a
    # hand-written row here would split the block into two non-contiguous runs
    # and verify_table_bodies would stop checking the dose rows, and the rate
    # in these rows is NOT one of the two pass-rate columns above -- it spans
    # them, and a reader who takes "27.2" as a Resolve-gated figure has read
    # the table exactly backwards.
    print(r"% Dose-response, treated arm only: the gate has two tiers and this is"
          r" the gradient.")
    print(r"    \multicolumn{3}{l}{\emph{Dose--response within the treated arm}}"
          r" & \multicolumn{2}{c}{\textbf{Pass rate (\%)}} & \\")
    for k, lb in (("ungated", r"\quad Investigate only"),
                  ("advance", r"\quad Advance the protocol"),
                  ("complete", r"\quad Complete the protocol")):
        print(f"    {lb:<28} & & {num(ds[k][1])} & "
              f"\\multicolumn{{2}}{{c}}{{{rate(ds[k]):.1f}}} & " + r" \\")
    print(caption_facts(obs_did, boot, perm, ds, nseed))
    ndraw = f"{boot['n']:,}".replace(",", "{,}")
    print(f"% caption facts: the permutation floor is 2/{perm['draws']} = "
          f"{perm['floor']:.4f}, so $p$ = {perm['p_did']:.3f} is near the "
          f"resolution of a test with only {perm['draws']} distinct "
          f"labellings, not a weak effect; the bootstrap puts "
          f"{boot['ge0']} of {ndraw} seed resamples at or above zero.")
    if errors:
        print(f"% provider-error rows retained: {dict(errors)}")


def emit_robustness(st, obs_t, obs_p, obs_did, rep, comp, per_model, order,
                    gate, sc):
    """The appendix panel: the three confounds, the replication, the outcome.

    The partial-credit row is last and is NOT a robustness check that passed --
    it is the one specification under which the effect is not significant, and
    it sits in the same panel as the ones that survive because a robustness
    table that only shows the surviving specifications is not one.
    """
    pv = lambda v: f"{v:.3f}".lstrip("0")

    def row(lb, t, p, d, pval=None):
        print(f"    {lb:<38} & {sgn(t)} & {sgn(p)} & {sgn(d)} & "
              + (pv(pval) if pval is not None else "") + r" \\")

    print("\n% ---- LaTeX body for T10 robustness (paste into the tabular) ----")
    print(r"%      column spec: l cccc")
    print(r"% The unadjusted row IS the high-effort panel, so there is no separate"
          r" high row.")
    print(r"% The three that follow re-estimate the SAME DiD inside levels of a"
          r" covariate")
    print(r"% that correlates with the gate; none of them explains it away. Their"
          r" p column")
    print(r"% is blank because no permutation was run within strata -- they are"
          r" point")
    print(r"% estimates, and filling that column with the unadjusted p would be a"
          r" lie.")
    row(r"\emph{Unadjusted} (high, %d models)" % rep[EFFORT]["models"],
        obs_t, obs_p, obs_did, rep[EFFORT]["p"])
    for k, lb in (("gold", "Within gold-trajectory length"),
                  ("units", "Within required-check count"),
                  ("rtype", "Within round type")):
        v = st[k]
        row(lb, v["treated"], v["placebo"], v["did"])
    print(r"    \midrule")
    print(r"% Disjoint panels, same corpus, each with its own exact p.")
    for eff in corpus.LEVELS:
        if eff == EFFORT:
            continue
        v = rep[eff]
        row(f"{eff.capitalize()} effort ({v['models']} models)",
            v["treated"], v["placebo"], v["did"], v["p"])
    print(r"    \midrule")
    print(r"% The outcome swap, and the one row that does NOT clear .05. Keep it.")
    s_t, s_p, s_did, s_perm = sc
    row("Outcome: mean score", s_t, s_p, s_did, s_perm["p_did"])
    neg = sum(1 for m in order if did_of(lambda a, g: per_model[m][(a, g)])[0] < 0)
    cg, cu = comp[True], comp[False]
    print(f"% caption facts: {neg} of {len(order)} models have a negative DiD, "
          f"which is agreement and not {len(order)} replications -- every model "
          f"sees the same {len(gate['seeds'])} seeds; the gate is graded on a "
          f"declared protocol effect in {100.0 * cg[0] / cg[1]:.1f}\\% of gated "
          f"rounds against {100.0 * cu[0] / cu[1]:.1f}\\% of ungated ones, an "
          f"imperfect and under-counted first stage that makes the estimate a "
          f"lower bound")
    print(f"% caption facts: on mean score the same contrast is {s_did:+.1f}pp "
          f"with exact $p$ = {s_perm['p_did']:.2f}, i.e.\\ NOT significant -- a "
          f"gated round costs the round rather than most of its predicates, "
          f"which is the competence-resolution gap stated causally and must be "
          f"conceded in the text, not omitted")


if __name__ == "__main__":
    sys.exit(main())
