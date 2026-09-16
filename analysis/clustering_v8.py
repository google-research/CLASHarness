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

"""T2 -- do failures concentrate, and does that explain who wins?

A round passes only if EVERY required check passes. So two models with the same
per-check ability can land at very different pass rates depending on whether
their failures pile onto a few rounds (the rest come out clean, more rounds
pass) or spread one-per-round (few rounds pass). T1 shows the pass/score gap
exists; T2 asks what it is made of.

THE TRAP THIS TABLE IS BUILT AROUND
-----------------------------------
The obvious statistic -- lift against an independence null over the whole
corpus -- is confounded here, and the factory report says so at
build_behavior_report.py:605. The corpus spans a wide difficulty range, so
failures land on the hard rounds even for a model whose checks fail
independently, and every model comes out looking clustered. Whole-corpus lift
is above 1.0 for every model in the panel, and reading that as "these models
cluster" would be reading the corpus, not the models.

Two corrections, both from the factory:

  STRATIFIED   Rank rounds by the pooled required-check failure rate of the
               OTHER models (leave-one-out, so a model never defines the axis
               it is judged on), cut into terciles, recompute lift where rounds
               are of comparable difficulty.

  PAIRED       Resample rounds once per draw and recompute ALL models on that
               same resample, then take each model's lift minus its peers'.
               Corpus difficulty cancels. This is the only one of the three
               that supports a sentence about a model.

A second caveat that has to travel with the number: lift is bounded above by
1/q where q is the model's own failure rate. The ceiling ranges from about 1.5
to 2.7 across the panel, so lift RANKS are comparable but MAGNITUDES are not.
The table prints the ceiling next to the value.

WHAT THE PAPER IS ALLOWED TO CLAIM comes out of the bottom three blocks, not
the first one. The first block is printed anyway, because the report shows it
and a reader who compares the two documents must find the same numbers.
"""
import json
import os
import sys
import time

import corpus

# Matches the factory -- now by construction rather than by hand. This module
# reproduces stats.json and prints tables the appendix parses from the frozen
# report, so its draw count is fixed by those documents, NOT by corpus.DRAWS
# (10,000). Aliasing it here is what keeps the two from being bumped together
# by someone who reads the 4000 as an oversight. See corpus.FACTORY_DRAWS.
DRAWS = corpus.FACTORY_DRAWS
# "Same raw ability" window for the exemplar pair. The factory used 3.0pp,
# which at high effort admits pairs almost 3pp apart -- enough for the ability
# term to swamp the concentration term and turn the exemplar into an argument
# for the opposite conclusion. 1.0pp keeps the comparison honest; the full
# candidate list is printed so the cut can be second-guessed.
SAME_ABILITY_PP = 1.0
# Display name -> stats.json key. corpus.py owns every alias a
# model has; see the naming block there.
FACTORY_KEY = corpus.STATS_KEY
FACTORY_STATS = os.path.join(os.path.dirname(corpus.DESIGN_AXES),
                             "cross-model-behavioral-comparison.stats.json")


def verify_against_factory(eps_m, eff):
    """Whole-corpus clustering must reproduce stats.json at its own effort.

    POINT ESTIMATES are required to be exact -- they are pure sums over the
    same records, so any difference means a predicate is wrong.

    BOOTSTRAP INTERVALS are allowed to differ by one rounding step, and cannot
    be made exact. The resample draws an index into the task sequence, and the
    factory's sequence is its own ingestion order while ours is sorted by
    task_id (see corpus.required_vectors -- sorting is what makes OUR intervals
    reproducible off this machine). Same seed, same indices, different tasks
    behind them. The tolerance is one unit in the last reported place, and it
    is asserted, not waved at: a real predicate bug moves point estimates too,
    and a bootstrap that disagreed by more than rounding would mean the
    resample unit itself was different.
    """
    stats = json.load(open(FACTORY_STATS))
    assert stats["primary_effort"] == eff, (
        f"loaded the {eff} tree but the artifact is a "
        f"{stats['primary_effort']} build")
    bad, n, ci_off = [], 0, 0
    for label, skey in FACTORY_KEY.items():
        got = corpus.failure_clustering(
            list(corpus.required_vectors(eps_m[label]).values()), draws=DRAWS)
        want = stats[skey]["clustering"]
        for f in ("n_tasks", "n_checks", "mean_k", "per_check_pct",
                  "obs_pass_pct", "indep_pass_pct", "excess_pp", "lift",
                  "lift_ceiling", "overdispersion"):
            n += 1
            if got[f] != want[f]:
                bad.append(f"{label}.{f}: ours {got[f]} vs factory {want[f]}")
        for i, side in enumerate(("lo", "hi")):
            n += 1
            d = abs(got["lift_ci"][i] - want["lift_ci"][i])
            if d > 0.0101:
                bad.append(f"{label}.lift_ci.{side}: ours {got['lift_ci'][i]} "
                           f"vs factory {want['lift_ci'][i]} (delta {d:.3f} > "
                           f"one rounding step -- not a resample-order effect)")
            elif d > 1e-9:
                ci_off += 1
    assert not bad, ("clustering drifted from the factory report:\n  "
                     + "\n  ".join(bad))
    return n, ci_off


def decompose(kf, common, a, b, peers):
    """Split the a-b pass-rate gap into per-check ability and clustering.

    For a set of rounds, independence predicts a pass rate of mean_i(p**k_i)
    from the per-check rate alone. The part of the observed gap that prediction
    accounts for is the ABILITY component; whatever is left is what differing
    failure concentration bought. Computed pooled over the corpus and again
    within difficulty terciles, because the pooled version inherits exactly the
    confound described in the module docstring.
    """
    def rates(model, ids):
        ks = [kf[model][t][0] for t in ids]
        fs = [kf[model][t][1] for t in ids]
        p = 1 - sum(fs) / sum(ks)
        return {
            "pass": 100.0 * sum(1 for f in fs if f == 0) / len(ids),
            "indep": 100.0 * sum(p ** k for k in ks) / len(ids),
            "lift": corpus.co_failure_lift(list(zip(ks, fs))),
            "per_check": 100.0 * p,
        }

    def block(ids):
        ra, rb = rates(a, ids), rates(b, ids)
        gap = ra["pass"] - rb["pass"]
        pc = ra["indep"] - rb["indep"]
        return {"n": len(ids), "gap": gap, "per_check": pc,
                "clustering": gap - pc, "a": ra, "b": rb}

    # difficulty axis excludes BOTH members of the pair
    diff = {}
    for t in common:
        pk = sum(kf[m][t][0] for m in peers)
        pf = sum(kf[m][t][1] for m in peers)
        diff[t] = pf / pk if pk else 0.0
    order = sorted(common, key=lambda t: diff[t])
    n = len(order)
    tiers = [("easy", order[:n // 3]), ("middle", order[n // 3:2 * n // 3]),
             ("hard", order[2 * n // 3:])]
    out = {"pooled": block(common), "tiers": [], "n_peers": len(peers)}
    strat_cl = 0.0
    for name, ids in tiers:
        blk = block(ids)
        blk["name"] = name
        blk["weighted_clustering"] = blk["clustering"] * len(ids) / n
        strat_cl += blk["weighted_clustering"]
        out["tiers"].append(blk)
    out["strat_clustering"] = strat_cl
    out["strat_per_check"] = out["pooled"]["gap"] - strat_cl
    return out


def main():
    t0 = time.time()
    # The factory calibration stays on corpus.REPORT_MODELS on purpose: it
    # replicates stats.json, which has rows only for the models the frozen
    # report covers. Only the panel below grows. Note that the tree it reads is
    # WIDER than the set actually compared -- FACTORY_KEY selects the subset --
    # so the count reported must be the one that was checked, not the one that
    # was loaded.
    #
    # The effort comes from the artifact (see frozen_effort), so this stays a
    # replication check rather than becoming a cross-effort comparison the
    # moment the report is rebuilt at a different setting.
    feff = corpus.frozen_effort()
    eps_m, _ = corpus.load_eval(feff, keep_checks=True)
    n_fields, ci_off = verify_against_factory(eps_m, feff)
    print(f"factory clustering replication at {feff}: {n_fields} fields across "
          f"the {len(FACTORY_KEY)} models the frozen report contains "
          f"(of {len(eps_m)} loaded at {feff}) -- every point estimate EXACT; "
          f"{ci_off} "
          f"bootstrap CI endpoints differ by one rounding step (resample "
          f"order, see verify_against_factory) ({time.time()-t0:.0f}s)")

    eps_h, _ = corpus.load_panel(keep_checks=True)
    order = corpus.model_order(eps_h)
    vecs = {m: corpus.required_vectors(eps_h[m]) for m in order}
    whole = {m: corpus.failure_clustering(list(vecs[m].values()), draws=DRAWS)
             for m in order}
    kf, common = corpus.kf_table(vecs)

    print(f"\nT2  failure concentration   population={corpus.PRIMARY_POPULATION} "
          f"(n={corpus.N_RAW})   effort={corpus.PRIMARY_EFFORT}   "
          f"{len(common)} rounds graded for all {len(order)} models")

    # ---- block 1: the confounded view, printed because the report shows it --
    hdr = (f"\n{'model':<{corpus.NAME_W}} | {'perchk%':>7} {'obs%':>6} {'indep%':>7} "
           f"{'excess':>7} | {'lift':>5} {'95% CI':>13} {'ceil':>5} "
           f"{'l/ceil':>6} | {'odisp':>5}")
    print("WHOLE-CORPUS (confounded by difficulty spread -- see below, do not "
          "quote as a model property)")
    print(hdr)
    print("-" * (len(hdr) - 1))
    for m in order:
        w = whole[m]
        print(f"{m:<{corpus.NAME_W}} | {w['per_check_pct']:>7.1f} {w['obs_pass_pct']:>6.1f} "
              f"{w['indep_pass_pct']:>7.1f} {w['excess_pp']:>+7.1f} | "
              f"{w['lift']:>5.2f} "
              f"{'[%.2f, %.2f]' % tuple(w['lift_ci']):>13} "
              f"{w['lift_ceiling']:>5.2f} "
              f"{w['lift']/w['lift_ceiling']:>6.2f} | "
              f"{w['overdispersion']:>5.2f}")
    print(f"  every model reads above 1.0, which is the tell that this null is "
          f"measuring the corpus. Note lift/ceiling: {order[-1]} sits at "
          f"{whole[order[-1]]['lift']/whole[order[-1]]['lift_ceiling']:.2f} of "
          f"its own maximum, the HIGHEST of the {len(order)}, so its low raw lift of "
          f"{whole[order[-1]]['lift']:.2f} does not mean its failures are the "
          f"most independent.")

    # ---- block 2: stratified ------------------------------------------------
    print(f"\nSTRATIFIED by leave-one-out difficulty tercile "
          f"({DRAWS} bootstrap draws over rounds)")
    hdr = (f"{'model':<{corpus.NAME_W}} | {'easy':>20} {'middle':>20} {'hard':>20}")
    print(hdr)
    print("-" * len(hdr))
    strat = {}
    for i, m in enumerate(order):
        tiers = corpus.stratified_lift(kf, common, m, draws=DRAWS,
                                       seed=90000 + i * 10)
        strat[m] = tiers
        cells = [f"{t['lift']:.2f}[{t['ci'][0]:.2f},{t['ci'][1]:.2f}]"
                 f"{'*' if t['verdict'] != 'flat' else ' '}" for t in tiers]
        print(f"{m:<{corpus.NAME_W}} | {cells[0]:>20} {cells[1]:>20} {cells[2]:>20}")
    up_easy = sum(1 for m in order if strat[m][0]["verdict"] == "up")
    up_hard = sum(1 for m in order if strat[m][2]["verdict"] == "up")
    hard_lo = min(strat[m][2]["lift"] for m in order)
    hard_hi = max(strat[m][2]["lift"] for m in order)
    print(f"  * = 95% CI excludes 1.0")
    print(f"  {up_easy}/{len(order)} models cluster on the EASY tercile; {up_hard}/{len(order)} do on "
          f"the HARD tercile, where every model sits in "
          f"[{hard_lo:.2f}, {hard_hi:.2f}] -- indistinguishable from "
          f"independent failure.")
    print(f"  -> the shape is not 'failures cluster'. It is that on rounds the "
          f"field finds easy, a model that fails fails WHOLESALE, while on hard "
          f"rounds everyone bleeds units one at a time. Most of the "
          f"whole-corpus effect above was the corpus.")

    # ---- block 3: paired ----------------------------------------------------
    print(f"\nPAIRED against peers on the SAME resampled rounds "
          f"(the model-level statistic)")
    peer = corpus.paired_peer_lift(kf, common, draws=DRAWS)
    hdr = (f"{'model':<{corpus.NAME_W}} | {'lift':>5} {'peers':>6} {'diff':>7} "
           f"{'95% CI':>18} {'p':>7}  verdict")
    print(hdr)
    print("-" * len(hdr))
    for m in sorted(order, key=lambda x: -peer[x]["diff"]):
        v = peer[m]
        print(f"{m:<{corpus.NAME_W}} | {v['lift']:>5.2f} {v['peer_mean']:>6.2f} "
              f"{v['diff']:>+7.3f} "
              f"{'[%+.3f, %+.3f]' % v['ci']:>18} {v['p']:>7.4f}  "
              f"{v['verdict'].upper()}")
    ups = [m for m in order if peer[m]["verdict"] == "up"]
    downs = [m for m in order if peer[m]["verdict"] == "down"]
    print(f"  once difficulty cancels, {len(ups)}/{len(order)} cluster more than their "
          f"peers ({', '.join(ups) or 'none'}) and {len(downs)}/{len(order)} cluster less "
          f"({', '.join(downs) or 'none'}); the rest are flat. THIS is the "
          f"sentence the paper is entitled to.")

    # ---- block 3b: what actually separates these models? -------------------
    # The harness-first argument, computed rather than asserted. If pass rate
    # cannot tell the frontier apart but concentration can, then a leaderboard
    # is the wrong instrument and this paper has a reason to exist. If it turns
    # out the other way, the paper has to say so.
    pairs_all = [(a, b) for i, a in enumerate(order) for b in order[i + 1:]]
    alpha = corpus.bonferroni(len(pairs_all))
    pvals = {}
    for a, b in pairs_all:
        pa = {t: kf[a][t][1] == 0 for t in common}
        pb = {t: kf[b][t][1] == 0 for t in common}
        nb = sum(1 for t in common if pa[t] and not pb[t])
        nc = sum(1 for t in common if pb[t] and not pa[t])
        pvals[(a, b)] = corpus.exact_mcnemar(nb, nc)
    # THE SET USED TO BE order[:6] WHILE THE LABEL SAID "top 6 by pass rate".
    # order is model_order -- the provider-blocked display order -- so the two
    # agreed only for as long as the leading providers happened to occupy the
    # first six rows. gemini-3.7-flash ended that: it outranks gpt-5.6-luna by
    # 0.5 points but sorts after it, so the printed set stopped being the set
    # the label named. Rather than swap in rank_order[:6] and keep an arbitrary
    # cut, this now uses the one frontier definition the paper has --
    # corpus.leading_group, the models this same all-pairs test cannot separate
    # from the best -- so the label and the set are the same object.
    lead = corpus.leading_group(eps_h)
    top_pairs = [pr for pr in pairs_all if pr[0] in lead and pr[1] in lead]
    sig_top = [pr for pr in top_pairs if pvals[pr] < alpha]
    sep_lift = [m for m in order if peer[m]["verdict"] != "flat"]
    print(f"\nWHAT SEPARATES THESE MODELS")
    print(f"  pass rate, the {len(lead)} leading models: "
          f"{len(sig_top)}/{len(top_pairs)} "
          f"pairs differ significantly (exact McNemar, Bonferroni "
          f"alpha=0.05/{len(pairs_all)}={alpha:.5f})"
          + (f" -- {', '.join(a + '>' + b for a, b in sig_top)}" if sig_top else ""))
    print(f"  concentration: {len(sep_lift)}/{len(order)} models sit significantly off "
          f"their peer mean on the same rounds ({', '.join(sep_lift)})")
    print(f"  -> the frontier is nearly flat on the headline metric and not "
          f"flat at all on HOW the failures land. That gap is the argument for "
          f"grading units rather than rounds.")

    # ---- block 4: gap decomposition ----------------------------------------
    # The exemplar is supposed to isolate concentration from ability, so the
    # ability gap must be SMALL. The factory's 3.0pp window was tuned on the
    # medium sweep, where its chosen pair happened to sit 0.05pp apart; at high
    # effort the same window admits pairs ~3pp apart, where the ability term
    # dominates and the exemplar demonstrates nothing. Every admissible pair is
    # printed so the choice is auditable rather than asserted.
    # Ability and pass rate are measured over `common`, NOT over each model's
    # own task set. The whole-corpus columns are rounded to one decimal and
    # computed on slightly different denominators (a provider error costs one
    # model a round), so using them for a 1.0pp selection window would let
    # rounding decide which pairs are admissible.
    def _pc(m, ids):
        K = sum(kf[m][t][0] for t in ids)
        F = sum(kf[m][t][1] for t in ids)
        return 100.0 * (1 - F / K)

    def _pass(m, ids):
        return 100.0 * sum(1 for t in ids if kf[m][t][1] == 0) / len(ids)

    def gaps(a, b):
        return (abs(_pc(a, common) - _pc(b, common)),
                abs(peer[a]["lift"] - peer[b]["lift"]))

    # An exemplar needs BOTH constraints, and the second one is the one that
    # is easy to forget: the pass gap being decomposed has to exist. Splitting
    # a non-significant difference into an ability part and a concentration
    # part yields two precise numbers that describe noise.
    cands = [pr for pr in pairs_all if gaps(*pr)[0] <= SAME_ABILITY_PP]
    print(f"\nEXEMPLAR SELECTION: needs a SMALL ability gap (<= "
          f"{SAME_ABILITY_PP}pp) and a REAL pass gap to decompose")
    print(f"  {'pair':<25} {'ability':>8} {'lift gap':>9} {'pass gap':>9} "
          f"{'McNemar p':>10}  usable")
    usable = []
    for pr in sorted(cands, key=lambda p: -gaps(*p)[1]):
        ab, lg = gaps(*pr)
        pg = abs(_pass(pr[0], common) - _pass(pr[1], common))
        ok = pvals[pr] < alpha
        usable += [pr] if ok else []
        print(f"  {pr[0] + ' vs ' + pr[1]:<25} {ab:>7.2f}p {lg:>9.2f} "
              f"{pg:>8.1f}p {pvals[pr]:>10.4f}  {'yes' if ok else 'NO'}")
    near = [pr for pr in pairs_all if pvals[pr] < alpha]
    if not usable:
        print(f"  NO PAIR QUALIFIES. Every pair of comparable per-check ability "
              f"has a pass gap indistinguishable from zero, and all "
              f"{len(near)} significant pairs differ by more than "
              f"{SAME_ABILITY_PP}pp of ability (closest: "
              f"{min((pr for pr in near), key=lambda p: gaps(*p)[0])[0]} vs "
              f"{min((pr for pr in near), key=lambda p: gaps(*p)[0])[1]} at "
              f"{gaps(*min((pr for pr in near), key=lambda p: gaps(*p)[0]))[0]:.2f}pp).")
        print(f"  => the decomposition below is an INSTRUMENT DEMONSTRATION, "
              f"not a finding about these two models. The paper must label it "
              f"that way and print the p-value beside it.")

    # The behavioural report's own exemplar at medium was opus-5 vs sol (its
# names for them), and
    # the paper's draft claim came from it. It is re-run here at high whether
    # or not it qualifies, because a claim inherited from another effort
    # setting has to be re-tested rather than carried over.
    a, b = max(cands, key=lambda pr: (gaps(*pr)[1], pr))
    if peer[a]["lift"] < peer[b]["lift"]:
        a, b = b, a
    report_pair = (corpus.MODEL_LABEL["opus_5"], corpus.MODEL_LABEL["5_6_sol"])
    todo = [(a, b, f"widest lift gap at comparable ability, "
                   f"p={pvals[tuple(sorted((a, b), key=order.index))]:.4f}")]
    if {a, b} != set(report_pair):
        rp = tuple(sorted(report_pair, key=order.index))
        todo.append((report_pair[0], report_pair[1],
                     f"the report's medium exemplar, re-checked at high, "
                     f"p={pvals[rp]:.4f}"))

    for a, b, why in todo:
        peers = [m for m in order if m not in (a, b)]
        d = decompose(kf, common, a, b, peers)
        p = d["pooled"]
        print(f"\nGAP DECOMPOSITION: {a} vs {b}   [{why}]")
        print(f"  difficulty terciles defined by the other {d['n_peers']} models")
        print(f"  per-check   {a} {p['a']['per_check']:.2f}%  vs  "
              f"{b} {p['b']['per_check']:.2f}%   "
              f"({p['a']['per_check']-p['b']['per_check']:+.2f}pp)")
        print(f"  pass rate   {a} {p['a']['pass']:.2f}%  vs  "
              f"{b} {p['b']['pass']:.2f}%   (gap {p['gap']:+.2f}pp)")
        print(f"\n  {'tier':<8} {'n':>4} {'gap':>7} {'= ability':>10} "
              f"{'+ clustering':>13} {'wtd cl':>8} | {'lift a':>7} {'lift b':>7}")
        for t in d["tiers"]:
            print(f"  {t['name']:<8} {t['n']:>4} {t['gap']:>+7.2f} "
                  f"{t['per_check']:>+10.2f} {t['clustering']:>+13.2f} "
                  f"{t['weighted_clustering']:>+8.2f} | "
                  f"{t['a']['lift']:>7.2f} {t['b']['lift']:>7.2f}")
        print(f"  {'pooled':<8} {p['n']:>4} {p['gap']:>+7.2f} "
              f"{p['per_check']:>+10.2f} {p['clustering']:>+13.2f}")
        print(f"  {'strat':<8} {'':>4} {p['gap']:>+7.2f} "
              f"{d['strat_per_check']:>+10.2f} {d['strat_clustering']:>+13.2f}")
        # Sign-aware: the interesting case (clustering carries the lead) and
        # the opposite case (clustering eats an ability lead) are both real
        # outcomes, and the summary must read off the numbers, not a template.
        pc, cl = d["strat_per_check"], d["strat_clustering"]
        lead = "ahead" if p["gap"] > 0 else "behind"
        if cl > 0 and pc < 0:
            verdict = (f"{a} is {abs(pc):.2f}pp WORSE per check and still "
                       f"{p['gap']:+.2f}pp {lead}: the entire lead, and then "
                       f"some, is bought by concentrating failures.")
        elif cl > 0:
            verdict = (f"ability explains {pc:+.2f}pp of the {p['gap']:+.2f}pp "
                       f"gap and concentration adds {cl:+.2f}pp on top -- "
                       f"{cl/p['gap']*100:.0f}% of the lead is concentration.")
        else:
            verdict = (f"ability alone would predict {pc:+.2f}pp; {a} only "
                       f"manages {p['gap']:+.2f}pp because its failures are "
                       f"LESS concentrated than {b}'s ({cl:+.2f}pp). "
                       f"Concentration works against the stronger model here.")
        print(f"  -> {verdict}")
        # Note the pooled/stratified sign flip when it happens: it is the
        # confound from the module docstring showing up inside a single
        # comparison, and it is the clearest available demonstration that the
        # stratified split is not cosmetic.
        if (p["per_check"] > 0) != (pc > 0):
            print(f"     pooled and stratified DISAGREE IN SIGN on the ability "
                  f"term ({p['per_check']:+.2f} vs {pc:+.2f}). Pooling mixes "
                  f"easy and hard rounds, where the two models' relative "
                  f"strengths differ; the stratified split is the one to read.")
        pr_key = tuple(sorted((a, b), key=order.index))
        if pvals[pr_key] >= alpha:
            print(f"     CAVEAT, and it governs the sentence above: the "
                  f"{p['gap']:+.2f}pp pass gap is NOT significant "
                  f"(p={pvals[pr_key]:.4f} vs alpha={alpha:.5f}). The "
                  f"arithmetic is exact but it is partitioning a difference "
                  f"the data does not establish. Usable to show what the "
                  f"instrument does; NOT usable as a claim about {a} and {b}.")

    emit_body(order, whole, strat, peer,
              {m: (len(vecs[m]), len(eps_h[m]),
                   corpus.pass_pct(list(eps_h[m].values())),
                   sum(1 for x in vecs[m].values() if all(x)),
                   sum(1 for e in eps_h[m].values() if e["success"]))
               for m in order},
              len(common))
    print(f"\ntotal {time.time()-t0:.0f}s")
    return 0


def emit_body(order, whole, strat, peer, denoms, n_common):
    """The body table for section 6.2, blocked to match the argument.

    Three column groups, in the order the module docstring argues them: the
    confounded whole-corpus view (printed because the appendix reproduces it
    and a reader comparing the two documents must find the same numbers), the
    stratified correction showing the effect lives on the easy tercile, and the
    paired statistic, the only one of the three that supports a sentence about
    a model.

    Lift travels with its own ceiling inside one cell. Lift is bounded by 1/q
    and the ceiling spans 1.4 to 2.7 across the panel, so printing the two in
    separate columns invites a reader to compare magnitudes that are not
    comparable.

    No verdict word column. The tercile stars and the paired interval already
    carry the significance, and a column reading "up" only restates the
    interval in a form the reader cannot check.

    Two denominators are in play and the emitter derives a disclosure for
    both. The first four columns are whole-corpus, each model on its own graded
    rounds: a provider error yields an episode with zero graded checks, and
    such a round has no failure vector to place, so it cannot enter a
    concentration statistic at all. Dropping it is forced; printing a pass rate
    that silently disagrees with the main table by a tenth of a point is not.
    The last five columns compare models to each other, so they run on the
    intersection -- a round one model lost must leave the comparison for all of
    them -- and that denominator is the same for every row.
    """
    print(f"\n% ---- LaTeX body for T2 (paste into the tabular) ----")
    print(r"%      column spec: l cccc ccc cc")
    for i, (_prov, ms) in enumerate(corpus.group_by_provider(order)):
        if i:
            print(r"    \midrule")
        for m in ms:
            w, s, v = whole[m], strat[m], peer[m]
            tiers = [f"{t['lift']:.2f}"
                     + (r"\rlap{$^{*}$}" if t["verdict"] != "flat" else "")
                     for t in s]
            cells = ([f"{w['per_check_pct']:.1f}", f"{w['obs_pass_pct']:.1f}",
                      f"{w['indep_pass_pct']:.1f}",
                      f"{w['lift']:.2f} ({w['lift_ceiling']:.2f})"]
                     + tiers
                     + [f"${v['diff']:+.3f}$", "$[%+.3f, %+.3f]$" % v["ci"]])
            print(f"    \\texttt{{{m}}}" + " " * (18 - len(m))
                  + " & " + " & ".join(cells) + r" \\")

    # Facts the caption and the prose quote, derived here so they cannot drift
    # from the table above.
    up_e = sum(1 for m in order if strat[m][0]["verdict"] == "up")
    up_h = sum(1 for m in order if strat[m][2]["verdict"] == "up")
    off = [m for m in order if peer[m]["verdict"] != "flat"]
    print(f"% caption facts: whole-corpus lift "
          f"{min(whole[m]['lift'] for m in order):.2f}--"
          f"{max(whole[m]['lift'] for m in order):.2f} for all {len(order)}; "
          f"{up_e}/{len(order)} cluster on the easy tercile and "
          f"{up_h}/{len(order)} on the hard, where every model sits in "
          f"[{min(strat[m][2]['lift'] for m in order):.2f}, "
          f"{max(strat[m][2]['lift'] for m in order):.2f}]; "
          f"{len(off)}/{len(order)} sit off their peer mean")
    print(f"% rows follow the order of the main results table")

    full = sorted({d[1] for d in denoms.values()})
    short = [m for m in order if denoms[m][0] != denoms[m][1]]
    if not short:
        print(f"% denominator: all {len(order)} models graded on "
              f"{full[0]} rounds, same as the main table")
    else:
        parts = []
        for m in short:
            graded, total, headline, won_graded, won_all = denoms[m]
            # The clause below says "same successes over a smaller
            # denominator". That holds only while every dropped round is a
            # failure. Assert it on the counts themselves -- a check-less round
            # recorded as a success would break the sentence and nothing else
            # in this script would notice.
            assert won_graded == won_all, (
                f"{m}: {won_graded} successes among graded rounds but "
                f"{won_all} overall -- a round with no required checks "
                f"passed, so the dropped rounds are not all failures and the "
                f"denominator clause must be rewritten")
            parts.append(
                f"{m} on {graded} ({total - graded} provider-error round(s) "
                f"carry no graded check, so they have no failure vector to "
                f"place); its pass reads {whole[m]['obs_pass_pct']:.1f} here "
                f"against {headline:.1f} in the main table, the same "
                f"{won_all} successes over the smaller denominator")
        print(f"% denominator: {len(order) - len(short)} of {len(order)} "
              f"models graded on all {max(full)} rounds; " + "; ".join(parts))
    print(f"% denominator: the first four columns are each model on its own "
          f"graded rounds; the tercile and paired columns are all "
          f"{len(order)} models on the {n_common} rounds every model was "
          f"graded on, since those columns compare models to each other")


if __name__ == "__main__":
    sys.exit(main())
