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

"""T3 -- what the rubric actually measures, and what the judge actually reached.

This is the table the paper's honesty rests on. The abstract claims the harness
grades atomic units rather than whole transcripts; T3 is where that claim is
made checkable, including the parts that are inconvenient.

Three things it has to show.

1. THE RUBRIC IS MIXED, AND THE MIX IS STATED.
   A round's score is the fraction of its required checks that passed. Those
   units are not homogeneous: roughly two thirds are LLM semantic matches and
   one third are deterministic string/set conditions. Reporting one blended
   "score" without this split would let a reader assume either that everything
   is deterministic (it is not) or that everything is model-judged (it is not).

2. THE JUDGE DOES NOT ALWAYS ARRIVE.
   A semantic check on a run that produced no transcript has nothing to judge.
   The factory records that as `not_invoked` and, crucially, reports BOTH
   denominators: pass_pct over every check, graded_pass_pct over the checks a
   judge actually reached. For most models the two agree. For 3.5-flash at
   medium they are 24.6% and 49.3%, because half its semantic checks were never
   reached -- an artefact of exhausting the step budget, not of semantic
   ability. Publishing only pass_pct would report an ability difference that is
   really a truncation difference; publishing only graded_pass_pct would hide
   a real failure. The paper prints both and says which is which.

3. THE COMPONENT ABILITIES ARE NOT ONE ABILITY.
   Inside field_contains the factory already separates two questions:
     asset_gate  did the agent reach the right artefact at all
     content     did what it wrote carry the exact required string
   Across the API models these rank almost independently. That is the
   evidence for the paper's claim that a single pass/fail number compresses
   several distinct skills, and it is what makes the harness worth having.

DENOMINATORS AND ROLES -- read before changing anything
   The bucket predicates are corpus.check_bucket / check_subbucket /
   check_status, mirrored verbatim from build_behavior_report.py. They bucket
   checks of EVERY role, because that is what the factory does; corpus.py's
   self-test reproduces the shipped stats.json bit-for-bit on all 9 models
   the report contains (corpus.REPORT_MODELS -- NOT the current API panel,
   which is larger) x 5
   buckets at medium, which is the licence for recomputing at high here.
   Diagnostic (non-required) checks live entirely inside field_contains, so the
   semantic and other buckets are required-only either way; the script asserts
   this rather than trusting the note.

   The paper reports HIGH effort. The shipped stats.json is MEDIUM. Nothing is
   copied across -- the medium column below is recomputed and then checked
   against the factory file, and it is a hard failure if they disagree.
"""
import json
import os
import sys
from collections import defaultdict

import corpus

BUCKETS = ("semantic_match", "field_contains_asset_gate",
           "field_contains_content", "other")
# Everything decided without an LLM: the exact complement of semantic_match.
DETERMINISTIC = ("field_contains_asset_gate", "field_contains_content", "other")
# stats.json spells the asset sub-bucket without "_gate".
FACTORY_KEY = {"field_contains_asset_gate": "field_contains_asset"}


def tally(episodes):
    """bucket -> counts, using the factory's own predicates and precedence.

    Every field is kept twice: once over checks of EVERY role (the factory's
    denominator, which `verify_against_factory` reproduces bit-for-bit) and
    once over required checks only, prefixed `req_`. The paper publishes the
    required-only view because a round's score divides by required checks;
    the all-role view exists so the replication gate still has something to
    replicate. See the DENOMINATORS note in the module docstring.
    """
    out = {b: defaultdict(int) for b in BUCKETS + ("field_contains",)}
    for e in episodes:
        for c in (e["checks"] or []):
            st = corpus.check_status(c)
            sub = corpus.check_subbucket(c)
            names = [corpus.check_bucket(c)]
            if sub:
                names.append(f"field_contains_{sub}")
            required = c.get("role") == "required"
            for b in names:
                d = out[b]
                d["total"] += 1
                d[st] += 1
                if required:
                    d["required"] += 1
                    d["req_" + st] += 1
    for d in out.values():
        d["graded"] = d["total"] - d["infra"] - d["not_invoked"]
        d["passed"] = d["graded_pass"]
        d["req_graded"] = d["required"] - d["req_infra"] - d["req_not_invoked"]
        d["req_passed"] = d["req_graded_pass"]
    return out


def pct(num, den):
    return 100.0 * num / den if den else float("nan")


def spearman(xs, ys):
    """Rank correlation with average ranks for ties. No scipy on this box."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx and vy else float("nan")


def verify_against_factory(medium_eps):
    """Recompute at medium and demand bit-equality with the shipped stats.

    corpus.py already does this, but T3 is the table that publishes these
    specific fields, so it re-checks the exact fields it prints rather than
    trusting that some other module checked something similar.
    """
    path = os.path.join(os.path.dirname(corpus.DESIGN_AXES),
                        "cross-model-behavioral-comparison.stats.json")
    stats = json.load(open(path))
    key = corpus.STATS_KEY
    bad = []
    for label, skey in key.items():
        got = tally(medium_eps[label].values())
        want = stats[skey]["check_split"]
        for b in BUCKETS + ("field_contains",):
            w = want[FACTORY_KEY.get(b, b)]
            g = got[b]
            for f in ("total", "passed", "not_invoked", "graded"):
                if g[f] != w[f]:
                    bad.append(f"{label}.{b}.{f}: ours {g[f]} vs factory {w[f]}")
            for f, num, den in (("pass_pct", g["passed"], g["total"]),
                                ("graded_pass_pct", g["passed"], g["graded"])):
                if abs(round(pct(num, den), 1) - w[f]) > 0.051:
                    bad.append(f"{label}.{b}.{f}: ours {pct(num,den):.1f} "
                               f"vs factory {w[f]}")
    assert not bad, "check_split drifted from the factory report:\n  " + "\n  ".join(bad)
    return len(key) * (len(BUCKETS) + 1)


def main():
    eps_h, err_h = corpus.load_eval("high", keep_checks=True)
    # The replication tree is whichever effort the frozen artifact holds --
    # never a literal. See corpus.frozen_effort for what naming one cost.
    feff = corpus.frozen_effort()
    eps_m, _ = corpus.load_eval(feff, keep_checks=True)

    n_cells = verify_against_factory(eps_m)
    print(f"factory check_split replication at {feff}: EXACT on {n_cells} "
          f"model x bucket cells (total/passed/not_invoked/graded and both "
          f"pass percentages)")

    order = corpus.model_order(eps_h)
    tab = {m: tally(eps_h[m].values()) for m in order}

    # ---- the rubric's composition, and the role claim, asserted not assumed --
    # The check inventory is a property of the corpus, not of the solver, so
    # every model that produced no provider errors must face byte-identical
    # bucket totals. If that ever stops holding, some model is being graded
    # against a different rubric and none of this table is comparable.
    ref = tab[order[0]]
    clean_models = [m for m in order if err_h.get(m, 0) == 0]
    assert order[0] in clean_models, "reference model has error rows"
    for m in order:
        t = tab[m]
        for b in ("semantic_match", "other"):
            assert t[b]["required"] == t[b]["total"], (
                f"{m}: {b} has {t[b]['total'] - t[b]['required']} non-required "
                f"checks; T3 says diagnostics live only in field_contains")
        if m in clean_models:
            for b in BUCKETS:
                assert t[b]["total"] == ref[b]["total"], (
                    f"{m} sees {t[b]['total']} {b} checks but {order[0]} sees "
                    f"{ref[b]['total']}; the rubric is not identical across "
                    f"models and the table is not comparable")
    req_total = {m: sum(tab[m][b]["required"] for b in BUCKETS) for m in order}
    diag = {m: tab[m]["field_contains"]["total"] - tab[m]["field_contains"]["required"]
            for m in order}

    print(f"\nT3  rubric composition and judge reach   population="
          f"{corpus.PRIMARY_POPULATION} (n={corpus.N_RAW})   effort="
          f"{corpus.PRIMARY_EFFORT}")
    print(f"\nthe scored rubric, per model-run (checks are fixed by the corpus, "
          f"so these are identical across models except where a provider error "
          f"cost a round its checks):")
    print(f"  {'bucket':<26} {'checks':>7} {'required':>9} {'share of required':>18}")
    for b in BUCKETS:
        r = ref[b]["required"]
        print(f"  {b:<26} {ref[b]['total']:>7} {r:>9} "
              f"{pct(r, req_total[order[0]]):>17.1f}%")
    print(f"  {'-- required total':<26} {'':>7} {req_total[order[0]]:>9}"
          f"   ({diag[order[0]]} diagnostic checks, all inside field_contains)")
    sem = ref["semantic_match"]["required"]
    print(f"  -> {pct(sem, req_total[order[0]]):.1f}% of scored units are LLM "
          f"semantic matches, {pct(req_total[order[0]] - sem, req_total[order[0]]):.1f}% "
          f"are deterministic conditions")

    # ---- the main table -----------------------------------------------------
    hdr = (f"\n{'model':<{corpus.NAME_W}} | {'sem pass%':>9} {'graded%':>8} {'notinv%':>8} | "
           f"{'asset%':>7} {'content%':>8} {'delta':>7} | {'other%':>7}")
    print(hdr)
    print("-" * (len(hdr) - 1))
    for m in order:
        t = tab[m]
        s, a, c, o = (t["semantic_match"], t["field_contains_asset_gate"],
                      t["field_contains_content"], t["other"])
        d = pct(c["passed"], c["total"]) - pct(a["passed"], a["total"])
        print(f"{m:<{corpus.NAME_W}} | {pct(s['passed'], s['total']):>9.1f} "
              f"{pct(s['passed'], s['graded']):>8.1f} "
              f"{pct(s['not_invoked'], s['total']):>8.1f} | "
              f"{pct(a['passed'], a['total']):>7.1f} "
              f"{pct(c['passed'], c['total']):>8.1f} "
              f"{d:>+7.1f} | {pct(o['passed'], o['total']):>7.1f}")

    # ---- claim 2: attrition is a truncation artefact, not a judge artefact ---
    print("\njudge reach: semantic checks the judge never got to")
    att = {m: pct(tab[m]["semantic_match"]["not_invoked"],
                  tab[m]["semantic_match"]["total"]) for m in order}
    worst = max(order, key=lambda m: att[m])
    best = min(order, key=lambda m: att[m])
    w = tab[worst]["semantic_match"]
    print(f"  every model loses some: {att[best]:.1f}% ({best}) to "
          f"{att[worst]:.1f}% ({worst}). No cell is attrition-free, so the "
          f"correction has to be reported for all {len(order)}, not just the "
          f"outlier.")
    print(f"  {worst} is the extreme: {w['not_invoked']}/{w['total']} semantic "
          f"checks never reached, which is why its two denominators read "
          f"{pct(w['passed'], w['total']):.1f}% (of all) and "
          f"{pct(w['passed'], w['graded']):.1f}% (of judged)")
    caps = {m: 100.0 * sum(e["cap_hit"] for e in eps_h[m].values()) / len(eps_h[m])
            for m in order}
    rho_cap = spearman([caps[m] for m in order], [att[m] for m in order])
    print(f"  it is truncation, not semantics: Spearman rho between cap-hit "
          f"rate and not-reached rate = {rho_cap:+.2f} (n={len(order)})")

    # Which denominator you publish changes the ordering. This is the concrete
    # reason the paper prints both instead of picking one.
    by_all = sorted(order, key=lambda m: -pct(tab[m]["semantic_match"]["passed"],
                                              tab[m]["semantic_match"]["total"]))
    by_judged = sorted(order, key=lambda m: -pct(tab[m]["semantic_match"]["passed"],
                                                 tab[m]["semantic_match"]["graded"]))
    moved = [(m, by_all.index(m) + 1, by_judged.index(m) + 1) for m in order
             if by_all.index(m) != by_judged.index(m)]
    if moved:
        print(f"  the choice of denominator re-ranks {len(moved)}/{len(order)} models "
              f"(rank over all checks -> rank over judged checks):")
        for m, a, b in sorted(moved, key=lambda t: -abs(t[1] - t[2]))[:4]:
            print(f"    {m:<{corpus.NAME_W}} #{a} -> #{b}")
        print(f"  -> reporting only the all-checks rate would sell a truncation "
              f"artefact as a semantic-ability gap; reporting only the judged "
              f"rate would hide a real failure to finish. Both, always.")

    # ---- claim 3: the two component abilities rank independently ------------
    # Required-only, matching the published table. The 476 diagnostic checks
    # all land in asset_gate, so using the all-role denominator here would
    # print a different asset rate than the table two screens up.
    xs = [pct(tab[m]["field_contains_asset_gate"]["req_passed"],
              tab[m]["field_contains_asset_gate"]["required"]) for m in order]
    ys = [pct(tab[m]["field_contains_content"]["req_passed"],
              tab[m]["field_contains_content"]["required"]) for m in order]
    rho = spearman(xs, ys)
    # The outlier is named through corpus.MODEL_LABEL, not spelled out: a
    # literal that stops matching leaves this line silently keeping all
    # every model and reporting the drop-one figure as if it had dropped one.
    outlier = corpus.MODEL_LABEL["3_5_flash"]
    keep = [i for i, m in enumerate(order) if m != outlier]
    assert len(keep) == len(order) - 1, f"{outlier} is not in {order}"
    rho_no = spearman([xs[i] for i in keep], [ys[i] for i in keep])
    print(f"\nreaching the artefact and transcribing it are different abilities")
    print(f"  Spearman rho(asset_gate, content) = {rho:+.2f} over "
          f"{len(order)} models, "
          f"{rho_no:+.2f} with {outlier} dropped. Both are weak, but dropping "
          f"the outlier RAISES the correlation, so the near-zero value is not "
          f"produced by one extreme point -- the ranks simply do not transfer. "
          f"With n={len(order)} this is an effect size and not a test; the "
          f"paper says "
          f"'ranks barely transfer', never 'no relationship'.")
    # The near-universal direction, and its 13x spread, is the sturdier claim:
    # it is about every model rather than an anecdote about two.
    delta = {m: ys[i] - xs[i] for i, m in enumerate(order)}
    neg = [m for m in order if delta[m] < 0]
    lo = min(order, key=lambda m: delta[m])
    hi = max(order, key=lambda m: delta[m])
    print(f"  the drop from 'found it' to 'wrote it correctly' is near-"
          f"universal ({len(neg)}/{len(order)} models) but its SIZE is not a "
          f"constant: "
          f"{delta[hi]:+.1f}pp ({hi}) to {delta[lo]:+.1f}pp ({lo}), a "
          f"{abs(delta[lo] / delta[sorted(neg, key=lambda m: -delta[m])[0]]):.0f}x "
          f"spread across the models that drop at all")
    for m in (hi, lo):
        i = order.index(m)
        print(f"    {m:<{corpus.NAME_W}} asset {xs[i]:5.1f}  content {ys[i]:5.1f}  "
              f"delta {delta[m]:+6.1f}")

    # ---- claim 4: the judge reproduces the deterministic ranking ------------
    # This is the WEAK half of the answer to "is the judge measuring the models
    # or itself?" and it is labelled as such. Compare the LLM bucket against
    # everything that is NOT LLM-decided -- the three deterministic buckets
    # pooled, which is exactly the complement of the semantic bucket.
    #
    # REQUIRED-ONLY, and that is not cosmetic. Pooling the 476 diagnostic
    # asset checks (which carry no score) used to drag this from +0.90 to
    # +0.83 and made gemini-3.7-flash look like a lone outlier: dropping it
    # recovered +0.93 while dropping anything else left ~0.80. That "outlier"
    # was an artefact of the diagnostic padding, not a property of the model.
    # The diagnostics sit entirely in asset_gate, a bucket that runs 78-93%
    # for everyone and therefore separates nobody, so mixing them in was
    # adding noise to the instrument. On the scored denominator the leave-one-
    # out range is +0.87..+0.93 and no single model moves it much.
    #
    # Pooled rather than the `other` bucket alone: `other` is 545 of the 1147
    # scored deterministic checks, and half the evidence is a noisier
    # instrument than all of it. Reported both ways so the choice is visible.
    det_num = lambda m: sum(tab[m][b]["req_passed"] for b in DETERMINISTIC)
    det_den = lambda m: sum(tab[m][b]["required"] for b in DETERMINISTIC)
    det = [pct(det_num(m), det_den(m)) for m in order]
    sem_all = [pct(tab[m]["semantic_match"]["req_passed"],
                   tab[m]["semantic_match"]["required"]) for m in order]
    sem_jdg = [pct(tab[m]["semantic_match"]["req_passed"],
                   tab[m]["semantic_match"]["req_graded"]) for m in order]
    rho_all = spearman(sem_all, det)
    rho_jdg = spearman(sem_jdg, det)
    rho_jdg_no = spearman([sem_jdg[i] for i in keep], [det[i] for i in keep])
    rho_other = spearman(sem_jdg, [pct(tab[m]["other"]["req_passed"],
                                       tab[m]["other"]["required"]) for m in order])
    loo = []
    for i in range(len(order)):
        k2 = [j for j in range(len(order)) if j != i]
        loo.append(spearman([sem_jdg[j] for j in k2], [det[j] for j in k2]))
    print(f"\nis the judge measuring the models or itself?")
    print(f"  Spearman rho(LLM semantic, pooled deterministic) over "
          f"{len(order)} models = "
          f"{rho_jdg:+.2f} on the judged denominator, {rho_jdg_no:+.2f} with "
          f"{outlier} dropped. The LLM bucket reproduces the ranking that "
          f"{det_den(order[0])} scored deterministic checks produce on their "
          f"own.")
    print(f"  leave-one-out range {min(loo):+.2f}..{max(loo):+.2f} "
          f"(worst single model: "
          f"{order[min(range(len(order)), key=lambda i: loo[i])]}), so no one "
          f"model is carrying it.")
    print(f"  the same correlation on the ALL-checks denominator is "
          f"{rho_all:+.2f}. Correcting for unreached checks RAISES agreement "
          f"with the deterministic bucket ({rho_all:+.2f} -> {rho_jdg:+.2f}), "
          f"which is independent support for claim 2: what the all-checks "
          f"denominator removes is truncation, not semantic ability.")
    print(f"  against the `other` bucket alone ({tab[order[0]]['other']['required']} "
          f"checks) it is {rho_other:+.2f}; pooling all "
          f"{det_den(order[0])} scored deterministic checks is the less noisy "
          f"comparison and is the one the paper quotes.")
    print(f"  n={len(order)}, so this is an effect size and not a test. It also "
          f"compares CHECK-level pass rates while the headline metric is a "
          f"TASK-level conjunction, and the two deterministic sub-buckets "
          f"themselves rank near-independently -- so this number bounds "
          f"nothing on its own. The paper says 'tracks', never 'validates', "
          f"and rests the validity claim on the two sections below instead.")

    jr = judge_reliability(order, eps_h, tab)

    if err_h:
        print(f"\nnote: provider-error rows retained in this population contribute "
              f"no checks -- {dict(err_h)} at {corpus.PRIMARY_EFFORT}, which is "
              f"why that model's bucket totals sit slightly below the others")

    emit_body(order, tab, ref, req_total, by_all, by_judged, det,
              dict(rho_cap=rho_cap, rho_all=rho_all, rho_jdg=rho_jdg,
                   rho_jdg_no=rho_jdg_no, rho_asset_content=rho,
                   n_moved=len(moved), att_lo=att[best], att_hi=att[worst],
                   best=best, worst=worst, outlier=outlier,
                   loo_lo=min(loo), loo_hi=max(loo), **jr))
    return 0


def judge_reliability(order, eps_h, tab):
    """The claim that actually carries weight: how noisy is the judge, and
    does the paper's ordering survive the noise?

    Two facts, both measured on ~19k checks rather than on ten models.

    A. THE JUDGE IS AN ENSEMBLE AND IT MOSTLY AGREES WITH ITSELF.
       Every semantic verdict is a 3-sample self-consistency vote whose tally
       the harness records. Unanimous (0/3 or 3/3) means the judge reproduced
       its own verdict under identical input; split (1/3, 2/3) means it did
       not. The split rate is a direct, per-check measurement of judge noise
       -- no proxy, no rank correlation, no n=10.

    B. THE ORDERING SURVIVES THE WORST CASE.
       Take every split verdict and flip it. Not resample it -- flip it, all
       at once, in the direction that hurts most. If the leaderboard is
       unchanged under that, then the judge's residual uncertainty cannot be
       what produced the leaderboard, and this holds without assuming the
       judge is CORRECT about anything.

       The adversarial variant is stronger still: fail every split check for
       one model while passing every split check for all its rivals. That is
       not a scenario anyone believes; it is the bound.
    """
    def votes(m):
        eps = eps_h[m].values()
        una = split = 0
        agree = 0
        for e in eps:
            for c in corpus.scored_checks(e):
                v = corpus.judge_votes(c)
                if v is None:
                    continue
                if corpus.judge_split(c):
                    split += 1
                else:
                    una += 1
                if bool(c.get("passed")) == (v[0] * 2 > v[1]):
                    agree += 1
        return una, split, agree

    print("\nhow reliable is the judge itself? (self-consistency ensemble)")
    tot_una = tot_split = tot_agree = 0
    rows = []
    for m in order:
        u, s, a = votes(m)
        rows.append((m, u, s, a))
        tot_una += u
        tot_split += s
        tot_agree += a
    n_tagged = tot_una + tot_split
    assert tot_agree == n_tagged, (
        f"the recorded verdict disagrees with its own majority vote on "
        f"{n_tagged - tot_agree} checks; the aggregation rule is not a "
        f"majority and every flip bound below is invalid")
    print(f"  every semantic verdict is a 3-sample vote; {n_tagged} of them "
          f"carry a recorded tally across {len(order)} models")
    print(f"  unanimous {tot_una} ({pct(tot_una, n_tagged):.2f}%)   "
          f"split {tot_split} ({pct(tot_split, n_tagged):.2f}%)")
    print(f"  the stored pass/fail equals the majority vote on all {n_tagged} "
          f"-- the aggregation rule is a clean majority with no hidden logic")
    hi = max(rows, key=lambda r: pct(r[2], r[1] + r[2]))
    lo = min(rows, key=lambda r: pct(r[2], r[1] + r[2]))
    print(f"  per-model split rate runs {pct(lo[2], lo[1]+lo[2]):.2f}% ({lo[0]}) "
          f"to {pct(hi[2], hi[1]+hi[2]):.2f}% ({hi[0]}): no model is being "
          f"graded more ambiguously than the rest")

    # ---- B: flip every split verdict ---------------------------------------
    def success(m, mode, target=None):
        n = 0
        for e in eps_h[m].values():
            req = corpus.scored_checks(e)
            if not req:
                n += bool(e["success"])
                continue
            ok = True
            for c in req:
                p = bool(c.get("passed"))
                if mode != "base" and corpus.judge_split(c):
                    p = (m != target) if mode == "adv" else (mode == "pass")
                if not p:
                    ok = False
                    break
            n += ok
        return pct(n, len(eps_h[m]))

    base = {m: success(m, "base") for m in order}
    af = {m: success(m, "fail") for m in order}
    ap = {m: success(m, "pass") for m in order}
    rank = lambda d: {m: i + 1 for i, m in
                      enumerate(sorted(order, key=lambda x: -d[x]))}
    rb, ra, rp = rank(base), rank(af), rank(ap)
    swing = max(ap[m] - af[m] for m in order)
    move = max(max(abs(rb[m] - ra[m]), abs(rb[m] - rp[m])) for m in order)
    print("\ndoes the ordering survive the judge's own uncertainty?")
    print(f"  {'model':<{corpus.NAME_W}} {'base':>6} {'allFAIL':>8} "
          f"{'allPASS':>8} {'swing':>6}  rank")
    for m in sorted(order, key=lambda x: -base[x]):
        print(f"  {m:<{corpus.NAME_W}} {base[m]:6.1f} {af[m]:8.1f} "
              f"{ap[m]:8.1f} {ap[m]-af[m]:6.1f}  "
              f"{rb[m]:>2}/{ra[m]:>2}/{rp[m]:>2}")
    print(f"  flipping ALL {tot_split} split verdicts at once moves task "
          f"success by at most {swing:.2f}pp and any rank by at most {move} "
          f"position(s)")

    adv = {}
    for t in order:
        d = {m: success(m, "adv", target=t) for m in order}
        adv[t] = (d[t], sorted(order, key=lambda x: -d[x]).index(t) + 1)
    worst = max(order, key=lambda m: adv[m][1] - rb[m])
    top = sorted(order, key=lambda m: -base[m])[0]
    print(f"  adversarial (fail every split check for one model, pass every "
          f"split check for all rivals): {top} still ranks #{adv[top][1]}; "
          f"largest drop anywhere is {worst} #{rb[worst]} -> #{adv[worst][1]}")
    print(f"  -> the judge's residual uncertainty is {pct(tot_split, n_tagged):.1f}% "
          f"of semantic verdicts and cannot produce the leaderboard. This is "
          f"a bound, not a correlation, and it does not assume the judge is "
          f"right about anything.")
    return dict(n_tagged=n_tagged, unanimous=tot_una, split=tot_split,
                split_pct=pct(tot_split, n_tagged), swing=swing, move=move,
                adv_top_rank=adv[top][1], lo=lo[0], hi=hi[0],
                lo_pct=pct(lo[2], lo[1]+lo[2]), hi_pct=pct(hi[2], hi[1]+hi[2]))


def emit_body(order, tab, ref, req_total, by_all, by_judged, det, facts):
    """The body table for the validity section.

    Two column groups matching the two things a reviewer needs to accept the
    rubric: what the LLM judge did, and what it did relative to the checks no
    LLM touched.

    EVERY percentage here is over REQUIRED checks, matching the composition
    shares in the caption and matching what a round's score actually divides
    by. It did not always be so: the table used to print the factory's
    all-role denominator, which padded the asset gate from 346 to 822 with
    diagnostic checks that carry no score. That made the table's deterministic
    total read 1623 against a prose sentence about 3284 required checks -- two
    denominators a reader cannot reconcile -- and it corrupted the headline
    correlation (see the note at claim 4). The all-role view still exists in
    `tally` and `verify_against_factory` still reproduces the factory on it;
    it is simply not what gets published.

    The rank column earns its width: it is the whole argument for printing two
    semantic denominators instead of picking one. A reader who sees only the
    percentages can tell they differ but not that the difference reorders the
    field.
    """
    print("\n% ---- LaTeX body for T3 (paste into the tabular) ----")
    print(r"%      column spec: l cccc cccc")
    ra = {m: by_all.index(m) + 1 for m in order}
    rj = {m: by_judged.index(m) + 1 for m in order}
    for i, (_prov, ms) in enumerate(corpus.group_by_provider(order)):
        if i:
            print(r"    \midrule")
        for m in ms:
            t = tab[m]
            s, a, c = (t["semantic_match"], t["field_contains_asset_gate"],
                       t["field_contains_content"])
            rank = ("--" if ra[m] == rj[m]
                    else f"{ra[m]}$\\,\\to\\,${rj[m]}")
            av = pct(a["req_passed"], a["required"])
            cv = pct(c["req_passed"], c["required"])
            cells = [f"{pct(s['req_passed'], s['required']):.1f}",
                     f"{pct(s['req_passed'], s['req_graded']):.1f}",
                     f"{pct(s['req_not_invoked'], s['required']):.1f}",
                     rank,
                     f"{av:.1f}",
                     f"{cv:.1f}",
                     f"${cv - av:+.1f}$",
                     f"{det[order.index(m)]:.1f}"]
            print(f"    \\texttt{{{m}}}" + " " * (18 - len(m))
                  + " & " + " & ".join(cells) + r" \\")

    sem_req = ref["semantic_match"]["required"]
    tot_req = req_total[order[0]]
    det_checks = sum(ref[b]["required"] for b in DETERMINISTIC)
    assert sem_req + det_checks == tot_req, (
        f"published denominators do not add up: {sem_req} semantic + "
        f"{det_checks} deterministic != {tot_req} required")
    print(f"%      denominator row: {sem_req} & varies & "
          f"{sem_req} & -- & "
          f"{ref['field_contains_asset_gate']['required']} & "
          f"{ref['field_contains_content']['required']} & -- & {det_checks}")
    print(f"% caption facts: {pct(sem_req, tot_req):.1f}% of the {tot_req} "
          f"required units are LLM semantic matches ({sem_req}) and "
          f"{pct(det_checks, tot_req):.1f}% are deterministic ({det_checks}); "
          f"{facts['n_tagged']} semantic verdicts carry a self-consistency "
          f"tally, {facts['split_pct']:.1f}% of them split; flipping every "
          f"split verdict moves task success by <={facts['swing']:.1f}pp and "
          f"any rank by <={facts['move']}; "
          f"unreached semantic checks run {facts['att_lo']:.1f}% "
          f"({facts['best']}) to {facts['att_hi']:.1f}% ({facts['worst']}) and "
          f"correlate with cap-hit at rho={facts['rho_cap']:+.2f}; the "
          f"denominator reorders {facts['n_moved']}/{len(order)} models; "
          f"rho(LLM, deterministic)={facts['rho_jdg']:+.2f} judged "
          f"(leave-one-out {facts['loo_lo']:+.2f}..{facts['loo_hi']:+.2f}) and "
          f"{facts['rho_all']:+.2f} over all checks; "
          f"rho(asset, content)={facts['rho_asset_content']:+.2f}")
    print(f"% rows follow the order of the main results table")


if __name__ == "__main__":
    sys.exit(main())
