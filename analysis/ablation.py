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

"""Budget and modality ablations: the two arms that re-run corpus A itself.

The other two ablations change the corpus (generator_ablation.py) or the dial
on the model (the effort sweep, whose exhibits live in the appendix). These two
change the HARNESS and hold everything else fixed, which is what makes them
paired at the round level: every arm below scores the same 600 task
identifiers, so each model is compared against itself round by round and the
between-task variance that dominates a corpus this heterogeneous drops out.
Exact McNemar is therefore the test, not a bootstrap over rounds.

  BUDGET    the step budget the solver is given, 20 (shipped) / 30 / 40.
            The corpus is untouched: same tasks, same gold trajectories, same
            grader. Only the cap moves. 4 models.
  MODALITY  the rendered assets are withheld from the solver ("blind"); the
            task text is unchanged. 3 models.

WHY THE TWO ARMS HAVE DIFFERENT ROSTERS. gemini-3.5-flash was run at all three
budgets but never blind. It is the row where the budget effect is largest
(+10.0 points, and it is still capping out on 44% of rounds at b=40), so
dropping it to make the two arms symmetric would delete the single most
informative cell in the table. The table is therefore panelled, each panel
carries its own roster, and the caption says so. Do not "fix" this by
averaging the two panels into one arm-level number: they are not over the same
models.

!! THE MODALITY ARM NEEDS THE ASSET-GATE SPLIT OR IT IS CIRCULAR !!
346 of the 600 rounds carry a required check of effect_type "assets_accessed"
-- the solver is graded on having opened the asset. Per the harness comment at
scripts/run_effort_panel.sh:117-120, that check is satisfied by CALLING
`asset open`, which still succeeds blind, so it is not an automatic zero; but
it is still a check about touching an asset, and letting it into a headline
"removing images costs 126 rounds" would be partly restating the gate. The
strict quantity is the other 254 rounds, where no required check mentions
asset access at all. Both strata are reported and the prose leads with the
254.

DO NOT CALL THE 254 STRATUM "TEXT-SOLVABLE". That asserts they can be solved
without pixels, which is exactly the thing under test and is NOT something
this split establishes -- all it establishes is that no GRADING CHECK demands
asset access. An earlier draft of this file used that name; it overclaimed.
The honest label is "no asset check", and the finding is stronger stated
honestly: on rounds where the grader never asks whether the agent opened the
asset, the blind arm still keeps only 11-13% of its sighted pass rate.

!! "required" MEANS TWO DIFFERENT THINGS ON THE TWO SIDES OF THIS CORPUS !!
Both sides hold the same 3,760 checks and disagree completely about roles:

  spec side  ground_truth.evaluation_checks in specs.json
             346 required / 3,414 diagnostic, and every one of the 346 is an
             assets_accessed check. This is what corpus.load_specs() counts
             as `units`.
  eval side  metadata.check_results on the episode
             3,284 required / 476 diagnostic, and effect_type is dropped
             (None on every row). This is what corpus.scored_checks() returns
             and what Table~\\ref{tab:main}'s "required per round" counts.

So `units` from load_specs and len(scored_checks(ep)) are NOT the same
quantity despite both being called required, and effect_type is unavailable on
the eval side -- which is why corpus.ASSET_GATE_PREFIX exists: on the eval
side the gate is identifiable only by the check NAME, "effect:assets_accessed
<condition>". Verified here: the two definitions select the IDENTICAL 346
rounds, exactly one gate check each, and success == (all required eval checks
passed) on 600/600. asset_gated() asserts the first of those every run.

ONE ERROR ROW IS RETAINED AS A FAILURE. budget40/gemini-3.5-flash has a single
504 DEADLINE_EXCEEDED row (payment_chargeback_dispute_v03_r09, steps 0). Under
corpus.PRIMARY_POPULATION == "full" error rows stay in the 600 and count as
failures, which is exactly what Table~\\ref{tab:main} does with its five. The
caption states it. Do not silently drop it -- that would make b=40 a 599-round
arm compared against 600-round arms.

CACHING TRAP: results_facts.run() keys on the mtime of the script it calls.
This module is imported BY results_facts, so editing it does not invalidate a
cached run of anything else. Use `--fresh` after touching it.

USAGE
    python3 ablation.py          # facts panel + the LaTeX body for tab:ablation
"""
import decimal
import glob
import itertools
import json
import os
import sys

import corpus

# The eval trees for every arm, fetched by tools/download_data.py. Moves with
# CLASHARNESS_DATA_ROOT via corpus.
EVALS = os.path.join(corpus.DATA_ROOT, "eval_results")

# arm key -> (display label, eval tree, roster)
#
# The roster is passed to corpus.load_eval, which checks it by SET EQUALITY.
# A cell that failed to produce output therefore aborts here rather than
# quietly shrinking the panel.
BUDGETS = [
    (20, f"{EVALS}/data_3_6_flash_v8"),
    (30, f"{EVALS}/data_3_6_flash_v8_budget30"),
    (40, f"{EVALS}/data_3_6_flash_v8_budget40"),
]
BLIND_TREE = f"{EVALS}/data_3_6_flash_v8_blind"

ROSTER_BUDGET = ["claude-opus-5", "claude-sonnet-5",
                 "gemini-3.6-flash", "gemini-3.5-flash"]

ROSTER_BLIND = ["claude-opus-5", "claude-sonnet-5", "gemini-3.6-flash"]

# WHAT EACH ARM'S TREE IS EXPECTED TO HOLD ON DISK, which is NOT the same thing
# as the roster the table reports. This is the integrity guard: it is handed to
# corpus.load_eval, which checks it by SET EQUALITY, so a cell that failed to
# produce output still aborts the run instead of quietly shrinking the panel.
#
# WHY THE TWO DIVERGED. Complete 600-round gemini-3.7-flash runs landed in the
# b=30 and b=40 trees after this panel was frozen (their r01 directories are
# dated Aug 27-28, the shipped table predates them). The data is real and
# complete -- 60 scenario files, 600 rows, each arm -- but tab:ablation panel
# (a) reports the four models above and growing it to five would change a
# published table. Spelling ONE constant for both jobs is exactly what broke
# this module: a set-equality guard written against the REPORTED roster aborts
# the moment anyone adds a cell to a tree, and widening that constant to
# silence the abort would have reported the extra row. So the guard is stated
# against the tree and the subset down to the reported roster is asserted
# separately -- the same two-step load() already performs on the b=20 tree.
#
# THE PAPER REPOSITORY'S COPY OF THIS FILE DECIDED THE OPPOSITE, DELIBERATELY.
# There are two divergent checkouts of this package. The one under the paper
# repo (Google_Conflict_VLM/Google/analysis, git-tracked) put gemini-3.7-flash
# straight into ROSTER_BUDGET, ROSTER_BLIND and ROW_ORDER -- it intends to
# REPORT the fifth row. It loads cleanly and stops much later, on the prose
# guard in build() demanding sec:budget-ablation be rewritten first, because
# the extra model creates a second "gained" separation (claude-sonnet-5 /
# gemini-3.7-flash, p 0.828 -> 3.7e-4). This checkout is not that one and does
# not publish; it is held to the four rows already in main.tex, which it
# reproduces byte-for-byte. Do NOT reconcile the two by copying constants
# across -- they answer different questions, and four rows here is a decision,
# not staleness.
#
# DO NOT DELETE THE `r02 -> r01` SYMLINKS under Gemini/3_7_flash in these two
# trees. corpus.RUN_OVERRIDE is keyed on `<setting>/<vendor>/<model>`, a path
# that does not include the eval root, so the main tree's 3_7_flash -> r02
# selection also applies here, where no second run was ever drawn. The symlink
# is what makes the one real run (r01) reachable under the name the override
# demands. Removing it makes the cell vanish and trips the `missing=` guard
# below rather than corrupting a number, but it still stops the module dead.
TREE_CONTENTS = {
    20: None,                                     # main tree: full shipped panel
    30: ROSTER_BUDGET + ["gemini-3.7-flash"],
    40: ROSTER_BUDGET + ["gemini-3.7-flash"],
}
assert {b for b, _ in BUDGETS} == set(TREE_CONTENTS), (
    "every arm in BUDGETS needs a declared on-disk roster")
assert all(c is None or set(ROSTER_BUDGET) <= set(c)
           for c in TREE_CONTENTS.values()), (
    "a tree is declared to hold less than the roster the table reports")

# The blind tree caught the same fifth cell in the same sweep, so it gets the
# same treatment: guard against what is on disk, report the three models the
# shipped tab:ablation panel (b) names. Kept as its own constant rather than
# folded into TREE_CONTENTS because this arm withholds a modality instead of
# moving the step budget -- they are different conditions that happen to have
# been extended by the same person on the same day.
BLIND_CONTENTS = ROSTER_BLIND + ["gemini-3.7-flash"]
assert set(ROSTER_BLIND) <= set(BLIND_CONTENTS)

# Row order inside each panel: descending shipped-configuration pass rate, so
# the first row is the strongest model, matching every other table in the
# paper. Hard-coded rather than sorted at runtime because the two panels must
# agree on the order of the models they share, and sorting each panel by its
# own arm would let them disagree.
ROW_ORDER = ["claude-opus-5", "claude-sonnet-5",
             "gemini-3.6-flash", "gemini-3.5-flash"]

N_GATED = 346          # rounds carrying a required assets_accessed check
N_UNGATED = 254        # the rest; 346 + 254 == corpus.N_RAW


def asset_gated(tasks=None, cross_check=True):
    """{task_id} carrying a required check about having opened an asset.

    Computed on the SPEC side (effect_type == "assets_accessed" AND role ==
    "required"), the predicate section3_facts.py:492-495 already uses. Do not
    re-spell it.

    `cross_check` re-derives the same set on the EVAL side, where effect_type
    is unavailable and the gate is identifiable only by check name via
    corpus.ASSET_GATE_PREFIX, and asserts the two agree. The two sides label
    checks "required" completely differently (346 vs 3,284 -- see the module
    docstring), so the fact that they nevertheless select the same 346 rounds
    is the thing worth guarding: it is what licenses stratifying the modality
    arm on a spec-side field while grading happens on the eval side. It costs
    ~2s.
    """
    tasks = tasks or corpus.TASKS_V8
    gated, seen = set(), 0
    for p in glob.glob(f"{tasks}/*/specs.json"):
        d = json.load(open(p))
        seen += 1
        checks = (d.get("ground_truth") or {}).get("evaluation_checks") or []
        if any(c.get("effect_type") == "assets_accessed"
               and c.get("role") == "required" for c in checks):
            gated.add(d["task_id"])
    assert seen == corpus.N_RAW, f"expected {corpus.N_RAW} specs, saw {seen}"
    assert len(gated) == N_GATED, (
        f"asset-gated stratum moved: {len(gated)} != {N_GATED}. The modality "
        f"panel's strata are frozen; re-derive them deliberately.")
    if cross_check:
        eps, _ = corpus.load_eval(keep_checks=True)
        ref = {t for t, e in next(iter(eps.values())).items()
               if any(str(c.get("name") or "").startswith(corpus.ASSET_GATE_PREFIX)
                      for c in corpus.scored_checks(e))}
        assert ref == gated, (
            f"spec-side and eval-side asset gates disagree: "
            f"only-eval={len(ref - gated)} only-spec={len(gated - ref)}")
    return gated


def load():
    """arm key -> (episodes, errors). All arms, high effort, full population.

    EVERY arm is loaded against the roster its tree actually holds
    (TREE_CONTENTS) and then subset to the four models the table reports, with
    the subset asserted. b=20 is the main tree and its declared contents are
    the full shipped panel, so the strongest guard in load_eval -- set equality
    against MODEL_LABEL -- stays available on the one tree every other table in
    the paper also reads. The b=30 and b=40 trees carry a fifth model that the
    shipped panel does not report; see TREE_CONTENTS for why that is a
    reporting decision and not a data problem.
    """
    dead = corpus.load_dead()
    out = {}
    for b, root in BUDGETS:
        eps, errs = corpus.load_eval(
            root=root, roster=TREE_CONTENTS[b],
            keep_checks=False, dead=dead)
        missing = set(ROSTER_BUDGET) - set(eps)
        assert not missing, f"b{b} tree ({root}) is missing {sorted(missing)}"
        eps = {m: eps[m] for m in ROSTER_BUDGET}
        errs = {m: v for m, v in errs.items() if m in ROSTER_BUDGET}
        out[f"b{b}"] = (eps, errs)
    beps, berrs = corpus.load_eval(
        root=BLIND_TREE, roster=BLIND_CONTENTS, keep_checks=False, dead=dead)
    missing = set(ROSTER_BLIND) - set(beps)
    assert not missing, f"blind tree is missing {sorted(missing)}"
    out["blind"] = ({m: beps[m] for m in ROSTER_BLIND},
                    {m: v for m, v in berrs.items() if m in ROSTER_BLIND})
    return out


def rate(eps, ids=None):
    """(passes, n, pass %) over `ids` (all rounds when None)."""
    sel = eps if ids is None else {t: e for t, e in eps.items() if t in ids}
    n = len(sel)
    k = sum(1 for e in sel.values() if e["success"])
    return k, n, 100.0 * k / n


def capped(eps):
    return 100.0 * sum(1 for e in eps.values() if e["cap_hit"]) / len(eps)


def paired(a, b, ids=None):
    """Exact McNemar for arm A against arm B over the shared task ids.

    Returns (gain, loss, p): `gain` is rounds arm A solved and B did not.
    The two scripts already in the repo name these b and c the OTHER way
    round from each other (see the note in main.tex's effort section), so
    nothing here uses those letters.
    """
    keys = set(a) & set(b)
    if ids is not None:
        keys &= ids
    assert keys, "no shared rounds"
    gain = sum(1 for t in keys if a[t]["success"] and not b[t]["success"])
    loss = sum(1 for t in keys if b[t]["success"] and not a[t]["success"])
    return gain, loss, corpus.exact_mcnemar(gain, loss)


def _sig(p, digits=2):
    """A p-value spelled the way main.tex quotes it, or an abort.

    ABORTS ON A ROUNDING BOUNDARY, deliberately, because round-half-up and
    round-half-even disagree there and the printed digits give the reader no
    way to tell which was used. main.tex already carries a hand-written note
    about one such case in the effort section ($p = 1.65\\times10^{-15}$,
    quoted at three figures for exactly this reason).

    Where the boundary is actually REACHABLE, precisely, because the docstring
    this replaced overclaimed it: exact_mcnemar returns a sum of binomial
    coefficients over 2**n, so every p here is a dyadic rational, and the
    two-decimal branch really can land on one -- p = 0.125 goes to 0.13
    half-up and 0.12 half-even, and the guard fires on it. The scientific
    branch is defensive rather than load-bearing: a mantissa exactly on a half
    needs p = k.k5 * 10**e with e < 0, which is not dyadic, so the float always
    falls just off it (1.25e-9 is really 1.25000...0261e-9). Keep the check on
    both branches regardless; it costs nothing, and the argument above is a
    property of this estimator rather than of p-values in general.
    """
    assert p > 0, f"non-positive p: {p!r}"
    d = decimal.Decimal(p)                      # exact: floats are dyadic
    if p >= 0.001:
        unit = decimal.Decimal("0.01")
        up = d.quantize(unit, rounding=decimal.ROUND_HALF_UP)
        ev = d.quantize(unit, rounding=decimal.ROUND_HALF_EVEN)
        assert up == ev, f"p={p!r} is on a 2-decimal rounding boundary"
        return str(up)
    e = d.adjusted()
    man = d.scaleb(-e)
    unit = decimal.Decimal(1).scaleb(-(digits - 1))
    up = man.quantize(unit, rounding=decimal.ROUND_HALF_UP)
    ev = man.quantize(unit, rounding=decimal.ROUND_HALF_EVEN)
    assert up == ev, (
        f"p={p!r} is exactly on a {digits}-significant-figure boundary: "
        f"half-up gives {up}, half-even {ev}. Quote it at three figures in "
        f"the prose and say why, as the effort section does for 1.65e-15.")
    if up >= 10:
        up, e = up.scaleb(-1), e + 1
    return f"{up}\\times10^{{{e}}}"


def _dec(p, digits=2):
    """A p-value as a PLAIN decimal at `digits` significant figures.

    _sig rounds to two DECIMALS above 0.001 and switches to scientific
    notation below it, so it renders the gained pair's two p-values as "0.00"
    and "7.7\\times10^{-4}" -- one of them destroyed, the other in a different
    notation from its partner. Those two numbers exist in the prose precisely
    to be compared with each other and with a threshold of 0.00139, so they
    have to be the same shape. Same rounding-boundary abort as _sig, for the
    same reason.
    """
    assert p > 0, f"non-positive p: {p!r}"
    d = decimal.Decimal(p)                      # exact: floats are dyadic
    unit = decimal.Decimal(1).scaleb(d.adjusted() - (digits - 1))
    up = d.quantize(unit, rounding=decimal.ROUND_HALF_UP)
    ev = d.quantize(unit, rounding=decimal.ROUND_HALF_EVEN)
    assert up == ev, (
        f"p={p!r} is exactly on a {digits}-significant-figure boundary: "
        f"half-up {up}, half-even {ev}. Quote it at three figures and say why.")
    return f"{up:f}"


# The all-pairs separation set that sec:main-results reports. It is recomputed
# below rather than cited, and the count is asserted, because the budget arm's
# headline is "one of THOSE separating pairs stops separating" -- a claim about
# the paper, not about an arbitrary set of pairs, and only true while the count
# here is the count mcnemar_v8.py prints.
#
# THE PAIR COUNT IS DERIVED AND THE SEPARATION COUNT IS NOT, on purpose. The
# number of pairs is arithmetic on the roster and must follow it silently; the
# number that separate is an empirical result the prose quotes, so it has to be
# re-typed by a human who has looked at mcnemar_v8.py's output. Freezing both
# turned a roster change into a failure here that reads like a data problem;
# deriving both would let a claim in sec:main-results go stale in silence.
N_PAIRS_ALL = corpus.N_MODELS * (corpus.N_MODELS - 1) // 2
N_SEP_ALL = 21


def separations(arms):
    """What opening the cap does to the separations the paper already reports.

    DO NOT INVENT A SECOND CORRECTION HERE. An earlier draft of this analysis
    corrected over the six within-roster pairs at alpha = 0.05/6, which is
    LAXER than the paper's own threshold and would have had the ablation and
    the main analysis counting the same kind of thing at different odds.
    Everything below is held to alpha = 0.05/N_PAIRS_ALL -- the main
    analysis's number -- at both budgets, so the two are like-for-like.

    SCOPE, WHICH THE PROSE MUST CARRY. Only a handful of the separating
    pairs are testable here: a pair is testable only if BOTH its members were
    re-run, and the budget arm re-ran four of the API models. The rest all
    involve a model that was never re-run, and this arm is silent about them.
    The testable count is returned as `n_testable` rather than written here,
    because it moves whenever the roster or alpha moves. In particular it cannot speak to any of the four top-six
    separations sec:main-results highlights, which involve gpt-5.6-terra,
    gpt-5.6-luna and gpt-5.6-sol.

    Returns a dict; `lost` and `gained` are lists of (pair, p20, p40).
    """
    full = corpus.load_eval(keep_checks=False, dead=corpus.load_dead())[0]
    pairs = list(itertools.combinations(corpus.rank_order(full), 2))
    assert len(pairs) == N_PAIRS_ALL, (
        f"{len(pairs)} pairs over {len(full)} models, expected {N_PAIRS_ALL}")
    alpha = corpus.bonferroni(len(pairs))
    sep20_all = {p: paired(full[p[0]], full[p[1]])[2] for p in pairs}
    separating = {p for p, v in sep20_all.items() if v < alpha}
    assert len(separating) == N_SEP_ALL, (
        f"the all-pairs separation set moved: {len(separating)} != "
        f"{N_SEP_ALL} over {len(pairs)} pairs. "
        f"mcnemar_v8.py and sec:main-results quote {N_SEP_ALL}; re-derive all "
        f"three together, or the prose's claim about 'those separating pairs' "
        f"is false.")

    b20, b40 = arms["b20"][0], arms["b40"][0]
    out = dict(alpha=alpha, n_pairs=len(pairs), n_sep=len(separating),
               lost=[], gained=[], survived=[], rows=[])
    for p in itertools.combinations(ROW_ORDER, 2):
        key = p if p in sep20_all else (p[1], p[0])
        g2, l2, p2 = paired(b20[p[0]], b20[p[1]])
        # b=20 IS the main tree, so this must reproduce the full-panel run
        # exactly. If it ever does not, the subset load has drifted.
        assert abs(p2 - sep20_all[key]) < 1e-12, (
            f"b=20 disagrees with the full-panel run on {p}: "
            f"{p2!r} vs {sep20_all[key]!r}")
        g4, l4, p4 = paired(b40[p[0]], b40[p[1]])
        testable = key in separating
        out["rows"].append(dict(pair=p, p20=p2, p40=p4, g20=g2, l20=l2,
                                g40=g4, l40=l4, testable=testable))
        if testable and p4 >= alpha:
            out["lost"].append((p, p2, p4))
        elif testable:
            out["survived"].append((p, p2, p4))
        elif p4 < alpha:
            out["gained"].append((p, p2, p4))
    out["n_testable"] = len(out["lost"]) + len(out["survived"])
    return out


def build():
    arms = load()
    gated = asset_gated()
    allids = set(arms["b20"][0]["claude-opus-5"])
    ungated = allids - gated
    assert len(ungated) == N_UNGATED, (
        f"ungated stratum: {len(ungated)} != {N_UNGATED}")

    f = {}
    # ---- panel (a): step budget -------------------------------------
    budget = {}
    for m in ROW_ORDER:
        row = {}
        for b, _ in BUDGETS:
            eps = arms[f"b{b}"][0][m]
            k, n, p = rate(eps)
            row[b] = dict(passes=k, n=n, pass_pct=p, cap=capped(eps),
                          steps=sum(e["steps"] or 0 for e in eps.values()) / n)
        row["delta"] = row[40]["pass_pct"] - row[20]["pass_pct"]
        for b in (30, 40):
            g, l, p = paired(arms[f"b{b}"][0][m], arms["b20"][0][m])
            row[f"mc{b}"] = dict(gain=g, loss=l, p=p)
        budget[m] = row

    # ---- panel (b): visual modality ---------------------------------
    mod = {}
    for m in ROSTER_BLIND:
        sight, blind = arms["b20"][0][m], arms["blind"][0][m]
        row = {}
        for name, ids in (("all", None), ("gated", gated), ("ungated", ungated)):
            ks, n, ps = rate(sight, ids)
            kb, _, pb = rate(blind, ids)
            g, l, p = paired(sight, blind, ids)
            row[name] = dict(n=n, sighted=ps, blind=pb, s_k=ks, b_k=kb,
                             gain=g, loss=l, p=p,
                             retained=(pb / ps if ps else float("nan")))
        mod[m] = row

    # ---- facts ------------------------------------------------------
    n_budget_tests = len(ROW_ORDER) * 2                  # 4 models x {30,40}
    n_mod_tests = len(ROSTER_BLIND) * 2                  # 3 models x {gated,ungated}
    f["ab_budget_tests"] = n_budget_tests
    f["ab_budget_alpha"] = corpus.bonferroni(n_budget_tests)
    f["ab_mod_tests"] = n_mod_tests
    f["ab_mod_alpha"] = corpus.bonferroni(n_mod_tests)
    f["ab_gated_n"] = N_GATED
    f["ab_ungated_n"] = N_UNGATED
    f["ab_err_b40_flash35"] = arms["b40"][1].get("gemini-3.5-flash", 0)

    for m in ROW_ORDER:
        s = m.split("-", 1)[1].replace("-", "").replace(".", "")
        f[f"ab_bud_{s}_20"] = round(budget[m][20]["pass_pct"], 1)
        f[f"ab_bud_{s}_40"] = round(budget[m][40]["pass_pct"], 1)
        f[f"ab_bud_{s}_delta"] = round(budget[m]["delta"], 1)
        f[f"ab_bud_{s}_cap20"] = round(budget[m][20]["cap"], 1)
        f[f"ab_bud_{s}_cap40"] = round(budget[m][40]["cap"], 1)
        f[f"ab_bud_{s}_steps40"] = round(budget[m][40]["steps"], 2)
        f[f"ab_bud_{s}_gain40"] = budget[m]["mc40"]["gain"]
        f[f"ab_bud_{s}_loss40"] = budget[m]["mc40"]["loss"]
        f[f"ab_bud_{s}_p40"] = _sig(budget[m]["mc40"]["p"])
    for m in ROSTER_BLIND:
        s = m.split("-", 1)[1].replace("-", "").replace(".", "")
        f[f"ab_mod_{s}_ung_sight"] = round(mod[m]["ungated"]["sighted"], 1)
        f[f"ab_mod_{s}_ung_blind"] = round(mod[m]["ungated"]["blind"], 1)
        f[f"ab_mod_{s}_ung_only"] = mod[m]["ungated"]["gain"]
        f[f"ab_mod_{s}_ung_blindonly"] = mod[m]["ungated"]["loss"]
        f[f"ab_mod_{s}_ung_p"] = _sig(mod[m]["ungated"]["p"])
        f[f"ab_mod_{s}_all_only"] = mod[m]["all"]["gain"]
        f[f"ab_mod_{s}_all_delta"] = mod[m]["all"]["b_k"] - mod[m]["all"]["s_k"]

    # ---- the cross-arm tie-in the budget prose turns on ----------------
    # sec:effort explains gemini-3.5-flash's effort INVERSION as the budget
    # binding, and could only check that against corpus B, which changes the
    # generator as well as the budget. The budget arm tests it by direct
    # manipulation on corpus A: hold effort at high and open the cap instead.
    # Recomputed here rather than cited from eff_* so the comparison lives in
    # one script and cannot drift against a renamed fact elsewhere.
    low, _ = corpus.load_eval(effort="low", keep_checks=False)
    lo = low["gemini-3.5-flash"]
    f["ab_35f_low20_pass"] = round(rate(lo)[2], 1)
    f["ab_35f_low20_cap"] = round(capped(lo), 1)
    f["ab_35f_low20_steps"] = round(
        sum(e["steps"] or 0 for e in lo.values()) / len(lo), 2)

    # ---- what the budget does to Table 2's headline spread -------------
    # sec:main-results quotes an 8.4x pass spread over all NINE models. The
    # four models here happen to contain both of its extremes (claude-opus-5
    # at the top, gemini-3.5-flash at the bottom), so the b=20 spread computed
    # over these four is the SAME number, not a coincidence to be glossed --
    # asserted below, because if a future rerun moved either extreme out of
    # this roster the cost qualifier in sec:main-results would silently start
    # comparing two different quantities.
    # ---- does the ranking survive the cap? -----------------------------
    # Not eyeballed off the table: the b=20 and b=40 orderings are compared
    # pair by pair. They are NOT identical -- claude-sonnet-5 and
    # gemini-3.6-flash trade places -- and the prose has to say so, so the
    # count is computed rather than asserted.
    def order(b):
        return sorted(ROW_ORDER, key=lambda m: -budget[m][b]["passes"])
    o20, o40 = order(20), order(40)
    f["ab_rank_swaps"] = sum(
        1 for i, a in enumerate(ROW_ORDER) for z in ROW_ORDER[i + 1:]
        if (o20.index(a) < o20.index(z)) != (o40.index(a) < o40.index(z)))
    f["ab_rank_top_stable"] = o20[0] == o40[0]
    f["ab_rank_bot_stable"] = o20[-1] == o40[-1]
    f["ab_rank_swap_pair"] = " / ".join(sorted(
        {a for i, a in enumerate(ROW_ORDER) for z in ROW_ORDER[i + 1:]
         if (o20.index(a) < o20.index(z)) != (o40.index(a) < o40.index(z))}
        | {z for i, a in enumerate(ROW_ORDER) for z in ROW_ORDER[i + 1:]
           if (o20.index(a) < o20.index(z)) != (o40.index(a) < o40.index(z))}))

    # ---- churn: rounds that flip either way between b=20 and b=40 -------
    # The aggregate being flat does not mean the rounds are. Reported as a
    # share of all rounds so the two Claude rows can be described honestly.
    #
    # GROSS CHURN IS NOT NET MOVEMENT AND A SHIPPED DRAFT CONFLATED THEM.
    # claude-opus-5 churns 8.2% of the corpus and moves the aggregate +1.8
    # points; the earlier prose invited the reader to treat the 8.2 as the
    # size of a re-run wobble in Table~\ref{tab:main}'s units. It is not.
    # Both quantities are shipped as facts so the prose cannot quote one for
    # the other again.
    for m in ROW_ORDER:
        s = m.split("-", 1)[1].replace("-", "").replace(".", "")
        t = budget[m]["mc40"]
        f[f"ab_bud_{s}_churn40"] = round(
            100.0 * (t["gain"] + t["loss"]) / budget[m][40]["n"], 1)

    # ---- how many of the gained rounds actually SPENT the new budget ----
    # A round can only have been bought by the extra room if its b=40 rollout
    # ran past the shipped cap. Counted on `steps`, deliberately NOT on
    # `cap_hit`: 11 rounds across the arms end with steps == budget but an
    # empty stop_reason and so are not flagged cap_hit, which makes that flag
    # a slight undercount. `steps > 20` needs no flag at all.
    #
    # WHAT THIS DOES AND DOES NOT LICENCE. It bounds the MECHANISM: on a round
    # whose b=40 rollout used 20 steps or fewer, the extra room went unspent,
    # so whatever moved that round, it was not the extra room. It does not
    # make the arm a same-configuration re-run, because the cap is also
    # announced in the prompt (train_eval/agents.py:3124, "You have at most N
    # steps", with the hint bound to the enforced cap), so the two arms differ
    # in what the solver is told as well as in what it is allowed. The
    # announcement is a global constant across models at a fixed budget, so it
    # costs the main tables nothing -- but DO NOT write this arm up as a pure
    # estimate of re-run noise.
    for m in ROW_ORDER:
        s = m.split("-", 1)[1].replace("-", "").replace(".", "")
        a20, a40 = arms["b20"][0][m], arms["b40"][0][m]
        keys = set(a20) & set(a40)
        gained = [t for t in keys if a40[t]["success"] and not a20[t]["success"]]
        f[f"ab_bud_{s}_gainspent"] = sum(1 for t in gained
                                         if (a40[t]["steps"] or 0) > 20)
        f[f"ab_bud_{s}_over20_b40"] = sum(1 for t in keys
                                          if (a40[t]["steps"] or 0) > 20)
        f[f"ab_bud_{s}_net40"] = round(
            100.0 * (len(gained) - budget[m]["mc40"]["loss"]) / len(keys), 1)

    top20 = max(budget[m][20]["passes"] for m in ROW_ORDER)
    bot20 = min(budget[m][20]["passes"] for m in ROW_ORDER)
    all20 = [sum(1 for e in v.values() if e["success"])
             for v in corpus.load_eval(keep_checks=False)[0].values()]
    assert top20 == max(all20) and bot20 == min(all20), (
        "the budget roster no longer contains both extremes of the nine-model "
        "panel; sec:main-results' spread qualifier must be rewritten")
    f["ab_spread20"] = round(top20 / bot20, 1)
    f["ab_spread40"] = round(max(budget[m][40]["passes"] for m in ROW_ORDER)
                             / min(budget[m][40]["passes"] for m in ROW_ORDER), 1)

    sig_b = sum(1 for m in ROW_ORDER
                if budget[m]["mc40"]["p"] < f["ab_budget_alpha"])
    f["ab_budget_sig40"] = sig_b
    f["ab_mod_sig_ungated"] = sum(1 for m in ROSTER_BLIND
                                  if mod[m]["ungated"]["p"] < f["ab_mod_alpha"])
    f["ab_mod_worst_retained"] = round(
        min(mod[m]["ungated"]["retained"] for m in ROSTER_BLIND), 3)
    f["ab_mod_best_retained"] = round(
        max(mod[m]["ungated"]["retained"] for m in ROSTER_BLIND), 3)

    # ---- which separations survive the cap ------------------------------
    sep = separations(arms)
    f["ab_sep_pairs_all"] = sep["n_pairs"]
    f["ab_sep_n36"] = sep["n_sep"]
    f["ab_sep_testable"] = sep["n_testable"]
    f["ab_sep_survived"] = len(sep["survived"])
    f["ab_sep_lost"] = len(sep["lost"])
    f["ab_sep_gained"] = len(sep["gained"])
    assert len(sep["lost"]) == 1 and len(sep["gained"]) == 1, (
        f"the prose names ONE lost and ONE gained separation; got "
        f"{len(sep['lost'])} and {len(sep['gained'])}. Rewrite sec:budget-"
        f"ablation before re-shipping.")
    (lp, lp20, lp40), = sep["lost"]
    (gp, gp20, gp40), = sep["gained"]
    lrow, = [r for r in sep["rows"] if r["pair"] == lp]
    f["ab_sep_lost_pair"] = " / ".join(lp)
    f["ab_sep_lost_p20"] = _sig(lp20)
    f["ab_sep_lost_p40"] = _sig(lp40)
    f["ab_sep_lost_disc20"] = lrow["g20"] + lrow["l20"]
    f["ab_sep_lost_disc40"] = lrow["g40"] + lrow["l40"]
    f["ab_sep_lost_gl20"] = f"{lrow['g20']}/{lrow['l20']}"
    f["ab_sep_lost_gl40"] = f"{lrow['g40']}/{lrow['l40']}"
    f["ab_sep_lost_eff20"] = round(100.0 * (lrow["g20"] - lrow["l20"]) / 600, 1)
    f["ab_sep_lost_eff40"] = round(100.0 * (lrow["g40"] - lrow["l40"]) / 600, 1)
    # The discordant count RISING while significance is lost is the whole
    # argument that this is a shrinking effect and not a thinner test.
    f["ab_sep_lost_power_up"] = (f["ab_sep_lost_disc40"]
                                 > f["ab_sep_lost_disc20"])
    f["ab_sep_gained_pair"] = " / ".join(gp)
    f["ab_sep_gained_p20"] = _dec(gp20)
    f["ab_sep_gained_p40"] = _dec(gp40)
    f["ab_sep_alpha_all"] = _dec(sep["alpha"], 3)
    # Both claims must hold at every correction the paper uses, or the prose
    # cannot say "at any of the three".
    # 6 within-roster pairs, 15 top-six pairs, N_PAIRS_ALL all-pairs. The
    # last one is derived from the roster rather than typed, so the "at any of
    # the three" claim keeps meaning the corrections the paper actually uses.
    f["ab_sep_corrections"] = (6, 15, N_PAIRS_ALL)
    f["ab_sep_robust"] = all(
        lp40 >= a and gp40 < a and lp20 < a
        for a in (corpus.bonferroni(n) for n in f["ab_sep_corrections"]))
    # The spread narrows, but every separation involving the model that
    # carries the narrowing survives. sec:budget-ablation says so explicitly,
    # because "the widest gap is a budget gap" was the earlier, wrong reading.
    w = [r for r in sep["rows"]
         if "gemini-3.5-flash" in r["pair"] and r["testable"]]
    assert w and all(r["p40"] < sep["alpha"] for r in w), (
        "a gemini-3.5-flash separation stopped surviving at b=40; the prose "
        "claims all of them do")
    f["ab_sep_worst_survive"] = len(w)
    f["ab_sep_worst_maxp40"] = _sig(max(r["p40"] for r in w))
    f["_sep"] = sep
    return budget, mod, f


# PANELS (a) AND (b) ARE SEVEN COLUMNS: one label plus six numeric. They used
# to be eight, and to draw their group rules with a per-cell \multicolumn
# wrapper, because all three panels of tab:ablation-app shared one preamble and
# the three disagree about where their groups end:
#
#     (a) Model | 20 30 40 | D | 20 40        breaks after 1, 4, 5
#     (b) Model | s b | s b | s b             breaks after 1, 3, 5
#     (c) Corpus| seven model columns         breaks after 1 only
#
# Only the break after column 1 was common to all three, so only that one
# could live in the preamble; the rest had to be set per cell, and (a) and (b)
# had to carry a trailing empty seventh numeric cell to reach the width panel
# (c) needed. The float now sets each panel in its own tabular*, so each has
# its own preamble and its own width, and none of that applies: the vertical
# rules are gone, the trailing cell is gone, and (c) is free to be eight
# columns while these two are seven. Keep the panels in separate environments
# and this stays true.
ABL_PANEL_COLS = 7


def _panel_a(budget, w):
    """Panel (a), step budget: heading through data, no top or bottom rule."""
    out = [r"    \multicolumn{" + str(ABL_PANEL_COLS)
           + r"}{@{}l}{\emph{(a) Step budget.} "
           r"Same 600 rounds, high effort; only the cap moves.} \\",
           r"    \addlinespace[2pt]",
           r"    & \multicolumn{4}{c}{\textbf{Pass Rate (\%)}} & "
           r"\multicolumn{2}{c}{\textbf{Ended at Cap (\%)}} \\",
           r"    \cmidrule(lr){2-5}\cmidrule(lr){6-7}",
           r"    \textbf{Model} & \textbf{20} & \textbf{30} & \textbf{40} & "
           r"\textbf{$\Delta$} & \textbf{20} & \textbf{40} \\",
           r"    \midrule"]
    for m in ROW_ORDER:
        r = budget[m]
        cols = [f"{r[b]['pass_pct']:.1f}" for b, _ in BUDGETS]
        cols += [f"${r['delta']:+.1f}$", f"{r[20]['cap']:.1f}",
                 f"{r[40]['cap']:.1f}"]
        out.append(f"    \\texttt{{{m}}}" + " " * (w - len(m))
                   + " & " + " & ".join(cols) + r" \\")
    return out


def _panel_b(mod, w):
    """Panel (b), visual modality: heading through data, no top or bottom rule."""
    out = [r"    \multicolumn{" + str(ABL_PANEL_COLS)
           + r"}{@{}l}{\emph{(b) Visual modality.} "
           r"Budget 20, high effort; assets withheld, task text unchanged.} \\",
           r"    \addlinespace[2pt]",
           r"    & \multicolumn{2}{c}{\textbf{All (600)}} "
           rf"& \multicolumn{{2}}{{c}}{{\textbf{{Asset-Gated ({N_GATED})}}}} "
           rf"& \multicolumn{{2}}{{c}}{{\textbf{{No Asset Check ({N_UNGATED})}}}} \\",
           r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
           r"    \textbf{Model} & \textbf{Sighted} & \textbf{Blind} & "
           r"\textbf{Sighted} & \textbf{Blind} & \textbf{Sighted} & "
           r"\textbf{Blind} \\",
           r"    \midrule"]
    for m in ROSTER_BLIND:
        r = mod[m]
        cols = []
        for name in ("all", "gated", "ungated"):
            cols += [f"{r[name]['sighted']:.1f}", f"{r[name]['blind']:.1f}"]
        out.append(f"    \\texttt{{{m}}}" + " " * (w - len(m))
                   + " & " + " & ".join(cols) + r" \\")
    return out


def latex(budget, mod):
    """The two tab:ablation-app panels this script owns, as separate blocks.

    Separate, not one: they are set in two tabular* environments now, so the
    rules that close (a) and open (b) sit between them, and a single block
    would have to either contain that seam or fail to be contiguous.
    verify_table_bodies.py matches a block as one unbroken run of lines, so
    the second is what would happen -- and it is what did happen, silently,
    for as long as the two panels were emitted together.

    The markers are deliberately NOT "LaTeX body for T3": check_split_v8.py
    already prints a block by that name for tab:validity, and
    verify_table_bodies.py keys on (script, marker), so two identical markers
    would work but would read as a collision to anyone auditing it. Neither
    marker below is a substring of the other, which `gen_block` requires.
    """
    w = max(len(m) for m in ROW_ORDER)
    return _panel_a(budget, w), _panel_b(mod, w)


def main():
    budget, mod, f = build()

    print("=== (a) step budget, high effort, n=600 per cell ===")
    print(f"{'model':18} {'b20':>14} {'b30':>14} {'b40':>14} "
          f"{'cap20':>6} {'cap40':>6}")
    for m in ROW_ORDER:
        r = budget[m]
        cells = " ".join(f"{r[b]['passes']:4}/{r[b]['n']} "
                         f"{r[b]['pass_pct']:5.1f}%" for b, _ in BUDGETS)
        print(f"{m:18} {cells} {r[20]['cap']:5.1f}% {r[40]['cap']:5.1f}%")
    print(f"\npaired exact McNemar vs b=20, Bonferroni "
          f"{f['ab_budget_tests']} tests, alpha={f['ab_budget_alpha']:.5f}")
    for m in ROW_ORDER:
        for b in (30, 40):
            t = budget[m][f"mc{b}"]
            star = "*" if t["p"] < f["ab_budget_alpha"] else " "
            print(f"  {m:18} b={b}: gained {t['gain']:3}, lost {t['loss']:3}, "
                  f"p={t['p']:.3g} {star}")

    print("\n=== (b) visual modality, budget 20, high effort ===")
    print(f"{'model':18} {'stratum':14} {'n':>4} {'sighted':>8} {'blind':>8} "
          f"{'retained':>9} {'gain/loss':>11} {'p':>10}")
    for m in ROSTER_BLIND:
        for name in ("all", "gated", "ungated"):
            r = mod[m][name]
            print(f"{m:18} {name:14} {r['n']:4} {r['sighted']:7.1f}% "
                  f"{r['blind']:7.1f}% {r['retained']:9.3f} "
                  f"{r['gain']:5}/{r['loss']:<5} {r['p']:10.3g}")
    print(f"\nmodality Bonferroni {f['ab_mod_tests']} tests "
          f"(3 models x gated/ungated), alpha={f['ab_mod_alpha']:.5f}; "
          f"{f['ab_mod_sig_ungated']}/{len(ROSTER_BLIND)} significant "
          f"on the ungated stratum")
    print(f"blind retains {f['ab_mod_worst_retained']:.3f}-"
          f"{f['ab_mod_best_retained']:.3f} of the sighted pass rate there")
    if f["ab_err_b40_flash35"]:
        print(f"\nprovider errors retained as failures: "
              f"b40/gemini-3.5-flash {f['ab_err_b40_flash35']}")

    # Quantities the PROSE asserts that no row above carries. They are printed
    # rather than left in the returned dict because results_facts.py parses
    # this stdout instead of importing build(); anything not printed here has
    # no guard on it once it reaches main.tex.
    print("\n=== (c) prose quantities ===")
    print(f"budget cap opened, gemini-3.5-flash at b=40 : "
          f"{f['ab_bud_35flash_40']}% pass, {f['ab_bud_35flash_cap40']}% cap, "
          f"{f['ab_bud_35flash_steps40']} steps")
    print(f"same model at b=20 LOW effort (sec:effort)  : "
          f"{f['ab_35f_low20_pass']}% pass, {f['ab_35f_low20_cap']}% cap, "
          f"{f['ab_35f_low20_steps']} steps")
    print(f"pass spread over these 4 models             : "
          f"b=20 {f['ab_spread20']}x, b=40 {f['ab_spread40']}x "
          f"(both extremes of the nine-model panel are in this roster)")
    print(f"mean steps at b=40                          : "
          + ", ".join(f"{m} {budget[m][40]['steps']:.2f}" for m in ROW_ORDER))
    print(f"rank order b=20 vs b=40                     : "
          f"{f['ab_rank_swaps']} discordant pair(s), top stable "
          f"{f['ab_rank_top_stable']}, bottom stable {f['ab_rank_bot_stable']}"
          + (f", swapped: {f['ab_rank_swap_pair']}" if f["ab_rank_swaps"]
             else ""))
    print(f"rounds flipping either way, b=20 vs b=40    : "
          + ", ".join(
              f"{m} {f['ab_bud_' + m.split('-', 1)[1].replace('-', '').replace('.', '') + '_churn40']}%"
              for m in ROW_ORDER))

    print("\n=== (d) did the gained rounds spend the new budget? ===")
    print(f"{'model':18} {'gained':>7} {'of those >20 steps':>19} "
          f"{'any >20 steps':>14} {'churn':>7} {'net':>7}")
    for m in ROW_ORDER:
        s = m.split("-", 1)[1].replace("-", "").replace(".", "")
        print(f"{m:18} {budget[m]['mc40']['gain']:7} "
              f"{f['ab_bud_' + s + '_gainspent']:19} "
              f"{f['ab_bud_' + s + '_over20_b40']:>10}/600 "
              f"{f['ab_bud_' + s + '_churn40']:6.1f}% "
              f"{f['ab_bud_' + s + '_net40']:+6.1f}")

    sep = f["_sep"]
    print("\n=== (e) separations from sec:main-results, under the cap ===")
    print(f"all {f['ab_sep_pairs_all']} pairs over {corpus.N_MODELS} models at "
          f"alpha={sep['alpha']:.6f}: {f['ab_sep_n36']} separate; "
          f"{f['ab_sep_testable']} of those involve only re-run models")
    for r in sorted(sep["rows"], key=lambda r: not r["testable"]):
        tag = ("testable" if r["testable"]
               else f"not in the {N_SEP_ALL}")
        mark = ""
        if r["testable"]:
            mark = "SURVIVES" if r["p40"] < sep["alpha"] else "LOST"
        elif r["p40"] < sep["alpha"]:
            mark = "GAINED"
        print(f"  {r['pair'][0]:17} vs {r['pair'][1]:17} "
              f"b20 {r['g20']:3}/{r['l20']:<3} p={r['p20']:9.3g} | "
              f"b40 {r['g40']:3}/{r['l40']:<3} p={r['p40']:9.3g} "
              f" [{tag}] {mark}")
    print(f"survived {f['ab_sep_survived']}, lost {f['ab_sep_lost']}, "
          f"gained {f['ab_sep_gained']}")
    print(f"lost pair   : {f['ab_sep_lost_pair']} "
          f"({f['ab_sep_lost_gl20']} -> {f['ab_sep_lost_gl40']}), "
          f"discordant {f['ab_sep_lost_disc20']} -> "
          f"{f['ab_sep_lost_disc40']} (power up: "
          f"{f['ab_sep_lost_power_up']}), effect "
          f"{f['ab_sep_lost_eff20']:+.1f} -> {f['ab_sep_lost_eff40']:+.1f} pts")
    print(f"gained pair : {f['ab_sep_gained_pair']} "
          f"(p {f['ab_sep_gained_p20']} -> {f['ab_sep_gained_p40']}, "
          f"threshold {f['ab_sep_alpha_all']})")
    print(f"robust at alpha 0.05/"
          f"{{{','.join(str(n) for n in f['ab_sep_corrections'])}}}: "
          f"{f['ab_sep_robust']}")
    print(f"gemini-3.5-flash separations surviving b=40: "
          f"{f['ab_sep_worst_survive']}, largest p = "
          f"{f['ab_sep_worst_maxp40']}")

    panel_a, panel_b = latex(budget, mod)
    print("\n% ---- LaTeX body for the step-budget panel ----")
    print("%      column spec: l cccc cc, its own tabular*; \\toprule and")
    print("%      \\bottomrule are hand-written")
    for line in panel_a:
        print(line)
    print("\n% ---- LaTeX body for the visual-modality panel ----")
    print("%      column spec: l cc cc cc, its own tabular*; \\toprule and")
    print("%      \\bottomrule are hand-written")
    for line in panel_b:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
