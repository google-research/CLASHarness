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

"""T1 -- main results on the difficulty axis: pass rate and score by gold length.

The paper's declared `difficulty` spec field is constant at 3 on all 600 rounds
(see coverage_v8.py), so it cannot carry a difficulty axis. What does vary, and
what the solver actually feels, is how much of the step budget a round leaves
spare: slack = BUDGET_V8 / gold length. Since the budget is the constant 20,
banding on slack and banding on gold length are the SAME partition -- gold
length is used here because it is an integer and reads better in a caption.

  easy   gold <=  9   n=236   slack >= 2.22
  mid    gold 10-11   n=224   slack 1.82-2.00
  hard   gold >= 12   n=140   slack <  1.82

(The gap between 2.00 and 2.22 is not a rounding slip: gold length is an
integer, so no round can land there.)

WHICH MODELS ARE ON THE TABLE. High effort carries the full fourteen-model
panel -- the ten API models plus the four open-weight ones, which were only
ever run at high. Medium and low carry the ten. The counts printed below
therefore have a denominator that moves with the effort argument, and every one
of them is printed as a fraction rather than a bare numerator for exactly that
reason. Three of the fourteen are text-only checkpoints that received the
rendered assets as an OCR transcription instead of as an image
(corpus.ASSET_MODALITY); their rows are daggered in the LaTeX body and starred
in the console view, and the sentence explaining the mark is generated beneath
the body rather than typed into the caption.

WHAT THIS AXIS ORDERS, AND WHAT IT DOES NOT.
Round pass rate is monotone easy > mid > hard for 13 of the 14 at high and for
10 of 10 at each of the other two. IT IS NO LONGER UNANIMOUS AT HIGH, and this
paragraph asserted for several drafts that it was: deepseek-v4-flash passes
8.5% easy against 11.2% mid, so the easy end inverts for it. Do not restore the
"every model" wording without re-reading the printed count.
Mean score does not order the axis at all: AT HIGH EFFORT it is monotone for
0 of 14, and all 14 score HIGHER on mid than on easy. That is not an averaging
artifact and the reader will suspect that it is, so the tool checks: the micro
average (every required check in the band over one denominator, which removes
the per-round denominator weighting completely) is also 0 of 14 and inverts
slightly harder. The axis is a CONJUNCTION-LENGTH axis.
Per-check accuracy is flat-to-inverted across it while the number of required
checks a round must clear grows 4.19 -> 5.48 -> 7.63, so rounds get harder to
win without individual checks getting harder to pass. Sec 6.2's failure
concentration is the same observation from the other side.

THE SCORE INVERSION IS AN EFFORT-DEPENDENT COUNT. DO NOT QUOTE IT UNSCOPED.
"Every model scores higher on mid than on easy" is unanimous at high and is
not at the other two levels, so the sentence needs its effort said out loud.
The count is also not the same proposition as the monotone count beside it --
easy > mid > hard can fail through the hard end -- and the paper leans on the
mid > easy reading specifically. Both are therefore printed, at every level.

WHY THE HEADLINE STATISTIC IS STRATIFIED AND NOT POOLED.
The bands are badly unbalanced across scenario seeds: three of the twenty
(incident_commander_selection, warehouse_robotics_safety_halt,
satellite_ground_station_handover) have NO hard rounds at all and contribute 69
easy ones, while multi_device_pairing alone supplies 21 of the 140 hard rounds
against 4 easy. A pooled easy-hard difference therefore partly measures which
scenarios happen to be short.

WHICH WAY IT BIASES IS A PER-MODEL FACT, NOT A CONSTANT. This paragraph used
to say the pooled gap "overstates by 2-6 points on every model" and that was
wrong in both directions: at medium and low the tool was already printing a
negative minimum. On the fourteen-model panel at high it runs -2.8 to +5.8,
overstated for 9 and understated for 5, and the understating rows are the four
open-weight models plus gemini-3.7-flash. That last one is the reason the
grouping is not "the OCR arm": an API vision model sits in it, so the sign is a
property of where a model's per-seed strengths happen to fall. The tool prints
the direction, the split and the understating models by name; nothing in the
paper quotes the range, and nothing should start.

The reported delta is the within-seed contrast, CMH-weighted
(w = n_easy*n_hard/(n_easy+n_hard), the inverse-variance weight for a
difference of proportions), over the seventeen seeds that hold both bands.

The interval is a nonparametric CLUSTER bootstrap: whole seeds are resampled
with replacement, all thirty of their rounds kept. corpus.boot_ci is not used
here and must not be -- it resamples a flat list of rounds, which treats the
600 as independent and would report an interval for a question nobody asked.
The estimand is "what would this gap be on a scenario drawn from the same
generator", and that question has n = 20, not n = 600.

EFFORT, AND WHY THE ROW ORDER DOES NOT FOLLOW IT.
The reasoning-effort level is the one argument this tool takes. It defaults to
corpus.PRIMARY_EFFORT and the paper ships all three, stacked. The ROW ORDER and
the provider block boundaries are always computed at PRIMARY_EFFORT, never at
the effort being tabulated. That is deliberate. The three tables exist to be
read down a column -- one model tracked across three efforts on the same row --
and a table that re-sorted itself by its own pass rate would silently destroy
that comparison while looking untouched. The cost is real and is paid in the
caption: the pass column does not descend on the medium and low tables, so they
carry corpus.blocking_ref (a pointer to Table 1's ordering) instead of
corpus.blocking_note (a claim about this table's own ranking, which would be
false there).

Since the open-weight group runs at high only, the two lower tables are now a
SUBSET of Table 1's rows rather than a permutation of them: the ten appear in
the same relative order, with the four open-weight rows absent rather than
moved. "Rows are in the order of Table~\\ref{tab:main}" is still true of what
is printed, but a table that lets a reader expect fourteen rows and hands them
ten is the kind of thing that gets read as a dropped model, so the paragraph
introducing the three tables says which ten they are and why.

Prints a LaTeX body as well as the console view; both come from one dict so
they cannot drift.
"""
import collections
import random
import re
import sys

import corpus
# The "all" column and its interval come from T1's own cell(), not from a
# reimplementation here. Two bootstraps over the same quantity would be two
# defensible numbers that disagree in the last digit, and the paper would then
# have a table and a sentence sourced from different draws. One function, one
# interval. (Consequence to remember: editing table1_v8.py changes this
# script's output without changing this script's mtime, which is what
# results_facts.py keys its cache on -- use --fresh after touching it.)
import table1_v8

# Row marker for models whose rendered assets arrived as an OCR transcription
# rather than as an image -- the membership is corpus.ASSET_MODALITY's, never
# a list retyped here. Emitted by the generator rather than pencilled into the
# caption afterwards, for the reason src/verify_table_bodies.py exists: a
# marker added by hand to main.tex is a marker that survives the next
# regeneration without being re-earned. The sentence it points at is printed
# below the body, so the caption is generated too.
OCR_MARK = r"$^\dagger$"
# Its console counterpart. One character, so it fits inside the existing name
# column for every name on the current roster and the nine-model console table
# is unchanged to the byte.
OCR_MARK_TXT = "*"

# Band edges in gold-length units. Inclusive upper bounds.
EASY_MAX, MID_MAX = 9, 11
BANDS = ("easy", "mid", "hard")
# "all" is the whole 600 and is carried alongside the bands on purpose: it is
# the number Sec 6.1 quotes throughout, and it is what the rows are ordered by,
# so the blocking clause in the caption stays checkable against the table it
# sits under. COLS is the display order.
COLS = ("all",) + BANDS
# Asserted below rather than trusted: a regenerated corpus changes these.
BAND_N = {"easy": 236, "mid": 224, "hard": 140}
# Table 1 is ours to compute, so it gets the live budget. See corpus.DRAWS for
# why it is 10,000 and corpus.FACTORY_DRAWS for why the appendix keeps 4,000.
DRAWS = corpus.DRAWS


def band_of(gold):
    return "easy" if gold <= EASY_MAX else ("mid" if gold <= MID_MAX else "hard")


def cmh_delta(strata):
    """Within-seed easy-hard pass difference in points, CMH-weighted.

    `strata` maps an opaque stratum key to {"easy": [success...], "hard": [...]}.
    Seeds holding only one band drop out: they carry no within-seed contrast,
    and including them as a pooled remainder is exactly the confound this
    function exists to remove.
    """
    num = den = 0.0
    for c in strata.values():
        ne, nh = len(c["easy"]), len(c["hard"])
        if not ne or not nh:
            continue
        w = ne * nh / (ne + nh)
        num += w * (100.0 * sum(c["easy"]) / ne - 100.0 * sum(c["hard"]) / nh)
        den += w
    return num / den if den else float("nan")


def cluster_ci(per_seed, seed, draws=DRAWS):
    """95% percentile interval, resampling SEEDS with replacement.

    A seed drawn twice enters as two independent strata under distinct keys, so
    it is weighted twice -- that is the point of a cluster bootstrap, not a bug.
    """
    seeds = sorted(per_seed)
    rnd = random.Random(seed)
    n = len(seeds)
    vals = []
    for _ in range(draws):
        drawn = {}
        for i in range(n):
            s = seeds[rnd.randrange(n)]
            drawn[(s, i)] = per_seed[s]
        v = cmh_delta(drawn)
        if v == v:
            vals.append(v)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def band_stream(v, m, b):
    """The band cell's resample replicates: (pass rate list, mean score list).

    Split out of band_sd so that zero_bound_cells() can take a PERCENTILE INTERVAL
    off the very stream whose standard deviation the table prints, instead of
    drawing a second one. corpus.pct_ci exists for the same reason and says so.
    A cell's +- and its bracket therefore cannot disagree: they are two
    summaries of one list.

    The stream is named after the model, the band and the band's pass count
    rather than seeded from a bare count: two models with the same number of
    passes in the same band would otherwise share a stream, which is not wrong
    but makes their +- move together for no reason a reader could infer.
    """
    return corpus.boot_vals(
        [(e["task_id"], e["success"], e["score"]) for e in v],
        (lambda ps: 100.0 * sum(t[1] for t in ps) / len(ps),
         lambda ps: sum(t[2] for t in ps) / len(ps)),
        seed=corpus.stable_seed("bands-cell", m, b,
                                sum(bool(e["success"]) for e in v)))


def band_sd(v, m, b):
    """Bootstrap standard errors of a band's pass rate and mean score.

    ONE resample stream feeding both, for the reason corpus.boot_vals gives.
    The whole-corpus band does NOT come through here -- it is table1_v8.cell()'s,
    so the "all" column keeps having exactly one derivation in the paper.
    """
    pv, sv = band_stream(v, m, b)
    return corpus.boot_sd(pv), corpus.boot_sd(sv)


_PM_RE = re.compile(r"^(-?\d+\.\d+)\{\\scriptsize\$\\pm\$(\d+\.\d+)\}$")


def printed_pm(val, sd):
    """The two numbers a reader actually SEES in a tab:main cell.

    Rendered through _pm and read back, rather than rounded again here, so this
    file has exactly one rounding of a tab:main cell. A second `round(x, 1)`
    would be banker's rounding set against _pm's format-spec rounding -- the
    very pair of disagreeing renderings that print_multi_effort's comment
    records having already been fixed once.
    """
    hit = _PM_RE.match(_pm(val, sd))
    assert hit, f"_pm changed shape; printed_pm cannot read it: {_pm(val, sd)!r}"
    return float(hit.group(1)), float(hit.group(2))


def zero_bound_cells(levels, specs):
    """Band cells whose PRINTED "pt +- sd" reconstructs to a negative pass rate.

    tab:main gives each cell a standard error, so a reader forming the habitual
    symmetric pt +- 1.96 sd lands below zero on a cell pressed against the zero
    bound -- an impossible pass rate, and one the paper never intended to
    claim. The caption names those cells and prints the percentile interval
    that actually applies; this returns what the caption has to agree with.

    Three deliberate choices:

    * The test is run on the ROUNDED numbers, via printed_pm. The negative
      limit is something a READER produces with a calculator from the printed
      digits, so it is the printed digits that have to be tested. A cell can
      therefore be flagged whose full-precision reconstruction clears zero;
      that is correct, because full precision is not on the page.

    * The set is DISCOVERED, never hardcoded: every band cell of every effort
      is tested. If a regenerated corpus presses a third cell to the bound, or
      lifts one of these off it, results_facts fails on the count instead of
      the caption quietly going stale.

    * The interval comes off band_stream, the same replicates whose standard
      deviation the cell prints. A freshly drawn stream would put a bracket and
      a +- from different draws in one caption.

    `at_zero` is the share of replicates in which the cell scored nothing at
    all. It is what makes the asymmetry legible: it is the mass piled against
    the bound that pushes the lower percentile up to 0.0 while the naive
    reconstruction runs past it.
    """
    out = []
    for label, eps, tab in levels:
        for m in sorted(tab):
            for b in BANDS:
                c = tab[m][b]
                pt, sd = printed_pm(c["pass"], c["pass_sd"])
                naive = pt - 1.96 * sd
                if naive >= 0:
                    continue
                v = [e for tid, e in sorted(eps[m].items())
                     if band_of(specs[tid]["gold"]) == b]
                pv, _ = band_stream(v, m, b)
                lo, hi = corpus.pct_ci(pv)
                out.append({
                    "effort": label, "model": m, "band": b,
                    "n": len(v), "k": sum(1 for e in v if e["success"]),
                    "pass": pt, "sd": sd, "naive_lo": naive,
                    "lo": lo, "hi": hi,
                    "at_zero": sum(1 for x in pv if x == 0.0) / len(pv),
                })
    return out


def print_zero_bound_cells(levels, specs):
    """Console block behind tab:main's caption sentence on these cells.

    "zero-bound", not "floor": this file and results_facts.py already use
    "floor" for the bottom-ranked MODEL (main_floor_pass and friends, all of
    them gemini-3.5-flash's whole-corpus row). These are CELLS pressed against
    the zero bound -- a different statement that happens to land on the same
    model, and reusing the word would make the two impossible to grep apart.
    """
    zb = zero_bound_cells(levels, specs)
    print("\n% ---- tab:main cells whose printed +- reconstructs below zero ----")
    print(f"  zero-bound cells: {len(zb)}")
    for c in zb:
        print(f"  zero-bound cell: {c['model']} {c['effort']} {c['band']} "
              f"{c['k']}/{c['n']} pt={c['pass']:.1f} sd={c['sd']:.1f} "
              f"naive_lo={c['naive_lo']:+.2f} pctl=[{c['lo']:.1f},{c['hi']:.1f}] "
              f"at_zero={c['at_zero']:.3f}")
    return zb


def build(eps, specs, with_sd=False):
    """model -> {band: {...}, 'delta': ..., 'ci': ..., 'pooled': ...}.

    `with_sd` adds a bootstrap standard error to each BAND's pass rate and mean
    score. It is off by default and only tab:main (print_multi_effort) turns it
    on: it costs six extra bootstraps per model and the three single-effort
    appendix tables print intervals rather than +-, so paying for it there
    would slow every run of this script to produce numbers nothing reads. The
    keys are ABSENT rather than None when it is off, so a future caller that
    forgets the flag gets a KeyError on the spot instead of a blank cell.
    """
    out = {}
    for m, E in eps.items():
        cells = {b: [] for b in COLS}
        micro = {b: [0, 0] for b in COLS}
        per_seed = collections.defaultdict(lambda: {b: [] for b in BANDS})
        for tid in sorted(E):                    # sorted: see required_vectors
            e = E[tid]
            s = specs[tid]
            b = band_of(s["gold"])
            req = corpus.scored_checks(e)
            # The score-is-the-required-pass-fraction invariant is asserted by
            # table1_v8.cell() on the same episodes, so it is not repeated here.
            for k in (b, "all"):
                micro[k][0] += sum(1 for c in req if c.get("passed"))
                micro[k][1] += len(req)
            cells[b].append(e)
            cells["all"].append(e)
            per_seed[s["seed"]][b].append(bool(e["success"]))

        row = {}
        for b in COLS:
            v = cells[b]
            want = corpus.N_RAW if b == "all" else BAND_N[b]
            assert len(v) == want, (
                f"{m}: band {b} has {len(v)} rounds, expected {want}; "
                f"the corpus changed and the band edges need re-deciding")
            row[b] = {
                "n": len(v),
                "pass": 100.0 * sum(e["success"] for e in v) / len(v),
                "score": sum(e["score"] for e in v) / len(v),
                "micro": micro[b][0] / micro[b][1],
                "req_per_round": micro[b][1] / len(v),
            }
            if with_sd and b != "all":
                row[b]["pass_sd"], row[b]["score_sd"] = band_sd(v, m, b)
        # Overwrite the whole-corpus cell with T1's, so the pass rate, the mean
        # score and the interval in this table are the same objects the paper
        # has been quoting -- and so a drift between the two scripts is a
        # failed assertion here rather than a silent disagreement on the page.
        allc = table1_v8.cell(E)
        for k in ("pass", "score"):
            assert abs(allc[k] - row["all"][k]) < 1e-9, (
                f"{m}: our whole-corpus {k} {row['all'][k]} != T1's {allc[k]}")
        row["all"] = dict(row["all"], **allc)
        row["delta"] = cmh_delta(per_seed)
        row["pooled"] = row["easy"]["pass"] - row["hard"]["pass"]
        row["ci"] = cluster_ci(
            per_seed,
            seed=corpus.stable_seed("bands-delta", m,
                                    sum(e["success"] for e in cells["easy"]),
                                    sum(e["success"] for e in cells["hard"])))
        row["both_bands"] = sum(
            1 for c in per_seed.values() if c["easy"] and c["hard"])
        out[m] = row
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    effort = argv[0] if argv else corpus.PRIMARY_EFFORT
    if effort in ("all", "multi"):
        specs = corpus.load_specs()
        return print_multi_effort(specs)
    if len(argv) > 1 or effort not in corpus.LEVELS:
        print(f"usage: difficulty_bands.py [{'|'.join(corpus.LEVELS)}|all]"
              f"   (default: {corpus.PRIMARY_EFFORT})", file=sys.stderr)
        return 2

    specs = corpus.load_specs()
    # load_panel, not load_eval: at high this is the twelve-model panel, and
    # at medium and low it returns exactly what load_eval returns, because the
    # open-weight group has no run at those levels. The branch lives in corpus
    # so that this script cannot disagree with the other tables about which
    # models exist at which effort.
    eps, errs = corpus.load_panel(effort, keep_checks=True)
    # Row order and block boundaries are pinned to PRIMARY_EFFORT at every
    # level -- see the module docstring. At PRIMARY_EFFORT itself this is the
    # same object, so Table 1 is untouched by the argument existing. At the
    # other two the order must be derived from the NINE, not from the panel:
    # load_eval is therefore the right call here even though the line above is
    # not, and the assert below is what holds the two together.
    order_eps = eps if effort == corpus.PRIMARY_EFFORT else (
        corpus.load_eval(corpus.PRIMARY_EFFORT)[0])
    order = corpus.model_order(order_eps)
    assert set(order) == set(eps), (
        f"{effort} carries {sorted(eps)} but the {corpus.PRIMARY_EFFORT} row "
        f"order carries {sorted(order)}; the three tables would not line up")
    tab = build(eps, specs)

    # Which rows are marked, resolved once from corpus and reused by both the
    # console view and the LaTeX body so the two cannot mark different rows.
    ocr = [m for m in order if m in corpus.ASSET_MODALITY]
    assert all(corpus.ASSET_MODALITY[m] == "ocr" for m in ocr), (
        "ASSET_MODALITY has grown a value this table has no marker for; "
        "a third modality needs a second symbol, not a silent reuse of the "
        "dagger")

    counts = collections.Counter(band_of(v["gold"]) for v in specs.values())
    assert dict(counts) == BAND_N, f"band sizes moved: {dict(counts)}"

    edge = {"all": f"n={corpus.N_RAW}",
            "easy": f"gold<={EASY_MAX}", "mid": f"gold {EASY_MAX+1}-{MID_MAX}",
            "hard": f"gold>={MID_MAX+1}"}
    print(f"T1  difficulty bands   effort={effort}   "
          f"population={corpus.PRIMARY_POPULATION} (n={corpus.N_RAW})"
          + ("" if effort == corpus.PRIMARY_EFFORT else
             f"   rows ordered at {corpus.PRIMARY_EFFORT}"))
    print(f"the axis is slack = budget/gold with budget fixed at "
          f"{corpus.BUDGET_V8}, so banding on slack and on gold length is one "
          f"partition:")
    for b in BANDS:
        lo_s = corpus.BUDGET_V8 / (MID_MAX + 1 if b == "hard" else
                                   EASY_MAX if b == "easy" else MID_MAX)
        print(f"  {b:<5} {edge[b]:<12} n={BAND_N[b]:<4} slack "
              + (f">= {corpus.BUDGET_V8/EASY_MAX:.2f}" if b == "easy" else
                 f"{corpus.BUDGET_V8/MID_MAX:.2f}-{corpus.BUDGET_V8/(EASY_MAX+1):.2f}"
                 if b == "mid" else f"<  {corpus.BUDGET_V8/MID_MAX:.2f}"))
    print()
    # Name column width. corpus.NAME_W already spans the combined roster; the
    # max() only widens it if a marked name would outgrow it, which on the
    # current roster it does not -- so both the nine-model and the
    # twelve-model console tables sit on the same left edge as before.
    cmark = {m: (OCR_MARK_TXT if m in corpus.ASSET_MODALITY else "")
             for m in order}
    cw = max(corpus.NAME_W, max(len(m) + len(cmark[m]) for m in order))
    hdr = (f"{'model':<{cw}} | " + " ".join(f"{b:>7}" for b in COLS)
           + " | " + " ".join(f"{b:>7}" for b in COLS)
           + " | " + " ".join(f"{b:>7}" for b in BANDS)
           + f" | {'strat':>7} {'95% CI':>16} {'pooled':>7}")
    print(f"{'':<{cw}} | {'pass (%)':^31} | {'score (macro)':^31} "
          f"| {'score (micro)':^23} | {'easy - hard (points)':^32}")
    print(hdr)
    print("-" * len(hdr))
    pmono = amono = imono = inv = 0
    for m in order:
        r = tab[m]
        pmono += r["easy"]["pass"] > r["mid"]["pass"] > r["hard"]["pass"]
        amono += r["easy"]["score"] > r["mid"]["score"] > r["hard"]["score"]
        imono += r["easy"]["micro"] > r["mid"]["micro"] > r["hard"]["micro"]
        # Counted separately from amono on purpose: see the docstring. This is
        # the one the prose quotes.
        inv += r["mid"]["score"] > r["easy"]["score"]
        lo, hi = r["ci"]
        print(f"{m + cmark[m]:<{cw}} | "
              + " ".join(f"{r[b]['pass']:>7.1f}" for b in COLS) + " | "
              + " ".join(f"{r[b]['score']:>7.3f}" for b in COLS) + " | "
              + " ".join(f"{r[b]['micro']:>7.3f}" for b in BANDS)
              + f" | {r['delta']:>+7.1f} [{lo:>+6.1f},{hi:>+6.1f}] "
              f"{r['pooled']:>+7.1f}")
    if ocr:
        print(f"{OCR_MARK_TXT} text-only checkpoint, no vision path: the "
              f"rendered assets reached it as an OCR transcription, not as an "
              f"image. Every other row was sent the image.")

    print()
    print(f"pass monotone easy>mid>hard        : {pmono}/{len(order)}")
    print(f"score monotone, macro (per-round)  : {amono}/{len(order)}")
    print(f"score monotone, micro (per-check)  : {imono}/{len(order)}")
    print(f"score inverted, mid > easy         : {inv}/{len(order)}")
    print("  -> the axis orders rounds won, not checks passed. Required checks "
          "per round: " + ", ".join(
              f"{b} {tab[order[0]][b]['req_per_round']:.2f}" for b in BANDS))

    sig = [m for m in order if tab[m]["ci"][0] > 0]
    print(f"\n{len(sig)}/{len(order)} models have a cluster-bootstrap interval "
          f"strictly above zero: " + ", ".join(sig))
    flat = [m for m in order if tab[m]["ci"][0] <= 0]
    if flat:
        print(f"  not separated: " + ", ".join(
            f"{m} ({tab[m]['delta']:+.1f}, pass "
            f"{tab[m]['easy']['pass']:.1f}/{tab[m]['hard']['pass']:.1f})"
            for m in flat))

    # THE SIGN IS NOT FIXED, AND THIS LINE USED TO ASSERT THAT IT WAS. It read
    # "pooling overstates the gap by {min}-{max} points" -- fine at high on the
    # nine API models, where the spread ran +0.1 to +5.8, but already printing
    # "-1.0-4.5" at medium and low, i.e. claiming an overstatement while
    # displaying an understatement, with a stray hyphen for a minus sign. The
    # direction of the confound is a property of how a MODEL's per-seed
    # strengths line up with which seeds are short, so it is a per-model fact
    # and has to be reported as one.
    infl = {m: tab[m]["pooled"] - tab[m]["delta"] for m in order}
    over = [m for m in order if infl[m] > 0]
    under = [m for m in order if infl[m] < 0]
    comp = collections.defaultdict(collections.Counter)
    for v in specs.values():
        comp[v["seed"]][band_of(v["gold"])] += 1
    nohard = [s for s in comp if not comp[s]["hard"]]
    worst = max(comp, key=lambda s: comp[s]["hard"])
    print(f"\nscenario confound: pooling shifts the gap by "
          f"{min(infl.values()):+.1f} to {max(infl.values()):+.1f} points "
          f"({tab[order[0]]['both_bands']}/{len(comp)} seeds hold both bands)")
    print(f"  overstated for {len(over)}/{len(order)} models "
          f"(largest {max(infl.values()):+.1f})"
          + (f", understated for {len(under)} "
             f"(largest {min(infl.values()):+.1f}): "
             + ", ".join(f"{m} ({infl[m]:+.1f})" for m in under)
             if under else ""))
    print(f"  {len(nohard)} seeds hold NO hard rounds and contribute "
          f"{sum(comp[s]['easy'] for s in nohard)} easy ones: "
          + ", ".join(sorted(nohard)))
    print(f"  heaviest hard seed is {worst} with {comp[worst]['hard']} of "
          f"{BAND_N['hard']} hard against {comp[worst]['easy']} easy")
    # Printed even when there are none. A caption that says "no provider errors
    # at this level" has to be reading a statement, not inferring from a line
    # that failed to appear -- which is indistinguishable from a broken print.
    print(f"provider errors retained as failures: {sum(errs.values())} "
          f"({errs or 'none'})")

    # Regrouped from the frozen order, NOT provider_blocks(eps): the midrules
    # have to fall between the same two models on all three tables.
    blocks = corpus.group_by_provider(order)
    print("\n% ---- caption clause forced by the provider blocking ----")
    print("% " + (corpus.blocking_note(eps)
                  if effort == corpus.PRIMARY_EFFORT
                  else corpus.blocking_ref()))
    # Generated for the same reason blocking_note is: a dagger in the body
    # whose meaning is typed by hand into the caption is one regeneration away
    # from pointing at nothing, or at the wrong rows.
    if ocr:
        print("% ---- caption clause forced by the marked rows ----")
        print(f"% {OCR_MARK} "
              + _and_list(f"\\texttt{{{m}}}" for m in ocr)
              + (" are" if len(ocr) > 1 else " is")
              + " text-only, with no vision variant released; the rendered "
                "assets were supplied to "
              + ("them" if len(ocr) > 1 else "it")
              + " as an OCR transcription of the same image. Every other row "
                "received the image itself.")

    print(f"\n% ---- LaTeX body for T1 at {effort} effort "
          f"(paste into the tabular) ----")
    print("%      column spec: l c c ccc cccc c c")
    # Source-alignment width for the name column, over the rows being printed
    # and counting the marker's source length. With no marked row and a
    # 16-character longest name this is 18, which is the constant the body
    # carried before the open-weight group landed -- so the medium and low
    # bodies come out byte-identical and only tab:main moves.
    mark = {m: (OCR_MARK if m in corpus.ASSET_MODALITY else "") for m in order}
    name_w = max(len(m) + len(mark[m]) for m in order) + 2
    for i, (_prov, ms) in enumerate(blocks):
        if i:
            print(r"    \midrule")
        for m in ms:
            r = tab[m]
            alo, ahi = r["all"]["ci"]
            lo, hi = r["ci"]
            # Score as a percentage, matching tab:main -- these three tables
            # say they restate its rows, so a reader who checks one against the
            # other must not have to convert on the way. See _score_pct. The
            # intervals stay: tab:main trades them for a +- to fit its extra
            # effort rows, and this is where the full percentile interval on
            # the pass rate is still printed.
            cols = ([f"{r['all']['pass']:.1f}", f"[{alo:.1f},\\,{ahi:.1f}]"]
                    + [f"{r[b]['pass']:.1f}" for b in BANDS]
                    + [f"{_score_pct(r[b]['score']):.1f}" for b in COLS]
                    + [f"{r['delta']:+.1f}", f"[{lo:+.1f},\\,{hi:+.1f}]"])
            print(f"    \\texttt{{{m}}}{mark[m]}"
                  + " " * (name_w - len(m) - len(mark[m]))
                  + " & " + " & ".join(cols) + r" \\")
    return 0


# Was a local copy. tab:capability builds the same sentence from the same
# ASSET_MODALITY set and had the ungrammatical three-item join this helper
# exists to prevent, so the helper moved to corpus and both callers share it.
_and_list = corpus.and_list


def _pm(val, sd):
    """A tab:main body cell: point estimate, then its bootstrap SE.

    WIDTH IS THE WHOLE DESIGN HERE, so the reasoning is written down rather
    than left to look like fussiness. tab:main measured 478.75pt against a
    472.28pt text block BEFORE this column group grew a +- on every cell (see
    the \\adjustbox note at the top of main.tex), and \\adjustbox silently
    shrinks the paper's headline table by whatever it overruns. Rendered the
    obvious way -- full-size "$\\pm$", thin spaces around it, the mean score
    left as a 0.xxx fraction -- the table came out 650.6pt, i.e. set at 72.6%,
    which is a 6.5pt body font. The three economies below bring it to 513.8pt
    and 91.9%, within half a point of the size it has always been printed at:

      no thin space before the SE   the \\scriptsize step already separates the
                                    two numbers, and \\tabcolsep separates the
                                    columns; a third gap only blurs which of
                                    the two the eye should group
      SE set \\scriptsize            it is a second-order quantity and reads as
                                    one
      score as a percentage         see _score_pct

    Do not "tidy" any of them back without re-measuring; the note in main.tex
    explains how (comment the \\adjustbox out and read the overfull report).
    """
    return f"{val:.1f}{{\\scriptsize$\\pm${sd:.1f}}}"


def _score_pct(x):
    """Mean score as a percentage, which is what the paper has always CALLED it.

    The tables printed 0.642 while tab:main's caption said "Mean Score (%):
    average percentage of atomic checks satisfied per round" and Sec 6.1's
    Finding 1 quoted "64.2% of atomic evaluation checks" -- three renderings of
    one quantity, one of which made the reader do the conversion. They now
    agree, and the eight statistic columns of tab:main are homogeneous: every
    one is a percentage carried to one decimal.

    Only the LaTeX bodies are converted. The console view and every dict key
    stay fractions, because analysis/results_facts.py pins the score off the
    console (main_top_score = 0.642) and a display change must not be able to
    move a checked number.
    """
    return 100.0 * x


def _t1_cols(r, c, _grp):
    """The ten statistic cells of one tab:main row -- everything the caller
    prints after the Model and Think Level columns, which is why the table is
    twelve columns wide and this returns ten.

    Shared by the three-effort and the high-only branch below, which used to
    carry two copies of this expression -- and would now carry two copies of
    the +- formatting, one of which would eventually be edited alone.

    THE COST PAIR TAKES NO +-. cost_total and tok_total are census sums over
    the 600 rounds that actually ran, not estimates of anything resampling
    could vary; a bootstrap SE on them would be a fabricated uncertainty for a
    quantity that has none. The pass rates and mean scores are sample means
    over rounds and do carry one.
    """
    return ([_pm(r["all"]["pass"], r["all"]["pass_sd"])]
            + [_pm(r[b]["pass"], r[b]["pass_sd"]) for b in BANDS]
            + [_pm(_score_pct(r[b]["score"]), _score_pct(r[b]["score_sd"]))
               for b in COLS]
            + [_grp(c["cost_total"]), _grp(c["tok_total"] / 1e6)])


def print_multi_effort(specs):
    import cost

    # The two run columns are the SAME quantities Table E1 prints, so they are
    # grouped by the same function rather than by a local copy of it. The copy
    # that used to live here rounded with python's round(), which is banker's
    # rounding, while cost._grp rounds half-up and says at length why -- two
    # renderings of one number that settle a tie in opposite directions. It
    # also carries the no-published-rate cell, so an unpriced model renders
    # here exactly as it does there instead of dying on int(None).
    _grp = cost._grp

    eps_high, _ = corpus.load_panel("high", keep_checks=True)
    eps_med, _ = corpus.load_eval("medium", keep_checks=True)
    eps_low, _ = corpus.load_eval("low", keep_checks=True)
    # with_sd: tab:main is the one table that prints a +- per cell rather than
    # an interval on the primary column. See build()'s docstring for why the
    # three single-effort tables below do not pay for it.
    tab_high = build(eps_high, specs, with_sd=True)
    tab_med = build(eps_med, specs, with_sd=True)
    tab_low = build(eps_low, specs, with_sd=True)

    cost_high, _, _ = cost.build("high", panel=True)
    cost_med, _, _ = cost.build("medium", panel=False)
    cost_low, _, _ = cost.build("low", panel=False)

    order = corpus.model_order(eps_high)
    blocks = corpus.group_by_provider(order)
    mark = {m: (OCR_MARK if m in corpus.ASSET_MODALITY else "") for m in order}
    name_w = max(len(m) + len(mark[m]) for m in order) + 2

    print("\n% ---- LaTeX body for T1 across all efforts (paste into the tabular) ----")
    print("%      column spec: l c | cccc | cccc | cc")
    for i, (_prov, ms) in enumerate(blocks):
        if i:
            print(r"    \midrule")
        for j, m in enumerate(ms):
            if j > 0 and (m in tab_med):
                print(r"    \addlinespace[2.5pt]")
            if m in tab_med and m in tab_low:
                efforts = [("High", tab_high[m], cost_high[m]),
                           ("Med", tab_med[m], cost_med[m]),
                           ("Low", tab_low[m], cost_low[m])]
                for k, (eff_label, r, c) in enumerate(efforts):
                    cols = _t1_cols(r, c, _grp)
                    if k == 0:
                        prefix = f"    \\multirow{{3}}{{*}}{{\\texttt{{{m}}}{mark[m]}}}" + " " * (name_w - len(m) - len(mark[m]))
                    else:
                        prefix = " " * (name_w + 26)
                    print(f"{prefix} & {eff_label:<4} & " + " & ".join(cols) + r" \\")
            else:
                cols = _t1_cols(tab_high[m], cost_high[m], _grp)
                prefix = f"    \\texttt{{{m}}}{mark[m]}" + " " * (name_w - len(m) - len(mark[m]) + 13)
                print(f"{prefix} & High & " + " & ".join(cols) + r" \\")

    # The Think Level labels are the table's own ("High"/"Med"/"Low"), so a
    # flagged cell can be found on the page by reading the caption.
    print_zero_bound_cells([("High", eps_high, tab_high),
                            ("Med", eps_med, tab_med),
                            ("Low", eps_low, tab_low)], specs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
