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

"""T8 -- capability by conflict type: 6 conflict categories x the high panel.

Feeds Section 6.2. The headline table says the frontier models are hard to
tell apart on one pooled number; this one asks whether the pooled number is
hiding a division of labour -- some model that is mediocre overall but strong
on, say, ordering conflicts.

THE PANEL IS THE ONE TABLE 1 SHOWS, and it is twelve rows: the nine API models
plus the three open-weight ones, which were run at high effort only -- which is
the only effort this table is built at. Two of the twelve are text-only
checkpoints that received each rendered asset as an OCR transcription rather
than as an image (corpus.ASSET_MODALITY). Their rows are marked, and the mark
matters more here than in Table 1: this table invites the reader to compare
DOWN a column, and two of the cells in every column were produced under a
different input modality.

HOW TO READ THIS TABLE -- two warnings, both load-bearing.

(1) The category buckets OVERLAP and do not partition the corpus. A seed
    carries every category its plugins route to, so the 20 seeds land in 32
    category slots (9/6/6/6/4/1) and the per-category round counts sum to 960,
    not 600. Column counts must never be added. Each column carries its own
    denominator for exactly this reason, and the script asserts the overlap is
    still there rather than silently emitting a table that looks like a
    partition.

(2) Read DOWN a column, not ACROSS a row. Down a column is a within-seed
    comparison: every model saw the same rounds, so a difference is about the
    models. Across a row is confounded with seed identity -- `main_supervised`
    is ONE seed (30 rounds), so "main_supervised is the hardest category" and
    "that one seed is hard" are the same statement and this design cannot
    separate them. The per-category `n` and the bootstrap half-width are
    printed so that the 30-round column is visibly not a measurement of a
    category.

`challenge` is deliberately absent: all 20 seeds carry challenge plugins, so
the bucket is the whole corpus. corpus.CAT_ORDER makes the same exclusion for
the same reason, and design_axis_slice.py did before it.

Prints a console view and a LaTeX body from one dict, so they cannot drift.
"""
import sys
from collections import defaultdict

import corpus

EFFORT = corpus.PRIMARY_EFFORT


def category_seeds():
    """category -> frozenset of base seed names, from the generator registry."""
    amap = corpus.load_axis_map()
    out = defaultdict(set)
    for seed, v in amap["seeds"].items():
        for c in v["categories"]:
            out[c].add(seed)
    assert set(out) == set(corpus.CAT_ORDER), (
        f"axis map categories {sorted(out)} != CAT_ORDER {list(corpus.CAT_ORDER)}")
    slots = sum(len(s) for s in out.values())
    assert slots > len(amap["seeds"]), (
        f"{slots} category slots over {len(amap['seeds'])} seeds: the buckets no "
        "longer overlap, so warning (1) in this docstring is stale -- rewrite it "
        "before emitting a table that now IS a partition")
    return {c: frozenset(out[c]) for c in corpus.CAT_ORDER}, amap


def slice_rows(rows, seeds):
    """The episodes whose seed (slug minus the _vNN variant) is in `seeds`."""
    return [e for e in rows if corpus.base_slug(e["slug"]) in seeds]


def specialization(eps, order, catseeds, amap, draws=4000, rng_seed=20260806):
    """Is any model better at a category than at the corpus as a whole?

    The eyeball test on the grid is not a result: six spreads read off nine
    noisy cells will always show a widest one. The statistic here is the
    SPECIALIZATION residual

        d(m, c) = pass(m, c) - pass(m, all)

    which is zero for every model under "no division of labour", whatever the
    category's difficulty is. The test statistic per category is the spread of
    d over models -- one model pulling ahead OF ITS OWN AVERAGE where another
    falls behind is exactly what specialization would look like.

    The null is a permutation over SEEDS, not over rounds: draw a random set of
    the same size from the 20 seeds and recompute. This is the right null
    because it holds the thing that is actually confounded -- seed identity and
    seed count -- fixed by construction. A round-level bootstrap would let the
    30-round column borrow precision it does not have.

    The frontier cut is corpus.leading_group: the models an all-pairs exact
    McNemar cannot separate from the best one, which on this panel is nine of
    the fourteen. It used to be `rank_order(eps)[:6]`, chosen over the
    provider-blocked `order[:6]` because those two had quietly stopped naming
    the same models. Taking the ranked six fixed the immediate mismatch but
    left "six" as a free parameter, and three sections went on meaning three
    different sets by "the top six". leading_group has no free parameter, is
    read off the same test the section already reports, and cuts at the same
    place on the 10-model API panel as on this one -- see its docstring.

    Returns (text, res) where res maps category -> the spreads and p-values.
    The table emitter needs the numbers, and re-deriving them from the printed
    text, or recomputing them under a second RNG, would be two ways to publish
    a p-value that does not match the one in this console block.
    """
    import random
    rnd = random.Random(rng_seed)
    seeds = sorted(amap["seeds"])
    by_seed = {m: defaultdict(list) for m in order}
    for m in order:
        for e in eps[m].values():
            by_seed[m][corpus.base_slug(e["slug"])].append(e["success"])
    overall = {m: 100.0 * sum(sum(v) for v in by_seed[m].values())
               / sum(len(v) for v in by_seed[m].values()) for m in order}

    def resid(m, sset):
        hit = sum(sum(by_seed[m][s]) for s in sset)
        n = sum(len(by_seed[m][s]) for s in sset)
        return 100.0 * hit / n - overall[m]

    def spread(sset, labs):
        d = [resid(m, sset) for m in labs]
        return max(d) - min(d)

    top6 = corpus.leading_group(eps)
    out = ["specialization test -- H0: no model is better at a category than at",
           "the corpus overall. d(m,c) = pass(m,c) - pass(m,all); statistic is",
           f"spread of d across models; null = {draws} random seed sets of the",
           "same size drawn from the 20 seeds.",
           "",
            f"  {'category':<24} {'k':>2} {f'spread{len(order)}':>8} "
            f"{'p' + str(len(order)):>7} {f'spread{len(top6)}':>8} "
            f"{'p' + str(len(top6)):>7}"]
    worst = []
    res = {}
    for c in corpus.CAT_ORDER:
        k = len(catseeds[c])
        obs9, obs6 = spread(catseeds[c], order), spread(catseeds[c], top6)
        ge9 = ge6 = 0
        for _ in range(draws):
            s = frozenset(rnd.sample(seeds, k))
            ge9 += spread(s, order) >= obs9
            ge6 += spread(s, top6) >= obs6
        p9, p6 = (ge9 + 1) / (draws + 1), (ge6 + 1) / (draws + 1)
        worst.append((p9, p6, c))
        res[c] = {"k": k, "spread9": obs9, "p9": p9, "spread6": obs6, "p6": p6}
        out.append(f"  {c:<24} {k:>2} {obs9:8.1f} {p9:7.3f} {obs6:8.1f} {p6:7.3f}")

    a = corpus.bonferroni(len(corpus.CAT_ORDER))
    sig9 = [c for p9, _, c in worst if p9 < a]
    sig6 = [c for _, p6, c in worst if p6 < a]
    out += ["",
            f"  Bonferroni alpha = 0.05/{len(corpus.CAT_ORDER)} = {a:.4f}",
            f"  significant over all {len(order)} models: {sig9 or 'none'}",
            f"  significant over the leading {len(top6)}:  {sig6 or 'none'}",
            f"  leading group: {', '.join(top6)}"]
    if not sig9 and not sig6:
        out.append("  => no detectable division of labour: the per-category "
                   "grid is the")
        out.append("     pooled ranking plus sampling noise, at every "
                   "denominator tested.")
    return "\n".join(out), res


def main():
    catseeds, amap = category_seeds()
    # load_panel, not load_eval: this table is built at high effort only, which
    # is exactly the effort the open-weight group exists at, so it shows the
    # same twelve rows Table 1 does. keep_checks stays off -- nothing here
    # reads a check, and the panel is a third larger than it was.
    eps, errors = corpus.load_panel(EFFORT, keep_checks=False)
    order = corpus.model_order(eps)
    mark = {m: (r"$^\dagger$" if m in corpus.ASSET_MODALITY else "")
            for m in order}
    cmark = {m: ("*" if m in corpus.ASSET_MODALITY else "") for m in order}
    nw = max(corpus.NAME_W, max(len(m) + len(cmark[m]) for m in order))

    tab = {}
    ns = {}
    for m in order:
        rows = list(eps[m].values())
        tab[m] = {"all": corpus.pass_pct(rows)}
        for c in corpus.CAT_ORDER:
            sub = slice_rows(rows, catseeds[c])
            tab[m][c] = corpus.pass_pct(sub)
            ns.setdefault(c, len(sub))
            assert ns[c] == len(sub), (
                f"{m}: category {c} has {len(sub)} rounds, {ns[c]} for an earlier "
                "model -- the models are not seeing the same slice")
    ns["all"] = len(eps[order[0]])

    # Bootstrap half-width per category, pooled over models: how much of the
    # spread down a column could be sampling noise at that denominator.
    half = {}
    for c in list(corpus.CAT_ORDER) + ["all"]:
        best = order[0]
        rows = (slice_rows(list(eps[best].values()), catseeds[c])
                if c != "all" else list(eps[best].values()))
        pairs = [(e["task_id"], e["success"]) for e in rows]
        lo, hi = corpus.boot_ci(
            pairs, lambda ps: 100.0 * sum(s for _, s in ps) / len(ps),
            seed=corpus.stable_seed("capability-half", c))
        half[c] = (hi - lo) / 2.0

    cols = ["all"] + list(corpus.CAT_ORDER)
    short = {"inconsistency": "incons", "coordination": "coord", "order": "order",
             "timeout": "timeout", "subagent_collaboration": "subagent",
             "main_supervised": "mainsup", "all": "ALL"}

    print(f"T8  capability by conflict type -- {EFFORT} effort, "
          f"{corpus.N_RAW} rounds/model, pass %")
    print(f"    buckets overlap: 20 seeds -> "
          f"{sum(len(s) for s in catseeds.values())} category slots; "
          f"column n sums to {sum(ns[c] for c in corpus.CAT_ORDER)}, not "
          f"{corpus.N_RAW}. DO NOT ADD COLUMNS.\n")

    print(f"{'model':<{nw}} " + " ".join(f"{short[c]:>8}" for c in cols))
    print(f"{'seeds':<{nw}} " + " ".join(
        f"{(len(catseeds[c]) if c != 'all' else len(amap['seeds'])):>8}" for c in cols))
    print(f"{'n rounds':<{nw}} " + " ".join(f"{ns[c]:>8}" for c in cols))
    print(f"{'boot +-':<{nw}} " + " ".join(f"{half[c]:>8.1f}" for c in cols))
    print("-" * (11 + 9 * len(cols)))
    for m in order:
        print(f"{m + cmark[m]:<{nw}} " + " ".join(f"{tab[m][c]:>8.1f}" for c in cols))
    if any(cmark.values()):
        print("* text-only checkpoint: assets arrived as an OCR transcription, "
              "not as an image")

    print(f"\n{'spread':<{nw}} " + " ".join(
        f"{max(tab[m][c] for m in order) - min(tab[m][c] for m in order):>8.1f}"
        for c in cols))
    # The six best BY PASS RATE, not the first six rows -- see specialization().
    top6 = corpus.leading_group(eps)
    print(f"{'top-6':<{nw}} " + " ".join(
        f"{max(tab[m][c] for m in top6) - min(tab[m][c] for m in top6):>8.1f}"
        for c in cols))

    # The question Section 6.2 is actually asking: does any model's ranking
    # move between categories, or is the pooled order just repeated six times?
    #
    # THE BASELINE IS rank_order, NOT `order`. `order` is corpus.model_order --
    # the table's presentation order, which blocks by provider and pins the
    # open-weight block last "ahead of no one, whatever it scored". Measuring a
    # per-category rank against THAT scores a presentation choice as a move:
    # kimi-k3 is fourth by pass rate and tenth by display, so every category it
    # does not lose counts as a flip. model_order's own docstring sends "who
    # beat whom" callers here instead. The sentence below claims a move against
    # the POOLED ORDER, and the pooled order is the pass-rate ranking -- 43 of
    # 72 against it, against 60 of 72 for the display order.
    print("\nper-category ranking of the models (best first):")
    flips = 0
    base = corpus.rank_order(eps)
    for c in corpus.CAT_ORDER:
        rank = sorted(base, key=lambda m: -tab[m][c])
        moved = [m for i, m in enumerate(rank) if base.index(m) != i]
        flips += len(moved)
        print(f"  {c:<24} " + " > ".join(rank))
    print(f"\n  {flips} of {len(corpus.CAT_ORDER) * len(order)} model-category "
          "cells sit at a different rank than the pooled order")

    spec_text, spec = specialization(eps, order, catseeds, amap)
    print("\n" + spec_text)

    print("\n% ---- caption clause forced by the provider blocking ----")
    # T1 owns the explanation of why the pass column is not monotone; this
    # table shares that column and only points at it, which is one caption
    # line cheaper than saying it twice.
    print("% " + corpus.blocking_ref())
    emit_body(eps, order, cols, tab, ns, half, catseeds, amap, spec, errors,
              mark)
    return 0


def emit_body(eps, order, cols, tab, ns, half, catseeds, amap, spec, errors,
              mark):
    """The grid, wrapped in the denominators above and the test below.

    The grid alone reads as a "who is best at what" table. It is not one --
    the result here is a NULL -- and three things have to be on the page for a
    reader to see that.

    ROWS ABOVE THE GRID: seeds, rounds, and the bootstrap half-width per
    column. Without the half-width the 30-round column and the 270-round
    column carry identical visual weight, and the 30-round one is +-15.0: its
    entire range is noise. This is the single most important row in the table
    and it is why the grid is not a ranking.

    ROWS BELOW THE GRID: the test. The statistic is the spread of the
    SPECIALIZATION RESIDUAL d(m,c) = pass(m,c) - pass(m,all), NOT the spread of
    the pass rates in the column -- those are different numbers (order: 7.4 for
    d against 28.9 for pass) and printing the pass spread next to this p-value
    would tell the reader the test tests something it does not. The `all`
    column is blank throughout because d(m,all) = 0 by construction.

    Both denominators are shown because the frontier cluster is the interesting
    case: nine models include gemini-3.5-flash, whose collapse can dominate a
    spread on its own.
    """
    print("\n% ---- LaTeX body for T8 (paste into the tabular) ----")
    print(r"%      column spec: l ccccccc")
    print(r"% Header rows (denominators; the counts OVERLAP and must not be summed):")
    print("    \\textbf{Seeds} & " + " & ".join(
        str(len(amap["seeds"]) if c == "all" else len(catseeds[c]))
        for c in cols) + r" \\")
    print("    \\textbf{Rounds} & " + " & ".join(str(ns[c]) for c in cols) + r" \\")
    print(r"    \textbf{Bootstrap $\pm$} & " + " & ".join(f"{half[c]:.1f}" for c in cols)
          + r" \\")
    # The rules between the three panels are emitted here, not typed into
    # main.tex, so the whole block is one contiguous paste that can be
    # diffed against this generator. Two hand-inserted \midrule lines used
    # to sit where these comments are, which broke that diff.
    print(r"    \midrule")
    print(r"% Grid, blocked by provider to match the main table row for row:")
    # Source-alignment width over the rows actually printed, counting the
    # marker. With no marked row and a 16-character longest name this is the
    # literal 18 the emitter used before the open-weight group landed.
    name_w = max(len(m) + len(mark[m]) for m in order) + 2
    for i, (prov, ms) in enumerate(corpus.provider_blocks(eps)):
        if i:
            print(r"    \midrule")
        for m in ms:
            print(f"    \\texttt{{{m}}}{mark[m]}"
                  + " " * (name_w - len(m) - len(mark[m])) + " & "
                  + " & ".join(f"{tab[m][c]:.1f}" for c in cols) + r" \\")
    print(r"    \midrule")
    print(r"% Footer rows (the test; blank under `all` because d(m,all)=0):")

    def frow(label, key, fmt):
        return (f"    {label} & " + " & ".join(
            "" if c == "all" else fmt(spec[c][key]) for c in cols) + r" \\")

    num = lambda v: f"{v:.1f}"
    pv = lambda v: f"{v:.3f}".lstrip("0")
    print(frow(r"Spread of $d$", "spread9", num))
    print(frow(r"\quad $p$", "p9", pv))
    print(frow(rf"Spread of $d$, leading {len(corpus.leading_group(eps))}",
               "spread6", num))
    print(frow(r"\quad $p$", "p6", pv))

    a = corpus.bonferroni(len(corpus.CAT_ORDER))
    n_sig = sum(1 for c in corpus.CAT_ORDER
                if spec[c]["p9"] < a or spec[c]["p6"] < a)
    print(f"% caption facts: bootstrap half-width runs "
          f"{min(half[c] for c in cols):.1f} to {max(half[c] for c in cols):.1f} "
          f"across columns, so the {ns[corpus.CAT_ORDER[-1]]}-round column is "
          f"+-{half[corpus.CAT_ORDER[-1]]:.1f} and its whole range is noise; "
          f"the specialization test returns {n_sig or 'no'} significant "
          f"categor{'y' if n_sig == 1 else 'ies'} of "
          f"{len(corpus.CAT_ORDER)} at Bonferroni alpha = "
          f"{a:.4f}, over all {len(order)} models and over the "
          f"leading {len(corpus.leading_group(eps))} alike")
    ocr = [m for m in order if m in corpus.ASSET_MODALITY]
    if ocr:
        print(r"% caption clause forced by the marked rows: $^\dagger$ "
              # corpus.and_list, not " and ".join: this set is derived from
              # ASSET_MODALITY and reached three entries, at which point the
              # naive join prints "a and b and c" into the caption.
              + corpus.and_list(f"\\texttt{{{m}}}" for m in ocr)
              + (" are" if len(ocr) > 1 else " is")
              + " text-only, with no vision variant released; every rendered "
                "asset reached "
              + ("them" if len(ocr) > 1 else "it")
              + " as an OCR transcription rather than as an image.")
    if errors:
        print(f"% provider-error rows retained: {dict(errors)}")


if __name__ == "__main__":
    sys.exit(main())
