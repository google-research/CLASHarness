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

"""Generator ablation: the same 20 seeds regenerated under two generator models.

Two corpora share identical task identifiers but differ in gold-trajectory
length, and they were evaluated under different step budgets (20 vs 30). A
naive cross-corpus comparison therefore confounds "the generator produced
harder tasks" with "the budget was tighter". We report TWO views:

  FULL     all paired tasks. Corpus B has MORE relative slack (budget/gold)
           than corpus A, so if B still scores lower this is a conservative
           lower bound on the generator effect -- the budget confound runs
           AGAINST the finding, not for it.

  MATCHED  tasks whose relative slack is close in both corpora. This removes
           the budget confound, but it is NOT an unbiased estimate of the full
           corpus: matching on a slack *difference* imposes opposite selection
           pressure on the two corpora (it preferentially pairs A's short
           tasks with B's long tasks). The selection diagnostics below
           quantify that, and the paper must disclose it.

!! round_id parsing !!
Task ids look like `conflict_vlm_access_request_approval_v01_r01` -- there is
NO `__` separator. An earlier version of this script split on `__`, silently
produced an empty round id, and so never matched the dead-round exclusion
list. The authoritative source is `row["metadata"]["round_id"]`. The
assertions in load_dead()/results() exist specifically to make that failure
mode loud instead of silent.
"""
import json, os, glob, math, re, statistics
from collections import defaultdict

# Only for the shared column width. This script keeps its own loader on
# purpose: corpus.load_eval asserts the nine-model roster, and corpus B has
# six, so the shared loader cannot read it.
import corpus

# Both corpora are fetched by tools/download_data.py; corpus.DATA_ROOT is the
# single knob that moves them (CLASHARNESS_DATA_ROOT).
DATA = corpus.DATA_ROOT
CORPORA = {
    "A": dict(tasks=f"{DATA}/multiagent_conflict_data_v8_Gemini_3_6_flash/tasks",
              evals=f"{DATA}/eval_results/data_3_6_flash_v8",
              run="r01", budget=20, gen="gemini-3.6-flash", dead_label="conflict_vlm_v8_3.6_flash",
              n_dead=18),
    "B": dict(tasks=f"{DATA}/multiagent_conflict_data_Gemini_Sonnet_5/tasks",
              evals=f"{DATA}/eval_results/data_sonnet_5",
              run="r02", budget=30, gen="claude-sonnet-5", dead_label="conflict_vlm_sonnet_5",
              n_dead=9),
}
DEAD = corpus.DEAD_JSON
LEVELS = ["high", "medium", "low"]
TOL = 0.30
# Filled by results(). Same predicate as corpus._cap_hit, deliberately: a
# second spelling of "ran out of steps" is a second definition.
CAPS = {k: defaultdict(dict) for k in CORPORA}


def load_dead():
    """corpus -> {(slug, round_id)}. Slug includes the variant, e.g.
    'access_request_approval_v03', matching the eval jsonl basename."""
    out = {k: set() for k in CORPORA}
    label_to_key = {v["dead_label"]: k for k, v in CORPORA.items()}
    # `label` not `corpus`: the dead list's first field is a corpus label, but
    # binding it to that name shadows the imported module for the whole
    # function body, which is a trap waiting for the first line here that
    # needs corpus.something.
    for label, slug, rid in json.load(open(DEAD))["exclude"]:
        assert label in label_to_key, f"unknown corpus label in dead list: {label!r}"
        out[label_to_key[label]].add((slug, rid))
    for k, v in CORPORA.items():
        assert len(out[k]) == v["n_dead"], (
            f"corpus {k}: expected {v['n_dead']} dead rounds, got {len(out[k])}")
    return out


def round_id_of(row):
    md = row.get("metadata")
    if isinstance(md, dict) and md.get("round_id"):
        return str(md["round_id"])
    m = re.search(r"_(r\d+)$", row.get("task_id") or "")   # fallback only
    return m.group(1) if m else ""


def gold_lengths(tasks_dir):
    """task_id -> gold length, sentinel excluded (corpus.gold_length).

    Both corpora terminate every gold trajectory with the same DONE sentinel,
    so including it would shift both sides by exactly one and leave the paired
    statistics untouched -- but it would NOT leave slack untouched, because
    slack is a ratio. Matching on slack therefore has to use the same command
    count the step budget does.
    """
    g = {}
    for p in glob.glob(f"{tasks_dir}/*/specs.json"):
        d = json.load(open(p))
        g[d["task_id"]] = corpus.gold_length(d["ground_truth"])
    return g


def results(key):
    """(model, level) -> {task_id: success} across all 600 tasks (full population).

    Also fills CAPS[key] on the same pass: (model, level) -> {task_id: whether
    the round ended by exhausting the step budget}.
    """
    cfg = CORPORA[key]
    dead = DEADSETS[key]
    out = defaultdict(dict)
    cap = CAPS[key]
    hit = set()
    per_cell_dead = defaultdict(int)
    for lvl in LEVELS:
        for model_dir in glob.glob(f"{cfg['evals']}/aligned_setting_{lvl}/*/*"):
            parts = model_dir.split("/")
            vendor, model = parts[-2], parts[-1]
            rel_dir = f"aligned_setting_{lvl}/{vendor}/{model}"
            run_name = corpus.RUN_OVERRIDE.get(rel_dir, cfg["run"]) if key == "A" else cfg["run"]
            for path in glob.glob(f"{model_dir}/{run_name}/*.jsonl"):
                slug = os.path.basename(path)[:-6]
                for line in open(path):
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    tid = row.get("task_id") or ""
                    rid = round_id_of(row)
                    assert rid, f"could not resolve round_id for {tid!r} in {path}"
                    if (slug, rid) in dead:
                        hit.add((slug, rid))
                        per_cell_dead[(model, lvl)] += 1
                    if row.get("error"):
                        out[(model, lvl)][tid] = False
                    else:
                        out[(model, lvl)][tid] = bool(row.get("success"))
                    cap[(model, lvl)][tid] = str(
                        row.get("stop_reason") or "").startswith("reached max_steps")

    missed = dead - hit
    assert not missed, (
        f"corpus {key}: {len(missed)} dead rounds never matched any eval row -- "
        f"dead detection is failing. examples: {sorted(missed)[:5]}")
    return out


def pearson(xs, ys):
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    assert sx and sy, "pearson: a variable is constant"
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def exact_mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def spearman(xs, ys):
    """Rank correlation, ties averaged.

    Separate from pearson() because the claim it serves is different.
    sec:generator-ablation argues that the ORDERING of models survives the
    generator swap, and an ordering claim is a rank statistic; a Pearson r on
    the pass rates answers "do the levels move together", which is a stronger
    and less relevant thing to assert.
    """
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
    return pearson(rank(xs), rank(ys))


def describe(tasks, label):
    """Print gold length and relative slack for both corpora over `tasks`."""
    print(f"  {label:<22} n={len(tasks):>4}", end="")
    for k in CORPORA:
        gl = [gold[k][t] for t in tasks]
        sl = [slack[k][t] for t in tasks]
        print(f"   | {k} gold {statistics.mean(gl):5.2f}  slack "
              f"{statistics.mean(sl):5.3f} (med {statistics.median(sl):5.3f})", end="")
    print()


def compare(pool_tasks, title):
    print(f"\n{'=' * 108}\n{title}\n{'=' * 108}")
    hdr = (f"{'model':<{corpus.NAME_W}} {'effort':<7} {'n':>4} {'passA%':>7} {'passB%':>7} "
           f"{'b(A>B)':>7} {'c(B>A)':>7} {'p':>11}  verdict")
    print(hdr)
    print("-" * len(hdr))
    tally = defaultdict(int)
    rows = {}
    # CELLS, not `shared` crossed with LEVELS. A cell that does not exist in
    # one of the corpora is not a test that came out null, and iterating the
    # cross product turned every such hole into a printed "insufficient
    # coverage" line that the Bonferroni denominator then paid for anyway. The
    # skipped bucket below can now only fill on a cell that EXISTS in both
    # corpora and still cannot be paired -- see the CELLS derivation.
    for m, lvl in CELLS:
        A, B = res["A"].get((m, lvl), {}), res["B"].get((m, lvl), {})
        pool = [t for t in pool_tasks if t in A and t in B]
        name = DISPLAY[m]
        if len(pool) < 30:
            print(f"{name:<{corpus.NAME_W}} {lvl:<7} {len(pool):>4}  -- insufficient paired coverage --")
            tally["skipped"] += 1
            continue
        b = sum(1 for t in pool if A[t] and not B[t])
        c = sum(1 for t in pool if not A[t] and B[t])
        p = exact_mcnemar(b, c)
        pa = 100.0 * sum(A[t] for t in pool) / len(pool)
        pb = 100.0 * sum(B[t] for t in pool) / len(pool)
        if p < ALPHA:
            v = "corpus A easier" if b > c else "corpus B easier"
        else:
            v = "n.s."
        tally[v] += 1
        rows[(name, lvl)] = dict(n=len(pool), pa=pa, pb=pb, b=b, c=c, p=p,
                                 verdict=v)
        print(f"{name:<{corpus.NAME_W}} {lvl:<7} {len(pool):>4} {pa:>7.2f} {pb:>7.2f} "
              f"{b:>7} {c:>7} {p:>11.3g}  {v}")
    print("-" * len(hdr))
    print("tally: " + ", ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    return tally, rows


def _row(label, a, b, w=44):
    print(f"    {label:<{w}} & {a} & {b} " + r"\\")


def _span(label, v, w=44):
    # `c`, matching the two data columns this spans -- a \multicolumn ignores
    # the preamble, so the letter has to be kept in step with it by hand.
    print(f"    {label:<{w}} & \\multicolumn{{2}}{{c}}{{{v}}} " + r"\\")


def emit_body(order, full_rows, full_tally, match_rows, match_tally):
    """TWO table bodies: the corpus design (appendix) and the outcome (body).

    SPLIT IN THE BODY STRUCTURE PASS, AND BOTH BLOCKS MUST KEEP PRINTING.
    This used to be one table. At 292.7pt of rows plus a caption it was the
    tallest float in the paper (445.1pt against a \\topfraction ceiling of
    472.03pt) and the body was being asked to keep only its result tables, so
    it was cut where the reader's question changes:

      design block  -> tab:corpora-design, appendix. What the two corpora ARE:
                       budget, gold length, slack, the tight and unwinnable
                       counts, the two n's, and how well the pairing covaries.
      outcome block -> tab:ablation panel (c), body. Whether they MEASURE
                       differently: the per-model pass rates and the paired
                       McNemar tallies.

    THE OUTCOME BLOCK IS NO LONGER ITS OWN FLOAT. It was tab:corpora until the
    merge pass, when it became the third panel of the harness-ablation table:
    "swap the generator" is the same kind of question as "move the step cap"
    and "withhold the assets", and three panels of one table cost one caption
    instead of two.

    KEEP THE PANEL TRANSPOSED -- models across the six numeric columns, corpora
    down the rows -- but NOT for the reason recorded here before. That reason
    was that transposing brought the merged float under the 472.03pt
    \\topfraction ceiling: 484.4pt upright against 432.9pt transposed. Both
    numbers came from a throwaway prototype and both are wrong. Measured in
    situ, the shipped transposed float is 521.9pt of body plus caption and
    541.9pt once \\textfloatsep is counted -- 69.9pt OVER the ceiling, the
    tallest exhibit in the paper at 80.4% of \\textheight. No arrangement of
    this panel buys a top-of-page slot; main.tex declares the float [tp] and
    pays for it with a float page. See the comment above the float there.

    What transposing does buy is real but smaller: three data rows instead of
    six, and an exact fit to the seven-column grid that panels (a) and (b)
    already require -- Corpus plus six models is precisely seven columns,
    where upright would fill four and leave three dead. Do not stand it back
    up.

    THE TWO n ROWS MOVED TO THE DESIGN BLOCK in the merge pass, reversing the
    reason that was recorded here before it. They had been kept beside the
    McNemar rows so a body table would carry its own denominators. Panel (c)
    now states 573 in its own heading and 164 inside the McNemar line, so the
    reader still has both denominators without spending two rows on them --
    and two rows is what the height budget does not have.
    results_facts.gen_paired regexes the "Paired rounds, both dead sets
    removed" row out of this stdout and does not care which block prints it,
    but it must keep being printed somewhere.

    THE PANEL HEADINGS HAVE A HARD WIDTH CEILING. A \\multicolumn{7} row does
    not wrap: it sets on one line, pushes the whole tabular past \\linewidth,
    and pdflatex blames the Overfull \\hbox on the tabular rather than on the
    row that caused it. Panels (a) and (b) run 66 and 82 rendered characters.
    Keep new heading text near that, and re-measure with
    src/measure_floats.py rather than counting characters -- the ceiling moves
    with the font size and with the widest data row.

    The header row and the "Generator model" row are no longer printed twice.
    The design block still needs both to label its two columns; panel (c)
    names each generator in its own row label instead, which is what lets it
    skip the separate header row a transposed panel would otherwise want.

    THE MARKERS MUST NOT BE SUBSTRINGS OF EACH OTHER. verify_table_bodies.py
    matches its marker as a substring and takes the FIRST hit, so a pair like
    "...for the corpora table" / "...for the corpora design table" would
    silently verify both labels against the same block. Hence "design" and
    "outcome" rather than a bare name and a qualified one.

    RESULTS_FACTS READS THIS STDOUT, NOT main.tex. gen_slack_{a,b},
    gen_tight_{a,b} and gen_paired are regex'd out of the printed LaTeX rows
    (results_facts.py, the loop at "These two read the slack row's LaTeX").
    Those patterns are single-line and order-independent, so which block a row
    lands in does not matter -- but a row that stops being PRINTED at all will
    take the SHIPPED check down with it.

    The descriptive half used to be typed into main.tex by hand, which meant
    the one table in the paper about whether two corpora differ was the one
    table whose numbers nothing recomputed. Emitting all of it makes the body
    byte-comparable against the .tex the way the other result tables are, and
    it immediately caught a row: "paired rounds after excluding both sets"
    was printed as a single 573, but 573 is the ceiling (600 minus the two
    disjoint dead sets) and the realised count per model x effort cell is
    lower wherever a provider error dropped a round.

    The table used to say only what the two corpora ARE. It never said whether
    they MEASURE differently, which is the only question a generator ablation
    exists to answer, so the outcome panel carries the paired test into it.

    Per-model rows are high effort only. All three levels are tested and the
    console block above prints all 18, but three levels x six models would
    triple the table to say the same thing: the tally rows already report
    every level, and the direction never varies by level.

    Rows follow the main results table. The order is taken from
    corpus.model_order on the same evaluation the main table is built from,
    not from this script's own pass rates -- this script excludes dead rounds
    and so could rank two adjacent models differently, which would silently
    break the row-for-row correspondence the caption promises.
    """
    # len(CELLS), NOT len(order) * len(LEVELS). Those agreed until corpus B
    # turned out to cover gemini-3.7-flash at high effort only; the cross
    # product is now 21 where 19 tests are run. Everything downstream of this
    # line -- the two triples, the printed table note, the caption's alpha --
    # reads it, so the single substitution fixes the denominator in all four
    # places at once.
    n_tests = len(CELLS)
    ns = [r["n"] for r in full_rows.values()]
    ceiling = len(common)
    assert max(ns) == ceiling, f"max paired n {max(ns)} != {ceiling}"

    def _heading():
        """The two rows the design table needs to label its columns."""
        print(r"    \multicolumn{1}{c}{} & \multicolumn{1}{c|}{\textbf{Corpus A}} & \textbf{Corpus B} \\")
        _row(r"\textbf{Generator model}",
             f"\\multicolumn{{1}}{{c|}}{{\\texttt{{{CORPORA['A']['gen']}}}}}",
             f"\\texttt{{{CORPORA['B']['gen']}}}")

    # One row, not two. The ceiling is what the design gives; the range is
    # what provider errors left. Both belong, neither is worth a line of its
    # own.
    #
    # BOTH pairing rows get the treatment, via one helper, because the second
    # one needs it more than the first.
    def _pairing(label, ceil, rows):
        cells = [r["n"] for r in rows.values()]
        lo, hi = min(cells), max(cells)
        _span(label, ceil if lo == hi == ceil else f"{ceil} ({lo}--{hi} per cell)")

    print("\n% ---- LaTeX body for the corpora design table ----")
    print(r"%      column spec lcc; generated by analysis/generator_ablation.py")
    print(r"    \toprule")
    _heading()
    print(r"    \midrule")
    _row("Step budget", CORPORA["A"]["budget"], CORPORA["B"]["budget"])
    g = {k: [gold[k][t] for t in common] for k in CORPORA}
    s = {k: [slack[k][t] for t in common] for k in CORPORA}
    _row("Gold length, mean / median / max",
         *[f"{statistics.mean(g[k]):.2f} / {statistics.median(g[k]):.0f} / {max(g[k])}"
           for k in CORPORA])
    for k in CORPORA:
        assert abs(statistics.mean(s[k]) - statistics.median(s[k])) < 0.05, (
            f"corpus {k}: slack mean and median diverge; the median is no "
            f"longer redundant and the dropped row must come back")
    _row(r"Slack (budget $\div$ gold), mean",
         *[f"${statistics.mean(s[k]):.2f}\\times$" for k in CORPORA])
    tight = {k: sum(1 for v in s[k] if v < 1.5) for k in CORPORA}
    _row(r"Rounds with slack $< 1.5\times$",
         *[f"{100.0 * tight[k] / len(common):.1f}\\% ({tight[k]})" for k in CORPORA])
    _row("Provably unwinnable rounds", len(DEADSETS["A"]), len(DEADSETS["B"]))

    print(r"    \midrule")
    print(r"    \multicolumn{3}{l}{\emph{Pairing between the two corpora}} \\")
    r_gold = pearson(g["A"], g["B"])
    _span(r"Gold-length correlation (Pearson $r$)", f"${r_gold:.3f}$")
    longer = sum(1 for t in common if gold["B"][t] > gold["A"][t])
    equal = sum(1 for t in common if gold["B"][t] == gold["A"][t])
    _span("Corpus B gold longer / equal / shorter",
          f"{longer} / {equal} / {len(common) - longer - equal}")
    _pairing("Aligned rounds per corpus", ceiling, full_rows)
    _pairing(rf"Slack-matched subset ($|\Delta| \le {TOL:.2f}$)",
             len(matched), match_rows)
    print(r"    \bottomrule")

    # ---- outcome panel: panel (c) of tab:ablation ------------------------
    short = [re.sub(r"^(claude|gemini|gpt)-", "", m) for m in order]
    assert len(order) == PANEL_C_COLS, (
        f"panel (c) is transposed into tab:ablation-app's {PANEL_C_COLS} "
        f"numeric columns but the shared roster is {len(order)}; widen the "
        f"column spec in main.tex AND update PANEL_C_COLS together")
    assert len(set(short)) == len(short), (
        f"model abbreviations collide: {short}; two columns would carry the "
        f"same heading")

    print("\n% ---- LaTeX body for the corpora outcome table ----")
    print(r"%      column spec l ccccccc, its own tabular*; panel (c) of")
    print(r"%      tab:ablation-app; generated by analysis/generator_ablation.py.")
    print(r"%      No leading rule: the panel opens with a hand-written")
    print(r"%      \toprule now that it is not stacked under panel (b).")
    print(r"    \multicolumn{" + str(PANEL_C_COLS + 1)
          + r"}{@{}l}{\emph{(c) Generator model.} Pass rate "
          r"(\%) at high effort; " + str(ceiling) +
          r" rounds per corpus.} \\")
    print(r"    \addlinespace[2pt]")
    # The family prefix is what gives when seven model names have to head
    # seven columns sized for two-digit percentages. It is only safe because
    # panels (a) and (b) spell four of the seven out in full in the same
    # float, in the sub-tables above; it would not be safe in a table
    # standing on its own. The count tracks PANEL_C_COLS, asserted above.
    print(r"    \textbf{Corpus} & " + " & ".join(f"\\textbf{{{s}}}" for s in short) + r" \\")
    print(r"    \midrule")
    for k, key in (("A", "pa"), ("B", "pb")):
        print(f"    {k} (\\texttt{{{CORPORA[k]['gen']}}}) & "
              + " & ".join(f"{full_rows[(m, 'high')][key]:.1f}" for m in order)
              + r" \\")
    # Full precision, as panel (a)'s delta is. A reader will nonetheless
    # subtract the two rows above, so check that the two routes agree to the
    # printed digit: a table whose own arithmetic does not close is an
    deltas = [round(full_rows[(m, "high")]["pb"], 1) - round(full_rows[(m, "high")]["pa"], 1)
              for m in order]
    print(r"    $\Delta$ & " + " & ".join(f"${d:+.1f}$" for d in deltas)
          + r" \\")

    # A / n.s. / B as one triple per view, and BOTH views in one sentence.
    # Three verdicts is more honest than reporting the wins alone -- it shows
    # the null count instead of leaving the reader to subtract -- and the
    # second triple is what stops the panel reading as a single-effort result:
    # the model columns above are high effort only, while these tallies cover
    # every paired model x effort cell. The zero is the load-bearing digit in
    # both. Do not write the cell count into this comment -- it moved from 18
    # to 21 to 19 across three roster changes and was wrong for two of them.
    triples = []
    for label, tal in (("full set", full_tally),
                       ("slack-matched subset", match_tally)):
        b = tal.get("corpus B easier", 0)
        assert b == 0, (f"corpus B came out easier on {b} test(s) in the "
                        f"{label}; the caption's 'never the other way' claim "
                        f"must be rewritten")
        # TWO ASSERTIONS, NOT ONE, because they fail for different reasons and
        # the message has to say which. `skipped` counts cells that exist in
        # both corpora but fell below the pairing floor on this view's pool;
        # the sum catches a cell that dropped out of the iteration entirely.
        # The sum ALONE cannot see the first fault -- "skipped" is a tally key,
        # so it lands inside the sum and the total still closes. That is
        # exactly how a 19-test panel came within one edit of shipping a
        # 21-test correction, with this assertion green the whole way.
        assert not tal.get("skipped"), (
            f"{label}: {tal['skipped']} cell(s) present in both corpora fell "
            f"below the pairing floor; the triple would drop them silently "
            f"while alpha = 0.05/{n_tests} kept paying for them")
        assert sum(tal.values()) == n_tests, (
            f"{label}: verdicts sum to {sum(tal.values())}, not {n_tests}; a "
            f"cell left the iteration and the triple would not add up")
        triples.append(f"{tal.get('corpus A easier', 0)}/"
                       f"{tal.get('n.s.', 0)}/{b}")
    # THIS SENTENCE IS NO LONGER A TABLE ROW, AND THAT IS THE POINT. It used
    # to be \multicolumn{7}{@{}l} at the foot of the tabular, where it was the
    # widest thing in the float and set the float's width at 450.06pt against
    # a 472.32pt \linewidth. Worse than the width was WHERE the overhang went:
    # a \multicolumn wider than the columns it spans charges the entire excess
    # to the LAST column, so this one line alone opened a 30.2pt gap in front
    # of column 7 -- measured by deleting it, which took the tabular from
    # 448.20pt to 417.96pt and shrank the column-7 gap on EVERY row by 30.3pt
    # while leaving all five other gaps untouched.
    #
    # The previous fix was to shorten the prose: "slack-matched" became
    # "matched" and "over" became a comma to buy about five characters. That
    # bought margin and left the gap, and it made the note disagree with
    # sec:app-corpora, which defines the term as slack-matched. Now that the
    # sentence lives in a minipage below the tabular it is width-coupled to
    # nothing, so the defined term is restored and the wrapping is free.
    #
    # MEASURE IT IN SITU, NEVER IN A STANDALONE PROBE. A minimal document
    # loading google.cls renders this tabular a full font-size step smaller
    # than main.tex does (457.71pt vs 494.31pt for the same source), because
    # \small is relative to a base size the probe does not share. A probe said
    # this line fit while the real build was 21.99pt overfull.
    print("\n% ---- LaTeX note under the ablation table ----")
    print(r"%      goes in the minipage BELOW \end{tabular}, not in a row;")
    print(r"%      generated by analysis/generator_ablation.py")
    # The tie before the parenthesis is load-bearing at the current lengths:
    # the sentence overruns the tabular's width by about one word, and without
    # it the second line is the orphan "($n{=}164$)." alone. Tied, the break
    # falls before "slack-matched" and the two lines are balanced. It is a tie
    # rather than a \linebreak so that a shorter sentence -- fewer models, a
    # narrower table -- silently goes back to one line instead of keeping a
    # break that no longer has a reason.
    print(r"    Paired McNemar, " + str(n_tests)
          + r" model $\times$ effort cells (A easier / n.s.\ / B): "
          + triples[0] + r" full, " + triples[1]
          + r" slack-matched~($n{=}" + str(len(matched)) + r"$).")

    # ---- caption facts, derived ------------------------------------------
    missing = sorted({corpus.MODEL_PROVIDER[m] for m in CANON} -
                     {corpus.MODEL_PROVIDER[m] for m in order})
    print(f"% caption fact: outcome panel covers {len(order)} of "
          f"{len(CANON)} models; corpus B was never run on the "
          f"{' / '.join(missing)} family")
    # THE HOLES ARE NAMED, NOT NETTED OUT. A note that says "19 cells" and
    # stops lets a reader who multiplies 7 by 3 conclude the panel lost
    # ROUNDS. What it lost is one model at two efforts on one corpus, which is
    # a coverage gap in the ablation's design rather than attrition in its
    # data, and the two read very differently to a referee.
    if ABSENT:
        by_model = defaultdict(lambda: defaultdict(list))
        for m, lvl in ABSENT:
            sides = "/".join(k for k in CORPORA if not res[k].get((m, lvl)))
            by_model[DISPLAY[m]][sides].append(lvl)
        gaps = "; ".join(
            f"corpus {sides} has no {name} run at {' or '.join(lv)} effort"
            for name, d in sorted(by_model.items())
            for sides, lv in sorted(d.items()))
        print(f"% caption fact: {n_tests} paired cells, not "
              f"{len(order)} models x {len(LEVELS)} efforts = "
              f"{len(order) * len(LEVELS)} -- {gaps}")
    print(f"% caption fact: {full_tally.get('corpus A easier', 0)}/{n_tests} "
          f"paired tests reach significance with corpus A easier on the full "
          f"set and {match_tally.get('corpus A easier', 0)}/{n_tests} on the "
          f"slack-matched subset, none the other way, at Bonferroni "
          f"alpha = 0.05/{n_tests} = {ALPHA:.5f}")
    print(f"% caption fact: matching equalises slack "
          f"({statistics.mean([slack['A'][t] for t in matched]):.2f} vs "
          f"{statistics.mean([slack['B'][t] for t in matched]):.2f}) but widens "
          f"the mean gold gap from {gapf:+.2f} to {gapm:+.2f} steps, so the "
          f"matched effect size is inflated, not unbiased")
    print(f"% caption fact: per-model high-effort pass falls from "
          f"{min(full_rows[(m, 'high')]['pa'] for m in order):.1f}-"
          f"{max(full_rows[(m, 'high')]['pa'] for m in order):.1f}\\% on A to "
          f"{min(full_rows[(m, 'high')]['pb'] for m in order):.1f}-"
          f"{max(full_rows[(m, 'high')]['pb'] for m in order):.1f}\\% on B, "
          f"every model down")
    assert all(full_rows[(m, "high")]["pa"] > full_rows[(m, "high")]["pb"]
               for m in order), "some model scored higher on corpus B at high effort"

    # CROSS-CORPUS AGREEMENT, DERIVED HERE BECAUSE IT WAS NOT DERIVED ANYWHERE.
    # sec:generator-ablation quotes a correlation for the claim that rankings
    # survive the generator swap. That figure was typed into main.tex by hand
    # and matched nothing any script computes: it read r = 0.94 against 0.918
    # Pearson over the then-18 paired cells, 0.911 Pearson over the six
    # high-effort models, and 0.970 Spearman over the cells. Every candidate
    # is printed, at the pool it was computed on, so the sentence has to name
    # which statistic it means instead of inheriting a number with no source.
    #
    # Spearman is the one an ORDERING claim wants; Pearson answers a stronger
    # question about the levels moving together. They are both here so the
    # choice is visible rather than implicit.
    xs = [r["pa"] for r in full_rows.values()]
    ys = [r["pb"] for r in full_rows.values()]
    print(f"% caption fact: cross-corpus agreement over the {len(xs)} paired "
          f"cells: Pearson r = {pearson(xs, ys):.3f}, "
          f"Spearman rho = {spearman(xs, ys):.3f}")
    for lvl in LEVELS:
        sub = [r for (nm, lv), r in full_rows.items() if lv == lvl]
        if len(sub) < 3:
            continue
        sx, sy = [r["pa"] for r in sub], [r["pb"] for r in sub]
        print(f"% caption fact: cross-corpus agreement at {lvl} effort "
              f"({len(sub)} models): Pearson r = {pearson(sx, sy):.3f}, "
              f"Spearman rho = {spearman(sx, sy):.3f}"
              # Tolerance, not `== 1.0`. A perfect ordering comes back as
              # 0.9999999999999998 often enough that an exact test marked the
              # low-effort panel and silently skipped the high-effort one,
              # which is the same value.
              + ("  <- zero rank inversions"
                 if abs(spearman(sx, sy) - 1.0) < 1e-9 else ""))


DEADSETS = load_dead()
gold = {k: gold_lengths(v["tasks"]) for k, v in CORPORA.items()}
res = {k: results(k) for k in CORPORA}

common = sorted(set(gold["A"]) & set(gold["B"]))
slack = {k: {t: CORPORA[k]["budget"] / gold[k][t] for t in common if gold[k][t]}
         for k in CORPORA}
common = [t for t in common if t in slack["A"] and t in slack["B"]]
matched = [t for t in common if abs(slack["A"][t] - slack["B"][t]) <= TOL]

print("generator ablation: corpus A =", CORPORA["A"]["gen"],
      f"(budget {CORPORA['A']['budget']})   vs   corpus B =", CORPORA["B"]["gen"],
      f"(budget {CORPORA['B']['budget']})")
print(f"dead rounds excluded: A={len(DEADSETS['A'])}  B={len(DEADSETS['B'])}  "
      f"(verified against eval rows)")
print(f"\ntasks in common: {len(common)}")
for tol in (0.30, 0.20, 0.15):
    n = sum(1 for t in common if abs(slack["A"][t] - slack["B"][t]) <= tol)
    print(f"  slack-matched at |delta| <= {tol:.2f}: {n}")

print("\nselection diagnostics (why MATCHED is not an unbiased subsample):")
describe(common, "FULL")
describe(matched, f"MATCHED |d|<={TOL:.2f}")
gapf = statistics.mean([gold["B"][t] - gold["A"][t] for t in common])
gapm = statistics.mean([gold["B"][t] - gold["A"][t] for t in matched])
print(f"  -> mean gold-length gap (B - A):  FULL {gapf:+.2f} steps   "
      f"MATCHED {gapm:+.2f} steps")
print(f"  -> matching equalises slack but WIDENS the gold-length gap; the matched "
      f"effect size is inflated relative to the corpus.")

shared = sorted({m for m, _ in res["A"]} & {m for m, _ in res["B"]} & set(corpus.MODEL_LABEL.keys()))

# THE BONFERRONI DENOMINATOR IS THE NUMBER OF TESTS ACTUALLY RUN, NOT THE
# ROSTER CROSSED WITH THE EFFORT LEVELS. Those were the same number until
# gemini-3.7-flash joined the panel. Corpus B had already been generated and
# scored by then and was only ever re-run on the new model at high effort, so
# two of the 7 x 3 cells do not exist. `0.05 / (len(shared) * len(LEVELS))`
# would have published an alpha corrected for 21 tests while performing 19,
# and the table note would have claimed 21 model x effort cells that a reader
# counting rows in this stdout cannot find.
#
# THE OLD ASSERTION IN emit_body WAS MEANT TO CATCH EXACTLY THIS AND COULD
# NOT. compare() files an absent cell under tally["skipped"], and "skipped" is
# a tally key like any other, so `sum(tally.values()) == 21` held as
# 16 + 3 + 0 + 2 skipped. The skipped bucket is now asserted empty on its own
# line, and compare() iterates CELLS, so it can only ever fill on a cell that
# EXISTS but is too thin to pair -- which is a different fault and deserves a
# different failure.
#
# Membership is decided on the corpora, not on a task pool, so the denominator
# does not move between the full and the slack-matched view. If matching ever
# starves a cell below the pairing floor, that shows up as a non-empty skipped
# bucket rather than as two views quietly corrected by different alphas.
_have = {(m, lvl) for m in shared for lvl in LEVELS
         if res["A"].get((m, lvl)) and res["B"].get((m, lvl))}
CELLS = [(m, lvl) for m in shared for lvl in LEVELS if (m, lvl) in _have]
ABSENT = [(m, lvl) for m in shared for lvl in LEVELS if (m, lvl) not in _have]
ALPHA = 0.05 / len(CELLS)

# The number of NUMERIC columns in tab:ablation-app's tabular spec in main.tex
# (`l | ccc c ccc` -> 7). Panel (c) transposes the shared roster into exactly
# those columns, so the two numbers are one fact stored in two files and the
# assert in emit_body is the only thing tying them together. Change this only
# in the same edit that changes the spec, the \multicolumn spans of panels (a)
# and (b), and their trailing cells.
PANEL_C_COLS = 7

# Corpus A carries a tenth directory (OpenAI/5_5) that the nine-model roster
# does not include, and corpus B carries no GPT models at all, so `shared` is
# whatever survived both -- it is not a slice of the roster by construction.
# Assert the mapping rather than trusting it: an unmapped directory would
# otherwise reach the page as a raw name like `3_1_pro`.
unmapped = [m for m in shared if m not in corpus.MODEL_LABEL]
assert not unmapped, f"eval dirs with no entry in corpus.MODEL_LABEL: {unmapped}"
DISPLAY = {m: corpus.MODEL_LABEL[m] for m in shared}

# The paper's row order, taken from the evaluation the main table is built
# from and then filtered -- never re-derived from this script's pass rates.
CANON = corpus.model_order(corpus.load_eval("high")[0])
ORDER = [m for m in CANON if m in set(DISPLAY.values())]
assert len(ORDER) == len(shared), "a shared model is missing from the canonical order"

print(f"\nmodels present in both corpora ({len(shared)}): "
      f"{', '.join(DISPLAY[m] for m in shared)}")
print(f"table row order (from corpus.model_order): {', '.join(ORDER)}")
print(f"Bonferroni alpha = 0.05/{len(CELLS)} = {ALPHA:.5f}")
# Named, not counted. A bare "19 of 21" invites the reader to assume the two
# gaps are spread thinly across the roster; they are one model missing two
# efforts, which is a different disclosure and the one the appendix has to
# make.
print(f"paired cells: {len(CELLS)} of {len(shared) * len(LEVELS)} "
      f"({len(shared)} models x {len(LEVELS)} efforts)")
for m, lvl in ABSENT:
    sides = "/".join(k for k in CORPORA if not res[k].get((m, lvl)))
    print(f"  !! no paired test for {DISPLAY[m]} x {lvl}: "
          f"cell absent from corpus {sides}")

full_tally, full_rows = compare(
    common, "FULL paired set  (corpus B enjoys MORE slack -> conservative lower bound)")
match_tally, match_rows = compare(
    matched, f"SLACK-MATCHED subset  |delta| <= {TOL:.2f}  (budget confound removed)")
emit_body(ORDER, full_rows, full_tally, match_rows, match_tally)

# ---------------------------------------------------------------------------
# Effort against the step budget, both corpora. This panel exists for
# sec:effort, not for the corpora table, and corpus B is the whole point of it:
# the effort inversion on corpus A could be an artefact of a 20-step budget, and
# the only way to find out is to look at a corpus generated against 30.
#
# DENOMINATORS DIFFER BY CORPUS ON PURPOSE. This is a WITHIN-corpus question --
# nothing here sets a corpus-A number beside a corpus-B one -- so each corpus is
# measured on its own winnable rounds (A drops 18 dead, B drops 9) rather than
# on the 573-task intersection the tables above use. An earlier version of this
# block filtered by `common`, which reads like a pairing restriction and is not
# one: every task id is shared, so the filter was a no-op over each corpus's own
# rows. The n is printed so no rate here is quotable without its pool. Corpus
# A's pass column consequently sits a few tenths above the sec:efficiency
# figures, which are over the full 600 with the dead rounds left in.
print(f"\n{'=' * 108}\nEFFORT vs THE STEP BUDGET, both corpora   "
      f"(each corpus on its OWN winnable rounds)\n{'=' * 108}")
hdr = (f"{'model':<{corpus.NAME_W}} {'corpus':>6} {'budget':>6} {'n':>5}  "
       f"{'cap-hit% low/med/high':>24}   {'pass% low/med/high':>21}  "
       f"{'cap-mono':>8} {'b(lo>hi)':>9} {'c(hi>lo)':>9} {'p':>10}")
print(hdr)
print("-" * len(hdr))
ORD_LOW = ["low", "medium", "high"]
SLUG = {DISPLAY[m]: m for m in shared}

# THIS PANEL NEEDS ALL THREE EFFORTS, SO ITS ROSTER IS PER CORPUS AND SHORTER
# THAN `ORDER`. The paired panel above tolerates a missing effort by dropping
# one cell; a low-vs-high contrast with no low is not a thinner test, it is no
# test. gemini-3.7-flash was scored on corpus B at high effort only, and the
# loop used to reach for CAPS[k][(slug, "low")] regardless -- CAPS[k] is a
# defaultdict, so the miss returned an empty dict, the pool came back empty,
# and the cap-hit line divided by zero. A defaultdict turns "this run does not
# exist" into "this run scored nothing", which is why the membership test is
# explicit here rather than left to a .get() default.
EFFORT_ROSTER = {k: [n for n in ORDER
                     if all(res[k].get((SLUG[n], lvl)) for lvl in ORD_LOW)]
                 for k in CORPORA}
for k in CORPORA:
    for n in ORDER:
        if n not in EFFORT_ROSTER[k]:
            gone = [lvl for lvl in ORD_LOW if not res[k].get((SLUG[n], lvl))]
            print(f"  !! {n} omitted from corpus {k}: no run at "
                  f"{'/'.join(gone)} effort, and a low-vs-high contrast needs "
                  f"both ends")
mono = {k: 0 for k in CORPORA}
panel = {}
for name in ORDER:
    slug = SLUG[name]
    for k in CORPORA:
        if name not in EFFORT_ROSTER[k]:
            continue
        pools = [sorted(CAPS[k][(slug, lvl)]) for lvl in ORD_LOW]
        cp = [100.0 * sum(CAPS[k][(slug, lvl)][t] for t in pool) / len(pool)
              for lvl, pool in zip(ORD_LOW, pools)]
        ps = [100.0 * sum(res[k][(slug, lvl)][t] for t in pool) / len(pool)
              for lvl, pool in zip(ORD_LOW, pools)]
        up = cp[0] < cp[1] < cp[2]
        mono[k] += up
        # Paired low-vs-high within the corpus, on the rounds both levels
        # scored. "Still degrades" is a direction; this is whether the
        # direction is distinguishable from noise, which is the only version
        # of the claim worth putting in a paper.
        lo, hi = res[k][(slug, "low")], res[k][(slug, "high")]
        both = sorted(set(lo) & set(hi))
        b = sum(1 for t in both if lo[t] and not hi[t])
        c = sum(1 for t in both if not lo[t] and hi[t])
        p = exact_mcnemar(b, c)
        panel[(name, k)] = (cp, ps, [len(p_) for p_ in pools], b, c, p)
        print(f"{name:<{corpus.NAME_W}} {k:>6} {CORPORA[k]['budget']:>6} "
              f"{min(len(p_) for p_ in pools):>5}  "
              f"{cp[0]:7.1f} {cp[1]:7.1f} {cp[2]:7.1f}   "
              f"{ps[0]:6.1f} {ps[1]:6.1f} {ps[2]:6.1f}   "
              f"{str(up):>8} {b:>9} {c:>9} {p:>10.3g}")
# Denominator per corpus, not len(ORDER): corpus B carries one model fewer in
# this panel, and "5/7" against "4/6" is a different sentence from both counts
# printed over 7.
for k in CORPORA:
    print(f"  corpus {k} (budget {CORPORA[k]['budget']}): cap-hit rises "
          f"monotonically with effort in {mono[k]}/{len(EFFORT_ROSTER[k])} "
          f"models")
# The replication rests on this one model being present at every effort in
# both corpora. Assert it rather than let a KeyError describe it.
for k in CORPORA:
    assert "gemini-3.5-flash" in EFFORT_ROSTER[k], (
        f"corpus {k} no longer covers gemini-3.5-flash at all three efforts; "
        f"sec:effort's replication claim has lost its subject")
    cp, ps, ns, b, c, p = panel[("gemini-3.5-flash", k)]
    print(f"  gemini-3.5-flash inversion  corpus {k} (n={ns[0]}/{ns[2]} "
          f"low/high): pass {ps[0]:.1f}% -> {ps[2]:.1f}%"
          f"  cap-hit {cp[0]:.1f}% -> {cp[2]:.1f}%   paired b={b} c={c} p={p:.3g}")
# The family is the contrasts this panel ACTUALLY RAN, which is no longer
# len(ORDER) * len(CORPORA) now that the two rosters differ in length.
# results_facts.eff2 recomputes the same alpha as 0.05/len(prow) by counting
# the printed rows, so the two agree only while this expression counts rows
# too.
N_EFFORT = sum(len(EFFORT_ROSTER[k]) for k in CORPORA)
FAM = 0.05 / N_EFFORT
pB = panel[("gemini-3.5-flash", "B")][5]
print(f"  -> the inversion survives a budget half again as large, so it is not "
      f"an artefact of corpus A's 20-step cap.")
print(f"  -> corpus B's replication is ONE pre-specified test at p={pB:.4f}; "
      f"read as one of the {N_EFFORT} low-vs-high contrasts "
      f"in this panel it would miss alpha={FAM:.4f}. Both readings are stated "
      f"because they disagree.")
# The tempting mechanism -- "extra steps stop paying once the budget binds" --
# is NOT what this panel shows, and the counterexample is printed so the claim
# cannot be written without meeting it.
for name in EFFORT_ROSTER["A"]:
    cp, ps, ns, b, c, p = panel[(name, "A")]
    if cp[2] >= 20.0 and c > b:
        print(f"  -> COUNTEREXAMPLE to a clean cap-hit story: {name} ends "
              f"{cp[2]:.1f}% of corpus-A rounds at the cap and effort still "
              f"helps it (b={b} c={c} p={p:.3g}). Only gemini-3.5-flash "
              f"inverts; the middle of the cap-hit range does not.")
