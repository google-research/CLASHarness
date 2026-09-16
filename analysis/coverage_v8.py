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

"""T9 -- what the corpus actually covers. Feeds Section 4's coverage table.

Diversity is the first thing a benchmark paper is asked to substantiate and the
easiest thing to overstate, so this script is built to report the corpus's
coverage AND its degeneracies from the same pass, with nothing transcribed by
hand.

WHAT THIS TABLE IS NOT. It is not a difficulty-axis analysis. The report's
Section 8 already is one, it is reproduced verbatim in the appendix, and its
verdict is not restated or re-derived here: of the fourteen CANDIDATE fields
that scan_difficulty_axes.py probes, ten are constant across all 600 specs and
scored-unit count is the only clean axis. Those fourteen are slots in the
generic task-spec schema (`weight`, `template_id`, `availability_tier`,
`online_requirement`, ...) that the report tried as difficulty proxies and
ruled out -- they were never axes this generator claims to vary, so their
constancy is a fact about the schema, not a coverage gap.

What Section 4 needs is the complementary thing: the axes the generator DOES
vary, which are the ones in design_axes.json -- seeds, conflict categories,
conflict plugins, challenge plugins, vision, async -- plus the round-level
descriptors (round type, tool families) and the two continuous size axes. All
of those do vary; that is the table below.

The constant-field census is still emitted, but as a DRIFT GUARD on the
appendix, not as a finding: KNOWN_CONSTANT is the appendix table's verdict
frozen into an assertion, so if a regenerated corpus starts populating
`hidden_constraints`, this fails loudly instead of leaving the appendix
asserting something the data no longer supports. Same pattern as
make_body_figures.taxonomy(): the factory's artifact is an input to be checked,
not an authority to be trusted.

Two counts in the table are deliberately NOT mutually exclusive and are marked
as such: conflict categories (a seed carries every category its plugins route
to, 20 seeds -> 32 slots) and plugins (a seed carries several).

Prints a console view and a LaTeX body from one dict, so they cannot drift.
"""
import json
import sys
from collections import Counter, defaultdict

import corpus

# The appendix's Section 8 table, frozen: the candidate difficulty fields it
# rules out as constant on all 600 specs. Named rather than discovered so that
# a field which STARTS varying trips the assertion below instead of quietly
# vanishing from the degeneracy list while the appendix still claims it.
KNOWN_CONSTANT = {
    "availability_tier", "conflict_taxonomy", "decision_requirements",
    "difficulty", "domains", "hidden_constraints", "online_requirement",
    "template_id", "visible_constraints", "weight",
}


def load_difficulty():
    d = json.load(open(corpus.DIFFICULTY_AXES, encoding="utf-8"))
    tasks = d["tasks"]
    assert len(tasks) == corpus.N_RAW, (len(tasks), corpus.N_RAW)
    ids = {t["task_id"] for t in tasks}
    eps, _ = corpus.load_eval(corpus.PRIMARY_EFFORT)
    for m, rows in eps.items():
        assert set(rows) == ids, (
            f"difficulty_axes.json describes different tasks than the {m} "
            f"episodes ({len(ids ^ set(rows))} symmetric difference)")
    return d, tasks


def constant_fields(card):
    """Declared spec fields with no variation, recomputed and cross-checked."""
    const, varying = {}, {}
    for name, c in sorted(card.items()):
        n = c["n_distinct"]
        (const if n <= 1 else varying)[name] = c
    assert set(const) == KNOWN_CONSTANT, (
        "the set of degenerate spec fields moved.\n"
        f"  now constant but not listed: {sorted(set(const) - KNOWN_CONSTANT)}\n"
        f"  listed but now varying:      {sorted(KNOWN_CONSTANT - set(const))}\n"
        "Update KNOWN_CONSTANT *and* the sentence in Section 4 that quotes it.")
    return const, varying


def quart(xs):
    s = sorted(xs)
    return s[0], s[len(s) // 2], s[-1]


def main():
    doc, tasks = load_difficulty()
    amap = corpus.load_axis_map()
    seeds = amap["seeds"]
    design = json.load(open(corpus.DESIGN_AXES, encoding="utf-8"))
    axes = next(iter(design["by_model"].values()))["axes"]

    const, varying = constant_fields(doc["axis_cardinality"])

    cat_seeds = defaultdict(int)
    for v in seeds.values():
        for c in v["categories"]:
            cat_seeds[c] += 1
    slots = sum(cat_seeds.values())

    plugins = sorted({p for v in seeds.values() for p in v["plugins"]})
    challenges = sorted({c for v in seeds.values() for c in v["challenges"]})
    assert len(plugins) == len(axes["plugin"]), (len(plugins), len(axes["plugin"]))
    assert len(challenges) == len(axes["challenge"]), len(challenges)
    # The challenge plugins are a SUBSET of the plugin union, not a disjoint
    # second family: plugin_category == "challenge" for exactly these 7. The
    # table used to list "Conflict plugins 23" and "Challenge plugins 7" on
    # consecutive rows, which reads as 30. Assert the containment here so the
    # conflict count below is derived, never typed.
    assert set(challenges) <= set(plugins), sorted(set(challenges) - set(plugins))
    n_conflict = len(plugins) - len(challenges)
    # The effect-declaring ("mechanic") plugins are all conflict plugins; none
    # of them is a challenge plugin, so "13 of the 16" is exact.
    assert not (set(design["mechanic_plugins"]) & set(challenges))
    assert set(design["mechanic_plugins"]) <= set(plugins)

    n_vision = sum(1 for v in seeds.values() if v["vision"])
    n_async = sum(1 for v in seeds.values() if v["async"])
    assert axes["vision"]["vision_graded"]["seeds"] == n_vision
    assert axes["async"]["async_required"]["seeds"] == n_async

    rt = Counter(t["round_type"] for t in tasks)
    tf = Counter(t["tool_extra"] for t in tasks)
    ni = sum(1 for t in tasks if t["natural_image"])
    gl, un = quart([t["gold_len"] for t in tasks]), quart([t["units"] for t in tasks])
    variants = len({t["scen"] for t in tasks})

    # rows: (axis, levels, spread, exclusive?)
    rows = [
        ("Scenario seeds", len(seeds),
         f"{variants} variants, {corpus.N_RAW} rounds", True),
        ("Conflict categories", len(corpus.CAT_ORDER),
         "/".join(str(cat_seeds[c]) for c in corpus.CAT_ORDER) + " seeds", False),
        # "mechanic" is the factory's own term (design_axes.mechanic_plugins).
        # It is no longer a guess what it means: section3_facts asserts this
        # set is exactly the plugins with a non-empty emits_effects, which is
        # what sec:extensions calls "declare effects" when it points here --
        # so the Spread column says "with effects" and the guard is still
        # keyed to the factory's term, one indirection away.
        # The Levels column is now the union (23) on the first row and the
        # challenge subset (7) on the second, with the subset spelled out, so
        # the two rows can no longer be added together.
        ("Plugins", len(plugins),
         f"{n_conflict} conflict, "
         f"{len(design['mechanic_plugins'])} with effects", False),
        ("Challenge plugins", len(challenges),
         f"of the {len(plugins)}; >=1 per seed", False),
        ("Round type", len(rt),
         "/".join(str(n) for _, n in rt.most_common()) + " rounds", True),
        ("Tool families", len(tf), f"{min(tf.values())}--{max(tf.values())} rounds each", True),
        ("Vision-graded seeds", 2, f"{n_vision} of {len(seeds)}", True),
        ("Natural photograph", 2, f"{ni} of {corpus.N_RAW} rounds", True),
        ("Async required", 2, f"{n_async} of {len(seeds)}", True),
        ("Gold trajectory length", "cont.", f"{gl[0]}--{gl[2]}, median {gl[1]}", None),
        ("Scored units / round", "cont.", f"{un[0]}--{un[2]}, median {un[1]}", None),
    ]

    print(f"T9  corpus coverage -- {corpus.N_RAW} rounds, run {corpus.RUN_V8}, "
          f"{corpus.BUDGET_V8}-step budget\n")
    print(f"{'axis':<24} {'levels':>7}  {'spread':<34} excl?")
    print("-" * 78)
    for name, lv, spread, excl in rows:
        mark = "-" if excl is None else ("yes" if excl else "NO -- overlaps")
        print(f"{name:<24} {str(lv):>7}  {spread:<34} {mark}")
    print(f"\n  category slots: {len(seeds)} seeds -> {slots} category memberships; "
          f"per-category round counts sum to {slots * corpus.N_RAW // len(seeds)}, "
          f"not {corpus.N_RAW}")

    print(f"\ndrift guard on appendix S8 (candidate DIFFICULTY fields, not "
          f"coverage axes):\n  {len(const) + len(varying)} probed, "
          f"{len(varying)} vary, {len(const)} constant on all {corpus.N_RAW} "
          f"specs -- matches the appendix table")
    print("  constant (appendix verdict: not usable as a difficulty axis):")
    for name in sorted(const):
        v = const[name]["values"]
        val = v[0] if v else "<absent>"
        print(f"    {name:<24} = {val}")
    print("  varying:")
    for name in sorted(varying):
        print(f"    {name:<24} {varying[name]['n_distinct']} distinct")

    emit_body(rows, const, varying, slots, len(seeds))
    return 0


def _tt(names, lead, width=84):
    r"""Pack field names into \texttt{} runs of at most `width` VISIBLE chars.

    Visible, not source: `\texttt{...}` and the `\_` escapes add nine-odd
    characters a reader never sees, and measuring the source wraps the list
    about a third too early. `lead` is the visible width of the label the
    first line has to share space with.

    Wrapped here rather than in the .tex because the census is derived -- a
    field that stops being constant moves between the two lists and the line
    breaks must move with it. 84 is calibrated for \small typewriter at this
    linewidth; the build's overfull count is the check that matters.
    """
    out, cur, used = [], [], lead
    for i, n in enumerate(sorted(names)):
        sep = "" if i == len(names) - 1 else ","
        w = len(n) + len(sep) + 1
        if cur and used + w > width:
            out.append(" ".join(cur))
            cur, used = [], 0
        cur.append("\\texttt{" + n.replace("_", r"\_") + "}" + sep)
        used += w
    if cur:
        out.append(" ".join(cur))
    return out


def emit_body(rows, const, varying, slots, n_seeds):
    r"""The whole tabular, toprule to bottomrule, for tab:coverage.

    Two things changed the day this stopped being a paste of bare rows.

    First, overlap is a COLUMN now, not a dagger. Three of the nine axes do
    not partition the corpus, and a footnote glyph asks the reader to hold
    that fact in their head while reading a count that looks summable. A
    column states it on the row it applies to and gives the caption its two
    sentences back.

    Second, the declared-field census comes into the table. It used to live in
    the caption as prose, and the prose was wrong: it said difficulty "and
    seven further declared fields" were constant when the guard above counts
    ten. That is the failure mode of a hand-typed caption over a generated
    body, so the census is emitted from the same dict the drift guard asserts
    on, and the caption quotes nothing the table does not print.

    The spread strings also get their LaTeX here. The old emitter printed a
    literal ">=1", which someone repaired to $\ge$1 in main.tex, and from that
    moment the body could not be diffed against its generator.
    """
    print("\n% ---- LaTeX body for T9 (paste into the tabular) ----")
    print(r"%      column spec lclc; generated by analysis/coverage_v8.py")
    print(r"    \toprule")
    print(r"    \textbf{Axis} & \multicolumn{1}{c|}{\textbf{Levels}} & \textbf{Spread} & \multicolumn{1}{c}{\textbf{Disjoint}} \\")
    prev = object()
    for name, lv, spread, excl in rows:
        # A rule wherever the kind of axis changes: categorical (yes), the
        # three overlapping ones, then the two continuous. Driven off the
        # exclusivity flag so a new row lands in the right block by itself.
        block = excl is None
        if block != prev:
            print(r"    \midrule")
            prev = block
        mark = "---" if excl is None else ("yes" if excl else "no")
        print(f"    {name} & {lv} & {spread.replace('>=', r'$\ge$')} & {mark} \\\\")
    n_probed = len(const) + len(varying)
    # The field NAMES used to be printed here as four further rows of
    # monospace, wrapped by _tt. They are gone on purpose. The census in the
    # appendix (tab:rep-12) carries the same list plus each field's
    # cardinality and a verdict on whether it is usable as an axis, so the
    # rows here were the strictly poorer copy; and the four that DO vary are
    # descriptors this table already devotes its own rows to, which made the
    # list a restatement of the table from inside the table.
    #
    # What must not follow the names out is the COUNT. The caption is only
    # allowed to say "4 of 14" because the float prints "4 of 14" -- that is
    # the drift guard this table was rebuilt around after the caption twice
    # claimed a number the body did not show. The census sentence moved out of
    # the tabular and into a minipage below it (see the note block just
    # below), which keeps it generated, keeps it verified and keeps it on the
    # page; what it must never become is a hand-typed caption clause.
    print(r"    \bottomrule")

    # OUT OF THE TABULAR ON PURPOSE. As \multicolumn{4}{l} this sentence was
    # by far the widest thing in a four-column table whose other cells are
    # short, and LaTeX charges a \multicolumn's overhang entirely to the last
    # column it spans: the Disjoint column, all of 8pt of content, was being
    # held open to a gap of up to 130pt -- 17.5x the table's median column
    # gap, the worst offender of the eight hand-written tables in the paper.
    # It also had to be kept to one line by hand, since a second line would
    # have gone overfull. In a minipage it wraps freely and costs the columns
    # nothing.
    print("\n% ---- LaTeX note under the coverage table ----")
    print(r"%      goes in the minipage BELOW \end{tabular}, not in a row;")
    print(r"%      generated by analysis/coverage_v8.py")
    # Roman and full-stopped, where the row version was italic and had neither.
    # As a \multicolumn row the italics were what marked it as not-a-data-row;
    # below the rule at \footnotesize that job is already done, a whole
    # sentence in italics is heavier than a note wants to be, and it would not
    # have matched the McNemar note under tab:ablation, which is the only
    # other note in the paper.
    # \ref, not the literal 8: appendix G used to set its own headings as
    # "\S8. Difficulty landscape", so "appendix \S8" matched what was on the
    # page. It is numbered off the counter now and reads G.8, and a bare \S8
    # points at nothing in a paper whose sections are 1-6 and A-G.
    print(r"    Declared spec fields, appendix~\S\ref{rep:s8}: %d of %d "
          r"probed vary; "
          r"the full census is Table~\ref{tab:rep-12}."
          % (len(varying), n_probed))

    # ---- caption facts, derived ------------------------------------------
    overlap = [r[0] for r in rows if r[3] is False]
    print(f"% caption fact: {len(overlap)} axes do not partition the corpus "
          f"({', '.join(overlap)}); {n_seeds} seeds fill {slots} category slots")
    dv = const.get("difficulty", {}).get("values") or ["?"]
    print(f"% caption fact: the specs' own difficulty field is the constant "
          f"{dv[0]} on all {corpus.N_RAW} rounds and {len(const) - 1} further "
          f"declared fields are likewise constant -- NOT seven, which is what "
          f"the caption said before this was generated")
    print(f"% caption fact: the {len(varying)} that do vary are "
          f"{', '.join(sorted(varying))} -- descriptors the table already "
          f"covers as its own rows, so nothing is lost by not treating the "
          f"declared schema as a difficulty axis")


if __name__ == "__main__":
    sys.exit(main())
