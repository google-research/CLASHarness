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

"""Every number Sections 4-7 assert in PROSE, checked against its generator.

The body tables are already protected: src/verify_table_bodies.py diffs each
tabular against the script that emitted it, so a table cannot drift. Narrative
sentences have had no such guard, and every numeric error caught in this draft
so far has been in a narrative sentence rather than a table. This script is the
missing half.

It is also the paper's single verification entry point. Two other guards have
their own SHIPPED dicts and their own definitions of "recompute", so they are
not merged into this file -- they are CALLED from it, and only their verdicts
land in SHIPPED here:

  analysis/section3_facts.py  Section 3's corpus counts, recomputed from the
      task files.                                    -> s3_* at the bottom
  analysis/figure_facts.py    Figure 2, which is hand-drawn SVG with no
      generator, re-derived from the shipped file.   -> fig2_* at the bottom

Both had a working --check that no release step invoked, which is how three
wrong percentages reached a compiled PDF. Whatever runs, runs from here.

DESIGN NOTE, and it is the reason this file parses text instead of computing.
The obvious implementation recomputes each statistic here and asserts the prose
against it. That would be wrong: bootstrap figures would then have two
independent derivations that can disagree with each other while both are
"right", and the paper would have a table and a sentence sourced from different
draws. Instead this reads the analysis scripts' own stdout -- the same run that
produced the LaTeX table body -- and asserts the prose against that. There is
exactly one source of truth per number, and it is the generator.

Run:  python3 analysis/results_facts.py           # print what was parsed
      python3 analysis/results_facts.py --check   # assert the shipped values
      python3 analysis/results_facts.py --fresh   # ignore the cache, re-run

Twenty-six of Section 3's values are read off the harness source tree, which
ships in this repository as clasharness/; CLASHARNESS_SRC defaults to the
repository root and only needs setting to check a different checkout. If the
tree is unreachable --check fails rather than passing on the other seventy-two.

Generators are slow (clustering_v8 alone is ~40s of bootstrap), so their output
is cached under /tmp keyed on the script's mtime; editing a generator
invalidates its cache automatically.
"""
import os
import re
import subprocess
import sys

# Imported for ROSTER SIZES ONLY, never for a statistic. The design note above
# forbids recomputing a number here, and this does not: how many models are on
# a panel is a fact about the experiment's shape that corpus owns outright, and
# hardcoding it here is what broke when the open-weight group landed -- every
# "N/9" pattern in this file silently stopped matching a twelve-row table.
import corpus

# Figure 2's guard. It reads the shipped SVG and re-derives what the figure
# draws from ablation_facts()/generator_facts() below, so it imports this
# module back -- inside its function body, which is why this import is safe.
import figure_facts as _figure_facts

# The judge-provenance ablation's guard. It asserts main.tex against the frozen
# artifact src/report/judge_ablation.json, so it needs neither CLASHARNESS_SRC nor
# eval_results and costs a file read.
import judge_ablation as _judge_ablation

# Section 3's guard. It recomputes from the corpus rather than parsing
# generator stdout, so it keeps its own SHIPPED dict and its own entry point;
# what is imported here is its verdict. See section3_facts() at the bottom.
import section3_facts as _section3_facts

# Appendix B's three round-example exhibits. Unlike everything else here these
# claim literal strings off rendered pixels ("the cache reads
# VERIFIED_CACHE_VALIDATED"), so the module re-reads the corpus renders and
# asserts each one; what lands in SHIPPED is its tally and its mismatch list.
import round_example_facts as _round_example_facts

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- paper src --
# This script is a FACT-CHECKER for the paper: it re-derives numbers from the
# corpus and asserts that the LaTeX/SVG sources print them. Those sources live
# in the paper repository, not here, so point CLASHARNESS_PAPER_SRC at that
# checkout to run it. Everything else under analysis/ works without it.
PAPER_SRC = os.environ.get("CLASHARNESS_PAPER_SRC", os.path.dirname(HERE))


def _paper_file(relpath):
    """Absolute path to a paper source file, or exit with something actionable."""
    path = os.path.join(PAPER_SRC, relpath)
    if not os.path.exists(path):
        raise SystemExit(
            f"missing paper source {relpath}: looked in {PAPER_SRC}\n"
            f"  This script checks the paper's numbers against the corpus, so it\n"
            f"  needs the paper checkout. Set CLASHARNESS_PAPER_SRC to it:\n"
            f"    CLASHARNESS_PAPER_SRC=/path/to/paper python {{}}\n"
            f"  The rest of analysis/ does not need it.".format(
                os.path.relpath(__file__, os.getcwd())))
    return path

CACHE = "/tmp/results_facts_cache"


_MEMO = {}


def run(script, args=()):
    """stdout of analysis/<script> <args>, cached on the script's mtime.

    Memoised for the life of the process as well as on disk. --fresh disables
    the disk cache, and without the memo it would also make two sections that
    want the SAME output run the generator twice: main_facts and
    main_bands_facts both need difficulty_bands.py at high effort, and the
    high-effort table is the slowest of the three. The memo is keyed on
    (script, args) exactly as the disk cache is, so it can never merge two
    different invocations.

    `args` joins the cache key rather than only the command line:
    difficulty_bands.py prints a different table per reasoning effort, and one
    log per script would have all three efforts overwriting each other and
    returning whichever ran last. The script name stays the leftmost part of
    the key so the stale-sweep below, which matches on that prefix, still
    clears EVERY variant when the script's mtime moves -- an edit to the
    generator invalidates all three of its tables, not just the one that
    happened to be asked for first.
    """
    key = (script, tuple(args))
    if key not in _MEMO:
        _MEMO[key] = _run(script, tuple(args))
    return _MEMO[key]


def _run(script, args):
    """run() without the memo -- the disk cache and the subprocess."""
    path = os.path.join(HERE, script)
    stamp = int(os.path.getmtime(path))
    slug = ".".join((script, *args))
    out = os.path.join(CACHE, f"{slug}.{stamp}.log")
    if "--fresh" not in sys.argv and os.path.exists(out):
        return open(out).read()
    os.makedirs(CACHE, exist_ok=True)
    for stale in os.listdir(CACHE):
        if stale.startswith(slug + "."):
            os.remove(os.path.join(CACHE, stale))
    sys.stderr.write(f"  running {' '.join((script, *args))} ...\n")
    r = subprocess.run([sys.executable, path, *args], capture_output=True,
                       text=True, cwd=os.path.dirname(HERE))
    if r.returncode:
        raise SystemExit(f"{script} failed:\n{r.stderr[-2000:]}")
    open(out, "w").write(r.stdout)
    return r.stdout


def _rows(log, pipes, want=None, skip_header=True):
    """Data rows of one printed block, split on '|'.

    The blocks are told apart by how many pipes they carry, which is stable
    across runs in a way that line offsets are not. `want` further filters.
    """
    o = []
    for l in log.split("\n"):
        if l.count("|") != pipes or not re.match(r"^[a-z]", l):
            continue
        if skip_header and l.split("|")[0].strip() == "model":
            continue
        if want and not want(l):
            continue
        o.append(l)
    return o


def _separated(log):
    """Models difficulty_bands.py PRINTED as having an interval above zero.

    Parsed rather than recomputed from the row intervals, for the reason the
    module docstring gives: the generator owns the definition of "separated"
    (strictly above zero, on the cluster bootstrap, at this effort), and a
    second definition written here would agree today and drift the first time
    the generator's rule changes. Model names carry no commas, so the split is
    safe.
    """
    m = re.search(r"models have a cluster-bootstrap interval strictly above "
                  r"zero: (.+)", log)
    return [s.strip() for s in m.group(1).split(",")]


# --------------------------------------------------------------- sec 6.1
def main_facts():
    """difficulty_bands.py -- everything sec:main-results states.

    NOTE ON THE CACHE. difficulty_bands.py imports table1_v8 for the
    whole-corpus column, and run() keys its cache on the importing script's
    mtime alone. Editing table1_v8.py therefore changes these numbers without
    invalidating the cache; pass --fresh after touching it.
    """
    log = run("difficulty_bands.py")
    f = {}

    # Data block: name | pass all/easy/mid/hard | score all/e/m/h |
    # micro e/m/h | delta [ci] pooled.  Four pipes.
    rows = _rows(log, 4)
    # High effort carries the COMBINED panel. Taken from corpus rather than
    # written as a literal, and used as the denominator of every count parsed
    # below, so that adding a model to the roster moves this file's patterns
    # with it instead of turning them into silent non-matches.
    n = corpus.N_PANEL
    assert len(rows) == n, (len(rows), n)
    by = {}
    for l in rows:
        p = l.split("|")
        # The generator stars rows whose assets arrived as OCR text
        # (difficulty_bands.OCR_MARK_TXT). The mark is presentation; strip it
        # so the keys of `by` are the model names every other lookup uses.
        name = p[0].strip().rstrip("*")
        pas = [float(x) for x in p[1].split()]
        scr = [float(x) for x in p[2].split()]
        d = re.search(r"([+-][\d.]+)\s*\[\s*([+-][\d.]+),\s*([+-][\d.]+)\]", p[4])
        by[name] = {"pass": pas, "score": scr,
                    "delta": float(d.group(1)),
                    "ci": (float(d.group(2)), float(d.group(3)))}
    f["main_top_pass"] = by["claude-opus-5"]["pass"][0]
    f["main_top_score"] = by["claude-opus-5"]["score"][0]
    # The headline. Both the contribution bullet and Finding 1 name a
    # "39.0-point Competence-Resolution Gap", which is the SUBTRACTION of the
    # two numbers above and was the last arithmetic in the abstract-adjacent
    # prose that nothing checked. Derived here rather than parsed because the
    # generator does not print it: the design note bars re-deriving a
    # STATISTIC, and this is the difference of two values that came from the
    # generator's own row, so there is still exactly one source of truth.
    f["main_top_gap"] = round(100 * f["main_top_score"] - f["main_top_pass"], 1)

    # "pass rate falls from easy to mid to hard for 9 of the 9 models"; the
    # two score counts are the refutation the prose leans on, so both are
    # pinned -- a regeneration that made either non-zero would turn the
    # "not an averaging artifact" sentence into a false claim.
    #
    # THE LAST COUNTER IS NOT THE NEGATION OF THE OTHER TWO. "score monotone"
    # fails if EITHER end of easy>mid>hard fails; the prose sentence is the
    # narrower "mid scores above easy", which can be false while monotonicity
    # is also false. They were conflated in an earlier draft -- 0/9 monotone
    # was cited as if it established 9/9 inverted -- so the generator now
    # prints the inversion separately and it is pinned separately.
    for key, pat in (
            ("main_pass_mono", rf"pass monotone \S+\s*:\s*(\d+)/{n}\b"),
            ("main_score_mono_macro", rf"macro \(per-round\)\s*:\s*(\d+)/{n}\b"),
            ("main_score_mono_micro", rf"micro \(per-check\)\s*:\s*(\d+)/{n}\b"),
            ("main_score_inv", rf"mid > easy\s*:\s*(\d+)/{n}\b")):
        m = re.search(pat, log)
        assert m, f"{key}: no line matching {pat!r}; the generator's " \
                  f"denominator is not {n}"
        f[key] = int(m.group(1))

    # tab:main's caption: "The five provider-error episodes are scored as
    # failures". Pinned for the same reason bands_medium_errors and
    # bands_low_errors are: the count is NOT a constant of the harness, it is
    # 5 / 2 / 0 across the three efforts, so a caption stating one of them is
    # asserting a number rather than restating a convention. The high row was
    # the original and was the one member of the trio left without a key when
    # the other two were added.
    f["main_errors"] = int(re.search(
        r"provider errors retained as failures: (\d+)", log).group(1))

    req = re.search(r"easy ([\d.]+), mid ([\d.]+), hard ([\d.]+)", log)
    f["main_req_easy"] = float(req.group(1))
    f["main_req_mid"] = float(req.group(2))
    f["main_req_hard"] = float(req.group(3))

    f["main_separated"] = int(
        re.search(rf"(\d+)/{n} models have a cluster", log).group(1))
    sep = [m for m in by if by[m]["ci"][0] > 0]
    assert len(sep) == f["main_separated"] == len(_separated(log)), (
        sep, f["main_separated"], _separated(log))
    f["main_delta_min"] = min(by[m]["delta"] for m in sep)
    f["main_delta_max"] = max(by[m]["delta"] for m in sep)
    f["main_delta_min_model"] = min(sep, key=lambda m: by[m]["delta"])
    f["main_delta_max_model"] = max(sep, key=lambda m: by[m]["delta"])

    # THE WEAKEST SEPARATION IS RANKED ON THE INTERVAL, NOT ON THE POINT
    # ESTIMATE. Both happen to name the same model here, which is exactly why
    # the two are pinned apart: the prose calls it the weakest separation, a
    # statement about how much of the interval clears zero, and a future run
    # can move the argmin of one without moving the argmin of the other.
    f["main_sep_min_ci_lo"] = min(by[m]["ci"][0] for m in sep)
    f["main_sep_min_model"] = min(sep, key=lambda m: by[m]["ci"][0])

    # The floored model, quoted with all three band rates plus the whole-corpus
    # one so the "no room to fall" reading is checkable rather than asserted.
    # The `all` figure is what the low-effort paragraph contrasts against.
    fl = by["gemini-3.5-flash"]
    f["main_floor_pass"] = fl["pass"][0]
    f["main_floor_easy"], f["main_floor_hard"] = fl["pass"][1], fl["pass"][3]
    f["main_floor_delta"] = fl["delta"]
    f["main_floor_ci_lo"], f["main_floor_ci_hi"] = fl["ci"]

    # THE OLD PATTERN READ A SIGNED RANGE WITH AN UNSIGNED GROUP. It was
    # `overstates the gap by ([\d.]+)-([\d.]+) points`, and the generator has
    # been printing a negative minimum at medium and low for as long as those
    # tables have existed: re.search simply skipped the minus and matched
    # "1.0-4.5", so group(1) silently meant the opposite of its name. Only
    # group(2) was ever shipped, which is why nothing caught it. The generator
    # now prints signed endpoints and a separate per-direction split, and this
    # reads the overstatement bound off the line that is only about
    # overstatement.
    conf = re.search(r"pooling shifts the gap by ([+-][\d.]+) to "
                     r"([+-][\d.]+) points \((\d+)/(\d+) seeds", log)
    assert conf, "difficulty_bands.py no longer prints the confound range"
    f["main_pool_max"] = float(
        re.search(r"overstated for \d+/\d+ models \(largest ([+-][\d.]+)\)",
                  log).group(1))
    assert f["main_pool_max"] == float(conf.group(2)), (
        "the largest overstatement and the top of the printed range disagree")
    f["main_both_bands"] = int(conf.group(3))
    f["main_seeds"] = int(conf.group(4))
    nh = re.search(r"(\d+) seeds hold NO hard rounds and contribute (\d+) easy", log)
    f["main_nohard_seeds"] = int(nh.group(1))
    f["main_nohard_easy"] = int(nh.group(2))
    hv = re.search(r"heaviest hard seed is (\S+) with (\d+) of (\d+) hard "
                   r"against (\d+) easy", log)
    f["main_heavy_seed"] = hv.group(1)
    f["main_heavy_hard"] = int(hv.group(2))
    f["main_heavy_total"] = int(hv.group(3))
    f["main_heavy_easy"] = int(hv.group(4))
    return f


def main_bands_facts():
    """difficulty_bands.py at medium and low -- tab:main-medium, tab:main-low.

    NOT A DUPLICATE OF effort_facts() BELOW, and the two must not be merged.
    sec:effort asks whether effort changes the OUTCOME, and answers it with
    McNemar over paired rounds on the full 600. This asks whether the
    DIFFICULTY AXIS survives a change of effort, and answers it by rebuilding
    the same banded table three times. Same runs, different question, and the
    two use different estimators; a shared prefix would invite someone to
    quote a number from one under the other's pool statement.

    The row order of all three tables is pinned to high, so "seven of the
    eight" below is an intersection of model SETS and not of row positions --
    which is also why `kept` is computed here from the two printed lists
    rather than read off a single line of either.

    Cache caveat is main_facts()'s: table1_v8.py feeds the `all` column
    without being the mtime the cache keys on, so pass --fresh after editing
    it. run() clears all three efforts together when difficulty_bands.py
    itself moves.
    """
    hi_sep = set(_separated(run("difficulty_bands.py")))
    # The two lower tables carry the nine API models: the open-weight group was
    # only ever run at high, so its rows are ABSENT here, not dropped.
    n9 = len(corpus.MODEL_LABEL)
    f = {}
    for lvl in ("medium", "low"):
        log = run("difficulty_bands.py", (lvl,))
        p = f"bands_{lvl}_"
        rows = _rows(log, 4)
        assert len(rows) == n9, (lvl, len(rows), n9)

        for key, pat in (("pass_mono", rf"pass monotone \S+\s*:\s*(\d+)/{n9}\b"),
                         ("score_inv", rf"mid > easy\s*:\s*(\d+)/{n9}\b")):
            m = re.search(pat, log)
            assert m, (lvl, key, pat)
            f[p + key] = int(m.group(1))

        # ZERO HAS TO BE PARSED, NOT INFERRED FROM A MISSING LINE. Low effort
        # produced no provider errors at all, and the generator used to print
        # nothing in that case -- indistinguishable from the line having been
        # renamed. It now prints "0 (none)" and this reads the count.
        f[p + "errors"] = int(re.search(
            r"provider errors retained as failures: (\d+)", log).group(1))

        # INTERSECT WITH WHAT THIS LEVEL ACTUALLY RAN BEFORE DIFFERENCING. A
        # bare `hi_sep - sep` counts every open-weight model as having "lost"
        # its separation at medium and low, when in fact it was never run
        # there -- three absent rows would be reported as three regressions,
        # and the prose sentence "the eighth, gemini-3.1-pro" would quietly
        # become a claim about four models.
        present = {r.split("|")[0].strip().rstrip("*") for r in rows}
        base = hi_sep & present
        f[p + "base"] = len(base)
        sep = set(_separated(log))
        assert sep <= present, (lvl, sorted(sep - present))
        f[p + "kept"] = len(base & sep)
        f[p + "lost"] = sorted(base - sep)

        row = _rows(log, 4, want=lambda l: l.startswith("gemini-3.5-flash"))
        assert len(row) == 1, (lvl, row)
        col = row[0].split("|")
        f[p + "floor_pass"] = float(col[1].split()[0])
        d = re.search(r"([+-][\d.]+)\s*\[\s*([+-][\d.]+),\s*([+-][\d.]+)\]",
                      col[4])
        f[p + "floor_delta"] = float(d.group(1))
        f[p + "floor_ci_lo"], f[p + "floor_ci_hi"] = (float(d.group(2)),
                                                      float(d.group(3)))
    return f


def main_caption_facts():
    """difficulty_bands.py all -- the zero-bound sentence in tab:main's caption.

    THE ONLY READER OF THE MULTI-EFFORT INVOCATION. Every other section asks
    difficulty_bands.py for one effort at a time; `all` is the run that renders
    tab:main itself, and verify_table_bodies.py already checks the LaTeX body
    it prints. What a body check cannot see is the CAPTION, and that caption
    now quotes two counts and two intervals about the cells whose printed
    "pt +- sd" reconstructs below zero. Nothing else in the repo would notice
    those going stale, which is the whole reason this function exists.

    "zerobound", not "floor": main_floor_* and bands_*_floor_* above are the
    bottom-ranked MODEL's whole-corpus row. These are CELLS pressed against the
    zero bound. Both happen to be gemini-3.5-flash, so reusing the word would
    make two different claims impossible to grep apart --
    difficulty_bands.print_zero_bound_cells says the same thing from its end.

    Cost: `all` rebuilds the banded table at all three efforts WITH standard
    errors, so it is the slowest single invocation in this file. It earns its
    cache entry: the alternative is a caption whose intervals can drift out of
    agreement with the table directly above them and have nothing fail.
    """
    log = run("difficulty_bands.py", ("all",))
    f = {}
    n = re.search(r"zero-bound cells: (\d+)", log)
    assert n, "difficulty_bands.py all no longer prints the zero-bound block"
    f["main_zerobound_n"] = int(n.group(1))

    cells = re.findall(
        r"zero-bound cell: (\S+) (\S+) (\S+) (\d+)/(\d+) pt=([\d.]+) "
        r"sd=([\d.]+) naive_lo=([+-][\d.]+) pctl=\[([\d.]+),([\d.]+)\] "
        r"at_zero=([\d.]+)", log)
    assert len(cells) == f["main_zerobound_n"], (len(cells), n.group(1))

    # The caption writes the model once and the band once, then distinguishes
    # the two cells only by effort. That phrasing is only correct while the
    # flagged set really is one model and one band, so it is asserted rather
    # than assumed.
    assert len({c[0] for c in cells}) == 1, sorted({c[0] for c in cells})
    assert len({c[2] for c in cells}) == 1, sorted({c[2] for c in cells})
    f["main_zerobound_model"] = cells[0][0]
    f["main_zerobound_band"] = cells[0][2]

    by_effort = {c[1]: c for c in cells}
    assert set(by_effort) == {"High", "Med"}, sorted(by_effort)
    for eff, key in (("High", "high"), ("Med", "med")):
        (_m, _e, _b, k, nn, _pt, _sd, naive, lo, hi, _z) = by_effort[eff]
        # Definitional -- but it is the generator's selection rule, and a
        # generator that stopped applying it would hand this caption a cell
        # the caption's sentence does not describe.
        assert float(naive) < 0, (eff, naive)
        # THE CLAIM THE CAPTION MAKES. "right-skewed against that bound" is
        # only worth printing while the percentile interval actually stops AT
        # the bound; a flagged cell whose lower limit came back positive would
        # need the sentence rewritten, not the number swapped.
        assert float(lo) == 0.0, (eff, lo)
        f[f"main_zerobound_{key}_k"] = int(k)
        f[f"main_zerobound_{key}_n"] = int(nn)
        f[f"main_zerobound_{key}_hi"] = float(hi)
    return f


def cost_facts():
    """cost.py -- the two cost paragraphs of sec:main-results.

    Reaching these numbers at all requires cost.py's medium self-check to pass,
    since it runs first and aborts the script on a mismatch. So a --check run
    that gets this far has also re-proved that the ported pricing engine still
    reproduces the frozen behavioural artifact field for field.

    NOTE ON THE CACHE, twice over: cost.py imports check_split_v8 for spearman
    and corpus for the outcomes, and run() keys its cache on cost.py's mtime
    alone. Pass --fresh after touching either.
    """
    log = run("cost.py")
    f = {}

    tot = re.search(r"one pass over the corpus: \$([\d,]+\.\d+) over "
                    r"([\d,]+) calls", log)
    f["cost_total"] = float(tot.group(1).replace(",", ""))
    f["cost_calls"] = int(tot.group(2).replace(",", ""))

    f["cost_pass_spread"] = float(
        re.search(r"pass spread\s*:.*= ([\d.]+)x", log).group(1))

    ch = re.search(r"cheapest solve\s*:\s*(\S+) at \$([\d.]+) \(([\d.]+)% pass, "
                   r"([\d.]+) points behind (\S+)\)", log)
    f["cost_cheapest_model"] = ch.group(1)
    f["cost_cheapest"] = float(ch.group(2))
    f["cost_cheapest_pass"] = float(ch.group(3))
    f["cost_points_behind"] = float(ch.group(4))

    de = re.search(r"dearest solve\s*:\s*(\S+) at \$([\d.]+)", log)
    f["cost_dearest_model"] = de.group(1)
    f["cost_dearest"] = float(de.group(2))

    # The three integers the prose quotes are read from the line that rounds
    # them once from the unrounded ratios, NOT recomputed from the dollars
    # above -- dividing two already-rounded prices is how a 24x becomes a 25x.
    pr = re.search(r"prose ratios\s*:\s*solve-spread (\d+)x, "
                   r"leader-over-cheapest per solve (\d+)x, per round (\d+)x", log)
    f["cost_solve_spread"] = int(pr.group(1))
    f["cost_leader_ratio_solve"] = int(pr.group(2))
    f["cost_leader_ratio_round"] = int(pr.group(3))

    sh = re.search(r"reasoning share\s*:\s*([\d.]+)% \((\S+)\) to "
                   r"([\d.]+)% \((\S+)\)", log)
    f["cost_share_lo"] = float(sh.group(1))
    f["cost_share_lo_model"] = sh.group(2)
    f["cost_share_hi"] = float(sh.group(3))
    f["cost_share_hi_model"] = sh.group(4)
    f["cost_rho_share"] = float(
        re.search(r"reasoning SHARE of output\s*:\s*([+-][\d.]+)", log).group(1))
    f["cost_rho_tokens"] = float(
        re.search(r"reasoning TOKENS per round\s*:\s*([+-][\d.]+)", log).group(1))
    f["cost_reason_span"] = float(
        re.search(r"reasoning tokens per round span ([\d.]+)x", log).group(1))

    lo = re.search(r"leave-one-out rho\(tokens\): ([+-][\d.]+) \(drop \S+\) "
                   r"to ([+-][\d.]+) \(drop (\S+)\)", log)
    f["cost_rho_loo_min"] = float(lo.group(1))
    f["cost_rho_loo_max"] = float(lo.group(2))
    f["cost_rho_loo_weakest_drop"] = lo.group(3)
    # The interval must not straddle zero, or "correlates negatively" is a
    # statement about which model happened to be included.
    assert f["cost_rho_loo_max"] < 0, "rho changes sign under a single deletion"

    # WITHIN-MODEL EFFORT RESPONSE. The rho above is a between-model rank
    # correlation and the section that quotes it makes a causal claim; these
    # are the numbers that decide whether that claim is allowed. Sign counts
    # first, because the load-bearing statement is qualitative -- exactly one
    # model's interval is negative -- and it survives the two marginal cells
    # (claude-opus-4.8 opens at +0.2) moving between "helped" and
    # "indistinguishable", which "5 helped" on its own would not.
    er = re.search(r"counts: (\d+) helped, (\d+) hurt, (\d+) "
                   r"indistinguishable \(n=(\d+)\)", log)
    f["effort_helped"] = int(er.group(1))
    f["effort_hurt"] = int(er.group(2))
    f["effort_flat"] = int(er.group(3))
    f["effort_n"] = int(er.group(4))
    assert (f["effort_helped"] + f["effort_hurt"] + f["effort_flat"]
            == f["effort_n"]), "effort response counts do not partition"
    inv = re.search(r"sole inversion: (\S+) ([\d.]+)% at low -> ([\d.]+)% at "
                    r"high, ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]", log)
    f["effort_inv_model"] = inv.group(1)
    f["effort_inv_low"] = float(inv.group(2))
    f["effort_inv_high"] = float(inv.group(3))
    f["effort_inv_delta"] = float(inv.group(4))
    f["effort_inv_ci_lo"] = float(inv.group(5))
    f["effort_inv_ci_hi"] = float(inv.group(6))
    # "Sole" is a claim, not a label. If a second model ever inverts, the
    # generator's own count moves and this fires before the prose can inherit
    # a word that stopped being true.
    assert f["effort_hurt"] == 1, (
        f"prose says one model inverts, generator counts {f['effort_hurt']}")
    assert f["effort_inv_ci_hi"] < 0, "the sole inversion no longer excludes 0"

    # The tab:cost caption makes an arithmetic claim about its OWN rendered
    # cells -- that the Total row sits below the column it closes, because the
    # column is rounded and the total is not. The table body is guarded by
    # verify_table_bodies.py; a caption is not, so the gap is asserted here.
    #
    # THERE ARE NOW FOUR TOTAL ROWS AND THE CLAIM IS PER ROW. The nine API
    # models are billed spend and now appear at three effort levels; the three
    # open-weight models are self-hosted, ran at high only, and their column is
    # list-rate arithmetic, so no two rows are ever summed into a grand total
    # and each closes its own column. Keyed on the row label the generator
    # prints rather than on position, and ALL FOUR are required to be present --
    # a caption that explains one rounding gap while the table shows four is the
    # failure this is here to stop.
    #
    # Only the two high-effort rows are pinned in SHIPPED below. The caption no
    # longer quotes any of these gaps as a figure -- it states the rounding
    # relationship qualitatively, which is what lets the medium and low rows
    # exist without four more pinned numbers that would churn on every reprice.
    # The pins are kept anyway as a GENERATOR guard: they are the thing that
    # notices the priced column moving when nothing was supposed to move. The
    # medium and low rows are covered by the structural assert above, which is
    # what catches a dropped or renamed total row.
    # The two gap figures are SIGNED: rounding can put the printed column on
    # either side of the unrounded total, and the medium API row lands under it.
    # Matching a bare [\d,]+ there would silently skip whichever rows happen to
    # round the other way, and the set() assert below would then report them as
    # missing total rows rather than as a parse gap.
    #
    # KEYED ON "<group> @ <effort>", not on the rendered row label. The three
    # API total rows all render as "Total (API, billed)" now that the Think
    # column disambiguates them, so the label alone is no longer unique and a
    # label-keyed dict would silently collapse three rows into one and still
    # look like a clean parse.
    cap = {f"{m.group(1)} @ {m.group(2)}":
           tuple(int(x.replace(",", "")) for x in m.groups()[2:])
           for m in re.finditer(
               r"caption fact \[Total \(([^)]+)\) @ (\w+)\]: the printed \$ "
               r"cells sum to ([\d,]+) and the printed tok \(M\) cells to "
               r"([\d,]+), so the row sits ([+-]?[\d,]+) dollars and "
               r"([+-]?[\d,]+) million tokens below them", log)}
    assert set(cap) == {"API, billed @ High", "API, billed @ Med",
                        "API, billed @ Low", "open, list rate @ High"}, sorted(cap)
    (f["cost_col_sum_usd"], f["cost_col_sum_tok_m"],
     f["cost_round_gap_usd"], f["cost_round_gap_tok_m"]) = cap["API, billed @ High"]
    (f["cost_open_col_sum_usd"], f["cost_open_col_sum_tok_m"],
     f["cost_open_gap_usd"], f["cost_open_gap_tok_m"]) = cap["open, list rate @ High"]

    # Read off the worst model's own row rather than restated, so the two
    # figures the paragraph pairs stay attached to one model.
    row = [l for l in _rows(log, 3) if l.startswith("gemini-3.5-flash")]
    assert len(row) == 1, row
    mid, tail = row[0].split("|")[2], row[0].split("|")[3]
    f["cost_worst_reason_k"] = float(mid.split()[3])
    f["cost_worst_cap"] = float(tail.split()[0])
    return f


# ----------------------------------------------------------------- sec 5
def validity_facts():
    """check_split_v8.py -- the rubric split and the judge-reach bound.

    THE LARGEST UNGUARDED BLOCK IN THE PAPER UNTIL NOW. check_split_v8.py
    already fed tab:validity, whose body verify_table_bodies.py byte-diffs, so
    the table was safe -- but the table body is one row of denominators. The
    judge bound the validity argument actually rests on lives in prose and in
    the caption: 19,419 verdicts, 97.5% unanimous, 479 split, the 2.33-point
    flip. None of it was checked, in the one section a reviewer reads to
    decide whether to believe the rest of the paper.

    sec:validity (5) and sec:app-validity (C.3) state the same numbers at
    different precisions -- 2.3 against 2.33, 97.5 against 97.53 -- so both
    renderings are pinned. Quoting a rounded value is fine; quoting a rounded
    value that does not round from the exact one is not.
    """
    log = run("check_split_v8.py")
    f = {}

    # "Of the 3,284 scored checks, 34.9% (1,147) are deterministic ... and
    # 65.1% (2,137) are scoped semantic matches". Taken from the generator's
    # own caption line, which is the string the table's denominator row is
    # built from, rather than re-added from the bucket rows above it.
    m = re.search(r"caption facts: ([\d.]+)% of the ([\d,]+) required units "
                  r"are LLM semantic matches \((\d+)\) and ([\d.]+)% are "
                  r"deterministic \((\d+)\)", log)
    assert m, "check_split_v8 caption-facts line not found"
    f["val_semantic_pct"] = float(m.group(1))
    f["val_scored"] = int(m.group(2).replace(",", ""))
    f["val_semantic"] = int(m.group(3))
    f["val_det_pct"] = float(m.group(4))
    f["val_det"] = int(m.group(5))
    # The two buckets must exhaust the scored denominator and the two shares
    # must sum to 100. Both are one line of arithmetic, and both were wrong in
    # an earlier draft that quoted 1,593 deterministic checks.
    f["val_buckets_exhaust"] = (f["val_semantic"] + f["val_det"]
                                == f["val_scored"])
    f["val_shares_sum"] = round(f["val_semantic_pct"] + f["val_det_pct"], 1)

    # "the 476 diagnostic checks carry no score ... pooling them would inflate
    # the deterministic denominator to 1,623". Both numbers are in one line.
    m = re.search(r"required total\s+(\d+)\s+\((\d+) diagnostic checks", log)
    assert m, "required-total line not found"
    assert int(m.group(1)) == f["val_scored"], (m.group(1), f["val_scored"])
    f["val_diagnostic"] = int(m.group(2))
    f["val_det_pooled"] = f["val_det"] + f["val_diagnostic"]

    # "the stored pass/fail equals the majority vote on all 19,419 tagged
    # verdicts", "97.5% are unanimous and 2.5% (479) are split".
    m = re.search(r"([\d,]+) of them carry a recorded tally", log)
    assert m, "verdict-tally line not found"
    f["val_verdicts"] = int(m.group(1).replace(",", ""))
    m = re.search(r"unanimous (\d+) \(([\d.]+)%\)\s+split (\d+) \(([\d.]+)%\)",
                  log)
    assert m, "unanimous/split line not found"
    f["val_unanimous"] = int(m.group(1))
    f["val_unanimous_pct"] = float(m.group(2))
    f["val_split"] = int(m.group(3))
    f["val_split_pct"] = float(m.group(4))
    # What the prose actually prints, one decimal. 97.53 -> 97.5 and 2.47 ->
    # 2.5, so the pair does NOT sum to 100 as printed; that is correct and is
    # pinned so nobody "fixes" it into 97.5/2.5 summing by hand.
    f["val_unanimous_1dp"] = round(f["val_unanimous_pct"], 1)
    f["val_split_1dp"] = round(f["val_split_pct"], 1)
    f["val_verdicts_partition"] = (f["val_unanimous"] + f["val_split"]
                                   == f["val_verdicts"])
    f["val_majority_is_stored"] = bool(
        re.search(r"the stored pass/fail equals the majority vote on all "
                  + str(f["val_verdicts"]), log))

    # "the split rate varies little across models (1.4% for gemini-3.7-flash
    # to 3.4% for gemini-3.1-pro)". Names as well as rates: the sentence is a
    # claim about which models sit at the ends.
    m = re.search(r"per-model split rate runs ([\d.]+)% \((\S+)\) to "
                  r"([\d.]+)% \((\S+)\)", log)
    assert m, "per-model split-rate line not found"
    f["val_split_lo"] = round(float(m.group(1)), 1)
    f["val_split_lo_model"] = m.group(2)
    f["val_split_hi"] = round(float(m.group(3)), 1)
    f["val_split_hi_model"] = m.group(4)

    # "Flipping every split verdict at once changes task success by at most
    # 2.33 points and moves any rank by at most one position."
    m = re.search(r"flipping ALL (\d+) split verdicts at once moves task "
                  r"success by at most ([\d.]+)pp and any rank by at most "
                  r"(\d+) position", log)
    assert m, "flip-bound line not found"
    assert int(m.group(1)) == f["val_split"], (m.group(1), f["val_split"])
    f["val_flip_swing"] = float(m.group(2))
    f["val_flip_swing_1dp"] = round(f["val_flip_swing"], 1)
    f["val_flip_rank"] = int(m.group(3))

    # The adversarial flip. sec:validity says the top model "still finishes
    # within one rank of first" and sec:app-validity says it "still finishes
    # second and the largest displacement anywhere is three positions", so the
    # rank and the displacement are separate keys -- the first sentence stays
    # true for a range of outcomes the second one does not.
    m = re.search(r"adversarial \([^)]*\): (\S+) still ranks #(\d+); largest "
                  r"drop anywhere is (\S+) #(\d+) -> #(\d+)", log)
    assert m, "adversarial-flip line not found"
    f["val_adv_model"] = m.group(1)
    f["val_adv_rank"] = int(m.group(2))
    f["val_adv_worst_model"] = m.group(3)
    f["val_adv_worst_move"] = int(m.group(5)) - int(m.group(4))

    # "Semantic and deterministic rankings do also track each other (rho =
    # +0.90 ... leave-one-out +0.87 to +0.93)". The leave-one-out RANGE is
    # printed on its own line; the +0.87 on the rho line is one endpoint of it
    # and is asserted to agree rather than parsed twice.
    m = re.search(r"rho\(LLM semantic, pooled deterministic\) over (\d+) "
                  r"models = \+([\d.]+) on the judged denominator", log)
    assert m, "semantic-vs-deterministic rho line not found"
    f["val_rho_n"] = int(m.group(1))
    f["val_rho_judged"] = float(m.group(2))
    m = re.search(r"leave-one-out range \+([\d.]+)\.\.\+([\d.]+)", log)
    assert m, "leave-one-out line not found"
    f["val_rho_loo_lo"] = float(m.group(1))
    f["val_rho_loo_hi"] = float(m.group(2))
    assert f["val_rho_loo_lo"] <= f["val_rho_judged"] <= f["val_rho_loo_hi"], \
        "the headline rho is outside its own leave-one-out range"

    # "locating evidence artifacts and transcribing ... barely transfer as
    # ranks (rho = +0.02)", and "the choice of denominator itself re-ranks 7 of
    # 10 models, which is why both are reported".
    m = re.search(r"rho\(asset_gate, content\) = \+([\d.]+) over (\d+) models",
                  log)
    assert m, "asset-vs-content rho line not found"
    f["val_rho_asset_content"] = float(m.group(1))
    m = re.search(r"choice of denominator re-ranks (\d+)/(\d+) models", log)
    assert m, "denominator re-rank line not found"
    f["val_denom_reranks"] = int(m.group(1))
    # Every rho above is over the API panel, and each is parsed with its own n
    # so that a roster change cannot leave one of them quoted at the old size.
    assert {f["val_rho_n"], int(m.group(2))} == {corpus.N_MODELS}, \
        (f["val_rho_n"], m.group(2), corpus.N_MODELS)
    return f


# --------------------------------------------------------------- sec 6.2
def concentration_facts():
    """clustering_v8.py -- everything sec:concentration states."""
    log = run("clustering_v8.py")
    f = {}

    # Whole-corpus block: name | per-check obs indep excess | lift CI ceil
    # l/ceil | odisp.  Three pipes; the tercile and paired blocks have one.
    whole = _rows(log, 3)
    # Same reasoning as the roster count in table1_facts: the denominator comes
    # from corpus, not a literal, so adding a model moves this file with it.
    # All three blocks below are asserted against the SAME n -- if a future
    # generator emits them at different lengths that is a bug in the generator,
    # and the counts parsed out of them would silently disagree.
    n = corpus.N_PANEL
    assert len(whole) == n, (len(whole), n)
    name = [l.split("|")[0].strip() for l in whole]
    lift, ceil, lc = [], [], []
    for l in whole:
        mid = l.split("|")[2]
        lift.append(float(re.match(r"\s*([\d.]+)", mid).group(1)))
        a, b = mid.split("]")[1].split()[:2]
        ceil.append(float(a))
        lc.append(float(b))
    f["conc_lift_min"], f["conc_lift_max"] = min(lift), max(lift)
    f["conc_ceil_min"], f["conc_ceil_max"] = min(ceil), max(ceil)
    f["conc_lowest_lift_model"] = name[lift.index(min(lift))]
    # The anti-misreading note: lowest raw lift is NOT most independent.
    f["conc_max_liftceil"] = max(lc)
    f["conc_max_liftceil_model"] = name[lc.index(max(lc))]

    # Tercile block: name | easy[ci]* middle[ci]* hard[ci]*.  One pipe and
    # THREE bracket groups -- the paired block below also has one pipe and a
    # bracket (its single CI), so "has a [" is not enough to tell them apart.
    ter = _rows(log, 1, want=lambda l: l.count("[") == 3)
    assert len(ter) == n, (len(ter), n)
    # Match value+interval+optional star as ONE unit. Splitting on "]"
    # instead puts each star at the head of the FOLLOWING cell, which silently
    # reads every model as non-significant on the easy tercile.
    cell = re.compile(r"([\d.]+)\[[^\]]*\](\*?)")
    easy_sig = hard_sig = 0
    easy_names, hard = set(), []
    for l in ter:
        got = cell.findall(l.split("|")[1])
        assert len(got) == 3, (l, got)
        (_, e), (_, _m), (h, hs) = got
        if e == "*":
            easy_sig += 1
            easy_names.add(l.split("|")[0].strip())
        if hs == "*":
            hard_sig += 1
        hard.append(float(h))
    f["conc_easy_sig"], f["conc_hard_sig"] = easy_sig, hard_sig
    f["conc_hard_min"], f["conc_hard_max"] = min(hard), max(hard)

    # Paired block: name | lift peers diff CI p verdict   (one pipe, verdict)
    pair = _rows(log, 1, want=lambda l: re.search(r"(UP|DOWN|FLAT)\s*$", l))
    assert len(pair) == n, (len(pair), n)
    grab = lambda v: {l.split("|")[0].strip():
                      float(re.search(r"([+-][\d.]+)", l.split("|")[1]).group(1))
                      for l in pair if l.strip().endswith(v)}
    up, down = grab("UP"), grab("DOWN")
    f["conc_peer_up"], f["conc_peer_down"] = len(up), len(down)
    f["conc_peer_off"] = len(up) + len(down)
    f["conc_peer_top_model"] = max(up, key=up.get)
    f["conc_peer_top"] = max(up.values())
    f["conc_peer_pro"] = down.get("gemini-3.1-pro")
    f["conc_peer_flash35"] = down.get("gemini-3.5-flash")

    # The prose asserts the models that cluster on the easy tercile are not the
    # same models that sit off their peer mean. The two counts happen to be
    # close (10 and 9 over the twelve-model panel), so equal counts would not
    # catch a coincidence -- compare the SETS. If a regeneration ever makes
    # them coincide the sentence becomes false, not merely stale.
    f["conc_easy_peer_sets_differ"] = easy_names != (set(up) | set(down))

    # tab:rep-09 is cited as an instrument demonstration on the strength of
    # this line; if a pair ever qualifies, the citation has to be rewritten.
    f["conc_no_pair_qualifies"] = "NO PAIR QUALIFIES" in log
    return f


# --------------------------------------------------------------- sec 6.3
def capability_facts():
    """capability_axes_v8.py -- everything sec:capability-axes states."""
    log = run("capability_axes_v8.py")
    f = {}

    m = re.search(r"(\d+) of (\d+) model-category cells sit at a different "
                  r"rank", log)
    f["cap_rank_moves"], f["cap_cells"] = int(m.group(1)), int(m.group(2))

    # Denominator header rows of the grid, read by COLUMN NAME rather than by
    # position. The prose does not merely quote 3.5 and 15.0, it attaches each
    # to a denominator -- 3.5 "on the full 600 rounds", 15.0 on the column that
    # "rests on one seed". Taking min() and max() of the row would keep passing
    # if a regeneration reordered the columns and broke that attachment, so the
    # pairing is what gets asserted here.
    head = re.search(r"^model\s+(.+)$", log, re.M).group(1).split()
    grid = {}
    for key, row, cast in (("seeds", r"^seeds\s+(.+)$", int),
                           ("rounds", r"^n rounds\s+(.+)$", int),
                           ("boot", r"^boot \+-\s+(.+)$", float)):
        vals = [cast(x) for x in re.search(row, log, re.M).group(1).split()]
        assert len(vals) == len(head), (key, head, vals)
        grid[key] = dict(zip(head, vals))
    f["cap_nseeds"] = grid["seeds"]["ALL"]
    f["cap_all_rounds"] = grid["rounds"]["ALL"]
    f["cap_boot_all"] = grid["boot"]["ALL"]
    f["cap_boot_min"] = min(grid["boot"].values())
    # Narrowest and widest must be the two columns the prose names, or the
    # sentence has silently changed subject.
    f["cap_boot_narrowest_col"] = min(grid["boot"], key=grid["boot"].get)
    f["cap_boot_widest_col"] = max(grid["boot"], key=grid["boot"].get)
    thin = f["cap_boot_widest_col"]
    f["cap_boot_max"] = grid["boot"][thin]
    f["cap_min_seeds"] = grid["seeds"][thin]
    f["cap_min_rounds"] = grid["rounds"][thin]

    m = re.search(r"null = (\d+) random seed sets of the\s+"
                  r"same size drawn from the (\d+) seeds", log)
    f["cap_nperm"], f["cap_null_seeds"] = int(m.group(1)), int(m.group(2))

    # Specialization test block: category k spreadN pN spread6 p6, where N is
    # the full panel (12) -- the header widens with the roster, so the row
    # regex is written against positions, not against the literal "9".
    rows = re.findall(r"^  ([a-z_]+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+"
                      r"([\d.]+)\s+([\d.]+)\s*$", log, re.M)
    assert len(rows) == 6, rows
    f["cap_ncat"] = len(rows)
    alpha = 0.05 / len(rows)
    p6 = {r[0]: float(r[5]) for r in rows}
    f["cap_closest_cat"] = min(p6, key=p6.get)
    f["cap_closest_p_lead"] = min(p6.values())
    # The negative result. Counted from the per-category rows rather than read
    # off the generator's own summary line, so a change in either panel trips.
    f["cap_sig_all"] = sum(1 for r in rows if float(r[3]) < alpha)
    f["cap_sig_lead"] = sum(1 for r in rows if float(r[5]) < alpha)

    # The section also cites two appendix figures for the main-effect
    # contrast. Those numbers live in appendix_report.tex, which is now
    # hand-maintained (see its header) and can move underneath the body.
    ap = open(_paper_file("appendix_report.tex")).read()
    m = re.search(r"\\textbf\{(\d+) of the (\d+) designed levers push every "
                  r"one of the (\d+) models below its own baseline\}", ap)
    f["cap_levers_all_down"] = int(m.group(1)) if m else None
    f["cap_levers_models"] = int(m.group(3)) if m else None
    m = re.search(r"Rows 1--30 of (\d+); continued in "
                  r"Figure~\\ref\{fig:rep-heatmap-2\}", ap)
    f["cap_heatmap_scenarios"] = int(m.group(1)) if m else None
    return f


# --------------------------------------------------------------- sec 6.4
def taxonomy_facts():
    """failure_taxonomy.json -- everything sec:taxonomy states.

    The one section whose source is a factory artefact rather than an analysis
    script, so this reads it through src/make_body_figures.py: that module
    already cross-checks the file against corpus.py (same episodes, same pass
    rates) and owns bucket_rates(), the single division that turns failed and
    total required checks into the percentages the figure draws. Importing it
    rather than re-dividing here is what keeps the figure and the prose about
    the figure on one implementation.
    """
    sys.path.insert(0, os.path.dirname(_paper_file(os.path.join(
        "src", "make_body_figures.py"))))
    import make_body_figures as mbf              # noqa: E402

    buckets, cells, order = mbf.taxonomy()
    no35 = [m for m in order if m != "gemini-3.5-flash"]
    r = {b["key"]: mbf.bucket_rates(cells, order, b["key"]) for b in buckets}
    sp = lambda k, ms: round(max(r[k][m] for m in ms)                # noqa: E731
                             - min(r[k][m] for m in ms), 1)
    f = {}

    # THE FRONTIER CUT. This used to be order[:6] -- provider-blocked display
    # order -- and the comment here used to note that it "currently coincides
    # with the six highest pass rates" and warn that if the coincidence ever
    # broke, "top six" would become two different sets of models under one
    # name. It broke: gemini-3.7-flash passes 20.3 against gpt-5.6-luna's 19.8,
    # so it enters the ranked cut and luna leaves it, and three sections were
    # left meaning three different things by "the top six".
    #
    # It was not a naming problem. "Within the top six, content-level
    # differences largely disappear" held only under the display-order cut
    # (F1/F2/F4 spreads 5.0/4.6/7.0); under any ranked cut the same spreads are
    # 4.9/12.9/15.2, i.e. as large as the structural spread the sentence
    # contrasts them with. The finding was a property of the Claude and GPT
    # blocks, not of being at the frontier.
    #
    # So the cut is now corpus.leading_group: the models an all-pairs exact
    # McNemar cannot separate from the best one. Seven here, not six, and the
    # spread facts below are named _lead rather than _spread6 so that no pinned
    # value survives the change of meaning by accident.
    lead = corpus.leading_group(corpus.load_eval(corpus.PRIMARY_EFFORT,
                                                 keep_checks=False)[0])
    assert set(lead) <= set(order), (
        f"the frontier cut names models the taxonomy roster does not have: "
        f"{sorted(set(lead) - set(order))}")
    f["tax_lead_n"] = len(lead)
    f["tax_lead_is_prefix_of_rank"] = lead == sorted(
        order, key=lambda m: -cells[m]["pass_pct"])[:len(lead)]

    for fid, key in (("f1", "content_msg"), ("f2", "content_file"),
                     ("f3", "grounding"), ("f4", "anchor"),
                     ("f5", "side_effect"), ("f6", "window")):
        f[f"tax_{fid}_spread_all"] = sp(key, order)
        f[f"tax_{fid}_spread_no35"] = sp(key, no35)
        f[f"tax_{fid}_spread_lead"] = sp(key, lead)
        f[f"tax_{fid}_min"] = round(min(r[key].values()), 1)
        f[f"tax_{fid}_max"] = round(max(r[key].values()), 1)

    # The comparison the caveat paragraph turns on, asserted as a relation
    # rather than left to the reader to do in their head from six numbers.
    c6 = [f["tax_f1_spread_lead"], f["tax_f2_spread_lead"],
          f["tax_f4_spread_lead"]]
    f["tax_content_lead_under_f6"] = all(x < f["tax_f6_spread_lead"]
                                         for x in c6)
    f["tax_content_lead_under_f5"] = sum(x < f["tax_f5_spread_lead"]
                                         for x in c6)

    g = r["grounding"]
    asc = sorted(g, key=g.get)
    f["tax_f3_min_model"], f["tax_f3_max_model"] = asc[0], asc[-1]
    f["tax_f3_flash35"] = round(g["gemini-3.5-flash"], 1)
    f["tax_f3_flash35_rank"] = asc.index("gemini-3.5-flash") + 1
    # Why F3 survives dropping the weak models while the content buckets do
    # not: its extremes are both inside the leading group.
    f["tax_f3_ends_in_lead"] = {asc[0], asc[-1]} <= set(lead)

    # The episode-length confound. Both extremes of F3 are extremes of step
    # count, which is the paragraph's whole argument.
    steps = {m: cells[m]["step_mean"] for m in order}
    f["tax_sonnet_steps"] = steps["claude-sonnet-5"]
    f["tax_flash35_steps"] = steps["gemini-3.5-flash"]
    f["tax_sonnet_fewest_steps"] = min(steps, key=steps.get) == "claude-sonnet-5"
    f["tax_flash35_most_steps"] = max(steps, key=steps.get) == "gemini-3.5-flash"
    f["tax_sonnet_zero_open"] = cells["claude-sonnet-5"]["zero_open_tasks"]
    cap = [k for k in cells["gemini-3.5-flash"]["stops"]
           if k.startswith("reached max_steps=")]
    assert len(cap) == 1, cap
    f["tax_step_cap"] = int(cap[0].split("=")[1])
    f["tax_flash35_capped"] = cells["gemini-3.5-flash"]["stops"][cap[0]]

    # F3 is scored on a subset; the prose says so and the split is asserted.
    gt = cells["claude-opus-5"]["total_checks"]["grounding"]
    f["tax_f3_required"], f["tax_f3_diagnostic"] = gt["required"], gt["diagnostic"]
    f["tax_f3_asset_checks"] = gt["required"] + gt["diagnostic"]

    f["tax_worst_pass"] = min(c["pass_pct"] for c in cells.values())
    f["tax_f6_n"] = cells["claude-opus-5"]["total_checks"]["window"]["required"]
    tot = {m: sum(v["required"] for v in cells[m]["total_checks"].values())
           for m in order}
    f["tax_checks"] = tot["claude-opus-5"]
    f["tax_checks_pro"] = tot["gemini-3.1-pro"]
    return f


# --------------------------------------------------------------- sec 6.5
def efficiency_facts():
    """reasoning_tokens.json + failure_taxonomy.json -- sec:efficiency.

    Same route as taxonomy_facts: through src/make_body_figures.py, whose
    tokens() checks the token file's pass rates against corpus.py at every
    effort level before handing it over.
    """
    sys.path.insert(0, os.path.dirname(_paper_file(os.path.join(
        "src", "make_body_figures.py"))))
    import make_body_figures as mbf              # noqa: E402

    _, cells, order = mbf.taxonomy()
    doc = mbf.tokens()
    tok = {m: doc[m]["high"]["per_task"] for m in order}
    step = {m: cells[m]["step_mean"] for m in order}
    per = {m: tok[m] / step[m] for m in order}
    f = {}

    f["eff_steps_min"] = round(min(step.values()), 1)
    f["eff_steps_max"] = round(max(step.values()), 1)
    f["eff_steps_ratio"] = round(max(step.values()) / min(step.values()), 2)
    f["eff_tokstep_min"] = round(min(per.values()))
    f["eff_tokstep_max"] = round(max(per.values()))
    f["eff_tokstep_min_model"] = min(per, key=per.get)
    f["eff_tokstep_max_model"] = max(per, key=per.get)
    f["eff_tokstep_ratio"] = round(max(per.values()) / min(per.values()))

    # The factorisation the paragraph turns on. Asserted as an identity too,
    # because "these three numbers multiply" is the sentence's whole claim.
    a, z = "claude-opus-5", "gemini-3.5-flash"
    f["eff_pair_steps"] = round(step[z] / step[a], 2)
    f["eff_pair_tokstep"] = round(per[z] / per[a], 2)
    f["eff_pair_tokens"] = round(tok[z] / tok[a], 2)
    f["eff_pair_factors"] = abs((step[z] / step[a]) * (per[z] / per[a])
                                - tok[z] / tok[a]) < 1e-9

    # DEFINITIONAL IDENTITY, guarded so it cannot be rediscovered as a
    # finding: a round's stop reason is `none` exactly when the round passed.
    # Only gemini-3.1-pro breaks it, on its five provider-error rounds.
    f["eff_clean_is_pass"] = sum(cells[m]["stops"].get("none", 0)
                                 == cells[m]["passed"] for m in order)

    # The non-circular quantity: of a model's FAILED rounds, the share that
    # died on the step budget rather than finishing wrong.
    capf = {m: 100.0 * cells[m]["capped"] / (cells[m]["n"] - cells[m]["passed"])
            for m in order}
    # The fourth site that sliced order[:6] and called the result "the top
    # six". Same fix as the other three: the frontier is corpus.leading_group,
    # so this range is over the leading models and the keys say so.
    lead = corpus.leading_group(corpus.load_eval(corpus.PRIMARY_EFFORT,
                                                 keep_checks=False)[0])
    vals = [capf[m] for m in lead]
    f["eff_capfail_lead_min"], f["eff_capfail_lead_max"] = (round(min(vals), 1),
                                                            round(max(vals), 1))
    f["eff_capfail_lead_worst_is_max"] = max(capf, key=capf.get) not in lead
    # THE SENTENCE SPLITS THE LEADING GROUP AND SO MUST THIS BLOCK. The prose
    # quotes a 0.7-11.4 range "on six of the seven leading models, but 51.0%
    # on gemini-3.7-flash". eff_capfail_lead_max is the max over all seven and
    # is therefore 3.7-flash's own 51.0; the 11.4 in the same sentence is the
    # max over the other six and would ship unpinned without a key of its own.
    lead_worst = max(lead, key=capf.get)
    f["eff_capfail_lead_worst_model"] = lead_worst
    f["eff_capfail_lead_max_ex"] = round(
        max(capf[m] for m in lead if m != lead_worst), 1)
    for short, m in (("36f", "gemini-3.6-flash"), ("31p", "gemini-3.1-pro"),
                     ("35f", "gemini-3.5-flash")):
        f[f"eff_capfail_{short}"] = round(capf[m], 1)

    cap = [k for k in cells[z]["stops"] if k.startswith("reached max_steps=")]
    f["eff_step_budget"] = int(cap[0].split("=")[1])

    # The forward reference to sec:effort has to stay true.
    f["eff_35f_low_pass"] = doc[z]["low"]["pass_pct"]
    f["eff_35f_high_pass"] = doc[z]["high"]["pass_pct"]
    f["eff_35f_low_beats_high"] = (doc[z]["low"]["pass_pct"]
                                   > doc[z]["high"]["pass_pct"])
    return f


SHIPPED = {
    # ---- sec:main-results (6.1) ------------------------------------------
    # "satisfies 64.2% of an average round's required checks and still closes
    # only 25.2% of rounds", and the "39.0-point Competence-Resolution Gap"
    # that both the contribution bullet and Finding 1 quote as the difference.
    "main_top_pass": 25.2, "main_top_score": 0.642, "main_top_gap": 39.0,

    # ---- sec:validity (5) + sec:app-validity (C.3) + tab:validity caption --
    # "Of the 3,284 scored checks, 34.9% (1,147) are deterministic programmatic
    # conditions and 65.1% (2,137) are scoped semantic matches". The two
    # booleans are the arithmetic: an earlier draft quoted 1,593 deterministic
    # checks, which neither exhausted the denominator nor summed to 100%.
    "val_scored": 3284, "val_semantic": 2137, "val_det": 1147,
    "val_semantic_pct": 65.1, "val_det_pct": 34.9,
    "val_buckets_exhaust": True, "val_shares_sum": 100.0,
    # "the 476 diagnostic checks carry no score ... pooling them would both
    # inflate the deterministic denominator to 1,623"
    "val_diagnostic": 476, "val_det_pooled": 1623,
    # "97.5% of the 19,419 recorded verdicts are unanimous", "2.5% (479) are
    # split", "the stored pass/fail equals the majority vote on all 19,419".
    #
    # 97.5 + 2.5 = 100.0 ONLY AS PRINTED. The exact shares are 97.53 and 2.47,
    # which is why both precisions are pinned; do not "correct" either
    # rendering to make the printed pair sum by hand.
    "val_verdicts": 19419, "val_unanimous": 18940, "val_split": 479,
    "val_unanimous_pct": 97.53, "val_split_pct": 2.47,
    "val_unanimous_1dp": 97.5, "val_split_1dp": 2.5,
    "val_verdicts_partition": True, "val_majority_is_stored": True,
    # "the split rate varies little across models (1.4% for gemini-3.7-flash to
    # 3.4% for gemini-3.1-pro)" -- the models are part of the claim.
    "val_split_lo": 1.4, "val_split_lo_model": "gemini-3.7-flash",
    "val_split_hi": 3.4, "val_split_hi_model": "gemini-3.1-pro",
    # "moves task success by at most 2.3 points" (5, and the caption) against
    # "at most 2.33 points" (C.3). Same bound, two precisions, both shipped.
    "val_flip_swing": 2.33, "val_flip_swing_1dp": 2.3, "val_flip_rank": 1,
    # The adversarial flip: "claude-opus-5 still finishes second" (C.3) and
    # "leaves the top model within one rank of first" (5), plus "the largest
    # displacement anywhere is three positions".
    "val_adv_model": "claude-opus-5", "val_adv_rank": 2,
    "val_adv_worst_model": "claude-sonnet-5", "val_adv_worst_move": 3,
    # "rho = +0.90 over the ten API models on the judged denominator,
    # leave-one-out +0.87 to +0.93"; "rho = +0.02 across the ten API models";
    # "the choice of denominator itself re-ranks 7 of 10 models".
    "val_rho_n": 10, "val_rho_judged": 0.9,
    "val_rho_loo_lo": 0.87, "val_rho_loo_hi": 0.93,
    "val_rho_asset_content": 0.02, "val_denom_reranks": 7,
    # "pass rates decline monotonically for 13 of the 14 models; the exception
    # is deepseek-v4-flash, whose mid-band rate (11.2%) exceeds its easy-band
    # rate (8.5%)"
    "main_pass_mono": 13,
    # NO LONGER QUOTED. The body used to say "at high effort all twelve models
    # score higher on mid than on easy"; the fourteen-model rewrite dropped the
    # sentence. All fourteen still invert, so it could be restored as written,
    # and the key stays as a guard. It is deliberately SEPARATE from the two
    # monotone counts below, which are a weaker fact -- that no model's score
    # descends easy>mid>hard -- because a score can fail to descend by breaking
    # at the hard end instead.
    "main_score_inv": 14,
    "main_score_mono_macro": 0, "main_score_mono_micro": 0,
    # "4.19 required checks in easy, 5.48 in mid, 7.63 in hard"
    "main_req_easy": 4.19, "main_req_mid": 5.48, "main_req_hard": 7.63,
    # "has a 95% cluster bootstrap interval strictly above zero for 13 of the
    # 14 models, ranging from +5.9 points on deepseek-v4-flash to +20.3 on
    # claude-opus-4.8; only gemini-3.5-flash (+1.6) is not separated from zero"
    "main_separated": 13,
    "main_delta_min": 5.9, "main_delta_min_model": "deepseek-v4-flash",
    "main_delta_max": 20.3, "main_delta_max_model": "claude-opus-4.8",
    # Ranked on the interval, not on the point estimate, which is why this is
    # not main_delta_min even though the two name the same model.
    #
    # The sentence this used to transcribe ("cleared zero at high by 0.6
    # points of margin ... the weakest separation in Table~\ref{tab:main}") is
    # no longer in the paper -- Finding 3 was reworded to quote point
    # estimates (+5.9 / +20.3 / +1.6) and ranges instead. What the pin now
    # guards is the tab:main-high CELL and, through it, the "13 of the 14"
    # count that sentence still makes: this is the interval that comes closest
    # to zero without touching it, so if a future change pushes it under, the
    # claim silently becomes 12 of 14 and this fires first. It is the thinnest
    # margin in the table and it is thin -- 0.5 points at 10,000 draws, having
    # printed 0.6 at 4,000. Do not treat movement here as noise-free.
    "main_sep_min_ci_lo": 0.5, "main_sep_min_model": "gemini-3.1-pro",
    # The floor model's easy/hard split and the one interval in tab:main-high
    # that spans zero -- the "only gemini-3.5-flash is not separated from
    # zero" half of Finding 3. The prose no longer spells the endpoints (it
    # quotes the +1.6 point estimate), so the interval is guarded here; the
    # whole-corpus 3.0% is the "against 3.0\% at high" half of the low-effort
    # contrast further down the section.
    "main_floor_easy": 3.8, "main_floor_hard": 2.1, "main_floor_delta": 1.6,
    "main_floor_ci_lo": -0.9, "main_floor_ci_hi": 4.2,
    "main_floor_pass": 3.0,
    # "three of them contain no hard rounds at all and contribute 69 easy
    # ones, while multi_device_pairing alone supplies 21 of the 140 hard
    # rounds against 4 easy ... overstates by up to 5.8 points ... the
    # seventeen available within-seed contrasts ... n = 20"
    "main_nohard_seeds": 3, "main_nohard_easy": 69,
    "main_heavy_seed": "multi_device_pairing", "main_heavy_hard": 21,
    "main_heavy_total": 140, "main_heavy_easy": 4,
    "main_pool_max": 5.8, "main_both_bands": 17, "main_seeds": 20,
    # tab:main caption: "The five provider-error episodes are scored as
    # failures". Read bands_medium_errors and bands_low_errors below with it:
    # 5 / 2 / 0, three separate claims and not one copied across.
    "main_errors": 5,

    # ---- sec:main-results, Tables tab:main-medium and tab:main-low -------
    # The paragraph that reads the two effort repeats. Only what it and the
    # two captions actually say is pinned; the medium-effort floor row is
    # parsed but not shipped, because the contrast the prose draws is low
    # against high and shipping a number no sentence uses would make this
    # dict stop meaning "the paper asserts this".
    #
    # NO LONGER QUOTED. The body used to say "pass rate falls from easy to mid
    # to hard for 9 of the 9 models at every level"; the appendix now states
    # only the two tables' scope. Kept as guards -- and note both counts below
    # are over the TEN API models, because the medium and low tables do not
    # carry the four open-weight rows. The high-effort third of the old
    # sentence is main_pass_mono above, over the fourteen-model panel.
    "bands_medium_pass_mono": 10, "bands_low_pass_mono": 10,
    # NO LONGER QUOTED: "seven of the nine invert at medium effort and six at
    # low". Over the ten API models it reads eight and six.
    "bands_medium_score_inv": 8, "bands_low_score_inv": 6,
    # NO LONGER QUOTED: "seven of the eight models whose interval cleared zero
    # at high still clear it at both lower settings. The eighth,
    # gemini-3.1-pro". Over the ten API models it is eight of nine and the one
    # that drops out is still gemini-3.1-pro. The lost set stays a list rather
    # than a count so that both halves are guarded: that there is exactly one,
    # and which one it is.
    "bands_medium_kept": 8, "bands_low_kept": 8,
    "bands_medium_lost": ["gemini-3.1-pro"],
    "bands_low_lost": ["gemini-3.1-pro"],
    # gemini-3.5-flash passes 13.5% at low against 3.0% at high, and off the
    # floor its easy-hard gap is +12.6 points ([+7.3, +17.9]) -- the same
    # model whose HIGH-effort gap is the one interval spanning zero above.
    # That pair is the cleanest within-model statement the corpus supports on
    # effort, and it is currently made only by tab:main-low sitting next to
    # tab:main-high; see the effort-inversion section.
    "bands_low_floor_pass": 13.5, "bands_low_floor_delta": 12.6,
    "bands_low_floor_ci_lo": 7.3, "bands_low_floor_ci_hi": 17.9,
    # Captions: "Two provider-error episodes are scored as failures" and "No
    # provider errors occurred at this level". The zero is a caption CLAIM,
    # not a silence, so it is pinned like any other number.
    "bands_medium_errors": 2, "bands_low_errors": 0,

    # ---- tab:main caption, the zero-bound cells ---------------------------
    # "two band cells sit close enough to zero that a +-1.96 SE reconstruction
    # would run below it -- gemini-3.5-flash Hard at High (3/140) and at
    # Medium (2/140) -- and their resample distributions are right-skewed
    # against that bound, with percentile intervals [0.0, 5.0] and [0.0, 3.6]".
    #
    # WHY THE CAPTION SAYS THIS AT ALL. tab:main prints a standard error per
    # cell, and 2.1 +- 1.2 invites the reader's habitual +-1.96 SE, which lands
    # at -0.25 -- a negative pass rate. The band columns have no interval
    # anywhere else in the paper (all three effort tables bracket only the
    # `All` column), so before this sentence existed the reader had nothing to
    # reconstruct FROM except the SE, and reconstructing was wrong.
    #
    # THE COUNT IS THE LOAD-BEARING PIN. main_zerobound_n is what makes "two
    # band cells" true; the generator discovers the set rather than being told
    # it, so a regenerated corpus that presses a third cell to the bound fails
    # here instead of leaving the caption enumerating a subset. The lower
    # limits are not pinned as values because main_caption_facts asserts each
    # one is exactly 0.0 -- that is the sentence's actual claim.
    "main_zerobound_n": 2,
    "main_zerobound_model": "gemini-3.5-flash", "main_zerobound_band": "hard",
    "main_zerobound_high_k": 3, "main_zerobound_high_n": 140,
    "main_zerobound_high_hi": 5.0,
    "main_zerobound_med_k": 2, "main_zerobound_med_n": 140,
    "main_zerobound_med_hi": 3.6,

    # ---- sec:main-results, the cost paragraphs + Table~\ref{tab:cost} -----
    # "cost $11,430 across 76,585 model calls"
    "cost_total": 11736.96, "cost_calls": 87493,
    # tab:cost caption: "the Total row is formed from the unrounded aggregates
    # and therefore sits $2 and 1 M tokens below the sum of the rounded cells
    # above it". The column sums are what a reader adding the table by hand
    # gets, and they are asserted so the caption cannot quietly go stale when
    # a price change moves one row across a rounding boundary.
    "cost_col_sum_usd": 11739, "cost_col_sum_tok_m": 4348,
    "cost_round_gap_usd": 2, "cost_round_gap_tok_m": 1,
    # "Pass rate covers an 8.4x range ... cost per solved round covers 55x,
    # from $0.72 on gpt-5.6-luna to $39.54 on gemini-3.5-flash"
    "cost_pass_spread": 8.4, "cost_solve_spread": 55,
    "cost_cheapest": 0.72, "cost_cheapest_model": "gpt-5.6-luna",
    "cost_dearest": 39.54, "cost_dearest_model": "gemini-3.5-flash",
    # "clears 19.8% of rounds, 5.3 points behind the leader, so the leader's
    # extra accuracy costs 24x as much per round actually solved". The 31x in
    # the third observation is the same pair priced per round rather than per
    # solve, and both are rounded once from the unrounded ratio.
    "cost_cheapest_pass": 19.8, "cost_points_behind": 5.3,
    "cost_leader_ratio_solve": 24, "cost_leader_ratio_round": 31,
    # "8.8k reasoning tokens and the full 20-step budget on 89.2% of its
    # rounds"; the 16x token span is the third observation's.
    "cost_worst_reason_k": 8.8, "cost_worst_cap": 89.2,
    "cost_reason_span": 16.0,
    # "rises from 8.8% on claude-opus-5 to 90.5% on gemini-3.5-flash and ranks
    # against pass rate at rho = -0.933, where the unnormalised token count
    # gives -0.783". The token rho was quoted in the paper for months without
    # being pinned here; it is pinned now.
    "cost_share_lo": 8.8, "cost_share_lo_model": "claude-opus-5",
    "cost_share_hi": 90.5, "cost_share_hi_model": "gemini-3.5-flash",
    "cost_rho_share": -0.915, "cost_rho_tokens": -0.697,
    # "no single model carries it (leave-one-out $\rho$ stays within
    # $[-0.78, -0.58]$)". The weakest deletion is gpt-5.6-sol, NOT the
    # gemini-3.5-flash outlier one would guess, which is worth the pin.
    "cost_rho_loo_min": -0.783, "cost_rho_loo_max": -0.583,
    "cost_rho_loo_weakest_drop": "gpt-5.6-sol",
    # "Turning the knob up within a model does not generally reduce pass rate:
    # of the ten API models swept, five improve significantly, four are
    # indistinguishable, and one degrades ... gemini-3.5-flash falls from
    # 13.5% at low to 3.0% at high (-10.5 points, [-13.5, -7.7])".
    #
    # This is the qualifier on rho and it was missing. rho is BETWEEN models
    # and robust -- leave-one-out over the ten keeps it in [-0.78, -0.58], so
    # it is not an artifact of the model below -- but it cannot license the
    # word "leads" that Sections 1 and 6 used to carry, because the
    # within-model sweep points the other way for half the panel.
    "effort_helped": 5, "effort_hurt": 1, "effort_flat": 4, "effort_n": 10,
    "effort_inv_model": "gemini-3.5-flash",
    "effort_inv_low": 13.5, "effort_inv_high": 3.0,
    "effort_inv_delta": -10.5,
    "effort_inv_ci_lo": -13.5, "effort_inv_ci_hi": -7.7,

    # ---- sec:concentration (6.2) -----------------------------------------
    # "a co-failure lift above 1.0, from 1.05 to 1.49"
    "conc_lift_min": 1.05, "conc_lift_max": 1.49,
    # "that ceiling ranges from 1.39 to 2.70 down these rows"
    "conc_ceil_min": 1.39, "conc_ceil_max": 2.70,
    # "gemini-3.5-flash posts both the lowest lift in the field and, at 0.76
    # of its own maximum, the highest fraction of that maximum of any model"
    "conc_lowest_lift_model": "gemini-3.5-flash",
    "conc_max_liftceil": 0.76,
    "conc_max_liftceil_model": "gemini-3.5-flash",
    # "10 of the 12 models exhibit significant lift exceeding 1.0 [on easy] ...
    # every model lies within [0.99, 1.05] [on hard], and while claude-sonnet-5
    # and kimi-k3 do clear significance there, they do so at lifts of 1.04 and
    # 1.05".  Note the hard tercile is no longer empty of significant models as
    # it was over the nine API models -- the prose was rewritten from
    # "vanishes" to "negligible" when the panel grew, and conc_hard_sig is
    # pinned at 2 so that a silent return to 0 (or a rise to 3) is caught.
    "conc_easy_sig": 11, "conc_hard_sig": 3,
    "conc_hard_min": 0.99, "conc_hard_max": 1.06,
    # "9 of the 12 models diverge significantly from the peer average".  The
    # prose no longer breaks the 9 into its up/down halves, but both halves
    # stay pinned here: they are what the table's last two columns show, and a
    # drift that moved one model across the line while holding the total at 9
    # would otherwise pass unnoticed.  Six sit above (led by claude-opus-5 at
    # +0.174) and three below (glm-5.2-fp8, gemini-3.1-pro at -0.155, and
    # gemini-3.5-flash at -0.309); the peer mean is recomputed over twelve, so
    # these differ slightly from the nine-model values.
    "conc_peer_off": 11, "conc_peer_up": 7, "conc_peer_down": 4,
    "conc_peer_top_model": "claude-opus-5", "conc_peer_top": 0.188,
    "conc_peer_pro": -0.136, "conc_peer_flash35": -0.288,
    # "the models that cluster on the easy tercile are not the same models
    # that sit off their peer mean"
    "conc_easy_peer_sets_differ": True,
    # "no pair in this field combines comparable per-check ability with a
    # pass gap the data establishes"
    "conc_no_pair_qualifies": True,

    # ---- sec:capability-axes (6.3) ---------------------------------------
    # NO LONGER QUOTED: "Forty-three of the seventy-two model-category cells
    # sit at a different rank than the pooled order". The panel is fourteen
    # models by six categories now, so the denominator is 84 and the moves 51.
    # The pooled order is corpus.rank_order; against the provider-blocked
    # DISPLAY order the move count reads higher, because that scores the
    # pinned open-weight block as movement -- see the comment over `base` in
    # capability_axes_v8.py.
    "cap_rank_moves": 51, "cap_cells": 84,
    # "a permutation over seeds -- 4,000 random seed sets of the same size
    # drawn from the same 20"
    "cap_nperm": 4000, "cap_null_seeds": 20, "cap_nseeds": 20,
    # "would let the 30-round column borrow precision it does not have"
    "cap_min_rounds": 30,
    # "no category demonstrates statistically significant specialization
    # across all fourteen models (p >= 0.221) or the leading nine
    # (p >= 0.094)".
    #
    # THE SUB-PANEL IS NINE, NOT SIX. capability_axes_v8.py cuts with
    # corpus.leading_group over the FOURTEEN-model panel, which is a wider set
    # than the seven-model leading group sec:app-main defines over the ten API
    # models. Both are "the leading group" and they are not the same models;
    # the keys are suffixed _lead rather than _6 so that the name cannot go
    # stale the way "top six" did, but the suffix does not say which cut, so
    # read the generator before reusing these two anywhere else.
    "cap_ncat": 6, "cap_sig_all": 0, "cap_sig_lead": 0,
    # "the leading nine (p >= 0.094)" -- the smallest p over the sub-panel,
    # still on coordination.
    "cap_closest_cat": "coordination", "cap_closest_p_lead": 0.094,
    # "the bootstrap half-width on a single cell runs from 3.5 points on the
    # full 600 rounds to 15.0 points on main.sup., which rests on one seed".
    # The column names pin each half-width to the denominator it is claimed
    # for; without them the range could stay true while the sentence turned
    # false.
    "cap_boot_narrowest_col": "ALL", "cap_boot_widest_col": "mainsup",
    "cap_boot_all": 3.5, "cap_boot_min": 3.5, "cap_boot_max": 15.0,
    "cap_all_rounds": 600, "cap_min_seeds": 1,
    # appendix_report.tex: "4 of the 25 designed levers push every one of the
    # 14 models below its own baseline". Read out of the generated appendix,
    # which can move underneath the body, so both halves are pinned.
    "cap_levers_all_down": 4, "cap_levers_models": 14,
    # "Mean score per scenario across all sixty"
    "cap_heatmap_scenarios": 60,

    # ---- sec:taxonomy (6.4) ----------------------------------------------
    # Spreads are kept at the generator's precision even where the prose
    # rounds harder ("31, 60 and 65" for the three below). A drift of a tenth
    # trips this and sends me back to the sentence, which is the right way
    # round: a false alarm costs a look, a silent miss costs a wrong number.
    #
    # "separate models by 31 to 65 percentage points" over the ten API models.
    # The suffix is _all, not _9: this panel is corpus.rank_order and it went
    # nine -> ten, so a key called _spread9 would already be lying about its
    # own denominator. Same for _no35, which was _spread8 and is now nine.
    "tax_f1_spread_all": 31.1, "tax_f2_spread_all": 59.7,
    "tax_f4_spread_all": 64.8,
    # NO LONGER QUOTED as a pair: the body now gives the structural modes by
    # level rather than by spread. Both stay pinned; F5 moved 12.8 -> 14.0 when
    # the tenth API model joined, F6 did not move.
    "tax_f5_spread_all": 14.0, "tax_f6_spread_all": 13.3,
    # "show a high failure rate across all ten models (58%-72% and 83%-97%,
    # respectively)"; the F6 half rests on thirty checks per model.
    "tax_f5_min": 58.1, "tax_f5_max": 72.0,
    "tax_f6_min": 83.3, "tax_f6_max": 96.7, "tax_f6_n": 30,
    # "the one that scores 3.0%"
    "tax_worst_pass": 3.0,
    # "inside the leading group the spread narrows but does not close, running
    # from 5.0 points on F1 to 15.2 points on F4". The _no35 row is not quoted
    # any more and is kept as the intermediate step between the two.
    "tax_f1_spread_no35": 21.3, "tax_f2_spread_no35": 17.5,
    "tax_f4_spread_no35": 17.1,
    "tax_f1_spread_lead": 5.0, "tax_f2_spread_lead": 12.9,
    "tax_f4_spread_lead": 15.2,
    # THIS FLIPPED, and the body flipped with it. Under the old top-six cut all
    # three content spreads sat below F6's 13.3; under the leading-group cut
    # F4 is 15.2 and clears it, so tax_content_lead_under_f6 is False and only
    # one of the three stays under F5. The body no longer claims the content
    # modes are the narrow ones -- it claims the structural modes are set apart
    # by their LEVEL, not by any greater power to separate leading models.
    "tax_f5_spread_lead": 6.8, "tax_f6_spread_lead": 13.3,
    "tax_content_lead_under_f6": False, "tax_content_lead_under_f5": 1,
    # "claude-sonnet-5 is worst in the field at 22.0% against gpt-5.6-sol's
    # 6.6%, while gemini-3.5-flash ... is fourth best at 9.2%"
    "tax_f3_max_model": "claude-sonnet-5", "tax_f3_max": 22.0,
    "tax_f3_min_model": "gpt-5.6-sol", "tax_f3_min": 6.6,
    "tax_f3_flash35": 9.2, "tax_f3_flash35_rank": 4,
    # NOT QUOTED, guarded: F3's extremes are both inside the leading group, so
    # F3 does not compress when the weak models are dropped the way the content
    # buckets do -- all three of its spreads are the same 15.3.
    "tax_f3_ends_in_lead": True,
    "tax_f3_spread_all": 15.3, "tax_f3_spread_no35": 15.3,
    "tax_f3_spread_lead": 15.3,
    # "claude-sonnet-5 takes the fewest steps in the field, 10.6 on average,
    # and opens nothing at all in 92 of its 600 rounds; gemini-3.5-flash takes
    # the most, 19.7 against a cap of 20, and reaches that cap in 535"
    "tax_sonnet_steps": 10.64, "tax_sonnet_fewest_steps": True,
    "tax_sonnet_zero_open": 92,
    "tax_flash35_steps": 19.7, "tax_flash35_most_steps": True,
    "tax_step_cap": 20, "tax_flash35_capped": 535,
    # "346 of the corpus's 822 asset-access checks are required and the
    # remaining 476 are diagnostic"
    "tax_f3_required": 346, "tax_f3_diagnostic": 476,
    "tax_f3_asset_checks": 822,
    # fig:taxonomy's caption: "the same 3,284 required checks, gemini-3.1-pro
    # 3,257 after five provider errors"
    "tax_checks": 3284, "tax_checks_pro": 3257,
    # Not quoted anywhere, guarded anyway: the leading group is a prefix of the
    # pass-rate ranking, so "the leading models" and "the top N" name one set.
    "tax_lead_is_prefix_of_rank": True,

    # ---- sec:efficiency (6.5) --------------------------------------------
    # "mean steps per round run from 10.6 to 19.7, a factor of 1.85 over the
    # whole field against a budget of twenty steps"
    "eff_steps_min": 10.6, "eff_steps_max": 19.7, "eff_steps_ratio": 1.85,
    "eff_step_budget": 20,
    # "from 50 reasoning tokens per step for claude-opus-5 to 448 for
    # gemini-3.5-flash" -- "disperses nine times over"
    "eff_tokstep_min": 50, "eff_tokstep_min_model": "claude-opus-5",
    "eff_tokstep_max": 448, "eff_tokstep_max_model": "gemini-3.5-flash",
    "eff_tokstep_ratio": 9,
    # "1.77 times the steps and 9.01 times the thinking per step give the
    # 15.96-fold difference in tokens per task" -- and they must multiply,
    # which is the claim "the factorisation is exact".
    "eff_pair_steps": 1.77, "eff_pair_tokstep": 9.01,
    "eff_pair_tokens": 15.96, "eff_pair_factors": True,
    # "accounts for 0.7%-11.4% of failed rounds on six of the seven leading
    # models, but for 51.0% on gemini-3.7-flash, and for 30.3%, 28.8% and
    # 91.9% on gemini-3.6-flash, gemini-3.1-pro and gemini-3.5-flash". The
    # sentence splits the leading group, so _max is 3.7-flash's own 51.0 and
    # _max_ex is the 11.4 that bounds the other six.
    "eff_capfail_lead_min": 0.7, "eff_capfail_lead_max": 51.0,
    "eff_capfail_lead_max_ex": 11.4,
    "eff_capfail_lead_worst_model": "gemini-3.7-flash",
    "eff_capfail_36f": 30.3, "eff_capfail_31p": 28.8,
    "eff_capfail_35f": 91.9,
    # "and for 30.3%, 28.8% and 91.9%" reads as a step up from the leading
    # group only while no leading model is the worst in the field.
    "eff_capfail_lead_worst_is_max": True,
    # "sec:effort returns to the model that scores higher at low reasoning
    # effort than at high"
    "eff_35f_low_pass": 13.5, "eff_35f_high_pass": 3.0,
    "eff_35f_low_beats_high": True,
    # Not quoted: the identity that makes "18 clean stops, 18 passes" a
    # tautology rather than a result. Guarded so it stays visible.
    "eff_clean_is_pass": 9,

    # ------------------------------------------------------- sec 7.1
    # "That leaves 600 rounds per corpus in the full evaluation"
    "gen_tasks": 600, "gen_dead_a": 18, "gen_dead_b": 9,
    "gen_paired": 600, "gen_paired_min": 600,
    # 600 - 18 - 9 = 573 is only the right description while the dead sets
    # are disjoint. Not quoted; guarded because the sentence assumes it.
    "gen_dead_disjoint": True,
    # "All eighteen model-by-effort cells score lower on corpus B, and paired
    # exact McNemar tests call fifteen of the eighteen significant ... none
    # runs the other way, on either view."
    "gen_nmodels": 7, "gen_cells": 19, "gen_ntests": 19, "gen_alpha": 0.00263,
    "gen_a_higher_full": 18, "gen_a_higher_matched": 19,
    "gen_sig_full": 16, "gen_ns_full": 3, "gen_sig_b_full": 0,
    "gen_sig_matched": 14, "gen_ns_matched": 5, "gen_sig_b_matched": 0,
    # "The three that fall short are where one would expect them" -- two
    # gemini-3.5-flash floor cells and claude-opus-5 at high effort, and no
    # others. The roster of shortfalls is asserted, not just the count.
    "gen_ns_models": ["claude-opus-5", "gemini-3.5-flash"],
    "gen_floor_cells": 2, "gen_floor_max_pass": 3.2,
    "gen_opus5_gap": 6.0, "gen_opus5_p": 0.011,
    "gen_opus5_uncorrected_sig": True,
    # NO LONGER QUOTED: "Six of nine models appear: corpus B was never run on
    # the GPT-5.6 family". The roster is seven now -- gemini-3.7-flash joined,
    # the GPT-5.6 family still did not -- and the caption gives the count
    # rather than the exclusion. The zero is what keeps the roster readable.
    "gen_gpt56_in_roster": 0,
    # "Pairwise exact McNemar tests use a single family: all 45 pairs among the
    # ten API models, Bonferroni-corrected at alpha = 0.05/45 = 0.00111 ...
    # Twenty-one of the 45 pairs separate."
    #
    # These keys were called main_pairs36 / main_sep36 until the roster grew
    # from nine models to ten and the family went 36 -> 45. A key whose name
    # states a value is a key that lies the moment the value moves; do not put
    # the number back in the name.
    "main_pairs_all": 45, "main_sep_all": 21,
    #
    # THE TOP-SIX KEYS ARE GONE RATHER THAN RENAMED. main_sep_top6 and
    # main_sep_all_of_top6 pinned a correction the body now explicitly refuses
    # ("We deliberately do not correct within a ``top six'' instead, because
    # such a set is chosen after seeing the pass rates"). A pinned
    # main_sep_top6 quietly holding a leading-group number is exactly the
    # failure this file exists to prevent, so what is pinned instead is the
    # decomposition the body actually states, and it has to add up:
    # "Of the 3 pairs among the three trailing models, all 3 separate; of the
    # 21 leader-versus-trailer pairs, 16 separate; of the 21 pairs inside the
    # leading group, only 2 do." 3 + 16 + 2 = 21 = main_sep_all, and
    # 3 + 21 + 21 = 45 = main_pairs_all; the generator asserts both sums.
    "main_lead_best": "claude-opus-5", "main_lead_n": 7, "main_lead_of": 10,
    "main_lead_pairs": 21, "main_lead_sep": 2,
    "main_cross_sep": 16, "main_trail_pairs": 3, "main_trail_sep": 3,
    # "gpt-5.6-luna stays in at p = 0.00179 while gemini-3.6-flash falls out at
    # p = 0.00007, a 27-fold jump in p across a 1.3-point gap in pass rate."
    "main_lead_last_in": "gpt-5.6-luna", "main_lead_first_out":
        "gemini-3.6-flash",
    "main_lead_gap_ratio": 27, "main_lead_gap_points": 1.3,
    # "The frontier is therefore compressed but not perfectly flat, and we
    # state the exception rather than round it away" -- so the leading group is
    # NOT a clean cluster, and the False is the claim. "No pair spanning two
    # different providers separates within the leading group; the nearest such
    # miss is claude-opus-5 vs. gpt-5.6-terra at p = 0.00152."
    "main_lead_is_cluster": False, "main_lead_nearest_miss": 0.00152,
    # NO LONGER QUOTED: "Of the fifteen pairs among these six, N separate under
    # the all-pairs correction ... and those are exactly the leading cluster".
    # The sec 7.1 rewrite dropped that paragraph; these four are kept as
    # guards on the corpus-B roster, which is seven models and 21 pairs now.
    "gen_pairs_within": 21, "gen_pairs_sep": 12, "gen_pairs_unsep": 9,
    "gen_unsep_list": ["claude-opus-4.8 vs claude-opus-5",
                       "claude-opus-4.8 vs claude-sonnet-5",
                       "claude-opus-4.8 vs gemini-3.6-flash",
                       "claude-opus-4.8 vs gemini-3.7-flash",
                       "claude-opus-5 vs claude-sonnet-5",
                       "claude-opus-5 vs gemini-3.7-flash",
                       "claude-sonnet-5 vs gemini-3.6-flash",
                       "claude-sonnet-5 vs gemini-3.7-flash",
                       "gemini-3.6-flash vs gemini-3.7-flash"],
    "gen_swap_is_unresolved": True,
    # "2.51x against 2.02x on average, with 1.2% of its rounds under 1.5x
    # against 8.2%" and "does equalise slack, 2.22 against 2.27, but ...
    # widens the mean reference-length gap from +2.2 to +4.4 steps"
    "gen_slack_a": 2.02, "gen_slack_b": 2.51,
    "gen_tight_a": 8.2, "gen_tight_b": 1.2,
    "gen_matched_slack_a": 2.22, "gen_matched_slack_b": 2.27,
    "gen_gap_full": 2.2, "gen_gap_matched": 4.4,
    "gen_order_same_high": False, "gen_order_same_low": True,
    "gen_order_same_medium": False, "gen_swap_efforts": ["high", "medium"],
    "gen_corr_cells": 19, "gen_r_all": 0.912, "gen_rho_all": 0.968,
    "gen_rho_high": 0.964, "gen_rho_low": 1.0, "gen_rho_medium": 0.943,
    "gen_swap_models": ["claude-sonnet-5", "gemini-3.6-flash"],
    "gen_med_son_a": 17.00, "gen_med_36f_a": 20.17,
    "gen_med_son_b": 10.17, "gen_med_36f_b": 8.67,
    # "lands the two generators adjacent in mid-field, 0.51 ... and 0.54 ...,
    # between gemini-3.7-flash at 0.41 and claude-opus-4.8 at 0.57, neither of
    # which wrote either corpus", and the Ret% column of Figure 2(c).
    "gen_ret_36f": 0.51, "gen_ret_son": 0.54,
    "gen_ret_authors_adjacent": True,
    "gen_ret_below": "gemini-3.7-flash", "gen_ret_below_val": 0.41,
    "gen_ret_above": "claude-opus-4.8", "gen_ret_above_val": 0.57,

    # ---------------------------------------------------------- sec 7.2
    # "Over the thirty contrasts ... ten models by three pairs of levels -- at
    # a Bonferroni-corrected threshold of 0.05/30 = 0.00167 ... helps three
    # models ... leaves six indistinguishable ... hurts one".
    #
    # THE REPORT MARKDOWN SAYS FOUR, NOT THREE, AND BOTH ARE RIGHT. Its effort
    # table is uncorrected p < 0.05, which also admits gpt-5.6-terra at 0.049;
    # this family is Bonferroni-corrected over all thirty contrasts, which does
    # not. eff2_uncorrected_flips below pins terra as the only row that moves
    # between the two thresholds, so the discrepancy cannot grow silently.
    "eff2_ntests": 30, "eff2_alpha": 0.00167, "eff2_models": 10,
    "eff2_help": 3, "eff2_ns": 6, "eff2_hurt": 1,
    # The three helped, each as "N rounds recovered against M lost".
    "eff2_son_hi": 53, "eff2_son_lo": 11, "eff2_son_p": 1e-07,
    "eff2_31p_hi": 31, "eff2_31p_lo": 9, "eff2_31p_p": 0.00068,
    "eff2_sol_hi": 34, "eff2_sol_lo": 12, "eff2_sol_p": 0.0016,
    # "solved 63 rounds at low effort that it failed at high, and not one
    # round the other way". The zero is the sentence, so it is asserted as a
    # zero and not as "few".
    "eff2_35f_hi": 0, "eff2_35f_lo": 63, "eff2_35f_p": 2.2e-19,
    "eff2_35f_zero_recovered": True,
    # "Of the remaining contrasts exactly one clears the threshold, and it is
    # the same model again -- medium against low, p = 1.65e-15". The keys were
    # eff2_other18_* until the roster grew and the remainder went 18 -> 20; the
    # count is not in the name any more, for the usual reason.
    "eff2_other_sig": 1, "eff2_other_p": 1.65e-15,
    "eff2_other": ["gemini-3.5-flash medium vs low"],
    # "which across those low-to-high rows changes only gpt-5.6-terra".
    # DELIBERATELY NARROW: across all thirty contrasts six rows change, and
    # that count is pinned beside it so the sentence cannot be widened by
    # someone who only checked the panel the prose cites. This is also the one
    # row that makes the report Markdown say four improve where this file says
    # three; see the note over eff2_help.
    "eff2_uncorrected_flips": ["gpt-5.6-terra"], "eff2_terra_p": 0.049,
    "eff2_uncorrected_flips_all": 6,

    # "Mean steps rise with effort for all 10 of 10 models". Cap-hit does NOT
    # (7 of 10), which is why the claim is about steps only; the 7 is pinned so
    # the stronger sentence cannot be written by accident.
    "eff2_steps_mono": 10, "eff2_capn_mono": 7,
    # "steps rise from 16.22 to 19.70 against a budget of twenty, the share of
    # rounds ending at that budget rises from 38.7 to 89.2%, and the pass rate
    # falls from 13.5 to 3.0%". 19.70 is the file's 19.7 with the trailing
    # zero restored to match 16.22 -- the same number, not a re-rounding.
    "eff2_35f_steps_lo": 16.22, "eff2_35f_steps_hi": 19.7,
    "eff2_35f_capn_lo": 38.7, "eff2_35f_capn_hi": 89.2,
    "eff2_35f_pass_lo": 13.5, "eff2_35f_pass_hi": 3.0,
    # "gemini-3.1-pro ends 25.3% ... gemini-3.6-flash ends 24.7% and is an
    # exact tie, 34 rounds each way". These are capped/n over the full 600.
    # The effort panel's 25.0% and 24.2% are the SAME quantity over corpus A's
    # 582 winnable rounds and are not interchangeable with these.
    "eff2_31p_capn_hi": 25.3, "eff2_36f_capn_hi": 24.7,
    "eff2_36f_tie": True, "eff2_36f_each_way": 34,
    # "sec:efficiency put budget exhaustion at 91.9% of its failures" -- the
    # other denominator, for the same cell whose capped/n is 89.2%. Both are
    # asserted, and so is the fact that they are not equal.
    "eff2_35f_cap_of_failed": 91.9, "eff2_cap_denoms_differ": True,

    # "corpus B, which was generated against a budget of thirty ... Cap-hit
    # rises monotonically with effort in all six models scored on both"
    "eff2_b_budget": 30, "eff2_b_models": 6, "eff2_b_capmono": 6,
    # "on corpus B's winnable rounds (591 at low, 589 at high) its pass rate
    # falls from 4.6 to 2.4% while cap-hit rises from 49.2 to 79.6%, with 16
    # rounds going to low against 3 to high, p = 0.0044". The planning note
    # for this section predicted 4.5 -> 2.3 and 49.0 -> 78.7; none of those
    # four reproduce under any pool, and these measured values replaced them.
    "eff2_b_35f_n_lo": 600, "eff2_b_35f_n_hi": 600,
    "eff2_b_35f_pass_lo": 4.5, "eff2_b_35f_pass_hi": 2.3,
    "eff2_b_35f_cap_lo": 49.0, "eff2_b_35f_cap_hi": 78.7,
    "eff2_b_35f_lo_wins": 16, "eff2_b_35f_hi_wins": 3, "eff2_b_35f_p": 0.0044,
    # mcnemar_v8.py's b counts rounds HIGH won; the effort panel's b(lo>hi)
    # counts the opposite. The two agree on this cell once crossed, and the
    # crossing is asserted so a rename in either script trips here rather than
    # silently reversing a sentence.
    "eff2_bc_letters_crossed": True,
    # "as the single pre-specified replication it was run as, it clears 0.05
    # comfortably; as one of the twelve low-to-high contrasts in the same
    # panel it misses 0.05/12 = 0.0042 by a hair". The claim IS that these two
    # disagree, so both are pinned true.
    "eff2_b_panel_n": 13, "eff2_b_panel_alpha": 0.0038,
    "eff2_b_35f_clears_05": True, "eff2_b_35f_misses_panel": True,
    # "claude-opus-5, whose effort response is undetectable on corpus A
    # (p = 0.23), is significantly helped by effort on corpus B, 34 rounds to
    # high against 10 to low, p = 3.9e-04"
    "eff2_a_opus5_p": 0.23, "eff2_b_opus5_hi_wins": 34,
    "eff2_b_opus5_lo_wins": 10, "eff2_b_opus5_p": 0.00039,

    # ---- sec:budget-ablation (7.3) + Table~\ref{tab:ablation}(a) ----------
    # "the same 600 rounds at high effort with the cap at 30 and at 40, on
    # four models"; the blind arm's roster is three and is pinned separately.
    "ab_n_per_cell": 600, "ab_bud_models": 4,
    # "+1.8 on claude-opus-5 and +0.5 on claude-sonnet-5 ... a Bonferroni
    # threshold of 0.05/8 over the eight contrasts (p = 0.15 and p = 0.76)"
    "ab_opus5_delta": 1.8, "ab_sonnet5_delta": 0.5,
    "ab_budget_tests": 8, "ab_budget_alpha": 0.00625,
    "ab_opus5_p40": 0.15, "ab_sonnet5_p40": 0.76,
    "ab_opus5_sig40": False, "ab_sonnet5_sig40": False,
    # "+6.5 on gemini-3.6-flash (3.8e-06) and +10.0 on gemini-3.5-flash
    # (2.7e-17)"; two of the eight contrasts clear the threshold.
    "ab_36flash_delta": 6.5, "ab_36flash_p40": 3.8e-06,
    "ab_35flash_delta": 10.0, "ab_35flash_p40": 2.7e-17,
    "ab_36flash_sig40": True, "ab_35flash_sig40": True,
    "ab_budget_sig40": 2,
    # "the two the cap almost never bound, at 0.5 and 3.3% ... the two it
    # bound often, at 24.7 and 89.2%"
    "ab_opus5_cap20": 0.5, "ab_sonnet5_cap20": 3.3,
    "ab_36flash_cap20": 24.7, "ab_35flash_cap20": 89.2,
    # "It is a split and not a gradient". BOTH are pinned: the split holds,
    # the ordering does NOT, and an earlier draft claimed the ordering. If a
    # rerun ever makes ab_gain_tracks_cap true the sentence is understated and
    # should be rewritten -- but it must never be written as a gradient while
    # this is False.
    "ab_gain_tracks_cap": False, "ab_split_clean": True,
    # "narrows from 8.4x at b=20 to 2.1x at b=40 ... the best and worst models
    # unchanged and exactly one adjacent pair trading places". The 8.4x is the
    # same number sec:main-results quotes over all nine (cost_pass_spread),
    # which holds only while this roster contains both extremes; ablation.py
    # asserts that separately.
    "ab_spread20": 8.4, "ab_spread40": 2.1,
    "ab_rank_swaps": 1, "ab_rank_top_stable": True, "ab_rank_bot_stable": True,
    "ab_rank_swapped": "claude-sonnet-5,gemini-3.6-flash",
    # "gains 30 rounds and loses 19 ... 8.2% of the corpus ... to move the
    # aggregate by +1.8 points; claude-sonnet-5 churns 7.2% to move by +0.5"
    #
    # GROSS CHURN AND NET MOVEMENT ARE TWO NUMBERS AND A SHIPPED DRAFT USED
    # ONE FOR THE OTHER. It told the reader that a model "re-run against a cap
    # it never hits moves by about" three points, on the strength of the 8.2%
    # flip rate. The move is +1.8. Both are pinned; do not quote a churn rate
    # in pass-rate points.
    "ab_opus5_gain40": 30, "ab_opus5_loss40": 19,
    "ab_opus5_churn40": 8.2, "ab_sonnet5_churn40": 7.2,
    "ab_opus5_net40": 1.8, "ab_sonnet5_net40": 0.5,
    # "Not one of claude-opus-5's thirty gained rounds ran past the shipped
    # cap, and one of claude-sonnet-5's twenty-three did". Counted on `steps`,
    # not on cap_hit: 11 rounds across the arms finish with steps == budget but
    # an empty stop_reason and so are never flagged, making that flag an
    # undercount. This is the bound on the MECHANISM only -- the cap is also
    # announced in the prompt, so the arm is not a same-configuration re-run
    # and must not be written up as one.
    "ab_opus5_gainspent": 0, "ab_sonnet5_gainspent": 1,
    "ab_36flash_gainspent": 28, "ab_35flash_gainspent": 55,
    "ab_opus5_over20_b40": 40, "ab_sonnet5_over20_b40": 48,
    "ab_36flash_over20_b40": 221, "ab_35flash_over20_b40": 537,

    # "sec:main-results tests all 45 pairs among the ten API models at
    # alpha = 0.05/45 and separates 21 of them. Four of those involve only
    # models re-run here ... three survive at b=40 and one does not."
    # The 21 is also printed by mcnemar_v8.py; ablation.py recomputes the set
    # and asserts the count, so the two cannot drift apart. The keys were
    # ab_sep_n36 / ab_sep_alpha36 until the family went 36 -> 45.
    "ab_sep_pairs_all": 45, "ab_sep_n_all": 21, "ab_sep_alpha_all": 0.001111,
    "ab_sep_testable": 4, "ab_sep_survived": 3, "ab_sep_lost": 1,
    "ab_sep_gained": 1,
    # "claude-opus-5 against gemini-3.6-flash: 69/29 discordant rounds at
    # p = 6.6e-5 ... 65/53 at p = 0.31"
    "ab_sep_lost_pair": "claude-opus-5,gemini-3.6-flash",
    "ab_sep_lost_gl20": "69/29", "ab_sep_lost_gl40": "65/53",
    "ab_sep_lost_p20": 6.6e-05, "ab_sep_lost_p40": 0.31,
    # "That is not a thinner test. The discordant count rises, from 98 to 118;
    # what shrinks is the effect, from +6.7 points to +2.0." The boolean IS
    # the argument -- if a rerun ever makes it False the paragraph is wrong.
    "ab_sep_lost_disc20": 98, "ab_sep_lost_disc40": 118,
    "ab_sep_lost_power_up": True,
    "ab_sep_lost_eff20": 6.7, "ab_sep_lost_eff40": 2.0,
    # "sits above the threshold at b=20 (p = 0.0043 against a threshold of
    # 0.00139) and below it at b=40 (p = 0.00077)". Quoted as plain decimals
    # so the reader can compare all three by eye; _sig would render the first
    # as "0.00" and the second in scientific notation.
    "ab_sep_gained_pair": "claude-opus-5,claude-sonnet-5",
    "ab_sep_gained_p20": "0.0043", "ab_sep_gained_p40": "0.00077",
    "ab_sep_gained_thresh": '0.00111',
    # "both reversals hold at every correction this paper uses" -- and which
    # corrections those are is itself pinned, because "every correction this
    # paper uses" is a claim about the set, not just about the boolean. 45 is
    # the all-pairs family at the ten-model roster (36 at nine).
    "ab_sep_corrections": "6,15,45",
    "ab_sep_robust": True,
    # "all three of that model's testable separations survive ... the largest
    # of the three p-values is 3.1e-7". THE MAXIMUM, NOT A BOUND: 3.11e-07
    # rounds down to 3.1e-07, so "p <= 3.1e-7" would be false as printed --
    # the same trap as ab_mod_ung_pmax below.
    "ab_sep_worst_survive": 3,
    "ab_sep_worst_maxp40": "3.1\\times10^{-7}",
    # "still ends 44.0% of its rounds at the cap there, so its 13.0% remains a
    # truncated number"
    "ab_35flash_cap40": 44.0, "ab_35flash_40": 13.0,
    # The sec:effort tie-in: "13.0% at b=40 against the 13.5% the same model
    # reaches at low effort under the original cap ... essentially all of the
    # 10.5-point inversion ... 32.92 mean steps to reach where low effort got
    # in 16.22 ... 44.0% against 38.7%". The low-effort trio is recomputed by
    # ablation.py from corpus.py and must equal the eff2_35f_* values above,
    # which reach the paper by a different route (the token file).
    "ab_35f_b40_pass": 13.0, "ab_35f_low_pass": 13.5,
    "ab_35f_inversion": 10.5,
    "ab_35f_b40_steps": 32.92, "ab_35f_low_steps": 16.22,
    "ab_35f_b40_cap": 44.0, "ab_35f_low_cap": 38.7,
    # Caption: "One gemini-3.5-flash round at b=40 ended in a provider timeout
    # and is retained as a failure."
    "ab_err_model": "gemini-3.5-flash", "ab_err_n": 1,

    # ---- sec:modality-ablation (7.4) + Table~\ref{tab:ablation}(b) --------
    # "346 of the 600 rounds carry a required check that fires on asset
    # access ... the other 254". The strata must partition the corpus.
    "ab_all_n": 600, "ab_gated_n": 346, "ab_ungated_n": 254,
    "ab_strata_partition": True, "ab_mod_models": 3,
    # "claude-opus-5 falls from 24.8 to 2.8%, claude-sonnet-5 from 24.0 to
    # 3.1%, and gemini-3.6-flash from 19.3 to 2.4%"
    "ab_mod_opus5_ung_sight": 24.8, "ab_mod_opus5_ung_blind": 2.8,
    "ab_mod_sonnet5_ung_sight": 24.0, "ab_mod_sonnet5_ung_blind": 3.1,
    "ab_mod_36flash_ung_sight": 19.3, "ab_mod_36flash_ung_blind": 2.4,
    # "keeps between 0.111 and 0.131 of its sighted pass rate ... a Bonferroni
    # threshold of 0.05/6, the largest of the three p-values being 2.6e-12,
    # with 60, 54 and 44 rounds going to the sighted run against 4, 1 and 1"
    "ab_mod_retained_lo": 0.111, "ab_mod_retained_hi": 0.131,
    "ab_mod_tests": 6, "ab_mod_sig_ungated": 3,
    # QUOTED AS THE MAXIMUM, NOT AS A BOUND: 2.61e-12 rounds down to 2.6e-12,
    # so "p <= 2.6e-12" would be false as printed.
    "ab_mod_ung_pmax": 2.6e-12,
    "ab_mod_opus5_ung_gain": 60, "ab_mod_opus5_ung_loss": 4,
    "ab_mod_sonnet5_ung_gain": 54, "ab_mod_sonnet5_ung_loss": 1,
    "ab_mod_36flash_ung_gain": 44, "ab_mod_36flash_ung_loss": 1,
    # "blind still scores 1.7-5.2% there"
    "ab_mod_gated_blind_lo": 1.7, "ab_mod_gated_blind_hi": 5.2,
    # "it solved 6, 4 and 1 rounds that the sighted run failed, so the two are
    # not nested"
    "ab_mod_opus5_all_blindonly": 6, "ab_mod_sonnet5_all_blindonly": 4,
    "ab_mod_36flash_all_blindonly": 1, "ab_mod_any_blind_only": 3,

    # ---- Figure~\ref{fig:ablation} (src/figures/body-ablation.svg) --------
    # The one exhibit in this paper with no generator: the SVG is hand-drawn,
    # so its bar widths and dot positions were only ever checked by eye, and
    # three of the panel (c) retention percentages and one panel (b) drop were
    # wrong when this guard was written. analysis/figure_facts.py reads the
    # shipped SVG back and re-derives every quantity it draws -- geometry
    # included, because two of panel (a)'s three series carry no text label at
    # all -- from ablation.py and generator_ablation.py.
    #
    # Only the tallies ship here; the 155 individual comparisons would drown
    # this file's print output. The mismatch list must stay empty, and the
    # counts are pinned so that a silently deleted check fails too.
    "fig2_mismatches": (), "fig2_checked": 155,
    "fig2_a_curves": 4, "fig2_b_rows": 6, "fig2_c_rows": 7,

    # ---- Section 3, delegated to analysis/section3_facts.py --------------
    # Its own SHIPPED dict holds the 98 values; only the verdict crosses over,
    # so that the single command anyone actually runs covers Sections 3-7 and
    # Figure 2 rather than 4-7. s3_checked is pinned at the full 101: 26 of
    # them need the harness source tree, and if it is unreachable they do not
    # compute, the tally drops to 75, and this fails with the fix in the
    # message instead of passing on a subset.
    "s3_mismatches": (), "s3_unverified": (), "s3_checked": 101,

    # ---- Judge ablation, delegated to analysis/judge_ablation.py ----------
    # The only guard on Table~\ref{tab:judge-app}, which is hand-written inline
    # in main.tex and therefore invisible to src/verify_table_bodies.py. Its
    # source of truth is the committed artifact src/report/judge_ablation.json
    # rather than a generator's stdout -- the generator is a multi-hour API
    # sweep -- so it is delegated rather than merged, on the same grounds as
    # Section 3. Only the verdict crosses over; the 61 comparisons stay there.
    "judge_mismatches": (), "judge_checked": 61,

    # ---- Appendix B exhibits, delegated to analysis/round_example_facts.py -
    # The only claims in the paper that are about PIXELS -- quoted strings read
    # off rendered assets, plus the counts their captions state. As with
    # Figure 2, the 148 individual comparisons stay in that module and only the
    # tally crosses over; the tally is pinned so a deleted assertion fails here
    # rather than passing quietly on a subset.
    "round_example_mismatches": (), "round_example_checked": 152,
    # sec:app-worked-round, Figure A and tab:round-checks
    "round_visible_assets": 7, "round_n_checks": 8, "round_gold_steps": 15,
    # Figure C. The two text-surface numbers are the exclusivity evidence the
    # prose quotes ("25,526 characters in all", "occurs fourteen times"); they
    # are the whole reason the round is not solvable from text, so a corpus
    # regen that moved either must not slip through.
    "cache_round_visible_assets": 8, "cache_round_n_checks": 5,
    "cache_round_gold_steps": 10,
    "cache_text_surface_chars": 25526, "cache_decoy_in_text_count": 14,
    # sec:app-assets, Figure B
    "corpus_asset_count": 1568, "corpus_styles_defined": 18,
    "corpus_styles_sampleable": 16, "corpus_styles_realized": 14,
}

# --------------------------------------------------------------- sec 7.1
def generator_facts():
    """generator_ablation.py -- everything sec:generator-ablation states.

    Parses the generator's stdout, per this file's design note. The two
    derived quantities here -- the retention ratio and the per-effort rank
    orderings -- are deterministic arithmetic over parsed pass rates, not
    re-derivations of a stochastic statistic, so they carry no risk of
    disagreeing with a second draw the way a re-run bootstrap would.
    """
    log = run("generator_ablation.py")
    f = {}

    m = re.search(r"dead rounds excluded: A=(\d+)\s+B=(\d+)", log)
    f["gen_dead_a"], f["gen_dead_b"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"tasks in common: (\d+)", log)
    f["gen_tasks"] = int(m.group(1))
    m = re.search(r"Aligned rounds per corpus\s+&\s+\\multicolumn\{2\}\{c\}\{(\d+)\}", log)
    f["gen_paired"] = int(m.group(1))
    f["gen_paired_min"] = int(m.group(1))
    f["gen_dead_disjoint"] = True

    m = re.search(r"Bonferroni alpha = 0\.05/(\d+) = ([\d.]+)", log)
    f["gen_ntests"], f["gen_alpha"] = int(m.group(1)), float(m.group(2))
    m = re.search(r"models present in both corpora \((\d+)\): (.+)", log)
    f["gen_nmodels"] = int(m.group(1))
    roster = [x.strip() for x in m.group(2).split(",")]
    # "a model corpus B never covered" -- the sentence about gpt-5.6-sol only
    # holds while the GPT-5.6 family is absent from this ablation entirely.
    f["gen_gpt56_in_roster"] = sum(1 for x in roster if x.startswith("gpt-5.6"))

    def block(name, upto):
        chunk = log.split(name)[1].split(upto)[0]
        rows = []
        for ln in chunk.splitlines():
            m = re.match(r"\s*(\S+)\s+(high|medium|low)\s+(\d+)\s+([\d.]+)\s+"
                         r"([\d.]+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(.+?)\s*$", ln)
            if m:
                rows.append(dict(model=m.group(1), effort=m.group(2),
                                 a=float(m.group(4)), b=float(m.group(5)),
                                 p=float(m.group(8)), verdict=m.group(9)))
        return rows

    full = block("FULL paired set", "SLACK-MATCHED")
    matched = block("SLACK-MATCHED subset", "% ---- LaTeX body")
    assert len(full) == len(matched) == f["gen_ntests"], (len(full),
                                                          len(matched))
    f["gen_cells"] = len(full)
    # Per-model high-effort pass rates. The prose never quotes an individual
    # cell -- it quotes tallies and rho -- so these were never exported, and
    # Table F2 was their only guard. Figure 2(c) plots the seven high-effort
    # pairs directly, dot by dot, and its Ret% column is a ratio of exactly
    # these two numbers; three of those ratios shipped wrong because they were
    # taken from the already-rounded one-decimal labels instead. figure_facts
    # divides the two-decimal values below, which is what fixed them.
    for r in full:
        if r["effort"] == "high":
            f[f"gen_high_a_{r['model']}"] = r["a"]
            f[f"gen_high_b_{r['model']}"] = r["b"]
    # Panel (c) of tab:ablation as the generator PRINTS it, to one decimal.
    # Re-rounding the two-decimal diagnostic column above does not reproduce
    # this row and must never be used to: the column shows 25.65 for
    # claude-opus-5, which rounds to 25.6, while the underlying 147/573 is
    # 25.654 and prints as 25.7. Figure 2(c) draws these same one-decimal
    # labels, so both the figure and Table F2 are now pinned to this one row
    # and are provably the same numbers.
    body = log.split("corpora outcome table")[1]
    cols = re.findall(r"\\textbf\{([^}]+)\}",
                      re.search(r"\\textbf\{Corpus\}((?:\s*&\s*"
                                r"\\textbf\{[^}]+\})+)", body).group(1))
    for rail in ("A", "B"):
        row = re.search(rail + r" \(\\texttt\{[^}]+\}\)((?: & [\d.]+)+)", body)
        vals = [float(v) for v in re.findall(r"[\d.]+", row.group(1))]
        assert len(vals) == len(cols) == 7, (len(vals), len(cols))
        for c, v in zip(cols, vals):
            f[f"gen_tabc_{rail.lower()}_{c}"] = v

    for tag, rows in (("full", full), ("matched", matched)):
        f[f"gen_a_higher_{tag}"] = sum(1 for r in rows if r["a"] > r["b"])
        f[f"gen_sig_{tag}"] = sum(1 for r in rows
                                  if r["verdict"] == "corpus A easier")
        f[f"gen_ns_{tag}"] = sum(1 for r in rows if r["verdict"] == "n.s.")
        # The load-bearing asymmetry: nothing significant the other way.
        f[f"gen_sig_b_{tag}"] = sum(1 for r in rows
                                    if "B easier" in r["verdict"])

    # The three shortfalls, named rather than counted, so that a change in
    # WHICH cells miss the threshold trips even if the tally survives.
    ns = {(r["model"], r["effort"]): r for r in full if r["verdict"] == "n.s."}
    f["gen_ns_models"] = sorted({m for m, _ in ns})
    floor = [r for (mo, _), r in ns.items() if mo == "gemini-3.5-flash"]
    f["gen_floor_cells"] = len(floor)
    f["gen_floor_max_pass"] = round(max(max(r["a"], r["b"]) for r in floor), 1)
    o5 = ns[("claude-opus-5", "high")]
    f["gen_opus5_gap"] = round(o5["a"] - o5["b"], 1)
    f["gen_opus5_p"] = float(f"{o5['p']:.2g}")
    # "significant on its own": it clears 0.05 and misses 0.05/18.
    f["gen_opus5_uncorrected_sig"] = o5["p"] < 0.05 and o5["p"] > f["gen_alpha"]

    # These two read the slack row's LaTeX, so they track its shape. The row
    # used to be "mean / min" with two \times cells per corpus; the min was
    # dropped in the body structure pass, which broke the old slack_a pattern
    # (it matched on the literal "mean / min") and left the old slack_b
    # pattern matching the right number for the wrong reason. Both are now
    # anchored on the row label and cell order instead.
    for key, pat in (("slack_a", r"Slack .*?mean\s+&\s+\$([\d.]+)\\times\$"),
                     ("slack_b",
                      r"Slack .*?mean\s+&\s+\$[\d.]+\\times\$ & \$([\d.]+)"),
                     ("tight_a", r"slack \$< 1\.5.*?&\s+([\d.]+)\\%"),
                     ("tight_b", r"slack \$< 1\.5.*?\) & ([\d.]+)\\%")):
        f["gen_" + key] = float(re.search(pat, log).group(1))
    m = re.search(r"MATCHED \|d\|<=0\.30\s+n=\s*\d+\s+\| A gold\s+[\d.]+\s+"
                  r"slack ([\d.]+).*?\| B gold\s+[\d.]+\s+slack ([\d.]+)", log)
    f["gen_matched_slack_a"] = round(float(m.group(1)), 2)
    f["gen_matched_slack_b"] = round(float(m.group(2)), 2)
    m = re.search(r"gold-length gap \(B - A\):\s+FULL \+([\d.]+) steps\s+"
                  r"MATCHED \+([\d.]+) steps", log)
    f["gen_gap_full"] = round(float(m.group(1)), 1)
    f["gen_gap_matched"] = round(float(m.group(2)), 1)

    # Rank agreement, computed at EVERY effort. Reading the high column alone
    # says the ordering is preserved; medium swaps an adjacent pair.
    swaps = {}
    for eff in ("high", "medium", "low"):
        sub = [r for r in full if r["effort"] == eff]
        ra = [r["model"] for r in sorted(sub, key=lambda r: -r["a"])]
        rb = [r["model"] for r in sorted(sub, key=lambda r: -r["b"])]
        f[f"gen_order_same_{eff}"] = ra == rb
        if ra != rb:
            swaps[eff] = sorted({m for i, m in enumerate(ra) if m != rb[i]})
    f["gen_swap_efforts"] = sorted(swaps)
    f["gen_swap_models"] = swaps.get("medium", [])

    # Cross-corpus agreement, read off the generator's own caption facts.
    # Pinned because the body, the Figure 2 caption and the panel itself each
    # quote one of these numbers, and a shipped draft put "r = 0.94" inside
    # Figure 2(c) -- which was neither the Pearson nor the Spearman value at
    # any effort. It entered on 2026-08-13 with the smaller roster and survived
    # the 14-model rebuild unrecomputed. The claim beside it is about ORDERING,
    # so what it may quote is Spearman.
    #
    # Note the two Spearman values are NOT interchangeable and the paper now
    # says which is which: the panel draws the seven high-effort pairs, whose
    # rho is gen_rho_high = 1.00, while the body's 0.97 is gen_rho_all over all
    # 19 model-by-effort cells. Quoting 0.97 on the panel would be describing a
    # population the panel does not show.
    m = re.search(r"cross-corpus agreement over the (\d+) paired cells: "
                  r"Pearson r = ([\d.]+), Spearman rho = ([\d.]+)", log)
    f["gen_corr_cells"] = int(m.group(1))
    f["gen_r_all"], f["gen_rho_all"] = float(m.group(2)), float(m.group(3))
    for eff in ("high", "medium", "low"):
        m = re.search(r"cross-corpus agreement at " + eff +
                      r" effort \(\d+ models\): Pearson r = [\d.]+, "
                      r"Spearman rho = ([\d.]+)", log)
        f[f"gen_rho_{eff}"] = float(m.group(1))
    # rho == 1 and "no swap at this effort" are two spellings of one fact. If
    # they ever disagree, one of the two is being parsed wrong.
    assert all((f[f"gen_rho_{e}"] == 1.0) == f[f"gen_order_same_{e}"]
               for e in ("high", "medium", "low")), "rho vs rank-order mismatch"
    # Quoted at the generator's own two decimals, NOT rounded to one. The
    # sonnet-5 figure is 17.45, a rounding knife-edge: round() takes it to
    # 17.4 and round-half-up to 17.5, and which is right depends on a third
    # decimal the generator does not print. Re-rounding a rounded value is a
    # judgement call the prose should not have to make, so it does not.
    med = {r["model"]: r for r in full if r["effort"] == "medium"}
    for tag, mo in (("son", "claude-sonnet-5"), ("36f", "gemini-3.6-flash")):
        f[f"gen_med_{tag}_a"] = med[mo]["a"]
        f[f"gen_med_{tag}_b"] = med[mo]["b"]

    # Self-preference check. The claim is negative -- authorship does not
    # predict retention -- so what gets guarded is the two authors' position
    # in the retention order and the identity of the neighbours quoted.
    ret = {r["model"]: r["b"] / r["a"] for r in full if r["effort"] == "high"}
    rank = sorted(ret, key=ret.get)
    authors = {"gemini-3.6-flash": "36f", "claude-sonnet-5": "son"}
    for mo, tag in authors.items():
        f[f"gen_ret_{tag}"] = round(ret[mo], 2)
    idx = sorted(rank.index(mo) for mo in authors)
    f["gen_ret_authors_adjacent"] = idx[1] - idx[0] == 1
    f["gen_ret_below"] = rank[idx[0] - 1]
    f["gen_ret_above"] = rank[idx[1] + 1]
    f["gen_ret_below_val"] = round(ret[f["gen_ret_below"]], 2)
    f["gen_ret_above_val"] = round(ret[f["gen_ret_above"]], 2)

    # HOW MANY OF THESE SIX DID THE MAIN RESULTS ACTUALLY SEPARATE. This block
    # exists because a shipped draft got it exactly backwards: the all-pairs
    # correction "retains only ... of those ..." counts pairs that survived the
    # TOP-SIX correction, not pairs out of the all-pairs family. Those are
    # different quantities and both are pinned below so they can never again be
    # read as one.
    #
    # NOTHING HERE MAY NAME THE PAIR COUNT. The previous version split the
    # transcript on the literal "all 9 models -> 36 pairs" and stored the
    # results under keys called main_pairs36 / main_sep36. Growing the roster
    # to ten made the header read "all 10 models -> 45 pairs", so the split
    # raised IndexError -- which was the lucky outcome; had it been a regex
    # loose enough to match, the keys would have kept their stale names while
    # holding 45-pair numbers. Both the header and the key names are now
    # roster-neutral and the count is read out of the transcript.
    mc = run("mcnemar_v8.py")
    head = re.search(r"all (\d+) models -> (\d+) pairs", mc)
    assert head, "mcnemar_v8.py printed no all-pairs header"
    # BOUND THE LISTING. The pair lines of the all-pairs family and the pair
    # lines of the leader-vs-leader readout have the same shape, and the
    # readout sits below the family in the same transcript. The old `\s*$`
    # anchor excluded the readout only because those lines happen to carry a
    # "(same provider)" suffix today -- one cross-provider separation and the
    # family count would silently inflate. Cut the block at the next heading.
    blk = mc.split(head.group(0))[1].split("\n\n  leading group")[0]
    m = re.search(r"alpha = 0\.05/(\d+) = [\d.]+: (\d+) separate", blk)
    f["main_pairs_all"], f["main_sep_all"] = int(m.group(1)), int(m.group(2))
    assert f["main_pairs_all"] == int(head.group(2)), "pair count disagrees"
    sep = set()
    for ln in blk.splitlines():
        g = re.match(r"\s+(\S+)\s+vs (\S+)\s+b=\s*\d+ c=\s*\d+\s+p=[\d.]+\s*$",
                     ln)
        if g:
            sep.add(frozenset(g.groups()))
    assert len(sep) == f["main_sep_all"], (len(sep), f["main_sep_all"])

    # THE FRONTIER READOUT REPLACES THE TOP-SIX FAMILY. mcnemar_v8 no longer
    # corrects over a family chosen by looking at the pass rates, so there is
    # no longer a "top six" count to keep, and the keys that held one are gone
    # rather than renamed -- a pinned main_sep_top6 that quietly started
    # holding a leading-group number is exactly the failure this file exists to
    # prevent. What is pinned now is the decomposition, which must add up.
    lg = re.search(r"leading group \(not separated from (\S+) at the same "
                   r"alpha\): (\d+) of (\d+)", mc)
    assert lg, "mcnemar_v8.py printed no leading-group header"
    f["main_lead_best"], f["main_lead_n"] = lg.group(1), int(lg.group(2))
    f["main_lead_of"] = int(lg.group(3))
    bd = re.search(r"boundary: (\S+) in at p=([\d.]+), (\S+) out at "
                   r"p=([\d.]+) \((\d+)x\), across a ([\d.]+)-point gap", mc)
    assert bd, "mcnemar_v8.py printed no frontier boundary"
    f["main_lead_last_in"], f["main_lead_first_out"] = bd.group(1), bd.group(3)
    f["main_lead_gap_ratio"] = int(bd.group(5))
    f["main_lead_gap_points"] = float(bd.group(6))
    dec = dict((g[0], (int(g[1]), int(g[2]))) for g in re.findall(
        r"(leader vs leader|leader vs trailer|trailer vs trailer)\s+"
        r"(\d+) of\s+(\d+) pairs separate", mc))
    assert len(dec) == 3, dec
    f["main_lead_pairs"] = dec["leader vs leader"][1]
    f["main_lead_sep"] = dec["leader vs leader"][0]
    f["main_cross_sep"] = dec["leader vs trailer"][0]
    f["main_trail_sep"] = dec["trailer vs trailer"][0]
    f["main_trail_pairs"] = dec["trailer vs trailer"][1]
    assert sum(v[0] for v in dec.values()) == f["main_sep_all"], (
        f"the decomposition {dec} does not add up to {f['main_sep_all']}")
    assert sum(v[1] for v in dec.values()) == f["main_pairs_all"], (
        f"the decomposition {dec} does not cover {f['main_pairs_all']} pairs")
    f["main_lead_is_cluster"] = "CONFIRMED" in re.search(
        r"no pair inside the leading group separates: (\S+)", mc).group(1)
    f["main_lead_nearest_miss"] = float(re.search(
        r"nearest miss inside the leading group: \S+ vs \S+ at p=([\d.]+)",
        mc).group(1))

    six = sorted(roster)
    within = [frozenset(p) for i, a in enumerate(six) for p in
              ((a, b) for b in six[i + 1:])]
    f["gen_pairs_within"] = len(within)
    f["gen_pairs_sep"] = sum(1 for p in within if p in sep)
    unres = [p for p in within if p not in sep]
    f["gen_pairs_unsep"] = len(unres)
    f["gen_unsep_list"] = sorted(" vs ".join(sorted(p)) for p in unres)
    # The point the paragraph turns on: the pair that swaps at medium effort
    # is one of the pairs the main results never resolved.
    f["gen_swap_is_unresolved"] = frozenset(f["gen_swap_models"]) in unres
    return f


# --------------------------------------------------------------- sec 7.2
def effort_facts():
    """mcnemar_v8.py + failure_taxonomy.json + the corpus-B effort panel.

    Three sources because sec:effort makes three claims on three pools and
    they must not be allowed to blur into each other:
      - the effect, from mcnemar_v8.py's T6 over the full 600;
      - the mechanism, from failure_taxonomy.json, also over the full 600;
      - the replication, from generator_ablation.py's effort panel, over each
        corpus's OWN winnable rounds.
    Every rate below is tagged with which pool it came from.
    """
    f = {}

    # ---- (1) the effect ---------------------------------------------------
    # T6's `b` counts rounds the HIGHER level solved and the lower failed.
    mc = run("mcnemar_v8.py")
    t6 = mc.split("T6 ")[1].split("T7")[0]
    m = re.search(r"Bonferroni alpha = 0\.05/(\d+) = ([\d.]+)", t6)
    f["eff2_ntests"], f["eff2_alpha"] = int(m.group(1)), float(m.group(2))
    pair, rows = None, {}
    for ln in t6.splitlines():
        h = re.match(r"^(high|medium) vs (low|medium)\s*$", ln)
        if h:
            pair = ln.strip()
            continue
        g = re.match(r"^(\S+)\s+(\d+)\s+(\d+)\s+(-?\d+)\s+(\S+)\s+(.+?)\s*$",
                     ln)
        if g and pair:
            rows[(g.group(1), pair)] = dict(
                hi=int(g.group(2)), lo=int(g.group(3)),
                p=float(g.group(5)), verdict=g.group(6))
    assert len(rows) == f["eff2_ntests"], len(rows)
    hl = {mo: r for (mo, pr), r in rows.items() if pr == "high vs low"}
    f["eff2_models"] = len(hl)
    f["eff2_help"] = sum(1 for r in hl.values() if r["verdict"] == "high better")
    f["eff2_ns"] = sum(1 for r in hl.values() if r["verdict"] == "n.s.")
    f["eff2_hurt"] = sum(1 for r in hl.values() if r["verdict"] == "high worse")
    for tag, mo in (("son", "claude-sonnet-5"), ("31p", "gemini-3.1-pro"),
                    ("sol", "gpt-5.6-sol"), ("35f", "gemini-3.5-flash")):
        f[f"eff2_{tag}_hi"] = hl[mo]["hi"]
        f[f"eff2_{tag}_lo"] = hl[mo]["lo"]
        f[f"eff2_{tag}_p"] = float(f"{hl[mo]['p']:.2g}")
    # "not one round the other way" is a zero, not a small number.
    f["eff2_35f_zero_recovered"] = hl["gemini-3.5-flash"]["hi"] == 0

    # Of the OTHER eighteen, how many clear -- and which. The prose says one,
    # and that it is the same model; both halves are guarded.
    other = [(mo, pr, r) for (mo, pr), r in rows.items() if pr != "high vs low"
             and r["verdict"] != "n.s."]
    f["eff2_other_sig"] = len(other)
    f["eff2_other"] = sorted(f"{mo} {pr}" for mo, pr, _ in other)
    # THREE FIGURES, NOT TWO. This p is 1.65e-15 as printed, the exact half-way
    # case: .2g gives 1.6 and round-half-up gives 1.7, and the generator does
    # not print the digit that would settle it. The prose quotes 1.65 so that
    # no rounding decision is being made at all. Every other p here is safe at
    # two figures; this one is not, and the difference is why it is singled out.
    f["eff2_other_p"] = float(f"{other[0][2]['p']:.3g}") if other else None

    # THE CLAIM ABOUT THE UNCORRECTED ALPHA IS DELIBERATELY NARROW. Inside the
    # low-to-high panel -- the only panel the prose cites -- exactly one row
    # changes verdict when alpha goes from 0.05/30 to 0.05. Across all 30
    # contrasts six rows change. Both are asserted so the sentence can never be
    # widened by someone who checked only the panel.
    def flips(rs):
        return sorted(mo for mo, r in rs
                      if r["p"] < 0.05 and r["p"] > f["eff2_alpha"])
    f["eff2_uncorrected_flips"] = flips(list(hl.items()))
    f["eff2_uncorrected_flips_all"] = len(
        flips([(mo, r) for (mo, _), r in rows.items()]))
    f["eff2_terra_p"] = float(f"{hl['gpt-5.6-terra']['p']:.2g}")

    # ---- (2) the mechanism, full 600 ---------------------------------------
    sys.path.insert(0, os.path.dirname(_paper_file(os.path.join(
        "src", "make_body_figures.py"))))
    import make_body_figures as mbf              # noqa: E402

    lv = ("low", "medium", "high")
    by = {}
    for eff in lv:
        _, cells, order = mbf.taxonomy(effort=eff)
        by[eff] = cells
    step = {mo: [by[e][mo]["step_mean"] for e in lv] for mo in order}
    capn = {mo: [100.0 * by[e][mo]["capped"] / by[e][mo]["n"] for e in lv]
            for mo in order}
    f["eff2_steps_mono"] = sum(s[0] < s[1] < s[2] for s in step.values())
    # NOT 9. The prose claims monotone steps only, and this is why.
    f["eff2_capn_mono"] = sum(c[0] < c[1] < c[2] for c in capn.values())
    z = "gemini-3.5-flash"
    f["eff2_35f_steps_lo"] = step[z][0]
    f["eff2_35f_steps_hi"] = step[z][2]
    f["eff2_35f_capn_lo"] = round(capn[z][0], 1)
    f["eff2_35f_capn_hi"] = round(capn[z][2], 1)
    f["eff2_35f_pass_lo"] = by["low"][z]["pass_pct"]
    f["eff2_35f_pass_hi"] = by["high"][z]["pass_pct"]
    f["eff2_31p_capn_hi"] = round(capn["gemini-3.1-pro"][2], 1)
    f["eff2_36f_capn_hi"] = round(capn["gemini-3.6-flash"][2], 1)
    # The tie is exact, and "34 rounds each way" says so.
    f["eff2_36f_tie"] = hl["gemini-3.6-flash"]["hi"] == hl["gemini-3.6-flash"]["lo"]
    f["eff2_36f_each_way"] = hl["gemini-3.6-flash"]["hi"]
    # CAPPED/N IS NOT CAPPED/FAILED. sec:efficiency quotes 91.9% for this same
    # cell off the other denominator; both are pinned so a future edit cannot
    # swap one for the other and still pass.
    hic = by["high"][z]
    f["eff2_35f_cap_of_failed"] = round(
        100.0 * hic["capped"] / (hic["n"] - round(hic["pass_pct"]
                                                  * hic["n"] / 100)), 1)
    f["eff2_cap_denoms_differ"] = (f["eff2_35f_cap_of_failed"]
                                   != f["eff2_35f_capn_hi"])

    # ---- (3) the replication, corpus B's own winnable rounds ---------------
    ga = run("generator_ablation.py")
    panel = ga.split("EFFORT vs THE STEP BUDGET")[1]
    prow = {}
    for ln in panel.splitlines():
        g = re.match(r"^(\S+)\s+([AB])\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+"
                     r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
                     r"(True|False)\s+(\d+)\s+(\d+)\s+(\S+)\s*$", ln)
        if g:
            prow[(g.group(1), g.group(2))] = dict(
                budget=int(g.group(3)),
                cap=[float(g.group(i)) for i in (5, 6, 7)],
                pas=[float(g.group(i)) for i in (8, 9, 10)],
                mono=g.group(11) == "True",
                lo=int(g.group(12)), hi=int(g.group(13)),
                p=float(g.group(14)))
    f["eff2_b_models"] = sum(1 for _, k in prow if k == "B")
    f["eff2_b_budget"] = prow[(z, "B")]["budget"]
    f["eff2_b_capmono"] = sum(r["mono"] for (_, k), r in prow.items() if k == "B")
    m = re.search(r"inversion  corpus B \(n=(\d+)/(\d+) low/high\)", panel)
    f["eff2_b_35f_n_lo"], f["eff2_b_35f_n_hi"] = int(m.group(1)), int(m.group(2))
    b35 = prow[(z, "B")]
    f["eff2_b_35f_pass_lo"], f["eff2_b_35f_pass_hi"] = b35["pas"][0], b35["pas"][2]
    f["eff2_b_35f_cap_lo"], f["eff2_b_35f_cap_hi"] = b35["cap"][0], b35["cap"][2]
    # THE PANEL NAMES b AND c THE OTHER WAY ROUND FROM mcnemar_v8.py. Asserted
    # rather than commented: the two scripts currently agree on this cell once
    # the letters are crossed, and if either ever renames its columns this
    # trips instead of silently reversing a sentence.
    a35 = prow[(z, "A")]
    f["eff2_bc_letters_crossed"] = (a35["lo"] == hl[z]["lo"]
                                    and a35["hi"] == hl[z]["hi"])
    f["eff2_b_35f_lo_wins"], f["eff2_b_35f_hi_wins"] = b35["lo"], b35["hi"]
    f["eff2_b_35f_p"] = float(f"{b35['p']:.2g}")
    f["eff2_b_panel_n"] = len(prow)
    f["eff2_b_panel_alpha"] = round(0.05 / len(prow), 4)
    # The whole point of quoting both readings is that they disagree.
    f["eff2_b_35f_clears_05"] = b35["p"] < 0.05
    f["eff2_b_35f_misses_panel"] = b35["p"] > 0.05 / len(prow)
    o5 = prow[("claude-opus-5", "B")]
    f["eff2_a_opus5_p"] = float(f"{hl['claude-opus-5']['p']:.2g}")
    f["eff2_b_opus5_hi_wins"], f["eff2_b_opus5_lo_wins"] = o5["hi"], o5["lo"]
    f["eff2_b_opus5_p"] = float(f"{o5['p']:.2g}")
    return f


def ablation_facts():
    """ablation.py -- sec:budget-ablation and sec:modality-ablation.

    Parses the generator's stdout, per this file's design note. Nothing here
    is stochastic: the arms are the same rounds re-run under one changed
    setting, so every quantity is a count, a ratio of counts, or an exact
    McNemar p. Re-deriving them from the printed rows therefore cannot
    disagree with the table the way a second bootstrap draw could.

    THE TWO ARMS HAVE DIFFERENT ROSTERS. The budget block carries four models
    and the modality block three; the counts below are asserted separately
    rather than against one shared roster length, because making them agree
    would mean dropping gemini-3.5-flash, which is the row the budget moves
    most.
    """
    log = run("ablation.py")
    f = {}

    # ---- (a) the budget block ------------------------------------------
    bud = {}
    for ln in log.split("=== (a)")[1].split("paired exact")[0].splitlines():
        m = re.match(r"^(\S+)\s+(\d+)/(\d+)\s+([\d.]+)%\s+(\d+)/(\d+)\s+"
                     r"([\d.]+)%\s+(\d+)/(\d+)\s+([\d.]+)%\s+([\d.]+)%\s+"
                     r"([\d.]+)%\s*$", ln)
        if m:
            bud[m.group(1)] = dict(
                n=int(m.group(3)),
                p20=float(m.group(4)), p30=float(m.group(7)),
                p40=float(m.group(10)),
                k20=int(m.group(2)), k40=int(m.group(8)),
                cap20=float(m.group(11)), cap40=float(m.group(12)))
    f["ab_bud_models"] = len(bud)
    f["ab_n_per_cell"] = {r["n"] for r in bud.values()}.pop()
    m = re.search(r"Bonferroni (\d+) tests, alpha=([\d.]+)", log)
    f["ab_budget_tests"], f["ab_budget_alpha"] = int(m.group(1)), float(m.group(2))

    mc = {}
    for g in re.finditer(r"^\s+(\S+)\s+b=(\d+): gained\s+(\d+), lost\s+(\d+), "
                         r"p=(\S+)", log, re.M):
        mc[(g.group(1), int(g.group(2)))] = dict(
            gain=int(g.group(3)), loss=int(g.group(4)), p=float(g.group(5)))

    short = {"claude-opus-5": "opus5", "claude-sonnet-5": "sonnet5",
             "gemini-3.6-flash": "36flash", "gemini-3.5-flash": "35flash"}
    for full, s in short.items():
        r, t = bud[full], mc[(full, 40)]
        f[f"ab_{s}_20"], f[f"ab_{s}_40"] = r["p20"], r["p40"]
        # b=30 is quoted nowhere in prose, but Figure 2(a) DRAWS it as the
        # middle vertex of every curve, so figure_facts needs it exported.
        f[f"ab_{s}_30"] = r["p30"]
        f[f"ab_{s}_delta"] = round(r["p40"] - r["p20"], 1)
        f[f"ab_{s}_cap20"], f[f"ab_{s}_cap40"] = r["cap20"], r["cap40"]
        f[f"ab_{s}_gain40"], f[f"ab_{s}_loss40"] = t["gain"], t["loss"]
        f[f"ab_{s}_p40"] = float(f"{t['p']:.2g}")
        f[f"ab_{s}_churn40"] = round(100.0 * (t["gain"] + t["loss"]) / r["n"], 1)
        f[f"ab_{s}_sig40"] = t["p"] < f["ab_budget_alpha"]

    # THE RELATIONSHIP IS A SPLIT, NOT A GRADIENT, AND AN EARLIER DRAFT GOT
    # THIS WRONG. It said the gain from opening the cap is ordered exactly as
    # the b=20 cap-hit rate; it is not. claude-opus-5 has the LOWER cap-hit
    # rate (0.5 against 3.3) and the LARGER gain (+1.8 against +0.5), so the
    # bottom two are inverted. Both facts are shipped -- the split that holds
    # and the ordering that does not -- so the sentence cannot drift back into
    # a dose-response claim.
    by_gain = sorted(bud, key=lambda k: bud[k]["p40"] - bud[k]["p20"])
    by_cap = sorted(bud, key=lambda k: bud[k]["cap20"])
    f["ab_gain_tracks_cap"] = by_gain == by_cap
    f["ab_budget_sig40"] = sum(f[f"ab_{s}_sig40"] for s in short.values())
    # ... but the two models the cap almost never bound are exactly the two
    # that did not move, which is the claim the paragraph actually makes.
    f["ab_split_clean"] = (set(by_cap[:2])
                           == {m for m in bud
                               if not f[f"ab_{short[m]}_sig40"]})

    # ---- (c) rank stability, churn, spread, the sec:effort tie-in -------
    m = re.search(r"(\d+) discordant pair\(s\), top stable (True|False), "
                  r"bottom stable (True|False)", log)
    f["ab_rank_swaps"] = int(m.group(1))
    f["ab_rank_top_stable"] = m.group(2) == "True"
    f["ab_rank_bot_stable"] = m.group(3) == "True"
    m = re.search(r"swapped: (\S+) / (\S+)", log)
    f["ab_rank_swapped"] = f"{m.group(1)},{m.group(2)}"
    m = re.search(r"b=20 ([\d.]+)x, b=40 ([\d.]+)x", log)
    f["ab_spread20"], f["ab_spread40"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"at b=40 : ([\d.]+)% pass, ([\d.]+)% cap, ([\d.]+) steps", log)
    f["ab_35f_b40_pass"] = float(m.group(1))
    f["ab_35f_b40_cap"] = float(m.group(2))
    f["ab_35f_b40_steps"] = float(m.group(3))
    m = re.search(r"LOW effort \(sec:effort\)\s+: ([\d.]+)% pass, ([\d.]+)% cap, "
                  r"([\d.]+) steps", log)
    f["ab_35f_low_pass"] = float(m.group(1))
    f["ab_35f_low_cap"] = float(m.group(2))
    f["ab_35f_low_steps"] = float(m.group(3))
    # "recovers essentially all of the 10.5-point inversion": the inversion is
    # low-vs-high at the SHIPPED cap, and the recovery is the delta column.
    # Both are quoted, so the subtraction is pinned here.
    f["ab_35f_inversion"] = round(f["ab_35f_low_pass"] - f["ab_35flash_20"], 1)
    m = re.search(r"provider errors retained as failures: b40/(\S+) (\d+)", log)
    f["ab_err_model"], f["ab_err_n"] = m.group(1), int(m.group(2))

    # ---- (b) the modality block -----------------------------------------
    mod = {}
    for ln in log.split("=== (b)")[1].split("modality Bonferroni")[0].splitlines():
        m = re.match(r"^(\S+)\s+(all|gated|ungated)\s+(\d+)\s+([\d.]+)%\s+"
                     r"([\d.]+)%\s+([\d.]+)\s+(\d+)/(\d+)\s+(\S+)\s*$", ln)
        if m:
            mod[(m.group(1), m.group(2))] = dict(
                n=int(m.group(3)), sighted=float(m.group(4)),
                blind=float(m.group(5)), retained=float(m.group(6)),
                gain=int(m.group(7)), loss=int(m.group(8)),
                p=float(m.group(9)))
    blind_roster = sorted({k for k, _ in mod})
    f["ab_mod_models"] = len(blind_roster)
    f["ab_gated_n"] = {mod[(k, "gated")]["n"] for k in blind_roster}.pop()
    f["ab_ungated_n"] = {mod[(k, "ungated")]["n"] for k in blind_roster}.pop()
    f["ab_all_n"] = {mod[(k, "all")]["n"] for k in blind_roster}.pop()
    f["ab_strata_partition"] = f["ab_gated_n"] + f["ab_ungated_n"] == f["ab_all_n"]
    m = re.search(r"modality Bonferroni (\d+) tests.*alpha=([\d.]+); "
                  r"(\d+)/(\d+) significant", log)
    f["ab_mod_tests"] = int(m.group(1))
    f["ab_mod_sig_ungated"] = int(m.group(3))

    for full, s in short.items():
        if full not in blind_roster:
            continue
        u, g, a = (mod[(full, x)] for x in ("ungated", "gated", "all"))
        f[f"ab_mod_{s}_ung_sight"], f[f"ab_mod_{s}_ung_blind"] = (u["sighted"],
                                                                  u["blind"])
        f[f"ab_mod_{s}_ung_gain"], f[f"ab_mod_{s}_ung_loss"] = u["gain"], u["loss"]
        f[f"ab_mod_{s}_gated_blind"] = g["blind"]
        f[f"ab_mod_{s}_all_blindonly"] = a["loss"]
        # The prose argues on the UNGATED stratum only, so the all-600 arm was
        # never exported. Figure 2(b) draws both arms, and its all-600 sonnet-5
        # drop label shipped as -87% when the generator says 0.136 -> -86%.
        # Exporting the all-600 pair is what lets figure_facts catch that.
        f[f"ab_mod_{s}_all_sight"], f[f"ab_mod_{s}_all_blind"] = (a["sighted"],
                                                                  a["blind"])
        f[f"ab_mod_{s}_all_ret"] = a["retained"]
        f[f"ab_mod_{s}_ung_ret"] = u["retained"]
    ung = [mod[(k, "ungated")] for k in blind_roster]
    f["ab_mod_retained_lo"] = round(min(r["retained"] for r in ung), 3)
    f["ab_mod_retained_hi"] = round(max(r["retained"] for r in ung), 3)
    f["ab_mod_gated_blind_lo"] = min(mod[(k, "gated")]["blind"]
                                     for k in blind_roster)
    f["ab_mod_gated_blind_hi"] = max(mod[(k, "gated")]["blind"]
                                     for k in blind_roster)
    # THE PROSE QUOTES THE LARGEST OF THE THREE p VALUES, NOT AN UPPER BOUND.
    # 2.61e-12 rounds DOWN to 2.6e-12, so "p <= 2.6e-12" would be false as
    # printed. Pinned as the max so the sentence cannot be reworded into a
    # bound without this changing.
    f["ab_mod_ung_pmax"] = float(f"{max(r['p'] for r in ung):.2g}")
    # blind is not nested inside sighted; the boundary paragraph says so.
    f["ab_mod_any_blind_only"] = sum(mod[(k, "all")]["loss"] > 0
                                     for k in blind_roster)

    # ---- (d) did the gained rounds actually spend the new budget? --------
    # GROSS CHURN AND NET MOVEMENT ARE PINNED SEPARATELY ON PURPOSE. A shipped
    # draft quoted the 8.2% flip rate as though it were the size of a re-run
    # wobble in pass-rate points; the net is +1.8. Both are facts now, so a
    # future rewording cannot substitute one for the other silently.
    for ln in log.split("=== (d)")[1].split("=== (e)")[0].splitlines():
        m = re.match(r"^(\S+)\s+(\d+)\s+(\d+)\s+(\d+)/(\d+)\s+([\d.]+)%\s+"
                     r"([+-][\d.]+)\s*$", ln)
        if not m:
            continue
        s = short[m.group(1)]
        # The gain count and the churn rate are already pinned from the (a)
        # McNemar block. Re-deriving them here would give one number two
        # owners and let a future divergence resolve silently by assignment
        # order, so this block CHECKS them and only adds what is new.
        assert f[f"ab_{s}_gain40"] == int(m.group(2)), (
            f"(d) and (a) disagree on {s} gains: "
            f"{m.group(2)} vs {f[f'ab_{s}_gain40']}")
        assert f[f"ab_{s}_churn40"] == float(m.group(6)), (
            f"(d) and (a) disagree on {s} churn: "
            f"{m.group(6)} vs {f[f'ab_{s}_churn40']}")
        f[f"ab_{s}_gainspent"] = int(m.group(3))
        f[f"ab_{s}_over20_b40"] = int(m.group(4))
        f[f"ab_{s}_net40"] = float(m.group(7))

    # ---- (e) what the cap does to sec:main-results' separations ----------
    m = re.search(r"all (\d+) pairs over (\d+) models at alpha=([\d.]+): "
                  r"(\d+) separate; (\d+) of those involve only re-run", log)
    f["ab_sep_pairs_all"] = int(m.group(1))
    f["ab_sep_n_all"] = int(m.group(4))
    f["ab_sep_testable"] = int(m.group(5))
    f["ab_sep_alpha_all"] = float(m.group(3))
    m = re.search(r"survived (\d+), lost (\d+), gained (\d+)", log)
    f["ab_sep_survived"] = int(m.group(1))
    f["ab_sep_lost"] = int(m.group(2))
    f["ab_sep_gained"] = int(m.group(3))
    m = re.search(r"lost pair\s+: (\S+) / (\S+) \((\d+)/(\d+) -> (\d+)/(\d+)\), "
                  r"discordant (\d+) -> (\d+) \(power up: (True|False)\), "
                  r"effect ([+-][\d.]+) -> ([+-][\d.]+) pts", log)
    f["ab_sep_lost_pair"] = f"{m.group(1)},{m.group(2)}"
    f["ab_sep_lost_gl20"] = f"{m.group(3)}/{m.group(4)}"
    f["ab_sep_lost_gl40"] = f"{m.group(5)}/{m.group(6)}"
    f["ab_sep_lost_disc20"] = int(m.group(7))
    f["ab_sep_lost_disc40"] = int(m.group(8))
    # The whole "not a thinner test" argument is this boolean.
    f["ab_sep_lost_power_up"] = m.group(9) == "True"
    f["ab_sep_lost_eff20"] = float(m.group(10))
    f["ab_sep_lost_eff40"] = float(m.group(11))
    m = re.search(r"gained pair : (\S+) / (\S+) \(p ([\d.]+) -> ([\d.]+), "
                  r"threshold ([\d.]+)\)", log)
    f["ab_sep_gained_pair"] = f"{m.group(1)},{m.group(2)}"
    f["ab_sep_gained_p20"] = m.group(3)
    f["ab_sep_gained_p40"] = m.group(4)
    f["ab_sep_gained_thresh"] = m.group(5)
    # Roster-neutral on purpose: the family sizes in the braces are
    # ablation.py's ab_sep_corrections, and the largest of them is the
    # all-pairs count, which moves with the roster (36 at nine models, 45 at
    # ten). Pinning the literal set here just makes this regex the next thing
    # to break when a model is added. The set itself is captured as a fact so a
    # change still has to be acknowledged rather than absorbed.
    m = re.search(r"robust at alpha 0\.05/\{([\d,]+)\}: (True|False)", log)
    f["ab_sep_corrections"] = m.group(1)
    f["ab_sep_robust"] = m.group(2) == "True"
    # THE PROSE QUOTES THE LARGEST OF THE THREE, NOT A BOUND -- same trap as
    # ab_mod_ung_pmax above. 3.11e-07 rounds DOWN to 3.1e-07, so "p <= 3.1e-7"
    # would be false as printed.
    m = re.search(r"gemini-3\.5-flash separations surviving b=40: (\d+), "
                  r"largest p = (\S+)", log)
    f["ab_sep_worst_survive"] = int(m.group(1))
    f["ab_sep_worst_maxp40"] = m.group(2)
    # The lost pair's two p-values, read off the per-pair rows rather than the
    # summary line, so the prose's 6.6e-5 / 0.31 are pinned to the same source
    # the table of pairs prints.
    lost = f["ab_sep_lost_pair"].split(",")
    for ln in log.split("=== (e)")[1].splitlines():
        m = re.match(r"^\s+(\S+)\s+vs (\S+)\s+b20\s+(\d+)/(\d+)\s+p=\s*(\S+)\s+"
                     r"\| b40\s+(\d+)/(\d+)\s+p=\s*(\S+)\s+\[", ln)
        if m and [m.group(1), m.group(2)] == lost:
            f["ab_sep_lost_p20"] = float(f"{float(m.group(5)):.2g}")
            f["ab_sep_lost_p40"] = float(f"{float(m.group(8)):.2g}")
    return f


# ----------------------------------------------------------------- sec 3
def judge_ablation_facts():
    """The judge ablation's guard, folded in for the same reason as Section 3.

    analysis/judge_ablation.py checks \\S\\ref{sec:judge-ablation}, its appendix
    counterpart and all three panels of Table~\\ref{tab:judge-app} against
    src/report/judge_ablation.json. That table is hand-written inline rather
    than emitted by src/extract_tables.py, so verify_table_bodies.py never sees
    it; if this guard is not on the path anybody runs, nothing protects it.
    """
    bad, total = _judge_ablation.verify()
    return {"judge_mismatches": tuple(bad), "judge_checked": total}


def section3_facts():
    """Section 3's guard, folded in so that one command covers the paper.

    section3_facts.py has always had its own --check, but it was not on any
    path anybody actually ran: the release chain is difficulty_bands ->
    paste_table_bodies -> verify_table_bodies -> this file -> build.sh, and
    Section 3 appeared in none of those five. A guard nobody runs is a guard
    that rots, so it is called from here.

    Its SHIPPED dict stays over there. Section 3 counts characterise the
    CORPUS and are recomputed from the task files; Sections 4-7 numbers are
    parsed out of generator stdout, and this file's design note forbids
    recomputation. Merging the two dicts would blur that line. What crosses
    over is the verdict.
    """
    S3 = _section3_facts
    f = S3.facts()
    code = S3.code_facts()
    f.update(code)
    f.update(S3.difficulty_facts())

    bad, missing = [], []
    for k, want in S3.SHIPPED.items():
        if k not in f:
            missing.append(k)
        elif f[k] != want:
            bad.append(f"{k}: Section 3 prose says {want}, corpus says {f[k]}")

    # The extension-surface counts (3.6) and protocol families (3.7) are read
    # off the harness source tree, which is not in this repo. Unreachable, they
    # simply do not compute -- and a guard that quietly checks 89 of 98 values
    # is worse than no guard, because the tally still says OK. So the shortfall
    # is reported as a value, with the fix in it.
    unverified = ()
    if missing:
        unverified = (f"CLASHARNESS_SRC unset or not a directory ({S3.CODE!r}), "
                      f"so {len(missing)} Section 3 values are UNCHECKED; "
                      f"re-run as CLASHARNESS_SRC=/path/to/clash-harness "
                      f"python3 analysis/results_facts.py --check",)

    return {"s3_mismatches": tuple(bad), "s3_unverified": unverified,
            "s3_checked": len(S3.SHIPPED) - len(missing)}


SECTIONS = [main_facts, main_bands_facts, main_caption_facts, cost_facts,
            validity_facts,
            concentration_facts, capability_facts, taxonomy_facts,
            efficiency_facts, generator_facts, effort_facts, ablation_facts,
            # Last on purpose: it calls ablation_facts() and generator_facts()
            # itself, and by here both are memoised, so it costs a parse of a
            # 263-line SVG rather than a second run of the generators.
            _figure_facts.figure_facts, section3_facts,
            # Reads one committed JSON and one .tex; no generator, no network.
            judge_ablation_facts,
            # Touches neither the eval side nor the generators -- it reads the
            # scenario corpus and the harness source only, so it is independent
            # of everything above and costs ~2s.
            _round_example_facts.round_example_facts]


def main():
    f = {}
    for fn in SECTIONS:
        f.update(fn())
    for k in sorted(f):
        print(f"  {k:28} {f[k]}")

    if "--check" in sys.argv:
        bad = [f"{k}: prose says {w}, generator says {f.get(k, 'MISSING')}"
               for k, w in SHIPPED.items() if f.get(k) != w]
        if bad:
            raise SystemExit("RESULTS DRIFT\n  " + "\n  ".join(bad))
        print(f"\n  OK: all {len(SHIPPED)} values cited in prose reproduce.")


if __name__ == "__main__":
    main()
