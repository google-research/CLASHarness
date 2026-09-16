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

"""Every number the judge-provenance ablation asserts, checked against its artifact.

The ablation itself lives in analysis/rejudge/: it replays all 9,000 recorded
high-effort rounds offline, re-adjudicates every semantic check with three
substitute judge ensembles, and writes ~90MB of prompts and verdicts into a
working tree outside this repository. None of that can be committed, so
analysis/rejudge/phase3_export.py freezes the result into one compact artifact,
src/report/judge_ablation.json, following the convention corpus.py:69 sets for
the other report JSONs -- "committed under src/report, so the self-check runs
from a clean clone with no source checkout present".

This file is the guard on that artifact's consumers. Unlike results_facts.py it
does not parse a generator's stdout, because the generator is a multi-hour API
sweep that cannot be re-run on demand; the artifact IS the frozen generator
output, and everything the paper says about the ablation is asserted against it.

Both consumers are checked:

  * prose   -- \\S\\ref{sec:judge-ablation} in the body, \\S\\ref{sec:app-judge-
               ablation} in the appendix, and the two sentences elsewhere that
               forward-reference them (the Rubric Composition paragraph of
               \\S\\ref{sec:validity} and limitation (4)).
  * tables  -- Table~\\ref{tab:judge-app}, every cell of it.

Table~\\ref{tab:judge-app} is hand-written inline in main.tex rather than emitted
by src/extract_tables.py, so src/verify_table_bodies.py does not cover it. This
script is the only thing standing between that table and silent drift, which is
why it re-derives every cell rather than spot-checking a few.

Run:  python3 analysis/judge_ablation.py           # print what was derived
      python3 analysis/judge_ablation.py --check   # assert the shipped values

Needs no CLASHARNESS_SRC and no eval_results: artifact plus main.tex, nothing else.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ARTIFACT = os.path.join(ROOT, "src", "report", "judge_ablation.json")
PAPER = None  # resolved lazily via _paper_file("main.tex")

# ---------------------------------------------------------------- paper src --
# This script is a FACT-CHECKER for the paper: it re-derives numbers from the
# corpus and asserts that the LaTeX/SVG sources print them. Those sources live
# in the paper repository, not here, so point CLASHARNESS_PAPER_SRC at that
# checkout to run it. Everything else under analysis/ works without it.
PAPER_SRC = os.environ.get("CLASHARNESS_PAPER_SRC", ROOT)


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


# The judges, in the order the paper presents them. "gemini36" is the incumbent
# re-run against itself, so it is a same-judge noise floor rather than a swap;
# the paper leans on that distinction, so the ordering is load-bearing.
JUDGES = ("gemini36", "sonnet5", "opus5")

# Slug -> the name printed in Table~\ref{tab:judge-app}. Hardcoded rather than
# imported from analysis/rejudge/replay_core.py, which needs the harness
# checkout and the scenario corpus; this file must run from a clean clone.
# check_roster() asserts it still matches the artifact exactly, so adding a
# model to the panel breaks the guard loudly instead of skipping a row.
PRETTY = {
    "opus_5": "claude-opus-5",
    "5_6_sol": "gpt-5.6-sol",
    "opus_4_8": "claude-opus-4.8",
    "kimi_k3": "kimi-k3",
    "sonnet_5": "claude-sonnet-5",
    "3_7_flash": "gemini-3.7-flash",
    "5_6_terra": "gpt-5.6-terra",
    "5_6_luna": "gpt-5.6-luna",
    "v4_pro": "deepseek-v4-pro",
    "3_6_flash": "gemini-3.6-flash",
    "glm_5_2": "glm-5.2-fp8",
    "3_1_pro": "gemini-3.1-pro",
    "dsv4_flash": "deepseek-v4-flash",
    "3_5_flash": "gemini-3.5-flash",
}

# The one row of Table~\ref{tab:main} whose recorded verdicts were issued by the
# model under test. The paper decomposes it in prose rather than in the table,
# and the "excluding the self-scored row" correlations drop it.
SELF_SCORED = "3_7_flash"


# --------------------------------------------------------------------------
# formatting: the paper's conventions, in one place


def n(x: int) -> str:
    """12345 -> '12{,}345', LaTeX's thousands separator."""
    return f"{x:,}".replace(",", "{,}")


def signed(x: float, dp: int = 2) -> str:
    return f"{x:+.{dp}f}"


# The body subsection is one prose paragraph and spells small counts out, so a
# few guarded strings need the word rather than the digit. Kept as a lookup
# rather than a formatter because the only values that reach it are rank
# displacements and the panel size, all small and all bounded by the artifact.
WORDS = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 14: "fourteen"}


def places(k: int) -> str:
    return f"{WORDS[k]} place" + ("" if k == 1 else "s")


# --------------------------------------------------------------------------
# score-side statistics, recomputed from the artifact
#
# Table~\ref{tab:judge-app} leads with mean score. The artifact was exported
# when pass rate was the headline, so d["score_summary"] carries only rho and
# largest_shift; the rest of the score block -- rank displacement, how many
# models move, the mean shift and the direction test on it -- is derived here
# from d["score"], which holds the same frozen per-model numbers the export
# summarised. score_stats() re-derives rho too and check_structure() asserts it
# reproduces d["score_summary"]["rho"] exactly, so this recomputation cannot
# quietly disagree with the artifact it is standing in for.


def _ranks(vals: dict[str, float]) -> dict[str, int]:
    """Descending rank, 1 = highest. Ties are impossible here and not handled."""
    return {m: i + 1 for i, m in enumerate(sorted(vals, key=lambda m: -vals[m]))}


def spearman(a: dict[str, float], b: dict[str, float], keys) -> float:
    keys = list(keys)
    ra, rb = _ranks({m: a[m] for m in keys}), _ranks({m: b[m] for m in keys})
    k = len(keys)
    dd = sum((ra[m] - rb[m]) ** 2 for m in keys)
    return 1 - 6 * dd / (k * (k * k - 1))


def sign_test_p(up: int, total: int) -> float:
    """Exact two-sided binomial test against p=1/2.

    Two-sided on purpose: the question is whether a judge shifts scores at all,
    not whether it shifts them upward, and the one-sided figure would overstate
    the case against the two Claude judges by a factor of two.
    """
    from math import comb
    tail = sum(comb(total, i) for i in range(min(up, total - up) + 1))
    return min(1.0, 2 * tail / 2 ** total)


def score_stats(d: dict) -> dict[str, dict]:
    sc, models = d["score"], list(d["rate"]["recorded"])
    base = sc["recorded"]
    out = {}
    for j in JUDGES:
        disp = {m: _ranks(sc[j])[m] - _ranks(base)[m] for m in models}
        deltas = [sc[j][m] - base[m] for m in models]
        up = sum(1 for x in deltas if x > 0)
        out[j] = {
            "rho": spearman(base, sc[j], models),
            "rho_ex_self_graded": spearman(
                base, sc[j], [m for m in models if m != SELF_SCORED]),
            "max_disp": max(abs(v) for v in disp.values()),
            "moved": sum(1 for v in disp.values() if v),
            "mean_shift": sum(deltas) / len(deltas),
            "up": up,
            "sign_p": sign_test_p(up, len(models)),
            "largest_shift": max(abs(x) for x in deltas),
        }
    return out


def load() -> dict:
    if not os.path.exists(ARTIFACT):
        raise SystemExit(
            f"missing {ARTIFACT}\n"
            "  regenerate with: cd analysis/rejudge && python3 phase3_export.py\n"
            "  (needs the rejudge working tree; see that file's docstring)")
    return json.load(open(ARTIFACT, encoding="utf-8"))


def regions() -> dict[str, str]:
    """Slice main.tex into the four places that talk about this ablation.

    Whitespace is collapsed so that table alignment padding -- which is
    cosmetic and gets reflowed by hand -- cannot fail a check on its own.
    """
    src = open(_paper_file("main.tex"), encoding="utf-8").read()

    def cut(start: str, end: str, label: str) -> str:
        i = src.find(start)
        if i < 0:
            raise SystemExit(f"cannot find the {label} region: {start!r} absent "
                             f"from main.tex")
        j = src.find(end, i)
        if j < 0:
            raise SystemExit(f"{label} region is unterminated: no {end!r} after "
                             f"{start!r}")
        return re.sub(r"[ \t]+", " ", src[i:j])

    return {
        "body": cut(r"\subsection{Judge provenance and cross-judge stability}",
                    r"\section{Limitations}", "body subsection"),
        "apx": cut(r"\subsection{Judge Provenance and Cross-Judge Stability}",
                   r"\input{appendix_report}", "appendix subsection"),
        "validity": cut(r"\noindent\smallskip\textbf{Rubric Composition",
                        r"\noindent\smallskip\textbf{Findings That Do Not",
                        "validity paragraph"),
        # Anchored on the stem, not the count: the enumeration has grown once
        # already (five -> seven, when sec:conflict-gate added the missing
        # conflict-free baseline and the temporal-versus-informational null),
        # and a literal "five" turned that edit into a hard SystemExit here
        # rather than a diff in the claims this file actually checks. The
        # stem plus the trailing colon is still unique in main.tex.
        "limits": cut(r"main scope limitations:",
                      r"\section{Conclusion}", "limitations paragraph"),
    }


# --------------------------------------------------------------------------
# the claims


def check_roster(d: dict, fail) -> None:
    """The panel PRETTY describes must be the panel the artifact measured."""
    got = set(d["rate"]["recorded"])
    want = set(PRETTY)
    if got != want:
        fail("roster", f"artifact panel {sorted(got - want)} unknown to PRETTY, "
                       f"PRETTY has {sorted(want - got)} the artifact lacks")
    for s in d["settings"]:
        if s != "recorded" and s not in JUDGES:
            fail("roster", f"artifact has judge {s!r} this guard does not check")
    for j in JUDGES:
        if j not in d["settings"]:
            fail("roster", f"judge {j!r} missing from the artifact")


def derive(d: dict) -> dict[str, list[tuple[str, str]]]:
    """Build, per region, the list of (label, exact substring) the text must contain.

    Everything is formatted from the artifact, so a change to the artifact
    changes what the paper is required to say.
    """
    rate = d["rate"]
    summ, agr = d["summary"], d["agreement"]
    cov, health = d["panel_coverage"], d["ensemble_health"]
    sc = d["score_summary"]
    fid, dec = d["replay_fidelity"], d["gemini_3_7_decomposition"]
    g37 = dec["under"]["gemini36"]

    panel_total = cov["gemini36"]["total"]
    common = agr["gemini36"]["n"]
    common_pct = 100.0 * common / panel_total
    opus_decl_pct = 100.0 * cov["opus5"]["declined"] / panel_total
    son_decl_pct = 100.0 * cov["sonnet5"]["declined"] / panel_total
    aux = fid["judge_pairs"] - panel_total

    # Largest single move under any substitute, used in the appendix.
    any_worst = max(summ[j]["largest_move_pp"] for j in JUDGES)

    # The body is one paragraph now, and it makes its stability claim on mean
    # score rather than on pass rate, matching Table~\ref{tab:judge-app}. The
    # pass-rate rho figures it used to carry still live in \S\ref{sec:validity},
    # which is guarded separately below. Two numbers the body needs are the
    # worst score displacement over the two substitute families and the recorded
    # pass-rate margin between the pair that trades places; both are derived
    # here so the sentence cannot drift away from the artifact.
    ssc_body = score_stats(d)
    claude_disp = max(ssc_body[j]["max_disp"] for j in ("sonnet5", "opus5"))
    lead_margin = abs(rate["recorded"]["opus_5"] - rate["recorded"]["5_6_sol"])

    body = [
        ("rounds replayed", f"all ${n(fid['rounds'])}$ recorded high-effort rounds"),
        ("deterministic gate", f"all ${n(fid['deterministic_checks'])}$ deterministic checks"),
        ("opus refusal rate", f"declined ${opus_decl_pct:.1f}\\%$ of the judge prompts"),
        ("common subset", f"${n(common)}$ of ${n(panel_total)}$ panel checks "
                          f"(${common_pct:.1f}\\%$)"),
        ("opus F->T", f"${agr['opus5']['false_to_true']}$ recorded failures into passes"),
        ("opus T->F", f"against ${agr['opus5']['true_to_false']}$ the other way"),
        ("opus lift breadth", f"raising all {WORDS[ssc_body['opus5']['up']]} models at once"),
        ("score displacement",
         f"by more than {places(claude_disp)}, against "
         f"{places(ssc_body['gemini36']['max_disp'])}"),
        ("sonnet agreement", f"${100 * agr['sonnet5']['agreement']:.1f}\\%$ "
                             f"($\\kappa = {agr['sonnet5']['kappa']:.2f}$)"),
        ("opus agreement", f"${100 * agr['opus5']['agreement']:.1f}\\%$ "
                           f"($\\kappa = {agr['opus5']['kappa']:.2f}$)"),
        ("gemini36 agreement", f"${100 * agr['gemini36']['agreement']:.1f}\\%$ "
                               f"($\\kappa = {agr['gemini36']['kappa']:.2f}$)"),
        ("leader margin", f"margin of ${lead_margin:.2f}$ pass-rate points"),
        ("observed gap", f"${signed(dec['observed_gap'])}$-point rollout gain"),
        ("judge term", f"${signed(g37['judge_term'])}$ points of it is judge change"),
    ]

    apx = [
        ("rounds replayed", f"${n(fid['rounds'])}$ recorded high-effort rounds"),
        ("panel rounds", f"${n(600 * len(PRETTY))}$ rounds of the fourteen-model panel"),
        ("deterministic gate", f"all ${n(fid['deterministic_checks'])}$ of them reproduce"),
        ("fidelity", f"a fidelity of ${100 * fid['fidelity']:.1f}\\%$"),
        ("prompt counts", f"${n(fid['judge_pairs'])}$ judge prompts (${n(panel_total)}$ "
                          f"from the panel, ${n(aux)}$ from the auxiliary rollout)"),
        ("sonnet declined", f"declined ${n(cov['sonnet5']['declined'])}$ of the "
                            f"${n(panel_total)}$ panel prompts (${son_decl_pct:.2f}\\%$)"),
        ("opus declined", f"declined ${n(cov['opus5']['declined'])}$ "
                          f"(${opus_decl_pct:.1f}\\%$)"),
        ("common subset", f"${n(common)}$ panel checks (${common_pct:.1f}\\%$)"),
        ("fallback bound", "at most $%.2f$, $%.2f$ and $%.2f$ points" % tuple(
            d["missing_pair_bounds"][j]["max_swing_pp"] for j in JUDGES)),
        ("thin counts", f"${health['sonnet5']['thin']}$ and ${health['opus5']['thin']}$ checks "
                        f"(${100 * health['sonnet5']['thin'] / health['sonnet5']['n']:.1f}\\%$ and "
                        f"${100 * health['opus5']['thin'] / health['opus5']['n']:.1f}\\%$)"),
        ("thin swing", f"${health['sonnet5']['thin_flip_max_swing_pp']:.2f}$ and "
                       f"${health['opus5']['thin_flip_max_swing_pp']:.2f}$ points"),
        ("split rates", "on $%.2f\\%%$ and $%.2f\\%%$ of prompts against $%.2f\\%%$" % (
            100 * health["sonnet5"]["split"] / health["sonnet5"]["n"],
            100 * health["opus5"]["split"] / health["opus5"]["n"],
            100 * health["gemini36"]["split"] / health["gemini36"]["n"])),
        ("score rho", "score rank correlation is $%.3f$, $%.3f$ and $%.3f$" % tuple(
            sc[j]["rho"] for j in JUDGES)),
        ("score shift", "largest absolute score shift for any model is $%.3f$, $%.3f$ and $%.3f$"
                        % tuple(sc[j]["largest_shift"] for j in JUDGES)),
        ("largest move", f"largest single pass-rate move under any substitute judge is "
                         f"${signed(any_worst, 1)}$ points"),
        ("observed gap", f"the ${signed(dec['observed_gap'])}$-point difference"),
        ("judge term", f"judge term dominates at ${signed(g37['judge_term'])}$ points"),
        ("rollout term", f"rollout term is ${signed(g37['rollout_term'])}$ points"),
        ("mcnemar", f"($b = {g37['mcnemar_b']}$, $c = {g37['mcnemar_c']}$, "
                    f"$p = {g37['mcnemar_p']:.2f}$, $n = {g37['n_paired']}$)"),
        # caption
        ("caption fidelity", f"${n(fid['deterministic_checks'])}/"
                             f"{n(fid['deterministic_match'])}$ deterministic verdicts"),
        ("caption subset", f"${n(common)}$ of ${n(panel_total)}$ panel checks "
                           f"(${common_pct:.1f}\\%$)"),
    ]

    # ---- The table: one judge-level panel, mean score first ------------------
    # Mean score is the primary metric. It is the right one for a judge
    # ablation: pass rate is conjunctive, so a single flipped check can move a
    # model across the whole panel, whereas mean score responds in proportion
    # to how many checks a judge actually sees differently. Pass rate follows
    # as the secondary block because Table~\ref{tab:main} leads with it, and
    # the check-level block underneath is metric-independent.
    ssc = score_stats(d)

    # The score and pass-rate blocks share four row labels, which reads fine
    # under their headings but would give two guard failures the same name, so
    # the block tag is carried separately from the printed label.
    def row(block: str, label: str, fmt) -> tuple[str, str]:
        return (f"table {block}/{label.strip()}",
                f"{label} & " + " & ".join(fmt(j) for j in JUDGES) + r" \\")

    apx += [
        ("table score header", r"\emph{Mean score (primary).}"),
        row("score", r"Rank correlation $\rho$", lambda j: f"{ssc[j]['rho']:.3f}"),
        row("score", r"\quad excluding the self-scored row",
            lambda j: f"{ssc[j]['rho_ex_self_graded']:.3f}"),
        row("score", "Maximum rank displacement", lambda j: str(ssc[j]["max_disp"])),
        row("score", "Models whose rank moves",
            lambda j: f"{ssc[j]['moved']}/{len(PRETTY)}"),
        row("score", "Mean shift", lambda j: f"${signed(ssc[j]['mean_shift'], 3)}$"),
        row("score", "Largest absolute shift",
            lambda j: f"{ssc[j]['largest_shift']:.3f}"),
        row("score", "Models scored higher", lambda j: f"{ssc[j]['up']}/{len(PRETTY)}"),
        row("score", r"\quad sign test $p$ (two-sided)",
            lambda j: f"{ssc[j]['sign_p']:.2g}"),

        ("table rate header", r"\emph{Pass rate.}"),
        row("rate", r"Rank correlation $\rho$", lambda j: f"{summ[j]['rho']:.3f}"),
        row("rate", r"\quad excluding the self-scored row",
            lambda j: f"{summ[j]['rho_ex_self_graded']:.3f}"),
        row("rate", "Maximum rank displacement", lambda j: str(summ[j]["max_disp"])),
        row("rate", "Models whose rank moves",
            lambda j: f"{summ[j]['moved']}/{len(PRETTY)}"),
        row("rate", "Largest move (points)",
            lambda j: f"${signed(summ[j]['largest_move_pp'], 1)}$"),

        ("table check header",
         r"\emph{Per-check agreement with the recorded verdicts.}"),
        row("check", "Agreement", lambda j: f"{100 * agr[j]['agreement']:.2f}\\%"),
        row("check", r"Cohen's $\kappa$", lambda j: f"{agr[j]['kappa']:.3f}"),
        row("check", r"Fail $\to$ pass flips",
            lambda j: str(agr[j]["false_to_true"])),
        row("check", r"Pass $\to$ fail flips",
            lambda j: str(agr[j]["true_to_false"])),
        row("check", f"Declined prompts (of {n(panel_total)})",
            lambda j: n(cov[j]["declined"])),
        row("check", "Non-unanimous ensembles",
            lambda j: f"{100 * health[j]['split'] / health[j]['n']:.2f}\\%"),
    ]

    validity = [
        ("forward reference", r"\S\ref{sec:judge-ablation}"),
        ("sonnet rho", f"($\\rho = {summ['sonnet5']['rho']:.3f}$)"),
        ("gemini36 rho", f"($\\rho = {summ['gemini36']['rho']:.3f}$)"),
    ]
    limits = [("forward reference", r"\S\ref{sec:judge-ablation}")]

    return {"body": body, "apx": apx, "validity": validity, "limits": limits}


def check_structure(d: dict, fail) -> None:
    """Claims the paper makes about the ranking that are not single numbers."""
    ranks = d["ranks"]
    cols = ("recorded",) + JUDGES

    # "the same four models occupy the top four places under every judge"
    tops = {c: frozenset(s for s, r in ranks[c].items() if r <= 4) for c in cols}
    if len(set(tops.values())) != 1:
        fail("top four", f"top-4 sets differ across judges: "
                         f"{ {c: sorted(v) for c, v in tops.items()} }")

    # "the bottom four are unchanged in both membership and order"
    bots = {c: tuple(sorted((r, s) for s, r in ranks[c].items() if r >= 11))
            for c in cols}
    if len(set(bots.values())) != 1:
        fail("bottom four", f"bottom-4 orderings differ: {bots}")

    # "displaces one model by three [positions]" -- exactly one, or the body
    # sentence should read differently.
    far = [s for s in ranks["recorded"]
           if abs(ranks["gemini36"][s] - ranks["recorded"][s])
           == d["summary"]["gemini36"]["max_disp"]]
    if len(far) != 1:
        fail("gemini36 displacement",
             f"body says one model moves by {d['summary']['gemini36']['max_disp']}, "
             f"artifact has {len(far)}: {sorted(far)}")

    # The whole framing rests on this: swapping the family perturbs the ranking
    # LESS than re-running the incumbent judge. \S\ref{sec:validity} states it on
    # pass rate and \S\ref{sec:judge-ablation} on mean score, so both have to
    # hold or one of the two sections is making a claim the artifact does not
    # support, and no single-number check would catch it.
    if not (d["summary"]["sonnet5"]["rho"] > d["summary"]["gemini36"]["rho"]
            and d["summary"]["sonnet5"]["max_disp"]
            < d["summary"]["gemini36"]["max_disp"]):
        fail("headline claim (pass rate)",
             "the validity paragraph claims the family swap perturbs the "
             "leaderboard less than the same-judge re-run, but sonnet5 no longer "
             "beats gemini36 on both rho and max displacement")
    if max(score_stats(d)[j]["max_disp"] for j in ("sonnet5", "opus5")) \
            > score_stats(d)["gemini36"]["max_disp"]:
        fail("headline claim (score)",
             "the body claims neither Claude judge displaces a model further on "
             "score than the incumbent's own resampling noise does, but one now "
             "does")

    # The body says claude-opus-5 raises "all fourteen models at once". Assert
    # the count before derive() formats it, so a re-export that broke the claim
    # fails here with a sentence rather than downstream on a missing word.
    up = score_stats(d)["opus5"]["up"]
    if up != len(PRETTY):
        fail("opus lift breadth",
             f"body says claude-opus-5 scores every one of the {len(PRETTY)} "
             f"models above its recorded value; the artifact has {up}")

    # score_stats() stands in for fields the export never wrote. It is only
    # trustworthy if the two fields it CAN be checked against come back exact,
    # so check those before believing the rest of the score block.
    ssc = score_stats(d)
    for j in JUDGES:
        for field in ("rho", "largest_shift"):
            got, want = ssc[j][field], d["score_summary"][j][field]
            if abs(got - want) > 1e-9:
                fail("score recomputation",
                     f"{j} {field} recomputed from d['score'] is {got!r}, but "
                     f"the artifact's score_summary says {want!r}; the derived "
                     f"score rows cannot be trusted while these disagree")

    # The table's headline claim, and the one thing a reader takes from it:
    # substituting the judge family can hand first place to a different model.
    # Assert both halves -- that the recorded leader is the one the caption
    # names, and that opus5 is the only column that unseats it -- because if a
    # re-export moved the flip to another judge every number in the table would
    # still check out while the caption silently became false.
    lead = {c: min(d["score"][c], key=lambda m: -d["score"][c][m])
            for c in ("recorded",) + JUDGES}
    flipped = [j for j in JUDGES if lead[j] != lead["recorded"]]
    if lead["recorded"] != "opus_5" or flipped != ["opus5"]:
        fail("score leader",
             f"caption says opus_5 leads on score and only claude-opus-5 as "
             f"judge overturns it; artifact leaders are {lead}")

    # Panel (c) was dropped from the table but its identity is still asserted
    # in prose, in both the body and the appendix.
    g = d["gemini_3_7_decomposition"]
    got = (g["under"]["gemini36"]["judge_term"]
           + g["under"]["gemini36"]["rollout_term"]
           + g["under"]["gemini36"]["self_replication"])
    if abs(got - g["observed_gap"]) > 1e-9:
        fail("decomposition identity",
             f"terms sum to {got:.6f}, observed gap is {g['observed_gap']:.6f}")

    # The replay fidelity gate is the reason the columns are comparable at all.
    f = d["replay_fidelity"]
    if f["deterministic_match"] != f["deterministic_checks"] or f["fidelity"] != 1.0:
        fail("replay fidelity",
             f"paper claims an exact replay; artifact has "
             f"{f['deterministic_match']}/{f['deterministic_checks']}")

    # The recorded column must still be the published leaderboard.
    if d["rate"]["recorded"][SELF_SCORED] != d["gemini_3_7_decomposition"]["r02_recorded"]:
        fail("self-scored row",
             "the recorded column's gemini-3.7-flash rate is not the rollout the "
             "decomposition calls r02, so Table 2 and the appendix disagree")


def verify(show: bool = False) -> tuple[list[str], int]:
    """Return (mismatches, values checked). Also results_facts.py's entry point."""
    d = load()
    bad: list[str] = []

    def fail(label: str, msg: str) -> None:
        bad.append(f"{label}: {msg}")

    check_roster(d, fail)
    check_structure(d, fail)

    want = derive(d)
    reg = regions()
    total = 0
    for name, claims in want.items():
        text = reg[name]
        for label, needle in claims:
            total += 1
            if show:
                print(f"  {name:9s} {label:34s} {needle}")
            if needle not in text:
                fail(f"{name}/{label}", f"main.tex does not contain {needle!r}")
    return bad, total


def main() -> int:
    check = "--check" in sys.argv
    bad, total = verify(show=not check)
    if check:
        if bad:
            raise SystemExit("JUDGE ABLATION DRIFT\n  " + "\n  ".join(bad))
        print(f"  OK: all {total} judge-ablation values in main.tex reproduce "
              f"from src/report/judge_ablation.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
