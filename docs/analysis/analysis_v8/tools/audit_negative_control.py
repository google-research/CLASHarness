#!/usr/bin/env python3

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

"""Injection-based negative control for `audit_report.py`.

A green audit only means something if the audit can go red. This script proves
it can: for each mutation below it corrupts exactly one number (or one phrase)
in a committed artifact, re-runs the audit as a subprocess, and requires that

  1. the audit exits non-zero, and
  2. the *specific* check that is supposed to notice actually names itself in
     the failure list.

Requirement (2) is the part that matters. An audit can fail for the wrong
reason and still look like it caught the bug; without pinning the expected
label, a mutation that trips some unrelated invariant would be scored as a
success and the check under test would stay unproven.

Every mutation is reverted from an in-memory copy of the original bytes and the
file is verified byte-identical afterwards, so a crash mid-run cannot leave a
corrupted artifact behind (and the final restore is checked, not assumed).

The comparison is against the *baseline set of failures*, not against a green
audit: while the nine deep-dives still have unwritten prose the audit is
legitimately red, and a control that demanded green could not run at all.
Requiring the expected label to be newly introduced is also strictly stronger
than requiring a non-zero exit, because a mutation can no longer be scored as
caught by a failure that was already there.

Usage:
    python tools/audit_negative_control.py            # run every mutation
    python tools/audit_negative_control.py D7 E9      # only those prefixes
"""

import hashlib
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
AUDIT = HERE / "audit_report.py"
CAP = ROOT / "eval-results-capability-analysis.md"
MDR = ROOT / "cross-model-behavioral-comparison.md"
FTJ = ROOT / "failure_taxonomy.json"
# Section 5.3's inputs. The audit recomputes that section from sources the
# reporting tools do not share, so the sources are mutation targets too --
# a structural guard like the seed join or the cross-derivation agreement is
# unreachable from the rendered markdown and would otherwise stay unproven.
DXJ = ROOT / "design_axes.json"
AXJ = ROOT / "difficulty_axes.json"
STJ = ROOT / "cross-model-behavioral-comparison.stats.json"
DSL = HERE / "design_axis_slice.py"
BHR = HERE / "build_behavior_report.py"
ESF = HERE / "emit_solver_failures.py"
# The plugin sources. Pass D13 rebuilds the conflict registry by PARSING these
# modules, specifically so it shares no derivation with the two reporting tools
# (one imports the registry, one hand-copies it). That independence is only
# worth something if a change to the sources is shown to move the audit, so two
# plugin modules are mutation targets in their own right.
PLUGINS = ROOT.parents[2] / "clasharness" / "scenario" / "plugins"
PLG_DEADLOCK = PLUGINS / "deadlock.py"
PLG_DLOCK = PLUGINS / "distributed_lock.py"
# credential_lock lives in db_audit.py -- module name and plugin name differ for
# four of the 23, which is why the audit keys off the parsed `name=` and not the
# filename. It is the plugin 5.3 calls out as prompt-only, and the one whose two
# seeds are also the two with no hard mechanic at all, so making it emit effects
# is the single edit that moves both of those claims.
PLG_DBAUDIT = PLUGINS / "db_audit.py"
# The seed spec: the archetype map 5.3's footnote contrasts with the plugin
# specs, and the three difficulty knobs 5.3 reports as built-but-never-set.
VSPEC = ROOT.parents[2] / "clasharness" / "scenario" / "conflict_vlm_spec.py"
SEEDS_DIR = (ROOT.parents[2] / "clasharness" / "data" / "scenario_seeds"
             / "conflict_vlm_v1")
SEED_DHS = SEEDS_DIR / "multi_tab_slot_monitoring.json"

SLUG = {"flash": "gemini-3-5-flash", "flash36": "gemini-3-6-flash",
        "pro": "gemini-3-1-pro", "opus": "claude-opus-4-8",
        "opus5": "claude-opus-5", "sonnet": "claude-sonnet-5",
        "sol": "gpt-5-6-sol", "luna": "gpt-5-6-luna", "terra": "gpt-5-6-terra"}
DD = {k: ROOT / f"{s}-solver-failures.md" for k, s in SLUG.items()}
TRACKED = ([CAP, MDR, FTJ, DXJ, AXJ, STJ, DSL, BHR, ESF, PLG_DEADLOCK,
            PLG_DLOCK, PLG_DBAUDIT, VSPEC, SEED_DHS] + list(DD.values()))

def swap_first_two_appendix_rows(text):
    """Reorder Appendix A without changing any value in it.

    Corrupting a field mean to break the sort order also breaks the field-mean
    check, so it cannot prove the sort check independently. Swapping two rows
    leaves every cell correct and violates only the ordering invariant.
    """
    head, _, tail = text.partition("<!-- APPENDIX-A-BEGIN -->")
    rows = [ln for ln in tail.splitlines() if ln.startswith("| `")]
    a, b = rows[0], rows[1]
    return head + "<!-- APPENDIX-A-BEGIN -->" + tail.replace(
        a + "\n" + b, b + "\n" + a, 1)


def drop_every_opus5_index_link(text):
    """Remove *all* links to one deep-dive, not just the first.

    The document indexes each deep-dive twice (section 4 and Cross-references),
    so a single-occurrence rewrite leaves the check satisfied and the mutation
    invisible.
    """
    return text.replace("./claude-opus-5-solver-failures.md",
                        "./claude-opus-5-solver-failure.md")


def clear_prose(*slots):
    """Blank the named prose slots of a deep-dive.

    The prose gate is the one check that fires on something *missing*, so
    corrupting a value cannot exercise it. Emptying slots in a document whose
    prose is written, and requiring the gate to name exactly those slots, is
    the only test that proves it is sensitive rather than permanently silent.
    """
    def f(text):
        for slot in slots:
            b = f"<!-- PROSE-{slot}-opus5-BEGIN -->"
            e = f"<!-- PROSE-{slot}-opus5-END -->"
            head, rest = text.split(b, 1)
            _stale, tail = rest.split(e, 1)
            text = f"{head}{b}\n\n\n\n{e}{tail}"
        return text
    return f


def write_prose(slot, body):
    """Replace the body of one prose slot, leaving the other nine alone."""
    def f(text):
        b = f"<!-- PROSE-{slot}-opus5-BEGIN -->"
        e = f"<!-- PROSE-{slot}-opus5-END -->"
        head, rest = text.split(b, 1)
        _stale, tail = rest.split(e, 1)
        return f"{head}{b}\n\n{body}\n\n{e}{tail}"
    return f


REAL = ("Its per-check accuracy of 61.7% sits second of nine, and the 1,342 "
        "failed units are the smallest but one in the field, which is the "
        "shape of a model that is losing units rather than whole tasks.")

def forge_foreign_sentinel(text):
    """Retag one block with a different model's key, as a copy-paste would."""
    return text.replace("SF7-opus5-", "SF7-sonnet-")


def swap_first_two_rows(tag):
    """Reorder two table rows inside one block without changing a value.

    Corrupting a number to break a sort order also breaks that number's own
    check, so it cannot prove the ordering invariant independently.
    """
    def f(text):
        i = text.index(f"<!-- {tag}-BEGIN -->")
        j = text.index(f"<!-- {tag}-END -->")
        body = text[i:j]
        rows = [l for l in body.split("\n") if l.startswith("| `")]
        a, b = rows[0], rows[1]
        return text[:i] + body.replace(a + "\n" + b, b + "\n" + a, 1) + text[j:]
    return f


def drop_last_row(tag):
    """Delete a whole row, which no per-row check can see."""
    def f(text):
        i = text.index(f"<!-- {tag}-BEGIN -->")
        j = text.index(f"<!-- {tag}-END -->")
        body = text[i:j]
        rows = [l for l in body.split("\n") if l.startswith("| `")]
        return text[:i] + body.replace(rows[-1] + "\n", "", 1) + text[j:]
    return f


def drop_row(tag, label):
    """Delete one *named* row, so the deletion lands where the sibling
    value mutations land and the two are comparable."""
    def f(text):
        i = text.index(f"<!-- {tag}-BEGIN -->")
        j = text.index(f"<!-- {tag}-END -->")
        body = text[i:j]
        row = next(l for l in body.split("\n") if l.startswith(f"| `{label}`"))
        return text[:i] + body.replace(row + "\n", "", 1) + text[j:]
    return f


def swap_rows(tag, a, b):
    """Swap two *named*, adjacent rows without touching a single value.

    Aimed at orderings whose tie-break is the row label: picking two rows that
    agree on the primary sort key means an ordering check that only reproduced
    that key would accept the result, and only one that reproduces the
    tie-break can fail it.
    """
    def f(text):
        i = text.index(f"<!-- {tag}-BEGIN -->")
        j = text.index(f"<!-- {tag}-END -->")
        body = text[i:j]
        ra = next(l for l in body.split("\n") if l.startswith(f"| `{a}`"))
        rb = next(l for l in body.split("\n") if l.startswith(f"| `{b}`"))
        return text[:i] + body.replace(ra + "\n" + rb, rb + "\n" + ra, 1) + text[j:]
    return f


def edit_json(mutate):
    """Parse, mutate one field, re-serialise.

    A textual substitution into a pretty-printed dump of tens of thousands of
    lines would have to pin the surrounding whitespace to stay unique, and a
    near-miss would corrupt a neighbouring model's numbers instead of the
    intended one. Going through the parser addresses exactly one field. The
    re-serialised layout does not matter: the audit parses these files, and it
    is the original bytes that get restored afterwards.
    """
    def f(text):
        d = json.loads(text)
        mutate(d)
        return json.dumps(d)
    return f


def bump(*path, by=1):
    """Add `by` to the integer at a nested key path."""
    def m(d):
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] += by
    return edit_json(m)


def orphan_first_task(text):
    """Rename one task's scenario so it joins to no seed.

    Every task in the corpus is generated from one of the 20 seeds, so the
    join is total by construction and nothing in the rendered text can make it
    fail. Only the dump can.
    """
    return edit_json(
        lambda d: d["tasks"][0].__setitem__("scen", "no_such_seed_v01"))(text)


# (id, target file, needle, replacement, expected substring in the failure list)
#
# `needle` may instead be a callable taking the file text and returning the
# mutated text, for mutations that cannot be expressed as one substitution.
# Otherwise only the FIRST occurrence is replaced, which is deliberate: a single
# corrupted cell is a sharper test than a global rewrite.
MUTATIONS = [
    # --- Pass A. The dollar columns had NO mutation coverage at all until the
    # 2026-07-30 terra/luna reprice shipped stale through a fully green audit.
    # The reason it got through is structural: A2 only compares the md against
    # stats.json, and both descend from the same PRICING table, so a wrong rate
    # is copied faithfully into both and neither notices. The A2b mutations pin
    # the rate itself; the first two prove the long-standing A2 cells can go red.
    ("A2-costtask", MDR, "| 3.2318 | 14.05 |", "| 3.2319 | 14.05 |",
     "S1 sol: $/task"),
    ("A2-costsolved", MDR, "| 3.7420 | 14.77 |", "| 3.7420 | 14.78 |",
     "S1 opus-5: $/solved"),

    # Exactly the regression that shipped: terra and luna carrying their
    # pre-cut rates. Reinstating those rates must now turn the audit red.
    ("A2b-terrarate", BHR, '"terra":   {"in": 2.00,', '"terra":   {"in": 2.50,',
     "A2b terra: PRICING input rate"),
    ("A2b-lunarate", BHR, '"luna":    {"in": 0.20,', '"luna":    {"in": 1.00,',
     "A2b luna: PRICING input rate"),
    # An Anthropic entry carries no rate-shape constraint, so this isolates the
    # golden-rate pin from the ratio invariants that the two above also trip.
    ("A2b-sonnetrate", BHR, '"sonnet":  {"in": 2.00, "out": 10.00}',
     '"sonnet":  {"in": 2.50, "out": 10.00}',
     "A2b sonnet-5: PRICING input rate"),
    # The mirror image: `in` is left alone so the golden pin still passes and
    # only the tier shape breaks. Anchored on the trailing comma because
    # 3.1-pro carries the identical long_in/long_out pair, closing-brace apart.
    ("A2b-terratier", BHR, '"long_in": 4.00, "long_out": 18.00,',
     '"long_in": 4.00, "long_out": 20.00,',
     "A2b terra: long_out/out"),
    ("A2b-lunathresh", BHR, '"long_threshold": 128_000, "long_in": 0.40,',
     '"long_threshold": 132_000, "long_in": 0.40,',
     "A2b luna: long_threshold"),

    # Bolding marks the cheapest and dearest per solved task, and A2 strips **
    # before comparing, so these three are invisible to every other check.
    ("A2b-unbold", MDR, "| 0.1138 | **0.64** |", "| 0.1138 | 0.64 |",
     "A2b luna: $/solved cell"),
    ("A2b-boldextra", MDR, "| 1.0608 | 5.30 |", "| 1.0608 | **5.30** |",
     "A2b terra: $/solved cell"),
    ("A2b-boldtask", MDR, "| 4.4098 | 18.50 |", "| **4.4098** | 18.50 |",
     "A2b opus-4.8: $/task cell"),

    # The four derived dollar claims in the section-1 bullets. None of them is
    # a table cell, so A2 never looked at any of them -- which is how a stale
    # luna figure sat in two of these bullets across every previous green run.
    ("A2b-ratio", MDR, "57.5\u00d7 luna's 0.64", "52.5\u00d7 luna's 0.64",
     "A2b: the 3.5-flash bullet must read"),
    ("A2b-cheapest", MDR, "solved task (0.64) at 17.7% pass",
     "solved task (0.65) at 17.7% pass",
     "must quote luna at 0.64 per solved task"),
    ("A2b-dearest", MDR, "(4.4098) and does not lead",
     "(4.4090) and does not lead",
     "must quote opus-4.8 at 4.4098 per task"),
    ("A2b-spread", MDR, "from 7.8\u00d7 to 38.8\u00d7 per task",
     "from 7.8\u00d7 to 34.8\u00d7 per task",
     "A2b: the orthogonality bullet must read 'from"),

    ("D0-rows", FTJ, '"rows": 600', '"rows": 599',
     "D0:"),
    ("D0-okfalse", FTJ, '"ok_false": 0', '"ok_false": 1',
     "has ok=False check records"),

    ("D1-score", CAP, "| **0.6328** |", "| **0.6329** |",
     "D1 opus5: avg_score"),
    ("D1-order", CAP, "| **25.3% (152/600)** |", "| **12.3% (152/600)** |",
     "capability ladder is not ordered by pass rate"),
    ("D1-percheck", CAP, "| 61.7% |", "| 63.7% |",
     "D1 opus5: per-check"),
    ("D1-gap", CAP, "leads sol by 0.0052", "leads sol by 0.0053",
     "opus-5 - sol avg-score gap"),
    ("D1-spread", CAP, "is a 0.3033 avg-score range", "is a 0.3133 avg-score range",
     "Gemini avg-score spread"),

    ("D2-names", CAP, "| `any_field_equals` | 23 |", "| `any_field_equals` | 22 |",
     "D2 any_field_equals: distinct names"),
    ("D2-checks", CAP, "| `field_semantic_match` | 2 | 2,137 |",
     "| `field_semantic_match` | 2 | 2,136 |",
     "D2 field_semantic_match: check count"),
    ("D2-total", CAP, "| **28** | **3,760** |", "| **28** | **3,761** |",
     "D2: total checks"),
    ("D2-roles", CAP, "3,284 required / 476 diagnostic",
     "3,283 required / 476 diagnostic",
     "D2: role split"),
    ("D2-largest", CAP, "at 82 checks", "at 83 checks",
     "D2: side-effect name #1"),
    ("D2-tail", CAP, "ten names have fewer than 15", "nine names have fewer than 15",
     "side-effect names have <15 checks"),

    ("D3-cell", CAP, "| 2,132 | **1,070** |", "| 2,132 | **1,071** |",
     "D3 flash: fail_closed"),
    ("D3-headline", CAP, "1,070 of\n2,132, or 50.2%", "1,070 of\n2,132, or 50.3%",
     "D3: headline fail-closed pct"),
    ("D3-partic", CAP, "483 of\nthem loops", "484 of\nthem loops",
     "D3: the participation triple"),

    ("D4-episodes", CAP, "| **7** |", "| **8** |",
     "D4: hard-failure episode count"),
    ("D4-denom", CAP, "| medium / gemini-3.5-flash | 3,278 |",
     "| medium / gemini-3.5-flash | 3,279 |",
     "D4 medium/flash: required units"),

    ("D5-required", CAP, "| 1,194 | 0 |", "| 1,195 | 0 |",
     "D5 F1: required"),
    ("D5-diag", CAP, "| 346 | 476 |", "| 346 | 475 |",
     "D5 F3: diagnostic"),
    ("D5-identity", CAP, "semantic 2,137 = F1 1,194", "semantic 2,137 = F1 1,195",
     "D5: the semantic refinement identity"),

    ("D6-count", CAP, "| gemini-3.5-flash | 894/0 |", "| gemini-3.5-flash | 895/0 |",
     "D6 flash/F1"),
    ("D6-total", CAP, "| 28/0 | 2,321 |", "| 28/0 | 2,322 |",
     "D6 flash: failed_total"),
    ("D6-rate", CAP, "| gemini-3.5-flash | 75.1 |", "| gemini-3.5-flash | 76.1 |",
     "D6 flash/F1: rate"),
    ("D6-share", CAP, "required units, 36.4%)", "required units, 37.4%)",
     "D6: the F1 share"),

    ("D7-rho9", CAP, "| F1 content_msg | \u22120.80 |", "| F1 content_msg | \u22120.81 |",
     "D7 F1: rho9"),
    ("D7-rho8", CAP, "| \u22120.80 | \u22120.71 |", "| \u22120.80 | \u22120.72 |",
     "D7 F1: rho8"),
    ("D7-spread", CAP, "| **9.3** | **9.3** |", "| **9.4** | **9.3** |",
     "D7 F5: spread9"),
    ("D7-invariant", CAP, "F3 and F6 are spread-invariant too",
     "F2 and F6 are spread-invariant too",
     "prose names the wrong co-invariant buckets"),
    ("D7-shrink", CAP, "spreads shrink on removal \u2014 F1, F2, F4 \u2014",
     "spreads shrink on removal \u2014 F1, F2, F3 \u2014",
     "prose names the wrong shrinking buckets"),
    ("D7-interior", CAP, "(13.3%, between gpt-5.6-sol's 9.0%",
     "(13.4%, between gpt-5.6-sol's 9.0%",
     "D7: the F3 interior figures"),
    ("D7-f5rank", CAP, "| 5 | **claude-opus-5** | 63.1 | **1** |",
     "| 5 | **claude-opus-5** | 63.1 | **2** |",
     "F5 rank 5 overall rank"),
    ("D7-ordinal", CAP, "The overall winner is **fifth** on F5",
     "The overall winner is **fourth** on F5",
     "on F5, not as written"),

    ("D8-low", CAP, "| **F3 grounding** | 693 |", "| **F3 grounding** | 694 |",
     "D8 F3/low"),
    ("D8-pct", CAP, "| **\u221248.9%** |", "| **\u221248.8%** |",
     "D8 F3: lo->hi"),
    ("D8-mono", CAP, "| **8/9** |", "| **7/9** |",
     "D8 F3: monotone count"),
    ("D8-net", CAP, "sonnet-5 \u2212112", "sonnet-5 \u2212113",
     "D8: F3 net delta"),

    ("D9-zerorow", CAP, "| `multi_tab_slot_monitoring_v02` | 0.438 |",
     "| `multi_tab_slot_monitoring_v02` | 0.439 |",
     "field mean"),
    ("D9-easiest", CAP, "field mean **0.779** and 55/90",
     "field mean **0.779** and 56/90",
     "easiest scenario line is stale"),
    ("D9-zerocount", CAP, "gemini-3.1-pro\nin 23,", "gemini-3.1-pro\nin 24,",
     "per-model zero-pass sentence disagrees"),
    ("D9-h2hwin", CAP, "| claude-opus-5 | 340 |", "| claude-opus-5 | 341 |",
     "D9 opus5: h2h wins"),
    ("D9-h2hnet", CAP, "| 8 | **+208** |", "| 8 | **+209** |",
     "D9 opus5: h2h net"),
    ("D9-split", CAP, "is 30\u201330 with zero ties", "is 31\u201329 with zero ties",
     "opus-5/sol split line is stale"),
    ("D9-idg", CAP, "passes it 4 times", "passes it 5 times",
     "passes it_deployment_gate_v02 4 times"),

    ("D10-mean", CAP, "| 0.246 | 0.19<br>0 |", "| 0.246 | 0.18<br>0 |",
     "D10 warehouse_robotics_safety_halt_v03/flash: mean"),
    ("D10-passed", CAP, "| 0.312 | 0.17<br>0 |", "| 0.312 | 0.17<br>1 |",
     "D10 distributed_lock_shared_resource_v02/flash: passed"),
    ("D10-bold", CAP, "**0.38**<br>1", "0.38<br>1",
     "bold marker disagrees"),
    ("D10-sort", CAP, swap_first_two_appendix_rows, None,
     "not sorted hardest-first"),

    ("D11-gpt55", CAP, "## Cross-references", "## Cross-references\n\ngpt-5.5\n",
     "gpt-5.5 is reported somewhere"),
    ("D11-index", CAP, drop_every_opus5_index_link, None,
     "does not index claude-opus-5"),
    ("D11-bare", CAP, "## Cross-references",
     "## Cross-references\n\nthe opus model\n",
     "bare family word"),
    # The on-disk check iterates a fixed slug list, so no edit to the text can
    # exercise it; the file itself has to go away.
    ("D11-ondisk", DD["opus5"], None, None,
     "indexed deep-dive claude-opus-5-solver-failures.md does not exist",
     "unlink"),

    # ---- D12: the main report's index of the companion documents ---------
    # Each mutation is shaped to isolate one D12 check: the count mutation
    # keeps the slug string present, and the capability mutation lands on an
    # extension the link scanner ignores, so neither borrows a neighbour's
    # failure.
    ("D12-section", MDR, "## Cross-references", "## Companion documents",
     "main report has no Cross-references section"),
    ("D12-cap", MDR, "](./eval-results-capability-analysis.md)",
     "](./eval-results-capability-analysis.htm)",
     "main report does not link the capability analysis"),
    ("D12-index", MDR, "](./gpt-5-6-terra-solver-failures.md)",
     "](./gpt-5-6-terra-solver-failure.md)",
     "main report does not index gpt-5-6-terra"),
    ("D12-count", MDR, "[terra](./gpt-5-6-terra-solver-failures.md)",
     "terra `./gpt-5-6-terra-solver-failures.md`",
     "does not link exactly 9 deep-dives"),
    # The exact leak the retired A9 token ban existed to stop: the reference
    # corpus's tenth link, to a model v8 has no logs for.
    ("D12-deadlink", MDR, "  - *OpenAI*: [sol]",
     "  - *OpenAI*: [gpt-5.5](./gpt-5-5-solver-failures.md) · [sol]",
     "link target does not exist: gpt-5-5-solver-failures.md"),
    # Lands on the first occurrence, in the intro, which a Cross-references
    # only scan would not have seen.
    ("D12-bodylink", MDR, "](./cross-model-behavioral-comparison.html)",
     "](./cross-model-behavioral-comparison-v7.html)",
     "link target does not exist: cross-model-behavioral-comparison-v7.html"),

    # ---- D13/D14: section 5.3, the designed difficulty axes --------------
    # 5.3 is the one section whose *grouping* can be wrong rather than just its
    # arithmetic, so the audit rebuilds it from the scenario seeds plus a
    # per-task dump and compares three independent derivations. The mutations
    # come in two kinds: corrupt the rendered markdown, and corrupt a source.
    # Only the second kind reaches the structural guards.
    ("D13-cell", CAP, "| gap_completion | challenge | 3 | 90 | 2.2% (0.26) |",
     "| gap_completion | challenge | 3 | 90 | 2.3% (0.26) |",
     "D13 gap_completion/3.5-flash"),
    # The 7 challenge tokens are plugins too, so each owns a row in *two* tables
    # under the same label. The audit indexes rows by label, so an index that
    # kept one row per label would verify the plugin table and silently skip the
    # challenge table. Corrupting the second copy is the only mutation that can
    # tell those two implementations apart.
    ("D13-cell2", CAP, "| gap_completion | 3 | 90 | 2.2% (0.26) |",
     "| gap_completion | 3 | 90 | 2.1% (0.26) |",
     "D13 gap_completion/3.5-flash"),
    # `category` is a column of the plugin table only. This mutation writes the
    # exact error the retired ARCHETYPE_CATEGORY map made: it filed
    # deadlock_dependency under inconsistency where the plugin declares
    # coordination.
    ("D13-cat", CAP, "| deadlock_dependency | coordination | 2 | 60 |",
     "| deadlock_dependency | inconsistency | 2 | 60 |",
     "D13 deadlock_dependency: category"),
    # seeds and tasks are separate cells, and reading a 60-task bucket as 60
    # independent trials is the misreading the seeds column was added to stop.
    # A seeds count that drifts while tasks stays right has to fail on its own.
    ("D13-seeds", CAP, "| credential_lock | coordination | 2 | 60 |",
     "| credential_lock | coordination | 3 | 60 |",
     "D13 credential_lock: seed count"),
    # Bold in these tables is generated as the row maximum. Moving it without
    # touching a value isolates the emphasis check from the value check, which
    # corrupting the maximum itself could not.
    ("D13-bold", CAP, "| 41.7% (0.70) | 31.7% (0.64) | **45.0% (0.71)** |",
     "| **41.7% (0.70)** | 31.7% (0.64) | 45.0% (0.71) |",
     "D13 duplicate_avoidance/opus-5: bold=True"),
    ("D13-delta", CAP, "| state_repair | 10 | 300 | -0.5 | +3.5 |",
     "| state_repair | 10 | 300 | -0.5 | +3.6 |",
     "D13 delta state_repair/3.6-flash"),
    # The 'below baseline' tally summarises the nine deltas beside it, so it can
    # drift out of step with a row whose own cells are all correct.
    ("D13-below", CAP, "| -4.7 | -6.0 | -8.3 | 9/9 |",
     "| -4.7 | -6.0 | -8.3 | 8/9 |",
     "D13 vision_decoy: 'below baseline'"),
    ("D13-baseline", CAP, "): 3.2% (0.30) | 20.2% (0.60)",
     "): 3.2% (0.30) | 20.3% (0.60)",
     "D13: the baseline line under the 5.3 tables is stale"),
    # The coverage paragraph counts *seeds* while every table counts tasks, and
    # conflating the two is the mistake this sentence exists to prevent.
    ("D13-coverage", CAP, "`multi_source_reasoning` by 12",
     "`multi_source_reasoning` by 11",
     "D13: coverage paragraph does not say `multi_source_reasoning` is carried "
     "by 12 seeds"),
    ("D13-slamean", CAP, "score (0.533)", "score (0.534)",
     "D13: sla_support_response field mean"),
    ("D13-asyncmax", CAP, "claude-opus-5's 13.3% is the",
     "claude-opus-5's 13.4% is the",
     "D13: the async field maximum is claude-opus-5"),
    # Emphasis only: the claim stays true and every number stays right, so this
    # can only be caught by the check that reads the claim itself. The audit
    # matches it against a line-flattened copy of 5.3; the needle here is the
    # on-disk text, which wraps mid-claim.
    ("D13-asyncone", CAP, "**`force_async_wait`\non exactly one**",
     "`force_async_wait`\non exactly one",
     "D13: the 'exactly one async seed' claim is gone"),
    # 5.3's plugin registry is derived three ways that share no code: the audit
    # parses the 23 plugin modules, design_axis_slice.py imports the live
    # registry, and build_behavior_report.py and emit_solver_failures.py each
    # carry a hand copy. The next six mutations break each leg of that
    # arrangement separately. A suite that only ever corrupted the rendered
    # markdown would leave every one of them unproven, which is how the retired
    # archetype version of this check managed to stay green while grouping the
    # report by a map the router never reads.
    #
    # (a) a hand copy moves, the sources do not. Both copies are pinned, and
    # each is pinned to the PARSED sources rather than to the other -- two hand
    # copies agreeing proves only that they were typed from the same place.
    ("D13-drift", BHR, '"deadlock_dependency": "coordination",',
     '"deadlock_dependency": "order",',
     "D13: tools/build_behavior_report.py PLUGIN_CATEGORY has drifted"),
    ("D13-esfdrift", ESF, '"scope_reconciliation": "main_supervised",',
     '"scope_reconciliation": "coordination",',
     "D13: tools/emit_solver_failures.py PLUGIN_CATEGORY has drifted"),
    # (b) the sources move, the hand copies do not. Same check, opposite
    # direction: a comparison accidentally written one-way passes one of these
    # two and fails the other, and only running both can tell.
    ("D13-spec", PLG_DEADLOCK, 'category="coordination",', 'category="order",',
     "D13: tools/build_behavior_report.py PLUGIN_CATEGORY has drifted"),
    # (c) design_axis_slice.py is the leg that must NOT hand-copy. If it ever
    # grows a literal it stops being an independent derivation, and D14's
    # three-way agreement quietly becomes a two-way one that cannot fail.
    ("D13-handcopy", DSL,
     "PLUGIN_CATEGORY = {s.name: s.category for s in PLUGIN_SPECS}",
     'PLUGIN_CATEGORY = {\n    "deadlock_dependency": "coordination",\n}',
     "D13: tools/design_axis_slice.py now hand-copies PLUGIN_CATEGORY"),
    # (d) 13 of 23 plugins gating real command execution is the split section
    # 5.3 reads as depth of enforcement. It is derived from the sources rather
    # than stated in any artifact, so no edit to the markdown can reach it.
    ("D13-mechanic", PLG_DLOCK,
     'emits_effects=("lock_acquired", "lock_released"),', "emits_effects=(),",
     "plugins declare emits_effects, expected 13"),
    # (e) every bucket in 5.3 rests on token -> plugin being 1:1. A single
    # plugin declaring an alias breaks that silently: the buckets still compute,
    # just over the wrong seeds, so the assumption is asserted rather than
    # assumed and the assertion needs proving.
    ("D13-alias", PLG_DLOCK, '      skill_families=("lock",),',
     '      taxonomy_tokens=("dlock",),\n      skill_families=("lock",),',
     "D13: plugins now declare taxonomy_tokens"),
    # Wide blast radius by construction -- an orphaned task drops out of every
    # bucket it fed, so most of 5.3 goes red with it. That is the point: the
    # join guard has to name *itself* among the wreckage, or a silently
    # dropped task would look like an ordinary arithmetic drift.
    ("D13-unmapped", AXJ, orphan_first_task, None,
     "D13: 1 scenario(s) join to no seed"),

    # ---- pass D13b: the four analysis subsections under the 5.3 tables ----
    # Everything above lands on a generated table cell. But the tables are the
    # part of 5.3 that cannot go stale -- they are re-emitted from the raw
    # trajectories on every run. The prose that READS them is typed, and it is
    # where the section commits to a conclusion: which bucket is hardest,
    # which lever is null, whether conflict category is an axis at all. That
    # asymmetry is the reason D13b exists, and it is why no mutation above can
    # stand in for the ones below: a correct table under a stale paragraph is
    # exactly the failure the generated cells cannot express.
    #
    # Three of these corrupt a source rather than the markdown. The claims
    # they carry -- credential_lock is prompt-only, the three knobs are set on
    # no seed, the archetype map disagrees on 5 of its 15 entries -- are
    # properties of the plugin specs and the seed corpus, not of the document,
    # so a prose-only control would leave them resting on nothing.

    # (i) the two unanimity rosters. Count, membership and the seed bound are
    # three separate readings of the same 27 buckets, and a roster can be
    # right while its headline count is wrong (or the reverse), so each moves
    # alone. Membership is deliberately not order-sensitive in the audit; the
    # mutation therefore substitutes a bucket rather than reordering one.
    ("D13b-negcount", CAP, "Five buckets put **all nine models",
     "Six buckets put **all nine models",
     "D13b: 5 buckets put all nine models below their own baseline"),
    ("D13b-negroster", CAP,
     "`duplicate_hold_stale`, `credential_lock`, `gap_completion` and "
     "`vision_decoy`.",
     "`state_repair`, `credential_lock`, `gap_completion` and `vision_decoy`.",
     "D13b: the unanimously negative buckets are"),
    ("D13b-negseeds", CAP, "Every one of them is carried by 3 seeds",
     "Every one of them is carried by 2 seeds",
     "D13b: the unanimously negative buckets top out at 3 seeds"),
    # The easy roster is the one that must include the `no_async` control arm.
    # Scoping either roster to the designed levers would silently drop it, so
    # the mutation removes exactly that member.
    ("D13b-easyroster", CAP,
     "0 of 9 below baseline: `no_async`, `workflow_completion`,",
     "0 of 9 below baseline: `state_repair`, `workflow_completion`,",
     "D13b: the unanimously easy buckets are"),

    # (ii) the four bullets. Each bullet states a superlative and then the
    # numbers behind it; the superlative is recomputed and the numbers are
    # matched, so both halves need moving separately.
    ("D13b-dhspool", CAP, "benchmark**: 3.3% pooled,", "benchmark**: 3.4% pooled,",
     "D13b: duplicate_hold_stale is 3.3% pooled"),
    ("D13b-dhszero", CAP, "with three models at 0.0%", "with two models at 0.0%",
     "D13b: 3 of the nine models score 0.0%"),
    ("D13b-dhsmax", CAP, "no model above 10.0%", "no model above 11.0%",
     "D13b: the best model on duplicate_hold_stale reaches 10.0%"),
    # Naming the wrong seed is the mistake a one-seed caveat exists to prevent,
    # and it is invisible to every arithmetic check in the section.
    ("D13b-dhsseed", CAP, "one seed (`multi_tab_slot_monitoring`)",
     "one seed (`clinical_trial_dosage_conflict`)",
     "D13b: duplicate_hold_stale rests on 1 seed(s)"),
    ("D13b-clrange", CAP, "pooled, \u22122.0 to \u221211.7)",
     "pooled, \u22122.0 to \u221211.8)",
     "D13b: credential_lock is 11.1% pooled"),
    # (source) "it declares no `emits_effects`" is a claim about db_audit.py.
    # Giving credential_lock a mechanic also moves the 13/10 split and the
    # mechanic-depth grouping, which is why the expected label pins the
    # prompt-only claim specifically rather than any failure at all.
    ("D13b-clmech", PLG_DBAUDIT, 'category="coordination",',
     'category="coordination",\n      emits_effects=("x",),',
     "D13b: credential_lock declares emits_effects"),
    # The unanimity threshold is read out of the sentence, not fixed in the
    # audit, so widening it in the prose has to re-derive the roster. Moving
    # the count and moving the threshold are different bugs: the first is a
    # miscount, the second silently changes what is being counted.
    ("D13b-unancount", CAP, "Exactly two plugins carried by 6",
     "Exactly three plugins carried by 6",
     "D13b: 2 plugins with 6+ seeds are unanimous"),
    ("D13b-unanthresh", CAP, "6 or more seeds return a unanimous",
     "3 or more seeds return a unanimous",
     "D13b: 3 plugins with 3+ seeds are unanimous"),
    # Order matters in this one and nowhere else in the section, because the
    # next clause says "The first is a near-constant".
    ("D13b-unanorder", CAP,
     "\u2014 `workflow_completion` and\n  `dependency_ordering`, 0 of 9 below "
     "baseline each.",
     "\u2014 `dependency_ordering` and\n  `workflow_completion`, 0 of 9 below "
     "baseline each.",
     "D13b: 5.3 names the well-sampled unanimous plugins"),
    ("D13b-wcseeds", CAP, "(18 of 20 seeds) and cannot discriminate",
     "(17 of 20 seeds) and cannot discriminate",
     "D13b: workflow_completion tags 18 of 20 seeds (near-constant claim)"),
    ("D13b-docoverage", CAP, "+0.8 to +6.1", "+0.8 to +6.2",
     "D13b: dependency_ordering covers 6 seeds"),
    ("D13b-noabest", CAP, "up to +29.7 for claude-sonnet-5)",
     "up to +29.7 for claude-opus-5)",
     "D13b: non_overlapping_assignment is 33.0% pooled, best +29.7 "
     "(claude-sonnet-5)"),

    # (iii) the category paragraph. Its whole argument is that three of the
    # six buckets are aliases for one plugin and the other three aggregate
    # without separating, so alias-ness, the split, the aggregating roster and
    # its spread are four claims that can each rot independently.
    ("D13b-catalias", CAP, "\u2014 `order` to `dependency_ordering`,",
     "\u2014 `controlled_deployment_gate` to `dependency_ordering`,",
     "D13b: 5.3 does not report category `order` as identical to "
     "`dependency_ordering`"),
    ("D13b-catcount", CAP, "Three of its six buckets are",
     "Four of its six buckets are",
     "D13b: 3 of 6 category buckets duplicate a single plugin"),
    # Widest-first ordering, and only here: the sentence pairs each name with
    # its seed count, so a re-ordered list misattributes both.
    ("D13b-catorder", CAP, "\u2014 `inconsistency` (9 seeds), `coordination` (6)",
     "\u2014 `coordination` (6), `inconsistency` (9 seeds)",
     "D13b: 5.3 names the aggregating categories"),
    ("D13b-catbelow", CAP, "(4, 6 and 6 of 9 below", "(5, 6 and 6 of 9 below",
     "D13b: the aggregating categories are"),
    ("D13b-catspan", CAP, "deltas inside \u22124.3 to +2.6)",
     "deltas inside \u22124.2 to +2.6)",
     "D13b: the aggregating categories span"),

    # (iv) depth of enforcement. This subsection reports a NULL, which is the
    # easiest kind of claim to leave stale: nothing about it looks wrong.
    ("D13b-mechsplit", CAP, "split into 13 that gate", "split into 12 that gate",
     "D13b: 13 of 23 plugins declare emits_effects"),
    ("D13b-depthsizes", CAP, "2 seeds with none, 12 with one, and 6 with two",
     "2 seeds with none, 11 with one, and 6 with two",
     "D13b: the mechanic-depth groups hold [2, 12, 6] seeds"),
    ("D13b-depthpool", CAP, "18.2% pooled pass rate respectively",
     "18.3% pooled pass rate respectively",
     "D13b: the mechanic-depth groups pool to"),
    ("D13b-null12", CAP, "18.7% against\n18.2%,", "18.7% against\n18.1%,",
     "D13b: the 1-versus-2 mechanic contrast is"),
    ("D13b-eachgroup", CAP,
     "with 3 of 9 models below their own baseline in each group",
     "with 4 of 9 models below their own baseline in each group",
     "D13b: the 1- and 2-mechanic groups put 3 and 3"),
    ("D13b-zeromech", CAP, "(11.1%, 9 of 9 below baseline)",
     "(11.1%, 8 of 9 below baseline)",
     "D13b: the 0-mechanic group is 11.1%, 9 of 9 below baseline"),
    # The confound is the paragraph's conclusion -- "no hard mechanic" and
    # "has credential_lock" are the same two seeds -- so naming the wrong seed
    # destroys the argument while leaving every number in it correct.
    ("D13b-confound", CAP,
     "its two seeds \u2014 `clinical_trial_dosage_conflict` and",
     "its two seeds \u2014 `multi_tab_slot_monitoring` and",
     "D13b: 5.3 does not name the confounded seed "
     "clinical_trial_dosage_conflict"),

    # (v) the three levers that do not produce difficulty, plus the two that
    # do. These are the sentences section 6 quotes, so a drift here escapes
    # 5.3 into the interpretation.
    ("D13b-wcnull", CAP, "+0.2 and +1.3", "+0.2 and +1.4",
     "D13b: workflow_completion is 540 tasks"),
    ("D13b-msrspread", CAP, "\u22121.1 to +0.8 points", "\u22121.2 to +0.8 points",
     "D13b: multi_source_reasoning spreads"),
    ("D13b-msrbelow", CAP, "Seven of nine models sit", "Six of nine models sit",
     "D13b: 7 of nine models sit below baseline"),
    ("D13b-dabest", CAP, "(gpt-5.6-sol 45.0%)", "(gpt-5.6-luna 45.0%)",
     "D13b: duplicate_avoidance is 60 tasks"),
    ("D13b-gcvd", CAP, "\u22120.5 to \u22128.3 points", "\u22120.5 to \u22128.4 points",
     "D13b: gap_completion/vision_decoy are"),
    ("D13b-itrange", CAP, "(8 of 9, \u22123.0 to +0.4)", "(8 of 9, \u22123.0 to +0.5)",
     "D13b: information_transfer is 8 of 9"),
    ("D13b-vgworst", CAP, "never worse\nthan \u22123.4)", "never worse\nthan \u22123.5)",
     "D13b: vision_graded is 8 of 9"),

    # (vi) statistical power. The bound and the coverage threshold are both
    # read out of the sentence by regex, so the mutation has to prove the
    # audit re-derives against the value it just read rather than a constant.
    ("D13b-powerbound", CAP, "more than 3.5 pass", "more than 3.6 pass",
     "D13b: the levers carried by 10+ seeds move a model up to 3.5 pass "
     "points"),
    ("D13b-sixset", CAP, "`state_repair`) together span",
     "`evidence_mismatch`) together span",
     "D13b: the six widest-coverage buckets are"),
    ("D13b-sixspan", CAP, "span only \u22123.4 to +3.5", "span only \u22123.4 to +3.4",
     "D13b: the six widest-coverage buckets span"),
    ("D13b-tally", CAP, "`gap_completion` 90 \u2014 while",
     "`gap_completion` 80 \u2014 while", "D13b: the coverage tally is"),
    ("D13b-solo", CAP, "Ten of the 23 plugins cannot",
     "Nine of the 23 plugins cannot",
     "D13b: 10 of 23 plugins rest on a single seed"),

    # (vii) the three built-but-unused knobs. "Set on 0 of the 20 seeds" is
    # the one claim in 5.3 that no artifact records -- it is an absence in the
    # seed corpus. Only setting one can show the audit would notice.
    ("D13b-deadknob", SEED_DHS, '"force_vision_locus": true,',
     '"force_vision_locus": true,\n  "force_decoy": true,',
     "D13b: force_decoy is set on 1 of 20 seeds"),
    # A line citation that has drifted is the failure a prose check cannot
    # see: :283 is a comment two lines above the knob it would be claiming.
    ("D13b-knobcite", CAP, "(`conflict_vlm_spec.py:282`",
     "(`conflict_vlm_spec.py:283`",
     "D13b: conflict_vlm_spec.py:283 does not declare force_round_type_plan"),
    ("D13b-knobtest", CAP, "a 569-line test", "a 570-line test",
     "is 569 lines, 5.3 says 570"),

    # (viii) the archetype footnote. It is the record of a grouping that was
    # wrong, which makes it the one part of 5.3 with no generated table
    # backing it at all -- every contrast in it is re-derived from
    # ARCHETYPE_CATEGORY against the plugin specs.
    ("D13b-archentries", CAP, "a 15-entry hand-written map",
     "a 16-entry hand-written map", "D13b: ARCHETYPE_CATEGORY has 15 entries"),
    ("D13b-archdis", CAP, "on 5 of its 15 entries", "on 4 of its 15 entries",
     "D13b: the archetype map disagrees with the plugin specs on 5 of 15 "
     "entries"),
    ("D13b-archroster", CAP,
     "`barrier_synchronization`, `deadlock_dependency` and",
     "`barrier_synchronization` and", "D13b: the archetype map files"),
    ("D13b-archnotplug", CAP, "(`inconsistency_expense`)", "(`inconsistency`)",
     "D13b: the archetype entries that are not plugins are"),
    ("D13b-archminp", CAP, "each activate 6 or more", "each activate 5 or more",
     "D13b: the archetype-less seeds activate as few as 6 plugins"),
    # (source) the same disagreement count from the other side. This one has
    # to report 4 of 15 where D13b-archdis reports 5 of 15: if the audit ever
    # hard-coded the count instead of deriving it, exactly one of the pair
    # would still pass, and running only one of them could not tell.
    ("D13b-archsource", VSPEC, '"distributed_lock": "inconsistency",',
     '"distributed_lock": "coordination",',
     "D13b: the archetype map disagrees with the plugin specs on 4 of 15 "
     "entries"),

    # D14 compares three derivations that share no code. Corrupting one of them
    # must be attributed to that one by name; a check that only compared two
    # would not be able to say which.
    ("D14-chain1", DXJ, bump("by_model", "opus5", "axes", "async",
                             "async_required", "passed"), None,
     "D14 design_axes.json/opus-5/async_required"),
    ("D14-chain2", STJ, bump("design_axes", "by_model", "opus5", "axes",
                             "vision", "vision_graded", "passed"), None,
     "D14 stats.design_axes/opus-5/vision_graded"),

    # ---- pass E: the nine deep-dives ------------------------------------
    # Mutations land in the opus-5 document unless the check under test is
    # about a different model's data.
    ("E0-title", DD["opus5"], "# claude-opus-5 \u2014 solver failure",
     "# claude-opus-4.8 \u2014 solver failure",
     "E0 claude-opus-5: title does not name"),
    ("E0-provider", DD["opus5"], "`vertex-anthropic`", "`bedrock-gpt`",
     "provenance does not name provider"),
    ("E0-oldstyle", DD["opus5"], "### Cross-references",
     "### Cross-references\n\n<!-- PROSE: signature -->\n",
     "old-style single-marker prose placeholder"),
    ("E0-foreign", DD["opus5"], forge_foreign_sentinel, None,
     "sentinels for other models present"),
    ("E0-prosecount", DD["opus5"], clear_prose("interpretation"), None,
     "E0 claude-opus-5: 1 prose slot(s) still unwritten: interpretation"),
    ("E0-prosegate", DD["opus5"],
     clear_prose("signature", "waits", "interpretation"), None,
     "E0 claude-opus-5: 3 prose slot(s) still unwritten: signature, waits, "
     "interpretation"),

    ("E1-pass", DD["opus5"], "| pass@budget | **25.3% (152/600)** |",
     "| pass@budget | **25.4% (152/600)** |",
     "E1 claude-opus-5: pass_pct"),
    ("E1-passed", DD["opus5"], "| pass@budget | **25.3% (152/600)** |",
     "| pass@budget | **25.3% (153/600)** |",
     "E1 claude-opus-5: passed in"),
    ("E1-rank", DD["opus5"], "| avg_score | **0.6328** | 1 | 0.5909 |",
     "| avg_score | **0.6328** | 2 | 0.5909 |",
     "E-rank avg_score/claude-opus-5"),
    ("E1-median", DD["opus5"], "| avg_score | **0.6328** | 1 | 0.5909 |",
     "| avg_score | **0.6328** | 1 | 0.5910 |",
     "E-med avg_score/claude-opus-5"),

    ("E2-int", DD["opus5"], "| near-miss tasks | **264** |",
     "| near-miss tasks | **265** |",
     "E2 claude-opus-5: near-miss tasks"),
    ("E2-float", DD["opus5"], "| mean steps | **9.65** |",
     "| mean steps | **9.66** |",
     "E2 claude-opus-5: mean steps"),
    ("E2-failclosed", DD["opus5"], "**20** of 2,137 (0.9%)",
     "**21** of 2,137 (0.9%)",
     "E2 claude-opus-5: fail_closed"),

    ("E3-rate", DD["opus5"], "| 325/0 | 515 | **63.1%** |",
     "| 325/0 | 515 | **63.2%** |",
     "E3 claude-opus-5/F5: rate"),
    ("E3-failed", DD["opus5"], "| 325/0 | 515 |", "| 326/0 | 515 |",
     "E3 claude-opus-5/F5: failed"),
    ("E3-what", DD["opus5"], "| **F6** window | time window |",
     "| **F6** window | time budget |",
     "E3 claude-opus-5/F6: description"),
    ("E3-total", DD["opus5"], "| **1,257/85** |", "| **1,257/86** |",
     "E3 claude-opus-5: total failed"),

    ("E4-avg", DD["opus5"], "| 0.6328 | 0 | 9.65 |", "| 0.6329 | 0 | 9.65 |",
     "E4 claude-opus-5/medium: avg_score"),
    ("E4-bucket", DD["opus5"], "| 597 | 224 | 39 | 48 | 325 | 24 |",
     "| 597 | 224 | 40 | 48 | 325 | 24 |",
     "E4 claude-opus-5/medium/F3"),
    ("E4-mcnemar", DD["opus5"], "(33 gained, 23 lost", "(34 gained, 23 lost",
     "E4 claude-opus-5: McNemar gained/lost"),

    ("E5-count", DD["opus5"], "| `message_send` | 1,504 |",
     "| `message_send` | 1,505 |",
     "E5 claude-opus-5/message_send: count"),
    ("E5-share", DD["opus5"], "| `wait` | 73 | 0.12 | 0.27 | 1.3% |",
     "| `wait` | 73 | 0.12 | 0.27 | 1.4% |",
     "E5 claude-opus-5/wait: share"),
    ("E5-sort", DD["opus5"], swap_first_two_rows("SF5-opus5"), None,
     "E5 claude-opus-5: rows are not sorted by count"),

    ("E6-stat", DD["opus5"], "| mean wait (s) | **86.9** |",
     "| mean wait (s) | **86.8** |",
     "E6 claude-opus-5: mean wait (s)"),
    ("E6-hist", DD["opus5"], "`wait 60` x47", "`wait 60` x48",
     "E6 claude-opus-5: histogram entries differ"),
    ("E6-truncation", DD["pro"], "of the histogram, ", "of the histogram, all ",
     "E6 gemini-3-1-pro: histogram sentence not found"),
    # The rank column reverses direction inside this one table, so the note
    # saying which way each row goes is load-bearing. Deleting it and lying
    # about the direction are different failures; a presence-only check would
    # only see the first.
    ("E6-note", DD["opus5"],
     "Rank 1 is the lowest value for wait actions, waits shorter than 30 s, "
     "and total simulated wait (s), and the highest value for mean wait (s), "
     "median wait (s), and longest wait (s).\n\n", "",
     "E6 claude-opus-5: rank-direction note is missing"),
    ("E6-notedir", DD["opus5"], "Rank 1 is the lowest value for wait actions",
     "Rank 1 is the highest value for wait actions",
     "E6 claude-opus-5: rank-direction note is missing"),

    ("E7-lift", DD["opus5"], "| lift | **1.48\u00d7** |", "| lift | **1.49\u00d7** |",
     "E7 claude-opus-5: lift"),
    ("E7-ci", DD["opus5"], "**[1.39, 1.57]**", "**[1.39, 1.58]**",
     "E7 claude-opus-5: lift CI"),
    ("E7-scored", DD["opus5"], "**3,284 / 600**", "**3,285 / 600**",
     "E7 claude-opus-5: checks/tasks"),

    ("E8-share", DD["opus5"], "| `none` | 152 | 25.3% |",
     "| `none` | 152 | 25.4% |",
     "E8 claude-opus-5/none: share"),
    ("E8-gating", DD["opus5"], "**378** involve a content check",
     "**379** involve a content check",
     "E8 claude-opus-5: content_involved"),
    ("E8-rowset", DD["opus5"], drop_last_row("SF8-opus5"), None,
     "E8 claude-opus-5: stop-reason rows"),

    ("E9-gap", DD["opus5"], "| 0.452 | 0.588 | \u22120.136 | 0/10 |",
     "| 0.452 | 0.588 | \u22120.137 | 0/10 |",
     "E9 claude-opus-5/it_deployment_gate_v02: gap"),
    ("E9-passes", DD["opus5"], "| 0.452 | 0.588 | \u22120.136 | 0/10 |",
     "| 0.452 | 0.588 | \u22120.136 | 1/10 |",
     "E9 claude-opus-5/it_deployment_gate_v02: passes"),
    ("E9-order", DD["opus5"], swap_first_two_rows("SF9-opus5"), None,
     "E9 claude-opus-5: positive gap table"),
    ("E9-zerolist", DD["opus5"],
     "`multi_tab_slot_monitoring_v02`, `it_deployment_gate_v02`.",
     "`multi_tab_slot_monitoring_v02`.",
     "E9 claude-opus-5: zero-pass list differs"),

    ("E10-quote", DD["opus5"], "> approval_recorded emitted for target",
     "> approval_recorded was not emitted for target",
     "E10 claude-opus-5/F5: quote is not verbatim"),
    ("E10-task", DD["opus5"], "`conflict_vlm_sla_support_response_v01_r01`",
     "`conflict_vlm_sla_support_response_v01_r02`",
     "E10 claude-opus-5/F6: task"),

    # E14. The per-document view of the same designed levers as D13, but built
    # from a different dump. Every value mutation lands on the `state_repair`
    # row on purpose: none of its figures are quoted in this document's prose,
    # so E12 stays out of the way and each failure is attributable to the one
    # E14 check under test.
    ("E14-passed", DD["opus5"], "| 300 | 72 | **24.0%** |",
     "| 300 | 73 | **24.0%** |",
     "E14 claude-opus-5/state_repair: 300/73"),
    ("E14-rate", DD["opus5"], "| **24.0%** | \u22121.3 |",
     "| **24.1%** | \u22121.3 |",
     "E14 claude-opus-5/state_repair: rate"),
    ("E14-delta", DD["opus5"], "| \u22121.3 | 0.6072 |", "| \u22121.4 | 0.6072 |",
     "E14 claude-opus-5/state_repair: delta"),
    # Fourth decimal: the column is printed at 4dp, so anything coarser would
    # not prove the comparison is exact rather than approximate.
    ("E14-avg", DD["opus5"], "| 0.6072 | 3 | 19.0% |", "| 0.6073 | 3 | 19.0% |",
     "E14 claude-opus-5/state_repair: avg"),
    ("E14-rank", DD["opus5"], "| 0.6072 | 3 | 19.0% |", "| 0.6072 | 4 | 19.0% |",
     "E-rank design/state_repair/claude-opus-5"),
    ("E14-median", DD["opus5"], "| 3 | 19.0% |", "| 3 | 19.1% |",
     "E-med design/state_repair/claude-opus-5"),
    # The two columns the plugin rewrite added. `category` is the cell the
    # retired archetype map got wrong -- state_repair is a challenge token, and
    # `inconsistency` is exactly what the old map called it -- and `seeds` is
    # the cell that tells the reader how thin the row's evidence is. Both sit
    # next to a correct `tasks` count, so neither is implied by the others.
    ("E14-cat", DD["opus5"], "| `state_repair` | challenge |",
     "| `state_repair` | inconsistency |",
     "E14 claude-opus-5/state_repair: category"),
    ("E14-seeds", DD["opus5"], "| `state_repair` | challenge | 10 | 300 |",
     "| `state_repair` | challenge | 11 | 300 |",
     "E14 claude-opus-5/state_repair: seed count"),
    ("E14-baseline", DD["opus5"],
     "| *all 600 tasks (baseline)* | \u2014 | 20 | 600 | 152 |",
     "| *all 600 tasks (baseline)* | \u2014 | 20 | 600 | 153 |",
     "E14 claude-opus-5: baseline 600/153"),
    ("E14-baseseeds", DD["opus5"],
     "| *all 600 tasks (baseline)* | \u2014 | 20 |",
     "| *all 600 tasks (baseline)* | \u2014 | 21 |",
     "E14 claude-opus-5: baseline seeds"),
    ("E14-rowgone", DD["opus5"], drop_row("SF11-opus5", "state_repair"), None,
     "E14 claude-opus-5: no `state_repair` row in SF11"),
    # The table claims to be sorted by coverage and 4.3 tells the reader to
    # read the `seeds` column first, so the order is a claim like any other.
    # These two rows both rest on one seed, so only a check that reproduces the
    # name tie-break can fail this; sorting by seed count alone would not.
    ("E14-roworder", DD["opus5"],
     swap_rows("SF11-opus5", "reversible_action_hold", "scope_reconciliation"),
     None, "E14 claude-opus-5: SF11 row order"),
    # The closing sentence is generated from the table above it, and it is the
    # part a reader actually quotes, so each of its claims -- the count, the
    # thin-evidence count, the sharpest lever, the list and the evidence base
    # of the sharpest -- is separately provable.
    ("E14-count", DD["opus5"], "**13 of 25 designed levers**",
     "**12 of 25 designed levers**",
     "E14 claude-opus-5: the summary does not say 13 of 25"),
    ("E14-thin", DD["opus5"],
     "**10 of those 13 are carried by 3 seeds or fewer**",
     "**9 of those 13 are carried by 3 seeds or fewer**",
     "E14 claude-opus-5: the summary does not say 10 of 13"),
    ("E14-sharpest", DD["opus5"], "The sharpest are `duplicate_hold_stale` at",
     "The sharpest are `credential_lock` at",
     "E14 claude-opus-5: the sharpest lever is duplicate_hold_stale"),
    # Dropping a name proves the list is compared as a whole rather than probed
    # name by name -- a wildcard probe would find every lever in the table
    # regardless of what the sentence says.
    ("E14-leverdrop", DD["opus5"],
     "`credential_lock` at \u221210.3 pp, `barrier_synchronization` at \u22128.7 pp,",
     "`barrier_synchronization` at \u22128.7 pp,",
     "E14 claude-opus-5: the summary lists"),
    # Reordering proves the same comparison is order-sensitive: the list is
    # sorted by depth, so a set-equality check would accept this.
    ("E14-leverorder", DD["opus5"],
     "`async_required` at \u221212.0 pp, `credential_lock` at \u221210.3 pp",
     "`credential_lock` at \u221210.3 pp, `async_required` at \u221212.0 pp",
     "E14 claude-opus-5: the summary lists"),
    # The named list is the five sharpest EXTENDED THROUGH the tie it lands
    # inside -- four buckets sit at -8.7 here. Truncating it back to a clean
    # five would look right to a check that only took below[:5], and would
    # silently tell the reader that three of the four are worse than the one
    # that was cut.
    ("E14-tiecut", DD["opus5"],
     "`distributed_lock` at \u22128.7 pp, and `scope_reconciliation` at \u22128.7 pp.",
     "and `distributed_lock` at \u22128.7 pp.",
     "E14 claude-opus-5: the summary lists"),
    # The tail sentence is the one that stops a headline delta from being read
    # as a finding: the sharpest lever here is a single seed.
    ("E14-evidence", DD["opus5"],
     "is measured on 30 tasks from 1 seed, where this model ranks 2 of 9",
     "is measured on 30 tasks from 1 seed, where this model ranks 3 of 9",
     "E14 claude-opus-5: the evidence-base sentence for duplicate_hold_stale"),
    # SF11 is built off failure_taxonomy.json while D13 works off
    # difficulty_axes.json; the audit reconciles the two before it trusts
    # either. Only a change to one of the dumps can put them out of step.
    ("E14-crosssource", FTJ, bump("cells", "medium", "models", "opus5",
                                  "scenarios", "access_request_approval_v01",
                                  "passed"), None,
     "E14 claude-opus-5: failure_taxonomy baseline"),

    # E12 needs prose to exist before it can judge it, so these write one slot.
    # The control here is REAL: the same sentence with only the number changed
    # must pass when the number is real and fail when it is not, which is what
    # separates a provenance check from a check that just dislikes digits.
    ("E12-real", DD["opus5"], write_prose("signature", REAL), None,
     "E12 claude-opus-5/signature", "quiet"),
    ("E12-fabricated", DD["opus5"],
     write_prose("signature", REAL.replace("1,342", "1,344")), None,
     "E12 claude-opus-5/signature: prose cites '1,344'"),
    ("E12-fabpct", DD["opus5"],
     write_prose("signature", REAL.replace("61.7%", "61.8%")), None,
     "E12 claude-opus-5/signature: prose cites '61.8'"),
    ("E12-short", DD["opus5"], write_prose("signature", "Good at this."), None,
     "E12 claude-opus-5/signature: prose is 13 chars"),
    ("E12-bare", DD["opus5"],
     write_prose("signature", REAL + " Overall the opus model leads."), None,
     "E12 claude-opus-5/signature: bare family word"),

    # E13. The disclosure has to track the data in both directions: absent when
    # it is needed is the bug that prompted it, and present when it is not
    # needed would be a fabricated caveat. gemini-3-1-pro is the only medium
    # document with unscored tasks at both efforts, so it carries the positive
    # cases and claude-opus-5 the negative one.
    ("E13-drop", DD["pro"], "That residual is 1 task whose episode",
     "Those tasks are described elsewhere. The residual is 1 case whose episode",
     "E13 gemini-3-1-pro: SF8 discloses unscored tasks nowhere"),
    ("E13-count", DD["pro"], "That residual is 1 task whose",
     "That residual is 2 tasks whose",
     "E13 gemini-3-1-pro: SF8 discloses 2 unscored, not 1"),
    ("E13-id", DD["pro"], "(`conflict_vlm_distributed_lock_shared_resource_v02_r02`)",
     "(`conflict_vlm_distributed_lock_shared_resource_v02_r03`)",
     "E13 gemini-3-1-pro: SF8 unscored task ids differ"),
    ("E13-scored", DD["pro"], "nor enters the 599 tasks scored",
     "nor enters the 600 tasks scored",
     "E13 gemini-3-1-pro: SF8 says 600 scored"),
    ("E13-sf4", DD["pro"], "being scored: 1 at medium, 5 at high.",
     "being scored: 1 at medium, 4 at high.",
     "E13 gemini-3-1-pro: SF4 unscored note"),
    ("E13-sf4drop", DD["pro"],
     "\nTasks whose episode produced no check results at all, counted as "
     "failures in the denominator above without being scored: 1 at medium, "
     "5 at high. See section 4.1.\n", "\n",
     "E13 gemini-3-1-pro: SF4 unscored note present=False"),
    ("E13-spurious", DD["opus5"],
     "Of 448 failed tasks,",
     "That residual is 1 task whose episode produced no check results at all "
     "(`conflict_vlm_x_v01_r01`). Nothing failed, so the stop reason is "
     "`none`; nothing was scored either, so such a task neither passes nor "
     "enters the 600 tasks scored in section 3.3.\n\nOf 448 failed tasks,",
     "E13 claude-opus-5: SF8 discloses unscored tasks but there are none"),
    ("E13-invariant", FTJ, '"none": 152', '"none": 153',
     "E13 claude-opus-5/medium: none 153 - passed 152 != 0 unscored"),
]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run_audit():
    r = subprocess.run([sys.executable, str(AUDIT)], capture_output=True,
                       text=True, cwd=ROOT)
    out = r.stdout + r.stderr
    hits = frozenset(ln[4:].strip() for ln in out.splitlines()
                     if ln.startswith("  - "))
    return r.returncode, out, hits


def matches(expect, hits):
    return any(expect in h for h in hits)


def main():
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    todo = [m for m in MUTATIONS
            if not only or any(m[0].startswith(p) for p in only)]

    baseline_sha = {p: sha(p) for p in TRACKED}
    rc, out, base_hits = run_audit()
    print(f"control: {out.splitlines()[0]}, "
          f"{len(base_hits)} pre-existing failure(s)\n")

    originals = {p: p.read_text() for p in TRACKED}
    bad = []
    for mut in todo:
        mid, path, old, new, expect = mut[:5]
        sense = mut[5] if len(mut) > 5 else "add"
        text = originals[path]
        if sense == "unlink":
            if matches(expect, base_hits):
                bad.append(f"{mid}: {expect!r} already fails at baseline")
                print(f"  SKIP  {mid:<16} expected label already red")
                continue
            aside = path.with_name(path.name + ".negctl")
            try:
                path.rename(aside)
                rc, out, hits = run_audit()
            finally:
                aside.rename(path)
            if not matches(expect, hits):
                bad.append(f"{mid}: removing {path.name} did not trip "
                           f"{expect!r}; got {sorted(hits - base_hits)[:3]}")
                print(f"  MISS  {mid:<16} deletion went unnoticed")
            else:
                print(f"  ok    {mid:<16} caught by {expect!r}")
            continue
        if callable(old):
            mutated = old(text)
        elif text.count(old) == 0:
            bad.append(f"{mid}: needle not present in {path.name}: {old!r}")
            print(f"  SKIP  {mid:<16} needle missing")
            continue
        else:
            mutated = text.replace(old, new, 1)
        if mutated == text:
            bad.append(f"{mid}: the mutation was a no-op")
            print(f"  SKIP  {mid:<16} mutation changed nothing")
            continue

        # An 'add' mutation must introduce the expected failure; a 'clear' one
        # must remove a failure that the baseline already has. Scoring against
        # the baseline set stops a mutation from being credited to a failure
        # that was there before it.
        if sense == "add" and matches(expect, base_hits):
            bad.append(f"{mid}: {expect!r} is already failing at baseline, so "
                       f"this mutation proves nothing")
            print(f"  SKIP  {mid:<16} expected label already red")
            continue
        if sense == "quiet" and matches(expect, base_hits):
            bad.append(f"{mid}: {expect!r} already fails at baseline, so a "
                       f"quiet result proves nothing")
            print(f"  SKIP  {mid:<16} expected label already red")
            continue
        if sense == "clear" and not matches(expect, base_hits):
            bad.append(f"{mid}: {expect!r} is not failing at baseline, so it "
                       f"cannot be cleared")
            print(f"  SKIP  {mid:<16} expected label not red at baseline")
            continue

        try:
            path.write_text(mutated)
            rc, out, hits = run_audit()
        finally:
            path.write_text(text)

        if sense == "quiet":
            # A provenance check that rejects every sentence is useless; this
            # proves valid prose passes it.
            if matches(expect, hits):
                bad.append(f"{mid}: valid prose still tripped {expect!r}: "
                           f"{sorted(h for h in hits - base_hits)[:2]}")
                print(f"  NOISY {mid:<16} valid input rejected")
            else:
                print(f"  ok    {mid:<16} valid input accepted, no {expect!r}")
            continue
        if sense == "clear":
            if matches(expect, hits):
                bad.append(f"{mid}: {expect!r} survived a mutation that "
                           f"satisfies it -- the check is stuck on")
                print(f"  STUCK {mid:<16} failure did not clear")
            else:
                print(f"  ok    {mid:<16} cleared {expect!r}")
            continue
        if rc == 0:
            bad.append(f"{mid}: the audit stayed GREEN under mutation")
            print(f"  MISS  {mid:<16} audit did not notice")
        elif not matches(expect, hits):
            new_hits = sorted(hits - base_hits)
            bad.append(f"{mid}: audit went red but not on {expect!r}; "
                       f"got {new_hits[:3]}")
            print(f"  WRONG {mid:<16} red, but on the wrong check")
        else:
            print(f"  ok    {mid:<16} caught by {expect!r}")

    for p, want in baseline_sha.items():
        if sha(p) != want:
            bad.append(f"{p.name} was NOT restored to its original bytes")
    rc, out, hits = run_audit()
    if hits != base_hits:
        bad.append("the audit's failure set changed after restore -- "
                   "artifacts are dirty")

    print(f"\nmutations run: {len(todo)}")
    if bad:
        print(f"NEGATIVE CONTROL FAILED: {len(bad)}")
        for b in bad:
            print("  -", b)
        return 1
    print("NEGATIVE CONTROL PASS: every mutation was caught by its own check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
