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

"""Regression audit for cross-model-behavioral-comparison.md.

!! SUPERSEDED -- DO NOT ACT ON THIS SCRIPT'S OUTPUT. It was written against the
!! nine-model / medium-effort corpus and has not been ported to the fourteen-
!! model / high-effort rebuild. On the current report it prints roughly 4,300
!! failures out of 14,300 checks, and they are overwhelmingly this script's
!! staleness rather than the report's: it still expects the model key
!! `gemini-3.1-pro-preview`, hard-codes the nine-model scored-unit band sizes
!! 192/258/150 and the 17.1pp pooled spread, hard-codes "8 of 9" and "3 of 9"
!! as the S8.1 monotonicity headlines, and caps every rank axis at 9. Its
!! `notes:` pass reports every shared passage as not-verbatim while
!! sync_notes_md.py --check reports all 26 shared passages and 16 generated
!! blocks matching -- so that pass is broken, not the notes.
!!
!! Porting it means re-deriving every pinned constant against the fourteen-
!! model panel at high effort. Until someone does that, a green run here would
!! be meaningless and a red one says nothing about the report. Fixing the two
!! or three most visible assertions in place would be worse than leaving it
!! obviously broken, because it would make the remainder look trustworthy.
!!
!! What is live and green on the current corpus, and what to run instead:
!!   tools/sync_notes_md.py --check     notes <-> md/html, verbatim
!!   paper/Google_Conflict_VLM/analysis/results_facts.py --check
!!                                      every number the paper asserts
!!   paper/Google_Conflict_VLM/src/verify_table_bodies.py
!!                                      every generated table body

Every number in the report has to come from one of three committed sources:

* ``cross-model-behavioral-comparison.stats.json`` -- emitted by
  ``build_behavior_report.py`` straight off the eval logs.
* ``behavior_notes.json``  -- the hand-authored prose, which the md must quote
  verbatim rather than paraphrase (the HTML renders the same strings, so any
  drift would make the two artefacts disagree).
* ``reasoning_tokens.json`` -- emitted by ``scan_reasoning_tokens.py``.

Pass A checks the md against those sources directly. Pass B re-derives, from
scratch, the tables that are *not* straight copies -- the catmix percentages,
the reasoning-token ratios, the difficulty splits, the field_contains
decomposition -- and checks the md's arithmetic rather than trusting it.

Anything a human typed by hand is a candidate for drift, which is the point:
an earlier revision of this report shipped a `passed` count back-derived from a
rounded percentage (19 written as 20), and pass A now catches exactly that.

Usage::

    python tools/audit_report.py

Exits non-zero on any failure.
"""

from __future__ import annotations

import collections
import glob
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD = os.path.join(HERE, "cross-model-behavioral-comparison.md")
NOTES = os.path.join(HERE, "behavior_notes.json")
STATS = os.path.join(HERE, "cross-model-behavioral-comparison.stats.json")
REASON = os.path.join(HERE, "reasoning_tokens.json")
CURATED = os.path.join(HERE, "curated_rounds.json")
AXES = os.path.join(HERE, "difficulty_axes.json")

md = open(MD, encoding="utf-8").read()
notes = json.load(open(NOTES, encoding="utf-8"))
S = json.load(open(STATS, encoding="utf-8"))
RT = json.load(open(REASON, encoding="utf-8"))
CR = json.load(open(CURATED, encoding="utf-8"))["rounds"]
AX = json.load(open(AXES, encoding="utf-8"))

# Typographic minus vs ASCII hyphen: the prose uses U+2212, stats.json uses "-".
mdn = md.replace("\u2212", "-")

K = ["flash", "flash36", "pro", "opus", "opus5", "sonnet", "sol", "luna", "terra"]
NAME = {"flash": "3.5-flash", "flash36": "3.6-flash", "pro": "3.1-pro",
        "opus": "opus-4.8", "opus5": "opus-5", "sonnet": "sonnet-5",
        "sol": "sol", "luna": "luna", "terra": "terra"}
ID = {"flash": "gemini-3.5-flash", "flash36": "gemini-3.6-flash",
      "pro": "gemini-3.1-pro-preview", "opus": "claude-opus-4-8",
      "opus5": "claude-opus-5", "sonnet": "claude-sonnet-5",
      "sol": "openai.gpt-5.6-sol", "luna": "openai.gpt-5.6-luna",
      "terra": "openai.gpt-5.6-terra"}

fails: list[str] = []
count = 0


def ck(cond, label):
    global count
    count += 1
    if not cond:
        fails.append(label)


def num_in(v, text=None):
    """True if v appears, tolerating -0 and trailing-zero formatting."""
    t = mdn if text is None else text
    cands = {str(v)}
    if isinstance(v, float):
        cands |= {f"{v:.1f}", f"{v:.2f}", f"{v:.3f}"}
    return any(c in t for c in cands)


def section(start, end):
    return md.split(start)[1].split(end)[0]


# ==========================================================================
# PASS A -- the md against its committed sources
# ==========================================================================

# ---- A1. verbatim parity with behavior_notes.json ------------------------
cs = notes["command_surface"]
ck(cs["intro"] in md, "notes: command_surface.intro not verbatim in md")
for i, b in enumerate(cs["bullets"]):
    ck(b in md, f"notes: command_surface.bullets[{i}] not verbatim in md")
ck(cs["kicker"] in md, "notes: command_surface.kicker not verbatim in md")
for k, v in notes["scenarios"].items():
    ck(v in md, f"notes: scenarios[{k}] not verbatim in md")
    ck(k in md, f"notes: scenario key {k} has no heading in md")

# ---- A2. section-1 main panel -------------------------------------------
for k in K:
    m = re.search(r"^\| `" + re.escape(ID[k]) + r"` \|(.+)$", md, re.M)
    ck(m is not None, f"S1: no row for {ID[k]}")
    if not m:
        continue
    cells = [c.strip() for c in m.group(1).split("|")]
    st = S[k]
    ck(cells[0].replace("**", "").startswith(f"{st['pass_pct']}%"),
       f"S1 {NAME[k]}: pass {cells[0]!r} != {st['pass_pct']}%")
    # The passed count must be quoted, never back-derived from the percentage.
    ck(f"({st['passed']})" in cells[0],
       f"S1 {NAME[k]}: passed count {cells[0]!r} != {st['passed']}")
    for idx, want, lbl in [
        (1, f"{st['avg_score']:.3f}", "score"),
        (5, f"{st['open_per_task']:.2f}", "opens"),
        (6, f"{st['write_per_task']:.2f}", "writes"),
        (7, str(st["zero_open"]), "zero_open"),
        (8, f"{st['cost']['cost_per_task']:.4f}", "$/task"),
        (9, f"{st['cost']['cost_per_solved']:.2f}", "$/solved"),
    ]:
        ck(cells[idx].replace("**", "") == want,
           f"S1 {NAME[k]}: {lbl} {cells[idx]!r} != {want}")

# ---- A2b. the cost column's provenance, and the prose derived from it ----
# A2 above compares the md's dollar cells against stats.json -- but BOTH descend
# from the same PRICING table in build_behavior_report.py, so agreement there
# proves the copy is faithful, not that the RATE is right. A stale rate ships
# clean through A2. That is not hypothetical: this report published gpt-5.6-terra
# and gpt-5.6-luna at their pre-2026-07-30 rates for a run that executed
# 2026-08-02/03 -- after the cut took effect, so the run was actually billed at
# the lower rates -- and all 14,377 checks passed. Everything below pins the
# rates themselves, and re-derives every dollar figure the section-1 prose quotes.
GENSRC = os.path.join(HERE, "tools", "build_behavior_report.py")


def parse_pricing(path):
    """The PRICING literal as WRITTEN in `path` -> {key: {field: float}}.

    Parsed from source text rather than imported, for the same reason D13 parses
    the plugin modules: importing would re-run the generator's own definition and
    could only ever agree with itself.
    """
    body = re.search(r"^PRICING\s*=\s*\{[ \t]*\n(.*?)^\}",
                     open(path, encoding="utf-8").read(), re.S | re.M)
    if not body:
        return None
    return {m.group(1): {f: float(v.replace("_", ""))
                         for f, v in re.findall(r'"(\w+)"\s*:\s*([0-9_.]+)',
                                                m.group(2))}
            for m in re.finditer(r'"(\w+)"\s*:\s*\{(.*?)\}', body.group(1), re.S)}


PRICING = parse_pricing(GENSRC)
ck(PRICING is not None,
   "A2b: could not parse a PRICING literal out of tools/build_behavior_report.py")
PRICING = PRICING or {}

# Short-context input rate per model, USD/1M tokens, verified 2026-08-05 against
# the first-party pricing pages. This is a deliberate second copy: it is the only
# thing in the audit that is not derived from the generator, so it is what makes
# a rate change visible instead of silent.
IN_RATE = {"flash": 1.50, "flash36": 1.50, "pro": 2.00,
           "opus": 5.00, "opus5": 5.00, "sonnet": 2.00,
           "sol": 5.00, "luna": 0.20, "terra": 2.00}
for _k, _want in IN_RATE.items():
    ck(_k in PRICING, f"A2b: PRICING has no entry for {NAME.get(_k, _k)}")
    if _k in PRICING:
        ck(abs(PRICING[_k].get("in", -1) - _want) < 1e-9,
           f"A2b {NAME[_k]}: PRICING input rate {PRICING[_k].get('in')} != "
           f"{_want} verified 2026-08-05")

# The four OpenAI entries are one rate SHAPE scaled by the input rate -- so the
# other seven fields are each redundant with `in`, and a single mistyped tier
# breaks the ratio. This catches a transcription slip that a golden copy of the
# same numbers cannot, because it does not share the copy's failure mode.
OPENAI_SHAPE = [("out", "in", 6.0), ("cached", "in", 0.10),
                ("cache_write", "in", 1.25), ("long_in", "in", 2.0),
                ("long_out", "out", 1.5), ("long_cached", "cached", 2.0),
                ("long_cache_write", "cache_write", 2.0)]
for _k in ("sol", "luna", "terra", "gpt55"):
    _pr = PRICING.get(_k, {})
    ck(bool(_pr), f"A2b: PRICING lost its {_k} entry")
    for _num, _den, _ratio in OPENAI_SHAPE:
        if _num not in _pr or _den not in _pr:
            continue
        ck(abs(_pr[_num] / _pr[_den] - _ratio) < 1e-9,
           f"A2b {_k}: {_num}/{_den} = {_pr[_num] / _pr[_den]:.6g}, expected "
           f"{_ratio} (OpenAI prices every tier off the input rate)")
    ck(_pr.get("long_threshold") == 128_000,
       f"A2b {_k}: long_threshold {_pr.get('long_threshold')} != 128000")

# Bolding in the section-1 dollar columns. A2 strips ** before comparing, so
# without this the emphasis could mark any row at all and nothing would notice.
S1ROW = {}
for _k in K:
    _m = re.search(r"^\| `" + re.escape(ID[_k]) + r"` \|(.+)$", md, re.M)
    if _m:
        S1ROW[_k] = [c.strip() for c in _m.group(1).split("|")]
CPS = {k: S[k]["cost"]["cost_per_solved"] for k in K}
CPT = {k: S[k]["cost"]["cost_per_task"] for k in K}
_cheap_solved = min(CPS, key=lambda k: CPS[k])
_dear_solved = max(CPS, key=lambda k: CPS[k])
_cheap_task = min(CPT, key=lambda k: CPT[k])
_dear_task = max(CPT, key=lambda k: CPT[k])
for _k in K:
    if _k not in S1ROW:
        continue
    _want_bold = _k in (_cheap_solved, _dear_solved)
    ck(S1ROW[_k][9].startswith("**") == _want_bold,
       f"A2b {NAME[_k]}: $/solved cell {S1ROW[_k][9]!r} is "
       f"{'not ' if _want_bold else ''}bolded; bold marks exactly the cheapest "
       f"({NAME[_cheap_solved]}) and dearest ({NAME[_dear_solved]}) per solved task")
    ck(not S1ROW[_k][8].startswith("**"),
       f"A2b {NAME[_k]}: $/task cell {S1ROW[_k][8]!r} is bolded; the $/task "
       f"column marks no extreme")

# The three derived dollar claims in the section-1 bullets, recomputed. None of
# these is a table cell, so A2 never saw them: the 3.5-flash ratio sentence and
# the orthogonality bullet both carried a luna figure that A2 could not check.
_ratio_solved = CPS[_dear_solved] / CPS[_cheap_solved]
ck(f"{CPS[_dear_solved]:.2f}** \u2014 {_ratio_solved:.1f}\u00d7 "
   f"{NAME[_cheap_solved]}'s {CPS[_cheap_solved]:.2f}" in md,
   f"A2b: the 3.5-flash bullet must read '{CPS[_dear_solved]:.2f}** \u2014 "
   f"{_ratio_solved:.1f}\u00d7 {NAME[_cheap_solved]}'s {CPS[_cheap_solved]:.2f}'")
ck(f"solved task ({CPS[_cheap_solved]:.2f}) at {S[_cheap_solved]['pass_pct']}% pass" in md,
   f"A2b: the orthogonality bullet must quote {NAME[_cheap_solved]} at "
   f"{CPS[_cheap_solved]:.2f} per solved task and "
   f"{S[_cheap_solved]['pass_pct']}% pass")
ck(f"({CPT[_dear_task]:.4f}) and does not lead" in md,
   f"A2b: the orthogonality bullet must quote {NAME[_dear_task]} at "
   f"{CPT[_dear_task]:.4f} per task as the dearest")
# The spread the bullet cites, at both ends of the 2026-07-30 cut. AWS: "prices
# ... for GPT-5.6 Luna are reduced by 80%", so the pre-cut luna rate -- and, the
# generator's cost being linear in the rate, its pre-cut $/task -- is 1/0.2 of
# today's. Deriving the historical figure keeps it checkable instead of frozen.
LUNA_CUT = 0.2
_spread_now = CPT[_dear_task] / CPT[_cheap_task]
_spread_pre = CPT[_dear_task] / (CPT[_cheap_task] / LUNA_CUT)
ck(_cheap_task == "luna",
   f"A2b: the cheapest model per task is now {NAME[_cheap_task]}, not luna; the "
   f"orthogonality bullet's pre-cut spread is derived assuming luna sits there")
ck(f"from {_spread_pre:.1f}\u00d7 to {_spread_now:.1f}\u00d7 per task" in md,
   f"A2b: the orthogonality bullet must read 'from {_spread_pre:.1f}\u00d7 to "
   f"{_spread_now:.1f}\u00d7 per task'")

# ---- A3. command mix / category totals ----------------------------------
for k in K:
    cm = S[k]["cmdmix"]
    for verb in ["asset list", "asset open", "file read", "message search",
                 "file create", "message send"]:
        n = cm.get(verb, 0)
        ck(f"{n:,}" in md or str(n) in md, f"S3 {NAME[k]}: {verb}={n} absent")
    t = S[k]["catmix_total"]
    ck(f"{t:,}" in md or str(t) in md, f"S2 {NAME[k]}: catmix_total {t} absent")

# ---- A4. stop reasons ----------------------------------------------------
for k, key, expect in [("flash", "reached max_steps", 510),
                       ("opus5", "reached max_steps", 0),
                       ("sonnet", "missing_required_side_effect", 293)]:
    got = S[k]["stops"].get(key, 0)
    ck(got == expect, f"S5 {NAME[k]}: {key} = {got} != {expect}")

# ---- A5. clustering ------------------------------------------------------
for k in K:
    c = S[k]["clustering"]
    ck(f"{c['per_check_pct']}%" in md, f"S7 {NAME[k]}: per_check {c['per_check_pct']} absent")
    ck(str(c["lift"]) in md, f"S7 {NAME[k]}: lift {c['lift']} absent")
    lo, hi = c["lift_ci"]
    ck(f"[{lo}, {hi}]" in md, f"S7 {NAME[k]}: ci [{lo}, {hi}] absent")
    ck(num_in(c["overdispersion"]), f"S7 {NAME[k]}: overdispersion absent")

# ---- A6. effort sweep ----------------------------------------------------
cells = S["effort"]["cells"]
for k in K:
    for e in ["low", "medium", "high"]:
        c = cells[f"{k}@{e}"]
        ck(str(c["pass_pct"]) in md, f"S9 {NAME[k]}@{e}: pass {c['pass_pct']} absent")
        ck(f"{c['mean_steps']:.2f}" in md, f"S9 {NAME[k]}@{e}: steps absent")
        ck(str(c["capped_pct"]) in md, f"S9 {NAME[k]}@{e}: capped absent")

cp = S["effort"]["cap_pass"]
ck(f"{cp['capped_n']:,}" in md, f"S6: capped_n {cp['capped_n']} absent")
ck(cp["capped_passed"] == 0, "cap_pass: capped_passed is no longer 0")
ck(f"{cp['uncapped_n']:,}" in md, f"S6: uncapped_n absent")
ck(f"{cp['uncapped_passed']:,}" in md, "S6: uncapped_passed absent")
ck(f"{100 * cp['uncapped_passed'] / cp['uncapped_n']:.2f}%" in md,
   "S6: uncapped pass rate absent")

for k in K:
    m = S["effort"]["mcnemar"][k]
    ck(f"{abs(m['delta_pp']):.1f}" in md, f"S9 {NAME[k]}: mcnemar delta absent")
    if m["significant"]:
        ck(str(m["lost"]) in md and str(m["gained"]) in md,
           f"S9 {NAME[k]}: lost/gained {m['lost']}/{m['gained']} absent")

# ---- A7. check-type split + the field_contains decomposition -------------
for k in K:
    cs2 = S[k]["check_split"]
    for kind in ["semantic_match", "field_contains", "other",
                 "field_contains_asset", "field_contains_content"]:
        v = cs2[kind]["graded_pass_pct"]
        ck(str(v) in md, f"S7 {NAME[k]}: check_split {kind} {v} absent")

# ---- A8. gap decomposition and strata ------------------------------------
gd = S["gap_decomposition"]
for label, blk in [("pooled", gd["pooled"]), ("stratified", gd["stratified"])]:
    for f2 in ["per_check", "clustering", "total"]:
        ck(num_in(blk[f2]), f"S7 gap {label}.{f2} = {blk[f2]} absent")
ov = gd["overall"]
ck(str(ov["a_lift"]) in md and str(ov["b_lift"]) in md, "S7 gap overall lifts absent")
ck(f"[{ov['paired_ci'][0]}, {ov['paired_ci'][1]}]" in md, "S7 gap paired_ci absent")

pm = S["clustering_strata"]["per_model"]
for k in ["flash", "flash36"]:
    p = pm[k]["peer"]
    ck(num_in(p["diff"]), f"S7 strata {NAME[k]}: diff {p['diff']} absent")
    ck(p["verdict"] in md, f"S7 strata {NAME[k]}: verdict absent")

# ---- A9. forbidden content ----------------------------------------------
# gpt-5.5 has no v8 logs; it may appear only in the single exclusion note.
ck(md.count("gpt-5.5") == 1, f"gpt-5.5 appears {md.count('gpt-5.5')}x (expected 1)")
ck("`gpt-5.5` is not in this corpus" in md, "gpt-5.5 exclusion note missing")
# This used to ban the strings "eval-results-capability-analysis" and
# "solver-failures.md" outright. That was a scaffolding guard from when no v8
# companion document existed yet: the risk then was pasting the reference
# corpus's cross-reference block, whose ten links included a
# gpt-5-5-solver-failures.md that v8 has no logs for. Those companions now
# exist, so the ban is obsolete and D12 replaces it with a stronger rule --
# every relative link in this document must resolve to a file on disk, which
# still rejects a pasted gpt-5.5 link, plus the ten companions must actually
# be indexed.

# A bare family word is ambiguous across releases ("opus" spans 4.8 and 5), so
# it is banned outside the header-mapping table, which quotes literal rendered
# column headers, and plural family references.
body = md
tbl = re.search(r"\| header \| model id \|.*?\n\n", md, re.S)
if tbl:
    body = md.replace(tbl.group(0), "")
for bare, allowed in [(" opus ", ["the two opus models", "The two opus models"]),
                      (" sonnet ", []), (" flash ", []), (" pro ", [])]:
    probe = body
    for a in allowed:
        probe = probe.replace(a, "")
    ck(probe.count(bare) == 0,
       f"bare family word {bare!r} appears {probe.count(bare)}x outside allowed contexts")

ck(not [c for c in md if "\u4e00" <= c <= "\u9fff"], "CJK characters present in md")

# ==========================================================================
# PASS B -- re-derive what the md computes by hand
# ==========================================================================

# ---- B1. section-2 catmix percentages ------------------------------------
COLS = ["asset_open", "asset_inspect", "file_read", "file_list", "file_write",
        "message_read", "message_send", "AGENT", "wait", "other"]
sec2 = section("## 2. Action-category mix", "## 3.")
for k in K:
    row = re.search(r"^\| " + re.escape(NAME[k]) + r" \| ([\d,*]+) \|(.+)$", sec2, re.M)
    ck(row is not None, f"S2: no row for {NAME[k]}")
    if not row:
        continue
    tot = int(row.group(1).replace("**", "").replace(",", ""))
    ck(tot == S[k]["catmix_total"], f"S2 {NAME[k]}: total {tot} != {S[k]['catmix_total']}")
    got_cells = [c.strip().replace("**", "").rstrip("%")
                 for c in row.group(2).split("|") if c.strip()]
    ck(len(got_cells) == len(COLS),
       f"S2 {NAME[k]}: {len(got_cells)} cells, expected {len(COLS)}")
    cm = S[k]["catmix"]
    for col, got in zip(COLS, got_cells):
        raw = (cm.get("agent_delegate", 0) + cm.get("agent_status", 0)
               if col == "AGENT" else cm.get(col, 0))
        want = round(100.0 * raw / tot, 1)
        ck(abs(float(got) - want) < 0.05,
           f"S2 {NAME[k]}.{col}: md {got} != recomputed {want}")

# ---- B2. reasoning tokens -------------------------------------------------
sec9 = section("## 9. The reasoning-effort sweep", "## 10.")
for disp, e in RT.items():
    for eff in ["low", "medium", "high"]:
        ck(f"{e[eff]['per_attempt']:.1f}" in sec9, f"S9.1 {disp}@{eff}: per_attempt absent")
        ck(f"{e[eff]['per_task']:,}" in sec9 or str(e[eff]["per_task"]) in sec9,
           f"S9.1 {disp}@{eff}: per_task absent")
        ck(f"{e[eff]['zero_pct']:.1f}%" in sec9, f"S9.1 {disp}@{eff}: zero_pct absent")
    ck(f"{e['mult_low_high']:.2f}\u00d7" in sec9, f"S9.1 {disp}: multiplier absent")

o5 = RT["opus-5"]["medium"]["per_task"]
ck(round(RT["3.5-flash"]["medium"]["per_task"] / o5) == 48, "S9.1: 48x ratio wrong")
ck(round(RT["opus-4.8"]["medium"]["per_task"] / o5) == 8, "S9.1: 8x ratio wrong")

# ---- B3. difficulty landscape, recomputed from difficulty_axes.json ------
#
# Section 8 is the one part of the report that is not a view onto stats.json:
# it joins the 600 specs against all nine runs itself. Presence checks ("is the
# string 26.0 somewhere in the section") cannot tell a right table from a wrong
# one, and two errors survived exactly that treatment -- a tool-family row that
# merged two disjoint scenario groups, and a monotonicity claim that contradicted
# the table printed directly above it. So every cell here is recomputed.
sec8 = section("## 8. Difficulty landscape", "## 9.")
# en-dash in band labels, U+2212 in negative diffs
sec8n = sec8.replace("\u2212", "-").replace("\u2013", "-")

AXT = AX["tasks"]
UBANDS = ("2-4", "5-6", "7+")
GBANDS = ("short", "mid", "long")


def uband(u):
    return "2-4" if u <= 4 else ("5-6" if u <= 6 else "7+")


def gband(g):
    """Gold-length tertile. Boundaries follow build_behavior_report.py, which
    drops the terminal DONE sentinel; counting it would shift every label by
    one while leaving the band sizes 113/347/140 untouched."""
    return "short" if g <= 8 else ("mid" if g <= 11 else "long")


def pool_pass(rs):
    """Pass rate pooled over all nine models -- tasks x models attempts."""
    return 100 * sum(t[NAME[k]][1] for t in rs for k in K) / (len(rs) * len(K))


def _plain(s):
    """Strip md emphasis so '**pooled**' and '*(none)*' match their labels."""
    return s.replace("**", "").replace("`", "").replace("*", "").strip()


def cells_of(sec, label):
    """Cells of the md table row whose first column is `label`.

    Rows are matched on the de-emphasised first cell, because the report bolds
    winning values and italicises '(none)'. Sections must be scoped by the
    caller: several tables reuse the same row labels (every model appears in
    both 8.1 and 8.5), so a document-wide search would silently read the wrong
    table and still find plausible numbers.
    """
    for line in sec.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        parts = [c for c in line.strip("|").split("|")]
        if parts and _plain(parts[0]) == label:
            return [_plain(c) for c in parts[1:]]
    return None


def prose(sec, pattern, label):
    """Pull a number straight out of the prose so the audit compares the text
    the reader sees, not a constant typed into this file.

    A constant here can silently drift from the sentence it is supposed to be
    guarding -- that is exactly how a wrong Spearman value and two wrong
    natural-image figures survived an audit that claimed to check them.
    """
    m = re.search(pattern, sec, re.M)
    ck(m is not None, f"{label}: prose pattern not found ({pattern})")
    return float(m.group(1).replace("\u2212", "-")) if m else None


def subsec(head, nxt):
    """The body of one '### 8.x' block, with dashes normalised."""
    body = sec8n.split(head)[1]
    return body.split(nxt)[0] if nxt else body


SEC80 = sec8n.split("### 8.1")[0]
SEC81 = subsec("### 8.1", "### 8.2")
SEC82 = subsec("### 8.2", "### 8.3")
SEC83 = subsec("### 8.3", "### 8.4")
SEC84 = subsec("### 8.4", "### 8.5")
SEC85 = subsec("### 8.5", None)


def near(a, b, tol=0.051):
    return a is not None and b is not None and abs(a - b) < tol


def as_f(s):
    """Parse a cell like '27.2', '+2.8', '22.3%', '25.1 [31]' or '\u22120.80'.

    U+2212 MINUS SIGN is normalised to ASCII: the prose and the tables use the
    typographic minus, and silently returning None for those cells would turn
    every signed comparison into a vacuous pass.
    """
    m = re.match(r"^([+-]?\d+(?:\.\d+)?)", s.strip().replace("\u2212", "-"))
    return float(m.group(1)) if m else None


def as_n(s):
    """Bracketed sample size in a cell like '27.2 [119]'."""
    m = re.search(r"\[(\d+)\]", s)
    return int(m.group(1)) if m else None


# provenance: this table must describe the same run the rest of the report does
ck(AX["effort"] == "medium", "S8.0: difficulty_axes.json is not the medium sweep")
ck(len(AXT) == 600, f"S8.0: {len(AXT)} tasks, expected 600")
ck(sum(t["units"] for t in AXT) == 3284, "S8.0: scored units != 3,284")
ck(min(t["units"] for t in AXT) == 2 and max(t["units"] for t in AXT) == 14,
   "S8.0: unit range is not 2-14")
ck(all(t[NAME[k]] is not None for t in AXT for k in K),
   "S8.0: some task is missing a model's result")
# the unit counts must agree with the check totals the rest of the report uses
ck(sum(t["units"] for t in AXT) == S["opus5"]["check_split"]["all"]["total"]
   - S["opus5"]["check_split"]["diagnostic"]["total"]
   if "diagnostic" in S["opus5"]["check_split"] else True,
   "S8.0: unit total disagrees with stats.json required-check total")

# -- 8.0 the degenerate-axis table -----------------------------------------
# Each row asserts a cardinality; recompute it rather than trust the prose.
CARD = AX["axis_cardinality"]
# The cardinality column is read out of the md rather than hardcoded here, so
# that editing the table without re-deriving it is a failure rather than a
# no-op. The row labels are the md's, backticks and all.
for field, label in (("difficulty", "difficulty"), ("domains", "domains"),
                     ("template_id", "template_id"), ("weight", "weight"),
                     ("round_type", "round_type"),
                     ("required_modalities", "required_modalities"),
                     ("allowed_tool_families", "allowed_tool_families"),
                     ("realism_tags", "realism_tags")):
    got = CARD[field]["n_distinct"]
    row = cells_of(SEC80, label)
    ck(row is not None, f"S8.0: no table row for {field}")
    if row:
        ck(near(as_f(row[0]), got, 0.001),
           f"S8.0: {field} cardinality is {got}, table says {row[0]!r}")
# the axis the section actually endorses
uu = cells_of(SEC80, "scored-unit count")
ck(uu is not None, "S8.0: no scored-unit-count row")
if uu:
    m = re.match(r"(\d+)-(\d+) per task", uu[0])
    ck(m is not None, f"S8.0: scored-unit range cell is {uu[0]!r}")
    if m:
        ck((int(m.group(1)), int(m.group(2)))
           == (min(t["units"] for t in AXT), max(t["units"] for t in AXT)),
           f"S8.0: stated unit range {uu[0]!r} is wrong")
for field in ("visible_constraints", "hidden_constraints"):
    ck(CARD[field]["list_lengths"] == [0],
       f"S8.0: {field} is not empty on every task")
ck(CARD["decision_requirements"]["list_lengths"] == [1],
   "S8.0: decision_requirements is not length 1 on every task")
for field in ("availability_tier", "online_requirement"):
    ck(CARD[field]["n_null"] == 600, f"S8.0: {field} is not null on every task")
ck(CARD["conflict_taxonomy"]["n_present"] == 0,
   "S8.0: conflict_taxonomy exists in v8, contradicting the table")
ck("**absent from v8 entirely**" in sec8, "S8.0: conflict_taxonomy verdict missing")
# round_type has 5 values but one is n=1, which 8.2 excludes as noise
rt_counts = collections.Counter(t["round_type"] for t in AXT)
ck(sorted(rt_counts.values())[0] == 1 and len(rt_counts) == 5,
   "S8.0: round_type is not '5 (one with n=1)'")

# -- 8.1 scored-unit bands --------------------------------------------------
ubn = collections.Counter(uband(t["units"]) for t in AXT)
ck([ubn[b] for b in UBANDS] == [192, 258, 150],
   f"S8.1: band sizes {[ubn[b] for b in UBANDS]}, prose says 192/258/150")
ck("192 /\n258 / 150" in sec8 or "192 / 258 / 150" in sec8.replace("\n", " "),
   "S8.1: band sizes not stated in prose")

mono_pass, mono_score = [], []
for k in K:
    disp = NAME[k]
    row = cells_of(SEC81, disp)
    ck(row is not None and len(row) == 6, f"S8.1 {disp}: row missing or malformed")
    if not row or len(row) != 6:
        continue
    ps, sc = [], []
    for i, b in enumerate(UBANDS):
        rs = [t for t in AXT if uband(t["units"]) == b]
        p = 100 * sum(t[disp][1] for t in rs) / len(rs)
        s = sum(t[disp][0] for t in rs) / len(rs)
        ps.append(p)
        sc.append(s)
        ck(near(as_f(row[i]), p),
           f"S8.1 {disp} {b} pass: md {row[i]} != recomputed {p:.1f}")
        ck(near(as_f(row[3 + i]), s, 0.00051),
           f"S8.1 {disp} {b} score: md {row[3 + i]} != recomputed {s:.3f}")
    if ps[0] > ps[1] > ps[2]:
        mono_pass.append(disp)
    if sc[0] > sc[1] > sc[2]:
        mono_score.append(disp)

prow = cells_of(SEC81, "pooled")
ck(prow is not None and len(prow) == 6, "S8.1: pooled row missing or malformed")
pooled_p = []
if prow and len(prow) == 6:
    for i, b in enumerate(UBANDS):
        rs = [t for t in AXT if uband(t["units"]) == b]
        p = pool_pass(rs)
        s = sum(t[NAME[k]][0] for t in rs for k in K) / (len(rs) * len(K))
        pooled_p.append(p)
        ck(near(as_f(prow[i]), p),
           f"S8.1 pooled {b} pass: md {prow[i]} != recomputed {p:.1f}")
        ck(near(as_f(prow[3 + i]), s, 0.00051),
           f"S8.1 pooled {b} score: md {prow[3 + i]} != recomputed {s:.3f}")
    ck(near(pooled_p[0] - pooled_p[2], 17.1),
       f"S8.1: spread {pooled_p[0] - pooled_p[2]:.1f}pp, prose says 17.1pp")
    ck("17.1pp" in sec8, "S8.1: 17.1pp absent")

# The two monotonicity counts are the section's headline claims.
ck(len(mono_pass) == 8 and "3.5-flash" not in mono_pass,
   f"S8.1: {len(mono_pass)} models have monotone pass, prose says 8 of 9 "
   f"with 3.5-flash the exception")
ck("8 of 9 models** (3.5-flash" in sec8, "S8.1: 8-of-9 pass claim not stated")
ck(set(mono_score) == {"sonnet-5", "sol", "terra"},
   f"S8.1: monotone-score set is {sorted(mono_score)}, prose says sonnet-5/sol/terra")
ck("only 3 of 9 models" in sec8, "S8.1: 3-of-9 score claim not stated")


def band_mean(disp, b, fn):
    rs = [t for t in AXT if uband(t["units"]) == b]
    return sum(fn(t, disp) for t in rs) / len(rs)


# satisfied units = fractional score x rubric size, which is what "absolute
# satisfied-unit count" means; the rubric column is just the mean unit count
sat_flash = [band_mean("3.5-flash", b, lambda t, d: t[d][0] * t["units"])
             for b in UBANDS]
sat_o5 = [band_mean("opus-5", b, lambda t, d: t[d][0] * t["units"]) for b in UBANDS]
rubric = [band_mean("opus-5", b, lambda t, d: t["units"]) for b in UBANDS]
for got, want, tag in ((sat_flash, [0.74, 1.88, 2.83], "3.5-flash satisfied"),
                       (sat_o5, [2.28, 3.62, 4.37], "opus-5 satisfied"),
                       (rubric, [3.51, 5.44, 8.05], "rubric size")):
    for g, w in zip(got, want):
        ck(near(g, w, 0.0051), f"S8.1 {tag}: recomputed {g:.2f}, md says {w}")
    ck(all(str(w) in sec8 for w in want), f"S8.1 {tag}: figures absent from prose")
ck(near(sat_flash[2] / sat_flash[0],
        prose(SEC81, r"2\.83 \((\d\.\d)\u00d7\)", "S8.1 flash ratio")),
   "S8.1: the 3.5-flash growth ratio in the prose is wrong")
ck(near(rubric[2] / rubric[0],
        prose(SEC81, r"8\.05 \((\d\.\d)\u00d7\)", "S8.1 rubric ratio")),
   "S8.1: the rubric growth ratio in the prose is wrong")
ck(near(sat_o5[2] / sat_o5[0],
        prose(SEC81, r"4\.37 \((\d\.\d)\u00d7\)", "S8.1 opus-5 ratio")),
   "S8.1: the opus-5 growth ratio in the prose is wrong")
# super- vs sub-proportional is the actual argument
ck(sat_flash[2] / sat_flash[0] > rubric[2] / rubric[0],
   "S8.1: 3.5-flash is not super-proportional, contradicting the prose")
ck(sat_o5[2] / sat_o5[0] < rubric[2] / rubric[0],
   "S8.1: opus-5 is not sub-proportional, contradicting the prose")
cost = 100 * sum(1 / t["units"] for t in AXT if uband(t["units"]) == "2-4") / ubn["2-4"]
ck(near(cost, prose(SEC81, r"costs ~(\d+)% of the score", "S8.1 unit cost"), 0.5),
   f"S8.1: one missed unit costs {cost:.1f}%, which the prose contradicts")

# -- 8.2 round type ---------------------------------------------------------
for rt in ("investigation", "verification", "action", "decision"):
    g = [t for t in AXT if t["round_type"] == rt]
    row = cells_of(SEC82, rt)
    ck(row is not None and len(row) == 6, f"S8.2 {rt}: row missing or malformed")
    if not row or len(row) != 6:
        continue
    ck(int(row[0]) == len(g), f"S8.2 {rt}: n {row[0]} != {len(g)}")
    mu = sum(t["units"] for t in g) / len(g)
    ck(near(as_f(row[1]), mu, 0.0051), f"S8.2 {rt}: mean units {row[1]} != {mu:.2f}")
    for i, b in enumerate(UBANDS):
        sub = [t for t in g if uband(t["units"]) == b]
        ck(near(as_f(row[2 + i]), pool_pass(sub)),
           f"S8.2 {rt} {b}: md {row[2 + i]} != {pool_pass(sub):.1f}")
        ck(as_n(row[2 + i]) == len(sub),
           f"S8.2 {rt} {b}: bracketed n {as_n(row[2 + i])} != {len(sub)}")
    ck(near(as_f(row[5]), pool_pass(g)),
       f"S8.2 {rt} marginal: md {row[5]} != {pool_pass(g):.1f}")
ck(rt_counts["reconciliation"] == 1, "S8.2: reconciliation is not n=1")

# the floor claim, recomputed end to end
floor = [t for t in AXT if t["round_type"] == "investigation" and uband(t["units"]) == "7+"]
n_att = len(floor) * len(K)
n_succ = sum(t[NAME[k]][1] for t in floor for k in K)
ck(len(floor) == 43 and n_att == 387,
   f"S8.2 floor: {len(floor)} tasks x 9 = {n_att}, prose says 43 x 9 = 387")
ck(n_succ == 7, f"S8.2 floor: {n_succ} successes, prose says roughly 7")
ck(near(100 * n_succ / n_att, 1.8), "S8.2 floor: 1.8% wrong")
ck("43 tasks \u00d7 9 models" in sec8 and "7 successes out of 387" in sec8,
   "S8.2: floor claim not stated as 43 x 9 / 7 of 387")
# "action easiest, investigation hardest, in both hard bands" -- the actual claim
for b in ("5-6", "7+"):
    per_rt = {rt: pool_pass([t for t in AXT if t["round_type"] == rt
                             and uband(t["units"]) == b])
              for rt in ("investigation", "verification", "action", "decision")}
    ck(max(per_rt, key=lambda r: per_rt[r]) == "action",
       f"S8.2: action is not easiest in the {b} band")
    ck(min(per_rt, key=lambda r: per_rt[r]) == "investigation",
       f"S8.2: investigation is not hardest in the {b} band")
# ... and it must hold despite action carrying the most units
mu_rt = {rt: sum(t["units"] for t in AXT if t["round_type"] == rt)
             / rt_counts[rt] for rt in ("investigation", "verification",
                                        "action", "decision")}
ck(max(mu_rt, key=lambda r: mu_rt[r]) == "action",
   "S8.2: action does not carry the most units")
ck(min(mu_rt, key=lambda r: mu_rt[r]) == "investigation",
   "S8.2: investigation does not carry the fewest units")

# -- 8.3 natural images (Simpson reversal) ----------------------------------
img = [t for t in AXT if t["natural_image"]]
noimg = [t for t in AXT if not t["natural_image"]]
ck(near(pool_pass(img),
        prose(SEC83, r"pass at \*\*(\d+\.\d)%\*\*", "S8.3 img marginal")),
   f"S8.3: img marginal is {pool_pass(img):.1f}, which the prose contradicts")
ck(near(pool_pass(noimg),
        prose(SEC83, r"versus \*\*(\d+\.\d)%\*\*", "S8.3 no-img marginal")),
   f"S8.3: no-img marginal is {pool_pass(noimg):.1f}, which the prose contradicts")
mu_img = sum(t["units"] for t in img) / len(img)
mu_no = sum(t["units"] for t in noimg) / len(noimg)
ck(near(mu_img, prose(SEC83, r"average \*\*(\d\.\d\d)\*\*", "S8.3 img units"), 0.0051),
   f"S8.3: img mean units is {mu_img:.2f}, which the prose contradicts")
ck(near(mu_no, prose(SEC83, r"against \*\*(\d\.\d\d)\*\*", "S8.3 no-img units"), 0.0051),
   f"S8.3: no-img mean units is {mu_no:.2f}, which the prose contradicts")
# the reversal itself: marginal favours images, every hard band does not
ck(pool_pass(img) > pool_pass(noimg), "S8.3: marginal does not favour images")
signs = []
for b in UBANDS:
    a = [t for t in noimg if uband(t["units"]) == b]
    c = [t for t in img if uband(t["units"]) == b]
    row = cells_of(SEC83, b)
    ck(row is not None and len(row) == 5, f"S8.3 {b}: row missing or malformed")
    if not row or len(row) != 5:
        continue
    ck(int(row[0]) == len(a), f"S8.3 {b}: n(no-img) {row[0]} != {len(a)}")
    ck(int(row[1]) == len(c), f"S8.3 {b}: n(img) {row[1]} != {len(c)}")
    ck(near(as_f(row[2]), pool_pass(a)), f"S8.3 {b}: no-img pass wrong")
    ck(near(as_f(row[3]), pool_pass(c)), f"S8.3 {b}: img pass wrong")
    diff = pool_pass(c) - pool_pass(a)
    ck(near(as_f(row[4]), diff), f"S8.3 {b}: diff {row[4]} != {diff:+.1f}")
    signs.append(diff)
ck(len(signs) == 3 and signs[0] > 0 > signs[1] and signs[1] > signs[2],
   "S8.3: the stratified diffs are not a sign reversal deepening with band")

# -- 8.4 tool family --------------------------------------------------------
byfam = collections.defaultdict(list)
for t in AXT:
    byfam[t["tool_extra"] or "(none)"].append(t)
ck(len(byfam) == 10, f"S8.4: {len(byfam)} family groups, expected 10")
# The merge that this section previously got wrong: lock and lock+review are
# disjoint scenario sets and must not be pooled into one row.
ck("lock" in byfam and "lock+review" in byfam,
   "S8.4: lock / lock+review are no longer distinct groups")
ck(not ({t["scen"] for t in byfam["lock"]} & {t["scen"] for t in byfam["lock+review"]}),
   "S8.4: lock and lock+review share a scenario")
LABEL = {"lock+review": "lock + review", "(none)": "(none)"}
seen = 0
for fam, g in byfam.items():
    row = cells_of(SEC84, LABEL.get(fam, fam))
    ck(row is not None and len(row) == 2, f"S8.4 {fam}: row missing or malformed")
    if not row or len(row) != 2:
        continue
    seen += 1
    ck(int(row[0]) == len(g), f"S8.4 {fam}: n {row[0]} != {len(g)}")
    ck(near(as_f(row[1]), pool_pass(g)),
       f"S8.4 {fam}: pooled {row[1]} != {pool_pass(g):.1f}")
ck(seen == 10, f"S8.4: only {seen} of 10 family rows found in the md")
fam_pass = {f: pool_pass(g) for f, g in byfam.items()}
spread = max(fam_pass.values()) - min(fam_pass.values())
ck(near(spread, prose(SEC84, r"^(\d+\.\d)pp spread", "S8.4 spread")),
   f"S8.4: spread is {spread:.1f}pp, which the prose contradicts")
ck(sorted(f for f, v in fam_pass.items() if v < 12)
   == sorted(["tasks", "booking", "lock", "containment"]),
   "S8.4: the under-12% set is not tasks/booking/lock/containment")
ck(near(fam_pass["lock+review"] - fam_pass["lock"],
        prose(SEC84, r"hiding a (\d+\.\d)pp gap", "S8.4 lock gap")),
   "S8.4: the lock/lock+review gap in the prose is wrong")
# guard the old wrong grouping from coming back
merged = byfam["lock"] + byfam["lock+review"]
ck(near(pool_pass(merged), 21.1), "S8.4: merged-lock control value drifted")
ck("| lock | 60 |" not in sec8, "S8.4: the merged lock row is back")
ck("17.5pp spread" not in sec8, "S8.4: the understated 17.5pp spread is back")

# -- 8.5 gold-trajectory length ---------------------------------------------
gbn = collections.Counter(gband(t["gold_len"]) for t in AXT)
gm = re.search(r"\(n = (\d+) / (\d+) / (\d+); gold max (\d+) steps", SEC85)
ck(gm is not None, "S8.5: the band-size/gold-max sentence is missing")
if gm:
    ck([gbn[b] for b in GBANDS] == [int(gm.group(i)) for i in (1, 2, 3)],
       f"S8.5: gold bands are {[gbn[b] for b in GBANDS]}, which the prose contradicts")
    ck(max(t["gold_len"] for t in AXT) == int(gm.group(4)),
       "S8.5: the stated gold max is wrong")
gmono = []
for k in K:
    disp = NAME[k]
    row = cells_of(SEC85, disp)
    ck(row is not None and len(row) == 4, f"S8.5 {disp}: row missing or malformed")
    if not row or len(row) != 4:
        continue
    v = []
    for i, b in enumerate(GBANDS):
        rs = [t for t in AXT if gband(t["gold_len"]) == b]
        p = 100 * sum(t[disp][1] for t in rs) / len(rs)
        v.append(p)
        ck(near(as_f(row[i]), p), f"S8.5 {disp} {b}: md {row[i]} != {p:.1f}")
    allp = 100 * sum(t[disp][1] for t in AXT) / len(AXT)
    ck(near(as_f(row[3]), allp), f"S8.5 {disp} all: md {row[3]} != {allp:.1f}")
    # must agree with the section-1 headline for the same model
    ck(near(allp, S[k]["pass_pct"]),
       f"S8.5 {disp}: overall {allp:.1f} disagrees with stats.json pass_pct")
    if v[0] > v[1] > v[2]:
        gmono.append(disp)
ck(len(gmono) == 8 and "3.5-flash" not in gmono,
   f"S8.5: {len(gmono)} models are monotone, prose says eight of nine "
   f"with 3.5-flash the exception")
ck("**eight of nine**" in sec8, "S8.5: the eight-of-nine correction is missing")
ck("Monotone decreasing for all nine" not in sec8,
   "S8.5: the false all-nine monotonicity claim is back")
fl = [100 * sum(t["3.5-flash"][1] for t in AXT if gband(t["gold_len"]) == b)
      / gbn[b] for b in GBANDS]
ck(fl[1] > fl[0], "S8.5: 3.5-flash does not tick up from short to mid")
npass = [sum(t["3.5-flash"][1] for t in AXT if gband(t["gold_len"]) == b)
         for b in GBANDS]
pm = re.search(r"on (\d+) / (\d+) / (\d+)\s*\npasses", SEC85)
ck(pm is not None, "S8.5: the 3.5-flash per-band pass counts are missing")
if pm:
    ck(npass == [int(pm.group(i)) for i in (1, 2, 3)],
       f"S8.5: 3.5-flash band passes are {npass}, which the prose contradicts")
ck(sum(npass) == S["flash"]["passed"],
   "S8.5: 3.5-flash band passes do not sum to its overall passed count")

# ---- B4. field_contains decomposition (section 7.3) ----------------------
sec73 = section("### 7.3 `field_contains`, decomposed", "## 8.")
asset_pct, content_pct = {}, {}
for k in K:
    d = S[k]["check_split"]
    a, c, u = d["field_contains_asset"], d["field_contains_content"], d["field_contains"]
    # The sub-buckets must partition the union exactly, or the pooled column and
    # the split describe different sets of checks.
    for field in ("total", "graded", "passed"):
        ck(a[field] + c[field] == u[field],
           f"S7.3 {NAME[k]}: {field} {a[field]}+{c[field]} != union {u[field]}")
    # These are deterministic checkers; a judge is never consulted, so the
    # not-invoked attrition that distorts semantic_match cannot apply.
    ck(a["graded"] == a["total"] and c["graded"] == c["total"],
       f"S7.3 {NAME[k]}: field_contains has judge attrition, which contradicts the prose")
    asset_pct[k] = a["graded_pass_pct"]
    content_pct[k] = c["graded_pass_pct"]

    row = re.search(r"^\| " + re.escape(NAME[k]) + r" \|(.+)$", sec73, re.M)
    ck(row is not None, f"S7.3: no row for {NAME[k]}")
    if not row:
        continue
    vals = [x.strip().replace("**", "") for x in row.group(1).split("|") if x.strip()]
    ck(len(vals) == 4, f"S7.3 {NAME[k]}: {len(vals)} cells, expected 4")
    ck(float(vals[0]) == a["graded_pass_pct"],
       f"S7.3 {NAME[k]}: asset {vals[0]} != {a['graded_pass_pct']}")
    ck(float(vals[1]) == c["graded_pass_pct"],
       f"S7.3 {NAME[k]}: content {vals[1]} != {c['graded_pass_pct']}")
    ck(float(vals[2]) == u["graded_pass_pct"],
       f"S7.3 {NAME[k]}: pooled {vals[2]} != {u['graded_pass_pct']}")
    want_gap = round(c["graded_pass_pct"] - a["graded_pass_pct"], 1)
    ck(abs(float(vals[3].replace("\u2212", "-")) - want_gap) < 0.05,
       f"S7.3 {NAME[k]}: gap {vals[3]} != recomputed {want_gap}")

# structural counts quoted in the warning
o5cs = S["opus5"]["check_split"]
ck(o5cs["field_contains_asset"]["total"] == 822, "S7.3: asset-gate count is not 822")
ck(o5cs["field_contains_content"]["total"] == 256, "S7.3: content count is not 256")
ck(822 + 256 == o5cs["field_contains"]["total"], "S7.3: 822+256 != pooled total")
ck(abs(100 * 822 / 1078 - 76.3) < 0.05, "S7.3: 76.3% wrong")
ck(822 - 346 == 476, "S7.3: 346 required + 476 diagnostic != 822")


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: -v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for t in range(i, j + 1):
                out[order[t]] = (i + j) / 2 + 1
            i = j + 1
        return out

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den


# Read both coefficients out of the prose rather than hardcoding them, so the
# audit cannot silently drift away from what the report actually claims.
m_rho = re.search(
    r"Spearman .{0,40}?\*\*(?P<all>[-\u2212+]?\d\.\d\d)\*\* \(n=9; "
    r"(?P<ex>[-\u2212+]?\d\.\d\d) with 3\.5-flash removed\)",
    sec73, re.S)
ck(m_rho is not None, "S7.3: cannot find the Spearman sentence in the prose")
if m_rho:
    md_rho = float(m_rho.group("all").replace("\u2212", "-"))
    md_rho2 = float(m_rho.group("ex").replace("\u2212", "-"))
    rho = spearman([asset_pct[k] for k in K], [content_pct[k] for k in K])
    ck(abs(rho - md_rho) < 0.005,
       f"S7.3: Spearman recomputed {rho:+.2f}, md says {md_rho:+.2f}")
    k2 = [k for k in K if k != "flash"]
    rho2 = spearman([asset_pct[k] for k in k2], [content_pct[k] for k in k2])
    ck(abs(rho2 - md_rho2) < 0.005,
       f"S7.3: Spearman ex-3.5-flash recomputed {rho2:+.2f}, md says {md_rho2:+.2f}")
    # Both must round to "essentially zero", which is the claim the prose makes.
    ck(abs(rho) < 0.10 and abs(rho2) < 0.10,
       "S7.3: the 'uncorrelated' claim no longer holds")

# ranked claims made in the prose
a_rank = sorted(K, key=lambda k: -asset_pct[k])
c_rank = sorted(K, key=lambda k: -content_pct[k])
ck(a_rank.index("flash") == 3, "S7.3: 3.5-flash is not 4th on the asset gate")
ck(asset_pct["flash"] > asset_pct["opus"], "S7.3: 3.5-flash no longer above opus-4.8 on the gate")
ck(c_rank[-1] == "flash", "S7.3: 3.5-flash is not last on literal content")
ck(abs(round(content_pct[c_rank[-2]] - content_pct["flash"], 1) - 41.3) < 0.05,
   "S7.3: 41.3-point content gap wrong")
ck(a_rank[-1] == "sonnet", "S7.3: sonnet-5 is not last on the asset gate")
ck(abs(round(asset_pct[a_rank[-2]] - asset_pct["sonnet"], 1) - 16.9) < 0.05,
   "S7.3: 16.9-point gate gap wrong")
ck(a_rank.index("opus5") < 3 and c_rank.index("opus5") < 3,
   "S7.3: opus-5 is not top-3 in both")
ck(sum(1 for k in K if a_rank.index(k) < 3 and c_rank.index(k) < 3) == 1,
   "S7.3: opus-5 is not the ONLY model top-3 in both")
ck(a_rank[0] == "sol" and c_rank.index("sol") == 5,
   "S7.3: sol is no longer 1st on the gate / 6th on content")
ck(asset_pct["terra"] == content_pct["terra"], "S7.3: terra is no longer flat")

# ---- B5. provenance and internal arithmetic ------------------------------
for v in ["16,200", "191,126", "1,620", "202,831", "3,760", "3,284",
          "822", "346", "254", "42.3%", "476", "76.3%"]:
    ck(v in md, f"provenance: {v} absent")
ck(3760 - 476 == 3284, "scored-unit arithmetic broken")
ck(600 - 346 == 254, "grounding arithmetic broken")
ck(abs(100 * 254 / 600 - 42.3) < 0.05, "42.3% wrong")

for eff, want in [("low", 15.2), ("medium", 22.1), ("high", 22.2)]:
    vals = [cells[f"{k}@{eff}"]["pass_pct"] for k in K]
    got = round(max(vals) - min(vals), 1)
    ck(abs(got - want) < 0.05, f"S9.3 {eff} spread recomputed {got}, md says {want}")

# ---- C1. section 10 and 11.3 against the raw curated dump ----------------
# Section 10 is the densest part of the report and the easiest to get wrong,
# because every claim is about one run rather than an aggregate. An earlier
# draft misattributed the only credited notification in the warehouse round to
# the wrong recipient and built a "severe limitation" on top of it. These
# checks recompute the claims from curated_rounds.json.

MODEL_NAMES = ["3.5-flash", "3.6-flash", "3.1-pro", "opus-4.8", "opus-5",
               "sonnet-5", "sol", "luna", "terra"]

# The agent each round's notification criterion names. Taken from the grader
# rationales, which quote the recipient and channel they credit.
NAMED_AGENT = {
    "payment_chargeback_dispute_v01_r04": ["Casey_Support_Specialist"],
    "warehouse_robotics_safety_halt_v02_r01": ["Avery_Safety_Officer"],
    "satellite_ground_station_handover_v02_r01": ["Devon_Ground_Ops_Lead"],
    "clinical_trial_dosage_conflict_v02_r03": ["Avery_Trial_Safety_Monitor"],
    "coordinated_containment_v02_r04": ["Owen_Platform_On_Call"],
    "distributed_lock_shared_resource_v01_r02": ["Casey_Inventory_Controller"],
    "expense_reimbursement_conflict_v02_r05": ["Blair_Audit_Investigator",
                                               "Casey_Approver"],
    "expiring_resource_hold_v03_r04": ["Rowan_Research_Lead"],
}

ck(set(CR) == set(NAMED_AGENT), "S10: curated dump does not hold the eight rounds")
ck(all(set(v) == set(MODEL_NAMES) for v in CR.values()),
   "S10: a curated round is missing a model")

UNIT_WORD = {2: "two", 3: "three", 4: "four", 5: "five"}
for rk, mm in CR.items():
    sec = section(f"### 10.{list(NAMED_AGENT).index(rk) + 1} ", "\n### ") \
        if False else None
    # Locate the round's paragraph by its heading rather than by index, so
    # reordering section 10 cannot silently point a check at the wrong prose.
    m = re.search(r"^### 10\.\d+ `" + re.escape(rk) + r"`\n+(.+?)$", md, re.M)
    ck(m is not None, f"S10 {rk}: no heading/paragraph in the report")
    if not m:
        continue
    para = m.group(1)

    # Unit counts must agree with the number of required checks in the run.
    units = {d["n_required"] for d in mm.values()}
    ck(len(units) == 1, f"S10 {rk}: models disagree on the required-check count")
    n_units = units.pop()
    word = UNIT_WORD.get(n_units, str(n_units))
    ck(re.search(rf"\b{word}\b", para, re.I) is not None,
       f"S10 {rk}: prose does not state {word} units ({n_units} required checks)")

    # Every "<model> @N" citation is a report-write step, not a step total.
    # \b on both sides matters: a bare "sol" otherwise matches inside "sole".
    for mm_name in MODEL_NAMES:
        tok = r"\b" + re.escape(mm_name) + r"\b"
        for hit in re.finditer(tok + r" @(\d+)", para):
            want = int(hit.group(1))
            steps = [f["step"] for f in mm[mm_name]["files"]]
            ck(want in steps,
               f"S10 {rk}: {mm_name} @{want} is not a file-create step {steps}")

        # Any "<model> ... N steps" style citation must match the run length.
        for hit in re.finditer(tok + r"[^.|]{0,80}?\*\*(\d+)\*\* steps", para):
            ck(int(hit.group(1)) == mm[mm_name]["steps"],
               f"S10 {rk}: {mm_name} cited at {hit.group(1)} steps, "
               f"actual {mm[mm_name]['steps']}")

# 11.3: the routing table, recomputed.
sec113 = section("### 11.3 ", "### 11.4")
credited_total = 0
for rk, agents in NAMED_AGENT.items():
    mm = CR[rk]
    for agent in agents:
        routed = sum(1 for d in mm.values()
                     if any(agent in (x["target"] or "") for x in d["messages"]))
        row = re.search(r"^\| `" + re.escape(rk) + r"` \| `" + re.escape(agent)
                        + r"` \| (\d+)/9 \| \*{0,2}(\d+)/9\*{0,2} \|$",
                        sec113, re.M)
        ck(row is not None, f"S11.3: no table row for {rk} / {agent}")
        if row:
            ck(int(row.group(1)) == routed,
               f"S11.3 {rk}/{agent}: routed {row.group(1)}/9, recomputed {routed}/9")
    # Credited is per model, counting the round's message units as a set.
    cred = sum(1 for d in mm.values()
               if any(c["passed"] for c in d["checks"]
                      if "messages_sent" in c["name"]))
    for agent in agents:
        row = re.search(r"^\| `" + re.escape(rk) + r"` \| `" + re.escape(agent)
                        + r"` \| \d+/9 \| \*{0,2}(\d+)/9\*{0,2} \|$", sec113, re.M)
        if row:
            ck(int(row.group(1)) == cred,
               f"S11.3 {rk}/{agent}: credited {row.group(1)}/9, recomputed {cred}/9")
    credited_total += sum(
        1 for d in mm.values() for c in d["checks"]
        if "messages_sent" in c["name"] and c["passed"])

ck(credited_total == 37,
   f"S11.3: credited notification units recomputed {credited_total}, md says 37")
ck("**37**" in sec113, "S11.3: the 37 figure is not in the prose")

# The invariant the corrected text rests on: no credited notification was
# addressed to anyone other than the agent the criterion names.
off_target = []
for rk, mm in CR.items():
    for name, d in mm.items():
        for c in d["checks"]:
            if "messages_sent" in c["name"] and c["passed"] \
                    and not any(a in c["judge"] for a in NAMED_AGENT[rk]):
                off_target.append(f"{rk}/{name}")
ck(not off_target,
   f"S11.3: credited message addressed off-target: {off_target}")
ck("no observed case of a message credited to a" in sec113,
   "S11.3: the off-target invariant is not stated in the prose")

# The retracted claim must not come back.
for banned in ["Casey_Line_Supervisor", "went to somebody else",
               "went to a *different* agent",
               "routing is not a necessary condition"]:
    ck(banned not in md, f"S11.3: retracted claim resurfaced: {banned!r}")

# Cross-round facts section 10 asserts.
empty_caps = {rk for rk, mm in CR.items()
              if sum(1 for d in mm.values()
                     if d["steps"] == 20 and not d["files"]) >= 2}
ck(empty_caps == {"warehouse_robotics_safety_halt_v02_r01",
                  "expiring_resource_hold_v03_r04"},
   f"S10: rounds with >1 empty cap changed: {sorted(empty_caps)}")

max_open = max((len(d["opens"]), rk, n)
               for rk, mm in CR.items() for n, d in mm.items())
ck(max_open[0] == 10 and max_open[2] == "3.5-flash"
   and max_open[1] == "expense_reimbursement_conflict_v02_r05",
   f"S10.7: the most-opens claim is stale: {max_open}")

# 10.7: opus-5 is the only model that never writes the passcode target.
exp = CR["expense_reimbursement_conflict_v02_r05"]
no_cfg = {n for n, d in exp.items()
          if not any("telemetry_settings.json" in (f["path"] or "")
                     for f in d["files"])}
ck(no_cfg == {"opus-5"},
   f"S10.7: models never writing telemetry_settings.json changed: {sorted(no_cfg)}")

# ==========================================================================
# PASS D -- eval-results-capability-analysis.md, recomputed from
#           failure_taxonomy.json
#
# Same rule as pass B: nothing here compares the md against a constant typed
# into this file. Every expected value is derived from the taxonomy dump, and
# every observed value is parsed back out of the rendered markdown, so a table
# that drifts from the data fails even if the prose around it still reads well.
# ==========================================================================

CAPMD = os.path.join(HERE, "eval-results-capability-analysis.md")
TAXON = os.path.join(HERE, "failure_taxonomy.json")

cap = open(CAPMD, encoding="utf-8").read()
capn = cap.replace("\u2212", "-")
FT = json.load(open(TAXON, encoding="utf-8"))

FTM = FT["cells"]["medium"]["models"]
DISP = {k: FTM[k]["display"] for k in K}
FBUCKET = [("F1", "content_msg"), ("F2", "content_file"), ("F3", "grounding"),
           ("F4", "anchor"), ("F5", "side_effect"), ("F6", "window")]


def capsec(start, end=None):
    """Body of one capability-doc section, scoped by heading."""
    if start not in cap:
        ck(False, f"capability doc: heading {start!r} missing")
        return ""
    body = cap.split(start, 1)[1]
    return body.split(end, 1)[0] if end and end in body else body


def flat(s):
    """Collapse whitespace runs so md re-wrapping cannot break a prose check.

    Prose assertions that embed a literal '\\n' are hostage to where the
    paragraph happens to wrap; three of them broke the first time this pass
    ran. Collapsing cannot manufacture a false pass -- distinct numbers still
    have to match -- it only removes the line-break dependency.
    """
    return re.sub(r"\s+", " ", s)


WORD = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
        6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def numword(n):
    """Every rendering the prose may legitimately use for a small integer."""
    w = WORD.get(n)
    return (str(n),) if w is None else (str(n), w, w.capitalize())


def caprow(sec, label):
    """Cells of the md row whose de-emphasised first column is `label`."""
    for line in sec.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        parts = line.strip("|").split("|")
        if parts and _plain(parts[0]) == label:
            return [_plain(c) for c in parts[1:]]
    return None


def capprose(sec, pattern, label):
    m = re.search(pattern, sec.replace("\u2212", "-"), re.S)
    ck(m is not None, f"{label}: prose pattern not found ({pattern})")
    return float(m.group(1)) if m else None


def req_rate(key, bucket, eff="medium"):
    a = FT["cells"][eff]["models"][key]
    return 100.0 * a["failed_checks"][bucket]["required"] \
        / a["total_checks"][bucket]["required"]


# ---- D0. provenance and structural invariants ---------------------------
ck(sorted(FT["cells"]) == ["high", "low", "medium"],
   "D0: failure_taxonomy.json does not cover all three efforts")
for eff in ("low", "medium", "high"):
    for k in K:
        a = FT["cells"][eff]["models"][k]
        ck(a["rows"] == 600, f"D0: {eff}/{k} has {a['rows']} rows, expected 600")
        ck(a["dupe_rows"] == 0, f"D0: {eff}/{k} has duplicate rows")
        ck(a["ok_false"] == 0, f"D0: {eff}/{k} has ok=False check records")
        ck(not a["unbucketed_names"],
           f"D0: {eff}/{k} has unbucketed check names")
        ck(sum(v["required"] + v["diagnostic"]
               for v in a["total_checks"].values()) == a["checks_total"],
           f"D0: {eff}/{k} bucket totals do not sum to checks_total")

# ---- D1. section 1 capability ladder ------------------------------------
CAP1 = capsec("### Capability ladder", "## 2. Measurement validity")
for k in K:
    a = FTM[k]
    row = caprow(CAP1, a["display"])
    ck(row is not None, f"D1: no ladder row for {a['display']}")
    if not row:
        continue
    # cols: provider | pass@budget | avg_score | per-check | near_miss |
    #       zero_score | capped | zero-write
    ck(f"{a['pass_pct']}%" in row[1],
       f"D1 {k}: pass_pct {row[1]!r} != {a['pass_pct']}")
    ck(f"({a['passed']}/600)" in row[1],
       f"D1 {k}: passed count {row[1]!r} != {a['passed']}")
    ck(as_f(row[2]) == a["avg_score"],
       f"D1 {k}: avg_score {row[2]!r} != {a['avg_score']}")
    ck(near(as_f(row[3]), S[k]["clustering"]["per_check_pct"], 0.051),
       f"D1 {k}: per-check {row[3]!r} != {S[k]['clustering']['per_check_pct']}")
    for col, field in ((4, "near_miss"), (5, "zero_score"),
                       (6, "capped"), (7, "zero_write_tasks")):
        ck(as_f(row[col]) == a[field],
           f"D1 {k}: {field} {row[col]!r} != {a[field]}")

# the ladder must be sorted by pass rate, and the doc's own gap arithmetic
_ladder = [caprow(CAP1, DISP[k])
           for k in sorted(K, key=lambda x: -FTM[x]["pass_pct"])]
ck(all(r is not None for r in _ladder), "D1: a ladder row is missing")
ck([as_f(r[1]) for r in _ladder if r is not None]
   == sorted((FTM[k]["pass_pct"] for k in K), reverse=True),
   "D1: capability ladder is not ordered by pass rate")
d_sol = round(FTM["opus5"]["avg_score"] - FTM["sol"]["avg_score"], 4)
d_o48 = round(FTM["opus5"]["avg_score"] - FTM["opus"]["avg_score"], 4)
d_2nd = round(FTM["sol"]["avg_score"] - FTM["opus"]["avg_score"], 4)
ck(capprose(CAP1, r"leads sol by (\d\.\d+)\s*\n?> ?avg-score", "D1 gap sol")
   == d_sol, f"D1: opus-5 - sol avg-score gap != {d_sol}")
ck(capprose(CAP1, r"opus-4\.8 by (\d\.\d+) avg-score", "D1 gap opus-4.8")
   == d_o48, f"D1: opus-5 - opus-4.8 avg-score gap != {d_o48}")
ck(capprose(CAP1, r"avg-score \(\+(\d\.\d+)\)", "D1 second-place gap") == d_2nd,
   f"D1: sol - opus-4.8 avg-score gap != {d_2nd}")
ck(near(capprose(CAP1, r"pass rate \(-(\d\.\d)", "D1 second-place pass gap"),
        round(FTM["opus"]["pass_pct"] - FTM["sol"]["pass_pct"], 1), 0.001),
   "D1: sol - opus-4.8 pass-rate gap is wrong")
gem_spread = round(FTM["flash36"]["avg_score"] - FTM["flash"]["avg_score"], 4)
ck(capprose(CAP1, r"is a (\d\.\d+) avg-score range", "D1 gemini spread")
   == gem_spread, f"D1: Gemini avg-score spread != {gem_spread}")
rest = [FTM[k]["avg_score"] for k in K if not DISP[k].startswith("gemini")]
ck(f"({min(rest):.4f}\u2013{max(rest):.4f})" in cap,
   "D1: non-Gemini avg-score range is stale")
ck(gem_spread > max(rest) - min(rest),
   "D1: the 'wider than the other six' claim no longer holds")

# ---- D2. section 2.2, the rubric's shape --------------------------------
CAP22 = capsec("### 2.2 The rubric", "### 2.3 The judge")
names = FTM["opus5"]["bucket_names"]
cond = collections.defaultdict(lambda: [0, 0])   # condition -> [names, checks]
for b, inv in names.items():
    for nm, cnt in inv.items():
        c = ("any_field_equals" if nm.endswith(" any_field_equals")
             else nm.rsplit(" ", 1)[-1])
        cond[c][0] += 1
        cond[c][1] += cnt
for c in ("any_field_equals", "field_semantic_match", "field_contains",
          "within_window"):
    row = caprow(CAP22, c)          # caprow strips the label's backticks
    ck(row is not None, f"D2: no rubric row for {c}")
    if row:
        ck(as_f(row[0]) == cond[c][0],
           f"D2 {c}: distinct names {row[0]!r} != {cond[c][0]}")
        ck(as_n(f"[{row[1].replace(',', '')}]") == cond[c][1]
           or as_f(row[1].replace(",", "")) == cond[c][1],
           f"D2 {c}: check count {row[1]!r} != {cond[c][1]}")
tot_names = sum(v[0] for v in cond.values())
tot_checks = sum(v[1] for v in cond.values())
trow = caprow(CAP22, "total")
ck(trow is not None and as_f(trow[0]) == tot_names,
   f"D2: total distinct names != {tot_names}")
ck(trow is not None and as_f(trow[1].replace(",", "")) == tot_checks,
   f"D2: total checks != {tot_checks}")
req_all = sum(v["required"] for v in FTM["opus5"]["total_checks"].values())
diag_all = sum(v["diagnostic"] for v in FTM["opus5"]["total_checks"].values())
ck(f"{req_all:,} required / {diag_all} diagnostic" in cap,
   f"D2: role split is not {req_all:,}/{diag_all}")
# field_contains is the only split-role condition
fc_req = (FTM["opus5"]["total_checks"]["grounding"]["required"]
          + FTM["opus5"]["total_checks"]["anchor"]["required"])
ck(f"{fc_req} `required`, {diag_all} `diagnostic`" in cap,
   f"D2: field_contains role split is not {fc_req}/{diag_all}")
# the long tail: the three largest side-effect names, then the tail count


def eff_of(name):
    """'effect:booking_held any_field_equals' -> 'booking_held'."""
    return name.split(":", 1)[1].rsplit(" ", 1)[0]


se = sorted(names["side_effect"].items(), key=lambda kv: -kv[1])
for i, verb in enumerate(("at", "has", "")):
    frag = f"`effect:{eff_of(se[i][0])}`" + (f" {verb}" if verb else "")
    ck(f"{frag} {se[i][1]}" in CAP22,
       f"D2: side-effect name #{i + 1} is "
       f"effect:{eff_of(se[i][0])} at {se[i][1]} checks")
n_small = sum(1 for _, v in se if v < 15)
ck(any(f"{v} names have fewer than 15" in CAP22 for v in numword(n_small)),
   f"D2: {n_small} side-effect names have <15 checks")

# ---- D3. section 2.4, fail-closed ---------------------------------------
CAP24 = capsec("### 2.4 Fail-closed", "### 2.5 Seven episodes")
for k in K:
    a = FTM[k]
    row = caprow(CAP24, a["display"])
    ck(row is not None, f"D3: no fail-closed row for {a['display']}")
    if not row:
        continue
    ck(as_f(row[0].replace(",", "")) == a["semantic"]["n"],
       f"D3 {k}: semantic n {row[0]!r} != {a['semantic']['n']}")
    ck(as_f(row[1].replace(",", "")) == a["semantic"]["fail_closed"],
       f"D3 {k}: fail_closed {row[1]!r} != {a['semantic']['fail_closed']}")
    ck(near(as_f(row[2]), a["semantic"]["fail_closed_pct"], 0.051),
       f"D3 {k}: fail_closed_pct {row[2]!r} != {a['semantic']['fail_closed_pct']}")
worst = max(K, key=lambda x: FTM[x]["semantic"]["fail_closed_pct"])
ck(worst == "flash", f"D3: worst fail-closed model is now {DISP[worst]}")
# the headline restates the worst row; all three of its numbers are recomputed
sem = FTM["flash"]["semantic"]
mh = re.search(r"([\d,]+) of ([\d,]+), or (\d+\.\d)%", flat(CAP24))
ck(mh is not None, "D3: the 3.5-flash fail-closed headline is missing")
if mh:
    ck(int(mh.group(1).replace(",", "")) == sem["fail_closed"],
       f"D3: headline fail-closed count != {sem['fail_closed']}")
    ck(int(mh.group(2).replace(",", "")) == sem["n"],
       f"D3: headline semantic-check count != {sem['n']}")
    ck(float(mh.group(3)) == sem["fail_closed_pct"],
       f"D3: headline fail-closed pct != {sem['fail_closed_pct']}")
ck(sem["fail_closed_pct"] >= 50.0,
   "D3: 3.5-flash is below 50% fail-closed -- the word 'Half' is now wrong")
# the participation reading the same paragraph offers instead
fa = FTM["flash"]
ck(flat(f"{fa['zero_write_tasks']} zero-write tasks, {fa['capped']} step-caps, "
        f"{fa['loopy_capped']} of them loops") in flat(CAP24),
   f"D3: the participation triple != {fa['zero_write_tasks']}/"
   f"{fa['capped']}/{fa['loopy_capped']}")

# ---- D4. section 2.5, episode integrity ---------------------------------
CAP25 = capsec("### 2.5 Seven episodes", "## 3. How a task fails")
miss = {(eff, k): FT["cells"][eff]["models"][k]["missing_check_results"]
        for eff in ("low", "medium", "high") for k in K}
n_missing_eps = sum(miss.values())
ck(f"| **{n_missing_eps}** |" in CAP25,
   f"D4: hard-failure episode count != {n_missing_eps}")
affected = {ek: v for ek, v in miss.items() if v}
ck(len(affected) == 3, f"D4: affected cells changed: {sorted(affected)}")
label_of = {("medium", "flash"): "medium / gemini-3.5-flash",
            ("medium", "pro"): "medium / gemini-3.1-pro",
            ("high", "pro"): "high / gemini-3.1-pro"}
base_req = sum(v["required"]
               for v in FT["cells"]["low"]["models"]["opus5"]["total_checks"].values())
ck(f"vs. {base_req:,}" in CAP25, f"D4: baseline required units != {base_req}")
for (eff, k), _n in sorted(affected.items()):
    a = FT["cells"][eff]["models"][k]
    got = sum(v["required"] for v in a["total_checks"].values())
    row = caprow(CAP25, label_of[(eff, k)])
    ck(row is not None, f"D4: no denominator row for {eff}/{k}")
    if row:
        ck(as_f(row[0].replace(",", "")) == got,
           f"D4 {eff}/{k}: required units {row[0]!r} != {got}")
        ck(as_f(row[1].replace(",", "")) == got - base_req,
           f"D4 {eff}/{k}: delta {row[1]!r} != {got - base_req}")
# the favourable-omission bound the prose quotes, recomputed
for k, pat in (("flash", r"gemini-3\.5-flash\s*\n?from (\d+\.\d+)% to (\d+\.\d+)%"),
               ("pro", r"gemini-3\.1-pro from (\d+\.\d+)% to (\d+\.\d+)%")):
    a = FT["cells"]["medium"]["models"][k]
    got = sum(v["required"] for v in a["total_checks"].values())
    failed = sum(v["required"] for v in a["failed_checks"].values())
    now = 100.0 * (got - failed) / got
    then = 100.0 * (got - failed) / base_req
    m = re.search(pat, CAP25)
    ck(m is not None, f"D4 {k}: cannot find the omission-bound sentence")
    if m:
        ck(near(float(m.group(1)), now, 0.006),
           f"D4 {k}: stated {m.group(1)}%, recomputed {now:.2f}%")
        ck(near(float(m.group(2)), then, 0.006),
           f"D4 {k}: stated {m.group(2)}%, recomputed {then:.2f}%")
        ck(now - then < 0.1,
           f"D4 {k}: the 'under 0.1 points' bound no longer holds")

# ---- D5. section 3.1, the bucket table ----------------------------------
CAP31 = capsec("### 3.1 The six buckets", "### 3.2 Per-bucket")
LABEL = {"F1": "F1 content_msg", "F2": "F2 content_file", "F3": "F3 grounding",
         "F4": "F4 anchor", "F5": "F5 side_effect", "F6": "F6 window"}
for fid, b in FBUCKET:
    tc = FTM["opus5"]["total_checks"][b]
    row = caprow(CAP31, LABEL[fid])
    ck(row is not None, f"D5: no bucket row for {fid}")
    if row:
        ck(as_f(row[-2].replace(",", "")) == tc["required"],
           f"D5 {fid}: required {row[-2]!r} != {tc['required']}")
        ck(as_f(row[-1].replace(",", "")) == tc["diagnostic"],
           f"D5 {fid}: diagnostic {row[-1]!r} != {tc['diagnostic']}")
        ck(str(len(names[b])) in row[-3] or len(names[b]) == 1,
           f"D5 {fid}: name count in {row[-3]!r} != {len(names[b])}")
# the refinement identity against the cross-model report's check-type split
sem = (FTM["opus5"]["total_checks"]["content_msg"]["required"]
       + FTM["opus5"]["total_checks"]["content_file"]["required"])
fcs = sum(FTM["opus5"]["total_checks"][b][r] for b in ("grounding", "anchor")
          for r in ("required", "diagnostic"))
oth = (FTM["opus5"]["total_checks"]["side_effect"]["required"]
       + FTM["opus5"]["total_checks"]["window"]["required"])
capf = flat(cap)
f3_all = sum(FTM["opus5"]["total_checks"]["grounding"].values())
f4_all = sum(FTM["opus5"]["total_checks"]["anchor"].values())
ck(f"semantic {sem:,} = F1 "
   f"{FTM['opus5']['total_checks']['content_msg']['required']:,} + F2 "
   f"{FTM['opus5']['total_checks']['content_file']['required']:,}" in capf,
   f"D5: the semantic refinement identity is stale (sem={sem})")
ck(f"`field_contains` {fcs:,} = F3 {f3_all:,} + F4 {f4_all:,}" in capf,
   f"D5: the field_contains refinement identity is stale (fc={fcs})")
ck(f"remaining {oth} = F5 "
   f"{FTM['opus5']['total_checks']['side_effect']['required']} + F6 "
   f"{FTM['opus5']['total_checks']['window']['required']}" in capf,
   f"D5: the residual refinement identity is stale (other={oth})")
ck(sem + fcs + oth == req_all + diag_all,
   "D5: the three refinement identities do not partition the rubric")

# ---- D6. section 3.2, counts and rates ----------------------------------
CAP32 = capsec("### 3.2 Per-bucket required failure rates", "### 3.3 Two buckets")
tbl_counts, tbl_rates = CAP32.split("As required-failure")
for k in K:
    a = FTM[k]
    row = caprow(tbl_counts, a["display"])
    ck(row is not None, f"D6: no count row for {a['display']}")
    if row:
        for i, (fid, b) in enumerate(FBUCKET):
            want = (f"{a['failed_checks'][b]['required']}/"
                    f"{a['failed_checks'][b]['diagnostic']}")
            ck(row[i] == want, f"D6 {k}/{fid}: {row[i]!r} != {want}")
        ck(as_f(row[6].replace(",", "")) == a["failed_total"],
           f"D6 {k}: failed_total {row[6]!r} != {a['failed_total']}")
    rrow = caprow(tbl_rates, a["display"])
    ck(rrow is not None, f"D6: no rate row for {a['display']}")
    if rrow:
        for i, (fid, b) in enumerate(FBUCKET):
            ck(near(as_f(rrow[i]), req_rate(k, b), 0.051),
               f"D6 {k}/{fid}: rate {rrow[i]!r} != {req_rate(k, b):.1f}")
f1_min = min(req_rate(k, "content_msg") for k in K)
ck(near(f1_min, 50.0, 0.051) and f1_min >= 50.0,
   f"D6: the 'F1 at or above 50% for everyone' claim broke (min {f1_min:.1f})")
f1_req = FTM["opus5"]["total_checks"]["content_msg"]["required"]
ck(near(capprose(CAP32, r"required units, (\d+\.\d)%\)", "D6 F1 share"),
        100.0 * f1_req / req_all, 0.051),
   "D6: the F1 share of required units is stale")

# ---- D7. section 3.3, discrimination ------------------------------------
CAP33 = capsec("### 3.3 Two buckets", "### 3.4 What reasoning effort")
pass_all = [FTM[k]["pass_pct"] for k in K]
K8 = [k for k in K if k != "flash"]
pass_8 = [FTM[k]["pass_pct"] for k in K8]
for fid, b in FBUCKET:
    row = caprow(CAP33, LABEL[fid]) or caprow(CAP33, f"**{LABEL[fid]}**")
    ck(row is not None, f"D7: no discrimination row for {fid}")
    if not row:
        continue
    r9 = spearman([req_rate(k, b) for k in K], pass_all)
    r8 = spearman([req_rate(k, b) for k in K8], pass_8)
    s9 = max(req_rate(k, b) for k in K) - min(req_rate(k, b) for k in K)
    s8 = max(req_rate(k, b) for k in K8) - min(req_rate(k, b) for k in K8)
    # exact at the printed precision, not near(): a +-0.006 window around a
    # value like -0.7143 admits BOTH -0.71 and -0.72, so a corrupted cell slips
    # through. The negative control caught this.
    for i, (val, dp, nm) in enumerate(((r9, 2, "rho9"), (r8, 2, "rho8"),
                                       (s9, 1, "spread9"), (s8, 1, "spread8"))):
        ck(as_f(row[i]) == float(f"{val:.{dp}f}"),
           f"D7 {fid}: {nm} {row[i]!r} != {val:.{dp}f}")
# F5 is the only bucket whose spread is unchanged by dropping 3.5-flash
unchanged = [fid for fid, b in FBUCKET
             if near(max(req_rate(k, b) for k in K) - min(req_rate(k, b) for k in K),
                     max(req_rate(k, b) for k in K8) - min(req_rate(k, b) for k in K8),
                     0.001)]
ck("F5" in unchanged, "D7: F5's spread now moves when 3.5-flash is dropped")
sp9 = {fid: max(req_rate(k, b) for k in K) - min(req_rate(k, b) for k in K)
       for fid, b in FBUCKET}
sp8 = {fid: max(req_rate(k, b) for k in K8) - min(req_rate(k, b) for k in K8)
       for fid, b in FBUCKET}
ck(min(sp9, key=lambda f: sp9[f]) == "F5"
   and min(sp8, key=lambda f: sp8[f]) == "F5",
   "D7: F5 is no longer the narrowest bucket at n=9 and n=8")
# the prose must name the *other* invariant buckets, and the shrinking ones
m_inv = re.search(r"(F\d(?:(?:,| and) F\d)*) are spread-invariant too",
                  flat(CAP33))
ck(m_inv is not None, "D7: the spread-invariance sentence is missing")
ck(m_inv is not None
   and re.findall(r"F\d", m_inv.group(1)) == [f for f in unchanged if f != "F5"],
   f"D7: prose names the wrong co-invariant buckets; recomputed "
   f"{[f for f in unchanged if f != 'F5']}")
m_shr = re.search(r"spreads shrink on removal \u2014 ([^\u2014]+) \u2014",
                  flat(CAP33))
ck(m_shr is not None, "D7: the shrinking-buckets sentence is missing")
ck(m_shr is not None
   and re.findall(r"F\d", m_shr.group(1))
   == [f for f, _ in FBUCKET if f not in unchanged],
   f"D7: prose names the wrong shrinking buckets; recomputed "
   f"{[f for f, _ in FBUCKET if f not in unchanged]}")
# the interior-of-F3 argument the prose gives, recomputed
f3_lo = min(K, key=lambda k: req_rate(k, "grounding"))
f3_hi = max(K, key=lambda k: req_rate(k, "grounding"))
ck(f3_lo != "flash" and f3_hi != "flash",
   "D7: 3.5-flash is now an F3 extreme -- the interior argument is wrong")
ck(flat(f"({req_rate('flash', 'grounding'):.1f}%, between {DISP[f3_lo]}'s "
        f"{req_rate(f3_lo, 'grounding'):.1f}% and {DISP[f3_hi]}'s "
        f"{req_rate(f3_hi, 'grounding'):.1f}%)") in flat(CAP33),
   "D7: the F3 interior figures are stale")
# the F5-ordering table, and the two rank inversions the prose calls out
f5_order = sorted(K, key=lambda k: req_rate(k, "side_effect"))
overall = sorted(K, key=lambda k: -FTM[k]["pass_pct"])
for i, k in enumerate(f5_order, 1):
    row = caprow(CAP33, str(i))
    ck(row is not None, f"D7: no F5-ordering row {i}")
    if row:
        ck(_plain(row[0]).replace("**", "") == DISP[k],
           f"D7: F5 rank {i} is {row[0]!r}, recomputed {DISP[k]}")
        ck(near(as_f(row[1]), req_rate(k, "side_effect"), 0.051),
           f"D7: F5 rank {i} rate {row[1]!r} wrong")
        ck(as_f(row[2]) == overall.index(k) + 1,
           f"D7: F5 rank {i} overall rank {row[2]!r} != {overall.index(k) + 1}")
ORD = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
       6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth"}
ck(overall[0] == "opus5", f"D7: the overall winner is now {DISP[overall[0]]}")
ck(f"The overall winner is **{ORD[f5_order.index(overall[0]) + 1]}** on F5"
   in flat(CAP33),
   f"D7: the winner ranks {f5_order.index(overall[0]) + 1} on F5, not as written")
ck(f"the model ranked last overall is "
   f"**{ORD[f5_order.index(overall[-1]) + 1]}, not {ORD[len(K)]}**"
   in flat(CAP33),
   f"D7: the last-overall model ranks {f5_order.index(overall[-1]) + 1} on F5")
f6_best = min(FTM[k]["failed_checks"]["window"]["required"] for k in K)
f6_tot = FTM["opus5"]["total_checks"]["window"]["required"]
ck(f"{f6_best} of {f6_tot} failed" in CAP33,
   f"D7: best F6 result is {f6_best}/{f6_tot}")
ck(f6_tot - f6_best > 0,
   "D7: F6 now has zero passes -- it would be an unwinnable bucket, not a "
   "difficulty finding, and the prose must be rewritten")
f6_pass = f6_tot - f6_best
ck(any(f"{v} passes" in flat(CAP33) for v in numword(f6_pass)),
   f"D7: the F6 pass count is not {f6_pass}")
ck(any(f"{v} of the {f6_tot} window checks were passed" in flat(CAP33)
       for v in numword(f6_pass)),
   f"D7: the F6 callout does not say {f6_pass} of {f6_tot}")

# ---- D8. section 3.4, the effort sweep by bucket ------------------------
CAP34 = capsec("### 3.4 What reasoning effort", "## 4. Per-model profiles")
for fid, b in FBUCKET:
    tot, tot8 = {}, {}
    for eff in ("low", "medium", "high"):
        M = FT["cells"][eff]["models"]
        tot[eff] = sum(M[k]["failed_checks"][b]["required"] for k in K)
        tot8[eff] = sum(M[k]["failed_checks"][b]["required"] for k in K8)
    mono = sum(1 for k in K
               if FT["cells"]["low"]["models"][k]["failed_checks"][b]["required"]
               > FT["cells"]["medium"]["models"][k]["failed_checks"][b]["required"]
               > FT["cells"]["high"]["models"][k]["failed_checks"][b]["required"])
    d9 = 100.0 * (tot["high"] - tot["low"]) / tot["low"]
    d8 = 100.0 * (tot8["high"] - tot8["low"]) / tot8["low"]
    row = caprow(CAP34, LABEL[fid]) or caprow(CAP34, f"**{LABEL[fid]}**")
    ck(row is not None, f"D8: no effort row for {fid}")
    if not row:
        continue
    for i, eff in enumerate(("low", "medium", "high")):
        ck(as_f(row[i].replace(",", "")) == tot[eff],
           f"D8 {fid}/{eff}: {row[i]!r} != {tot[eff]}")
    ck(near(as_f(row[3]), d9, 0.051), f"D8 {fid}: lo->hi {row[3]!r} != {d9:.1f}%")
    ck(near(as_f(row[4]), d8, 0.051), f"D8 {fid}: excl {row[4]!r} != {d8:.1f}%")
    ck(row[5].startswith(f"{mono}/9"),
       f"D8 {fid}: monotone count {row[5]!r} != {mono}/9")
# the headline: F3 is the only bucket that materially improves, for everyone
f3_net = {k: FT["cells"]["high"]["models"][k]["failed_checks"]["grounding"]["required"]
          - FT["cells"]["low"]["models"][k]["failed_checks"]["grounding"]["required"]
          for k in K}
ck(all(v < 0 for v in f3_net.values()),
   f"D8: F3 is no longer net-negative for every model: {f3_net}")
for k, v in f3_net.items():
    ck(f"{DISP[k]} {v}" in capn or f"{DISP[k]} \u2212{-v}" in cap,
       f"D8: F3 net delta for {DISP[k]} is not {v}")
f2 = {k: (FT["cells"]["low"]["models"][k]["failed_checks"]["content_file"]["required"],
          FT["cells"]["high"]["models"][k]["failed_checks"]["content_file"]["required"])
      for k in K}
ck(f"F2 {f2['flash'][0]} \u2192 {f2['flash'][1]}" in cap,
   f"D8: the 3.5-flash F2 regression is not {f2['flash']}")
f4 = (FT["cells"]["low"]["models"]["flash"]["failed_checks"]["anchor"]["required"],
      FT["cells"]["high"]["models"]["flash"]["failed_checks"]["anchor"]["required"])
ck(f"F4 {f4[0]} \u2192 {f4[1]}" in cap,
   f"D8: the 3.5-flash F4 regression is not {f4}")

# ---- D9. section 5, difficulty ------------------------------------------
CAP5 = capsec("## 5. Difficulty is a property", "## 6. Interpretation")
scen_pass = collections.Counter()
scen_field = {}
for s in FTM["opus5"]["scenarios"]:
    scen_pass[s] = sum(FTM[k]["scenarios"][s]["passed"] for k in K)
    scen_field[s] = sum(FTM[k]["scenarios"][s]["mean_score"] for k in K) / 9.0
zero_pass = sorted((s for s, v in scen_pass.items() if v == 0),
                   key=lambda s: scen_field[s])
ck(len(zero_pass) == 3, f"D9: {len(zero_pass)} scenarios have zero passes")
for s in zero_pass:
    row = caprow(CAP5, s)
    ck(row is not None, f"D9: no zero-pass row for {s}")
    if row:
        ck(near(as_f(row[0]), scen_field[s], 0.0006),
           f"D9 {s}: field mean {row[0]!r} != {scen_field[s]:.3f}")
        ck(row[1].replace(" ", "") == "0/90", f"D9 {s}: passes {row[1]!r} != 0/90")
hardest = min(scen_field, key=lambda s: scen_field[s])
ck(f"`{hardest}` is the hardest scenario" in CAP5,
   f"D9: hardest scenario is {hardest}")
ck(sum(1 for k in K if FTM[k]["scenarios"][hardest]["passed"] == 0) == 9,
   f"D9: {hardest} is no longer in every model's zero-pass list")
cap5f = flat(CAP5)
easiest = max(scen_field, key=lambda s: scen_field[s])
ck(f"`{easiest}` is the easiest scenario at field mean "
   f"**{scen_field[easiest]:.3f}** and {scen_pass[easiest]}/90" in cap5f,
   f"D9: easiest scenario line is stale ({easiest}, {scen_field[easiest]:.3f}, "
   f"{scen_pass[easiest]}/90)")
for s, n in ((s, scen_pass[s]) for s in scen_pass if 0 < scen_pass[s] <= 3):
    ck(f"`{s}` ({n}" in cap5f, f"D9: near-zero scenario {s} ({n}/90) not listed")
zp_counts = {k: sum(1 for s in scen_pass if FTM[k]["scenarios"][s]["passed"] == 0)
             for k in K}
# parse the per-model zero-pass sentence rather than probing for bare digits:
# "gemini-3.6-flash and gpt-5.6-sol in 13" attaches one count to two models, and
# a bare `str(n) in CAP5` would pass on any stray digit anywhere in the section.
zp_split = cap5f.split("Zero-pass scenario counts per model")
ck(len(zp_split) == 2, "D9: the per-model zero-pass sentence is missing")
if len(zp_split) == 2:
    sent = zp_split[1].split(". ")[0]
    parts = re.split("(" + "|".join(re.escape(DISP[k]) for k in K) + ")", sent)
    seen, pending = {}, []
    for j in range(1, len(parts), 2):
        pending.append(parts[j])
        nums = re.findall(r"(?<![\d.])(\d+)(?![\d.])",
                          parts[j + 1] if j + 1 < len(parts) else "")
        if nums:
            for nm in pending:
                seen[nm] = int(nums[0])
            pending = []
    ck(not pending, f"D9: no count follows {pending} in the zero-pass sentence")
    ck(seen == {DISP[k]: zp_counts[k] for k in K},
       f"D9: the per-model zero-pass sentence disagrees with the recomputed "
       f"counts {sorted((DISP[k], zp_counts[k]) for k in K)}: got "
       f"{sorted(seen.items())}")
ck(f"in **{zp_counts['flash']} of 60**" in cap5f,
   f"D9: 3.5-flash zero-pass count != {zp_counts['flash']}")
ck(zp_counts["opus"] == zp_counts["opus5"]
   and f"in just {zp_counts['opus5']} each" in cap5f,
   f"D9: the two Claude opus zero-pass counts are {zp_counts['opus']}/"
   f"{zp_counts['opus5']}")

# head-to-head
H2H = FT["cells"]["medium"]["head_to_head"]
# NB: the end marker must be a standalone rule. Splitting on a bare '---' cuts
# the section at the table's own header separator and hides every data row.
CAP52 = capsec("### 5.2 Head-to-head", "\n---\n")
net = {}
for k in K:
    w = sum(H2H[k][o]["win"] for o in K if o != k)
    l = sum(H2H[k][o]["loss"] for o in K if o != k)
    t = sum(H2H[k][o]["tie"] for o in K if o != k)
    net[k] = (w, l, t, w - l)
for k in K:
    row = caprow(CAP52, DISP[k])
    ck(row is not None, f"D9: no head-to-head row for {DISP[k]}")
    if row:
        w, l, t, n = net[k]
        ck(as_f(row[0]) == w, f"D9 {k}: h2h wins {row[0]!r} != {w}")
        ck(as_f(row[1]) == l, f"D9 {k}: h2h losses {row[1]!r} != {l}")
        ck(as_f(row[2]) == t, f"D9 {k}: h2h ties {row[2]!r} != {t}")
        ck(as_f(row[3]) == n, f"D9 {k}: h2h net {row[3]!r} != {n}")
o5s = H2H["opus5"]["sol"]
ck(o5s["win"] == o5s["loss"] and o5s["tie"] == 0,
   f"D9: opus-5 vs sol is no longer an even split: {o5s}")
ck(f"is {o5s['win']}\u2013{o5s['loss']} with zero ties" in cap5f,
   f"D9: the opus-5/sol split line is stale ({o5s})")
o54 = H2H["opus5"]["opus"]
ck(f"is {o54['win']}\u2013{o54['loss']} with {o54['tie']} ties" in cap5f,
   f"D9: the opus-5/opus-4.8 line is stale ({o54})")
ck(f"takes {net['flash'][0]} scenario comparisons, including "
   f"{H2H['flash']['pro']['win']} against gemini-3.1-pro" in cap5f,
   f"D9: the 3.5-flash win line is stale ({net['flash'][0]}, "
   f"{H2H['flash']['pro']['win']})")

# the it_deployment_gate_v02 inversion
IDG = "it_deployment_gate_v02"
ck(FTM["opus5"]["scenarios"][IDG]["passed"] == 0,
   "D9: opus-5 now passes it_deployment_gate_v02")
n_flash = FTM["flash"]["scenarios"][IDG]["passed"]
ck(f"passes it {n_flash} times" in cap5f,
   f"D9: the prose does not say 3.5-flash passes {IDG} {n_flash} times")
ck(n_flash > 0, f"D9: 3.5-flash no longer passes {IDG} -- the inversion is gone")
share = round(100.0 * n_flash / FTM["flash"]["passed"])
ck(f"**{share}% of" in CAP5,
   f"D9: those passes are {share}% of 3.5-flash's haul")
ck(near(capprose(CAP5, rf"field mean (\d\.\d+), {scen_pass[IDG]}/90",
                 "D9 idg field"),
        scen_field[IDG], 0.0006), "D9: it_deployment_gate_v02 field mean is stale")
ck(f"{scen_pass[IDG]}/90 passes" in CAP5,
   f"D9: it_deployment_gate_v02 pass count != {scen_pass[IDG]}")
o5_zero = sorted(s for s in scen_pass if FTM["opus5"]["scenarios"][s]["passed"] == 0)
ck(IDG in o5_zero and len(o5_zero) == 5,
   f"D9: opus-5's zero-pass list changed: {o5_zero}")
gap = FTM["opus5"]["scenarios"][IDG]["mean_score"] - scen_field[IDG]
ck(near(capprose(CAP5, r"gap against the field \(-(\d\.\d+)\)", "D9 idg gap"),
        -gap, 0.006), f"D9: the it_deployment_gate_v02 gap != {gap:.2f}")
worst_gap = min(scen_field,
                key=lambda s: FTM["opus5"]["scenarios"][s]["mean_score"] - scen_field[s])
ck(worst_gap == IDG,
   f"D9: opus-5's largest negative gap is now {worst_gap}")

# ---- D10. Appendix A, all 540 cells recomputed --------------------------
APX = capsec("## Appendix A", "## Cross-references")
ck("<!-- APPENDIX-A-BEGIN -->" in APX and "<!-- APPENDIX-A-END -->" in APX,
   "D10: Appendix A sentinels are missing")
apx_rows = [ln for ln in APX.splitlines()
            if ln.strip().startswith("| `") and "<br>" in ln]
ck(len(apx_rows) == 60, f"D10: Appendix A has {len(apx_rows)} rows, expected 60")
seen = set()
prev_field = -1.0
for ln in apx_rows:
    parts = [c.strip() for c in ln.strip().strip("|").split("|")]
    slug = _plain(parts[0])
    seen.add(slug)
    ck(slug in scen_field, f"D10: unknown scenario {slug!r}")
    if slug not in scen_field:
        continue
    ck(near(float(parts[1]), scen_field[slug], 0.0006),
       f"D10 {slug}: field mean {parts[1]!r} != {scen_field[slug]:.3f}")
    ck(float(parts[1]) >= prev_field - 1e-9,
       f"D10: Appendix A is not sorted hardest-first at {slug}")
    prev_field = float(parts[1])
    for i, k in enumerate(K):
        cell = parts[2 + i]
        mean_s, _, pass_s = cell.partition("<br>")
        want = FTM[k]["scenarios"][slug]
        ck(near(float(_plain(mean_s)), want["mean_score"], 0.0051),
           f"D10 {slug}/{k}: mean {mean_s!r} != {want['mean_score']}")
        ck(int(pass_s) == want["passed"],
           f"D10 {slug}/{k}: passed {pass_s!r} != {want['passed']}")
        ck(("**" in mean_s) == (want["passed"] > 0),
           f"D10 {slug}/{k}: bold marker disagrees with passed={want['passed']}")
ck(seen == set(scen_field), "D10: Appendix A scenario set != the taxonomy's 60")

# ---- D11. cross-document consistency and retracted claims ---------------
ck("gpt-5.5" not in cap.replace("`gpt-5.5` is not in this corpus", "")
   .replace("The `OpenAI/5_5` directories", ""),
   "D11: gpt-5.5 is reported somewhere in the capability document")
for banned in ["ten models", "four OpenAI", "nine models (three Gemini, three "
               "Claude, four OpenAI)"]:
    ck(banned not in cap, f"D11: reference-corpus phrasing leaked in: {banned!r}")
# every deep-dive this document indexes must exist -- as a link *and* on disk,
# because an index entry pointing at a file nobody wrote is the failure mode a
# link-text check cannot see
for slug in ["gemini-3-5-flash", "gemini-3-6-flash", "gemini-3-1-pro",
             "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
             "gpt-5-6-sol", "gpt-5-6-luna", "gpt-5-6-terra"]:
    ck(f"./{slug}-solver-failures.md" in cap,
       f"D11: capability doc does not index {slug}")
    ck(os.path.exists(os.path.join(HERE, f"{slug}-solver-failures.md")),
       f"D11: indexed deep-dive {slug}-solver-failures.md does not exist")
# bare family words are ambiguous across the nine-model field
for bare in [r"\bthe opus model\b", r"\bthe flash model\b", r"\bthe sonnet\b",
             r"\bthe pro model\b"]:
    ck(re.search(bare, cap) is None, f"D11: bare family word {bare!r} in prose")

# ---- D12. the main report's Cross-references section ---------------------
# The main report is the entry point, so its index is the one that decides
# whether the companion documents are discoverable at all. Three failure modes
# are distinct and all silent: an entry that is missing, an entry whose target
# was never written, and an entry that rots when a file is renamed.
ck("## Cross-references" in md, "D12: main report has no Cross-references section")
# Fall back to the empty string rather than indexing a one-element split: a
# missing section must be reported as a failed check, not an audit traceback.
xref = md.split("## Cross-references", 1)[1] if "## Cross-references" in md else ""
ck("./eval-results-capability-analysis.md" in xref,
   "D12: main report does not link the capability analysis")
for slug in ["gemini-3-5-flash", "gemini-3-6-flash", "gemini-3-1-pro",
             "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
             "gpt-5-6-sol", "gpt-5-6-luna", "gpt-5-6-terra"]:
    ck(f"./{slug}-solver-failures.md" in xref,
       f"D12: main report does not index {slug}")
ck(len(re.findall(r"\]\([^)]*solver-failures\.md\)", xref)) == 9,
   "D12: Cross-references does not link exactly 9 deep-dives")
# Every relative link in the *whole* document, not just this section, so a typo
# in an artifact path or a stale link pasted into the body is caught too.
links = sorted(set(re.findall(r"\]\((?:\./)?([A-Za-z0-9_./-]+"
                              r"\.(?:md|html|json|py))\)", md)))
ck(len(links) >= 6, f"D12: main report has only {len(links)} relative links")
for tgt in links:
    ck(os.path.exists(os.path.join(HERE, tgt)),
       f"D12: link target does not exist: {tgt}")

# ---- D13. section 5.3, the designed difficulty axes ----------------------
# Section 5.3 is the only part of the analysis that reads the benchmark's
# *intent* rather than its outcome: it joins each task to the scenario seed it
# was generated from and groups by the knobs that seed declares. Every other
# pass audits a number against a derived artifact; here the grouping itself can
# be wrong, so this pass rebuilds the buckets from two sources the reporting
# tools do not share -- the seed JSON, and difficulty_axes.json, which is a
# per-task dump written by scan_difficulty_axes.py straight off the raw
# trajectories -- and recomputes all 444 rendered figures from scratch.
#
# WHAT GROUPS A TASK. The generator concatenates the seed's `conflict_taxonomy`
# and `difficulty_challenges` into one token list and resolves it through the
# conflict plugin registry (conflict_llm_pipeline.py:1582). So the grouping is
# the PLUGIN SET, and `archetype` -- which an earlier version of this pass keyed
# off -- never reaches the router at all. The archetype map survives only as a
# footnote in 5.3 contrasting it with the plugin specs; it groups nothing, and
# the checks below no longer read it.
#
# The registry is rebuilt here by PARSING the 23 plugin modules, not importing
# them. design_axis_slice.py imports the live registry and build_behavior_report.py
# carries a hand copy, so a textual third derivation keeps all three honest:
# importing here would collapse this pass into design_axis_slice's chain, and
# checking the hand copies only against each other would pass if they were wrong
# in the same way.
PLUGIN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
                          "clasharness", "scenario", "plugins")
SEED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
                        "clasharness", "data", "scenario_seeds",
                        "conflict_vlm_v1")
# Modules in the plugin package that are machinery, not plugins.
PLUGIN_SKIP = {"__init__.py", "registry.py", "spec.py", "base.py"}


def parse_plugin_specs(dirpath):
    """name -> (category, is_mechanic, taxonomy_tokens) from the plugin sources.

    Deliberately textual, for the reason given above. The `SPEC =
    ConflictPluginSpec(...)` block is a class attribute, so the closing paren is
    at two-space indent; anchoring on that is brittle by design -- a reformat
    should fail this pass loudly rather than let it silently parse fewer
    plugins than exist.
    """
    out = {}
    for f in sorted(glob.glob(os.path.join(dirpath, "*.py"))):
        if os.path.basename(f) in PLUGIN_SKIP:
            continue
        src = open(f, encoding="utf-8").read()
        for m in re.finditer(r"SPEC\s*=\s*ConflictPluginSpec\((.*?)\n  \)", src, re.S):
            b = m.group(1)
            nm = re.search(r'name\s*=\s*"([^"]+)"', b)
            cat = re.search(r'category\s*=\s*"([^"]+)"', b)
            eff = re.search(r"emits_effects\s*=\s*\(([^)]*)\)", b, re.S)
            tok = re.search(r"taxonomy_tokens\s*=\s*\(([^)]*)\)", b, re.S)
            if nm and cat:
                out[nm.group(1)] = (cat.group(1),
                                    bool(eff and eff.group(1).strip()),
                                    bool(tok and tok.group(1).strip()))
    return out


def parse_literal_map(path, var):
    """A multi-line `var = {...}` LITERAL as written in `path`, else None.

    The opening brace must end its line. Without that anchor the lazy `.*?`
    runs across newlines and a one-line dict COMPREHENSION -- which is what
    design_axis_slice.py builds PLUGIN_CATEGORY with, since it imports the
    registry -- swallows everything up to the next literal's closing brace and
    reports that unrelated map as this variable's contents.
    """
    if not os.path.exists(path):
        return None
    body = re.search(rf"^{var}\s*[:=][^{{\n]*\{{[ \t]*\n(.*?)^\}}",
                     open(path, encoding="utf-8").read(), re.S | re.M)
    return dict(re.findall(r'"(\w+)"\s*:\s*"(\w+)"', body.group(1))) if body else None


PLUGIN_SPECS = parse_plugin_specs(PLUGIN_DIR)
ck(len(PLUGIN_SPECS) == 23,
   f"D13: parsed {len(PLUGIN_SPECS)} conflict plugins from {PLUGIN_DIR}, "
   f"expected 23")
SPEC_CAT = {n: v[0] for n, v in PLUGIN_SPECS.items()}
MECHANIC = {n for n, v in PLUGIN_SPECS.items() if v[1]}
CHALLENGE = {n for n, c in SPEC_CAT.items() if c == "challenge"}
ck(len(CHALLENGE) == 7,
   f"D13: {len(CHALLENGE)} plugins are category=challenge, expected 7")
ck(len(MECHANIC) == 13,
   f"D13: {len(MECHANIC)} plugins declare emits_effects, expected 13")
# Routing is "keep the known names" only because no plugin aliases itself. If
# one ever declares taxonomy_tokens, seed_buckets below silently stops matching
# the generator, so this is asserted rather than assumed.
ck(not [n for n, v in PLUGIN_SPECS.items() if v[2]],
   f"D13: plugins now declare taxonomy_tokens "
   f"{[n for n, v in PLUGIN_SPECS.items() if v[2]]}; token->plugin is no longer "
   f"1:1 and the bucket derivation below is wrong")

# The hand copies. build_behavior_report.py and emit_solver_failures.py must
# each carry one (they are the chains that deliberately do not import);
# design_axis_slice.py must NOT, because it imports the live registry and a copy
# there would fake an extra derivation. Both copies are pinned to the PARSED
# sources, never to each other -- two hand copies agreeing proves only that they
# were typed from the same wrong place.
for _rel in ("build_behavior_report.py", "emit_solver_failures.py"):
    _copy = parse_literal_map(os.path.join(HERE, "tools", _rel),
                              "PLUGIN_CATEGORY")
    ck(_copy is not None,
       f"D13: tools/{_rel} no longer carries a PLUGIN_CATEGORY literal; its "
       f"derivation is supposed to be independent of the registry import")
    ck(_copy == SPEC_CAT,
       f"D13: tools/{_rel} PLUGIN_CATEGORY has drifted from the plugin sources"
       + (f": {sorted(set(_copy.items()) ^ set(SPEC_CAT.items()))[:4]}"
          if _copy is not None and _copy != SPEC_CAT else ""))
ck(parse_literal_map(os.path.join(HERE, "tools", "design_axis_slice.py"),
                     "PLUGIN_CATEGORY") is None,
   "D13: tools/design_axis_slice.py now hand-copies PLUGIN_CATEGORY; it is "
   "supposed to import the registry so the derivations stay independent")
# No tool may still be grouping by the archetype map -- it does not route.
for _rel in ("design_axis_slice.py", "build_behavior_report.py",
             "emit_solver_failures.py"):
    _p = os.path.join(HERE, "tools", _rel)
    _arch = parse_literal_map(_p, "ARCHETYPE_CATEGORY")
    if _arch is None:
        continue
    _txt = open(_p, encoding="utf-8").read()
    ck(re.search(r"ARCHETYPE_CATEGORY\s*[.\[]|ARCHETYPE_CATEGORY\.get", _txt) is None,
       f"D13: tools/{_rel} still reads ARCHETYPE_CATEGORY; `archetype` never "
       f"reaches the router, so it must not group anything")

SEEDS = {os.path.basename(f)[:-len(".json")]: json.load(open(f, encoding="utf-8"))
         for f in sorted(glob.glob(os.path.join(SEED_DIR, "*.json")))}
ck(len(SEEDS) == 20, f"D13: {len(SEEDS)} scenario seeds found, expected 20")


def seed_plugins(sd):
    """The plugins one seed routes to, off the PARSED registry.

    Mirrors conflict_llm_pipeline.py:1582: the two token fields are
    concatenated and unknown tokens are dropped (`sla_support_response` carries
    a `subagent_delay` token that is not a registered plugin). An empty token
    list would resolve to every plugin in the real registry, so it is refused
    here rather than quietly producing 23 buckets.
    """
    toks = ((sd.get("conflict_taxonomy") or [])
            + (sd.get("difficulty_challenges") or []))
    out = []
    for t in toks:
        if t in SPEC_CAT and t not in out:
            out.append(t)
    return out


def seed_buckets(sd):
    """One seed -> every (axis, bucket) it belongs to. Mirrors the emitters,
    but off the parsed plugin sources rather than off any tool's copy."""
    plugins = seed_plugins(sd)
    return ([("plugin", p) for p in plugins]
            + [("category", c) for c in sorted(
                {SPEC_CAT[p] for p in plugins if p not in CHALLENGE})]
            + [("vision", "vision_graded" if sd.get("force_vision_locus")
                else "not_vision_graded"),
               ("async", "async_required" if sd.get("force_async_wait")
                else "no_async")]
            + [("challenge", t) for t in (sd.get("difficulty_challenges") or [])])


# Seeds per bucket, model-independent. The rendered tables carry this column so
# a reader can see that most plugin rows rest on one or two scenarios.
# NOTE the dedupe by NAME, not by (axis, bucket). The seven challenge tokens are
# plugins too, so seed_buckets yields each of them on BOTH the plugin and the
# challenge axis; every accumulator below is keyed by bucket name alone and
# would count those seeds and tasks twice without this. The doc renders those
# buckets in two tables on purpose and D13 checks the two agree.
DXSEEDS = collections.Counter()
for _slug, _sd in SEEDS.items():
    for _b in {b for _a, b in seed_buckets(_sd)}:
        DXSEEDS[_b] += 1
# Every seed must route somewhere. An empty token list resolves to the WHOLE
# registry in the real router, so a seed that fell through would quietly join
# all 23 plugin buckets and inflate every one of them.
ck(all(seed_plugins(_sd) for _sd in SEEDS.values()),
   f"D13: seed(s) route to no conflict plugin: "
   f"{sorted(s for s, d in SEEDS.items() if not seed_plugins(d))}")

# The rendered row order shared by 5.3's delta table and 4.3's SF11: the four
# knob arms first, then the plugins by coverage, widest first, ties by name.
# Recomputed from DXSEEDS rather than transcribed, so a row that moves without
# its seed count moving fails the order checks that read this.
DXKNOBS = ["async_required", "no_async", "vision_graded", "not_vision_graded"]
DXORDER = DXKNOBS + sorted(SPEC_CAT, key=lambda p: (-DXSEEDS[p], p))
# Control arms are not levers: nothing was designed to make `no_async` hard, it
# is just the complement of a knob that was. Both sections count levers in
# their closing claims, so the audit has to draw the line the emitters draw.
DXLEVERS = [b for b in DXORDER if b not in ("no_async", "not_vision_graded")]
ck(len(DXORDER) == 27 and len(DXLEVERS) == 25,
   f"D13: {len(DXORDER)} rows / {len(DXLEVERS)} levers, expected 27 / 25")

# [n, passed, score_total] per (model, bucket), and the same per model overall
DXB = collections.defaultdict(lambda: [0, 0, 0.0])
DXBASE = collections.defaultdict(lambda: [0, 0, 0.0])
dx_unmapped = set()
for _t in AX["tasks"]:
    _sd = SEEDS.get(re.sub(r"_v\d+$", "", _t["scen"]))
    if _sd is None:
        dx_unmapped.add(_t["scen"])
        continue
    _pairs = {b for _a, b in seed_buckets(_sd)}     # by name; see DXSEEDS above
    for k in K:
        _score, _passed = _t[NAME[k]]
        for _acc in [DXBASE[k]] + [DXB[(k, b)] for b in _pairs]:
            _acc[0] += 1
            _acc[1] += 1 if _passed else 0
            _acc[2] += _score
ck(not dx_unmapped,
   f"D13: {len(dx_unmapped)} scenario(s) join to no seed: "
   f"{sorted(dx_unmapped)[:3]}")
ck(all(DXBASE[k][0] == 600 for k in K),
   f"D13: baseline is not 600 tasks per model: "
   f"{sorted({DXBASE[k][0] for k in K})}")


def dxpct(k, b):
    c = DXB[(k, b)]
    return 100.0 * c[1] / c[0] if c[0] else 0.0


def dxbase_pct(k):
    return 100.0 * DXBASE[k][1] / DXBASE[k][0]


def dxpool(b):
    """One bucket's pass rate with all nine models pooled."""
    cs = [DXB[(k, b)] for k in K]
    n = sum(c[0] for c in cs)
    return 100.0 * sum(c[1] for c in cs) / n if n else 0.0


def dxbelow(b):
    """How many of the nine models this bucket puts below their own baseline."""
    return sum(1 for k in K if dxpct(k, b) < dxbase_pct(k))


def dxrange(b):
    """(most negative, most positive) delta against own baseline, over the nine."""
    ds = sorted(dxpct(k, b) - dxbase_pct(k) for k in K)
    return ds[0], ds[-1]


def dxsig(v):
    """A signed delta rendered the way the emitters render it: U+2212, 1dp."""
    return f"{v:+.1f}".replace("-", "\u2212")


CAP53 = capsec("### 5.3", "## 6. Interpretation")
# Column order is load-bearing, so it is asserted rather than assumed. These are
# the RENDERED labels (design_axis_slice.MODEL_LABEL), which are not NAME[] --
# NAME hyphenates.
DX_MODEL_LABEL = ["3.5 flash", "3.6 flash", "3.1 pro", "opus 4.8", "opus 5",
                  "sonnet 5", "5.6-sol", "5.6-luna", "5.6-terra"]

# Attribute every row to the header above it. Sniffing the column layout from
# the row width no longer works: the plugin table carries an extra `category`
# column and the delta table an extra `below baseline` column, so both are 13
# wide and mean different things. Cells are then addressed BY HEADER NAME.
DX_TABLES = []
_cur = None
for _ln in CAP53.splitlines():
    if _ln.startswith("|---"):
        continue
    if _ln.startswith("|"):
        _c = [x.strip() for x in _ln.strip("|").split("|")]
        if _c[0] in ("bucket", "lever"):
            _cur = (_c, [])
            DX_TABLES.append(_cur)
        elif _cur is not None:
            _cur[1].append(_c)
    else:
        _cur = None

ck(len(DX_TABLES) == 6,
   f"D13: {len(DX_TABLES)} tables in 5.3, expected 6 "
   f"(plugin, category, vision, async, challenge, delta)")
ck(len(re.findall(r"^#### ", CAP53, re.M)) == 10,
   f"D13: {len(re.findall(chr(94) + '#### ', CAP53, re.M))} subheads in 5.3, "
   f"expected 10 (6 tables + 4 analysis)")
for _hdr, _rs in DX_TABLES:
    ck([n for n in _hdr if n in set(DX_MODEL_LABEL)] == DX_MODEL_LABEL,
       f"D13: table {_hdr[0]!r} does not carry the nine models in order: {_hdr}")
    ck("seeds" in _hdr and "tasks" in _hdr,
       f"D13: table {_hdr[0]!r} is missing the seeds/tasks columns: {_hdr}")
    ck(len(_rs) > 0, f"D13: table {_hdr[0]!r} has no rows")

dx_rows = collections.defaultdict(list)
for _hdr, _rs in DX_TABLES:
    _ci = {n: i for i, n in enumerate(_hdr)}
    for _c in _rs:
        ck(len(_c) == len(_hdr),
           f"D13: row {_c[0]!r} has {len(_c)} cells, header has {len(_hdr)}")
        dx_rows[_c[0]].append((_hdr, _ci, _c))
ck(len(dx_rows) == 33,
   f"D13: {len(dx_rows)} distinct bucket labels in 5.3, expected 33 "
   f"(23 plugin + 6 category + 2 vision + 2 async; the 7 challenge tokens are "
   f"plugins too and reuse those labels)")

# Every plugin must appear as a row, or a lever silently vanished from the doc.
for _p in sorted(SPEC_CAT):
    ck(_p in dx_rows, f"D13: plugin {_p!r} has no row in the 5.3 tables")

for _name, _variants in sorted(dx_rows.items()):
    ck(all(DXB[(k, _name)][0] for k in K),
       f"D13: table row {_name!r} matches no bucket recomputed from the seeds")
    for _hdr, _ci, _cells in _variants:
        ck(as_f(_cells[_ci["tasks"]]) == DXB[(K[0], _name)][0],
           f"D13 {_name}: task count {_cells[_ci['tasks']]!r} != "
           f"{DXB[(K[0], _name)][0]}")
        ck(as_f(_cells[_ci["seeds"]]) == DXSEEDS[_name],
           f"D13 {_name}: seed count {_cells[_ci['seeds']]!r} != "
           f"{DXSEEDS[_name]}")
        if "category" in _ci:
            ck(_cells[_ci["category"]] == SPEC_CAT.get(_name),
               f"D13 {_name}: category {_cells[_ci['category']]!r} != "
               f"{SPEC_CAT.get(_name)!r}")
        _pcts = [dxpct(k, _name) for k in K]
        if "below baseline" in _ci:      # the delta table
            for k in K:
                _d = _pcts[K.index(k)] - dxbase_pct(k)
                ck(_cells[_ci[DX_MODEL_LABEL[K.index(k)]]] == f"{_d:+.1f}",
                   f"D13 delta {_name}/{NAME[k]}: "
                   f"{_cells[_ci[DX_MODEL_LABEL[K.index(k)]]]!r} != {_d:+.1f}")
            _below = sum(1 for _i, k in enumerate(K)
                         if _pcts[_i] - dxbase_pct(k) < 0)
            ck(_cells[_ci["below baseline"]] == f"{_below}/9",
               f"D13 {_name}: 'below baseline' "
               f"{_cells[_ci['below baseline']]!r} != {_below}/9")
            continue
        # Bold is generated as the row maximum, so it is checked as one: a
        # hand-applied emphasis anywhere in these tables fails here, and so
        # does a maximum that moved without the bold moving with it.
        _mx = max(_pcts)
        for _i, k in enumerate(K):
            _cell = _cells[_ci[DX_MODEL_LABEL[_i]]]
            _c3 = DXB[(k, _name)]
            ck(_cell.strip("*") == f"{_pcts[_i]:.1f}% ({_c3[2] / _c3[0]:.2f})",
               f"D13 {_name}/{NAME[k]}: {_cell!r} != "
               f"{_pcts[_i]:.1f}% ({_c3[2] / _c3[0]:.2f})")
            ck(_cell.startswith("**") == (_pcts[_i] == _mx),
               f"D13 {_name}/{NAME[k]}: bold={_cell.startswith('**')} but "
               f"{_pcts[_i]:.1f}% vs row max {_mx:.1f}%")

ck(" | ".join(f"{dxbase_pct(k):.1f}% ({DXBASE[k][2] / DXBASE[k][0]:.2f})"
              for k in K) in CAP53,
   "D13: the baseline line under the 5.3 tables is stale")

# The coverage paragraph: seed-level counts, not task-level. The per-token seed
# counts that used to be asserted from prose here are now a `seeds` COLUMN, and
# every one of its cells is checked against DXSEEDS in the table loop above --
# a stronger check over 33 buckets instead of a regex over 7. What is left to
# check is the framing the paragraph puts around those columns.
_CAP53F = flat(CAP53)
ck(f"All {len(SPEC_CAT)} registered conflict plugins are active" in _CAP53F,
   f"D13: the paragraph does not claim all {len(SPEC_CAT)} plugins are active")
_via_ch, _via_tx = set(), set()
for _sd in SEEDS.values():
    _via_ch |= {t for t in (_sd.get("difficulty_challenges") or []) if t in SPEC_CAT}
    _via_tx |= {t for t in (_sd.get("conflict_taxonomy") or []) if t in SPEC_CAT}
ck(_via_ch | _via_tx == set(SPEC_CAT),
   f"D13: the seeds do not activate every plugin; missing "
   f"{sorted(set(SPEC_CAT) - (_via_ch | _via_tx))}")
ck(not (_via_ch & _via_tx),
   f"D13: {sorted(_via_ch & _via_tx)} appear in BOTH seed token fields, so the "
   f"'no token appearing in both fields' claim is wrong")
ck(f"{len(_via_ch)} reached through `difficulty_challenges` and the other "
   f"{len(_via_tx)} through `conflict_taxonomy`" in _CAP53F,
   f"D13: the {len(_via_ch)}/{len(_via_tx)} split by seed field is stale")
# per-seed activation spread, e.g. "4 to 7 of them (6 seeds activate 4, ...)"
_spread = collections.Counter(len(seed_plugins(_sd)) for _sd in SEEDS.values())
ck(f"Each seed activates {min(_spread)} to {max(_spread)} of them" in _CAP53F,
   f"D13: the per-seed activation range is {min(_spread)}-{max(_spread)}")
for _cnt, _ns in sorted(_spread.items()):
    _w = "one" if _ns == 1 else str(_ns)
    ck(re.search(rf"\b{_w} (?:seeds? )?activates? {_cnt}\b", _CAP53F),
       f"D13: the paragraph does not say {_w} seed(s) activate {_cnt} plugins")
_solo = sorted(p for p in SPEC_CAT if DXSEEDS[p] == 1)
ck(f"**{len(_solo)} of the {len(SPEC_CAT)} plugins are carried by a single "
   f"seed**" in _CAP53F,
   f"D13: {len(_solo)} plugins rest on one seed {_solo}, prose disagrees")
# the three widest buckets, named with their counts
for _p, _n in collections.Counter(
        {p: DXSEEDS[p] for p in SPEC_CAT}).most_common(3):
    ck(re.search(rf"`{_p}`[^.]*?\b{_n}\b|\b{_n}\b[^.]*?`{_p}`", _CAP53F),
       f"D13: coverage paragraph does not say `{_p}` is carried by {_n} seeds")
ck(f"`force_vision_locus` is set on "
   f"{sum(1 for s in SEEDS.values() if s.get('force_vision_locus'))} seeds"
   in _CAP53F, "D13: the force_vision_locus seed count is stale")
ck(f"`archetype` is declared on "
   f"{sum(1 for s in SEEDS.values() if s.get('archetype'))} seeds" in _CAP53F,
   "D13: the archetype seed count is stale")
ck("never reaches the router" in _CAP53F,
   "D13: 5.3 no longer says `archetype` does not route, but it does not")
_async_seeds = sorted(n for n, s in SEEDS.items() if s.get("force_async_wait"))
ck(len(_async_seeds) == 1 and f"`{_async_seeds[0]}`" in CAP53,
   f"D13: the corpus has {len(_async_seeds)} async seeds {_async_seeds}, and "
   f"the prose names one")
# flat(), not raw: the bold span wraps a line in the source markdown.
ck("**`force_async_wait` on exactly one**" in _CAP53F,
   "D13: the 'exactly one async seed' claim is gone while the seed set still "
   "has one")

# the async callout: three models at zero, opus-5 the field maximum, and the
# delta range quoted in the box
_azero = sorted(FTM[k]["display"] for k in K if dxpct(k, "async_required") == 0.0)
ck(len(_azero) == 3 and all(d in CAP53 for d in _azero),
   f"D13: {len(_azero)} models score 0.0% on async_required {_azero}, and the "
   f"callout names three")
_amax = max(K, key=lambda k: dxpct(k, "async_required"))
ck(f"{FTM[_amax]['display']}'s {dxpct(_amax, 'async_required'):.1f}% is the "
   f"field maximum" in flat(CAP53),
   f"D13: the async field maximum is {FTM[_amax]['display']} at "
   f"{dxpct(_amax, 'async_required'):.1f}%")
_adelta = sorted(dxpct(k, "async_required") - dxbase_pct(k) for k in K)
# Matched without the trailing "pass points": the claim sits in a blockquote,
# and flat() collapses the newline but leaves the "> " marker, so any pattern
# spanning the wrap would be asserting the position of a line break.
ck(f"here: \u2212{abs(_adelta[-1]):.1f} to \u2212{abs(_adelta[0]):.1f}"
   in CAP53,
   f"D13: the async delta range is {_adelta[-1]:.1f} to {_adelta[0]:.1f}")
ck("pass points" in flat(CAP53).split(
       f"\u2212{abs(_adelta[-1]):.1f} to \u2212{abs(_adelta[0]):.1f}")[-1][:20],
   "D13: the async delta range is not stated in pass points")

# sla_support_response is the point-like-failure claim, and it is a ranking:
# 7th hardest of 20 by score but 2nd lowest by passes is the whole argument.
_sm, _sp = {}, collections.Counter()
for _scen in FTM["opus5"]["scenarios"]:
    _b = re.sub(r"_v\d+$", "", _scen)
    _sm.setdefault(_b, []).append(
        sum(FTM[k]["scenarios"][_scen]["mean_score"] for k in K) / 9.0)
    _sp[_b] += sum(FTM[k]["scenarios"][_scen]["passed"] for k in K)
_smean = {b: sum(v) / len(v) for b, v in _sm.items()}
SLA = "sla_support_response"
ck(sorted(_smean, key=lambda b: _smean[b]).index(SLA) + 1 == 7
   and "7th hardest of the 20 seeds" in flat(CAP53),
   f"D13: {SLA} is #{sorted(_smean, key=lambda b: _smean[b]).index(SLA) + 1} "
   f"hardest by field mean, not 7th")
ck(f"score ({_smean[SLA]:.3f})" in flat(CAP53),
   f"D13: {SLA} field mean != {_smean[SLA]:.3f}")
ck(sorted(_sp, key=lambda b: _sp[b]).index(SLA) + 1 == 2
   and "2nd lowest by pass rate" in flat(CAP53),
   f"D13: {SLA} is #{sorted(_sp, key=lambda b: _sp[b]).index(SLA) + 1} lowest "
   f"by pass rate, not 2nd")
ck(f"({_sp[SLA]} passes out of 270)" in flat(CAP53),
   f"D13: {SLA} pass count != {_sp[SLA]}/270")

# ---- D13b. the four analysis subsections under the 5.3 tables ------------
# The six tables above are generated; the prose that reads them is typed, and
# it is where the section states its conclusions -- "this bucket is the hardest
# in the benchmark", "that lever is null", "category is not a difficulty axis".
# A table can be right while the paragraph summarising it is stale, so every
# claim below is recomputed from DXB and matched against the sentence that
# makes it. Failures print the value the prose should have carried.
#
# The archetype footnote is a blockquote, so flat() collapses its newlines but
# leaves a "> " marker at every wrap inside it. Those are stripped here: a
# claim that spans a line break in the footnote is otherwise unmatchable, and
# "unmatchable" would quietly become "unchecked" the next time it rewraps.
F53 = flat(CAP53).replace(" > ", " ")


def dxsent(marker):
    """The sentence of 5.3 that starts at `marker`, or '' if it is gone.

    Sentences end at '. '; the decimals in these paragraphs never carry a
    following space, so a rate like 3.3% cannot truncate one early.
    """
    return F53.split(marker, 1)[1].split(". ", 1)[0] if marker in F53 else ""


def buckets_in(marker):
    """The backticked names listed in that sentence, in the order given."""
    return re.findall(r"`(\w+)`", dxsent(marker))


def says(tmpl, *counts):
    """Is `tmpl` in 5.3 under any legitimate rendering of its small integers?

    '{}' placeholders are filled with each of "3" / "three" / "Three", so the
    check pins the VALUE without dictating whether the prose spells it out.
    """
    outs = [tmpl]
    for n in counts:
        outs = [o.replace("{}", w, 1) for o in outs for w in numword(n)]
    return any(o in F53 for o in outs)


# -- the two unanimous rosters, over all 27 rendered buckets ----------------
# Control arms are included on purpose here: `no_async` genuinely is one of the
# unanimously easy buckets, and scoping this to DXLEVERS would let a roster
# that quietly dropped it pass.
_neg = sorted(b for b in DXORDER if dxbelow(b) == 9)
_pos = sorted(b for b in DXORDER if dxbelow(b) == 0)
ck(says("{} buckets put **all nine models below their own baseline**",
        len(_neg)),
   f"D13b: {len(_neg)} buckets put all nine models below their own baseline")
ck(says("{} buckets are unanimously *easy*", len(_pos)),
   f"D13b: {len(_pos)} buckets are unanimously easy")
# Membership, not order: the doc lists these in the delta table's row order,
# but the claim is about which buckets qualify, and pinning the order here
# would fail a re-sorted sentence that is still true.
for _mark, _want, _what in (
        ("**all nine models below their own baseline**:", _neg, "unanimously negative"),
        ("0 of 9 below baseline:", _pos, "unanimously easy")):
    _got = buckets_in(_mark)
    ck(_got != [], f"D13b: 5.3 no longer lists the {_what} buckets")
    ck(set(_got) == set(_want),
       f"D13b: the {_what} buckets are {_want}, 5.3 lists {sorted(_got)}")
_negmax = max(DXSEEDS[b] for b in _neg)
ck(says("Every one of them is carried by {} seeds or fewer", _negmax),
   f"D13b: the unanimously negative buckets top out at {_negmax} seeds")

# -- the four bullets -------------------------------------------------------
DHS, CL = "duplicate_hold_stale", "credential_lock"
WC, DO = "workflow_completion", "dependency_ordering"
NOA, MSR = "non_overlapping_assignment", "multi_source_reasoning"
DA, GC, VD = "duplicate_avoidance", "gap_completion", "vision_decoy"
IT, VG = "information_transfer", "vision_graded"

_hardest = min(DXORDER, key=lambda b: (dxpool(b), b))
ck(_hardest == DHS and f"**`{DHS}` is the hardest bucket in the benchmark**" in F53,
   f"D13b: the hardest bucket by pooled pass rate is {_hardest}, not {DHS}")
_dn = sum(DXB[(k, DHS)][0] for k in K)
_dp = sum(DXB[(k, DHS)][1] for k in K)
ck(f"{dxpool(DHS):.1f}% pooled, {_dp} passes in {_dn} attempts" in F53,
   f"D13b: {DHS} is {dxpool(DHS):.1f}% pooled, {_dp} passes in {_dn} attempts")
_zero = sum(1 for k in K if dxpct(k, DHS) == 0.0)
ck(says("with {} models at 0.0%", _zero),
   f"D13b: {_zero} of the nine models score 0.0% on {DHS}")
ck(f"no model above {max(dxpct(k, DHS) for k in K):.1f}%" in F53,
   f"D13b: the best model on {DHS} reaches "
   f"{max(dxpct(k, DHS) for k in K):.1f}%")
_dseeds = sorted(s for s, d in SEEDS.items() if DHS in seed_plugins(d))
ck(DXSEEDS[DHS] == 1 and f"It is one seed (`{_dseeds[0]}`)" in F53,
   f"D13b: {DHS} rests on {DXSEEDS[DHS]} seed(s): {_dseeds}")

_hardmulti = min((b for b in DXORDER if DXSEEDS[b] > 1),
                 key=lambda b: (dxpool(b), b))
ck(_hardmulti == CL
   and f"**`{CL}` is the hardest bucket with more than one seed**" in F53,
   f"D13b: the hardest bucket carried by more than one seed is {_hardmulti}, "
   f"not {CL}")
_clo, _chi = dxrange(CL)
ck(f"({dxpool(CL):.1f}% pooled, {dxsig(_chi)} to {dxsig(_clo)})" in F53,
   f"D13b: {CL} is {dxpool(CL):.1f}% pooled, {_chi:+.1f} to {_clo:+.1f}")
ck(CL not in MECHANIC
   and "It is also prompt-only: it declares no `emits_effects`" in F53,
   f"D13b: {CL} declares emits_effects, so it is not prompt-only")

# The unanimous verdicts among the well-sampled plugins. The threshold is read
# out of the sentence rather than fixed here, so widening it in the prose
# without re-deriving the roster fails.
_wm = re.search(r"plugins carried by (\d+) or more seeds return a unanimous "
                r"verdict", F53)
ck(_wm is not None, "D13b: 5.3 no longer states the well-sampled unanimity claim")
if _wm:
    _wide = sorted((b for b in SPEC_CAT if DXSEEDS[b] >= int(_wm.group(1))),
                   key=lambda b: (-DXSEEDS[b], b))
    _unan = [b for b in _wide if dxbelow(b) in (0, 9)]
    ck(says("**Exactly {} plugins carried by " + _wm.group(1)
            + " or more seeds return a unanimous verdict", len(_unan)),
       f"D13b: {len(_unan)} plugins with {_wm.group(1)}+ seeds are unanimous: "
       f"{_unan}")
    ck(all(dxbelow(b) == 0 for b in _unan)
       and "and both point the wrong way" in F53,
       f"D13b: the well-sampled unanimous plugins are not all unanimously easy: "
       f"{[(b, dxbelow(b)) for b in _unan]}")
    # Order matters in this one: the next clause says "The first is...".
    ck(buckets_in("and both point the wrong way** —") == _unan,
       f"D13b: 5.3 names the well-sampled unanimous plugins "
       f"{buckets_in('and both point the wrong way** —')}, expected {_unan}")
    ck(len(_unan) == 2 and f"{dxbelow(_unan[0])} of 9 below baseline each" in F53,
       f"D13b: the unanimous plugins are {[(b, dxbelow(b)) for b in _unan]}")
ck(f"near-constant ({DXSEEDS[WC]} of {len(SEEDS)} seeds)" in F53,
   f"D13b: {WC} tags {DXSEEDS[WC]} of {len(SEEDS)} seeds (near-constant claim)")
_olo, _ohi = dxrange(DO)
ck(f"`{DO}` covers {DXSEEDS[DO]} seeds and {DXB[(K[0], DO)][0]} tasks per model "
   f"at {dxsig(_olo)} to {dxsig(_ohi)}" in F53,
   f"D13b: {DO} covers {DXSEEDS[DO]} seeds / {DXB[(K[0], DO)][0]} tasks at "
   f"{_olo:+.1f} to {_ohi:+.1f}")

_easiest = max(DXORDER, key=lambda b: (dxpool(b), b))
_nbest = max(K, key=lambda k: dxpct(k, NOA) - dxbase_pct(k))
ck(_easiest == NOA and f"**`{NOA}` is the easiest bucket anywhere**" in F53,
   f"D13b: the easiest bucket by pooled pass rate is {_easiest}, not {NOA}")
ck(f"({dxpool(NOA):.1f}% pooled, up to "
   f"{dxsig(dxpct(_nbest, NOA) - dxbase_pct(_nbest))} for {DISP[_nbest]})" in F53,
   f"D13b: {NOA} is {dxpool(NOA):.1f}% pooled, best "
   f"{dxpct(_nbest, NOA) - dxbase_pct(_nbest):+.1f} ({DISP[_nbest]})")
ck(DXSEEDS[NOA] == 1 and "and again is a single seed" in F53,
   f"D13b: {NOA} rests on {DXSEEDS[NOA]} seeds, not one")

# -- the category paragraph -------------------------------------------------
# Half the category table is an alias for a single plugin. That is asserted as
# NUMERIC identity per model, not as the seed arithmetic behind it, because the
# claim a reader checks is that the two rows carry the same cells.
CATPLUG = collections.defaultdict(set)
for _sd in SEEDS.values():
    for _p in seed_plugins(_sd):
        if _p not in CHALLENGE:
            CATPLUG[SPEC_CAT[_p]].add(_p)
_alias, _agg = {}, {}
for _c, _ps in CATPLUG.items():
    _same = sorted(p for p in _ps
                   if all(DXB[(k, p)][:2] == DXB[(k, _c)][:2] for k in K))
    (_alias if _same else _agg)[_c] = _same or sorted(_ps)
ck(says("{} of its {} buckets are numerically identical to a single plugin",
        len(_alias), len(CATPLUG)),
   f"D13b: {len(_alias)} of {len(CATPLUG)} category buckets duplicate a single "
   f"plugin: {sorted(_alias)}")
for _c, _same in sorted(_alias.items()):
    ck(len(_same) == 1,
       f"D13b: category {_c} is numerically identical to several plugins: {_same}")
    ck(f"`{_c}` to `{_same[0]}`" in F53,
       f"D13b: 5.3 does not report category `{_c}` as identical to `{_same[0]}`")
    ck(DXSEEDS[_same[0]] == DXSEEDS[_c],
       f"D13b: {_same[0]} does not cover every seed of category {_c}")
# The complement, and the property that makes it the complement.
_aorder = sorted(_agg, key=lambda c: (-DXSEEDS[c], c))
ck(says("The other three — ") or len(_aorder) == 3,
   f"D13b: {len(_aorder)} category buckets aggregate more than one plugin")
ck(buckets_in("The other three — ")[:len(_aorder)] == _aorder,
   f"D13b: 5.3 names the aggregating categories "
   f"{buckets_in('The other three — ')}, expected {_aorder} widest-first")
for _c in _aorder:
    ck(max(DXSEEDS[p] for p in _agg[_c]) < DXSEEDS[_c],
       f"D13b: category {_c} pools no more seeds ({DXSEEDS[_c]}) than its "
       f"widest plugin ({max(DXSEEDS[p] for p in _agg[_c])})")
    ck(f"`{_c}` ({DXSEEDS[_c]} seeds)" in F53 or f"`{_c}` ({DXSEEDS[_c]})" in F53,
       f"D13b: 5.3 does not report {DXSEEDS[_c]} seeds for category `{_c}`")
_ab = [dxbelow(c) for c in _aorder]
ck(len(_ab) == 3
   and f"({_ab[0]}, {_ab[1]} and {_ab[2]} of 9 below baseline" in F53,
   f"D13b: the aggregating categories are {list(zip(_aorder, _ab))} below baseline")
_ad = sorted(dxpct(k, c) - dxbase_pct(k) for c in _agg for k in K)
ck(f"deltas inside {dxsig(_ad[0])} to {dxsig(_ad[-1])})" in F53,
   f"D13b: the aggregating categories span {_ad[0]:+.1f} to {_ad[-1]:+.1f}")

# -- depth of enforcement ---------------------------------------------------
ck(says("The {} plugins split into {} that gate real command execution",
        len(SPEC_CAT), len(MECHANIC)),
   f"D13b: {len(MECHANIC)} of {len(SPEC_CAT)} plugins declare emits_effects")
ck(says("and {} that only inject prompt text", len(SPEC_CAT) - len(MECHANIC)),
   f"D13b: {len(SPEC_CAT) - len(MECHANIC)} plugins are prompt-only")
ck(not (CHALLENGE & MECHANIC)
   and says("All {} `challenge` plugins are in the prompt-only group",
            len(CHALLENGE)),
   f"D13b: challenge plugins that also emit effects: {sorted(CHALLENGE & MECHANIC)}")
_depth = collections.defaultdict(list)
for _s, _sd in SEEDS.items():
    _depth[len([p for p in seed_plugins(_sd) if p in MECHANIC])].append(_s)
ck(sorted(_depth) == [0, 1, 2],
   f"D13b: seeds route to {sorted(_depth)} hard mechanics, expected 0 / 1 / 2")


def dxseedstat(names):
    """(pooled pass rate, models below their own baseline) for a set of seeds.

    Off AX["tasks"] rather than off DXB: these groups are not rendered buckets,
    so there is no bucket accumulator to read them out of.
    """
    per = {k: [0, 0] for k in K}
    for t in AX["tasks"]:
        if re.sub(r"_v\d+$", "", t["scen"]) not in names:
            continue
        for k in K:
            per[k][0] += 1
            per[k][1] += 1 if t[NAME[k]][1] else 0
    n = sum(per[k][0] for k in K)
    return (100.0 * sum(per[k][1] for k in K) / n if n else 0.0,
            sum(1 for k in K
                if per[k][0] and 100.0 * per[k][1] / per[k][0] < dxbase_pct(k)))


_dstat = {d: dxseedstat(set(_depth[d])) for d in (0, 1, 2)}
ck(says("gives {} seeds with none, {} with one, and {} with two",
        len(_depth[0]), len(_depth[1]), len(_depth[2])),
   f"D13b: the mechanic-depth groups hold "
   f"{[len(_depth[d]) for d in (0, 1, 2)]} seeds")
ck(f"at {_dstat[0][0]:.1f}%, {_dstat[1][0]:.1f}% and {_dstat[2][0]:.1f}% pooled "
   f"pass rate respectively" in F53,
   f"D13b: the mechanic-depth groups pool to "
   f"{[f'{_dstat[d][0]:.1f}' for d in (0, 1, 2)]}")
ck(f"it is **null**: {_dstat[1][0]:.1f}% against {_dstat[2][0]:.1f}%" in F53,
   f"D13b: the 1-versus-2 mechanic contrast is {_dstat[1][0]:.1f}% against "
   f"{_dstat[2][0]:.1f}%")
ck(_dstat[1][1] == _dstat[2][1]
   and says("with {} of 9 models below their own baseline in each group",
            _dstat[1][1]),
   f"D13b: the 1- and 2-mechanic groups put {_dstat[1][1]} and {_dstat[2][1]} "
   f"models below their own baseline, so 'in each group' is wrong")
ck(f"The 0-mechanic group looks dramatic ({_dstat[0][0]:.1f}%, "
   f"{_dstat[0][1]} of 9 below baseline)" in F53,
   f"D13b: the 0-mechanic group is {_dstat[0][0]:.1f}%, {_dstat[0][1]} of 9 "
   f"below baseline")
# The confound is the point of the paragraph, so it is checked as an identity
# of seed SETS rather than as two counts that happen to match.
_clseeds = sorted(s for s, d in SEEDS.items() if CL in seed_plugins(d))
ck(sorted(_depth[0]) == _clseeds,
   f"D13b: the 0-mechanic seeds {sorted(_depth[0])} are not the `{CL}` seeds "
   f"{_clseeds}, so the two buckets are distinguishable after all")
for _s in _clseeds:
    ck(f"`{_s}`" in F53, f"D13b: 5.3 does not name the confounded seed {_s}")

# -- the levers that do not produce difficulty ------------------------------
_wlo, _whi = dxrange(WC)
ck(f"**`{WC}` ({DXB[(K[0], WC)][0]} tasks) is null and slightly positive** — "
   f"{dxbelow(WC)} of 9 models below baseline, every delta between "
   f"{dxsig(_wlo)} and {dxsig(_whi)}" in F53,
   f"D13b: {WC} is {DXB[(K[0], WC)][0]} tasks, {dxbelow(WC)} of 9 below "
   f"baseline, {_wlo:+.1f} to {_whi:+.1f}")
ck(says("It tags {} of {} seeds", DXSEEDS[WC], len(SEEDS)),
   f"D13b: {WC} tags {DXSEEDS[WC]} of {len(SEEDS)} seeds (null-lever paragraph)")
_mlo, _mhi = dxrange(MSR)
ck(f"**`{MSR}` ({DXB[(K[0], MSR)][0]} tasks) is null.**" in F53,
   f"D13b: {MSR} covers {DXB[(K[0], MSR)][0]} tasks")
ck(says("{} of nine models sit below baseline", dxbelow(MSR)),
   f"D13b: {dxbelow(MSR)} of nine models sit below baseline on {MSR}")
ck(f"the entire spread is {dxsig(_mlo)} to {dxsig(_mhi)} points" in F53,
   f"D13b: {MSR} spreads {_mlo:+.1f} to {_mhi:+.1f} points")
_abest = max(K, key=lambda k: dxpct(k, DA) - dxbase_pct(k))
ck(f"**`{DA}` ({DXB[(K[0], DA)][0]} tasks) is the easiest challenge token**, "
   f"{dxbelow(DA)} of 9 below baseline and up to "
   f"{dxsig(dxpct(_abest, DA) - dxbase_pct(_abest))} points "
   f"({DISP[_abest]} {dxpct(_abest, DA):.1f}%)" in F53,
   f"D13b: {DA} is {DXB[(K[0], DA)][0]} tasks, {dxbelow(DA)} of 9 below "
   f"baseline, best {dxpct(_abest, DA) - dxbase_pct(_abest):+.1f} "
   f"({DISP[_abest]} {dxpct(_abest, DA):.1f}%)")
ck(DA in CHALLENGE, f"D13b: {DA} is not a challenge token")
_glo, _ghi = dxrange(GC)
_vlo, _vhi = dxrange(VD)
ck(dxbelow(GC) == dxbelow(VD) == 9
   and f"`{GC}` ({DXB[(K[0], GC)][0]} tasks) and `{VD}` "
       f"({DXB[(K[0], VD)][0]} tasks) do work, putting {dxbelow(GC)} of 9 "
       f"models below baseline at {dxsig(_ghi)} to {dxsig(_glo)} and "
       f"{dxsig(_vhi)} to {dxsig(_vlo)} points" in F53,
   f"D13b: {GC}/{VD} are {dxbelow(GC)}/{dxbelow(VD)} of 9 below baseline at "
   f"{_ghi:+.1f}..{_glo:+.1f} and {_vhi:+.1f}..{_vlo:+.1f}")
_ilo, _ihi = dxrange(IT)
ck(f"`{IT}` is mildly negative ({dxbelow(IT)} of 9, {dxsig(_ilo)} to "
   f"{dxsig(_ihi)})" in F53,
   f"D13b: {IT} is {dxbelow(IT)} of 9 below baseline at {_ilo:+.1f} to "
   f"{_ihi:+.1f}")
ck(f"`{VG}` likewise ({dxbelow(VG)} of 9, never worse than "
   f"{dxsig(dxrange(VG)[0])})" in F53,
   f"D13b: {VG} is {dxbelow(VG)} of 9 below baseline, worst "
   f"{dxrange(VG)[0]:+.1f}")

# -- statistical power ------------------------------------------------------
ck(says("Every lever with a unanimous negative verdict is carried by {} seeds "
        "or fewer", _negmax),
   f"D13b: the unanimously negative levers top out at {_negmax} seeds")
_pm = re.search(r"no lever carried by (\d+) or more seeds moves any model more "
                r"than (\d+\.\d) pass", F53)
ck(_pm is not None, "D13b: 5.3 no longer bounds the widest levers' movement")
if _pm:
    _widest = [b for b in DXLEVERS if DXSEEDS[b] >= int(_pm.group(1))]
    _mx = max(abs(d) for b in _widest for d in dxrange(b))
    ck(f"{_mx:.1f}" == _pm.group(2),
       f"D13b: the levers carried by {_pm.group(1)}+ seeds move a model up to "
       f"{_mx:.1f} pass points, not {_pm.group(2)}")
# Widest by TASKS, which is what "widest-coverage" means here and is why the
# `no_async` control arm is in the list. Membership only: the doc's ordering of
# the six is not derived from anything.
_six = sorted(DXORDER, key=lambda b: (-DXB[(K[0], b)][0], b))[:6]
_sixnames = re.findall(r"`(\w+)`",
                       F53.split("the six widest-coverage buckets", 1)[1]
                       .split(")", 1)[0]) if \
    "the six widest-coverage buckets" in F53 else []
ck(set(_sixnames) == set(_six),
   f"D13b: the six widest-coverage buckets are {sorted(_six)}, 5.3 lists "
   f"{sorted(_sixnames)}")
_slo = min(dxrange(b)[0] for b in _six)
_shi = max(dxrange(b)[1] for b in _six)
ck(f"together span only {dxsig(_slo)} to {dxsig(_shi)} across all nine models"
   in F53,
   f"D13b: the six widest-coverage buckets span {_slo:+.1f} to {_shi:+.1f}")
ck(DXB[(K[0], CL)][0] == DXB[(K[0], VD)][0],
   f"D13b: {CL} ({DXB[(K[0], CL)][0]} tasks) and {VD} "
   f"({DXB[(K[0], VD)][0]}) no longer cover the same number of tasks")
ck(f"`force_async_wait` covers {DXB[(K[0], 'async_required')][0]} tasks, "
   f"`{DHS}` {DXB[(K[0], DHS)][0]}, `{CL}` and `{VD}` {DXB[(K[0], CL)][0]} "
   f"each, `{GC}` {DXB[(K[0], GC)][0]} — while the two null plugins cover "
   f"{DXB[(K[0], WC)][0]} and {DXB[(K[0], MSR)][0]}" in F53,
   f"D13b: the coverage tally is async_required={DXB[(K[0], 'async_required')][0]}"
   f", {DHS}={DXB[(K[0], DHS)][0]}, {CL}={DXB[(K[0], CL)][0]}, "
   f"{VD}={DXB[(K[0], VD)][0]}, {GC}={DXB[(K[0], GC)][0]}, "
   f"{WC}={DXB[(K[0], WC)][0]}, {MSR}={DXB[(K[0], MSR)][0]}")
_solo = sorted(p for p in SPEC_CAT if DXSEEDS[p] == 1)
ck(says("{} of the {} plugins cannot be distinguished from their single "
        "scenario at all", len(_solo), len(SPEC_CAT)),
   f"D13b: {len(_solo)} of {len(SPEC_CAT)} plugins rest on a single seed")

# The three built-but-unused knobs. Both halves of the claim are checked: the
# seeds never set them, and the cited declarations really are there -- a line
# reference that has drifted is the failure mode a prose check cannot see.
DEAD = ("force_round_type_plan", "force_decoy", "force_no_signpost")
VSPEC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
                     "clasharness", "scenario", "conflict_vlm_spec.py")
_vsrc = (open(VSPEC, encoding="utf-8").read().splitlines()
         if os.path.exists(VSPEC) else [])
ck(_vsrc != [], f"D13b: {VSPEC} is missing")
for _knob in DEAD:
    _on = sorted(s for s, d in SEEDS.items() if d.get(_knob))
    ck(not _on,
       f"D13b: {_knob} is set on {len(_on)} of {len(SEEDS)} seeds ({_on[:3]}), "
       f"so 5.3 is wrong that the corpus never pulls it")
    ck(f"`{_knob}`" in F53, f"D13b: 5.3 no longer names the unused knob {_knob}")
ck(says("**set on 0 of the {} seeds**", len(SEEDS)),
   f"D13b: 5.3 does not state that the unused knobs are set on 0 of "
   f"{len(SEEDS)} seeds")
_cite = re.search(r"\(`conflict_vlm_spec\.py:(\d+)`, `:(\d+)`, `:(\d+)`\)", F53)
ck(_cite is not None, "D13b: 5.3 no longer cites the unused knobs' declarations")
if _cite and _vsrc:
    for _knob, _ln in zip(DEAD, (int(g) for g in _cite.groups())):
        ck(1 <= _ln <= len(_vsrc) and _knob in _vsrc[_ln - 1],
           f"D13b: conflict_vlm_spec.py:{_ln} does not declare {_knob} "
           f"({_vsrc[_ln - 1].strip()[:48]!r})"
           if 1 <= _ln <= len(_vsrc) else
           f"D13b: conflict_vlm_spec.py:{_ln} is past the end of the file")
_km = re.search(r"a (\d+)-line test \(`tests/([\w./]+)`\)", F53)
ck(_km is not None, "D13b: 5.3 no longer cites the knob test")
if _km:
    _kp = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
                       "tests", _km.group(2))
    ck(os.path.exists(_kp), f"D13b: the cited knob test {_kp} does not exist")
    if os.path.exists(_kp):
        _kl = len(open(_kp, encoding="utf-8").read().splitlines())
        ck(_kl == int(_km.group(1)),
           f"D13b: {_km.group(2)} is {_kl} lines, 5.3 says {_km.group(1)}")

# -- the archetype footnote -------------------------------------------------
# The footnote's whole job is to say why the retired grouping was wrong, so
# every one of its contrasts is re-derived against the plugin specs.
ARCH = parse_literal_map(VSPEC, "ARCHETYPE_CATEGORY")
ck(ARCH is not None,
   "D13b: conflict_vlm_spec.py no longer carries an ARCHETYPE_CATEGORY literal")
if ARCH:
    ck(says("a {}-entry hand-written map", len(ARCH)),
       f"D13b: ARCHETYPE_CATEGORY has {len(ARCH)} entries")
    _dis = {a: c for a, c in ARCH.items() if SPEC_CAT.get(a) != c}
    ck(says("on {} of its {} entries", len(_dis), len(ARCH)),
       f"D13b: the archetype map disagrees with the plugin specs on "
       f"{len(_dis)} of {len(ARCH)} entries: {sorted(_dis)}")
    ck("coordination" not in set(ARCH.values())
       and "it has no `coordination` category at all" in F53,
       f"D13b: the archetype map does carry a coordination category: "
       f"{sorted(a for a, c in ARCH.items() if c == 'coordination')}")
    # Scoped to the clause, not the sentence: this footnote runs three
    # contrasts together without a full stop between them, so a sentence-level
    # slice would sweep up the names belonging to the other two.
    _wrongly_inc = sorted(a for a, c in _dis.items()
                          if c == "inconsistency" and a in SPEC_CAT)
    _im = re.search(r"files ((?:`\w+`(?:, | and )?)+) under `inconsistency`", F53)
    ck(_im is not None and re.findall(r"`(\w+)`", _im.group(1)) == _wrongly_inc,
       f"D13b: the archetype map files {_wrongly_inc} under `inconsistency`, "
       f"5.3 names {re.findall(r'`(\w+)`', _im.group(1)) if _im else None}")
    ck(ARCH.get("evidence_mismatch") == "main_supervised"
       and SPEC_CAT.get("evidence_mismatch") == "inconsistency"
       and "files `evidence_mismatch` under `main_supervised` where the plugin "
           "says `inconsistency`" in F53,
       f"D13b: evidence_mismatch is archetype="
       f"{ARCH.get('evidence_mismatch')} / plugin="
       f"{SPEC_CAT.get('evidence_mismatch')}")
    _notplug = sorted(a for a in ARCH if a not in SPEC_CAT)
    ck(len(_notplug) == 1 and f"(`{_notplug[0]}`) that is not a plugin" in F53,
       f"D13b: the archetype entries that are not plugins are {_notplug}")
_noarch = sorted(s for s, d in SEEDS.items() if not d.get("archetype"))
ck(says('an `other` bucket of "the {} seeds with no archetype"', len(_noarch)),
   f"D13b: {len(_noarch)} seeds carry no archetype")
_minp = min((len(seed_plugins(SEEDS[s])) for s in _noarch), default=0)
ck(says("those {} seeds each activate {} or more conflict plugins",
        len(_noarch), _minp),
   f"D13b: the archetype-less seeds activate as few as {_minp} plugins")

# ---- D14. the three derivations of the designed axes must agree -----------
# design_axes.json (tools/design_axis_slice.py, off the raw jsonl), the
# design_axes block of the stats dump (build_behavior_report's own task table)
# and this pass (difficulty_axes.json) compute the same buckets from three
# sources with no shared code. Agreement is therefore evidence; agreement
# between two of them that share an ancestor would not be.
#
# Integers only. The three carry mean scores at different precisions on
# purpose -- build_behavior_report rounds each task score to 2dp before it
# aggregates -- so their averages differ in the third decimal by construction
# and comparing them would fail for a reason that is not a bug.
DXJ = os.path.join(HERE, "design_axes.json")
ck(os.path.exists(DXJ), "D14: design_axes.json is missing")
if os.path.exists(DXJ):
    _j = json.load(open(DXJ, encoding="utf-8"))["by_model"]
    _s = S.get("design_axes", {}).get("by_model", {})
    ck(_s != {}, "D14: the stats dump has no design_axes block")
    ck(sorted(_j) == sorted(K), f"D14: design_axes.json covers {sorted(_j)}")
    for k in K:
        for _src, _lbl in ((_j.get(k), "design_axes.json"),
                           (_s.get(k), "stats.design_axes")):
            ck(_src is not None, f"D14 {_lbl}: no entry for {NAME[k]}")
            if _src is None:
                continue
            ck(not _src.get("unmapped"),
               f"D14 {_lbl}/{NAME[k]}: unmapped scenarios {_src.get('unmapped')}")
            ck((_src["baseline"]["n"], _src["baseline"]["passed"])
               == (DXBASE[k][0], DXBASE[k][1]),
               f"D14 {_lbl}/{NAME[k]}: baseline {_src['baseline']} != "
               f"{DXBASE[k][0]}/{DXBASE[k][1]}")
            _seen = set()
            for _axis, _bks in _src["axes"].items():
                for _b, _c in _bks.items():
                    _seen.add(_b)
                    ck((_c["n"], _c["passed"]) == (DXB[(k, _b)][0],
                                                   DXB[(k, _b)][1]),
                       f"D14 {_lbl}/{NAME[k]}/{_b}: {_c['n']}/{_c['passed']} "
                       f"!= {DXB[(k, _b)][0]}/{DXB[(k, _b)][1]}")
            ck(_seen == set(dx_rows),
               f"D14 {_lbl}/{NAME[k]}: buckets differ from the 5.3 tables: "
               f"{sorted(_seen ^ set(dx_rows))}")

# ==========================================================================
# PASS E -- the nine <model>-solver-failures.md deep-dives, recomputed
#
# Those tables are emitted by tools/emit_solver_failures.py. Running that tool
# with --check only proves the files match what the tool *would* write today,
# so a bug inside the tool round-trips cleanly and stays invisible. This pass
# therefore re-derives every emitted cell from failure_taxonomy.json and the
# stats dump using its own arithmetic, parses the observed value back out of
# the rendered markdown, and adds cross-document invariants (a "rank of 9"
# column has to be a consistent ranking of the nine documents' values) that no
# single-document check can see.
# ==========================================================================

SLUG = {"flash": "gemini-3-5-flash", "flash36": "gemini-3-6-flash",
        "pro": "gemini-3-1-pro", "opus": "claude-opus-4-8",
        "opus5": "claude-opus-5", "sonnet": "claude-sonnet-5",
        "sol": "gpt-5-6-sol", "luna": "gpt-5-6-luna", "terra": "gpt-5-6-terra"}
PROSE_SLOTS = ["signature", "termination", "buckets", "effort", "actions",
               "waits", "clustering", "stops", "scenarios", "design_axes",
               "interpretation"]
SF_BLOCKS = [f"SF{i}" for i in range(1, 12)]
EFF3 = ("low", "medium", "high")
PROVIDER = {"flash": "gemini", "flash36": "gemini", "pro": "gemini",
            "opus": "vertex-anthropic", "opus5": "vertex-anthropic",
            "sonnet": "vertex-anthropic", "sol": "bedrock-gpt",
            "luna": "bedrock-gpt", "terra": "bedrock-gpt"}


def med(vals):
    """Median, computed here rather than imported, so this pass does not share
    an implementation with the emitter it is auditing."""
    v = sorted(vals)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0


def dpeq(cell, val, places):
    """Cell equals `val` rendered at `places` decimals -- exact, not 'near'.

    A tolerance window wide enough to absorb rounding is also wide enough to
    absorb a typo in the last printed digit, which is the drift this exists to
    catch: at tol=0.006 a true -0.7143 accepts both '-0.71' and '-0.72'.
    """
    # thousands separators first: as_f stops at the comma, so '2,321' would
    # otherwise parse as 2.0 and every large count would compare as wrong
    got = as_f(cell.replace(",", "")) if isinstance(cell, str) else cell
    return got is not None and got == float(f"{val:.{places}f}")


def as_int(s):
    """First integer in a cell, tolerating thousands separators."""
    m = re.search(r"-?\d[\d,]*", str(s).replace("\u2212", "-"))
    return int(m.group(0).replace(",", "")) if m else None


def rank9(key, vals, best="min"):
    order = sorted(vals.values(), reverse=(best == "max"))
    return order.index(vals[key]) + 1


RANKCOL = collections.defaultdict(dict)   # metric -> {model: (rank, value)}


def rankcell(metric, k, cell, vals, best):
    """Check one 'rank of 9' cell and remember it for the cross-doc invariant."""
    want = rank9(k, vals, best)
    got = as_int(cell)
    ck(got == want, f"E-rank {metric}/{SLUG[k]}: rank {cell!r} != {want}")
    RANKCOL[metric][k] = (got, vals[k])


def medcell(metric, k, cell, vals, places):
    ck(dpeq(cell, med(list(vals.values())), places),
       f"E-med {metric}/{SLUG[k]}: field median {cell!r} != "
       f"{med(list(vals.values())):.{places}f}")


DD = {}
for k in K:
    _p = os.path.join(HERE, f"{SLUG[k]}-solver-failures.md")
    ck(os.path.exists(_p), f"E0: deep-dive {SLUG[k]}-solver-failures.md missing")
    DD[k] = open(_p, encoding="utf-8").read() if os.path.exists(_p) else ""


def ddblk(k, tag):
    b, e = f"<!-- {tag}-{k}-BEGIN -->", f"<!-- {tag}-{k}-END -->"
    t = DD[k]
    if b not in t or e not in t:
        ck(False, f"E0 {SLUG[k]}: sentinel pair {tag}-{k} missing")
        return ""
    return t.split(b, 1)[1].split(e, 1)[0]


def ddrow(k, tag, label):
    return cells_of(ddblk(k, tag), label)


# ---- E0. structure, provenance and the unwritten-prose gate --------------
for k in K:
    doc, a = DD[k], FTM[k]
    ck(doc.startswith(f"# {a['display']} \u2014 solver failure deep-dive"),
       f"E0 {SLUG[k]}: title does not name {a['display']}")
    ck(f"`{PROVIDER[k]}`" in doc,
       f"E0 {SLUG[k]}: provenance does not name provider {PROVIDER[k]}")
    ck("<!-- PROSE: " not in doc,
       f"E0 {SLUG[k]}: old-style single-marker prose placeholder survives")
    # a sentinel carrying another model's key means a copy-paste across files
    foreign = sorted({m for m in re.findall(
        r"<!-- (?:SF\d+|PROSE-[a-z]+)-(\w+)-BEGIN -->", doc) if m != k})
    ck(not foreign, f"E0 {SLUG[k]}: sentinels for other models present: {foreign}")
    empty = []
    for slot in PROSE_SLOTS:
        b, e = f"<!-- PROSE-{slot}-{k}-BEGIN -->", f"<!-- PROSE-{slot}-{k}-END -->"
        if b not in doc or e not in doc:
            ck(False, f"E0 {SLUG[k]}: prose sentinel {slot} missing")
            continue
        if not doc.split(b, 1)[1].split(e, 1)[0].strip():
            empty.append(slot)
    ck(not empty,
       f"E0 {SLUG[k]}: {len(empty)} prose slot(s) still unwritten: "
       f"{', '.join(empty)}")
    # three sources must agree on this model's headline before any table is read
    ck(a["pass_pct"] == round(100.0 * a["passed"] / 600, 1),
       f"E0 {SLUG[k]}: taxonomy pass_pct {a['pass_pct']} != passed/600")
    ck(S["effort"]["cells"][f"{k}@medium"]["passed"] == a["passed"],
       f"E0 {SLUG[k]}: stats and taxonomy disagree on medium passes")

# ---- E1. capability snapshot --------------------------------------------
for k in K:
    a, sec = FTM[k], ddblk(k, "SF1")
    pas = {m: FTM[m]["pass_pct"] for m in K}
    r = cells_of(sec, "pass@budget")
    ck(r is not None, f"E1 {SLUG[k]}: no pass@budget row")
    if r:
        ck(dpeq(r[0], a["pass_pct"], 1),
           f"E1 {SLUG[k]}: pass_pct {r[0]!r} != {a['pass_pct']}")
        m = re.search(r"\((\d+)/600\)", r[0])
        ck(m and int(m.group(1)) == a["passed"],
           f"E1 {SLUG[k]}: passed in {r[0]!r} != {a['passed']}")
        rankcell("pass_pct", k, r[1], pas, "max")
        medcell("pass_pct", k, r[2], pas, 1)
    for label, field, src, places, best in (
            ("avg_score", "avg_score", "ft", 4, "max"),
            ("per-check accuracy", "per_check_pct", "cl", 1, "max"),
            ("failed required+diagnostic units", "failed_total", "ft", 0, "min")):
        vals = {m: (FTM[m][field] if src == "ft"
                    else S[m]["clustering"][field]) for m in K}
        r = cells_of(sec, label)
        ck(r is not None, f"E1 {SLUG[k]}: no {label} row")
        if not r:
            continue
        ck(dpeq(r[0], vals[k], places),
           f"E1 {SLUG[k]}: {label} {r[0]!r} != {vals[k]}")
        rankcell(label, k, r[1], vals, best)
        medcell(label, k, r[2], vals, places)

# ---- E2. termination profile --------------------------------------------
E2FIELDS = [("near-miss tasks", "near_miss", "max"),
            ("zero-score tasks", "zero_score", "min"),
            ("step-capped tasks", "capped", "min"),
            ("of those, single-verb loops", "loopy_capped", "min"),
            ("tasks with zero asset opens", "zero_open_tasks", "min"),
            ("tasks with zero writes", "zero_write_tasks", "min"),
            ("median steps", "step_p50", "min"),
            ("mean steps", "step_mean", "min")]
for k in K:
    sec = ddblk(k, "SF2")
    for label, field, best in E2FIELDS:
        vals = {m: FTM[m][field] for m in K}
        places = 2 if isinstance(vals[k], float) else 0
        r = cells_of(sec, label)
        ck(r is not None, f"E2 {SLUG[k]}: no {label} row")
        if not r:
            continue
        ck(dpeq(r[0], vals[k], places),
           f"E2 {SLUG[k]}: {label} {r[0]!r} != {vals[k]}")
        rankcell(f"term:{field}", k, r[1], vals, best)
    sem = FTM[k]["semantic"]
    r = cells_of(sec, "fail-closed semantic checks")
    ck(r is not None, f"E2 {SLUG[k]}: no fail-closed row")
    if r:
        m = re.search(r"^(\d+) of ([\d,]+) \(([\d.]+)%\)$", r[0])
        ck(m is not None, f"E2 {SLUG[k]}: fail-closed cell unparsable: {r[0]!r}")
        if m:
            ck(int(m.group(1)) == sem["fail_closed"],
               f"E2 {SLUG[k]}: fail_closed {m.group(1)} != {sem['fail_closed']}")
            ck(int(m.group(2).replace(",", "")) == sem["n"],
               f"E2 {SLUG[k]}: semantic n {m.group(2)} != {sem['n']}")
            ck(float(m.group(3)) == round(sem["fail_closed_pct"], 1),
               f"E2 {SLUG[k]}: fail_closed_pct {m.group(3)} != "
               f"{sem['fail_closed_pct']}")
            ck(near(100.0 * sem["fail_closed"] / sem["n"],
                    sem["fail_closed_pct"], 0.051),
               f"E2 {SLUG[k]}: fail_closed_pct disagrees with its own ratio")

# ---- E3. the F1-F6 bucket table -----------------------------------------
BWHAT = {"content_msg": "message content", "content_file": "file content",
         "grounding": "asset grounding", "anchor": "literal anchor",
         "side_effect": "terminal action", "window": "time window"}
for k in K:
    sec, a = ddblk(k, "SF3"), FTM[k]
    treq = tdiag = tunits = 0
    for fid, b in FBUCKET:
        rates = {m: req_rate(m, b) for m in K}
        fc, tot = a["failed_checks"][b], a["total_checks"][b]["required"]
        treq += fc["required"]
        tdiag += fc["diagnostic"]
        tunits += tot
        r = cells_of(sec, f"{fid} {b}")
        ck(r is not None, f"E3 {SLUG[k]}: no {fid} row")
        if not r:
            continue
        ck(r[0] == BWHAT[b], f"E3 {SLUG[k]}/{fid}: description {r[0]!r} drifted")
        ck(r[1] == f"{fc['required']}/{fc['diagnostic']}",
           f"E3 {SLUG[k]}/{fid}: failed {r[1]!r} != "
           f"{fc['required']}/{fc['diagnostic']}")
        ck(as_int(r[2]) == tot,
           f"E3 {SLUG[k]}/{fid}: required units {r[2]!r} != {tot}")
        ck(dpeq(r[3], rates[k], 1),
           f"E3 {SLUG[k]}/{fid}: rate {r[3]!r} != {rates[k]:.1f}%")
        rankcell(f"bucket:{fid}", k, r[4], rates, "min")
        medcell(f"bucket:{fid}", k, r[5], rates, 1)
    r = cells_of(sec, "total")
    ck(r is not None, f"E3 {SLUG[k]}: no total row")
    if r:
        ck(r[1] == f"{treq:,}/{tdiag}",
           f"E3 {SLUG[k]}: total failed {r[1]!r} != {treq:,}/{tdiag}")
        ck(as_int(r[2]) == tunits,
           f"E3 {SLUG[k]}: total required {r[2]!r} != {tunits}")
    # the six buckets must exhaust the model's required units
    ck(tunits == sum(v["required"] for v in a["total_checks"].values()),
       f"E3 {SLUG[k]}: F1-F6 do not cover every required check")
    ck(treq == sum(v["required"] for v in a["failed_checks"].values()),
       f"E3 {SLUG[k]}: F1-F6 do not cover every required failure")

# ---- E4. the effort sweep ------------------------------------------------
for k in K:
    sec = ddblk(k, "SF4")
    for eff in EFF3:
        c, fm = S["effort"]["cells"][f"{k}@{eff}"], FT["cells"][eff]["models"][k]
        r = cells_of(sec, eff)
        ck(r is not None, f"E4 {SLUG[k]}/{eff}: no row")
        if not r:
            continue
        ck(dpeq(r[0], c["pass_pct"], 1),
           f"E4 {SLUG[k]}/{eff}: pass_pct {r[0]!r} != {c['pass_pct']}")
        m = re.search(r"\((\d+)/600\)", r[0])
        ck(m and int(m.group(1)) == c["passed"],
           f"E4 {SLUG[k]}/{eff}: passed in {r[0]!r} != {c['passed']}")
        ck(dpeq(r[1], c["avg_score"], 4),
           f"E4 {SLUG[k]}/{eff}: avg_score {r[1]!r} != {c['avg_score']}")
        ck(as_int(r[2]) == c["capped"],
           f"E4 {SLUG[k]}/{eff}: capped {r[2]!r} != {c['capped']}")
        ck(dpeq(r[3], c["mean_steps"], 2),
           f"E4 {SLUG[k]}/{eff}: mean_steps {r[3]!r} != {c['mean_steps']}")
        for i, (fid, b) in enumerate(FBUCKET):
            want = fm["failed_checks"][b]["required"]
            ck(as_int(r[4 + i]) == want,
               f"E4 {SLUG[k]}/{eff}/{fid}: failed {r[4 + i]!r} != {want}")
        # the two dumps must agree cell by cell, not just at medium
        ck(c["passed"] == fm["passed"],
           f"E4 {SLUG[k]}/{eff}: stats passed != taxonomy passed")
    mc = S["effort"]["mcnemar"][k]
    m = re.search(r"low \u2192 high McNemar: \*\*([+-]?[\d.]+) pp\*\* "
                  r"\((\d+) gained, (\d+) lost, p = ([\d.eE+-]+)\)",
                  ddblk(k, "SF4").replace("\u2212", "-"))
    ck(m is not None, f"E4 {SLUG[k]}: McNemar sentence not found")
    if m:
        ck(float(m.group(1)) == float(f"{mc['delta_pp']:.1f}"),
           f"E4 {SLUG[k]}: McNemar delta {m.group(1)} != {mc['delta_pp']:.1f}")
        ck(int(m.group(2)) == mc["gained"] and int(m.group(3)) == mc["lost"],
           f"E4 {SLUG[k]}: McNemar gained/lost drifted")
        ck(m.group(4) == f"{mc['p']:.2g}",
           f"E4 {SLUG[k]}: McNemar p {m.group(4)!r} != {mc['p']:.2g}")
        # delta_pp must be the gained/lost difference over 600 tasks
        ck(near(mc["delta_pp"], 100.0 * (mc["gained"] - mc["lost"]) / 600, 0.051),
           f"E4 {SLUG[k]}: McNemar delta_pp disagrees with gained-lost")

# ---- E5. action mix ------------------------------------------------------
for k in K:
    sec, mix = ddblk(k, "SF5"), S[k]["catmix"]
    total, n = S[k]["catmix_total"], FTM[k]["rows"]
    ck(sum(mix.values()) == total,
       f"E5 {SLUG[k]}: catmix does not sum to catmix_total")
    seen, prev = [], None
    for line in sec.splitlines():
        if not line.strip().startswith("| `"):
            continue
        parts = [_plain(c) for c in line.strip().strip("|").split("|")]
        cat = parts[0]
        seen.append(cat)
        ck(cat in mix, f"E5 {SLUG[k]}: unknown action category {cat!r}")
        if cat not in mix:
            continue
        v = mix[cat]
        ck(as_int(parts[1]) == v, f"E5 {SLUG[k]}/{cat}: count {parts[1]!r} != {v}")
        ck(dpeq(parts[2], v / n, 2),
           f"E5 {SLUG[k]}/{cat}: per task {parts[2]!r} != {v / n:.2f}")
        fieldv = med([S[m]["catmix"].get(cat, 0) / FTM[m]["rows"] for m in K])
        ck(dpeq(parts[3], fieldv, 2),
           f"E5 {SLUG[k]}/{cat}: field median {parts[3]!r} != {fieldv:.2f}")
        ck(dpeq(parts[4], 100.0 * v / total, 1),
           f"E5 {SLUG[k]}/{cat}: share {parts[4]!r} != {100.0 * v / total:.1f}%")
        ck(prev is None or v <= prev,
           f"E5 {SLUG[k]}: rows are not sorted by count at {cat!r}")
        prev = v
    ck(sorted(seen) == sorted(mix), f"E5 {SLUG[k]}: row set != catmix keys")
    r = cells_of(sec, "total")
    ck(r is not None and as_int(r[0]) == total,
       f"E5 {SLUG[k]}: total row {r[0] if r else None!r} != {total}")

# ---- E6. waiting ---------------------------------------------------------
E6FIELDS = [("wait actions", "n", "min"),
            ("waits shorter than 30 s", "sub_30", "min"),
            ("mean wait (s)", "mean", "max"),
            ("median wait (s)", "median", "max"),
            ("longest wait (s)", "max", "max"),
            ("total simulated wait (s)", "total", "min")]
for k in K:
    sec, w = ddblk(k, "SF6"), FTM[k]["waits"]
    for label, field, best in E6FIELDS:
        vals = {m: FTM[m]["waits"][field] for m in K}
        r = cells_of(sec, label)
        ck(r is not None, f"E6 {SLUG[k]}: no {label} row")
        if not r:
            continue
        ck(as_f(r[0]) == float(f"{vals[k]:g}"),
           f"E6 {SLUG[k]}: {label} {r[0]!r} != {vals[k]:g}")
        rankcell(f"wait:{field}", k, r[1], vals, best)
    # SF6 is the only table whose rank column reverses direction between rows,
    # so it carries a note saying which way each row goes. Rebuild that
    # sentence from E6FIELDS -- the same list this pass ranks against, not the
    # emitter's -- so a flipped direction cannot stay consistent with itself.
    def _oxford(xs):
        return " and ".join(xs) if len(xs) < 3 else \
            ", ".join(xs[:-1]) + ", and " + xs[-1]
    want_note = ("Rank 1 is the lowest value for "
                 + _oxford([l for l, _, b in E6FIELDS if b == "min"])
                 + ", and the highest value for "
                 + _oxford([l for l, _, b in E6FIELDS if b == "max"]) + ".")
    ck(flat(sec).count(want_note) == 1,
       f"E6 {SLUG[k]}: rank-direction note is missing or not exactly "
       f"{want_note!r}")
    shown = sum(w["hist"].values())
    m = re.search(r"Most-used durations \(top (\d+) of the histogram, "
                  r"(\d+) of (\d+) waits\): (.*)\.$", flat(sec).strip())
    ck(m is not None, f"E6 {SLUG[k]}: histogram sentence not found")
    if m:
        ck(int(m.group(1)) == len(w["hist"]),
           f"E6 {SLUG[k]}: histogram claims {m.group(1)} entries, "
           f"has {len(w['hist'])}")
        ck(int(m.group(2)) == shown,
           f"E6 {SLUG[k]}: histogram claims {m.group(2)} waits, sums to {shown}")
        ck(int(m.group(3)) == w["n"],
           f"E6 {SLUG[k]}: histogram total {m.group(3)} != {w['n']}")
        # the dump keeps only most_common(8); the sentence must not imply more
        ck(shown <= w["n"],
           f"E6 {SLUG[k]}: histogram sums above the wait count")
        got = dict(re.findall(r"`([^`]+)` x(\d+)", m.group(4)))
        ck({a: int(b) for a, b in got.items()} == {a: int(b) for a, b
                                                   in w["hist"].items()},
           f"E6 {SLUG[k]}: histogram entries differ from the dump")

# ---- E7. failure clustering ---------------------------------------------
E7FIELDS = [("per-check pass rate", "per_check_pct", "max"),
            ("observed task pass rate", "obs_pass_pct", "max"),
            ("pass rate if checks were independent", "indep_pass_pct", "max"),
            ("excess (observed \u2212 independent)", "excess_pp", "max"),
            ("lift", "lift", "max"),
            ("overdispersion", "overdispersion", "max")]
for k in K:
    sec, c = ddblk(k, "SF7"), S[k]["clustering"]
    for label, field, best in E7FIELDS:
        vals = {m: S[m]["clustering"][field] for m in K}
        r = cells_of(sec, _plain(label))
        ck(r is not None, f"E7 {SLUG[k]}: no {label!r} row")
        if not r:
            continue
        ck(as_f(r[0]) == float(vals[k]),
           f"E7 {SLUG[k]}: {label} {r[0]!r} != {vals[k]}")
        rankcell(f"clust:{field}", k, r[1], vals, best)
    lo, hi = c["lift_ci"]
    r = cells_of(sec, "lift 95% CI")
    ck(r is not None and r[0] == f"[{lo}, {hi}]",
       f"E7 {SLUG[k]}: lift CI {r[0] if r else None!r} != [{lo}, {hi}]")
    ck(lo <= c["lift"] <= hi, f"E7 {SLUG[k]}: lift lies outside its own CI")
    r = cells_of(sec, "checks / tasks scored")
    ck(r is not None and r[0] == f"{c['n_checks']:,} / {c['n_tasks']}",
       f"E7 {SLUG[k]}: checks/tasks {r[0] if r else None!r} != "
       f"{c['n_checks']:,} / {c['n_tasks']}")
    # The excess must be the difference the row names. The dump rounds all
    # three to one decimal independently, so a difference of rounded values can
    # legitimately sit 0.1 away from the rounded difference; anything larger is
    # a wrong operand, not rounding.
    ck(near(c["excess_pp"], c["obs_pass_pct"] - c["indep_pass_pct"], 0.1001),
       f"E7 {SLUG[k]}: excess_pp {c['excess_pp']} is not observed "
       f"{c['obs_pass_pct']} minus independent {c['indep_pass_pct']}")

# ---- E8. how runs end ----------------------------------------------------
ALLSTOPS = sorted({s for m in K for s in FTM[m]["stops"]})
for k in K:
    sec, stops, n = ddblk(k, "SF8"), FTM[k]["stops"], FTM[k]["rows"]
    rows = [_plain(l.strip().strip("|").split("|")[0])
            for l in sec.splitlines() if l.strip().startswith("| `")]
    ck(sorted(rows) == ALLSTOPS,
       f"E8 {SLUG[k]}: stop-reason rows {sorted(rows)} != union {ALLSTOPS}")
    prev = None
    for s in rows:
        r = cells_of(sec, s)
        v = stops.get(s, 0)
        ck(r is not None, f"E8 {SLUG[k]}: stop row {s!r} unreadable")
        if r is None:
            continue
        ck(as_int(r[0]) == v, f"E8 {SLUG[k]}/{s}: tasks {r[0]!r} != {v}")
        ck(dpeq(r[1], 100.0 * v / n, 1),
           f"E8 {SLUG[k]}/{s}: share {r[1]!r} != {100.0 * v / n:.1f}%")
        fm = med([FTM[m]["stops"].get(s, 0) for m in K])
        ck(as_f(r[2]) == float(f"{fm:g}"),
           f"E8 {SLUG[k]}/{s}: field median {r[2]!r} != {fm:g}")
        ck(prev is None or v <= prev,
           f"E8 {SLUG[k]}: stop rows are not sorted by count at {s!r}")
        prev = v
    ck(sum(stops.values()) == n,
       f"E8 {SLUG[k]}: stop reasons sum to {sum(stops.values())}, not {n}")
    g = FTM[k]["gating"]
    m = re.search(r"Of (\d+) failed tasks, \*\*(\d+)\*\* involve a content "
                  r"check \(F1/F2\) and \*\*(\d+)\*\* fail on grounding alone; "
                  r"(\d+) fail with no required check failing at all", flat(sec))
    ck(m is not None, f"E8 {SLUG[k]}: gating sentence not found")
    if m:
        ck(int(m.group(1)) == g["n_fail"],
           f"E8 {SLUG[k]}: gating n_fail {m.group(1)} != {g['n_fail']}")
        ck(int(m.group(2)) == g["content_involved"],
           f"E8 {SLUG[k]}: content_involved {m.group(2)} != "
           f"{g['content_involved']}")
        ck(int(m.group(3)) == g["grounding_only"],
           f"E8 {SLUG[k]}: grounding_only {m.group(3)} != {g['grounding_only']}")
        ck(int(m.group(4)) == g["no_required_check_failed"],
           f"E8 {SLUG[k]}: no_required_check_failed {m.group(4)} != "
           f"{g['no_required_check_failed']}")
        ck(g["n_fail"] == n - FTM[k]["passed"],
           f"E8 {SLUG[k]}: gating n_fail disagrees with 600 - passed")

# ---- E13. unscored tasks are disclosed wherever they distort a table -----
# An episode that yields no check results is graded fail-closed: it sits in the
# 600-task denominator but scores nothing. Two consequences are visible in the
# documents and look like errors if left unexplained -- the `none` stop row
# exceeds the pass count, and the gating sentence reports failures with no
# required check failing. Both equal the unscored count exactly, so the
# invariant is asserted against the data first and the prose disclosure is then
# required to be present exactly when it bites.
for eff in EFF3:
    for k in K:
        a = FT["cells"][eff]["models"][k]
        miss = a["missing_check_results"]
        ck(a["stops"].get("none", 0) - a["passed"] == miss,
           f"E13 {SLUG[k]}/{eff}: none {a['stops'].get('none', 0)} - passed "
           f"{a['passed']} != {miss} unscored")
        ck(a["gating"]["no_required_check_failed"] == miss,
           f"E13 {SLUG[k]}/{eff}: gating residual "
           f"{a['gating']['no_required_check_failed']} != {miss} unscored")
        ck(len(a["missing_check_result_ids"]) == miss,
           f"E13 {SLUG[k]}/{eff}: {len(a['missing_check_result_ids'])} ids "
           f"listed for {miss} unscored tasks")

for k in K:
    miss = FTM[k]["missing_check_results"]
    sec = flat(ddblk(k, "SF8"))
    m = re.search(r"That residual is (\d+) tasks? whose episode produced no "
                  r"check results at all \(([^)]*)\)\..*?neither passes nor "
                  r"enters the ([\d,]+) tasks scored", sec)
    ck((m is not None) == bool(miss),
       f"E13 {SLUG[k]}: SF8 discloses unscored tasks "
       f"{'but there are none' if m else 'nowhere, but there are'} {miss}")
    if m and miss:
        ck(int(m.group(1)) == miss,
           f"E13 {SLUG[k]}: SF8 discloses {m.group(1)} unscored, not {miss}")
        ck(re.findall(r"`([^`]+)`", m.group(2))
           == FTM[k]["missing_check_result_ids"],
           f"E13 {SLUG[k]}: SF8 unscored task ids differ from the dump")
        ck(as_int(m.group(3)) == S[k]["clustering"]["n_tasks"],
           f"E13 {SLUG[k]}: SF8 says {m.group(3)} scored, section 3.3 says "
           f"{S[k]['clustering']['n_tasks']}")
    permiss = {e: FT["cells"][e]["models"][k]["missing_check_results"]
               for e in EFF3}
    m4 = re.search(r"counted as failures in the denominator above without "
                   r"being scored: ([^.]+)\.", flat(ddblk(k, "SF4")))
    ck((m4 is not None) == any(permiss.values()),
       f"E13 {SLUG[k]}: SF4 unscored note present={m4 is not None}, "
       f"needed={any(permiss.values())}")
    if m4:
        ck(m4.group(1) == ", ".join(f"{v} at {e}" for e, v in permiss.items()
                                    if v),
           f"E13 {SLUG[k]}: SF4 unscored note {m4.group(1)!r} disagrees with "
           f"{permiss}")

# ---- E9. scenario extremes ----------------------------------------------
for k in K:
    sec, a = ddblk(k, "SF9"), FTM[k]
    slugs = sorted(a["scenarios"])
    ck(len(slugs) == 60, f"E9 {SLUG[k]}: {len(slugs)} scenarios, expected 60")
    field = {s: sum(FTM[m]["scenarios"][s]["mean_score"] for m in K) / len(K)
             for s in slugs}
    gap = {s: a["scenarios"][s]["mean_score"] - field[s] for s in slugs}
    parts = sec.split("**Largest negative gaps against the field**")
    ck(len(parts) == 2, f"E9 {SLUG[k]}: cannot split the two gap tables")
    if len(parts) != 2:
        continue
    tail = parts[1].split("**Zero-pass scenarios")
    for tbl, want, tag in ((parts[0], sorted(slugs, key=lambda s: -gap[s])[:5],
                            "positive"),
                           (tail[0], sorted(slugs, key=lambda s: gap[s])[:5],
                            "negative")):
        got = [_plain(l.strip().strip("|").split("|")[0])
               for l in tbl.splitlines() if l.strip().startswith("| `")]
        ck(got == want, f"E9 {SLUG[k]}: {tag} gap table is {got}, expected {want}")
        for s in got:
            if s not in gap:
                continue
            r = cells_of(tbl, s)
            ck(r is not None, f"E9 {SLUG[k]}: {tag} row {s!r} unreadable")
            if r is None:
                continue
            ck(dpeq(r[0], a["scenarios"][s]["mean_score"], 3),
               f"E9 {SLUG[k]}/{s}: mean {r[0]!r} != "
               f"{a['scenarios'][s]['mean_score']:.3f}")
            ck(dpeq(r[1], field[s], 3),
               f"E9 {SLUG[k]}/{s}: field mean {r[1]!r} != {field[s]:.3f}")
            ck(dpeq(r[2], gap[s], 3),
               f"E9 {SLUG[k]}/{s}: gap {r[2]!r} != {gap[s]:+.3f}")
            ck(r[3] == f"{a['scenarios'][s]['passed']}/10",
               f"E9 {SLUG[k]}/{s}: passes {r[3]!r} != "
               f"{a['scenarios'][s]['passed']}/10")
    zero = sorted((s for s in slugs if a["scenarios"][s]["passed"] == 0),
                  key=lambda s: field[s])
    m = re.search(r"\*\*Zero-pass scenarios: (\d+) of 60\.\*\* (.*)\.",
                  flat(parts[1]).strip())
    ck(m is not None, f"E9 {SLUG[k]}: zero-pass sentence not found")
    if m:
        ck(int(m.group(1)) == len(zero),
           f"E9 {SLUG[k]}: zero-pass count {m.group(1)} != {len(zero)}")
        got = re.findall(r"`([^`]+)`", m.group(2))
        ck(got == zero, f"E9 {SLUG[k]}: zero-pass list differs from the dump")

# ---- E10. verbatim failure evidence -------------------------------------
for k in K:
    sec, ex = ddblk(k, "SF10"), FTM[k]["examples"]
    ck("not selected for effect" in flat(sec),
       f"E10 {SLUG[k]}: the selection rule is not stated")
    for fid, b in FBUCKET:
        items = ex.get(b, [])
        want = next((e for e in items if e["role"] == "required"),
                    items[0] if items else None)
        m = re.search(rf"\*\*{fid}\*\* `{b}` \u2014 `([^`]+)` \((\w+)\):\s*> (.+)",
                      sec)
        ck(m is not None, f"E10 {SLUG[k]}: no {fid} bullet")
        if not m or want is None:
            continue
        ck(m.group(1) == want["task_id"],
           f"E10 {SLUG[k]}/{fid}: task {m.group(1)!r} != {want['task_id']!r}")
        ck(m.group(2) == want["role"],
           f"E10 {SLUG[k]}/{fid}: role {m.group(2)!r} != {want['role']!r}")
        ck(m.group(2) == "required",
           f"E10 {SLUG[k]}/{fid}: quoted a {m.group(2)} failure, which cannot "
           f"gate a task")
        ck(m.group(3).strip().replace("\\|", "|")
           == " ".join(want["message"].split()),
           f"E10 {SLUG[k]}/{fid}: quote is not verbatim")

# ---- E14. section 4.3, the designed difficulty levers -------------------
# The same buckets as D13, but per document and from a fourth source: SF11 is
# built off failure_taxonomy.json's per-scenario aggregates, while D13 works
# off difficulty_axes.json's per-task rows. Recomputing here rather than
# reusing DXB is the point -- if the two disagree, one of the two dumps is
# wrong about this model and no single-source check would say so.
# Row order and the lever/control-arm split are the ones D13 recomputed off the
# parsed registry. Aliased rather than redefined: 4.3 and 5.3 render the same
# 27 buckets and close on the same lever count, and two local copies of that
# ordering could drift into ranking the same table two ways.
E14KNOBS, E14ORDER, E14LEVERS = DXKNOBS, DXORDER, DXLEVERS
E14ROWS = E14ORDER
FTAX = {}          # model -> {bucket: [n, passed, score_total]}, plus baseline
for k in K:
    _acc = collections.defaultdict(lambda: [0, 0, 0.0])
    for _scen, _c in FTM[k]["scenarios"].items():
        _sd = SEEDS.get(re.sub(r"_v\d+$", "", _scen))
        if _sd is None:
            continue
        _tot = _c["mean_score"] * _c["n"]
        for _tgt in ["__base__"] + sorted({b for _a, b in seed_buckets(_sd)}):
            _acc[_tgt][0] += _c["n"]
            _acc[_tgt][1] += _c["passed"]
            _acc[_tgt][2] += _tot
    FTAX[k] = _acc
    # the two dumps must agree before either is used to judge a rendered cell
    ck(_acc["__base__"][:2] == [DXBASE[k][0], DXBASE[k][1]],
       f"E14 {SLUG[k]}: failure_taxonomy baseline {_acc['__base__'][:2]} != "
       f"difficulty_axes {[DXBASE[k][0], DXBASE[k][1]]}")
    for _b in E14ROWS:
        ck(_acc[_b][:2] == [DXB[(k, _b)][0], DXB[(k, _b)][1]],
           f"E14 {SLUG[k]}/{_b}: failure_taxonomy {_acc[_b][:2]} != "
           f"difficulty_axes {[DXB[(k, _b)][0], DXB[(k, _b)][1]]}")


def e14pct(k, b):
    c = FTAX[k][b]
    return 100.0 * c[1] / c[0] if c[0] else 0.0


for k in K:
    sec = ddblk(k, "SF11")
    base = FTAX[k]["__base__"]
    brow = cells_of(sec, f"all {base[0]} tasks (baseline)")
    ck(brow is not None, f"E14 {SLUG[k]}: no baseline row in SF11")
    if brow:
        ck(brow[0] == "\u2014",
           f"E14 {SLUG[k]}: baseline category cell {brow[0]!r} is not an em dash")
        ck(as_int(brow[1]) == len(SEEDS),
           f"E14 {SLUG[k]}: baseline seeds {brow[1]!r} != {len(SEEDS)}")
        ck(as_int(brow[2]) == base[0] and as_int(brow[3]) == base[1],
           f"E14 {SLUG[k]}: baseline {brow[2]}/{brow[3]} != {base[0]}/{base[1]}")
        ck(dpeq(brow[4], e14pct(k, "__base__"), 1),
           f"E14 {SLUG[k]}: baseline rate {brow[4]!r} != "
           f"{e14pct(k, '__base__'):.1f}%")
        ck(brow[5] == "\u2014",
           f"E14 {SLUG[k]}: baseline delta cell {brow[5]!r} is not an em dash")
        ck(dpeq(brow[6], base[2] / base[0], 4),
           f"E14 {SLUG[k]}: baseline avg {brow[6]!r} != {base[2] / base[0]:.4f}")
        rankcell("design/baseline", k, brow[7],
                 {m: FTAX[m]["__base__"][1] for m in K}, "max")
        medcell("design/baseline", k, brow[8],
                {m: e14pct(m, "__base__") for m in K}, 1)
    # Row order is load-bearing: the table claims to be sorted by coverage, and
    # the reader is told in 4.3 to read the `seeds` column first. Checking each
    # row by label alone would pass on a table shuffled into any order.
    seen = []
    for _ln in sec.splitlines():
        _m = re.match(r"\|\s*`([^`]+)`\s*\|", _ln.strip())
        if _m:
            seen.append(_m.group(1))
    ck(seen == E14ORDER,
       f"E14 {SLUG[k]}: SF11 row order is not (knobs, then plugins by "
       f"descending seed count then name); first difference at "
       f"{next((i for i, (a, b) in enumerate(zip(seen, E14ORDER)) if a != b), len(seen))}")
    for b in E14ROWS:
        row, cell = cells_of(sec, b), FTAX[k][b]
        ck(row is not None, f"E14 {SLUG[k]}: no `{b}` row in SF11")
        if row is None:
            continue
        # `category` is the plugin's own declared category, or `knob` for the
        # four arms that are scenario switches rather than plugins. This is the
        # column the retired archetype map got wrong, so it is checked against
        # the parsed sources, not against any tool's copy.
        ck(row[0] == SPEC_CAT.get(b, "knob"),
           f"E14 {SLUG[k]}/{b}: category {row[0]!r} != "
           f"{SPEC_CAT.get(b, 'knob')!r}")
        ck(as_int(row[1]) == DXSEEDS[b],
           f"E14 {SLUG[k]}/{b}: seed count {row[1]!r} != {DXSEEDS[b]}")
        ck(as_int(row[2]) == cell[0] and as_int(row[3]) == cell[1],
           f"E14 {SLUG[k]}/{b}: {row[2]}/{row[3]} != {cell[0]}/{cell[1]}")
        ck(dpeq(row[4], e14pct(k, b), 1),
           f"E14 {SLUG[k]}/{b}: rate {row[4]!r} != {e14pct(k, b):.1f}%")
        ck(dpeq(row[5], e14pct(k, b) - e14pct(k, "__base__"), 1),
           f"E14 {SLUG[k]}/{b}: delta {row[5]!r} != "
           f"{e14pct(k, b) - e14pct(k, '__base__'):+.1f}")
        ck(dpeq(row[6], cell[2] / cell[0], 4),
           f"E14 {SLUG[k]}/{b}: avg {row[6]!r} != {cell[2] / cell[0]:.4f}")
        # Ranked on the integer pass count, never on the rate: the denominator
        # is the same for all nine inside a bucket, so the two orderings are
        # identical, but only the integer one is free of a float tie-break.
        rankcell(f"design/{b}", k, row[7], {m: FTAX[m][b][1] for m in K}, "max")
        medcell(f"design/{b}", k, row[8], {m: e14pct(m, b) for m in K}, 1)
    # The two knob axes partition the seed set, so their arms must sum to it.
    # A seeds column computed off the wrong axis still looks plausible per row.
    for _a, _b in (("async_required", "no_async"),
                   ("vision_graded", "not_vision_graded")):
        ck(DXSEEDS[_a] + DXSEEDS[_b] == len(SEEDS),
           f"E14: {_a}+{_b} cover {DXSEEDS[_a] + DXSEEDS[_b]} seeds, not "
           f"{len(SEEDS)}")
    # ---- the closing sentence ------------------------------------------
    below = sorted(((e14pct(k, b) - e14pct(k, "__base__"), b)
                    for b in E14LEVERS
                    if e14pct(k, b) < e14pct(k, "__base__")),
                   key=lambda r: (r[0], r[1]))
    if below:
        worst = below[0][1]
        thin = [b for _d, b in below if DXSEEDS[b] <= 3]
        ck(f"**{len(below)} of {len(E14LEVERS)} designed levers**" in sec,
           f"E14 {SLUG[k]}: the summary does not say {len(below)} of "
           f"{len(E14LEVERS)} levers are below baseline")
        # The thin-evidence count is the whole point of the row: most of what
        # looks like a designed difficulty is one or two scenarios deep.
        ck(f"**{len(thin)} of those {len(below)} are carried by 3 seeds or "
           f"fewer**" in flat(sec),
           f"E14 {SLUG[k]}: the summary does not say {len(thin)} of "
           f"{len(below)} below-baseline levers rest on <=3 seeds")
        # The named list is parsed out of the sentence itself and compared
        # whole. Probing for each name with a wildcard would search the table
        # too, where every lever appears by construction, so dropping one from
        # the sentence would still find it there and pass.
        #
        # The emitter names the five sharpest and extends through any tie it
        # lands inside, comparing the RENDERED delta -- most buckets are 30
        # tasks, so exact ties are common and cutting one off mid-tie would
        # invent a distinction the data does not make. Reproduced, not trusted.
        want = below[:5]
        while len(want) < len(below) and (f"{below[len(want)][0]:+.1f}"
                                          == f"{want[-1][0]:+.1f}"):
            want.append(below[len(want)])
        m = re.search(r"(?:The sharpest are|The only one is) (.*?)\. `",
                      flat(sec))
        ck(m is not None, f"E14 {SLUG[k]}: the summary sentence does not parse")
        ck(("The only one is " in sec) == (len(below) == 1),
           f"E14 {SLUG[k]}: singular/plural lead does not match "
           f"{len(below)} below-baseline lever(s)")
        if m:
            got = [(n, d.replace("\u2212", "-"))
                   for n, d in re.findall(r"`(\w+)` at (\S+) pp", m.group(1))]
            ck(bool(got) and got[0][0] == worst,
               f"E14 {SLUG[k]}: the sharpest lever is {worst}")
            ck(got == [(b, f"{d:+.1f}") for d, b in want],
               f"E14 {SLUG[k]}: the summary lists {got}, recomputed "
               f"{[(b, f'{d:+.1f}') for d, b in want]}")
        # The tail sentence sizes the sharpest lever's evidence base. Without
        # it the reader has no way to see that the headline number is one seed.
        _wn, _ws = FTAX[k][worst][0], DXSEEDS[worst]
        _wr = rank9(k, {m2: FTAX[m2][worst][1] for m2 in K}, "max")
        ck(f"`{worst}` is measured on {_wn} tasks from {_ws} "
           f"seed{'' if _ws == 1 else 's'}, where this model ranks {_wr} of 9 "
           f"by passes." in flat(sec),
           f"E14 {SLUG[k]}: the evidence-base sentence for {worst} is stale "
           f"(expected {_wn} tasks / {_ws} seeds / rank {_wr})")
    else:
        ck("**No designed lever** sits below" in sec,
           f"E14 {SLUG[k]}: no lever is below baseline but the summary claims "
           f"otherwise")
    ck("matching no seed" not in sec,
       f"E14 {SLUG[k]}: SF11 reports scenarios that join to no seed")

# ---- E12. no prose number without a generated table behind it -----------
# Prose is the one part of these documents a human types, so it is the only
# place an unaudited number can enter. Requiring every numeral in a prose slot
# to also appear in a generated block of the same document closes that hole: a
# derived figure has to go into the emitter, where pass E recomputes it, before
# it can be quoted. Integers of ten or less are exempt as structural (ranks,
# bucket counts, "one of nine"); everything larger, and anything with a decimal
# point, has to be traceable.
NUMTOK = re.compile(r"\d[\d,]*(?:\.\d+)?")
NOISE = re.compile(r"\bF[1-6]\b|\bSF\d+\b|\bv\d+\b|\[[^\]]*\]\([^)]*\)|`[^`]*`")

for k in K:
    doc = DD[k]
    pool = set()
    for tag in SF_BLOCKS:
        for tok in NUMTOK.findall(ddblk(k, tag)):
            pool.add(tok.replace(",", ""))
    for slot in PROSE_SLOTS:
        b, e = f"<!-- PROSE-{slot}-{k}-BEGIN -->", f"<!-- PROSE-{slot}-{k}-END -->"
        if b not in doc or e not in doc:
            continue
        body = doc.split(b, 1)[1].split(e, 1)[0].strip()
        if not body:
            continue          # E0 already reports the slot as unwritten
        ck(len(body) >= 120,
           f"E12 {SLUG[k]}/{slot}: prose is {len(body)} chars, too short to "
           f"be an interpretation")
        for tok in NUMTOK.findall(NOISE.sub(" ", body)):
            bare = tok.replace(",", "")
            if "." not in bare and int(bare) <= 10:
                continue
            ck(bare in pool,
               f"E12 {SLUG[k]}/{slot}: prose cites {tok!r}, which is in no "
               f"generated table of this document")
        for pat in (r"\bthe opus model\b", r"\bthe flash model\b",
                    r"\bthe sonnet\b", r"\bthe pro model\b",
                    r"\bthe gemini model\b"):
            ck(re.search(pat, body) is None,
               f"E12 {SLUG[k]}/{slot}: bare family word {pat!r} in prose")

# ---- E11. cross-document rank invariants --------------------------------
# A "rank of 9" column is only meaningful if, read across the nine documents,
# it actually ranks the nine values: better value => strictly smaller rank,
# equal values => equal rank. This catches a rank helper that is wrong in the
# same direction everywhere, which a per-document recomputation cannot.
ck(len(RANKCOL) >= 24, f"E11: only {len(RANKCOL)} rank columns were checked")
for metric, data in sorted(RANKCOL.items()):
    ck(len(data) == 9, f"E11 {metric}: {len(data)} of 9 documents reported")
    ck(min(r for r, _ in data.values()) == 1, f"E11 {metric}: no document ranks 1")
    ck(max(r for r, _ in data.values()) <= 9, f"E11 {metric}: a rank exceeds 9")
    by_rank = sorted(data.items(), key=lambda kv: kv[1][0])
    for (ka, (ra, va)), (kb, (rb, vb)) in zip(by_rank, by_rank[1:]):
        if ra == rb:
            ck(va == vb,
               f"E11 {metric}: {ka} and {kb} share rank {ra} on {va} vs {vb}")
        else:
            ck(va != vb,
               f"E11 {metric}: {ka} and {kb} both score {va} but rank "
               f"{ra} vs {rb}")

# ==========================================================================

print(f"checks run: {count}")
if fails:
    print(f"FAILURES: {len(fails)}")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ALL PASS")
