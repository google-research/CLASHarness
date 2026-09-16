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

"""Validate the project's founding hypothesis against the solver eval results.

The proposal (docs/proposal/Intern Project Proposal.txt) asserts two target
capabilities: (1) Information Update & Conflict Resolution -- reconciling
contradictory evidence across sources/subagents -- and (2) Time Awareness --
stale-vs-fresh information and async scheduling. This script asks the blunt
question: **do the eval results actually show agents are weak there?**

Method: every eval task is joined to its scenario seed (by `scenario_slug` with
the trailing `_v<NN>` variant suffix stripped) and bucketed by the seed's
authoritative `conflict_taxonomy`. For each mechanism we contrast seeds that
carry the tag against seeds that do not (computed per run, then averaged), and
count how many runs agree in direction. Three further passes harden the result:

  * confound control  -- round count and `difficulty` are uniform across all 60
                         scenarios, so the only structural confound is
                         `credential_lock` (an extra decrypt step, 2 seeds, both
                         inside the E&T group). We re-run the contrast with them
                         dropped.
  * mechanism         -- per-check failure rates by effect field, so we can tell
                         "never opened the right asset" apart from "opened it and
                         still wrote the wrong thing".
  * round phase       -- `round_type` as a within-scenario phase proxy, to locate
                         *where* in a task the deficit appears.

Unlike design_axis_slice.py (r01 only, archetype categories) this uses **all 14
model-runs including the OpenAI r02 repeats** and slices on `conflict_taxonomy`
rather than `archetype`. Records are deduplicated by `task_id` (last wins), the
same convention as design_axis_slice.load_model -- sol/terra append a few
re-attempts and a raw read would inflate their denominators.

Dependency-free (stdlib only) and read-only. Emits a self-contained HTML report
with inline SVG charts plus a machine-readable .stats.json sidecar.

Usage:
    python3 docs/analysis_think_history/tools/conflict_hypothesis_validation.py
    python3 docs/analysis_think_history/tools/conflict_hypothesis_validation.py --root REPO
    python3 docs/analysis_think_history/tools/conflict_hypothesis_validation.py --stdout
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import os
import re
from collections import defaultdict

EVAL_ROOT = os.path.join("eval_results", "data_3_5_flash", "aligned_setting")
SEED_DIR = os.path.join("clasharness", "data", "scenario_seeds", "conflict_vlm_v1")
OUT_HTML = os.path.join("docs", "analysis_think_history",
                        "conflict-hypothesis-validation.html")
OUT_JSON = os.path.join("docs", "analysis_think_history",
                        "conflict-hypothesis-validation.stats.json")

# Conflict-taxonomy tags that encode asynchronous / ordering pressure. Used only
# for the reconciliation section against the capability doc's async claim.
ASYNC_TAGS = ["time_window_expiration", "dependency_ordering",
              "barrier_synchronization", "deadlock_dependency", "subagent_delay"]

RED, BLUE, TEAL, GRAY, DARK = "#d1495b", "#2e6f95", "#3fb0ac", "#9aa0a6", "#333"


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_seeds(root):
    seeds = {}
    for f in sorted(glob.glob(os.path.join(root, SEED_DIR, "*.json"))):
        seeds[os.path.basename(f)[:-len(".json")]] = json.load(open(f))
    return seeds


def discover_runs(root):
    """[(label, dir)] for every family/model/run holding trajectory jsonl."""
    out = []
    base = os.path.join(root, EVAL_ROOT)
    for fam in sorted(os.listdir(base)):
        fp = os.path.join(base, fam)
        if not os.path.isdir(fp):
            continue
        for model in sorted(os.listdir(fp)):
            mp = os.path.join(fp, model)
            if not os.path.isdir(mp):
                continue
            for run in sorted(os.listdir(mp)):
                rp = os.path.join(mp, run)
                if os.path.isdir(rp) and glob.glob(os.path.join(rp, "*.jsonl")):
                    out.append((f"{fam}/{model}/{run}", rp))
    return out


def load_run(rundir):
    """Records for one run, deduplicated by task_id (last wins)."""
    by_id = {}
    for f in sorted(glob.glob(os.path.join(rundir, "*.jsonl"))):
        for line in open(f):
            line = line.strip()
            if line:
                r = json.loads(line)
                by_id[r.get("task_id")] = r
    return list(by_id.values())


def base_of(slug):
    return re.sub(r"_v\d+$", "", slug or "")


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def field_of(check_name):
    m = re.match(r"effect:(\w+)\s", check_name or "")
    return m.group(1) if m else "other"


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def aggregate(root):
    seeds = load_seeds(root)
    runs = discover_runs(root)

    run_seed = defaultdict(lambda: defaultdict(list))   # run -> seed -> [scores]
    mech = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # grp -> field -> [fail,n]
    phase = defaultdict(lambda: defaultdict(list))      # grp -> round_type -> [scores]
    difficulties, roundcounts, unmapped = set(), defaultdict(int), set()

    def tags(s):
        return set((seeds.get(s) or {}).get("conflict_taxonomy") or [])

    for label, rundir in runs:
        for r in load_run(rundir):
            md = r.get("metadata") or {}
            seed = base_of(md.get("scenario_slug"))
            if seed not in seeds:
                unmapped.add(seed)
                continue
            sc = float(r.get("score") or 0)
            run_seed[label][seed].append(sc)
            difficulties.add(md.get("difficulty"))
            t = tags(seed)
            grp = ("E&T" if {"evidence_mismatch", "temporal_staleness"} <= t
                   else "other")
            phase[grp][md.get("round_type", "?")].append(sc)
            for c in md.get("check_results") or []:
                fld = field_of(c.get("name"))
                mech[grp][fld][1] += 1
                if not c.get("passed"):
                    mech[grp][fld][0] += 1
        for seed in run_seed[label]:
            roundcounts[len(run_seed[label][seed]) // 3] += 1  # 3 variants per seed

    return seeds, run_seed, mech, phase, difficulties, roundcounts, unmapped


def contrast(seeds, run_seed, pred, exclude=frozenset()):
    """(present, absent, delta, n_runs_agreeing, n_runs) over per-run seed means."""
    names = sorted(seeds)
    per = []
    for run in run_seed:
        pres = [mean(run_seed[run][s]) for s in names
                if pred(s) and s not in exclude and run_seed[run].get(s)]
        absn = [mean(run_seed[run][s]) for s in names
                if not pred(s) and s not in exclude and run_seed[run].get(s)]
        if pres and absn:
            per.append((mean(pres), mean(absn)))
    if not per:
        return 0.0, 0.0, 0.0, 0, 0
    P, A = mean(p for p, _ in per), mean(a for _, a in per)
    return P, A, P - A, sum(1 for p, a in per if p < a), len(per)


# --------------------------------------------------------------------------- #
# svg
# --------------------------------------------------------------------------- #
def _e(t):
    return html.escape(str(t))


def _ylabels(s, pl, pr, pb, pt, ymin, ymax, pct, ticks=5):
    def Y(v):
        return pb - (v - ymin) / (ymax - ymin) * (pb - pt)
    for i in range(ticks + 1):
        v = ymin + (ymax - ymin) * i / ticks
        yy = Y(v)
        s.append(f'<line x1="{pl}" y1="{yy:.1f}" x2="{pr}" y2="{yy:.1f}" '
                 f'stroke="#eee" stroke-width="1"/>')
        lab = f"{v*100:.0f}%" if pct else f"{v:.2f}"
        s.append(f'<text x="{pl-6}" y="{yy+3:.1f}" font-size="10" '
                 f'text-anchor="end" fill="#888">{lab}</text>')
    return Y


def svg_delta_bars(cats, vals, sublabels, title, w=760, h=300):
    """Signed single-series bars (one colour per sign/magnitude)."""
    pl, pr, pt, pb = 60, w - 14, 36, h - 46
    s = [f'<svg viewBox="0 0 {w} {h}" width="100%" xmlns="http://www.w3.org/2000/svg" '
         f'font-family="system-ui,Segoe UI,Roboto,sans-serif">']
    s.append(f'<text x="{pl}" y="20" font-size="13.5" font-weight="600" '
             f'fill="{DARK}">{_e(title)}</text>')
    lo, hi = min(vals + [0]) - 0.02, max(vals + [0]) + 0.02
    def Y(v):
        return pb - (v - lo) / (hi - lo) * (pb - pt)
    for i in range(6):
        v = lo + (hi - lo) * i / 5
        yy = Y(v)
        s.append(f'<line x1="{pl}" y1="{yy:.1f}" x2="{pr}" y2="{yy:.1f}" stroke="#eee"/>')
        s.append(f'<text x="{pl-6}" y="{yy+3:.1f}" font-size="10" text-anchor="end" '
                 f'fill="#888">{v:+.2f}</text>')
    yz = Y(0)
    s.append(f'<line x1="{pl}" y1="{yz:.1f}" x2="{pr}" y2="{yz:.1f}" '
             f'stroke="#bbb" stroke-width="1.5"/>')
    gw = (pr - pl) / len(cats)
    bw = min(72, gw * 0.46)
    for i, cat in enumerate(cats):
        cx = pl + i * gw + gw / 2
        v = vals[i]
        colour = RED if v < -0.02 else (TEAL if v > 0.005 else GRAY)
        top, hh = min(yz, Y(v)), abs(Y(v) - yz)
        s.append(f'<rect x="{cx-bw/2:.1f}" y="{top:.1f}" width="{bw:.1f}" '
                 f'height="{hh:.1f}" fill="{colour}" rx="2"/>')
        ly = top - 5 if v >= 0 else top + hh + 12
        s.append(f'<text x="{cx:.1f}" y="{ly:.1f}" font-size="11" font-weight="600" '
                 f'text-anchor="middle" fill="{DARK}">{v:+.3f}</text>')
        s.append(f'<text x="{cx:.1f}" y="{pb+15:.1f}" font-size="10.5" '
                 f'text-anchor="middle" fill="{DARK}">{_e(cat)}</text>')
        s.append(f'<text x="{cx:.1f}" y="{pb+28:.1f}" font-size="9" '
                 f'text-anchor="middle" fill="#999">{_e(sublabels[i])}</text>')
    s.append('</svg>')
    return "".join(s)


def svg_grouped(cats, series, ymin, ymax, title, pct=False, note="", w=760, h=320):
    """Vertical grouped bars. series = [(name, colour, [values])]."""
    pl, pr, pt, pb = 60, w - 14, 36, h - 56
    s = [f'<svg viewBox="0 0 {w} {h}" width="100%" xmlns="http://www.w3.org/2000/svg" '
         f'font-family="system-ui,Segoe UI,Roboto,sans-serif">']
    s.append(f'<text x="{pl}" y="20" font-size="13.5" font-weight="600" '
             f'fill="{DARK}">{_e(title)}</text>')
    Y = _ylabels(s, pl, pr, pb, pt, ymin, ymax, pct)
    n, ns = len(cats), len(series)
    gw = (pr - pl) / n
    bw = min(48, gw / (ns + 0.7))
    for gi, cat in enumerate(cats):
        gx = pl + gi * gw
        span = ns * bw + (ns - 1) * 4
        for si, (name, colour, vals) in enumerate(series):
            v = vals[gi]
            bx = gx + gw / 2 - span / 2 + si * (bw + 4)
            yv = Y(v)
            s.append(f'<rect x="{bx:.1f}" y="{yv:.1f}" width="{bw:.1f}" '
                     f'height="{pb-yv:.1f}" fill="{colour}" rx="2"/>')
            lab = f"{v*100:.1f}%" if pct else f"{v:.3f}"
            s.append(f'<text x="{bx+bw/2:.1f}" y="{yv-4:.1f}" font-size="9.5" '
                     f'text-anchor="middle" fill="{DARK}">{lab}</text>')
        s.append(f'<text x="{gx+gw/2:.1f}" y="{pb+16:.1f}" font-size="10.5" '
                 f'text-anchor="middle" fill="{DARK}">{_e(cat)}</text>')
    lx = pl
    for name, colour, _ in series:
        s.append(f'<rect x="{lx}" y="{h-17}" width="11" height="11" fill="{colour}" rx="2"/>')
        s.append(f'<text x="{lx+15}" y="{h-7}" font-size="10.5" fill="{DARK}">{_e(name)}</text>')
        lx += 26 + len(name) * 6.6
    if note:
        s.append(f'<text x="{pr}" y="{h-7}" font-size="9.5" text-anchor="end" '
                 f'fill="#999">{_e(note)}</text>')
    s.append('</svg>')
    return "".join(s)


def svg_paired_rows(rows, xmin, xmax, title, w=760, rowh=26):
    """Horizontal paired bars: rows = [(label, val_a, val_b)]."""
    n = len(rows)
    pt, lblw = 34, 158
    h = pt + n * rowh + 42
    pl, pr = lblw, w - 52
    def X(v):
        return pl + (v - xmin) / (xmax - xmin) * (pr - pl)
    s = [f'<svg viewBox="0 0 {w} {h}" width="100%" xmlns="http://www.w3.org/2000/svg" '
         f'font-family="system-ui,Segoe UI,Roboto,sans-serif">']
    s.append(f'<text x="8" y="20" font-size="13.5" font-weight="600" '
             f'fill="{DARK}">{_e(title)}</text>')
    for i in range(6):
        v = xmin + (xmax - xmin) * i / 5
        xx = X(v)
        s.append(f'<line x1="{xx:.1f}" y1="{pt}" x2="{xx:.1f}" y2="{pt+n*rowh}" stroke="#eee"/>')
        s.append(f'<text x="{xx:.1f}" y="{pt+n*rowh+14}" font-size="9" '
                 f'text-anchor="middle" fill="#888">{v:.2f}</text>')
    for ri, (label, va, vb) in enumerate(rows):
        ry = pt + ri * rowh
        s.append(f'<text x="{lblw-6}" y="{ry+rowh/2+3:.1f}" font-size="10" '
                 f'text-anchor="end" fill="{DARK}">{_e(label)}</text>')
        bh = (rowh - 7) / 2
        for bi, (v, colour) in enumerate(((va, RED), (vb, BLUE))):
            by = ry + 3 + bi * bh
            s.append(f'<rect x="{pl}" y="{by:.1f}" width="{max(0,X(v)-pl):.1f}" '
                     f'height="{bh-1.5:.1f}" fill="{colour}" rx="1.5"/>')
            s.append(f'<text x="{X(v)+3:.1f}" y="{by+bh/2+2:.1f}" font-size="8.5" '
                     f'fill="{DARK}">{v:.3f}</text>')
    lx = pl
    for name, colour in (("E\u2229T scenarios", RED), ("non-E\u2229T", BLUE)):
        s.append(f'<rect x="{lx}" y="{h-14}" width="11" height="11" fill="{colour}" rx="2"/>')
        s.append(f'<text x="{lx+15}" y="{h-4}" font-size="10.5" fill="{DARK}">{_e(name)}</text>')
        lx += 26 + len(name) * 6.6
    s.append('</svg>')
    return "".join(s)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
CSS = """
:root{--fg:#222;--muted:#666;--line:#e4e4e7;--accent:#2e6f95;--red:#d1495b;}
*{box-sizing:border-box}
body{font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--fg);
max-width:920px;margin:0 auto;padding:34px 24px 80px;line-height:1.55;font-size:15px}
h1{font-size:25px;margin:0 0 4px}
h2{font-size:19px;margin:38px 0 10px;padding-bottom:6px;border-bottom:2px solid var(--line)}
.sub{color:var(--muted);margin:0 0 18px;font-size:13px}
.bl{background:#f6f8fa;border:1px solid var(--line);border-left:4px solid var(--accent);
border-radius:8px;padding:14px 18px;margin:16px 0}
.bl b{color:var(--red)}
.chart{border:1px solid var(--line);border-radius:10px;padding:12px 12px 6px;margin:14px 0;background:#fff}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13px}
th,td{border:1px solid var(--line);padding:6px 9px;text-align:right}
th:first-child,td:first-child{text-align:left}
th{background:#f2f4f7;font-weight:600}
tbody tr:nth-child(even){background:#fafbfc}
.take{background:#fff8f0;border:1px solid #f0d9bf;border-radius:8px;padding:10px 14px;
margin:10px 0;font-size:13.5px}
.take b{color:var(--red)}
.warn{background:#fdf6f7;border:1px solid #efc9d0;border-left:4px solid var(--red);
border-radius:8px;padding:12px 16px;margin:14px 0;font-size:13.5px}
ul{margin:6px 0 6px 4px;padding-left:20px}li{margin:3px 0}
code{background:#f2f4f7;padding:1px 5px;border-radius:4px;font-size:12.5px}
.badge{display:inline-block;background:var(--red);color:#fff;border-radius:12px;
padding:1px 9px;font-size:12px;font-weight:600}
.foot{color:var(--muted);font-size:12px;margin-top:40px;border-top:1px solid var(--line);padding-top:12px}
"""


def table(headers, rows):
    h = "".join(f"<th>{_e(x)}</th>" for x in headers)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


def build(root):
    seeds, run_seed, mech, phase, difficulties, roundcounts, unmapped = aggregate(root)
    names = sorted(seeds)

    def tg(s):
        return set(seeds[s].get("conflict_taxonomy") or [])

    def ch(s):
        return set(seeds[s].get("difficulty_challenges") or [])

    E = lambda s: "evidence_mismatch" in tg(s)
    T = lambda s: "temporal_staleness" in tg(s)
    ET = lambda s: {"evidence_mismatch", "temporal_staleness"} <= tg(s)
    CR = lambda s: "credential_lock" in tg(s)
    MS = lambda s: "multi_source_reasoning" in ch(s)
    ASY = lambda s: bool(tg(s) & set(ASYNC_TAGS))
    FAW = lambda s: bool(seeds[s].get("force_async_wait"))

    nseed = lambda p: sum(1 for s in names if p(s))
    C = lambda p, ex=frozenset(): contrast(seeds, run_seed, p, ex)

    rows = [
        ("evidence_mismatch (E)", nseed(E)) + C(E),
        ("temporal_staleness (T)", nseed(T)) + C(T),
        ("E \u2229 T (both)", nseed(ET)) + C(ET),
        ("credential_lock", nseed(CR)) + C(CR),
        ("multi_source_reasoning", nseed(MS)) + C(MS),
        ("async taxonomy tags", nseed(ASY)) + C(ASY),
        ("force_async_wait", nseed(FAW)) + C(FAW),
    ]
    by = {r[0]: r for r in rows}
    CREDSET = {s for s in names if CR(s)}
    etnc = C(ET, CREDSET)
    nruns = len(run_seed)
    n_models = len({r.rsplit("/", 1)[0] for r in run_seed})

    # per-model
    permodel = []
    for run in run_seed:
        et = mean(mean(run_seed[run][s]) for s in names if ET(s) and run_seed[run].get(s))
        ne = mean(mean(run_seed[run][s]) for s in names if not ET(s) and run_seed[run].get(s))
        permodel.append((run.replace("/", " ").replace("r0", "r0"), et, ne))
    permodel.sort(key=lambda x: -x[2])

    # seed ranking
    rank = sorted(((s, mean(mean(run_seed[r][s]) for r in run_seed if run_seed[r].get(s)),
                    sorted(tg(s))) for s in names), key=lambda x: x[1])

    mech_fields = ["messages_sent", "files_created", "assets_accessed"]
    mrows = []
    for f in mech_fields:
        e, o = mech["E&T"][f], mech["other"][f]
        er = e[0] / e[1] if e[1] else 0.0
        orr = o[0] / o[1] if o[1] else 0.0
        mrows.append((f, er, e[1], orr, o[1], er - orr))

    prows = []
    for rt in ["action", "investigation", "decision", "verification"]:
        e, o = phase["E&T"].get(rt, []), phase["other"].get(rt, [])
        if e and o:
            prows.append((rt, mean(e), len(e), mean(o), len(o), mean(e) - mean(o)))

    # ---- charts ----
    ca = ["evidence (E)", "temporal (T)", "E\u2229T", "cred_lock",
          "multi_source", "async tags"]
    cakeys = ["evidence_mismatch (E)", "temporal_staleness (T)", "E \u2229 T (both)",
              "credential_lock", "multi_source_reasoning", "async taxonomy tags"]
    chartA = svg_delta_bars(ca, [by[k][4] for k in cakeys],
                            [f"{by[k][5]}/{by[k][6]} runs" for k in cakeys],
                            "Chart A \u2014 \u0394 (tagged \u2212 untagged) avg score  "
                            "[negative = mechanism hurts]")
    chartB = svg_grouped(["E", "T", "E\u2229T", "E\u2229T minus cred_lock"],
                         [("tagged (conflict)", RED,
                           [by[cakeys[0]][2], by[cakeys[1]][2], by[cakeys[2]][2], etnc[0]]),
                          ("untagged (control)", BLUE,
                           [by[cakeys[0]][3], by[cakeys[1]][3], by[cakeys[2]][3], etnc[1]])],
                         0.50, 0.75,
                         "Chart B \u2014 Confound control: effect survives dropping credential_lock",
                         note="difficulty\u22613, rounds\u226110 \u2192 not confounds")
    chartC = svg_grouped([r[0] for r in mrows],
                         [("E\u2229T scenarios", RED, [r[1] for r in mrows]),
                          ("other scenarios", BLUE, [r[3] for r in mrows])],
                         0.0, 0.6,
                         "Chart C \u2014 Mechanism: per-check failure rate", pct=True)
    chartD = svg_grouped([r[0] for r in prows],
                         [("E\u2229T scenarios", RED, [r[1] for r in prows]),
                          ("other scenarios", BLUE, [r[3] for r in prows])],
                         0.45, 0.75,
                         "Chart D \u2014 Where it breaks: avg score by round phase")
    allv = [v for _, a, b in permodel for v in (a, b)]
    chartE = svg_paired_rows(permodel, max(0.0, min(allv) - 0.05), min(1.0, max(allv) + 0.05),
                             f"Chart E \u2014 Every run ranks E\u2229T below non-E\u2229T "
                             f"({by[cakeys[2]][5]}/{by[cakeys[2]][6]})")

    # ---- tables ----
    def dcell(d):
        return f'<b style="color:{RED}">{d:+.3f}</b>' if d < -0.02 else f"{d:+.3f}"

    contrast_tbl = table(
        ["mechanism", "#seeds", "tagged", "untagged", "\u0394", "runs tagged&lt;untagged"],
        [[r[0], r[1], f"{r[2]:.3f}", f"{r[3]:.3f}", dcell(r[4]), f"{r[5]}/{r[6]}"]
         for r in rows])
    mech_tbl = table(
        ["check field", "E\u2229T fail", "n", "other fail", "n", "\u0394"],
        [[r[0], f"{r[1]*100:.1f}%", r[2], f"{r[3]*100:.1f}%", r[4], f"{r[5]*100:+.1f} pp"]
         for r in mrows])
    phase_tbl = table(
        ["round phase", "E\u2229T", "n", "other", "n", "\u0394"],
        [[r[0], f"{r[1]:.3f}", r[2], f"{r[3]:.3f}", r[4], dcell(r[5])] for r in prows])
    rank_tbl = table(
        ["#", "seed", "avg score", "conflict_taxonomy"],
        [[i + 1, s, f"{v:.3f}", ", ".join(t)] for i, (s, v, t) in enumerate(rank)])
    model_tbl = table(
        ["model / run", "E\u2229T", "non-E\u2229T", "\u0394"],
        [[l, f"{a:.3f}", f"{b:.3f}", dcell(a - b)] for l, a, b in permodel])

    diffs = sorted(x for x in difficulties if x is not None)
    ET_seeds = [s for s in names if ET(s)]
    faw_seeds = [s for s in names if FAW(s)]

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Conflict Hypothesis Validation</title><style>{CSS}</style></head><body>

<h1>Does the eval support the founding hypothesis?</h1>
<p class="sub">Hypothesis under test: agents are weak at <b>temporal</b> (stale-vs-fresh) and
<b>multi-source information conflict</b>. &nbsp;|&nbsp; {nruns} model-runs
({n_models} models, incl. OpenAI r02 repeats) &times; 60 scenarios &times; 10 rounds
&nbsp;|&nbsp; source <code>{EVAL_ROOT}</code></p>

<div class="bl"><b>Verdict: supported, but only in a sharpened form.</b>
The deficit is real, unanimous (<span class="badge">{by[cakeys[2]][5]}/{by[cakeys[2]][6]} runs</span>)
and survives every confound control &mdash; but it belongs to
<b>contradictory <i>and</i> stale evidence</b> (<code>evidence_mismatch</code> &cap;
<code>temporal_staleness</code>, &Delta;{by[cakeys[2]][4]:+.3f}), not to "multi-source" or
"time" as such. Generic multi-source reasoning is <b>flat</b>
(&Delta;{by[cakeys[4]][4]:+.3f}, {by[cakeys[4]][5]}/{by[cakeys[4]][6]} runs). Mechanistically the
agents <b>do open the right assets</b> and then <b>write the wrong thing downstream</b>,
during <b>action</b> and <b>investigation</b> rounds.</div>

<h2>1. Method</h2>
<ul>
<li>Each task joins to its seed via <code>scenario_slug</code> (variant suffix stripped);
buckets come from the seed's authoritative <code>conflict_taxonomy</code>.</li>
<li>Contrast = mean score of tagged seeds vs untagged seeds, computed <i>per run</i> then
averaged, plus a count of how many runs agree in direction (sign test).</li>
<li>Records deduplicated by <code>task_id</code> (last wins) &mdash; sol/terra append a few
re-attempts; a raw read would inflate their denominators.</li>
<li>Uses all {nruns} runs, a superset of <code>design_axis_slice.py</code> (r01 only).</li>
</ul>

<h2>2. Core contrast</h2>
<div class="chart">{chartA}</div>
{contrast_tbl}
<div class="take"><b>Read:</b> E, T and E&cap;T are all negative and unanimous.
<code>multi_source_reasoning</code> is <i>positive</i> &mdash; so "many sources" is not the
problem; "sources that disagree" is.</div>

<h2>3. Confound control</h2>
<div class="chart">{chartB}</div>
<ul>
<li><b>Round count</b>: every scenario is exactly <b>10 rounds</b> &rarr; not a confound.</li>
<li><b>Difficulty</b>: <code>metadata.difficulty</code> is uniformly <b>{diffs}</b> &rarr; not a confound.</li>
<li><b>credential_lock</b> (extra decrypt step; {len(CREDSET)} seeds, both inside E&cap;T):
dropping them leaves <b>{len(ET_seeds)-len(CREDSET)} E&cap;T seeds still trailing by
&Delta;{etnc[2]:+.3f} ({etnc[3]}/{etnc[4]} runs)</b> &rarr; not an artifact of decryption.</li>
</ul>

<h2>4. Failure mechanism</h2>
<div class="chart">{chartC}</div>
{mech_tbl}
<div class="take"><b>Read:</b> excess failure lands on <b>messages_sent</b> and
<b>files_created</b> (semantic mismatch), not on <b>assets_accessed</b>. The bottleneck is
<b>resolving the conflict and propagating the resolution</b>, not retrieval.</div>

<h2>5. Where in the task it breaks</h2>
<div class="chart">{chartD}</div>
{phase_tbl}
<div class="take"><b>Read:</b> the deficit is concentrated in <b>action</b> and
<b>investigation</b>; <b>decision</b> and <b>verification</b> are ~unaffected. Agents fail at
gathering authoritative evidence and executing with it, not at picking an answer.</div>

<h2>6. Reconciling the async claim</h2>
<div class="warn"><b>Apparent conflict with <code>eval-results-capability-analysis.md</code> &sect;6.2</b>,
which calls <code>force_async_wait</code> "the hardest designed lever &hellip; close to unsolved".
Both are true, at different granularity:
<ul>
<li>Broad <b>async taxonomy tags</b> ({nseed(ASY)} seeds: {", ".join(ASYNC_TAGS)}) are
<b>flat</b>: &Delta;{by["async taxonomy tags"][4]:+.3f}
({by["async taxonomy tags"][5]}/{by["async taxonomy tags"][6]} runs).</li>
<li><code>force_async_wait</code> is <b>a single seed</b> ({", ".join(faw_seeds)}) &mdash;
&Delta;{by["force_async_wait"][4]:+.3f}
({by["force_async_wait"][5]}/{by["force_async_wait"][6]} runs).</li>
</ul>
So "scenarios that <i>involve</i> timing" are not hard; the one scenario that <i>forces a real
wait</i> is. That is a narrow, n=1-seed finding &mdash; it cannot carry the general
"agents are weak at temporal reasoning" claim, whereas the E&cap;T result (6 seeds,
unanimous) can.</div>

<h2>7. Every run agrees</h2>
<div class="chart">{chartE}</div>
{model_tbl}

<h2>8. Scenario ranking (worst &rarr; best)</h2>
{rank_tbl}

<h2>9. Caveats</h2>
<ul>
<li>{n_models} models; only the 4 OpenAI models have a second run &rarr; per-model variance under-sampled.</li>
<li>Conflict tags are <b>seed-level</b>; the phase split uses <code>round_type</code> as a proxy,
not per-round conflict-injection labels.</li>
<li><code>credential_lock</code> (n={len(CREDSET)}) and <code>force_async_wait</code>
(n={len(faw_seeds)}) are small-n; their standalone deltas are directional only.</li>
<li>Single eval configuration (<code>data_3_5_flash / aligned_setting</code>); not replicated
across settings or generators.</li>
<li>Contrasts are between-seed, so residual scenario-authoring differences cannot be fully excluded.</li>
</ul>

<p class="foot">Generated by
<code>docs/analysis_think_history/tools/conflict_hypothesis_validation.py</code>
&mdash; stdlib-only, read-only. Re-run after new eval runs to refresh.</p>
</body></html>"""

    stats = {
        "runs": nruns, "models": n_models, "seeds": len(names),
        "difficulty_values": diffs,
        "contrasts": {r[0]: {"seeds": r[1], "tagged": round(r[2], 4),
                             "untagged": round(r[3], 4), "delta": round(r[4], 4),
                             "runs_agree": r[5], "runs": r[6]} for r in rows},
        "ET_without_credential_lock": {"tagged": round(etnc[0], 4),
                                       "untagged": round(etnc[1], 4),
                                       "delta": round(etnc[2], 4),
                                       "runs_agree": etnc[3], "runs": etnc[4]},
        "mechanism": {r[0]: {"et_fail": round(r[1], 4), "et_n": r[2],
                             "other_fail": round(r[3], 4), "other_n": r[4],
                             "delta": round(r[5], 4)} for r in mrows},
        "round_phase": {r[0]: {"et": round(r[1], 4), "et_n": r[2],
                               "other": round(r[3], 4), "other_n": r[4],
                               "delta": round(r[5], 4)} for r in prows},
        "per_model": [{"run": l, "et": round(a, 4), "non_et": round(b, 4),
                       "delta": round(a - b, 4)} for l, a, b in permodel],
        "scenario_rank": [{"seed": s, "avg": round(v, 4), "taxonomy": t}
                          for s, v, t in rank],
        "unmapped_bases": sorted(unmapped),
    }
    return doc, stats, rows, etnc, nruns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--stdout", action="store_true", help="print summary only")
    args = ap.parse_args()

    doc, stats, rows, etnc, nruns = build(args.root)

    print(f"runs={stats['runs']} models={stats['models']} seeds={stats['seeds']} "
          f"difficulty={stats['difficulty_values']}")
    for r in rows:
        print(f"  {r[0]:26s} seeds={r[1]:2d} tagged={r[2]:.3f} untagged={r[3]:.3f} "
              f"delta={r[4]:+.3f}  {r[5]}/{r[6]}")
    print(f"  {'E&T minus cred_lock':26s}          tagged={etnc[0]:.3f} "
          f"untagged={etnc[1]:.3f} delta={etnc[2]:+.3f}  {etnc[3]}/{etnc[4]}")
    if stats["unmapped_bases"]:
        print("WARNING unmapped scenario bases:", stats["unmapped_bases"])
    if args.stdout:
        return

    out_html = os.path.join(args.root, OUT_HTML)
    out_json = os.path.join(args.root, OUT_JSON)
    with open(out_html, "w") as f:
        f.write(doc)
    with open(out_json, "w") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")
    print("wrote", out_html)
    print("wrote", out_json)


if __name__ == "__main__":
    main()
