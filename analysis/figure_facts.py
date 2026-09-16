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

"""Every number Figure 2 DRAWS, checked against the generator that owns it.

Figure 2 (src/figures/body-ablation.svg) was the one exhibit in the paper with
no generator and no verifier: src/README.md said so outright. Everything else
is guarded -- src/verify_table_bodies.py diffs each tabular against the script
that emitted it, analysis/results_facts.py asserts the prose, and
analysis/section3_facts.py asserts Section 3 -- but this panel was hand-drawn
SVG, so its forty-odd numbers were only ever checked by eye. Three of them were
wrong when this was written:

  * (c) Ret% for claude-opus-5, claude-sonnet-5 and gemini-3.1-pro were
    computed by dividing two ALREADY-ROUNDED one-decimal labels. 19.4/25.7
    gives 0.755 and 11.2/21.3 gives 0.526, which round to the same 76 and 53
    the panel showed -- but the generator's two-decimal values give 0.7552 and
    0.5247, so sonnet-5 is 52, not 53. results_facts.py had a comment warning
    about exactly this division since 2026-08-13; nothing enforced it on the
    figure.
  * (b) the all-600 drop for claude-sonnet-5 read -87%. retained is 0.136, so
    the drop is 86.4 points: -86%.

This module closes that hole. It does NOT recompute anything: per the design
note at the top of results_facts.py there is exactly one source of truth per
number and it is the generator, so every expected value here is read out of
results_facts.ablation_facts() and results_facts.generator_facts(), which parse
the generators' own stdout.

WHY IT PARSES COLOUR AND GEOMETRY RATHER THAN ANNOTATIONS. The obvious
alternative is to tag each <text> with a data-* key and look the keys up. That
would check only the typed labels. Half of Figure 2(a) is not labelled at all
-- b=20 and b=30 are drawn as bare vertices, and only b=40 carries text -- so a
label-only checker would leave two thirds of that panel unguarded. Reading the
coordinates back through each panel's stated scale checks what the reader
actually sees, and it needs no edit to the SVG, hence no re-render of the PDF.

The file is served to headless Chrome inside an HTML page (see
src/extract_figures.py), not as image/svg+xml, so it legitimately uses HTML
entities -- &minus;, &ndash;, &rarr; -- that are undefined in XML. ElementTree
therefore cannot read it and html.parser can.

Run:  python3 analysis/figure_facts.py          # print every drawn value
      python3 analysis/figure_facts.py --check  # assert them (also via
                                                # results_facts.py --check)
"""
import os
import re
import sys
from html.parser import HTMLParser

HERE = os.path.dirname(os.path.abspath(__file__))
SVG = None  # resolved lazily via _paper_file("src/figures/body-ablation.svg")

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


# Panel bounding boxes, from the three background <rect>s.
PANEL_A = (8, 288)
PANEL_B = (298, 582)
PANEL_C = (592, 892)

# --- panel scales -----------------------------------------------------------
# (a) y gridlines sit at 195/146/97/48 and are labelled 0/10/20/30%, so the
#     axis is 147px for 30 points.
A_Y0, A_PX = 195.0, 4.9
# (b) has no axis; the bars are 4.5px per point. Asserted against every bar
#     rather than trusted, so a single mis-scaled bar cannot hide.
B_PX = 4.5
# (c) is a slopegraph with no axis either; the file documents its own mapping
#     as y = 62.7 + 5.2222 * (25.2 - value), anchored on the top-left point.
C_Y0, C_V0, C_PX = 62.7, 25.2, 5.2222
C_DOT_R = 3.6  # slopegraph dot radius; two dots closer than 2r occlude

# Rounding a drawn coordinate back through the scale loses ~0.01 of a point;
# 0.06 is loose enough to survive that and tight enough that a 0.1 typo trips.
GEOM_TOL = 0.06

SHORT2FULL = {
    "opus-5": "claude-opus-5", "opus-4.8": "claude-opus-4.8",
    "sonnet-5": "claude-sonnet-5", "3.7-flash": "gemini-3.7-flash",
    "3.6-flash": "gemini-3.6-flash", "3.5-flash": "gemini-3.5-flash",
    "3.1-pro": "gemini-3.1-pro",
}
# results_facts abbreviates the ablation roster differently again.
SHORT2AB = {"opus-5": "opus5", "sonnet-5": "sonnet5",
            "3.6-flash": "36flash", "3.5-flash": "35flash"}


class _Svg(HTMLParser):
    """Flat element list in document order: (tag, attrs, text).

    Document order is load-bearing. Figure 2(c) deliberately paints
    gemini-3.7-flash before gemini-3.6-flash so the upper dot ends up on top of
    the lower one, and (a), (b) and (c) are all read as row blocks delimited by
    their first element, so a reordering must be visible here.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.els = []
        self._depth = 0

    def handle_starttag(self, tag, attrs):
        self.els.append([tag, dict(attrs), ""])
        if tag in ("text", "tspan"):
            self._depth += 1

    def handle_startendtag(self, tag, attrs):
        self.els.append([tag, dict(attrs), ""])

    def handle_endtag(self, tag):
        if tag in ("text", "tspan"):
            self._depth = max(0, self._depth - 1)

    def handle_data(self, data):
        # Accumulate onto the innermost open <text>, tspans included, so the
        # cap-hit strip in (a) reads as one string.
        if self._depth:
            for el in reversed(self.els):
                if el[0] == "text":
                    el[2] += data
                    return


def _num(s, default=None):
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def parse_svg(path=None):
    p = _Svg()
    if path is None:
        path = _paper_file(os.path.join("src", "figures", "body-ablation.svg"))
    with open(path, encoding="utf-8") as fh:
        p.feed(fh.read())
    out = []
    for tag, a, txt in p.els:
        a = dict(a)
        a["_x"] = _num(a.get("x", a.get("cx", a.get("x1"))))
        a["_y"] = _num(a.get("y", a.get("cy", a.get("y1"))))
        if a["_x"] is None and a.get("d"):
            # <path> carries no x/cx, and the slopes in (c) are the row
            # delimiters, so locate one by the moveto that starts it.
            m = re.match(r"\s*M\s*([\d.]+)[ ,]+([\d.]+)", a["d"])
            if m:
                a["_x"], a["_y"] = float(m.group(1)), float(m.group(2))
        out.append((tag, a, re.sub(r"\s+", " ", txt).strip()))
    return out


def _in(panel, a):
    x = a["_x"]
    return x is not None and panel[0] <= x <= panel[1]


def _colour(a):
    """The series colour of a shape: whichever of fill/stroke is not white."""
    for k in ("fill", "stroke"):
        v = a.get(k)
        if v and v != "#ffffff" and not v.startswith("url("):
            return v
    return None


def _verts(d):
    """The explicit coordinate pairs of a path, in order."""
    return [(float(x), float(y))
            for x, y in re.findall(r"([\d.]+),([\d.]+)", d)]


def _pct(txt):
    m = re.match(r"^&?(?:minus)?[\u2212-]?([\d.]+)%$", txt.replace("\u2212", "-"))
    return float(m.group(1)) if m else None


def figure_facts():
    """Summary facts for results_facts.py's SECTIONS/SHIPPED contract.

    The per-quantity detail stays out of the returned dict on purpose: it is
    155 entries and results_facts.py prints every fact it holds. What ships is
    the tally, the three roster sizes, and the mismatch list, which must be
    empty. A wrong number lands in the list, a deleted one drops the tally and
    a new unwired one raises it, so none of the three can move silently.
    """
    drawn, bad = _compare()
    return {"fig2_checked": len(drawn), "fig2_mismatches": tuple(bad),
            "fig2_a_curves": drawn["a.curves"], "fig2_b_rows": drawn["b.rows"],
            "fig2_c_rows": drawn["c.rows"]}


def _results_facts():
    """The results_facts module, reusing the running instance if there is one.

    Resolved here rather than at module scope because results_facts imports
    this file, so a top-level import back would be a cycle.

    The sys.modules lookup is not paranoia. Under `python3 results_facts.py`
    that file is __main__, so a plain `import results_facts` would build a
    SECOND module object with its own empty run() memo -- and under --fresh,
    which disables the disk cache as well, ablation.py and
    generator_ablation.py would each be run a second time. Same numbers, twice
    the bootstrap.
    """
    for name in ("__main__", "results_facts"):
        mod = sys.modules.get(name)
        if mod is not None and hasattr(mod, "ablation_facts"):
            return mod
    import results_facts
    return results_facts


def _compare():
    """Compare every quantity Figure 2 draws against its generator."""
    RF = _results_facts()
    ab, gen = RF.ablation_facts(), RF.generator_facts()
    els = parse_svg()

    bad, drawn = [], {}

    def check(key, got, want, tol=0.0):
        drawn[key] = got
        ok = (abs(got - want) <= tol) if isinstance(got, (int, float)) \
            and isinstance(want, (int, float)) else got == want
        if not ok:
            bad.append(f"{key}: figure draws {got!r}, generator says {want!r}")

    # ================= panel (a): step budget ==============================
    a_els = [(t, a, x) for t, a, x in els if _in(PANEL_A, a)]

    # Gridlines carry the scale; if they move, every curve below is misread.
    grid = sorted(a["_y"] for t, a, _ in a_els
                  if t == "text" and a["_x"] == 34)
    check("a.scale", [round((A_Y0 - (y - 4)) / A_PX) for y in grid],
          [30, 20, 10, 0])

    # colour -> (short name, drawn b40 label), from the curve end labels.
    a_lbl = {}
    for t, a, txt in a_els:
        if t == "text" and a["_x"] == 182:
            m = re.match(r"^(\S+) ([\d.]+)%$", txt)
            a_lbl[a["fill"]] = (m.group(1), float(m.group(2)))
    check("a.curves", len(a_lbl), 4)

    a_dots, a_line = {}, {}
    for t, a, _ in a_els:
        if t == "circle":
            a_dots.setdefault(_colour(a), []).append((a["_x"], a["_y"]))
        elif t == "path" and a.get("stroke") in a_lbl:
            a_line[a["stroke"]] = _verts(a["d"])
    for col, (short, lbl40) in a_lbl.items():
        s = SHORT2AB[short]
        pts = sorted(a_dots[col])
        check(f"a.{short}.vertices", len(pts), 3)
        for (_, y), b in zip(pts, (20, 30, 40)):
            check(f"a.{short}.b{b}.dot", round((A_Y0 - y) / A_PX, 2),
                  ab[f"ab_{s}_{b}"], GEOM_TOL)
        # The polyline is what the reader actually follows, and it is a second
        # copy of the same three coordinates. Only b=40 carries a label, so
        # without this a moved b=20 or b=30 vertex would be drawn wrong with
        # nothing on the panel to contradict it.
        check(f"a.{short}.polyline", a_line[col], pts)
        check(f"a.{short}.b40.label", lbl40, ab[f"ab_{s}_40"])

    hdr = next(x for t, a, x in a_els if t == "text" and a["_x"] == 278)
    m = re.match(r"^(\S+) \+([\d.]+) pp$", hdr)
    check("a.hdr.model", SHORT2AB[m.group(1).lower()], "35flash")
    check("a.hdr.delta", float(m.group(2)), ab["ab_35flash_delta"])
    arrow = next(x for t, a, x in a_els if t == "text" and a["_x"] == 96)
    check("a.arrow.delta", float(re.match(r"^\+([\d.]+) pp$", arrow).group(1)),
          ab["ab_35flash_delta"])

    # Cap-hit strip. Both figures are integers on the page; 24.7 -> 25 is a
    # rounding the strip is allowed to make, 0.3 is printed as given.
    strip = next(x for t, a, x in a_els if t == "text" and a["_x"] == 148)
    m = re.search(r"3\.5-flash \(([\d.]+)%\u2192([\d.]+)%\).*?"
                  r"3\.6-flash \(([\d.]+)%\u2192([\d.]+)%\)", strip)
    check("a.cap.35flash.b20", float(m.group(1)), round(ab["ab_35flash_cap20"]))
    check("a.cap.35flash.b40", float(m.group(2)), round(ab["ab_35flash_cap40"]))
    check("a.cap.36flash.b20", float(m.group(3)), round(ab["ab_36flash_cap20"]))
    check("a.cap.36flash.b40", float(m.group(4)), ab["ab_36flash_cap40"])

    # ================= panel (b): visual modality ==========================
    b_els = [(t, a, x) for t, a, x in els if _in(PANEL_B, a)]
    blocks, cur = [], None
    for t, a, txt in b_els:
        if t == "text" and a["_x"] == 358:
            cur = {"model": txt, "y": a["_y"], "els": []}
            blocks.append(cur)
        elif cur is not None:
            cur["els"].append((t, a, txt))
    check("b.rows", len(blocks), 6)

    drops = []
    for i, blk in enumerate(blocks):
        arm = "all" if i < 3 else "ung"
        short, s = blk["model"], SHORT2AB[blk["model"]]
        tag = f"b.{arm}.{short}"
        # The 254-task card starts at y=119; the three all-600 rows must sit
        # above it and the three subset rows inside it.
        check(f"{tag}.arm", blk["y"] < 119, arm == "all")
        sight, blind = ab[f"ab_mod_{s}_{arm}_sight"], ab[f"ab_mod_{s}_{arm}_blind"]

        # x=366 is the bar origin. Filtering on it drops the two panel
        # furniture rects -- the 254-task card and the verdict strip -- which
        # fall inside these blocks in document order.
        bars = [a for t, a, _ in blk["els"] if t == "rect" and a["_x"] == 366]
        check(f"{tag}.bars", len(bars), 3)
        solid = [a for a in bars if a.get("fill") == "#5072a7"]
        hatch = [a for a in bars if str(a.get("fill", "")).startswith("url(")]
        red = [a for a in bars if a.get("fill") == "#c7585c"]
        check(f"{tag}.sight.bar", _num(solid[0]["width"]) / B_PX, sight, GEOM_TOL)
        check(f"{tag}.sight.hatch", _num(hatch[0]["width"]),
              _num(solid[0]["width"]))
        check(f"{tag}.blind.bar", _num(red[0]["width"]) / B_PX, blind, GEOM_TOL)

        txts = [(a, x) for t, a, x in blk["els"] if t == "text"]
        st = [x for a, x in txts if a.get("fill") == "#5072a7"]
        bt = [x for a, x in txts
              if a.get("fill") == "#c7585c" and "text-anchor" not in a]
        dt = [x for a, x in txts
              if a.get("fill") == "#c7585c" and a.get("text-anchor") == "end"]
        check(f"{tag}.sight.label", _pct(st[0]), sight)
        check(f"{tag}.blind.label", _pct(bt[0]), blind)
        # The drop is one minus the generator's own retained ratio. Deriving it
        # from the two rounded pass rates instead is the (c) Ret% bug.
        want_drop = round(100 * (1 - ab[f"ab_mod_{s}_{arm}_ret"]))
        check(f"{tag}.drop", _pct(dt[0]), want_drop)
        drops.append(want_drop)

    hdr = next(x for t, a, x in b_els
               if t == "text" and a["_x"] == 572 and a["_y"] == 24)
    m = re.match(r"^\u2212(\d+) to \u2212(\d+)%$", hdr)
    check("b.hdr.lo", int(m.group(1)), min(drops))
    check("b.hdr.hi", int(m.group(2)), max(drops))

    card = next(x for t, a, x in b_els if t == "text" and a["_x"] == 312
                and a["_y"] == 133)
    check("b.card.n", int(re.search(r"\((\d+) Tasks\)", card).group(1)),
          ab["ab_ungated_n"])

    verdict = next(x for t, a, x in b_els if t == "text" and a["_x"] == 440)
    m = re.search(r"Keeps (\d+)\u2013(\d+)% of Sighted", verdict)
    check("b.verdict.lo", int(m.group(1)),
          int(100 * ab["ab_mod_retained_lo"]))
    check("b.verdict.hi", int(m.group(2)),
          int(100 * ab["ab_mod_retained_hi"]))

    # ================= panel (c): generator difference =====================
    c_els = [(t, a, x) for t, a, x in els if _in(PANEL_C, a)]
    blocks, cur = [], None
    for t, a, txt in c_els:
        if t == "path":
            cur = {"slope": _verts(a["d"]), "els": []}
            blocks.append(cur)
        elif cur is not None:
            cur["els"].append((t, a, txt))
    check("c.rows", len(blocks), 7)

    seen, row_y, dot_y = [], {}, {}
    for blk in blocks:
        txts = [(a, x) for t, a, x in blk["els"] if t == "text"]
        name = next((a, x) for a, x in txts if a["_x"] == 798)
        short = name[1]
        full = SHORT2FULL[short]
        seen.append(short)
        row_y[short] = name[0]["_y"]
        # ONE-DECIMAL LABELS COME FROM THE GENERATOR'S OWN TABLE ROW, never
        # from re-rounding its two-decimal diagnostic column: 25.65 there
        # rounds to 25.6, while 147/573 = 25.654 prints as 25.7. Sourcing them
        # here also makes Figure 2(c) and Table F2 the same seven numbers.
        la, lb = gen[f"gen_tabc_a_{short}"], gen[f"gen_tabc_b_{short}"]
        # The RATIO, by contrast, must use the two-decimal values. Dividing the
        # one-decimal labels is what shipped 53 for claude-sonnet-5 (11.2/21.3
        # = 0.526) where 11.17/21.29 = 0.5247 gives 52.
        va, vb = gen[f"gen_high_a_{full}"], gen[f"gen_high_b_{full}"]

        check(f"c.{short}.a.label",
              _pct(next(x for a, x in txts if a["_x"] == 636)), la)
        check(f"c.{short}.b.label",
              _pct(next(x for a, x in txts if a["_x"] == 758)), lb)
        ret = next(x for a, x in txts if a["_x"] == 880)
        check(f"c.{short}.ret", int(ret.rstrip("%")), round(100 * vb / va))

        dots = {a["_x"]: a["_y"] for t, a, _ in blk["els"] if t == "circle"}
        check(f"c.{short}.dots", sorted(dots), [644.0, 744.0])
        dot_y[short] = dots[644.0]
        for x, v, rail in ((644.0, la, "a"), (744.0, lb, "b")):
            check(f"c.{short}.{rail}.dot",
                  round(C_V0 - (dots[x] - C_Y0) / C_PX, 2), v, GEOM_TOL)
        # The slope segment restates both endpoints, and the dashed leader
        # restates the right one again on its way to the label. All three must
        # be the same two points or the panel contradicts itself.
        check(f"c.{short}.slope", blk["slope"],
              [(644.0, dots[644.0]), (744.0, dots[744.0])])
        lead = next(a for t, a, _ in blk["els"] if t == "line")
        check(f"c.{short}.leader", (_num(lead["x1"]), _num(lead["y1"])),
              (744.0, dots[744.0]))

    # Zero crossings is a claim about ORDER, and the panel is read top-down in
    # corpus A order, so it holds exactly when corpus B agrees rank for rank.
    order_a = sorted(seen, key=lambda s: -gen[f"gen_high_a_{SHORT2FULL[s]}"])
    order_b = sorted(seen, key=lambda s: -gen[f"gen_high_b_{SHORT2FULL[s]}"])
    by_y = sorted(seen, key=lambda s: row_y[s])
    check("c.label_order", by_y, order_a)
    check("c.zero_crossings", order_a == order_b, gen["gen_order_same_high"])

    # PAINT ORDER IS DELIBERATELY NOT LABEL ORDER, and the exception is load
    # bearing. gemini-3.6-flash and gemini-3.7-flash differ by 0.35 points, so
    # their left-rail dots overlap; whichever is painted last covers the other.
    # Painting the LOWER dot first leaves the upper dot visible and level with
    # the upper label, so dot order and label order agree for the reader.
    # Reverse it and the 19.2% label sits beside the 18.8% model's dot.
    for hi, lo in zip(by_y, by_y[1:]):
        overlap = abs(dot_y[hi] - dot_y[lo]) < 2 * C_DOT_R
        check(f"c.paint.{hi}_before_{lo}",
              seen.index(hi) < seen.index(lo), not overlap)

    hdr = next(x for t, a, x in c_els
               if t == "text" and a["_x"] == 880 and a["_y"] == 24)
    m = re.match(r"^\u03c1 = ([\d.]+) \(high effort\)$", hdr)
    # Spearman, not Pearson, and the HIGH-effort value: the panel draws seven
    # pairs. The body's 0.97 is gen_rho_all over all 19 model-by-effort cells
    # and would be describing a population this panel does not show.
    check("c.hdr.rho", float(m.group(1)), round(gen["gen_rho_high"], 2))
    check("c.hdr.rho.dp", m.group(1), f"{gen['gen_rho_high']:.2f}")

    foot = next(x for t, a, x in c_els if t == "text" and a["_x"] == 742)
    m = re.search(r"\((\u03c1|&rho;) = ([\d.]+)\)", foot)
    if m:
        check("c.foot.rho", float(m.group(2)), round(gen["gen_rho_high"], 2))
    else:
        m_cross = re.search(r"\((\d+) Crossings\)", foot)
        check("c.foot.crossings", int(m_cross.group(1)) if m_cross else -1,
              0 if gen["gen_order_same_high"] else -1)

    return drawn, bad


def main():
    drawn, bad = _compare()
    for k, v in drawn.items():
        print(f"  {k:34} {v}")
    if bad:
        raise SystemExit("\nFIGURE 2 DRIFT\n  " + "\n  ".join(bad))
    print(f"\n  OK: all {len(drawn)} quantities drawn in Figure 2 "
          f"reproduce from their generators.")


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    main()
