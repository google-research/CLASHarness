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

"""WCAG 2.1 AA contrast gate for the generated behaviour-report HTML.

Two passes:

1. **DOM text.** Walks every text node, resolves the effective foreground
   (nearest ancestor declaring ``color``) and background (nearest ancestor
   declaring a solid ``background``/``background-color``), then checks the
   contrast ratio against the AA threshold for the node's computed font size
   (4.5:1 normal, 3:1 for large text).

2. **SVG text.** The report draws its panels as inline SVG, where colour comes
   from ``fill=`` attributes that CSS never sees. Those are checked against the
   panel fills the report actually paints, plus an explicit table of captions
   that are deliberately drawn *on top of* a coloured data mark.

The CSS subset modelled is the one the report emits: a flat stylesheet of
simple and descendant selectors plus inline ``style=`` attributes. Descendant
combinators are honoured; ``>`` is relaxed to a descendant match, which is safe
here because the report never uses child-vs-descendant to disambiguate two
conflicting rules.

Usage::

    python tools/audit_contrast.py cross-model-behavioral-comparison.html

Exits non-zero if anything is below AA.
"""

from __future__ import annotations

import collections
import re
import sys
from html.parser import HTMLParser

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

# Backgrounds the report paints behind SVG text. Every SVG text node must clear
# AA against ALL of these unless it is listed in SVG_ON_MARK below.
SVG_PANEL_BACKGROUNDS = ("#ffffff", "#fbfcfe", "#f8fafc")

# Captions deliberately drawn inside a coloured data mark, so the panel
# background does not apply. Keyed by the text fill; the value is the solid
# colour of the mark behind it. Marks must be fully opaque -- a translucent
# mark blends toward the track and silently loses contrast.
SVG_ON_MARK = {
    "#ffffff": "#dc2626",  # in-bar "x% -> y% capped" caption on the capped bar
}

# --------------------------------------------------------------------------
# colour helpers
# --------------------------------------------------------------------------

RGB = tuple[int, int, int]

_NAMED: dict[str, RGB | None] = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "red": (255, 0, 0),
    "transparent": None,
    "none": None,
    "inherit": None,
    "currentcolor": None,
}


def parse_color(raw: str) -> RGB | None:
    """Return an (r, g, b) tuple, or None when the value is not a solid colour."""
    if not raw:
        return None
    v = raw.strip().lower()
    if v in _NAMED:
        return _NAMED[v]
    m = re.fullmatch(r"#([0-9a-f]{3})", v)
    if m:
        h = m.group(1)
        return (int(h[0] * 2, 16), int(h[1] * 2, 16), int(h[2] * 2, 16))
    m = re.fullmatch(r"#([0-9a-f]{6})", v)
    if m:
        h = m.group(1)
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    m = re.fullmatch(r"rgba?\(([^)]*)\)", v)
    if m:
        parts = [p.strip() for p in m.group(1).replace("/", ",").split(",")]
        if len(parts) >= 3:
            try:
                r, g, b = (int(round(float(p.rstrip("%")))) for p in parts[:3])
            except ValueError:
                return None
            if len(parts) == 4:
                try:
                    if float(parts[3]) < 0.999:
                        return None  # translucent: skip rather than guess
                except ValueError:
                    return None
            return (r, g, b)
    return None


def luminance(rgb) -> float:
    def chan(c):
        s = c / 255.0
        return s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4

    r, g, b = (chan(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(fg, bg) -> float:
    lf, lb = luminance(fg), luminance(bg)
    hi, lo = max(lf, lb), min(lf, lb)
    return (hi + 0.05) / (lo + 0.05)


def threshold(size_px: float, weight: int) -> float:
    """AA minimum for the given computed type size."""
    large = size_px >= 24 or (size_px >= 18.66 and weight >= 700)
    return 3.0 if large else 4.5


# --------------------------------------------------------------------------
# minimal stylesheet model
# --------------------------------------------------------------------------

# A compound is one selector segment, e.g. "td.exn" -> ("td", ("exn",)).
Rule = collections.namedtuple("Rule", "parts spec decls raw")


def _compound(seg: str):
    seg = re.sub(r":[a-z-]+(\([^)]*\))?", "", seg).strip()
    if not seg:
        return None
    tag_m = re.match(r"^([a-z0-9]+)", seg)
    tag = tag_m.group(1) if tag_m else None
    return (tag, tuple(re.findall(r"\.([A-Za-z0-9_-]+)", seg)))


def parse_stylesheet(css: str):
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    css = re.sub(r"@[a-z-]+[^{]*\{(?:[^{}]|\{[^{}]*\})*\}", "", css, flags=re.S)
    rules = []
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        sel_blob, body = m.group(1), m.group(2)
        decls = {}
        for d in body.split(";"):
            if ":" not in d:
                continue
            k, _, v = d.partition(":")
            decls[k.strip().lower()] = v.strip().rstrip("!important").strip()
        if not decls:
            continue
        for sel in sel_blob.split(","):
            sel = sel.strip()
            if not sel or "::" in sel:
                continue
            segs = [s for s in sel.replace(">", " ").split() if s.strip()]
            raw_parts = [_compound(s) for s in segs]
            if not raw_parts or any(p is None for p in raw_parts):
                continue
            parts = [p for p in raw_parts if p is not None]
            ntag = sum(1 for t, _ in parts if t)
            ncls = sum(len(c) for _, c in parts)
            rules.append(Rule(tuple(parts), (ncls, ntag, len(parts)), decls, sel))
    return rules


def _compound_matches(part, tag: str, classes: set) -> bool:
    ptag, pcls = part
    if ptag and ptag != tag:
        return False
    return all(c in classes for c in pcls)


def matches(rule: Rule, stack) -> bool:
    """rule.parts must appear in order along the ancestor path, with the final
    compound matching the element itself."""
    if not _compound_matches(rule.parts[-1], stack[-1][0], stack[-1][1]):
        return False
    remaining = list(rule.parts[:-1])
    if not remaining:
        return True
    for anc in reversed(stack[:-1]):
        if _compound_matches(remaining[-1], anc[0], anc[1]):
            remaining.pop()
            if not remaining:
                return True
    return False


# --------------------------------------------------------------------------
# DOM pass
# --------------------------------------------------------------------------

VOID = {"br", "hr", "img", "input", "meta", "link", "source", "col"}
SKIP = {"script", "style", "head", "title"}


class DomScan(HTMLParser):
    def __init__(self, rules):
        super().__init__(convert_charrefs=True)
        self.rules = rules
        self.stack = [("html", set(), (17, 24, 39), (255, 255, 255), 14.0, 400, "html")]
        self.findings = []
        self.total = 0
        self.skipping = 0

    def _resolve(self, tag, attrs):
        d = dict(attrs)
        classes = set((d.get("class") or "").split())
        parent = self.stack[-1]
        color, bg, size, weight = parent[2], parent[3], parent[4], parent[5]

        probe = self.stack + [(tag, classes)]
        hits = sorted((r for r in self.rules if matches(r, probe)), key=lambda r: r.spec)
        decls = {}
        for r in hits:
            decls.update(r.decls)
        for k, v in re.findall(r"([a-z-]+)\s*:\s*([^;]+)", d.get("style", "").lower()):
            decls[k.strip()] = v.strip()

        if "color" in decls:
            c = parse_color(decls["color"])
            if c:
                color = c
        for key in ("background-color", "background"):
            if key in decls:
                c = parse_color(decls[key].split()[0] if decls[key] else "")
                if c:
                    bg = c
                    break
        if "font-size" in decls:
            m = re.match(r"([\d.]+)px", decls["font-size"])
            if m:
                size = float(m.group(1))
        if "font" in decls:
            m = re.search(r"([\d.]+)px", decls["font"])
            if m:
                size = float(m.group(1))
        if "font-weight" in decls:
            w = decls["font-weight"].strip()
            if w.isdigit():
                weight = int(w)
            elif w == "bold":
                weight = 700
            elif w == "normal":
                weight = 400
        if tag in ("b", "strong", "th"):
            weight = max(weight, 700)

        sig = "." + ".".join(sorted(classes)) if classes else ""
        return (tag, classes, color, bg, size, weight, f"{tag}{sig}")

    def handle_starttag(self, tag, attrs):
        if tag in SKIP:
            self.skipping += 1
            return
        if tag in VOID:
            return
        self.stack.append(self._resolve(tag, attrs))

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        if tag in SKIP:
            self.skipping = max(0, self.skipping - 1)
            return
        if tag in VOID:
            return
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break

    def handle_data(self, data):
        if self.skipping or not data.strip():
            return
        _, _, color, bg, size, weight, path = self.stack[-1]
        self.total += 1
        ratio = contrast(color, bg)
        need = threshold(size, weight)
        if ratio < need:
            self.findings.append(
                (path, "#%02x%02x%02x" % color, "#%02x%02x%02x" % bg,
                 round(ratio, 2), size, need, data.strip()[:40])
            )


# --------------------------------------------------------------------------
# SVG pass
# --------------------------------------------------------------------------

def _num(attrs: str, name: str):
    m = re.search(name + r'="(-?[\d.]+)"', attrs)
    return float(m.group(1)) if m else None


def translucent_overlaps(html: str):
    """Find <text> nodes painted on top of a translucent <rect>.

    A translucent mark blends with whatever is behind it, so neither the panel
    background nor the mark's nominal fill is the true background for text
    sitting on it, and SVG_ON_MARK cannot describe it. Rather than warn about
    every translucent mark forever, test geometrically whether any text
    actually lands on one; only those are unresolvable and must be fixed
    (either make the mark opaque or move the label off it).

    Geometry is compared within a single <svg> so the coordinate systems match.
    The vertical band is widened upward by one em-ish ascent because an SVG
    text ``y`` is a baseline, not a top edge.
    """
    overlaps, marks = [], []
    for sm in re.finditer(r"<svg\b.*?</svg>", html, flags=re.S):
        svg = sm.group(0)
        rects = []
        for rm in re.finditer(r"<rect\b([^>]*)>", svg):
            a = rm.group(1)
            om = re.search(r'\bopacity="([\d.]+)"', a)
            if not om or float(om.group(1)) >= 0.999:
                continue
            x, y = _num(a, "x"), _num(a, "y")
            w, h = _num(a, "width"), _num(a, "height")
            fm = re.search(r'fill="([^"]+)"', a)
            desc = f"{fm.group(1) if fm else '?'}@{om.group(1)}"
            marks.append(rm.group(0)[:90])
            if None not in (x, y, w, h):
                rects.append((x, y, w, h, desc))
        if not rects:
            continue
        for tm in re.finditer(r"<text\b([^>]*)>(.*?)</text>", svg, flags=re.S):
            a, body = tm.group(1), re.sub(r"<[^>]+>", "", tm.group(2)).strip()
            if not body:
                continue
            tx, ty = _num(a, "x"), _num(a, "y")
            if tx is None or ty is None:
                continue
            asc = (_num(a, "font-size") or 12.0)
            for (x, y, w, h, desc) in rects:
                if x - 2 <= tx <= x + w + 2 and y - asc <= ty <= y + h + 3:
                    overlaps.append((body, tx, ty, desc))
    return overlaps, marks


def scan_svg(html: str):
    """Check every inline-SVG <text> node's fill against the panel backgrounds."""
    findings, total = [], 0

    for m in re.finditer(r"<text\b([^>]*)>(.*?)</text>", html, flags=re.S):
        attrs, body = m.group(1), re.sub(r"<[^>]+>", "", m.group(2)).strip()
        if not body:
            continue
        total += 1
        fm = re.search(r'fill="([^"]+)"', attrs)
        sm = re.search(r'font-size="([\d.]+)"', attrs)
        wm = re.search(r'font-weight="(\d+)"', attrs)
        fill = fm.group(1) if fm else "#000000"
        size = float(sm.group(1)) if sm else 12.0
        weight = int(wm.group(1)) if wm else 400
        fg = parse_color(fill)
        if fg is None:
            continue
        need = threshold(size, weight)
        if fill.lower() in SVG_ON_MARK:
            backgrounds = (SVG_ON_MARK[fill.lower()],)
        else:
            backgrounds = SVG_PANEL_BACKGROUNDS
        for bgs in backgrounds:
            ratio = contrast(fg, parse_color(bgs))
            if ratio < need:
                findings.append((fill, bgs, round(ratio, 2), size, need, body[:40]))
    return findings, total


# --------------------------------------------------------------------------

def main(path: str) -> int:
    html = open(path, encoding="utf-8").read()
    css = "\n".join(re.findall(r"<style[^>]*>(.*?)</style>", html, flags=re.S))
    rules = parse_stylesheet(css)

    dom = DomScan(rules)
    dom.feed(html)
    svg_bad, svg_total = scan_svg(html)

    print(f"stylesheet rules parsed : {len(rules)}")
    print(f"DOM text nodes checked  : {dom.total}")
    print(f"SVG text nodes checked  : {svg_total}")
    print()

    if dom.findings:
        agg = collections.Counter()
        sample = {}
        for path_, fg, bg, ratio, size, need, txt in dom.findings:
            key = (path_, fg, bg, ratio, size, need)
            agg[key] += 1
            sample.setdefault(key, txt)
        print(f"DOM below AA: {len(dom.findings)} nodes")
        print(f"{'count':>6}  {'ratio':>5}  {'need':>4}  {'px':>5}  {'fg':<9}{'bg':<9} selector")
        for key, n in agg.most_common():
            p, fg, bg, ratio, size, need = key
            print(f"{n:>6}  {ratio:>5.2f}  {need:>4.1f}  {size:>5.1f}  {fg:<9}{bg:<9} "
                  f"{p}   e.g. {sample[key]!r}")
    else:
        print("DOM below AA: 0")

    print()
    if svg_bad:
        print(f"SVG below AA: {len(svg_bad)} nodes")
        for fill, bg, ratio, size, need, txt in svg_bad[:40]:
            print(f"  {ratio:>5.2f} (need {need}) {fill} on {bg} @{size}px  {txt!r}")
    else:
        print("SVG below AA: 0")

    overlaps, marks = translucent_overlaps(html)
    print()
    if overlaps:
        print(f"Text on translucent marks: {len(overlaps)} (UNRESOLVABLE "
              f"background -- make the mark opaque or move the label off it)")
        for body, tx, ty, desc in overlaps[:20]:
            print(f"   {body!r} at ({tx},{ty}) over {desc}")
    else:
        print(f"Text on translucent marks: 0 "
              f"({len(marks)} translucent marks, none carry a label)")

    fails = len(dom.findings) + len(svg_bad) + len(overlaps)
    print()
    print("RESULT: PASS -- everything clears WCAG AA" if fails == 0
          else f"RESULT: FAIL -- {fails} node(s) below AA")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "cross-model-behavioral-comparison.html"
    sys.exit(main(target))
