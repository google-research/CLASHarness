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

"""Emit Appendix A (60 scenarios x 9 models) into eval-results-capability-analysis.md.

The table is *generated*, never hand-written, so it cannot drift from the data.
It is injected between the two HTML-comment sentinels

    <!-- APPENDIX-A-BEGIN -->
    <!-- APPENDIX-A-END -->

and `tools/audit_report.py` re-parses the emitted rows back out of the markdown
and recomputes every cell against `failure_taxonomy.json`. Running this script
twice in a row is a no-op (idempotent).

Usage:
    python tools/emit_appendix_a.py            # inject (in place)
    python tools/emit_appendix_a.py --check    # verify in-place table is current
    python tools/emit_appendix_a.py --stdout   # print the table only
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

TAXONOMY = os.path.join(ROOT, "failure_taxonomy.json")
DOC = os.path.join(ROOT, "eval-results-capability-analysis.md")

BEGIN = "<!-- APPENDIX-A-BEGIN -->"
END = "<!-- APPENDIX-A-END -->"

# (json key, column header). Order is the ladder order used everywhere else in
# the v8 documents: Gemini, Claude, OpenAI.
COLUMNS = [
    ("flash", "3.5-flash"),
    ("flash36", "3.6-flash"),
    ("pro", "3.1-pro"),
    ("opus", "opus-4.8"),
    ("opus5", "opus-5"),
    ("sonnet", "sonnet-5"),
    ("sol", "sol"),
    ("luna", "luna"),
    ("terra", "terra"),
]

EFFORT = "medium"


def build_rows(taxonomy):
    """Return [(scenario, field_mean, [(mean, passed), ...])] sorted hardest first."""
    models = taxonomy["cells"][EFFORT]["models"]
    missing = [k for k, _ in COLUMNS if k not in models]
    if missing:
        raise SystemExit(f"failure_taxonomy.json is missing model keys: {missing}")

    scenarios = sorted(models[COLUMNS[0][0]]["scenarios"])
    if len(scenarios) != 60:
        raise SystemExit(f"expected 60 scenarios, found {len(scenarios)}")

    rows = []
    for slug in scenarios:
        cells = []
        for key, _hdr in COLUMNS:
            per = models[key]["scenarios"].get(slug)
            if per is None:
                raise SystemExit(f"{key} has no record for scenario {slug}")
            cells.append((per["mean_score"], per["passed"]))
        field = sum(c[0] for c in cells) / len(cells)
        rows.append((slug, field, cells))
    rows.sort(key=lambda r: (r[1], r[0]))
    return rows


def render(rows):
    out = []
    out.append("| scenario | field | " + " | ".join(h for _, h in COLUMNS) + " |")
    out.append("|---|---:|" + "---:|" * len(COLUMNS))
    for slug, field, cells in rows:
        rendered = []
        for mean, passed in cells:
            # Bold marks "this model passed this scenario at least once".
            body = f"**{mean:.2f}**" if passed else f"{mean:.2f}"
            rendered.append(f"{body}<br>{passed}")
        out.append(f"| `{slug}` | {field:.3f} | " + " | ".join(rendered) + " |")
    return "\n".join(out)


def splice(doc_text, table):
    if BEGIN not in doc_text or END not in doc_text:
        raise SystemExit(f"sentinels {BEGIN} / {END} not found in {DOC}")
    head, rest = doc_text.split(BEGIN, 1)
    _stale, tail = rest.split(END, 1)
    return f"{head}{BEGIN}\n\n{table}\n\n{END}{tail}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify without writing")
    ap.add_argument("--stdout", action="store_true", help="print table only")
    args = ap.parse_args()

    with open(TAXONOMY) as fh:
        taxonomy = json.load(fh)
    table = render(build_rows(taxonomy))

    if args.stdout:
        print(table)
        return 0

    with open(DOC) as fh:
        doc = fh.read()
    updated = splice(doc, table)

    if args.check:
        if updated != doc:
            print("STALE: Appendix A does not match failure_taxonomy.json.")
            print("       Run: python tools/emit_appendix_a.py")
            return 1
        print(f"OK: Appendix A is current ({len(table.splitlines()) - 2} scenario rows).")
        return 0

    if updated == doc:
        print(f"unchanged: Appendix A already current "
              f"({len(table.splitlines()) - 2} scenario rows).")
        return 0
    with open(DOC, "w") as fh:
        fh.write(updated)
    print(f"wrote Appendix A ({len(table.splitlines()) - 2} scenario rows) -> {DOC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
