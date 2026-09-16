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

"""Splice hand-written prose into a solver-failures document, safely.

The prose slots are the only part of these documents a human types, so they are
the only place an unaudited number can enter. `audit_report.py` pass E12 is the
authority on that rule, but it only runs *after* the text is on disk. This tool
applies the same three rules at authoring time and refuses to write anything at
all if any slot violates one, so a bad number never reaches the file:

  1. every numeral in a slot must also appear in a generated block of the same
     document (integers of ten or less are exempt as structural: ranks, bucket
     counts, "two of nine");
  2. a slot is at least MIN_CHARS long, so the gate cannot be cleared with a
     one-liner;
  3. no bare family word ("the opus model", "the flash model", ...).

Rule 1 is the load-bearing one: to quote a derived figure you must first add it
to `emit_solver_failures.py`, where pass E recomputes it.

Input is a JSON object mapping slot name to prose body, read from a file or
stdin. The document itself remains the only home for the text; the JSON is a
transient carrier.

    python tools/write_prose.py --model opus5 --json prose.json
    python tools/write_prose.py --model opus5 --json - < prose.json
    python tools/write_prose.py --model opus5 --json prose.json --check

The rules are re-implemented here rather than imported: `audit_report.py` is a
script, not a module, and an independent implementation means a bug has to be
made twice to escape.
"""

import argparse
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from emit_solver_failures import (  # noqa: E402
    BLOCKS, MODELS, PROSE, extract, splice,
)

HERE = pathlib.Path(__file__).resolve().parent.parent

MIN_CHARS = 120
NUMTOK = re.compile(r"\d[\d,]*(?:\.\d+)?")
# Bucket ids, block ids, corpus versions, markdown links and anything inside
# backticks are identifiers rather than measurements.
NOISE = re.compile(r"\bF[1-6]\b|\bSF\d+\b|\bv\d+\b|\[[^\]]*\]\([^)]*\)|`[^`]*`")
FAMILY = (r"\bthe opus model\b", r"\bthe flash model\b", r"\bthe sonnet\b",
          r"\bthe pro model\b", r"\bthe gemini model\b")


def sourced(text, k):
    """Every number printed by a generated block of this document."""
    pool = set()
    for tag in BLOCKS:
        body = extract(text, f"{tag}-{k}")
        if body is None:
            raise SystemExit(f"{tag}-{k}: generated block is missing")
        for tok in NUMTOK.findall(body):
            pool.add(tok.replace(",", ""))
    return pool


def problems(slot, body, pool):
    out = []
    if len(body) < MIN_CHARS:
        out.append(f"{slot}: {len(body)} chars, under the {MIN_CHARS} minimum")
    for tok in NUMTOK.findall(NOISE.sub(" ", body)):
        bare = tok.replace(",", "")
        if "." not in bare and int(bare) <= 10:
            continue
        if bare not in pool:
            out.append(f"{slot}: cites {tok!r}, which is in no generated table "
                       f"of this document")
    for pat in FAMILY:
        if re.search(pat, body):
            out.append(f"{slot}: bare family word {pat!r}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--json", required=True,
                    help="file with {slot: prose}, or - for stdin")
    ap.add_argument("--check", action="store_true",
                    help="validate and report, write nothing")
    args = ap.parse_args()

    k = args.model
    path = HERE / f"{MODELS[k][1]}-solver-failures.md"
    text = path.read_text()

    raw = sys.stdin.read() if args.json == "-" else pathlib.Path(args.json).read_text()
    slots = json.loads(raw)

    bad = [s for s in slots if s not in PROSE]
    if bad:
        raise SystemExit(f"unknown prose slot(s): {', '.join(sorted(bad))}")

    pool = sourced(text, k)
    found = []
    for slot in PROSE:
        if slot not in slots:
            continue
        found += problems(slot, " ".join(slots[slot].split()), pool)
    if found:
        print(f"REFUSED: {len(found)} problem(s) in {path.name}")
        for p in found:
            print("  -", p)
        return 1

    for slot in PROSE:
        if slot in slots:
            text = splice(text, f"PROSE-{slot}-{k}", slots[slot].strip())

    n = sum(1 for s in PROSE if s in slots)
    if args.check:
        same = text == path.read_text()
        print(f"{path.name}: {n} slot(s) valid, "
              f"{'unchanged' if same else 'WOULD CHANGE'}")
        return 0 if same else 1
    path.write_text(text)
    empty = [s for s in PROSE if not extract(text, f"PROSE-{s}-{k}")]
    print(f"{path.name}: wrote {n} slot(s); {len(empty)} still empty"
          + (f" ({', '.join(empty)})" if empty else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
