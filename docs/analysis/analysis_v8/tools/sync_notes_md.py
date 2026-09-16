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

"""Keep the markdown mirror's shared prose identical to the rendered report.

Fifteen passages -- the eight curated-round notes, the command-surface intro,
its five bullets and its kicker -- appear in three places: the
`behavior_notes.json` template, the generated HTML, and
`cross-model-behavioral-comparison.md`. They must read identically in the last
two, because they are the same claims about the same corpus.

They stopped being identical, and the failure was silent. The notes were
rewritten against thirteen models at reasoning_effort=high; the markdown was
left at the previous nine-model, medium cut, where it went on asserting things
the data had reversed -- that opus-5 was the only model to score 0.00 on the
expense round when it in fact passes it, that 3.5-flash leads the field's
`asset info`, that search collapses monotonically as capability rises. Nothing
complained, because the only thing enforcing agreement was a byte-comparison
against the template, and the template still matched the stale copy.

Two changes make that failure mode impossible to repeat.

First, the comparison is against the RESOLVED prose
(`cross-model-behavioral-comparison.notes.json`, written by
build_behavior_report.py on every run) rather than against the template. The
template holds `{{tokens}}`, so it can never again be byte-equal to prose --
comparing to it would have "passed" forever while saying nothing.

Second, each passage in the markdown carries an HTML comment naming its notes
key. Comments do not render, so the document is unchanged for a reader, but the
boundaries no longer have to be recovered by searching for the old text -- which
only works while the old text is still there, i.e. exactly until the first time
it matters.

The marker sits at the END of the passage's own line:

    - Some bullet text. <!-- notes:command_surface.bullets.0 prefix="- " -->

and not on lines of its own around the passage. That detail is load-bearing.
These passages are not free-floating paragraphs: the intro and kicker are
blockquotes and the five bullets are list items, so each line opens with `> `
or `- `. A marker on a separate line strands that prefix -- the comment
inherits the blockquote and the prose below it silently stops being a list
item. An inline trailing comment is structurally invisible, and recording the
prefix in the marker lets --write put it back verbatim.

    python tools/sync_notes_md.py --check     # non-zero exit if any drift
    python tools/sync_notes_md.py --write     # rewrite the markdown passages
    python tools/sync_notes_md.py --migrate --from-notes OLD.json
    python tools/sync_notes_md.py --repair    # collapse legacy block markers

`--migrate` is the one-time bootstrap: it finds the passages by their old text,
marks them, and substitutes the resolved text. `--repair` is a second one-time
step that converts the earlier three-line block markers to inline form.
Afterwards only --check and --write are needed.
"""

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MD = os.path.join(ROOT, "cross-model-behavioral-comparison.md")
RESOLVED = os.path.join(ROOT, "cross-model-behavioral-comparison.notes.json")
BLOCKS = os.path.join(ROOT, "cross-model-behavioral-comparison.blocks.json")

_KEY = r"[A-Za-z0-9_.]+"
INLINE = re.compile(r'[ \t]*<!--\s*notes:(?P<key>' + _KEY + r')'
                    r'(?:\s+prefix="(?P<prefix>[^"]*)")?\s*-->[ \t]*$')
# Generated multi-line blocks -- currently the command-surface table. These DO
# get markers on lines of their own, and that is safe for the reason the prose
# markers are not: a table is a top-level construct with no `> ` or `- ` prefix
# to strand, and an HTML comment closes its own block on the line holding
# `-->`, so the table below it still parses as a table.
BLOCK_O = re.compile(r'^<!--\s*block:(?P<key>' + _KEY + r')\s*-->[ \t]*$')
BLOCK_C = re.compile(r'^<!--\s*/block:(?P<key>' + _KEY + r')\s*-->[ \t]*$')
# The superseded three-line form, recognised only so --repair can migrate off it.
BLOCK_OPEN = re.compile(r'^(?P<prefix>.*?)<!--\s*notes:(?P<key>' + _KEY
                        + r')\s*-->[ \t]*$')
BLOCK_CLOSE = re.compile(r'^<!--\s*/notes:(?P<key>' + _KEY + r')\s*-->[ \t]*$')

# A passage may be indented, quoted and/or bulleted; it may not be prefixed by
# anything that carries meaning of its own, because --write would reproduce it
# blindly onto text that no longer says the same thing.
STRUCTURAL = re.compile(r'^(?:\s*>\s?)*(?:\s*(?:[-*+]|\d+\.)\s+)?$')


def marker(key, prefix):
    if prefix:
        return f'<!-- notes:{key} prefix="{prefix}" -->'
    return f"<!-- notes:{key} -->"


SKIP = {"_comment"}


def slots(notes):
    """(key, text) for every passage the markdown mirrors, in document order.

    A generic walk, matching the one resolve_notes() uses on the template. The
    enumeration used to be hand-written -- scenarios, then the intro, then the
    bullets, then the kicker -- which meant a new prose section was invisible
    here: the build would resolve its tokens and render it into the HTML, and
    --check would report everything in sync while the markdown carried no copy
    of it at all. Anything that is a string leaf is a passage.
    """
    out = []

    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                if not path and k in SKIP:
                    continue
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}.{i}")
        elif isinstance(node, str):
            out.append((path, node))

    walk(notes, "")
    return out


def load_resolved():
    if not os.path.exists(RESOLVED):
        sys.exit(f"{RESOLVED} not found -- run build_behavior_report.py first")
    with open(RESOLVED) as f:
        notes = json.load(f)
    bad = [k for k, v in slots(notes) if "{{" in v]
    if bad:
        sys.exit("resolved notes still contain tokens: " + ", ".join(bad))
    multi = [k for k, v in slots(notes) if "\n" in v]
    if multi:
        sys.exit("passages must be single-line: " + ", ".join(multi))
    return notes


def load_blocks():
    if not os.path.exists(BLOCKS):
        sys.exit(f"{BLOCKS} not found -- run build_behavior_report.py first")
    with open(BLOCKS) as f:
        return json.load(f)


def block_span(lines, key):
    """(open_idx, close_idx) for a generated block, or None if unmarked."""
    o = [i for i, ln in enumerate(lines)
         if (m := BLOCK_O.match(ln)) and m.group("key") == key]
    if not o:
        return None
    if len(o) > 1:
        sys.exit(f"block marker {key} opens {len(o)} times")
    c = [i for i, ln in enumerate(lines)
         if i > o[0] and (m := BLOCK_C.match(ln)) and m.group("key") == key]
    if not c:
        sys.exit(f"block {key} is opened but never closed")
    return o[0], c[0]


def check_blocks(lines, blocks, quiet=False):
    bad = []
    for key, want in blocks.items():
        span = block_span(lines, key)
        if span is None:
            bad.append((key, "no block marker in markdown"))
        elif "\n".join(lines[span[0] + 1:span[1]]) != want:
            bad.append((key, "block differs from the generated table"))
    if bad and not quiet:
        print(f"{len(bad)} of {len(blocks)} generated blocks out of sync:")
        for key, why in bad:
            print(f"  {key}: {why}")
    return bad


def write_blocks(lines, blocks):
    for key, want in blocks.items():
        span = block_span(lines, key)
        if span is None:
            sys.exit(f"no block marker for {key}; add "
                     f"<!-- block:{key} --> / <!-- /block:{key} --> to the "
                     "markdown around the region it replaces")
        o, c = span
        lines = lines[:o + 1] + want.split("\n") + lines[c:]
    return lines


def locate(lines, key):
    """Index of the line carrying key's inline marker, or None."""
    hits = []
    for i, ln in enumerate(lines):
        m = INLINE.search(ln)
        if m and m.group("key") == key:
            hits.append(i)
    if not hits:
        return None
    if len(hits) > 1:
        sys.exit(f"marker for {key} appears on {len(hits)} lines")
    return hits[0]


def current(lines, key):
    """(prefix, text) for key, or None if unmarked. Exits on a broken marker."""
    i = locate(lines, key)
    if i is None:
        return None
    ln = lines[i]
    m = INLINE.search(ln)
    if m is None:                             # locate() only returns matches
        sys.exit(f"{key}: marker vanished between passes")
    head, prefix = ln[:m.start()], m.group("prefix") or ""
    if not head.startswith(prefix):
        sys.exit(f"{key}: line does not open with its recorded prefix "
                 f"{prefix!r}")
    return prefix, head[len(prefix):]


def do_check(lines, notes, quiet=False):
    bad = []
    for key, want in slots(notes):
        got = current(lines, key)
        if got is None:
            bad.append((key, "no marker in markdown"))
        elif got[1] != want:
            bad.append((key, "text differs from the rendered report"))
    if bad and not quiet:
        print(f"{len(bad)} of {len(slots(notes))} passages out of sync:")
        for key, why in bad:
            print(f"  {key}: {why}")
    return bad


def do_write(lines, notes):
    lines = list(lines)
    for key, want in slots(notes):
        i = locate(lines, key)
        if i is None:
            sys.exit(f"no marker for {key}; run --migrate first")
        got = current(lines, key)
        assert got is not None                # locate() found the marker
        prefix = got[0]
        lines[i] = prefix + want + " " + marker(key, prefix)
    return lines


def _place(lines, key, want, needle):
    """Replace the unique line containing `needle` with a marked `want`."""
    hits = [i for i, ln in enumerate(lines) if needle in ln]
    if len(hits) != 1:
        sys.exit(f"old text for {key} matches {len(hits)} lines; "
                 "cannot place a marker unambiguously")
    i = hits[0]
    ln = lines[i]
    if ln.count(needle) != 1:
        sys.exit(f"old text for {key} appears twice on line {i + 1}")
    at = ln.index(needle)
    prefix, tail = ln[:at], ln[at + len(needle):]
    if tail.strip():
        sys.exit(f"{key}: trailing text after the passage on line {i + 1}: "
                 f"{tail.strip()[:60]!r}")
    if not STRUCTURAL.match(prefix):
        sys.exit(f"{key}: unexpected prefix on line {i + 1}: {prefix!r}")
    lines[i] = prefix + want + " " + marker(key, prefix)
    return lines


def do_migrate(lines, notes, old_path):
    """Bootstrap: locate each passage by its pre-rewrite text, mark, replace."""
    with open(old_path) as f:
        old = dict(slots(json.load(f)))
    missing = [k for k, _ in slots(notes) if k not in old]
    if missing:
        sys.exit("old notes lack these keys: " + ", ".join(missing))
    lines = list(lines)
    for key, want in slots(notes):
        if locate(lines, key) is not None:
            continue                          # already marked
        lines = _place(lines, key, want, old[key])
    return lines


def do_repair(lines, notes):
    """Collapse the legacy three-line block form into an inline marker.

    The block form put the open marker on its own line, which left any `> ` or
    `- ` prefix stranded above the passage. The prefix is still sitting on that
    line, so the damage is undone exactly rather than guessed at.
    """
    want = dict(slots(notes))
    out, i, fixed = [], 0, 0
    while i < len(lines):
        m = BLOCK_OPEN.match(lines[i])
        close = (BLOCK_CLOSE.match(lines[i + 2])
                 if m and i + 2 < len(lines) else None)
        if not (m and close and close.group("key") == m.group("key")):
            out.append(lines[i])
            i += 1
            continue
        key, prefix = m.group("key"), m.group("prefix")
        if key not in want:
            sys.exit(f"block marker {key} is not a known passage")
        if not STRUCTURAL.match(prefix):
            sys.exit(f"{key}: unexpected stranded prefix {prefix!r}")
        if lines[i + 1] != want[key]:
            sys.exit(f"{key}: block body does not match the rendered report; "
                     "repair by hand")
        out.append(prefix + want[key] + " " + marker(key, prefix))
        i += 3
        fixed += 1
    print(f"collapsed {fixed} block markers")
    return out


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--write", action="store_true")
    g.add_argument("--migrate", action="store_true")
    g.add_argument("--repair", action="store_true")
    ap.add_argument("--from-notes", help="pre-rewrite behavior_notes.json")
    ap.add_argument("--md", default=MD)
    args = ap.parse_args()

    notes = load_resolved()
    blocks = load_blocks()
    with open(args.md) as f:
        lines = f.read().split("\n")

    if args.check:
        bad = do_check(lines, notes) + check_blocks(lines, blocks)
        if bad:
            sys.exit(1)
        print(f"all {len(slots(notes))} shared passages and {len(blocks)} "
              "generated blocks match the report")
        return

    if args.migrate:
        if not args.from_notes:
            sys.exit("--migrate needs --from-notes")
        out = do_migrate(lines, notes, args.from_notes)
    elif args.repair:
        out = do_repair(lines, notes)
    else:
        out = write_blocks(do_write(lines, notes), blocks)

    if out == lines:
        print("markdown already up to date")
        return
    with open(args.md, "w") as f:
        f.write("\n".join(out))
    left = do_check(out, notes, quiet=True) + check_blocks(out, blocks, True)
    if left:
        sys.exit(f"wrote {args.md} but {len(left)} passages still differ")
    print(f"wrote {args.md} ({len(slots(notes))} passages, {len(blocks)} "
          "blocks in sync)")


if __name__ == "__main__":
    main()
