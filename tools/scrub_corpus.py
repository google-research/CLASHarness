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

"""Redact infrastructure identifiers from the corpus before publishing it.

WHY THIS EXISTS. The evaluation trees were produced on internal infrastructure,
and a handful of episodes failed with runner or transport errors whose messages
embed that infrastructure: cluster job addresses, workspace paths, and the
operator's account name. The messages were captured verbatim into the episode
record, so they travel with the corpus unless something removes them.

WHAT IS AND IS NOT TOUCHED. An audit of all 893,951 records in every
published component found every such identifier in exactly one place: the
``error`` string of episodes that already failed (``success: false``,
``score: 0.0``, empty ``final_response``). 524 episodes have a non-empty
``error`` and they take six distinct shapes. No successful episode carries an
identifier, and neither do ``trajectory``, ``input``, ``metadata`` or any score
field. Redaction therefore cannot move a number -- it rewrites failure
diagnostics on runs that were already recorded as failures, and the
``error_type`` field that analysis actually reads is a separate field and is
left alone. `--check` re-establishes that claim on whatever tree you point it
at, so it is worth running before and after.

Two of those six shapes -- 31 episodes -- are transport errors carrying tens of
kilobytes of serialised internal stack detail. Most of it cannot be matched
corpus-wide without rewriting scenario prose, so it is handled by a rule scoped
to the ``error`` field which drops the structured body and keeps the leading
sentence: ``504 DEADLINE_EXCEEDED. <internal-diagnostic-trimmed>``. The other
four shapes -- rate limits, context-length errors, a timeout -- carry nothing
internal and survive intact.

WHAT IS COPIED. Only the component directories listed in the download_data.py
manifest, which is imported rather than restated so the staging tree and the
fetch layout stay in step. The scoping is a safety property, not a convenience:
the corpus root also holds superseded eval trees, run provenance, staging
scratch and a `logs/` sibling beside each published `tasks/`, and those carry
far more operator paths than the published trees do. Redacting and publishing
them would be strictly worse than leaving them behind. Point `--src` at the
corpus root and the tool takes only the subtrees it should; pass `--whole-tree`
only for a corpus you generated yourself, which has no manifest layout.
One file is dropped by name inside those directories: ``visualize.html``, a
browser view of the JSON published beside it. It is 9.0 GB across 200 files,
nothing in this repository reads it, and it is a second copy of every string in
the file it renders. See the EXCLUDE_FILES comment below.

Legacy environment names in unreleased or historical run artifacts are
optionally normalized with ``--rename-brand``, rewriting historical name
variants to OpenClaw. This is off by default to preserve exact hash parity
with reference evaluations.

USAGE
    python tools/scrub_corpus.py --src <corpus-root> --check
    python tools/scrub_corpus.py --src <corpus-root> --dst <staging> --apply
    python tools/scrub_corpus.py --src <corpus-root> --dst <staging> --apply \
        --component eval-blind
    python tools/scrub_corpus.py --src <my-own-corpus> --check --whole-tree
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from download_data import COMPONENTS  # noqa: E402

# The directory prefixes that are actually published, read out of the
# download_data.py manifest so the staging tree and the fetch layout cannot
# drift apart.
#
# This is a allowlist rather than a walk of the corpus root, and the difference
# matters. The root also holds superseded eval trees (data_3_6_flash and v3..v7,
# _jspace), run provenance, staging scratch, and a `logs/` and `datasets/`
# sibling beside each published `tasks/`. None of that is part of the release
# and several of those trees are dense with operator paths, so walking the root
# and trusting redaction to clean up would publish them redacted instead of not
# publishing them at all.
PUBLISHED = {k: dest for k, (dest, _size, _why) in COMPONENTS.items()}

# Second line of defence, applied inside a published prefix. Matched as a
# substring rather than by equality: the corpus also contains
# `_staging_minimax_m3`, which an equality test lets straight through while
# looking like it excludes minimax_m3.
EXCLUDE_DIRS = ("minimax_m3",)

# Files excluded by name inside a published prefix. `visualize.html` is a
# self-contained browser view of the JSON that sits next to it, and there are
# two of them: 80 files / 4.1 GB in the eval trees rendering the episode
# records, and 120 files / 4.9 GB in the scenario trees rendering scenario.json.
# 9.0 GB in total, a fifth of the release.
#
# Both are dropped. Neither is read by anything in analysis/, clasharness/ or
# tools/ -- the string "visualize" does not occur in this repository outside
# this comment -- both are byte-derived from a file published beside them, and
# both give every string in that file a second place to hide. The scenario one
# additionally loads Tailwind, Mermaid and svg-pan-zoom from public CDNs, so it
# does not render offline and would put a third-party fetch in the middle of
# someone inspecting the dataset.
#
# The generator for these is not part of this release, so dropping them is not
# reversible from the published artifacts. Say so in the dataset card.
EXCLUDE_FILES = ("visualize.html",)

# Ordered: the path rules must run before the bare-identifier rules, or the
# identifier rule rewrites a fragment of a path the path rule was about to
# replace wholesale.
# Every rule is case-insensitive on purpose: DETECT is, and a rule that is not
# produces the worst possible outcome -- --check reports a leak that --apply
# believes it already fixed. "Uniserve" next to "uniserve" is exactly that case.
_DOM = "google" + ".com"
_G3 = "google" + "3"
_R = [
    # Cluster job addresses, e.g. /bns/rt/borg/rt/bns/uniserve-...-prod-jobs/...
    (r"/bns/[\w./:@+-]*", "<cluster-address-redacted>"),
    # Source-control workspace paths, e.g. /google/src/cloud/<user>/.../defaults
    (r"/google/src/[\w./:@+-]*", "<workspace-path-redacted>"),
    # Batch-scheduler home directories, e.g. /home/<user>_google_com/deploy/...
    (r"/home/[\w.-]*_google_com[\w./:@+-]*", "<home-path-redacted>"),
    (r"/usr/local/google[\w./:@+-]*", "<home-path-redacted>"),
    # Account names. `_` is a word character, so \btestoperator\b does NOT match
    # inside "testoperator_google_com"; the compound form needs its own rule and
    # has to run first, or the bare rule leaves "_google_com" behind.
    (r"[\w.-]*_google_com\b", "<operator-redacted>"),
    (rf"[\w.+-]+@{_DOM}", "<email-redacted>"),
    (r"\btestoperator\b", "<operator-redacted>"),
    # Production hosts, cluster service targets and internal protobuf type
    # URLs. Unambiguous enough to be corpus-wide: none of them can occur in
    # scenario prose by accident.
    (rf"[\w-]+\.prod\.{_DOM}", "<host-redacted>"),
    (r"blade:[\w.-]+", "<service-target-redacted>"),
    (r"type\.googleapis\.com/(?!google\.rpc\b)[\w.]+",
     "<internal-proto-redacted>"),
    # Bare identifiers that survive outside any path.
    (r"\buniserve[\w.-]*", "<service-redacted>"),
    (r"\bborg\b", "<scheduler-redacted>"),
    (rf"\b{_G3}\b", "<monorepo-redacted>"),
    (r"\bopenclaw_env\b", "clasharness"),
]
REDACTIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(p, re.I), s) for p, s in _R]

# Detection is deliberately broader than redaction: --check must fail on a
# pattern even where no rewrite rule would fire, rather than quietly passing.
DETECT = re.compile(
    rf"/bns/|\bborg\b|\buniserve|/google/src/|\b{_G3}\b|/usr/local/google"
    rf"|@{_DOM}|\btestoperator\b|_google_com|\bopenclaw_env\b"
    rf"|\.prod\.{_DOM}|blade:[\w.-]"
    r"|type\.googleapis\.com/(?!google\.rpc\b)", re.I)

# Field-scoped rules for `.error`, which is the only place the corpus carries
# machine-generated diagnostics rather than authored text.
#
# 31 of the 524 failed episodes carry a transport error whose body is a few
# tens of kilobytes of internal stack detail: model revision strings, service
# job names, monorepo source paths, internal protobuf type URLs and debug
# links. Most of that cannot be matched corpus-wide without hitting scenario
# prose -- "groot", "orcas" and "go/" are all things a scenario may legitimately
# say. So the payload is detected inside `.error` only, and the whole structured
# body is dropped, keeping the leading sentence that names the failure.
#
# Dropping it costs nothing analytically: analysis/ reads `bool(row["error"])`
# and the separate `error_type` field, never the text. See analysis/corpus.py
# and analysis/generator_ablation.py.
_ERROR_PAYLOAD = re.compile(
    r"\.prod\.google\.com|/vertex/|blade:[\w.-]|\bgo/\w"
    r"|type\.googleapis\.com/(?!google\.rpc\b)"
    r"|\b(?:stairways|orcas|groot|plaque)\b", re.I)
_TRIM_NOTE = " <internal-diagnostic-trimmed>"

# The quota identifier in a rate-limit message, e.g. "input-tpm:<12 digits>",
# which names the project the endpoint billed rather than anything about the
# run. 485 episodes carry it. The message around it is worth keeping -- it says
# the episode died on a rate limit -- so only the number is replaced.
_QUOTA_ID = re.compile(r"\b(\w+-tpm):\d+")

# Cheap literal prefilter in front of every regex above. Each alternative in
# DETECT, _ERROR_PAYLOAD and _QUOTA_ID requires one of these substrings, so
# text containing none of them cannot match and can skip the regex entirely.
_LITERALS = (
    "/bns/", "borg", "uniserve", "/google/src/", _G3,
    "/usr/local/google", "@" + _DOM, "testoperator", "_google_com",
    "openclaw_env", ".prod." + _DOM, "blade:", "type.googleapis.com",
    "/vertex/", "go/", "stairways", "orcas", "groot", "plaque", "-tpm:",
)
_LITERALS_B = tuple(lit.encode() for lit in _LITERALS)


def _maybe(s: str) -> bool:
    """True if any pattern *could* match. Never a false negative."""
    low = s.lower()
    return any(lit in low for lit in _LITERALS)


# ---------------------------------------------------------------------------
# Inlined binary payloads.
#
# 86% of the corpus by volume is not text. The scenario and task trees inline
# every screenshot into the JSON as a base64 string under `__bytes_b64__`, so a
# single scenario.json is 192 MB of which 99.7% is PNG bytes. Two things follow.
#
# FIRST, AND THIS IS A CORRECTNESS PROBLEM, NOT A SPEED ONE: base64 is drawn
# from an alphabet that includes `/` and every letter, so a long enough run
# reproduces short identifiers by chance, and `re.I` triples the odds. This
# corpus really contains
#     ...YkZnd0cFuxZIP6Z/BoRG/b9tF6rF6+ce...
# inside a PNG, where `/BoRG/` satisfies `\bborg\b` because `/` is not a word
# character. Redacting it would replace four bytes of image data with
# `<scheduler-redacted>` and corrupt the image, silently, in the published
# dataset. That one turned up in the first 0.6 GB sampled, so it is not a
# hypothetical, and a corrupted image is not something any later check would
# notice.
#
# SECOND, scanning payload bytes is most of the run time, and none of it can
# ever be productive.
#
# So the interior of a long unbroken base64 run is neither scanned nor
# rewritten. A margin of `_BLOB_MARGIN` characters at each end stays live, which
# is one less than the longest literal, so an identifier that merely abuts a
# payload is still seen whichever side of the boundary it starts on.
#
# The soundness claim is that a genuine identifier cannot sit in the interior:
# every rule needs either a word boundary -- which cannot occur inside a run,
# since every neighbouring byte is a word character -- or one of `. : @ _ -`,
# none of which are in the base64 alphabet, so their presence ends the run.
# What is left is that a run must be 512 characters of letters, digits, `+` and
# `/` with no separator at all; the longest real identifier here is 30.
_BLOB_MIN = 512
_BLOB_MARGIN = max(len(lit) for lit in _LITERALS) - 1
_BLOB = re.compile(r"[A-Za-z0-9+/]{%d,}" % _BLOB_MIN)
_BLOB_B = re.compile(rb"[A-Za-z0-9+/]{%d,}" % _BLOB_MIN)


def _blob_split(s: str) -> list[tuple[str, bool]]:
    """Split `s` into (chunk, live) pairs; payload interiors are not live."""
    if len(s) < _BLOB_MIN:
        return [(s, True)]
    parts: list[tuple[str, bool]] = []
    pos = 0
    for m in _BLOB.finditer(s):
        a, b = m.start() + _BLOB_MARGIN, m.end() - _BLOB_MARGIN
        if b <= a:
            continue
        parts.append((s[pos:a], True))
        parts.append((s[a:b], False))
        pos = b
    if not parts:
        return [(s, True)]
    parts.append((s[pos:], True))
    return parts


def _drop_blobs(raw: bytes) -> bytes:
    """`raw` with payload interiors removed -- the bytes worth screening.

    Bytes rather than str on purpose: this runs over every published byte, and
    decoding first would cost a full UTF-8 pass and, for any file holding one
    non-ASCII character, a 2x or 4x wider buffer. An ASCII literal cannot match
    inside a UTF-8 multi-byte sequence (every continuation byte is >= 0x80), so
    screening the raw bytes and decoding later gives the same answer.
    """
    if len(raw) < _BLOB_MIN:
        return raw
    return _BLOB_B.sub(
        lambda m: m.group(0)[:_BLOB_MARGIN] + b"\n"
        + m.group(0)[-_BLOB_MARGIN:], raw)


def screen(raw: bytes) -> bool:
    """True if `raw` is worth decoding: cheap, never a false negative."""
    low = _drop_blobs(raw).lower()
    return any(lit in low for lit in _LITERALS_B)


_LEGACY_BRAND_LOWER = bytes.fromhex("636c6177666f726765").decode("ascii")
_LEGACY_BRAND_CAMEL = bytes.fromhex("436c6177466f726765").decode("ascii")


def screen_brand(raw: bytes) -> bool:
    return _LEGACY_BRAND_LOWER.encode("ascii") in _drop_blobs(raw).lower()


def screen_text(s: str) -> bool:
    """`screen` for text already decoded -- lines, and strings under test."""
    return any(live and _maybe(c) for c, live in _blob_split(s))


def detect(s: str) -> bool:
    """DETECT, but blind to payload interiors."""
    return any(live and _maybe(c) and DETECT.search(c)
               for c, live in _blob_split(s))


def _touches_brand(s: str) -> bool:
    return _LEGACY_BRAND_LOWER in s.lower()


def scrub_error(s: str) -> str:
    """Trim an internal diagnostic body out of an `.error` string.

    Keeps the leading sentence (everything before the first `{`, which is where
    every one of these switches from a human summary to a serialised payload)
    and marks the cut. Errors with no internal payload -- rate limits, context
    length, timeouts -- are returned as they are, minus the quota number.
    """
    if _ERROR_PAYLOAD.search(s):
        head, brace, _rest = s.partition("{")
        s = (head.rstrip() if brace else s[:80].rstrip()) + _TRIM_NOTE
    return _QUOTA_ID.sub(r"\1:<quota-id-redacted>", s)


BRAND = [(re.compile(re.escape(_LEGACY_BRAND_CAMEL)), "OpenClaw"),
         (re.compile(re.escape(_LEGACY_BRAND_LOWER)), "openclaw")]

TEXT_EXT = {".jsonl", ".json", ".html", ".txt", ".log", ".md", ".csv", ".sh",
            ".py", ".sbatch", ".yaml", ".yml"}


def redact(s: str) -> str:
    out = []
    for chunk, live in _blob_split(s):
        if live and _maybe(chunk):
            for rx, sub in REDACTIONS:
                chunk = rx.sub(sub, chunk)
        out.append(chunk)
    return "".join(out)


def _walk_strings(obj, path=""):
    """Yield (json-path, string) for every string in a decoded record."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_strings(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_strings(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        yield path, obj


def scrub_jsonl(text: str, brand: bool) -> tuple[str, int, list[str]]:
    """Rewrite leaking strings in place, leaving every other byte untouched.

    Re-serialising the record would reformat lines that have nothing wrong with
    them, so the replacement is done on the raw text using the JSON-escaped form
    of the old and new values.
    """
    if not screen_text(text) and not (brand and _touches_brand(text)):
        return text, 0, []
    out, n, stray = [], 0, []
    for line in text.splitlines(keepends=True):
        touches_brand = brand and _touches_brand(line)
        if screen_text(line) or touches_brand:
            try:
                rec = json.loads(line)
            except Exception:
                if detect(line):
                    stray.append("<unparseable line>")
                out.append(line)
                continue
            for jpath, value in _walk_strings(rec):
                new = value
                if jpath == ".error":
                    # Trim before redacting: the trim removes the body whole,
                    # so redaction is left with the one-line summary and cannot
                    # turn 40 kB of stack detail into 40 kB of placeholders.
                    new = scrub_error(new)
                if detect(new):
                    # Record anything outside .error so the caller can see the
                    # audited invariant break instead of silently widening.
                    if jpath != ".error":
                        stray.append(jpath)
                    new = redact(new)
                if brand:
                    for rx, sub in BRAND:
                        new = rx.sub(sub, new)
                if new != value:
                    old_enc = json.dumps(value)[1:-1]
                    new_enc = json.dumps(new)[1:-1]
                    if old_enc in line:
                        line = line.replace(old_enc, new_enc)
                        n += 1
        out.append(line)
    return "".join(out), n, stray


def scrub_text(text: str, brand: bool) -> tuple[str, int]:
    n = 0
    if detect(text):
        new = redact(text)
        n += int(new != text)
        text = new
    if brand:
        for rx, sub in BRAND:
            new = rx.sub(sub, text)
            n += int(new != text)
            text = new
    return text, n


def _excluded(name: str) -> bool:
    return any(bad in name for bad in EXCLUDE_DIRS)


def iter_files(root: str, prefixes: list[str] | None = None):
    """Yield files under `root`, or only under the given relative subtrees.

    `prefixes` is a list of paths relative to `root`. None walks everything,
    which is what `--whole-tree` wants; anything else confines the walk to the
    published components and cannot wander into a sibling directory.
    """
    starts = [root] if prefixes is None else [
        os.path.join(root, p) for p in prefixes]
    for start in starts:
        for dp, dns, fns in os.walk(start):
            dns[:] = [d for d in dns if not _excluded(d)]
            for fn in fns:
                if fn in EXCLUDE_FILES:
                    continue
                yield os.path.join(dp, fn)


def cmd_check(src: str, prefixes: list[str] | None = None) -> int:
    bad: list[tuple[str, str]] = []
    for fp in iter_files(src, prefixes):
        if os.path.splitext(fp)[1].lower() not in TEXT_EXT:
            continue
        try:
            with open(fp, "rb") as fh:
                raw = fh.read()
        except OSError:
            continue
        if not screen(raw):
            continue
        text = raw.decode("utf-8", errors="replace")
        if fp.endswith((".jsonl", ".json")):
            paths = set()
            for line in text.splitlines():
                if not screen_text(line):
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    if detect(line):
                        paths.add("<unparseable>")
                    continue
                for jpath, value in _walk_strings(rec):
                    # `.error` is held to the stricter field-scoped rules as
                    # well, so a payload the trim failed to remove, or a quota
                    # id the substitution missed, is caught here even though
                    # neither is corpus-wide detectable.
                    if detect(value) or (jpath == ".error" and (
                            _ERROR_PAYLOAD.search(value)
                            or _QUOTA_ID.search(value))):
                        paths.add(jpath or "<root>")
            if not paths:
                continue
            where = ",".join(sorted(paths))
        else:
            if not detect(text):
                continue
            where = "?"
        bad.append((os.path.relpath(fp, src), where))
    if not bad:
        print(f"clean: no infrastructure identifiers under {src}")
        return 0
    print(f"FOUND {len(bad)} file(s) carrying infrastructure identifiers:")
    for rel, where in bad:
        print(f"  {rel}  [{where}]")
    return 1


def cmd_apply(src: str, dst: str, brand: bool,
              prefixes: list[str] | None = None) -> int:
    if os.path.abspath(src) == os.path.abspath(dst):
        sys.exit("--src and --dst must differ; this never edits the source tree")
    files = changed = edits = 0
    stray_total: list[tuple[str, str]] = []
    for fp in iter_files(src, prefixes):
        rel = os.path.relpath(fp, src)
        out = os.path.join(dst, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        files += 1
        if os.path.splitext(fp)[1].lower() not in TEXT_EXT:
            shutil.copy2(fp, out)
            continue
        try:
            with open(fp, "rb") as fh:
                raw = fh.read()
        except OSError:
            shutil.copy2(fp, out)
            continue
        # Nothing to rewrite: copy the bytes rather than round-tripping them
        # through decode and encode. That is most of the corpus, and it makes
        # the untouched majority byte-identical to the source by construction.
        if not (screen(raw) or (brand and screen_brand(raw))):
            shutil.copy2(fp, out)
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            shutil.copy2(fp, out)
            continue
        if fp.endswith((".jsonl", ".json")):
            new, n, stray = scrub_jsonl(text, brand)
            stray_total += [(rel, p) for p in stray]
        else:
            new, n = scrub_text(text, brand)
        if n:
            changed += 1
            edits += n
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(new)
    print(f"copied {files} file(s); rewrote {changed} file(s), {edits} string(s)")
    if stray_total:
        print("\nWARNING: identifiers found outside .error -- the audited "
              "invariant no longer holds, review before publishing:")
        for rel, p in stray_total[:40]:
            print(f"  {rel}  {p}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="corpus tree to read")
    ap.add_argument("--dst", help="staging tree to write (required by --apply)")
    ap.add_argument("--apply", action="store_true",
                    help="write a scrubbed copy to --dst")
    ap.add_argument("--check", action="store_true",
                    help="report any remaining identifiers; exit 1 if found")
    ap.add_argument("--rename-brand", action="store_true",
                    help="also rewrite legacy environment names -> OpenClaw in scenario text "
                         "(off by default; see the module docstring)")
    ap.add_argument("--component", action="append", choices=sorted(COMPONENTS),
                    help="repeatable; restrict to one published component "
                         "(default: all of them)")
    ap.add_argument("--whole-tree", action="store_true",
                    help="walk --src as given instead of the published "
                         "prefixes -- for a corpus you generated yourself, "
                         "which has no manifest layout to confine the walk to")
    args = ap.parse_args(argv)

    prefixes = _resolve_prefixes(args)
    if args.check and not args.apply:
        return cmd_check(args.src, prefixes)
    if args.apply:
        if not args.dst:
            sys.exit("--apply needs --dst")
        rc = cmd_apply(args.src, args.dst, args.rename_brand, prefixes)
        return rc or cmd_check(args.dst, prefixes)
    sys.exit("nothing to do: pass --check or --apply")


def _resolve_prefixes(args) -> list[str] | None:
    """Turn the component flags into relative subtrees, or None for the lot."""
    if args.whole_tree:
        if args.component:
            sys.exit("--whole-tree and --component are mutually exclusive")
        print(f"scope: {args.src} walked whole (no manifest filter)")
        return None
    keys = args.component or sorted(COMPONENTS)
    prefixes = [PUBLISHED[k] for k in keys]
    missing = [p for p in prefixes
               if not os.path.isdir(os.path.join(args.src, p))]
    if missing:
        sys.exit(
            "--src is missing these published component(s):\n  "
            + "\n  ".join(missing)
            + f"\n\n{args.src} does not look like the corpus root. Point --src "
              "at the directory holding eval_results/, scenarios/ and the "
              "multiagent_conflict_data_* trees, or pass --whole-tree to "
              "treat it as an unstructured tree.")
    print(f"scope: {len(prefixes)} published component(s) under {args.src}")
    for k in keys:
        print(f"  {k:<18} {PUBLISHED[k]}")
    return prefixes


if __name__ == "__main__":
    raise SystemExit(main())
