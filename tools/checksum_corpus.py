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

"""Fingerprint the published corpus, and verify a download against it.

WHY THIS EXISTS. The corpus is 46.3 GB fetched over the network into a tree that
`analysis/` then reads without further checking. A truncated component does not
announce itself: the scripts happily compute a number over whatever files
arrived, and the result is wrong in a way nobody notices. This produces one
digest per component so that "I have the same bytes the paper was computed on"
is a command rather than an assumption.

WHAT THE DIGEST COVERS. For each component, every file underneath it is hashed
and folded into a single SHA-256 over the sorted sequence of
``<relpath>\\0<size>\\0<sha256>\\n``. Sorting is what makes it reproducible
across filesystems that walk in different orders; including the relative path
means a file renamed or moved between components changes the digest even when
its contents did not; including the size makes a truncation visible in the line
itself and not only in the hash.

The rollup is deliberately one line per component rather than one line per file.
The scenario trees alone hold tens of thousands of files, and a per-file
manifest would be a multi-megabyte artifact in a repository whose point is to be
small. Pass ``--per-file`` when a mismatch needs localising -- it writes the full
listing so a diff names the offending file.

USAGE
    python tools/checksum_corpus.py --root <staging> --write CHECKSUMS.txt
    python tools/checksum_corpus.py --root data --verify CHECKSUMS.txt
    python tools/checksum_corpus.py --root data --verify CHECKSUMS.txt \\
        --component eval-v8
    python tools/checksum_corpus.py --root <staging> --component scenarios-v8 \\
        --per-file scenarios-v8.files.txt
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from download_data import COMPONENTS  # noqa: E402

# Streamed rather than read whole: single scenario assets run to megabytes and
# there is no reason to hold one in memory to hash it.
CHUNK = 1 << 20


def file_sha256(path: str) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(CHUNK)
            if not buf:
                break
            n += len(buf)
            h.update(buf)
    return h.hexdigest(), n


def component_lines(root: str, prefix: str):
    """Yield `<relpath>\\0<size>\\0<sha256>` for every file, sorted by path.

    Paths are relative to the component, not to `root`, so a component keeps its
    digest if the corpus is rehomed under a different parent directory.
    """
    base = os.path.join(root, prefix)
    paths = []
    for dp, dns, fns in os.walk(base):
        dns.sort()
        for fn in sorted(fns):
            paths.append(os.path.join(dp, fn))
    paths.sort()
    for p in paths:
        digest, size = file_sha256(p)
        rel = os.path.relpath(p, base).replace(os.sep, "/")
        yield f"{rel}\0{size}\0{digest}"


def component_digest(root: str, prefix: str, per_file=None):
    h = hashlib.sha256()
    files = total = 0
    for line in component_lines(root, prefix):
        h.update(line.encode("utf-8"))
        h.update(b"\n")
        files += 1
        total += int(line.split("\0")[1])
        if per_file is not None:
            per_file.write(line.replace("\0", "  ") + "\n")
    return h.hexdigest(), files, total


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0


def parse_manifest(path: str) -> dict[str, tuple[str, int, int]]:
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, digest, files, total = line.split()
            out[key] = (digest, int(files), int(total))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="tree holding the component directories")
    ap.add_argument("--component", action="append", choices=sorted(COMPONENTS),
                    help="repeatable; default is every component present")
    ap.add_argument("--write", help="write the rollup manifest here")
    ap.add_argument("--verify", help="compare against this manifest; exit 1 on "
                                     "any mismatch")
    ap.add_argument("--per-file", help="also write the full per-file listing "
                                       "here (use with one --component)")
    args = ap.parse_args(argv)

    keys = args.component or sorted(COMPONENTS)
    present, absent = [], []
    for k in keys:
        (present if os.path.isdir(os.path.join(args.root, COMPONENTS[k][0]))
         else absent).append(k)
    if not present:
        sys.exit(f"no component directories found under {args.root}")
    # Absent components are normal -- almost nobody downloads all eleven -- so
    # this reports rather than fails, and --verify only checks what is here.
    if absent:
        print(f"# not present, skipping: {', '.join(absent)}")

    expected = parse_manifest(args.verify) if args.verify else None
    pf = open(args.per_file, "w", encoding="utf-8") if args.per_file else None
    rows, bad = [], []
    try:
        for k in present:
            prefix = COMPONENTS[k][0]
            digest, files, total = component_digest(args.root, prefix, pf)
            rows.append((k, digest, files, total))
            status = ""
            if expected is not None:
                if k not in expected:
                    status = "  NOT IN MANIFEST"
                    bad.append(k)
                elif expected[k] != (digest, files, total):
                    ed, ef, et = expected[k]
                    status = (f"  MISMATCH (manifest: {ef} files, "
                              f"{human(et)}, {ed[:16]}...)")
                    bad.append(k)
                else:
                    status = "  ok"
            print(f"{k:<18} {digest}  {files:>7} files  {human(total):>9}{status}")
    finally:
        if pf:
            pf.close()

    if args.write:
        with open(args.write, "w", encoding="utf-8") as fh:
            fh.write("# CLASHarness corpus rollup digests\n")
            fh.write("# component  sha256  files  bytes\n")
            for k, digest, files, total in rows:
                fh.write(f"{k} {digest} {files} {total}\n")
        print(f"\nwrote {args.write} ({len(rows)} component(s))")

    if expected is not None:
        if bad:
            print(f"\nFAILED: {len(bad)} component(s) differ from the "
                  f"manifest: {', '.join(bad)}")
            print("Re-fetch them with tools/download_data.py, or rerun with "
                  "--per-file to see which files differ.")
            return 1
        print(f"\nOK: {len(rows)} component(s) match the manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
