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

"""Fetch the published CLASHarness corpus into the directory analysis/ reads.

WHY THIS EXISTS. The repository ships the generator, the harness, the 20
scenario seeds and the frozen report artifacts, but not the corpus those
artifacts were computed from: the eval trees and compiled task specs run to
tens of gigabytes and cannot live in git. They are hosted as a Hugging Face
dataset instead, and this script lays them out under the one path every
analysis script resolves against -- `data/` at the repository root, or
CLASHARNESS_DATA_ROOT if you point it elsewhere.

The layout is not a convention this script invents. `analysis/corpus.py`
derives ROOT_V8 and TASKS_V8 from DATA_ROOT, and `ablation.py`,
`generator_ablation.py`, `round_example_facts.py` and `rejudge/replay_core.py`
all read through it, so a component placed here is found with no further
configuration.

WHICH COMPONENTS YOU NEED depends on what you are reproducing. `--list` prints
the table; the short version is that `eval-v8` plus `tasks-v8` covers every
main-paper figure, `splits-v8` is the 58 KB train/dev/test definition you want
if you are treating this as a benchmark rather than replicating the paper, and
the rest are ablation arms. Nothing here is needed to run the test suite or to
generate a fresh corpus of your own -- only to recompute the published numbers.

USAGE
    python tools/download_data.py --list
    python tools/download_data.py --repo <hf-user>/<hf-dataset>
    python tools/download_data.py --repo ... --component eval-v8 --component tasks-v8
    CLASHARNESS_HF_REPO=<hf-user>/<hf-dataset> python tools/download_data.py --all

Requires `huggingface_hub` (`pip install huggingface_hub`), which is not a
dependency of the harness itself.
"""
from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get(
    "CLASHARNESS_DATA_ROOT", os.path.join(REPO_ROOT, "data"))

# key -> (destination under DATA_ROOT, approximate on-disk size, what it is for)
#
# Sizes are measured from the published tree, not estimated, and they are the
# sizes after `tools/scrub_corpus.py` drops `visualize.html` -- so they are what
# you actually download. Totals are summed from this table (`--list`,
# `--dry-run`) rather than restated. The destination is also the path prefix
# inside the dataset repository, so the manifest is the single place the two
# layouts are kept in step.
COMPONENTS = {
    "eval-v8": (
        "eval_results/data_3_6_flash_v8",
        "2.8 GB",
        "Primary panel: 14 models, the effort sweep, and every main-paper "
        "figure. Required by almost everything in analysis/.",
    ),
    "tasks-v8": (
        "multiagent_conflict_data_v8_Gemini_3_6_flash/tasks",
        "5.7 GB",
        "Compiled task specs for the primary corpus (generator "
        "gemini-3.6-flash). Needed to score or re-judge eval-v8.",
    ),
    "splits-v8": (
        "multiagent_conflict_data_v8_Gemini_3_6_flash/datasets",
        "58 KB",
        "The official train/dev/test split for the primary corpus: 420/90/90 "
        "of 600 rounds, held out by seed. Not read by analysis/ -- the paper "
        "reports the whole corpus -- but it is what makes this usable as a "
        "benchmark rather than a dump, so it is fetched by default.",
    ),
    "eval-sonnet5": (
        "eval_results/data_sonnet_5",
        "1.9 GB",
        "Generator ablation, arm B. Needed by generator_ablation.py.",
    ),
    "tasks-sonnet5": (
        "multiagent_conflict_data_Gemini_Sonnet_5/tasks",
        "7.1 GB",
        "Task specs for the arm-B corpus (generator claude-sonnet-5).",
    ),
    "splits-sonnet5": (
        "multiagent_conflict_data_Gemini_Sonnet_5/datasets",
        "58 KB",
        "The train/dev/test split for the arm-B corpus, same 420/90/90 shape.",
    ),
    "eval-budget30": (
        "eval_results/data_3_6_flash_v8_budget30",
        "440 MB",
        "Step-budget ablation at 30 steps. Needed by ablation.py.",
    ),
    "eval-budget40": (
        "eval_results/data_3_6_flash_v8_budget40",
        "577 MB",
        "Step-budget ablation at 40 steps. Needed by ablation.py.",
    ),
    "eval-blind": (
        "eval_results/data_3_6_flash_v8_blind",
        "219 MB",
        "Text-only (no image) arm. Needed by ablation.py.",
    ),
    "scenarios-v8": (
        "scenarios/conflict_vlm_v8_3.6_flash",
        "12.3 GB",
        "Materialised scenario packages for the primary corpus, including the "
        "rendered visual assets. Needed by round_example_facts.py; not needed "
        "for the aggregate tables.",
    ),
    "scenarios-sonnet5": (
        "scenarios/conflict_vlm_sonnet_5",
        "15.3 GB",
        "Materialised scenario packages for arm B. Only needed to regenerate "
        "analysis/dead_rounds_exclusion.json from scratch.",
    ),
}

# What `--default` fetches: enough to recompute the main-paper tables, plus the
# 58 KB split that makes the corpus usable as a benchmark. Everything omitted
# here is an ablation arm or raw assets.
DEFAULT = ["eval-v8", "tasks-v8", "splits-v8"]


def _print_manifest() -> None:
    print(f"Destination root: {DATA_ROOT}")
    print(f"  (override with CLASHARNESS_DATA_ROOT)\n")
    width = max(len(k) for k in COMPONENTS)
    for key, (dest, size, why) in COMPONENTS.items():
        have = "present" if os.path.isdir(os.path.join(DATA_ROOT, dest)) else "-"
        mark = "*" if key in DEFAULT else " "
        print(f"{mark} {key:<{width}}  {size:>7}  [{have}]  {dest}")
        for line in _wrap(why, 72):
            print(f"  {'':<{width}}  {'':>7}         {line}")
    print(f"\n* = fetched by --default ({_total(DEFAULT)}); "
          f"--all is {_total(COMPONENTS)}")


_UNITS = {"KB": 10**3, "MB": 10**6, "GB": 10**9}


def _total(keys) -> str:
    """Sum the manifest sizes of `keys`, in the manifest's own decimal units."""
    total = 0.0
    for key in keys:
        number, unit = COMPONENTS[key][1].split()
        total += float(number) * _UNITS[unit]
    for unit in ("GB", "MB"):
        if total >= _UNITS[unit]:
            return f"{total / _UNITS[unit]:.1f} {unit}"
    return f"{total / _UNITS['KB']:.0f} KB"


def _has_files(path: str) -> bool:
    for _, _, files in os.walk(path):
        if files:
            return True
    return False


def _wrap(text: str, width: int) -> list[str]:
    out, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=os.environ.get("CLASHARNESS_HF_REPO"),
                    help="Hugging Face dataset repo id, e.g. user/clash-harness-corpus "
                         "(or set CLASHARNESS_HF_REPO)")
    # One selection mode per run: letting --all or --default quietly win over
    # --component would fetch something other than what was named.
    which = ap.add_mutually_exclusive_group()
    which.add_argument("--component", action="append", choices=sorted(COMPONENTS),
                       help="repeatable; defaults to the --default set")
    which.add_argument("--all", action="store_true", help="fetch every component")
    which.add_argument("--default", action="store_true",
                       help=f"fetch {', '.join(DEFAULT)} (the implicit default)")
    ap.add_argument("--list", action="store_true",
                    help="print the manifest and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be fetched and exit")
    args = ap.parse_args(argv)

    if args.list:
        _print_manifest()
        return 0

    if args.all:
        wanted = list(COMPONENTS)
    elif args.component:
        wanted = args.component
    else:
        wanted = list(DEFAULT)

    total_dest = os.path.abspath(DATA_ROOT)
    print(f"Destination root: {total_dest}")
    for key in wanted:
        dest, size, _ = COMPONENTS[key]
        print(f"  {key:<18} {size:>7}  -> {dest}")
    print(f"  {'total':<18} {_total(wanted):>7}")
    if args.dry_run:
        return 0

    if not args.repo:
        sys.exit(
            "No dataset repository given. Pass --repo <hf-user>/<hf-dataset> or "
            "set CLASHARNESS_HF_REPO. See the README for the published id.")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub is required: pip install huggingface_hub")

    # allow_patterns is what keeps this from pulling the whole repository when
    # the caller asked for one arm. Each component is a directory prefix in the
    # dataset repo identical to its destination under DATA_ROOT.
    patterns = [f"{COMPONENTS[k][0]}/**" for k in wanted]
    os.makedirs(total_dest, exist_ok=True)
    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        local_dir=total_dest,
        allow_patterns=patterns,
    )
    # huggingface_hub logs only a warning when no pattern matches, and nothing
    # when just some of them miss, so a component the repository does not
    # publish would otherwise end in "Done." and surface later as a missing
    # file inside analysis/.
    missing = [k for k in wanted
               if not _has_files(os.path.join(total_dest, COMPONENTS[k][0]))]
    if missing:
        print(f"\nThese components are missing or empty under {total_dest}:",
              file=sys.stderr)
        for key in missing:
            print(f"  {key:<18} -> {COMPONENTS[key][0]}", file=sys.stderr)
        print(f"Check that {args.repo} publishes them under those paths.",
              file=sys.stderr)
        return 1
    # analysis/ needs no flag, but the evaluator does: --task-data-dir takes the
    # parent of tasks/, which is not the path the component is listed under.
    print(f"\nDone. {len(wanted)} component(s) under {total_dest}.")
    for key in wanted:
        dest = COMPONENTS[key][0]
        if os.path.basename(dest) == "tasks":
            print(f"  {key}: evaluate with --task-data-dir "
                  f"{os.path.join(total_dest, os.path.dirname(dest))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
