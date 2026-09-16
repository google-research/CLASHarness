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

"""Mirror the reconciled ``mock_subagents`` / ``subagent_completion`` structures
from the patched ``conflict_vlm_v1`` source tree into a flat eval task-data-dir.

Why this exists
---------------
``scripts/reconcile_conflict_vlm_channels.py`` surgically patched the *source*
scenario tree (``clasharness/data/scenarios/conflict_vlm_v1/<scenario>/tasks/
<task>/specs.json``) so every mock subagent / ``subagent_completion`` carries a
``reply_channel`` and a canonical ``--channel``. But the evaluator does NOT read
that tree - it reads the flattened export produced by
``generate_conflict_vlm_scenarios.py --output_data_dir`` (default
``/tmp/clasharness-conflict-vlm-data``), whose ``tasks/<task_id>/specs.json`` were
exported BEFORE the reconciliation logic existed and are therefore stale (no
``reply_channel``). Running the eval against the stale export silently negates
the entire R1/R2 + reconciliation fix.

Re-running generation would rebuild the export correctly (the compiler now bakes
in the reconciliation) but is expensive and non-deterministic. Instead this tool
copies ONLY the two already-patched, verified structures from source into the
export, matched by ``task_id`` (identical in both trees):

  * ``data.private.mock_subagents``           (replaced wholesale)
  * ``data.private.background_events[*]`` where ``type == "subagent_completion"``
    (replaced in order; non-completion events are left untouched)

Safety: before replacing anything it asserts each source/export entry is
identical *modulo the patch itself* (i.e. after dropping ``reply_channel`` and
blanking the ``--channel`` value in message-send effects). If any entry differs
beyond that, the task is refused and reported - the tool never blindly
overwrites divergent data. Idempotent; dry-run by default.

Usage::

    venv/bin/python scripts/mirror_reconcile_to_eval_data.py           # dry run
    venv/bin/python scripts/mirror_reconcile_to_eval_data.py --write    # apply
    venv/bin/python scripts/mirror_reconcile_to_eval_data.py \
        --eval-data-dir /tmp/clasharness-conflict-vlm-data --write
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import shlex
import sys
from pathlib import Path

_SCENARIO_ROOT = (
    Path(__file__).resolve().parent.parent / "clasharness/data/scenarios"
)
# Historical default. Generation runs pass an explicit --output_dir (e.g.
# conflict_vlm_v1_3.5_flash), so this bare name frequently does not exist; it is
# only a default now and --scenario-dir overrides it. An unresolvable source
# tree used to surface as a bare "no source specs" line, which reads like "the
# corpus is empty" rather than "you are pointed at the wrong directory".
_DEFAULT_SRC = _SCENARIO_ROOT / "conflict_vlm_v1"
_DEFAULT_EVAL_DIR = Path("/tmp/clasharness-conflict-vlm-data")


def _expand(raw) -> Path:
    """Expand '~' before Path() sees it.

    A shell does not expand '~' after '=', so --eval-data-dir=~/Data/x arrives
    as the literal string '~/Data/x', which Path() treats as a *relative* path
    rooted at the CWD. Always route user-supplied paths through here.
    """
    return Path(os.path.expanduser(str(raw)))


def _available_scenario_dirs() -> list[str]:
    """Scenario dirs that actually contain compiled task specs."""
    if not _SCENARIO_ROOT.is_dir():
        return []
    found = []
    for child in sorted(_SCENARIO_ROOT.iterdir()):
        if child.is_dir() and glob.glob(
            os.path.join(str(child), "*", "tasks", "*", "specs.json")
        ):
            found.append(child.name)
    return found


def _canon_effect(x: object) -> object:
    """Canonicalize one state_effect so patch-only diffs (the --channel value)
    are erased, leaving everything else to compare structurally."""
    if isinstance(x, str) and x.strip().lower().startswith(
        ("message send", "message broadcast")
    ):
        try:
            toks = shlex.split(x)
        except ValueError:
            return ("RAW", x)
        if "--channel" in toks:
            i = toks.index("--channel")
            if i + 1 < len(toks):
                toks[i + 1] = "<CH>"
        return ("MSG", toks)
    if isinstance(x, str):
        try:
            return ("CMD", shlex.split(x))
        except ValueError:
            return ("RAW", x)
    return ("OBJ", x)


def _canon_entry(entry: dict) -> dict:
    """A mock / completion dict reduced to its patch-invariant form."""
    e = copy.deepcopy(entry)
    e.pop("reply_channel", None)
    e["state_effects"] = [_canon_effect(x) for x in (e.get("state_effects") or [])]
    return e


def _subagent_completions(events: list) -> list[dict]:
    return [
        e
        for e in (events or [])
        if isinstance(e, dict) and e.get("type") == "subagent_completion"
    ]


def _build_source_index(src: Path) -> dict[str, dict]:
    idx: dict[str, dict] = {}
    for sp in glob.glob(os.path.join(str(src), "*", "tasks", "*", "specs.json")):
        d = json.loads(Path(sp).read_text(encoding="utf-8"))
        idx[d["task_id"]] = d
    return idx


def _mirror_task(export_spec_path: Path, src_spec: dict) -> dict:
    """Return stats; mutate the parsed export spec in place (caller writes)."""
    export = json.loads(export_spec_path.read_text(encoding="utf-8"))
    s_priv = (src_spec.get("data") or {}).get("private") or {}
    e_priv = (export.get("data") or {}).get("private") or {}

    s_mocks = s_priv.get("mock_subagents") or []
    e_mocks = e_priv.get("mock_subagents") or []
    s_comps = _subagent_completions(s_priv.get("background_events") or [])
    e_comps = _subagent_completions(e_priv.get("background_events") or [])

    stats = {"changed": False, "refused": None, "_export": export}

    # --- safety: refuse unless identical modulo the patch --------------------
    if len(s_mocks) != len(e_mocks):
        stats["refused"] = f"mock count {len(s_mocks)}!={len(e_mocks)}"
        return stats
    if len(s_comps) != len(e_comps):
        stats["refused"] = f"completion count {len(s_comps)}!={len(e_comps)}"
        return stats
    for i, (a, b) in enumerate(zip(s_mocks, e_mocks)):
        if _canon_entry(a) != _canon_entry(b):
            stats["refused"] = f"mock[{i}] differs beyond patch"
            return stats
    for i, (a, b) in enumerate(zip(s_comps, e_comps)):
        if _canon_entry(a) != _canon_entry(b):
            stats["refused"] = f"completion[{i}] differs beyond patch"
            return stats

    # --- mirror: mock_subagents wholesale ------------------------------------
    before = json.dumps(e_priv.get("mock_subagents"), sort_keys=True,
                        ensure_ascii=False)
    e_priv["mock_subagents"] = copy.deepcopy(s_mocks)
    after = json.dumps(e_priv["mock_subagents"], sort_keys=True,
                       ensure_ascii=False)
    if before != after:
        stats["changed"] = True

    # --- mirror: subagent_completion entries in order ------------------------
    it = iter(copy.deepcopy(s_comps))
    new_events = []
    for ev in (e_priv.get("background_events") or []):
        if isinstance(ev, dict) and ev.get("type") == "subagent_completion":
            src_ev = next(it)
            b0 = json.dumps(ev, sort_keys=True, ensure_ascii=False)
            a0 = json.dumps(src_ev, sort_keys=True, ensure_ascii=False)
            if b0 != a0:
                stats["changed"] = True
            new_events.append(src_ev)
        else:
            new_events.append(ev)
    e_priv["background_events"] = new_events
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="apply changes in place (default: dry run)")
    ap.add_argument("--eval-data-dir", default=str(_DEFAULT_EVAL_DIR),
                    help="flat eval task-data-dir (contains tasks/<id>/specs.json)")
    ap.add_argument("--scenario-dir", default=str(_DEFAULT_SRC),
                    help="source scenario tree (contains <slug>/tasks/<id>/specs.json). "
                         "Must match the --output_dir the corpus was generated with.")
    args = ap.parse_args()

    src = _expand(args.scenario_dir)
    tasks_root = _expand(args.eval_data_dir) / "tasks"
    if not tasks_root.is_dir():
        print(f"[FATAL] not a directory: {tasks_root}", file=sys.stderr)
        return 2

    if not src.is_dir():
        avail = _available_scenario_dirs()
        print(f"[FATAL] scenario dir does not exist: {src}", file=sys.stderr)
        print(
            "[FATAL] scenario dirs with compiled specs: "
            + (", ".join(avail) if avail else "(none found)"),
            file=sys.stderr,
        )
        print("[FATAL] pass the right one with --scenario-dir", file=sys.stderr)
        return 2

    src_index = _build_source_index(src)
    if not src_index:
        avail = _available_scenario_dirs()
        print(f"[FATAL] no source specs under {src}", file=sys.stderr)
        print(
            "[FATAL] scenario dirs with compiled specs: "
            + (", ".join(avail) if avail else "(none found)"),
            file=sys.stderr,
        )
        print("[FATAL] pass the right one with --scenario-dir", file=sys.stderr)
        return 2

    totals = {"tasks": 0, "changed": 0, "refused": 0, "no_source": 0}
    refusals = []
    for spec_path in sorted(tasks_root.glob("*/specs.json")):
        totals["tasks"] += 1
        export = json.loads(spec_path.read_text(encoding="utf-8"))
        tid = export.get("task_id")
        src = src_index.get(tid)
        if src is None:
            totals["no_source"] += 1
            refusals.append((tid, "no matching source task"))
            continue
        st = _mirror_task(spec_path, src)
        if st["refused"]:
            totals["refused"] += 1
            refusals.append((tid, st["refused"]))
            continue
        if st["changed"]:
            totals["changed"] += 1
            if args.write:
                # Match the export's formatting: 2-space indent, non-ASCII kept,
                # no trailing newline.
                spec_path.write_text(
                    json.dumps(st["_export"], indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

    mode = "WROTE" if args.write else "DRY-RUN (no files written)"
    print(f"\n=== mirror_reconcile_to_eval_data [{mode}] ===")
    print(f"eval data dir      : {args.eval_data_dir}")
    print(f"tasks scanned      : {totals['tasks']}")
    print(f"tasks changed      : {totals['changed']}")
    print(f"tasks refused      : {totals['refused']}")
    print(f"tasks w/o source   : {totals['no_source']}")
    if refusals:
        print("\nrefused / unmatched (NOT written):")
        for tid, why in refusals[:50]:
            print(f"  {tid}: {why}")
    if not args.write:
        print("\nRe-run with --write to apply.")
    return 1 if refusals else 0


if __name__ == "__main__":
    raise SystemExit(main())
