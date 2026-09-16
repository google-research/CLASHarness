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

"""Approach B: surgically patch shipped ``conflict_vlm_v1`` task ``specs.json``.

Why this exists
---------------
``_compile_round_to_task`` now (a) reconciles the channel of every mock-subagent
message-send ``state_effect`` to the canonical shipped ``session_id`` and (b)
stamps each mock / ``subagent_completion`` event with a ``reply_channel`` so the
runtime (R1/R2) can route the subagent's spoken reply into
``effects["messages_sent"]`` - the sink the grader actually reads. The 180
already-shipped tasks predate that logic: their mock effects still carry the
LLM's invented ``#ops-triage``-style channels and no ``reply_channel``.

Re-materializing through ``scripts/generate_conflict_vlm_scenarios.py`` with
``resume=True`` would re-run the LLM auditor + F1 repair (non-deterministic,
needs model access, and would churn the auditor/F1 fixes already baked into the
shipped specs). Instead we surgically patch, in place and deterministically,
ONLY the two structures the new logic touches:

  * ``data.private.mock_subagents[*]``
  * ``data.private.background_events[*]`` where ``type == "subagent_completion"``

For each we (1) rewrite any non-canonical ``--channel`` inside its message-send
``state_effects`` to the matching ``session_id`` and (2) add ``reply_channel``
(the subagent's own reconciled effect channel, else the round's primary
coordination channel). Everything else in the file is left untouched. The exact
reconciliation helpers are imported from the materializer so patched output is
identical to a fresh compile. Idempotent.

NOTE on grading: all 259 ``messages_sent`` checks are ``field_semantic_match``
with only ``field``+``criterion`` (no structured channel/target filter), so the
grader's pass/fail does NOT depend on the reply channel. Channel reconciliation
here is for realism/consistency (solver ``message search``, coherent persisted
session files, parity with fresh compile); ``reply_channel`` matters because
R1/R2 route the reply onto it.

Usage::

    # dry run (default): report what would change, write nothing
    venv/bin/python scripts/reconcile_conflict_vlm_channels.py

    # apply in place
    venv/bin/python scripts/reconcile_conflict_vlm_channels.py --write
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Import the SAME reconciliation helpers the compiler uses, so a patched spec is
# byte-for-byte what _compile_round_to_task would now emit for these fields.
from clasharness.scenario.conflict_vlm_materializer import (
    _primary_round_channel,
    _reconcile_subagent_channels,
    _reply_channel_for,
)

_DATA = (
    Path(__file__).resolve().parent.parent
    / "clasharness/data/scenarios/conflict_vlm_v1"
)


def _session_ids(scenario_dir: Path) -> list[str]:
    """Canonical channel identities, in the scenario's declared order.

    ``scenario.json`` lists sessions in canonical order (matches ``spec.sessions``
    used by the compiler) and is plain JSON, so we avoid importing the heavier
    ``ConflictVLMScenarioSpec`` (which pulls PIL). The per-session ``.jsonl``
    files under ``sessions/`` are the runtime message stores keyed by these ids.
    """
    sj = json.loads((scenario_dir / "scenario.json").read_text(encoding="utf-8"))
    return [
        s["session_id"]
        for s in sj.get("sessions", [])
        if isinstance(s, dict) and s.get("session_id")
    ]


def _patch_entry(entry: dict, session_ids: list[str], primary: str) -> bool:
    """Reconcile one mock / completion dict in place. Returns True if changed."""
    before = json.dumps(entry, sort_keys=True, ensure_ascii=False)
    effects = entry.get("state_effects", []) or []
    reconciled = _reconcile_subagent_channels(list(effects), session_ids)
    entry["state_effects"] = reconciled
    entry["reply_channel"] = _reply_channel_for(reconciled, session_ids, primary)
    after = json.dumps(entry, sort_keys=True, ensure_ascii=False)
    return before != after


def _patch_task(spec_path: Path, session_ids: list[str]) -> dict:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    priv = spec.get("data", {}).get("private", {})
    gt = spec.get("ground_truth", {})
    solution_commands = [
        c for c in gt.get("solution_commands", []) if isinstance(c, str)
    ]
    primary = _primary_round_channel(solution_commands, session_ids)

    stats = {
        "mocks": 0,
        "mocks_changed": 0,
        "completions": 0,
        "completions_changed": 0,
        "bad_reply_channel": 0,
    }

    for m in priv.get("mock_subagents", []) or []:
        if not isinstance(m, dict):
            continue
        stats["mocks"] += 1
        if _patch_entry(m, session_ids, primary):
            stats["mocks_changed"] += 1
        if session_ids and m.get("reply_channel") not in session_ids:
            stats["bad_reply_channel"] += 1

    for e in priv.get("background_events", []) or []:
        if not isinstance(e, dict) or e.get("type") != "subagent_completion":
            continue
        stats["completions"] += 1
        if _patch_entry(e, session_ids, primary):
            stats["completions_changed"] += 1
        if session_ids and e.get("reply_channel") not in session_ids:
            stats["bad_reply_channel"] += 1

    stats["changed"] = bool(
        stats["mocks_changed"] or stats["completions_changed"]
    )
    stats["_spec"] = spec
    stats["_primary"] = primary
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--write",
        action="store_true",
        help="apply changes in place (default: dry run, write nothing)",
    )
    ap.add_argument(
        "--data",
        default=str(_DATA),
        help="conflict_vlm_v1 scenarios root",
    )
    args = ap.parse_args()

    root = Path(args.data)
    if not root.is_dir():
        print(f"[FATAL] not a directory: {root}", file=sys.stderr)
        return 2

    totals = {
        "scenarios": 0,
        "tasks": 0,
        "tasks_changed": 0,
        "mocks": 0,
        "mocks_changed": 0,
        "completions": 0,
        "completions_changed": 0,
        "bad_reply_channel": 0,
        "empty_sessions": 0,
    }

    for scen in sorted(os.listdir(root)):
        sdir = root / scen
        tp = sdir / "tasks"
        if not tp.is_dir():
            continue
        totals["scenarios"] += 1
        session_ids = _session_ids(sdir)
        if not session_ids:
            totals["empty_sessions"] += 1
            print(f"[WARN] {scen}: no sessions declared; reply_channel will be ''")

        for task in sorted(os.listdir(tp)):
            spec_path = tp / task / "specs.json"
            if not spec_path.is_file():
                continue
            totals["tasks"] += 1
            st = _patch_task(spec_path, session_ids)
            for k in (
                "mocks",
                "mocks_changed",
                "completions",
                "completions_changed",
                "bad_reply_channel",
            ):
                totals[k] += st[k]
            if st["changed"]:
                totals["tasks_changed"] += 1
                if args.write:
                    # Match shipped formatting: 2-space indent, non-ASCII kept,
                    # no trailing newline.
                    spec_path.write_text(
                        json.dumps(st["_spec"], indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )

    mode = "WROTE" if args.write else "DRY-RUN (no files written)"
    print(f"\n=== reconcile_conflict_vlm_channels [{mode}] ===")
    print(f"scenarios scanned      : {totals['scenarios']}")
    print(f"tasks scanned          : {totals['tasks']}")
    print(f"tasks changed          : {totals['tasks_changed']}")
    print(f"mock_subagents         : {totals['mocks']} "
          f"(changed {totals['mocks_changed']})")
    print(f"subagent_completions   : {totals['completions']} "
          f"(changed {totals['completions_changed']})")
    print(f"bad reply_channel (not in sessions): {totals['bad_reply_channel']}")
    print(f"scenarios w/o sessions : {totals['empty_sessions']}")
    if not args.write:
        print("\nRe-run with --write to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
