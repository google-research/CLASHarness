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

"""Export CLASHarness trajectories into reusable episode, step, and SFT JSONL datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clasharness.utils.trajectory_schema import (
    filter_records,
    iter_normalized_steps,
    normalize_episode,
    read_trajectory_jsonl,
    select_records,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Trajectory JSONL file(s)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--all-records", action="store_true", help="Use every record instead of latest record per task")
    parser.add_argument("--success-only", action="store_true", help="Keep only successful records")
    parser.add_argument("--scenario-slug", default=None, help="Filter to one scenario slug")
    parser.add_argument("--round-type", default=None, choices=["observation", "reconciliation", "update_aware", "action", "unknown"], help="Filter to one round type")
    parser.add_argument("--agent", default=None, help="Filter to one agent name")
    parser.add_argument("--min-score", type=float, default=None, help="Keep records with score >= value")
    return parser.parse_args()


def _sft_records(record: dict[str, Any]) -> list[dict[str, Any]]:
    episode = normalize_episode(record)
    if not episode["success"] and episode["agent"] != "expert":
        return []
    instruction = str(record.get("instruction") or episode.get("metadata", {}).get("instruction") or "")
    if not instruction:
        instruction = str(record.get("metadata", {}).get("canonical_instruction") or "")
    rows: list[dict[str, Any]] = []
    prior: list[str] = []
    for step in iter_normalized_steps(record):
        action = str(step.get("action") or "").strip()
        if not action or action.upper() == "DONE":
            continue
        context = "\n".join(prior[-6:])
        rows.append({
            "task_id": episode["task_id"],
            "agent": episode["agent"],
            "scenario_slug": episode["scenario_slug"],
            "round_id": episode["round_id"],
            "round_type": episode["round_type"],
            "instruction": instruction,
            "context": context,
            "target_command": action,
            "success": episode["success"],
            "final_score": episode["score"],
        })
        stdout = str(step.get("stdout") or "").strip()
        stderr = str(step.get("stderr") or "").strip()
        prior.append(f"$ {action}\nSTDOUT: {stdout}\nSTDERR: {stderr}")
    return rows


def main() -> None:
    args = parse_args()
    records = read_trajectory_jsonl(args.paths)
    records = select_records(records, latest_only=not args.all_records)
    records = filter_records(
        records,
        success_only=args.success_only,
        scenario_slug=args.scenario_slug,
        round_type=args.round_type,
        agent=args.agent,
        min_score=args.min_score,
    )

    out_dir = Path(args.output_dir)
    episodes = [normalize_episode(record) for record in records]
    steps = [step for record in records for step in iter_normalized_steps(record)]
    sft = [row for record in records for row in _sft_records(record)]

    write_jsonl(out_dir / "episodes.jsonl", episodes)
    write_jsonl(out_dir / "steps.jsonl", steps)
    write_jsonl(out_dir / "sft_commands.jsonl", sft)

    print(f"Wrote {len(episodes)} episodes to {out_dir / 'episodes.jsonl'}")
    print(f"Wrote {len(steps)} steps to {out_dir / 'steps.jsonl'}")
    print(f"Wrote {len(sft)} SFT command rows to {out_dir / 'sft_commands.jsonl'}")


if __name__ == "__main__":
    main()
