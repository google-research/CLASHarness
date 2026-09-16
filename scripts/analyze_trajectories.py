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

"""Summarize CLASHarness trajectory JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clasharness.utils.trajectory_schema import (
    filter_records,
    read_trajectory_jsonl,
    select_records,
    summarize_records,
)


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Trajectory JSONL file(s)")
    parser.add_argument("--all-records", action="store_true", help="Use every record instead of latest record per task")
    parser.add_argument("--success-only", action="store_true", help="Keep only successful records")
    parser.add_argument("--scenario-slug", default=None, help="Filter to one scenario slug")
    parser.add_argument("--round-type", default=None, choices=["observation", "reconciliation", "update_aware", "action", "unknown"], help="Filter to one round type")
    parser.add_argument("--agent", default=None, help="Filter to one agent name")
    parser.add_argument("--min-score", type=float, default=None, help="Keep records with score >= value")
    parser.add_argument("--json-out", default=None, help="Optional path for machine-readable summary JSON")
    return parser.parse_args()


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
    summary = summarize_records(records)

    print("CLASHarness trajectory summary")
    print(f"Records: {summary['total']}")
    print(f"Success: {summary['successful']}/{summary['total']} ({_fmt_pct(summary['success_rate'])})")
    print(f"Provider impacted: {summary['provider_impacted']}")
    print(f"Avg steps: {summary['avg_steps']:.1f} (success={summary['avg_steps_success']:.1f}, failed={summary['avg_steps_failed']:.1f})")

    print("\nBy round type:")
    for name, stats in summary["by_round_type"].items():
        print(f"  {name:<16} {stats['passed']:>3}/{stats['total']:<3} {_fmt_pct(stats['success_rate'])}")

    print("\nBy scenario:")
    for name, stats in summary["by_scenario"].items():
        print(f"  {name:<42} {stats['passed']:>3}/{stats['total']:<3} {_fmt_pct(stats['success_rate'])}")

    print("\nFailure modes:")
    for name, count in summary["failure_modes"].items():
        print(f"  {count:>4}  {name}")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
