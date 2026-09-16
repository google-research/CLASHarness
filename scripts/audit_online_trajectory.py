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

"""Audit trajectory JSONL quality for online data collection."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _has_openclaw_step(ep: dict[str, Any]) -> bool:
    return any(
        (step.get("action") or "").strip().startswith("openclaw ")
        for step in ep.get("trajectory", [])
    )


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    passed = sum(1 for r in rows if r.get("success"))
    with_openclaw = [r for r in rows if _has_openclaw_step(r)]
    with_openclaw_passed = sum(1 for r in with_openclaw if r.get("success"))

    compat_status = Counter()
    error_tags = Counter()
    skipped_actions = Counter()
    uncovered = Counter()

    for ep in rows:
        for step in ep.get("trajectory", []):
            status = step.get("compat_status")
            if status:
                compat_status[status] += 1

            tags = step.get("error_tags", []) or []
            for tag in tags:
                error_tags[tag] += 1

            if status == "skipped_incompatible":
                action = (step.get("action") or "").strip()
                if action:
                    skipped_actions[action] += 1
                for tag in tags:
                    if tag.startswith("incompatible_"):
                        uncovered[tag] += 1

    return {
        "total_episodes": total,
        "passed_episodes": passed,
        "success_rate": (passed / total) if total else 0.0,
        "with_openclaw_episodes": len(with_openclaw),
        "with_openclaw_passed": with_openclaw_passed,
        "with_openclaw_success_rate": (
            (with_openclaw_passed / len(with_openclaw)) if with_openclaw else 0.0
        ),
        "compat_status_counts": dict(compat_status),
        "error_tag_counts": dict(error_tags.most_common()),
        "skipped_actions_top20": dict(skipped_actions.most_common(20)),
        "uncovered_compat_rules": dict(uncovered.most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", help="Trajectory JSONL file path")
    parser.add_argument(
        "--out",
        default=None,
        help="Output report path (default: <jsonl>_quality_report.json)",
    )
    args = parser.parse_args()

    input_path = Path(args.jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"JSONL not found: {input_path}")

    rows = _load_jsonl(input_path)
    report = build_report(rows)

    out_path = (
        Path(args.out)
        if args.out
        else input_path.with_name(f"{input_path.stem}_quality_report.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(f"Audited {input_path} ({len(rows)} episodes)")
    print(f"Report saved -> {out_path}")


if __name__ == "__main__":
    main()
