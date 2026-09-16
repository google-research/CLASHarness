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

"""Utilities for normalizing CLASHarness trajectory JSONL records."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROUND_RE = re.compile(r"_q(\d+)$")
PRO_TASK_RE = re.compile(r"^pro_llm_(?P<slug>.+)_q(?P<round>\d+)$")


def read_trajectory_jsonl(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
    return records


def latest_records_by_task(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        task_id = str(record.get("task_id") or "")
        if task_id:
            latest[task_id] = record
    return latest


def select_records(records: list[dict[str, Any]], *, latest_only: bool = True) -> list[dict[str, Any]]:
    if latest_only:
        return list(latest_records_by_task(records).values())
    return list(records)


def infer_scenario_slug(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    slug = metadata.get("scenario_slug")
    if isinstance(slug, str) and slug:
        return slug
    task_id = str(record.get("task_id") or "")
    match = PRO_TASK_RE.match(task_id)
    if match:
        return match.group("slug")
    return "unknown"


def infer_round_id(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    round_id = metadata.get("round_id")
    if isinstance(round_id, str) and round_id:
        return round_id
    task_id = str(record.get("task_id") or "")
    match = ROUND_RE.search(task_id)
    if match:
        return f"q{int(match.group(1)):02d}"
    return ""


def infer_round_type(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    round_type = metadata.get("round_type")
    if isinstance(round_type, str) and round_type:
        return round_type
    round_id = infer_round_id(record)
    try:
        idx = int(round_id.lstrip("q"))
    except ValueError:
        return "unknown"
    if 1 <= idx <= 7:
        return "observation"
    if 8 <= idx <= 14:
        return "reconciliation"
    if 15 <= idx <= 20:
        return "update_aware"
    if 21 <= idx <= 24:
        return "action"
    return "unknown"


def failure_mode(record: dict[str, Any]) -> str:
    if bool(record.get("success")):
        return "success"
    stop_reason = str(record.get("stop_reason") or "").strip()
    if stop_reason:
        return stop_reason
    error_type = str(record.get("error_type") or "").strip()
    if error_type:
        return error_type
    error = str(record.get("error") or "").strip()
    if error:
        return "error"
    return "logical_failure"


def normalize_episode(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    return {
        "task_id": record.get("task_id"),
        "agent": record.get("agent"),
        "mode": record.get("mode") or metadata.get("mode"),
        "scenario_slug": infer_scenario_slug(record),
        "round_id": infer_round_id(record),
        "round_type": infer_round_type(record),
        "success": bool(record.get("success")),
        "score": float(record.get("score") or 0.0),
        "steps": int(record.get("steps") or 0),
        "duration_s": float(record.get("duration_s") or 0.0),
        "provider_impacted": bool(record.get("provider_impacted")),
        "error_type": record.get("error_type") or "",
        "stop_reason": record.get("stop_reason") or "",
        "failure_mode": failure_mode(record),
        "final_response": record.get("final_response") or "",
        "metadata": metadata,
        "trajectory": record.get("trajectory") if isinstance(record.get("trajectory"), list) else [],
    }


def iter_normalized_steps(record: dict[str, Any]) -> Iterable[dict[str, Any]]:
    episode = normalize_episode(record)
    trajectory = episode.pop("trajectory", [])
    final_score = episode["score"]
    for idx, step in enumerate(trajectory):
        if not isinstance(step, dict):
            continue
        yield {
            "task_id": episode["task_id"],
            "agent": episode["agent"],
            "mode": episode["mode"],
            "scenario_slug": episode["scenario_slug"],
            "round_id": episode["round_id"],
            "round_type": episode["round_type"],
            "step_index": int(step.get("step") if step.get("step") is not None else idx),
            "action": step.get("action") or "",
            "executed_action": step.get("executed_action") or step.get("action") or "",
            "stdout": step.get("clean_stdout") if step.get("clean_stdout") is not None else step.get("stdout", ""),
            "stderr": step.get("clean_stderr") if step.get("clean_stderr") is not None else step.get("stderr", ""),
            "exit_code": step.get("exit_code"),
            "reward": float(step.get("reward") or 0.0),
            "done": bool(step.get("done")),
            "current_score": step.get("current_score"),
            "score_delta": step.get("score_delta"),
            "passed_checks": step.get("passed_checks") or [],
            "failed_checks": step.get("failed_checks") or [],
            "final_score": final_score,
            "success": episode["success"],
            "stop_reason": episode["stop_reason"],
            "failure_mode": episode["failure_mode"],
            "provider_impacted": episode["provider_impacted"],
        }


def filter_records(
    records: Iterable[dict[str, Any]],
    *,
    success_only: bool = False,
    scenario_slug: str | None = None,
    round_type: str | None = None,
    agent: str | None = None,
    min_score: float | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for record in records:
        episode = normalize_episode(record)
        if success_only and not episode["success"]:
            continue
        if scenario_slug and episode["scenario_slug"] != scenario_slug:
            continue
        if round_type and episode["round_type"] != round_type:
            continue
        if agent and episode["agent"] != agent:
            continue
        if min_score is not None and episode["score"] < min_score:
            continue
        selected.append(record)
    return selected


def summarize_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    episodes = [normalize_episode(record) for record in records]
    total = len(episodes)
    successes = sum(1 for ep in episodes if ep["success"])
    provider_impacted = sum(1 for ep in episodes if ep["provider_impacted"])
    stage_counts: dict[str, Counter[str]] = defaultdict(Counter)
    scenario_counts: dict[str, Counter[str]] = defaultdict(Counter)
    failure_counts: Counter[str] = Counter()
    step_sum = 0
    success_step_sum = 0
    fail_step_sum = 0
    success_total = 0
    fail_total = 0

    for ep in episodes:
        label = "passed" if ep["success"] else "failed"
        stage_counts[ep["round_type"]]["total"] += 1
        stage_counts[ep["round_type"]][label] += 1
        scenario_counts[ep["scenario_slug"]]["total"] += 1
        scenario_counts[ep["scenario_slug"]][label] += 1
        failure_counts[ep["failure_mode"]] += 1
        step_sum += ep["steps"]
        if ep["success"]:
            success_step_sum += ep["steps"]
            success_total += 1
        else:
            fail_step_sum += ep["steps"]
            fail_total += 1

    def _rate(passed: int, denom: int) -> float:
        return (passed / denom) if denom else 0.0

    return {
        "total": total,
        "successful": successes,
        "success_rate": _rate(successes, total),
        "provider_impacted": provider_impacted,
        "avg_steps": (step_sum / total) if total else 0.0,
        "avg_steps_success": (success_step_sum / success_total) if success_total else 0.0,
        "avg_steps_failed": (fail_step_sum / fail_total) if fail_total else 0.0,
        "by_round_type": {
            key: {
                "passed": value["passed"],
                "total": value["total"],
                "success_rate": _rate(value["passed"], value["total"]),
            }
            for key, value in sorted(stage_counts.items())
        },
        "by_scenario": {
            key: {
                "passed": value["passed"],
                "total": value["total"],
                "success_rate": _rate(value["passed"], value["total"]),
            }
            for key, value in sorted(scenario_counts.items())
        },
        "failure_modes": dict(failure_counts.most_common()),
    }


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out
