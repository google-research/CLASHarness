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

"""Generate tasks from all registered generators and split into datasets."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import random
import re
import shutil
import sys
from pathlib import Path


def _parse_hard_decision_scenario_counts(raw: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not raw.strip():
        return counts
    for item in raw.split(','):
        piece = item.strip()
        if not piece:
            continue
        if '=' not in piece:
            raise ValueError(
                f"Invalid hard-decision scenario count override: {piece!r}. Expected scenario=count."
            )
        scenario_slug, count_text = piece.split('=', 1)
        scenario_slug = scenario_slug.strip()
        count_text = count_text.strip()
        if not scenario_slug:
            raise ValueError('Scenario slug cannot be empty in --hard-decision-scenario-counts.')
        try:
            count = int(count_text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid count for scenario {scenario_slug!r}: {count_text!r}. Expected an integer."
            ) from exc
        if count < 1:
            raise ValueError('Scenario counts must be >= 1.')
        counts[scenario_slug] = count
    return counts

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from clasharness.core.task import save_task
from clasharness.scenario.export import (
    compile_target_scenarios_in_task_list,
    scenario_specs_for_tasks,
    write_scenario_specs,
)
from clasharness.tasks.generation_options import set_generation_options
from clasharness.tasks.registry import generate_all_tasks, get_all_generators
from clasharness.tools.agent_tool_spec import validate_task_solution_commands

# Importing the generators package auto-discovers and registers every generator
# (see clasharness/tasks/generators/__init__.py). No hardcoded list to maintain.
import clasharness.tasks.generators  # noqa: F401

_BANNED_PATTERNS = [
    "openclaw configure ",
    "openclaw security set-token",
    "openclaw channels config",
    "openclaw plugins remove",
    "openclaw webhooks add",
    "openclaw agents add --name",
    "openclaw agents set-identity --name",
    "openclaw message poll --question",
    "openclaw cron add --schedule",
]


def _default_output_data_dir() -> Path:
    return Path(__file__).parent.parent / "clasharness" / "data"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate all task specs and dataset splits.",
    )
    parser.add_argument(
        "--message-dry-run",
        dest="message_dry_run",
        action="store_true",
        help="Append --dry-run to generated openclaw message commands.",
    )
    parser.add_argument(
        "--no-message-dry-run",
        dest="message_dry_run",
        action="store_false",
        help="Do not append --dry-run to generated message commands (default).",
    )
    parser.add_argument(
        "--plugin-install-mode",
        choices=["stable", "mixed"],
        default="mixed",
        help=(
            "Plugin task generation mode: "
            "'stable' keeps enable/disable only; "
            "'mixed' uses real plugins install for install-task subset."
        ),
    )
    parser.add_argument(
        "--output-data-dir",
        default="",
        help=(
            "Output data root containing tasks/ and datasets/. "
            "Default: clasharness/data."
        ),
    )
    parser.add_argument(
        "--complex-task-pack",
        choices=["off", "standard"],
        default="standard",
        help=(
            "Complex composed workflow generation mode. "
            "'off' disables complex task pack, "
            "'standard' adds the selected complex scenario pack."
        ),
    )
    parser.add_argument(
        "--complex-scenario-profile",
        choices=["legacy", "life_work"],
        default="life_work",
        help=(
            "Scenario set used by complex workflows. "
            "'legacy' keeps the older 120-task templates; "
            "'life_work' uses 160 realistic life/work scenarios (default)."
        ),
    )
    parser.add_argument(
        "--complex-min-steps",
        type=int,
        default=3,
        help="Minimum step count allowed for generated complex composed tasks.",
    )
    parser.add_argument(
        "--complex-max-steps",
        type=int,
        default=5,
        help="Maximum step count allowed for generated complex composed tasks (<=5).",
    )
    parser.add_argument(
        "--hard-decision-variants-per-scenario",
        type=int,
        default=16,
        help=(
            "Base hard-decision count to use per scenario. "
            "Default: 16. With the default hard profile and no explicit per-scenario overrides, "
            "this yields 272 hard_decision_workflow tasks, including two longer conflict/resume scenarios."
        ),
    )
    parser.add_argument(
        "--hard-decision-scenario-counts",
        default="",
        help=(
            "Optional comma-separated per-scenario count overrides for hard_decision_workflow, "
            "for example: existing_state_followthrough=24,state_repair_followthrough=20. "
            "When you pass explicit overrides, any unspecified hard scenario falls back to the shared variants-per-scenario count."
        ),
    )
    parser.add_argument(
        "--include-branch-sensitive",
        dest="include_branch_sensitive",
        action="store_true",
        help="Generate branch_sensitive_workflow tasks as an experimental family.",
    )
    parser.add_argument(
        "--no-include-branch-sensitive",
        dest="include_branch_sensitive",
        action="store_false",
        help="Do not generate branch_sensitive_workflow tasks (default).",
    )
    parser.add_argument(
        "--branch-sensitive-variants-per-scenario",
        type=int,
        default=0,
        help=(
            "Number of branch-sensitive variants to generate per scenario. "
            "Default: 0. Only used when --include-branch-sensitive is enabled."
        ),
    )
    parser.add_argument(
        "--include-pro-scenarios",
        action="store_true",
        help="Generate the scenario-forge-style PRO task family.",
    )
    parser.add_argument(
        "--pro-rounds-per-scenario",
        type=int,
        default=24,
        help="Number of PRO rounds per generated scenario (12-30, default: 24).",
    )
    parser.add_argument(
        "--emit-scenario-specs",
        action="store_true",
        help=(
            "Write PRO scenario.json specs for hard_decision_workflow and "
            "complex_composed_workflow without changing default task generation."
        ),
    )
    parser.add_argument(
        "--scenario-output-dir",
        default="",
        help=(
            "Output root for PRO scenario specs. "
            "Default: <output-data-dir>/scenarios."
        ),
    )
    parser.add_argument(
        "--scenario-series",
        default="",
        help=(
            "Optional scenario series subdirectory and metadata label, "
            "for example 'pro' writes <scenario-output-dir>/pro/..."
        ),
    )
    parser.add_argument(
        "--scenario-spec-scope",
        choices=["all", "pro"],
        default="all",
        help=(
            "Which generated tasks to export as scenario specs. "
            "'all' exports hard/complex/PRO specs; 'pro' exports only the new PRO scenario family."
        ),
    )
    parser.add_argument(
        "--compile-scenarios-to-tasks",
        action="store_true",
        help=(
            "Compile emitted in-memory scenario specs back to current Task specs "
            "before writing tasks. This is a compatibility smoke path for PRO."
        ),
    )
    parser.set_defaults(message_dry_run=False, include_branch_sensitive=False)
    return parser.parse_args()


def _lint_solution_commands(tasks) -> None:
    violations: list[str] = []
    violations.extend(validate_task_solution_commands(tasks))
    for task in tasks:
        if not task.ground_truth:
            continue
        for cmd in task.ground_truth.solution_commands:
            lowered = cmd.lower()
            for banned in _BANNED_PATTERNS:
                if banned in lowered:
                    violations.append(f"{task.task_id}: {cmd}")
                    break

    if violations:
        print("\nInvalid or legacy command patterns detected in generated solutions:")
        for v in violations[:50]:
            print(f"  - {v}")
        if len(violations) > 50:
            print(f"  ... and {len(violations)-50} more")
        raise SystemExit("Generation aborted due to invalid command syntax.")


def _write_profile_reports(tasks, data_dir: Path, command_profile: str) -> None:
    datasets_dir = data_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    by_generator: Counter[str] = Counter()
    for task in tasks:
        if task.generator_id:
            by_generator[task.generator_id] += 1

    coverage = {
        "command_profile": command_profile,
        "total_tasks": len(tasks),
        "generator_count": len(by_generator),
        "by_generator": dict(sorted(by_generator.items())),
    }
    (datasets_dir / "generator_coverage_report.json").write_text(
        json.dumps(coverage, indent=2, ensure_ascii=False) + "\n"
    )

    online_dir = datasets_dir / "online_types"
    online_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[str]] = {
        "openclaw_cli": [],
        "http_api_curl": [],
        "weather_cli_online_flag": [],
        "gcalcli_online_flag": [],
        "online_any": [],
    }
    for task in tasks:
        if not task.ground_truth:
            continue
        cmds = task.ground_truth.solution_commands
        tid = task.task_id
        if any(cmd.strip().startswith("openclaw ") for cmd in cmds):
            groups["openclaw_cli"].append(tid)
        if any(cmd.strip().startswith("curl ") for cmd in cmds):
            groups["http_api_curl"].append(tid)
        if any(cmd.strip().startswith("weather ") and "--online" in cmd for cmd in cmds):
            groups["weather_cli_online_flag"].append(tid)
        if any(cmd.strip().startswith("gcalcli ") and "--online" in cmd for cmd in cmds):
            groups["gcalcli_online_flag"].append(tid)
        if any("--online" in cmd or cmd.strip().startswith("curl ") for cmd in cmds):
            groups["online_any"].append(tid)

    for key, items in groups.items():
        unique = sorted(set(items))
        (online_dir / f"{key}.txt").write_text("\n".join(unique) + ("\n" if unique else ""))
        groups[key] = unique

    report = {
        "command_profile": command_profile,
        "output_dir": str(online_dir),
        "categories": {
            key: {
                "count": len(items),
                "file": str(online_dir / f"{key}.txt"),
            }
            for key, items in groups.items()
            if key != "online_any"
        },
        "online_any": {
            "count": len(groups["online_any"]),
            "file": str(online_dir / "online_any.txt"),
        },
    }
    (online_dir / "online_types_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )


def _write_complex_workflow_report(tasks, data_dir: Path) -> None:
    datasets_dir = data_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    complex_tasks = [
        t
        for t in tasks
        if t.generator_id == "complex_composed_workflow" and t.ground_truth
    ]
    if not complex_tasks:
        report = {
            "complex_task_count": 0,
            "step_distribution": {},
            "openclaw_step_ratio": 0.0,
            "template_coverage": {},
            "scenario_coverage": {},
            "scenario_distribution": {},
            "high_volatility_ratio": 0.0,
            "strict_online_expected_failure_risk": {},
            "causal_chain_score": {
                "avg": 0.0,
                "min": 0.0,
                "max": 0.0,
            },
            "filler_step_ratio": 0.0,
            "entity_consistency_pass_rate": 0.0,
        }
        (datasets_dir / "complex_workflow_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        )
        return

    step_distribution: Counter[int] = Counter()
    openclaw_steps = 0
    total_steps = 0
    template_coverage: Counter[str] = Counter()
    scenario_coverage: Counter[str] = Counter()
    high_volatility_count = 0
    risk_coverage: Counter[str] = Counter()
    causal_scores: list[float] = []
    filler_steps = 0
    entity_consistency_pass = 0

    for task in complex_tasks:
        assert task.ground_truth is not None
        commands = task.ground_truth.solution_commands
        step_distribution[len(commands)] += 1
        total_steps += len(commands)
        openclaw_steps += sum(1 for cmd in commands if cmd.strip().startswith("openclaw "))
        template_id = str(task.data.public.get("complex_template", "unknown"))
        template_coverage[template_id] += 1
        scenario_slug = str(task.data.public.get("complex_scenario_slug", "unknown"))
        scenario_coverage[scenario_slug] += 1
        if bool(task.data.public.get("complex_high_volatility", False)):
            high_volatility_count += 1
        strict_risk = str(task.data.public.get("strict_online_risk", "unknown"))
        risk_coverage[strict_risk] += 1
        causal_scores.append(float(task.data.public.get("causal_chain_score", 0.0)))
        filler_steps += int(task.data.public.get("filler_step_count", 0))
        if bool(task.data.public.get("entity_consistency_pass", False)):
            entity_consistency_pass += 1

    report = {
        "complex_task_count": len(complex_tasks),
        "step_distribution": dict(sorted(step_distribution.items())),
        "openclaw_step_ratio": (
            round(openclaw_steps / total_steps, 4) if total_steps else 0.0
        ),
        "template_coverage": dict(sorted(template_coverage.items())),
        "scenario_coverage": dict(sorted(scenario_coverage.items())),
        "scenario_distribution": dict(sorted(scenario_coverage.items())),
        "high_volatility_ratio": round(high_volatility_count / len(complex_tasks), 4),
        "strict_online_expected_failure_risk": dict(sorted(risk_coverage.items())),
        "causal_chain_score": {
            "avg": round(sum(causal_scores) / len(causal_scores), 4) if causal_scores else 0.0,
            "min": round(min(causal_scores), 4) if causal_scores else 0.0,
            "max": round(max(causal_scores), 4) if causal_scores else 0.0,
        },
        "filler_step_ratio": round(filler_steps / total_steps, 4) if total_steps else 0.0,
        "entity_consistency_pass_rate": (
            round(entity_consistency_pass / len(complex_tasks), 4) if complex_tasks else 0.0
        ),
    }
    (datasets_dir / "complex_workflow_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )


def _count_tokens(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _instruction_has_internal_id(text: str) -> bool:
    patterns = (
        r"\bagent-[A-Za-z0-9_-]+\b",
        r"\b(?:task|evt|email)_[0-9]{4}\b",
        r"\b(?:email_seed|task_seed)_[A-Za-z0-9_-]+\b",
        r"\[[^\]]+\]",
    )
    return any(re.search(pattern, text or "") for pattern in patterns)


def _first_ngram_signature(text: str, n: int = 4) -> str:
    tokens = re.findall(r"[A-Za-z0-9']+", (text or "").lower())
    if not tokens:
        return ""
    return " ".join(tokens[:n])


def _repeated_ngram_rate(texts: list[str], n: int = 4) -> float:
    grams: list[str] = []
    for text in texts:
        tokens = re.findall(r"[A-Za-z0-9']+", (text or "").lower())
        if len(tokens) < n:
            continue
        grams.extend(" ".join(tokens[idx:idx + n]) for idx in range(len(tokens) - n + 1))
    if not grams:
        return 0.0
    counts = Counter(grams)
    repeated = sum(count for count in counts.values() if count > 1)
    return round(repeated / len(grams), 4)


def _write_instruction_quality_report(tasks, data_dir: Path) -> None:
    datasets_dir = data_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    total = len(tasks)
    if total == 0:
        report = {
            "task_count": 0,
            "avg_token_length": 0.0,
            "prompt_style_distribution": {},
            "tasks_with_hidden_constraints_pct": 0.0,
            "tasks_with_internal_id_leaks_pct": 0.0,
            "avg_instruction_variant_count": 0.0,
            "avg_surface_forms_per_hard_scenario": 0.0,
            "hard_scenario_surface_form_counts": {},
            "top_instruction_openers": {},
            "instruction_lead_signature_reuse_rate": 0.0,
            "instruction_fourgram_repetition_rate": 0.0,
        }
    else:
        token_counts = [_count_tokens(task.instruction) for task in tasks]
        prompt_styles = Counter(
            str(task.data.public.get("prompt_style", "direct")) for task in tasks
        )
        scenario_surface_forms: dict[str, set[str]] = {}
        all_texts: list[str] = []
        opener_signatures: list[str] = []
        for task in tasks:
            scenario = str(task.data.public.get("hard_decision_scenario", "unknown"))
            bucket = scenario_surface_forms.setdefault(scenario, set())
            surface_forms = [task.instruction, task.canonical_instruction or task.instruction, *task.variant_texts()]
            bucket.update(text.strip() for text in surface_forms if text and text.strip())
            all_texts.extend(text.strip() for text in surface_forms if text and text.strip())
            lead = _first_ngram_signature(task.instruction)
            if lead:
                opener_signatures.append(lead)
        hidden_count = sum(1 for task in tasks if task.hidden_constraints)
        leak_count = sum(
            1
            for task in tasks
            if _instruction_has_internal_id(task.instruction)
            or any(_instruction_has_internal_id(text) for text in task.variant_texts())
        )
        opener_counts = Counter(opener_signatures)
        report = {
            "task_count": total,
            "avg_token_length": round(sum(token_counts) / total, 4),
            "prompt_style_distribution": dict(sorted(prompt_styles.items())),
            "tasks_with_hidden_constraints_pct": round(hidden_count / total, 4),
            "tasks_with_internal_id_leaks_pct": round(leak_count / total, 4),
            "avg_instruction_variant_count": round(
                sum(len(task.variant_texts()) for task in tasks) / total,
                4,
            ),
            "avg_surface_forms_per_hard_scenario": round(
                (
                    sum(len(forms) for forms in scenario_surface_forms.values())
                    / len(scenario_surface_forms)
                )
                if scenario_surface_forms
                else 0.0,
                4,
            ),
            "hard_scenario_surface_form_counts": {
                key: len(value)
                for key, value in sorted(scenario_surface_forms.items())
            },
            "top_instruction_openers": dict(opener_counts.most_common(10)),
            "instruction_lead_signature_reuse_rate": round(
                (
                    sum(count for count in opener_counts.values() if count > 1)
                    / len(opener_signatures)
                )
                if opener_signatures
                else 0.0,
                4,
            ),
            "instruction_fourgram_repetition_rate": _repeated_ngram_rate(all_texts, n=4),
        }
    (datasets_dir / "instruction_quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )


def _check_is_command_history_lock(check: dict[str, object]) -> bool:
    return (
        check.get("type") == "output"
        and check.get("output_field") == "command_history"
        and check.get("condition") == "regex"
    )


def _check_is_exact_lock(check: dict[str, object]) -> bool:
    if check.get("type") == "config" and check.get("condition") == "equals":
        return True
    if check.get("type") == "state" and check.get("condition") == "equals":
        return True
    if check.get("type") == "effect" and check.get("condition") == "field_equals":
        return True
    return False


def _write_evaluation_rigidity_report(tasks, data_dir: Path) -> None:
    datasets_dir = data_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    command_history_locks = 0
    hidden_parameter_locks = 0
    family_rigidity: Counter[str] = Counter()

    for task in tasks:
        if not task.ground_truth:
            continue
        hidden_values = {
            str(item.get("value", "")).lower()
            for item in task.hidden_constraints
            if item.get("value")
        }
        rigidity_units = 0
        for check in task.ground_truth.evaluation_checks:
            if _check_is_command_history_lock(check):
                command_history_locks += 1
                rigidity_units += 1
                continue
            if _check_is_exact_lock(check):
                expected = str(check.get("expected", "")).lower()
                if hidden_values and expected in hidden_values:
                    hidden_parameter_locks += 1
                    rigidity_units += 1
        if rigidity_units:
            family_rigidity[str(task.generator_id or "unknown")] += rigidity_units

    report = {
        "task_count": len(tasks),
        "command_history_lock_count": command_history_locks,
        "exact_hidden_parameter_lock_count": hidden_parameter_locks,
        "families_with_highest_rigidity": dict(
            sorted(family_rigidity.items(), key=lambda item: (-item[1], item[0]))
        ),
    }
    (datasets_dir / "evaluation_rigidity_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )


def _write_decision_load_report(tasks, data_dir: Path) -> None:
    datasets_dir = data_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    family_counts: Counter[str] = Counter()
    family_decision_total: Counter[str] = Counter()
    family_underspecified: Counter[str] = Counter()
    infer_tasks = 0

    for task in tasks:
        family = str(task.generator_id or "unknown")
        family_counts[family] += 1
        req_count = len(task.decision_requirements)
        family_decision_total[family] += req_count
        if req_count:
            infer_tasks += 1
        if "underspecified" in task.realism_tags:
            family_underspecified[family] += 1

    per_family = {}
    for family, count in sorted(family_counts.items()):
        per_family[family] = {
            "task_count": count,
            "avg_decision_requirements": round(
                family_decision_total[family] / count, 4
            ),
            "underspecified_ratio": round(
                family_underspecified[family] / count, 4
            ),
        }

    total = len(tasks) or 1
    report = {
        "task_count": len(tasks),
        "tasks_requiring_inference_pct": round(infer_tasks / total, 4),
        "avg_decision_requirements_per_task": round(
            sum(len(task.decision_requirements) for task in tasks) / total, 4
        ),
        "per_family_ambiguity": per_family,
    }
    (datasets_dir / "decision_load_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )


def _write_online_readiness_report(tasks, data_dir: Path) -> None:
    datasets_dir = data_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    provider_counts: Counter[str] = Counter()
    online_requirement_counts: Counter[str] = Counter()
    availability_counts: Counter[str] = Counter()
    auth_requirements: dict[str, list[str]] = {
        "google_calendar": ["CLASHARNESS_GOOGLE_CLIENT_SECRET_FILE", "CLASHARNESS_GOOGLE_TOKEN_FILE"],
        "email_provider": ["CLASHARNESS_EMAIL_PROVIDER", "CLASHARNESS_GOOGLE_TOKEN_FILE"],
        "google_tasks": ["CLASHARNESS_GOOGLE_CLIENT_SECRET_FILE", "CLASHARNESS_GOOGLE_TOKEN_FILE"],
        "weather_provider": ["CLASHARNESS_ENABLE_ONLINE_DATA"],
        "channel_provider": ["openclaw channel auth/config"],
    }
    expected_failure_modes: Counter[str] = Counter()

    for task in tasks:
        online_requirement_counts[str(task.online_requirement or "none")] += 1
        availability_counts[str(task.availability_tier or "stable")] += 1
        for dep in task.provider_dependencies:
            provider_counts[dep] += 1
        tier = str(task.availability_tier or "stable")
        if tier == "stable":
            expected_failure_modes["logical_failure"] += 1
        elif tier == "flaky":
            expected_failure_modes["auth_or_channel_unavailable"] += 1
        else:
            expected_failure_modes["provider_unavailable_or_rate_limited"] += 1

    report = {
        "task_count": len(tasks),
        "provider_coverage": dict(sorted(provider_counts.items())),
        "online_requirement_distribution": dict(sorted(online_requirement_counts.items())),
        "availability_tier_distribution": dict(sorted(availability_counts.items())),
        "auth_requirements": auth_requirements,
        "expected_failure_modes": dict(sorted(expected_failure_modes.items())),
    }
    (datasets_dir / "online_readiness_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )


def _write_benchmark_slices(tasks, data_dir: Path) -> None:
    datasets_dir = data_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    slice_dir = datasets_dir / "benchmark_slices"
    slice_dir.mkdir(parents=True, exist_ok=True)

    slices: dict[str, list[str]] = {
        "baseline_core": sorted(
            task.task_id
            for task in tasks
            if task.online_requirement == "none"
            and "high_volatility" not in task.realism_tags
            and task.generator_id != "complex_composed_workflow"
            and task.generator_id != "branch_sensitive_workflow"
        ),
        "complex_realistic": sorted(
            task.task_id
            for task in tasks
            if task.generator_id == "complex_composed_workflow"
            and "life_work" in task.realism_tags
        ),
        "online_strict": sorted(
            task.task_id for task in tasks if task.online_requirement == "required"
        ),
        "challenge": sorted(
            task.task_id for task in tasks if "challenge" in task.realism_tags
        ),
        "branch_challenge": sorted(
            task.task_id for task in tasks if task.generator_id == "branch_sensitive_workflow"
        ),
    }
    for name, task_ids in slices.items():
        (slice_dir / f"{name}.txt").write_text(
            "\n".join(task_ids) + ("\n" if task_ids else "")
        )

    report = {
        "output_dir": str(slice_dir),
        "slices": {
            name: {
                "count": len(task_ids),
                "file": str(slice_dir / f"{name}.txt"),
                "experimental": name == "branch_challenge",
            }
            for name, task_ids in sorted(slices.items())
        },
    }
    (slice_dir / "benchmark_slice_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )


def _remove_generated_challenge_task_dirs(data_dir: Path) -> int:
    tasks_dir = data_dir / "tasks"
    if not tasks_dir.exists():
        return 0
    removed = 0
    for pattern in ("complex_*", "hard_decision_*", "branch_sensitive_*"):
        for path in tasks_dir.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                removed += 1
    return removed


def _split_counts(total: int) -> tuple[int, int, int]:
    if total <= 0:
        return (0, 0, 0)
    if total == 1:
        return (1, 0, 0)
    if total == 2:
        return (1, 1, 0)
    train = max(1, int(total * 0.6))
    dev = max(1, int(total * 0.2))
    test = total - train - dev
    if test <= 0:
        if train > dev:
            train -= 1
        else:
            dev -= 1
        test = 1
    return (train, dev, test)


def _stratified_hard_splits(tasks, seed: int = 42) -> dict[str, list[str]]:
    by_scenario: dict[str, list] = {}
    for task in tasks:
        scenario = str(task.data.public.get("hard_decision_scenario", "unknown"))
        by_scenario.setdefault(scenario, []).append(task)

    splits: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    for scenario, scenario_tasks in sorted(by_scenario.items()):
        ordered = list(scenario_tasks)
        random.Random(f"hard-split:{seed}:{scenario}").shuffle(ordered)
        train_n, dev_n, test_n = _split_counts(len(ordered))
        splits["train"].extend(task.task_id for task in ordered[:train_n])
        splits["dev"].extend(task.task_id for task in ordered[train_n:train_n + dev_n])
        splits["test"].extend(task.task_id for task in ordered[train_n + dev_n:train_n + dev_n + test_n])
    for split_name, task_ids in splits.items():
        random.Random(f"hard-split:{seed}:{split_name}:merge").shuffle(task_ids)
    return splits


def _default_splits(tasks, seed: int = 42) -> dict[str, list[str]]:
    ordered = list(tasks)
    random.Random(seed).shuffle(ordered)
    train_n, dev_n, test_n = _split_counts(len(ordered))
    return {
        "train": [t.task_id for t in ordered[:train_n]],
        "dev": [t.task_id for t in ordered[train_n:train_n + dev_n]],
        "test": [t.task_id for t in ordered[train_n + dev_n:train_n + dev_n + test_n]],
    }


def _build_dataset_splits(tasks, seed: int = 42) -> dict[str, list[str]]:
    hard_tasks = [t for t in tasks if t.generator_id == "hard_decision_workflow"]
    other_tasks = [t for t in tasks if t.generator_id != "hard_decision_workflow"]
    hard_splits = _stratified_hard_splits(hard_tasks, seed=seed)
    other_splits = _default_splits(other_tasks, seed=seed)
    merged: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    for split_name in merged:
        merged[split_name] = list(hard_splits[split_name]) + list(other_splits[split_name])
        random.Random(f"dataset-split:{seed}:{split_name}").shuffle(merged[split_name])
    return merged


def _write_hard_split_coverage_report(tasks, splits: dict[str, list[str]], data_dir: Path, seed: int) -> None:
    datasets_dir = data_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    task_by_id = {task.task_id: task for task in tasks if task.generator_id == "hard_decision_workflow"}
    report = {"seed": seed, "hard_total": len(task_by_id), "splits": {}}
    for split_name, task_ids in splits.items():
        hard_ids = [task_id for task_id in task_ids if task_id in task_by_id]
        scenario_counts = Counter(
            str(task_by_id[task_id].data.public.get("hard_decision_scenario", "unknown"))
            for task_id in hard_ids
        )
        report["splits"][split_name] = {
            "hard_count": len(hard_ids),
            "by_scenario": dict(sorted(scenario_counts.items())),
        }
    (datasets_dir / "hard_split_coverage_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )


def main():
    args = _parse_args()
    command_profile = "local_skill"
    data_dir = (
        Path(args.output_data_dir).expanduser()
        if args.output_data_dir
        else _default_output_data_dir()
    )
    hard_decision_scenario_counts = _parse_hard_decision_scenario_counts(
        args.hard_decision_scenario_counts
    )
    set_generation_options(
        message_dry_run=args.message_dry_run,
        plugin_install_mode=args.plugin_install_mode,
        command_profile=command_profile,
        complex_task_pack=args.complex_task_pack,
        complex_scenario_profile=args.complex_scenario_profile,
        complex_min_steps=args.complex_min_steps,
        complex_max_steps=args.complex_max_steps,
        hard_decision_variants_per_scenario=args.hard_decision_variants_per_scenario,
        hard_decision_scenario_counts=hard_decision_scenario_counts,
        include_branch_sensitive=args.include_branch_sensitive,
        branch_sensitive_variants_per_scenario=args.branch_sensitive_variants_per_scenario,
        include_pro_scenarios=args.include_pro_scenarios,
        pro_rounds_per_scenario=args.pro_rounds_per_scenario,
    )

    # Show registered generators
    generators = get_all_generators()
    print(f"Registered generators: {len(generators)}")
    for gid, cls in generators.items():
        print(f"  - {gid} (domains={cls.required_domains}, difficulty={cls.difficulty})")

    # Generate all tasks
    print("\nGenerating tasks...")
    tasks = generate_all_tasks()
    print(f"Generated {len(tasks)} tasks total")
    print(
        "Generation options: "
        f"message_dry_run={args.message_dry_run}, "
        f"plugin_install_mode={args.plugin_install_mode}, "
        f"command_profile={command_profile}, "
        f"complex_task_pack={args.complex_task_pack}, "
        f"complex_scenario_profile={args.complex_scenario_profile}, "
        f"complex_min_steps={args.complex_min_steps}, "
        f"complex_max_steps={args.complex_max_steps}, "
        f"hard_decision_variants_per_scenario={args.hard_decision_variants_per_scenario}, "
        f"include_branch_sensitive={args.include_branch_sensitive}, "
        f"branch_sensitive_variants_per_scenario={args.branch_sensitive_variants_per_scenario}, "
        f"include_pro_scenarios={args.include_pro_scenarios}, "
        f"pro_rounds_per_scenario={args.pro_rounds_per_scenario}"
    )
    _lint_solution_commands(tasks)

    scenario_specs = []
    if args.emit_scenario_specs or args.compile_scenarios_to_tasks:
        scenario_specs = scenario_specs_for_tasks(tasks)
        if args.scenario_spec_scope == "pro":
            scenario_specs = [
                spec
                for spec in scenario_specs
                if spec.generator_id == "pro_scenario_workflow"
            ]
        print(
            f"Built {len(scenario_specs)} PRO scenario specs "
            "for selected scenario families"
        )
    if args.compile_scenarios_to_tasks:
        tasks = compile_target_scenarios_in_task_list(tasks, scenario_specs)
        print(f"Compiled PRO scenarios back to {len(tasks)} runnable task specs")
    if args.emit_scenario_specs:
        scenario_output_dir = (
            Path(args.scenario_output_dir).expanduser()
            if args.scenario_output_dir
            else data_dir / "scenarios"
        )
        scenario_series = args.scenario_series.strip()
        if scenario_series:
            if "/" in scenario_series or "\\" in scenario_series:
                raise SystemExit("--scenario-series must be a simple path segment")
            scenario_output_dir = scenario_output_dir / scenario_series
            for spec in scenario_specs:
                spec.metadata.setdefault("scenario_series", scenario_series)
        paths = write_scenario_specs(scenario_specs, scenario_output_dir)
        print(f"Wrote {len(paths)} PRO scenario specs under {scenario_output_dir}")

    # Print distribution
    domain_counts = Counter()
    difficulty_counts = Counter()
    for t in tasks:
        for d in t.domains:
            domain_counts[d] += 1
        difficulty_counts[t.difficulty] += 1

    print("\nBy domain:")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count}")

    print("\nBy difficulty:")
    for diff, count in sorted(difficulty_counts.items()):
        print(f"  Level {diff}: {count}")

    removed = _remove_generated_challenge_task_dirs(data_dir)
    if removed:
        print(f"\nRemoved {removed} generated challenge task directories")

    # Save tasks
    print(f"\nSaving tasks to {data_dir / 'tasks'}...")
    for task in tasks:
        save_task(task, data_dir=data_dir)

    # Split into train/dev/test.
    split_seed = 42
    splits = _build_dataset_splits(tasks, seed=split_seed)

    datasets_dir = data_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    for split_name, task_ids in splits.items():
        split_path = datasets_dir / f"{split_name}.txt"
        split_path.write_text("\n".join(task_ids) + "\n")
        print(f"  {split_name}: {len(task_ids)} tasks")

    total_task_ids = [task.task_id for task in tasks]
    total_path = datasets_dir / "total.txt"
    total_path.write_text("\n".join(total_task_ids) + "\n")
    print(f"  total: {len(total_task_ids)} tasks")

    _write_profile_reports(tasks, data_dir, command_profile)
    _write_hard_split_coverage_report(tasks, splits, data_dir, split_seed)
    _write_complex_workflow_report(tasks, data_dir)
    _write_instruction_quality_report(tasks, data_dir)
    _write_evaluation_rigidity_report(tasks, data_dir)
    _write_decision_load_report(tasks, data_dir)
    _write_online_readiness_report(tasks, data_dir)
    _write_benchmark_slices(tasks, data_dir)
    print(f"Reports saved under {datasets_dir}")

    print("\nDone!")


if __name__ == "__main__":
    main()
