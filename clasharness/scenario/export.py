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

"""Export generated tasks into PRO scenario specs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

from clasharness.core.task import Task
from clasharness.scenario.compiler import compile_scenario_to_tasks
from clasharness.scenario.spec import (
    ScenarioEvalItem,
    ScenarioManifest,
    ScenarioRound,
    ScenarioSpec,
    ScenarioWorkspace,
)
from clasharness.scenario.validator import validate_scenario

_TARGET_GENERATORS = {
    "hard_decision_workflow",
    "complex_composed_workflow",
    "pro_scenario_workflow",
}


def _safe_slug(value: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value.strip().lower())
    return slug.strip("_") or "unknown"


def _scenario_title(slug: str) -> str:
    return slug.replace("_", " ").replace("-", " ").title()


def _round_from_task(task: Task, index: int) -> ScenarioRound:
    solution_commands = task.ground_truth.solution_commands if task.ground_truth else []
    return ScenarioRound(
        round_id=f"round_{index:03d}",
        task_id=task.task_id,
        instruction=task.instruction,
        task_spec=task.to_dict(),
        expert_trajectory=list(solution_commands),
        metadata={
            "generator_id": task.generator_id,
            "template_id": task.template_id,
            "realism_tags": list(task.realism_tags),
            "decision_requirements": list(task.decision_requirements),
        },
    )


def _eval_from_task(task: Task, round_id: str) -> ScenarioEvalItem:
    checks = task.ground_truth.evaluation_checks if task.ground_truth else []
    answer = task.ground_truth.answer if task.ground_truth else None
    return ScenarioEvalItem(
        eval_id=f"eval_{round_id}",
        round_id=round_id,
        checks=list(checks),
        answer=answer,
        metadata={
            "task_id": task.task_id,
            "generator_id": task.generator_id,
        },
    )


def _build_spec(generator_id: str, scenario_slug: str, tasks: list[Task]) -> ScenarioSpec:
    ordered = list(tasks)
    rounds = [_round_from_task(task, idx + 1) for idx, task in enumerate(ordered)]
    eval_items = [
        _eval_from_task(task, round_item.round_id)
        for task, round_item in zip(ordered, rounds, strict=True)
    ]
    first = ordered[0]
    public = dict(first.data.public)
    title = (
        str(public.get("complex_scenario") or public.get("hard_decision_scenario_title") or "")
        or _scenario_title(scenario_slug)
    )
    metadata = {
        "round_count": len(rounds),
        "source_task_ids": [task.task_id for task in ordered],
    }
    if generator_id == "hard_decision_workflow":
        metadata.update(
            {
                "hard_decision_scenario": scenario_slug,
                "hard_decision_ability": public.get("hard_decision_ability"),
                "hard_decision_ability_tags": public.get("hard_decision_ability_tags", []),
            }
        )
    elif generator_id == "complex_composed_workflow":
        metadata.update(
            {
                "complex_scenario_slug": scenario_slug,
                "complex_template_examples": sorted(
                    {
                        str(task.data.public.get("complex_template"))
                        for task in ordered
                        if task.data.public.get("complex_template")
                    }
                ),
                "complex_variant_examples": sorted(
                    {
                        str(task.data.public.get("complex_variant"))
                        for task in ordered
                        if task.data.public.get("complex_variant")
                    }
                ),
                "realism_tiers": sorted(
                    {
                        str(task.data.public.get("realism_tier"))
                        for task in ordered
                        if task.data.public.get("realism_tier")
                    }
                ),
            }
        )

    spec = ScenarioSpec(
        scenario_id=f"{generator_id}:{scenario_slug}",
        generator_id=generator_id,
        scenario_slug=scenario_slug,
        title=title,
        description=(
            "Scenario-first PRO package compiled from existing generated task specs."
        ),
        narrative={
            "layer": "layer0-narrative",
            "goal": str(public.get("complex_scenario") or title),
            "source": "compiled_from_current_task_generator",
        },
        workspace=ScenarioWorkspace(
            notes={
                "layer": "layer1-workspace",
                "initial_state": first.initial_state,
                "public_seed_metadata": public,
            }
        ),
        sessions=[],
        updates=[],
        rounds=rounds,
        eval_items=eval_items,
        manifest=ScenarioManifest(
            source_generator=generator_id,
            compiled_task_ids=[task.task_id for task in ordered],
            metadata={"scenario_forge_layers": ["narrative", "workspace", "eval"]},
        ),
        metadata=metadata,
    )
    if generator_id == "pro_scenario_workflow":
        _enrich_pro_spec(spec, ordered)
    return spec


def _enrich_pro_spec(spec: ScenarioSpec, tasks: list[Task]) -> None:
    first = tasks[0]
    public = first.data.public
    scenario = dict(public.get("pro_scenario", {}))
    spec.description = str(scenario.get("layer0_narrative", {}).get("core_narrative", spec.description))
    spec.narrative = {
        "layer": "layer0-narrative",
        **dict(scenario.get("layer0_narrative", {})),
    }
    spec.workspace.files = [
        {
            "id": str(item.get("path", "")),
            "path": str(item.get("path", "")),
            "type": str(item.get("type", "")),
            "facts": str(item.get("facts", "")),
        }
        for item in scenario.get("layer1_workspace", {}).get("files", [])
    ]
    spec.workspace.notes = {
        "layer": "layer1-workspace",
        "initial_file_count": scenario.get("layer1_workspace", {}).get("initial_file_count", 0),
    }
    from clasharness.scenario.spec import ScenarioSession, ScenarioUpdate

    spec.sessions = [
        ScenarioSession(
            session_id=str(item.get("session_id", "")),
            kind=str(item.get("type", "text")),
            channel=str(item.get("channel", "default")),
            metadata={
                "role": item.get("role"),
                "participants": item.get("participants", []),
                "phase_count": item.get("phase_count", 0),
            },
        )
        for item in scenario.get("layer2_sessions", [])
    ]
    spec.updates = [
        ScenarioUpdate(
            update_id=str(item.get("update_id", "")),
            before_round=f"q{int(item.get('before_round', 0)):02d}",
            workspace_adds=[
                {"path": path, "source_update": item.get("update_id")}
                for path in item.get("workspace_adds", [])
            ],
            session_appends=[
                {"session_id": session_id, "source_update": item.get("update_id")}
                for session_id in item.get("session_appends", [])
            ],
            metadata={"reveals": item.get("reveals")},
        )
        for item in scenario.get("layer4_dynamic", [])
    ]
    session_ids = [session.session_id for session in spec.sessions]
    for round_item, task in zip(spec.rounds, tasks, strict=True):
        round_item.round_id = str(task.data.public.get("pro_round_id", round_item.round_id))
        round_item.update_ids = list(task.data.public.get("pro_active_updates", []))
        round_item.session_refs = session_ids
        round_item.workspace_refs = [str(item.get("path", "")) for item in spec.workspace.files]
        round_item.metadata.update(
            {
                "round_type": task.data.public.get("pro_round_type"),
                "target": task.data.public.get("pro_target", {}),
            }
        )
    for eval_item, round_item, task in zip(spec.eval_items, spec.rounds, tasks, strict=True):
        eval_item.round_id = round_item.round_id
        eval_item.eval_id = f"eval_{round_item.round_id}"
        eval_item.kind = str(task.data.public.get("pro_round_type", "task_checks"))
    spec.metadata.update(
        {
            "scenario_series": public.get("pro_series", "pro_strict_v1"),
            "pro_profile": "scenario_forge_style",
            "round_type_distribution": {
                "multi_choice": sum(1 for task in tasks if task.data.public.get("pro_round_type") == "multi_choice"),
                "exec_check": sum(1 for task in tasks if task.data.public.get("pro_round_type") == "exec_check"),
            },
            "session_count": len(spec.sessions),
            "workspace_file_count": len(spec.workspace.files),
            "update_count": len(spec.updates),
        }
    )


def hard_tasks_to_scenario_specs(tasks: Iterable[Task]) -> list[ScenarioSpec]:
    grouped: dict[str, list[Task]] = defaultdict(list)
    for task in tasks:
        if task.generator_id != "hard_decision_workflow":
            continue
        slug = _safe_slug(str(task.data.public.get("hard_decision_scenario", "unknown")))
        grouped[slug].append(task)
    return [
        _build_spec("hard_decision_workflow", slug, grouped[slug])
        for slug in sorted(grouped)
    ]


def complex_tasks_to_scenario_specs(tasks: Iterable[Task]) -> list[ScenarioSpec]:
    grouped: dict[str, list[Task]] = defaultdict(list)
    for task in tasks:
        if task.generator_id != "complex_composed_workflow":
            continue
        slug = _safe_slug(str(task.data.public.get("complex_scenario_slug", "unknown")))
        grouped[slug].append(task)
    return [
        _build_spec("complex_composed_workflow", slug, grouped[slug])
        for slug in sorted(grouped)
    ]


def pro_tasks_to_scenario_specs(tasks: Iterable[Task]) -> list[ScenarioSpec]:
    grouped: dict[str, list[Task]] = defaultdict(list)
    for task in tasks:
        if task.generator_id != "pro_scenario_workflow":
            continue
        slug = _safe_slug(str(task.data.public.get("pro_scenario_slug", "unknown")))
        grouped[slug].append(task)
    return [
        _build_spec("pro_scenario_workflow", slug, grouped[slug])
        for slug in sorted(grouped)
    ]


def scenario_specs_for_tasks(tasks: Iterable[Task]) -> list[ScenarioSpec]:
    task_list = list(tasks)
    return (
        hard_tasks_to_scenario_specs(task_list)
        + complex_tasks_to_scenario_specs(task_list)
        + pro_tasks_to_scenario_specs(task_list)
    )


def write_scenario_specs(specs: Iterable[ScenarioSpec], output_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for spec in specs:
        validate_scenario(spec).raise_for_errors()
        path = output_dir / spec.generator_id / spec.scenario_slug / "scenario.json"
        paths.append(spec.write_json(path))
    return paths


def compile_target_scenarios_in_task_list(tasks: list[Task], specs: Iterable[ScenarioSpec]) -> list[Task]:
    """Replace target-generator tasks with compiled scenario output."""

    compiled_by_id: dict[str, Task] = {}
    for spec in specs:
        for task in compile_scenario_to_tasks(spec):
            compiled_by_id[task.task_id] = task
    return [
        compiled_by_id.get(task.task_id, task)
        if task.generator_id in _TARGET_GENERATORS
        else task
        for task in tasks
    ]
