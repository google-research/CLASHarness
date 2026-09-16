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

"""Materialize generated PRO scenario layers into scenario packages and tasks."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from clasharness.core.task import Difficulty, GroundTruth, Task, TaskData, save_task
from clasharness.scenario._common import _round_index, _round_type, _update_anchor_id
from clasharness.scenario.llm_pipeline import ScenarioSeed
from clasharness.scenario.spec import (
    ScenarioEvalItem,
    ScenarioManifest,
    ScenarioRound,
    ScenarioSession,
    ScenarioSpec,
    ScenarioUpdate,
    ScenarioWorkspace,
)
from clasharness.tools.agent_tool_spec import validate_task_solution_commands

_SAFE_PATH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/\-]*$")
_ANCHOR_RE = re.compile(r"\b(?:DECISION|RULE|UPDATE|EVIDENCE|FILE|NOTE|LOG|SUMMARY|CHECKLIST):[A-Z0-9_]+\b")
_FUTURE_ROUND_REF_RE = re.compile(r"\b(?:ROUND|R)\s*(?:0?[8-9]|1[0-9]|2[0-9])\b", re.IGNORECASE)
_FUTURE_COMPACT_REF_RE = re.compile(r"\b(?:q|r|round)[_\- ]?(?:0?[8-9]|1[0-9]|2[0-9])\b", re.IGNORECASE)
_FILE_TOKEN_RE = re.compile(
    r"\b[A-Za-z0-9][A-Za-z0-9._\-]*\."
    r"(?:md|markdown|jsonl|eml|log|csv|tsv|txt|json|yaml|yml|pdf|docx?|xlsx?|ics)\b",
    re.IGNORECASE,
)
_EARLY_FORBIDDEN_RUNTIME_MARKERS = (
    "DEPRECATION NOTICE",
    "APPENDED ROUND",
    "ROUND 24",
    "POST-LAUNCH",
    "SUPERSEDED",
    "DO NOT TREAT",
    "RECONCILIATION NOTICE",
    "CANONICAL DECISION",
    "GOVERNING RULE",
    "AUTHORITATIVE SOURCES",
    "SUPERSEDING",
    "NEWER OWNER",
    "EVIDENCE LOG",
    "HAVE SINCE",
    "DEPRECATED",
    "GATED LAUNCH",
    "GATED —",
    "GATED -",
    "NOT READY TO PROCEED",
    "LAUNCH IS NOT READY",
    "WHY THIS DOCUMENT IS STALE",
    "FORMAL WITHDRAWAL",
    "CORRECT LAUNCH STATUS",
    "FIRST FORMAL OVERRIDE",
    "STRONGEST EVIDENTIARY ANCHOR",
    "MANDATORY AND NOT OBTAINED",
    "HOLD-LIFT",
    "HOLD LIFT",
    "FINAL APPROVAL",
    "FINAL HOLD",
    "FINAL OWNER",
    "FINAL AUDIT",
    "LATE OWNER",
    "COMPLIANCE HOLD",
    "DO NOT ACT",
    "BEFORE COMPLIANCE FLAGS",
    "RETAINED FOR AUDIT",
    "AUDIT AND RECONCILIATION",
)
_PRO_TOOL_FAMILIES = ["file", "email", "tasks", "calendar"]


def materialize_generated_scenario(
    seed: ScenarioSeed,
    layers: dict[str, Any],
    *,
    scenario_series_dir: Path,
    output_data_dir: Path,
    clean: bool = True,
) -> ScenarioSpec:
    """Write a generated scenario package and compiled runnable tasks."""

    scenario_dir = scenario_series_dir / "pro_scenario_workflow" / seed.slug
    if clean and scenario_dir.exists():
        shutil.rmtree(scenario_dir)
    (scenario_dir / "workspace").mkdir(parents=True, exist_ok=True)
    (scenario_dir / "sessions").mkdir(parents=True, exist_ok=True)
    (scenario_dir / "updates").mkdir(parents=True, exist_ok=True)
    (scenario_dir / "internal").mkdir(parents=True, exist_ok=True)

    narrative = dict(layers["layer0_narrative"])
    workspace_layer = dict(layers["layer1_workspace"])
    session_layer = dict(layers["layer2_sessions"])
    eval_layer = dict(layers["layer3_eval"])
    dynamic_layer = dict(layers["layer4_dynamic"])

    workspace_entries: list[dict[str, Any]] = []
    base_workspace_content: dict[str, str] = {}
    workspace_id_to_path: dict[str, str] = {}
    workspace_files = list(workspace_layer.get("files", []))
    canonical_ids = [
        _evidence_string(narrative.get("canonical_decision_id")),
        _evidence_string(narrative.get("canonical_rule_id")),
    ]
    for item in workspace_files:
        file_id = str(item.get("id") or item.get("path") or "")
        rel_path = _safe_rel_path(str(item.get("path") or file_id))
        content = str(item.get("content", ""))
        available_from_round = _availability_for_index(
            len(workspace_entries) + 1,
            total=len(workspace_files) or 1,
        )
        available_from_round = _availability_not_before_path_hint(available_from_round, rel_path)
        content = _ensure_anchor_text(
            content,
            _anchors_for_stage(
                available_from_round,
                [item for item in canonical_ids if item],
                prefix=f"FILE:{len(workspace_entries) + 1:02d}_VISIBLE",
            ),
        )
        _write_text(scenario_dir / "workspace" / rel_path, content)
        base_workspace_content[rel_path] = content
        if file_id:
            workspace_id_to_path[file_id] = rel_path
        workspace_entries.append(
            {
                "id": file_id or rel_path,
                "path": rel_path,
                "materialized_path": f"workspace/{rel_path}",
                "source": "llm_layer1_workspace",
                "substrate": "file",
                "available_from_round": available_from_round,
                "anchor_ids": _extract_anchor_ids(content),
            }
        )
    canonical_content = _canonical_evidence_content(narrative, dynamic_layer)
    _write_text(scenario_dir / "internal" / "canonical_evidence.md", canonical_content)

    sessions: list[ScenarioSession] = []
    session_content: dict[str, str] = {}
    for item in session_layer.get("sessions", []):
        session_id = str(item.get("session_id") or item.get("id") or "")
        if not session_id:
            raise ValueError("session is missing session_id")
        session_path = f"sessions/{session_id}.jsonl"
        messages = list(item.get("messages", []))
        lines = [json.dumps(message, ensure_ascii=False) for message in messages]
        content = "\n".join(lines) + ("\n" if lines else "")
        _write_text(scenario_dir / session_path, content)
        session_content[session_id] = content
        sessions.append(
            ScenarioSession(
                session_id=session_id,
                kind=str(item.get("kind", "chat")),
                channel=str(item.get("channel", "default")),
                path=session_path,
                metadata={k: v for k, v in item.items() if k not in {"messages"}},
            )
        )

    updates: list[ScenarioUpdate] = []
    update_content_by_id: dict[str, dict[str, str]] = {}
    for item in dynamic_layer.get("updates", []):
        update_id = str(item.get("update_id") or item.get("id") or "")
        if not update_id:
            raise ValueError("update is missing update_id")
        workspace_adds: list[dict[str, Any]] = []
        update_files: dict[str, str] = {}
        for add in item.get("workspace_adds", []):
            file_id = str(add.get("id") or add.get("path") or "")
            rel_path = _safe_rel_path(str(add.get("path") or file_id))
            content = str(add.get("content", ""))
            before_round = _availability_not_before_path_hint(
                str(item.get("before_round") or "q99"),
                rel_path,
            )
            item["before_round"] = before_round
            content = _ensure_anchor_text(
                content,
                [
                    str(item.get("anchor_id") or _update_anchor_id(update_id)),
                    *[anchor for anchor in canonical_ids if anchor],
                ],
            )
            materialized_path = f"updates/{update_id}/workspace/{rel_path}"
            _write_text(scenario_dir / materialized_path, content)
            update_files[rel_path] = content
            workspace_adds.append(
                {
                    "id": file_id or rel_path,
                    "path": rel_path,
                    "materialized_path": materialized_path,
                }
            )
        session_appends: list[dict[str, Any]] = []
        for append in item.get("session_appends", []):
            session_id = str(append.get("session_id", ""))
            messages = list(append.get("messages", []))
            lines = [json.dumps(message, ensure_ascii=False) for message in messages]
            content = "\n".join(lines) + ("\n" if lines else "")
            append_path = f"updates/{update_id}/sessions/{session_id}.jsonl"
            _write_text(scenario_dir / append_path, content)
            update_files[f"sessions/{session_id}.jsonl"] = content
            session_appends.append(
                {"session_id": session_id, "materialized_path": append_path}
            )
        update_content_by_id[update_id] = update_files
        updates.append(
            ScenarioUpdate(
                update_id=update_id,
                before_round=item.get("before_round"),
                anchor_id=str(item.get("anchor_id") or _update_anchor_id(update_id)),
                workspace_adds=workspace_adds,
                session_appends=session_appends,
                metadata={k: v for k, v in item.items() if k not in {"workspace_adds", "session_appends"}},
            )
        )

    evidence_bundle = _build_evidence_bundle(
        seed=seed,
        narrative=narrative,
        workspace_entries=workspace_entries,
        workspace_content=base_workspace_content,
        sessions=sessions,
        session_content=session_content,
        updates=updates,
        update_content_by_id=update_content_by_id,
    )

    update_ids_by_round = _active_update_ids_by_round(eval_layer.get("rounds", []), updates)
    rounds: list[ScenarioRound] = []
    eval_items: list[ScenarioEvalItem] = []
    tasks: list[Task] = []
    for round_item in eval_layer.get("rounds", []):
        round_id = str(round_item.get("round_id", ""))
        if not round_id:
            raise ValueError("round is missing round_id")
        active_update_ids = update_ids_by_round.get(round_id, [])
        task = _compile_round_to_task(
            seed,
            narrative,
            round_item,
            evidence_bundle,
            workspace_id_to_path,
            update_content_by_id,
            active_update_ids,
        )
        tasks.append(task)
        rounds.append(
            ScenarioRound(
                round_id=round_id,
                task_id=task.task_id,
                instruction=task.instruction,
                task_spec=task.to_dict(),
                update_ids=active_update_ids,
                session_refs=[str(v) for v in round_item.get("session_refs", [])],
                workspace_refs=[str(v) for v in round_item.get("workspace_refs", [])],
                visible_refs=[str(v) for v in round_item.get("visible_refs", [])],
                expected_anchor_ids=list(task.ground_truth.answer.get("expected_anchor_ids", []))
                if task.ground_truth
                else [],
                expert_trajectory=task.ground_truth.solution_commands if task.ground_truth else [],
                metadata={k: v for k, v in round_item.items() if k not in {"question"}},
            )
        )

    eval_by_round = {str(item.get("round_id")): item for item in eval_layer.get("eval_items", [])}
    for round_item in eval_layer.get("rounds", []):
        round_id = str(round_item.get("round_id", ""))
        item = eval_by_round.get(round_id, {})
        eval_items.append(
            ScenarioEvalItem(
                eval_id=str(item.get("eval_id") or f"eval_{round_id}"),
                round_id=round_id,
                kind=str(item.get("kind", "task_checks")),
                checks=next(task for task in tasks if task.task_id.endswith(round_id)).ground_truth.evaluation_checks,
                answer=round_item.get("answer"),
                metadata={
                    "required_evidence": _evidence_strings(
                        item.get("required_evidence") or round_item.get("required_evidence") or []
                    ),
                    "required_evidence_ids": list(
                        next(task for task in tasks if task.task_id.endswith(round_id))
                        .ground_truth.answer.get("expected_anchor_ids", [])
                    ),
                },
            )
        )

    spec = ScenarioSpec(
        scenario_id=f"pro_scenario_workflow:{seed.slug}",
        generator_id="pro_scenario_workflow",
        scenario_slug=seed.slug,
        title=seed.title,
        description=str(narrative.get("summary", "")),
        version="pro_llm_v1",
        narrative=narrative,
        workspace=ScenarioWorkspace(files=workspace_entries, notes=dict(workspace_layer.get("notes", {}))),
        sessions=sessions,
        updates=updates,
        rounds=rounds,
        eval_items=eval_items,
        manifest=ScenarioManifest(
            source_generator="pro_scenario_workflow",
            compiled_task_ids=[task.task_id for task in tasks],
            created_by="clasharness-llm-pipeline",
            metadata={"series": "pro_llm_v1", "seed": seed.to_dict()},
        ),
        metadata={"scenario_series": "pro_llm_v1", "seed": seed.to_dict()},
    )
    from clasharness.scenario.consistency_layer import audit_and_repair_scenario
    spec = audit_and_repair_scenario(spec)
    tasks = [Task.from_dict(round_item.task_spec) for round_item in spec.rounds]

    command_errors = validate_task_solution_commands(
        tasks,
        allowed_families=sorted(set(seed.allowed_tool_families) | set(_PRO_TOOL_FAMILIES)),
    )
    if command_errors:
        raise ValueError(
            "Generated PRO task commands violate AgentToolSpec: "
            + "; ".join(command_errors[:10])
        )
    spec.write_json(scenario_dir / "scenario.json")
    _write_tasks_and_splits(tasks, output_data_dir)
    return spec


def _compile_round_to_task(
    seed: ScenarioSeed,
    narrative: dict[str, Any],
    round_item: dict[str, Any],
    evidence_bundle: dict[str, Any],
    workspace_id_to_path: dict[str, str],
    update_content_by_id: dict[str, dict[str, str]],
    active_update_ids: list[str],
) -> Task:
    round_id = str(round_item["round_id"])
    round_index = _round_index(round_id)
    report_path = str(round_item.get("answer", {}).get("report_path") or f"/reports/{seed.slug}_{round_id}.md")
    evidence = _evidence_strings(round_item.get("required_evidence", []))
    requested_anchor_ids = _normalize_expected_anchor_ids(
        narrative,
        round_item.get("answer", {}).get("expected_anchor_ids")
        or round_item.get("required_evidence_ids")
        or [],
        active_update_ids=active_update_ids,
    )
    truth_anchor = _evidence_string(
        round_item.get("answer", {}).get("truth_anchor")
        or narrative.get("truth_anchor")
        or (evidence[0] if evidence else "FINAL_EVIDENCE")
    )
    preference_rule = _evidence_string(
        round_item.get("answer", {}).get("preference_rule")
        or narrative.get("preference_rule")
        or (evidence[1] if len(evidence) > 1 else "prefer newer direct evidence")
    )
    if not requested_anchor_ids:
        requested_anchor_ids = [
            _evidence_string(narrative.get("canonical_decision_id")),
            _evidence_string(narrative.get("canonical_rule_id")),
        ]
    for update_id in active_update_ids:
        requested_anchor_ids.append(_update_anchor_id(update_id))
    requested_anchor_ids = list(
        dict.fromkeys(item for item in requested_anchor_ids if item and not _is_weak_anchor_id(item))
    )
    task_id = f"pro_llm_{seed.slug}_{round_id}"

    visible = _visible_evidence_for_round(evidence_bundle, round_index, active_update_ids)
    overrides = _state_overrides_for_visible_evidence(visible, round_index=round_index)
    visible_anchor_ids = _runtime_anchor_ids(overrides)
    round_type = _round_type(round_index)
    if round_type == "observation":
        expected_anchor_ids = _observation_expected_anchor_ids(overrides, round_index)
    else:
        expected_anchor_ids = [item for item in requested_anchor_ids if item in visible_anchor_ids]
    if not expected_anchor_ids:
        expected_anchor_ids = _fallback_expected_anchor_ids(visible_anchor_ids, round_index)

    visible_workspace_paths = {str(item["path"]) for item in visible["files"]}
    workspace_ref_text = _readable_workspace_refs(
        round_item.get("workspace_refs", []),
        workspace_id_to_path,
        visible_workspace_paths,
    )
    visible_file_text = ", ".join(f"/{item['path']}" for item in visible["files"]) or "none"
    visible_email_text = ", ".join(str(item["id"]) for item in visible["emails"]) or "none"
    allowed_tools = ", ".join(_PRO_TOOL_FAMILIES)
    round_prompt = _sanitize_round_prompt(
        str(round_item.get("question", "")),
        round_index=round_index,
    )
    round_prompt = _sanitize_prompt_for_round_type(round_prompt, round_type)
    evidence_read_commands = _evidence_read_commands(
        round_item,
        workspace_id_to_path,
        visible,
        active_update_ids,
        update_content_by_id,
    )
    action_instruction, action_commands, checks = _mixed_action_contract(
        seed=seed,
        round_id=round_id,
        round_index=round_index,
        round_type=round_type,
        report_path=report_path,
        expected_anchor_ids=expected_anchor_ids,
        rationale_summary=str(
            round_item.get("answer", {}).get("rationale_summary")
            or "resolved from visible evidence"
        ),
    )
    instruction = (
        f"Submission requirement: {action_instruction} `DONE` before completing that operation is invalid. "
        f"Resolve PRO scenario '{seed.title}' round {round_id}. "
        f"Round type: {round_type}. Allowed tool families: {allowed_tools}. "
        "Use file, email, tasks, and calendar commands to inspect only the currently visible state. "
        f"Currently visible files are: {visible_file_text}. Currently visible scenario emails are: {visible_email_text}. "
        "If the round prompt names a file, session, or email that is not currently visible, ignore that future reference. "
        "Reconcile conflicts using timestamps, ownership, and provenance in the evidence. "
        f"Prioritize workspace refs {workspace_ref_text}, but only when they are visible now. "
        "Your operation must cite the short evidence anchor ids that appear in the available evidence. "
        f"Round prompt: {round_prompt}"
    )
    solution_commands = [*evidence_read_commands, *action_commands, "done"]
    return Task(
        task_id=task_id,
        instruction=instruction,
        domains=["composite", "file", "email", "tasks", "calendar"],
        difficulty=int(Difficulty.REASONING),
        initial_state="default",
        template_id="pro_llm_v1_round",
        generator_id="pro_scenario_workflow",
        data=TaskData(
            public={
                "scenario_series": "pro_llm_v1",
                "scenario_slug": seed.slug,
                "round_id": round_id,
                "round_kind": round_item.get("kind", "ec"),
                "round_type": round_type,
                "workspace_hint": "Use file, email, tasks, and calendar commands; only round-visible evidence is preloaded.",
            },
            private={"initial_state_overrides": overrides},
        ),
        ground_truth=GroundTruth(
            solution_commands=solution_commands,
            evaluation_checks=checks,
            answer={
                "truth_anchor": truth_anchor,
                "preference_rule": preference_rule,
                "expected_anchor_ids": expected_anchor_ids,
            },
            metadata={
                "scenario_slug": seed.slug,
                "round_id": round_id,
                "expected_anchor_ids": expected_anchor_ids,
            },
        ),
        decision_requirements=["reconcile_conflicting_evidence", "apply_preference_rule"],
        realism_tags=["pro_llm_v1", "workspace_evidence", "state_conflict"],
    )


def _evidence_read_commands(
    round_item: dict[str, Any],
    workspace_id_to_path: dict[str, str],
    visible: dict[str, Any],
    active_update_ids: list[str],
    update_content_by_id: dict[str, dict[str, str]],
) -> list[str]:
    commands = ["file list --path /"]
    for path in _visible_workspace_paths_for_round(round_item, workspace_id_to_path, visible["files"]):
        commands.append(f"file read --path /{path}")
    for email in visible["emails"][:2]:
        commands.append(f"email read --id {email['id']}")
    if visible["tasks"]:
        commands.append("tasks list --status all")
    if visible["calendar_events"]:
        commands.append("calendar list")
    for path in _update_workspace_paths_for_round(active_update_ids, update_content_by_id):
        if path in visible["update_files"]:
            commands.append(f"file read --path /{path}")
    return commands


def _mixed_action_contract(
    *,
    seed: ScenarioSeed,
    round_id: str,
    round_index: int,
    round_type: str,
    report_path: str,
    expected_anchor_ids: list[str],
    rationale_summary: str,
) -> tuple[str, list[str], list[dict[str, Any]]]:
    anchor_text = _anchor_phrase(expected_anchor_ids)
    safe_round_id = round_id.lower()
    if round_type == "observation":
        title = f"PRO {seed.slug} {safe_round_id} current evidence review | {anchor_text}"
        instruction = (
            "record the current blocker/review state on the task board with `tasks add`; "
            "include visible evidence anchor ids in the task title; do not create a report file."
        )
        commands = [
            f"tasks add --title {json.dumps(title)} --due 2026-05-05 --priority high",
            "tasks list --status all",
        ]
        checks = [
            _effect_exists("tasks_created", "task board entry created"),
            *_effect_anchor_checks("tasks_created", "title", expected_anchor_ids),
            *_output_anchor_checks("last_stdout", expected_anchor_ids),
        ]
    elif round_type == "reconciliation":
        mode = round_index % 4
        if mode == 0:
            subject = f"PRO {seed.slug} {safe_round_id} reconciliation path"
            body = (
                "Use the gated/review path rather than stale launch evidence. "
                f"{anchor_text}. Rationale: {rationale_summary}"
            )
            instruction = (
                "notify the responsible owner/team with `email send --to owner@example.com --subject ... --body ...`; "
                "include visible evidence anchor ids in the email body; do not create a report file."
            )
            commands = [
                f"email send --to owner@example.com --subject {json.dumps(subject)} --body {json.dumps(body)}",
            ]
            checks = [
                _effect_field_contains("emails_sent", "subject", "reconciliation", "reconciliation email sent"),
                *_effect_anchor_checks("emails_sent", "body", expected_anchor_ids),
            ]
        elif mode == 1:
            title = f"PRO {seed.slug} {safe_round_id} reconciliation follow-up | {anchor_text}"
            instruction = (
                "record the correct reconciliation follow-up with `tasks add --title ...`; "
                "include visible evidence anchor ids in the task title; do not create a report file."
            )
            commands = [
                f"tasks add --title {json.dumps(title)} --due 2026-05-06 --priority high",
                "tasks list --status all",
            ]
            checks = [
                _effect_exists("tasks_created", "reconciliation task created"),
                *_effect_anchor_checks("tasks_created", "title", expected_anchor_ids),
            ]
        elif mode == 2:
            title = f"PRO {seed.slug} {safe_round_id} reconciliation review | {anchor_text}"
            instruction = (
                "schedule the reconciliation review with `calendar add-event --title ... --start ...`; "
                "include visible evidence anchor ids in the event title; do not create a report file."
            )
            commands = [
                f"calendar add-event --title {json.dumps(title)} --start 2026-05-06T09:00:00 --location ReviewRoom",
                "calendar list",
            ]
            checks = [
                _effect_exists("calendar_events_created", "reconciliation calendar event created"),
                *_effect_anchor_checks("calendar_events_created", "title", expected_anchor_ids),
            ]
        else:
            action_path = f"/actions/{seed.slug}_{safe_round_id}_reconciliation.md"
            content = (
                f"Reconciliation action note for {seed.slug} {round_id}.\n"
                f"{anchor_text}.\n"
                f"Rationale summary: {rationale_summary}."
            )
            instruction = (
                f"create the reconciliation action note at `{action_path}` with `file create --path ... --content ...`; "
                "include visible evidence anchor ids; do not create a report file."
            )
            commands = [
                f"file create --path {action_path} --content {json.dumps(content)}",
                f"file read --path {action_path}",
            ]
            checks = [
                _effect_field_equals("files_created", "path", action_path, "reconciliation action note created"),
                *_effect_anchor_checks("files_created", "content", expected_anchor_ids),
                _effect_field_contains("files_created", "content", "Evidence anchors", "action note cites evidence anchors"),
            ]
    elif round_type == "update_aware":
        mode = round_index % 3
        if mode == 0:
            action_path = f"/actions/{seed.slug}_{safe_round_id}_update.md"
            content = (
                f"Update-aware action log for {seed.slug} {round_id}.\n"
                f"{anchor_text}.\n"
                f"Rationale summary: {rationale_summary}."
            )
            instruction = (
                f"create or update the decision action log at `{action_path}` with `file create --path ... --content ...`; "
                "include visible evidence anchor ids, read the action log back, and do not create a report file."
            )
            commands = [
                f"file create --path {action_path} --content {json.dumps(content)}",
                f"file read --path {action_path}",
            ]
            checks = [
                _effect_field_equals("files_created", "path", action_path, "update action log created"),
                *_effect_anchor_checks("files_created", "content", expected_anchor_ids),
                _effect_field_contains("files_created", "content", "Evidence anchors", "action log cites evidence anchors"),
            ]
        elif mode == 1:
            title = f"PRO {seed.slug} {safe_round_id} update repair | {anchor_text}"
            instruction = (
                "record the update-aware repair item with `tasks add --title ...`; "
                "include visible evidence anchor ids in the task title; do not create a report file."
            )
            commands = [
                f"tasks add --title {json.dumps(title)} --due 2026-05-07 --priority high",
                "tasks list --status all",
            ]
            checks = [
                _effect_exists("tasks_created", "update repair task created"),
                *_effect_anchor_checks("tasks_created", "title", expected_anchor_ids),
            ]
        else:
            subject = f"PRO {seed.slug} {safe_round_id} update correction"
            body = f"Apply the current update-aware correction. {anchor_text}. Rationale: {rationale_summary}"
            instruction = (
                "send the update-aware correction with `email send --to owner@example.com --subject ... --body ...`; "
                "include visible evidence anchor ids in the email body; do not create a report file."
            )
            commands = [
                f"email send --to owner@example.com --subject {json.dumps(subject)} --body {json.dumps(body)}",
            ]
            checks = [
                _effect_field_contains("emails_sent", "subject", "update", "update correction email sent"),
                *_effect_anchor_checks("emails_sent", "body", expected_anchor_ids),
            ]
    else:
        content = (
            f"Decision report for {seed.slug} {round_id}.\n"
            f"{anchor_text}.\n"
            f"Rationale summary: {rationale_summary}."
        )
        instruction = (
            f"create the final handoff report at `{report_path}` with `file create`, "
            "include visible evidence anchor ids, then read it back with `file read`."
        )
        commands = [
            f"file create --path {report_path} --content {json.dumps(content)}",
            f"file read --path {report_path}",
        ]
        checks = [
            _effect_field_equals("files_created", "path", report_path, "final report file created"),
            *_effect_anchor_checks("files_created", "content", expected_anchor_ids),
            _effect_field_contains("files_created", "content", "Evidence anchors", "report cites evidence anchors"),
        ]
    return instruction, commands, checks


def _anchor_phrase(anchor_ids: list[str]) -> str:
    return f"Evidence anchors: {', '.join(anchor_ids)}"


def _effect_field_equals(effect_type: str, field: str, value: str, name: str) -> dict[str, Any]:
    return {
        "type": "effect",
        "effect_type": effect_type,
        "condition": "field_equals",
        "expected": {"field": field, "value": value},
        "name": name,
        "weight": 1.0,
    }


def _effect_exists(effect_type: str, name: str) -> dict[str, Any]:
    return {
        "type": "effect",
        "effect_type": effect_type,
        "condition": "exists",
        "expected": None,
        "name": name,
        "weight": 1.0,
    }


def _effect_field_contains(effect_type: str, field: str, value: str, name: str) -> dict[str, Any]:
    return {
        "type": "effect",
        "effect_type": effect_type,
        "condition": "field_contains",
        "expected": {"field": field, "value": value},
        "name": name,
        "weight": 1.0,
    }


def _effect_anchor_checks(effect_type: str, field: str, anchor_ids: list[str]) -> list[dict[str, Any]]:
    return [
        _effect_field_contains(effect_type, field, anchor_id, f"anchor cited in {effect_type}: {anchor_id}")
        for anchor_id in anchor_ids
    ]


def _output_contains(output_field: str, expected: str, name: str) -> dict[str, Any]:
    return {
        "type": "output",
        "output_field": output_field,
        "condition": "contains",
        "expected": expected,
        "name": name,
        "weight": 1.0,
    }


def _output_anchor_checks(output_field: str, anchor_ids: list[str]) -> list[dict[str, Any]]:
    return [
        _output_contains(output_field, anchor_id, f"anchor cited: {anchor_id}")
        for anchor_id in anchor_ids
    ]


def _active_update_ids_by_round(rounds: list[dict[str, Any]], updates: list[ScenarioUpdate]) -> dict[str, list[str]]:
    round_ids = [str(item.get("round_id", "")) for item in rounds]
    active: list[str] = []
    out: dict[str, list[str]] = {}
    by_before: dict[str, list[str]] = {}
    for update in updates:
        if update.before_round:
            by_before.setdefault(update.before_round, []).append(update.update_id)
    for round_id in round_ids:
        active.extend(by_before.get(round_id, []))
        out[round_id] = list(active)
    return out


def _build_evidence_bundle(
    *,
    seed: ScenarioSeed,
    narrative: dict[str, Any],
    workspace_entries: list[dict[str, Any]],
    workspace_content: dict[str, str],
    sessions: list[ScenarioSession],
    session_content: dict[str, str],
    updates: list[ScenarioUpdate],
    update_content_by_id: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Derive round-visible evidence substrates from generated scenario layers."""

    decision_id = _evidence_string(narrative.get("canonical_decision_id"))
    rule_id = _evidence_string(narrative.get("canonical_rule_id"))
    canonical_ids = [item for item in (decision_id, rule_id) if item]

    files: list[dict[str, Any]] = []
    for index, entry in enumerate(workspace_entries, start=1):
        path = str(entry["path"])
        available_from_round = str(entry.get("available_from_round") or _availability_for_index(index, total=len(workspace_entries)))
        available_from_round = _availability_not_before_path_hint(available_from_round, path)
        content = _ensure_anchor_text(
            workspace_content[path],
            _anchors_for_stage(available_from_round, canonical_ids, prefix=f"FILE:{index:02d}_VISIBLE"),
        )
        workspace_content[path] = content
        entry["anchor_ids"] = _extract_anchor_ids(content)
        entry["available_from_round"] = available_from_round
        files.append(
            {
                "id": str(entry["id"]),
                "path": path,
                "content": content,
                "available_from_round": available_from_round,
                "anchor_ids": entry["anchor_ids"],
            }
        )

    emails: list[dict[str, Any]] = []
    # Derive the evidence-mail host from the scenario seed rather than a fixed
    # product domain, so generated inboxes carry no harness brand name.
    _seed_host = re.sub(r"[^a-z0-9]+", "-", seed.slug.lower()).strip("-") or "scenario"
    sender_host = f"{_seed_host}.example.com"
    for index, session in enumerate(sessions, start=1):
        available_from_round = _availability_for_index(index, total=max(len(sessions), 1))
        body = _ensure_anchor_text(
            session_content.get(session.session_id, ""),
            _anchors_for_stage(available_from_round, canonical_ids, prefix=f"EVIDENCE:EMAIL_{index:02d}_VISIBLE"),
        )
        emails.append(
            {
                "id": f"pro_{session.session_id}",
                "sender": f"{session.channel}@{sender_host}",
                "to": "agent@example.com",
                "subject": f"{seed.title} evidence thread {session.session_id}",
                "body": body,
                "folder": "inbox",
                "read": False,
                "starred": index == 1,
                "available_from_round": available_from_round,
                "anchor_ids": _extract_anchor_ids(body),
            }
        )

    tasks = [
        {
            "id": "pro_task_gate_review",
            "title": "Review current evidence status EVIDENCE:TASK_REVIEW_OPEN",
            "due": "2026-05-05",
            "status": "pending",
            "priority": "high",
            "available_from_round": "q01",
            "anchor_ids": ["EVIDENCE:TASK_REVIEW_OPEN"],
        },
        {
            "id": "pro_task_precedence_rule",
            "title": f"Apply provenance precedence {rule_id or 'RULE:PROVENANCE_REQUIRED'}",
            "due": "2026-05-06",
            "status": "pending",
            "priority": "high",
            "available_from_round": "q08",
            "anchor_ids": [rule_id] if rule_id else [],
        },
    ]

    calendar_events = [
        {
            "id": "pro_cal_review_window",
            "title": "Evidence review window EVIDENCE:CAL_REVIEW_WINDOW",
            "start": "2026-05-04T09:00:00",
            "end": "2026-05-04T10:00:00",
            "location": "Launch review board",
            "attendees": ["ops@example.com", "compliance@example.com"],
            "available_from_round": "q01",
            "anchor_ids": ["EVIDENCE:CAL_REVIEW_WINDOW"],
        },
        {
            "id": "pro_cal_update_cutoff",
            "title": f"Final update cutoff {decision_id or 'DECISION:FINAL_REVIEW'}",
            "start": "2026-05-04T16:00:00",
            "end": "2026-05-04T16:30:00",
            "location": "Release room",
            "attendees": ["owner@example.com"],
            "available_from_round": "q21",
            "anchor_ids": [decision_id] if decision_id else [],
        },
    ]

    update_files: list[dict[str, Any]] = []
    for update in updates:
        for path, content in update_content_by_id.get(update.update_id, {}).items():
            if path.startswith("sessions/"):
                continue
            anchored_content = _ensure_anchor_text(content, [update.anchor_id, *canonical_ids])
            update_content_by_id[update.update_id][path] = anchored_content
            update_files.append(
                {
                    "id": update.update_id,
                    "path": path,
                    "content": anchored_content,
                    "available_from_round": update.before_round or "q99",
                    "anchor_ids": _extract_anchor_ids(anchored_content),
                }
            )

    return {
        "files": files,
        "emails": emails,
        "tasks": tasks,
        "calendar_events": calendar_events,
        "update_files": update_files,
    }


def _visible_evidence_for_round(
    evidence_bundle: dict[str, Any],
    round_index: int,
    active_update_ids: list[str],
) -> dict[str, Any]:
    visible: dict[str, Any] = {}
    for key in ("files", "emails", "tasks", "calendar_events"):
        visible[key] = [
            item
            for item in evidence_bundle.get(key, [])
            if _round_index(str(item.get("available_from_round") or "q01")) <= round_index
        ]
    active = set(active_update_ids)
    visible["update_files"] = {
        str(item["path"]): item
        for item in evidence_bundle.get("update_files", [])
        if str(item.get("id")) in active
        and _round_index(str(item.get("available_from_round") or "q99")) <= round_index
    }
    return visible


def _visible_anchor_ids(visible: dict[str, Any]) -> list[str]:
    anchors: list[str] = []
    for key in ("files", "emails", "tasks", "calendar_events"):
        for item in visible.get(key, []):
            anchors.extend(str(anchor) for anchor in item.get("anchor_ids", []) if anchor)
            if "content" in item:
                anchors.extend(_extract_anchor_ids(str(item.get("content", ""))))
            if key == "tasks":
                anchors.extend(_extract_anchor_ids(str(item.get("title", ""))))
            if key == "calendar_events":
                anchors.extend(_extract_anchor_ids(str(item.get("title", ""))))
    for item in visible.get("update_files", {}).values():
        anchors.extend(str(anchor) for anchor in item.get("anchor_ids", []) if anchor)
        anchors.extend(_extract_anchor_ids(str(item.get("content", ""))))
    return list(dict.fromkeys(anchor for anchor in anchors if anchor and not _is_weak_anchor_id(anchor)))


def _runtime_anchor_ids(overrides: dict[str, Any]) -> list[str]:
    """Extract expected-check candidates from the actual agent-visible state."""

    corpus = json.dumps(overrides, ensure_ascii=False, sort_keys=True)
    return [
        anchor
        for anchor in _extract_anchor_ids(corpus)
        if anchor and not _is_weak_anchor_id(anchor)
    ]


def _state_overrides_for_visible_evidence(visible: dict[str, Any], *, round_index: int) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    visible_paths = {str(item["path"]) for item in visible.get("files", [])}
    visible_paths.update(str(path) for path in visible.get("update_files", {}))
    for item in visible.get("files", []):
        content = str(item.get("content", ""))
        if _is_index_path(str(item["path"])):
            content = _round_visible_index_content(visible, round_index=round_index)
        overrides[f"workspace/{item['path']}"] = _mask_future_evidence(
            content,
            round_index=round_index,
            visible_paths=visible_paths,
        )
    for path, item in visible.get("update_files", {}).items():
        overrides[f"workspace/{path}"] = _mask_future_evidence(
            str(item.get("content", "")),
            round_index=round_index,
            visible_paths=visible_paths,
        )
    overrides["emails.json"] = {
        "replace_seeded": True,
        "emails": [
        _mask_email_payload(
            {k: v for k, v in item.items() if k not in {"available_from_round", "anchor_ids"}},
            round_index=round_index,
            visible_paths=visible_paths,
        )
        for item in visible.get("emails", [])
        ],
    }
    overrides["tasks.json"] = {
        "replace_seeded": True,
        "tasks": [
            _mask_runtime_payload(
                {k: v for k, v in item.items() if k not in {"available_from_round", "anchor_ids"}},
                round_index=round_index,
                visible_paths=visible_paths,
            )
            for item in visible.get("tasks", [])
        ],
    }
    overrides["calendar_events.json"] = {
        "events": [
            _mask_runtime_payload(
                {k: v for k, v in item.items() if k not in {"available_from_round", "anchor_ids"}},
                round_index=round_index,
                visible_paths=visible_paths,
            )
            for item in visible.get("calendar_events", [])
        ],
    }
    return overrides


def _write_tasks_and_splits(tasks: list[Task], output_data_dir: Path) -> None:
    for task in tasks:
        save_task(task, data_dir=output_data_dir)
    datasets_dir = output_data_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    tasks_dir = output_data_dir / "tasks"
    task_ids = sorted(
        path.name
        for path in tasks_dir.iterdir()
        if path.is_dir() and (path / "specs.json").exists()
    )
    for split in ("train", "dev", "test", "total"):
        (datasets_dir / f"{split}.txt").write_text("\n".join(task_ids) + "\n", encoding="utf-8")


def _safe_rel_path(path: str) -> str:
    rel = path.strip().lstrip("/")
    if not rel or ".." in rel.split("/") or not _SAFE_PATH_RE.match(rel):
        raise ValueError(f"Unsafe generated path: {path!r}")
    return rel


def _dedupe_rel_path(path: str, existing: set[str]) -> str:
    rel = _safe_rel_path(path)
    if rel not in existing:
        return rel
    stem, suffix = (rel.rsplit(".", 1) + [""])[:2] if "." in rel else (rel, "")
    for index in range(2, 100):
        candidate = f"{stem}_{index}.{suffix}" if suffix else f"{stem}_{index}"
        if candidate not in existing:
            return candidate
    raise ValueError(f"Could not dedupe generated path: {path!r}")


def _canonical_evidence_content(
    narrative: dict[str, Any],
    dynamic_layer: dict[str, Any],
) -> str:
    decision_id = _evidence_string(narrative.get("canonical_decision_id"))
    rule_id = _evidence_string(narrative.get("canonical_rule_id"))
    lines = [
        "# Canonical Evidence Anchors",
        "",
        "This file is generated by the materializer so evaluator-facing anchors are exactly traceable.",
        "",
    ]
    if decision_id:
        lines.extend(
            [
                f"- {decision_id}",
                f"  - Decision summary: {_evidence_string(narrative.get('truth_anchor'))}",
            ]
        )
    if rule_id:
        lines.extend(
            [
                f"- {rule_id}",
                f"  - Rule summary: {_evidence_string(narrative.get('preference_rule'))}",
            ]
        )
    timeline = narrative.get("status_timeline")
    if isinstance(timeline, list) and timeline:
        lines.extend(["", "## Status Timeline"])
        for item in timeline:
            if isinstance(item, dict):
                lines.append(
                    f"- {item.get('from_round', 'q01')}: {item.get('anchor_id', '')} "
                    f"{item.get('summary', '')}".strip()
                )
    updates = dynamic_layer.get("updates", []) if isinstance(dynamic_layer, dict) else []
    if updates:
        lines.extend(["", "## Dynamic Update Anchors"])
    for item in updates:
        if not isinstance(item, dict):
            continue
        lines.append(f"- {item.get('anchor_id') or _update_anchor_id(str(item.get('update_id') or item.get('id') or ''))}")
    return "\n".join(lines).strip() + "\n"


def _readable_workspace_refs(
    refs: Any,
    id_to_path: dict[str, str],
    visible_paths: set[str],
) -> str:
    paths = []
    for ref in refs if isinstance(refs, list) else []:
        text = str(ref)
        path = id_to_path.get(text, text).lstrip("/")
        if path in visible_paths:
            paths.append(f"{text}->{path}")
    return ", ".join(paths) if paths else "visible workspace files"


def _visible_workspace_paths_for_round(
    round_item: dict[str, Any],
    id_to_path: dict[str, str],
    visible_files: list[dict[str, Any]],
) -> list[str]:
    visible_paths = {str(item["path"]) for item in visible_files}
    paths: list[str] = []
    for ref in round_item.get("workspace_refs", []):
        path = id_to_path.get(str(ref), str(ref).lstrip("/"))
        if path in visible_paths:
            paths.append(path)
    if not paths and visible_files:
        paths.append(str(visible_files[0]["path"]))
    return list(dict.fromkeys(paths))


def _update_workspace_paths_for_round(
    active_update_ids: list[str],
    update_content_by_id: dict[str, dict[str, str]],
) -> list[str]:
    paths: list[str] = []
    for update_id in active_update_ids:
        for rel_path in update_content_by_id.get(update_id, {}):
            if not rel_path.startswith("sessions/"):
                paths.append(rel_path)
    return list(dict.fromkeys(paths))


def _session_ids_for_round(
    round_item: dict[str, Any],
    session_content: dict[str, str],
) -> list[str]:
    ids = [str(ref) for ref in round_item.get("session_refs", []) if str(ref) in session_content]
    if not ids and session_content:
        ids.append(next(iter(session_content)))
    return list(dict.fromkeys(ids))


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _availability_for_index(index: int, *, total: int) -> str:
    """Spread base evidence over the 24-round PRO timeline."""

    if index <= max(1, total // 4):
        return "q01"
    if index <= max(2, total // 2):
        return "q08"
    if index <= max(3, (total * 3) // 4):
        return "q15"
    return "q21"


def _availability_not_before_path_hint(available_from_round: str, path: str) -> str:
    hint = _round_hint_from_path(path)
    current = _round_index(available_from_round)
    if hint and (current <= 0 or hint > current):
        return f"q{hint:02d}"
    return available_from_round


def _round_hint_from_path(path: str) -> int:
    text = str(path or "")
    for pattern in (
        r"(?:^|[._/\-])(?:q|r|round)[_\-]?(\d{1,2})(?:[._/\-]|$)",
        r"(?:^|[._/\-])round[_\- ]?(\d{1,2})(?:[._/\-]|$)",
    ):
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            try:
                value = int(match.group(1))
            except ValueError:
                continue
            if 1 <= value <= 24:
                return value
    return 0


def _anchors_for_stage(
    available_from_round: str,
    canonical_ids: list[str],
    *,
    prefix: str,
) -> list[str]:
    anchors = [prefix]
    round_idx = _round_index(available_from_round)
    if round_idx >= 8 and len(canonical_ids) >= 2:
        anchors.append(canonical_ids[1])
    if round_idx >= 15 and canonical_ids:
        anchors.append(canonical_ids[0])
    return list(dict.fromkeys(anchor for anchor in anchors if anchor))


def _extract_anchor_ids(text: str) -> list[str]:
    return list(dict.fromkeys(match.group(0) for match in _ANCHOR_RE.finditer(text or "")))


def _ensure_anchor_text(text: str, anchors: list[str]) -> str:
    content = str(text or "").strip()
    missing = [anchor for anchor in anchors if anchor and anchor not in content]
    if missing:
        content += "\n\nEvidence anchors: " + ", ".join(missing)
    return content


def _mask_future_evidence(
    content: str,
    *,
    round_index: int,
    visible_paths: set[str] | None = None,
) -> str:
    """Hide future-state hints from round-visible runtime state.

    Scenario packages retain the full generated artifact for audit, but the
    compiled task state must reflect what an agent could know at that round.
    """

    text = str(content or "")
    kept: list[str] = []
    skip_invisible_file_block = False
    for line in text.splitlines():
        file_heading = _workspace_file_heading_path(line)
        if file_heading is not None:
            skip_invisible_file_block = bool(visible_paths is not None and file_heading not in visible_paths)
            if skip_invisible_file_block:
                continue
        elif line.startswith("### "):
            skip_invisible_file_block = False
        if skip_invisible_file_block:
            continue
        if _line_leaks_future_state(line, round_index=round_index):
            continue
        if visible_paths is not None and _line_references_invisible_workspace_path(line, visible_paths):
            continue
        kept.append(line)
    masked = "\n".join(kept).strip()
    return masked + "\n" if masked else ""


def _sanitize_round_prompt(prompt: str, *, round_index: int) -> str:
    """Remove machine anchors/future-state hints from agent-visible prompts."""

    if not prompt:
        return ""
    kept: list[str] = []
    for line in str(prompt).splitlines():
        if _line_leaks_future_state(line, round_index=round_index):
            continue
        if _ANCHOR_RE.search(line):
            # Prompt text should use natural language; anchors must come from
            # visible evidence, not from the task instruction.
            continue
        kept.append(line)
    cleaned = " ".join(part.strip() for part in kept if part.strip())
    return cleaned or "Inspect the currently visible evidence and complete the required operation."


def _sanitize_prompt_for_round_type(prompt: str, round_type: str) -> str:
    """Align LLM-authored scenario prose with the compiled action contract."""

    text = str(prompt or "")
    if round_type != "action":
        text = re.sub(
            r"[^.?!]*(?:/reports/|report file|write report|write a report|final report)[^.?!]*[.?!]?",
            " ",
            text,
            flags=re.IGNORECASE,
        )
    if round_type == "observation":
        text = re.sub(
            r"\b(?:produce|write|draft|prepare|create)\s+(?:a\s+)?(?:written\s+)?(?:memo|report|status memo|compliance memo)\b",
            "record the current blocker state on the task board",
            text,
            flags=re.IGNORECASE,
        )
    elif round_type == "reconciliation":
        text = re.sub(
            r"\b(?:produce|write|draft|prepare|create)\s+(?:a\s+)?(?:written\s+)?(?:memo|report|status memo|compliance memo)\b",
            "send the responsible owner/team an email",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\bwritten\s+(?:compliance\s+)?status\s+memo\b",
            "owner notification email",
            text,
            flags=re.IGNORECASE,
        )
    elif round_type == "update_aware":
        text = re.sub(
            r"\b(?:produce|write|draft|prepare|create)\s+(?:a\s+)?(?:final\s+)?(?:memo|report|handoff)\b",
            "write the update action log",
            text,
            flags=re.IGNORECASE,
        )
    cleaned = " ".join(text.split())
    return cleaned or "Inspect the currently visible evidence and complete the required operation."


def _line_leaks_future_state(line: str, *, round_index: int = 0) -> bool:
    upper = str(line or "").upper()
    if round_index < 8 and ("DECISION:" in upper or "RULE:" in upper or "UPDATE:" in upper):
        return True
    if round_index > 0 and _line_references_future_round(upper, round_index=round_index):
        return True
    if round_index > 0:
        for match in re.finditer(r"\bROUND\s*(\d{1,2})\b", upper):
            if int(match.group(1)) > round_index:
                return True
    return round_index < 8 and any(marker in upper for marker in _EARLY_FORBIDDEN_RUNTIME_MARKERS)


def _line_references_future_round(line: str, *, round_index: int) -> bool:
    upper = str(line or "").upper()
    if "TOTAL ROUNDS" in upper or "TARGET ROUNDS" in upper:
        return False
    patterns = (
        r"\bQ(\d{1,2})\b",
        r"\bR(\d{1,2})\b",
        r"\bROUND[_\- ]?(\d{1,2})\b",
        r"\bR[_\- ]?(\d{1,2})\b",
        r"\bQ[_\- ]?(\d{1,2})\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, upper):
            try:
                value = int(match.group(1))
            except ValueError:
                continue
            if round_index < value <= 24:
                return True
    return False


def _workspace_file_heading_path(line: str) -> str | None:
    match = re.search(r"^###\s+f\d+\s+[—-]\s+`([^`]+)`", str(line or "").strip())
    return match.group(1).lstrip("/") if match else None


def _line_references_invisible_workspace_path(line: str, visible_paths: set[str]) -> bool:
    text = str(line or "")
    visible_names = {Path(path).name for path in visible_paths}
    for path in re.findall(r"`([^`]+\.[A-Za-z0-9]+)`", text):
        if path.lstrip("/") not in visible_paths:
            return True
    for token in _FILE_TOKEN_RE.findall(text):
        if token not in visible_names and token.lstrip("/") not in visible_paths:
            return True
    return False


def _is_index_path(path: str) -> bool:
    name = Path(str(path)).name.lower()
    return name in {"index.md", "workspace_index.md"} or name.endswith("_index.md")


def _round_visible_index_content(visible: dict[str, Any], *, round_index: int) -> str:
    """Use a minimal generated index so rounds do not inherit final-state summaries."""

    lines = [
        "# Round-Visible Workspace Index",
        "",
        f"This index lists only evidence currently visible in q{round_index:02d}.",
        "",
        "## Visible files",
    ]
    for item in visible.get("files", []):
        lines.append(f"- /{item['path']}")
    if visible.get("emails"):
        lines.extend(["", "## Visible emails"])
        for item in visible.get("emails", []):
            lines.append(f"- {item['id']}: {item.get('subject', 'scenario evidence')}")
    return "\n".join(lines).strip() + "\n"


def _mask_email_payload(
    email: dict[str, Any],
    *,
    round_index: int,
    visible_paths: set[str] | None = None,
) -> dict[str, Any]:
    out = dict(email)
    out["body"] = _mask_future_evidence(
        str(out.get("body", "")),
        round_index=round_index,
        visible_paths=visible_paths,
    )
    return out


def _mask_runtime_payload(
    value: Any,
    *,
    round_index: int,
    visible_paths: set[str] | None = None,
) -> Any:
    if isinstance(value, dict):
        return {
            key: _mask_runtime_payload(
                item,
                round_index=round_index,
                visible_paths=visible_paths,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _mask_runtime_payload(
                item,
                round_index=round_index,
                visible_paths=visible_paths,
            )
            for item in value
        ]
    if isinstance(value, str):
        masked = _mask_future_evidence(
            value,
            round_index=round_index,
            visible_paths=visible_paths,
        )
        return masked.strip() if "\n" not in value else masked
    return value


def _fallback_expected_anchor_ids(visible_anchor_ids: list[str], round_index: int) -> list[str]:
    if not visible_anchor_ids:
        return ["EVIDENCE:VISIBLE_STATE_REVIEWED"]
    if round_index <= 7:
        preferred = [item for item in visible_anchor_ids if item.startswith("EVIDENCE:")]
        return (preferred or visible_anchor_ids)[:2]
    return visible_anchor_ids[: min(3, len(visible_anchor_ids))]


def _observation_expected_anchor_ids(overrides: dict[str, Any], round_index: int) -> list[str]:
    """Use only explicitly readable file/email evidence for early observations.

    Task-board and calendar seeds are useful context, but q01-q07 prompts do
    not require the model to cite them. Keeping expected anchors on file/email
    evidence avoids penalizing an otherwise valid task-board side effect for
    not citing a substrate that was not clearly required.
    """

    evidence_only: dict[str, Any] = {}
    for key, value in overrides.items():
        text_key = str(key)
        if text_key.startswith("workspace/") or text_key == "emails.json":
            evidence_only[text_key] = value
    anchors = _runtime_anchor_ids(evidence_only)
    return _fallback_expected_anchor_ids(anchors, round_index)


def _evidence_strings(values: Any) -> list[str]:
    if not isinstance(values, list):
        values = [values]
    out: list[str] = []
    for value in values:
        text = _evidence_string(value)
        if text:
            out.append(text)
    return out


def _normalize_expected_anchor_ids(
    narrative: dict[str, Any],
    values: Any,
    *,
    active_update_ids: list[str],
) -> list[str]:
    ids = _evidence_strings(values)
    if not ids:
        ids = [
            _evidence_string(narrative.get("canonical_decision_id")),
            _evidence_string(narrative.get("canonical_rule_id")),
        ]
    update_map = {
        f"UPDATE:U{index:02d}_APPLIED": _update_anchor_id(update_id)
        for index, update_id in enumerate(active_update_ids, start=1)
    }
    normalized: list[str] = []
    for item in ids:
        text = item.strip()
        upper = text.upper()
        if _is_weak_anchor_id(upper):
            continue
        normalized.append(update_map.get(upper, text))
    return list(dict.fromkeys(item for item in normalized if item))


def _evidence_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "; ".join(_evidence_string(item) for item in value if _evidence_string(item))
    if isinstance(value, dict):
        priority = value.get("decision_priority")
        if isinstance(priority, list) and priority:
            return _evidence_string(priority[0])
        for key in ("truth_anchor", "preference_rule", "claim", "summary"):
            if key in value:
                text = _evidence_string(value[key])
                if text:
                    return text
        return ""
    return str(value).strip()


def _is_weak_anchor_id(anchor_id: str) -> bool:
    return bool(re.match(r"^DECISION:STATUS_[0-9]+$", str(anchor_id).upper()))
