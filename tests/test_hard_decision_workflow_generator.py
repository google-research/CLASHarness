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

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collections import Counter
import re

import clasharness.tasks.generators  # noqa: F401  (auto-registers all generators)
from clasharness.skills.impl.weather_skill import _get_weather
from clasharness.tasks.generation_options import set_generation_options
from clasharness.tasks.registry import generate_all_tasks


def _cmds(task) -> list[str]:
    return list(task.ground_truth.solution_commands) if task.ground_truth else []


def _first_cmd(commands: list[str], prefix: str) -> str:
    return next((cmd for cmd in commands if cmd.startswith(prefix)), "")


def _sentence_count(text: str) -> int:
    return len(re.findall(r"[.!?](?=\s|$)", text))


_RISKY_WEATHER_CONDITIONS = {"rainy", "heavy rain", "rain showers", "heavy rain showers", "thunderstorm", "drizzle"}


def _generate_tasks(*, variants_per_scenario: int = 16, scenario_counts: dict[str, int] | None = None):
    set_generation_options(
        message_dry_run=False,
        plugin_install_mode="mixed",
        command_profile="local_skill",
        complex_task_pack="standard",
        complex_scenario_profile="life_work",
        complex_min_steps=3,
        complex_max_steps=5,
        hard_decision_variants_per_scenario=variants_per_scenario,
        hard_decision_scenario_counts=scenario_counts,
    )
    return generate_all_tasks(generator_ids=["hard_decision_workflow"])


def test_hard_decision_workflow_generator_produces_parameterized_challenge_tasks():
    tasks = _generate_tasks()
    assert len(tasks) == 362

    scenario_counts = Counter(
        str(task.data.public.get("hard_decision_scenario", "unknown")) for task in tasks
    )
    assert scenario_counts == {
        "inbox_followthrough": 12,
        "release_recovery_runbook": 10,
        "channel_incident_recovery": 10,
        "daily_ops_commitment_loop": 10,
        "release_gate_followthrough": 30,
        "delivery_update_followthrough": 10,
        "ops_review_followthrough": 30,
        "existing_state_followthrough": 10,
        "duplicate_avoidance_followthrough": 10,
        "multi_source_decision_followthrough": 20,
        "state_repair_followthrough": 46,
        "completion_gap_followthrough": 12,
        "branch_resolution_followthrough": 40,
        "already_done_skip_followthrough": 10,
        "wrong_state_replacement_followthrough": 46,
        "interrupted_workflow_resume": 30,
        "contradictory_source_resolution": 26,
    }

    family_has_openclaw = False
    family_has_non_openclaw = False
    for task in tasks:
        commands = _cmds(task)
        assert 5 <= len(commands) <= 9
        family_has_openclaw = family_has_openclaw or any(
            cmd.startswith("openclaw ") for cmd in commands
        )
        family_has_non_openclaw = family_has_non_openclaw or any(
            not cmd.startswith("openclaw ") for cmd in commands
        )
        assert task.task_id.startswith("hard_decision_workflow_")
        assert "challenge" in task.realism_tags
        assert "hard_decision" in task.realism_tags
        assert "underspecified" in task.realism_tags
        assert task.decision_requirements
        assert task.data.public.get("prompt_style") in {"brief", "underspecified"}
        assert task.data.public.get("hard_decision_ability") in {
            "information_transfer",
            "workflow_completion",
            "gap_completion",
            "duplicate_avoidance",
            "state_repair",
            "multi_source_reasoning",
        }
        ability_tags = task.data.public.get("hard_decision_ability_tags")
        assert isinstance(ability_tags, list) and ability_tags
        assert task.data.public["hard_decision_ability"] in ability_tags
        assert task.online_requirement in {"optional", "required"}
        assert task.provider_dependencies
    assert family_has_openclaw
    assert family_has_non_openclaw



def test_hard_decision_scenario_to_ability_mapping_is_present():
    tasks = _generate_tasks()
    mapping = {
        str(task.data.public.get("hard_decision_scenario")): task.data.public.get("hard_decision_ability")
        for task in tasks
    }
    assert mapping["existing_state_followthrough"] == "gap_completion"
    assert mapping["state_repair_followthrough"] == "state_repair"
    assert mapping["branch_resolution_followthrough"] == "multi_source_reasoning"
    assert mapping["already_done_skip_followthrough"] == "duplicate_avoidance"
    assert mapping["inbox_followthrough"] == "information_transfer"
    assert mapping["interrupted_workflow_resume"] == "gap_completion"
    assert mapping["contradictory_source_resolution"] == "multi_source_reasoning"


def test_hard_decision_workflow_default_front_of_schedule_stays_grouped_by_scenario():
    tasks = _generate_tasks()
    first_scenarios = [task.data.public["hard_decision_scenario"] for task in tasks[:30]]

    assert first_scenarios[:6] == ["inbox_followthrough"] * 6
    assert first_scenarios[6:12] == ["release_recovery_runbook"] * 6
    assert first_scenarios[12:18] == ["channel_incident_recovery"] * 6
    assert first_scenarios[18:24] == ["daily_ops_commitment_loop"] * 6
    assert first_scenarios[24:30] == ["inbox_followthrough"] * 6


def test_hard_decision_workflow_generation_is_stable_for_same_options():
    tasks_a = _generate_tasks()
    tasks_b = _generate_tasks()

    sig_a = [(task.task_id, task.instruction, tuple(_cmds(task))) for task in tasks_a]
    sig_b = [(task.task_id, task.instruction, tuple(_cmds(task))) for task in tasks_b]
    assert sig_a == sig_b


def test_hard_decision_workflow_emits_canonical_instruction_and_surface_variants():
    tasks = _generate_tasks()
    for task in tasks:
        assert task.canonical_instruction
        assert task.canonical_instruction != ""
        variants = task.variant_texts()
        assert len(variants) >= 6
        assert task.instruction not in variants
        assert len(set(variants)) == len(variants)


def test_hard_decision_workflow_scenario_counts_are_overridable():
    tasks = _generate_tasks(
        scenario_counts={
            "existing_state_followthrough": 24,
            "state_repair_followthrough": 10,
            "already_done_skip_followthrough": 6,
        }
    )
    scenario_counts = Counter(
        str(task.data.public.get("hard_decision_scenario", "unknown")) for task in tasks
    )
    assert scenario_counts["existing_state_followthrough"] == 24
    assert scenario_counts["state_repair_followthrough"] == 10
    assert scenario_counts["already_done_skip_followthrough"] == 6
    assert scenario_counts["inbox_followthrough"] == 16
    assert scenario_counts["wrong_state_replacement_followthrough"] == 16
    assert scenario_counts["interrupted_workflow_resume"] == 16
    assert scenario_counts["contradictory_source_resolution"] == 16


def test_hard_decision_workflow_variant_count_is_configurable():
    tasks = _generate_tasks(variants_per_scenario=8)
    assert len(tasks) == 136
    scenario_counts = Counter(
        str(task.data.public.get("hard_decision_scenario", "unknown")) for task in tasks
    )
    assert scenario_counts == {
        "inbox_followthrough": 8,
        "release_recovery_runbook": 8,
        "channel_incident_recovery": 8,
        "daily_ops_commitment_loop": 8,
        "release_gate_followthrough": 8,
        "delivery_update_followthrough": 8,
        "ops_review_followthrough": 8,
        "existing_state_followthrough": 8,
        "duplicate_avoidance_followthrough": 8,
        "multi_source_decision_followthrough": 8,
        "state_repair_followthrough": 8,
        "completion_gap_followthrough": 8,
        "branch_resolution_followthrough": 8,
        "already_done_skip_followthrough": 8,
        "wrong_state_replacement_followthrough": 8,
        "interrupted_workflow_resume": 8,
        "contradictory_source_resolution": 8,
    }


def test_hard_decision_workflow_has_no_duplicate_instruction_command_signatures():
    tasks = _generate_tasks()
    signatures = [(task.instruction, tuple(_cmds(task))) for task in tasks]
    assert len(signatures) == len(set(signatures))


def test_hard_decision_workflow_instruction_and_gt_alignment_invariants_hold():
    tasks = _generate_tasks()
    for task in tasks:
        instruction = task.instruction.lower()
        commands = _cmds(task)
        if any(cmd.startswith("email read --id ") for cmd in commands):
            assert any(token in instruction for token in ("review", "read", "email", "note", "thread"))
        if any(cmd.startswith("tasks add ") for cmd in commands):
            assert any(token in instruction for token in ("task", "board", "next step", "follow-up", "decision-log", "gap", "missing pieces", "queued up", "tracked", "reminder"))
        if any(cmd.startswith("calendar add-event ") or cmd.startswith("calendar update-event ") for cmd in commands):
            assert any(token in instruction for token in ("schedule", "sync", "block", "event", "review", "slot", "calendar", "live"))
        if any(cmd.startswith("email send ") for cmd in commands):
            assert "send" in instruction or "email" in instruction
        if any(cmd.startswith("openclaw cron add ") for cmd in commands):
            assert "daily" in instruction or "recurring" in instruction
        if "check the current model" in instruction or "confirm the current model" in instruction:
            assert any(cmd.startswith("openclaw config get agent.model") for cmd in commands)
        if any(token in instruction for token in ("avoid duplicating", "do not recreate", "without duplicating")):
            check_names = {check.get("name", "") for check in task.ground_truth.evaluation_checks}
            assert any(name.startswith("no duplicate") for name in check_names)


def test_hard_decision_inbox_followthrough_does_not_lock_email_subject_keyword():
    tasks = _generate_tasks()
    inbox_tasks = [
        task for task in tasks if task.data.public.get("hard_decision_scenario") == "inbox_followthrough"
    ]
    assert inbox_tasks
    for task in inbox_tasks:
        check_names = {check.get("name", "") for check in task.ground_truth.evaluation_checks}
        assert not any(name.startswith("email subject references ") for name in check_names)


def test_hard_decision_incident_recovery_does_not_require_channel_connected_state():
    tasks = _generate_tasks()
    incident_tasks = [
        task for task in tasks if task.data.public.get("hard_decision_scenario") == "channel_incident_recovery"
    ]
    assert incident_tasks
    for task in incident_tasks:
        check_names = {check.get("name", "") for check in task.ground_truth.evaluation_checks}
        assert not any(name.endswith(" connected") for name in check_names)




def test_release_like_hard_scenarios_use_stable_handoff_path():
    tasks = _generate_tasks()
    release_scenarios = {
        "release_recovery_runbook",
        "release_gate_followthrough",
        "state_repair_followthrough",
        "completion_gap_followthrough",
        "wrong_state_replacement_followthrough",
        "interrupted_workflow_resume",
    }
    for task in tasks:
        if task.data.public.get("hard_decision_scenario") not in release_scenarios:
            continue
        file_cmds = [cmd for cmd in _cmds(task) if cmd.startswith("file create --path ")]
        if not file_cmds:
            continue
        handoff_cmds = [cmd for cmd in file_cmds if "handoff" in cmd]
        if not handoff_cmds:
            continue
        assert all("handoff" in cmd for cmd in handoff_cmds)
        check_names = {check.get("name", "") for check in task.ground_truth.evaluation_checks}
        assert "handoff file path references handoff" in check_names
        assert "handoff file created under /ops/" not in check_names


def test_hard_decision_state_repair_non_model_variants_do_not_tell_model_review():
    tasks = _generate_tasks()
    repair_tasks = [
        task for task in tasks if task.data.public.get("hard_decision_scenario") == "state_repair_followthrough"
    ]
    assert repair_tasks
    for task in repair_tasks:
        commands = _cmds(task)
        instruction = task.instruction.lower()
        if any(cmd.startswith("openclaw models set ") for cmd in commands):
            assert "current model" in instruction
        else:
            assert "current model" not in instruction


def test_hard_decision_existing_state_wording_points_to_existing_setup_and_missing_only():
    tasks = _generate_tasks()
    existing_tasks = [
        task for task in tasks if task.data.public.get("hard_decision_scenario") == "existing_state_followthrough"
    ]
    assert existing_tasks
    for task in existing_tasks:
        instruction = task.instruction.lower()
        assert "artifacts" not in instruction
        assert "already" in instruction or "partway" in instruction
        assert any(token in instruction for token in ("fill only the gaps", "finish only what's still missing", "finish only what's missing", "finish just the missing pieces", "only add what's still missing", "only fill the gaps", "finish only the missing piece", "finish just the missing piece", "finish what's missing", "finish the missing piece"))
        assert "next step on the board" in instruction
        assert "review slot on the calendar" in instruction
        assert "recurring daily check" in instruction
        assert "short recap" in instruction


def test_hard_decision_inbox_wording_is_anchored_to_email_threads():
    tasks = _generate_tasks()
    inbox_tasks = [
        task for task in tasks if task.data.public.get("hard_decision_scenario") == "inbox_followthrough"
    ]
    assert inbox_tasks
    for task in inbox_tasks:
        instruction = task.instruction.lower()
        assert "email thread" in instruction
        assert "in the inbox" in instruction
        assert "next step" in instruction
        assert "board" in instruction
        assert "calendar" in instruction
        assert "handoff file" in instruction
        assert any(token in instruction for token in ("live follow-up", "calendar alone", "what is still missing"))


def test_hard_decision_state_repair_wording_names_staged_objects():
    tasks = _generate_tasks()
    repair_tasks = [
        task for task in tasks if task.data.public.get("hard_decision_scenario") == "state_repair_followthrough"
    ]
    assert repair_tasks
    for task in repair_tasks:
        instruction = task.instruction.lower()
        commands = _cmds(task)
        if any(cmd.startswith("openclaw models set ") for cmd in commands):
            assert "current model" in instruction
            assert "next-step task on the board" in instruction
            assert "sync on the calendar" in instruction
        elif any(cmd.startswith("calendar update-event --id ") for cmd in commands):
            assert "next-step task on the board" in instruction
            assert "sync on the calendar" in instruction
        elif any(cmd.startswith("tasks complete --title ") for cmd in commands):
            assert "next-step task on the board" in instruction
            assert "sync on the calendar" in instruction


def test_hard_decision_cron_creating_ops_wording_mentions_schedule():
    tasks = _generate_tasks()
    cron_scenarios = {
        "daily_ops_commitment_loop",
        "ops_review_followthrough",
        "multi_source_decision_followthrough",
    }
    schedule_tokens = ("recurring daily check", "daily check schedule", "scheduled")
    cron_tasks = [
        task
        for task in tasks
        if task.data.public.get("hard_decision_scenario") in cron_scenarios
    ]
    assert cron_tasks
    for task in cron_tasks:
        instruction = task.instruction.lower()
        if any(cmd.startswith("openclaw cron add ") for cmd in _cmds(task)):
            assert any(token in instruction for token in schedule_tokens)


def test_channel_incident_recovery_uses_recipient_email_check_and_result_anchors():
    tasks = _generate_tasks()
    incident_tasks = [
        task for task in tasks if task.data.public.get("hard_decision_scenario") == "channel_incident_recovery"
    ]
    assert incident_tasks
    for task in incident_tasks:
        instruction = task.instruction.lower()
        names = {check.get("name", "") for check in task.ground_truth.evaluation_checks}
        assert "update" in instruction
        assert "next step" in instruction
        assert "on the board" in instruction
        assert any(token in instruction for token in ("already being tracked", "already on the board", "missing piece"))
        topic = next((item.get("value", "") for item in task.hidden_constraints if item.get("type") == "title"), "")
        if topic:
            assert topic.split()[0].lower() in instruction
        assert any(name.startswith("email sent to ") for name in names)
        assert "recap email sent" not in names


def test_delivery_update_followthrough_wording_points_to_path_next_step_and_recipient():
    tasks = _generate_tasks()
    delivery_tasks = [
        task for task in tasks if task.data.public.get("hard_decision_scenario") == "delivery_update_followthrough"
    ]
    assert delivery_tasks
    for task in delivery_tasks:
        instruction = task.instruction.lower()
        assert any(token in instruction for token in ("delivery path", "fix the delivery path", "repair the path", "fix the path", "clean up the path"))
        assert "next step" in instruction
        assert "on the board" in instruction
        names = {check.get("name", "") for check in task.ground_truth.evaluation_checks}
        if "calendar event created" in names:
            assert "calendar" in instruction
        if "no duplicate task created" in names:
            assert "already be on the board" in instruction or "already on the board" in instruction
        assert any(token in instruction for token in ("target is shared", "target is direct", "missing", "already be on the board"))
        assert "recap" in instruction
        assert any(item.get("type") == "email" and item.get("value", "").lower() in instruction for item in task.visible_constraints)


def test_duplicate_avoidance_wording_makes_primary_and_backup_rules_explicit():
    tasks = _generate_tasks()
    duplicate_tasks = [
        task for task in tasks if task.data.public.get("hard_decision_scenario") == "duplicate_avoidance_followthrough"
    ]
    assert duplicate_tasks
    for task in duplicate_tasks:
        instruction = task.instruction.lower()
        assert "forecast" in instruction
        assert "calendar" in instruction
        assert "backup review block" in instruction
        assert "primary review block" in instruction
        assert "on the calendar" in instruction


def test_daily_ops_wording_keeps_schedule_and_next_step_visible():
    tasks = _generate_tasks()
    daily_tasks = [
        task for task in tasks if task.data.public.get("hard_decision_scenario") == "daily_ops_commitment_loop"
    ]
    assert daily_tasks
    for task in daily_tasks:
        instruction = task.instruction.lower()
        assert "recurring daily check" in instruction or "daily check schedule" in instruction
        assert "next step" in instruction
        assert "on the board" in instruction


def test_ops_review_wording_mentions_review_calendar_next_step_and_schedule():
    tasks = _generate_tasks()
    review_tasks = [
        task for task in tasks if task.data.public.get("hard_decision_scenario") == "ops_review_followthrough"
    ]
    assert review_tasks
    for task in review_tasks:
        instruction = task.instruction.lower()
        assert "forecast" in instruction
        assert "calendar" in instruction
        assert "review" in instruction
        assert "on the calendar" in instruction
        assert "next step" in instruction
        assert "on the board" in instruction
        assert "recurring daily check" in instruction or "daily check schedule" in instruction


def test_release_gate_wording_mentions_handoff_sync_and_next_step():
    tasks = _generate_tasks()
    gate_tasks = [
        task for task in tasks if task.data.public.get("hard_decision_scenario") == "release_gate_followthrough"
    ]
    assert gate_tasks
    for task in gate_tasks:
        instruction = task.instruction.lower()
        assert "handoff" in instruction
        assert "sync" in instruction
        assert "calendar" in instruction
        assert "next step" in instruction
        assert "on the board" in instruction


def test_release_recovery_wording_mentions_handoff_review_and_decision_log():
    tasks = _generate_tasks()
    release_tasks = [
        task for task in tasks if task.data.public.get("hard_decision_scenario") == "release_recovery_runbook"
    ]
    assert release_tasks
    for task in release_tasks:
        instruction = task.instruction.lower()
        assert "handoff" in instruction
        assert "review" in instruction
        assert "calendar" in instruction
        assert "decision-log task" in instruction
        assert "board" in instruction


def test_multi_source_wording_mentions_live_calendar_branch_and_async_email_branch():
    tasks = _generate_tasks()
    multi_source_tasks = [
        task for task in tasks if task.data.public.get("hard_decision_scenario") == "multi_source_decision_followthrough"
    ]
    assert multi_source_tasks
    for task in multi_source_tasks:
        instruction = task.instruction.lower()
        assert "next step" in instruction
        assert "on the board" in instruction
        assert "email note" in instruction
        assert "review" in instruction
        assert "calendar" in instruction
        assert "if live" in instruction or "only if async" in instruction or "only if it" in instruction or "if it stays live" in instruction


def test_completion_gap_wording_mentions_next_step_when_task_creation_is_required():
    tasks = _generate_tasks()
    gap_tasks = [
        task
        for task in tasks
        if task.data.public.get("hard_decision_scenario") == "completion_gap_followthrough"
    ]
    assert gap_tasks
    for task in gap_tasks:
        commands = _cmds(task)
        instruction = task.instruction.lower()
        if any(cmd.startswith("tasks add ") for cmd in commands):
            assert "next step" in instruction or "next-step task" in instruction
            assert "board" in instruction
        if any(cmd.startswith("calendar add-event ") for cmd in commands):
            assert "sync" in instruction
            assert "calendar" in instruction
        if any(cmd.startswith("file create --path ") for cmd in commands):
            assert "handoff" in instruction


def test_hard_decision_branch_resolution_underspecified_wording_avoids_staged_abstraction():
    tasks = _generate_tasks()
    branch_tasks = [
        task
        for task in tasks
        if task.data.public.get("hard_decision_scenario") == "branch_resolution_followthrough"
        and task.data.public.get("prompt_style") == "underspecified"
    ]
    assert branch_tasks
    for task in branch_tasks:
        instruction = task.instruction.lower()
        assert "what's already staged" not in instruction
        assert "task already on the board" in instruction
        assert "recurring daily check schedule" in instruction


def test_hard_decision_already_done_skip_wording_points_to_board_and_calendar():
    tasks = _generate_tasks()
    skip_tasks = [
        task
        for task in tasks
        if task.data.public.get("hard_decision_scenario") == "already_done_skip_followthrough"
    ]
    assert skip_tasks
    for task in skip_tasks:
        instruction = task.instruction.lower()
        assert "staged next step" not in instruction
        assert "next-step task on the board" in instruction
        assert "sync on the calendar" in instruction
        assert "before you send" in instruction


def test_interrupted_resume_wording_and_checks_stay_gap_oriented():
    tasks = _generate_tasks()
    resume_tasks = [
        task
        for task in tasks
        if task.data.public.get("hard_decision_scenario") == "interrupted_workflow_resume"
    ]
    assert resume_tasks
    for task in resume_tasks:
        instruction = task.instruction.lower()
        names = {check.get("name", "") for check in task.ground_truth.evaluation_checks}
        assert any(
            token in instruction
            for token in (
                "already in motion",
                "already underway",
                "partly underway",
                "already has",
                "already have part",
                "part of the",
                "pieces moving",
            )
        )
        assert any(token in instruction for token in ("missing pieces", "missing handoff", "missing pieces, send", "finish what's still missing", "finish the missing", "finish what's missing"))
        assert any(
            token in instruction
            for token in (
                "don't recreate",
                "avoid recreating",
                "don't rebuild",
                "avoid rebuilding",
                "leave the existing setup alone",
                "leave the existing pieces alone",
                "leave the rest alone",
            )
        )
        assert any(name.startswith("no duplicate") for name in names)
        assert task.data.private.get("initial_state_overrides")
        assert len(_cmds(task)) >= 7


def test_existing_state_variants_use_two_existing_objects_and_no_duplicate_guards():
    tasks = _generate_tasks()
    existing_tasks = [
        task for task in tasks if task.data.public.get("hard_decision_scenario") == "existing_state_followthrough"
    ]
    assert existing_tasks
    for task in existing_tasks:
        commands = _cmds(task)
        overrides = task.data.private.get("initial_state_overrides") or {}
        names = {check.get("name", "") for check in task.ground_truth.evaluation_checks}
        existing_objects = sum(
            1 for key in ("tasks.json", "calendar_events.json", "cron_jobs.json") if key in overrides
        )
        assert existing_objects == 2
        assert any(cmd.startswith("email send --to ") for cmd in commands)
        assert any(name.startswith("email sent to ") for name in names)
        assert sum(1 for name in names if name.startswith("no duplicate")) >= 2


def test_completion_gap_variants_require_real_state_inspection_and_handoff():
    tasks = _generate_tasks()
    gap_tasks = [
        task for task in tasks if task.data.public.get("hard_decision_scenario") == "completion_gap_followthrough"
    ]
    assert gap_tasks
    for task in gap_tasks:
        commands = _cmds(task)
        names = {check.get("name", "") for check in task.ground_truth.evaluation_checks}
        assert "openclaw config get agent.model" in commands
        assert "tasks list --status pending" in commands
        assert any(cmd == "calendar list" for cmd in commands)
        assert any(cmd.startswith("calendar today --timezone ") for cmd in commands)
        assert "file read --path '/ops/release-review.txt'" in commands
        assert any(cmd.startswith("file create --path ") for cmd in commands)
        assert not any(cmd.startswith("email send --to ") for cmd in commands)
        assert not any(name.startswith("email sent to ") for name in names)


def test_multi_source_decision_variants_require_email_reading_without_existing_state_modes():
    tasks = _generate_tasks()
    multi_source_tasks = [
        task for task in tasks if task.data.public.get("hard_decision_scenario") == "multi_source_decision_followthrough"
    ]
    assert multi_source_tasks
    for task in multi_source_tasks:
        commands = _cmds(task)
        names = {check.get("name", "") for check in task.ground_truth.evaluation_checks}
        assert "email search --query 'review'" in commands
        assert "email read --id email_seed_5" in commands
        assert "next-step task already exists" not in names
        assert "daily check already exists" not in names


def test_interrupted_resume_variants_have_partial_state_and_longer_closure_sequences():
    tasks = _generate_tasks()
    resume_tasks = [
        task for task in tasks if task.data.public.get("hard_decision_scenario") == "interrupted_workflow_resume"
    ]
    assert resume_tasks
    for task in resume_tasks:
        commands = _cmds(task)
        overrides = task.data.private.get("initial_state_overrides") or {}
        names = {check.get("name", "") for check in task.ground_truth.evaluation_checks}
        assert len(commands) >= 8
        existing_objects = sum(
            1 for key in ("tasks.json", "calendar_events.json", "cron_jobs.json") if key in overrides
        )
        assert existing_objects >= 2
        assert any(key.endswith(".txt") for key in overrides)
        assert any(name.startswith("no duplicate") for name in names)
        assert not any("ops-resume" in cmd for cmd in commands)
        assert not any("resumed release follow-through" in cmd for cmd in commands)


def test_contradictory_source_resolution_wording_and_outcomes_are_explicit():
    tasks = _generate_tasks()
    conflict_tasks = [
        task
        for task in tasks
        if task.data.public.get("hard_decision_scenario") == "contradictory_source_resolution"
    ]
    assert conflict_tasks
    saw_live = False
    saw_async = False
    for task in conflict_tasks:
        instruction = task.instruction.lower()
        commands = _cmds(task)
        names = {check.get("name", "") for check in task.ground_truth.evaluation_checks}
        assert "email note" in instruction
        assert "forecast" in instruction
        assert "calendar" in instruction
        assert "next step" in instruction
        assert "on the board" in instruction
        assert "recurring daily check" in instruction or "daily check schedule" in instruction
        assert len(commands) >= 8
        if any(cmd.startswith("calendar add-event ") for cmd in commands):
            saw_live = True
            assert "calendar event created" in names
        if any(cmd.startswith("email send ") for cmd in commands):
            saw_async = True
            assert any(name.startswith("email sent to ") for name in names)
            assert "no duplicate calendar event created" in names
    assert saw_live
    assert saw_async

def test_hard_decision_workflow_diversity_covers_multiple_entities_per_scenario():
    tasks = _generate_tasks()
    inbox_cities = set()
    release_models = set()
    incident_targets = set()
    daily_ops_cities = set()
    gate_followthrough_models = set()
    delivery_targets = set()
    ops_review_cities = set()
    existing_state_kinds = set()
    duplicate_recipients = set()
    multi_source_cities = set()
    repair_kinds = set()
    completion_gap_kinds = set()
    branch_resolution_modes = set()
    already_done_recipients = set()
    replacement_kinds = set()

    for task in tasks:
        commands = _cmds(task)
        scenario = task.data.public.get("hard_decision_scenario")
        if scenario == "inbox_followthrough":
            task_cmd = _first_cmd(commands, "tasks add --title '")
            if task_cmd:
                inbox_cities.add(task_cmd)
            else:
                inbox_cities.add(
                    _first_cmd(commands, "calendar add-event --title '")
                    or _first_cmd(commands, "file create --path '")
                )
        elif scenario == "release_recovery_runbook":
            release_models.add(next((cmd for cmd in commands if cmd.startswith("openclaw models set ")), ""))
        elif scenario == "channel_incident_recovery":
            incident_targets.add(_first_cmd(commands, "openclaw message send --channel "))
        elif scenario == "daily_ops_commitment_loop":
            daily_ops_cities.add(commands[0])
        elif scenario == "release_gate_followthrough":
            gate_followthrough_models.add(
                next((cmd for cmd in commands if cmd.startswith("openclaw models set ")), "")
            )
        elif scenario == "delivery_update_followthrough":
            delivery_targets.add(_first_cmd(commands, "openclaw message send --channel "))
        elif scenario == "ops_review_followthrough":
            ops_review_cities.add(commands[0])
        elif scenario == "existing_state_followthrough":
            existing_state_kinds.add(task.initial_state)
        elif scenario == "duplicate_avoidance_followthrough":
            email_cmd = _first_cmd(commands, "email send --to ")
            duplicate_recipients.add(email_cmd or _first_cmd(commands, "calendar add-event --title '") or _first_cmd(commands, "tasks add --title '"))
        elif scenario == "multi_source_decision_followthrough":
            multi_source_cities.add(commands[0])
        elif scenario == "state_repair_followthrough":
            if any(cmd.startswith("openclaw models set ") for cmd in commands):
                repair_kinds.add("model")
            elif any(cmd.startswith("calendar update-event --id ") for cmd in commands):
                repair_kinds.add("calendar")
            elif any(cmd.startswith("tasks complete --title ") for cmd in commands):
                repair_kinds.add("task")
        elif scenario == "completion_gap_followthrough":
            if any(cmd.startswith("tasks add ") for cmd in commands) and any(cmd.startswith("calendar add-event ") for cmd in commands):
                completion_gap_kinds.add("task_and_calendar")
            elif any(cmd.startswith("calendar add-event ") for cmd in commands):
                completion_gap_kinds.add("calendar_only")
            elif any(cmd.startswith("tasks add ") for cmd in commands):
                completion_gap_kinds.add("task_only")
        elif scenario == "branch_resolution_followthrough":
            branch_resolution_modes.add("backup" if any(cmd.startswith("email send ") for cmd in commands) else "primary")
        elif scenario == "already_done_skip_followthrough":
            already_done_recipients.add(_first_cmd(commands, "email send --to "))
        elif scenario == "wrong_state_replacement_followthrough":
            if any(cmd.startswith("openclaw models set ") for cmd in commands):
                replacement_kinds.add("model")
            elif any(cmd.startswith("calendar update-event --id ") for cmd in commands):
                replacement_kinds.add("calendar")
            elif any(cmd.startswith("tasks complete --title ") for cmd in commands):
                replacement_kinds.add("task")

    assert len(inbox_cities) >= 10
    assert len(release_models) >= 5
    assert len(incident_targets) >= 3
    assert len(daily_ops_cities) >= 3
    assert len(gate_followthrough_models) >= 4
    assert len(delivery_targets) >= 3
    assert len(ops_review_cities) >= 3
    assert existing_state_kinds == {"default"}
    assert len(duplicate_recipients) >= 3
    assert len(multi_source_cities) >= 7
    assert repair_kinds == {"model", "calendar", "task"}
    assert "task_and_calendar" in completion_gap_kinds
    assert "task_only" in completion_gap_kinds or "calendar_only" in completion_gap_kinds
    assert branch_resolution_modes == {"backup", "primary"}
    assert len(already_done_recipients) >= 3
    assert replacement_kinds == {"model", "calendar", "task"}


def test_hard_decision_followthrough_checks_are_result_first():
    tasks = _generate_tasks()
    for task in tasks:
        scenario = task.data.public.get("hard_decision_scenario")
        names = {check.get("name", "") for check in task.ground_truth.evaluation_checks}
        if scenario == "release_gate_followthrough":
            assert "task created" in names
            assert "calendar event created" in names
            assert not any(name.startswith("task title references ") for name in names)
        elif scenario == "delivery_update_followthrough":
            assert "task created" in names or "no duplicate task created" in names
            if "calendar event created" in names:
                assert "calendar event created" in names
            else:
                assert "no duplicate calendar event created" in names
            assert not any(name.startswith("task title references ") for name in names)
        elif scenario == "ops_review_followthrough":
            assert "task created" in names or "no duplicate task created" in names
            assert "calendar event created" in names
            assert not any(name.startswith("calendar title references ") for name in names)
        elif scenario == "existing_state_followthrough":
            assert any(name.startswith("no duplicate") for name in names)
        elif scenario == "inbox_followthrough":
            assert "file created" in names
            assert "calendar event created" in names or "no duplicate calendar event created" in names
            assert any(name.startswith("task title references ") for name in names) or "no duplicate task created" in names
            assert any(name.startswith("email sent to ") for name in names)
            assert not any(name.startswith("calendar title references ") for name in names)
        elif scenario == "duplicate_avoidance_followthrough":
            assert "no duplicate task created" in names
            assert "no duplicate cron job created" in names
            assert any(name.startswith("calendar title references ") for name in names) or "no duplicate calendar event created" in names
        elif scenario == "multi_source_decision_followthrough":
            assert "task created" in names
            assert "cron job created" in names
        elif scenario == "state_repair_followthrough":
            assert "file created" in names
            assert not any(name.startswith("task title references ") for name in names)
        elif scenario == "completion_gap_followthrough":
            assert "file created" in names
            assert not any(name.startswith("task title references ") for name in names)
        elif scenario == "branch_resolution_followthrough":
            assert "no duplicate task created" in names
            assert "no duplicate cron job created" in names
            assert not any(name.startswith("task title references ") for name in names)
        elif scenario == "already_done_skip_followthrough":
            assert any(name.startswith("email sent to ") for name in names)
            assert "no duplicate task created" in names
            assert "no duplicate calendar event created" in names
        elif scenario == "wrong_state_replacement_followthrough":
            assert "file created" not in names
            assert not any(name.startswith("task title references ") for name in names)
        elif scenario == "interrupted_workflow_resume":
            assert any(name.startswith("email sent to ") for name in names)
            assert any(name.startswith("no duplicate") for name in names)
        elif scenario == "contradictory_source_resolution":
            assert "task created" in names
            assert "cron job created" in names


def test_existing_state_followthrough_specs_keep_exactly_two_existing_pieces():
    tasks = _generate_tasks()
    existing_state_tasks = [
        task for task in tasks if task.data.public.get("hard_decision_scenario") == "existing_state_followthrough"
    ]
    assert existing_state_tasks

    seen_missing = set()
    for task in existing_state_tasks:
        overrides = task.data.private.get("initial_state_overrides", {})
        has_task = "tasks.json" in overrides
        has_calendar = "calendar_events.json" in overrides
        has_cron = "cron_jobs.json" in overrides
        override_count = sum([has_task, has_calendar, has_cron])
        check_names = {check.get("name", "") for check in task.ground_truth.evaluation_checks}

        assert task.initial_state == "default"
        assert override_count == 2

        if "cron job created" in check_names:
            seen_missing.add("cron")
            assert has_task and has_calendar and not has_cron
        elif "task created" in check_names:
            seen_missing.add("task")
            assert has_calendar and has_cron and not has_task
        elif "calendar event created" in check_names:
            seen_missing.add("calendar")
            assert has_task and has_cron and not has_calendar
        else:
            raise AssertionError(f"could not infer missing piece from checks: {check_names}")

    assert seen_missing == {"cron", "task", "calendar"}


def test_inbox_followthrough_variants_cover_task_existing_and_calendar_existing_modes():
    tasks = _generate_tasks()
    inbox_tasks = [
        task for task in tasks if task.data.public.get("hard_decision_scenario") == "inbox_followthrough"
    ]
    assert inbox_tasks
    saw_task_existing = False
    saw_task_missing = False
    saw_sync_required = False
    saw_async_only = False
    for task in inbox_tasks:
        names = {check.get("name", "") for check in task.ground_truth.evaluation_checks}
        overrides = task.data.private.get("initial_state_overrides") or {}
        if "follow-up task already exists" in names:
            saw_task_existing = True
            assert "tasks.json" in overrides
            assert "no duplicate task created" in names
            assert "file created" in names
            assert not any(name.startswith("task title references ") for name in names)
        else:
            saw_task_missing = True
            assert "tasks.json" not in overrides
            assert "task created" in names
            assert "file created" in names
            assert any(name.startswith("task title references ") for name in names)
        if "calendar event created" in names:
            saw_sync_required = True
        else:
            saw_async_only = True
            assert "no duplicate calendar event created" in names
    assert saw_task_existing
    assert saw_task_missing
    assert saw_sync_required
    assert saw_async_only


def test_inbox_followthrough_instruction_explicitly_names_which_piece_already_exists():
    tasks = _generate_tasks()
    inbox_tasks = [
        task for task in tasks if task.data.public.get("hard_decision_scenario") == "inbox_followthrough"
    ]
    assert inbox_tasks
    saw_task_existing = False
    saw_task_missing = False
    for task in inbox_tasks:
        instruction = task.instruction.lower()
        names = {check.get("name", "") for check in task.ground_truth.evaluation_checks}
        if "follow-up task already exists" in names:
            saw_task_existing = True
            assert "already on the board" in instruction
        else:
            saw_task_missing = True
            assert "board still lacks the next step" in instruction
        if "calendar event created" in names:
            assert "live follow-up" in instruction
        else:
            assert "calendar alone" in instruction
    assert saw_task_existing
    assert saw_task_missing


def test_channel_incident_recovery_uses_existing_task_and_no_duplicate_task_checks():
    tasks = _generate_tasks()
    incident_tasks = [
        task for task in tasks if task.data.public.get("hard_decision_scenario") == "channel_incident_recovery"
    ]
    assert incident_tasks
    for task in incident_tasks:
        commands = _cmds(task)
        names = {check.get("name", "") for check in task.ground_truth.evaluation_checks}
        overrides = task.data.private.get("initial_state_overrides") or {}
        assert "tasks list --status pending" in commands
        assert "tasks.json" in overrides
        assert "next-step task already exists" in names
        assert "no duplicate task created" in names
        assert any(name.startswith("message sent to ") for name in names)
        assert any(name.startswith("email sent to ") for name in names)


def test_delivery_update_followthrough_uses_existing_task_and_no_duplicate_task_checks():
    tasks = _generate_tasks()
    delivery_tasks = [
        task for task in tasks if task.data.public.get("hard_decision_scenario") == "delivery_update_followthrough"
    ]
    assert delivery_tasks
    saw_task_existing = False
    saw_task_missing = False
    for task in delivery_tasks:
        commands = _cmds(task)
        names = {check.get("name", "") for check in task.ground_truth.evaluation_checks}
        overrides = task.data.private.get("initial_state_overrides") or {}
        assert "tasks list --status pending" in commands
        assert any(name.startswith("message sent to ") for name in names)
        assert any(name.startswith("email sent to ") for name in names)
        if "next-step task already exists" in names:
            saw_task_existing = True
            assert "tasks.json" in overrides
            assert "no duplicate task created" in names
        else:
            saw_task_missing = True
            assert "tasks.json" not in overrides
            assert "task created" in names
        if "calendar event created" in names:
            assert "no duplicate calendar event created" not in names
        else:
            assert "no duplicate calendar event created" in names
    assert saw_task_existing
    assert saw_task_missing


def test_duplicate_avoidance_variants_cover_primary_and_backup_modes():
    tasks = _generate_tasks()
    duplicate_tasks = [
        task for task in tasks if task.data.public.get("hard_decision_scenario") == "duplicate_avoidance_followthrough"
    ]
    assert duplicate_tasks
    saw_primary = False
    saw_backup = False
    for task in duplicate_tasks:
        commands = _cmds(task)
        names = {check.get("name", "") for check in task.ground_truth.evaluation_checks}
        overrides = task.data.private.get("initial_state_overrides") or {}
        assert "tasks.json" in overrides
        assert "cron_jobs.json" in overrides
        assert "next-step task already exists" in names
        assert "daily check already exists" in names
        assert "no duplicate task created" in names
        assert "no duplicate cron job created" in names
        assert "calendar event created" in names
        assert not any(cmd.startswith("email send --to ") for cmd in commands)
        if "calendar title references backup" in names:
            saw_backup = True
        if "calendar title references primary" in names:
            saw_primary = True
    assert saw_primary
    assert saw_backup


def test_duplicate_avoidance_forecast_branch_matches_visible_weather():
    tasks = _generate_tasks()
    duplicate_tasks = [
        task for task in tasks if task.data.public.get("hard_decision_scenario") == "duplicate_avoidance_followthrough"
    ]
    assert duplicate_tasks
    for task in duplicate_tasks:
        commands = _cmds(task)
        names = {check.get("name", "") for check in task.ground_truth.evaluation_checks}
        instruction = task.instruction.lower()
        assert "forecast" in instruction
        assert "calendar" in instruction
        assert "board" in instruction or "setup" in instruction
        forecast_cmd = next(cmd for cmd in commands if cmd.startswith("weather forecast --location "))
        location = forecast_cmd.split("--location ", 1)[1].split(" --days", 1)[0].strip().strip("'\"")
        risky = any(
            str(_get_weather(location, f"2026-03-{day:02d}").get("condition", "")).lower() in _RISKY_WEATHER_CONDITIONS
            for day in (1, 2, 3)
        )
        if risky:
            assert "calendar title references backup" in names
        else:
            assert "calendar title references primary" in names


def test_email_followthrough_scenarios_keep_visible_recipient_and_email_check_aligned():
    tasks = _generate_tasks()
    scenarios = {
        "inbox_followthrough",
        "channel_incident_recovery",
        "delivery_update_followthrough",
        "existing_state_followthrough",
        "already_done_skip_followthrough",
    }
    for task in tasks:
        scenario = task.data.public.get("hard_decision_scenario")
        if scenario not in scenarios:
            continue
        names = {check.get("name", "") for check in task.ground_truth.evaluation_checks}
        recipients = [item.get("value", "") for item in task.visible_constraints if item.get("type") == "email"]
        assert recipients
        assert any(name.startswith("email sent to ") for name in names)
        assert any(recipient and recipient.lower() in task.instruction.lower() for recipient in recipients)


def test_followthrough_file_closure_scenarios_use_broad_handoff_file_checks():
    tasks = _generate_tasks()
    scenarios = {"inbox_followthrough"}
    for task in tasks:
        if task.data.public.get("hard_decision_scenario") not in scenarios:
            continue
        names = {check.get("name", "") for check in task.ground_truth.evaluation_checks}
        assert "file created" in names
        assert "handoff file path references handoff" in names


def test_hard_decision_instruction_quality_avoids_generator_checklist_phrasing():
    tasks = _generate_tasks()
    banned_phrases = (
        "artifacts",
        "keep the next step on the board",
        "make sure the daily check is in place",
        "refresh the handoff notes",
        "follow-through",
    )
    for task in tasks:
        instruction = task.instruction.lower()
        assert all(phrase not in instruction for phrase in banned_phrases)
        assert _sentence_count(task.instruction) <= 2
        assert len(task.instruction) <= 320


def test_hard_decision_existing_state_wording_stays_human_and_gap_oriented():
    tasks = _generate_tasks()
    existing_tasks = [
        task for task in tasks if task.data.public.get("hard_decision_scenario") == "existing_state_followthrough"
    ]
    assert existing_tasks
    for task in existing_tasks:
        instruction = task.instruction.lower()
        assert "existing next-step task" not in instruction
        assert "already" in instruction or "partway" in instruction
        assert any(token in instruction for token in ("fill only the gaps", "finish only what's still missing", "only add what's still missing", "only fill the gaps", "finish only what's missing", "finish just the missing pieces", "finish only the missing piece", "finish just the missing piece", "finish what's missing", "finish the missing piece"))


def test_completion_gap_wording_names_missing_release_work_without_old_or_template_phrase():
    tasks = _generate_tasks()
    gap_tasks = [
        task for task in tasks if task.data.public.get("hard_decision_scenario") == "completion_gap_followthrough"
    ]
    assert gap_tasks
    saw_task_and_calendar = False
    saw_calendar_only = False
    saw_task_only = False
    for task in gap_tasks:
        instruction = task.instruction.lower()
        commands = _cmds(task)
        assert "release work" in instruction
        assert "release setup" not in instruction
        assert "or the sync on the calendar only if it's missing" not in instruction
        if any(cmd.startswith("tasks add ") for cmd in commands) and any(cmd.startswith("calendar add-event ") for cmd in commands):
            saw_task_and_calendar = True
            assert "the next step and the review sync are still missing" in instruction
            assert "finish the remaining next step on the board and sync on the calendar" in instruction
        elif any(cmd.startswith("calendar add-event ") for cmd in commands):
            saw_calendar_only = True
            assert "the review sync is the missing piece" in instruction
            assert "add only the missing sync on the calendar" in instruction
        elif any(cmd.startswith("tasks add ") for cmd in commands):
            saw_task_only = True
            assert "the next step is the missing piece" in instruction
            assert "add only the missing next step on the board" in instruction
    assert saw_task_and_calendar
    assert saw_calendar_only
    assert saw_task_only


def test_release_recovery_wording_uses_grounded_review_slot_and_decision_log_task():
    tasks = _generate_tasks()
    release_tasks = [
        task for task in tasks if task.data.public.get("hard_decision_scenario") == "release_recovery_runbook"
    ]
    assert release_tasks
    for task in release_tasks:
        instruction = task.instruction.lower()
        assert "review piece" not in instruction
        assert "decision-log reminder" not in instruction
        assert "review slot" in instruction
        assert "decision-log task" in instruction


def test_wrong_state_replacement_wording_drops_piece_and_right_one_language():
    tasks = _generate_tasks()
    replacement_tasks = [
        task for task in tasks if task.data.public.get("hard_decision_scenario") == "wrong_state_replacement_followthrough"
    ]
    assert replacement_tasks
    for task in replacement_tasks:
        instruction = task.instruction.lower()
        assert "bad piece" not in instruction
        assert "right one" not in instruction
        assert "stale piece" not in instruction
        assert "correct version" in instruction
        assert "correctly set" in instruction
        assert "handoff file" not in instruction


def test_state_repair_wording_names_stale_target_without_piece_language():
    tasks = _generate_tasks()
    repair_tasks = [
        task for task in tasks if task.data.public.get("hard_decision_scenario") == "state_repair_followthrough"
    ]
    assert repair_tasks
    for task in repair_tasks:
        instruction = task.instruction.lower()
        assert "stale piece" not in instruction
        assert "bad piece" not in instruction
        assert "stale model setting" in instruction or "stale sync" in instruction or "stale next step" in instruction
        assert "correctly set" in instruction


def test_release_recovery_variants_are_partial_state_completion_not_straight_creation():
    tasks = _generate_tasks()
    release_tasks = [
        task for task in tasks if task.data.public.get("hard_decision_scenario") == "release_recovery_runbook"
    ]
    assert release_tasks
    saw_existing_task = False
    saw_existing_calendar = False
    for task in release_tasks:
        names = {check.get("name", "") for check in task.ground_truth.evaluation_checks}
        commands = _cmds(task)
        assert "openclaw config get agent.model" in commands
        assert "tasks list --status pending" in commands
        assert "calendar list" in commands
        if "decision-log reminder already exists" in names:
            saw_existing_task = True
            assert "no duplicate task created" in names
        if "review slot already exists" in names:
            saw_existing_calendar = True
            assert "no duplicate calendar event created" in names
    assert saw_existing_task
    assert saw_existing_calendar


def test_daily_ops_and_ops_review_include_partial_state_variants():
    tasks = _generate_tasks()
    for scenario in ("daily_ops_commitment_loop", "ops_review_followthrough"):
        scenario_tasks = [
            task for task in tasks if task.data.public.get("hard_decision_scenario") == scenario
        ]
        assert scenario_tasks
        saw_task_existing = False
        saw_cron_existing = False
        for task in scenario_tasks:
            names = {check.get("name", "") for check in task.ground_truth.evaluation_checks}
            if "next-step task already exists" in names:
                saw_task_existing = True
                assert "no duplicate task created" in names
            if "daily check already exists" in names:
                saw_cron_existing = True
                assert "no duplicate cron job created" in names
        assert saw_task_existing
        assert saw_cron_existing


def test_hard_decision_already_done_skip_wording_emphasizes_confirm_and_recap():
    tasks = _generate_tasks()
    skip_tasks = [
        task for task in tasks if task.data.public.get("hard_decision_scenario") == "already_done_skip_followthrough"
    ]
    assert skip_tasks
    for task in skip_tasks:
        instruction = task.instruction.lower()
        assert "short recap" in instruction
        assert "verify" in instruction
        assert (
            "without rebuilding anything" in instruction
            or "leave it alone if it is already right" in instruction
            or "leave it untouched if it is already right" in instruction
            or "leave it alone." in instruction
            or "leave it alone if it's already right" in instruction
        )


def test_channel_incident_recovery_instruction_names_actual_recap_recipient():
    tasks = _generate_tasks()
    incident_tasks = [
        task for task in tasks if task.data.public.get("hard_decision_scenario") == "channel_incident_recovery"
    ]
    assert incident_tasks
    for task in incident_tasks:
        recipient = next(
            (
                item["value"]
                for item in task.visible_constraints
                if item.get("type") == "email"
            ),
            "",
        )
        assert recipient
        instruction = task.instruction.lower()
        assert recipient.lower() in instruction
        assert "send leadership a " not in instruction


def test_hard_decision_brief_and_underspecified_styles_sound_different():
    tasks = _generate_tasks()
    by_scenario = {}
    for task in tasks:
        scenario = str(task.data.public.get("hard_decision_scenario"))
        by_scenario.setdefault(scenario, {}).setdefault(task.data.public.get("prompt_style"), []).append(task.instruction)
    for scenario, styles in by_scenario.items():
        if "brief" not in styles or "underspecified" not in styles:
            continue
        assert set(styles["brief"]) != set(styles["underspecified"])


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__]))
