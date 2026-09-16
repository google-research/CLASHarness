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

from collections import Counter
import json

import scripts.generate_tasks as generate_tasks
from clasharness.backend.mock_backend import MockBackend
from clasharness.tasks.generation_options import set_generation_options
from clasharness.tasks.registry import generate_all_tasks
from clasharness.utils.state_manager import StateManager


def _hard_tasks(*, variants_per_scenario: int = 16):
    set_generation_options(
        message_dry_run=False,
        plugin_install_mode="mixed",
        command_profile="local_skill",
        complex_task_pack="standard",
        complex_scenario_profile="life_work",
        complex_min_steps=3,
        complex_max_steps=5,
        hard_decision_variants_per_scenario=variants_per_scenario,
    )
    return generate_all_tasks(generator_ids=["hard_decision_workflow"])


def test_hard_split_is_stratified_by_scenario_and_stable():
    tasks = _hard_tasks()
    splits_a = generate_tasks._stratified_hard_splits(tasks, seed=42)
    splits_b = generate_tasks._stratified_hard_splits(tasks, seed=42)
    assert splits_a == splits_b

    task_by_id = {task.task_id: task for task in tasks}
    scenarios = {
        str(task.data.public.get("hard_decision_scenario", "unknown"))
        for task in tasks
    }
    assert scenarios

    for split_name, ids in splits_a.items():
        counts = Counter(
            str(task_by_id[task_id].data.public.get("hard_decision_scenario", "unknown"))
            for task_id in ids
        )
        assert set(counts) == scenarios
        assert all(counts[scenario] > 0 for scenario in scenarios)


def test_hard_wrong_model_base_config_exposes_agent_model_path():
    state = StateManager(base_config_name="hard_wrong_model")
    state.initialize()
    try:
        backend = MockBackend()
        backend.initialize(str(state.state_dir), state.get_env_vars())
        result = backend.execute_cli("openclaw config get agent.model")
        assert result.exit_code == 0
        assert result.stdout.strip() == "openai/gpt-4o"
    finally:
        state.cleanup()


def test_existing_ops_base_configs_use_explicit_existing_resource_names():
    state = StateManager(base_config_name="hard_existing_full")
    state.initialize()
    try:
        tasks_text = (state.state_dir / "tasks.json").read_text()
        calendar_text = (state.state_dir / "calendar_events.json").read_text()
        cron_text = (state.state_dir / "cron_jobs.json").read_text()
        assert "Existing ops next-step task" in tasks_text
        assert "Existing ops review block" in calendar_text
        assert "existing-daily-ops-check" in cron_text
        assert "Run existing ops daily check" in cron_text
    finally:
        state.cleanup()


def test_state_manager_applies_initial_state_overrides():
    state = StateManager(base_config_name="hard_existing_task")
    state.initialize()
    try:
        state.apply_state_overrides({
            "tasks.json": [
                {
                    "id": "task_existing_ops",
                    "title": "Tokyo existing ops next-step task",
                    "due": "2026-03-12",
                    "status": "pending",
                    "priority": "medium",
                    "duration": 1,
                }
            ]
        })
        tasks_text = (state.state_dir / "tasks.json").read_text()
        assert "Tokyo existing ops next-step task" in tasks_text
        assert "Existing ops next-step task" not in tasks_text
    finally:
        state.cleanup()


def test_hard_tasks_include_city_specific_initial_state_overrides():
    tasks = _hard_tasks()
    existing_tasks = [
        task for task in tasks if task.data.public.get("hard_decision_scenario") == "existing_state_followthrough"
    ]
    assert existing_tasks
    assert any(
        any("existing ops" in str(value).lower() for value in task.data.private.get("initial_state_overrides", {}).values())
        for task in existing_tasks
    )

    skip_tasks = [
        task for task in tasks if task.data.public.get("hard_decision_scenario") == "already_done_skip_followthrough"
    ]
    assert skip_tasks
    assert any(
        any("existing release" in str(value).lower() for value in task.data.private.get("initial_state_overrides", {}).values())
        for task in skip_tasks
    )


def test_instruction_quality_report_tracks_surface_diversity(tmp_path):
    tasks = _hard_tasks()
    generate_tasks._write_instruction_quality_report(tasks, tmp_path)
    report = json.loads((tmp_path / "datasets" / "instruction_quality_report.json").read_text())
    assert report["avg_instruction_variant_count"] >= 6.0
    assert report["avg_surface_forms_per_hard_scenario"] >= 8.0
    assert "existing_state_followthrough" in report["hard_scenario_surface_form_counts"]
    assert report["instruction_fourgram_repetition_rate"] > 0.0
    assert report["top_instruction_openers"]
