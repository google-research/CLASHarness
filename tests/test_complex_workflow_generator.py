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

import re

import clasharness.tasks.generators  # noqa: F401  (auto-registers all generators)
from clasharness.tasks.generation_options import set_generation_options
from clasharness.tasks.registry import generate_all_tasks


def _cmds(task) -> list[str]:
    if not task.ground_truth:
        return []
    return task.ground_truth.solution_commands


def test_complex_workflow_pack_standard_generates_120_tasks():
    set_generation_options(
        message_dry_run=False,
        plugin_install_mode="mixed",
        command_profile="local_skill",
        complex_task_pack="standard",
        complex_scenario_profile="life_work",
        complex_min_steps=3,
        complex_max_steps=5,
    )
    tasks = generate_all_tasks(generator_ids=["complex_composed_workflow"])
    assert len(tasks) == 160

    template_counts: dict[str, int] = {}
    scenario_counts: dict[str, int] = {}
    high_volatility = 0
    first_status = 0
    final_tasks_add = 0
    for task in tasks:
        assert re.match(r"^complex_[a-z0-9_]+_[1-9][0-9]*$", task.task_id)
        assert not task.task_id.startswith("complex_composed_workflow_")

        commands = _cmds(task)
        joined = " ".join(commands)
        assert "email_seed_" not in joined
        assert "task_seed_" not in joined
        assert 3 <= len(commands) <= 5
        assert any(cmd.startswith("openclaw ") for cmd in commands)
        assert any(not cmd.startswith("openclaw ") for cmd in commands)
        if commands and commands[0].strip() == "openclaw status":
            first_status += 1
        if commands and commands[-1].startswith("tasks add "):
            final_tasks_add += 1
        template_id = str(task.data.public.get("complex_template", ""))
        assert template_id
        template_counts[template_id] = template_counts.get(template_id, 0) + 1
        scenario_slug = str(task.data.public.get("complex_scenario_slug", ""))
        assert scenario_slug
        scenario_counts[scenario_slug] = scenario_counts.get(scenario_slug, 0) + 1
        assert task.canonical_instruction == task.instruction
        assert task.online_requirement in {"optional", "required"}
        assert task.availability_tier in {"flaky", "external-risk"}
        assert "multi_step" in task.realism_tags
        assert "cross_domain" in task.realism_tags
        assert task.provider_dependencies
        if task.data.public.get("complex_high_volatility"):
            high_volatility += 1
            assert "high_volatility" in task.realism_tags
        assert task.data.public.get("realism_tier") == "life_work"
        assert bool(task.data.public.get("entity_consistency_pass", False))
        assert float(task.data.public.get("causal_chain_score", 0.0)) >= 0.8

    assert len(template_counts) == 16
    assert len(scenario_counts) == 16
    assert all(v == 10 for v in scenario_counts.values())
    assert high_volatility > 0
    assert first_status / len(tasks) <= 0.2
    assert final_tasks_add / len(tasks) <= 0.4


def test_complex_workflow_pack_off_generates_zero_tasks():
    set_generation_options(
        message_dry_run=False,
        plugin_install_mode="mixed",
        command_profile="local_skill",
        complex_task_pack="off",
        complex_scenario_profile="life_work",
        complex_min_steps=3,
        complex_max_steps=5,
    )
    tasks = generate_all_tasks(generator_ids=["complex_composed_workflow"])
    assert tasks == []


def test_complex_workflow_step_filtering_works():
    set_generation_options(
        message_dry_run=False,
        plugin_install_mode="mixed",
        command_profile="local_skill",
        complex_task_pack="standard",
        complex_scenario_profile="life_work",
        complex_min_steps=5,
        complex_max_steps=5,
    )
    tasks = generate_all_tasks(generator_ids=["complex_composed_workflow"])
    assert len(tasks) == 40
    assert all(len(_cmds(task)) == 5 for task in tasks)


def test_total_task_count_off_vs_standard():
    set_generation_options(
        message_dry_run=False,
        plugin_install_mode="mixed",
        command_profile="local_skill",
        complex_task_pack="off",
        complex_scenario_profile="life_work",
        complex_min_steps=3,
        complex_max_steps=5,
    )
    baseline = generate_all_tasks()
    assert len(baseline) == 1456

    set_generation_options(
        message_dry_run=False,
        plugin_install_mode="mixed",
        command_profile="local_skill",
        complex_task_pack="standard",
        complex_scenario_profile="life_work",
        complex_min_steps=3,
        complex_max_steps=5,
    )
    expanded = generate_all_tasks()
    assert len(expanded) == 1616


def test_complex_workflow_legacy_profile_keeps_120_tasks():
    set_generation_options(
        message_dry_run=False,
        plugin_install_mode="mixed",
        command_profile="local_skill",
        complex_task_pack="standard",
        complex_scenario_profile="legacy",
        complex_min_steps=3,
        complex_max_steps=5,
    )
    tasks = generate_all_tasks(generator_ids=["complex_composed_workflow"])
    assert len(tasks) == 120


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__]))
