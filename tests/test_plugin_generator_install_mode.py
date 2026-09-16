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

import clasharness.tasks.generators  # noqa: F401  (auto-registers all generators)
from clasharness.tasks.generation_options import set_generation_options
from clasharness.tasks.registry import generate_all_tasks


ONLINE_PACKAGE_BY_ID = {
    "discord": "@openclaw/discord",
    "slack": "@openclaw/slack",
    "telegram": "@openclaw/telegram",
    "whatsapp": "@openclaw/whatsapp",
}


def _tasks_by_name(tasks):
    out = {}
    for task in tasks:
        if not task.ground_truth:
            continue
        params = task.ground_truth.metadata.get("params", {})
        name = params.get("name")
        if name is not None:
            out[name] = task
    return out


def test_install_plugin_uses_real_install_for_online_subset_in_mixed_mode():
    set_generation_options(
        message_dry_run=False,
        plugin_install_mode="mixed",
        command_profile="local_skill",
    )
    try:
        tasks = generate_all_tasks(generator_ids=["install_plugin"])
        by_name = _tasks_by_name(tasks)

        for plugin_id, package in ONLINE_PACKAGE_BY_ID.items():
            task = by_name[plugin_id]
            commands = task.ground_truth.solution_commands
            assert commands == [f"openclaw plugins install {package}"]
            check = task.ground_truth.evaluation_checks[0]
            assert check["condition"] == "regex"
            # The pattern must bind the verb to *this* plugin. It used to be
            # the bare `(install|already)`, which any output mentioning either
            # word satisfied -- including installing a different plugin.
            assert "install\\w*|already" in check["expected"]
            assert plugin_id in check["expected"]
            assert package in check["expected"]
    finally:
        set_generation_options(
            message_dry_run=False,
            plugin_install_mode="mixed",
            command_profile="local_skill",
        )


def test_install_and_verify_uses_install_and_list_checks_in_mixed_mode():
    set_generation_options(
        message_dry_run=False,
        plugin_install_mode="mixed",
        command_profile="local_skill",
    )
    try:
        tasks = generate_all_tasks(generator_ids=["install_and_verify_plugin"])
        by_name = _tasks_by_name(tasks)
        task = by_name["discord"]
        commands = task.ground_truth.solution_commands
        assert commands[0] == "openclaw plugins install @openclaw/discord"
        assert commands[1] == "openclaw plugins list"

        checks = task.ground_truth.evaluation_checks
        assert checks[0]["output_field"] == "command_history"
        assert "openclaw\\ plugins\\ install\\ @openclaw/discord" in checks[0]["expected"]
        assert checks[1]["type"] == "effect"
        assert checks[1]["expected"]["value"] == "@openclaw/discord"
        assert checks[2]["type"] == "output"
        assert checks[2]["condition"] == "regex"
    finally:
        set_generation_options(
            message_dry_run=False,
            plugin_install_mode="mixed",
            command_profile="local_skill",
        )


def test_plugin_install_mode_stable_keeps_enable_commands():
    set_generation_options(
        message_dry_run=False,
        plugin_install_mode="stable",
        command_profile="local_skill",
    )
    try:
        tasks = generate_all_tasks(
            generator_ids=["install_plugin", "install_and_verify_plugin"]
        )
        for task in tasks:
            commands = task.ground_truth.solution_commands
            assert all("openclaw plugins install " not in c for c in commands)
            assert "openclaw plugins enable " in commands[0]
    finally:
        set_generation_options(
            message_dry_run=False,
            plugin_install_mode="mixed",
            command_profile="local_skill",
        )


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__]))
