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


BANNED_PATTERNS = [
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


def _all_solution_commands(tasks) -> list[str]:
    cmds: list[str] = []
    for task in tasks:
        if not task.ground_truth:
            continue
        cmds.extend(task.ground_truth.solution_commands)
    return cmds


def test_generated_tasks_do_not_use_legacy_openclaw_syntax():
    set_generation_options(
        message_dry_run=False,
        plugin_install_mode="mixed",
        command_profile="local_skill",
    )
    tasks = generate_all_tasks()
    commands = _all_solution_commands(tasks)
    lowered = [c.lower() for c in commands]

    for banned in BANNED_PATTERNS:
        assert all(banned not in cmd for cmd in lowered), banned


def test_generated_tasks_include_local_skill_profile_markers():
    set_generation_options(
        message_dry_run=False,
        plugin_install_mode="mixed",
        command_profile="local_skill",
    )
    tasks = generate_all_tasks()
    commands = _all_solution_commands(tasks)

    assert any(cmd.startswith("openclaw config set gateway.port") for cmd in commands)
    assert any("openclaw agents add " in cmd and "--name" not in cmd for cmd in commands)
    assert any("openclaw plugins enable " in cmd for cmd in commands)
    assert any("openclaw plugins disable " in cmd for cmd in commands)
    assert any("openclaw plugins install @openclaw/" in cmd for cmd in commands)
    assert any("openclaw cron add " in cmd and "--cron" in cmd and "--name" in cmd for cmd in commands)
    assert any(cmd.startswith("weather get --location ") for cmd in commands)
    assert any(cmd.startswith("calendar add-event --title ") for cmd in commands)
    assert all(not cmd.startswith("gcalcli ") for cmd in commands)
    assert all(not cmd.startswith("curl -sG ") for cmd in commands)


def test_email_review_wording_requires_email_read_command():
    set_generation_options(
        message_dry_run=False,
        plugin_install_mode="mixed",
        command_profile="local_skill",
    )
    tasks = generate_all_tasks(generator_ids=["email_search_and_read", "hard_decision_workflow"])

    for task in tasks:
        instruction = (task.instruction or "").lower()
        commands = list(task.ground_truth.solution_commands) if task.ground_truth else []
        if "email" in instruction and ("review " in instruction or " read the matching email" in instruction):
            assert any(cmd.startswith("email read --id ") for cmd in commands), task.task_id


def test_universal_profile_is_not_supported():
    try:
        set_generation_options(
            message_dry_run=False,
            plugin_install_mode="mixed",
            command_profile="universal",
        )
    except ValueError as exc:
        assert "Only 'local_skill' is supported" in str(exc)
    else:
        raise AssertionError("expected ValueError for command_profile=universal")


def test_message_dry_run_generation_option_applies_to_message_generators():
    set_generation_options(
        message_dry_run=True,
        plugin_install_mode="mixed",
        command_profile="local_skill",
    )
    tasks = generate_all_tasks(
        generator_ids=[
            "msg_send_text",
            "msg_broadcast",
            "msg_create_poll",
            "msg_search",
            "msg_react",
        ]
    )
    try:
        commands = _all_solution_commands(tasks)
        assert commands
        assert all("--dry-run" in cmd for cmd in commands)
        assert all(
            task.ground_truth.metadata["generation_options"]["message_dry_run"] is True
            for task in tasks
            if task.ground_truth
        )
    finally:
        set_generation_options(
            message_dry_run=False,
            plugin_install_mode="mixed",
            command_profile="local_skill",
        )


def test_plugin_install_mode_stable_has_no_real_install_commands():
    set_generation_options(
        message_dry_run=False,
        plugin_install_mode="stable",
        command_profile="local_skill",
    )
    try:
        tasks = generate_all_tasks(
            generator_ids=["install_plugin", "install_and_verify_plugin"]
        )
        commands = _all_solution_commands(tasks)
        assert commands
        assert all("openclaw plugins install " not in c for c in commands)
        assert any("openclaw plugins enable " in c for c in commands)
    finally:
        set_generation_options(
            message_dry_run=False,
            plugin_install_mode="mixed",
            command_profile="local_skill",
        )


def test_generated_tasks_do_not_include_legacy_plugin_names():
    set_generation_options(
        message_dry_run=False,
        plugin_install_mode="mixed",
        command_profile="local_skill",
    )
    tasks = generate_all_tasks()
    lowered = [c.lower() for c in _all_solution_commands(tasks)]
    banned_legacy_names = ("weather-plugin", "slack-integration", "github-notifier")
    for token in banned_legacy_names:
        assert all(token not in cmd for cmd in lowered), token


TIGHTENED_OUTPUT_CHECK_GENERATORS = [
    "msg_search",
    "msg_react",
    "install_skill",
    "install_plugin",
    "install_then_remove_plugin",
]


def _run_ground_truth(commands: list[str]) -> str:
    """Execute a solution against the backend evaluation actually uses."""
    import tempfile

    from clasharness.backend.multi_app_backend import MultiAppBackend

    backend = MultiAppBackend()
    backend.initialize(tempfile.mkdtemp(), {})
    stdout = ""
    for command in commands:
        stdout = backend.execute_cli(command).stdout
    return stdout


def _checks_pass(checks, stdout: str) -> bool:
    from clasharness.evaluation.checkers.output_checker import OutputChecker

    state = {"last_stdout": stdout, "command_history": []}
    return all(
        OutputChecker.from_spec(check).evaluate(state, {}).passed
        for check in checks
        if check["type"] == "output"
    )


def _tightened_cases():
    set_generation_options(
        message_dry_run=False,
        plugin_install_mode="mixed",
        command_profile="local_skill",
    )
    cases = {}
    for generator_id in TIGHTENED_OUTPUT_CHECK_GENERATORS:
        rows = []
        for task in generate_all_tasks(generator_ids=[generator_id]):
            if not task.ground_truth:
                continue
            rows.append(
                (
                    task.task_id,
                    task.ground_truth.evaluation_checks,
                    _run_ground_truth(list(task.ground_truth.solution_commands)),
                )
            )
        cases[generator_id] = rows
    return cases


def test_tightened_output_checks_still_accept_the_ground_truth():
    """Guard against tightening a pattern into something GT cannot satisfy."""
    for generator_id, rows in _tightened_cases().items():
        assert rows, generator_id
        for task_id, checks, stdout in rows:
            assert _checks_pass(checks, stdout), f"{task_id}: GT stdout {stdout!r}"


def test_tightened_output_checks_reject_a_sibling_tasks_output():
    """The real defect: patterns so loose they matched the wrong task's output.

    `msg_react` accepted `(React|react|emoji)` -- reacting with the wrong emoji
    on the wrong channel passed. `msg_search` accepted the bare word `message`
    plus a `dry-run` form no code path emits. `install_skill` accepted
    `(skill|name|description|missing|not found)`, scoring even a failed lookup
    as success. `install_plugin` accepted `(install|already)`, and its offline
    branch plus `install_then_remove_plugin` used bare ``contains`` checks on
    "enable"/"disable" -- so acting on the *wrong* plugin passed. Each pattern
    must now be specific enough that a sibling task's output cannot satisfy it.
    """
    for generator_id, rows in _tightened_cases().items():
        assert len(rows) > 1, generator_id
        for task_id, checks, stdout in rows:
            for other_id, _, other_stdout in rows:
                if other_id == task_id or other_stdout == stdout:
                    continue
                assert not _checks_pass(checks, other_stdout), (
                    f"{generator_id}: {task_id}'s check is satisfied by "
                    f"{other_id}'s output {other_stdout!r}"
                )


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__]))
