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

"""Task generators for Plugin & Skill domain (D6)."""

from __future__ import annotations

import re
from typing import Any, Iterator

from clasharness.core.task import TaskData
from clasharness.tasks.generation_options import get_generation_options
from clasharness.tasks.registry import BaseTaskGenerator, Pass, SetupResult

_PLUGINS = [
    "discord",
    "slack",
    "telegram",
    "whatsapp",
    "matrix",
    "mattermost",
    "signal",
    "line",
    "nostr",
    "irc",
]

_ONLINE_INSTALL_PLUGIN_PACKAGE_BY_ID = {
    "discord": "@openclaw/discord",
    "slack": "@openclaw/slack",
    "telegram": "@openclaw/telegram",
    "whatsapp": "@openclaw/whatsapp",
}
_ONLINE_INSTALL_PLUGIN_IDS = list(_ONLINE_INSTALL_PLUGIN_PACKAGE_BY_ID.keys())

_SKILLS = [
    "healthcheck",
    "skill-creator",
    "tmux",
    "weather",
    "healthcheck",
    "weather",
]


def _use_online_install(plugin_id: str) -> bool:
    options = get_generation_options()
    return (
        options.plugin_install_mode == "mixed"
        and plugin_id in _ONLINE_INSTALL_PLUGIN_PACKAGE_BY_ID
    )


def _plugin_install_command(plugin_id: str) -> str:
    if _use_online_install(plugin_id):
        pkg = _ONLINE_INSTALL_PLUGIN_PACKAGE_BY_ID[plugin_id]
        return f"openclaw plugins install {pkg}"
    return f"openclaw plugins enable {plugin_id}"


def _plugin_list_presence_regex(plugin_id: str) -> str:
    pkg = _ONLINE_INSTALL_PLUGIN_PACKAGE_BY_ID.get(plugin_id, "")
    if pkg:
        return f"({re.escape(plugin_id)}|{re.escape(pkg)})"
    return re.escape(plugin_id)


@BaseTaskGenerator.register("install_plugin")
class InstallPluginGenerator(BaseTaskGenerator):
    """Install a named plugin."""

    required_domains = ("plugin_skill",)
    difficulty = 1
    parameters = {
        "name": _PLUGINS,
    }

    def setup(
        self, params: dict[str, Any], initial_config: dict[str, Any]
    ) -> Iterator[tuple[SetupResult, TaskData]]:
        data = TaskData(
            public={"name": params["name"]},
            private={"expected_plugin": params["name"]},
        )
        yield Pass(), data

    def get_instruction(self, params: dict[str, Any], data: TaskData) -> str:
        if _use_online_install(params["name"]):
            return f"Install the '{params['name']}' plugin."
        return f"Enable the '{params['name']}' plugin."

    def get_solution(self, params: dict[str, Any], data: TaskData) -> list[str]:
        return [_plugin_install_command(params["name"])]

    def get_evaluation_checks(
        self, params: dict[str, Any], data: TaskData
    ) -> list[dict[str, Any]]:
        if _use_online_install(params["name"]):
            # `plugins install` emits "Plugin 'PKG' installed." (or
            # "... is already installed."). The previous pattern was the bare
            # `(install|already)`, which any output mentioning either word
            # satisfied -- including installing a completely different plugin.
            # Tie the verb to the plugin that was actually asked for.
            return [
                {
                    "type": "output",
                    "condition": "regex",
                    "expected": (
                        rf"{_plugin_list_presence_regex(params['name'])}"
                        r"[^\n]*(install\w*|already)"
                    ),
                    "output_field": "last_stdout",
                    "name": f"plugin '{params['name']}' installed",
                    "ignore_case": True,
                }
            ]
        # Same defect as the online branch above, just spelled as a `contains`
        # check so it never showed up in a regex audit: the bare substring
        # "enable" is satisfied by *any* plugin being enabled.
        return [
            {
                "type": "output",
                "condition": "regex",
                "expected": (
                    rf"{_plugin_list_presence_regex(params['name'])}"
                    r"[^\n]*enable\w*"
                ),
                "output_field": "last_stdout",
                "name": f"plugin '{params['name']}' enabled",
                "ignore_case": True,
            }
        ]


@BaseTaskGenerator.register("install_skill")
class InstallSkillGenerator(BaseTaskGenerator):
    """Install a named skill."""

    required_domains = ("plugin_skill",)
    difficulty = 1
    parameters = {
        "name": _SKILLS,
    }

    def setup(
        self, params: dict[str, Any], initial_config: dict[str, Any]
    ) -> Iterator[tuple[SetupResult, TaskData]]:
        data = TaskData(
            public={"name": params["name"]},
            private={"expected_skill": params["name"]},
        )
        yield Pass(), data

    def get_instruction(self, params: dict[str, Any], data: TaskData) -> str:
        return f"Show detailed information for the '{params['name']}' skill."

    def get_solution(self, params: dict[str, Any], data: TaskData) -> list[str]:
        return [f"openclaw skills info {params['name']}"]

    def get_evaluation_checks(
        self, params: dict[str, Any], data: TaskData
    ) -> list[dict[str, Any]]:
        # `skills info X` emits "Skill: X\nDescription: ...".
        # The previous pattern `(skill|name|description|missing|not found)`
        # matched on any of those bare words, so it passed for a *different*
        # skill -- and `missing|not found` actively scored a failed lookup as a
        # success even though no code path emits either phrase. Require the
        # requested skill to be the one described.
        return [
            {
                "type": "output",
                "condition": "regex",
                "expected": rf"skill[^\n]*{re.escape(params['name'])}",
                "output_field": "last_stdout",
                "name": f"skill '{params['name']}' info shown",
                "ignore_case": True,
            }
        ]


@BaseTaskGenerator.register("list_plugins")
class ListPluginsGenerator(BaseTaskGenerator):
    """List all installed plugins."""

    required_domains = ("plugin_skill",)
    difficulty = 1
    parameters = {}

    def setup(
        self, params: dict[str, Any], initial_config: dict[str, Any]
    ) -> Iterator[tuple[SetupResult, TaskData]]:
        yield Pass(), TaskData()

    def get_instruction(self, params: dict[str, Any], data: TaskData) -> str:
        return "List all currently installed plugins."

    def get_solution(self, params: dict[str, Any], data: TaskData) -> list[str]:
        return ["openclaw plugins list"]

    def get_evaluation_checks(
        self, params: dict[str, Any], data: TaskData
    ) -> list[dict[str, Any]]:
        return [
            {
                "type": "output",
                "condition": "regex",
                "expected": r"(plugin|Plugin|No plugins)",
                "output_field": "last_stdout",
                "name": "plugins listed",
            }
        ]


@BaseTaskGenerator.register("install_and_verify_plugin")
class InstallAndVerifyPluginGenerator(BaseTaskGenerator):
    """Multi-step: install a plugin then verify it appears in the list."""

    required_domains = ("plugin_skill",)
    difficulty = 2
    parameters = {
        "name": _PLUGINS[:8],  # 8 variations
    }

    def setup(
        self, params: dict[str, Any], initial_config: dict[str, Any]
    ) -> Iterator[tuple[SetupResult, TaskData]]:
        data = TaskData(
            public={"name": params["name"]},
            private={"expected_plugin": params["name"]},
        )
        yield Pass(), data

    def get_instruction(self, params: dict[str, Any], data: TaskData) -> str:
        if _use_online_install(params["name"]):
            return (
                f"Install the '{params['name']}' plugin, then verify it appears "
                f"in plugin list output."
            )
        return (
            f"Enable the '{params['name']}' plugin, then verify it appears "
            f"in plugin list output."
        )

    def get_solution(self, params: dict[str, Any], data: TaskData) -> list[str]:
        return [
            _plugin_install_command(params["name"]),
            "openclaw plugins list",
        ]

    def get_evaluation_checks(
        self, params: dict[str, Any], data: TaskData
    ) -> list[dict[str, Any]]:
        checks: list[dict[str, Any]] = []
        expected_effect_plugin = params["name"]
        if _use_online_install(params["name"]):
            install_cmd = _plugin_install_command(params["name"])
            expected_effect_plugin = _ONLINE_INSTALL_PLUGIN_PACKAGE_BY_ID[params["name"]]
            checks.append(
                {
                    "type": "output",
                    "condition": "regex",
                    "expected": re.escape(install_cmd) + r".*?'exit_code':\s*0",
                    "output_field": "command_history",
                    "name": "install step executed successfully",
                    "ignore_case": False,
                }
            )

        checks.extend(
            [
                {
                    "type": "effect",
                    "effect_type": "plugins_installed",
                    "condition": "field_equals",
                    "expected": {"field": "name", "value": expected_effect_plugin},
                    "name": f"plugin '{params['name']}' installed/enabled",
                },
                {
                    "type": "output",
                    "condition": "regex",
                    "expected": _plugin_list_presence_regex(params["name"]),
                    "output_field": "last_stdout",
                    "name": "plugin appears in list output",
                    "ignore_case": True,
                },
            ]
        )
        return checks


@BaseTaskGenerator.register("install_then_remove_plugin")
class InstallThenRemovePluginGenerator(BaseTaskGenerator):
    """Multi-step: install a plugin then remove it (tests lifecycle)."""

    required_domains = ("plugin_skill",)
    difficulty = 2
    parameters = {
        "name": _PLUGINS[:4],  # keep 4 variations while using valid built-in plugin ids
    }

    def setup(
        self, params: dict[str, Any], initial_config: dict[str, Any]
    ) -> Iterator[tuple[SetupResult, TaskData]]:
        data = TaskData(
            public={"name": params["name"]},
            private={},
        )
        yield Pass(), data

    def get_instruction(self, params: dict[str, Any], data: TaskData) -> str:
        return (
            f"Enable the '{params['name']}' plugin, verify it appears "
            f"by listing plugins, then disable it."
        )

    def get_solution(self, params: dict[str, Any], data: TaskData) -> list[str]:
        return [
            f"openclaw plugins enable {params['name']}",
            "openclaw plugins list",
            f"openclaw plugins disable {params['name']}",
        ]

    def get_evaluation_checks(
        self, params: dict[str, Any], data: TaskData
    ) -> list[dict[str, Any]]:
        # The bare substring "disable" was satisfied by disabling any plugin,
        # so the lifecycle task could be completed against the wrong one.
        return [
            {
                "type": "output",
                "condition": "regex",
                "expected": (
                    rf"{_plugin_list_presence_regex(params['name'])}"
                    r"[^\n]*disable\w*"
                ),
                "output_field": "last_stdout",
                "name": "plugin disabled",
                "ignore_case": True,
            },
        ]
