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

import importlib

import pytest

from clasharness.backend.base import CommandResult
from clasharness.skills.base import Skill
from clasharness.skills.registry import SkillRegistry


class _NoopSkill(Skill):
    def __init__(self, prefixes: tuple[str, ...]) -> None:
        super().__init__(prefixes=prefixes)

    def initialize(self, state_dir: str, env_vars: dict[str, str]) -> None:
        pass

    def execute(self, command: str) -> CommandResult:
        return CommandResult(stdout=command, stderr="", exit_code=0)

    def cleanup(self) -> None:
        pass


def test_register_and_resolve_prefix():
    registry = SkillRegistry()
    skill = _NoopSkill(("weather",))
    registry.register("weather", skill)
    assert registry.resolve("weather") is skill


def test_register_skill_registers_all_prefixes_once():
    registry = SkillRegistry()
    skill = _NoopSkill(("calendar", "gcalcli"))
    registry.register_skill(skill)
    assert registry.resolve("calendar") is skill
    assert registry.resolve("gcalcli") is skill
    assert len(registry.iter_skills()) == 1


def test_register_duplicate_prefix_raises():
    registry = SkillRegistry()
    registry.register("weather", _NoopSkill(("weather",)))
    with pytest.raises(ValueError, match="already registered"):
        registry.register("weather", _NoopSkill(("weather",)))


def test_list_prefixes_sorted():
    registry = SkillRegistry()
    registry.register("tasks", _NoopSkill(("tasks",)))
    registry.register("calendar", _NoopSkill(("calendar",)))
    assert registry.list_prefixes() == ["calendar", "tasks"]


def test_adapters_package_exports_only_live_adapters():
    """Guard against dead adapter shells creeping back in.

    Only backends without a native unified-spec skill are adapted
    (asset/file/http). Calendar, email, tasks and weather are implemented
    directly in ``clasharness.skills.impl`` and registered from there, so
    their retired ``BackendSkillAdapter`` shells must stay deleted -- both as
    package exports and as importable submodules.
    """
    from clasharness.skills import adapters

    assert sorted(adapters.__all__) == [
        "AssetSkillAdapter",
        "FileSkillAdapter",
        "HttpSkillAdapter",
    ]

    for dead in ("calendar", "email", "tasks", "weather"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(f"clasharness.skills.adapters.{dead}")
