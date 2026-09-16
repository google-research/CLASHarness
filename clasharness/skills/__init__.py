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

"""Skill layer for backend command routing."""

from clasharness.skills.base import Skill, BackendSkillAdapter
from clasharness.skills.impl import CalendarSkill, EmailSkill, TasksSkill, WeatherSkill
from clasharness.skills.registry import SkillRegistry
from clasharness.skills.runtime import SkillRuntime
from clasharness.skills.spec import CommandSpec, RealKind, SkillSpec
from clasharness.skills.extensions import (
    SkillContext,
    build_extension_skills,
    clear_extension_skills,
    extension_registry_sandbox,
    register_skill,
    registered_extension_skills,
)

__all__ = [
    "Skill",
    "BackendSkillAdapter",
    "CalendarSkill",
    "EmailSkill",
    "TasksSkill",
    "WeatherSkill",
    "SkillRegistry",
    "SkillRuntime",
    # Unified spec
    "SkillSpec",
    "CommandSpec",
    "RealKind",
    # Extension interface
    "register_skill",
    "SkillContext",
    "build_extension_skills",
    "registered_extension_skills",
    "clear_extension_skills",
    "extension_registry_sandbox",
]
