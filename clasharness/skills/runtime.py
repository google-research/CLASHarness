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

"""Runtime for dispatching CLI commands to registered skills."""

from __future__ import annotations

from typing import Any

from clasharness.backend.base import CommandResult
from clasharness.skills.registry import SkillRegistry


class SkillRuntime:
    """Coordinates skill lifecycle and command dispatch."""

    def __init__(self, registry: SkillRegistry) -> None:
        self._registry = registry
        # Latest virtual-clock reading published by the environment (virtual
        # seconds). Pushed onto the resolved skill just before dispatch so
        # protocol skills can expire time-limited records; ``0`` until the
        # environment first calls :meth:`set_now`.
        self._now: int = 0

    def initialize(self, state_dir: str, env_vars: dict[str, str]) -> None:
        for skill in self._registry.iter_skills():
            skill.initialize(state_dir, env_vars)

    def set_now(self, now: int) -> None:
        """Record the environment's current virtual time (see :meth:`execute_cli`)."""
        self._now = int(now)

    def execute_cli(self, command: str) -> CommandResult:
        prefix = command.strip().split()[0] if command.strip() else ""
        skill = self._registry.resolve(prefix)
        if skill is None:
            return CommandResult(
                stdout="",
                stderr=(
                    f"Unknown command prefix '{prefix}'. "
                    f"Available: {', '.join(self._registry.list_prefixes())}"
                ),
                exit_code=1,
            )
        # Thread the virtual clock into stateful protocol skills (those that
        # declare a ``_now`` slot) without changing the ``execute`` signature.
        # Skills that do not expire records simply never read it.
        if hasattr(skill, "_now"):
            skill._now = self._now
        return skill.execute(command)

    def get_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        for skill in self._registry.iter_skills():
            state.update(skill.get_state())
        return state

    def snapshot_state(self) -> dict[str, Any]:
        """Capture each skill's mutable state, keyed by skill class name."""
        snapshot: dict[str, Any] = {}
        for skill in self._registry.iter_skills():
            skill_state = skill.to_state()
            if skill_state:
                snapshot[type(skill).__name__] = skill_state
        return snapshot

    def restore_state(self, snapshot: dict[str, Any]) -> None:
        """Restore skill state previously captured by :meth:`snapshot_state`."""
        if not isinstance(snapshot, dict):
            return
        for skill in self._registry.iter_skills():
            key = type(skill).__name__
            if key in snapshot:
                skill.from_state(snapshot[key])

    def cleanup(self) -> None:
        for skill in self._registry.iter_skills():
            skill.cleanup()
