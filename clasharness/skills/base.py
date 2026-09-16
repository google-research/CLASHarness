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

"""Core skill interfaces used by backend runtime."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from clasharness.backend.base import BaseBackend, CommandResult
from clasharness.skills.spec import SkillSpec


class Skill(ABC):
    """Abstract executable capability bound to one or more command prefixes."""

    # Subclasses declare the instance attributes that constitute episode-mutable
    # state so they can be snapshotted/restored (for `undo`). Only plain data
    # attributes (lists/dicts/scalars) should be listed — never object refs.
    _snapshot_attrs: tuple[str, ...] = ()

    # Unified declarative spec (the single source of truth for routing, MCP
    # facade, docs and tests). Every concrete skill should attach one. See
    # ``clasharness/skills/spec.py`` and ``docs/mcp-extension-guide.md``.
    SPEC: ClassVar[SkillSpec | None] = None

    @classmethod
    def get_spec(cls) -> SkillSpec | None:
        """Return this skill's :class:`SkillSpec` (or ``None`` if undeclared)."""
        return cls.SPEC

    @classmethod
    def from_context(cls, context: "SkillContext") -> "Skill":
        """Build an instance from a :class:`~clasharness.skills.extensions.SkillContext`.

        Used by the additive extension layer (see ``@register_skill`` in
        ``clasharness/skills/extensions.py``). The default builds a
        dependency-free skill via ``cls()``; override when construction needs
        shared collaborators::

            @classmethod
            def from_context(cls, context):
                return cls(file_backend=context.file_backend)
        """
        return cls()

    def __init__(self, *, prefixes: tuple[str, ...]) -> None:
        if not prefixes:
            raise ValueError("Skill must declare at least one command prefix")
        self._prefixes = tuple(prefixes)

    @property
    def prefixes(self) -> tuple[str, ...]:
        return self._prefixes

    def can_handle(self, command_prefix: str) -> bool:
        return command_prefix in self._prefixes

    @abstractmethod
    def initialize(self, state_dir: str, env_vars: dict[str, str]) -> None:
        """Initialize skill state for a new episode."""

    @abstractmethod
    def execute(self, command: str) -> CommandResult:
        """Execute one command routed to this skill."""

    def get_state(self) -> dict[str, Any]:
        """Return skill state contributions for evaluator state."""
        return {}

    def to_state(self) -> dict[str, Any]:
        """Return a deep-copied snapshot of this skill's mutable state."""
        return {
            attr: copy.deepcopy(getattr(self, attr))
            for attr in self._snapshot_attrs
            if hasattr(self, attr)
        }

    def from_state(self, data: dict[str, Any]) -> None:
        """Restore this skill's mutable state from a snapshot produced by to_state."""
        if not isinstance(data, dict):
            return
        for attr in self._snapshot_attrs:
            if attr in data:
                setattr(self, attr, copy.deepcopy(data[attr]))

    @abstractmethod
    def cleanup(self) -> None:
        """Release skill resources."""


class BackendSkillAdapter(Skill):
    """Thin adapter that exposes an existing ``BaseBackend`` as a skill."""

    def __init__(self, backend: BaseBackend, *, prefixes: tuple[str, ...]) -> None:
        super().__init__(prefixes=prefixes)
        self.backend = backend

    def initialize(self, state_dir: str, env_vars: dict[str, str]) -> None:
        self.backend.initialize(state_dir, env_vars)

    def execute(self, command: str) -> CommandResult:
        return self.backend.execute_cli(command)

    def get_state(self) -> dict[str, Any]:
        return self.backend.get_state()

    def to_state(self) -> dict[str, Any]:
        snap = getattr(self.backend, "to_state", None)
        if callable(snap):
            result = snap()
            return result if isinstance(result, dict) else {}
        return {}

    def from_state(self, data: dict[str, Any]) -> None:
        restore = getattr(self.backend, "from_state", None)
        if callable(restore):
            restore(data)

    def cleanup(self) -> None:
        self.backend.cleanup()
