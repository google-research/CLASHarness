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

"""Base backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

#: Canonical environment flag signalling that skills should take their real
#: (non-mock) execution path. Real/hybrid backends set this to ``"1"`` during
#: :meth:`BaseBackend.initialize`; mock/multi backends leave it unset. Skills
#: read it in ``initialize()`` to choose between their mock and real backends.
REAL_ENV_FLAG = "CLASHARNESS_REAL"

#: Legacy flag kept for backward compatibility with external-service skills
#: (calendar/email/weather/tasks). Set alongside :data:`REAL_ENV_FLAG` when a
#: backend runs in a real mode.
ONLINE_DATA_ENV_FLAG = "CLASHARNESS_ENABLE_ONLINE_DATA"


@dataclass
class CommandResult:
    """Result of executing a command."""

    stdout: str
    stderr: str
    exit_code: int
    state_changes: dict[str, Any] | None = None
    meta: dict[str, Any] | None = None
    execution_trace: list[dict[str, Any]] | None = None


class BaseBackend(ABC):
    """Abstract backend for executing openclaw commands."""

    @abstractmethod
    def initialize(self, state_dir: str, env_vars: dict[str, str]) -> None:
        """Initialize the backend with a state directory."""

    @abstractmethod
    def execute_cli(self, command: str) -> CommandResult:
        """Execute a CLI command and return the result."""

    @abstractmethod
    def execute_python(self, code: str) -> CommandResult:
        """Execute Python code and return the result."""

    @abstractmethod
    def get_gateway_status(self) -> dict[str, Any] | None:
        """Get the current gateway status, or None if not running."""

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """Get the current openclaw configuration."""

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up backend resources."""

    @property
    def real_enabled(self) -> bool:
        """Whether this backend runs skills in their real (non-mock) path.

        Mock/multi backends return ``False`` (fully simulated). Real/hybrid
        backends override this to return ``True`` so the environment threads
        the canonical :data:`REAL_ENV_FLAG` signal into skill initialization.
        """
        return False

    def set_virtual_time(self, now: int) -> None:
        """Publish the environment's virtual clock to this backend (no-op default).

        The environment calls this before every command dispatch and after every
        clock advance so stateful protocol skills can grade time-limited records
        (leases, holds, pairing codes) against a real virtual clock rather than a
        skill-local logical tick. Backends that route to skills override this to
        thread ``now`` down to them; simple/mock backends ignore it.
        """
        return None

    def get_state(self) -> dict[str, Any]:
        """Return backend-provided evaluator state fragments."""
        return {}

    def snapshot_state(self) -> dict[str, Any]:
        """Return an in-memory snapshot of mutable backend state for undo.

        Backends that return a non-empty mapping opt into fast in-memory
        snapshot/restore. Returning ``{}`` (the default) makes the environment
        fall back to the legacy on-disk copytree snapshot.
        """
        return {}

    def restore_state(self, snapshot: dict[str, Any]) -> None:
        """Restore mutable backend state captured by :meth:`snapshot_state`."""
        return None
