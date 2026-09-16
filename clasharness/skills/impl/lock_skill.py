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

"""Distributed-lock protocol skill (M1).

The ``lock`` family provides advisory mutual exclusion over named resources
(optionally a sub-``section`` of a resource):

* ``lock acquire --resource R [--section S] [--holder H]`` — take the lock.
  Acquiring a resource already held by a *different* holder fails with exit 1.
  This is the mutual-exclusion mechanic the ``distributed_lock`` scenarios grade
  (a correct agent acquires before mutating a shared resource; an agent that
  clobbers a peer's lock is caught).
* ``lock release --resource R [--section S]`` — release the lock.
* ``lock list`` / ``lock status --resource R`` — read-only introspection.

Successful ``acquire``/``release`` emit ``lock_acquired``/``lock_released``
effects through the shared protocol-effect channel
(:meth:`StatefulProtocolSkill._emit`), so the checker scaffolder can assert them.

Holder identity resolves as ``--holder`` → env ``CLASHARNESS_ACTOR`` →
``"coordinator"``. State persists to ``<state_dir>/lock.json`` via the base and
is snapshotted for ``undo``.
"""

from __future__ import annotations

import shlex
from typing import Any

from clasharness.backend.base import CommandResult
from clasharness.skills.spec import CommandSpec, RealKind, SkillSpec
from clasharness.skills.stateful import StatefulProtocolSkill

_DEFAULT_HOLDER = "coordinator"


class LockSkill(StatefulProtocolSkill):
    """Advisory locks over named resources with holder-conflict detection."""

    SPEC = SkillSpec(
        name="lock",
        category="coordination",
        summary="Advisory locks over named resources with holder-conflict detection.",
        prefixes=("lock",),
        real_kind=RealKind.LOCAL_PERSISTENT,
        commands=(
            CommandSpec(
                "lock acquire",
                "Acquire a lock on a resource (fails if held by another holder).",
                args={
                    "--resource": {"type": "string"},
                    "--section": {"type": "string"},
                    "--holder": {"type": "string"},
                },
                required=("--resource",),
                emits_effects=("lock_acquired",),
            ),
            CommandSpec(
                "lock release",
                "Release a lock on a resource.",
                args={
                    "--resource": {"type": "string"},
                    "--section": {"type": "string"},
                },
                required=("--resource",),
                emits_effects=("lock_released",),
            ),
            CommandSpec("lock list", "List currently held locks."),
            CommandSpec(
                "lock status",
                "Show the status of a resource lock.",
                args={"--resource": {"type": "string"}},
            ),
        ),
        smoke=(
            "lock acquire --resource db --holder alice",
            "lock release --resource db",
            "lock list",
        ),
    )

    def __init__(self) -> None:
        super().__init__(prefixes=("lock",), family="lock")
        self._actor = ""

    def _on_initialize(self, env_vars: dict[str, str]) -> None:
        self._actor = str(env_vars.get("CLASHARNESS_ACTOR", "") or "").strip()

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    def _resolve_holder(self, explicit: str | None) -> str:
        if explicit:
            return explicit
        if self._actor:
            return self._actor
        return _DEFAULT_HOLDER

    def _find(self, resource: str, section: str) -> dict[str, Any] | None:
        for rec in self._records:
            if rec.get("resource") == resource and rec.get("section", "") == section:
                return rec
        return None

    @staticmethod
    def _label(resource: str, section: str) -> str:
        return f"{resource}/{section}" if section else resource

    # ------------------------------------------------------------------ #
    # Dispatch                                                           #
    # ------------------------------------------------------------------ #
    def execute(self, command: str) -> CommandResult:
        try:
            parts = shlex.split(command.strip())
        except ValueError as exc:
            return CommandResult(stdout="", stderr=f"parse error: {exc}", exit_code=1)
        if not parts or parts[0] != "lock":
            return CommandResult(stdout="", stderr="Not a lock command", exit_code=1)
        if len(parts) < 2:
            return CommandResult(
                stdout="lock <acquire|release|list|status> [options]",
                stderr="",
                exit_code=0,
            )

        sub = parts[1]
        args = parts[2:]
        handlers = {
            "acquire": self._cmd_acquire,
            "release": self._cmd_release,
            "list": self._cmd_list,
            "status": self._cmd_status,
        }
        handler = handlers.get(sub)
        if handler is None:
            return CommandResult(
                stdout="", stderr=f"Unknown lock subcommand: {sub}", exit_code=1
            )
        return handler(args)

    def _cmd_acquire(self, args: list[str]) -> CommandResult:
        resource = _get_arg(args, "--resource")
        if not resource:
            return CommandResult(stdout="", stderr="--resource is required", exit_code=1)
        section = _get_arg(args, "--section", "") or ""
        holder = self._resolve_holder(_get_arg(args, "--holder"))

        existing = self._find(resource, section)
        if existing is not None and existing.get("holder") != holder:
            return CommandResult(
                stdout="",
                stderr=(
                    f"lock conflict: {self._label(resource, section)} already held "
                    f"by {existing.get('holder')}"
                ),
                exit_code=1,
            )
        if existing is None:
            self._records.append(
                {
                    "resource": resource,
                    "section": section,
                    "holder": holder,
                    "created_at": self._advance_tick(),
                }
            )
            self._save()
        # else: idempotent re-acquire by the same holder — still a success.

        payload = {"resource": resource, "section": section, "holder": holder}
        return CommandResult(
            stdout=f"lock acquired: {self._label(resource, section)} by {holder}",
            stderr="",
            exit_code=0,
            state_changes=self._emit("lock_acquired", payload),
        )

    def _cmd_release(self, args: list[str]) -> CommandResult:
        resource = _get_arg(args, "--resource")
        if not resource:
            return CommandResult(stdout="", stderr="--resource is required", exit_code=1)
        section = _get_arg(args, "--section", "") or ""

        existing = self._find(resource, section)
        if existing is not None:
            self._records = [r for r in self._records if r is not existing]
            self._save()

        payload = {"resource": resource, "section": section}
        return CommandResult(
            stdout=f"lock released: {self._label(resource, section)}",
            stderr="",
            exit_code=0,
            state_changes=self._emit("lock_released", payload),
        )

    def _cmd_list(self, args: list[str]) -> CommandResult:
        if not self._records:
            return CommandResult(stdout="No locks held.", stderr="", exit_code=0)
        lines = ["Locks:"]
        for r in self._records:
            lines.append(
                f"  {self._label(r.get('resource', ''), r.get('section', ''))}"
                f" -> {r.get('holder', '')}"
            )
        return CommandResult(stdout="\n".join(lines), stderr="", exit_code=0)

    def _cmd_status(self, args: list[str]) -> CommandResult:
        resource = _get_arg(args, "--resource")
        if not resource:
            return CommandResult(stdout="", stderr="--resource is required", exit_code=1)
        section = _get_arg(args, "--section", "") or ""
        existing = self._find(resource, section)
        label = self._label(resource, section)
        if existing is None:
            return CommandResult(stdout=f"{label}: free", stderr="", exit_code=0)
        return CommandResult(
            stdout=f"{label}: held by {existing.get('holder')}", stderr="", exit_code=0
        )


def _get_arg(args: list[str], flag: str, default: str | None = None) -> str | None:
    """Return ``--flag value`` or ``--flag=value``, else ``default``."""
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(flag + "="):
            return a[len(flag) + 1:]
    return default
