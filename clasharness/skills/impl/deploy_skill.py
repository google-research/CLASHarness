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

"""Deployment / barrier-synchronization protocol skill (M1).

The ``deploy`` family models a release guarded by readiness *gates*. Its crux is
the **cutover barrier**: ``deploy cutover`` only succeeds once at least one gate
exists and *every* gate is ``ready``; otherwise it fails with exit 1. This is the
mechanic the ``barrier_synchronization`` scenarios grade (a correct agent waits
for all gates before cutting over; an agent that cuts over early is caught).

Verbs::

    deploy request  --change C                     # register a deployment
    deploy approve  --change C --approver R         # record an approval
    deploy gate-set --gate G --state ready|pending|failed|stale|expired
    deploy execute  --change C                      # -> deployment_executed
    deploy cutover  --change C                      # BARRIER -> deployment_cutover
    deploy rollback --change C                      # -> deployment_rolledback
    deploy monitor  --change C                      # read-only post-deploy check
    deploy status   [--change C]                    # read-only introspection

Effect-bearing verbs emit ``deployment_executed`` / ``deployment_cutover`` /
``deployment_rolledback`` through the shared protocol-effect channel
(:meth:`StatefulProtocolSkill._emit`). All records live in the base ``_records``
list tagged by ``kind`` (``gate`` | ``deployment`` | ``approval``), so base
persistence to ``<state_dir>/deploy.json``, snapshot, and ``undo`` work
unchanged.
"""

from __future__ import annotations

import shlex
from typing import Any

from clasharness.backend.base import CommandResult
from clasharness.skills.spec import (
    CommandSpec,
    EffectPrecondition,
    RealKind,
    SkillSpec,
)
from clasharness.skills.stateful import StatefulProtocolSkill

_VALID_GATE_STATES = frozenset(
    {"ready", "pending", "failed", "stale", "expired"}
)


class DeploySkill(StatefulProtocolSkill):
    """Release workflow with a readiness-gate cutover barrier."""

    SPEC = SkillSpec(
        name="deploy",
        category="coordination",
        summary="Release workflow with a readiness-gate cutover barrier.",
        prefixes=("deploy",),
        real_kind=RealKind.LOCAL_PERSISTENT,
        commands=(
            CommandSpec(
                "deploy request",
                "Register a deployment for a change.",
                args={"--change": {"type": "string"}},
                required=("--change",),
                emits_effects=("deployment_requested",),
            ),
            CommandSpec(
                "deploy approve",
                "Record an approval for a change.",
                args={
                    "--change": {"type": "string"},
                    "--approver": {"type": "string"},
                },
                required=("--change",),
                emits_effects=("deployment_approved",),
            ),
            CommandSpec(
                "deploy gate-set",
                "Set a readiness gate's state.",
                args={
                    "--gate": {"type": "string"},
                    "--state": {
                        "type": "string",
                        "enum": ["ready", "pending", "failed", "stale", "expired"],
                    },
                },
                required=("--gate", "--state"),
                emits_effects=("deployment_gate_set",),
            ),
            CommandSpec(
                "deploy execute",
                "Execute a deployment.",
                args={"--change": {"type": "string"}},
                required=("--change",),
                emits_effects=("deployment_executed",),
                requires=(
                    EffectPrecondition(
                        chain=("deploy request",), identifier_arg="--change"
                    ),
                ),
            ),
            CommandSpec(
                "deploy cutover",
                "Cutover barrier: succeeds only once every gate is ready.",
                args={"--change": {"type": "string"}},
                required=("--change",),
                emits_effects=("deployment_cutover",),
                requires=(
                    EffectPrecondition(
                        chain=("deploy request",),
                        identifier_arg="--change",
                        all_ready=("deploy gate-set", "--state", "ready"),
                    ),
                ),
            ),
            CommandSpec(
                "deploy rollback",
                "Roll back a deployment.",
                args={"--change": {"type": "string"}},
                required=("--change",),
                emits_effects=("deployment_rolledback",),
                requires=(
                    EffectPrecondition(
                        chain=("deploy request",), identifier_arg="--change"
                    ),
                ),
            ),
            CommandSpec(
                "deploy monitor",
                "Read-only post-deploy health check.",
                args={"--change": {"type": "string"}},
            ),
            CommandSpec(
                "deploy status",
                "Read-only deployment introspection.",
                args={"--change": {"type": "string"}},
            ),
        ),
        smoke=(
            "deploy request --change c1",
            "deploy execute --change c1",
            "deploy status",
        ),
    )

    def __init__(self) -> None:
        super().__init__(prefixes=("deploy",), family="deploy")

    # ------------------------------------------------------------------ #
    # Record helpers                                                     #
    # ------------------------------------------------------------------ #
    def _gates(self) -> list[dict[str, Any]]:
        return [r for r in self._records if r.get("kind") == "gate"]

    def _deployments(self) -> list[dict[str, Any]]:
        return [r for r in self._records if r.get("kind") == "deployment"]

    def _find_gate(self, gate: str) -> dict[str, Any] | None:
        for r in self._records:
            if r.get("kind") == "gate" and r.get("gate") == gate:
                return r
        return None

    def _find_deployment(self, change: str) -> dict[str, Any] | None:
        for r in self._records:
            if r.get("kind") == "deployment" and r.get("change") == change:
                return r
        return None

    def _ok(
        self, stdout: str, effect: tuple[str, dict[str, Any]] | None = None
    ) -> CommandResult:
        changes = self._emit(effect[0], effect[1]) if effect else None
        return CommandResult(
            stdout=stdout, stderr="", exit_code=0, state_changes=changes
        )

    @staticmethod
    def _err(msg: str) -> CommandResult:
        return CommandResult(stdout="", stderr=msg, exit_code=1)

    # ------------------------------------------------------------------ #
    # Dispatch                                                           #
    # ------------------------------------------------------------------ #
    def execute(self, command: str) -> CommandResult:
        try:
            parts = shlex.split(command.strip())
        except ValueError as exc:
            return self._err(f"parse error: {exc}")
        if not parts or parts[0] != "deploy":
            return self._err("Not a deploy command")
        if len(parts) < 2:
            return CommandResult(
                stdout=(
                    "deploy <request|approve|gate-set|execute|cutover|rollback|"
                    "monitor|status> [options]"
                ),
                stderr="",
                exit_code=0,
            )

        sub = parts[1]
        args = parts[2:]
        handlers = {
            "request": self._cmd_request,
            "approve": self._cmd_approve,
            "gate-set": self._cmd_gate_set,
            "execute": self._cmd_execute,
            "cutover": self._cmd_cutover,
            "rollback": self._cmd_rollback,
            "monitor": self._cmd_monitor,
            "status": self._cmd_status,
        }
        handler = handlers.get(sub)
        if handler is None:
            return self._err(f"Unknown deploy subcommand: {sub}")
        return handler(args)

    def _cmd_request(self, args: list[str]) -> CommandResult:
        change = _get_arg(args, "--change")
        if not change:
            return self._err("--change is required")
        dep = self._find_deployment(change)
        if dep is None:
            self._records.append(
                {
                    "kind": "deployment",
                    "change": change,
                    "status": "requested",
                    "created_at": self._advance_tick(),
                }
            )
        else:
            dep["status"] = "requested"
        self._save()
        # Emitted so the prerequisite half of the gate protocol is gradable,
        # not just the terminal cutover. These are real state changes -- they
        # are exactly what ``deploy cutover``'s EffectPrecondition checks.
        return self._ok(
            f"deployment requested: {change}",
            ("deployment_requested", {"change": change}),
        )

    def _cmd_approve(self, args: list[str]) -> CommandResult:
        change = _get_arg(args, "--change")
        approver = _get_arg(args, "--approver")
        if not change:
            return self._err("--change is required")
        if not approver:
            return self._err("--approver is required")
        self._records.append(
            {
                "kind": "approval",
                "change": change,
                "approver": approver,
                "created_at": self._advance_tick(),
            }
        )
        self._save()
        return self._ok(
            f"deployment approved: {change} by {approver}",
            ("deployment_approved", {"change": change, "approver": approver}),
        )

    def _cmd_gate_set(self, args: list[str]) -> CommandResult:
        gate = _get_arg(args, "--gate")
        state = _get_arg(args, "--state")
        if not gate:
            return self._err("--gate is required")
        if not state:
            return self._err("--state is required")
        if state not in _VALID_GATE_STATES:
            return self._err(
                "--state must be one of: " + ", ".join(sorted(_VALID_GATE_STATES))
            )
        g = self._find_gate(gate)
        if g is None:
            self._records.append(
                {
                    "kind": "gate",
                    "gate": gate,
                    "state": state,
                    "created_at": self._advance_tick(),
                }
            )
        else:
            g["state"] = state
        self._save()
        return self._ok(
            f"gate set: {gate} = {state}",
            ("deployment_gate_set", {"gate": gate, "state": state}),
        )

    def _cmd_execute(self, args: list[str]) -> CommandResult:
        change = _get_arg(args, "--change")
        if not change:
            return self._err("--change is required")
        dep = self._find_deployment(change)
        if dep is None:
            return self._err(f"no requested deployment for change: {change}")
        dep["status"] = "executed"
        self._save()
        return self._ok(
            f"deployment executed: {change}",
            effect=("deployment_executed", {"change": change}),
        )

    def _cmd_cutover(self, args: list[str]) -> CommandResult:
        change = _get_arg(args, "--change")
        if not change:
            return self._err("--change is required")
        dep = self._find_deployment(change)
        if dep is None:
            return self._err(f"no deployment for change: {change}")

        gates = self._gates()
        if not gates:
            return self._err("cutover blocked: no readiness gates defined")
        not_ready = [g.get("gate", "") for g in gates if g.get("state") != "ready"]
        if not_ready:
            return self._err(
                "cutover blocked: gates not ready: " + ", ".join(not_ready)
            )

        dep["status"] = "cutover"
        self._save()
        return self._ok(
            f"cutover complete: {change}",
            effect=("deployment_cutover", {"change": change}),
        )

    def _cmd_rollback(self, args: list[str]) -> CommandResult:
        change = _get_arg(args, "--change")
        if not change:
            return self._err("--change is required")
        dep = self._find_deployment(change)
        if dep is not None:
            dep["status"] = "rolledback"
            self._save()
        return self._ok(
            f"deployment rolled back: {change}",
            effect=("deployment_rolledback", {"change": change}),
        )

    def _cmd_monitor(self, args: list[str]) -> CommandResult:
        change = _get_arg(args, "--change")
        if not change:
            return self._err("--change is required")
        dep = self._find_deployment(change)
        status = dep.get("status") if dep else "unknown"
        return self._ok(f"monitor {change}: status={status}")

    def _cmd_status(self, args: list[str]) -> CommandResult:
        change = _get_arg(args, "--change")
        lines: list[str] = []
        gates = self._gates()
        if gates:
            lines.append("Gates:")
            for g in gates:
                lines.append(f"  {g.get('gate', '')} = {g.get('state', '')}")
        deployments = self._deployments()
        if change:
            deployments = [d for d in deployments if d.get("change") == change]
        if deployments:
            lines.append("Deployments:")
            for d in deployments:
                lines.append(f"  {d.get('change', '')}: {d.get('status', '')}")
        if not lines:
            return self._ok("No deployments or gates.")
        return self._ok("\n".join(lines))


def _get_arg(args: list[str], flag: str, default: str | None = None) -> str | None:
    """Return ``--flag value`` or ``--flag=value``, else ``default``."""
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(flag + "="):
            return a[len(flag) + 1:]
    return default
