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

"""Agent skill implementation with local in-memory registry."""

from __future__ import annotations

import shlex
import uuid
from typing import Any

from clasharness.backend.base import CommandResult
from clasharness.skills.base import Skill
from clasharness.skills.spec import CommandSpec, RealKind, SkillSpec


def _get_arg(args: list[str], flag: str, default: str | None = None) -> str | None:
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
    return default


class AgentSkill(Skill):
    """Agent skill handles registering agents, updating identities, and delegating tasks."""

    SPEC = SkillSpec(
        name="agent",
        category="agent",
        summary="Sub-agent registry and task delegation (deterministic, no live LLM).",
        prefixes=("agent",),
        real_kind=RealKind.DETERMINISTIC,
        commands=(
            CommandSpec(
                "agent list",
                "List registered agents.",
            ),
            CommandSpec(
                "agent add",
                "Register a new agent by name.",
                args={"name": {"type": "string"}, "--model": {"type": "string"}},
                required=("name",),
                emits_effects=("agents",),
            ),
            CommandSpec(
                "agent delete",
                "Remove an agent by name.",
                args={"name": {"type": "string"}},
                required=("name",),
                emits_effects=("agents",),
            ),
            CommandSpec(
                "agent set-identity",
                "Set an agent's identity/model.",
                args={"--id": {"type": "string"}, "--model": {"type": "string"}},
                emits_effects=("agents",),
            ),
            CommandSpec(
                "agent delegate",
                "Delegate a task to an agent.",
                args={
                    "--id": {"type": "string"},
                    "--instruction": {"type": "string"},
                },
                required=("--instruction",),
                emits_effects=("delegated_tasks",),
            ),
            CommandSpec(
                "agent status",
                "Show an agent's status.",
                args={"--id": {"type": "string"}},
            ),
        ),
        smoke=(
            "agent add alice",
            "agent list",
        ),
    )

    _snapshot_attrs = ("_agents", "_delegated_tasks")

    def __init__(self) -> None:
        super().__init__(prefixes=("agent",))
        self._agents: dict[str, dict[str, Any]] = {}
        self._delegated_tasks: list[dict[str, Any]] = []
        self._state_dir: str = ""
        self._initialized = False

    def initialize(self, state_dir: str, env_vars: dict[str, str]) -> None:
        self._state_dir = state_dir
        self._agents = {}
        self._delegated_tasks = []
        self._initialized = True

    def execute(self, command: str) -> CommandResult:
        parts = shlex.split(command.strip())
        if not parts or parts[0] != "agent":
            return CommandResult(stdout="", stderr="Not an agent command", exit_code=1)

        if len(parts) < 2:
            return CommandResult(
                stdout="Usage: agent <add|delete|list|set-identity|delegate|status>",
                stderr="",
                exit_code=0,
            )

        action = parts[1]
        args = parts[2:]

        handlers = {
            "list": self._action_list,
            "add": self._action_add,
            "delete": self._action_delete,
            "set-identity": self._action_set_identity,
            "delegate": self._action_delegate,
            "status": self._action_status,
        }

        handler = handlers.get(action)
        if handler is None:
            return CommandResult(
                stdout="", stderr=f"Unknown agent action: {action}", exit_code=1
            )
        return handler(args)

    def _action_list(self, args: list[str]) -> CommandResult:
        if not self._agents:
            return CommandResult(stdout="No agents configured.", stderr="", exit_code=0)
        lines = ["Agents:"]
        for name, agent in self._agents.items():
            model = agent.get("model", "default")
            emoji = agent.get("emoji", "")
            role = agent.get("role", "")
            info = f" (model: {model}"
            if emoji:
                info += f", emoji: {emoji}"
            if role:
                info += f", role: {role}"
            info += ")"
            lines.append(f"  - {name}{info}")
        return CommandResult(stdout="\n".join(lines), stderr="", exit_code=0)

    def _action_add(self, args: list[str]) -> CommandResult:
        name = _get_arg(args, "--name")
        if not name and args and not args[0].startswith("-"):
            name = args[0]
        if not name:
            name = "agent-" + uuid.uuid4().hex[:6]
        model = _get_arg(args, "--model", "anthropic/claude-sonnet-4-5-20250929")

        if name in self._agents:
            return CommandResult(
                stdout="", stderr=f"Agent '{name}' already exists.", exit_code=1
            )
        self._agents[name] = {"name": name, "model": model}
        return CommandResult(
            stdout=f"Agent '{name}' created with model {model}.",
            stderr="",
            exit_code=0,
            state_changes={"agents": {name: self._agents[name]}},
        )

    def _action_delete(self, args: list[str]) -> CommandResult:
        name = _get_arg(args, "--name") or (args[0] if args and not args[0].startswith("-") else None)
        if not name or name not in self._agents:
            return CommandResult(
                stdout="", stderr=f"Agent '{name}' not found.", exit_code=1
            )
        del self._agents[name]
        return CommandResult(
            stdout=f"Agent '{name}' deleted.",
            stderr="",
            exit_code=0,
        )

    def _action_set_identity(self, args: list[str]) -> CommandResult:
        name = _get_arg(args, "--agent") or _get_arg(args, "--name")
        if not name and args and not args[0].startswith("-"):
            name = args[0]
        if not name or name not in self._agents:
            return CommandResult(
                stdout="", stderr=f"Agent '{name}' not found.", exit_code=1
            )
        
        emoji = _get_arg(args, "--emoji")
        role = _get_arg(args, "--role")
        if emoji:
            self._agents[name]["emoji"] = emoji
        if role:
            self._agents[name]["role"] = role

        return CommandResult(
            stdout=f"Agent '{name}' identity updated.",
            stderr="",
            exit_code=0,
        )

    def _action_delegate(self, args: list[str]) -> CommandResult:
        agent_name = _get_arg(args, "--agent", "default")
        instruction = _get_arg(args, "--instruction", "")
        if not instruction:
            return CommandResult(
                stdout="", stderr="--instruction is required", exit_code=1
            )

        task_id = f"task_{uuid.uuid4().hex[:8]}"
        task_record = {
            "id": task_id,
            "agent": agent_name,
            "instruction": instruction,
            "status": "pending",
        }
        self._delegated_tasks.append(task_record)

        # In a real system, this could invoke the subagent process or API.
        # In mock/default execution, we record it.
        return CommandResult(
            stdout=f"Task delegated to '{agent_name}'. Session: {task_id}",
            stderr="",
            exit_code=0,
            state_changes={"delegated_tasks": [task_record]},
        )

    def _action_status(self, args: list[str]) -> CommandResult:
        agent_name = _get_arg(args, "--agent")
        results = [
            t for t in self._delegated_tasks
            if not agent_name or t["agent"] == agent_name
        ]
        if not results:
            return CommandResult(stdout="No active tasks.", stderr="", exit_code=0)
        
        lines = ["Delegated Tasks Status:"]
        for t in results:
            lines.append(f"  - [{t['id']}] to {t['agent']}: status={t['status']}, instruction='{t['instruction']}'")
        return CommandResult(stdout="\n".join(lines), stderr="", exit_code=0)

    def get_state(self) -> dict[str, Any]:
        return {
            "agents": self._agents,
            "delegated_tasks": self._delegated_tasks,
        }

    def cleanup(self) -> None:
        self._agents = {}
        self._delegated_tasks = []
