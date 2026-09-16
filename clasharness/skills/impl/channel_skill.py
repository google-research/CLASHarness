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

"""Channel skill implementation with local in-memory states."""

from __future__ import annotations

import json
import shlex
from typing import Any

from clasharness.backend.base import CommandResult
from clasharness.skills.base import Skill
from clasharness.skills.spec import CommandSpec, RealKind, SkillSpec


def _get_arg(args: list[str], flag: str, default: str | None = None) -> str | None:
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
    return default


class ChannelSkill(Skill):
    """Channel skill handles logging in to message channels and configuring them."""

    SPEC = SkillSpec(
        name="channel",
        category="comms",
        summary="Message-channel registration, login and configuration.",
        prefixes=("channel",),
        real_kind=RealKind.LOCAL_PERSISTENT,
        commands=(
            CommandSpec("channel list", "List known channels."),
            CommandSpec(
                "channel status",
                "Show a channel's status.",
                args={"--channel": {"type": "string"}},
            ),
            CommandSpec(
                "channel add",
                "Register a new channel.",
                args={"--channel": {"type": "string"}},
                required=("--channel",),
            ),
            CommandSpec(
                "channel login",
                "Authenticate to a channel.",
                args={"--channel": {"type": "string"}},
                required=("--channel",),
            ),
            CommandSpec(
                "channel config",
                "Show a channel's configuration.",
                args={"--channel": {"type": "string"}},
            ),
            CommandSpec(
                "channel set",
                "Set a channel configuration value.",
                args={
                    "--channel": {"type": "string"},
                    "--key": {"type": "string"},
                    "--value": {"type": "string"},
                },
            ),
        ),
        smoke=(
            "channel add --channel slack",
            "channel list",
        ),
    )

    _snapshot_attrs = ("_channels",)

    def __init__(self) -> None:
        super().__init__(prefixes=("channel",))
        self._channels: dict[str, dict[str, Any]] = {}
        self._state_dir: str = ""
        self._initialized = False

    def initialize(self, state_dir: str, env_vars: dict[str, str]) -> None:
        self._state_dir = state_dir
        self._channels = {}
        self._initialized = True

    def execute(self, command: str) -> CommandResult:
        parts = shlex.split(command.strip())
        if not parts or parts[0] != "channel":
            return CommandResult(stdout="", stderr="Not a channel command", exit_code=1)

        if len(parts) < 2:
            return CommandResult(
                stdout="Usage: channel <login|add|config|set|list|status>",
                stderr="",
                exit_code=0,
            )

        action = parts[1]
        args = parts[2:]

        handlers = {
            "list": self._action_list,
            "status": self._action_list,
            "add": self._action_add,
            "login": self._action_login,
            "config": self._action_config,
            "set": self._action_config,
        }

        handler = handlers.get(action)
        if handler is None:
            return CommandResult(
                stdout="", stderr=f"Unknown channel action: {action}", exit_code=1
            )
        return handler(args)

    def _action_list(self, args: list[str]) -> CommandResult:
        if not self._channels:
            return CommandResult(stdout="No channels configured.", stderr="", exit_code=0)
        if "--json" in args:
            payload = {
                name: {
                    "status": ch.get("status", "unknown"),
                    "config": ch.get("config", {}),
                }
                for name, ch in self._channels.items()
            }
            return CommandResult(
                stdout=json.dumps(payload, indent=2),
                stderr="",
                exit_code=0,
            )
        lines = ["Channels:"]
        for name, ch in self._channels.items():
            lines.append(f"  - {name}: {ch['status']}")
        return CommandResult(stdout="\n".join(lines), stderr="", exit_code=0)

    def _action_add(self, args: list[str]) -> CommandResult:
        channel = _get_arg(args, "--channel", "")
        if not channel:
            return CommandResult(stdout="", stderr="--channel is required", exit_code=1)
        self._channels[channel] = {
            "name": channel,
            "status": "connected",
            "config": {},
        }
        return CommandResult(
            stdout=f"Channel '{channel}' added.",
            stderr="",
            exit_code=0,
        )

    def _action_login(self, args: list[str]) -> CommandResult:
        channel = _get_arg(args, "--channel")
        if not channel:
            return CommandResult(stdout="", stderr="--channel is required", exit_code=1)
        self._channels[channel] = {
            "name": channel,
            "status": "connected",
            "config": {},
        }
        return CommandResult(
            stdout=f"Logged in to {channel}.",
            stderr="",
            exit_code=0,
        )

    def _action_config(self, args: list[str]) -> CommandResult:
        channel = _get_arg(args, "--channel") or (args[0] if args and not args[0].startswith("-") else "")
        if channel not in self._channels:
            return CommandResult(
                stdout="",
                stderr=f"Channel '{channel}' not found.",
                exit_code=1,
            )
        key = _get_arg(args, "--key")
        value = _get_arg(args, "--value")
        if key is not None and value is not None:
            self._channels[channel].setdefault("config", {})[key] = value
            return CommandResult(
                stdout=f"Configured channel '{channel}': {key}={value}.",
                stderr="",
                exit_code=0,
            )
        return CommandResult(
            stdout="",
            stderr="--key and --value are required",
            exit_code=1,
        )

    def get_state(self) -> dict[str, Any]:
        return {"channels": self._channels}

    def cleanup(self) -> None:
        self._channels = {}
