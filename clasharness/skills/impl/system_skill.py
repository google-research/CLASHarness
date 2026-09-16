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

"""System skill implementation with local states and config management."""

from __future__ import annotations

import json
import os
import shlex
import uuid
from pathlib import Path
from typing import Any

from clasharness.backend.base import CommandResult
from clasharness.skills.base import Skill
from clasharness.skills.spec import CommandSpec, RealKind, SkillSpec


def _get_arg(args: list[str], flag: str, default: str | None = None) -> str | None:
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
    return default


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value


def _set_path(cfg: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def _get_path(cfg: dict[str, Any], path: str) -> Any:
    parts = path.split(".")
    d = cfg
    for p in parts:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def _unset_path(cfg: dict[str, Any], path: str) -> None:
    parts = path.split(".")
    d = cfg
    for p in parts[:-1]:
        if not isinstance(d, dict) or p not in d:
            return
        d = d[p]
    if isinstance(d, dict):
        d.pop(parts[-1], None)


class SystemSkill(Skill):
    """System skill handles workspace setup, gateway control, configurations, crons, wait and undo."""

    SPEC = SkillSpec(
        name="system",
        category="system",
        summary="Workspace setup, gateway/health, config, cron, plugins, wait and undo.",
        prefixes=("system", "wait", "undo", "plugin"),
        real_kind=RealKind.LOCAL_PERSISTENT,
        commands=(
            CommandSpec("system status", "Show overall system status."),
            CommandSpec("system health", "Report component health."),
            CommandSpec("system doctor", "Run diagnostics."),
            CommandSpec(
                "system gateway",
                "Control/inspect the local gateway.",
                args={"--port": {"type": "integer"}},
            ),
            CommandSpec(
                "system config",
                "Get/set/unset configuration values.",
                args={
                    "--set": {"type": "string"},
                    "--get": {"type": "string"},
                    "--unset": {"type": "string"},
                },
                emits_effects=("config",),
            ),
            CommandSpec(
                "system cron",
                "Manage scheduled cron jobs.",
                emits_effects=("cron_jobs",),
            ),
            CommandSpec("system security", "Inspect/manage security tokens."),
            CommandSpec("system setup", "Set up the workspace."),
            CommandSpec("system devices", "List/pair devices."),
            CommandSpec("system logs", "Show recent logs."),
            CommandSpec("system plugins", "List installed plugins."),
            CommandSpec(
                "wait",
                "Wait for N simulated ticks.",
                args={"n": {"type": "integer"}},
            ),
            CommandSpec(
                "undo",
                "Undo the last N mutating commands.",
                args={"n": {"type": "integer"}},
            ),
            CommandSpec("plugin", "Alias for `system plugins`."),
        ),
        smoke=(
            "system status",
            "system health",
        ),
    )

    _snapshot_attrs = ("_config", "_cron_jobs", "_plugins", "_sessions", "_gateway_running")

    def __init__(self, agent_skill: Skill | None = None, channel_skill: Skill | None = None) -> None:
        super().__init__(prefixes=("system", "wait", "undo", "plugin"))
        self._agent_skill = agent_skill
        self._channel_skill = channel_skill
        
        self._state_dir: str = ""
        self._gateway_running = True
        self._gateway_port = 19999
        self._cron_jobs: list[dict[str, Any]] = []
        self._plugins: list[dict[str, Any]] = []
        self._sessions: list[dict[str, Any]] = []
        self._config: dict[str, Any] = {
            "gateway": {
                "port": self._gateway_port,
                "auth": {"mode": "none", "token": ""}
            },
            "agent": {
                "model": "anthropic/claude-sonnet-4-5-20250929"
            },
            "agents": {
                "defaults": {
                    "model": {
                        "primary": "anthropic/claude-sonnet-4-5-20250929"
                    }
                }
            }
        }
        self._initialized = False

    def initialize(self, state_dir: str, env_vars: dict[str, str]) -> None:
        self._state_dir = state_dir
        self._initialized = True
        
        # Preload configuration from disk if exists
        if self._state_dir:
            config_path = Path(self._state_dir) / "openclaw.json"
            if config_path.exists():
                try:
                    payload = json.loads(config_path.read_text(encoding="utf-8"))
                    if isinstance(payload, dict):
                        _deep_merge(self._config, payload)
                except Exception:
                    pass

    def execute(self, command: str) -> CommandResult:
        parts = shlex.split(command.strip())
        if not parts:
            return CommandResult(stdout="", stderr="Empty command", exit_code=1)

        family = parts[0]
        if family not in ("system", "wait", "undo", "plugin"):
            return CommandResult(stdout="", stderr=f"Unknown system command: {family}", exit_code=1)

        if family == "plugin":
            return self._cmd_plugins(parts[1:])


        if family == "wait":
            try:
                wait_seconds = int(parts[1]) if len(parts) > 1 else 0
            except ValueError:
                wait_seconds = 0
            return CommandResult(
                stdout=f"Waited for {wait_seconds} seconds.",
                stderr="",
                exit_code=0,
            )

        if family == "undo":
            try:
                steps = int(parts[1]) if len(parts) > 1 else 1
            except ValueError:
                steps = 1
            return CommandResult(
                stdout=f"Undid {steps} steps.",
                stderr="",
                exit_code=0,
            )

        # Handle system subcommands
        if len(parts) < 2:
            return CommandResult(
                stdout="Usage: system <status|health|doctor|gateway|config|models|cron|security|setup|devices|logs|plugins>",
                stderr="",
                exit_code=0,
            )

        subcmd = parts[1]
        args = parts[2:]

        handlers = {
            "status": self._cmd_status,
            "health": self._cmd_health,
            "doctor": self._cmd_doctor,
            "gateway": self._cmd_gateway,
            "config": self._cmd_config,
            "models": self._cmd_models,
            "cron": self._cmd_cron,
            "security": self._cmd_security,
            "setup": self._cmd_setup,
            "devices": self._cmd_devices,
            "logs": self._cmd_logs,
            "plugins": self._cmd_plugins,
        }

        handler = handlers.get(subcmd)
        if handler is None:
            return CommandResult(
                stdout="", stderr=f"Unknown system subcommand: {subcmd}", exit_code=1
            )
        return handler(args)

    def _cmd_status(self, args: list[str]) -> CommandResult:
        use_json = "--json" in args
        
        # Query sibling skills for counts
        agents = []
        if self._agent_skill and hasattr(self._agent_skill, "get_state"):
            agents = list(self._agent_skill.get_state().get("agents", {}).keys())
            
        channels = {}
        if self._channel_skill and hasattr(self._channel_skill, "get_state"):
            channels = self._channel_skill.get_state().get("channels", {})

        status = {
            "gateway": {
                "running": self._gateway_running,
                "port": self._gateway_port,
            },
            "channels": {
                name: {"status": ch["status"]}
                for name, ch in channels.items()
            },
            "agents": agents,
            "sessions": len(self._sessions),
        }
        if use_json:
            return CommandResult(
                stdout=json.dumps(status, indent=2), stderr="", exit_code=0
            )
            
        lines = [
            f"Gateway: {'running' if self._gateway_running else 'stopped'} "
            f"(port {self._gateway_port})",
            f"Channels: {len(channels)} configured",
        ]
        for name, ch in channels.items():
            lines.append(f"  - {name}: {ch['status']}")
        lines.append(f"Agents: {len(agents)}")
        lines.append(f"Sessions: {len(self._sessions)}")
        return CommandResult(stdout="\n".join(lines), stderr="", exit_code=0)

    def _cmd_health(self, args: list[str]) -> CommandResult:
        if not self._gateway_running:
            return CommandResult(
                stdout="",
                stderr="Gateway is not running.",
                exit_code=1,
            )
        return CommandResult(
            stdout="Gateway health: OK\nUptime: 1h 0m\nMemory: 128MB",
            stderr="",
            exit_code=0,
        )

    def _cmd_doctor(self, args: list[str]) -> CommandResult:
        agents_cnt = 0
        if self._agent_skill and hasattr(self._agent_skill, "get_state"):
            agents_cnt = len(self._agent_skill.get_state().get("agents", {}))
            
        channels_cnt = 0
        if self._channel_skill and hasattr(self._channel_skill, "get_state"):
            channels_cnt = len(self._channel_skill.get_state().get("channels", {}))

        checks = [
            ("Config file", "OK"),
            ("Gateway", "running" if self._gateway_running else "stopped"),
            ("Channels", f"{channels_cnt} configured"),
            ("Agents", f"{agents_cnt} configured"),
            ("Workspace", "OK"),
        ]
        lines = ["OpenClaw Doctor\n"]
        for name, status in checks:
            icon = "+" if status == "OK" or "configured" in status or status == "running" else "!"
            lines.append(f"  [{icon}] {name}: {status}")
        return CommandResult(stdout="\n".join(lines), stderr="", exit_code=0)

    def _cmd_gateway(self, args: list[str]) -> CommandResult:
        if not args:
            return CommandResult(
                stdout="Usage: system gateway <start|stop|restart|status>",
                stderr="",
                exit_code=0,
            )
        action = args[0]
        if action == "start":
            self._gateway_running = True
            return CommandResult(stdout=f"Gateway started on port {self._gateway_port}.", stderr="", exit_code=0)
        elif action == "stop":
            self._gateway_running = False
            return CommandResult(stdout="Gateway stopped.", stderr="", exit_code=0)
        elif action == "restart":
            self._gateway_running = True
            return CommandResult(stdout=f"Gateway restarted on port {self._gateway_port}.", stderr="", exit_code=0)
        elif action == "status":
            return self._cmd_status(args[1:])
        else:
            return CommandResult(stdout="", stderr=f"Unknown gateway action: {action}", exit_code=1)

    def _cmd_config(self, args: list[str]) -> CommandResult:
        if not args:
            return CommandResult(
                stdout=json.dumps(self._config, indent=2), stderr="", exit_code=0
            )

        action = args[0]
        if action == "get":
            if len(args) < 2:
                return CommandResult(stdout="", stderr="config path required", exit_code=1)
            value = _get_path(self._config, args[1])
            if value is None:
                return CommandResult(stdout="", stderr="path not found", exit_code=1)
            return CommandResult(stdout=str(value), stderr="", exit_code=0)

        if action == "set":
            if len(args) < 3:
                return CommandResult(stdout="", stderr="path and value required", exit_code=1)
            path = args[1]
            value = args[2]
            _set_path(self._config, path, value)
            
            # Persist back to openclaw.json on disk
            if self._state_dir:
                config_path = Path(self._state_dir) / "openclaw.json"
                try:
                    config_path.write_text(json.dumps(self._config, indent=2), encoding="utf-8")
                except Exception:
                    pass
            return CommandResult(
                stdout=f"Updated {path}.",
                stderr="",
                exit_code=0,
                state_changes={"config": {path: value}},
            )

        if action == "unset":
            if len(args) < 2:
                return CommandResult(stdout="", stderr="path required", exit_code=1)
            path = args[1]
            _unset_path(self._config, path)
            if self._state_dir:
                config_path = Path(self._state_dir) / "openclaw.json"
                try:
                    config_path.write_text(json.dumps(self._config, indent=2), encoding="utf-8")
                except Exception:
                    pass
            return CommandResult(stdout=f"Unset {path}.", stderr="", exit_code=0)

        return CommandResult(stdout="", stderr=f"Unknown config action: {action}", exit_code=1)

    def _cmd_models(self, args: list[str]) -> CommandResult:
        """Mirror of MockBackend._cmd_models.

        The ``system.models_set`` spec entry declares ``multi`` as a supported
        mode, but this handler used to be missing entirely, so every
        ``openclaw models set ...`` fell through to "Unknown system
        subcommand: models" and left ``agent.model`` at its default.
        """
        current = (
            _get_path(self._config, "agents.defaults.model.primary")
            or self._config.get("agent", {}).get("model", "not set")
        )
        if not args:
            return CommandResult(
                stdout=f"Current model: {current}\n\n"
                "Available models:\n"
                "  - anthropic/claude-opus-4-6\n"
                "  - anthropic/claude-sonnet-4-5-20250929\n"
                "  - openai/gpt-4o\n"
                "  - google/gemini-2.0-flash",
                stderr="",
                exit_code=0,
            )

        if args[0] == "set":
            model = args[1] if len(args) > 1 else ""
            self._config.setdefault("agent", {})["model"] = model
            _set_path(self._config, "agents.defaults.model.primary", model)
            return CommandResult(
                stdout=f"Model set to {model}.",
                stderr="",
                exit_code=0,
                state_changes={
                    "config": {
                        "agent.model": model,
                        "agents.defaults.model.primary": model,
                    }
                },
            )

        return CommandResult(
            stdout="", stderr=f"Unknown models action: {args[0]}", exit_code=1
        )

    def _cmd_cron(self, args: list[str]) -> CommandResult:
        if not args:
            return CommandResult(
                stdout="Usage: system cron <add|list|delete>",
                stderr="",
                exit_code=0,
            )

        action = args[0]
        if action == "list":
            if not self._cron_jobs:
                return CommandResult(stdout="No cron jobs configured.", stderr="", exit_code=0)
            lines = ["Cron jobs:"]
            for job in self._cron_jobs:
                schedule = job.get("schedule") or job.get("cron") or job.get("every", "")
                message = job.get("message") or job.get("command", "")
                lines.append(f"  [{job['id']}] {schedule} - {message}")
            return CommandResult(stdout="\n".join(lines), stderr="", exit_code=0)

        if action == "add":
            schedule = _get_arg(args, "--cron") or _get_arg(args, "--schedule", "")
            every = _get_arg(args, "--every", "")
            message = _get_arg(args, "--message", "")
            command = _get_arg(args, "--command", "")
            name = _get_arg(args, "--name", f"job-{uuid.uuid4().hex[:6]}")
            normalized_message = message or command
            if not name:
                return CommandResult(stdout="", stderr="--name is required", exit_code=1)
            if not schedule and not every:
                return CommandResult(stdout="", stderr="--cron or --schedule is required", exit_code=1)
            if not normalized_message:
                return CommandResult(stdout="", stderr="--message or --command is required", exit_code=1)
                
            payload: dict[str, Any] = {
                "id": uuid.uuid4().hex[:6],
                "name": name,
                "status": "active",
            }
            if schedule:
                payload["schedule"] = schedule
            if every:
                payload["every"] = every
            if message:
                payload["message"] = message
            elif command:
                payload["command"] = command
            self._cron_jobs.append(payload)
            return CommandResult(
                stdout=f"Cron job created: {payload['id']}",
                stderr="",
                exit_code=0,
                state_changes={"cron_jobs": [payload]},
            )

        if action in ("delete", "rm"):
            job_id = args[1] if len(args) > 1 else ""
            self._cron_jobs = [j for j in self._cron_jobs if j["id"] != job_id]
            return CommandResult(stdout=f"Cron job '{job_id}' deleted.", stderr="", exit_code=0)

        return CommandResult(stdout="", stderr=f"Unknown cron action: {action}", exit_code=1)

    def _cmd_security(self, args: list[str]) -> CommandResult:
        if args and args[0] == "token" and len(args) > 1 and args[1] == "set":
            args = ["set-token"] + args[2:]

        auth = self._config.setdefault("gateway", {}).setdefault("auth", {})
        if not args:
            return CommandResult(
                stdout=f"Auth mode: {auth.get('mode', 'none')}\nToken configured: {'yes' if auth.get('token') else 'no'}",
                stderr="",
                exit_code=0,
            )

        if args[0] == "audit":
            return CommandResult(
                stdout="Security audit summary: 0 critical · 0 warn · 1 info",
                stderr="",
                exit_code=0,
            )

        if args[0] == "set-token":
            token = args[1] if len(args) > 1 else uuid.uuid4().hex
            auth["token"] = token
            auth["mode"] = "token"
            return CommandResult(stdout="Security token updated.", stderr="", exit_code=0)

        return CommandResult(stdout="", stderr=f"Unknown security action: {args[0]}", exit_code=1)

    def _cmd_setup(self, args: list[str]) -> CommandResult:
        return CommandResult(stdout="Workspace initialized successfully.", stderr="", exit_code=0)

    def _cmd_devices(self, args: list[str]) -> CommandResult:
        if not args:
            return CommandResult(stdout="Usage: system devices <pair|list|remove>", stderr="", exit_code=0)
        if args[0] == "list":
            return CommandResult(stdout="No devices paired.", stderr="", exit_code=0)
        if args[0] == "pair":
            return CommandResult(stdout="Device pairing initiated. Follow instructions on your device.", stderr="", exit_code=0)
        return CommandResult(stdout="", stderr=f"Unknown devices action: {args[0]}", exit_code=1)

    def _cmd_logs(self, args: list[str]) -> CommandResult:
        return CommandResult(
            stdout="[2025-01-01 12:00:00] Gateway started\n[2025-01-01 12:00:02] Agent turn processed",
            stderr="",
            exit_code=0,
        )

    def _cmd_plugins(self, args: list[str]) -> CommandResult:
        if not args:
            return CommandResult(
                stdout="Usage: system plugins <enable|disable|install|list>",
                stderr="",
                exit_code=0,
            )
        action = args[0]

        # ``openclaw skills info NAME`` is rewritten to ``plugin skill info
        # NAME`` (see ``openclaw_compat.rewrite_command``) and ``plugin skill
        # info NAME`` is a documented command in ``agent_tools.v1.json``
        # (``plugin.skill_info``). This skill owns the ``plugin`` prefix in the
        # multi-app backend, but had no ``skill`` branch, so the documented
        # command fell through to "Unknown plugins action: skill" -- exit 1 with
        # empty stdout. MockBackend has implemented it all along; the two
        # backends had simply drifted.
        if action == "skill":
            return self._cmd_skill(args[1:])

        if action == "list":
            if not self._plugins:
                return CommandResult(stdout="No plugins installed.", stderr="", exit_code=0)
            lines = ["Installed plugins:"]
            for p in self._plugins:
                lines.append(f"  - {p['name']} v{p.get('version', '1.0.0')}")
            return CommandResult(stdout="\n".join(lines), stderr="", exit_code=0)

        if action in ("install", "enable"):
            name = args[1] if len(args) > 1 else ""
            if not name:
                return CommandResult(stdout="", stderr="Plugin name required", exit_code=1)
            # ``install`` used to report "enabled" too, so an installed plugin
            # never reported an install. Mirror MockBackend, which already
            # distinguishes the two verbs and reports the already-present case.
            action_word = "installed" if action == "install" else "enabled"
            if any(p["name"] == name for p in self._plugins):
                return CommandResult(
                    stdout=f"Plugin '{name}' is already {action_word}.",
                    stderr="",
                    exit_code=0,
                )
            plugin = {"name": name, "version": "1.0.0", "status": "active"}
            self._plugins.append(plugin)
            # ``plugins`` is the only state-change key this skill used to drop:
            # Environment._track_effects maps it to the ``plugins_installed``
            # effect (see core/environment.py), which 114 shipped tasks assert
            # on. Without it every one of them was unsatisfiable under the
            # multi-app backend even for the expert solution, while the same
            # tasks passed under MockBackend -- which has always emitted it.
            return CommandResult(
                stdout=f"Plugin '{name}' {action_word}.",
                stderr="",
                exit_code=0,
                state_changes={"plugins": [plugin]},
            )

        # ``plugin disable NAME`` is documented (``plugin.disable``) but was
        # never implemented here either.
        if action == "disable":
            name = args[1] if len(args) > 1 else ""
            if not name:
                return CommandResult(stdout="", stderr="Plugin name required", exit_code=1)
            self._plugins = [p for p in self._plugins if p["name"] != name]
            return CommandResult(stdout=f"Plugin '{name}' disabled.", stderr="", exit_code=0)

        return CommandResult(stdout="", stderr=f"Unknown plugins action: {action}", exit_code=1)

    # Skill catalogue served by ``plugin skill …``. Kept identical to
    # ``MockBackend._cmd_skills`` so the mock and multi-app backends cannot
    # answer the same documented command differently.
    _SKILL_CATALOGUE = ("healthcheck", "skill-creator", "tmux", "weather")

    def _cmd_skill(self, args: list[str]) -> CommandResult:
        if not args or args[0] == "list":
            listing = "\n".join(f"  - {s}" for s in self._SKILL_CATALOGUE)
            return CommandResult(stdout=f"Skills:\n{listing}", stderr="", exit_code=0)
        if args[0] == "info":
            name = args[1] if len(args) > 1 else ""
            return CommandResult(
                stdout=f"Skill: {name}\nDescription: mock skill details",
                stderr="",
                exit_code=0,
            )
        return CommandResult(
            stdout="", stderr=f"Unknown skill action: {args[0]}", exit_code=1
        )

    def get_state(self) -> dict[str, Any]:
        return {
            "config": self._config,
            "cron_jobs": self._cron_jobs,
            "plugins": self._plugins,
        }

    def cleanup(self) -> None:
        self._cron_jobs = []
        self._plugins = []
        self._sessions = []
