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

"""MultiAppBackend — routes CLI commands to per-app sub-backends."""

from __future__ import annotations

import json
import re
from typing import Any

from clasharness.backend.base import (
    ONLINE_DATA_ENV_FLAG,
    REAL_ENV_FLAG,
    BaseBackend,
    CommandResult,
)
from clasharness.backend.calendar_backend import CalendarBackend
from clasharness.backend.email_backend import EmailBackend
from clasharness.backend.weather_backend import WeatherBackend
from clasharness.backend.file_backend import FileSystemBackend
from clasharness.backend.asset_backend import AssetBackend
from clasharness.backend.tasks_backend import TasksBackend
from clasharness.backend.http_backend import HttpBackend
from clasharness.backend.openclaw_compat import rewrite_command
from clasharness.skills.adapters import (
    AssetSkillAdapter,
    FileSkillAdapter,
    HttpSkillAdapter,
)
from clasharness.skills.impl import (
    CalendarSkill,
    EmailSkill,
    TasksSkill,
    WeatherSkill,
    MessageSkill,
    AgentSkill,
    ChannelSkill,
    SystemSkill,
    GitSkill,
    JqSkill,
    GrepSkill,
    ShellExecutionSkill,
    LockSkill,
    DeploySkill,
    BookingSkill,
    PairingSkill,
    ApprovalSkill,
    ContainmentSkill,
    ReviewSkill,
)

from clasharness.skills.registry import SkillRegistry
from clasharness.skills.runtime import SkillRuntime
from clasharness.skills.extensions import SkillContext, build_extension_skills


class MultiAppBackend(BaseBackend):
    """Routing backend that delegates command execution to skills.

    Command routing is based on the first token of the command:
        calendar *  → CalendarBackend
        gcalcli *   → CalendarBackend
        email *     → EmailBackend
        weather *   → WeatherBackend
        file *      → FileSystemBackend
        asset *     → AssetBackend
        tasks *     → TasksBackend
        message *   → MessageSkill
        agent *     → AgentSkill
        channel *   → ChannelSkill
        system *    → SystemSkill
        curl *      → HttpBackend
    """

    def __init__(
        self,
        real_openclaw: bool = False,
        real_openclaw_kwargs: dict[str, Any] | None = None,
        fallback_openclaw_network_to_mock: bool = False,
        strict_online_data: bool = True,
        real_apps: bool = False,
    ) -> None:
        self._fallback_openclaw_network_to_mock = fallback_openclaw_network_to_mock
        self._strict_online_data = strict_online_data
        # ``real_openclaw`` = real curl/HTTP path. ``real_apps`` = real
        # (non-mock) execution path for every skill (calendar/email/weather/
        # tasks + the arena-internal skills). They are independent: HybridBackend
        # sets real_openclaw but keeps real_apps=False ("real gateway, mock
        # apps"); mode="real" sets both.
        self._real_openclaw = bool(real_openclaw)
        self._real_apps = bool(real_apps)

        self._message_skill = MessageSkill()
        self._email_skill = EmailSkill()
        self._email = EmailBackend(self._email_skill)
        self._file = FileSystemBackend()
        self._tasks_skill = TasksSkill()
        self._tasks = TasksBackend(self._tasks_skill)
        self._http = HttpBackend(real_http=real_openclaw, message_skill=self._message_skill)

        self._agent_skill = AgentSkill()
        self._channel_skill = ChannelSkill()
        self._system_skill = SystemSkill(
            agent_skill=self._agent_skill,
            channel_skill=self._channel_skill,
        )


        self._git_skill = GitSkill()
        self._jq_skill = JqSkill(file_backend=self._file)
        self._grep_skill = GrepSkill(file_backend=self._file)
        self._shell_exec_skill = ShellExecutionSkill()
        self._lock_skill = LockSkill()
        self._deploy_skill = DeploySkill()
        self._booking_skill = BookingSkill()
        self._pairing_skill = PairingSkill()
        self._approval_skill = ApprovalSkill()
        self._containment_skill = ContainmentSkill()
        self._review_skill = ReviewSkill()

        self._skill_registry = SkillRegistry()
        self._calendar_skill = CalendarSkill()
        self._weather_skill = WeatherSkill()
        
        self._calendar = CalendarBackend(self._calendar_skill)
        self._weather = WeatherBackend(self._weather_skill)
        self._asset = AssetBackend()

        # Register all skills to the registry
        self._skill_registry.register_skill(self._calendar_skill)
        self._skill_registry.register_skill(self._email_skill)
        self._skill_registry.register_skill(self._weather_skill)
        self._skill_registry.register_skill(FileSkillAdapter(self._file))
        self._skill_registry.register_skill(AssetSkillAdapter(self._asset))
        self._skill_registry.register_skill(self._tasks_skill)
        self._skill_registry.register_skill(HttpSkillAdapter(self._http))
        
        self._skill_registry.register_skill(self._message_skill)
        self._skill_registry.register_skill(self._agent_skill)
        self._skill_registry.register_skill(self._channel_skill)
        self._skill_registry.register_skill(self._system_skill)
        self._skill_registry.register_skill(self._git_skill)
        self._skill_registry.register_skill(self._jq_skill)
        self._skill_registry.register_skill(self._grep_skill)
        self._skill_registry.register_skill(self._shell_exec_skill)
        self._skill_registry.register_skill(self._lock_skill)
        self._skill_registry.register_skill(self._deploy_skill)
        self._skill_registry.register_skill(self._booking_skill)
        self._skill_registry.register_skill(self._pairing_skill)
        self._skill_registry.register_skill(self._approval_skill)
        self._skill_registry.register_skill(self._containment_skill)
        self._skill_registry.register_skill(self._review_skill)

        # Additive extension layer: instantiate & register any user-contributed
        # skills declared via @register_skill (see docs/mcp-extension-guide.md).
        # The core arena ships none, so this is a no-op unless extensions are
        # installed. Failures are collected (not raised) so a broken extension
        # cannot destabilize the core arena; inspect ``_extension_errors``.
        self._extension_errors: list[str] = []
        self._register_extension_skills()

        self._runtime = SkillRuntime(self._skill_registry)

    def _register_extension_skills(self) -> None:
        """Build and register user extension skills onto the registry."""
        try:
            import clasharness.skills.contrib  # noqa: F401  (runs @register_skill)
        except Exception as exc:  # pragma: no cover - defensive
            self._extension_errors.append(f"contrib import failed: {exc}")
        context = SkillContext(
            real_apps=self._real_apps,
            real_openclaw=self._real_openclaw,
            strict_online_data=self._strict_online_data,
            file_backend=self._file,
            message_skill=self._message_skill,
            asset_backend=self._asset,
            http_backend=self._http,
            agent_skill=self._agent_skill,
            channel_skill=self._channel_skill,
        )

        def _on_build_error(cls: type, exc: Exception) -> None:
            self._extension_errors.append(f"{cls.__name__}: build failed: {exc}")

        for skill in build_extension_skills(context, on_error=_on_build_error):
            try:
                self._skill_registry.register_skill(skill)
            except Exception as exc:  # noqa: BLE001 - surfaced, not fatal
                self._extension_errors.append(
                    f"{type(skill).__name__}: register failed: {exc}"
                )

    # ------------------------------------------------------------------ #
    # BaseBackend interface                                                 #
    # ------------------------------------------------------------------ #

    def initialize(self, state_dir: str, env_vars: dict[str, str]) -> None:
        merged_env = dict(env_vars)
        if self._real_apps:
            # Canonical real signal consumed by skills to pick their real
            # (non-mock) path, plus the legacy online-data flag still read by
            # external-service skills (calendar/email/weather/tasks).
            merged_env[REAL_ENV_FLAG] = "1"
            merged_env.setdefault(ONLINE_DATA_ENV_FLAG, "1")
        if self._strict_online_data:
            merged_env["CLASHARNESS_STRICT_ONLINE_DATA"] = "1"
        self._runtime.initialize(state_dir, merged_env)

    def set_virtual_time(self, now: int) -> None:
        """Thread the environment's virtual clock down to the skill runtime.

        Stateful protocol skills (locks, holds, pairing) read it to expire
        time-limited records. See :meth:`SkillRuntime.set_now`.
        """
        self._runtime.set_now(now)

    def execute_cli(self, command: str) -> CommandResult:
        # Pre-process command to normalize legacy openclaw commands (e.g. openclaw status)
        command = command.strip()
        decision = rewrite_command(command, skip_incompatible=False)
        executed_command = decision.executed_action.strip()
        
        # Route to native skills layer
        return self._runtime.execute_cli(executed_command)

    def execute_python(self, code: str) -> CommandResult:
        # Emulate python interactions by calling equivalent CLI skills
        if "client.status" in code:
            return self._system_skill.execute("system status")
        if "client.agents.add" in code:
            match = re.search(r"name=['\"](\w+)['\"]", code)
            name = match.group(1) if match else "default"
            return self._agent_skill.execute(f"agent add {name}")
        if "client.agents.list" in code:
            return self._agent_skill.execute("agent list")
        return CommandResult(
            stdout="Mock Python execution: OK",
            stderr="",
            exit_code=0,
        )

    def get_gateway_status(self) -> dict[str, Any] | None:
        # Gateway status is maintained inside system skill
        status_res = self._system_skill.execute("system status --json")
        if status_res.exit_code == 0:
            try:
                data = json.loads(status_res.stdout)
                return data.get("gateway")
            except Exception:
                pass
        return None

    def get_config(self) -> dict[str, Any]:
        # Config dictionary is maintained inside system skill
        return self._system_skill.get_state().get("config", {})

    def cleanup(self) -> None:
        self._runtime.cleanup()

    # ------------------------------------------------------------------ #
    # Multi-app state access                                               #
    # ------------------------------------------------------------------ #

    def get_state(self) -> dict[str, Any]:
        """Merge state from all registered skills into one dict."""
        return self._runtime.get_state()

    def snapshot_state(self) -> dict[str, Any]:
        """Capture all skill state for fast in-memory snapshot/restore (undo)."""
        return self._runtime.snapshot_state()

    def restore_state(self, snapshot: dict[str, Any]) -> None:
        """Restore skill state captured by :meth:`snapshot_state`."""
        self._runtime.restore_state(snapshot)

    # Legacy compatibility accessors for existing tests/callers.
    @property
    def calendar_backend(self) -> CalendarBackend:
        return self._calendar

    @property
    def email_backend(self) -> EmailBackend:
        return self._email

    @property
    def file_backend(self) -> FileSystemBackend:
        return self._file

    @property
    def asset_backend(self) -> AssetBackend:
        return self._asset

    @property
    def tasks_backend(self) -> TasksBackend:
        return self._tasks

    @property
    def mock_backend(self) -> Any:
        # Returns self, as MultiAppBackend now behaves as the mock backend natively
        return self

    @property
    def real_enabled(self) -> bool:
        # True when skills should run their real (non-mock) path. Driven by
        # ``real_apps`` (mode="real"), NOT by ``real_openclaw`` — HybridBackend
        # keeps real_apps=False so its app services stay mocked.
        return self._real_apps

    @property
    def real_backend(self) -> BaseBackend | None:
        # Legacy property, no longer required as there is no RealOpenClawBackend.
        # Real behavior is now selected per-skill via ``real_enabled`` /
        # CLASHARNESS_REAL rather than a single delegate backend.
        return None

    def _should_fallback_to_mock(self, command: str, result: CommandResult) -> bool:
        # Network fallback checks are no longer needed without openclaw subprocesses
        return False
