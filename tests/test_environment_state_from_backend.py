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

from __future__ import annotations

from typing import Any

import pytest

from clasharness.backend.base import BaseBackend, CommandResult
from clasharness.core.environment import CLASHarnessEnv


class _StateOnlyBackend(BaseBackend):
    def initialize(self, state_dir: str, env_vars: dict[str, str]) -> None:
        self._state = {
            "agents": [{"id": "main", "model": "vllm/Qwen"}],
            "channels": [{"name": "telegram", "status": "connected"}],
            "messages": [{"target": "+123", "message": "hi"}],
            "cron_jobs": [{"id": "job_1"}],
            "plugins": [{"name": "discord"}],
            "calendar_events": [{"id": "evt_1", "title": "Standup"}],
            "emails": [
                {"id": "email_1", "folder": "sent", "to": "alice@example.com"}
            ],
            "files": {"/tmp/report.txt": "hello"},
            "tasks": [{"id": "task_1", "title": "Write report", "status": "done"}],
        }

    def execute_cli(self, command: str) -> CommandResult:
        return CommandResult(stdout="ok", stderr="", exit_code=0)

    def execute_python(self, code: str) -> CommandResult:
        return CommandResult(stdout="", stderr="python unsupported", exit_code=1)

    def get_gateway_status(self) -> dict[str, Any] | None:
        return None

    def get_config(self) -> dict[str, Any]:
        return {}

    def get_state(self) -> dict[str, Any]:
        return dict(self._state)

    def cleanup(self) -> None:
        pass


def test_environment_uses_backend_get_state_only(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "clasharness.core.environment._create_backend",
        lambda backend_type, **kwargs: _StateOnlyBackend(),
    )

    with CLASHarnessEnv(task_id="agent_create_1", backend="mock") as env:
        env.reset()
        env.step("openclaw status")
        state = env._get_evaluation_state()

    assert state["agents"]["main"]["model"] == "vllm/Qwen"
    assert state["channels"]["telegram"]["status"] == "connected"
    assert state["tasks_list"][0]["title"] == "Write report"
    assert state["emails"][0]["to"] == "alice@example.com"
    assert state["files"]["/tmp/report.txt"] == "hello"
