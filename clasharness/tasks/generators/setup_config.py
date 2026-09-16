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

"""Task generators for Setup & Configuration domain (D1)."""

from __future__ import annotations

from typing import Any, Iterator

from clasharness.core.task import TaskData
from clasharness.tasks.registry import BaseTaskGenerator, Fail, Pass, SetupResult


@BaseTaskGenerator.register("setup_workspace")
class SetupWorkspaceGenerator(BaseTaskGenerator):
    """Generate tasks for initializing the openclaw workspace."""

    required_domains = ("setup_config",)
    difficulty = 1
    parameters = {}  # Single variation

    def setup(
        self, params: dict[str, Any], initial_config: dict[str, Any]
    ) -> Iterator[tuple[SetupResult, TaskData]]:
        data = TaskData()
        yield Pass(), data

    def get_instruction(self, params: dict[str, Any], data: TaskData) -> str:
        return "Initialize the openclaw workspace by running the setup command."

    def get_solution(self, params: dict[str, Any], data: TaskData) -> list[str]:
        return ["openclaw setup"]

    def get_evaluation_checks(
        self, params: dict[str, Any], data: TaskData
    ) -> list[dict[str, Any]]:
        return [
            {
                "type": "output",
                "condition": "contains",
                "expected": "initialized",
                "output_field": "last_stdout",
                "name": "workspace initialized",
            }
        ]


@BaseTaskGenerator.register("configure_model")
class ConfigureModelGenerator(BaseTaskGenerator):
    """Generate tasks for configuring the AI model."""

    required_domains = ("setup_config",)
    difficulty = 1
    parameters = {
        "model": [
            "anthropic/claude-opus-4-6",
            "anthropic/claude-sonnet-4-5-20250929",
            "openai/gpt-4o",
            "google/gemini-2.0-flash",
        ],
    }

    def setup(
        self, params: dict[str, Any], initial_config: dict[str, Any]
    ) -> Iterator[tuple[SetupResult, TaskData]]:
        data = TaskData(
            public={"model": params["model"]},
            private={"expected_model": params["model"]},
        )
        yield Pass(), data

    def get_instruction(self, params: dict[str, Any], data: TaskData) -> str:
        return f"Configure openclaw to use the model '{params['model']}' as the default agent model."

    def get_solution(self, params: dict[str, Any], data: TaskData) -> list[str]:
        return [f"openclaw models set {params['model']}"]

    def get_evaluation_checks(
        self, params: dict[str, Any], data: TaskData
    ) -> list[dict[str, Any]]:
        return [
            {
                "type": "config",
                "config_path": "agent.model",
                "condition": "equals",
                "expected": params["model"],
                "name": "model configured correctly",
            }
        ]


@BaseTaskGenerator.register("configure_gateway_port")
class ConfigureGatewayPortGenerator(BaseTaskGenerator):
    """Generate tasks for configuring the gateway port."""

    required_domains = ("setup_config",)
    difficulty = 1
    parameters = {
        "port": [8080, 9090, 18789, 3000],
    }

    def setup(
        self, params: dict[str, Any], initial_config: dict[str, Any]
    ) -> Iterator[tuple[SetupResult, TaskData]]:
        data = TaskData(
            public={"port": params["port"]},
            private={"expected_port": params["port"]},
        )
        yield Pass(), data

    def get_instruction(self, params: dict[str, Any], data: TaskData) -> str:
        return f"Configure the openclaw gateway to listen on port {params['port']}."

    def get_solution(self, params: dict[str, Any], data: TaskData) -> list[str]:
        return [f"openclaw config set gateway.port {params['port']}"]

    def get_evaluation_checks(
        self, params: dict[str, Any], data: TaskData
    ) -> list[dict[str, Any]]:
        return [
            {
                "type": "config",
                "config_path": "gateway.port",
                "condition": "equals",
                "expected": str(params["port"]),
                "name": "gateway port configured",
            }
        ]
