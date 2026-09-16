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

"""Task generators for Device & Node domain (D9)."""

from __future__ import annotations

from typing import Any, Iterator

from clasharness.core.task import TaskData
from clasharness.tasks.registry import BaseTaskGenerator, Pass, SetupResult


@BaseTaskGenerator.register("pair_device")
class PairDeviceGenerator(BaseTaskGenerator):
    """Initiate device pairing."""

    required_domains = ("device_node",)
    difficulty = 1
    parameters = {}

    def setup(
        self, params: dict[str, Any], initial_config: dict[str, Any]
    ) -> Iterator[tuple[SetupResult, TaskData]]:
        yield Pass(), TaskData()

    def get_instruction(self, params: dict[str, Any], data: TaskData) -> str:
        return "Initiate device pairing to connect a new device to openclaw."

    def get_solution(self, params: dict[str, Any], data: TaskData) -> list[str]:
        return ["openclaw devices pair"]

    def get_evaluation_checks(
        self, params: dict[str, Any], data: TaskData
    ) -> list[dict[str, Any]]:
        return [
            {
                "type": "output",
                "condition": "contains",
                "expected": "pairing",
                "output_field": "last_stdout",
                "name": "device pairing initiated",
                "ignore_case": True,
            }
        ]


@BaseTaskGenerator.register("list_devices")
class ListDevicesGenerator(BaseTaskGenerator):
    """List all paired devices."""

    required_domains = ("device_node",)
    difficulty = 1
    parameters = {}

    def setup(
        self, params: dict[str, Any], initial_config: dict[str, Any]
    ) -> Iterator[tuple[SetupResult, TaskData]]:
        yield Pass(), TaskData()

    def get_instruction(self, params: dict[str, Any], data: TaskData) -> str:
        return "List all devices currently paired with openclaw."

    def get_solution(self, params: dict[str, Any], data: TaskData) -> list[str]:
        return ["openclaw devices list"]

    def get_evaluation_checks(
        self, params: dict[str, Any], data: TaskData
    ) -> list[dict[str, Any]]:
        return [
            {
                "type": "output",
                "condition": "regex",
                "expected": r"(device|Device|No devices|paired)",
                "output_field": "last_stdout",
                "name": "devices listed",
                "ignore_case": True,
            }
        ]


@BaseTaskGenerator.register("pair_and_list_devices")
class PairAndListDevicesGenerator(BaseTaskGenerator):
    """Multi-step: pair a device and list devices to verify."""

    required_domains = ("device_node",)
    difficulty = 2
    parameters = {}

    def setup(
        self, params: dict[str, Any], initial_config: dict[str, Any]
    ) -> Iterator[tuple[SetupResult, TaskData]]:
        yield Pass(), TaskData()

    def get_instruction(self, params: dict[str, Any], data: TaskData) -> str:
        return (
            "Initiate device pairing to connect a new device, "
            "then list all paired devices to verify the pairing process started."
        )

    def get_solution(self, params: dict[str, Any], data: TaskData) -> list[str]:
        return [
            "openclaw devices pair",
            "openclaw devices list",
        ]

    def get_evaluation_checks(
        self, params: dict[str, Any], data: TaskData
    ) -> list[dict[str, Any]]:
        return [
            {
                "type": "output",
                "condition": "regex",
                "expected": r"(device|Device|No devices|paired)",
                "output_field": "last_stdout",
                "name": "devices listed after pairing",
                "ignore_case": True,
            }
        ]
