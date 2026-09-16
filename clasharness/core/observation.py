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

"""Observation and action space definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Back-compat re-export: EvaluationResult/CheckResult now live in
# clasharness.evaluation.results (canonical home). Imported here so existing
# ``from clasharness.core.observation import EvaluationResult, CheckResult``
# call sites keep working.
from clasharness.evaluation.results import (  # noqa: F401
    CheckResult,
    EvaluationResult,
)


@dataclass
class Observation:
    """What the agent sees after each step."""

    command_output: str = ""
    error_output: str | None = None
    exit_code: int = 0
    current_config: dict[str, Any] = field(default_factory=dict)
    gateway_status: dict[str, Any] | None = None
    available_commands: list[str] = field(default_factory=list)
    task_instruction: str = ""
    step_number: int = 0

    def to_text(self) -> str:
        """Render observation as text for agent consumption."""
        parts = []
        if self.task_instruction and self.step_number == 0:
            parts.append(f"[TASK] {self.task_instruction}")
        if self.command_output:
            parts.append(f"[STDOUT]\n{self.command_output}")
        if self.error_output:
            parts.append(f"[STDERR]\n{self.error_output}")
        parts.append(f"[EXIT CODE] {self.exit_code}")
        if self.available_commands:
            parts.append(
                f"[AVAILABLE COMMANDS]\n" + "\n".join(self.available_commands)
            )
        return "\n\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_output": self.command_output,
            "error_output": self.error_output,
            "exit_code": self.exit_code,
            "current_config": self.current_config,
            "gateway_status": self.gateway_status,
            "available_commands": self.available_commands,
            "task_instruction": self.task_instruction,
            "step_number": self.step_number,
        }


@dataclass
class StepRecord:
    """Record of a single agent step (for trajectory recording)."""

    step_number: int
    observation: dict[str, Any]
    action: str
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step_number,
            "observation": self.observation,
            "action": self.action,
            "reward": self.reward,
            "done": self.done,
            "info": self.info,
        }


@dataclass
class Trajectory:
    """Full trajectory of an episode."""

    task_id: str
    instruction: str
    steps: list[StepRecord] = field(default_factory=list)
    final_evaluation: EvaluationResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "trajectory": [s.to_dict() for s in self.steps],
            "evaluation": self.final_evaluation.to_dict()
            if self.final_evaluation
            else None,
            "metadata": self.metadata,
        }
