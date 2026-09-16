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

"""Trajectory recording for training data export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from clasharness.core.observation import (
    EvaluationResult,
    Observation,
    StepRecord,
    Trajectory,
)


class TrajectoryRecorder:
    """Records agent interactions for training data export."""

    def __init__(self, task_id: str, instruction: str):
        self._trajectory = Trajectory(task_id=task_id, instruction=instruction)

    def record_step(
        self,
        observation: Observation,
        action: str,
        reward: float,
        done: bool,
        info: dict[str, Any] | None = None,
    ) -> None:
        step = StepRecord(
            step_number=len(self._trajectory.steps),
            observation=observation.to_dict(),
            action=action,
            reward=reward,
            done=done,
            info=info or {},
        )
        self._trajectory.steps.append(step)

    def set_evaluation(self, result: EvaluationResult) -> None:
        self._trajectory.final_evaluation = result

    def set_metadata(self, metadata: dict[str, Any]) -> None:
        self._trajectory.metadata.update(metadata)

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    def save(self, output_dir: Path) -> Path:
        """Save trajectory to JSON file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{self._trajectory.task_id}.json"
        with open(path, "w") as f:
            json.dump(self._trajectory.to_dict(), f, indent=2, ensure_ascii=False)
        return path
