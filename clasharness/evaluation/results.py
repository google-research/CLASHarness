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

"""Evaluation result types.

Canonical home for the dataclasses produced by the evaluation subsystem.
Historically these lived in ``clasharness.core.observation``; that module now
re-exports them from here for back-compat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationResult:
    """Result of evaluating a task."""

    success: bool
    score: float  # 0.0 - 1.0
    details: list[CheckResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "score": self.score,
            "details": [d.to_dict() for d in self.details],
            "metadata": self.metadata,
        }


@dataclass
class CheckResult:
    """Result of a single evaluation check.

    ``role`` decouples binary success from the partial-credit score:
      * ``required``   - must pass for ``success``; also contributes to score.
      * ``scored``     - contributes to score only; does NOT gate success.
      * ``diagnostic`` - recorded for observability; affects neither.

    Defaulting to ``required`` preserves the historical ``success = all(passed)``
    semantics for every check set that does not opt into a role.
    """

    name: str
    passed: bool
    score: float = 1.0
    weight: float = 1.0
    message: str = ""
    role: str = "required"
    # ``ok`` distinguishes a check that ran to a genuine verdict from one whose
    # grader could not run at all (e.g. an LLM judge with no provider configured
    # or a failed provider call). A semantic check that returns ``ok=False`` is a
    # (loud) infrastructure failure, NOT evidence the content was wrong -- it is
    # still recorded as failed/0 for gating, but callers can separate the two.
    ok: bool = True
    error: str = ""
    # ``judge_invoked`` separates the two very different ways a semantic check
    # can fail. False means the grader was deliberately never called because
    # there was nothing to grade (e.g. a ``field_semantic_match`` whose effect
    # transcript is empty: the solver sent no message at all). That is a correct
    # fail-closed verdict, but it says "produced nothing", not "produced the
    # wrong thing" -- and the two are not comparable across models. Measured on
    # the conflict-VLM set the split is enormous: 39.4% of 3.5-flash's semantic
    # checks never reached the judge versus 1.4% for terra, so collapsing them
    # into one "failed semantic check" bucket misreads a non-acting model as a
    # semantically wrong one. Deterministic (non-judge) checks leave this True.
    judge_invoked: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "score": self.score,
            "weight": self.weight,
            "message": self.message,
            "role": self.role,
            "ok": self.ok,
            "error": self.error,
            "judge_invoked": self.judge_invoked,
        }
