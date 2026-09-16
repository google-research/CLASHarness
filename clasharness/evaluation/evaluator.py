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

"""Composable evaluation framework for task verification."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from clasharness.evaluation.results import CheckResult, EvaluationResult

# Allowed check roles. See ``CheckResult`` for semantics.
VALID_ROLES = ("required", "scored", "diagnostic")


class Evaluator(ABC):
    """Base evaluator interface."""

    @abstractmethod
    def evaluate(
        self, env_state: dict[str, Any], task_data: dict[str, Any]
    ) -> CheckResult:
        """Evaluate the current state against task expectations."""

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> "Evaluator":
        """Build a checker instance from a check-spec dict.

        Concrete checkers override this; declared here so ``build_evaluator``
        type-checks against the base interface.
        """
        raise NotImplementedError


class EvaluatorComb:
    """Combines multiple evaluators into a single evaluation.

    ``roles`` (parallel to ``evaluators``) decouples binary ``success`` from the
    continuous ``score``:
      * ``required``   gates ``success`` AND contributes to ``score``.
      * ``scored``     contributes to ``score`` only (partial credit).
      * ``diagnostic`` is recorded but affects neither ``success`` nor ``score``.

    When ``roles`` is omitted every check defaults to ``required``, which
    reproduces the historical ``success = all(passed)`` behaviour exactly.
    """

    def __init__(
        self,
        evaluators: list[Evaluator],
        roles: list[str] | None = None,
    ) -> None:
        self.evaluators = evaluators
        if roles is None:
            roles = ["required"] * len(evaluators)
        # Defensive: pad/truncate so ``roles`` always aligns with ``evaluators``.
        if len(roles) < len(evaluators):
            roles = roles + ["required"] * (len(evaluators) - len(roles))
        self.roles = roles[: len(evaluators)]

    def __call__(
        self, env_state: dict[str, Any], task_data: dict[str, Any]
    ) -> EvaluationResult:
        results: list[CheckResult] = []
        for index, (evaluator, role) in enumerate(zip(self.evaluators, self.roles)):
            # One misbehaving checker must not destroy the episode. ``evaluate``
            # is called from ``Environment.step`` (per-step dense evaluation as
            # well as at done), so an escaping exception aborts the ROLLOUT --
            # the agent's work is lost and the failure is misattributed to the
            # model. Contain it here: record the check as a loud infrastructure
            # failure (``ok=False``) and let the remaining checks run.
            #
            # Exactly one result is appended per evaluator, including on crash:
            # callers index ``details[i]`` against ``checks[i]`` (see
            # ``conflict_vlm_materializer``), so dropping a result would silently
            # misalign every check after it.
            try:
                result = evaluator.evaluate(env_state, task_data)
                result.role = role
            except Exception as exc:  # noqa: BLE001 - surfaced as ok=False
                name = (
                    getattr(evaluator, "name", "")
                    or f"{type(evaluator).__name__}[{index}]"
                )
                detail = f"{type(exc).__name__}: {exc}"
                result = CheckResult(
                    name=name,
                    passed=False,
                    score=0.0,
                    message=f"checker crashed: {detail}",
                    role=role,
                    ok=False,
                    error=detail,
                )
            results.append(result)

        # Diagnostic checks are observed only; scoring is over required+scored.
        scoring = [r for r in results if r.role != "diagnostic"]
        required = [r for r in results if r.role == "required"]
        total_weight = sum(r.weight for r in scoring)

        # A task with no effective (non-diagnostic, non-zero-weight) checks cannot
        # certify success. Returning ``all([]) == True`` here would let empty /
        # fully-pruned / diagnostic-only check sets pass vacuously (score 0.0 but
        # success True), which silently inflates pass rates and hides
        # non-discriminating rounds. Treat it as a hard failure with a flag.
        if not scoring or total_weight == 0:
            reason = "no checks defined"
            if results and not scoring:
                reason = "only diagnostic checks present"
            elif scoring:
                reason = "all scoring checks have zero weight"
            return EvaluationResult(
                success=False,
                score=0.0,
                details=results,
                metadata={"no_effective_checks": True, "reason": reason},
            )

        score = sum(r.score * r.weight for r in scoring) / total_weight

        # Success requires at least one gating (``required``) check and all of
        # them passing. A check set made entirely of ``scored`` checks cannot
        # certify success on its own (avoids vacuous ``all([]) == True``).
        if not required:
            return EvaluationResult(
                success=False,
                score=score,
                details=results,
                metadata={
                    "no_required_checks": True,
                    "reason": "no required (gating) checks defined",
                },
            )

        return EvaluationResult(
            success=all(r.passed for r in required),
            score=score,
            details=results,
            metadata={},
        )


def build_evaluator(checks: list[dict[str, Any]]) -> EvaluatorComb:
    """Build an EvaluatorComb from a list of check specifications.

    Each check dict has a "type" key and type-specific parameters.
    """
    from clasharness.evaluation.checkers.config_checker import ConfigChecker
    from clasharness.evaluation.checkers.effect_checker import EffectChecker
    from clasharness.evaluation.checkers.output_checker import OutputChecker
    from clasharness.evaluation.checkers.state_checker import StateChecker
    from clasharness.evaluation.checkers.llm_checker import LLMChecker
    from clasharness.evaluation.checkers.file_checker import FileChecker
    from clasharness.evaluation.checkers.command_checker import CommandChecker

    type_map: dict[str, type[Evaluator]] = {
        "state": StateChecker,
        "output": OutputChecker,
        "config": ConfigChecker,
        "effect": EffectChecker,
        "llm": LLMChecker,
        "file_check": FileChecker,
        "file": FileChecker,
        "command_check": CommandChecker,
    }

    evaluators: list[Evaluator] = []
    roles: list[str] = []
    for check in checks:
        check_type = check.get("type", "")
        cls = type_map.get(check_type)
        if cls is None:
            raise ValueError(f"Unknown check type: {check_type}")
        role = check.get("role", "required")
        if role not in VALID_ROLES:
            raise ValueError(
                f"Unknown check role: {role!r} (expected one of {VALID_ROLES})"
            )
        evaluators.append(cls.from_spec(check))
        roles.append(role)

    return EvaluatorComb(evaluators, roles)
