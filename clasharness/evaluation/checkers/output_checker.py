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

"""Output checker - verifies command output content."""

from __future__ import annotations

import re
from typing import Any

from clasharness.evaluation.results import CheckResult
from clasharness.evaluation.evaluator import Evaluator


class OutputChecker(Evaluator):
    """Checks that command output matches expected patterns.

    Similar to WebArena's string_match evaluator.
    """

    def __init__(
        self,
        condition: str,
        expected: str | None,
        output_field: str = "last_stdout",
        name: str = "",
        weight: float = 1.0,
        ignore_case: bool = True,
    ) -> None:
        self.condition = condition  # "exact", "contains", "regex", "not_contains"
        self.expected = expected
        self.output_field = output_field
        self.name = name or f"output:{condition}"
        self.weight = weight
        self.ignore_case = ignore_case

    def evaluate(
        self, env_state: dict[str, Any], task_data: dict[str, Any]
    ) -> CheckResult:
        actual_raw = env_state.get(self.output_field, "")
        actual = "" if actual_raw is None else str(actual_raw)
        expected_raw = "" if self.expected is None else str(self.expected)

        if self.condition == "exit_code_zero":
            exit_code = env_state.get("last_exit_code", None)
            passed = exit_code == 0
            msg = f"exit_code_zero check: actual={exit_code}"
        else:
            actual_cmp = actual
            expected_cmp = expected_raw
            if self.ignore_case:
                actual_cmp = actual_cmp.lower()
                expected_cmp = expected_cmp.lower()

            if self.condition == "exact":
                passed = actual_cmp.strip() == expected_cmp.strip()
            elif self.condition == "contains":
                passed = expected_cmp in actual_cmp
            elif self.condition == "not_contains":
                passed = expected_cmp not in actual_cmp
            elif self.condition == "regex":
                flags = re.IGNORECASE if self.ignore_case else 0
                passed = bool(re.search(expected_raw, actual, flags))
            else:
                passed = False

            # ``last_stdout`` only holds the most recent command, so both kinds
            # of assertion have to consult the whole command history -- but with
            # OPPOSITE quantifiers.
            if self.output_field == "last_stdout":
                history = env_state.get("command_history", [])

                if self.condition == "not_contains":
                    # A negative assertion is universally quantified: the
                    # forbidden text must be absent from EVERY observed output.
                    # Folding it into the positive any-match rescue below made
                    # it self-defeating -- one clean earlier command (an ``ls``,
                    # a ``cd``) was enough to "rescue" a run that really did
                    # print the forbidden text, so ``not_contains`` could
                    # essentially never fail. Note this scan must run even when
                    # ``passed`` is already True: the violation may sit in an
                    # earlier command rather than in ``last_stdout``.
                    for hist_item in history:
                        hist_actual = str(hist_item.get("stdout", ""))
                        hist_cmp = (
                            hist_actual.lower() if self.ignore_case else hist_actual
                        )
                        if expected_cmp in hist_cmp:
                            passed = False
                            break

                elif not passed:
                    # Positive assertions are existentially quantified: matching
                    # any single command's output satisfies them.
                    for hist_item in history:
                        hist_actual = str(hist_item.get("stdout", ""))
                        hist_cmp = (
                            hist_actual.lower() if self.ignore_case else hist_actual
                        )
                        if self.condition == "exact":
                            item_passed = hist_cmp.strip() == expected_cmp.strip()
                        elif self.condition == "contains":
                            item_passed = expected_cmp in hist_cmp
                        elif self.condition == "regex":
                            flags = re.IGNORECASE if self.ignore_case else 0
                            item_passed = bool(
                                re.search(expected_raw, hist_actual, flags)
                            )
                        else:
                            item_passed = False
                        if item_passed:
                            passed = True
                            break

            msg = (
                f"{self.condition} check: expected='{expected_raw[:50]}', "
                f"actual='{actual[:50]}'"
            )

        return CheckResult(
            name=self.name,
            passed=passed,
            score=1.0 if passed else 0.0,
            weight=self.weight,
            message=msg,
        )

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> OutputChecker:
        # ``match_type`` is the spelling persisted in the shipped task specs
        # (``clasharness/data/tasks/*/specs.json``); the generators emit
        # ``condition``. The alias used to be missing, so every task that stored
        # ``match_type`` silently fell back to the ``contains`` default and had
        # its pattern matched *literally* -- e.g. the alternation
        # ``(Gateway|gateway|running|stopped)`` was searched for as a raw
        # substring and could never match, failing even the expert solution.
        # 175 shipped checks use this key (159 of them ``regex``).
        condition = spec.get("condition") or spec.get("match_type") or "contains"
        return cls(
            condition=condition,
            expected=spec.get("expected"),
            output_field=spec.get("output_field", "last_stdout"),
            name=spec.get("name", ""),
            weight=spec.get("weight", 1.0),
            ignore_case=spec.get("ignore_case", True),
        )
