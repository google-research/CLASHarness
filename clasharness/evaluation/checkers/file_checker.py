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

"""File checker - verifies virtual workspace file contents."""

from __future__ import annotations
import posixpath
from typing import Any

from clasharness.evaluation.results import CheckResult
from clasharness.evaluation.evaluator import Evaluator


def _fold(text: str) -> str:
    """Lowercase and collapse every whitespace run to a single space.

    Makes ``contains`` forgiving about indentation, newlines, repeated spaces and
    capitalization, so a substring check on generated file content is not defeated
    by cosmetic formatting differences.
    """
    return " ".join(text.split()).lower()


class FileChecker(Evaluator):
    """Checks that a virtual workspace file matches expectations."""

    def __init__(
        self,
        path: str,
        condition: str,
        expected: Any,
        name: str = "",
        weight: float = 1.0,
    ) -> None:
        if not path.startswith("/"):
            path = "/" + path
        self.path = posixpath.normpath(path)
        self.condition = condition  # "exists", "contains", "equals", "not_exists"
        self.expected = expected
        self.name = name or f"file:{self.path} {condition}"
        self.weight = weight

    def evaluate(
        self, env_state: dict[str, Any], task_data: dict[str, Any]
    ) -> CheckResult:
        files = env_state.get("files", {})
        actual = files.get(self.path)
        score: float | None = None
        # Judge-infra status for the semantic branch; deterministic branches
        # leave these at their "ran fine" defaults.
        ok_flag = True
        err = ""
        # False only when a semantic check fails closed WITHOUT reaching the
        # judge (missing file / empty criterion). "Never asked" and "asked, and
        # the content was wrong" are different verdicts that must not be pooled,
        # and only a structured flag can separate them downstream -- the message
        # is prose and was already being string-matched by the report layer.
        judge_invoked = True

        if self.condition == "exists":
            passed = actual is not None
            msg = f"file {self.path} exists: {passed}"
        elif self.condition == "not_exists":
            passed = actual is None
            msg = f"file {self.path} not exists: {passed}"
        elif self.condition == "field_semantic_match":
            # File-scoped semantic grading: judge whether the file's content
            # satisfies ``criterion`` (an ambiguous / paraphrasable assertion such
            # as a verdict or a prose finding that a verbatim ``contains`` would
            # brittly reject). Mirrors EffectChecker's message ``field_semantic_match``.
            criterion = ""
            model = None
            if isinstance(self.expected, dict):
                criterion = str(self.expected.get("criterion", "") or "")
                model = self.expected.get("model")
            else:
                criterion = str(self.expected or "")
            if actual is None:
                # A positive semantic assertion cannot be satisfied by a missing
                # file. Fail closed WITHOUT calling the judge (saves a model call
                # and avoids a spurious pass hallucinated from placeholder text).
                passed = False
                judge_invoked = False
                msg = f"LLM semantic match: no file {self.path} to evaluate"
            elif not criterion:
                passed = False
                judge_invoked = False
                msg = f"LLM semantic match: empty criterion for {self.path}"
            else:
                from clasharness.evaluation.llm_judge import (
                    judge_majority as _llm_judge,
                )

                result = _llm_judge(str(actual), criterion, model=model)
                passed = result.passed
                score = result.score
                ok_flag = result.ok
                err = result.error
                msg = f"LLM semantic match: {result.reason or result.error}"
        elif self.condition == "equals":
            if actual is None:
                passed = False
            elif isinstance(actual, str) and isinstance(self.expected, str):
                # Fold cosmetic formatting (case + whitespace runs) so an
                # exact-content check is not defeated by indentation, repeated
                # spaces, or a trailing newline in generated file content.
                passed = _fold(actual) == _fold(self.expected)
            else:
                passed = actual == self.expected
            msg = f"file {self.path}: expected={self.expected}, actual={actual[:50] if actual else None}"
        elif self.condition == "contains":
            if actual is None:
                passed = False
            elif isinstance(self.expected, str) and isinstance(actual, str):
                passed = _fold(self.expected) in _fold(actual)
            else:
                passed = self.expected in actual
            msg = f"file {self.path} contains '{self.expected}': {passed}"
        else:
            passed = False
            msg = f"Unknown condition: {self.condition}"

        return CheckResult(
            name=self.name,
            passed=passed,
            score=score if score is not None else (1.0 if passed else 0.0),
            weight=self.weight,
            message=msg,
            ok=ok_flag,
            error=err,
            judge_invoked=judge_invoked,
        )

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> FileChecker:
        condition = spec.get("condition")
        expected = spec.get("expected")
        if condition == "field_semantic_match":
            # Criterion may be supplied top-level or nested under ``expected``.
            criterion = spec.get("criterion")
            model = spec.get("model")
            if isinstance(expected, dict):
                criterion = criterion or expected.get("criterion")
                model = model or expected.get("model")
            expected = {"criterion": str(criterion or ""), "model": model}
        elif not condition:
            if "contains" in spec:
                condition = "contains"
                expected = spec["contains"]
            elif "equals" in spec:
                condition = "equals"
                expected = spec["equals"]
            else:
                condition = "exists"
        return cls(
            path=spec.get("path", ""),
            condition=condition,
            expected=expected,
            name=spec.get("name", ""),
            weight=spec.get("weight", 1.0),
        )
