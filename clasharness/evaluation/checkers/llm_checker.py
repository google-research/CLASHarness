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

"""LLM-based fuzzy checker — semantic evaluation via Claude."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from clasharness.evaluation.results import CheckResult
from clasharness.evaluation.evaluator import Evaluator


class LLMChecker(Evaluator):
    """Uses Claude to semantically judge whether a task criterion was satisfied.

    Unlike rule-based checkers (EffectChecker, OutputChecker …), LLMChecker
    handles open-ended criteria such as:
    - "The agent's reply was polite and on-topic"
    - "The agent found the correct email and extracted the right date"
    - "The follow-up task title is meaningfully related to the email subject"

    The checker calls the Anthropic API (``anthropic`` package) and returns a
    structured result. If the package is missing, the API call fails, or the
    ANTHROPIC_API_KEY is unset, the check fails gracefully without raising.

    Task spec example::

        {
            "type": "llm",
            "criterion": "The agent found an email about 'proposal' and created a
                          follow-up task whose title relates to the email subject.",
            "name": "email→task semantic check",
            "weight": 1.0,
            "model": "claude-haiku-4-5-20251001"
        }
    """

    DEFAULT_MODEL = "claude-haiku-4-5-20251001"

    def __init__(
        self,
        criterion: str,
        name: str = "",
        weight: float = 1.0,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> None:
        self.criterion = criterion
        self.name = name or f"llm:{criterion[:50]}"
        self.weight = weight
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------ #
    # Evaluator interface                                                   #
    # ------------------------------------------------------------------ #

    def evaluate(
        self, env_state: dict[str, Any], task_data: dict[str, Any]
    ) -> CheckResult:
        try:
            return self._call_llm(env_state, task_data)
        except Exception as exc:
            # An exception escaping _call_llm is an INFRASTRUCTURE failure
            # (evidence assembly blew up, or the judge raised instead of
            # returning ok=False) -- it is not evidence the content was wrong.
            # Two things have to hold for the judge-health report to see it:
            #   * ``ok=False``, or it is counted as a genuine content failure
            #     and silently drags the model's semantic pass rate down;
            #   * the shared "LLM semantic match:" prefix, which is the ONLY
            #     way judge-originated checks are identified. Without it the
            #     check is invisible to the report rather than miscategorised,
            #     so a judge melting down looks like a clean run.
            return CheckResult(
                name=self.name,
                passed=False,
                score=0.0,
                weight=self.weight,
                message=f"LLM semantic match: evaluation error: {exc}",
                ok=False,
                error=str(exc),
            )

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> LLMChecker:
        return cls(
            criterion=spec["criterion"],
            name=spec.get("name", ""),
            weight=spec.get("weight", 1.0),
            model=spec.get("model", cls.DEFAULT_MODEL),
            temperature=spec.get("temperature", 0.0),
            max_tokens=spec.get("max_tokens", 256),
        )

    # ------------------------------------------------------------------ #
    # Internal                                                              #
    # ------------------------------------------------------------------ #

    def _call_llm(
        self, env_state: dict[str, Any], task_data: dict[str, Any]
    ) -> CheckResult:
        from clasharness.evaluation.llm_judge import judge as _llm_judge

        evidence = _build_evidence(env_state, task_data)
        result = _llm_judge(evidence, self.criterion, model=self.model)
        # Share the "LLM semantic match:" prefix with the effect/file semantic
        # checkers so judge-originated checks are identifiable in one place (the
        # judge-health report keys off this prefix).
        message = "LLM semantic match: " + (result.reason or result.error or "")[:200]
        return CheckResult(
            name=self.name,
            passed=result.passed,
            score=result.score,
            weight=self.weight,
            message=message,
            ok=result.ok,
            error=result.error,
        )


# ------------------------------------------------------------------ #
# Evidence construction                                                 #
# ------------------------------------------------------------------ #

def _build_evidence(
    env_state: dict[str, Any],
    task_data: dict[str, Any],
) -> str:
    task_instruction = (
        env_state.get("task_instruction")
        or task_data.get("public", {}).get("instruction", "")
        or "(not provided)"
    )

    # Command history: last 8 steps
    history = env_state.get("command_history", [])[-8:]
    if history:
        history_lines = []
        for i, h in enumerate(history, 1):
            stdout_preview = (h.get("stdout") or "").strip()[:300]
            history_lines.append(
                f"  [{i}] $ {h.get('action', '')}\n"
                f"      exit={h.get('exit_code', '?')}  "
                f"stdout: {stdout_preview or '(empty)'}"
            )
        history_block = "\n".join(history_lines)
    else:
        history_block = "  (no commands recorded)"

    # Effects summary
    effects = env_state.get("effects", {})
    effect_lines = [
        f"  {k}: {len(v)} event(s)"
        for k, v in effects.items()
        if v
    ]
    effects_block = "\n".join(effect_lines) or "  (none)"

    last_stdout = (env_state.get("last_stdout") or "").strip()[:600]
    last_exit = env_state.get("last_exit_code", "?")

    return f"""## Task instruction given to the agent
{task_instruction}

## Agent command history
{history_block}

## Side effects produced during the episode
{effects_block}

## Final command output
Exit code: {last_exit}
{last_stdout or "(empty)"}"""


# ------------------------------------------------------------------ #
# Response parsing (retained for backward compatibility)               #
# ------------------------------------------------------------------ #

def _parse_response(text: str) -> dict[str, Any]:
    """Parse LLM JSON response, with fallback for malformed output."""
    # Try strict parse first
    try:
        data = json.loads(text)
        return _normalise(data)
    except (json.JSONDecodeError, ValueError):
        pass

    # Extract embedded JSON object
    match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return _normalise(data)
        except (json.JSONDecodeError, ValueError):
            pass

    # Last resort: keyword scan
    lower = text.lower()
    passed = "true" in lower and "false" not in lower
    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": text[:200],
    }


def _normalise(data: dict[str, Any]) -> dict[str, Any]:
    """Ensure the parsed dict has the expected fields and types."""
    return {
        "passed": bool(data.get("passed", False)),
        "score": max(0.0, min(1.0, float(data.get("score", 0.0)))),
        "reason": str(data.get("reason", "")),
    }
