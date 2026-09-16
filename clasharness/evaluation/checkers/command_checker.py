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

"""Command checker - verifies command execution history.

Matching is *structural*, not literal-substring. The previous implementation
required the expected command to appear verbatim inside an executed action,
which made grading extremely brittle: a check for

    agent delegate --agent Ops_Agent --instruction "system gateway restart"

would fail if the agent quoted the name differently, reordered flags, or added
extra words to the instruction. In practice this caused ~76% of command checks
to fail even when the agent did the right thing.

The checker now parses both the expected command and each executed action into
``(verb_prefix, flags, positionals)`` and matches when:

  * the verb prefix is identical (so ``agent delegate`` never matches a bare
    ``system gateway restart`` - delegation is a distinct behaviour), AND
  * every flag named in the expected command is present in the action with a
    value that matches (normalized substring or token subsequence), AND
  * every positional in the expected command appears (in order) among the
    action positionals.

Extra flags/words in the action are ignored. A literal-substring match is kept
as a fast path so previously-passing exact checks keep passing.
"""

from __future__ import annotations

import shlex
from typing import Any

from clasharness.evaluation.results import CheckResult
from clasharness.evaluation.evaluator import Evaluator


def _safe_split(text: str) -> list[str]:
  try:
    return shlex.split(text)
  except ValueError:
    return text.split()


def _norm(value: str) -> str:
  """Collapse a token to alphanumerics for forgiving id/value comparison.

  ``"#Incident-Prod_504"`` and ``"incident prod 504"`` both normalize to
  ``"incidentprod504"``.
  """
  return "".join(ch for ch in value.lower() if ch.isalnum())


def _word_tokens(value: str) -> list[str]:
  return [_norm(tok) for tok in value.replace("/", " ").split() if _norm(tok)]


def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
  it = iter(haystack)
  return all(tok in it for tok in needle)


def _verb_prefix(tokens: list[str]) -> tuple[str, ...]:
  """Return the command's leading verb tokens (the family + subcommand)."""
  if not tokens:
    return ()
  if tokens[0] == "openclaw" and len(tokens) >= 3:
    return tuple(tokens[:3])
  if len(tokens) >= 2 and not tokens[1].startswith("-"):
    return tuple(tokens[:2])
  return (tokens[0],)


def _parse(tokens: list[str]) -> tuple[tuple[str, ...], dict[str, list[str]], list[str]]:
  """Split tokens into (verb_prefix, flags, positionals)."""
  verb = _verb_prefix(tokens)
  rest = tokens[len(verb):]
  flags: dict[str, list[str]] = {}
  positionals: list[str] = []
  i = 0
  while i < len(rest):
    tok = rest[i]
    if tok.startswith("--"):
      if "=" in tok:
        key, val = tok[2:].split("=", 1)
        flags.setdefault("--" + key, []).append(val)
      elif i + 1 < len(rest) and not rest[i + 1].startswith("--"):
        flags.setdefault(tok, []).append(rest[i + 1])
        i += 1
      else:
        flags.setdefault(tok, []).append("")
    else:
      positionals.append(tok)
    i += 1
  return verb, flags, positionals


def _value_match(expected: str, actual: str) -> bool:
  """True when ``actual`` satisfies the expected value.

  Forgiving on formatting: normalized substring (handles quoting/separators and
  extra surrounding words) or token subsequence (handles minor reordering).
  """
  if expected == actual:
    return True
  exp_norm, act_norm = _norm(expected), _norm(actual)
  if not exp_norm:
    return True
  if exp_norm in act_norm:
    return True
  return _is_subsequence(_word_tokens(expected), _word_tokens(actual))


# Verbs whose payload carries a *forwarded* command/instruction. A required
# direct command is frequently accomplished by delegating it to a sub-agent, so
# the required command text lives inside one of the payload flags below rather
# than being executed verbatim by the primary agent.
_DELEGATION_VERBS: frozenset[tuple[str, ...]] = frozenset(
    {
        ("agent", "delegate"),
        ("agent", "message"),
        ("agent", "instruct"),
        ("agent", "ask"),
    }
)
_PAYLOAD_FLAGS: tuple[str, ...] = (
    "--instruction",
    "--instructions",
    "--task",
    "--message",
    "--msg",
    "--prompt",
    "--body",
)


def _is_delegation(tokens: list[str]) -> bool:
    """True when ``tokens`` start with a delegation/forwarding verb."""
    verb = _verb_prefix(tokens)
    return tuple(_norm(t) for t in verb) in _DELEGATION_VERBS


def _delegated_payloads(action_tokens: list[str]) -> list[str]:
    """Return command-like payloads embedded in a delegation action.

    ``agent delegate --agent Ops --instruction "system gateway restart"`` carries
    the required command inside ``--instruction``. Returning that payload lets a
    check for the *direct* command match it structurally (not only as a verbatim
    substring of the whole delegate command). Non-delegation actions yield no
    payloads, so ordinary commands are unaffected.
    """
    if not _is_delegation(action_tokens):
        return []
    _verb, flags, _pos = _parse(action_tokens)
    payloads: list[str] = []
    for flag in _PAYLOAD_FLAGS:
        for val in flags.get(flag, []):
            val = val.strip()
            if val:
                payloads.append(val)
    return payloads


def _structural_match(expected_tokens: list[str], action_tokens: list[str]) -> bool:
  if not expected_tokens or not action_tokens:
    return False
  e_verb, e_flags, e_pos = _parse(expected_tokens)
  a_verb, a_flags, a_pos = _parse(action_tokens)

  if tuple(_norm(t) for t in e_verb) != tuple(_norm(t) for t in a_verb):
    return False

  for flag, exp_values in e_flags.items():
    act_values = a_flags.get(flag)
    if act_values is None:
      return False
    for exp_val in exp_values:
      if not any(_value_match(exp_val, act_val) for act_val in act_values):
        return False

  # Positionals must appear in order (forgiving on value formatting).
  cursor = 0
  for exp_p in e_pos:
    matched = False
    while cursor < len(a_pos):
      if _value_match(exp_p, a_pos[cursor]):
        cursor += 1
        matched = True
        break
      cursor += 1
    if not matched:
      return False
  return True


class CommandChecker(Evaluator):
    """Checks that a command was executed during the episode."""

    def __init__(
        self,
        command: str,
        name: str = "",
        weight: float = 1.0,
    ) -> None:
        self.command = command.strip()
        self.name = name or f"command:{self.command}"
        self.weight = weight

    def evaluate(
        self, env_state: dict[str, Any], task_data: dict[str, Any]
    ) -> CheckResult:
        command_history = env_state.get("command_history", [])
        expected_tokens = _safe_split(self.command)
        # A check that expects delegation must still be satisfied by a real
        # delegate action, so only mine delegated payloads when the expected
        # command is itself a *direct* (non-delegation) command.
        expected_is_delegation = _is_delegation(expected_tokens)
        passed = False
        actual_cmds = []
        for cmd_entry in command_history:
            action = (cmd_entry.get("action") or "").strip()
            actual_cmds.append(action)
            # An MCP-tool verb (``run_script`` / ``decompress_archive`` /
            # ``execute_db_query``) is expanded to its runtime executable
            # (``python3`` / ``bash`` / ``unzip`` / ``tar`` / ``sqlite3``) before
            # the environment records ``action``. Grading is compiled against the
            # agent-facing verb, which would never match the expanded executable,
            # so also consider the raw (pre-rewrite) form when present.
            candidates = [action]
            raw_action = (cmd_entry.get("raw_action") or "").strip()
            if raw_action and raw_action != action:
                candidates.append(raw_action)
            # Delegation-aware: a required direct command may have been carried
            # out by delegating it; add any command carried inside a delegate
            # action's instruction/task/message payload as extra candidates.
            if not expected_is_delegation:
                delegated: list[str] = []
                for cand in candidates:
                    delegated.extend(_delegated_payloads(_safe_split(cand)))
                candidates.extend(delegated)
            for cand in candidates:
                # Fast path: literal substring (preserves prior exact behaviour).
                if self.command and self.command in cand:
                    passed = True
                    break
                # Structural path: same verb, expected flags/positionals satisfied.
                if _structural_match(expected_tokens, _safe_split(cand)):
                    passed = True
                    break
            if passed:
                break

        msg = f"command '{self.command}' executed: {passed} (history: {actual_cmds})"
        return CheckResult(
            name=self.name,
            passed=passed,
            score=1.0 if passed else 0.0,
            weight=self.weight,
            message=msg,
        )

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> CommandChecker:
        return cls(
            command=spec.get("command", ""),
            name=spec.get("name", ""),
            weight=spec.get("weight", 1.0),
        )
