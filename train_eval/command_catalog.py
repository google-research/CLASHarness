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

"""Command catalog / spec helpers extracted from examples/train_and_eval.py."""
from __future__ import annotations

from dataclasses import dataclass

from clasharness.core.task import Task
from clasharness.tools.agent_tool_spec import (
    cli_verbs_by_family,
    iter_command_specs,
    required_flags_by_tokens,
    required_positionals_by_tokens,
)

from train_eval._constants import _DOMAIN_HINT_GROUPS, _PLACEHOLDER_VALUES


# ---------------------------------------------------------------------------
# Available CLI prefixes and command catalog
# ---------------------------------------------------------------------------
_CLI_VERBS: dict[str, list[str]] = cli_verbs_by_family()

@dataclass(frozen=True)
class CommandSpec:
  tokens: tuple[str, ...]
  syntax: str
  domains: tuple[str, ...]
  note: str = ""

def _command_specs_from_tool_spec() -> tuple[CommandSpec, ...]:
  return tuple(
      CommandSpec(
          tokens=spec.tokens,
          syntax=spec.syntax,
          domains=spec.domains,
          note=spec.note,
      )
      for spec in iter_command_specs(safe_for_gt=True)
  )

_COMMAND_SPECS: tuple[CommandSpec, ...] = _command_specs_from_tool_spec() + (
    CommandSpec(
        tokens=("wait",),
        syntax="wait <seconds>",
        domains=("composite", "conflict"),
        note="Progress virtual time by specified seconds to wait for background events or subagents.",
    ),
)

_REQUIRED_FLAGS_BY_PREFIX: dict[tuple[str, ...], tuple[str, ...]] = (
    required_flags_by_tokens()
)

_REQUIRED_POSITIONAL_ARGS_BY_PREFIX: dict[tuple[str, ...], int] = {
    **required_positionals_by_tokens(),
    ("wait",): 1,
}

def _strip_placeholder_token(token: str) -> str:
  return token.strip().strip("`'\"[](){}.,:;")

def _looks_like_catalog_placeholder(token: str) -> bool:
  stripped = token.strip()
  return stripped.startswith("[") and stripped.endswith("]")

def _contains_placeholder_value(tokens: tuple[str, ...]) -> str | None:
  for token in tokens:
    normalized = _strip_placeholder_token(token)
    if normalized in _PLACEHOLDER_VALUES:
      return normalized
  return None

def _spec_key_from_command(command: str) -> tuple[str, ...]:
  tokens = command.split()
  if not tokens:
    return tuple()
  if tokens[0] == "openclaw" and len(tokens) >= 3:
    return tuple(tokens[:3])
  if len(tokens) >= 2:
    return tuple(tokens[:2])
  return tuple(tokens[:1])

def _command_specs_for_task(task: Task | None) -> list[CommandSpec]:
  if task is None:
    return list(_COMMAND_SPECS)

  wanted = set(getattr(task, "domains", []) or [])
  wanted.discard("composite")
  for domain in tuple(wanted):
    wanted.update(_DOMAIN_HINT_GROUPS.get(domain, ()))

  specs: list[CommandSpec] = []
  seen: set[tuple[str, ...]] = set()

  if wanted:
    for spec in _COMMAND_SPECS:
      if wanted.intersection(spec.domains):
        specs.append(spec)
        seen.add(spec.tokens)

  ground_truth = getattr(task, "ground_truth", None)
  if ground_truth and getattr(ground_truth, "solution_commands", None):
    wanted_keys = {
        _spec_key_from_command(cmd)
        for cmd in ground_truth.solution_commands
        if cmd.strip()
    }
    for spec in _COMMAND_SPECS:
      if spec.tokens in wanted_keys and spec.tokens not in seen:
        specs.append(spec)
        seen.add(spec.tokens)

  return specs or list(_COMMAND_SPECS)

def _command_catalog_text(task: Task | None) -> str:
  lines = []
  for spec in _command_specs_for_task(task):
    line = f"  {spec.syntax}"
    if spec.note:
      line += f"  # {spec.note}"
    lines.append(line)
  return "\n".join(lines)

