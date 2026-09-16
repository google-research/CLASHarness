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

"""Unified MCP/skill specification — the single source of truth.

Every skill declares a :class:`SkillSpec` describing its command surface,
category, capabilities (mock/real), and effects. This one declaration drives:

* command-prefix routing / registration,
* the generated per-skill MCP tool surface (``SkillSpec.to_mcp_tools``); the
  curated ``tools/list`` + dispatch facade (``MCP_TOOLS``) is built in
  ``backend/mcp_facade.py``,
* generated documentation,
* the conformance and dual-path (mock+real) test suites.

Adding a new MCP/skill is therefore: subclass :class:`~clasharness.skills.base.Skill`
(or ``StatefulProtocolSkill``), attach a ``SPEC``, and register it. Everything
else is derived automatically. See ``docs/mcp-extension-guide.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Known skill categories (advisory grouping used in docs / MCP listing). Kept as
# plain strings so external extensions may introduce their own if needed.
KNOWN_CATEGORIES: frozenset[str] = frozenset(
    {
        "filesystem",   # file, asset
        "external",     # calendar, email, weather, tasks (real = network API)
        "http",         # curl / http
        "vcs",          # git
        "compute",      # jq, grep, shell (deterministic local computation)
        "comms",        # message, channel (multi-agent communication substrate)
        "coordination", # lock, deploy (conflict protocols under test)
        "agent",        # agent (sub-agent management)
        "system",       # system config / gateway / plugins
    }
)


class RealKind(str, Enum):
    """How a skill's *real* (non-mock) path is realized.

    This classifies what ``supports_real=True`` means for a skill so the
    dual-path tests and docs can treat each skill correctly.
    """

    #: Deterministic local computation; the real path == the mock path (there is
    #: no meaningful external side to reach). e.g. jq, grep, file, asset.
    PURE_COMPUTE = "pure_compute"
    #: Real path persists to the local filesystem / sqlite; mock is in-memory.
    #: e.g. lock, deploy, message, channel, system.
    LOCAL_PERSISTENT = "local_persistent"
    #: Real path calls an external network service / API; mock returns canned
    #: data. Hermetic tests cover the mock path plus the real path's pure helpers
    #: (query building / response parsing); the live real path is exercised by an
    #: opt-in network suite. e.g. calendar, email, weather, tasks, curl.
    EXTERNAL_NETWORK = "external_network"
    #: Real path shells out to a local binary. e.g. git, shell.
    SUBPROCESS = "subprocess"
    #: Real path is deterministic by design (real == mock) because a genuine
    #: implementation is intentionally out of scope. e.g. agent (no live LLM).
    DETERMINISTIC = "deterministic"


def _tool_name(command_name: str) -> str:
    """Convert a CLI command name (``"lock acquire"``) to an MCP tool name
    (``"lock_acquire"``)."""
    return re.sub(r"\s+", "_", command_name.strip())


@dataclass(frozen=True)
class EffectPrecondition:
    """A machine-readable precondition for an effect-emitting command.

    Declares what must already be true in the protocol state — established by
    *earlier commands in the same trajectory*, since every conflict round starts
    with empty protocol state — for this command to succeed and emit its effect.
    This is the generation-time analogue of a skill's runtime ``_cmd_*`` guards
    (e.g. ``booking confirm`` fails unless a live ``booking hold`` exists for the
    same resource): the *same* contract, made machine-readable so the generation
    pipeline can author, validate, and deterministically repair trajectories
    instead of relying on LLM prose alone.

    Two shapes cover every shipped multi-step protocol:

    * an **ordered same-identifier chain** — :attr:`chain` lists prerequisite
      command names (full names, e.g. ``"booking hold"``) that must appear, in
      order, before this command, each sharing the same value for
      :attr:`identifier_arg` (e.g. ``"--resource"``);
    * an optional **all-ready quantifier** — :attr:`all_ready` =
      ``(command_name, arg, value)`` requires that every occurrence of
      ``command_name`` before this command set ``arg`` to ``value`` (e.g. every
      ``deploy gate-set`` is ``--state ready``), with at least one occurrence.

    A command with no runtime precondition (a chain head such as ``booking hold``
    or ``lock acquire``) simply declares no ``requires``.
    """

    chain: tuple[str, ...] = ()
    """Ordered prerequisite command names (same family) that must precede this."""
    identifier_arg: str = ""
    """Arg whose value must match across ``chain`` and the owning command."""
    all_ready: tuple[str, str, str] | None = None
    """Optional ``(command_name, arg, value)`` all-instances-ready quantifier."""

    def validate(self) -> None:
        """Raise ``ValueError`` if the precondition is internally malformed."""
        if not self.chain and self.all_ready is None:
            raise ValueError("EffectPrecondition must declare chain or all_ready")
        if self.chain and not self.identifier_arg:
            raise ValueError(
                "EffectPrecondition with a chain must declare identifier_arg"
            )
        if self.all_ready is not None and len(self.all_ready) != 3:
            raise ValueError(
                "EffectPrecondition.all_ready must be (command_name, arg, value)"
            )


@dataclass(frozen=True)
class CommandSpec:
    """One command exposed by a skill (a verb under the skill's prefix)."""

    name: str
    """Full command name including the prefix, e.g. ``"lock acquire"``."""
    summary: str = ""
    args: dict[str, dict[str, Any]] = field(default_factory=dict)
    """JSON-Schema ``properties`` for the command arguments."""
    required: tuple[str, ...] = ()
    emits_effects: tuple[str, ...] = ()
    """Effect keys this command may emit (subset of the env effect dict)."""
    requires: tuple[EffectPrecondition, ...] = ()
    """Preconditions that must hold for this command to succeed (see
    :class:`EffectPrecondition`). Empty for chain-head / unconditional commands.
    Purely declarative generation-time metadata; it does not affect the MCP tool
    schema (:meth:`to_mcp_tool` ignores it)."""

    def input_schema(self) -> dict[str, Any]:
        """Return a JSON-Schema object for this command's arguments."""
        schema: dict[str, Any] = {"type": "object", "properties": dict(self.args)}
        if self.required:
            schema["required"] = list(self.required)
        return schema

    def to_mcp_tool(self, *, description_prefix: str = "") -> dict[str, Any]:
        desc = self.summary or self.name
        if description_prefix:
            desc = f"{description_prefix}: {desc}"
        return {
            "name": _tool_name(self.name),
            "description": desc,
            "inputSchema": self.input_schema(),
        }


@dataclass(frozen=True)
class SkillSpec:
    """Declarative description of a skill / MCP tool group."""

    name: str
    category: str
    summary: str
    prefixes: tuple[str, ...]
    commands: tuple[CommandSpec, ...] = ()
    supports_mock: bool = True
    supports_real: bool = True
    real_kind: RealKind = RealKind.PURE_COMPUTE
    smoke: tuple[str, ...] = ()
    """Representative commands (run in order) for the dual-path smoke test.

    Each must be executable from a freshly-initialized skill and return
    ``exit_code == 0`` in both mock and real modes.
    """

    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        """Raise ``ValueError`` if the spec is internally inconsistent."""
        if not self.name:
            raise ValueError("SkillSpec.name is required")
        if not self.prefixes:
            raise ValueError(f"SkillSpec '{self.name}' must declare prefixes")
        if not (self.supports_mock or self.supports_real):
            raise ValueError(
                f"SkillSpec '{self.name}' must support at least one of mock/real"
            )
        if not isinstance(self.real_kind, RealKind):
            raise ValueError(f"SkillSpec '{self.name}' has invalid real_kind")
        # Every command name must begin with one of the declared prefixes.
        cmd_names = {cmd.name for cmd in self.commands}
        for cmd in self.commands:
            if not any(
                cmd.name == p or cmd.name.startswith(p + " ") for p in self.prefixes
            ):
                raise ValueError(
                    f"SkillSpec '{self.name}': command '{cmd.name}' does not "
                    f"start with any declared prefix {self.prefixes}"
                )
            # Effect preconditions must reference real commands/args in this skill.
            for pre in cmd.requires:
                pre.validate()
                for prereq in pre.chain:
                    prereq_spec = self.command(prereq)
                    if prereq_spec is None:
                        raise ValueError(
                            f"SkillSpec '{self.name}': command '{cmd.name}' requires "
                            f"unknown prerequisite command '{prereq}'"
                        )
                    if pre.identifier_arg and pre.identifier_arg not in prereq_spec.args:
                        raise ValueError(
                            f"SkillSpec '{self.name}': prerequisite '{prereq}' of "
                            f"'{cmd.name}' has no arg '{pre.identifier_arg}'"
                        )
                if pre.identifier_arg and pre.identifier_arg not in cmd.args:
                    raise ValueError(
                        f"SkillSpec '{self.name}': command '{cmd.name}' has no "
                        f"identifier arg '{pre.identifier_arg}'"
                    )
                if pre.all_ready is not None:
                    ready_cmd, ready_arg, _ = pre.all_ready
                    if ready_cmd not in cmd_names:
                        raise ValueError(
                            f"SkillSpec '{self.name}': command '{cmd.name}' "
                            f"all_ready references unknown command '{ready_cmd}'"
                        )
                    target = self.command(ready_cmd)
                    if target is not None and ready_arg not in target.args:
                        raise ValueError(
                            f"SkillSpec '{self.name}': all_ready command "
                            f"'{ready_cmd}' has no arg '{ready_arg}'"
                        )

    def command(self, name: str) -> CommandSpec | None:
        for cmd in self.commands:
            if cmd.name == name:
                return cmd
        return None

    def to_mcp_tools(self) -> list[dict[str, Any]]:
        """Generate MCP tool entries for every command in this skill."""
        return [c.to_mcp_tool(description_prefix=self.name) for c in self.commands]


def build_mcp_tools(specs: list[SkillSpec]) -> list[dict[str, Any]]:
    """Flatten many :class:`SkillSpec` into a single MCP ``tools`` list."""
    tools: list[dict[str, Any]] = []
    for spec in specs:
        tools.extend(spec.to_mcp_tools())
    return tools
