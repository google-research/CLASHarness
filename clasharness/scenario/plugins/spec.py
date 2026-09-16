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

"""Unified conflict-plugin specification — the declarative source of truth.

Every conflict plugin declares a :class:`ConflictPluginSpec` describing the
taxonomy tokens it binds to, the skill command families its expert trajectory
exercises, the effect keys its evals assert, and which prompt hooks it
implements. This mirrors ``clasharness/skills/spec.py`` (:class:`SkillSpec`) so
the plugin surface is standardized the same way the MCP/skill surface is.

The ``SPEC`` is metadata *around* the prompt hooks, not a replacement for them:
the five ``get_*_instruction`` hooks still carry the actual prompt text. The
spec adds structured, machine-checkable declarations that drive:

* taxonomy-token routing / registration (``get_plugins_for_taxonomy``),
* conformance checks (``validate_plugins``), including the cross-surface check
  that a plugin only claims effects the declared skill families can emit,
* generated documentation and the plugin authoring guide.

See ``docs/conflict-plugin-guide.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Known plugin categories (advisory grouping used in docs / listings). Kept as
# plain strings so future conflict classes may introduce their own if needed.
KNOWN_PLUGIN_CATEGORIES: frozenset[str] = frozenset(
    {
        "inconsistency",           # temporal staleness, evidence mismatch, audits
        "order",                   # dependency ordering, controlled gates
        "timeout",                 # time-window / expiring holds
        "coordination",            # locks, barriers, deadlocks (protocol families)
        "subagent_collaboration",  # multi-party synchronization / transfer
        "main_supervised",         # supervisor-approved actions
        "challenge",               # multimodal / multi-agent difficulty challenges
    }
)

# The five prompt-hook slots a plugin may implement. Closed set (enforced).
KNOWN_HOOKS: frozenset[str] = frozenset(
    {"layer0", "layer1", "layer1b", "trajectory", "evals"}
)


@dataclass(frozen=True)
class ConflictPluginSpec:
    """Declarative description of a conflict plugin.

    Attributes:
      name: Canonical taxonomy token and unique plugin id (the primary match
        key). Required.
      summary: One-line description for docs / listings.
      category: Advisory grouping; see :data:`KNOWN_PLUGIN_CATEGORIES`.
      taxonomy_tokens: All seed ``conflict_taxonomy`` / ``difficulty_challenges``
        tokens that route to this plugin. Defaults to ``(name,)``. Declaring
        aliases here prevents the "typo silently matches nothing" failure mode.
      skill_families: Command families (``agent_tools.v1.json``) this plugin's
        expert trajectory exercises, e.g. ``("deploy",)``. Validated against the
        AgentToolSpec family set.
      emits_effects: Effect keys this plugin's evals assert, e.g.
        ``("deployment_cutover",)``. Cross-validated: must be a subset of the
        ``state_effects`` the declared ``skill_families`` can actually emit.
      hooks: Which prompt hooks this plugin implements (subset of
        :data:`KNOWN_HOOKS`). Advisory but validated to be legal hook names.
      reused: True if this plugin predates the seed(s) that use it (a reused
        mechanic rather than a newly-authored one). Metadata only.
    """

    name: str
    summary: str = ""
    category: str = ""
    taxonomy_tokens: tuple[str, ...] = ()
    skill_families: tuple[str, ...] = ()
    emits_effects: tuple[str, ...] = ()
    hooks: tuple[str, ...] = ()
    reused: bool = False

    @property
    def effective_tokens(self) -> tuple[str, ...]:
        """Tokens that route to this plugin (``taxonomy_tokens`` or ``(name,)``)."""
        return self.taxonomy_tokens if self.taxonomy_tokens else (self.name,)

    def validate(self) -> None:
        """Raise ``ValueError`` if the spec is internally inconsistent."""
        if not self.name:
            raise ValueError("ConflictPluginSpec.name is required")
        for token in self.taxonomy_tokens:
            if not token or not isinstance(token, str):
                raise ValueError(
                    f"ConflictPluginSpec '{self.name}': taxonomy_tokens must be "
                    f"non-empty strings, got {self.taxonomy_tokens!r}"
                )
        for hook in self.hooks:
            if hook not in KNOWN_HOOKS:
                raise ValueError(
                    f"ConflictPluginSpec '{self.name}': unknown hook '{hook}' "
                    f"(known: {sorted(KNOWN_HOOKS)})"
                )
