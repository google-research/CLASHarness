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

"""Conflict-plugin package: registration, taxonomy routing, and conformance.

Conflict plugins are standardized the same way the skill/MCP surface is: every
plugin carries a declarative :class:`~clasharness.scenario.plugins.spec.ConflictPluginSpec`
and self-registers via :func:`~clasharness.scenario.plugins.registry.register_conflict_plugin`.
Importing this package imports every plugin module (running the decorators),
builds ``_ALL_PLUGINS`` from the registry, and exposes:

* :func:`get_plugins_for_taxonomy` — route seed taxonomy/difficulty tokens to
  the plugins that implement them (via each plugin's declared tokens),
* :func:`validate_plugins` — a conformance checker that tests assert on: unique
  names, no taxonomy-token collisions, valid specs, and the cross-surface check
  that a plugin only claims effects (``emits_effects``) the skill families it
  declares (``skill_families``) can actually emit.

Adding a new conflict mechanic is therefore: subclass ``BaseConflictPlugin``,
attach a ``SPEC``, decorate with ``@register_conflict_plugin``, and import the
module below. See ``docs/conflict-plugin-guide.md``.
"""

from __future__ import annotations

from clasharness.scenario.plugins.base import BaseConflictPlugin
from clasharness.scenario.plugins.spec import ConflictPluginSpec
from clasharness.scenario.plugins.registry import (
    build_registered_plugins,
    register_conflict_plugin,
    registered_plugin_classes,
)

# Importing each plugin module runs its ``@register_conflict_plugin`` decorator.
# The import order below defines the registration order and therefore the
# ``_ALL_PLUGINS`` order (kept identical to the pre-standardization list).
from clasharness.scenario.plugins.staleness import TemporalStalenessPlugin
from clasharness.scenario.plugins.db_audit import DBAuditPlugin
from clasharness.scenario.plugins.dependency_ordering import DependencyOrderingPlugin
from clasharness.scenario.plugins.deadlock import DeadlockDependencyPlugin
from clasharness.scenario.plugins.evidence_mismatch import EvidenceMismatchPlugin
from clasharness.scenario.plugins.time_window import TimeWindowExpirationPlugin
from clasharness.scenario.plugins.distributed_lock import DistributedLockPlugin
from clasharness.scenario.plugins.barrier_synchronization import BarrierSynchronizationPlugin
from clasharness.scenario.plugins.controlled_deployment_gate import ControlledDeploymentGatePlugin
from clasharness.scenario.plugins.reversible_action_hold import ReversibleActionHoldPlugin
from clasharness.scenario.plugins.multi_device_pairing import MultiDevicePairingPlugin
from clasharness.scenario.plugins.scope_reconciliation import ScopeReconciliationPlugin
from clasharness.scenario.plugins.multi_party_synchronization import MultiPartySynchronizationPlugin
from clasharness.scenario.plugins.coordinated_containment import CoordinatedContainmentPlugin
from clasharness.scenario.plugins.non_overlapping_assignment import NonOverlappingAssignmentPlugin
from clasharness.scenario.plugins.duplicate_hold_stale import DuplicateHoldStalePlugin

# Challenge plugins (multimodal / multi-agent difficulty tokens).
from clasharness.scenario.plugins.duplicate_avoidance import DuplicateAvoidancePlugin
from clasharness.scenario.plugins.gap_completion import GapCompletionPlugin
from clasharness.scenario.plugins.information_transfer import InformationTransferPlugin
from clasharness.scenario.plugins.multi_source_reasoning import MultiSourceReasoningPlugin
from clasharness.scenario.plugins.state_repair import StateRepairPlugin
from clasharness.scenario.plugins.workflow_completion import WorkflowCompletionPlugin
from clasharness.scenario.plugins.vision_decoy import VisionDecoyPlugin

# Built from the decorator registry (import order preserved). Kept module-level
# and named ``_ALL_PLUGINS`` for backward compatibility with existing tooling.
_ALL_PLUGINS: list[BaseConflictPlugin] = build_registered_plugins()


def _effective_tokens(plugin: BaseConflictPlugin) -> tuple[str, ...]:
  """Taxonomy tokens a plugin answers to (``SPEC`` tokens, else its ``name``)."""
  spec = plugin.SPEC
  if spec is not None:
    return spec.effective_tokens
  return (plugin.name,)


def _build_token_index(
    plugins: list[BaseConflictPlugin],
) -> dict[str, BaseConflictPlugin]:
  """Map each taxonomy token to the (first) plugin that declares it."""
  index: dict[str, BaseConflictPlugin] = {}
  for plugin in plugins:
    for token in _effective_tokens(plugin):
      index.setdefault(token, plugin)
  return index


_TOKEN_INDEX: dict[str, BaseConflictPlugin] = _build_token_index(_ALL_PLUGINS)


def get_plugins_for_taxonomy(taxonomy: list[str]) -> list[BaseConflictPlugin]:
  """Return the conflict plugins that implement a seed's taxonomy tokens.

  An empty/falsy ``taxonomy`` returns every registered plugin (unchanged
  behavior). Otherwise each token is resolved through the token index; unknown
  tokens are ignored and each matched plugin appears at most once, preserving
  first-seen order.
  """
  if not taxonomy:
    return list(_ALL_PLUGINS)

  matched: list[BaseConflictPlugin] = []
  seen: set[int] = set()
  for token in taxonomy:
    plugin = _TOKEN_INDEX.get(token)
    if plugin is not None and id(plugin) not in seen:
      matched.append(plugin)
      seen.add(id(plugin))
  return matched


def _validate_cross_surface(plugins: list[BaseConflictPlugin]) -> list[str]:
  """Check ``skill_families``/``emits_effects`` against AgentToolSpec.

  A plugin may only declare skill families that exist in AgentToolSpec, and may
  only claim effects (``emits_effects``) that those families can actually emit
  (their commands' ``state_effects``). This is the plugin analogue of the
  skill-side ``CommandSpec.emits_effects`` check and catches a plugin whose
  evals assert an effect no reachable command produces.
  """
  relevant = [
      p
      for p in plugins
      if p.SPEC is not None and (p.SPEC.skill_families or p.SPEC.emits_effects)
  ]
  if not relevant:
    return []

  try:
    from clasharness.tools.agent_tool_spec import (  # pylint: disable=g-import-not-at-top
        iter_command_specs,
        load_agent_tool_spec,
        normalize_family_name,
    )

    tool_spec = load_agent_tool_spec()
  except Exception as exc:  # pylint: disable=broad-except
    return [f"cross-surface validation unavailable (AgentToolSpec): {exc}"]

  known_families = {family.family for family in tool_spec.families}
  errors: list[str] = []
  for plugin in relevant:
    spec = plugin.SPEC
    assert spec is not None  # narrowed by ``relevant`` filter
    norm_families: list[str] = []
    for family in spec.skill_families:
      normalized = normalize_family_name(family)
      if normalized not in known_families:
        errors.append(
            f"{spec.name}: skill_family {family!r} (normalized {normalized!r}) "
            f"is not an AgentToolSpec family"
        )
      else:
        norm_families.append(normalized)

    if spec.emits_effects:
      if not spec.skill_families:
        errors.append(
            f"{spec.name}: declares emits_effects {spec.emits_effects} but no "
            f"skill_families to emit them"
        )
        continue
      available: set[str] = set()
      for command in iter_command_specs(include_families=norm_families):
        available.update(command.state_effects)
      for effect in spec.emits_effects:
        if effect not in available:
          errors.append(
              f"{spec.name}: effect {effect!r} is not emitted by declared "
              f"families {tuple(norm_families)} (available: {sorted(available)})"
          )
  return errors


def validate_plugins(require_spec: bool = False) -> list[str]:
  """Return conformance violations for the registered plugins (empty = OK).

  Checks, in order:

  1. every plugin exposes a non-empty ``name``; names are unique;
  2. each attached ``SPEC`` is internally valid and ``SPEC.name == plugin.name``;
     when ``require_spec`` is True, a missing ``SPEC`` is itself a violation;
  3. no taxonomy token is claimed by two different plugins;
  4. cross-surface: declared ``skill_families``/``emits_effects`` are consistent
     with AgentToolSpec (see :func:`_validate_cross_surface`).

  ``require_spec`` starts False during the backfill window and is flipped to
  True (enforced by tests) once every plugin carries a spec.
  """
  errors: list[str] = []
  plugins = _ALL_PLUGINS

  seen_names: dict[str, str] = {}
  for plugin in plugins:
    cls = type(plugin).__name__
    try:
      name = plugin.name
    except Exception as exc:  # pylint: disable=broad-except
      errors.append(f"{cls}: .name is not resolvable ({exc!r})")
      continue
    if not name:
      errors.append(f"{cls}: empty .name")

    spec = plugin.SPEC
    if spec is None:
      if require_spec:
        errors.append(f"{cls} (name={name!r}): missing SPEC")
    else:
      try:
        spec.validate()
      except ValueError as exc:
        errors.append(f"{cls}: invalid SPEC ({exc})")
      if spec.name != name:
        errors.append(
            f"{cls}: SPEC.name={spec.name!r} != plugin.name={name!r}"
        )

    if name:
      if name in seen_names:
        errors.append(
            f"duplicate plugin name {name!r} ({seen_names[name]} and {cls})"
        )
      else:
        seen_names[name] = cls

  # Taxonomy-token collisions across distinct plugins.
  token_owner: dict[str, str] = {}
  for plugin in plugins:
    cls = type(plugin).__name__
    for token in _effective_tokens(plugin):
      owner = token_owner.get(token)
      if owner is None:
        token_owner[token] = cls
      elif owner != cls:
        errors.append(
            f"taxonomy token {token!r} claimed by both {owner} and {cls}"
        )

  errors.extend(_validate_cross_surface(plugins))
  return errors


__all__ = [
    "BaseConflictPlugin",
    "ConflictPluginSpec",
    "register_conflict_plugin",
    "registered_plugin_classes",
    "get_plugins_for_taxonomy",
    "validate_plugins",
    # Concrete plugins (public for direct import / testing).
    "TemporalStalenessPlugin",
    "DBAuditPlugin",
    "DependencyOrderingPlugin",
    "DeadlockDependencyPlugin",
    "EvidenceMismatchPlugin",
    "TimeWindowExpirationPlugin",
    "DistributedLockPlugin",
    "BarrierSynchronizationPlugin",
    "ControlledDeploymentGatePlugin",
    "ReversibleActionHoldPlugin",
    "MultiDevicePairingPlugin",
    "ScopeReconciliationPlugin",
    "MultiPartySynchronizationPlugin",
    "CoordinatedContainmentPlugin",
    "NonOverlappingAssignmentPlugin",
    "DuplicateHoldStalePlugin",
    "DuplicateAvoidancePlugin",
    "GapCompletionPlugin",
    "InformationTransferPlugin",
    "MultiSourceReasoningPlugin",
    "StateRepairPlugin",
    "WorkflowCompletionPlugin",
    "VisionDecoyPlugin",
]
