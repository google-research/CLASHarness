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

"""Base interface for CLASHarness PRO modular conflict plugins."""

from __future__ import annotations

from clasharness.scenario.conflict_vlm_spec import ConflictVLMSeed
from clasharness.scenario.plugins.spec import ConflictPluginSpec


class BaseConflictPlugin:
  """Base class for modular conflict plugins in CLASHarness PRO."""

  #: Declarative spec (the single source of truth for routing metadata,
  #: conformance, and docs). Concrete plugins should attach one; it is optional
  #: during the migration window and enforced by ``validate_plugins`` once every
  #: plugin carries a spec.
  SPEC: ConflictPluginSpec | None = None

  @property
  def name(self) -> str:
    """The unique name/identifier of this conflict plugin.

    Defaults to ``SPEC.name`` when a spec is attached; subclasses may still
    override this property directly (kept for backward compatibility).
    """
    if self.SPEC is not None:
      return self.SPEC.name
    raise NotImplementedError

  def mechanic_available(self, seed: ConflictVLMSeed) -> bool:
    """Whether this plugin's mechanized skill family is reachable for ``seed``.

    Taxonomy routing and tool availability are independent: a seed can match a
    plugin's taxonomy slug while its ``allowed_tool_families`` withholds the
    family the plugin mechanizes. ``_mechanic_command_sets`` already intersects
    with the seed's families, so the *gates* are correctly inert in that case --
    but the trajectory/evals hooks are collected by calling the methods and
    testing truthiness, with no family filter. Without this guard a mechanized
    plugin would instruct the generator to drive commands the seed does not
    expose.

    Plugins that declare no ``skill_families`` are purely narrative and are
    always available.
    """
    declared = set(getattr(self.SPEC, "skill_families", ()) or ())
    if not declared:
      return True
    return bool(declared & set(seed.allowed_tool_families or ()))

  def get_layer0_instruction(self, seed: ConflictVLMSeed) -> str:
    """Returns the phase-level instruction hook for this conflict."""
    return ""

  def get_layer1_instruction(self, seed: ConflictVLMSeed) -> str:
    """Returns the agent/persona instruction hook for this conflict."""
    return ""

  def get_layer1b_instruction(self, seed: ConflictVLMSeed) -> str:
    """Returns the background workspace/session instruction hook."""
    return ""

  def get_trajectory_instruction(self, seed: ConflictVLMSeed) -> str:
    """Returns the expert trajectory instruction hook."""
    return ""

  def get_evals_instruction(self, seed: ConflictVLMSeed) -> str:
    """Returns the automated evals check instruction hook."""
    return ""
