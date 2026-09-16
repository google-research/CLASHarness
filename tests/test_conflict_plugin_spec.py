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

"""Conformance tests for the standardized conflict-plugin surface.

These tests are the enforcement mechanism for ``ConflictPluginSpec`` (the plugin
analogue of the skill-side ``SkillRegistry.validate_specs``). They assert that:

* ``validate_plugins`` finds no conformance violations in the shipped plugins,
* taxonomy-token routing is stable and deduplicated,
* the cross-surface effect check actually rejects impossible effect claims,
* ``ConflictPluginSpec.validate`` rejects malformed specs.

Every plugin now carries a spec, so ``require_spec=True`` is enforced by
``test_every_plugin_has_spec`` (the warn-then-enforce backfill is complete). See
``docs/conflict-plugin-guide.md``.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from clasharness.scenario.plugins import _ALL_PLUGINS
from clasharness.scenario.plugins import _validate_cross_surface
from clasharness.scenario.plugins import get_plugins_for_taxonomy
from clasharness.scenario.plugins import validate_plugins
from clasharness.scenario.plugins.base import BaseConflictPlugin
from clasharness.scenario.plugins.registry import build_registered_plugins
from clasharness.scenario.plugins.registry import registered_plugin_classes
from clasharness.scenario.plugins.spec import ConflictPluginSpec
from clasharness.scenario.plugins.spec import KNOWN_HOOKS


# Every plugin carries a SPEC, so a missing spec is a hard failure
# (warn-then-enforce complete; see docs/conflict-plugin-guide.md).
_ENFORCE_SPEC_REQUIRED = True

# Every registered plugin now carries a SPEC (P1 seeded the deploy/lock coordination
# classes; P2 backfilled the core conflict and multimodal challenge plugins).
_SPEC_BACKFILLED = frozenset(
    {
        # core conflict plugins
        "temporal_staleness",
        "credential_lock",
        "dependency_ordering",
        "deadlock_dependency",
        "evidence_mismatch",
        "time_window_expiration",
        # coordination-protocol plugins
        "distributed_lock",
        "barrier_synchronization",
        "controlled_deployment_gate",
        # multimodal challenge plugins
        "duplicate_avoidance",
        "gap_completion",
        "information_transfer",
        "multi_source_reasoning",
        "state_repair",
        "workflow_completion",
        "vision_decoy",
    }
)


def _make_plugin(spec: ConflictPluginSpec) -> BaseConflictPlugin:
  """Build a throwaway plugin instance carrying ``spec`` (name via SPEC)."""

  class _TmpPlugin(BaseConflictPlugin):
    SPEC = spec

  return _TmpPlugin()


class TestConflictPluginConformance(unittest.TestCase):

  def test_no_conformance_violations(self):
    self.assertEqual(validate_plugins(require_spec=False), [])

  def test_plugin_names_unique(self):
    names = [p.name for p in _ALL_PLUGINS]
    self.assertEqual(len(names), len(set(names)), names)

  def test_registry_matches_all_plugins(self):
    self.assertEqual(len(registered_plugin_classes()), len(_ALL_PLUGINS))
    self.assertEqual(
        [type(p).__name__ for p in build_registered_plugins()],
        [type(p).__name__ for p in _ALL_PLUGINS],
    )

  def test_backfilled_specs_valid_and_consistent(self):
    by_name = {p.name: p for p in _ALL_PLUGINS}
    for name in _SPEC_BACKFILLED:
      plugin = by_name[name]
      spec = plugin.SPEC
      self.assertIsNotNone(spec, f"{name} should carry a SPEC")
      assert spec is not None  # narrow Optional for type-checkers
      spec.validate()  # raises ValueError on failure
      self.assertEqual(spec.name, plugin.name)
      for hook in spec.hooks:
        self.assertIn(hook, KNOWN_HOOKS)

  def test_deploy_lock_specs_declare_expected_surface(self):
    by_name = {p.name: p for p in _ALL_PLUGINS}
    lock_spec = by_name["distributed_lock"].SPEC
    assert lock_spec is not None
    self.assertEqual(lock_spec.skill_families, ("lock",))
    self.assertEqual(
        set(lock_spec.emits_effects), {"lock_acquired", "lock_released"}
    )
    for deploy_name in ("barrier_synchronization", "controlled_deployment_gate"):
      spec = by_name[deploy_name].SPEC
      assert spec is not None
      self.assertEqual(spec.skill_families, ("deploy",))
      self.assertEqual(spec.emits_effects, ("deployment_cutover",))

  @unittest.skipUnless(
      _ENFORCE_SPEC_REQUIRED,
      "SPEC backfill in progress; enforced in P3 via _ENFORCE_SPEC_REQUIRED.",
  )
  def test_every_plugin_has_spec(self):
    self.assertEqual(validate_plugins(require_spec=True), [])


class TestTaxonomyRouting(unittest.TestCase):

  def test_empty_returns_all(self):
    self.assertEqual(len(get_plugins_for_taxonomy([])), len(_ALL_PLUGINS))

  def test_unknown_token_ignored(self):
    self.assertEqual(get_plugins_for_taxonomy(["___not_a_token___"]), [])

  def test_single_tokens_resolve(self):
    for token in [
        "temporal_staleness",
        "credential_lock",
        "dependency_ordering",
        "deadlock_dependency",
        "evidence_mismatch",
        "time_window_expiration",
        "distributed_lock",
        "barrier_synchronization",
        "controlled_deployment_gate",
    ]:
      got = get_plugins_for_taxonomy([token])
      self.assertEqual(len(got), 1, token)
      self.assertEqual(got[0].name, token)

  def test_multi_token_dedup_and_order(self):
    got = get_plugins_for_taxonomy(
        [
            "controlled_deployment_gate",
            "dependency_ordering",
            "controlled_deployment_gate",
        ]
    )
    self.assertEqual(
        [p.name for p in got],
        ["controlled_deployment_gate", "dependency_ordering"],
    )


class TestCrossSurfaceValidation(unittest.TestCase):

  def test_rejects_effect_not_emitted_by_family(self):
    plugin = _make_plugin(
        ConflictPluginSpec(
            name="t", skill_families=("deploy",), emits_effects=("nope",)
        )
    )
    errors = _validate_cross_surface([plugin])
    self.assertTrue(errors)
    self.assertIn("nope", errors[0])

  def test_rejects_unknown_family(self):
    plugin = _make_plugin(
        ConflictPluginSpec(name="t", skill_families=("bogus_family",))
    )
    self.assertTrue(_validate_cross_surface([plugin]))

  def test_rejects_effects_without_families(self):
    plugin = _make_plugin(
        ConflictPluginSpec(name="t", emits_effects=("deployment_cutover",))
    )
    self.assertTrue(_validate_cross_surface([plugin]))

  def test_accepts_valid_surface(self):
    plugin = _make_plugin(
        ConflictPluginSpec(
            name="t", skill_families=("lock",), emits_effects=("lock_acquired",)
        )
    )
    self.assertEqual(_validate_cross_surface([plugin]), [])


class TestSpecValidate(unittest.TestCase):

  def test_empty_name_rejected(self):
    with self.assertRaises(ValueError):
      ConflictPluginSpec(name="").validate()

  def test_unknown_hook_rejected(self):
    with self.assertRaises(ValueError):
      ConflictPluginSpec(name="x", hooks=("bogus_hook",)).validate()

  def test_effective_tokens(self):
    self.assertEqual(ConflictPluginSpec(name="a").effective_tokens, ("a",))
    self.assertEqual(
        ConflictPluginSpec(name="a", taxonomy_tokens=("a", "b")).effective_tokens,
        ("a", "b"),
    )


class TestVisionDecoyPlugin(unittest.TestCase):
  """The selective evidence-contradiction (decoy) difficulty mechanic."""

  def _seed(self, tokens):
    from clasharness.scenario.conflict_vlm_spec import ConflictVLMSeed

    return ConflictVLMSeed.from_dict(
        {"slug": "t", "difficulty_challenges": list(tokens)}
    )

  def test_routes_on_difficulty_token(self):
    got = get_plugins_for_taxonomy(["vision_decoy"])
    self.assertEqual([p.name for p in got], ["vision_decoy"])

  def test_selective_not_default_on(self):
    # A seed that does not carry the token must NOT route to vision_decoy, so
    # the mechanic stays opt-in and does not homogenize other archetypes.
    got = get_plugins_for_taxonomy(["evidence_mismatch"])
    self.assertNotIn("vision_decoy", [p.name for p in got])

  def test_declared_hooks_render_nonempty(self):
    by_name = {p.name: p for p in _ALL_PLUGINS}
    plugin = by_name["vision_decoy"]
    spec = plugin.SPEC
    assert spec is not None
    self.assertEqual(spec.category, "challenge")
    self.assertEqual(
        set(spec.hooks), {"layer0", "layer1", "trajectory", "evals"}
    )
    seed = self._seed(["vision_decoy"])
    for hook in spec.hooks:
      text = getattr(plugin, f"get_{hook}_instruction")(seed)
      self.assertTrue(text.strip(), f"{hook} instruction should be non-empty")
      # The decisive-value-in-pixels contract is the point of the mechanic.
      self.assertIn("decoy", text.lower())

  def test_no_skill_families_or_effects(self):
    # Pure-prompt plugin: it must not claim command families / effects.
    by_name = {p.name: p for p in _ALL_PLUGINS}
    spec = by_name["vision_decoy"].SPEC
    assert spec is not None
    self.assertEqual(spec.skill_families, ())
    self.assertEqual(spec.emits_effects, ())


if __name__ == "__main__":
  unittest.main()
