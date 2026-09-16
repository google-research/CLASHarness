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

"""Tests for multimodal and multi-agent difficulty challenges via plugins."""

import unittest
from clasharness.scenario.conflict_vlm_spec import ConflictVLMSeed
from clasharness.scenario.plugins import get_plugins_for_taxonomy

class TestDifficultyChallenges(unittest.TestCase):

  def test_get_plugins_empty(self):
    plugins = get_plugins_for_taxonomy([])
    # If empty taxonomy, it defaults to returning all plugins registry
    self.assertGreater(len(plugins), 0)

  def test_get_challenge_plugins_and_hooks(self):
    challenges = [
        "duplicate_avoidance",
        "gap_completion",
        "information_transfer",
        "multi_source_reasoning",
        "state_repair",
        "workflow_completion",
    ]
    plugins = get_plugins_for_taxonomy(challenges)
    self.assertEqual(len(plugins), len(challenges))

    # Construct mock seed
    seed = ConflictVLMSeed(
        slug="test",
        domain="system",
        title="Test",
        target_rounds=2,
        core_conflict="",
        conflict_taxonomy=[],
        participants=["Alice", "Bob"],
        required_constraints=[],
        allowed_tool_families=["file"],
        difficulty_challenges=challenges,
    )

    # Verify hooks return correct constraint instructions
    l0_hooks = "\n".join(p.get_layer0_instruction(seed) for p in plugins if p.get_layer0_instruction(seed))
    traj_hooks = "\n".join(p.get_trajectory_instruction(seed) for p in plugins if p.get_trajectory_instruction(seed))
    eval_hooks = "\n".join(p.get_evals_instruction(seed) for p in plugins if p.get_evals_instruction(seed))

    self.assertIn("DUPLICATE AVOIDANCE", l0_hooks)
    self.assertIn("DUPLICATE AVOIDANCE CHALLENGE", traj_hooks)
    self.assertIn("GAP COMPLETION", l0_hooks)
    self.assertIn("GAP COMPLETION CHALLENGE", traj_hooks)
    self.assertIn("INFORMATION TRANSFER", l0_hooks)
    self.assertIn("INFORMATION TRANSFER CHALLENGE", traj_hooks)
    self.assertIn("EVALS", eval_hooks)
    self.assertIn("MULTI-SOURCE REASONING", l0_hooks)
    self.assertIn("STATE REPAIR", l0_hooks)
    self.assertIn("WORKFLOW COMPLETION", traj_hooks)

  def test_seed_from_dict_parses_difficulty_challenges(self):
    data = {
        "slug": "test_scenario",
        "domain": "test",
        "title": "Test Scenario",
        "target_rounds": 5,
        "core_conflict": "test conflict",
        "conflict_taxonomy": ["deadlock"],
        "participants": ["Alice", "Bob"],
        "required_constraints": [],
        "allowed_tool_families": ["file"],
        "difficulty_challenges": ["gap_completion", "state_repair"]
    }
    seed = ConflictVLMSeed.from_dict(data)
    self.assertEqual(seed.difficulty_challenges, ["gap_completion", "state_repair"])

if __name__ == "__main__":
  unittest.main()
