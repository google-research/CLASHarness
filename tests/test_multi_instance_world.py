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

"""Guards for the multi-candidate-target world requirement (supply lever B).

Measured on the v3 corpus, 16 of 30 mechanic scenarios contained exactly ONE
distinct target identifier across every graded protocol command. With a single
candidate in the world, binding the identifier in the checker grades nothing
that a bare ``exists`` did not already grade. These tests pin the requirement
that makes the binding meaningful, and -- more importantly -- pin the
distinction that keeps it from shipping unsatisfiable checks.

The load-bearing test here is
``test_set_member_flags_never_get_decoys``: ``deploy cutover`` requires EVERY
readiness gate to be ready and ``review sign`` requires EVERY aspect to be ok,
so a "decoy gate" or "decoy aspect" that is meant to stay unsatisfied would
block the terminal command forever and make the round unwinnable.
"""

import json
import os
import pathlib
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from clasharness.scenario import conflict_llm_pipeline as P
from clasharness.skills.impl.deploy_skill import DeploySkill
from clasharness.skills.impl.review_skill import ReviewSkill


_SEED_DIR = (
    # No `.resolve()`: under the build system `__file__` is a symlink into the
    # CAS, and resolving it walks out of the runfiles tree.
    pathlib.Path(__file__).parent.parent
    / "clasharness/data/scenario_seeds/conflict_vlm_v1"
)

# Seeds deliberately kept as an untouched control arm; every mechanic-driven
# change must be a no-op for them.
_CONTROL_SEEDS = {
    "clinical_trial_dosage_conflict",
    "payment_chargeback_dispute",
    "sla_support_response",
}


def _seeds():
    out = []
    for path in sorted(_SEED_DIR.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        slug = raw.get("scenario_slug", path.stem)
        tokens = list(raw.get("conflict_taxonomy") or []) + list(
            raw.get("difficulty_challenges") or []
        )
        mech = P._mechanic_command_sets(
            P.get_plugins_for_taxonomy(tokens),
            list(raw.get("allowed_tool_families") or []),
        )
        out.append((slug, mech))
    return out


class TestMechanicTargetFlags(unittest.TestCase):
    def test_no_mechanic_means_no_flags_and_no_instruction(self):
        empty = P.MechanicCommands()
        self.assertEqual(P._mechanic_target_flags(empty), ())
        self.assertEqual(P._mechanic_set_member_flags(empty), ())
        self.assertEqual(P._multi_instance_world_instruction(empty), "")

    def test_control_seeds_are_untouched(self):
        for slug, mech in _seeds():
            if slug not in _CONTROL_SEEDS:
                continue
            with self.subTest(slug=slug):
                self.assertEqual(mech.protocol, ())
                self.assertEqual(P._multi_instance_world_instruction(mech), "")

    def test_every_mechanic_seed_gets_at_least_one_selection_target(self):
        # A mechanic with no choosable target would make the whole lever a
        # no-op for that scenario, which would be a silent regression.
        for slug, mech in _seeds():
            if not mech.protocol:
                continue
            with self.subTest(slug=slug):
                self.assertTrue(
                    P._mechanic_target_flags(mech),
                    f"{slug} routes to a mechanic but exposes no target flag",
                )

    def test_target_and_set_member_flags_are_disjoint(self):
        for slug, mech in _seeds():
            if not mech.protocol:
                continue
            with self.subTest(slug=slug):
                self.assertFalse(
                    set(P._mechanic_target_flags(mech))
                    & set(P._mechanic_set_member_flags(mech))
                )

    def test_selection_targets_come_from_terminal_commands(self):
        for slug, mech in _seeds():
            if not mech.terminals:
                continue
            expected = {
                P._first_required_flag(name)
                for name in mech.terminals
                if P._first_required_flag(name)
            }
            with self.subTest(slug=slug):
                self.assertEqual(set(P._mechanic_target_flags(mech)), expected)

    def test_known_flag_classification(self):
        # Pins the classification actually produced for the shipped seeds, so a
        # future edit that reclassifies a flag has to say so out loud.
        expected = {
            "access_request_approval": (("--resource",), ()),
            "barrier_sync_deployment": (("--change",), ("--gate",)),
            "collaborative_contract_review": (("--document",), ("--aspect",)),
            "coordinated_containment": (
                ("--account", "--device", "--token"),
                (),
            ),
            # Newly mechanized when ``deadlock_dependency`` was bound to the
            # approval family (it names escalate/provisional/grant verbatim).
            "deadlock_multi_approval": (("--resource",), ()),
            "distributed_lock_shared_resource": (("--resource",), ()),
            # Newly mechanized when ``dependency_ordering`` was bound to the
            # deploy family (gate-set/status/cutover is the seed's own wording).
            "employee_onboarding_workflow": (("--change",), ("--gate",)),
            "expiring_resource_hold": (("--resource",), ()),
            "it_deployment_gate": (("--change",), ("--gate",)),
            "multi_device_pairing": (("--device",), ()),
            "multi_party_access_permit": (("--resource",), ()),
            "multi_tab_slot_monitoring": (("--resource",), ()),
        }
        seen = {}
        for slug, mech in _seeds():
            if mech.protocol:
                seen[slug] = (
                    P._mechanic_target_flags(mech),
                    P._mechanic_set_member_flags(mech),
                )
        self.assertEqual(seen, expected)


class TestUnwinnabilityGuard(unittest.TestCase):
    """The reason set-member flags are held apart from selection targets."""

    def test_an_unready_gate_blocks_cutover_forever(self):
        # If '--gate' were treated as a decoy flag, the world would ship gates
        # that are meant to stay un-ready -- and then no solver could ever
        # reach 'deployment_cutover'.
        import tempfile

        skill = DeploySkill()
        skill.initialize(tempfile.mkdtemp(), {})
        skill.execute("deploy request --change c1")
        skill.execute("deploy gate-set --gate g_correct --state ready")
        skill.execute("deploy gate-set --gate g_decoy --state pending")
        blocked = skill.execute("deploy cutover --change c1")
        self.assertEqual(blocked.exit_code, 1)
        self.assertIsNone(blocked.state_changes)

        skill.execute("deploy gate-set --gate g_decoy --state ready")
        unblocked = skill.execute("deploy cutover --change c1")
        self.assertEqual(unblocked.exit_code, 0)

    def test_a_pending_aspect_blocks_sign_forever(self):
        import tempfile

        skill = ReviewSkill()
        skill.initialize(tempfile.mkdtemp(), {})
        skill.execute("review assign --aspect legal --reviewer a")
        skill.execute("review submit --aspect legal --status ok")
        skill.execute("review assign --aspect decoy --reviewer b")
        blocked = skill.execute("review sign --document d1")
        self.assertEqual(blocked.exit_code, 1)

        skill.execute("review submit --aspect decoy --status ok")
        unblocked = skill.execute("review sign --document d1")
        self.assertEqual(unblocked.exit_code, 0)

    def test_set_member_flags_never_get_decoys(self):
        # The instruction text must ask for breadth for set members and decoys
        # only for selection targets -- never the other way round.
        for slug, mech in _seeds():
            if not mech.protocol:
                continue
            text = P._multi_instance_world_instruction(mech)
            members = P._mechanic_set_member_flags(mech)
            with self.subTest(slug=slug):
                if members:
                    self.assertIn("COMPLETE THE PRECONDITION SET", text)
                    self.assertIn("These are NOT decoys", text)
                    for flag in members:
                        # The set-member flag must appear only in the breadth
                        # block, never in the decoy block.
                        decoy_block = text.split("COMPLETE THE PRECONDITION SET")[0]
                        self.assertNotIn(f"`{flag}`", decoy_block)
                else:
                    self.assertNotIn("COMPLETE THE PRECONDITION SET", text)


class TestInstructionText(unittest.TestCase):
    def test_selection_block_states_the_fairness_contract(self):
        for slug, mech in _seeds():
            if not P._mechanic_target_flags(mech):
                continue
            text = P._multi_instance_world_instruction(mech)
            with self.subTest(slug=slug):
                self.assertIn("MULTIPLE CANDIDATE TARGETS", text)
                self.assertIn("EXACTLY ONE", text)
                # Difficulty must come from having to read the world, never
                # from an under-determined choice.
                self.assertIn("UNIQUELY determined", text)
                self.assertIn(str(P.MECHANIC_TARGET_CANDIDATES), text)

    def test_every_target_flag_is_named_in_the_text(self):
        for slug, mech in _seeds():
            flags = P._mechanic_target_flags(mech)
            if not flags:
                continue
            text = P._multi_instance_world_instruction(mech)
            with self.subTest(slug=slug):
                for flag in flags:
                    self.assertIn(f"`{flag}`", text)

    def test_candidate_count_is_greater_than_two(self):
        # 1-in-2 guessing is close enough to free that it would blunt rather
        # than sharpen the measurement.
        self.assertGreaterEqual(P.MECHANIC_TARGET_CANDIDATES, 3)


class TestMechanicHooksAreFamilyGated(unittest.TestCase):
    """Taxonomy routing and tool availability are independent axes.

    A seed can match a plugin's taxonomy slug while its
    ``allowed_tool_families`` withholds the family that plugin mechanizes.
    ``_mechanic_command_sets`` intersects with the seed's families so the gates
    are correctly inert, but the pipeline collects trajectory/evals hooks by
    calling the methods and testing truthiness -- with no family filter. An
    ungated mechanized hook therefore instructs the generator to drive commands
    the seed does not expose, and (worse) mutates the prompt of the control-arm
    seeds. This regression was shipped once; these tests keep it from
    returning for any plugin.
    """

    def _seed_objects(self):
        from clasharness.scenario.conflict_vlm_spec import ConflictVLMSeed

        for path in sorted(_SEED_DIR.glob("*.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            slug = raw.get("scenario_slug", path.stem)
            tokens = list(raw.get("conflict_taxonomy") or []) + list(
                raw.get("difficulty_challenges") or []
            )
            yield slug, ConflictVLMSeed(**raw), P.get_plugins_for_taxonomy(tokens)

    def test_unavailable_family_means_no_mechanized_hook(self):
        for slug, seed, plugins in self._seed_objects():
            families = set(seed.allowed_tool_families or ())
            for plugin in plugins:
                declared = set(getattr(plugin.SPEC, "skill_families", ()) or ())
                if not declared or (declared & families):
                    continue
                with self.subTest(slug=slug, plugin=plugin.name):
                    self.assertFalse(
                        plugin.mechanic_available(seed),
                        "mechanic_available must track allowed_tool_families",
                    )
                    self.assertEqual(
                        plugin.get_trajectory_instruction(seed),
                        "",
                        f"{plugin.name} would tell {slug} to drive "
                        f"{sorted(declared)} commands it cannot run",
                    )
                    self.assertEqual(plugin.get_evals_instruction(seed), "")

    def test_narrative_only_plugins_are_never_gated(self):
        # Plugins that mechanize nothing must keep firing everywhere; the guard
        # must not silently mute the 7 challenge / 6 narrative plugins.
        for slug, seed, plugins in self._seed_objects():
            for plugin in plugins:
                if getattr(plugin.SPEC, "skill_families", ()):
                    continue
                with self.subTest(slug=slug, plugin=plugin.name):
                    self.assertTrue(plugin.mechanic_available(seed))

    def test_control_seed_prompts_carry_no_mechanic_vocabulary(self):
        # The control arm's value is that it is untreated. Any command verb
        # from a family the seed cannot run is both unfair and contaminating.
        forbidden = (
            "deploy cutover",
            "deploy gate-set",
            "approval grant",
            "approval provisional",
            "approval escalate",
            "booking hold",
            "booking extend",
            "pairing start",
        )
        for slug, seed, plugins in self._seed_objects():
            if slug not in _CONTROL_SEEDS:
                continue
            text = "\n".join(
                getattr(plugin, f"get_{hook}_instruction")(seed)
                for plugin in plugins
                for hook in ("layer0", "layer1", "layer1b", "trajectory", "evals")
            )
            for verb in forbidden:
                with self.subTest(slug=slug, verb=verb):
                    self.assertNotIn(verb, text)


if __name__ == "__main__":
    unittest.main()
