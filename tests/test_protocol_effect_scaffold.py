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

"""Unit tests for the protocol-effect scaffolding path in the LLM pipeline.

Verifies that effect-bearing commands are (a) classified as ``protocol_effect``,
(b) scaffolded into an effect checker bound to the TARGET the command acts on,
and (c) that both generated-round validators accept the registered effect types
while still rejecting unknown ones.

Two deliberate behaviour changes are pinned here, because both look like
regressions against the original M1 tests and are not:

  * Protocol PREREQUISITE verbs (``deploy request``/``approve``/``gate-set``,
    ``approval request``/``approve``, ``pairing start``/``enter-code``, ...) are
    no longer ``plain``. They emit effects now, which is what lets a round be
    graded on advancing a protocol instead of only on completing it. Before this
    they were invisible to the checker scaffold, so 9 of every 10 protocol
    rounds had nothing to assert.
  * A protocol command scaffolds ``any_field_equals`` bound to its target rather
    than a bare ``exists``. ``exists`` passes for ANY invocation, so it graded
    "did you run the verb" when the difficulty on offer is "did you pick the
    right resource out of the conflicting evidence".

Read-only verbs (``status``/``list``/``monitor``) must still scaffold nothing;
that is the property that keeps "inspecting the protocol" from scoring.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from clasharness.scenario.conflict_llm_pipeline import (
    ALLOWED_EFFECT_TYPES,
    _EFFECT_COMMAND_NAMES,
    _classify_step_command,
    _scaffold_round_checkers,
    _validate_generated_round_steps,
    _validate_generated_round_trajectory_and_evals_compiled,
    command_spec_for_effect,
    validate_effect_command_registry,
)

# A valid command line for each effect-bearing command name.
_CMD_FOR_NAME = {
    "lock.acquire": "lock acquire --resource r --holder a",
    "lock.release": "lock release --resource r",
    "deploy.execute": "deploy execute --change c",
    "deploy.cutover": "deploy cutover --change c",
    "deploy.rollback": "deploy rollback --change c",
}
_FAMILIES = ["lock", "deploy"]


class TestClassifyStepCommand(unittest.TestCase):
    def test_effect_commands_classified_protocol_effect(self):
        for name, cmd in _CMD_FOR_NAME.items():
            with self.subTest(name=name):
                category, spec = _classify_step_command(cmd, _FAMILIES)
                self.assertEqual(category, "protocol_effect")
                self.assertEqual(getattr(spec, "name", ""), name)

    def test_read_only_verbs_stay_plain(self):
        """Inspecting the protocol must never scaffold a check."""
        for cmd in [
            "deploy status",
            "deploy monitor --change c",
            "lock status --resource r",
            "lock list",
        ]:
            with self.subTest(cmd=cmd):
                category, _ = _classify_step_command(cmd, _FAMILIES)
                self.assertEqual(category, "plain")

    def test_prerequisite_verbs_are_protocol_effects(self):
        """The supply fix: advancing a protocol is now gradable.

        These were ``plain`` before. If they regress to ``plain`` the advance
        gate becomes unsatisfiable except by repeating the terminal command.
        """
        for cmd in [
            "deploy gate-set --gate g --state ready",
            "deploy request --change c",
            "deploy approve --change c --approver r",
        ]:
            with self.subTest(cmd=cmd):
                category, _ = _classify_step_command(cmd, _FAMILIES)
                self.assertEqual(category, "protocol_effect")

    def test_disallowed_family_degrades_to_plain_none(self):
        # A lock command is not resolvable when only 'deploy' is allowed.
        category, spec = _classify_step_command(
            "lock acquire --resource r", ["deploy"]
        )
        self.assertEqual((category, spec), ("plain", None))


class TestScaffoldProtocolEffect(unittest.TestCase):
    def test_each_effect_command_scaffolds_target_bound_checker(self):
        for name, cmd in _CMD_FOR_NAME.items():
            with self.subTest(name=name):
                scaffolds, semantic_needed = _scaffold_round_checkers(
                    [{"step_number": 1, "command": cmd}], _FAMILIES
                )
                self.assertEqual(len(scaffolds), 1)
                self.assertEqual(
                    scaffolds[0]["checker"],
                    {
                        "type": "effect",
                        "effect_type": _EFFECT_COMMAND_NAMES[name],
                        "condition": "any_field_equals",
                        # every command in _CMD_FOR_NAME targets 'r' or 'c'
                        "expected": {"value": "r" if "--resource" in cmd else "c"},
                    },
                )
                # A protocol-effect step needs no semantic (LLM) pass.
                self.assertIsNone(scaffolds[0]["semantic"])
                self.assertIsNone(scaffolds[0]["extra_checkers"])
                self.assertEqual(semantic_needed, [])

    def test_distinct_targets_scaffold_distinct_expected_values(self):
        """Why no count/distinct condition is needed.

        N commands against N distinct targets produce N checks with N distinct
        expected values, so replaying a single target satisfies exactly one of
        them. ``lock acquire`` re-emits its effect on an idempotent re-acquire,
        so a raw ``count_gte`` here would have been spammable.
        """
        scaffolds, _ = _scaffold_round_checkers(
            [
                {"step_number": 1, "command": "lock acquire --resource db-a"},
                {"step_number": 2, "command": "lock acquire --resource db-b"},
            ],
            _FAMILIES,
        )
        values = [s["checker"]["expected"]["value"] for s in scaffolds]
        self.assertEqual(["db-a", "db-b"], values)

    def test_missing_target_falls_back_to_exists(self):
        """An unparsable target must degrade, never scaffold something unwinnable."""
        scaffolds, _ = _scaffold_round_checkers(
            [{"step_number": 1, "command": "lock acquire"}], _FAMILIES
        )
        checker = scaffolds[0]["checker"]
        if checker is not None:  # resolvable-but-targetless degrades to exists
            self.assertEqual("exists", checker["condition"])
            self.assertNotIn("expected", checker)

    def test_lock_acquire_maps_to_lock_acquired(self):
        scaffolds, _ = _scaffold_round_checkers(
            [{"step_number": 1, "command": "lock acquire --resource db"}], _FAMILIES
        )
        self.assertEqual(scaffolds[0]["checker"]["effect_type"], "lock_acquired")

    def test_deploy_cutover_maps_to_deployment_cutover(self):
        scaffolds, _ = _scaffold_round_checkers(
            [{"step_number": 1, "command": "deploy cutover --change c"}], _FAMILIES
        )
        self.assertEqual(scaffolds[0]["checker"]["effect_type"], "deployment_cutover")

    def test_non_effect_command_has_no_checker(self):
        scaffolds, _ = _scaffold_round_checkers(
            [{"step_number": 1, "command": "deploy status"}], _FAMILIES
        )
        self.assertIsNone(scaffolds[0]["checker"])


def _steps(*effect_types):
    """Build a 4-step steps_payload where each step carries an effect checker."""
    cmds = list(_CMD_FOR_NAME.values())
    steps = []
    for i, et in enumerate(effect_types):
        steps.append({
            "command": cmds[i % len(cmds)],
            "checker": {"type": "effect", "effect_type": et, "condition": "exists"},
        })
    return {"steps": steps}


class TestValidatorsAcceptNewEffectTypes(unittest.TestCase):
    def test_effect_command_targets_are_all_allowed(self):
        self.assertTrue(
            set(_EFFECT_COMMAND_NAMES.values()).issubset(ALLOWED_EFFECT_TYPES)
        )

    def test_steps_validator_accepts_new_effect_types(self):
        payload = _steps(
            "lock_acquired", "lock_released", "deployment_executed",
            "deployment_cutover",
        )
        # Should not raise.
        _validate_generated_round_steps(payload, _FAMILIES)

    def test_steps_validator_rejects_unknown_effect_type(self):
        payload = _steps(
            "lock_acquired", "lock_released", "deployment_executed", "bogus_effect",
        )
        with self.assertRaises(ValueError):
            _validate_generated_round_steps(payload, _FAMILIES)

    def test_compiled_validator_accepts_new_effect_types(self):
        round_payload = {
            "rounds": [{"solution_commands": list(_CMD_FOR_NAME.values())}],
            "evals": [{
                "checks": [
                    {"type": "effect", "effect_type": et, "condition": "exists"}
                    for et in _EFFECT_COMMAND_NAMES.values()
                ]
            }],
        }
        # Should not raise.
        _validate_generated_round_trajectory_and_evals_compiled(
            round_payload, _FAMILIES
        )

    def test_compiled_validator_rejects_unknown_effect_type(self):
        round_payload = {
            "rounds": [{"solution_commands": list(_CMD_FOR_NAME.values())}],
            "evals": [{
                "checks": [
                    {"type": "effect", "effect_type": "nope", "condition": "exists"}
                ]
            }],
        }
        with self.assertRaises(ValueError):
            _validate_generated_round_trajectory_and_evals_compiled(
                round_payload, _FAMILIES
            )


class TestEffectCommandRegistryIntegrity(unittest.TestCase):
    """The effect registry and the skill precondition SSOT stay in agreement.

    This guards the deterministic prerequisite resolver's lookup path
    (``effect_type`` -> command -> ``CommandSpec.requires``) against silent
    drift: adding a ``requires`` to a skill command without registering its
    effect, or renaming a command/effect, would otherwise leave the resolver
    unable to recover a failing effect's precondition contract.
    """

    def test_registry_is_conformant(self):
        errors = validate_effect_command_registry()
        self.assertEqual(errors, [], "\n".join(errors))

    def test_every_registered_effect_resolves_to_a_command_spec(self):
        for effect in _EFFECT_COMMAND_NAMES.values():
            with self.subTest(effect=effect):
                self.assertIsNotNone(command_spec_for_effect(effect))

    def test_unknown_effect_resolves_to_none(self):
        self.assertIsNone(command_spec_for_effect("not_a_real_effect"))

    def test_every_registered_command_is_reachable_from_the_scaffold(self):
        """Registering an effect is worthless if the scaffold cannot see it.

        The skill SSOT spells some verbs with a hyphen (``deploy gate-set``)
        while the AgentToolSpec the scaffold resolves against reports a few of
        those same verbs with an underscore (``deploy.gate_set``) -- and keeps
        the hyphen for others (``containment.block-token``). Three commands fell
        in that gap and were registered-but-ungradable: the protocol gate could
        demand them while the checker scaffold produced nothing to grade them
        with. This asserts the round-trip for EVERY registered command so the
        gap cannot silently reopen.
        """
        from clasharness.scenario.conflict_llm_pipeline import (
            _classify_step_command,
            _skill_command_specs,
        )

        sample = {
            "--resource": "R1", "--change": "C1", "--device": "D1",
            "--account": "A1", "--token": "T1", "--gate": "G1",
            "--state": "ready", "--approver": "boss", "--to": "boss",
            "--holder": "me", "--aspect": "legal", "--reviewer": "boss",
            "--status": "ok", "--document": "DOC1", "--code": "123456",
            "--ttl": "60", "--scope": "read", "--issue": "none",
            "--section": "main", "--approvers": "boss",
        }
        specs = _skill_command_specs()
        for dotted, effect in _EFFECT_COMMAND_NAMES.items():
            with self.subTest(command=dotted):
                cs = specs[dotted]
                family = dotted.split(".", 1)[0]
                # Every declared arg, not just the skill's ``required``: the two
                # SSOTs disagree on what is mandatory (``deploy approve`` needs
                # ``--approver`` per the AgentToolSpec but the skill spec lists
                # only ``--change``), and an incomplete line is rejected before
                # classification, which would make this guard vacuous.
                line = cs.name + "".join(
                    f" {f} {sample.get(f, 'X')}" for f in (cs.args or cs.required or ())
                )
                category, spec = _classify_step_command(line, [family])
                self.assertEqual(
                    "protocol_effect", category,
                    msg=f"{dotted} is registered but not classified as an effect",
                )
                scaffolds, _ = _scaffold_round_checkers(
                    [{"step_number": 1, "command": line}], [family]
                )
                checker = scaffolds[0]["checker"]
                self.assertIsNotNone(
                    checker, msg=f"{dotted} scaffolded no checker"
                )
                self.assertEqual(effect, checker["effect_type"])

    def test_multistep_terminals_are_registered(self):
        # Every command that declares a precondition contract must be reachable
        # from a registered effect (otherwise the resolver is blind to it).
        from clasharness.scenario.conflict_llm_pipeline import (
            _skill_command_specs,
        )

        for dotted, cmd in _skill_command_specs().items():
            if getattr(cmd, "requires", None):
                with self.subTest(command=dotted):
                    self.assertIn(dotted, _EFFECT_COMMAND_NAMES)


if __name__ == "__main__":
    unittest.main()
