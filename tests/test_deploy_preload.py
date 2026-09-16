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

"""Unit tests for DeploySkill (M1 barrier-synchronization protocol family).

Covers the cutover barrier (the graded crux), gate-state validation, effect
emission shape for the effect-bearing verbs, preload/persist, snapshot/undo,
and read-only introspection.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from clasharness.skills.impl.deploy_skill import DeploySkill


class TestDeploySkill(unittest.TestCase):
    def _fresh(self, env_vars=None, preload=None):
        d = tempfile.mkdtemp()
        if preload is not None:
            with open(os.path.join(d, "deploy.json"), "w", encoding="utf-8") as f:
                json.dump(preload, f)
        skill = DeploySkill()
        skill.initialize(d, env_vars or {})
        return skill, d

    # ---- request / approve: protocol steps emit --------------------------- #
    # Every state-changing step of the barrier protocol emits, not just the
    # terminal cutover. A round that only advances the protocol (request /
    # approve / gate-set) must still leave a gradable trace, otherwise the
    # checker scaffold has nothing to bind a check to for that round.
    def test_request_emits_effect(self):
        skill, _ = self._fresh()
        res = skill.execute("deploy request --change c1")
        self.assertEqual(res.exit_code, 0)
        self.assertEqual(
            res.state_changes,
            {"effects": {"deployment_requested": [{"change": "c1"}]}},
        )

    def test_request_requires_change(self):
        skill, _ = self._fresh()
        self.assertEqual(skill.execute("deploy request").exit_code, 1)

    def test_approve_emits_effect(self):
        skill, _ = self._fresh()
        res = skill.execute("deploy approve --change c1 --approver rel")
        self.assertEqual(res.exit_code, 0)
        self.assertEqual(
            res.state_changes,
            {
                "effects": {
                    "deployment_approved": [{"change": "c1", "approver": "rel"}]
                }
            },
        )

    def test_approve_requires_approver(self):
        skill, _ = self._fresh()
        res = skill.execute("deploy approve --change c1")
        self.assertEqual(res.exit_code, 1)

    # ---- gate-set validation --------------------------------------------- #
    def test_gate_set_invalid_state_rejected(self):
        skill, _ = self._fresh()
        res = skill.execute("deploy gate-set --gate g1 --state bogus")
        self.assertEqual(res.exit_code, 1)
        self.assertIn("--state must be one of", res.stderr)

    def test_gate_set_valid_emits_effect(self):
        skill, _ = self._fresh()
        res = skill.execute("deploy gate-set --gate g1 --state ready")
        self.assertEqual(res.exit_code, 0)
        self.assertEqual(
            res.state_changes,
            {
                "effects": {
                    "deployment_gate_set": [{"gate": "g1", "state": "ready"}]
                }
            },
        )

    def test_gate_set_rejected_state_emits_nothing(self):
        # A rejected command must not leave a trace: the effect stream is a
        # log of state changes, not of attempts.
        skill, _ = self._fresh()
        res = skill.execute("deploy gate-set --gate g1 --state bogus")
        self.assertEqual(res.exit_code, 1)
        self.assertIsNone(res.state_changes)

    def test_gate_set_requires_gate_and_state(self):
        skill, _ = self._fresh()
        self.assertEqual(skill.execute("deploy gate-set --state ready").exit_code, 1)
        self.assertEqual(skill.execute("deploy gate-set --gate g1").exit_code, 1)

    # ---- execute --------------------------------------------------------- #
    def test_execute_requires_existing_deployment(self):
        skill, _ = self._fresh()
        res = skill.execute("deploy execute --change c1")
        self.assertEqual(res.exit_code, 1)
        self.assertIsNone(res.state_changes)

    def test_execute_emits_effect(self):
        skill, _ = self._fresh()
        skill.execute("deploy request --change c1")
        res = skill.execute("deploy execute --change c1")
        self.assertEqual(res.exit_code, 0)
        self.assertEqual(
            res.state_changes,
            {"effects": {"deployment_executed": [{"change": "c1"}]}},
        )

    # ---- cutover barrier (the graded crux) ------------------------------- #
    def test_cutover_blocked_no_deployment(self):
        skill, _ = self._fresh()
        res = skill.execute("deploy cutover --change c1")
        self.assertEqual(res.exit_code, 1)
        self.assertIsNone(res.state_changes)

    def test_cutover_blocked_no_gates(self):
        skill, _ = self._fresh()
        skill.execute("deploy request --change c1")
        res = skill.execute("deploy cutover --change c1")
        self.assertEqual(res.exit_code, 1)
        self.assertIn("no readiness gates", res.stderr)

    def test_cutover_blocked_gate_not_ready(self):
        skill, _ = self._fresh()
        skill.execute("deploy request --change c1")
        skill.execute("deploy gate-set --gate g1 --state pending")
        res = skill.execute("deploy cutover --change c1")
        self.assertEqual(res.exit_code, 1)
        self.assertIn("g1", res.stderr)

    def test_cutover_blocked_one_of_two_gates_not_ready(self):
        skill, _ = self._fresh()
        skill.execute("deploy request --change c1")
        skill.execute("deploy gate-set --gate g1 --state ready")
        skill.execute("deploy gate-set --gate g2 --state pending")
        res = skill.execute("deploy cutover --change c1")
        self.assertEqual(res.exit_code, 1)
        self.assertIn("g2", res.stderr)
        self.assertNotIn("g1", res.stderr)

    def test_cutover_success_when_all_gates_ready(self):
        skill, _ = self._fresh()
        skill.execute("deploy request --change c1")
        skill.execute("deploy gate-set --gate g1 --state ready")
        skill.execute("deploy gate-set --gate g2 --state ready")
        res = skill.execute("deploy cutover --change c1")
        self.assertEqual(res.exit_code, 0)
        self.assertEqual(
            res.state_changes,
            {"effects": {"deployment_cutover": [{"change": "c1"}]}},
        )

    # ---- rollback -------------------------------------------------------- #
    def test_rollback_emits_effect(self):
        skill, _ = self._fresh()
        res = skill.execute("deploy rollback --change c1")
        self.assertEqual(res.exit_code, 0)
        self.assertEqual(
            res.state_changes,
            {"effects": {"deployment_rolledback": [{"change": "c1"}]}},
        )

    # ---- introspection --------------------------------------------------- #
    def test_status_lists_gates_and_deployments(self):
        skill, _ = self._fresh()
        skill.execute("deploy request --change c1")
        skill.execute("deploy gate-set --gate g1 --state ready")
        res = skill.execute("deploy status")
        self.assertIn("g1", res.stdout)
        self.assertIn("c1", res.stdout)
        # Read-only introspection stays outside the effect stream.
        self.assertIsNone(res.state_changes)

    def test_monitor_reports_status(self):
        skill, _ = self._fresh()
        skill.execute("deploy request --change c1")
        res = skill.execute("deploy monitor --change c1")
        self.assertIn("requested", res.stdout)
        self.assertIsNone(res.state_changes)

    # ---- persistence / preload ------------------------------------------- #
    def test_preload_replace_seeded(self):
        skill, _ = self._fresh(
            preload={"replace_seeded": True, "deploy": [
                {"kind": "deployment", "change": "c1", "status": "requested",
                 "created_at": 0}
            ]},
        )
        # A preloaded deployment is found by execute (would 500 otherwise).
        res = skill.execute("deploy execute --change c1")
        self.assertEqual(res.exit_code, 0)
        self.assertEqual(
            res.state_changes,
            {"effects": {"deployment_executed": [{"change": "c1"}]}},
        )

    def test_persist_across_reinitialize(self):
        skill, d = self._fresh()
        skill.execute("deploy gate-set --gate g1 --state ready")
        reloaded = DeploySkill()
        reloaded.initialize(d, {})
        self.assertEqual([r.get("gate") for r in reloaded._gates()], ["g1"])

    # ---- snapshot / undo -------------------------------------------------- #
    def test_snapshot_roundtrip_restores(self):
        skill, _ = self._fresh()
        skill.execute("deploy request --change c1")
        snap = skill.to_state()
        skill.execute("deploy rollback --change c1")
        self.assertEqual(skill._find_deployment("c1").get("status"), "rolledback")
        skill.from_state(snap)
        self.assertEqual(skill._find_deployment("c1").get("status"), "requested")

    def test_get_state_exposes_deploy_records(self):
        skill, _ = self._fresh()
        skill.execute("deploy gate-set --gate g1 --state ready")
        state = skill.get_state()
        self.assertIn("deploy", state)
        self.assertEqual(state["deploy"][0]["gate"], "g1")


if __name__ == "__main__":
    unittest.main()
