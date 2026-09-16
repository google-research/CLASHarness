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

"""Unit tests for ContainmentSkill (M4 coordinated multi-point containment).

Covers each of the four control-point verbs and its ``containment_action``
effect shape, the idempotency guard (re-actioning a contained point is a no-op
that emits nothing), the anti-spam property (a single verb cannot be spammed to
reach four effects — only four *distinct* points can), full containment across
all four points, required-flag validation, read-only status, preload/persist,
and snapshot/undo.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from clasharness.skills.impl.containment_skill import ContainmentSkill


class TestContainmentSkill(unittest.TestCase):
    def _fresh(self, env_vars=None, preload=None):
        d = tempfile.mkdtemp()
        if preload is not None:
            with open(os.path.join(d, "containment.json"), "w", encoding="utf-8") as f:
                json.dump(preload, f)
        skill = ContainmentSkill()
        skill.initialize(d, env_vars or {})
        return skill, d

    def _contain_all(self, skill, account="alice", device="laptop-7", token="tok_abc"):
        skill.execute(f"containment revoke-sessions --account {account}")
        skill.execute(f"containment disable-forwarding --account {account}")
        skill.execute(f"containment isolate-device --device {device}")
        skill.execute(f"containment block-token --token {token}")

    # ---- each verb contains its point + effect shape --------------------- #
    def test_revoke_sessions_emits_effect(self):
        skill, _ = self._fresh({})
        res = skill.execute("containment revoke-sessions --account alice")
        self.assertEqual(res.exit_code, 0)
        self.assertEqual(
            res.state_changes,
            {"effects": {"containment_action": [
                {"point": "sessions", "verb": "revoke-sessions", "target": "alice"}
            ]}},
        )

    def test_disable_forwarding_emits_effect(self):
        skill, _ = self._fresh({})
        res = skill.execute("containment disable-forwarding --account alice")
        self.assertEqual(res.exit_code, 0)
        self.assertEqual(
            res.state_changes["effects"]["containment_action"][0]["point"], "forwarding"
        )

    def test_isolate_device_emits_effect(self):
        skill, _ = self._fresh({})
        res = skill.execute("containment isolate-device --device laptop-7")
        self.assertEqual(res.exit_code, 0)
        payload = res.state_changes["effects"]["containment_action"][0]
        self.assertEqual(payload["point"], "device")
        self.assertEqual(payload["target"], "laptop-7")

    def test_block_token_emits_effect(self):
        skill, _ = self._fresh({})
        res = skill.execute("containment block-token --token tok_abc")
        self.assertEqual(res.exit_code, 0)
        self.assertEqual(
            res.state_changes["effects"]["containment_action"][0]["point"], "token"
        )

    # ---- idempotency guard (D1) ------------------------------------------ #
    def test_reaction_of_contained_point_is_noop(self):
        skill, _ = self._fresh({})
        first = skill.execute("containment revoke-sessions --account alice")
        self.assertEqual(first.exit_code, 0)
        second = skill.execute("containment revoke-sessions --account alice")
        self.assertEqual(second.exit_code, 1)
        self.assertIn("already contained", second.stderr)
        self.assertIsNone(second.state_changes)

    def test_reaction_with_different_target_still_noop(self):
        # The point is keyed by the control-point type (the verb), not the
        # target, so re-running the verb against a different account is still a
        # no-op — one verb == one point.
        skill, _ = self._fresh({})
        skill.execute("containment revoke-sessions --account alice")
        res = skill.execute("containment revoke-sessions --account bob")
        self.assertEqual(res.exit_code, 1)
        self.assertIsNone(res.state_changes)

    # ---- anti-spam: one verb cannot reach four --------------------------- #
    def test_spamming_one_verb_yields_a_single_effect(self):
        skill, _ = self._fresh({})
        emitted = 0
        for _ in range(4):
            res = skill.execute("containment revoke-sessions --account alice")
            if res.state_changes is not None:
                emitted += len(res.state_changes["effects"]["containment_action"])
        self.assertEqual(emitted, 1)

    # ---- full containment: four distinct points -> four effects ---------- #
    def test_all_four_points_emit_four_effects(self):
        skill, _ = self._fresh({})
        points = set()
        for cmd in (
            "containment revoke-sessions --account alice",
            "containment disable-forwarding --account alice",
            "containment isolate-device --device laptop-7",
            "containment block-token --token tok_abc",
        ):
            res = skill.execute(cmd)
            self.assertEqual(res.exit_code, 0)
            points.add(res.state_changes["effects"]["containment_action"][0]["point"])
        self.assertEqual(points, {"sessions", "forwarding", "device", "token"})

    # ---- required flags -------------------------------------------------- #
    def test_revoke_sessions_requires_account(self):
        skill, _ = self._fresh({})
        res = skill.execute("containment revoke-sessions")
        self.assertEqual(res.exit_code, 1)
        self.assertIsNone(res.state_changes)

    def test_isolate_device_requires_device(self):
        skill, _ = self._fresh({})
        self.assertEqual(skill.execute("containment isolate-device").exit_code, 1)

    def test_block_token_requires_token(self):
        skill, _ = self._fresh({})
        self.assertEqual(skill.execute("containment block-token").exit_code, 1)

    def test_unknown_subcommand_fails(self):
        skill, _ = self._fresh({})
        self.assertEqual(skill.execute("containment nuke --all").exit_code, 1)

    # ---- status ---------------------------------------------------------- #
    def test_status_reports_progress(self):
        skill, _ = self._fresh({})
        skill.execute("containment revoke-sessions --account alice")
        out = skill.execute("containment status").stdout
        self.assertIn("contained 1/4", out)
        self.assertIn("sessions", out)

    def test_status_after_full_containment(self):
        skill, _ = self._fresh({})
        self._contain_all(skill)
        out = skill.execute("containment status").stdout
        self.assertIn("contained 4/4", out)

    # ---- persistence / preload ------------------------------------------- #
    def test_preload_marks_point_contained(self):
        skill, _ = self._fresh(
            {},
            preload={"replace_seeded": True, "containment": [
                {"kind": "control_point", "point": "sessions",
                 "verb": "revoke-sessions", "target": "alice",
                 "status": "contained", "contained_at": 0},
            ]},
        )
        # The preloaded point is already contained -> re-actioning is a no-op.
        res = skill.execute("containment revoke-sessions --account alice")
        self.assertEqual(res.exit_code, 1)
        self.assertIn("contained 1/4", skill.execute("containment status").stdout)

    def test_persist_across_reinitialize(self):
        skill, d = self._fresh({})
        skill.execute("containment revoke-sessions --account alice")
        reloaded = ContainmentSkill()
        reloaded.initialize(d, {})
        # The contained point survives the reload -> re-actioning is a no-op.
        self.assertEqual(
            reloaded.execute("containment revoke-sessions --account alice").exit_code, 1
        )
        self.assertIn("contained 1/4", reloaded.execute("containment status").stdout)

    # ---- snapshot / undo -------------------------------------------------- #
    def test_snapshot_roundtrip_restores(self):
        skill, _ = self._fresh({})
        skill.execute("containment revoke-sessions --account alice")
        snap = skill.to_state()
        skill.execute("containment disable-forwarding --account alice")
        self.assertIn("contained 2/4", skill.execute("containment status").stdout)
        skill.from_state(snap)
        self.assertIn("contained 1/4", skill.execute("containment status").stdout)
        # After undo, forwarding is active again and can be re-actioned.
        res = skill.execute("containment disable-forwarding --account alice")
        self.assertEqual(res.exit_code, 0)

    def test_get_state_exposes_containment_records(self):
        skill, _ = self._fresh({})
        skill.execute("containment revoke-sessions --account alice")
        state = skill.get_state()
        self.assertIn("containment", state)
        self.assertEqual(state["containment"][0]["point"], "sessions")


if __name__ == "__main__":
    unittest.main()
