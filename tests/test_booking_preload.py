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

"""Unit tests for BookingSkill (M3 reversible-action-hold protocol family).

Covers holder resolution, the reversible hold/confirm/release lifecycle, effect
emission shape, the virtual-time confirmation window (confirm/extend against the
plumbed ``_now`` clock), mutual-exclusion conflict detection, read-only
introspection, preload/persist, and snapshot/undo.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from clasharness.skills.impl.booking_skill import BookingSkill


class TestBookingSkill(unittest.TestCase):
    def _fresh(self, env_vars=None, preload=None):
        d = tempfile.mkdtemp()
        if preload is not None:
            with open(os.path.join(d, "booking.json"), "w", encoding="utf-8") as f:
                json.dump(preload, f)
        skill = BookingSkill()
        skill.initialize(d, env_vars or {})
        return skill, d

    # ---- holder resolution ------------------------------------------------ #
    def test_holder_defaults_to_env_actor(self):
        skill, _ = self._fresh({"CLASHARNESS_ACTOR": "alice"})
        res = skill.execute("booking hold --resource room-a")
        self.assertEqual(res.exit_code, 0)
        self.assertEqual(
            res.state_changes,
            {"effects": {"booking_held": [
                {"resource": "room-a", "holder": "alice", "ttl": 300, "created_at": 0}
            ]}},
        )

    def test_holder_defaults_to_coordinator(self):
        skill, _ = self._fresh({})
        res = skill.execute("booking hold --resource room-a")
        self.assertEqual(
            res.state_changes["effects"]["booking_held"][0]["holder"], "coordinator"
        )

    def test_explicit_holder_overrides_env(self):
        skill, _ = self._fresh({"CLASHARNESS_ACTOR": "alice"})
        res = skill.execute("booking hold --resource room-a --holder bob")
        self.assertEqual(
            res.state_changes["effects"]["booking_held"][0]["holder"], "bob"
        )

    # ---- effect shape ----------------------------------------------------- #
    def test_confirm_emits_effect(self):
        skill, _ = self._fresh({})
        skill.execute("booking hold --resource room-a --holder alice")
        res = skill.execute("booking confirm --resource room-a")
        self.assertEqual(res.exit_code, 0)
        self.assertEqual(
            res.state_changes,
            {"effects": {"booking_confirmed": [
                {"resource": "room-a", "holder": "alice"}
            ]}},
        )

    def test_release_emits_effect(self):
        skill, _ = self._fresh({})
        skill.execute("booking hold --resource room-a --holder alice")
        res = skill.execute("booking release --resource room-a")
        self.assertEqual(res.exit_code, 0)
        self.assertEqual(
            res.state_changes,
            {"effects": {"booking_released": [{"resource": "room-a"}]}},
        )

    def test_custom_ttl_is_recorded(self):
        skill, _ = self._fresh({})
        res = skill.execute("booking hold --resource gpu --ttl 120")
        self.assertEqual(res.state_changes["effects"]["booking_held"][0]["ttl"], 120)

    # ---- virtual-time confirmation window --------------------------------- #
    def test_confirm_within_window_succeeds(self):
        skill, _ = self._fresh({})
        skill._now = 0
        skill.execute("booking hold --resource gpu --ttl 300")
        skill._now = 299  # still inside the window
        res = skill.execute("booking confirm --resource gpu")
        self.assertEqual(res.exit_code, 0)
        self.assertIn("booking_confirmed", res.state_changes["effects"])

    def test_confirm_after_expiry_fails_without_effect(self):
        skill, _ = self._fresh({})
        skill._now = 0
        skill.execute("booking hold --resource gpu --ttl 300")
        skill._now = 400  # window has closed (simulates `wait 400`)
        res = skill.execute("booking confirm --resource gpu")
        self.assertEqual(res.exit_code, 1)
        self.assertIn("expired", res.stderr)
        self.assertIsNone(res.state_changes)

    def test_created_at_stamps_virtual_clock(self):
        skill, _ = self._fresh({})
        skill._now = 42
        res = skill.execute("booking hold --resource gpu")
        self.assertEqual(
            res.state_changes["effects"]["booking_held"][0]["created_at"], 42
        )

    def test_extend_refreshes_window(self):
        skill, _ = self._fresh({})
        skill._now = 0
        skill.execute("booking hold --resource gpu --ttl 300")
        skill._now = 250
        ext = skill.execute("booking extend --resource gpu")  # created_at -> 250
        self.assertEqual(ext.exit_code, 0)
        skill._now = 500  # 500 - 250 = 250 < 300, still live
        res = skill.execute("booking confirm --resource gpu")
        self.assertEqual(res.exit_code, 0)
        self.assertIn("booking_confirmed", res.state_changes["effects"])

    def test_extend_after_expiry_fails(self):
        skill, _ = self._fresh({})
        skill._now = 0
        skill.execute("booking hold --resource gpu --ttl 300")
        skill._now = 400
        res = skill.execute("booking extend --resource gpu")
        self.assertEqual(res.exit_code, 1)

    def test_expired_hold_can_be_taken_over(self):
        skill, _ = self._fresh({})
        skill._now = 0
        skill.execute("booking hold --resource gpu --holder alice --ttl 300")
        skill._now = 400  # alice's hold expired
        res = skill.execute("booking hold --resource gpu --holder bob")
        self.assertEqual(res.exit_code, 0)
        self.assertEqual(
            res.state_changes["effects"]["booking_held"][0]["holder"], "bob"
        )

    # ---- mutual exclusion ------------------------------------------------- #
    def test_live_hold_blocks_different_holder(self):
        skill, _ = self._fresh({})
        skill.execute("booking hold --resource gpu --holder alice")
        res = skill.execute("booking hold --resource gpu --holder bob")
        self.assertEqual(res.exit_code, 1)
        self.assertIn("conflict", res.stderr)
        self.assertIsNone(res.state_changes)

    def test_release_frees_for_new_holder(self):
        skill, _ = self._fresh({})
        skill.execute("booking hold --resource gpu --holder alice")
        skill.execute("booking release --resource gpu")
        res = skill.execute("booking hold --resource gpu --holder bob")
        self.assertEqual(res.exit_code, 0)

    def test_confirm_wrong_holder_fails(self):
        skill, _ = self._fresh({})
        skill.execute("booking hold --resource gpu --holder alice")
        res = skill.execute("booking confirm --resource gpu --holder bob")
        self.assertEqual(res.exit_code, 1)
        self.assertIsNone(res.state_changes)

    def test_confirm_missing_hold_fails(self):
        skill, _ = self._fresh({})
        res = skill.execute("booking confirm --resource nope")
        self.assertEqual(res.exit_code, 1)
        self.assertIsNone(res.state_changes)

    # ---- required flags --------------------------------------------------- #
    def test_hold_requires_resource(self):
        skill, _ = self._fresh({})
        self.assertEqual(skill.execute("booking hold").exit_code, 1)

    def test_confirm_requires_resource(self):
        skill, _ = self._fresh({})
        self.assertEqual(skill.execute("booking confirm").exit_code, 1)

    # ---- introspection ---------------------------------------------------- #
    def test_status_free_and_held(self):
        skill, _ = self._fresh({})
        self.assertIn("free", skill.execute("booking status --resource gpu").stdout)
        skill.execute("booking hold --resource gpu --holder alice")
        self.assertIn(
            "held by alice", skill.execute("booking status --resource gpu").stdout
        )

    def test_status_reports_expired(self):
        skill, _ = self._fresh({})
        skill._now = 0
        skill.execute("booking hold --resource gpu --ttl 300")
        skill._now = 400
        self.assertIn(
            "expired", skill.execute("booking status --resource gpu").stdout
        )

    def test_search_lists_holds(self):
        skill, _ = self._fresh({})
        skill.execute("booking hold --resource gpu --holder alice")
        self.assertIn("gpu", skill.execute("booking search").stdout)

    # ---- persistence / preload ------------------------------------------- #
    def test_preload_replace_seeded(self):
        skill, _ = self._fresh(
            {},
            preload={"replace_seeded": True, "booking": [
                {"resource": "gpu", "holder": "carol", "status": "held",
                 "created_at": 0, "ttl": None}
            ]},
        )
        # A preloaded (non-expiring) hold by carol blocks a different holder.
        res = skill.execute("booking hold --resource gpu --holder alice")
        self.assertEqual(res.exit_code, 1)

    def test_persist_across_reinitialize(self):
        skill, d = self._fresh({})
        skill.execute("booking hold --resource gpu --holder alice")
        reloaded = BookingSkill()
        reloaded.initialize(d, {})
        self.assertEqual([r["resource"] for r in reloaded._records], ["gpu"])

    # ---- snapshot / undo -------------------------------------------------- #
    def test_snapshot_roundtrip_restores(self):
        skill, _ = self._fresh({})
        skill.execute("booking hold --resource gpu --holder alice")
        snap = skill.to_state()
        skill.execute("booking release --resource gpu")
        self.assertEqual(len(skill._records), 0)
        skill.from_state(snap)
        self.assertEqual([r["resource"] for r in skill._records], ["gpu"])

    def test_get_state_exposes_booking_records(self):
        skill, _ = self._fresh({})
        skill.execute("booking hold --resource gpu --holder alice")
        state = skill.get_state()
        self.assertIn("booking", state)
        self.assertEqual(state["booking"][0]["holder"], "alice")


if __name__ == "__main__":
    unittest.main()
