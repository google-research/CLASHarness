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

"""Unit tests for PairingSkill (M3 multi-device-pairing protocol family).

Covers the ordered handshake FSM (start -> show-code -> enter-code -> confirm),
out-of-order rejection, effect emission shape, the virtual-time session window
(each step against the plumbed ``_now`` clock), pairing-code match/mismatch,
read-only introspection, preload/persist, and snapshot/undo.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from clasharness.skills.impl.pairing_skill import PairingSkill, _derive_code


class TestPairingSkill(unittest.TestCase):
    def _fresh(self, env_vars=None, preload=None):
        d = tempfile.mkdtemp()
        if preload is not None:
            with open(os.path.join(d, "pairing.json"), "w", encoding="utf-8") as f:
                json.dump(preload, f)
        skill = PairingSkill()
        skill.initialize(d, env_vars or {})
        return skill, d

    def _walk_happy(self, skill, device="phone-1"):
        """Drive the full ordered handshake and return the confirm result."""
        skill.execute(f"pairing start --device {device}")
        skill.execute(f"pairing show-code --device {device}")
        skill.execute(f"pairing enter-code --device {device}")
        return skill.execute(f"pairing confirm --device {device}")

    # ---- ordered handshake / happy path ----------------------------------- #
    def test_full_handshake_emits_completed(self):
        skill, _ = self._fresh({})
        res = self._walk_happy(skill, "phone-1")
        self.assertEqual(res.exit_code, 0)
        self.assertEqual(
            res.state_changes,
            {"effects": {"pairing_completed": [{"device": "phone-1"}]}},
        )

    def test_every_protocol_step_emits(self):
        # start -> code_shown -> entered -> completed are all real transitions
        # of the pairing FSM, so each one emits. Only ``confirm`` used to emit,
        # which left a round that merely advanced the protocol ungradable.
        skill, _ = self._fresh({})

        started = skill.execute("pairing start --device p")
        self.assertEqual(started.exit_code, 0)
        self.assertEqual(
            list(started.state_changes["effects"]), ["pairing_started"]
        )
        self.assertEqual(
            started.state_changes["effects"]["pairing_started"][0]["device"], "p"
        )

        shown = skill.execute("pairing show-code --device p")
        self.assertEqual(shown.exit_code, 0)
        shown_payload = shown.state_changes["effects"]["pairing_code_shown"][0]
        self.assertEqual(shown_payload["device"], "p")
        code = shown_payload["code"]
        self.assertTrue(code)

        entered = skill.execute("pairing enter-code --device p")
        self.assertEqual(entered.exit_code, 0)
        entered_payload = entered.state_changes["effects"][
            "pairing_code_entered"
        ][0]
        self.assertEqual(entered_payload["device"], "p")
        self.assertEqual(entered_payload["code"], code)

    def test_status_is_read_only(self):
        skill, _ = self._fresh({})
        skill.execute("pairing start --device p")
        res = skill.execute("pairing status")
        self.assertEqual(res.exit_code, 0)
        self.assertIsNone(res.state_changes)

    # ---- out-of-order rejection ------------------------------------------- #
    def test_show_code_before_start_fails(self):
        skill, _ = self._fresh({})
        res = skill.execute("pairing show-code --device p")
        self.assertEqual(res.exit_code, 1)
        self.assertIsNone(res.state_changes)

    def test_enter_code_before_show_fails(self):
        skill, _ = self._fresh({})
        skill.execute("pairing start --device p")
        res = skill.execute("pairing enter-code --device p")
        self.assertEqual(res.exit_code, 1)
        self.assertIsNone(res.state_changes)

    def test_confirm_before_enter_fails_without_effect(self):
        skill, _ = self._fresh({})
        skill.execute("pairing start --device p")
        skill.execute("pairing show-code --device p")
        res = skill.execute("pairing confirm --device p")
        self.assertEqual(res.exit_code, 1)
        self.assertIsNone(res.state_changes)

    def test_confirm_directly_after_start_fails(self):
        skill, _ = self._fresh({})
        skill.execute("pairing start --device p")
        res = skill.execute("pairing confirm --device p")
        self.assertEqual(res.exit_code, 1)
        self.assertIsNone(res.state_changes)

    # ---- pairing code match / mismatch ------------------------------------ #
    def test_show_code_returns_derived_code(self):
        skill, _ = self._fresh({})
        skill.execute("pairing start --device phone-1")
        res = skill.execute("pairing show-code --device phone-1")
        self.assertEqual(res.exit_code, 0)
        self.assertIn(_derive_code("phone-1"), res.stdout)

    def test_enter_code_matching_succeeds(self):
        skill, _ = self._fresh({})
        skill.execute("pairing start --device phone-1")
        skill.execute("pairing show-code --device phone-1")
        code = _derive_code("phone-1")
        res = skill.execute(f"pairing enter-code --device phone-1 --code {code}")
        self.assertEqual(res.exit_code, 0)

    def test_enter_code_mismatch_fails(self):
        skill, _ = self._fresh({})
        skill.execute("pairing start --device phone-1")
        skill.execute("pairing show-code --device phone-1")
        res = skill.execute("pairing enter-code --device phone-1 --code 000000")
        self.assertEqual(res.exit_code, 1)
        self.assertIsNone(res.state_changes)

    # ---- virtual-time session window -------------------------------------- #
    def test_created_at_stamps_virtual_clock(self):
        skill, _ = self._fresh({})
        skill._now = 42
        skill.execute("pairing start --device p")
        self.assertEqual(skill._records[0]["created_at"], 42)

    def test_custom_ttl_is_recorded(self):
        skill, _ = self._fresh({})
        skill.execute("pairing start --device p --ttl 120")
        self.assertEqual(skill._records[0]["ttl"], 120)

    def test_full_handshake_within_window_succeeds(self):
        skill, _ = self._fresh({})
        skill._now = 0
        skill.execute("pairing start --device p --ttl 300")
        skill.execute("pairing show-code --device p")
        skill.execute("pairing enter-code --device p")
        skill._now = 299  # still inside the window
        res = skill.execute("pairing confirm --device p")
        self.assertEqual(res.exit_code, 0)
        self.assertIn("pairing_completed", res.state_changes["effects"])

    def test_confirm_after_expiry_fails_without_effect(self):
        skill, _ = self._fresh({})
        skill._now = 0
        skill.execute("pairing start --device p --ttl 300")
        skill.execute("pairing show-code --device p")
        skill.execute("pairing enter-code --device p")
        skill._now = 400  # window has closed (simulates `wait 400`)
        res = skill.execute("pairing confirm --device p")
        self.assertEqual(res.exit_code, 1)
        self.assertIn("expired", res.stderr)
        self.assertIsNone(res.state_changes)

    def test_show_code_after_expiry_fails(self):
        skill, _ = self._fresh({})
        skill._now = 0
        skill.execute("pairing start --device p --ttl 300")
        skill._now = 400
        res = skill.execute("pairing show-code --device p")
        self.assertEqual(res.exit_code, 1)
        self.assertIsNone(res.state_changes)

    def test_enter_code_after_expiry_fails(self):
        skill, _ = self._fresh({})
        skill._now = 0
        skill.execute("pairing start --device p --ttl 300")
        skill.execute("pairing show-code --device p")
        skill._now = 400
        res = skill.execute("pairing enter-code --device p")
        self.assertEqual(res.exit_code, 1)
        self.assertIsNone(res.state_changes)

    def test_restart_after_expiry_allows_new_session(self):
        skill, _ = self._fresh({})
        skill._now = 0
        skill.execute("pairing start --device p --ttl 300")
        skill._now = 400  # first session expired
        skill.execute("pairing start --device p --ttl 300")  # restart
        skill.execute("pairing show-code --device p")
        skill.execute("pairing enter-code --device p")
        res = skill.execute("pairing confirm --device p")
        self.assertEqual(res.exit_code, 0)
        self.assertIn("pairing_completed", res.state_changes["effects"])

    # ---- required flags --------------------------------------------------- #
    def test_start_requires_device(self):
        skill, _ = self._fresh({})
        self.assertEqual(skill.execute("pairing start").exit_code, 1)

    def test_confirm_requires_device(self):
        skill, _ = self._fresh({})
        self.assertEqual(skill.execute("pairing confirm").exit_code, 1)

    def test_unknown_subcommand_fails(self):
        skill, _ = self._fresh({})
        self.assertEqual(skill.execute("pairing frobnicate").exit_code, 1)

    # ---- introspection ---------------------------------------------------- #
    def test_status_unpaired_and_states(self):
        skill, _ = self._fresh({})
        self.assertIn("unpaired", skill.execute("pairing status --device p").stdout)
        skill.execute("pairing start --device p")
        self.assertIn("started", skill.execute("pairing status --device p").stdout)

    def test_status_reports_confirmed(self):
        skill, _ = self._fresh({})
        self._walk_happy(skill, "p")
        self.assertIn("confirmed", skill.execute("pairing status --device p").stdout)

    def test_status_reports_expired(self):
        skill, _ = self._fresh({})
        skill._now = 0
        skill.execute("pairing start --device p --ttl 300")
        skill._now = 400
        self.assertIn("expired", skill.execute("pairing status --device p").stdout)

    def test_status_lists_all_sessions(self):
        skill, _ = self._fresh({})
        skill.execute("pairing start --device a")
        skill.execute("pairing start --device b")
        out = skill.execute("pairing status").stdout
        self.assertIn("a", out)
        self.assertIn("b", out)

    # ---- persistence / preload ------------------------------------------- #
    def test_preload_replace_seeded(self):
        skill, _ = self._fresh(
            {},
            preload={"replace_seeded": True, "pairing": [
                {"device": "p", "state": "entered", "created_at": 0, "ttl": None,
                 "code": "123456"}
            ]},
        )
        # A preloaded session already at 'entered' can be confirmed directly.
        res = skill.execute("pairing confirm --device p")
        self.assertEqual(res.exit_code, 0)
        self.assertIn("pairing_completed", res.state_changes["effects"])

    def test_persist_across_reinitialize(self):
        skill, d = self._fresh({})
        skill.execute("pairing start --device p")
        reloaded = PairingSkill()
        reloaded.initialize(d, {})
        self.assertEqual([r["device"] for r in reloaded._records], ["p"])

    # ---- snapshot / undo -------------------------------------------------- #
    def test_snapshot_roundtrip_restores(self):
        skill, _ = self._fresh({})
        skill.execute("pairing start --device p")
        snap = skill.to_state()
        skill.execute("pairing start --device q")
        self.assertEqual(len(skill._records), 2)
        skill.from_state(snap)
        self.assertEqual([r["device"] for r in skill._records], ["p"])

    def test_get_state_exposes_pairing_records(self):
        skill, _ = self._fresh({})
        skill.execute("pairing start --device p")
        state = skill.get_state()
        self.assertIn("pairing", state)
        self.assertEqual(state["pairing"][0]["device"], "p")


if __name__ == "__main__":
    unittest.main()
