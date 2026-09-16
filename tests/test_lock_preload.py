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

"""Unit tests for LockSkill (M1 distributed-lock protocol family).

Covers holder resolution, mutual-exclusion conflict detection, effect emission
shape, preload/persist, snapshot/undo, and read-only introspection.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from clasharness.skills.impl.lock_skill import LockSkill


class TestLockSkill(unittest.TestCase):
    def _fresh(self, env_vars=None, preload=None):
        d = tempfile.mkdtemp()
        if preload is not None:
            with open(os.path.join(d, "lock.json"), "w", encoding="utf-8") as f:
                json.dump(preload, f)
        skill = LockSkill()
        skill.initialize(d, env_vars or {})
        return skill, d

    # ---- holder resolution ------------------------------------------------ #
    def test_holder_defaults_to_env_actor(self):
        skill, _ = self._fresh({"CLASHARNESS_ACTOR": "alice"})
        res = skill.execute("lock acquire --resource db")
        self.assertEqual(res.exit_code, 0)
        self.assertEqual(
            res.state_changes,
            {"effects": {"lock_acquired": [{"resource": "db", "section": "", "holder": "alice"}]}},
        )

    def test_holder_defaults_to_coordinator(self):
        skill, _ = self._fresh({})
        res = skill.execute("lock acquire --resource db")
        self.assertEqual(
            res.state_changes["effects"]["lock_acquired"][0]["holder"], "coordinator"
        )

    def test_explicit_holder_overrides_env(self):
        skill, _ = self._fresh({"CLASHARNESS_ACTOR": "alice"})
        res = skill.execute("lock acquire --resource db --holder bob")
        self.assertEqual(
            res.state_changes["effects"]["lock_acquired"][0]["holder"], "bob"
        )

    # ---- effect shape ----------------------------------------------------- #
    def test_release_emits_effect(self):
        skill, _ = self._fresh({})
        skill.execute("lock acquire --resource db --holder alice")
        res = skill.execute("lock release --resource db")
        self.assertEqual(res.exit_code, 0)
        self.assertEqual(
            res.state_changes,
            {"effects": {"lock_released": [{"resource": "db", "section": ""}]}},
        )

    def test_section_is_part_of_identity(self):
        skill, _ = self._fresh({})
        # Different sections of the same resource are independent locks.
        r1 = skill.execute("lock acquire --resource db --section a --holder alice")
        r2 = skill.execute("lock acquire --resource db --section b --holder bob")
        self.assertEqual(r1.exit_code, 0)
        self.assertEqual(r2.exit_code, 0)

    # ---- mutual exclusion ------------------------------------------------- #
    def test_conflict_returns_exit_1(self):
        skill, _ = self._fresh({})
        skill.execute("lock acquire --resource db --holder alice")
        res = skill.execute("lock acquire --resource db --holder bob")
        self.assertEqual(res.exit_code, 1)
        self.assertIn("conflict", res.stderr)
        self.assertIsNone(res.state_changes)

    def test_idempotent_reacquire_same_holder(self):
        skill, _ = self._fresh({})
        skill.execute("lock acquire --resource db --holder alice")
        res = skill.execute("lock acquire --resource db --holder alice")
        self.assertEqual(res.exit_code, 0)
        # No duplicate record after idempotent re-acquire.
        self.assertEqual(len(skill._records), 1)

    def test_release_frees_for_new_holder(self):
        skill, _ = self._fresh({})
        skill.execute("lock acquire --resource db --holder alice")
        skill.execute("lock release --resource db")
        res = skill.execute("lock acquire --resource db --holder bob")
        self.assertEqual(res.exit_code, 0)

    def test_acquire_requires_resource(self):
        skill, _ = self._fresh({})
        res = skill.execute("lock acquire")
        self.assertEqual(res.exit_code, 1)

    # ---- introspection ---------------------------------------------------- #
    def test_status_free_and_held(self):
        skill, _ = self._fresh({})
        self.assertIn("free", skill.execute("lock status --resource db").stdout)
        skill.execute("lock acquire --resource db --holder alice")
        self.assertIn("held by alice", skill.execute("lock status --resource db").stdout)

    def test_list_reports_held(self):
        skill, _ = self._fresh({})
        skill.execute("lock acquire --resource db --holder alice")
        self.assertIn("db", skill.execute("lock list").stdout)

    # ---- persistence / preload ------------------------------------------- #
    def test_preload_replace_seeded(self):
        skill, _ = self._fresh(
            {},
            preload={"replace_seeded": True, "lock": [
                {"resource": "db", "section": "", "holder": "carol", "created_at": 0}
            ]},
        )
        # A preloaded lock held by carol blocks a different holder.
        res = skill.execute("lock acquire --resource db --holder alice")
        self.assertEqual(res.exit_code, 1)

    def test_persist_across_reinitialize(self):
        skill, d = self._fresh({})
        skill.execute("lock acquire --resource db --holder alice")
        reloaded = LockSkill()
        reloaded.initialize(d, {})
        self.assertEqual([r["resource"] for r in reloaded._records], ["db"])

    # ---- snapshot / undo -------------------------------------------------- #
    def test_snapshot_roundtrip_restores(self):
        skill, _ = self._fresh({})
        skill.execute("lock acquire --resource db --holder alice")
        snap = skill.to_state()
        skill.execute("lock release --resource db")
        self.assertEqual(len(skill._records), 0)
        skill.from_state(snap)
        self.assertEqual([r["resource"] for r in skill._records], ["db"])

    def test_get_state_exposes_lock_records(self):
        skill, _ = self._fresh({})
        skill.execute("lock acquire --resource db --holder alice")
        state = skill.get_state()
        self.assertIn("lock", state)
        self.assertEqual(state["lock"][0]["holder"], "alice")


if __name__ == "__main__":
    unittest.main()
