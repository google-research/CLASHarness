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

"""Unit tests for the StatefulProtocolSkill base (M0 foundations).

The base ships with no concrete subclass yet, so these tests exercise it through
a trivial in-test subclass covering the three building blocks it provides:
preload/persist, effect emission, and virtual-time expiry + snapshot/undo.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from clasharness.backend.base import CommandResult
from clasharness.skills.stateful import StatefulProtocolSkill


class _DemoSkill(StatefulProtocolSkill):
    """Minimal protocol skill: ``demo add <id>`` appends a record + emits an effect."""

    def __init__(self) -> None:
        super().__init__(prefixes=("demo",), family="demo")

    def _default_records(self):
        return [{"id": "seed", "created_at": 0}]

    def execute(self, command: str) -> CommandResult:
        parts = command.split()
        if len(parts) >= 3 and parts[1] == "add":
            record = {"id": parts[2], "created_at": self._advance_tick()}
            self._records.append(record)
            self._save()
            return CommandResult(
                stdout=f"added {parts[2]}",
                stderr="",
                exit_code=0,
                state_changes=self._emit("demo_added", record),
            )
        return CommandResult(stdout="", stderr="unknown", exit_code=1)


class TestStatefulProtocolSkill(unittest.TestCase):
    def test_requires_family(self):
        with self.assertRaises(ValueError):

            class _Bad(StatefulProtocolSkill):
                def __init__(self):
                    super().__init__(prefixes=("x",), family="")

                def execute(self, command):
                    return CommandResult(stdout="", stderr="", exit_code=0)

            _Bad()

    def test_seed_records_present_without_preload(self):
        with tempfile.TemporaryDirectory() as d:
            skill = _DemoSkill()
            skill.initialize(d, {})
            self.assertEqual([r["id"] for r in skill._records], ["seed"])

    def test_preload_list_extends_seeds(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "demo.json"), "w", encoding="utf-8") as f:
                json.dump([{"id": "pre1", "created_at": 0}], f)
            skill = _DemoSkill()
            skill.initialize(d, {})
            self.assertEqual([r["id"] for r in skill._records], ["seed", "pre1"])

    def test_preload_dict_replace_seeded(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "demo.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {"replace_seeded": True, "demo": [{"id": "only", "created_at": 0}]},
                    f,
                )
            skill = _DemoSkill()
            skill.initialize(d, {})
            self.assertEqual([r["id"] for r in skill._records], ["only"])

    def test_emit_shape_single(self):
        payload = {"id": "L1"}
        self.assertEqual(
            StatefulProtocolSkill._emit("lock_acquired", payload),
            {"effects": {"lock_acquired": [payload]}},
        )

    def test_emit_batch_groups_by_type(self):
        out = StatefulProtocolSkill._emit_batch(
            [
                ("lock_acquired", {"id": "L1"}),
                ("lock_acquired", {"id": "L2"}),
                ("deploy_started", {"id": "D1"}),
            ]
        )
        self.assertEqual(
            out,
            {
                "effects": {
                    "lock_acquired": [{"id": "L1"}, {"id": "L2"}],
                    "deploy_started": [{"id": "D1"}],
                }
            },
        )

    def test_execute_emits_effect_and_persists(self):
        with tempfile.TemporaryDirectory() as d:
            skill = _DemoSkill()
            skill.initialize(d, {})
            res = skill.execute("demo add L1")
            self.assertEqual(res.exit_code, 0)
            self.assertEqual(res.state_changes, {"effects": {"demo_added": [{"id": "L1", "created_at": 1}]}})

            # A fresh skill pointed at the same state_dir loads the persisted row.
            reloaded = _DemoSkill()
            reloaded.initialize(d, {})
            self.assertIn("L1", [r["id"] for r in reloaded._records])

    def test_is_expired(self):
        self.assertFalse(StatefulProtocolSkill._is_expired(now=5, created_at=0, ttl=None))
        self.assertFalse(StatefulProtocolSkill._is_expired(now=4, created_at=0, ttl=5))
        self.assertTrue(StatefulProtocolSkill._is_expired(now=5, created_at=0, ttl=5))
        self.assertTrue(StatefulProtocolSkill._is_expired(now=12, created_at=3, ttl=5))

    def test_now_defaults_zero_and_expired_uses_plumbed_clock(self):
        skill = _DemoSkill()
        # Deterministic default before the environment publishes a clock.
        self.assertEqual(skill._now, 0)
        self.assertFalse(skill._expired(created_at=0, ttl=5))
        # The runtime pushes the environment's virtual time onto ``_now``.
        skill._now = 7
        self.assertTrue(skill._expired(created_at=0, ttl=5))
        self.assertFalse(skill._expired(created_at=5, ttl=5))  # 7-5 < 5
        self.assertFalse(skill._expired(created_at=0, ttl=None))  # never expires

    def test_now_is_not_snapshotted(self):
        # ``_now`` is a per-dispatch clock reading, not episode state: undo must
        # not rewind it (the environment keeps virtual time moving after undo).
        self.assertNotIn("_now", StatefulProtocolSkill._snapshot_attrs)
        skill = _DemoSkill()
        skill._now = 99
        self.assertNotIn("_now", skill.to_state())

    def test_snapshot_roundtrip_restores_records_and_clock(self):
        with tempfile.TemporaryDirectory() as d:
            skill = _DemoSkill()
            skill.initialize(d, {})
            skill.execute("demo add L1")
            snap = skill.to_state()
            self.assertEqual(skill._logical_tick, 1)

            skill.execute("demo add L2")
            self.assertEqual(skill._logical_tick, 2)
            self.assertEqual(len(skill._records), 3)  # seed + L1 + L2

            skill.from_state(snap)
            self.assertEqual(skill._logical_tick, 1)
            self.assertEqual([r["id"] for r in skill._records], ["seed", "L1"])

    def test_get_state_exposes_records(self):
        with tempfile.TemporaryDirectory() as d:
            skill = _DemoSkill()
            skill.initialize(d, {})
            skill.execute("demo add L1")
            state = skill.get_state()
            self.assertIn("demo", state)
            self.assertEqual([r["id"] for r in state["demo"]], ["seed", "L1"])


if __name__ == "__main__":
    unittest.main()
