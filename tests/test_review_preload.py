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

"""Unit tests for ReviewSkill (M4 non-overlapping review + document sign-gate).

Covers aspect assignment (no effect), the ``review_submitted`` effect shape for
ok/blocked verdicts, blocking-issue tracking and clearing, the document sign
gate (fails while any aspect is pending or blocked, and emits ``document_signed``
only once every aspect is ok), the multi-aspect count property (one
``review_submitted`` per distinct reviewed aspect), non-overlapping assignment
(re-assigning an aspect updates its owner rather than duplicating it), required
flags, read-only status, preload/persist, and snapshot/undo.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from clasharness.skills.impl.review_skill import ReviewSkill


class TestReviewSkill(unittest.TestCase):
    def _fresh(self, env_vars=None, preload=None):
        d = tempfile.mkdtemp()
        if preload is not None:
            with open(os.path.join(d, "review.json"), "w", encoding="utf-8") as f:
                json.dump(preload, f)
        skill = ReviewSkill()
        skill.initialize(d, env_vars or {})
        return skill, d

    def _clear_all(self, skill, aspects=("legal", "privacy", "security", "business")):
        for a in aspects:
            skill.execute(f"review assign --aspect {a} --reviewer r_{a}")
            skill.execute(f"review submit --aspect {a} --status ok")

    # ---- assign: partitions work and emits -------------------------------- #
    def test_assign_emits_effect(self):
        # Assignment is the step where the non-overlapping mechanic is decided
        # (who owns which aspect), so it must be gradable on its own rather
        # than only via the final signature.
        skill, _ = self._fresh({})
        res = skill.execute("review assign --aspect legal --reviewer counsel")
        self.assertEqual(res.exit_code, 0)
        self.assertEqual(
            res.state_changes,
            {
                "effects": {
                    "review_assigned": [
                        {"aspect": "legal", "reviewer": "counsel"}
                    ]
                }
            },
        )

    def test_rejected_assign_emits_nothing(self):
        skill, _ = self._fresh({})
        res = skill.execute("review assign --aspect legal")
        self.assertEqual(res.exit_code, 1)
        self.assertIsNone(res.state_changes)

    def test_assign_requires_aspect_and_reviewer(self):
        skill, _ = self._fresh({})
        self.assertEqual(skill.execute("review assign --aspect legal").exit_code, 1)
        self.assertEqual(skill.execute("review assign --reviewer counsel").exit_code, 1)

    def test_reassign_updates_owner_not_duplicate(self):
        # Non-overlapping: an aspect has exactly one owner; re-assigning swaps
        # the reviewer rather than creating a second record for the same aspect.
        skill, _ = self._fresh({})
        skill.execute("review assign --aspect legal --reviewer counsel")
        skill.execute("review assign --aspect legal --reviewer new_counsel")
        state = skill.get_state()
        legal = [r for r in state["review"] if r.get("aspect") == "legal"]
        self.assertEqual(len(legal), 1)
        self.assertEqual(legal[0]["assignee"], "new_counsel")

    # ---- submit: review_submitted effect shape --------------------------- #
    def test_submit_ok_emits_effect(self):
        skill, _ = self._fresh({})
        res = skill.execute("review submit --aspect legal --status ok")
        self.assertEqual(res.exit_code, 0)
        self.assertEqual(
            res.state_changes,
            {"effects": {"review_submitted": [{"aspect": "legal", "status": "ok"}]}},
        )

    def test_submit_blocked_records_issue(self):
        skill, _ = self._fresh({})
        res = skill.execute(
            "review submit --aspect privacy --status blocked --issue retention"
        )
        self.assertEqual(res.exit_code, 0)
        self.assertEqual(
            res.state_changes["effects"]["review_submitted"][0]["status"], "blocked"
        )
        rec = [r for r in skill.get_state()["review"] if r.get("aspect") == "privacy"][0]
        self.assertEqual(rec["blocking_issues"], ["retention"])

    def test_submit_invalid_status_fails(self):
        skill, _ = self._fresh({})
        res = skill.execute("review submit --aspect legal --status maybe")
        self.assertEqual(res.exit_code, 1)
        self.assertIsNone(res.state_changes)

    def test_submit_requires_aspect(self):
        skill, _ = self._fresh({})
        self.assertEqual(skill.execute("review submit --status ok").exit_code, 1)

    def test_resubmit_ok_clears_prior_block(self):
        skill, _ = self._fresh({})
        skill.execute("review submit --aspect privacy --status blocked --issue retention")
        skill.execute("review submit --aspect privacy --status ok")
        rec = [r for r in skill.get_state()["review"] if r.get("aspect") == "privacy"][0]
        self.assertEqual(rec["status"], "ok")
        self.assertEqual(rec["blocking_issues"], [])

    def test_multiple_aspects_emit_one_effect_each(self):
        skill, _ = self._fresh({})
        emitted = 0
        for a in ("legal", "privacy", "security", "business"):
            res = skill.execute(f"review submit --aspect {a} --status ok")
            emitted += len(res.state_changes["effects"]["review_submitted"])
        self.assertEqual(emitted, 4)

    # ---- sign gate ------------------------------------------------------- #
    def test_sign_without_aspects_fails(self):
        skill, _ = self._fresh({})
        res = skill.execute("review sign --document contract")
        self.assertEqual(res.exit_code, 1)
        self.assertIsNone(res.state_changes)

    def test_sign_with_pending_aspect_fails(self):
        skill, _ = self._fresh({})
        skill.execute("review assign --aspect legal --reviewer counsel")  # pending
        res = skill.execute("review sign --document contract")
        self.assertEqual(res.exit_code, 1)
        self.assertIsNone(res.state_changes)

    def test_sign_with_blocked_aspect_fails(self):
        skill, _ = self._fresh({})
        skill.execute("review submit --aspect legal --status ok")
        skill.execute("review submit --aspect privacy --status blocked --issue x")
        res = skill.execute("review sign --document contract")
        self.assertEqual(res.exit_code, 1)
        self.assertIsNone(res.state_changes)

    def test_sign_after_all_ok_emits_document_signed(self):
        skill, _ = self._fresh({})
        self._clear_all(skill)
        res = skill.execute("review sign --document contract")
        self.assertEqual(res.exit_code, 0)
        self.assertEqual(
            res.state_changes,
            {"effects": {"document_signed": [{"document": "contract"}]}},
        )

    def test_sign_succeeds_after_block_resolved(self):
        skill, _ = self._fresh({})
        skill.execute("review submit --aspect legal --status ok")
        skill.execute("review submit --aspect privacy --status blocked --issue x")
        self.assertEqual(skill.execute("review sign --document c").exit_code, 1)
        # Resolve the block, then the gate opens.
        skill.execute("review submit --aspect privacy --status ok")
        self.assertEqual(skill.execute("review sign --document c").exit_code, 0)

    def test_sign_requires_document(self):
        skill, _ = self._fresh({})
        self._clear_all(skill)
        self.assertEqual(skill.execute("review sign").exit_code, 1)

    def test_unknown_subcommand_fails(self):
        skill, _ = self._fresh({})
        self.assertEqual(skill.execute("review approve --all").exit_code, 1)

    # ---- status ---------------------------------------------------------- #
    def test_status_reports_aspects_and_blockers(self):
        skill, _ = self._fresh({})
        skill.execute("review assign --aspect legal --reviewer counsel")
        skill.execute("review submit --aspect privacy --status blocked --issue retention")
        out = skill.execute("review status").stdout
        self.assertIn("legal", out)
        self.assertIn("privacy", out)
        self.assertIn("retention", out)

    def test_status_empty(self):
        skill, _ = self._fresh({})
        self.assertIn("No review", skill.execute("review status").stdout)

    def _two_signed_docs(self):
        skill, _ = self._fresh({})
        self._clear_all(skill)
        skill.execute("review submit --aspect legal --status ok")
        skill.execute("review sign --document RB-9921-PARTNERSHIP")
        skill.execute("review sign --document VENDOR-SOC2")
        return skill

    def test_status_document_scopes_to_the_requested_document(self):
        """``--document`` is advertised as a filter and used to be ignored."""
        skill = self._two_signed_docs()
        out = skill.execute("review status --document RB-9921-PARTNERSHIP").stdout
        self.assertIn("RB-9921-PARTNERSHIP", out)
        self.assertNotIn("VENDOR-SOC2", out)

    def test_status_document_matches_a_looser_spelling(self):
        skill = self._two_signed_docs()
        out = skill.execute("review status --document rb-9921").stdout
        self.assertIn("RB-9921-PARTNERSHIP", out)
        self.assertNotIn("VENDOR-SOC2", out)

    def test_status_document_unresolved_falls_back_to_all(self):
        """An unmatched name must never hide a signature that was visible."""
        skill = self._two_signed_docs()
        out = skill.execute("review status --document 'some other contract'").stdout
        self.assertIn("RB-9921-PARTNERSHIP", out)
        self.assertIn("VENDOR-SOC2", out)
        self.assertIn("no document record matching", out)

    def test_status_document_still_lists_aspects(self):
        """Aspects gate signing any document, so they are never filtered out."""
        skill, _ = self._fresh({})
        self._clear_all(skill)
        skill.execute("review assign --aspect legal --reviewer counsel")
        out = skill.execute("review status --document anything").stdout
        self.assertIn("legal", out)

    # ---- persistence / preload ------------------------------------------- #
    def test_preload_aspect_gates_sign(self):
        skill, _ = self._fresh(
            {},
            preload={"replace_seeded": True, "review": [
                {"kind": "aspect", "aspect": "legal", "assignee": "counsel",
                 "status": "blocked", "blocking_issues": ["open item"]},
            ]},
        )
        # Preloaded blocked aspect keeps the gate shut.
        self.assertEqual(skill.execute("review sign --document c").exit_code, 1)
        skill.execute("review submit --aspect legal --status ok")
        self.assertEqual(skill.execute("review sign --document c").exit_code, 0)

    def test_persist_across_reinitialize(self):
        skill, d = self._fresh({})
        skill.execute("review submit --aspect legal --status ok")
        reloaded = ReviewSkill()
        reloaded.initialize(d, {})
        rec = [r for r in reloaded.get_state()["review"] if r.get("aspect") == "legal"]
        self.assertEqual(rec[0]["status"], "ok")

    # ---- snapshot / undo -------------------------------------------------- #
    def test_snapshot_roundtrip_restores(self):
        skill, _ = self._fresh({})
        skill.execute("review submit --aspect legal --status ok")
        snap = skill.to_state()
        skill.execute("review submit --aspect privacy --status ok")
        self.assertEqual(len(skill.get_state()["review"]), 2)
        skill.from_state(snap)
        self.assertEqual(len(skill.get_state()["review"]), 1)

    def test_get_state_exposes_review_records(self):
        skill, _ = self._fresh({})
        skill.execute("review submit --aspect legal --status ok")
        state = skill.get_state()
        self.assertIn("review", state)
        self.assertEqual(state["review"][0]["aspect"], "legal")


if __name__ == "__main__":
    unittest.main()
