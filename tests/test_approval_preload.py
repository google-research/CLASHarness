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

"""Unit tests for ApprovalSkill (M4 multi-party access-grant protocol family).

Covers the request/approve lifecycle, the grant gate (every required approver
must have a live, unexpired, scope-reconciled approval), the multi-party
synchronization mechanic (all approvals simultaneously valid against the plumbed
``_now`` clock), scope reconciliation, the deadlock-breaking provisional/escalate
verbs and their effect shapes, revoke, read-only introspection, preload/persist,
and snapshot/undo.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from clasharness.skills.impl.approval_skill import ApprovalSkill


class TestApprovalSkill(unittest.TestCase):
    def _fresh(self, env_vars=None, preload=None):
        d = tempfile.mkdtemp()
        if preload is not None:
            with open(os.path.join(d, "approval.json"), "w", encoding="utf-8") as f:
                json.dump(preload, f)
        skill = ApprovalSkill()
        skill.initialize(d, env_vars or {})
        return skill, d

    def _request_and_approve_all(self, skill, resource="vault",
                                 scope="read", approvers=("sec", "legal")):
        skill.execute(
            f"approval request --resource {resource} --scope {scope} "
            f"--approvers {','.join(approvers)}"
        )
        for role in approvers:
            skill.execute(
                f"approval approve --resource {resource} --approver {role} "
                f"--scope {scope}"
            )

    # ---- grant gate: happy path ------------------------------------------ #
    def test_grant_emits_effect_when_all_valid(self):
        skill, _ = self._fresh({})
        self._request_and_approve_all(skill)
        res = skill.execute("approval grant --resource vault")
        self.assertEqual(res.exit_code, 0)
        self.assertEqual(
            res.state_changes,
            {"effects": {"approval_granted": [
                {"resource": "vault", "scope": "read"}
            ]}},
        )

    def test_grant_marks_request_granted(self):
        skill, _ = self._fresh({})
        self._request_and_approve_all(skill)
        skill.execute("approval grant --resource vault")
        self.assertIn("granted", skill.execute("approval status --resource vault").stdout)

    # ---- grant gate: multi-party requirement ----------------------------- #
    def test_missing_approver_blocks_grant(self):
        skill, _ = self._fresh({})
        skill.execute("approval request --resource db --approvers sec,legal")
        skill.execute("approval approve --resource db --approver sec")
        res = skill.execute("approval grant --resource db")
        self.assertEqual(res.exit_code, 1)
        self.assertIn("missing approval from legal", res.stderr)
        self.assertIsNone(res.state_changes)

    def test_grant_requires_a_request(self):
        skill, _ = self._fresh({})
        res = skill.execute("approval grant --resource ghost")
        self.assertEqual(res.exit_code, 1)
        self.assertIsNone(res.state_changes)

    def test_grant_with_no_named_approvers_needs_one_live_approval(self):
        skill, _ = self._fresh({})
        skill.execute("approval request --resource open")  # no --approvers
        # No approvals recorded yet -> cannot grant.
        self.assertEqual(skill.execute("approval grant --resource open").exit_code, 1)
        skill.execute("approval approve --resource open --approver anyone")
        res = skill.execute("approval grant --resource open")
        self.assertEqual(res.exit_code, 0)
        self.assertIn("approval_granted", res.state_changes["effects"])

    # ---- grant gate: virtual-time synchronization ------------------------ #
    def test_expired_approval_blocks_grant(self):
        skill, _ = self._fresh({})
        skill.execute("approval request --resource gpu --approvers a,b")
        skill._now = 0
        skill.execute("approval approve --resource gpu --approver a --ttl 100")
        skill._now = 200  # a's window has closed (simulates `wait 200`)
        skill.execute("approval approve --resource gpu --approver b --ttl 100")
        res = skill.execute("approval grant --resource gpu")
        self.assertEqual(res.exit_code, 1)
        self.assertIn("expired", res.stderr)
        self.assertIsNone(res.state_changes)

    def test_grant_succeeds_while_all_windows_overlap(self):
        skill, _ = self._fresh({})
        skill.execute("approval request --resource gpu --approvers a,b")
        skill._now = 0
        skill.execute("approval approve --resource gpu --approver a --ttl 300")
        skill._now = 50
        skill.execute("approval approve --resource gpu --approver b --ttl 300")
        skill._now = 200  # both still live (a: 200<300, b: 150<300)
        res = skill.execute("approval grant --resource gpu")
        self.assertEqual(res.exit_code, 0)
        self.assertIn("approval_granted", res.state_changes["effects"])

    def test_renewing_lapsing_approval_restores_grant(self):
        skill, _ = self._fresh({})
        skill.execute("approval request --resource gpu --approvers a,b")
        skill._now = 0
        skill.execute("approval approve --resource gpu --approver a --ttl 100")
        skill._now = 200  # a lapsed
        skill.execute("approval approve --resource gpu --approver b --ttl 300")
        self.assertEqual(skill.execute("approval grant --resource gpu").exit_code, 1)
        # Re-record a's approval so both are live at the same moment.
        skill.execute("approval approve --resource gpu --approver a --ttl 300")
        res = skill.execute("approval grant --resource gpu")
        self.assertEqual(res.exit_code, 0)

    # ---- grant gate: scope reconciliation -------------------------------- #
    def test_scope_mismatch_blocks_grant(self):
        skill, _ = self._fresh({})
        skill.execute("approval request --resource api --scope write --approvers sec")
        skill.execute("approval approve --resource api --approver sec --scope read")
        res = skill.execute("approval grant --resource api")
        self.assertEqual(res.exit_code, 1)
        self.assertIn("reconcile", res.stderr)
        self.assertIsNone(res.state_changes)

    def test_broad_approval_scope_reconciles(self):
        skill, _ = self._fresh({})
        skill.execute("approval request --resource api --scope write --approvers sec")
        skill.execute("approval approve --resource api --approver sec --scope all")
        res = skill.execute("approval grant --resource api")
        self.assertEqual(res.exit_code, 0)
        self.assertIn("approval_granted", res.state_changes["effects"])

    def test_matching_scope_reconciles(self):
        skill, _ = self._fresh({})
        skill.execute("approval request --resource api --scope write --approvers sec")
        skill.execute("approval approve --resource api --approver sec --scope write")
        self.assertEqual(skill.execute("approval grant --resource api").exit_code, 0)

    # ---- provisional / escalate (deadlock-breaking) ---------------------- #
    def test_provisional_counts_as_live_for_grant(self):
        skill, _ = self._fresh({})
        skill.execute("approval request --resource ns --approvers platform,security")
        skill.execute("approval approve --resource ns --approver security")
        skill.execute("approval provisional --resource ns --approver platform")
        res = skill.execute("approval grant --resource ns")
        self.assertEqual(res.exit_code, 0)
        self.assertIn("approval_granted", res.state_changes["effects"])

    def test_escalate_emits_effect(self):
        skill, _ = self._fresh({})
        skill.execute("approval request --resource ns --approvers platform,security")
        res = skill.execute("approval escalate --resource ns --to release_manager")
        self.assertEqual(res.exit_code, 0)
        self.assertEqual(
            res.state_changes,
            {"effects": {"approval_escalated": [
                {"resource": "ns", "to": "release_manager"}
            ]}},
        )

    def test_escalate_requires_to(self):
        skill, _ = self._fresh({})
        skill.execute("approval request --resource ns --approvers a,b")
        res = skill.execute("approval escalate --resource ns")
        self.assertEqual(res.exit_code, 1)
        self.assertIsNone(res.state_changes)

    # ---- revoke ----------------------------------------------------------- #
    def test_revoke_single_approver_blocks_grant(self):
        skill, _ = self._fresh({})
        self._request_and_approve_all(skill)
        skill.execute("approval revoke --resource vault --approver sec")
        res = skill.execute("approval grant --resource vault")
        self.assertEqual(res.exit_code, 1)
        self.assertIn("missing approval from sec", res.stderr)

    def test_revoke_whole_request_removes_records(self):
        skill, _ = self._fresh({})
        self._request_and_approve_all(skill)
        skill.execute("approval revoke --resource vault")
        self.assertEqual(
            [r for r in skill._records if r.get("resource") == "vault"], []
        )

    # ---- required flags --------------------------------------------------- #
    def test_request_requires_resource(self):
        skill, _ = self._fresh({})
        self.assertEqual(skill.execute("approval request").exit_code, 1)

    def test_approve_requires_approver(self):
        skill, _ = self._fresh({})
        skill.execute("approval request --resource r --approvers a")
        self.assertEqual(skill.execute("approval approve --resource r").exit_code, 1)

    def test_unknown_subcommand_fails(self):
        skill, _ = self._fresh({})
        self.assertEqual(skill.execute("approval frobnicate").exit_code, 1)

    # ---- default ttl ------------------------------------------------------ #
    def test_default_ttl_recorded(self):
        skill, _ = self._fresh({})
        skill.execute("approval request --resource r --approvers a")
        skill.execute("approval approve --resource r --approver a")
        appr = skill._approval("r", "a")
        self.assertEqual(appr["ttl_seconds"], 300)

    # ---- introspection ---------------------------------------------------- #
    def test_status_reports_expired(self):
        skill, _ = self._fresh({})
        skill.execute("approval request --resource r --approvers a")
        skill._now = 0
        skill.execute("approval approve --resource r --approver a --ttl 100")
        skill._now = 200
        self.assertIn("expired", skill.execute("approval status --resource r").stdout)

    def test_status_empty(self):
        skill, _ = self._fresh({})
        self.assertIn("No approval records", skill.execute("approval status").stdout)

    # ---- persistence / preload ------------------------------------------- #
    def test_preload_replace_seeded(self):
        skill, _ = self._fresh(
            {},
            preload={"replace_seeded": True, "approval": [
                {"kind": "request", "resource": "vault", "scope": "read",
                 "approvers": ["sec"], "status": "open"},
                {"kind": "approval", "resource": "vault", "approver_role": "sec",
                 "scope": "read", "status": "approved", "granted_at": 0,
                 "ttl_seconds": None},
            ]},
        )
        res = skill.execute("approval grant --resource vault")
        self.assertEqual(res.exit_code, 0)
        self.assertIn("approval_granted", res.state_changes["effects"])

    def test_persist_across_reinitialize(self):
        skill, d = self._fresh({})
        self._request_and_approve_all(skill)
        reloaded = ApprovalSkill()
        reloaded.initialize(d, {})
        self.assertEqual(reloaded.execute("approval grant --resource vault").exit_code, 0)

    # ---- snapshot / undo -------------------------------------------------- #
    def test_snapshot_roundtrip_restores(self):
        skill, _ = self._fresh({})
        self._request_and_approve_all(skill)
        snap = skill.to_state()
        skill.execute("approval revoke --resource vault")
        self.assertEqual(len(skill._records), 0)
        skill.from_state(snap)
        self.assertEqual(skill.execute("approval grant --resource vault").exit_code, 0)

    def test_get_state_exposes_approval_records(self):
        skill, _ = self._fresh({})
        skill.execute("approval request --resource vault --approvers sec")
        state = skill.get_state()
        self.assertIn("approval", state)
        self.assertEqual(state["approval"][0]["resource"], "vault")


if __name__ == "__main__":
    unittest.main()
