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

"""Access-grant / multi-party-approval protocol skill (M4).

The ``approval`` family models an access grant that becomes valid only once
**every required approver** has approved, each approval is **still within its
time window**, and the approved scopes **reconcile** with what was requested.
This is the mechanic the ``scope_reconciliation`` (#2) and
``multi_party_synchronization`` (#3) scenarios grade, and it also carries the
deadlock-breaking ``provisional`` / ``escalate`` verbs the
``deadlock_dependency`` (#14) scenarios exercise.

Verbs::

    approval request     --resource R --scope S --approvers roleA,roleB
    approval approve     --resource R --approver role [--scope S] [--ttl N]
    approval grant       --resource R          # -> approval_granted (all valid)
    approval revoke      --resource R [--approver role]
    approval provisional --resource R --approver role   # deadlock-breaking (#14)
    approval escalate    --resource R --to role         # -> approval_escalated (#14)
    approval status      [--resource R]                 # read-only

``grant`` is the gate: it fails with exit 1 (and emits **no**
``approval_granted`` effect) unless, for the resource's request, every required
approver has a live (``approved`` or ``provisional``) approval whose window has
not closed and whose scope reconciles with the requested scope. Because the
window is checked against the plumbed virtual clock
(:attr:`~StatefulProtocolSkill._now`), a set of approvals must be *simultaneously*
valid at grant time: if the agent waits (`wait N`) until one approval expires,
the grant fails — the ``multi_party_synchronization`` (#3) mechanic. ``escalate``
emits ``approval_escalated`` so a circular approval deadlock can be broken by
routing to a higher authority. Records persist to ``<state_dir>/approval.json``
via the base and are snapshotted for ``undo``.
"""

from __future__ import annotations

import shlex
from typing import Any

from clasharness.backend.base import CommandResult
from clasharness.skills.spec import (
    CommandSpec,
    EffectPrecondition,
    RealKind,
    SkillSpec,
)
from clasharness.skills.stateful import StatefulProtocolSkill

#: Default approval lifetime in virtual seconds when ``--ttl`` is omitted.
_DEFAULT_TTL = 300
#: Approval statuses that count as a live "yes" when computing a grant.
_LIVE_STATUSES = ("approved", "provisional")
#: Scope values that count as broad enough to cover any requested scope.
_BROAD_SCOPES = ("", "all", "*")


class ApprovalSkill(StatefulProtocolSkill):
    """Multi-party access grants gated on unexpired, scope-reconciled approvals."""

    SPEC = SkillSpec(
        name="approval",
        category="coordination",
        summary=(
            "Multi-party access grants that require every approver to approve, "
            "within a time window and with reconciling scopes."
        ),
        prefixes=("approval",),
        real_kind=RealKind.LOCAL_PERSISTENT,
        commands=(
            CommandSpec(
                "approval request",
                "Open an access request naming the required approvers and scope.",
                args={
                    "--resource": {"type": "string"},
                    "--scope": {"type": "string"},
                    "--approvers": {"type": "string"},
                },
                required=("--resource",),
                emits_effects=("approval_requested",),
            ),
            CommandSpec(
                "approval approve",
                "Record one approver's approval (optionally scoped, time-limited).",
                args={
                    "--resource": {"type": "string"},
                    "--approver": {"type": "string"},
                    "--scope": {"type": "string"},
                    "--ttl": {"type": "integer"},
                },
                required=("--resource", "--approver"),
                emits_effects=("approval_recorded",),
            ),
            CommandSpec(
                "approval grant",
                "Grant access once every required approval is valid and reconciled.",
                args={"--resource": {"type": "string"}},
                required=("--resource",),
                emits_effects=("approval_granted",),
                requires=(
                    EffectPrecondition(
                        chain=("approval request", "approval approve"),
                        identifier_arg="--resource",
                    ),
                ),
            ),
            CommandSpec(
                "approval revoke",
                "Revoke a resource's request, or one approver's approval.",
                args={
                    "--resource": {"type": "string"},
                    "--approver": {"type": "string"},
                },
                required=("--resource",),
                emits_effects=("approval_revoked",),
            ),
            CommandSpec(
                "approval provisional",
                "Grant a provisional (deadlock-breaking) approval for an approver.",
                args={
                    "--resource": {"type": "string"},
                    "--approver": {"type": "string"},
                    "--scope": {"type": "string"},
                    "--ttl": {"type": "integer"},
                },
                required=("--resource", "--approver"),
                emits_effects=("approval_recorded",),
            ),
            CommandSpec(
                "approval escalate",
                "Escalate a stuck approval to a higher authority.",
                args={
                    "--resource": {"type": "string"},
                    "--to": {"type": "string"},
                },
                required=("--resource", "--to"),
                emits_effects=("approval_escalated",),
            ),
            CommandSpec(
                "approval status",
                "Show requests and approvals (read-only).",
                args={"--resource": {"type": "string"}},
            ),
        ),
        smoke=(
            "approval request --resource vault --scope read --approvers sec,legal",
            "approval approve --resource vault --approver sec --scope read",
            "approval approve --resource vault --approver legal --scope read",
            "approval grant --resource vault",
            "approval status",
        ),
    )

    def __init__(self) -> None:
        super().__init__(prefixes=("approval",), family="approval")

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    def _request(self, resource: str) -> dict[str, Any] | None:
        for rec in self._records:
            if rec.get("kind") == "request" and rec.get("resource") == resource:
                return rec
        return None

    def _approval(self, resource: str, approver: str) -> dict[str, Any] | None:
        for rec in self._records:
            if (
                rec.get("kind") == "approval"
                and rec.get("resource") == resource
                and rec.get("approver_role") == approver
            ):
                return rec
        return None

    def _approvals(self, resource: str) -> list[dict[str, Any]]:
        return [
            r
            for r in self._records
            if r.get("kind") == "approval" and r.get("resource") == resource
        ]

    def _rec_expired(self, rec: dict[str, Any]) -> bool:
        return self._expired(int(rec.get("granted_at", 0)), rec.get("ttl_seconds"))

    @staticmethod
    def _scope_reconciles(requested: str, approved: str) -> bool:
        """True iff an ``approved`` scope is broad enough for ``requested``."""
        if requested in _BROAD_SCOPES:
            return True
        if approved in _BROAD_SCOPES:
            return True
        return approved == requested

    @staticmethod
    def _parse_ttl(raw: str | None, default: int | None) -> int | None:
        if raw is None:
            return default
        try:
            return int(raw)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _parse_roles(raw: str | None) -> list[str]:
        if not raw:
            return []
        return [r.strip() for r in raw.split(",") if r.strip()]

    @staticmethod
    def _err(msg: str) -> CommandResult:
        return CommandResult(stdout="", stderr=msg, exit_code=1)

    def _record_approval(
        self, args: list[str], status: str
    ) -> CommandResult:
        """Shared body for ``approve`` and ``provisional``."""
        resource = _get_arg(args, "--resource")
        approver = _get_arg(args, "--approver")
        if not resource:
            return self._err("--resource is required")
        if not approver:
            return self._err("--approver is required")
        req = self._request(resource)
        default_scope = req.get("scope", "") if req else ""
        scope = _get_arg(args, "--scope")
        if scope is None:
            scope = default_scope
        ttl = self._parse_ttl(_get_arg(args, "--ttl"), _DEFAULT_TTL)

        existing = self._approval(resource, approver)
        record = {
            "kind": "approval",
            "resource": resource,
            "approver_role": approver,
            "scope": scope,
            "status": status,
            "granted_at": self._now,
            "ttl_seconds": ttl,
        }
        if existing is None:
            self._records.append(record)
        else:
            existing.update(record)
        self._save()
        label = "provisional approval" if status == "provisional" else "approval"
        # Emitted so the *per-approver* step is gradable, not just the final
        # grant. Recording an approval is a real state change (it is what
        # ``approval grant``'s EffectPrecondition chain checks), but it used to
        # leave no effect record, which made every round of a multi-party
        # protocol invisible to the checker except the last one.
        return CommandResult(
            stdout=(
                f"{label} recorded: {resource} by {approver} "
                f"(scope={scope or 'all'}, ttl={ttl})"
            ),
            stderr="",
            exit_code=0,
            state_changes=self._emit(
                "approval_recorded",
                {
                    "resource": resource,
                    "approver": approver,
                    "scope": scope,
                    "status": status,
                    "ttl_seconds": ttl,
                },
            ),
        )

    # ------------------------------------------------------------------ #
    # Dispatch                                                           #
    # ------------------------------------------------------------------ #
    def execute(self, command: str) -> CommandResult:
        try:
            parts = shlex.split(command.strip())
        except ValueError as exc:
            return self._err(f"parse error: {exc}")
        if not parts or parts[0] != "approval":
            return self._err("Not an approval command")
        if len(parts) < 2:
            return CommandResult(
                stdout=(
                    "approval <request|approve|grant|revoke|provisional|"
                    "escalate|status> [options]"
                ),
                stderr="",
                exit_code=0,
            )

        sub = parts[1]
        args = parts[2:]
        handlers = {
            "request": self._cmd_request,
            "approve": self._cmd_approve,
            "grant": self._cmd_grant,
            "revoke": self._cmd_revoke,
            "provisional": self._cmd_provisional,
            "escalate": self._cmd_escalate,
            "status": self._cmd_status,
        }
        handler = handlers.get(sub)
        if handler is None:
            return self._err(f"Unknown approval subcommand: {sub}")
        return handler(args)

    def _cmd_request(self, args: list[str]) -> CommandResult:
        resource = _get_arg(args, "--resource")
        if not resource:
            return self._err("--resource is required")
        scope = _get_arg(args, "--scope") or ""
        approvers = self._parse_roles(_get_arg(args, "--approvers"))

        existing = self._request(resource)
        record = {
            "kind": "request",
            "resource": resource,
            "scope": scope,
            "approvers": approvers,
            "status": "open",
        }
        if existing is None:
            self._records.append(record)
        else:
            existing.update(record)
        self._save()
        who = ", ".join(approvers) if approvers else "(any)"
        return CommandResult(
            stdout=(
                f"access requested: {resource} (scope={scope or 'all'}) "
                f"needs approval from {who}"
            ),
            stderr="",
            exit_code=0,
            state_changes=self._emit(
                "approval_requested",
                {
                    "resource": resource,
                    "scope": scope,
                    "approvers": ", ".join(approvers),
                },
            ),
        )

    def _cmd_approve(self, args: list[str]) -> CommandResult:
        return self._record_approval(args, "approved")

    def _cmd_provisional(self, args: list[str]) -> CommandResult:
        return self._record_approval(args, "provisional")

    def _cmd_grant(self, args: list[str]) -> CommandResult:
        resource = _get_arg(args, "--resource")
        if not resource:
            return self._err("--resource is required")
        req = self._request(resource)
        if req is None:
            return self._err(f"no access request for resource: {resource}")

        requested_scope = req.get("scope", "")
        required = list(req.get("approvers") or [])
        approvals = self._approvals(resource)

        # With no explicit approver list, require at least one live approval.
        if not required:
            live = [
                a
                for a in approvals
                if a.get("status") in _LIVE_STATUSES and not self._rec_expired(a)
            ]
            if not live:
                return self._err(
                    f"cannot grant {resource}: no valid approvals recorded"
                )
            required = [a.get("approver_role", "") for a in live]

        for role in required:
            appr = self._approval(resource, role)
            if appr is None or appr.get("status") not in _LIVE_STATUSES:
                return self._err(
                    f"cannot grant {resource}: missing approval from {role}"
                )
            if self._rec_expired(appr):
                return self._err(
                    f"cannot grant {resource}: approval from {role} has expired "
                    f"(its window closed before all approvals were in place)"
                )
            if not self._scope_reconciles(requested_scope, appr.get("scope", "")):
                return self._err(
                    f"cannot grant {resource}: approval from {role} covers scope "
                    f"'{appr.get('scope')}', which does not reconcile with the "
                    f"requested scope '{requested_scope}'"
                )

        req["status"] = "granted"
        self._save()
        payload = {"resource": resource, "scope": requested_scope}
        return CommandResult(
            stdout=f"access granted: {resource} (scope={requested_scope or 'all'})",
            stderr="",
            exit_code=0,
            state_changes=self._emit("approval_granted", payload),
        )

    def _cmd_revoke(self, args: list[str]) -> CommandResult:
        resource = _get_arg(args, "--resource")
        if not resource:
            return self._err("--resource is required")
        approver = _get_arg(args, "--approver")
        # Emitted only when a record was actually removed, so the effect stream
        # stays a faithful log of state changes rather than of attempts.
        if approver:
            appr = self._approval(resource, approver)
            removed = appr is not None
            if removed:
                self._records = [r for r in self._records if r is not appr]
                self._save()
            return CommandResult(
                stdout=f"approval revoked: {resource} by {approver}",
                stderr="",
                exit_code=0,
                state_changes=(
                    self._emit(
                        "approval_revoked",
                        {"resource": resource, "approver": approver},
                    )
                    if removed
                    else None
                ),
            )
        # Revoke the whole request + its approvals.
        kept = [r for r in self._records if r.get("resource") != resource]
        removed = len(kept) != len(self._records)
        self._records = kept
        self._save()
        return CommandResult(
            stdout=f"access request revoked: {resource}",
            stderr="",
            exit_code=0,
            state_changes=(
                self._emit(
                    "approval_revoked", {"resource": resource, "approver": ""}
                )
                if removed
                else None
            ),
        )

    def _cmd_escalate(self, args: list[str]) -> CommandResult:
        resource = _get_arg(args, "--resource")
        if not resource:
            return self._err("--resource is required")
        to = _get_arg(args, "--to")
        if not to:
            return self._err("--to is required")
        req = self._request(resource)
        if req is not None:
            req["escalated_to"] = to
            self._save()
        payload = {"resource": resource, "to": to}
        return CommandResult(
            stdout=f"approval escalated: {resource} -> {to}",
            stderr="",
            exit_code=0,
            state_changes=self._emit("approval_escalated", payload),
        )

    def _cmd_status(self, args: list[str]) -> CommandResult:
        resource = _get_arg(args, "--resource")
        records = self._records
        if resource:
            records = [r for r in records if r.get("resource") == resource]
        if not records:
            return CommandResult(stdout="No approval records.", stderr="", exit_code=0)
        lines = ["Approvals:"]
        for r in records:
            if r.get("kind") == "request":
                who = ", ".join(r.get("approvers") or []) or "(any)"
                lines.append(
                    f"  request {r.get('resource')} scope={r.get('scope') or 'all'} "
                    f"status={r.get('status')} needs=[{who}]"
                )
            else:
                state = "expired" if self._rec_expired(r) else r.get("status")
                lines.append(
                    f"  approval {r.get('resource')} by {r.get('approver_role')} "
                    f"scope={r.get('scope') or 'all'} ({state})"
                )
        return CommandResult(stdout="\n".join(lines), stderr="", exit_code=0)


def _get_arg(args: list[str], flag: str, default: str | None = None) -> str | None:
    """Return ``--flag value`` or ``--flag=value``, else ``default``."""
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(flag + "="):
            return a[len(flag) + 1:]
    return default
