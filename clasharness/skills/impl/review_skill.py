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

"""Collaborative review + document sign-gate protocol skill (M4, #5).

The ``review`` family models a document (a contract, a launch plan, a design)
that can be signed off only once **every review aspect has been reviewed and
cleared**. Distinct aspects — legal, privacy, security, business, layout — are
assigned to different reviewers, each of whom submits an ``ok`` or ``blocked``
verdict (a block carrying its blocking issue). The document's ``sign`` gate
fails until every aspect in play is ``ok`` with no outstanding blockers. This is
the mechanic the ``non_overlapping_assignment`` (#5) scenarios grade.

Verbs::

    review assign --aspect A --reviewer ROLE            # route an aspect
    review submit --aspect A --status ok|blocked [--issue TEXT]  # -> review_submitted
    review sign   --document D    # -> document_signed (all aspects ok, no blockers)
    review status                                        # read-only

``submit`` records one aspect's verdict and emits ``review_submitted``. ``sign``
is the gate: it fails with exit 1 (emitting **no** ``document_signed`` effect)
unless at least one aspect has been reviewed and every aspect that is in play
(assigned or submitted) is ``ok`` with no blocking issues. Because a blocked or
still-pending aspect keeps the gate shut, the reviewers' verdicts must all be
collected and cleared before the document can be signed. Records persist to
``<state_dir>/review.json`` and are snapshotted for ``undo``.
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

#: Verdict values ``review submit --status`` accepts.
_VALID_STATUSES = ("ok", "blocked")


class ReviewSkill(StatefulProtocolSkill):
    """Collaborative multi-aspect review with a document sign-off gate."""

    SPEC = SkillSpec(
        name="review",
        category="coordination",
        summary=(
            "Collaborative document review: distinct aspects (legal, privacy, "
            "security, business, layout) are assigned to reviewers and the "
            "document can be signed only once every aspect is cleared with no "
            "outstanding blockers."
        ),
        prefixes=("review",),
        real_kind=RealKind.LOCAL_PERSISTENT,
        commands=(
            CommandSpec(
                "review assign",
                "Assign a review aspect to a reviewer (non-overlapping roles).",
                args={
                    "--aspect": {"type": "string"},
                    "--reviewer": {"type": "string"},
                },
                required=("--aspect", "--reviewer"),
                emits_effects=("review_assigned",),
            ),
            CommandSpec(
                "review submit",
                "Submit one aspect's verdict (ok or blocked, with an issue).",
                args={
                    "--aspect": {"type": "string"},
                    "--status": {"type": "string"},
                    "--issue": {"type": "string"},
                },
                required=("--aspect", "--status"),
                emits_effects=("review_submitted",),
            ),
            CommandSpec(
                "review sign",
                "Sign the document once every aspect is ok with no blockers.",
                args={"--document": {"type": "string"}},
                required=("--document",),
                emits_effects=("document_signed",),
                requires=(
                    EffectPrecondition(
                        all_ready=("review submit", "--status", "ok"),
                    ),
                ),
            ),
            CommandSpec(
                "review status",
                "Show aspects, verdicts, and sign state (read-only).",
                args={"--document": {"type": "string"}},
            ),
        ),
        smoke=(
            "review assign --aspect legal --reviewer counsel",
            "review submit --aspect legal --status ok",
            "review sign --document contract",
            "review status",
        ),
    )

    def __init__(self) -> None:
        super().__init__(prefixes=("review",), family="review")

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    def _aspect(self, aspect: str) -> dict[str, Any] | None:
        for rec in self._records:
            if rec.get("kind") == "aspect" and rec.get("aspect") == aspect:
                return rec
        return None

    def _aspects(self) -> list[dict[str, Any]]:
        return [r for r in self._records if r.get("kind") == "aspect"]

    @staticmethod
    def _err(msg: str) -> CommandResult:
        return CommandResult(stdout="", stderr=msg, exit_code=1)

    # ------------------------------------------------------------------ #
    # Dispatch                                                           #
    # ------------------------------------------------------------------ #
    def execute(self, command: str) -> CommandResult:
        try:
            parts = shlex.split(command.strip())
        except ValueError as exc:
            return self._err(f"parse error: {exc}")
        if not parts or parts[0] != "review":
            return self._err("Not a review command")
        if len(parts) < 2:
            return CommandResult(
                stdout="review <assign|submit|sign|status> [options]",
                stderr="",
                exit_code=0,
            )

        sub = parts[1]
        args = parts[2:]
        handlers = {
            "assign": self._cmd_assign,
            "submit": self._cmd_submit,
            "sign": self._cmd_sign,
            "status": self._cmd_status,
        }
        handler = handlers.get(sub)
        if handler is None:
            return self._err(f"Unknown review subcommand: {sub}")
        return handler(args)

    def _cmd_assign(self, args: list[str]) -> CommandResult:
        aspect = _get_arg(args, "--aspect")
        reviewer = _get_arg(args, "--reviewer")
        if not aspect:
            return self._err("--aspect is required")
        if not reviewer:
            return self._err("--reviewer is required")
        existing = self._aspect(aspect)
        if existing is None:
            self._records.append({
                "kind": "aspect",
                "aspect": aspect,
                "assignee": reviewer,
                "status": "pending",
                "blocking_issues": [],
            })
        else:
            existing["assignee"] = reviewer
        self._save()
        # Emitted so the non-overlapping-assignment mechanic is gradable per
        # assignment (who owns which aspect), not only at the final signature.
        return CommandResult(
            stdout=f"assigned {aspect} -> {reviewer}",
            stderr="",
            exit_code=0,
            state_changes=self._emit(
                "review_assigned", {"aspect": aspect, "reviewer": reviewer}
            ),
        )

    def _cmd_submit(self, args: list[str]) -> CommandResult:
        aspect = _get_arg(args, "--aspect")
        status = _get_arg(args, "--status")
        if not aspect:
            return self._err("--aspect is required")
        if status not in _VALID_STATUSES:
            return self._err(
                f"--status must be one of {', '.join(_VALID_STATUSES)}"
            )
        issue = _get_arg(args, "--issue")
        rec = self._aspect(aspect)
        if rec is None:
            rec = {
                "kind": "aspect",
                "aspect": aspect,
                "assignee": "",
                "status": "pending",
                "blocking_issues": [],
            }
            self._records.append(rec)
        rec["status"] = status
        if status == "blocked":
            issues = list(rec.get("blocking_issues") or [])
            issues.append(issue or "unspecified")
            rec["blocking_issues"] = issues
        else:
            # A fresh ``ok`` verdict clears any earlier block.
            rec["blocking_issues"] = []
        self._save()
        payload = {"aspect": aspect, "status": status}
        return CommandResult(
            stdout=f"review submitted: {aspect} = {status}",
            stderr="",
            exit_code=0,
            state_changes=self._emit("review_submitted", payload),
        )

    def _cmd_sign(self, args: list[str]) -> CommandResult:
        document = _get_arg(args, "--document")
        if not document:
            return self._err("--document is required")
        aspects = self._aspects()
        if not aspects:
            return self._err(
                f"cannot sign {document}: no aspects have been reviewed"
            )
        for rec in aspects:
            if rec.get("status") != "ok" or rec.get("blocking_issues"):
                reason = (
                    "has blocking issues"
                    if rec.get("blocking_issues")
                    else f"is {rec.get('status')}"
                )
                return self._err(
                    f"cannot sign {document}: aspect '{rec.get('aspect')}' "
                    f"{reason}"
                )

        doc = next(
            (
                r
                for r in self._records
                if r.get("kind") == "document" and r.get("document") == document
            ),
            None,
        )
        if doc is None:
            self._records.append({
                "kind": "document",
                "document": document,
                "status": "signed",
            })
        else:
            doc["status"] = "signed"
        self._save()
        payload = {"document": document}
        return CommandResult(
            stdout=f"document signed: {document}",
            stderr="",
            exit_code=0,
            state_changes=self._emit("document_signed", payload),
        )

    def _cmd_status(self, args: list[str]) -> CommandResult:
        document = _get_arg(args, "--document")
        aspects = self._aspects()
        docs = [r for r in self._records if r.get("kind") == "document"]
        # ``review status [--document DOCUMENT]`` advertises a filter, but
        # ``--document`` used to be read nowhere: a question about one document
        # was answered with the entire register. Scope the document lines when
        # the caller's name resolves, and fall back to the full list when it
        # does not -- callers spell the same document many ways ("partnership
        # contract" vs "RB-9921-PARTNERSHIP"), and a strict filter would hide a
        # signature that is visible today. Aspects are never document-scoped
        # (they gate signing any document), so they are always listed.
        scoped = docs
        unresolved = False
        if document:
            needle = document.strip().lower()
            scoped = [
                d
                for d in docs
                if needle in str(d.get("document", "")).lower()
                or str(d.get("document", "")).lower() in needle
            ]
            if not scoped:
                unresolved = True
                scoped = docs
        if not aspects and not docs:
            return CommandResult(stdout="No review records.", stderr="", exit_code=0)
        lines = ["Review:"]
        for rec in aspects:
            blockers = rec.get("blocking_issues") or []
            suffix = f" blockers={blockers}" if blockers else ""
            lines.append(
                f"  aspect {rec.get('aspect')} reviewer={rec.get('assignee') or '(none)'} "
                f"status={rec.get('status')}{suffix}"
            )
        for doc in scoped:
            lines.append(f"  document {doc.get('document')} status={doc.get('status')}")
        if document and unresolved:
            lines.append(
                f"  (no document record matching '{document}'; showing all)"
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
