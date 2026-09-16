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

"""Coordinated multi-point security-containment protocol skill (M4, #4).

The ``containment`` family models an incident response in which a compromise is
only truly contained once **every one of four independent control points** has
been actioned: the account's active sessions are revoked, its forwarding rules
are disabled, the compromised device is isolated from the network, and the
leaked access token is blocked. This is the mechanic the
``coordinated_containment`` (#4) scenarios grade.

Verbs::

    containment revoke-sessions    --account A   # -> containment_action {sessions}
    containment disable-forwarding --account A   # -> containment_action {forwarding}
    containment isolate-device     --device D    # -> containment_action {device}
    containment block-token        --token T     # -> containment_action {token}
    containment status                           # read-only

There is exactly **one** of each control point per incident, and each is
**idempotency-guarded**: the first time a point is actioned it transitions from
``active`` to ``contained`` and emits a single ``containment_action`` effect;
re-actioning an already-``contained`` point fails with exit 1 and emits **no**
effect. The effect count therefore tops out at four — one per distinct point —
so the graders' ``count_gte(4)`` on ``containment_action`` is satisfied *iff*
all four points are contained. An agent cannot reach four by spamming a single
verb. None of the action verbs declare ``requires`` (there is no ordering
prerequisite between the four points), which keeps the lossy
``_EFFECT_COMMAND_NAMES`` inverse out of the picture for this family. Records
persist to ``<state_dir>/containment.json`` and are snapshotted for ``undo``.
"""

from __future__ import annotations

import shlex
from typing import Any

from clasharness.backend.base import CommandResult
from clasharness.skills.spec import (
    CommandSpec,
    RealKind,
    SkillSpec,
)
from clasharness.skills.stateful import StatefulProtocolSkill

#: The four independent control points, keyed by their CLI verb. The value is
#: the canonical point name recorded once the point is contained. There is
#: exactly one of each per incident, so full containment emits exactly four
#: distinct ``containment_action`` effects.
_POINT_BY_VERB = {
    "revoke-sessions": "sessions",
    "disable-forwarding": "forwarding",
    "isolate-device": "device",
    "block-token": "token",
}
#: The target flag each control-point verb reads (recorded + echoed in payload).
_TARGET_FLAG = {
    "revoke-sessions": "--account",
    "disable-forwarding": "--account",
    "isolate-device": "--device",
    "block-token": "--token",
}


class ContainmentSkill(StatefulProtocolSkill):
    """Coordinated containment: four control points, each contained at most once."""

    SPEC = SkillSpec(
        name="containment",
        category="coordination",
        summary=(
            "Coordinated multi-point security containment: revoke sessions, "
            "disable forwarding, isolate the device, and block the token. Each "
            "control point can be contained only once; full containment requires "
            "all four to be actioned."
        ),
        prefixes=("containment",),
        real_kind=RealKind.LOCAL_PERSISTENT,
        commands=(
            CommandSpec(
                "containment revoke-sessions",
                "Revoke all active sessions for the compromised account.",
                args={"--account": {"type": "string"}},
                required=("--account",),
                emits_effects=("containment_action",),
            ),
            CommandSpec(
                "containment disable-forwarding",
                "Disable mail/message forwarding rules on the account.",
                args={"--account": {"type": "string"}},
                required=("--account",),
                emits_effects=("containment_action",),
            ),
            CommandSpec(
                "containment isolate-device",
                "Isolate the compromised device from the network.",
                args={"--device": {"type": "string"}},
                required=("--device",),
                emits_effects=("containment_action",),
            ),
            CommandSpec(
                "containment block-token",
                "Revoke and block the leaked access token.",
                args={"--token": {"type": "string"}},
                required=("--token",),
                emits_effects=("containment_action",),
            ),
            CommandSpec(
                "containment status",
                "Show which control points are contained (read-only).",
                args={},
            ),
        ),
        smoke=(
            "containment revoke-sessions --account alice",
            "containment disable-forwarding --account alice",
            "containment isolate-device --device laptop-7",
            "containment block-token --token tok_abc",
            "containment status",
        ),
    )

    def __init__(self) -> None:
        super().__init__(prefixes=("containment",), family="containment")

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    def _point_record(self, point: str) -> dict[str, Any] | None:
        for rec in self._records:
            if rec.get("point") == point:
                return rec
        return None

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
        if not parts or parts[0] != "containment":
            return self._err("Not a containment command")
        if len(parts) < 2:
            return CommandResult(
                stdout=(
                    "containment <revoke-sessions|disable-forwarding|"
                    "isolate-device|block-token|status> [options]"
                ),
                stderr="",
                exit_code=0,
            )

        sub = parts[1]
        args = parts[2:]
        if sub == "status":
            return self._cmd_status(args)
        if sub in _POINT_BY_VERB:
            return self._cmd_contain(sub, args)
        return self._err(f"Unknown containment subcommand: {sub}")

    def _cmd_contain(self, verb: str, args: list[str]) -> CommandResult:
        point = _POINT_BY_VERB[verb]
        target_flag = _TARGET_FLAG[verb]
        target = _get_arg(args, target_flag)
        if not target:
            return self._err(f"{target_flag} is required")

        existing = self._point_record(point)
        if existing is not None and existing.get("status") == "contained":
            # Idempotency guard (D1): re-actioning a contained point is a no-op
            # and emits NO effect, so the effect count tops out at four (one per
            # distinct point) and a single verb cannot be spammed to reach four.
            return self._err(
                f"control point '{point}' is already contained "
                f"(target {existing.get('target')!r}); no action taken"
            )

        record = {
            "kind": "control_point",
            "point": point,
            "verb": verb,
            "target": target,
            "status": "contained",
            "contained_at": self._now,
        }
        if existing is None:
            self._records.append(record)
        else:
            existing.update(record)
        self._save()
        payload = {"point": point, "verb": verb, "target": target}
        return CommandResult(
            stdout=f"contained {point}: {verb} {target}",
            stderr="",
            exit_code=0,
            state_changes=self._emit("containment_action", payload),
        )

    def _cmd_status(self, args: list[str]) -> CommandResult:
        contained = {
            r.get("point"): r
            for r in self._records
            if r.get("status") == "contained"
        }
        lines = ["Containment control points:"]
        for verb, point in _POINT_BY_VERB.items():
            rec = contained.get(point)
            if rec is not None:
                lines.append(
                    f"  {point} ({verb}): contained target={rec.get('target')}"
                )
            else:
                lines.append(f"  {point} ({verb}): active")
        lines.append(f"contained {len(contained)}/{len(_POINT_BY_VERB)}")
        return CommandResult(stdout="\n".join(lines), stderr="", exit_code=0)


def _get_arg(args: list[str], flag: str, default: str | None = None) -> str | None:
    """Return ``--flag value`` or ``--flag=value``, else ``default``."""
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(flag + "="):
            return a[len(flag) + 1:]
    return default
