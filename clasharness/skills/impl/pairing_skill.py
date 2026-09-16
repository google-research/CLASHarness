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

"""Multi-device pairing-session protocol skill (M3).

The ``pairing`` family models a short-lived, multi-step device pairing handshake
that must be driven to completion *within a single time window* and *in order*.
This is the mechanic the ``multi_device_pairing`` scenarios grade (in combination
with ``time_window_expiration``): a correct agent walks the handshake
``start → show-code → enter-code → confirm`` before the session's window closes;
an agent that lets the session expire, or that skips a step (e.g. confirms before
the code is entered), is caught — those transitions fail with exit 1 and produce
no ``pairing_completed`` effect.

Verbs::

    pairing start      --device D [--ttl N]           # open a pairing session
    pairing show-code   --device D [--code C]           # reveal the pairing code
    pairing enter-code  --device D [--code C]           # peer enters the code
    pairing confirm     --device D                      # -> pairing_completed
    pairing status      [--device D]                    # read-only introspection

Finite-state machine (per device session)::

    (none) --start--> started --show-code--> code_shown
           --enter-code--> entered --confirm--> confirmed

Any step attempted out of order fails; any step attempted after the session's
window has closed fails (the session is ``expired``). The window is a single TTL
measured from ``start`` (matching the ``time_window_expiration`` framing: a
temporary session valid for only N seconds from start). Expiration rides the
plumbed virtual clock (:attr:`~StatefulProtocolSkill._now`); only ``wait``
advances it. ``confirm`` emits ``pairing_completed`` through the shared
protocol-effect channel (:meth:`StatefulProtocolSkill._emit`). State persists to
``<state_dir>/pairing.json`` and is snapshotted for ``undo``.
"""

from __future__ import annotations

import hashlib
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

#: Default pairing-session lifetime in virtual seconds when ``--ttl`` is omitted.
_DEFAULT_TTL = 300


def _derive_code(device: str) -> str:
    """Deterministic 6-digit pairing code for a device (stable across runs)."""
    digest = hashlib.sha256(device.encode("utf-8")).hexdigest()
    return str(int(digest, 16) % 1_000_000).zfill(6)


class PairingSkill(StatefulProtocolSkill):
    """Ordered, time-bounded multi-device pairing handshake."""

    SPEC = SkillSpec(
        name="pairing",
        category="coordination",
        summary="Ordered, time-bounded multi-device pairing handshake.",
        prefixes=("pairing",),
        real_kind=RealKind.LOCAL_PERSISTENT,
        commands=(
            CommandSpec(
                "pairing start",
                "Open a time-bounded pairing session for a device.",
                args={
                    "--device": {"type": "string"},
                    "--ttl": {"type": "integer"},
                },
                required=("--device",),
                emits_effects=("pairing_started",),
            ),
            CommandSpec(
                "pairing show-code",
                "Reveal the pairing code for an open session.",
                args={
                    "--device": {"type": "string"},
                    "--code": {"type": "string"},
                },
                required=("--device",),
                emits_effects=("pairing_code_shown",),
            ),
            CommandSpec(
                "pairing enter-code",
                "Enter the pairing code from the peer device.",
                args={
                    "--device": {"type": "string"},
                    "--code": {"type": "string"},
                },
                required=("--device",),
                emits_effects=("pairing_code_entered",),
            ),
            CommandSpec(
                "pairing confirm",
                "Complete the pairing once the code is entered (fails if expired).",
                args={"--device": {"type": "string"}},
                required=("--device",),
                emits_effects=("pairing_completed",),
                requires=(
                    EffectPrecondition(
                        chain=(
                            "pairing start",
                            "pairing show-code",
                            "pairing enter-code",
                        ),
                        identifier_arg="--device",
                    ),
                ),
            ),
            CommandSpec(
                "pairing status",
                "Show the status of pairing sessions (read-only).",
                args={"--device": {"type": "string"}},
            ),
        ),
        smoke=(
            "pairing start --device phone-1",
            "pairing show-code --device phone-1",
            "pairing enter-code --device phone-1",
            "pairing confirm --device phone-1",
        ),
    )

    def __init__(self) -> None:
        super().__init__(prefixes=("pairing",), family="pairing")

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    def _find(self, device: str) -> dict[str, Any] | None:
        for rec in self._records:
            if rec.get("device") == device:
                return rec
        return None

    def _rec_expired(self, rec: dict[str, Any]) -> bool:
        return self._expired(int(rec.get("created_at", 0)), rec.get("ttl"))

    @staticmethod
    def _parse_ttl(raw: str | None, default: int | None) -> int | None:
        if raw is None:
            return default
        try:
            return int(raw)
        except (TypeError, ValueError):
            return default

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
        if not parts or parts[0] != "pairing":
            return self._err("Not a pairing command")
        if len(parts) < 2:
            return CommandResult(
                stdout=(
                    "pairing <start|show-code|enter-code|confirm|status> [options]"
                ),
                stderr="",
                exit_code=0,
            )

        sub = parts[1]
        args = parts[2:]
        handlers = {
            "start": self._cmd_start,
            "show-code": self._cmd_show_code,
            "enter-code": self._cmd_enter_code,
            "confirm": self._cmd_confirm,
            "status": self._cmd_status,
        }
        handler = handlers.get(sub)
        if handler is None:
            return self._err(f"Unknown pairing subcommand: {sub}")
        return handler(args)

    def _cmd_start(self, args: list[str]) -> CommandResult:
        device = _get_arg(args, "--device")
        if not device:
            return self._err("--device is required")
        ttl = self._parse_ttl(_get_arg(args, "--ttl"), _DEFAULT_TTL)

        existing = self._find(device)
        record = {
            "device": device,
            "state": "started",
            "created_at": self._now,
            "ttl": ttl,
            "code": None,
        }
        if existing is None:
            self._records.append(record)
        else:
            existing.update(record)
        self._save()
        # Each step of the pairing state machine (started -> code_shown ->
        # entered -> completed) is a real transition, so each emits. Without
        # this only the final ``confirm`` was observable to the checker.
        return CommandResult(
            stdout=f"pairing session started: {device} (ttl={ttl})",
            stderr="",
            exit_code=0,
            state_changes=self._emit(
                "pairing_started", {"device": device, "ttl": ttl}
            ),
        )

    def _cmd_show_code(self, args: list[str]) -> CommandResult:
        device = _get_arg(args, "--device")
        if not device:
            return self._err("--device is required")
        session = self._find(device)
        if session is None:
            return self._err(f"no pairing session for device: {device}")
        if self._rec_expired(session):
            return self._err(f"pairing session for {device} has expired")
        if session.get("state") not in ("started", "code_shown"):
            return self._err(
                f"cannot show code for {device} in state {session.get('state')}"
            )
        code = _get_arg(args, "--code") or _derive_code(device)
        session["code"] = code
        session["state"] = "code_shown"
        self._save()
        return CommandResult(
            stdout=f"pairing code for {device}: {code}",
            stderr="",
            exit_code=0,
            state_changes=self._emit(
                "pairing_code_shown", {"device": device, "code": code}
            ),
        )

    def _cmd_enter_code(self, args: list[str]) -> CommandResult:
        device = _get_arg(args, "--device")
        if not device:
            return self._err("--device is required")
        session = self._find(device)
        if session is None:
            return self._err(f"no pairing session for device: {device}")
        if self._rec_expired(session):
            return self._err(f"pairing session for {device} has expired")
        if session.get("state") != "code_shown":
            return self._err(
                f"no code has been shown for {device} (state {session.get('state')})"
            )
        provided = _get_arg(args, "--code")
        if provided is not None and provided != session.get("code"):
            return self._err(f"pairing code mismatch for {device}")
        session["state"] = "entered"
        self._save()
        return CommandResult(
            stdout=f"pairing code accepted: {device}",
            stderr="",
            exit_code=0,
            state_changes=self._emit(
                "pairing_code_entered",
                {"device": device, "code": session.get("code") or ""},
            ),
        )

    def _cmd_confirm(self, args: list[str]) -> CommandResult:
        device = _get_arg(args, "--device")
        if not device:
            return self._err("--device is required")
        session = self._find(device)
        if session is None:
            return self._err(f"no pairing session for device: {device}")
        if self._rec_expired(session):
            return self._err(
                f"pairing session for {device} has expired; its window has closed"
            )
        if session.get("state") != "entered":
            return self._err(
                f"cannot confirm {device}: code has not been entered "
                f"(state {session.get('state')})"
            )
        session["state"] = "confirmed"
        self._save()
        payload = {"device": device}
        return CommandResult(
            stdout=f"pairing completed: {device}",
            stderr="",
            exit_code=0,
            state_changes=self._emit("pairing_completed", payload),
        )

    def _cmd_status(self, args: list[str]) -> CommandResult:
        device = _get_arg(args, "--device")
        if device:
            session = self._find(device)
            if session is None:
                return CommandResult(
                    stdout=f"{device}: unpaired", stderr="", exit_code=0
                )
            state = (
                "expired" if self._rec_expired(session)
                else session.get("state", "started")
            )
            return CommandResult(
                stdout=f"{device}: {state}", stderr="", exit_code=0
            )
        if not self._records:
            return CommandResult(stdout="No pairing sessions.", stderr="", exit_code=0)
        lines = ["Pairing sessions:"]
        for r in self._records:
            state = "expired" if self._rec_expired(r) else r.get("state", "started")
            lines.append(f"  {r.get('device', '')} -> {state}")
        return CommandResult(stdout="\n".join(lines), stderr="", exit_code=0)


def _get_arg(args: list[str], flag: str, default: str | None = None) -> str | None:
    """Return ``--flag value`` or ``--flag=value``, else ``default``."""
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(flag + "="):
            return a[len(flag) + 1:]
    return default
