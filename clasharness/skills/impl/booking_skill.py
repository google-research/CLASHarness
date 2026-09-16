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

"""Reversible-action booking-hold protocol skill (M3).

The ``booking`` family models a **reversible hold** on a scarce resource that is
only tentative until it is *confirmed*, and which auto-expires if left
unconfirmed past its time window. This is the mechanic the
``reversible_action_hold`` scenarios grade (in combination with
``time_window_expiration``): a correct agent confirms (or releases) its hold
*before the window closes*; an agent that dawdles past the TTL and then tries to
confirm is caught, because confirming an expired hold fails with exit 1 and
produces no ``booking_confirmed`` effect.

Verbs::

    booking search  [--resource R] [--query Q]      # read-only availability
    booking hold    --resource R [--holder H] [--ttl N]  # -> booking_held
    booking confirm --resource R [--holder H]        # -> booking_confirmed
    booking release --resource R                     # -> booking_released
    booking extend  --resource R [--ttl N]           # refresh the window
    booking status  [--resource R]                   # read-only introspection

Effect-bearing verbs emit ``booking_held`` / ``booking_confirmed`` /
``booking_released`` through the shared protocol-effect channel
(:meth:`StatefulProtocolSkill._emit`).

Expiration rides the plumbed virtual clock: a ``hold`` stamps ``created_at`` with
:attr:`~StatefulProtocolSkill._now` (virtual seconds published by the
environment before each dispatch and after every ``wait``), and ``confirm`` /
``extend`` guard the hold with :meth:`~StatefulProtocolSkill._expired`. Only
``wait`` advances the virtual clock, so the window closes exactly when the
trajectory waits past the TTL. Records persist to ``<state_dir>/booking.json``
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

_DEFAULT_HOLDER = "coordinator"
#: Default hold lifetime in virtual seconds when ``--ttl`` is omitted. Matches
#: the ``time_window_expiration`` framing ("valid for only 300 seconds").
_DEFAULT_TTL = 300


class BookingSkill(StatefulProtocolSkill):
    """Reversible resource holds with a time-window confirmation deadline."""

    SPEC = SkillSpec(
        name="booking",
        category="coordination",
        summary="Reversible resource holds with a time-window confirmation deadline.",
        prefixes=("booking",),
        real_kind=RealKind.LOCAL_PERSISTENT,
        commands=(
            CommandSpec(
                "booking search",
                "List resource holds and their availability (read-only).",
                args={
                    "--resource": {"type": "string"},
                    "--query": {"type": "string"},
                },
            ),
            CommandSpec(
                "booking hold",
                "Place a reversible, time-limited hold on a resource.",
                args={
                    "--resource": {"type": "string"},
                    "--holder": {"type": "string"},
                    "--ttl": {"type": "integer"},
                },
                required=("--resource",),
                emits_effects=("booking_held",),
            ),
            CommandSpec(
                "booking confirm",
                "Confirm a hold before its window closes (fails if expired).",
                args={
                    "--resource": {"type": "string"},
                    "--holder": {"type": "string"},
                },
                required=("--resource",),
                emits_effects=("booking_confirmed",),
                requires=(
                    EffectPrecondition(
                        chain=("booking hold",), identifier_arg="--resource"
                    ),
                ),
            ),
            CommandSpec(
                "booking release",
                "Release (reverse) a hold on a resource.",
                args={"--resource": {"type": "string"}},
                required=("--resource",),
                emits_effects=("booking_released",),
            ),
            CommandSpec(
                "booking extend",
                "Refresh a hold's time window before it expires.",
                args={
                    "--resource": {"type": "string"},
                    "--ttl": {"type": "integer"},
                },
                required=("--resource",),
                emits_effects=("booking_extended",),
            ),
            CommandSpec(
                "booking status",
                "Show the status of resource holds (read-only).",
                args={"--resource": {"type": "string"}},
            ),
        ),
        smoke=(
            "booking hold --resource room-a --holder alice",
            "booking confirm --resource room-a --holder alice",
            "booking status",
        ),
    )

    def __init__(self) -> None:
        super().__init__(prefixes=("booking",), family="booking")
        self._actor = ""

    def _on_initialize(self, env_vars: dict[str, str]) -> None:
        self._actor = str(env_vars.get("CLASHARNESS_ACTOR", "") or "").strip()

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    def _resolve_holder(self, explicit: str | None) -> str:
        if explicit:
            return explicit
        if self._actor:
            return self._actor
        return _DEFAULT_HOLDER

    def _find(self, resource: str) -> dict[str, Any] | None:
        """Return the (single) hold record for ``resource``, if any."""
        for rec in self._records:
            if rec.get("resource") == resource:
                return rec
        return None

    def _rec_expired(self, rec: dict[str, Any]) -> bool:
        """True iff the hold has outlived its TTL against the virtual clock."""
        return self._expired(int(rec.get("created_at", 0)), rec.get("ttl"))

    @staticmethod
    def _parse_ttl(raw: str | None, default: int | None) -> int | None:
        if raw is None:
            return default
        try:
            return int(raw)
        except (TypeError, ValueError):
            return default

    # ------------------------------------------------------------------ #
    # Dispatch                                                           #
    # ------------------------------------------------------------------ #
    def execute(self, command: str) -> CommandResult:
        try:
            parts = shlex.split(command.strip())
        except ValueError as exc:
            return CommandResult(stdout="", stderr=f"parse error: {exc}", exit_code=1)
        if not parts or parts[0] != "booking":
            return CommandResult(stdout="", stderr="Not a booking command", exit_code=1)
        if len(parts) < 2:
            return CommandResult(
                stdout="booking <search|hold|confirm|release|extend|status> [options]",
                stderr="",
                exit_code=0,
            )

        sub = parts[1]
        args = parts[2:]
        handlers = {
            "search": self._cmd_search,
            "hold": self._cmd_hold,
            "confirm": self._cmd_confirm,
            "release": self._cmd_release,
            "extend": self._cmd_extend,
            "status": self._cmd_status,
        }
        handler = handlers.get(sub)
        if handler is None:
            return CommandResult(
                stdout="", stderr=f"Unknown booking subcommand: {sub}", exit_code=1
            )
        return handler(args)

    def _cmd_search(self, args: list[str]) -> CommandResult:
        resource = _get_arg(args, "--resource")
        query = _get_arg(args, "--query")
        recs = self._records
        if resource:
            recs = [r for r in recs if r.get("resource") == resource]
        if query:
            recs = [r for r in recs if query in str(r.get("resource", ""))]
        if not recs:
            return CommandResult(
                stdout="No matching resources are held.", stderr="", exit_code=0
            )
        lines = ["Holds:"]
        for r in recs:
            state = "expired" if self._rec_expired(r) else r.get("status", "held")
            lines.append(
                f"  {r.get('resource', '')} -> {r.get('holder', '')} ({state})"
            )
        return CommandResult(stdout="\n".join(lines), stderr="", exit_code=0)

    def _cmd_hold(self, args: list[str]) -> CommandResult:
        resource = _get_arg(args, "--resource")
        if not resource:
            return CommandResult(stdout="", stderr="--resource is required", exit_code=1)
        holder = self._resolve_holder(_get_arg(args, "--holder"))
        ttl = self._parse_ttl(_get_arg(args, "--ttl"), _DEFAULT_TTL)

        existing = self._find(resource)
        # A live hold by a *different* holder blocks a new hold (mutual
        # exclusion); an expired or released hold may be taken over.
        if (
            existing is not None
            and existing.get("holder") != holder
            and existing.get("status") != "released"
            and not self._rec_expired(existing)
        ):
            return CommandResult(
                stdout="",
                stderr=(
                    f"booking conflict: {resource} already held by "
                    f"{existing.get('holder')}"
                ),
                exit_code=1,
            )

        if existing is None:
            self._records.append(
                {
                    "resource": resource,
                    "holder": holder,
                    "status": "held",
                    "created_at": self._now,
                    "ttl": ttl,
                }
            )
        else:
            existing.update(
                {
                    "holder": holder,
                    "status": "held",
                    "created_at": self._now,
                    "ttl": ttl,
                }
            )
        self._save()

        payload = {
            "resource": resource,
            "holder": holder,
            "ttl": ttl,
            "created_at": self._now,
        }
        return CommandResult(
            stdout=f"booking held: {resource} by {holder} (ttl={ttl})",
            stderr="",
            exit_code=0,
            state_changes=self._emit("booking_held", payload),
        )

    def _cmd_confirm(self, args: list[str]) -> CommandResult:
        resource = _get_arg(args, "--resource")
        if not resource:
            return CommandResult(stdout="", stderr="--resource is required", exit_code=1)
        holder = _get_arg(args, "--holder")

        existing = self._find(resource)
        if existing is None or existing.get("status") == "released":
            return CommandResult(
                stdout="", stderr=f"no active hold for resource: {resource}", exit_code=1
            )
        if holder and existing.get("holder") != holder:
            return CommandResult(
                stdout="",
                stderr=(
                    f"hold on {resource} is owned by {existing.get('holder')}, "
                    f"not {holder}"
                ),
                exit_code=1,
            )
        if self._rec_expired(existing):
            return CommandResult(
                stdout="",
                stderr=(
                    f"hold on {resource} has expired; its confirmation window "
                    f"has closed"
                ),
                exit_code=1,
            )

        existing["status"] = "confirmed"
        self._save()
        payload = {"resource": resource, "holder": existing.get("holder")}
        return CommandResult(
            stdout=f"booking confirmed: {resource} for {existing.get('holder')}",
            stderr="",
            exit_code=0,
            state_changes=self._emit("booking_confirmed", payload),
        )

    def _cmd_release(self, args: list[str]) -> CommandResult:
        resource = _get_arg(args, "--resource")
        if not resource:
            return CommandResult(stdout="", stderr="--resource is required", exit_code=1)

        existing = self._find(resource)
        if existing is not None:
            self._records = [r for r in self._records if r is not existing]
            self._save()

        payload = {"resource": resource}
        return CommandResult(
            stdout=f"booking released: {resource}",
            stderr="",
            exit_code=0,
            state_changes=self._emit("booking_released", payload),
        )

    def _cmd_extend(self, args: list[str]) -> CommandResult:
        resource = _get_arg(args, "--resource")
        if not resource:
            return CommandResult(stdout="", stderr="--resource is required", exit_code=1)

        existing = self._find(resource)
        if existing is None or existing.get("status") == "released":
            return CommandResult(
                stdout="", stderr=f"no active hold for resource: {resource}", exit_code=1
            )
        if self._rec_expired(existing):
            return CommandResult(
                stdout="",
                stderr=f"hold on {resource} has already expired; cannot extend",
                exit_code=1,
            )

        ttl = self._parse_ttl(_get_arg(args, "--ttl"), existing.get("ttl"))
        existing["created_at"] = self._now
        existing["ttl"] = ttl
        self._save()
        # Extending resets the hold's window -- a real change to when the hold
        # expires, and the whole point of the expiring-hold mechanic, so it is
        # emitted rather than left invisible.
        return CommandResult(
            stdout=f"booking extended: {resource} (ttl={ttl})",
            stderr="",
            exit_code=0,
            state_changes=self._emit(
                "booking_extended", {"resource": resource, "ttl": ttl}
            ),
        )

    def _cmd_status(self, args: list[str]) -> CommandResult:
        resource = _get_arg(args, "--resource")
        if resource:
            existing = self._find(resource)
            if existing is None:
                return CommandResult(
                    stdout=f"{resource}: free", stderr="", exit_code=0
                )
            state = (
                "expired" if self._rec_expired(existing)
                else existing.get("status", "held")
            )
            return CommandResult(
                stdout=f"{resource}: {state} by {existing.get('holder')}",
                stderr="",
                exit_code=0,
            )
        if not self._records:
            return CommandResult(stdout="No holds.", stderr="", exit_code=0)
        lines = ["Holds:"]
        for r in self._records:
            state = "expired" if self._rec_expired(r) else r.get("status", "held")
            lines.append(
                f"  {r.get('resource', '')} -> {r.get('holder', '')} ({state})"
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
