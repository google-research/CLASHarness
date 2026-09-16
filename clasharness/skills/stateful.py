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

"""Reusable base for stateful, multi-round *protocol* skills.

Concrete data skills (:class:`~clasharness.skills.impl.tasks_skill.TasksSkill`,
:class:`~clasharness.skills.impl.email_skill.EmailSkill`) all repeat the same
three chores:

1. **Preload + persist** a list of records to ``<state_dir>/<family>.json`` so a
   scenario can seed starting state and so mutations survive across the per-round
   environments the materializer spins up.
2. **Emit effects** for the grader by attaching them to
   :attr:`CommandResult.state_changes`; the environment folds them into its
   ``_effects`` ledger which :class:`EffectChecker` reads.
3. **Expire** records against a virtual clock (leases, locks, deploy windows).

Protocol families (locks, deploys, approvals, ...) need exactly these same three
building blocks, so this base factors them out once. The M1 command families are
its first real subclasses:
:class:`~clasharness.skills.impl.lock_skill.LockSkill` (``lock``) and
:class:`~clasharness.skills.impl.deploy_skill.DeploySkill` (``deploy``) both
subclass it and are registered by ``MultiAppBackend``. It remains
*behaviour-neutral for the data skills above*, which continue to embed these
chores directly rather than inheriting them.

Design notes
------------
* Effects ride a single generic ``{"effects": {<type>: [payload, ...]}}`` key
  (see :meth:`_emit`). ``Environment._track_effects`` accepts this shape
  generically, so a new protocol family needs *no* environment edit to surface
  its effects.
* The environment now plumbs its ``_virtual_time`` (virtual seconds) into skills:
  before every dispatch — and after every clock advance in ``_tick`` — it pushes
  the reading onto :attr:`_now` via ``SkillRuntime.set_now`` /
  ``MultiAppBackend.set_virtual_time``. Subclasses expire records with
  :meth:`_expired` (``self._now``-relative) or :meth:`_is_expired` (explicit
  ``now``); the self-contained monotonic :attr:`_logical_tick` remains available
  for record ordering. All three are unit-agnostic.
* :attr:`_now` is **deliberately not** in :attr:`_snapshot_attrs`: it is not
  episode state but a per-dispatch clock reading, and the environment keeps
  virtual time moving forward across ``undo`` (it never rewinds the clock), so a
  snapshotted ``_now`` would fight that. Mutable state (``_records`` +
  ``_logical_tick``) *is* snapshotted, so the base's ``to_state``/``from_state``
  (inherited from :class:`Skill`) give correct snapshot/undo for free.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable

from clasharness.backend.base import CommandResult
from clasharness.skills.base import Skill

Effect = tuple[str, dict[str, Any]]


class StatefulProtocolSkill(Skill):
    """Abstract base for skills that persist records and emit expiring effects.

    Subclasses must set a ``family`` (the json filename stem and the default
    records key) and implement :meth:`execute`. They typically call
    :meth:`_save` after mutating :attr:`_records`, return :meth:`_emit` output in
    ``state_changes``, and guard time-limited records with :meth:`_is_expired`.
    """

    # ``_records`` holds the persisted rows; ``_logical_tick`` is the optional
    # built-in monotonic clock. Both are episode-mutable, so snapshot both for
    # correct undo.
    _snapshot_attrs = ("_records", "_logical_tick")

    def __init__(self, *, prefixes: tuple[str, ...], family: str) -> None:
        super().__init__(prefixes=prefixes)
        if not family:
            raise ValueError("StatefulProtocolSkill requires a non-empty 'family'")
        self._family = family
        self._records: list[dict[str, Any]] = []
        self._state_dir: str = ""
        self._logical_tick: int = 0
        # Virtual clock (virtual seconds) published by the environment before
        # each dispatch. NOT snapshotted (see module docstring). Defaults to 0
        # so a skill invoked outside an environment behaves deterministically.
        self._now: int = 0
        self._initialized = False

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #
    def _preload_path(self) -> Path:
        return Path(self._state_dir) / f"{self._family}.json"

    def _records_key(self) -> str:
        """Key under which records live in the on-disk dict payload."""
        return self._family

    def initialize(self, state_dir: str, env_vars: dict[str, str]) -> None:
        self._state_dir = state_dir
        self._records = [copy.deepcopy(r) for r in self._default_records()]
        self._logical_tick = 0
        self._initialized = True

        path = self._preload_path()
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                payload = None
            self._load_payload(payload)

        self._on_initialize(env_vars)

    def _load_payload(self, payload: Any) -> None:
        """Merge a preload payload into ``_records``.

        Mirrors the tasks/email convention: a bare ``list`` extends the seeded
        records; a ``dict`` may carry ``{"replace_seeded": true}`` to drop seeds
        first, plus the records under :meth:`_records_key`.
        """
        if isinstance(payload, list):
            self._records.extend(copy.deepcopy(payload))
        elif isinstance(payload, dict):
            if payload.get("replace_seeded"):
                self._records = []
            rows = payload.get(self._records_key())
            if isinstance(rows, list):
                self._records.extend(copy.deepcopy(rows))

    def _save(self) -> None:
        """Persist ``_records`` back to ``<state_dir>/<family>.json`` (best effort)."""
        if not self._state_dir:
            return
        payload = {"replace_seeded": True, self._records_key(): self._records}
        try:
            self._preload_path().write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Effect emission                                                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _emit(effect_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Build a ``state_changes`` dict carrying one protocol effect.

        Shape: ``{"effects": {effect_type: [payload]}}`` -- consumed generically
        by ``Environment._track_effects`` and graded by ``EffectChecker``.
        """
        return {"effects": {effect_type: [payload]}}

    @staticmethod
    def _emit_batch(effects: Iterable[Effect]) -> dict[str, Any]:
        """Build a ``state_changes`` dict carrying several effects at once."""
        bucket: dict[str, list[dict[str, Any]]] = {}
        for effect_type, payload in effects:
            bucket.setdefault(effect_type, []).append(payload)
        return {"effects": bucket}

    # ------------------------------------------------------------------ #
    # Virtual-time helpers                                               #
    # ------------------------------------------------------------------ #
    def _advance_tick(self, ticks: int = 1) -> int:
        """Advance and return the built-in monotonic logical clock."""
        self._logical_tick += max(0, int(ticks))
        return self._logical_tick

    @staticmethod
    def _is_expired(now: int, created_at: int, ttl: int | None) -> bool:
        """True iff a record created at ``created_at`` has outlived ``ttl`` by ``now``.

        ``ttl is None`` means "never expires". ``now`` and ``created_at`` share
        the caller's own unit (virtual seconds or logical ticks); the base makes
        no assumption about which.
        """
        if ttl is None:
            return False
        return (now - created_at) >= ttl

    def _expired(self, created_at: int, ttl: int | None) -> bool:
        """:meth:`_is_expired` against the plumbed virtual clock :attr:`_now`.

        The preferred check for records whose ``created_at`` is a virtual-second
        stamp: ``now`` is supplied by the environment, so subclasses never have
        to thread it through their handlers.
        """
        return self._is_expired(self._now, created_at, ttl)

    # ------------------------------------------------------------------ #
    # Subclass hooks                                                     #
    # ------------------------------------------------------------------ #
    def _default_records(self) -> list[dict[str, Any]]:
        """Seed records installed before any preload file is merged."""
        return []

    def _on_initialize(self, env_vars: dict[str, str]) -> None:
        """Hook for subclass-specific setup after records are loaded."""
        return None

    # ------------------------------------------------------------------ #
    # Skill lifecycle                                                    #
    # ------------------------------------------------------------------ #
    def get_state(self) -> dict[str, Any]:
        return {self._records_key(): list(self._records)}

    def cleanup(self) -> None:
        self._records = []
        self._logical_tick = 0
        self._initialized = False
