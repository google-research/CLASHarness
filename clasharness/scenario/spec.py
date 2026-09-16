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

"""Canonical scenario specification dataclasses.

The scenario layer is the PRO source format. Existing runnable `Task` specs are
kept as compiled artifacts by storing one current task snapshot per round.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MockSubagent:
    """A strongly-typed representation of a subagent's role and state."""
    agent_role: str
    mock_response: str
    capabilities: list[str] = field(default_factory=list)
    private_knowledge: list[str] = field(default_factory=list)
    virtual_duration_seconds: int = 10
    state_effects: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MockSubagent":
        raw_pk = data.get("private_knowledge") or []
        if isinstance(raw_pk, str):
            pk = [raw_pk]
        elif isinstance(raw_pk, list):
            pk = [str(x) for x in raw_pk]
        else:
            pk = []

        raw_caps = data.get("capabilities") or []
        if isinstance(raw_caps, str):
            caps = [raw_caps]
        elif isinstance(raw_caps, list):
            caps = [str(x) for x in raw_caps]
        else:
            caps = []

        return cls(
            agent_role=str(data.get("agent_role", "")),
            mock_response=str(data.get("mock_response", "")),
            capabilities=caps,
            private_knowledge=pk,
            virtual_duration_seconds=int(data.get("virtual_duration_seconds", 10)),
            state_effects=_parse_state_effects(data.get("state_effects")),
        )



@dataclass
class SessionMessage:
    """A rich message inside a ScenarioSession."""
    message_id: str
    timestamp: str
    sender: str
    text: str
    attachments: list[str] = field(default_factory=list)
    is_redacted: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionMessage":
        sender_val = data.get("sender") or data.get("sender_id") or data.get("sender_name") or data.get("author") or ""
        text_val = data.get("text") or data.get("content") or data.get("body") or data.get("message") or ""
        return cls(
            message_id=str(data.get("message_id") or data.get("id") or ""),
            timestamp=str(data.get("timestamp") or data.get("time") or data.get("date") or ""),
            sender=str(sender_val),
            text=str(text_val),
            attachments=list(data.get("attachments", [])),
            is_redacted=bool(data.get("is_redacted", False)),
        )

@dataclass
class ScenarioWorkspace:
    """Workspace assets and notes used by a scenario."""

    files: list[dict[str, Any]] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | list[Any] | None) -> "ScenarioWorkspace":
        data = data or {}
        if isinstance(data, list):
            return cls(files=data, notes={})
        return cls(
            files=list(data.get("files", [])),
            notes=dict(data.get("notes", {})),
        )


def session_channel_value(data: dict[str, Any]) -> str:
    """The channel a raw session dict declares, across every spelling accepted.

    THE single source of truth for "which key holds the channel name". It exists
    because there used to be two readers of that question and they disagreed:
    ``ScenarioSession.from_dict`` accepted three spellings while the generator's
    channel gate (``conflict_vlm_materializer._session_display_channel``) read
    only ``channel``. When a session declared its name under ``channel_name`` or
    ``name``, the gate saw no channel at all, fell back to deriving one from the
    ``session_id``, and then enforced THAT invented vocabulary on the round's
    command author -- rejecting the real channel as nonexistent and rewriting it
    into the derived one. Materialization then shipped the real name, so every
    gold command and every criterion derived from it pointed at a channel the
    scenario does not have. Measured on the v7 corpus: 433 of 1085 gold
    '--channel' values (26 scenarios) and 320 criterion mentions were phantom,
    and the split was perfectly clean -- 0 phantoms among the 28 scenarios whose
    session_id-derived name happened to match the shipped one, 433 among the 32
    where they diverged.

    Callers MUST route through this rather than reading ``data["channel"]``
    directly, so a future fourth spelling cannot re-open the same divergence.

    Returns the literal ``"default"`` sentinel when the session declares none,
    matching :class:`ScenarioSession`'s field default; callers that want a
    display name derive one from the ``session_id`` in that case.
    """
    return str(
        data.get("channel")
        or data.get("channel_name")
        or data.get("name")
        or "default"
    )


@dataclass
class ScenarioSession:
    """A conversation, command, or external-log session attached to a scenario."""

    session_id: str
    kind: str = "text"
    channel: str = "default"
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    messages: list[SessionMessage] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScenarioSession":
        session_id_val = data.get("session_id") or data.get("channel_id") or data.get("id") or data.get("channel_name") or ""
        channel_val = session_channel_value(data)
        messages_list = data.get("messages") or data.get("chat") or data.get("dialogs") or []
        return cls(
            session_id=str(session_id_val),
            kind=str(data.get("kind", "text")),
            channel=str(channel_val),
            path=data.get("path"),
            metadata=dict(data.get("metadata", {})),
            messages=[SessionMessage.from_dict(m) for m in messages_list if isinstance(m, dict)],
        )


@dataclass
class ScenarioUpdate:
    """Dynamic update that can be applied before a round."""

    update_id: str
    before_round: str | None = None
    anchor_id: str | None = None
    workspace_adds: list[dict[str, Any]] = field(default_factory=list)
    session_appends: list[dict[str, Any]] = field(default_factory=list)
    session_redactions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScenarioUpdate":
        return cls(
            update_id=str(data.get("update_id", "")),
            before_round=data.get("before_round"),
            anchor_id=data.get("anchor_id"),
            workspace_adds=list(data.get("workspace_adds", [])),
            session_appends=list(data.get("session_appends", [])),
            session_redactions=list(data.get("session_redactions", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ScenarioEvalItem:
    """Evaluation item attached to a scenario round."""

    eval_id: str
    round_id: str
    kind: str = "task_checks"
    checks: list[dict[str, Any]] = field(default_factory=list)
    answer: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScenarioEvalItem":
        return cls(
            eval_id=str(data.get("eval_id", "")),
            round_id=str(data.get("round_id", "")),
            kind=str(data.get("kind", "task_checks")),
            checks=list(data.get("checks", [])),
            answer=data.get("answer"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ScenarioRound:
    """One executable round in a scenario."""

    round_id: str
    task_id: str
    instruction: str
    task_spec: dict[str, Any]
    update_ids: list[str] = field(default_factory=list)
    session_refs: list[str] = field(default_factory=list)
    workspace_refs: list[str] = field(default_factory=list)
    visible_refs: list[str] = field(default_factory=list)
    expected_anchor_ids: list[str] = field(default_factory=list)
    expert_trajectory: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScenarioRound":
        return cls(
            round_id=str(data.get("round_id", "")),
            task_id=str(data.get("task_id", "")),
            instruction=str(data.get("instruction", "")),
            task_spec=dict(data.get("task_spec", {})),
            update_ids=[str(item) for item in data.get("update_ids", [])],
            session_refs=[str(item) for item in data.get("session_refs", [])],
            workspace_refs=[str(item) for item in data.get("workspace_refs", [])],
            visible_refs=[str(item) for item in data.get("visible_refs", [])],
            expected_anchor_ids=[str(item) for item in data.get("expected_anchor_ids", [])],
            expert_trajectory=[str(item) for item in data.get("expert_trajectory", [])],
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ScenarioManifest:
    """Manifest for compiled artifacts and provenance."""

    source_generator: str
    compiled_task_ids: list[str] = field(default_factory=list)
    created_by: str = "clasharness"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ScenarioManifest":
        data = data or {}
        return cls(
            source_generator=str(data.get("source_generator", "")),
            compiled_task_ids=[str(item) for item in data.get("compiled_task_ids", [])],
            created_by=str(data.get("created_by", "clasharness")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ScenarioSpec:
    """Canonical PRO scenario spec."""

    scenario_id: str
    generator_id: str
    scenario_slug: str
    title: str
    description: str = ""
    version: str = "1.0"
    narrative: dict[str, Any] = field(default_factory=dict)
    workspace: ScenarioWorkspace = field(default_factory=ScenarioWorkspace)
    sessions: list[ScenarioSession] = field(default_factory=list)
    updates: list[ScenarioUpdate] = field(default_factory=list)
    rounds: list[ScenarioRound] = field(default_factory=list)
    eval_items: list[ScenarioEvalItem] = field(default_factory=list)
    manifest: ScenarioManifest | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.manifest is None:
            data["manifest"] = None
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScenarioSpec":
        return cls(
            scenario_id=str(data.get("scenario_id", "")),
            generator_id=str(data.get("generator_id", "")),
            scenario_slug=str(data.get("scenario_slug", "")),
            title=str(data.get("title", "")),
            description=str(data.get("description", "")),
            version=str(data.get("version", "1.0")),
            narrative=dict(data.get("narrative", {})),
            workspace=ScenarioWorkspace.from_dict(data.get("workspace")),
            sessions=[ScenarioSession.from_dict(item) for item in data.get("sessions", [])],
            updates=[ScenarioUpdate.from_dict(item) for item in data.get("updates", [])],
            rounds=[ScenarioRound.from_dict(item) for item in data.get("rounds", [])],
            eval_items=[ScenarioEvalItem.from_dict(item) for item in data.get("eval_items", [])],
            manifest=ScenarioManifest.from_dict(data.get("manifest"))
            if data.get("manifest") is not None
            else None,
            metadata=dict(data.get("metadata", {})),
        )

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False) + "\n"

    @classmethod
    def from_json(cls, raw: str) -> "ScenarioSpec":
        return cls.from_dict(json.loads(raw))

    def write_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def read_json(cls, path: Path) -> "ScenarioSpec":
        return cls.from_json(path.read_text(encoding="utf-8"))


def _parse_state_effects(raw_effects: Any) -> list[Any]:
    if not raw_effects:
        return []
    if isinstance(raw_effects, str):
        return [raw_effects]
    if isinstance(raw_effects, list):
        return list(raw_effects)
    return []

