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

"""Scenario seed to LLM layer orchestration for PRO data generation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clasharness.scenario._common import ScenarioPipelineBase, _update_anchor_id
from clasharness.scenario.llm_client import ScenarioLLMError

_LAYER_ORDER = [
    "layer0_narrative",
    "layer1_workspace",
    "layer2_sessions",
    "layer3_eval",
    "layer4_dynamic",
]
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]*$")

_TARGET_WORKSPACE_FILES = 8
_TARGET_SESSIONS = 6
_TARGET_MESSAGES_PER_SESSION = 4
_TARGET_UPDATES = 3
_UPDATE_BEFORE_ROUNDS = ["q08", "q15", "q21"]
_MESSAGE_ROLES = ["participant", "reviewer", "owner", "participant"]


@dataclass
class ScenarioSeed:
    """Brief seed consumed by the LLM scenario pipeline."""

    slug: str
    domain: str
    title: str
    target_rounds: int = 24
    core_conflict: str = ""
    participants: list[str] = field(default_factory=list)
    required_constraints: list[str] = field(default_factory=list)
    allowed_tool_families: list[str] = field(
        default_factory=lambda: ["file", "email", "tasks", "calendar"]
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScenarioSeed":
        seed = cls(
            slug=str(data.get("slug", "")),
            domain=str(data.get("domain", "")),
            title=str(data.get("title", "")),
            target_rounds=int(data.get("target_rounds", 24)),
            core_conflict=str(data.get("core_conflict", "")),
            participants=[str(item) for item in data.get("participants", [])],
            required_constraints=[str(item) for item in data.get("required_constraints", [])],
            allowed_tool_families=[
                str(item)
                for item in data.get(
                    "allowed_tool_families",
                    ["file", "email", "tasks", "calendar"],
                )
            ],
            metadata=dict(data.get("metadata", {})),
        )
        seed.validate()
        return seed

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "domain": self.domain,
            "title": self.title,
            "target_rounds": self.target_rounds,
            "core_conflict": self.core_conflict,
            "participants": self.participants,
            "required_constraints": self.required_constraints,
            "allowed_tool_families": self.allowed_tool_families,
            "metadata": self.metadata,
        }

    def validate(self) -> None:
        errors: list[str] = []
        if not self.slug or not _SLUG_RE.match(self.slug):
            errors.append("slug must be a non-empty lowercase path-safe id")
        if not self.domain:
            errors.append("domain is required")
        if not self.title:
            errors.append("title is required")
        if self.target_rounds < 1:
            errors.append("target_rounds must be positive")
        if not self.allowed_tool_families:
            errors.append("allowed_tool_families must include at least one family")
        if errors:
            raise ValueError("Invalid scenario seed: " + "; ".join(errors))


def load_seed_file(path: Path) -> ScenarioSeed:
    return ScenarioSeed.from_dict(json.loads(path.read_text(encoding="utf-8")))


class LLMScenarioPipeline(ScenarioPipelineBase):
    """Generate scenario-forge-style layers from a brief seed."""

    def generate(self, seed: ScenarioSeed) -> dict[str, Any]:
        """Generate layers through LLM-authored outlines plus small fill calls.

        The scenario seed and spec constraints define the legal shape. The LLM
        first proposes compact outlines for workspace files, sessions/messages,
        and dynamic updates. The pipeline validates those outlines, assigns
        stable ids, then calls the LLM again for each small content slot. This
        keeps generation extensible without asking the model to emit one large,
        fragile JSON layer.
        """

        layers: dict[str, Any] = {}

        self._log("[1/5] generating layer0_narrative from seed and spec constraints")
        narrative = self._call(
            "layer0_narrative_core",
            seed=seed,
            layers=layers,
            slot={
                "required_fields": [
                    "summary",
                    "core_conflict",
                    "truth_anchor",
                    "preference_rule",
                    "canonical_decision_id",
                    "canonical_rule_id",
                    "status_timeline",
                ]
            },
        )
        narrative = _normalize_narrative(narrative)
        layers["layer0_narrative"] = narrative
        self._validate_layer_shape("layer0_narrative", narrative)
        self._log(f"[1/5] completed layer0_narrative: {_layer_summary('layer0_narrative', narrative)}")

        self._log("[2/5] generating layer1_workspace outline")
        workspace_outline_raw = self._call(
            "layer1_workspace_outline",
            seed=seed,
            layers=layers,
            slot={
                "target_file_count": _TARGET_WORKSPACE_FILES,
                "requirements": [
                    "include an index file",
                    "include at least one stale or misleading artifact",
                    "include at least one authoritative current artifact",
                    "include a reliability or evidence matrix artifact",
                    "use short relative paths only",
                    "do not generate file content in the outline",
                ],
            },
        )
        workspace_outline = _normalize_workspace_outline(
            workspace_outline_raw,
            target_count=_TARGET_WORKSPACE_FILES,
        )
        self._log(f"[2/5] outline accepted: {len(workspace_outline)} workspace files")

        self._log("[2/5] filling layer1_workspace file-by-file")
        files: list[dict[str, Any]] = []
        for index, outline_item in enumerate(workspace_outline, start=1):
            file_id = str(outline_item["id"])
            path = str(outline_item["path"])
            purpose = str(outline_item["purpose"])
            self._log(f"  workspace file {index}/{len(workspace_outline)}: {path}")
            item = self._call(
                "layer1_workspace_file",
                seed=seed,
                layers=layers,
                slot={
                    "id": file_id,
                    "path": path,
                    "purpose": purpose,
                    "workspace_outline": workspace_outline,
                    "canonical_evidence_ids": _canonical_evidence_ids(narrative),
                },
            )
            content = _workspace_file_content(item, path=path)
            if not content:
                raise ScenarioLLMError(f"workspace file {path} returned empty content")
            files.append({"id": file_id, "path": path, "content": content})
        layers["layer1_workspace"] = {
            "files": files,
            "notes": {
                "truth_anchor": narrative.get("truth_anchor"),
                "preference_rule": narrative.get("preference_rule"),
                "generation": "spec_guided_outline_then_slot_fill",
                "outline": workspace_outline,
            },
        }
        self._validate_layer_shape("layer1_workspace", layers["layer1_workspace"])
        self._log(f"[2/5] completed layer1_workspace: {_layer_summary('layer1_workspace', layers['layer1_workspace'])}")

        self._log("[3/5] generating layer2_sessions outline")
        session_outline_raw = self._call(
            "layer2_session_outline",
            seed=seed,
            layers=layers,
            slot={
                "target_session_count": _TARGET_SESSIONS,
                "messages_per_session": _TARGET_MESSAGES_PER_SESSION,
                "message_roles": _MESSAGE_ROLES,
                "requirements": [
                    "sessions should cover different evidence channels",
                    "each session must contain message purposes, not full text",
                    "at least one session should preserve stale assumptions",
                    "at least one session should point to authoritative evidence",
                    "do not generate message text in the outline",
                ],
            },
        )
        session_outline = _normalize_session_outline(
            session_outline_raw,
            target_sessions=_TARGET_SESSIONS,
            messages_per_session=_TARGET_MESSAGES_PER_SESSION,
        )
        self._log(f"[3/5] outline accepted: {len(session_outline)} sessions")

        self._log("[3/5] filling layer2_sessions message-by-message")
        sessions: list[dict[str, Any]] = []
        for index, session_spec in enumerate(session_outline, start=1):
            session_id = str(session_spec["session_id"])
            channel = str(session_spec["channel"])
            purpose = str(session_spec["purpose"])
            self._log(f"  session {index}/{len(session_outline)}: {session_id} ({channel})")
            messages: list[dict[str, str]] = []
            message_specs = list(session_spec["messages"])
            for message_index, message_spec in enumerate(message_specs, start=1):
                role = str(message_spec["role"])
                message_purpose = str(message_spec["message_purpose"])
                speaker = str(message_spec.get("speaker") or _speaker_for(seed, message_index))
                self._log(
                    f"    message {message_index}/{len(message_specs)}: "
                    f"{role} / {message_purpose}"
                )
                item = self._call(
                    "layer2_session_message_text",
                    seed=seed,
                    layers=layers,
                    slot={
                        "session_id": session_id,
                        "channel": channel,
                        "session_purpose": purpose,
                        "message_index": message_index,
                        "role": role,
                        "speaker": speaker,
                        "message_purpose": message_purpose,
                        "session_outline": session_spec,
                    },
                )
                text = _clean_slot_text(item.get("text"))
                if not text:
                    raise ScenarioLLMError(
                        f"session {session_id} message {message_index} returned empty text"
                    )
                messages.append({"role": role, "speaker": speaker, "text": text})
            sessions.append(
                {
                    "session_id": session_id,
                    "kind": "chat",
                    "channel": channel,
                    "messages": messages,
                    "purpose": purpose,
                }
            )
        layers["layer2_sessions"] = {"sessions": sessions}
        self._validate_layer_shape("layer2_sessions", layers["layer2_sessions"])
        self._log(f"[3/5] completed layer2_sessions: {_layer_summary('layer2_sessions', layers['layer2_sessions'])}")

        session_ids = [str(item["session_id"]) for item in session_outline]
        workspace_ids = [str(item["id"]) for item in workspace_outline]
        primary_workspace_refs = _primary_workspace_refs(workspace_ids)

        self._log("[4/5] generating layer4_dynamic outline")
        update_outline_raw = self._call(
            "layer4_update_outline",
            seed=seed,
            layers=layers,
            slot={
                "target_update_count": _TARGET_UPDATES,
                "suggested_before_rounds": _UPDATE_BEFORE_ROUNDS,
                "available_sessions": session_ids,
                "requirements": [
                    "updates should change or clarify evidence after earlier rounds",
                    "updates should be traceable to workspace/session material",
                    "do not generate update content in the outline",
                ],
            },
        )
        update_outline = _normalize_update_outline(
            update_outline_raw,
            target_count=_TARGET_UPDATES,
            before_rounds=_UPDATE_BEFORE_ROUNDS,
            session_ids=session_ids,
        )
        self._log(f"[4/5] outline accepted: {len(update_outline)} updates")

        self._log("[4/5] filling layer4_dynamic update-by-update")
        updates: list[dict[str, Any]] = []
        for index, update_spec in enumerate(update_outline, start=1):
            update_id = str(update_spec["update_id"])
            before_round = str(update_spec["before_round"])
            purpose = str(update_spec["purpose"])
            workspace_path = str(update_spec["workspace_path"])
            session_id = str(update_spec["session_id"])
            self._log(f"  update {index}/{len(update_outline)}: {update_id} before {before_round}")
            note_item = self._call(
                "layer4_update_workspace_content",
                seed=seed,
                layers=layers,
                slot={
                    "update_id": update_id,
                    "before_round": before_round,
                    "purpose": purpose,
                    "workspace_path": workspace_path,
                    "update_outline": update_spec,
                    "canonical_evidence_ids": _canonical_evidence_ids(narrative),
                    "update_anchor_id": _update_anchor_id(update_id),
                },
            )
            message_item = self._call(
                "layer4_update_message_text",
                seed=seed,
                layers=layers,
                slot={
                    "update_id": update_id,
                    "before_round": before_round,
                    "purpose": purpose,
                    "session_id": session_id,
                    "role": "owner",
                    "speaker": _speaker_for(seed, index),
                    "update_outline": update_spec,
                    "canonical_evidence_ids": _canonical_evidence_ids(narrative),
                    "update_anchor_id": _update_anchor_id(update_id),
                },
            )
            update_content = _clean_slot_text(note_item.get("content"))
            message_text = _clean_slot_text(message_item.get("text"))
            if not update_content:
                raise ScenarioLLMError(f"update {update_id} returned empty workspace_content")
            if not message_text:
                raise ScenarioLLMError(f"update {update_id} returned empty session message")
            updates.append(
                {
                    "update_id": update_id,
                    "before_round": before_round,
                    "anchor_id": _update_anchor_id(update_id),
                    "workspace_adds": [
                        {
                            "id": f"uw{index:02d}",
                            "path": workspace_path,
                            "content": _ensure_anchor_text(
                                update_content,
                                [_update_anchor_id(update_id), *_canonical_evidence_ids(narrative)],
                            ),
                        }
                    ],
                    "session_appends": [
                        {
                            "session_id": session_id,
                            "messages": [
                                {
                                    "role": "owner",
                                    "speaker": _speaker_for(seed, index),
                                    "text": _ensure_anchor_text(
                                        message_text,
                                        [_update_anchor_id(update_id)],
                                    ),
                                }
                            ],
                        }
                    ],
                }
            )
        layers["layer4_dynamic"] = {"updates": updates}
        self._validate_layer_shape("layer4_dynamic", layers["layer4_dynamic"])
        self._log(f"[4/5] completed layer4_dynamic: {_layer_summary('layer4_dynamic', layers['layer4_dynamic'])}")

        self._log("[5/5] generating layer3_eval round-by-round")
        rounds: list[dict[str, Any]] = []
        eval_items: list[dict[str, Any]] = []
        update_specs = layers["layer4_dynamic"]["updates"]
        for index in range(1, seed.target_rounds + 1):
            round_id = f"q{index:02d}"
            round_kind = "mc" if index <= 7 else "ec"
            self._log(f"  eval round {index}/{seed.target_rounds}: {round_id} ({round_kind})")
            session_refs = _round_session_refs(session_ids, index)
            active_update_ids = _active_update_ids_for_round_index(update_specs, index)
            expected_anchor_ids = _round_expected_anchor_ids(
                narrative,
                index,
                updates=update_specs,
            )
            item = self._call(
                "layer3_eval_round",
                seed=seed,
                layers=layers,
                slot={
                    "round_id": round_id,
                    "kind": round_kind,
                    "report_path": f"/reports/{seed.slug}_{round_id}.md",
                    "session_refs": session_refs,
                    "workspace_refs": primary_workspace_refs,
                    "active_update_ids": active_update_ids,
                    "workspace_outline": workspace_outline,
                    "session_outline": session_outline,
                    "forbidden_anchor_ids": expected_anchor_ids,
                },
            )
            required_evidence = [
                str(narrative.get("truth_anchor", "")).strip(),
                str(narrative.get("preference_rule", "")).strip(),
            ]
            question = _sanitize_question(
                str(item.get("question", "")).strip()
                or f"Resolve {seed.title} round {round_id}.",
                forbidden_ids=expected_anchor_ids,
            )
            rounds.append(
                {
                    "round_id": round_id,
                    "kind": round_kind,
                    "question": question,
                    "session_refs": session_refs,
                    "workspace_refs": primary_workspace_refs,
                    "visible_refs": [
                        *[f"workspace:{ref}" for ref in primary_workspace_refs],
                        *[f"session:{ref}" for ref in session_refs],
                        *[f"update:{ref}" for ref in active_update_ids],
                    ],
                    "update_ids": active_update_ids,
                    "required_evidence": required_evidence,
                    "required_evidence_ids": expected_anchor_ids,
                    "answer": {
                        "report_path": f"/reports/{seed.slug}_{round_id}.md",
                        "truth_anchor": required_evidence[0],
                        "preference_rule": required_evidence[1],
                        "expected_anchor_ids": expected_anchor_ids,
                        "rationale_summary": str(item.get("rationale_summary", "")).strip(),
                    },
                }
            )
            eval_items.append(
                {
                    "eval_id": f"eval_{round_id}",
                    "round_id": round_id,
                    "kind": "task_checks",
                    "required_evidence": required_evidence,
                    "required_evidence_ids": expected_anchor_ids,
                }
            )
        layers["layer3_eval"] = {"rounds": rounds, "eval_items": eval_items}
        self._validate_layer_shape("layer3_eval", layers["layer3_eval"])
        self._log(f"[5/5] completed layer3_eval: {_layer_summary('layer3_eval', layers['layer3_eval'])}")

        return layers

    def _call(
        self,
        layer: str,
        *,
        seed: ScenarioSeed,
        layers: dict[str, Any],
        slot: dict[str, Any],
    ) -> dict[str, Any]:
        return self.client.generate_layer(
            layer,
            {
                "seed": seed.to_dict(),
                "context": _compact_layers_for_prompt(layers),
                "slot": slot,
            },
        )

    @staticmethod
    def _validate_layer_shape(layer: str, data: dict[str, Any]) -> None:
        required = {
            "layer0_narrative": ["summary", "core_conflict"],
            "layer1_workspace": ["files"],
            "layer2_sessions": ["sessions"],
            "layer3_eval": ["rounds", "eval_items"],
            "layer4_dynamic": ["updates"],
        }.get(layer, [])
        missing = [key for key in required if key not in data]
        if missing:
            raise ScenarioLLMError(f"{layer} missing required keys: {', '.join(missing)}")


def _layer_summary(layer: str, data: dict[str, Any]) -> str:
    if layer == "layer1_workspace":
        return f"{len(data.get('files', []))} files"
    if layer == "layer2_sessions":
        return f"{len(data.get('sessions', []))} sessions"
    if layer == "layer3_eval":
        return f"{len(data.get('rounds', []))} rounds, {len(data.get('eval_items', []))} eval items"
    if layer == "layer4_dynamic":
        return f"{len(data.get('updates', []))} updates"
    keys = ", ".join(sorted(data.keys())[:5])
    return f"keys={keys}"


def _normalize_workspace_outline(data: dict[str, Any], *, target_count: int) -> list[dict[str, str]]:
    files = data.get("files")
    if not isinstance(files, list):
        raise ScenarioLLMError("layer1_workspace_outline must return a files list")
    if len(files) < 3:
        raise ScenarioLLMError("layer1_workspace_outline must include at least 3 files")
    normalized: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for index, item in enumerate(files[: max(target_count + 4, target_count)], start=1):
        if not isinstance(item, dict):
            raise ScenarioLLMError("workspace outline file entries must be objects")
        path = _safe_outline_path(item.get("path"), fallback=f"evidence_{index:02d}.md")
        path = _dedupe_path(path, seen_paths)
        purpose = _clean_slot_text(item.get("purpose")) or f"scenario evidence artifact {index}"
        file_id = _safe_id(item.get("id"), prefix="w", index=index)
        normalized.append({"id": file_id, "path": path, "purpose": purpose})
        seen_paths.add(path)
    return normalized


def _normalize_session_outline(
    data: dict[str, Any],
    *,
    target_sessions: int,
    messages_per_session: int,
) -> list[dict[str, Any]]:
    sessions = data.get("sessions")
    if not isinstance(sessions, list):
        raise ScenarioLLMError("layer2_session_outline must return a sessions list")
    if len(sessions) < 2:
        raise ScenarioLLMError("layer2_session_outline must include at least 2 sessions")
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(sessions[: max(target_sessions + 2, target_sessions)], start=1):
        if not isinstance(item, dict):
            raise ScenarioLLMError("session outline entries must be objects")
        session_id = _safe_id(item.get("session_id") or item.get("id"), prefix="s", index=index)
        while session_id in seen_ids:
            session_id = f"s{index:02d}_{len(seen_ids) + 1}"
        seen_ids.add(session_id)
        channel = _safe_channel(item.get("channel"), fallback=f"scenario-channel-{index}")
        purpose = _clean_slot_text(item.get("purpose")) or f"scenario discussion thread {index}"
        raw_messages = item.get("messages")
        if not isinstance(raw_messages, list) or len(raw_messages) < 2:
            raise ScenarioLLMError(f"session outline {session_id} must include at least 2 message specs")
        messages: list[dict[str, str]] = []
        for message_index, raw_message in enumerate(raw_messages[: max(messages_per_session + 1, messages_per_session)], start=1):
            if not isinstance(raw_message, dict):
                raise ScenarioLLMError(f"session outline {session_id} message specs must be objects")
            role = _clean_role(raw_message.get("role"), fallback=_MESSAGE_ROLES[(message_index - 1) % len(_MESSAGE_ROLES)])
            speaker = _clean_slot_text(raw_message.get("speaker"))
            message_purpose = (
                _clean_slot_text(raw_message.get("message_purpose"))
                or _clean_slot_text(raw_message.get("purpose"))
                or f"{role} evidence contribution {message_index}"
            )
            messages.append(
                {
                    "role": role,
                    "speaker": speaker,
                    "message_purpose": message_purpose,
                }
            )
        normalized.append(
            {
                "session_id": session_id,
                "channel": channel,
                "purpose": purpose,
                "messages": messages,
            }
        )
    return normalized


def _normalize_update_outline(
    data: dict[str, Any],
    *,
    target_count: int,
    before_rounds: list[str],
    session_ids: list[str],
) -> list[dict[str, str]]:
    updates = data.get("updates")
    if not isinstance(updates, list):
        raise ScenarioLLMError("layer4_update_outline must return an updates list")
    if len(updates) < 1:
        raise ScenarioLLMError("layer4_update_outline must include at least 1 update")
    normalized: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    fallback_session = session_ids[0] if session_ids else "s01"
    for index, item in enumerate(updates[: max(target_count + 2, target_count)], start=1):
        if not isinstance(item, dict):
            raise ScenarioLLMError("update outline entries must be objects")
        update_id = _safe_id(item.get("update_id") or item.get("id"), prefix="u", index=index)
        before_round = _round_ref(item.get("before_round"), fallback=before_rounds[min(index - 1, len(before_rounds) - 1)])
        purpose = _clean_slot_text(item.get("purpose")) or f"late evidence update {index}"
        workspace_path = _safe_outline_path(
            item.get("workspace_path") or item.get("path"),
            fallback=f"updates/late_evidence_{index:02d}.md",
        )
        workspace_path = _dedupe_path(workspace_path, seen_paths)
        seen_paths.add(workspace_path)
        session_id = _clean_slot_text(item.get("session_id"))
        if session_id not in session_ids:
            session_id = fallback_session
        normalized.append(
            {
                "update_id": update_id,
                "before_round": before_round,
                "purpose": purpose,
                "workspace_path": workspace_path,
                "session_id": session_id,
            }
        )
    return normalized


def _primary_workspace_refs(workspace_ids: list[str]) -> list[str]:
    if not workspace_ids:
        return []
    refs = workspace_ids[: min(3, len(workspace_ids))]
    if workspace_ids[-1] not in refs and len(refs) < 4:
        refs.append(workspace_ids[-1])
    return refs


def _round_session_refs(session_ids: list[str], round_index: int) -> list[str]:
    if not session_ids:
        return []
    refs = [session_ids[0]]
    other = session_ids[(round_index - 1) % len(session_ids)]
    if other not in refs:
        refs.append(other)
    return refs


def _safe_id(value: Any, *, prefix: str, index: int) -> str:
    text = _clean_slot_text(value).lower().replace("-", "_")
    if not text or not re.match(r"^[a-z][a-z0-9_]*$", text):
        return f"{prefix}{index:02d}"
    return text[:48]


def _safe_channel(value: Any, *, fallback: str) -> str:
    text = _clean_slot_text(value).strip("#").lower().replace(" ", "-")
    if not text or not re.match(r"^[a-z0-9][a-z0-9_\-]*$", text):
        return fallback
    return text[:64]


def _safe_outline_path(value: Any, *, fallback: str) -> str:
    text = _clean_slot_text(value).lstrip("/")
    if not text or ".." in text.split("/") or not re.match(r"^[A-Za-z0-9][A-Za-z0-9._/\-]*$", text):
        return fallback
    return text[:120]


def _dedupe_path(path: str, seen_paths: set[str]) -> str:
    if path not in seen_paths:
        return path
    stem, suffix = (path.rsplit(".", 1) + [""])[:2] if "." in path else (path, "")
    for index in range(2, 100):
        candidate = f"{stem}_{index}.{suffix}" if suffix else f"{stem}_{index}"
        if candidate not in seen_paths:
            return candidate
    raise ScenarioLLMError(f"could not deduplicate generated path: {path}")


def _clean_role(value: Any, *, fallback: str) -> str:
    role = _clean_slot_text(value).lower().replace(" ", "_")
    if not role or not re.match(r"^[a-z][a-z0-9_]*$", role):
        return fallback
    return role[:32]


def _round_ref(value: Any, *, fallback: str) -> str:
    text = _clean_slot_text(value).lower()
    if re.match(r"^q[0-9]{2}$", text):
        return text
    if text.isdigit():
        return f"q{int(text):02d}"
    return fallback


def _compact_layers_for_prompt(layers: dict[str, Any]) -> dict[str, Any]:
    """Keep slot prompts small by passing structure summaries, not full content."""

    compact: dict[str, Any] = {}
    narrative = layers.get("layer0_narrative")
    if isinstance(narrative, dict):
        compact["narrative"] = {
            key: narrative.get(key)
            for key in (
                "summary",
                "core_conflict",
                "truth_anchor",
                "preference_rule",
                "canonical_decision_id",
                "canonical_rule_id",
                "status_timeline",
            )
            if key in narrative
        }

    workspace = layers.get("layer1_workspace")
    if isinstance(workspace, dict):
        compact["workspace_files"] = [
            {"id": item.get("id"), "path": item.get("path")}
            for item in workspace.get("files", [])
            if isinstance(item, dict)
        ]

    sessions = layers.get("layer2_sessions")
    if isinstance(sessions, dict):
        compact["sessions"] = [
            {
                "session_id": item.get("session_id"),
                "kind": item.get("kind"),
                "channel": item.get("channel"),
            }
            for item in sessions.get("sessions", [])
            if isinstance(item, dict)
        ]

    dynamic = layers.get("layer4_dynamic")
    if isinstance(dynamic, dict):
        compact["updates"] = [
            {
                "update_id": item.get("update_id"),
                "before_round": item.get("before_round"),
                "anchor_id": item.get("anchor_id"),
            }
            for item in dynamic.get("updates", [])
            if isinstance(item, dict)
        ]

    eval_layer = layers.get("layer3_eval")
    if isinstance(eval_layer, dict):
        rounds = eval_layer.get("rounds", [])
        compact["round_count"] = len(rounds) if isinstance(rounds, list) else 0

    return compact


def _normalize_narrative(data: dict[str, Any]) -> dict[str, Any]:
    """Force model-filled narrative slots back into scalar evidence strings."""

    normalized = dict(data)
    normalized["summary"] = _coerce_text(normalized.get("summary"))
    normalized["core_conflict"] = _coerce_text(normalized.get("core_conflict"))
    normalized["truth_anchor"] = _coerce_text(normalized.get("truth_anchor"))
    normalized["preference_rule"] = _coerce_text(normalized.get("preference_rule"))
    normalized["canonical_decision_id"] = _canonical_id(
        normalized.get("canonical_decision_id"),
        prefix="DECISION",
        fallback_source=normalized.get("truth_anchor") or normalized.get("core_conflict"),
        fallback="RECONCILE_CURRENT_EVIDENCE",
    )
    normalized["canonical_rule_id"] = _canonical_id(
        normalized.get("canonical_rule_id"),
        prefix="RULE",
        fallback_source=normalized.get("preference_rule"),
        fallback="PREFER_TRACEABLE_CURRENT_EVIDENCE",
    )
    normalized["status_timeline"] = _normalize_status_timeline(normalized)
    return normalized


def _canonical_id(
    value: Any,
    *,
    prefix: str,
    fallback_source: Any,
    fallback: str,
) -> str:
    text = _coerce_text(value).upper().strip()
    if re.match(rf"^{re.escape(prefix)}:[A-Z0-9_]+$", text):
        body = text.split(":", 1)[1]
        if prefix == "DECISION" and body.startswith("STATUS_"):
            text = ""
        else:
            return text[:96]
    source = _coerce_text(fallback_source).upper()
    if prefix == "DECISION" and "GATED" in source and "LAUNCH" in source:
        slug = "GATED_LAUNCH_REQUIRED"
    elif prefix == "DECISION" and "UNCONDITIONAL" in source and "RELEASE" in source:
        slug = "NO_UNCONDITIONAL_RELEASE"
    elif prefix == "RULE" and "OWNER" in source and "STALE" in source:
        slug = "NEWER_OWNER_EVIDENCE_OVERRIDES_STALE_SUMMARY"
    elif prefix == "RULE" and "EVIDENCE" in source:
        slug = "PREFER_TRACEABLE_CURRENT_EVIDENCE"
    else:
        slug = fallback
    return f"{prefix}:{slug}"


def _normalize_status_timeline(narrative: dict[str, Any]) -> list[dict[str, str]]:
    raw = narrative.get("status_timeline")
    if isinstance(raw, list):
        normalized: list[dict[str, str]] = []
        for index, item in enumerate(raw, start=1):
            if not isinstance(item, dict):
                continue
            anchor = _canonical_id(
                item.get("anchor_id"),
                prefix="DECISION",
                fallback_source=item.get("status") or item.get("summary"),
                fallback=str(narrative["canonical_decision_id"]).split(":", 1)[-1],
            )
            normalized.append(
                {
                    "from_round": _round_ref(item.get("from_round"), fallback="q01"),
                    "anchor_id": anchor,
                    "summary": _clean_slot_text(item.get("summary") or item.get("status")),
                }
            )
        if normalized:
            return normalized
    return [
        {
            "from_round": "q01",
            "anchor_id": str(narrative["canonical_decision_id"]),
            "summary": _clean_slot_text(narrative.get("truth_anchor")),
        }
    ]


def _canonical_evidence_ids(narrative: dict[str, Any]) -> list[str]:
    ids = [
        str(narrative.get("canonical_decision_id", "")).strip(),
        str(narrative.get("canonical_rule_id", "")).strip(),
    ]
    return [item for item in ids if item]


def _round_expected_anchor_ids(
    narrative: dict[str, Any],
    round_index: int,
    *,
    updates: list[dict[str, Any]] | None = None,
) -> list[str]:
    ids = _canonical_evidence_ids(narrative)
    timeline = narrative.get("status_timeline")
    active_status = ""
    if isinstance(timeline, list):
        for item in timeline:
            if not isinstance(item, dict):
                continue
            ref = _round_ref(item.get("from_round"), fallback="q01")
            try:
                ref_index = int(ref[1:])
            except ValueError:
                ref_index = 1
            if ref_index <= round_index:
                active_status = str(item.get("anchor_id", "")).strip()
    if active_status and not _is_weak_anchor_id(active_status) and active_status not in ids:
        ids.insert(0, active_status)
    for update in updates or []:
        if not isinstance(update, dict):
            continue
        try:
            before_index = int(_round_ref(update.get("before_round"), fallback="q99")[1:])
        except ValueError:
            continue
        if before_index <= round_index:
            ids.append(str(update.get("anchor_id") or _update_anchor_id(str(update.get("update_id") or ""))))
    return list(dict.fromkeys(item for item in ids if item))


def _active_update_ids_for_round_index(
    updates: list[dict[str, Any]],
    round_index: int,
) -> list[str]:
    active: list[str] = []
    for update in updates:
        if not isinstance(update, dict):
            continue
        try:
            before_index = int(_round_ref(update.get("before_round"), fallback="q99")[1:])
        except ValueError:
            continue
        update_id = str(update.get("update_id") or "").strip()
        if update_id and before_index <= round_index:
            active.append(update_id)
    return active


_ANCHOR_RE = re.compile(r"\b(?:DECISION|RULE|UPDATE):[A-Z0-9_]+\b", re.IGNORECASE)


def _sanitize_question(question: str, *, forbidden_ids: list[str]) -> str:
    """Remove machine-check anchor ids from agent-facing prompts."""

    cleaned = question.strip()
    for anchor_id in sorted(set(forbidden_ids), key=len, reverse=True):
        if not anchor_id:
            continue
        cleaned = re.sub(
            re.escape(anchor_id),
            _anchor_label(anchor_id),
            cleaned,
            flags=re.IGNORECASE,
        )
    cleaned = _ANCHOR_RE.sub(lambda match: _anchor_label(match.group(0)), cleaned)
    return " ".join(cleaned.split()).strip()


def _anchor_label(anchor_id: str) -> str:
    upper = anchor_id.upper()
    if upper.startswith("DECISION:"):
        return "the current scenario decision"
    if upper.startswith("RULE:"):
        return "the evidence precedence rule"
    if upper.startswith("UPDATE:"):
        return "the relevant dynamic update"
    return "the relevant evidence anchor"


def _is_weak_anchor_id(anchor_id: str) -> bool:
    upper = anchor_id.upper()
    return bool(re.match(r"^DECISION:STATUS_[0-9]+$", upper))


def _ensure_anchor_text(text: str, anchors: list[str]) -> str:
    content = text.strip()
    missing = [anchor for anchor in anchors if anchor and anchor not in content]
    if missing:
        content += "\n\nCanonical evidence anchors: " + ", ".join(missing)
    return content


def _workspace_file_content(item: dict[str, Any], *, path: str) -> str:
    """Extract materialized file content from an LLM slot response.

    Live providers sometimes return a JSON artifact directly, e.g.
    {"path": "index.json", "files": [...]}, instead of wrapping it in a
    "content" field. Treat that as valid file content rather than failing the
    whole scenario, while keeping genuinely empty responses invalid.
    """

    explicit = item.get("content")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    if explicit is not None and not isinstance(explicit, str):
        text = _coerce_text(explicit)
        if text:
            return text

    payload = {
        key: value
        for key, value in item.items()
        if key not in {"id", "path", "content"} and value not in (None, "", [], {})
    }
    if not payload:
        return ""

    suffix = Path(path).suffix.lower()
    if suffix in {".json", ".jsonl"} or isinstance(payload, dict):
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    return _coerce_text(payload)


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "; ".join(_coerce_text(item) for item in value if _coerce_text(item))
    if isinstance(value, dict):
        priority = value.get("decision_priority")
        if isinstance(priority, list) and priority:
            primary = _coerce_text(priority[0])
            style = _coerce_text(value.get("communication_style"))
            return f"{primary} {style}".strip()
        parts = []
        for key in sorted(value):
            text = _coerce_text(value[key])
            if text:
                parts.append(f"{key}: {text}")
        return "; ".join(parts)
    return str(value).strip()


def _speaker_for(seed: ScenarioSeed, index: int) -> str:
    participants = seed.participants or ["Avery", "Blair", "Casey", "Devon"]
    return participants[(index - 1) % len(participants)]


def _clean_slot_text(value: Any) -> str:
    text = _coerce_text(value)
    return " ".join(text.split()).strip()
