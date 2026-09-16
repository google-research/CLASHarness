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

"""Strict package validation for materialized PRO scenario packages."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from clasharness.scenario._common import _round_index
from clasharness.scenario.spec import ScenarioSpec
from clasharness.scenario.validator import ScenarioValidationResult, validate_scenario
from clasharness.scenario.vlm_spec import VLMScenarioSpec, validate_vlm_scenario_spec

_FUTURE_ROUND_REF_RE = re.compile(r"\b(?:ROUND|R)\s*(?:0?[8-9]|1[0-9]|2[0-9])\b", re.IGNORECASE)
_FUTURE_COMPACT_REF_RE = re.compile(r"\b(?:q|r|round)[_\- ]?(?:0?[8-9]|1[0-9]|2[0-9])\b", re.IGNORECASE)
_FILE_TOKEN_RE = re.compile(
    r"\b[A-Za-z0-9][A-Za-z0-9._\-]*\."
    r"(?:md|markdown|jsonl|eml|log|csv|tsv|txt|json|yaml|yml|pdf|docx?|xlsx?|ics|svg|png|jpe?g|gif|webp|html)\b",
    re.IGNORECASE,
)
_ROUND_IN_TOKEN_RE = re.compile(r"(?:^|[_\-])(?:q|r|round)[_\-]?(\d{1,2})(?:[_\-]|$)", re.IGNORECASE)
_SYSTEM_PRELOAD_FILES = {"emails.json", "tasks.json", "calendar_events.json"}
_EARLY_FORBIDDEN_RUNTIME_MARKERS = (
    "DEPRECATION NOTICE",
    "APPENDED ROUND",
    "ROUND 24",
    "POST-LAUNCH",
    "SUPERSEDED",
    "DO NOT TREAT",
    "RECONCILIATION NOTICE",
    "CANONICAL DECISION",
    "GOVERNING RULE",
    "AUTHORITATIVE SOURCES",
    "SUPERSEDING",
    "NEWER OWNER",
    "EVIDENCE LOG",
    "HAVE SINCE",
    "DEPRECATED",
    "GATED LAUNCH",
    "GATED —",
    "GATED -",
    "NOT READY TO PROCEED",
    "LAUNCH IS NOT READY",
    "WHY THIS DOCUMENT IS STALE",
    "FORMAL WITHDRAWAL",
    "CORRECT LAUNCH STATUS",
    "FIRST FORMAL OVERRIDE",
    "STRONGEST EVIDENTIARY ANCHOR",
    "MANDATORY AND NOT OBTAINED",
    "HOLD-LIFT",
    "HOLD LIFT",
    "FINAL APPROVAL",
    "FINAL HOLD",
    "FINAL OWNER",
    "FINAL AUDIT",
    "LATE OWNER",
    "COMPLIANCE HOLD",
    "DO NOT ACT",
    "BEFORE COMPLIANCE FLAGS",
    "RETAINED FOR AUDIT",
    "AUDIT AND RECONCILIATION",
)


def strict_validate_scenario_package(spec: ScenarioSpec, scenario_dir: Path) -> ScenarioValidationResult:
    """Validate references, materialized files, and evidence traceability."""

    result = validate_scenario(spec)
    scenario_dir = Path(scenario_dir)
    if not scenario_dir.exists():
        result.errors.append(f"scenario_dir does not exist: {scenario_dir}")
        return result

    evidence_text_parts: list[str] = []

    for item in spec.workspace.files:
        path = Path(str(item.get("materialized_path") or f"workspace/{item.get('path', '')}"))
        file_path = scenario_dir / path
        if not file_path.exists():
            result.errors.append(f"missing workspace file: {path.as_posix()}")
        else:
            evidence_text_parts.append(file_path.read_text(encoding="utf-8", errors="replace"))

    for session in spec.sessions:
        if not session.path:
            result.errors.append(f"session {session.session_id}: path is required")
            continue
        path = Path(session.path)
        file_path = scenario_dir / path
        if not file_path.exists():
            result.errors.append(f"missing session file: {path.as_posix()}")
        else:
            raw = file_path.read_text(encoding="utf-8", errors="replace")
            evidence_text_parts.append(raw)
            for idx, line in enumerate(raw.splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError as exc:
                    result.errors.append(f"session {session.session_id}:{idx} invalid JSONL: {exc}")

    for update in spec.updates:
        for item in update.workspace_adds:
            rel_path_str = item.get("materialized_path")
            if not rel_path_str:
                result.errors.append(f"update {update.update_id}: workspace_add missing materialized_path")
                continue
            path = Path(str(rel_path_str))
            file_path = scenario_dir / path
            if not file_path.exists():
                result.errors.append(f"missing update workspace file: {path.as_posix()}")
            elif file_path.is_dir():
                result.errors.append(f"update workspace path is a directory, not a file: {path.as_posix()}")
            else:
                evidence_text_parts.append(file_path.read_text(encoding="utf-8", errors="replace"))

    asset_manifest_path = scenario_dir / "asset_manifest.json"
    if asset_manifest_path.exists():
        try:
            manifest = json.loads(asset_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            result.errors.append(f"asset_manifest.json invalid JSON: {exc}")
            manifest = {}
        assets = manifest.get("assets", manifest if isinstance(manifest, list) else [])
        if not isinstance(assets, list):
            result.errors.append("asset_manifest.json must contain an assets list")
            assets = []
        for asset in assets:
            if not isinstance(asset, dict):
                result.errors.append("asset_manifest.json assets must be objects")
                continue
            _validate_asset_manifest_item(asset, scenario_dir, evidence_text_parts, result)
        vlm_spec = None
    else:
        vlm_spec = None
    vlm_spec_path = scenario_dir / "vlm_spec.json"
    if vlm_spec_path.exists():
        try:
            vlm_spec = VLMScenarioSpec.read_json(vlm_spec_path)
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            result.errors.append(f"vlm_spec.json invalid: {exc}")
        else:
            for error in validate_vlm_scenario_spec(vlm_spec):
                result.errors.append(f"vlm_spec.json: {error}")
            _validate_vlm_package_assets(vlm_spec, scenario_dir, result)

    for update in spec.updates:
        for item in update.session_appends:
            rel_path_str = item.get("materialized_path")
            if not rel_path_str:
                result.errors.append(f"update {update.update_id}: session_append missing materialized_path")
                continue
            path = Path(str(rel_path_str))
            file_path = scenario_dir / path
            if not file_path.exists():
                result.errors.append(f"missing update session append: {path.as_posix()}")
            elif file_path.is_dir():
                result.errors.append(f"update session append path is a directory, not a file: {path.as_posix()}")
            else:
                evidence_text_parts.append(file_path.read_text(encoding="utf-8", errors="replace"))

    target_rounds = spec.metadata.get("seed", {}).get("target_rounds")
    if target_rounds is not None and len(spec.rounds) != int(target_rounds):
        result.errors.append(f"expected {target_rounds} rounds, found {len(spec.rounds)}")
    if len(spec.rounds) < 12:
        result.errors.append("PRO scenario packages should contain at least 12 rounds")

    corpus = "\n".join(evidence_text_parts).lower()
    update_before_by_id = {
        update.update_id: str(update.before_round or "") for update in spec.updates
    }
    rounds_by_id = {round_item.round_id: round_item for round_item in spec.rounds}
    for eval_item in spec.eval_items:
        round_state_corpus = ""
        round_item = rounds_by_id.get(eval_item.round_id)
        if round_item is not None:
            visible_text = _round_visible_corpus(round_item, result)
            _validate_runtime_mask(round_item, visible_text, result)
            round_state_corpus = visible_text.lower()
        required_ids = eval_item.metadata.get("required_evidence_ids", [])
        if required_ids:
            if not isinstance(required_ids, list):
                result.errors.append(
                    f"eval item {eval_item.eval_id}: required_evidence_ids must be a list"
                )
                required_ids = []
            for evidence_id in required_ids:
                if not isinstance(evidence_id, str):
                    result.errors.append(
                        f"eval item {eval_item.eval_id}: required_evidence_ids must contain strings"
                    )
                    continue
                needle = evidence_id.strip().lower()
                if _is_weak_anchor_id(evidence_id):
                    result.errors.append(
                        f"eval item {eval_item.eval_id}: weak evidence id is not allowed: {evidence_id}"
                    )
                target_corpus = round_state_corpus or corpus
                # Check if it is a visible VLM anchor for this round
                is_visible_vlm_anchor = False
                if round_item is not None and vlm_spec is not None:
                    r_idx = _round_index(round_item.round_id)
                    for asset in vlm_spec.assets:
                        if asset.available_from_round <= r_idx:
                            if any(aid.strip().lower() == needle for aid in asset.anchor_ids):
                                is_visible_vlm_anchor = True
                                break
                if needle and needle not in target_corpus and not is_visible_vlm_anchor:
                    result.errors.append(
                        f"eval item {eval_item.eval_id}: evidence id is not traceable in round-visible evidence: {evidence_id}"
                    )
        # Long narrative evidence is retained as metadata for human audit. Do
        # not make it a hard traceability requirement once short canonical ids
        # are present.
        if required_ids:
            continue
        for evidence in eval_item.metadata.get("required_evidence", []):
            if not isinstance(evidence, str):
                result.errors.append(
                    f"eval item {eval_item.eval_id}: required_evidence must be a string, got {type(evidence).__name__}"
                )
                continue
            needle = evidence.strip().lower()
            if needle and needle not in corpus:
                result.errors.append(
                    f"eval item {eval_item.eval_id}: evidence is not traceable in materialized files: {evidence}"
                )

    for round_item in spec.rounds:
        round_index = _round_index(round_item.round_id)
        expected_ids = list(round_item.expected_anchor_ids or [])
        visible_text = _round_visible_corpus(round_item, result)
        _validate_runtime_mask(round_item, visible_text, result)
        round_state_corpus = visible_text.lower()
        if not expected_ids and isinstance(round_item.metadata, dict):
            expected_ids = [
                str(item)
                for item in round_item.metadata.get("required_evidence_ids", [])
                if isinstance(item, str)
            ]
        for evidence_id in expected_ids:
            if _is_weak_anchor_id(evidence_id):
                result.errors.append(
                    f"round {round_item.round_id}: weak evidence id is not allowed: {evidence_id}"
                )
            needle = str(evidence_id).strip().lower()
            if needle and round_state_corpus and needle not in round_state_corpus:
                # Check if it is a visible VLM anchor for this round
                is_visible_vlm_anchor = False
                if vlm_spec is not None:
                    for asset in vlm_spec.assets:
                        if asset.available_from_round <= round_index:
                            if any(aid.strip().lower() == needle for aid in asset.anchor_ids):
                                is_visible_vlm_anchor = True
                                break
                if not is_visible_vlm_anchor:
                    result.errors.append(
                        f"round {round_item.round_id}: expected evidence id is not visible in task state: {evidence_id}"
                    )
        for update_id in round_item.update_ids:
            before_round = update_before_by_id.get(update_id)
            if before_round is None:
                result.errors.append(
                    f"round {round_item.round_id}: references unknown update {update_id}"
                )
                continue
            before_index = _round_index(before_round)
            if before_index and round_index and before_index > round_index:
                result.errors.append(
                    f"round {round_item.round_id}: update {update_id} is not active until {before_round}"
                )

    _validate_no_answer_leakage(spec, result)

    return result



def _validate_asset_manifest_item(
    asset: dict[str, Any],
    scenario_dir: Path,
    evidence_text_parts: list[str],
    result: ScenarioValidationResult,
) -> None:
    asset_id = str(asset.get("id") or "")
    rel_path = str(asset.get("path") or "")
    if not asset_id:
        result.errors.append("asset manifest item missing id")
    if not rel_path:
        result.errors.append(f"asset {asset_id or '<missing>'}: path is required")
        return
    for required in ("kind", "mime_type", "available_from_round"):
        if asset.get(required) in (None, ""):
            result.errors.append(f"asset {asset_id or '<missing>'}: {required} is required")
    if asset.get("generator") == "google_image" and (not asset.get("prompt") or not asset.get("caption")):
        result.errors.append(f"asset {asset_id}: google_image requires prompt and caption")
    if str(asset.get("generator") or "") in {"code_chart", "template_chart"} and not (asset.get("chart_spec_path") or asset.get("render_source_path")):
        result.errors.append(f"asset {asset_id}: chart asset requires render source or chart spec")
    if any(key.lower() in {"api_key", "authorization", "x-goog-api-key", "raw_response"} for key in asset):
        result.errors.append(f"asset {asset_id}: raw provider response or secret-like key must not be packaged")
    asset_path = scenario_dir / "assets" / rel_path.lstrip("/")
    if not asset_path.exists():
        result.errors.append(f"missing asset file: assets/{rel_path}")
    else:
        evidence_text_parts.append(asset_path.read_text(encoding="utf-8", errors="replace"))
    evidence_text_parts.append(
        "\n".join(
            str(asset.get(key) or "")
            for key in ("description", "caption", "ocr_text", "rich_text", "prompt")
        )
    )


def _validate_vlm_package_assets(
    vlm_spec: VLMScenarioSpec,
    scenario_dir: Path,
    result: ScenarioValidationResult,
) -> None:
    for asset in vlm_spec.assets:
        asset_path = scenario_dir / "assets" / asset.path.lstrip("/")
        if not asset_path.exists():
            result.errors.append(f"vlm_spec asset {asset.id}: missing generated file {asset.path}")
        if asset.metadata_path and not (scenario_dir / "assets" / asset.metadata_path).exists():
            result.errors.append(f"vlm_spec asset {asset.id}: missing metadata file {asset.metadata_path}")
        if asset.chart_spec_path and not (scenario_dir / "assets" / asset.chart_spec_path).exists():
            result.errors.append(f"vlm_spec asset {asset.id}: missing chart spec {asset.chart_spec_path}")
        if asset.render_source_path and not (scenario_dir / "assets" / asset.render_source_path).exists():
            result.errors.append(f"vlm_spec asset {asset.id}: missing render source {asset.render_source_path}")
        forbidden_meta = {"api_key", "authorization", "x-goog-api-key", "raw_response"}
        if any(str(key).lower() in forbidden_meta for key in asset.provider_metadata):
            result.errors.append(f"vlm_spec asset {asset.id}: provider metadata contains secret/raw response key")

def _round_visible_corpus(round_item: Any, result: ScenarioValidationResult) -> str:
    task_spec = round_item.task_spec or {}
    data = task_spec.get("data", {}) if isinstance(task_spec, dict) else {}
    private = data.get("private", {}) if isinstance(data, dict) else {}
    overrides = private.get("initial_state_overrides", {}) if isinstance(private, dict) else {}
    if not isinstance(overrides, dict):
        return ""
    for path in overrides:
        if "canonical_evidence.md" in str(path):
            result.errors.append(
                f"round {round_item.round_id}: canonical_evidence.md must not be agent-visible"
            )
    return json.dumps(_strip_binary_override_payloads(overrides), ensure_ascii=False, sort_keys=True)


def _strip_binary_override_payloads(value: Any) -> Any:
    if isinstance(value, dict):
        if "__bytes_b64__" in value:
            return {"__bytes_b64__": "<binary omitted>"}
        return {key: _strip_binary_override_payloads(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strip_binary_override_payloads(item) for item in value]
    return value


def _validate_runtime_mask(
    round_item: Any,
    visible_text: str,
    result: ScenarioValidationResult,
) -> None:
    round_index = _round_index(getattr(round_item, "round_id", ""))
    if round_index <= 0:
        return
    upper = str(visible_text or "").upper()
    if round_index < 8:
        for anchor_prefix in ("DECISION:", "RULE:", "UPDATE:"):
            if anchor_prefix in upper:
                _append_error_once(
                    result,
                    f"round {round_item.round_id}: early runtime state leaks {anchor_prefix}",
                )
                break
        for marker in _EARLY_FORBIDDEN_RUNTIME_MARKERS:
            if marker in upper:
                _append_error_once(
                    result,
                    f"round {round_item.round_id}: early runtime state leaks future marker {marker}",
                )
                break
    if _visible_text_references_future_round(upper, round_index=round_index):
        _append_error_once(
            result,
            f"round {round_item.round_id}: runtime state leaks future round reference"
        )
    visible_names = _visible_workspace_file_names(round_item)
    token_text = _normalize_json_escaped_text(str(visible_text or ""))
    if _references_future_file_token(token_text, round_index=round_index, visible_names=visible_names):
        _append_error_once(
            result,
            f"round {round_item.round_id}: runtime state leaks future file reference"
        )
    for token in _FILE_TOKEN_RE.findall(token_text):
        if token not in visible_names and token not in _SYSTEM_PRELOAD_FILES:
            _append_error_once(
                result,
                f"round {round_item.round_id}: runtime state references invisible file {token}",
            )
            break


# Future-round leak heuristic tuning (kept in sync with
# conflict_strict_validator._visible_text_references_future_round): bare
# ``R<N>``/``Q<N>`` tokens are flagged ONLY with a nearby round-context cue, so
# domain noise (release tag ``R2``, rack id ``R12``, fiscal quarter ``Q3``) no
# longer false-positives.
_FUTURE_ROUND_CAP = 24
_BARE_TOKEN_CONTEXT_WINDOW = 24
_ROUND_CONTEXT_CUE = re.compile(
    r"ROUND|QUESTION|NEXT|LATER|UPCOMING|SUBSEQUENT|FUTURE"
)
_EXPLICIT_ROUND_RE = re.compile(r"\bROUND[_\- ]?(\d{1,2})\b")
_BARE_ROUND_TOKEN_RE = re.compile(r"\b[RQ][_\- ]?(\d{1,2})\b")


def _visible_text_references_future_round(text: str, *, round_index: int) -> bool:
    """True if visible ``text`` references a FUTURE round/question index.

    EXPLICIT ``ROUND[_- ]?N`` is always decisive; a bare ``R<N>``/``Q<N>`` is a
    leak only when a round-context cue sits within
    ``_BARE_TOKEN_CONTEXT_WINDOW`` chars (C: false-positive reduction).
    """
    upper = str(text or "").upper()
    if "TOTAL ROUNDS" in upper or "TARGET ROUNDS" in upper:
        return False

    # Explicit round references are always decisive.
    for match in _EXPLICIT_ROUND_RE.finditer(upper):
        try:
            value = int(match.group(1))
        except ValueError:
            continue
        if round_index < value <= _FUTURE_ROUND_CAP:
            return True

    # Bare R<N>/Q<N> tokens require a nearby round-context cue.
    for match in _BARE_ROUND_TOKEN_RE.finditer(upper):
        try:
            value = int(match.group(1))
        except ValueError:
            continue
        if not (round_index < value <= _FUTURE_ROUND_CAP):
            continue
        lo = max(0, match.start() - _BARE_TOKEN_CONTEXT_WINDOW)
        hi = min(len(upper), match.end() + _BARE_TOKEN_CONTEXT_WINDOW)
        window = upper[lo:match.start()] + upper[match.end():hi]
        if _ROUND_CONTEXT_CUE.search(window):
            return True
    return False


def _visible_workspace_file_names(round_item: Any) -> set[str]:
    task_spec = getattr(round_item, "task_spec", {}) or {}
    data = task_spec.get("data", {}) if isinstance(task_spec, dict) else {}
    private = data.get("private", {}) if isinstance(data, dict) else {}
    overrides = private.get("initial_state_overrides", {}) if isinstance(private, dict) else {}
    names: set[str] = set()
    if isinstance(overrides, dict):
        for key in overrides:
            text = str(key)
            names.add(Path(text).name)
    return names


def _normalize_json_escaped_text(text: str) -> str:
    return (
        str(text or "")
        .replace("\\n", "\n")
        .replace("\\r", "\r")
        .replace("\\t", "\t")
    )


def _references_future_file_token(text: str, *, round_index: int, visible_names: set[str]) -> bool:
    """Return true when a file token explicitly points to a future round.

    File names often include round ids for provenance, e.g. ``*_r11.eml``.
    That is only a future leak if the embedded round is after the current
    round and the file is not already part of the round-visible workspace.
    """

    for token in _FILE_TOKEN_RE.findall(str(text or "")):
        if token in visible_names or token in _SYSTEM_PRELOAD_FILES:
            continue
        stem = Path(token).stem
        for match in _ROUND_IN_TOKEN_RE.finditer(stem):
            try:
                token_round = int(match.group(1))
            except ValueError:
                continue
            if round_index < token_round <= 24:
                return True
    return False


def _append_error_once(result: ScenarioValidationResult, message: str) -> None:
    if message not in result.errors:
        result.errors.append(message)


def _validate_no_answer_leakage(
    spec: ScenarioSpec,
    result: ScenarioValidationResult,
) -> None:
    narrative = spec.narrative or {}
    leaked_long_values = []
    for key in ("truth_anchor", "preference_rule"):
        text = str(narrative.get(key) or "").strip()
        if len(text) >= 80:
            leaked_long_values.append((key, text.lower()))

    eval_ids_by_round = {
        eval_item.round_id: [
            str(item).strip().lower()
            for item in eval_item.metadata.get("required_evidence_ids", [])
            if isinstance(item, str) and item.strip()
        ]
        for eval_item in spec.eval_items
    }

    for round_item in spec.rounds:
        instruction = (round_item.instruction or "").lower()
        for key, text in leaked_long_values:
            if text and text in instruction:
                result.errors.append(
                    f"round {round_item.round_id}: instruction leaks narrative {key}"
                )
        for evidence_id in eval_ids_by_round.get(round_item.round_id, []):
            if evidence_id and evidence_id in instruction:
                result.errors.append(
                    f"round {round_item.round_id}: instruction leaks expected evidence id {evidence_id}"
                )
        for evidence_id in round_item.expected_anchor_ids:
            needle = str(evidence_id).strip().lower()
            if needle and needle in instruction:
                result.errors.append(
                    f"round {round_item.round_id}: instruction leaks expected evidence id {needle}"
                )


def _is_weak_anchor_id(anchor_id: str) -> bool:
    text = str(anchor_id).upper()
    return text.startswith("DECISION:STATUS_")
