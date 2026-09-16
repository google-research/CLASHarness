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

"""Required-effect / stop-reason checks extracted from examples/train_and_eval.py."""
from __future__ import annotations

import re
import shlex
from typing import Any

from clasharness.core.observation import EvaluationResult
from clasharness.core.task import Task

from train_eval._constants import _SIDE_EFFECT_HINTS


def _instruction_implies_side_effect(task: Task | None) -> bool:
  instruction = ""
  if task is not None:
    instruction = str(
        getattr(task, "instruction", "")
        or getattr(task, "canonical_instruction", "")
        or ""
    )
  text = instruction.lower()
  return any(marker in text for marker in _SIDE_EFFECT_HINTS)

def _required_report_path(task: Task | None) -> str | None:
  if task is None:
    return None
  ground_truth = getattr(task, "ground_truth", None)
  checks = (
      getattr(ground_truth, "evaluation_checks", None) if ground_truth else None
  )
  if isinstance(checks, list):
    for check in checks:
      if not isinstance(check, dict):
        continue
      expected = check.get("expected")
      if (
          check.get("type") == "effect"
          # Only required checks gate the missing_required_report stop_reason;
          # mirror the evaluator role convention (see _required_effect_checks).
          and check.get("role", "required") == "required"
          and isinstance(expected, dict)
          and expected.get("field") == "path"
          and str(expected.get("value", "")).startswith("/reports/")
      ):
        return str(expected["value"])
  instruction = str(getattr(task, "instruction", "") or "")
  match = re.search(r"(/reports/[A-Za-z0-9._/\-]+)", instruction)
  return match.group(1) if match else None

def _created_required_report(
    actions: list[str], report_path: str | None
) -> bool:
  if not report_path:
    return True
  for action in actions:
    try:
      parts = shlex.split(action)
    except ValueError:
      continue
    if len(parts) >= 2 and parts[0] == "file" and parts[1] == "create":
      for idx, part in enumerate(parts[:-1]):
        if part == "--path" and parts[idx + 1] == report_path:
          return True
  return False

def _missing_required_report_stop_reason(
    task: Task | None, actions: list[str]
) -> str:
  report_path = _required_report_path(task)
  if report_path and not _created_required_report(actions, report_path):
    return f"missing_required_report: {report_path}"
  return ""

def _required_submission_stop_reason(
    task: Task | None,
    actions: list[str],
    eval_result: EvaluationResult | None = None,
) -> str:
  missing = _missing_required_side_effect_stop_reason(task, actions)
  if missing:
    return missing
  if eval_result is not None and not eval_result.success:
    failed_details = [
        detail for detail in eval_result.details if not detail.passed
    ]
    if any(
        "anchor" in str(detail.name).lower()
        or "evidence" in str(detail.name).lower()
        for detail in failed_details
    ):
      return "failed_required_anchor_check"
    if failed_details:
      return "incomplete_required_submission"
  return ""

def _missing_required_side_effect_stop_reason(
    task: Task | None, actions: list[str]
) -> str:
  for check in _required_effect_checks(task):
    effect_type = str(check.get("effect_type", ""))
    expected = check.get("expected")
    if _effect_satisfied_by_actions(effect_type, expected, actions, task=task):
      continue
    if effect_type == "files_created" and isinstance(expected, dict):
      path = str(expected.get("value", ""))
      if path.startswith("/reports/"):
        return f"missing_required_report: {path}"
      if path.startswith("/actions/"):
        return f"missing_required_action_log: {path}"
    if effect_type:
      return f"missing_required_side_effect: {effect_type}"
  return ""

def _required_effect_checks(task: Task | None) -> list[dict[str, Any]]:
  if task is None:
    return []
  ground_truth = getattr(task, "ground_truth", None)
  checks = (
      getattr(ground_truth, "evaluation_checks", None) if ground_truth else None
  )
  if not isinstance(checks, list):
    return []
  # Only checks that gate ``success`` count as *required* side effects. Mirror
  # the evaluator's role convention (see clasharness/evaluation/evaluator.py:
  # ``VALID_ROLES`` and ``role = check.get("role", "required")``): a missing role
  # defaults to "required", while "scored" (partial credit) and "diagnostic"
  # (observability only, e.g. ``assets_accessed`` tagged by the conflict-VLM
  # materializer) must NOT be reported as ``missing_required_side_effect``.
  return [
      check
      for check in checks
      if isinstance(check, dict)
      and check.get("type") == "effect"
      and check.get("role", "required") == "required"
  ]

def _get_flag_value(parts: list[str], flag: str) -> str | None:
  for idx, part in enumerate(parts[:-1]):
    if part == flag:
      return parts[idx + 1]
  return None

def _effect_satisfied_by_actions(
    effect_type: str,
    expected: Any,
    actions: list[str],
    task: Task | None = None,
) -> bool:
  from clasharness.evaluation.checkers.effect_checker import _tokenize_and_clean, _is_subsequence

  for action in actions:
    try:
      parts = shlex.split(action)
    except ValueError:
      continue
    if not parts:
      continue
    if effect_type == "tasks_created" and parts[:2] == ["tasks", "add"]:
      return True
    if effect_type == "emails_sent" and parts[:2] in (
        ["email", "send"],
        ["email", "reply"],
    ):
      return True
    if effect_type == "calendar_events_created" and parts[:2] == [
        "calendar",
        "add-event",
    ]:
      return True
    if effect_type == "files_created":
      if parts[:2] == ["file", "create"]:
        if isinstance(expected, dict) and expected.get("field") == "path":
          if _has_flag_value(parts, "--path", str(expected.get("value", ""))):
            return True
        else:
          return True
      elif parts[0] == "echo" and (">" in parts or ">>" in parts):
        try:
          redir_idx = parts.index(">") if ">" in parts else parts.index(">>")
          if len(parts) > redir_idx + 1:
            filepath = parts[redir_idx + 1].strip()
            if isinstance(expected, dict) and expected.get("field") == "path":
              val = str(expected.get("value", "")).strip().lstrip("/")
              if filepath.lstrip("/").endswith(val):
                return True
            else:
              return True
        except ValueError:
          pass
    if effect_type == "assets_accessed":
      import re
      if isinstance(expected, dict) and expected.get("field") == "id":
        expected_id = str(expected.get("value", "")).lower()
        norm_expected = expected_id.replace("_", "")
        clean_expected = re.sub(r"_r\d+", "", expected_id).lower().replace("_", "")

        # A. Check asset open/info commands
        if parts[0] == "asset" and len(parts) > 1 and parts[1] in ("info", "read-text", "open", "describe"):
          actual_id = _get_flag_value(parts, "--id")
          if actual_id:
            actual_id = actual_id.lower()
            def normalize_base(aid: str) -> str:
              aid = re.sub(r'_\d{8}_\d{6}', '', aid)
              aid = re.sub(r'_r\d+', '', aid)
              if aid.endswith('_chart'):
                aid = aid[:-6]
              return aid.replace("_", "")
            if normalize_base(actual_id) == normalize_base(expected_id):
              return True

            # Manifest path resolution fallbacks
            if task is not None:
              try:
                private_data = getattr(task.data, "private", {})
                manifest = private_data.get("initial_state_overrides", {}).get("asset_manifest.json", {})
                assets_list = manifest.get("assets", []) if isinstance(manifest, dict) else []
                asset_paths = {str(a.get("id")).lower(): str(a.get("path")).lower() for a in assets_list if a.get("id") and a.get("path")}
                expected_path = asset_paths.get(expected_id)
                if expected_path and actual_id and asset_paths.get(actual_id) == expected_path:
                  return True
                expected_asset_entry = next((a for a in assets_list if str(a.get("id")).lower() == expected_id), None)
                if expected_asset_entry:
                  platform_id = str(expected_asset_entry.get("platform_id", "")).lower()
                  if platform_id and platform_id == actual_id:
                    return True
              except Exception:
                pass

        # B. Check direct file_read commands (e.g. reading routing_target_latency_chart.json directly on disk)
        if parts[:2] == ["file", "read"]:
          filepath = _get_flag_value(parts, "--path")
          if filepath:
            filepath = filepath.lower()
            clean_root = expected_id.replace("ops_dashboard_", "").replace("support_portal_", "").split("_20")[0]
            clean_root = re.sub(r"_r\d+", "", clean_root).lower().replace("_", "")
            normalized_path = filepath.replace("_", "").replace("-", "")
            if clean_root in normalized_path:
              return True
      else:
        return True
    if effect_type == "messages_sent":
      if parts[:2] == ["agent", "delegate"]:
        if isinstance(expected, dict) and expected.get("field") == "message":
          instruction_val = _get_flag_value(parts, "--instruction")
          if not instruction_val:
            continue
          expected_val = str(expected.get("value", ""))
          expected_tokens = _tokenize_and_clean(expected_val)
          actual_tokens = _tokenize_and_clean(instruction_val)
          if _is_subsequence(expected_tokens, actual_tokens):
            return True
          else:
            continue
        return True
      elif parts[:2] == ["message", "send"]:
        if isinstance(expected, dict) and expected.get("field") == "message":
          message_val = _get_flag_value(parts, "--message")
          if not message_val:
            continue
          return True
        return True
      elif parts[0] == "curl":
        data_val = None
        for idx, p in enumerate(parts[:-1]):
          if p in ("-d", "--data", "--data-raw"):
            data_val = parts[idx + 1]
            break
        if data_val:
          if isinstance(expected, dict) and expected.get("field") == "message":
            # Extract text/message if JSON to verify it has message content
            try:
              import json
              payload = json.loads(data_val)
              message_text = payload.get("text") or payload.get("message")
              if not message_text:
                continue
            except Exception:
              pass
            return True
          return True
  return False

def _has_flag_value_contains(parts: list[str], flag: str, value: str) -> bool:
  for idx, part in enumerate(parts[:-1]):
    if part == flag and value.lower() in parts[idx + 1].lower():
      return True
  return False

def _has_flag_value(parts: list[str], flag: str, value: str) -> bool:
  for idx, part in enumerate(parts[:-1]):
    if part == flag and parts[idx + 1] == value:
      return True
  return False

