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

"""Agent classes + command/response helpers extracted from examples/train_and_eval.py."""
from __future__ import annotations

import base64
import json
import os
import random
import re
import shlex
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Callable

from clasharness.core.observation import Observation
from clasharness.core.task import Task
from clasharness.utils.episode_memory import (
    build_memory_summary,
    build_openai_request_messages,
    clip_text,
    is_read_only_action,
)

from train_eval._constants import (
    _ANTHROPIC_ADAPTIVE_EFFORT,
    _ANTHROPIC_BYTE_BUDGET_EDGE_LADDER,
    _ANTHROPIC_FIXED_THINKING_BUDGET,
    _ANTHROPIC_IMAGE_EDGE_FALLBACKS,
    _ANTHROPIC_MIN_EFFORT,
    _ANTHROPIC_MODEL_IMAGE_EDGE,
    _ANTHROPIC_MODELS_OMIT_TEMPERATURE,
    _ANTHROPIC_REQUEST_BYTE_BUDGET,
    _ANTHROPIC_THINKING_OUTPUT_HEADROOM,
    _EFFORT_OFF_VALUES,
    _GEMINI_MODEL_MIN_THINKING_LEVEL,
    _VALID_CONFIG_PATHS,
)
from train_eval.command_catalog import (
    _CLI_VERBS,
    _REQUIRED_FLAGS_BY_PREFIX,
    _REQUIRED_POSITIONAL_ARGS_BY_PREFIX,
    _command_catalog_text,
    _command_specs_for_task,
    _contains_placeholder_value,
    _looks_like_catalog_placeholder,
)
from train_eval.effect_checks import (
    _instruction_implies_side_effect,
    _required_effect_checks,
    _required_report_path,
)
from clasharness.evaluation.llm_retry import (
    anthropic_kwargs_carry_temperature,
    anthropic_temperature_kwargs,
)
from train_eval.image_utils import (
    _anthropic_text_from_response,
    _downscale_image_b64,
    _fit_anthropic_messages_to_byte_budget,
    _is_anthropic_image_dimension_error,
    _is_anthropic_reasoning_param_unsupported,
    _is_anthropic_temperature_unsupported,
    _normalize_anthropic_max_image_edge,
    _parse_anthropic_image_pixel_limit,
)
from train_eval.llm_errors import (
    _is_content_filter_llm_error,
    _is_deadline_error,
    _is_flaky_safety_llm_error,
    _is_gemini_thinking_level_unsupported,
    _is_retryable_llm_error,
    _retry_backoff_delay,
)
from train_eval.tool_calling import (
    ToolCall,
    chat_reasoning_text,
    assistant_tool_turn,
    parse_anthropic_tool_call,
    parse_gemini_tool_call,
    parse_openai_chat_tool_call,
    parse_openai_responses_tool_call,
    to_anthropic_messages,
    to_gemini_history_items,
    to_openai_chat_messages,
    to_openai_responses_items,
    tool_result_turn,
)
from train_eval.tool_schemas import (
    build_tool_defs,
    command_from_tool_call,
    to_anthropic_tools,
    to_gemini_tools,
    to_openai_chat_tools,
    to_openai_responses_tools,
)


def _effective_multimodal_input_mode(
    *,
    requested: str,
    provider: str,
    task: Task | None,
) -> str:
  """Return the actual multimodal mode used for this episode."""
  if requested not in {"direct", "both"}:
    return "path"
  # 'vertex-anthropic' (Claude on Vertex) accepts inline images too; the
  # OpenAI-style image_url parts are converted to Anthropic image blocks in
  # _anthropic_response_text. Plain 'anthropic' stays path-only (unchanged).
  # 'vertex-xai' (Grok on Vertex) uses the OpenAI-compatible image_url parts
  # directly, so it is enabled alongside 'openai'.
  # 'bedrock-gpt' (GPT on Bedrock) is OpenAI Responses-API compatible and also
  # consumes image_url parts directly.
  if provider not in {
      "openai",
      "gemini",
      "vertex-anthropic",
      "vertex-xai",
      "bedrock-gpt",
  }:
    return "path"
  if not _multimodal_image_parts_from_task(task, mode=requested):
    return "path"
  return requested

def _multimodal_image_parts_from_task(
    task: Task | None,
    *,
    mode: str,
    max_edge: int | None = None,
) -> list[dict[str, Any]]:
  """Build OpenAI-compatible image_url parts from round-visible assets."""
  if mode not in {"direct", "both"} or task is None:
    return []
  private = (
      getattr(task.data, "private", {}) if getattr(task, "data", None) else {}
  )
  overrides = (
      private.get("initial_state_overrides", {})
      if isinstance(private, dict)
      else {}
  )
  if not isinstance(overrides, dict):
    return []
  manifest = overrides.get("asset_manifest.json") or overrides.get(
      "assets/asset_manifest.json"
  )
  if isinstance(manifest, str):
    try:
      manifest = json.loads(manifest)
    except json.JSONDecodeError:
      manifest = {}
  assets = manifest.get(
      "assets", manifest if isinstance(manifest, list) else []
  )
  if not isinstance(assets, list):
    return []

  parts: list[dict[str, Any]] = []
  for asset in assets:
    if not isinstance(asset, dict):
      continue
    mime_type = (
        str(asset.get("mime_type") or "").strip() or "application/octet-stream"
    )
    if not mime_type.startswith("image/"):
      continue
    rel_path = str(asset.get("path") or "").strip().lstrip("/")
    if not rel_path:
      continue
    raw = overrides.get(f"assets/{rel_path}") or overrides.get(rel_path)
    if raw is None:
      continue
    if isinstance(raw, dict) and "__bytes_b64__" in raw:
      payload = base64.b64decode(str(raw["__bytes_b64__"]))
    elif isinstance(raw, str):
      payload = raw.encode("utf-8")
    else:
      payload = json.dumps(raw, ensure_ascii=False).encode("utf-8")
    b64 = base64.b64encode(payload).decode("ascii")
    if max_edge:
      # gpt-5.x bills image tokens ~linearly with pixel area (no server-side
      # downscale cap), so shrinking oversized assets here cuts cost directly.
      # _downscale_image_b64 is memoized and no-ops on already-small images.
      b64 = _downscale_image_b64(b64, mime_type, max_edge)
    data_url = f"data:{mime_type};base64,{b64}"
    parts.append({"type": "image_url", "image_url": {"url": data_url}})
  return parts

def _validate_command(action: str, task: Task | None) -> tuple[bool, str]:
  from clasharness.backend.openclaw_compat import rewrite_command
  from clasharness.backend.file_backend import _peel_content

  raw = (action or "").strip()
  decision = rewrite_command(raw, skip_incompatible=False)
  raw = decision.executed_action.strip()
  if not raw:
    return False, "empty response"
  if raw.upper() == "DONE":
    return True, ""
  # A free-text ``--content`` value (report body) may legitimately contain
  # unescaped quotes/newlines that make ``shlex.split`` raise. The executor
  # peels this value off before tokenizing (``file_backend._peel_content``), so
  # validation mirrors that: tokenize with the content replaced by a neutral
  # sentinel. Non-content commands are untouched and still reject on a genuine
  # parse error (cheap to retry), preserving the strict contract for short flags.
  head, content = _peel_content(raw)
  parse_target = f"{head} --content x" if content is not None else raw
  try:
    tokens = tuple(shlex.split(parse_target))
  except ValueError as exc:
    return False, f"shell parse error: {exc}"
  if any(_looks_like_catalog_placeholder(token) for token in tokens):
    return False, (
        f"invalid command `{raw}`. Remove catalog placeholders like `[--json]`"
        " and output only the executable command."
    )
  for spec in _command_specs_for_task(task):
    if (
        len(tokens) >= len(spec.tokens)
        and tokens[: len(spec.tokens)] == spec.tokens
    ):
      required_flags = _REQUIRED_FLAGS_BY_PREFIX.get(spec.tokens, ())
      missing = [flag for flag in required_flags if flag not in tokens]
      if missing:
        return False, (
            f"invalid command `{raw}`. Missing required flag(s):"
            f" {', '.join(missing)}"
        )
      required_positional = _REQUIRED_POSITIONAL_ARGS_BY_PREFIX.get(
          spec.tokens, 0
      )
      positional_tokens = [
          token
          for token in tokens[len(spec.tokens) :]
          if not token.startswith("--")
      ]
      if len(positional_tokens) < required_positional:
        return False, (
            f"invalid command `{raw}`. Missing required value after"
            f" `{ ' '.join(spec.tokens) }`."
        )
      placeholder = _contains_placeholder_value(tokens[len(spec.tokens) :])
      if placeholder:
        return False, (
            f"invalid command `{raw}`. Replace template placeholder"
            f" `{placeholder}` with a task-specific value."
        )
      if spec.tokens == ("openclaw", "config", "get"):
        path = positional_tokens[0] if positional_tokens else ""
        if path not in _VALID_CONFIG_PATHS:
          valid_examples = ", ".join(sorted(_VALID_CONFIG_PATHS)[:4])
          return False, (
              f"invalid command `{raw}`. Use an exact config path such as"
              f" {valid_examples}."
          )
      return True, ""
  allowed = ", ".join(spec.syntax for spec in _command_specs_for_task(task)[:8])
  return False, (
      f"invalid command `{raw}`. Use only the documented commands for this"
      f" task. Examples: {allowed}"
  )

def _quote_is_open(text: str) -> bool:
  """True when ``text`` has an unterminated quote/escape (shlex would raise).

  Used to detect a command whose quoted value (typically a multi-line
  ``--content`` report body) continues onto following physical lines, so the
  normalizer does not truncate it to the first line and orphan the opening
  quote.
  """
  try:
    shlex.split(text)
    return False
  except ValueError:
    return True

def _normalize_model_command(raw: str) -> str:
  text = (raw or "").strip()
  if not text:
    return ""
  multiline = False
  code_block_match = re.search(r"```(?:[a-zA-Z0-9_-]+)?\n(.*?)```", text, re.DOTALL)
  if code_block_match:
    text = code_block_match.group(1).strip()
    if "\n" in text and _quote_is_open(text.split("\n", 1)[0]):
      multiline = True
  else:
    raw_lines = text.splitlines()
    valid_command_prefixes = {
        "file ",
        "email ",
        "tasks ",
        "calendar ",
        "done",
        "install_skill ",
        "agent_run ",
        "msg_react ",
        "read_file ",
        "write_file ",
    }
    picked_idx = None
    for idx, line in enumerate(raw_lines):
      stripped = line.strip()
      if not stripped:
        continue
      cleaned_line = stripped.strip("`$ ")
      if any(
          cleaned_line.lower().startswith(prefix)
          for prefix in valid_command_prefixes
      ):
        picked_idx = idx
        text = cleaned_line
        break
    if picked_idx is None:
      # No command-prefixed line: fall back to the first non-empty line
      # (previous behavior; no multi-line reconstruction).
      text = next((ln.strip() for ln in raw_lines if ln.strip()), text)
    elif _quote_is_open(text):
      # The command's quoted value (e.g. a multi-line ``--content`` report)
      # continues on the following lines. Re-slice from the command line
      # through end-of-text, preserving inner newlines, so the opening quote is
      # not orphaned -- otherwise shlex.split raises "No closing quotation" and
      # the whole report is rejected (the dominant Gemini failure mode).
      tail = "\n".join(raw_lines[picked_idx:]).lstrip().lstrip("`").lstrip()
      if tail.startswith("$ "):
        tail = tail[2:].lstrip()
      text = tail
      multiline = True
  if multiline:
    # A multi-line command is already the full intended command. Skip the
    # single-line hygiene rewrites below: they strip trailing ``[...]``/``json``
    # decorations and split on `` - ``, all of which legitimately occur inside
    # report bodies (citations, bullet lists) and would corrupt the content.
    text = text.strip()
    if text.upper() == "DONE":
      return "DONE"
    return text
  text = text.strip("`")
  if text.startswith("$ "):
    text = text[2:].strip()
  while re.search(r"\s+\[[^\]]+\]\s*$", text):
    text = re.sub(r"\s+\[[^\]]+\]\s*$", "", text).strip()
  text = re.sub(
      r"[。【】]+\s*(?:json(?:_[a-zA-Z0-9]+)?(?:\s+to=\w+.*)?|analysis(?:\s+to=\w+.*)?)$",
      "",
      text,
      flags=re.IGNORECASE,
  ).strip()
  if text.upper() == "DONE":
    return "DONE"
  if " - " in text and not any(ch in text for ch in "\"'"):
    text = re.split(r"\s+-\s+", text, maxsplit=1)[0].strip()
    if text.upper() == "DONE":
      return "DONE"
  try:
    shlex.split(text)
    return text
  except ValueError:
    text = re.split(r"\s+-\s+", text, maxsplit=1)[0].strip()
  text = re.sub(
      r"[。【】]+\s*(?:json(?:_[a-zA-Z0-9]+)?(?:\s+to=\w+.*)?|analysis(?:\s+to=\w+.*)?)$",
      "",
      text,
      flags=re.IGNORECASE,
  ).strip()
  if text.upper() == "DONE":
    return "DONE"
  return text

def _has_required_artifact(task: Task | None) -> bool:
  """True when the task has a concrete required side-effect the gate can name.

  Used to keep the early-DONE gate conservative: fire only when a real required
  artifact backs it (a required report path, or a required effect check), never
  on a bare instruction-keyword guess. This removes false positives where an
  incidental word like "report"/"add"/"keep" tripped the gate on a task that
  needs no side effect.
  """
  return bool(_required_report_path(task)) or bool(_required_effect_checks(task))

def _gemini_extract_text(response: Any) -> str:
  """Best-effort text extraction from a Gemini response.

  ``response.text`` is a convenience accessor that raises when the only
  candidate was blocked (SAFETY/RECITATION) or carries no text parts. We must
  not let that surface as a network error, so fall back to manually walking the
  candidate parts and return "" for a genuinely empty/blocked reply.
  """
  try:
    text = response.text
    if text:
      return str(text)
  except Exception:  # pragma: no cover - SDK raises on blocked/empty candidates
    pass
  collected: list[str] = []
  try:
    for candidate in getattr(response, "candidates", None) or []:
      content = getattr(candidate, "content", None)
      for part in getattr(content, "parts", None) or []:
        part_text = getattr(part, "text", None)
        if part_text:
          collected.append(str(part_text))
  except Exception:  # pragma: no cover - defensive against SDK shape changes
    return ""
  return "".join(collected)

def _gemini_response_diagnostics(response: Any) -> str:
  """Render finish_reason + token usage for observability.

  Empty replies are indistinguishable at the text layer; the finish_reason
  (STOP vs MAX_TOKENS vs SAFETY/RECITATION) and usage counts are what let us
  tell a benign sampling blank apart from a truncation/safety block. Returns a
  leading-space suffix so it can be appended to the ``api:`` debug line, or ""
  when nothing is available.
  """
  fields: list[str] = []
  try:
    candidates = getattr(response, "candidates", None) or []
    if candidates:
      finish_reason = getattr(candidates[0], "finish_reason", None)
      if finish_reason is not None:
        fields.append(
            f"finish_reason={getattr(finish_reason, 'name', finish_reason)}"
        )
  except Exception:  # pragma: no cover - defensive
    pass
  try:
    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
      prompt = getattr(usage, "prompt_token_count", None)
      candidates_tokens = getattr(usage, "candidates_token_count", None)
      total = getattr(usage, "total_token_count", None)
      fields.append(
          f"usage=(prompt={prompt},candidates={candidates_tokens},total={total})"
      )
  except Exception:  # pragma: no cover - defensive
    pass
  return (" " + " ".join(fields)) if fields else ""

def _gemini_extract_parts(response: Any) -> tuple[str, str]:
  """Split a Gemini response into ``(answer_text, thinking_text)``.

  With ``ThinkingConfig(include_thoughts=True)`` the model's thought summaries
  arrive as ordinary text parts flagged ``part.thought == True``. They MUST be
  kept out of the answer text (which feeds command parsing) and routed to the
  thinking channel instead. Returns ``("", "")`` for a genuinely empty/blocked
  reply. Only used on the capture path; the default path keeps
  ``_gemini_extract_text`` so non-capture runs are byte-for-byte unchanged.
  """
  answer: list[str] = []
  thinking: list[str] = []
  try:
    for candidate in getattr(response, "candidates", None) or []:
      content = getattr(candidate, "content", None)
      for part in getattr(content, "parts", None) or []:
        part_text = getattr(part, "text", None)
        if not part_text:
          continue
        if getattr(part, "thought", False):
          thinking.append(str(part_text))
        else:
          answer.append(str(part_text))
  except Exception:  # pragma: no cover - defensive against SDK shape changes
    return "", ""
  return "".join(answer), "".join(thinking)

def _gemini_response_meta(response: Any) -> dict[str, Any]:
  """Structured finish_reason / safety / block / usage for the interaction log.

  Complements the human-readable ``api:`` debug line with machine-parseable
  fields that let us tell a benign blank apart from a truncation, a safety
  block, or a MALFORMED_FUNCTION_CALL when diagnosing generation instability.
  """
  meta: dict[str, Any] = {
      "finish_reason": None,
      "finish_message": None,
      "safety_ratings": [],
      "block_reason": None,
      "usage": {},
  }
  try:
    candidates = getattr(response, "candidates", None) or []
    if candidates:
      cand = candidates[0]
      finish_reason = getattr(cand, "finish_reason", None)
      if finish_reason is not None:
        meta["finish_reason"] = getattr(finish_reason, "name", finish_reason)
      meta["finish_message"] = getattr(cand, "finish_message", None)
      ratings = getattr(cand, "safety_ratings", None) or []
      serialized = []
      for rating in ratings:
        category = getattr(rating, "category", None)
        probability = getattr(rating, "probability", None)
        serialized.append({
            "category": getattr(category, "name", category),
            "probability": getattr(probability, "name", probability),
            "blocked": getattr(rating, "blocked", None),
        })
      meta["safety_ratings"] = serialized
  except Exception:  # pragma: no cover - defensive
    pass
  try:
    feedback = getattr(response, "prompt_feedback", None)
    if feedback is not None:
      block_reason = getattr(feedback, "block_reason", None)
      if block_reason is not None:
        meta["block_reason"] = getattr(block_reason, "name", block_reason)
  except Exception:  # pragma: no cover - defensive
    pass
  try:
    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
      meta["usage"] = {
          "prompt": getattr(usage, "prompt_token_count", None),
          "candidates": getattr(usage, "candidates_token_count", None),
          "total": getattr(usage, "total_token_count", None),
          "thoughts": getattr(usage, "thoughts_token_count", None),
      }
  except Exception:  # pragma: no cover - defensive
    pass
  return meta

def _gemini_text_from_response(response: Any) -> str:
  """Visible assistant text from a Gemini response, best effort.

  Used by the no-tool-call salvage path in ``_act_native`` to keep a text-only
  turn in the history instead of discarding it. Thought parts are skipped: they
  are recorded separately as ``thinking_text`` and must not be replayed as
  ordinary assistant text. Never raises -- a shape we do not recognise yields
  "" and the caller simply falls back to the bare nudge.
  """
  parts: list[str] = []
  try:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
      return ""
    content = getattr(candidates[0], "content", None)
    for part in getattr(content, "parts", None) or []:
      if getattr(part, "thought", False):
        continue
      text = getattr(part, "text", None)
      if text:
        parts.append(str(text))
  except Exception:  # pragma: no cover - defensive
    return ""
  return "".join(parts).strip()

_GEMINI_MEDIA_RESOLUTIONS = ("low", "medium", "high", "ultra_high")

def _normalize_gemini_media_resolution(value: str | None) -> str | None:
  """Validate a --gemini-media-resolution value; None/"" means "leave unset".

  Raises on an unknown level rather than silently ignoring it: a typo here
  would otherwise look like a successful run at the default budget, which is
  exactly the kind of silent-misconfiguration bug this flag exists to expose.
  """
  if value is None:
    return None
  text = str(value).strip().lower()
  if not text or text == "unspecified":
    return None
  if text not in _GEMINI_MEDIA_RESOLUTIONS:
    raise ValueError(
        f"Unsupported gemini media resolution: {value!r}. Choose from:"
        f" {', '.join(_GEMINI_MEDIA_RESOLUTIONS)}"
    )
  return text

def _gemini_part_media_resolution(types: Any, level: str | None) -> Any:
  """Build a ``types.PartMediaResolution`` for ``level``, or None.

  Returns None when unset or when the installed google-genai SDK is too old to
  expose the per-part enum, so an older SDK degrades to the server default
  instead of crashing the run.
  """
  if not level:
    return None
  enum_cls = getattr(types, "PartMediaResolutionLevel", None)
  wrapper_cls = getattr(types, "PartMediaResolution", None)
  if enum_cls is None or wrapper_cls is None:  # pragma: no cover - old SDK
    return None
  member = getattr(enum_cls, f"MEDIA_RESOLUTION_{level.upper()}", None)
  if member is None:  # pragma: no cover - old SDK
    return None
  return wrapper_cls(level=member)

def _openai_response_meta(payload: dict[str, Any]) -> dict[str, Any]:
  """finish_reason / usage / reasoning text from an OpenAI-compatible payload.

  Mirrors ``_gemini_response_meta`` so the OpenAI/vertex-xai providers populate
  the same interaction-record fields Gemini already does. Handles both wire
  shapes:

  * Chat Completions: ``choices[0].message`` (+ ``reasoning_content``) and a
    top-level ``usage`` with ``prompt_tokens`` / ``completion_tokens`` plus
    ``prompt_tokens_details.cached_tokens`` and
    ``completion_tokens_details.reasoning_tokens``.
  * Responses API: ``output[]`` items (``reasoning`` items carry a ``summary``)
    and a ``usage`` with ``input_tokens`` / ``output_tokens`` plus
    ``input_tokens_details.cached_tokens`` and
    ``output_tokens_details.reasoning_tokens``.

  The raw ``usage`` dict is preserved verbatim so the exact cached / reasoning
  token counts survive for cost accounting. ``thinking_text`` collects whatever
  reasoning summary the endpoint returned; OpenAI only emits one when the model
  chooses to, so it is frequently empty even when reasoning tokens were billed
  (the count still lives in ``usage``).
  """
  meta: dict[str, Any] = {
      "finish_reason": None,
      "finish_message": None,
      "safety_ratings": [],
      "block_reason": None,
      "usage": {},
      "thinking_text": "",
  }
  try:
    meta["usage"] = payload.get("usage") or {}
  except Exception:  # pragma: no cover - defensive
    pass
  # Responses API shape.
  if payload.get("object") == "response" or "output" in payload:
    try:
      meta["finish_reason"] = payload.get("status")
      incomplete = payload.get("incomplete_details")
      if isinstance(incomplete, dict):
        meta["finish_message"] = incomplete.get("reason")
      thinking: list[str] = []
      for item in payload.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "reasoning":
          continue
        for chunk in (item.get("summary") or []) + (item.get("content") or []):
          if isinstance(chunk, dict) and chunk.get("text"):
            thinking.append(str(chunk["text"]))
      meta["thinking_text"] = "".join(thinking)
    except Exception:  # pragma: no cover - defensive
      pass
    return meta
  # Chat Completions shape.
  try:
    choices = payload.get("choices") or []
    if choices:
      choice = choices[0] or {}
      meta["finish_reason"] = choice.get("finish_reason")
      message = choice.get("message") or {}
      meta["thinking_text"] = chat_reasoning_text(message) or ""
  except Exception:  # pragma: no cover - defensive
    pass
  return meta

def _openai_text_from_payload(payload: dict[str, Any]) -> str:
  """Visible assistant text from an OpenAI-compatible payload, best effort.

  A pure, non-raising counterpart to ``_extract_openai_message_text`` for the
  no-tool-call salvage path in ``_act_native``: it never issues the followup
  request that method can, and returns "" instead of raising when the payload
  carries no text. Reasoning items are skipped -- only ``message`` output is
  visible assistant text.
  """
  parts: list[str] = []
  try:
    if payload.get("object") == "response" or "output" in payload:
      for item in payload.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
          continue
        for chunk in item.get("content") or []:
          if (
              isinstance(chunk, dict)
              and chunk.get("type") in {"output_text", "text"}
              and chunk.get("text")
          ):
            parts.append(str(chunk["text"]))
    else:
      choices = payload.get("choices") or []
      message = (choices[0] or {}).get("message") or {} if choices else {}
      content = message.get("content")
      if isinstance(content, str):
        parts.append(content)
      elif isinstance(content, list):
        for chunk in content:
          if isinstance(chunk, dict) and chunk.get("text"):
            parts.append(str(chunk["text"]))
  except Exception:  # pragma: no cover - defensive
    return ""
  return "".join(parts).strip()

def _anthropic_response_meta(response: Any) -> dict[str, Any]:
  """stop_reason / usage / thinking text from an Anthropic ``Message``.

  Extended thinking IS requested on the native-tool path (see
  ``_anthropic_create``: ``thinking={"type":"adaptive"}`` +
  ``output_config.effort``), so responses routinely carry ``thinking`` /
  ``redacted_thinking`` blocks whose tokens are billed inside ``output_tokens``.
  The whole ``usage`` object is preserved verbatim -- crucially including
  ``output_tokens_details.thinking_tokens`` -- so the visible/thinking split can
  be recovered downstream; folding thinking into an opaque ``output_tokens``
  total otherwise makes Claude's ``output_tokens`` look inflated next to
  providers that report a separate reasoning count. NOTE: on Vertex the
  ``thinking`` blocks arrive with an empty ``.thinking`` text (signature only),
  so ``thinking_text`` is usually blank even when thinking occurred -- rely on
  ``usage.output_tokens_details.thinking_tokens`` for that signal, not the text.
  Tolerates both SDK-object and dict-shaped responses (as in the unit tests).
  """
  meta: dict[str, Any] = {
      "finish_reason": None,
      "finish_message": None,
      "safety_ratings": [],
      "block_reason": None,
      "usage": {},
      "thinking_text": "",
  }

  def _get(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
      return obj.get(key)
    return getattr(obj, key, None)

  try:
    meta["finish_reason"] = _get(response, "stop_reason")
  except Exception:  # pragma: no cover - defensive
    pass
  try:
    usage = _get(response, "usage")
    if isinstance(usage, dict):
      meta["usage"] = dict(usage)
    elif usage is not None:
      # Prefer the SDK's own serializer so newer usage fields survive without a
      # code change here -- most importantly ``output_tokens_details`` (whose
      # ``thinking_tokens`` records how many billed output tokens were internal
      # reasoning). A hand-picked allowlist silently dropped that, which made
      # Claude's reasoning cost invisible while it was still billed inside
      # ``output_tokens``.
      dumped = None
      for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(usage, attr, None)
        if callable(fn):
          try:
            dumped = fn()
          except Exception:  # pragma: no cover - defensive
            dumped = None
          if isinstance(dumped, dict):
            break
          dumped = None
      if isinstance(dumped, dict):
        meta["usage"] = dumped
      else:
        # Fallback: read the known scalar fields plus the reasoning detail so
        # the split is preserved even when the object cannot be serialized.
        details = getattr(usage, "output_tokens_details", None)
        if details is not None and not isinstance(details, dict):
          details = {
              "thinking_tokens": getattr(details, "thinking_tokens", None),
              "reasoning_tokens": getattr(details, "reasoning_tokens", None),
          }
        meta["usage"] = {
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
            "cache_read_input_tokens": getattr(
                usage, "cache_read_input_tokens", None
            ),
            "cache_creation_input_tokens": getattr(
                usage, "cache_creation_input_tokens", None
            ),
            "output_tokens_details": details,
        }
  except Exception:  # pragma: no cover - defensive
    pass
  try:
    content = _get(response, "content")
    thinking: list[str] = []
    for block in content or []:
      btype = _get(block, "type")
      if btype not in {"thinking", "redacted_thinking"}:
        continue
      # `thinking` blocks carry `.thinking`; redacted blocks are opaque (`data`).
      text = _get(block, "thinking") or _get(block, "text")
      if text:
        thinking.append(str(text))
    meta["thinking_text"] = "".join(thinking)
  except Exception:  # pragma: no cover - defensive
    pass
  return meta

# Where each provider puts its token counts, and -- the part that bites -- whether
# the reasoning count is already *inside* the visible output count. The providers
# disagree, so ``visible = output - reasoning`` is correct for two of them and
# wrong for the third. Verified against the logged corpora:
#
#   gemini            total == prompt + candidates + thoughts   (30,092/30,092)
#                     -> ``candidates`` EXCLUDES reasoning; visible IS candidates
#   openai-shaped     total == input + output, output >= reasoning  (55,570/55,570)
#                     -> ``output_tokens`` INCLUDES reasoning
#   vertex-anthropic  ``output_tokens`` INCLUDES ``thinking_tokens``; both are
#                     billed at the output rate, so the *cost* is right either
#                     way and only the visible/reasoning split is at stake
#
# ``vertex-xai`` is assumed to follow the OpenAI-compatible (inclusive)
# convention because it speaks that wire format, but no corpus has been logged
# for it, so that one branch is unverified rather than measured.
#
# Reasoning is reported under four different key paths (``thoughts``,
# ``completion_tokens_details.reasoning_tokens``,
# ``output_tokens_details.reasoning_tokens``,
# ``output_tokens_details.thinking_tokens``). Every consumer that hand-rolled
# that union got Anthropic wrong and silently reported zero reasoning for
# Claude. Normalising once, here, is what stops that recurring.
_USAGE_REASONING_PATHS: tuple[tuple[str, ...], ...] = (
    ("thoughts",),
    ("completion_tokens_details", "reasoning_tokens"),
    ("output_tokens_details", "reasoning_tokens"),
    ("output_tokens_details", "thinking_tokens"),
)

def _usage_dig(usage: dict[str, Any], path: tuple[str, ...]) -> Any:
  """``usage`` value at a dotted path, or None if any hop is missing."""
  cur: Any = usage
  for key in path:
    if not isinstance(cur, dict):
      return None
    cur = cur.get(key)
  return cur

def _normalized_usage(provider: str, usage: Any) -> dict[str, Any]:
  """Provider-agnostic token split, recorded next to the verbatim ``usage``.

  Returns ``input`` / ``output_visible`` / ``output_reasoning`` /
  ``output_billed``. Downstream cost and behaviour reporting should read this
  instead of reaching into provider-specific keys.

  ``output_reasoning`` is ``None`` -- deliberately not ``0`` -- when the
  provider did not report a reasoning channel at all. "Not captured" and
  "captured, genuinely zero" are different claims, and collapsing them to 0 is
  exactly how Claude's reasoning spend was mistaken for zero thinking for so
  long. Callers must treat None as unknown.
  """
  out: dict[str, Any] = {
      "input": None,
      "output_visible": None,
      "output_reasoning": None,
      "output_billed": None,
  }
  if not isinstance(usage, dict) or not usage:
    return out

  reasoning = None
  for path in _USAGE_REASONING_PATHS:
    val = _usage_dig(usage, path)
    if isinstance(val, (int, float)):
      reasoning = int(val)
      break

  if provider == "gemini":
    # ``candidates`` is already reasoning-free; do not subtract.
    visible = usage.get("candidates")
    out["input"] = usage.get("prompt")
    out["output_visible"] = visible
    out["output_reasoning"] = reasoning
    if isinstance(visible, int):
      out["output_billed"] = visible + (reasoning or 0)
    return out

  # OpenAI-shaped and Anthropic: ``output_tokens`` is the billed total and
  # already contains the reasoning tokens.
  billed = usage.get("output_tokens")
  if billed is None:
    billed = usage.get("completion_tokens")
  out["input"] = usage.get("input_tokens", usage.get("prompt_tokens"))
  out["output_billed"] = billed
  out["output_reasoning"] = reasoning
  if isinstance(billed, int):
    out["output_visible"] = billed - (reasoning or 0)
  return out

def _record_dropped_parallel_calls(
    meta: dict[str, Any] | None,
    call: Any,
    debug: list[str],
) -> None:
  """Record parallel tool calls the one-action-per-step loop cannot execute.

  Every provider may answer a forced tool-call turn with several calls at once.
  The benchmark runs exactly one action per step, so the extras are dropped and
  never replayed into history. This makes that observable -- and symmetric
  across providers, so per-provider parallel-call rates are comparable -- by
  putting the extra call names on the interaction record and in the attempt
  debug string. It changes no behaviour.
  """
  dropped = list(call.dropped_parallel_calls) if call else []
  if meta is not None:
    meta["dropped_parallel_calls"] = dropped
  if dropped:
    debug.append(f"api: dropped parallel tool calls: {dropped!r}")

def _sha1(text: str) -> str:
  import hashlib
  return hashlib.sha1((text or "").encode("utf-8", "replace")).hexdigest()

def _sha1_bytes(data: bytes) -> str:
  import hashlib
  return hashlib.sha1(data).hexdigest()

def _collapse_response_text_parts(text_parts: list[str]) -> tuple[str, int]:
  cleaned_parts = [str(part) for part in text_parts if str(part).strip()]
  if not cleaned_parts:
    return "", 0

  deduped_parts: list[str] = []
  previous_key: str | None = None
  removed_duplicates = 0
  for part in cleaned_parts:
    key = _normalize_model_command(part) or part.strip()
    if deduped_parts and key and key == previous_key:
      removed_duplicates += 1
      continue
    deduped_parts.append(part)
    previous_key = key

  collapsed = "".join(deduped_parts).strip()
  if collapsed:
    return collapsed, removed_duplicates
  return "".join(cleaned_parts).strip(), removed_duplicates

def _extract_command_from_reasoning_text(
    reasoning_text: str, task: Task | None
) -> str:
  text = (reasoning_text or "").strip()
  if not text:
    return ""

  candidates: list[str] = []
  lines = [line.strip() for line in text.splitlines() if line.strip()]
  prefixes = [" ".join(spec.tokens) for spec in _command_specs_for_task(task)]

  for line in lines:
    cleaned = re.sub(r"^[\-\*\d\.\)\s]+", "", line).strip()
    if cleaned:
      candidates.append(cleaned)
    candidates.extend(re.findall(r"`([^`]+)`", line))
    for prefix in prefixes:
      idx = line.find(prefix)
      if idx >= 0:
        candidates.append(line[idx:].strip())

  for candidate in candidates:
    normalized = _normalize_model_command(candidate)
    ok, _ = _validate_command(normalized, task)
    if ok and normalized.upper() != "DONE":
      return normalized
  return ""

# The single canonical --llm-reasoning-effort vocabulary. Users control
# reasoning with ONE ladder regardless of provider; each provider maps every
# rung onto its own wire vocabulary via the per-provider tables below. This
# keeps the flag's surface identical across vendors instead of leaking each
# API's idiosyncratic names (Gemini's MINIMAL..HIGH, OpenAI's none..xhigh,
# Anthropic's low..high).
#
# Ordered low->high. Every per-provider table below MUST be total over this
# tuple so a validated effort can never miss a key; construction only has to
# reject values that are not on this ladder at all.
_REASONING_EFFORT_LADDER = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)

# Gemini expresses reasoning effort as ThinkingConfig.thinking_level, an enum
# with exactly MINIMAL/LOW/MEDIUM/HIGH (google-genai 2.10.0). It has no true
# "off" ('none'/'minimal' collapse to MINIMAL = "think as little as possible")
# and no tier above HIGH ('xhigh'/'max' cap at HIGH).
_GEMINI_THINKING_LEVELS = {
    "none": "MINIMAL",
    "minimal": "MINIMAL",
    "low": "LOW",
    "medium": "MEDIUM",
    "high": "HIGH",
    "xhigh": "HIGH",
    "max": "HIGH",
}

# Ordered low->high so a model that rejects a level (e.g. Gemini 3 Pro rejects
# MINIMAL) can be stepped UP to the next one it accepts. See
# _GEMINI_MODEL_MIN_THINKING_LEVEL and LLMAgent._escalate_gemini_thinking_floor.
_GEMINI_THINKING_LEVEL_LADDER = ("MINIMAL", "LOW", "MEDIUM", "HIGH")

# Reverse of _GEMINI_THINKING_LEVELS, for reporting the level actually sent in
# the shared vocabulary. Both 'none' and 'minimal' send MINIMAL, so either is
# reported as 'minimal' (Gemini has no true "off").
_GEMINI_LEVEL_TO_EFFORT = {
    "MINIMAL": "minimal",
    "LOW": "low",
    "MEDIUM": "medium",
    "HIGH": "high",
}

# OpenAI Responses ``reasoning.effort`` accepts none/low/medium/high/xhigh (the
# Bedrock GPT proxy rejects 'minimal' and 'max' with an HTTP 400). 'none' is a
# real off and is preserved; 'minimal' maps to the nearest real tier 'low';
# 'max' caps at the top tier 'xhigh'.
_OPENAI_EFFORT_LEVELS = {
    "none": "none",
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "xhigh",
}

# Anthropic ``output_config.effort`` accepts low/medium/high only: there is no
# tier above high, and reasoning is turned off via ``thinking.type`` (the
# adaptive/budget/plain ladder), NOT via this field. 'none' is preserved as the
# sentinel the send site has always forwarded; every other rung clamps into
# low/medium/high.
_ANTHROPIC_EFFORT_LEVELS = {
    "none": "none",
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}

# Kimi K3 reads a top-level Chat Completions ``reasoning_effort`` and renders it
# into an injected 'thinking-effort' system message. It accepts EXACTLY
# low/high/max and answers anything else with HTTP 500 ("Unsupported
# thinking_effort='medium'; supported values are ['high', 'low', 'max']"), so
# unlike the maps above -- where an unsupported rung merely clamps -- this one
# must never emit 'medium'. There is no "off" either: K3 always thinks, so
# 'none'/'minimal' fold onto the lowest real tier.
#
# NOTE this map STRETCHES where the others CLAMP: 'medium'->high, 'high'->max.
# That is deliberate. K3 has exactly three tiers and the panel runs exactly
# three levels, so an order-preserving bijection keeps all three cells on
# DISTINCT tiers; clamping ('medium'->high, 'high'->high) would collapse two
# panel cells onto one tier and make the study degenerate. It also puts the
# panel's top cell on 'max', the tier K3's published benchmarks use. The
# consequence -- a run recorded at harness 'medium' went out at K3 'high' -- is
# what ``reasoning_effort_effective`` exists to state, and it does.
_KIMI_EFFORT_LEVELS = {
    "none": "low",
    "minimal": "low",
    "low": "low",
    "medium": "high",
    "high": "max",
    "xhigh": "max",
    "max": "max",
}


def _is_kimi_k3_model(model: str | None) -> bool:
  """True for Kimi K3, however the served model id spells it.

  Matches 'moonshotai/Kimi-K3', 'kimi_k3' and local snapshot paths alike by
  dropping separators before the test, so re-serving the same weights under a
  different id cannot silently turn the effort flag back into a no-op.
  """
  return "kimik3" in re.sub(r"[^a-z0-9]", "", (model or "").lower())


def _gemini_clamp_thinking_level(level: str, floor: str | None) -> str:
  """Raise ``level`` to ``floor`` when the model cannot think that low.

  Both are ThinkingConfig enum names; returns whichever is higher on the
  MINIMAL<LOW<MEDIUM<HIGH ladder. ``floor`` None means no clamp.
  """
  if floor is None:
    return level
  order = _GEMINI_THINKING_LEVEL_LADDER
  return order[max(order.index(level), order.index(floor))]


class BaseAgent(ABC):
  """Abstract agent interface.

  On each step the agent receives an :class:`Observation` and returns a
  single CLI command string.  The episode loop calls :meth:`reset` at the
  start of every task and :meth:`act` at every step.
  """

  name: str = "base"

  def reset(self, task_instruction: str, task: Task | None = None) -> None:
    """Called once at the start of each episode."""

  @abstractmethod
  def act(self, observation: Observation) -> str:
    """Return the next CLI command given the current observation."""

  def should_stop(self) -> bool:
    """Whether the episode runner should stop before sending another command."""
    return False

  def debug_last_reply(self) -> str | None:
    """Optional debug string describing the latest model/agent reply."""
    return None

  def debug_reply_attempts(self) -> list[str]:
    """Optional list of raw/parsed reply attempts from the latest act() call."""
    return []

  def debug_run_config(self) -> dict[str, str]:
    """Optional metadata to stamp into logs and saved reports."""
    return {}

class ExpertAgent(BaseAgent):
  """Replays the ground-truth solution commands.

  This is the oracle upper bound — it always succeeds when evaluation checks
  are correct.  Use it to verify the environment and collect demonstration
  trajectories for behavior cloning.
  """

  name = "expert"

  def __init__(self) -> None:
    self._commands: list[str] = []
    self._idx: int = 0

  def reset(self, task_instruction: str, task: Task | None = None) -> None:
    if task and task.ground_truth:
      self._commands = list(task.ground_truth.solution_commands)
    else:
      self._commands = []
    self._idx = 0

  def act(self, observation: Observation) -> str:
    if self._idx < len(self._commands):
      cmd = self._commands[self._idx]
      self._idx += 1
      return cmd
    # Fallback no-op if caller does not check should_stop().
    return "tasks list"

  def should_stop(self) -> bool:
    return self._idx >= len(self._commands)

class RandomAgent(BaseAgent):
  """Issues random CLI commands.  Lower-bound baseline."""

  name = "random"

  def __init__(self, seed: int = 42) -> None:
    self._rng = random.Random(seed)
    self._prefixes = list(_CLI_VERBS.keys())

  def act(self, observation: Observation) -> str:
    prefix = self._rng.choice(self._prefixes)
    verb = self._rng.choice(_CLI_VERBS[prefix])
    return f"{prefix} {verb}"

class RuleAgent(BaseAgent):
  """Simple heuristic agent that parses the task instruction.

  Demonstrates how to build a domain-specific rule-based baseline without
  using an LLM.  It detects keywords in the instruction and maps them to
  CLI commands.
  """

  name = "rule"

  _KEYWORD_MAP: list[tuple[list[str], str]] = [
      # (keywords, command_template)
      (["calendar", "event", "add", "block"], "calendar add-event"),
      (["calendar", "reschedule", "move", "update"], "calendar update-event"),
      (["calendar", "list", "schedule", "show"], "calendar list"),
      (["email", "send", "draft"], "email send"),
      (["email", "reply", "respond"], "email reply"),
      (["email", "move", "folder"], "email move"),
      (["email", "mark", "read", "seen"], "email mark-read"),
      (["email", "search", "find", "inbox"], "email search"),
      (["weather", "forecast", "week"], "weather forecast"),
      (["weather", "alert"], "weather alerts"),
      (["weather", "get", "check"], "weather get"),
      (["file", "create", "save", "write"], "file create"),
      (["file", "move", "rename"], "file move"),
      (["file", "append", "add"], "file append"),
      (["file", "read", "show"], "file read"),
      (["task", "add", "track", "item"], "tasks add"),
      (["task", "complete", "done", "finish"], "tasks complete"),
      (["task", "search", "find"], "tasks search"),
      (["task", "list", "show", "plate"], "tasks list"),
  ]

  def __init__(self) -> None:
    self._instruction = ""
    self._step = 0

  def reset(self, task_instruction: str, task: Task | None = None) -> None:
    self._instruction = task_instruction.lower()
    self._step = 0

  def act(self, observation: Observation) -> str:
    self._step += 1
    if self._step == 1:
      # On the first step, pick the best-matching command
      words = set(self._instruction.split())
      best_cmd = "tasks list"
      best_score = 0
      for keywords, cmd in self._KEYWORD_MAP:
        score = sum(1 for kw in keywords if kw in words)
        if score > best_score:
          best_score = score
          best_cmd = cmd
      return best_cmd
    # Subsequent steps: just list to avoid errors
    return "tasks list"


class _EmptyProviderReply(RuntimeError):
  """Sentinel: the provider returned a successful (HTTP 200) but empty reply.

  Raised inside the per-provider request loops when the call succeeded yet
  carried no usable content (Anthropic ``stop_reason=refusal`` with no text /
  tool_use block, Gemini ``finish_reason=SAFETY`` or an otherwise empty
  candidate, or an OpenAI-compatible response with empty message content). It is
  a retryable condition consumed from the dedicated ``_empty_reply_retries``
  budget (NOT the general request-retry budget); once that budget is exhausted
  the caller converts it back to an empty reply so the existing stall handling
  coerces DONE and records ``generation_stall_empty_reply`` unchanged.
  """


class LLMAgent(BaseAgent):
  """Multi-turn LLM agent.

  Each step appends the observation to a running conversation and asks the
  model for the next CLI command.  The model is instructed to output a
  single bare command or the literal token ``DONE`` when finished.

  Requirements
  ------------
  Anthropic:
  - ``pip install anthropic``
  - ``ANTHROPIC_API_KEY`` environment variable set

  OpenAI-compatible:
  - No extra package required
  - API compatible with ``POST /v1/chat/completions``
  - ``OPENAI_API_KEY`` or configured key env var set

  Parameters
  ----------
  provider:
      ``anthropic`` or ``openai``. ``openai`` also covers OpenAI-compatible
      proxies and Azure-style gateways that expose the chat completions API.
  model:
      Model ID for the selected provider.
  base_url:
      Optional base URL for OpenAI-compatible providers. Defaults to
      ``https://api.openai.com/v1`` when ``provider=openai``.
  api_key_env:
      Environment variable holding the API key. Defaults to
      ``ANTHROPIC_API_KEY`` or ``OPENAI_API_KEY`` based on provider.
  max_steps_hint:
      Soft cap communicated to the model in the system prompt.
  """

  name = "llm"

  BASE_SYSTEM_PROMPT = """\
You are selecting the next command for a benchmark task.

Response format:
- Return one shell command line.
- If the task is complete, return DONE.
- Do not add commentary or formatting.

Execution guidance:
- Choose commands from the task-specific command catalog.
- After each step, use the command output to decide the next action.
- If a step fails, choose a different command from the catalog.
- Do not return DONE until the task's requested state changes have actually happened.
- You MUST proactively list visible assets via `asset list`. If there are any newly visible or relevant visual evidence assets (charts, diagrams, images) introduced in the current round, you MUST inspect them by calling `asset open` or `asset info` on them immediately, to verify the evidence yourself before delegating tasks or finishing.
- If you have delegated tasks to subagents or are waiting for asynchronous events to occur (e.g. file updates, messages from other agents), you must use the `wait <seconds>` command to progress virtual time. Events will NOT trigger and subagents will NOT complete unless you progress time by waiting.
- When subagents complete their tasks, they often generate new files, write reports, or update logs in the workspace. You MUST read and verify these generated files/reports (using `file read` or `asset read-text` or `asset open`) to inspect the subagents' outputs and complete your analysis before returning DONE.
"""

  # Native function/tool-calling counterpart of BASE_SYSTEM_PROMPT. The
  # execution guidance is intentionally kept semantically identical (so the
  # native-vs-text ablation isolates the calling surface, not the coaching); only
  # the response-format section and the command references are re-expressed as
  # tool calls. The task's command surface is conveyed by the provider ``tools``
  # payload, so no textual command catalog is appended in native mode.
  NATIVE_SYSTEM_PROMPT = """\
You are selecting the next action for a benchmark task.

Response format:
- Call exactly one tool per step to act in the workspace.
- Uppercase words like TIMEZONE, LOCATION, TITLE, NAME, and SCHEDULE in tool descriptions are templates, not literal values.
- Call the `done` tool when, and only when, the task is complete.
- Do not add commentary; act only through tool calls.

Execution guidance:
- Choose from the task-specific tools provided.
- After each step, use the tool result to decide the next action.
- If a step fails, call a different tool.
- Do not call the `done` tool until the task's requested state changes have actually happened.
- You MUST proactively list visible assets via the `asset_list` tool. If there are any newly visible or relevant visual evidence assets (charts, diagrams, images) introduced in the current round, you MUST inspect them by calling the `asset_open` or `asset_info` tool on them immediately, to verify the evidence yourself before delegating tasks or finishing.
- If you have delegated tasks to subagents or are waiting for asynchronous events to occur (e.g. file updates, messages from other agents), you must call the `wait` tool (with a number of seconds) to progress virtual time. Events will NOT trigger and subagents will NOT complete unless you progress time by waiting.
- When subagents complete their tasks, they often generate new files, write reports, or update logs in the workspace. You MUST read and verify these generated files/reports (using the `file_read` or `asset_read_text` or `asset_open` tool) to inspect the subagents' outputs and complete your analysis before calling the `done` tool.
"""

  _LITELLM_PROXY_MODEL_PREFIXES = ("claude-",)
  _LITELLM_PROXY_BASE_URL_ENV_CANDIDATES = (
      "LITELLM_PROXY_BASE_URL",
      "BEDROCK_LITELLM_BASE_URL",
  )
  _LITELLM_PROXY_KEY_ENV_CANDIDATES = (
      "LITELLM_PROXY_KEY",
      "BEDROCK_LITELLM_KEY",
  )
  _LITELLM_PROXY_DEFAULT_BASE_URL = "http://127.0.0.1:4000/v1"

  def __init__(
      self,
      provider: str = "anthropic",
      model: str = "claude-haiku-4-5-20251001",
      base_url: str | None = None,
      api_key_env: str | None = None,
      history_mode: str = "full",
      max_steps_hint: int = 15,
      temperature: float = 0.0,
      response_max_tokens: int = 256,
      request_timeout_s: int = 25,
      request_retries: int = 1,
      safety_request_retries: int = 2,
      empty_reply_retries: int = 5,
      retry_backoff_s: float = 1.5,
      transient_max_backoff_s: float = 12.0,
      deadline_retry_backoff_s: float = 60.0,
      deadline_max_backoff_s: float = 60.0,
      multimodal_input_mode: str = "path",
      verbose: bool = False,
      vertex_project: str = "",
      vertex_location: str = "us",
      anthropic_max_image_edge: str | int = "auto",
      gemini_max_image_edge: int | None = None,
      gemini_media_resolution: str | None = None,
      openai_max_image_edge: int | None = None,
      capture_thinking: bool = False,
      capture_interactions: bool = False,
      allow_compact_fallback: bool = True,
      tool_mode: str = "text",
      tool_schema_mode: str = "native",
      native_reasoning: bool = True,
      reasoning_effort: str | None = None,
      chat_template_kwargs: dict[str, Any] | None = None,
      replay_chat_reasoning: bool = False,
  ) -> None:
    if provider not in {
        "anthropic",
        "openai",
        "gemini",
        "vertex-anthropic",
        "vertex-xai",
        "bedrock-gpt",
    }:
      raise ValueError(
          f"Unsupported LLM provider: {provider!r}. Choose from: anthropic,"
          " openai, gemini, vertex-anthropic, vertex-xai, bedrock-gpt"
      )
    if history_mode not in {"auto", "full", "summary"}:
      raise ValueError(
          f"Unsupported history mode: {history_mode!r}. Choose from: auto,"
          " full, summary"
      )
    if tool_mode not in {"text", "native"}:
      raise ValueError(
          f"Unsupported tool mode: {tool_mode!r}. Choose from: text, native"
      )
    if tool_schema_mode not in {"native", "unified"}:
      raise ValueError(
          f"Unsupported tool schema mode: {tool_schema_mode!r}. Choose from:"
          " native, unified"
      )
    self._provider = provider
    self._model = model
    # Grok on Vertex AI is served through the OpenAI-compatible endpoint and
    # requires an 'xai/' publisher prefix on the model id. Add it automatically
    # so callers can pass either 'grok-4.20-reasoning' or 'xai/grok-4.20-reasoning'.
    if provider == "vertex-xai" and model and not model.startswith("xai/"):
      self._model = f"xai/{model}"
    if base_url is not None:
      self._base_url = base_url.rstrip("/")
    elif provider == "gemini":
      self._base_url = "https://generativelanguage.googleapis.com/v1beta"
    else:
      self._base_url = ""
    self._api_key_env = api_key_env
    self._history_mode = history_mode
    self._max_steps_hint = max_steps_hint
    if model == "o3":
      self._temperature = 1.0
    else:
      self._temperature = temperature
    self._response_max_tokens = max(32, response_max_tokens)
    self._request_timeout_s = request_timeout_s
    self._request_retries = max(0, request_retries)
    # Small, dedicated retry budget for provider-side INPUT safety pre-check
    # false positives (see _is_flaky_safety_llm_error). These fire flakily on
    # benign benchmark prompts, so a bounded verbatim retry usually clears them;
    # kept separate from (and never larger than) the general request-retry budget
    # so a genuinely deterministic safety rejection does not burn all 20 retries.
    self._safety_request_retries = max(0, min(safety_request_retries, self._request_retries))
    # Small, dedicated retry budget for a successful (HTTP 200) but EMPTY /
    # contentless reply: Anthropic ``stop_reason=refusal`` (no text/tool_use),
    # Gemini ``finish_reason=SAFETY`` (blocked), or an OpenAI-compatible response
    # with empty message content. These are re-sent verbatim (same request) with
    # backoff -- helping a genuinely transient empty (e.g. OpenAI 'completed'
    # with no text) -- but kept SMALL and never larger than the general
    # request-retry budget so a DETERMINISTIC refusal does not burn the full
    # (e.g. 20) budget every stalling step for an outcome that will not change.
    self._empty_reply_retries = max(0, min(empty_reply_retries, self._request_retries))
    self._retry_backoff_s = max(0.0, retry_backoff_s)
    # Per-retry ceiling for the ordinary transient class (429/503/network).
    # Defaults to 12.0, the value this was hardcoded to, so an unset flag leaves
    # the published-run behavior byte-identical. Raise it when the endpoint is
    # quota-limited on a window measured in hours rather than seconds: at the
    # 12s ceiling the backoff stops growing, so an exhausted quota is met with
    # sustained ~7 req/min/worker hammering instead of a widening wait.
    self._transient_max_backoff_s = max(0.0, transient_max_backoff_s)
    self._deadline_backoff_s = max(0.0, deadline_retry_backoff_s)
    self._deadline_max_backoff_s = max(0.0, deadline_max_backoff_s)
    if multimodal_input_mode not in {"path", "direct", "both"}:
      raise ValueError(
          "multimodal_input_mode must be one of: path, direct, both"
      )
    self._multimodal_input_mode = multimodal_input_mode
    self._multimodal_input_mode_effective = "path"
    # 'auto' = send originals (max fidelity) then learn the server's cap on the
    # first oversized-image rejection; an int forces a fixed long-edge cap.
    self._anthropic_max_image_edge = _normalize_anthropic_max_image_edge(
        anthropic_max_image_edge
    )
    # Fixed long-edge cap (pixels) for inline images sent to gemini; None (or a
    # non-positive value) disables resizing so originals are sent. Unlike
    # anthropic there is no 'auto' learning - gemini/Vertex never rejects
    # oversized images, so the cap is operator-chosen.
    self._gemini_max_image_edge = (
        gemini_max_image_edge
        if isinstance(gemini_max_image_edge, int)
        and not isinstance(gemini_max_image_edge, bool)
        and gemini_max_image_edge > 0
        else None
    )
    # Per-image visual token budget for gemini ("low"|"medium"|"high"
    # |"ultra_high"), or None to leave the request untouched (server default).
    #
    # This is the ONLY knob that changes how much visual detail gemini actually
    # receives. Unlike anthropic/openai -- which tokenize by pixel area, so
    # resizing the upload directly changes what the model sees -- gemini
    # resamples every image to a FIXED token budget server-side, so
    # --gemini-max-image-edge is effectively a no-op for model-visible detail
    # and only trims upload bytes. Measured on
    # eval_results/data_3_5_flash (n=5509 images, R^2=0.9999): an 8.29 MP image
    # and a 1.08 MP image both cost ~1100 tokens, i.e. the default budget is
    # already saturated by ~1 MP.
    #
    # Set per-Part rather than via GenerateContentConfig.media_resolution
    # because "ultra_high" is only accepted on individual media parts -- the
    # global enum (types.MediaResolution) has no ULTRA_HIGH member at all.
    self._gemini_media_resolution = _normalize_gemini_media_resolution(
        gemini_media_resolution
    )
    # Fixed long-edge cap (pixels) for inline images sent to OpenAI-compatible
    # providers (openai, vertex-xai/Grok). gpt-5.x bills image tokens ~linearly
    # with pixel area (no server-side downscale cap), and history_mode='full'
    # re-sends every image each step, so uncapped 4K assets dominate cost.
    # Downscaling once at construction shrinks every re-send. None (or a
    # non-positive value) disables resizing so originals are sent.
    self._openai_max_image_edge = (
        openai_max_image_edge
        if isinstance(openai_max_image_edge, int)
        and not isinstance(openai_max_image_edge, bool)
        and openai_max_image_edge > 0
        else None
    )
    self._verbose = verbose
    self._vertex_project = vertex_project
    self._vertex_location = vertex_location
    self._messages: list[dict[str, Any]] = []
    self._client: Any = None
    # Cached Google ADC credentials for provider 'vertex-xai' (Grok on Vertex).
    # Refreshed lazily when the access token expires (~1h lifetime).
    self._vertex_credentials: Any = None
    self._task: Task | None = None
    self._system_prompt = self.BASE_SYSTEM_PROMPT
    # Extra attempts allowed within a single step when the model's reply cannot
    # be turned into a valid command (text mode) or tool call (native mode);
    # ``N`` means ``N + 1`` requests before the step gives up and the stall
    # handling coerces DONE. Kept at 2 for backward compatibility, but exposed
    # as an attribute because ``tool_choice=auto`` makes the "returned text, no
    # tool call" branch a normal occurrence rather than a near-impossibility --
    # a run that wants to be generous about re-prompting can raise it without
    # editing this file.
    self._invalid_retry_limit = 2
    self._pending_action: str = ""
    self._history: list[dict[str, Any]] = []
    self._done_requested = False
    self._last_reply_debug: str | None = None
    self._reply_attempts_debug: list[str] = []
    self._stall_reason: str | None = None
    self._task_instruction: str = ""
    # Complete-interaction logging (opt-in, off by default so default eval runs
    # are byte-for-byte unchanged). ``capture_thinking`` adds a Gemini
    # ThinkingConfig so thought summaries are returned; ``capture_interactions``
    # assembles a structured per-attempt record (including the terminal turn
    # that coerces DONE) for a sidecar log. See ``debug_interactions``.
    self._capture_thinking = bool(capture_thinking)
    self._capture_interactions = bool(capture_interactions)
    # When False, a content-filter / invalid_prompt error is NOT silently
    # salvaged with a degraded compact-prompt retry; it propagates so the task
    # is recorded as a hard provider failure. Lets strict eval runs avoid
    # contaminating results with degraded (image-dropped) trajectories.
    self._allow_compact_fallback = bool(allow_compact_fallback)
    # Per-call metadata set by each provider path (Gemini, OpenAI/vertex-xai,
    # Anthropic) and consumed while assembling the interaction record in
    # ``act``. Reset to None before every attempt; stays None only when capture
    # is off or the call errored before a response was parsed.
    self._last_response_meta: dict[str, Any] | None = None
    # Visible assistant text from the most recent native tool-call request that
    # returned NO tool call ("" whenever a call was parsed). Set by each
    # provider's ``_*_tool_call`` and consumed by the no-call path in
    # ``_act_native``, which replays it as a real assistant turn so the model's
    # own words survive into the next request instead of being discarded.
    #
    # Deliberately NOT routed through ``_last_response_meta``: that is only
    # populated when capture_thinking/capture_interactions is on, and history
    # content must not depend on whether logging is enabled.
    self._last_no_call_text: str = ""
    # Structured per-attempt interaction records for the current ``act`` call;
    # reset each ``act`` (mirrors ``_reply_attempts_debug``) so parallel
    # ``copy.copy`` agent clones stay isolated after their first step.
    self._interaction_log: list[dict[str, Any]] = []
    # Native function/tool-calling mode (WS3-5). In 'native' mode the provider
    # returns a structured tool call which is reverse-mapped to the workspace CLI
    # string; ``self._messages`` then holds provider-native tool_use/tool_result
    # turns ("typed structured history"). 'text' mode is the legacy ReAct path.
    self._tool_mode = tool_mode
    # Schema *dialect* for native tool defs (WS: --llm-tool-schema ablation).
    # 'native' renders each provider's idiomatic argument schema (Gemini's
    # uppercase OpenAPI form); 'unified' feeds every provider the identical
    # canonical JSON-Schema (Gemini via ``parameters_json_schema``) so a
    # cross-provider comparison is not confounded by an *avoidable* schema-format
    # difference. Mandatory per-API envelopes are left untouched either way.
    # Only affects Gemini today (OpenAI/Anthropic already emit canonical schema).
    self._tool_schema_mode = tool_schema_mode
    # Native tool_choice policy, shared by all three providers so the setting is
    # not silently confounded with the vendor. "auto" (the default) lets the
    # model decide whether to emit a tool call; "forced" sends each vendor's
    # mandatory-call form (OpenAI ``required`` / Anthropic ``{"type":"any"}`` /
    # Gemini ``mode="ANY"``).
    #
    # The default is ``auto`` because the previous per-vendor defaults were
    # asymmetric -- OpenAI and Gemini were always forced while Anthropic relaxed
    # to ``auto`` whenever native reasoning was on -- which makes tool_choice
    # perfectly collinear with the provider and therefore unmeasurable. ``auto``
    # is also the only value legal in every reasoning configuration (a forced
    # choice is rejected outright under Anthropic's *manual* extended thinking),
    # so it is the safe common denominator.
    #
    # Forcing is not free of side effects: ``required``/``any``/``ANY`` prefill
    # the assistant message, so the model cannot emit natural-language text
    # before the tool call -- under ``auto`` each step may carry a short
    # scratchpad. Empirically ``auto`` does not make models skip tools: measured
    # over the existing Claude eval runs only 13/8553 (0.15%) of opus-4-8 turns
    # and 1/7517 sonnet-5 turns returned no tool call.
    #
    # NOTE for the forced arm of an ablation: Anthropic only accepts a forced
    # choice under *adaptive* thinking. Under the "budget" rung of
    # ``_anthropic_reasoning_mode`` the request is rejected, and the rejection
    # matches ``_is_anthropic_reasoning_param_unsupported``, so
    # ``_anthropic_create`` would silently degrade all the way to "plain" --
    # yielding forced tool_choice with thinking disabled and nothing in the log
    # to say so. ``_anthropic_tool_choice`` therefore keeps ``auto`` on the
    # non-adaptive rungs even when ``forced`` is requested.
    self._native_tool_choice = "auto"
    # Reasoning-threading switch for native tool-calling (native mode only).
    # When True, Claude and GPT thread their opaque per-turn reasoning back into
    # each forced tool-call request -- mirroring what Gemini already does with
    # ``thought_signature`` -- so the chain of thought stays continuous across
    # steps and all three providers are treated consistently. For Anthropic this
    # turns on extended thinking (adaptive, with an ``effort`` level); for OpenAI
    # Responses it adds ``store=false`` +
    # ``include=["reasoning.encrypted_content"]`` and replays the reasoning
    # items. False reproduces the prior (reasoning-dropping) native behavior --
    # no thinking, which is what regressed the native-tool scores against the
    # text baseline; text mode ignores it.
    #
    # This no longer affects ``tool_choice``: that is now the independent
    # ``_native_tool_choice`` knob, so the two variables can be ablated
    # separately instead of moving together.
    self._native_reasoning = bool(native_reasoning)
    # Optional explicit reasoning-effort override for native tool calls, applied
    # to whichever vendor knob carries it (OpenAI ``reasoning.effort``, Anthropic
    # ``output_config.effort``). ``None`` -- the default -- sends no effort key
    # at all for OpenAI and keeps Anthropic's ``_ANTHROPIC_ADAPTIVE_EFFORT``, so
    # every request is byte-identical to the pre-flag behavior.
    #
    # This exists because the vendors' OWN defaults differ: Anthropic documents
    # ``high`` as the API default for Sonnet 5 / Opus 4.8, while gpt-5.5 defaults
    # to ``medium`` (confirmed against the Bedrock proxy, which echoes
    # ``reasoning: {"effort": "medium"}`` when the key is omitted). Comparing the
    # families at their own defaults is therefore an unequal-effort comparison;
    # this knob makes that variable controllable instead of implicit.
    self._reasoning_effort = (reasoning_effort or "").strip() or None
    # Server-side chat-template variables, forwarded verbatim on the Chat
    # Completions path as ``chat_template_kwargs``.
    #
    # This is the only reasoning control that some open-weight models expose.
    # ``reasoning_effort`` is a no-op for them twice over: the harness only
    # sends it on the Responses API path (gpt-* models), and vLLM answers a
    # chat/completions request carrying it with HTTP 200 while ignoring it, so
    # a run "at low effort" is silently a run at the model's default. Qwen3.6
    # instead reads ``enable_thinking`` (thinking on/off) and
    # ``preserve_thinking`` (keep earlier turns' thinking in the rendered
    # prompt) out of its chat template. Kept as a free-form dict because these
    # names are per-model template variables, not a cross-vendor vocabulary.
    self._chat_template_kwargs = dict(chat_template_kwargs or {})
    # Re-inline each past tool turn's thinking on the Chat Completions path so
    # the model keeps a continuous chain of thought across steps (the Responses
    # and Anthropic paths already thread reasoning through their own native
    # item shapes; this is the equivalent for chat-format thinking models).
    #
    # Opt-in rather than automatic: the replay writes literal
    # ``<think>...</think>`` tags into the assistant content, which is only
    # inert for models whose chat template parses them back out (Qwen does).
    # It is additionally self-limiting -- a model that returns no reasoning has
    # nothing to inline -- but the tag convention is still model-specific.
    self._replay_chat_reasoning = bool(replay_chat_reasoning)
    # Fail fast on a typo'd level for ANY provider. The flag exposes one
    # canonical ladder (_REASONING_EFFORT_LADDER) and each provider maps every
    # rung onto its own wire vocabulary at the send site, so the only thing to
    # reject here is a value that is not on the ladder at all. (A rung a given
    # provider can't express natively -- e.g. Gemini above HIGH, OpenAI's
    # 'minimal' -- is folded onto the nearest real tier by the per-provider map,
    # not rejected, so the same flag value works everywhere.)
    if (
        self._reasoning_effort
        and self._reasoning_effort not in _REASONING_EFFORT_LADDER
    ):
      raise ValueError(
          f"--llm-reasoning-effort={self._reasoning_effort!r} is not a valid "
          f"level. Pick one of {list(_REASONING_EFFORT_LADDER)}, or drop the "
          f"flag to use the model's own default."
      )
    # Anthropic reasoning-control degradation ladder for native tool calls,
    # stepped down in ``_anthropic_create`` when a backend rejects the preferred
    # request. "adaptive" = ``thinking={"type":"adaptive"}`` + top-level
    # ``output_config`` effort; "budget" = fixed ``thinking={"type":"enabled",
    # "budget_tokens": _ANTHROPIC_FIXED_THINKING_BUDGET}``; "plain" = no
    # thinking. Learned once per process-lived agent so we waste at most one
    # call per downgrade. ``tool_choice`` is independent of this ladder
    # (``_native_tool_choice``), except that a *forced* choice is only sent on
    # the "adaptive" rung -- see ``_anthropic_tool_choice``.
    self._anthropic_reasoning_mode = "adaptive"
    # Separate one-shot fallback for the "no think" request shape (native tool
    # path, ``--llm-reasoning-effort`` in ``_EFFORT_OFF_VALUES``, i.e. ``none``):
    # Claude has no ``none`` effort rung, so we send
    # ``thinking={"type":"disabled"}`` + effort=low. If a backend rejects that
    # param combo we clear this flag and fall back to a plain (no-thinking)
    # request, which still satisfies "off". (``minimal`` is not "off" for Claude
    # -- it maps to a real ``low`` rung and uses the adaptive path below.)
    # Kept apart from the adaptive/budget ladder above because it is the opposite
    # request (thinking off, not sized) and must not be re-enabled by it.
    self._anthropic_disable_thinking_supported = True
    # The tool call emitted on the previous successful step; its result (the next
    # observation) is replayed as a paired tool_result turn at the start of the
    # following ``act`` (native mode only).
    self._pending_call: ToolCall | None = None
    # Per-task tool surface (native mode), built in ``reset``: the provider-
    # agnostic defs plus each vendor's pre-formatted ``tools`` payload.
    self._native_tool_defs: list[Any] = []
    self._native_tools_openai_chat: list[dict[str, Any]] = []
    self._native_tools_openai_responses: list[dict[str, Any]] = []
    self._native_tools_anthropic: list[dict[str, Any]] = []
    self._native_tools_gemini: list[dict[str, Any]] = []

  def _uses_litellm_proxy_defaults(self) -> bool:
    if self._provider != "openai":
      return False
    model = (self._model or "").strip().lower()
    return any(
        model.startswith(prefix)
        for prefix in self._LITELLM_PROXY_MODEL_PREFIXES
    )

  def _litellm_proxy_env_name(self) -> str:
    for name in self._LITELLM_PROXY_KEY_ENV_CANDIDATES:
      if os.environ.get(name):
        return name
    return self._LITELLM_PROXY_KEY_ENV_CANDIDATES[0]

  def _api_key_name(self) -> str:
    if self._api_key_env:
      return self._api_key_env
    if self._uses_litellm_proxy_defaults():
      return self._litellm_proxy_env_name()
    if self._provider == "gemini":
      return "GEMINI_API_KEY"
    if self._provider == "bedrock-gpt":
      # GPT on Bedrock uses a dedicated key env so it never collides with a
      # direct-OpenAI run sharing the same shell (OPENAI_API_KEY).
      return "BEDROCK_OPENAI_API_KEY"
    return (
        "OPENAI_API_KEY" if self._provider == "openai" else "ANTHROPIC_API_KEY"
    )

  def _api_key(self) -> str:
    api_key = os.environ.get(self._api_key_name(), "")
    if not api_key:
      raise RuntimeError(
          f"{self._api_key_name()} environment variable is not set."
      )
    return api_key

  def _vertex_access_token(self) -> str:
    """Return a fresh Google OAuth access token for Vertex (provider vertex-xai).

    Grok on Vertex authenticates via Application Default Credentials rather than
    a static API key. Access tokens are short-lived (~1h), so we cache the
    credentials object and refresh the token whenever it has expired -- this
    matters for long eval runs that outlast a single token lifetime.
    """
    try:
      import google.auth
      import google.auth.transport.requests
    except ImportError as e:
      raise RuntimeError(
          "google-auth not installed. Run: pip install 'google-auth>=2.29.0'"
      ) from e
    if self._vertex_credentials is None:
      self._vertex_credentials, _ = google.auth.default(
          scopes=["https://www.googleapis.com/auth/cloud-platform"]
      )
    creds = self._vertex_credentials
    if not creds.valid:
      creds.refresh(google.auth.transport.requests.Request())
    if not creds.token:
      raise RuntimeError(
          "Failed to obtain a Vertex access token from Application Default"
          " Credentials. Run: gcloud auth application-default login"
      )
    return creds.token

  def _bearer_token(self) -> str:
    """Authorization bearer value for OpenAI-compatible requests.

    'vertex-xai' uses a Google ADC access token; all other OpenAI-style
    providers use the configured static API key.
    """
    if self._provider == "vertex-xai":
      return self._vertex_access_token()
    return self._api_key()

  def _get_client(self) -> Any:
    if self._provider not in {"anthropic", "vertex-anthropic"}:
      raise RuntimeError("_get_client is only valid for Anthropic providers")
    if self._client is not None:
      return self._client
    if self._provider == "vertex-anthropic":
      try:
        from anthropic import AnthropicVertex
      except ImportError as e:
        raise RuntimeError(
            "anthropic[vertex] not installed. Run: pip install"
            " 'anthropic[vertex]'"
        ) from e
      # Claude on Vertex AI authenticates via Application Default Credentials
      # (no API key). The model id must be a Vertex publisher id such as
      # 'claude-sonnet-4@20250514' and --vertex-location must serve Claude
      # (e.g. 'us-east5'). Same messages.create() interface as the direct
      # Anthropic client, so _anthropic_response_text is reused verbatim.
      self._client = AnthropicVertex(
          project_id=self._vertex_project, region=self._vertex_location
      )
      return self._client
    try:
      import anthropic
    except ImportError as e:
      raise RuntimeError(
          "anthropic package not installed. Run: pip install anthropic"
      ) from e
    self._client = anthropic.Anthropic(api_key=self._api_key())
    return self._client

  def _get_gemini_client(self) -> Any:
    if self._client is not None:
      return self._client
    try:
      from google import genai
      from google.genai import types
    except ImportError as e:
      raise RuntimeError(
          "google-genai package not installed or not in dependency."
      ) from e
    # Two ways to reach the same models. The Gemini Developer API (an API key
    # in GEMINI_API_KEY) is the default because it needs no cloud project and
    # no ADC, which is what a reader reproducing the benchmark has. Vertex AI
    # is kept for the original run configuration and is selected by supplying
    # --vertex-project with no key set; the model ids are identical on both, so
    # nothing else in this class branches on the choice.
    api_key = os.environ.get(self._api_key_name(), "").strip()
    if api_key:
      client_kwargs: dict[str, Any] = dict(api_key=api_key)
    elif self._vertex_project:
      client_kwargs = dict(
          vertexai=True,
          project=self._vertex_project,
          location=self._vertex_location,
      )
    else:
      raise RuntimeError(
          f"{self._api_key_name()} is not set and no --vertex-project was"
          " given, so there is no way to authenticate to Gemini. Set"
          f" {self._api_key_name()} for the Gemini Developer API, or pass"
          " --vertex-project (with Application Default Credentials) to use"
          " Vertex AI."
      )
    # Pass an explicit per-request timeout so gemini calls honor --llm-timeout-s.
    # HttpOptions.timeout is in milliseconds. Without it the SDK can hang far
    # past the intended cap and produce a latency fat-tail, unlike the
    # anthropic/openai paths which already thread the timeout through.
    if self._request_timeout_s and self._request_timeout_s > 0:
      client_kwargs["http_options"] = types.HttpOptions(
          timeout=int(self._request_timeout_s * 1000)
      )
    self._client = genai.Client(**client_kwargs)
    return self._client

  def _current_anthropic_image_edge(self) -> int | None:
    """Effective image long-edge cap (pixels) for the current model.

    ``auto`` starts uncapped (send originals at max fidelity) and returns the
    cap learned from a prior oversized-image rejection. A fixed integer cap
    always applies, further reduced by any learned cap for the same model.
    """
    learned = _ANTHROPIC_MODEL_IMAGE_EDGE.get(self._model)
    if isinstance(self._anthropic_max_image_edge, int):
      fixed = self._anthropic_max_image_edge
      return fixed if learned is None else min(fixed, learned)
    return learned

  def _sleep_before_retry(self, exc: Exception, attempt: int) -> None:
    """Sleep between retries using a class-aware backoff.

    Server-side deadline errors (504) use a longer, jittered backoff so an
    overloaded backend can drain before we re-send the same (heavy) request;
    ordinary transient errors (429/503/network) keep the short backoff used for
    throttling/unavailability.
    """
    if _is_deadline_error(exc):
      base = self._deadline_backoff_s
      cap = self._deadline_max_backoff_s
      label = " (deadline)"
    else:
      base = self._retry_backoff_s
      cap = self._transient_max_backoff_s
      label = ""
    delay = _retry_backoff_delay(base, attempt, cap)
    if delay > 0:
      self._reply_attempts_debug.append(
          f"api-backoff: sleeping {delay:.1f}s before retry{label}"
      )
      time.sleep(delay)

  def _anthropic_native_reasoning_enabled(self) -> bool:
    """Whether extended thinking is threaded through native Claude tool calls.

    Only in native mode with ``native_reasoning`` on and an Anthropic provider.
    When on, each tool step carries its thinking blocks back so the chain of
    thought stays continuous. The exact request shape follows the per-agent
    degradation ladder in ``self._anthropic_reasoning_mode`` (adaptive -> fixed
    budget -> plain); the "plain" rung disables thinking entirely, so this
    returns False for it.

    This does NOT by itself determine ``tool_choice`` -- see
    ``_anthropic_tool_choice``. Contrary to an earlier reading of the docs, only
    *manual* extended thinking (``thinking={"type": "enabled", ...}``) forbids a
    forced tool choice; the *adaptive* form we prefer explicitly supports it.
    """
    return (
        self._tool_mode == "native"
        and self._native_reasoning
        and self._anthropic_reasoning_mode != "plain"
        and self._provider in {"anthropic", "vertex-anthropic"}
    )

  def _anthropic_tool_choice(self) -> dict[str, Any]:
    """The ``tool_choice`` block for a native Claude tool call.

    ``auto`` unless ``_native_tool_choice`` asks for the forced form, in which
    case ``{"type": "any"}`` -- but only on reasoning rungs that accept it.

    The "budget" rung sends manual extended thinking, which rejects a forced
    choice with an error that ``_is_anthropic_reasoning_param_unsupported``
    also matches; ``_anthropic_create`` would read that as "this backend does
    not support the reasoning params", step down again, and land on "plain".
    The run would then be forced-with-no-thinking while the log still claimed
    reasoning was on. Staying on ``auto`` for that rung keeps the degradation
    ladder measuring what it says it measures.
    """
    if self._native_tool_choice != "forced":
      return {"type": "auto"}
    # "No think" (``thinking={"type":"disabled"}``, chosen in _anthropic_create
    # when the shared effort knob is in _EFFORT_OFF_VALUES) accepts a forced
    # choice: the forbid-forced rule applies only to *manual* enabled thinking.
    # Honor the forced request directly, independent of the adaptive/budget
    # ladder state, so "off" runs still get the forced surface they asked for.
    if (
        self._reasoning_effort in _EFFORT_OFF_VALUES
        and self._anthropic_disable_thinking_supported
    ):
      return {"type": "any"}
    if self._anthropic_native_reasoning_enabled():
      # Only the adaptive rung is documented to accept a forced choice.
      return (
          {"type": "any"}
          if self._anthropic_reasoning_mode == "adaptive"
          else {"type": "auto"}
      )
    return {"type": "any"}

  def _anthropic_create(
      self,
      base_messages: list[dict[str, Any]],
      *,
      tools: list[dict[str, Any]] | None = None,
      tool_choice: dict[str, Any] | None = None,
      is_empty: Callable[[Any], bool] | None = None,
  ) -> Any:
    """Issue a Messages request with image byte-budget healing; return the raw response.

    Shared by the text path (:meth:`_anthropic_response_text`) and the native
    tool path (:meth:`_anthropic_tool_call`). ``tools``/``tool_choice`` are
    attached only when given, so the text path's request is byte-for-byte
    unchanged.

    ``is_empty``, when given, decides whether a SUCCESSFUL (HTTP 200) response
    carried no usable content (e.g. ``stop_reason=refusal`` with no text /
    tool_use block). Such a response is re-sent verbatim under the small,
    dedicated ``_empty_reply_retries`` budget with backoff before it is returned,
    so a transient empty may clear while a deterministic refusal stops after the
    cap instead of burning the full request-retry budget.
    """
    client = self._get_client()

    def _create() -> Any:
      # Bounded self-healing loop: at most one temperature drop plus one retry
      # per image-cap step (parsed limit, then fallbacks). Both failure modes
      # mutate module-level caches and re-issue the request on the next pass.
      fallbacks = iter(_ANTHROPIC_IMAGE_EDGE_FALLBACKS)
      max_heal_attempts = 2 + len(_ANTHROPIC_IMAGE_EDGE_FALLBACKS) + 1
      last_exc: Exception | None = None
      for _ in range(max_heal_attempts):
        edge = self._current_anthropic_image_edge()
        # Proactively shrink the WHOLE request under the total-payload byte
        # budget before sending. Vertex rejects requests over 30MB with a 400
        # "request size ... exceeds 30.000MB limit" that the per-image pixel-cap
        # healing below never triggers on (no single image is individually
        # oversized) -- conflict-VLM rounds accumulate every round-visible image
        # (r10 ~= 25 images, several 4K) and the full-history payload trips it.
        # This descends the edge ladder only as far as needed to fit and returns
        # the edge it actually used, so the pixel-heal fallback below stays
        # consistent (it only accepts a cap strictly smaller than ``edge``).
        request_messages, edge = _fit_anthropic_messages_to_byte_budget(
            base_messages,
            edge,
            _ANTHROPIC_REQUEST_BYTE_BUDGET,
            _ANTHROPIC_BYTE_BUDGET_EDGE_LADDER,
        )
        # Extended thinking is threaded only on the native tool path (tools set).
        # ``mode`` selects the request shape from the per-agent degradation
        # ladder: "adaptive" (preferred) sends ``thinking={"type":"adaptive"}``
        # plus a top-level ``output_config`` effort; "budget" sends a fixed
        # ``budget_tokens``; "disabled" turns thinking OFF ("no think", below);
        # "plain" (and every non-native path) sends neither.
        #
        # "disabled" wins over the adaptive/budget ladder and does NOT depend on
        # ``native_reasoning``: the shared --llm-reasoning-effort asked Claude to
        # stop thinking, and a plain (no-param) request would NOT achieve that --
        # Sonnet 5 / Opus 4.8 default to high effort when no thinking param is
        # sent -- so "off" has to be requested explicitly. Only the native tool
        # path opts in (the text path keeps its byte-identical plain request).
        disable_thinking = (
            tools is not None
            and self._reasoning_effort in _EFFORT_OFF_VALUES
            and self._anthropic_disable_thinking_supported
        )
        if disable_thinking:
          mode = "disabled"
        else:
          thinking_on = (
              tools is not None and self._anthropic_native_reasoning_enabled()
          )
          mode = self._anthropic_reasoning_mode if thinking_on else "plain"
        # ``max_tokens`` must exceed the thinking budget; the default (256) is far
        # below a useful budget, so raise the ceiling to leave the model room for
        # both the thinking and the tool_use block it emits. Adaptive thinking
        # sizes its own budget, so reserve the same generous fixed headroom.
        max_tokens = self._response_max_tokens
        if mode in ("adaptive", "budget"):
          max_tokens = max(
              max_tokens,
              _ANTHROPIC_FIXED_THINKING_BUDGET + _ANTHROPIC_THINKING_OUTPUT_HEADROOM,
          )
        kwargs: dict[str, Any] = dict(
            model=self._model,
            max_tokens=max_tokens,
            system=self._system_prompt,
            messages=request_messages,
            # Pass an explicit timeout so the SDK does not raise
            # "Streaming is required for operations that may take longer than
            # 10 minutes" when max_tokens is large; also honors --llm-timeout-s.
            timeout=self._request_timeout_s,
        )
        if tools is not None:
          kwargs["tools"] = tools
          if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if mode == "adaptive":
          kwargs["thinking"] = {"type": "adaptive"}
          # ``effort`` lives on the top-level ``output_config`` object, NOT inside
          # ``thinking``. It is supported on Sonnet 5 / Opus 4.8 and also nudges
          # the model to reason before each tool step.
          kwargs["output_config"] = {
              "effort": (
                  _ANTHROPIC_EFFORT_LEVELS[self._reasoning_effort]
                  if self._reasoning_effort
                  else _ANTHROPIC_ADAPTIVE_EFFORT
              )
          }
        elif mode == "budget":
          kwargs["thinking"] = {
              "type": "enabled",
              "budget_tokens": _ANTHROPIC_FIXED_THINKING_BUDGET,
          }
        elif mode == "disabled":
          # "No think": explicitly turn extended thinking off and pin effort to
          # the documented floor. Anthropic's ``output_config.effort`` has no
          # ``none`` rung, and ``disabled`` thinking is rejected alongside the
          # xhigh/max rungs, so ``low`` is the correct partner here (see
          # _ANTHROPIC_MIN_EFFORT). This -- not the plain path -- is what makes
          # the shared "none"/"off" request actually stop Claude reasoning.
          # (``minimal`` never reaches here: it is a real ``low`` rung and takes
          # the adaptive path above.)
          kwargs["thinking"] = {"type": "disabled"}
          kwargs["output_config"] = {"effort": _ANTHROPIC_MIN_EFFORT}
        # Extended thinking requires the default temperature: send an explicit
        # one only when thinking is off ("plain" or "disabled") and the model
        # accepts a value.
        if (
            mode in ("plain", "disabled")
            and self._model not in _ANTHROPIC_MODELS_OMIT_TEMPERATURE
        ):
          kwargs.update(anthropic_temperature_kwargs(self._temperature))
        try:
          return client.messages.create(**kwargs)
        except Exception as exc:  # pragma: no cover - provider dependent
          last_exc = exc
          # Backend rejects the adaptive/effort (or fixed-budget) reasoning
          # parameters: step down the ladder (adaptive -> budget -> plain) for
          # this agent and retry, so we degrade gracefully instead of failing.
          if mode in ("adaptive", "budget") and _is_anthropic_reasoning_param_unsupported(
              exc
          ):
            self._anthropic_reasoning_mode = (
                "budget" if mode == "adaptive" else "plain"
            )
            continue
          # Backend rejects the "no think" params (``thinking={"type":
          # "disabled"}`` and/or the paired effort floor): stop sending them and
          # also pin the ladder to "plain" so the retry does not re-enable
          # thinking via ``native_reasoning``. Plain leaves the model at its own
          # default effort rather than truly off -- the closest we can get on
          # this backend without failing -- and _reasoning_effort_effective
          # reports "" once this flag is cleared.
          if mode == "disabled" and _is_anthropic_reasoning_param_unsupported(exc):
            self._anthropic_disable_thinking_supported = False
            self._anthropic_reasoning_mode = "plain"
            continue
          # Some newer models reject `temperature`; remember that for this
          # model and retry without it so we do not waste a call every step.
          if anthropic_kwargs_carry_temperature(
              kwargs
          ) and _is_anthropic_temperature_unsupported(
              exc
          ):
            _ANTHROPIC_MODELS_OMIT_TEMPERATURE.add(self._model)
            continue
          # Oversized images: learn the server's cap (or fall back), remember
          # it for this model, and retry with pre-shrunk images.
          if _is_anthropic_image_dimension_error(exc):
            limit = _parse_anthropic_image_pixel_limit(exc)
            if limit is None:
              limit = next(fallbacks, None)
            # Only accept a cap that actually shrinks the current edge.
            while limit is not None and edge is not None and limit >= edge:
              limit = next(fallbacks, None)
            if limit is None:
              raise
            _ANTHROPIC_MODEL_IMAGE_EDGE[self._model] = limit
            continue
          raise
      if last_exc is not None:
        raise last_exc
      raise RuntimeError("Anthropic request did not produce a response")

    last_exc: Exception | None = None
    empty_attempts = 0
    for attempt in range(self._request_retries + 1):
      try:
        response = _create()
      except Exception as exc:  # pragma: no cover - network/provider dependent
        last_exc = exc
        if not _is_retryable_llm_error(exc):
          raise
        if attempt < self._request_retries:
          self._sleep_before_retry(exc, attempt)
        continue
      # Successful call. Re-send an empty (contentless) reply under the dedicated
      # empty-reply budget; see ``is_empty`` in the docstring. ``empty_attempts``
      # is bounded by ``_empty_reply_retries`` (already capped at
      # ``_request_retries``), independent of the exception-retry budget above.
      if is_empty is not None and is_empty(response):
        if empty_attempts < self._empty_reply_retries:
          empty_attempts += 1
          self._reply_attempts_debug.append(
              "api-empty: provider returned no usable content; retry"
              f" {empty_attempts}/{self._empty_reply_retries}"
          )
          self._sleep_before_retry(_EmptyProviderReply(), empty_attempts - 1)
          continue
        self._reply_attempts_debug.append(
            "api-empty: still empty after"
            f" {empty_attempts} retr{'y' if empty_attempts == 1 else 'ies'};"
            " giving up (generation stall)"
        )
      return response
    assert last_exc is not None
    raise last_exc

  def _anthropic_response_text(self) -> str:
    response = self._anthropic_create(
        self._request_messages(self._messages),
        is_empty=lambda r: not _anthropic_text_from_response(r),
    )
    text = _anthropic_text_from_response(response)
    if self._capture_thinking or self._capture_interactions:
      meta = _anthropic_response_meta(response)
      meta["answer_text"] = text
      self._last_response_meta = meta
    self._reply_attempts_debug.append(f"api: content={text!r}")
    return text

  def _anthropic_tool_call(self) -> ToolCall | None:
    # Native mode replays the full structured conversation (tool_use/tool_result
    # blocks); summary/auto history trimming is text-mode only.
    base_messages = to_anthropic_messages(self._messages)
    # ``auto`` by default (see ``_native_tool_choice``). The system prompt still
    # instructs exactly one tool per turn; the no-call path guards the rare
    # text-only turn.
    response = self._anthropic_create(
        base_messages,
        tools=self._native_tools_anthropic,
        tool_choice=self._anthropic_tool_choice(),
        # Contentless only when there is neither a tool call NOR visible text; a
        # text-only reply (model answered in prose) is NOT empty -- it is handled
        # by the native reprompt/nudge path, not re-sent verbatim.
        is_empty=lambda r: parse_anthropic_tool_call(r) is None
        and not _anthropic_text_from_response(r),
    )
    call = parse_anthropic_tool_call(response)
    self._last_no_call_text = (
        "" if call else _anthropic_text_from_response(response)
    )
    meta = None
    if self._capture_thinking or self._capture_interactions:
      meta = _anthropic_response_meta(response)
      meta["answer_text"] = call.name if call else ""
      self._last_response_meta = meta
    _record_dropped_parallel_calls(meta, call, self._reply_attempts_debug)
    self._reply_attempts_debug.append(
        f"api: tool_call={(call.name if call else None)!r}"
        f" args={(call.arguments if call else None)!r}"
    )
    return call

  def _openai_base_url(self) -> str:
    if self._base_url:
      return self._base_url
    if self._provider == "bedrock-gpt":
      # GPT on Bedrock's OpenAI-compatible ('mantle') endpoint. Region is fixed
      # to us-east-2; override with --llm-base-url for other regions.
      return "https://bedrock-mantle.us-east-2.api.aws/openai/v1"
    if self._provider == "vertex-xai":
      # Grok is served via Vertex AI's OpenAI-compatible ('openapi') endpoint.
      # Default to the global endpoint; a real region (e.g. 'us-central1') is
      # honored when given. The bare 'us' default is not a valid openapi region,
      # so it is treated as global.
      loc = (self._vertex_location or "").strip()
      if loc in ("", "global", "us"):
        return (
            "https://aiplatform.googleapis.com/v1/projects/"
            f"{self._vertex_project}/locations/global/endpoints/openapi"
        )
      return (
          f"https://{loc}-aiplatform.googleapis.com/v1/projects/"
          f"{self._vertex_project}/locations/{loc}/endpoints/openapi"
      )
    if self._uses_litellm_proxy_defaults():
      for name in self._LITELLM_PROXY_BASE_URL_ENV_CANDIDATES:
        value = (os.environ.get(name) or "").strip()
        if value:
          return value.rstrip("/")
      return self._LITELLM_PROXY_DEFAULT_BASE_URL
    return "https://api.openai.com/v1"

  def _trim_request_messages(
      self, messages: list[dict[str, Any]]
  ) -> list[dict[str, Any]]:
    return build_openai_request_messages(messages, self._history)

  def _request_messages(
      self, messages: list[dict[str, Any]]
  ) -> list[dict[str, Any]]:
    if self._history_mode == "full":
      return list(messages)
    if self._history_mode == "summary":
      return self._trim_request_messages(messages)
    return self._trim_request_messages(messages)

  def _compact_system_prompt(self) -> str:
    return (
        "Return exactly one shell command from the catalog or DONE. "
        "Do not explain. Do not use markdown."
    )

  def _compact_messages(self) -> list[dict[str, Any]]:
    first_content = self._messages[0]["content"] if self._messages else ""
    if isinstance(first_content, list):
      text_parts = [
          str(part.get("text", ""))
          for part in first_content
          if isinstance(part, dict) and part.get("type") == "text"
      ]
      first_content = "\n".join(text_parts)
    task_instruction = (
        self._task_instruction.strip()
        if self._task_instruction
        else str(first_content)
    )
    lines = [task_instruction.strip()]
    summary = build_memory_summary(self._history, compact=True)
    if summary:
      lines.append(summary)
    elif self._history:
      lines.append("Known context:")
      for item in self._history[-2:]:
        lines.append(
            f"- {item['action']} => exit={item.get('exit_code', 0)};"
            f" stdout={clip_text(item.get('stdout', ''), limit=100) or '(empty)'};"
            f" stderr={clip_text(item.get('stderr', ''), limit=80) or '(none)'}"
        )
    lines.append(
        "Reply with exactly one next command from the catalog or DONE."
    )
    return [{"role": "user", "content": "\n".join(lines)}]

  def _openai_token_limit_field(self) -> str:
    model = (self._model or "").lower()
    if model.startswith("gpt-5") or model.startswith("o3"):
      return "max_completion_tokens"
    return "max_tokens"

  def _openai_prefers_responses_api(self) -> bool:
    # GPT on Bedrock only serves the Responses API ('/responses'); its model ids
    # carry an 'openai.' publisher prefix (e.g. 'openai.gpt-5.5') that would not
    # match the 'gpt-' heuristic, so force it on by provider.
    if self._provider == "bedrock-gpt":
      return True
    model = (self._model or "").lower()
    return model.startswith("gpt-")

  def _openai_reasoning_enabled(self) -> bool:
    """Whether native GPT tool calls thread reasoning across tool steps.

    Only in native mode with ``native_reasoning`` on and an OpenAI-family
    provider served over the Responses API. When on, each request opts into
    stateless encrypted reasoning (``store=false`` +
    ``include=["reasoning.encrypted_content"]``) and the reasoning items are
    replayed before their ``function_call`` on the next step, so the reasoning is
    not dropped between tool turns -- dropping it drove the model into terminal
    no-op loops, which was most acute under a forced ``tool_choice``. Effort
    comes from ``_reasoning_effort`` (or the API default when unset); there is
    no numeric budget to thread, so this is a plain on/off.
    """
    return (
        self._tool_mode == "native"
        and self._native_reasoning
        and self._provider in {"openai", "vertex-xai", "bedrock-gpt"}
        and self._openai_prefers_responses_api()
    )

  def _openai_omits_temperature(self) -> bool:
    """Reasoning models reject an explicit (non-default) ``temperature``.

    gpt-5* and the o-series only accept their default temperature; sending one
    (even 0.0) makes the Responses/Chat endpoints return HTTP 400, so it must
    be omitted from the payload entirely rather than pinned to a value.
    """
    # GPT on Bedrock serves gpt-5-class reasoning models under an 'openai.'
    # prefixed id that would not match the checks below; omit by provider.
    if self._provider == "bedrock-gpt":
      return True
    model = (self._model or "").lower()
    return (
        model.startswith("gpt-5")
        or model.startswith("o1")
        or model.startswith("o3")
        or model.startswith("o4")
    )

  def _kimi_chat_effort(self) -> str:
    """K3's wire-level thinking effort for this run, "" when not applicable.

    The blanket claim next to ``_chat_template_kwargs`` -- that a chat/completions
    server answers HTTP 200 while ignoring ``reasoning_effort``, making the flag a
    no-op for open-weight models -- does not hold for Kimi K3, which validates the
    field server-side (an unsupported level is a hard HTTP 500) and measurably
    scales the thinking it generates with it: low/high/max produced 1448/2091/3890
    completion tokens on one fixed prompt while the visible answer stayed ~1.57k
    characters, i.e. the extra budget went into the ``reasoning`` channel.

    Gated to K3 rather than applied to the whole chat path, because for the other
    open-weight models served here the field really is inert (Qwen's knob lives in
    ``chat_template_kwargs``) and a server that validates it would fail the run
    outright rather than degrade quietly.

    Shared by the send site and ``_reasoning_effort_effective`` so what goes on
    the wire and what the run records cannot drift apart.
    """
    if not self._reasoning_effort:
      return ""
    if self._openai_prefers_responses_api():
      return ""
    if not _is_kimi_k3_model(self._model):
      return ""
    return _KIMI_EFFORT_LEVELS[self._reasoning_effort]

  def _to_responses_input(
      self, messages: list[dict[str, Any]]
  ) -> list[dict[str, Any]]:
    """Adapt chat-format messages to the Responses API ``input`` schema.

    Chat Completions and the Responses API disagree on multimodal content
    parts: chat uses ``{"type": "text"}`` and ``{"type": "image_url",
    "image_url": {"url": ...}}`` whereas Responses expects ``input_text`` and
    ``input_image`` (with ``image_url`` as a bare string) for user content, and
    ``output_text`` for assistant content. Without this conversion, image
    parts are silently rejected (HTTP 400) and gpt-5 never sees the picture.

    Messages whose ``content`` is a plain string are accepted verbatim by the
    Responses API and are passed through untouched.
    """
    converted: list[dict[str, Any]] = []
    for message in messages:
      content = message.get("content")
      if not isinstance(content, list):
        converted.append(message)
        continue
      text_type = (
          "output_text" if message.get("role") == "assistant" else "input_text"
      )
      parts: list[dict[str, Any]] = []
      for part in content:
        if not isinstance(part, dict):
          parts.append({"type": text_type, "text": str(part)})
          continue
        part_type = part.get("type")
        if part_type == "text":
          parts.append({"type": text_type, "text": part.get("text", "")})
        elif part_type == "image_url":
          image_url = part.get("image_url")
          url = (
              image_url.get("url", "")
              if isinstance(image_url, dict)
              else (image_url or "")
          )
          parts.append({"type": "input_image", "image_url": url})
        else:
          # Already Responses-shaped (input_text/input_image/...) or unknown;
          # pass the part through unchanged.
          parts.append(part)
      converted.append({**message, "content": parts})
    return converted

  def _openai_request_once(
      self,
      messages: list[dict[str, Any]],
      max_tokens: int | None = None,
      *,
      system_prompt: str | None = None,
      use_responses_api: bool = False,
      tools: list[dict[str, Any]] | None = None,
      tool_choice: Any | None = None,
  ) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {self._bearer_token()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "clasharness-train-eval/1.0",
    }
    if use_responses_api:
      payload = {
          "model": self._model,
          "instructions": system_prompt or self._system_prompt,
          "input": self._to_responses_input(messages),
          "max_output_tokens": max_tokens or self._response_max_tokens,
      }
      if self._temperature is not None and not self._openai_omits_temperature():
        payload["temperature"] = self._temperature
      if self._openai_reasoning_enabled():
        # Return the reasoning items with their opaque encrypted payload but keep
        # the request stateless (we replay full history every step rather than
        # relying on server-side ``previous_response_id``), matching how Gemini
        # and Claude thread reasoning here. The reasoning items are replayed
        # before their function_call by ``to_openai_responses_items``.
        payload["store"] = False
        payload["include"] = ["reasoning.encrypted_content"]
        # Only send ``reasoning`` when an effort was explicitly requested: with
        # no override the key is omitted entirely and the endpoint applies the
        # model's own default (``medium`` for gpt-5.5), keeping the payload
        # byte-identical to the pre-flag behavior.
        if self._reasoning_effort:
          payload["reasoning"] = {
              "effort": _OPENAI_EFFORT_LEVELS[self._reasoning_effort]
          }
      if tools is not None:
        payload["tools"] = tools
        if tool_choice is not None:
          payload["tool_choice"] = tool_choice
      request = urllib.request.Request(
          f"{self._openai_base_url()}/responses",
          data=json.dumps(payload).encode("utf-8"),
          headers=headers,
          method="POST",
      )
    else:
      token_limit_field = self._openai_token_limit_field()
      payload = {
          "model": self._model,
          "messages": [
              {
                  "role": "system",
                  "content": system_prompt or self._system_prompt,
              },
              *messages,
          ],
      }
      if self._temperature is not None and not self._openai_omits_temperature():
        payload["temperature"] = self._temperature
      payload[token_limit_field] = max_tokens or self._response_max_tokens
      # Template variables (e.g. Qwen's enable_thinking / preserve_thinking).
      # Chat Completions only: the Responses API has no chat template.
      if self._chat_template_kwargs:
        payload["chat_template_kwargs"] = dict(self._chat_template_kwargs)
      # Kimi K3's reasoning control, which -- unlike Qwen's -- is a top-level
      # field rather than a template variable. Omitted entirely for every other
      # model on this path, so their payloads stay byte-identical.
      kimi_effort = self._kimi_chat_effort()
      if kimi_effort:
        payload["reasoning_effort"] = kimi_effort
      if tools is not None:
        payload["tools"] = tools
        if tool_choice is not None:
          payload["tool_choice"] = tool_choice
      request = urllib.request.Request(
          f"{self._openai_base_url()}/chat/completions",
          data=json.dumps(payload).encode("utf-8"),
          headers=headers,
          method="POST",
      )
    with urllib.request.urlopen(
        request, timeout=self._request_timeout_s
    ) as response:
      return json.loads(response.read().decode("utf-8"))

  def _openai_request(
      self,
      messages: list[dict[str, Any]],
      max_tokens: int | None = None,
      *,
      system_prompt: str | None = None,
      trim_messages: bool = True,
      allow_compact_fallback: bool = False,
      tools: list[dict[str, Any]] | None = None,
      tool_choice: Any | None = None,
  ) -> dict[str, Any]:
    request_messages = (
        self._request_messages(messages) if trim_messages else list(messages)
    )
    last_exc: Exception | None = None
    use_responses_api = self._openai_prefers_responses_api()
    # Separate counter for flaky provider input-safety false positives: they get
    # only ``_safety_request_retries`` bounded verbatim retries, independent of
    # the (much larger) general ``_request_retries`` budget.
    safety_attempts = 0
    for attempt in range(self._request_retries + 1):
      try:
        return self._openai_request_once(
            request_messages,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            use_responses_api=use_responses_api,
            tools=tools,
            tool_choice=tool_choice,
        )
      except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        err = RuntimeError(
            f"OpenAI-compatible request failed: HTTP {exc.code}: {detail}"
        )
        if (
            not use_responses_api
            and exc.code == 400
            and "unsupported" in detail.lower()
        ):
          self._reply_attempts_debug.append(
              "api-fallback: chat/completions unsupported; retrying with"
              " responses API"
          )
          return self._openai_request_once(
              request_messages,
              max_tokens=max_tokens,
              system_prompt=system_prompt,
              use_responses_api=True,
              tools=tools,
              tool_choice=tool_choice,
          )
        if allow_compact_fallback and _is_content_filter_llm_error(err):
          self._reply_attempts_debug.append(
              "api-fallback: content filter triggered; retrying with compact"
              " prompt"
          )
          return self._openai_request(
              self._compact_messages(),
              max_tokens=min(max_tokens or self._response_max_tokens, 96),
              system_prompt=self._compact_system_prompt(),
              trim_messages=False,
              allow_compact_fallback=False,
              tools=tools,
              tool_choice=tool_choice,
          )
        # Some providers wrap transient server-side failures as an HTTP 400
        # (e.g. Bedrock returns
        # {"code":"validation_error","message":"Internal server error"}). The
        # body carries the real signal, so classify on ``err`` -- which embeds
        # the response body -- rather than ``exc``, whose str() omits it. A
        # non-429 4xx stays fatal unless the body looks transient; genuine
        # validation / content-policy 400s have no transient marker and still
        # raise immediately.
        body_retryable = _is_retryable_llm_error(err)
        is_flaky_safety = _is_flaky_safety_llm_error(err)
        if exc.code < 500 and exc.code != 429 and not body_retryable:
          # A non-429 4xx is normally fatal. Exception: a FLAKY provider
          # input-safety pre-check false positive (benign benchmark prompt
          # sporadically flagged) gets a SMALL bounded verbatim retry -- capped
          # at ``_safety_request_retries`` so a genuinely deterministic
          # rejection still fails fast instead of burning the full budget.
          if not is_flaky_safety:
            raise err from exc
          if safety_attempts >= self._safety_request_retries:
            self._reply_attempts_debug.append(
                "api-error: flaky safety pre-check still flagged after"
                f" {safety_attempts} retr{'y' if safety_attempts == 1 else 'ies'};"
                " giving up (provider_failure)"
            )
            raise err from exc
          safety_attempts += 1
          self._reply_attempts_debug.append(
              "api-fallback: flaky safety pre-check flagged; bounded verbatim"
              f" retry {safety_attempts}/{self._safety_request_retries}"
          )
        last_exc = err
        if attempt < self._request_retries:
          self._sleep_before_retry(exc, attempt)
      except (urllib.error.URLError, TimeoutError, OSError) as exc:
        err = RuntimeError(f"OpenAI-compatible request failed: {exc}")
        if not _is_retryable_llm_error(err):
          raise err from exc
        last_exc = err
        if attempt < self._request_retries:
          self._sleep_before_retry(exc, attempt)
    assert last_exc is not None
    raise last_exc

  def _extract_openai_message_text(
      self,
      payload: dict[str, Any],
      allow_followup: bool = True,
  ) -> str:
    if payload.get("object") == "response" or "output" in payload:
      output = payload.get("output") or []
      content_parts: list[dict[str, Any]] = []
      for item in output:
        if isinstance(item, dict) and item.get("type") == "message":
          content_parts.extend(item.get("content") or [])
      self._reply_attempts_debug.append(
          "api:"
          f" responses_parts={json.dumps(content_parts, ensure_ascii=False)}"
      )
      text_parts = [
          str(part.get("text", ""))
          for part in content_parts
          if isinstance(part, dict)
          and part.get("type") in {"output_text", "text"}
      ]
      text, removed_duplicates = _collapse_response_text_parts(text_parts)
      if removed_duplicates:
        self._reply_attempts_debug.append(
            f"api: deduped {removed_duplicates} adjacent duplicate response"
            " text part(s)"
        )
      if text:
        return text
      raise _EmptyProviderReply(
          "OpenAI-compatible responses API contained no text parts: "
          f"{json.dumps(payload, ensure_ascii=False)}"
      )
    try:
      choice_message = payload["choices"][0]["message"]
      message = choice_message["content"]
    except Exception as exc:  # pragma: no cover - defensive
      raise RuntimeError(
          f"Unexpected OpenAI-compatible response payload: {payload}"
      ) from exc
    reasoning_text = chat_reasoning_text(choice_message) or ""
    if message is None:
      self._reply_attempts_debug.append(
          f"api: content=None reasoning={reasoning_text!r}"
      )
      fallback = _extract_command_from_reasoning_text(
          reasoning_text, self._task
      )
      if fallback:
        return fallback
      if allow_followup and reasoning_text:
        followup_payload = self._openai_request(
            [
                *self._messages,
                {
                    "role": "user",
                    "content": (
                        "Your previous reply included reasoning but no final"
                        " command. Output exactly one next command from the"
                        " catalog now. Do not include reasoning or commentary."
                    ),
                },
            ],
            max_tokens=64,
            allow_compact_fallback=self._allow_compact_fallback,
        )
        return self._extract_openai_message_text(
            followup_payload,
            allow_followup=False,
        )
      raise _EmptyProviderReply(
          "OpenAI-compatible response contained empty message content: "
          f"{json.dumps(choice_message, ensure_ascii=False)}"
      )
    if isinstance(message, str):
      text = message.strip()
      self._reply_attempts_debug.append(f"api: content={text!r}")
      return text
    if isinstance(message, list):
      self._reply_attempts_debug.append(
          f"api: content_parts={json.dumps(message, ensure_ascii=False)}"
          f" reasoning={reasoning_text!r}"
      )
      text_parts = [
          str(part.get("text", ""))
          for part in message
          if isinstance(part, dict) and part.get("type") in {None, "text"}
      ]
      text = "".join(text_parts).strip()
      if text:
        return text
      fallback = _extract_command_from_reasoning_text(
          reasoning_text, self._task
      )
      if fallback:
        return fallback
      if allow_followup and reasoning_text:
        followup_payload = self._openai_request(
            [
                *self._messages,
                {
                    "role": "user",
                    "content": (
                        "Your previous reply did not include a final command. "
                        "Output exactly one next command from the catalog now. "
                        "Do not include reasoning or commentary."
                    ),
                },
            ],
            max_tokens=64,
            allow_compact_fallback=self._allow_compact_fallback,
        )
        return self._extract_openai_message_text(
            followup_payload,
            allow_followup=False,
        )
      raise _EmptyProviderReply(
          "OpenAI-compatible response contained no text parts in message"
          f" content: {json.dumps(choice_message, ensure_ascii=False)}"
      )
    raise RuntimeError(
        "OpenAI-compatible response returned unsupported message content type:"
        f" {type(message).__name__}"
        f" payload={json.dumps(choice_message, ensure_ascii=False)}"
    )

  def _openai_response_text(self) -> str:
    empty_attempts = 0
    while True:
      payload = self._openai_request(
          self._messages, allow_compact_fallback=self._allow_compact_fallback
      )
      try:
        text = self._extract_openai_message_text(payload)
      except _EmptyProviderReply:
        # A successful (HTTP 200) but contentless reply (empty message content /
        # no text parts / responses API 'completed' with no output). Re-send
        # verbatim under the small, dedicated empty-reply budget
        # (``_empty_reply_retries``): a transient empty may clear, while a
        # deterministic refusal stops after the cap instead of burning the full
        # request-retry budget.
        if empty_attempts < self._empty_reply_retries:
          empty_attempts += 1
          self._reply_attempts_debug.append(
              "api-empty: provider returned no usable content; retry"
              f" {empty_attempts}/{self._empty_reply_retries}"
          )
          self._sleep_before_retry(_EmptyProviderReply(), empty_attempts - 1)
          continue
        self._reply_attempts_debug.append(
            "api-empty: still empty after"
            f" {empty_attempts} retr{'y' if empty_attempts == 1 else 'ies'};"
            " giving up (generation stall)"
        )
        # Return "" (not raise) so the caller's stall handling mirrors
        # Anthropic/Gemini: DONE is coerced and generation_stall_empty_reply is
        # recorded (kept a provider_failure).
        return ""
      break
    if self._capture_thinking or self._capture_interactions:
      # Record usage (exact input/cached/output/reasoning token counts) and any
      # reasoning summary the endpoint returned. Note: if extraction issued a
      # no-command followup request, ``payload`` is the primary call, so its
      # usage reflects the dominant (first) request; the small followup's usage
      # is not separately captured.
      meta = _openai_response_meta(payload)
      meta["answer_text"] = text
      self._last_response_meta = meta
    return text

  def _openai_tool_call(self) -> ToolCall | None:
    # Native mode replays the full structured conversation (function_call /
    # function_call_output items). History trimming (``_request_messages``) is
    # chat-text-format-specific and would not understand tool turns, so it is
    # bypassed with ``trim_messages=False``; the rendered turns are passed
    # straight through. ``_openai_request`` picks the responses/chat endpoint via
    # ``_openai_prefers_responses_api`` -- the same predicate used here to choose
    # the matching renderer and parser.
    use_responses = self._openai_prefers_responses_api()
    if use_responses:
      rendered = to_openai_responses_items(self._messages)
      tools = self._native_tools_openai_responses
    else:
      rendered = to_openai_chat_messages(
          self._messages, replay_reasoning=self._replay_chat_reasoning
      )
      tools = self._native_tools_openai_chat
    empty_attempts = 0
    while True:
      payload = self._openai_request(
          rendered,
          trim_messages=False,
          tools=tools,
          # ``auto`` by default; see ``_native_tool_choice``.
          tool_choice=(
              "required" if self._native_tool_choice == "forced" else "auto"
          ),
          allow_compact_fallback=self._allow_compact_fallback,
      )
      call = (
          parse_openai_responses_tool_call(payload)
          if use_responses
          else parse_openai_chat_tool_call(payload)
      )
      no_call_text = "" if call else _openai_text_from_payload(payload)
      self._last_no_call_text = no_call_text
      meta = None
      if self._capture_thinking or self._capture_interactions:
        meta = _openai_response_meta(payload)
        meta["answer_text"] = call.name if call else ""
        self._last_response_meta = meta
      _record_dropped_parallel_calls(meta, call, self._reply_attempts_debug)
      self._reply_attempts_debug.append(
          f"api: tool_call={(call.name if call else None)!r}"
          f" args={(call.arguments if call else None)!r}"
      )
      # Empty native reply == no function call AND no prose fallback text. A
      # prose-but-no-call turn is the separate "reasoned without acting" case
      # that ``_act_native`` nudges, so it is NOT treated as empty here. Re-send
      # under the dedicated empty-reply budget before giving up.
      if call is None and not no_call_text and empty_attempts < self._empty_reply_retries:
        empty_attempts += 1
        self._reply_attempts_debug.append(
            "api-empty: provider returned no usable content; retry"
            f" {empty_attempts}/{self._empty_reply_retries}"
        )
        self._sleep_before_retry(_EmptyProviderReply(), empty_attempts - 1)
        continue
      return call

  def _gemini_content_from_message(self, types: Any, msg: dict[str, Any]) -> Any:
    """Convert one plain chat message into a Gemini ``types.Content``.

    Shared by the text path (:meth:`_gemini_response_text`) and the native tool
    path (:meth:`_gemini_tool_call`), which also emits function_call/response
    Content objects around these. Behaviour is byte-for-byte the text path's
    original inline loop.
    """
    role = "model" if msg["role"] == "assistant" else "user"
    content = msg["content"]
    parts = []
    if isinstance(content, str):
      parts.append(types.Part.from_text(text=content))
    elif isinstance(content, list):
      for part in content:
        if part["type"] == "text":
          parts.append(types.Part.from_text(text=part["text"]))
        elif part["type"] == "image_url":
          data_url = part["image_url"]["url"]
          if data_url.startswith("data:"):
            header, b64_data = data_url.split(",", 1)
            mime = header.split(";")[0].split(":")[1]
            # By default send gemini the original-size bytes (these flat UI
            # renders compress ~57:1). When --gemini-max-image-edge is set,
            # downscale any image whose long edge exceeds the cap down to it
            # (images already within the cap pass through unchanged). This trims
            # the accumulated full-history payload that pushes deep-step requests
            # past the Vertex deadline (504 DEADLINE_EXCEEDED). Caveat: LANCZOS
            # re-encoding can slightly INFLATE a few very flat renders at loose
            # caps (~2048).
            #
            # This does NOT change what the model sees: Gemini resamples every
            # image to a fixed per-image token budget server-side (measured
            # ~1100 tok/image for both 8.29 MP and 1.08 MP inputs). Use
            # ``--gemini-media-resolution`` to change the visual token budget;
            # this flag only trims upload bytes and latency.
            if self._gemini_max_image_edge:
              b64_data = _downscale_image_b64(
                  b64_data, mime, self._gemini_max_image_edge
              )
            import base64
            raw_bytes = base64.b64decode(b64_data)
            image_part = types.Part.from_bytes(
                data=raw_bytes, mime_type=mime
            )
            # Attach the per-image visual token budget, if requested. Applied
            # here (per media Part) rather than globally because "ultra_high"
            # is only valid on individual parts; see
            # ``self._gemini_media_resolution``. Left untouched when unset so
            # the default request is byte-for-byte what it was before.
            part_resolution = _gemini_part_media_resolution(
                types, self._gemini_media_resolution
            )
            if part_resolution is not None:
              image_part.media_resolution = part_resolution
            parts.append(image_part)
    return types.Content(role=role, parts=parts)

  def _gemini_thinking_config(self, types_mod):
    """Build the ThinkingConfig for a Gemini request, or None.

    Two independent switches feed ONE object: ``capture_thinking`` (return
    thought summaries) and ``reasoning_effort`` (how hard to think). They were
    previously mutually exclusive by accident -- the effort flag was never read
    on either Gemini path, so all three Gemini models ran at the model default
    while Claude and GPT ran pinned, making the cross-family comparison an
    unequal-effort one. Setting them as two separate ThinkingConfig objects
    would be just as wrong: the second assignment would drop the first.
    """
    kwargs = {}
    if self._capture_thinking:
      # Opt-in only: returns the model's thought summaries so we can inspect the
      # "failed think" behind an empty/invalid command.
      kwargs["include_thoughts"] = True
    if self._reasoning_effort:
      # Validated at construction, so the lookup cannot miss. Clamp UP to any
      # per-model floor learned at runtime (e.g. Gemini 3 Pro rejects MINIMAL),
      # so "as little as possible" becomes the lowest the model actually allows
      # instead of a level it 400s on.
      level = _GEMINI_THINKING_LEVELS[self._reasoning_effort]
      kwargs["thinking_level"] = _gemini_clamp_thinking_level(
          level, _GEMINI_MODEL_MIN_THINKING_LEVEL.get(self._model)
      )
    if not kwargs:
      # Unset on both switches: omit thinking_config entirely so the request
      # stays byte-identical to the pre-flag behavior.
      return None
    return types_mod.ThinkingConfig(**kwargs)

  def _escalate_gemini_thinking_floor(self, rejected_level: str) -> bool:
    """Raise this model's floor to one rung above ``rejected_level``.

    Keys off the level that was ACTUALLY rejected in this request, not the
    current shared floor, so it is idempotent and safe under --parallel-workers:
    many agents that all sent MINIMAL and got rejected each raise the floor to
    LOW without overshooting to MEDIUM/HIGH just because a sibling thread already
    bumped it. Only ever advances the shared floor, never lowers it. Returns
    False only when the rejected level was already the top (HIGH) -- nothing left
    to try, so the caller re-raises.
    """
    order = _GEMINI_THINKING_LEVEL_LADDER
    idx = order.index(rejected_level)
    if idx + 1 >= len(order):
      return False
    target = order[idx + 1]
    current = _GEMINI_MODEL_MIN_THINKING_LEVEL.get(self._model)
    if current is None or order.index(current) < idx + 1:
      _GEMINI_MODEL_MIN_THINKING_LEVEL[self._model] = target
    return True

  def _gemini_generate_content(self, client, types_mod, contents, config_kwargs):
    """Issue a generate_content call, healing an unsupported thinking_level.

    ``config_kwargs`` must NOT already contain ``thinking_config``: it is built
    here from the current per-model floor so that a retry after escalation picks
    up the raised level. Only the thinking-level rejection is healed in this
    loop (bounded by the ladder length); every other error propagates to the
    caller's request-retry handling unchanged.
    """
    heal_attempts = 1 + len(_GEMINI_THINKING_LEVEL_LADDER)
    last_exc: Exception | None = None
    for _ in range(heal_attempts):
      cfg = dict(config_kwargs)
      # The level actually put on the wire this attempt: the requested effort
      # clamped up to the current per-model floor -- identical to what
      # _gemini_thinking_config bakes into the ThinkingConfig. Escalation keys
      # off THIS level (not the shared floor) so it stays idempotent when 20
      # workers all send MINIMAL and get rejected at once.
      sent_level: str | None = None
      if self._reasoning_effort:
        sent_level = _gemini_clamp_thinking_level(
            _GEMINI_THINKING_LEVELS[self._reasoning_effort],
            _GEMINI_MODEL_MIN_THINKING_LEVEL.get(self._model),
        )
      thinking_config = self._gemini_thinking_config(types_mod)
      if thinking_config is not None:
        cfg["thinking_config"] = thinking_config
      try:
        return client.models.generate_content(
            model=self._model,
            contents=contents,
            config=types_mod.GenerateContentConfig(**cfg),
        )
      except Exception as exc:  # pragma: no cover - provider dependent
        last_exc = exc
        # A model that cannot go as low as the requested level: step UP to the
        # lowest it accepts, remember it, and retry -- instead of failing the
        # run on a non-retryable INVALID_ARGUMENT.
        if (
            sent_level is not None
            and _is_gemini_thinking_level_unsupported(exc)
            and self._escalate_gemini_thinking_floor(sent_level)
        ):
          continue
        raise
    if last_exc is not None:
      raise last_exc
    raise RuntimeError("Gemini request did not produce a response")

  def _gemini_response_text(self) -> str:
    client = self._get_gemini_client()
    from google.genai import types

    gemini_contents = [
        self._gemini_content_from_message(types, msg) for msg in self._messages
    ]

    config_kwargs: dict[str, Any] = dict(
        temperature=self._temperature,
        max_output_tokens=self._response_max_tokens,
        system_instruction=self._system_prompt,
    )
    # ``thinking_config`` is built inside ``_gemini_generate_content`` so an
    # unsupported thinking_level can be healed (stepped up) per model.
    last_exc = None
    empty_attempts = 0
    for attempt in range(self._request_retries + 1):
      try:
        response = self._gemini_generate_content(
            client, types, gemini_contents, config_kwargs
        )
        if self._capture_thinking or self._capture_interactions:
          answer_text, thinking_text = _gemini_extract_parts(response)
          text = answer_text.strip()
          meta = _gemini_response_meta(response)
          meta["answer_text"] = text
          meta["thinking_text"] = thinking_text
          self._last_response_meta = meta
        else:
          text = _gemini_extract_text(response).strip()
          self._last_response_meta = None
        diag = _gemini_response_diagnostics(response)
        self._reply_attempts_debug.append(f"api: content={text!r}{diag}")
        # A successful (HTTP 200) but empty reply -- typically
        # ``finish_reason=SAFETY``/``RECITATION`` or an otherwise contentless
        # candidate -- is re-sent verbatim under the small, dedicated
        # empty-reply budget (``_empty_reply_retries``, already capped at
        # ``_request_retries``): a transient block may clear, while a
        # deterministic one stops after the cap instead of burning the full
        # request-retry budget.
        if not text and empty_attempts < self._empty_reply_retries:
          empty_attempts += 1
          self._reply_attempts_debug.append(
              "api-empty: provider returned no usable content; retry"
              f" {empty_attempts}/{self._empty_reply_retries}"
          )
          self._sleep_before_retry(_EmptyProviderReply(), empty_attempts - 1)
          continue
        return text
      except Exception as exc:  # pragma: no cover - network/provider dependent
        last_exc = exc
        if not _is_retryable_llm_error(exc):
          raise
        if attempt < self._request_retries:
          self._sleep_before_retry(exc, attempt)
    assert last_exc is not None
    raise last_exc

  def _gemini_tool_call(self) -> ToolCall | None:
    client = self._get_gemini_client()
    from google.genai import types

    # Native mode replays the full structured conversation: plain chat turns are
    # converted exactly as the text path does, and tool turns become
    # function_call (model) / function_response (user) Content objects. The
    # neutral descriptors come from ``to_gemini_history_items`` so tool_calling
    # stays free of the SDK dependency.
    contents = []
    for item in to_gemini_history_items(self._messages):
      kind = item["kind"]
      if kind == "message":
        contents.append(self._gemini_content_from_message(types, item["entry"]))
      elif kind == "function_call":
        fc_part = types.Part.from_function_call(
            name=item["name"], args=item["args"]
        )
        # Gemini 2.5+ thinking models require the opaque thought_signature that
        # accompanied the original functionCall to be replayed on the history
        # part; otherwise multi-step turns fail with 400 INVALID_ARGUMENT
        # ("Function call is missing a thought_signature"). It is carried
        # base64-encoded through the canonical turn, so decode it back to bytes.
        signature_b64 = item.get("thought_signature")
        if signature_b64:
          fc_part.thought_signature = base64.b64decode(signature_b64)
        contents.append(types.Content(role="model", parts=[fc_part]))
      elif kind == "function_response":
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name=item["name"],
                        response={"output": item["output"]},
                    )
                ],
            )
        )

    declarations = (
        self._native_tools_gemini[0]["functionDeclarations"]
        if self._native_tools_gemini
        else []
    )
    tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(**decl) for decl in declarations
        ]
    )
    config_kwargs: dict[str, Any] = dict(
        temperature=self._temperature,
        max_output_tokens=self._response_max_tokens,
        system_instruction=self._system_prompt,
        tools=[tool],
        # "AUTO" by default, mirroring OpenAI 'auto' / Anthropic
        # {"type":"auto"}; "ANY" forces exactly one function call per turn. See
        # ``_native_tool_choice``. The `done` tool signals completion.
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode="ANY" if self._native_tool_choice == "forced" else "AUTO"
            )
        ),
    )
    # ``thinking_config`` is built inside ``_gemini_generate_content`` so an
    # unsupported thinking_level can be healed (stepped up) per model.
    last_exc = None
    empty_attempts = 0
    for attempt in range(self._request_retries + 1):
      try:
        response = self._gemini_generate_content(
            client, types, contents, config_kwargs
        )
        call = parse_gemini_tool_call(response)
        no_call_text = "" if call else _gemini_text_from_response(response)
        self._last_no_call_text = no_call_text
        meta = None
        if self._capture_thinking or self._capture_interactions:
          meta = _gemini_response_meta(response)
          meta["answer_text"] = call.name if call else ""
          self._last_response_meta = meta
        else:
          self._last_response_meta = None
        _record_dropped_parallel_calls(meta, call, self._reply_attempts_debug)
        diag = _gemini_response_diagnostics(response)
        self._reply_attempts_debug.append(
            f"api: tool_call={(call.name if call else None)!r}"
            f" args={(call.arguments if call else None)!r}{diag}"
        )
        # Empty native reply == no function call AND no prose fallback text
        # (typically ``finish_reason=SAFETY``). A prose-but-no-call turn is a
        # separate "reasoned without acting" case that ``_act_native`` nudges,
        # so it is NOT treated as empty here. Re-send under the dedicated
        # empty-reply budget before giving up.
        if call is None and not no_call_text and empty_attempts < self._empty_reply_retries:
          empty_attempts += 1
          self._reply_attempts_debug.append(
              "api-empty: provider returned no usable content; retry"
              f" {empty_attempts}/{self._empty_reply_retries}"
          )
          self._sleep_before_retry(_EmptyProviderReply(), empty_attempts - 1)
          continue
        return call
      except Exception as exc:  # pragma: no cover - network/provider dependent
        last_exc = exc
        if not _is_retryable_llm_error(exc):
          raise
        if attempt < self._request_retries:
          self._sleep_before_retry(exc, attempt)
    assert last_exc is not None
    raise last_exc

  def _native_tool_call(self) -> ToolCall | None:
    """Dispatch a single native tool-call request to the active provider.

    Mirrors the provider grouping used by the text path in :meth:`act`.
    """
    if self._provider in {"openai", "vertex-xai", "bedrock-gpt"}:
      return self._openai_tool_call()
    if self._provider == "gemini":
      return self._gemini_tool_call()
    # 'anthropic' (direct) and 'vertex-anthropic' (Claude on Vertex).
    return self._anthropic_tool_call()

  def reset(self, task_instruction: str, task: Task | None = None) -> None:
    self._task = task
    self._task_instruction = task_instruction
    self._messages = []
    self._pending_action = ""
    self._pending_call = None
    self._history = []
    self._done_requested = False
    self._last_reply_debug = None
    self._reply_attempts_debug = []
    self._stall_reason = None
    self._multimodal_input_mode_effective = _effective_multimodal_input_mode(
        requested=self._multimodal_input_mode,
        provider=self._provider,
        task=task,
    )
    if self._tool_mode == "native":
      # Build the per-task native tool surface once per episode: the provider-
      # agnostic defs plus each vendor's pre-formatted ``tools`` payload. The
      # command surface is conveyed by these tools, so no textual catalog is
      # appended to the system prompt.
      self._native_tool_defs = build_tool_defs(task)
      self._native_tools_openai_chat = to_openai_chat_tools(self._native_tool_defs)
      self._native_tools_openai_responses = to_openai_responses_tools(
          self._native_tool_defs
      )
      self._native_tools_anthropic = to_anthropic_tools(self._native_tool_defs)
      self._native_tools_gemini = to_gemini_tools(
          self._native_tool_defs,
          unified=(self._tool_schema_mode == "unified"),
      )
      self._system_prompt = self.NATIVE_SYSTEM_PROMPT
    else:
      self._system_prompt = (
          f"{self.BASE_SYSTEM_PROMPT}\n\nCommand catalog for this"
          f" task:\n{_command_catalog_text(task)}\n\nCatalog notes:\n- Reply"
          " with one command line or DONE.\n- Uppercase words like TIMEZONE,"
          " LOCATION, TITLE, NAME, and SCHEDULE are templates, not literal"
          " values.\n- `file` by itself is not part of this catalog.\n"
      )
    # Prime the conversation with the task. Direct/both mode attaches
    # visible image assets as OpenAI-compatible multimodal message parts;
    # path mode keeps the benchmark purely tool-mediated.
    if self._tool_mode == "native":
      task_prompt = (
          f"Task: {task_instruction}\n\n"
          "Complete the task using the available tools. "
          f"You have at most {self._max_steps_hint} steps. "
          "Call your first tool now."
      )
    else:
      task_prompt = (
          f"Task: {task_instruction}\n\n"
          "Complete the task using the available CLI commands. "
          f"You have at most {self._max_steps_hint} steps. "
          "Output your first command now."
      )
    content: str | list[dict[str, Any]] = task_prompt
    image_parts = _multimodal_image_parts_from_task(
        task,
        mode=self._multimodal_input_mode_effective,
        # Only OpenAI-compatible providers use these image_url parts directly;
        # gemini/anthropic apply their own edge caps elsewhere.
        max_edge=(
            self._openai_max_image_edge
            if self._provider in {"openai", "vertex-xai", "bedrock-gpt"}
            else None
        ),
    )
    if image_parts:
      content = [{"type": "text", "text": task_prompt}, *image_parts]
    self._messages.append({
        "role": "user",
        "content": content,
    })

  def _coerce_stall(self, reply: str) -> str:
    """Tag why generation stalled and return the episode-ending coerced DONE.

    Shared by the invalid-retry loop's natural exhaustion and the empty-reply
    short-circuit (text and native paths). A contentless ``reply`` is a
    generation stall (``generation_stall_empty_reply``); a non-empty reply that
    never validated is ``invalid_command_exhausted``. Tagging WHY keeps a
    coerced DONE from being mistaken for a genuine agent-chosen DONE, and marks
    the last captured interaction record with the same reason.
    """
    if not (reply or "").strip():
      self._stall_reason = "generation_stall_empty_reply"
    else:
      self._stall_reason = "invalid_command_exhausted"
    self._reply_attempts_debug.append(f"stall: {self._stall_reason}")
    if self._capture_interactions and self._interaction_log:
      self._interaction_log[-1]["stall_reason"] = self._stall_reason
    return "DONE"

  def act(self, observation: Observation) -> str:
    if self._tool_mode == "native":
      return self._act_native(observation)
    self._reply_attempts_debug = []
    self._interaction_log = []
    self._stall_reason = None
    # On step > 0, append the previous observation before asking for next cmd
    if observation.step_number > 0:
      if self._pending_action:
        self._history.append({
            "action": self._pending_action,
            "stdout": observation.command_output or "",
            "stderr": observation.error_output or "",
            "exit_code": observation.exit_code,
        })
      obs_text = (
          f"[STDOUT]\n{observation.command_output or '(empty)'}\n"
          f"[STDERR]\n{observation.error_output or '(none)'}\n"
          f"[EXIT CODE] {observation.exit_code}\n\n"
          "What is your next command? (or DONE if finished)"
      )
      if len(self._history) >= 3:
        recent = self._history[-3:]
        if all(
            item["action"] == recent[0]["action"]
            and item["stdout"] == recent[0]["stdout"]
            and item["stderr"] == recent[0]["stderr"]
            and item["exit_code"] == recent[0]["exit_code"]
            for item in recent[1:]
        ):
          obs_text += (
              f"\n\nRecent attempts repeated `{recent[0]['action']}` with no"
              " new result. Pick a different command from the catalog."
          )
      if self._history and all(
          is_read_only_action(item["action"]) for item in self._history
      ):
        if _instruction_implies_side_effect(self._task) and _has_required_artifact(
            self._task
        ):
          report_path = _required_report_path(self._task)
          if report_path:
            obs_text += (
                "\n\nDONE is invalid right now. You have only inspected"
                " context. "
                f"The next command must create the required report, e.g. "
                f'`file create --path {report_path} --content "<your report'
                ' with evidence anchors>"`, '
                f"then read it with `file read --path {report_path}`."
            )
          else:
            obs_text += (
                "\n\nSo far you have only inspected context. If the task asks"
                " you to create, schedule, send, or update something, choose a"
                " state-changing command before DONE."
            )
      self._messages.append({"role": "user", "content": obs_text})

    normalized = ""
    for attempt_idx in range(self._invalid_retry_limit + 1):
      # Reset per attempt so a record never reads stale metadata from a prior
      # attempt (e.g. an api-error attempt, or a non-Gemini provider).
      self._last_response_meta = None
      input_snapshot = (
          self._input_snapshot() if self._capture_interactions else None
      )
      try:
        if self._provider in {"openai", "vertex-xai", "bedrock-gpt"}:
          # 'openai', 'vertex-xai' (Grok on Vertex) and 'bedrock-gpt' (GPT on
          # Bedrock) all speak the OpenAI chat/Responses protocol; only auth,
          # base URL and endpoint choice differ.
          raw = self._openai_response_text()
        elif self._provider == "gemini":
          raw = self._gemini_response_text()
        else:
          # 'anthropic' (direct API) and 'vertex-anthropic' (Claude on Vertex)
          # share the same messages.create() interface and response shape.
          raw = self._anthropic_response_text()
      except RuntimeError as exc:
        message = str(exc)
        self._reply_attempts_debug.append(f"api-error: {message}")
        if self._provider in {"openai", "vertex-xai", "bedrock-gpt"} and (
            "empty message content" in message
            or "no text parts in message content" in message
        ):
          raw = ""
        else:
          raise
      normalized = _normalize_model_command(raw)
      self._last_reply_debug = f"raw={raw!r} normalized={normalized!r}"
      self._reply_attempts_debug.append(f"parsed: {self._last_reply_debug}")
      self._messages.append({"role": "assistant", "content": normalized or raw})
      ok, reason = _validate_command(normalized, self._task)
      if ok and normalized.upper() == "DONE":
        if self._history and all(
            is_read_only_action(item["action"]) for item in self._history
        ):
          if _instruction_implies_side_effect(
              self._task
          ) and _has_required_artifact(self._task):
            report_path = _required_report_path(self._task)
            ok = False
            if report_path:
              reason = (
                  "DONE is invalid before creating and reading the required"
                  " report. "
                  f"Reply with `file create --path {report_path} --content"
                  ' "<report>"`.'
              )
            else:
              reason = (
                  "the task likely still requires a state-changing command"
                  " before DONE"
              )
      if self._capture_interactions:
        # Built BEFORE the ``raw = normalized`` reassignment below so ``raw``
        # still holds the model's pre-normalization output for this attempt.
        self._interaction_log.append(
            self._build_interaction_record(
                attempt_idx, input_snapshot, raw, normalized, ok, reason
            )
        )
      if ok:
        raw = normalized
        break
      self._reply_attempts_debug.append(f"validator: {reason}")
      # Short-circuit an empty reply straight to a stall. The provider layer
      # already re-sent it under its dedicated empty-reply budget (with
      # backoff); re-prompting here would launch a whole new provider round --
      # each re-running those empty-reply retries -- so a deterministic refusal
      # would cost (invalid_retry_limit+1) x (empty_reply_retries+1) calls for
      # an outcome that will not change. A non-empty invalid reply still earns
      # the catalog nudge below.
      if not (normalized or "").strip():
        raw = self._coerce_stall(normalized)
        break
      self._messages.append({
          "role": "user",
          "content": (
              "Your last reply did not match the command catalog:"
              f" {reason}\nReply again with one catalog command. Do not reply"
              " DONE until the required state change is complete."
          ),
      })
    else:
      # The retry budget was exhausted without a valid command. Coercing to
      # DONE ends the episode, but we must NOT let it masquerade as a genuine
      # agent-chosen DONE (which the runner would bucket as "done early"). Tag
      # WHY generation stalled so the runner/report can account for it
      # separately: a persistently empty reply is a generation failure, not a
      # premature stop.
      raw = self._coerce_stall(normalized)

    self._pending_action = raw
    if raw.upper() == "DONE" or not raw:
      self._done_requested = True
      return "DONE"
    return raw

  def _act_native(self, observation: Observation) -> str:
    """Native function/tool-calling counterpart of :meth:`act`.

    Structurally identical to the text path -- same per-step observation
    framing, repeated-command / read-only guardrails, invalid-retry loop and
    exhausted->coerce-DONE stall handling -- but the model emits one forced,
    structured tool call per turn that is reverse-mapped to the same workspace
    CLI string text mode produces, and the conversation is preserved as typed
    tool_call / tool_result turns (full assistant-turn preservation).
    """
    self._reply_attempts_debug = []
    self._interaction_log = []
    self._stall_reason = None
    # On step > 0, replay the previous observation as the tool_result paired
    # with the tool call emitted last step. The guardrail text mirrors text
    # mode's semantics exactly, re-expressed for the tool-calling surface so the
    # native-vs-text ablation isolates the calling mechanism.
    if observation.step_number > 0:
      if self._pending_action:
        self._history.append({
            "action": self._pending_action,
            "stdout": observation.command_output or "",
            "stderr": observation.error_output or "",
            "exit_code": observation.exit_code,
        })
      obs_text = (
          f"[STDOUT]\n{observation.command_output or '(empty)'}\n"
          f"[STDERR]\n{observation.error_output or '(none)'}\n"
          f"[EXIT CODE] {observation.exit_code}\n\n"
          "Call your next tool (or the `done` tool if finished)."
      )
      if len(self._history) >= 3:
        recent = self._history[-3:]
        if all(
            item["action"] == recent[0]["action"]
            and item["stdout"] == recent[0]["stdout"]
            and item["stderr"] == recent[0]["stderr"]
            and item["exit_code"] == recent[0]["exit_code"]
            for item in recent[1:]
        ):
          obs_text += (
              f"\n\nRecent attempts repeated `{recent[0]['action']}` with no"
              " new result. Call a different tool."
          )
      if self._history and all(
          is_read_only_action(item["action"]) for item in self._history
      ):
        if _instruction_implies_side_effect(self._task) and _has_required_artifact(
            self._task
        ):
          report_path = _required_report_path(self._task)
          if report_path:
            obs_text += (
                "\n\nThe `done` tool is invalid right now. You have only"
                " inspected context. Call the `file_create` tool to create the"
                f" required report at {report_path} (with your report and"
                " evidence anchors), then read it with the `file_read` tool."
            )
          else:
            obs_text += (
                "\n\nSo far you have only inspected context. If the task asks"
                " you to create, schedule, send, or update something, call a"
                " state-changing tool before the `done` tool."
            )
      if self._pending_call is not None:
        self._messages.append(tool_result_turn(self._pending_call, obs_text))
      else:
        # Defensive: no tool call was pending (e.g. a prior coerced DONE). Give
        # the observation as a plain user turn instead.
        self._messages.append({"role": "user", "content": obs_text})

    cli = ""
    for attempt_idx in range(self._invalid_retry_limit + 1):
      # Reset per attempt so a record never reads stale metadata from a prior
      # attempt.
      self._last_response_meta = None
      self._last_no_call_text = ""
      input_snapshot = (
          self._input_snapshot() if self._capture_interactions else None
      )
      call = self._native_tool_call()
      if call is None:
        # The model answered in text instead of calling a tool. Routine under
        # ``tool_choice=auto`` (the default) and rare but not impossible under a
        # forced choice. Measured over the existing Claude eval runs: 13/8553
        # (0.15%) of opus-4-8 turns and 1/7517 sonnet-5 turns.
        no_call_text = self._last_no_call_text
        self._reply_attempts_debug.append("validator: no tool call returned")
        if self._capture_interactions:
          self._interaction_log.append(
              self._build_interaction_record(
                  attempt_idx,
                  input_snapshot,
                  no_call_text,
                  "",
                  False,
                  "no tool call returned",
              )
          )
        # Short-circuit a truly empty reply (no tool call AND no prose) straight
        # to a stall: the provider layer already re-sent it under its dedicated
        # empty-reply budget, so nudging would launch a whole new provider round
        # (each re-running those retries). Only a prose-but-no-call turn -- the
        # model reasoned without acting -- still earns the nudge below.
        if not no_call_text:
          cli = self._coerce_stall("")
          break
        nudge = (
            "You did not call a tool. Call exactly one tool now (or the"
            " `done` tool if the task is complete)."
        )
        # Keep the model's own words in the history. Dropping them made the
        # nudge look like a non-sequitur (the model saw its turn vanish and
        # often reproduced it verbatim), and it silently deleted any reasoning
        # the text turn contained. The thinking blocks themselves still cannot
        # be replayed -- they are only carried on a ToolCall -- so this records
        # the visible text as an ordinary assistant turn, which every renderer
        # passes through unchanged.
        if no_call_text:
          self._messages.append({"role": "assistant", "content": no_call_text})
        # Avoid emitting two consecutive user turns (rejected by some providers'
        # role-alternation checks): fold the nudge into a trailing plain user
        # turn when the last message is one. Appending the assistant turn above
        # makes that case impossible, which is why this runs after it.
        last = self._messages[-1] if self._messages else None
        if (
            isinstance(last, dict)
            and last.get("role") == "user"
            and isinstance(last.get("content"), str)
        ):
          last["content"] = f"{last['content']}\n\n{nudge}"
        else:
          self._messages.append({"role": "user", "content": nudge})
        continue

      try:
        cli = command_from_tool_call(call.name, call.arguments)
        map_error: str | None = None
      except KeyError as exc:
        cli = ""
        map_error = f"invalid tool call: {exc}"
      raw_repr = f"{call.name}({call.arguments_json()})"
      self._last_reply_debug = f"tool_call={raw_repr} cli={cli!r}"
      self._reply_attempts_debug.append(f"parsed: {self._last_reply_debug}")
      # Record the assistant tool call turn; it is paired below with either the
      # next observation (on success) or the rejection nudge (on failure).
      self._messages.append(assistant_tool_turn(call))
      self._pending_call = call

      if map_error is not None:
        ok, reason = False, map_error
      else:
        ok, reason = _validate_command(cli, self._task)
        if ok and cli.upper() == "DONE":
          if self._history and all(
              is_read_only_action(item["action"]) for item in self._history
          ):
            if _instruction_implies_side_effect(
                self._task
            ) and _has_required_artifact(self._task):
              report_path = _required_report_path(self._task)
              ok = False
              if report_path:
                reason = (
                    "The `done` tool is invalid before creating and reading the"
                    f" required report. Call the `file_create` tool for"
                    f" {report_path}."
                )
              else:
                reason = (
                    "the task likely still requires a state-changing tool"
                    " before the `done` tool"
                )
      if self._capture_interactions:
        self._interaction_log.append(
            self._build_interaction_record(
                attempt_idx, input_snapshot, raw_repr, cli, ok, reason
            )
        )
      if ok:
        break
      self._reply_attempts_debug.append(f"validator: {reason}")
      # Pair the just-recorded assistant tool call with a tool_result carrying
      # the nudge: every tool_use MUST be followed by a tool_result before the
      # next model turn (Anthropic requirement; harmless for the others).
      self._messages.append(
          tool_result_turn(
              call,
              f"Your tool call was rejected: {reason}\nCall one valid tool now."
              " Do not call the `done` tool until the required state change is"
              " complete.",
          )
      )
    else:
      # Retry budget exhausted without a valid tool call. Coerce DONE but tag
      # WHY so the runner does not count it as a genuine agent-chosen DONE.
      cli = self._coerce_stall(cli)

    self._pending_action = cli
    if cli.upper() == "DONE" or not cli:
      self._done_requested = True
      return "DONE"
    return cli

  def should_stop(self) -> bool:
    return self._done_requested

  def debug_last_reply(self) -> str | None:
    return self._last_reply_debug

  def debug_reply_attempts(self) -> list[str]:
    return list(self._reply_attempts_debug)

  def _image_ref(self, part: dict[str, Any]) -> dict[str, Any]:
    """Compact reference to an inline image (never store the bytes)."""
    try:
      data_url = part["image_url"]["url"]
    except Exception:
      return {}
    if not isinstance(data_url, str) or not data_url.startswith("data:"):
      return {"url": str(data_url)[:200]}
    try:
      header, b64_data = data_url.split(",", 1)
      mime = header.split(";")[0].split(":")[1]
      import base64
      raw = base64.b64decode(b64_data)
    except Exception:
      return {}
    ref: dict[str, Any] = {
        "mime": mime,
        "bytes_len": len(raw),
        "sha1": _sha1_bytes(raw),
    }
    try:
      import io
      from PIL import Image
      with Image.open(io.BytesIO(raw)) as image:
        ref["dims"] = list(image.size)
    except Exception:  # pragma: no cover - PIL optional / decode best-effort
      pass
    return ref

  def _input_snapshot(self) -> dict[str, Any]:
    """Compact record of what the model was shown for this attempt.

    Full message history is re-sent every step (quadratic) and carries huge
    base64 images, so we store only the newest user turn's text + image
    references (dims/size/hash, never bytes) plus a system-prompt hash. The full
    system prompt is emitted once per episode via ``debug_system_prompt``.
    """
    last_user_text = ""
    image_refs: list[dict[str, Any]] = []
    for msg in reversed(self._messages):
      # Native mode delivers the newest observation as a tool_result turn
      # (role 'tool'); surface its output as the last input shown to the model.
      if msg.get("role") == "tool" and "tool_result" in msg:
        last_user_text = str((msg["tool_result"] or {}).get("output", ""))
        break
      if msg.get("role") != "user":
        continue
      content = msg.get("content")
      if isinstance(content, str):
        last_user_text = content
      elif isinstance(content, list):
        texts = []
        for part in content:
          if not isinstance(part, dict):
            continue
          if part.get("type") == "text":
            texts.append(str(part.get("text", "")))
          elif part.get("type") == "image_url":
            image_refs.append(self._image_ref(part))
        last_user_text = "\n".join(texts)
      break
    return {
        "system_prompt_sha1": _sha1(self._system_prompt),
        "last_user_text": last_user_text,
        "image_refs": image_refs,
        "history_len": len(self._messages),
    }

  def _build_interaction_record(
      self,
      attempt_idx: int,
      input_snapshot: dict[str, Any] | None,
      raw: str,
      normalized: str,
      ok: bool,
      reason: str | None,
  ) -> dict[str, Any]:
    """One structured record per model attempt for the sidecar log."""
    meta = self._last_response_meta or {}
    return {
        "attempt_index": attempt_idx,
        "provider": self._provider,
        "model": self._model,
        "input": input_snapshot,
        "response": {
            "answer_text": meta.get("answer_text", raw),
            "thinking_text": meta.get("thinking_text", ""),
            "finish_reason": meta.get("finish_reason"),
            "finish_message": meta.get("finish_message"),
            "safety_ratings": meta.get("safety_ratings", []),
            "block_reason": meta.get("block_reason"),
            "usage": meta.get("usage", {}),
            # Provider-agnostic visible/reasoning split derived from the raw
            # ``usage`` above, which stays verbatim for audit. Consumers should
            # read this rather than re-deriving the four provider key paths.
            "usage_normalized": _normalized_usage(
                self._provider, meta.get("usage", {})
            ),
            # Names of any additional tool calls the model emitted in the same
            # response that the one-action-per-step loop could not run. Almost
            # always empty; non-empty means the model's plan was truncated, which
            # would otherwise be invisible in the logs.
            "dropped_parallel_calls": meta.get("dropped_parallel_calls", []),
        },
        "raw": raw,
        "normalized": normalized,
        "validation": {"ok": bool(ok), "reason": None if ok else reason},
        "stall_reason": None,
        "accepted": bool(ok),
    }

  def debug_interactions(self) -> list[dict[str, Any]]:
    """Structured per-attempt interaction records for the last act() call.

    Populated only when the agent was built with ``capture_interactions=True``.
    Unlike the trajectory ``model_debug`` strings, this includes the terminal
    turn that coerces DONE (invalid_command_exhausted / generation stall) plus
    per-call response metadata for every provider (Gemini, OpenAI/vertex-xai,
    Anthropic): token ``usage`` and, when the model returns it, the reasoning /
    thought text.
    """
    return [dict(rec) for rec in self._interaction_log]

  def debug_system_prompt(self) -> str:
    return self._system_prompt

  def debug_stall_reason(self) -> str | None:
    """Non-None when the last act() coerced DONE after exhausting retries.

    Distinguishes a generation stall (empty/invalid-command exhaustion) from a
    genuine agent-chosen DONE so the runner records an accurate stop_reason.
    """
    return self._stall_reason

  def _reasoning_effort_effective(self) -> str:
    """The reasoning level the requests actually carry, or "" if none do.

    ``reasoning_effort`` records what was ASKED for; this records what the wire
    format can express for the current provider/mode combination. They diverge
    silently, which is how all three Gemini models came to be reported at an
    effort they never ran at:

    * gemini  - sent on both the text and native paths (ThinkingConfig), but
                clamped UP to a per-model floor learned at runtime (some models,
                e.g. Gemini 3 Pro, reject MINIMAL), so this can read one rung
                above what was requested.
    * openai  - on the Responses path, only under ``_openai_reasoning_enabled``
                (native mode + native reasoning); a text-mode run drops it. On
                the Chat Completions path only Kimi K3 carries it, reported at
                the low/high/max tier actually sent rather than the rung typed
                (the map stretches: harness 'medium' goes out as K3 'high').
    * claude  - the adaptive rung of the reasoning ladder (thinking ON), OR a
                "no think" request (``_EFFORT_OFF_VALUES`` on the native tool
                path, thinking explicitly disabled). Both can degrade at runtime,
                so a non-empty value here means "sent at the start", not "sent on
                every call".
    """
    if not self._reasoning_effort:
      return ""
    if self._provider == "gemini":
      # Report the level actually sent: clamped UP to any learned per-model floor
      # (e.g. Gemini 3 Pro's MINIMAL->LOW) and mapped back to the shared
      # vocabulary. Both 'none' and 'minimal' send MINIMAL -> reported 'minimal'
      # (Gemini has no true "off"). The floor is learned at runtime, so before
      # the first call this reflects the requested level.
      level = _gemini_clamp_thinking_level(
          _GEMINI_THINKING_LEVELS[self._reasoning_effort],
          _GEMINI_MODEL_MIN_THINKING_LEVEL.get(self._model),
      )
      return _GEMINI_LEVEL_TO_EFFORT[level]
    if self._provider in {"openai", "vertex-xai", "bedrock-gpt"}:
      # Kimi K3 is the one model on the Chat Completions path that carries the
      # field, so it would otherwise be recorded as "" -- an effort-pinned run
      # filed as if it had run at the model default, exactly the Gemini failure
      # this method exists to prevent.
      kimi_effort = self._kimi_chat_effort()
      if kimi_effort:
        return kimi_effort
      # Map onto OpenAI's wire vocabulary so the recorded level is what the
      # request actually carried (e.g. 'minimal'->'low', 'max'->'xhigh'), not
      # the canonical name the user typed.
      return (
          _OPENAI_EFFORT_LEVELS[self._reasoning_effort]
          if self._openai_reasoning_enabled()
          else ""
      )
    # "No think" on the native Claude path: thinking is turned OFF explicitly
    # (``thinking={"type":"disabled"}``), a request shape distinct from the
    # adaptive/budget ladder that ``_anthropic_native_reasoning_enabled`` tracks.
    # Report the requested off level while it is still being sent; it degrades to
    # "" if a backend rejects the disabled params (flag cleared).
    if (
        self._provider in {"anthropic", "vertex-anthropic"}
        and self._tool_mode == "native"
        and self._reasoning_effort in _EFFORT_OFF_VALUES
    ):
      return (
          self._reasoning_effort
          if self._anthropic_disable_thinking_supported
          else ""
      )
    if self._anthropic_native_reasoning_enabled():
      # Same: report the effort as clamped into Anthropic's low/medium/high
      # (e.g. 'xhigh'/'max'->'high'), which is what output_config carried.
      return _ANTHROPIC_EFFORT_LEVELS[self._reasoning_effort]
    return ""

  def debug_run_config(self) -> dict[str, str]:
    return {
        "provider": self._provider,
        "model": self._model,
        "history_mode": self._history_mode,
        "multimodal_input_mode": self._multimodal_input_mode,
        "multimodal_input_mode_effective": (
            self._multimodal_input_mode_effective
        ),
        # Records native vs text function-calling surface for run provenance
        # (surfaces in the per-episode trajectory sidecar's run_config).
        "tool_mode": self._tool_mode,
        # Schema dialect (native per-provider vs unified canonical JSON-Schema);
        # provenance for the --llm-tool-schema ablation.
        "tool_schema_mode": self._tool_schema_mode,
        # Explicit reasoning-effort override, or "" when each vendor's own
        # default applies (which is NOT the same level across vendors: Anthropic
        # defaults to high, gpt-5.5 to medium). Recorded so a result set states
        # which effort produced it instead of leaving it to be inferred.
        "reasoning_effort": self._reasoning_effort or "",
        # What the requests actually carry. Differs from the line above whenever
        # the provider/mode combination cannot express the requested level -- the
        # gap that let a Gemini run be recorded as pinned while every request
        # went out at the model default.
        "reasoning_effort_effective": self._reasoning_effort_effective(),
        # Server-side chat-template variables actually sent. For models whose
        # only reasoning control lives in the template (Qwen's enable_thinking),
        # this -- not reasoning_effort -- is the field that says whether the run
        # was a thinking run, so it has to be in the provenance record.
        "chat_template_kwargs": json.dumps(
            self._chat_template_kwargs, sort_keys=True
        ) if self._chat_template_kwargs else "",
        # Whether past turns' thinking was re-inlined into the chat history.
        # Changes what the model sees on every step after the first, so a run
        # with it on is not comparable to one with it off.
        "replay_chat_reasoning": "true" if self._replay_chat_reasoning else "",
        # "auto" (model may answer without calling a tool) vs "forced" (exactly
        # one call per step). Recorded because it materially changes the task
        # and is not otherwise recoverable from a finished run.
        "native_tool_choice": self._native_tool_choice,
        # Image sizing, recorded per provider because the three caps have
        # ASYMMETRIC defaults (anthropic "auto"=uncapped, gemini None=off,
        # openai 1920) and an unset flag therefore does not mean "unchanged".
        # A finished run must state the size it used rather than leave it to be
        # reverse-engineered from token counts.
        "max_image_edge": _run_config_image_edge(self),
        # Gemini-only: the per-image visual token budget. "" means the server
        # default (1120 tokens for Gemini 3 images). Unlike max_image_edge this
        # is what actually governs Gemini's model-visible detail.
        "gemini_media_resolution": self._gemini_media_resolution or "",
    }

def _run_config_image_edge(agent: Any) -> str:
  """Effective image long-edge cap for ``agent``'s provider, as a string.

  Returns "" when the provider applies no client-side cap. Anthropic's "auto"
  is reported as "auto" (uncapped until the server rejects an oversized image)
  or as the learned cap once one has been observed, so the recorded value
  reflects what the run actually sent rather than the flag as typed.
  """
  provider = agent._provider  # pylint: disable=protected-access
  if provider in {"openai", "vertex-xai", "bedrock-gpt"}:
    edge = agent._openai_max_image_edge  # pylint: disable=protected-access
    return str(edge) if edge else ""
  if provider == "gemini":
    edge = agent._gemini_max_image_edge  # pylint: disable=protected-access
    return str(edge) if edge else ""
  if provider in {"anthropic", "vertex-anthropic"}:
    edge = agent._current_anthropic_image_edge()  # pylint: disable=protected-access
    if edge:
      return str(edge)
    # Fixed-but-zero is not reachable; a falsy edge here means "auto, nothing
    # learned yet", i.e. originals were sent and the server did its own cap.
    return "auto"
  return ""

