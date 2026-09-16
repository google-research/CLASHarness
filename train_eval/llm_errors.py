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

"""LLM error-classification helpers extracted from examples/train_and_eval.py."""
from __future__ import annotations

# ``_is_retryable_llm_error`` and ``_retry_backoff_delay`` were relocated to the
# lower ``clasharness`` layer so the in-layer LLM judge can reuse ONE retry
# policy without a backwards (train_eval -> clasharness is fine; the reverse is
# not) dependency. They are re-exported here so existing importers
# (agents.py, train_and_eval.py, tests) keep working unchanged.
from clasharness.evaluation.llm_retry import (  # noqa: F401
    _is_flaky_safety_llm_error,
    _is_retryable_llm_error,
    _retry_backoff_delay,
)


def _is_deadline_error(exc: Exception) -> bool:
  """True for the "ran out of time" class: HTTP 504 / DEADLINE_EXCEEDED or a
  client-side request timeout.

  Distinct from throttling/unavailability (429/503): here the backend accepted
  the request but could not finish in time (usually under concurrent load), so a
  longer backoff lets it drain before we re-send the same (heavy) request. Two
  manifestations of the same overload are treated alike:

  * Server-side 504 -- when our per-request timeout is looser than the backend's
    own deadline. gemini/Vertex APIError carries ``.code == 504``, anthropic
    carries ``.status_code == 504``, and the openai-compatible path wraps it as
    ``"HTTP 504: ..."``.
  * Client-side timeout -- when ``--llm-timeout-s`` fires first (e.g. the 90s cap
    aborts before the backend's ~180s deadline). Without this branch the long
    backoff would rarely trigger, since the overload surfaces as a timeout, not a
    504.
  """
  for attr in ("code", "status_code"):
    code = getattr(exc, attr, None)
    if isinstance(code, bool):
      continue
    if isinstance(code, int) and code == 504:
      return True
  msg = str(exc).lower()
  deadline_markers = (
      "deadline_exceeded",
      "deadline exceeded",
      "deadline expired",
      "gateway timeout",
      "http 504",
      '"code":504',
      '"code": 504',
      '"status":504',
      '"status": 504',
      "timed out",
      "timeout",
  )
  return any(marker in msg for marker in deadline_markers)

def _is_gemini_thinking_level_unsupported(exc: Exception) -> bool:
  """True when Gemini rejects the requested ThinkingConfig.thinking_level.

  Not every model accepts every level: Gemini 3 Pro rejects MINIMAL (its floor
  is LOW) and 400s with "thinking_level MINIMAL is not supported by this model",
  while Flash accepts MINIMAL. Detected so the caller can step the level UP to
  the lowest the model accepts and retry, instead of failing the run on an
  INVALID_ARGUMENT that the normal retry path treats as non-retryable.
  """
  msg = str(exc).lower()
  if "thinking_level" not in msg and "thinking level" not in msg:
    return False
  markers = ("not supported", "not allowed", "unsupported", "invalid")
  return any(marker in msg for marker in markers)

def _is_content_filter_llm_error(exc: Exception | str) -> bool:
  msg = str(exc).lower()
  markers = (
      "content_filter",
      "filtered due to the prompt",
      "responsibleaipolicyviolation",
      '"param":"prompt"',
      '"param": "prompt"',
      '"code":"content_filter"',
      '"code": "content_filter"',
      '"jailbreak"',
      # Some providers reject a policy-violating prompt as a 400 with an
      # ``invalid_prompt`` code rather than a ``content_filter`` marker. Treat
      # it the same so the caller can fall back to a compact prompt.
      "invalid_prompt",
      '"code":"invalid_prompt"',
      '"code": "invalid_prompt"',
      "flagged as potentially violating",
  )
  return any(marker in msg for marker in markers)

# A terminal empty-reply generation stall carries NO error string: the HTTP
# call succeeded and the provider simply returned an empty body, repeatedly,
# until the invalid-reply retry budget was exhausted. Without this the stall
# falls through to "logical_failure" below (empty ``error`` -> reasoning
# mistake), which charges a provider non-response to the model.
#
# This also removes a VENDOR ASYMMETRY. The same refusal surfaces differently
# per provider: the OpenAI-compatible gateway rejects with an explicit HTTP 400
# ("flagged for potentially high-risk cyber activity"), which already matches
# ``provider_markers`` and classifies as provider_failure, while Anthropic and
# Gemini return an empty body and classified as logical_failure. Observed on the
# SAME task (conflict_vlm_payment_chargeback_dispute_v03_r08): gpt-5.5 and terra
# -> provider_failure (retry-eligible), opus-5 -> logical_failure (not
# retry-eligible). Treat both shapes as the provider failing to answer.
#
# Visibility is NOT lost by reclassifying: ``EvalSummary.generation_stall_tasks``
# keys off ``stop_reason`` independently, so the stall count stays reported.
_GENERATION_STALL_EMPTY_REPLY = "generation_stall_empty_reply"


def _classify_error_type(
    error: str, success: bool, stop_reason: str = ""
) -> str:
  if success:
    return ""
  if (stop_reason or "").strip() == _GENERATION_STALL_EMPTY_REPLY:
    return "provider_failure"
  text = (error or "").strip().lower()
  if not text:
    return "logical_failure"
  provider_markers = (
      "http 429",
      "too many requests",
      "queue is full",
      "content_filter",
      "http 500",
      "http 502",
      "http 503",
      "http 504",
      # Vertex/gRPC surface deadlines and transient failures without an "http NNN"
      # prefix, e.g. "504 DEADLINE_EXCEEDED ... Deadline expired ...". Match the
      # status names directly so gemini deadlines classify as provider (not runner)
      # failures.
      "deadline_exceeded",
      "deadline expired",
      "resource_exhausted",
      "unavailable",
      "read operation timed out",
      "timed out",
      "openai-compatible request failed",
      "empty message content",
      "no text parts in message content",
      "rate limit",
      # Vertex rejects requests over 30MB with a 400 "The request size (N bytes)
      # exceeds 30.000MB limit. FAILED_PRECONDITION". This is a provider-side
      # payload cap, not a solver mistake: the agent has no control over how many
      # round-visible images the scenario accumulates. Even with proactive
      # byte-budget downscaling (see _fit_anthropic_messages_to_byte_budget), a
      # residual oversize must be charged to the provider, not the model.
      "request size",
      "failed_precondition",
  )
  if any(marker in text for marker in provider_markers):
    return "provider_failure"
  return "runner_failure"

def _has_provider_noise(
    debug_lines: list[str] | None = None, error: str = ""
) -> bool:
  text_parts = [str(item) for item in debug_lines or [] if str(item).strip()]
  if error:
    text_parts.append(str(error))
  haystack = "\n".join(text_parts).lower()
  if not haystack.strip():
    return False
  markers = (
      "api-fallback: content filter triggered",
      "content_filter",
      "empty message content",
      "no text parts in message content",
      "api-error: openai-compatible request failed",
      "api-error: openai-compatible response contained empty message content",
      "api: content=none reasoning=",
      # Emitted as ``stall: generation_stall_empty_reply`` by the agent's
      # retry-exhausted branch, which lives in provider-INDEPENDENT code, so all
      # vendors leave the same breadcrumb. Needed because only the
      # OpenAI-compatible path raises (and therefore logs an ``api-error: ...
      # empty message content`` line) on an empty body; Anthropic and Gemini
      # return "" silently and would otherwise leave no provider-noise trace at
      # all, making ``provider_impacted`` fire for one vendor but not the others
      # on the exact same event.
      _GENERATION_STALL_EMPTY_REPLY,
  )
  return any(marker in haystack for marker in markers)

