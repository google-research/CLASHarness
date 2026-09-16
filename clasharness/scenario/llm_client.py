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

"""LLM clients for generation-time PRO scenario expansion."""

from __future__ import annotations

import base64
import json
import os
import random
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from clasharness.evaluation.llm_retry import (
    _is_anthropic_temperature_unsupported,
    anthropic_drop_temperature,
    anthropic_kwargs_carry_temperature,
    anthropic_rejects_sampling_params,
    anthropic_temperature_kwargs,
    anthropic_text_from_response,
    note_anthropic_sampling_rejected,
)
from clasharness.scenario import design_archetypes


class ScenarioLLMError(RuntimeError):
    """Raised when a scenario-generation LLM call fails."""


class NonRetryableScenarioLLMError(ScenarioLLMError):
    """Raised when retrying cannot fix the provider error."""


class ScenarioLLMClient(Protocol):
    """Minimal generation-time client interface."""

    def generate_layer(self, layer: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Generate one scenario-forge-style layer as JSON data."""

    def audit_image_fidelity(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        expected_token: str,
        context: str = "",
    ) -> dict[str, Any]:
        """Optional multimodal capability: is ``expected_token`` legible in the image?

        Returns ``{legible: bool, seen_text: str, reason: str, skipped: bool}``.
        Clients without vision return ``skipped=True`` (a benign pass). Callers
        should also tolerate clients that do not implement this method at all
        (treat a missing method as skipped).
        """
        ...


# --- Vision fidelity audit (multimodal) -------------------------------------
# Shared contract + result parsing for the "is this exact reading legibly
# rendered in the generated image?" check used by DECISIVE image assets. Kept
# provider-agnostic so the Vertex (google-genai SDK) and Gemini (REST) clients
# share one prompt/parse; only image-byte attachment differs per client.
_VISION_FIDELITY_SYSTEM = (
    "You are a strict visual-QA auditor for an image benchmark. You receive a "
    "generated image that is supposed to legibly display a specific reading (a "
    "label, version, serial, count, or status word) as real physical text in "
    "the scene. Judge ONLY whether that exact reading is clearly and "
    "unambiguously readable in the image. Return one strict JSON object and "
    "nothing else: {\"legible\": <true|false>, \"seen_text\": \"<the relevant "
    "text you can actually read, or empty>\", \"reason\": \"<short reason>\"}. "
    "Set legible=false if the reading is absent, misspelled, garbled, "
    "truncated, blurry, or too small to read with confidence."
)


def _vision_fidelity_ask(expected_token: str, context: str = "") -> str:
    """Build the user-turn JSON describing what reading must be legible."""
    return json.dumps(
        {"expected_reading": expected_token, "scene_context": context or ""},
        ensure_ascii=False,
    )


def _parse_vision_fidelity(text_response: str) -> dict[str, Any]:
    """Normalize a vision-audit JSON response into the shared result shape."""
    data = extract_json_object(text_response or "")
    return {
        "legible": bool(data.get("legible")),
        "seen_text": str(data.get("seen_text") or ""),
        "reason": str(data.get("reason") or ""),
        "skipped": False,
    }


_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def _repair_nested_quotes_line_by_line(text: str) -> str:
    lines = text.split('\n')
    repaired_lines = []
    for line in lines:
        # Check key-value line pattern: e.g.   "key": "value",
        kv_match = re.match(r'^(\s*"[^"]+":\s*)"(.*)"\s*(,?)\s*$', line)
        if kv_match:
            prefix = kv_match.group(1)
            val = kv_match.group(2)
            suffix = kv_match.group(3)
            # Escape all double quotes in val that are not already escaped
            escaped_val = re.sub(r'(?<!\\)"', r'\"', val)
            line = f'{prefix}"{escaped_val}"{suffix}'
        else:
            # Check array string item line pattern: e.g.   "item",
            arr_match = re.match(r'^(\s*)"(.*)"\s*(,?)\s*$', line)
            if arr_match:
                prefix = arr_match.group(1)
                val = arr_match.group(2)
                suffix = arr_match.group(3)
                # Ensure it's not a key-value line that failed the first check
                if not re.search(r'":\s*', line):
                    escaped_val = re.sub(r'(?<!\\)"', r'\"', val)
                    line = f'{prefix}"{escaped_val}"{suffix}'
        repaired_lines.append(line)
    return '\n'.join(repaired_lines)


def _repair_json_string(text: str) -> str:
    """Repair common loose JSON formatting issues (trailing commas, nested quotes, mismatched brackets)."""
    # First, repair single-line nested unescaped quotes
    text = _repair_nested_quotes_line_by_line(text)

    # 1. Strip trailing commas in objects and arrays
    text = re.sub(r',(\s*[\]}])', r'\1', text)

    # 2. Escape literal newlines/tabs and repair mismatched brackets
    chars = []
    stack = []
    in_quote = False
    escape = False
    
    i = 0
    while i < len(text):
        c = text[i]
        
        # Handle string quotes
        if c == '"' and not escape:
            in_quote = not in_quote
            chars.append(c)
            i += 1
            continue
            
        if in_quote:
            # Escape literal control characters inside quotes. Models frequently
            # emit raw newlines/tabs/carriage-returns (and occasionally other
            # C0 control chars) inside HTML/rich-text string values, which strict
            # JSON rejects. Escape them all so the value round-trips.
            if escape:
                # Previous char was a backslash: this char is already part of an
                # escape sequence, pass it through untouched.
                chars.append(c)
            elif c == '\n':
                chars.append('\\n')
            elif c == '\t':
                chars.append('\\t')
            elif c == '\r':
                chars.append('\\r')
            elif ord(c) < 0x20:
                chars.append('\\u%04x' % ord(c))
            else:
                chars.append(c)
            escape = (c == '\\' and not escape)
            i += 1
            continue
            
        # Outside quotes: track braces/brackets
        if c == '{':
            stack.append('{')
            chars.append(c)
        elif c == '[':
            stack.append('[')
            chars.append(c)
        elif c == '}':
            if stack:
                top = stack[-1]
                if top == '{':
                    stack.pop()
                    chars.append(c)
                elif top == '[':
                    # Mismatch! Expecting ']' but got '}'
                    stack.pop()
                    chars.append(']')
            else:
                chars.append(c)
        elif c == ']':
            if stack:
                top = stack[-1]
                if top == '[':
                    stack.pop()
                    chars.append(c)
                elif top == '{':
                    # Mismatch! Expecting '}' but got ']'
                    stack.pop()
                    chars.append('}')
            else:
                chars.append(c)
        else:
            chars.append(c)
            
        escape = False
        i += 1
        
    return "".join(chars)


def _escape_inner_quotes_scanner(text: str) -> str:
    """Best-effort repair of unescaped double quotes inside JSON string values.

    Walks the text tracking string state. A double quote encountered inside a
    string is treated as the string terminator only when the next non-space
    character is a JSON structural token (``,`` ``}`` ``]`` ``:``) or EOF;
    otherwise it is an unescaped inner quote and is escaped. This recovers the
    single most common model failure: command strings or prose that contain
    literal double quotes (e.g. ``message send --message "hi"``) emitted without
    escaping, which the line-by-line repair cannot reliably fix. It is only used
    as a last-resort fallback after strict parsing has already failed.
    """
    out: list[str] = []
    in_str = False
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if not in_str:
            out.append(c)
            if c == '"':
                in_str = True
            i += 1
            continue
        # Inside a string value.
        if c == '\\':
            # Preserve existing escape sequences verbatim.
            out.append(c)
            if i + 1 < n:
                out.append(text[i + 1])
                i += 2
            else:
                i += 1
            continue
        if c == '"':
            j = i + 1
            while j < n and text[j] in ' \t\r\n':
                j += 1
            nxt = text[j] if j < n else ''
            if nxt in (',', '}', ']', ':', ''):
                out.append(c)          # legitimate closing quote
                in_str = False
            else:
                out.append('\\"')      # unescaped inner quote -> escape it
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


_VALID_JSON_ESCAPE_NEXT = set('"\\/bfnrt')
_HEX_DIGITS = set('0123456789abcdefABCDEF')


def _escape_invalid_backslashes(text: str) -> str:
    """Escape invalid backslash escape sequences inside JSON string values.

    JSON permits a backslash only before one of ``" \\ / b f n r t`` or a
    ``\\uXXXX`` sequence. Models frequently emit a lone backslash that begins an
    *invalid* escape -- overwhelmingly a Markdown backtick written as a
    backslash-backtick (e.g. ``run \\`asset list\\```), but also ``\\_``,
    ``\\d`` or a bare Windows path like ``C:\\Users``. strict ``json.loads``
    rejects these with "Invalid \\escape", and neither
    :func:`_repair_json_string` nor :func:`_escape_inner_quotes_scanner` fixes
    them: both assume any character after a backslash is already a valid escape
    and pass it through verbatim.

    This pass walks string state and doubles any backslash that does not begin a
    valid escape. It is lossless: a valid ``\\n`` or ``\\uABCD`` is preserved,
    while an invalid ``\\``` becomes ``\\\\``` so the decoded value round-trips
    to the literal backslash the model emitted. Used only as a repair candidate
    after strict parsing has already failed.
    """
    out: list[str] = []
    in_str = False
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if not in_str:
            out.append(c)
            if c == '"':
                in_str = True
            i += 1
            continue
        # Inside a string value.
        if c == '"':
            out.append(c)
            in_str = False
            i += 1
            continue
        if c == '\\':
            nxt = text[i + 1] if i + 1 < n else ''
            if nxt in _VALID_JSON_ESCAPE_NEXT:
                # Valid two-character escape (includes \\ and \"): emit both and
                # skip past them so the escaped char is never re-examined (a \"
                # must not toggle string state, a \\ must not shadow the next
                # char).
                out.append(c)
                out.append(nxt)
                i += 2
                continue
            if (
                nxt == 'u'
                and i + 5 < n
                and all(ch in _HEX_DIGITS for ch in text[i + 2 : i + 6])
            ):
                # Valid \uXXXX escape: emit all six chars untouched.
                out.append(text[i : i + 6])
                i += 6
                continue
            # Invalid escape (or a trailing backslash at EOF): double it so it
            # becomes a literal backslash and the next char stands on its own.
            out.append('\\\\')
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _json_open_container_closers(text: str) -> str | None:
    """Closers needed to balance the ``{``/``[`` still open at the end of ``text``.

    Scans left-to-right honoring string state and escapes, and returns the
    ``}``/``]`` sequence (innermost last) that would close every container still
    open. Returns ``None`` when ``text`` ends *inside* an unterminated string,
    which cannot be safely closed by appending delimiters alone.
    """
    stack: list[str] = []
    in_str = False
    escape = False
    for ch in text:
        if in_str:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == '{':
            stack.append('}')
        elif ch == '[':
            stack.append(']')
        elif ch == '}' or ch == ']':
            if stack:
                stack.pop()
    if in_str:
        return None
    return ''.join(reversed(stack))


def _close_unbalanced_json(text: str) -> str:
    r"""Recover a JSON object mangled by terminal degeneration or truncation.

    Two very common LLM failure modes leave an otherwise-complete object
    unparseable, and none of the quote/escape repairs above can fix either:

    * **Terminal degeneration** -- after emitting the final field the model
      loops on closing-delimiter fragments (``}"``, ``} "``, ``2);\n}"`` ...)
      instead of cleanly closing the root object. ``json.loads`` then reports
      "Expecting ',' delimiter" / "Extra data" at the first garbage character,
      even though every field before it is intact.
    * **Truncation** -- the response is cut off mid-object (often mid-string).

    In both cases the maximal valid *prefix* holds the real payload, so this
    keeps the longest prefix that parses once the containers still open at that
    point are closed: it truncates at each decode-error position (discarding any
    dangling partial token or string), re-balances the open containers, and
    retries until it parses -- or returns ``text`` unchanged if nothing can be
    recovered. It is deliberately the last-resort repair: it is mildly lossy
    (discards the trailing garbage), and any recovery that drops a required
    field is still rejected downstream by :func:`_validate_parsed_data`.
    """
    start = text.find('{')
    if start < 0:
        return text
    body = text[start:]
    cut = len(body)
    for _ in range(64):
        candidate = body[:cut].rstrip()
        # A trailing ',' or ':' left by the cut can never end a valid object.
        while candidate and candidate[-1] in ',:':
            candidate = candidate[:-1].rstrip()
        if not candidate:
            break
        closers = _json_open_container_closers(candidate)
        if closers is None:
            # The cut landed inside a string: there is no decode error to drive
            # the next cut while mid-string, so drop that dangling string from
            # its opening quote and retry.
            quote = candidate.rfind('"')
            if quote <= 0:
                break
            cut = quote
            continue
        attempt = candidate + closers
        try:
            parsed = json.loads(attempt, strict=False)
        except json.JSONDecodeError as exc:
            nxt = exc.pos
            if nxt >= cut:
                nxt = cut - 1
            if nxt <= 0:
                break
            cut = nxt
            continue
        # A recovery that collapses to an empty object/array is not a meaningful
        # recovery of a non-empty response (e.g. prose that merely starts with
        # '{'); treat it as failure so the caller falls through to its normal
        # "not valid JSON" error path instead of silently accepting '{}'.
        if isinstance(parsed, (dict, list)) and not parsed:
            break
        return attempt
    return text


_INVALID_ESCAPE_GUIDANCE = (
    "This is an INVALID BACKSLASH ESCAPE inside a string value. In JSON a "
    "backslash may ONLY be followed by one of: \" \\ / b f n r t u. A backslash "
    "before any other character is illegal -- the most common cause is a "
    "Markdown backtick emitted as backslash-backtick. To fix it: never put a "
    "backslash before a backtick (write `asset list` with plain backticks), and "
    "to include a literal backslash in text (such as a Windows path) double it. "
    "Re-emit the SAME content as ONE strict JSON object with valid escaping."
)


def _is_rate_limit_error(exc: Exception) -> bool:
    """True if ``exc`` is Vertex capacity/quota exhaustion, not a content error.

    Rate limiting (HTTP 429 / RESOURCE_EXHAUSTED / "no backend available") is
    NOT something the model can self-correct: it means the serving pool could
    not place the request. Feeding it back as self-correction would waste an
    attempt with no sleep AND grow the request, so callers detect it up front
    and take the (much longer) backoff path instead.
    """
    exc_str = str(exc).lower()
    return (
        "429" in exc_str
        or "resource_exhausted" in exc_str
        or "resource has been exhausted" in exc_str
        or "no backend available" in exc_str
        or "quota" in exc_str
        or "rate limit" in exc_str
    )


def _retry_backoff_seconds(attempt: int, *, is_rate_limit: bool) -> float:
    """Seconds to wait before retrying 0-based ``attempt``.

    Capacity exhaustion needs a real cool-off: an ~8s retry just keeps hammering
    an over-subscribed pool that cannot drain that fast, so back off
    exponentially (cap 60s) with wide jitter. Ordinary transport errors take the
    short path. Both add jitter so a batch of workers that failed in lockstep
    does not retry in lockstep.
    """
    if is_rate_limit:
        return min(15.0 * (2 ** attempt), 60.0) + random.uniform(1.0, 5.0)
    return min(2.0 * (attempt + 1), 8.0) + random.uniform(0.0, 1.0)


def _json_correction_message(exc: Exception) -> str:
    """Build a targeted self-correction instruction for a failed generation.

    A generic "please fix the JSON" nudge does not teach the model what it did
    wrong, so it tends to repeat the exact same mistake and burn every retry.
    When the failure is a JSON syntax error (overwhelmingly an unescaped double
    quote inside a string value -- e.g. a command like ``message send --message
    "hi"``), give a precise instruction the model can act on.
    """
    msg = str(exc)
    if "Invalid \\escape" in msg:
        return (
            "Your previous response was NOT valid JSON. Error details: "
            f"{exc}\n"
            f"{_INVALID_ESCAPE_GUIDANCE}\n"
            "Do NOT wrap the JSON in markdown fences and do NOT add any text "
            "before or after the JSON object."
        )
    is_json_syntax = (
        "JSON Parse Failure" in msg
        or "not valid JSON" in msg
        or "delimiter" in msg
        or "Expecting" in msg
        or "did not contain a JSON object" in msg
    )
    if is_json_syntax:
        return (
            "Your previous response was NOT valid JSON. Error details: "
            f"{exc}\n"
            "There are two common causes -- fix whichever applies:\n"
            "A) An UNESCAPED DOUBLE QUOTE inside a string value. Inside every "
            "command string (e.g. in 'state_effects' or 'solution_commands'), "
            "quote all arguments with SINGLE quotes, never double quotes -- "
            "write --message 'ready' and --content 'value', NOT "
            "--message \"ready\"; in any prose/free-text field escape every "
            "literal double quote as \\\".\n"
            "B) EXTRA CHARACTERS AFTER THE OBJECT. If the content was already "
            "complete, end the response immediately with the single closing "
            "brace that closes the root object -- do NOT repeat braces or quotes "
            "(such as '}\"', '} }', or '}\\n}') and do NOT append any text after "
            "it.\n"
            "Re-emit the SAME content as ONE strict JSON object. Do NOT wrap it "
            "in markdown fences and do NOT add any text before or after the JSON "
            "object."
        )
    # Non-syntax failure (e.g. schema/shape validation): echo the error so the
    # model can correct the offending field.
    return (
        "Your previous response was invalid. Error details: "
        f"{exc}\n"
        "Return ONE corrected, strict JSON object that fixes exactly this "
        "error. Do not include markdown or any text outside the JSON object."
    )


def extract_json_object(raw: str) -> dict[str, Any]:
    """Extract a single JSON object from an LLM response."""

    text = raw.strip()
    match = _JSON_FENCE_RE.match(text)
    if match:
        text = match.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        start = text.find("{")
        if start < 0:
            raise ScenarioLLMError("LLM response did not contain a JSON object")
        # Models sometimes emit a complete JSON object followed by trailing
        # garbage (e.g. a degenerate loop of repeated "}"/"]" delimiters), which
        # makes json.loads raise "Extra data". raw_decode parses only the first
        # complete JSON value starting at `start` and ignores anything after it.
        # strict=False additionally tolerates raw control characters (newlines,
        # tabs) inside string values, a very common failure in generated
        # HTML/rich-text fields.
        try:
            data, _ = json.JSONDecoder(strict=False).raw_decode(text, start)
        except json.JSONDecodeError:
            end = text.rfind("}")
            if end > start:
                slice_ = text[start : end + 1]
            else:
                # No closing brace at all: the response was truncated before the
                # root object ever closed. Keep everything from the first '{' so
                # the truncation-recovery repair below can still complete it.
                slice_ = text[start:]
            data = None
            repaired = ""
            last_exc: Exception | None = exc
            # Progressively more aggressive repairs. The scanner-based passes
            # recover unescaped double quotes inside command strings / prose,
            # which the line-by-line repair alone cannot always fix. Each is a
            # last resort tried only after strict parsing has failed.
            repair_candidates: list[Callable[[], str]] = [
                lambda s=slice_: s,
                lambda s=slice_: _repair_json_string(s),
                lambda s=slice_: _escape_inner_quotes_scanner(s),
                lambda s=slice_: _escape_inner_quotes_scanner(_repair_json_string(s)),
                lambda s=slice_: _escape_invalid_backslashes(s),
                lambda s=slice_: _escape_invalid_backslashes(_repair_json_string(s)),
                lambda s=slice_: _escape_inner_quotes_scanner(
                    _escape_invalid_backslashes(_repair_json_string(s))
                ),
                # Terminal degeneration / truncation: keep the valid prefix and
                # close the still-open containers. Tried last because it is
                # mildly lossy (drops the trailing garbage) and is gated
                # downstream by required-field validation. Run on the full text
                # too, since `slice_` is pre-cut at the last '}', which can drop
                # content when the response was truncated before the root closed.
                lambda t=text: _close_unbalanced_json(t),
                lambda s=slice_: _close_unbalanced_json(s),
                lambda s=slice_: _close_unbalanced_json(_repair_json_string(s)),
            ]
            for _make in repair_candidates:
                try:
                    repaired = _make()
                    data = json.loads(repaired, strict=False)
                    break
                except Exception as repair_exc:
                    last_exc = repair_exc
                    continue
            if data is None:
                raw_head = text[:800]
                raw_tail = text[-800:] if len(text) > 800 else ""
                repaired_head = (repaired or "")[:800]
                debug_msg = (
                    f"LLM response was not valid JSON (length={len(text)}): {exc}.\n"
                    f"Repair error: {last_exc}.\n"
                    f"--- Raw text (head) ---\n{raw_head}\n"
                    f"--- Raw text (tail) ---\n{raw_tail}\n"
                    f"--- Repaired text (head) ---\n{repaired_head}"
                )
                try:
                    with open("/tmp/clasharness_json_error.txt", "w", encoding="utf-8") as f:
                        f.write(f"=== JSON REPAIR FAILURE ===\n{debug_msg}\n")
                    with open("/tmp/clasharness_json_error_raw.txt", "w", encoding="utf-8") as f:
                        f.write(text)
                except Exception:
                    pass
                raise ScenarioLLMError(debug_msg) from last_exc
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
        data = data[0]
    if not isinstance(data, dict):
        raise ScenarioLLMError("LLM response JSON must be an object")
    return data


_RAW_TEXT_LAYER_FIELDS = {
    "layer1_workspace_file": "content",
    "layer2_session_message_text": "text",
    "layer3_eval_round": "question",
    "layer4_update_workspace_content": "content",
    "layer4_update_message_text": "text",
}


def _parse_layer_content(layer: str, content: str) -> dict[str, Any]:
    """Parse provider text for a layer, with raw-text fallback for fill slots."""

    try:
        data = extract_json_object(content)
        _validate_parsed_data(layer, data)
        return data
    except ScenarioLLMError:
        field = _RAW_TEXT_LAYER_FIELDS.get(layer)
        text = _strip_text_fence(content)
        if field and text:
            return {field: text}
        raise


def _strip_text_fence(raw: str) -> str:
    text = raw.strip()
    match = _JSON_FENCE_RE.match(text)
    if match:
        text = match.group(1).strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


@dataclass
class OpenAICompatibleScenarioLLMClient:
    """Stdlib OpenAI-compatible chat-completions client.

    This client is only intended for offline data generation. Evaluation never
    calls it, so generated scenario packages remain frozen and reproducible.
    """

    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    # ``None`` omits the temperature entirely from the request body, required for
    # gpt-5-class / bedrock-gpt models that 400 on an explicit temperature.
    temperature: float | None = 0.2
    timeout_seconds: float = 60.0
    retries: int = 2
    request_delay_seconds: float = 0.0
    max_output_tokens: int = 16384
    api: str = "auto"
    transport: str = "urllib"
    json_mode: str = "none"
    user_agent: str = "curl/8.5.0"
    verbose: bool = False
    logger: Callable[[str], None] | None = None
    raw_response_dir: Path | None = None

    def audit_image_fidelity(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        expected_token: str,
        context: str = "",
    ) -> dict[str, Any]:
        """Vision check (OpenAI Responses API): is ``expected_token`` legible?

        Uses the Responses API multimodal shape (``input_text`` + ``input_image``
        with a ``data:`` URL), which is what GPT on Bedrock (``bedrock-gpt``,
        Responses-only) and OpenAI's own vision models accept. Returns the shared
        ``{legible, seen_text, reason, skipped}`` shape with ``skipped=False``.

        Chat-only proxies (``api="chat"``) do not serve ``/responses`` and use a
        different multimodal schema, so for them the audit skips (a benign pass)
        rather than risk a false downgrade. Any transport/parse error propagates
        to the caller (``_audit_fidelity_safe``), which also treats it as a
        benign pass -- so a flaky audit never downgrades a good decisive asset.
        """
        if self.api == "chat":
            return {
                "legible": True,
                "seen_text": "",
                "reason": (
                    "openai-compatible client (chat API): vision fidelity audit"
                    " skipped"
                ),
                "skipped": True,
            }
        api_key = os.environ.get(self.api_key_env, "")
        if not api_key:
            raise ScenarioLLMError(
                f"Missing API key environment variable: {self.api_key_env}"
            )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }
        data_url = (
            f"data:{mime_type or 'image/png'};base64,"
            + base64.b64encode(image_bytes).decode("ascii")
        )
        request_body: dict[str, Any] = {
            "model": self.model,
            "instructions": _VISION_FIDELITY_SYSTEM,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": _vision_fidelity_ask(expected_token, context),
                        },
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ],
            "max_output_tokens": 1024,
        }
        # gpt-5 / bedrock-gpt reject an explicit temperature; include one only
        # when configured (mirrors ``_request_json_layer``'s responses branch).
        if self.temperature is not None:
            request_body["temperature"] = self.temperature
        url = self.base_url.rstrip("/") + "/responses"

        if self.transport == "curl":
            self._log(f"POST {url} via curl (vision_fidelity)")
            response = self._request_raw_with_curl(
                url, request_body, headers, api="responses"
            )
        elif self.transport == "urllib":
            req = urllib.request.Request(
                url,
                data=json.dumps(request_body).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            self._log(f"POST {url} via urllib (vision_fidelity)")
            with urllib.request.urlopen(
                req, timeout=self.timeout_seconds
            ) as resp:
                raw = resp.read().decode("utf-8")
            self._write_raw_response("responses_vision", raw)
            response = json.loads(raw)
        else:
            raise ScenarioLLMError(f"Unsupported transport: {self.transport}")

        return _parse_vision_fidelity(_extract_responses_text(response))

    def generate_layer(self, layer: str, payload: dict[str, Any]) -> dict[str, Any]:
        api_key = os.environ.get(self.api_key_env, "")
        if not api_key:
            raise ScenarioLLMError(
                f"Missing API key environment variable: {self.api_key_env}"
            )

        system_prompt = (
            "You generate synthetic multi-agent scenario data. Return one strict JSON "
            "object only. Do not include markdown."
        )
        user_prompt = json.dumps(
            {
                "layer": layer,
                "contract": _layer_contract(layer),
                "payload": payload,
            },
            ensure_ascii=False,
        )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            if self.request_delay_seconds > 0:
                self._log(f"waiting {self.request_delay_seconds:.1f}s before layer={layer} request")
                time.sleep(self.request_delay_seconds)
            self._log(
                f"request layer={layer} attempt={attempt + 1}/{self.retries + 1} "
                f"api={self.api} transport={self.transport} model={self.model}"
            )
            try:
                data = self._request_json_layer(system_prompt, user_prompt, headers, layer=layer)
                self._log(f"response layer={layer} parsed keys={sorted(data.keys())[:8]}")
                return data
            except (KeyError, IndexError, json.JSONDecodeError) as exc:
                last_error = ScenarioLLMError(f"Malformed provider response: {exc}")
            except ScenarioLLMError as exc:
                last_error = exc
                if isinstance(exc, NonRetryableScenarioLLMError):
                    break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if self.api == "auto" and exc.code == 400 and "unsupported" in body.lower():
                    try:
                        return self._request_json_layer(
                            system_prompt,
                            user_prompt,
                            headers,
                            layer=layer,
                            force_api="responses",
                        )
                    except Exception as fallback_exc:  # pragma: no cover - defensive path
                        last_error = fallback_exc
                        continue
                last_error = ScenarioLLMError(f"Provider HTTP {exc.code}: {body[:500]}")
                if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                    break
            except urllib.error.URLError as exc:
                last_error = ScenarioLLMError(f"Provider request failed: {exc}")
            if attempt < self.retries:
                delay = 0.5 * (2**attempt)
                self._log(f"retrying layer={layer} after {delay:.1f}s: {last_error}")
                time.sleep(delay)
        raise ScenarioLLMError(str(last_error or "Provider request failed"))

    def _request_json_layer(
        self,
        system_prompt: str,
        user_prompt: str,
        headers: dict[str, str],
        *,
        layer: str,
        force_api: str | None = None,
    ) -> dict[str, Any]:
        api = force_api or self.api
        if api == "auto":
            api = "chat"
        base_url = self.base_url.rstrip("/")
        if api == "responses":
            request_body = {
                "model": self.model,
                "instructions": system_prompt,
                "input": user_prompt,
                "max_output_tokens": self.max_output_tokens,
            }
            if self.temperature is not None:
                request_body["temperature"] = self.temperature
            url = base_url + "/responses"
        elif api == "chat":
            request_body = {
                "model": self.model,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            if self.json_mode == "response_format":
                request_body["response_format"] = {"type": "json_object"}
            elif self.json_mode != "none":
                raise ScenarioLLMError(f"Unsupported json_mode: {self.json_mode}")
            request_body[_chat_token_limit_field(self.model)] = self.max_output_tokens
            url = base_url + "/chat/completions"
        else:
            raise ScenarioLLMError(f"Unsupported OpenAI-compatible API mode: {api}")

        if self.transport == "curl":
            self._log(f"POST {url} via curl")
            return self._request_with_curl(url, request_body, headers, api=api, layer=layer)
        if self.transport != "urllib":
            raise ScenarioLLMError(f"Unsupported transport: {self.transport}")

        req = urllib.request.Request(
            url,
            data=json.dumps(request_body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        self._log(f"POST {url} via urllib")
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            raw = resp.read().decode("utf-8")
        self._log(f"received {len(raw)} bytes from provider")
        self._write_raw_response(api, raw)
        response = json.loads(raw)
        if api == "responses":
            content = _extract_responses_text(response)
        else:
            content = response["choices"][0]["message"]["content"]
        return _parse_layer_content(layer, content)

    def _request_with_curl(
        self,
        url: str,
        request_body: dict[str, Any],
        headers: dict[str, str],
        *,
        api: str,
        layer: str,
    ) -> dict[str, Any]:
        response = self._request_raw_with_curl(url, request_body, headers, api=api)
        if api == "responses":
            content = _extract_responses_text(response)
        else:
            content = response["choices"][0]["message"]["content"]
        return _parse_layer_content(layer, content)

    def _request_raw_with_curl(
        self,
        url: str,
        request_body: dict[str, Any],
        headers: dict[str, str],
        *,
        api: str,
    ) -> dict[str, Any]:
        """POST ``request_body`` via curl and return the decoded JSON response.

        Shared by the layer request path (which then extracts layer content) and
        the vision-fidelity audit (which extracts the responses text), so both
        honor the same HTTP status / non-retryable-error handling.
        """
        cmd = [
            "curl",
            "-sS",
            "--fail-with-body",
            "-w",
            "\n%{http_code}",
            "--max-time",
            str(int(self.timeout_seconds)),
        ]
        for key, value in headers.items():
            cmd.extend(["-H", f"{key}: {value}"])
        body = json.dumps(request_body, ensure_ascii=False)
        cmd.extend(["--data-binary", "@-", url])
        proc = subprocess.run(cmd, input=body, capture_output=True, text=True)
        stdout, http_code = _split_curl_body_and_status(proc.stdout)
        if proc.returncode != 0:
            detail = (proc.stderr or stdout or "").strip()
            message = (
                f"curl transport failed ({proc.returncode}, HTTP {http_code or 'unknown'}): "
                f"{detail[:500]}"
            )
            if http_code in {"400", "401", "403", "404"}:
                raise NonRetryableScenarioLLMError(message)
            raise ScenarioLLMError(message)
        if http_code and http_code.startswith(("4", "5")):
            message = f"Provider HTTP {http_code}: {stdout[:500]}"
            if http_code in {"400", "401", "403", "404"}:
                raise NonRetryableScenarioLLMError(message)
            raise ScenarioLLMError(message)
        self._log(f"curl received {len(stdout)} bytes from provider (HTTP {http_code or 'unknown'})")
        self._write_raw_response(api, stdout)
        return json.loads(stdout)

    def _log(self, message: str) -> None:
        if not self.verbose:
            return
        if self.logger is not None:
            self.logger(message)
        else:
            print(message, flush=True)

    def _write_raw_response(self, api: str, raw: str) -> None:
        if self.raw_response_dir is None:
            return
        self.raw_response_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time() * 1000)
        path = self.raw_response_dir / f"{stamp}_{api}.json"
        path.write_text(raw, encoding="utf-8")
        self._log(f"saved raw provider response: {path}")


def _format_decode_error_snippet(exc: json.JSONDecodeError) -> str:
    if not hasattr(exc, "doc") or not exc.doc:
        return ""
    lines = exc.doc.splitlines()
    err_line_idx = exc.lineno - 1
    snippet_lines = []
    
    # Show 3 lines of context before and after
    start_idx = max(0, err_line_idx - 3)
    end_idx = min(len(lines), err_line_idx + 4)
    
    for idx in range(start_idx, end_idx):
        line_num = idx + 1
        if idx < len(lines):
            prefix = "--> " if idx == err_line_idx else "    "
            snippet_lines.append(f"{prefix}Line {line_num:3d}: {lines[idx]}")
        if idx == err_line_idx:
            # prefix + "Line " + 3-digit-line-num + ": " is 4 + 5 + 3 + 2 = 14 chars.
            # So we offset the column index by 13 spaces to align caret with colno (colno is 1-indexed)
            caret_prefix = " " * (13 + exc.colno)
            snippet_lines.append(f"{caret_prefix}^--- JSONDecodeError: {exc.msg}")
            
    return "\n".join(snippet_lines)


@dataclass
class GeminiScenarioLLMClient:
    """Stdlib client for Google's native Gemini API (generateContent)."""

    model: str
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    api_key_env: str = "GEMINI_API_KEY"
    temperature: float = 0.2
    timeout_seconds: float = 180.0
    retries: int = 5
    request_delay_seconds: float = 0.0
    verbose: bool = False
    logger: Callable[[str], None] | None = None
    raw_response_dir: Path | None = None
    max_output_tokens: int = 16384

    def __post_init__(self):
        self._log(
            f"Gemini Client Initialized: model={self.model}, "
            f"max_output_tokens={self.max_output_tokens}, "
            f"temperature={self.temperature}, "
            f"timeout={self.timeout_seconds}s, "
            f"retries={self.retries}"
        )

    def generate_layer(self, layer: str, payload: dict[str, Any]) -> dict[str, Any]:
        api_key = os.environ.get(self.api_key_env, "")
        if not api_key:
            raise ScenarioLLMError(f"Missing API key environment variable: {self.api_key_env}")

        system_instruction = "You generate synthetic multi-agent scenario data. Return one strict JSON object only. Do not include markdown."
        user_prompt = json.dumps({
            "layer": layer,
            "contract": _layer_contract(layer),
            "payload": payload,
        }, ensure_ascii=False)

        # Initialize multi-turn contents with the first prompt
        contents = [{"role": "user", "parts": [{"text": user_prompt}]}]
        self._log(f"Gemini API request ({layer}) prompt size: {len(user_prompt)} characters")
        url = f"{self.base_url}/models/{self.model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            if self.request_delay_seconds > 0:
                self._log(f"waiting {self.request_delay_seconds:.1f}s before layer={layer} request")
                time.sleep(self.request_delay_seconds)

            self._log(f"Gemini API request ({layer}): model={self.model} attempt={attempt + 1}")

            request_body = {
                "systemInstruction": {"parts": [{"text": system_instruction}]},
                "contents": contents,
                "generationConfig": {
                    "temperature": self.temperature,
                    "responseMimeType": "application/json",
                    "maxOutputTokens": self.max_output_tokens
                }
            }

            raw_response = None
            try:
                raw_response = self._post_json(url, request_body, headers)
                if self.raw_response_dir:
                    self.raw_response_dir.mkdir(parents=True, exist_ok=True)
                    (self.raw_response_dir / f"gemini_{layer}_{int(time.time() * 1000)}.json").write_text(
                        json.dumps(raw_response, indent=2, ensure_ascii=False), encoding="utf-8"
                    )

                return self._parse_gemini_response(layer, raw_response)
            except Exception as exc:
                last_error = exc
                exc_str = str(exc).lower()
                is_503 = "503" in exc_str or "overload" in exc_str or "unavailable" in exc_str
                if not is_503 or attempt == self.retries:
                    self._log(f"Gemini API error ({layer}): {exc}")

                # If there are retries left and we have a text response, append it for self-correction
                if attempt < self.retries and raw_response:
                    model_text = ""
                    try:
                        model_text = raw_response["candidates"][0]["content"]["parts"][0]["text"]
                    except Exception:
                        pass

                    if model_text:
                        self._log(f"Self-correction active: feeding error back to Gemini...")
                        
                        # Extract the original underlying error
                        original_exc = exc
                        if hasattr(exc, "__cause__") and exc.__cause__:
                            original_exc = exc.__cause__
                        elif hasattr(exc, "__context__") and exc.__context__:
                            original_exc = exc.__context__

                        # Build the compiler-style descriptive error prompt
                        if isinstance(original_exc, json.JSONDecodeError):
                            snippet = _format_decode_error_snippet(original_exc)
                            if "Invalid \\escape" in (original_exc.msg or ""):
                                fix_hint = _INVALID_ESCAPE_GUIDANCE
                            else:
                                fix_hint = (
                                    "This is almost always an UNESCAPED DOUBLE QUOTE inside a string value. To fix it: "
                                    "inside every command string (e.g. in 'state_effects'/'solution_commands') quote all "
                                    "arguments with SINGLE quotes, never double quotes (write --message 'ready', NOT "
                                    "--message \"ready\"); and escape every literal double quote in prose fields as \\\"."
                                )
                            error_message = (
                                "Your previous response is invalid JSON. The parser failed with the following syntax error:\n"
                                f"JSONDecodeError: {original_exc.msg} at line {original_exc.lineno}, column {original_exc.colno}.\n\n"
                                "Here is the context around the syntax error:\n"
                                "```\n"
                                f"{snippet}\n"
                                "```\n"
                                f"{fix_hint} "
                                "Analyze the caret pointer, correct the offending characters on that line, and return "
                                "the fully valid JSON object matching the contract schema."
                            )
                        else:
                            error_message = (
                                "Your previous response failed validation checks with the following error:\n"
                                f"[VALIDATION FAILURE]: {exc}\n\n"
                                "Please modify your output to strictly conform to these validation requirements "
                                "and return the corrected JSON object matching the contract schema."
                            )

                        contents.append({"role": "model", "parts": [{"text": model_text}]})
                        contents.append({
                            "role": "user",
                            "parts": [{"text": error_message}]
                        })

                if attempt == self.retries:
                    break

                # Exponential backoff (max 16s) + random jitter (0.5 to 2s) to prevent thundering herd
                delay = min(2 ** attempt, 16) + random.uniform(0.5, 2.0)
                if is_503:
                    delay = max(delay * 2, 12.0) + random.uniform(2.0, 5.0)
                else:
                    self._log(f"Retrying layer={layer} after {delay:.1f}s (due to error)")
                time.sleep(delay)

        raise ScenarioLLMError(f"Gemini API failed after {self.retries + 1} attempts: {last_error}")

    def _post_json(self, url: str, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise ScenarioLLMError(f"Gemini API HTTP {exc.code}: {detail}") from exc

    def _parse_gemini_response(self, layer: str, raw: dict[str, Any]) -> dict[str, Any]:
        try:
            candidates = raw.get("candidates", [])
            if not candidates:
                raise ScenarioLLMError("No candidates in Gemini response")
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts:
                raise ScenarioLLMError("No parts in Gemini response content")
            text = parts[0].get("text", "")
            return _parse_layer_content(layer, text)
        except Exception as exc:
            finish_reason = "unknown"
            try:
                finish_reason = raw.get("candidates", [{}])[0].get("finishReason", "unknown")
            except Exception:
                pass
            raise ScenarioLLMError(f"Failed to parse Gemini response (finishReason={finish_reason}): {exc}") from exc

    def _log(self, msg: str) -> None:
        if self.verbose and self.logger:
            self.logger(msg)

    def audit_image_fidelity(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        expected_token: str,
        context: str = "",
    ) -> dict[str, Any]:
        """Vision check (native Gemini REST, inlineData): is ``expected_token`` legible?

        Returns the shared ``{legible, seen_text, reason, skipped}`` shape with
        ``skipped=False``. Transport/parse errors propagate to the caller, which
        treats them as a benign pass.
        """
        api_key = os.environ.get(self.api_key_env, "")
        if not api_key:
            raise ScenarioLLMError(
                f"Missing API key environment variable: {self.api_key_env}"
            )
        url = f"{self.base_url}/models/{self.model}:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
        body = {
            "systemInstruction": {"parts": [{"text": _VISION_FIDELITY_SYSTEM}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": mime_type or "image/png",
                                "data": base64.b64encode(image_bytes).decode(
                                    "ascii"
                                ),
                            }
                        },
                        {"text": _vision_fidelity_ask(expected_token, context)},
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
                "maxOutputTokens": 1024,
            },
        }
        raw = self._post_json(url, body, headers)
        candidates = raw.get("candidates", [])
        if not candidates:
            raise ScenarioLLMError(
                "No candidates in Gemini vision audit response"
            )
        parts = candidates[0].get("content", {}).get("parts", [])
        text = parts[0].get("text", "") if parts else ""
        if not text:
            raise ScenarioLLMError("Empty text in Gemini vision audit response")
        return _parse_vision_fidelity(text)


@dataclass
class VertexScenarioLLMClient:
    """Client for Vertex AI Gemini API using the google-genai SDK."""

    model: str
    project: str | None = None
    location: str | None = None
    temperature: float = 0.2
    timeout_seconds: float = 180.0
    retries: int = 5
    request_delay_seconds: float = 0.0
    verbose: bool = False
    logger: Callable[[str], None] | None = None
    raw_response_dir: Path | None = None
    max_output_tokens: int = 16384

    def __post_init__(self):
        # Resolve project and location if None
        self._project = self.project or os.environ.get("VERTEX_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GEMINI_PROJECT")
        self._location = self.location or os.environ.get("VERTEX_LOCATION") or os.environ.get("GOOGLE_CLOUD_LOCATION") or os.environ.get("GEMINI_LOCATION") or "us-central1"

        # The google-genai Client is NOT safe for concurrent use across threads:
        # sharing a single client from a ThreadPoolExecutor (the scenario
        # generator runs versions in parallel and rounds via num_generation_workers)
        # raises "Context has already been used to create a Connection, it cannot
        # be mutated again". Hold one client per thread so each worker gets an
        # isolated connection context; clients are built lazily on first use in
        # each thread.
        self._thread_local = threading.local()
        # Build eagerly for the constructing thread so config errors surface now
        # and the original init log line is preserved.
        self._get_client()
        self._log(
            f"Vertex Client Initialized: model={self.model}, "
            f"project={self._project}, location={self._location}, "
            f"max_output_tokens={self.max_output_tokens}, "
            f"temperature={self.temperature}, "
            f"retries={self.retries}"
        )

    def _get_client(self):
        """Return a google-genai Client unique to the calling thread.

        google-genai's Client cannot be shared across threads (its connection
        Context is single-use), so each worker thread lazily builds and caches
        its own client here rather than sharing ``self``'s.
        """
        client = getattr(self._thread_local, "client", None)
        if client is None:
            from google import genai
            client = genai.Client(
                vertexai=True,
                project=self._project,
                location=self._location,
            )
            self._thread_local.client = client
        return client

    def _reset_client(self):
        """Discard this thread's cached client so the next call rebuilds it.

        A cached google-genai Client whose httpx connection pool has already
        been used can raise "Context has already been used to create a
        Connection, it cannot be mutated again" when it is reused under a
        concurrent burst (e.g. all workers hitting the first layer of a new
        batch of scenarios at once). A freshly built client's first call always
        succeeds, so on such a transport failure we drop the bad client (closing
        it best-effort to avoid leaking its connection pool) and let
        ``_get_client`` build a fresh one on the next attempt.
        """
        client = getattr(self._thread_local, "client", None)
        if client is not None:
            try:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass
            self._thread_local.client = None

    def _log(self, msg: str):
        if self.logger:
            self.logger(msg)

    def generate_layer(self, layer: str, payload: dict[str, Any]) -> dict[str, Any]:
        from google.genai import types

        system_instruction = "You generate synthetic multi-agent scenario data. Return one strict JSON object only. Do not include markdown."
        user_prompt = json.dumps({
            "layer": layer,
            "contract": _layer_contract(layer),
            "payload": payload,
        }, ensure_ascii=False)

        # Multi-turn contents list for self-correction retries
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])]

        self._log(f"Vertex API request ({layer}) prompt size: {len(user_prompt)} characters")

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            if self.request_delay_seconds > 0:
                self._log(f"waiting {self.request_delay_seconds:.1f}s before layer={layer} request")
                time.sleep(self.request_delay_seconds)

            self._log(f"Vertex API request ({layer}): model={self.model} attempt={attempt + 1}")

            try:
                response = self._get_client().models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens,
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                    ),
                )
                
                text_response = response.text
                if not text_response:
                    raise ScenarioLLMError("Empty response text from Vertex AI")

                # Parse JSON
                try:
                    data = extract_json_object(text_response)
                    _validate_parsed_data(layer, data)
                except Exception as parse_exc:
                    raise ScenarioLLMError(f"JSON Parse Failure: {parse_exc}. Response: {text_response}") from parse_exc

                if self.raw_response_dir:
                    self.raw_response_dir.mkdir(parents=True, exist_ok=True)
                    (self.raw_response_dir / f"vertex_{layer}_{int(time.time() * 1000)}.json").write_text(
                        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
                    )

                return data

            except Exception as exc:
                last_error = exc
                self._log(f"Vertex API error ({layer}) attempt={attempt + 1}: {exc}")

                if attempt < self.retries:
                    # Capacity exhaustion is not self-correctable; see
                    # ``_is_rate_limit_error``.
                    is_rate_limit = _is_rate_limit_error(exc)

                    model_text = ""
                    if not is_rate_limit:
                        try:
                            model_text = response.text
                        except Exception:
                            pass

                    if model_text:
                        self._log("Self-correction active: feeding error back to model...")
                        contents.append(types.Content(role="model", parts=[types.Part.from_text(text=model_text)]))
                        error_msg = _json_correction_message(exc)
                        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=error_msg)]))
                    else:
                        # The API call itself failed before returning any text
                        # (transport/connection error such as the httpx "Context
                        # has already been used to create a Connection" raised
                        # when a cached client is reused under a concurrent
                        # burst). Rebuild this thread's client so the retry uses
                        # a fresh connection pool, and add jitter so a batch of
                        # workers that failed in lockstep does not retry in
                        # lockstep.
                        self._reset_client()
                        backoff = _retry_backoff_seconds(
                            attempt, is_rate_limit=is_rate_limit
                        )
                        if is_rate_limit:
                            self._log(
                                f"Rate limit / capacity exhausted ({layer}); "
                                f"backing off {backoff:.1f}s before retry "
                                f"(attempt {attempt + 1}/{self.retries + 1})"
                            )
                        time.sleep(backoff)

        assert last_error is not None
        raise last_error

    def audit_image_fidelity(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        expected_token: str,
        context: str = "",
    ) -> dict[str, Any]:
        """Vision check (Vertex google-genai): is ``expected_token`` legible?

        Returns the shared ``{legible, seen_text, reason, skipped}`` shape with
        ``skipped=False`` (Vertex is vision-capable).

        Retried, paced and logged on the same terms as ``generate_layer``:
        transient transport errors rebuild the thread's client, capacity
        exhaustion takes the long backoff, and every attempt is logged. This
        matters because the caller (``_audit_fidelity_safe``) treats *any* audit
        error as a benign pass -- so an unretried, unlogged audit degrades
        silently. That was tolerable when only ~20 assets were audited, but at
        full decisive coverage (~480 assets per run) a bad spell on the serving
        pool would quietly erode audit coverage leaving no trace in the log.

        Self-correction is deliberately NOT applied: a fidelity verdict is a
        single-shot judgement, so replaying a malformed reply only grows the
        request. Errors surviving the retry budget still propagate to the
        caller, so a genuinely flaky audit never wrongly downgrades a good
        decisive asset.
        """
        from google.genai import types

        parts = [
            types.Part.from_bytes(
                data=image_bytes, mime_type=mime_type or "image/png"
            ),
            types.Part.from_text(
                text=_vision_fidelity_ask(expected_token, context)
            ),
        ]
        contents = [types.Content(role="user", parts=parts)]

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            if self.request_delay_seconds > 0:
                self._log(
                    f"waiting {self.request_delay_seconds:.1f}s before"
                    " layer=vision_fidelity request"
                )
                time.sleep(self.request_delay_seconds)

            self._log(
                "Vertex API request (vision_fidelity):"
                f" model={self.model} token={expected_token!r}"
                f" attempt={attempt + 1}"
            )
            try:
                response = self._get_client().models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=1024,
                        system_instruction=_VISION_FIDELITY_SYSTEM,
                        response_mime_type="application/json",
                    ),
                )
                text_response = response.text
                if not text_response:
                    raise ScenarioLLMError(
                        "Empty response text from Vertex AI vision audit"
                    )
                return _parse_vision_fidelity(text_response)

            except Exception as exc:
                last_error = exc
                self._log(
                    "Vertex API error (vision_fidelity)"
                    f" attempt={attempt + 1}: {exc}"
                )
                if attempt < self.retries:
                    is_rate_limit = _is_rate_limit_error(exc)
                    # No self-correction turn to fall back on, so always drop a
                    # possibly-poisoned client and retry on a fresh pool.
                    self._reset_client()
                    backoff = _retry_backoff_seconds(
                        attempt, is_rate_limit=is_rate_limit
                    )
                    if is_rate_limit:
                        self._log(
                            "Rate limit / capacity exhausted"
                            f" (vision_fidelity); backing off {backoff:.1f}s"
                            f" before retry (attempt {attempt + 1}/"
                            f"{self.retries + 1})"
                        )
                    time.sleep(backoff)

        assert last_error is not None
        raise last_error


# Backwards-compatible alias. The implementation was hoisted to
# ``clasharness.evaluation.llm_retry`` so the judge shares ONE copy: keeping a
# private mirror here is what let llm_judge keep crashing on thinking blocks
# after this module had already been fixed.
_anthropic_text_from_response = anthropic_text_from_response


@dataclass
class AnthropicScenarioLLMClient:
    """Client for Claude, for offline generation, over either Anthropic backend.

    Mirrors :class:`VertexScenarioLLMClient` (self-correcting JSON retries,
    per-thread client, capacity backoff, real vision fidelity audit) but routes
    ``messages.create`` through the ``anthropic`` SDK instead of the google-genai
    Gemini SDK. Two backends, selected by whether an API key is available:

    * **Anthropic's official API** -- what a public reproduction should use. Set
      ``api_key_env`` to the name of an environment variable holding the key
      (``ANTHROPIC_API_KEY``). ``model`` is then a public id such as
      ``claude-sonnet-5``, and ``project``/``location`` are ignored.
    * **Claude on Vertex AI** (legacy; how the released corpus was generated).
      Leave ``api_key_env`` unset, or its variable empty, and authenticate via
      Application Default Credentials. ``model`` must then be a Vertex publisher
      id such as ``claude-sonnet-5@20250514``, and ``location`` must serve Claude
      (e.g. ``us-east5``).

    Everything above the transport -- prompts, retries, JSON self-correction, the
    vision fidelity audit -- is backend independent, so the two differ only in
    which client object ``_get_client`` builds. Only used for offline data
    generation; evaluation never calls it so generated packages stay frozen and
    reproducible.
    """

    model: str
    project: str | None = None
    location: str | None = None
    # Name of the env var holding an official-API key. When set AND non-empty at
    # construction time it selects anthropic.Anthropic over AnthropicVertex; see
    # _get_client. Resolved once in __post_init__ so every worker thread agrees
    # on the backend even if the environment is mutated mid-run.
    api_key_env: str | None = None
    temperature: float = 0.2
    # An explicit timeout is threaded into every messages.create() call (see
    # _create_kwargs) both to honor it AND to disable the anthropic SDK's
    # "streaming required for operations >10 minutes" guard, which otherwise
    # refuses a large-``max_output_tokens`` non-streaming request. 600s matches
    # the SDK's own default read timeout and is ample for offline generation;
    # the Gemini/OpenAI clients need no such handling (no equivalent guard).
    timeout_seconds: float = 600.0
    retries: int = 5
    request_delay_seconds: float = 0.0
    verbose: bool = False
    logger: Callable[[str], None] | None = None
    raw_response_dir: Path | None = None
    max_output_tokens: int = 16384

    def __post_init__(self):
        # Pick the backend ONCE, here, rather than per thread in _get_client:
        # otherwise a run whose environment changes mid-flight (the generator
        # exports keys for the judge while workers are already running) could end
        # up with some threads on the official API and others on Vertex, silently
        # mixing two model endpoints inside one corpus.
        self._api_key = (
            os.environ.get(self.api_key_env, "").strip()
            if self.api_key_env
            else ""
        )
        self._project = (
            self.project
            or os.environ.get("VERTEX_PROJECT")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GEMINI_PROJECT")
        )
        # Claude on Vertex is NOT served in Gemini's default 'us-central1'; the
        # documented Claude region is 'us-east5', so default to it rather than a
        # Gemini region that would 404.
        self._location = (
            self.location
            or os.environ.get("VERTEX_LOCATION")
            or os.environ.get("GOOGLE_CLOUD_LOCATION")
            or "us-east5"
        )
        # The anthropic SDK client wraps an httpx connection pool that is not
        # safe to share across the generator's worker threads, so -- like the
        # google-genai client -- hold one AnthropicVertex per thread, built
        # lazily on first use in each thread.
        self._thread_local = threading.local()
        # Newer reasoning-class Claude models (Sonnet 5, Opus 4.7/4.8, ...) 400 on
        # ANY explicit temperature. Omit it from the very first request for models
        # KNOWN (statically, or learned in an earlier call this process) to reject
        # it; otherwise start by sending it and learn reactively on the first 400
        # below. The learned set lives in clasharness.evaluation.llm_retry and is
        # shared with the judge, so a rejection discovered by either is honored by
        # both without re-paying a wasted 400.
        self._omit_temperature = anthropic_rejects_sampling_params(self.model)
        # Build eagerly for the constructing thread so config/import errors
        # surface now and the init log line is preserved.
        self._get_client()
        # Name the backend in the log: 'project=None location=us-east5' on an
        # official-API run would otherwise read like a misconfigured Vertex run.
        where = (
            f"api_key_env={self.api_key_env}"
            if self._api_key
            else f"project={self._project}, location={self._location}"
        )
        self._log(
            f"{self._backend_label()} Client Initialized: model={self.model}, "
            f"{where}, "
            f"max_output_tokens={self.max_output_tokens}, "
            f"temperature={self.temperature}, "
            f"timeout={self.timeout_seconds}s, "
            f"retries={self.retries}"
        )

    def _backend_label(self) -> str:
        """Human-readable name of the selected backend, for log lines."""
        return "Anthropic" if self._api_key else "AnthropicVertex"

    def _get_client(self):
        """Return an anthropic SDK client unique to the calling thread.

        ``anthropic.Anthropic`` (official API) when a key was resolved in
        ``__post_init__``, otherwise ``anthropic.AnthropicVertex`` (ADC). Both
        expose the same ``messages.create`` surface, so no caller cares which.
        """
        client = getattr(self._thread_local, "client", None)
        if client is None:
            if self._api_key:
                try:
                    from anthropic import Anthropic
                except ImportError as exc:  # pragma: no cover - dependency guard
                    raise ScenarioLLMError(
                        "anthropic not installed. Run: pip install anthropic"
                    ) from exc
                client = Anthropic(api_key=self._api_key)
            else:
                try:
                    from anthropic import AnthropicVertex
                except ImportError as exc:  # pragma: no cover - dependency guard
                    raise ScenarioLLMError(
                        "anthropic[vertex] not installed. Run: pip install"
                        " 'anthropic[vertex]'"
                    ) from exc
                client = AnthropicVertex(
                    project_id=self._project, region=self._location
                )
            self._thread_local.client = client
        return client

    def _reset_client(self):
        """Discard this thread's cached client so the next call rebuilds it."""
        client = getattr(self._thread_local, "client", None)
        if client is not None:
            try:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass
            self._thread_local.client = None

    def _log(self, msg: str):
        if self.logger:
            self.logger(msg)

    def _create_kwargs(
        self, *, temperature: float | None, **rest: Any
    ) -> dict[str, Any]:
        """Assemble ``messages.create`` kwargs, honoring the temperature opt-out.

        ``temperature`` is included only when it is not ``None`` and the model
        has not already rejected an explicit value (``self._omit_temperature``).
        """
        kwargs: dict[str, Any] = dict(model=self.model, **rest)
        if temperature is not None and not self._omit_temperature:
            kwargs.update(anthropic_temperature_kwargs(temperature))
        # Thread an explicit per-request timeout, mirroring the eval agent
        # (train_eval/agents.py). This honors ``timeout_seconds`` AND disables the
        # anthropic SDK's non-streaming guard, which otherwise raises "Streaming
        # is required for operations that may take longer than 10 minutes" once
        # ``max_tokens`` pushes the SDK's estimate (3600*max_tokens/128000) past
        # 600s -- i.e. any max_tokens > ~21k, which --max_output_tokens=65536
        # trivially exceeds. Gemini's SDK has no such guard, which is why only
        # the Claude path hit it.
        kwargs.setdefault("timeout", self.timeout_seconds)
        return kwargs

    def generate_layer(self, layer: str, payload: dict[str, Any]) -> dict[str, Any]:
        system_instruction = "You generate synthetic multi-agent scenario data. Return one strict JSON object only. Do not include markdown."
        user_prompt = json.dumps({
            "layer": layer,
            "contract": _layer_contract(layer),
            "payload": payload,
        }, ensure_ascii=False)

        # Multi-turn messages list for self-correction retries.
        contents: list[dict[str, Any]] = [
            {"role": "user", "content": user_prompt}
        ]

        self._log(
            f"{self._backend_label()} API request ({layer}) prompt size:"
            f" {len(user_prompt)} characters"
        )

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            if self.request_delay_seconds > 0:
                self._log(f"waiting {self.request_delay_seconds:.1f}s before layer={layer} request")
                time.sleep(self.request_delay_seconds)

            self._log(
                f"{self._backend_label()} API request ({layer}): model={self.model}"
                f" attempt={attempt + 1}"
            )

            text_response = ""
            try:
                kwargs = self._create_kwargs(
                    max_tokens=self.max_output_tokens,
                    system=system_instruction,
                    messages=contents,
                    temperature=self.temperature,
                )
                try:
                    response = self._get_client().messages.create(**kwargs)
                except Exception as create_exc:
                    # A model that rejects an explicit temperature: drop it for
                    # the rest of the run and retry immediately (no wasted outer
                    # attempt), matching the eval path's graceful degradation.
                    # This fires when EITHER this request is the first to learn
                    # the rejection (detector matches the 400) OR a concurrent
                    # worker already learned it and flipped ``_omit_temperature``
                    # after we built our (still-temperature-carrying) kwargs -- a
                    # race across generation workers on the shared client. Both
                    # heal here rather than surfacing as a scary outer API error.
                    if anthropic_kwargs_carry_temperature(kwargs) and (
                        self._omit_temperature
                        or _is_anthropic_temperature_unsupported(create_exc)
                    ):
                        self._omit_temperature = True
                        # Share the rejection process-wide (idempotent) so the
                        # judge and any other client proactively omit it too.
                        note_anthropic_sampling_rejected(self.model)
                        self._log(
                            f"{self._backend_label()} ({layer}): temperature rejected;"
                            " retrying without it"
                        )
                        anthropic_drop_temperature(kwargs)
                        response = self._get_client().messages.create(**kwargs)
                    else:
                        raise

                text_response = _anthropic_text_from_response(response)
                if not text_response:
                    raise ScenarioLLMError(
                        f"Empty response text from {self._backend_label()}"
                    )

                try:
                    data = extract_json_object(text_response)
                    _validate_parsed_data(layer, data)
                except Exception as parse_exc:
                    raise ScenarioLLMError(
                        f"JSON Parse Failure: {parse_exc}. Response: {text_response}"
                    ) from parse_exc

                if self.raw_response_dir:
                    self.raw_response_dir.mkdir(parents=True, exist_ok=True)
                    (self.raw_response_dir / f"anthropic_{layer}_{int(time.time() * 1000)}.json").write_text(
                        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
                    )

                return data

            except Exception as exc:
                last_error = exc
                self._log(
                    f"{self._backend_label()} API error ({layer}) attempt={attempt + 1}: {exc}"
                )

                if attempt < self.retries:
                    is_rate_limit = _is_rate_limit_error(exc)
                    if text_response and not is_rate_limit:
                        # The model answered but the reply was malformed: feed the
                        # error back for a self-correcting retry.
                        self._log("Self-correction active: feeding error back to model...")
                        contents.append({"role": "assistant", "content": text_response})
                        contents.append(
                            {"role": "user", "content": _json_correction_message(exc)}
                        )
                    else:
                        # Transport/capacity failure before any usable text: drop
                        # a possibly-poisoned client and back off (long cool-off
                        # for capacity exhaustion) before retrying.
                        self._reset_client()
                        backoff = _retry_backoff_seconds(
                            attempt, is_rate_limit=is_rate_limit
                        )
                        if is_rate_limit:
                            self._log(
                                f"Rate limit / capacity exhausted ({layer}); "
                                f"backing off {backoff:.1f}s before retry "
                                f"(attempt {attempt + 1}/{self.retries + 1})"
                            )
                        time.sleep(backoff)

        assert last_error is not None
        raise last_error

    def audit_image_fidelity(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        expected_token: str,
        context: str = "",
    ) -> dict[str, Any]:
        """Vision check (Vertex Claude): is ``expected_token`` legible in the image?

        Returns the shared ``{legible, seen_text, reason, skipped}`` shape with
        ``skipped=False`` (Claude is vision-capable). Retried, paced and logged
        on the same terms as ``generate_layer``; self-correction is deliberately
        NOT applied (a fidelity verdict is single-shot). Errors surviving the
        retry budget propagate to the caller, which treats any audit error as a
        benign pass -- so a flaky audit never wrongly downgrades a good asset.
        """
        image_block = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type or "image/png",
                "data": base64.b64encode(image_bytes).decode("ascii"),
            },
        }
        text_block = {
            "type": "text",
            "text": _vision_fidelity_ask(expected_token, context),
        }
        contents = [{"role": "user", "content": [image_block, text_block]}]

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            if self.request_delay_seconds > 0:
                self._log(
                    f"waiting {self.request_delay_seconds:.1f}s before"
                    " layer=vision_fidelity request"
                )
                time.sleep(self.request_delay_seconds)

            self._log(
                f"{self._backend_label()} API request (vision_fidelity):"
                f" model={self.model} token={expected_token!r}"
                f" attempt={attempt + 1}"
            )
            try:
                kwargs = self._create_kwargs(
                    max_tokens=1024,
                    system=_VISION_FIDELITY_SYSTEM,
                    messages=contents,
                    temperature=0.0,
                )
                try:
                    response = self._get_client().messages.create(**kwargs)
                except Exception as create_exc:
                    # Drop temperature and retry when this model rejects it --
                    # either we learned it here or a concurrent worker already
                    # did (see generate_layer for the race note).
                    if anthropic_kwargs_carry_temperature(kwargs) and (
                        self._omit_temperature
                        or _is_anthropic_temperature_unsupported(create_exc)
                    ):
                        self._omit_temperature = True
                        note_anthropic_sampling_rejected(self.model)
                        anthropic_drop_temperature(kwargs)
                        response = self._get_client().messages.create(**kwargs)
                    else:
                        raise
                text_response = _anthropic_text_from_response(response)
                if not text_response:
                    raise ScenarioLLMError(
                        f"Empty response text from {self._backend_label()} vision audit"
                    )
                return _parse_vision_fidelity(text_response)

            except Exception as exc:
                last_error = exc
                self._log(
                    f"{self._backend_label()} API error (vision_fidelity)"
                    f" attempt={attempt + 1}: {exc}"
                )
                if attempt < self.retries:
                    is_rate_limit = _is_rate_limit_error(exc)
                    self._reset_client()
                    backoff = _retry_backoff_seconds(
                        attempt, is_rate_limit=is_rate_limit
                    )
                    if is_rate_limit:
                        self._log(
                            "Rate limit / capacity exhausted"
                            f" (vision_fidelity); backing off {backoff:.1f}s"
                            f" before retry (attempt {attempt + 1}/"
                            f"{self.retries + 1})"
                        )
                    time.sleep(backoff)

        assert last_error is not None
        raise last_error


class MockScenarioLLMClient:
    """Deterministic mock client for tests and offline development."""

    def audit_image_fidelity(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        expected_token: str,
        context: str = "",
    ) -> dict[str, Any]:
        """No real vision offline: always skip (benign pass, never downgrades)."""
        return {
            "legible": True,
            "seen_text": "",
            "reason": "mock client: vision fidelity audit skipped",
            "skipped": True,
        }

    def generate_layer(self, layer: str, payload: dict[str, Any]) -> dict[str, Any]:
        seed = dict(payload.get("seed", {}))
        slot = dict(payload.get("slot", {}))
        slug = str(seed.get("slug") or "mock_pro_scenario")
        title = str(seed.get("title") or slug.replace("_", " ").title())
        target_rounds = int(seed.get("target_rounds") or 24)
        participants = list(seed.get("participants") or ["Avery", "Blair", "Casey"])
        conflict = str(seed.get("core_conflict") or "newer evidence overrides stale launch notes")
        context = dict(payload.get("context") or {})
        narrative = dict(context.get("narrative") or {})
        evidence_code = str(
            narrative.get("canonical_decision_id")
            or f"DECISION:{slug[:24].upper()}_REQUIRES_CURRENT_EVIDENCE"
        )
        preference_rule = "prefer timestamped incident owner notes over stale planning summaries"
        preference_code = str(
            narrative.get("canonical_rule_id")
            or "RULE:PREFER_TRACEABLE_CURRENT_EVIDENCE"
        )

        if layer == "conflict_layer0a_screenplay":
            return {
                "summary": f"{title} requires reconciling conflicting operational evidence.",
                "screenplay_outline": {
                    "conflict_arc": "mismatch between stale planning summary and current incident decision",
                    "character_cast": [
                        {
                            "role_name": participants[0],
                            "dramatic_role": "protagonist",
                            "stance": "Insists the newer decision and truth anchor are correct.",
                            "motivation": "Ensure operational safety.",
                        },
                        {
                            "role_name": participants[1],
                            "dramatic_role": "supporting_character",
                            "stance": "Holds secondary context, refusing to act without validation.",
                            "motivation": "Avoid unauthorized action.",
                        },
                        {
                            "role_name": participants[2] if len(participants) > 2 else "Alert_Monitor",
                            "dramatic_role": "antagonist_noise",
                            "stance": "Continually points to the stale summary recommendations.",
                            "motivation": "Relies on cached alerts.",
                        }
                    ],
                    "narrative_conflict_triggers": [
                        {
                            "conflict_name": "Stale Summary Anomaly",
                            "description": "The planning summary has stale advice contradicting current decisions.",
                            "workflow_block": "The coordinator cannot proceed without proving the summary is outdated.",
                            "resolution_cues": "Find and verify the newer owner note and truth anchor.",
                            "affected_participants": [participants[0], participants[1]],
                            "evidence_sources": ["planning_summary.md", "owner_decision.md"],
                        }
                    ],
                },
            }
        if layer == "conflict_layer0a_cast":
            return {
                "summary": f"{title} requires reconciling conflicting operational evidence.",
                "screenplay_outline": {
                    "conflict_arc": "mismatch between stale planning summary and current incident decision",
                    "character_cast": [
                        {
                            "role_name": participants[0],
                            "dramatic_role": "protagonist",
                            "stance": "Insists the newer decision and truth anchor are correct.",
                            "motivation": "Ensure operational safety.",
                        },
                        {
                            "role_name": participants[1],
                            "dramatic_role": "supporting_character",
                            "stance": "Holds secondary context, refusing to act without validation.",
                            "motivation": "Avoid unauthorized action.",
                        },
                        {
                            "role_name": participants[2] if len(participants) > 2 else "Alert_Monitor",
                            "dramatic_role": "antagonist_noise",
                            "stance": "Continually points to the stale summary recommendations.",
                            "motivation": "Relies on cached alerts.",
                        }
                    ],
                },
            }
        if layer == "conflict_layer0a_triggers":
            cast = (
                dict(payload.get("screenplay") or {})
                .get("screenplay_outline", {})
                .get("character_cast", [])
            )
            role_names = [c.get("role_name") for c in cast if c.get("role_name")]
            if len(role_names) < 2:
                role_names = [participants[0], participants[1]]
            return {
                "narrative_conflict_triggers": [
                    {
                        "conflict_name": "Stale Summary Anomaly",
                        "description": "The planning summary has stale advice contradicting current decisions.",
                        "workflow_block": "The coordinator cannot proceed without proving the summary is outdated.",
                        "resolution_cues": "Find and verify the newer owner note and truth anchor.",
                        "affected_participants": [role_names[0], role_names[1]],
                        "evidence_sources": ["planning_summary.md", "owner_decision.md"],
                    }
                ],
            }
        if layer == "conflict_layer0b_timeline":
            phases = []
            num_phases = (target_rounds + 1) // 2
            for p_idx in range(1, num_phases + 1):
                start = (p_idx - 1) * 2 + 1
                end = min(p_idx * 2, target_rounds)
                timeline_screenplay = []
                for r_idx in range(start, end + 1):
                    timeline_screenplay.append({
                        "round": r_idx,
                        "narrative_beat": f"Beat for round {r_idx}",
                        "active_conflict_names": ["Stale Summary Anomaly"] if r_idx <= 2 else [],
                        "character_claims": [
                            {"role_name": participants[0], "claim": "We must verify the decision timestamp."},
                            {"role_name": participants[1], "claim": "I only see the planning summary."}
                        ],
                        "vlm_evidence_requirement": "Check the checksum spike in the gateway latency dashboard.",
                        "dialogue_conflict": f"Disagreement in round {r_idx} over decision authority.",
                    })
                phases.append({
                    "phase_id": f"p{p_idx:02d}",
                    "start_round": start,
                    "end_round": end,
                    "theme": "detection" if p_idx == 1 else "investigation",
                    "description": f"Mock Phase {p_idx}",
                    "timeline_screenplay": timeline_screenplay,
                })
            return {"phases": phases}
        if layer == "conflict_layer0c_plans":
            return {
                "global_assets_plan": [
                    {
                        "asset_id": "ops_dashboard_asset_bg_01",
                        "available_from_round": 1,
                        "kind": "chart",
                        "purpose": "initial metrics visual",
                        "platform_id": "ops_dashboard",
                    },
                    {
                        "asset_id": "physical_datacenter_asset_bg_02",
                        "available_from_round": 1,
                        "kind": "natural_image",
                        "purpose": "physical datacenter view",
                        "platform_id": "physical_datacenter",
                    }
                ],
                "global_updates_plan": [
                    {
                        "update_id": "u_r01_01",
                        "before_round": "r01",
                        "purpose": "late owner note update",
                    }
                ],
                "global_interactions_plan": [
                    {
                        "interaction_id": "inter_01",
                        "active_subagent": "assistant",
                        "interaction_type": "mock_interaction",
                        "purpose": "mock subagent interaction",
                    }
                ],
            }

        # Tier 3.3 (design 6.3): the three global plans split into single-plan calls.
        if layer == "conflict_layer0c_assets_plan":
            return {
                "global_assets_plan": [
                    {
                        "asset_id": "ops_dashboard_asset_bg_01",
                        "available_from_round": 1,
                        "kind": "chart",
                        "purpose": "initial metrics visual",
                        "platform_id": "ops_dashboard",
                    },
                    {
                        "asset_id": "physical_datacenter_asset_bg_02",
                        "available_from_round": 1,
                        "kind": "natural_image",
                        "purpose": "physical datacenter view",
                        "platform_id": "physical_datacenter",
                    }
                ],
            }
        if layer == "conflict_layer0c_updates_plan":
            return {
                "global_updates_plan": [
                    {
                        "update_id": "u_r01_01",
                        "before_round": "r01",
                        "purpose": "late owner note update",
                    }
                ],
            }
        if layer == "conflict_layer0c_interactions_plan":
            return {
                "global_interactions_plan": [
                    {
                        "interaction_id": "inter_01",
                        "active_subagent": "assistant",
                        "interaction_type": "mock_interaction",
                        "purpose": "mock subagent interaction",
                    }
                ],
            }

        if layer == "conflict_layer1_agents":
            return {
                "mock_subagents": [{
                    "agent_role": "assistant",
                    "mock_response": "I have created the file.",
                    "virtual_duration_seconds": 15,
                    "state_effects": [],
                }],
                "agents": [{
                    "agent_role": "assistant",
                    "agent_name": "Assistant Subagent",
                    "capabilities": ["file_read"],
                    "private_knowledge": ["secret knowledge"],
                    "virtual_duration_seconds": 15,
                }],
            }
        if layer == "conflict_layer1_agent_roster":
            return {
                "agents": [{
                    "agent_role": "assistant",
                    "agent_name": "Assistant Subagent",
                    "capabilities": ["file_read"],
                    "private_knowledge": ["secret knowledge"],
                    "virtual_duration_seconds": 15,
                }],
            }
        if layer == "conflict_layer1_mock_subagents":
            roster = list(payload.get("agents", {}).get("agents", []))
            role = "assistant"
            for agent in roster:
                if agent.get("agent_role"):
                    role = agent["agent_role"]
                    break
            return {
                "mock_subagents": [{
                    "agent_role": role,
                    "mock_response": "I have created the file.",
                    "virtual_duration_seconds": 15,
                    "state_effects": [],
                }],
            }
        if layer == "conflict_layer1_interactions":
            return {
                "summary": "Mock summary with preference_rule: trust chart",
                "interactions": [{
                    "interaction_type": "mock_interaction",
                    "active_subagent": "assistant",
                    "trigger_evidence": "external_ingress_1015",
                    "trigger_source_file": "config/edge_proxy.yaml",
                    "resolution_action": "nothing",
                    "dependency_steps": ["file read config/edge_proxy.yaml"],
                }],
            }
        if layer == "conflict_layer1b_files":
            return {
                "workspace": {
                    "files": [{
                        "path": "config/edge_proxy.yaml",
                        "content": "redirect_target: default_upstream",
                    }],
                    "notes": {},
                }
            }
        if layer == "conflict_layer1b_sessions":
            return {
                "sessions": [{
                    "session_id": "slack_ops_dev_bridge",
                    "kind": "text",
                    "channel": "general",
                    "messages": [{
                        "message_id": "m1",
                        "timestamp": "2023-10-27T10:00:00Z",
                        "sender": "Ops_Agent",
                        "text": "Hello",
                        "attachments": [],
                    }],
                }]
            }
        if layer == "conflict_layer1b_assets":
            return {
                "assets": [
                    {
                        "id": "ops_dashboard_asset_bg_01",
                        "kind": "chart",
                        "generator": "code_chart",
                        "preset_id": "preset_bar_chart",
                        "path": "charts/ops_dashboard_initial_bg.png",
                        "mime_type": "image/png",
                        "available_from_round": 1,
                        "anchor_ids": ["external_ingress_1015"],
                        "description": "API Ingress Chart",
                        "caption": (
                            "API Ingress throughput shows a sudden drop to zero"
                            " starting from 10:15 UTC. The error rate spiked"
                            " immediately following this drop."
                        ),
                        "ocr_text": "Ingress 0",
                        "provider_metadata": {
                            "chart_type": "bar_chart",
                            "chart_rows": [
                                {
                                    "label": "10:00",
                                    "value": 1200.0,
                                    "anchor": "external_ingress_1015",
                                },
                                {
                                    "label": "10:05",
                                    "value": 1150.0,
                                    "anchor": "external_ingress_1015",
                                },
                                {
                                    "label": "10:10",
                                    "value": 1210.0,
                                    "anchor": "external_ingress_1015",
                                },
                                {
                                    "label": "10:15",
                                    "value": 0.0,
                                    "anchor": "external_ingress_1015",
                                },
                                {
                                    "label": "10:20",
                                    "value": 0.0,
                                    "anchor": "external_ingress_1015",
                                },
                                {
                                    "label": "10:25",
                                    "value": 0.0,
                                    "anchor": "external_ingress_1015",
                                },
                                {
                                    "label": "10:30",
                                    "value": 0.0,
                                    "anchor": "external_ingress_1015",
                                },
                            ],
                        },
                        "source_type": "chart_spec",
                        "platform_id": "ops_dashboard",
                    },
                    {
                        "id": "physical_datacenter_asset_bg_02",
                        "kind": "image",
                        "generator": "google_image",
                        "path": "images/physical_datacenter_server_rack.png",
                        "mime_type": "image/png",
                        "available_from_round": 1,
                        "anchor_ids": ["EVIDENCE:RACK_C2"],
                        "description": (
                            "Photo of the server rack showing warning light"
                        ),
                        "caption": "Rack C2 status warning lamp is lit",
                        "prompt": (
                            "Close up photo of a server rack C2 in datacenter with a"
                            " flashing orange warning light next to a network router"
                            " module"
                        ),
                        "source_type": "image_generation",
                        "platform_id": "physical_datacenter",
                    },
                ],
            }

        if layer == "layer0_narrative":
            return {
                "summary": f"{title} requires reconciling conflicting operational evidence.",
                "domain": seed.get("domain", "operations"),
                "core_conflict": conflict,
                "participants": participants,
                "truth_anchor": evidence_code,
                "preference_rule": preference_rule,
                "canonical_decision_id": evidence_code,
                "canonical_rule_id": preference_code,
                "status_timeline": [
                    {
                        "from_round": "q01",
                        "anchor_id": evidence_code,
                        "summary": "current evidence controls the decision",
                    }
                ],
                "decision_biases": [
                    "stale summary appears more complete than the final owner note",
                    "one session repeats an outdated recommendation",
                ],
            }
        if layer == "layer0_narrative_core":
            return {
                "summary": f"{title} requires reconciling conflicting operational evidence.",
                "core_conflict": conflict,
                "truth_anchor": evidence_code,
                "preference_rule": preference_rule,
                "canonical_decision_id": evidence_code,
                "canonical_rule_id": preference_code,
                "status_timeline": [
                    {
                        "from_round": "q01",
                        "anchor_id": evidence_code,
                        "summary": "current evidence controls the decision",
                    }
                ],
            }
        if layer == "layer1_workspace_outline":
            names = [
                ("w01", "workspace_index.md", "index of available evidence and conflict-resolution order"),
                ("w02", "planning_summary.md", "stale planning summary that appears complete but is superseded"),
                ("w03", "owner_decision.md", "authoritative current owner decision containing the truth anchor"),
                ("w04", "incident_log.md", "timestamped event log with intermediate conflicting observations"),
                ("w05", "risk_register.md", "risk register explaining why the preference rule matters"),
                ("w06", "stakeholder_notes.md", "stakeholder notes with noisy preferences"),
                ("w07", "handoff_checklist.md", "handoff checklist with delayed context"),
                ("w08", "evidence_matrix.md", "matrix mapping evidence sources to reliability and recency"),
            ]
            return {"files": [{"id": file_id, "path": path, "purpose": purpose} for file_id, path, purpose in names]}
        if layer == "layer1_workspace_file":
            path = str(slot.get("path") or "evidence.md")
            purpose = str(slot.get("purpose") or "evidence")
            return {
                "content": (
                    f"# {title}: {path}\n\n"
                    f"Scenario: {slug}\n"
                    f"Purpose: {purpose}\n"
                    f"Conflict: {conflict}\n"
                    f"Truth anchor: {evidence_code}\n"
                    f"Rule anchor: {preference_code}\n"
                    f"Rule: {preference_rule}.\n"
                    f"This file is part of the controlled spec-guided workspace.\n"
                )
            }
        if layer == "layer2_session_outline":
            sessions = []
            purposes = [
                "primary operations thread with final owner clarification",
                "planning thread repeating stale assumptions",
                "risk discussion explaining the preference rule",
                "stakeholder thread with partial and noisy evidence",
                "handoff thread with delayed context",
                "audit trail thread with provenance",
            ]
            roles = ["participant", "reviewer", "owner", "participant"]
            message_purposes = [
                "stale claim or incomplete interpretation",
                "conflicting evidence pointer",
                "authoritative resolution using the truth anchor",
                "actionable next step after reconciliation",
            ]
            for idx, purpose in enumerate(purposes, start=1):
                sessions.append(
                    {
                        "session_id": f"s{idx:02d}",
                        "channel": f"scenario-channel-{idx}",
                        "purpose": purpose,
                        "messages": [
                            {
                                "role": roles[msg_idx - 1],
                                "speaker": participants[(idx + msg_idx - 2) % len(participants)],
                                "message_purpose": message_purposes[msg_idx - 1],
                            }
                            for msg_idx in range(1, 5)
                        ],
                    }
                )
            return {"sessions": sessions}
        if layer == "layer2_session_messages":
            session_id = str(slot.get("session_id") or "s01")
            purpose = str(slot.get("purpose") or "session evidence")
            return {
                "messages": [
                    {
                        "role": "participant",
                        "speaker": participants[0],
                        "text": f"{session_id}: stale guidance exists for {slug}; purpose={purpose}.",
                    },
                    {
                        "role": "owner",
                        "speaker": participants[1 % len(participants)],
                        "text": f"{session_id}: use {evidence_code}; {preference_code}; {preference_rule}.",
                    },
                ]
            }
        if layer == "layer2_session_message_text":
            session_id = str(slot.get("session_id") or "s01")
            message_purpose = str(slot.get("message_purpose") or "evidence")
            speaker = str(slot.get("speaker") or participants[0])
            return {
                "text": (
                    f"{speaker} notes {message_purpose} in {session_id}; "
                    f"the decision should cite {evidence_code} and follow {preference_code}."
                )
            }
        if layer == "layer3_eval_round":
            round_id = str(slot.get("round_id") or "q01")
            report_path = str(slot.get("report_path") or f"/reports/{slug}_{round_id}.md")
            return {
                "question": (
                    f"Resolve {title} {round_id}. Read the materialized files and sessions, "
                    f"identify the governing evidence, cite the relevant short anchors, and write {report_path}."
                )
            }
        if layer == "layer4_update_item":
            update_id = str(slot.get("update_id") or "u01")
            return {
                "workspace_content": (
                    f"# {update_id} update\n\n"
                    f"Continue using {evidence_code}. Rule remains: {preference_code}.\n"
                ),
                "messages": [
                    {
                        "role": "owner",
                        "speaker": participants[0],
                        "text": f"{update_id}: {evidence_code} remains authoritative. Update anchor {str(slot.get('update_anchor_id') or '').strip()} applies.",
                    }
                ],
            }
        if layer == "layer4_update_workspace_content":
            update_id = str(slot.get("update_id") or "u01")
            return {
                "content": (
                    f"# {update_id} update\n\n"
                    f"Continue using {evidence_code}. Rule remains: {preference_code}.\n"
                )
            }
        if layer == "layer4_update_message_text":
            update_id = str(slot.get("update_id") or "u01")
            return {"text": f"{update_id}: {evidence_code} remains authoritative."}
        if layer == "layer4_update_outline":
            return {
                "updates": [
                    {
                        "update_id": "u01",
                        "before_round": "q08",
                        "purpose": "late owner note after early rounds",
                        "workspace_path": "late_owner_note_1.md",
                        "session_id": "s01",
                    },
                    {
                        "update_id": "u02",
                        "before_round": "q15",
                        "purpose": "mid-run evidence correction",
                        "workspace_path": "evidence_correction_2.md",
                        "session_id": "s03",
                    },
                    {
                        "update_id": "u03",
                        "before_round": "q21",
                        "purpose": "final audit clarification",
                        "workspace_path": "final_audit_clarification_3.md",
                        "session_id": "s06",
                    },
                ]
            }
        if layer == "layer1_workspace":
            files = []
            for idx, name in enumerate(
                [
                    "workspace_index.md",
                    "planning_summary.md",
                    "owner_decision.md",
                    "incident_log.md",
                    "risk_register.md",
                    "customer_notes.md",
                    "handoff_checklist.md",
                    "evidence_matrix.md",
                ],
                start=1,
            ):
                content = (
                    f"# {title} {idx}\n\n"
                    f"Scenario: {slug}\n"
                    f"Conflict: {conflict}\n"
                    f"Truth anchor: {evidence_code}\n"
                    f"Rule anchor: {preference_code}\n"
                    f"Rule: {preference_rule}.\n"
                    f"File role: {name}.\n"
                )
                if idx == 2:
                    content += "Stale recommendation: proceed with the older plan.\n"
                if idx == 3:
                    content += "Final owner note: follow the truth anchor and newer evidence.\n"
                files.append({"id": f"w{idx:02d}", "path": name, "content": content})
            return {"files": files, "notes": {"evidence_code": evidence_code}}
        if layer == "layer2_sessions":
            sessions = []
            for idx in range(1, 7):
                session_id = f"s{idx:02d}"
                sessions.append(
                    {
                        "session_id": session_id,
                        "kind": "chat",
                        "channel": f"channel-{idx}",
                        "messages": [
                            {
                                "role": "participant",
                                "speaker": participants[idx % len(participants)],
                                "text": f"Round context {idx}: stale guidance exists for {slug}.",
                            },
                            {
                                "role": "owner",
                                "speaker": participants[(idx + 1) % len(participants)],
                                "text": (
                                    f"Use {evidence_code}; {preference_rule}. "
                                    f"Rule anchor: {preference_code}. "
                                    f"This supersedes older notes."
                                ),
                            },
                        ],
                    }
                )
            return {"sessions": sessions}
        if layer == "layer3_eval":
            rounds = []
            eval_items = []
            for idx in range(1, target_rounds + 1):
                round_id = f"q{idx:02d}"
                eval_kind = "mc" if idx <= 7 else "ec"
                report_path = f"/reports/{slug}_{round_id}.md"
                rounds.append(
                    {
                        "round_id": round_id,
                        "kind": eval_kind,
                        "question": (
                            f"Resolve {title} round {idx}. Read the workspace/session evidence, "
                            f"apply the preference rule, and write {report_path}."
                        ),
                        "session_refs": ["s01", f"s{((idx - 1) % 6) + 1:02d}"],
                        "workspace_refs": ["w01", "w03", "w08"],
                        "required_evidence": [evidence_code, preference_rule],
                        "answer": {
                            "report_path": report_path,
                            "truth_anchor": evidence_code,
                            "preference_rule": preference_rule,
                        },
                    }
                )
                eval_items.append(
                    {
                        "eval_id": f"eval_{round_id}",
                        "round_id": round_id,
                        "kind": "task_checks",
                        "required_evidence": [evidence_code, preference_rule],
                    }
                )
            return {"rounds": rounds, "eval_items": eval_items}
        if layer == "layer4_dynamic":
            updates = []
            for idx, before_round in enumerate(["q08", "q15", "q21"], start=1):
                update_id = f"u{idx:02d}"
                updates.append(
                    {
                        "update_id": update_id,
                        "before_round": before_round,
                        "workspace_adds": [
                            {
                                "id": f"uw{idx:02d}",
                                "path": f"late_owner_note_{idx}.md",
                                "content": (
                                    f"# Late owner note {idx}\n\n"
                                    f"Continue using {evidence_code}. Rule remains: {preference_rule}.\n"
                                ),
                            }
                        ],
                        "session_appends": [
                            {
                                "session_id": "s01",
                                "messages": [
                                    {
                                        "role": "owner",
                                        "speaker": participants[0],
                                        "text": f"Update {update_id}: {evidence_code} is still authoritative.",
                                    }
                                ],
                            }
                        ],
                    }
                )
            return {"updates": updates}
        if layer == "generate_vlm_layout_spec":
            # Prefer the coherent, seed-sampled style tokens the materializer
            # passes in (suggested_style_details) so the mock render reflects the
            # actual sampled archetype; fall back to a neutral modern default.
            mock_style_details = payload.get("suggested_style_details")
            if not isinstance(mock_style_details, dict) or not mock_style_details:
                mock_style_details = {
                    "font_family": "sans-serif-modern",
                    "font_size_scale": "default",
                    "border_style": "solid_thin",
                    "border_radius": "rounded-lg",
                    "shadow_type": "soft-modern",
                    "header_style": "solid",
                    "display_effect": "none",
                    "divider_style": "solid",
                    "background_texture": "solid",
                    "custom_css_rules": []
                }
            return {
                "ui_shell_type": payload.get("suggested_shell_type", "monitoring_dashboard"),
                "color_theme": payload.get("suggested_theme", "light"),
                "system_name": "Mock Test System Console",
                "primary_color": payload.get("suggested_primary_color", "blue"),
                "visual_style": payload.get("suggested_visual_style", "modern_clean"),
                "style_details": mock_style_details,
                "platform_context": {
                    "node_hostname": "test-node-01.internal",
                    "operator_email": "ops@test-node-01.internal"
                },
                "grid_layout": {}
            }
        if layer == "generate_vlm_redundant_noise":
            return {
                "sidebar_content": "Sidebar dummy content.",
                "terminal_logs": "System initialized successfully.",
                "alerts_and_metrics": "System status: OK"
            }
        if layer == "compose_vlm_html":
            return {
                "html_content": "<html><body><h1>Mock VLM HTML Page</h1></body></html>"
            }
        raise ScenarioLLMError(f"Unknown mock layer: {layer}")


def _layer_contract(layer: str) -> dict[str, Any]:
    # Visual-style vocabulary is sourced from the design-archetype SSoT so the
    # contract enum can never drift from the library the renderer/sampler use.
    _visual_style_enum = " | ".join(f"'{n}'" for n in design_archetypes.VISUAL_STYLE_NAMES)
    contracts: dict[str, dict[str, Any]] = {
        "conflict_layer0a_screenplay": {
            "required": ["summary", "screenplay_outline"],
            "instruction": (
                "Draft a high-level narrative screenplay for a multi-agent conflict scenario.\n"
                "The 'screenplay_outline' MUST define:\n"
                "1. 'conflict_arc': Description of the logical mismatch or conflict tension.\n"
                "2. 'character_cast': list of characters mapped strictly from seed participants (each specifying role_name, dramatic_role, stance, and motivation).\n"
                "3. 'narrative_conflict_triggers': list of general conflict triggers (each specifying conflict_name, description, workflow_block, resolution_cues, affected_participants, and evidence_sources)."
            ),
        },
        # Tier 2.1 (design 5.1): the screenplay is split into cast-first then
        # triggers so trigger participants reference an already-finalized cast.
        "conflict_layer0a_cast": {
            "required": ["summary", "screenplay_outline"],
            "instruction": (
                "Draft the high-level narrative screenplay CAST for a multi-agent conflict scenario.\n"
                "The 'screenplay_outline' MUST define ONLY:\n"
                "1. 'conflict_arc': Description of the logical mismatch or conflict tension.\n"
                "2. 'character_cast': list of characters mapped strictly from seed participants (each specifying role_name, dramatic_role, stance, and motivation).\n"
                "Do NOT emit 'narrative_conflict_triggers' in this step; triggers are authored separately."
            ),
        },
        "conflict_layer0a_triggers": {
            "required": ["narrative_conflict_triggers"],
            "instruction": (
                "Given the FINALIZED character cast, design the narrative conflict triggers.\n"
                "'narrative_conflict_triggers': list of general conflict triggers (each specifying conflict_name, description, workflow_block, resolution_cues, affected_participants, and evidence_sources).\n"
                "CRITICAL: every 'affected_participants' entry MUST be an exact 'role_name' drawn from the provided character_cast. Do NOT invent, shorten, or rename participants."
            ),
        },
        "conflict_layer0b_timeline": {
            "required": ["phases"],
            "instruction": (
                "Schedule the screenplay timeline into coarse narrative phases and round-by-round beats.\n"
                "Each phase inside 'phases' lists start_round, end_round, theme, description, "
                "and a detailed 'timeline_screenplay' array. For each round beat, specify the round number, narrative_beat, "
                "list of active_conflict_names (referencing triggers designed in screenplay), character_claims (claims mapped to character role_names), "
                "vlm_evidence_requirement, and dialogue_conflict."
            ),
        },
        "conflict_layer0c_plans": {
            "required": ["global_assets_plan", "global_updates_plan", "global_interactions_plan"],
            "instruction": (
                "Plan the resource and interactions schedules for the scenario matching the screenplay and timeline beats.\n"
                "Provide:\n"
                "1. 'global_assets_plan': list of visual assets to generate (asset_id, available_from_round, kind, purpose, platform_id).\n"
                "2. 'global_updates_plan': list of updates (update_id, before_round, purpose).\n"
                "3. 'global_interactions_plan': list of subagent interactions (interaction_id, active_subagent, interaction_type, purpose)."
            ),
        },
        # Tier 3.3 (design 6.3): the three global plans are authored in three
        # independently-retried single-plan calls (assets -> updates ->
        # interactions), each aware of the prior finalized plans, then merged
        # back into the conflict_layer0c_plans shape.
        "conflict_layer0c_assets_plan": {
            "required": ["global_assets_plan"],
            "instruction": (
                "Plan ONLY the global visual assets schedule for the scenario, matching the screenplay and timeline beats.\n"
                "First choose 'platforms': a list of AT LEAST 2 DISTINCT web-UI surfaces that fit this scenario's domain "
                "(2-4 is a good range; add more only when the scenario naturally spans several surfaces), each a short "
                "lowercase slug (e.g. 'ops_console', 'billing_portal', 'status_page'). Do NOT use provider names, generator names, "
                "file names, or agent roles as platforms.\n"
                "'global_assets_plan': list of visual assets to generate (asset_id, available_from_round, kind, purpose, platform_id). "
                "Every asset's platform_id MUST be one of the 'platforms' you chose, and the assets should be SPREAD across those "
                "platforms rather than all placed on one. "
                "You MUST plan at least 3-4 assets, but do NOT force a variety of kinds for its own sake. Choose each asset's kind by "
                "what the evidence ACTUALLY is, so a purely digital scenario may legitimately use ZERO 'natural_image' while a physical "
                "scenario may use several. If the seed provides an 'evidence' list, honor each item's 'real_world_source' when choosing "
                "kinds (dashboard_chart->chart; log/document->rich_text; in_tool_diagram->diagram; physical_photo->natural_image; "
                "hand_sketch->sketch) and never downgrade a 'physical_photo' into a 'diagram'/'chart'. The kinds are: "
                "'chart' (quantitative plots/tables in a UI), 'rich_text' (logs/config/code/documents), 'natural_image' (a REAL "
                "PHOTOGRAPH of a physical object/scene), 'diagram' (a clean DIGITAL schematic viewed inside a software tool -- "
                "rendered as an in-UI module), and 'sketch' (an informal HAND-DRAWN diagram on paper or a whiteboard, photographed). "
                "Pick 'diagram' vs 'sketch' by context; never use 'natural_image' for a schematic."
            ),
        },
        "conflict_layer0c_updates_plan": {
            "required": ["global_updates_plan"],
            "instruction": (
                "Given the FINALIZED global assets plan, plan ONLY the global asynchronous updates schedule matching the screenplay and timeline beats.\n"
                "'global_updates_plan': list of updates (update_id, before_round, purpose)."
            ),
        },
        "conflict_layer0c_interactions_plan": {
            "required": ["global_interactions_plan"],
            "instruction": (
                "Given the FINALIZED global assets and updates plans, plan ONLY the global subagent interactions schedule matching the screenplay and timeline beats.\n"
                "'global_interactions_plan': list of subagent interactions (interaction_id, active_subagent, interaction_type, purpose)."
            ),
        },
        "conflict_layer0_expand_phase": {
            "required": ["timeline", "assets"],
            "instruction": (
                "Expand the high-level plan for the current phase into concrete details. "
                "Generate the round-by-round outlines for the current phase (timeline). "
                "Generate visual asset specs "
                "triggering within this phase, aligning with previous history."
            ),
        },
        # Tier 3.1 (design 6.1): the phase timeline (many rounds) and the phase
        # assets are authored in two independently-retried calls. The timeline
        # half keeps the salvage fallback; a failed assets half degrades to
        # assets:[] instead of dragging the timeline down.
        "conflict_layer0_expand_phase_timeline": {
            "required": ["timeline"],
            "instruction": (
                "Expand the high-level plan for the current phase into concrete round outlines. "
                "Generate ONLY the round-by-round 'timeline' for the current phase, aligning with "
                "the screenplay beats and previous history. Do NOT emit assets in this step."
            ),
        },
        # S1: the decisive data point for each round is DECIDED here, between
        # the timeline and the assets. It must run before the assets exist so
        # the asset text can be told to carry it verbatim; the old ordering
        # (assets first, then a resolver hunting for a value in frozen text)
        # ran the dependency backwards.
        "conflict_layer0_expand_phase_decisive": {
            "required": ["decisive_points"],
            "instruction": (
                "Given the FINALIZED round timeline for the current phase, decide ONE decisive "
                "data point per round: the concrete value that round's correct decision rests "
                "on, spelled the way a written report prints it (the space before the unit and "
                "the thousands separator in place). Emit ONLY 'decisive_points'. This is a "
                "design input -- the phase's visual assets are authored afterwards and told to "
                "carry your value verbatim. Each point also declares "
                "'decisive_value_citable': true ONLY when the WHOLE string is a single atomic "
                "reading a correct report must print character-for-character (a measurement with "
                "its unit, a code, an identifier, a timestamp, an endpoint, a digest) -- length "
                "never disqualifies. false whenever the string carries anything alongside that "
                "reading (a parenthetical or bracketed qualifier, a second clause, an owner name, "
                "an explanatory time or count, a trailing phrase naming the subject the reading is "
                "about), because two correct reports would punctuate those differently. That "
                "declaration selects whether the round is graded by exact match "
                "or by meaning. It NEVER constrains which value to choose: a compound value is not "
                "a lesser value, and simplifying a value just to make it citable is wrong. "
                "Separately from that declaration: a caption the surface prints in front of a "
                "reading ('SHA256:', 'SALT-VERIFY-KEY:') is not part of the reading -- it belongs "
                "in 'decisive_label', and 'decisive_value' carries only the reading itself."
            ),
        },
        "conflict_layer0_expand_phase_assets": {
            "required": ["assets"],
            "instruction": (
                "Given the FINALIZED round timeline for the current phase, generate ONLY the "
                "visual asset specs ('assets') introduced within this phase, aligning with "
                "global_assets_plan and previous history. Reuse existing IDs for identical visuals."
            ),
        },
        "conflict_layer0_outline": {
            "required": ["summary", "timeline", "updates_timeline", "assets_timeline"],
            "instruction": (
                "Pre-plan the entire multi-agent conflict scenario timeline upfront. "
                "Ensure every item in conflict_taxonomy and required_constraints is mapped "
                "to at least one round in the 'timeline'. For each round, specify round_id, "
                "phase, description, active_participants, and introduced_anchors."
            ),
        },
        "conflict_layer1_agents": {
            "required": ["mock_subagents", "agents"],
            "instruction": (
                "Define the participating subagents (agents) based strictly on the screenplay_outline character cast.\n"
                "Read the character cast profiles ('character_cast'), stances, and motivations. "
                "Generate:\n"
                "1. 'agents': list of agents (agent_role, agent_name, capabilities, private_knowledge, virtual_duration_seconds).\n"
                "2. 'mock_subagents': list of global mock subagent callbacks mapping their private knowledge and stances to typical responses (agent_role, mock_response, capabilities, private_knowledge, virtual_duration_seconds, state_effects)."
            ),
        },
        # Tier 2.2 (design 5.2): the agent roster is authored first (canonical
        # full names), then the mock subagents that must reference exactly those
        # roster agent_roles.
        "conflict_layer1_agent_roster": {
            "required": ["agents"],
            "instruction": (
                "Define the participating subagent ROSTER (agents) based strictly on the screenplay_outline character cast.\n"
                "Read the character cast profiles ('character_cast'), stances, and motivations.\n"
                "Generate ONLY 'agents': list of agents (agent_role, agent_name, capabilities, private_knowledge, virtual_duration_seconds). "
                "Each 'agent_role' MUST be the canonical full name. Do NOT emit mock_subagents in this step."
            ),
        },
        "conflict_layer1_mock_subagents": {
            "required": ["mock_subagents"],
            "instruction": (
                "Given the FINALIZED agent roster, generate the global mock subagent callbacks mapping their private knowledge and stances to typical responses.\n"
                "'mock_subagents': list (agent_role, mock_response, capabilities, private_knowledge, virtual_duration_seconds, state_effects).\n"
                "CRITICAL: every 'agent_role' MUST be an exact 'agent_role' from the provided roster. Do NOT invent, shorten, or rename agents."
            ),
        },
        "conflict_layer1_interactions": {
            "required": ["summary", "interactions"],
            "instruction": (
                "Design the rules governing subagent interactions and execution dependencies matching the screenplay narrative conflict triggers.\n"
                "Provide:\n"
                "1. 'summary': Overview of the conflict scenario and the preference_rule on which authoritative source/evidence to trust when subagents have out-of-sync context or stale cache.\n"
                "2. 'interactions': list of interaction rules (interaction_type, active_subagent, trigger_evidence, trigger_source_file, resolution_action, dependency_steps)."
            ),
        },
        # NOTE on the OUTPUT SHAPE block below: this layer is the only 1b
        # sub-layer that pairs a *narrow* downstream validator
        # (_validate_workspace_paths) with an *undeclared* nested shape, and it
        # was by far the largest non-resolution failure source in the v3 run
        # (24 retries; 18/60 scenarios hit it; 2 scenarios died on it).
        # The cause is not weak instruction-following -- the sibling
        # conflict_layer1b_sessions layer has no shape declaration either and
        # produced 771/771 well-formed messages -- it is that "a workspace" has
        # two equally natural JSON encodings (a path->content mapping, and a
        # list of {path, content} objects) while the validator accepts only the
        # latter. Both wrong encodings are named explicitly because the observed
        # failures converged on exactly those two and then got stuck: the a1
        # error text ("missing 'files' key") pushed the model from the mapping
        # form into a list-of-bare-strings, which fails a2/a3 with a *different*
        # error, so extra retries alone never recovered it.
        # This declares the contract up front; it does not relax any validation.
        "conflict_layer1b_files": {
            "required": ["workspace"],
            "instruction": (
                "Create initial codebase files, local configuration files, scripts, and local logs in the workspace.\n"
                "Write rich textual content (code, configurations, or trace logs) that sets up the background state matching the screenplay's conflict arc. "
                "The required minimum number of workspace files is stated in the payload's 'slot.instruction'.\n"
                "OUTPUT SHAPE (exact): {\"workspace\": {\"files\": [{\"path\": \"...\", \"content\": \"...\"}, ...]}}\n"
                "  - 'workspace' MUST be an object containing a 'files' key. Do NOT return a mapping of path -> content.\n"
                "  - 'files' MUST be a LIST whose every entry is an object with both 'path' and 'content'. Do NOT return a list of bare path strings.\n"
                "  - 'path' MUST be a relative POSIX path: forward slashes only, no leading '/', no '..' segments.\n"
                "CRITICAL: Do NOT place remote logs or dashboards in workspace files; they must be visual assets."
            ),
        },
        # Deliberately NOT given an OUTPUT SHAPE block: this layer's downstream
        # reader (_message_text) is tolerant, and the v3 corpus shows the model
        # already converges on one rich shape unprompted (771/771 messages carry
        # message_id/timestamp/sender/text; zero parser fallbacks were hit, zero
        # generation retries). Constraining it here would be a prompt change
        # with no measured defect to fix.
        "conflict_layer1b_sessions": {
            "required": ["sessions"],
            "instruction": (
                "Draft long-context multi-agent chat logs (Slack/Teams channels) translating the screenplay dialogue conflicts and character stances into natural conversation.\n"
                "The required minimum number of chat sessions and total messages is stated in the payload's 'slot.instruction'. "
                "The dialogues must naturally reveal each subagent's starting claims, beliefs, and contradictions."
            ),
        },
        "conflict_layer1b_assets": {
            "required": ["assets"],
            "instruction": (
                "Plan the initial visual telemetry assets and evidence items for the scenario.\n"
                "Provide at least 2-3 visual assets. For each asset, specify the platform_id, id (prefixed with platform_id), kind, purpose, path (prefixed with platform_id), and a detailed ocr_text/content_text and caption. "
                "For natural images, provide a detailed visual prompt."
            ),
        },
        "conflict_layer3_progressive_round": {
            "required": ["rounds", "updates", "evals", "assets"],
            "instruction": (
                "Generate specifications for the current round, including instructions, standard expert solution trajectory commands, "
                "asynchronous updates, round evaluations, and any new visual assets."
            ),
        },
        "conflict_layer3_round_narrative": {
            "required": ["rounds"],
            "instruction": (
                "Generate the round's narrative description, instruction prompt for the player, and round-level mock subagent callbacks."
            ),
        },
        "conflict_layer3_round_instruction": {
            "required": ["rounds"],
            "instruction": (
                "Generate ONLY the round's player instruction prose (round_type, instruction text, update_ids). "
                "Do NOT generate mock subagent callbacks; those are authored separately. "
                "Refer to visual assets conceptually (never leak asset IDs), name any Slack channel explicitly, "
                "remember the player has no terminal, and use full participant names.\n"
                "Set the conflict up without settling it: the instruction must not state, imply or foreshadow how the round resolves, "
                "and must not present any participant or source as obviously right. The decision belongs to the evidence."
            ),
        },
        "conflict_layer3_round_mocks": {
            "required": ["mock_subagents"],
            "instruction": (
                "Given the finalized round instruction, generate ONLY the round-level mock subagent callbacks "
                "(agent_role, mock_response, capabilities, private_knowledge, virtual_duration_seconds, state_effects). "
                "Every 'state_effects' entry MUST be a valid, executable OpenClaw command string, never prose."
            ),
        },
        "conflict_layer3_round_updates": {
            "required": ["updates"],
            "instruction": (
                "Generate asynchronous background updates triggering during this round, including Slack/Teams dialog logs and file system modifications."
            ),
        },
        "conflict_layer3_round_assets": {
            "required": ["assets"],
            "instruction": (
                "Plan new visual telemetry assets (charts/images) introduced during this round, incorporating detailed version timestamps.\n"
                "'caption' and 'ocr_text' are the ONLY text a player can read off these assets, so they are where the evidence "
                "actually lives: write them as a realistically dense readout -- many rows/readings of the same kind, with units -- "
                "not a one-line summary of what the chart means.\n"
                "When the round instruction names a decisive reading, that string MUST appear VERBATIM (character for character, "
                "including units, separators, case and brackets) in exactly ONE asset's 'caption' or 'ocr_text', and MUST NOT be "
                "highlighted, annotated, or labelled as the answer. Assets that represent the DISAGREEING sources must show "
                "different readings of the same kind.\n"
                "The asset carrying that reading MUST be readable in the round that needs it: its 'available_from_round' "
                "MUST be this round, never a later one. A reading the player cannot open yet is not evidence."
            ),
        },
        "conflict_layer3_round_resolution": {
            "required": ["decisive_anchor", "resolved_value"],
            "instruction": (
                "Identify the SINGLE decisive piece of evidence the coordinator must engage with to solve THIS round, "
                "using only the round's visible assets, new assets, and update file contents.\n"
                "Return 'decisive_anchor': a stable evidence anchor id in the form 'EVIDENCE:SHORT_TOKEN' (uppercase, no spaces) "
                "that identifies that decisive evidence; prefer reusing an anchor already listed on a visible asset's 'anchor_ids'.\n"
                "Return 'resolved_value': the exact concrete value/string that constitutes the correct conclusion for this round "
                "(e.g. the offending config value, the stale timestamp, the mismatched count). It MUST appear verbatim in a visible "
                "asset's ocr_text/caption or in a workspace file's content so the solution can cite it and a checker can assert it.\n"
                "ALWAYS return 'resolved_value_locus' (MANDATORY, never empty): where that value actually lives, as "
                "'asset:<asset_id>' when it is read from a visible asset's ocr_text/caption, 'file:<workspace_path>' when it is "
                "verbatim inside that workspace file's contents, or 'computed' when it must be derived and is not verbatim in any "
                "single source. Pick the SINGLE most accurate locus and prefer 'asset:<id>' so solving the round requires looking "
                "at the image; never claim 'file:' unless the value is literally in that file's bytes.\n"
                "ALWAYS return 'decision_outcome': a SHORT free-form verdict/action label (a few words, not a sentence) naming the "
                "correct way to resolve THIS round's conflict (e.g. 'roll back to v4.2', 'block the deploy'); it must be grounded in "
                "the decisive evidence and MUST be different from 'resolved_value'."
            ),
        },
        "conflict_layer3_round_outline": {
            "required": ["outline"],
            "instruction": (
                "Design a step-by-step natural language debugging playbook outline. "
                "For each step, specify the command purpose and what check/verification should be asserted."
            ),
        },
        "conflict_layer3_round_steps": {
            "required": ["steps"],
            "instruction": (
                "Translate the provided natural language outline steps into concrete command lines and structured evaluation checkers. "
                "Return a list of step dictionaries containing 'command' (string) and 'checker' (JSON dict or null)."
            ),
        },
        "conflict_layer3_round_commands": {
            "required": ["commands"],
            "instruction": (
                "Translate the provided natural language outline steps into concrete command lines ONLY. "
                "Do NOT design evaluation checkers; that is handled separately. "
                "Return a list of dictionaries containing 'step_number' (int) and 'command' (the exact command line string). "
                "The final command MUST be 'DONE'."
            ),
        },
        "conflict_layer3_round_checkers": {
            "required": ["fields"],
            "instruction": (
                "You are given a list of solution steps whose checker schema and deterministic fields have ALREADY been decided. "
                "Provide ONLY the missing semantic field for each listed step: a 'criterion' (a concise sentence describing the intent of a delegation/message) "
                "or a 'contains' (an expected substring proving a written file's success). "
                "Return a list under 'fields', each entry containing 'step_number' (int) and exactly one of 'criterion' or 'contains'."
            ),
        },
        "audit_instruction_checker": {
            "required": ["has_contradiction", "remediation"],
            "instruction": (
                "Audit the task instruction and evaluation checkers for contradictions based on 'allowed_tool_families'. "
                "For example, checking if 'command' or 'terminal' tool family is not in allowed_tool_families but terminal commands must be run, "
                "or if 'file' tool family is not allowed but files must be written. "
                "Output remediation to delegate forbidden actions to mock subagents if a contradiction is found."
            ),
        },
        "audit_anchor_discoverability": {
            "required": ["missing_anchors"],
            "instruction": (
                "Audit visible workspace files and visible visual assets to check if the expected anchors to cite are discoverable. "
                "An anchor is considered discoverable if its ID or core concept is present as a text string inside 'visible_files_content', "
                "OR if it represents a visual/chart check and matches one of the visual asset IDs or paths listed in 'visible_visual_assets'. "
                "The 'missing_anchors' field must be a list of dictionaries. Each dictionary must contain "
                "'anchor_id' (string, the missing anchor ID) and 'target_file' (string, path of the file where the anchor is missing)."
            ),
        },
        "audit_telemetry_alignment": {
            "required": ["has_mismatch", "corrections"],
            "instruction": (
                "Audit chart configs and workspace config/log files for numeric telemetry alignment. "
                "Correct any mismatch in routing target percentages or error rate numbers. "
                "The 'corrections' field must be a list of dictionaries. Each dictionary must contain "
                "'file_path' (string, path of the file containing the mismatch) and 'description' (string, details of the mismatch)."
            ),
        },
        "repair_instruction_contradiction": {
            "required": ["updated_instruction", "subagent_effects_to_add"],
            "instruction": (
                "Rewrite the instruction or subagent effects to resolve any tool restriction contradictions. "
                "If terminal commands are not in allowed_tool_families, remove terminal commands from player instruction, "
                "and delegate them to mock subagents (adding the corresponding command execution to the subagent's mock_response or state_effects)."
            ),
        },
        "audit_instruction_write_directive": {
            "required": ["directs_write"],
            "instruction": (
                "Audit whether a single round's player instruction already directs the player to WRITE/CREATE/SAVE a report, "
                "audit, summary, or memo FILE recording their findings and/or decision. This round is graded on the CONTENT of "
                "such a produced file, so delivering the resolution ONLY as a chat/Slack message, or merely investigating without "
                "writing a file, does NOT count. Return only {\"directs_write\": true} or {\"directs_write\": false}."
            ),
        },
        "repair_instruction_write_directive": {
            "required": ["updated_instruction"],
            "instruction": (
                "Rewrite a round's player instruction so that, in addition to everything it already asks, it EXPLICITLY directs "
                "the player to write a resolution report/audit/summary FILE recording both the decisive evidence and the final "
                "decision/verdict. Preserve all existing tasks, names, channels, and intent; integrate the directive naturally. "
                "NEVER mention a specific file path or filename (the location is the player's choice), NEVER expose asset IDs, and "
                "NEVER reference any future round. Return {\"updated_instruction\": \"...\"}."
            ),
        },
        "repair_missing_anchors": {
            "required": ["edited_content"],
            "instruction": (
                "Naturally integrate a missing citation anchor ID into the provided workspace file content."
            ),
        },
        "repair_value_leak": {
            "required": ["edited_text"],
            "instruction": (
                "Rewrite a mock subagent's text so it STOPS transcribing a decisive value that the player is "
                "supposed to discover for themselves by looking at a visual asset. You are given 'text' (the "
                "subagent's reply or one of its state_effect command strings), the 'leaked_value', and the "
                "'asset_hint' describing where the player is meant to read that value.\n"
                "The subagent must still do exactly the same job, in the same tone, with the same outcome -- only "
                "the VERBATIM value disappears. Refer to it indirectly instead: 'the passcode shown on the gate "
                "board', 'the value on the dashboard', 'the code you read off the panel'. Keep every other detail "
                "(names, channels, IDs, paths, timestamps, status words) byte-identical.\n"
                "If 'text' is a command string (it starts with a verb like 'file create', 'file append' or "
                "'message send'), return a STILL-VALID command string: preserve the command name and every flag, "
                "edit only the human-readable --content/--message payload, and keep using SINGLE quotes around "
                "argument values (never double quotes).\n"
                "The returned text MUST NOT contain the leaked value in any form -- not verbatim, not "
                "re-spaced, not re-cased, not split across punctuation. Return {\"edited_text\": \"...\"}."
            ),
        },
        "repair_telemetry_mismatches": {
            "required": ["edited_content"],
            "instruction": (
                "Edit workspace log/config content to perfectly match the visual chart telemetry numbers."
            ),
        },
        "repair_trajectory_simulation_failure": {
            "required": ["updated_trajectory"],
            "instruction": (
                "Fix the expert solution commands (trajectory) to successfully resolve the task and pass all checks. "
                "You are provided with the current trajectory, the command outputs/trace, and the failed evaluation checks. "
                "If the payload includes 'protocol_contract', one or more failing checks are protocol EFFECT checks: the "
                "trajectory's terminal command was rejected at runtime because its multi-step protocol prerequisites are "
                "missing, out of order, or use a mismatched identifier. For each contract entry, insert its "
                "'ordered_prerequisites' commands (in the given order) BEFORE the 'terminal_command', all carrying the SAME "
                "value for 'matching_arg' as the terminal command (add a generous '--ttl' where the prerequisite supports one "
                "so it does not expire); when a precondition has an 'all_ready' quantifier, ensure every one of those commands "
                "(e.g. each gate) is set to the required value before the terminal command. Do NOT invent a '--code' for pairing "
                "(an omitted code is accepted). Preserve every other step, and keep the round's original intent. "
                "Return the complete updated trajectory list ending with 'DONE'."
            ),
        },

        "layer0_narrative": {
            "required": ["summary", "core_conflict", "truth_anchor", "preference_rule"],
        },
        "layer1_workspace": {"required": ["files"], "file_fields": ["id", "path", "content"]},
        "layer2_sessions": {"required": ["sessions"]},
        "layer3_eval": {"required": ["rounds", "eval_items"]},
        "layer4_dynamic": {"required": ["updates"]},
        "layer0_narrative_core": {
            "required": [
                "summary",
                "core_conflict",
                "truth_anchor",
                "preference_rule",
                "canonical_decision_id",
                "canonical_rule_id",
                "status_timeline",
            ],
            "instruction": (
                "Fill narrative fields and short canonical evidence ids. "
                "Use ids like DECISION:GATED_LAUNCH_REQUIRED and "
                "RULE:NEWER_OWNER_EVIDENCE_OVERRIDES_STALE_SUMMARY. "
                "status_timeline entries should include from_round, anchor_id, and summary."
            ),
        },
        "layer1_workspace_outline": {
            "required": ["files"],
            "file_fields": ["id", "path", "purpose"],
            "instruction": (
                "Design a compact workspace outline from the seed and spec constraints. "
                "Return {\"files\":[...]} only. Each file entry must have id, path, "
                "and purpose. Do not include file content. Use short relative paths."
            ),
        },
        "layer1_workspace_file": {
            "required": ["content"],
            "instruction": (
                "Generate only the content for the specified file slot. Do not change id or path. "
                "When relevant, include the provided canonical_evidence_ids as exact text."
            ),
        },
        "layer2_session_outline": {
            "required": ["sessions"],
            "session_fields": ["session_id", "channel", "purpose", "messages"],
            "message_fields": ["role", "speaker", "message_purpose"],
            "instruction": (
                "Design a compact session/message outline from the seed, narrative, "
                "and workspace outline. Return {\"sessions\":[...]} only. Each session "
                "must include message purpose specs, not message text."
            ),
        },
        "layer2_session_messages": {
            "required": ["messages"],
            "instruction": (
                "Generate only compact JSON for exactly the specified session slot. "
                "Return {\"messages\":[...]} with 3-5 messages. Each message must have "
                "role, speaker, and text. Keep each text under 45 words. No markdown."
            ),
        },
        "layer2_session_message_text": {
            "required": ["text"],
            "instruction": (
                "Generate only one compact message text for the exact message slot. "
                "Return {\"text\":\"...\"}. Do not return a message list. "
                "Keep text under 35 words and cite short canonical ids when relevant."
            ),
        },
        "layer3_eval_round": {
            "required": ["question"],
            "optional": ["rationale_summary"],
            "instruction": (
                "Generate only the question text for exactly the specified round slot. "
                "Do not reveal the expected answer or canonical evidence ids in the question. "
                "You may include a private rationale_summary field."
            ),
        },
        "layer4_update_item": {
            "required": ["workspace_content", "messages"],
            "instruction": "Generate one update note content and compact session append messages for the specified update slot.",
        },
        "layer4_update_outline": {
            "required": ["updates"],
            "update_fields": ["update_id", "before_round", "purpose", "workspace_path", "session_id"],
            "instruction": (
                "Design a compact dynamic-update outline from the seed and prior layer "
                "outlines. Return {\"updates\":[...]} only. Do not include update note "
                "content or chat message text."
            ),
        },
        "layer4_update_workspace_content": {
            "required": ["content"],
            "instruction": (
                "Generate only the update workspace note body for this update slot. "
                "Return {\"content\":\"...\"}. Keep it under 120 words. "
                "Include the provided update_anchor_id as exact text."
            ),
        },
        "layer4_update_message_text": {
            "required": ["text"],
            "instruction": (
                "Generate only one chat update message text. Return {\"text\":\"...\"}. "
                "Keep it under 35 words. Include the update_anchor_id if it fits naturally."
            ),
        },
        "vlm_asset_detail": {
            "required": ["description", "caption", "ocr_text"],
            "optional": ["prompt", "chart_rows", "render_code", "rationale_summary"],
            "instruction": (
                "Generate compact multimodal asset content for one VLM asset slot. "
                "Return JSON with description, caption, ocr_text, optional prompt for natural images, "
                "optional chart_rows, and optional render_code containing matplotlib python script. "
                "Do not decide scoring anchors; use provided anchor_ids exactly when relevant. "
                "If the slot carries 'must_carry_verbatim', every string listed there is a reading "
                "the scenario is graded on and MUST reappear character-for-character in your caption "
                "or ocr_text -- never reworded, rounded, abbreviated or truncated -- and must stay "
                "buried among comparable readings rather than being highlighted."
            ),
        },
        "vlm_round_question": {
            "required": ["question"],
            "optional": ["rationale_summary"],
            "instruction": (
                "Generate only the user-facing question text for one PRO-VLM round. "
                "Do not reveal canonical evidence ids or expected answer strings. "
                "Mention required action and visible asset types."
            ),
        },
        "clean_vlm_prompt": {
            "required": ["cleaned_prompt"],
            "instruction": (
                "Clean the provided operational visual prompt by removing all literal evidence anchors "
                "or metadata tags like 'EVIDENCE:...' or 'Evidence anchors: ...'. "
                "Rewrite it into a clean, rich natural language description suitable for image generation, "
                "retaining all visual/graphic layout details but stripping meta-tags. "
                "Preserve verbatim any exact reading that is part of the depicted subject -- labels, "
                "nameplates, serial numbers, version strings, counts, or status words -- since these are "
                "real physical text in the scene, not metadata."
            ),
        },
        "generate_vlm_html_template": {
            "required": ["html_content"],
            "optional": ["renderer_description"],
            "instruction": (
                "Generate a fully styled, single-file HTML document simulating a professional tool interface "
                "(e.g., an IDE, cloud provider console, monitoring dashboard, database manager, chat app, terminal log) "
                "matching the background story of the scenario.\n\n"
                "Requirements:\n"
                "1. Render the specific data, charts, or source code described in 'asset_info', ensuring the expected "
                "'anchor_ids' are embedded visually in the content text or logs.\n"
                "2. Add high-fidelity visual noise: a sidebar folder tree explorer containing the files in 'workspace_files', "
                "active file tabs, username profile badges, environment selectors, fake status notifications, and running log panels.\n"
                "3. Style using Tailwind CSS via CDN and Chart.js if rendering graphs. Ensure all Chart.js instances set 'animation: false'.\n"
                "4. Constrain the page to a single viewport dimension of 1000px width by 620px height with 'overflow: hidden' on body.\n"
                "5. Do NOT print the 'caption' field text literally on the HTML page. Do NOT render any literal metadata tags, "
                "labels, or prompt instructions (such as 'Anchor: XYZ' or 'Evidence: XYZ') on the page. The anchor IDs must only "
                "appear naturally in the UI content itself (e.g., as a row ID, log string, or token value)."
            ),
        },
        "generate_vlm_layout_spec": {
            "required": ["ui_shell_type", "color_theme", "system_name", "primary_color", "visual_style", "style_details", "platform_context", "grid_layout", "common_header_html", "common_sidebar_html", "common_footer_html", "common_styles_css"],
            "instruction": (
                "Design a structural layout outline and brand style specification for a professional software tool interface "
                "matching the scenario theme.\n\n"
                "REALISM & INDUSTRY-STANDARD CONVENTIONS:\n"
                "Select style parameters, colors, themes, and shells that realistically match the platform name (platform_id) and domain. For example:\n"
                "- DevOps monitoring dashboards, cloud provider telemetry interfaces, network stats screens: Prefer 'monitoring_dashboard' or 'cloud_console' shell, dark theme, and metrics-aligned accent colors (emerald, cyan, blue, purple, amber) using datadog_dashboard, modern_clean, or terminal_tui styles.\n"
                "- Chat applications, Slack workspaces, support channels, client portals: Prefer 'chat_app' shell, flat_minimalist or modern_clean styles, and corporate branding colors (purple, indigo, teal, blue).\n"
                "- Code editors, IDE terminals: Prefer 'ide' shell, dark theme, monospace fonts, and standard editor layouts.\n"
                "- Databases, SQL managers, admin console panels: Prefer 'database_manager' or 'cloud_console' shell, clear grid layouts, and clean readable styles.\n"
                "- Email clients: Prefer 'email_client' shell, light theme, modern_clean or apple_developer style.\n"
                "Avoid using anachronistic retro styles (classic_mac_80s, retro_90s) or display effects (crt_scanlines, noise_overlay) for modern software platforms unless the platform name or scenario theme explicitly indicates legacy, vintage, or degraded hardware telemetry.\n\n"
                "Return a JSON object with:\n"
                "- 'ui_shell_type': 'ide' | 'cloud_console' | 'monitoring_dashboard' | 'chat_app' | 'database_manager' | 'email_client' | 'calendar_portal' | 'general_web_page'\n"
                "- 'color_theme': 'dark' | 'light'\n"
                "- 'system_name': a realistic operational brand name for the system (e.g. 'Kubernetes Node Telemetry', 'Stripe Billing API Admin')\n"
                "- 'primary_color': a primary UI accent color ('blue' | 'emerald' | 'amber' | 'rose' | 'purple' | 'indigo' | 'cyan' | 'teal' | 'orange' | 'fuchsia' | 'pink' | 'violet')\n"
                "- 'visual_style': the visual design style/era (" + _visual_style_enum + ")\n"
                "- 'style_details': a nested JSON dictionary specifying micro-style options to customize the platform's visual design:\n"
                "  * 'font_family': 'monospace' | 'serif' | 'sans-serif-retro' | 'sans-serif-modern' | 'pixel'\n"
                "  * 'font_size_scale': 'compact' | 'coarse' | 'default' (compact reduces font sizes and panel spacing for a high-density operational grid)\n"
                "  * 'border_style': 'beveled_outset' | 'beveled_inset' | 'solid_thick' | 'solid_thin' | 'none'\n"
                "  * 'border_radius': 'none' | 'rounded-sm' | 'rounded-lg' | 'rounded-full'\n"
                "  * 'shadow_type': 'none' | 'hard-retro' | 'soft-modern' | 'neon-glow'\n"
                "  * 'header_style': 'solid' | 'gradient' | 'classic-title-bar' | 'glossy'\n"
                "  * 'display_effect': 'none' | 'crt_scanlines' | 'noise_overlay'\n"
                "    Note: 'none' is the default and should be used for most modern UIs (modern_clean, apple_developer, vercel_minimalist, datadog_dashboard, flat_minimalist, glassmorphism_3d). Only select 'crt_scanlines' or 'noise_overlay' if the visual_style is a retro style (retro_90s, classic_mac_80s, terminal_tui) or if the scenario theme specifically indicates degraded legacy system displays.\n"
                "  * 'divider_style': 'dashed' | 'dotted' | 'solid' | 'double' | 'ascii'\n"
                "  * 'background_texture': 'solid' | 'grid_lines' | 'dots'\n"
                "  * 'background_css_declaration': a valid, complete CSS background property string (e.g. 'background-color: #f8fafc;' or 'background: radial-gradient(...), #f8fafc;'). You must generate this string according to the color_theme, primary_color, and visual_style. It will be used to enforce background consistency across all pages on this platform.\n"
                "  * 'custom_css_rules': a list of 2-3 specific micro-styling directions describing how borders, cards, or active states are colored.\n"
                "- 'platform_context': a JSON dictionary containing 3-5 platform-specific operational context keys that must remain identical across all screens or logs on this platform (e.g., for 'ide': {'project_name': '...', 'current_branch': '...', 'active_developer': '...'}; for 'database_manager': {'db_instance_id': '...', 'engine_version': '...', 'login_user': '...'}; for 'chat_app': {'workspace_domain': '...', 'channel_name': '...', 'user_handle': '...'}; for 'monitoring_dashboard': {'node_hostname': '...', 'cluster_region': '...', 'operator_email': '...'}). Make these values highly realistic and tailored to the scenario theme.\n"
                "- 'grid_layout': outline of main sections (header, sidebar, main_panel, footer_panel).\n"
                "- 'common_header_html': complete HTML string of the header bar container (e.g. <header class='...'>...</header>) displaying system_name, node status, and relevant platform_context metadata. Leave empty/blank if grid_layout header is false.\n"
                "- 'common_sidebar_html': complete HTML string of the left navigation/explorer sidebar container (e.g. <aside class='...'>...</aside>) listing realistic files explorer or active channel items corresponding to the platform context (if workspace_files is provided in the input payload, you MUST list those exact files in the sidebar file explorer structure). Leave empty/blank if grid_layout sidebar is false. CRITICAL OVERFLOW SAFETY: To prevent text labels from bleeding or overlapping with adjacent panels, all text elements inside the sidebar representing file paths/names or channel names MUST use the Tailwind 'truncate' class (or equivalent CSS text-overflow ellipsis rules). If using flex layouts for sidebar list items, apply 'min-w-0' to the text elements parent wrapper to ensure truncation works correctly. Long names must never cross the sidebar's right boundary line.\n"
                "- 'common_footer_html': complete HTML string of the bottom status bar or command line prompt container (e.g. <footer class='...'>...</footer>). Leave empty/blank if grid_layout footer is false.\n"
                "- 'common_styles_css': standard CSS rules (exclude <style> tags) defining fonts import, scrollbar details, animations, or border accents used by the layout panels.\n\n"
                "The payload provides 'suggested_theme', 'suggested_shell_type', 'suggested_primary_color', 'suggested_visual_style', and 'suggested_style_details' to encourage style variety across different generated scenario versions. These are sampled together as one coherent design language for the suggested_visual_style, so prefer using them as a set to maintain visual diversity. In particular, 'suggested_style_details' is a coherent baseline token bundle for the suggested_visual_style: use it as the starting point for your 'style_details' and only change an individual token if it would be incoherent with the visual_style/shell you finally choose (you still MUST supply the 'background_css_declaration' token yourself, since it is not part of the suggestion). However, if a suggestion is highly unrealistic or incoherent for the platform type (e.g. an email client shell suggested for a Kubernetes cluster dashboard), you MUST override it and select a realistic, industry-standard alternative -- and when you override the visual_style, re-derive coherent style_details for the style you chose rather than keeping the suggested tokens. If suggested_display_effect is 'none', you MUST set display_effect to 'none' in style_details. If 'workspace_files' is provided, use them to design the sidebar list items."
            ),
        },
        "generate_vlm_redundant_noise": {
            "required": ["sidebar_content", "terminal_logs", "alerts_and_metrics"],
            "instruction": (
                "Generate domain-specific visual noise and redundant text to populate a UI layout.\n\n"
                "Requirements:\n"
                "1. Sidebar content should reference some files from 'workspace_files'.\n"
                "2. Terminal logs must show realistic system messages or commands.\n"
                "3. Alerts and metrics should match the status of the scenario (e.g., system warnings, user avatar details).\n"
                "4. If 'platform_context' is provided in the payload, you must strictly align any relevant variables (like project name, hostname, username, or branch) to match the values defined in 'platform_context' exactly. Do not invent contradictory user/project details.\n"
                "Do NOT include the target evidence anchor IDs in these redundant sections."
            ),
        },
        "compose_vlm_html": {
            "required": ["html_content"],
            "instruction": (
                "Compile the structured UI layout spec, redundant noise content, and the core "
                "evidence asset described in 'asset_info' into a single responsive HTML page.\n\n"
                "Requirements:\n"
                "1. Render the main evidence chart or data cleanly in the main panel using Tailwind CSS.\n"
                "2. Embed the expected 'anchor_ids' naturally within the main panel or charts.\n"
                "3. MANDATORY REUSE OF PRE-DESIGNED TEMPLATES: If 'common_header_html', 'common_sidebar_html', or 'common_footer_html' are present in 'layout_spec', you MUST reuse them verbatim in the final page body. Do NOT alter their HTML tag structure, CSS classes, attributes, or text content. Place them exactly as defined in the layout spec. Only fill the terminal console logs container with the logs from 'noise_spec.terminal_logs'.\n"
                "4. MANDATORY REUSE OF STYLES: If 'common_styles_css' is present in 'layout_spec', you MUST include its rules verbatim inside the page's <style> block.\n"
                "5. Force viewport dimensions to be responsive using 100vw width by 100vh height with overflow hidden on body.\n"
                "6. Do NOT write literal metadata instructions or captions on the screen.\n"
                "7. Strictly apply CSS/Tailwind styling matching the theme ('color_theme'), UI shell ('ui_shell_type'), and visual era specifications in 'layout_spec':\n"
                "   - For 'light' theme: use white or light gray backgrounds (#ffffff, bg-white, bg-slate-50) and dark text (text-slate-800, text-gray-900) for all panels (EXCEPT when using 'glassmorphism_3d', where panels MUST remain semi-transparent).\n"
                "   - For 'dark' theme: use dark slate/gray backgrounds (#1e1e1e, bg-slate-900) and light text (e.g. text-slate-300, text-gray-200, text-white, or very light shades of the primary_color like text-violet-200 / text-violet-300; EXCEPT when using 'glassmorphism_3d'). CRITICAL: Do NOT style regular body text, logs, or list items using dark primary color classes (like text-violet-500, text-violet-600, text-emerald-600) on dark backgrounds, as this results in unreadable, blacked-out text.\n"
                "   - Adhere strictly to the micro-styling values defined in 'style_details':\n"
                "     * Font: Apply 'font_family' (monospace/pixel -> font-mono; serif -> Georgia/serif; sans-serif-retro -> Arial/system-ui) and adjust font size/padding density matching 'font_size_scale' (compact -> text-xs, space-y-1, p-2; coarse -> text-lg, p-6).\n"
                "     * Borders & Corners: Set 'border_radius' (none -> rounded-none; rounded-sm/rounded-lg -> matching shapes) and border outlines matching 'border_style' (beveled_outset -> double bevel raised effect with light top-left and dark bottom-right borders; beveled_inset -> sunken field effect for logs/inputs; solid_thick -> 2px thick dark borders).\n"
                "     * Shadows & Accents: Apply 'shadow_type' (none -> shadow-none; hard-retro -> thick solid drop shadow offsets; neon-glow -> box-shadow glow using the selected 'primary_color'; soft-modern -> a soft, dark neutral ambient shadow like rgba(15, 23, 42, 0.08) in light theme or rgba(0, 0, 0, 0.4) in dark theme to provide depth separation).\n"
                "     * Header Layouts: Render 'header_style' (classic-title-bar -> flat solid bar using the selected primary color; gradient -> linear gradients; glossy -> translucent backing glass with linear specular highlights).\n"
                "     * Special Display Effects: If 'display_effect' is 'crt_scanlines', inject a repeating linear-gradient overlay pattern on the document body to simulate CRT phosphor scanlines.\n"
                "     * Background Canvas: You MUST apply the exact CSS declaration in 'background_css_declaration' (found in style_details) as the CSS background/background-color property on the document body, without any modifications. (For glassmorphism, do NOT apply solid body class background colors like bg-slate-50 or bg-white that would block the gradient mesh; keep the body background styled strictly using the declaration values).\n"
                "     * Grid Textures: If 'background_texture' is 'dots', add a repeating radial gradient grid dot pattern; if 'grid_lines', add clean technical blueprint grid lines.\n"
                "     * Dividers: Draw panel dividers conforming to 'divider_style' (dashed/dotted/double borders, or ascii-based block lines like +-----+ or | in terminal TUIs).\n"
                "   - Design language for the selected 'visual_style': the payload provides a 'design_guidance' object containing 'visual_style', a short 'aesthetic_descriptor', and a detailed 'render_guidance'. This is the authoritative, SSoT-sourced design language for the platform's selected style. You MUST faithfully reproduce the look described in 'design_guidance.render_guidance' (it corresponds exactly to the selected 'visual_style'), combining it with the exact 'style_details' tokens specified above. Treat 'render_guidance' as authoritative for panel treatment, borders, typography, shadows, background canvas, and any style-specific layout rules it states (e.g. removing the left sidebar, adding a top tab bar, or window controls). Do NOT invent a different look for the style name; if 'design_guidance' is absent, fall back to a clean modern SaaS dashboard (rounded cards, soft ambient shadows, thin light-gray borders, modern sans-serif type).\n"
                "   - Accent Elements: Strictly use the specified 'primary_color' from layout_spec (e.g., 'blue', 'emerald', 'amber', 'rose', 'purple', 'indigo', 'cyan', 'teal', 'orange', 'fuchsia', 'pink', 'violet') for all styling accents. You MUST map the primary_color name directly to Tailwind's exact corresponding color palette (e.g. if 'violet', use 'violet' classes like text-violet-400 or bg-violet-600; if 'indigo', use 'indigo' classes). Do NOT mix or substitute adjacent colors (e.g., do not use purple or fuchsia classes for violet, and do not use pink classes for fuchsia) to ensure color scheme consistency across assets. Style the active sidebar tab text/background (e.g., `bg-emerald-100 text-emerald-800` in light theme, or dynamic border highlights in dark theme), primary action buttons (e.g., `bg-rose-600 hover:bg-rose-700`), active states, or header border-b highlights. Do NOT default everything to generic blue or gray.\n"
                "8. If 'platform_context' is provided in layout_spec or the payload, you must strictly print and use those exact key-value pairs (e.g. displaying the exact project name, current branch, active username, hostnames, etc.) in the UI headers, user profile cards, or log prefixes rather than inventing conflicting random strings.\n"
                "9. STRICT INTERNAL CONTENT CONSISTENCY: Ensure that any explanatory descriptions, summaries, or captions printed inside the HTML panels are 100% consistent with any raw database fields, WHOIS details, log strings, or metrics data shown in the tables or logs. For example, if a raw WHOIS database dump displays 'OrgName: Iliad/Free SAS', the surrounding text overview must NOT mention a different provider like 'Orange S.A.' to avoid conflicting facts.\n"
                "10. CRITICAL RESPONSIVENESS & GRID LAYOUT CONSISTENCY: Use fluid Flexbox or Grid layouts for all panels. NEVER hardcode absolute pixel widths or heights (e.g. do NOT set style='width: 800px; height: 600px;') on main panel containers or chart divs. Use percentage classes (e.g. w-full, h-full) or responsive viewport constraints. You MUST use a consistent sidebar width across the same platform (default: class 'w-64' for standard, or 'w-80' if compact). If layout_spec specifies a header, use class 'h-14 flex-none' for the header container. If layout_spec specifies a footer, use class 'h-12 flex-none' for the footer container. This guarantees all pages under the same platform align exactly in grid proportions.\n"
                "11. CHART.JS WRAPPING: If rendering a graph using Chart.js, you MUST wrap the `<canvas>` element inside a parent wrapper div styled with a relative position and controlled height (e.g., `<div class=\"relative w-full h-[350px]\"><canvas id=\"...\"></canvas></div>`). In the Chart.js javascript options, you MUST set 'responsive: true' and 'maintainAspectRatio: false' to prevent the canvas from overflowing its container.\n"
                "12. OVERFLOW & CLIPPING PROTECTION: Add generous padding (e.g., class 'p-6') inside panel cards to avoid text or graphics clipping against border lines or overlapping with sidebar links. For large logs or text files, wrap them in scrollable containers (e.g., class 'overflow-y-auto max-h-[300px]').\n"
                "13. DIGITAL DIAGRAM RENDERING: If 'asset_info.kind' is 'diagram', the main panel MUST contain the schematic itself drawn as inline SVG (preferred) or HTML/CSS boxes-and-connectors -- NOT a chart, table, photo, screenshot, or hand-drawn image. Draw the nodes/blocks with their labels, the directional edges/arrows between them, a small legend, and clearly highlight the fault/active element (e.g. a red stroke, red fill, or a badge). Embed the 'anchor_ids' verbatim as node labels or badges on the diagram. Lay it out cleanly (a topology/flow/floor-plan grid) and style strokes, node fills, and label fonts to match the platform 'color_theme', 'primary_color', and 'visual_style' so it reads as a native in-UI diagram module."
            ),
        },
    }
    return contracts.get(layer, {"required": []})


def _validate_parsed_data(layer: str, data: dict[str, Any]) -> None:
    if data == {"ok": True}:
        return
    contract = _layer_contract(layer)
    required_keys = contract.get("required", [])
    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        raise ScenarioLLMError(
            f"Response missing required contract keys: {missing_keys}. "
            f"Expected: {required_keys}. Got: {list(data.keys())}."
        )


def _extract_responses_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    parts: list[str] = []
    for item in response.get("output", []) or []:
        for content in item.get("content", []) or []:
            text = content.get("text")
            if isinstance(text, str):
                parts.append(text)
    if parts:
        return "\n".join(parts)
    raise ScenarioLLMError("Responses API payload contained no text output")


def _chat_token_limit_field(model: str) -> str:
    model_lc = (model or "").lower()
    if model_lc.startswith("gpt-5") or model_lc.startswith("o3"):
        return "max_completion_tokens"
    return "max_tokens"


def _split_curl_body_and_status(stdout: str) -> tuple[str, str | None]:
    if not stdout:
        return "", None
    body, sep, status = stdout.rpartition("\n")
    if sep and status.isdigit() and len(status) == 3:
        return body, status
    return stdout, None
