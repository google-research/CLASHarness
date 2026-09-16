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

"""Provider-native tool-call parsing + structured history (WS3-5).

Pure, provider-agnostic helpers that let :class:`train_eval.agents.LLMAgent`
run in *native tool-calling* mode while keeping a single canonical conversation
representation:

* **Parsing.** ``parse_*_tool_call`` extracts the first tool call out of each
  provider's raw response into a common :class:`ToolCall`.
* **Structured history.** A canonical conversation entry (produced by
  :func:`assistant_tool_turn` / :func:`tool_result_turn`) is rendered into each
  vendor's own request/history format by the ``to_*`` serializers, so a tool
  call and its result are replayed to the model the way that vendor expects
  ("typed structured history" / full assistant-turn preservation).

The module is intentionally free of provider SDK imports so it stays pure and
independently unit-testable (mirroring :mod:`train_eval.tool_schemas`). Gemini
``Content``/``Part`` objects are constructed in the agent from the neutral
descriptors returned by :func:`to_gemini_history_items`.

Canonical conversation entries understood by the serializers:

* a plain chat turn ``{"role": "user"|"assistant"|"system", "content": ...}``
  (passed through unchanged), or
* an assistant tool-call turn ``{"role": "assistant", "tool_call": {...}}``, or
* a tool-result turn ``{"role": "tool", "tool_result": {...}}``.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any


# --------------------------------------------------------------------------- #
# Canonical tool call
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ToolCall:
    """A single provider tool call, normalized across vendors.

    ``arguments`` is always a decoded ``dict``. ``raw_arguments`` preserves the
    exact JSON string the provider sent (OpenAI-family returns arguments as a
    string) so it can be replayed verbatim in history; it is synthesized from
    ``arguments`` when the provider delivered a structured object instead
    (Anthropic ``input`` / Gemini ``args``). ``id`` is the provider's call
    correlation id (``None`` for Gemini, which correlates by function name).

    ``thought_signature`` is a base64-encoded string of the opaque signature
    Gemini 2.5+ thinking models attach to each ``functionCall`` part. It is
    ``None`` for every other provider (and for Gemini responses without a
    signature). When present it MUST be replayed on the corresponding history
    part, otherwise Gemini rejects multi-step turns with ``400
    INVALID_ARGUMENT: Function call is missing a thought_signature``. It is
    stored base64-encoded (not raw ``bytes``) so canonical turns stay
    JSON-serializable for snapshots and trajectory sidecars.

    ``reasoning`` is the provider's per-turn state that must be threaded back
    into the next tool-call request so the model keeps a continuous chain of
    thought across steps (the Gemini ``thought_signature`` plays the same role
    for that provider). It is a JSON-serializable list of provider-native items
    replayed verbatim, in order, on the assistant tool turn:

    * Anthropic: the content blocks that precede the ``tool_use`` block, e.g.
      ``[{"type": "thinking", "thinking": "...", "signature": "..."}]``. When
      extended thinking is enabled these MUST be replayed before the paired
      ``tool_use`` block or the API rejects the turn (HTTP 400). Visible
      ``text`` blocks are included too -- under ``tool_choice=auto`` Claude may
      narrate before calling a tool, and that narration is part of the turn.
    * OpenAI Responses: the ``reasoning`` output items (carrying
      ``encrypted_content`` under ``store=false``) that precede the
      ``function_call`` item; replayed before it so the reasoning is not
      dropped between steps.

    It is ``None`` for providers/turns without threaded reasoning (Gemini, and
    any turn produced with reasoning threading disabled).

    ``dropped_parallel_calls`` names the additional tool calls a provider emitted
    in the same response beyond the first. The benchmark executes exactly one
    action per step, so the extras cannot be run and are not replayed into
    history; this field exists purely so that discarding them is observable
    rather than silent. Empty for the overwhelmingly common single-call case.
    """

    name: str
    arguments: dict = field(default_factory=dict)
    id: str | None = None
    raw_arguments: str | None = None
    thought_signature: str | None = None
    reasoning: Any | None = None
    dropped_parallel_calls: list[str] = field(default_factory=list)

    def arguments_json(self) -> str:
        """The arguments as a JSON string (for OpenAI-style history replay)."""
        if self.raw_arguments is not None:
            return self.raw_arguments
        return json.dumps(self.arguments, ensure_ascii=False)


def _as_arguments(value: Any) -> tuple[dict, str | None]:
    """Normalize a provider arguments payload to ``(dict, raw_json_or_None)``."""
    if value is None:
        return {}, None
    if isinstance(value, str):
        raw = value
        try:
            parsed = json.loads(value) if value.strip() else {}
        except (ValueError, TypeError):
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}
        return parsed, raw
    if isinstance(value, dict):
        return dict(value), None
    return {}, None


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a dict or an SDK object (attribute) uniformly."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# --------------------------------------------------------------------------- #
# Parsers: raw provider response -> ToolCall | None
# --------------------------------------------------------------------------- #
def chat_reasoning_text(message: dict) -> str | None:
    """The turn's thinking text from a Chat Completions message, if any.

    Servers that split a ``<think>`` block out of ``content`` put it in a side
    field whose name is NOT standardized: vLLM's reasoning parsers currently
    emit ``reasoning``, while OpenAI's own naming (and older vLLM) uses
    ``reasoning_content``. Reading only one of them silently drops every
    thinking block the other kind of server produces, so both are accepted.
    """
    for key in ("reasoning", "reasoning_content"):
        value = (message or {}).get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def parse_openai_chat_tool_call(payload: dict) -> ToolCall | None:
    """Parse the first tool call from a Chat Completions response payload.

    Parallel calls beyond the first are recorded in ``dropped_parallel_calls``
    (see :class:`ToolCall`) rather than discarded silently.

    The turn's thinking text is captured onto ``ToolCall.reasoning`` as a plain
    ``str`` (the Responses and Anthropic paths store a ``list`` there instead;
    every renderer type-checks before using it, so the two shapes coexist).
    ``to_openai_chat_messages`` replays it on the next request.
    """
    try:
        message = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None
    tool_calls = (message or {}).get("tool_calls") or []
    chosen: dict | None = None
    extra_calls: list[str] = []
    for tc in tool_calls:
        if (tc or {}).get("type", "function") != "function":
            continue
        fn = tc.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        if chosen is None:
            chosen = tc
        else:
            extra_calls.append(name)
    if chosen is None:
        return None
    fn = chosen.get("function") or {}
    arguments, raw = _as_arguments(fn.get("arguments"))
    return ToolCall(
        name=fn["name"],
        arguments=arguments,
        id=chosen.get("id"),
        raw_arguments=raw,
        reasoning=chat_reasoning_text(message),
        dropped_parallel_calls=extra_calls,
    )


def parse_openai_responses_tool_call(payload: dict) -> ToolCall | None:
    """Parse the first ``function_call`` item from a Responses API payload.

    Any ``reasoning`` items that precede the ``function_call`` (present when the
    request opts into ``include=["reasoning.encrypted_content"]`` with
    ``store=false``) are captured verbatim so they can be replayed before the
    call on the next request, keeping reasoning continuous across forced steps.
    """
    reasoning: list[dict] = []
    chosen: dict | None = None
    # The model may emit PARALLEL calls. The benchmark executes exactly one
    # action per step, so only the first is kept -- but the extras are recorded
    # rather than discarded silently, because a step where the model planned N
    # actions and was credited with 1 otherwise looks identical in the logs to a
    # step where it only ever planned 1.
    extra_calls: list[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "reasoning":
            if chosen is None:
                reasoning.append(item)
            continue
        if item.get("type") != "function_call":
            continue
        name = item.get("name")
        if not name:
            continue
        if chosen is None:
            chosen = item
        else:
            extra_calls.append(name)
    if chosen is None:
        return None
    arguments, raw = _as_arguments(chosen.get("arguments"))
    return ToolCall(
        name=chosen["name"],
        arguments=arguments,
        id=chosen.get("call_id") or chosen.get("id"),
        raw_arguments=raw,
        reasoning=reasoning or None,
        dropped_parallel_calls=extra_calls,
    )


def _anthropic_reasoning_blocks(content: Any) -> list[dict] | None:
    """Collect the blocks that precede ``tool_use`` and must be replayed.

    Covers ``thinking`` / ``redacted_thinking`` and the visible ``text`` block:

    * Extended-thinking responses emit thinking blocks before the ``tool_use``
      block. They carry an opaque ``signature`` (or encrypted ``data``) and MUST
      be replayed verbatim on the assistant turn preceding a ``tool_result`` or
      the Messages API rejects the multi-step turn.
    * Under ``tool_choice=auto`` Claude may also narrate in a ``text`` block
      before calling the tool (a forced choice prefills the assistant message,
      so it cannot). That text was previously dropped, which silently deleted a
      per-step scratchpad from Claude's history that the other providers'
      histories did not lose -- an asymmetry working against Claude.

    Original block order is preserved rather than regrouped by type: it is
    exactly the order Claude produced, so the API's "thinking block first"
    requirement is satisfied by construction. Empty ``text`` blocks are skipped
    (the API rejects them). SDK block objects are normalized to plain
    JSON-serializable dicts in the shape the API accepts on input.
    """
    blocks: list[dict] = []
    for block in content or []:
        btype = _get(block, "type")
        if btype == "thinking":
            entry: dict = {"type": "thinking", "thinking": _get(block, "thinking") or ""}
            signature = _get(block, "signature")
            if signature is not None:
                entry["signature"] = signature
            blocks.append(entry)
        elif btype == "redacted_thinking":
            blocks.append({"type": "redacted_thinking", "data": _get(block, "data") or ""})
        elif btype == "text":
            text = _get(block, "text") or ""
            if text.strip():
                blocks.append({"type": "text", "text": text})
    return blocks or None


def parse_anthropic_tool_call(response: Any) -> ToolCall | None:
    """Parse the first ``tool_use`` block from an Anthropic Messages response.

    Anthropic emits parallel tool calls as several ``tool_use`` blocks in one
    assistant message. Only the first is executable here (one action per step);
    the rest are recorded in ``dropped_parallel_calls`` so the truncation is
    observable and comparable across providers.
    """
    content = _get(response, "content") or []
    reasoning = _anthropic_reasoning_blocks(content)
    chosen: Any | None = None
    extra_calls: list[str] = []
    for block in content:
        if _get(block, "type") != "tool_use":
            continue
        name = _get(block, "name")
        if not name:
            continue
        if chosen is None:
            chosen = block
        else:
            extra_calls.append(name)
    if chosen is None:
        return None
    arguments, raw = _as_arguments(_get(chosen, "input"))
    return ToolCall(
        name=_get(chosen, "name"),
        arguments=arguments,
        id=_get(chosen, "id"),
        raw_arguments=raw,
        reasoning=reasoning,
        dropped_parallel_calls=extra_calls,
    )


def parse_gemini_tool_call(response: Any) -> ToolCall | None:
    """Parse the first ``function_call`` part from a Gemini response object.

    Gemini emits parallel tool calls as several ``function_call`` parts inside
    one candidate's content; those beyond the first are recorded in
    ``dropped_parallel_calls``. Extra *candidates* are alternative samples
    rather than parallel calls, so they are deliberately not counted: only the
    first candidate that carries a call is inspected.
    """
    candidates = _get(response, "candidates") or []
    for candidate in candidates:
        content = _get(candidate, "content")
        parts = _get(content, "parts") or []
        chosen_part: Any | None = None
        chosen_fc: Any | None = None
        extra_calls: list[str] = []
        for part in parts:
            fc = _get(part, "function_call")
            if fc is None:
                continue
            name = _get(fc, "name")
            if not name:
                continue
            if chosen_fc is None:
                chosen_part, chosen_fc = part, fc
            else:
                extra_calls.append(name)
        if chosen_fc is None:
            continue
        args = _get(chosen_fc, "args")
        arguments, raw = _as_arguments(dict(args) if args else {})
        # Gemini 2.5+ thinking models attach an opaque ``thought_signature``
        # (bytes) to the functionCall part; it must be replayed on the
        # history part or multi-step turns get rejected. Store it b64 so the
        # canonical turn stays JSON-serializable.
        signature = _encode_signature(_get(chosen_part, "thought_signature"))
        # Gemini has no call id; results correlate by function name.
        return ToolCall(
            name=_get(chosen_fc, "name"),
            arguments=arguments,
            id=None,
            raw_arguments=raw,
            thought_signature=signature,
            dropped_parallel_calls=extra_calls,
        )
    return None


def _encode_signature(value: Any) -> str | None:
    """Normalize a Gemini ``thought_signature`` to a base64 string or ``None``."""
    if value is None:
        return None
    if isinstance(value, str):
        # Already a string (e.g. round-tripped canonical turn): pass through.
        return value
    if isinstance(value, (bytes, bytearray)):
        return base64.b64encode(bytes(value)).decode("ascii")
    return None


# --------------------------------------------------------------------------- #
# Canonical conversation entries
# --------------------------------------------------------------------------- #
def assistant_tool_turn(call: ToolCall) -> dict:
    """Canonical assistant turn recording an emitted tool call."""
    return {
        "role": "assistant",
        "tool_call": {
            "id": call.id,
            "name": call.name,
            "arguments": dict(call.arguments),
            "raw_arguments": call.raw_arguments,
            "thought_signature": call.thought_signature,
            "reasoning": call.reasoning,
        },
    }


def tool_result_turn(call: ToolCall, output: str) -> dict:
    """Canonical turn recording the workspace result of a tool call."""
    return {
        "role": "tool",
        "tool_result": {
            "id": call.id,
            "name": call.name,
            "output": output,
        },
    }


def _call_from_entry(entry: dict) -> ToolCall:
    tc = entry["tool_call"]
    return ToolCall(
        name=tc["name"],
        arguments=dict(tc.get("arguments") or {}),
        id=tc.get("id"),
        raw_arguments=tc.get("raw_arguments"),
        thought_signature=tc.get("thought_signature"),
        reasoning=tc.get("reasoning"),
    )


def _is_tool_call_entry(entry: dict) -> bool:
    return isinstance(entry, dict) and "tool_call" in entry


def _is_tool_result_entry(entry: dict) -> bool:
    return isinstance(entry, dict) and "tool_result" in entry


# --------------------------------------------------------------------------- #
# Serializers: canonical conversation -> provider request messages
# --------------------------------------------------------------------------- #
def to_openai_chat_messages(
    conversation: list[dict], *, replay_reasoning: bool = False
) -> list[dict]:
    """Render a canonical conversation into Chat Completions messages.

    With ``replay_reasoning`` the thinking text captured on each assistant tool
    turn is re-inserted as an inline ``<think>...</think>`` prefix of that
    turn's ``content``, so the model keeps a continuous chain of thought across
    tool steps instead of seeing its own past reasoning vanish.

    Inlining looks redundant -- the message could just carry a
    ``reasoning_content`` field, which is what the Qwen chat template reads --
    but that does not survive the trip: vLLM validates request messages against
    the OpenAI schema and drops unknown fields BEFORE applying the template, so
    a side field renders as an empty ``<think></think>``. Verified against the
    live server: the inline form round-trips, the side field does not. The
    template's own fallback branch splits an inline block back out of
    ``content``, so this reconstructs exactly the shape it expects.

    Off by default: the tag text is only inert for models whose template strips
    it, so it must be opted into per model rather than applied to every
    OpenAI-compatible endpoint.
    """
    out: list[dict] = []
    for entry in conversation:
        if _is_tool_call_entry(entry):
            call = _call_from_entry(entry)
            # Only the chat path stores a str here; the Responses path stores a
            # list of native items, which this renderer must not touch.
            content = None
            if replay_reasoning and isinstance(call.reasoning, str):
                thinking = call.reasoning.strip()
                if thinking:
                    content = f"<think>\n{thinking}\n</think>\n\n"
            out.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": call.id or "",
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments_json(),
                            },
                        }
                    ],
                }
            )
        elif _is_tool_result_entry(entry):
            tr = entry["tool_result"]
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": tr.get("id") or "",
                    "content": tr.get("output", ""),
                }
            )
        else:
            out.append(entry)
    return out


def to_openai_responses_items(conversation: list[dict]) -> list[dict]:
    """Render tool turns into Responses API ``input`` items.

    Plain chat turns are returned unchanged; the agent's existing
    ``_to_responses_input`` converts their multimodal content parts afterwards.
    """
    out: list[dict] = []
    for entry in conversation:
        if _is_tool_call_entry(entry):
            call = _call_from_entry(entry)
            # Replay the reasoning items (with encrypted_content) before the call
            # so the model's chain of thought is not dropped between tool steps.
            if isinstance(call.reasoning, list):
                out.extend(item for item in call.reasoning if isinstance(item, dict))
            out.append(
                {
                    "type": "function_call",
                    "call_id": call.id or "",
                    "name": call.name,
                    "arguments": call.arguments_json(),
                }
            )
        elif _is_tool_result_entry(entry):
            tr = entry["tool_result"]
            out.append(
                {
                    "type": "function_call_output",
                    "call_id": tr.get("id") or "",
                    "output": tr.get("output", ""),
                }
            )
        else:
            out.append(entry)
    return out


def to_anthropic_messages(conversation: list[dict]) -> list[dict]:
    """Render a canonical conversation into Anthropic Messages ``messages``.

    Tool calls become an assistant ``tool_use`` content block; results become a
    user ``tool_result`` content block (Anthropic requires tool results to be
    delivered in the user turn).
    """
    out: list[dict] = []
    for entry in conversation:
        if _is_tool_call_entry(entry):
            call = _call_from_entry(entry)
            content: list[dict] = []
            # Extended thinking requires the thinking/redacted_thinking blocks to
            # be replayed before the tool_use block they preceded; otherwise the
            # Messages API rejects the turn (HTTP 400). Any visible text block
            # from the same turn rides along in the same original order.
            if isinstance(call.reasoning, list):
                content.extend(
                    block for block in call.reasoning if isinstance(block, dict)
                )
            content.append(
                {
                    "type": "tool_use",
                    "id": call.id or "",
                    "name": call.name,
                    "input": dict(call.arguments),
                }
            )
            out.append({"role": "assistant", "content": content})
        elif _is_tool_result_entry(entry):
            tr = entry["tool_result"]
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tr.get("id") or "",
                            "content": tr.get("output", ""),
                        }
                    ],
                }
            )
        else:
            out.append(entry)
    return out


def to_gemini_history_items(conversation: list[dict]) -> list[dict]:
    """Render a canonical conversation into neutral Gemini history descriptors.

    The agent turns each descriptor into ``types.Content``/``types.Part``
    (this module stays free of the ``google.genai`` dependency). Descriptor
    shapes:

    * ``{"kind": "message", "entry": <plain chat turn>}`` -- convert as today.
    * ``{"kind": "function_call", "name": str, "args": dict}`` -- model turn.
      Carries an extra ``"thought_signature": <base64 str>`` only when the
      original Gemini response supplied one (so the agent can replay it on the
      reconstructed part).
    * ``{"kind": "function_response", "name": str, "output": str}`` -- user turn.
    """
    out: list[dict] = []
    for entry in conversation:
        if _is_tool_call_entry(entry):
            call = _call_from_entry(entry)
            item = {"kind": "function_call", "name": call.name, "args": dict(call.arguments)}
            # Only surface the key when present so non-Gemini / unsigned turns
            # keep the minimal descriptor shape.
            if call.thought_signature is not None:
                item["thought_signature"] = call.thought_signature
            out.append(item)
        elif _is_tool_result_entry(entry):
            tr = entry["tool_result"]
            out.append(
                {
                    "kind": "function_response",
                    "name": tr.get("name", ""),
                    "output": tr.get("output", ""),
                }
            )
        else:
            out.append({"kind": "message", "entry": entry})
    return out
