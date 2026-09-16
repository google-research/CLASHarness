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

"""Unit tests for the provider-native tool-call layer (WS3-5).

Pure tests: no provider I/O. They cover (a) parsing a single tool call out of
each vendor's raw response shape, and (b) serializing a canonical tool-call /
tool-result conversation into each vendor's request/history format.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import train_eval.tool_calling as tc


# --------------------------------------------------------------------------- #
# ToolCall
# --------------------------------------------------------------------------- #
def test_arguments_json_prefers_raw_then_falls_back():
    with_raw = tc.ToolCall(name="x", arguments={"a": 1}, raw_arguments='{"a": 1}')
    assert with_raw.arguments_json() == '{"a": 1}'

    without_raw = tc.ToolCall(name="x", arguments={"a": 1})
    assert without_raw.arguments_json() == '{"a": 1}'


# --------------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------------- #
def test_parse_openai_chat_tool_call():
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "email_send",
                                "arguments": '{"to": "a@x.com"}',
                            },
                        }
                    ],
                }
            }
        ]
    }
    call = tc.parse_openai_chat_tool_call(payload)
    assert call is not None
    assert call.name == "email_send"
    assert call.arguments == {"to": "a@x.com"}
    assert call.id == "call_1"
    assert call.raw_arguments == '{"to": "a@x.com"}'


def test_parse_openai_chat_no_tool_call_returns_none():
    assert tc.parse_openai_chat_tool_call({"choices": [{"message": {"content": "hi"}}]}) is None
    assert tc.parse_openai_chat_tool_call({}) is None


def test_parse_openai_responses_tool_call_prefers_call_id():
    payload = {
        "output": [
            {"type": "reasoning", "id": "rs_1"},
            {
                "type": "function_call",
                "id": "fc_item_1",
                "call_id": "call_9",
                "name": "system_status",
                "arguments": "{}",
            },
        ]
    }
    call = tc.parse_openai_responses_tool_call(payload)
    assert call is not None
    assert call.name == "system_status"
    assert call.arguments == {}
    assert call.id == "call_9"  # call_id wins over item id


def test_parse_anthropic_tool_call_from_sdk_objects():
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="let me do that"),
            SimpleNamespace(
                type="tool_use",
                id="toolu_1",
                name="calendar_add_event",
                input={"title": "Sync", "start": "2026-01-01"},
            ),
        ]
    )
    call = tc.parse_anthropic_tool_call(response)
    assert call is not None
    assert call.name == "calendar_add_event"
    assert call.arguments == {"title": "Sync", "start": "2026-01-01"}
    assert call.id == "toolu_1"
    # Anthropic delivers a structured object, so raw JSON is synthesized on demand.
    assert call.raw_arguments is None
    assert call.arguments_json() == '{"title": "Sync", "start": "2026-01-01"}'


def test_parse_gemini_tool_call_from_sdk_objects():
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(text="thinking", function_call=None),
                        SimpleNamespace(
                            function_call=SimpleNamespace(
                                name="email_mark", args={"id": "m1", "flag": "read"}
                            )
                        ),
                    ]
                )
            )
        ]
    )
    call = tc.parse_gemini_tool_call(response)
    assert call is not None
    assert call.name == "email_mark"
    assert call.arguments == {"id": "m1", "flag": "read"}
    assert call.id is None
    # This response part carries no signature attribute -> None (not every model
    # or turn emits one; only Gemini 2.5+ thinking functionCall parts do).
    assert call.thought_signature is None


def test_parse_gemini_no_function_call_returns_none():
    response = SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text="hi", function_call=None)]))]
    )
    assert tc.parse_gemini_tool_call(response) is None


def test_parse_malformed_arguments_degrade_to_empty_dict():
    payload = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"id": "c", "type": "function", "function": {"name": "x", "arguments": "not json"}}
                    ]
                }
            }
        ]
    }
    call = tc.parse_openai_chat_tool_call(payload)
    assert call is not None
    assert call.arguments == {}
    assert call.raw_arguments == "not json"  # preserved verbatim


# --------------------------------------------------------------------------- #
# Canonical conversation + serializers
# --------------------------------------------------------------------------- #
def _sample_conversation() -> list[dict]:
    call = tc.ToolCall(name="email_send", arguments={"to": "a@x.com"}, id="call_1", raw_arguments='{"to": "a@x.com"}')
    return [
        {"role": "user", "content": "do the task"},
        tc.assistant_tool_turn(call),
        tc.tool_result_turn(call, "[STDOUT]\nok\n[EXIT CODE] 0"),
        {"role": "user", "content": "next?"},
    ]


def test_openai_chat_serialization_roundtrips_shape():
    msgs = tc.to_openai_chat_messages(_sample_conversation())
    assert msgs[0] == {"role": "user", "content": "do the task"}

    assistant = msgs[1]
    assert assistant["role"] == "assistant"
    assert assistant["content"] is None
    (call,) = assistant["tool_calls"]
    assert call["id"] == "call_1"
    assert call["type"] == "function"
    assert call["function"] == {"name": "email_send", "arguments": '{"to": "a@x.com"}'}

    result = msgs[2]
    assert result["role"] == "tool"
    assert result["tool_call_id"] == "call_1"
    assert result["content"] == "[STDOUT]\nok\n[EXIT CODE] 0"

    assert msgs[3] == {"role": "user", "content": "next?"}


def test_openai_responses_serialization_shape():
    items = tc.to_openai_responses_items(_sample_conversation())
    fc = items[1]
    assert fc == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "email_send",
        "arguments": '{"to": "a@x.com"}',
    }
    fco = items[2]
    assert fco == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "[STDOUT]\nok\n[EXIT CODE] 0",
    }
    # Plain turns pass through untouched (agent converts their content parts).
    assert items[0] == {"role": "user", "content": "do the task"}


def test_anthropic_serialization_shape():
    msgs = tc.to_anthropic_messages(_sample_conversation())
    assistant = msgs[1]
    assert assistant["role"] == "assistant"
    (block,) = assistant["content"]
    assert block == {
        "type": "tool_use",
        "id": "call_1",
        "name": "email_send",
        "input": {"to": "a@x.com"},
    }
    result = msgs[2]
    assert result["role"] == "user"  # Anthropic requires tool results in a user turn
    (rblock,) = result["content"]
    assert rblock == {
        "type": "tool_result",
        "tool_use_id": "call_1",
        "content": "[STDOUT]\nok\n[EXIT CODE] 0",
    }


def test_gemini_history_descriptors():
    items = tc.to_gemini_history_items(_sample_conversation())
    kinds = [it["kind"] for it in items]
    assert kinds == ["message", "function_call", "function_response", "message"]
    assert items[1] == {"kind": "function_call", "name": "email_send", "args": {"to": "a@x.com"}}
    assert items[2] == {
        "kind": "function_response",
        "name": "email_send",
        "output": "[STDOUT]\nok\n[EXIT CODE] 0",
    }
    assert items[0] == {"kind": "message", "entry": {"role": "user", "content": "do the task"}}


def test_serializers_leave_plain_conversation_untouched():
    plain = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]
    assert tc.to_openai_chat_messages(plain) == plain
    assert tc.to_openai_responses_items(plain) == plain
    assert tc.to_anthropic_messages(plain) == plain


def test_gemini_id_absent_uses_name_for_result_correlation():
    # A Gemini-origin call (id=None) still round-trips through the canonical
    # turns because results correlate by function name, not id.
    call = tc.ToolCall(name="system_status", arguments={}, id=None)
    conv = [tc.assistant_tool_turn(call), tc.tool_result_turn(call, "ok")]
    items = tc.to_gemini_history_items(conv)
    assert items[0]["name"] == "system_status"
    assert items[1]["name"] == "system_status"


# --------------------------------------------------------------------------- #
# Gemini thought_signature (2.5+ thinking models)
# --------------------------------------------------------------------------- #
def test_parse_gemini_tool_call_captures_thought_signature():
    import base64

    sig = b"\x00\x01SIG\xff"
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            function_call=SimpleNamespace(name="asset_list", args={}),
                            thought_signature=sig,
                        )
                    ]
                )
            )
        ]
    )
    call = tc.parse_gemini_tool_call(response)
    assert call is not None
    assert call.name == "asset_list"
    # Stored base64-encoded so the canonical turn stays JSON-serializable.
    assert call.thought_signature == base64.b64encode(sig).decode("ascii")


def test_thought_signature_roundtrips_through_canonical_turns():
    call = tc.ToolCall(
        name="asset_list", arguments={"x": 1}, id=None, thought_signature="c2ln"
    )
    turn = tc.assistant_tool_turn(call)
    assert turn["tool_call"]["thought_signature"] == "c2ln"
    restored = tc._call_from_entry(turn)
    assert restored.thought_signature == "c2ln"


def test_gemini_descriptor_includes_signature_only_when_present():
    signed = tc.ToolCall(name="asset_list", arguments={}, thought_signature="c2ln")
    unsigned = tc.ToolCall(name="done", arguments={})
    conv = [tc.assistant_tool_turn(signed), tc.assistant_tool_turn(unsigned)]
    items = tc.to_gemini_history_items(conv)
    assert items[0] == {
        "kind": "function_call",
        "name": "asset_list",
        "args": {},
        "thought_signature": "c2ln",
    }
    # An unsigned call keeps the minimal descriptor shape (no extra key), so
    # existing non-Gemini / pre-signature expectations are unaffected.
    assert items[1] == {"kind": "function_call", "name": "done", "args": {}}
    assert "thought_signature" not in items[1]


def test_non_gemini_serializers_ignore_thought_signature():
    # The signature is Gemini-only; OpenAI/Anthropic history must be unchanged
    # whether or not a signature is present on the canonical turn.
    call = tc.ToolCall(
        name="asset_list", arguments={"a": 1}, id="call_1", thought_signature="c2ln"
    )
    conv = [tc.assistant_tool_turn(call)]
    (chat,) = tc.to_openai_chat_messages(conv)
    assert "thought_signature" not in json.dumps(chat)
    (resp,) = tc.to_openai_responses_items(conv)
    assert "thought_signature" not in json.dumps(resp)
    (anth,) = tc.to_anthropic_messages(conv)
    assert "thought_signature" not in json.dumps(anth)


# --------------------------------------------------------------------------- #
# Reasoning threading (Anthropic extended thinking / OpenAI encrypted reasoning)
# --------------------------------------------------------------------------- #
def test_parse_anthropic_tool_call_captures_thinking_blocks():
    response = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="thinking", thinking="let me plan", signature="sig-abc"
            ),
            SimpleNamespace(type="redacted_thinking", data="enc-xyz"),
            SimpleNamespace(
                type="tool_use", id="toolu_1", name="asset_list", input={}
            ),
        ]
    )
    call = tc.parse_anthropic_tool_call(response)
    assert call is not None
    assert call.name == "asset_list"
    # Thinking blocks are normalized to plain, replayable dicts in API-input shape.
    assert call.reasoning == [
        {"type": "thinking", "thinking": "let me plan", "signature": "sig-abc"},
        {"type": "redacted_thinking", "data": "enc-xyz"},
    ]


def test_parse_anthropic_tool_call_captures_visible_text_block():
    """A ``text`` block preceding the call is carried, not dropped.

    Under ``tool_choice=auto`` Claude may narrate before calling a tool. That
    narration used to be discarded, silently shortening Claude's history in a
    way the other providers' histories were not shortened.
    """
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="ok"),
            SimpleNamespace(type="tool_use", id="t1", name="done", input={}),
        ]
    )
    call = tc.parse_anthropic_tool_call(response)
    assert call is not None
    assert call.reasoning == [{"type": "text", "text": "ok"}]


def test_parse_anthropic_tool_call_with_bare_tool_use_has_no_reasoning():
    response = SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", id="t1", name="done", input={})]
    )
    call = tc.parse_anthropic_tool_call(response)
    assert call is not None
    assert call.reasoning is None


def test_parse_anthropic_tool_call_skips_empty_text_block():
    """The Messages API rejects empty text blocks, so they must not be replayed."""
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="   "),
            SimpleNamespace(type="tool_use", id="t1", name="done", input={}),
        ]
    )
    call = tc.parse_anthropic_tool_call(response)
    assert call is not None
    assert call.reasoning is None


def test_anthropic_replay_preserves_thinking_then_text_then_tool_use_order():
    """Blocks replay in Claude's original order, thinking first.

    The Messages API requires the assistant turn to lead with its thinking
    block; preserving the emitted order satisfies that by construction.
    """
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="plan", signature="sig-1"),
            SimpleNamespace(type="text", text="Checking the roster."),
            SimpleNamespace(type="tool_use", id="t1", name="done", input={}),
        ]
    )
    call = tc.parse_anthropic_tool_call(response)
    assert call is not None

    (turn,) = tc.to_anthropic_messages([tc.assistant_tool_turn(call)])
    assert [block["type"] for block in turn["content"]] == [
        "thinking",
        "text",
        "tool_use",
    ]
    assert turn["content"][1]["text"] == "Checking the roster."


def test_parse_openai_responses_tool_call_captures_reasoning():
    payload = {
        "output": [
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [],
                "encrypted_content": "ENC==",
            },
            {
                "type": "function_call",
                "call_id": "call_9",
                "name": "system_status",
                "arguments": "{}",
            },
        ]
    }
    call = tc.parse_openai_responses_tool_call(payload)
    assert call is not None
    assert call.id == "call_9"
    assert call.reasoning == [
        {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "ENC=="}
    ]


def test_reasoning_roundtrips_through_canonical_turns():
    reasoning = [{"type": "reasoning", "id": "rs_1", "encrypted_content": "ENC=="}]
    call = tc.ToolCall(name="asset_list", arguments={}, id="c1", reasoning=reasoning)
    turn = tc.assistant_tool_turn(call)
    assert turn["tool_call"]["reasoning"] == reasoning
    restored = tc._call_from_entry(turn)
    assert restored.reasoning == reasoning


def test_anthropic_replays_thinking_before_tool_use():
    reasoning = [
        {"type": "thinking", "thinking": "plan", "signature": "sig"},
        {"type": "redacted_thinking", "data": "enc"},
    ]
    call = tc.ToolCall(
        name="asset_list", arguments={"a": 1}, id="call_1", reasoning=reasoning
    )
    (msg,) = tc.to_anthropic_messages([tc.assistant_tool_turn(call)])
    assert msg["role"] == "assistant"
    # Thinking blocks precede the tool_use block, in order.
    assert msg["content"][0] == {
        "type": "thinking",
        "thinking": "plan",
        "signature": "sig",
    }
    assert msg["content"][1] == {"type": "redacted_thinking", "data": "enc"}
    assert msg["content"][2]["type"] == "tool_use"
    assert msg["content"][2]["name"] == "asset_list"


def test_openai_responses_replays_reasoning_before_function_call():
    reasoning = [{"type": "reasoning", "id": "rs_1", "encrypted_content": "ENC=="}]
    call = tc.ToolCall(
        name="system_status", arguments={}, id="call_9", reasoning=reasoning
    )
    items = tc.to_openai_responses_items([tc.assistant_tool_turn(call)])
    # The reasoning item is emitted immediately before its function_call.
    assert items[0] == {
        "type": "reasoning",
        "id": "rs_1",
        "encrypted_content": "ENC==",
    }
    assert items[1]["type"] == "function_call"
    assert items[1]["call_id"] == "call_9"


def test_reasoning_absent_leaves_history_shapes_unchanged():
    # A call with no threaded reasoning must produce exactly the prior shapes.
    call = tc.ToolCall(name="done", arguments={}, id="call_1")
    conv = [tc.assistant_tool_turn(call)]
    (anth,) = tc.to_anthropic_messages(conv)
    assert [b["type"] for b in anth["content"]] == ["tool_use"]
    (resp,) = tc.to_openai_responses_items(conv)
    assert resp["type"] == "function_call"
    # Chat history never carries reasoning items.
    (chat,) = tc.to_openai_chat_messages(conv)
    assert "reasoning" not in json.dumps(chat)
