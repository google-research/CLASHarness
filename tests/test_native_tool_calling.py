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

"""Native function/tool-calling mode wiring tests (WS3-5).

Two complementary layers:

* **Provider integration** -- mock ``urllib.request.urlopen`` to return a raw
  OpenAI Chat/Responses tool-call payload and assert the request is shaped with
  ``tools``/``tool_choice`` and the returned tool call is reverse-mapped to the
  same workspace CLI string text mode produces.
* **Orchestration** -- monkeypatch :meth:`LLMAgent._native_tool_call` to script
  :class:`ToolCall` sequences and assert the ``_act_native`` loop's DONE
  handling, invalid-retry + tool_result pairing, multi-step tool_result replay,
  and exhausted->coerce-DONE stall handling, independent of any provider.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

import train_eval.agents as agents
from clasharness.evaluation.llm_retry import (
    anthropic_kwargs_carry_temperature as _has_temp,
)
from train_eval.tool_calling import (
    ToolCall,
    parse_anthropic_tool_call,
    parse_gemini_tool_call,
    parse_openai_chat_tool_call,
)


def _obs(step: int, stdout: str = "", stderr: str = "", exit_code: int = 0):
    return SimpleNamespace(
        step_number=step,
        command_output=stdout,
        error_output=stderr,
        exit_code=exit_code,
    )


class _Resp:
    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def _chat_tool_call_payload(name: str, arguments: dict) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                }
            }
        ]
    }


def _responses_tool_call_payload(name: str, arguments: dict) -> dict:
    return {
        "object": "response",
        "output": [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": name,
                "arguments": json.dumps(arguments),
            }
        ],
    }


def _mock_urlopen(monkeypatch, payloads: list[dict], bodies: list[dict]):
    queue = list(payloads)

    def _urlopen(req, timeout=0):
        bodies.append(json.loads(req.data.decode("utf-8")))
        return _Resp(queue.pop(0))

    monkeypatch.setattr(agents.urllib.request, "urlopen", _urlopen)


# --------------------------------------------------------------------------- #
# Provider integration (OpenAI-compatible)
# --------------------------------------------------------------------------- #
def test_native_chat_tool_call_reverse_maps_and_shapes_request(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    bodies: list[dict] = []
    _mock_urlopen(
        monkeypatch, [_chat_tool_call_payload("weather_get", {"location": "Paris"})], bodies
    )

    agent = agents.LLMAgent(
        provider="openai",
        model="legacy-openai-chat-model",
        base_url="https://example.test/v1",
        tool_mode="native",
    )
    agent.reset("Check the weather in Paris.")
    action = agent.act(_obs(0))

    assert action == "weather get --location Paris"
    body = bodies[0]
    assert body["tool_choice"] == "auto"
    assert isinstance(body["tools"], list) and body["tools"]
    assert any(t["function"]["name"] == "weather_get" for t in body["tools"])
    # Native mode uses the tool-calling system prompt with no textual catalog.
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"] == agents.LLMAgent.NATIVE_SYSTEM_PROMPT
    assert "Command catalog" not in body["messages"][0]["content"]


def test_native_responses_tool_call_uses_flat_tools_and_reverse_maps(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    bodies: list[dict] = []
    _mock_urlopen(
        monkeypatch,
        [_responses_tool_call_payload("weather_get", {"location": "Paris"})],
        bodies,
    )

    agent = agents.LLMAgent(
        provider="openai",
        model="gpt-5.2",
        base_url="https://example.test/v1",
        tool_mode="native",
    )
    agent.reset("Check the weather in Paris.")
    action = agent.act(_obs(0))

    assert action == "weather get --location Paris"
    body = bodies[0]
    assert body["tool_choice"] == "auto"
    # Responses API uses the flat function-tool shape (no nested "function").
    assert body["tools"][0]["type"] == "function"
    assert "name" in body["tools"][0]
    assert "function" not in body["tools"][0]


def test_native_done_tool_stops_episode(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    bodies: list[dict] = []
    _mock_urlopen(monkeypatch, [_chat_tool_call_payload("done", {})], bodies)

    agent = agents.LLMAgent(
        provider="openai",
        model="legacy-openai-chat-model",
        base_url="https://example.test/v1",
        tool_mode="native",
    )
    agent.reset("Check the weather in Paris.")
    action = agent.act(_obs(0))

    assert action == "DONE"
    assert agent.should_stop() is True


# --------------------------------------------------------------------------- #
# Orchestration (_act_native), provider-agnostic
# --------------------------------------------------------------------------- #
def _native_agent() -> agents.LLMAgent:
    return agents.LLMAgent(
        provider="openai",
        model="legacy-openai-chat-model",
        base_url="https://example.test/v1",
        tool_mode="native",
    )


def test_native_invalid_tool_call_retries_and_pairs_result(monkeypatch):
    agent = _native_agent()
    agent.reset("Check the weather in Paris.")
    scripted = iter(
        [
            ToolCall(name="definitely_not_a_tool", arguments={}, id="c1"),
            ToolCall(name="weather_get", arguments={"location": "Paris"}, id="c2"),
        ]
    )
    monkeypatch.setattr(agent, "_native_tool_call", lambda: next(scripted))

    action = agent.act(_obs(0))

    assert action == "weather get --location Paris"
    # The rejected call is recorded as an assistant tool_call turn, immediately
    # followed by a tool_result turn carrying the nudge (Anthropic pairing rule).
    tool_call_turns = [m for m in agent._messages if "tool_call" in m]
    tool_result_turns = [m for m in agent._messages if m.get("role") == "tool"]
    assert len(tool_call_turns) == 2  # invalid attempt + accepted attempt
    assert len(tool_result_turns) == 1  # the rejection nudge for the invalid one
    assert "rejected" in tool_result_turns[0]["tool_result"]["output"]


def test_native_multistep_replays_observation_as_tool_result(monkeypatch):
    agent = _native_agent()
    agent.reset("Check the weather in Paris.")
    scripted = iter(
        [
            ToolCall(name="weather_get", arguments={"location": "Paris"}, id="c1"),
            ToolCall(name="done", arguments={}, id="c2"),
        ]
    )
    monkeypatch.setattr(agent, "_native_tool_call", lambda: next(scripted))

    a0 = agent.act(_obs(0))
    assert a0 == "weather get --location Paris"
    assert agent._pending_call is not None
    assert agent._pending_call.name == "weather_get"

    a1 = agent.act(_obs(1, stdout="Sunny, 22C"))
    assert a1 == "DONE"

    # Step 1 must have replayed the observation as a tool_result paired with the
    # step-0 call before requesting the next tool.
    tool_results = [m for m in agent._messages if m.get("role") == "tool"]
    assert tool_results
    assert any("Sunny, 22C" in m["tool_result"]["output"] for m in tool_results)
    assert tool_results[0]["tool_result"]["id"] == "c1"


def test_native_exhausted_retries_coerce_done_with_stall_reason(monkeypatch):
    agent = _native_agent()
    agent.reset("Check the weather in Paris.")
    # A known tool called with no args reverse-maps to a non-empty CLI that fails
    # validation (missing required --location), so the stall is classified as an
    # exhausted-invalid rather than an empty-generation stall.
    monkeypatch.setattr(
        agent,
        "_native_tool_call",
        lambda: ToolCall(name="weather_get", arguments={}, id="x"),
    )

    action = agent.act(_obs(0))

    assert action == "DONE"
    assert agent.should_stop() is True
    assert agent.debug_stall_reason() == "invalid_command_exhausted"


def test_native_no_tool_call_is_treated_as_invalid(monkeypatch):
    agent = _native_agent()
    agent.reset("Check the weather in Paris.")
    # Under the default tool_choice=auto a text-only turn is a normal outcome;
    # the loop should nudge and eventually coerce DONE rather than crash.
    monkeypatch.setattr(agent, "_native_tool_call", lambda: None)

    action = agent.act(_obs(0))

    assert action == "DONE"
    assert agent.should_stop() is True
    assert agent.debug_stall_reason() == "generation_stall_empty_reply"


def test_native_no_tool_call_keeps_assistant_text_in_history(monkeypatch):
    """A text-only turn is preserved, not silently deleted.

    Dropping it made the nudge read as a non-sequitur and erased whatever
    reasoning the model had just written. ``tool_choice=auto`` turns this from a
    near-unreachable branch into a routine one, so the loss now matters.
    """
    agent = _native_agent()
    agent.reset("Check the weather in Paris.")
    texts = iter(["I should look up Paris first.", "", ""])

    def _no_call():
        agent._last_no_call_text = next(texts)
        return None

    monkeypatch.setattr(agent, "_native_tool_call", _no_call)

    agent.act(_obs(0))

    assistant_texts = [
        m["content"]
        for m in agent._messages
        if m.get("role") == "assistant" and isinstance(m.get("content"), str)
    ]
    assert "I should look up Paris first." in assistant_texts
    # The nudge still arrives, as its own user turn (never merged into the
    # assistant turn, and never doubled up with another user turn).
    nudges = [
        m
        for m in agent._messages
        if m.get("role") == "user"
        and isinstance(m.get("content"), str)
        and "did not call a tool" in m["content"]
    ]
    assert nudges
    roles = [m.get("role") for m in agent._messages if "tool_call" not in m]
    assert not any(a == "user" == b for a, b in zip(roles, roles[1:]))


def test_native_no_tool_call_without_text_still_only_nudges(monkeypatch):
    """No text to salvage -> behave exactly as before (bare nudge)."""
    agent = _native_agent()
    agent.reset("Check the weather in Paris.")
    monkeypatch.setattr(agent, "_native_tool_call", lambda: None)

    agent.act(_obs(0))

    assert not [
        m
        for m in agent._messages
        if m.get("role") == "assistant" and isinstance(m.get("content"), str)
    ]


def test_text_mode_remains_the_constructor_default():
    agent = agents.LLMAgent(
        provider="openai",
        model="legacy-openai-chat-model",
        base_url="https://example.test/v1",
    )
    assert agent._tool_mode == "text"


def test_invalid_tool_mode_rejected():
    with pytest.raises(ValueError):
        agents.LLMAgent(
            provider="openai",
            model="legacy-openai-chat-model",
            base_url="https://example.test/v1",
            tool_mode="bogus",
        )


# --------------------------------------------------------------------------- #
# Reasoning threading: OpenAI Responses (encrypted_content)
# --------------------------------------------------------------------------- #
def _responses_payload_with_reasoning(name: str, arguments: dict) -> dict:
    return {
        "object": "response",
        "output": [
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [],
                "encrypted_content": "ENC==",
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": name,
                "arguments": json.dumps(arguments),
            },
        ],
    }


def test_native_responses_threads_reasoning_when_native_reasoning_on(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    bodies: list[dict] = []
    _mock_urlopen(
        monkeypatch,
        [
            _responses_payload_with_reasoning("weather_get", {"location": "Paris"}),
            _responses_tool_call_payload("done", {}),
        ],
        bodies,
    )

    agent = agents.LLMAgent(
        provider="openai",
        model="gpt-5.2",
        base_url="https://example.test/v1",
        tool_mode="native",
        native_reasoning=True,
    )
    agent.reset("Check the weather in Paris.")

    agent.act(_obs(0))
    # Step 0 request opts into stateless encrypted reasoning.
    assert bodies[0]["store"] is False
    assert bodies[0]["include"] == ["reasoning.encrypted_content"]

    agent.act(_obs(1, stdout="Sunny, 22C"))
    # Step 1 replays the reasoning item immediately before its function_call.
    input_items = bodies[1]["input"]
    types = [it.get("type") for it in input_items if isinstance(it, dict)]
    assert "reasoning" in types
    ri = types.index("reasoning")
    fci = types.index("function_call")
    assert ri < fci
    reasoning_item = input_items[ri]
    assert reasoning_item["encrypted_content"] == "ENC=="


def test_native_responses_no_reasoning_when_native_reasoning_off(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    bodies: list[dict] = []
    _mock_urlopen(
        monkeypatch,
        [_responses_tool_call_payload("weather_get", {"location": "Paris"})],
        bodies,
    )

    agent = agents.LLMAgent(
        provider="openai",
        model="gpt-5.2",
        base_url="https://example.test/v1",
        tool_mode="native",
        native_reasoning=False,
    )
    agent.reset("Check the weather in Paris.")
    agent.act(_obs(0))

    # native_reasoning=False reproduces the prior reasoning-dropping behavior.
    assert "store" not in bodies[0]
    assert "include" not in bodies[0]


# --------------------------------------------------------------------------- #
# Reasoning effort override
# --------------------------------------------------------------------------- #
def test_openai_omits_reasoning_key_when_no_effort_requested(monkeypatch):
    """Unset effort must leave the payload byte-identical to prior runs.

    The endpoint applies the model's own default (medium for gpt-5.5) when the
    key is absent, so sending nothing is NOT the same as sending "medium" -- and
    existing result sets were produced with the key absent.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    bodies: list[dict] = []
    _mock_urlopen(
        monkeypatch,
        [_responses_payload_with_reasoning("weather_get", {"location": "Paris"})],
        bodies,
    )
    agent = agents.LLMAgent(
        provider="openai",
        model="gpt-5.2",
        base_url="https://example.test/v1",
        tool_mode="native",
        native_reasoning=True,
    )
    agent.reset("Check the weather in Paris.")
    agent.act(_obs(0))

    assert "reasoning" not in bodies[0]


def test_openai_sends_requested_reasoning_effort(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    bodies: list[dict] = []
    _mock_urlopen(
        monkeypatch,
        [_responses_payload_with_reasoning("weather_get", {"location": "Paris"})],
        bodies,
    )
    agent = agents.LLMAgent(
        provider="openai",
        model="gpt-5.2",
        base_url="https://example.test/v1",
        tool_mode="native",
        native_reasoning=True,
        reasoning_effort="high",
    )
    agent.reset("Check the weather in Paris.")
    agent.act(_obs(0))

    assert bodies[0]["reasoning"] == {"effort": "high"}
    # The effort override must not disturb the stateless-reasoning opt-in.
    assert bodies[0]["store"] is False
    assert bodies[0]["include"] == ["reasoning.encrypted_content"]


@pytest.mark.parametrize(
    "effort,wire",
    [
        # Canonical rungs OpenAI expresses verbatim.
        ("none", "none"), ("low", "low"), ("medium", "medium"),
        ("high", "high"), ("xhigh", "xhigh"),
        # Rungs the OpenAI/Bedrock-GPT wire vocabulary lacks, folded onto the
        # nearest real tier so the shared flag value is accepted everywhere:
        # 'minimal' (proxy 400s on it) -> 'low', 'max' -> the top tier 'xhigh'.
        ("minimal", "low"), ("max", "xhigh"),
    ],
)
def test_openai_maps_canonical_effort_onto_wire_vocabulary(
    monkeypatch, effort, wire
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    bodies: list[dict] = []
    _mock_urlopen(
        monkeypatch,
        [_responses_payload_with_reasoning("weather_get", {"location": "Paris"})],
        bodies,
    )
    agent = agents.LLMAgent(
        provider="openai",
        model="gpt-5.2",
        base_url="https://example.test/v1",
        tool_mode="native",
        native_reasoning=True,
        reasoning_effort=effort,
    )
    agent.reset("Check the weather in Paris.")
    agent.act(_obs(0))

    assert bodies[0]["reasoning"] == {"effort": wire}
    # The recorded effective effort matches the wire value, not the raw request.
    cfg = agent.debug_run_config()
    assert cfg["reasoning_effort"] == effort
    assert cfg["reasoning_effort_effective"] == wire


def test_openai_effort_ignored_when_native_reasoning_off(monkeypatch):
    """Effort rides on the reasoning opt-in; with it off nothing is sent."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    bodies: list[dict] = []
    _mock_urlopen(
        monkeypatch,
        [_responses_tool_call_payload("weather_get", {"location": "Paris"})],
        bodies,
    )
    agent = agents.LLMAgent(
        provider="openai",
        model="gpt-5.2",
        base_url="https://example.test/v1",
        tool_mode="native",
        native_reasoning=False,
        reasoning_effort="high",
    )
    agent.reset("Check the weather in Paris.")
    agent.act(_obs(0))

    assert "reasoning" not in bodies[0]


# --------------------------------------------------------------------------- #
# Parallel tool calls: only the first runs, but the extras are recorded
# --------------------------------------------------------------------------- #
def test_openai_parallel_tool_calls_are_recorded_not_silently_dropped(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    bodies: list[dict] = []
    parallel = {
        "object": "response",
        "output": [
            {"type": "reasoning", "id": "rs_1", "summary": [],
             "encrypted_content": "ENC=="},
            {"type": "function_call", "call_id": "call_1",
             "name": "weather_get", "arguments": json.dumps({"location": "Paris"})},
            {"type": "function_call", "call_id": "call_2",
             "name": "weather_get", "arguments": json.dumps({"location": "Rome"})},
            {"type": "function_call", "call_id": "call_3",
             "name": "done", "arguments": "{}"},
        ],
    }
    _mock_urlopen(monkeypatch, [parallel], bodies)
    agent = agents.LLMAgent(
        provider="openai",
        model="gpt-5.2",
        base_url="https://example.test/v1",
        tool_mode="native",
        native_reasoning=True,
        capture_interactions=True,
    )
    agent.reset("Check the weather in Paris.")
    action = agent.act(_obs(0))

    # Behavior is unchanged: the FIRST call is the one that runs.
    assert action == "weather get --location Paris"
    # But the discarded plan is now visible instead of vanishing.
    rec = agent.debug_interactions()[-1]
    assert rec["response"]["dropped_parallel_calls"] == ["weather_get", "done"]


def test_openai_single_tool_call_reports_no_dropped_calls(monkeypatch):
    """The common case must stay clean -- no false positives."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    bodies: list[dict] = []
    _mock_urlopen(
        monkeypatch,
        [_responses_payload_with_reasoning("weather_get", {"location": "Paris"})],
        bodies,
    )
    agent = agents.LLMAgent(
        provider="openai",
        model="gpt-5.2",
        base_url="https://example.test/v1",
        tool_mode="native",
        native_reasoning=True,
        capture_interactions=True,
    )
    agent.reset("Check the weather in Paris.")
    agent.act(_obs(0))

    assert agent.debug_interactions()[-1]["response"]["dropped_parallel_calls"] == []


def test_anthropic_parallel_tool_use_blocks_are_recorded():
    """Anthropic emits parallel calls as extra ``tool_use`` blocks."""
    call = parse_anthropic_tool_call(
        SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="plan", signature="sig-1"),
                SimpleNamespace(type="tool_use", id="toolu_1", name="weather_get",
                                input={"location": "Paris"}),
                SimpleNamespace(type="tool_use", id="toolu_2", name="weather_get",
                                input={"location": "Rome"}),
                SimpleNamespace(type="tool_use", id="toolu_3", name="done", input={}),
            ]
        )
    )
    # The first call still wins; the rest are visible rather than gone.
    assert call.name == "weather_get"
    assert call.arguments == {"location": "Paris"}
    assert call.id == "toolu_1"
    assert call.dropped_parallel_calls == ["weather_get", "done"]
    # Reasoning threading must be unaffected by the extra blocks.
    assert call.reasoning == [
        {"type": "thinking", "thinking": "plan", "signature": "sig-1"}
    ]


def test_anthropic_single_tool_use_reports_no_dropped_calls():
    call = parse_anthropic_tool_call(
        SimpleNamespace(
            content=[
                SimpleNamespace(type="tool_use", id="toolu_1", name="weather_get",
                                input={"location": "Paris"}),
            ]
        )
    )
    assert call.dropped_parallel_calls == []


def test_anthropic_dropped_calls_reach_the_interaction_record():
    """Parser -> agent wiring, so the count is comparable to OpenAI's."""
    bodies: list[dict] = []
    agent = agents.LLMAgent(
        provider="anthropic",
        model="claude-sonnet-5",
        tool_mode="native",
        native_reasoning=True,
        capture_interactions=True,
    )
    agent._client = _FakeAnthropicClient(
        [
            _anthropic_response(
                [
                    _tool_use_block("weather_get", {"location": "Paris"}, "toolu_1"),
                    _tool_use_block("done", {}, "toolu_2"),
                ]
            )
        ],
        bodies,
    )
    agent.reset("Check the weather in Paris.")
    action = agent.act(_obs(0))

    assert action == "weather get --location Paris"
    rec = agent.debug_interactions()[-1]
    assert rec["response"]["dropped_parallel_calls"] == ["done"]


def _gemini_part(name: str, args: dict) -> SimpleNamespace:
    return SimpleNamespace(
        function_call=SimpleNamespace(name=name, args=args),
        thought_signature=None,
    )


def _gemini_response(parts_per_candidate: list[list]) -> SimpleNamespace:
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(content=SimpleNamespace(parts=parts))
            for parts in parts_per_candidate
        ]
    )


def _gemini_text_part(text: str, thought: bool = False) -> SimpleNamespace:
    return SimpleNamespace(function_call=None, text=text, thought=thought)


def test_gemini_text_extractor_skips_thought_parts():
    """Thought parts are logged as thinking_text; replaying them as visible
    assistant text would leak the chain of thought back in the wrong shape."""
    response = _gemini_response(
        [[
            _gemini_text_part("internal deliberation", thought=True),
            _gemini_text_part("Let me check the roster."),
        ]]
    )
    assert agents._gemini_text_from_response(response) == "Let me check the roster."


def test_gemini_text_extractor_returns_empty_on_unknown_shape():
    assert agents._gemini_text_from_response(SimpleNamespace()) == ""
    assert agents._gemini_text_from_response(object()) == ""


def test_openai_text_extractor_handles_both_wire_shapes():
    responses_payload = {
        "object": "response",
        "output": [
            {"type": "reasoning", "summary": [{"text": "hidden"}]},
            {"type": "message", "content": [{"type": "output_text", "text": "Hello."}]},
        ],
    }
    # Reasoning items are not visible assistant text.
    assert agents._openai_text_from_payload(responses_payload) == "Hello."

    chat_payload = {"choices": [{"message": {"content": "Hello."}}]}
    assert agents._openai_text_from_payload(chat_payload) == "Hello."


def test_openai_text_extractor_returns_empty_instead_of_raising():
    """Unlike ``_extract_openai_message_text`` this must never raise, and must
    never issue the followup request that method can."""
    assert agents._openai_text_from_payload({}) == ""
    assert agents._openai_text_from_payload({"object": "response", "output": []}) == ""
    assert agents._openai_text_from_payload({"choices": [{"message": {}}]}) == ""


def test_native_chat_no_tool_call_preserves_text_end_to_end(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    bodies: list[dict] = []
    _mock_urlopen(
        monkeypatch,
        [
            {"choices": [{"message": {"content": "Let me think about Paris."}}]},
            _chat_tool_call_payload("weather_get", {"location": "Paris"}),
        ],
        bodies,
    )
    agent = agents.LLMAgent(
        provider="openai",
        model="legacy-openai-chat-model",
        base_url="https://example.test/v1",
        tool_mode="native",
    )
    agent.reset("Check the weather in Paris.")
    action = agent.act(_obs(0))

    assert action == "weather get --location Paris"
    # The retry request carries the model's own text back to it.
    retry_messages = bodies[1]["messages"]
    assert {
        "role": "assistant",
        "content": "Let me think about Paris.",
    } in retry_messages


def _gemini_agent_capturing_config(monkeypatch, configs: list):
    """A native Gemini agent whose generate_content config is captured."""
    agent = agents.LLMAgent(
        provider="gemini", model="gemini-3-flash", tool_mode="native"
    )

    def _generate_content(*, model, contents, config):
        configs.append(config)
        return _gemini_response([[_gemini_part("weather_get", {"location": "Paris"})]])

    monkeypatch.setattr(
        agent,
        "_get_gemini_client",
        lambda: SimpleNamespace(
            models=SimpleNamespace(generate_content=_generate_content)
        ),
    )
    return agent


def test_gemini_native_tool_config_defaults_to_auto(monkeypatch):
    configs: list = []
    agent = _gemini_agent_capturing_config(monkeypatch, configs)
    agent.reset("Check the weather in Paris.")

    assert agent.act(_obs(0)) == "weather get --location Paris"
    mode = configs[0].tool_config.function_calling_config.mode
    assert getattr(mode, "name", mode) == "AUTO"


def test_gemini_native_tool_config_forced_opt_in(monkeypatch):
    configs: list = []
    agent = _gemini_agent_capturing_config(monkeypatch, configs)
    agent._native_tool_choice = "forced"
    agent.reset("Check the weather in Paris.")

    assert agent.act(_obs(0)) == "weather get --location Paris"
    mode = configs[0].tool_config.function_calling_config.mode
    assert getattr(mode, "name", mode) == "ANY"


# ---- Gemini reasoning effort (ThinkingConfig.thinking_level) ----
#
# The three Gemini models in the shipped aligned_setting tree ran with NO effort
# pin while Claude and gpt-5.5 ran at `high`: the runbook never passed the flag
# to Gemini, and the two Gemini request paths did not read it even when it was
# passed, so the cross-family comparison was an unequal-effort one.


def _gemini_agent_capturing_config_kwargs(monkeypatch, configs, **agent_kwargs):
    kwargs = dict(provider="gemini", model="gemini-3-flash", tool_mode="native")
    kwargs.update(agent_kwargs)
    agent = agents.LLMAgent(**kwargs)

    def _generate_content(*, model, contents, config):
        configs.append(config)
        return _gemini_response([[_gemini_part("weather_get", {"location": "Paris"})]])

    monkeypatch.setattr(
        agent,
        "_get_gemini_client",
        lambda: SimpleNamespace(
            models=SimpleNamespace(generate_content=_generate_content)
        ),
    )
    return agent


def test_gemini_omits_thinking_config_when_nothing_requested(monkeypatch):
    # No effort, no thought capture -> the request must stay byte-identical to
    # the pre-flag behavior.
    configs: list = []
    agent = _gemini_agent_capturing_config_kwargs(monkeypatch, configs)
    agent.reset("Check the weather in Paris.")
    agent.act(_obs(0))
    assert configs[0].thinking_config is None


def test_gemini_native_sends_requested_reasoning_effort(monkeypatch):
    configs: list = []
    agent = _gemini_agent_capturing_config_kwargs(
        monkeypatch, configs, reasoning_effort="high"
    )
    agent.reset("Check the weather in Paris.")
    agent.act(_obs(0))
    level = configs[0].thinking_config.thinking_level
    assert getattr(level, "name", level) == "HIGH"


def test_gemini_text_path_sends_requested_reasoning_effort(monkeypatch):
    # Unlike OpenAI/Anthropic, Gemini honours the flag in text mode too: the
    # thinking_level is a plain generate_content config, not a tool-call opt-in.
    configs: list = []
    agent = _gemini_agent_capturing_config_kwargs(
        monkeypatch, configs, tool_mode="text", reasoning_effort="low"
    )
    agent.reset("Check the weather in Paris.")
    agent.act(_obs(0))
    level = configs[0].thinking_config.thinking_level
    assert getattr(level, "name", level) == "LOW"


@pytest.mark.parametrize(
    "effort,expected",
    [("none", "MINIMAL"), ("minimal", "MINIMAL"), ("low", "LOW"),
     ("medium", "MEDIUM"), ("high", "HIGH"),
     # 'xhigh'/'max' have no Gemini equivalent above HIGH, so the unified
     # ladder folds them onto HIGH instead of rejecting the flag.
     ("xhigh", "HIGH"), ("max", "HIGH")],
)
def test_gemini_effort_level_mapping(monkeypatch, effort, expected):
    configs: list = []
    agent = _gemini_agent_capturing_config_kwargs(
        monkeypatch, configs, reasoning_effort=effort
    )
    agent.reset("Check the weather in Paris.")
    agent.act(_obs(0))
    level = configs[0].thinking_config.thinking_level
    assert getattr(level, "name", level) == expected


@pytest.mark.parametrize("effort", ["xhigh", "max"])
def test_gemini_accepts_above_high_effort_by_folding_to_high(monkeypatch, effort):
    # The unified ladder means every rung is a valid flag value everywhere; a
    # rung a provider can't express natively is folded onto its nearest real
    # tier (HIGH for Gemini) and REPORTED as that, so nothing is mislabeled.
    configs: list = []
    agent = _gemini_agent_capturing_config_kwargs(
        monkeypatch, configs, reasoning_effort=effort
    )
    agent.reset("Check the weather in Paris.")
    agent.act(_obs(0))
    level = configs[0].thinking_config.thinking_level
    assert getattr(level, "name", level) == "HIGH"
    assert agent.debug_run_config()["reasoning_effort_effective"] == "high"


@pytest.mark.parametrize("provider", ["gemini", "openai", "anthropic"])
def test_invalid_effort_rejected_at_construction_for_every_provider(
    monkeypatch, provider
):
    # A typo (not on the canonical ladder) must fail loudly for ANY provider,
    # not silently ride through as an unmapped value.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(ValueError, match="is not a valid level"):
        agents.LLMAgent(
            provider=provider, model="m", tool_mode="native",
            reasoning_effort="bogus",
        )


@pytest.mark.parametrize("effort", ["xhigh", "max"])
def test_non_gemini_providers_still_accept_high_efforts(monkeypatch, effort):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent = agents.LLMAgent(
        provider="openai", model="gpt-5.2", base_url="https://example.test/v1",
        tool_mode="native", native_reasoning=True, reasoning_effort=effort,
    )
    # The requested value is preserved verbatim; only the wire form is mapped.
    assert agent._reasoning_effort == effort


def test_gemini_effort_and_thought_capture_share_one_thinking_config(monkeypatch):
    # Two switches, ONE ThinkingConfig. Assigning them separately would drop the
    # first, silently disabling whichever was set first.
    configs: list = []
    agent = _gemini_agent_capturing_config_kwargs(
        monkeypatch, configs, reasoning_effort="medium", capture_thinking=True
    )
    agent.reset("Check the weather in Paris.")
    agent.act(_obs(0))
    tc = configs[0].thinking_config
    assert tc.include_thoughts is True
    assert getattr(tc.thinking_level, "name", tc.thinking_level) == "MEDIUM"


def test_gemini_capture_thinking_alone_still_omits_level(monkeypatch):
    configs: list = []
    agent = _gemini_agent_capturing_config_kwargs(
        monkeypatch, configs, capture_thinking=True
    )
    agent.reset("Check the weather in Paris.")
    agent.act(_obs(0))
    tc = configs[0].thinking_config
    assert tc.include_thoughts is True
    assert tc.thinking_level is None


# ---- Gemini per-model thinking_level floor: heal unsupported MINIMAL ----
#
# Some Gemini models (e.g. Gemini 3 Pro) reject the lowest thinking level
# MINIMAL with a 400 "thinking_level MINIMAL is not supported by this model",
# while Flash accepts it. 'none'/'minimal' map to MINIMAL, so on such a model
# the first request must be healed by stepping UP to the lowest level the model
# accepts, not failed on a non-retryable INVALID_ARGUMENT.


def _gemini_client_rejecting_minimal(configs):
    def _generate_content(*, model, contents, config):
        configs.append(config)
        level = config.thinking_config.thinking_level
        if getattr(level, "name", level) == "MINIMAL":
            raise RuntimeError(
                "thinking_level MINIMAL is not supported by this model"
            )
        return _gemini_response(
            [[_gemini_part("weather_get", {"location": "Paris"})]]
        )

    return SimpleNamespace(
        models=SimpleNamespace(generate_content=_generate_content)
    )


def test_gemini_escalates_unsupported_minimal_thinking_level(monkeypatch):
    agents._GEMINI_MODEL_MIN_THINKING_LEVEL.pop("gemini-3-pro", None)
    configs: list = []
    agent = agents.LLMAgent(
        provider="gemini", model="gemini-3-pro", tool_mode="native",
        reasoning_effort="none",
    )
    monkeypatch.setattr(
        agent, "_get_gemini_client",
        lambda: _gemini_client_rejecting_minimal(configs),
    )
    agent.reset("Check the weather in Paris.")
    action = agent.act(_obs(0))

    assert action == "weather get --location Paris"
    # First MINIMAL (rejected), then escalated to LOW (accepted) within ONE step
    # -- the run heals instead of failing.
    names = [
        getattr(c.thinking_config.thinking_level, "name",
                c.thinking_config.thinking_level)
        for c in configs
    ]
    assert names == ["MINIMAL", "LOW"]
    # The floor is remembered so later steps skip the rejected level.
    assert agents._GEMINI_MODEL_MIN_THINKING_LEVEL["gemini-3-pro"] == "LOW"


def test_gemini_thinking_floor_persists_across_steps(monkeypatch):
    agents._GEMINI_MODEL_MIN_THINKING_LEVEL.pop("gemini-3-pro", None)
    configs: list = []
    agent = agents.LLMAgent(
        provider="gemini", model="gemini-3-pro", tool_mode="native",
        reasoning_effort="minimal",
    )
    monkeypatch.setattr(
        agent, "_get_gemini_client",
        lambda: _gemini_client_rejecting_minimal(configs),
    )
    agent.reset("Check the weather in Paris.")
    agent.act(_obs(0))
    agent.act(_obs(1, stdout="Sunny"))

    # Step 0: MINIMAL rejected -> LOW. Step 1 goes straight to LOW (no wasted
    # MINIMAL request) because the floor was learned once for the process.
    names = [
        getattr(c.thinking_config.thinking_level, "name",
                c.thinking_config.thinking_level)
        for c in configs
    ]
    assert names == ["MINIMAL", "LOW", "LOW"]


def test_gemini_effective_effort_reflects_learned_floor():
    agents._GEMINI_MODEL_MIN_THINKING_LEVEL.pop("gemini-3-pro", None)
    agent = agents.LLMAgent(
        provider="gemini", model="gemini-3-pro", tool_mode="native",
        reasoning_effort="none",
    )
    # Before any call the floor is unknown, so it reads the requested level.
    assert agent.debug_run_config()["reasoning_effort_effective"] == "minimal"
    # Once the floor is learned, it reflects the level actually sent.
    agents._GEMINI_MODEL_MIN_THINKING_LEVEL["gemini-3-pro"] = "LOW"
    assert agent.debug_run_config()["reasoning_effort_effective"] == "low"


def test_gemini_none_effective_effort_is_minimal_not_none():
    # Gemini has no true "off": 'none' sends MINIMAL, so the effective level is
    # reported as 'minimal', not 'none'.
    agents._GEMINI_MODEL_MIN_THINKING_LEVEL.pop("gemini-3-flash", None)
    agent = agents.LLMAgent(
        provider="gemini", model="gemini-3-flash", tool_mode="native",
        reasoning_effort="none",
    )
    assert agent.debug_run_config()["reasoning_effort_effective"] == "minimal"


# ---- reasoning_effort_effective: what the wire actually carried ----


def test_run_config_records_effective_effort_for_gemini(monkeypatch):
    agent = agents.LLMAgent(
        provider="gemini", model="gemini-3-flash", tool_mode="native",
        reasoning_effort="high",
    )
    cfg = agent.debug_run_config()
    assert cfg["reasoning_effort"] == "high"
    assert cfg["reasoning_effort_effective"] == "high"


def test_run_config_effective_effort_empty_when_unset(monkeypatch):
    agent = agents.LLMAgent(
        provider="gemini", model="gemini-3-flash", tool_mode="native"
    )
    cfg = agent.debug_run_config()
    assert cfg["reasoning_effort"] == ""
    assert cfg["reasoning_effort_effective"] == ""


def test_run_config_effective_effort_blank_when_openai_text_mode(monkeypatch):
    # The requested level is recorded, but text-mode OpenAI never sends it.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent = agents.LLMAgent(
        provider="openai", model="gpt-5.2", base_url="https://example.test/v1",
        tool_mode="text", reasoning_effort="high",
    )
    cfg = agent.debug_run_config()
    assert cfg["reasoning_effort"] == "high"
    assert cfg["reasoning_effort_effective"] == ""


def test_gemini_parallel_function_call_parts_are_recorded():
    """Gemini emits parallel calls as extra parts inside ONE candidate."""
    call = parse_gemini_tool_call(
        _gemini_response(
            [[
                _gemini_part("weather_get", {"location": "Paris"}),
                _gemini_part("weather_get", {"location": "Rome"}),
                _gemini_part("done", {}),
            ]]
        )
    )
    assert call.name == "weather_get"
    assert call.arguments == {"location": "Paris"}
    assert call.dropped_parallel_calls == ["weather_get", "done"]


def test_gemini_extra_candidates_are_not_counted_as_parallel_calls():
    """Extra candidates are alternative samples, not a multi-call plan.

    Counting them would inflate Gemini's parallel rate against the other
    providers, which is exactly the comparison this field exists to support.
    """
    call = parse_gemini_tool_call(
        _gemini_response(
            [
                [_gemini_part("weather_get", {"location": "Paris"})],
                [_gemini_part("done", {})],
            ]
        )
    )
    assert call.name == "weather_get"
    assert call.dropped_parallel_calls == []


def test_gemini_preserves_thought_signature_of_the_chosen_part():
    """The signature must come from the chosen part, not a later one."""
    chosen = _gemini_part("weather_get", {"location": "Paris"})
    chosen.thought_signature = b"\x01\x02"
    extra = _gemini_part("done", {})
    extra.thought_signature = b"\xff\xff"
    call = parse_gemini_tool_call(_gemini_response([[chosen, extra]]))
    assert call.thought_signature == base64.b64encode(b"\x01\x02").decode("ascii")
    assert call.dropped_parallel_calls == ["done"]


def test_openai_chat_parallel_tool_calls_are_recorded():
    call = parse_openai_chat_tool_call(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"id": "call_1", "type": "function",
                             "function": {"name": "weather_get",
                                          "arguments": json.dumps({"location": "Paris"})}},
                            {"id": "call_2", "type": "function",
                             "function": {"name": "done", "arguments": "{}"}},
                        ]
                    }
                }
            ]
        }
    )
    assert call.name == "weather_get"
    assert call.arguments == {"location": "Paris"}
    assert call.id == "call_1"
    assert call.dropped_parallel_calls == ["done"]


# --------------------------------------------------------------------------- #
# Reasoning threading: Anthropic extended thinking
# --------------------------------------------------------------------------- #
class _FakeAnthropicMessages:
    def __init__(self, responses: list, bodies: list[dict]):
        self._responses = list(responses)
        self._bodies = bodies

    def create(self, **kwargs):
        self._bodies.append(kwargs)
        return self._responses.pop(0)


class _FakeAnthropicClient:
    def __init__(self, responses: list, bodies: list[dict]):
        self.messages = _FakeAnthropicMessages(responses, bodies)


def _anthropic_response(blocks: list) -> SimpleNamespace:
    return SimpleNamespace(content=blocks)


def _thinking_block(text: str, signature: str) -> SimpleNamespace:
    return SimpleNamespace(type="thinking", thinking=text, signature=signature)


def _tool_use_block(name: str, arguments: dict, block_id: str = "toolu_1") -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=arguments)


def test_native_anthropic_enables_thinking_with_auto_tool_choice():
    bodies: list[dict] = []
    agent = agents.LLMAgent(
        provider="anthropic",
        model="claude-sonnet-5",
        tool_mode="native",
        native_reasoning=True,
    )
    agent._client = _FakeAnthropicClient(
        [
            _anthropic_response(
                [
                    _thinking_block("plan the weather lookup", "sig-1"),
                    _tool_use_block("weather_get", {"location": "Paris"}),
                ]
            ),
            _anthropic_response([_tool_use_block("done", {}, "toolu_2")]),
        ],
        bodies,
    )
    agent.reset("Check the weather in Paris.")

    action = agent.act(_obs(0))
    assert action == "weather get --location Paris"
    body = bodies[0]
    # Preferred request: adaptive extended thinking + top-level output_config
    # effort (NOT a fixed budget_tokens, and effort is NOT nested in thinking).
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "high"}
    # Default tool_choice policy is auto for every provider.
    assert body["tool_choice"] == {"type": "auto"}
    # max_tokens is raised well above the fixed fallback budget so adaptive
    # thinking has room before the tool_use block.
    assert body["max_tokens"] > 8192
    # Temperature must be omitted (default) when thinking is enabled.
    assert not _has_temp(body)

    agent.act(_obs(1, stdout="Sunny, 22C"))
    # Step 1 replays the thinking block before the tool_use block it preceded.
    msgs = bodies[1]["messages"]
    assistant_turns = [m for m in msgs if m["role"] == "assistant"]
    assert assistant_turns
    first = assistant_turns[0]["content"]
    assert first[0]["type"] == "thinking"
    assert first[0]["signature"] == "sig-1"
    assert first[1]["type"] == "tool_use"


def test_native_anthropic_effort_override_replaces_default():
    """Anthropic's own default is already high; the flag must still override.

    Pinning both families to the same level is the point of the flag, so it has
    to be able to move Claude DOWN as well as leave it where it is.
    """
    bodies: list[dict] = []
    agent = agents.LLMAgent(
        provider="anthropic",
        model="claude-sonnet-5",
        tool_mode="native",
        native_reasoning=True,
        reasoning_effort="medium",
    )
    agent._client = _FakeAnthropicClient(
        [_anthropic_response([_tool_use_block("weather_get", {"location": "Paris"})])],
        bodies,
    )
    agent.reset("Check the weather in Paris.")
    agent.act(_obs(0))

    assert bodies[0]["output_config"] == {"effort": "medium"}


def test_native_anthropic_keeps_default_effort_when_unset():
    bodies: list[dict] = []
    agent = agents.LLMAgent(
        provider="anthropic",
        model="claude-sonnet-5",
        tool_mode="native",
        native_reasoning=True,
    )
    agent._client = _FakeAnthropicClient(
        [_anthropic_response([_tool_use_block("weather_get", {"location": "Paris"})])],
        bodies,
    )
    agent.reset("Check the weather in Paris.")
    agent.act(_obs(0))

    assert bodies[0]["output_config"] == {"effort": "high"}


@pytest.mark.parametrize(
    "effort,wire",
    [
        # Canonical rungs Anthropic expresses verbatim (thinking ON, adaptive).
        ("low", "low"), ("medium", "medium"), ("high", "high"),
        # 'minimal' is NOT "off" for Claude: it is the lowest real thinking rung
        # and folds onto 'low' (thinking ON, adaptive), same as an explicit 'low'.
        ("minimal", "low"),
        # Rungs Anthropic's low/medium/high vocabulary lacks, folded onto the
        # nearest real tier: 'xhigh'/'max' -> 'high' (no rejected call). Only the
        # 'none' off rung is NOT here -- it routes through the disabled-thinking
        # path; see test_native_anthropic_off_effort_*.
        ("xhigh", "high"), ("max", "high"),
    ],
)
def test_anthropic_maps_canonical_effort_onto_wire_vocabulary(effort, wire):
    bodies: list[dict] = []
    agent = agents.LLMAgent(
        provider="anthropic",
        model="claude-sonnet-5",
        tool_mode="native",
        native_reasoning=True,
        reasoning_effort=effort,
    )
    agent._client = _FakeAnthropicClient(
        [_anthropic_response([_tool_use_block("weather_get", {"location": "Paris"})])],
        bodies,
    )
    agent.reset("Check the weather in Paris.")
    agent.act(_obs(0))

    assert bodies[0]["output_config"] == {"effort": wire}
    cfg = agent.debug_run_config()
    assert cfg["reasoning_effort"] == effort
    assert cfg["reasoning_effort_effective"] == wire


# --------------------------------------------------------------------------- #
# "No think": ONE --llm-reasoning-effort knob turning thinking off
# --------------------------------------------------------------------------- #
# 'none' means "turn thinking off". The families express that differently, so
# the shared knob maps per provider on the native path:
#   * gemini - clamps to MINIMAL (covered by test_gemini_effort_level_mapping);
#              Gemini has no true "off".
#   * openai - passes 'none' through to reasoning.effort (documented as bypassing
#              reasoning).
#   * claude - has NO 'none' effort rung, so we send
#              thinking={"type":"disabled"} + effort=low. A plain (no-param)
#              request would NOT be off: Sonnet 5 / Opus 4.8 default to high.
# NOTE 'minimal' is NOT "off" for Claude -- it is the lowest *real* thinking rung
# and maps to 'low' (thinking ON); see the canonical-vocabulary test above and
# test_native_anthropic_minimal_effort_uses_low_adaptive below.


@pytest.mark.parametrize("effort", ["none"])
def test_native_anthropic_off_effort_disables_thinking(effort):
    bodies: list[dict] = []
    agent = agents.LLMAgent(
        provider="anthropic",
        model="claude-sonnet-5",
        tool_mode="native",
        native_reasoning=True,
        reasoning_effort=effort,
    )
    agent._client = _FakeAnthropicClient(
        [_anthropic_response([_tool_use_block("weather_get", {"location": "Paris"})])],
        bodies,
    )
    agent.reset("Check the weather in Paris.")
    action = agent.act(_obs(0))

    assert action == "weather get --location Paris"
    body = bodies[0]
    # Thinking OFF explicitly + effort pinned to the low floor -- NOT adaptive
    # /high, and NOT a plain (no-param) request that leaves the model at high.
    assert body["thinking"] == {"type": "disabled"}
    assert body["output_config"] == {"effort": "low"}
    # Thinking off -> temperature is sent and max_tokens is not inflated for a
    # thinking budget the request will not use.
    assert _has_temp(body)
    assert body["max_tokens"] == agent._response_max_tokens


def test_native_anthropic_off_effort_disables_thinking_without_native_reasoning():
    """'off' does not depend on native_reasoning.

    A plain native request would leave Sonnet 5 / Opus 4.8 at their default high
    effort, so 'none' must still disable thinking explicitly even when reasoning
    threading is off.
    """
    bodies: list[dict] = []
    agent = agents.LLMAgent(
        provider="anthropic",
        model="claude-sonnet-5",
        tool_mode="native",
        native_reasoning=False,
        reasoning_effort="none",
    )
    agent._client = _FakeAnthropicClient(
        [_anthropic_response([_tool_use_block("weather_get", {"location": "Paris"})])],
        bodies,
    )
    agent.reset("Check the weather in Paris.")
    agent.act(_obs(0))

    assert bodies[0]["thinking"] == {"type": "disabled"}
    assert bodies[0]["output_config"] == {"effort": "low"}


def test_native_anthropic_minimal_effort_uses_low_adaptive():
    """'minimal' is NOT "off" for Claude.

    Unlike 'none', 'minimal' is the lowest *real* thinking rung: it must send
    adaptive thinking ON at effort=low (identical to an explicit 'low'), NOT
    thinking={"type":"disabled"}.
    """
    bodies: list[dict] = []
    agent = agents.LLMAgent(
        provider="anthropic",
        model="claude-sonnet-5",
        tool_mode="native",
        native_reasoning=True,
        reasoning_effort="minimal",
    )
    agent._client = _FakeAnthropicClient(
        [_anthropic_response([_tool_use_block("weather_get", {"location": "Paris"})])],
        bodies,
    )
    agent.reset("Check the weather in Paris.")
    agent.act(_obs(0))

    body = bodies[0]
    # Thinking ON (adaptive) at the low floor -- NOT disabled.
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "low"}
    assert agent.debug_run_config()["reasoning_effort_effective"] == "low"


def test_run_config_effective_effort_anthropic_minimal_no_threading():
    # 'minimal' rides the normal adaptive path, so (like 'low') it is only
    # effective when native_reasoning threading is on; without it, nothing is
    # sent and the effective level is blank.
    agent = agents.LLMAgent(
        provider="anthropic",
        model="claude-sonnet-5",
        tool_mode="native",
        native_reasoning=False,
        reasoning_effort="minimal",
    )
    assert agent.debug_run_config()["reasoning_effort_effective"] == ""


def test_native_anthropic_off_effort_forced_tool_choice_sends_any():
    """Disabled thinking accepts a forced tool choice.

    The forbid-forced rule applies only to manual *enabled* thinking, so an
    'off' run still gets the forced surface it asked for.
    """
    bodies: list[dict] = []
    agent = agents.LLMAgent(
        provider="anthropic",
        model="claude-sonnet-5",
        tool_mode="native",
        native_reasoning=True,
        reasoning_effort="none",
    )
    agent._native_tool_choice = "forced"
    agent._client = _FakeAnthropicClient(
        [_anthropic_response([_tool_use_block("weather_get", {"location": "Paris"})])],
        bodies,
    )
    agent.reset("Check the weather in Paris.")
    agent.act(_obs(0))

    assert bodies[0]["thinking"] == {"type": "disabled"}
    assert bodies[0]["tool_choice"] == {"type": "any"}


def test_native_anthropic_off_effort_falls_back_to_plain_when_rejected():
    """A backend that rejects thinking={"type":"disabled"} degrades to a plain
    (no-param) request instead of failing, and remembers the downgrade."""
    bodies: list[dict] = []
    success = _anthropic_response(
        [_tool_use_block("weather_get", {"location": "Paris"})]
    )

    class _RejectDisabledMessages:
        def create(self, **kwargs):
            bodies.append(kwargs)
            if kwargs.get("thinking") == {"type": "disabled"}:
                raise RuntimeError(
                    "thinking: disabled is not supported on this model"
                )
            return success

    agent = agents.LLMAgent(
        provider="vertex-anthropic",
        model="claude-sonnet-5",
        tool_mode="native",
        native_reasoning=True,
        reasoning_effort="none",
    )
    agent._client = SimpleNamespace(messages=_RejectDisabledMessages())
    agent.reset("Check the weather in Paris.")

    action = agent.act(_obs(0))
    assert action == "weather get --location Paris"
    # First the disabled request, then a plain fallback (no reasoning params).
    assert len(bodies) == 2
    assert bodies[0]["thinking"] == {"type": "disabled"}
    assert "thinking" not in bodies[1]
    assert "output_config" not in bodies[1]
    # The downgrade is remembered (so later steps skip the rejected param) and
    # pins the ladder to plain so native_reasoning cannot re-enable thinking.
    assert agent._anthropic_disable_thinking_supported is False
    assert agent._anthropic_reasoning_mode == "plain"


def test_openai_none_effort_passes_through_to_bypass_reasoning(monkeypatch):
    """OpenAI supports a real 'none' effort (bypass reasoning); it rides the same
    native reasoning opt-in as every other level."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    bodies: list[dict] = []
    _mock_urlopen(
        monkeypatch,
        [_responses_payload_with_reasoning("weather_get", {"location": "Paris"})],
        bodies,
    )
    agent = agents.LLMAgent(
        provider="openai",
        model="gpt-5.2",
        base_url="https://example.test/v1",
        tool_mode="native",
        native_reasoning=True,
        reasoning_effort="none",
    )
    agent.reset("Check the weather in Paris.")
    agent.act(_obs(0))

    assert bodies[0]["reasoning"] == {"effort": "none"}


def test_run_config_effective_effort_anthropic_off_native():
    agent = agents.LLMAgent(
        provider="anthropic",
        model="claude-sonnet-5",
        tool_mode="native",
        native_reasoning=True,
        reasoning_effort="none",
    )
    cfg = agent.debug_run_config()
    assert cfg["reasoning_effort"] == "none"
    assert cfg["reasoning_effort_effective"] == "none"


def test_run_config_effective_effort_anthropic_off_native_no_threading():
    # 'off' is sent even without native_reasoning threading, so it is effective.
    agent = agents.LLMAgent(
        provider="anthropic",
        model="claude-sonnet-5",
        tool_mode="native",
        native_reasoning=False,
        reasoning_effort="none",
    )
    assert agent.debug_run_config()["reasoning_effort_effective"] == "none"


def test_run_config_effective_effort_anthropic_off_text_blank():
    # The text path is deliberately not wired for 'off'; nothing is sent.
    agent = agents.LLMAgent(
        provider="anthropic",
        model="claude-sonnet-5",
        tool_mode="text",
        reasoning_effort="none",
    )
    cfg = agent.debug_run_config()
    assert cfg["reasoning_effort"] == "none"
    assert cfg["reasoning_effort_effective"] == ""


def test_run_config_effective_effort_anthropic_off_blank_after_fallback():
    # Once the disabled params are rejected and the flag cleared, the effective
    # level drops to "" (the request no longer carries an off signal).
    agent = agents.LLMAgent(
        provider="anthropic",
        model="claude-sonnet-5",
        tool_mode="native",
        native_reasoning=True,
        reasoning_effort="none",
    )
    agent._anthropic_disable_thinking_supported = False
    assert agent.debug_run_config()["reasoning_effort_effective"] == ""


def test_native_anthropic_reasoning_off_keeps_auto_choice_no_thinking():
    """Turning reasoning off must no longer flip tool_choice as a side effect.

    ``native_reasoning`` and ``_native_tool_choice`` used to move together, which
    made them impossible to ablate independently.
    """
    bodies: list[dict] = []
    agent = agents.LLMAgent(
        provider="anthropic",
        model="claude-haiku-4-5",
        tool_mode="native",
        native_reasoning=False,
    )
    agent._client = _FakeAnthropicClient(
        [_anthropic_response([_tool_use_block("weather_get", {"location": "Paris"})])],
        bodies,
    )
    agent.reset("Check the weather in Paris.")
    action = agent.act(_obs(0))

    assert action == "weather get --location Paris"
    body = bodies[0]
    assert body["tool_choice"] == {"type": "auto"}
    assert "thinking" not in body
    assert "output_config" not in body
    # Non-thinking path still sends temperature (model not in the omit set).
    assert _has_temp(body)


def test_native_forced_tool_choice_opt_in_per_provider(monkeypatch):
    """The forced arm of the ablation reaches every provider's wire format."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    bodies: list[dict] = []
    _mock_urlopen(
        monkeypatch, [_chat_tool_call_payload("weather_get", {"location": "Paris"})], bodies
    )
    openai_agent = agents.LLMAgent(
        provider="openai",
        model="legacy-openai-chat-model",
        base_url="https://example.test/v1",
        tool_mode="native",
    )
    openai_agent._native_tool_choice = "forced"
    openai_agent.reset("Check the weather in Paris.")
    openai_agent.act(_obs(0))
    assert bodies[0]["tool_choice"] == "required"

    anthropic_bodies: list[dict] = []
    anthropic_agent = agents.LLMAgent(
        provider="anthropic",
        model="claude-sonnet-5",
        tool_mode="native",
        native_reasoning=True,
    )
    anthropic_agent._native_tool_choice = "forced"
    anthropic_agent._client = _FakeAnthropicClient(
        [_anthropic_response([_tool_use_block("weather_get", {"location": "Paris"})])],
        anthropic_bodies,
    )
    anthropic_agent.reset("Check the weather in Paris.")
    anthropic_agent.act(_obs(0))
    # Adaptive thinking is documented to support a forced tool choice.
    assert anthropic_bodies[0]["thinking"] == {"type": "adaptive"}
    assert anthropic_bodies[0]["tool_choice"] == {"type": "any"}


def test_native_forced_choice_stays_auto_on_non_adaptive_rungs():
    """Guards against a silent degradation to forced-with-no-thinking.

    Only *adaptive* thinking accepts a forced tool choice. On the "budget" rung
    the request would be rejected with a message that
    ``_is_anthropic_reasoning_param_unsupported`` also matches, so
    ``_anthropic_create`` would read it as "backend lacks the reasoning params",
    step down again, and land on "plain" -- leaving the run forced with thinking
    disabled and nothing in the log to show it.
    """
    agent = agents.LLMAgent(
        provider="anthropic",
        model="claude-sonnet-5",
        tool_mode="native",
        native_reasoning=True,
    )
    agent._native_tool_choice = "forced"

    agent._anthropic_reasoning_mode = "adaptive"
    assert agent._anthropic_tool_choice() == {"type": "any"}

    agent._anthropic_reasoning_mode = "budget"
    assert agent._anthropic_tool_choice() == {"type": "auto"}

    # "plain" disables thinking entirely, so forcing is safe again.
    agent._anthropic_reasoning_mode = "plain"
    assert agent._anthropic_tool_choice() == {"type": "any"}


def test_native_anthropic_reasoning_param_fallback_ladder():
    """adaptive -> fixed budget -> plain when the backend rejects each rung."""
    bodies: list[dict] = []
    success = _anthropic_response(
        [_tool_use_block("weather_get", {"location": "Paris"})]
    )

    class _LadderMessages:
        def create(self, **kwargs):
            bodies.append(kwargs)
            # Adaptive request carries output_config; reject it first.
            if "output_config" in kwargs:
                raise RuntimeError("output_config: unsupported parameter")
            # Fixed-budget request carries thinking; reject it next.
            if "thinking" in kwargs:
                raise RuntimeError("thinking: not supported on this model")
            # Plain request (no reasoning params) succeeds.
            return success

    agent = agents.LLMAgent(
        provider="vertex-anthropic",
        model="claude-sonnet-5",
        tool_mode="native",
        native_reasoning=True,
    )
    agent._client = SimpleNamespace(messages=_LadderMessages())
    agent.reset("Check the weather in Paris.")

    action = agent.act(_obs(0))
    assert action == "weather get --location Paris"
    # Three attempts within one call: adaptive, fixed budget, then plain.
    assert len(bodies) == 3
    assert bodies[0]["thinking"] == {"type": "adaptive"}
    assert bodies[0]["output_config"] == {"effort": "high"}
    assert bodies[1]["thinking"] == {"type": "enabled", "budget_tokens": 8192}
    assert "output_config" not in bodies[1]
    assert "thinking" not in bodies[2]
    assert "output_config" not in bodies[2]
    # The plain fallback re-adds temperature (model not in the omit set).
    assert _has_temp(bodies[2])
    # The downgrade is remembered so later steps skip the rejected rungs.
    assert agent._anthropic_reasoning_mode == "plain"


# ---- Gemini thinking_level floor: idempotent under --parallel-workers ----
#
# Regression test for a concurrency race. The floor lives in a process-global
# dict shared by every worker thread. The buggy escalation stepped UP from the
# CURRENT shared floor, so 20 workers that all sent MINIMAL and got rejected at
# once would each bump it a rung -- a thundering herd that pushed the floor
# MINIMAL->LOW->MEDIUM->HIGH. Late threads then found the floor already at HIGH
# (nothing above it), escalation returned False, and their MINIMAL rejection was
# re-raised, failing those tasks. The fix keys escalation off the level the
# request ACTUALLY sent, so every MINIMAL-rejected worker only ever raises the
# floor to LOW, no matter how many escalate concurrently.


def _gemini_client_rejecting_minimal_concurrent(sent_levels, lock, barrier):
    """Fake client where MINIMAL 400s, and every first-wave MINIMAL request is
    held at ``barrier`` so all workers get rejected from a shared floor=None --
    the exact thundering herd the fix has to survive."""

    def _generate_content(*, model, contents, config):
        level = config.thinking_config.thinking_level
        name = getattr(level, "name", level)
        with lock:
            sent_levels.append(name)
        if name == "MINIMAL":
            try:
                barrier.wait(timeout=10)
            except threading.BrokenBarrierError:  # pragma: no cover - timing
                pass
            raise RuntimeError(
                "400 INVALID_ARGUMENT thinking_level MINIMAL is not supported "
                "by this model"
            )
        return SimpleNamespace(candidates=[])  # accepted; body unused here

    return SimpleNamespace(
        models=SimpleNamespace(generate_content=_generate_content)
    )


def test_gemini_thinking_floor_escalation_idempotent_under_concurrency():
    from google.genai import types

    workers = 20
    agents._GEMINI_MODEL_MIN_THINKING_LEVEL.pop("gemini-3-pro", None)
    sent_levels: list[str] = []
    lock = threading.Lock()
    # Barrier sized to the herd: floor is only written during escalation, which
    # happens AFTER release, so until all 20 are waiting the floor stays None and
    # every worker computes MINIMAL -- guaranteeing the barrier fills (no
    # deadlock) and the escalation storm starts from a single shared floor.
    barrier = threading.Barrier(workers)
    client = _gemini_client_rejecting_minimal_concurrent(
        sent_levels, lock, barrier
    )

    agent = agents.LLMAgent(
        provider="gemini", model="gemini-3-pro", tool_mode="native",
        reasoning_effort="none",
    )

    errors: list[Exception] = []

    def _one(_i):
        try:
            agent._gemini_generate_content(
                client, types, ["hi"],
                dict(temperature=0.0, max_output_tokens=16),
            )
        except Exception as exc:  # noqa: BLE001 - record instead of failing thread
            with lock:
                errors.append(exc)

    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_one, range(workers)))
        final_floor = agents._GEMINI_MODEL_MIN_THINKING_LEVEL.get("gemini-3-pro")
    finally:
        agents._GEMINI_MODEL_MIN_THINKING_LEVEL.pop("gemini-3-pro", None)

    # No worker re-raised: the race never exhausted the ladder.
    assert errors == [], [str(e) for e in errors]
    # The floor converged to exactly LOW -- one rung above the rejected MINIMAL,
    # never overshooting because 20 siblings escalated at once.
    assert final_floor == "LOW", final_floor
    # Nothing above LOW was ever put on the wire: the herd healed MINIMAL->LOW
    # and stopped, instead of climbing to MEDIUM/HIGH.
    assert set(sent_levels) <= {"MINIMAL", "LOW"}, set(sent_levels)
    # The race really was exercised: the herd tried MINIMAL first, then healed.
    assert "MINIMAL" in sent_levels
    assert "LOW" in sent_levels
