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

"""Tests that --capture-log records usage + reasoning for every provider.

Historically only the Gemini path populated ``_last_response_meta`` (so
``thinking_text`` and ``usage`` were always empty for OpenAI/Anthropic runs even
with ``--capture-log``). These tests lock in that the OpenAI-compatible and
Anthropic paths now capture the exact token ``usage`` (including cached /
reasoning sub-counts) and any reasoning text the model returned, and that it
flows through into the sidecar interaction record.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import train_eval.agents as agents


# ---------------------------------------------------------------------------
# Pure meta-extraction helpers
# ---------------------------------------------------------------------------
def test_openai_response_meta_chat_shape_preserves_usage_and_reasoning():
    payload = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": "ls", "reasoning_content": "think first"},
            }
        ],
        "usage": {
            "prompt_tokens": 1200,
            "completion_tokens": 30,
            "prompt_tokens_details": {"cached_tokens": 1024},
            "completion_tokens_details": {"reasoning_tokens": 256},
        },
    }

    meta = agents._openai_response_meta(payload)

    assert meta["finish_reason"] == "stop"
    assert meta["thinking_text"] == "think first"
    # Raw usage is preserved verbatim so cached/reasoning survive for costing.
    assert meta["usage"]["prompt_tokens_details"]["cached_tokens"] == 1024
    assert meta["usage"]["completion_tokens_details"]["reasoning_tokens"] == 256


def test_openai_response_meta_responses_shape_reads_status_and_summary():
    payload = {
        "object": "response",
        "status": "completed",
        "output": [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "plan: read"}],
            },
            {"type": "message", "content": [{"type": "output_text", "text": "DONE"}]},
        ],
        "usage": {
            "input_tokens": 2000,
            "input_tokens_details": {"cached_tokens": 1900},
            "output_tokens": 40,
            "output_tokens_details": {"reasoning_tokens": 300},
        },
    }

    meta = agents._openai_response_meta(payload)

    assert meta["finish_reason"] == "completed"
    assert meta["thinking_text"] == "plan: read"
    assert meta["usage"]["input_tokens_details"]["cached_tokens"] == 1900
    assert meta["usage"]["output_tokens_details"]["reasoning_tokens"] == 300


def test_openai_response_meta_tolerates_missing_usage():
    meta = agents._openai_response_meta({"choices": [{"message": {"content": "ls"}}]})
    assert meta["usage"] == {}
    assert meta["thinking_text"] == ""
    assert meta["finish_reason"] is None


def test_anthropic_response_meta_dict_shape():
    resp = {
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 500,
            "output_tokens": 20,
            "cache_read_input_tokens": 480,
            "cache_creation_input_tokens": 0,
        },
        "content": [
            {"type": "thinking", "thinking": "reasoning here"},
            {"type": "text", "text": "ls -la"},
        ],
    }

    meta = agents._anthropic_response_meta(resp)

    assert meta["finish_reason"] == "end_turn"
    assert meta["thinking_text"] == "reasoning here"
    assert meta["usage"]["cache_read_input_tokens"] == 480


def test_anthropic_response_meta_object_shape():
    resp = SimpleNamespace(
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=10,
            cache_read_input_tokens=90,
            cache_creation_input_tokens=0,
        ),
        content=[
            SimpleNamespace(type="thinking", thinking="obj thoughts"),
            SimpleNamespace(type="text", text="cat file"),
        ],
    )

    meta = agents._anthropic_response_meta(resp)

    assert meta["thinking_text"] == "obj thoughts"
    assert meta["usage"]["cache_read_input_tokens"] == 90
    assert meta["usage"]["output_tokens"] == 10


def test_anthropic_response_meta_uses_sdk_serializer_for_thinking_tokens():
    """model_dump() output is preserved verbatim, incl. output_tokens_details.

    Extended thinking is billed inside ``output_tokens``; the only record of the
    split is ``usage.output_tokens_details.thinking_tokens``. A hand-picked
    allowlist used to drop it, hiding Claude's reasoning cost. Real SDK usage
    objects expose ``model_dump`` (pydantic), so prefer it.
    """

    class _Usage:
        def model_dump(self):
            return {
                "input_tokens": 100,
                "output_tokens": 3148,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "output_tokens_details": {"thinking_tokens": 2395},
            }

    resp = SimpleNamespace(
        stop_reason="tool_use",
        usage=_Usage(),
        content=[SimpleNamespace(type="text", text="file_create")],
    )

    meta = agents._anthropic_response_meta(resp)

    assert meta["usage"]["output_tokens"] == 3148
    assert meta["usage"]["output_tokens_details"]["thinking_tokens"] == 2395


def test_anthropic_response_meta_fallback_keeps_output_tokens_details():
    """When usage exposes no serializer, the fallback still keeps the detail.

    Guards against a future SDK object shape that lacks ``model_dump`` but still
    carries ``output_tokens_details`` -- the reasoning split must survive.
    """
    resp = SimpleNamespace(
        stop_reason="tool_use",
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=500,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            output_tokens_details=SimpleNamespace(thinking_tokens=380),
        ),
        content=[SimpleNamespace(type="text", text="asset list")],
    )

    meta = agents._anthropic_response_meta(resp)

    assert meta["usage"]["output_tokens"] == 500
    assert meta["usage"]["output_tokens_details"]["thinking_tokens"] == 380


# ---------------------------------------------------------------------------
# Call-site wiring: meta is set and flows into the interaction record
# ---------------------------------------------------------------------------
def test_openai_call_site_sets_meta_and_flows_into_record():
    agent = agents.LLMAgent(
        provider="openai",
        model="gpt-5.6-luna",
        capture_interactions=True,
        request_retries=0,
    )
    payload = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": "ls", "reasoning_content": "let me think"},
            }
        ],
        "usage": {
            "prompt_tokens": 800,
            "completion_tokens": 12,
            "prompt_tokens_details": {"cached_tokens": 512},
            "completion_tokens_details": {"reasoning_tokens": 64},
        },
    }
    # Stub the transport so no network/key is needed.
    agent._openai_request = lambda *a, **k: payload  # type: ignore[assignment]

    text = agent._openai_response_text()
    assert text == "ls"

    meta = agent._last_response_meta
    assert meta is not None
    assert meta["answer_text"] == "ls"
    assert meta["thinking_text"] == "let me think"
    assert meta["usage"]["prompt_tokens_details"]["cached_tokens"] == 512

    record = agent._build_interaction_record(
        attempt_idx=0,
        input_snapshot={"history_len": 1},
        raw="ls",
        normalized="ls",
        ok=True,
        reason=None,
    )
    resp = record["response"]
    assert resp["answer_text"] == "ls"
    assert resp["thinking_text"] == "let me think"
    assert resp["usage"]["completion_tokens_details"]["reasoning_tokens"] == 64


def test_anthropic_call_site_sets_meta_from_fake_client():
    agent = agents.LLMAgent(
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        capture_interactions=True,
        request_retries=0,
    )
    resp = SimpleNamespace(
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=300,
            output_tokens=8,
            cache_read_input_tokens=256,
            cache_creation_input_tokens=0,
        ),
        content=[
            SimpleNamespace(type="thinking", thinking="anthropic reasoning"),
            SimpleNamespace(type="text", text="asset list"),
        ],
    )

    class _Messages:
        def create(self, **_kwargs):
            return resp

    agent._client = SimpleNamespace(messages=_Messages())
    agent._messages = []

    text = agent._anthropic_response_text()
    assert text == "asset list"

    meta = agent._last_response_meta
    assert meta is not None
    assert meta["answer_text"] == "asset list"
    assert meta["thinking_text"] == "anthropic reasoning"
    assert meta["usage"]["cache_read_input_tokens"] == 256


def test_openai_capture_off_leaves_meta_none():
    agent = agents.LLMAgent(
        provider="openai",
        model="gpt-5.6-luna",
        capture_interactions=False,
        capture_thinking=False,
        request_retries=0,
    )
    agent._openai_request = lambda *a, **k: {  # type: ignore[assignment]
        "choices": [{"finish_reason": "stop", "message": {"content": "ls"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }
    assert agent._openai_response_text() == "ls"
    # Default runs stay byte-for-byte unchanged: no meta captured.
    assert agent._last_response_meta is None


# ---------------------------------------------------------------------------
# Cross-provider usage normalisation
# ---------------------------------------------------------------------------
def test_normalized_usage_gemini_does_not_subtract_thoughts():
    """Gemini's ``candidates`` already excludes reasoning -- subtracting is wrong.

    Verified against the logged corpus: 30,092/30,092 Gemini attempts satisfy
    ``total == prompt + candidates + thoughts``. Treating Gemini like the
    OpenAI/Anthropic shape would double-subtract and understate visible output.
    """
    n = agents._normalized_usage(
        "gemini", {"prompt": 1000, "candidates": 40, "thoughts": 300, "total": 1340}
    )
    assert n["input"] == 1000
    assert n["output_visible"] == 40  # NOT 40 - 300
    assert n["output_reasoning"] == 300
    assert n["output_billed"] == 340


def test_normalized_usage_openai_subtracts_reasoning_from_output():
    """OpenAI's ``output_tokens`` is the billed total and contains reasoning."""
    n = agents._normalized_usage(
        "bedrock-gpt",
        {
            "input_tokens": 2000,
            "output_tokens": 500,
            "output_tokens_details": {"reasoning_tokens": 300},
        },
    )
    assert n["input"] == 2000
    assert n["output_visible"] == 200
    assert n["output_reasoning"] == 300
    assert n["output_billed"] == 500


def test_normalized_usage_anthropic_reads_thinking_tokens():
    """Anthropic reports the same quantity under a *different* key name."""
    n = agents._normalized_usage(
        "vertex-anthropic",
        {
            "input_tokens": 900,
            "output_tokens": 3148,
            "output_tokens_details": {"thinking_tokens": 2395},
        },
    )
    assert n["output_visible"] == 753
    assert n["output_reasoning"] == 2395
    assert n["output_billed"] == 3148


def test_normalized_usage_reports_none_not_zero_when_uncaptured():
    """Uncaptured reasoning must stay None so it cannot be read as "no thinking".

    This is the regression guard for the original bug: Anthropic usage without
    ``output_tokens_details`` (every pre-fix log) means "we did not capture the
    split", not "the model thought zero tokens". Collapsing that to 0 is what
    made Claude look like it never reasoned.
    """
    n = agents._normalized_usage(
        "vertex-anthropic", {"input_tokens": 900, "output_tokens": 3148}
    )
    assert n["output_reasoning"] is None
    assert n["output_reasoning"] != 0
    # Visible falls back to the billed total rather than inventing a split.
    assert n["output_visible"] == 3148
    assert n["output_billed"] == 3148


def test_normalized_usage_handles_empty_and_degenerate_usage():
    """Empty responses report None rather than a fabricated zero.

    28/30,156 logged Gemini attempts came back with ``candidates=None`` (the
    model emitted nothing); the normaliser must not turn that into 0 output.
    """
    assert agents._normalized_usage("gemini", {}) == {
        "input": None,
        "output_visible": None,
        "output_reasoning": None,
        "output_billed": None,
    }
    degenerate = agents._normalized_usage(
        "gemini", {"prompt": 30361, "candidates": None, "thoughts": None}
    )
    assert degenerate["input"] == 30361
    assert degenerate["output_visible"] is None
    assert degenerate["output_billed"] is None


def test_interaction_record_carries_normalized_usage():
    """The split reaches the sidecar so consumers need no provider knowledge."""
    agent = agents.LLMAgent(
        provider="anthropic",
        model="claude-opus-4-5-20251101",
        capture_interactions=True,
        request_retries=0,
    )
    agent._last_response_meta = {
        "answer_text": "file_create",
        "usage": {
            "input_tokens": 900,
            "output_tokens": 3148,
            "output_tokens_details": {"thinking_tokens": 2395},
        },
    }

    record = agent._build_interaction_record(
        attempt_idx=0,
        input_snapshot={"history_len": 3},
        raw="file_create",
        normalized="file_create",
        ok=True,
        reason=None,
    )

    norm = record["response"]["usage_normalized"]
    assert norm["output_reasoning"] == 2395
    assert norm["output_visible"] == 753
    # Raw usage is still preserved verbatim alongside it.
    assert record["response"]["usage"]["output_tokens"] == 3148
