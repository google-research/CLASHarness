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

from __future__ import annotations

import json
import os
import sys
import types
from typing import Any

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from clasharness.evaluation import llm_retry as _llm_retry
from clasharness.evaluation.llm_retry import (
    anthropic_kwargs_carry_temperature as _has_temp,
    anthropic_kwargs_temperature as _temp_of,
)
from clasharness.scenario.llm_client import (
    AnthropicScenarioLLMClient,
    MockScenarioLLMClient,
    OpenAICompatibleScenarioLLMClient,
    ScenarioLLMError,
    VertexScenarioLLMClient,
    _close_unbalanced_json,
    _escape_invalid_backslashes,
    _json_correction_message,
    _repair_json_string,
    extract_json_object,
)


@pytest.fixture(autouse=True)
def _reset_anthropic_sampling_learned():
    """Isolate the process-wide 'model rejects sampling params' learned set.

    ``anthropic_rejects_sampling_params`` consults a module-level set that
    ``note_anthropic_sampling_rejected`` populates at runtime and that is shared
    across the generation client and the judge. Clear it around every test so a
    rejection learned in one test cannot flip an unknown model to the proactive
    path in another and silently change its call count.
    """
    _llm_retry._ANTHROPIC_SAMPLING_DEPRECATED_LEARNED.clear()
    yield
    _llm_retry._ANTHROPIC_SAMPLING_DEPRECATED_LEARNED.clear()


def test_extract_json_object_from_fenced_response() -> None:
    assert extract_json_object('```json\n{"ok": true}\n```') == {"ok": True}


def test_extract_json_object_tolerates_raw_control_chars_in_strings() -> None:
    # Models routinely emit raw newlines/tabs inside HTML/rich-text string
    # values, which strict json.loads rejects. extract_json_object must recover
    # the value with the control characters preserved.
    raw = '{"html": "<div>\n\t<p>hi</p>\n</div>"}'
    assert extract_json_object(raw) == {"html": "<div>\n\t<p>hi</p>\n</div>"}


def test_extract_json_object_repairs_trailing_comma() -> None:
    # Trailing commas survive neither json.loads nor raw_decode; the repair pass
    # must strip them so the object parses.
    raw = '{"files": [{"path": "a.txt", "content": "x"},], "ok": true,}'
    assert extract_json_object(raw) == {
        "files": [{"path": "a.txt", "content": "x"}],
        "ok": True,
    }


def test_repair_json_string_escapes_control_chars_in_strings() -> None:
    # Direct unit check of the P0-4 escaping: raw control chars inside a quoted
    # value are escaped so the repaired text is valid under strict json.loads,
    # and the decoded value round-trips to the original control characters.
    raw = '{"html": "line1\nline2\tend\rzed"}'
    repaired = _repair_json_string(raw)
    assert json.loads(repaired) == {"html": "line1\nline2\tend\rzed"}


def test_repair_json_string_strips_trailing_commas() -> None:
    assert json.loads(_repair_json_string('{"a": [1, 2,], "b": 3,}')) == {
        "a": [1, 2],
        "b": 3,
    }


def test_extract_json_object_ignores_trailing_delimiter_garbage() -> None:
    # Repro of the real failure: the model emits a complete object, then loops
    # on closing delimiters, producing "Extra data" that broke json.loads and
    # the rfind('}')-based fallback. raw_decode must take the leading object.
    raw = '{"workspace": {"files": [{"path": "a.txt", "content": "x"}]}}\n}\n  ]\n}\n}'
    assert extract_json_object(raw) == {
        "workspace": {"files": [{"path": "a.txt", "content": "x"}]}
    }


def test_extract_json_object_ignores_trailing_prose() -> None:
    raw = '{"ok": true}\n\nHere is the JSON you requested. Hope it helps!'
    assert extract_json_object(raw) == {"ok": True}


def test_extract_json_object_repairs_invalid_backslash_escape() -> None:
    # Repro of the real production failure: the model wrote a Markdown backtick as
    # a backslash-backtick (e.g. run \`asset list\`), which is an INVALID JSON
    # escape. strict json.loads rejects it with "Invalid \escape", and neither the
    # control-char repair nor the inner-quote scanner touches it. The backslash
    # pass must double the invalid escape so the value parses and round-trips to
    # the literal characters the model emitted.
    raw = r'{"remediation": "run \`asset list\` now"}'
    assert extract_json_object(raw) == {"remediation": r"run \`asset list\` now"}


def test_extract_json_object_backslash_repair_preserves_valid_escapes() -> None:
    # The backslash repair must be lossless: only invalid escapes are doubled;
    # valid ones (\n \t \" \/ \uXXXX) pass through untouched. Here \q is invalid
    # (doubled to a literal backslash) while the rest survive as real escapes.
    raw = r'{"a": "line\nbreak", "b": "quote \" slash \/", "c": "\u00e9", "bad": "x\qy"}'
    assert extract_json_object(raw) == {
        "a": "line\nbreak",
        "b": 'quote " slash /',
        "c": "\u00e9",
        "bad": r"x\qy",
    }


def test_escape_invalid_backslashes_is_idempotent_on_valid_json() -> None:
    # A string that is already valid JSON must be returned unchanged, so the pass
    # never corrupts a response that only needed a different repair.
    valid = r'{"a": "l1\nl2 \"q\" \\ end \u00e9"}'
    assert _escape_invalid_backslashes(valid) == valid


def test_json_correction_message_targets_invalid_backslash_escape() -> None:
    # An "Invalid \escape" failure must get backslash-specific guidance, not the
    # default "unescaped double quote" nudge -- the wrong nudge misdirects the
    # model and burns every retry repeating the same mistake.
    exc = ScenarioLLMError(
        "JSON Parse Failure: Invalid \\escape: line 3 column 449 (char 479)."
    )
    msg = _json_correction_message(exc)
    assert "INVALID BACKSLASH ESCAPE" in msg
    assert "backtick" in msg
    assert "UNESCAPED DOUBLE QUOTE" not in msg


def test_json_correction_message_targets_unescaped_quote() -> None:
    # A generic delimiter syntax error keeps the quote-focused guidance.
    exc = ScenarioLLMError("Expecting ':' delimiter: line 2 column 10 (char 20)")
    msg = _json_correction_message(exc)
    assert "UNESCAPED DOUBLE QUOTE" in msg
    assert "INVALID BACKSLASH ESCAPE" not in msg


def test_json_correction_message_covers_terminal_degeneration() -> None:
    # A delimiter error is NOT always an unescaped quote: the observed
    # generate_vlm_layout_spec stall was terminal degeneration -- the model
    # looped on closing-delimiter fragments after a complete object. The nudge
    # must also tell the model to stop after the single root-closing brace, or
    # self-correction keeps burning retries on the wrong fix.
    exc = ScenarioLLMError("Expecting ',' delimiter: line 41 column 1 (char 4631)")
    msg = _json_correction_message(exc)
    assert "UNESCAPED DOUBLE QUOTE" in msg  # keep cause A
    assert "EXTRA CHARACTERS AFTER THE OBJECT" in msg  # add cause B
    assert "do NOT repeat braces" in msg


def test_extract_json_object_recovers_terminal_degeneration() -> None:
    # Real production failure (generate_vlm_layout_spec): the model emits a
    # complete object whose final value is a CSS string ending in '}', then
    # loops on closing-delimiter fragments ('2);\\n}"', '}"', '} "', '} \\n}"')
    # instead of cleanly closing the root. The garbage sits BEFORE the root's
    # closing brace, so raw_decode cannot take a leading value and the
    # quote/backslash repairs cannot strip it. The truncate-and-close repair
    # must recover every field with the trailing garbage discarded.
    raw = (
        '{\n'
        '  "system_name": "LedgerLock Console",\n'
        '  "custom_css_rules": ["border: 1px solid #f43f5e;"],\n'
        '  "common_styles_css": "/* c */\\n.ascii-border {\\n  '
        'box-shadow: inset 0 0 4px rgba(244, 63, 94, 0.2);\\n}"\n'
        '2);\\n}"\n'
        '}"\n'
        '} "\n'
        '} \\n}"\n'
        '}\n'
        '}'
    )
    data = extract_json_object(raw)
    assert data["system_name"] == "LedgerLock Console"
    assert data["custom_css_rules"] == ["border: 1px solid #f43f5e;"]
    # The last real field is intact and none of the degenerate fragments leaked.
    assert data["common_styles_css"].endswith("rgba(244, 63, 94, 0.2);\n}")
    assert "2);" not in data["common_styles_css"][-4:]


def test_extract_json_object_recovers_truncated_object() -> None:
    # The response was cut off mid-string, so there is no closing brace at all.
    # The old rfind('}')-based guard bailed out with "did not contain a JSON
    # object"; recovery must keep the complete prefix and close the containers.
    raw = (
        '{\n'
        '  "summary": "done",\n'
        '  "rounds": [\n'
        '    {"round_id": "r01", "instruction": "read the confi'
    )
    data = extract_json_object(raw)
    assert data["summary"] == "done"
    assert data["rounds"][0]["round_id"] == "r01"


def test_close_unbalanced_json_preserves_valid_object() -> None:
    # The recovery pass must never mutate text that is already valid JSON (note
    # the literal '}' inside the "c" string value must not be miscounted).
    valid = '{"a": {"b": [1, 2, 3]}, "c": "x}y"}'
    assert _close_unbalanced_json(valid) == valid


def test_close_unbalanced_json_rejects_empty_collapse() -> None:
    # Prose that merely starts with '{' would "recover" to '{}', which is not a
    # meaningful recovery. The pass must return the input unchanged so the caller
    # still raises its normal "not valid JSON" error instead of accepting '{}'.
    prose = "{ this is only prose with no fields at all"
    assert _close_unbalanced_json(prose) == prose
    try:
        extract_json_object(prose)
        assert False, "expected ScenarioLLMError for unrecoverable prose"
    except ScenarioLLMError:
        pass


def test_extract_json_object_raises_when_nothing_recoverable() -> None:
    # A response truncated before even the first complete field yields no usable
    # data and must still raise rather than silently return an empty object.
    try:
        extract_json_object('{"a": "b')
        assert False, "expected ScenarioLLMError for truncated input"
    except ScenarioLLMError:
        pass


def test_mock_client_generates_required_layers() -> None:
    client = MockScenarioLLMClient()
    payload = {
        "seed": {
            "slug": "mock_case",
            "title": "Mock Case",
            "domain": "ops",
            "target_rounds": 24,
        }
    }
    assert "summary" in client.generate_layer("layer0_narrative", payload)
    assert len(client.generate_layer("layer1_workspace", payload)["files"]) >= 7
    assert len(client.generate_layer("layer2_sessions", payload)["sessions"]) >= 5
    layer3 = client.generate_layer("layer3_eval", payload)
    assert len(layer3["rounds"]) == 24
    assert sum(1 for item in layer3["rounds"] if item["kind"] == "mc") == 7
    assert len(client.generate_layer("layer4_dynamic", payload)["updates"]) == 3


def test_openai_compatible_client_supports_responses_api(monkeypatch) -> None:
    captured = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"output_text": "{\"ok\": true}"}).encode("utf-8")

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _Response()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = OpenAICompatibleScenarioLLMClient(
        model="Kimi-K2.5",
        base_url="https://proxy.example/v1",
        api="responses",
    )
    assert client.generate_layer("layer0_narrative", {"seed": {}}) == {"ok": True}
    assert captured["url"] == "https://proxy.example/v1/responses"
    assert captured["body"]["model"] == "Kimi-K2.5"
    assert isinstance(captured["body"]["input"], str)
    assert captured["body"]["max_output_tokens"] == 16384


def test_openai_compatible_client_supports_curl_transport(monkeypatch) -> None:
    captured = {}

    class _Proc:
        returncode = 0
        stdout = json.dumps({"output_text": "{\"ok\": true}"}) + "\n200"
        stderr = ""

    def fake_run(cmd, input, capture_output, text):
        captured["cmd"] = cmd
        captured["input"] = input
        captured["capture_output"] = capture_output
        captured["text"] = text
        return _Proc()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("subprocess.run", fake_run)

    client = OpenAICompatibleScenarioLLMClient(
        model="Kimi-K2.5",
        base_url="https://proxy.example/v1",
        api="responses",
        transport="curl",
    )
    assert client.generate_layer("layer0_narrative", {"seed": {}}) == {"ok": True}
    assert "curl" in captured["cmd"][0]
    assert "https://proxy.example/v1/responses" in captured["cmd"]
    assert "--data-binary" in captured["cmd"]
    assert captured["input"]


def test_client_retries_invalid_json(monkeypatch) -> None:
    responses = iter(
        [
            {"output_text": "{\"sessions\": ["},
            {"output_text": "{\"sessions\": []}"},
        ]
    )

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(next(responses)).encode("utf-8")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: _Response())

    client = OpenAICompatibleScenarioLLMClient(
        model="gpt-5.2",
        base_url="https://proxy.example/v1",
        api="responses",
        retries=1,
    )
    assert client.generate_layer("layer2_sessions", {"seed": {}}) == {"sessions": []}


def test_chat_json_mode_is_opt_in(monkeypatch) -> None:
    captured = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "{\"ok\": true}"}}]}).encode("utf-8")

    def fake_urlopen(req, timeout):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _Response()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = OpenAICompatibleScenarioLLMClient(
        model="claude-sonnet-4.6",
        base_url="http://127.0.0.1:4000/v1",
        api="chat",
    )
    assert client.generate_layer("layer0_narrative", {"seed": {}}) == {"ok": True}
    assert "response_format" not in captured["body"]


def test_vertex_scenario_llm_client(monkeypatch) -> None:
    captured = {}

    class MockModels:
        def generate_content(self, model, contents, config):
            captured["model"] = model
            captured["contents"] = contents
            captured["config"] = config
            
            class MockResponse:
                text = '{"ok": true}'
                
            return MockResponse()

    class MockClient:
        def __init__(self, **kwargs):
            captured["client_args"] = kwargs
            self.models = MockModels()

    # Mock the genai client module
    from google import genai
    monkeypatch.setattr(genai, "Client", MockClient)

    client = VertexScenarioLLMClient(
        model="gemini-2.5-flash",
        project="test-project",
        location="us-east4",
    )
    
    assert client.generate_layer("layer0_narrative", {"seed": {}}) == {"ok": True}
    assert captured["client_args"]["project"] == "test-project"
    assert captured["client_args"]["location"] == "us-east4"
    assert captured["model"] == "gemini-2.5-flash"
    assert captured["config"].max_output_tokens == 16384
    assert captured["config"].temperature == 0.2
    assert "layer0_narrative" in captured["contents"][0].parts[0].text


def test_vertex_client_rebuilds_on_stale_connection_error(monkeypatch) -> None:
    """Stale-client transport errors must self-heal by rebuilding the client.

    A cached google-genai Client whose httpx connection pool has already been
    used can raise "Context has already been used to create a Connection, it
    cannot be mutated again" when reused under a concurrent burst (all workers
    hitting the first layer of a new batch of scenarios at once). The client must
    discard that thread's stale client and rebuild a fresh one on retry so the
    call still succeeds. Regression for the conflict_layer0a_cast
    batch-transition failure that stalled parallel generation.
    """
    constructions = {"n": 0}
    calls = {"n": 0}

    class MockModels:
        def generate_content(self, model, contents, config):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError(
                    "Context has already been used to create a Connection, "
                    "it cannot be mutated again"
                )

            class MockResponse:
                text = '{"ok": true}'

            return MockResponse()

    class MockClient:
        def __init__(self, **kwargs):
            constructions["n"] += 1
            self.models = MockModels()

    from google import genai

    monkeypatch.setattr(genai, "Client", MockClient)
    # Do not actually sleep through the retry backoff.
    monkeypatch.setattr(
        "clasharness.scenario.llm_client.time.sleep", lambda *a, **k: None
    )

    client = VertexScenarioLLMClient(
        model="gemini-2.5-flash",
        project="test-project",
        location="us-east4",
        retries=3,
    )
    # One client is built eagerly during construction.
    assert constructions["n"] == 1

    result = client.generate_layer("layer0_narrative", {"seed": {}})

    assert result == {"ok": True}
    assert calls["n"] == 2  # failed once on the stale client, then retried
    assert constructions["n"] == 2  # stale client discarded and rebuilt fresh


def test_vertex_client_backs_off_longer_on_rate_limit(monkeypatch) -> None:
    """HTTP 429 / RESOURCE_EXHAUSTED must trigger a long capacity cool-off.

    A 429 "No backend available" means the Vertex serving pool could not place
    the request -- it is not a content error the model can self-correct. The
    client must (a) NOT feed it back as a self-correction turn (which would
    waste an attempt and grow the request) and (b) back off far longer than the
    ~8s transport retry so it stops hammering an over-subscribed pool. Regression
    for the conflict_layer3_round_commands stall that spun 31 times on ~8s
    retries.
    """
    constructions = {"n": 0}
    calls = {"n": 0, "last_contents_len": 0}
    sleeps: list[float] = []

    class MockModels:
        def generate_content(self, model, contents, config):
            calls["n"] += 1
            # Record turn count so we can prove self-correction did NOT append a
            # model/user turn pair for the rate-limit error.
            calls["last_contents_len"] = len(contents)
            if calls["n"] == 1:
                raise RuntimeError(
                    "429 RESOURCE_EXHAUSTED: No backend available for the model"
                )

            class MockResponse:
                text = '{"ok": true}'

            return MockResponse()

    class MockClient:
        def __init__(self, **kwargs):
            constructions["n"] += 1
            self.models = MockModels()

    from google import genai

    monkeypatch.setattr(genai, "Client", MockClient)
    # Capture backoff durations instead of actually sleeping through them.
    monkeypatch.setattr(
        "clasharness.scenario.llm_client.time.sleep",
        lambda s, *a, **k: sleeps.append(s),
    )

    client = VertexScenarioLLMClient(
        model="gemini-2.5-flash",
        project="test-project",
        location="us-east4",
        retries=3,
    )

    result = client.generate_layer("layer3_round_commands", {"seed": {}})

    assert result == {"ok": True}
    assert calls["n"] == 2  # failed once on 429, then retried and succeeded
    # Self-correction must be skipped: the retry re-sends the same single user
    # turn rather than appending a model turn + correction turn.
    assert calls["last_contents_len"] == 1
    # The stale client is discarded and rebuilt on the capacity path too.
    assert constructions["n"] == 2
    # Long capacity backoff: attempt 0 sleeps 15 * 2**0 = 15s (+1-5s jitter),
    # well above the ~8s transport-error cap.
    assert len(sleeps) == 1
    assert sleeps[0] >= 15.0


def _vision_audit_client(monkeypatch, failures: list[Exception]):
    """Build a Vertex client whose vision audit fails ``failures`` then succeeds.

    Returns ``(client, calls, constructions, sleeps, logs)`` so a test can assert
    on retry count, client rebuilds, backoff length and log output.
    """
    constructions = {"n": 0}
    calls = {"n": 0}
    sleeps: list[float] = []
    logs: list[str] = []

    class MockModels:
        def generate_content(self, model, contents, config):
            calls["n"] += 1
            if calls["n"] <= len(failures):
                raise failures[calls["n"] - 1]

            class MockResponse:
                text = '{"legible": true, "seen_text": "v4.2", "reason": "ok"}'

            return MockResponse()

    class MockClient:
        def __init__(self, **kwargs):
            constructions["n"] += 1
            self.models = MockModels()

    from google import genai

    monkeypatch.setattr(genai, "Client", MockClient)
    monkeypatch.setattr(
        "clasharness.scenario.llm_client.time.sleep",
        lambda s, *a, **k: sleeps.append(s),
    )

    client = VertexScenarioLLMClient(
        model="gemini-2.5-flash",
        project="test-project",
        location="us-east4",
        retries=3,
        logger=logs.append,
    )
    return client, calls, constructions, sleeps, logs


def test_vision_audit_retries_and_logs_transient_error(monkeypatch) -> None:
    """The vision fidelity audit must retry and log like any other Vertex call.

    ``_audit_fidelity_safe`` treats *any* audit exception as a benign pass, so an
    unretried, unlogged audit degrades invisibly: the asset is silently recorded
    as verified when it was never actually checked. Auditing every decisive asset
    (~480/run instead of ~20) makes that erosion likely enough to matter, so the
    audit must survive a transient transport error and leave a trace when it does.
    """
    client, calls, constructions, _sleeps, logs = _vision_audit_client(
        monkeypatch,
        [RuntimeError(
            "Context has already been used to create a Connection, "
            "it cannot be mutated again"
        )],
    )
    assert constructions["n"] == 1  # one built eagerly at construction

    result = client.audit_image_fidelity(
        image_bytes=b"\x89PNG-not-a-real-image",
        mime_type="image/png",
        expected_token="v4.2",
    )

    assert result == {
        "legible": True,
        "seen_text": "v4.2",
        "reason": "ok",
        "skipped": False,
    }
    assert calls["n"] == 2  # failed once, retried, succeeded
    assert constructions["n"] == 2  # stale client discarded and rebuilt
    # The failure is no longer silent: both the attempt and the error are logged.
    joined = "\n".join(logs)
    assert "vision_fidelity" in joined
    assert "Context has already been used" in joined


def test_vision_audit_backs_off_longer_on_rate_limit(monkeypatch) -> None:
    """A 429 during the audit takes the long capacity cool-off, not the ~8s path.

    Auditing every decisive asset adds a few hundred vision calls to a run that
    already saturates the serving pool, so the audit is a realistic place to meet
    RESOURCE_EXHAUSTED. It must cool off on the same terms as ``generate_layer``
    rather than hammering an over-subscribed pool.
    """
    client, calls, _constructions, sleeps, logs = _vision_audit_client(
        monkeypatch,
        [RuntimeError("429 RESOURCE_EXHAUSTED: No backend available for the model")],
    )

    result = client.audit_image_fidelity(
        image_bytes=b"\x89PNG-not-a-real-image",
        mime_type="image/png",
        expected_token="v4.2",
    )

    assert result["legible"] is True
    assert calls["n"] == 2
    # attempt 0 sleeps 15 * 2**0 = 15s (+1-5s jitter), above the ~8s transport cap.
    assert len(sleeps) == 1
    assert sleeps[0] >= 15.0
    assert "Rate limit / capacity exhausted (vision_fidelity)" in "\n".join(logs)


def test_vision_audit_propagates_error_after_retry_budget(monkeypatch) -> None:
    """A genuinely broken audit still raises, so the caller can treat it as a pass.

    Retrying must not turn a persistent failure into a false verdict: the error
    propagates to ``_audit_fidelity_safe``, which converts it into an explicit
    ``skipped`` result. That keeps a flaky audit from ever *downgrading* a good
    decisive asset -- the retry only removes the silence, not the safety valve.
    """
    boom = RuntimeError("vision backend permanently unavailable")
    client, calls, _constructions, _sleeps, _logs = _vision_audit_client(
        monkeypatch, [boom] * 99
    )

    try:
        client.audit_image_fidelity(
            image_bytes=b"x", mime_type="image/png", expected_token="v4.2"
        )
    except RuntimeError as exc:
        assert "permanently unavailable" in str(exc)
    else:
        raise AssertionError("expected the audit error to propagate")

    assert calls["n"] == 4  # retries=3 -> 4 total attempts, then give up


# ---- Claude, official API or Vertex AI (AnthropicScenarioLLMClient) ----


def _fake_anthropic_vertex(monkeypatch, captured: dict, replies):
    """Inject a fake ``anthropic`` module recording BOTH client constructors.

    ``AnthropicVertex`` (ADC path) and ``Anthropic`` (official-API path) are both
    stubbed, since the client chooses between them; ``captured['backend']``
    records which one was actually built.

    ``replies`` is either a single reply string or a list; each ``messages.create``
    pops the next entry, and any ``Exception`` instance is raised instead of
    returned (to exercise the temperature-downgrade / retry paths). Every call's
    kwargs are appended to ``captured['calls']``.
    """
    if isinstance(replies, str):
        replies = [replies]
    queue = list(replies)
    captured["calls"] = []
    captured["constructions"] = 0

    class _Messages:
        def create(self, **kwargs):
            captured["calls"].append(kwargs)
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return {"content": [{"type": "text", "text": item}]}

    class _FakeVertex:
        def __init__(self, project_id=None, region=None):
            captured["backend"] = "vertex"
            captured["project_id"] = project_id
            captured["region"] = region
            captured["constructions"] += 1
            self.messages = _Messages()

        def close(self):
            pass

    class _FakeAnthropic:
        def __init__(self, api_key=None):
            captured["backend"] = "official"
            captured["api_key"] = api_key
            captured["constructions"] += 1
            self.messages = _Messages()

        def close(self):
            pass

    fake_mod = types.ModuleType("anthropic")
    fake_mod.AnthropicVertex = _FakeVertex
    fake_mod.Anthropic = _FakeAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)


def test_anthropic_vertex_client_generate_layer(monkeypatch) -> None:
    captured: dict = {}
    _fake_anthropic_vertex(monkeypatch, captured, '{"ok": true}')

    client = AnthropicScenarioLLMClient(
        model="claude-sonnet-4@20250514",
        project="test-project",
        location="us-east5",
    )
    assert client.generate_layer("layer0_narrative", {"seed": {}}) == {"ok": True}

    # Routed through AnthropicVertex with the configured project/region/model.
    assert captured["backend"] == "vertex"
    assert captured["project_id"] == "test-project"
    assert captured["region"] == "us-east5"
    call = captured["calls"][0]
    assert call["model"] == "claude-sonnet-4@20250514"
    assert call["max_tokens"] == 16384
    assert _temp_of(call) == 0.2
    assert call["messages"][0]["role"] == "user"
    assert "layer0_narrative" in call["messages"][0]["content"]


def test_anthropic_client_uses_official_api_when_key_env_is_set(
    monkeypatch,
) -> None:
    # api_key_env naming a populated variable selects anthropic.Anthropic, and
    # the ADC constructor must not be touched at all -- a run that quietly fell
    # back to Vertex would fail much later, on credentials the user does not have.
    captured: dict = {}
    _fake_anthropic_vertex(monkeypatch, captured, '{"ok": true}')
    monkeypatch.setenv("MY_CLAUDE_KEY", "sk-ant-test")

    client = AnthropicScenarioLLMClient(
        model="claude-sonnet-5", api_key_env="MY_CLAUDE_KEY"
    )
    assert client.generate_layer("layer0_narrative", {"seed": {}}) == {"ok": True}

    assert captured["backend"] == "official"
    assert captured["api_key"] == "sk-ant-test"
    assert "project_id" not in captured
    assert captured["calls"][0]["model"] == "claude-sonnet-5"


def test_anthropic_client_falls_back_to_vertex_when_key_env_is_empty(
    monkeypatch,
) -> None:
    # An api_key_env pointing at an unset (or blank) variable is NOT a request
    # for the official API -- it is a missing key, and the documented behaviour
    # is to keep the ADC path rather than construct a keyless official client
    # that would 401 on the first call.
    captured: dict = {}
    _fake_anthropic_vertex(monkeypatch, captured, '{"ok": true}')
    monkeypatch.setenv("MY_CLAUDE_KEY", "   ")

    client = AnthropicScenarioLLMClient(
        model="claude-sonnet-4@20250514",
        project="test-project",
        api_key_env="MY_CLAUDE_KEY",
    )
    assert client.generate_layer("layer0_narrative", {"seed": {}}) == {"ok": True}
    assert captured["backend"] == "vertex"
    assert captured["project_id"] == "test-project"


def test_anthropic_client_backend_is_pinned_at_construction(monkeypatch) -> None:
    # The backend is resolved once in __post_init__, so a key exported AFTER the
    # client exists must not flip later worker threads onto the official API --
    # that would silently mix two endpoints inside one generated corpus.
    captured: dict = {}
    _fake_anthropic_vertex(monkeypatch, captured, '{"ok": true}')
    monkeypatch.delenv("MY_CLAUDE_KEY", raising=False)

    client = AnthropicScenarioLLMClient(
        model="claude-sonnet-4@20250514",
        project="test-project",
        api_key_env="MY_CLAUDE_KEY",
    )
    assert captured["backend"] == "vertex"
    assert captured["constructions"] == 1

    monkeypatch.setenv("MY_CLAUDE_KEY", "sk-ant-late")
    client._reset_client()  # force a rebuild, as a fresh worker thread would
    assert client.generate_layer("layer0_narrative", {"seed": {}}) == {"ok": True}
    assert captured["constructions"] == 2  # it really did rebuild
    assert captured["backend"] == "vertex"


def test_anthropic_client_log_names_the_active_backend(monkeypatch) -> None:
    # 'project=None location=us-east5' on an official-API run reads like a
    # broken Vertex run; the init line must say which backend is live.
    monkeypatch.setenv("MY_CLAUDE_KEY", "sk-ant-test")
    lines: list[str] = []
    _fake_anthropic_vertex(monkeypatch, {}, '{"ok": true}')
    AnthropicScenarioLLMClient(
        model="claude-sonnet-5",
        api_key_env="MY_CLAUDE_KEY",
        logger=lines.append,
    )
    assert any(
        line.startswith("Anthropic Client Initialized:")
        and "api_key_env=MY_CLAUDE_KEY" in line
        for line in lines
    ), lines

    lines.clear()
    _fake_anthropic_vertex(monkeypatch, {}, '{"ok": true}')
    AnthropicScenarioLLMClient(
        model="claude-sonnet-4@20250514",
        project="test-project",
        logger=lines.append,
    )
    assert any(
        line.startswith("AnthropicVertex Client Initialized:")
        and "project=test-project" in line
        for line in lines
    ), lines


def test_anthropic_vertex_client_threads_timeout_into_create(monkeypatch) -> None:
    # Regression: a large --max_output_tokens must NOT hit the anthropic SDK's
    # "streaming required for operations >10 minutes" guard. The client threads
    # an explicit timeout into messages.create() (mirroring the eval agent),
    # which disables that guard. 65536 tokens would otherwise trip it
    # (3600*65536/128000 >> 600s). Assert the (default) timeout reaches create.
    captured: dict = {}
    _fake_anthropic_vertex(monkeypatch, captured, '{"ok": true}')

    client = AnthropicScenarioLLMClient(
        model="claude-sonnet-5",
        project="test-project",
        max_output_tokens=65536,
    )
    assert client.generate_layer("layer0_narrative", {"seed": {}}) == {"ok": True}
    call = captured["calls"][0]
    assert call["max_tokens"] == 65536
    assert call["timeout"] == 600.0

    # The vision-audit path shares _create_kwargs, so it is covered too.
    captured_audit: dict = {}
    _fake_anthropic_vertex(
        monkeypatch,
        captured_audit,
        '{"legible": true, "seen_text": "v1", "reason": "ok"}',
    )
    client2 = AnthropicScenarioLLMClient(
        model="claude-sonnet-5",
        project="test-project",
        timeout_seconds=900.0,
    )
    client2.audit_image_fidelity(
        image_bytes=b"\x89PNG", mime_type="image/png", expected_token="v1"
    )
    assert captured_audit["calls"][0]["timeout"] == 900.0


def test_anthropic_vertex_client_defaults_region_us_east5(monkeypatch) -> None:
    # location unset (and no region env) -> Claude backend uses us-east5, not
    # Gemini's us-central1/global.
    for var in ("VERTEX_LOCATION", "GOOGLE_CLOUD_LOCATION"):
        monkeypatch.delenv(var, raising=False)
    captured: dict = {}
    _fake_anthropic_vertex(monkeypatch, captured, '{"ok": true}')

    client = AnthropicScenarioLLMClient(
        model="claude-sonnet-4@20250514",
        project="test-project",
    )
    assert client.generate_layer("layer0_narrative", {"seed": {}}) == {"ok": True}
    assert captured["region"] == "us-east5"


def test_anthropic_vertex_client_drops_temperature_on_rejection(monkeypatch) -> None:
    # A model that 400s on an explicit temperature: the client retries the same
    # request immediately without it (no wasted attempt), matching the eval path.
    captured: dict = {}
    _fake_anthropic_vertex(
        monkeypatch,
        captured,
        [
            RuntimeError("temperature is not supported for this model"),
            '{"ok": true}',
        ],
    )

    client = AnthropicScenarioLLMClient(
        model="claude-reasoning@20250101",
        project="test-project",
    )
    assert client.generate_layer("layer0_narrative", {"seed": {}}) == {"ok": True}
    # Two create calls: first with temperature, second without it.
    assert len(captured["calls"]) == 2
    assert _has_temp(captured["calls"][0])
    assert not _has_temp(captured["calls"][1])


def test_anthropic_vertex_client_drops_temperature_on_deprecated_wording(
    monkeypatch,
) -> None:
    # The exact 400 wording newer Claude models return ("`temperature` is
    # deprecated for this model.") must be recognized so the client drops
    # temperature and retries instead of failing the layer. Uses a model NOT in
    # the statically-known reject list (claude-sonnet-5 is now omitted
    # proactively -- see test_anthropic_vertex_client_omits_temperature_for_known
    # _model) so the REACTIVE detection of this wording is what is exercised here.
    captured: dict = {}
    _fake_anthropic_vertex(
        monkeypatch,
        captured,
        [
            RuntimeError(
                "Error code: 400 - {'type': 'error', 'error': {'type': "
                "'invalid_request_error', 'message': '`temperature` is "
                "deprecated for this model.'}}"
            ),
            '{"ok": true}',
        ],
    )

    client = AnthropicScenarioLLMClient(
        model="claude-newthinker@20250101", project="test-project"
    )
    assert client.generate_layer("layer0_narrative", {"seed": {}}) == {"ok": True}
    assert len(captured["calls"]) == 2
    assert _has_temp(captured["calls"][0])
    assert not _has_temp(captured["calls"][1])
    # The reactive drop is remembered process-wide so later calls omit it.
    assert _llm_retry.anthropic_rejects_sampling_params(
        "claude-newthinker@20250101"
    )


def test_anthropic_vertex_client_omits_temperature_for_known_model(
    monkeypatch,
) -> None:
    # claude-sonnet-5 is KNOWN to reject an explicit temperature, so the client
    # omits it from the very first request -- no wasted 400, a single create call
    # carrying no temperature. This is the fix for the flood of "temperature is
    # deprecated" 400s seen when generating with sonnet-5.
    captured: dict = {}
    _fake_anthropic_vertex(monkeypatch, captured, '{"ok": true}')

    client = AnthropicScenarioLLMClient(
        model="claude-sonnet-5", project="test-project"
    )
    # Learned at construction, before any request is sent.
    assert client._omit_temperature is True
    assert client.generate_layer("layer0_narrative", {"seed": {}}) == {"ok": True}
    assert len(captured["calls"]) == 1
    assert not _has_temp(captured["calls"][0])
    # Other kwargs (and the streaming-guard timeout) are unaffected.
    assert captured["calls"][0]["max_tokens"] == 16384
    assert captured["calls"][0]["timeout"] == 600.0


def test_anthropic_vertex_client_omits_temperature_when_learned_elsewhere(
    monkeypatch,
) -> None:
    # A rejection learned by ANOTHER component (e.g. the judge) is shared via the
    # process-wide set, so a freshly-constructed generation client for that model
    # omits temperature proactively too -- the learning crosses components.
    _llm_retry.note_anthropic_sampling_rejected("claude-shared@1")
    captured: dict = {}
    _fake_anthropic_vertex(monkeypatch, captured, '{"ok": true}')

    client = AnthropicScenarioLLMClient(
        model="claude-shared@1", project="test-project"
    )
    assert client._omit_temperature is True
    assert client.generate_layer("layer0_narrative", {"seed": {}}) == {"ok": True}
    assert len(captured["calls"]) == 1
    assert not _has_temp(captured["calls"][0])


def test_anthropic_vertex_client_heals_temperature_race_loser(monkeypatch) -> None:
    # Race across generation workers on the shared client: a concurrent worker
    # flips ``_omit_temperature`` True AFTER we built our (temperature-carrying)
    # kwargs, and our create fails with wording the detector does NOT itself
    # recognize. The client must still drop temperature and retry INLINE -- no
    # outer retry, no backoff sleep -- because the flag is already set. Uses a
    # model NOT in the statically-known reject list so temperature IS sent on the
    # first request (the precondition for this race); a known model would omit it
    # proactively and the race could not arise.
    captured: dict = {"calls": []}
    state: dict = {"client": None}

    class _Messages:
        def create(self, **kwargs):
            captured["calls"].append(dict(kwargs))
            if len(captured["calls"]) == 1:
                # A peer worker already learned the rejection.
                state["client"]._omit_temperature = True
                raise RuntimeError("Error code: 400 - unrelated wording")
            return {"content": [{"type": "text", "text": '{"ok": true}'}]}

    class _FakeVertex:
        def __init__(self, project_id=None, region=None):
            self.messages = _Messages()

        def close(self):
            pass

    fake_mod = types.ModuleType("anthropic")
    fake_mod.AnthropicVertex = _FakeVertex
    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)

    sleeps: list = []
    monkeypatch.setattr(
        "clasharness.scenario.llm_client.time.sleep", lambda s: sleeps.append(s)
    )

    client = AnthropicScenarioLLMClient(
        model="claude-newthinker@20250102", project="test-project"
    )
    state["client"] = client
    assert client.generate_layer("layer0_narrative", {"seed": {}}) == {"ok": True}
    # Healed inline: exactly two create calls (fail + temperature-dropped retry)
    # and NO backoff sleep (the old code re-raised, took the outer backoff).
    assert len(captured["calls"]) == 2
    assert _has_temp(captured["calls"][0])
    assert not _has_temp(captured["calls"][1])
    assert sleeps == []


def test_anthropic_vertex_client_vision_audit(monkeypatch) -> None:
    captured: dict = {}
    _fake_anthropic_vertex(
        monkeypatch,
        captured,
        '{"legible": true, "seen_text": "v4.2", "reason": "ok"}',
    )

    client = AnthropicScenarioLLMClient(
        model="claude-sonnet-4@20250514",
        project="test-project",
    )
    result = client.audit_image_fidelity(
        image_bytes=b"\x89PNG-not-a-real-image",
        mime_type="image/png",
        expected_token="v4.2",
    )
    assert result == {
        "legible": True,
        "seen_text": "v4.2",
        "reason": "ok",
        "skipped": False,
    }
    # The image is sent as a base64 Anthropic image block alongside the ask.
    content = captured["calls"][0]["messages"][0]["content"]
    blocks = {block["type"]: block for block in content}
    assert blocks["image"]["source"]["type"] == "base64"
    assert blocks["image"]["source"]["media_type"] == "image/png"
    assert blocks["image"]["source"]["data"]  # non-empty base64 payload
    assert "v4.2" in blocks["text"]["text"]


# ---- OpenAI-compatible / bedrock-gpt vision fidelity ----


def test_openai_compatible_responses_vision_audit(monkeypatch) -> None:
    # bedrock-gpt shape: Responses API vision with an input_image data URL and no
    # temperature (temperature=None), returning the shared verdict dict.
    captured: dict = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {"output_text": '{"legible": true, "seen_text": "v4.2", "reason": "ok"}'}
            ).encode("utf-8")

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _Response()

    monkeypatch.setenv("BEDROCK_OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = OpenAICompatibleScenarioLLMClient(
        model="openai.gpt-5.5",
        base_url="https://bedrock.example/openai/v1",
        api_key_env="BEDROCK_OPENAI_API_KEY",
        api="responses",
        temperature=None,
    )
    result = client.audit_image_fidelity(
        image_bytes=b"\x89PNG-not-a-real-image",
        mime_type="image/png",
        expected_token="v4.2",
    )
    assert result == {
        "legible": True,
        "seen_text": "v4.2",
        "reason": "ok",
        "skipped": False,
    }
    assert captured["url"] == "https://bedrock.example/openai/v1/responses"
    # temperature=None omits the field (gpt-5-class 400s on an explicit value).
    assert "temperature" not in captured["body"]
    content = captured["body"]["input"][0]["content"]
    kinds = {part["type"] for part in content}
    assert kinds == {"input_text", "input_image"}
    image = next(p for p in content if p["type"] == "input_image")
    assert image["image_url"].startswith("data:image/png;base64,")


def test_openai_compatible_chat_api_skips_vision_audit(monkeypatch) -> None:
    # Chat-only proxies do not serve /responses vision; the audit skips (benign
    # pass) rather than risk a false downgrade. No network call is made.
    def explode(*a, **k):  # pragma: no cover - must never be called
        raise AssertionError("chat api must not issue a vision request")

    monkeypatch.setattr("urllib.request.urlopen", explode)

    client = OpenAICompatibleScenarioLLMClient(
        model="some-chat-model",
        base_url="https://proxy.example/v1",
        api="chat",
    )
    result = client.audit_image_fidelity(
        image_bytes=b"x", mime_type="image/png", expected_token="v4.2"
    )
    assert result["skipped"] is True
    assert result["legible"] is True


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__]))
