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

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from clasharness.evaluation import llm_judge
from clasharness.evaluation import llm_retry as _llm_retry
from clasharness.evaluation.llm_retry import (
    anthropic_kwargs_carry_temperature as _has_temp,
    anthropic_kwargs_temperature as _temp_of,
)

from clasharness.evaluation.llm_judge import (
    JudgeResult,
    _pick_model,
    active_judge_config,
    clear_cache,
    judge,
    judge_majority,
    resolve_provider,
    sync_judge_env,
)


@pytest.fixture(autouse=True)
def _reset_anthropic_sampling_learned():
    """Isolate the process-wide 'model rejects sampling params' learned set so a
    rejection learned by one test cannot change another test's call count."""
    _llm_retry._ANTHROPIC_SAMPLING_DEPRECATED_LEARNED.clear()
    yield
    _llm_retry._ANTHROPIC_SAMPLING_DEPRECATED_LEARNED.clear()


def _clear(monkeypatch) -> None:
    monkeypatch.delenv("LLM_JUDGE_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_JUDGE_MODEL", raising=False)


def _clear_all(monkeypatch) -> None:
    # Every variable resolve_provider() consults, so a key that happens to be
    # exported in the developer's shell cannot decide which backend a test sees.
    for var in (
        "LLM_JUDGE_PROVIDER",
        "LLM_JUDGE_MODEL",
        "GEMINI_API_KEY",
        "VERTEX_PROJECT",
        "VERTEX_LOCATION",
        "ANTHROPIC_API_KEY",
        "BEDROCK_OPENAI_API_KEY",
        "BEDROCK_OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(var, raising=False)


def _fake_anthropic_vertex(monkeypatch, captured: dict, reply: str) -> None:
    """Inject a fake ``anthropic`` module exposing ``AnthropicVertex``."""

    class _Resp:
        content = [type("_C", (), {"text": reply})()]

    class _Messages:
        def create(self, **kwargs):
            captured["kwargs"] = kwargs
            return _Resp()

    class _FakeVertex:
        def __init__(self, project_id=None, region=None):
            captured["project_id"] = project_id
            captured["region"] = region
            self.messages = _Messages()

    fake_mod = types.ModuleType("anthropic")
    fake_mod.AnthropicVertex = _FakeVertex
    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)


def test_sync_judge_env_vertex_uses_generation_model(monkeypatch) -> None:
    _clear(monkeypatch)
    sync_judge_env("vertex", "gemini-3.5-flash")
    assert os.environ["LLM_JUDGE_PROVIDER"] == "gemini"
    assert os.environ["LLM_JUDGE_MODEL"] == "gemini-3.5-flash"


def test_sync_judge_env_fixes_404_fallback(monkeypatch) -> None:
    # Repro of the bug: checkers request a Claude model, but the active provider
    # is gemini, so without sync _pick_model falls back to gemini-2.5-flash
    # (which 404s on this Vertex endpoint). After syncing to the generation
    # model, the same request resolves to gemini-3.5-flash instead.
    _clear(monkeypatch)
    assert _pick_model("gemini", "claude-haiku-4-5-20251001") == "gemini-2.5-flash"
    sync_judge_env("vertex", "gemini-3.5-flash")
    assert _pick_model("gemini", "claude-haiku-4-5-20251001") == "gemini-3.5-flash"


def test_sync_judge_env_respects_explicit_override(monkeypatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("LLM_JUDGE_MODEL", "gemini-2.0-flash")
    sync_judge_env("vertex", "gemini-3.5-flash")
    assert os.environ["LLM_JUDGE_MODEL"] == "gemini-2.0-flash"


def test_sync_judge_env_anthropic(monkeypatch) -> None:
    _clear(monkeypatch)
    sync_judge_env("anthropic", "claude-haiku-4-5-20251001")
    assert os.environ["LLM_JUDGE_PROVIDER"] == "anthropic"
    assert os.environ["LLM_JUDGE_MODEL"] == "claude-haiku-4-5-20251001"


def test_sync_judge_env_mock_leaves_env_untouched(monkeypatch) -> None:
    _clear(monkeypatch)
    sync_judge_env("mock", "whatever")
    assert "LLM_JUDGE_PROVIDER" not in os.environ
    assert "LLM_JUDGE_MODEL" not in os.environ


def test_sync_judge_env_skips_non_gemini_model_for_gemini_provider(monkeypatch) -> None:
    # provider is gemini but the requested model is not a gemini model: pin the
    # provider, but leave the model to the judge default rather than pinning a
    # model the gemini backend cannot serve.
    _clear(monkeypatch)
    sync_judge_env("gemini", "claude-haiku-4-5-20251001")
    assert os.environ["LLM_JUDGE_PROVIDER"] == "gemini"
    assert "LLM_JUDGE_MODEL" not in os.environ


# ---- k-way self-consistency voting (judge_majority) ----


def test_judge_majority_takes_strict_majority(monkeypatch) -> None:
    verdicts = [
        JudgeResult(passed=True, score=1.0, reason="ok1"),
        JudgeResult(passed=True, score=0.8, reason="ok2"),
        JudgeResult(passed=False, score=0.0, reason="no"),
    ]
    calls: list[str] = []

    def fake_judge(content, criterion, *, model=None, use_cache=True,
                   temperature=0.0, cache_salt=""):
        calls.append(cache_salt)
        return verdicts[len(calls) - 1]

    monkeypatch.setattr(llm_judge, "judge", fake_judge)
    res = judge_majority("content", "crit", k=3)
    assert res.passed is True
    assert res.ok is True
    assert len(calls) == 3
    # Each vote must use a distinct cache salt so votes don't collapse/collide.
    assert len(set(calls)) == 3


def test_judge_majority_minority_pass_fails(monkeypatch) -> None:
    verdicts = iter([
        JudgeResult(passed=True, score=1.0, reason="ok"),
        JudgeResult(passed=False, score=0.0, reason="no1"),
        JudgeResult(passed=False, score=0.0, reason="no2"),
    ])
    monkeypatch.setattr(llm_judge, "judge", lambda *a, **k: next(verdicts))
    assert judge_majority("content", "crit", k=3).passed is False


def test_judge_majority_all_fail_to_run_returns_not_ok(monkeypatch) -> None:
    monkeypatch.setattr(
        llm_judge,
        "judge",
        lambda *a, **k: JudgeResult(
            passed=False, score=0.0, reason="", ok=False, error="boom"
        ),
    )
    res = judge_majority("content", "crit", k=3)
    assert res.ok is False
    assert res.passed is False


def test_judge_majority_k1_is_single_deterministic_call(monkeypatch) -> None:
    calls: list[tuple[float, str]] = []

    def fake_judge(content, criterion, *, model=None, use_cache=True,
                   temperature=0.0, cache_salt=""):
        calls.append((temperature, cache_salt))
        return JudgeResult(passed=True, score=1.0, reason="ok")

    monkeypatch.setattr(llm_judge, "judge", fake_judge)
    assert judge_majority("content", "crit", k=1).passed is True
    assert calls == [(0.0, "")]  # one deterministic, unsalted call


def test_judge_majority_default_is_three_greedy_votes(monkeypatch) -> None:
    """The DEFAULT keeps k-way voting, but at temperature 0.

    These are two independent properties and must not be collapsed into one
    another: temperature 0 buys reproducibility, k=3 absorbs the residual
    nondeterminism that greedy decoding on a hosted backend still exhibits
    (batching / MoE routing / fp reassociation). Dropping to k=1 because
    "greedy is deterministic anyway" also silently destroys the split-vote
    health signal, which is the only way to observe that jitter.
    """
    calls: list[tuple[float, str]] = []

    def fake_judge(content, criterion, *, model=None, use_cache=True,
                   temperature=0.0, cache_salt=""):
        calls.append((temperature, cache_salt))
        return JudgeResult(passed=True, score=1.0, reason="ok")

    monkeypatch.setattr(llm_judge, "judge", fake_judge)
    assert judge_majority("content", "crit").passed is True
    assert len(calls) == 3
    # Every vote greedy ...
    assert {t for t, _ in calls} == {0.0}
    # ... and separately salted, or the three votes collide in the cache and
    # degenerate into one answer repeated three times.
    assert len({salt for _, salt in calls}) == 3


def test_judge_majority_default_temperature_is_zero() -> None:
    """Reproducibility is a signature-level guarantee, not a caller convention.

    Both production call sites (effect_checker / file_checker
    ``field_semantic_match``) invoke ``judge_majority`` WITHOUT passing k or
    temperature, so the defaults here are the scored grading configuration.
    """
    import inspect

    params = inspect.signature(judge_majority).parameters
    assert params["temperature"].default == 0.0
    assert params["k"].default == 3


def test_judge_majority_default_emits_vote_annotation(monkeypatch) -> None:
    # The annotation is what eval_report's JudgeHealth parses; without it every
    # check is misfiled as single_shot and split_vote_rate reads a constant 0.
    monkeypatch.setattr(
        llm_judge,
        "judge",
        lambda *a, **k: JudgeResult(passed=True, score=1.0, reason="looks right"),
    )
    res = judge_majority("content", "crit")
    assert "[self-consistency 3/3 passed]" in res.reason


def test_judge_majority_split_at_temperature_zero_is_still_reported(
    monkeypatch,
) -> None:
    """A split at temperature 0 means the BACKEND was nondeterministic.

    That is the one thing k=3 exists to surface once sampling is off, so the
    annotation must record the true tally rather than the majority verdict.
    """
    verdicts = iter([
        JudgeResult(passed=True, score=1.0, reason="yes"),
        JudgeResult(passed=True, score=0.9, reason="yes"),
        JudgeResult(passed=False, score=0.0, reason="no"),
    ])
    monkeypatch.setattr(llm_judge, "judge", lambda *a, **k: next(verdicts))
    res = judge_majority("content", "crit")
    assert res.passed is True
    assert "[self-consistency 2/3 passed]" in res.reason


def test_judge_majority_k1_forwards_explicit_temperature(monkeypatch) -> None:
    # k=1 used to silently drop the caller's temperature; a diagnostic caller
    # asking for sampling must actually get it.
    calls: list[float] = []

    def fake_judge(content, criterion, *, model=None, use_cache=True,
                   temperature=0.0, cache_salt=""):
        calls.append(temperature)
        return JudgeResult(passed=True, score=1.0, reason="ok")

    monkeypatch.setattr(llm_judge, "judge", fake_judge)
    judge_majority("content", "crit", k=1, temperature=0.7)
    assert calls == [0.7]


def test_judge_majority_default_reuses_cache_across_identical_criteria(
    monkeypatch,
) -> None:
    """Voting costs 3 calls per distinct pair -- not 3 per grading.

    The per-vote salts are stable (``vote:0/1/2``), so a repeat of the same
    (content, criterion) pair inside one process hits all three cached entries.
    A salt that varied per invocation (a random nonce, a counter) would make the
    cache useless and every re-grade cost full price again.
    """
    _use_gemini(monkeypatch)
    calls = {"n": 0}

    def fake_gemini(content, criterion, model, temperature=0.0):
        calls["n"] += 1
        assert temperature == 0.0  # sampling must stay off for scored runs
        return JudgeResult(passed=True, score=1.0, reason="ok")

    monkeypatch.setattr(llm_judge, "_judge_gemini", fake_gemini)
    first = judge_majority("content", "crit")
    second = judge_majority("content", "crit")
    assert first.passed is second.passed is True
    assert calls["n"] == 3  # 3 for the first grading, 0 for the second


def test_judge_backend_returning_none_is_reported_not_raised(monkeypatch) -> None:
    # A backend that hands back None must surface as ok=False, not as an
    # AttributeError escaping judge() and killing the episode.
    _use_gemini(monkeypatch)
    monkeypatch.setattr(llm_judge, "_judge_gemini", lambda *a, **k: None)
    res = judge("content", "crit")
    assert res.ok is False
    assert res.passed is False
    assert "no result" in res.error


# ---- vertex-anthropic (Claude on Vertex AI) judge backend ----


def test_resolve_provider_vertex_anthropic_override(monkeypatch) -> None:
    _clear_all(monkeypatch)
    for alias in ("vertex-anthropic", "vertex-claude", "claude-vertex", "VERTEX-ANTHROPIC"):
        monkeypatch.setenv("LLM_JUDGE_PROVIDER", alias)
        assert resolve_provider() == "vertex-anthropic"


def test_resolve_provider_never_autodetects_vertex_anthropic(monkeypatch) -> None:
    # A bare VERTEX_PROJECT always means Gemini; vertex-anthropic is opt-in only.
    _clear_all(monkeypatch)
    monkeypatch.setenv("VERTEX_PROJECT", "proj-x")
    assert resolve_provider() == "gemini"


def test_pick_model_vertex_anthropic_accepts_any_claude_id(monkeypatch) -> None:
    _clear_all(monkeypatch)
    # A Vertex '@'-form Claude id is honored.
    assert (
        _pick_model("vertex-anthropic", "claude-sonnet-4@20250514")
        == "claude-sonnet-4@20250514"
    )
    # A plain dash id is now passed through verbatim too (matching the agent):
    # the judge trusts the caller's id and lets the endpoint decide resolution.
    assert (
        _pick_model("vertex-anthropic", "claude-opus-4-8") == "claude-opus-4-8"
    )
    # The 'claude' family filter is preserved: a non-Claude id (e.g. a stray
    # gemini id) is rejected -> falls back to LLM_JUDGE_MODEL (empty here, since
    # there is no vertex-anthropic default).
    assert _pick_model("vertex-anthropic", "gemini-2.5-flash") == ""
    # LLM_JUDGE_MODEL is used whenever the requested id does not match the family.
    monkeypatch.setenv("LLM_JUDGE_MODEL", "claude-opus-4@20250101")
    assert _pick_model("vertex-anthropic", None) == "claude-opus-4@20250101"
    assert (
        _pick_model("vertex-anthropic", "gemini-2.5-flash")
        == "claude-opus-4@20250101"
    )


def test_sync_judge_env_vertex_anthropic(monkeypatch) -> None:
    _clear(monkeypatch)
    sync_judge_env("vertex-anthropic", "claude-sonnet-4@20250514")
    assert os.environ["LLM_JUDGE_PROVIDER"] == "vertex-anthropic"
    assert os.environ["LLM_JUDGE_MODEL"] == "claude-sonnet-4@20250514"


def test_sync_judge_env_vertex_anthropic_pins_dash_model(monkeypatch) -> None:
    # dash-form id is now pinned verbatim for the Vertex backend, matching the
    # generation client (so a --provider=vertex-anthropic --model=claude-opus-4-8
    # run makes the judge auto-follow to the same Claude).
    _clear(monkeypatch)
    sync_judge_env("vertex-anthropic", "claude-opus-4-8")
    assert os.environ["LLM_JUDGE_PROVIDER"] == "vertex-anthropic"
    assert os.environ["LLM_JUDGE_MODEL"] == "claude-opus-4-8"


def test_sync_judge_env_vertex_anthropic_skips_non_claude_model(monkeypatch) -> None:
    # The 'claude' family filter still holds: a non-Claude id is not pinned.
    _clear(monkeypatch)
    sync_judge_env("vertex-anthropic", "gemini-2.5-flash")
    assert os.environ["LLM_JUDGE_PROVIDER"] == "vertex-anthropic"
    assert "LLM_JUDGE_MODEL" not in os.environ


def test_judge_dispatches_to_vertex_anthropic(monkeypatch) -> None:
    _clear_all(monkeypatch)
    clear_cache()
    monkeypatch.setenv("LLM_JUDGE_PROVIDER", "vertex-anthropic")
    monkeypatch.setenv("VERTEX_PROJECT", "proj-x")
    monkeypatch.setenv("VERTEX_LOCATION", "us-east5")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "claude-sonnet-4@20250514")

    captured: dict = {}
    _fake_anthropic_vertex(
        monkeypatch, captured, '{"passed": true, "score": 1.0, "reason": "ok"}'
    )

    res = judge("some content", "some criterion")
    assert res.ok is True
    assert res.passed is True
    # Routed through AnthropicVertex with the resolved project/region/model.
    assert captured["project_id"] == "proj-x"
    assert captured["region"] == "us-east5"
    assert captured["kwargs"]["model"] == "claude-sonnet-4@20250514"


def test_judge_vertex_anthropic_defaults_region_us_east5(monkeypatch) -> None:
    # VERTEX_LOCATION unset -> the Claude backend uses us-east5, not Gemini's 'us'.
    _clear_all(monkeypatch)
    clear_cache()
    monkeypatch.setenv("LLM_JUDGE_PROVIDER", "vertex-anthropic")
    monkeypatch.setenv("VERTEX_PROJECT", "proj-x")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "claude-sonnet-4@20250514")

    captured: dict = {}
    _fake_anthropic_vertex(
        monkeypatch, captured, '{"passed": false, "score": 0.0, "reason": "no"}'
    )

    res = judge("c", "crit")
    assert res.ok is True
    assert res.passed is False
    assert captured["region"] == "us-east5"


def test_judge_vertex_anthropic_omits_temperature_for_known_model(
    monkeypatch,
) -> None:
    # claude-sonnet-5 rejects an explicit temperature (400 "temperature is
    # deprecated for this model"). The judge must omit it proactively so it does
    # not fail on every call (the judge does NOT retry invalid_request 400s).
    _clear_all(monkeypatch)
    clear_cache()
    monkeypatch.setenv("LLM_JUDGE_PROVIDER", "vertex-anthropic")
    monkeypatch.setenv("VERTEX_PROJECT", "proj-x")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "claude-sonnet-5")

    captured: dict = {}
    _fake_anthropic_vertex(
        monkeypatch, captured, '{"passed": true, "score": 1.0, "reason": "ok"}'
    )

    res = judge("c", "crit")
    assert res.ok is True
    assert res.passed is True
    assert not _has_temp(captured["kwargs"])


def test_judge_vertex_anthropic_keeps_temperature_for_old_model(
    monkeypatch,
) -> None:
    # Older Claude models accept (and are reproducible with) temperature=0, so the
    # judge keeps sending it -- only the newer reject-list models omit it.
    _clear_all(monkeypatch)
    clear_cache()
    monkeypatch.setenv("LLM_JUDGE_PROVIDER", "vertex-anthropic")
    monkeypatch.setenv("VERTEX_PROJECT", "proj-x")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "claude-sonnet-4@20250514")

    captured: dict = {}
    _fake_anthropic_vertex(
        monkeypatch, captured, '{"passed": true, "score": 1.0, "reason": "ok"}'
    )

    res = judge("c", "crit")
    assert res.ok is True
    assert _temp_of(captured["kwargs"]) == 0.0


def test_judge_vertex_anthropic_drops_temperature_on_rejection(monkeypatch) -> None:
    # An UNKNOWN model that still 400s on an explicit temperature: the judge drops
    # it and retries once (and remembers it process-wide), rather than surfacing
    # the failure -- because judge() does not retry invalid_request 400s itself.
    _clear_all(monkeypatch)
    clear_cache()
    monkeypatch.setenv("LLM_JUDGE_PROVIDER", "vertex-anthropic")
    monkeypatch.setenv("VERTEX_PROJECT", "proj-x")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "claude-newjudge@20250101")

    calls: list = []

    class _Resp:
        content = [
            type("_C", (), {"text": '{"passed": true, "score": 1.0, "reason": "ok"}'})()
        ]

    class _Messages:
        def create(self, **kwargs):
            calls.append(dict(kwargs))
            if len(calls) == 1 and _has_temp(kwargs):
                raise RuntimeError(
                    "Error code: 400 - '`temperature` is deprecated for this model.'"
                )
            return _Resp()

    class _FakeVertex:
        def __init__(self, project_id=None, region=None):
            self.messages = _Messages()

    fake_mod = types.ModuleType("anthropic")
    fake_mod.AnthropicVertex = _FakeVertex
    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)

    res = judge("c", "crit")
    assert res.ok is True
    assert res.passed is True
    assert len(calls) == 2
    assert _has_temp(calls[0])
    assert not _has_temp(calls[1])
    assert _llm_retry.anthropic_rejects_sampling_params("claude-newjudge@20250101")


def test_anthropic_temperature_kwargs_bind_to_the_installed_sdk() -> None:
    # anthropic >= 1.0 removed the sampling params from messages.create(), so a
    # literal ``temperature=`` raises TypeError LOCALLY, before any request --
    # which the HTTP-400 detector never saw, so every Claude model outside the
    # hardcoded reject-list hard-failed the judge and the generation client.
    # Whatever spelling the helper picks must bind against the real signature of
    # whichever SDK major is installed.
    import inspect

    pytest.importorskip("anthropic")
    from anthropic.resources.messages import Messages

    kwargs = _llm_retry.anthropic_temperature_kwargs(0.0)
    inspect.signature(Messages.create).bind_partial(
        None, model="m", max_tokens=1, messages=[], **kwargs
    )
    # ...and it must still read back as an explicit temperature, either spelling.
    assert _llm_retry.anthropic_kwargs_carry_temperature(kwargs)
    assert _llm_retry.anthropic_kwargs_temperature(kwargs) == 0.0


def test_anthropic_sdk_typeerror_counts_as_a_temperature_rejection() -> None:
    # Belt and braces for the above: any path that still passes the parameter
    # literally must degrade through the existing drop-and-retry rather than
    # crashing the run.
    exc = TypeError(
        "Messages.create() got an unexpected keyword argument 'temperature'"
    )
    assert _llm_retry._is_anthropic_temperature_unsupported(exc)


def test_anthropic_drop_temperature_does_not_rewrite_the_sent_request() -> None:
    # ``**kwargs`` rebuilds only the TOP level, so a nested ``extra_body`` is the
    # same object the just-sent request holds. Dropping the temperature for the
    # retry must not retroactively edit that request.
    kwargs = {"model": "m", **_llm_retry.anthropic_temperature_kwargs(0.0)}
    sent_extra_body = kwargs.get("extra_body")
    _llm_retry.anthropic_drop_temperature(kwargs)
    assert not _llm_retry.anthropic_kwargs_carry_temperature(kwargs)
    if sent_extra_body is not None:
        assert sent_extra_body == {"temperature": 0.0}


# ---- run-level judge config recording (active_judge_config) ----


def test_active_judge_config_unconfigured_returns_empty(monkeypatch) -> None:
    _clear_all(monkeypatch)
    assert active_judge_config() == ("", "")


def test_active_judge_config_gemini_default(monkeypatch) -> None:
    _clear_all(monkeypatch)
    monkeypatch.setenv("VERTEX_PROJECT", "proj-x")
    provider, model = active_judge_config()
    assert provider == "gemini"
    assert model == "gemini-2.5-flash"  # provider default when LLM_JUDGE_MODEL unset


def test_active_judge_config_honors_explicit_model(monkeypatch) -> None:
    _clear_all(monkeypatch)
    monkeypatch.setenv("LLM_JUDGE_PROVIDER", "gemini")
    monkeypatch.setenv("VERTEX_PROJECT", "proj-x")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "gemini-3.1-flash-lite")
    assert active_judge_config() == ("gemini", "gemini-3.1-flash-lite")


# ---- vote annotation is machine-parseable (judge-health report keys off it) ----


def test_judge_majority_reason_carries_vote_annotation(monkeypatch) -> None:
    verdicts = iter([
        JudgeResult(passed=True, score=1.0, reason="looks right"),
        JudgeResult(passed=True, score=0.9, reason="also right"),
        JudgeResult(passed=False, score=0.0, reason="nope"),
    ])
    monkeypatch.setattr(llm_judge, "judge", lambda *a, **k: next(verdicts))
    res = judge_majority("content", "crit", k=3)
    # The "[self-consistency X/N passed]" token is the contract the eval-report
    # judge-health aggregator parses; keep it stable.
    assert "[self-consistency 2/3 passed]" in res.reason


# ---- transient-failure auto-retry inside judge() ----


def _use_gemini(monkeypatch) -> None:
    """Resolve the judge to the Gemini backend so retry tests have a provider."""
    _clear_all(monkeypatch)
    clear_cache()
    monkeypatch.setenv("LLM_JUDGE_PROVIDER", "gemini")
    monkeypatch.setenv("VERTEX_PROJECT", "proj-x")


def test_judge_retries_transient_then_succeeds(monkeypatch) -> None:
    _use_gemini(monkeypatch)
    monkeypatch.setenv("LLM_JUDGE_REQUEST_RETRIES", "3")
    sleeps: list[float] = []
    monkeypatch.setattr(llm_judge.time, "sleep", lambda s: sleeps.append(s))

    calls = {"n": 0}

    def fake_gemini(content, criterion, model, temperature=0.0):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("503 Service Unavailable")
        return JudgeResult(passed=True, score=1.0, reason="ok")

    monkeypatch.setattr(llm_judge, "_judge_gemini", fake_gemini)
    res = judge("content", "crit")
    assert res.ok is True
    assert res.passed is True
    assert calls["n"] == 2  # one transient failure, then success
    assert len(sleeps) == 1  # backed off exactly once before the retry


def test_judge_retries_malformed_reply_then_succeeds(monkeypatch) -> None:
    """An unparseable reply is retried: the provider answered, just off-contract.

    The judge prompt demands ``{"passed": ..., "score": ..., "reason": ...}``.
    When a model returns prose instead, re-asking usually yields a well-formed
    object -- but the malformed result used to be returned as the FINAL verdict
    (with a guessed ``passed``) even when retries were still available.
    """
    _use_gemini(monkeypatch)
    monkeypatch.setenv("LLM_JUDGE_REQUEST_RETRIES", "3")
    sleeps: list[float] = []
    monkeypatch.setattr(llm_judge.time, "sleep", lambda s: sleeps.append(s))

    calls = {"n": 0}

    def fake_gemini(content, criterion, model, temperature=0.0):
        calls["n"] += 1
        if calls["n"] == 1:
            return llm_judge._parse_judge_text("sure, that looks true to me")
        return JudgeResult(passed=False, score=0.0, reason="not satisfied")

    monkeypatch.setattr(llm_judge, "_judge_gemini", fake_gemini)
    res = judge("content", "crit")

    assert calls["n"] == 2, "the malformed reply must be re-asked"
    assert len(sleeps) == 1
    assert res.ok is True
    # The retry's real verdict wins over the first reply's guessed "true".
    assert res.passed is False


def test_judge_malformed_reply_exhausted_is_not_ok(monkeypatch) -> None:
    """If every attempt is unparseable, report a grader failure -- never a pass."""
    _use_gemini(monkeypatch)
    monkeypatch.setenv("LLM_JUDGE_REQUEST_RETRIES", "2")
    monkeypatch.setattr(llm_judge.time, "sleep", lambda s: None)

    calls = {"n": 0}

    def fake_gemini(content, criterion, model, temperature=0.0):
        calls["n"] += 1
        return llm_judge._parse_judge_text("this looks true to me")

    monkeypatch.setattr(llm_judge, "_judge_gemini", fake_gemini)
    res = judge("content", "crit")

    assert calls["n"] == 3  # initial attempt + 2 retries
    assert res.ok is False
    assert res.passed is False
    assert "malformed judge reply" in res.error


def test_judge_does_not_retry_non_transient(monkeypatch) -> None:
    # A non-retryable error (malformed request / content policy) must fail fast as
    # ok=False without burning retries.
    _use_gemini(monkeypatch)
    monkeypatch.setenv("LLM_JUDGE_REQUEST_RETRIES", "3")
    sleeps: list[float] = []
    monkeypatch.setattr(llm_judge.time, "sleep", lambda s: sleeps.append(s))

    calls = {"n": 0}

    def fake_gemini(content, criterion, model, temperature=0.0):
        calls["n"] += 1
        raise ValueError("400 invalid request: malformed prompt")

    monkeypatch.setattr(llm_judge, "_judge_gemini", fake_gemini)
    res = judge("content", "crit")
    assert res.ok is False
    assert calls["n"] == 1  # fatal error, no wasted retries
    assert sleeps == []


def test_judge_retries_exhausted_returns_not_ok(monkeypatch) -> None:
    _use_gemini(monkeypatch)
    monkeypatch.setenv("LLM_JUDGE_REQUEST_RETRIES", "2")
    sleeps: list[float] = []
    monkeypatch.setattr(llm_judge.time, "sleep", lambda s: sleeps.append(s))

    calls = {"n": 0}

    def fake_gemini(content, criterion, model, temperature=0.0):
        calls["n"] += 1
        raise RuntimeError("429 too many requests")

    monkeypatch.setattr(llm_judge, "_judge_gemini", fake_gemini)
    res = judge("content", "crit")
    assert res.ok is False
    assert "gemini judge call failed" in res.error
    assert calls["n"] == 3  # initial attempt + 2 retries
    assert len(sleeps) == 2


def test_judge_no_retries_by_default(monkeypatch) -> None:
    # Without LLM_JUDGE_REQUEST_RETRIES the judge keeps its historical single-shot
    # behavior, so non-CLI callers (scenario generator, tests) are unchanged even
    # for an otherwise-retryable error.
    _use_gemini(monkeypatch)
    monkeypatch.delenv("LLM_JUDGE_REQUEST_RETRIES", raising=False)
    sleeps: list[float] = []
    monkeypatch.setattr(llm_judge.time, "sleep", lambda s: sleeps.append(s))

    calls = {"n": 0}

    def fake_gemini(content, criterion, model, temperature=0.0):
        calls["n"] += 1
        raise RuntimeError("503 service unavailable")

    monkeypatch.setattr(llm_judge, "_judge_gemini", fake_gemini)
    res = judge("content", "crit")
    assert res.ok is False
    assert calls["n"] == 1  # single shot
    assert sleeps == []


def test_judge_majority_inherits_retry(monkeypatch) -> None:
    # judge_majority routes each vote through the retry-capable judge(), so a
    # single transient blip in one vote no longer fails the whole majority.
    _use_gemini(monkeypatch)
    monkeypatch.setenv("LLM_JUDGE_REQUEST_RETRIES", "2")
    monkeypatch.setattr(llm_judge.time, "sleep", lambda s: None)

    calls = {"n": 0}

    def fake_gemini(content, criterion, model, temperature=0.0):
        calls["n"] += 1
        if calls["n"] == 1:  # only the very first provider call blips
            raise RuntimeError("503 service unavailable")
        return JudgeResult(passed=True, score=1.0, reason="ok")

    monkeypatch.setattr(llm_judge, "_judge_gemini", fake_gemini)
    res = judge_majority("content", "crit", k=3)
    assert res.ok is True
    assert res.passed is True
    assert calls["n"] == 4  # 1 transient + retry, then 2 clean votes


# ---------------------------------------------------------------------------
# C-1: strict parsing of the judge's ``passed`` field.
#
# Regression guard for a single-directional scoring bug: the old
# ``bool(data.get("passed", False))`` turned the STRING "false" (and "no",
# "0", any non-empty string) into a PASS, while a real ``false`` boolean
# correctly stayed a fail. Small judge models emit stringly-typed verdicts, so
# this inflated semantic checks upward only. These tests pin the corrected
# semantics AND explicitly assert the old bug is gone.
# ---------------------------------------------------------------------------


def test_parse_stringly_false_is_a_fail_not_a_pass() -> None:
    """The core bug: ``{"passed": "false"}`` must be a FAIL, never a pass."""
    res = llm_judge._parse_judge_text('{"passed": "false"}')
    assert res.passed is False
    assert res.ok is True  # a recognizable (garbled) verdict, not an infra error
    assert res.score == 0.0


def test_parse_stringly_no_and_zero_are_fails() -> None:
    for reply in ('{"passed": "no"}', '{"passed": "0"}', '{"passed": "False"}'):
        res = llm_judge._parse_judge_text(reply)
        assert res.passed is False, reply
        assert res.ok is True, reply


def test_parse_stringly_true_is_a_pass_case_and_space_insensitive() -> None:
    for reply in ('{"passed": "true"}', '{"passed": "True"}', '{"passed": "  TRUE  "}'):
        res = llm_judge._parse_judge_text(reply)
        assert res.passed is True, reply
        assert res.ok is True, reply
        assert res.score == 1.0, reply


def test_parse_real_booleans_still_work() -> None:
    t = llm_judge._parse_judge_text('{"passed": true}')
    assert t.passed is True and t.ok is True and t.score == 1.0
    f = llm_judge._parse_judge_text('{"passed": false}')
    assert f.passed is False and f.ok is True and f.score == 0.0


def test_parse_missing_passed_field_is_ok_false_not_a_verdict() -> None:
    """A dict reply with no ``passed`` key is undeterminable -> infra failure."""
    res = llm_judge._parse_judge_text('{"reason": "looks fine to me"}')
    assert res.ok is False
    assert res.passed is False
    assert "passed" in res.error


def test_parse_non_bool_non_string_passed_is_ok_false() -> None:
    """A numeric / null / structured ``passed`` value cannot be trusted."""
    for reply in ('{"passed": 1}', '{"passed": null}', '{"passed": [true]}'):
        res = llm_judge._parse_judge_text(reply)
        assert res.ok is False, reply
        assert res.passed is False, reply


def test_parse_score_is_honored_with_stringly_true() -> None:
    res = llm_judge._parse_judge_text('{"passed": "true", "score": 0.5}')
    assert res.passed is True
    assert res.score == 0.5


def test_parse_embedded_object_salvage_still_parses_passed() -> None:
    """Prose wrapping a JSON object is still salvaged (existing behavior)."""
    res = llm_judge._parse_judge_text('Sure: {"passed": true, "reason": "ok"}')
    assert res.passed is True
    assert res.ok is True


def test_coerce_passed_unit_table() -> None:
    c = llm_judge._coerce_passed
    # unambiguous booleans
    assert c(True) is True
    assert c(False) is False
    # only 'true' (case/space-insensitive) is a pass
    assert c("true") is True
    assert c("TRUE") is True
    assert c("  true  ") is True
    # every other string is a conservative fail (never invent a pass)
    for s in ("false", "no", "0", "maybe", "pass", ""):
        assert c(s) is False, s
    # non-bool / non-string / missing -> undeterminable (None -> ok=False)
    for v in (None, 1, 0, 1.0, [True], {"x": 1}, llm_judge._MISSING):
        assert c(v) is None, v


# ---- GPT on Bedrock (bedrock-gpt) ----


def _fake_bedrock_urlopen(monkeypatch, captured: dict, reply: str) -> None:
    """Patch urllib to capture the Bedrock Responses POST and return ``reply``."""

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"output_text": reply}).encode("utf-8")

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)


def test_resolve_provider_bedrock_gpt_override(monkeypatch) -> None:
    _clear_all(monkeypatch)
    for alias in ("bedrock-gpt", "bedrock", "bedrock-openai"):
        monkeypatch.setenv("LLM_JUDGE_PROVIDER", alias)
        assert resolve_provider() == "bedrock-gpt", alias


def test_resolve_provider_autodetects_bedrock_from_api_key(monkeypatch) -> None:
    # Auto-detected only from BEDROCK_OPENAI_API_KEY, and only after Vertex /
    # Anthropic, so it never shadows the established defaults.
    _clear_all(monkeypatch)
    monkeypatch.setenv("BEDROCK_OPENAI_API_KEY", "sk-test")
    assert resolve_provider() == "bedrock-gpt"
    # A Vertex project still wins over a bare Bedrock key.
    monkeypatch.setenv("VERTEX_PROJECT", "proj-x")
    assert resolve_provider() == "gemini"


def test_pick_model_bedrock_gpt_requires_openai_prefix(monkeypatch) -> None:
    _clear_all(monkeypatch)
    # openai.-prefixed and bare gpt- ids are accepted as-is.
    assert _pick_model("bedrock-gpt", "openai.gpt-5.5") == "openai.gpt-5.5"
    assert _pick_model("bedrock-gpt", "gpt-5") == "gpt-5"
    # A mismatched family falls through to LLM_JUDGE_MODEL (no built-in default).
    assert _pick_model("bedrock-gpt", "claude-sonnet-4@20250514") == ""
    monkeypatch.setenv("LLM_JUDGE_MODEL", "openai.gpt-5.5")
    assert _pick_model("bedrock-gpt", "gemini-2.5-flash") == "openai.gpt-5.5"


def test_sync_judge_env_bedrock_gpt(monkeypatch) -> None:
    _clear(monkeypatch)
    sync_judge_env("bedrock-gpt", "openai.gpt-5.5")
    assert os.environ["LLM_JUDGE_PROVIDER"] == "bedrock-gpt"
    assert os.environ["LLM_JUDGE_MODEL"] == "openai.gpt-5.5"


def test_sync_judge_env_bedrock_gpt_skips_mismatched_model(monkeypatch) -> None:
    _clear(monkeypatch)
    # A non-GPT id is left for explicit config rather than pinned.
    sync_judge_env("bedrock-gpt", "claude-sonnet-4@20250514")
    assert os.environ["LLM_JUDGE_PROVIDER"] == "bedrock-gpt"
    assert "LLM_JUDGE_MODEL" not in os.environ


def test_judge_dispatches_to_bedrock_gpt(monkeypatch) -> None:
    _clear_all(monkeypatch)
    clear_cache()
    monkeypatch.setenv("LLM_JUDGE_PROVIDER", "bedrock-gpt")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "openai.gpt-5.5")
    monkeypatch.setenv("BEDROCK_OPENAI_API_KEY", "sk-test")

    captured: dict = {}
    _fake_bedrock_urlopen(
        monkeypatch, captured, '{"passed": true, "score": 1.0, "reason": "ok"}'
    )

    res = judge("some content", "some criterion")
    assert res.ok is True
    assert res.passed is True
    # Posts to the Responses API with the resolved model and a bearer key; the
    # gpt-5-class temperature opt-out means no temperature field is sent.
    assert captured["url"].endswith("/responses")
    assert captured["body"]["model"] == "openai.gpt-5.5"
    assert "temperature" not in captured["body"]
    auth = {k.lower(): v for k, v in captured["headers"].items()}
    assert auth["authorization"] == "Bearer sk-test"


def test_judge_bedrock_gpt_honors_base_url_override(monkeypatch) -> None:
    _clear_all(monkeypatch)
    clear_cache()
    monkeypatch.setenv("LLM_JUDGE_PROVIDER", "bedrock-gpt")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "openai.gpt-5.5")
    monkeypatch.setenv("BEDROCK_OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("BEDROCK_OPENAI_BASE_URL", "https://custom.example/openai/v1")

    captured: dict = {}
    _fake_bedrock_urlopen(
        monkeypatch, captured, '{"passed": true, "score": 1.0, "reason": "ok"}'
    )

    res = judge("c", "crit")
    assert res.ok is True
    assert captured["url"] == "https://custom.example/openai/v1/responses"


def test_judge_bedrock_gpt_missing_api_key_is_not_ok(monkeypatch) -> None:
    _clear_all(monkeypatch)
    clear_cache()
    monkeypatch.setenv("LLM_JUDGE_PROVIDER", "bedrock-gpt")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "openai.gpt-5.5")
    # No BEDROCK_OPENAI_API_KEY -> reported as an infra failure, not a crash.

    def explode(*a, **k):  # pragma: no cover - must never be called
        raise AssertionError("no request should be made without an API key")

    monkeypatch.setattr("urllib.request.urlopen", explode)

    res = judge("c", "crit")
    assert res.ok is False
    assert res.passed is False
    assert "BEDROCK_OPENAI_API_KEY" in (res.error or "")


# ---- GPT on OpenAI's official API (openai) ----


def test_resolve_provider_openai_override(monkeypatch) -> None:
    _clear_all(monkeypatch)
    for alias in ("openai", "gpt"):
        monkeypatch.setenv("LLM_JUDGE_PROVIDER", alias)
        assert resolve_provider() == "openai", alias


def test_resolve_provider_autodetects_openai_last(monkeypatch) -> None:
    # OPENAI_API_KEY is a very commonly exported variable, so it is checked
    # AFTER every other backend: setting it for some unrelated tool must not
    # silently move an already-configured judge onto GPT.
    _clear_all(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    assert resolve_provider() == "openai"
    for var, expected in (
        ("BEDROCK_OPENAI_API_KEY", "bedrock-gpt"),
        ("ANTHROPIC_API_KEY", "anthropic"),
        ("VERTEX_PROJECT", "gemini"),
        ("GEMINI_API_KEY", "gemini"),
    ):
        monkeypatch.setenv(var, "x")
        assert resolve_provider() == expected, var


def test_pick_model_openai_strips_bedrock_publisher_prefix(monkeypatch) -> None:
    _clear_all(monkeypatch)
    # OpenAI's own API does not know Bedrock's 'openai.' publisher prefix, so a
    # user who switched only the provider still gets a resolvable id.
    assert _pick_model("openai", "openai.gpt-5.5") == "gpt-5.5"
    assert _pick_model("openai", "gpt-5.5") == "gpt-5.5"
    # A mismatched family falls through to LLM_JUDGE_MODEL (no built-in default).
    assert _pick_model("openai", "claude-sonnet-5") == ""
    monkeypatch.setenv("LLM_JUDGE_MODEL", "gpt-5.5")
    assert _pick_model("openai", "gemini-2.5-flash") == "gpt-5.5"


def test_sync_judge_env_openai(monkeypatch) -> None:
    _clear(monkeypatch)
    sync_judge_env("openai", "gpt-5.5")
    assert os.environ["LLM_JUDGE_PROVIDER"] == "openai"
    assert os.environ["LLM_JUDGE_MODEL"] == "gpt-5.5"


def test_sync_judge_env_openai_strips_publisher_prefix(monkeypatch) -> None:
    _clear(monkeypatch)
    sync_judge_env("openai", "openai.gpt-5.5")
    assert os.environ["LLM_JUDGE_PROVIDER"] == "openai"
    assert os.environ["LLM_JUDGE_MODEL"] == "gpt-5.5"


def test_sync_judge_env_openai_skips_mismatched_model(monkeypatch) -> None:
    _clear(monkeypatch)
    sync_judge_env("openai", "claude-sonnet-5")
    assert os.environ["LLM_JUDGE_PROVIDER"] == "openai"
    assert "LLM_JUDGE_MODEL" not in os.environ


def test_judge_dispatches_to_openai(monkeypatch) -> None:
    _clear_all(monkeypatch)
    clear_cache()
    monkeypatch.setenv("LLM_JUDGE_PROVIDER", "openai")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "gpt-5.5")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")

    captured: dict = {}
    _fake_bedrock_urlopen(
        monkeypatch, captured, '{"passed": true, "score": 1.0, "reason": "ok"}'
    )

    res = judge("some content", "some criterion")
    assert res.ok is True
    assert res.passed is True
    assert captured["url"] == f"{llm_judge._OPENAI_DEFAULT_BASE_URL}/responses"
    assert captured["body"]["model"] == "gpt-5.5"
    # Same gpt-5-class temperature opt-out as the Bedrock path.
    assert "temperature" not in captured["body"]
    auth = {k.lower(): v for k, v in captured["headers"].items()}
    assert auth["authorization"] == "Bearer sk-openai"


def test_judge_openai_honors_base_url_override(monkeypatch) -> None:
    _clear_all(monkeypatch)
    clear_cache()
    monkeypatch.setenv("LLM_JUDGE_PROVIDER", "openai")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "gpt-5.5")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example/v1")

    captured: dict = {}
    _fake_bedrock_urlopen(
        monkeypatch, captured, '{"passed": true, "score": 1.0, "reason": "ok"}'
    )

    assert judge("c", "crit").ok is True
    assert captured["url"] == "https://proxy.example/v1/responses"


def test_judge_openai_missing_api_key_is_not_ok(monkeypatch) -> None:
    _clear_all(monkeypatch)
    clear_cache()
    monkeypatch.setenv("LLM_JUDGE_PROVIDER", "openai")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "gpt-5.5")

    def explode(*a, **k):  # pragma: no cover - must never be called
        raise AssertionError("no request should be made without an API key")

    monkeypatch.setattr("urllib.request.urlopen", explode)

    res = judge("c", "crit")
    assert res.ok is False
    assert res.passed is False
    assert "OPENAI_API_KEY" in (res.error or "")


def test_openai_and_bedrock_send_the_same_request_body(monkeypatch) -> None:
    # The two GPT judges must stay one implementation: same Responses-API body,
    # differing only in host and bearer key. A divergence here means one of them
    # has grown behaviour the other silently lacks.
    bodies = {}
    for provider, key_var, key in (
        ("openai", "OPENAI_API_KEY", "sk-openai"),
        ("bedrock-gpt", "BEDROCK_OPENAI_API_KEY", "sk-bedrock"),
    ):
        _clear_all(monkeypatch)
        clear_cache()
        monkeypatch.setenv("LLM_JUDGE_PROVIDER", provider)
        monkeypatch.setenv("LLM_JUDGE_MODEL", "gpt-5.5")
        monkeypatch.setenv(key_var, key)
        captured: dict = {}
        _fake_bedrock_urlopen(
            monkeypatch, captured, '{"passed": true, "score": 1.0, "reason": "ok"}'
        )
        assert judge("same content", "same criterion").ok is True
        bodies[provider] = captured["body"]
    assert bodies["openai"] == bodies["bedrock-gpt"]


# ---------------------------------------------------------------------------
# Reasoning-model response shapes (thinking blocks).
#
# Claude returns ``content`` as a LIST of typed blocks. A reasoning model such
# as claude-sonnet-5 prefixes ``thinking`` blocks, so ``response.content[0]`` is
# a ThinkingBlock with no ``.text`` -- which is how the judge came to die with
#     vertex-anthropic judge call failed:
#     'ThinkingBlock' object has no attribute 'text'
# mid-generation, silently disabling the gold-satisfiability gate for the whole
# round (its caller ``return``s on judge error -- it fails OPEN).
#
# NOTE the pre-existing ``_fake_anthropic_vertex`` above hands back a
# single-text-block response, which is exactly why 44 judge tests never caught
# this. These fakes deliberately model the real multi-block shape.
# ---------------------------------------------------------------------------


def _thinking_block(text: str = "let me reason about this"):
    """A ThinkingBlock-alike: has ``.type``/``.thinking`` but NO ``.text``."""
    return type("_Thinking", (), {"type": "thinking", "thinking": text})()


def _redacted_block():
    return type("_Redacted", (), {"type": "redacted_thinking", "data": "enc"})()


def _tool_use_block():
    return type("_ToolUse", (), {"type": "tool_use", "id": "t1", "input": {}})()


def _text_block(text: str):
    return type("_Text", (), {"type": "text", "text": text})()


def _resp(blocks):
    return type("_Resp", (), {"content": list(blocks)})()


def _fake_anthropic_blocks(monkeypatch, blocks, *, vertex: bool = True) -> None:
    """Inject a fake ``anthropic`` module returning an arbitrary block list."""

    class _Messages:
        def create(self, **kwargs):
            return _resp(blocks)

    class _Client:
        def __init__(self, *a, **k):
            self.messages = _Messages()

    fake_mod = types.ModuleType("anthropic")
    fake_mod.AnthropicVertex = _Client
    fake_mod.Anthropic = _Client
    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)


_VERDICT = '{"passed": true, "score": 1.0, "reason": "ok"}'


def test_thinking_block_first_does_not_crash_vertex_judge(monkeypatch) -> None:
    """The exact reported crash: sonnet-5 prefixes a thinking block."""
    _clear_all(monkeypatch)
    clear_cache()
    monkeypatch.setenv("LLM_JUDGE_PROVIDER", "vertex-anthropic")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("VERTEX_PROJECT", "p")
    _fake_anthropic_blocks(monkeypatch, [_thinking_block(), _text_block(_VERDICT)])

    res = judge("transcript", "criterion")
    assert res.ok is True, f"judge must survive a thinking block: {res.error!r}"
    assert res.passed is True
    assert "ThinkingBlock" not in (res.error or "")


def test_thinking_block_first_does_not_crash_direct_anthropic_judge(
    monkeypatch,
) -> None:
    """Same latent bug on the direct (non-Vertex) Anthropic judge path."""
    _clear_all(monkeypatch)
    clear_cache()
    monkeypatch.setenv("LLM_JUDGE_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    _fake_anthropic_blocks(monkeypatch, [_thinking_block(), _text_block(_VERDICT)])

    res = judge("transcript", "criterion")
    assert res.ok is True, f"judge must survive a thinking block: {res.error!r}"
    assert res.passed is True


def test_judge_survives_redacted_thinking_and_tool_use_prefix(monkeypatch) -> None:
    _clear_all(monkeypatch)
    clear_cache()
    monkeypatch.setenv("LLM_JUDGE_PROVIDER", "vertex-anthropic")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("VERTEX_PROJECT", "p")
    _fake_anthropic_blocks(
        monkeypatch,
        [_thinking_block(), _redacted_block(), _tool_use_block(), _text_block(_VERDICT)],
    )

    res = judge("transcript", "criterion")
    assert res.ok is True, f"unexpected judge error: {res.error!r}"
    assert res.passed is True


def test_judge_text_only_response_still_works(monkeypatch) -> None:
    """Positive control: the ordinary single-text-block shape is unaffected."""
    _clear_all(monkeypatch)
    clear_cache()
    monkeypatch.setenv("LLM_JUDGE_PROVIDER", "vertex-anthropic")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "claude-haiku-4-5")
    monkeypatch.setenv("VERTEX_PROJECT", "p")
    _fake_anthropic_blocks(monkeypatch, [_text_block(_VERDICT)])

    res = judge("transcript", "criterion")
    assert res.ok is True
    assert res.passed is True


def test_thinking_only_response_is_reported_not_crashed(monkeypatch) -> None:
    """A turn with NO text block must fail closed (reported), never raise."""
    _clear_all(monkeypatch)
    clear_cache()
    monkeypatch.setenv("LLM_JUDGE_PROVIDER", "vertex-anthropic")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("VERTEX_PROJECT", "p")
    _fake_anthropic_blocks(monkeypatch, [_thinking_block()])

    res = judge("transcript", "criterion")
    assert res.ok is False
    assert res.passed is False


# --- the shared extractor itself -------------------------------------------


def test_extractor_skips_thinking_and_returns_text() -> None:
    f = _llm_retry.anthropic_text_from_response
    assert f(_resp([_thinking_block(), _text_block("answer")])) == "answer"


def test_extractor_concatenates_multiple_text_blocks_in_order() -> None:
    f = _llm_retry.anthropic_text_from_response
    assert f(_resp([_text_block("a"), _thinking_block(), _text_block("b")])) == "ab"


def test_extractor_handles_dict_shaped_blocks() -> None:
    f = _llm_retry.anthropic_text_from_response
    resp = _resp(
        [{"type": "thinking", "thinking": "x"}, {"type": "text", "text": " hi "}]
    )
    assert f(resp) == "hi"


def test_extractor_handles_dict_shaped_response() -> None:
    f = _llm_retry.anthropic_text_from_response
    assert f({"content": [{"type": "text", "text": "hi"}]}) == "hi"


def test_extractor_tolerates_typeless_fake_with_text() -> None:
    """Legacy test fakes expose ``.text`` without a ``.type``; keep them working."""
    f = _llm_retry.anthropic_text_from_response
    block = type("_C", (), {"text": "hi"})()
    assert f(_resp([block])) == "hi"


@pytest.mark.parametrize(
    "response",
    [
        _resp([]),
        _resp([_thinking_block()]),
        _resp([_tool_use_block()]),
        type("_NoContent", (), {})(),
        {"content": None},
        {},
    ],
)
def test_extractor_returns_empty_when_no_text_block(response) -> None:
    """Empty means 'no answer' -> callers retry. Must never raise."""
    assert _llm_retry.anthropic_text_from_response(response) == ""


# --- structural guards against the copy drifting back ----------------------


def test_scenario_client_reuses_the_shared_extractor() -> None:
    """One definition tree-wide.

    A private mirror in the scenario client is what let the judge keep crashing
    long after the client had been fixed; assert they are literally the same
    object so a future copy-paste fix cannot diverge again.
    """
    from clasharness.scenario import llm_client

    assert (
        llm_client._anthropic_text_from_response
        is _llm_retry.anthropic_text_from_response
    )


def test_no_module_indexes_anthropic_content_block_zero() -> None:
    """No caller may go back to ``response.content[0].text``.

    This is the structural regression test for the original defect: it fails on
    the pattern itself, in any clasharness module, not just the two call sites
    that happened to be known.
    """
    import pathlib
    import re

    # No `.resolve()`: under the build system `__file__` is a symlink into the
    # CAS, and resolving it walks out of the runfiles tree.
    root = pathlib.Path(__file__).parent.parent / "clasharness"
    pattern = re.compile(r"\.content\[0\]\s*\.\s*text")
    offenders = [
        f"{path.relative_to(root.parent)}:{i}"
        for path in sorted(root.rglob("*.py"))
        for i, line in enumerate(path.read_text().splitlines(), 1)
        if pattern.search(line) and not line.lstrip().startswith("#")
    ]
    assert not offenders, (
        "use llm_retry.anthropic_text_from_response instead of indexing block 0: "
        + ", ".join(offenders)
    )


def test_extractor_accepts_magicmock_block_with_text() -> None:
    """MagicMock auto-creates ``.type``, defeating a ``btype is None`` fallback.

    Several suites fake Anthropic responses with ``MagicMock(text=...)``; those
    must keep working, so the fallback keys on "``.type`` is not a *string*"
    (real SDK blocks always have a str type) rather than on it being ``None``.
    """
    from unittest.mock import MagicMock

    f = _llm_retry.anthropic_text_from_response
    assert f(_resp([MagicMock(text='{"passed": true}')])) == '{"passed": true}'


def test_extractor_ignores_magicmock_block_without_explicit_text() -> None:
    """A bare MagicMock has an auto-created ``.text`` that is NOT a str; it must
    not be spliced into the answer (and must not blow up ``str.join``)."""
    from unittest.mock import MagicMock

    assert _llm_retry.anthropic_text_from_response(_resp([MagicMock()])) == ""


def test_extractor_ignores_non_string_text_payload() -> None:
    """A non-str ``.text`` is dropped rather than raising inside ``str.join``."""
    f = _llm_retry.anthropic_text_from_response
    block = type("_Weird", (), {"type": "text", "text": 42})()
    assert f(_resp([block, _text_block("ok")])) == "ok"


def test_extractor_excludes_typed_non_text_block_even_when_it_carries_text() -> None:
    """Block *type* decides, not the mere presence of a ``.text`` payload.

    Without this, a selector that stopped filtering on type would splice model
    reasoning into the verdict and no other test would notice (the ThinkingBlock
    fakes above carry ``.thinking``, not ``.text``).
    """
    f = _llm_retry.anthropic_text_from_response
    sneaky_dict = {"type": "thinking", "text": "SECRET REASONING"}
    assert f(_resp([sneaky_dict, _text_block("answer")])) == "answer"

    sneaky_obj = type("_Sneaky", (), {"type": "tool_use", "text": "TOOL NOISE"})()
    assert f(_resp([sneaky_obj, _text_block("answer")])) == "answer"
