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

"""Unified LLM judge for semantic evaluation.

Both the EffectChecker (``field_semantic_match``) and the LLMChecker route
through this single entry point so the benchmark has ONE provider-selection
policy, ONE prompt/response contract, consistent failure semantics, and a shared
in-process response cache.

Failure semantics: the judge never raises. When no provider is configured or a
provider call fails, it returns ``JudgeResult(ok=False, passed=False, ...)`` with
a diagnostic ``error``. Callers treat that as a (loud) failed check rather than a
vacuous pass or a hard crash of the whole evaluation.

Four of the providers below reach a vendor's official public API with nothing but
an API key, which is what a public reproduction of the benchmark should use. The
other two -- ``vertex-anthropic`` and ``bedrock-gpt`` -- reach the same model
families through Google Cloud and AWS instead; they are what the paper's runs
used and are kept working, but they need cloud credentials and are not required.

Provider selection precedence:
  1. ``LLM_JUDGE_PROVIDER`` env override:
       - "gemini"/"vertex"                    -> Gemini (API key, or Vertex AI via ADC)
       - "anthropic"/"claude"                 -> Anthropic Claude (official API)
       - "openai"/"gpt"                       -> OpenAI GPT (official Responses API)
       - "vertex-anthropic"/"vertex-claude"   -> Claude on Vertex AI (AnthropicVertex)
       - "bedrock-gpt"/"bedrock"              -> GPT on Bedrock (OpenAI Responses API)
  2. ``GEMINI_API_KEY`` set -> Gemini (official API)
  3. ``VERTEX_PROJECT`` set  -> Gemini on Vertex AI
  4. ``ANTHROPIC_API_KEY`` set -> Anthropic Claude (official API)
  5. ``BEDROCK_OPENAI_API_KEY`` set -> GPT on Bedrock
  6. ``OPENAI_API_KEY`` set -> OpenAI GPT (official API)
  7. otherwise -> no provider (ok=False)

``vertex-anthropic`` is never auto-detected (a bare ``VERTEX_PROJECT`` already
means Gemini): it is reachable only via an explicit ``LLM_JUDGE_PROVIDER``
override (i.e. the ``--llm-judge-provider vertex-anthropic`` CLI flag), and
requires an explicitly supplied Claude model id (there is no default). The id is
passed through verbatim -- exactly like the agent/generation client -- so both
``'@'``-form publisher ids (``claude-sonnet-4@20250514``) and plain dash ids
(``claude-opus-4-8``) are accepted; whether a given id resolves is left to the
Vertex endpoint.

``bedrock-gpt`` likewise requires an explicitly supplied model id (an
``openai.``-prefixed publisher id such as ``openai.gpt-5.5``, or a plain
``gpt-`` id) -- there is no built-in default. It authenticates with
``BEDROCK_OPENAI_API_KEY`` and posts to the OpenAI Responses API at
``BEDROCK_OPENAI_BASE_URL`` (default: the us-east-2 'mantle' endpoint).

``openai`` is the same Responses-API request against OpenAI's own host, so it
too needs an explicit model id -- but written WITHOUT Bedrock's publisher prefix
(``gpt-5.5``, not ``openai.gpt-5.5``). It reads ``OPENAI_API_KEY`` and
``OPENAI_BASE_URL`` (default ``https://api.openai.com/v1``). It is auto-detected
last so that setting ``OPENAI_API_KEY`` for some other tool cannot silently
displace an already-configured Gemini/Claude/Bedrock judge.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass

from clasharness.evaluation.llm_retry import (
    _is_anthropic_temperature_unsupported,
    _is_retryable_llm_error,
    _retry_backoff_delay,
    anthropic_rejects_sampling_params,
    anthropic_temperature_kwargs,
    anthropic_text_from_response,
    note_anthropic_sampling_rejected,
)


@dataclass
class JudgeResult:
    """Outcome of a single semantic judgement."""

    passed: bool
    score: float
    reason: str
    ok: bool = True  # False when the judge could not run (config/provider error)
    error: str = ""


_CACHE: dict[str, JudgeResult] = {}

_GEMINI_DEFAULT = "gemini-2.5-flash"
_ANTHROPIC_DEFAULT = "claude-haiku-4-5-20251001"

# GPT on Bedrock ('bedrock-gpt'): OpenAI Responses-API compatible, served from
# the 'mantle' endpoint (region fixed to us-east-2; override with
# BEDROCK_OPENAI_BASE_URL). Authenticated with a static bearer key. Model ids
# carry an 'openai.' publisher prefix (e.g. 'openai.gpt-5.5') and there is no
# built-in default -- the id must be supplied explicitly, like vertex-anthropic.
_BEDROCK_GPT_DEFAULT_BASE_URL = "https://bedrock-mantle.us-east-2.api.aws/openai/v1"
_BEDROCK_GPT_API_KEY_ENV = "BEDROCK_OPENAI_API_KEY"
_BEDROCK_GPT_BASE_URL_ENV = "BEDROCK_OPENAI_BASE_URL"

# GPT on OpenAI's own API ('openai'): byte-for-byte the same Responses-API
# request as bedrock-gpt, only the host and the key differ -- which is why both
# share _judge_openai_responses below. Model ids drop Bedrock's publisher prefix
# ('gpt-5.5', not 'openai.gpt-5.5'), and as with bedrock-gpt there is no
# built-in default; the id must be supplied explicitly.
_OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
_OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"


def clear_cache() -> None:
    """Clear the in-process judge cache (used by tests)."""
    _CACHE.clear()


def _judge_retries() -> int:
    """Number of retries for transient judge failures (0 = single attempt).

    Transported via env (``LLM_JUDGE_REQUEST_RETRIES``) rather than a function
    argument because ``judge`` is called deep inside evaluators with no access to
    CLI args, and env is the only channel that also propagates to parallel worker
    processes. ``examples/train_and_eval.py`` seeds it from ``--llm-request-retries``.
    Defaults to 0 when unset so non-CLI callers (scenario generator, tests) keep
    the historical single-shot behavior unless they opt in.
    """
    try:
        return max(0, int(os.environ.get("LLM_JUDGE_REQUEST_RETRIES", "0")))
    except (TypeError, ValueError):
        return 0


def _judge_retry_backoff_s() -> float:
    """Base backoff (seconds) for judge retries; seeded from --llm-retry-backoff-s."""
    try:
        return max(0.0, float(os.environ.get("LLM_JUDGE_RETRY_BACKOFF_S", "1.5")))
    except (TypeError, ValueError):
        return 1.5


def resolve_provider() -> str:
    """Return the active judge provider, or '' when none is configured.

    One of 'gemini', 'anthropic', 'openai', 'vertex-anthropic', 'bedrock-gpt'.

    ``vertex-anthropic`` (Claude on Vertex AI) is intentionally never
    auto-detected -- ``GEMINI_API_KEY`` or a bare ``VERTEX_PROJECT`` both map to
    Gemini -- so it only resolves via an explicit ``LLM_JUDGE_PROVIDER``
    override. ``bedrock-gpt`` and ``openai`` resolve via the explicit override,
    and also auto-detect from ``BEDROCK_OPENAI_API_KEY`` / ``OPENAI_API_KEY``
    respectively -- both checked last, after Gemini/Anthropic, so neither ever
    shadows the established defaults.
    """
    override = os.environ.get("LLM_JUDGE_PROVIDER", "").strip().lower()
    if override in {"vertex-anthropic", "vertex-claude", "claude-vertex"}:
        return "vertex-anthropic"
    if override in {"bedrock-gpt", "bedrock", "bedrock-openai"}:
        return "bedrock-gpt"
    if override in {"gemini", "vertex"}:
        return "gemini"
    if override in {"anthropic", "claude"}:
        return "anthropic"
    if override in {"openai", "gpt"}:
        return "openai"
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if os.environ.get("VERTEX_PROJECT"):
        return "gemini"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get(_BEDROCK_GPT_API_KEY_ENV):
        return "bedrock-gpt"
    if os.environ.get(_OPENAI_API_KEY_ENV):
        return "openai"
    return ""


def sync_judge_env(provider: str | None, model: str | None) -> None:
    """Align the semantic-match judge with a driving generation/eval backend.

    The judge resolves its backend from ``LLM_JUDGE_PROVIDER`` /
    ``LLM_JUDGE_MODEL``. A caller that has already chosen a provider+model (e.g.
    the scenario generator running on ``--provider=vertex --model=gemini-3.5-flash``)
    calls this so ``field_semantic_match`` runs on the SAME backend, instead of
    falling back to ``_GEMINI_DEFAULT`` (gemini-2.5-flash), which 404s on Vertex
    endpoints that only serve the generation model. An explicit ``LLM_JUDGE_*``
    env var the user set always wins (we never overwrite it). ``mock``/unknown
    providers leave the environment untouched.
    """
    provider = (provider or "").strip().lower()
    model = (model or "").strip()
    if provider in {"vertex-anthropic", "vertex-claude", "claude-vertex"}:
        os.environ.setdefault("LLM_JUDGE_PROVIDER", "vertex-anthropic")
        # Pin any Claude id the caller chose, passing it through verbatim just
        # like the agent/generation client does (no '@'-form requirement): a
        # Vertex endpoint may serve either '@'-form publisher ids or plain dash
        # ids, so trust the caller and let the endpoint decide what resolves.
        if model.startswith("claude"):
            os.environ.setdefault("LLM_JUDGE_MODEL", model)
    elif provider in {"bedrock-gpt", "bedrock", "bedrock-openai"}:
        os.environ.setdefault("LLM_JUDGE_PROVIDER", "bedrock-gpt")
        # Bedrock GPT ids carry an 'openai.' publisher prefix (e.g.
        # 'openai.gpt-5.5'); a plain 'gpt-' id is also accepted. Pin only a
        # matching id so a mismatched family is left to explicit config.
        if model.startswith("openai.") or model.startswith("gpt-"):
            os.environ.setdefault("LLM_JUDGE_MODEL", model)
    elif provider in {"gemini", "vertex"}:
        os.environ.setdefault("LLM_JUDGE_PROVIDER", "gemini")
        if model.startswith("gemini") or model.startswith("publishers/"):
            os.environ.setdefault("LLM_JUDGE_MODEL", model)
    elif provider in {"anthropic", "claude"}:
        os.environ.setdefault("LLM_JUDGE_PROVIDER", "anthropic")
        if model.startswith("claude"):
            os.environ.setdefault("LLM_JUDGE_MODEL", model)
    elif provider in {"openai", "gpt"}:
        os.environ.setdefault("LLM_JUDGE_PROVIDER", "openai")
        # Official OpenAI ids carry no publisher prefix. Accept a Bedrock-style
        # 'openai.' id too but strip the prefix, so a caller that switched only
        # --provider (bedrock-gpt -> openai) and left the model id alone gets a
        # judge id the official endpoint can actually resolve.
        if model.startswith("openai."):
            os.environ.setdefault("LLM_JUDGE_MODEL", model[len("openai."):])
        elif model.startswith("gpt-"):
            os.environ.setdefault("LLM_JUDGE_MODEL", model)


def _pick_model(provider: str, requested: str | None) -> str:
    """Use the requested model only if it matches the active provider family."""
    requested = (requested or "").strip()
    if provider == "gemini":
        if requested and (requested.startswith("gemini") or requested.startswith("publishers/")):
            return requested
        return os.environ.get("LLM_JUDGE_MODEL") or _GEMINI_DEFAULT
    if provider == "vertex-anthropic":
        # Accept any Claude id verbatim, matching the agent/generation client
        # which passes --model straight to AnthropicVertex with no format check:
        # both '@'-form publisher ids ('claude-sonnet-4@20250514') and plain
        # dash ids ('claude-opus-4-8') are honored, and whether one resolves is
        # left to the endpoint. The 'claude' prefix is still required so a
        # mismatched family (e.g. a stray gemini id) falls back rather than being
        # sent to Claude. There is no built-in default -- the CLI hard-errors
        # when LLM_JUDGE_MODEL is unset.
        if requested.startswith("claude"):
            return requested
        return os.environ.get("LLM_JUDGE_MODEL", "")
    if provider == "bedrock-gpt":
        # Bedrock GPT uses 'openai.'-prefixed ids (e.g. 'openai.gpt-5.5'); a
        # plain 'gpt-' id is also accepted. Like vertex-anthropic there is no
        # built-in default -- the CLI hard-errors when LLM_JUDGE_MODEL is unset.
        if requested.startswith("openai.") or requested.startswith("gpt-"):
            return requested
        return os.environ.get("LLM_JUDGE_MODEL", "")
    if provider == "openai":
        # Official OpenAI ids have no publisher prefix; accept a Bedrock-style
        # 'openai.gpt-...' spelling but hand the endpoint the bare id. As with
        # bedrock-gpt there is no built-in default.
        if requested.startswith("openai."):
            return requested[len("openai."):]
        if requested.startswith("gpt-"):
            return requested
        return os.environ.get("LLM_JUDGE_MODEL", "")
    if provider == "anthropic":
        if requested.startswith("claude"):
            return requested
        return os.environ.get("LLM_JUDGE_MODEL") or _ANTHROPIC_DEFAULT
    return requested


def active_judge_config() -> tuple[str, str]:
    """Return the ``(provider, model)`` the judge uses by default this run.

    Provider is ``''`` when no judge is configured. ``model`` is the run-level
    default -- ``LLM_JUDGE_MODEL`` or the provider's built-in default -- resolved
    before any per-check ``spec['model']`` override. Recorded in the run summary
    for reproducibility (the judge is frequently a different, smaller model than
    the solver under evaluation).
    """
    provider = resolve_provider()
    if not provider:
        return "", ""
    return provider, _pick_model(provider, None)


def _cache_key(
    provider: str, model: str, criterion: str, content: str, salt: str = ""
) -> str:
    digest = hashlib.sha256()
    digest.update(
        f"{provider}\x00{model}\x00{criterion}\x00{content}\x00{salt}".encode("utf-8")
    )
    return digest.hexdigest()


def _build_prompt(content: str, criterion: str) -> str:
    return f"""You are an automated task-evaluation judge.
Decide whether the provided content satisfies the evaluation criterion, based
solely on the content. Do not invent information.

## Content to evaluate
{content}

## Evaluation criterion
{criterion}

Respond with ONLY a JSON object in this exact format (no other text):
{{"passed": true_or_false, "score": 0.0_to_1.0, "reason": "one concise sentence"}}

Scoring guide:
  1.0 - criterion fully satisfied
  0.5 - partially satisfied (right intent, wrong details)
  0.0 - not satisfied"""


def judge(
    content: str,
    criterion: str,
    *,
    model: str | None = None,
    use_cache: bool = True,
    temperature: float = 0.0,
    cache_salt: str = "",
) -> JudgeResult:
    """Judge whether ``content`` satisfies ``criterion`` using the active provider.

    ``temperature`` controls sampling (default 0.0 keeps single-shot grading
    deterministic). ``cache_salt`` is mixed into the cache key so callers that
    deliberately want independent samples of the same (content, criterion) - e.g.
    self-consistency voting - do not collide on one cached verdict.
    """
    provider = resolve_provider()
    if not provider:
        return JudgeResult(
            passed=False,
            score=0.0,
            reason="",
            ok=False,
            error=(
                "No LLM judge configured. Set GEMINI_API_KEY (Gemini), "
                "ANTHROPIC_API_KEY (Claude), OPENAI_API_KEY (GPT), or "
                "VERTEX_PROJECT (Gemini on Vertex AI); or set "
                "LLM_JUDGE_PROVIDER (gemini / anthropic / openai / "
                "vertex-anthropic / bedrock-gpt)."
            ),
        )

    resolved_model = _pick_model(provider, model)
    key = _cache_key(provider, resolved_model, criterion, content, cache_salt)
    if use_cache and key in _CACHE:
        return _CACHE[key]

    # Retry only TRANSIENT provider failures (429 / 5xx / timeouts / quota),
    # reusing the same classifier + backoff as the agent path so a single blip
    # in the judge no longer scores an otherwise-passing check as failed. A
    # non-retryable error (content policy, malformed request) still fails fast as
    # ``ok=False`` with no wasted calls. ``retries`` defaults to 0 (single shot)
    # unless seeded from --llm-request-retries, so non-CLI callers are unchanged.
    retries = _judge_retries()
    backoff = _judge_retry_backoff_s()
    result: JudgeResult | None = None
    for attempt in range(retries + 1):
        try:
            if provider == "gemini":
                result = _judge_gemini(
                    content, criterion, resolved_model, temperature
                )
            elif provider == "vertex-anthropic":
                result = _judge_vertex_anthropic(
                    content, criterion, resolved_model, temperature
                )
            elif provider == "bedrock-gpt":
                result = _judge_bedrock_gpt(
                    content, criterion, resolved_model, temperature
                )
            elif provider == "openai":
                result = _judge_openai(
                    content, criterion, resolved_model, temperature
                )
            else:
                result = _judge_anthropic(
                    content, criterion, resolved_model, temperature
                )
            # A reply we could not parse (ok=False from _parse_judge_text) is
            # treated like a transient fault: the provider answered, it just did
            # not honour the response contract, and re-asking usually yields a
            # well-formed object. Without this the malformed verdict would be the
            # final answer even with retries left. ``ok=False`` results are never
            # cached (see below), so a retry is always a fresh call.
            if result is not None and not result.ok and attempt < retries:
                time.sleep(_retry_backoff_delay(backoff, attempt))
                continue
            break
        except Exception as exc:  # noqa: BLE001 - surfaced as ok=False, never crashes eval
            if attempt < retries and _is_retryable_llm_error(exc):
                time.sleep(_retry_backoff_delay(backoff, attempt))
                continue
            result = JudgeResult(
                passed=False,
                score=0.0,
                reason="",
                ok=False,
                error=f"{provider} judge call failed: {exc}",
            )
            break

    if result is None:
        # Unreachable for the shipped backends (``retries`` is clamped to >= 0,
        # so the loop always runs and every path assigns ``result``), but a
        # backend that returns None instead of a JudgeResult would otherwise
        # raise AttributeError straight out of ``judge`` and take the episode
        # with it. Fail as a loud infrastructure error like any other.
        return JudgeResult(
            passed=False,
            score=0.0,
            reason="",
            ok=False,
            error=f"{provider} judge returned no result",
        )

    if use_cache and result.ok:
        _CACHE[key] = result
    return result


def judge_majority(
    content: str,
    criterion: str,
    *,
    model: str | None = None,
    k: int = 3,
    temperature: float = 0.0,
) -> JudgeResult:
    """Judge ``content`` against ``criterion`` via k-way self-consistency voting.

    DEFAULT (``k=3``, ``temperature=0.0``): three GREEDY votes, strict majority.

    The two knobs answer different questions and are deliberately not tied:

    ``temperature=0.0`` is what makes grading reproducible. The ``_CACHE`` is
    process-local and never persisted, so the previous ``0.5`` default made
    every fresh eval process re-roll every borderline check. Measured on 10
    models x 600 tasks (17533 voted checks), 3.6% of verdicts were non-unanimous
    under ``k=3``/``0.5``, putting a ~1pp wobble on each model's pass rate --
    wider than the gap separating the middle of the leaderboard.

    ``k=3`` is NOT made redundant by that. Greedy decoding on the hosted
    backends is not bit-deterministic (batching, MoE routing, floating-point
    reassociation), so three greedy votes still disagree occasionally. Majority
    voting absorbs that residual jitter, and -- because a split at temperature 0
    can only come from provider nondeterminism -- the split rate reported by
    ``eval_report.JudgeHealth`` becomes a direct health signal rather than a
    measure of deliberate sampling.

    Cost is 3 provider calls per distinct (content, criterion) pair: each vote
    carries its own ``cache_salt`` so the votes neither collapse onto one answer
    nor collide in the shared cache. The salts are per-index and stable, so
    REPEATS of the same pair within a process still hit the cache -- 3 calls
    total, not 3 per grading.

    ``k=1`` drops to a single unsalted call (cheapest, no health signal). Pass
    an explicit ``temperature > 0`` only for diagnostic sweeps that intend to
    measure how borderline a criterion is; it forfeits reproducibility.

    If every vote fails to run (provider/config error), the last failing result
    is returned so the caller still observes ``ok=False``.
    """
    k = max(1, int(k))
    if k == 1:
        return judge(content, criterion, model=model, temperature=temperature)

    results = [
        judge(
            content,
            criterion,
            model=model,
            temperature=temperature,
            cache_salt=f"vote:{i}",
        )
        for i in range(k)
    ]
    ok_results = [r for r in results if r.ok]
    if not ok_results:
        return results[-1]

    passed_votes = sum(1 for r in ok_results if r.passed)
    passed = passed_votes * 2 > len(ok_results)  # strict majority
    mean_score = sum(r.score for r in ok_results) / len(ok_results)
    winning = [r for r in ok_results if r.passed == passed]
    base_reason = (winning[0].reason if winning else ok_results[0].reason) or ""
    reason = f"{base_reason} [self-consistency {passed_votes}/{len(ok_results)} passed]"
    return JudgeResult(
        passed=passed, score=mean_score, reason=reason, ok=True, error=""
    )


def _judge_gemini(
    content: str, criterion: str, model: str, temperature: float = 0.0
) -> JudgeResult:
    from google import genai
    from google.genai import types

    # Same two backends as the agent (see LLMAgent._get_gemini_client): an API
    # key selects the Gemini Developer API, a project selects Vertex AI. The
    # `publishers/...` form is Vertex-only -- the Developer API wants the bare
    # model id and 404s on the prefixed one -- so the two must be decided
    # together rather than in separate branches.
    api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    model_name = model
    if api_key:
        client = genai.Client(api_key=api_key)
        model_name = model_name.removeprefix("publishers/google/models/")
    else:
        if not model_name.startswith("publishers/"):
            model_name = f"publishers/google/models/{model_name}"
        client = genai.Client(
            vertexai=True,
            project=os.environ.get("VERTEX_PROJECT"),
            location=os.environ.get("VERTEX_LOCATION", "us"),
        )
    response = client.models.generate_content(
        model=model_name,
        contents=_build_prompt(content, criterion),
        config=types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
        ),
    )
    return _parse_judge_text(response.text)


def _judge_anthropic(
    content: str, criterion: str, model: str, temperature: float = 0.0
) -> JudgeResult:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    def _create(include_temperature: bool):
        if include_temperature:
            return client.messages.create(
                model=model,
                max_tokens=256,
                **anthropic_temperature_kwargs(temperature),
                messages=[
                    {"role": "user", "content": _build_prompt(content, criterion)}
                ],
            )
        return client.messages.create(
            model=model,
            max_tokens=256,
            messages=[
                {"role": "user", "content": _build_prompt(content, criterion)}
            ],
        )

    # Newer Claude models (Sonnet 5, Opus 4.7/4.8, ...) 400 on ANY explicit
    # temperature; omit it proactively for those, keep temp=0 for older models so
    # their judgements stay reproducible. If an unknown model still rejects it,
    # drop it and retry once (remembering it process-wide so we never re-pay).
    include_temperature = not anthropic_rejects_sampling_params(model)
    try:
        response = _create(include_temperature)
    except Exception as exc:
        if include_temperature and _is_anthropic_temperature_unsupported(exc):
            note_anthropic_sampling_rejected(model)
            response = _create(False)
        else:
            raise
    # NOT response.content[0].text: a reasoning model prefixes thinking blocks.
    raw_text = anthropic_text_from_response(response)
    return _parse_judge_text(raw_text)


def _judge_vertex_anthropic(
    content: str, criterion: str, model: str, temperature: float = 0.0
) -> JudgeResult:
    """Judge via Claude on Vertex AI (AnthropicVertex), authenticated with ADC.

    Mirrors ``_judge_anthropic`` -- same ``messages.create`` contract -- but
    routes through Vertex instead of the direct Anthropic API. ``model`` is sent
    verbatim (an ``'@'``-form publisher id such as ``claude-sonnet-4@20250514``
    or a plain dash id such as ``claude-opus-4-8``, depending on what the
    endpoint serves) and the region must serve Claude (defaults to ``us-east5``;
    the Gemini judge's ``us`` does not serve Claude).
    """
    from anthropic import AnthropicVertex

    client = AnthropicVertex(
        project_id=os.environ.get("VERTEX_PROJECT"),
        region=os.environ.get("VERTEX_LOCATION") or "us-east5",
    )

    def _create(include_temperature: bool):
        if include_temperature:
            return client.messages.create(
                model=model,
                max_tokens=256,
                **anthropic_temperature_kwargs(temperature),
                messages=[
                    {"role": "user", "content": _build_prompt(content, criterion)}
                ],
            )
        return client.messages.create(
            model=model,
            max_tokens=256,
            messages=[
                {"role": "user", "content": _build_prompt(content, criterion)}
            ],
        )

    # See _judge_anthropic: omit temperature proactively for models known to
    # reject it, keep temp=0 for older models (reproducibility), and drop+retry
    # once if an unknown model 400s on it (remembering it process-wide).
    include_temperature = not anthropic_rejects_sampling_params(model)
    try:
        response = _create(include_temperature)
    except Exception as exc:
        if include_temperature and _is_anthropic_temperature_unsupported(exc):
            note_anthropic_sampling_rejected(model)
            response = _create(False)
        else:
            raise
    # NOT response.content[0].text: a reasoning model prefixes thinking blocks.
    raw_text = anthropic_text_from_response(response)
    return _parse_judge_text(raw_text)


def _bedrock_gpt_base_url() -> str:
    """Base URL for the Bedrock GPT (OpenAI-compatible) endpoint.

    Honors a ``BEDROCK_OPENAI_BASE_URL`` override (e.g. a different region);
    otherwise defaults to the us-east-2 'mantle' endpoint.
    """
    return (
        os.environ.get(_BEDROCK_GPT_BASE_URL_ENV, "").strip()
        or _BEDROCK_GPT_DEFAULT_BASE_URL
    ).rstrip("/")


def _responses_api_text(payload: dict) -> str:
    """Extract the assistant text from an OpenAI Responses API payload.

    Tolerates both the convenience ``output_text`` field and the structured
    ``output[].content[].text`` shape.
    """
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    parts: list[str] = []
    for item in payload.get("output", []) or []:
        for content in item.get("content", []) or []:
            text = content.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def _openai_base_url() -> str:
    """Base URL for OpenAI's official API.

    Honors an ``OPENAI_BASE_URL`` override (a proxy, or an OpenAI-compatible
    gateway); otherwise the public endpoint.
    """
    return (
        os.environ.get(_OPENAI_BASE_URL_ENV, "").strip()
        or _OPENAI_DEFAULT_BASE_URL
    ).rstrip("/")


def _judge_bedrock_gpt(
    content: str, criterion: str, model: str, temperature: float = 0.0
) -> JudgeResult:
    """Judge via GPT on Bedrock (OpenAI Responses API), authenticated with a bearer key.

    Bedrock's 'mantle' gateway serves the OpenAI Responses API only, so this is
    ``_judge_openai_responses`` pointed at ``BEDROCK_OPENAI_BASE_URL`` with the
    ``BEDROCK_OPENAI_API_KEY`` bearer key.
    """
    return _judge_openai_responses(
        content,
        criterion,
        model,
        temperature,
        provider="bedrock-gpt",
        base_url=_bedrock_gpt_base_url(),
        api_key_env=_BEDROCK_GPT_API_KEY_ENV,
    )


def _judge_openai(
    content: str, criterion: str, model: str, temperature: float = 0.0
) -> JudgeResult:
    """Judge via GPT on OpenAI's official API (Responses API).

    Same request as :func:`_judge_bedrock_gpt`, against OpenAI's own host with
    ``OPENAI_API_KEY``. This is the cloud-account-free way to run a GPT judge.
    """
    return _judge_openai_responses(
        content,
        criterion,
        model,
        temperature,
        provider="openai",
        base_url=_openai_base_url(),
        api_key_env=_OPENAI_API_KEY_ENV,
    )


def _judge_openai_responses(
    content: str,
    criterion: str,
    model: str,
    temperature: float = 0.0,
    *,
    provider: str,
    base_url: str,
    api_key_env: str,
) -> JudgeResult:
    """Judge through an OpenAI Responses API endpoint (Bedrock or OpenAI itself).

    Posts to ``{base_url}/responses`` with the prompt split into
    ``instructions`` + ``input`` and reads the reply back through
    ``_responses_api_text``, authenticating with a bearer key from
    ``api_key_env``. ``temperature`` is accepted for signature parity with the
    other judges and deliberately never sent: the gpt-5-class reasoning models
    on both endpoints reject an explicit value (HTTP 400), matching the eval
    agent's GPT paths.
    """
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        return JudgeResult(
            passed=False,
            score=0.0,
            reason="",
            ok=False,
            error=f"{provider} judge requires {api_key_env} to be set.",
        )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    request_body = {
        "model": model,
        "instructions": (
            "You are an automated task-evaluation judge. Reply with ONLY the "
            "requested JSON object."
        ),
        "input": _build_prompt(content, criterion),
        "max_output_tokens": 256,
    }
    req = urllib.request.Request(
        f"{base_url}/responses",
        data=json.dumps(request_body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    timeout = _bedrock_gpt_timeout_s()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return _parse_judge_text(_responses_api_text(payload))


def _bedrock_gpt_timeout_s() -> float:
    """Per-request timeout (seconds) for a Responses-API judge call."""
    try:
        return max(1.0, float(os.environ.get("LLM_JUDGE_REQUEST_TIMEOUT_S", "60")))
    except (TypeError, ValueError):
        return 60.0


_MISSING = object()


def _coerce_passed(value: object) -> bool | None:
    """Strictly interpret a judge reply's ``passed`` field.

    The historical ``bool(data.get("passed", False))`` was a single-directional
    scoring bug: a judge that emitted the STRING ``"false"`` (or ``"no"``,
    ``"0"``, or any other non-empty string) produced ``bool("false") == True``
    -- a PASS -- while a real ``false`` boolean correctly stayed a fail. Small
    judge models (e.g. the flash-lite grader backing ``command.txt``) routinely
    emit stringly-typed verdicts, so this inflated semantic checks upward only,
    never downward.

    Rules (consistent with the bare-boolean branch, which honours only an
    unambiguous boolean):
      * a real ``bool``          -> that boolean
      * the string ``'true'``    -> True   (case/space-insensitive)
      * any other string         -> False  (conservative: never invent a pass)
      * anything else / missing  -> None   (undeterminable; caller -> ok=False)
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return None


def _parse_judge_text(text: str) -> JudgeResult:
    """Parse a judge JSON reply, tolerating extra surrounding text."""
    raw = (text or "").strip()
    data = None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except (json.JSONDecodeError, ValueError):
                data = None

    if isinstance(data, bool):
        # A bare JSON boolean is off-contract but UNAMBIGUOUS, so honour it
        # rather than discarding a usable verdict.
        return JudgeResult(
            passed=data,
            score=1.0 if data else 0.0,
            reason=raw[:200],
        )

    if not isinstance(data, dict):
        # Anything else is prose, truncation, or a refusal -- we do not know what
        # the judge decided.
        #
        # This used to guess: ``passed = "true" in lower and "false" not in
        # lower``, so a reply like "this looks true to me" scored a PASS, and
        # because the guess was returned with the default ``ok=True`` it was also
        # written to the shared cache and reused for every identical
        # (content, criterion) pair. A grader that cannot be parsed must not
        # manufacture a verdict: report ``ok=False`` so the caller records an
        # infrastructure failure (still gated as not-passed) instead of a
        # hallucinated pass, and so ``judge()`` can retry it.
        return JudgeResult(
            passed=False,
            score=0.0,
            reason="",
            ok=False,
            error=(
                "malformed judge reply: expected a JSON object with a "
                f"'passed' field, got {raw[:200]!r}"
            ),
        )

    coerced = _coerce_passed(data.get("passed", _MISSING))
    if coerced is None:
        # A dict reply whose ``passed`` field is missing or is neither a boolean
        # nor a string is off-contract: we cannot know the verdict. Do NOT guess
        # (the old default turned a missing field into a fail-with-ok=True, and
        # ``bool(...)`` turned stringly verdicts into passes). Report ok=False so
        # the caller records an infrastructure failure (still gated as
        # not-passed) and ``judge()`` can retry -- mirroring the malformed
        # non-dict branch above and the bare-boolean branch's "honour only an
        # unambiguous verdict" rule.
        return JudgeResult(
            passed=False,
            score=0.0,
            reason="",
            ok=False,
            error=(
                "unparseable 'passed' field in judge reply: expected a boolean "
                f"or 'true'/'false', got {data.get('passed', None)!r}"
            ),
        )
    passed = coerced
    if "score" in data:
        try:
            score = max(0.0, min(1.0, float(data["score"])))
        except (TypeError, ValueError):
            score = 1.0 if passed else 0.0
    else:
        score = 1.0 if passed else 0.0
    return JudgeResult(
        passed=passed,
        score=score,
        reason=str(data.get("reason", "")),
    )
