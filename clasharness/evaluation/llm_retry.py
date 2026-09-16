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

"""Low-layer LLM retry and provider-shape primitives shared by agent and judge.

These are pure helpers (no third-party deps) intentionally placed in the
``clasharness`` layer so BOTH the higher ``train_eval`` agent stack and the
in-layer ``clasharness.evaluation.llm_judge`` can reuse ONE retry-classification
and backoff policy. ``train_eval.llm_errors`` re-exports these for backward
compatibility, so existing importers keep working while the dependency direction
stays ``train_eval -> clasharness`` (never the reverse).

Alongside the retry policy this module owns the small set of *provider quirk*
helpers that every Anthropic caller needs (sampling-param rejection, response
block extraction). They live here for the same reason: there must be exactly
one copy, or a fix applied to one caller silently skips the others -- which is
precisely how ``llm_judge`` came to crash on ``'ThinkingBlock' object has no
attribute 'text'`` long after the scenario client had been fixed.
"""
from __future__ import annotations

import functools
import inspect
import random
from collections.abc import Mapping
from typing import Any


def _is_retryable_llm_error(exc: Exception) -> bool:
  # Structured numeric status codes are the most reliable signal for provider
  # backends. gemini/Vertex raise an APIError carrying .code == 429/503 with a
  # message (e.g. "Quota exceeded") that need not contain any text marker below,
  # so relying on string matching alone would mis-classify real throttling as
  # fatal. Guard against bool (a subclass of int) so a stray True is ignored.
  for attr in ("code", "status_code"):
    code = getattr(exc, attr, None)
    if isinstance(code, bool):
      continue
    if isinstance(code, int) and (code == 429 or 500 <= code <= 599):
      return True
  msg = str(exc).lower()
  retry_markers = (
      "timed out",
      "timeout",
      "temporarily unavailable",
      "connection reset",
      "connection aborted",
      "connection refused",
      "remote end closed connection",
      "network is unreachable",
      "try again",
      "service unavailable",
      "bad gateway",
      "gateway timeout",
      "too many requests",
      "rate limit",
      "http 429",
      "http 500",
      "http 502",
      "http 503",
      "503",
      "server_error",
      "internal server error",
      "unavailable",
      "overloaded",
      "prefill queue",
      "queue is full",
      '"code":429',
      '"code": 429',
      '"status":429',
      '"status": 429',
      '"code":503',
      '"code": 503',
      "resource_exhausted",
      "resource exhausted",
      "quota",
  )
  return any(marker in msg for marker in retry_markers)


# Newer reasoning-class Claude models (Sonnet 5, Opus 4.7/4.8, Opus 5, ...) 400
# on ANY explicit sampling parameter (temperature/top_p/top_k) -- only the model
# default is accepted. This is a permanent per-model property, not a transient
# error, so once observed it is remembered process-wide and shared across BOTH
# the offline generation client and the in-layer judge (which key off the same
# helpers) so a rejection learned by one is not re-paid by the other.
_ANTHROPIC_SAMPLING_DEPRECATED_LEARNED: set[str] = set()

# Substrings (matched against the lower-cased model id, dash- or '@'-form) of
# model families KNOWN to reject explicit sampling params. Kept as substrings so
# both ``claude-sonnet-5@20250929`` and ``claude-sonnet-5`` match.
_ANTHROPIC_SAMPLING_DEPRECATED_MARKERS = (
    "sonnet-5",
    "opus-5",
    "opus-4-7",
    "opus-4.7",
    "opus-4-8",
    "opus-4.8",
)


def _is_anthropic_temperature_unsupported(exc: Exception) -> bool:
    """True when an Anthropic error indicates an explicit sampling param is rejected.

    Newer Claude models accept only their default temperature (and top_p/top_k)
    and return an HTTP 400 such as ``'temperature' is deprecated for this model``
    on an explicit value. Detect that specific rejection so callers can drop the
    sampling param and retry instead of failing the layer. Inspects the
    exception's ``message``/``body`` too because the anthropic SDK carries the
    provider detail there rather than in ``str(exc)``.
    """
    parts = [str(exc)]
    for attr in ("message", "body"):
        val = getattr(exc, attr, None)
        if val:
            parts.append(str(val))
    msg = " ".join(parts).lower()
    if "temperature" not in msg and "top_p" not in msg and "top_k" not in msg:
        return False
    return any(
        marker in msg
        for marker in (
            "deprecated",
            "not supported",
            "unsupported",
            "not allowed",
            # anthropic >= 1.0 dropped the sampling params from the typed
            # messages.create() signature, so passing one raises a local
            # TypeError ("unexpected keyword argument 'temperature'") before any
            # request is made. Callers normally avoid this via
            # anthropic_temperature_kwargs(), but treat it as a rejection too so
            # any path that still passes one degrades instead of hard-failing.
            "unexpected keyword argument",
        )
    )


def anthropic_rejects_sampling_params(model: str) -> bool:
    """True if this Claude model is known to reject explicit sampling params.

    Combines a static list of known reasoning-class families with anything
    learned at runtime via :func:`note_anthropic_sampling_rejected`, so callers
    can PROACTIVELY omit ``temperature`` on the first request instead of paying a
    wasted 400 to discover the rejection. Substring match handles both the
    dash-form (``claude-opus-4-8``) and the ``'@'``-form publisher id
    (``claude-sonnet-5@20250929``).
    """
    if not model:
        return False
    key = model.strip().lower()
    if key in _ANTHROPIC_SAMPLING_DEPRECATED_LEARNED:
        return True
    return any(
        marker in key for marker in _ANTHROPIC_SAMPLING_DEPRECATED_MARKERS
    )


def note_anthropic_sampling_rejected(model: str) -> None:
    """Remember (process-wide) that ``model`` rejects explicit sampling params.

    Called after a request 400s on an explicit temperature so every subsequent
    request -- in this client AND the judge -- proactively omits it.
    """
    if model:
        _ANTHROPIC_SAMPLING_DEPRECATED_LEARNED.add(model.strip().lower())


@functools.lru_cache(maxsize=1)
def anthropic_sdk_takes_temperature() -> bool:
    """True if the installed anthropic SDK types ``temperature`` on messages.create.

    anthropic >= 1.0 removed the sampling params from that signature, so passing
    ``temperature=`` raises a local ``TypeError`` before any request is made.
    The field is still honored on the wire, so on those SDKs it has to travel
    via ``extra_body`` instead -- see :func:`anthropic_temperature_kwargs`.

    Probed once off the real signature rather than off a version string, so a
    back-port, a fork or a future re-addition is read correctly.
    """
    try:
        from anthropic.resources.messages import Messages

        return "temperature" in inspect.signature(Messages.create).parameters
    except Exception:
        # No SDK installed, or an unreadable signature. Assume the modern shape:
        # ``extra_body`` is accepted by both SDK majors, and if it somehow is
        # not, the resulting TypeError is classified as a sampling rejection by
        # _is_anthropic_temperature_unsupported and the caller degrades.
        return False


def anthropic_temperature_kwargs(temperature: float) -> dict[str, Any]:
    """``messages.create`` kwargs that put an explicit ``temperature`` on the wire.

    ``{"temperature": t}`` on SDKs that still type the parameter, and
    ``{"extra_body": {"temperature": t}}`` on anthropic >= 1.0, which serializes
    to the identical JSON body. Models that reject an explicit value still
    answer with the usual HTTP 400, so the drop-and-retry path is unchanged.
    """
    if anthropic_sdk_takes_temperature():
        return {"temperature": temperature}
    return {"extra_body": {"temperature": temperature}}


def anthropic_kwargs_carry_temperature(kwargs: Mapping[str, Any]) -> bool:
    """True if ``kwargs`` sends an explicit temperature, in either spelling."""
    return anthropic_kwargs_temperature(kwargs) is not None


def anthropic_kwargs_temperature(kwargs: Mapping[str, Any]) -> float | None:
    """The explicit temperature ``kwargs`` will put on the wire, else ``None``.

    Reads both spellings, so a caller (or a test) can assert on the value that
    actually reaches the API without knowing which anthropic SDK major is
    installed.
    """
    if "temperature" in kwargs:
        return kwargs["temperature"]
    extra = kwargs.get("extra_body")
    if isinstance(extra, Mapping) and "temperature" in extra:
        return extra["temperature"]
    return None


def anthropic_drop_temperature(kwargs: dict[str, Any]) -> None:
    """Strip an explicit temperature from ``kwargs``, in either spelling.

    Used by the retry paths, which re-issue the SAME kwargs dict after learning
    that the model rejects a temperature.

    ``extra_body`` is REPLACED rather than edited in place: the already-sent
    request shares that nested dict (``**kwargs`` only rebuilds the top level),
    so mutating it would retroactively rewrite the request we just made.
    """
    kwargs.pop("temperature", None)
    extra = kwargs.get("extra_body")
    if isinstance(extra, Mapping):
        remaining = {k: v for k, v in extra.items() if k != "temperature"}
        if remaining:
            kwargs["extra_body"] = remaining
        else:
            kwargs.pop("extra_body", None)


def anthropic_text_from_response(response: Any) -> str:
    """Concatenate the ``text`` blocks of an Anthropic Messages response.

    Claude returns ``content`` as a LIST of typed blocks. A reasoning model may
    prefix ``thinking`` / ``redacted_thinking`` blocks (and a tool-using one may
    emit ``tool_use``), so ``content[0]`` is NOT guaranteed to be the answer
    text: indexing it blindly raises ``'ThinkingBlock' object has no attribute
    'text'``. Select only the ``text`` blocks, tolerating dict- or object-shaped
    blocks and test fakes that expose ``.text`` without a real ``.type``.

    Real SDK blocks always carry a *string* ``type``. Anything else -- ``None``
    on a hand-rolled stub, or an auto-created attribute on a ``MagicMock`` --
    means "this is a fake", so fall back to "does it carry a string ``.text``".
    Demanding a *str* (not merely non-``None``) additionally stops a MagicMock's
    auto-created ``.text`` from being spliced into the answer, and stops a
    non-str payload from blowing up ``str.join``.

    Returns ``""`` when the response carries no text block at all (a pure
    thinking/tool_use turn); callers treat empty as "no answer" and retry, which
    is the correct fail-closed behaviour.
    """
    content = getattr(response, "content", None)
    if content is None and isinstance(response, dict):
        content = response.get("content")
    parts: list[str] = []
    for block in content or []:
        if isinstance(block, dict):
            btype = block.get("type")
            text = block.get("text")
        else:
            btype = getattr(block, "type", None)
            text = getattr(block, "text", None)
        is_text = btype == "text" or (
            not isinstance(btype, str) and isinstance(text, str)
        )
        if is_text and isinstance(text, str) and text:
            parts.append(text)
    return "".join(parts).strip()


def _is_flaky_safety_llm_error(exc: Exception | str) -> bool:
  """True for provider-side INPUT safety pre-check rejections that are observed
  to fire FLAKILY on identical benign input.

  The OpenAI-compatible gateway runs a "safety-checks" classifier on the request
  before the model sees it and returns an HTTP 400
  ``{"code":"validation_error", ... "flagged for potentially high-risk cyber
  activity" ... /docs/guides/safety-checks/cybersecurity}`` for benign benchmark
  prompts whose narrative merely *sounds* security-adjacent (e.g. "obtain the
  signing key to clear the audit block"). Empirically this is probabilistic: the
  SAME task passes on most rounds and trips only occasionally (see the eval
  jsonl -- one family had r06 flagged while r03/r04 passed), so a SMALL bounded
  verbatim retry usually clears it.

  This is deliberately kept SEPARATE from ``_is_retryable_llm_error``: ordinary
  transient errors (429/5xx) use the full ``--llm-request-retries`` budget,
  whereas these safety false-positives get only a small dedicated cap so a
  genuinely deterministic rejection does not burn the whole budget (and the
  in-layer judge, which keys off ``_is_retryable_llm_error``, is unaffected).

  Matches on markers specific to the safety-checks family so generic validation
  400s (e.g. "invalid argument") stay fatal.
  """
  msg = str(exc).lower()
  markers = (
      "safety-checks",  # the /docs/guides/safety-checks/... URL these carry
      "high-risk cyber activity",
      "flagged for potentially high-risk",
  )
  return any(marker in msg for marker in markers)


def _retry_backoff_delay(
    base_delay_s: float, attempt_index: int, cap_s: float = 12.0
) -> float:
  if base_delay_s <= 0:
    return 0.0
  # Equal-jitter exponential backoff: keep half of the capped delay
  # deterministic and randomize the other half so parallel workers do not retry
  # in lockstep, which would re-create the thundering-herd bursts we want to
  # spread out. Returned delay lies in [capped/2, capped]. ``cap_s`` bounds the
  # per-retry wait: the default matches ordinary transient errors, while the
  # server-side deadline class (504) passes a larger cap for a longer wait that
  # gives the overloaded backend time to recover before we re-send.
  capped = min(cap_s, base_delay_s * (2**attempt_index))
  return capped * 0.5 + random.uniform(0.0, capped * 0.5)
