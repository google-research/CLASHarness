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

"""Shared constant tables extracted from examples/train_and_eval.py.

Leaf module (no intra-package imports) so every other train_eval submodule and
the facade can import these with no circular-import risk.
"""
from __future__ import annotations


_VALID_CONFIG_PATHS = {
    "agent.model",
    "agents.defaults.model.primary",
    "gateway.port",
    "gateway.bind",
    "gateway.auth.mode",
    "gateway.auth.token",
    "gateway.remote.url",
}

_PLACEHOLDER_VALUES = {
    "TIMEZONE",
    "LOCATION",
    "TITLE",
    "DATETIME",
    "DATE",
    "NAME",
    "SCHEDULE",
    "QUERY",
    "BODY",
    "SUBJECT",
    "EMAIL",
    "ADDR",
    "TARGET",
    "TEXT",
    "ROLE",
    "PATH",
    "SRC",
    "DST",
    "ID",
    "FOLDER",
    "PRIORITY",
    "MODEL",
    "LOC",
    "TZ",
    "N",
}

_DOMAIN_HINT_GROUPS: dict[str, tuple[str, ...]] = {
    "messaging": ("channel_mgmt",),
    "channel_mgmt": ("messaging",),
    "setup_config": ("monitoring",),
    "monitoring": ("setup_config",),
    "cron_webhook": ("monitoring",),
    "conflict": ("asset", "multimodal", "file", "agent_mgmt", "messaging", "tasks", "setup_config", "monitoring"),
}

# Models that reject the `temperature` parameter (e.g. some newer Claude
# reasoning models on Vertex). Populated lazily on the first 400 response so we
# only waste at most one request per model per process.
_ANTHROPIC_MODELS_OMIT_TEMPERATURE: set[str] = set()

# Extra output tokens to add on top of the extended-thinking ``budget_tokens``
# when raising ``max_tokens`` for a native Claude tool call. The Messages API
# requires ``max_tokens > budget_tokens`` (the budget is consumed by thinking
# before any visible output), so this headroom leaves room for the ``tool_use``
# block the model still has to emit after it finishes thinking.
_ANTHROPIC_THINKING_OUTPUT_HEADROOM: int = 1024

# Effort level requested (via the top-level ``output_config`` object) alongside
# adaptive extended thinking for native Claude tool calls. "high" is the API
# default and the level supported on Claude Sonnet 5 / Opus 4.8; it also nudges
# the model to reason before each forced tool step. See
# https://platform.claude.com/docs/en/build-with-claude/effort.
_ANTHROPIC_ADAPTIVE_EFFORT: str = "high"

# Shared --llm-reasoning-effort values that mean "turn thinking off" for Claude.
# Only ``none`` is a true "off" switch: for Claude ``minimal`` is NOT off, it is
# the lowest *real* thinking rung and maps to ``low`` (see
# _ANTHROPIC_EFFORT_LEVELS in agents.py). This set is consumed only by the native
# Claude tool path, where a value here sends ``thinking={"type":"disabled"}`` +
# effort=_ANTHROPIC_MIN_EFFORT; ``minimal`` instead falls through to the normal
# adaptive-thinking path at effort=low.
#
# The other two families do their own mapping and do not consult this set:
#   * gemini - clamps to MINIMAL; Gemini 3's ThinkingConfig has no true "off",
#              MINIMAL is the floor (see _GEMINI_THINKING_LEVELS in agents.py).
#   * openai - passed straight through to ``reasoning.effort``; the Responses API
#              documents ``none`` as bypassing reasoning (model-dependent), and
#              ``minimal`` is a real low rung, so no special-casing is needed.
_EFFORT_OFF_VALUES: frozenset[str] = frozenset({"none"})

# Effort paired with ``thinking={"type":"disabled"}`` when the shared
# --llm-reasoning-effort asks Claude to stop thinking (a value in
# _EFFORT_OFF_VALUES, i.e. ``none``). Anthropic's ``output_config.effort`` has no
# ``none`` rung -- ``low`` is the documented floor -- and ``disabled`` thinking
# is rejected alongside the xhigh/max rungs, so ``low`` is the correct partner.
# See https://platform.claude.com/docs/en/build-with-claude/effort.
_ANTHROPIC_MIN_EFFORT: str = "low"

# Fixed extended-thinking ``budget_tokens`` used as the fallback when a backend
# rejects the preferred ``thinking={"type":"adaptive"}`` + ``output_config``
# effort request (e.g. an older Vertex/Anthropic endpoint that predates adaptive
# thinking). Kept deliberately generous so the fallback still leaves Claude real
# room to reason before emitting its ``tool_use`` block.
_ANTHROPIC_FIXED_THINKING_BUDGET: int = 8192

# Learned per-model minimum ThinkingConfig.thinking_level. Not every Gemini
# model accepts every level: Gemini 3 Pro rejects MINIMAL (its floor is LOW)
# with a 400 "thinking_level MINIMAL is not supported by this model", while
# Flash accepts MINIMAL. The shared --llm-reasoning-effort 'none'/'minimal' both
# map to MINIMAL, so we learn each model's floor from that rejection and clamp
# UP to it, wasting at most one call per model per process. Keyed by model id.
_GEMINI_MODEL_MIN_THINKING_LEVEL: dict[str, str] = {}

# Learned per-model image long-edge caps (pixels). Anthropic on Vertex rejects
# oversized images with a 400 that names the limit (e.g. Sonnet high-fidelity
# 2576, standard 1568). We start at max fidelity (send originals), learn the cap
# from the first such rejection, and pre-shrink to it for later calls. Keyed by
# model id and populated lazily so we waste at most one call per model.
_ANTHROPIC_MODEL_IMAGE_EDGE: dict[str, int] = {}

# Memoize downscaled images so full-history requests do not re-encode the same
# asset on every step. Keyed by (sha1 of source base64, target long edge).
_DOWNSCALED_IMAGE_CACHE: dict[tuple[str, int], str] = {}

# Fallback long-edge caps to try (in order) when a dimension error does not name
# a parseable pixel limit.
_ANTHROPIC_IMAGE_EDGE_FALLBACKS: tuple[int, ...] = (2000, 1568)

# Total-payload byte budget for an Anthropic/Vertex request. Vertex rejects
# requests larger than 30MB with a 400 "The request size (N bytes) exceeds
# 30.000MB limit. FAILED_PRECONDITION" -- a TOTAL-payload cap that the per-image
# pixel-dimension healing (_ANTHROPIC_IMAGE_EDGE_FALLBACKS above) never triggers
# on, because no single image is individually oversized. Conflict-VLM rounds
# accumulate every round-visible image (r10 ~= 25 images, several 4K), so the
# full-history payload trips this. We proactively downscale images until the
# estimated request is under this budget, kept a couple MB below the hard 30MB
# limit to leave headroom for text, JSON overhead, and estimate slack.
_ANTHROPIC_REQUEST_BYTE_BUDGET: int = 28_000_000

# Long-edge caps (pixels), most-fidelity first, tried in order when shrinking a
# request to fit _ANTHROPIC_REQUEST_BYTE_BUDGET. Descends well below the pixel
# fallbacks so a many-image round can still be squeezed under the byte cap.
_ANTHROPIC_BYTE_BUDGET_EDGE_LADDER: tuple[int, ...] = (2000, 1568, 1280, 1024, 768)

_SIDE_EFFECT_HINTS: tuple[str, ...] = (
    "add ",
    "create ",
    "send ",
    "schedule",
    "reschedule",
    "set up",
    "book ",
    "keep ",
    "put ",
    "configure",
    "report",
    "save ",
    "update ",
    "move ",
    "write ",
)

_WEATHER_CODE_TEXT = {
    0: "clear",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "rime fog",
    51: "light drizzle",
    53: "drizzle",
    55: "dense drizzle",
    61: "rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "snow",
    73: "moderate snow",
    75: "heavy snow",
    80: "rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
}

