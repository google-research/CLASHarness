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

"""Strict validation for Conflict-VLM scenario generation."""

from __future__ import annotations

from typing import Any

from clasharness.scenario.conflict_vlm_spec import ConflictVLMScenarioSpec
from clasharness.scenario.conflict_vlm_spec import ConflictVLMSeed
from clasharness.tools.agent_tool_spec import ToolSpecViolation
from clasharness.tools.agent_tool_spec import validate_command_against_spec


import re

_FUTURE_ROUND_REF_RE = re.compile(r"\b(?:ROUND|R|q|r)[_\- ]?(?:0?[2-9]|[1-9][0-9])\b", re.IGNORECASE)

# Effect ``type``/``effect_type`` values that denote a deterministic file-write
# primitive rather than a CLI command. These are valid runtime effects (applied
# directly to the virtual workspace) and are therefore skipped by command
# validation. Mirrors ``conflict_vlm_materializer._SUBAGENT_FILE_EFFECT_TYPES``.
_MOCK_FILE_EFFECT_TYPES = frozenset(
    {"file_write", "file_create", "write_file", "create_file"}
)


def iter_mock_command_effects(effects: Any) -> list[str]:
  """Yield the executable-command ``state_effects`` entries for a mock subagent.

  A subagent ``state_effects`` list may hold: a raw CLI string, a
  ``{command|cmd: ...}`` dict, or a file-write dict
  (``{type: file_create, path, content}``). Only the first two forms are CLI
  commands that must satisfy ``validate_command_against_spec``; file-write dicts
  are deterministic file primitives and are skipped. This mirrors the runtime
  normalization in ``conflict_vlm_materializer._flatten_subagent_state_effects``
  so that what we validate matches what actually executes.
  """
  out: list[str] = []
  if not effects:
    return out
  if not isinstance(effects, list):
    effects = [effects]
  for eff in effects:
    if isinstance(eff, str):
      if eff.strip():
        out.append(eff.strip())
      continue
    if not isinstance(eff, dict):
      continue
    kind = str(eff.get("type") or eff.get("effect_type") or "").strip().lower()
    path = eff.get("path") or eff.get("file") or eff.get("filename")
    if kind in _MOCK_FILE_EFFECT_TYPES and path:
      continue  # file-write primitive, not a CLI command
    command = eff.get("command") or eff.get("cmd")
    if command and str(command).strip():
      out.append(str(command).strip())
  return out


class ConflictValidationError(ValueError):
  """Raised when a generated conflict scenario fails strict validation."""


def validate_conflict_scenario_strict(
    spec: ConflictVLMScenarioSpec,
    seed: ConflictVLMSeed | None = None,
) -> None:
  """Validate that generated conflicts match taxonomy.

  ``seed`` is optional and backward compatible: when provided, the seed-gated
  coordination-integrity check (:func:`_validate_protocol_completion`, M8 CL1)
  additionally runs. Callers that only have the materialized spec may omit it;
  the seed-gated check is then skipped.
  """
  errors = []

  # 1. Temporal Staleness Check
  if "temporal_staleness" in spec.conflict_taxonomy:
    has_temporal_trigger = any(
        update.trigger_virtual_time is not None for update in spec.updates
    )
    if not has_temporal_trigger:
      errors.append(
          "Taxonomy includes 'temporal_staleness' but no "
          "ConflictScenarioUpdate defines a 'trigger_virtual_time'. "
          "Temporal conflicts require async injection."
      )

  # 2. Mock Subagents Check
  if (
      "subagent_hallucination" in spec.conflict_taxonomy
      or "subagent_delay" in spec.conflict_taxonomy
  ):
    has_mock_subagents = any(
        len(round_spec.mock_subagents) > 0 for round_spec in spec.rounds
    )
    if not has_mock_subagents:
      errors.append(
          "Taxonomy requires subagent conflicts but no "
          "MockSubAgentBehavior was generated in any round."
      )

  # 3. Valid Mock Subagent Behavior
  for round_spec in spec.rounds:
    for mock_idx, mock in enumerate(round_spec.mock_subagents):
      if not mock.agent_role:
        errors.append(
            f"Round {round_spec.round_id} Mock {mock_idx} is missing "
            "'agent_role'"
        )
      if (
          "subagent_delay" in spec.conflict_taxonomy
          and mock.virtual_duration_seconds <= 0
      ):
        errors.append(
            f"Round {round_spec.round_id} Mock {mock_idx} has duration <= 0 "
            "but taxonomy requires 'subagent_delay'."
        )
      # Decision 4 (design 4.2): every command-shaped state_effect must be a
      # valid, executable OpenClaw command (prose/invented commands are caught
      # here as a final gate, complementing the in-loop generator validation).
      for effect in iter_mock_command_effects(mock.state_effects):
        try:
          validate_command_against_spec(
              effect, allowed_families=spec.allowed_tool_families
          )
        except ToolSpecViolation as exc:
          errors.append(
              f"Round {round_spec.round_id} Mock {mock_idx} state_effect is "
              f"not an executable command: '{effect}' ({exc})"
          )

  # 4. Leakage Validation (Borrowing from LLM PRO)
  for round_spec in spec.rounds:
    round_idx = _round_index(round_spec.round_id)
    instruction_lower = round_spec.instruction.lower()

    # Prevent literal Asset ID leakage
    for asset_id in round_spec.visible_asset_ids:
      if asset_id.lower() in instruction_lower:
        errors.append(
            f"Round {round_spec.round_id}: instruction leaks visible asset ID: '{asset_id}'"
        )

    # Prevent future round reference leaks
    if _visible_text_references_future_round(round_spec.instruction, round_index=round_idx):
      errors.append(
          f"Round {round_spec.round_id}: instruction leaks future round references"
      )

  # 5. Decisive-value confidentiality (LOCUS-AWARE backstop).
  #    Post-generation safety net behind the resolver-side confidentiality gate
  #    (``_validate_round_confidentiality``); it mirrors that gate's locus
  #    awareness so it never rejects a round the gate accepted:
  #      * INSTRUCTION check -- UNIVERSAL (every round): the question must never
  #        spell out its own answer, wherever the value is grounded.
  #      * WORKSPACE/CHAT check -- ONLY for an ``asset:`` locus (a vision round).
  #        There the value is meant to live only in the rendered pixels, so it
  #        must NOT appear in any agent-visible TEXT the round injects, else the
  #        round is text-solvable without looking at the image (the "difficulty
  #        collapse"). A ``file``/``computed``/unknown locus is a legitimately
  #        text-grounded round (the value is READ from a workspace file), so its
  #        presence in workspace text is expected and is NOT a leak.
  #    A pure ASSERTION (fails generation -> retry): never scrubs/repairs, and
  #    never scans the locus asset itself (the value legitimately lives there).
  updates_by_id = {str(u.update_id): u for u in spec.updates}
  for round_spec in spec.rounds:
    value = str(round_spec.resolved_value or "").strip()
    if not value:
      continue
    if _value_leaks_in_text(value, round_spec.instruction):
      errors.append(
          f"Round {round_spec.round_id}: instruction leaks decisive "
          f"resolved_value '{value}' (the question must not state its own answer)"
      )
    # Vision-only confinement applies solely to asset-locus (vision) rounds.
    if not _is_asset_locus(round_spec.resolved_value_locus):
      continue
    for update_id in round_spec.update_ids:
      update = updates_by_id.get(str(update_id))
      if update is None:
        continue
      if _value_leaks_in_text(value, _update_visible_text(update)):
        errors.append(
            f"Round {round_spec.round_id}: update '{update_id}' leaks decisive "
            f"resolved_value '{value}' into agent-visible workspace/chat text "
            f"(asset-locus round: it must be readable only from the pixels)"
        )
    # Base chat sessions (``spec.sessions``) are scenario-wide and authored
    # before the round's decisive value is chosen, so they are the leak channel
    # the round-scoped ``update`` scan above cannot see. They are rendered into
    # the session JSONL the agent reads, so a verbatim decisive value here makes
    # an asset-locus round text-solvable without opening the image.
    for session in getattr(spec, "sessions", None) or []:
      if _value_leaks_in_text(value, _session_visible_text(session)):
        errors.append(
            f"Round {round_spec.round_id}: base chat session "
            f"'{getattr(session, 'session_id', '?')}' leaks decisive "
            f"resolved_value '{value}' into agent-visible chat text "
            f"(asset-locus round: it must be readable only from the pixels)"
        )

  # 6. Coordination-integrity (M8 CL1, seed-gated / behaviour-gated).
  #    Only runs when a ``seed`` is supplied AND it opts in via
  #    ``require_protocol_completion``; a no-op for every existing seed (E6). It
  #    asserts the seed's core protocol mechanic is actually driven to completion
  #    in the graded gold trajectory, catching the "coordination theatre" failure
  #    where the scenario narrates a protocol it never completes or grades.
  errors.extend(_validate_protocol_completion(spec, seed))

  # 7. Forced-async contract (M8 CL2, seed-gated / behaviour-gated). Only runs
  #    when a ``seed`` opts in via ``force_async_wait``; a no-op for every other
  #    seed (E6). It asserts the scenario can actually grade its forced-async
  #    mechanic -- a delayed sub-agent completion exists (so the within_window
  #    check is injected) and the gold trajectory genuinely delegates-then-waits
  #    -- catching the "async theatre" failure where the knob is set but nothing
  #    async is graded.
  errors.extend(_validate_async_wait_contract(spec, seed))

  if errors:
    raise ConflictValidationError(
        "Strict Validation Failed:\n- " + "\n- ".join(errors)
    )


def _value_leaks_in_text(value: str, text: str) -> bool:
  """True if decisive ``value`` appears verbatim in agent-visible ``text``.

  Case-insensitive and whitespace-normalized. Very short values (<3 non-space
  chars) are ignored to avoid matching ubiquitous substrings. Simple
  alphanumeric tokens (e.g. ``v4.2``, ``SALT_2024``) are matched on token
  boundaries; multi-word / punctuated values fall back to a normalized substring
  match (e.g. ``roll back to v4.2``).
  """
  v = re.sub(r"\s+", " ", str(value or "")).strip().lower()
  if len(v.replace(" ", "")) < 3:
    return False
  hay = re.sub(r"\s+", " ", str(text or "")).lower()
  if not hay:
    return False
  if re.fullmatch(r"[a-z0-9][a-z0-9_.\-]*", v):
    return re.search(rf"(?<![a-z0-9]){re.escape(v)}(?![a-z0-9])", hay) is not None
  return v in hay


def _is_asset_locus(resolved_value_locus: str | None) -> bool:
  """True if the decisive value's declared locus is an ``asset:<id>`` -- a vision
  round whose value must live only in the rendered pixels.

  Mirrors the ``asset`` case of ``conflict_llm_pipeline._parse_resolved_locus``
  (kept self-contained here to avoid importing the pipeline). ``file:``,
  ``computed`` and unknown/empty loci are legitimately text-grounded and return
  ``False`` -- their value may appear in agent-visible workspace text.
  """
  return str(resolved_value_locus or "").strip().lower().startswith("asset:")


def _update_visible_text(update: Any) -> str:
  """Concatenate the agent-visible text an update injects into workspace/chat.

  Only fields that surface to the solving agent (file bodies/paths and chat
  message content) are included; internal bookkeeping (ids, anchors, metadata)
  is ignored. Tolerant of loosely-typed dict rows emitted by the generator.
  """
  parts: list[str] = []
  for add in getattr(update, "workspace_adds", []) or []:
    if isinstance(add, dict):
      for key in ("content", "text", "body", "path", "filename", "name"):
        val = add.get(key)
        if val:
          parts.append(str(val))
  for msg in getattr(update, "session_appends", []) or []:
    if isinstance(msg, dict):
      for key in ("content", "text", "message", "body"):
        val = msg.get(key)
        if val:
          parts.append(str(val))
  return "\n".join(parts)


def _session_visible_text(session: Any) -> str:
  """Concatenate the agent-visible message text of a base chat session.

  Base sessions (``spec.sessions``) are the scenario-wide chat transcripts the
  materializer renders into session JSONL the solving agent reads. Handles both
  the parsed ``ScenarioSession``/``SessionMessage`` objects (``.messages`` /
  ``.text``) and the loosely-typed dict rows the generator may still emit.
  """
  parts: list[str] = []
  if isinstance(session, dict):
    messages = (
        session.get("messages")
        or session.get("chat")
        or session.get("dialogs")
        or []
    )
  else:
    messages = getattr(session, "messages", None) or []
  for msg in messages:
    text = getattr(msg, "text", None)
    if text is None and isinstance(msg, dict):
      for key in ("text", "content", "message", "body"):
        val = msg.get(key)
        if val:
          text = val
          break
    if text:
      parts.append(str(text))
  return "\n".join(parts)


def _round_index(round_id: str | None) -> int:
  text = str(round_id or "").lower()
  if len(text) == 3 and (text.startswith("r") or text.startswith("q")) and text[1:].isdigit():
    return int(text[1:])
  return 0


# Future-round leak heuristic tuning. A round/question index ``N`` counts as a
# leak only when ``round_index < N <= _FUTURE_ROUND_CAP``. Bare ``R<N>``/``Q<N>``
# tokens are flagged ONLY when a round-context cue word sits within
# ``_BARE_TOKEN_CONTEXT_WINDOW`` characters, so domain noise (a release tag
# ``R2``, a rack id ``R12``, a fiscal quarter ``Q3``) no longer false-positives.
_FUTURE_ROUND_CAP = 24
_BARE_TOKEN_CONTEXT_WINDOW = 24
_ROUND_CONTEXT_CUE = re.compile(
    r"ROUND|QUESTION|NEXT|LATER|UPCOMING|SUBSEQUENT|FUTURE"
)
_EXPLICIT_ROUND_RE = re.compile(r"\bROUND[_\- ]?(\d{1,2})\b")
_BARE_ROUND_TOKEN_RE = re.compile(r"\b[RQ][_\- ]?(\d{1,2})\b")


def _visible_text_references_future_round(text: str, *, round_index: int) -> bool:
  """True if visible ``text`` references a FUTURE round/question index.

  Two matching tiers keep the check precise (C: false-positive reduction):
    * EXPLICIT ``ROUND[_- ]?N`` -- always a leak (an unambiguous round ref).
    * BARE ``R<N>`` / ``Q<N>`` -- a leak ONLY when a round-context cue word
      (``round``, ``next``, ``later`` ...) sits within
      ``_BARE_TOKEN_CONTEXT_WINDOW`` chars. Without a cue, a bare ``R2``/``Q3``
      is domain noise (release tag / rack id / fiscal quarter), not a round
      reference, so flagging it is a false positive that aborts otherwise-valid
      generation (the it_deployment_gate_v01 failure mode).
  """
  upper = str(text or "").upper()

  # Explicit round references are always decisive.
  for match in _EXPLICIT_ROUND_RE.finditer(upper):
    try:
      value = int(match.group(1))
    except ValueError:
      continue
    if round_index < value <= _FUTURE_ROUND_CAP:
      return True

  # Bare R<N>/Q<N> tokens require a nearby round-context cue.
  for match in _BARE_ROUND_TOKEN_RE.finditer(upper):
    try:
      value = int(match.group(1))
    except ValueError:
      continue
    if not (round_index < value <= _FUTURE_ROUND_CAP):
      continue
    lo = max(0, match.start() - _BARE_TOKEN_CONTEXT_WINDOW)
    hi = min(len(upper), match.end() + _BARE_TOKEN_CONTEXT_WINDOW)
    window = upper[lo:match.start()] + upper[match.end():hi]
    if _ROUND_CONTEXT_CUE.search(window):
      return True
  return False


def enforce_protocol_completion(
    spec: ConflictVLMScenarioSpec,
    seed: ConflictVLMSeed | None,
) -> None:
  """Raise if an opted-in seed's gold trajectory never completes its mechanic.

  The generation-time subset of the strict validator: it runs ONLY the seed-gated
  coordination-integrity check (:func:`_validate_protocol_completion`), so wiring
  it into materialization enforces the M8 CL1 contract without newly turning on
  the other strict checks (asset/future-round leaks, decisive-value
  confidentiality, mock-subagent validity), which remain post-generation test
  backstops via :func:`validate_conflict_scenario_strict`. A no-op unless ``seed``
  opts in via ``require_protocol_completion`` (E6).
  """
  errors = _validate_protocol_completion(spec, seed)
  if errors:
    raise ConflictValidationError(
        "Strict Validation Failed:\n- " + "\n- ".join(errors)
    )


def _required_completion_effects(seed: ConflictVLMSeed) -> set[str]:
  """Effects the seed's CORE mechanic must drive to completion.

  Resolved from the conflict-plugin registry: the seed's ``archetype`` token (or,
  when absent, its ``conflict_taxonomy`` tokens) routes to the plugin(s) that
  implement the mechanic, and each plugin declares the effect keys its evals
  assert (``ConflictPluginSpec.emits_effects``, e.g. the ``multi_device_pairing``
  plugin declares ``("pairing_completed",)``). Restricting to the archetype token
  keeps the requirement surgical -- only the seed's primary mechanic must
  complete, not every effect any routed difficulty plugin might emit. Lazy import
  keeps this module free of an import cycle with the plugins package.
  """
  from clasharness.scenario.plugins import get_plugins_for_taxonomy  # pylint: disable=g-import-not-at-top

  archetype = str(getattr(seed, "archetype", "") or "").strip()
  tokens = (
      [archetype]
      if archetype
      else [str(t) for t in (getattr(seed, "conflict_taxonomy", None) or [])]
  )
  effects: set[str] = set()
  for plugin in get_plugins_for_taxonomy(tokens):
    spec = getattr(plugin, "SPEC", None)
    if spec is not None and getattr(spec, "emits_effects", None):
      effects.update(spec.emits_effects)
  return effects


def _completion_command_prefixes(required_effects: set[str]) -> dict[str, list[str]]:
  """Map each required effect -> the gold-command prefixes whose success emits it.

  Reverses the pipeline's ``_EFFECT_COMMAND_NAMES`` (dotted command name ->
  effect, e.g. ``pairing.confirm`` -> ``pairing_completed``); the dotted key
  becomes a spaced command prefix (``pairing confirm``) that a
  ``solution_commands`` entry must start with to count as completing the effect.
  Lazy import avoids an import cycle with the pipeline (which imports this
  module).
  """
  from clasharness.scenario.conflict_llm_pipeline import _EFFECT_COMMAND_NAMES  # pylint: disable=g-import-not-at-top

  out: dict[str, list[str]] = {effect: [] for effect in required_effects}
  for dotted, effect in _EFFECT_COMMAND_NAMES.items():
    if effect in required_effects:
      out[effect].append(str(dotted).replace(".", " ").strip().lower())
  return out


def _validate_protocol_completion(
    spec: ConflictVLMScenarioSpec,
    seed: ConflictVLMSeed | None,
) -> list[str]:
  """Assert an opted-in seed drives its core protocol mechanic to completion.

  Behaviour-gated: returns ``[]`` unless ``seed`` is supplied and opts in via
  ``require_protocol_completion`` (default off -> every existing seed unaffected,
  E6). When enabled, some round's gold ``solution_commands`` must perform the
  completion command whose success emits the mechanic's declared effect (e.g.
  ``pairing confirm`` -> ``pairing_completed``). The checker scaffolder
  deterministically turns that command into a required ``exists`` effect check,
  so verifying the gold trajectory here guarantees the effect is graded; a seed
  that narrates the protocol but never completes it (the "coordination theatre"
  failure) is flagged. A pure assertion: never repairs, only reports.
  """
  if seed is None or not getattr(seed, "require_protocol_completion", False):
    return []

  errors: list[str] = []
  required = _required_completion_effects(seed)
  if not required:
    return [
        "require_protocol_completion is set but the seed's archetype/taxonomy "
        "routes to no conflict plugin declaring emits_effects, so there is no "
        "protocol completion to enforce (misconfiguration)."
    ]

  effect_to_prefixes = _completion_command_prefixes(required)
  gold_commands = [
      str(cmd).strip().lower()
      for round_spec in spec.rounds
      for cmd in (round_spec.solution_commands or [])
  ]

  for effect in sorted(required):
    prefixes = effect_to_prefixes.get(effect) or []
    if not prefixes:
      errors.append(
          f"require_protocol_completion: declared effect '{effect}' has no known "
          "completion command in _EFFECT_COMMAND_NAMES, so its completion cannot "
          "be verified (register the emitting command or drop the effect)."
      )
      continue
    if not any(cmd.startswith(pfx) for cmd in gold_commands for pfx in prefixes):
      wanted = " | ".join(sorted(prefixes))
      errors.append(
          "require_protocol_completion: no round's gold solution_commands "
          f"performs the completion command for effect '{effect}' (expected a "
          f"command starting with: {wanted}). The scenario would grade no "
          f"'{effect}' effect check and collapses into coordination theatre "
          "(the protocol is narrated but never completed)."
      )
  return errors


def enforce_async_wait_contract(
    spec: ConflictVLMScenarioSpec,
    seed: ConflictVLMSeed | None,
) -> None:
  """Raise if an opted-in async seed cannot actually grade its async mechanic.

  The generation-time subset of the strict validator for M8 CL2 (``force_async_wait``,
  F6): it runs ONLY the seed-gated async-wait contract
  (:func:`_validate_async_wait_contract`), so wiring it into materialization
  enforces the CL2 contract without newly turning on the other strict checks. A
  no-op unless ``seed`` opts in via ``force_async_wait`` (E6).
  """
  errors = _validate_async_wait_contract(spec, seed)
  if errors:
    raise ConflictValidationError(
        "Strict Validation Failed:\n- " + "\n- ".join(errors)
    )


def _round_has_delayed_subagent(round_spec: Any) -> bool:
  """Whether a round declares a mock sub-agent with a delayed completion.

  The materializer injects the sender-scoped ``within_window`` temporal check
  (:func:`conflict_vlm_materializer._build_async_within_window_check`) ONLY when a
  round has a mock sub-agent with a non-empty ``agent_role`` AND a positive
  ``virtual_duration_seconds`` -- the latter becomes the completion event's
  ``trigger_time`` (the derived SLA deadline). Without such a delayed completion
  there is nothing async to grade and the ``force_async_wait`` requirement is
  vacuous ("async theatre").
  """
  for mock in getattr(round_spec, "mock_subagents", None) or []:
    role = str(getattr(mock, "agent_role", "") or "").strip()
    try:
      duration = int(getattr(mock, "virtual_duration_seconds", 0) or 0)
    except (TypeError, ValueError):
      duration = 0
    if role and duration > 0:
      return True
  return False


def _gold_delegates_then_waits(
    round_spec: Any, allowed_families: list[str]
) -> bool:
  """Whether a round's gold ``solution_commands`` delegates then waits.

  Reuses the pipeline's :func:`_has_delegate_then_wait` (single source of truth,
  so the materialize-time backstop cannot drift from the generation-time gate):
  the reference trajectory must issue an ``agent delegate`` and THEN a ``wait``
  that advances the virtual clock past the sub-agent's completion, so the delayed
  reply lands in ``messages_sent`` and the within-window check can grade it. Lazy
  import avoids the import cycle with the pipeline (which imports this module).
  """
  from clasharness.scenario.conflict_llm_pipeline import _has_delegate_then_wait  # pylint: disable=g-import-not-at-top

  commands = [
      {"command": str(cmd or "")}
      for cmd in (getattr(round_spec, "solution_commands", None) or [])
  ]
  return _has_delegate_then_wait(commands, list(allowed_families or []))


def _validate_async_wait_contract(
    spec: ConflictVLMScenarioSpec,
    seed: ConflictVLMSeed | None,
) -> list[str]:
  """Assert an opted-in async seed can actually grade its forced-async mechanic.

  Behaviour-gated: returns ``[]`` unless ``seed`` is supplied and opts in via
  ``force_async_wait`` (default off -> every existing seed unaffected, E6). When
  enabled it enforces two correct-by-construction preconditions for the F6
  ablation gate (a no-wait trajectory must fail):

    1. Some round declares a delayed mock sub-agent (:func:`_round_has_delayed_subagent`),
       so the materializer actually injects the sender-scoped ``within_window``
       check; otherwise the async requirement is never graded.
    2. Some *such* round's gold ``solution_commands`` delegates to a sub-agent and
       THEN waits (:func:`_gold_delegates_then_waits`), so the reference
       trajectory is a genuine async-delegation playbook rather than a synchronous
       one that merely narrates delegation.

  A pure assertion: never repairs, only reports. Mirrors the generation-time
  pipeline gate for seeds that reach materialization from disk (bypassing it).
  """
  if seed is None or not getattr(seed, "force_async_wait", False):
    return []

  delayed_rounds = [r for r in spec.rounds if _round_has_delayed_subagent(r)]
  if not delayed_rounds:
    return [
        "force_async_wait is set but no round declares a delayed mock sub-agent "
        "(a mock_subagent with agent_role and virtual_duration_seconds > 0). The "
        "materializer would inject no within_window temporal check, so the forced-"
        "async requirement is never graded (async theatre)."
    ]

  allowed_families = list(getattr(spec, "allowed_tool_families", None) or [])
  if not any(
      _gold_delegates_then_waits(r, allowed_families) for r in delayed_rounds
  ):
    return [
        "force_async_wait is set but no delayed-sub-agent round's gold "
        "solution_commands delegates to a sub-agent and THEN waits (expected an "
        "'agent delegate ...' command followed later by a 'wait <seconds>'). The "
        "reference trajectory does not actually go async, so it cannot exercise "
        "the injected within_window check."
    ]

  return []

