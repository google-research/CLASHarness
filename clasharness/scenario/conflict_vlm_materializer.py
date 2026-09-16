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

"""Materializer for building Conflict-VLM scenarios from seeds."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shlex
from pathlib import Path
from typing import Any, Callable, NamedTuple

from clasharness.core.observation import CheckResult
# M2/M3 borrow the CHECKER's tokenizer on purpose: both mechanisms are only
# sound while "would this text satisfy that check?" is answered exactly the way
# grading answers it. A private reimplementation here would drift out of sync
# and silently void the decoy-safety and gold-solvability guarantees.
from clasharness.evaluation.checkers.effect_checker import _is_subsequence
from clasharness.evaluation.checkers.effect_checker import _tokenize_and_clean
from clasharness.core.task import Difficulty, GroundTruth, Task, TaskData, save_task
from clasharness.evaluation.evaluator import Evaluator
from clasharness.scenario.conflict_llm_pipeline import ConflictLLMPipeline
from clasharness.scenario.conflict_llm_pipeline import _is_code_file
from clasharness.scenario.conflict_llm_pipeline import _MIN_PLATFORM_VOCAB
from clasharness.scenario.conflict_llm_pipeline import _extract_flag_value
from clasharness.scenario.conflict_llm_pipeline import _parse_resolved_locus
from clasharness.scenario.conflict_llm_pipeline import command_spec_for_effect
from clasharness.scenario.conflict_strict_validator import enforce_async_wait_contract
from clasharness.scenario.conflict_strict_validator import enforce_protocol_completion
from clasharness.scenario import design_archetypes
from clasharness.scenario.conflict_vlm_spec import (
    ConflictAgentSpec,
    ConflictInteractionSpec,
    ConflictScenarioUpdate,
    ConflictVLMRoundSpec,
    ConflictVLMScenarioSpec,
    ConflictVLMSeed,
)
from clasharness.scenario.llm_client import ScenarioLLMClient
from clasharness.scenario.spec import MockSubagent, ScenarioEvalItem, ScenarioRound, ScenarioSession, ScenarioSpec, ScenarioWorkspace, session_channel_value
from clasharness.scenario.vlm_materializer import generate_vlm_assets, llm_refine_vlm_assets
from clasharness.scenario.vlm_spec import IMAGE_GENERATORS, IMAGE_KINDS, VLMAssetManifest, VLMAssetSpec, VLMEvalSpec
from clasharness.tools.agent_tool_spec import validate_task_solution_commands


def _tag_decisive_assets(spec: ConflictVLMScenarioSpec) -> int:
  """Tag visual assets that legibly carry a round's decisive value.

  Role (decisive/corroborating) is orthogonal to kind: most assets are
  corroborating and stay untagged. An asset is tagged ``decisive`` only when
  both hold:

    (a) one of its ``anchor_ids`` is an ``expected_anchor_ids`` of some round
        (so the asset backs a deciding anchor the coordinator must cite), and
    (b) that round's non-empty ``resolved_value`` actually appears in the
        asset's grounded text (caption / OCR) -- a precision guard confirming
        the asset really bears that deciding token/state.

  The tag (``evidence_role="decisive"`` + ``expected_token``) drives the
  decisive-photo prompt augmentation and the vision fidelity audit downstream.

  This deliberately spans EVERY visual kind, not just the image-model ones.
  Charts, rich_text and diagrams render to a PNG via headless Chrome and the
  runtime strips their ``ocr_text``/``content_text``/``caption``/``description``
  exactly as it does an image's, so they are every bit as pixel-exclusive -- and
  they are the overwhelming majority of decisive assets (~93-95% measured across
  both generated sets). Restricting the tag to ``IMAGE_KINDS`` meant the
  downstream fidelity audit only ever saw the remaining ~5%, leaving the bulk of
  decisive evidence unverified.

  ``VLMAssetSpec`` is frozen, so this mutates each ``provider_metadata`` dict in
  place and never reassigns the attribute. Returns the count of tagged assets.
  """
  if not spec.assets:
    return 0
  tagged = 0
  for round_spec in spec.rounds:
    resolved = (round_spec.resolved_value or "").strip()
    if not resolved:
      continue
    expected = {
        str(x).strip().upper()
        for x in (round_spec.expected_anchor_ids or [])
        if str(x).strip()
    }
    if not expected:
      continue
    resolved_lc = resolved.lower()
    for asset in spec.assets:
      # First decisive round to claim an asset wins its expected_token.
      if asset.provider_metadata.get("evidence_role") == "decisive":
        continue
      anchors = {str(x).strip().upper() for x in (asset.anchor_ids or [])}
      if not (anchors & expected):
        continue
      grounded = f"{asset.caption or ''}\n{asset.ocr_text or ''}".lower()
      if resolved_lc not in grounded:
        continue
      asset.provider_metadata["evidence_role"] = "decisive"
      asset.provider_metadata["expected_token"] = resolved
      tagged += 1
  return tagged


# ``initial_state_overrides`` entries that are NOT agent-readable text: the
# rendered pixels themselves (stored as a base64 bytes blob) and their manifest.
# Everything else under that key IS readable -- not just ``workspace/``, but the
# app-state documents the email/tasks/approval tool families serve.
_NON_TEXT_STATE_PREFIXES = ("assets/",)
_NON_TEXT_STATE_KEYS = frozenset({"asset_manifest.json"})


def _task_visible_text_surfaces(task_spec: dict[str, Any]) -> dict[str, list[str]]:
  """The three agent-readable TEXT surfaces of one COMPILED task.

  A round only "requires opening the asset" if its decisive value is absent from
  every channel the agent can reach by reading rather than looking. This reads
  those channels off the SHIPPED task rather than re-deriving them from the
  authoring spec, which matters for three reasons:

  * ``task_spec`` is what actually ships and what grades. Every repair layer
    (LLM auditor, consistency, state/protocol resolvers, F1) rewrites it, and
    each can put the decisive value back into a file or a message after the
    generation-time confidentiality gate has already passed the round;
  * re-deriving requires replaying the compiler's update-accumulation rules, and
    a reconstruction that drifts from the compiler reports fiction. Measured on
    the 3.6-flash corpus, 299 of 675 updates (44%) are referenced by no round's
    ``update_ids`` and therefore never ship -- a reconstruction that scanned all
    updates would invent leaks, one that mirrored ``update_ids`` imperfectly
    would miss real ones;
  * the compiled state has surfaces the authoring spec does not expose at all.
    ``initial_state_overrides`` carries app-state documents (``emails.json``,
    ``tasks.json``, ``approval.json``, ...) that the email/tasks tool families
    serve to the agent; scanning only ``spec.workspace.files`` missed them.

  Surfaces returned:

  * ``workspace`` - every readable ``initial_state_overrides`` document (all but
    the asset blobs and their manifest), plus ``file_update`` background events;
  * ``session`` - the session JSONL overrides plus ``session_append`` events;
  * ``mock`` - ``mock_subagents`` and ``subagent_completion`` events, limited to
    ``mock_response`` (reachable by delegating and running ``message search``)
    and ``state_effects`` (executed for real). ``private_knowledge`` and
    ``capabilities`` never reach a solving agent and are not collected.
  """
  private = (task_spec.get("data") or {}).get("private") or {}
  workspace: list[str] = []
  session: list[str] = []
  mock: list[str] = []

  for key, value in (private.get("initial_state_overrides") or {}).items():
    name = str(key)
    if name in _NON_TEXT_STATE_KEYS or name.startswith(_NON_TEXT_STATE_PREFIXES):
      continue
    (session if name.startswith("sessions/") else workspace).append(str(value))

  for event in private.get("background_events") or []:
    if not isinstance(event, dict):
      continue
    kind = event.get("type")
    if kind == "file_update":
      workspace.append(str(event.get("new_content") or ""))
    elif kind == "session_append":
      session.append(str(event.get("message") or ""))
    elif kind == "subagent_completion":
      mock.append(str(event.get("mock_response") or ""))
      mock.append(str(event.get("state_effects") or ""))

  for sub in private.get("mock_subagents") or []:
    if not isinstance(sub, dict):
      continue
    mock.append(str(sub.get("mock_response") or ""))
    mock.append(str(sub.get("state_effects") or ""))

  return {"workspace": workspace, "session": session, "mock": mock}


def _classify_round_vision_exclusivity(
    spec: ConflictVLMScenarioSpec,
    round_spec: ConflictVLMRoundSpec,
    task_spec: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
  """Classify one round as vision-exclusive / text-solvable / etc.

  Returns ``(verdict, leaked_surfaces)``. This is a MEASUREMENT taken on the
  finished spec, deliberately not a restatement of the generation-time
  confidentiality gate's own verdict: between that gate and here the value
  passes through the LLM auditor, the consistency layer, the state and protocol
  resolvers and the F1 self-repair, any of which can write the decisive value
  back into a workspace file or a chat message. Re-deriving it from the shipped
  artifact is the only way to know what the dataset actually contains.

  Verdicts:

  * ``no_resolved_value`` - nothing decisive to protect;
  * ``non_asset_locus`` - the resolver did not place the value in an asset AND
    no text surface carries it either, so this round was never meant to require
    looking and is not answerable off a text channel;
  * ``non_asset_locus_text_solvable`` - a non-asset locus whose value IS
    readable on a text surface. Split out from ``non_asset_locus`` because
    collapsing the two made this report structurally blind to the very rounds
    most likely to be text-solvable: the surface scan used to run only for
    ``asset:`` loci, so the reported denominator was exactly the asset-locus
    population and a value laundered into a ``computed`` locus disappeared from
    its own audit. Measured on two full corpora, the ``rounds:`` line counted
    521 of 521 and 492 of 492 asset-locus rounds -- it reported 1.8%
    text-solvable on a corpus whose true rate was 9.3%, because 54 rounds had
    been re-labelled ``computed``/``file`` after a leak rejection and were not
    in the denominator. A report that hides the failure it exists to find is
    worse than no report;
  * ``not_compiled`` - no compiled ``task_spec`` was supplied, so the shipped
    text surfaces cannot be read. Reported rather than silently guessed from the
    authoring spec: a reconstruction that drifts from the compiler would report
    an exclusivity rate the dataset does not actually have. Checked BEFORE the
    locus split, so an uncompiled round is never silently filed as a clean
    ``non_asset_locus``;
  * ``text_solvable`` - an ``asset:`` locus whose value is readable on at least
    one text surface; the agent can answer without ever opening the asset.
    ``leaked_surfaces`` names which ones;
  * ``unsolvable`` - nothing carries the value in text AND the decisive asset
    failed its fidelity audit, so it is not legible in pixels either. The
    value-asserting checks are relaxed to ``diagnostic`` downstream, so this
    does not mis-score, but the round tests nothing;
  * ``unverified_exclusive`` - text-exclusive, but no vision model ever
    confirmed the pixels are readable (audit skipped, disabled, or the asset was
    never tagged decisive). Probably fine, unproven;
  * ``vision_exclusive`` - text-exclusive AND a vision model confirmed the token
    is legible. The only bucket that is positively known to require looking.
  """
  resolved = (round_spec.resolved_value or "").strip()
  if not resolved:
    return "no_resolved_value", []
  if not isinstance(task_spec, dict) or not task_spec:
    return "not_compiled", []

  # The surface scan runs for EVERY round that has a decisive value, whatever
  # its locus. Gating it on ``asset:`` (as it used to be) meant the one
  # population that most needs auditing -- a value re-labelled ``computed``
  # after the confidentiality gate rejected it for leaking -- was never scanned
  # and never counted, so the report certified a corpus it had not looked at.
  resolved_lc = resolved.lower()
  surfaces = _task_visible_text_surfaces(task_spec)
  leaked = sorted(
      name
      for name, texts in surfaces.items()
      if any(resolved_lc in text.lower() for text in texts)
  )

  locus = (round_spec.resolved_value_locus or "").strip()
  if not locus.startswith("asset:"):
    # A non-asset locus is not required to live in pixels, so a clean one is
    # simply out of scope for the vision question -- but a LEAKING one is
    # text-solvable regardless of what its locus claims, and must be visible.
    if leaked:
      return "non_asset_locus_text_solvable", leaked
    return "non_asset_locus", []

  if leaked:
    return "text_solvable", leaked

  asset_id = locus[len("asset:") :].strip()
  asset = next((a for a in spec.assets if a.id == asset_id), None)
  pm = (asset.provider_metadata or {}) if asset is not None else {}
  if pm.get("fidelity_failed"):
    return "unsolvable", []
  if pm.get("vision_verified") is True:
    return "vision_exclusive", []
  return "unverified_exclusive", []


def _summarize_vision_audit(spec: ConflictVLMScenarioSpec) -> dict[str, Any]:
  """Per-asset vision-audit tallies for the end-of-generation report.

  ``expected_token`` (not ``evidence_role``) marks the decisive population,
  because a failed audit downgrades ``evidence_role`` to ``corroborating`` and
  counting on that would make the failures disappear from their own report.
  """
  decisive = [
      a for a in spec.assets
      if str((a.provider_metadata or {}).get("expected_token") or "").strip()
  ]
  verified = failed = never_ran = 0
  skipped_reasons: dict[str, int] = {}
  channels: dict[str, int] = {}
  for asset in decisive:
    pm = asset.provider_metadata or {}
    channel = pm.get("vision_audit_channel")
    if channel:
      channels[str(channel)] = channels.get(str(channel), 0) + 1
    else:
      never_ran += 1
    if pm.get("fidelity_failed"):
      failed += 1
    elif pm.get("vision_verified") is True:
      verified += 1
    elif pm.get("vision_audit_skipped"):
      reason = str(pm["vision_audit_skipped"])
      skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
  return {
      "decisive": len(decisive),
      "verified": verified,
      "failed": failed,
      "never_ran": never_ran,
      "skipped_reasons": skipped_reasons,
      "channels": channels,
  }


def _vision_exclusivity_report(
    spec: ConflictVLMScenarioSpec,
    state: Any | None = None,
) -> list[str]:
  """Render the two-layer 'does this scenario actually require looking?' report.

  Layer 1 (assets) answers "were the pixels verified readable"; layer 2 (rounds)
  answers "was the value withheld from every text channel". Both are needed: a
  verified asset whose value is also sitting in a chat message is not a vision
  round, and a perfectly withheld value in an illegible chart is not a solvable
  one. Reporting only the headline count of decisive assets hid both failures.

  ``state`` is the compiled ``ScenarioSpec``; its ``rounds[].task_spec`` supplies
  the shipped text surfaces layer 2 scans.
  """
  audit = _summarize_vision_audit(spec)
  tasks_by_round: dict[str, dict[str, Any]] = {}
  for compiled in getattr(state, "rounds", []) or []:
    task_spec = getattr(compiled, "task_spec", None)
    if isinstance(task_spec, dict):
      tasks_by_round[str(getattr(compiled, "round_id", ""))] = task_spec

  verdicts: dict[str, int] = {}
  leak_surfaces: dict[str, int] = {}
  for round_spec in spec.rounds:
    verdict, leaked = _classify_round_vision_exclusivity(
        spec, round_spec, tasks_by_round.get(round_spec.round_id)
    )
    verdicts[verdict] = verdicts.get(verdict, 0) + 1
    for surface in leaked:
      leak_surfaces[surface] = leak_surfaces.get(surface, 0) + 1

  lines = ["Vision-exclusivity report:"]
  if audit["decisive"]:
    parts = [f"{audit['verified']} vision-verified"]
    if audit["failed"]:
      parts.append(f"{audit['failed']} illegible (downgraded)")
    if audit["never_ran"]:
      parts.append(f"{audit['never_ran']} never audited")
    unverified = (
        audit["decisive"] - audit["verified"] - audit["failed"] - audit["never_ran"]
    )
    if unverified > 0:
      parts.append(f"{unverified} audit skipped")
    lines.append(f"  assets: {audit['decisive']} decisive -- " + ", ".join(parts))
    if audit["channels"]:
      lines.append(
          "    channels: "
          + ", ".join(f"{k}={v}" for k, v in sorted(audit["channels"].items()))
      )
    for reason, count in sorted(audit["skipped_reasons"].items()):
      lines.append(f"    skipped ({count}x): {reason}")
  else:
    lines.append("  assets: no decisive assets tagged")

  asset_locus_rounds = sum(
      count for verdict, count in verdicts.items()
      if verdict not in (
          "no_resolved_value", "non_asset_locus",
          "non_asset_locus_text_solvable",
      )
  )
  lines.append(
      f"  rounds: {verdicts.get('vision_exclusive', 0)} vision-exclusive,"
      f" {verdicts.get('unverified_exclusive', 0)} exclusive-but-unverified,"
      f" {verdicts.get('text_solvable', 0)} text-solvable,"
      f" {verdicts.get('unsolvable', 0)} unsolvable"
      f" (of {asset_locus_rounds} asset-locus round(s);"
      f" {verdicts.get('non_asset_locus', 0)} non-asset-locus)"
  )
  # The headline number, reported against EVERY round that has a decisive value
  # rather than against the asset-locus subset. The per-locus line above answers
  # "of the rounds meant to require looking, how many do"; this one answers the
  # question that actually matters for the dataset -- "how many rounds can be
  # answered off a text channel" -- and is the line that would have surfaced the
  # 54 laundered rounds the old report filed under a clean ``non_asset_locus``.
  laundered = verdicts.get("non_asset_locus_text_solvable", 0)
  # ``not_compiled`` is excluded from the denominator as well as
  # ``no_resolved_value``: those rounds have a value but were never scanned, so
  # counting them as "not text-solvable" would dilute the very rate this line
  # exists to stop understating. They are reported on their own line below.
  total_valued = sum(
      count for verdict, count in verdicts.items()
      if verdict not in ("no_resolved_value", "not_compiled")
  )
  if total_valued:
    solvable = verdicts.get("text_solvable", 0) + laundered
    lines.append(
        f"    TEXT-SOLVABLE (all loci): {solvable}/{total_valued}"
        f" ({solvable / total_valued:.1%})"
        + (f" -- {laundered} of them on a non-asset locus" if laundered else "")
    )
  if leak_surfaces:
    lines.append(
        "    leaked via: "
        + ", ".join(f"{k}={v}" for k, v in sorted(leak_surfaces.items()))
    )
  if verdicts.get("not_compiled"):
    lines.append(
        f"    WARNING: {verdicts['not_compiled']} asset-locus round(s) had no"
        " compiled task_spec and could not be measured"
    )
  return lines


def _normalize_append_messages(app: dict[str, Any]) -> list[dict[str, Any]]:
  """Normalize a ``session_appends`` entry into a list of message dicts.

  Tolerates the two shapes the generator emits for an appended chat message:

    * canonical - ``{"session_id": .., "messages": [ {..}, .. ]}`` where each
      message is already a dict;
    * flat/singular - ``{"session_id": .., "sender": .., "timestamp": ..,
      "message": <str|dict>}``, the shape the LLM sometimes returns for a single
      appended message.

  A bare-string ``message`` is coerced into a message dict that HOISTS the
  app-level ``sender``/``timestamp``/``message_id`` (in the flat shape those live
  on the entry, not inside the string), so downstream ``msg.get(...)`` is always
  safe and no author/timestamp metadata is lost. Dict messages pass through
  unchanged; any other element type is skipped.
  """
  raw = app.get("messages")
  if not raw and "message" in app:
    raw = [app["message"]]
  normalized: list[dict[str, Any]] = []
  for m in raw or []:
    if isinstance(m, dict):
      normalized.append(m)
    elif isinstance(m, str):
      normalized.append({
          "message_id": app.get("message_id", ""),
          "timestamp": app.get("timestamp", ""),
          "sender": app.get("sender", ""),
          "text": m,
      })
  return normalized


def materialize_conflict_scenario(
    seed: ConflictVLMSeed,
    llm_client: ScenarioLLMClient,
    output_dir: Path | None = None,
    progress: Callable[[str], None] | None = None,
    image_provider: str = "mock",
    chart_provider: str = "mock",
    google_model: str = "gemini-pro-vision",
    google_api_key_env: str = "GOOGLE_API_KEY",
    google_base_url: str = "https://generativelanguage.googleapis.com/v1beta",
    raw_response_dir: Path | None = None,
    google_project: str | None = None,
    vertex_project: str | None = None,
    vertex_location: str | None = None,
    request_delay_seconds: float = 0.0,
    resume: bool = False,
    style_seed: int = 0,
    num_render_workers: int | None = None,
    num_generation_workers: int | None = None,
    auditor_max_iterations: int = 3,
    retries: int = 5,
    phase_expand_attempts: int | None = None,
    vision_audit: bool = True,
    checkpoint_dir: Path | None = None,
) -> tuple[ConflictVLMScenarioSpec, ScenarioSpec]:
  """Generate and package a full Conflict-VLM scenario from a seed.

  ``vision_audit`` (default on) gates the decisive-image fidelity audit; pass
  ``False`` (the CLI ``--novision_audit``) to skip it and save vision API calls.

  ``checkpoint_dir`` relocates the pipeline's resume checkpoint away from
  ``output_dir``. Callers that delete ``output_dir`` when a version fails MUST
  set it, or the failure cleanup destroys the very checkpoint a retry would
  resume from (see the note at the ``pipeline.generate`` call). Defaults to
  ``output_dir``, preserving the historical single-directory layout.
  """

  def _log(msg: str) -> None:
    if progress:
      progress(msg)
    else:
      logging.info(msg)

  _log(f"Starting materialization for seed: {seed.slug}")

  spec_path = output_dir / f"{seed.slug}_spec.json" if output_dir else None
  spec = None

  if resume and spec_path and spec_path.is_file():
    _log(f"Resuming generation. Loading existing spec from {spec_path}...")
    try:
      spec = ConflictVLMScenarioSpec.read_json(spec_path)
      # Validate the resumed spec actually corresponds to the requested seed so a
      # stale / mismatched cache is never reused silently.
      loaded_slug = getattr(spec, "slug", None)
      if loaded_slug and loaded_slug != seed.slug:
        msg = (
            f"Resumed spec slug '{loaded_slug}' != requested seed slug "
            f"'{seed.slug}'. Discarding cache and regenerating."
        )
        logging.warning(msg)
        _log(f"WARNING: {msg}")
        spec = None
      elif not getattr(spec, "rounds", None):
        msg = "Resumed spec contains no rounds. Discarding cache and regenerating."
        logging.warning(msg)
        _log(f"WARNING: {msg}")
        spec = None
    except Exception as e:
      logging.warning("Failed to load existing spec from %s: %s. Regenerating.", spec_path, e)
      _log(f"WARNING: failed to load existing spec ({e}). Starting fresh generation...")

  if spec is None:
    spec = _generate_fresh_spec(
        seed=seed,
        llm_client=llm_client,
        progress=progress,
        image_provider=image_provider,
        chart_provider=chart_provider,
        output_dir=output_dir,
        resume=resume,
        num_generation_workers=num_generation_workers,
        phase_expand_attempts=phase_expand_attempts,
        style_seed=style_seed,
        checkpoint_dir=checkpoint_dir,
    )
    if output_dir and spec_path:
      output_dir.mkdir(parents=True, exist_ok=True)
      spec.write_json(spec_path)
      _log(f"Saved initial scenario spec checkpoint to {spec_path}")

  # Repair mismatched update IDs in rounds of ConflictVLMScenarioSpec
  updates_in_spec = {u.update_id for u in spec.updates}
  for round_spec in spec.rounds:
    round_id = round_spec.round_id
    new_update_ids = []
    for u_id in round_spec.update_ids:
      if u_id in updates_in_spec:
        new_update_ids.append(u_id)
      else:
        matched_id = None
        for spec_u_id in updates_in_spec:
          if round_id in spec_u_id:
            matched_id = spec_u_id
            break
        if matched_id:
          new_update_ids.append(matched_id)
        else:
          new_update_ids.append(u_id)
    round_spec.update_ids = new_update_ids

  # M8 CL1 coordination-integrity gate. Seed-gated / behaviour-gated: runs only
  # for a seed opting in via ``require_protocol_completion`` (a no-op for every
  # other seed, E6), failing fast if the graded gold trajectory never completes
  # the seed's core protocol mechanic ("coordination theatre"). Deliberately the
  # narrow ``enforce_protocol_completion`` rather than the full strict validator,
  # so generation does not newly enforce the other strict checks (which remain
  # post-generation test backstops).
  enforce_protocol_completion(spec, seed)

  # M8 CL2 forced-async contract gate. Seed-gated / behaviour-gated: runs only
  # for a seed opting in via ``force_async_wait`` (a no-op for every other seed,
  # E6), failing fast if the scenario cannot actually grade its forced-async
  # mechanic -- no delayed sub-agent completion (so no within_window check is
  # injected) or a gold trajectory that never delegates-then-waits ("async
  # theatre"). Narrow companion to ``enforce_protocol_completion``.
  enforce_async_wait_contract(spec, seed)

  if output_dir:
    output_dir.mkdir(parents=True, exist_ok=True)

    for rel in ("assets", "workspace", "sessions", "updates"):
      (output_dir / rel).mkdir(parents=True, exist_ok=True)

    # Tag decisive assets (role is orthogonal to kind, and this spans charts and
    # HTML-rendered assets as well as photos) so the prompt augmentation and the
    # vision fidelity audit can enforce that the deciding token/state is legibly
    # visible in the rendered pixels.
    num_decisive_assets = _tag_decisive_assets(spec)
    if num_decisive_assets:
      _log(
          f"Tagged {num_decisive_assets} decisive asset(s) for fidelity audit"
      )

    # 5. Generate Multi-modal assets (render pixels/charts to disk)
    _log("Generating visual assets...")
    spec = generate_vlm_assets(
        spec,
        scenario_dir=output_dir,
        image_provider=image_provider,
        chart_provider=chart_provider,
        google_model=google_model,
        google_api_key_env=google_api_key_env,
        google_base_url=google_base_url,
        raw_response_dir=raw_response_dir,
        google_project=google_project,
        vertex_project=vertex_project,
        vertex_location=vertex_location,
        request_delay_seconds=request_delay_seconds,
        llm_client=llm_client,
        progress=progress,
        style_seed=style_seed,
        num_render_workers=num_render_workers,
        num_generation_workers=num_generation_workers,
        retries=retries,
        vision_audit=vision_audit,
    )

    # Write asset_manifest.json
    manifest = VLMAssetManifest(spec.assets)
    _write_json(output_dir / "asset_manifest.json", manifest.to_package_dict())

    # Write workspace files
    _log("Writing workspace files...")
    for f in spec.workspace.files:
      rel_path = f.get("path")
      content = f.get("content")
      if rel_path and content:
        rel_path = rel_path.lstrip("/")
        _write_text(output_dir / "workspace" / rel_path, content)

    # Write sessions
    _log("Writing chat sessions...")
    for session in spec.sessions:
      if not session.path:
        session.path = f"sessions/{session.session_id}.jsonl"
      lines = []
      for msg in session.messages:
        lines.append(
            json.dumps(
                {
                    "message_id": msg.message_id,
                    "timestamp": msg.timestamp,
                    "sender": msg.sender,
                    "text": msg.text,
                    "attachments": msg.attachments,
                    "is_redacted": msg.is_redacted,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
      _write_text(output_dir / session.path, "".join(lines))

    # Write updates files
    _log("Writing scenario updates...")
    for u in spec.updates:
      for add in u.workspace_adds:
        rel_path = add.get("path")
        content = add.get("content")
        if rel_path and content:
          rel_path = rel_path.lstrip("/")
          mat_path = f"updates/{u.update_id}/workspace/{rel_path}"
          add["materialized_path"] = mat_path
          _write_text(output_dir / mat_path, content)
      for app in u.session_appends:
        session_id = app.get("session_id")
        messages_list = _normalize_append_messages(app)
        if session_id and messages_list:
          mat_path = f"updates/{u.update_id}/sessions/{session_id}.jsonl"
          app["materialized_path"] = mat_path
          lines = []
          for msg in messages_list:
            lines.append(
                json.dumps(
                    {
                        "message_id": msg.get("message_id", ""),
                        "timestamp": msg.get("timestamp", ""),
                        "sender": msg.get("sender", ""),
                        "text": msg.get("text") or msg.get("content") or "",
                        "attachments": msg.get("attachments", []),
                        "is_redacted": msg.get("is_redacted", False),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
          _write_text(output_dir / mat_path, "".join(lines))

    spec_path = output_dir / f"{seed.slug}_spec.json"
    spec.write_json(spec_path)
    _log(f"Saved scenario spec to {spec_path}")

  # 6. Compile down to standard OpenClaw ScenarioSpec for evaluator
  tasks: list[Task] = []
  scenario_rounds: list[ScenarioRound] = []
  eval_items: list[ScenarioEvalItem] = []

  for round_spec in spec.rounds:
    round_updates = [
        u for u in spec.updates if u.update_id in round_spec.update_ids
    ]
    round_mocks = round_spec.mock_subagents
    task = _compile_round_to_task(
        seed, round_spec, spec.evals, round_updates, round_mocks, spec
    )
    tasks.append(task)
    if output_dir:
      save_task(task, data_dir=output_dir)

    scenario_rounds.append(
        ScenarioRound(
            round_id=round_spec.round_id,
            task_id=task.task_id,
            instruction=task.instruction,
            task_spec=task.to_dict(),
            update_ids=list(round_spec.update_ids),
            session_refs=[],
            workspace_refs=[],
            visible_refs=list(round_spec.visible_asset_ids),
            expected_anchor_ids=list(round_spec.expected_anchor_ids),
            expert_trajectory=list(
                task.ground_truth.solution_commands if task.ground_truth else []
            ),
            metadata={
                "round_type": round_spec.round_type,
                # ③: surface the round's graded conclusion on the shipped
                # ScenarioRound for inspection/coverage of the decision.
                "decision_outcome": getattr(round_spec, "decision_outcome", ""),
                "resolved_value": getattr(round_spec, "resolved_value", ""),
            },
        )
    )
    # Find eval items for this round
    round_evals = [e for e in spec.evals if e.round_id == round_spec.round_id]
    if round_evals:
      eval_items.append(
          ScenarioEvalItem(
              eval_id=f"eval_{round_spec.round_id}",
              round_id=round_spec.round_id,
              kind="task_checks",
              checks=round_evals[0].checks,
              metadata={},
          )
      )

  openclaw_spec = ScenarioSpec(
      scenario_id=f"conflict_vlm_v1:{seed.slug}",
      generator_id="conflict_vlm_v1",
      scenario_slug=seed.slug,
      title=seed.title,
      description=seed.core_conflict,
      version="1.0",
      narrative={
          "domain": seed.domain,
          "core_conflict": seed.core_conflict,
          "allowed_tool_families": seed.allowed_tool_families,
          "conflict_vlm_spec": f"{seed.slug}_spec.json",
      },
      workspace=spec.workspace,
      sessions=spec.sessions,
      updates=spec.updates,
      rounds=scenario_rounds,
      eval_items=eval_items,
  )

  # Auditor + consistency are best-effort repair layers, NOT gates. They run
  # before scenario.json is written, so an uncaught failure here would abort
  # materialization and leave a half-written version (the root cause of broken
  # scenario dirs with no scenario.json). Guard each so a failure degrades to
  # "ship the un-audited/un-repaired spec" and still flows into C2/C4/F1 and the
  # scenario.json write. F1 (_verify_and_repair) remains the non-discarding
  # verification backstop, so nothing is ever dropped on account of these steps.
  from clasharness.scenario.llm_auditor import audit_scenario_with_llm
  try:
    openclaw_spec = audit_scenario_with_llm(openclaw_spec, llm_client, max_iterations=auditor_max_iterations)
  except Exception as e:  # pylint: disable=broad-except
    _log(f"WARNING: LLM auditor failed; proceeding with un-audited spec (F1 backstop still applies). Error: {e}")

  from clasharness.scenario.consistency_layer import audit_and_repair_scenario
  try:
    openclaw_spec = audit_and_repair_scenario(openclaw_spec)
  except Exception as e:  # pylint: disable=broad-except
    _log(f"WARNING: consistency layer failed; proceeding with un-repaired spec (F1 backstop still applies). Error: {e}")

  if output_dir:
    # Re-save all compiled tasks to disk to ensure they have all LLM auditor
    # repairs, so the calibration + F1 gates below replay exactly what would ship.
    for round_spec in openclaw_spec.rounds:
      task = Task.from_dict(round_spec.task_spec)
      save_task(task, data_dir=output_dir)

    # C2 (State Resolver calibration, verification LEFT-SHIFT): before the F1
    # backstop, replay each gold trajectory once and deterministically reconcile
    # any file_check the gold cannot satisfy against the resolved final state
    # (degrade doomed content assertions to existence, or drop them). This makes
    # the shipped gold pass its own file_checks by construction, so F1 rarely has
    # anything left to repair.
    _log("Running State Resolver calibration (file_check ↔ final state)...")
    n_adjusted = _calibrate_checks_against_state(openclaw_spec, output_dir, _log)
    _log(f"State Resolver: adjusted {n_adjusted} file_check(s) to the resolved state.")

    # C3 (Zero-Action Discrimination Resolver): immediately AFTER C2, which is
    # itself a source of the defect - degrading a doomed ``contains`` to a bare
    # ``exists`` on an ``initial_state_overrides``-seeded file yields a check that
    # is true before the agent acts. Probe every round with an EMPTY trajectory and
    # demote each check that already passes to ``diagnostic``, so it stops gating
    # success and stops inflating the score. Free and deterministic: the judge env
    # is cleared for the probe, and an ungradable check is left alone.
    _log("Running Zero-Action Discrimination Resolver (vacuous check demotion)...")
    n_vacuous, n_guarded = _demote_vacuous_checks(openclaw_spec, output_dir, _log)
    _log(
        f"Zero-Action Resolver: demoted {n_vacuous} vacuous check(s) to diagnostic."
        + (
            f" WARNING: {n_guarded} round(s) had EVERY required check pass at zero "
            "actions and were left unchanged to avoid an empty required set."
            if n_guarded else ""
        )
    )

    # C4 (Protocol Prerequisite Resolver): before the LLM backstop, deterministically
    # repair the leading cause of kept-with-failures rounds — a gold trajectory that
    # fails its own protocol ``exists`` effect check because the required multi-step
    # prerequisites (hold→confirm, request→execute, …) are missing/misordered. Reads
    # the machine-readable precondition contract (``CommandSpec.requires``) and inserts
    # the missing prerequisite commands, but ONLY for rounds already failing an effect
    # check and ONLY when it strictly reduces failures without regressing any other
    # check — so passing/diverse trajectories are left byte-identical.
    _log("Running Protocol Prerequisite Resolver (effect precondition repair)...")
    n_proto = _resolve_protocol_prerequisites(openclaw_spec, output_dir, _log)
    _log(f"Protocol Prerequisite Resolver: repaired {n_proto} round(s) deterministically.")

    # 7. F1: verification-driven LLM self-repair backstop. NEVER discards a round.
    # Role after C3: with C2 reconciling file_checks up front, F1 is now a thin
    # safety net - it re-replays each gold solution and, for any residual failure
    # (chiefly effect/message-semantics checks that C2 does not own), invokes the
    # LLM 'repair_trajectory_simulation_failure' layer up to --auditor_max_iterations
    # times. If repair does not fully succeed within the budget we KEEP THE LAST
    # VERSION rather than dropping the round (product decision: preserve dataset
    # size; best-effort well-posedness).
    _log("Running F1 verification backstop + LLM self-repair (no discard)...")
    repair_report = _verify_and_repair(
        openclaw_spec, output_dir, _log, llm_client,
        max_iterations=auditor_max_iterations,
    )

    if repair_report:
      failed_tasks_path = output_dir / "failed_tasks.json"
      _log(
          f"F1: {len(repair_report)} round(s) kept with unresolved issues; "
          f"details -> {failed_tasks_path}"
      )
      with open(failed_tasks_path, "w", encoding="utf-8") as f:
        json.dump(repair_report, f, indent=2)
    else:
      _log("✅ F1: every compiled round's gold solution passes its own checks.")

    # Report (never a gate) how much of this scenario genuinely requires looking
    # at an asset, measured on the FINAL post-repair state rather than on the
    # generation-time gate's own verdict. Wrapped because a reporting bug must
    # never abort a scenario that has otherwise fully materialized.
    try:
      for line in _vision_exclusivity_report(spec, openclaw_spec):
        _log(line)
    except Exception as e:  # pylint: disable=broad-except
      _log(f"WARNING: vision-exclusivity report failed (scenario unaffected): {e}")

    # Write scenario.json (ALL rounds kept; nothing discarded). Atomic: write to
    # a temp file then rename, so scenario.json only ever exists in its complete
    # form. A crash mid-write leaves scenario.json.tmp (ignored) rather than a
    # truncated scenario.json that downstream would mistake for a valid
    # completion marker.
    scenario_path = output_dir / "scenario.json"
    scenario_tmp = output_dir / "scenario.json.tmp"
    openclaw_spec.write_json(scenario_tmp)
    scenario_tmp.replace(scenario_path)
    _log(f"Saved standard scenario to {scenario_path}")
    _generate_html_visualizer(spec, output_dir, progress=progress)

  return spec, openclaw_spec


def _is_semantic_check(check: Any) -> bool:
  """True for any check graded by the LLM judge (``field_semantic_match``).

  Such a check calls ``judge_majority`` (a live model) at run time -- whether it
  targets a written file (``file_check``/``file``) or an effect stream (a
  ``files_created`` report deliverable or a ``messages_sent`` delegation check).
  It must never be evaluated during compile-time replay: it is non-deterministic,
  costs a model call, and is validated only at real run-time grading. Matching on
  the ``field_semantic_match`` condition alone covers every judge-graded shape.
  """
  return (
      isinstance(check, dict)
      and check.get("condition") == "field_semantic_match"
  )


class _ReplaySafeStub(Evaluator):
  """Compile-time placeholder for a run-time-only (LLM-judge) check.

  Substituted for a ``field_semantic_match`` file checker during the throwaway
  replay so the compile-time state solver neither invokes the judge nor lets the
  C2/F1 passes act on an assertion that is only graded at run time. Always passes
  deterministically and preserves the check's name/weight so ``details[i]`` still
  aligns with ``evaluation_checks[i]``.
  """

  def __init__(self, name: str, weight: float) -> None:
    self.name = name
    self.weight = weight

  def evaluate(
      self, env_state: dict[str, Any], task_data: dict[str, Any]
  ) -> CheckResult:
    return CheckResult(
        name=self.name,
        passed=True,
        score=1.0,
        weight=self.weight,
        message="semantic check deferred to run-time grading",
    )


def _replay_trajectory(
    task_id: str, output_dir: Path, replay: list[str] | None
) -> tuple[Any, str, dict[str, Any]]:
  """Replay a trajectory in a throwaway env; return (result, trace, env_state).

  Shared compile-time state-solver used by the State Resolver calibration pass
  (C2) and the F1 verify+repair backstop (F1). ``build_evaluator`` /
  ``EvaluatorComb`` preserve check order, so ``result.details[i]`` aligns with
  ``evaluation_checks[i]``. ``env_state`` is the same dict the evaluator graded
  against (so file existence can be re-derived deterministically).

  Semantic (``field_semantic_match``) file checks are LLM-judge graded at run
  time; here they are swapped for an always-pass stub so replay stays fully
  deterministic and never issues a model call during generation.
  """
  from clasharness.core.environment import CLASHarnessEnv

  trace: list[str] = []
  with CLASHarnessEnv(
      task_id=task_id, task_data_dir=output_dir, backend="multi"
  ) as env:
    env.reset()
    for cmd in replay or []:
      if not isinstance(cmd, str):
        continue
      if cmd.strip().lower() in {"done", "exit", "quit"}:
        break
      try:
        obs, _r, _d, _i = env.step(cmd)
        trace.append(f"$ {cmd}\n{str(obs)[:400]}")
      except Exception as e:  # keep going; the eval reflects the failure
        trace.append(f"$ {cmd}\n[step error: {e}]")
    _stub_semantic_file_checks(env)
    result = env.evaluate()
    try:
      env_state = env._get_evaluation_state()
    except Exception:
      env_state = {}
    return result, "\n".join(trace), env_state


def _stub_semantic_file_checks(env: Any) -> None:
  """Replace semantic file checkers in ``env``'s evaluator with pass stubs.

  Keeps positional alignment (``EvaluatorComb`` preserves check order, matching
  ``ground_truth.evaluation_checks``) so downstream ``details[i]`` indexing in the
  C2/F1 passes is unaffected. The env is a throwaway replay instance, so mutating
  its evaluator in place is safe.
  """
  evaluator = getattr(env, "_evaluator", None)
  task = getattr(env, "_task", None)
  if evaluator is None or task is None or not getattr(task, "ground_truth", None):
    return
  checks = list(task.ground_truth.evaluation_checks or [])
  for idx, chk in enumerate(checks):
    if idx >= len(evaluator.evaluators):
      break
    if _is_semantic_check(chk):
      inst = evaluator.evaluators[idx]
      evaluator.evaluators[idx] = _ReplaySafeStub(
          name=getattr(inst, "name", "") or "",
          weight=getattr(inst, "weight", 1.0),
      )


# Punctuation dropped when re-matching an authored content substring against the
# actually-written file body. Quote/backtick/asterisk drift between the two
# independent LLM calls (one writes the file, one authors the ``contains``
# substring) is the sole systemic cause of ``files_created``/``field_contains``
# misses; run-time grading (effect_checker) is a raw lowercase substring match, so
# only these format characters ever diverge for an otherwise-present value.
_CONTENT_MATCH_DROP_CHARS = ('"', "'", "`", "*")


def _is_files_created_contains(check: Any) -> bool:
  """True for a ``{"type":"effect","effect_type":"files_created",
  "condition":"field_contains"}`` evaluation check."""
  return (
      isinstance(check, dict)
      and check.get("type") == "effect"
      and check.get("effect_type") == "files_created"
      and check.get("condition") == "field_contains"
  )


# Character class of "insignificant" separators tolerated between the significant
# characters of an authored substring: whitespace plus the dropped quote/markdown
# punctuation. Kept in a raw class string so it can be embedded in a regex.
_CONTENT_SEP_CLASS = r"[\s\"'`*]*"
_CONTENT_DROP_CLASS = r"[\"'`*]*"


def _rederive_contains_value(needle: str, content: str) -> str | None:
  """Return the raw slice of ``content`` matching ``needle`` modulo quote/
  whitespace/markdown drift, or ``None`` if the value is genuinely absent.

  The substring is authored by a separate LLM call from the one that writes the
  file, so an otherwise-present value can differ only in "insignificant"
  separators: quote/backtick/asterisk punctuation (which may appear *anywhere* in
  the real content, e.g. JSON quotes) and whitespace (tolerated where the needle
  itself has a token boundary). We build a regex from the needle's *significant*
  characters, allowing those separators between them, and match it against the raw
  content. Crucially, only separator characters may be skipped -- never letters,
  digits, or other punctuation -- so a coincidental cross-token match cannot
  weaken grading. The returned span is a literal substring of ``content``
  (expanded over adjacent dropped punctuation for readability), so
  ``value.lower() in content.lower()`` -- exactly what ``EffectChecker`` evaluates
  at run time -- is guaranteed True.
  """
  parts: list[str] = []
  pending_ws = False
  seen_sig = False
  for ch in needle:
    if ch in _CONTENT_MATCH_DROP_CHARS:
      continue
    if ch.isspace():
      pending_ws = True
      continue
    if seen_sig:
      parts.append(_CONTENT_SEP_CLASS if pending_ws else _CONTENT_DROP_CLASS)
    parts.append(re.escape(ch))
    seen_sig = True
    pending_ws = False
  if not parts:
    return None
  match = re.search("".join(parts), content, re.IGNORECASE)
  if match is None:
    return None
  raw_start, raw_end = match.start(), match.end()
  # Expand over immediately-adjacent dropped punctuation (e.g. the quotes around a
  # JSON value) so the re-derived substring reads naturally; still a raw slice.
  while raw_start > 0 and content[raw_start - 1] in _CONTENT_MATCH_DROP_CHARS:
    raw_start -= 1
  while raw_end < len(content) and content[raw_end] in _CONTENT_MATCH_DROP_CHARS:
    raw_end += 1
  candidate = content[raw_start:raw_end]
  if str(candidate).lower() in str(content).lower():
    return candidate
  return None


def _calibrate_files_created_contains(
    chk: dict[str, Any],
    env_state: dict[str, Any],
    log: Callable[[str], None],
    task_id: str,
) -> dict[str, Any] | None:
  """Reconcile a failing ``files_created``/``field_contains`` effect check against
  the actually-written content in the resolved final state.

  1. Re-derive the expected substring from the real content when it differs only
     by quote/whitespace/markdown drift (the systemic case): the grading intent
     (the decisive value is present) is preserved, just format-aligned.
  2. If the value is genuinely absent but some file was created, degrade to a bare
     ``exists`` check (mirroring the file_check degrade-to-existence policy).

  Returns the adjusted check, or ``None`` to leave the original untouched.
  """
  raw_expected = chk.get("expected")
  expected: dict[str, Any] = raw_expected if isinstance(raw_expected, dict) else {}
  field = expected.get("field") or "content"
  needle = str(expected.get("value", "")).strip()
  if not needle:
    return None
  effects = env_state.get("effects", {}) if isinstance(env_state, dict) else {}
  effect_list = effects.get("files_created", []) if isinstance(effects, dict) else []
  contents = [
      str(e.get(field, "")) for e in effect_list if isinstance(e, dict)
  ]
  # Defensive: if the value already satisfies run-time grading (raw lowercase
  # substring), there is nothing to reconcile. In the normal C2 flow this helper
  # is only reached for a *failing* check, so this is a no-op guard that keeps the
  # helper correct if ever called on a passing check.
  raw_value = str(expected.get("value", ""))
  if any(raw_value.lower() in content.lower() for content in contents):
    return None
  for content in contents:
    rederived = _rederive_contains_value(needle, content)
    if rederived is not None and rederived != expected.get("value"):
      new_chk = dict(chk)
      new_expected = dict(expected)
      new_expected["value"] = rederived
      new_chk["expected"] = new_expected
      log(
          f"    ↳ {task_id}: re-derived files_created contains "
          f"{needle!r} -> {rederived!r}"
      )
      return new_chk
  if effect_list:
    new_chk = dict(chk)
    new_chk["condition"] = "exists"
    log(
        f"    ↳ {task_id}: degraded unsatisfiable files_created contains "
        f"{needle!r} to existence"
    )
    return new_chk
  return None


def _calibrate_checks_against_state(
    openclaw_spec: ScenarioSpec,
    output_dir: Path,
    log: Callable[[str], None],
) -> int:
  """State Resolver calibration (C2): reconcile file_checks to the resolved state.

  Verification left-shift. Before the F1 backstop, replay each round's gold
  (expert) trajectory once and deterministically reconcile every ``file_check``
  the gold cannot satisfy against the *resolved final state*:

    * a failing content assertion (``contains``/``equals``) is degraded to a bare
      existence check when the file exists in the final state - the decisive value
      is already graded through the round's outcome (message/report) checkers, so
      the read/write file need not re-assert it (A/C-class residue, and any
      B-class the effect-drop missed);
    * if the file does not exist in the final state, the check is dropped.

  This makes the shipped gold pass its own file_checks *by construction*, which is
  what turns "prevented-by-construction" from aspiration into fact: F1 then finds
  nothing to repair, so the wasted 5-attempt repair cycles disappear. Edits the
  shipped ``task_spec`` (what F1 replays and what grades at run time) and re-saves
  the task. Returns the number of checks adjusted.
  """
  from clasharness.evaluation.checkers.file_checker import FileChecker

  adjusted = 0
  for round_spec in openclaw_spec.rounds:
    task_id = round_spec.task_id
    task_spec = round_spec.task_spec
    if not isinstance(task_spec, dict):
      continue
    gt = task_spec.get("ground_truth")
    if not isinstance(gt, dict):
      continue
    checks = gt.get("evaluation_checks")
    if not isinstance(checks, list) or not checks:
      continue
    sol_cmds = list(round_spec.expert_trajectory or [])
    if not sol_cmds:
      sol_cmds = list(gt.get("solution_commands") or [])
    if not sol_cmds:
      continue
    try:
      res, _trace, env_state = _replay_trajectory(task_id, output_dir, sol_cmds)
    except Exception as e:
      log(f"  • {task_id}: calibration skipped (replay error: {e})")
      continue
    details = list(getattr(res, "details", None) or [])

    new_checks: list[Any] = []
    round_adjusted = 0
    for i, chk in enumerate(checks):
      passed = details[i].passed if i < len(details) else True
      if passed:
        new_checks.append(chk)
        continue
      is_filecheck = (
          isinstance(chk, dict) and chk.get("type") in {"file_check", "file"}
      )
      if is_filecheck:
        # Only reconcile content assertions; a failing existence/not_exists check
        # is a structural issue left to the F1 backstop.
        cond = chk.get("condition")
        has_content = (
            cond in {"contains", "equals"}
            or bool(str(chk.get("contains", "")).strip())
            or bool(str(chk.get("equals", "")).strip())
        )
        if not has_content:
          new_checks.append(chk)
          continue
        # Degrade to existence (keeping name/weight identity); keep only if the
        # file actually exists in the resolved final state - verified via the real
        # FileChecker so the prediction matches run-time grading exactly.
        degraded: dict[str, Any] = {"type": "file_check", "path": chk.get("path", "")}
        if chk.get("name"):
          degraded["name"] = chk["name"]
        if chk.get("weight") is not None:
          degraded["weight"] = chk["weight"]
        try:
          exists_ok = FileChecker.from_spec(degraded).evaluate(env_state, {}).passed
        except Exception:
          exists_ok = False
        if exists_ok:
          new_checks.append(degraded)
          log(
              f"    ↳ {task_id}: degraded doomed file_check on "
              f"{chk.get('path')!r} to existence"
          )
        else:
          log(
              f"    ↳ {task_id}: dropped unsatisfiable file_check on "
              f"{chk.get('path')!r}"
          )
        round_adjusted += 1
        continue
      # Effect content calibration (C2 extension). A failing
      # ``files_created``/``field_contains`` effect check almost always means the
      # gold wrote the decisive value in a form that differs from the authored
      # substring only by quote/whitespace/markdown drift - the substring is
      # authored by a *separate* LLM call from the one that writes the file, with
      # no consistency guarantee, and run-time grading is a raw substring match.
      # Re-derive the expected substring from the actually-written content so gold
      # passes by construction; degrade to existence only if genuinely absent.
      if _is_files_created_contains(chk):
        calibrated = _calibrate_files_created_contains(chk, env_state, log, task_id)
        if calibrated is not None:
          new_checks.append(calibrated)
          round_adjusted += 1
          continue
      new_checks.append(chk)

    if round_adjusted:
      gt["evaluation_checks"] = new_checks
      round_spec.task_spec = task_spec
      try:
        task = Task.from_dict(task_spec)
        save_task(task, data_dir=output_dir)
      except Exception as e:
        log(f"  • {task_id}: calibration re-save failed: {e}")
      adjusted += round_adjusted

  return adjusted


# Judge-provider environment variables cleared for the duration of the zero-action
# probe. A ``field_semantic_match`` check would otherwise bill k model calls per
# task just to grade an empty transcript. With them cleared the grader fails
# closed, which the probe records as "verdict unknown" (``ok=False``) and therefore
# never acts on -- the conservative direction.
_ZERO_ACTION_JUDGE_ENV = (
    "LLM_JUDGE_PROVIDER",
    "LLM_JUDGE_MODEL",
    "GEMINI_API_KEY",
    "ANTHROPIC_API_KEY",
    "VERTEX_PROJECT",
)

# Roles that actually decide something: ``required`` gates success AND contributes
# to the score, ``scored`` contributes to the score only. ``diagnostic`` affects
# neither, so a vacuous diagnostic check is already harmless and is left alone.
_DISCRIMINATING_ROLES = frozenset({"required", "scored"})


def _is_negative_assertion(check: Any) -> bool:
  """True for a check asserting the ABSENCE of something (``not_exists``, ...).

  A negative assertion is *supposed* to hold with zero agent actions -- it says
  "do not break/delete this". Its discriminating power is whether an *acting*
  agent violates it, so passing the zero-action probe is the expected, correct
  behaviour and must never be read as vacuity.
  """
  if not isinstance(check, dict):
    return False
  return str(check.get("condition") or "").startswith("not_")


def _replay_zero_action(task_id: str, output_dir: Path) -> Any:
  """Evaluate a task after ``reset()`` with ZERO agent actions.

  Deliberately NOT ``_replay_trajectory(task_id, output_dir, [])``: that helper
  installs ``_stub_semantic_file_checks``, an always-PASS stub. Under a zero-action
  probe an always-pass stub reports every ``field_semantic_match`` check as
  "already satisfied before the agent did anything" -- which is precisely the
  vacuity signal this pass acts on, so reusing it would neutralise the entire
  semantic check population. The probe therefore builds its own env and leaves the
  real graders in place; the cleared judge env keeps it free and deterministic.
  """
  from clasharness.core.environment import CLASHarnessEnv

  with CLASHarnessEnv(
      task_id=task_id, task_data_dir=output_dir, backend="multi"
  ) as env:
    env.reset()
    return env.evaluate()


def _demote_vacuous_checks(
    openclaw_spec: ScenarioSpec,
    output_dir: Path,
    log: Callable[[str], None],
) -> tuple[int, int]:
  """C3 (Zero-Action Discrimination Resolver): neutralise checks already satisfied
  before the agent acts.

  A check that passes with zero agent actions measures nothing: every model scores
  it identically, so it inflates the score and -- when ``required`` -- weakens the
  success gate into a free pass. Such a check is demoted to ``role: diagnostic``
  (retained for observability, but no longer gating success nor contributing to
  the score) rather than deleted.

  Ordered AFTER C2 because C2 is itself a source of the defect: degrading a doomed
  ``contains`` to a bare ``exists`` on a file that ``initial_state_overrides``
  pre-seeds yields a check that is true at t=0 by construction.

  Conservative by construction -- a check is demoted only when ALL hold:
    * its role currently decides something (``required``/``scored``);
    * it is not a negative assertion (those SHOULD hold at t=0);
    * the probe reached a genuine verdict (``ok``) -- an ungradable check has an
      *unknown* zero-action verdict, not a clean one;
    * that verdict is ``passed``.

  Invariant: a round never has its entire ``required`` set demoted. ``success`` is
  ``all(required_passed)``, and ``all([])`` is True, so emptying the required set
  turns the round into a guaranteed win -- strictly worse than the vacuity being
  repaired. When the demotions would empty it, NONE are applied to that round and
  the case is logged loudly, leaving the generate-side zero-action audit gate to
  fail the run.

  Returns ``(n_checks_demoted, n_rounds_guarded)``.
  """
  import os as _os

  demoted_total = 0
  guarded_total = 0
  saved = {k: _os.environ.pop(k, None) for k in _ZERO_ACTION_JUDGE_ENV}
  try:
    for round_spec in openclaw_spec.rounds:
      task_id = round_spec.task_id
      task_spec = round_spec.task_spec
      if not isinstance(task_spec, dict):
        continue
      gt = task_spec.get("ground_truth")
      if not isinstance(gt, dict):
        continue
      checks = gt.get("evaluation_checks")
      if not isinstance(checks, list) or not checks:
        continue
      try:
        res = _replay_zero_action(task_id, output_dir)
      except Exception as e:  # pylint: disable=broad-except
        log(f"  • {task_id}: zero-action probe skipped (replay error: {e})")
        continue
      details = list(getattr(res, "details", None) or [])

      victims: list[int] = []
      for i, chk in enumerate(checks):
        if not isinstance(chk, dict):
          continue
        if chk.get("role", "required") not in _DISCRIMINATING_ROLES:
          continue
        if _is_negative_assertion(chk):
          continue
        if i >= len(details):
          continue
        detail = details[i]
        if not getattr(detail, "ok", True):
          continue
        if not getattr(detail, "passed", False):
          continue
        victims.append(i)

      if not victims:
        continue

      required_before = sum(
          1 for c in checks
          if isinstance(c, dict) and c.get("role", "required") == "required"
      )
      required_demoted = sum(
          1 for i in victims
          if checks[i].get("role", "required") == "required"
      )
      if required_before and required_demoted >= required_before:
        guarded_total += 1
        log(
            f"  ! {task_id}: ALL {required_before} required check(s) pass at zero "
            "actions; refusing to demote (an empty required set would make the "
            "round a guaranteed win). Round shipped unchanged - the generate-side "
            "zero-action audit gate will flag it."
        )
        continue

      victim_set = set(victims)
      new_checks: list[Any] = []
      for i, chk in enumerate(checks):
        if i not in victim_set:
          new_checks.append(chk)
          continue
        demoted = dict(chk)
        demoted["role"] = "diagnostic"
        new_checks.append(demoted)
        # Identify the demotion precisely: these checks frequently carry no
        # ``name``, and "demoted 'file_check'" is unauditable.
        label = (
            chk.get("name")
            or chk.get("path")
            or chk.get("effect_type")
            or chk.get("condition")
            or chk.get("type")
        )
        log(
            f"    ↳ {task_id}: demoted vacuous {chk.get('role', 'required')} check "
            f"{label!r} to diagnostic (passes with zero agent actions)"
        )

      gt["evaluation_checks"] = new_checks
      round_spec.task_spec = task_spec
      try:
        task = Task.from_dict(task_spec)
        save_task(task, data_dir=output_dir)
      except Exception as e:  # pylint: disable=broad-except
        log(f"  • {task_id}: zero-action re-save failed: {e}")
      demoted_total += len(victims)
  finally:
    for k, v in saved.items():
      if v is not None:
        _os.environ[k] = v

  return demoted_total, guarded_total


# TTL stamped on injected prerequisite holds/sessions. Large enough that no
# virtual-clock advance (background waits are trigger-time bounded, and the
# terminal command runs right after the injected prefix) can expire them.
_INJECTED_PREREQ_TTL = 10**9


def _is_effect_check(check: Any) -> bool:
  """True for a protocol ``{"type": "effect", ...}`` evaluation check."""
  return isinstance(check, dict) and check.get("type") == "effect"


def _failing_indices(checks: list[Any], details: list[Any]) -> set[int]:
  """Indices of checks the replay marked failed (order-aligned with details)."""
  return {
      i
      for i in range(len(checks))
      if i < len(details) and not details[i].passed
  }


def _failing_effect_types(
    checks: list[Any], failing_idx: set[int]
) -> list[str]:
  """``effect_type`` of each failing ``type=="effect"`` check."""
  out: list[str] = []
  for i in sorted(failing_idx):
    chk = checks[i]
    if _is_effect_check(chk):
      et = str(chk.get("effect_type", "") or "")
      if et:
        out.append(et)
  return out


def _first_command_index(cmds: list[str], name: str) -> int | None:
  """Index of the first command line whose leading tokens are ``name`` (spaced,
  e.g. ``"booking confirm"``), else ``None``."""
  target = name.split()
  n = len(target)
  for i, c in enumerate(cmds):
    if not isinstance(c, str):
      continue
    toks = c.strip().split()
    if toks[:n] == target:
      return i
  return None


def _gate_names_before(cmds: list[str], idx: int, gate_set_name: str,
                       gate_flag: str) -> list[str]:
  """Gate identifiers set by ``gate_set_name`` commands before ``idx``."""
  names: list[str] = []
  prefix = gate_set_name.split()
  n = len(prefix)
  for c in cmds[:idx]:
    if not isinstance(c, str):
      continue
    if c.strip().split()[:n] != prefix:
      continue
    v = _extract_flag_value(c, gate_flag)
    if v and v not in names:
      names.append(v)
  return names


def _copy_shared_provided_args(
    terminal_cmd: str, terminal_spec: Any, chain_spec: Any
) -> list[str]:
  """``[flag, value, ...]`` for every arg the chain command declares that the
  terminal command also declares AND actually provides (identifier, holder, …).

  This carries the matching identifier (and any owner/holder the terminal set,
  which the runtime guard cross-checks) from the terminal command onto its
  injected prerequisites, without hard-coding per-skill arg names.
  """
  tokens: list[str] = []
  for flag in getattr(chain_spec, "args", {}) or {}:
    if flag not in (getattr(terminal_spec, "args", {}) or {}):
      continue
    val = _extract_flag_value(terminal_cmd, flag)
    if val is not None:
      tokens.extend([flag, shlex.quote(val)])
  return tokens


def _build_prerequisite_commands(
    terminal_cmd: str, terminal_spec: Any, gates_before: list[str]
) -> list[str]:
  """Construct the prerequisite command lines that must precede ``terminal_cmd``.

  Driven entirely by ``terminal_spec.requires`` (the machine-readable protocol
  contract declared on the skill SSOT): for each precondition emit its ordered
  same-identifier chain (with a long TTL where the chain command supports one),
  then, for an all-ready quantifier (a barrier such as ``deploy cutover``), flip
  every already-seen gate ready — or seed one if none exist — so the barrier can
  cross. Returns the commands to insert, in order. Empty when the terminal
  command does not carry the matching identifier value (nothing constructible).
  """
  from clasharness.scenario.conflict_llm_pipeline import (  # pylint: disable=g-import-not-at-top
      _skill_command_specs,
  )

  specs = _skill_command_specs()
  pre_cmds: list[str] = []
  for pre in getattr(terminal_spec, "requires", ()) or ():
    ident = pre.identifier_arg
    ident_val = _extract_flag_value(terminal_cmd, ident) if ident else None
    if ident and ident_val is None:
      continue  # cannot build a matching chain without the identifier value
    for chain_name in pre.chain:
      chain_spec = specs.get(chain_name.replace(" ", "."))
      if chain_spec is None:
        continue
      parts = [chain_name]
      parts += _copy_shared_provided_args(terminal_cmd, terminal_spec, chain_spec)
      if "--ttl" in (getattr(chain_spec, "args", {}) or {}):
        parts += ["--ttl", str(_INJECTED_PREREQ_TTL)]
      pre_cmds.append(" ".join(parts))
    if pre.all_ready is not None:
      ready_cmd, ready_arg, ready_val = pre.all_ready
      ready_spec = specs.get(ready_cmd.replace(" ", "."))
      # The gate identifier flag is the gate-set's required flag other than the
      # state flag (data-driven, no hard-coded ``--gate``).
      gate_flag = next(
          (
              f
              for f in getattr(ready_spec, "required", ()) or ()
              if f != ready_arg
          ),
          "--gate",
      )
      gates = list(gates_before)
      if not gates:
        gates = [f"{ident_val}-gate" if ident_val else "gate-1"]
      for g in gates:
        pre_cmds.append(
            f"{ready_cmd} {gate_flag} {shlex.quote(g)} {ready_arg} {ready_val}"
        )
  return pre_cmds


def _inject_prerequisites_for_effects(
    sol_cmds: list[str], failing_effects: set[str]
) -> list[str]:
  """Return a copy of ``sol_cmds`` with deterministic prerequisite commands
  inserted before the first terminal command of each failing protocol effect.

  Effects whose emitting command declares no precondition (a chain head such as
  ``lock acquire``) are skipped — there is nothing to construct.
  """
  result = list(sol_cmds)
  for effect_type in sorted(failing_effects):
    terminal_spec = command_spec_for_effect(effect_type)
    if terminal_spec is None or not getattr(terminal_spec, "requires", ()):
      continue
    idx = _first_command_index(result, terminal_spec.name)
    if idx is None:
      continue
    terminal_cmd = result[idx]
    # Derive the gate-set command name from the all-ready quantifier (if any).
    gates_before: list[str] = []
    for pre in terminal_spec.requires:
      if pre.all_ready is not None:
        ready_cmd, ready_arg, _ = pre.all_ready
        from clasharness.scenario.conflict_llm_pipeline import (  # pylint: disable=g-import-not-at-top
            _skill_command_specs,
        )

        rspec = _skill_command_specs().get(ready_cmd.replace(" ", "."))
        gate_flag = next(
            (f for f in getattr(rspec, "required", ()) or () if f != ready_arg),
            "--gate",
        )
        gates_before = _gate_names_before(result, idx, ready_cmd, gate_flag)
    prereqs = _build_prerequisite_commands(
        terminal_cmd, terminal_spec, gates_before
    )
    if not prereqs:
      continue
    result[idx:idx] = prereqs  # insert immediately before the terminal command
  return result


def _resolve_protocol_prerequisites(
    openclaw_spec: ScenarioSpec,
    output_dir: Path,
    log: Callable[[str], None],
) -> int:
  """Deterministic protocol-prerequisite resolver (runs between C2 and F1).

  Conflict rounds start with EMPTY protocol state, so a gold trajectory must
  build each multi-step protocol from scratch, in order, with a matching
  identifier: ``booking hold``→``confirm``; ``pairing start``→``show-code``→
  ``enter-code``→``confirm``; ``deploy request``(→gates ready)→``execute`` /
  ``cutover``. When the LLM-authored gold omits, misorders, or mis-identifies
  those prerequisites, the terminal command's runtime guard rejects it, no
  effect is emitted, and the round's ``exists`` effect check fails — the leading
  cause of kept-with-failures rounds.

  This pass replays each gold once; for any FAILING ``type=="effect"`` check it
  recovers the terminal command's declared precondition contract
  (``CommandSpec.requires`` — the machine-readable SSOT) and deterministically
  inserts the missing prerequisite commands immediately before the terminal
  command. It re-replays and COMMITS the rewrite only when no previously-passing
  check regresses (failing-check indices after ⊆ before) AND the number of
  failing effect checks strictly drops; otherwise it reverts and leaves the
  round to the F1 LLM backstop.

  Diversity / no-regression invariant: a round with no failing effect check is
  never touched (its trajectory stays byte-identical), and a trajectory that
  already passes is never rewritten. Only already-broken rounds gain a
  deterministic prerequisite prefix. Returns the number of rounds repaired.
  """
  repaired = 0
  for round_spec in openclaw_spec.rounds:
    task_id = round_spec.task_id
    try:
      task = Task.from_dict(round_spec.task_spec)
    except Exception as e:  # pylint: disable=broad-except
      log(f"  • {task_id}: prerequisite resolver skipped (load error: {e})")
      continue
    checks = (
        list(task.ground_truth.evaluation_checks) if task.ground_truth else []
    )
    sol_cmds = list(round_spec.expert_trajectory or [])
    if not sol_cmds and task.ground_truth:
      sol_cmds = list(task.ground_truth.solution_commands)
    if not checks or not sol_cmds:
      continue

    try:
      res, _trace, _state = _replay_trajectory(task_id, output_dir, sol_cmds)
    except Exception as e:  # pylint: disable=broad-except
      log(f"  • {task_id}: prerequisite resolver skipped (replay error: {e})")
      continue
    details = list(getattr(res, "details", None) or [])
    before_idx = _failing_indices(checks, details)
    failing_effects = _failing_effect_types(checks, before_idx)
    if not failing_effects:
      continue  # healthy for effects: never touch -> byte-identical trajectory

    new_cmds = _inject_prerequisites_for_effects(sol_cmds, set(failing_effects))
    if new_cmds == sol_cmds:
      continue  # nothing constructible (e.g. missing identifier value)

    try:
      res2, _t2, _s2 = _replay_trajectory(task_id, output_dir, new_cmds)
    except Exception as e:  # pylint: disable=broad-except
      log(f"  • {task_id}: prerequisite re-replay error: {e}; left for F1")
      continue
    details2 = list(getattr(res2, "details", None) or [])
    after_idx = _failing_indices(checks, details2)
    n_eff_before = len(failing_effects)
    n_eff_after = len(_failing_effect_types(checks, after_idx))

    # Commit only if strictly fewer failing EFFECT checks AND no previously
    # passing check regressed (after-failures are a subset of before-failures).
    if not (after_idx <= before_idx and n_eff_after < n_eff_before):
      continue

    if task.ground_truth is not None:
      task.ground_truth.solution_commands = new_cmds
    round_spec.expert_trajectory = new_cmds
    round_spec.task_spec = task.to_dict()
    try:
      save_task(task, data_dir=output_dir)
    except Exception as e:  # pylint: disable=broad-except
      log(f"  • {task_id}: prerequisite re-save failed: {e}")
      continue
    repaired += 1
    log(
        f"  🔩 {task_id}: injected protocol prerequisites "
        f"(effect failures {n_eff_before} -> {n_eff_after})"
    )
  return repaired


def _protocol_contract_for_failed_checks(
    failed: list[dict[str, Any]]
) -> list[dict[str, Any]]:
  """Declarative precondition contract for the failing ``effect`` checks.

  For every failed ``{"type": "effect", ...}`` check, recover the terminal
  command's machine-readable precondition contract (``CommandSpec.requires`` via
  the effect registry SSOT) and render it as a compact, LLM-consumable structure.
  This tells the F1 repair layer *exactly* which ordered, identifier-matched
  prerequisite commands the round is missing — the same contract the
  deterministic resolver uses — so an LLM backstop repair converges instead of
  guessing. Empty when no failing check is a protocol effect (repair proceeds
  without protocol hints, as before).
  """
  seen: set[str] = set()
  out: list[dict[str, Any]] = []
  for f in failed:
    chk = f.get("check") if isinstance(f, dict) else None
    if not isinstance(chk, dict) or chk.get("type") != "effect":
      continue
    et = str(chk.get("effect_type", "") or "")
    if not et or et in seen:
      continue
    seen.add(et)
    spec = command_spec_for_effect(et)
    if spec is None or not getattr(spec, "requires", None):
      continue
    preconditions: list[dict[str, Any]] = []
    for pre in spec.requires:
      entry: dict[str, Any] = {}
      if pre.chain:
        entry["ordered_prerequisites"] = list(pre.chain)
        if pre.identifier_arg:
          entry["matching_arg"] = pre.identifier_arg
      if pre.all_ready is not None:
        ready_cmd, ready_arg, ready_val = pre.all_ready
        entry["all_ready"] = {
            "command": ready_cmd,
            "arg": ready_arg,
            "value": ready_val,
        }
      if entry:
        preconditions.append(entry)
    if preconditions:
      out.append({
          "effect_type": et,
          "terminal_command": spec.name,
          "preconditions": preconditions,
      })
  return out


def _verify_and_repair(
    openclaw_spec: ScenarioSpec,
    output_dir: Path,
    log: Callable[[str], None],
    llm_client: Any = None,
    max_iterations: int = 5,
) -> dict[str, Any]:
  """F1 verification with LLM self-repair. NEVER discards a round.

  For every compiled round we replay its reference (expert) trajectory in a
  throwaway env. Because ``build_evaluator`` + ``EvaluatorComb`` preserve check
  order, ``result.details[i]`` corresponds to ``evaluation_checks[i]``, so we can
  tell exactly which checks the gold solution fails.

  When the gold trajectory fails one or more of its own checks (a broken /
  self-inconsistent task), we invoke the LLM
  ``repair_trajectory_simulation_failure`` layer to rewrite the trajectory so it
  satisfies the checks, re-verify, and retry up to ``max_iterations`` times (the
  ``--auditor_max_iterations`` budget).

  Product decision: a round is ALWAYS kept. If repair does not fully succeed
  within the budget, we keep the LAST repaired version rather than discarding the
  round, so ``openclaw_spec.rounds`` is left intact. Returns a {task_id: report}
  dict for every round that is not cleanly passing (kept-with-failures /
  unverifiable), purely for transparency.
  """
  def _failing(checks: list[Any], details: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, c in enumerate(checks):
      if i < len(details) and not details[i].passed:
        out.append({"check": c, "reason": getattr(details[i], "message", "")})
    return out

  report: dict[str, Any] = {}

  for round_spec in openclaw_spec.rounds:
    task_id = round_spec.task_id
    task = Task.from_dict(round_spec.task_spec)
    checks = list(task.ground_truth.evaluation_checks) if task.ground_truth else []
    sol_cmds = list(round_spec.expert_trajectory or [])
    if not sol_cmds and task.ground_truth:
      sol_cmds = list(task.ground_truth.solution_commands)

    # Nothing to verify against -> keep as-is (never discard), just record it.
    if not checks or not sol_cmds:
      reason = "no_checks" if not checks else "no_trajectory"
      report[task_id] = {"status": "kept_unverified", "reason": reason}
      log(f"  • {task_id}: kept (unverified: {reason})")
      continue

    try:
      res, trace, _state = _replay_trajectory(task_id, output_dir, sol_cmds)
    except Exception as e:  # env failure -> cannot certify, but still keep
      report[task_id] = {"status": "kept_unverified", "reason": f"verify_error: {e}"}
      log(f"  • {task_id}: kept (verify error: {e})")
      continue

    instruction = getattr(task, "instruction", "") or str(
        getattr(round_spec, "instruction", "") or ""
    )
    attempts = 0
    while _failing(checks, res.details) and attempts < max_iterations:
      attempts += 1
      failed = _failing(checks, res.details)
      log(
          f"  🔧 {task_id}: gold fails {len(failed)} check(s); "
          f"LLM repair attempt {attempts}/{max_iterations}"
      )
      if llm_client is None:
        log("    no llm_client available; cannot self-repair")
        break
      payload = {
          "instruction": instruction,
          "trajectory": sol_cmds,
          "trace": trace[:8000],
          "failed_checks": failed,
          "all_checks": checks,
      }
      # B3: when a failing check is a protocol effect, hand the LLM the exact
      # declarative precondition contract (ordered, identifier-matched
      # prerequisite commands) recovered from the skill SSOT, so the repair
      # targets the real cause (a missing/misordered protocol prefix) instead of
      # guessing. Absent for non-protocol failures.
      protocol_contract = _protocol_contract_for_failed_checks(failed)
      if protocol_contract:
        payload["protocol_contract"] = protocol_contract
      try:
        rr = llm_client.generate_layer(
            "repair_trajectory_simulation_failure", payload
        )
      except Exception as e:
        log(f"    repair LLM call failed: {e}; keeping current version")
        break
      updated = rr.get("updated_trajectory")
      if not isinstance(updated, list) or not updated:
        log("    repair returned no trajectory; keeping current version")
        break
      sol_cmds = [str(c) for c in updated]
      if task.ground_truth is not None:
        task.ground_truth.solution_commands = sol_cmds
      round_spec.expert_trajectory = sol_cmds
      round_spec.task_spec = task.to_dict()
      save_task(task, data_dir=output_dir)
      try:
        res, trace, _state = _replay_trajectory(task_id, output_dir, sol_cmds)
      except Exception as e:
        log(f"    re-verify failed: {e}; keeping current version")
        break

    remaining = _failing(checks, res.details)
    if remaining:
      report[task_id] = {
          "status": "kept_with_failures",
          "repair_attempts": attempts,
          "failed_checks": [f["check"] for f in remaining],
      }
      log(
          f"  ⚠️ {task_id}: kept LAST version after {attempts} repair attempt(s); "
          f"still {len(remaining)} failing check(s)"
      )
    elif attempts:
      log(
          f"  ✅ {task_id}: repaired — gold now passes all checks "
          f"(after {attempts} attempt(s))"
      )
    else:
      log(f"  ✅ {task_id}: gold passes all checks")

  n = len(openclaw_spec.rounds)
  log(
      f"F1 verify+repair: {n} round(s) kept (never discarded); "
      f"{n - len(report)} clean, {len(report)} kept-with-issues."
  )
  return report


def _generate_html_visualizer(
    spec: ConflictVLMScenarioSpec,
    output_dir: Path,
    progress: Callable[[str], None] | None = None,
) -> None:
  """Compile and write visualize.html inside output_dir embedding spec & assets."""
  import base64
  import json
  from copy import deepcopy

  def _log(msg: str) -> None:
    if progress:
      progress(msg)
    else:
      logging.info(msg)

  _log("Generating scenario HTML visualizer...")
  spec_data = deepcopy(spec.to_dict())

  # Base64 encode image assets
  for asset in spec_data.get("assets", []):
    asset_path = output_dir / "assets" / asset.get("path", "")
    if not asset_path.exists():
      asset_path = output_dir / asset.get("path", "")

    if asset_path.is_file():
      try:
        mime_type = asset.get("mime_type") or "image/png"
        with open(asset_path, "rb") as img_f:
          b64_data = base64.b64encode(img_f.read()).decode("utf-8")
        asset["base64_data"] = f"data:{mime_type};base64,{b64_data}"
      except Exception as e:
        _log(f"  Warning: Failed to base64 encode asset {asset.get('id')}: {e}")

  # Read HTML template
  template_path = (
      Path(__file__).parent.parent.parent
      / "scripts"
      / "templates"
      / "visualize_template.html"
  )
  if not template_path.exists():
    _log(f"Warning: Visualizer template not found at {template_path}, skipping visualizer generation.")
    return

  try:
    with open(template_path, "r", encoding="utf-8") as f:
      template_content = f.read()

    spec_str = json.dumps(spec_data, indent=2, ensure_ascii=False)
    output_content = template_content.replace(
        "/* SCENARIO_DATA_PLACEHOLDER */", spec_str
    )

    output_file = output_dir / "visualize.html"
    with open(output_file, "w", encoding="utf-8") as f:
      f.write(output_content)

    _log(f"SUCCESS: Visualizer HTML compiled at: file://{output_file}")
  except Exception as e:
    _log(f"Warning: Failed to compile visualizer HTML: {e}")


def _decisive_values_by_asset(rounds: Any) -> dict[str, list[str]]:
  """Map ``asset_id -> decisive readings that asset must keep carrying``.

  Reads the resolutions the pipeline already produced. A round contributes only
  when its ``resolved_value_locus`` names an ASSET: that is precisely the case
  ``_validate_round_resolution`` grounded, so it is the only case where a
  post-refine survival check is asserting something the pipeline actually
  promised. ``file:`` loci point at workspace files the refiner never touches,
  and ``computed`` loci were never grounded to begin with -- both are skipped
  rather than guessed at.

  Deliberately NOT a text search over every asset: the locus is the designated
  carrier, and widening protection to incidental mentions would raise the
  carrier count, which measurably lowers difficulty.
  """
  out: dict[str, list[str]] = {}
  for r_dict in rounds or []:
    if not isinstance(r_dict, dict):
      continue
    value = str(r_dict.get("resolved_value") or "").strip()
    if not value:
      continue
    kind, target = _parse_resolved_locus(r_dict.get("resolved_value_locus"))
    if kind != "asset" or not target:
      continue
    bucket = out.setdefault(target, [])
    if value not in bucket:
      bucket.append(value)
  return out


def _generate_fresh_spec(
    seed: ConflictVLMSeed,
    llm_client: ScenarioLLMClient,
    progress: Callable[[str], None] | None = None,
    image_provider: str = "mock",
    chart_provider: str = "mock",
    output_dir: Path | None = None,
    resume: bool = False,
    num_generation_workers: int | None = None,
    phase_expand_attempts: int | None = None,
    style_seed: int = 0,
    checkpoint_dir: Path | None = None,
) -> ConflictVLMScenarioSpec:
  """Generate core text and rounds details from scratch."""
  from dataclasses import replace

  # Synchronize seed generation policy with runtime argument values
  # so that Stage 5 prompt generation is aware of the actual rendering providers
  seed = replace(
      seed,
      generation_policy=replace(
          seed.generation_policy,
          chart_provider=chart_provider,
          natural_image_provider=image_provider,
      ),
  )

  # 1. Run the LLM Pipeline to generate the core scenario data.
  #
  # The pipeline checkpoint deliberately defaults to ``output_dir`` but SHOULD
  # be pointed somewhere else by callers that delete ``output_dir`` when a
  # version fails. Measured on a 60-scenario run: the checkpoint is written
  # after Stage 1c and again after Stage 2, i.e. BEFORE Stage 3 -- so when
  # Stage 3 died there was a complete, resumable checkpoint on disk, and the
  # caller's failure cleanup (``shutil.rmtree(scenario_output_dir)``) then
  # deleted it along with the half-written spec. Two scenarios were lost that
  # way, each discarding ~85s and ~10 layer calls of already-VALID Stage 1-2
  # work that a resume would have skipped entirely. Keeping the checkpoint
  # outside the deleted tree is what makes a retry cheap instead of total.
  pipeline = ConflictLLMPipeline(client=llm_client, progress=progress)
  raw_data = pipeline.generate(
      seed,
      checkpoint_dir=checkpoint_dir if checkpoint_dir is not None else output_dir,
      resume=resume,
      phase_expand_attempts=phase_expand_attempts,
  )

  logging.info("Structuring scenario specification...")

  # 2. Build Updates
  updates: list[ConflictScenarioUpdate] = []
  for u_dict in raw_data.get("updates", []):
    updates.append(ConflictScenarioUpdate.from_dict(u_dict))

  # 2.5 Build Assets
  assets: list[VLMAssetSpec] = []
  for a_dict in raw_data.get("assets", []):
    asset_spec = VLMAssetSpec.from_dict(a_dict)
    # Map layout overrides from seed-level generation policy
    layout = seed.generation_policy.chart_layouts.get(asset_spec.id, "devops_dashboard")
    if not isinstance(asset_spec.provider_metadata, dict):
      asset_spec.provider_metadata = {}
    asset_spec.provider_metadata["layout"] = layout
    asset_spec = _backfill_asset_anchors(asset_spec)
    assets.append(asset_spec)

  # Automatically map round expected anchors to visible assets for OCR refinement backfill
  assets = _backfill_missing_round_anchors_to_assets(assets, raw_data)

  # Assign one *coherent* styling bundle per platform_id, sampled from the SSoT
  # design-archetype library (design_archetypes). This retires the historical
  # 6-item ``STYLE_PRESETS`` (a crude theme+color pair chosen by ``hash % 6``),
  # which set only 2 of the 4 styling fields and could clash with the platform's
  # archetype -- there is now a single style path (the SSoT sampler) shared with
  # the base materializer's design + no-LLM fallback paths. Every chart on a
  # platform gets the same theme / primary_color / visual_style / style_details,
  # and the choice is deterministic for a given (slug, platform_id, style_seed),
  # so a pinned ``style_seed`` reproduces the platform's look (WD5).
  platform_styles: dict[str, dict[str, Any]] = {}
  styled_assets = []
  for asset in assets:
    if asset.kind == "chart":
      pid = asset.platform_id or "default"
      if pid not in platform_styles:
        design = design_archetypes.sample_platform_design(
            f"{seed.slug}_{pid}_{style_seed}"
        )
        platform_styles[pid] = {
            "theme": design["color_theme"],
            "primary_color": design["primary_color"],
            "visual_style": design["visual_style"],
            "style_details": design["style_details"],
        }
      style = platform_styles[pid]
      new_overrides = dict(asset.custom_overrides)
      if "styling" not in new_overrides:
        new_overrides["styling"] = {}
      else:
        new_overrides["styling"] = dict(new_overrides["styling"])
      # Platform bundle wins for the four coherent axes (enforces per-platform
      # consistency); any other author-set styling keys are preserved.
      new_overrides["styling"].update(style)
      asset = replace(asset, custom_overrides=new_overrides)
    styled_assets.append(asset)
  assets = styled_assets

  # Soft diversity check (WARN-only, never aborts): a healthy scenario renders
  # its assets across at least a couple of distinct web-UI surfaces. Collapsing
  # onto a single surface usually means the platform vocabulary/remap
  # under-performed for this seed (>= 2 distinct is acceptable).
  distinct_platforms = {a.platform_id for a in assets if a.platform_id}
  if 0 < len(distinct_platforms) < _MIN_PLATFORM_VOCAB:
    _diversity_warning = (
        f"WARNING: scenario '{seed.slug}' renders assets on only "
        f"{len(distinct_platforms)} distinct platform(s) "
        f"({sorted(distinct_platforms)}); soft target is >= {_MIN_PLATFORM_VOCAB}."
    )
    if progress:
      progress(_diversity_warning)
    else:
      logging.warning(_diversity_warning)

  if llm_client is not None and assets:
    logging.info("Refining visual asset definitions using LLM...")
    assets = llm_refine_vlm_assets(
        seed,
        assets=assets,
        client=llm_client,
        progress=progress,
        num_generation_workers=num_generation_workers,
        decisive_values_by_asset=_decisive_values_by_asset(raw_data.get("rounds", [])),
    )

  # 2.8 Build Workspace and Sessions
  logging.info("Structuring workspace and sessions...")
  workspace = ScenarioWorkspace.from_dict(raw_data.get("workspace", {}))
  sessions = []
  for s_dict in raw_data.get("sessions", []):
    s_obj = ScenarioSession.from_dict(s_dict)
    sessions.append(s_obj)

  # 3. Build Rounds
  logging.info("Compiling evaluation rounds to executable round specs...")
  rounds: list[ConflictVLMRoundSpec] = []
  global_mock_subagents = [
      MockSubagent.from_dict(m)
      for m in raw_data.get("mock_subagents_global", [])
  ]

  for r_dict in raw_data.get("rounds", []):
    round_id = str(r_dict.get("round_id", ""))
    try:
      round_index = int(round_id.lstrip("r"))
    except ValueError:
      round_index = 1

    # Filter assets based on available_from_round (effective round, so an
    # untrusted bare id is not surfaced before its first real timestamped
    # snapshot exists).
    visible_asset_ids = [
        a.id
        for a in assets
        if _effective_available_from_round(a, assets) <= round_index
    ]

    # Find expected_anchor_ids from evals
    round_evals = [
        e for e in raw_data.get("evals", []) if e.get("round_id") == round_id
    ]
    expected_anchors = (
        round_evals[0].get("expected_anchor_ids", []) if round_evals else []
    )

    # We merge global mock subagents with any round-specific ones
    r_mock_dicts = r_dict.get("mock_subagents", [])
    r_mocks = [MockSubagent.from_dict(m) for m in r_mock_dicts]

    # If the LLM didn't define round-specific mocks, use the global ones
    final_mocks = r_mocks if r_mocks else global_mock_subagents

    rounds.append(
        ConflictVLMRoundSpec(
            round_id=round_id,
            round_type=str(r_dict.get("round_type", "mc")),
            visible_asset_ids=visible_asset_ids,
            required_modalities=seed.modalities,
            expected_anchor_ids=expected_anchors,
            required_tool_families=seed.allowed_tool_families,
            instruction=_sanitize_instruction(str(r_dict.get("instruction", ""))),
            update_ids=[str(i) for i in r_dict.get("update_ids", [])],
            solution_commands=[
                str(c) for c in r_dict.get("solution_commands", [])
            ],
            mock_subagents=final_mocks,
            # ③: carry the round's graded conclusion onto the shipped round spec.
            decision_outcome=str(r_dict.get("decision_outcome", "")),
            resolved_value=str(r_dict.get("resolved_value", "")),
            # M7 (Axis A): thread the decisive value's locus onto the shipped round
            # so the materializer can identify the decisive asset and promote its
            # access check to ``required`` for verifiably vision-confined rounds.
            resolved_value_locus=str(r_dict.get("resolved_value_locus", "")),
        )
    )

  # 4. Create the Final Spec
  spec = ConflictVLMScenarioSpec(
      series="conflict_vlm_v1",
      scenario_slug=seed.slug,
      title=seed.title,
      domain=seed.domain,
      core_conflict=seed.core_conflict,
      conflict_taxonomy=seed.conflict_taxonomy,
      modalities=seed.modalities,
      input_policy=seed.input_policy,
      generation_policy=seed.generation_policy,
      allowed_tool_families=seed.allowed_tool_families,
      workspace=workspace,
      sessions=sessions,
      assets=assets,
      updates=updates,
      rounds=rounds,
      evals=[VLMEvalSpec.from_dict(e) for e in raw_data.get("evals", [])],
      agents=[ConflictAgentSpec.from_dict(a) for a in raw_data.get("agents", [])],
      interactions=[ConflictInteractionSpec.from_dict(i) for i in raw_data.get("interactions", [])],
      metadata={
          "summary": raw_data.get("summary", ""),
          "screenplay_outline": raw_data.get("screenplay_outline", {}),
          "global_phases": raw_data.get("phases", []),
      },
      platforms=raw_data.get("platforms", []),
  )
  return spec


def _true_value_is_recoverable_from(true_value: str, text: str) -> bool:
  """Would a ``field_contains`` check for ``true_value`` pass against ``text``?

  Delegates to the CHECKER's own tokenizer rather than reimplementing it. That
  is deliberate: both M2 (the decoy must not accidentally satisfy the real
  check) and M3 (stripping must not destroy a check the gold relies on) are
  only sound if this predicate agrees exactly with grading. A local copy would
  drift silently and turn both guarantees into decoration.
  """
  return _is_subsequence(
      _tokenize_and_clean(str(true_value).lower()),
      _tokenize_and_clean(str(text).lower()),
  )


def _decoy_value_for(true_value: str) -> str | None:
  """Derive a deterministic competing reading for a decisive value (M2).

  Perturbs the LAST number in the value, which is where these values actually
  discriminate -- both for metrics (``1,820 ms``) and for the identifier forms
  that dominate this corpus (``SIG-APP-STAGE-8821``, ``SEC-9942-Z``). Keeping
  the surrounding text intact is the point: a decoy that reads
  ``SIG-APP-STAGE-8824`` is confusable, whereas an unrelated string is not
  tempting and tests nothing.

  Returns ``None`` when no candidate can be made SAFE. Safety is not
  "tokens are disjoint" -- that is far too strict and would reject the
  confusable identifier case above, whose tokens necessarily overlap. The
  precise condition is that the true value must NOT be recoverable from the
  decoy text, i.e. an agent that reports the decoy cannot accidentally satisfy
  the true-value check.
  """
  text = str(true_value or "").strip()
  if not text:
    return None
  matches = list(re.finditer(r"\d+", text))
  if not matches:
    # Non-numeric values (19% of the corpus) would need domain vocabulary to
    # contradict credibly; inventing one risks an incoherent or accidentally
    # correct decoy. Skip rather than guess.
    return None
  m = matches[-1]
  head, digits, tail = text[: m.start()], m.group(0), text[m.end() :]
  # Deterministic candidate ladder, tried in order. Same-width replacements
  # first so the decoy keeps the shape of a real reading.
  candidates: list[str] = []
  n = int(digits)
  for delta in (3, 7, 11, 23, 41, 104):
    alt = n + delta
    if len(str(alt)) == len(digits):
      candidates.append(str(alt).zfill(len(digits)))
  for delta in (3, 7, 11, 23, 41):
    alt = n - delta
    if alt > 0 and len(str(alt)) == len(digits):
      candidates.append(str(alt).zfill(len(digits)))
  for cand in candidates:
    decoy = f"{head}{cand}{tail}"
    if decoy == text:
      continue
    if not _true_value_is_recoverable_from(text, decoy):
      return decoy
  return None


# Phrasings for the injected decoy. The decoy is attributed to a STALE or
# SECONDARY read, never presented as the authoritative surface: the round must
# stay solvable by an agent that checks the decisive asset, so the decoy has to
# be defeasible evidence, not a contradiction of the ground truth.
_DECOY_TEMPLATES: tuple[str, ...] = (
    "Heads up -- my cached copy of the panel still shows {decoy}; I have not"
    " refreshed it, so confirm against the live surface before you record it.",
    "For what it's worth, the mirrored readout on my side reads {decoy}."
    " That export lags, so treat it as indicative only.",
    "Note: the summary I was handed lists {decoy}. I did not pull it myself,"
    " so please verify it directly rather than taking my number.",
)


def _inject_decoy(
    mock_subagents: list[dict[str, Any]],
    background_events: list[dict[str, Any]],
    round_spec: "ConflictVLMRoundSpec",
    slug: str,
) -> str | None:
  """M2: add a competing (wrong) reading to one sub-agent's reply.

  Grading is deliberately NOT touched. The round's existing
  ``field_contains``/``field_semantic_match`` checks against the TRUE value do
  all the work: an agent that believes the decoy reports the wrong value and
  fails them. A "must not say the decoy" negative check was considered and
  rejected -- ``not_exists`` is whole-effect-type only (it cannot filter on a
  field value), and ``field_contains`` on ``messages_sent`` matches by token
  SUBSEQUENCE, so such a check would fire whenever the decoy's tokens happened
  to appear in order in an otherwise-correct report.

  Returns the decoy string actually injected, or ``None`` if the round was left
  untouched.
  """
  decoy = _decoy_value_for(str(round_spec.resolved_value or ""))
  if decoy is None or not mock_subagents:
    return None
  # Place it on a deterministic sub-agent, chosen per round so the decoy is not
  # always the same voice.
  digest = hashlib.sha1(f"{slug}:{round_spec.round_id}".encode("utf-8")).hexdigest()
  idx = int(digest, 16) % len(mock_subagents)
  template = _DECOY_TEMPLATES[int(digest, 16) % len(_DECOY_TEMPLATES)]
  line = template.format(decoy=decoy)
  target_role = mock_subagents[idx].get("agent_role")
  original = str(mock_subagents[idx].get("mock_response") or "")
  # Belt and braces: if the decoy line somehow makes the TRUE value readable
  # (e.g. the true value is a bare digit the template reproduces), skip.
  if _true_value_is_recoverable_from(str(round_spec.resolved_value or ""), line):
    return None
  updated = f"{original.rstrip()} {line}".strip()
  mock_subagents[idx]["mock_response"] = updated
  # Keep the background event in sync: it is the copy the environment actually
  # replays to the solver, so updating only ``mock_subagents`` would inject a
  # decoy the agent never sees.
  for ev in background_events:
    if (
        ev.get("type") == "subagent_completion"
        and ev.get("agent_name") == target_role
        and str(ev.get("mock_response") or "") == original
    ):
      ev["mock_response"] = updated
  return decoy


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# G1(b) appends this to the task text whenever an ``assets_accessed`` check
# grades the round. M3 MUST NOT strip it: it names no asset, and it is the
# deterministic guarantee that a de-signposted round stays solvable -- it tells
# the agent the discovery primitive (``asset list``) exists. Without it,
# removing the asset's name from the instruction turns a search problem into a
# guessing game. Kept as a module constant so the strip and the append can never
# drift apart.
_ASSET_LIST_GUIDANCE = (
    "Before deciding, run `asset list` to see the available visual evidence "
    "and inspect every relevant chart, screenshot, or diagram with "
    "`asset open` (or `asset info`) so your analysis is grounded in what they "
    "actually show."
)


class _SignpostStrip(NamedTuple):
  """Outcome of one round's de-signposting, across both channels.

  ``outcome`` keeps the four original values so existing consumers and the
  ``difficulty_knobs`` record stay comparable across corpus generations;
  ``channel`` is the new field that says WHERE the strip landed, which is the
  distinction the shipped corpus cannot reconstruct after the fact.
  """

  removed: list[str]
  outcome: str
  instruction: str
  channel: str
  instruction_removed: int


def _asset_mention_terms(asset_id: str) -> list[str]:
  """Surface forms by which a reply might name the decisive asset."""
  base = asset_id.strip()
  if not base:
    return []
  terms = {base.lower()}
  # ``lock_manager_console_concurrency_status_panel`` -> the prose form
  # ``lock manager console concurrency status panel`` and its platform head
  # ``lock manager console``, which is how sub-agents actually sign-post.
  words = re.sub(r"[_/]+", " ", base).strip()
  words = re.sub(r"\s*\d{6,}\s*$", "", words).strip()  # drop timestamp suffix
  if words:
    terms.add(words.lower())
    parts = words.split()
    for n in (4, 3, 2):
      if len(parts) > n:
        terms.add(" ".join(parts[:n]).lower())
  return sorted(t for t in terms if len(t) >= 6)


def _strip_signposts(
    mock_subagents: list[dict[str, Any]],
    background_events: list[dict[str, Any]],
    decisive_asset_id: str,
    checks: list[Any],
    instruction: str = "",
) -> _SignpostStrip:
  """M3: remove the text that names the decisive asset outright.

  TWO channels, because rewriting only one of them made the first cut of this
  knob near-inert. On the shipped v3 corpus (598 asset-locus rounds; a round
  "names" the asset when the text contains >=2 CONSECUTIVE words of the asset's
  prose name, at least one of them not a generic platform word like "console"
  or "panel"):

      the round instruction  526 (88.0%)
      a sub-agent reply      427 (71.4%)
      either                 543 (90.8%)
      a reply and NOT the instruction ... 17 (2.8%)

  The original implementation rewrote replies only, so its reachable ceiling was
  that last line -- 17 rounds out of 598. Stripping a reply while the round's
  own instruction still names the asset does not de-signpost the round; it only
  makes the telemetry say ``stripped``. The instruction is the dominant channel
  and is handled here too. That is the entire justification for this function:
  it is a CORRECTNESS fix, making the knob do what its name claims.

  It is NOT, on current evidence, a difficulty lever. An earlier revision of
  this docstring claimed de-signposted rounds are 9.4pp harder. That was
  withdrawn: recomputed with the predicate above, paired within
  ``seed x model x round_type``, the effect is -6.8pp n-weighted but -2.0pp
  unweighted with a MEDIAN cell effect of +0.0pp, only 46% of cells pointing
  that way, and 37% of the weighted total coming from five cells. The
  n-weighted number was an artifact of a handful of extreme cells. Whether
  forcing the condition makes rounds harder is untested -- the observational
  data can neither establish it nor rule it out.

  Why it stays fair: ``asset list`` is a real command every scenario exposes,
  and ``_ASSET_LIST_GUIDANCE`` (which this function refuses to strip) tells the
  agent to run it. De-signposting turns the round into a search over an
  enumerable set, not a guess. Gold never needed that search -- ``asset list``
  appears 0 times in 5757 gold commands -- which is precisely the slack being
  removed.

  Rollback granularity:
    * gold-solvability (a graded check's literal value would stop being
      recoverable) rolls back the WHOLE round, both channels;
    * "this strip would leave nothing behind" rolls back only ITS OWN channel,
      so an unstrippable instruction does not also forfeit the reply strip.

  Outcomes (unchanged vocabulary, so records stay comparable across corpus
  generations). An empty ``removed`` list is AMBIGUOUS in exactly the way that
  ruins the measurement: "never signposted" and "signposted but rolled back"
  both ship a round with no strip applied, and the shipped corpus cannot tell
  them apart afterwards, so the outcome is recorded here at the source:
    ``stripped``            -- text removed from at least one channel.
    ``no_mention``          -- nothing named the asset; knob was a no-op.
    ``rolled_back_empty``   -- every channel that matched would have emptied.
    ``rolled_back_check``   -- stripping would have broken a graded check.
  ``channel`` reports where it landed (``instruction``/``mock``/``both``/``""``).
  """
  terms = _asset_mention_terms(decisive_asset_id)
  if not terms:
    return _SignpostStrip([], "no_mention", instruction, "", 0)
  # Every literal value a text check depends on. If stripping would take one of
  # these with it, the gold trajectory stops grading and the round is broken.
  guarded: list[str] = []
  for c in checks:
    spec_dict = c if isinstance(c, dict) else getattr(c, "__dict__", {})
    exp = spec_dict.get("expected")
    if isinstance(exp, dict) and exp.get("value"):
      guarded.append(str(exp["value"]))

  def _split_on_terms(text: str) -> tuple[list[str], list[str]]:
    """Sentence-split ``text`` into (kept, dropped) by asset mention."""
    keep, drop = [], []
    for s in _SENTENCE_SPLIT_RE.split(text):
      # Never drop the ``asset list`` guidance: it names no asset and it is what
      # keeps a de-signposted round solvable. Compared on stripped text because
      # G1(b) joins it on with "\n\n".
      if s.strip() == _ASSET_LIST_GUIDANCE:
        keep.append(s)
        continue
      (drop if any(t in s.lower() for t in terms) else keep).append(s)
    return keep, drop

  # --- channel 1: sub-agent replies -----------------------------------------
  proposed: list[tuple[int, str, str]] = []  # (idx, original, stripped)
  removed: list[str] = []
  mock_matched = False
  mock_empty_rollback = False
  for i, sub in enumerate(mock_subagents):
    original = str(sub.get("mock_response") or "")
    if not original:
      continue
    keep, drop = _split_on_terms(original)
    if not drop:
      continue
    mock_matched = True
    stripped = " ".join(keep).strip()
    if not stripped:
      # Stripping would leave this sub-agent replying with nothing. Measured on
      # the shipped corpus a naive strip empties 66 replies, which is a broken
      # artifact in its own right (the coordinator delegates and hears
      # silence). It also breaks M8/CL2: ``_build_async_within_window_check``
      # runs AFTER this and binds its temporal check to a sub-agent's REPLY, so
      # an emptied reply would quietly void the async gate on the seeds that
      # use both mechanisms. Roll THIS channel back rather than ship it -- the
      # instruction channel is independent and still gets its chance.
      mock_empty_rollback = True
      proposed.clear()
      removed.clear()
      break
    proposed.append((i, original, stripped))
    removed.extend(drop)

  # --- channel 2: the round instruction (the dominant one, 88.1%) -----------
  instr_matched = False
  instr_empty_rollback = False
  instr_removed: list[str] = []
  new_instruction = instruction
  if instruction.strip():
    keep, drop = _split_on_terms(instruction)
    if drop:
      instr_matched = True
      # Rejoin. Measured on the shipped corpus, a round instruction is a single
      # paragraph of single-space-separated prose (600/600 rounds: no newline at
      # all), so the ONLY structure to preserve is the blank line G1(b) puts
      # before the guidance. Preserve it: a plain " ".join would leave
      # de-signposted instructions formatted differently from signposted ones,
      # which is an avoidable second difference between the two arms when the
      # effect gets attributed.
      candidate = ""
      for k in (k.strip() for k in keep):
        if not k:
          continue
        if not candidate:
          candidate = k
        else:
          candidate += ("\n\n" if k == _ASSET_LIST_GUIDANCE else " ") + k
      # An instruction stripped down to nothing (or down to the guidance
      # boilerplate alone) states no task at all. That is not a hard round, it
      # is a broken one.
      if not candidate or candidate == _ASSET_LIST_GUIDANCE:
        instr_empty_rollback = True
      else:
        new_instruction = candidate
        instr_removed = drop

  if not mock_matched and not instr_matched:
    return _SignpostStrip([], "no_mention", instruction, "", 0)
  if not proposed and not instr_removed:
    # Something named the asset everywhere it appeared, but every channel that
    # matched had to be rolled back for emptiness.
    assert mock_empty_rollback or instr_empty_rollback
    return _SignpostStrip([], "rolled_back_empty", instruction, "", 0)

  # Gold-solvability gate: re-evaluate every guarded value against the FULL
  # post-strip text of BOTH channels. A value that was recoverable before and is
  # not after means we just deleted evidence the grader needs. The instruction
  # has to be in here: a check whose literal lives only in the task text would
  # otherwise be silently un-gradable.
  def _corpus(use_stripped: bool) -> str:
    out = []
    replacement = {i: s for i, _o, s in proposed}
    for i, sub in enumerate(mock_subagents):
      if use_stripped and i in replacement:
        out.append(replacement[i])
      else:
        out.append(str(sub.get("mock_response") or ""))
    out.append(new_instruction if use_stripped else instruction)
    return " ".join(out)

  before, after = _corpus(False), _corpus(True)
  for value in guarded:
    if _true_value_is_recoverable_from(value, before) and not (
        _true_value_is_recoverable_from(value, after)
    ):
      # Roll the whole round back, both channels: the two strips were validated
      # together and only their combination is known to be safe.
      return _SignpostStrip([], "rolled_back_check", instruction, "", 0)

  for i, original, stripped in proposed:
    mock_subagents[i]["mock_response"] = stripped
    role = mock_subagents[i].get("agent_role")
    for ev in background_events:
      if (
          ev.get("type") == "subagent_completion"
          and ev.get("agent_name") == role
          and str(ev.get("mock_response") or "") == original
      ):
          ev["mock_response"] = stripped

  if proposed and instr_removed:
    channel = "both"
  elif proposed:
    channel = "mock"
  else:
    channel = "instruction"
  return _SignpostStrip(
      removed + instr_removed, "stripped", new_instruction, channel,
      len(instr_removed),
  )


def _compile_round_to_task(
    seed: ConflictVLMSeed,
    round_spec: ConflictVLMRoundSpec,
    evals: list[VLMEvalSpec],
    round_updates: list[ConflictScenarioUpdate],
    round_mocks: list[MockSubagent],
    spec: ConflictVLMScenarioSpec,
) -> Task:
  task_id = f"conflict_vlm_{seed.slug}_{round_spec.round_id}"

  # Extract checks from evals for this round
  checks = []
  for e in evals:
    if e.round_id == round_spec.round_id:
      checks.extend(e.checks)

  # F3 (modality/strategy neutrality): asset-access is a *ritual*, not an
  # outcome. Scoring ``assets_accessed`` rewards a specific tool sequence
  # (``asset open/read-text/describe``) and penalizes an agent that solves the
  # task from the equivalent text channel, so it must not gate success or affect
  # the score. Previously such checks were dropped outright because the old
  # evaluator computed ``success = all(passed)`` and could not carry a
  # non-gating check. With role-aware evaluation (``EvaluatorComb``) we instead
  # KEEP them tagged ``role="diagnostic"`` - recorded for observability (did the
  # agent look at the evidence?) but affecting neither success nor score. This
  # preserves the behavioural signal without losing data.
  #
  # M7 (Axis A): a round is "verifiably vision-confined" when the seed opts in
  # (``force_vision_locus``) AND the resolver shipped an ``asset:<id>`` locus --
  # which only happens after the grounding + confidentiality gates passed, i.e. the
  # decisive value is readable ONLY from that image. For such rounds we promote that
  # one asset's access check to ``role="required"`` (graded) so the round cannot be
  # solved without looking at the image; every other ``assets_accessed`` check stays
  # ``diagnostic``. Non-opted seeds (``force_vision_locus=False``) keep the legacy
  # all-diagnostic behaviour and stay byte-identical (E6).
  decisive_asset_id: str | None = None
  if seed.force_vision_locus:
    locus = (round_spec.resolved_value_locus or "").strip()
    if locus.startswith("asset:"):
      decisive_asset_id = locus[len("asset:") :].strip() or None
  checks = _tag_asset_access_checks_diagnostic(
      checks, decisive_asset_id=decisive_asset_id
  )

  # Use solution_commands directly from the round_spec
  solution_commands = list(round_spec.solution_commands)
  if not solution_commands:
    solution_commands = ["DONE"]

  # Build initial_state_overrides
  initial_state_overrides: dict[str, Any] = {}

  # 1. Add baseline workspace files
  for f in spec.workspace.files:
    rel_path = f.get("path")
    content = f.get("content")
    if rel_path and content:
      rel_path = rel_path.lstrip("/")
      initial_state_overrides[f"workspace/{rel_path}"] = content

  # 2. Add sessions (as JSONL files)
  for session in spec.sessions:
    session_path = session.path or f"sessions/{session.session_id}.jsonl"
    lines = []
    for msg in session.messages:
      lines.append(
          json.dumps(
              {
                  "message_id": msg.message_id,
                  "timestamp": msg.timestamp,
                  "sender": msg.sender,
                  "text": msg.text,
                  "attachments": msg.attachments,
                  "is_redacted": msg.is_redacted,
              },
              ensure_ascii=False,
          )
          + "\n"
      )
    initial_state_overrides[session_path] = "".join(lines)

  # 3. Add visible assets and asset_manifest.json
  try:
    round_index = int(round_spec.round_id.lstrip("r"))
  except ValueError:
    round_index = 1

  visible_assets = [
      asset
      for asset in spec.assets
      if _effective_available_from_round(asset, spec.assets) <= round_index
  ]
  # G3: collapse bare/timestamped twins of the same snapshot so the agent sees
  # one canonical id per visual instead of confusing look-alikes.
  visible_assets = _dedup_visible_assets(visible_assets)

  for asset in visible_assets:
    # Add asset content to overrides
    if asset.content_b64:
      initial_state_overrides[f"assets/{asset.path}"] = {
          "__bytes_b64__": asset.content_b64
      }
    else:
      initial_state_overrides[f"assets/{asset.path}"] = asset.content_text

  # Generate asset manifest for this round
  manifest = VLMAssetManifest(visible_assets)
  initial_state_overrides["asset_manifest.json"] = manifest.to_runtime_dict()

  # G2: drop assets_accessed checks whose target asset is not visible this round
  # (graded before its available_from_round, or a hallucinated id). Such checks
  # are impossible to satisfy and unfairly reduce scores.
  checks, dropped_asset_checks = _reconcile_asset_checks(checks, visible_assets)
  if dropped_asset_checks:
    logging.warning(
        "Round %s: dropped %d unsatisfiable assets_accessed check(s): %s",
        round_spec.round_id,
        len(dropped_asset_checks),
        dropped_asset_checks,
    )

  # P1: point `asset ... --id X` solution commands at ids that actually exist in
  # this round's manifest (dedup may have pruned the bare twin the LLM wrote),
  # so replaying the expert trajectory records the assets_accessed effect and the
  # matching check passes.
  reconciled_commands = _reconcile_solution_command_asset_ids(
      solution_commands, visible_assets
  )
  if reconciled_commands != solution_commands:
    logging.info(
        "Round %s: canonicalized asset ids in %d solution command(s).",
        round_spec.round_id,
        sum(1 for a, b in zip(solution_commands, reconciled_commands) if a != b),
    )
    solution_commands = reconciled_commands

  # P1: drop file_check contains/equals checks the expert's own file rewrite
  # contradicts (e.g. asserting a pre-repair value is still present after the
  # round repairs it), which would make the expert trajectory fail its own check.
  checks, dropped_file_checks = _reconcile_file_checks(checks, solution_commands)
  if dropped_file_checks:
    logging.warning(
        "Round %s: dropped %d self-contradictory file_check(s): %s",
        round_spec.round_id,
        len(dropped_file_checks),
        dropped_file_checks,
    )

  # F5: if a decisive photo could not be made to render its token legibly
  # (``fidelity_failed``) and no other legible channel carries the round's
  # ``resolved_value``, the checks asserting that value are no longer fairly
  # satisfiable from the evidence. Relax them to ``role="diagnostic"`` (recorded,
  # non-gating) rather than penalize a correct agent - the user-mandated
  # auto-degrade action for an unreadable decisive image.
  checks, relaxed_fidelity_checks = _reconcile_decisive_fidelity_checks(
      checks, round_spec, visible_assets, spec
  )
  if relaxed_fidelity_checks:
    logging.warning(
        "Round %s: decisive photo unreadable and %r has no redundant legible "
        "carrier; relaxed %d value-asserting check(s) to diagnostic.",
        round_spec.round_id,
        round_spec.resolved_value,
        len(relaxed_fidelity_checks),
    )

  # G1(b): state the graded "inspect the visual evidence" step in the task text.
  round_instruction = _augment_instruction_for_asset_checks(
      round_spec.instruction, checks
  )

  # 4. Add emails.json and tasks.json (default/empty)
  initial_state_overrides["emails.json"] = {
      "replace_seeded": True,
      "emails": [],
  }
  initial_state_overrides["tasks.json"] = {"replace_seeded": True, "tasks": []}

  # 4b. Seed empty lock/deploy protocol state, but ONLY for seeds that expose
  # those families. Gating on ``allowed_tool_families`` keeps the compiled state
  # byte-identical for every existing (non-protocol) scenario, so this is
  # behaviour-neutral until an M1 protocol seed opts in.
  seed_families = set(seed.allowed_tool_families or [])
  if "lock" in seed_families:
    initial_state_overrides["lock.json"] = {"replace_seeded": True, "lock": []}
  if "deploy" in seed_families:
    initial_state_overrides["deploy.json"] = {"replace_seeded": True, "deploy": []}
  if "booking" in seed_families:
    initial_state_overrides["booking.json"] = {
        "replace_seeded": True,
        "booking": [],
    }
  if "pairing" in seed_families:
    initial_state_overrides["pairing.json"] = {
        "replace_seeded": True,
        "pairing": [],
    }
  if "approval" in seed_families:
    initial_state_overrides["approval.json"] = {
        "replace_seeded": True,
        "approval": [],
    }
  if "containment" in seed_families:
    initial_state_overrides["containment.json"] = {
        "replace_seeded": True,
        "containment": [],
    }
  if "review" in seed_families:
    initial_state_overrides["review.json"] = {
        "replace_seeded": True,
        "review": [],
    }

  # 5. Apply previous rounds' updates workspace_adds
  for prev_round_spec in spec.rounds:
    try:
      prev_round_index = int(prev_round_spec.round_id.lstrip("r"))
    except ValueError:
      prev_round_index = 0
    if prev_round_index < round_index:
      # Apply state effects (e.g. file creations) from previous rounds' mock agents
      import shlex
      for prev_mock in prev_round_spec.mock_subagents:
        for effect in prev_mock.state_effects:
          if isinstance(effect, str) and effect.startswith("file create"):
            try:
              parts = shlex.split(effect)
              path_val = None
              content_val = None
              for idx, part in enumerate(parts):
                if part == "--path" and idx + 1 < len(parts):
                  path_val = parts[idx + 1]
                elif part == "--content" and idx + 1 < len(parts):
                  content_val = parts[idx + 1]
              if path_val and content_val:
                rel_path = path_val.lstrip("/").replace("workspace/", "", 1)
                initial_state_overrides[f"workspace/{rel_path}"] = content_val
            except Exception:
              pass

      for prev_update_id in prev_round_spec.update_ids:
        prev_updates = [
            u for u in spec.updates if u.update_id == prev_update_id
        ]
        for pu in prev_updates:
          for add in pu.workspace_adds:
            rel_path = add.get("path")
            content = add.get("content")
            if rel_path and content:
              initial_state_overrides[f"workspace/{rel_path}"] = content

          for app in pu.session_appends:
            session_id = app.get("session_id")
            messages_list = _normalize_append_messages(app)
            if session_id and messages_list:
              session_path = f"sessions/{session_id}.jsonl"
              for s in spec.sessions:
                if s.session_id == session_id:
                  session_path = s.path or session_path
                  break
              current_content = initial_state_overrides.get(session_path, "")
              new_lines = []
              for msg in messages_list:
                new_lines.append(
                    json.dumps(
                        {
                            "message_id": msg.get("message_id", ""),
                            "timestamp": msg.get("timestamp", ""),
                            "sender": msg.get("sender", ""),
                            "text": msg.get("text") or msg.get("content") or "",
                            "attachments": msg.get("attachments", []),
                            "is_redacted": msg.get("is_redacted", False),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
              initial_state_overrides[session_path] = current_content + "".join(new_lines)

          for red in pu.session_redactions:
            for s_path in list(initial_state_overrides.keys()):
              if s_path.startswith("sessions/") and s_path.endswith(".jsonl"):
                lines = initial_state_overrides[s_path].splitlines()
                updated_lines = []
                for line in lines:
                  if not line.strip():
                    continue
                  try:
                    msg_obj = json.loads(line)
                    if msg_obj.get("message_id") == red:
                      msg_obj["is_redacted"] = True
                      msg_obj["text"] = "[Message redacted by compliance]"
                    updated_lines.append(json.dumps(msg_obj, ensure_ascii=False) + "\n")
                  except Exception:
                    updated_lines.append(line + "\n")
                initial_state_overrides[s_path] = "".join(updated_lines)

  # Calculate round starting virtual time to offset background event triggers
  try:
    current_round_index = int(round_spec.round_id.lstrip("r"))
  except ValueError:
    current_round_index = 0

  round_start_time = 0
  for prev_round in spec.rounds:
    try:
      prev_round_index = int(prev_round.round_id.lstrip("r"))
    except ValueError:
      continue
    if prev_round_index < current_round_index:
      for prev_update_id in prev_round.update_ids:
        for u in spec.updates:
          if u.update_id == prev_update_id:
            if u.trigger_virtual_time is not None:
              round_start_time = max(round_start_time, u.trigger_virtual_time)

  background_events = []
  for u in round_updates:
    trigger_time = u.trigger_virtual_time or 0
    trigger_time = max(0, trigger_time - round_start_time)
    for add in u.workspace_adds:
      background_events.append({
          "type": "file_update",
          "trigger_time": trigger_time,
          "path": add.get("path", ""),
          "new_content": add.get("content", ""),
      })
    for app in u.session_appends:
      messages_list = _normalize_append_messages(app)
      for msg in messages_list:
        background_events.append({
            "type": "session_append",
            "trigger_time": trigger_time,
            "session_id": app.get("session_id", ""),
            "message": {
                "message_id": msg.get("message_id", ""),
                "timestamp": msg.get("timestamp", ""),
                "sender": msg.get("sender", ""),
                "text": msg.get("text") or msg.get("content") or "",
                "attachments": msg.get("attachments", []),
                "is_redacted": msg.get("is_redacted", False),
            },
        })
    for red in u.session_redactions:
      background_events.append({
          "type": "session_redaction",
          "trigger_time": trigger_time,
          "message_id": red,
      })
  # B3: files the reference trajectory writes itself - used to suppress
  # conflicting subagent writes that would clobber the expert's own content.
  expert_files = _solution_touched_file_paths(solution_commands)

  # G1: canonical channel registry. The graded channel identity is a session's
  # DISPLAY name ('#fulfillment-ops'), not its storage key
  # ('channel_fulfillment_ops'): eval criteria name the channel the way the prose
  # does, `message search --channel` compares the value literally against what
  # was posted, and the reference solution speaks display names too. Mock reply
  # effects are authored in a separate LLM call against an invented convention
  # (e.g. '#ops-triage'); reconcile them onto that shared display vocabulary so a
  # subagent's reply lands where both the grader and the solver's
  # `message search` can see it, and record each subagent's canonical
  # ``reply_channel`` so the runtime can route its spoken reply.
  session_channels = _session_channel_registry(spec.sessions)
  primary_channel = _primary_round_channel(solution_commands, session_channels)

  mock_subagents = []
  for m in round_mocks:
    # Timing frame: each round runs in its own env with virtual_time starting at
    # 0. Update triggers above are converted to this round-local frame via
    # ``- round_start_time``; a subagent's ``virtual_duration_seconds`` is already
    # a round-local duration (time from delegation, which happens near t=0), so
    # the two event sources share one consistent round-local clock.
    #
    # B3: flatten the LLM's nested/inconsistently-keyed ``state_effects`` into
    # the flat CLI-string / ``file_create`` forms that
    # ``environment._apply_background_event`` actually applies; otherwise the
    # subagent's file writes and command executions are silently dropped and the
    # matching file_check/command_check can never pass. Then drop any file effect
    # that targets a file the expert authors itself (would clobber it).
    flat_effects = _drop_conflicting_subagent_file_effects(
        _flatten_subagent_state_effects(m.state_effects), expert_files
    )
    # G1: reconcile authored channels onto the canonical display-name registry
    # and resolve the channel this subagent replies on (its own effect's
    # channel, else the round's primary coordination channel).
    flat_effects = _reconcile_subagent_channels(flat_effects, session_channels)
    reply_channel = _reply_channel_for(
        flat_effects, session_channels, primary_channel
    )
    background_events.append({
        "type": "subagent_completion",
        "trigger_time": m.virtual_duration_seconds,
        "agent_name": m.agent_role,
        "mock_response": m.mock_response,
        "state_effects": flat_effects,
        "reply_channel": reply_channel,
    })
    mock_subagents.append({
        "agent_role": m.agent_role,
        "mock_response": m.mock_response,
        "virtual_duration_seconds": m.virtual_duration_seconds,
        "state_effects": flat_effects,
        "reply_channel": reply_channel,
    })

  # Per-round record of what the difficulty knobs ACTUALLY did, stamped into
  # ``ground_truth.metadata`` below. This is deliberately recorded rather than
  # logged: whether a knob fired is not recoverable from the shipped artifact
  # (a successfully de-signposted round is textually indistinguishable from one
  # that never signposted), and attributing the round's effect size means
  # splitting "the knob fired here" from "the knob was a no-op here". Written
  # only when a knob is enabled, so seeds that opt out keep byte-identical
  # output.
  knob_record: dict[str, Any] = {}

  # M2 (``force_decoy``): plant a competing reading that an agent which does not
  # actually consult the decisive surface can be taken in by. No checker change:
  # believing the decoy makes the agent report the wrong value, which the
  # round's existing true-value checks already fail. No-op for non-opted seeds.
  if seed.force_decoy:
    _decoy = _inject_decoy(
        mock_subagents, background_events, round_spec, seed.slug
    )
    knob_record["decoy"] = {
        "applied": _decoy is not None,
        "value": _decoy,
        "true_value": round_spec.resolved_value,
    }
    if _decoy is None:
      logging.info(
          "[M2] %s %s: no safe decoy (resolved_value=%r); round left unchanged.",
          seed.slug,
          round_spec.round_id,
          round_spec.resolved_value,
      )
    else:
      logging.info(
          "[M2] %s %s: injected decoy %r against true %r.",
          seed.slug,
          round_spec.round_id,
          _decoy,
          round_spec.resolved_value,
      )

  # M3 (``force_no_signpost``): delete the text that names the decisive asset
  # outright, in BOTH channels -- the sub-agent replies AND the round
  # instruction. The instruction is the dominant one (88.1% of asset-locus
  # rounds name the asset there vs 81.9% in a reply), and rewriting only the
  # replies left the knob a reachable ceiling of 5 rounds in 598. See
  # ``_strip_signposts`` for the rollback rules; it refuses to strip if doing so
  # would break a check the gold relies on, or leave either channel empty.
  if seed.force_no_signpost:
    _locus = (round_spec.resolved_value_locus or "").strip()
    _m3_asset = _locus[len("asset:") :].strip() if _locus.startswith("asset:") else ""
    if not _m3_asset:
      knob_record["no_signpost"] = {
          "applied": False,
          "outcome": "not_asset_locus",
          "asset": "",
          "removed_count": 0,
          "channel": "",
          "instruction_removed_count": 0,
      }
      logging.info(
          "[M3] %s %s: locus %r is not an asset; nothing to de-signpost.",
          seed.slug,
          round_spec.round_id,
          _locus,
      )
    else:
      _m3 = _strip_signposts(
          mock_subagents,
          background_events,
          _m3_asset,
          checks,
          round_instruction,
      )
      # The instruction is consumed below (``instruction=round_instruction``),
      # so the de-signposted text has to be written back here.
      round_instruction = _m3.instruction
      knob_record["no_signpost"] = {
          "applied": _m3.outcome == "stripped",
          "outcome": _m3.outcome,
          "asset": _m3_asset,
          "removed_count": len(_m3.removed),
          # Which channel actually got de-signposted. Recorded because the
          # shipped corpus cannot tell "the instruction was stripped" from "the
          # instruction never named it", and the two mean opposite things when
          # attributing the knob's effect.
          "channel": _m3.channel,
          "instruction_removed_count": _m3.instruction_removed,
      }
      logging.info(
          "[M3] %s %s: %s (%s, channel=%s)",
          seed.slug,
          round_spec.round_id,
          f"stripped {len(_m3.removed)} signpost sentence(s) naming {_m3_asset}"
          f" ({_m3.instruction_removed} from the instruction)"
          if _m3.removed
          else f"NO strip applied for {_m3_asset}",
          _m3.outcome,
          _m3.channel or "none",
      )

  # B3: a time-triggered ``file_update`` conflict event whose path the expert
  # also authors would, once ``wait`` advances the clock, fire after the expert's
  # own edit and clobber it - failing the round's own file_check. Drop those so
  # the expert's write stays authoritative (same policy as for subagent file
  # effects and _reconcile_file_checks).
  background_events = _drop_conflicting_file_update_events(
      background_events, expert_files
  )

  # B3: the expert trajectory must advance virtual time so the time-triggered
  # background effects above actually fire before the round is graded.
  solution_commands = _inject_background_wait(solution_commands, background_events)

  # M8/CL2 (L1+L3): when the seed opts in (``force_async_wait``), grade that the
  # decisive sub-agent's delayed result actually arrived on time -- i.e. the
  # coordinator WAITED for the async completion rather than acting on a stale
  # synchronous value. We add a sender-scoped ``within_window`` check on
  # ``messages_sent`` bound to the slowest mock sub-agent's reply, with the
  # deadline derived from the round's max background trigger. This is the
  # temporal analogue of the M7 vision-blind ablation: a no-wait trajectory never
  # fires the completion, so no matching sender effect exists and the check fails
  # closed (F6). The effect is stamped at the event's ``trigger_time`` by the
  # environment, so the reference trajectory (whose single injected ``wait``
  # advances the clock to ``max_trigger``) is on time by construction, and an
  # over-waiting solver is NOT penalized. No-op for non-opted seeds (E6).
  async_check = _build_async_within_window_check(mock_subagents, background_events)
  if async_check is not None and seed.force_async_wait:
    checks.append(async_check)
    # L2 (runtime arrival gate): the decisive sub-agent's reply must reach the
    # grader ONLY via its DELAYED completion (Route 1, stamped at trigger_time),
    # never via the environment's synchronous Route-2 mirror (which would stamp
    # it at virtual_time=0 the instant a solver DMs the agent). Mark that
    # sub-agent ``async_only`` so ``environment._track_effects`` withholds its
    # synchronous reply until ``wait`` fires the completion event. The decisive
    # role is taken from the check itself (single source of truth for "slowest
    # sub-agent"), so this can never drift from what the window check grades.
    _decisive_role = async_check["expected"]["value"]
    for _m in mock_subagents:
      if str(_m.get("agent_role") or "").strip() == _decisive_role:
        _m["async_only"] = True

  # Order-preserving dedup of evaluation checks. A final-verification re-read of
  # an already-inspected asset scaffolds a byte-identical ``assets_accessed``
  # check (keyed only on the asset id), so ~23% of tasks carried duplicate
  # checks that inflated the all-pass denominator with free passes. Collapse
  # exact duplicates while keeping first-occurrence order; keyed on canonical
  # JSON so dict field order is irrelevant.
  _seen_check_keys: set[str] = set()
  _deduped_checks: list[Any] = []
  for _chk in checks:
    try:
      _key = json.dumps(_chk, sort_keys=True, default=str)
    except (TypeError, ValueError):
      _key = repr(_chk)
    if _key in _seen_check_keys:
      continue
    _seen_check_keys.add(_key)
    _deduped_checks.append(_chk)
  checks = _deduped_checks

  # P2: a ``file_check contains <value>`` scaffolded from a *read* step needs the
  # decisive value to already live in the seeded evidence file. When the LLM
  # populated that source file without it - and no expert command, subagent, or
  # background update writes the file - replaying the reference trajectory fails
  # the check through no fault of the solution. Seed the value into the
  # pre-existing source data file (never fabricating files, never touching code
  # files, and never touching a file the round itself writes: those are graded by
  # the write, tracked in ``reserved_paths``).
  reserved_paths = set(expert_files) | _runtime_written_paths(background_events)
  seeded_source_values = _seed_source_files_for_read_checks(
      checks, initial_state_overrides, reserved_paths
  )
  if seeded_source_values:
    logging.info(
        "Round %s: seeded %d read-evidence value(s) into source files: %s",
        round_spec.round_id,
        len(seeded_source_values),
        seeded_source_values,
    )

  task = Task(
      task_id=task_id,
      instruction=round_instruction,
      canonical_instruction=round_spec.instruction,
      domains=["composite", "conflict"],
      difficulty=int(Difficulty.REASONING),
      initial_state="default",
      template_id="conflict_vlm_mixed_action",
      generator_id="conflict_vlm_v1",
      data=TaskData(
          public={
              "scenario_slug": seed.slug,
              "round_id": round_spec.round_id,
              "round_type": round_spec.round_type,
              "allowed_tool_families": list(seed.allowed_tool_families),
              "required_modalities": list(round_spec.required_modalities),
          },
          private={
              "expected_anchor_ids": list(round_spec.expected_anchor_ids),
              "visible_asset_ids": list(round_spec.visible_asset_ids),
              "background_events": background_events,
              "mock_subagents": mock_subagents,
              "initial_state_overrides": initial_state_overrides,
              "vlm_asset_anchors": {asset.id: list(asset.anchor_ids) for asset in visible_assets},
          },
      ),
      ground_truth=GroundTruth(
          solution_commands=solution_commands,
          evaluation_checks=checks,
          metadata={
              "expected_anchor_ids": list(round_spec.expected_anchor_ids),
              "round_type": round_spec.round_type,
              # Only present when a knob was enabled, so opted-out seeds keep
              # byte-identical ground truth. ``ground_truth`` is never shown to
              # the agent and no checker reads this key, so it cannot leak or
              # change grading -- it exists purely so the post-hoc effect-size
              # analysis reads what happened instead of inferring it.
              **({"difficulty_knobs": knob_record} if knob_record else {}),
          },
      ),
      decision_requirements=["Resolve conflicts dynamically."],
      realism_tags=["conflict_vlm_v1", round_spec.round_type],
  )
  errors = validate_task_solution_commands([task], allowed_families=seed.allowed_tool_families)
  if errors:
    raise ValueError("Invalid Conflict-VLM solution commands:\n" + "\n".join(errors))
  return task


def lint_assets_accessed_satisfiability(task_spec: dict) -> list[str]:
  """Flag ``required`` assets_accessed checks that no agent could ever satisfy.

  The grader (``effect_checker``) matches an expected asset id exactly first,
  and only falls back to the timestamp-stripped base when that base picks out a
  single visible asset -- because collapsing ``<asset>_20241024_085500`` onto a
  stale ``<asset>`` would erase the temporal_staleness distinction the scenario
  exists to test. That makes a check unsatisfiable when its expected id is
  neither a concrete visible asset nor an unambiguous base, so no sequence of
  agent actions can pass it. Such a check silently caps the achievable score,
  which is why this runs at generation time rather than being discovered later
  as an unexplained accuracy ceiling.

  Returns a list of human-readable problems (empty when the task is clean).
  Kept in sync with ``effect_checker`` -- change both together.
  """
  data = task_spec.get("data") or {}
  priv = data.get("private") or task_spec.get("private") or {}
  gt = task_spec.get("ground_truth") or {}
  checks = gt.get("evaluation_checks") or []

  visible = {str(x).lower() for x in (priv.get("visible_asset_ids") or [])}
  manifest = (priv.get("initial_state_overrides") or {}).get("asset_manifest.json") or {}
  if isinstance(manifest, dict):
    visible |= {
        str(a.get("id")).lower()
        for a in (manifest.get("assets") or [])
        if a.get("id")
    }

  problems: list[str] = []
  for idx, check in enumerate(checks):
    if not isinstance(check, dict):
      continue
    if check.get("effect_type") != "assets_accessed":
      continue
    if check.get("role") != "required":
      continue
    needle = str((check.get("expected") or {}).get("value", "")).lower()
    if not needle:
      continue
    if needle in visible:
      continue  # exact match is reachable
    base = _asset_base_key(needle)
    colliding = sorted(a for a in visible if _asset_base_key(a) == base)
    if len(colliding) == 1:
      continue  # unambiguous normalized fallback is reachable
    if not colliding:
      problems.append(
          f"check[{idx}] required assets_accessed '{needle}' matches no visible "
          f"asset (exactly or after normalization); it can never pass"
      )
    else:
      problems.append(
          f"check[{idx}] required assets_accessed '{needle}' is not a visible "
          f"asset id and its normalized base '{base}' is ambiguous across "
          f"{colliding}; the grader refuses to guess, so it can never pass"
      )
  return problems


def _asset_base_key(asset_id: str) -> str:
  """Normalize an asset id for grouping/matching.

  Strips an auto-generated ``_YYYYMMDD_HHMMSS`` timestamp, a ``_r<N>`` round
  suffix, and a trailing ``_chart`` so that a bare id and its timestamped twin
  collapse to the same key. Also folds a doubled platform prefix / path
  separators (e.g. ``workspace_workspace/flight_schedule`` or
  ``workspace/flight_schedule``) that older data may carry, so those legacy
  phantom ids collapse to the same key as the clean ``workspace_flight_schedule``
  the generator now emits. Kept in sync with
  ``effect_checker._normalize_asset_base`` so compile-time reconciliation and
  runtime grading agree.
  """
  import re

  s = re.sub(r"_\d{8}_\d{6}", "", asset_id)
  s = re.sub(r"_r\d+", "", s)
  s = re.sub(r"^([^/_]+)_\1/", r"\1_", s)  # workspace_workspace/ -> workspace_
  s = s.replace("/", "_")  # any residual path separator -> '_'
  if s.endswith("_chart"):
    s = s[:-6]
  return s.lower()


def _has_timestamp_suffix(asset_id: str) -> bool:
  import re

  return bool(re.search(r"_\d{8}_\d{6}", asset_id))


# Fields carrying an asset's OWN substantive visual content. Used to tell a
# real, separately authored visual apart from a placeholder bare id. Identity
# and timing fields (id, path, available_from_round) are deliberately excluded:
# those differ between genuine twins by construction, so including them would
# make every twin look "distinct" and disable the deferral entirely.
_VISIBILITY_CONTENT_FIELDS = (
    "ocr_text", "caption", "description", "chart_rows", "render_code",
    "regions", "prompt", "content_text",
)


def _asset_content_fingerprint(asset: Any) -> str:
  """Digest of an asset's own visual payload; "" when it carries none.

  An empty result means the object is a bare reference with no authored
  content -- the placeholder case ``_effective_available_from_round`` exists to
  defer. Anything non-empty is content a solver could actually read.
  """
  import json as _json

  parts = []
  for name in _VISIBILITY_CONTENT_FIELDS:
    value = getattr(asset, name, None)
    if not value:
      continue
    parts.append(f"{name}={_json.dumps(value, sort_keys=True, default=str)}")
  return "\n".join(parts)


def _effective_available_from_round(asset: Any, all_assets: list[Any]) -> int:
  """Availability round used for visibility, correcting untrusted bare ids.

  A bare (non-timestamped) asset id can carry an unreliable
  ``available_from_round`` (often 1) even though the real, timestamped snapshots
  of the same visual first appear in a later round. Grading an
  ``assets_accessed`` check against such a bare id before any real snapshot
  exists is impossible and unfairly reduces scores, so a bare PLACEHOLDER defers
  to the earliest snapshot round among its timestamped siblings.

  The deferral is restricted to actual placeholders. A bare asset that carries
  its own authored payload, different from every timestamped sibling's, is a
  real independently rendered visual rather than a stand-in for one, and keeps
  its declared round. Deferring those was a defect, not a safeguard: it hid the
  asset from the very round whose answer lived in it, producing rounds no
  solver could win while gold still passed (gold is authored from
  ``resolved_value``, so it writes the answer without reading anything). Across
  the v8 and sonnet-5 corpora that accounted for all 27 unwinnable rounds --
  0/162 solver successes on them versus 19.1% elsewhere -- and every one of the
  112 assets this rule deferred had payload distinct from its siblings, i.e.
  the deferral never once fired on the placeholder it was written for.

  Redundant twins and ungradable ids are handled downstream where they belong,
  by ``_dedup_visible_assets`` and ``_reconcile_asset_checks``; hiding the asset
  is the wrong lever for either.
  """
  own = int(getattr(asset, "available_from_round", 1))
  asset_id = str(getattr(asset, "id", ""))
  if _has_timestamp_suffix(asset_id):
    return own
  base = _asset_base_key(asset_id)
  siblings = [
      a
      for a in all_assets
      if a is not asset
      and _has_timestamp_suffix(str(getattr(a, "id", "")))
      and _asset_base_key(str(getattr(a, "id", ""))) == base
  ]
  if not siblings:
    return own
  own_content = _asset_content_fingerprint(asset)
  if own_content and all(
      _asset_content_fingerprint(s) != own_content for s in siblings
  ):
    return own  # a real visual of its own, not a stand-in for a later snapshot
  return min(int(getattr(a, "available_from_round", 1)) for a in siblings)


def _dedup_visible_assets(visible_assets: list[Any]) -> list[Any]:
  """G3: drop bare (un-timestamped) twins when a timestamped sibling exists.

  The LLM frequently emits two asset specs for a single snapshot: a bare id
  (e.g. ``devops_dashboard_db_connections_chart``) and a fully timestamped id
  (``devops_dashboard_db_connections_chart_20231027_101500``). Listing both
  makes the agent waste steps guessing which to open. Timestamped members are
  always kept (so genuine time series like 10:15 vs 10:18 survive); only bare
  duplicates that have a timestamped sibling in the same visible set are dropped.
  """
  groups: dict[str, list[Any]] = {}
  for asset in visible_assets:
    groups.setdefault(_asset_base_key(asset.id), []).append(asset)

  kept: list[Any] = []
  for asset in visible_assets:
    group = groups[_asset_base_key(asset.id)]
    has_timestamped_sibling = any(_has_timestamp_suffix(m.id) for m in group)
    if (
        len(group) > 1
        and not _has_timestamp_suffix(asset.id)
        and has_timestamped_sibling
    ):
      continue  # redundant bare duplicate of a timestamped snapshot
    kept.append(asset)
  return kept


def _canonical_visible_asset_map(visible_assets: list[Any]) -> dict[str, str]:
  """Map each visible asset's base key to a single canonical (real) id.

  After ``_dedup_visible_assets`` prunes bare twins, a check or solution command
  written against the bare id names something absent from the manifest. This map
  resolves a base key to the surviving id that the agent can actually open,
  preferring the latest snapshot (highest ``available_from_round``, then the
  lexically-greatest id, i.e. the newest timestamp) for determinism.
  """
  best: dict[str, Any] = {}
  for asset in visible_assets:
    key = _asset_base_key(asset.id)
    cur = best.get(key)
    if cur is None or (
        int(getattr(asset, "available_from_round", 1)),
        str(asset.id),
    ) > (
        int(getattr(cur, "available_from_round", 1)),
        str(cur.id),
    ):
      best[key] = asset
  return {key: str(asset.id) for key, asset in best.items()}


def _reconcile_asset_command_id(
    cmd: str, visible_exact: set[str], canonical: dict[str, str]
) -> tuple[str | None, bool]:
  """Reconcile the ``--id`` of a single ``asset <sub> --id X`` command string.

  Shared by ``_reconcile_solution_command_asset_ids`` (solution/expert commands)
  and ``_reconcile_asset_checks`` (``command_check`` entries) so a command and the
  check that grades it always make the *same* decision about the same id.

  Returns ``(new_cmd, changed)``:
    * ``(cmd, False)`` - not an ``asset {open,info,read-text,describe} --id X``
      command, unparseable, or the id already names a visible manifest asset.
    * ``(rewritten, True)`` - the exact id is absent (its bare twin was pruned by
      ``_dedup_visible_assets``) but its base key resolves to a canonical visible
      id; the id is rewritten to that canonical id (P1).
    * ``(None, True)`` - the id names no asset visible this round at all (not even
      by base key): the M14 dangling read. Caller should drop it (P1b).
  """
  import shlex

  if not isinstance(cmd, str) or not cmd.strip().startswith("asset "):
    return cmd, False
  try:
    parts = shlex.split(cmd)
  except ValueError:
    return cmd, False
  asset_subcommands = {"open", "info", "read-text", "describe"}
  if len(parts) < 2 or parts[0] != "asset" or parts[1] not in asset_subcommands:
    return cmd, False
  for idx, part in enumerate(parts):
    if part == "--id" and idx + 1 < len(parts):
      cur_id = parts[idx + 1]
      if cur_id.lower() in visible_exact:
        return cmd, False  # already a real manifest id
      base_key = _asset_base_key(cur_id)
      if base_key in canonical:
        if canonical[base_key] != cur_id:
          parts[idx + 1] = canonical[base_key]
          return shlex.join(parts), True
        return cmd, False
      # P1b: names no asset visible this round -> dangling read, drop.
      return None, True
  return cmd, False


def _reconcile_asset_checks(
    checks: list[dict[str, Any]], visible_assets: list[Any]
) -> tuple[list[dict[str, Any]], list[str]]:
  """G2/P1: drop unsatisfiable ``assets_accessed``/``command_check`` and canonicalize ids.

  Adjustments so a graded id always names something in the manifest:
    * ``assets_accessed`` effect checks: drop when the target asset is not visible
      this round (graded before its ``available_from_round`` or hallucinated) -
      impossible to satisfy; else rewrite a pruned bare-twin id to the canonical
      visible id of the same base key. Matching is timestamp/``_chart`` agnostic,
      kept in sync with ``effect_checker._normalize_asset_base``.
    * ``command_check`` entries whose command is ``asset <sub> --id X``: apply the
      *same* per-command reconciliation as the solution/expert trajectory
      (``_reconcile_asset_command_id``). When P1b drops the dangling ``asset open``
      from the trajectory, the paired ``command_check`` would otherwise be an
      orphan that can never pass (the command is never issued), so it is dropped
      too; a canonicalized command id is rewritten in lock-step.
  """
  visible_exact = {str(a.id).lower() for a in visible_assets}
  visible_keys = {_asset_base_key(a.id) for a in visible_assets}
  canonical = _canonical_visible_asset_map(visible_assets)

  reconciled: list[dict[str, Any]] = []
  dropped: list[str] = []
  for check in checks:
    if isinstance(check, dict) and check.get("effect_type") == "assets_accessed":
      expected = check.get("expected") or {}
      if isinstance(expected, dict) and expected.get("field") == "id":
        asset_id = str(expected.get("value", ""))
        base_key = _asset_base_key(asset_id)
        if asset_id.lower() not in visible_exact and base_key not in visible_keys:
          dropped.append(asset_id)
          continue
        # Surviving check whose exact id is not a manifest id (bare twin pruned):
        # point it at the canonical visible id so the agent can open it.
        if asset_id.lower() not in visible_exact and base_key in canonical:
          new_id = canonical[base_key]
          if new_id != asset_id:
            check = {**check, "expected": {**expected, "value": new_id}}
    elif isinstance(check, dict) and check.get("type") == "command_check":
      cmd = check.get("command")
      if isinstance(cmd, str) and cmd.strip().startswith("asset "):
        new_cmd, changed = _reconcile_asset_command_id(cmd, visible_exact, canonical)
        if new_cmd is None:
          # P1b residue: the paired dangling `asset ... --id X` command was dropped
          # from the trajectory, so this command_check can never pass. Drop it.
          dropped.append(cmd)
          continue
        if changed:
          check = {**check, "command": new_cmd}
    reconciled.append(check)
  return reconciled, dropped


def _reconcile_solution_command_asset_ids(
    solution_commands: list[str], visible_assets: list[Any]
) -> list[str]:
  """P1: rewrite ``asset <sub> --id X`` ids to a canonical visible id.

  When ``_dedup_visible_assets`` prunes a bare twin, a solution command that
  opens the bare id hits "Asset not found", records no ``assets_accessed``
  effect, and the corresponding check fails even though the visual is on screen.
  Rewrite such ids to the surviving canonical id (same base key) so the command
  succeeds. Ids already present in the manifest are left untouched.

  P1b: if the id resolves to no asset visible this round at all (not even by base
  key) - the M14 residue where a bare/phantom twin's real snapshot only appears
  in a later round and ``_effective_available_from_round`` has hidden it here -
  the pure-read command is a guaranteed "Asset not found" no-op whose matching
  ``assets_accessed`` and ``command_check`` checks are dropped in lock-step by
  ``_reconcile_asset_checks`` (both share ``_reconcile_asset_command_id``). Drop
  it so the reference trajectory references only assets that actually exist this
  round. A guard restores the originals in the pathological case where every
  command was such a dangling read.
  """
  visible_exact = {str(a.id).lower() for a in visible_assets}
  canonical = _canonical_visible_asset_map(visible_assets)

  out: list[str] = []
  dropped_dangling: list[str] = []
  for cmd in solution_commands:
    new_cmd, _changed = _reconcile_asset_command_id(cmd, visible_exact, canonical)
    if new_cmd is None:
      dropped_dangling.append(cmd)
      continue
    out.append(new_cmd)
  # Guard: never let dangling-drop empty the trajectory. If every command was a
  # dangling read (pathological), keep the originals rather than emit nothing.
  if not out and dropped_dangling:
    return list(solution_commands)
  return out


def _norm_file_path(path: str) -> str:
  """Normalize a workspace-relative file path for check/command matching.

  Strips a leading ``/`` and a ``workspace/`` prefix so a ``file_check`` path
  (``config.json`` -> ``/config.json``) and a solution command's
  ``--path config.json`` / ``--path workspace/config.json`` compare equal.
  """
  import posixpath

  p = str(path).strip().lstrip("/")
  if p.startswith("workspace/"):
    p = p[len("workspace/") :]
  return posixpath.normpath(p) if p else p


def _solution_authored_files(solution_commands: list[str]) -> dict[str, str]:
  """Map normalized path -> final content the expert writes via ``file create``.

  Last write wins, mirroring how replaying the trajectory leaves the file.
  """
  import shlex

  authored: dict[str, str] = {}
  for cmd in solution_commands:
    if not isinstance(cmd, str) or "file create" not in cmd:
      continue
    try:
      parts = shlex.split(cmd)
    except ValueError:
      continue
    if len(parts) < 2 or parts[0] != "file" or parts[1] != "create":
      continue
    path_val = None
    content_val = None
    for idx, part in enumerate(parts):
      if part == "--path" and idx + 1 < len(parts):
        path_val = parts[idx + 1]
      elif part == "--content" and idx + 1 < len(parts):
        content_val = parts[idx + 1]
    if path_val is not None and content_val is not None:
      authored[_norm_file_path(path_val)] = content_val
  return authored


def _reconcile_file_checks(
    checks: list[dict[str, Any]], solution_commands: list[str]
) -> tuple[list[dict[str, Any]], list[str]]:
  """P1: drop ``file_check`` contains/equals checks the expert edit contradicts.

  The LLM sometimes emits a ``file_check contains <OLD>`` against a file the task
  requires the agent to rewrite (e.g. asserting a broken config value is still
  present after the round explicitly repairs it - a self-contradictory pair with
  the ``contains <NEW>`` check). Replaying the expert trajectory can never pass
  such a check. For any file the expert explicitly (re)writes, evaluate its
  contains/equals checks against that authored content and drop the ones it
  fails. Files the expert does not write are left untouched (they may be produced
  at runtime by subagents), and non-content conditions are never dropped here.
  """
  authored = _solution_authored_files(solution_commands)
  if not authored:
    return checks, []

  reconciled: list[dict[str, Any]] = []
  dropped: list[str] = []
  for check in checks:
    if isinstance(check, dict) and check.get("type") in {"file_check", "file"}:
      norm = _norm_file_path(check.get("path", ""))
      if norm in authored:
        condition = check.get("condition")
        expected = check.get("expected")
        if not condition:
          if "contains" in check:
            condition, expected = "contains", check["contains"]
          elif "equals" in check:
            condition, expected = "equals", check["equals"]
          else:
            condition = "exists"
        content = authored[norm]
        if condition == "contains" and str(expected) not in content:
          dropped.append(f"{check.get('path')} contains {expected!r}")
          continue
        if condition == "equals" and content != expected:
          dropped.append(f"{check.get('path')} equals (mismatch)")
          continue
    reconciled.append(check)
  return reconciled, dropped


def _check_asserts_value(check: dict[str, Any], value_lc: str) -> bool:
  """True when a check's serialized content asserts on ``value_lc`` (lowercase).

  A broad substring match over the check's JSON form: an ``expected.value``, a
  ``file_check contains``/``equals``, a ``command_check`` command, or an effect
  payload that carries the decisive value all serialize to a blob that contains
  it. Intentionally liberal - it only ever runs in the already-degraded
  sole-carrier branch below, where relaxing anything that references the now
  unreadable value is the safe direction.
  """
  if not isinstance(check, dict) or not value_lc:
    return False
  try:
    blob = json.dumps(check, ensure_ascii=False, default=str).lower()
  except (TypeError, ValueError):
    blob = str(check).lower()
  return value_lc in blob


def _resolved_value_has_readable_carrier(
    resolved: str,
    failed_ids: set[str],
    visible_assets: list[Any],
    spec: ConflictVLMScenarioSpec,
) -> bool:
  """True when ``resolved`` is legibly obtainable from a channel other than the failed photo(s).

  The decisive value is guaranteed to appear in at least the failed decisive
  asset's grounded text (that is how ``_tag_decisive_assets`` tagged it). This
  asks whether a SECOND, reliably legible source also carries it, in which case
  an agent can still satisfy the value-asserting checks fairly and they must NOT
  be relaxed. Reliable sources at runtime are:

    * a workspace file's content (plain text the agent can ``file read``);
    * a session message's text (delivered as JSONL);
    * a visible NON-image asset (chart/rich_text/diagram/table/doc) whose grounded
      text carries the value - these render their text from that same text, so it
      is normally legible in pixels, unlike a generative ``natural_image``/
      ``sketch``. "Normally" is not "always" (an LLM-composed page can drop the
      value, and a layout can clip or overflow it), which is why such assets are
      now fidelity-audited too; a chart that FAILED that audit is excluded by the
      ``fidelity_failed`` guard below just like a failed photo;
    * a visible decisive image asset whose fidelity audit PASSED for this exact
      token (audit success means the pixels are legible).

  Other fidelity-failed assets and untagged/corroborating generative images are
  never treated as carriers - their text may be illegible.
  """
  resolved_lc = resolved.lower()

  for f in spec.workspace.files:
    if resolved_lc in str(f.get("content") or "").lower():
      return True

  for session in spec.sessions:
    for msg in session.messages:
      if resolved_lc in str(getattr(msg, "text", "") or "").lower():
        return True

  for asset in visible_assets:
    if asset.id in failed_ids:
      continue
    pm = asset.provider_metadata or {}
    if pm.get("fidelity_failed"):
      continue  # another degraded photo - not a reliable source
    is_image = asset.kind in IMAGE_KINDS or asset.generator in IMAGE_GENERATORS
    if is_image:
      # Only an audit-passed decisive photo of the same token is legible enough
      # to count; corroborating/untagged generative images may not be.
      if (
          pm.get("evidence_role") == "decisive"
          and str(pm.get("expected_token") or "").strip().lower() == resolved_lc
      ):
        return True
      continue
    grounded = (
        f"{asset.caption or ''}\n{asset.ocr_text or ''}\n{asset.content_text or ''}"
    ).lower()
    if resolved_lc in grounded:
      return True

  return False


def _reconcile_decisive_fidelity_checks(
    checks: list[dict[str, Any]],
    round_spec: ConflictVLMRoundSpec,
    visible_assets: list[Any],
    spec: ConflictVLMScenarioSpec,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  """Relax value-asserting checks a now-illegible decisive photo left unsatisfiable.

  When ``_maybe_audit_and_fix_fidelity`` cannot make a decisive photo render its
  ``expected_token`` legibly it flags the asset ``fidelity_failed`` and downgrades
  it to ``corroborating`` (the user-mandated failure action: auto-degrade, never
  fail the build). If that photo was the SOLE legible carrier of the round's
  ``resolved_value``, then any check that asserts that value - the
  ``field_semantic_match`` decisive answer, a ``file_check contains``, a matching
  ``command_check`` - can no longer be satisfied from the evidence, so grading it
  would unfairly penalize a correct agent. Such checks are tagged
  ``role="diagnostic"`` (recorded, non-gating, non-scored) exactly like the
  asset-access relaxation, rather than dropped, so the signal is preserved.

  A no-op unless (a) the round has a non-empty ``resolved_value``, and (b) a
  visible decisive image asset for that exact token is flagged ``fidelity_failed``,
  and (c) no redundant legible carrier exists
  (``_resolved_value_has_readable_carrier``). Returns ``(reconciled, relaxed)``.
  """
  resolved = (round_spec.resolved_value or "").strip()
  if not resolved:
    return checks, []
  resolved_lc = resolved.lower()

  failed_ids = {
      asset.id
      for asset in visible_assets
      if (asset.provider_metadata or {}).get("fidelity_failed")
      and str((asset.provider_metadata or {}).get("expected_token") or "")
      .strip()
      .lower()
      == resolved_lc
  }
  if not failed_ids:
    return checks, []

  if _resolved_value_has_readable_carrier(
      resolved, failed_ids, visible_assets, spec
  ):
    return checks, []

  reconciled: list[dict[str, Any]] = []
  relaxed: list[dict[str, Any]] = []
  for check in checks:
    if (
        isinstance(check, dict)
        and check.get("role") != "diagnostic"
        and _check_asserts_value(check, resolved_lc)
    ):
      reconciled.append({**check, "role": "diagnostic"})
      relaxed.append(check)
    else:
      reconciled.append(check)
  return reconciled, relaxed


def _runtime_written_paths(background_events: list[dict[str, Any]]) -> set[str]:
  """Normalized paths written at runtime by background updates or subagents.

  Combined with the expert's own writes (``_solution_touched_file_paths``), these
  mark files whose final content the round itself produces (a time-triggered
  ``file_update``, or a subagent ``file_create`` / flat CLI ``file <verb>``
  effect). The P2 source-seeding pass must not pre-fill such a file: the write is
  what its check grades, and pre-seeding would both defeat that and reveal the
  value before the producing event fires.
  """
  import shlex

  written: set[str] = set()
  for ev in background_events:
    if not isinstance(ev, dict):
      continue
    if ev.get("type") in {"file_update", "file_create"} and ev.get("path"):
      written.add(_norm_file_path(str(ev.get("path", ""))))
    for eff in ev.get("state_effects") or []:
      if isinstance(eff, dict):
        if eff.get("path"):
          written.add(_norm_file_path(str(eff.get("path", ""))))
      elif isinstance(eff, str) and eff.strip().startswith("file "):
        try:
          parts = shlex.split(eff)
        except ValueError:
          continue
        for idx, part in enumerate(parts):
          if part == "--path" and idx + 1 < len(parts):
            written.add(_norm_file_path(parts[idx + 1]))
  return written


def _seed_source_files_for_read_checks(
    checks: list[dict[str, Any]],
    initial_state_overrides: dict[str, Any],
    reserved_paths: set[str],
) -> list[str]:
  """P2: seed a decisive ``contains`` value into the source file a check reads.

  A ``file_check contains <value>`` scaffolded from a *read* step asserts the
  agent surfaces a decisive value that is supposed to already live in a seeded
  evidence file. When the LLM populated that source file without the value (and
  nothing in the round writes it), replaying the reference trajectory can never
  make the read reveal it, so the check fails through no fault of the solution.
  Amend the *pre-existing* source data file so the value is present.

  Strictly conservative - only touches a path that:
    (a) already exists as a ``workspace/`` text file in the overrides (never
        fabricates a new file, so it cannot invent a file the task forbids);
    (b) is a data / report / config file, never a source/script file
        (``_is_code_file`` - a decisive value does not belong inside code);
    (c) is not written by the expert, a subagent, or a background update
        (``reserved_paths`` - those files are graded by the write, not the seed,
        and pre-seeding would defeat the grade); and
    (d) does not already contain the value.

  Returns human-readable descriptions of the seeds applied (for logging).
  """
  # Map normalized path -> the "workspace/..." override key holding its text.
  source_text_files: dict[str, str] = {}
  for key, val in initial_state_overrides.items():
    if isinstance(key, str) and key.startswith("workspace/") and isinstance(
        val, str
    ):
      source_text_files[_norm_file_path(key)] = key

  seeded: list[str] = []
  for check in checks:
    if not (isinstance(check, dict) and check.get("type") in {"file_check", "file"}):
      continue
    # Extract a 'contains' expectation (direct key or condition/expected form).
    if "contains" in check:
      expected = check.get("contains")
    elif check.get("condition") == "contains":
      expected = check.get("expected")
    else:
      continue
    value = str(expected if expected is not None else "").strip()
    if not value:
      continue
    norm = _norm_file_path(check.get("path", ""))
    if not norm or norm in reserved_paths or _is_code_file(norm):
      continue
    key = source_text_files.get(norm)
    if key is None:  # not a pre-existing source text file -> never fabricate
      continue
    content = initial_state_overrides[key]
    if value in content:
      continue
    initial_state_overrides[key] = (
        content.rstrip() + f"\n\n[System note] Authoritative record: {value}\n"
    )
    # Reflect the amended content locally so a second check on the same file
    # sees it (and a repeat value becomes a no-op above).
    source_text_files[norm] = key
    seeded.append(f"{check.get('path')} += {value!r}")
  return seeded


_SUBAGENT_FILE_EFFECT_TYPES = frozenset(
    {"file_write", "file_create", "write_file", "create_file"}
)
_SUBAGENT_COMMAND_EFFECT_TYPES = frozenset({
    "command_execution",
    "command_executed",
    "execute_command",
    "command_check_mock",
    "command_check",
    "run_command",
    "command",
})
# Keys whose values hold further nested effect lists/dicts to recurse into.
_SUBAGENT_NESTED_EFFECT_KEYS = ("state_effects", "effects", "actions")


def _as_effect_list(value: Any) -> list[Any]:
  if value is None:
    return []
  if isinstance(value, list):
    return value
  return [value]


def _flatten_subagent_state_effects(effects: Any, _depth: int = 0) -> list[Any]:
  """Normalize LLM-authored subagent ``state_effects`` to runtime forms.

  ``environment._apply_background_event`` only applies two effect shapes for a
  ``subagent_completion`` event: a raw CLI ``str`` (executed and recorded in
  ``command_history``) and a dict with ``type`` in
  ``{file_write, file_create, write_file}`` (written to the virtual workspace).

  The generator's LLM instead emits deeply nested, inconsistently-keyed dicts,
  e.g.::

      {"subagent_name": "system_admin",
       "trigger_conditions": {"message_contains": ["restart"]},
       "state_effects": [
           {"effect_type": "file_write", "path": "config/x.json", "content": ...},
           {"effect_type": "command_execution", "command": "system gateway restart"}]}

  which the runtime silently drops. Every file the subagent was meant to write
  and every command it was meant to run then never happens, so the matching
  ``file_check`` / ``command_check`` can never pass - not even for the reference
  solution. Recursively flatten those structures into the two supported forms so
  the intended effects actually fire. Command effects become raw CLI strings;
  file effects become ``{"type": "file_create", ...}`` dicts. Duplicate effects
  (the LLM often emits several trigger variants of one action) are collapsed
  while preserving order.
  """
  out: list[Any] = []
  if _depth > 6:
    return out
  for eff in _as_effect_list(effects):
    if isinstance(eff, str):
      s = eff.strip()
      if s:
        out.append(s)
      continue
    if not isinstance(eff, dict):
      continue
    kind = str(eff.get("type") or eff.get("effect_type") or "").strip().lower()
    path = eff.get("path") or eff.get("file") or eff.get("filename")
    command = eff.get("command") or eff.get("cmd")

    if kind in _SUBAGENT_FILE_EFFECT_TYPES and path:
      content = eff.get("content", eff.get("new_content", ""))
      if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
      out.append(
          {"type": "file_create", "path": str(path), "content": content}
      )
      continue
    if command and (kind in _SUBAGENT_COMMAND_EFFECT_TYPES or not kind):
      if isinstance(command, str) and command.strip():
        out.append(command.strip())
      continue

    # Container-only dict (a nested subagent definition): recurse into any
    # nested effect holders and dict-shaped nested state.
    for key in _SUBAGENT_NESTED_EFFECT_KEYS:
      if key in eff:
        out.extend(_flatten_subagent_state_effects(eff[key], _depth + 1))
    for cmd in _as_effect_list(eff.get("commands_executed")) + _as_effect_list(
        eff.get("commands")
    ):
      if isinstance(cmd, str) and cmd.strip():
        out.append(cmd.strip())
      elif isinstance(cmd, dict):
        out.extend(_flatten_subagent_state_effects(cmd, _depth + 1))
    for fw in _as_effect_list(eff.get("files_written")) + _as_effect_list(
        eff.get("files_created")
    ):
      if isinstance(fw, dict) and (fw.get("path") or fw.get("file")):
        fp = fw.get("path") or fw.get("file")
        fc = fw.get("content", fw.get("new_content", ""))
        if not isinstance(fc, str):
          fc = json.dumps(fc, ensure_ascii=False)
        out.append({"type": "file_create", "path": str(fp), "content": fc})

  # Collapse duplicates while preserving order.
  seen: set[str] = set()
  deduped: list[Any] = []
  for eff in out:
    key = eff if isinstance(eff, str) else json.dumps(eff, sort_keys=True)
    if key in seen:
      continue
    seen.add(key)
    deduped.append(eff)
  return deduped


_CHANNEL_FLAG_RE = re.compile(r"(--channel\s+)('[^']*'|\"[^\"]*\"|\S+)")
_CHANNEL_STOP_TOKENS = frozenset({"channel", "chan", "the", "a", "an"})


def _channel_slug_tokens(raw: str) -> list[str]:
  """Split a channel-ish string into lowercase alphanumeric tokens.

  ``'#ops-triage'`` -> ``['ops', 'triage']``;
  ``'operations_triage_channel'`` -> ``['operations', 'triage']``.
  """
  s = str(raw or "").strip().lower().lstrip("#")
  parts = re.split(r"[^a-z0-9]+", s)
  return [p for p in parts if p and p not in _CHANNEL_STOP_TOKENS]


# Storage prefixes stack in practice ('session_channel_ops_coordination'), so
# strip them repeatedly rather than one deep.
_SESSION_ID_CHANNEL_PREFIX_RE = re.compile(r"^(?:channel_|session_)+")


def _session_display_channel(session: Any) -> str:
  """The human-readable channel name a shipped session is known by.

  A ``ScenarioSession`` carries TWO identifiers and they are not interchangeable:

    * ``session_id`` -- the runtime storage key (``sessions/<session_id>.jsonl``).
    * ``channel``    -- the display name humans (and the graded criteria) use.

  The DISPLAY name is the graded channel identity: an eval criterion names the
  channel the way the scenario prose does (``#fulfillment-ops``), the runtime's
  ``message search --channel`` compares the value LITERALLY against what was
  posted, and the reference solution's own ``message send --channel`` overwhelm-
  ingly speaks display names. Normalizing to ``session_id`` instead would rewrite
  every channel into a token no participant ever utters.

  Falls back to deriving a display name from ``session_id`` when the session
  declares none (``ScenarioSession.channel`` defaults to the literal
  ``"default"``): drop a leading ``channel_``/``session_`` storage prefix, swap
  underscores for hyphens, and prefix ``#``.

  Accepts either a ``ScenarioSession`` or the raw ``layer1b`` session dict it is
  built from, because the generator needs this same vocabulary one stage EARLIER
  than materialization -- while the round's author can still be told which
  channel names are real (see ``conflict_llm_pipeline`` round generation). Both
  carry the same two keys, so one accessor serves both and the two stages cannot
  drift onto different registries.
  """
  if isinstance(session, dict):
    # Route through the shared accessor, never ``session["channel"]``: the raw
    # layer1b dict may declare its name under 'channel_name' or 'name', and
    # reading only 'channel' here is precisely the divergence that made this
    # gate enforce a session_id-derived vocabulary against the real one (see
    # ``spec.session_channel_value``).
    raw = session_channel_value(session).strip()
    sid = str(session.get("session_id") or "").strip()
  else:
    raw = str(getattr(session, "channel", "") or "").strip()
    sid = str(getattr(session, "session_id", "") or "").strip()
  if raw and raw.lower() != "default":
    return raw if raw.startswith("#") else f"#{raw}"
  # A session_id that already carries the '#' would otherwise be re-prefixed
  # into '##incident-ops' -- a name no participant ever utters, which then gets
  # stamped onto every reconciled sub-agent reply. Observed twice in the v5
  # corpus, so strip it before prefixing rather than trusting the id's shape.
  sid = sid.lstrip("#").strip()
  if not sid:
    return ""
  slug = _SESSION_ID_CHANNEL_PREFIX_RE.sub("", sid).replace("_", "-").strip("-")
  return f"#{slug}" if slug else ""


def _session_channel_registry(sessions: list[Any]) -> list[str]:
  """Ordered, de-duplicated display-name registry for a scenario's sessions.

  Two sessions may legitimately share one display name (a single real channel
  transcribed as two time-sliced session files); they collapse to one registry
  entry rather than being forced apart, because they ARE the same channel.
  """
  out: list[str] = []
  seen: set[str] = set()
  for s in sessions or []:
    name = _session_display_channel(s)
    if name and name not in seen:
      seen.add(name)
      out.append(name)
  return out


def update_channel_registry(updates: list[Any]) -> list[str]:
  """Channels the scenario's async updates actually post into.

  A SECOND, independent witness of the scenario's channel vocabulary, for the
  case where the first one is missing. ``coordinated_containment_v02`` ships a
  single degenerate session (``session_id: ""``, ``channel: "default"``), so
  :func:`_session_channel_registry` returns nothing for it -- yet its updates
  append into ``#incident-inc7921-war-room`` and
  ``#secops-endpoint-coordination`` in all ten rounds, and its gold speaks those
  two names. The names are there; the sessions list just fails to declare them.

  Deliberately NOT merged into the session registry: on the v6 corpus updates
  introduce a name sessions lack in exactly one scenario -- that one -- so a
  blanket union would buy nothing and could only weaken the vocabulary by
  admitting an update's own invented name as if it were canonical. Callers use
  this strictly as a fallback when the session vocabulary is empty.
  """
  out: list[str] = []
  seen: set[str] = set()
  for upd in updates or []:
    appends = (
        upd.get("session_appends")
        if isinstance(upd, dict)
        else getattr(upd, "session_appends", None)
    )
    for app in appends or []:
      raw = (
          app.get("channel_or_thread")
          if isinstance(app, dict)
          else getattr(app, "channel_or_thread", "")
      )
      name = str(raw or "").strip()
      if not name or name.lower() == "default":
        continue
      name = name if name.startswith("#") else f"#{name.lstrip('#')}"
      if name not in seen:
        seen.add(name)
        out.append(name)
  return out


def update_channel_aliases(updates: list[Any]) -> dict[str, str]:
  """Provable aliases for :func:`update_channel_registry`, keyed casefolded.

  Narrower than :func:`_session_channel_aliases` by construction: an update
  names its channel directly, so there is no ``session_id`` spelling to also
  accept -- only the name itself and the same name without its ``#``.
  """
  alias: dict[str, str] = {}
  for display in update_channel_registry(updates):
    for key in (display, display.lstrip("#")):
      k = key.strip().lower()
      if k and k not in alias:
        alias[k] = display
  return alias


def _session_channel_aliases(sessions: list[Any]) -> dict[str, str]:
  """Aliases that PROVABLY denote a shipped channel, keyed casefolded.

  Distinct from :func:`_match_canonical_channel`, which scores token overlap and
  will happily map ``#prod-cutover-war-room`` onto ``#war-room``. That guess is
  tolerable for ambient sub-agent chatter, but not for the reference solution:
  merging two channels the scenario deliberately keeps apart would silently
  rewrite what the gold trajectory did. So this admits only forms that ARE the
  same channel by construction:

    * the display name itself (``#ops-triage``);
    * the ``session_id`` that derived it (``channel_ops_triage``) -- the storage
      key of that exact session file;
    * the display name without its ``#`` (``ops-triage``), the single most
      common way an author writes it.

  Anything else is a genuine authoring choice and is left for the author to
  correct, not for this function to guess.
  """
  alias: dict[str, str] = {}
  for s in sessions or []:
    display = _session_display_channel(s)
    if not display:
      continue
    sid = (
        str(s.get("session_id") or "")
        if isinstance(s, dict)
        else str(getattr(s, "session_id", "") or "")
    ).strip()
    # First writer wins, matching the registry's own de-duplication order, so
    # two time-sliced session files that share a display name agree here too.
    for key in (display, display.lstrip("#"), sid):
      k = key.strip().lower()
      if k and k not in alias:
        alias[k] = display
  return alias


def canonicalize_gold_channels(
    commands: list[str], aliases: dict[str, str]
) -> tuple[list[str], list[str]]:
  """Speak the shipped channel vocabulary in the gold trajectory's own commands.

  The display name is the graded channel identity, but it is DERIVED here and
  never shown to the round's command author, which sees only raw ``session_id``s
  in its background payload. The author therefore writes ``--channel
  'channel_ops_triage'`` or ``--channel 'ops-triage'`` while every reconciled
  sub-agent posts to ``#ops-triage`` -- one conversation split across two channel
  strings. A ``messages_sent`` criterion then fails on the gold's own transcript
  for a reason that has nothing to do with the conflict being tested ("the
  message is in #channel_ops_triage instead of #ops-triage").

  Returns ``(commands, unresolved)``. Only provable aliases are rewritten;
  ``unresolved`` collects channel values that match no shipped session so the
  CALLER can bounce them back to the author (who has the semantics to choose)
  rather than having a similarity score pick for it.
  """
  if not aliases:
    return list(commands), []
  out: list[str] = []
  unresolved: list[str] = []
  for cmd in commands:
    if not (
        isinstance(cmd, str) and _is_gold_channel_command(cmd) and "--channel" in cmd
    ):
      out.append(cmd)
      continue
    raw = _effect_channel_value(cmd)
    if not raw:
      out.append(cmd)
      continue
    canon = aliases.get(raw.strip().lower())
    if canon is None:
      unresolved.append(raw)
    elif canon != raw:
      cmd = _rewrite_effect_channel(cmd, canon)
    out.append(cmd)
  return out, unresolved


def assign_unresolved_channels(
    commands: list[str], aliases: dict[str, str], registry: list[str]
) -> tuple[list[str], dict[str, str]]:
  """Last resort: bind each still-unknown channel name to a channel that exists.

  For the FINAL generation attempt only, where the alternative is discarding an
  otherwise-valid scenario -- the same A3-style escape the path-grounding and
  anti-repetition gates take. By then the author has been shown the vocabulary
  explicitly and still not used it, and ``canonicalize_gold_channels``
  deliberately refuses to guess.

  Assignment is a stable order-of-first-appearance mapping onto the shipped
  channels, NOT a similarity score. This is a deliberately weaker claim: it does
  not pretend to know WHICH real channel the author meant. What it does
  guarantee is the property that decides whether the round can be graded fairly
  -- every channel the gold names is one the solver can actually find, and two
  names the author kept apart stay apart, so a round that ran two conversations
  still runs two. A similarity score cannot promise either: measured on v6 the
  invented names share no tokens with the real ones (``prod-access-permits`` and
  ``prod-release-ops`` against ``#proj-migration-sync`` and ``#ops-escalations``),
  so scoring would merge both into whichever name happened to overlap.

  Returns ``(commands, mapping)``; ``mapping`` is casefolded-invented -> real,
  for the caller to log. When more names were invented than exist, the mapping
  wraps around and stops being injective -- unavoidable, and still better than
  naming a channel that is not there.
  """
  if not registry:
    return list(commands), {}
  mapping: dict[str, str] = {}
  out: list[str] = []
  for cmd in commands:
    if not (
        isinstance(cmd, str) and _is_gold_channel_command(cmd) and "--channel" in cmd
    ):
      out.append(cmd)
      continue
    raw = _effect_channel_value(cmd)
    if not raw:
      out.append(cmd)
      continue
    key = raw.strip().lower()
    if aliases.get(key) is not None:
      out.append(cmd)  # provable; canonicalize_gold_channels already handled it
      continue
    if key not in mapping:
      mapping[key] = registry[len(mapping) % len(registry)]
    out.append(_rewrite_effect_channel(cmd, mapping[key]))
  return out, mapping


def _match_canonical_channel(raw: str, session_ids: list[str]) -> str | None:
  """Map an LLM-authored channel name to the nearest canonical ``session_id``.

  Mock ``state_effects`` are authored in a *separate* LLM call that invents its
  own channel convention (``#ops-triage``), while the graded channel identity is
  the ``session_id`` of a shipped session file (``operations_triage_channel``).
  Token-match the two (allowing abbreviation prefixes like ``ops`` ~
  ``operations``) so the subagent's reply lands on a channel the grader and the
  solver's ``message search`` actually see. Returns ``None`` when nothing
  overlaps (caller falls back to the round's primary channel).
  """
  raw_tokens = [t for t in _channel_slug_tokens(raw) if not t.isdigit()]
  if not raw_tokens:
    return None
  best: str | None = None
  best_score = 0
  for sid in session_ids:
    sid_tokens = [t for t in _channel_slug_tokens(sid) if not t.isdigit()]
    if not sid_tokens:
      continue
    score = 0
    for rt in raw_tokens:
      for st in sid_tokens:
        if rt == st:
          score += 2
          break
        if st.startswith(rt) or rt.startswith(st):
          score += 1
          break
    if score > best_score:
      best_score = score
      best = sid
  return best if best_score > 0 else None


def _effect_channel_value(cmd: str) -> str | None:
  """Return the (unquoted) ``--channel`` value of a CLI-string effect, if any."""
  m = _CHANNEL_FLAG_RE.search(cmd)
  if not m:
    return None
  v = m.group(2)
  if len(v) >= 2 and v[0] in "'\"" and v[-1] == v[0]:
    v = v[1:-1]
  return v


def _rewrite_effect_channel(cmd: str, new_channel: str) -> str:
  """Replace the first ``--channel <x>`` value with ``'<new_channel>'``."""
  return _CHANNEL_FLAG_RE.sub(
      lambda m: f"{m.group(1)}'{new_channel}'", cmd, count=1
  )


def _is_message_send_effect(cmd: str) -> bool:
  s = cmd.strip().lower()
  return s.startswith("message send") or s.startswith("message broadcast")


def _is_gold_channel_command(cmd: str) -> bool:
  """Does this gold command name a channel that has to actually exist?

  Wider than :func:`_is_message_send_effect`, which asks a different question
  ("does this utterance land in the graded transcript?") and is therefore right
  to exclude reads. For the gold trajectory the test is whether the channel is
  REACHABLE: ``message search --channel`` on a channel that does not exist
  returns nothing, so the reference solution silently reads no evidence and the
  step it was supposed to justify rests on air. Measured on v6 all 88 gold
  searches happened to name a real channel, but nothing was enforcing that.
  """
  s = cmd.strip().lower()
  return (
      s.startswith("message send")
      or s.startswith("message broadcast")
      or s.startswith("message search")
  )


def _reconcile_subagent_channels(
    flat_effects: list[Any], session_ids: list[str]
) -> list[Any]:
  """Rewrite each message-send effect's ``--channel`` to a canonical session_id."""
  if not session_ids:
    return flat_effects
  out: list[Any] = []
  for eff in flat_effects:
    if isinstance(eff, str) and "--channel" in eff:
      raw = _effect_channel_value(eff)
      if raw:
        canon = _match_canonical_channel(raw, session_ids)
        if canon and canon != raw:
          eff = _rewrite_effect_channel(eff, canon)
    out.append(eff)
  return out


def _primary_round_channel(
    solution_commands: list[str], session_ids: list[str]
) -> str:
  """Pick the round's primary coordination channel.

  Prefer the canonical channel the reference solution talks to most; fall back to
  the first shipped session. Used as the reply channel for a subagent whose own
  effects carry no resolvable channel.
  """
  counts: dict[str, int] = {}
  for cmd in solution_commands:
    if (
        isinstance(cmd, str)
        and _is_message_send_effect(cmd)
        and "--channel" in cmd
    ):
      raw = _effect_channel_value(cmd)
      canon = _match_canonical_channel(raw, session_ids) if raw else None
      if canon:
        counts[canon] = counts.get(canon, 0) + 1
  if counts:
    return max(counts.items(), key=lambda kv: kv[1])[0]
  return session_ids[0] if session_ids else ""


def _reply_channel_for(
    flat_effects: list[Any], session_ids: list[str], primary: str
) -> str:
  """Canonical channel a subagent should reply on.

  Derived from the subagent's first message-send effect (channel-reconciled);
  falls back to the round's ``primary`` channel when none resolves.
  """
  for eff in flat_effects:
    if (
        isinstance(eff, str)
        and _is_message_send_effect(eff)
        and "--channel" in eff
    ):
      raw = _effect_channel_value(eff)
      canon = _match_canonical_channel(raw, session_ids) if raw else None
      if canon:
        return canon
      break
  return primary


def project_round_subagents(
    mock_subagents: list[Any],
    sessions: list[Any],
    solution_commands: list[str],
) -> list[dict[str, Any]]:
  """Apply materialization's sub-agent transforms WITHOUT materializing.

  The generator has to answer "can this round's gold satisfy the criterion we
  just wrote?" while the author is still in the retry loop -- which means
  predicting the ``messages_sent`` stream the runtime will produce. That stream
  depends on transforms that only happen later, at materialization: nested
  ``state_effects`` get flattened, their ``--channel`` values get reconciled onto
  the session registry, and each sub-agent gets the ``reply_channel`` the runtime
  routes its spoken reply through.

  Predicting them by hand is how the gate came to grade a fiction: it read the
  raw authored mocks, where ``reply_channel`` is absent, so every sub-agent reply
  it rendered was missing the ``in #channel`` the real transcript carries -- on a
  measured sample the reconstructed line set matched the real one 0/18 times, on
  BOTH the v5 and v6 corpora. Rather than restate the transforms, this runs the
  very same functions the materializer calls, in the same order, so the two
  cannot drift: a change to reconciliation is automatically a change to what the
  gate predicts.

  Returns copies; the caller's mocks are left untouched (they are still the
  authored artifact, and materialization -- not the gate -- owns rewriting them).
  """
  channels = _session_channel_registry(sessions)
  primary = _primary_round_channel(list(solution_commands or []), channels)
  out: list[dict[str, Any]] = []
  for mock in mock_subagents or []:
    if not isinstance(mock, dict):
      continue
    flat = _reconcile_subagent_channels(
        _flatten_subagent_state_effects(mock.get("state_effects")), channels
    )
    out.append({
        **mock,
        "state_effects": flat,
        "reply_channel": _reply_channel_for(flat, channels, primary),
    })
  return out


_FILE_WRITE_VERBS = frozenset({
    "create",
    "write",
    "append",
    "update",
    "edit",
    "set",
    "modify",
    "delete",
    "remove",
})


def _solution_touched_file_paths(solution_commands: list[str]) -> set[str]:
  """Normalized paths the expert writes/appends/modifies via ``file <verb>``.

  Broader than ``_solution_authored_files`` (which only handles ``file create``):
  used to detect when a subagent background effect would clobber a file the
  reference trajectory authors itself.
  """
  import shlex

  touched: set[str] = set()
  for cmd in solution_commands:
    if not isinstance(cmd, str) or not cmd.strip().startswith("file "):
      continue
    try:
      parts = shlex.split(cmd)
    except ValueError:
      continue
    if len(parts) < 2 or parts[0] != "file" or parts[1] not in _FILE_WRITE_VERBS:
      continue
    for idx, part in enumerate(parts):
      if part == "--path" and idx + 1 < len(parts):
        touched.add(_norm_file_path(parts[idx + 1]))
  return touched


_SUBAGENT_FILE_EFFECT_DICT_TYPES = frozenset({
    "file_create",
    "file_write",
    "write_file",
})


def _effect_target_file_path(eff: Any) -> str | None:
  """Return the normalized path a flattened subagent effect writes, or None.

  ``_flatten_subagent_state_effects`` emits file writes in *two* runtime shapes:
  a dict ``{"type": "file_create", "path": ...}`` and a raw CLI ``str`` such as
  ``file append --path reports/x.md --content ...`` (when the LLM authored the
  effect already as a command line). Both mutate the workspace when the event
  fires. This resolves the target path for either shape (any ``file`` write
  verb), so conflict detection covers both instead of only the dict form.
  """
  import shlex

  if isinstance(eff, dict):
    if str(eff.get("type", "")) in _SUBAGENT_FILE_EFFECT_DICT_TYPES:
      path = eff.get("path")
      if path:
        return _norm_file_path(str(path))
    return None
  if isinstance(eff, str):
    s = eff.strip()
    if not s.startswith("file "):
      return None
    try:
      parts = shlex.split(s)
    except ValueError:
      return None
    if len(parts) < 2 or parts[0] != "file" or parts[1] not in _FILE_WRITE_VERBS:
      return None
    for idx, part in enumerate(parts):
      if part == "--path" and idx + 1 < len(parts):
        return _norm_file_path(parts[idx + 1])
    return None
  return None


def _drop_conflicting_subagent_file_effects(
    effects: list[Any], expert_files: set[str]
) -> list[Any]:
  """Remove subagent file-write effects the expert also authors.

  Such an effect fires (once time is advanced) after the expert's own write and
  overwrites it with conflicting content, so the round's own ``file_check`` -
  which expects the expert's content - would fail. Covers both flattened effect
  shapes (``file_create`` dicts *and* raw ``file <verb> --path ...`` CLI strings)
  across every ``file`` write verb, so string-form writes/appends no longer slip
  through. Command effects and files the subagent alone produces are left intact.
  """
  if not expert_files:
    return effects
  kept: list[Any] = []
  for eff in effects:
    target = _effect_target_file_path(eff)
    if target is not None and target in expert_files:
      continue
    kept.append(eff)
  return kept


def _drop_conflicting_file_update_events(
    events: list[dict[str, Any]], expert_files: set[str]
) -> list[dict[str, Any]]:
  """Remove ``file_update`` background events the expert also authors.

  A ``file_update`` event (an injected "conflicting workspace update" from a
  round's ``workspace_adds``) is time-triggered, so once the reference trajectory
  advances virtual time via the injected ``wait`` it fires *after* the expert's
  own edits and clobbers them with the conflicting content. The round's
  ``file_check`` expects the expert's final content, so replaying the reference
  solution would then fail on a file it correctly wrote. When the expert authors
  the same path itself, drop the event so the expert's write is the authoritative
  final state - mirroring ``_drop_conflicting_subagent_file_effects`` and the
  ``_reconcile_file_checks`` policy that the expert's content wins. Events
  targeting files the expert never writes (the update is the only source of that
  file's content) are left intact.
  """
  if not expert_files:
    return events
  kept: list[dict[str, Any]] = []
  for ev in events:
    if (
        isinstance(ev, dict)
        and ev.get("type") == "file_update"
        and _norm_file_path(str(ev.get("path", ""))) in expert_files
    ):
      continue
    kept.append(ev)
  return kept


def _build_async_within_window_check(
    mock_subagents: list[dict[str, Any]],
    background_events: list[dict[str, Any]],
) -> dict[str, Any] | None:
  """Build the M8/CL2 sender-scoped ``within_window`` check, or ``None``.

  Returns a ``messages_sent`` / ``within_window`` effect check bound (by
  ``sender``) to the *slowest* mock sub-agent's reply -- the one whose latency
  forces the coordinator to wait -- with ``max_virtual_time`` set to the round's
  maximum background-event ``trigger_time`` (the derived SLA deadline, per the
  "derive = max background trigger" decision).

  Why this grades the async requirement (F6): the sub-agent's reply lands in
  ``messages_sent`` (via ``_record_subagent_message``) with ``sender=<role>``
  ONLY when the completion event fires, which happens ONLY when the solver
  advances the virtual clock with ``wait``. A no-wait trajectory (acting on a
  stale synchronous value) never produces that sender's effect, so the check
  fails closed. The environment stamps the effect at the event's ``trigger_time``
  (its true completion time), so the reference trajectory -- whose single
  injected ``wait`` reaches ``max_trigger`` -- is on time by construction, and an
  over-waiting solver is not penalized.

  Returns ``None`` (nothing async to grade) when the round declares no mock
  sub-agent or has no delayed background event. The caller additionally gates on
  ``seed.force_async_wait`` so non-opted seeds are unaffected (E6).
  """
  subs = [
      m
      for m in (mock_subagents or [])
      if isinstance(m, dict) and str(m.get("agent_role") or "").strip()
  ]
  if not subs:
    return None
  trigger_times = [
      int(be.get("trigger_time", 0) or 0)
      for be in (background_events or [])
      if isinstance(be, dict)
  ]
  deadline = max(trigger_times, default=0)
  if deadline <= 0:
    return None
  decisive = max(
      subs, key=lambda m: int(m.get("virtual_duration_seconds", 0) or 0)
  )
  role = str(decisive.get("agent_role")).strip()
  # L1 lower bound (no-wait bypass fix): the decisive reply legitimately arrives
  # only when its delayed completion fires, stamped at the sub-agent's own
  # ``trigger_time`` (== ``virtual_duration_seconds``). Requiring
  # ``virtual_time >= min_vt`` rejects a Route-2 *synchronous* mirror stamped at
  # ``virtual_time=0`` -- i.e. a no-wait solver eliciting the decisive value
  # without ever advancing the clock. ``min_vt <= deadline`` always holds because
  # the decisive completion is itself one of ``background_events`` (so its
  # trigger is included in ``max(...)``). Omitted when 0 (degenerate seed) so the
  # check is never stricter than the deadline.
  min_vt = int(decisive.get("virtual_duration_seconds", 0) or 0)
  expected: dict[str, Any] = {
      "max_virtual_time": deadline,
      "field": "sender",
      "value": role,
  }
  if min_vt > 0:
    expected["min_virtual_time"] = min_vt
  return {
      "type": "effect",
      "effect_type": "messages_sent",
      "condition": "within_window",
      "expected": expected,
  }


def _inject_background_wait(
    solution_commands: list[str], background_events: list[dict[str, Any]]
) -> list[str]:
  """Ensure the expert trajectory advances virtual time past background events.

  Background events (subagent completions, config/session updates) only fire
  when the environment's virtual clock is advanced via a ``wait`` command
  (``environment._tick``). LLM-authored expert trajectories almost never
  ``wait``, so time-triggered subagent file/command effects never materialize
  and the round's ``file_check`` / ``command_check`` fail against the reference
  solution itself. Insert one ``wait`` long enough to trigger every pending
  background event, just before the terminal ``DONE`` marker. ``system.wait`` is
  a core env command, exempt from tool-family gating, so this stays valid for
  every scenario.
  """
  trigger_times = [
      int(be.get("trigger_time", 0) or 0)
      for be in background_events
      if isinstance(be, dict)
  ]
  max_trigger = max(trigger_times, default=0)
  if max_trigger <= 0:
    return solution_commands

  existing_wait = 0
  for cmd in solution_commands:
    if isinstance(cmd, str) and cmd.strip().lower().startswith("wait "):
      parts = cmd.split()
      if len(parts) > 1:
        try:
          existing_wait = max(existing_wait, int(parts[1]))
        except ValueError:
          pass
  if existing_wait >= max_trigger:
    return solution_commands

  wait_cmd = f"wait {max_trigger}"
  result = list(solution_commands)
  terminal = {"done", "exit", "quit"}
  for idx, cmd in enumerate(result):
    if isinstance(cmd, str) and cmd.strip().lower() in terminal:
      result.insert(idx, wait_cmd)
      return result
  result.append(wait_cmd)
  return result


def _tag_asset_access_checks_diagnostic(
    checks: list[dict[str, Any]],
    *,
    decisive_asset_id: str | None = None,
) -> list[dict[str, Any]]:
  """F3: mark ``assets_accessed`` effect checks as ``role="diagnostic"``.

  See the call site in ``_compile_round_to_task`` for the rationale: scoring
  asset access rewards a modality-specific ritual rather than the task outcome,
  which breaks strategy/modality neutrality. Rather than DROP such checks (the
  old behaviour, which lost the "did the agent look at the evidence?" signal
  because the pre-role evaluator could not carry a non-gating check), we keep
  them tagged ``diagnostic`` so the role-aware ``EvaluatorComb`` records them
  without letting them affect success or score. Kept as a tiny, testable helper
  so the policy lives in exactly one place.

  M7 (Axis A): when ``decisive_asset_id`` is given, the round has been verified
  vision-confined (the decisive value is readable ONLY from that image), so the
  neutrality objection no longer applies -- there is no equivalent text channel to
  penalize. That one asset's access check is promoted to ``role="required"`` so
  opening the decisive image gates success; every other ``assets_accessed`` check
  (corroborating / distractor) stays ``diagnostic``. When ``decisive_asset_id`` is
  ``None`` (legacy / non-opted seeds) every check stays ``diagnostic`` exactly as
  before, so those scenarios remain byte-identical.
  """
  decisive = (decisive_asset_id or "").strip()
  tagged: list[dict[str, Any]] = []
  for c in checks:
    if (
        isinstance(c, dict)
        and c.get("type") == "effect"
        and c.get("effect_type") == "assets_accessed"
    ):
      check_asset_id = str((c.get("expected") or {}).get("value") or "").strip()
      is_decisive = bool(decisive) and check_asset_id == decisive
      tagged.append({**c, "role": "required" if is_decisive else "diagnostic"})
    else:
      tagged.append(c)
  return tagged


def _augment_instruction_for_asset_checks(
    instruction: str, checks: list[dict[str, Any]]
) -> str:
  """G1(b): make the implicit "inspect the visual evidence" step explicit.

  ``assets_accessed`` checks require the agent to actually open the evidence
  (``asset open``/``asset info``), but instructions often only say "analyze the
  chart". Append one generic, non-leaking nudge when such a check exists so the
  graded behaviour is stated in the task itself.
  """
  needs_assets = any(
      isinstance(c, dict) and c.get("effect_type") == "assets_accessed"
      for c in checks
  )
  if not needs_assets:
    return instruction
  # Module constant, not a local: M3 has to be able to recognise this sentence
  # to refuse to strip it. See ``_ASSET_LIST_GUIDANCE``.
  if _ASSET_LIST_GUIDANCE in instruction:
    return instruction
  return f"{instruction.rstrip()}\n\n{_ASSET_LIST_GUIDANCE}"


def _write_text(path: Path, content: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(content, encoding="utf-8")


def _write_json(path: Path, data: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)


def _backfill_asset_anchors(asset: VLMAssetSpec) -> VLMAssetSpec:
  from dataclasses import replace

  anchors = set(asset.anchor_ids)

  # 1. Extract from provider_metadata (charts)
  if isinstance(asset.provider_metadata, dict):
    chart_rows = asset.provider_metadata.get("chart_rows", [])
    if isinstance(chart_rows, list):
      for row in chart_rows:
        if isinstance(row, dict):
          anchor = row.get("anchor")
          if anchor:
            anchors.add(str(anchor))

  # 2. Extract from regions (visual assets)
  for region in asset.regions:
    for r_anchor in region.anchor_ids:
      if r_anchor:
        anchors.add(str(r_anchor))

  if len(anchors) > len(asset.anchor_ids):
    # Recreate the asset spec with updated anchor_ids
    return replace(asset, anchor_ids=sorted(list(anchors)))
  return asset


def _backfill_missing_round_anchors_to_assets(
    assets: list[VLMAssetSpec], raw_data: dict[str, Any]
) -> list[VLMAssetSpec]:
  from dataclasses import replace
  # Create a mapping of round_id to its expected anchors
  round_anchors: dict[str, list[str]] = {}
  for e_dict in raw_data.get("evals", []):
    r_id = e_dict.get("round_id")
    anchors = e_dict.get("expected_anchor_ids", [])
    if r_id and anchors:
      round_anchors[r_id] = [str(a) for a in anchors]

  # For each round's expected anchors, ensure at least one visible asset has it
  assets_by_id = {a.id: a for a in assets}

  for r_dict in raw_data.get("rounds", []):
    r_id = r_dict.get("round_id")
    if not r_id or r_id not in round_anchors:
      continue

    try:
      r_index = int(r_id.lstrip("r"))
    except ValueError:
      r_index = 1

    # Visible assets for this round (query updated assets)
    visible_assets = [a for a in assets_by_id.values() if a.available_from_round <= r_index]
    if not visible_assets:
      continue

    for anchor in round_anchors[r_id]:
      # Check if any visible asset already has this anchor
      has_anchor = any(anchor in a.anchor_ids for a in visible_assets)
      if not has_anchor:
        # Assign it to the first visible asset of matching kind or just the first visible asset
        target_asset = visible_assets[0]
        anchor_lower = anchor.lower()

        # Semantic heuristic matching:
        chart_keywords = ["chart", "traffic", "metrics", "throughput", "revenue", "connections", "lag", "drift", "dashboard"]
        text_keywords = ["log", "config", "text", "script", "yaml", "yml", "json", "txt", "timestamp", "suspicion", "bypass", "cron", "timeline", "loopback", "error", "timeout"]
        # Physical-photo anchors vs. schematic anchors. A schematic may be a
        # DIGITAL ``diagram`` (rendered as an in-UI HTML/SVG module) or a
        # hand-drawn ``sketch`` (an image), while photos are ``natural_image``.
        # Matching on kind in {natural_image, sketch, diagram} fixes the old
        # ``kind == "image"`` check that never matched a real asset.
        photo_keywords = ["router", "photo", "led", "indicator", "rack", "robot", "antenna", "hardware", "device", "cable"]
        diagram_keywords = ["topology", "diagram", "flow", "flowchart", "schematic", "layout", "wiring", "floorplan", "floor_plan", "map", "sketch", "whiteboard"]
        image_keywords = photo_keywords + diagram_keywords

        for asset in visible_assets:
          if any(kw in anchor_lower for kw in chart_keywords) and asset.kind == "chart":
            target_asset = asset
            break
          if any(kw in anchor_lower for kw in text_keywords) and asset.kind == "rich_text":
            target_asset = asset
            break
          if any(kw in anchor_lower for kw in image_keywords) and asset.kind in ("natural_image", "sketch", "diagram", "image"):
            target_asset = asset
            break

        # Update the target asset's anchor_ids
        updated_anchors = sorted(list(set(target_asset.anchor_ids + [anchor])))
        assets_by_id[target_asset.id] = replace(
            target_asset, anchor_ids=updated_anchors
        )

        # Sync the visible_assets list for subsequent anchors in the same round
        for idx, a in enumerate(visible_assets):
          if a.id == target_asset.id:
            visible_assets[idx] = assets_by_id[target_asset.id]

  return list(assets_by_id.values())


def _sanitize_instruction(text: str) -> str:
  import re
  # 1. Strip parenthesized anchors
  text = re.sub(r"\s*\((?:EVIDENCE|DECISION|RULE|UPDATE):[A-Z0-9_\-]+\)", "", text)
  # 2. Replace remaining unparenthesized anchors
  text = re.sub(r"\b(?:EVIDENCE|DECISION|RULE|UPDATE):[A-Z0-9_\-]+", "the relevant evidence", text)
  return " ".join(text.split())
