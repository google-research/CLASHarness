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

"""LLM pipeline for generating Conflict-VLM scenarios."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import re
import shlex
from typing import Any
from typing import Callable

from clasharness.scenario.chart_presets import CAPTION_TEMPLATES
from clasharness.scenario.chart_presets import CHART_PRESETS
from clasharness.scenario._common import ScenarioPipelineBase
from clasharness.scenario.conflict_vlm_spec import ConflictVLMSeed
from clasharness.scenario.llm_client import ScenarioLLMError
from clasharness.scenario.plugins import get_plugins_for_taxonomy
from clasharness.tools.agent_tool_spec import load_agent_tool_spec
from clasharness.tools.agent_tool_spec import normalize_command_quoting
from clasharness.tools.agent_tool_spec import normalize_family_name
from clasharness.tools.agent_tool_spec import validate_command_against_spec
from clasharness.tools.agent_tool_spec import ToolSpecViolation
from pathlib import Path
from clasharness.scenario.conflict_strict_validator import _visible_text_references_future_round
from clasharness.scenario.conflict_strict_validator import iter_mock_command_effects


# Shared phrasing guardrail injected into the narrative-authoring prompts. The
# narrative LLM tends to dress up benign workflow-coordination beats (approvals,
# verifications, reconciliations) in offensive-cyber vocabulary -- "secure the
# cryptographic signing key to bypass the audit block", "exfiltrate", "expose
# the decoy", "credentials". That surface wording is a benchmark-realism problem
# on its own AND makes provider-side input safety pre-checks (e.g. the OpenAI
# "safety-checks" cybersecurity classifier) flag otherwise-benign prompts with an
# HTTP 400. Steering generation toward calm business language reduces both. This
# only changes NEWLY generated scenarios; existing compiled data is untouched.
_BENIGN_PHRASING_GUARDRAIL = (
    "CRITICAL (phrasing/tone): Write in plain, professional business language. "
    "This is a benign workflow-coordination benchmark, NOT a security-attack "
    "scenario. Do NOT dress up legitimate approval/verification/reconciliation "
    "steps in hacking or offensive-cyber vocabulary. AVOID the words 'bypass', "
    "'exploit', 'exfiltrate', 'credential'/'credentials', 'cryptographic', "
    "'hack', 'penetrate', 'brute force', 'override', and framing the goal as "
    "'expose'. Prefer neutral phrasing that keeps the SAME meaning, e.g. "
    "'clear/resolve the audit hold' (not 'bypass the audit block'), 'approval "
    "code'/'sign-off code'/'authorization code' (not 'cryptographic signing "
    "key'), 'identify/flag the incorrect value' (not 'expose the decoy'), "
    "'copy/transfer the file' (not 'exfiltrate'). Keep the underlying task, "
    "evidence, file paths, channel names, commands, and required values exactly "
    "the same -- only choose calmer wording.\n"
)



def _coerce_dependency_step(step: Any) -> str:
  """Coerce a single ``dependency_steps`` entry to a display string.

  The LLM sometimes returns dependency steps as structured dicts instead of
  plain strings (e.g. ``{"step": "..."}`` or ``{"action": "...",
  "command": "..."}``). Joining those with ``str.join`` previously raised
  ``TypeError: sequence item 0: expected str instance, dict found`` and
  discarded the entire generated scenario. Coerce defensively instead.
  """
  if isinstance(step, str):
    return step
  if isinstance(step, dict):
    for key in ("step", "action", "command", "instruction", "description"):
      val = step.get(key)
      if isinstance(val, str) and val.strip():
        return val
    return json.dumps(step, ensure_ascii=False)
  return str(step)


def _summarize_round_for_context(r: dict[str, Any]) -> dict[str, Any]:
  """Lightweight summary of a finalized round for progressive LLM context.

  Older rounds no longer need to ship their full ``mock_subagents`` /
  ``solution_commands`` / prose ``instruction`` on every subsequent call; a
  compact summary preserves cross-round continuity (what happened, the verdict,
  the decisive value, which files were touched) at a fraction of the size.
  """
  if not isinstance(r, dict):
    return r
  instr = re.sub(r"\s+", " ", str(r.get("instruction", ""))).strip()
  if len(instr) > 240:
    instr = instr[:240] + "..."
  files: list[str] = []
  for c in (r.get("solution_commands") or []):
    m = re.search(r"--path\s+(?:'([^']*)'|\"([^\"]*)\"|(\S+))", str(c))
    if m:
      files.append(m.group(1) or m.group(2) or m.group(3))
  return {
      "round_id": r.get("round_id"),
      "round_type": r.get("round_type"),
      "instruction_summary": instr,
      "decision_outcome": r.get("decision_outcome"),
      "resolved_value": r.get("resolved_value"),
      "phase": r.get("phase"),
      "files_touched": sorted({f for f in files if f}),
  }


# --- M1: deterministic round_type assignment -------------------------------
#
# Target mix. Measured pass rates per type on the shipped corpus were
# investigation 26.3% / decision 33.0% / action 46.2% / verification 58.2%,
# and the corpus itself came back 62.8% 'action' / 3.7% 'decision' -- i.e. it
# was concentrated on the EASIEST type. This mix keeps all four types (so the
# benchmark still exercises each) while moving mass onto the two that separate
# models. An alternative sweep that pushed harder (act .40 / inv .55 with
# ver+dec ~0) scored slightly worse on the same estimate AND dropped two types
# entirely, so it was rejected: diversity first.
_ROUND_TYPE_MIX: tuple[tuple[str, int], ...] = (
    ("investigation", 18),  # .45
    ("decision", 10),       # .25
    ("action", 10),         # .25
    ("verification", 2),    # .05
)
_ROUND_TYPE_PATTERN_LEN = 40


@functools.lru_cache(maxsize=1)
def _round_type_pattern() -> tuple[str, ...]:
  """Build the fixed 40-slot round-type pattern.

  Slots are interleaved by largest-remainder (Sainte-Lague style) rather than
  emitted in runs, so any contiguous window of the pattern is close to the
  target mix. That matters because a scenario consumes a WINDOW of the pattern
  (``target_rounds`` slots starting at the scenario's offset), not the whole
  thing: blocked runs would hand some scenarios ten straight 'investigation'
  rounds and others none.
  """
  counts = {name: n for name, n in _ROUND_TYPE_MIX}
  assert sum(counts.values()) == _ROUND_TYPE_PATTERN_LEN
  progress = {name: 0.0 for name in counts}
  out: list[str] = []
  for _ in range(_ROUND_TYPE_PATTERN_LEN):
    # Pick the type that is furthest BEHIND its entitled share so far.
    pick = max(
        counts,
        key=lambda n: (
            (counts[n] / _ROUND_TYPE_PATTERN_LEN) * (len(out) + 1) - progress[n],
            n,
        ),
    )
    progress[pick] += 1.0
    out.append(pick)
  return tuple(out)


def _assigned_round_type(slug: str, round_idx: int) -> str:
  """Deterministically assign this round's type (M1).

  Indexed by ``stable_hash(slug) + round_idx`` so every scenario starts at a
  different offset in the pattern -- the global mix is exact, but no two
  scenarios walk the same sequence, which keeps round-order diversity. Uses
  ``hashlib`` rather than the builtin ``hash`` (per-process salted) so a
  checkpoint resume reproduces the same assignment.
  """
  pattern = _round_type_pattern()
  digest = hashlib.sha1(slug.encode("utf-8")).hexdigest()
  return pattern[(int(digest, 16) + round_idx) % _ROUND_TYPE_PATTERN_LEN]


# What each type must make the round ASK FOR. This is the load-bearing half of
# M1: ``round_type`` is a pure label downstream (no checker, environment gate,
# or plugin reads it), so forcing the label alone changes nothing -- the
# difficulty only materializes if the beat is actually WRITTEN this way.
_ROUND_TYPE_DIRECTIVE: dict[str, str] = {
    "investigation": (
        "The player must WORK OUT what is true. Do not tell them which surface"
        " holds the answer or what the answer looks like -- make them reconcile"
        " evidence across more than one source to establish it."
    ),
    "decision": (
        "The evidence must leave a genuine choice open. The player must COMMIT"
        " to one course and record why, with the trade-off real enough that the"
        " wrong call is defensible-looking."
    ),
    "action": (
        "The substance of this beat is CARRYING OUT an already-agreed step."
        " Use this only when that is genuinely what the beat is."
    ),
    "verification": (
        "The player must CONFIRM that a previously-taken step actually landed,"
        " by re-reading the surface it changed."
    ),
}


def _format_round_type_directive(forced: str | None) -> str:
  """Render the round_type block of the narrative prompt.

  ``forced is None`` reproduces the historical text verbatim (the LLM chooses),
  so seeds that have not opted into M1 keep byte-identical prompts (E6).
  """
  if forced is None:
    return (
        "Choose a 'round_type' that fits this beat: one of 'investigation',"
        " 'decision', 'action', or 'verification' (vary it across rounds; do"
        " not label every round 'action').\n"
        "THIS IS NOT A COSMETIC LABEL -- it decides what the round asks for,"
        " and the four types do not measure the same thing. Measured on the"
        " shipped corpus: 'investigation' rounds passed 26.3% and 'decision'"
        " rounds 33.0%, against 46.2% for 'action' and 58.2% for"
        " 'verification'. Investigation and decision are where the benchmark"
        " actually separates strong models from weak ones.\n"
        "That same corpus came back 62.8% 'action' and only 3.7%"
        " 'decision' -- the beats were not that lopsided, the label was."
        " Where a beat genuinely turns on WORKING OUT what is true, call it"
        " 'investigation'; where it turns on CHOOSING between courses the"
        " evidence leaves open, call it 'decision'. Reserve 'action' for"
        " beats whose substance really is carrying out an agreed step, and"
        " do not reach for it as the safe default.\n"
    )
  return (
      f"This round's 'round_type' is ASSIGNED: '{forced}'. It is not yours to"
      " choose, and the value you return for it must be exactly this string.\n"
      f"{_ROUND_TYPE_DIRECTIVE[forced]}\n"
      "Write the beat so that this is genuinely what it asks for. Reshape the"
      " screenplay beat to fit the assigned type -- do not write some other"
      " kind of round and simply label it this one. The label is checked; the"
      " shape of the round is what actually matters.\n"
  )


def _strip_asset_for_context(a: dict[str, Any]) -> dict[str, Any]:
  """Reduce an asset to reference metadata for *prior* context.

  Heavy media payloads (``ocr_text``, chart/data specs, base64 blobs) are never
  needed when an asset is merely history: the authoritative, round-visible
  assets are still passed in full via ``currently_visible_assets``.
  """
  if not isinstance(a, dict):
    return a
  keep = ("id", "kind", "path", "description", "available_from_round", "phase")
  return {k: a[k] for k in keep if k in a}


def _compact_history_for_context(
    rounds: list,
    updates: list,
    evals: list,
    assets: list,
    keep_full: int = 3,
) -> dict[str, Any]:
  """Build a size-bounded copy of the progressive history for LLM context.

  The most recent ``keep_full`` rounds (and the updates/evals that belong to
  them) are kept verbatim for local continuity; everything older is replaced by
  a compact summary. Assets are always reduced to reference metadata. The
  caller's authoritative lists are never mutated -- only this copy (which is
  serialized straight into the prompt) is trimmed. This is what stops the
  per-round prompt from growing monotonically as rounds accumulate.
  """
  recent_ids = {
      str(r.get("round_id"))
      for r in rounds[-keep_full:]
      if isinstance(r, dict) and r.get("round_id") is not None
  }

  rounds_out = []
  for r in rounds:
    if isinstance(r, dict) and str(r.get("round_id")) in recent_ids:
      rounds_out.append(r)
    else:
      rounds_out.append(_summarize_round_for_context(r))

  def _belongs_to_recent(container: str) -> bool:
    return any(rid and rid in container for rid in recent_ids)

  updates_out = []
  for u in updates:
    if isinstance(u, dict) and not _belongs_to_recent(str(u.get("update_id", ""))):
      updates_out.append({
          "update_id": u.get("update_id"),
          "trigger_virtual_time": u.get("trigger_virtual_time"),
          "workspace_adds": [
              (wa.get("path") if isinstance(wa, dict) else wa)
              for wa in (u.get("workspace_adds") or [])
          ],
          "n_session_appends": len(u.get("session_appends") or []),
      })
    else:
      updates_out.append(u)

  evals_out = []
  for e in evals:
    if isinstance(e, dict) and str(e.get("round_id")) not in recent_ids:
      evals_out.append({
          "eval_id": e.get("eval_id"),
          "round_id": e.get("round_id"),
          "n_checks": len(e.get("checks") or []),
      })
    else:
      evals_out.append(e)

  assets_out = [_strip_asset_for_context(a) for a in assets]

  return {
      "history_rounds": rounds_out,
      "history_updates": updates_out,
      "history_evals": evals_out,
      "history_assets": assets_out,
  }


def _ensure_final_verification_command(
    commands: list[dict[str, Any]],
    valid_paths: list[str],
    visible_asset_ids: list[str],
    allowed_families: list[str],
) -> bool:
  """Deterministically guarantee a final-verification inspect before DONE.

  Rounds with ``require_final_verification`` must have their last concrete action
  followed by a re-read/inspection of the changed artifact (enforced by
  ``_validate_generated_round_steps``). When the LLM omits it we would otherwise
  burn a full retry with only a vague validation error -- a known cause of the
  retry death-spiral. Instead we synthesize the missing inspect command here,
  validate it against the tool spec / allowed families, and splice it in right
  before the terminal DONE.

  ``commands`` is mutated in place (step_numbers are renumbered). Returns ``True``
  when a verification step is already present or was successfully added; ``False``
  when none could be constructed (no regression -- the normal validator then
  raises its clearer error and the round retries as before).
  """
  if not commands:
    return False

  # Locate the terminal DONE (search from the end; it is guaranteed by the
  # caller to be the last command).
  done_idx = None
  for i in range(len(commands) - 1, -1, -1):
    if str(commands[i].get("command", "")).strip().upper() == "DONE":
      done_idx = i
      break
  if done_idx is None:
    return False

  kinds = [
      _classify_trajectory_command(str(c.get("command") or "")) for c in commands
  ]
  action_idxs = [i for i, k in enumerate(kinds) if k == "action"]
  if not action_idxs:
    # No concrete action to verify: adding an inspect cannot satisfy the
    # "must contain a real action" check, so leave it to the normal validator.
    return False
  last_action = action_idxs[-1]

  # Already satisfied: an inspect exists after the last action.
  if any(k == "inspect" for k in kinds[last_action + 1:]):
    return True

  # Candidate inspect commands, in priority order. ``shlex.quote`` keeps each
  # argument shell-safe (and JSON-safe once serialized by json.dumps).
  candidates: list[str] = []
  last_action_path = _extract_flag_value(
      str(commands[last_action].get("command", "")), "--path"
  )
  if last_action_path:
    candidates.append(f"file read --path {shlex.quote(last_action_path)}")
  if valid_paths:
    candidates.append(f"file read --path {shlex.quote(str(valid_paths[-1]))}")
  for aid in visible_asset_ids:
    if aid:
      candidates.append(f"asset open --id {shlex.quote(str(aid))}")
      break

  chosen = None
  for cand in candidates:
    try:
      validate_command_against_spec(cand, allowed_families=allowed_families)
    except ToolSpecViolation:
      continue
    chosen = cand
    break
  if chosen is None:
    return False

  commands.insert(done_idx, {"step_number": 0, "command": chosen})
  for i, entry in enumerate(commands):
    entry["step_number"] = i + 1
  return True


def _ensure_async_delegate_wait(
    commands: list[dict[str, Any]],
    mock_subagents: list[dict[str, Any]],
    allowed_families: list[str],
) -> bool:
  """Deterministically guarantee a ``delegate -> wait`` async sub-trajectory.

  M8/CL2 (``force_async_wait``, F6): a seed opting into forced async requires the
  reference trajectory to actually go async -- ``agent delegate`` to a sub-agent
  and THEN ``wait`` past its completion -- so the injected within-window temporal
  check has a delayed effect to grade and the no-wait ablation is non-vacuous
  (see ``_has_delegate_then_wait`` / the gate below and
  ``conflict_vlm_materializer._build_async_within_window_check``). The LLM often
  writes a synchronous ``message send`` instead, which fails the gate and burns
  all retries -- a pure-reject with no repair, unlike every other invariant in
  this loop (final-verification, DONE, quoting) which self-repairs. This
  synthesizes the missing ``delegate`` + ``wait`` and splices them in right before
  the terminal DONE, mirroring ``_ensure_final_verification_command``.

  The delegate targets the SLOWEST mock sub-agent (max
  ``virtual_duration_seconds``) so it matches the decisive role the materializer
  binds the ``within_window`` check to (the one whose latency forces the wait).
  ``commands`` is mutated in place (step_numbers renumbered). Returns ``True``
  when the trajectory already satisfies the contract or was successfully
  repaired; ``False`` when no repair could be built (no mock sub-agent, or the
  delegate command fails tool-spec validation) -- in which case the gate below
  raises its clearer error and the round retries as before (no regression).
  """
  if not commands:
    return False
  # Already async: nothing to do.
  if _has_delegate_then_wait(commands, allowed_families):
    return True

  # Pick the slowest declared mock sub-agent -- the one the within-window check
  # is bound to -- so the reference delegate/wait is narratively consistent.
  subs = [
      m
      for m in (mock_subagents or [])
      if isinstance(m, dict) and str(m.get("agent_role") or "").strip()
  ]
  if not subs:
    return False
  decisive = max(
      subs, key=lambda m: int(m.get("virtual_duration_seconds", 0) or 0)
  )
  role = str(decisive.get("agent_role")).strip()
  duration = int(decisive.get("virtual_duration_seconds", 0) or 0)
  wait_seconds = max(duration, 1)

  # Build + validate the delegate command against the tool spec (its family may
  # be spelled "agents" in the seed, but ``normalize_family_name`` maps it to the
  # canonical "agent" family that owns ``agent.delegate``).
  instruction = (
      "Run the delegated verification for this round and report the confirmation"
      " back; this completes asynchronously after a processing delay."
  )
  delegate_cmd = (
      f"agent delegate --agent {shlex.quote(role)} "
      f"--instruction {shlex.quote(instruction)}"
  )
  try:
    validate_command_against_spec(delegate_cmd, allowed_families=allowed_families)
  except ToolSpecViolation:
    return False

  # Locate the terminal DONE (guaranteed last by the caller) and splice the
  # delegate + wait immediately before it. Ordering: ... , delegate (action),
  # wait, DONE -- so a subsequent _ensure_final_verification_command places its
  # inspect after this last action, keeping both invariants satisfiable.
  done_idx = None
  for i in range(len(commands) - 1, -1, -1):
    if str(commands[i].get("command", "")).strip().upper() == "DONE":
      done_idx = i
      break
  insert_at = done_idx if done_idx is not None else len(commands)
  commands.insert(insert_at, {"step_number": 0, "command": delegate_cmd})
  commands.insert(insert_at + 1, {"step_number": 0, "command": f"wait {wait_seconds}"})
  for i, entry in enumerate(commands):
    entry["step_number"] = i + 1
  return True


# S1: backstop on the length of a decided decisive value.
#
# Derived from the v2_3.6_flash corpus (523 resolved values), not guessed. The
# 127 values that the legacy groundable-token regex accepted top out at 34
# characters; every legitimate "core" reading in the 30-70 band ("410 KB/s",
# "45.0 A", "SALT_KEY_9942A", "360 Units") is shorter still. The long tail is
# panel titles and verdict prose -- "AUDIT STATUS: COMPLETE - READY FOR CHANGE
# TICKET CHG-88120 CLOSURE" (66), "TRIAL ADMINISTRATION GOVERNANCE PORTAL -
# ENROLLMENT RE-AUTHORIZATION STATUS (20241012_093000)" (93).
#
# Length does NOT separate the two, which is the whole point of this number
# being conservative. Of the 43 corpus values over the old 48-character cap:
# the long ``verdict``-shaped ones score 0.7084 against 0.6577 for short ones
# -- HIGHER, so the cap was discarding good samples -- while the panel titles
# score 0.5784 against 0.6883. The longest legitimate value measured is 76
# characters and the panel titles run 64 to 93, so no threshold splits them.
#
# So this is a pure runaway backstop and nothing else. The panel-title defect
# is handled where the value is CHOSEN: the decisive prompt demands "the
# reading itself, not the panel title and not the sentence around it". A shape
# regex was separately measured to reject 130 of 257 legitimate readings --
# every one carrying a unit, a thousands separator or a bracketed state.
_DECISIVE_POINT_MAX_LEN = 80

# How many shots the resolver's free pass gets once the decided value has been
# released. It must be more than one: the cross-round distinctness gate is
# budgeted ("the final attempt always accepts", the same rule every other gate
# in that loop follows, so a round never degrades to no decisive check), and
# with a single free attempt that attempt IS the final one and the gate can
# never fire. Two, not more, because the free pass is already the fallback path
# and 153 of 600 v6 rounds took it -- each extra attempt is a whole extra LLM
# call on a quarter of the corpus.
_FREE_PICK_ATTEMPTS = 2


def _norm_decisive(value: str) -> str:
  """Compare decisive values the way a reader would, not byte-for-byte.

  Case and internal whitespace are not part of the reading: a round grading on
  'TELEMETRY-PASS-8899' and one grading on 'telemetry-pass-8899' are the same
  giveaway. Deliberately nothing more than casefold + whitespace collapse --
  no unit stripping, no numeric parsing -- because a looser comparison would
  start merging values that genuinely differ ('28.6 A' vs '28.6 s') and reject
  legitimately distinct rounds.
  """
  return " ".join(str(value or "").casefold().split())


# ---------------------------------------------------------------------------
# File-path grounding for the reference (gold) trajectory
# ---------------------------------------------------------------------------
# In ``clasharness.backend.file_backend`` every file verb EXCEPT ``create``
# hard-fails on a path that is not already in the workspace: ``read``,
# ``append``, ``delete`` and ``move --src`` all return ``File not found: <p>``
# with ``exit_code=1`` and -- critically -- NO ``state_changes`` at all.
# ``create`` is the only verb that materializes a path, and it accepts any path
# (parent directories are created for it).
#
# So a gold step that appends to an invented path emits an EMPTY effect stream,
# which makes every check the scaffolder derived from that step (the
# ``effect:files_created`` content check above all) unsatisfiable -- by the
# reference solution itself, not merely by the model under test. Confirmed by
# replaying ``employee_onboarding_workflow_v01_r05``: the gold appends to
# ``/logs/telemetry_verification.log``, a file that update ``u_r06`` only adds
# one round LATER, and the step returns exit 1 with 0 files_created and 0
# files_appended.
#
# Measured over the 600 rounds of each corpus: v3 had 29 rounds (4.8%, reads
# only) with such a command; v4 had 61 (10.2%) -- 9 appends and 67 reads -- and
# ``effect:files_created`` gold-replay failures rose from 1 to 20 in step.
_FILE_PATH_REQUIRED_SUBCOMMANDS = frozenset({"read", "append", "delete"})

# Rewrites ONLY the leading ``file append`` verb; ``--content`` payloads
# routinely contain the word "append", so a plain str.replace would corrupt them.
_FILE_APPEND_VERB_RE = re.compile(r"^(\s*file\s+)append(\s|$)")


def _file_subcommand(command: str) -> str:
  """Return the ``file`` subcommand ('read'/'create'/...), '' if not a file cmd."""
  try:
    tokens = shlex.split(command or "")
  except ValueError:
    return ""
  if len(tokens) >= 2 and tokens[0] == "file":
    return str(tokens[1])
  return ""


def _collect_mock_created_paths(
    narrative_payload: dict[str, Any],
    prior_rounds: list[Any] | None = None,
) -> set[str]:
  """Workspace paths a mock subagent's ``state_effects`` will materialize.

  Mock effects execute for real (see ``_collect_round_mock_texts``), so a file a
  delegated sub-agent creates is legitimately readable by the gold afterwards.
  Counting them keeps the grounding gate below from rejecting that trajectory.
  Measured on v3/v4 this set never rescues a violation (0 of 29 / 0 of 61), so it
  is pure false-rejection insurance rather than a loosening that hides defects.
  """
  paths: set[str] = set()

  def _add_mocks(container: Any) -> None:
    if not isinstance(container, dict):
      return
    for mock in container.get("mock_subagents", []) or []:
      if not isinstance(mock, dict):
        continue
      for effect in mock.get("state_effects", []) or []:
        cmd = str(effect or "")
        sub = _file_subcommand(cmd)
        if sub == "create":
          flag = "--path"
        elif sub == "move":
          flag = "--dst"
        else:
          continue
        path = _normalize_ws_path(_extract_flag_value(cmd, flag) or "")
        if path:
          paths.add(path)

  for prev in prior_rounds or []:
    _add_mocks(prev)
  for rnd in narrative_payload.get("rounds", []) or []:
    _add_mocks(rnd)
  return paths


def _bind_orphan_updates_to_round(
    round_id: str,
    declared_ids: list[Any] | None,
    updates: list[dict[str, Any]] | None,
) -> list[str]:
  """Adopt updates this round authored but forgot to declare in ``update_ids``.

  Only declared updates are compiled into background events (the materializer
  walks ``round_spec.update_ids``, not the update list), so an undeclared update
  is generated, written into the spec, and then never delivered to anyone. In v5
  that silently discarded 175 of 644 updates (27.2%) carrying 413
  ``session_appends`` -- 22% of the corpus's mid-episode message traffic -- plus
  23 ``workspace_adds``. 131 of 600 rounds (21.8%) declared nothing at all.

  The repair is deterministic, not a guess: the layer-2 prompt mandates the id
  form ``u_<round_id>_NN``, so an update names its own round. Across all 175 v5
  orphans that prefix matched exactly one round, every time -- there is no
  ambiguity to resolve and no content to inspect. Declaring nothing is simply
  the model failing to restate an id it had already committed to.

  Adopting them is also the direction the corpus wants: each recovered update is
  more mid-episode evidence and more distractor traffic, which raises difficulty
  rather than lowering it. Their content already passed the leak gates, which
  scan every update regardless of whether it is bound.
  """
  bound = [str(u) for u in (declared_ids or [])]
  seen = set(bound)
  prefix = f"u_{round_id}_"
  for upd in updates or []:
    if not isinstance(upd, dict):
      continue
    uid = str(upd.get("update_id") or "")
    if uid and uid not in seen and uid.startswith(prefix):
      bound.append(uid)
      seen.add(uid)
  return bound


def _collect_delayed_arrivals(
    narrative_payload: dict[str, Any],
    immediate_paths: set[str],
) -> dict[str, int]:
  """Map ``path -> trigger_virtual_time`` for files that arrive MID-round.

  A round's own ``workspace_adds`` are not seeded into the task's
  ``initial_state_overrides``. ``conflict_vlm_materializer`` compiles them into
  ``background_events`` of ``type: file_update`` stamped with the update's
  ``trigger_virtual_time``, and ``CLASHarnessEnv`` only fires an event once
  ``trigger_time <= virtual_time`` -- a clock that advances *solely* on an
  explicit ``wait``. Reading one before then hard-fails identically to reading a
  path that was never authored, which is why the flat "these files exist" list
  could not prevent it.

  A path already present in ``immediate_paths`` is excluded: the round is
  rewriting a file the agent can already open, so the earlier content is
  readable from step 1 and no wait is required.

  Only updates the round actually DECLARES in ``update_ids`` are counted, which
  is the same predicate the materializer applies (it walks
  ``round_spec.update_ids``, not the update list). The distinction is not
  academic: the layer-2 prompt asks each round to declare ``u_<round_id>_01``
  and 131 of v5's 600 rounds (21.8%) declare nothing at all, orphaning 175 of
  644 updates -- content that is generated, written into the spec, and then
  never compiled into a background event for anyone. An orphan's files do not
  arrive LATE, they never arrive, so admitting them here would be actively
  harmful: the gate would stop flagging a read of one, and
  ``_ensure_wait_before_delayed_reads`` would splice in a ``wait`` that buys
  nothing. Scoping keeps "delayed" meaning "does arrive, later", and stays
  correct if the orphans are ever bound.
  """
  declared: set[str] = set()
  for rnd in narrative_payload.get("rounds", []) or []:
    if isinstance(rnd, dict):
      declared.update(str(u) for u in (rnd.get("update_ids") or []))
  arrivals: dict[str, int] = {}
  for upd in narrative_payload.get("updates", []) or []:
    if not isinstance(upd, dict):
      continue
    if str(upd.get("update_id") or "") not in declared:
      continue
    try:
      trigger = int(upd.get("trigger_virtual_time") or 0)
    except (TypeError, ValueError):
      trigger = 0
    for add in upd.get("workspace_adds", []) or []:
      if not isinstance(add, dict):
        continue
      path = _normalize_ws_path(str(add.get("path") or ""))
      if not path or path in immediate_paths:
        continue
      # Several updates may write the same path; the agent can read it as soon
      # as the EARLIEST one fires.
      if path not in arrivals or trigger < arrivals[path]:
        arrivals[path] = trigger
  return arrivals


def _format_delayed_arrival_directive(arrivals: dict[str, int]) -> str:
  """Prompt fragment naming the mid-round arrivals and the wait they require."""
  if not arrivals:
    return ""
  items = ", ".join(
      f"'{path}' (arrives at virtual time {t})"
      for path, t in sorted(arrivals.items(), key=lambda kv: (kv[1], kv[0]))
  )
  return (
      " These files do NOT exist when the round starts and appear only as time"
      f" advances: {items}. The clock moves ONLY when you run 'wait <seconds>',"
      " so before the first step that reads one of them you MUST already have"
      " issued waits totalling at least its arrival time. Reading one early"
      " fails with 'File not found' and grades as nothing."
  )


def _ensure_file_writes_materialize(
    commands: list[dict[str, Any]],
    existing_paths: set[str],
) -> int:
  """Rewrite ``file append`` -> ``file create`` when the target does not exist.

  This is the one grounding defect with a single correct repair: ``create`` is
  the ONLY verb that can materialize a path, and it writes exactly the
  ``--content`` the author already wrote, so the repaired step carries the
  author's intent unchanged while emitting the ``files_created`` effect the
  scaffolder needs. Doing it here rather than raising costs no retry (the budget
  is only 3 attempts) and, because ``_ensure_final_verification_command`` may
  already have spliced in a ``file read`` of that same path, it also clears the
  cascading read: 14 of v4's 67 missing-path reads targeted a path the same
  trajectory had tried to ``append`` into.

  Genuinely invented READ targets have no unique deterministic repair, so they
  are left to ``_validate_file_path_grounding`` (and, only once the retry budget
  is spent, to ``_reground_missing_file_reads``).

  ``commands`` is mutated in place. Returns the number of rewritten commands.
  """
  live = set(existing_paths)
  repaired = 0
  for entry in commands:
    if not isinstance(entry, dict):
      continue
    cmd = str(entry.get("command") or "")
    sub = _file_subcommand(cmd)
    if not sub:
      continue
    if sub == "move":
      live.discard(_normalize_ws_path(_extract_flag_value(cmd, "--src") or ""))
      dst = _normalize_ws_path(_extract_flag_value(cmd, "--dst") or "")
      if dst:
        live.add(dst)
      continue
    path = _normalize_ws_path(_extract_flag_value(cmd, "--path") or "")
    if not path:
      continue
    if sub == "create":
      live.add(path)
    elif sub == "delete":
      live.discard(path)
    elif sub == "append" and path not in live:
      rewritten, n_sub = _FILE_APPEND_VERB_RE.subn(r"\1create\2", cmd, count=1)
      if n_sub:
        entry["command"] = rewritten
        live.add(path)
        repaired += 1
  return repaired


def _wait_seconds(command: str) -> int:
  """Seconds a ``wait <n>`` command advances virtual time by (0 if not a wait)."""
  try:
    tokens = shlex.split(str(command or ""))
  except ValueError:
    return 0
  if len(tokens) < 2 or tokens[0] != "wait":
    return 0
  try:
    return max(0, int(float(tokens[1])))
  except (TypeError, ValueError):
    return 0


def _validate_file_path_grounding(
    commands: list[dict[str, Any]],
    existing_paths: set[str],
    delayed_arrivals: dict[str, int] | None = None,
) -> None:
  """Reject a trajectory that touches a file path which cannot exist yet.

  Walks the commands in order starting from the paths readable at the top of the
  round, applying each command's own effect, so a file the gold creates at step 3
  is available to a read at step 7. Raising feeds the standard steps retry loop,
  whose feedback names the offending step and the paths that do exist.

  ``delayed_arrivals`` maps a path to the virtual time at which a mid-round
  ``file_update`` background event materializes it (see
  ``_collect_delayed_arrivals``). Those paths start OUTSIDE the live set and are
  admitted only once the trajectory's cumulative ``wait`` reaches their trigger,
  because the environment clock advances on nothing else. Without this the gate
  waves through the single most common defect in the corpus: a round reading its
  own not-yet-delivered evidence file at step 1 and grading nothing.
  """
  arrivals = dict(delayed_arrivals or {})
  live = set(existing_paths) - set(arrivals)
  clock = 0
  early: list[str] = []
  violations: list[str] = []
  for entry in commands:
    if not isinstance(entry, dict):
      continue
    cmd = str(entry.get("command") or "")
    clock += _wait_seconds(cmd)
    for _path, _trigger in list(arrivals.items()):
      if _trigger <= clock:
        live.add(_path)
        arrivals.pop(_path, None)
    sub = _file_subcommand(cmd)
    if not sub:
      continue
    step_no = entry.get("step_number", "?")
    if sub == "move":
      src = _normalize_ws_path(_extract_flag_value(cmd, "--src") or "")
      dst = _normalize_ws_path(_extract_flag_value(cmd, "--dst") or "")
      if src and src not in live:
        violations.append(f"step {step_no}: 'file move --src {src}'")
      live.discard(src)
      if dst:
        live.add(dst)
      continue
    path = _normalize_ws_path(_extract_flag_value(cmd, "--path") or "")
    if not path:
      continue
    if sub in _FILE_PATH_REQUIRED_SUBCOMMANDS and path not in live:
      if path in arrivals:
        early.append(
            f"step {step_no}: 'file {sub} --path {path}' runs at virtual time"
            f" {clock} but that file only arrives at {arrivals[path]}"
        )
      else:
        violations.append(f"step {step_no}: 'file {sub} --path {path}'")
    if sub == "create":
      live.add(path)
      arrivals.pop(path, None)
    elif sub == "delete":
      live.discard(path)
  if early:
    raise ValueError(
        "File path grounding (timing): these commands read a file before the"
        f" background event that creates it has fired -- {'; '.join(early)}."
        " Virtual time advances ONLY via 'wait <seconds>', so insert waits"
        " totalling at least the arrival time BEFORE the step that reads the"
        " file, or drop that read. Reading early fails with 'File not found',"
        " emits no effect and grades as nothing."
    )
  if violations:
    available = ", ".join(sorted(set(existing_paths) - set(arrivals))[:40]) or "(none)"
    raise ValueError(
        "File path grounding: these commands target a file that does not exist"
        f" at that point in the trajectory -- {'; '.join(violations)}."
        " 'file read', 'file append', 'file delete' and 'file move --src' all"
        " FAIL with 'File not found' on a path that was never created, so the"
        " step produces no output and no effect and can never be graded. Use"
        " ONLY these existing workspace paths:"
        f" [{available}] -- or create the file first with"
        " \"file create --path '<path>' --content '<text>'\" earlier in the same"
        " trajectory, and only then read or append to it."
    )


def _reground_missing_file_reads(
    commands: list[dict[str, Any]],
    ordered_paths: list[str],
) -> int:
  """Last-attempt escape: retarget file commands whose path cannot exist.

  ``_validate_file_path_grounding`` is a hard gate for the first attempts, which
  is where it belongs -- the author LLM should re-write a coherent trajectory
  once it is told which paths are real. But the steps budget is only 3 attempts
  and exhausting it raises ``ScenarioLLMError``, which discards the whole
  SCENARIO, not the round. Measured on v4, 9.5% of rounds would enter that gate,
  spread over 14 of 60 seeds, so a pure-reject policy trades a grading defect for
  a much larger loss of corpus diversity.

  So on the final attempt the offending command is repointed at a real artifact,
  chosen deterministically and with no guessing about content: the most recent
  file the trajectory itself wrote before that step, else the most recent file it
  successfully read, else the newest workspace file. This is the same fallback
  ladder ``_ensure_final_verification_command`` already uses for the inspect it
  synthesizes. A ``delete``/``move --src`` of a non-existent path has no
  meaningful substitute and is dropped instead (neither appears in either
  corpus). The step stays an inspection of a real file, and because the checker
  scaffold is derived from the FINAL command list, its checks stay consistent
  with the repaired gold by construction.

  ``commands`` is mutated in place. Returns the number of repaired commands.
  """
  live = set(ordered_paths)
  newest = ordered_paths[-1] if ordered_paths else ""
  last_written = ""
  last_read = ""
  repaired = 0
  survivors: list[dict[str, Any]] = []
  for entry in commands:
    if not isinstance(entry, dict):
      survivors.append(entry)
      continue
    cmd = str(entry.get("command") or "")
    sub = _file_subcommand(cmd)
    if not sub:
      survivors.append(entry)
      continue
    if sub == "move":
      src = _normalize_ws_path(_extract_flag_value(cmd, "--src") or "")
      dst = _normalize_ws_path(_extract_flag_value(cmd, "--dst") or "")
      if src and src not in live:
        repaired += 1
        continue  # drop
      live.discard(src)
      if dst:
        live.add(dst)
        last_written = dst
      survivors.append(entry)
      continue
    path = _normalize_ws_path(_extract_flag_value(cmd, "--path") or "")
    if not path:
      survivors.append(entry)
      continue
    if sub == "create":
      live.add(path)
      last_written = path
    elif sub == "append":
      # ``_ensure_file_writes_materialize`` runs first and converts every
      # append-to-missing into a create, so a missing target cannot reach here.
      last_written = path
    elif sub == "delete":
      if path not in live:
        repaired += 1
        continue  # drop
      live.discard(path)
    elif sub == "read":
      if path not in live:
        target = last_written or last_read or newest
        if not target:
          repaired += 1
          continue  # nothing real to read: drop
        entry["command"] = f"file read --path {shlex.quote(target)}"
        repaired += 1
        path = target
      last_read = path
    survivors.append(entry)
  if repaired:
    commands[:] = survivors
    for i, entry in enumerate(commands, 1):
      if isinstance(entry, dict):
        entry["step_number"] = i
  return repaired


def _ensure_wait_before_delayed_reads(
    commands: list[dict[str, Any]],
    arrivals: dict[str, int],
) -> int:
  """Insert the ``wait`` a mid-round file arrival requires, before its reader.

  The unique deterministic repair for the timing arm of
  ``_validate_file_path_grounding``: the trajectory already says *what* to read,
  and the environment already says *when* the file lands, so the only missing
  token is the ``wait`` that carries the clock from here to there. Repointing the
  read at some other artifact (what ``_reground_missing_file_reads`` does for a
  path that never exists) would be strictly worse here -- the evidence the round
  was designed around does arrive, just later.

  Prefer this over dropping the step: it preserves the authored trajectory and
  the ``wait`` it adds is itself a graded temporal action. Steps are renumbered
  after insertion so downstream check derivation keeps its 1-based ordering.
  """
  if not arrivals:
    return 0
  pending = dict(arrivals)
  clock = 0
  inserted = 0
  out: list[dict[str, Any]] = []

  def _land(now: int) -> None:
    for path, trigger in list(pending.items()):
      if trigger <= now:
        pending.pop(path, None)

  for entry in commands:
    if not isinstance(entry, dict):
      out.append(entry)
      continue
    cmd = str(entry.get("command") or "")
    clock += _wait_seconds(cmd)
    _land(clock)
    sub = _file_subcommand(cmd)
    path = (
        _normalize_ws_path(_extract_flag_value(cmd, "--path") or "")
        if sub
        else ""
    )
    if sub in _FILE_PATH_REQUIRED_SUBCOMMANDS and path in pending:
      delta = pending[path] - clock
      if delta > 0:
        out.append({"step_number": 0, "command": f"wait {delta}"})
        clock += delta
        inserted += 1
        _land(clock)
    if sub == "create" and path:
      pending.pop(path, None)
    out.append(entry)
  if inserted:
    commands[:] = out
    for i, entry in enumerate(commands, 1):
      if isinstance(entry, dict):
        entry["step_number"] = i
  return inserted


def _assets_visible_in_round(
    assets: Any,
    round_idx: int,
) -> list[dict[str, Any]]:
  """The assets a round can actually open and read.

  ``available_from_round`` is the only thing deciding whether a player ever
  sees an asset in a given round, and the round loop asks that question in
  four places: it spelled the predicate out inline in three of them and forgot
  it entirely in the fourth -- the list handed to the resolver as groundable
  evidence. The cost of that omission is measured: 8 corpus rounds ship a
  ``resolved_value_locus`` naming an asset whose availability starts AFTER
  that round, and their 216 graded observations average 0.3731 against 0.6590
  for clean rounds. Naming the predicate once is what makes a site that skips
  it visible.

  Availability is read exactly as ``vlm_spec.VLMAssetSpec.from_dict`` reads it
  (``int(x or 1)``), so an asset that never declared a round stays visible
  from round 1 here and downstream alike. Returns the ORIGINAL dicts in their
  original order -- callers read ``caption``/``ocr_text`` straight off them.
  """
  visible: list[dict[str, Any]] = []
  for asset in (assets or []):
    if not isinstance(asset, dict):
      continue
    if int(asset.get("available_from_round") or 1) <= round_idx:
      visible.append(asset)
  return visible


def _asset_kind_by_id(assets: Any, asset_id: str) -> str:
  """The ``kind`` of the asset a locus names, or '' when it names nothing.

  Read off the SAME list the resolver was shown, so a locus naming an asset
  this round cannot open resolves to '' and the decisive-carrier gate abstains
  instead of guessing: the visibility check inside
  ``_validate_round_resolution`` already owns that failure and reports it with
  the right message. Abstaining is also the safe default for an unrecognised
  kind -- the gate only ever fires on a POSITIVE 'rich_text' match.
  """
  want = (asset_id or "").strip()
  if not want:
    return ""
  for asset in (assets or []):
    if isinstance(asset, dict) and str(asset.get("id") or "").strip() == want:
      return str(asset.get("kind") or "").strip()
  return ""


def _format_round_carrier_directive(
    round_outline: dict[str, Any] | None,
    visible_assets: list[dict[str, Any]] | None,
) -> str:
  """Tell the round asset step whether it is the carrier or one of the foils.

  This step is where most carriers are actually born. Of the 523 corpus rounds
  that named a value, 427 had it printed on an asset the round could see; 241
  of those (56.4%) had every visible carrier introduced at that very round and
  another 131 (30.7%) had at least one, so 87.1% of carried rounds depend on
  it. Its prompt said nothing about the decisive value at all.

  But it must not simply inherit the phase brief. The phase step was already
  told to carry the value; repeating that order here mints a second carrier
  for free, and a second visible carrier is measurably easier -- 0.6310 for
  one against 0.7545 for two-to-four, holding value reuse and value length
  fixed. The corpus shows the leak has already happened: only 209 of 523
  rounds (40.0%) have exactly one visible carrier and 56 (10.7%) have six or
  more.

  So the two jobs are opposites, and which one applies is a fact the pipeline
  already holds: does an asset visible in this round literally print the
  value? That is the same folded-substring predicate
  ``_validate_round_resolution`` enforces and ``FileChecker`` grades with --
  reused here to ROUTE a prompt, not to judge or patch a result. Returns ""
  when S1 decided nothing, so a degraded phase never leaves this step under
  orders about a value that does not exist.
  """
  value = str((round_outline or {}).get("decisive_value") or "").strip()
  if not value:
    return ""
  label = str((round_outline or {}).get("decisive_label") or "").strip()
  needle = _fold_for_grounding(value)
  carriers = [
      str(a.get("id") or "")
      for a in (visible_assets or [])
      if isinstance(a, dict) and needle in _fold_for_grounding(
          _asset_grounding_blob(a))
  ]
  carriers = [c for c in carriers if c]

  head = (
      "CRITICAL (decisive evidence): this round's decisive reading has already"
      f" been chosen, and it is: {value}"
      + (f"   (the player reads it from: {label})" if label else "")
      + "\n"
  )
  if carriers:
    named = ", ".join(carriers[:3])
    return head + (
        f"It is ALREADY printed on an asset this round can see: {named}. Do"
        " not print it again. The assets you create here are the sources that"
        " DISAGREE with it -- give them different readings of the same kind,"
        " close enough in magnitude to be genuinely confusable. Every extra"
        " surface repeating the same value is one less surface the player has"
        " to reconcile. Repeat it only if this round's beat is explicitly two"
        " sources corroborating against a third, and never beyond that one"
        " twin.\n"
    )
  return head + (
      "No asset this round can already see prints it, so exactly ONE of the"
      " assets you create here must carry it VERBATIM in its caption or"
      " ocr_text -- character for character, including units, separators,"
      " case and brackets ('1,420 queries/min' is not '1420 queries per"
      " minute'). It must be readable in THIS round: leave its"
      " available_from_round at the round number already filled into the JSON"
      " schema below, and never defer it to a later round -- a reading the"
      " player cannot open yet is not evidence, and the round loses the value"
      " it was built around. Bury it: that carrier must show at least 8 other"
      " readings of"
      " the same kind, several of them close in magnitude, so it is not the"
      " only number on the surface. Do not highlight it -- no bold, no arrows,"
      " no 'NOTE:' or 'CRITICAL:', no callout in the description, and do not"
      " place it first or last. Every OTHER asset you create must show a"
      " DIFFERENT reading; those are the sources that disagree, and that"
      " disagreement is the conflict the player has to resolve.\n"
  )


def _format_decisive_stakes(round_outline: dict[str, Any] | None) -> str:
  """Order the round's prose to leave the decision genuinely unsettled.

  This is the positive dual of ``_format_decisive_secrecy_guard``. The guard
  stops the text from PRINTING the reading; this stops the text from ANSWERING
  the round without it.

  Measured on the last corpus run, splitting each run by what it could
  actually reach before it wrote anything:

  * 548 runs opened no asset and had no decisive value in any output they had
    seen -- prose only. Their verdict check passed 88.9% of the time.
  * 8,341 runs did open evidence. 88.4%.

  Opening the evidence was worth -0.5 percentage points. Some scenarios
  (``access_request_approval_v03``, ``employee_onboarding_workflow_v02``,
  ``multi_tab_slot_monitoring_v02``) were solved from prose alone every single
  time. Nor is the verdict simply a coin flip: only 16 of 58 scenarios reuse
  two or fewer distinct ``decision_outcome`` values across their rounds.

  The obvious explanation -- the value leaking into the text -- is measurably
  NOT the mechanism. The decisive string appears in 0.6% of round instructions
  (78/13,801) and is readable in pre-write stdout in 3.3% of runs
  (458/13,816), and S5's guard already targets that path. What remains is that
  the prose settles the conflict by how it is written.

  The concrete handle below -- write the side that loses just as credibly as
  the side that wins -- is a hypothesis about that mechanism, not a measured
  cause. It is stated as an operational requirement because "leave the
  decision open" alone is too abstract for an author to act on.

  Returns "" when S1 decided nothing, so a degraded phase never leaves a
  sub-step balancing a conflict whose decisive reading does not exist. It
  deliberately does NOT restate the value: the guard above already names it
  once, and repeating it under a positive instruction is the likeliest way to
  licence writing it.
  """
  value = str((round_outline or {}).get("decisive_value") or "").strip()
  if not value:
    return ""
  label = str((round_outline or {}).get("decisive_label") or "").strip()
  subject = label or "the quantity that reading measures"
  return (
      "CRITICAL (the decision must stay open): the conflict this round grades"
      f" turns on {subject}, and the reading named just above is what settles"
      " it.\n"
      "Measured on the last run of this generator: agents that never opened a"
      " single asset still reached the correct verdict 88.9% of the time (548"
      " runs), against 88.4% for the 8,341 runs that did open the evidence."
      " Looking at the evidence was worth -0.5 percentage points, because the"
      " text was already answering the question.\n"
      "So your text MUST leave the decision genuinely unsettled. Do not state,"
      " imply or foreshadow how this round resolves. No participant may be"
      " written as obviously right or obviously wrong, and no line may hint"
      " which source will turn out to be the trustworthy one.\n"
      "Concretely: you know which side that reading favours, so you know which"
      " side is mistaken -- write the mistaken side just as credibly as the"
      " correct one. Equal specificity, equal confidence, equal corroboration,"
      " equal seniority. If the mistaken party sounds vague, hedging or"
      " flustered while the correct one sounds precise and calm, tone alone"
      " gives the answer away and the player never has to look.\n"
      "Knowing which side is mistaken is the ONLY reason you were told the"
      " reading. Build that symmetry without writing the reading or any"
      " reformatting of it -- the confidentiality rule above still binds"
      " absolutely.\n"
  )


def _format_decisive_secrecy_guard(round_outline: dict[str, Any] | None) -> str:
  """Name the decided value to one text sub-step so it can refuse to write it.

  Deciding the value in S1 buys the asset authors a target to carry, but the
  same ``round_outline`` is handed to the three sub-steps that write the player
  instruction, the teammate chatter and the async updates -- exactly the
  surfaces the confidentiality gate forbids it to appear on. A round whose
  instruction quotes the reading is solvable without opening the evidence.

  It is also self-defeating: a leak makes ``_validate_round_confidentiality``
  reject every bound resolution attempt, the binding is released, and the round
  falls back to a free pick -- silently bypassing S1, S2 and S5 for that round.

  The existing lines in those prompts are generic ("do not write any specific
  decisive reading"), which asks the model to infer what counts. Since the
  string is known by then, hand it over: an exact string is a far easier
  constraint to obey than a category. Returns "" when nothing was decided, so
  no prompt grows a dangling order to avoid a value that was never chosen.
  """
  value = str((round_outline or {}).get("decisive_value") or "").strip()
  if not value:
    return ""
  label = str((round_outline or {}).get("decisive_label") or "").strip()
  return (
      "CRITICAL (confidentiality -- the exact string): this round's decisive"
      f" reading has already been chosen, and it is: {value}"
      + (f"   (the player reads it from: {label})" if label else "")
      + "\nIt is printed on a visual asset, and finding it there IS the round."
      " Your text MUST NOT contain that string. It MUST NOT contain a"
      " reformatting of it either -- different units, separators, spacing,"
      " case, rounding or spelled out in words all count as writing it. Do not"
      " let a character quote it, confirm it, or narrow it down to it. Refer to"
      " it only indirectly ('the figure on the dashboard', 'whatever the panel"
      " reports'). Writing it makes the round solvable without looking at the"
      " evidence, and the round will be rejected and regenerated.\n"
  )


def _format_decisive_brief(round_outlines: list[dict[str, Any]]) -> str:
  """Brief the asset steps on the values S1 already decided for these rounds.

  Returns "" when nothing was decided (the decisive half degraded), so the
  asset prompt never grows a header with no values under it.

  The two rules below pull in opposite directions and both are load-bearing:

  * VERBATIM, or the round has no exact anchor at all. 82 of 523 corpus rounds
    named a value that appeared in no asset's caption, ocr_text or
    description; they passed their decisive checks 2.8% of the time against
    53.1% for rounds whose value was findable.

  * FEW carriers, buried in a dense surface, or the search disappears.
    Across 9,267 graded findable rounds, score climbs with the number of
    assets repeating the value -- 1 carrier 0.6336, 2-4 carriers 0.7137, 5-9
    carriers 0.7424 -- and the gradient survives every control available:
    restricted to values no other round reuses it is 0.6314 -> 0.7331, and
    restricted further to values of 15+ characters it is 0.6310 -> 0.7545. So
    it is not an artifact of the short-passcode genre; repeating the value
    really does cost 0.05-0.12 of difficulty.

    It is deliberately NOT capped, and only discouraged in the prompt. A hard
    "exactly one" would forbid a legitimate and common reconciliation shape --
    two sources agreeing against a third that disagrees -- which the decisive
    half explicitly asks for. Difficulty and diversity genuinely conflict
    here; the floor (the value must appear at all) is the part that is a bug
    fix, the ceiling is a design tradeoff left open pending carrier-count data
    from a run made under this contract.

  The density floor (8 sibling readings) is the 10th percentile of the
  measured carrier distribution -- corpus carriers hold a median 17 numeric
  readings across a median 857 characters, with the decisive value occupying
  a median 2.1% of the text. It is a floor, not a target.
  """
  decided = [
      r for r in round_outlines
      if isinstance(r, dict) and str(r.get("decisive_value") or "").strip()
  ]
  if not decided:
    return ""

  lines = [
      "=== DECISIVE VALUES DECIDED FOR THIS PHASE ===",
      "These values are already fixed. Your job is to build the evidence that"
      " carries them; do NOT invent different ones.",
  ]
  for r in decided:
    label = str(r.get("decisive_label") or "").strip()
    lines.append(
        f"  - {r.get('round_id')}: {r['decisive_value']}"
        + (f"   (read from: {label})" if label else "")
    )
  lines += [
      "RULES FOR CARRYING THEM:",
      " - VERBATIM. The value must appear in an asset's caption or ocr_text"
      " character for character, including units, separators, case and"
      " brackets. '1,420 queries/min' is not '1420 queries per minute'.",
      " - ONE PRIMARY CARRIER. Put the value on the asset it is genuinely read"
      " from, and that asset's available_from_round must be at or before its"
      " round. A second surface may show the same value ONLY when the conflict"
      " structure actually calls for corroboration (two sources agreeing"
      " against a third that disagrees) -- never as insurance to make sure it"
      " appears somewhere. Every extra surface printing the same value is one"
      " less surface the player has to reconcile.",
      " - BURY IT. The carrier must be a realistically dense readout, not a"
      " panel built around the answer: put at least 8 other readings of the"
      " same kind on the same surface, several of them close in magnitude to"
      " the decisive one, so it does not stand out by being the only number"
      " there.",
      " - Do not highlight it. No bold, stars, arrows, 'NOTE:', 'CRITICAL:',"
      " no callout in the description, no ordering that puts it first or"
      " last. It must read as one ordinary row among many.",
      " - The surfaces that DISAGREE with it must show DIFFERENT readings --"
      " that disagreement is the conflict the player has to resolve. Only the"
      " true value above is printed verbatim.",
      "=== END DECISIVE VALUES ===",
  ]
  return "\n".join(lines) + "\n"


def _format_retry_feedback(errors: list[str] | None) -> str:
  """Render EVERY prior attempt's error, not just the most recent one.

  The bounded retry loops originally kept a single ``error_feedback = str(e)``
  slot that was overwritten on each failure, so attempt N only ever saw
  attempt N-1's rejection and nothing stopped it from re-proposing a value
  that attempt N-2 had already had rejected. Measured on the 67 degraded
  rounds of the 3.6-flash batch: 56 of them (83.6%) had a third attempt that
  returned the SAME resolved_value as its own first attempt.

  Duplicates are deliberately NOT collapsed -- the identical error twice is
  precisely the "you are looping" signal the model needs to see.

  A single error renders bare (byte-identical to the old behaviour, since
  every call site already prefixes its own "PREVIOUS ATTEMPT FAILED" header),
  so this only adds information from the third attempt onward. Returns "" for
  an empty history so call sites can keep guarding on plain truthiness.
  """
  cleaned = [str(e).strip() for e in (errors or []) if str(e).strip()]
  if not cleaned:
    return ""
  if len(cleaned) == 1:
    return cleaned[0]
  numbered = "\n".join(f"  {i}. {e}" for i, e in enumerate(cleaned, 1))
  return (
      f"ALL {len(cleaned)} PREVIOUS ATTEMPTS FAILED -- do not repeat any of"
      f" them:\n{numbered}\n"
      "Every numbered item above is a rejection you have ALREADY received."
      " Your new output must differ from all of them."
  )


class ConflictLLMPipeline(ScenarioPipelineBase):
  """Generate conflict-focused multi-agent scenarios."""

  # Channel-fairness defect counters for the run's acceptance report. Both
  # record a channel decision that was made FOR the author because it ran out of
  # retries, which is exactly the class of defect that shipped 433 phantom gold
  # channels and 320 phantom criteria in v7. Kept as counts rather than hard
  # failures on purpose: aborting the scenario would cost a whole conflict
  # archetype from the corpus, and corpus diversity is the more expensive thing
  # to lose. Class-level defaults so no ``__init__`` override is needed; reset
  # per scenario in ``generate``.
  _unsatisfiable_criterion_channels: int = 0
  _blind_bound_channels: int = 0

  def _save_checkpoint(
      self,
      checkpoint_path: Path,
      layer0_phases: dict[str, Any] | None,
      layer1_payload: dict[str, Any] | None,
      layer1b_payload: dict[str, Any] | None,
      updates: list[dict[str, Any]],
      rounds: list[dict[str, Any]],
      evals: list[dict[str, Any]],
      phase_expanded_assets: list[dict[str, Any]],
      expanded_rounds: dict[str, dict[str, Any]],
  ) -> None:
    data = {
        "layer0_phases": layer0_phases,
        "layer1_payload": layer1_payload,
        "layer1b_payload": layer1b_payload,
        "updates": updates,
        "rounds": rounds,
        "evals": evals,
        "phase_expanded_assets": phase_expanded_assets,
        "expanded_rounds": expanded_rounds,
    }
    try:
      checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
      with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
      self._log(f"Warning: Failed to save pipeline checkpoint: {e}")

  def _fallback_round_outline(
      self,
      round_id: str,
      current_phase: dict[str, Any],
      seed: ConflictVLMSeed,
  ) -> dict[str, Any]:
    """Synthesize a minimal round outline when phase expansion fails.

    Used so a single failed ``conflict_layer0_expand_phase`` call degrades one
    phase to a bare-bones outline instead of discarding the entire scenario.
    Fields mirror the shape produced by a successful expansion so downstream
    round-narrative generation has the context keys it expects.
    """
    phase = current_phase if isinstance(current_phase, dict) else {}
    description = (
        phase.get("narrative_beat")
        or phase.get("description")
        or phase.get("theme")
        or f"Advance the core conflict during {round_id}."
    )
    participants = (
        phase.get("active_participants")
        or list(getattr(seed, "participants", []) or [])
    )
    return {
        "round_id": round_id,
        "phase": phase.get("theme", "action"),
        "description": description,
        "active_participants": participants,
        "introduced_anchors": [],
        "_synthesized_fallback": True,
    }

  def _generate_with_retry(
      self,
      build_fn: Callable[[int, str], Any],
      *,
      attempts: int,
      label: str,
      failure_message: str,
      include_last_error: bool = False,
  ) -> Any:
    """Run ``build_fn(attempt, error_feedback)`` with bounded validation retries.

    Factors out the screenplay/narrative retry scaffolding that was duplicated
    across the generation stages. ``build_fn`` receives the 1-based attempt
    number and EVERY earlier attempt's error, rendered by
    ``_format_retry_feedback`` ("" on the first attempt), and returns its
    successful result. Every raised exception is accumulated as
    feedback and logged with the same ``[<label> Attempt N Failed]: <e>.
    Retrying...`` line the inline loops used. Once ``attempts`` failures
    accumulate a ``ScenarioLLMError(failure_message)`` is raised; when
    ``include_last_error`` is set the final error (truncated to 800 chars) is
    appended, preserving the round loops' ``Last error: ...`` diagnostics.
    """
    attempt_errors: list[str] = []
    for attempt in range(1, attempts + 1):
      try:
        return build_fn(attempt, _format_retry_feedback(attempt_errors))
      except Exception as e:  # noqa: BLE001 - retried as validation feedback
        attempt_errors.append(str(e))
        self._log(f"  [{label} Attempt {attempt} Failed]: {e}. Retrying...")
    if include_last_error:
      last_error = attempt_errors[-1] if attempt_errors else ""
      raise ScenarioLLMError(
          f"{failure_message} Last error: {last_error[:800]}"
      )
    raise ScenarioLLMError(failure_message)

  def generate(
      self,
      seed: ConflictVLMSeed,
      checkpoint_dir: Path | None = None,
      resume: bool = False,
      phase_expand_attempts: int | None = None,
  ) -> dict[str, Any]:
    """Generate the full scenario data including subagents and updates."""
    self._unsatisfiable_criterion_channels = 0
    self._blind_bound_channels = 0
    seed_dict = dataclasses.asdict(seed)
    # De-contaminate the seed the LLM sees: generation_policy carries backend
    # provider flags (chart_provider='template', natural_image_provider='mock',
    # google_model='gemini-...') that the model tends to copy verbatim into asset
    # platform_ids. They are irrelevant to scenario authoring and are applied from
    # the real seed at render time, so strip them from this payload copy only
    # (asdict returned a fresh dict; the real seed is untouched).
    _gen_policy = seed_dict.get("generation_policy")
    if isinstance(_gen_policy, dict):
      for _provider_key in ("chart_provider", "natural_image_provider", "google_model"):
        _gen_policy.pop(_provider_key, None)
    # ``require_protocol_completion`` is a grading/validation-side control (the
    # strict validator asserts the gold trajectory completes the mechanic); the
    # LLM never authors against it -- steering is the archetype plugin's job. Strip
    # it from this prompt copy so enabling the flag on a seed leaves every LLM
    # prompt (and therefore the generated bytes) unchanged (E6).
    seed_dict.pop("require_protocol_completion", None)
    # ``force_async_wait`` (M8/CL2) is likewise a generation/grading-side knob: it
    # steers the round resolver + checker scaffolder + strict validator (require a
    # delegate->wait->read reference and a temporal check), not the prose the LLM
    # authors. Strip it too so toggling it leaves every prompt payload byte
    # identical (E6); the real seed is untouched and still drives the resolver.
    seed_dict.pop("force_async_wait", None)
    # M1/M2/M3 are likewise not for the bulk seed payload, and these three MUST
    # be popped unconditionally (not just when set): ``dataclasses.asdict``
    # emits every field, so leaving them in would change the prompt bytes of
    # every seed the moment the fields exist, including seeds that never opt in
    # -- exactly the E6 regression the pops above exist to prevent.
    #   M1 IS generation-side, but it is delivered per-round as a concrete
    # assigned type (see ``_format_round_type_directive``); a bare boolean here
    # would tell the LLM nothing and only add noise.
    #   M2/M3 are materialization-side and never reach the LLM at all.
    seed_dict.pop("force_round_type_plan", None)
    seed_dict.pop("force_decoy", None)
    seed_dict.pop("force_no_signpost", None)
    target_rounds = seed.target_rounds

    # Build Entity Registry mapping first/short names to full participant IDs
    character_map = {}
    for p in seed.participants:
      short_name = p.split("_")[0]
      character_map[short_name] = p
      # Also add role prefix if it has underscores but model refers to it differently
      parts = p.split("_")
      if len(parts) > 1:
        character_map[parts[0]] = p

    checkpoint_path = None
    if checkpoint_dir:
      checkpoint_path = checkpoint_dir / f"{seed.slug}_pipeline_checkpoint.json"

    # Default empty states
    layer0_phases = None
    layer1_payload = None
    layer1b_payload = None
    updates = []
    rounds = []
    evals = []
    phase_expanded_assets = []
    expanded_rounds = {}

    presets_help = self._presets_prompt_instructions()
    max_attempts = 3
    # Phase expansion is the single most abort-prone step (one failure used to
    # discard the whole scenario), so give it a larger, independently
    # configurable retry budget.
    phase_attempts = (
        phase_expand_attempts
        if phase_expand_attempts is not None
        else max(max_attempts, 6)
    )

    # Load checkpoint if resuming
    if resume and checkpoint_path and checkpoint_path.is_file():
      self._log(f"[Resume] Loading pipeline checkpoint from {checkpoint_path}...")
      try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
          ckpt = json.load(f)
        layer0_phases = ckpt.get("layer0_phases")
        layer1_payload = ckpt.get("layer1_payload")
        layer1b_payload = ckpt.get("layer1b_payload")
        updates = ckpt.get("updates", [])
        rounds = ckpt.get("rounds", [])
        evals = ckpt.get("evals", [])
        phase_expanded_assets = ckpt.get("phase_expanded_assets", [])
        expanded_rounds = ckpt.get("expanded_rounds", {})
        self._log(
            f"[Resume] Restored state: rounds_count={len(rounds)},"
            f" phases_loaded={bool(layer0_phases)}"
        )
      except Exception as e:
        self._log(f"[Resume] Failed to load checkpoint: {e}. Starting fresh...")

    active_slugs = (seed.conflict_taxonomy or []) + (seed.difficulty_challenges or [])
    plugins = get_plugins_for_taxonomy(active_slugs)
    # The conflict mechanic's command vocabulary is a property of the scenario
    # (plugins x allowed families), not of a round or a layer, so derive it once
    # here and reuse it for the world-building hooks below and the per-round
    # gates further down. Empty for the 10 seeds that route to no mechanic
    # plugin, which makes every mechanic injection and gate a no-op for them.
    mech = _mechanic_command_sets(plugins, seed.allowed_tool_families)
    multi_instance_hook = _multi_instance_world_instruction(mech)

    layer0_hooks = "\n".join(f"- {p.get_layer0_instruction(seed)}" for p in plugins if p.get_layer0_instruction(seed))
    layer1_hooks = "\n".join(f"- {p.get_layer1_instruction(seed)}" for p in plugins if p.get_layer1_instruction(seed))
    layer1b_hooks = "\n".join(f"- {p.get_layer1b_instruction(seed)}" for p in plugins if p.get_layer1b_instruction(seed))
    trajectory_hooks = "\n".join(f"- {p.get_trajectory_instruction(seed)}" for p in plugins if p.get_trajectory_instruction(seed))
    evals_hooks = "\n".join(f"- {p.get_evals_instruction(seed)}" for p in plugins if p.get_evals_instruction(seed))

    if multi_instance_hook:
      # One requirement, injected at the three layers that each own a piece of
      # it: layer 0 plans the beat where the coordinator works out WHICH
      # instance is in play, layer 1b plants the candidates and the evidence
      # that singles one out, and the trajectory has to act on that one (the
      # checker binds the identifier, so a gold trajectory that wandered onto a
      # decoy would author a check the scenario's own evidence argues against).
      layer0_hooks += (
          "\n- The scenario contains SEVERAL candidate targets of the same kind "
          "and only one is correct; the phase plan must include a beat where the "
          "coordinator works out WHICH one the conflict is about before acting "
          "on it, and must not reveal the answer by having only one candidate."
      )
      layer1b_hooks += f"\n- {multi_instance_hook}"
      trajectory_hooks += (
          "\n- CORRECT TARGET: several plausible candidates exist for the "
          "protocol's target flags and only one is right. Every protocol command "
          "in the expert trajectory must name the candidate the shipped evidence "
          "singles out -- never a decoy, and never a placeholder. Where the "
          "decisive action requires a set of preconditions to be complete, the "
          "trajectory must satisfy every member of that set before invoking it."
      )

    has_run_cmd = "shell_scripting" in seed.allowed_tool_families or "run_command" in seed.allowed_tool_families
    capability_negatives = ""
    if not has_run_cmd:
      capability_negatives = (
          "\nCRITICAL: The coordinator agent (the player) does NOT have terminal/command-line access, and cannot execute shell commands or run python/bash scripts directly. "
          "You MUST NOT design any screenplay beats, conflict triggers, or actions that require the player to run scripts (e.g. 'python decrypt.py'), compile code, or calculate SHA256/MD5 checksums of files. "
          "All tasks must be solved via simple configuration edits, Slack message alignment, or reading details from visual charts."
      )

    # Stage 1: High-level phase planning (Multi-Step Screenplay Planning)
    if not layer0_phases:
      # 1a: Screenplay
      self._log("[1a/4] Drafting screenplay characters and conflict triggers...")
      def _gen_l0a(attempt, error_feedback):
        # Tier 2.1 (design 5.1): author the character cast first, then author
        # the conflict triggers referencing that finalized cast, then merge
        # back into the single screenplay_outline shape. Both calls share this
        # retry loop (independent retry granularity is deferred to Tier 3.2).
        cast_instruction = (
            "Draft the high-level narrative screenplay CAST for a multi-agent conflict scenario.\n"
            "The 'screenplay_outline' MUST define ONLY:\n"
            "1. 'conflict_arc': Description of the logical mismatch or conflict tension.\n"
            "2. 'character_cast': list of characters mapped strictly from seed participants (each specifying role_name, dramatic_role, stance, and motivation).\n"
            "Do NOT emit 'narrative_conflict_triggers' in this step; triggers are authored separately."
        )
        if capability_negatives:
          cast_instruction += f"\n{capability_negatives}"
        if layer0_hooks:
          cast_instruction += f"\n\nCONFLICT & DIFFICULTY CONSTRAINTS:\n{layer0_hooks}"
        if error_feedback:
          cast_instruction += f"\n\n⚠️ PREVIOUS ATTEMPT FAILED WITH VALIDATION ERROR:\n{error_feedback}\nPlease correct this error in your new output."

        l0a_cast = self.client.generate_layer(
            "conflict_layer0a_cast",
            {
                "seed": seed_dict,
                "slot": {
                    "instruction": cast_instruction
                }
            }
        )

        finalized_outline = dict(l0a_cast.get("screenplay_outline") or {})
        cast_role_names = [
            c.get("role_name")
            for c in (finalized_outline.get("character_cast") or [])
            if isinstance(c, dict) and c.get("role_name")
        ]

        triggers_instruction = (
            "Given the FINALIZED character cast below, design the narrative conflict triggers.\n"
            f"--- FINALIZED CHARACTER CAST role_names ---\n{cast_role_names}\n--- END CAST ---\n"
            "'narrative_conflict_triggers': list of general conflict triggers (each specifying conflict_name, description, workflow_block, resolution_cues, affected_participants, and evidence_sources).\n"
            "CRITICAL: every 'affected_participants' entry MUST be an exact 'role_name' from the cast above. Do NOT invent, shorten, or rename participants."
        )
        if capability_negatives:
          triggers_instruction += f"\n{capability_negatives}"
        if layer0_hooks:
          triggers_instruction += f"\n\nCONFLICT & DIFFICULTY CONSTRAINTS:\n{layer0_hooks}"
        if error_feedback:
          triggers_instruction += f"\n\n⚠️ PREVIOUS ATTEMPT FAILED WITH VALIDATION ERROR:\n{error_feedback}\nPlease correct this error in your new output."

        l0a_triggers = self.client.generate_layer(
            "conflict_layer0a_triggers",
            {
                "seed": seed_dict,
                "screenplay": l0a_cast,
                "slot": {
                    "instruction": triggers_instruction
                }
            }
        )
        _validate_generated_screenplay_triggers(l0a_cast, l0a_triggers)

        # Merge cast + triggers into the single l0a_res screenplay_outline
        # shape that the timeline/plans layers and checkpoint expect.
        finalized_outline["narrative_conflict_triggers"] = l0a_triggers.get(
            "narrative_conflict_triggers", []
        )
        return {
            "summary": l0a_cast["summary"],
            "screenplay_outline": finalized_outline,
        }
      l0a_res = self._generate_with_retry(
          _gen_l0a,
          attempts=max_attempts,
          label="Stage 1a",
          failure_message=f"Failed to generate screenplay outline after {max_attempts} attempts.",
      )

      # 1b: Timeline
      self._log("[1b/4] Scheduling phases and round-by-round screenplay beats...")
      def _gen_l0b(attempt, error_feedback):
        instruction_str = (
            "Schedule the screenplay timeline into coarse narrative phases and round-by-round beats matching the screenplay draft.\n"
            f"Map the taxonomy and required constraints across {target_rounds} rounds.\n"
            "Each phase inside 'phases' lists start_round, end_round, theme, description, "
            "and a detailed 'timeline_screenplay' array. For each round beat, specify the round number, narrative_beat, "
            "list of active_conflict_names (referencing triggers designed in screenplay), character_claims (claims mapped to character role_names), "
            "vlm_evidence_requirement, and dialogue_conflict."
        )
        if capability_negatives:
          instruction_str += f"\n{capability_negatives}"
        if error_feedback:
          instruction_str += f"\n\n⚠️ PREVIOUS ATTEMPT FAILED WITH VALIDATION ERROR:\n{error_feedback}\nPlease correct this error in your new output."

        return self.client.generate_layer(
            "conflict_layer0b_timeline",
            {
                "seed": seed_dict,
                "screenplay": l0a_res,
                "slot": {
                    "instruction": instruction_str
                }
            }
        )
      l0b_res = self._generate_with_retry(
          _gen_l0b,
          attempts=max_attempts,
          label="Stage 1b",
          failure_message=f"Failed to generate screenplay timeline after {max_attempts} attempts.",
      )

      # 1c: Plans
      # Tier 3.3 (design 6.3): the three global plans are authored in three
      # independently-retried single-plan calls (assets -> updates ->
      # interactions), each aware of the prior finalized plans, then merged into
      # the layer0c shape. Splitting keeps a late failure from re-calling the
      # already-valid earlier plans.
      self._log("[1c/4] Planning assets, updates, and interactions schedules...")

      # Plan 1: global assets plan (independent).
      def _gen_l0c_assets(attempt, error_feedback):
        instruction_assets_plan = (
            "Plan ONLY the global visual assets schedule for the scenario, matching the screenplay and timeline beats.\n"
            "First choose 'platforms': a list of AT LEAST 2 DISTINCT web-UI surfaces that fit this scenario's domain "
            "(2-4 is a good range; add more only when the scenario naturally spans several surfaces), each a short "
            "lowercase slug (e.g. 'ops_console', 'billing_portal', 'status_page'). Do NOT use provider names, generator names, "
            "file names, or agent roles as platforms.\n"
            "'global_assets_plan': list of visual assets to generate (asset_id, available_from_round, kind, purpose, platform_id). "
            "Every asset's platform_id MUST be one of the 'platforms' you chose, and the assets should be SPREAD across those "
            "platforms rather than all placed on one. "
            "You MUST plan at least 3-4 assets, but do NOT force a variety of kinds for its own sake. Choose each asset's kind by "
            "what the evidence ACTUALLY is, so a purely digital scenario may legitimately use ZERO 'natural_image' while a physical "
            "scenario may use several. If the seed provides an 'evidence' list, honor each item's 'real_world_source' when choosing "
            "kinds (dashboard_chart->chart; log/document->rich_text; in_tool_diagram->diagram; physical_photo->natural_image; "
            "hand_sketch->sketch) and never downgrade a 'physical_photo' into a 'diagram'/'chart'. The kinds are: "
            "'chart' (quantitative plots/tables in a UI), 'rich_text' (logs, configs, code, documents), "
            "'natural_image' (a REAL PHOTOGRAPH of a physical object/scene - e.g. a router's status LEDs, a robot cell, an "
            "antenna dish, a server rack), 'diagram' (a clean DIGITAL schematic viewed INSIDE a software tool - e.g. a network "
            "topology, a facility/floor-plan layout, a wiring/one-line diagram, or a process flowchart - rendered as an in-UI "
            "module), and 'sketch' (an informal HAND-DRAWN schematic on paper or a whiteboard, photographed - e.g. an incident "
            "war-room whiteboard sketch or a field technician's notebook diagram). "
            "Pick 'diagram' when the schematic would realistically live inside a tool's UI, and 'sketch' when it is an informal "
            "hand-sketch; decide by context. Never use 'natural_image' for a schematic/screen, and never use 'chart' for a "
            "non-quantitative schematic.\n"
            "DOMAIN-AWARENESS: when this scenario's domain centers on physical equipment or environments (industrial robots / "
            "robot cells, antennas & dishes, hardware / server racks, vehicles, lab instruments or samples, field sites), you "
            "SHOULD include at least one 'natural_image' photograph of the physical subject; and when a round's conflict hinges "
             "on a physically-observable state, make that photo carry the DECISIVE reading by placing the deciding value in its "
             "caption/ocr_text. Keep ZERO 'natural_image' for purely digital domains (dashboards, logs, code, web tooling).\n"
             "PLAN A SURFACE WORTH READING. Later layers pick one asset per round to carry that round's decisive value, and a value "
             "printed as a lone line on a free-form text panel is read at a glance -- measured, such rounds passed 54.6% against "
             "38.5% when the value sat on a quantitative panel. So this plan must supply enough quantitative surfaces (charts, "
             "tables, timelines) or physical photographs for every round to have one. Plan 'rich_text' only where the evidence "
             "genuinely IS a log, a config or a document -- not as the default shape for an information panel."
         )
        if error_feedback:
          instruction_assets_plan += f"\n\n⚠️ PREVIOUS ATTEMPT FAILED WITH VALIDATION ERROR:\n{error_feedback}\nPlease correct this error in your new output."

        return self.client.generate_layer(
            "conflict_layer0c_assets_plan",
            {
                "seed": seed_dict,
                "screenplay": l0a_res,
                "timeline": l0b_res,
                "slot": {
                    "instruction": instruction_assets_plan
                }
            }
        )
      l0c_assets = self._generate_with_retry(
          _gen_l0c_assets,
          attempts=max_attempts,
          label="Stage 1c Assets Plan",
          failure_message=f"Failed to generate screenplay plans after {max_attempts} attempts.",
      )

      # Plan 2: global updates plan (aware of the finalized assets plan).
      def _gen_l0c_updates(attempt, error_feedback):
        instruction_updates_plan = (
            "Given the FINALIZED global assets plan, plan ONLY the global asynchronous updates schedule matching the screenplay and timeline beats.\n"
            "'global_updates_plan': list of updates (update_id, before_round, purpose)."
        )
        if error_feedback:
          instruction_updates_plan += f"\n\n⚠️ PREVIOUS ATTEMPT FAILED WITH VALIDATION ERROR:\n{error_feedback}\nPlease correct this error in your new output."

        return self.client.generate_layer(
            "conflict_layer0c_updates_plan",
            {
                "seed": seed_dict,
                "screenplay": l0a_res,
                "timeline": l0b_res,
                "assets_plan": l0c_assets,
                "slot": {
                    "instruction": instruction_updates_plan
                }
            }
        )
      l0c_updates = self._generate_with_retry(
          _gen_l0c_updates,
          attempts=max_attempts,
          label="Stage 1c Updates Plan",
          failure_message=f"Failed to generate screenplay plans after {max_attempts} attempts.",
      )

      # Plan 3: global interactions plan (aware of assets + updates plans).
      def _gen_l0c_interactions(attempt, error_feedback):
        instruction_interactions_plan = (
            "Given the FINALIZED global assets and updates plans, plan ONLY the global subagent interactions schedule matching the screenplay and timeline beats.\n"
            "'global_interactions_plan': list of subagent interactions (interaction_id, active_subagent, interaction_type, purpose)."
        )
        if error_feedback:
          instruction_interactions_plan += f"\n\n⚠️ PREVIOUS ATTEMPT FAILED WITH VALIDATION ERROR:\n{error_feedback}\nPlease correct this error in your new output."

        return self.client.generate_layer(
            "conflict_layer0c_interactions_plan",
            {
                "seed": seed_dict,
                "screenplay": l0a_res,
                "timeline": l0b_res,
                "assets_plan": l0c_assets,
                "updates_plan": l0c_updates,
                "slot": {
                    "instruction": instruction_interactions_plan
                }
            }
        )
      l0c_interactions = self._generate_with_retry(
          _gen_l0c_interactions,
          attempts=max_attempts,
          label="Stage 1c Interactions Plan",
          failure_message=f"Failed to generate screenplay plans after {max_attempts} attempts.",
      )

      # Merge the three plans back into the single layer0c shape.
      l0c_res = {
          "global_assets_plan": l0c_assets["global_assets_plan"],
          "global_updates_plan": l0c_updates["global_updates_plan"],
          "global_interactions_plan": l0c_interactions["global_interactions_plan"],
      }

      layer0_phases = {
          "summary": l0a_res["summary"],
          "screenplay_outline": l0a_res["screenplay_outline"],
          "phases": l0b_res["phases"],
          # Scenario platform vocabulary: LLM-authored where available, sanitized,
          # and backfilled to a soft minimum. Persisted inside layer0_phases so it
          # survives checkpoint/resume, and consumed by every downstream asset
          # step via self._platforms(layer0_phases).
          "platforms": _build_platform_vocab(
              l0c_assets.get("platforms"), l0c_res["global_assets_plan"]),
          "global_assets_plan": l0c_res["global_assets_plan"],
          "global_updates_plan": l0c_res["global_updates_plan"],
          "global_interactions_plan": l0c_res["global_interactions_plan"],
      }

      if checkpoint_path:
        self._save_checkpoint(
            checkpoint_path,
            layer0_phases,
            layer1_payload,
            layer1b_payload,
            updates,
            rounds,
            evals,
            phase_expanded_assets,
            expanded_rounds,
        )

    # Stage 2: Narrative and mock subagents
    if not layer1_payload:
      # 2a: Subagents
      self._log("[2a/4] Generating participating subagents specs...")
      def _gen_l1_agents(attempt, error_feedback):
        # Tier 2.2 (design 5.2): author the agent roster first (canonical full
        # names), then the mock subagents that must reference exactly those
        # roster agent_roles, then merge into the single l1_agents shape. Both
        # calls share this retry loop (independent retry deferred to Tier 3.2).
        roster_instruction = (
            "Define the participating subagent ROSTER (agents) based strictly on the screenplay_outline character cast.\n"
            "Read the character cast profiles ('character_cast'), stances, and motivations.\n"
            "Generate ONLY 'agents': list of agents (agent_role, agent_name, capabilities, private_knowledge, virtual_duration_seconds). "
            "Each 'agent_role' MUST be the canonical full name. Do NOT emit mock_subagents in this step."
        )
        if error_feedback:
          roster_instruction += f"\n\n⚠️ PREVIOUS ATTEMPT FAILED WITH VALIDATION ERROR:\n{error_feedback}\nPlease correct this error in your new output."

        l1_roster = self.client.generate_layer(
            "conflict_layer1_agent_roster",
            {
                "seed": seed_dict,
                "phases": layer0_phases,
                "slot": {
                    "instruction": roster_instruction
                }
            }
        )

        roster_role_names = [
            a.get("agent_role")
            for a in (l1_roster.get("agents") or [])
            if isinstance(a, dict) and a.get("agent_role")
        ]

        mocks_instruction = (
            "Given the FINALIZED agent roster below, generate the global mock subagent callbacks mapping their private knowledge and stances to typical responses.\n"
            f"--- FINALIZED AGENT ROSTER agent_roles ---\n{roster_role_names}\n--- END ROSTER ---\n"
            "'mock_subagents': list (agent_role, mock_response, capabilities, private_knowledge, virtual_duration_seconds, state_effects).\n"
            "CRITICAL: every 'agent_role' MUST be an exact 'agent_role' from the roster above. Do NOT invent, shorten, or rename agents."
        )
        if error_feedback:
          mocks_instruction += f"\n\n⚠️ PREVIOUS ATTEMPT FAILED WITH VALIDATION ERROR:\n{error_feedback}\nPlease correct this error in your new output."

        l1_mocks = self.client.generate_layer(
            "conflict_layer1_mock_subagents",
            {
                "seed": seed_dict,
                "phases": layer0_phases,
                "agents": l1_roster,
                "slot": {
                    "instruction": mocks_instruction
                }
            }
        )
        _validate_generated_agent_mock_subagents(l1_roster, l1_mocks)

        # Merge roster + mocks into the single l1_agents shape the interactions
        # layer and layer1_payload expect.
        return {
            "agents": l1_roster.get("agents", []),
            "mock_subagents": l1_mocks.get("mock_subagents", []),
        }
      l1_agents = self._generate_with_retry(
          _gen_l1_agents,
          attempts=max_attempts,
          label="Stage 2a",
          failure_message=f"Failed to generate participating subagents after {max_attempts} attempts.",
      )

      # 2b: Interactions
      self._log("[2b/4] Designing subagent interaction and dependency rules...")
      families_set = {normalize_family_name(f) for f in seed.allowed_tool_families or []}
      def _gen_l1_inter(attempt, error_feedback):
        instruction_str = (
            "Design the rules governing subagent interactions and execution dependencies matching the screenplay narrative conflict triggers.\n"
            "CRITICAL: The commands in `dependency_steps` MUST strictly belong to the allowed tool families defined in the seed's allowed_tool_families.\n"
            "Provide:\n"
            "1. 'summary': Overview of the conflict scenario and the preference_rule on which authoritative source/evidence to trust when subagents have out-of-sync context or stale cache.\n"
            f"{layer1_hooks}\n"
            "2. 'interactions': list of interaction rules (interaction_type, active_subagent, trigger_evidence, trigger_source_file, resolution_action, dependency_steps)."
        )
        if error_feedback:
          instruction_str += f"\n\n⚠️ PREVIOUS ATTEMPT FAILED WITH VALIDATION ERROR:\n{error_feedback}\nPlease correct this error in your new output."

        return self.client.generate_layer(
            "conflict_layer1_interactions",
            {
                "seed": seed_dict,
                "phases": layer0_phases,
                "agents": l1_agents,
                "slot": {
                    "instruction": instruction_str
                }
            }
        )
      l1_inter = self._generate_with_retry(
          _gen_l1_inter,
          attempts=max_attempts,
          label="Stage 2b",
          failure_message=f"Failed to generate interactions after {max_attempts} attempts.",
      )

      layer1_payload = {
          "summary": l1_inter.get("summary") or "Verify the anomalous platform configuration.",
          "mock_subagents": l1_agents["mock_subagents"],
          "agents": l1_agents["agents"],
          "interactions": l1_inter["interactions"],
      }

      if checkpoint_path:
        self._save_checkpoint(
            checkpoint_path,
            layer0_phases,
            layer1_payload,
            layer1b_payload,
            updates,
            rounds,
            evals,
            phase_expanded_assets,
            expanded_rounds,
        )

    # Stage 3: Background workspace and sessions
    if not layer1b_payload:
      self._log("[3/4] Generating long-context background workspace and sessions...")
      # Tier 3.2 (design 6.2): files, sessions, and assets each get their own
      # bounded retry loop with independent error_feedback, so a late failure
      # (e.g. assets) no longer forces re-generation of already-valid files or
      # sessions. The sub-calls stay sequential because sessions depend on files
      # and assets depend on both.

      # Sub-step 1: workspace files (independent).
      def _gen_bg_files(attempt, error_feedback):
        self._log(f"  [Files Attempt {attempt}] Creating workspace files...")
        instruction_files = (
            "Create initial codebase files, local configuration files, scripts, and local logs in the workspace.\n"
            f"Create at least {seed.min_workspace_files} different workspace files. Write rich textual content (code, configurations, or trace logs) that sets up the background state matching the screenplay's conflict arc.\n"
            "CRITICAL: Do NOT place remote logs or dashboards in workspace files; they must be visual assets."
        )
        # layer1b is the "plant the conflict evidence in the workspace" hook.
        # It was computed at the top of this method and then never used, so the
        # mechanic-specific evidence each plugin asks for (an invoice whose
        # amount contradicts the transaction log, a pair of tickets that wait on
        # each other, an approval record already past its expiry) was never
        # requested and the background stayed generic. Wired here, mirroring the
        # layer1 hook injection in _gen_l1_inter.
        if layer1b_hooks:
          instruction_files += (
              f"\n\nCONFLICT EVIDENCE TO PLANT (required):\n{layer1b_hooks}\n"
              "These files are the substrate of the conflict: the contradiction"
              " must be discoverable by reading them, not merely asserted in the"
              " narrative. Do NOT label the planted evidence as wrong, stale, or"
              " conflicting -- record each source in the neutral voice it would"
              " really have, and let the disagreement emerge only from comparing"
              " them."
          )
        if error_feedback:
          instruction_files += f"\n\n⚠️ PREVIOUS ATTEMPT FAILED WITH VALIDATION ERROR:\n{error_feedback}\nPlease correct this error in your new output."

        l1b_files = self.client.generate_layer(
            "conflict_layer1b_files",
            {
                "seed": seed_dict,
                "phases": layer0_phases,
                "context": layer1_payload,
                "slot": {
                    "instruction": instruction_files
                }
            }
        )
        # Validate only the workspace portion: sessions=[] satisfies the
        # presence check and omitting the assets key skips asset validation.
        _validate_background_payload(
            {"workspace": l1b_files.get("workspace"), "sessions": []}
        )
        return l1b_files

      l1b_files = self._generate_with_retry(
          _gen_bg_files,
          attempts=max_attempts,
          label="Background Files",
          failure_message=f"Failed to generate background payload after {max_attempts} attempts.",
      )

      # Sub-step 2: chat sessions (depend on the finalized files).
      def _gen_bg_sessions(attempt, error_feedback):
        self._log(f"  [Sessions Attempt {attempt}] Generating chat sessions logs...")
        instruction_sessions = (
            "Draft long-context multi-agent chat logs (Slack/Teams channels) translating the screenplay dialogue conflicts and character stances into natural conversation.\n"
            f"Create at least {seed.min_chat_sessions} chat sessions with at least {seed.min_chat_messages} messages combined. "
            "The dialogues must naturally reveal each subagent's starting claims, beliefs, and contradictions."
        )
        # Same layer1b hook, applied to the chat substrate: the participants must
        # be talking past each other *about the planted evidence*, so the
        # transcript is a second, independent route to the same contradiction.
        if layer1b_hooks:
          instruction_sessions += (
              f"\n\nCONFLICT EVIDENCE TO SURFACE IN DIALOGUE (required):\n"
              f"{layer1b_hooks}\n"
              "Each speaker states their own view as settled fact. No participant"
              " may summarize the discrepancy, name the correct source, or tell"
              " the reader which side to believe."
          )
        if error_feedback:
          instruction_sessions += f"\n\n⚠️ PREVIOUS ATTEMPT FAILED WITH VALIDATION ERROR:\n{error_feedback}\nPlease correct this error in your new output."

        l1b_sessions = self.client.generate_layer(
            "conflict_layer1b_sessions",
            {
                "seed": seed_dict,
                "phases": layer0_phases,
                "context": layer1_payload,
                "files": l1b_files,
                "slot": {
                    "instruction": instruction_sessions
                }
            }
        )
        if "sessions" not in l1b_sessions:
          raise ValueError("Background payload missing 'sessions'")
        return l1b_sessions

      l1b_sessions = self._generate_with_retry(
          _gen_bg_sessions,
          attempts=max_attempts,
          label="Background Sessions",
          failure_message=f"Failed to generate background payload after {max_attempts} attempts.",
      )

      # Sub-step 3: initial visual assets (depend on files + sessions). The full
      # _validate_background_payload runs here as the final net on the merge.
      def _gen_bg_assets(attempt, error_feedback):
        self._log(f"  [Assets Attempt {attempt}] Structuring initial visual telemetry assets...")
        platform_vocab = self._platforms(layer0_phases)
        instruction_assets = (
            "Plan the initial visual telemetry assets and evidence items for the scenario.\n"
            "Provide at least 2-3 visual assets. For each asset, specify the platform_id, id (prefixed with platform_id), kind, purpose, path (prefixed with platform_id), and a detailed ocr_text/content_text and caption. "
            "For natural images, provide a detailed visual prompt.\n"
            f"Every asset's platform_id MUST be one of these scenario platforms: {platform_vocab}. Spread the assets across them; do not invent new platforms."
        )
        if error_feedback:
          instruction_assets += f"\n\n⚠️ PREVIOUS ATTEMPT FAILED WITH VALIDATION ERROR:\n{error_feedback}\nPlease correct this error in your new output."

        l1b_assets = self.client.generate_layer(
            "conflict_layer1b_assets",
            {
                "seed": seed_dict,
                "phases": layer0_phases,
                "context": layer1_payload,
                "files": l1b_files,
                "sessions": l1b_sessions,
                "platform_vocabulary": platform_vocab,
                "slot": {
                    "instruction": instruction_assets
                }
            }
        )

        candidate_payload = {
            "workspace": l1b_files["workspace"],
            "sessions": l1b_sessions["sessions"],
            "assets": l1b_assets["assets"],
        }

        candidate_payload = _normalize_and_prefix_assets(
            candidate_payload, self._platforms(layer0_phases))
        _validate_background_payload(candidate_payload)
        return candidate_payload

      layer1b_payload = self._generate_with_retry(
          _gen_bg_assets,
          attempts=max_attempts,
          label="Background Assets",
          failure_message=f"Failed to generate background payload after {max_attempts} attempts.",
      )

      if checkpoint_path:
        self._save_checkpoint(
            checkpoint_path,
            layer0_phases,
            layer1_payload,
            layer1b_payload,
            updates,
            rounds,
            evals,
            phase_expanded_assets,
            expanded_rounds,
        )

    # Format dynamic prompt constraints from interactions spec
    families_set = {normalize_family_name(f) for f in seed.allowed_tool_families or []}
    interactions_raw = layer1_payload.get("interactions", []) if layer1_payload else []
    interactions_prompt_block = ""
    if interactions_raw:
      interactions_prompt_block = " You MUST design and implement round mock subagents and trajectories to enforce these multi-agent interaction dependency rules:\n"
      for inter_idx, inter in enumerate(interactions_raw, start=1):
        active_agent = inter.get("active_subagent", "Dev_Agent")
        interaction_type = inter.get("interaction_type", "confrontation_decryption")
        trigger_ev = inter.get("trigger_evidence", "")
        trigger_file = inter.get("trigger_source_file", "")
        res_act = inter.get("resolution_action", "")
        dep_steps = " -> ".join(
            _coerce_dependency_step(s)
            for s in inter.get("dependency_steps", [])
        )

        interactions_prompt_block += (
            f"Rule {inter_idx}: The interaction is a '{interaction_type}'. "
            f"The active subagent is '{active_agent}'. They hold a stale local cache or hallucinated status and will remain out-of-sync "
            f"until the coordinator aligns/corrects them in their instructions containing the exact trigger evidence '{trigger_ev}' "
            f"which must first be extracted from '{trigger_file}'. Once corrected with this evidence, the active agent will update its context and perform the resolution action: '{res_act}'. "
            f"The expert solution commands trajectory MUST follow this exact sequential dependency: {dep_steps}.\n"
        )

    # Collapse the raw-dict / parsed-object split in the sessions BEFORE any
    # round reads them, so the channel vocabulary the gate enforces is the same
    # one materialization will ship. Placed here rather than at generation so
    # the checkpoint-resume path (which restores a payload written by an older
    # build) is covered by the same single call.
    _chan_normalized = _normalize_session_channels(
        layer1b_payload.get("sessions", []) if layer1b_payload else []
    )
    if _chan_normalized:
      self._log(
          f"  [ChannelVocab] normalized {_chan_normalized} session channel"
          " declaration(s) onto the 'channel' key"
      )

    self._log(
        "[4/4] Generating evaluation rounds, updates, and subagents"
        f" round-by-round for rounds 1-{target_rounds}..."
    )

    # ``mech`` is derived once alongside the plugin hooks above.
    final_mechanic_round = _final_mechanic_round(target_rounds)
    # Scenario-level invariant: the protocol must be COMPLETED at least once, or
    # the plugin's declared terminal effect is never emitted and never graded.
    # Tracked across rounds rather than pinned to one round so a scenario that
    # naturally completes early is never forced to repeat itself: on the corpus
    # 27 of 30 mechanic instances already complete before the last selected
    # round, so the completion gate fires in only 10% of scenarios.
    mechanic_completed = False

    for round_idx in range(1, target_rounds + 1):
      round_id = f"r{round_idx:02d}"
      # M1: when the seed opts in, this round's type is decided HERE, not by the
      # LLM. ``None`` leaves the existing behaviour (LLM chooses) untouched.
      forced_round_type = (
          _assigned_round_type(seed.slug, round_idx)
          if seed.force_round_type_plan
          else None
      )

      if len(rounds) >= round_idx:
        self._log(f"  [Resume] Skipping round {round_idx}/{target_rounds} ({round_id}) - already generated.")
        continue

      # 1. Identify current phase
      phases = layer0_phases.get("phases", [])
      current_phase = {}
      for phase in phases:
        if (
            phase.get("start_round", 1)
            <= round_idx
            <= phase.get("end_round", 1)
        ):
          current_phase = phase
          break

      phase_id = current_phase.get("phase_id", "p01")

      # 2. Trigger Phase Expansion if not yet expanded
      if round_id not in expanded_rounds:
        self._log(
            f"[Phase Expansion] Expanding details for Phase {phase_id} (Rounds"
            f" {current_phase.get('start_round')}-{current_phase.get('end_round')})..."
        )
        start = current_phase.get("start_round", 1)
        end = current_phase.get("end_round", 1)
        expected_round_ids = [f"r{idx:02d}" for idx in range(start, end + 1)]

        # History up to current round (compacted so phase-expansion prompts do
        # not carry the full accumulated round history verbatim).
        _compact_so_far = _compact_history_for_context(
            rounds, updates, evals, phase_expanded_assets
        )
        history_so_far = {
            "history_updates": _compact_so_far["history_updates"],
            "history_rounds": _compact_so_far["history_rounds"],
            "history_evals": _compact_so_far["history_evals"],
            "history_assets": _compact_so_far["history_assets"],
        }

        # Tier 3.1 (design 6.1): the phase round timeline and the phase assets
        # are authored in two independently-retried calls. The timeline half
        # keeps the salvage fallback (synthesize minimal outlines) so the
        # scenario is never discarded; the assets half degrades to assets:[] on
        # failure instead of dragging the (already-finalized) timeline down.

        # --- Timeline half (independently retried; salvage fallback) ---
        timeline_success = False
        timeline_payload = {}
        attempt_errors: list[str] = []
        for attempt in range(1, phase_attempts + 1):
          try:
            timeline_instruction = (
              f"You are expanding the ROUND TIMELINE for Phase {phase_id} (Rounds {current_phase.get('start_round')} to {current_phase.get('end_round')}).\n"
              "CRITICAL: Meticulously read and implement the 'timeline_screenplay' beats designed for this phase in 'global_phases.phases'. "
              "For each round in this phase, generate a detailed round outline ('timeline') that aligns perfectly with the screenplay's narrative_beat, active_conflict_names, character_claims, vlm_evidence_requirement, and dialogue_conflict.\n"
              "Generate ONLY:\n"
              "1. `timeline`: round-by-round outlines for the rounds within this phase. Every round entry must reflect the screenplay's active conflicts and character claims.\n"
              "Do NOT emit assets in this step; visual assets are authored separately.\n"
              + _BENIGN_PHRASING_GUARDRAIL
              + f"Return {{"
              + '"timeline": [{"round_id": "' + round_id + '", "phase": "' + current_phase.get("theme", "action") + '", "description": "...", "active_participants": ["..."], "introduced_anchors": []}]}.'
            )
            if attempt_errors:
              timeline_instruction += f"\n\n⚠️ PREVIOUS ATTEMPT FAILED WITH VALIDATION ERROR:\n{_format_retry_feedback(attempt_errors)}\nPlease inspect and correct this error in your new output."

            timeline_payload = self.client.generate_layer(
                "conflict_layer0_expand_phase_timeline",
                {
                    "seed": seed_dict,
                    "global_phases": layer0_phases,
                    "agents": layer1_payload,
                    "background": layer1b_payload,
                    "current_phase_id": phase_id,
                    "history": history_so_far,
                    "slot": {
                        "instruction": timeline_instruction
                    },
                },
            )
            _validate_expanded_phase(
                {"timeline": timeline_payload.get("timeline")}, expected_round_ids
            )
            timeline_success = True
            break
          except Exception as e:
            attempt_errors.append(str(e))
            self._log(f"  [Phase Timeline {phase_id} Attempt {attempt} Failed]: {e}. Retrying...")

        if not timeline_success:
          # Resilience: a failed timeline must NOT discard the whole scenario
          # (previously this raised and lost all rounds). Synthesize a minimal
          # outline for the phase's rounds from the screenplay/phase context so
          # downstream round generation can proceed; only a scenario that ends
          # up with zero rounds is fatal (checked at the end of the loop).
          self._log(
              f"  [Phase Timeline {phase_id}] FAILED after {phase_attempts} "
              f"attempts: {attempt_errors[-1] if attempt_errors else ''}. Falling back to a minimal synthesized"
              " outline so the scenario is not discarded."
          )
          timeline_payload = {
              "timeline": [
                  self._fallback_round_outline(r_id, current_phase, seed)
                  for r_id in expected_round_ids
              ],
          }

        # --- Decisive half (independently retried; degrades to no points) ---
        # Root-cause inversion: the value a round turns on is DECIDED here,
        # against the finalized round timeline, while this phase's asset text
        # does not yet exist. The asset halves below are then told which value
        # they must carry verbatim.
        #
        # The previous order was backwards: assets were frozen first and a
        # later resolution step had to hunt through them for something that
        # could pass as decisive. Measured cost of that order -- 84 rounds
        # ended up naming a value that appeared in NO asset, and those rounds
        # passed their decisive checks 2.8% of the time against 53.1% for
        # rounds whose value was actually findable.
        # Values earlier phases already spent. Derived from the round outlines
        # rather than kept as separate state, so it survives a checkpoint
        # resume for free (expanded_rounds is already checkpointed).
        #
        # Both fields, not just the decided one. When an earlier round's binding
        # was released its GRADED value is ``resolved_value`` and differs from
        # the ``decisive_value`` recorded here, so checking only the latter
        # leaves this phase free to decide a value an earlier round is already
        # graded on -- the same leak as the free pass, pointing the other way.
        previously_decided = sorted({
            _v
            for r_prev in expanded_rounds.values()
            if isinstance(r_prev, dict)
            for _v in (
                str(r_prev.get("decisive_value") or "").strip(),
                str(r_prev.get("resolved_value") or "").strip(),
            )
            if _v
        })
        # The text the background half ALREADY shipped. Stage 3 authors the base
        # workspace files and chat sessions before this stage runs, so by the
        # time S1 picks a value that text is frozen -- and a value that collides
        # with it is text-solvable the moment it is chosen. Passing an empty
        # round payload to the round-scoped collectors yields exactly the base
        # layer (this phase's own ``workspace_adds``/``session_appends`` do not
        # exist yet), which is the dominant leak source: the observed binding-
        # release rate FELL with round index (r01 50% -> r06 20%), the signature
        # of a fixed base corpus rather than accumulating per-round history.
        #
        # Checking it HERE is the point. The confidentiality gate downstream
        # catches the same collision, but by then the value is bound and the
        # resolver's only escape is to re-label the locus ``computed`` -- 43 of
        # 43 ``computed`` picks on the last corpus were emitted only after a
        # leak rejection, none were a first choice. At S1 the value is still
        # soft, so the retry can actually change it.
        _base_workspace_texts = _collect_round_workspace_texts(
            layer1b_payload or {}, {})
        _base_session_texts = _collect_round_session_texts(
            layer1b_payload or {}, {})
        decisive_by_round: dict[str, dict[str, Any]] = {}
        attempt_errors: list[str] = []
        for attempt in range(1, phase_attempts + 1):
          try:
            # NO SPECIES BAN. This prompt once carried an "It must be EVIDENCE,
            # not a VERDICT" clause naming 'APPROVED', 'BLOCKED (3/4 Ready)' and
            # 'AUDIT VERDICT: COMPLETE' as wrong. It was Gate A's deleted theory
            # re-entering as prose, and it was measured to be false twice over:
            #
            #  * Gate A's own post-mortem (see the note at the grounding gate)
            #    found verdict values reach the round's own text 1.2% of the
            #    time against 0.4% for readings, overlap ``decision_outcome``
            #    0% of the time, and SCORE LOWER -- they are harder rounds, and
            #    they were half the corpus (266 of 523).
            #  * Shipping the clause cost exactly the species it named. Values
            #    carrying no digit fell from 33.7% of the corpus (176/523) to
            #    4.9% (27/548); the specific 'APPROVED'-shaped family fell from
            #    21.6% to 1.6%. Those rounds scored 0.6628 against 0.6782 for
            #    numeric ones over 17,804 graded observations -- the ban
            #    deleted the HARDER third of the value space.
            #
            # The failure it claimed to prevent -- the value being the round's
            # own verdict, so no evidence need be read -- is already blocked
            # downstream by ``_validate_round_decision_outcome``, which rejects
            # ``decision_outcome == resolved_value`` into the resolution retry
            # loop. That gate is hard and structural; this clause was prose that
            # cost a species. A narrower restatement here would just re-create a
            # diluted version of the same pressure, so there is none.
            decisive_instruction = (
              f"You are choosing the DECISIVE VALUE for each round of Phase {phase_id} (Rounds {current_phase.get('start_round')} to {current_phase.get('end_round')}).\n"
              "The round timeline is already FINAL (see 'phase_timeline'); this phase's visual assets do NOT exist yet. "
              "Your output is a DESIGN INPUT: the very next step authors those assets and is required to print each value you choose VERBATIM onto the evidence the player inspects. "
              "So DECIDE the value each round turns on -- do not describe evidence that already exists, because none does yet.\n"
              "For EVERY round listed in 'phase_timeline', emit one entry:\n"
              " - `round_id`: the round it belongs to.\n"
              " - `decisive_value`: the exact reading, written exactly as it will be printed on the evidence.\n"
              " - `decisive_label`: a short phrase naming what the value is and which surface it is read from (e.g. 'ALT result on the 06:14 lab panel').\n"
              " - `decisive_value_citable`: is your value ONE reading a correct report must print character-for-character, or a phrase two equally correct reports would lay out differently?\n"
              "     true  -- the WHOLE string is a single atomic reading: a measurement with its unit ('182.1 U/L', '1,420 queries/min'), a code or identifier ('SEC-CONTAIN-8842'), a timestamp ('2024-05-20T14:42:00Z'), an endpoint ('db-prod-02.internal:5432'), a digest ('f2c9a18d04b3e7125a89601d2e4f5139'). "
              "LENGTH NEVER DISQUALIFIES: '1,420 queries/min' and 'https://telemetry-alpha.cluster.internal/v1/sync' are long AND atomic, so both are true.\n"
              "     false -- the string carries anything ALONGSIDE the reading: a parenthetical or bracketed qualifier, a second clause, an owner or actor name, a time or count that explains the reading, or a status word plus an explanatory phrase. "
              "Two correct reports would punctuate those differently, so an exact match on your spelling would fail a report that got the finding right.\n"
              "     FALSE, measured on the shipped corpus: 'LOCKED_EXCLUSIVE (Holder: Avery_Fulfillment_Coordinator, TTL: 120 s)', 'ALL RECONCILIATION CHECKS PASSED (Commit ID: COMMIT-8842-FINAL)', "
              "'16 open tickets (1 P1 critical, 1 P2, 14 low-severity)', 'BLOCKED (3/4 gates ready)', '9.0% pool utilization (18 / 200 active connections at 10:15 UTC)', 'LOADED EPOCH: 2023-11-14T08:12:00Z (STALE)'.\n"
              "That declaration decides how the round is GRADED: true grades the player's report by an exact match on your string, false grades it by meaning.\n"
              "WHY IT MATTERS -- measured over 18,194 graded round-runs: single atomic readings matched exactly passed 66.1%, and compound phrases graded by MEANING passed 63.1% (very slightly harder). "
              "But compound phrases put through an EXACT match passed 0 of 437 times. Declaring a compound value true does not make its round harder, it makes the round ungradable.\n"
              "THIS DECLARATION NEVER CONSTRAINS WHICH VALUE YOU CHOOSE. A compound value is not a lesser value and its round is not easier -- the two grading routes are within 3 points of each other. "
              "NEVER simplify, truncate or re-shape a value to make it citable: choose the value the round actually needs, then declare it honestly. Answer it about the value you actually chose.\n"
              "A CAPTION IS NOT PART OF THE READING. Where the surface prints a field name in front of the value -- 'SHA256: 7f8a91b2c3d4e5f60718293a4b5c6d7e', 'SALT-VERIFY-KEY: c1cd_sal7_8f92a3' -- the caption names the reading and the characters after it ARE the reading. "
              "One correct report reprints the caption, another writes 'the digest is' and then the same characters; both read the panel correctly, so a caption carried inside `decisive_value` fails the second one. "
              "Put the caption in `decisive_label`, which exists for exactly that, and put only the reading in `decisive_value`. The asset still prints the caption, and the player still has to find that line and read it.\n"
              "This is NOT the simplification banned just above: the value, its species and its round are unchanged -- you are spelling the SAME reading without the caption sitting beside it. "
              "It applies ONLY where a field name and a colon stand in front of the reading. Where there is no caption to lift there is nothing to do: 'db-prod-02.internal:5432' is one endpoint and 'FULL CONTAINMENT VERIFIED' is one status phrase, neither has a field name in front, and cutting either would destroy the reading.\n"
              "HARD REQUIREMENT -- the value must be one that can ONLY be settled by RECONCILING SOURCES THAT DISAGREE. "
              "This round's conflict must be the reason the value is in doubt: a stale dashboard against a fresh log, a teammate's quoted figure against the photographed gauge, two systems reporting counts that do not match. "
              "A value the player can copy straight off the first surface they open is the WRONG choice -- there would be nothing to resolve.\n"
              # There is deliberately NO species ban here -- see the note above
              # this prompt. An "EVIDENCE, not a VERDICT" clause used to sit on
              # this line and cost the corpus an entire species of value.
              "COMPOUND VALUES ARE ALLOWED AND ARE OFTEN CORRECT. Where a bare number would identify nothing on its own, carry the qualifier that makes it specific: '100% packet drop on port 443' is right where a bare '100%' is meaningless. "
              "THEN DECLARE THAT VALUE false. A reading that needs a trailing qualifier to mean anything is, by construction, carrying something alongside the reading -- the false case defined above. "
              "Choosing the compound value and declaring it false is the correct pair and its round is no easier; choosing it and declaring it true is the one combination that makes the round ungradable. "
              "A unit is part of the reading and stays true -- '1,420 queries/min' is atomic -- but a phrase naming what dropped, which port, or which error is not part of the reading.\n"
              "SPELL IT THE WAY A WRITTEN REPORT PRINTS IT -- with the space before the unit and the thousands separator in place: '3,850 ms', never '3850ms'. "
              "The asset authored next prints your value verbatim and the player copies it from there into their report, so a spelling nobody would type is scored wrong even when the player read the panel correctly. "
              "Measured: a single value spelled '3850ms' failed 107 otherwise-correct answers, all of which had written '3,850 ms'. "
              "Keep units, thousands separators and identifier suffixes exactly as they will be printed.\n"
              f"Each `decisive_value` must be a SINGLE LINE of at most {_DECISIVE_POINT_MAX_LEN} characters: the reading itself, not the panel title and not the sentence around it.\n"
              "VARY THE KIND of value across the rounds of this phase -- do not make every round turn on the same species of token. Draw on whatever this scenario's domain actually measures.\n"
              "THREE SPECIES ARE EQUALLY VALID and a good phase uses more than one:\n"
              "  (a) MEASUREMENTS -- '182.1 U/L', '1,420 queries/min', '3,850 ms', '410 KB/s';\n"
              "  (b) CODES AND IDENTIFIERS -- 'SEC-CONTAIN-8842', 'CHG-88120', '2024-05-20T14:42:00Z';\n"
              "  (c) STATE AND STATUS PHRASES -- 'FULLY CONTAINED', 'PENDING FINAL SIGN-OFF', 'BLOCKED (3/4 Ready)', "
              "'ALL POST-DEPLOYMENT CHECKS PASSED', 'transitioned from LOCKED_EXCLUSIVE to UNLOCKED'.\n"
              "Species (c) is NOT a lesser choice and must not be avoided: a status phrase printed on a panel still has to be READ off that panel -- "
              "the player cannot know it says CONTAINED rather than BREACHED without looking -- and such rounds measured HARDER than numeric ones, not easier. "
              "A phase in which every value carries a digit has lost a whole species and is a worse phase.\n"
              "EVERY ROUND MUST TURN ON A DIFFERENT VALUE. Never repeat a value across rounds, and never reuse one listed in 'previously_decided_values' (those belong to earlier phases of this same scenario). "
              "A round whose answer the player already holds from an earlier round grades nothing.\n"
              "THE VALUE MUST BE NEW TEXT. 'background' holds the workspace files and chat logs that are ALREADY WRITTEN and that the player can read and search freely. "
              "A value that already appears there is handed to the player for free -- they grep the workspace and never open the evidence. "
              "So do not lift a string out of 'background', and do not choose one that happens to match text already in it.\n"
              "This constrains WHICH value, never WHAT KIND. If the phrase you want is already written somewhere, make it MORE SPECIFIC -- "
              "a different reading, a different state, a narrower phrasing of the same status -- and keep the species you chose. "
              "Switching from a status phrase to an invented ID just to dodge a collision is the wrong repair and makes the phase worse.\n"
              + _BENIGN_PHRASING_GUARDRAIL
              + 'Return {"decisive_points": [{"round_id": "' + round_id + '", "decisive_value": "...", "decisive_label": "...", "decisive_value_citable": true}]}.'
            )
            if attempt_errors:
              decisive_instruction += f"\n\n⚠️ PREVIOUS ATTEMPT FAILED WITH VALIDATION ERROR:\n{_format_retry_feedback(attempt_errors)}\nPlease inspect and correct this error in your new output."

            decisive_raw = self.client.generate_layer(
                "conflict_layer0_expand_phase_decisive",
                {
                    "seed": seed_dict,
                    "global_phases": layer0_phases,
                    "agents": layer1_payload,
                    "background": layer1b_payload,
                    "current_phase_id": phase_id,
                    "phase_timeline": timeline_payload.get("timeline", []),
                    "previously_decided_values": previously_decided,
                    "history": history_so_far,
                    "slot": {
                        "instruction": decisive_instruction
                    },
                },
            )
            citable_downgrades = _validate_expanded_phase_decisive(
                decisive_raw,
                expected_round_ids,
                already_decided={v.casefold() for v in previously_decided},
                workspace_texts=_base_workspace_texts,
                session_texts=_base_session_texts,
            )
            # Not a failure and not a retry -- the value is kept, only its
            # grading arm moved. Logged per round so the rate at which the
            # model over-declares citability stays countable from the run log,
            # the way the degrade path above is.
            for _dg_round_id, _dg_value in citable_downgrades:
              self._log(
                  f"  [Phase Decisive {phase_id} {_dg_round_id}] citable"
                  f" DOWNGRADED: {_dg_value!r} carries more than the reading, so"
                  " an exact substring match would fail a correct report that"
                  " punctuates it differently. Value kept; graded semantically."
              )
            decisive_by_round = {
                str(point.get("round_id")): point
                for point in decisive_raw["decisive_points"]
            }
            break
          except Exception as e:
            attempt_errors.append(str(e))
            self._log(f"  [Phase Decisive {phase_id} Attempt {attempt} Failed]: {e}. Retrying...")

        if not decisive_by_round:
          # Degrade this half only: the asset steps fall back to their previous
          # behaviour (author freely, let the resolution step pick a value out
          # of the finished text). Logged distinctly so the rate at which the
          # inversion is bypassed stays countable.
          self._log(
              f"  [Phase Decisive {phase_id}] FAILED after {phase_attempts} "
              f"attempts: {attempt_errors[-1] if attempt_errors else ''}. Degrading to no pre-decided values"
              " (timeline retained; the resolution step falls back to selecting one)."
          )

        # Ride the decision along on the round outline itself. The assets half
        # below is handed this same list as 'phase_timeline', and each round's
        # own sub-steps read their outline out of it, so every consumer sees
        # the value without a parallel side table to keep in sync.
        for r_out in timeline_payload.get("timeline", []):
          point = decisive_by_round.get(str(r_out.get("round_id")))
          if point:
            r_out["decisive_value"] = str(point.get("decisive_value") or "").strip()
            r_out["decisive_label"] = str(point.get("decisive_label") or "").strip()
            # Normalized to a real bool by the validator above.
            r_out["decisive_value_citable"] = bool(
                point.get("decisive_value_citable"))

        # The brief handed to BOTH asset steps. Empty when the decisive half
        # degraded, so the asset prompt never grows a header with nothing
        # under it (which would read as an order to carry a value that was
        # never chosen).
        decisive_brief = _format_decisive_brief(
            timeline_payload.get("timeline", []))

        # --- Assets half (independently retried; degrades to assets:[]) ---
        assets_success = False
        phase_assets = []
        attempt_errors: list[str] = []
        for attempt in range(1, phase_attempts + 1):
          try:
            platform_vocab = self._platforms(layer0_phases)
            assets_instruction = (
              f"You are specifying NEW VISUAL ASSETS introduced in Phase {phase_id} (Rounds {current_phase.get('start_round')} to {current_phase.get('end_round')}).\n"
              "CRITICAL: Read the FINALIZED round timeline for this phase (provided in 'phase_timeline') and the screenplay beats in 'global_phases.phases'.\n"
              "Generate ONLY:\n"
              "`assets`: concrete specs for any new visual assets introduced in this phase, aligning with the planned assets in global_assets_plan. "
              "Define their id, kind, generator, path, mime_type, available_from_round, description, caption, ocr_text, source_type, and platform_id.\n"
              "KIND -> GENERATOR -> SOURCE_TYPE (pick by what the evidence ACTUALLY is; do NOT force variety - a purely digital phase "
              "may use ZERO 'natural_image'; if the seed 'evidence' list names a 'real_world_source', honor it: dashboard_chart->chart, "
              "log/document->rich_text, in_tool_diagram->diagram, physical_photo->natural_image, hand_sketch->sketch):\n"
              "  - 'chart' (quantitative plot/table in a UI) -> generator 'template_chart', source_type 'chart_spec'.\n"
              "  - 'rich_text' (logs, config, code, documents) -> generator 'template_chart', source_type 'chart_spec'.\n"
              "  - 'diagram' (a clean DIGITAL schematic viewed INSIDE a software tool: network topology, facility/floor-plan layout, wiring/one-line diagram, process flowchart - rendered as an in-UI module) -> generator 'template_chart', source_type 'chart_spec'.\n"
              "  - 'natural_image' (a REAL PHOTOGRAPH of a physical object/scene: router status LEDs, robot cell, antenna dish, server rack) -> generator 'google_image', source_type 'image_generation'.\n"
              "  - 'sketch' (an informal HAND-DRAWN schematic on paper or a whiteboard, photographed) -> generator 'sketch_image', source_type 'image_generation'.\n"
              "Pick 'diagram' when the schematic would realistically live inside a tool's UI, and 'sketch' when it is an informal hand-sketch; decide by context. Never use 'natural_image' for a schematic/screen (use 'diagram'/'sketch'/'chart'); never use 'diagram'/'sketch' for a photographable physical object (use 'natural_image').\n"
              "DOMAIN-AWARENESS: if this scenario's domain centers on physical equipment or environments (industrial robots / robot cells, antennas & dishes, hardware / server racks, vehicles, lab instruments or samples, field sites), introduce at least one 'natural_image' photograph of the physical subject in the phase where it is first observed; and when a round's conflict hinges on a physically-observable state, make that photo DECISIVE by putting the deciding value in its caption/ocr_text. Keep ZERO 'natural_image' for purely digital domains (dashboards, logs, code, web tooling).\n"
              "WHICH SURFACE CARRIES THE DECISIVE VALUE. Printing the value verbatim is required (see below) -- but WHERE you print it decides whether the round measures visual grounding or just reading. "
              "The decisive value belongs on a surface the player has to WORK to read: a quantitative panel where the value sits among competing rows, series or categories and must be located and attributed to the right row; a dense table; or a timeline. "
              "Do NOT park the decisive value on a free-form text panel -- a log dump, a config listing, a plain document -- UNLESS the evidence genuinely IS a log, a config or a document. A deciding value printed as a lone line of prose is taken in at a glance and the round stops testing whether the player can read evidence at all.\n"
              "MEASURED on the shipped corpus (4,496 graded round-runs): rounds whose decisive value sat on a 'rich_text' surface passed 54.9%, against 38.2% for every other carrier combined -- and 'rich_text' was the easier carrier inside EVERY round type (investigation +17.3, decision +5.0, action +11.8, verification +12.5 points), so this is not an artifact of which round types happen to use it. It also carried 36.5% of all decisive values, making it the single largest difficulty leak in the corpus.\n"
              "Non-decisive supporting assets are free to be logs, configs and documents -- this constrains only the ONE surface each round turns on.\n"
              + decisive_brief
              + "CRITICAL DE-DUPLICATION & REUSE RULES FOR PHASE ASSETS:\n"
              " - Inspect the list of initial background assets in `background.assets` and previously planned assets in `history.history_assets`.\n"
              " - Reuse existing IDs if referring to the exact same visual content to avoid duplicates.\n"
              " - Suffix versioned/evolving asset IDs and path filenames with virtual timestamps (e.g. `_YYYYMMDD_HHMMSS`) to show timeline progression.\n"
              " - Use EXACTLY ONE id per snapshot: never emit both a bare (un-timestamped) and a timestamped copy of the same visual; each visual must appear once in the asset list.\n"
              f" Every asset's platform_id MUST be one of these scenario platforms: {platform_vocab}. Spread the assets across them; do not invent new platforms.\n"
              f" {presets_help} Return {{"
              # Every discriminating field in this shape example is "..." ON
              # PURPOSE. A concrete literal here is not read as an illustration,
              # it is read as a default: the shipped corpus carried
              # '"kind": "rich_text", "generator": "template_chart",
              # "source_type": "chart_spec"' on this line and 33.7% of decisive
              # assets came back 'rich_text' against 4.9% in the corpus before
              # it -- a 7x swing that the kind rules on the lines above did not
              # cause and could not hold back. Keep id/kind/generator/
              # source_type as "..." and let those rules decide.
              '"assets": [{"id": "...", "kind": "...", "generator": "...", "path": "...", "mime_type": "image/png", "available_from_round": 2, "anchor_ids": [], "description": "...", "caption": "...", "ocr_text": "...", "source_type": "...", "platform_id": "..."}]}.'
            )
            if attempt_errors:
              assets_instruction += f"\n\n⚠️ PREVIOUS ATTEMPT FAILED WITH VALIDATION ERROR:\n{_format_retry_feedback(attempt_errors)}\nPlease inspect and correct this error in your new output."

            assets_raw = self.client.generate_layer(
                "conflict_layer0_expand_phase_assets",
                {
                    "seed": seed_dict,
                    "global_phases": layer0_phases,
                    "agents": layer1_payload,
                    "background": layer1b_payload,
                    "current_phase_id": phase_id,
                    "phase_timeline": timeline_payload.get("timeline", []),
                    "history": history_so_far,
                    "platform_vocabulary": platform_vocab,
                    "slot": {
                        "instruction": assets_instruction
                    },
                },
            )
            assets_raw = _normalize_and_prefix_assets(
                assets_raw, self._platforms(layer0_phases))
            _validate_expanded_phase_assets(assets_raw)
            phase_assets = assets_raw.get("assets", [])
            assets_success = True
            break
          except Exception as e:
            attempt_errors.append(str(e))
            self._log(f"  [Phase Assets {phase_id} Attempt {attempt} Failed]: {e}. Retrying...")

        if not assets_success:
          # design 6.1: a failed assets half degrades to assets:[] rather than
          # dragging the (already-finalized) timeline down.
          self._log(
              f"  [Phase Assets {phase_id}] FAILED after {phase_attempts} "
              f"attempts: {attempt_errors[-1] if attempt_errors else ''}. Degrading to assets:[] (timeline retained)."
          )
          phase_assets = []

        # Merge the two independently-generated halves back into the single
        # expanded_payload shape the rest of the loop expects.
        expanded_payload = {
            "timeline": timeline_payload.get("timeline", []),
            "assets": phase_assets,
        }

        # Register all expanded rounds
        for r_out in expanded_payload.get("timeline", []):
          expanded_rounds[r_out["round_id"]] = r_out

        # Accumulate phase level assets
        phase_expanded_assets.extend(expanded_payload.get("assets", []))

      # Retrieve expanded round outline for current round
      round_outline = expanded_rounds.get(round_id, {})
      self._log(
          f"  [Round {round_idx}/{target_rounds}] Generating round details"
          f" ({round_id})..."
      )

      # Progressive history context for Layer 3 (only what has been finalized
      # so far). The history is compacted (recent rounds full, older ones
      # summarized, assets reduced to metadata) so the per-round prompt does not
      # grow without bound as rounds accumulate. The authoritative lists
      # (rounds/updates/evals/phase_expanded_assets) are left untouched.
      _compact = _compact_history_for_context(
          rounds, updates, evals, phase_expanded_assets
      )
      current_context = {
          "phases": layer0_phases,
          "agents": layer1_payload,
          "background": layer1b_payload,
          "history_updates": _compact["history_updates"],
          "history_rounds": _compact["history_rounds"],
          "history_evals": _compact["history_evals"],
          "history_assets": _compact["history_assets"],
      }

      allowed_families = seed.allowed_tool_families
      allowed_prompt = self._build_allowed_commands_prompt(allowed_families)
      # (1) Machine-generated protocol-prerequisite contract (from the skill
      # SSOT). Steers Layer 3a to author complete prerequisite chains for
      # multi-step protocols so their terminal commands actually emit effects.
      protocol_contract = _build_protocol_contract_prompt(allowed_families)
      # (2) Whether THIS round's gold must actually act on the conflict
      # protocol rather than only query it. ``protocol_contract`` above only
      # says "if you use one of these, chain the prerequisites"; it never makes
      # the model use one. Empty mechanic => scenario routes to no mechanic
      # plugin => both of the injections below and the gate are no-ops.
      #
      # Two tiers, because "perform" and "complete" are different demands. Most
      # selected rounds must only ADVANCE the protocol gradably (any registered
      # protocol command -- request, gate-set, enter-code -- now emits its own
      # effect, so advancing is assertable). Exactly one round per scenario must
      # COMPLETE it. A 10-round scenario cannot plausibly grant the same access
      # six times: 7 of the 10 mechanic seeds have a single distinct terminal
      # command line across all 30 corpus rounds, so demanding completion every
      # selected round would force either absurd repetition or generation
      # failure (70.6% of selected rounds would enter the retry loop).
      round_requires_mechanic = bool(mech) and _round_requires_mechanic(round_idx)
      round_requires_terminal = (
          bool(mech)
          and round_idx == final_mechanic_round
          and not mechanic_completed
      )
      mechanic_instruction = ""
      if round_requires_terminal:
        mechanic_instruction = (
            "\n\nCOMPLETE THE PROTOCOL (required this round):\n"
            "This round's correct solution must CARRY OUT the coordination"
            " protocol to completion, not merely inspect or advance it."
            " Reading status is evidence-gathering, not resolution.\n"
            f"- The gold steps MUST include at least one of: "
            f"{', '.join(repr(t) for t in mech.terminals)}.\n"
            f"- Executing it is what produces the effect(s) the conflict is"
            f" judged on: {', '.join(mech.effects)}.\n"
            "- Satisfy that command's prerequisites first (see the protocol"
            " prerequisite contract); do not skip straight to the terminal"
            " step.\n"
            "- Choosing the RIGHT target/parameter is the difficulty. Keep the"
            " conflicting evidence genuinely ambiguous so the wrong target is"
            " plausible -- do NOT make the correct one obvious, and do NOT"
            " announce which source is authoritative."
        )
      elif round_requires_mechanic:
        mechanic_instruction = (
            "\n\nADVANCE THE PROTOCOL (required this round):\n"
            "This round's correct solution must move the coordination protocol"
            " forward with a real state-changing step, not merely inspect it."
            " Reading status is evidence-gathering, not resolution.\n"
            f"- The gold steps MUST include at least one of: "
            f"{', '.join(repr(t) for t in mech.protocol)}.\n"
            "- A prerequisite step (requesting, gating, approving) counts: each"
            " records its own effect. A status/read-only command does not.\n"
            "- Do NOT force the protocol to its terminal step every round; a"
            " partial, correctly-sequenced advance is the right answer here.\n"
            "- Choosing the RIGHT target/parameter is the difficulty. Keep the"
            " conflicting evidence genuinely ambiguous so the wrong target is"
            " plausible -- do NOT make the correct one obvious, and do NOT"
            " announce which source is authoritative."
        )
      currently_visible_assets = _assets_visible_in_round(
          layer1b_payload.get("assets", []) + phase_expanded_assets, round_idx)
      visible_assets_text = "\n".join([
          f"- ID: '{a.get('id')}', Kind: '{a.get('kind')}', Path: '{a.get('path')}', Description: '{a.get('description')}'"
          for a in currently_visible_assets
      ])

      tool_families_prompt = (
          f"The allowed tool families for the coordinator in this scenario are: {', '.join(allowed_families)}. "
          "You MUST align the instruction text with these tool families. For example: "
          "if 'run_command' (terminal command execution) is NOT in the allowed tool families, do NOT instruct the coordinator to execute shell commands, scripts, or decryption tools directly. "
          "Instead, instruct them to inspect the relevant visual assets (e.g., decrypted views on dashboards) or delegate tasks to active subagents."
      )

      # A1 (anti-repetition): a compact digest of already-finalized rounds so
      # this round's prompts can steer the model to make THIS round materially
      # different (distinct evidence/anchor, targets, files, and decisions).
      prior_rounds_brief = ""
      if rounds:
        _brief_lines = []
        for _r in rounds[-4:]:
          _instr = re.sub(r"\s+", " ", str(_r.get("instruction", ""))).strip()
          if len(_instr) > 160:
            _instr = _instr[:160] + "..."
          _brief_lines.append(f"- {_r.get('round_id')}: {_instr}")
        prior_rounds_brief = (
            "\nANTI-REPETITION: The following rounds were already authored. THIS"
            " round MUST be materially different from all of them -- advance the"
            " narrative with different evidence, targets, files, and decisions;"
            " do NOT re-run the same investigation or repeat prior actions:\n"
            + "\n".join(_brief_lines)
            + "\n"
        )

      # Step 1: Generate Round Narrative & Updates
      # Tier 3.2 (design 6.2): the round narrative bundle is split into four
      # independently-retried sub-calls -- instruction -> mocks -> updates ->
      # assets -- each with its own bounded loop and independent error_feedback,
      # so a late failure no longer re-calls already-valid earlier sub-steps.
      # The calls stay sequential (mocks need the finalized instruction; updates
      # and assets need the merged narrative). The full
      # _validate_generated_round_narrative still runs as the final net in the
      # assets loop (it needs the round's own new asset IDs to detect leaks).
      narrative_payload = {}

      # Sub-step 1: player instruction (free-form prose).
      def _gen_round_instruction(attempt, error_feedback):
        self._log(f"  [Round {round_id} Instruction Attempt {attempt}] Generating round player instructions...")
        # Name the channels for the INSTRUCTION author too, not just the command
        # author. This rule used to carry a worked example that was a raw
        # session id ('session_01_outage_triage'), which taught every scenario
        # to write storage keys where the player expects a channel -- and the
        # player instruction is read by the SOLVER, so a channel named there
        # that does not exist misdirects the very agent under test. Naming the
        # scenario's own channels also widens the corpus rather than narrowing
        # it: the fixed example pulled unrelated scenarios toward one spelling,
        # while each scenario's real vocabulary is whatever its author invented.
        _instr_chan_names, _ = _channel_vocabulary(
            layer1b_payload.get("sessions", []) if layer1b_payload else [],
            updates,
        )
        if _instr_chan_names:
          _instr_chan_rule = (
              "CRITICAL: When instructing the player to communicate with mock"
              " subagents or send/search messages, you MUST explicitly name the"
              " target chat channel inside the generated player instruction"
              " text, and it MUST be one of this scenario's real channels,"
              " copied VERBATIM:"
              f" {', '.join(repr(c) for c in _instr_chan_names)}. Never refer"
              " to channels generically (like 'the chat channel'), never invent"
              " a name however well it fits the story, and never write a raw"
              " session id -- any of those sends the player somewhere that does"
              " not exist.\n"
          )
        else:
          _instr_chan_rule = (
              "CRITICAL: When instructing the player to communicate with mock"
              " subagents or send/search messages, you MUST explicitly name the"
              " target chat channel inside the generated player instruction"
              " text. Never refer to channels generically (like 'the chat"
              " channel') without giving the exact name of the channel.\n"
          )
        instruction_instr = (
            f"You are generating the narrative instruction for round {round_idx} of {target_rounds}.\n"
            "Review the screenplay outline ('phases.screenplay_outline') and the timelines. "
            "Align the instruction to materialize the narrative_beat and dialogue_conflict. "
            "The instruction should direct the player to investigate visual telemetry conceptual assets and delegate specific Capabilites. "
            "\n"
            "CRITICAL: The coordinator agent (the player) does NOT have terminal/command-line access, and cannot execute shell commands or run python/bash scripts directly. You MUST NOT design task instructions or updates that require the player to run scripts (e.g. 'python decrypt.py'), compile code, or calculate SHA256/MD5 checksums of files. If a file needs to be decrypted or a checksum checked, either design it so a subagent performs that action and reports the result in Slack, or allow the player to resolve it via a simple configuration edit or by reading details from a visual asset.\n"
            f"CRITICAL: You MUST strictly use the exact, full participant/subagent names in all instructions, dialogues, and targets. Mapping registry: {character_map}. (e.g. use 'Devon_Operations_Coordinator' instead of 'Devon', 'Casey_Database_Admin' instead of 'Casey', 'Blair_Backend_Developer' instead of 'Blair'). Never use short/first names.\n"
            "CRITICAL: Any file path checked by a file_check or written by a solution command MUST exactly match a file path that actually exists in the workspace or is added in the update's 'workspace_adds' list. Do NOT invent arbitrary file paths (like using an asset ID as a file path).\n"
            "CRITICAL: This round's FINAL deliverable is a written resolution report. The instruction MUST explicitly direct the player to WRITE a report/audit/summary file that records both the decisive evidence they found AND their final decision/verdict for the conflict -- this written artifact is what the round's outcome is graded on (by its CONTENT, not its location), so the player must be told to produce it and must never be left to deliver the resolution only as a chat message.\n"
            "CRITICAL: Do NOT expose exact asset IDs in the instruction text; refer to them conceptually. "
            "CRITICAL (confidentiality): Do NOT write any specific decisive reading into the instruction -- no concrete metric value, count, percentage, version, timestamp, passcode, or verdict that the player is meant to DISCOVER. Refer to such evidence indirectly ('the latency shown on the dashboard', 'the version in the build chart'); the player must obtain the actual value by inspecting the visual asset, not by reading it from the prompt. "
            "CRITICAL: Clearly differentiate between visual conceptual assets (which the player must inspect using asset open/info) and normal files in the workspace. Do not direct the player to read visual assets as if they were local files in the workspace. "
            + _instr_chan_rule
            + "CRITICAL: You MUST ONLY refer to or require the player to inspect visual assets from the following list of currently visible/available assets for this round. Do NOT reference any other assets that are not in this list:\n"
            f"{visible_assets_text}\n"
            f"{interactions_prompt_block} {tool_families_prompt}\n"
            f"{prior_rounds_brief}"
            "\n"
            + _format_round_type_directive(forced_round_type)
            + _format_decisive_secrecy_guard(round_outline)
            + _format_decisive_stakes(round_outline)
            + _BENIGN_PHRASING_GUARDRAIL
            # "round_type" is "..." ON PURPOSE -- see the note on the assets
            # shape example. This line used to read '"round_type": "action"',
            # three lines under an instruction that says in as many words not to
            # label every round 'action'; the shipped corpus came back 62.8%
            # 'action' and 3.7% 'decision'. The example won. Do not put a
            # literal round_type back here.
            + 'Return JSON format with "rounds": [{"round_id": "' + round_id + '", "round_type": "...", "instruction": "...", "update_ids": ["u_' + round_id + '_01"]}].'
        )
        if error_feedback:
          instruction_instr += f"\n\n⚠️ PREVIOUS ATTEMPT FAILED WITH VALIDATION ERROR:\n{error_feedback}\nPlease correct this error in your new output."

        l3_instr = self.client.generate_layer(
            "conflict_layer3_round_instruction",
            {
                "seed": seed_dict,
                "context": current_context,
                "slot": {
                    "round_index": round_idx,
                    "target_rounds": target_rounds,
                    "round_outline": round_outline,
                    "instruction": instruction_instr,
                    "currently_visible_assets": currently_visible_assets
                }
            }
        )

        # Instruction-level gate (mirrors the rounds/instruction checks at the
        # head of _validate_generated_round_narrative) so an empty/missing
        # instruction retries only this sub-call.
        if not isinstance(l3_instr.get("rounds"), list) or not l3_instr["rounds"]:
          raise ValueError("Response missing 'rounds' or it is empty")
        finalized_instruction = str(l3_instr["rounds"][0].get("instruction", ""))
        if not finalized_instruction:
          raise ValueError("Round instruction is empty")
        # A (root-cause fix): enforce the future-round-leak contract HERE, inside
        # the instruction's OWN retry loop, so a leaky sample is regenerated at
        # the source. This check used to run only in the assets sub-step's loop
        # (_validate_generated_round_narrative), against the already-frozen
        # instruction -- so a leak there could never be re-sampled and
        # deterministically exhausted all retries, aborting the whole version
        # (the it_deployment_gate_v01 failure mode). The identical check in
        # _validate_generated_round_narrative now stays purely as a
        # defense-in-depth net (the instruction is frozen and already clean).
        if _visible_text_references_future_round(
            finalized_instruction, round_index=round_idx
        ):
          raise ValueError("Instruction leaks future round references")
        return (l3_instr, finalized_instruction)

      l3_instr, finalized_instruction = self._generate_with_retry(
          _gen_round_instruction,
          attempts=max_attempts,
          label=f"Round {round_id} Instruction",
          failure_message=f"Failed to generate round {round_id} narrative after {max_attempts} attempts.",
          include_last_error=True,
      )

      # Sub-step 2: round mock subagents (depend on the finalized instruction).
      def _gen_round_mocks(attempt, error_feedback):
        self._log(f"  [Round {round_id} Mocks Attempt {attempt}] Authoring round mock subagents...")
        # Option 4 (prompt nudge): only for delay scenarios, require a positive
        # duration on every mock so strict check #3 (all mocks > 0) is satisfied
        # up front and the deterministic clamp rarely has to fire. Gated on the
        # taxonomy so non-delay seeds' prompts (and thus their output) are
        # unchanged (E6).
        delay_duration_line = (
            "CRITICAL (async delay): This scenario grades a sub-agent processing "
            "delay, so EVERY mock subagent MUST declare a positive integer "
            "'virtual_duration_seconds' (the seconds its delegated work takes to "
            "complete) -- never 0. Give the decisive/delegated sub-agent the "
            "longest duration.\n"
            if "subagent_delay" in (seed.conflict_taxonomy or [])
            else ""
        )
        instruction_mocks = (
            f"You are generating the round-level mock subagent callbacks for round {round_idx} of {target_rounds}.\n"
            "You are given the FINALIZED player instruction for this round below. Design the mock subagents "
            "(the non-player teammates the coordinator delegates to or messages) whose responses and work outcomes support that instruction.\n"
            f"--- FINALIZED ROUND INSTRUCTION ---\n{finalized_instruction}\n--- END INSTRUCTION ---\n"
            f"CRITICAL: You MUST strictly use the exact, full participant/subagent names in every agent_role, mock_response, and target. Mapping registry: {character_map}. (e.g. use 'Devon_Operations_Coordinator' instead of 'Devon', 'Casey_Database_Admin' instead of 'Casey', 'Blair_Backend_Developer' instead of 'Blair'). Never use short/first names.\n"
            "CRITICAL: If the round instruction requires the player to retrieve a key, code, or passcode from a subagent and then use it in a local script or file action in the same round, the subagent's mock_response MUST directly provide that key, code, or passcode.\n"
            "CRITICAL: Any entry in 'state_effects' inside mock_subagents MUST be a valid, executable OpenClaw command string (e.g. 'file create --path workspace/proxy_key.txt --content ...' or 'file append ...') that performs the subagent's work outcome. Do NOT write natural language descriptions or explanations in 'state_effects'.\n"
            "CRITICAL (JSON SAFETY): Inside every command string (in 'state_effects' and anywhere a command appears), quote ALL argument values with SINGLE quotes, never double quotes -- e.g. write message send --channel '#risk-escalations-91' --message 'ready to clear the flag' and file create --path 'workspace/key.txt' --content 'SEC_KEY_9921A'. Because these strings are embedded in JSON (which is delimited by double quotes), any literal double quote inside them breaks the JSON. Do NOT use double quotes inside command strings; use single quotes so the output stays valid JSON.\n"
            f"{delay_duration_line}"
            f"{interactions_prompt_block} {allowed_prompt}\n"
            "\n"
            + _format_decisive_secrecy_guard(round_outline)
            + _format_decisive_stakes(round_outline)
            + _BENIGN_PHRASING_GUARDRAIL
            + 'Return JSON format with "mock_subagents": [{"agent_role": "...", "mock_response": "...", "capabilities": [], "private_knowledge": [], "virtual_duration_seconds": 10, "state_effects": []}].'
        )
        if error_feedback:
          instruction_mocks += f"\n\n⚠️ PREVIOUS ATTEMPT FAILED WITH VALIDATION ERROR:\n{error_feedback}\nPlease correct this error in your new output."

        l3_mocks = self.client.generate_layer(
            "conflict_layer3_round_mocks",
            {
                "seed": seed_dict,
                "context": current_context,
                "narrative": l3_instr,
                "slot": {
                    "round_index": round_idx,
                    "target_rounds": target_rounds,
                    "round_outline": round_outline,
                    "instruction": instruction_mocks
                }
            }
        )
        _normalize_mock_state_effect_quoting(l3_mocks)
        # Option 1 (deterministic repair): for subagent_delay scenarios, clamp any
        # non-positive virtual_duration_seconds up to the default so strict
        # check #3 cannot silently drop the version post-materialization.
        _normalize_mock_durations(l3_mocks, seed.conflict_taxonomy)
        _validate_generated_round_mocks(l3_mocks, allowed_families)

        # Merge instruction + mocks into the single shape the rest of the round
        # pipeline (and _validate_generated_round_narrative) expects.
        l3_narr = {"rounds": l3_instr.get("rounds", []) or []}
        if l3_narr["rounds"]:
          l3_narr["rounds"][0]["mock_subagents"] = l3_mocks.get("mock_subagents", [])
        return l3_narr

      l3_narr = self._generate_with_retry(
          _gen_round_mocks,
          attempts=max_attempts,
          label=f"Round {round_id} Mocks",
          failure_message=f"Failed to generate round {round_id} narrative after {max_attempts} attempts.",
          include_last_error=True,
      )

      # Sub-step 3: async updates (depend on the merged narrative).
      def _gen_round_updates(attempt, error_feedback):
        self._log(f"  [Round {round_id} Updates Attempt {attempt}] Drafting async updates and chat conversations...")
        instruction_upd = (
            f"Generate the asynchronous updates triggering during round {round_idx}.\n"
            "Translate the screenplay dialogue conflicts and claims into realistic Slack/Teams messages in session_appends. "
            "Ensure dialogue conversations support the current round instruction.\n"
            "CRITICAL (confidentiality): The chat messages and workspace_adds must NOT transcribe the decisive reading the player must discover from the visual assets -- no exact metric value, count, percentage, version, timestamp, or passcode. Characters may REFER to 'the figure on the dashboard' or surface the conflict, but must not write the actual value into text, otherwise the round becomes solvable without opening the asset.\n"
            + _format_decisive_secrecy_guard(round_outline)
            + _format_decisive_stakes(round_outline)
            + 'Return JSON format with "updates": [{"update_id": "u_' + round_id + '_01", "trigger_virtual_time": 15, "workspace_adds": [], "session_appends": [], "session_redactions": []}].'
        )
        if error_feedback:
          instruction_upd += f"\n\n⚠️ PREVIOUS ATTEMPT FAILED WITH VALIDATION ERROR:\n{error_feedback}\nPlease correct this error in your new output."

        l3_upd = self.client.generate_layer(
            "conflict_layer3_round_updates",
            {
                "seed": seed_dict,
                "context": current_context,
                "narrative": l3_narr,
                "slot": {
                    "round_index": round_idx,
                    "target_rounds": target_rounds,
                    "round_outline": round_outline,
                    "instruction": instruction_upd
                }
            }
        )

        # Updates-level gate (mirrors the updates checks in the final net) so an
        # invalid update path retries only this sub-call.
        updates_list = l3_upd.get("updates")
        if not isinstance(updates_list, list):
          raise ValueError("Updates must be a list")
        for idx, update in enumerate(updates_list):
          if not isinstance(update, dict):
            raise ValueError("Update must be a dictionary")
          _validate_workspace_paths(
              update.get("workspace_adds", []),
              f"Round {round_id} Update #{idx} workspace_adds",
          )
        return l3_upd

      l3_upd = self._generate_with_retry(
          _gen_round_updates,
          attempts=max_attempts,
          label=f"Round {round_id} Updates",
          failure_message=f"Failed to generate round {round_id} narrative after {max_attempts} attempts.",
          include_last_error=True,
      )

      # Sub-step 4: new round visual assets (depend on narrative + updates). The
      # full _validate_generated_round_narrative runs here as the final net.
      def _gen_round_assets(attempt, error_feedback):
        self._log(f"  [Round {round_id} Assets Attempt {attempt}] Designing new round visual telemetry evidence assets...")
        platform_vocab = self._platforms(layer0_phases)
        instruction_assets = (
            f"Generate concrete specifications for any new visual telemetry assets introduced during round {round_idx}.\n"
            f"Every asset MUST have a platform_id, and it MUST be one of these scenario platforms: {platform_vocab} (do not invent new platforms). "
            "Suffix versioned/evolving asset IDs and path filenames with virtual timestamps (e.g. `_YYYYMMDD_HHMMSS`) to show time progression. "
            "The asset's path, id, caption, and ocr_text MUST reference this timestamp.\n"
            "Use EXACTLY ONE id per snapshot: do NOT also emit a bare (un-timestamped) or short `_HHMM` copy of the same chart/screenshot. Each visual must appear once in the asset list.\n"
            f"{presets_help}"
            + _format_round_carrier_directive(
                round_outline, currently_visible_assets)
            + 'Return JSON format with '
            '"assets": [{"id": "...", "kind": "chart", "generator": "template_chart", "path": "...", "mime_type": "image/png", "available_from_round": ' + str(round_idx) + ', "anchor_ids": [], "description": "...", "caption": "...", "ocr_text": "...", "platform_id": "..."}].'
        )
        if error_feedback:
          instruction_assets += f"\n\n⚠️ PREVIOUS ATTEMPT FAILED WITH VALIDATION ERROR:\n{error_feedback}\nPlease correct this error in your new output."

        l3_assets = self.client.generate_layer(
            "conflict_layer3_round_assets",
            {
                "seed": seed_dict,
                "context": current_context,
                "narrative": l3_narr,
                "updates": l3_upd,
                "platform_vocabulary": platform_vocab,
                "slot": {
                    "round_index": round_idx,
                    "target_rounds": target_rounds,
                    "round_outline": round_outline,
                    "instruction": instruction_assets
                }
            }
        )

        narrative_payload = {
            "rounds": l3_narr["rounds"],
            "updates": l3_upd["updates"],
            "assets": l3_assets["assets"],
        }

        narrative_payload = _normalize_and_prefix_assets(
            narrative_payload, self._platforms(layer0_phases))
        visible_asset_ids = [
            a.get("id")
            for a in _assets_visible_in_round(
                layer1b_payload.get("assets", []) + phase_expanded_assets,
                round_idx)
        ]
        # This round's own new assets are added UNFILTERED on purpose. The list
        # drives the PROSE leak-check, and prose that foreshadows an asset the
        # round cannot open yet is legitimate setup, not a defect: the 17
        # corpus rounds that only mention one average 0.7251 against a 0.6590
        # clean baseline. Narrowing this is the one thing Gate F must not do.
        visible_asset_ids.extend([a.get("id") for a in narrative_payload.get("assets", []) if isinstance(a, dict)])
        _validate_generated_round_narrative(narrative_payload, round_id, visible_asset_ids)
        return narrative_payload, visible_asset_ids

      narrative_payload, visible_asset_ids = self._generate_with_retry(
          _gen_round_assets,
          attempts=max_attempts,
          label=f"Round {round_id} Assets",
          failure_message=f"Failed to generate round {round_id} narrative after {max_attempts} attempts.",
          include_last_error=True,
      )

      # Adopt any update this round authored but forgot to list in
      # ``update_ids``; undeclared updates are never compiled into background
      # events, so their content is generated and then thrown away. Done here,
      # on the narrative payload itself, so EVERY downstream consumer sees the
      # repaired declaration -- in particular ``_collect_delayed_arrivals``,
      # which scopes to declared ids and would otherwise keep treating a
      # recovered late arrival as if it never arrives.
      if narrative_payload.get("rounds"):
        _round0 = narrative_payload["rounds"][0]
        if isinstance(_round0, dict):
          _before = list(_round0.get("update_ids") or [])
          _round0["update_ids"] = _bind_orphan_updates_to_round(
              round_id, _before, narrative_payload.get("updates") or []
          )
          _adopted = len(_round0["update_ids"]) - len(_before)
          if _adopted:
            self._log(
                f"  [Updates] {round_id}: adopted {_adopted} undeclared"
                f" update(s) named for this round"
                f" ({_before or 'nothing was declared'} ->"
                f" {_round0['update_ids']})."
            )

      # Step 1.5 (difficulty): name THIS round's single decisive evidence. A
      # dedicated small call returns the anchor + the concrete value the solver
      # must surface, so downstream evals grade real problem-solving instead of
      # mere command syntax. Best-effort: a failure degrades to a synthesized
      # anchor (and no value) rather than aborting the round.
      # P1a: give the resolver the ACTUAL content it must ground the decisive
      # value in. ``visible_assets_text`` (above) only lists id/kind/path/desc,
      # so the model was historically asked to return a value "verbatim present
      # in an asset's ocr_text/caption or a workspace file" it was never shown --
      # inviting ungrounded guesses that downstream reconcilers then patched.
      # Assemble the round's readable asset text + file contents (capped) so it
      # can PICK a genuinely present value.
      #
      # Gate F: filter the round's OWN new assets by availability too. They
      # used to be appended wholesale, so an asset the narrative deferred to a
      # later round was still offered here as groundable -- and the visibility
      # check already inside ``_validate_round_resolution`` ("does not name a
      # visible asset this round") was comparing against that same polluted
      # list, so it could never fire. Showing the resolver only what the player
      # can open both prevents the pick and re-arms the existing gate; a
      # resolver that names something else now fails it and is regenerated with
      # that feedback, instead of shipping a citation into a panel nobody has.
      round_assets_for_grounding = currently_visible_assets + _assets_visible_in_round(
          narrative_payload.get("assets", []), round_idx)
      # Anchors are offered to the resolver as "already present on visible
      # assets (prefer reusing one)", so they come off the same list.
      round_visible_anchor_ids: list[str] = []
      for a in round_assets_for_grounding:
        for aid in (a.get("anchor_ids") or []):
          if isinstance(aid, str) and aid.strip():
            round_visible_anchor_ids.append(aid.strip())
      # ``updates`` accumulates every EARLIER round's updates (extended at the
      # end of each round iteration), and a task for round rN ships the workspace
      # accumulated over r1..rN. Feed those prior writes to the scan so the
      # confidentiality gate sees the same files the solving agent will, instead
      # of only the base workspace plus this round's adds.
      round_workspace_texts = _collect_round_workspace_texts(
          layer1b_payload, narrative_payload, prior_updates=updates
      )
      # Agent-visible chat/session text (base sessions + this round's async
      # session_appends). Fixed across the resolution retry loop, so collect once
      # here and feed the confidentiality gate below -- it is the side-channel a
      # leaky decisive value must NOT appear in (the difficulty-collapse bug).
      round_session_texts = _collect_round_session_texts(
          layer1b_payload, narrative_payload
      )
      # Agent-reachable mock subagent text (this round's mocks, already authored
      # by the mocks sub-layer above, plus every earlier round's -- their file and
      # message effects persist into this round's state). Like the two scans
      # above this is fixed across the resolution retry loop, so collect once.
      round_mock_texts = _collect_round_mock_texts(
          narrative_payload, prior_rounds=rounds
      )
      # Liveness tripwire for the three collectors above. Each one feeds one arm
      # of the confidentiality gate, and an arm whose corpus is empty reports
      # every round clean -- a confident false negative that is strictly worse
      # than no gate. Raise if a surface has content but its collector returned
      # nothing; the driver isolates and counts this per scenario version.
      _audit_extractor_liveness(
          layer1b_payload=layer1b_payload,
          narrative_payload=narrative_payload,
          prior_updates=updates,
          prior_rounds=rounds,
          workspace_texts=round_workspace_texts,
          session_texts=round_session_texts,
          mock_texts=round_mock_texts,
          round_id=round_id,
      )
      grounding_text = _build_resolution_grounding_text(
          round_assets_for_grounding, round_workspace_texts
      )

      decisive_anchor = ""
      resolved_value = ""
      resolved_value_locus = ""
      decision_outcome = ""
      attempt_errors: list[str] = []
      # S5: the value this round turns on was DECIDED by the phase layer before
      # any of this phase's evidence existed, and the asset authors were then
      # ordered to print it verbatim. The resolver's job here is to say WHERE it
      # ended up and WHAT the coordinator must do about it -- NOT to choose a
      # value out of the finished text. Without this binding none of the phase
      # layer's guarantees reach the graded field: a round designed around
      # '1,420 queries/min' could still be graded on a decorative '12', and the
      # cross-round de-duplication in particular would buy nothing (213 of 523
      # corpus rounds, 40.7%, shared their decisive value with another round of
      # the same scenario). Empty when the decisive half degraded, in which case
      # everything below is the pre-inversion behaviour, unchanged.
      decided_value = str(round_outline.get("decisive_value") or "").strip()
      decided_label = str(round_outline.get("decisive_label") or "").strip()
      # Values the OTHER rounds of this scenario are already spending. A round
      # graded on a value the player has already read in an earlier round is
      # free score. The phase layer does enforce distinctness -- but only over
      # the values it DECIDES, and that guarantee stops at the free pass below,
      # which until now picked with no cross-round constraint of any kind.
      #
      # MEASURED on v6 (60 scenarios, 600 rounds): 43 rounds shared their graded
      # value with another round of the same scenario, up from 18/600 on v5. The
      # free pass is where they come from -- free picks are 25.5% of all rounds
      # (153/600) but 55.8% of the sharing ones (24/43), a 2.2x enrichment. The
      # shape is a released round landing on a value some other round had
      # legitimately bound: expense_reimbursement_conflict_v01 ran 5 of its 10
      # rounds on 'telemetry-pass-8899', 4 of those 5 free picks.
      #
      # Derived from ``expanded_rounds`` -- which is checkpointed -- rather than
      # kept as separate state, so a resumed run cannot silently lose the
      # constraint (the same reason ``previously_decided`` is derived). Covers
      # both halves: what another round DECIDED (it is going to use it) and what
      # an earlier round RESOLVED to (the player has already seen it).
      spent_values: dict[str, str] = {}
      for _other_id, _other in expanded_rounds.items():
        if str(_other_id) == str(round_id) or not isinstance(_other, dict):
          continue
        for _key in ("decisive_value", "resolved_value"):
          _spent = str(_other.get(_key) or "").strip()
          if _spent:
            spent_values.setdefault(_norm_decisive(_spent), _spent)
      # Attempt budget. Hold the resolver to the decided value while budget
      # lasts, then release it for a free pass. A decided value the asset author
      # never printed cannot be rescued by retrying -- both the value and the
      # evidence are already fixed by this point -- so releasing keeps the round
      # graded on a GROUNDED value (the gates below still apply) instead of
      # degrading it to no decisive check at all. The release is logged
      # distinctly, once, so its rate stays countable.
      #
      # The free pass gets ``_FREE_PICK_ATTEMPTS`` shots, not one, because the
      # distinctness gate below has to have somewhere to send a rejection: with
      # a single free attempt every gate on that attempt is inert (it is the
      # final attempt, which always accepts) and the fix would be a no-op.
      attempt_modes = (
          ["bound"] * max_attempts + ["free"] * _FREE_PICK_ATTEMPTS
          if decided_value
          else ["free"] * max_attempts
      )
      # M7 (Axis A): when the seed opts in (``force_vision_locus``), the decisive
      # value MUST live in a visual asset so the round is unsolvable from text.
      # We both tell the resolver to require an ``asset:`` locus (below) AND hard-
      # reject any non-asset pick in the retry loop, so a text-solvable value is
      # regenerated at the source. Non-opted seeds keep the legacy "strongly
      # prefer" nudge and stay byte-identical (E6).
      if seed.force_vision_locus:
        vision_directive = (
            " REQUIRED: 'resolved_value_locus' MUST be 'asset:<id>' and the value MUST"
            " live ONLY in that VISUAL asset's ocr_text/caption. A 'file:' or 'computed'"
            " locus is NOT acceptable for this round -- the answer must be obtainable"
            " solely by looking at the image, never by reading any text source.\n"
        )
      else:
        vision_directive = (
            " Strongly prefer a value"
            " that lives in a VISUAL asset's ocr_text/caption (locus 'asset:<id>'), so"
            " solving the round requires looking at the image rather than reading the"
            " answer from text.\n"
        )
      for attempt, attempt_mode in enumerate(attempt_modes, start=1):
        bound_value = decided_value if attempt_mode == "bound" else ""
        # Once, on the FIRST free attempt. The free pass now spans several
        # attempts, and logging per attempt would silently inflate the release
        # rate this line exists to keep countable.
        if attempt_mode == "free" and decided_value and attempt == max_attempts + 1:
          self._log(
              f"  [Round {round_id} Resolution] BINDING RELEASED: the decided"
              f" value '{decided_value}' could not be grounded in"
              f" {max_attempts} attempts; falling back to a free pick so the"
              " round keeps a decisive check."
          )
        try:
          if bound_value:
            value_directive = (
                f"\nThe DECISIVE VALUE for this round has ALREADY BEEN DECIDED. It was"
                " chosen when this phase was designed, BEFORE any of this round's"
                " evidence was authored, and the evidence was then authored to carry"
                " it:\n"
                f"  decided value: '{bound_value}'\n"
                + (f"  what it is: '{decided_label}'\n" if decided_label else "")
                + " Return that exact string as 'resolved_value', VERBATIM, character for"
                " character. You are NOT choosing a value here and you may NOT substitute"
                " a different one -- your job is to say WHERE it now lives and WHAT the"
                " coordinator must do about it.\n"
                " Return 'decisive_anchor' (an 'EVIDENCE:SHORT_TOKEN' id) naming the piece"
                " of evidence that carries it."
            )
          else:
            value_directive = (
                " Return 'decisive_anchor' (an 'EVIDENCE:SHORT_TOKEN' id) and 'resolved_value'"
                " (the exact concrete value/string that is the correct conclusion). The"
                " resolved_value MUST appear verbatim in a visible asset's ocr_text/caption"
                " or in one of this round's workspace file contents so it can be cited and checked."
                # Told BEFORE the pick, not only enforced after it. The gate
                # below can reject a collision, but a rejection costs an LLM
                # call on a path a quarter of the corpus already takes, and the
                # observed failure is the model gravitating to the same salient
                # token again and again (4 of one scenario's 5 repeats of
                # 'telemetry-pass-8899' were free picks) -- naming them is what
                # actually moves it off that token.
                + (
                    "\nThese values are ALREADY TAKEN by other rounds of this same"
                    " scenario. Do NOT choose any of them, and do not choose a"
                    " trivial restatement of one: a player who solved the earlier"
                    " round already knows the answer, so a repeat grades nothing."
                    " Pick a DIFFERENT reading off this round's own evidence:\n"
                    + "".join(
                        f"  - {taken}\n"
                        for taken in sorted(spent_values.values())
                    )
                    if spent_values
                    else ""
                )
            )
          resolution_instruction = (
              f"For round {round_id}, identify the SINGLE decisive piece of evidence the"
              f" coordinator must engage with to solve this task instruction:\n"
              f"'{narrative_payload['rounds'][0]['instruction']}'\n"
              f"Anchors already present on visible assets (prefer reusing one): {round_visible_anchor_ids}\n"
              f"The ONLY visual assets visible in this round are:\n{visible_assets_text}\n"
              + value_directive
              + " ALSO return 'decision_outcome': a SHORT free-form label (a few words, NOT a"
              " sentence) naming the CORRECT way the coordinator must resolve THIS round's"
              " conflict -- the verdict/action category, e.g. 'roll back to v4.2', 'block the"
              " deploy', 'escalate to on-call', 'approve the migration'. It must be grounded in"
              " the conflict and the decisive evidence, and MUST be different from"
              " 'resolved_value' (the value is the evidence; the outcome is the decision)."
              " ALSO return 'resolved_value_locus' (MANDATORY -- this field must ALWAYS be"
              " present and non-empty) naming WHERE that value actually lives, so"
              " the deterministic checker is bound to the right artifact:\n"
              "  - 'asset:<asset_id>' if the value is read from a visible asset's ocr_text/caption"
              " (it will NOT be present in any workspace file);\n"
              "  - 'file:<workspace_path>' if the value is verbatim inside that workspace file's"
              " contents;\n"
              "  - 'computed' if the value must be derived/aggregated and is not verbatim in any"
              " single source.\n"
              " Pick the SINGLE most accurate locus; never claim 'file:' unless the value is"
              " literally in that file's bytes. Never leave 'resolved_value_locus' blank --"
              " if you are unsure which artifact holds it, use 'computed'.\n"
              " CONFIDENTIALITY: the resolved_value must be readable ONLY from the single"
              " locus you name. Do NOT pick a value that also appears in the round"
              " instruction above or in any OTHER workspace file."
              + vision_directive
              + (
                  " FIND the decided value among the concrete sources below and name the"
                  " one that carries it in 'resolved_value_locus'. Do not paraphrase it and"
                  " do not answer with a different string that happens to be easier to"
                  " locate:\n"
                  if bound_value
                  else " Choose 'resolved_value' as a substring that LITERALLY appears in one of the"
                  " concrete sources below (it must match the one you name in"
                  " 'resolved_value_locus'):\n"
              )
              + f"{grounding_text}"
          )
          if attempt_errors:
            resolution_instruction += (
                f"\n\n⚠️ PREVIOUS ATTEMPT FAILED:\n{_format_retry_feedback(attempt_errors)}\n"
                "Return a resolved_value that literally appears in the named source"
                " (or use locus 'computed' for a genuinely derived value)."
            )
          resolution_slot: dict[str, Any] = {
              "round_index": round_idx,
              "target_rounds": target_rounds,
              "round_outline": round_outline,
              "instruction": resolution_instruction,
              "currently_visible_assets": currently_visible_assets,
          }
          if bound_value:
            # Structured as well as in prose: the binding is a contract, not a
            # suggestion buried in a long instruction.
            resolution_slot["decided_value"] = bound_value
            resolution_slot["decided_label"] = decided_label
          l3_resolution = self.client.generate_layer(
              "conflict_layer3_round_resolution",
              {
                  "seed": seed_dict,
                  "context": current_context,
                  "narrative": narrative_payload,
                  "slot": resolution_slot,
              },
          )
          cand_anchor = str((l3_resolution or {}).get("decisive_anchor", "")).strip()
          cand_value = str((l3_resolution or {}).get("resolved_value", "")).strip()
          cand_locus = str(
              (l3_resolution or {}).get("resolved_value_locus", "")
          ).strip()
          cand_outcome = str(
              (l3_resolution or {}).get("decision_outcome", "")
          ).strip()
          if bound_value:
            # The decision is not the resolver's to make. Answering with a
            # different string is overridden rather than retried: the phase
            # layer committed to this value and the evidence was authored to
            # carry THIS one, so anything else is grounded nowhere by
            # construction. Folded comparison, so a pure whitespace/case
            # rewrap is not reported as a deviation.
            if cand_value and _fold_for_grounding(cand_value) != _fold_for_grounding(bound_value):
              self._log(
                  f"  [Round {round_id} Resolution] resolver answered"
                  f" '{cand_value}'; held to the decided value '{bound_value}'."
              )
            cand_value = bound_value
          # There is deliberately NO shape gate here. Gate A used to classify
          # the value as reading / buried / verdict and have an LLM rewrite the
          # last two, on the theory that a verdict-shaped value lets the round
          # be answered without reading anything. Measured on the 523-round
          # corpus -- generated before Gate A existed, so a clean sample of
          # what it would have rewritten -- the theory is false: verdict values
          # are reachable from the round's own text in 2 cases out of 169
          # (1.2%, against 0.4% for readings), overlap ``decision_outcome`` 0%
          # of the time, and score LOWER rather than higher (holding the check
          # arm fixed: -17.8pp on semantic(D), -4.9pp on semantic(verdict), an
          # arm that never quotes the value at all). They are HARDER rounds,
          # and they are half the corpus (266 of 523). The real concern -- the
          # answer being handed over in text -- is prevented at the source by
          # the secrecy guard, not repaired afterwards here.
          # P1b: grounding gate -- an ungrounded value / bogus locus is fed back
          # and regenerated at the source instead of leaking to reconcilers.
          _validate_round_resolution(
              cand_value,
              cand_locus,
              round_assets_for_grounding,
              round_workspace_texts,
              logger=self._log,
          )
          # Mock-leak repair, BEFORE the gate below judges. This round's mocks
          # were authored two sub-layers before the resolver picked a value, so a
          # collision is expected, not a modelling error -- rewriting the mock to
          # refer to "the value on the dashboard" preserves both the resolver's
          # natural pick and the narrative, where rejecting would burn the retry
          # budget on roughly a quarter of all rounds. Earlier rounds' mocks are
          # NOT touched (their checkers already exist); those collisions fall
          # through to the gate and are regenerated at the value instead.
          if cand_locus.startswith(("asset:", "file:")):
            n_repaired = _repair_round_mock_leaks(
                self.client,
                cand_value,
                cand_locus,
                narrative_payload,
                logger=self._log,
            )
            if n_repaired:
              self._log(
                  f"  [Resolution] rewrote {n_repaired} mock text(s) that"
                  f" transcribed '{cand_value}' so the value stays confined to"
                  f" '{cand_locus}'."
              )
              # The scan feeding the gate is a snapshot taken before the rewrite.
              round_mock_texts = _collect_round_mock_texts(
                  narrative_payload, prior_rounds=rounds
              )
          # I1 (narrative confidentiality): dual of the grounding gate. The value
          # must be readable ONLY from its locus, never from an agent-visible text
          # side-channel (the instruction or a non-locus workspace file), or the
          # round is text-solvable without engaging the evidence (for an asset
          # locus, without looking at the image). Feeds the SAME retry loop, so a
          # leaky pick is regenerated at the source instead of shipped.
          _validate_round_confidentiality(
              cand_value,
              cand_locus,
              str((narrative_payload["rounds"][0] or {}).get("instruction") or ""),
              round_workspace_texts,
              session_texts=round_session_texts,
              mock_texts=round_mock_texts,
              logger=self._log,
          )
          # M7 (Axis A) hard gate: when the seed forces a vision locus, a non-asset
          # pick is text-solvable and defeats the point -- reject it into the SAME
          # retry loop (regenerate at the source) rather than shipping it. No-op for
          # non-opted seeds, so their behaviour is unchanged (E6).
          if seed.force_vision_locus and not cand_locus.startswith("asset:"):
            raise ValueError(
                "force_vision_locus: resolved_value_locus must be 'asset:<id>' so"
                f" the round is unsolvable from text, got locus '{cand_locus}'."
            )
          # Decisive-carrier gate. WHERE the value sits decides whether the
          # round measures visual grounding or just reading: a value parked on
          # a free-text panel is taken in at a glance. MEASURED on the shipped
          # corpus (4,496 graded round-runs): a 'rich_text' decisive carrier
          # passed 54.9% against 38.2% for every other carrier combined, and it
          # was the EASIER carrier inside every round type (+17.3/+5.0/+11.8/
          # +12.5 pp), so the gap is not a round-type composition artifact. It
          # carried 36.5% of all decisive values -- the largest single
          # difficulty leak in the corpus.
          #
          # Reject into the SAME retry loop as the gates above, so a better
          # carrier is chosen AT THE SOURCE rather than patched afterwards.
          # BUDGETED: the FINAL attempt always accepts. A round whose evidence
          # genuinely IS a log, a config or a document therefore keeps its
          # decisive check instead of degrading to none, and the degrade path
          # below stays as rare as it is today (0.2% of corpus rounds).
          if (
              attempt < len(attempt_modes)
              and cand_locus.startswith("asset:")
              and _asset_kind_by_id(
                  round_assets_for_grounding,
                  cand_locus[len("asset:"):],
              ) == "rich_text"
          ):
            raise ValueError(
                "decisive carrier: the resolved value sits on a free-text"
                f" 'rich_text' surface ('{cand_locus}'), which is read at a"
                " glance rather than interpreted. Pick a decisive value that"
                " lives on a quantitative surface -- a chart, a dense table, a"
                " timeline -- where it must be located among competing rows,"
                " series or categories. Only fall back to the text panel if no"
                " other surface visible in this round carries a value that"
                " decides the round."
            )
          # Cross-round distinctness, on the FREE pick only. A bound pick is the
          # phase layer's decided value, which that layer already holds distinct
          # (``already_decided``); re-litigating it here would only burn the
          # remaining attempts and force the very free pass this gate exists to
          # constrain. The free pass is where the guarantee was being lost --
          # 24 of v6's 43 value-sharing rounds were free picks, against free
          # picks being 25.5% of all rounds.
          #
          # BUDGETED like every other gate in this loop: the final attempt
          # accepts. A round whose evidence genuinely carries only one decisive
          # reading keeps a graded check rather than degrading to none, and the
          # residue stays visible in the log instead of becoming silent.
          if (
              not bound_value
              and _norm_decisive(cand_value) in spent_values
          ):
            _clash = spent_values[_norm_decisive(cand_value)]
            if attempt < len(attempt_modes):
              raise ValueError(
                  f"cross-round distinctness: '{cand_value}' is already the"
                  f" decisive value of another round in this scenario"
                  f" ('{_clash}'). A player who solved that round already has"
                  " this answer, so grading it again measures nothing. Choose a"
                  " different reading from this round's own evidence."
              )
            self._log(
                f"  [Round {round_id} Resolution] DISTINCTNESS CONCEDED:"
                f" free pick '{cand_value}' repeats another round's value"
                f" ('{_clash}') and the attempt budget is spent; keeping it"
                " rather than dropping the decisive check."
            )
          # ③: the round's CORRECT verdict label must be present, concise, and
          # distinct from the decisive value; an invalid outcome feeds the same
          # retry loop rather than shipping an empty/degenerate classification.
          _validate_round_decision_outcome(cand_outcome, cand_value)
          decisive_anchor = cand_anchor
          resolved_value = cand_value
          resolved_value_locus = cand_locus
          decision_outcome = cand_outcome
          break
        except Exception as e:
          attempt_errors.append(str(e))
          self._log(
              f"  [Round {round_id} Resolution Attempt {attempt} Failed]: {e}."
              " Retrying..."
          )
      else:
        # Non-fatal degrade (as before) but now LOGGED, not silent. An empty
        # resolved_value simply means no decisive-value check is seeded (the
        # scaffolder guards on ``resolved``), never a doomed check.
        self._log(
            f"  [Round {round_id} Resolution] degraded after {max_attempts}"
            f" attempts ({attempt_errors[-1] if attempt_errors else ''});"
            " proceeding with empty resolved_value."
        )
        decisive_anchor = ""
        resolved_value = ""
        resolved_value_locus = ""
        decision_outcome = ""

      # Publish what this round actually ended up grading on, so the rounds
      # after it can be held off the same value. Written into the CHECKPOINTED
      # outline rather than a side accumulator, for the same reason
      # ``previously_decided`` is derived from it: a resumed run keeps the
      # constraint instead of quietly starting over with an empty set. This is
      # not the same string as ``decisive_value`` whenever the binding was
      # released -- and released rounds are exactly the ones that collide.
      if resolved_value and isinstance(expanded_rounds.get(round_id), dict):
        expanded_rounds[round_id]["resolved_value"] = resolved_value

      # Choose a well-formed anchor: model value -> an existing visible anchor ->
      # a synthesized fallback. Always non-empty so the anchor switch at assembly
      # is populated and stays discoverable (materialize-time backfill attaches
      # an otherwise-unmatched anchor to a visible asset).
      decisive_anchor = _coerce_evidence_anchor(decisive_anchor)
      if not decisive_anchor:
        decisive_anchor = (
            round_visible_anchor_ids[0]
            if round_visible_anchor_ids
            else _coerce_evidence_anchor(f"{seed.slug}_{round_id}")
        )

      # Step 2: Generate Playbook Outline (Turn 1)
      outline_context = dict(current_context)
      outline_context["current_round_narrative"] = narrative_payload

      def _gen_round_outline(attempt, error_feedback):
        outline_instruction = (
            f"You are generating a step-by-step natural language debugging playbook outline for round {round_id}."
            f" Solve the task instruction: '{narrative_payload['rounds'][0]['instruction']}'."
            " Design a step-by-step list of action outline dicts containing 'step_number', 'description' (the command purpose), and 'expected_check' (what state/action should be verified, or null if no verification needed)."
            " Each step must follow standard debugging workflow principles (Discovery -> Inspection -> Wait -> Action -> Final Verification)."
            f" Produce between {seed.target_round_steps_min} and {seed.target_round_steps_max} steps (NEVER fewer than {seed.min_round_steps}), fully realizing all five stages IN ORDER:"
            " (1) at least 2 Discovery/Inspection steps that read the relevant visible assets and/or real workspace files;"
            " (2) at least 1 concrete Action step (write a file, send a targeted Slack message, or delegate to a subagent);"
            " (3) a Wait step so the effect of the action can settle;"
            " (4) a FINAL VERIFICATION step whose 'expected_check' RE-READS the specific visible asset or real workspace file that your Action changed or affected, to confirm the fix landed."
            " This final verification MUST be a re-read/inspection of the artifact you changed (e.g. re-open the updated dashboard asset or re-read the edited config file) and MUST NOT be another status question sent to a subagent."
            " Avoid padding: every step must be a distinct, non-redundant operation."
            " Only plan delegation for tasks requiring specific subagent capabilities (e.g. database queries, file writing)."
            f" The ONLY visual assets visible/available in this round are:\n{visible_assets_text}\n"
            " Check the updates in the history (`history_updates`) and the current round's narrative updates to verify which files are added."
        )
        # B/C: pin the FINAL VERIFICATION to the round's decisive value so the
        # last step grades a real outcome instead of a generic "looks fixed".
        if resolved_value:
          _outcome_hint = (
              f" and record the decision/verdict '{decision_outcome}'"
              if decision_outcome
              else ""
          )
          outline_instruction += (
              f" DECISIVE OUTCOME: solving this round hinges on the concrete value"
              f" '{resolved_value}'{_outcome_hint}. Plan an Action step that WRITES"
              f" a resolution report (a data/report file, not code) capturing this"
              f" outcome, and a FINAL VERIFICATION step whose 'expected_check'"
              f" re-reads that report to confirm the exact value"
              f" '{resolved_value}' is present; do not settle for a generic"
              f" 'looks fixed' check."
          )
        # M8/CL2 (L1): when the seed forces an async trajectory, the round's
        # decisive information only becomes available AFTER a delegated sub-agent
        # finishes (a delayed completion). Upgrade the cosmetic "Wait" stage (3)
        # into a REQUIRED delegate -> wait -> read triad so the gold genuinely
        # goes async and the temporal grader has something on-time to assert. No
        # effect for non-opted seeds (E6).
        if seed.force_async_wait:
          outline_instruction += (
              " ASYNC (required): the decisive information for THIS round comes"
              " from a sub-agent that only finishes AFTER a delay. The Action ->"
              " Wait stages MUST be: (a) a 'delegate' step handing the"
              " investigation/query to that sub-agent; (b) a 'Wait' step long"
              " enough for the sub-agent to complete; (c) a step that READS and"
              " INCORPORATES the sub-agent's returned result into the resolution."
              " Do NOT resolve the round from any value available BEFORE the"
              " wait -- the answer is only knowable once the delegated sub-agent"
              " reports back."
          )
        # A1: steer THIS round's trajectory to differ from earlier rounds.
        if prior_rounds_brief:
          outline_instruction += prior_rounds_brief
        if trajectory_hooks:
          outline_instruction += f"\n\nDIFFICULTY CHALLENGES CONSTRAINT:\n{trajectory_hooks}"
        # The outline drives the commands, so the "act, don't just look"
        # requirement has to land here too -- otherwise Layer 3a is asked to
        # author a terminal command the outline never planned for.
        outline_instruction += mechanic_instruction
        outline_instruction += ' Return JSON format: {"outline": [{"step_number": 1, "description": "Verify DB stats", "expected_check": "Accessed DB dashboard"}]}.'
        if error_feedback:
          outline_instruction += f"\n\n⚠️ PREVIOUS ATTEMPT FAILED:\n{error_feedback}\nPlease correct the structure in your new response."

        outline_payload = self.client.generate_layer(
            "conflict_layer3_round_outline",
            {
                "seed": seed_dict,
                "context": outline_context,
                "slot": {
                    "round_index": round_idx,
                    "target_rounds": target_rounds,
                    "round_outline": round_outline,
                    "instruction": outline_instruction,
                    "currently_visible_assets": currently_visible_assets
                },
            },
        )
        _validate_generated_round_outline(
            outline_payload, min_steps=seed.min_round_steps
        )
        return outline_payload

      outline_payload = self._generate_with_retry(
          _gen_round_outline,
          attempts=max_attempts,
          label=f"Round {round_id} Outline",
          failure_message=f"Failed to generate round {round_id} outline after {max_attempts} attempts.",
          include_last_error=True,
      )

      # Step 3: Translate outline -> command lines (LLM 3a) -> deterministic
      # checker scaffold (code) -> minimal semantic fields (LLM 3b), all inside
      # one shared retry loop. See layer_split_design.md section 4.1.
      steps_success = False
      steps_payload = {}
      steps_context = dict(outline_context)
      steps_context["current_round_outline"] = outline_payload

      # Paths a delegated mock will materialize this round (or materialized in an
      # earlier one). Snapshotted here, AFTER the resolution loop's mock-leak
      # rewrites have settled, and consumed only by the file-path grounding gate
      # below so the gold may legitimately read a file its own sub-agent wrote.
      round_mock_created_paths = _collect_mock_created_paths(
          narrative_payload, prior_rounds=rounds
      )

      attempt_errors: list[str] = []
      for attempt in range(1, max_attempts + 1):
        try:
          # Workspace file paths the solving agent can actually read this round.
          # This used to be extracted inline as "base workspace + THIS round's
          # workspace_adds", which silently omits the middle term: a task for
          # round rN ships the workspace accumulated over r1..rN, so from r03 on
          # the prompt advertised a SHORTER list than the real workspace and the
          # author LLM improvised the difference. ``_collect_round_workspace_texts``
          # is the same collector the confidentiality gate already trusts, and it
          # folds in every earlier round's adds. Insertion order (base -> prior
          # adds -> this round's adds) is preserved on purpose:
          # ``_ensure_final_verification_command`` uses ``valid_paths[-1]`` as its
          # fallback re-read target and should keep picking the newest file.
          valid_paths = list(round_workspace_texts.keys())

          # Split that list by WHEN the file becomes readable. A round's own
          # ``workspace_adds`` are not part of the shipped
          # ``initial_state_overrides``; the materializer compiles them into
          # ``background_events`` of type ``file_update`` carrying a
          # ``trigger_time``, and virtual time only advances on an explicit
          # ``wait`` (``environment.py`` ``_advance_time``). So a step that reads
          # one of them BEFORE waiting past its trigger hard-fails exactly like a
          # missing path. Measured on v5, 41 of the 51 unreadable gold reads were
          # this, not an invented path -- the defect the flat ``valid_paths`` list
          # could not express.
          _immediate_texts = _collect_round_workspace_texts(
              layer1b_payload, {}, prior_updates=updates
          )
          immediate_paths = [p for p in valid_paths if p in _immediate_texts]
          delayed_arrivals = _collect_delayed_arrivals(
              narrative_payload, set(_immediate_texts)
          )

          # --- Layer 3a: author command lines only (the only taint-prone job).
          commands_instruction = (
              f"You are translating a natural language playbook outline for round {round_id} into concrete command lines ONLY."
              f" Outline:\n{json.dumps(outline_payload['outline'], indent=2)}\n\n"
              " For each outline step, emit a dictionary containing 'step_number' (int) and 'command' (the exact command line string to run)."
              " Do NOT design evaluation checkers or verification schemas here; those are derived deterministically from your commands.\n"
              f" {allowed_prompt}\n"
              f" CRITICAL: You MUST use the exact, full subagent names from the character cast and agent list in all delegate commands. Mapping registry: {character_map}. (e.g. use 'Devon_Operations_Coordinator' instead of 'Devon', 'Casey_Database_Admin' instead of 'Casey', 'Blair_Backend_Developer' instead of 'Blair'). Never use short/first names.\n"
              f" CRITICAL (FILE PATHS): files readable right now: {immediate_paths}."
              " These already exist: to add to one, use 'file append' -- that is"
              " the correct verb and it is preferred over re-creating the file."
              " Use 'file create' ONLY for a path that appears nowhere in this"
              " instruction; 'file create' on an existing path overwrites it and"
              " loses the seeded content."
              + _format_delayed_arrival_directive(delayed_arrivals)
              + " Any other path does not exist: 'file read'/'file append'/'file"
              " delete' fail with 'File not found', emit no effect, and cannot be"
              " graded. Create a genuinely new path before its first use."
              " CRITICAL (JSON SAFETY): Quote ALL command argument values with SINGLE quotes, never double quotes -- e.g. file write --path 'workspace/report.md' --content 'resolved value 4.2%'. These command strings are embedded in a JSON document delimited by double quotes, so a literal double quote inside a command breaks the JSON; always use single quotes."
          )
          # The evals hooks are written as GRADING requirements ("assert the
          # lock was acquired before the edit", "at least 4 distinct containment
          # actions"). They are kept here because checks are derived from the
          # commands: an assertion the gold never performs can never be graded,
          # so the command author has to satisfy them first. They are ALSO sent
          # to the checker stage below so the semantic fields it authors agree
          # with the same contract instead of restating a generic intent.
          if evals_hooks:
            commands_instruction += (
                f"\n\nGRADING CONTRACT -- your commands MUST make each of these"
                f" assertable:\n{evals_hooks}"
            )
          if protocol_contract:
            commands_instruction += protocol_contract
          # The contract above is conditional ("if you use one of these,
          # chain the prerequisites"); this makes using one mandatory on the
          # selected rounds. The hard gate below rejects a trajectory that
          # lacks it, so steering here saves a retry.
          commands_instruction += mechanic_instruction
          # M8/CL2 (L1): force the concrete command lines to realize the async
          # triad (delegate -> wait -> read). The hard gate below
          # (``_has_delegate_then_wait``) rejects a trajectory that lacks it, so
          # this steers the model to pass on the first attempt. No-op for
          # non-opted seeds (E6).
          if seed.force_async_wait:
            commands_instruction += (
                " ASYNC TRAJECTORY (required): include, IN ORDER, (1) an"
                " 'agent delegate ...' command to the sub-agent that owns the"
                " decisive finding, (2) a later 'wait <seconds>' command that"
                " advances virtual time past the sub-agent's completion, and (3)"
                " AFTER that wait, a command that reads or acts on the"
                " sub-agent's reply (e.g. 'message search' on the reply channel,"
                " or the resolution-report write that incorporates it). The"
                " 'wait' MUST come AFTER the 'agent delegate'. A synchronous"
                " 'message send' alone does NOT satisfy this requirement."
            )
          if seed.require_final_verification:
            commands_instruction += (
                " FINAL VERIFICATION: the command immediately before the terminal DONE"
                " (i.e. right after the final wait) MUST re-read the asset or workspace file"
                " that your action step changed or affected -- e.g."
                " 'asset open --id <changed_asset_id>' or 'file read --path <changed_path>'"
                " (the path must be one of the valid workspace paths above) -- to confirm the"
                " fix landed. Do NOT use a Slack message as the verification step."
            )
          if resolved_value:
            if valid_paths:
              # ⑤ prevent-by-construction: the gold must MATERIALIZE its conclusion
              # into a durable, inspectable data/report artifact (never source
              # code), so the decisive value -- and the round's decision_outcome
              # verdict -- are graded against a real written outcome. The
              # scaffolder derives deterministic file_check(s) from THIS write, so
              # the checks are consistent with the gold by construction.
              _outcome_clause = (
                  f" and the decision/verdict '{decision_outcome}'"
                  if decision_outcome
                  else ""
              )
              commands_instruction += (
                  " RESOLUTION REPORT (required): before the final wait, include a"
                  " step that WRITES a resolution report to a NON-code data file"
                  " (a .md/.txt/.json report-style path, chosen from the valid"
                  " workspace paths above -- never a source/script file). Its"
                  f" content MUST state, verbatim, the decisive value"
                  f" '{resolved_value}'{_outcome_clause}. Then the FINAL"
                  " VERIFICATION step MUST re-read THAT SAME report file to confirm"
                  " the resolution landed."
              )
            else:
              commands_instruction += (
                  f" That final re-read MUST target the artifact whose content proves"
                  f" the value '{resolved_value}', so the deterministic file/asset"
                  f" check can confirm the resolution."
              )
          # Name the channels. The graded channel identity is a session's
          # DISPLAY name ('#ops-triage'), derived downstream at materialization,
          # while this author's background payload carries only the raw
          # session_id ('channel_ops_triage') -- so it has never had any way to
          # learn what the channels are actually called. It does not guess
          # timidly either: measured on v6 it INVENTS a name that suits the
          # story and then uses it consistently for all 10 rounds. Four
          # scenarios wrote 54 '--channel' values naming a channel that does not
          # exist, and because the checker stage authors criteria FROM these
          # commands, 37 criteria ended up naming the invented channel and 0
          # named a real one -- criteria no solver can satisfy by exploring the
          # environment, i.e. lost points that have nothing to do with the
          # conflict under test. Telling the author the vocabulary is the fix;
          # the gate after generation is only its enforcement.
          _chan_names, _ = _channel_vocabulary(
              layer1b_payload.get("sessions", []) if layer1b_payload else [],
              narrative_payload.get("updates", []),
          )
          if _chan_names:
            commands_instruction += (
                " CHANNEL VOCABULARY (required): the ONLY chat channels that"
                " exist in this scenario are"
                f" {', '.join(repr(c) for c in _chan_names)}. Every"
                " '--channel' argument MUST be one of those exact strings,"
                " copied VERBATIM including the leading '#'. Do NOT invent a"
                " channel name however well it fits the story, and do NOT pass"
                " a raw session id: a message sent to a channel that does not"
                " exist is invisible to every other participant."
            )
          commands_instruction += ' The final command MUST be "DONE". Return JSON format: {"commands": [{"step_number": 1, "command": "file list"}, {"step_number": 2, "command": "DONE"}]}.'
          if attempt_errors:
            commands_instruction += f"\n\n⚠️ PREVIOUS ATTEMPT FAILED WITH VALIDATION ERROR:\n{_format_retry_feedback(attempt_errors)}\nPlease correct this error in your new output."

          commands_payload = self.client.generate_layer(
              "conflict_layer3_round_commands",
              {
                  "seed": seed_dict,
                  "context": steps_context,
                  "slot": {
                      "round_index": round_idx,
                      "target_rounds": target_rounds,
                      "round_outline": round_outline,
                      "instruction": commands_instruction,
                      "currently_visible_assets": currently_visible_assets
                  },
              },
          )

          # Normalize command entries and guarantee a DONE terminator before
          # validating each one against the tool spec.
          raw_commands = commands_payload.get("commands", []) or []
          norm_commands = []
          for cmd_i, cmd_entry in enumerate(raw_commands):
            if isinstance(cmd_entry, dict):
              cmd_str = str(cmd_entry.get("command", "")).strip()
              cmd_sn = cmd_entry.get("step_number", cmd_i + 1)
            else:
              cmd_str = str(cmd_entry).strip()
              cmd_sn = cmd_i + 1
            if cmd_str:
              # Repair invalid POSIX quoting (e.g. an apostrophe escaped as
              # ``\'`` inside single quotes) before the command is validated
              # against the tool spec, which would otherwise raise a shell parse
              # error and burn a retry / drop the scenario.
              cmd_str = normalize_command_quoting(cmd_str)
              norm_commands.append({"step_number": cmd_sn, "command": cmd_str})
          if not norm_commands or norm_commands[-1]["command"].strip().upper() != "DONE":
            next_sn = (norm_commands[-1]["step_number"] + 1) if norm_commands else 1
            norm_commands.append({"step_number": next_sn, "command": "DONE"})
          # M8/CL2 (F6): deterministically repair a missing ``delegate -> wait``
          # async sub-trajectory (LLM tends to write a synchronous ``message
          # send`` instead) BEFORE the final-verification repair, so its injected
          # inspect lands after the delegate action, and before the gate below --
          # which would otherwise pure-reject and burn every retry. No-op for
          # non-opted seeds (E6) and when already async.
          _round_mocks = (
              narrative_payload.get("rounds", [{}])[0].get("mock_subagents", [])
              if narrative_payload.get("rounds")
              else []
          )
          if seed.force_async_wait:
            _ensure_async_delegate_wait(
                norm_commands, _round_mocks, allowed_families
            )
          # Deterministically repair a missing final-verification step instead of
          # letting the (much later) step validator fail and burn a whole retry.
          if seed.require_final_verification:
            _visible_ids = [
                str(a.get("id"))
                for a in currently_visible_assets
                if isinstance(a, dict) and a.get("id")
            ]
            _ensure_final_verification_command(
                norm_commands, valid_paths, _visible_ids, allowed_families
            )
          _validate_generated_round_commands({"commands": norm_commands}, allowed_families)

          # Make the round speak the scenario's own channel vocabulary -- the
          # gold's commands AND the sub-agent effects that reply to them, since
          # a reply posted to a channel that does not exist is just as invisible
          # as a gold message sent there. Ordered before the checker stage so the
          # criteria are authored from already-corrected commands.
          #
          # That ordering used to be the WHOLE argument ("a channel corrected
          # now is a criterion corrected for free"), and it was not enough: it
          # only holds while the vocabulary being corrected TO is the real one.
          # On v7 it was not -- the gate read the raw session dict's 'channel'
          # key while materialization read three spellings, so the gate enforced
          # a session_id-derived vocabulary, rejected the real names, and the
          # criteria faithfully inherited the wrong ones. The vocabulary is now
          # single-sourced (``_normalize_session_channels``) and the criteria
          # carry their OWN channel gate downstream, so correctness no longer
          # rests on this ordering alone.
          #
          # Provable aliases ('ops-triage', 'channel_ops_triage') are rewritten
          # silently; a name that matches no shipped session is bounced back to
          # the author, the only party with the semantics to choose between two
          # real channels. No-op when the scenario has no vocabulary at all.
          _chan_stats: dict[str, int] = {}
          _unresolved_chans = _canonicalize_gold_channels(
              norm_commands,
              layer1b_payload.get("sessions", []) if layer1b_payload else [],
              mock_subagents=_round_mocks,
              updates=narrative_payload.get("updates", []),
              last_attempt=(attempt >= max_attempts),
              log=self._log,
              stats=_chan_stats,
          )
          self._blind_bound_channels += _chan_stats.get("blind_bound", 0)
          if _unresolved_chans:
            _valid_chans, _ = _channel_vocabulary(
                layer1b_payload.get("sessions", []) if layer1b_payload else [],
                narrative_payload.get("updates", []),
            )
            raise ValueError(
                "These '--channel' values name a channel that does not exist in"
                f" this scenario: {', '.join(repr(c) for c in sorted(set(_unresolved_chans)))}."
                " A message sent there is invisible to every other participant,"
                " and the grading criterion derived from it would be"
                " unsatisfiable. Re-issue those commands using ONLY these exact"
                f" channels: {', '.join(repr(c) for c in _valid_chans)}."
                " Copy the string verbatim, including the leading '#'."
            )

          # Ground every file command in a path that can actually exist at that
          # point in the trajectory. ``file read/append/delete`` hard-fail on a
          # missing path and emit NO effect, so a step built on an invented path
          # carries a check that not even the gold can satisfy (see
          # ``_validate_file_path_grounding``). The append -> create rewrite runs
          # first because it is the one defect with a unique deterministic repair
          # and it also clears the reads that ``_ensure_final_verification_command``
          # spliced in against that same not-yet-existing path.
          _round_existing_paths = valid_paths + sorted(
              round_mock_created_paths - set(valid_paths)
          )
          _n_repaired = _ensure_file_writes_materialize(
              norm_commands, set(_round_existing_paths)
          )
          if _n_repaired:
            self._log(
                f"  [PathGrounding] {round_id}: rewrote {_n_repaired} 'file"
                " append' to 'file create' (target did not exist yet)."
            )
          # Insert the wait a mid-round arrival needs on EVERY attempt, like the
          # append -> create rewrite above and for the same reason: the repair is
          # unique and deterministic (the file does arrive, just later; the only
          # token missing is the wait), so spending a retry on it buys nothing.
          # It also has to run early because a retry CANNOT fix this class of
          # defect when we caused it ourselves: ``_ensure_final_verification_
          # command`` falls back to re-reading ``valid_paths[-1]``, i.e. the
          # newest file, which is precisely this round's delayed add. Gating this
          # behind the last attempt would burn attempts 1 and 2 re-prompting the
          # author for a step the author never wrote.
          _n_waits = _ensure_wait_before_delayed_reads(
              norm_commands, delayed_arrivals
          )
          if _n_waits:
            self._log(
                f"  [PathGrounding] {round_id}: inserted {_n_waits} 'wait'"
                " before a read of a file that only arrives mid-round."
            )
          # A3-style final-attempt escape. Hard-reject while retries remain (the
          # author should re-write a coherent trajectory now that it has been
          # told which paths are real); on the last attempt repoint the command
          # instead, because exhausting the budget discards the whole scenario
          # and 9.5% of v4 rounds across 14 of 60 seeds would enter this gate.
          if attempt >= max_attempts:
            _n_reground = _reground_missing_file_reads(
                norm_commands, _round_existing_paths
            )
            if _n_reground:
              self._log(
                  f"  [PathGrounding] {round_id}: final attempt -- repointed"
                  f" {_n_reground} file command(s) at a real artifact rather"
                  " than discarding the scenario."
              )
          _validate_file_path_grounding(
              norm_commands, set(_round_existing_paths), delayed_arrivals
          )

          # M8/CL2 (F6): when the seed opts into forced async, the reference
          # trajectory MUST actually go async -- delegate to a sub-agent and THEN
          # ``wait`` past its completion -- otherwise the injected within-window
          # temporal check (Deliv 4/8) has no delayed effect to grade and the
          # ablation gate (no-wait => fail) is vacuous. Raising here feeds the
          # standard retry loop with feedback. No-op for non-opted seeds (E6).
          if seed.force_async_wait and not _has_delegate_then_wait(
              norm_commands, allowed_families
          ):
            raise ValueError(
                "This seed requires a forced-async trajectory: include an "
                "'agent delegate ...' command followed by a later 'wait <seconds>' "
                "that advances virtual time past the sub-agent's completion, then a "
                "step that consumes the sub-agent's reply. A synchronous "
                "'message send' alone does NOT satisfy this."
            )

          # On the rounds selected by ``_round_requires_mechanic``, the gold
          # must ACT ON the conflict protocol, not just read its status; on the
          # final selected round it must COMPLETE it. Enforced rather than only
          # prompted because the checker scaffold derives checks from these
          # commands: if the gold never performs the protocol, the plugin's
          # declared effect is never graded, which is exactly the
          # 92.3%-touched / 29.7%-performed gap measured on the corpus. Raising
          # feeds the standard retry loop with feedback. No-op when the scenario
          # routes to no mechanic plugin.
          if round_requires_terminal and not _commands_include(
              norm_commands, mech.terminals
          ):
            raise ValueError(
                "This round must COMPLETE the coordination protocol, not only"
                " query or advance it. Include at least one of these commands"
                f" (with its prerequisites satisfied first):"
                f" {', '.join(repr(t) for t in mech.terminals)}."
                " Status/read-only commands alone do NOT satisfy this."
            )
          # The advance requirement carries an A3-style final-attempt escape:
          # measured against the corpus, 59.4% of selected rounds would not
          # satisfy it unprompted, and a hard gate at that rate would discard
          # most scenarios rather than make them harder. The completion gate
          # above stays hard -- it is what makes the mechanic gradable at all,
          # and it fires in only 10% of scenarios.
          if (
              round_requires_mechanic
              and attempt < max_attempts
              and not _commands_include(norm_commands, mech.protocol)
          ):
            raise ValueError(
                "This round must ADVANCE the coordination protocol, not only"
                " query its state. Include at least one of these commands"
                f" (with its prerequisites satisfied first):"
                f" {', '.join(repr(t) for t in mech.protocol)}."
                " Status/read-only commands alone do NOT satisfy this."
            )

          # A3 (anti-repetition): reject a trajectory identical to an earlier
          # round and retry with feedback so consecutive rounds are materially
          # distinct (root cause of duplicated rounds like r05==r06). On the
          # final attempt we accept rather than abort -- repetition should never
          # discard an otherwise-valid scenario.
          this_sig = _round_command_signature(
              [c["command"] for c in norm_commands]
          )
          prior_sigs = {
              _round_command_signature(list(r.get("solution_commands", []) or []))
              for r in rounds
          }
          if this_sig and this_sig in prior_sigs and attempt < max_attempts:
            raise ValueError(
                "This round's solution trajectory is identical to an earlier "
                "round. Design a materially different sequence of actions "
                "(different evidence, targets, files, or decisions) for this round."
            )

          # --- Code: deterministic checker scaffolds from the command lines.
          round_asset_ids: list[str] = []
          for a in currently_visible_assets + narrative_payload.get("assets", []):
            if isinstance(a, dict) and isinstance(a.get("id"), str):
              round_asset_ids.append(a["id"])
          # S6: the declaration belongs to the value S1 decided, so it applies
          # only while the round is still resolving to THAT value. A resolver
          # that broke its binding and picked its own string is describing
          # something the declaration never saw.
          _decided = str(round_outline.get("decisive_value") or "").strip()
          _decided_citable = bool(
              _decided
              and round_outline.get("decisive_value_citable")
              and _fold_for_grounding(resolved_value)
              == _fold_for_grounding(_decided)
          )
          scaffolds, semantic_needed = _scaffold_round_checkers(
              norm_commands, allowed_families, round_asset_ids,
              resolved_value=resolved_value,
              resolved_value_locus=resolved_value_locus,
              decision_outcome=decision_outcome,
              resolved_value_citable=_decided_citable,
          )

          # --- Layer 3b: fill ONLY the semantic fields, and only if any remain.
          sem_map = {}
          if semantic_needed:
            # C3: align each requested semantic field with the outline step's
            # intended verification ('expected_check') so the LLM grades the right
            # thing instead of restating the command's surface intent.
            outline_by_sn = {}
            for _os in (outline_payload.get("outline") or []):
              if isinstance(_os, dict) and _os.get("step_number") is not None:
                outline_by_sn[_os["step_number"]] = _os
            target_lines = []
            for target in semantic_needed:
              _ec = str(
                  (outline_by_sn.get(target["step_number"], {}) or {}).get(
                      "expected_check", ""
                  )
                  or ""
              ).strip()
              _ec_hint = (
                  f' It must verify the outline\'s intended check: "{_ec}".'
                  if _ec
                  else ""
              )
              if target["field"] == "criterion":
                target_lines.append(
                    f"- step_number {target['step_number']} (command: {target['command']}): provide 'criterion', a concise sentence describing the intent of this delegation/message.{_ec_hint}"
                )
              else:
                _contains_hint = _ec_hint
                # B3: when the round has a decisive value, the 'contains' substring
                # is graded by literal membership, so it MUST be the exact token
                # verbatim (no paraphrase/rounding) or replay will never match.
                if resolved_value:
                  _contains_hint += (
                      f" If this step materializes the round's decisive value,"
                      f" 'contains' MUST be the exact literal token"
                      f" '{resolved_value}' verbatim (it is matched as a literal"
                      f" substring, not semantically)."
                  )
                target_lines.append(
                    f"- step_number {target['step_number']} (command: {target['command']}): provide 'contains', an expected substring that proves the written file's success.{_contains_hint}"
                )
            checkers_instruction = (
                f"For round {round_id}, provide ONLY the missing semantic checker fields for the following steps."
                " The checker schema and all deterministic fields (command text, file path, asset id) are already fixed; do NOT restate or change them.\n"
                + "\n".join(target_lines)
                + '\n Return JSON format: {"fields": [{"step_number": 1, "criterion": "asks the DB admin to confirm the deadlock"}]}.'
            )
            # C2: when this round has a decisive value, any criterion meant to
            # surface/confirm the resolution must assert that concrete value rather
            # than a vague intent, so offline grading checks a real outcome.
            if resolved_value:
              # The worked example here used to be "confirms the resolved value
              # <X>", and the LLM copied that phrasing verbatim into 23.8% of all
              # criteria. "Resolved value" is pipeline jargon: the offline judge
              # reads it as "this issue has been settled", which directly
              # contradicts a gold message reporting <X> as the reason something
              # is BLOCKED -- so the check failed against its own gold. State the
              # value requirement without the jargon, and without an example the
              # model can copy.
              checkers_instruction += (
                  f"\nIf a step above is meant to surface or confirm this round's"
                  f" decisive outcome, its 'criterion' MUST assert the concrete"
                  f" value '{resolved_value}' verbatim, not merely the general"
                  f" intent. Say in plain operational language what the message"
                  f" must convey ABOUT that value (what it is, and what follows"
                  f" from it). Never describe it with pipeline jargon such as"
                  f" \"the resolved value\" or \"the decisive value\", and never"
                  f" assert the situation is resolved unless the round's outcome"
                  f" actually is."
              )
            # The plugins' evals hooks state what this conflict is supposed to
            # be graded ON. Previously they reached only the command author, so
            # the semantic fields authored here were written blind to the
            # mechanic and fell back to generic intent ("confirms the update").
            # The deterministic schema is still fixed by the scaffold -- this
            # only steers the free-text criterion/contains to speak to the
            # mechanic the plugin declares.
            if evals_hooks:
              checkers_instruction += (
                  f"\n\nWHAT THIS CONFLICT IS GRADED ON -- where a step above"
                  f" bears on one of these, its criterion/contains MUST speak to"
                  f" it directly rather than to a generic 'the step"
                  f" succeeded':\n{evals_hooks}"
              )
            # Name the channels for the CRITERIA author. Until now this stage
            # had no channel constraint at all -- the commands were canonicalized
            # just above on the theory that "the criteria are authored FROM these
            # commands, so a channel corrected now is a criterion corrected for
            # free". That held only while the vocabulary the commands were
            # corrected TO was the real one; when it was not, the criteria
            # faithfully inherited the wrong name. Stating the vocabulary here
            # makes the criteria independently correct instead of merely
            # consistent with whatever the commands happen to say.
            _crit_chan_names, _crit_chan_aliases = _channel_vocabulary(
                layer1b_payload.get("sessions", []) if layer1b_payload else [],
                narrative_payload.get("updates", []),
            )
            if _crit_chan_names:
              checkers_instruction += (
                  "\n\nCHANNEL VOCABULARY (required): if a criterion names a"
                  " chat channel, it MUST be one of"
                  f" {', '.join(repr(c) for c in _crit_chan_names)}, copied"
                  " VERBATIM including the leading '#'. A criterion that names"
                  " any other channel is unsatisfiable -- the solver posts where"
                  " the sessions it was given actually are, and loses the point"
                  " for a reason unrelated to the conflict. Prefer describing"
                  " WHAT must be conveyed over WHERE it was posted; name a"
                  " channel only when the routing itself is the thing graded."
              )
            if attempt_errors:
              checkers_instruction += f"\n\n⚠️ PREVIOUS ATTEMPT FAILED WITH VALIDATION ERROR:\n{_format_retry_feedback(attempt_errors)}\nPlease correct this error in your new output."

            checkers_payload = self.client.generate_layer(
                "conflict_layer3_round_checkers",
                {
                    "seed": seed_dict,
                    "context": steps_context,
                    "slot": {
                        "round_index": round_idx,
                        "target_rounds": target_rounds,
                        "round_outline": round_outline,
                        "instruction": checkers_instruction,
                        "semantic_targets": semantic_needed,
                        "currently_visible_assets": currently_visible_assets
                    },
                },
            )
            _validate_generated_round_checkers(checkers_payload)
            sem_map = _parse_semantic_fields(checkers_payload)

            # Criterion channel gate. The commands were canonicalized before
            # this stage, but a criterion is free prose and can name a channel
            # the commands never mention, so consistency with the gold does not
            # imply satisfiability for the solver. Bounce it back into this same
            # retry loop with the real vocabulary in the feedback.
            #
            # On the LAST attempt the finding is recorded and the round is kept
            # rather than discarded: losing the scenario would cost a whole
            # conflict archetype from the corpus, which is a worse trade than
            # one imperfect criterion. The counter is what acceptance reads.
            _bad_crit_chans = _unknown_criterion_channels(
                [
                    str(v)
                    for fields in sem_map.values()
                    for v in (
                        fields.values() if isinstance(fields, dict) else []
                    )
                    if isinstance(v, str)
                ],
                _crit_chan_aliases,
            )
            # Never bounce a token that is part of the round's decisive value: a
            # 'contains' check is graded by literal membership and MUST carry
            # that value verbatim (B3), so demanding a rewrite here would trade
            # an unsatisfiable channel for an unsatisfiable decisive check.
            if _bad_crit_chans and resolved_value:
              _rv = str(resolved_value).lower()
              _bad_crit_chans = [c for c in _bad_crit_chans if c.lower() not in _rv]
            if _bad_crit_chans:
              if attempt < max_attempts:
                raise ValueError(
                    "These grading criteria name a chat channel that does not"
                    " exist in this scenario:"
                    f" {', '.join(repr(c) for c in _bad_crit_chans)}. No solver"
                    " can satisfy them -- the sessions it is given are named"
                    f" {', '.join(repr(c) for c in _crit_chan_names)}. Rewrite"
                    " those criteria using ONLY those exact names (verbatim,"
                    " including the leading '#'), or drop the channel reference"
                    " and state what the message must CONVEY instead."
                )
              self._unsatisfiable_criterion_channels += len(_bad_crit_chans)
              self._log(
                  "  [CriterionChannel] final attempt -- keeping round"
                  f" {round_id} with {len(_bad_crit_chans)} criterion"
                  " channel(s) that name no shipped session:"
                  f" {', '.join(repr(c) for c in _bad_crit_chans)}"
              )

          # --- Code: merge scaffolds + semantic fields back into the legacy
          # steps shape, then keep the original validator as a safety net.
          _apply_round_semantic_fields(scaffolds, sem_map)
          # Gold-satisfiability gate: grade the criteria we just wrote against
          # the trajectory we just wrote, using the same judge + the same
          # transcript rendering the real grader uses. A criterion its own gold
          # fails is unwinnable for every solver, so bounce it back into this
          # retry loop (both authors see the judge's reason) instead of shipping
          # dead weight.
          #
          # The mocks are handed over PROJECTED through materialization's own
          # sub-agent transforms. The authored mocks carry no ``reply_channel``
          # and un-reconciled effect channels -- both are filled in later, at
          # materialization -- so feeding them raw made the gate render replies
          # with no ``in #channel`` at all and judge a transcript the runtime
          # never produces (measured: 0/18 reconstructed line sets matched the
          # real replay, on v5 and v6 alike; with the projection and both reply
          # routes modelled, 40/40 on v6 with no predicted-but-absent lines).
          # Projecting reuses the materializer's functions rather
          # than restating them, so the prediction cannot drift from the thing
          # predicted.
          _gate_mocks = _project_round_subagents_for_gate(
              (
                  narrative_payload.get("rounds", [{}])[0].get(
                      "mock_subagents", []
                  )
                  if narrative_payload.get("rounds")
                  else []
              ),
              layer1b_payload.get("sessions", []) if layer1b_payload else [],
              [c["command"] for c in norm_commands],
          )
          _enforce_gold_satisfiable_message_criteria(
              scaffolds,
              norm_commands,
              _gate_mocks,
              last_attempt=(attempt >= max_attempts),
              log=self._log,
          )
          steps_payload = _steps_from_scaffolds(scaffolds)
          _validate_generated_round_steps(
              steps_payload,
              allowed_families,
              min_steps=seed.min_round_steps,
              require_final_verification=seed.require_final_verification,
          )
          steps_success = True
          break
        except Exception as e:
          attempt_errors.append(str(e))
          self._log(f"  [Round {round_id} Steps Translation Attempt {attempt} Failed]: {e}. Retrying...")

      if not steps_success:
        last_error = attempt_errors[-1] if attempt_errors else ""
        raise ScenarioLLMError(f"Failed to generate round {round_id} steps after {max_attempts} attempts. Last error: {last_error[:800]}")

      # Compile results into standard round structures
      sol_cmds = [step["command"] for step in steps_payload["steps"]]
      if not sol_cmds or sol_cmds[-1].strip().upper() != "DONE":
        sol_cmds.append("DONE")

      checks = []
      for step in steps_payload["steps"]:
        if step.get("checker") is not None:
          checks.append(step["checker"])
        # ③/⑤: a step may contribute additional deterministic checks (the
        # resolution report's decision_outcome assertion alongside its decisive
        # value); flatten them into the eval so both are graded.
        for _extra in (step.get("extra_checkers") or []):
          if _extra is not None:
            checks.append(_extra)

      # Honor the narrative's declared round_type (LLM may vary it, e.g.
      # investigation/decision) rather than hardcoding 'action'. The fallback
      # below still resolves to 'action', which is the EASIEST round type
      # measured (46.2% pass, against 26.3% for investigation) -- so a silent
      # fallback quietly buys the corpus difficulty it did not earn. Say so
      # when it fires instead of letting it pass unseen.
      _declared_round_type = narrative_payload["rounds"][0].get("round_type")
      if not _declared_round_type:
        self._log(
            f"  [RoundType] {round_id}: narrative declared no round_type;"
            " defaulting to 'action'. This biases the corpus toward the"
            " easiest round type -- check the narrative prompt if it recurs."
        )
      # M1: the assignment wins over whatever came back. Log DRIFT (the model
      # returned a different type than it was given) rather than swallowing it:
      # drift is the only observable signal of how well the generator complied,
      # and compliance is exactly what M1's value rests on, since the label
      # itself is inert downstream. A silent overwrite would make a generator
      # that ignored the directive look identical to one that obeyed it.
      if forced_round_type is not None:
        if _declared_round_type and str(_declared_round_type) != forced_round_type:
          self._log(
              f"  [RoundType] {round_id}: DRIFT -- assigned"
              f" '{forced_round_type}' but narrative returned"
              f" '{_declared_round_type}'; forcing the assignment. The prose was"
              " written against the returned type, so this round probably does"
              " not have the assigned shape."
          )
        _declared_round_type = forced_round_type
      round_payload = {
          "rounds": [{
              "round_id": round_id,
              "round_type": str(_declared_round_type or "action"),
              "instruction": narrative_payload["rounds"][0]["instruction"],
              "update_ids": narrative_payload["rounds"][0].get("update_ids", []),
              "mock_subagents": narrative_payload["rounds"][0].get("mock_subagents", []),
              "solution_commands": sol_cmds,
              # ③: record this round's graded conclusion so it flows through the
              # materializer into the round spec / ScenarioRound metadata.
              "decision_outcome": decision_outcome,
              "resolved_value": resolved_value,
              # I1 backstop: thread the decisive value's LOCUS to the shipped spec
              # so the post-generation confidentiality net can enforce vision-only
              # confinement for asset-locus rounds while correctly permitting
              # file/computed/unknown-grounded rounds.
              "resolved_value_locus": resolved_value_locus,
              "phase": phase.get("theme", "System Updates"),
          }],
          "updates": narrative_payload.get("updates", []),
          "evals": [{
              "eval_id": f"e_{round_id}",
              "round_id": round_id,
              # C1: bind the round's decisive anchor so the anchor-consistency
              # audits (consistency_layer/auditor/strict_validator) and VLM
              # grounding activate for this eval.
              "expected_anchor_ids": [decisive_anchor] if decisive_anchor else [],
              "checks": checks
          }],
          "assets": narrative_payload.get("assets", []),
      }

      # Consistency + self-repair: when this round is graded on the CONTENT of an
      # agent-produced report (a path-agnostic ``files_created`` deliverable
      # check), the instruction MUST direct the player to write that report --
      # otherwise the round is unwinnable no matter how capable the solver is.
      # Detect the mismatch and repair the instruction in place (LLM judge -> LLM
      # rewrite -> deterministic path-free fallback) so a graded deliverable
      # always has a matching directive.
      self._reconcile_instruction_with_checks(
          round_payload, round_id, round_idx, visible_asset_ids, seed_dict,
          current_context,
      )

      round_success = True


      # Finalize this round's outputs
      if "updates" in round_payload:
        updates.extend(round_payload["updates"])
      if "rounds" in round_payload:
        rounds.extend(round_payload["rounds"])
        # Scenario-level protocol-completion tracking (see ``mechanic_completed``
        # above). Read off the ACCEPTED trajectory so a round rejected by a later
        # validator never counts as having completed the protocol.
        if mech and not mechanic_completed:
          mechanic_completed = any(
              _commands_include(
                  [{"command": c} for c in (r.get("solution_commands") or [])],
                  mech.terminals,
              )
              for r in round_payload["rounds"]
          )
      if "evals" in round_payload:
        evals.extend(round_payload["evals"])
      # (Note: assets from progressive rounds are supported but
      # phase_expanded_assets is the primary source)
      if "assets" in round_payload:
        phase_expanded_assets.extend(round_payload["assets"])

      if checkpoint_path:
        self._save_checkpoint(
            checkpoint_path,
            layer0_phases,
            layer1_payload,
            layer1b_payload,
            updates,
            rounds,
            evals,
            phase_expanded_assets,
            expanded_rounds,
        )

    if len(rounds) != target_rounds:
      raise ScenarioLLMError(
          f"Progressive round generation failed: expected {target_rounds}"
          f" rounds, but only generated {len(rounds)} rounds. Check LLM output"
          " payload structures."
      )

    # Deduplicate round updates by update_id
    all_updates = {}
    for u in updates:
      if isinstance(u, dict) and "update_id" in u:
        all_updates[u["update_id"]] = u

    # A scenario with zero rounds is unusable. Keep the checkpoint (do not delete
    # below) and fail loudly so the caller can resume/retry rather than silently
    # emitting an empty scenario.
    if not rounds:
      raise ScenarioLLMError(
          f"Scenario '{seed.slug}' produced 0 rounds after generation."
      )

    # Channel-fairness summary for the run's acceptance report. Both numbers
    # should be 0; non-zero means some channel reference was decided for the
    # author by exhaustion rather than authored, which is how v7 shipped 433
    # phantom gold channels and 320 phantom criteria.
    if self._blind_bound_channels or self._unsatisfiable_criterion_channels:
      self._log(
          "  [ChannelFairness] blind-bound gold channels:"
          f" {self._blind_bound_channels}; criteria naming no shipped session:"
          f" {self._unsatisfiable_criterion_channels}"
      )

    # Success! Clean up checkpoint file
    if checkpoint_path and checkpoint_path.is_file():
      try:
        checkpoint_path.unlink()
      except Exception as e:
        self._log(f"Warning: Failed to delete checkpoint file: {e}")

    return {
        "summary": layer1_payload.get("summary", ""),
        "mock_subagents_global": layer1_payload.get("mock_subagents", []),
        "workspace": layer1b_payload.get("workspace", {}),
        "sessions": layer1b_payload.get("sessions", []),
        "assets": layer1b_payload.get("assets", []) + phase_expanded_assets,
        "updates": list(all_updates.values()),
        "rounds": rounds,
        "evals": evals,
        "agents": layer1_payload.get("agents", []),
        "interactions": layer1_payload.get("interactions", []),
        "platforms": self._platforms(layer0_phases),
    }

  def _reconcile_instruction_with_checks(
      self,
      round_payload: dict[str, Any],
      round_id: str,
      round_idx: int,
      visible_asset_ids: list[str],
      seed_dict: dict[str, Any],
      current_context: dict[str, Any],
  ) -> None:
    """Ensure a path-agnostic file deliverable has a matching write directive.

    When a round is graded on the CONTENT of an agent-produced report via the
    ``files_created`` effect stream (a discretionary-path deliverable), the player
    instruction must actually tell the player to WRITE that report; otherwise the
    round is unwinnable no matter how capable the solver is. This repairs the
    instruction IN PLACE rather than failing the round (mirroring
    ``_sanitize_leaked_asset_ids``): an LLM judge decides whether the instruction
    already directs a write; if not, a targeted LLM rewrite adds a path-free,
    asset-ID-free directive; if that rewrite is unavailable or unsafe, a
    deterministic path-free sentence is appended so the round stays winnable.
    Only ``effect``/``files_created`` deliverables are in scope.
    """
    evals = round_payload.get("evals") or []
    checks = (evals[0].get("checks") if evals else None) or []
    grades_created_file = any(
        isinstance(c, dict)
        and c.get("type") == "effect"
        and c.get("effect_type") == "files_created"
        for c in checks
    )
    if not grades_created_file:
      return
    rounds = round_payload.get("rounds") or []
    if not rounds:
      return
    round_data = rounds[0]
    instruction = str(round_data.get("instruction", "") or "")
    if not instruction:
      return

    # Call 1 (judge): does the instruction already direct writing a report file?
    if self._judge_instruction_directs_write(
        instruction, round_id, round_idx, seed_dict, current_context
    ):
      return

    self._log(
        f"  [Round {round_id} Reconcile] Round is graded on a written report but"
        " the instruction does not direct the player to write one; repairing."
    )

    # Call 2 (rewrite): add a path-free directive to write the report.
    rewritten = self._rewrite_instruction_add_report(
        instruction, round_id, round_idx, seed_dict, current_context
    )
    if rewritten:
      # Auto-clean any asset IDs the rewrite leaked (mirrors the narrative path),
      # then accept only if leak-free and free of future-round references.
      rewritten = _sanitize_leaked_asset_ids(rewritten, visible_asset_ids)
      lowered = rewritten.lower()
      leaks = any(aid and aid.lower() in lowered for aid in visible_asset_ids)
      if (
          rewritten.strip()
          and not leaks
          and not _visible_text_references_future_round(
              rewritten, round_index=round_idx
          )
      ):
        round_data["instruction"] = rewritten.strip()
        return

    # Fallback: deterministic, path-free, asset-free directive appended to the
    # already-validated (leak-free) original instruction. Guarantees the graded
    # deliverable has a matching directive even when the rewrite call fails.
    round_data["instruction"] = _append_report_directive(instruction)

  def _judge_instruction_directs_write(
      self,
      instruction: str,
      round_id: str,
      round_idx: int,
      seed_dict: dict[str, Any],
      current_context: dict[str, Any],
  ) -> bool:
    """LLM judge: does ``instruction`` already direct writing a report file?

    Conservative on failure: returns False (assume a repair is needed) so a graded
    deliverable is never left without a directive merely because a judge call
    errored out.
    """
    prompt = (
        "You are auditing a single round instruction given to a coordinator (the"
        " 'player'). This round is graded on the CONTENT of a report FILE the"
        " player is expected to WRITE into the workspace.\n"
        "Decide whether the instruction, exactly as written, already CLEARLY"
        " directs the player to write/create/save/produce a report, audit,"
        " summary, or memo FILE that records their findings and/or decision."
        " Delivering the resolution ONLY as a chat/Slack message, or merely"
        " investigating without writing a file, does NOT count.\n"
        "Instruction to audit:\n"
        f'"""\n{instruction}\n"""\n'
        'Return JSON: {"directs_write": true} or {"directs_write": false}.'
    )
    for attempt in range(1, 3):
      try:
        out = self.client.generate_layer(
            "audit_instruction_write_directive",
            {
                "seed": seed_dict,
                "context": current_context,
                "slot": {"round_index": round_idx, "instruction": prompt},
            },
        )
        val = out.get("directs_write")
        if isinstance(val, bool):
          return val
        if isinstance(val, str):
          return val.strip().lower() in ("true", "yes", "1")
        return bool(val)
      except Exception as e:  # pylint: disable=broad-except
        self._log(
            f"  [Round {round_id} Reconcile judge attempt {attempt} failed]: {e}."
        )
    return False

  def _rewrite_instruction_add_report(
      self,
      instruction: str,
      round_id: str,
      round_idx: int,
      seed_dict: dict[str, Any],
      current_context: dict[str, Any],
  ) -> str:
    """LLM rewrite: add a path-free directive to write the resolution report.

    Returns the rewritten instruction, or "" on failure (the caller then falls
    back to a deterministic append).
    """
    prompt = (
        "Rewrite the following round instruction so that, in addition to"
        " everything it already asks, it EXPLICITLY directs the player to WRITE a"
        " resolution report/audit/summary FILE into the workspace that records"
        " both the decisive evidence they found AND their final decision/verdict"
        " for the conflict.\n"
        "Hard constraints:\n"
        "- Preserve all existing tasks, names, channels, and intent; only add the"
        " missing 'write the report' directive and integrate it naturally.\n"
        "- Do NOT mention any specific file path or filename -- the location is"
        " the player's choice (the grade is on content, not location).\n"
        "- Do NOT expose any asset IDs; refer to assets conceptually.\n"
        "- Do NOT reference any future round.\n"
        "Instruction to rewrite:\n"
        f'"""\n{instruction}\n"""\n'
        'Return JSON: {"updated_instruction": "..."}.'
    )
    for attempt in range(1, 3):
      try:
        out = self.client.generate_layer(
            "repair_instruction_write_directive",
            {
                "seed": seed_dict,
                "context": current_context,
                "slot": {"round_index": round_idx, "instruction": prompt},
            },
        )
        new_instr = str(out.get("updated_instruction", "") or "")
        if new_instr.strip():
          return new_instr
      except Exception as e:  # pylint: disable=broad-except
        self._log(
            f"  [Round {round_id} Reconcile rewrite attempt {attempt} failed]:"
            f" {e}."
        )
    return ""

  def _platforms(self, layer0_phases: dict[str, Any] | None) -> list[str]:
    """Scenario platform vocabulary, rebuilt from checkpointable layer0_phases.

    Reads the LLM-authored ``platforms`` list, falling back to the platform_ids
    already present in ``global_assets_plan`` so the vocabulary survives a resume
    from a checkpoint written before the ``platforms`` key existed.
    """
    lp = layer0_phases or {}
    return _build_platform_vocab(lp.get("platforms"), lp.get("global_assets_plan"))

  def _presets_prompt_instructions(self) -> str:
    """Return formatted prompt instructions listing chart presets."""

    presets_desc = "Available General Chart Presets:\n"
    for pid, p in CHART_PRESETS.items():
      presets_desc += f"- Preset ID: '{pid}'\n"
      presets_desc += f"  Chart Type: '{p.chart_type}'\n"
      presets_desc += f"  Guidance: {p.guidance}\n"
      if p.series_names:
        presets_desc += (
            "  Default Series Names (please override via"
            f" custom_overrides.series_names): {p.series_names}\n"
        )

      if p.chart_type in {"bar_chart", "line_chart", "pie_chart"}:
        row_example = '{"label": "10:00", "value": 85}'
      elif p.chart_type in {
          "multi_line_chart",
          "grouped_bar_chart",
          "stacked_bar_chart",
      }:
        row_example = '{"label": "10:00", "values": [85, 15]}'
      elif p.chart_type == "scatter_plot":
        row_example = '{"label": "Sample 1", "x": 1.5, "y": 95.0}'
      elif p.chart_type == "gantt_chart":
        row_example = '{"label": "Step Name", "start": 0, "duration": 5}'
      else:
        row_example = '{"label": "...", "value": ...}'

      presets_desc += f"  Expected chart_rows format: List of {row_example}\n\n"

    presets_desc += "Available Caption Templates:\n"
    for cid, c in CAPTION_TEMPLATES.items():
      presets_desc += (
          f"- Template ID: '{cid}' (default if dict caption generated)\n"
      )
      presets_desc += f'  Template String: "{c.template_string}"\n'
      presets_desc += f"  Required Checklist items: {c.required_checklist}\n"

    instructions = (
        f"\n=== Visual Asset Guidelines ===\n"
        f"You can define visual assets. For charts"
        f" (generator='template_chart'), you should select a preset_id from the"
        f" list below. "
        f"To customize the chart (e.g., overriding title, subtitle, series"
        f" names, x_label, y_label, or styling theme to"
        f" 'cool'/'warm'/'default'), "
        f"use `custom_overrides`. You must populate `chart_rows` with dense"
        f" data rows matching the preset type. Alternatively, you can use "
        f"the mathematical data curve generator by specifying in "
        f"`custom_overrides`:\n"
        f"  `pattern`: String (one of 'flatline', 'spike', 'decay', "
        f"'periodic') OR list of strings (one per series)\n"
        f"  `pattern_params`: Dict OR list of dicts (one per series) "
        f"with keys:\n"
        f"    'baseline' (default 100.0), 'noise_std' (default 0.0),\n"
        f"    'spike_value' (default 0.0), 'spike_index' (index of spike),\n"
        f"    'start_value'/'end_value' (for decay),\n"
        f"    'amplitude'/'frequency' (for periodic).\n\n"
        f"{presets_desc}\n"
        f"For chart data format:\n"
        f"  `preset_id`: the selected preset ID.\n"
        f"  `chart_rows`: list of dicts conforming to the expected chart_rows"
        f" format above (can be omitted if using mathematical patterns).\n"
        f'  `custom_overrides`: dict of overridden fields (e.g. {{"title":'
        f' "Custom Title", "styling": {{"theme": "warm"}}}}).\n'
        f"  `caption`: can be a string directly, OR a dictionary matching the"
        f' variables in the Caption Template (e.g. {{"service_name": "...",'
        f' "visual_anomaly_details": "...", "anchors": "..."}}).\n'
        f"================================\n"
    )
    return instructions

  def _build_allowed_commands_prompt(self, allowed_families: list[str]) -> str:
    """Build system instructions restricting generated solution commands."""
    spec = load_agent_tool_spec()
    allowed_syntax = []

    families_set = {normalize_family_name(f) for f in allowed_families or []}
    for family in spec.families:
      spec_family_normalized = normalize_family_name(family.family)
      if not families_set or spec_family_normalized in families_set:
        for cmd in family.commands:
          allowed_syntax.append(cmd.syntax)

    if not allowed_syntax:
      return "You must only use the command: DONE"

    return (
        "CRITICAL: You are strictly restricted to using ONLY the following "
        "allowed command syntaxes for the solution trajectory. Do not use any "
        "other command verbs or formats:\n"
        + "\n".join(f"- {s}" for s in allowed_syntax)
    )


_PLATFORM_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

# Canonical fallback platform_id, used when a value is missing/unusable and no
# scenario vocabulary is available to remap onto.
_DEFAULT_PLATFORM_ID = "default_platform"

# platform_id values the model tends to hallucinate by copying adjacent prompt
# context (provider flags, generator names, kinds, agent roles, asset ids)
# instead of naming a real UI surface. Rejected during sanitization.
_PLATFORM_ID_BLOCKLIST = frozenset({
    "vertex", "template", "google_model", "mock", "gemini", "openai",
    "default", "default_platform", "none", "null", "na",
    "workspace", "vlm_evidence", "vlm_service", "asset", "assets",
    "platform", "platform_id", "chart", "image", "rich_text",
    "natural_image", "template_chart", "google_image",
    "diagram", "diagram_image", "sketch", "sketch_image",
})

# Fixed fallback vocabulary of realistic web-UI surfaces. The scenario's platform
# vocabulary is authored by the LLM per-scenario, but is backfilled from this set
# so every scenario has at least a few DISTINCT platforms even when the model
# under-specifies. The goal is diversity, not collapse onto a single dashboard.
_DEFAULT_PLATFORM_VOCAB = (
    "ops_console", "metrics_dashboard", "admin_portal",
    "incident_board", "status_page", "log_viewer",
)

# Soft floor for the number of distinct platforms per scenario. A scenario is
# acceptable with >= 2 distinct surfaces; more variety is encouraged softly (via
# the planning prompt) but never required. Used as a best-effort backfill target
# here and a WARN (never an abort) downstream.
_MIN_PLATFORM_VOCAB = 2


def _sanitize_platform_id(raw: Any) -> str:
  """Reduce an LLM-provided platform_id to a clean slug, or '' if unusable.

  platform_id names a web UI surface and must be a bare lowercase slug. The
  model frequently emits garbage copied from surrounding prompt context --
  provider flags (``vertex``/``template``/``google_model``), generator names
  (``template_chart``), agent roles (``Ops_Agent``), asset ids (``asset_bg_01``)
  or file paths/filenames (``docs/x.md``, ``chart.png``). Strip any directory
  and file extension, slugify, and reject a blocklist of known non-platform
  tokens. Returns '' when nothing usable remains so the caller can fall back to
  a scenario vocabulary member or the default platform.
  """
  if raw is None:
    return ""
  text = str(raw).strip().lower()
  if not text:
    return ""
  # Drop directory components (keep the final path segment).
  text = text.replace("\\", "/").rsplit("/", 1)[-1]
  # Drop a single trailing file extension (e.g. .png/.jpg/.md/.log/.json).
  text = re.sub(r"\.[a-z0-9]{1,5}$", "", text)
  # Slugify: collapse any run of disallowed chars into a single underscore,
  # preserving existing '_' and '-' separators (both are valid slug chars).
  text = re.sub(r"[^a-z0-9_-]+", "_", text).strip("_-")
  if not text:
    return ""
  # Agent-role ids (``*_agent``) and asset ids (``asset_*``) are never platforms.
  if text.endswith("_agent") or text.startswith("asset_"):
    return ""
  if text in _PLATFORM_ID_BLOCKLIST:
    return ""
  if not _PLATFORM_SLUG_RE.match(text):
    return ""
  return text


def _dedupe_preserve(items: list[str]) -> list[str]:
  """Order-preserving de-duplication that drops falsy entries."""
  seen: set[str] = set()
  out: list[str] = []
  for it in items:
    if it and it not in seen:
      seen.add(it)
      out.append(it)
  return out


def _build_platform_vocab(platforms_field: Any, global_assets_plan: Any) -> list[str]:
  """Build the scenario platform vocabulary (>= _MIN_PLATFORM_VOCAB slugs).

  Candidates are drawn first from the LLM-authored ``platforms`` list, then from
  the ``platform_id`` of each planned asset, each run through
  ``_sanitize_platform_id`` so garbage never enters the vocabulary. The result is
  deduped (order-preserving) and backfilled from ``_DEFAULT_PLATFORM_VOCAB`` so
  downstream always has a few distinct, realistic surfaces to spread assets
  across instead of collapsing onto one.
  """
  candidates: list[str] = []
  if isinstance(platforms_field, list):
    for p in platforms_field:
      candidates.append(_sanitize_platform_id(p))
  if isinstance(global_assets_plan, list):
    for a in global_assets_plan:
      if isinstance(a, dict):
        candidates.append(_sanitize_platform_id(a.get("platform_id")))
  vocab = _dedupe_preserve(candidates)
  # Backfill from the fixed fallback until we reach the soft minimum, skipping
  # any fallback already present so the vocabulary stays deduped.
  for fb in _DEFAULT_PLATFORM_VOCAB:
    if len(vocab) >= _MIN_PLATFORM_VOCAB:
      break
    if fb not in vocab:
      vocab.append(fb)
  return vocab


def _stable_platform_choice(key: str, vocab: list[str]) -> str:
  """Deterministically map an arbitrary key onto a vocabulary member.

  Uses a stable hash (``hashlib``, NOT the per-process-salted builtin ``hash``)
  so the same asset maps to the same platform across processes and checkpoint
  resumes. Used to spread out-of-vocabulary / missing platform_ids across the
  scenario vocabulary instead of collapsing them all onto a single surface.
  """
  if not vocab:
    return _DEFAULT_PLATFORM_ID
  digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
  return vocab[int(digest, 16) % len(vocab)]


def _normalize_and_prefix_assets(
    payload: dict[str, Any],
    allowed_platforms: list[str] | None = None,
) -> dict[str, Any]:
  if "assets" not in payload or not isinstance(payload["assets"], list):
    return payload
  import os
  vocab = allowed_platforms or []
  for idx, asset in enumerate(payload["assets"]):
    if not isinstance(asset, dict):
      continue
    platform_id = _sanitize_platform_id(asset.get("platform_id"))
    if vocab:
      # A scenario vocabulary is available: keep in-vocabulary choices (this
      # preserves the model's deliberate grouping and keeps versioned/evolving
      # copies of one asset on the same surface), but deterministically remap
      # anything missing or out-of-vocabulary onto a vocabulary member so assets
      # spread across platforms instead of collapsing onto a single default.
      if platform_id not in vocab:
        remap_key = str(asset.get("id") or asset.get("path") or idx)
        platform_id = _stable_platform_choice(remap_key, vocab)
    elif not platform_id:
      platform_id = _DEFAULT_PLATFORM_ID
    asset["platform_id"] = platform_id

    old_id = asset.get("id", "")
    # The downstream validators (`_validate_expanded_phase`,
    # `_validate_background_payload`) require the id/path filename to start with
    # ``basename(platform_id)_``. Prefix with the SAME basename here so
    # normalization always satisfies validation instead of tripping it (a
    # full-platform-id prefix would mismatch when platform_id contains a path
    # separator, burning the whole retry budget and aborting the scenario).
    prefix = f"{os.path.basename(str(platform_id))}_"

    # Auto-synthesize a missing id from the path/platform so the asset is
    # repairable rather than a hard validation failure.
    if not old_id:
      path_for_id = asset.get("path", "")
      base_name = os.path.splitext(os.path.basename(str(path_for_id)))[0]
      old_id = base_name or f"{os.path.basename(str(platform_id))}_asset"

    # Sanitize a path-like id before prefixing. When the model emits an id such
    # as ``workspace/flight_schedule`` and the platform basename is
    # ``workspace``, a naive prefix yields the doubled, slash-bearing
    # ``workspace_workspace/flight_schedule`` -- a phantom that never matches the
    # clean, timestamped snapshot id downstream (this is the shared root cause of
    # mis-scoped assets_accessed checks and asset-id double-prefix false
    # failures). Drop a leading ``basename(platform_id)/`` (the path form of the
    # prefix) and fold any remaining separators to ``_`` so the id is always a
    # single-prefixed, slash-free token.
    base = os.path.basename(str(platform_id))
    if base and old_id.startswith(f"{base}/"):
      old_id = old_id[len(base) + 1 :]
    old_id = old_id.strip("/").replace("/", "_")
    asset["id"] = old_id

    if not old_id.startswith(prefix):
      asset["id"] = f"{prefix}{old_id}"

    path = asset.get("path", "")
    if path:
      dir_part, filename = os.path.split(path) if "/" in path else ("", path)
      if not filename.startswith(prefix):
        new_filename = f"{prefix}{filename}"
        asset["path"] = os.path.join(dir_part, new_filename) if dir_part else new_filename

  return payload


def _validate_workspace_paths(files: list[dict[str, Any]], context_desc: str) -> None:
  """Asserts that all workspace file paths are safe, relative, and cleanly structured."""
  for f in files:
    if not isinstance(f, dict):
      raise ValueError(f"[{context_desc}] File entry must be a dictionary")
    path = f.get("path")
    if not path:
      raise ValueError(f"[{context_desc}] File entry missing 'path'")
    # Missing/empty file content is recoverable: materialize an empty file
    # rather than failing the whole task. Models routinely omit 'content' for
    # legitimately empty artifacts (e.g. `.gitkeep`, placeholder checksums),
    # so hard-failing here needlessly drops otherwise-valid scenarios. Coerce
    # to "" (and stringify non-str payloads) so the path check still runs.
    content = f.get("content")
    if content is None:
      f["content"] = ""
    elif not isinstance(content, str):
      f["content"] = str(content)
    
    path_str = str(path).strip()
    if path_str.startswith("/"):
        path_str = path_str.lstrip("/")
        f["path"] = path_str

    if "\\" in path_str:
      raise ValueError(
          f"[{context_desc}] File path '{path_str}' contains forbidden backslashes '\\'. Use '/' for directories."
      )
    if ".." in path_str.split("/"):
      raise ValueError(
          f"[{context_desc}] File path '{path_str}' contains forbidden directory traversal '..'."
      )


def _validate_background_payload(payload: dict[str, Any]) -> None:
  if "workspace" not in payload:
    raise ValueError("Background payload missing 'workspace'")
  workspace = payload["workspace"]
  if isinstance(workspace, list):
    workspace = {"files": workspace}
    payload["workspace"] = workspace

  if not isinstance(workspace, dict):
    raise ValueError("Workspace must be a dictionary or a list of files")
  if "files" not in workspace:
    raise ValueError("Workspace dictionary missing 'files' key")
  _validate_workspace_paths(workspace["files"], "Initial Background Workspace")

  if "sessions" not in payload:
    raise ValueError("Background payload missing 'sessions'")
  if "assets" in payload:
    assets = payload["assets"]
    if not isinstance(assets, list):
      raise ValueError("Assets must be a list")
    for asset in assets:
      if not isinstance(asset, dict):
        raise ValueError("Asset spec must be a dictionary")
      if not asset.get("id"):
        raise ValueError("Asset missing 'id'")
      if not asset.get("path"):
        raise ValueError(f"Asset '{asset.get('id')}' missing 'path'")
      platform_id = asset.get("platform_id")
      if not platform_id:
        raise ValueError(f"Asset '{asset.get('id')}' missing 'platform_id'")
      if not _PLATFORM_SLUG_RE.match(str(platform_id)):
        raise ValueError(
            f"Asset '{asset.get('id')}' has invalid platform_id "
            f"'{platform_id}' (must be a lowercase slug: [a-z0-9][a-z0-9_-]*)"
        )
      asset_id = asset.get("id")
      import os
      platform_basename = os.path.basename(platform_id)
      prefix = f"{platform_basename}_"
      if not asset_id.startswith(prefix):
        raise ValueError(f"Asset ID '{asset_id}' must start with platform prefix '{prefix}'")
      path = asset.get("path")
      _, filename = os.path.split(path)
      if not filename.startswith(prefix):
        raise ValueError(f"Asset path filename '{filename}' must start with platform prefix '{prefix}'")


def _validate_expanded_phase_assets(assets_payload: dict[str, Any]) -> None:
  """Validate the 'assets' half of a phase expansion (Tier 3.1 / design 6.1).

  The assets half is generated and retried independently from the timeline. A
  missing 'assets' key or a non-list is an error so the caller's retry loop can
  react; the caller degrades to ``assets:[]`` only after exhausting its retry
  budget. Each asset must carry id/path/platform_id and obey the platform-prefix
  rules enforced downstream.
  """
  if "assets" not in assets_payload:
    raise ValueError("Phase assets response missing 'assets'")
  assets = assets_payload["assets"]
  if not isinstance(assets, list):
    raise ValueError("Assets must be a list")
  import os
  for asset in assets:
    if not isinstance(asset, dict):
      raise ValueError("Asset spec must be a dictionary")
    if not asset.get("id"):
      raise ValueError("Asset missing 'id'")
    if not asset.get("path"):
      raise ValueError(f"Asset '{asset.get('id')}' missing 'path'")
    platform_id = asset.get("platform_id")
    if not platform_id:
      raise ValueError(f"Asset '{asset.get('id')}' missing 'platform_id'")
    if not _PLATFORM_SLUG_RE.match(str(platform_id)):
      raise ValueError(
          f"Asset '{asset.get('id')}' has invalid platform_id "
          f"'{platform_id}' (must be a lowercase slug: [a-z0-9][a-z0-9_-]*)"
      )
    asset_id = asset.get("id")
    platform_basename = os.path.basename(platform_id)
    prefix = f"{platform_basename}_"
    if not asset_id.startswith(prefix):
      raise ValueError(f"Asset ID '{asset_id}' must start with platform prefix '{prefix}'")
    path = asset.get("path")
    _, filename = os.path.split(path)
    if not filename.startswith(prefix):
      raise ValueError(f"Asset path filename '{filename}' must start with platform prefix '{prefix}'")


def _validate_expanded_phase(expanded_payload: dict[str, Any], expected_round_ids: list[str]) -> None:
  if "timeline" not in expanded_payload:
    raise ValueError("Phase expansion response missing 'timeline'")
  timeline = expanded_payload["timeline"]
  if not isinstance(timeline, list) or not timeline:
    raise ValueError("Timeline must be a non-empty list")
  generated_round_ids = {r.get("round_id") for r in timeline if isinstance(r, dict)}
  for r_id in expected_round_ids:
    if r_id not in generated_round_ids:
      raise ValueError(f"Phase expansion failed to generate details for expected round '{r_id}'")
  if "assets" in expanded_payload:
    _validate_expanded_phase_assets(expanded_payload)


# A bracketed segment that leaves content outside it, a comma that is not a
# thousands separator, or six-plus words. Deliberately NOT a length test: see
# ``_is_compound_decisive_value``.
_COMPOUND_BRACKET_RE = re.compile(r"[(\[][^)\]]+[)\]]")
_COMPOUND_CLAUSE_COMMA_RE = re.compile(r",(?!\d{3}(\D|$))")
_COMPOUND_MIN_WORDS = 6


def _is_compound_decisive_value(value: str) -> bool:
  """Whether a decisive value carries anything ALONGSIDE its reading.

  This is the guard on the ``decisive_value_citable`` DECLARATION, not on the
  value. A compound value is a perfectly good value -- it just cannot be graded
  by an exact substring match, because two equally correct reports punctuate it
  differently.

  Why the guard exists at all. S6 replaced a shape router with the model's own
  declaration, on the grounds that "a default would just be the deleted shape
  router again with a constant in place of the regex". The shipped corpus says
  the declaration alone does not hold: 178 of 400 exact-match values (44.5%)
  were compound phrases like ``LOCKED_EXCLUSIVE (Holder: Avery_Fulfillment_
  Coordinator, TTL: 120 s)``, whose round instruction asks only that the agent
  "record the state shown on the panel". Graded as a raw case-insensitive
  substring (``effect_checker`` tokenizes only ``messages_sent``, never
  ``files_created``), a correct report that writes "held by Avery, TTL 120s"
  fails. Measured on 8,337 run observations the exact arm passes 66% at 0-15
  characters, 25.5% at 21-34, 4.5% at 35-44 and 0.0% at 45+; across 437
  observations of compound needles it passed 0 times.

  The C2 calibration pass cannot catch this: it replays the GOLD, and the gold
  is prompt-steered to print the value verbatim, so the check always looks
  satisfiable (1 re-derive and 0 degrades across the whole last corpus).

  Downgrading costs no difficulty. Separating the arms over 18,194 run
  observations: atomic values matched exactly passed 66.1%, compound values
  graded semantically passed 63.1%. The downgrade moves a round from a check
  that never passes to one that is marginally HARDER than the exact arm.

  LENGTH IS DELIBERATELY NOT A CRITERION. It is the false-negative class S6 was
  right to remove: ``https://telemetry-alpha.cluster.internal/v1/sync`` (48
  chars), ``SHA256: f2c9a18d04b3e7125a89601d2e4f5139`` (40) and ``1,420
  queries/min`` are long AND atomic, and a length rule re-rejects exactly the
  readings the shape router used to lose.

  The ``SHA256: `` example is deliberately kept here even though the S1 prompt
  now steers AWAY from emitting a caption inside the value. The two are not in
  conflict and must not be reconciled by tightening this predicate: the prompt
  DISCOURAGES the shape at the source, while this guard stays LENIENT about it,
  because a caption is not evidence that the reading is compound and flagging
  it would downgrade a genuinely atomic digest. If the prompt fails to prevent
  the shape, the answer is prompt evidence from the next corpus, not a stricter
  rule here.

  Calibration: this predicate flags 178 of the 400 exact-arm values on the
  current corpus and 0 of the 127 on the previous one -- that older exact arm
  was routed by the shape regex and is a clean false-positive control.
  """
  text = str(value or "").strip()
  if not text:
    return False
  # Bracketed qualifier, but only when the reading survives outside it: a value
  # that is wholly bracketed is still one token.
  if _COMPOUND_BRACKET_RE.search(text) and _COMPOUND_BRACKET_RE.sub("", text).strip():
    return True
  if _COMPOUND_CLAUSE_COMMA_RE.search(text):
    return True
  return len(text.split()) >= _COMPOUND_MIN_WORDS


def _validate_expanded_phase_decisive(
    payload: dict[str, Any],
    expected_round_ids: list[str],
    already_decided: set[str] | frozenset[str] | None = None,
    workspace_texts: dict[str, str] | None = None,
    session_texts: list[str] | None = None,
) -> list[tuple[str, str]]:
  """Validate the S1 half of phase expansion: one decisive point per round.

  S1 runs between the timeline half and the assets half, so the value each
  round turns on is chosen from the FINALIZED round outlines while the phase's
  asset text does not yet exist. The asset steps are then told which value they
  must carry verbatim, instead of a later resolver having to hunt for one in
  text that is already frozen.

  Only STRUCTURAL facts are checked here -- presence, per-round coverage,
  single line, length. The SHAPE of the value is the layer contract's
  responsibility and NO value is ever rejected for its shape: a shape regex was
  measured to accept just 127 of 523 corpus values, rejecting every reading with
  a unit ("182.1 U/L"), a thousands separator ("1,420 queries/min") or a
  bracketed state ("SEC-CONTAIN-8842 [VERIFIED]").

  The one place shape IS read is ``decisive_value_citable``, and it is read as a
  guard on the DECLARATION, never on the value. When the model declares a
  compound phrase citable, the declaration is downgraded to false in place and
  the round is graded semantically; the value itself is passed through
  untouched, and the phase is NOT retried. Retrying would be actively harmful:
  a raise here re-runs the whole ``generate_layer`` call, so every OTHER round
  in the phase re-rolls its value too, and one wrong boolean would churn a whole
  phase's worth of species. Downgrades are RETURNED so the caller can log them
  and the trigger rate stays countable. See ``_is_compound_decisive_value`` for
  why the declaration cannot be trusted unguarded and why the downgrade costs no
  difficulty.

  Distinctness IS enforced, because reuse is measurable and it is free score:
  213 of 523 corpus rounds (40.7%) turned on a value another round in the same
  scenario had already used, and 8 scenarios ran all 10 of their rounds on a
  single value. Holding findability constant, those rounds scored 0.6950
  against 0.6601 -- once the player has the answer, every later round that
  reuses it is a giveaway. ``already_decided`` carries the casefolded values
  earlier phases already spent, since phases are expanded one at a time.

  COLLISION with already-written text is enforced for the same reason, and it is
  the reason S1 is worth validating at all rather than leaving it to the
  downstream confidentiality gate. ``workspace_texts``/``session_texts`` are the
  base workspace files and chat logs Stage 3 wrote BEFORE this call; a value that
  already appears in them is text-solvable the moment it is chosen -- the player
  greps the workspace instead of reading the evidence. The downstream gate does
  catch the same collision, but only after the value is bound to the round, and
  the resolver's cheapest escape from a bound value is to re-label the locus
  ``computed``, which the gate then skips: 43 of 43 ``computed`` picks on the
  last corpus appeared only after a leak rejection and none was a first choice,
  laundering 56 rounds into a text-solvable state the exclusivity report was
  not even scanning. Rejecting HERE feeds the S1 retry loop, where the value is
  still soft and can actually be changed.

  Matching is fold-insensitive via ``_fold_for_grounding`` -- deliberately the
  SAME comparison the downstream gate uses, so a value that satisfies this
  check cannot then fail that one on a whitespace or case difference and end up
  laundered anyway. Trivially short values (<3 non-space characters) are skipped
  for the same reason the downstream gate skips them: they match ubiquitous
  substrings.

  This check constrains WHICH value, never WHAT KIND. There is no species
  filter here and must not be one -- see the note above the S1 prompt for the
  measured cost of the last one.
  """
  if "decisive_points" not in payload:
    raise ValueError("Phase decisive response missing 'decisive_points'")
  points = payload["decisive_points"]
  if not isinstance(points, list) or not points:
    raise ValueError("'decisive_points' must be a non-empty list")

  by_round: dict[str, dict[str, Any]] = {}
  seen_values: dict[str, str] = {}
  downgraded: list[tuple[str, str]] = []
  decided_before = {str(v).strip().casefold() for v in (already_decided or set())}
  for point in points:
    if not isinstance(point, dict):
      raise ValueError(f"Each decisive point must be an object, got {type(point).__name__}")
    round_id = str(point.get("round_id") or "").strip()
    if not round_id:
      raise ValueError("A decisive point is missing 'round_id'")

    raw_value = point.get("decisive_value")
    value = str(raw_value).strip() if raw_value is not None else ""
    if not value:
      raise ValueError(f"Round '{round_id}' decisive point has an empty 'decisive_value'")
    if "\n" in value or "\r" in value:
      raise ValueError(
          f"Round '{round_id}' decisive value spans multiple lines; it must be a"
          " single token that can be stamped onto a caption verbatim:"
          f" {value!r}"
      )
    if len(value) > _DECISIVE_POINT_MAX_LEN:
      raise ValueError(
          f"Round '{round_id}' decisive value is {len(value)} characters, over the"
          f" {_DECISIVE_POINT_MAX_LEN}-character limit. Emit the bare reading the"
          " decision rests on, not the panel title or the sentence around it:"
          f" {value!r}"
      )

    # Collision with text the background half already shipped. The repair the
    # message asks for is deliberately "make it more specific", NOT "pick a
    # different kind of value": the cheapest way to dodge a collision is to
    # invent an ID, and a phase that does that for every clash has traded its
    # status-phrase species away to satisfy a uniqueness check.
    if len(value.replace(" ", "")) >= 3:
      folded_value = _fold_for_grounding(value)
      for path, text in (workspace_texts or {}).items():
        if folded_value in _fold_for_grounding(text):
          raise ValueError(
              f"Round '{round_id}' decisive value {value!r} is ALREADY WRITTEN in"
              f" the existing workspace file '{path}'. The player can read that"
              " file without opening any evidence, so the round would grade"
              " nothing. Choose a value that does not yet appear anywhere in"
              " 'background' -- keep the same KIND of value and make it more"
              " specific (a different reading, a different state, a narrower"
              " phrasing); do not switch to an invented identifier just to avoid"
              " the clash."
          )
      for text in (session_texts or []):
        if folded_value in _fold_for_grounding(text):
          raise ValueError(
              f"Round '{round_id}' decisive value {value!r} is ALREADY WRITTEN in"
              " an existing chat/session message. The player can find it by"
              " searching the chat log without opening any evidence. Choose a"
              " value that does not yet appear anywhere in 'background' -- keep"
              " the same KIND of value and make it more specific (a different"
              " reading, a different state, a narrower phrasing); do not switch"
              " to an invented identifier just to avoid the clash."
          )

    label = str(point.get("decisive_label") or "").strip()
    if not label:
      raise ValueError(
          f"Round '{round_id}' decisive point is missing 'decisive_label' (a short"
          " phrase naming what the value is and where it is read)"
      )

    # S6: the step that chose the value also declares how it must be graded.
    # Required, never defaulted -- a default would just be the deleted shape
    # router again with a constant in place of the regex. Normalized in place
    # to a real bool so every consumer can read it with ``is False``.
    raw_citable = point.get("decisive_value_citable")
    if isinstance(raw_citable, bool):
      citable = raw_citable
    elif (
        isinstance(raw_citable, str)
        and raw_citable.strip().lower() in ("true", "false")
    ):
      # Tolerating a quoted boolean is parsing, not judgement: a model that
      # spelled its boolean as a string has still answered the question, and
      # failing the whole phase over the quoting would degrade S1 for a typo.
      citable = raw_citable.strip().lower() == "true"
    else:
      raise ValueError(
          f"Round '{round_id}' decisive point is missing a boolean"
          " 'decisive_value_citable'. Declare true when a correct report cannot"
          " state the value except by printing those exact characters (a"
          " measurement with its unit, a code, a timestamp), false when its"
          f" meaning survives rewording. Got {raw_citable!r}"
      )
    # Guard the DECLARATION, never the value. A compound reading is a fine
    # value and is passed through byte-for-byte; it just cannot be graded by a
    # raw substring match, so it is routed to the semantic arm instead. No
    # raise: see the note in this function's docstring on why retrying here
    # would re-roll every other round in the phase.
    if citable and _is_compound_decisive_value(value):
      citable = False
      downgraded.append((round_id, value))
    point["decisive_value_citable"] = citable
    # Distinctness is compared casefolded and stripped so "SEC892" and
    # " sec892 " cannot pose as two different answers. The value is still
    # carried onto the asset exactly as it was written.
    key = value.casefold()
    if key in seen_values:
      raise ValueError(
          f"Round '{round_id}' reuses the decisive value already assigned to"
          f" round '{seen_values[key]}' ({value!r}). Every round must turn on a"
          " DIFFERENT value -- a repeat hands the player an answer they already"
          " hold, so that round grades nothing."
      )
    if key in decided_before:
      raise ValueError(
          f"Round '{round_id}' reuses {value!r}, which an earlier phase of this"
          " scenario already used as its decisive value. Choose a value this"
          " scenario has not resolved yet."
      )
    seen_values[key] = round_id
    by_round[round_id] = point

  for round_id in expected_round_ids:
    if round_id not in by_round:
      raise ValueError(
          f"No decisive point was decided for expected round '{round_id}'"
      )
  return downgraded


def _neutral_asset_phrase(asset_id: str) -> str:
  """Infer a neutral, id-free noun phrase for a leaked asset reference."""
  low = asset_id.lower()
  if any(k in low for k in (
      ".png", ".jpg", ".jpeg", ".gif", ".webp", "screenshot", "chart",
      "graph", "image", "dashboard", "snapshot", "diagram", "sketch",
      "schematic", "whiteboard", "plot")):
    return "the attached screenshot"
  if any(k in low for k in (
      ".pdf", ".doc", ".docx", "document", "report", "invoice", "receipt")):
    return "the attached document"
  if any(k in low for k in (
      ".csv", ".xls", ".xlsx", ".tsv", "spreadsheet", "sheet")):
    return "the attached spreadsheet"
  if any(k in low for k in (".log", "logfile", "trace")):
    return "the attached log"
  return "the attached evidence"


def _append_report_directive(instruction: str) -> str:
  """Append a neutral, path-free, asset-free directive to write the report.

  Deterministic fallback for ``_reconcile_instruction_with_checks`` when the LLM
  rewrite is unavailable, so a round graded on a produced report always tells the
  player to produce one. The base instruction has already passed narrative
  validation (leak-free), and the appended sentence names no path, filename,
  asset, or future round, so the result stays leak-free by construction.
  """
  directive = (
      " Finally, record your findings and your final decision in a written"
      " report file saved to the workspace (you may choose the filename); this"
      " written report is the graded deliverable for this round."
  )
  return f"{instruction.rstrip()}{directive}"


def _sanitize_leaked_asset_ids(
    instruction: str, visible_asset_ids: list[str]
) -> str:
  """Rewrite raw asset IDs the model leaked into a round instruction.

  Agents never see internal asset IDs (they view assets rendered in the
  platform UI), so an instruction naming one is malformed regardless of seed.
  Rather than hard-failing the round -- which wastes retries and can drop the
  whole task -- rewrite each leaked ID to a neutral, id-free noun phrase. This
  keeps the instruction coherent and fair while removing the leak. Longer IDs
  are replaced first so a shorter ID that is a substring of a longer one does
  not corrupt the replacement.
  """
  sanitized = instruction
  for asset_id in sorted({a for a in visible_asset_ids if a}, key=len, reverse=True):
    pattern = re.compile(re.escape(asset_id), re.IGNORECASE)
    if pattern.search(sanitized):
      sanitized = pattern.sub(_neutral_asset_phrase(asset_id), sanitized)
  # Collapse artifacts introduced by substitution (doubled articles/spaces).
  sanitized = re.sub(
      r"\b(?:the|a|an)\s+(the attached)\b", r"\1", sanitized, flags=re.IGNORECASE
  )
  sanitized = re.sub(r"[ \t]{2,}", " ", sanitized)
  return sanitized.strip()


def _validate_generated_round_narrative(
    round_payload: dict[str, Any],
    round_id: str,
    visible_asset_ids: list[str],
) -> None:
  if "rounds" not in round_payload or not round_payload["rounds"]:
    raise ValueError("Response missing 'rounds' or it is empty")
  round_data = round_payload["rounds"][0]
  instruction = round_data.get("instruction", "")
  if not instruction:
    raise ValueError("Round instruction is empty")
  # Leaked asset IDs are auto-sanitized in place (not hard-rejected) to reduce
  # wasted retries; the loop below is a safety net that still fails loudly if
  # anything survives sanitization, so a leak can never reach the final task.
  sanitized_instruction = _sanitize_leaked_asset_ids(instruction, visible_asset_ids)
  if sanitized_instruction != instruction:
    round_data["instruction"] = sanitized_instruction
    instruction = sanitized_instruction
  if not instruction:
    raise ValueError("Round instruction is empty after asset-ID sanitization")
  instruction_lower = instruction.lower()
  for asset_id in visible_asset_ids:
    if asset_id and asset_id.lower() in instruction_lower:
      raise ValueError(f"Instruction leaks visible asset ID: '{asset_id}'")
  round_idx = int(round_id.lstrip("r"))
  if _visible_text_references_future_round(instruction, round_index=round_idx):
    raise ValueError("Instruction leaks future round references")

  mock_subagents = round_data.get("mock_subagents", [])
  for idx, subagent in enumerate(mock_subagents):
    if not isinstance(subagent, dict):
      raise ValueError(f"Round {round_id} Mock subagent at index {idx} must be a dictionary")
    if not subagent.get("agent_role"):
      raise ValueError(f"Round {round_id} Mock subagent at index {idx} is missing 'agent_role'")
    if not subagent.get("mock_response"):
      raise ValueError(f"Round {round_id} Mock subagent '{subagent.get('agent_role')}' is missing 'mock_response'")

  if "updates" in round_payload:
    updates = round_payload["updates"]
    if not isinstance(updates, list):
      raise ValueError("Updates must be a list")
    for idx, update in enumerate(updates):
      if not isinstance(update, dict):
        raise ValueError("Update must be a dictionary")
      workspace_adds = update.get("workspace_adds", [])
      _validate_workspace_paths(workspace_adds, f"Round {round_id} Update #{idx} workspace_adds")

  if "assets" in round_payload:
    assets = round_payload["assets"]
    if not isinstance(assets, list):
      raise ValueError("Assets must be a list")
    for asset in assets:
      if not isinstance(asset, dict):
        raise ValueError("Asset spec must be a dictionary")
      if not asset.get("id"):
        raise ValueError("Asset missing 'id'")
      if not asset.get("path"):
        raise ValueError(f"Asset '{asset.get('id')}' missing 'path'")
      platform_id = asset.get("platform_id")
      if not platform_id:
        raise ValueError(f"Asset '{asset.get('id')}' missing 'platform_id'")
      if not _PLATFORM_SLUG_RE.match(str(platform_id)):
        raise ValueError(
            f"Asset '{asset.get('id')}' has invalid platform_id "
            f"'{platform_id}' (must be a lowercase slug: [a-z0-9][a-z0-9_-]*)"
        )
      asset_id = asset.get("id")
      import os
      platform_basename = os.path.basename(platform_id)
      prefix = f"{platform_basename}_"
      if not asset_id.startswith(prefix):
        raise ValueError(f"Asset ID '{asset_id}' must start with platform prefix '{prefix}'")
      path = asset.get("path")
      _, filename = os.path.split(path)
      if not filename.startswith(prefix):
        raise ValueError(f"Asset path filename '{filename}' must start with platform prefix '{prefix}'")



def _validate_delegation_check(check: dict[str, Any], context_desc: str) -> None:
  if check.get("type") == "effect" and check.get("effect_type") == "messages_sent":
    expected = check.get("expected", {})
    if expected.get("field") == "message":
      if check.get("condition") == "field_semantic_match":
        if not expected.get("criterion"):
          raise ValueError(f"[{context_desc}] Delegation check with field_semantic_match must contain a non-empty 'criterion' field")
      else:
        val = str(expected.get("value", ""))
        for word in ["your", "their", "my", "our", "the", "a", "an"]:
          if f" {word} " in f" {val.lower()} ":
            raise ValueError(f"[{context_desc}] Delegation check value contains forbidden word: '{word}' in '{val}'")


# ---------------------------------------------------------------------------
# Tier 1.1: round_steps split into commands (LLM) + deterministic checker
# scaffold (code) + minimal semantic fill (LLM). See layer_split_design.md 4.1.
# ---------------------------------------------------------------------------

# Commands whose success is verified by a semantic ``messages_sent`` checker
# (a delegation / outgoing message). The criterion is the only LLM-authored
# field; everything else is fixed by code.
_DELEGATION_COMMAND_NAMES = frozenset({
    "agent.delegate",
    "message.send",
    "message.broadcast",
    "message.poll",
})
# Commands that write a file whose success is verified by a ``file_check``.
_FILE_WRITE_COMMAND_NAMES = frozenset({"file.create", "file.append"})
# Commands that read a file. When the round has a known decisive value, a read
# of a real workspace file becomes a deterministic ``file_check`` asserting that
# value is present -- turning an otherwise-unchecked verification/re-read step
# into a real, graded step. Without a decisive value it stays 'plain'.
_FILE_READ_COMMAND_NAMES = frozenset({"file.read"})
# Commands that access a visual asset -> fully deterministic ``assets_accessed``.
_ASSET_ACCESS_COMMAND_NAMES = frozenset({
    "asset.open",
    "asset.info",
    "asset.read_text",
    "asset.describe",
})
# Per-delegation-command flag whose value seeds the semantic criterion fallback.
_DELEGATION_CRITERION_FLAGS = {
    "agent.delegate": "--instruction",
    "message.send": "--message",
    "message.broadcast": "--message",
    "message.poll": "--question",
}
# Source/script file extensions. A decisive *business* value (a token, a number,
# a status phrase) does not belong textually inside an executable script, so we
# never assert ``file_check contains <decisive_value>`` against these: a read of
# a script stays 'plain' and a write degrades to a ``command_check`` (grade that
# the action happened, not that the answer text is embedded in the code). Data /
# config / report files (.md/.txt/.json/.yaml/.conf/.log/.csv/...) are unaffected
# and keep their content checks.
_CODE_FILE_EXTENSIONS = frozenset({
    ".py", ".pyw", ".sh", ".bash", ".zsh", ".ksh", ".js", ".mjs", ".cjs",
    ".ts", ".tsx", ".jsx", ".rb", ".go", ".rs", ".java", ".kt", ".kts",
    ".scala", ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".cs",
    ".php", ".pl", ".pm", ".lua", ".swift", ".m", ".mm", ".dart",
})
# Effect-check ``effect_type`` values the generated-round validators accept. Kept
# as one shared constant so the step validator and the compiled-trajectory
# validator can never drift apart. New protocol families extend this set as they
# gain graded effects.
ALLOWED_EFFECT_TYPES: frozenset[str] = frozenset({
    "messages_sent",
    "assets_accessed",
    "files_created",
    "lock_acquired",
    "lock_released",
    "deployment_executed",
    "deployment_cutover",
    "deployment_rolledback",
    "booking_held",
    "booking_confirmed",
    "booking_released",
    "pairing_completed",
    "approval_granted",
    "approval_escalated",
    "containment_action",
    "review_submitted",
    "document_signed",
    # Prerequisite steps of the multi-step protocols. These are real state
    # transitions (they are what each terminal's ``EffectPrecondition`` chain
    # verifies), but they used to emit nothing, so every round of a protocol
    # was invisible to the checker except the last one -- measured: gold
    # touched the declared tool family in 92.3% of rounds but performed a
    # gradable protocol step in only 29.7%. Grading the intermediate steps is
    # what the mechanic plugins' own evals hooks already ask for ("following an
    # approval request and an approval approve from every required party").
    "approval_requested",
    "approval_recorded",
    "approval_revoked",
    "deployment_requested",
    "deployment_approved",
    "deployment_gate_set",
    "pairing_started",
    "pairing_code_shown",
    "pairing_code_entered",
    "review_assigned",
    "booking_extended",
})
# Registry mapping a command *name* (e.g. ``lock.acquire``) to the effect type
# its successful execution produces. The scaffolder branch and classifier hook
# below turn each entry into a ``{"type": "effect", ...}`` checker asserting the
# effect was emitted. Populated in M1 with the lock/deploy protocol families.
_EFFECT_COMMAND_NAMES: dict[str, str] = {
    "lock.acquire": "lock_acquired",
    "lock.release": "lock_released",
    "deploy.execute": "deployment_executed",
    "deploy.cutover": "deployment_cutover",
    "deploy.rollback": "deployment_rolledback",
    "booking.hold": "booking_held",
    "booking.confirm": "booking_confirmed",
    "booking.release": "booking_released",
    "pairing.confirm": "pairing_completed",
    "approval.grant": "approval_granted",
    "approval.escalate": "approval_escalated",
    "containment.revoke-sessions": "containment_action",
    "containment.disable-forwarding": "containment_action",
    "containment.isolate-device": "containment_action",
    "containment.block-token": "containment_action",
    "review.submit": "review_submitted",
    "review.sign": "document_signed",
    # Prerequisite steps (see the note in ALLOWED_EFFECT_TYPES). Registering
    # them here is what lets ``_scaffold_round_checkers`` grade a round whose
    # gold advances the protocol without completing it. Terminal detection is
    # unaffected: ``_mechanic_command_sets`` builds ``terminals`` by
    # intersecting the *plugin's* declared effects with the command's, and no
    # plugin declares these, so a prerequisite lands in ``protocol`` only and
    # can never satisfy the complete-the-protocol gate.
    "approval.request": "approval_requested",
    "approval.approve": "approval_recorded",
    "approval.provisional": "approval_recorded",
    # NOT registered: ``approval.revoke``. It is the one command in the skill
    # set that can exit 0 and emit NOTHING -- revoking a resource that was never
    # granted is a successful no-op, and the skill deliberately logs state
    # changes rather than attempts. Scaffolding a check off it would therefore
    # produce an UNWINNABLE round whenever a gold trajectory revokes something
    # it never granted. It buys nothing to offset that: the advance gate is
    # already satisfiable by five other approval verbs, and no plugin grades
    # ``approval_revoked``. The effect type stays in ALLOWED_EFFECT_TYPES so the
    # emission itself remains legal and gradable if a plugin ever asks for it.
    "deploy.request": "deployment_requested",
    "deploy.approve": "deployment_approved",
    "deploy.gate-set": "deployment_gate_set",
    "pairing.start": "pairing_started",
    "pairing.show-code": "pairing_code_shown",
    "pairing.enter-code": "pairing_code_entered",
    "review.assign": "review_assigned",
    "booking.extend": "booking_extended",
}


def _normalize_effect_command_key(name: str) -> str:
  """Fold a dotted command name so hyphen/underscore spellings compare equal.

  The two SSOTs disagree, and only for some verbs: a skill's ``CommandSpec``
  gives ``deploy gate-set`` -> ``deploy.gate-set``, while the AgentToolSpec that
  ``_classify_step_command`` resolves against reports ``deploy.gate_set`` --
  yet the very same AgentToolSpec keeps the hyphen for
  ``containment.block-token``. Three commands (``deploy gate-set``,
  ``pairing show-code``, ``pairing enter-code``) fell in that gap, so their
  registry entries were unreachable from the scaffold: the gate could require
  the command while the checker scaffold silently produced nothing to grade it
  with. Folding both spellings removes the class of bug rather than the three
  instances; ``test_protocol_effect_scaffold`` pins that every registered key
  stays reachable.
  """
  return str(name or "").strip().lower().replace("-", "_")


@functools.lru_cache(maxsize=1)
def _effect_command_names_normalized() -> dict[str, str]:
  """``_EFFECT_COMMAND_NAMES`` keyed by normalized command name."""
  return {
      _normalize_effect_command_key(k): v
      for k, v in _EFFECT_COMMAND_NAMES.items()
  }


def _effect_type_for_command_name(name: str) -> str | None:
  """The effect a command emits, tolerant of hyphen/underscore spelling."""
  return _effect_command_names_normalized().get(
      _normalize_effect_command_key(name)
  )


@functools.lru_cache(maxsize=1)
def _skill_command_specs() -> dict[str, Any]:
  """Map dotted command name (``booking.confirm``) -> skill ``CommandSpec``.

  Read once from the skill SSOT (each concrete skill's ``SPEC``). Dotted names
  match ``_EFFECT_COMMAND_NAMES`` and ``AgentCommandSpec.name``; skill specs
  store *spaced* names (``booking confirm``), so spaces normalize to dots. Lazy
  import keeps this free of an import cycle with the skills package.
  """
  from clasharness.skills import impl as _impl  # pylint: disable=g-import-not-at-top

  out: dict[str, Any] = {}
  for attr in getattr(_impl, "__all__", []):
    cls = getattr(_impl, attr, None)
    spec = getattr(cls, "SPEC", None)
    if spec is None:
      continue
    for cmd in spec.commands:
      out[cmd.name.replace(" ", ".")] = cmd
  return out


def command_spec_for_effect(effect_type: str) -> Any | None:
  """Return the skill ``CommandSpec`` whose success emits ``effect_type``.

  Resolves ``effect_type`` -> dotted command name (reverse of
  ``_EFFECT_COMMAND_NAMES``) -> skill ``CommandSpec`` (the skill SSOT). Returns
  ``None`` when the effect is not a registered protocol effect or its command is
  unknown. This is the lookup the deterministic prerequisite resolver uses to
  recover a failing effect's precondition contract (``CommandSpec.requires``).
  """
  # Several commands may emit the same effect (the four ``containment`` verbs;
  # ``approval approve`` and ``approval provisional``). Registry order is the
  # canonical order, so keep the FIRST entry rather than letting the last one
  # silently win -- otherwise the resolver's "how to satisfy this" hint names
  # an incidental alias (``approval provisional``) instead of the canonical
  # command (``approval approve``).
  by_effect: dict[str, str] = {}
  for name, eff in _EFFECT_COMMAND_NAMES.items():
    by_effect.setdefault(eff, name)
  dotted = by_effect.get(effect_type)
  if dotted is None:
    return None
  return _skill_command_specs().get(dotted)


def validate_effect_command_registry() -> list[str]:
  """Conformance: the effect registry and the skill precondition SSOT agree.

  Protects the deterministic prerequisite resolver's lookup path
  (``effect_type`` -> command -> ``CommandSpec.requires``) from silent drift.
  Returns a list of violation strings (empty == OK) so a test can assert on it.

  Checks:

  1. every ``_EFFECT_COMMAND_NAMES`` entry resolves to a real skill command that
     declares the mapped effect in ``emits_effects``, and that effect is in
     ``ALLOWED_EFFECT_TYPES``;
  2. every skill command that declares ``requires`` (a multi-step protocol
     terminal) also emits an effect that is registered in
     ``_EFFECT_COMMAND_NAMES`` pointing back to that command -- otherwise the
     resolver could never recover its precondition contract from a failing
     effect check (the exact gap that produced kept-with-failures rounds).
  """
  errors: list[str] = []
  specs = _skill_command_specs()

  # (1) registry -> skill-spec integrity.
  for dotted, effect in _EFFECT_COMMAND_NAMES.items():
    cmd = specs.get(dotted)
    if cmd is None:
      errors.append(f"_EFFECT_COMMAND_NAMES[{dotted!r}]: no such skill command")
      continue
    if effect not in ALLOWED_EFFECT_TYPES:
      errors.append(f"{dotted}: effect {effect!r} not in ALLOWED_EFFECT_TYPES")
    if effect not in cmd.emits_effects:
      errors.append(
          f"{dotted}: registry maps to {effect!r} but the command emits "
          f"{cmd.emits_effects}"
      )

  # (2) skill-requires -> registry integrity.
  for dotted, cmd in specs.items():
    if not cmd.requires:
      continue
    if not cmd.emits_effects:
      errors.append(f"{dotted}: declares requires but emits no effect")
      continue
    mapped = _EFFECT_COMMAND_NAMES.get(dotted)
    if mapped is None:
      errors.append(
          f"{dotted}: declares requires but is absent from "
          f"_EFFECT_COMMAND_NAMES (resolver cannot recover its precondition "
          f"from a failing effect check)"
      )
    elif mapped not in cmd.emits_effects:
      errors.append(
          f"{dotted}: _EFFECT_COMMAND_NAMES maps to {mapped!r}, not in "
          f"emits_effects {cmd.emits_effects}"
      )
  return errors


def _build_protocol_contract_prompt(allowed_families: list[str]) -> str:
  """Machine-generated protocol-prerequisite contract for the authoring prompt.

  Conflict rounds start from EMPTY protocol state, so a terminal command of a
  multi-step protocol (``booking confirm``, ``pairing confirm``, ``deploy
  execute``/``cutover``) is rejected by its runtime guard unless its declared
  prerequisites ran first, in order, sharing the same identifier. When the gold
  author omits that prefix the terminal emits no effect and the round's
  ``exists`` effect check fails (the leading cause of kept-with-failures rounds).

  This renders the contract straight from the skill SSOT
  (``CommandSpec.requires``) so the guidance can never drift from the runtime
  guards or the deterministic resolver. Only families allowed in this scenario
  are described; returns ``""`` when no allowed command declares a prerequisite
  (so plain scenarios get no extra prompt text).
  """
  specs = _skill_command_specs()
  families_set = {normalize_family_name(f) for f in allowed_families or []}
  lines: list[str] = []
  for dotted in sorted(specs):
    cmd = specs[dotted]
    if not getattr(cmd, "requires", None):
      continue
    family = normalize_family_name(dotted.split(".", 1)[0])
    if families_set and family not in families_set:
      continue
    clauses: list[str] = []
    for pre in cmd.requires:
      if pre.chain:
        chain_txt = " -> ".join(f"`{c}`" for c in pre.chain)
        ident = (
            f" (all sharing the same {pre.identifier_arg} value, within --ttl)"
            if pre.identifier_arg
            else ""
        )
        clauses.append(f"first run {chain_txt}{ident}")
      if pre.all_ready is not None:
        ready_cmd, ready_arg, ready_val = pre.all_ready
        clauses.append(
            f"and every `{ready_cmd}` must be {ready_arg} {ready_val}"
        )
    if not clauses:
      continue
    lines.append(f"- `{cmd.name}`: {'; '.join(clauses)}, THEN `{cmd.name}`.")
  if not lines:
    return ""
  return (
      "\n\nPROTOCOL PREREQUISITES (each round starts from EMPTY protocol state;"
      " these terminal commands are REJECTED — and emit no effect — unless their"
      " prerequisite commands ran first, in order, with a matching identifier."
      " If you use one of these, include its full prerequisite chain in the same"
      " round):\n"
      + "\n".join(lines)
  )


# Fraction of a mechanic scenario's rounds whose gold must actually PERFORM the
# conflict protocol (not merely query it). Measured motivation: across the 10
# corpus seeds that route to a mechanic plugin, gold touched the declared tool
# family in 92.3% of rounds but performed the protocol in only 29.7% -- the rest
# called read-only verbs (`lock status`, `approval status`). Because the checker
# scaffold derives checks from the commands, a round whose gold never performs
# the protocol also never grades it: TERMINAL and CHECK coverage were identical
# in all 10 seeds (89/89 rounds). Rounds that do perform it pass at 30.2% vs
# 53.2% for the rest (-23.0pp; -18.2pp after controlling for gold length), so
# this is the difficulty the design intends and currently does not collect.
#
# Not 100%: a scenario needs investigation rounds too, and forcing the terminal
# verb into every round would flatten the trajectory variety the seeds exist to
# provide. Spread deterministically (see ``_round_requires_mechanic``) rather
# than back-loaded, so protocol rounds do not clump at the end.
MECHANIC_ROUND_RATIO = 2.0 / 3.0


def _round_requires_mechanic(round_idx: int, ratio: float = MECHANIC_ROUND_RATIO) -> bool:
  """Whether round ``round_idx`` (1-based) must perform the conflict protocol.

  Deterministic even spread: round *i* is selected iff ``floor(i*ratio)``
  advances past ``floor((i-1)*ratio)``. For 10 rounds at 2/3 this selects rounds
  2, 3, 5, 6, 8, 9 -- six of ten, none adjacent to both neighbours. Deterministic
  so a regenerated scenario is reproducible and the allocation is unit-testable.
  """
  if round_idx < 1 or ratio <= 0:
    return False
  if ratio >= 1:
    return True
  return int(round_idx * ratio) > int((round_idx - 1) * ratio)


def _final_mechanic_round(
    target_rounds: int, ratio: float = MECHANIC_ROUND_RATIO
) -> int | None:
  """The last selected round, which must COMPLETE (not just advance) the protocol.

  Every scenario has to finish its protocol at least once or the plugin's
  declared terminal effect is never emitted and never graded. Requiring that in
  the *last* selected round rather than the first keeps the completion at the
  end of the trajectory, where the preceding rounds have already established the
  evidence that makes the target choice ambiguous.

  Returns ``None`` when no round is selected (``target_rounds`` too small or
  ratio 0), which makes the caller a no-op.
  """
  for idx in range(int(target_rounds), 0, -1):
    if _round_requires_mechanic(idx, ratio):
      return idx
  return None


@dataclasses.dataclass(frozen=True)
class MechanicCommands:
  """The conflict-protocol command vocabulary a scenario's mechanic exposes.

  ``terminals`` complete the protocol (they emit the effect the plugin declares
  it grades). ``protocol`` is the wider set that also includes the prerequisite
  steps -- ``approval approve``, ``deploy gate-set``, ``pairing enter-code`` --
  which are real state transitions (they are what each terminal's
  ``EffectPrecondition`` verifies) and, since they are registered in
  ``_EFFECT_COMMAND_NAMES``, now scaffold their own effect check.

  The distinction is what lets a round be required to *advance* the protocol
  gradably without forcing it to *complete* it. Completing is required once per
  scenario; a 10-round scenario cannot plausibly grant the same access six
  times, and demanding that would either fail generation or produce absurd
  repetition. Measured on the corpus: 7 of the 10 mechanic seeds have exactly
  one distinct terminal command line across all 30 rounds.
  """

  protocol: tuple[str, ...] = ()
  terminals: tuple[str, ...] = ()
  effects: tuple[str, ...] = ()

  def __bool__(self) -> bool:
    return bool(self.terminals)


def _mechanic_command_sets(
    plugins: list[Any], allowed_families: list[str]
) -> MechanicCommands:
  """Derive the mechanic's command vocabulary for this scenario.

  Derived, never hand-listed: ``ConflictPluginSpec.emits_effects`` says which
  effect keys the plugin asserts and ``CommandSpec.emits_effects`` says which
  commands emit them, so the intersection is the terminal set. The protocol set
  additionally takes every command of the *same families* that emits a
  registered effect. Both are restricted to families the scenario actually
  exposes, so a requirement can never be impossible to satisfy.

  Returns an empty ``MechanicCommands`` when the scenario routes to no mechanic
  plugin (half the seeds), which makes every caller a no-op there.
  """
  declared: set[str] = set()
  for plugin in plugins or []:
    spec = getattr(plugin, "SPEC", None)
    if spec is None:
      continue
    declared |= {str(e).strip() for e in (spec.emits_effects or ()) if str(e).strip()}
  if not declared:
    return MechanicCommands()
  allowed = {normalize_family_name(f) for f in allowed_families or []}
  specs = _skill_command_specs()

  terminals: set[str] = set()
  mechanic_families: set[str] = set()
  for dotted, cmd in specs.items():
    family = normalize_family_name(dotted.split(".", 1)[0])
    if allowed and family not in allowed:
      continue
    emits = {str(e).strip() for e in (getattr(cmd, "emits_effects", None) or ())}
    if declared & emits:
      terminals.add(cmd.name)
      mechanic_families.add(family)
  if not terminals:
    return MechanicCommands()

  protocol: set[str] = set(terminals)
  for dotted, cmd in specs.items():
    family = normalize_family_name(dotted.split(".", 1)[0])
    if family not in mechanic_families:
      continue
    if dotted in _EFFECT_COMMAND_NAMES:
      protocol.add(cmd.name)
  return MechanicCommands(
      protocol=tuple(sorted(protocol)),
      terminals=tuple(sorted(terminals)),
      effects=tuple(sorted(declared)),
  )


def _first_required_flag(command_name: str) -> str | None:
  """The first required flag of a registered command, or ``None``.

  This is the same rule ``_target_identifier_for_command`` uses to decide what
  a command acts on, resolved from the command name rather than a spec object.
  """
  specs = _skill_command_specs()
  key = _normalize_effect_command_key(command_name)
  for dotted, cmd in specs.items():
    name = getattr(cmd, "name", "") or dotted
    if _normalize_effect_command_key(name) != key:
      continue
    required = list(
        getattr(cmd, "required_flags", None)
        or getattr(cmd, "required", None)
        or ()
    )
    if not required:
      return None
    flag = str(required[0] or "").strip()
    return flag if flag.startswith("--") else None
  return None


def _mechanic_target_flags(mech: MechanicCommands) -> tuple[str, ...]:
  """Flags naming a target the solver must CHOOSE (selection semantics).

  SSOT-derived and deliberately restricted to the flags of the mechanic's
  TERMINAL commands. A terminal command's identifier is a thing the solver
  picks -- ``--resource`` for approval/booking/lock, ``--change`` for deploy,
  ``--device`` for pairing, ``--account``/``--device``/``--token`` for
  containment, ``--document`` for review -- so planting decoys next to it is
  safe and is exactly the discrimination we want.

  Flags that appear only on non-terminal steps are excluded on purpose; see
  :func:`_mechanic_set_member_flags` for why conflating the two would ship
  unsatisfiable checks.
  """
  flags: list[str] = []
  for command_name in mech.terminals:
    flag = _first_required_flag(command_name)
    if flag and flag not in flags:
      flags.append(flag)
  return tuple(sorted(flags))


def _mechanic_set_member_flags(mech: MechanicCommands) -> tuple[str, ...]:
  """Flags naming a member of a set the terminal step CONSUMES (set semantics).

  Some protocol steps do not name a thing to choose between; they name one
  member of a collection that the terminal step then requires to be *complete*.
  ``deploy gate-set --gate`` populates the readiness gates that ``deploy
  cutover`` requires ALL of to be ready; ``review submit --aspect`` populates
  the aspects that ``review sign`` requires ALL of to be ok.

  Keeping these apart from the selection targets is a correctness requirement,
  not a nicety. Asking the world for "three candidate gates, exactly one
  correct, the other two decoys" would plant two gates that must stay un-ready
  -- and an un-ready gate blocks ``deploy cutover`` forever, so the round's
  terminal check could never pass no matter what the solver did. The same trap
  exists for a decoy review aspect, which stays pending and blocks the
  signature. Set-member flags therefore get a *breadth* requirement (more
  members, all of which must be satisfied) instead of a decoy requirement.

  Derived structurally: a flag is a set member if some protocol step uses it
  and no terminal command does.
  """
  targets = set(_mechanic_target_flags(mech))
  flags: list[str] = []
  for command_name in mech.protocol:
    if command_name in mech.terminals:
      continue
    flag = _first_required_flag(command_name)
    if flag and flag not in targets and flag not in flags:
      flags.append(flag)
  return tuple(sorted(flags))


# How many candidate identifiers of each target kind the world must contain.
# 3 (one correct + two decoys) rather than 2: the check that grades a protocol
# step binds the target identifier, so a solver that cannot tell the candidates
# apart is guessing, and 1-in-2 guessing is close enough to free that it would
# blunt rather than sharpen the measurement. It is capped low because every
# extra candidate is extra world the generator has to keep self-consistent.
MECHANIC_TARGET_CANDIDATES = 3


def _multi_instance_world_instruction(mech: MechanicCommands) -> str:
  """Require the world to hold several candidate targets, not just one.

  Measured on the v3 corpus: 16 of 30 mechanic scenarios contained exactly ONE
  distinct target identifier across every graded protocol command (one
  ``--resource``, one ``--device``, ...). With a single candidate, binding the
  identifier in the checker grades nothing that ``exists`` did not already
  grade -- the only resource in the world is trivially the right one, so a
  solver that never reads the evidence still lands on it.

  Planting several *plausible* candidates and letting the evidence single one
  out is what turns "run the right verb" into "run the right verb on the thing
  the conflict is actually about", which is the skill under test.

  Fairness is the binding constraint here, so the instruction is explicit that
  the correct target must be UNIQUELY determinable from shipped evidence. The
  difficulty has to come from having to read the world, never from ambiguity:
  an under-determined choice would be a coin flip, and the gold trajectory
  would be arbitrary. Decoys must be plausible (same naming convention) yet
  distinguishable (each ruled out by something concrete and legible).

  Returns ``""`` for the seeds that route to no mechanic plugin, which makes
  this a no-op for them.
  """
  targets = _mechanic_target_flags(mech)
  members = _mechanic_set_member_flags(mech)
  if not targets and not members:
    return ""

  n = MECHANIC_TARGET_CANDIDATES
  blocks: list[str] = []

  if targets:
    flag_list = ", ".join(f"`{f}`" for f in targets)
    blocks.append(
        f"MULTIPLE CANDIDATE TARGETS (required): the decisive protocol commands "
        f"in this scenario name what they act on via {flag_list}. For EVERY one "
        f"of those flags, the world must contain at least {n} distinct, "
        f"plausible candidate identifiers -- not one. Requirements:\n"
        f"  (a) All {n} must look like real peers: same naming convention, same "
        f"kind of thing, all present in the workspace/system state, and all "
        f"individually valid to act on (acting on a decoy must silently "
        f"succeed, never error out).\n"
        f"  (b) EXACTLY ONE is the correct target of the conflict. The other "
        f"{n - 1} are decoys.\n"
        f"  (c) The correct one must be UNIQUELY determined by concrete "
        f"evidence that is actually shipped in the scenario (a chart value, a "
        f"log line, a ticket field, a policy rule, a message from a "
        f"participant). A reader who studies the evidence must be able to rule "
        f"out every decoy for a specific, stateable reason. Never leave the "
        f"choice ambiguous, under-determined, or decidable only by guessing -- "
        f"if two candidates both survive the evidence, the scenario is broken.\n"
        f"  (d) Decoys must be tempting: each should be the answer a solver "
        f"would reach by skimming (the one named first, the one in the title, "
        f"the one the loudest participant asks for), and wrong for a reason "
        f"that only surfaces on a careful read."
    )

  if members:
    member_list = ", ".join(f"`{f}`" for f in members)
    blocks.append(
        f"COMPLETE THE PRECONDITION SET (required): {member_list} name members "
        f"of a set that the decisive action requires to be COMPLETE -- every "
        f"gate ready, every aspect signed off. Give the scenario at least {n} "
        f"such members, each with its own owner and its own reason to be "
        f"outstanding, so finishing takes real coordination rather than one "
        f"call. These are NOT decoys: every member must end up satisfied, and "
        f"the expert trajectory must satisfy all of them before the decisive "
        f"action. Do not plant a member that is meant to stay unsatisfied -- "
        f"the decisive action is blocked while any member is outstanding, so "
        f"that would make the scenario unsolvable."
    )

  return "\n\n".join(blocks)


def _commands_include(
    commands: list[dict[str, Any]], names: tuple[str, ...] | list[str]
) -> bool:
  """True if any authored command line starts with one of ``names``."""
  if not names:
    return True
  lowered = [n.lower() for n in names]
  for entry in commands or []:
    text = str((entry or {}).get("command", "")).strip().lower()
    if any(text.startswith(n) for n in lowered):
      return True
  return False


def _target_identifier_for_command(command: str, spec: Any) -> str | None:
  """The identifier a protocol command acts on, e.g. ``DEV-4471``.

  Derived from the SSOT, not hand-listed: a protocol command's first REQUIRED
  flag is its target throughout the skill set -- ``--resource`` for
  approval/booking/lock, ``--change`` for deploy, ``--device`` for pairing and
  ``containment isolate-device``, ``--gate`` for ``deploy gate-set``,
  ``--document`` for ``review sign``. Verified against all 29 registered effect
  commands. Where a command also declares an ``EffectPrecondition``, the
  precondition's ``identifier_arg`` agrees with this flag, so the two views of
  "what this command is about" stay consistent.

  Returns ``None`` when the command declares no required flag or the value is
  absent/unparsable, which makes the caller fall back to the previous behaviour.

  Accepts either spec flavour: the AgentToolSpec's ``AgentCommandSpec`` (which
  is what the scaffold resolves and calls the field ``required_flags``) or a
  skill ``CommandSpec`` (``required``). They agree on the ordering.
  """
  required = list(
      getattr(spec, "required_flags", None)
      or getattr(spec, "required", None)
      or ()
  )
  if not required:
    return None
  flag = str(required[0] or "").strip()
  if not flag.startswith("--"):
    return None
  value = _extract_flag_value(command, flag)
  value = (value or "").strip()
  # A placeholder is not an identifier; grading against it would be unwinnable.
  if not value or value.startswith("-"):
    return None
  return value


def _extract_flag_value(command: str, flag: str) -> str | None:
  """Return the value of ``--flag value`` or ``--flag=value``, else None."""
  try:
    tokens = shlex.split(command or "")
  except ValueError:
    return None
  for i, tok in enumerate(tokens):
    if tok == flag:
      if i + 1 < len(tokens):
        return tokens[i + 1]
      return None
    if tok.startswith(flag + "="):
      return tok[len(flag) + 1:]
  return None


def _normalize_ws_path(path: str) -> str:
  """Strip a leading slash so paths match the workspace-relative convention."""
  p = (path or "").strip()
  if p.startswith("/"):
    p = p.lstrip("/")
  return p


def _is_code_file(path: str) -> bool:
  """True when ``path`` is a source/script file (see ``_CODE_FILE_EXTENSIONS``).

  Used by the checker scaffolder to avoid asserting a decisive *business* value
  is textually present inside executable code (fragile and semantically wrong):
  such a value belongs in a data / report / config file, not a script.
  """
  p = (path or "").strip().lower()
  dot = p.rfind(".")
  if dot <= p.rfind("/"):  # no extension (or the dot is in a directory name)
    return False
  return p[dot:] in _CODE_FILE_EXTENSIONS


def _classify_step_command(
    command: str, allowed_families: list[str]
) -> tuple[str, Any]:
  """Classify a command into delegation|file_write|asset_access|plain.

  Returns ``(category, spec)`` where ``spec`` is the resolved
  ``AgentCommandSpec`` (or ``None`` if the command could not be resolved). Any
  ``ToolSpecViolation`` degrades safely to ``('plain', None)``.
  """
  try:
    spec = validate_command_against_spec(
        str(command), allowed_families=allowed_families
    )
  except ToolSpecViolation:
    return ("plain", None)
  name = getattr(spec, "name", "") or ""
  if name in _DELEGATION_COMMAND_NAMES:
    return ("delegation", spec)
  if name in _FILE_WRITE_COMMAND_NAMES:
    return ("file_write", spec)
  if name in _FILE_READ_COMMAND_NAMES:
    return ("file_read", spec)
  if name in _ASSET_ACCESS_COMMAND_NAMES:
    return ("asset_access", spec)
  if _effect_type_for_command_name(name) is not None:
    return ("protocol_effect", spec)
  return ("plain", spec)


def _has_delegate_then_wait(
    commands: list[dict[str, Any]], allowed_families: list[str]
) -> bool:
  """Whether the trajectory delegates to a sub-agent and THEN waits for it.

  M8/CL2 (``force_async_wait``, F6) requires the reference trajectory to actually
  go async: an ``agent delegate`` followed later by a ``wait`` that advances the
  virtual clock so the sub-agent's delayed completion can fire (and be consumed).
  The delegate is detected by tool-spec classification (so a synchronous
  ``message send`` does NOT count), and the ``wait`` by prefix (it is a core env
  command, exempt from tool-family gating -- mirrors
  ``conflict_vlm_materializer._inject_background_wait``). Order matters: the wait
  must come after the delegate.
  """
  delegate_idx: int | None = None
  for i, entry in enumerate(commands):
    cmd = str((entry or {}).get("command", "")).strip()
    _, spec = _classify_step_command(cmd, allowed_families)
    if getattr(spec, "name", "") == "agent.delegate":
      delegate_idx = i
      break
  if delegate_idx is None:
    return False
  for entry in commands[delegate_idx + 1 :]:
    cmd = str((entry or {}).get("command", "")).strip().lower()
    if cmd == "wait" or cmd.startswith("wait "):
      return True
  return False


def _coerce_evidence_anchor(raw: str) -> str:
  """Normalize an anchor string to the canonical ``EVIDENCE:TOKEN`` form.

  Returns ``""`` if nothing usable remains. Accepts either a bare token or an
  already-prefixed ``EVIDENCE:...`` value; non-alphanumeric runs collapse to a
  single underscore and the token is upper-cased so anchors are stable and
  matchable across the pipeline.
  """
  s = str(raw or "").strip()
  if not s:
    return ""
  body = s.split(":", 1)[1] if s.upper().startswith("EVIDENCE:") else s
  token = re.sub(r"[^A-Za-z0-9]+", "_", body).strip("_").upper()
  return f"EVIDENCE:{token}" if token else ""


def _round_command_signature(commands: list[str]) -> str:
  """A normalized signature of a round's solution trajectory for dedup.

  Whitespace is collapsed, case is folded, and the always-present ``DONE``
  terminator is dropped so two rounds with the same substantive command
  sequence collide even under trivial formatting differences.
  """
  norm: list[str] = []
  for c in commands:
    s = re.sub(r"\s+", " ", str(c or "").strip().lower())
    if s and s != "done":
      norm.append(s)
  return " | ".join(norm)


def _parse_resolved_locus(raw: str | None) -> tuple[str, str]:
  """Parse a resolution ``resolved_value_locus`` into ``(kind, target)``.

  ``kind`` is one of ``'asset'``, ``'file'``, ``'computed'``, or ``''`` (unknown
  / not provided). ``target`` is the asset id (for ``asset``) or the normalized
  workspace path (for ``file``); ``''`` otherwise. Anything unrecognized degrades
  to unknown, which preserves the legacy behavior (the State Resolver calibration
  pass is the backstop that validates such checks against replayed state).
  """
  s = str(raw or "").strip()
  if not s:
    return "", ""
  low = s.lower()
  if low.startswith("asset:"):
    return "asset", s.split(":", 1)[1].strip()
  if low.startswith("file:"):
    return "file", _normalize_ws_path(s.split(":", 1)[1].strip())
  if low in {"computed", "derived", "aggregate", "aggregated", "calc", "calculated"}:
    return "computed", ""
  return "", ""


# Cap grounding snippets so injecting real asset/file content into the resolution
# prompt improves grounding (P1a) without blowing up prompt size.
_GROUNDING_OCR_CAP = 600
_GROUNDING_FILE_CAP = 600


def _fold_for_grounding(text: str) -> str:
  """Case/whitespace-insensitive fold matching ``FileChecker`` ``contains``.

  The deterministic ``file_check`` grades ``contains`` after collapsing
  whitespace runs and lower-casing (``file_checker._fold``); the grounding gate
  MUST use the SAME semantics so a value it accepts cannot later fail the literal
  check it was meant to guarantee.
  """
  return " ".join(str(text or "").split()).lower()


def _truncate_grounding(text: str, limit: int) -> str:
  s = str(text or "")
  return s if len(s) <= limit else s[:limit] + "…"


def _asset_grounding_blob(asset: dict[str, Any]) -> str:
  """Searchable text of an asset: ``caption`` (str or json) + ``ocr_text``."""
  cap = asset.get("caption")
  cap_s = (
      json.dumps(cap, ensure_ascii=False)
      if isinstance(cap, (dict, list))
      else str(cap or "")
  )
  return f"{cap_s}\n{str(asset.get('ocr_text') or '')}"


def _collect_round_workspace_texts(
    layer1b_payload: dict[str, Any],
    narrative_payload: dict[str, Any],
    prior_updates: list[Any] | None = None,
) -> dict[str, str]:
  """Map ``normalized_path -> content`` for files readable in this round.

  Combines three agent-visible sources, in the order the solving agent sees
  them, normalizing paths (leading '/' stripped) to match command/check paths:

  1. the base workspace (layer 1b);
  2. ``prior_updates`` -- ``workspace_adds`` written by EARLIER rounds. A task
     for round ``rN`` ships the workspace accumulated over ``r1..rN``, so a file
     authored back in ``r04`` is still readable in ``r09``. Omitting these was a
     confidentiality blind spot: measured on a full 60-scenario corpus, 36 of
     the 50 surviving workspace leak sites (72%) were values already present in
     an earlier round's copy of a file, which is why the observed leak rate rose
     monotonically with round index (r02: 2 -> r10: 9) even though the gate's
     workspace arm was live and rejecting 52 base-workspace picks;
  3. this round's update ``workspace_adds`` (``narrative_payload`` is already
     round-scoped).

  Later sources overwrite earlier ones for the same path, so a file rewritten in
  a later round is scanned in its most recent (agent-visible) form.
  """
  texts: dict[str, str] = {}

  def _add(path: Any, content: Any) -> None:
    p = str(path or "").strip()
    if not p:
      return
    if p.startswith("/"):
      p = p.lstrip("/")
    texts[p] = str(content or "")

  def _add_updates(updates: Any) -> None:
    for upd in updates or []:
      if not isinstance(upd, dict):
        continue
      for add in upd.get("workspace_adds", []) or []:
        if isinstance(add, dict):
          _add(add.get("path"), add.get("content"))

  workspace_data = layer1b_payload.get("workspace", {})
  base_files: list[Any] = []
  if isinstance(workspace_data, dict):
    base_files = workspace_data.get("files", []) or []
  elif isinstance(workspace_data, list):
    base_files = workspace_data
  for f in base_files:
    if isinstance(f, dict):
      _add(f.get("path"), f.get("content"))

  _add_updates(prior_updates)
  _add_updates(narrative_payload.get("updates", []))
  return texts


def _message_text(msg: Any) -> str:
  """Extract the agent-visible text of one chat message (loose dict/str shapes).

  Mirrors the key fallbacks ``SessionMessage.from_dict`` accepts
  (``text``/``content``/``body``/``message``) so the confidentiality scan sees
  the same string the materializer eventually renders into the session JSONL.
  """
  if isinstance(msg, str):
    return msg
  if isinstance(msg, dict):
    for key in ("text", "content", "body", "message"):
      val = msg.get(key)
      if val:
        return str(val)
  return ""


def _collect_round_session_texts(
    layer1b_payload: dict[str, Any],
    narrative_payload: dict[str, Any],
) -> list[str]:
  """Agent-visible CHAT/session message texts readable in this round.

  Combines the scenario-wide base chat sessions (layer 1b, generated in Stage 3
  BEFORE this round's decisive value is chosen -- so they can never have been
  nudged about it) with this round's async ``session_appends``
  (``narrative_payload`` is already round-scoped). Each list entry is one
  message's text. This is the chat side-channel the confidentiality gate must
  scan: the materializer renders exactly these strings into the session JSONL the
  solving agent reads (:func:`conflict_vlm_materializer` step 2), so a decisive
  value transcribed here is text-solvable without ever opening the asset.
  """
  texts: list[str] = []

  def _add_session_messages(session: Any) -> None:
    if not isinstance(session, dict):
      return
    msgs = (
        session.get("messages")
        or session.get("chat")
        or session.get("dialogs")
        or []
    )
    for m in msgs:
      t = _message_text(m)
      if t:
        texts.append(t)

  for session in layer1b_payload.get("sessions", []) or []:
    _add_session_messages(session)

  for upd in narrative_payload.get("updates", []) or []:
    if not isinstance(upd, dict):
      continue
    for msg in upd.get("session_appends", []) or []:
      t = _message_text(msg)
      if t:
        texts.append(t)
  return texts


def _collect_round_mock_texts(
    narrative_payload: dict[str, Any],
    prior_rounds: list[Any] | None = None,
) -> list[str]:
  """Agent-REACHABLE mock subagent text for this round (the delegation channel).

  A mock subagent is the third agent-visible text surface, alongside workspace
  files and chat sessions, and historically the leakiest: measured on two full
  60-scenario corpora, the decisive value was reachable through a mock in 25.6%
  (3.6-flash) and 26.6% (3.5-flash) of asset-locus rounds -- roughly 4.5x the
  workspace-file leak rate, and essentially unchanged between the two because no
  gate has ever scanned this surface. A round whose answer a mock will simply
  recite is solvable by delegating and running ``message search``, never opening
  the asset.

  Only the two fields the SOLVING agent can actually reach are collected:

  * ``mock_response`` -- delivered verbatim when the completion fires and
    mirrored into the searchable message pool
    (:mod:`clasharness.core.environment`, ``_record_subagent_message``), so the
    agent reads it with ``message search --channel <reply_channel>``;
  * ``state_effects`` -- executed for real, so a ``message send`` effect lands in
    the same searchable pool and a ``file create`` effect lands in the workspace.

  ``private_knowledge`` and ``capabilities`` are deliberately NOT collected: the
  materializer strips both before the task ships (only ``agent_role``,
  ``mock_response``, ``virtual_duration_seconds``, ``state_effects`` and
  ``reply_channel`` survive into ``TaskData.private``), so they are authoring-LLM
  context that no solving agent can ever read. Scanning them would reject or
  rewrite ~23% of rounds over a value the agent cannot see -- a pure loss of
  scenario diversity for zero confidentiality gain.

  ``prior_rounds`` supplies earlier rounds' mocks, whose file/message effects
  persist into this round's initial state.
  """
  texts: list[str] = []

  def _add_mocks(container: Any) -> None:
    if not isinstance(container, dict):
      return
    for mock in container.get("mock_subagents", []) or []:
      if not isinstance(mock, dict):
        continue
      response = mock.get("mock_response")
      if response:
        texts.append(str(response))
      for effect in mock.get("state_effects", []) or []:
        if effect:
          texts.append(str(effect))

  for prev in prior_rounds or []:
    _add_mocks(prev)

  for rnd in narrative_payload.get("rounds", []) or []:
    _add_mocks(rnd)
  return texts


# Message-list keys ``_collect_round_session_texts`` accepts on a session dict.
_SESSION_MESSAGE_KEYS = ("messages", "chat", "dialogs")
# Fields ``_collect_round_mock_texts`` reads off a mock subagent.
_MOCK_TEXT_KEYS = ("mock_response", "state_effects")


class ExtractorBlindError(RuntimeError):
  """A confidentiality collector went blind on a surface that has content.

  Raised when a round's payload declares an agent-visible text surface
  (workspace files, chat messages, mock subagent output) but the corresponding
  collector returned nothing from it. That combination means the matching arm of
  the confidentiality gate silently degraded to a no-op: it scans an empty
  corpus, finds no leak, and reports the round clean.

  This is the exact failure that hid the mock-subagent channel for two full
  60-scenario corpora -- the decisive value was reachable through a mock in
  25.6% / 26.6% of asset-locus rounds while the gate reported clean, because no
  arm was scanning that surface at all. A blind gate is worse than no gate: it
  produces a confident false negative. So it fails loudly rather than passing
  quietly. The driver isolates this per scenario version
  (``generate_conflict_vlm_scenarios._generate_version_task``), records it in
  ``stats_tracker.final_failures`` and continues with the remaining versions, so
  the blast radius is one counted version rather than the run.
  """


def _audit_extractor_liveness(
    *,
    layer1b_payload: dict[str, Any],
    narrative_payload: dict[str, Any],
    prior_updates: list[Any] | None,
    prior_rounds: list[Any] | None,
    workspace_texts: dict[str, str],
    session_texts: list[str],
    mock_texts: list[str],
    round_id: str = "",
) -> None:
  """Assert each confidentiality collector actually saw its surface.

  Two distinct anomalies are detected, both by re-deriving "was there anything
  to extract" from the payload independently of the collector's return value:

  * **populated-but-empty** -- the collector's own keys are present and hold
    content, yet it returned nothing. An unambiguous collector bug (a filter
    that drops every entry), with no legitimate cause;
  * **schema drift** -- a non-empty container exposes NONE of the keys the
    collector reads (e.g. an authoring prompt change renames ``messages`` to
    ``turns``, or ``workspace`` stops carrying ``files``). The collector then
    reads nothing and the arm dies silently.

  Both raise :class:`ExtractorBlindError`. Neither fires on a legitimately empty
  surface -- a scenario with no chat sessions, or a round with no mocks, has
  nothing declared and is skipped. Measured across the 118 shipped scenario
  specs of both generated corpora, the observed rate of both anomalies is 0, so
  this is a regression tripwire rather than a repair for a live defect.
  """
  where = f"round {round_id}: " if round_id else ""
  problems: list[str] = []

  # ---- surface 1: workspace files -----------------------------------------
  workspace_data = layer1b_payload.get("workspace")
  base_files: list[Any] = []
  if isinstance(workspace_data, dict):
    base_files = workspace_data.get("files") or []
    if workspace_data and "files" not in workspace_data:
      problems.append(
          f"workspace container is a non-empty dict with no 'files' key "
          f"(keys={sorted(workspace_data)[:6]}); _collect_round_workspace_texts "
          f"reads only 'files', so the base workspace was not scanned"
      )
  elif isinstance(workspace_data, list):
    base_files = workspace_data

  def _declared_adds(updates: Any) -> int:
    n = 0
    for upd in updates or []:
      if not isinstance(upd, dict):
        continue
      for add in upd.get("workspace_adds", []) or []:
        if isinstance(add, dict) and str(add.get("path") or "").strip():
          n += 1
    return n

  declared_files = sum(
      1 for f in base_files
      if isinstance(f, dict) and str(f.get("path") or "").strip()
  )
  declared_files += _declared_adds(prior_updates)
  declared_files += _declared_adds(narrative_payload.get("updates", []))
  if declared_files and not workspace_texts:
    problems.append(
        f"{declared_files} workspace file(s) declared with a path but "
        f"_collect_round_workspace_texts returned 0 entries"
    )

  # ---- surface 2: chat sessions -------------------------------------------
  sessions = layer1b_payload.get("sessions") or []
  declared_msgs = 0
  keyless_sessions = 0
  for ses in sessions:
    if not isinstance(ses, dict) or not ses:
      continue
    container = next(
        (ses[k] for k in _SESSION_MESSAGE_KEYS if ses.get(k)), None
    )
    if container is None:
      if not any(k in ses for k in _SESSION_MESSAGE_KEYS):
        keyless_sessions += 1
      continue
    declared_msgs += sum(1 for m in container if _message_text(m))
  for upd in narrative_payload.get("updates", []) or []:
    if isinstance(upd, dict):
      declared_msgs += sum(
          1 for m in (upd.get("session_appends") or []) if _message_text(m)
      )
  if keyless_sessions:
    problems.append(
        f"{keyless_sessions} session(s) expose none of "
        f"{list(_SESSION_MESSAGE_KEYS)}; the chat surface was not scanned"
    )
  if declared_msgs and not session_texts:
    problems.append(
        f"{declared_msgs} chat message(s) declared but "
        f"_collect_round_session_texts returned 0 texts"
    )

  # ---- surface 3: mock subagents ------------------------------------------
  declared_mock_texts = 0
  for container in list(prior_rounds or []) + list(
      narrative_payload.get("rounds", []) or []
  ):
    if not isinstance(container, dict):
      continue
    for mock in container.get("mock_subagents", []) or []:
      if not isinstance(mock, dict):
        continue
      if mock.get("mock_response"):
        declared_mock_texts += 1
      declared_mock_texts += sum(
          1 for e in (mock.get("state_effects") or []) if e
      )
  if declared_mock_texts and not mock_texts:
    problems.append(
        f"{declared_mock_texts} agent-reachable mock text(s) declared "
        f"({list(_MOCK_TEXT_KEYS)}) but _collect_round_mock_texts returned 0"
    )

  if problems:
    raise ExtractorBlindError(
        f"{where}confidentiality collector(s) went blind -- "
        + "; ".join(problems)
    )


def _repair_round_mock_leaks(
    client: Any,
    value: str,
    asset_hint: str,
    narrative_payload: dict[str, Any],
    logger: Any = None,
) -> int:
  """LLM-rewrite THIS round's mock text that transcribes ``value``, in place.

  The mocks sub-layer runs BEFORE resolution, so it authors its dialogue without
  knowing which value the resolver will later make decisive; a collision is
  therefore expected rather than a modelling error, and rejecting the value into
  the retry loop over it would burn the retry budget on ~26% of rounds and push
  them into degradation. Rewriting the mock instead keeps the resolver's natural
  choice of value AND the scenario's narrative variety -- the subagent still does
  the same job, it just refers to "the value on the dashboard" rather than
  reciting it.

  Only the CURRENT round's mocks are touched. Sub-layer order within a round is
  instruction -> mocks -> updates -> assets -> resolution -> commands -> checkers,
  so at this point no checker exists yet for this round and the rewrite cannot
  invalidate one. EARLIER rounds are deliberately left alone: their checkers are
  already generated, so a retroactive edit could silently break them. A value
  that collides with an earlier round's mock is left to the confidentiality gate,
  which rejects it into the resolution retry loop -- there the value, not the
  narrative, is the free variable.

  Never raises: any failure leaves the text untouched and lets the gate be the
  backstop. Returns the number of texts successfully rewritten.
  """
  folded_value = _fold_for_grounding(value)

  def _leaks(text: Any) -> bool:
    return bool(text) and folded_value in _fold_for_grounding(str(text))

  repair_fn = getattr(client, "generate_layer", None)
  if not callable(repair_fn):
    return 0

  def _rewrite(text: str) -> str | None:
    """Return a rewritten text with the value gone, or None to keep the original."""
    try:
      out = repair_fn(
          "repair_value_leak",
          {
              "text": text,
              "leaked_value": value,
              "asset_hint": asset_hint,
          },
      )
    except Exception as exc:  # noqa: BLE001 - repair must never break generation
      if logger:
        logger(f"  [Resolution] mock leak repair errored ({exc}); leaving text as-is")
      return None
    edited = ""
    if isinstance(out, dict):
      edited = str(out.get("edited_text") or "").strip()
    # Trust nothing: a rewrite that did not actually remove the value (or that
    # emptied the text) is worse than no rewrite, because it would silently
    # satisfy nothing while looking repaired.
    if not edited or _leaks(edited):
      return None
    return edited

  repaired = 0
  for rnd in narrative_payload.get("rounds", []) or []:
    if not isinstance(rnd, dict):
      continue
    for mock in rnd.get("mock_subagents", []) or []:
      if not isinstance(mock, dict):
        continue
      if _leaks(mock.get("mock_response")):
        edited = _rewrite(str(mock["mock_response"]))
        if edited is not None:
          mock["mock_response"] = edited
          repaired += 1
      effects = mock.get("state_effects") or []
      for idx, effect in enumerate(effects):
        if not _leaks(effect):
          continue
        edited = _rewrite(str(effect))
        if edited is not None:
          effects[idx] = edited
          repaired += 1
  return repaired


def _build_resolution_grounding_text(
    assets: list[dict[str, Any]],
    workspace_texts: dict[str, str],
) -> str:
  """Render capped asset text + file contents for the resolution prompt (P1a).

  Historically the resolver was only shown ``id/kind/path/description`` yet was
  told to return a value "verbatim present in an asset's ocr_text/caption or a
  workspace file" -- content it never saw. Surfacing the real (capped) content
  lets the model PICK a genuinely present value instead of inventing one.
  """
  lines: list[str] = []
  seen: set[str] = set()
  for a in assets:
    aid = str(a.get("id", "")).strip()
    if not aid or aid in seen:
      continue
    seen.add(aid)
    cap = a.get("caption")
    cap_s = (
        json.dumps(cap, ensure_ascii=False)
        if isinstance(cap, (dict, list))
        else str(cap or "").strip()
    )
    ocr_s = str(a.get("ocr_text") or "").strip()
    parts = [f"  [asset:{aid}]"]
    if cap_s:
      parts.append(f"caption: {cap_s}")
    if ocr_s:
      parts.append(f"ocr_text: {_truncate_grounding(ocr_s, _GROUNDING_OCR_CAP)}")
    if len(parts) > 1:
      lines.append(" | ".join(parts))
  for path, content in workspace_texts.items():
    c = str(content or "").strip()
    if not c:
      continue
    lines.append(f"  [file:{path}] {_truncate_grounding(c, _GROUNDING_FILE_CAP)}")
  return (
      "\n".join(lines)
      if lines
      else "  (no readable asset text or file content available this round)"
  )


def _validate_round_resolution(
    resolved_value: str,
    resolved_value_locus: str,
    assets: list[dict[str, Any]],
    workspace_texts: dict[str, str],
    logger: Any = None,
) -> None:
  """Grounding gate for a round's resolution (P1b).

  Raises ``ValueError`` (fed back into the resolution retry loop as
  ``error_feedback`` so the model regenerates a grounded value at the source)
  when:
    * ``resolved_value`` is empty;
    * an ``asset:``/``file:`` locus names an artifact not visible this round;
    * an ``asset:``/``file:`` value is not present (fold-insensitive, matching
      ``FileChecker`` semantics) in that named source;
    * the locus is missing/unknown AND the value appears in no visible source.

  ``computed`` is lenient: a hit anywhere is clean; no hit is accepted with a
  WARN (a genuinely derived/aggregated value need not appear verbatim, so
  rejecting it would needlessly drop valid rounds).
  """
  value = str(resolved_value or "").strip()
  if not value:
    raise ValueError(
        "resolved_value is empty; return the exact concrete value the solver"
        " must surface -- a token that literally appears in a named visible"
        " asset's ocr_text/caption or a workspace file."
    )
  kind, target = _parse_resolved_locus(resolved_value_locus)
  folded_value = _fold_for_grounding(value)
  asset_by_id = {
      str(a.get("id", "")).strip(): a
      for a in assets
      if str(a.get("id", "")).strip()
  }

  def _hits(text: str) -> bool:
    return folded_value in _fold_for_grounding(text)

  if kind == "asset":
    if target not in asset_by_id:
      raise ValueError(
          f"resolved_value_locus 'asset:{target}' does not name a visible asset"
          f" this round. Visible asset ids: {sorted(asset_by_id)}."
      )
    if not _hits(_asset_grounding_blob(asset_by_id[target])):
      raise ValueError(
          f"resolved_value '{value}' does not appear in asset '{target}'"
          " ocr_text/caption. Pick a value literally present there, or set the"
          " correct locus."
      )
    return
  if kind == "file":
    if target not in workspace_texts:
      raise ValueError(
          f"resolved_value_locus 'file:{target}' is not a readable workspace"
          f" file this round. Valid paths: {sorted(workspace_texts)}."
      )
    if not _hits(workspace_texts[target]):
      raise ValueError(
          f"resolved_value '{value}' does not appear in file '{target}'. Pick a"
          " value literally present there, or set the correct locus."
      )
    return

  grounded_anywhere = any(
      _hits(_asset_grounding_blob(a)) for a in assets
  ) or any(_hits(t) for t in workspace_texts.values())

  if kind == "computed":
    if not grounded_anywhere and logger:
      logger(
          f"  [Resolution] computed resolved_value '{value}' is not present"
          " verbatim in any visible source (accepted as a derived value)."
      )
    return

  # Unknown / missing locus: require grounding somewhere so an unlabeled guess
  # cannot slip through ungrounded.
  if not grounded_anywhere:
    raise ValueError(
        f"resolved_value_locus '{resolved_value_locus}' is missing/unknown and"
        f" resolved_value '{value}' is not found in any visible asset text or"
        " workspace file. Provide a locus ('asset:<id>' | 'file:<path>' |"
        " 'computed') and a grounded value."
    )


def _validate_round_confidentiality(
    resolved_value: str,
    resolved_value_locus: str,
    instruction: str,
    workspace_texts: dict[str, str],
    session_texts: list[str] | None = None,
    mock_texts: list[str] | None = None,
    logger: Any = None,
) -> None:
  """Confidentiality gate -- the DUAL of the grounding gate (I1, narrative).

  ``_validate_round_resolution`` guarantees the decisive value CAN be found in
  its locus (the round is answerable). This gate guarantees it can be found
  *only* there: the value must be ABSENT from every other agent-visible TEXT
  surface, so the round cannot be solved from a text side-channel without
  engaging the locus. For an ``asset:`` locus that means the value lives only in
  the rendered image -- the agent must actually LOOK. This is the
  correct-by-construction half of the difficulty-collapse fix; the post-generation
  strict validator is the backstop.

  Medium strictness (fold-insensitive, matching grounding/``FileChecker``
  semantics), it raises ``ValueError`` (fed back into the SAME resolution retry
  loop) when the value appears in:
    * the round INSTRUCTION -- never; the question must not state its own answer;
    * any workspace FILE other than a ``file:`` locus;
    * any agent-visible CHAT/session message (``session_texts``) -- chat is never
      a declared locus, so a decisive value transcribed there makes the round
      text-solvable from the session JSONL without opening the asset. This is the
      exact leak channel behind the observed difficulty collapse: base chat
      sessions are authored in Stage 3, before the round's decisive value is even
      chosen, so a soft "don't transcribe the value" prompt cannot protect them
      -- only this post-resolution scan can;
    * any agent-reachable MOCK SUBAGENT text (``mock_texts``) -- a mock's
      ``mock_response`` and its executed ``state_effects`` both land in the
      solving agent's searchable message pool, so a mock that recites the value
      makes the round solvable by "delegate, then ``message search``" without
      ever opening the asset. Same authored-before-resolution problem as chat.

  The three arms SHORT-CIRCUIT in the order above, so their rejection counts are
  conditional and do not rank the surfaces. An earlier version of this docstring
  read them as a ranking and called the mock arm "the LARGEST of the three
  surfaces (~26%)"; the logs do not support that on either corpus measured
  (rejections v2: workspace 73 / session 56 / mock 48; v3: workspace 256 /
  session 125 / mock 10), and because a round leaking on both workspace and mock
  is only ever REPORTED as workspace, neither set of counts can establish a
  ranking in the first place. Sizing the surfaces would need each arm scanned
  independently; until that is measured, no ranking is claimed here.

  It deliberately does NOT scan other visual assets: a corroborating chart/photo
  that shows the same value is still pixels, so it does not make the round
  text-solvable. All three scans run for EVERY locus -- see the note at the
  locus split below for why ``computed`` stopped being exempt. Trivially short
  values (<3 non-space chars) are skipped to avoid matching ubiquitous
  substrings.
  """
  value = str(resolved_value or "").strip()
  if len(value.replace(" ", "")) < 3:
    return
  folded_value = _fold_for_grounding(value)

  def _hits(text: str) -> bool:
    return folded_value in _fold_for_grounding(text)

  if _hits(instruction):
    raise ValueError(
        f"resolved_value '{value}' appears verbatim in the round instruction;"
        " the question must NOT spell out its own answer. Pick a decisive value"
        " that lives only in the evidence, not the prompt."
    )

  kind, target = _parse_resolved_locus(resolved_value_locus)
  # EVERY locus is scanned, including ``computed`` and unknown/empty ones. This
  # used to exempt them, on the theory that "a computed value legitimately
  # aggregates several visible sources" so finding it in a file proves nothing.
  # The theory was untestable when written and is now refuted: ``computed`` went
  # from 0 occurrences on the pre-binding corpus to 43 on the next one, of which
  # 43/43 were emitted only AFTER this gate had rejected the round for leaking
  # and 0/43 were a first choice. It was not a locus the resolver believed in;
  # it was the one channel the gate did not guard, and binding pressure found
  # it. Together with 13 similarly-relabelled ``file`` rounds it laundered 56
  # rounds into a text-solvable state -- 11.9% of the corpus, reported as 1.8%
  # because the exclusivity report was not scanning them either.
  #
  # Exempting them was never necessary for the stated reason: this scan only
  # rejects a VERBATIM (fold-insensitive) hit on the computed RESULT. A value
  # genuinely derived from several visible sources -- a difference, a total, a
  # count -- is not sitting written out in a file; if it is, the round is
  # text-solvable whatever the locus label claims.
  # ``file:`` remains exempt for its OWN declared path only (per-path, below);
  # every other path, and every other locus kind, is scanned.
  for path, text in workspace_texts.items():
    if kind == "file" and path == target:
      continue
    if not _hits(text):
      continue
    # Two different mistakes reach this line and they need DIFFERENT repairs.
    # A round that declared a locus is leaking and must change its value. A
    # round that declared NO locus may simply be an honest text-grounded round
    # that never said so -- telling it to "choose a unique value" is advice it
    # cannot act on, and an unactionable rejection burns the retry budget and
    # degrades the round instead of fixing it. Undeclared rounds are told to
    # declare instead. They are NOT exempted: exempting the unnamed locus is
    # how ``computed`` became the laundering channel (0 occurrences before the
    # pressure existed, 43 after, 43/43 only ever chosen post-rejection), and
    # an unnamed escape hatch is the same hatch one blank field away.
    if kind not in {"asset", "file", "computed"}:
      raise ValueError(
          f"resolved_value '{value}' appears in workspace file '{path}' but the"
          f" round declares no locus (resolved_value_locus={resolved_value_locus!r})."
          " An undeclared locus cannot be told apart from a text leak. If that"
          f" file is genuinely where the value is read, declare it:"
          f" resolved_value_locus='file:{path}'. If the value is meant to be read"
          " from an asset's pixels, keep the asset locus and choose a value that"
          " does not also appear in the workspace text."
      )
    raise ValueError(
        f"resolved_value '{value}' also appears in workspace file '{path}',"
        f" which is not the declared locus ('{resolved_value_locus}'). It would"
        " be text-solvable without engaging the decisive evidence -- choose a"
        " value that is unique to the locus (ideally read only from an asset's"
        " pixels)."
    )
  # Chat/session confinement: the value must not be transcribed into any
  # agent-visible chat message. Chat is never a declared locus, so no exemption
  # is needed (unlike the ``file:`` locus above). Characters may REFER to "the
  # figure on the dashboard" -- only a VERBATIM (fold-insensitive) hit trips this.
  for text in session_texts or []:
    if _hits(text):
      raise ValueError(
          f"resolved_value '{value}' also appears in an agent-visible chat/"
          f"session message, which is not the declared locus"
          f" ('{resolved_value_locus}'). It would be text-solvable from the"
          " session log without engaging the decisive evidence -- choose a value"
          " unique to the locus (ideally read only from an asset's pixels), and"
          " keep chat referring to 'the figure on the dashboard' rather than"
          " transcribing the actual value."
      )
  # Mock-subagent confinement. This comment used to call the delegation channel
  # "the leakiest surface of the three (~26% of asset-locus rounds)"; that
  # ranking is RETRACTED -- see the docstring, the arms short-circuit so their
  # counts cannot be compared. The arm is here on its mechanism, not its size: a
  # mock's ``mock_response`` and its executed ``state_effects`` both reach the
  # solving agent's searchable message pool, so a mock that recites the value
  # turns the round into "delegate, then ``message search``" and the asset is
  # never opened.
  # A mock may still ACT on the value and refer to it obliquely -- only a
  # VERBATIM hit trips this.
  for text in mock_texts or []:
    if _hits(text):
      raise ValueError(
          f"resolved_value '{value}' also appears in a mock subagent's"
          " response or state_effects, which is not the declared locus"
          f" ('{resolved_value_locus}'). The solving agent can read it by"
          " delegating and running 'message search', so the round is solvable"
          " without engaging the decisive evidence -- choose a value unique to"
          " the locus (ideally read only from an asset's pixels), and keep the"
          " subagent referring to 'the value on the dashboard' rather than"
          " transcribing it."
      )
  if logger:
    logger(
        f"  [Resolution] confidentiality ok: '{value}' is confined to locus"
        f" '{resolved_value_locus}'."
    )


def _validate_round_decision_outcome(outcome: str, resolved_value: str) -> None:
  """Light gate on a round's ``decision_outcome`` verdict label (item ③).

  Unlike ``resolved_value`` (the decisive evidence, which must be grounded in a
  visible source), ``decision_outcome`` is an authored short free-form label
  naming the CORRECT resolution/verdict. We only enforce that it is present,
  concise (a label, not a paragraph), and distinct from ``resolved_value`` -- the
  value is the evidence, the outcome is the decision. Raising feeds the shared
  resolution retry loop so a degenerate classification is regenerated at the
  source rather than shipped.
  """
  o = str(outcome or "").strip()
  if not o:
    raise ValueError(
        "decision_outcome is empty. Return a SHORT free-form label naming the"
        " correct resolution/verdict for this round (e.g. 'roll back to v4.2')."
    )
  if len(o) > 120:
    raise ValueError(
        "decision_outcome is too long; return a SHORT label (a few words), not a"
        " full sentence or explanation."
    )
  if o.lower() == str(resolved_value or "").strip().lower():
    raise ValueError(
        "decision_outcome must be DIFFERENT from resolved_value: the value is the"
        " decisive evidence; the outcome is the decision/verdict category."
    )


def _scaffold_round_checkers(
    commands: list[dict[str, Any]],
    allowed_families: list[str],
    asset_ids: list[str] | None = None,
    resolved_value: str | None = None,
    resolved_value_locus: str | None = None,
    decision_outcome: str | None = None,
    resolved_value_citable: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  """Build checker scaffolds deterministically from command lines.

  Returns ``(scaffolds, semantic_needed)``. Each scaffold is
  ``{step_number, command, checker, semantic, extra_checkers}`` where ``semantic``
  (or ``None``) records which checker field still needs an LLM-authored value plus
  a guaranteed non-empty fallback, and ``extra_checkers`` is a list of additional
  fully-formed deterministic checkers for that step (or ``None``).
  ``semantic_needed`` lists only the subset that requires the semantic pass:
  ``{step_number, field, command}``.

  Schema/verbatim-command/file-path/asset-id are chosen by code here; only the
  ``criterion``/``contains`` free-text remains for the LLM (layer 3b).

  ``resolved_value_locus`` (provenance of the decisive value) drives whether a
  *read* step becomes a ``file_check``: only when the value actually lives in that
  workspace file. Values read from an asset/OCR or that must be computed are not
  in the read file, so no doomed ``file_check`` is scaffolded - they are graded
  through the round's outcome (message/report) checkers instead.

  ``decision_outcome`` (item ③, the round's short verdict label) is asserted as a
  second deterministic ``file_check`` on the resolution-report *write* step, so
  the round's decision -- not just its decisive value -- is graded against the
  gold's own written outcome. It is only seeded where a decisive value is also
  written (the report), so it is correct-by-construction with the prompt-steered
  gold and the C2 calibration pass degrades it if the gold does not write it.
  """
  known_asset_ids = {a for a in (asset_ids or []) if a}
  resolved = str(resolved_value or "").strip()
  outcome = str(decision_outcome or "").strip()
  locus_kind, locus_target = _parse_resolved_locus(resolved_value_locus)
  scaffolds: list[dict[str, Any]] = []
  semantic_needed: list[dict[str, Any]] = []
  for entry in commands:
    step_number = entry.get("step_number")
    command = str(entry.get("command", "")).strip()
    category, spec = _classify_step_command(command, allowed_families)
    checker: dict[str, Any] | None = None
    semantic: dict[str, Any] | None = None
    extra_checkers: list[dict[str, Any]] | None = None

    if category == "delegation":
      name = getattr(spec, "name", "") or ""
      primary_flag = _DELEGATION_CRITERION_FLAGS.get(name, "--message")
      fallback = (_extract_flag_value(command, primary_flag) or "").strip()
      if not fallback:
        for alt in ("--instruction", "--message", "--question", "--target"):
          alt_val = (_extract_flag_value(command, alt) or "").strip()
          if alt_val:
            fallback = alt_val
            break
      if not fallback:
        fallback = "performs the delegated communication"
      checker = {
          "type": "effect",
          "effect_type": "messages_sent",
          "condition": "field_semantic_match",
          "expected": {"field": "message", "criterion": ""},
      }
      semantic = {"field": "criterion", "fallback": fallback}

    elif category == "file_write":
      norm_path = _normalize_ws_path(_extract_flag_value(command, "--path") or "")
      if not norm_path:
        checker = {"type": "command_check", "command": command}
      elif _is_code_file(norm_path):
        # Writing a source/script file: grade that the write action happened,
        # not that a decisive business value is embedded in the code. Asserting
        # ``contains <value>`` inside executable code is fragile and
        # semantically wrong -- the decisive value belongs in a data/report
        # file, so degrade to a command_check (no semantic pass).
        checker = {"type": "command_check", "command": command}
      else:
        # Path-agnostic deliverable grading: the report file the agent writes is
        # DISCRETIONARY, so grade the CONTENT it produced via the ``files_created``
        # effect stream (which lists only agent-created files -- never pre-seeded
        # workspace files, so a seeded doc can never satisfy it) instead of binding
        # a fixed path. Mirrors the text materializer (materializer.py:651) and
        # vlm_materializer.py:1704.
        # S6: grading follows the DECLARATION the step that chose the value
        # made about it, not a regex re-reading its shape afterwards. The shape
        # router accepted 127 of 523 corpus values (24.3%) and among the
        # rejections were 42 plain number+unit readings ('182.1 U/L') that are
        # perfectly safe to match exactly.
        #
        # The exact arm STAYS required. Over 11,295 aligned round-runs it
        # passed 61.4% where the verdict judge passed 88.0% -- it is the
        # sharpest discriminator in the suite, and 82.5% of its failures are
        # runs whose report never states the value in any form. Demoting it
        # would hand those rounds a pass for reaching the right verdict without
        # ever finding the evidence.
        #
        # An undeclared value defaults to semantic. That is the degraded path
        # (S1 produced nothing, or the resolver broke its binding), and it
        # costs no difficulty: the semantic arm passed 59.7% against the exact
        # arm's 61.4%.
        if resolved and resolved_value_citable:
          checker = {
              "type": "effect",
              "effect_type": "files_created",
              "condition": "field_contains",
              "expected": {"field": "content", "value": resolved},
          }
        elif resolved:
          checker = {
              "type": "effect",
              "effect_type": "files_created",
              "condition": "field_semantic_match",
              "expected": {
                  "field": "content",
                  "criterion": (
                      f"A written report states the decisive finding: {resolved}"
                  ),
              },
          }
        else:
          # No decisive value to seed: let the semantic pass (layer 3b) author a
          # specific substring the produced report must contain.
          checker = {
              "type": "effect",
              "effect_type": "files_created",
              "condition": "field_contains",
              "expected": {"field": "content", "value": ""},
          }
          semantic = {"field": "contains", "fallback": resolved}
        # ③: this non-code write is the round's resolution report (the prompt steers
        # the gold to write the decisive value AND the verdict here). Grade the
        # round's decision against the agent's own produced report content --
        # SEMANTICALLY, never verbatim (a decision/verdict phrase is ambiguous and
        # no independent solver reproduces it word-for-word) and path-agnostically
        # via the ``files_created`` stream. Only when a decisive value is also
        # written (a real report).
        if outcome and resolved:
          extra_checkers = [
              {
                  "type": "effect",
                  "effect_type": "files_created",
                  "condition": "field_semantic_match",
                  "expected": {
                      "field": "content",
                      "criterion": (
                          f"A written report states the decision/verdict: {outcome}"
                      ),
                  },
              }
          ]

    elif category == "file_read":
      # A read of a real workspace file is only a graded step when we know the
      # decisive value it must reveal; then it becomes a deterministic file_check
      # (no semantic pass). Otherwise it stays 'plain' (no checker). A read of a
      # source/script file never carries the decisive business value as text (it
      # lives in a data/report file), so such a read stays 'plain'.
      norm_path = _normalize_ws_path(_extract_flag_value(command, "--path") or "")
      if norm_path and resolved and not _is_code_file(norm_path):
        # Provenance gate (closes the A-class defect): only assert the decisive
        # value lives in THIS read file when it actually resides in a workspace
        # file. When the value is read from an asset/OCR ('asset:') or must be
        # computed ('computed'), the read file need not contain it, so a
        # 'file_check contains <value>' here can never pass on replay -- leave the
        # read 'plain' and let the round's outcome (message/report) checkers grade
        # the value. A 'file:' locus pointing at a different file likewise means
        # this read does not carry the value. Unknown locus keeps the legacy
        # file_check (the State Resolver calibration pass validates it).
        binds_here = not (
            locus_kind in {"asset", "computed"}
            or (locus_kind == "file" and locus_target and locus_target != norm_path)
        )
        if binds_here:
          # A read grades the FIXED materialized input file, not solver-authored
          # content: there is no paraphrase risk (the solver does not rewrite what
          # it reads), and the C2 calibration pass degrades/drops any doomed
          # ``contains`` here against the resolved final state. So a read stays an
          # exact ``contains`` -- reliable and free -- rather than a semantic check
          # (which would bypass C2's deterministic backstop and cost a judge call).
          checker = {"type": "file_check", "path": norm_path, "contains": resolved}

    elif category == "asset_access":
      asset_id = (_extract_flag_value(command, "--id") or "").strip()
      if not asset_id or (known_asset_ids and asset_id not in known_asset_ids):
        # Missing id, or an id we cannot confirm is visible -> degrade safely.
        checker = {"type": "command_check", "command": command}
      else:
        checker = {
            "type": "effect",
            "effect_type": "assets_accessed",
            "condition": "field_contains",
            "expected": {"field": "id", "value": asset_id},
        }

    elif category == "protocol_effect":
      # A protocol command (lock/deploy/approval/...) whose success is graded by
      # asserting the effect it emits exists. Active from M1: _EFFECT_COMMAND_NAMES
      # maps lock.acquire/release and deploy.execute/cutover/rollback to their
      # effect types, so those commands scaffold an ``exists`` effect check.
      effect_name = getattr(spec, "name", "") or ""
      effect_type = _effect_type_for_command_name(effect_name)
      target = _target_identifier_for_command(command, spec)
      if target:
        # Grade WHICH target the protocol acted on, not merely that it ran.
        # ``exists`` is satisfied by any invocation, so a solver that runs the
        # right verb against the wrong resource still scores -- and picking the
        # right resource out of the conflicting evidence IS the task. Binding the
        # identifier also makes repetition worthless: N commands against N
        # distinct targets scaffold N checks with N distinct expected values, so
        # replaying one target satisfies exactly one of them. That is why this
        # needs no count/distinctness condition, which matters because only
        # ``containment`` suppresses duplicate effects at the source (``lock
        # acquire`` re-emits on an idempotent re-acquire, so a raw ``count_gte``
        # would have been spammable).
        checker = {
            "type": "effect",
            "effect_type": effect_type,
            "condition": "any_field_equals",
            "expected": {"value": target},
        }
      else:
        # No required flag, or no parsable value -> keep the legacy assertion
        # rather than scaffold something unwinnable.
        checker = {
            "type": "effect",
            "effect_type": effect_type,
            "condition": "exists",
        }
    # 'plain' (list/read/wait/DONE/unresolved) -> no checker.

    scaffolds.append({
        "step_number": step_number,
        "command": command,
        "checker": checker,
        "semantic": semantic,
        "extra_checkers": extra_checkers,
    })
    if semantic is not None:
      semantic_needed.append({
          "step_number": step_number,
          "field": semantic["field"],
          "command": command,
      })
  return scaffolds, semantic_needed


def _parse_semantic_fields(
    fields_payload: dict[str, Any],
) -> dict[int, dict[str, str]]:
  """Index a round_checkers payload by step_number -> {criterion|contains}."""
  result: dict[int, dict[str, str]] = {}
  for entry in (fields_payload or {}).get("fields", []) or []:
    if not isinstance(entry, dict):
      continue
    sn_raw = entry.get("step_number")
    if not isinstance(sn_raw, (int, str)):
      continue
    try:
      sn_int = int(sn_raw)
    except (TypeError, ValueError):
      continue
    vals: dict[str, str] = {}
    if entry.get("criterion") not in (None, ""):
      vals["criterion"] = str(entry["criterion"])
    if entry.get("contains") not in (None, ""):
      vals["contains"] = str(entry["contains"])
    if vals:
      result[sn_int] = vals
  return result


def _apply_round_semantic_fields(
    scaffolds: list[dict[str, Any]],
    sem_map: dict[int, dict[str, str]],
) -> None:
  """Fill scaffold checker placeholders from the semantic map, else fallback.

  A delegation ``criterion`` is always forced non-empty (the strict validator
  requires it); a file ``contains`` may legitimately remain empty.
  """
  for scaffold in scaffolds:
    semantic = scaffold.get("semantic")
    if not semantic:
      continue
    checker = scaffold.get("checker")
    if not isinstance(checker, dict):
      continue
    field = semantic.get("field")
    sn_raw = scaffold.get("step_number")
    sn_int: int | None = None
    if isinstance(sn_raw, (int, str)):
      try:
        sn_int = int(sn_raw)
      except (TypeError, ValueError):
        sn_int = None
    provided = ""
    if sn_int is not None:
      provided = (sem_map.get(sn_int, {}) or {}).get(field, "") or ""
    provided = str(provided).strip()
    if field == "criterion":
      value = (
          provided
          or str(semantic.get("fallback", "")).strip()
          or "performs the delegated communication"
      )
      checker.setdefault("expected", {})
      checker["expected"]["criterion"] = value
    elif field == "contains":
      value = provided if provided else str(semantic.get("fallback", ""))
      if checker.get("type") == "effect":
        # Path-agnostic file deliverable: the substring the produced report must
        # contain lives in the ``files_created`` effect checker's expected.value,
        # not the legacy file_check's top-level ``contains`` key.
        checker.setdefault("expected", {})["value"] = value
      else:
        checker["contains"] = value


_TRAJECTORY_STOP_TOKENS = frozenset({"done", "exit", "quit"})


def _parse_outgoing_message_effects(command: str) -> list[dict[str, str]]:
  """The ``messages_sent`` effects a command contributes (mirrors the skills).

  Faithful to ``message_skill`` and the ``agent delegate`` handler -- including
  the cases that contribute NOTHING a judge can see, which is precisely what
  lets the gate catch a criterion no gold could satisfy:

  * ``message send`` requires ``--channel`` AND ``--target`` AND ``--message``;
    missing any one exits 1 and records no effect at all.
  * ``message broadcast`` never reads ``--target``. It reads ``--targets``
    (unreachable from the agent-facing tool spec) and otherwise fans out to the
    channel itself, so the documented form is a single ``in #chan`` line.
  * ``message poll`` records ``question``/``options`` but no ``message`` key, and
    ``message react`` records ``message: ""``. Both render to an empty transcript
    line, so a criterion written about a poll or a reaction is unsatisfiable by
    construction -- we return nothing for them so the gate says so.
  """
  cmd = (command or "").strip()
  try:
    tokens = shlex.split(cmd)
  except ValueError:
    return []
  if len(tokens) < 2:
    return []
  verb, action = tokens[0], tokens[1]

  if verb == "agent" and action == "delegate":
    agent = (_extract_flag_value(cmd, "--agent") or "").strip()
    instruction = (_extract_flag_value(cmd, "--instruction") or "").strip()
    if not agent or not instruction:
      return []
    return [{"target": agent, "message": instruction, "channel": ""}]

  if verb != "message":
    return []
  message = (_extract_flag_value(cmd, "--message") or "").strip()
  channel = (_extract_flag_value(cmd, "--channel") or "").strip()

  if action == "send":
    target = (_extract_flag_value(cmd, "--target") or "").strip()
    if not channel or not target or not message:
      return []
    return [{"target": target, "message": message, "channel": channel}]

  if action == "broadcast":
    targets: list[str] = []
    if "--targets" in tokens:
      idx = tokens.index("--targets") + 1
      while idx < len(tokens) and not tokens[idx].startswith("--"):
        targets.append(tokens[idx])
        idx += 1
    if len(targets) == 1 and "," in targets[0]:
      targets = [t.strip() for t in targets[0].split(",") if t.strip()]
    if not targets and channel:
      targets = [channel]
    if not targets or not message:
      return []
    return [
        {"target": t, "message": message, "channel": channel} for t in targets
    ]

  return []


# '#' followed by the characters a channel name may contain. Deliberately
# literal: the question is whether a name the criterion PRINTS exists, so
# anything fuzzier would start inventing matches of its own.
_CRITERION_CHANNEL_RE = re.compile(r"#[A-Za-z0-9][A-Za-z0-9._-]*")


def _criterion_channel_tokens(text: str) -> list[str]:
  """The '#name' channel references a graded criterion actually prints.

  Exactly two exclusions, and no more:

    * a '#' followed only by digits ('#802', '#4829') is a ticket number. No
      channel is all digits, so this costs nothing.
    * sentence punctuation glued on -- '#incident-response.' is the real channel
      '#incident-response' followed by a full stop, but '.' is legal inside a
      channel name so the regex cannot tell the two apart. Trailing separators
      are stripped before the lookup.

  A THIRD exclusion was tried and removed, and the reason is worth keeping. It
  read "an all-caps body is an identifier, not a channel" -- meant to spare
  '#R-904' and '#GR-8841'. Measured against v7 it excluded 25 mentions, of which
  19 were '#SESS-MIGRATION-OPS-01', '#SESS-RELEASE-GOV-02', '#SESS-GOV-02' and
  friends: not identifiers at all, but the session_id-derived names the broken
  channel gate FORCED onto the authors. The rule was blind to precisely the
  failure mode this gate exists to catch. Nothing distinguishes '#GR-8841' from
  '#SESS-GOV-02' by shape, so shape is the wrong question -- membership is the
  question, and the alias table already answers it.

  Bouncing a criterion over a genuine id costs one retry and, on the last
  attempt, nothing at all (the round is kept and counted). Being blind to a
  phantom channel costs every solver the point. The trade is not close. Real
  ids that must survive verbatim are protected upstream instead: the caller
  drops any token contained in the round's ``resolved_value``, so a 'contains'
  check keeps its decisive value (B3).
  """
  out: list[str] = []
  for tok in _CRITERION_CHANNEL_RE.findall(text or ""):
    tok = tok.rstrip("._-")
    body = tok[1:]
    if not body or body.isdigit():
      continue
    out.append(tok)
  return out


def _unknown_criterion_channels(
    texts: list[str], aliases: dict[str, str]
) -> list[str]:
  """Channel names a criterion prints that no shipped session can provide.

  The fairness question the whole channel effort exists to answer. A criterion
  reading "posts in #bay4-ops" when the scenario ships '#bay4-fulfillment-ops'
  is unwinnable: the solver reads the sessions it was given, posts where those
  sessions are, and loses the point for a reason that has nothing to do with the
  conflict under test. On the v7 corpus 199 graded checks failed exactly this
  way across 130 of 600 tasks.

  Empty when the scenario has no channel vocabulary at all -- with no vocabulary
  there is no valid answer, so enforcing would burn every retry for the scenario
  least able to afford it.
  """
  if not aliases:
    return []
  unknown: list[str] = []
  for text in texts:
    for tok in _criterion_channel_tokens(text):
      if aliases.get(tok.strip().lower()) is None and tok not in unknown:
        unknown.append(tok)
  return unknown


def _normalize_session_channels(sessions: list[Any] | None) -> int:
  """Give every raw session a ``channel`` key holding its declared name.

  Round generation reads the layer1b sessions as RAW DICTS, while everything
  downstream of materialization reads them as parsed ``ScenarioSession``
  objects. The parser accepts three spellings of the channel name
  (``channel`` / ``channel_name`` / ``name``); for a long time the raw-dict
  readers accepted only the first. A session that declared ``name:
  '#partnership-contract-review'`` therefore looked channel-less during
  generation, so the channel gate derived a vocabulary from the ``session_id``
  (``'#01-partnership-review'``), told the gold author that was the only legal
  channel, REJECTED the real name as nonexistent, and rewrote it away -- and
  then materialization shipped the real name, leaving the gold commands and
  every criterion derived from them pointing at a channel the scenario does not
  have.

  ``session_channel_value`` now makes both readers agree on which key to read;
  this collapses the two VIEWS as well, so no future raw-dict reader can
  reintroduce the split by reaching for ``s["channel"]`` directly. Run once
  before the round loop, on both the fresh and the checkpoint-resume path.

  Only the ``channel`` key is written -- the dict is NOT replaced by a parsed
  object -- so any other key the generator put on the session survives intact.
  Idempotent; returns how many sessions were changed (for logging).
  """
  from clasharness.scenario.spec import (  # pylint: disable=g-import-not-at-top
      session_channel_value,
  )

  changed = 0
  for session in sessions or []:
    if not isinstance(session, dict):
      continue
    declared = session_channel_value(session)
    if str(session.get("channel") or "") != declared:
      session["channel"] = declared
      changed += 1
  return changed


def _channel_vocabulary(
    sessions: list[Any], updates: list[Any] | None = None
) -> tuple[list[str], dict[str, str]]:
  """The scenario's real channel names, plus the aliases that provably mean them.

  Sourced from the shipped sessions, because that is where the graded channel
  identity comes from. Returned as ``(display names, alias -> display name)``:
  the names are what the round's author is TOLD it may use, the aliases are what
  can be repaired without guessing.

  ``updates`` is a FALLBACK witness, consulted only when the sessions yield
  nothing. One v6 scenario (``coordinated_containment_v02``) ships a single
  degenerate session -- ``session_id: ""``, ``channel: "default"`` -- so the
  session vocabulary is empty even though the scenario plainly has two channels:
  its updates post into them every round and its gold names them. Without the
  fallback the prompt block is omitted and the gate is a no-op for exactly the
  scenario least able to afford it, since the author is then told nothing at all
  about what channels exist. Strictly a fallback, never a union: on v6 the
  updates add a name the sessions lack in that one scenario only, so unioning
  everywhere would buy nothing while risking the promotion of an update's own
  invented name to canonical.

  Still empty when neither source yields a name, and every caller must treat
  "no vocabulary" as "enforce nothing" rather than "everything is invalid":
  there would be no valid answer for the author to give, so enforcing would burn
  every retry and drop the scenario outright.

  Imported lazily: ``conflict_vlm_materializer`` imports THIS module at import
  time, so a module-level import here would be circular. Deferring also keeps
  one implementation of the channel vocabulary instead of a second copy that can
  drift from the one materialization uses -- two copies drifting apart is the
  defect being fixed, not a pattern to repeat.
  """
  from clasharness.scenario.conflict_vlm_materializer import (  # pylint: disable=g-import-not-at-top
      _session_channel_aliases,
      _session_channel_registry,
      update_channel_aliases,
      update_channel_registry,
  )

  registry = _session_channel_registry(sessions)
  if registry:
    return registry, _session_channel_aliases(sessions)
  return update_channel_registry(updates or []), update_channel_aliases(
      updates or []
  )


def _canonicalize_gold_channels(
    commands: list[dict[str, Any]],
    sessions: list[Any],
    *,
    mock_subagents: list[Any] | None = None,
    updates: list[Any] | None = None,
    last_attempt: bool = False,
    log: Callable[[str], None] | None = None,
    stats: dict[str, int] | None = None,
) -> list[str]:
  """Rewrite provable channel aliases in the round; report what is left over.

  Mutates ``commands`` -- and, when given, each sub-agent's ``state_effects`` --
  in place (they are the working structures the round is being built from) and
  returns the channel values that match no shipped session, for the caller to
  bounce back to the author.

  ``mock_subagents`` is covered for the same reason the commands are, and it is
  not the smaller half: on v6 the four scenarios that invented channel names put
  54 of those names in gold commands and 104 more in sub-agent ``state_effects``
  (``payment_chargeback_dispute_v02`` alone: 10 in commands, 40 in effects). A
  sub-agent posting to a channel that does not exist is a message the solver
  cannot find by reading the session it was told about, so a criterion about
  that exchange is just as unwinnable as one about an invented gold channel.

  Commands and effects are repaired as ONE batch, not one batch each. The
  ``last_attempt`` fallback assigns by order of first appearance, so repairing
  them separately would let a single invented name be bound to one real channel
  in the gold's own ``message send`` and to a different one in the sub-agent
  reply to it -- splitting one conversation in the very step meant to keep it
  whole.

  Returns nothing to bounce when the scenario has no channel vocabulary at all:
  with no vocabulary there is no valid answer, so enforcing would burn every
  retry and discard the scenario.

  On ``last_attempt`` the leftovers are bound to real channels instead of
  reported, so a stubborn round costs one imperfect channel assignment rather
  than the whole scenario. See ``assign_unresolved_channels``.
  """
  from clasharness.scenario.conflict_vlm_materializer import (  # pylint: disable=g-import-not-at-top
      assign_unresolved_channels,
      canonicalize_gold_channels,
  )

  registry, aliases = _channel_vocabulary(sessions, updates)
  if not aliases:
    return []

  # Gather every channel-bearing string in the round into one flat batch, each
  # tagged with where to write the repaired version back.
  raw: list[str] = [str(c.get("command", "")) for c in commands]
  slots: list[tuple[Any, Any]] = [(entry, "command") for entry in commands]
  for sub in mock_subagents or []:
    if not isinstance(sub, dict):
      continue
    effects = sub.get("state_effects")
    if not isinstance(effects, list):
      continue
    for i, effect in enumerate(effects):
      # Only plain-string effects are commands; dict effects carry no
      # '--channel' and must survive untouched at their original index.
      if isinstance(effect, str):
        raw.append(effect)
        slots.append((effects, i))

  fixed, unresolved = canonicalize_gold_channels(raw, aliases)
  if unresolved and last_attempt:
    fixed, mapping = assign_unresolved_channels(fixed, aliases, registry)
    # A blind bind is a channel decision made FOR the author because retries ran
    # out; it is the same defect class the criterion gate counts, so the run's
    # acceptance report needs to see it even though it never fails the build.
    if stats is not None:
      stats["blind_bound"] = stats.get("blind_bound", 0) + len(mapping or {})
    if mapping and log is not None:
      log(
          "  [ChannelVocab] final attempt -- bound invented channel(s) to real"
          " ones rather than discarding the scenario: "
          + ", ".join(f"{k!r} -> {v!r}" for k, v in sorted(mapping.items()))
      )
    unresolved = []
  for (container, key), new_value in zip(slots, fixed):
    container[key] = new_value
  return unresolved


def _project_round_subagents_for_gate(
    mock_subagents: list[Any],
    sessions: list[Any],
    solution_commands: list[str],
) -> list[dict[str, Any]]:
  """Sub-agents as materialization will leave them, for the gate to predict from.

  Thin, deliberately logic-free wrapper: the transforms live in the materializer
  and are called there, so the gate's prediction and the thing it predicts stay
  the same code. See ``conflict_vlm_materializer.project_round_subagents``.
  """
  from clasharness.scenario.conflict_vlm_materializer import (  # pylint: disable=g-import-not-at-top
      project_round_subagents,
  )

  return project_round_subagents(mock_subagents, sessions, solution_commands)


def _reconstruct_gold_messages_sent(
    commands: list[dict[str, Any]],
    mock_subagents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
  """Rebuild the ``messages_sent`` effect stream the gold trajectory produces.

  A ``messages_sent | field_semantic_match`` check is graded by handing the LLM
  judge a transcript rendered from this effect stream. Reproducing the stream
  here -- at generation time, before anything is materialized -- is what lets us
  ask "can the gold trajectory we just wrote actually satisfy the criterion we
  just wrote?" while the author is still in the retry loop and can fix it.

  Two sources feed the stream at replay time, and both are reproduced:

  1. The solver's own outgoing messages: ``message send`` / ``message broadcast``
     (target/message/channel) and ``agent delegate`` (rendered ``To <agent>:
     <instruction>`` with no channel). ``message poll`` / ``message react`` are
     deliberately absent -- they record no ``message`` field, so
     ``_format_transcript_line`` skips them and the judge never sees them.
  2. The mock sub-agents' replies (``Environment._record_subagent_message``):
     the baked ``mock_response``, plus any ``message send`` hiding in the
     sub-agent's ``state_effects``. Idempotent on ``(sender, text, channel)``
     exactly as the environment is, so one utterance is never double-counted.

  Source 2 arrives by TWO runtime routes and both are reproduced, because they
  do not always agree on the channel. Route 1 is the baked
  ``subagent_completion`` firing on ``wait``, which speaks on the sub-agent's
  own ``reply_channel``. Route 2 is the solver addressing the sub-agent
  directly, which speaks on ``reply_channel or the channel the solver used``
  (``Environment._apply_message_triggers``). When ``reply_channel`` is set the
  two coincide and the shared ``(sender, text, channel)`` dedup collapses them
  into the single line the corpus shows. When it is empty they do not, and the
  runtime emits the same utterance twice -- once channel-less, once in the
  solver's channel. Modelling only Route 1 loses the second, which is how
  ``coordinated_containment_v02`` (the one scenario whose sessions declare no
  channel, so every ``reply_channel`` is empty) was the single content mismatch
  left in the 40-round fidelity probe.

  Ordering is approximated (a sub-agent fires right after the solver addresses
  it; the rest drain at the end) because the true order is decided by
  ``background_events.trigger_time``, which the materializer computes later.

  CALLERS MUST PASS PROJECTED MOCKS. The line CONTENT is only faithful if
  ``mock_subagents`` has already been through materialization's sub-agent
  transforms (see ``_project_round_subagents_for_gate``): the authored mocks
  carry ``reply_channel = None`` and un-reconciled effect channels, so replies
  rebuilt from them render as ``From Quinn_X: ...`` where the runtime produces
  ``From Quinn_X in #prod-access-requests: ...``.

  An earlier revision of this docstring claimed 94/94 identical line sets
  against real replays. That number was wrong -- it came from a probe that fed
  this function mocks read back AFTER materialization, an input the generation-
  time gate can never have. Re-measured with the input the gate actually holds
  (``/tmp/proto_fix2.py``, line sets diffed against real v6 replays): 0/18 on
  v5 and 0/18 on v6 before the projection; 39/40 with the projection and Route
  1 only; 40/40 with Route 2 as well, with zero rounds in which this function
  predicted a line the replay did not contain. Message ORDER is a separate,
  weaker claim and is NOT fixed by any of this: 33/40, unchanged, which is why
  the gate must not grade on order.
  """
  effects: list[dict[str, Any]] = []
  seen: set[tuple[str, str, str]] = set()

  def _sub_name(sub: dict[str, Any]) -> str:
    """The name the runtime speaks and matches under (``agent_name`` first)."""
    return (
        str(sub.get("agent_name") or "").strip()
        or str(sub.get("agent_role") or "").strip()
    )

  def _add_subagent(sender: str, text: str, channel: str, target: str = "") -> None:
    text = (text or "").strip()
    if not text:
      return
    key = (sender, text, channel or "")
    if key in seen:
      return
    seen.add(key)
    effects.append({
        "sender": sender,
        "target": target or channel,
        "message": text,
        "channel": channel,
    })

  def _fire(sub: dict[str, Any], elicited_channel: str = "") -> None:
    """One sub-agent speaking. ``elicited_channel`` is set only on Route 2."""
    role = _sub_name(sub)
    if not role:
      return
    # NOT guarded on "this sub-agent already spoke": the runtime lets both
    # routes run and relies on the (sender, text, channel) dedup, so guarding
    # here would drop Route 1's channel-less copy of a reply Route 2 already
    # placed in the solver's channel.
    reply_channel = str(sub.get("reply_channel") or "") or elicited_channel
    _add_subagent(role, str(sub.get("mock_response") or ""), reply_channel)
    for effect in sub.get("state_effects") or []:
      if not isinstance(effect, str):
        continue
      for parsed in _parse_outgoing_message_effects(effect):
        _add_subagent(
            role,
            parsed["message"],
            parsed["channel"] or reply_channel,
            parsed["target"],
        )

  subs = [s for s in (mock_subagents or []) if isinstance(s, dict)]

  for entry in commands or []:
    if not isinstance(entry, dict):
      continue
    command = str(entry.get("command", "")).strip()
    if command.lower() in _TRAJECTORY_STOP_TOKENS:
      break
    for parsed in _parse_outgoing_message_effects(command):
      effects.append({
          "target": parsed["target"],
          "message": parsed["message"],
          "channel": parsed["channel"],
      })
      target = parsed["target"].lower()
      for sub in subs:
        # Substring, not equality, and every match fires -- the runtime's own
        # test (``sub_name.lower() in target.lower()``). ``async_only`` is
        # excluded there so a no-wait solver cannot elicit the decisive value
        # at virtual_time 0; excluded here for the same reason.
        name = _sub_name(sub).lower()
        if name and name in target and not sub.get("async_only"):
          _fire(sub, elicited_channel=parsed["channel"])

  for sub in subs:
    _fire(sub)
  return effects


def _render_messages_sent_transcript(effects: list[dict[str, Any]]) -> str:
  """Render ``effects`` exactly as the grader hands them to the judge.

  Delegates to the grader's own ``_format_transcript_line`` rather than
  re-implementing it, so the gate can never drift from the thing it predicts
  (the renderer decides, among other things, that a channel-only message shows
  as ``in #chan`` and that an empty ``message`` is dropped entirely).
  """
  from clasharness.evaluation.checkers.effect_checker import (
      _format_transcript_line,
  )

  lines = []
  for idx, effect in enumerate(effects, 1):
    line = _format_transcript_line(idx, effect, "message")
    if line:
      lines.append(line)
  return "\n".join(lines)


def _enforce_gold_satisfiable_message_criteria(
    scaffolds: list[dict[str, Any]],
    commands: list[dict[str, Any]],
    mock_subagents: list[dict[str, Any]],
    *,
    last_attempt: bool,
    log: Any = None,
) -> None:
  """Reject (or, last resort, drop) message criteria the gold cannot satisfy.

  A ``messages_sent | field_semantic_match`` check whose criterion the round's
  OWN gold trajectory fails is dead weight: no solver can ever pass it, so it
  contributes zero variance and zero discrimination while silently taxing every
  model's score. Measured on the v3 corpus this was 109 of 1682 replayed checks
  (6.5% of all checks; 20.8% of the messages_sent semantic ones) -- the single
  largest source of unwinnable grading in the set.

  The check is deliberately NOT relaxed to make gold pass. The criterion is the
  design intent (notify every named party, cite the report path, state the
  status), so the fix is to send the author back to complete the trajectory --
  which raises difficulty rather than lowering it, because a strong model can
  do the extra steps and a weak one cannot. Raising here lands in the same
  ``attempt_errors`` retry loop that feeds BOTH the command author and the
  criterion author, so they get the judge's own reason as feedback.

  Only a CONFIDENT judgement acts. A judge that errored (``ok`` false) leaves
  the check untouched: keeping a possibly-dead check is the status quo, whereas
  dropping checks on an infrastructure blip would silently thin the corpus.

  On the final attempt we drop only the offending check -- never the scenario,
  never the criterion text -- so a stubborn round degrades by one check instead
  of shipping something unwinnable.
  """
  targets = []
  for scaffold in scaffolds:
    checker = scaffold.get("checker")
    if not isinstance(checker, dict):
      continue
    if (
        checker.get("effect_type") != "messages_sent"
        or checker.get("condition") != "field_semantic_match"
    ):
      continue
    criterion = str((checker.get("expected") or {}).get("criterion", "")).strip()
    if criterion:
      targets.append((scaffold, checker, criterion))
  if not targets:
    return

  transcript = _render_messages_sent_transcript(
      _reconstruct_gold_messages_sent(commands, mock_subagents)
  )

  failures: list[tuple[dict[str, Any], str, str]] = []
  for scaffold, checker, criterion in targets:
    if not transcript:
      # Mirrors the grader, which fails closed on an empty transcript without
      # spending a judge call (effect_checker: "a positive semantic assertion
      # cannot be satisfied by an empty transcript").
      failures.append(
          (scaffold, criterion, "the gold trajectory sends no message at all")
      )
      continue
    try:
      from clasharness.evaluation.llm_judge import judge_majority
      verdict = judge_majority(transcript, criterion)
    except Exception as exc:  # noqa: BLE001 - judge outage must not break gen
      if log:
        log(f"    [gold-satisfiability] judge unavailable ({exc}); skipping")
      return
    if not verdict.ok:
      if log:
        log(f"    [gold-satisfiability] judge error ({verdict.error}); skipping")
      return
    if not verdict.passed:
      failures.append((scaffold, criterion, verdict.reason or "(no reason)"))

  if not failures:
    return

  if last_attempt:
    for scaffold, criterion, reason in failures:
      scaffold["checker"] = None
      if log:
        log(
            "    [gold-satisfiability] dropping unsatisfiable check after final"
            f" attempt: {criterion[:120]!r} -- {reason[:160]}"
        )
    return

  detail = "\n".join(
      f"  - criterion {criterion!r} is NOT satisfied by the gold trajectory."
      f" The judge said: {reason}"
      for _, criterion, reason in failures
  )
  raise ValueError(
      "The gold solution_commands do not satisfy their own message criteria, so"
      " no solver could ever pass them. Do NOT weaken the criteria -- instead"
      " EXTEND the gold trajectory so it actually performs what they require"
      " (message every named recipient, quote the exact path/identifier, and"
      " state the status in the message body):\n"
      + detail
      + "\nThe transcript the grader saw was:\n"
      + (transcript[:1500] or "(no messages at all)")
  )


def _steps_from_scaffolds(scaffolds: list[dict[str, Any]]) -> dict[str, Any]:
  """Merge scaffolds back into the legacy ``{"steps": [{command, checker}]}``.

  ``extra_checkers`` (additional deterministic checkers a single step contributes,
  e.g. the resolution report's ③ decision_outcome assertion alongside its decisive
  value) is preserved so round assembly can flatten them into the eval's checks.
  """
  return {
      "steps": [
          {
              "command": s.get("command"),
              "checker": s.get("checker"),
              "extra_checkers": s.get("extra_checkers"),
          }
          for s in scaffolds
      ]
  }


def _validate_generated_round_commands(
    commands_payload: dict[str, Any],
    allowed_families: list[str],
) -> None:
  """Validate the layer-3a command list (non-empty, every command in-spec)."""
  commands = commands_payload.get("commands", [])
  if not commands:
    raise ValueError("Round commands payload missing or empty 'commands'.")
  for idx, entry in enumerate(commands):
    if not isinstance(entry, dict):
      raise ValueError(f"Command entry {idx} is not a dict.")
    cmd = entry.get("command")
    if not cmd or not str(cmd).strip():
      raise ValueError(f"Command entry {idx} is missing 'command'.")
    try:
      validate_command_against_spec(str(cmd), allowed_families=allowed_families)
    except ToolSpecViolation as exc:
      raise ValueError(
          f"Invalid command at entry {idx}: '{cmd}' ({exc})"
      ) from exc


def _validate_generated_screenplay_triggers(
    cast_payload: dict[str, Any],
    triggers_payload: dict[str, Any],
) -> None:
  """Validate layer-0a conflict triggers against the finalized cast (Tier 2.1).

  Per design 5.1, every ``affected_participants`` entry across all triggers must
  be an exact ``role_name`` present in the finalized ``character_cast``. This
  turns a silent name hallucination into a caught error fed back through the
  shared screenplay retry loop. If the cast declares no role_names there is
  nothing to anchor against, so the reference check is skipped.
  """
  triggers = triggers_payload.get("narrative_conflict_triggers")
  if not isinstance(triggers, list):
    raise ValueError(
        "Screenplay triggers payload missing 'narrative_conflict_triggers' list."
    )
  outline = cast_payload.get("screenplay_outline") or {}
  cast = outline.get("character_cast") or []
  role_names = {
      c.get("role_name")
      for c in cast
      if isinstance(c, dict) and c.get("role_name")
  }
  if not role_names:
    return
  for idx, trigger in enumerate(triggers):
    if not isinstance(trigger, dict):
      raise ValueError(f"Conflict trigger at index {idx} must be a dictionary")
    participants = trigger.get("affected_participants") or []
    if isinstance(participants, str):
      participants = [participants]
    for participant in participants:
      if participant not in role_names:
        raise ValueError(
            f"Conflict trigger '{trigger.get('conflict_name', idx)}' references "
            f"affected participant '{participant}' not in the character cast "
            f"{sorted(str(r) for r in role_names)}."
        )


def _validate_generated_agent_mock_subagents(
    roster_payload: dict[str, Any],
    mocks_payload: dict[str, Any],
) -> None:
  """Validate layer-1 mock subagents against the finalized roster (Tier 2.2).

  Per design 5.2, every mock subagent ``agent_role`` must be an exact
  ``agent_role`` present in the finalized roster. This structurally guarantees
  the subagent name registry so downstream name-drift cannot dangle. If the
  roster declares no agent_roles there is nothing to anchor against, so the
  reference check is skipped.
  """
  mocks = mocks_payload.get("mock_subagents")
  if not isinstance(mocks, list):
    raise ValueError("Agent mocks payload missing 'mock_subagents' list.")
  roster = roster_payload.get("agents") or []
  roster_roles = {
      a.get("agent_role")
      for a in roster
      if isinstance(a, dict) and a.get("agent_role")
  }
  if not roster_roles:
    return
  for idx, mock in enumerate(mocks):
    if not isinstance(mock, dict):
      raise ValueError(f"Mock subagent at index {idx} must be a dictionary")
    role = mock.get("agent_role")
    if role not in roster_roles:
      raise ValueError(
          f"Mock subagent at index {idx} references agent_role '{role}' not in "
          f"the agent roster {sorted(str(r) for r in roster_roles)}."
      )


def _normalize_mock_state_effect_quoting(mocks_payload: dict[str, Any]) -> None:
  """Repair invalid POSIX quoting in mock-subagent command ``state_effects``.

  Mutates command-shaped entries (raw strings and ``{command|cmd: ...}`` dicts)
  in place so a mis-quoted apostrophe (e.g. ``'...file\\'s...'``) does not fail
  ``validate_command_against_spec`` and burn a retry / drop the scenario.
  File-write dict primitives carry no command field and are left untouched.
  """
  mocks = mocks_payload.get("mock_subagents")
  if not isinstance(mocks, list):
    return
  for mock in mocks:
    if not isinstance(mock, dict):
      continue
    effects = mock.get("state_effects")
    if isinstance(effects, str):
      if effects.strip():
        mock["state_effects"] = normalize_command_quoting(effects)
      continue
    if not isinstance(effects, list):
      continue
    for i, eff in enumerate(effects):
      if isinstance(eff, str):
        if eff.strip():
          effects[i] = normalize_command_quoting(eff)
      elif isinstance(eff, dict):
        for key in ("command", "cmd"):
          val = eff.get(key)
          if isinstance(val, str) and val.strip():
            eff[key] = normalize_command_quoting(val)


_DEFAULT_MOCK_DURATION_SECONDS = 10


def _normalize_mock_durations(
    mocks_payload: dict[str, Any],
    conflict_taxonomy: list[str] | None,
    *,
    default_seconds: int = _DEFAULT_MOCK_DURATION_SECONDS,
) -> None:
  """Clamp non-positive mock ``virtual_duration_seconds`` for delay scenarios.

  Deterministic repair (Option 1). The strict validator (check #3,
  :func:`conflict_strict_validator.validate_conflict_scenario_strict`) requires
  EVERY mock sub-agent in a ``subagent_delay`` scenario to declare a positive
  ``virtual_duration_seconds`` -- a delegated task that completes instantly has
  no delay to grade. The LLM frequently emits ``0`` for the non-decisive helper
  mocks (the ``10`` dataclass default only applies when the key is ABSENT), which
  fails strict validation AFTER materialization and silently drops the whole
  version from the dataset. Mirroring the deterministic-repair philosophy of
  :func:`_ensure_async_delegate_wait` and
  :func:`_normalize_mock_state_effect_quoting`, this clamps any ``<= 0`` (or
  missing / non-integer) duration up to ``default_seconds`` in place rather than
  burning retries or depending on sampling luck. LLM-provided POSITIVE values are
  preserved untouched, so the decisive slow sub-agent the materializer binds the
  within-window check to keeps its intended latency.

  A no-op unless the taxonomy contains ``subagent_delay`` (E6: scenarios without
  the delay mechanic are byte-for-byte unaffected).
  """
  if not conflict_taxonomy or "subagent_delay" not in conflict_taxonomy:
    return
  mocks = mocks_payload.get("mock_subagents")
  if not isinstance(mocks, list):
    return
  for mock in mocks:
    if not isinstance(mock, dict):
      continue
    raw: Any = mock.get("virtual_duration_seconds", 0)
    try:
      dur = int(raw or 0)
    except (TypeError, ValueError):
      dur = 0
    if dur <= 0:
      mock["virtual_duration_seconds"] = default_seconds


def _validate_generated_round_mocks(
    mocks_payload: dict[str, Any],
    allowed_families: list[str],
) -> None:
  """Validate the layer-3 round mock-subagent payload (Tier 1.2, design 4.2).

  Each mock needs an ``agent_role`` and ``mock_response``. Per decision 4, every
  command-shaped ``state_effects`` entry is enforced to be a valid, executable
  OpenClaw command via ``validate_command_against_spec`` (file-write dicts are
  legitimate primitives and are skipped by ``iter_mock_command_effects``).
  Failures are raised so the shared round retry loop feeds them back as
  ``error_feedback``.
  """
  mocks = mocks_payload.get("mock_subagents")
  if not isinstance(mocks, list):
    raise ValueError("Round mocks payload missing 'mock_subagents' list.")
  for idx, mock in enumerate(mocks):
    if not isinstance(mock, dict):
      raise ValueError(f"Mock subagent at index {idx} must be a dictionary")
    if not mock.get("agent_role"):
      raise ValueError(f"Mock subagent at index {idx} is missing 'agent_role'")
    if not mock.get("mock_response"):
      raise ValueError(
          f"Mock subagent '{mock.get('agent_role')}' is missing 'mock_response'"
      )
    for eff_idx, effect in enumerate(
        iter_mock_command_effects(mock.get("state_effects"))
    ):
      try:
        validate_command_against_spec(str(effect), allowed_families=allowed_families)
      except ToolSpecViolation as exc:
        raise ValueError(
            f"Mock subagent '{mock.get('agent_role')}' state_effect {eff_idx} "
            f"is not an executable command: '{effect}' ({exc})"
        ) from exc
      # G2': a subagent's message-send/broadcast effect is the reply the grader
      # reads. At runtime it is parsed by --channel and routed into
      # messages_sent (Environment._parse_message_send_effect), and at compile
      # time that channel is reconciled to a canonical session channel so the
      # reply lands somewhere gradeable/searchable. Without --channel the reply
      # can only fall back to the round's primary channel and is never persisted
      # to a session file, so require one explicitly.
      #
      # In practice this is the guard for `message broadcast`: `message send`
      # already mandates --channel at the tool-spec/skill layer (so it is
      # rejected above as "not an executable command"). `message broadcast` is
      # channel-routed by convention in this pipeline (every shipped broadcast
      # carries --channel, none carry --targets), even though the base
      # MessageSkill would otherwise route a broadcast by --targets. Enforcing
      # --channel keeps that convention intact. Raised into the round retry loop
      # as error_feedback so the model self-corrects.
      eff_s = str(effect).strip()
      low = eff_s.lower()
      if (low.startswith("message send") or low.startswith("message broadcast")) \
          and "--channel" not in eff_s:
        raise ValueError(
            f"Mock subagent '{mock.get('agent_role')}' message-send state_effect "
            f"{eff_idx} must specify a '--channel' so its reply routes to a "
            f"gradeable channel: '{effect}'"
        )


def _validate_generated_round_checkers(
    fields_payload: dict[str, Any],
) -> None:
  """Validate the layer-3b semantic fields payload shape (lenient)."""
  fields = fields_payload.get("fields")
  if not isinstance(fields, list):
    raise ValueError("Round checkers payload missing 'fields' list.")
  for idx, entry in enumerate(fields):
    if not isinstance(entry, dict):
      raise ValueError(f"Checker field entry {idx} is not a dict.")
    if "step_number" not in entry:
      raise ValueError(f"Checker field entry {idx} missing 'step_number'.")
    if ("criterion" not in entry) and ("contains" not in entry):
      raise ValueError(
          f"Checker field entry {idx} must contain 'criterion' or 'contains'."
      )


def _classify_trajectory_command(cmd: str) -> str:
  """Coarse kind for a solution/step command line used by richness checks."""
  c = str(cmd or "").strip().lower()
  if c == "done":
    return "done"
  if c.startswith("wait"):
    return "wait"
  if c.startswith("asset open") or c.startswith("asset read"):
    return "inspect"
  if c.startswith("file read"):
    return "inspect"
  if c.startswith((
      "file write", "file update", "file append", "file create",
      "file edit", "file modify",
  )):
    return "action"
  if c.startswith("message send"):
    return "action"
  if c.startswith(("agent", "delegate", "subagent", "task ")):
    return "action"
  if c.startswith("run_script"):
    return "action"
  return "other"


def _validate_generated_round_steps(
    steps_payload: dict[str, Any],
    allowed_families: list[str],
    min_steps: int = 4,
    require_final_verification: bool = False,
) -> None:
  steps = steps_payload.get("steps", [])
  if len(steps) < min_steps:
    raise ValueError(
        f"Steps trajectory too short: {len(steps)} steps. Must be at least"
        f" {min_steps} steps."
    )
  if require_final_verification:
    kinds = [_classify_trajectory_command(step.get("command")) for step in steps]
    action_idxs = [i for i, k in enumerate(kinds) if k == "action"]
    if not action_idxs:
      raise ValueError(
          "Steps trajectory has no concrete action step (file write / message"
          " send / delegation / script). It must contain at least one real action."
      )
    last_action = action_idxs[-1]
    if not any(k == "inspect" for k in kinds[last_action + 1:]):
      raise ValueError(
          "Steps trajectory is missing a final verification step: the last action"
          " must be followed by a re-read/inspection (asset open/read or file"
          " read) of the changed asset/file to confirm the fix landed."
      )
  for idx, step in enumerate(steps):
    cmd = step.get("command")
    if not cmd:
      raise ValueError(f"Step {idx} is missing 'command' field.")
    try:
      validate_command_against_spec(str(cmd), allowed_families=allowed_families)
    except ToolSpecViolation as exc:
      raise ValueError(f"Invalid command at step {idx}: '{cmd}' ({exc})") from exc
    
    checker = step.get("checker")
    if checker:
      check_type = checker.get("type")
      if check_type == "effect":
        effect_type = checker.get("effect_type")
        if effect_type not in ALLOWED_EFFECT_TYPES:
          raise ValueError(f"Forbidden effect check type at step {idx}: '{effect_type}'")
      elif check_type not in ["file_check", "command_check"]:
        raise ValueError(f"Forbidden check type at step {idx}: '{check_type}'")

      _validate_delegation_check(checker, f"Step {idx} Checker")


def _validate_generated_round_outline(
    outline_payload: dict[str, Any],
    min_steps: int = 1,
) -> None:
  if "outline" not in outline_payload or not outline_payload["outline"]:
    raise ValueError("Response missing 'outline' or it is empty")
  outline = outline_payload["outline"]
  if len(outline) < min_steps:
    raise ValueError(
        f"Outline trajectory too short: {len(outline)} steps. Must plan at least"
        f" {min_steps} steps realizing Discovery -> Inspection -> Wait -> Action"
        " -> Final Verification (with at least 2 inspection steps, at least 1"
        " action step, and a final verification step that re-reads the changed"
        " asset/file)."
    )


def _validate_generated_round_trajectory_and_evals_compiled(
    round_payload: dict[str, Any],
    allowed_families: list[str],
) -> None:
  rounds = round_payload.get("rounds", [])
  if not rounds:
    raise ValueError("Compiled payload missing 'rounds'")
  sol_cmds = rounds[0].get("solution_commands", [])
  if len(sol_cmds) < 4:
    raise ValueError(f"Compiled solution_commands too short: {len(sol_cmds)} steps. Expected at least 4.")
  for cmd in sol_cmds:
    try:
      validate_command_against_spec(str(cmd), allowed_families=allowed_families)
    except ToolSpecViolation as exc:
      raise ValueError(f"Invalid command in compiled trajectory: '{cmd}' ({exc})") from exc

  evals = round_payload.get("evals", [])
  for ev in evals:
    for check in ev.get("checks", []):
      check_type = check.get("type")
      if check_type == "effect":
        effect_type = check.get("effect_type")
        if effect_type not in ALLOWED_EFFECT_TYPES:
          raise ValueError(f"Forbidden effect check type: '{effect_type}'")
      elif check_type not in ["file_check", "command_check"]:
        raise ValueError(f"Forbidden check type: '{check_type}'")

      _validate_delegation_check(check, "Compiled Eval Check")


def _validate_generated_round(
    round_payload: dict[str, Any],
    round_id: str,
    visible_asset_ids: list[str],
    allowed_families: list[str],
) -> None:
  _validate_generated_round_narrative(round_payload, round_id, visible_asset_ids)
  _validate_generated_round_trajectory_and_evals_compiled(round_payload, allowed_families)