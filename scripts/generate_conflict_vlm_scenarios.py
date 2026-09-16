#!/usr/bin/env python3

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

"""Generate CLASHarness Conflict-VLM scenario packages and task data."""

from datetime import datetime
from pathlib import Path
import os
import sys
import threading

import argparse
import hashlib
import re
from dataclasses import replace

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from clasharness.scenario.llm_client import (
    AnthropicScenarioLLMClient,
    GeminiScenarioLLMClient,
    OpenAICompatibleScenarioLLMClient,
    VertexScenarioLLMClient,
)
from clasharness.scenario.conflict_vlm_spec import load_conflict_vlm_seed_file
from clasharness.scenario.conflict_vlm_materializer import (
    _is_negative_assertion,
    lint_assets_accessed_satisfiability,
    materialize_conflict_scenario,
)
from clasharness.scenario.conflict_strict_validator import validate_conflict_scenario_strict
from clasharness.core.task import Task, save_task

def _str2bool(value):
  """Parse an absl-style boolean literal (true/false/1/0/yes/no/t/f)."""
  return str(value).strip().lower() in ("1", "true", "t", "yes", "y")


def _parse_list(value):
  """Parse an absl DEFINE_list value: comma-separated, whitespace-stripped."""
  if not value:
    return []
  return [item.strip() for item in value.split(",")]


def _flag_names(name):
  """Return the underscore spelling plus a hyphenated alias, mirroring absl.

  absl treats '-' and '_' as interchangeable in flag names; argparse does not,
  so both spellings are registered to keep existing generation commands working.
  """
  names = [f"--{name}"]
  if "_" in name:
    names.append(f"--{name.replace('_', '-')}")
  return names


def _add_bool_flag(parser, name, default, help_text):
  """Register an absl-compatible boolean flag.

  Reproduces absl's bare (``--name`` -> True), negated (``--noname`` -> False),
  and valued (``--name=false`` / ``--name=true``) spellings, plus hyphen
  aliases for each, so existing generation commands keep working unchanged.
  """
  pos = _flag_names(name)
  neg = [spelling.replace("--", "--no", 1) for spelling in pos]
  parser.add_argument(
      *pos, dest=name, nargs="?", const=True, default=default, type=_str2bool,
      metavar="BOOL", help=help_text,
  )
  parser.add_argument(
      *neg, dest=name, action="store_false", default=argparse.SUPPRESS,
      help=argparse.SUPPRESS,
  )


# Provider-aware defaults for the reused --base_url / --api_key_env flags.
# Gemini (API-key mode) is the historical default; every other API-key provider
# reuses the same two flags but needs a different endpoint + key env, so when the
# user leaves them at the Gemini defaults we swap in that provider's defaults
# (mirroring train_eval/agents.py). The per-provider base URL / key-env constants
# are sourced from llm_judge so the generator and the field_semantic_match judge
# always agree on where a given backend lives.
from clasharness.evaluation.llm_judge import (
    _BEDROCK_GPT_API_KEY_ENV as _BEDROCK_GPT_DEFAULT_API_KEY_ENV,
    _BEDROCK_GPT_DEFAULT_BASE_URL,
    _OPENAI_API_KEY_ENV as _OPENAI_DEFAULT_API_KEY_ENV,
    _OPENAI_DEFAULT_BASE_URL,
)

_GEMINI_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_GEMINI_DEFAULT_API_KEY_ENV = "GEMINI_API_KEY"
# Claude's official API needs no base URL -- the anthropic SDK owns the host --
# so only the key env is swappable for --provider=anthropic.
_ANTHROPIC_DEFAULT_API_KEY_ENV = "ANTHROPIC_API_KEY"


def _endpoint_from_args(
    args, default_base_url: str, default_api_key_env: str
) -> tuple[str, str]:
  """Resolve (base_url, api_key_env) for a non-Gemini API-key provider.

  Every API-key provider reuses the Gemini-oriented --base_url / --api_key_env
  flags. When either is left at its Gemini default this substitutes the chosen
  provider's default, so the user only has to pass --provider; an explicit
  override is honoured as-is. Shared by the generation-client constructor and
  the judge-env sync so both hit the same endpoint/key.
  """
  base_url = (
      default_base_url
      if args.base_url == _GEMINI_DEFAULT_BASE_URL
      else args.base_url
  )
  api_key_env = (
      default_api_key_env
      if args.api_key_env == _GEMINI_DEFAULT_API_KEY_ENV
      else args.api_key_env
  )
  return base_url, api_key_env


def _bedrock_endpoint_from_args(args) -> tuple[str, str]:
  """Resolve the (base_url, api_key_env) for --provider=bedrock-gpt."""
  return _endpoint_from_args(
      args, _BEDROCK_GPT_DEFAULT_BASE_URL, _BEDROCK_GPT_DEFAULT_API_KEY_ENV
  )


def _openai_endpoint_from_args(args) -> tuple[str, str]:
  """Resolve the (base_url, api_key_env) for --provider=openai."""
  return _endpoint_from_args(
      args, _OPENAI_DEFAULT_BASE_URL, _OPENAI_DEFAULT_API_KEY_ENV
  )


def _anthropic_key_env_from_args(args) -> str:
  """Resolve the api_key_env for --provider=anthropic (no base URL to pick)."""
  return (
      _ANTHROPIC_DEFAULT_API_KEY_ENV
      if args.api_key_env == _GEMINI_DEFAULT_API_KEY_ENV
      else args.api_key_env
  )


def _build_text_client(args, client_logger):
  """Construct the narrative-text generation client for --provider.

  Extracted from ``main`` so the provider dispatch (the provider-aware
  base_url/key defaults, and the ADC paths) is unit testable without running the
  pipeline.
  """
  if args.provider == "mock":
    from tests.test_conflict_pipeline import MockConflictLLMClient
    return MockConflictLLMClient()
  if args.provider == "anthropic":
    # Claude on Anthropic's official API. Same client class as
    # vertex-anthropic; passing api_key_env is what flips it off the ADC path,
    # so the model id is a public one ('claude-sonnet-5') rather than an
    # '@'-form Vertex publisher id, and no project/location is consulted.
    return AnthropicScenarioLLMClient(
        model=args.model,
        api_key_env=_anthropic_key_env_from_args(args),
        temperature=args.temperature,
        retries=args.retries,
        request_delay_seconds=args.request_delay,
        max_output_tokens=args.max_output_tokens,
        logger=client_logger,
        verbose=True,
    )
  if args.provider == "openai":
    # GPT on OpenAI's official API. Identical wiring to bedrock-gpt below --
    # Responses API, no temperature -- against OpenAI's host with OPENAI_API_KEY,
    # and with model ids that carry no 'openai.' publisher prefix.
    openai_base_url, openai_api_key_env = _openai_endpoint_from_args(args)
    return OpenAICompatibleScenarioLLMClient(
        model=args.model,
        base_url=openai_base_url,
        api_key_env=openai_api_key_env,
        temperature=None,
        api="responses",
        retries=args.retries,
        request_delay_seconds=args.request_delay,
        max_output_tokens=args.max_output_tokens,
        logger=client_logger,
        verbose=True,
    )
  if args.provider == "vertex":
    return VertexScenarioLLMClient(
        model=args.model,
        project=args.vertex_project,
        location=args.vertex_location,
        temperature=args.temperature,
        retries=args.retries,
        request_delay_seconds=args.request_delay,
        max_output_tokens=args.max_output_tokens,
        logger=client_logger,
        verbose=True,
    )
  if args.provider == "vertex-anthropic":
    # Claude on Vertex AI (ADC, no API key). Region defaults to a Claude-serving
    # region (us-east5) inside the client when --vertex_location is unset, since
    # Gemini's us-central1 does not serve Claude.
    return AnthropicScenarioLLMClient(
        model=args.model,
        project=args.vertex_project,
        location=args.vertex_location,
        temperature=args.temperature,
        retries=args.retries,
        request_delay_seconds=args.request_delay,
        max_output_tokens=args.max_output_tokens,
        logger=client_logger,
        verbose=True,
    )
  if args.provider == "bedrock-gpt":
    # GPT on Bedrock via the OpenAI Responses API. Reuse --base_url/--api_key_env
    # but swap the Gemini defaults for Bedrock-aware ones so the user only needs
    # to pass --provider=bedrock-gpt. api="responses" forces the Responses API
    # (gpt-5-class models reject chat-completions); temperature is omitted (those
    # models 400 on an explicit temperature).
    bedrock_base_url, bedrock_api_key_env = _bedrock_endpoint_from_args(args)
    return OpenAICompatibleScenarioLLMClient(
        model=args.model,
        base_url=bedrock_base_url,
        api_key_env=bedrock_api_key_env,
        temperature=None,
        api="responses",
        retries=args.retries,
        request_delay_seconds=args.request_delay,
        max_output_tokens=args.max_output_tokens,
        logger=client_logger,
        verbose=True,
    )
  return GeminiScenarioLLMClient(
      model=args.model,
      base_url=args.base_url,
      api_key_env=args.api_key_env,
      temperature=args.temperature,
      retries=args.retries,
      request_delay_seconds=args.request_delay,
      max_output_tokens=args.max_output_tokens,
      logger=client_logger,
      verbose=True,
  )


def _resolve_under(base: Path, raw) -> Path:
  """Expand a user-supplied path argument into an absolute Path.

  Closes a silent data-routing bug: a shell does NOT expand ``~`` after ``=``,
  so ``--output_data_dir=~/Data/foo`` arrives as the literal string
  ``~/Data/foo``. ``Path("~/Data/foo")`` is a *relative* path, so every write
  lands in ``<cwd>/~/Data/foo`` -- a real directory literally named ``~``. The
  evaluator receives the same path as a separate argv word (``--task-data-dir
  ~/Data/foo``), which the shell *does* expand, so it reads ``$HOME/Data/foo``.
  Generation and evaluation then point at two different trees with no error on
  either side; this has already produced a multi-GB orphan tree next to the
  repo. Route every user-supplied path through here.

  Relative paths keep their historical meaning by way of an explicit ``base``.
  """
  path = Path(os.path.expanduser(str(raw)))
  if not path.is_absolute():
    path = base / path
  return path.resolve()


def _human_bytes(n: int) -> str:
  """Format a byte count for human-facing log lines."""
  size = float(n)
  for unit in ("B", "KB", "MB", "GB"):
    if size < 1024.0:
      return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
    size /= 1024.0
  return f"{size:.1f} TB"


def _tree_stats(path: Path):
  """Return (file_count, total_bytes) for a directory tree, best effort."""
  n_files = 0
  total = 0
  for dirpath, _dirnames, filenames in os.walk(path):
    for name in filenames:
      n_files += 1
      try:
        total += os.lstat(os.path.join(dirpath, name)).st_size
      except OSError:
        pass
  return n_files, total


def _guarded_rmtree(path: Path, label: str) -> None:
  """Delete a tree, first reporting exactly what is about to be destroyed.

  ``--clean`` used to rmtree multi-GB trees silently, so a single mistyped
  ``--output_data_dir`` / ``--output_dir`` destroyed hours of generation output
  with no trace of what had been there. None of this data is in version
  control, so there is no way back.

  Passing ``--clean`` is itself the explicit request to delete, so this does
  NOT ask for a second confirmation flag -- it logs the file count and size
  first, so the run log records what was removed and a mistyped path is
  obvious after the fact. The guard that actually prevents the expensive
  failure mode lives elsewhere: ``--clean`` is rejected outright when combined
  with ``--resume`` (see parse/validate), because wiping the tree that
  ``--resume`` exists to continue from silently turns a resume into a
  full-price regeneration.
  """
  import shutil
  if not path.exists():
    return
  n_files, total = _tree_stats(path)
  _log(f"--clean: {label} {path} holds {n_files} files ({_human_bytes(total)}).")
  _log(f"--clean: deleting {label} {path} ...")
  try:
    shutil.rmtree(path)
  except Exception as e:
    _log(f"Warning: failed to clean {path}: {e}")


# Task ids are ``<seed>_v<NN>_r<NN>``. The ``_rNN`` suffix is NOT a replicate:
# it is round N of one cumulative episode, and the rounds share a world. Round
# r10 of a scenario literally contains r01's assets (measured on the shipped
# 600-task set: visible_asset_ids grows 4 -> 13 -> 21 across r01/r05/r10), so
# scattering an episode's rounds across splits leaks the test set into training.
_TASK_ID_RE = re.compile(r"^(?P<seed>.+?)_v(?P<version>\d+)_r(?P<round>\d+)$")

SPLIT_GROUP_CHOICES = ("seed", "version", "task")


def _split_group_key(task_id: str, group_by: str) -> str:
  """Return the unit that must stay wholly inside one split.

  ``seed``    - all versions and rounds of a scenario family (default). Tests
                generalisation to an UNSEEN scenario type. Chosen as the default
                because the seed, not the version, is the observed behavioural
                unit: 9 of opus-5's 11 generation stalls fell in the
                ``payment_chargeback_dispute`` family across all three versions.
  ``version`` - one scenario instance (all its rounds). Versions are genuinely
                distinct worlds -- v01/v02/v03 of a seed have disjoint asset
                namespaces -- so this is a legitimate, weaker split that tests
                generalisation to a new instance of a SEEN scenario type.
  ``task``    - no grouping. Leaks rounds of the same episode across splits;
                offered only for debugging, never for reporting a number.
  """
  if group_by == "task":
    return task_id
  m = _TASK_ID_RE.match(task_id)
  if not m:
    # Unparseable id: keep it atomic rather than silently lumping it in with
    # everything else that fails to parse.
    return task_id
  if group_by == "version":
    return f"{m.group('seed')}_v{m.group('version')}"
  return m.group("seed")


def _assign_splits(task_ids, ratios, group_by, salt):
  """Partition ``task_ids`` into train/dev/test with no group spanning splits.

  Deterministic and reproducible: groups are ordered by ``sha256(salt:group)``
  and cut at exact largest-remainder counts, so the same (id list, ratios,
  group_by, salt) always yields the same partition regardless of generation
  order or worker count.
  """
  groups: dict[str, list[str]] = {}
  for tid in task_ids:
    groups.setdefault(_split_group_key(tid, group_by), []).append(tid)

  ordered = sorted(
      groups,
      key=lambda g: (hashlib.sha256(f"{salt}:{g}".encode()).hexdigest(), g),
  )
  n = len(ordered)
  # Largest remainder, so the counts sum to n exactly instead of drifting.
  raw = [r * n for r in ratios]
  counts = [int(x) for x in raw]
  for i in sorted(range(3), key=lambda i: raw[i] - counts[i], reverse=True):
    if sum(counts) >= n:
      break
    counts[i] += 1

  out = {}
  start = 0
  for name, take in zip(("train", "dev", "test"), counts):
    chunk = ordered[start:start + take]
    start += take
    out[name] = sorted(tid for g in chunk for tid in groups[g])
  return out, {name: c for name, c in zip(("train", "dev", "test"), counts)}, n


def _parse_ratios(value: str):
  parts = [float(x) for x in value.split(",")]
  if len(parts) != 3:
    raise argparse.ArgumentTypeError(
        f"--split-ratios needs exactly 3 comma-separated numbers, got {value!r}"
    )
  if any(x < 0 for x in parts):
    raise argparse.ArgumentTypeError(f"--split-ratios must be non-negative, got {value!r}")
  total = sum(parts)
  if total <= 0:
    raise argparse.ArgumentTypeError("--split-ratios must sum to more than 0")
  return [x / total for x in parts]


# A required check that already passes before the agent does ANYTHING is a free
# point: it inflates every model's score by the same amount and measures nothing.
# Measured on the shipped 600-task set, all 4 ``file_check`` checks were exactly
# this -- they asserted content that ``initial_state_overrides`` had already
# seeded into the file. Nothing upstream catches it: the satisfiability lint asks
# "can this ever pass?", which is the opposite failure.
_ZERO_ACTION_JUDGE_ENV = (
    "LLM_JUDGE_PROVIDER",
    "LLM_JUDGE_MODEL",
    "GEMINI_API_KEY",
    "ANTHROPIC_API_KEY",
    "VERTEX_PROJECT",
)


def audit_zero_action_discrimination(task_ids, data_dir):
  """Replay every task with an EMPTY trajectory and report what still passes.

  Returns ``(vacuous, non_discriminating, unaudited, n_ok)`` where:
    * ``vacuous`` - (task_id, check_name, message) for each REQUIRED check that
      passes with zero agent actions.
    * ``non_discriminating`` - task_ids whose whole evaluation succeeds at zero
      actions (the severe case: the task is free).
    * ``unaudited`` - count of checks whose grader could not run (``ok=False``),
      so their zero-action verdict is unknown rather than clean.

  The LLM judge is deliberately DISABLED for the audit: a semantic check would
  otherwise bill k model calls per task just to confirm an empty transcript, and
  a disabled judge fails closed (``ok=False``), which lands in ``unaudited``
  rather than being silently miscounted as discriminating.

  Negative assertions (``not_exists`` / ``not_contains``) are exempt from
  ``vacuous``, matching the compile-time demotion pass
  (``conflict_vlm_materializer._demote_vacuous_checks``). Such a check says "do
  not break this"; it is SUPPOSED to hold before the agent acts, and its
  discriminating power is whether an acting agent violates it. Without the
  exemption the two passes disagree about the same check -- the materializer
  keeps it, this gate then fails the run over it. ``CheckResult`` carries no
  ``condition``, so the verdict is read off the authoring spec, which is
  positionally aligned with ``result.details``
  (``environment`` builds the evaluator from this very list and
  ``EvaluatorComb`` preserves order).

  The exemption is deliberately NOT applied to ``result.success`` /
  ``non_discriminating``. The two are different questions: a single negative
  assertion holding at t=0 is expected, but a task whose ENTIRE required set is
  negative assertions really is free -- an agent that does nothing wins it --
  and that is worth failing the run over.
  """
  import os as _os
  from clasharness.core.environment import CLASHarnessEnv

  saved = {k: _os.environ.pop(k, None) for k in _ZERO_ACTION_JUDGE_ENV}
  vacuous, non_disc, unaudited, n_ok = [], [], 0, 0
  try:
    for tid in task_ids:
      try:
        with CLASHarnessEnv(task_id=tid, task_data_dir=data_dir, backend="multi") as env:
          env.reset()
          result = env.evaluate()
          # Read inside the ``with``: the env is torn down on exit. Same list the
          # evaluator was built from, so ``specs[i]`` describes ``details[i]``.
          # Every hop is a ``getattr`` default: an env that does not expose the
          # spec must degrade to "no exemptions" (the loud direction), never to
          # an exception -- that would land in the handler below and drop the
          # task from the audit entirely, silently blinding the gate.
          _gt = getattr(getattr(env, "_task", None), "ground_truth", None)
          specs = list(getattr(_gt, "evaluation_checks", None) or [])
      except Exception as exc:  # noqa: BLE001 - audit must never abort the run
        _log(f"  ! zero-action audit could not replay {tid}: {exc}")
        continue
      n_ok += 1
      if result.success:
        non_disc.append(tid)
      for i, check in enumerate(result.details):
        if getattr(check, "role", "required") != "required":
          continue
        if not getattr(check, "ok", True):
          unaudited += 1
        elif check.passed:
          # A missing/short spec list yields None, which reads as "not negative"
          # -- the loud direction, so a desync can never silence the gate.
          if _is_negative_assertion(specs[i] if i < len(specs) else None):
            continue
          vacuous.append((tid, check.name, str(getattr(check, "message", ""))[:160]))
  finally:
    for k, v in saved.items():
      if v is not None:
        _os.environ[k] = v
  return vacuous, non_disc, unaudited, n_ok


# --- exact-arm fairness audit ------------------------------------------------
#
# WHY THIS EXISTS. A corpus once shipped with 178 of its 400 exact-graded
# rounds (44.5%) turning on a compound phrase put through a raw substring
# match -- a check no correct-but-differently-worded report can pass. Nothing
# in the run was looking at that number, so it went out and was only found by
# hand afterwards. This audit is the thing that looks.
#
# IT REPORTS AND NEVER REPAIRS. It reads the shipped specs, prints what it
# found, and returns. It raises nothing and rewrites nothing: the fix for an
# unfair value belongs at the source (the S1 prompt), and a post-hoc rewrite
# here would silently paper over a prompt that had stopped working.
#
# ON THE TWO CRITERIA. `_is_compound_decisive_value` is the pipeline's ONE
# decision rule and this audit adds no second one -- it never re-decides any
# value's grading arm. But it cannot measure with that rule either: everything
# the rule flags was already downgraded out of the exact arm, so asking it
# "how much compound survived?" returns 0 BY CONSTRUCTION. A guard's blind
# spot is not observable through the guard. So the audit measures with a
# deliberately BROAD instrument and reports the difference (broad-flagged AND
# NOT rule-flagged) as the residual -- that difference IS the blind spot.
#
# INSTRUMENT LIMITS, stated because the numbers are only as good as these:
#   * the broad probe is a heuristic. It excludes the bare comma on purpose:
#     treating ',' as a clause separator misreports thousands separators
#     ('12,450 ms'), which inflated an earlier reading of this same metric
#     from 4 hits to 12. It can still miss shapes it has no separator for.
#   * `_FAIRNESS_STATUS_WORDS` is a fixed word list, so the species-c number
#     is a DRIFT DETECTOR across runs on one instrument, not an absolute
#     census. It reads higher than a narrower list would; compare it only
#     against a previous run measured the same way.
#   * the exact arm is read off `evals[].checks[]`, because the shipped spec
#     does not persist `decisive_value_citable`. Granularity is the ROUND.
_FAIRNESS_SEP_RE = re.compile(r'[;()\[\]]|\s[-|/]\s|:\s')
_FAIRNESS_ID_RE = re.compile(
    r'\b[A-Z][A-Z0-9]+[-_][A-Z0-9]*\d[A-Z0-9]*\b|\b[A-Z]{2,}\d{3,}\b')
_FAIRNESS_STATUS_WORDS = frozenset({
    "PASSED", "PASS", "FAILED", "FAIL", "BLOCKED", "PENDING", "APPROVED",
    "REJECTED", "CONTAINED", "VERIFIED", "CONFIRMED", "COMPLETE", "COMPLETED",
    "LOCKED", "UNLOCKED", "STALE", "ACTIVE", "INACTIVE", "ENABLED", "DISABLED",
    "SUCCESS", "ERROR", "CRITICAL", "WARNING", "OPEN", "CLOSED", "READY",
    "DENIED", "GRANTED", "VALID", "INVALID", "MISMATCH", "DETECTED", "CLEARED",
    "ESCALATED", "RESOLVED", "UNRESOLVED", "DEGRADED", "HEALTHY", "UNHEALTHY",
    "TIMEOUT", "ABORTED", "CANCELLED", "SUSPENDED", "QUARANTINED", "BREACHED",
    "EXPIRED", "HALTED", "ONLINE", "OFFLINE", "SYNCED", "UNSYNCED", "DRIFT",
    "OVERRIDE", "ROLLBACK", "REVOKED", "ACKNOWLEDGED", "UNACKNOWLEDGED",
    "FULLY", "HOLD", "WAIT", "QUEUED", "RUNNING", "STOPPED", "STARTED",
    "IDLE", "BUSY", "SAFE", "UNSAFE",
})

# Coupled BY VALUE to the pipeline's downgrade log line. The pipeline keeps no
# run-level downgrade counter (the count is by design recoverable from the log
# only), so the rate is recovered by counting this marker. A silent 0 would be
# the worst failure mode here, so a test pins this string against the one the
# pipeline actually emits -- reword either and that test goes red.
_CITABLE_DOWNGRADE_MARKER = "citable DOWNGRADED:"

# Coupled BY VALUE to the pipeline's gold-satisfiability ABORT log lines, the
# same way as the downgrade marker above.
#
# That gate FAILS OPEN by design: a judge that errors leaves the round's checks
# untouched, because dropping checks on an infrastructure blip would silently
# thin the corpus. The cost of that (correct) choice is that a broken judge is
# INVISIBLE -- the sonnet-5 run aborted the gate 208 times (a reasoning model's
# thinking block crashed the judge's response parser) and nothing in the run
# summary said so. Counting the marker here does not change the fail-open
# semantics; it just makes the outage impossible to miss.
#
# Both wordings are matched: the gate logs "judge unavailable" when the call
# raised and "judge error" when it returned ok=False.
#
# The markers are the full phrase and not the "[gold-satisfiability]" prefix
# because the gate logs a THIRD line under that prefix -- "dropping unsatisfiable
# check after final attempt" -- which means the gate WORKED. Counting it would
# invert the metric, reporting an outage exactly when the gate did its job.
#
# The trailing " (" is narrower still: it pins the parenthesised-cause shape
# these two lines actually emit, so a future line that merely mentions a judge
# error ("judge error rate: 5%") is not swept in. It is not what excludes the
# dropping line -- that line contains neither wording.
_GOLD_SAT_ABORT_MARKERS = (
    "[gold-satisfiability] judge error (",
    "[gold-satisfiability] judge unavailable (",
)


def _fairness_exact_arm(spec):
  """Return ``(values_by_round, exact_arm_by_round)`` for one shipped spec.

  A round is in the EXACT arm when some check grades it by a raw substring
  match on written file content carrying that round's own resolved value --
  ``files_created`` AND ``field_contains`` AND ``field == "content"`` AND the
  check's value equals ``resolved_value``.

  All four are kept because every shape they exclude grades the same value MORE
  forgivingly, so admitting one would dilute every rate computed off the arm:
  ``assets_accessed`` is also ``field_contains`` but matches on ``field ==
  "id"`` (1,914 of those against 554 real exact-arm checks over the last two
  corpora), and ``messages_sent`` is also ``field_contains`` but is matched by
  ``effect_checker._effect_field_contains`` as a tokenized, order-tolerant,
  stop-word-stripped subsequence rather than as a substring -- a compound value
  survives that check, so it is not part of the unfair arm.

  DO NOT validate a conjunct by deleting it. Measured over the last two
  corpora, dropping any ONE of the four leaves the arm at exactly 412 rounds,
  because every excluded shape is over-determined: the 1,811 valued-round
  ``assets_accessed`` checks fail ``effect_type`` AND ``field`` AND ``value``,
  the 2,197 ``files_created`` semantic checks fail ``condition`` AND ``value``,
  and everything else fails all four. No single conjunct is individually
  load-bearing TODAY; the redundancy is exactly what keeps the arm from
  silently widening when a new check shape appears.

  One round can carry several checks, so the arm is keyed by round to keep the
  granularity at the round.
  """
  by_round = {}
  for rnd in spec.get("rounds") or []:
    value = str(rnd.get("resolved_value") or "").strip()
    if value:
      by_round[rnd.get("round_id")] = value
  exact = {}
  for ev in spec.get("evals") or []:
    round_id = ev.get("round_id")
    value = by_round.get(round_id)
    if not value:
      continue
    for check in ev.get("checks") or []:
      expected = check.get("expected") or {}
      if (check.get("effect_type") == "files_created"
          and check.get("condition") == "field_contains"
          and expected.get("field") == "content"
          and str(expected.get("value") or "").casefold() == value.casefold()):
        exact[round_id] = value
  return by_round, exact


def _fairness_is_broad(value):
  """The measurement instrument: does the string LOOK like it carries a clause?

  Deliberately looser than the pipeline's decision rule -- that is the point,
  see the note above. The bare comma is excluded so a thousands separator is
  not read as a clause boundary.
  """
  return len(value.split()) >= 4 or bool(_FAIRNESS_SEP_RE.search(value))


# Every payload field an asset can carry a reading in. Over-inclusive on
# purpose: this set feeds the REACHABLE side of the sourceability test, so a
# field missing here would invent an unwinnable round that is actually fine.
# `id` is added separately -- a value can be the asset's own identifier.
_FAIRNESS_ASSET_TEXT_FIELDS = ("ocr_text", "caption", "description",
                               "chart_rows", "render_code", "regions", "prompt")


def _fairness_asset_blob(asset, _json):
  """Everything about one asset a solver could possibly read a value out of."""
  return str(asset.get("id") or "") + "\n" + _json.dumps(
      {k: asset.get(k) for k in _FAIRNESS_ASSET_TEXT_FIELDS},
      ensure_ascii=False)


def _fairness_unwinnable(scenario_dir, exact, _json):
  """Which exact-arm values cannot be READ from what their round exposes?

  A different invariant from gold satisfiability, and not reducible to it. The
  gold-satisfiability gate replays the round's own reference trajectory and asks
  whether it passes the check -- but gold is AUTHORED from ``resolved_value``,
  so it writes the value whether or not any asset the round shows contains it.
  Two rounds on the last corpus prove the gap concretely: their gold
  ``solution_commands`` open no asset holding the value, yet gold satisfies the
  check. Gold knows the answer; a solver has to find it.

  So reachability is computed from the SOLVER's surface instead, and computed as
  a deliberate OVER-estimate: the entire workspace, every session, every update,
  plus the id and full payload of each asset in that round's ``visible_refs``.
  A value absent from that superset is obtainable by no route at all, so a
  ``field_contains`` check demanding it is unwinnable rather than merely hard.
  Erring this way means the audit under-reports; a false accusation here would
  send an author chasing a defect that is not there.

  Note ``available_from_round`` is deliberately NOT used as the visibility test.
  That field says when an asset ENTERS the scenario, not what a given round puts
  in front of the solver -- reading it as visibility is precisely the confusion
  that let this defect ship. ``visible_refs`` is the authority.

  Returns ``(rows, checked)`` where rows are ``(round_id, value, tier, holders)``
  and ``checked`` is True only when both companion files were readable, so a
  clean result can be told apart from an unperformed test.
  """
  scen_path = scenario_dir / "scenario.json"
  manifest_path = scenario_dir / "asset_manifest.json"
  if not (scen_path.is_file() and manifest_path.is_file()):
    return [], False
  try:
    scen = _json.loads(scen_path.read_text())
    assets = {a["id"]: a
              for a in (_json.loads(manifest_path.read_text()).get("assets")
                        or []) if a.get("id")}
  except Exception as exc:  # noqa: BLE001 - audit must never abort the run
    _log(f"  ! sourceability audit could not read {scenario_dir.name}: {exc}")
    return [], False

  return fairness_unwinnable_rows(scen, assets, exact, _json), True


def fairness_unwinnable_rows(scen, assets, exact, _json):
  """The reachability test itself, over ALREADY-PARSED scenario data.

  Split out of the file-reading wrapper above for one reason: the same question
  has to be asked of a scenario that is not the one on disk. Re-scoring the
  eval runs collected before the visibility repair means asking "was this round
  winnable under its OLD ``visible_refs``", and a scenario.json here runs to
  ~108 MB, so writing a reverted copy per scenario just to re-read it is
  gigabytes of pointless I/O.

  It exists so the offline tool can REUSE this test rather than restate it. A
  second implementation of "was this round winnable" would be free to drift
  from this one, and the two would then disagree about which eval rows to drop
  -- silently, and in the direction nobody audits.

  ``scen`` is the parsed scenario package, ``assets`` an id -> asset dict.
  """
  ambient = "\n".join(_json.dumps(scen.get(k), ensure_ascii=False)
                      for k in ("workspace", "sessions", "updates"))
  everywhere = _json.dumps(scen, ensure_ascii=False) + "\n" + "\n".join(
      _fairness_asset_blob(a, _json) for a in assets.values())
  rounds = {r.get("round_id"): r for r in (scen.get("rounds") or [])}

  rows = []
  for round_id, value in exact.items():
    rnd = rounds.get(round_id)
    if rnd is None:
      continue
    visible = [v for v in (rnd.get("visible_refs") or []) if v in assets]
    reachable = ambient + "\n" + "\n".join(
        _fairness_asset_blob(assets[v], _json) for v in visible)
    if value in reachable:
      continue
    holders = sorted(aid for aid, a in assets.items()
                     if value in _fairness_asset_blob(a, _json))
    if holders:
      tier = "invisible-asset"      # the reading exists, this round hides it
    elif value in everywhere:
      tier = "elsewhere"            # somewhere in the package, but in no asset
    else:
      tier = "unsourced"            # nowhere at all -- the value was invented
    rows.append((round_id, value, tier, holders))
  return rows


def audit_exact_arm_fairness(scenario_dirs, log_path=None):
  """Measure how gradable the shipped exact arm is. Report-only.

  Returns a dict of counts and rates plus ``residual``, the list of
  ``(scenario, round_id, value)`` triples that are exact-graded and look
  compound to the broad instrument yet were not caught by the pipeline rule.
  Those are the rounds a correct report can still fail on wording alone.
  """
  from clasharness.scenario.conflict_llm_pipeline import (
      _is_compound_decisive_value,
  )
  import json as _json
  import statistics as _statistics

  all_values, exact_values, residual = [], [], []
  unwinnable, n_sourceable_specs = [], 0
  reuse, n_specs = 0, 0
  for scenario_dir in scenario_dirs:
    spec_path = scenario_dir / f"{scenario_dir.name}_spec.json"
    if not spec_path.is_file():
      continue
    try:
      spec = _json.loads(spec_path.read_text())
    except Exception as exc:  # noqa: BLE001 - audit must never abort the run
      _log(f"  ! fairness audit could not read {spec_path.name}: {exc}")
      continue
    n_specs += 1
    by_round, exact = _fairness_exact_arm(spec)
    seen = set()
    for value in by_round.values():
      all_values.append(value)
      key = value.casefold()
      if key in seen:
        reuse += 1
      seen.add(key)
    for round_id, value in exact.items():
      exact_values.append(value)
      if _fairness_is_broad(value) and not _is_compound_decisive_value(value):
        residual.append((scenario_dir.name, round_id, value))
    # Sourceability runs off the SAME arm rather than a re-derived one, so it
    # can never disagree with the residual audit about what is exact-graded.
    _rows, _checked = _fairness_unwinnable(scenario_dir, exact, _json)
    n_sourceable_specs += 1 if _checked else 0
    for _rid, _v, _tier, _holders in _rows:
      unwinnable.append((scenario_dir.name, _rid, _v, _tier, _holders))

  n_all, n_exact = len(all_values), len(exact_values)
  lengths = sorted(len(v) for v in exact_values)
  coarse = [v for v in all_values
            if any(t.upper() in _FAIRNESS_STATUS_WORDS
                   for t in re.split(r'[^A-Za-z]+', v) if t)]
  welded = [v for v in coarse if _FAIRNESS_ID_RE.search(v)]

  # ONE read, several counts: these rates are recoverable from the run log only,
  # and a full generation log runs to several MB -- re-reading it per metric is
  # pure waste. ``None`` means "could not look", which the banner renders
  # differently from a genuine 0 so an unreadable log never masquerades as clean.
  downgrades = None
  gold_sat_aborts = None
  if log_path:
    try:
      log_text = Path(log_path).read_text(errors="replace")
    except Exception:  # noqa: BLE001 - a missing log must not abort the audit
      log_text = None
    if log_text is not None:
      downgrades = log_text.count(_CITABLE_DOWNGRADE_MARKER)
      gold_sat_aborts = sum(log_text.count(m) for m in _GOLD_SAT_ABORT_MARKERS)

  def pct(n, d):
    return 100.0 * n / d if d else 0.0

  return {
      "specs": n_specs,
      "rounds_with_value": n_all,
      "exact_arm": n_exact,
      "exact_arm_pct": pct(n_exact, n_all),
      "exact_len_median": _statistics.median(lengths) if lengths else 0,
      "exact_len_max": max(lengths) if lengths else 0,
      "exact_ge35": sum(1 for l in lengths if l >= 35),
      "exact_ge35_pct": pct(sum(1 for l in lengths if l >= 35), n_exact),
      "residual": residual,
      "residual_pct": pct(len(residual), n_exact),
      "unwinnable": unwinnable,
      "unwinnable_pct": pct(len(unwinnable), n_exact),
      # Reported so an empty `unwinnable` cannot be misread. Zero findings over
      # zero checkable scenarios is a test that did not run, not a clean bill.
      "sourceability_specs": n_sourceable_specs,
      "unique_value_pct": pct(len({v.casefold() for v in all_values}), n_all),
      "reuse_pct": pct(reuse, n_all),
      "status_phrase_pct": pct(len(coarse) - len(welded), n_all),
      "downgrades": downgrades,
      # How many times the gold-satisfiability gate gave up on a round because
      # the judge failed. The gate fails open, so this is the ONLY signal that
      # a fairness gate was offline for part of the run.
      "gold_sat_aborts": gold_sat_aborts,
  }


# Outcomes of the unwinnable-round audit, kept distinct because "found nothing"
# and "could not look" must never collapse into the same pass.
UNWINNABLE_AUDIT_OK = "ok"                            # ran to completion
UNWINNABLE_AUDIT_FAILED = "failed"                    # raised; result unknown
UNWINNABLE_AUDIT_NOTHING_SHIPPED = "nothing-shipped"  # no corpus to audit

# How many findings the exception text carries. The full list is already in the
# log; this is for the caller that only ever sees the traceback.
_GATE_PREVIEW = 10


def shipping_gate_failure(
    *,
    vacuous,
    non_discriminating,
    unwinnable,
    audit_status,
    untested_specs=0,
    fail_on_unwinnable=True,
):
  """Should this run exit non-zero? Returns the reason, or None to ship.

  Split out of ``main()`` and kept pure so the gate can be exercised directly:
  the condition it guards costs an eight-hour generation run to reproduce, and
  a gate that has only ever been observed passing is not a gate.

  Every reason is reported together rather than on a first-match raise. Fixing
  one defect and rediscovering the next one a full run later is the failure mode
  this shape exists to avoid.

  ``audit_status`` is one of the ``UNWINNABLE_AUDIT_*`` constants.
  ``UNWINNABLE_AUDIT_FAILED`` blocks too: 27 unwinnable rounds shipped precisely
  because nothing distinguished "clean" from "never checked", so a run that
  could not perform the check does not get credit for passing it.
  """
  blocks = []

  if vacuous or non_discriminating:
    blocks.append(
        f"Zero-action audit failed: {len(vacuous)} required check(s) pass "
        f"with no agent actions and {len(non_discriminating)} task(s) succeed "
        f"outright. These are free points -- every model scores them equally, "
        f"so they inflate the leaderboard without measuring anything.\n"
        f"The generated data was NOT deleted; see the VACUOUS / "
        f"NON-DISCRIMINATING lines above. Re-run with "
        f"--noaudit_discrimination to ship anyway."
    )

  if not fail_on_unwinnable:
    return "\n\n".join(blocks) or None

  if unwinnable:
    preview = "\n".join(
        f"  {_s} {_rid} [{_tier}]: {_v!r}"
        for _s, _rid, _v, _tier, _h in unwinnable[:_GATE_PREVIEW]
    )
    if len(unwinnable) > _GATE_PREVIEW:
      preview += f"\n  ... and {len(unwinnable) - _GATE_PREVIEW} more"
    blocks.append(
        f"Unwinnable-round audit failed: {len(unwinnable)} exact-graded "
        f"round(s) demand a value that is readable from NOTHING the round "
        f"shows the solver. Gold still passes them -- gold is authored from "
        f"resolved_value, not read off an asset -- so no other gate catches "
        f"this. Each one is a `required` check, so it caps its task below 1.0 "
        f"for every model forever and measures the corpus defect rather than "
        f"the model.\n"
        f"{preview}\n"
        f"The generated data was NOT deleted; see the UNWINNABLE lines above. "
        f"Re-run with --nofail_on_unwinnable to ship anyway."
    )

  if audit_status == UNWINNABLE_AUDIT_FAILED:
    blocks.append(
        "Unwinnable-round audit did not complete, so this corpus is "
        "UNVERIFIED, not clean. See the WARNING line above for the "
        "exception. Re-run with --nofail_on_unwinnable to ship anyway."
    )
  elif audit_status == UNWINNABLE_AUDIT_OK and untested_specs:
    blocks.append(
        f"Unwinnable-round audit could not test {untested_specs} shipped "
        f"scenario(s): asset_manifest.json was missing or unreadable. That "
        f"file is written before scenario.json on every successful "
        f"materialization, so its absence next to a finished scenario is "
        f"itself a defect -- and those scenarios were NOT checked. Re-run "
        f"with --nofail_on_unwinnable to ship anyway."
    )

  return "\n\n".join(blocks) or None


def parse_args():
  parser = argparse.ArgumentParser(
      description="Generate CLASHarness Conflict-VLM scenario packages and task data."
  )

  def add(name, **kwargs):
    parser.add_argument(*_flag_names(name), dest=name, **kwargs)

  add("provider", default="vertex",
      choices=["mock", "gemini", "anthropic", "openai", "vertex",
               "vertex-anthropic", "bedrock-gpt"],
      help="LLM provider for narrative text. The first four need only an API "
      "key and no cloud account: 'mock' is fully offline; 'gemini' is the "
      "official Gemini API (GEMINI_API_KEY); 'anthropic' is Claude's official "
      "API (ANTHROPIC_API_KEY); 'openai' is GPT's official Responses API "
      "(OPENAI_API_KEY). The last three are the cloud-hosted routes the "
      "released corpus was generated through, and need cloud credentials: "
      "'vertex' is Gemini on Vertex AI; "
      "'vertex-anthropic' is Claude on Vertex AI (ADC, model must be an "
      "'@'-form publisher id e.g. claude-sonnet-4@20250514, served in a Claude "
      "region such as us-east5); 'bedrock-gpt' is GPT on Bedrock via the OpenAI "
      "Responses API (reuses --base_url/--api_key_env with Bedrock-aware "
      "defaults; model carries an 'openai.' prefix e.g. openai.gpt-5.5).")
  add("model", default="gemini-1.5-pro",
      help="Text model for the chosen --provider. Gemini id for gemini/vertex, "
      "public Claude id (e.g. claude-sonnet-5) for anthropic, bare GPT id (e.g. "
      "gpt-5.5) for openai, '@'-form Claude publisher id for vertex-anthropic, "
      "'openai.'-prefixed id for bedrock-gpt.")
  add("vertex_project", default=None,
      help="GCP project ID for Vertex AI generator calls.")
  add("vertex_location", default=None,
      help="GCP location for Vertex AI generator calls (defaults to environment variables or 'us-central1').")
  add("image_vertex_location", default=None,
      help="GCP location for the Vertex AI IMAGE provider (--image_provider=vertex) only. Defaults to --vertex_location when unset. Set this when the image model (e.g. gemini-3-pro-image) is only served in a different region than the text model.")
  add("base_url", default=_GEMINI_DEFAULT_BASE_URL,
      help="API endpoint base URL for API-key HTTP providers (gemini, openai, "
      "bedrock-gpt). Left at the Gemini default it is auto-swapped to the "
      f"chosen provider's default ({_OPENAI_DEFAULT_BASE_URL} for openai, "
      f"{_BEDROCK_GPT_DEFAULT_BASE_URL} for bedrock-gpt). Ignored by "
      "anthropic (the SDK owns the host) and by vertex / vertex-anthropic (ADC).")
  add("api_key_env", default=_GEMINI_DEFAULT_API_KEY_ENV,
      help="Environment variable holding the text API key for API-key "
      "providers (gemini, anthropic, openai, bedrock-gpt). Left at the Gemini "
      "default it is auto-swapped to the chosen provider's default "
      f"({_ANTHROPIC_DEFAULT_API_KEY_ENV}, {_OPENAI_DEFAULT_API_KEY_ENV}, or "
      f"{_BEDROCK_GPT_DEFAULT_API_KEY_ENV}). Ignored by vertex / "
      "vertex-anthropic (ADC).")
  add("seed_dir", default="clasharness/data/scenario_seeds/conflict_vlm_v1",
      help="Directory containing VLM seed files.")
  add("output_dir", default="clasharness/data/scenarios/conflict_vlm_v1",
      help="Scenario output directory.")
  add("output_data_dir", default="/tmp/clasharness-conflict-vlm-data",
      help="Task data output directory.")
  add("image_provider", default="google",
      choices=["mock", "google", "vertex"],
      help="Visual assets image generator. 'vertex' uses Vertex AI (VERTEX_PROJECT/VERTEX_LOCATION) like --provider=vertex.")
  add("google_project", default=None,
      help="GCP project ID for Vertex AI image calls.")
  add("chart_provider", default="template",
      choices=["mock", "template", "code", "google"],
      help="Chart rendering strategy.")
  add("google_model", default="gemini-3.1-flash-image",
      help="VLM model to refine definitions.")
  add("google_api_key_env", default="GEMINI_API_KEY",
      help="API key environment variable for VLM model.")
  add("target_rounds", type=int, default=None,
      help="Override target rounds count.")
  add("seed_name", type=_parse_list, default=None,
      help="Run only the seed(s) with these names. Comma-separated to run multiple, e.g. --seed_name=a,b.")
  add("retries", type=int, default=5,
      help="Number of retries on API errors.")
  add("auditor_max_iterations", type=int, default=3,
      help="Max iterations for the inconsistency auditor repair loop.")
  _add_bool_flag(parser, "vision_audit", True,
      "Audit each DECISIVE image with the vision model to confirm its deciding token renders legibly (bounded regeneration, then auto-downgrade to corroborating and relax the bound check if still illegible). On by default for real image providers; --novision_audit skips it to save vision API calls (already a no-op for --image_provider=mock).")
  add("phase_expand_attempts", type=int, default=None,
      help="Retry budget for the abort-prone phase-expansion step. Defaults to max(retries-derived floor, 6). A single phase failure degrades to a salvage outline instead of discarding the scenario.")
  add("request_delay", type=float, default=0.0,
      help="Delay in seconds between API calls.")
  add("max_output_tokens", type=int, default=16384,
      help="Max output tokens for the LLM generator client.")
  _add_bool_flag(parser, "resume", False,
      "Resume generation from the existing spec and cached assets on disk.")
  add("version_retries", type=int, default=1,
      help="Extra whole-version attempts after a version fails. Each retry resumes from the pipeline checkpoint (kept outside the deleted scenario dir), so it re-runs only the stage that failed, not the whole scenario. Retries change nothing about validation -- every gate still applies. 0 restores the historical fail-once behaviour.")
  add("style_seed", type=int, default=None,
      help="Integer seed to pin visual styles and themes. Defaults to a random seed on every run.")
  add("num_render_workers", type=int, default=None,
      help="Number of worker threads to render assets in parallel. If not set or <=1, serial execution is used.")
  add("num_generation_workers", type=int, default=None,
      help="Number of worker threads to generate scenario specs in parallel. If not set or <=1, serial execution is used.")
  add("temperature", type=float, default=0.2,
      help="Model generation temperature.")
  add("num_versions", type=int, default=1,
      help="Number of scenario versions/variations to generate from each seed.")
  add("num_workers", type=int, default=1,
      help="Number of scenario versions/seeds to generate in parallel. If not set or <=1, serial execution is used.")
  _add_bool_flag(parser, "clean", False,
      "Delete all compiled tasks, datasets, and trajectories in the output data directory before starting.")
  _add_bool_flag(parser, "audit_discrimination", True,
      "After generating, replay every new task with an EMPTY trajectory and report REQUIRED checks that still pass (free points) and tasks that succeed outright (non-discriminating). Exits non-zero if any are found; the generated data is NOT deleted. --noaudit_discrimination skips it.")
  _add_bool_flag(parser, "fail_on_unwinnable", True,
      "Exit non-zero if any exact-graded round demands a value that is readable from NOTHING the round shows the solver. Such a round caps its task below 1.0 for every model forever, and gold still passes it (gold is authored from resolved_value), so no other gate catches it. The generated data is NOT deleted. --nofail_on_unwinnable ships anyway.")
  add("split_ratios", type=_parse_ratios, default="0.7,0.15,0.15",
      help="train,dev,test proportions (normalised to sum to 1). Applied to GROUPS, not to individual tasks.")
  add("split_group_by", default="seed", choices=list(SPLIT_GROUP_CHOICES),
      help="Unit kept wholly inside one split. 'seed' (default) holds out entire scenario families; 'version' holds out scenario instances; 'task' does not group and LEAKS rounds of the same episode across splits.")
  add("split_salt", default="clasharness-conflict-vlm",
      help="Salt for the deterministic group-to-split hash. Change it to deliberately reshuffle the split.")
  add("log_file", default=None,
      help="Mirror all _log() output to this file (append mode), so per-attempt failure reasons survive terminal scrollback and parallel runs. If unset, defaults to <output_data_dir>/logs/generation_<timestamp>.log so a run is always persisted next to its data.")

  return parser.parse_args()




_LOG_LOCK = threading.Lock()
_LOG_FH = None
_LOG_FILE = None  # Set from args.log_file in main().


def _log(message: str) -> None:
  """Emit one line to stdout and (if configured) to the run log.

  BOTH sinks are written under the ONE lock, and the stdout write is the reason
  the lock has to cover them.

  ``print(line, flush=True)`` is not one write. It is ``write(line)`` then
  ``write("\\n")`` then ``flush()`` -- three calls, with two scheduling points
  inside them, which is a fact about ``print`` and not about any particular
  stream. Runs use ``--num_workers 20``, so twenty threads hit those scheduling
  points constantly and another thread's text lands between a line and its own
  newline. Measured on a stream that yields inside ``write`` (which makes the
  interleave happen every time instead of occasionally -- it manufactures the
  SCHEDULING, not the splice): twenty logged lines came out as eleven, five of
  them two or three records welded together, and 20/20 clean once locked.

  Only the file mirror used to be locked, which is why the run log stayed
  trustworthy while the terminal did not.

  Holding one lock across both sinks also keeps them in the SAME ORDER, so a
  scrollback and a run log can be compared line for line instead of only
  line-set for line-set.
  """
  stamp = datetime.now().strftime("%H:%M:%S")
  line = f"[{stamp}] {message}"
  path = _LOG_FILE
  global _LOG_FH
  with _LOG_LOCK:
    print(line, flush=True)
    # Optionally mirror to a persistent log file (handle opened lazily).
    if path:
      if _LOG_FH is None:
        _LOG_FH = open(path, "a", encoding="utf-8")
      _LOG_FH.write(line + "\n")
      _LOG_FH.flush()


def main() -> None:
  global _LOG_FILE
  args = parse_args()
  # NOTE: _LOG_FILE stays None here on purpose. The default log path lives under
  # --output_data_dir, which --clean rmtree's; opening the handle now would let
  # clean delete the log out from under us. It is resolved+set after the --clean
  # block below (see "Resolve run log destination").
  import sys
  import time
  t_start_total = time.time()
  print(f"[Parent Process] Python version: {sys.version}")
  print(f"[Parent Process] sys.executable: {sys.executable}")
  try:
    import numpy as np
    print(f"[Parent Process] numpy imported successfully! version: {np.__version__}")
  except Exception as e:
    print(f"[Parent Process] numpy import failed: {e}")

  # Propagate the run's Vertex project/location (passed as flags) into the env
  # the judge and both Vertex-backed clients read. Generation resolves the flag
  # first, so without this a Gemini/Claude-on-Vertex judge invoked only via
  # flags would have no VERTEX_PROJECT and fail. Flags are the run's explicit
  # intent, so they overwrite any stale env to keep the judge on the SAME
  # project/region as generation.
  if args.vertex_project:
    os.environ["VERTEX_PROJECT"] = args.vertex_project
  if args.vertex_location:
    os.environ["VERTEX_LOCATION"] = args.vertex_location

  # bedrock-gpt: point the judge at the same endpoint/key the generation client
  # resolved (so field_semantic_match hits the same Bedrock deployment). The
  # judge reads BEDROCK_OPENAI_BASE_URL / BEDROCK_OPENAI_API_KEY; mirror the
  # resolved values there when they differ from the judge's own defaults.
  if args.provider == "bedrock-gpt":
    bedrock_base_url, bedrock_api_key_env = _bedrock_endpoint_from_args(args)
    os.environ["BEDROCK_OPENAI_BASE_URL"] = bedrock_base_url
    if bedrock_api_key_env != _BEDROCK_GPT_DEFAULT_API_KEY_ENV:
      key_value = os.environ.get(bedrock_api_key_env, "")
      if key_value:
        os.environ[_BEDROCK_GPT_DEFAULT_API_KEY_ENV] = key_value

  # Same idea for the two official-API providers whose key env is overridable:
  # the judge reads the canonical variable name, so copy the value across when
  # --api_key_env named a different one. (Gemini needs no such mirroring: its
  # judge path already reads GEMINI_API_KEY, the default.)
  if args.provider == "openai":
    openai_base_url, openai_api_key_env = _openai_endpoint_from_args(args)
    os.environ["OPENAI_BASE_URL"] = openai_base_url
    if openai_api_key_env != _OPENAI_DEFAULT_API_KEY_ENV:
      key_value = os.environ.get(openai_api_key_env, "")
      if key_value:
        os.environ[_OPENAI_DEFAULT_API_KEY_ENV] = key_value
  if args.provider == "anthropic":
    anthropic_api_key_env = _anthropic_key_env_from_args(args)
    if anthropic_api_key_env != _ANTHROPIC_DEFAULT_API_KEY_ENV:
      key_value = os.environ.get(anthropic_api_key_env, "")
      if key_value:
        os.environ[_ANTHROPIC_DEFAULT_API_KEY_ENV] = key_value

  # Keep the field_semantic_match LLM judge on the SAME provider+model as
  # generation. Without this it falls back to llm_judge._GEMINI_DEFAULT
  # (gemini-2.5-flash), which 404s on Vertex endpoints that only serve the
  # generation model (e.g. gemini-3.5-flash) -> every semantic check fails.
  from clasharness.evaluation.llm_judge import sync_judge_env
  sync_judge_env(args.provider, args.model)
  _log(
      "LLM judge backend: "
      f"provider={os.environ.get('LLM_JUDGE_PROVIDER') or '(env default)'} "
      f"model={os.environ.get('LLM_JUDGE_MODEL') or '(judge default)'}"
  )

  # --clean deletes precisely the spec/asset tree that --resume exists to
  # continue from, so the combination is always a mistake. Reject it up front
  # rather than after the delete has already happened.
  if args.clean and args.resume:
    raise SystemExit(
        "--clean and --resume are mutually exclusive: --clean deletes the "
        "spec and asset tree that --resume would continue from. Pass one."
    )

  audit_vacuous: list = []
  audit_non_disc: list = []

  seed_dir = _resolve_under(ROOT, args.seed_dir)
  output_dir = _resolve_under(ROOT, args.output_dir)
  # Historically CWD-relative, so the base stays CWD; only the literal '~'
  # mis-routing is corrected. Resolved exactly once and reused everywhere
  # below (including inside the _generate_version_task closure).
  output_data_path = _resolve_under(Path.cwd(), args.output_data_dir)

  # Echo the resolved absolute paths before doing anything destructive: a
  # mis-expanded '~' or a stale flag is obvious here and nowhere else.
  _log(f"Resolved scenario output dir : {output_dir}")
  _log(f"Resolved task data output dir: {output_data_path}")

  if args.clean:
    _guarded_rmtree(output_data_path, "task data dir")
    output_data_path.mkdir(parents=True, exist_ok=True)
    _guarded_rmtree(output_dir, "scenario spec dir")
    output_dir.mkdir(parents=True, exist_ok=True)

  # Resolve run log destination. Done AFTER --clean so the auto-default log
  # (which lives under output_data_dir) is not wiped by clean. An explicit
  # --log_file always wins; otherwise persist a per-run timestamped log next to
  # the task data so failure reasons survive terminal scrollback.
  if args.log_file:
    log_path = Path(args.log_file)
  else:
    log_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_data_path / "logs" / f"generation_{log_stamp}.log"
  log_path.parent.mkdir(parents=True, exist_ok=True)
  _LOG_FILE = str(log_path)
  _log(f"Writing run log to {log_path}")

  if not seed_dir.exists():
    raise SystemExit(f"Seed dir {seed_dir} does not exist.")

  seed_files = sorted(seed_dir.glob("*.json"))
  if args.seed_name:
    wanted = list(dict.fromkeys(args.seed_name))  # de-dup, keep order
    available = {f.stem for f in seed_files}
    missing = [n for n in wanted if n not in available]
    if missing:
      raise SystemExit(
          f"No seed matching name(s) {missing} found in {seed_dir}. "
          f"Available: {sorted(available)}"
      )
    seed_files = [f for f in seed_files if f.stem in set(wanted)]
  if not seed_files:
    raise SystemExit(f"No VLM seed files found under {seed_dir}")

  import threading
  from collections import Counter

  class GenerationStatsTracker:
    def __init__(self) -> None:
      self.lock = threading.Lock()
      self.total_errors = 0
      self.error_types = Counter()
      self.final_failures = {}
      self.durations = {}

    def record_error(self, exc_str: str) -> None:
      with self.lock:
        self.total_errors += 1
        exc_str_lower = exc_str.lower()
        if "json" in exc_str_lower or "parse" in exc_str_lower:
          self.error_types["JSON Decode Failure"] += 1
        elif "resourceexhausted" in exc_str_lower or "429" in exc_str_lower or "quota" in exc_str_lower:
          self.error_types["Rate Limit / Resource Exhausted (429)"] += 1
        elif "503" in exc_str_lower or "unavailable" in exc_str_lower or "overload" in exc_str_lower:
          self.error_types["Service Unavailable (503)"] += 1
        elif "blocked" in exc_str_lower or "safety" in exc_str_lower:
          self.error_types["Safety / Content Blocked"] += 1
        else:
          self.error_types[exc_str[:80] + ("..." if len(exc_str) > 80 else "")] += 1

    def record_final_failure(self, slug: str, exc: Exception) -> None:
      with self.lock:
        self.final_failures[slug] = str(exc)

    def record_task_duration(self, slug: str, duration: float) -> None:
      with self.lock:
        self.durations[slug] = duration

  stats_tracker = GenerationStatsTracker()

  def make_tracking_logger(stats: GenerationStatsTracker):
    def tracking_logger(message: str) -> None:
      _log(message)
      if "api error" in message.lower():
        parts = message.split(":", 1)
        if len(parts) > 1:
          stats.record_error(parts[1].strip())
    return tracking_logger

  client_logger = make_tracking_logger(stats_tracker)

  text_client = _build_text_client(args, client_logger)

  if args.image_provider == "vertex":
    _log(
        "Vertex locations: "
        f"text={args.vertex_location or '(env default)'} "
        f"image={args.image_vertex_location or args.vertex_location or '(env default)'}"
        f" (image model={args.google_model})"
    )

  task_ids = []

  # Pipeline resume checkpoints live OUTSIDE the per-version scenario dir.
  # That dir is deleted whenever a version fails (see the cleanup at the bottom
  # of _generate_version_once), and the checkpoint used to live inside it -- so
  # the cleanup destroyed the Stage 1c/Stage 2 checkpoint that a retry would
  # have resumed from, turning a recoverable late-stage failure into a total
  # loss of the version. Corpus-level placement keeps it clear of the
  # per-version delete while still being wiped by --clean (which rmtrees
  # output_dir wholesale), and the leading dot keeps it out of the
  # '<corpus>/<slug>/' scan pattern every downstream consumer uses.
  checkpoint_root = output_dir / ".checkpoints"

  # Shared helper function to materialize a single version (thread-safe)
  def _generate_version_once(
      seed_item,
      v_idx: int,
      *,
      resume_this_attempt: bool,
  ) -> list[str]:
    num_versions = args.num_versions
    if num_versions > 1:
      version_slug = f"{seed_item.slug}_v{v_idx:02d}"
    else:
      version_slug = seed_item.slug

    version_seed = replace(seed_item, slug=version_slug)
    scenario_output_dir = output_dir / version_slug

    import shutil
    if not resume_this_attempt:
      # Clean target scenario spec directory
      if scenario_output_dir.exists():
        _log(f"[{version_slug}] Cleaning scenario output directory: {scenario_output_dir}")
        shutil.rmtree(scenario_output_dir)

    # Exported task/trajectory debris is keyed on the USER's --resume flag, not
    # on this attempt's resume mode. A version can fail *after* materialization,
    # in the save_task loop below, and the failure handler only removes
    # scenario_output_dir -- so already-exported task folders survive. An
    # internal retry must still sweep them, otherwise a retry that yields fewer
    # rounds than the failed attempt would strand that attempt's orphan task
    # folders in the export tree. (Never observed in the v3 run, whose 2
    # failures both died inside materialization, but internal retries are new
    # and this is the only path that can reach the save_task loop twice.)
    if not args.resume:
      # Clean exported task specifications matching versioned slug.
      # output_data_path is the already-resolved value from main() (closure
      # capture); do not re-derive it from args or the '~' bug comes back.
      tasks_dir = output_data_path / "tasks"
      if tasks_dir.exists():
        for path in tasks_dir.glob(f"*{version_slug}_*"):
          _log(f"[{version_slug}] Cleaning exported task folder: {path}")
          if path.is_dir():
            shutil.rmtree(path)
          else:
            os.remove(path)

      # Clean exported baseline trajectories matching versioned slug
      traj_dir = output_data_path / "trajectories"
      if traj_dir.exists():
        for path in traj_dir.glob(f"*{version_slug}_*"):
          _log(f"[{version_slug}] Cleaning baseline trajectory: {path}")
          os.remove(path)

    # Generate random style seed per version (unless pinned) using high-precision timestamp
    import time
    import random
    if args.style_seed is not None:
      style_seed = args.style_seed
    else:
      # Guarantee unique seed using nanoseconds and version index offset
      unique_base = time.time_ns() + v_idx * 1000000
      # Create a thread-local random generator to avoid state contention in global random
      local_rand = random.Random(unique_base)
      style_seed = local_rand.randint(1000, 999999)
    _log(f"[{version_slug}] Starting generation style_seed={style_seed}")
    t_start_task = time.time()

    try:
      conflict_spec, openclaw_spec = materialize_conflict_scenario(
          seed=version_seed,
          llm_client=text_client,
          output_dir=scenario_output_dir,
          progress=lambda msg: _log(f"[{version_slug}] {msg}"),
          image_provider=args.image_provider,
          chart_provider=args.chart_provider,
          google_model=args.google_model,
          google_api_key_env=args.google_api_key_env,
          google_project=args.google_project,
          vertex_project=args.vertex_project,
          # Image provider gets its own location so an image-only model (e.g.
          # gemini-3-pro-image) can run in a region where it is served, while the
          # text model stays on --vertex_location. Falls back to --vertex_location.
          vertex_location=args.image_vertex_location or args.vertex_location,
          request_delay_seconds=args.request_delay,
          resume=resume_this_attempt,
          checkpoint_dir=checkpoint_root,
          style_seed=style_seed,
          num_render_workers=args.num_render_workers,
          num_generation_workers=args.num_generation_workers,
          auditor_max_iterations=args.auditor_max_iterations,
          retries=args.retries,
          phase_expand_attempts=args.phase_expand_attempts,
          vision_audit=args.vision_audit,
      )
      _log(f"[{version_slug}] Successfully generated scenario version at {scenario_output_dir}")

      validate_conflict_scenario_strict(conflict_spec)

      version_task_ids = []
      for round_item in openclaw_spec.rounds:
        # Gate: a required assets_accessed check whose expected id is neither
        # a concrete visible asset nor an unambiguous base can never pass, and
        # would ship as a silent score ceiling. Fail the version instead.
        lint_problems = lint_assets_accessed_satisfiability(round_item.task_spec)
        if lint_problems:
          raise ValueError(
              f"Unsatisfiable asset checks in {round_item.task_spec.get('task_id')}:\n  - "
              + "\n  - ".join(lint_problems)
          )
        task = Task.from_dict(round_item.task_spec)
        save_task(task, data_dir=output_data_path)
        version_task_ids.append(task.task_id)

      stats_tracker.record_task_duration(version_slug, time.time() - t_start_task)
      return version_task_ids

    except Exception as e:
      stats_tracker.record_task_duration(version_slug, time.time() - t_start_task)
      import traceback
      _log(f"[{version_slug}] Failed to generate: {e}")
      # Route the full stack through _log so it lands in the persisted log file
      # (traceback.print_exc() only reaches stderr and is lost from scrollback).
      _log(f"[{version_slug}] Traceback:\n{traceback.format_exc()}")
      # Cleanup-on-failure: a version that dies before scenario.json is written
      # leaves a half-populated workspace dir (task subdirs + spec.json but no
      # scenario.json). That debris is exactly what makes a failed version look
      # like a broken-but-complete scenario and desyncs from the /tmp export.
      # Remove it so scenario.json's presence stays the single source of truth
      # for "this version completed". Skipped under --resume so partial state
      # remains inspectable/resumable.
      #
      # Keyed on the USER's --resume flag, not on this attempt's resume mode:
      # an internal retry does not need this tree (it resumes from
      # ``checkpoint_root``, which lives outside it), so deleting the debris
      # keeps the "scenario.json == completed" invariant intact even across
      # retries.
      if not args.resume and scenario_output_dir.exists():
        _log(f"[{version_slug}] Removing half-written scenario dir: {scenario_output_dir}")
        shutil.rmtree(scenario_output_dir, ignore_errors=True)
      raise

  def _generate_version_task(seed_item, v_idx: int) -> list[str]:
    """Materialize one version, retrying a failed attempt from its checkpoint.

    Version failures are dominated by stochastic, per-attempt LLM schema misses
    rather than by anything wrong with the seed. Measured on a 60-scenario run:
    30% of versions emitted a malformed background workspace on their first
    try, and the two that exhausted the stage's own retry budget were lost
    outright -- yet BOTH of those seeds had generated cleanly in the previous
    corpus, so the deaths were luck, not content. Re-drawing once turns that
    into a near-certain recovery, and because the pipeline checkpoint now
    survives the failure cleanup the retry resumes mid-pipeline instead of
    repeating the ~85s of already-valid Stage 1-2 work.

    Retries deliberately do NOT loosen any validation: every gate the first
    attempt had to pass still applies. This buys back lost DATA, never quality.
    """
    version_slug = (
        f"{seed_item.slug}_v{v_idx:02d}" if args.num_versions > 1 else seed_item.slug
    )
    total_attempts = 1 + max(0, args.version_retries)
    for attempt in range(1, total_attempts + 1):
      # Attempt 1 honours the user's flag; every retry resumes so it can pick
      # up the checkpoint the failed attempt left behind.
      resume_this_attempt = args.resume or attempt > 1
      try:
        return _generate_version_once(
            seed_item, v_idx, resume_this_attempt=resume_this_attempt
        )
      except Exception as e:
        if attempt >= total_attempts:
          raise
        _log(
            f"[{version_slug}] Attempt {attempt}/{total_attempts} failed ({e});"
            " retrying from the pipeline checkpoint..."
        )
    raise AssertionError("unreachable: retry loop always returns or raises")

  # Build flat list of all version tasks to execute
  tasks_to_run = []
  for seed_file in seed_files:
    seed = load_conflict_vlm_seed_file(seed_file)
    if args.target_rounds is not None:
      seed = replace(seed, target_rounds=args.target_rounds)
    for v_idx in range(1, args.num_versions + 1):
      tasks_to_run.append((seed, v_idx))

  # Execute version generation loop (serial or parallel)
  version_workers = args.num_workers if args.num_workers is not None else 1
  from concurrent.futures import ThreadPoolExecutor, as_completed

  if version_workers > 1 and len(tasks_to_run) > 1:
    _log(f"Spawning parallel version generation with {version_workers} workers...")
    if (args.num_render_workers and args.num_render_workers > 1) or (args.num_generation_workers and args.num_generation_workers > 1):
      _log("Optimization Tip: Nested concurrency active. Consider reducing --num_render_workers or --num_generation_workers if you experience rate limit warnings or high memory usage.")

    with ThreadPoolExecutor(max_workers=version_workers) as executor:
      futures = {
          executor.submit(_generate_version_task, seed, v_idx): f"{seed.slug}_v{v_idx:02d}" if args.num_versions > 1 else seed.slug
          for seed, v_idx in tasks_to_run
      }
      for fut in as_completed(futures):
        version_slug = futures[fut]
        try:
          version_task_ids = fut.result()
          task_ids.extend(version_task_ids)
        except Exception as e:
          stats_tracker.record_final_failure(version_slug, e)
          _log(f"Warning: A version generation worker thread failed for {version_slug}: {e}")
  else:
    for seed, v_idx in tasks_to_run:
      version_slug = f"{seed.slug}_v{v_idx:02d}" if args.num_versions > 1 else seed.slug
      try:
        version_task_ids = _generate_version_task(seed, v_idx)
        task_ids.extend(version_task_ids)
      except Exception as e:
        stats_tracker.record_final_failure(version_slug, e)
        _log(f"Warning: Failed to generate version {version_slug}: {e}")

  if task_ids:
    datasets_dir = output_data_path / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    # Load existing IDs if the dataset index exists
    existing_ids = []
    total_file = datasets_dir / "total.txt"
    if total_file.exists():
      try:
        existing_ids = total_file.read_text("utf-8").strip().splitlines()
      except Exception:
        pass

    # Merge, filter by physical directory existence on disk, and sort
    merged_ids = set(task_ids + existing_ids)
    all_task_ids = sorted([
        tid for tid in merged_ids
        if (output_data_path / "tasks" / tid).exists()
    ])

    # train/dev/test used to be four byte-identical copies of the full id list,
    # so every "held-out" evaluation was scored on data the split claimed was
    # training data. Partition for real, holding out whole groups.
    splits, group_counts, n_groups = _assign_splits(
        all_task_ids, args.split_ratios, args.split_group_by, args.split_salt
    )
    for split_name, ids in splits.items():
      (datasets_dir / f"{split_name}.txt").write_text(
          "\n".join(ids) + "\n", encoding="utf-8"
      )
    (datasets_dir / "total.txt").write_text(
        "\n".join(all_task_ids) + "\n", encoding="utf-8"
    )

    # Fail loudly rather than ship a leaky split: these are the two invariants
    # the old writer violated.
    covered = sorted(tid for ids in splits.values() for tid in ids)
    if covered != all_task_ids:
      raise SystemExit(
          f"Split bug: partition covers {len(covered)} ids but {len(all_task_ids)} "
          f"are registered; refusing to write a lossy split."
      )
    seen: dict[str, str] = {}
    for split_name, ids in splits.items():
      for tid in ids:
        key = _split_group_key(tid, args.split_group_by)
        if seen.setdefault(key, split_name) != split_name:
          raise SystemExit(
              f"Split bug: group {key!r} spans {seen[key]} and {split_name}; "
              f"refusing to write a leaky split."
          )

    _log(
        f"Exported {len(task_ids)} new tasks (Total registered: "
        f"{len(all_task_ids)}) to {datasets_dir}"
    )
    _log(
        f"Split: grouped by {args.split_group_by} into {n_groups} groups -> "
        + ", ".join(
            f"{name}={len(splits[name])} tasks/{group_counts[name]} groups"
            for name in ("train", "dev", "test")
        )
        + f" (salt={args.split_salt!r})"
    )

    if args.audit_discrimination and task_ids:
      _log("-" * 60)
      _log(f"Zero-action discrimination audit over {len(task_ids)} new task(s) ...")
      audit_vacuous, audit_non_disc, audit_unaudited, audit_n = (
          audit_zero_action_discrimination(task_ids, output_data_path)
      )
      _log(f"  replayed {audit_n}/{len(task_ids)} task(s); "
           f"{len(audit_vacuous)} vacuous required check(s), "
           f"{len(audit_non_disc)} non-discriminating task(s), "
           f"{audit_unaudited} check(s) unaudited (grader unavailable)")
      for tid in audit_non_disc:
        _log(f"  NON-DISCRIMINATING {tid}: succeeds with zero actions")
      for tid, cname, cmsg in audit_vacuous:
        _log(f"  VACUOUS {tid} :: {cname} :: {cmsg}")

  # Exact-arm fairness audit. UNCONDITIONAL and unflagged, unlike the
  # discrimination audit above: that one replays every task in an environment
  # and costs real time, this one reads N json files. The defect it watches for
  # shipped once precisely because it was nobody's job to look, so it is not
  # something a run should be able to opt out of.
  #
  # The audit body still cannot abort the run -- everything it reports besides
  # `unwinnable` is advisory, and a crashing auditor must not destroy an
  # otherwise good eight-hour generation. Its VERDICT, though, is now blocking
  # (--fail_on_unwinnable). That verdict is therefore carried OUT of the `try`
  # in these three variables: the `except` below swallows every exception, so a
  # gate evaluated inside it would read as "clean" on exactly the runs where it
  # knows least. `audit_status` starts at FAILED for the same reason -- any path
  # that does not reach the assignment leaves the corpus unverified, not clean.
  audit_unwinnable = []
  audit_untested_specs = 0
  audit_status = UNWINNABLE_AUDIT_FAILED
  try:
    _run_slugs = [
        (f"{seed.slug}_v{v_idx:02d}" if args.num_versions > 1 else seed.slug)
        for seed, v_idx in tasks_to_run
    ]
    # Only versions that actually finished: scenario.json is the completion
    # marker, and a version that died mid-way can still have left an early
    # spec checkpoint behind. Auditing those would mix a half-written corpus
    # into the numbers.
    _audit_dirs = [
        output_dir / slug for slug in _run_slugs
        if (output_dir / slug / "scenario.json").is_file()
    ]
    if not _audit_dirs:
      # No finished scenario means nothing to ship and nothing to grade, so the
      # gate has no subject. Distinguished from OK so it can never read as a
      # clean bill of health on an empty corpus.
      audit_status = UNWINNABLE_AUDIT_NOTHING_SHIPPED
    else:
      fair = audit_exact_arm_fairness(_audit_dirs, log_path=_LOG_FILE)
      # Captured on the line after the call, BEFORE any of the reporting below:
      # a typo in a later _log() f-string must not be able to downgrade a real
      # finding into "the audit crashed, ship it".
      audit_unwinnable = fair["unwinnable"]
      audit_untested_specs = fair["specs"] - fair["sourceability_specs"]
      audit_status = UNWINNABLE_AUDIT_OK
      _log("-" * 60)
      _log(f"Exact-arm fairness audit over {fair['specs']} shipped scenario(s)")
      _log(f"  rounds with a decisive value : {fair['rounds_with_value']}")
      _log(f"  graded by exact match        : {fair['exact_arm']}"
           f"  ({fair['exact_arm_pct']:.1f}%)")
      _log(f"  exact-arm length med / max   : {fair['exact_len_median']:.0f}"
           f" / {fair['exact_len_max']}")
      _log(f"  exact-arm >= 35 chars        : {fair['exact_ge35']}"
           f"  ({fair['exact_ge35_pct']:.1f}%)")
      _log(f"  citability downgrades        : "
           f"{'unavailable (no run log)' if fair['downgrades'] is None else fair['downgrades']}")
      # Printed UNCONDITIONALLY, including when zero. The gate fails open, so
      # silence here is exactly what let 208 aborts ship unnoticed; "0" is the
      # reassuring reading and its absence must never be mistaken for it.
      _gsa = fair['gold_sat_aborts']
      _log(f"  gold-satisfiability aborts   : "
           f"{'unavailable (no run log)' if _gsa is None else _gsa}"
           + ("   <-- a fairness gate was OFFLINE for these rounds; the judge"
              " errored and the gate fails open" if _gsa else ""))
      _log(f"  unique values / in-scen reuse: "
           f"{fair['unique_value_pct']:.1f}% / {fair['reuse_pct']:.1f}%")
      _log(f"  status-phrase values         : {fair['status_phrase_pct']:.1f}%"
           "   (fixed word list; compare only against a run measured the same"
           " way)")
      # The headline. Every one of these is a round a correct report can fail
      # on punctuation alone, so they are listed individually -- a rate with no
      # values attached is not actionable.
      _log(f"  UNGRADABLE RESIDUAL          : {len(fair['residual'])}"
           f"  ({fair['residual_pct']:.1f}% of the exact arm)")
      for _s, _rid, _v in fair["residual"]:
        _log(f"    RESIDUAL {_s} {_rid}: {_v!r}")
      if not fair["residual"]:
        _log("    none -- every exact-graded value reads as a single atomic"
             " reading")
      # The harder headline. A residual round is one a correct report can fail
      # on WORDING; an unwinnable round cannot be passed at all, because the
      # value it demands is not in anything that round shows the solver. These
      # are `required` checks, so each one caps its task's score below 1.0 for
      # every model forever.
      _log(f"  UNWINNABLE (unreadable value)  : {len(fair['unwinnable'])}"
           f"  ({fair['unwinnable_pct']:.1f}% of the exact arm)")
      if fair["sourceability_specs"] != fair["specs"]:
        _log(f"    ! only {fair['sourceability_specs']}/{fair['specs']}"
             " scenario(s) carried both scenario.json and asset_manifest.json;"
             " the rest were NOT tested")
      for _s, _rid, _v, _tier, _holders in fair["unwinnable"]:
        _log(f"    UNWINNABLE [{_tier}] {_s} {_rid}: {_v!r}")
        _log(f"      readable in: {_holders or '(no asset in the package)'}")
      if not fair["unwinnable"] and fair["sourceability_specs"]:
        _log("    none -- every exact-graded value is readable from the"
             " round's own visible_refs")
  except Exception as exc:  # noqa: BLE001 - the audit must never abort the run
    _log(f"WARNING: exact-arm fairness audit failed (corpus unaffected): {exc}")

  # Generation Statistics Report. Routed through _log() so the summary (and, in
  # particular, the per-version failure reasons) is persisted to the log file
  # rather than only printed to stdout.
  total_duration = time.time() - t_start_total
  _log("=" * 60)
  _log("          CLASHarness GENERATION STATISTICS REPORT          ")
  _log("=" * 60)
  _log(f"Total API Intermediate Errors Occurred: {stats_tracker.total_errors}")
  if stats_tracker.total_errors > 0:
    _log("API Errors by Grouped Type:")
    for err_type, count in stats_tracker.error_types.most_common():
      _log(f"  - {err_type}: {count} times")

  total_attempted_versions = len(tasks_to_run)
  total_failed_versions = len(stats_tracker.final_failures)
  total_successful_versions = total_attempted_versions - total_failed_versions

  _log("Scenario Version Generation Summary:")
  _log(f"  - Total Attempted versions: {total_attempted_versions}")
  _log(f"  - Successfully materialized: {total_successful_versions}")
  _log(f"  - Failed (retries exhausted / validator checks failed): {total_failed_versions}")

  if stats_tracker.durations:
    _log("Generation Durations by Version:")
    for slug, dur in sorted(stats_tracker.durations.items()):
      _log(f"  * {slug}: {dur:.2f} seconds")

  if total_failed_versions > 0:
    _log("Failed Scenario Versions Details (All Retries Exhausted):")
    for slug, exc_desc in stats_tracker.final_failures.items():
      _log(f"  * {slug}:")
      _log(f"    Reason/Exception: {exc_desc}")
  _log(f"Total Run Execution Time: {total_duration:.2f} seconds")
  _log("=" * 60)

  # Reported last and raised last, on purpose: the statistics report above is
  # what makes a failed run diagnosable, and the generated data is deliberately
  # left on disk so the flagged tasks can be inspected (or the seeds fixed)
  # rather than silently regenerated from scratch.
  _gate = shipping_gate_failure(
      vacuous=audit_vacuous,
      non_discriminating=audit_non_disc,
      unwinnable=audit_unwinnable,
      audit_status=audit_status,
      untested_specs=audit_untested_specs,
      fail_on_unwinnable=args.fail_on_unwinnable,
  )
  if _gate:
    raise SystemExit(_gate)
if __name__ == "__main__":
  main()
