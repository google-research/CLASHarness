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

"""Complete example: training and evaluating agents with clasharness.

This script demonstrates four agent types and two usage patterns:
  - Evaluation: measure task success rate across a split
  - Data collection: record trajectories for offline RL / behavior cloning

Agent types
-----------
  expert   -- runs the ground-truth solution commands (upper bound)
  random   -- issues random commands (lower bound baseline)
  llm      -- Claude-powered agent via the Anthropic API (requires ANTHROPIC_API_KEY)
  rule     -- simple hand-written heuristics (example of a rule-based agent)

Usage examples
--------------
  # Evaluate expert on dev split
  python examples/train_and_eval.py --agent expert --split dev

  # Evaluate LLM agent on calendar domain, difficulty 1
  python examples/train_and_eval.py --agent llm --split dev \\
      --domain calendar --difficulty 1 --limit 10

  # Collect training data (saves JSONL trajectories)
  python examples/train_and_eval.py --agent expert --split train \\
      --collect-data --out-dir data/trajectories/expert

  # Quick smoke test (3 tasks)
  python examples/train_and_eval.py --agent expert --split dev --limit 3
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import argparse
import base64
import contextlib
from dataclasses import dataclass, field
from datetime import date, timedelta
import hashlib
import io
import json
import os
os.environ["GOOGLE_API_USE_CLIENT_CERTIFICATE"] = "false"
from pathlib import Path
import random
import re
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any, Callable
import urllib.error
import urllib.request

# Allow running from the repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

from clasharness import make_env
from clasharness.core.environment import CLASHarnessEnv
from clasharness.core.observation import EvaluationResult, Observation
from clasharness.core.task import Task, load_task, load_task_ids
from clasharness.utils.output_cleaner import clean_openclaw_output
from clasharness.tools.agent_tool_spec import (
    cli_verbs_by_family,
    iter_command_specs,
    required_flags_by_tokens,
    required_positionals_by_tokens,
)
from clasharness.utils.episode_memory import (
    build_memory_summary,
    build_openai_request_messages,
    clip_text,
    is_read_only_action,
)

# --- Extracted to train_eval/ submodules (re-exported for back-compat) ---
from train_eval import agents
from train_eval.agents import (
    BaseAgent,
    ExpertAgent,
    LLMAgent,
    RandomAgent,
    RuleAgent,
    _collapse_response_text_parts,
    _effective_multimodal_input_mode,
    _extract_command_from_reasoning_text,
    _multimodal_image_parts_from_task,
    _normalize_model_command,
    _validate_command,
)

# --- Extracted to train_eval/ submodules (re-exported for back-compat) ---
from train_eval._constants import (
    _ANTHROPIC_BYTE_BUDGET_EDGE_LADDER,
    _ANTHROPIC_IMAGE_EDGE_FALLBACKS,
    _ANTHROPIC_MODELS_OMIT_TEMPERATURE,
    _ANTHROPIC_MODEL_IMAGE_EDGE,
    _ANTHROPIC_REQUEST_BYTE_BUDGET,
    _DOMAIN_HINT_GROUPS,
    _DOWNSCALED_IMAGE_CACHE,
    _PLACEHOLDER_VALUES,
    _SIDE_EFFECT_HINTS,
    _VALID_CONFIG_PATHS,
    _WEATHER_CODE_TEXT,
)
from train_eval.llm_errors import (
    _classify_error_type,
    _has_provider_noise,
    _is_content_filter_llm_error,
    _is_deadline_error,
    _is_flaky_safety_llm_error,
    _is_retryable_llm_error,
    _retry_backoff_delay,
)
from train_eval.command_catalog import (
    CommandSpec,
    _CLI_VERBS,
    _COMMAND_SPECS,
    _REQUIRED_FLAGS_BY_PREFIX,
    _REQUIRED_POSITIONAL_ARGS_BY_PREFIX,
    _command_catalog_text,
    _command_specs_for_task,
    _command_specs_from_tool_spec,
    _contains_placeholder_value,
    _looks_like_catalog_placeholder,
    _spec_key_from_command,
    _strip_placeholder_token,
)
from train_eval.effect_checks import (
    _created_required_report,
    _effect_satisfied_by_actions,
    _get_flag_value,
    _has_flag_value,
    _has_flag_value_contains,
    _instruction_implies_side_effect,
    _missing_required_report_stop_reason,
    _missing_required_side_effect_stop_reason,
    _required_effect_checks,
    _required_report_path,
    _required_submission_stop_reason,
)
from train_eval.trajectory_summary import (
    _build_final_response,
    _extract_calendar_summary,
    _extract_location_from_geocoding,
    _extract_weather_summary,
    _safe_json_dict,
    _step_action,
    _step_output,
)
from train_eval.image_utils import (
    _anthropic_text_from_response,
    _downscale_image_b64,
    _estimate_anthropic_request_bytes,
    _fit_anthropic_messages_to_byte_budget,
    _is_anthropic_image_dimension_error,
    _is_anthropic_temperature_unsupported,
    _normalize_anthropic_max_image_edge,
    _parse_anthropic_image_pixel_limit,
    _pil_format_for_media_type,
    _to_anthropic_messages,
)
from train_eval.eval_report import (
    DomainMetrics,
    EpisodeResult,
    EvalSummary,
    _generate_evaluation_visualizer,
    _load_asset_refs,
)

DEFAULT_TASK_DATA_DIR = Path(__file__).parent.parent / "clasharness" / "data"
DEFAULT_MAX_STEPS = 100
# Default for reasoning-threading in native tool-calling. When on, native mode
# threads each provider's opaque per-turn reasoning back into every forced
# tool-call step (Claude extended thinking / GPT encrypted reasoning items) so
# the chain of thought stays continuous -- mirroring what Gemini already does
# with thought_signature, so all three providers are treated consistently.
# Disable (--no-llm-native-reasoning) to reproduce the prior reasoning-dropping
# native behavior (forced tool_choice, no thinking).
DEFAULT_LLM_NATIVE_REASONING = True
DEFAULT_BENCHMARK_DATE = "2026-03-01"
BACKEND_MODE_CHOICES = ["mock", "multi", "martin", "real", "hybrid"]


class _TeeStdout:
  """Mirror stdout writes to the original stream and an optional log file."""

  def __init__(self, primary: Any, secondary: Any) -> None:
    self._primary = primary
    self._secondary = secondary

  def write(self, data: str) -> int:
    written = self._primary.write(data)
    self._secondary.write(data)
    return written

  def flush(self) -> None:
    self._primary.flush()
    self._secondary.flush()

  def isatty(self) -> bool:
    return bool(getattr(self._primary, "isatty", lambda: False)())






















def _task_public_dict(task: Task | None) -> dict[str, Any]:
  public = getattr(getattr(task, "data", None), "public", None)
  return public if isinstance(public, dict) else {}


def _is_hard_decision_task(task: Task | None) -> bool:
  if task is None:
    return False
  task_id = str(getattr(task, "task_id", "") or "")
  if task_id.startswith("hard_decision_workflow_"):
    return True
  public = _task_public_dict(task)
  return any(
      isinstance(public.get(key), str) and public.get(key)
      for key in ("hard_decision_scenario", "hard_decision_ability")
  )


def _benchmark_anchor_timezone(task: Task | None) -> str | None:
  ground_truth = getattr(task, "ground_truth", None)
  commands = (
      getattr(ground_truth, "solution_commands", None) if ground_truth else None
  )
  if not commands:
    return None
  for cmd in commands:
    raw = str(cmd or "").strip()
    if raw.startswith("calendar today --timezone "):
      return raw.split("calendar today --timezone ", 1)[1].strip().strip("'\"")
  return None


def _rewrite_task_instruction_for_prompt(
    task_instruction: str, task: Task | None
) -> str:
  if not _is_hard_decision_task(task):
    return task_instruction
  return re.sub(r"\bboard\b", "task board", task_instruction)


def _benchmark_prompt_notes(task: Task | None) -> list[str]:
  if not _is_hard_decision_task(task):
    return []
  notes = [
      (
          "Benchmark time anchor: treat relative dates like 'today' and 'next"
          " Friday' as belonging to the benchmark world, not the host system"
          f" date. The benchmark world date anchor is {DEFAULT_BENCHMARK_DATE}"
          " unless command output gives a more specific benchmark time."
      ),
      (
          "In hard workflow tasks, 'board' means the task board managed with"
          " `tasks *` commands unless the command catalog explicitly includes a"
          " required channel or messaging action."
      ),
      (
          "Prefer `calendar today --timezone ...` before inferring date ranges"
          " from relative time phrases."
      ),
  ]
  timezone = _benchmark_anchor_timezone(task)
  if timezone:
    notes.append(
        f"For this task, prefer `calendar today --timezone {timezone}` when you"
        " need the current benchmark-local date."
    )
  return notes


def _is_hard_task_id(task_id: str | None) -> bool:
  return bool(task_id and task_id.startswith("hard_decision_workflow_"))


def _resolve_max_steps(
    max_steps: int | None,
    *,
    split: str | None = None,
    task_prefix: str | None = None,
    task_id: str | None = None,
) -> int:
  del split, task_prefix, task_id
  return max_steps if max_steps is not None else DEFAULT_MAX_STEPS


def _host_reference_date() -> date:
  return date.today()






















































































def _detect_date_anchor_mismatch(
    task: Task | None,
    executed_actions: list[str],
) -> bool:
  if not _is_hard_decision_task(task) or not executed_actions:
    return False
  host_today = _host_reference_date().isoformat()
  if host_today == DEFAULT_BENCHMARK_DATE:
    return False
  suspicious_dates = {
      host_today,
      (_host_reference_date() - timedelta(days=1)).isoformat(),
      (_host_reference_date() + timedelta(days=1)).isoformat(),
  }
  text = "\n".join(executed_actions)
  return any(marker in text for marker in suspicious_dates)


# ===========================================================================
# Episode result
# ===========================================================================










def _agent_run_config(agent: Any) -> dict[str, str]:
  debug_fn = getattr(agent, "debug_run_config", None)
  if not callable(debug_fn):
    return {}
  config = debug_fn()
  return config if isinstance(config, dict) else {}


# ===========================================================================
# Base agent interface
# ===========================================================================




# ===========================================================================
# Expert agent
# ===========================================================================




# ===========================================================================
# Random agent
# ===========================================================================




# ===========================================================================
# Rule-based agent
# ===========================================================================




# ===========================================================================
# LLM agent (Anthropic or OpenAI-compatible API)
# ===========================================================================




# ===========================================================================
# Episode runner

# ===========================================================================
# Episode metadata helpers
# ===========================================================================


def _episode_metadata(task: Task | None, mode: str) -> dict[str, Any]:
  public = _task_public_dict(task)
  return {
      "mode": mode,
      "domains": list(getattr(task, "domains", []) or []),
      "difficulty": getattr(task, "difficulty", None),
      "scenario_slug": (
          public.get("scenario_slug") or public.get("hard_decision_scenario")
      ),
      "round_id": public.get("round_id"),
      "round_type": public.get("round_type"),
      "generator_id": getattr(task, "generator_id", None),
      "template_id": getattr(task, "template_id", None),
      "instruction": getattr(task, "instruction", ""),
      "canonical_instruction": getattr(task, "canonical_instruction", ""),
  }


# ===========================================================================


def run_episode(
    task_id: str,
    agent: BaseAgent,
    task_data_dir: str | Path | None = None,
    mode: str = "multi",
    max_steps: int = DEFAULT_MAX_STEPS,
    record_trajectory: bool = False,
    online_clean: bool = False,
    backend_kwargs: dict[str, Any] | None = None,
    stop_on_success: bool = True,
    max_stagnant_steps: int = 3,
    record_step_eval: bool = False,
    verbose: bool = False,
    interaction_sink: Callable[[dict[str, Any]], None] | None = None,
) -> EpisodeResult:
  """Run one episode of `task_id` with `agent`.

  Parameters
  ----------
  task_id:
      Task identifier (e.g. ``"calendar_add_event_1"``).
  agent:
      The agent to run.
  mode:
      Backend mode: ``"mock"`` | ``"multi"`` | ``"martin"`` | ``"real"`` |
      ``"hybrid"``.
  max_steps:
      Hard cap on the number of steps per episode.
  record_trajectory:
      If True, store the full step-by-step trajectory in the result.
  record_step_eval:
      If True, include evaluator progress after each executed command.
  verbose:
      Print each step to stdout.
  """
  t_start = time.time()
  trajectory: list[dict[str, Any]] = []
  executed_actions: list[str] = []
  error = ""
  provider_impacted = False
  compact_fallback = False
  task: Task | None = None
  stop_reason = ""
  episode_metadata: dict[str, Any] = {"mode": mode}
  # Tracks whether this episode's one-time interaction-log header was emitted.
  interaction_header_state = {"written": False}

  def _drain_interactions(current_step: int) -> None:
    """Flush the agent's per-attempt interaction records for the turn just run.

    Called after EVERY act() (including the terminal turn that coerces DONE,
    which the trajectory drops), so the sidecar log is the complete record.
    """
    if interaction_sink is None:
      return
    getter = getattr(agent, "debug_interactions", None)
    if not callable(getter):
      return
    records = getter()
    if not isinstance(records, list) or not records:
      return
    if not interaction_header_state["written"]:
      interaction_header_state["written"] = True
      sys_getter = getattr(agent, "debug_system_prompt", None)
      cfg_getter = getattr(agent, "debug_run_config", None)
      interaction_sink({
          "record_type": "episode_header",
          "task_id": task_id,
          "system_prompt": sys_getter() if callable(sys_getter) else None,
          "run_config": cfg_getter() if callable(cfg_getter) else {},
      })
    for entry in records:
      interaction_sink({
          "record_type": "attempt",
          "task_id": task_id,
          "step": current_step,
          **entry,
      })

  try:
    with make_env(
        task_id,
        task_data_dir=task_data_dir,
        mode=mode,
        max_steps=max_steps,
        backend_kwargs=backend_kwargs,
        record_step_evaluation=record_step_eval,
    ) as env:
      obs: Observation = env.reset()
      task = env.task
      episode_metadata = _episode_metadata(task, mode)
      agent.reset(obs.task_instruction, task)

      if verbose:
        print(f"\n{'─'*60}")
        print(f"TASK  : {task_id}")
        print(f"AGENT : {agent.name}")
        run_config = _agent_run_config(agent)
        if run_config:
          provider = run_config.get("provider")
          model = run_config.get("model")
          history_mode = run_config.get("history_mode")
          if provider or model:
            llm_label = " / ".join(part for part in (provider, model) if part)
            print(f"LLM   : {llm_label}")
          if history_mode:
            print(f"HIST  : {history_mode}")
        print(f"INSTR : {obs.task_instruction}")
        print(f"{'─'*60}")

      done = False
      step = 0
      eval_result: EvaluationResult | None = None
      stagnant_steps = 0
      last_transition: tuple[str, str, str, int] | None = None
      while not done and step < max_steps:
        if agent.should_stop():
          stop_reason = "agent requested stop before selecting another command"
          break
        action = agent.act(obs)
        reply_attempts = agent.debug_reply_attempts()
        reply_debug = agent.debug_last_reply()
        # Drain BEFORE the DONE check below: act() fully populates the
        # interaction log (incl. the terminal coerced-DONE turn) before
        # returning, and this point runs for every turn, so the sidecar log
        # captures the fatal turn that the trajectory (which breaks on DONE)
        # drops.
        _drain_interactions(step)
        provider_impacted = provider_impacted or _has_provider_noise(
            reply_attempts or ([reply_debug] if reply_debug else [])
        )
        if not compact_fallback and any(
            "retrying with compact prompt" in a for a in (reply_attempts or [])
        ):
          # Compact fallback silently drops ALL images + full history to recover
          # from a content-filter / invalid_prompt rejection, so this task's
          # result is degraded. Surface it loudly (independent of --verbose),
          # once per episode on the False->True transition, tagged with task_id
          # so a degraded run is never mistaken for a clean one.
          compact_fallback = True
          print(
              f"WARNING: task {task_id}: compact-prompt fallback triggered at"
              f" step {step} (images + full history dropped; result degraded)"
          )
        if verbose and reply_attempts:
          for attempt in reply_attempts:
            print(f"       model: {attempt}")
        elif verbose and reply_debug:
          print(f"       model: {reply_debug}")
        if action.upper() == "DONE":
          eval_result = env.evaluate()
          # A DONE produced by exhausting the reply-retry budget (persistently
          # empty or invalid generation) is a generation stall, NOT an
          # agent-chosen early stop. Attribute it distinctly so it is not
          # bucketed as "done early" / missing-side-effect.
          stall_reason: str | None = None
          debug_stall = getattr(agent, "debug_stall_reason", None)
          if callable(debug_stall):
            candidate = debug_stall()
            stall_reason = candidate if isinstance(candidate, str) else None
          stop_reason = (
              stall_reason
              or _required_submission_stop_reason(
                  task, executed_actions, eval_result
              )
              or "agent returned DONE before completing required action"
          )
          break
        if verbose:
          print(f"  [{step+1:02d}] $ {action}")

        obs, reward, done, info = env.step(action)
        executed_actions.append(action)
        if stop_on_success:
          eval_result = env.evaluate()
          if eval_result.success:
            done = True

        transition = (
            action,
            obs.command_output or "",
            obs.error_output or "",
            int(obs.exit_code),
        )
        if transition == last_transition:
          stagnant_steps += 1
        else:
          stagnant_steps = 1
          last_transition = transition
        if max_stagnant_steps > 0 and stagnant_steps >= max_stagnant_steps:
          stop_reason = (
              f"stopped after {max_stagnant_steps} identical no-progress"
              " transitions"
          )
          done = True

        if verbose:
          out = (obs.command_output or "").strip()[:120]
          print(f"       exit={obs.exit_code}  {out}")
          step_eval = info.get("step_evaluation")
          if step_eval is not None:
            current_score = step_eval.get("current_score", 0.0)
            score_delta = step_eval.get("score_delta", 0.0)
            delta_str = f" ({score_delta:+.2f})" if score_delta != 0 else ""
            print(f"       step_score={current_score:.2f}{delta_str}")

        if record_trajectory:
          clean_stdout = obs.command_output or ""
          clean_stderr = obs.error_output or ""
          if online_clean:
            clean_stdout, clean_stderr = clean_openclaw_output(
                clean_stdout,
                clean_stderr,
            )

          command_meta = info.get("command_meta", {})
          step_evaluation = info.get("step_evaluation")
          step_record = {
              "step": step,
              "action": action,
              "model_debug": (
                  list(reply_attempts)
                  if reply_attempts
                  else ([reply_debug] if reply_debug else [])
              ),
              "stdout": obs.command_output,
              "stderr": obs.error_output,
              "executed_action": command_meta.get("executed_action", action),
              "clean_stdout": clean_stdout,
              "clean_stderr": clean_stderr,
              "error_tags": command_meta.get("error_tags", []),
              "compat_status": command_meta.get("compat_status", "ok"),
              "execution_trace": command_meta.get("execution_trace", []),
              "exit_code": obs.exit_code,
              "reward": reward,
              "done": done,
          }
          if step_evaluation is not None:
            step_record.update({
                "step_evaluation": step_evaluation,
                "current_score": step_evaluation.get("current_score"),
                "score_delta": step_evaluation.get("score_delta"),
                "passed_checks": step_evaluation.get("passed_checks", []),
                "failed_checks": step_evaluation.get("failed_checks", []),
            })
          trajectory.append(step_record)
        step += 1

      if eval_result is None:
        eval_result = env.evaluate()

      if not eval_result.success and not stop_reason:
        # Precedence matters for diagnostics: an episode that exhausted its step
        # budget is "step capped", even if it also happens to be missing a
        # required side effect. Reporting the missing-side-effect reason here
        # would wrongly bucket capped runs as "done early" and make
        # step_capped_tasks always read 0. Only attribute a missing-submission
        # reason when the agent stopped *before* the budget ran out (DONE,
        # should_stop, stagnation, or env-signalled done).
        if step >= max_steps:
          stop_reason = f"reached max_steps={max_steps}"
        else:
          stop_reason = _required_submission_stop_reason(
              task, executed_actions, eval_result
          )

      if verbose:
        status = "PASS" if eval_result.success else "FAIL"
        print(f"  [{status}] score={eval_result.score:.2f}  steps={step}")
        if not eval_result.success:
          if not stop_reason and step >= max_steps:
            stop_reason = f"reached max_steps={max_steps}"
          if stop_reason:
            print(f"       reason: {stop_reason}")
          failed_checks = [d for d in eval_result.details if not d.passed]
          if failed_checks:
            print("       failed checks:")
            for detail in failed_checks[:3]:
              message = (detail.message or "(no message)").strip()
              print(f"         - {detail.name}: {message}")

  except Exception as exc:
    error = str(exc)
    import traceback
    import sys
    tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    sys.stderr.write(f"\n[ERROR] Task '{task_id}' encountered exception:\n{tb_str}\n")
    sys.stderr.flush()
    # Best-effort: flush any interaction records logged before the raise (e.g.
    # earlier attempts in the failing step). Never let logging mask the error.
    try:
      _drain_interactions(locals().get("step", 0))
    except Exception:  # pragma: no cover - logging must not shadow the failure
      pass
    if verbose:
      reply_attempts = agent.debug_reply_attempts()
      reply_debug = agent.debug_last_reply()
      if reply_attempts:
        for attempt in reply_attempts:
          print(f"       model: {attempt}")
      elif reply_debug:
        print(f"       model: {reply_debug}")
      print(f"  [ERROR] {exc}")
    return EpisodeResult(
        task_id=task_id,
        agent_name=agent.name,
        success=False,
        score=0.0,
        steps=0,
        duration_s=time.time() - t_start,
        trajectory=trajectory,
        final_response="",
        error=error,
        error_type=_classify_error_type(error, False, stop_reason),
        provider_impacted=provider_impacted
        or _has_provider_noise(agent.debug_reply_attempts(), error),
        compact_fallback=compact_fallback
        or any(
            "retrying with compact prompt" in a
            for a in (agent.debug_reply_attempts() or [])
        ),
        stop_reason=stop_reason,
        date_anchor_mismatch=_detect_date_anchor_mismatch(
            task, executed_actions
        ),
        mode=mode,
        metadata=episode_metadata,
    )

  if eval_result is not None:
    episode_metadata["check_results"] = [d.to_dict() for d in eval_result.details]

  final_response = _build_final_response(task, trajectory)
  error_type = _classify_error_type(error, eval_result.success, stop_reason)
  return EpisodeResult(
      task_id=task_id,
      agent_name=agent.name,
      success=eval_result.success,
      score=eval_result.score,
      steps=step,
      duration_s=time.time() - t_start,
      trajectory=trajectory,
      final_response=final_response,
      error=error,
      error_type=error_type,
      provider_impacted=provider_impacted or _has_provider_noise(error=error),
      compact_fallback=compact_fallback,
      stop_reason=stop_reason,
      date_anchor_mismatch=_detect_date_anchor_mismatch(task, executed_actions),
      mode=mode,
      metadata=episode_metadata,
  )


# ===========================================================================
# Batch evaluation
# ===========================================================================






def _update_summary(
    summary: EvalSummary,
    result: EpisodeResult,
    task_id: str,
    data_dir: Path | None,
) -> None:
  core_hard_abilities = {
      "duplicate_avoidance",
      "gap_completion",
      "information_transfer",
      "multi_source_reasoning",
      "state_repair",
      "workflow_completion",
  }
  summary.results.append(result)
  summary.total += 1
  summary.total_score += result.score
  if result.success:
    summary.passed += 1
  if result.error_type == "provider_failure":
    summary.provider_failures += 1
  if result.provider_impacted:
    summary.provider_impacted_tasks += 1
  if result.compact_fallback:
    summary.compact_fallback_tasks += 1

  # Per-domain breakdown (load task metadata)
  try:
    task = load_task(task_id, data_dir=data_dir)
    for d in task.domains:
      dm = summary.by_domain.setdefault(d, DomainMetrics())
      dm.total += 1
      dm.total_score += result.score
      if result.success:
        dm.passed += 1
    diff = task.difficulty
    dfm = summary.by_difficulty.setdefault(diff, DomainMetrics())
    dfm.total += 1
    dfm.total_score += result.score
    if result.success:
      dfm.passed += 1

    public = task.data.public if getattr(task, "data", None) else {}
    hard_scenario = public.get("hard_decision_scenario") or public.get(
        "scenario_slug"
    )
    if isinstance(hard_scenario, str) and hard_scenario:
      hsm = summary.by_hard_scenario.setdefault(hard_scenario, DomainMetrics())
      hsm.total += 1
      hsm.total_score += result.score
      if result.success:
        hsm.passed += 1

    hard_ability = public.get("hard_decision_ability")
    if isinstance(hard_ability, str) and hard_ability:
      ham = summary.by_hard_ability.setdefault(hard_ability, DomainMetrics())
      ham.total += 1
      ham.total_score += result.score
      if result.success:
        ham.passed += 1

    hard_ability_tags = public.get("hard_decision_ability_tags")
    if isinstance(hard_ability_tags, list):
      for tag in hard_ability_tags:
        if not isinstance(tag, str) or tag not in core_hard_abilities:
          continue
        htm = summary.by_hard_ability_tag.setdefault(tag, DomainMetrics())
        htm.total += 1
        htm.total_score += result.score
        if result.success:
          htm.passed += 1
  except Exception:
    pass






def run_evaluation(
    agent: BaseAgent,
    split: str = "dev",
    task_data_dir: str | Path | None = None,
    mode: str = "multi",
    task_prefix: str | None = None,
    scenario_slugs: list[str] | None = None,
    domain: str | None = None,
    difficulty: int | None = None,
    limit: int | None = None,
    max_steps: int | None = None,
    max_stagnant_steps: int = 3,
    online_clean: bool | None = None,
    skip_incompatible_openclaw: bool = True,
    fallback_openclaw_network_to_mock: bool = False,
    strict_online_data: bool = True,
    online_openclaw_only: bool = False,
    inter_task_sleep_s: float = 0.0,
    record_step_eval: bool = False,
    verbose: bool = False,
    interaction_log: str | Path | None = None,
) -> EvalSummary:
  """Evaluate `agent` on a task split.

  Parameters
  ----------
  agent:
      The agent instance to evaluate.
  split:
      Dataset split: ``"train"`` | ``"dev"`` | ``"test"``.
  mode:
      Backend mode: ``"mock"`` | ``"multi"`` | ``"real"`` | ``"hybrid"``.
  domain:
      If set, only evaluate tasks in this domain (e.g. ``"calendar"``).
  difficulty:
      If set (1/2/3), only evaluate tasks at this difficulty level.
  limit:
      Maximum number of tasks to run (useful for quick tests).
  max_steps:
      Hard cap on steps per episode.
  verbose:
      Print per-episode details.
  """
  data_dir = Path(task_data_dir) if task_data_dir else None
  effective_max_steps = _resolve_max_steps(
      max_steps,
      split=split,
      task_prefix=task_prefix,
  )
  task_ids = load_task_ids(
      split, data_dir=data_dir, difficulty=difficulty, domain=domain
  )
  task_ids = _filter_task_ids_by_prefix(task_ids, task_prefix)
  task_ids = _filter_task_ids_by_scenario_slugs(
      task_ids, scenario_slugs, data_dir=data_dir
  )
  if limit:
    task_ids = task_ids[:limit]
  if online_openclaw_only:
    task_ids = _select_openclaw_only_task_ids(task_ids, data_dir=data_dir)

  if not task_ids:
    print(
        f"No tasks found for split='{split}' domain={domain}"
        f" difficulty={difficulty} task_prefix={task_prefix}"
        f" scenario_slugs={scenario_slugs}"
    )
    run_config = _agent_run_config(agent)
    return EvalSummary(
        agent_name=agent.name,
        split=split,
        total=0,
        passed=0,
        total_score=0.0,
        exec_mode=mode,
        llm_provider=run_config.get("provider"),
        llm_model=run_config.get("model"),
        llm_history_mode=run_config.get("history_mode"),
    )

  print(
      f"Evaluating '{agent.name}' on {len(task_ids)} tasks "
      f"(split={split}, mode={mode})..."
  )

  run_config = _agent_run_config(agent)
  summary = EvalSummary(
      agent_name=agent.name,
      split=split,
      total=0,
      passed=0,
      total_score=0.0,
      exec_mode=mode,
      llm_provider=run_config.get("provider"),
      llm_model=run_config.get("model"),
      llm_history_mode=run_config.get("history_mode"),
  )
  should_clean = _resolve_online_clean(mode, online_clean)
  backend_kwargs = _make_backend_kwargs(
      mode,
      skip_incompatible_openclaw,
      fallback_openclaw_network_to_mock,
      strict_online_data,
  )

  core_hard_abilities = {
      "duplicate_avoidance",
      "gap_completion",
      "information_transfer",
      "multi_source_reasoning",
      "state_repair",
      "workflow_completion",
  }

  inter_fh = None
  inter_sink = None
  if interaction_log is not None and _agent_captures_interactions(agent):
    inter_file = _interaction_log_file(interaction_log, f"{agent.name}_{split}")
    inter_fh = open(inter_file, "w", encoding="utf-8")
    inter_sink = _make_jsonl_sink(inter_fh)
    print(f"Interaction log → {inter_file}")

  for i, task_id in enumerate(task_ids, 1):
    if i > 1 and inter_task_sleep_s > 0:
      time.sleep(inter_task_sleep_s)
    if not verbose:
      print(f"  [{i:4d}/{len(task_ids)}] {task_id:<50}", end="\r", flush=True)

    result = run_episode(
        task_id,
        agent,
        task_data_dir=data_dir,
        mode=mode,
        max_steps=effective_max_steps,
        max_stagnant_steps=max_stagnant_steps,
        online_clean=should_clean,
        backend_kwargs=backend_kwargs,
        record_step_eval=record_step_eval or verbose,
        verbose=verbose,
        interaction_sink=inter_sink,
    )
    _update_summary(summary, result, task_id, data_dir)

  if not verbose:
    print()  # clear the \r line
  if inter_fh is not None:
    inter_fh.close()

  return summary


# ===========================================================================
# Training data collection (offline RL / behavior cloning)
# ===========================================================================


def _clean_output_dir(out_path: Path) -> None:
  """Delete existing contents of the trajectory output directory.

  Backs the ``--clean-out-dir`` flag so a fresh ``--collect-data`` run does not
  inherit stale per-scenario ``.jsonl`` files, ``.summary`` reports, or the
  dashboard from a previous run (which the aggregating visualizer would
  otherwise pick up). The directory itself is preserved; only its contents are
  removed. Symlinked directories are unlinked, not followed.

  Passing ``--clean-out-dir`` is itself the explicit request to delete, so no
  second confirmation flag is required. What IS reported first is the file
  count and size being removed, so a mistyped ``--out-dir`` is visible in the
  run log rather than silently costing a completed evaluation. The guard that
  actually prevents the expensive failure mode lives in the caller:
  ``--clean-out-dir`` combined with any ``--resume``/``--retry`` flag is
  rejected outright, because wiping the directory the resume logic is about to
  read turns "continue where I left off" into "re-run the whole split at full
  API cost, having already deleted what it would have resumed from".
  """
  resolved = Path(out_path).expanduser()
  try:
    resolved = resolved.resolve()
  except OSError:
    resolved = resolved.absolute()
  # Safety: never wipe a filesystem root (e.g. "/"), where parent == self.
  if resolved.parent == resolved:
    raise ValueError(f"Refusing to clean unsafe out-dir path: {resolved}")
  if not resolved.exists():
    return
  if not resolved.is_dir():
    raise NotADirectoryError(
        f"--clean-out-dir target exists but is not a directory: {resolved}"
    )
  # --clean-out-dir destroys completed evaluation results, which cost real API
  # budget to produce and are not in version control. Deleting is what the flag
  # is for, so this does not refuse -- but it states the damage first, so a
  # mistyped --out-dir leaves a trace in the run log.
  entries = sorted(resolved.iterdir())
  if entries:
    n_files = 0
    total = 0
    for dirpath, _dirnames, filenames in os.walk(resolved):
      for name in filenames:
        n_files += 1
        try:
          total += os.lstat(os.path.join(dirpath, name)).st_size
        except OSError:
          pass
    print(
        f"--clean-out-dir: removing {len(entries)} top-level entry(ies), "
        f"{n_files} file(s), {total / 1024 ** 2:.1f} MB from {resolved}"
    )

  removed = 0
  for entry in entries:
    if entry.is_dir() and not entry.is_symlink():
      shutil.rmtree(entry, ignore_errors=True)
    else:
      try:
        entry.unlink()
      except OSError:
        pass
    removed += 1
  print(f"Cleaned {removed} existing entry(ies) from out-dir: {resolved}")


def _interaction_log_file(log_dir: str | Path, stem: str) -> Path:
  """Path to a per-scenario/task interaction JSONL inside ``log_dir``.

  ``log_dir`` is created if needed (the '<out-dir>/log' folder by default). The
  filename is a sanitized ``stem`` (scenario slug / task id / trajectory stem)
  so each maps to one predictable '<stem>.interactions.jsonl'.
  """
  directory = Path(log_dir)
  directory.mkdir(parents=True, exist_ok=True)
  return directory / f"{_safe_filename_piece(stem)}.interactions.jsonl"


def _make_jsonl_sink(fh: Any) -> Callable[[dict[str, Any]], None]:
  def _sink(record: dict[str, Any]) -> None:
    fh.write(json.dumps(record, ensure_ascii=False) + "\n")

  return _sink


def _agent_captures_interactions(agent: BaseAgent) -> bool:
  return bool(getattr(agent, "_capture_interactions", False))


def collect_training_data(
    agent: BaseAgent,
    split: str = "train",
    out_dir: str | Path = "data/trajectories/expert",
    clean_out_dir: bool = False,
    task_data_dir: str | Path | None = None,
    mode: str = "multi",
    task_prefix: str | None = None,
    scenario_slugs: list[str] | None = None,
    domain: str | None = None,
    difficulty: int | None = None,
    limit: int | None = None,
    max_steps: int | None = None,
    max_stagnant_steps: int = 3,
    online_clean: bool | None = None,
    skip_incompatible_openclaw: bool = True,
    fallback_openclaw_network_to_mock: bool = False,
    strict_online_data: bool = True,
    online_openclaw_only: bool = False,
    resume_trajectories: bool = False,
    resume_successes_only: bool = False,
    retry_provider_failures_only: bool = False,
    inter_task_sleep_s: float = 0.0,
    record_step_eval: bool = False,
    parallel_workers: int = 1,
    verbose: bool = False,
    interaction_log: str | Path | None = None,
) -> Path:
  """Record trajectories from `agent` and save as JSONL.

  Each line of the output file is a JSON object with keys:
  ``task_id``, ``agent``, ``success``, ``score``, ``steps``,
  ``trajectory`` (list of step dicts), ``final_response``, ``duration_s``.

  Use the saved trajectories for:
  - **Behavior cloning**: train a policy to imitate successful expert episodes
  - **Offline RL**: reward-weighted regression, IQL, CQL, …
  - **Dataset inspection**: analyze failure modes and success patterns

  Parameters
  ----------
  out_dir:
      Directory where the JSONL file will be written.
      File name: ``{agent_name}_{split}.jsonl``.
  interaction_log:
      Directory for per-scenario interaction JSONL sidecars. Only used when
      the agent captures interactions. Defaults to ``<out_dir>/log`` when the
      agent captures interactions but no explicit directory is given.
  """
  from clasharness.core.task import load_task_ids

  out_path = Path(out_dir)
  # --clean-out-dir and the resume family are mutually exclusive, and silently
  # combining them is expensive: _clean_output_dir() wipes out_dir first, so the
  # resume logic below then finds no prior records, treats every task as "never
  # ran", and re-runs the entire split at full API cost -- while the results it
  # was meant to resume from have already been deleted. Fail loudly instead.
  _resume_flags = [
      name
      for name, on in (
          ("--retry-provider-failures-only", retry_provider_failures_only),
          ("--resume-trajectories", resume_trajectories),
          ("--resume-successes-only", resume_successes_only),
      )
      if on
  ]
  if clean_out_dir and _resume_flags:
    raise SystemExit(
        "--clean-out-dir cannot be combined with "
        + " / ".join(_resume_flags)
        + ":\n    --clean-out-dir deletes --out-dir before the resume logic "
        "reads it, so nothing is left to resume from and the whole split is "
        "re-run at full cost.\nPass --clean-out-dir to start fresh, or the "
        "resume flag to continue an existing run -- not both."
    )
  if clean_out_dir:
    _clean_output_dir(out_path)
  out_path.mkdir(parents=True, exist_ok=True)
  out_file = _trajectory_output_file(
      out_path, agent.name, split, scenario_slugs
  )

  interaction_log_dir = None
  if _agent_captures_interactions(agent):
    interaction_log_dir = (
        Path(interaction_log) if interaction_log else (out_path / "log")
    )
    interaction_log_dir.mkdir(parents=True, exist_ok=True)

  data_dir = Path(task_data_dir) if task_data_dir else None
  effective_max_steps = _resolve_max_steps(
      max_steps,
      split=split,
      task_prefix=task_prefix,
  )
  task_ids = load_task_ids(
      split, data_dir=data_dir, difficulty=difficulty, domain=domain
  )
  task_ids = _filter_task_ids_by_prefix(task_ids, task_prefix)
  task_ids = _filter_task_ids_by_scenario_slugs(
      task_ids, scenario_slugs, data_dir=data_dir
  )
  if limit:
    task_ids = task_ids[:limit]
  if online_openclaw_only:
    task_ids = _select_openclaw_only_task_ids(task_ids, data_dir=data_dir)
  if retry_provider_failures_only:
    provider_failed_task_ids = (
        _load_provider_failed_trajectory_task_ids_for_run(
            out_path,
            out_file,
            agent.name,
            split,
            scenario_slugs,
            run_task_ids=task_ids,
        )
    )
    # Rerun provider-failed tasks AND tasks that have no record yet (never ran /
    # didn't finish). Only skip tasks whose latest record is a non-provider
    # outcome (successes and genuine task failures), which should be left as-is.
    recorded_task_ids = _load_completed_trajectory_task_ids_for_run(
        out_path,
        out_file,
        agent.name,
        split,
        scenario_slugs,
        successes_only=False,
        run_task_ids=task_ids,
    )
    skip_task_ids = recorded_task_ids - provider_failed_task_ids
    task_ids = _select_retry_provider_failure_task_ids(
        task_ids, recorded_task_ids, provider_failed_task_ids
    )
    print(
        "Retrying provider-failed and unrun trajectories: "
        f"{len(task_ids)} task(s) selected from {out_path} "
        f"({len(provider_failed_task_ids)} provider-failed; "
        f"skipping {len(skip_task_ids)} completed record(s))"
    )
  if resume_trajectories or resume_successes_only:
    completed_task_ids = _load_completed_trajectory_task_ids_for_run(
        out_path,
        out_file,
        agent.name,
        split,
        scenario_slugs,
        successes_only=resume_successes_only,
        run_task_ids=task_ids,
    )
    if completed_task_ids:
      task_ids = [
          task_id for task_id in task_ids if task_id not in completed_task_ids
      ]
      scope = "successful" if resume_successes_only else "existing"
      print(
          "Resuming trajectory collection: skipping "
          f"{len(completed_task_ids)} {scope} task record(s) from {out_path}"
      )

  print(
      f"Collecting trajectories from '{agent.name}' on {len(task_ids)} tasks →"
      f" {out_file}"
  )

  n_success = 0
  run_config = _agent_run_config(agent)
  from clasharness.evaluation.llm_judge import active_judge_config
  judge_provider, judge_model = active_judge_config()
  summary = EvalSummary(
      agent_name=agent.name,
      split=split,
      total=0,
      passed=0,
      total_score=0.0,
      exec_mode=mode,
      llm_provider=run_config.get("provider"),
      llm_model=run_config.get("model"),
      llm_history_mode=run_config.get("history_mode"),
      judge_provider=judge_provider or None,
      judge_model=judge_model or None,
  )
  should_clean = _resolve_online_clean(mode, online_clean)
  backend_kwargs = _make_backend_kwargs(
      mode,
      skip_incompatible_openclaw,
      fallback_openclaw_network_to_mock,
      strict_online_data,
  )
  file_mode = (
      "a"
      if (
          resume_trajectories
          or resume_successes_only
          or retry_provider_failures_only
      )
      else "w"
  )
  if parallel_workers <= 1:
    with contextlib.ExitStack() as stack:
      fh = stack.enter_context(open(out_file, file_mode))
      inter_sink = None
      if interaction_log_dir is not None:
        inter_file = _interaction_log_file(
            interaction_log_dir, Path(out_file).stem
        )
        inter_fh = stack.enter_context(
            open(inter_file, file_mode, encoding="utf-8")
        )
        inter_sink = _make_jsonl_sink(inter_fh)
        print(f"Interaction log → {inter_file}")
      for i, task_id in enumerate(task_ids, 1):
        if i > 1 and inter_task_sleep_s > 0:
          time.sleep(inter_task_sleep_s)
        if not verbose:
          print(f"  [{i:4d}/{len(task_ids)}] {task_id:<50}", end="\r", flush=True)

        result = run_episode(
            task_id,
            agent,
            task_data_dir=data_dir,
            mode=mode,
            max_steps=effective_max_steps,
            max_stagnant_steps=max_stagnant_steps,
            record_trajectory=True,
            online_clean=should_clean,
            backend_kwargs=backend_kwargs,
            record_step_eval=record_step_eval or verbose,
            verbose=verbose,
            interaction_sink=inter_sink,
        )

        if result.success:
          n_success += 1
        elif result.error:
          print(f"\n[{task_id}] Failed with error: {result.error}\n")

        _update_summary(summary, result, task_id, data_dir)
        fh.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

    if not verbose:
      print()

    print(
        f"Saved {len(task_ids)} trajectories ({n_success} successful) to"
        f" {out_file}"
    )

    # For resume/retry runs the in-invocation `summary` only covers the subset of
    # tasks re-run this time; rebuild over the whole out-dir so the saved report
    # reflects the full cohort (latest record per task_id).
    report_summary = summary
    if file_mode == "a":
      report_summary = _aggregate_summary_over_out_dir(
          summary, out_path, agent.name, data_dir
      )
    report_summary.print_report()

    summary_file = Path(out_file).with_suffix(".summary")
    with open(summary_file, "w", encoding="utf-8") as f:
      with contextlib.redirect_stdout(f):
        report_summary.print_report()
    print(f"Summary report saved → {summary_file}")

    _generate_evaluation_visualizer(out_path)
  else:
    # Group tasks by scenario slug
    groups = {}
    for task_id in task_ids:
      slug = _extract_scenario_slug_from_task_id(task_id)
      groups.setdefault(slug, []).append(task_id)

    print(f"Starting parallel trajectory collection across {len(groups)} scenario groups with {parallel_workers} workers...")

    # Thread task handler
    def _run_scenario_group(slug: str, group_tasks: list[str]) -> tuple[str, int, int, Path, list[tuple[str, Any]]]:
      group_out_file = out_path / f"{slug}.jsonl"
      group_n_success = 0
      group_results = []
      
      # Clone agent for thread safety
      thread_agent = _clone_agent(agent)
      
      with contextlib.ExitStack() as stack:
        fh = stack.enter_context(
            open(group_out_file, file_mode, encoding="utf-8")
        )
        inter_sink = None
        if interaction_log_dir is not None:
          inter_fh = stack.enter_context(
              open(
                  _interaction_log_file(interaction_log_dir, slug),
                  file_mode,
                  encoding="utf-8",
              )
          )
          inter_sink = _make_jsonl_sink(inter_fh)
        for idx, task_id in enumerate(group_tasks, 1):
          if idx > 1 and inter_task_sleep_s > 0:
            time.sleep(inter_task_sleep_s)
          
          print(f"[{slug}] Running task {idx}/{len(group_tasks)}: {task_id}")
          
          result = run_episode(
              task_id,
              thread_agent,
              task_data_dir=data_dir,
              mode=mode,
              max_steps=effective_max_steps,
              max_stagnant_steps=max_stagnant_steps,
              record_trajectory=True,
              online_clean=should_clean,
              backend_kwargs=backend_kwargs,
              record_step_eval=record_step_eval or verbose,
              verbose=False, # Force False to prevent stdout interlacing
              interaction_sink=inter_sink,
          )
          if result.success:
            group_n_success += 1
          elif result.error:
            print(f"[{slug}] Task {task_id} failed with error: {result.error}")
          
          fh.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
          group_results.append((task_id, result))
          
      return slug, len(group_tasks), group_n_success, group_out_file, group_results

    # Run in parallel thread pool
    from concurrent.futures import ThreadPoolExecutor, as_completed
    total_n_success = 0
    total_run = 0
    
    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
      futures = {
          executor.submit(_run_scenario_group, slug, group_tasks): slug
          for slug, group_tasks in groups.items()
      }
      
      for fut in as_completed(futures):
        slug = futures[fut]
        try:
          slug, count, successes, group_file, group_results = fut.result()
          total_n_success += successes
          total_run += count
          print(f"[{slug}] Completed: {successes}/{count} succeeded. Saved to {group_file}")
          
          # Aggregate into main summary
          for task_id, res in group_results:
            _update_summary(summary, res, task_id, data_dir)
            
        except Exception as exc:
          print(f"Warning: Scenario group {slug} failed: {exc}")
          import traceback
          traceback.print_exc()

    print(f"\nCompleted parallel run: {total_n_success}/{total_run} succeeded total.")

    # For resume/retry runs the in-invocation `summary` only covers the subset of
    # tasks re-run this time; rebuild over the whole out-dir so the saved report
    # reflects the full cohort (latest record per task_id).
    report_summary = summary
    if file_mode == "a":
      report_summary = _aggregate_summary_over_out_dir(
          summary, out_path, agent.name, data_dir
      )
    report_summary.print_report()

    summary_file = out_path / "parallel_run.summary"
    with open(summary_file, "w", encoding="utf-8") as f:
      with contextlib.redirect_stdout(f):
        report_summary.print_report()
    print(f"Summary report saved → {summary_file}")

    _generate_evaluation_visualizer(out_path)

  return out_file


# ===========================================================================
# Single-task interactive demo
# ===========================================================================


def run_demo(
    task_id: str,
    agent: BaseAgent,
    task_data_dir: str | Path | None = None,
    mode: str = "multi",
    max_steps: int | None = None,
    max_stagnant_steps: int = 3,
    online_clean: bool | None = None,
    skip_incompatible_openclaw: bool = True,
    fallback_openclaw_network_to_mock: bool = False,
    strict_online_data: bool = True,
    record_step_eval: bool = False,
    interaction_log: str | Path | None = None,
) -> None:
  """Run a single episode with full verbose output."""
  with contextlib.ExitStack() as stack:
    inter_sink = None
    if interaction_log is not None and _agent_captures_interactions(agent):
      inter_file = _interaction_log_file(interaction_log, task_id)
      inter_fh = stack.enter_context(
          open(inter_file, "w", encoding="utf-8")
      )
      inter_sink = _make_jsonl_sink(inter_fh)
      print(f"Interaction log → {inter_file}")
    result = run_episode(
        task_id,
        agent,
        task_data_dir=task_data_dir,
        mode=mode,
        max_steps=_resolve_max_steps(max_steps, task_id=task_id),
        max_stagnant_steps=max_stagnant_steps,
        record_trajectory=True,
        online_clean=_resolve_online_clean(mode, online_clean),
        backend_kwargs=_make_backend_kwargs(
            mode,
            skip_incompatible_openclaw,
            fallback_openclaw_network_to_mock,
            strict_online_data,
        ),
        record_step_eval=record_step_eval,
        verbose=True,
        interaction_sink=inter_sink,
    )
  print(
      f"\nFinal: success={result.success}  score={result.score:.2f}  "
      f"steps={result.steps}  time={result.duration_s:.1f}s"
  )
  if result.error:
    print(f"Error: {result.error}")


# ===========================================================================
# CLI entry point
# ===========================================================================


def _resolve_vertex_project(explicit: str | None = None) -> str:
  """Resolve the Vertex AI project, mirroring the agent's fallback chain.

  Shared by the LLM agent and the LLM judge so they never diverge onto
  different projects. Returns ``""`` when nothing is configured: Vertex is the
  optional backend, and a project belongs to whoever is running, so guessing
  one would only turn a missing setting into a permission error against
  somebody else's project.
  """
  return (
      explicit or
      os.environ.get("VERTEX_PROJECT") or
      os.environ.get("GOOGLE_CLOUD_PROJECT") or
      os.environ.get("GEMINI_PROJECT") or
      ""
  )


def _resolve_vertex_location(explicit: str | None = None) -> str:
  """Resolve the Vertex AI location, mirroring the agent's fallback chain."""
  return (
      explicit or
      os.environ.get("VERTEX_LOCATION") or
      os.environ.get("GOOGLE_CLOUD_REGION") or
      os.environ.get("GEMINI_LOCATION") or
      "us"
  )


def _validate_judge_config() -> None:
  """Resolve + validate the LLM judge backend loudly, before evaluation starts.

  Reads the judge selection out of the environment (already populated from the
  CLI flags) and fails fast on a misconfiguration that would otherwise surface
  only as silent ``ok=False`` semantic checks midway through a run. A judge that
  is simply unconfigured is a loud warning (grading degrades to failed checks),
  not a hard error.
  """
  from clasharness.evaluation.llm_judge import resolve_provider

  provider = resolve_provider()
  model = os.environ.get("LLM_JUDGE_MODEL", "")
  project = os.environ.get("VERTEX_PROJECT", "")
  location = os.environ.get("VERTEX_LOCATION", "")

  if not provider:
    print(
        "WARNING: no LLM judge configured -- semantic checks "
        "(field_semantic_match / LLMChecker) will all record ok=False. Set "
        "--llm-judge-provider, or one of GEMINI_API_KEY / ANTHROPIC_API_KEY / "
        "OPENAI_API_KEY / VERTEX_PROJECT.",
        file=sys.stderr,
    )
    return

  # Gemini has two backends: an API key (Developer API) or a project (Vertex).
  # Only 'vertex-anthropic' is Vertex-only.
  if provider == "gemini" and not project and not os.environ.get(
      "GEMINI_API_KEY"
  ):
    raise SystemExit(
        "train_and_eval: error: judge provider 'gemini' needs either "
        "GEMINI_API_KEY (Gemini Developer API) or a Vertex project. Set "
        "GEMINI_API_KEY, or pass --vertex-project / set VERTEX_PROJECT."
    )
  if provider == "vertex-anthropic" and not project:
    raise SystemExit(
        "train_and_eval: error: judge provider 'vertex-anthropic' needs a "
        "Vertex project but VERTEX_PROJECT is empty. Pass --vertex-project or "
        "set VERTEX_PROJECT / GOOGLE_CLOUD_PROJECT."
    )
  if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
    raise SystemExit(
        "train_and_eval: error: judge provider 'anthropic' needs "
        "ANTHROPIC_API_KEY (direct Anthropic API)."
    )
  if provider == "vertex-anthropic":
    if not model:
      raise SystemExit(
          "train_and_eval: error: judge provider 'vertex-anthropic' requires an "
          "explicit --llm-judge-model (e.g. 'claude-sonnet-4@20250514' or "
          "'claude-opus-4-8'); there is no default."
      )
    # The model id is passed through to AnthropicVertex verbatim, exactly like
    # the agent, so both '@'-form publisher ids and plain dash ids are accepted
    # -- whether one resolves is left to the endpoint (no format check here).
    if location == "us":
      print(
          "WARNING: --vertex-location 'us' does not serve Claude on Vertex. Pass "
          "a Claude-capable region such as 'us-east5'.",
          file=sys.stderr,
      )
    elif not location:
      print(
          "Judge: vertex-anthropic location unset; using Claude default "
          "'us-east5'.",
          file=sys.stderr,
      )
  if provider == "bedrock-gpt":
    if not os.environ.get("BEDROCK_OPENAI_API_KEY"):
      raise SystemExit(
          "train_and_eval: error: judge provider 'bedrock-gpt' needs "
          "BEDROCK_OPENAI_API_KEY (GPT on Bedrock via the OpenAI Responses API)."
      )
    if not model:
      raise SystemExit(
          "train_and_eval: error: judge provider 'bedrock-gpt' requires an "
          "explicit --llm-judge-model with an 'openai.' prefix (e.g. "
          "'openai.gpt-5.5'); there is no default."
      )
    if not (model.startswith("openai.") or model.startswith("gpt-")):
      print(
          f"WARNING: --llm-judge-model {model!r} lacks an 'openai.'/'gpt-' "
          "prefix -- Bedrock GPT ids look like 'openai.gpt-5.5'.",
          file=sys.stderr,
      )
  if provider == "openai":
    if not os.environ.get("OPENAI_API_KEY"):
      raise SystemExit(
          "train_and_eval: error: judge provider 'openai' needs OPENAI_API_KEY "
          "(GPT on OpenAI's official Responses API)."
      )
    if not model:
      raise SystemExit(
          "train_and_eval: error: judge provider 'openai' requires an explicit "
          "--llm-judge-model (e.g. 'gpt-5.5'); there is no default."
      )
    # Official ids carry no publisher prefix; the judge strips a Bedrock-style
    # one rather than 404ing, so this is a nudge, not an error.
    if model.startswith("openai."):
      print(
          f"WARNING: --llm-judge-model {model!r} carries Bedrock's 'openai.' "
          f"publisher prefix; OpenAI's own API wants {model[len('openai.'):]!r}."
          " The prefix will be stripped.",
          file=sys.stderr,
      )

  print(
      f"Judge: {provider}/{model or '(provider default)'} "
      f"@ project={project or '(n/a)'} location={location or '(default)'}"
  )


def _parse_chat_template_kwargs(raw: str | None) -> dict[str, Any] | None:
  """Parse --llm-chat-template-kwargs, rejecting anything not a JSON object.

  Validated at startup rather than at the first request because the failure is
  otherwise invisible: an endpoint that receives a malformed or unexpected
  template variable answers HTTP 200 and silently ignores it, so a whole run
  would complete at the wrong setting and look successful.
  """
  if raw is None or not raw.strip():
    return None
  try:
    parsed = json.loads(raw)
  except ValueError as exc:
    raise ValueError(
        f"--llm-chat-template-kwargs is not valid JSON: {exc}"
    ) from exc
  if not isinstance(parsed, dict):
    raise ValueError(
        "--llm-chat-template-kwargs must be a JSON object, got "
        f"{type(parsed).__name__}"
    )
  return parsed


def _build_agent(
    name: str,
    model: str,
    llm_provider: str = "anthropic",
    llm_base_url: str | None = None,
    llm_api_key_env: str | None = None,
    llm_history_mode: str = "full",
    llm_temperature: float = 0.0,
    llm_max_tokens: int = 256,
    llm_timeout_s: int = 25,
    llm_request_retries: int = 1,
    llm_empty_reply_retries: int = 5,
    llm_retry_backoff_s: float = 1.5,
    llm_transient_max_backoff_s: float = 12.0,
    llm_deadline_retry_backoff_s: float = 60.0,
    llm_deadline_max_backoff_s: float = 60.0,
    multimodal_input_mode: str = "path",
    max_steps_hint: int = DEFAULT_MAX_STEPS,
    verbose: bool = False,
    vertex_project: str | None = None,
    vertex_location: str | None = None,
    anthropic_max_image_edge: str | int = "auto",
    gemini_max_image_edge: int | None = None,
    gemini_media_resolution: str | None = None,
    openai_max_image_edge: int | None = None,
    capture_thinking: bool = False,
    capture_interactions: bool = False,
    llm_allow_compact_fallback: bool = True,
    llm_tool_mode: str = "native",
    llm_tool_schema: str = "native",
    llm_native_reasoning: bool = DEFAULT_LLM_NATIVE_REASONING,
    llm_reasoning_effort: str | None = None,
    llm_chat_template_kwargs: dict[str, Any] | None = None,
    llm_replay_chat_reasoning: bool = False,
) -> BaseAgent:
  if name == "expert":
    return ExpertAgent()
  if name == "random":
    return RandomAgent()
  if name == "rule":
    return RuleAgent()
  if name == "llm":
    resolved_project = _resolve_vertex_project(vertex_project)
    resolved_location = _resolve_vertex_location(vertex_location)
    return LLMAgent(
        provider=llm_provider,
        model=model,
        base_url=llm_base_url,
        api_key_env=llm_api_key_env,
        history_mode=llm_history_mode,
        temperature=llm_temperature,
        response_max_tokens=llm_max_tokens,
        request_timeout_s=llm_timeout_s,
        request_retries=llm_request_retries,
        empty_reply_retries=llm_empty_reply_retries,
        retry_backoff_s=llm_retry_backoff_s,
        transient_max_backoff_s=llm_transient_max_backoff_s,
        deadline_retry_backoff_s=llm_deadline_retry_backoff_s,
        deadline_max_backoff_s=llm_deadline_max_backoff_s,
        multimodal_input_mode=multimodal_input_mode,
        max_steps_hint=max_steps_hint,
        verbose=verbose,
        vertex_project=resolved_project,
        vertex_location=resolved_location,
        anthropic_max_image_edge=anthropic_max_image_edge,
        gemini_max_image_edge=gemini_max_image_edge,
        gemini_media_resolution=gemini_media_resolution,
        openai_max_image_edge=openai_max_image_edge,
        capture_thinking=capture_thinking,
        capture_interactions=capture_interactions,
        allow_compact_fallback=llm_allow_compact_fallback,
        tool_mode=llm_tool_mode,
        tool_schema_mode=llm_tool_schema,
        native_reasoning=llm_native_reasoning,
        reasoning_effort=llm_reasoning_effort,
        chat_template_kwargs=llm_chat_template_kwargs,
        replay_chat_reasoning=llm_replay_chat_reasoning,
    )
  raise ValueError(
      f"Unknown agent: {name!r}. Choose from: expert, random, rule, llm"
  )


def _select_openclaw_only_task_ids(
    task_ids: list[str],
    data_dir: Path | None = None,
) -> list[str]:
  """Keep only tasks whose GT commands include at least one `openclaw *` step."""
  selected: list[str] = []
  for task_id in task_ids:
    try:
      task = load_task(task_id, data_dir=data_dir)
      cmds = (
          task.ground_truth.solution_commands
          if task and task.ground_truth
          else []
      )
      if any(c.strip().startswith("openclaw ") for c in cmds):
        selected.append(task_id)
    except Exception:
      continue
  return selected


def _filter_task_ids_by_prefix(
    task_ids: list[str], task_prefix: str | None
) -> list[str]:
  if not task_prefix:
    return task_ids
  return [task_id for task_id in task_ids if task_id.startswith(task_prefix)]


def _filter_task_ids_by_scenario_slugs(
    task_ids: list[str],
    scenario_slugs: list[str] | None,
    data_dir: Path | None = None,
) -> list[str]:
  """Keep only tasks whose public scenario slug matches one of the requested seeds."""
  if not scenario_slugs:
    return task_ids
  wanted = {slug for slug in scenario_slugs if slug}
  if not wanted:
    return task_ids

  selected: list[str] = []
  for task_id in task_ids:
    # PRO LLM task IDs include the seed slug. This fallback keeps filtering
    # usable even if a task file cannot be loaded.
    if any(task_id.startswith(f"pro_llm_{slug}_q") for slug in wanted):
      selected.append(task_id)
      continue
    try:
      task = load_task(task_id, data_dir=data_dir)
    except Exception:
      continue
    public = _task_public_dict(task)
    task_slug = public.get("scenario_slug") or public.get(
        "hard_decision_scenario"
    )
    if isinstance(task_slug, str):
      # conflict_vlm slugs are versioned (e.g. "..._v01"); accept the base slug
      # so "--scenario-slug foo" selects every "foo_vNN" (matches the PRO-data
      # UX, where base slugs already work via the pro_llm_{slug}_q fallback
      # above). Strip only a trailing version suffix and match the base exactly
      # so "it_deployment_gate" cannot leak into "it_deployment_gate_extended".
      task_base = re.sub(r"_v\d+$", "", task_slug)
      if task_slug in wanted or task_base in wanted:
        selected.append(task_id)
  return selected


def _trajectory_output_file(
    out_path: Path,
    agent_name: str,
    split: str,
    scenario_slugs: list[str] | None,
) -> Path:
  if not scenario_slugs:
    suffix = split
  elif len(scenario_slugs) == 1:
    suffix = f"{split}_{_safe_filename_piece(scenario_slugs[0])}"
  else:
    suffix = f"{split}_selected_scenarios"
  return out_path / f"{agent_name}_{suffix}.jsonl"



def _extract_scenario_slug_from_task_id(task_id: str) -> str:
  """Extracts the versioned scenario slug from a task ID."""
  if task_id.startswith("conflict_vlm_"):
    parts = task_id.split("_")
    if len(parts) >= 4 and parts[-1].startswith("r") and parts[-1][1:].isdigit():
      return "_".join(parts[2:-1])
  return task_id.split("_")[0]


def _clone_agent(agent: BaseAgent) -> BaseAgent:
  """Clones the agent instance to ensure thread-safety in parallel execution."""
  import copy
  new_agent = copy.copy(agent)
  if hasattr(new_agent, "_messages"):
    new_agent._messages = []
  if hasattr(new_agent, "_history"):
    new_agent._history = []
  if hasattr(new_agent, "_client"):
    new_agent._client = None
  return new_agent


def _safe_filename_piece(value: str) -> str:
  cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
  return cleaned.strip("._-") or "scenario"


def _load_completed_trajectory_task_ids(
    path: Path, *, successes_only: bool = False
) -> set[str]:
  if not path.exists():
    return set()
  completed: set[str] = set()
  with path.open(encoding="utf-8") as fh:
    for line in fh:
      raw = line.strip()
      if not raw:
        continue
      try:
        record = json.loads(raw)
      except json.JSONDecodeError:
        continue
      if successes_only and not bool(record.get("success")):
        continue
      task_id = record.get("task_id")
      if isinstance(task_id, str) and task_id:
        completed.add(task_id)
  return completed


def _trajectory_record_is_provider_failed(record: dict[str, Any]) -> bool:
  if bool(record.get("success")):
    return False
  if bool(record.get("provider_impacted")):
    return True
  if str(record.get("error_type") or "") == "provider_failure":
    return True
  text = " ".join(
      str(record.get(key) or "")
      for key in ("error", "stop_reason", "final_response")
  ).lower()
  return any(
      marker in text
      for marker in (
          "http 429",
          "rate limit",
          "ratelimit",
          "too many tokens",
          "provider failure",
          "openai-compatible request failed",
          # A terminal empty-reply stall is the provider returning nothing, not
          # a solver mistake -- see _classify_error_type. Matched here on
          # ``stop_reason`` (already folded into ``text`` above) rather than
          # relying solely on ``error_type`` so that trajectory records written
          # BEFORE that reclassification -- which carry the old
          # ``error_type="logical_failure"`` on disk -- also become eligible for
          # --retry-provider-failures-only without a full re-run.
          "generation_stall_empty_reply",
      )
  )


def _load_latest_trajectory_records(path: Path) -> dict[str, dict[str, Any]]:
  latest: dict[str, dict[str, Any]] = {}
  if not path.exists():
    return latest
  with path.open(encoding="utf-8") as fh:
    for line in fh:
      raw = line.strip()
      if not raw:
        continue
      try:
        record = json.loads(raw)
      except json.JSONDecodeError:
        continue
      task_id = record.get("task_id")
      if isinstance(task_id, str) and task_id:
        latest[task_id] = record
  return latest


def _load_provider_failed_trajectory_task_ids(path: Path) -> set[str]:
  return {
      task_id
      for task_id, record in _load_latest_trajectory_records(path).items()
      if _trajectory_record_is_provider_failed(record)
  }


def _run_trajectory_files(
    out_path: Path,
    out_file: Path,
    agent_name: str,
    split: str,
    scenario_slugs: list[str] | None,
    run_task_ids: list[str] | None,
) -> list[Path]:
  """Return the trajectory JSONL files a run may have written.

  Covers both naming schemes:
    * Sequential / single-worker: ``{agent}_{split}*.jsonl`` (see
      ``_trajectory_output_file`` and the ``parallel_workers <= 1`` path).
    * Parallel: one ``{scenario_slug}.jsonl`` per scenario group, written by
      ``_run_scenario_group``. These are keyed only by the scenario slug (no
      agent/split prefix), so they are matched from ``run_task_ids`` to avoid
      pulling in other agents' or other runs' group files.
  """
  files: list[Path] = []
  seen: set[Path] = set()

  def _add(path: Path) -> None:
    if path not in seen and path.exists():
      seen.add(path)
      files.append(path)

  # Always consider the primary output file for this run.
  _add(out_file)

  # Sequential naming: pick up sibling per-scenario files unless a specific
  # scenario set was requested (in which case out_file already targets it).
  if not scenario_slugs:
    for path in sorted(out_path.glob(f"{agent_name}_{split}*.jsonl")):
      _add(path)

  # Parallel naming: one file per scenario slug of THIS run's task IDs.
  if run_task_ids:
    slugs = sorted(
        {_extract_scenario_slug_from_task_id(task_id) for task_id in run_task_ids}
    )
    for slug in slugs:
      _add(out_path / f"{slug}.jsonl")

  return files


def _load_completed_trajectory_task_ids_for_run(
    out_path: Path,
    out_file: Path,
    agent_name: str,
    split: str,
    scenario_slugs: list[str] | None,
    *,
    successes_only: bool = False,
    run_task_ids: list[str] | None = None,
) -> set[str]:
  """Load completed task IDs for resume, including per-scenario JSONL files."""
  completed: set[str] = set()
  for path in _run_trajectory_files(
      out_path, out_file, agent_name, split, scenario_slugs, run_task_ids
  ):
    completed.update(
        _load_completed_trajectory_task_ids(path, successes_only=successes_only)
    )
  return completed


def _load_provider_failed_trajectory_task_ids_for_run(
    out_path: Path,
    out_file: Path,
    agent_name: str,
    split: str,
    scenario_slugs: list[str] | None,
    *,
    run_task_ids: list[str] | None = None,
) -> set[str]:
  """Load task IDs whose latest trajectory record is provider-impacted."""
  latest: dict[str, dict[str, Any]] = {}
  for path in _run_trajectory_files(
      out_path, out_file, agent_name, split, scenario_slugs, run_task_ids
  ):
    latest.update(_load_latest_trajectory_records(path))
  return {
      task_id
      for task_id, record in latest.items()
      if _trajectory_record_is_provider_failed(record)
  }


def _select_retry_provider_failure_task_ids(
    run_task_ids: list[str],
    recorded_task_ids: set[str],
    provider_failed_task_ids: set[str],
) -> list[str]:
  """Task IDs to (re)run for ``--retry-provider-failures-only``.

  Reruns provider-failed tasks AND tasks with no record yet (never ran /
  didn't finish), preserving the input order. Skips only tasks whose latest
  record is a non-provider outcome (successes and genuine task failures).
  """
  skip_task_ids = recorded_task_ids - provider_failed_task_ids
  return [task_id for task_id in run_task_ids if task_id not in skip_task_ids]


def _episode_result_from_record(record: dict[str, Any]) -> EpisodeResult:
  """Reconstruct an EpisodeResult from a persisted trajectory JSONL record.

  Mirrors ``EpisodeResult.to_dict()`` (note the ``"agent"`` -> ``agent_name``
  remap) so a summary rebuilt from disk matches one accumulated in-memory,
  including the fields the diagnostics rely on (``stop_reason``, ``trajectory``,
  ``score``, ``date_anchor_mismatch``, ...).
  """
  return EpisodeResult(
      task_id=str(record.get("task_id") or ""),
      agent_name=str(record.get("agent") or ""),
      success=bool(record.get("success")),
      score=float(record.get("score") or 0.0),
      steps=int(record.get("steps") or 0),
      duration_s=float(record.get("duration_s") or 0.0),
      trajectory=record.get("trajectory") or [],
      final_response=str(record.get("final_response") or ""),
      error=str(record.get("error") or ""),
      error_type=str(record.get("error_type") or ""),
      provider_impacted=bool(record.get("provider_impacted")),
      compact_fallback=bool(record.get("compact_fallback")),
      stop_reason=str(record.get("stop_reason") or ""),
      date_anchor_mismatch=bool(record.get("date_anchor_mismatch")),
      mode=str(record.get("mode") or ""),
      metadata=record.get("metadata") or {},
  )


def _aggregate_summary_over_out_dir(
    template: EvalSummary,
    out_path: Path,
    agent_name: str,
    data_dir: Path | None,
) -> EvalSummary:
  """Rebuild an EvalSummary over EVERY trajectory record in ``out_path``.

  Resume/retry runs only execute a subset of tasks, so the in-invocation summary
  (and the ``.summary`` file it writes) would report just that subset. This
  re-derives the full cohort from disk, using the latest record per ``task_id``
  across both the sequential ``{agent}_{split}*.jsonl`` and the parallel
  ``{slug}.jsonl`` files, so the saved summary reflects all tasks. Header/run
  metadata (agent, split, mode, solver + judge provider/model) is copied from
  ``template``; records written by a different agent into the same directory are
  skipped.
  """
  full = EvalSummary(
      agent_name=template.agent_name,
      split=template.split,
      total=0,
      passed=0,
      total_score=0.0,
      exec_mode=template.exec_mode,
      llm_provider=template.llm_provider,
      llm_model=template.llm_model,
      llm_history_mode=template.llm_history_mode,
      judge_provider=template.judge_provider,
      judge_model=template.judge_model,
  )
  latest: dict[str, dict[str, Any]] = {}
  for path in sorted(out_path.glob("*.jsonl")):
    latest.update(_load_latest_trajectory_records(path))
  for task_id, record in sorted(latest.items()):
    record_agent = str(record.get("agent") or "")
    if agent_name and record_agent and record_agent != agent_name:
      continue
    _update_summary(full, _episode_result_from_record(record), task_id, data_dir)
  return full


def _resolve_online_clean(mode: str, online_clean: bool | None) -> bool:
  if online_clean is None:
    return mode in {"real", "hybrid"}
  return online_clean


def _make_backend_kwargs(
    mode: str,
    skip_incompatible_openclaw: bool,
    fallback_openclaw_network_to_mock: bool,
    strict_online_data: bool,
) -> dict[str, Any] | None:
  if mode not in {"real", "hybrid"}:
    return None
  kwargs: dict[str, Any] = {
      "skip_incompatible_openclaw": skip_incompatible_openclaw,
  }
  if fallback_openclaw_network_to_mock:
    kwargs["fallback_openclaw_network_to_mock"] = True
  if strict_online_data:
    kwargs["strict_online_data"] = True
  return kwargs


def _preflight_mode(mode: str, status_timeout_s: int = 8) -> None:
  """Fail fast for real/hybrid runs when openclaw runtime is not ready."""
  if mode not in {"real", "hybrid"}:
    return

  cli_path = shutil.which("openclaw")
  if not cli_path:
    raise RuntimeError(
        "Mode requires `openclaw` in PATH, but it was not found. "
        "Install OpenClaw CLI and ensure `which openclaw` succeeds."
    )

  try:
    proc = subprocess.run(
        ["openclaw", "status", "--json"],
        capture_output=True,
        text=True,
        timeout=status_timeout_s,
    )
  except subprocess.TimeoutExpired as exc:
    raise RuntimeError(
        f"`openclaw status --json` timed out after {status_timeout_s}s. "
        "Check gateway health with `openclaw gateway status`."
    ) from exc
  except Exception as exc:  # pragma: no cover - defensive
    raise RuntimeError(
        f"Failed to run `openclaw status --json`: {exc}"
    ) from exc

  if proc.returncode != 0:
    details = (proc.stderr or proc.stdout or "").strip()
    detail_msg = f" Details: {details}" if details else ""
    raise RuntimeError(
        "OpenClaw preflight failed: `openclaw status --json` did not succeed. "
        "Start/check the gateway (`openclaw gateway status`) and profile setup."
        f"{detail_msg}"
    )


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Train / evaluate agents with clasharness",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=__doc__,
  )
  parser.add_argument(
      "--agent",
      default="expert",
      choices=["expert", "random", "rule", "llm"],
      help="Agent type (default: expert)",
  )
  parser.add_argument(
      "--split",
      default="dev",
      help=(
          "Dataset split or dataset list name under clasharness/data/datasets"
          " (default: dev)"
      ),
  )
  parser.add_argument(
      "--mode",
      default="multi",
      choices=BACKEND_MODE_CHOICES,
      help="Backend mode (default: multi; martin aliases multi)",
  )
  parser.add_argument(
      "--domain", default=None, help="Filter by domain, e.g. 'calendar'"
  )
  parser.add_argument(
      "--task-prefix",
      default=None,
      help="Filter task IDs by prefix, e.g. 'hard_decision_workflow_'",
  )
  parser.add_argument(
      "--scenario-slug",
      action="append",
      default=None,
      help=(
          "Filter by scenario/seed slug. Can be passed multiple times. Useful"
          " for PRO data, e.g. --scenario-slug privacy_deletion_request_hold."
      ),
  )
  parser.add_argument(
      "--difficulty",
      type=int,
      default=None,
      choices=[1, 2, 3],
      help="Filter by difficulty level 1/2/3",
  )
  parser.add_argument(
      "--limit", type=int, default=None, help="Max number of tasks to run"
  )
  parser.add_argument(
      "--max-steps",
      type=int,
      default=15,
      help="Max steps per episode (default: 15)",
  )
  parser.add_argument(
      "--max-stagnant-steps",
      type=int,
      default=3,
      help=(
          "Stop an episode after this many identical no-progress transitions "
          "(same command, stdout, stderr, exit code). Default: 3"
      ),
  )
  parser.add_argument(
      "--parallel-workers",
      type=int,
      default=1,
      help="Number of parallel scenario groups to evaluate (default: 1)",
  )
  parser.add_argument(
      "--model",
      default="claude-haiku-4-5-20251001",
      help=(
          "LLM model ID (only used with --agent llm). When"
          " --llm-provider=openai and --model starts with 'claude-', the client"
          " defaults to a local LiteLLM proxy at http://127.0.0.1:4000/v1 and"
          " reads LITELLM_PROXY_KEY unless overridden."
      ),
  )
  parser.add_argument(
      "--llm-provider",
      default="anthropic",
      choices=[
          "anthropic",
          "openai",
          "gemini",
          "vertex-anthropic",
          "vertex-xai",
          "bedrock-gpt",
      ],
      help=(
          "LLM API provider for --agent llm (default: anthropic). "
          "'anthropic' and 'openai' are the vendors' official APIs "
          "(ANTHROPIC_API_KEY / OPENAI_API_KEY) and need no cloud account. "
          "'gemini' serves Gemini models over the official Gemini API when "
          "GEMINI_API_KEY is set, and via Vertex AI (ADC + a Vertex project) "
          "otherwise. The remaining three are cloud-hosted routes to the same "
          "model families, which is how the paper's runs were made: "
          "'vertex-anthropic' "
          "serves Claude models via Vertex AI (Anthropic Model Garden) using "
          "ADC -- pass a Vertex model id like 'claude-sonnet-4@20250514' and a "
          "Claude-capable --vertex-location such as 'us-east5'. 'vertex-xai' "
          "serves xAI Grok via Vertex AI's OpenAI-compatible endpoint using ADC "
          "-- pass a model id like 'grok-4.20-reasoning' (the 'xai/' prefix is "
          "added automatically) and use --vertex-location global. 'bedrock-gpt' "
          "serves GPT on AWS Bedrock via its OpenAI Responses-API-compatible "
          "endpoint (default base URL "
          "https://bedrock-mantle.us-east-2.api.aws/openai/v1, override with "
          "--llm-base-url; reads BEDROCK_OPENAI_API_KEY) -- pass an explicit "
          "--model like 'openai.gpt-5.5'."
      ),
  )
  parser.add_argument(
      "--llm-base-url",
      default=None,
      help=(
          "Base URL for OpenAI-compatible chat completions API. Example:"
          " https://api.openai.com/v1 or a compatible proxy. Claude Bedrock"
          " proxy models can omit this and use LITELLM_PROXY_BASE_URL (default:"
          " http://127.0.0.1:4000/v1). '--llm-provider bedrock-gpt' defaults to"
          " the us-east-2 mantle endpoint; set this to target another region."
      ),
  )
  parser.add_argument(
      "--llm-api-key-env",
      default=None,
      help=(
          "Environment variable containing the LLM API key. "
          "Defaults to ANTHROPIC_API_KEY or OPENAI_API_KEY based on provider. "
          "Claude Bedrock proxy models default to LITELLM_PROXY_KEY; "
          "'--llm-provider bedrock-gpt' defaults to BEDROCK_OPENAI_API_KEY."
      ),
  )
  parser.add_argument(
      "--llm-history-mode",
      default="full",
      choices=["auto", "full", "summary"],
      help=(
          "How to construct multi-turn message history for LLM requests. 'full'"
          " sends full history by default, 'summary' always uses the compact"
          " history summary, and 'auto' uses the existing automatic trim logic."
      ),
  )
  parser.add_argument(
      "--llm-tool-mode",
      default="native",
      choices=["native", "text"],
      help=(
          "How the LLM selects each action. 'native' (default) uses the"
          " provider's function/tool-calling API with tool_choice=auto - the"
          " model may also answer in plain text, which is kept in the history"
          " rather than discarded; the `done` tool signals completion - and"
          " replays a typed tool_call/tool_result history. 'text' uses the"
          " legacy ReAct path (the model emits a plain command line parsed"
          " from text). Native mode uses full history regardless of"
          " --llm-history-mode."
      ),
  )
  parser.add_argument(
      "--llm-tool-schema",
      default="native",
      choices=["native", "unified"],
      help=(
          "Argument-schema dialect for native tool-calling (ignored in text"
          " mode). 'native' (default) sends each provider its idiomatic schema"
          " (Gemini's uppercase OpenAPI form). 'unified' sends every provider"
          " the identical canonical JSON-Schema (Gemini via"
          " parameters_json_schema), removing an avoidable cross-provider format"
          " confound; the mandatory per-API envelope is unchanged. Only affects"
          " Gemini (OpenAI/Anthropic already emit the canonical schema)."
      ),
  )
  parser.add_argument(
      "--llm-native-reasoning",
      action=argparse.BooleanOptionalAction,
      default=DEFAULT_LLM_NATIVE_REASONING,
      help=(
          "Thread native reasoning through native tool-calling (ignored in text"
          f" mode). Default: {DEFAULT_LLM_NATIVE_REASONING}. When on, Claude and"
          " GPT thread their opaque per-turn reasoning back into every forced"
          " tool-call step (Claude adaptive extended thinking + effort +"
          " tool_choice=auto, falling back to a fixed budget then plain; GPT"
          " store=false + include=reasoning.encrypted_content), mirroring"
          " Gemini's thought_signature so the chain of thought stays continuous"
          " and all three providers are treated consistently. Use"
          " --no-llm-native-reasoning to reproduce the prior reasoning-dropping"
          " native behavior (forced tool_choice, no thinking)."
      ),
  )
  parser.add_argument(
      "--llm-reasoning-effort",
      type=str,
      default=None,
      choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
      help=(
          "Explicit reasoning effort. Default: unset, meaning each vendor"
          " applies its OWN default -- which are NOT the same level: Anthropic"
          " documents 'high' as the API default for Sonnet 5 / Opus 4.8, while"
          " gpt-5.5 defaults to 'medium'. Comparing the families unset is"
          " therefore an unequal-effort comparison; set this to pin them to the"
          " same level. Where it applies is PER PROVIDER:"
          " OpenAI ('reasoning.effort') and Anthropic ('output_config.effort')"
          " honour it for native tool calls only -- it is ignored in text mode"
          " and under --no-llm-native-reasoning."
          " Gemini ('ThinkingConfig.thinking_level') honours it on BOTH the text"
          " and native paths. The one canonical ladder maps onto each vendor's"
          " own wire vocabulary rather than exposing per-vendor names: Gemini"
          " tops out at HIGH so 'xhigh'/'max' fold onto HIGH (and 'none'/'minimal'"
          " onto MINIMAL -- no true off); OpenAI passes 'none' through to bypass"
          " reasoning and folds 'minimal'->'low', 'max'->'xhigh'; Anthropic folds"
          " 'minimal'->'low' and 'xhigh'/'max'->'high'. Only a value off the"
          " ladder is rejected at startup. The level the requests actually"
          " carried is recorded as 'reasoning_effort_effective' in the run config."
          " Leaving it unset keeps every request byte-identical to prior runs."
          " NOTE this flag reaches the wire only on the OpenAI Responses API"
          " (gpt-* models), Gemini and Anthropic. An OpenAI-COMPATIBLE endpoint"
          " served over chat/completions (vLLM, SGLang) never receives it, and"
          " vLLM answers a request carrying it with HTTP 200 while ignoring it"
          " -- for those models use --llm-chat-template-kwargs instead."
      ),
  )
  parser.add_argument(
      "--llm-chat-template-kwargs",
      type=str,
      default=None,
      metavar="JSON",
      help=(
          "JSON object of server-side chat-template variables, forwarded"
          " verbatim as 'chat_template_kwargs' on the Chat Completions path."
          " This is how open-weight models expose reasoning control, which"
          " --llm-reasoning-effort cannot reach for them. Qwen3.6 reads"
          " 'enable_thinking' (bool; false emits an empty <think></think> and"
          " the model answers directly) and 'preserve_thinking' (bool; keep"
          " EARLIER turns' thinking in the rendered prompt -- without it the"
          " template drops any thinking that precedes the last user-role turn,"
          " which the nudge paths here do inject). Example:"
          ' \'{"enable_thinking": false}\'. Recorded in the run config.'
      ),
  )
  parser.add_argument(
      "--llm-replay-chat-reasoning",
      action="store_true",
      help=(
          "Re-inline each past tool turn's thinking into the chat history as"
          " '<think>...</think>', so a thinking model keeps a continuous chain"
          " of thought across tool steps instead of seeing its own reasoning"
          " vanish every step. Chat Completions only (the Responses and"
          " Anthropic paths already thread reasoning natively). Inlining is"
          " required because vLLM drops a non-standard 'reasoning_content'"
          " message field before applying the template; the inline form is the"
          " one the template parses back out. Only meaningful for models whose"
          " template strips these tags, so it is off by default."
      ),
  )
  parser.add_argument(
      "--llm-temperature",
      type=float,
      default=0.0,
      help="Sampling temperature for --agent llm (default: 0.0)",
  )
  parser.add_argument(
      "--llm-max-tokens",
      type=int,
      default=256,
      help="Max output tokens for --agent llm API calls (default: 256)",
  )
  parser.add_argument(
      "--llm-timeout-s",
      type=int,
      default=90,
      help=(
          "Per-request timeout for --agent llm API calls in seconds"
          " (default: 90)"
      ),
  )
  parser.add_argument(
      "--llm-request-retries",
      type=int,
      default=5,
      help="Retry count for transient --agent llm API failures (default: 5)",
  )
  parser.add_argument(
      "--llm-empty-reply-retries",
      type=int,
      default=5,
      help=(
          "Max re-sends when the provider returns a SUCCESSFUL (HTTP 200) but"
          " EMPTY/contentless reply -- Anthropic stop_reason=refusal (no"
          " text/tool_use), Gemini finish_reason=SAFETY, or an OpenAI-compatible"
          " response with empty message content. Re-sent verbatim with backoff"
          " (reusing --llm-retry-backoff-s); kept small and capped by"
          " --llm-request-retries so a deterministic refusal does not burn the"
          " full budget every stalling step. An empty reply that persists past"
          " this budget still coerces DONE and records"
          " generation_stall_empty_reply (provider_failure), unchanged."
          " (default: 5)"
      ),
  )
  parser.add_argument(
      "--llm-retry-backoff-s",
      type=float,
      default=1.5,
      help=(
          "Base backoff in seconds before retrying transient --agent llm API"
          " failures (default: 1.5)"
      ),
  )
  parser.add_argument(
      "--llm-transient-max-backoff-s",
      type=float,
      default=12.0,
      help=(
          "Per-retry ceiling in seconds for the ordinary transient error class"
          " (429/503/network). DEFAULT 12.0, the value this was previously"
          " hardcoded to, so leaving it unset reproduces the published runs"
          " exactly. Raise it when the endpoint is quota-limited on a window of"
          " hours rather than seconds (a personal subscription, say): at the 12s"
          " ceiling the exponential backoff stops growing, so an exhausted quota"
          " is met with sustained ~7 req/min/worker hammering rather than a"
          " widening wait, and --llm-request-retries is consumed in minutes."
          " Pair a large value here with a large --llm-request-retries to wait"
          " out a multi-hour quota window instead of failing the task."
      ),
  )
  parser.add_argument(
      "--llm-deadline-retry-backoff-s",
      type=float,
      default=60.0,
      help=(
          "Base backoff in seconds for the deadline/timeout class (HTTP 504 /"
          " DEADLINE_EXCEEDED or a client-side request timeout). Longer than"
          " --llm-retry-backoff-s so an overloaded backend can drain before we"
          " re-send the same request (default: 60.0)"
      ),
  )
  parser.add_argument(
      "--llm-deadline-max-backoff-s",
      type=float,
      default=60.0,
      help=(
          "Cap in seconds on the per-retry wait for the deadline/timeout class."
          " With equal-jitter each wait lands in [cap/2, cap]; the default"
          " 60/60 yields a flat 30-60s band (default: 60.0)"
      ),
  )
  parser.add_argument(
      "--vertex-project",
      default=None,
      help="Google Cloud project ID for Vertex AI. Fallback: VERTEX_PROJECT, GOOGLE_CLOUD_PROJECT, or GEMINI_PROJECT env vars.",
  )
  parser.add_argument(
      "--vertex-location",
      default=None,
      help="Google Cloud region/location for Vertex AI. Fallback: VERTEX_LOCATION, GOOGLE_CLOUD_REGION, or GEMINI_LOCATION env vars.",
  )
  parser.add_argument(
      "--multimodal-input-mode",
      default="both",
      choices=["path", "direct", "both"],
      help=(
          "Multimodal asset injection mode for --agent llm. path exposes asset"
          " ids through tools only; direct injects visible image assets as"
          " provider message parts; both does both. Default: both (falls back to"
          " path for providers/tasks without direct image support)"
      ),
  )
  parser.add_argument(
      "--anthropic-max-image-edge",
      default="auto",
      help=(
          "Long-edge cap (pixels) for inline images sent to Claude/Anthropic"
          " (incl. vertex-anthropic). 'auto' (default) sends originals at max"
          " fidelity, then learns the server's cap from the first oversized-"
          " image rejection and pre-shrinks subsequent calls. A positive"
          " integer forces a fixed cap. Ignored by non-Anthropic providers."
      ),
  )
  parser.add_argument(
      "--gemini-max-image-edge",
      type=int,
      default=None,
      help=(
          "Long-edge cap (pixels) for inline images sent to Gemini/Vertex. Off"
          " by default (originals sent unchanged). A positive integer downscales"
          " any image whose long edge exceeds it down to the cap (same format);"
          " images already within the cap pass through unchanged. This trims the"
          " accumulated multimodal history that causes deep-step 504"
          " DEADLINE_EXCEEDED. Ignored by non-Gemini providers. NOTE: unlike the"
          " Anthropic/OpenAI caps this does NOT change how much detail the model"
          " sees - Gemini resamples every image to a fixed per-image token"
          " budget server-side (see --gemini-media-resolution), and that budget"
          " is already saturated by ~1 MP. Measured on eval_results/"
          "data_3_5_flash (n=5509 images): 8.29 MP and 1.08 MP both cost ~1100"
          " tokens. So treat this purely as an upload-bytes/latency knob; use"
          " --gemini-media-resolution to change model-visible fidelity."
      ),
  )
  parser.add_argument(
      "--gemini-media-resolution",
      choices=("low", "medium", "high", "ultra_high"),
      default=None,
      help=(
          "Per-image visual token budget for Gemini/Vertex. Unset by default"
          " (server default, which for Gemini 3 images is 1120 tokens - the"
          " same as 'high'). Approximate tokens per image: low=280,"
          " medium=560, high=1120, ultra_high=2240. This is the ONLY knob that"
          " changes how much visual detail Gemini actually receives; resizing"
          " the upload (--gemini-max-image-edge) does not, because Gemini"
          " resamples to this fixed budget server-side. Use 'ultra_high' to"
          " narrow the gap against Anthropic (4784 visual tokens per 4K image)"
          " and OpenAI (~2040 at 1920px, ~3726 at 2576px); it roughly doubles"
          " Gemini image tokens. Applied per media Part, since 'ultra_high' is"
          " not accepted as a global generation-config value. Ignored by"
          " non-Gemini providers."
      ),
  )
  parser.add_argument(
      "--openai-max-image-edge",
      type=int,
      default=1920,
      help=(
          "Long-edge cap (pixels) for inline images sent to OpenAI-compatible"
          " providers (openai, vertex-xai/Grok). DEFAULT 1920 (1080p). gpt-5.x"
          " bills image tokens ~linearly with pixel area (no server-side"
          " downscale cap) and history_mode=full re-sends every image each"
          " step, so uncapped 4K assets dominate cost: on the conflict_vlm_v1"
          " corpus (~99%% of input tokens are images, ~9.9k tok per 4K image)"
          " capping to 1920 cuts input ~75%% while keeping dashboard evidence"
          " legible. Any image whose long edge exceeds the cap is downscaled"
          " (same format); smaller images pass through unchanged. Set 0 (or a"
          " negative value) to disable resizing and send originals. Ignored by"
          " non-OpenAI-compatible providers. Guide: 1568 ~ -82%%; 2048 (DCI 2K)"
          " ~ -71%%; 1280 (720p) ~ -89%% but risks small-text legibility."
      ),
  )
  parser.add_argument(
      "--no-compact-fallback",
      action="store_true",
      help=(
          "Disable the OpenAI-compatible compact-prompt fallback. By default,"
          " when a provider rejects a request with a content-filter /"
          " invalid_prompt 400, the agent silently retries once with a compact"
          " text-only prompt (ALL images + full history dropped), which recovers"
          " the task but in a degraded mode. Passing this flag makes such"
          " rejections raise instead, so the task is recorded as a"
          " provider_failure (excluded from accuracy and eligible for"
          " --retry-provider-failures-only) rather than a degraded pass. Only"
          " affects OpenAI-compatible providers."
      ),
  )
  parser.add_argument(
      "--interaction-log",
      type=str,
      default=None,
      help=(
          "Directory for the complete per-attempt interaction sidecar log"
          " (JSONL). Only used together with --capture-log; when omitted it"
          " defaults to '<out-dir>/log'. One '<scenario>.interactions.jsonl' is"
          " written inside it per scenario/task, one line per attempt: full"
          " input snapshot, raw + normalized output, validation outcome,"
          " finish_reason/safety/usage, PLUS the terminal turn that coerces DONE"
          " (invalid_command_exhausted / generation stall) which the trajectory"
          " model_debug drops."
      ),
  )
  parser.add_argument(
      "--capture-log",
      action="store_true",
      help=(
          "Master switch for complete interaction logging (off by default so"
          " default runs are byte-for-byte unchanged). When set it writes the"
          " per-attempt sidecar log AND records per-call response metadata for"
          " every provider: exact token usage (input/cached/output/reasoning)"
          " plus any reasoning/thinking text the model returned. For Gemini it"
          " also adds ThinkingConfig(include_thoughts=True) so thought summaries"
          " are returned; OpenAI/Anthropic usage and reasoning are read from the"
          " normal response (no extra request, so no latency cost). Output goes"
          " to --interaction-log or, by default, '<out-dir>/log'."
      ),
  )
  parser.add_argument(
      "--inter-task-sleep",
      type=float,
      default=0.0,
      help=(
          "Sleep this many seconds between tasks during batch eval/data"
          " collection to reduce provider pressure (default: 0.0)"
      ),
  )
  parser.add_argument(
      "--collect-data",
      action="store_true",
      help="Save trajectories as JSONL instead of printing metrics",
  )
  parser.add_argument(
      "--record-step-eval",
      action="store_true",
      help=(
          "Record evaluator progress after each executed command in"
          " trajectories."
      ),
  )
  parser.add_argument(
      "--resume-trajectories",
      action="store_true",
      help=(
          "With --collect-data, append to the trajectory file and skip task_ids"
          " already present in that JSONL file."
      ),
  )
  parser.add_argument(
      "--resume-successes-only",
      action="store_true",
      help=(
          "With --collect-data, append to the trajectory file and skip only"
          " previously successful task_ids. Failed/provider-impacted records"
          " are rerun."
      ),
  )
  parser.add_argument(
      "--retry-provider-failures-only",
      action="store_true",
      help=(
          "With --collect-data, append to the trajectory file and rerun "
          "task_ids whose latest record is provider-impacted/provider_failure "
          "AND task_ids that have no record yet (never ran / didn't finish). "
          "Successes and genuine (non-provider) failures are left untouched."
      ),
  )
  parser.add_argument(
      "--out-dir",
      default="data/trajectories",
      help="Output dir for --collect-data (default: data/trajectories)",
  )
  parser.add_argument(
      "--clean-out-dir",
      dest="clean_out_dir",
      action="store_true",
      help=(
          "Before --collect-data writes, delete existing contents of --out-dir "
          "so the run starts fresh (avoids stale per-scenario .jsonl files, "
          ".summary reports, and dashboard from a previous run polluting the "
          "aggregated visualizer). No effect without --collect-data."
      ),
  )
  parser.add_argument(
      "--task-data-dir",
      default=str(DEFAULT_TASK_DATA_DIR),
      help=(
          "Task data root containing tasks/ and datasets/. "
          "Default: clasharness/data"
      ),
  )
  parser.add_argument(
      "--task-id", default=None, help="Run a single task demo and exit"
  )
  parser.add_argument(
      "-v", "--verbose", action="store_true", help="Print per-step details"
  )
  parser.add_argument(
      "--verbose-log",
      default=None,
      help=(
          "Optional file path that receives the same stdout stream as the"
          " terminal, including -v per-step output."
      ),
  )
  parser.add_argument(
      "--online-clean",
      dest="online_clean",
      action="store_true",
      help="Store cleaned stdout/stderr fields in trajectories",
  )
  parser.add_argument(
      "--no-online-clean",
      dest="online_clean",
      action="store_false",
      help="Disable output cleaning in trajectories",
  )
  parser.set_defaults(online_clean=None)
  parser.add_argument(
      "--skip-incompatible-openclaw",
      dest="skip_incompatible_openclaw",
      action="store_true",
      default=True,
      help="Skip incompatible real openclaw commands with labels (default: on)",
  )
  parser.add_argument(
      "--no-skip-incompatible-openclaw",
      dest="skip_incompatible_openclaw",
      action="store_false",
      help="Do not skip incompatible real openclaw commands",
  )
  parser.add_argument(
      "--online-openclaw-only",
      action="store_true",
      help="Only run tasks whose GT includes at least one `openclaw *` command",
  )
  parser.add_argument(
      "--fallback-openclaw-network-to-mock",
      action="store_true",
      help=(
          "In real/hybrid mode, fallback openclaw commands to mock when "
          "real execution fails due to network/auth/dependency errors"
      ),
  )
  parser.add_argument(
      "--strict-online-data",
      dest="strict_online_data",
      action="store_true",
      help=(
          "In real/hybrid mode, disable mock fallback for"
          " weather/calendar/email/tasks online reads and fail the command if"
          " live fetch is unavailable (default: on)"
      ),
  )
  parser.add_argument(
      "--no-strict-online-data",
      dest="strict_online_data",
      action="store_false",
      help=(
          "Allow weather/calendar/email/tasks mock fallback when live fetch is"
          " unavailable."
      ),
  )
  parser.set_defaults(strict_online_data=True)
  parser.add_argument(
      "--llm-judge-model",
      default=None,
      help="Model to use for LLM semantic validation judge (e.g. gemini-2.5-flash)",
  )
  parser.add_argument(
      "--llm-judge-provider",
      default=None,
      choices=[
          "gemini",
          "anthropic",
          "openai",
          "vertex-anthropic",
          "bedrock-gpt",
      ],
      help=(
          "Backend for the LLM semantic judge. Names match --llm-provider. "
          "Official-API backends, which need only a key: 'gemini' = Gemini "
          "(GEMINI_API_KEY, or a Vertex project for the ADC path); 'anthropic' "
          "= Claude via the Anthropic API (ANTHROPIC_API_KEY); 'openai' = GPT "
          "via OpenAI's Responses API (OPENAI_API_KEY and an explicit "
          "--llm-judge-model such as 'gpt-5.5'; override the endpoint via "
          "OPENAI_BASE_URL). Cloud backends, which the paper's runs used and "
          "which need cloud credentials: 'vertex-anthropic' = Claude on Vertex "
          "AI via ADC (needs a Vertex project, a Claude-capable "
          "--vertex-location such as us-east5, and an explicit "
          "--llm-judge-model -- passed through verbatim, e.g. "
          "'claude-sonnet-4@20250514' or 'claude-opus-4-8'); 'bedrock-gpt' = "
          "GPT on AWS Bedrock via its OpenAI Responses API (needs "
          "BEDROCK_OPENAI_API_KEY and an explicit --llm-judge-model with an "
          "'openai.' prefix, e.g. 'openai.gpt-5.5'; override the endpoint via "
          "BEDROCK_OPENAI_BASE_URL). When omitted, the judge auto-selects from "
          "the environment (GEMINI_API_KEY or VERTEX_PROJECT -> gemini, else "
          "ANTHROPIC_API_KEY -> anthropic, else BEDROCK_OPENAI_API_KEY -> "
          "bedrock-gpt, else OPENAI_API_KEY -> openai); 'vertex-anthropic' is "
          "only reachable via this flag."
      ),
  )
  parser.add_argument(
      "--save-report",
      default=None,
      help="Save JSON evaluation report to this path",
  )

  args = parser.parse_args()
  if args.llm_judge_model:
    os.environ["LLM_JUDGE_MODEL"] = args.llm_judge_model
  if args.llm_judge_provider:
    os.environ["LLM_JUDGE_PROVIDER"] = args.llm_judge_provider
  if args.vertex_project:
    os.environ["VERTEX_PROJECT"] = args.vertex_project
  if args.vertex_location:
    os.environ["VERTEX_LOCATION"] = args.vertex_location
  # Give the LLM judge the SAME transient-failure resilience as the agent by
  # reusing --llm-request-retries / --llm-retry-backoff-s. These are transported
  # via env (not a function arg) because the judge is invoked deep inside
  # evaluators with no access to args, and env also propagates to parallel worker
  # processes. Not a new user-facing knob -- purely the wire for existing flags.
  os.environ["LLM_JUDGE_REQUEST_RETRIES"] = str(args.llm_request_retries)
  os.environ["LLM_JUDGE_RETRY_BACKOFF_S"] = str(args.llm_retry_backoff_s)

  # Keep the judge and the agent on the same Vertex project/location: when the
  # judge resolves to a Vertex backend (Gemini or Claude-on-Vertex) but
  # VERTEX_PROJECT is empty, backfill it from the same fallback chain the agent
  # uses so the judge cannot silently drop out of its provider branch.
  from clasharness.evaluation.llm_judge import resolve_provider as _resolve_judge_provider
  _judge_provider = _resolve_judge_provider()
  if _judge_provider in {"gemini", "vertex-anthropic"} and not os.environ.get(
      "VERTEX_PROJECT"
  ):
    _backfilled_project = _resolve_vertex_project(args.vertex_project)
    # Only export a real project. Setting it to "" would make the variable
    # *present* and empty, which reads as "configured" further down.
    if _backfilled_project:
      os.environ["VERTEX_PROJECT"] = _backfilled_project
  # Only backfill the location for Gemini (whose default region is 'us'). For
  # vertex-anthropic 'us' does not serve Claude, so leave the location to the
  # judge backend's own Claude-capable default (us-east5) when it is unset.
  if _judge_provider == "gemini" and not os.environ.get("VERTEX_LOCATION"):
    os.environ["VERTEX_LOCATION"] = _resolve_vertex_location(args.vertex_location)

  _validate_judge_config()

  def _run() -> None:
    _preflight_mode(args.mode)
    resolved_max_steps = _resolve_max_steps(
        args.max_steps,
        split=args.split,
        task_prefix=args.task_prefix,
        task_id=args.task_id,
    )

    if args.agent == "llm" and args.llm_provider == "bedrock-gpt" and (
        not args.model or args.model.lower().startswith("claude")
    ):
      raise SystemExit(
          "train_and_eval: error: --llm-provider bedrock-gpt requires an "
          "explicit --model (e.g. 'openai.gpt-5.5'); there is no default and "
          f"{args.model!r} is not a Bedrock GPT model id."
      )

    agent = _build_agent(
        args.agent,
        args.model,
        llm_provider=args.llm_provider,
        llm_base_url=args.llm_base_url,
        llm_api_key_env=args.llm_api_key_env,
        llm_history_mode=args.llm_history_mode,
        llm_temperature=args.llm_temperature,
        llm_max_tokens=args.llm_max_tokens,
        llm_timeout_s=args.llm_timeout_s,
        llm_request_retries=args.llm_request_retries,
        llm_empty_reply_retries=args.llm_empty_reply_retries,
        llm_retry_backoff_s=args.llm_retry_backoff_s,
        llm_transient_max_backoff_s=args.llm_transient_max_backoff_s,
        llm_deadline_retry_backoff_s=args.llm_deadline_retry_backoff_s,
        llm_deadline_max_backoff_s=args.llm_deadline_max_backoff_s,
        multimodal_input_mode=args.multimodal_input_mode,
        max_steps_hint=resolved_max_steps,
        verbose=args.verbose,
        vertex_project=args.vertex_project,
        vertex_location=args.vertex_location,
        anthropic_max_image_edge=args.anthropic_max_image_edge,
        gemini_max_image_edge=args.gemini_max_image_edge,
        gemini_media_resolution=args.gemini_media_resolution,
        openai_max_image_edge=args.openai_max_image_edge,
        capture_thinking=args.capture_log,
        capture_interactions=args.capture_log,
        llm_allow_compact_fallback=not args.no_compact_fallback,
        llm_tool_mode=args.llm_tool_mode,
        llm_tool_schema=args.llm_tool_schema,
        llm_native_reasoning=args.llm_native_reasoning,
        llm_reasoning_effort=args.llm_reasoning_effort,
        llm_chat_template_kwargs=_parse_chat_template_kwargs(
            args.llm_chat_template_kwargs
        ),
        llm_replay_chat_reasoning=args.llm_replay_chat_reasoning,
    )

    # --capture-log is the single master switch for complete interaction
    # logging: it turns on BOTH the per-attempt sidecar record and (for Gemini)
    # thought capture. The log directory defaults to '<out-dir>/log' but can be
    # overridden with --interaction-log. Resolved once here and threaded into
    # whichever run mode executes below.
    interaction_log_dir: str | None = None
    if args.capture_log:
      interaction_log_dir = args.interaction_log or str(
          Path(args.out_dir) / "log"
      )
    elif args.interaction_log is not None:
      print(
          "WARNING: --interaction-log has no effect without --capture-log;"
          " no interaction log will be written."
      )

    # ------------------------------------------------------------------
    # Single-task demo mode
    # ------------------------------------------------------------------
    if args.task_id:
      run_demo(
          args.task_id,
          agent,
          task_data_dir=args.task_data_dir,
          mode=args.mode,
          max_steps=resolved_max_steps,
          max_stagnant_steps=args.max_stagnant_steps,
          online_clean=args.online_clean,
          skip_incompatible_openclaw=args.skip_incompatible_openclaw,
          fallback_openclaw_network_to_mock=args.fallback_openclaw_network_to_mock,
          strict_online_data=args.strict_online_data,
          record_step_eval=args.record_step_eval,
          interaction_log=interaction_log_dir,
      )
      return

    # ------------------------------------------------------------------
    # Data collection mode
    # ------------------------------------------------------------------
    if args.collect_data:
      collect_training_data(
          agent,
          split=args.split,
          out_dir=args.out_dir,
          clean_out_dir=args.clean_out_dir,
          task_data_dir=args.task_data_dir,
          mode=args.mode,
          task_prefix=args.task_prefix,
          scenario_slugs=args.scenario_slug,
          domain=args.domain,
          difficulty=args.difficulty,
          limit=args.limit,
          max_steps=resolved_max_steps,
          max_stagnant_steps=args.max_stagnant_steps,
          online_clean=args.online_clean,
          skip_incompatible_openclaw=args.skip_incompatible_openclaw,
          fallback_openclaw_network_to_mock=args.fallback_openclaw_network_to_mock,
          strict_online_data=args.strict_online_data,
          online_openclaw_only=args.online_openclaw_only,
          resume_trajectories=args.resume_trajectories,
          resume_successes_only=args.resume_successes_only,
          retry_provider_failures_only=args.retry_provider_failures_only,
          inter_task_sleep_s=args.inter_task_sleep,
          record_step_eval=args.record_step_eval,
          parallel_workers=args.parallel_workers,
          verbose=args.verbose,
          interaction_log=interaction_log_dir,
      )
      return

    # ------------------------------------------------------------------
    # Evaluation mode (default)
    # ------------------------------------------------------------------
    summary = run_evaluation(
        agent,
        split=args.split,
        task_data_dir=args.task_data_dir,
        mode=args.mode,
        task_prefix=args.task_prefix,
        scenario_slugs=args.scenario_slug,
        domain=args.domain,
        difficulty=args.difficulty,
        limit=args.limit,
        max_steps=resolved_max_steps,
        max_stagnant_steps=args.max_stagnant_steps,
        online_clean=args.online_clean,
        skip_incompatible_openclaw=args.skip_incompatible_openclaw,
        fallback_openclaw_network_to_mock=args.fallback_openclaw_network_to_mock,
        strict_online_data=args.strict_online_data,
        online_openclaw_only=args.online_openclaw_only,
        inter_task_sleep_s=args.inter_task_sleep,
        record_step_eval=args.record_step_eval,
        verbose=args.verbose,
        interaction_log=interaction_log_dir,
    )
    summary.print_report()

    if args.save_report:
      report = {
          "agent": summary.agent_name,
          "split": summary.split,
          "mode": summary.exec_mode,
          "max_steps_budget": resolved_max_steps,
          "llm_provider": summary.llm_provider,
          "llm_model": summary.llm_model,
          "llm_history_mode": summary.llm_history_mode,
          "scenario_slugs": args.scenario_slug,
          "tgc": summary.tgc,
          "pass_at_budget": summary.pass_at_budget,
          "raw_accuracy": summary.tgc,
          "avg_score": summary.avg_score,
          "near_miss_rate": summary.near_miss_rate,
          "near_miss_tasks": summary.near_miss_tasks,
          "done_early_tasks": summary.done_early_tasks,
          "step_capped_tasks": summary.step_capped_tasks,
          "date_anchor_mismatch_tasks": summary.date_anchor_mismatch_tasks,
          "score_rank_note": summary.score_rank_note,
          "passed": summary.passed,
          "total": summary.total,
          "completed_tasks": summary.completed_tasks,
          "clean_completed_tasks": summary.clean_completed_tasks,
          "provider_failures": summary.provider_failures,
          "provider_impacted_tasks": summary.provider_impacted_tasks,
          "compact_fallback_tasks": summary.compact_fallback_tasks,
          "provider_adjusted_accuracy": summary.provider_adjusted_tgc,
          "clean_accuracy": summary.clean_tgc,
          "by_domain": {
              d: {
                  "passed": m.passed,
                  "total": m.total,
                  "tgc": m.tgc,
                  "avg_score": m.avg_score,
              }
              for d, m in summary.by_domain.items()
          },
          "by_difficulty": {
              str(d): {
                  "passed": m.passed,
                  "total": m.total,
                  "tgc": m.tgc,
                  "avg_score": m.avg_score,
              }
              for d, m in summary.by_difficulty.items()
          },
          "by_hard_scenario": {
              k: {
                  "passed": m.passed,
                  "total": m.total,
                  "tgc": m.tgc,
                  "avg_score": m.avg_score,
              }
              for k, m in summary.by_hard_scenario.items()
          },
          "by_hard_ability": {
              k: {
                  "passed": m.passed,
                  "total": m.total,
                  "tgc": m.tgc,
                  "avg_score": m.avg_score,
              }
              for k, m in summary.by_hard_ability.items()
          },
          "by_hard_ability_tag": {
              k: {
                  "passed": m.passed,
                  "total": m.total,
                  "tgc": m.tgc,
                  "avg_score": m.avg_score,
              }
              for k, m in summary.by_hard_ability_tag.items()
          },
          "results": [r.to_dict() for r in summary.results],
      }
      Path(args.save_report).parent.mkdir(parents=True, exist_ok=True)
      with open(args.save_report, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
      print(f"Report saved → {args.save_report}")

  if args.verbose_log:
    log_path = Path(args.verbose_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_fh:
      with contextlib.redirect_stdout(_TeeStdout(sys.stdout, log_fh)):
        _run()
  else:
    _run()


if __name__ == "__main__":
  main()
