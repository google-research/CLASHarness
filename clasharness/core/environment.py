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

"""Main environment class for CLI agent testing and training."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from clasharness.backend.base import BaseBackend
from clasharness.backend.mock_backend import MockBackend
from clasharness.backend.multi_app_backend import MultiAppBackend
from clasharness.core.observation import EvaluationResult, Observation
from clasharness.core.task import Task, load_task
from clasharness.evaluation.evaluator import EvaluatorComb, build_evaluator
from clasharness.utils.safety_guard import (
    SafetyViolation,
    check_command_safety,
    check_python_safety,
)
from clasharness.utils.state_manager import StateManager
from clasharness.utils.trajectory_recorder import TrajectoryRecorder
from clasharness.skills.state_merge import (
    as_named_mapping,
    merge_backend_state_into_eval,
    merge_named_entities,
)
from clasharness.core.agent_manager import AgentManager


def _parse_delegate_args(tokens: list[str]) -> tuple[str | None, str | None]:
    """Parse ``agent delegate`` flags, tolerating an unquoted multi-word instruction.

    The agent-facing tool template documents ``agent delegate --agent <name>
    --instruction <text>`` without any quoting guidance, so models routinely emit
    an *unquoted* free-text instruction. ``shlex.split`` then explodes that prose
    into many argv tokens, and a strict ``argparse`` parser (``--instruction``
    consumes a single token) rejects the trailing words with
    ``error: unrecognized arguments: ...`` -> the whole delegation is silently
    dropped and the caller is told, misleadingly, that the flags are missing.

    Instead treat ``--agent`` as a single-token flag and ``--instruction`` as
    consuming the remainder of the line (up to a later ``--agent``). This is
    identical to the intended result for a correctly quoted instruction and
    recovers the far more common unquoted case. Both ``--flag value`` and
    ``--flag=value`` spellings are accepted.

    Returns ``(agent, instruction)`` where either element is ``None`` when the
    corresponding flag is absent or empty.
    """
    agent: str | None = None
    instruction: str | None = None
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if tok == "--agent" or tok.startswith("--agent="):
            if "=" in tok:
                agent = tok.split("=", 1)[1] or None
                i += 1
            else:
                agent = tokens[i + 1] if i + 1 < n else None
                i += 2
        elif tok == "--instruction" or tok.startswith("--instruction="):
            rest: list[str] = []
            if "=" in tok:
                first = tok.split("=", 1)[1]
                if first:
                    rest.append(first)
            i += 1
            # The instruction is free text: swallow every following token until a
            # later --agent flag (the only other recognized delegate flag), so an
            # unquoted multi-word instruction is reassembled intact.
            while (
                i < n
                and tokens[i] != "--agent"
                and not tokens[i].startswith("--agent=")
            ):
                rest.append(tokens[i])
                i += 1
            instruction = " ".join(rest).strip() or None
        else:
            i += 1
    return agent, instruction


class CLASHarnessEnv:
    """Main environment for CLI agent testing and training.

    Supports CLI and Python API dual-mode interaction with composable
    evaluation and state isolation.

    Usage:
        with CLASHarnessEnv(task_id="msg_send_basic_1") as env:
            obs = env.reset()
            while True:
                action = agent.act(obs)
                obs, reward, done, info = env.step(action)
                if done:
                    break
            result = env.evaluate()
    """

    def __init__(
        self,
        task_id: str,
        experiment_name: str = "default",
        backend: str = "mock",
        interface: str = "cli",
        max_steps: int = 50,
        timeout_seconds: int = 120,
        raise_on_safety_violation: bool = True,
        record_trajectory: bool = True,
        record_step_evaluation: bool = False,
        instruction_variant_mode: str = "canonical",
        backend_kwargs: dict | None = None,
        task_data_dir: str | Path | None = None,
    ) -> None:
        self.task_id = task_id
        self.experiment_name = experiment_name
        self.interface = interface
        self.max_steps = max_steps
        self.timeout_seconds = timeout_seconds
        self.raise_on_safety_violation = raise_on_safety_violation
        self.record_trajectory = record_trajectory
        self.record_step_evaluation = record_step_evaluation
        self._last_step_eval_score = 0.0
        if instruction_variant_mode not in {"canonical", "sampled"}:
            raise ValueError(
                "instruction_variant_mode must be 'canonical' or 'sampled'."
            )
        self.instruction_variant_mode = instruction_variant_mode
        self._task_data_dir = Path(task_data_dir) if task_data_dir else None

        self._task: Task | None = None
        self._backend: BaseBackend = _create_backend(backend, **(backend_kwargs or {}))
        self._state_manager: StateManager | None = None
        self._evaluator: EvaluatorComb | None = None
        self._recorder: TrajectoryRecorder | None = None
        self._agent_manager: AgentManager | None = None

        self._step_count = 0
        self._done = False
        self._last_observation: Observation | None = None
        self._cumulative_reward = 0.0
        self._command_history: list[dict] = []
        self._active_instruction = ""
        self._virtual_time = 0
        self._background_events = []
        self._active_sub_agents = {}
        self._history_stack: list[str] = []
        # In-memory snapshot store: snapshot_id -> {"backend": ..., "env": ...}.
        # Enables fast, sound undo without per-step full-directory copies.
        self._snapshots: dict[str, dict[str, Any]] = {}

        # Track effects for evaluation
        self._effects: dict[str, list[dict[str, Any]]] = {
            "messages_sent": [],
            # Environment-INJECTED incoming messages (``session_append`` background
            # events: a peer/system posting INTO a session the solver is reading).
            # Kept in a separate bucket from ``messages_sent`` -- which grades what
            # the solver (and mock sub-agents) *sent* -- so an injected provocation
            # can never satisfy a ``messages_sent`` content check about the agent's
            # own outgoing message. Still mirrored into the searchable message pool
            # so the solver can ``message search`` for it.
            "incoming_messages": [],
            "agents_created": [],
            "agents_deleted": [],
            "channels_configured": [],
            "cron_jobs_created": [],
            "plugins_installed": [],
            "config_changes": [],
            # New app effects
            "calendar_events_created": [],
            "calendar_events_updated": [],
            "calendar_events_deleted": [],
            "emails_sent": [],
            "emails_moved": [],
            "files_created": [],
            "files_appended": [],
            "files_deleted": [],
            "tasks_created": [],
            "tasks_completed": [],
            "assets_accessed": [],
            "assets_opened": [],
            "agent_status_checked": [],
        }

    def __enter__(self) -> CLASHarnessEnv:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def reset(self) -> Observation:
        """Initialize environment to the task's initial state."""
        # Load task
        self._task = load_task(self.task_id, data_dir=self._task_data_dir)

        # Set up isolated state
        self._state_manager = StateManager(
            base_config_name=self._task.initial_state
        )
        self._state_manager.initialize()
        self._state_manager.apply_state_overrides(
            self._task.data.private.get("initial_state_overrides", {})
        )

        # Initialize backend
        env_vars = self._state_manager.get_env_vars()
        # Thread the canonical real-path signal for real/hybrid backends. This
        # replaces the old (dead) ``real_backend is not None`` gate: real
        # behavior is now selected per-skill via CLASHARNESS_REAL. Mock/multi
        # backends report ``real_enabled=False`` and stay fully simulated.
        if getattr(self._backend, "real_enabled", False):
            env_vars["CLASHARNESS_REAL"] = "1"
            env_vars.setdefault("CLASHARNESS_ENABLE_ONLINE_DATA", "1")
        self._backend.initialize(str(self._state_manager.state_dir), env_vars)

        # Pre-populate backend with mock subagents defined in task metadata
        mock_subagents = self._task.data.private.get("mock_subagents", [])
        for subagent in mock_subagents:
            role = subagent.get("agent_role") or subagent.get("agent_name")
            if role:
                self._backend.execute_cli(f"agent add {role}")

        # Build evaluator from task ground truth
        if self._task.ground_truth:
            self._evaluator = build_evaluator(
                self._task.ground_truth.evaluation_checks
            )

        # Set up trajectory recorder
        if self.record_trajectory:
            self._recorder = TrajectoryRecorder(
                task_id=self.task_id,
                instruction=self._resolve_task_instruction(self._task),
            )
            self._recorder.set_metadata({
                "domains": self._task.domains,
                "difficulty": self._task.difficulty,
                "experiment": self.experiment_name,
                "interface": self.interface,
                "instruction_variant_mode": self.instruction_variant_mode,
                "canonical_instruction": self._task.canonical_instruction
                or self._task.instruction,
            })

        # Reset counters
        self._step_count = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._last_step_eval_score = 0.0
        self._command_history.clear()
        for k in self._effects:
            self._effects[k].clear()
        self._active_instruction = self._resolve_task_instruction(self._task)
        
        self._virtual_time = 0
        events = self._task.data.private.get("background_events", []) if self._task.data else []
        self._background_events = sorted(list(events), key=lambda x: x.get("trigger_time", 0))
        self._active_sub_agents = {}
        self._history_stack.clear()
        
        mode_str = "mock" if self._backend.__class__.__name__ == "MockBackend" else "real"
        self._agent_manager = AgentManager(mode=mode_str, env=self)

        # Build initial observation
        obs = Observation(
            command_output="Environment initialized. Ready for commands.",
            task_instruction=self._active_instruction,
            current_config=self._backend.get_config(),
            gateway_status=self._backend.get_gateway_status(),
            available_commands=_get_available_commands(
                self._task.domains,
            ),
            step_number=0,
        )
        self._last_observation = obs
        return obs

    def _is_real_execution(self) -> bool:
        """Whether the active backend runs real (non-mock) tools.

        Real backends run real tools (curl/git/openclaw) that the strict
        simulator safety would otherwise reject, so real execution relaxes
        safety and passes incompatible commands through. Uses the canonical
        ``real_enabled`` signal plus the legacy ``real_backend`` delegate
        (e.g. HybridBackend) for backward-compatible behavior.
        """
        backend = self._backend
        return bool(
            getattr(backend, "real_enabled", False)
            or getattr(backend, "real_backend", None) is not None
        )

    def step(self, action: str) -> tuple[Observation, float, bool, dict[str, Any]]:
        """Execute an action and return (observation, reward, done, info)."""
        if self._done:
            raise RuntimeError("Episode already done. Call reset() first.")
        if self._task is None:
            raise RuntimeError("No task loaded. Call reset() first.")

        # Determine if real execution is active (relaxes safety + passes real
        # commands through). See ``_is_real_execution``.
        is_real = self._is_real_execution()

        # Apply compatibility rewrite so both safety and execution run on canonical commands
        from clasharness.backend.openclaw_compat import rewrite_command
        action = (action or "").strip()
        # Preserve the agent-facing command verbatim. ``action`` is reassigned
        # below to the rewritten/expanded runtime form (e.g. an MCP verb like
        # ``run_script --path X`` -> ``python3 X``) for safety + execution, but
        # command_history must also retain the raw form: CommandChecker grades
        # against the agent-facing verb the scenario was compiled with, and the
        # expanded executable (``python3``/``bash``/``unzip``/``tar``/``sqlite3``)
        # would never match ``command:run_script ...`` etc.
        raw_action = action
        decision = rewrite_command(action, skip_incompatible=is_real)
        
        if decision.compat_status == "skipped_incompatible":
            obs = Observation(
                command_output="",
                error_output=decision.skip_reason or "Skipped incompatible command.",
                exit_code=125,
                current_config=self._backend.get_config(),
                gateway_status=self._backend.get_gateway_status(),
                task_instruction=self._active_instruction,
                step_number=self._step_count,
            )
            return self._finalize_step(action, obs, {})

        action = (decision.executed_action or "").strip()

        # Intercept wait to skip budget
        import shlex
        try:
            action_parts = shlex.split(action)
        except ValueError:
            action_parts = action.strip().split()
        is_wait_cmd = bool(action_parts and action_parts[0] == "wait")
        is_undo_cmd = bool(action_parts and action_parts[0] == "undo")

        if not is_undo_cmd:
            snapshot_id = self.save_state()
            self._history_stack.append(snapshot_id)
            while len(self._history_stack) > 50:
                self._evict_snapshot(self._history_stack.pop(0))

        if not is_wait_cmd:
            self._step_count += 1
        
        info: dict[str, Any] = {"step": self._step_count}

        # Handle wait command completely within environment
        if is_wait_cmd:
            try:
                wait_seconds = int(action_parts[1]) if len(action_parts) > 1 else 0
            except ValueError:
                wait_seconds = 0
            triggered_events = self._tick(wait_seconds)
            
            output_lines = [f"Waited for {wait_seconds} seconds."]
            if triggered_events:
                output_lines.append(f"{len(triggered_events)} background events completed.")
                
            obs = Observation(
                command_output="\n".join(output_lines),
                error_output=None,
                exit_code=0,
                current_config=self._backend.get_config(),
                gateway_status=self._backend.get_gateway_status(),
                task_instruction=self._active_instruction,
                step_number=self._step_count,
            )
            # Record command
            self._command_history.append({
                "action": action,
                "raw_action": raw_action,
                "stdout": obs.command_output,
                "stderr": "",
                "exit_code": 0,
            })
            if len(self._command_history) > 50:
                self._command_history.pop(0)

            return self._finalize_step(action, obs, info)
            
        if is_undo_cmd:
            try:
                n_steps = int(action_parts[1]) if len(action_parts) > 1 else 1
            except ValueError:
                n_steps = 1
                
            if not self._history_stack:
                obs = Observation(
                    command_output="Error: No history available to undo.",
                    error_output=None,
                    exit_code=1,
                    current_config=self._backend.get_config(),
                    gateway_status=self._backend.get_gateway_status(),
                    task_instruction=self._active_instruction,
                    step_number=self._step_count,
                )
                self._last_observation = obs
                if self._recorder:
                    self._recorder.record_step(obs, action, 0.0, self._done, info)
                return obs, 0.0, self._done, info

            n_steps = max(1, min(n_steps, len(self._history_stack)))
            popped = [self._history_stack.pop() for _ in range(n_steps)]
            target_snapshot_id = popped[-1]

            # Virtual time keeps moving forward even after an undo; restored
            # background events are shifted so they still fire at the same
            # wall-clock offset relative to "now".
            current_time = self._virtual_time
            snapshot = self._snapshots.get(target_snapshot_id, {})
            snapshot_time = snapshot.get("env", {}).get("virtual_time", current_time)

            self.load_state(target_snapshot_id)
            self._virtual_time = current_time
            time_delta = current_time - snapshot_time
            for event in self._background_events:
                event["trigger_time"] = event.get("trigger_time", 0) + time_delta

            # Consumed snapshots (including intermediates) are freed.
            for sid in popped:
                self._evict_snapshot(sid)

            obs = Observation(
                command_output=f"Successfully undid {n_steps} steps. Virtual time did not revert.",
                error_output=None,
                exit_code=0,
                current_config=self._backend.get_config(),
                gateway_status=self._backend.get_gateway_status(),
                task_instruction=self._active_instruction,
                step_number=self._step_count,
            )
            self._command_history.append({
                "action": action,
                "raw_action": raw_action,
                "stdout": obs.command_output,
                "stderr": "",
                "exit_code": 0,
            })
            if len(self._command_history) > 50:
                self._command_history.pop(0)

            return self._finalize_step(action, obs, info)

        is_delegate_cmd = bool(action_parts and action_parts[:2] == ["agent", "delegate"])
        is_status_cmd = bool(action_parts and action_parts[:2] == ["agent", "status"])

        if is_delegate_cmd:
            # Tolerant parse: an unquoted multi-word --instruction must NOT be
            # rejected as "unrecognized arguments" (that silently drops the whole
            # delegation and misreports the flags as missing). See
            # ``_parse_delegate_args``.
            agent_name, instruction = _parse_delegate_args(action_parts[2:])
            if agent_name and instruction:
                self._agent_manager.delegate(agent_name, instruction)
                # L3 (M8/CL2): remember that the solver actually delegated to this
                # sub-agent, so ``_tick`` may fire an ``async_only`` decisive
                # completion (see ``_async_gate_blocks``). Recorded in the
                # env-level, snapshot/undo-wired ``_active_sub_agents`` (NOT the
                # agent manager's ``active_tasks``, which undo does not revert) so
                # undoing a delegate correctly revokes it and cannot re-open the
                # no-delegate bypass. A strict no-op for every non-async_only
                # scenario -- the gate only consults this for async_only agents.
                self._active_sub_agents[agent_name] = {
                    "instruction": instruction,
                    "delegated_at": self._virtual_time,
                }
                # Record the delegation through the SAME single messages_sent
                # sink used by message send/broadcast/poll/react (via
                # _track_effects), rather than a bespoke direct mutation. This
                # keeps messages_sent single-source: every communication action
                # feeds one sink and is treated uniformly.
                self._track_effects({"messages": [{
                    "recipient": agent_name,
                    "message": instruction,
                    "type": "delegation",
                }]})
                output = f"Task delegated to {agent_name} successfully."
            else:
                # Accurate, actionable error naming ONLY the truly-missing flag(s)
                # (the old message always claimed both were required, even when the
                # real cause was an unquoted instruction that argparse rejected).
                missing = []
                if not agent_name:
                    missing.append("--agent")
                if not instruction:
                    missing.append("--instruction")
                output = (
                    "Error: agent delegate requires "
                    + " and ".join(missing)
                    + ". Usage: agent delegate --agent <name> --instruction <text>"
                )
                
            obs = Observation(
                command_output=output,
                error_output=None,
                exit_code=0,
                current_config=self._backend.get_config(),
                gateway_status=self._backend.get_gateway_status(),
                task_instruction=self._active_instruction,
                step_number=self._step_count,
            )
            self._command_history.append({
                "action": action,
                "raw_action": raw_action,
                "stdout": obs.command_output[:500],
                "stderr": "",
                "exit_code": 0,
            })
            if len(self._command_history) > 50:
                self._command_history.pop(0)
            return self._finalize_step(action, obs, info)

        if is_status_cmd:
            import json
            import argparse
            parser = argparse.ArgumentParser()
            parser.add_argument("--agent", required=False)
            try:
                args = parser.parse_args(action_parts[2:])
                status_dict = self._agent_manager.status(args.agent)
                output = json.dumps(status_dict, indent=2)
                
                # Record agent status check effect
                if "agent_status_checked" not in self._effects:
                    self._effects["agent_status_checked"] = []
                
                if args.agent:
                    # Resolve to role
                    role = args.agent
                    mock_subagents = self._task.data.private.get("mock_subagents", [])
                    for sub in mock_subagents:
                        role_name = sub.get("agent_role", "")
                        sub_name = sub.get("agent_name", "")
                        if args.agent.lower() in role_name.lower() or args.agent.lower() in sub_name.lower():
                            role = role_name
                            break
                    self._effects["agent_status_checked"].append({
                        "agent_name": args.agent,
                        "agent_role": role,
                        "virtual_time": self._virtual_time,
                    })
                else:
                    # If general status check, record status checks for all mock subagents defined in this round
                    mock_subagents = self._task.data.private.get("mock_subagents", [])
                    for sub in mock_subagents:
                        role = sub.get("agent_role", "")
                        self._effects["agent_status_checked"].append({
                            "agent_name": sub.get("agent_name", role),
                            "agent_role": role,
                            "virtual_time": self._virtual_time,
                        })
            except SystemExit:
                output = "Error: Invalid arguments."
                
            obs = Observation(
                command_output=output,
                error_output=None,
                exit_code=0,
                current_config=self._backend.get_config(),
                gateway_status=self._backend.get_gateway_status(),
                task_instruction=self._active_instruction,
                step_number=self._step_count,
            )
            self._command_history.append({
                "action": action,
                "raw_action": raw_action,
                "stdout": obs.command_output[:500],
                "stderr": "",
                "exit_code": 0,
            })
            if len(self._command_history) > 50:
                self._command_history.pop(0)
            return self._finalize_step(action, obs, info)

        # Safety check. ``is_real`` (computed above) relaxes strict simulator
        # safety for real execution.
        try:
            if self.interface == "python":
                check_python_safety(action)
            else:
                check_command_safety(action, strict=not is_real)
        except SafetyViolation as e:
            obs = Observation(
                error_output=f"Safety violation: {e}",
                exit_code=1,
                task_instruction=self._active_instruction,
                step_number=self._step_count,
                current_config=self._backend.get_config(),
            )
            info["safety_violation"] = True
            if self.raise_on_safety_violation:
                self._done = True
            return self._finalize_step(action, obs, info)

        # Publish the virtual clock so stateful protocol skills can expire
        # time-limited records (holds/pairing codes/leases) at dispatch time.
        self._backend.set_virtual_time(self._virtual_time)

        # Execute action.
        #
        # A malformed model command (e.g. an unterminated quote in a
        # ``--content`` value) makes ``shlex.split`` deep inside a skill raise
        # ``ValueError: No closing quotation``. Without this guard that
        # exception propagates all the way up and crashes the whole episode as a
        # ``runner_failure``. Degrade it to a normal failed step instead: return
        # an error Observation (mirroring the ``skipped_incompatible`` path
        # above) so the agent gets feedback and can self-correct on the next
        # step. This single chokepoint covers every skill's ``shlex.split``.
        try:
            if self.interface == "python":
                result = self._backend.execute_python(action)
            else:
                result = self._backend.execute_cli(action)
        except ValueError as exc:
            obs = Observation(
                command_output="",
                error_output=f"shell parse error: {exc}",
                exit_code=2,
                current_config=self._backend.get_config(),
                gateway_status=self._backend.get_gateway_status(),
                task_instruction=self._active_instruction,
                step_number=self._step_count,
            )
            return self._finalize_step(action, obs, info)
        command_meta: dict[str, Any] = dict(result.meta or {})
        if result.execution_trace:
            command_meta["execution_trace"] = result.execution_trace
        if command_meta:
            info["command_meta"] = command_meta

        # Record to command history (capped at 50 to bound memory)
        self._command_history.append({
            "action": action,
            "raw_action": raw_action,
            "stdout": result.stdout[:500],
            "stderr": result.stderr[:200],
            "exit_code": result.exit_code,
        })
        if len(self._command_history) > 50:
            self._command_history.pop(0)

        # Track effects
        if result.state_changes:
            self._track_effects(result.state_changes)

        # Check if agent signaled completion
        if action.strip().lower() in ("done", "exit", "quit"):
            self._done = True

        obs = Observation(
            command_output=result.stdout,
            error_output=result.stderr if result.stderr else None,
            exit_code=result.exit_code,
            current_config=self._backend.get_config(),
            gateway_status=self._backend.get_gateway_status(),
            task_instruction=self._active_instruction,
            step_number=self._step_count,
        )

        return self._finalize_step(action, obs, info)

    def _finalize_step(
        self,
        action: str,
        obs: Observation,
        info: dict[str, Any],
    ) -> tuple[Observation, float, bool, dict[str, Any]]:
        """Common post-execution logic for all step types (standard and internal)."""
        # Check step limit
        if self._step_count >= self.max_steps:
            self._done = True
            info["truncated"] = True

        # Compute optional dense/progress evaluation
        step_eval_result: EvaluationResult | None = None
        if self.record_step_evaluation and self._evaluator:
            step_eval_result = self.evaluate()
            current_score = float(step_eval_result.score)
            info["step_evaluation"] = _summarize_evaluation_result(
                step_eval_result,
                previous_score=self._last_step_eval_score,
            )
            self._last_step_eval_score = current_score

        # Compute intermediate reward (0 unless done)
        reward = 0.0
        if self._done and self._evaluator:
            eval_result = step_eval_result or self.evaluate()
            reward = eval_result.score

        self._cumulative_reward += reward

        # Update last observation in env
        self._last_observation = obs

        if self._recorder:
            self._recorder.record_step(obs, action, reward, self._done, info)

        return obs, reward, self._done, info

    def evaluate(self) -> EvaluationResult:
        """Evaluate the current state against task goals."""
        if self._evaluator is None:
            return EvaluationResult(success=False, score=0.0)

        env_state = self._get_evaluation_state()
        task_data = {}
        if self._task and self._task.data:
            task_data = {
                "public": self._task.data.public,
                "private": self._task.data.private,
            }

        result = self._evaluator(env_state, task_data)

        if self._recorder:
            self._recorder.set_evaluation(result)

        return result

    def save_state(self) -> str:
        """Save a snapshot of the current state. Returns snapshot ID.

        Uses a fast in-memory snapshot when the backend supports it (capturing
        skill state plus environment bookkeeping). Backends that do not expose
        in-memory snapshots fall back to the legacy on-disk copytree snapshot,
        preserving the public API and snapshot format.
        """
        if self._state_manager is None:
            raise RuntimeError("Environment not initialized.")

        try:
            backend_state = self._backend.snapshot_state()
        except Exception:
            backend_state = {}

        snapshot: dict[str, Any] = {
            "backend": backend_state,
            "env": {
                "virtual_time": self._virtual_time,
                "effects": copy.deepcopy(self._effects),
                "active_sub_agents": copy.deepcopy(self._active_sub_agents),
                "background_events": copy.deepcopy(self._background_events),
                "command_history": copy.deepcopy(self._command_history),
                "step_count": self._step_count,
                "done": self._done,
                "active_instruction": self._active_instruction,
            },
        }

        # Backends without in-memory snapshot support (e.g. MockBackend) use the
        # legacy on-disk copytree path.
        if not backend_state:
            snapshot["disk_snapshot_id"] = self._disk_save_state()

        import uuid
        snapshot_id = uuid.uuid4().hex[:12]
        self._snapshots[snapshot_id] = snapshot
        return snapshot_id

    def _disk_save_state(self) -> str:
        """Legacy on-disk snapshot (copytree) used as a fallback / compat path."""
        metadata = {
            "virtual_time": self._virtual_time,
            "effects": self._effects,
            "active_sub_agents": self._active_sub_agents,
            "background_events": self._background_events,
        }
        metadata_file = Path(self._state_manager.state_dir) / "env_metadata.json"
        try:
            metadata_file.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

        snapshot_id = self._state_manager.save_snapshot()

        if metadata_file.exists():
            try:
                metadata_file.unlink()
            except Exception:
                pass

        return snapshot_id

    def load_state(self, state_id: str) -> None:
        """Restore state from a snapshot id produced by :meth:`save_state`."""
        if self._state_manager is None:
            raise RuntimeError("Environment not initialized.")

        snapshot = self._snapshots.get(state_id)
        if snapshot is None:
            # Backward compatibility: treat unknown ids as raw StateManager
            # snapshots (e.g. created directly via the legacy disk path).
            self._disk_load_state(state_id, restore_env_metadata=True)
            return

        if hasattr(self, "_agent_manager") and self._agent_manager:
            self._agent_manager.cancel_all_tasks()

        backend_state = snapshot.get("backend") or {}
        if backend_state:
            try:
                self._backend.restore_state(copy.deepcopy(backend_state))
            except Exception:
                pass
        elif "disk_snapshot_id" in snapshot:
            self._disk_load_state(
                snapshot["disk_snapshot_id"], restore_env_metadata=False
            )

        # Restore world state only. Episode-progress counters (step_count,
        # done, active_instruction) and virtual time keep moving forward so that
        # `undo` reverts effects/files/history without rewinding the budget/clock.
        env = snapshot.get("env", {})
        self._virtual_time = env.get("virtual_time", 0)
        self._effects = copy.deepcopy(env.get("effects", {})) or self._effects
        self._active_sub_agents = copy.deepcopy(env.get("active_sub_agents", {}))
        self._background_events = copy.deepcopy(env.get("background_events", []))
        if "command_history" in env:
            self._command_history = copy.deepcopy(env["command_history"])

    def _disk_load_state(
        self, state_id: str, *, restore_env_metadata: bool = True
    ) -> None:
        """Legacy on-disk restore (copytree + backend re-init)."""
        if hasattr(self, "_agent_manager") and self._agent_manager:
            self._agent_manager.cancel_all_tasks()
        self._state_manager.load_snapshot(state_id)
        env_vars = self._state_manager.get_env_vars()
        self._backend.initialize(str(self._state_manager.state_dir), env_vars)

        metadata_file = Path(self._state_manager.state_dir) / "env_metadata.json"
        if restore_env_metadata and metadata_file.exists():
            try:
                metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
                self._virtual_time = metadata.get("virtual_time", 0)
                self._effects = copy.deepcopy(metadata.get("effects", {}))
                self._active_sub_agents = copy.deepcopy(metadata.get("active_sub_agents", {}))
                self._background_events = copy.deepcopy(metadata.get("background_events", []))
            except Exception:
                pass
        # Remove restored transient metadata so it never leaks into the workspace.
        if metadata_file.exists():
            try:
                metadata_file.unlink()
            except Exception:
                pass

    def _evict_snapshot(self, snapshot_id: str) -> None:
        """Drop a snapshot and free any on-disk resources it referenced."""
        snapshot = self._snapshots.pop(snapshot_id, None)
        if not snapshot:
            return
        disk_id = snapshot.get("disk_snapshot_id")
        if disk_id and self._state_manager is not None:
            self._state_manager.drop_snapshot(disk_id)

    @property
    def task(self) -> Task | None:
        return self._task

    @property
    def trajectory(self):
        return self._recorder.trajectory if self._recorder else None

    def close(self) -> None:
        """Clean up resources."""
        self._backend.cleanup()
        self._snapshots.clear()
        self._history_stack.clear()
        if self._state_manager:
            self._state_manager.cleanup()

    def _tick(self, seconds: int) -> list[dict[str, Any]]:
        """Advance virtual time and process triggered events."""
        self._virtual_time += seconds
        # Refresh the skills' clock here too: background-event ``state_effects``
        # dispatch through ``execute_cli`` from inside ``_apply_background_event``
        # (below), bypassing ``step``'s publish, so without this those effects
        # would run against a stale ``_now``.
        self._backend.set_virtual_time(self._virtual_time)
        triggered = []
        remaining = []
        for event in self._background_events:
            due = event.get("trigger_time", 0) <= self._virtual_time
            # L3 (M8/CL2, F6): an ``async_only`` decisive sub-agent's baked
            # completion must NOT fire until the solver actually delegated to it.
            # Otherwise a no-delegate solver could advance the clock with a bare
            # ``wait`` and elicit the decisive value for free, satisfying the
            # sender-scoped ``within_window`` gate without ever going async. Keep
            # such a due-but-ungated event pending (deferred, not dropped): once
            # the solver issues ``agent delegate`` a later ``wait`` fires it. A
            # strict no-op for every non-async_only event (E6), so existing baked
            # completions still fire on a bare ``wait``.
            if due and self._async_gate_blocks(event):
                remaining.append(event)
                continue
            if due:
                self._apply_background_event(event)
                triggered.append(event)
            else:
                remaining.append(event)
        self._background_events = remaining
        return triggered

    @staticmethod
    def _normalize_agent(name: str | None) -> str:
        """Normalize an agent name/role for tolerant matching.

        Mirrors ``AgentManager.delegate``'s ``_normalize`` (lowercase, alnum-only)
        so the L3 gate matches sub-agents the same way delegation resolves them.
        """
        if not name:
            return ""
        return "".join(c for c in name.lower() if c.isalnum())

    def _is_async_only_agent(self, agent_name: str) -> bool:
        """Whether ``agent_name`` resolves to an ``async_only`` mock sub-agent.

        Reads the live ``mock_subagents`` (the same source Route 2's synchronous
        gate consults), matching by normalized ``agent_role``/``agent_name``
        substring in both directions so a completion event's full role name
        (e.g. ``Iris_SLA_Monitor``) matches its mock entry. Returns ``False`` when
        no ``async_only`` sub-agent matches -> the L3 gate is inert for every
        existing scenario until regeneration stamps ``async_only`` (E6).
        """
        target = self._normalize_agent(agent_name)
        if not target:
            return False
        mock_subagents: list[dict[str, Any]] = []
        if self._task is not None and self._task.data is not None:
            mock_subagents = self._task.data.private.get("mock_subagents", []) or []
        for sub in mock_subagents:
            if not sub.get("async_only"):
                continue
            for cand in (sub.get("agent_role"), sub.get("agent_name")):
                c = self._normalize_agent(cand)
                if c and (target in c or c in target):
                    return True
        return False

    def _was_delegated(self, agent_name: str) -> bool:
        """Whether the solver has delegated to ``agent_name`` (undo-aware).

        Consults the env-level ``_active_sub_agents`` (populated in the ``agent
        delegate`` branch and snapshotted, so undo of a delegate revokes it),
        matched by the same normalized two-way substring rule used for resolution.
        """
        target = self._normalize_agent(agent_name)
        if not target:
            return False
        for key in self._active_sub_agents:
            k = self._normalize_agent(key)
            if k and (target in k or k in target):
                return True
        return False

    def _async_gate_blocks(self, event: dict[str, Any]) -> bool:
        """Whether L3 must defer this baked completion until delegation.

        ``True`` only for a ``subagent_completion`` whose target sub-agent is
        marked ``async_only`` (the decisive one in a ``force_async_wait`` round)
        AND has not yet been reached by an ``agent delegate``. Gated on
        ``async_only`` so it is a strict no-op for every existing scenario/test:
        non-async_only baked completions (and manually appended test completions)
        still fire on a bare ``wait``.
        """
        if event.get("type") != "subagent_completion":
            return False
        name = event.get("agent_name", "")
        if not self._is_async_only_agent(name):
            return False
        return not self._was_delegated(name)

    def _apply_background_event(self, event: dict[str, Any]) -> None:
        """Apply a mocked background event to the environment state."""
        event_type = event.get("type")
        if event_type == "file_create" or event_type == "file_update":
            path = event.get("path", "")
            content = event.get("new_content", "")
            if path and self._state_manager:
                import posixpath
                # Write to disk inside workspace
                target = self._state_manager.state_dir / "workspace" / path.lstrip("/")
                target.parent.mkdir(parents=True, exist_ok=True)
                # Write background-event content verbatim. (Previously a regex
                # rewrote f-string quoting in .py files, which could silently
                # corrupt file content the grader later inspects.)
                target.write_text(content)

                # Update backend memory if available
                if hasattr(self._backend, "file_backend"):
                    file_be = self._backend.file_backend
                    norm_path = path if path.startswith("/") else "/" + path
                    norm_path = posixpath.normpath(norm_path)
                    file_be._files[norm_path] = content
        elif event_type == "subagent_completion":
            agent_name = event.get("agent_name", "unknown_agent")
            mock_response = event.get("mock_response", "Done.")
            state_effects = event.get("state_effects", [])
            reply_channel = event.get("reply_channel", "")
            
            # Update AgentManager status
            if hasattr(self, "_agent_manager") and agent_name in self._agent_manager.active_tasks:
                self._agent_manager.active_tasks[agent_name]["status"] = "completed"
                
            # Deliver response message
            self._command_history.append({
                "action": f"[System] Sub-agent '{agent_name}' completed.",
                "stdout": mock_response,
                "stderr": "",
                "exit_code": 0,
            })

            # R1: surface the subagent's spoken reply to the messages_sent grader
            # (Route 1: delegation / baked completion firing on `wait`). Without
            # this the reply lives only in command_history and is never graded.
            self._record_subagent_message(
                sender=agent_name, text=mock_response, channel=reply_channel
            )
            
            # Apply state effects
            for effect in state_effects:
                # If the effect is a raw CLI command, we execute it silently
                if isinstance(effect, str):
                    res = self._backend.execute_cli(effect)
                    self._command_history.append({
                        "action": effect,
                        "stdout": res.stdout[:500],
                        "stderr": res.stderr[:200],
                        "exit_code": res.exit_code,
                    })
                    if len(self._command_history) > 50:
                        self._command_history.pop(0)
                    # R1: a message-send effect must also reach messages_sent;
                    # executing it on the backend alone is invisible to grading.
                    parsed = self._parse_message_send_effect(effect)
                    if parsed and parsed.get("message"):
                        self._record_subagent_message(
                            sender=agent_name,
                            text=parsed["message"],
                            channel=parsed.get("channel") or reply_channel,
                            target=parsed.get("target") or None,
                        )
                elif isinstance(effect, dict):
                    action_type = effect.get("type")
                    if action_type in ("file_write", "file_create", "write_file"):
                        path = effect.get("path", "")
                        content = effect.get("content", "")
                        if path:
                            self._apply_background_event({
                                "type": "file_create",
                                "path": path,
                                "new_content": content,
                            })

        elif event_type == "session_append":
            session_id = event.get("session_id", "")
            message = event.get("message", {})
            sender = message.get("sender", "system")
            text = message.get("text", "")
            self._command_history.append({
                "action": f"[System: {session_id}] New incoming message.",
                "stdout": f"Sender: {sender}\nText: {text}",
                "stderr": "",
                "exit_code": 0,
            })
            # Route into the INCOMING bucket, not the graded ``messages_sent``
            # sink. This is a peer/system message arriving into a session the
            # solver reads -- not something the solver sent -- so counting it in
            # ``messages_sent`` let an injected provocation spuriously satisfy a
            # ``field_contains`` / ``field_semantic_match`` check about the agent's
            # OWN outgoing message (the nested ``{"session_id","message":{...}}``
            # shape was also stringified into the judge transcript). Sender-scoped
            # temporal checks (``within_window`` on ``messages_sent`` with
            # ``field=sender``) are unaffected: those decisive replies arrive via
            # the ``subagent_completion`` path (``_record_subagent_message``),
            # which still writes the flat, top-level-``sender`` record they match.
            self._effects["incoming_messages"].append({"session_id": session_id, "message": message})

            # Sync into the searchable message pool so background/late messages
            # are findable via `message search`. The MultiAppBackend keeps messages
            # inside MessageSkill (no top-level `_messages`), so the previous
            # `hasattr(self._backend, "_messages")` check silently no-op'd there
            # and injected conflict messages were invisible to the agent.
            message_skill = getattr(self._backend, "_message_skill", None)
            if message_skill is not None and hasattr(message_skill, "add_mock_message"):
                message_skill.add_mock_message(
                    channel=session_id,
                    target=message.get("target") or sender,
                    message=text,
                    media=message.get("media"),
                )
            elif hasattr(self._backend, "_messages"):
                import uuid
                self._backend._messages.append({
                    "id": message.get("message_id") or uuid.uuid4().hex[:8],
                    "target": message.get("target") or sender,
                    "message": text,
                    "channel": session_id,
                    "media": message.get("media"),
                    "type": "text",
                })
            
            # Persist back to the actual active sessions file on disk
            if self._state_manager:
                sessions_file = self._state_manager.state_dir / "sessions" / f"{session_id}.jsonl"
                sessions_file.parent.mkdir(parents=True, exist_ok=True)
                with open(sessions_file, "a") as f:
                    f.write(json.dumps(message) + "\n")
        elif event_type == "session_redaction":
            msg_id = event.get("message_id", "")
            self._command_history.append({
                "action": f"[System] Message redacted/withdrawn.",
                "stdout": f"Message ID {msg_id} has been redacted from the active session.",
                "stderr": "",
                "exit_code": 0,
            })
            # Redaction withdraws a previously INJECTED session message, which now
            # lives in ``incoming_messages`` (the ``session_append`` sink), matched
            # by the nested ``message.message_id``. Filtering ``messages_sent`` here
            # would no longer find it (and must not touch the solver's own sends).
            self._effects["incoming_messages"] = [
                m for m in self._effects["incoming_messages"]
                if m.get("message", {}).get("message_id") != msg_id
            ]
        # Note: Additional event types will be added as required by scenarios.

        # Stamp effects delivered by this event with the event's own
        # ``trigger_time`` (when the subagent/update actually completed), so the
        # grader's temporal operators read the true occurrence time -- NOT the
        # post-``_tick`` clock, which would inflate the stamp when the solver
        # over-waits. Falls back to the current clock for events without an
        # explicit trigger_time (e.g. the recursive file_create below, which
        # emits no graded effect). Idempotent for effects already stamped (M8/CL2).
        self._stamp_new_effects(at=event.get("trigger_time"))

    @staticmethod
    def _parse_message_send_effect(cmd: str) -> dict[str, str] | None:
        """Parse a ``message send``/``broadcast`` CLI-string effect.

        Mock ``state_effects`` are shipped as raw CLI strings. When a subagent's
        effect is a message send, we want its content routed into graded
        ``messages_sent`` (not just executed on the backend, which the grader
        never sees). Returns ``{target, message, channel}`` or ``None``.
        """
        s = (cmd or "").strip()
        low = s.lower()
        if not (low.startswith("message send") or low.startswith("message broadcast")):
            return None
        import shlex
        try:
            toks = shlex.split(s)
        except ValueError:
            return None

        def _flag(name: str) -> str:
            if name in toks:
                i = toks.index(name)
                if i + 1 < len(toks):
                    return toks[i + 1]
            return ""

        return {
            "target": _flag("--target"),
            "message": _flag("--message"),
            "channel": _flag("--channel"),
        }

    def _record_subagent_message(
        self,
        sender: str,
        text: str,
        channel: str,
        target: str | None = None,
    ) -> None:
        """Route a mock subagent's spoken reply into graded ``messages_sent``.

        The ``subagent_completion`` handler previously only wrote to
        ``command_history`` and executed effects on the backend, so a subagent's
        reply was invisible to the ``messages_sent`` grader
        (``merge_backend_state_into_eval`` intentionally never backfills effects
        from the backend snapshot). Append the reply in the SAME flat shape a
        solver ``message send`` produces so ``field_semantic_match`` /
        ``field_contains`` checks about the subagent can match, carrying
        ``sender`` for attribution. Idempotent on ``(sender, text, channel)`` so
        the completion path and the message-trigger path cannot double-count one
        utterance.
        """
        text = (text or "").strip()
        if not text:
            return
        channel = channel or ""
        tgt = target or channel
        for m in self._effects["messages_sent"]:
            if (
                m.get("sender") == sender
                and m.get("message") == text
                and (m.get("channel") or "") == channel
            ):
                return
        import uuid
        record = {
            "id": uuid.uuid4().hex[:8],
            "sender": sender,
            "target": tgt,
            "message": text,
            "channel": channel,
            "type": "text",
        }
        self._effects["messages_sent"].append(record)

        # Mirror into the searchable pool so the solver's `message search` finds
        # it, matching the `session_append` sync path above.
        message_skill = getattr(self._backend, "_message_skill", None)
        if message_skill is not None and hasattr(message_skill, "add_mock_message"):
            message_skill.add_mock_message(
                channel=channel, target=tgt, message=text, media=None
            )
        elif hasattr(self._backend, "_messages"):
            self._backend._messages.append({
                "id": record["id"],
                "target": tgt,
                "message": text,
                "channel": channel,
                "media": None,
                "type": "text",
            })

        # Persist to the active session file on disk for on-disk consistency.
        if self._state_manager and channel:
            sessions_file = self._state_manager.state_dir / "sessions" / f"{channel}.jsonl"
            sessions_file.parent.mkdir(parents=True, exist_ok=True)
            with open(sessions_file, "a") as f:
                f.write(json.dumps({"sender": sender, "text": text}) + "\n")

    def _get_evaluation_state(self) -> dict[str, Any]:
        """Build the state dict used for evaluation."""
        state: dict[str, Any] = {
            "config": self._backend.get_config(),
            "gateway_status": self._backend.get_gateway_status(),
            "effects": dict(self._effects),
            "step_count": self._step_count,
            "task_instruction": self._active_instruction,
            "canonical_instruction": (
                (self._task.canonical_instruction or self._task.instruction)
                if self._task
                else ""
            ),
            "command_history": list(self._command_history),
        }

        if self._last_observation:
            state["last_stdout"] = self._last_observation.command_output
            state["last_stderr"] = self._last_observation.error_output or ""
            state["last_exit_code"] = self._last_observation.exit_code

        backend_state = self._backend.get_state()

        # In real mode, ensure openclaw live state is still available even when
        # upstream backend returns sparse state.
        if isinstance(self._backend, MultiAppBackend) and self._backend.real_backend is not None:
            real_state = self._backend.real_backend.get_state()
            if "agents" not in backend_state and "agents" in real_state:
                backend_state["agents"] = real_state["agents"]
            if "channels" not in backend_state and "channels" in real_state:
                backend_state["channels"] = real_state["channels"]

        merge_backend_state_into_eval(state, state["effects"], backend_state)

        # Normalize possibly list-shaped live state and backfill from effects.
        state["agents"] = as_named_mapping(state.get("agents"))
        state["channels"] = as_named_mapping(state.get("channels"))
        merge_named_entities(state["agents"], state["effects"].get("agents_created", []))
        merge_named_entities(state["channels"], state["effects"].get("channels_configured", []))

        return state

    def _track_effects(self, changes: dict[str, Any]) -> None:
        """Track state changes as effects for evaluation."""
        if "agents" in changes:
            for name, agent_data in changes["agents"].items():
                self._effects["agents_created"].append(
                    {"name": name, **agent_data}
                )
        if "messages" in changes:
            self._effects["messages_sent"].extend(changes["messages"])
            # Process message-based triggers from mock subagents
            if self._task and hasattr(self._task, "data") and self._task.data:
                mock_subagents = self._task.data.private.get("mock_subagents", [])
                for msg in changes["messages"]:
                    target = msg.get("target", "")
                    text = msg.get("message", "")
                    channel = msg.get("channel", "")
                    
                    for subagent in mock_subagents:
                        sub_name = subagent.get("agent_name", subagent.get("agent_role", ""))
                        # If the sent message is targeted to this subagent
                        if sub_name.lower() in target.lower():
                            # L2 async-arrival gate (M8/CL2, F6): an
                            # ``async_only`` sub-agent (the decisive one in a
                            # ``force_async_wait`` round) must NOT surface any
                            # effect synchronously here. Its preset reply and
                            # state_effects are delivered ONLY when ``wait``
                            # advances the clock and fires its
                            # ``subagent_completion`` background event (Route 1),
                            # stamped at the true ``trigger_time``. Skipping the
                            # synchronous mirror prevents a no-wait solver from
                            # eliciting the decisive value at ``virtual_time=0``
                            # and satisfying the sender-scoped ``within_window``
                            # gate without ever waiting. Safe: the materializer
                            # always pairs such a sub-agent with a matching
                            # ``subagent_completion`` event, so nothing is lost --
                            # only deferred to completion time.
                            if subagent.get("async_only"):
                                continue
                            # R2 (Route 2): the solver directly contacted this
                            # subagent, so elicit its preset reply into graded
                            # messages_sent - even when the subagent's effects
                            # are plain CLI strings (the shipped form the
                            # dict-trigger branch below never handled). Reply on
                            # the subagent's canonical reply_channel, falling
                            # back to the channel the solver used. Deduped inside
                            # _record_subagent_message so Route 1 (completion on
                            # `wait`) cannot double-count the same utterance.
                            r2_channel = subagent.get("reply_channel") or channel
                            r2_resp = subagent.get("mock_response", "")
                            if r2_resp:
                                self._record_subagent_message(
                                    sender=sub_name,
                                    text=r2_resp,
                                    channel=r2_channel,
                                )
                            for _eff in subagent.get("state_effects", []):
                                if isinstance(_eff, str):
                                    _parsed = self._parse_message_send_effect(_eff)
                                    if _parsed and _parsed.get("message"):
                                        self._record_subagent_message(
                                            sender=sub_name,
                                            text=_parsed["message"],
                                            channel=_parsed.get("channel") or r2_channel,
                                            target=_parsed.get("target") or None,
                                        )
                            effects = subagent.get("state_effects", [])
                            for effect in effects:
                                if not isinstance(effect, dict):
                                    continue
                                trigger = effect.get("trigger", {})
                                trigger_type = trigger.get("type")
                                
                                # Match trigger conditions
                                matched = False
                                if trigger_type == "message":
                                    conditions = trigger.get("conditions", [])
                                    matched = True
                                    for cond in conditions:
                                        val = cond.get("value", "")
                                        op = cond.get("operator", "contains")
                                        if op == "contains" and val.lower() not in text.lower():
                                            matched = False
                                            break
                                elif trigger_type == "message_received":
                                    # Semantic matching trigger (fallback to keyword match or match target)
                                    matched = True
                                
                                if matched:
                                    response = effect.get("response", {})
                                    resp_text = response.get("message")
                                    if not resp_text and "actions" in effect:
                                        # Extract from actions list
                                        for action in effect["actions"]:
                                            if action.get("type") == "send_message":
                                                resp_text = action.get("message")
                                                break
                                    
                                    if resp_text:
                                        # Append the response message to the chat channel
                                        import uuid
                                        self._apply_background_event({
                                            "type": "session_append",
                                            "session_id": channel,
                                            "message": {
                                                "message_id": uuid.uuid4().hex[:8],
                                                "sender": sub_name,
                                                "timestamp": "2024-10-24T14:40:00Z",
                                                "text": resp_text,
                                                "attachments": [],
                                                "is_redacted": False
                                            }
                                        })
        if "config" in changes:

            self._effects["config_changes"].append(changes["config"])
        if "plugins" in changes:
            self._effects["plugins_installed"].extend(changes["plugins"])
        if "cron_jobs" in changes:
            self._effects["cron_jobs_created"].extend(changes["cron_jobs"])
        # New app effects
        if "calendar_events_created" in changes:
            self._effects["calendar_events_created"].extend(changes["calendar_events_created"])
        if "calendar_events_updated" in changes:
            self._effects["calendar_events_updated"].extend(changes["calendar_events_updated"])
        if "calendar_events_deleted" in changes:
            self._effects["calendar_events_deleted"].extend(changes["calendar_events_deleted"])
        if "emails_sent" in changes:
            self._effects["emails_sent"].extend(changes["emails_sent"])
        if "emails_moved" in changes:
            self._effects["emails_moved"].extend(changes["emails_moved"])
        if "files_created" in changes:
            self._effects["files_created"].extend(changes["files_created"])
        if "files_appended" in changes:
            # An append is recorded as a first-class ``files_appended`` effect AND
            # folded into ``files_created`` so content-based grading observes the
            # appended text. Rationale: ``effect_checker.field_contains`` matches
            # against ``files_created[].content``; before this, ``file append``
            # emitted no effect at all, so appending the decisive answer to a
            # report was invisible to grading. ``full_content`` is the
            # authoritative post-append file body from the backend.
            for entry in changes["files_appended"]:
                if not isinstance(entry, dict):
                    continue
                path = entry.get("path")
                delta = entry.get("content", "")
                full_content = entry.get("full_content")
                self._effects["files_appended"].append(
                    {"path": path, "content": delta}
                )
                folded = False
                for created in self._effects["files_created"]:
                    if created.get("path") == path:
                        created["content"] = (
                            full_content
                            if full_content is not None
                            else str(created.get("content", "")) + str(delta)
                        )
                        folded = True
                        break
                if not folded:
                    self._effects["files_created"].append(
                        {
                            "path": path,
                            "content": full_content if full_content is not None else delta,
                        }
                    )
        if "files_deleted" in changes:
            self._effects["files_deleted"].extend(changes["files_deleted"])
        if "tasks_created" in changes:
            self._effects["tasks_created"].extend(changes["tasks_created"])
        if "tasks_completed" in changes:
            self._effects["tasks_completed"].extend(changes["tasks_completed"])
        if "assets_accessed" in changes:
            self._effects["assets_accessed"].extend(changes["assets_accessed"])
        if "assets_opened" in changes:
            self._effects["assets_opened"].extend(changes["assets_opened"])

        # Generic protocol-effect passthrough. Skills built on
        # StatefulProtocolSkill emit effects under a nested ``effects`` key
        # (``{"effects": {<effect_type>: [payload, ...]}}``) so a new command
        # family surfaces its effects without editing this method. Buckets are
        # created lazily; ``reset()`` clears them like any other effect bucket.
        # No shipped producer sets ``changes["effects"]``, so this is a no-op for
        # all current skills/backends.
        nested_effects = changes.get("effects")
        if isinstance(nested_effects, dict):
            for effect_type, effect_list in nested_effects.items():
                if isinstance(effect_list, list):
                    self._effects.setdefault(effect_type, []).extend(effect_list)

        self._stamp_new_effects()

    def _stamp_new_effects(self, at: float | None = None) -> None:
        """Stamp every effect dict lacking ``virtual_time`` with an emission time.

        Enables the grader's temporal operators (``within_window``; future
        ``ordered_before``) to reason about *when* an effect was emitted (M8/CL2,
        F5/F11). Idempotent via ``setdefault``: an effect recorded in an earlier
        step keeps the timestamp captured when it was emitted, so only the
        effects added by the current step/tick pick up the new stamp.

        ``at`` is the emission time to stamp:

        * ``_track_effects`` (solver's own actions) calls with ``at=None`` so the
          effect is stamped with ``self._virtual_time`` -- *when the solver
          acted*.
        * ``_apply_background_event`` calls with ``at=<event trigger_time>`` so a
          delegated ``subagent_completion`` (or a ``session_append``) fired on a
          ``wait`` is stamped with the time the event actually *occurred*
          (``trigger_time``), NOT the post-``_tick`` clock. This matters for the
          temporal grader: the subagent finishes at its ``trigger_time`` no matter
          how long the solver over-waits, so an on-time assertion stays correct
          (and correct-by-construction for the reference trajectory, whose single
          injected ``wait`` advances the clock to ``max_trigger``).

        Purely additive to runtime effect dicts; ``self._effects``/``env_state``
        are never written into generated scenario JSON, so this does not affect
        the E6/M0 byte gate (only in-memory effect snapshots gain a key).
        """
        vt = self._virtual_time if at is None else at
        for bucket in self._effects.values():
            if not isinstance(bucket, list):
                continue
            for effect in bucket:
                if isinstance(effect, dict):
                    effect.setdefault("virtual_time", vt)

    def _resolve_task_instruction(self, task: Task) -> str:
        """Resolve the instruction shown to the agent for this episode."""
        canonical = task.canonical_instruction or task.instruction
        if self.instruction_variant_mode != "sampled":
            return task.instruction

        variants = task.variant_texts()
        if not variants:
            return task.instruction

        seed_material = f"{task.task_id}:{self.experiment_name}:{canonical}"
        seed = int.from_bytes(
            hashlib.sha256(seed_material.encode("utf-8")).digest()[:8],
            "big",
        )
        return variants[seed % len(variants)]


def _summarize_evaluation_result(
    result: EvaluationResult,
    *,
    previous_score: float,
) -> dict[str, Any]:
    """Return a compact serializable per-step evaluation summary."""
    passed = [detail.name for detail in result.details if detail.passed]
    failed = [detail.name for detail in result.details if not detail.passed]
    current_score = float(result.score)
    return {
        "success": bool(result.success),
        "current_score": current_score,
        "score_delta": current_score - float(previous_score),
        "passed_checks": passed,
        "failed_checks": failed,
    }


def _create_backend(backend_type: str, **kwargs) -> BaseBackend:
    """Create a backend instance by type name.

    Extra keyword arguments are forwarded to the backend constructor
    (only meaningful for ``"hybrid"`` currently).
    """
    if backend_type == "mock":
        return MockBackend()
    if backend_type in {"multi", "martin"}:
        return MultiAppBackend()
    if backend_type == "real":
        real_kwargs = dict(kwargs)
        fallback_to_mock = bool(real_kwargs.pop("fallback_openclaw_network_to_mock", False))
        strict_online_data = bool(real_kwargs.pop("strict_online_data", True))
        return MultiAppBackend(
            real_openclaw=True,
            real_openclaw_kwargs=real_kwargs,
            fallback_openclaw_network_to_mock=fallback_to_mock,
            strict_online_data=strict_online_data,
            real_apps=True,
        )
    if backend_type == "hybrid":
        from clasharness.backend.hybrid_backend import HybridBackend
        return HybridBackend(**kwargs)
    raise ValueError(f"Unknown backend type: {backend_type}")


def make_env(
    task_id: str,
    task_data_dir: str | Path | None = None,
    mode: str = "mock",
    backend_kwargs: dict | None = None,
    **kwargs,
) -> CLASHarnessEnv:
    """Convenience factory for creating a CLASHarnessEnv.

    Args:
        task_id: Task identifier (e.g. ``"calendar_add_event_1"``).
        task_data_dir: Optional task-data root containing ``tasks/`` and
            ``datasets/``. Defaults to built-in ``clasharness/data``.
        mode: Backend mode — ``'mock'``, ``'multi'``, ``'martin'``,
            ``'real'``, or ``'hybrid'``. ``martin`` is an alias for
            the local multi-app sandbox.
        backend_kwargs: Extra keyword arguments forwarded to the backend
            constructor. Currently only used by ``'hybrid'`` mode
            (e.g. ``{"auto_start_gateway": True, "gateway_port": 19000}``).
        **kwargs: Forwarded to :class:`CLASHarnessEnv`.
    """
    _MODE_MAP = {"mock": "mock", "multi": "multi", "martin": "martin", "real": "real", "hybrid": "hybrid"}
    backend = _MODE_MAP.get(mode, "mock")
    return CLASHarnessEnv(
        task_id=task_id,
        task_data_dir=task_data_dir,
        backend=backend,
        backend_kwargs=backend_kwargs,
        **kwargs,
    )


def _get_available_commands(
    domains: list[str],
) -> list[str]:
    """Get command hints based on task domains."""
    domain_commands: dict[str, list[str]] = {
        "setup_config": [
            "system setup",
            "system config set <path> <value>",
        ],
        "messaging": [
            "message send --target <target> --message <text>",
            "message broadcast --targets <t1> <t2> --message <text>",
            "message poll --target <target> --poll-question <q> --poll-option <o1> --poll-option <o2>",
            "message search --channel discord --guild-id <gid> --channel-id <cid> --query <text>",
        ],
        "agent_mgmt": [
            "agent add <name> --model <model>",
            "agent list",
            "agent delete --name <name>",
            "agent set-identity --agent <name> --emoji <emoji>",
        ],
        "channel_mgmt": [
            "channel login --channel <name>",
            "channel list",
            "system config set channels.<name>.config.<key> <value>",
        ],
        "monitoring": [
            "system status",
            "system status --json",
            "system health",
            "system logs",
            "system doctor",
            "system sessions",
        ],
        "plugin_skill": [
            "plugin enable <name>",
            "plugin disable <name>",
            "plugin list",
            "plugin skill info <name>",
        ],
        "cron_webhook": [
            "system cron add --name <job> --cron <expr> --message <text>",
            "system cron list",
            "system config set integrations.webhooks.primary.url <url>",
        ],
        "security": [
            "system security",
            "system config set gateway.auth.token <token>",
        ],
        "device_node": [
            "system devices pair",
            "system devices list",
        ],
        "calendar": [
            "calendar list [--from DATE] [--to DATE]",
            "calendar add-event --title TITLE --start DATETIME [--location LOC] [--attendees A,B]",
            "calendar update-event --id ID [--title TITLE] [--start DATETIME] [--location LOC]",
            "calendar delete-event --id ID",
            "calendar search --query QUERY",
            "calendar today [--timezone TZ]",
        ],
        "email": [
            "email list [--folder FOLDER] [--unread] [--from SENDER]",
            "email read --id ID",
            "email send --to ADDR --subject SUBJ --body BODY",
            "email reply --id ID --body BODY",
            "email search --query QUERY",
            "email move --id ID --folder FOLDER",
            "email mark --id ID --flag {read,unread,starred}",
        ],
        "weather": [
            "weather get --location LOC [--date DATE]",
            "weather forecast --location LOC [--days N]",
            "weather alerts --location LOC",
        ],
        "file": [
            "file create --path PATH --content CONTENT",
            "file read --path PATH",
            "file delete --path PATH",
            "file list [--path PATH]",
            "file move --src SRC --dst DST",
            "file append --path PATH --content CONTENT",
        ],
        "tasks": [
            "tasks list [--status {all,pending,done}] [--priority {high,medium,low}]",
            "tasks add --title TITLE [--due DATE] [--priority PRIORITY]",
            "tasks complete --id ID",
            "tasks delete --id ID",
            "tasks search --query QUERY",
        ],
        "asset": [
            "asset list [--kind KIND]",
            "asset info --id ID",
            "asset read-text --id ID",
            "asset describe --id ID",
            "asset open --id ID",
        ],
        "multimodal": [
            "asset list [--kind KIND]",
            "asset info --id ID",
            "asset read-text --id ID",
            "asset describe --id ID",
            "asset open --id ID",
        ],
    }

    commands: list[str] = []
    for domain in domains:
        commands.extend(domain_commands.get(domain, []))
    # Always include general commands plus the temporal / async control
    # primitives. These are the crux of conflict scenarios (temporal staleness,
    # subagent delay) and must be discoverable, not hidden from the agent.
    commands.extend([
        "system status",
        "system status --json",
        "wait <seconds>  # advance virtual time so async/background tasks can surface",
        "agent delegate --agent <name> --instruction <text>  # start an async subtask",
        "agent status [--agent <name>]  # check pending async subagent tasks",
        "undo [N]  # revert the last N steps (state reverts; virtual time does not)",
        "done  # signal task completion",
    ])
    return list(dict.fromkeys(commands))  # deduplicate preserving order


def _as_named_mapping(raw: Any) -> dict[str, Any]:
    """Normalize backend entity containers to ``{name: payload}`` mappings."""
    return as_named_mapping(raw)


def _merge_named_entities(dst: dict[str, Any], entries: Any) -> None:
    """Merge effect entries that carry a ``name`` field into ``dst``."""
    merge_named_entities(dst, entries)


# Legacy alias for backwards compatibility
OpenClawEnv = CLASHarnessEnv
