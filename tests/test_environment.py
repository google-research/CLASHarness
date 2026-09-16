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

"""Integration tests for CLASHarnessEnv."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from clasharness.core.environment import CLASHarnessEnv
from clasharness.core.task import load_task_ids


# Use a few concrete task IDs that are always generated
_TASK_IDS = [
    "agent_create_1",
    "msg_send_text_1",
    "check_system_status_1",
    "setup_workspace_1",
]


class TestEnvironmentBasic:
    def test_reset_returns_observation(self):
        with CLASHarnessEnv(task_id="agent_create_1", backend="mock") as env:
            obs = env.reset()
            assert obs.task_instruction != ""
            assert obs.step_number == 0

    def test_step_increments_count(self):
        with CLASHarnessEnv(task_id="agent_create_1", backend="mock") as env:
            env.reset()
            obs, reward, done, info = env.step("openclaw status")
            assert obs.step_number == 1
            assert info["step"] == 1

    def test_done_after_done_signal(self):
        with CLASHarnessEnv(task_id="agent_create_1", backend="mock") as env:
            env.reset()
            _, _, done, _ = env.step("done")
            assert done is True

    def test_context_manager(self):
        with CLASHarnessEnv(task_id="msg_send_text_1", backend="mock") as env:
            obs = env.reset()
            assert env.task is not None

    def test_step_requires_reset(self):
        env = CLASHarnessEnv(task_id="agent_create_1", backend="mock")
        with pytest.raises(RuntimeError, match="reset"):
            env.step("openclaw status")

    def test_step_after_done_raises(self):
        with CLASHarnessEnv(task_id="agent_create_1", backend="mock") as env:
            env.reset()
            env.step("done")
            with pytest.raises(RuntimeError, match="done"):
                env.step("openclaw status")


class TestExpertAgent:
    """Test that expert solutions achieve full scores."""

    @pytest.mark.parametrize("task_id", _TASK_IDS)
    def test_expert_solution_passes(self, task_id):
        with CLASHarnessEnv(task_id=task_id, backend="mock") as env:
            obs = env.reset()
            task = env.task
            assert task is not None
            assert task.ground_truth is not None

            for command in task.ground_truth.solution_commands:
                obs, reward, done, info = env.step(command)
                if done:
                    break

            result = env.evaluate()
            assert result.success is True, (
                f"Expert solution failed for {task_id}: "
                f"{[(d.name, d.message) for d in result.details if not d.passed]}"
            )
            assert result.score == pytest.approx(1.0)

    def test_agent_create_passes(self):
        with CLASHarnessEnv(task_id="agent_create_1", backend="mock") as env:
            obs = env.reset()
            task = env.task
            for cmd in task.ground_truth.solution_commands:
                env.step(cmd)
            result = env.evaluate()
            assert result.success is True

    def test_msg_send_passes(self):
        with CLASHarnessEnv(task_id="msg_send_text_1", backend="mock") as env:
            obs = env.reset()
            task = env.task
            for cmd in task.ground_truth.solution_commands:
                env.step(cmd)
            result = env.evaluate()
            assert result.success is True

    def test_wrong_action_fails(self):
        """Wrong action should fail evaluation."""
        with CLASHarnessEnv(task_id="agent_create_1", backend="mock") as env:
            env.reset()
            # Do something unrelated instead
            env.step("openclaw status")
            result = env.evaluate()
            assert result.success is False


class TestSafetyGuard:
    def test_dangerous_command_blocked(self):
        with CLASHarnessEnv(
            task_id="agent_create_1",
            backend="mock",
            raise_on_safety_violation=True,
        ) as env:
            env.reset()
            obs, reward, done, info = env.step("rm -rf /")
            assert info.get("safety_violation") is True
            assert done is True

    def test_safe_command_allowed(self):
        with CLASHarnessEnv(task_id="agent_create_1", backend="mock") as env:
            env.reset()
            obs, reward, done, info = env.step("openclaw status")
            assert info.get("safety_violation") is None


class TestTrajectory:
    def test_trajectory_recorded(self):
        with CLASHarnessEnv(
            task_id="agent_create_1", backend="mock", record_trajectory=True
        ) as env:
            env.reset()
            env.step("openclaw agents add --name alice --model openai/gpt-4o")
            env.evaluate()

            traj = env.trajectory
            assert traj is not None
            assert len(traj.steps) == 1
            assert traj.steps[0].action == "agent add alice --model openai/gpt-4o"

    def test_trajectory_dict(self):
        with CLASHarnessEnv(
            task_id="agent_create_1", backend="mock", record_trajectory=True
        ) as env:
            env.reset()
            env.step("openclaw status")

            traj = env.trajectory
            d = traj.to_dict()
            assert "task_id" in d
            assert "trajectory" in d
            assert len(d["trajectory"]) == 1


class TestStateSnapshot:
    def test_save_and_restore(self):
        with CLASHarnessEnv(task_id="agent_create_1", backend="mock") as env:
            env.reset()
            snapshot_id = env.save_state()
            assert snapshot_id is not None

            env.step("openclaw agents add --name temp --model openai/gpt-4o")
            env.load_state(snapshot_id)
            # After restore, backend is re-initialized from snapshot


class TestMessagesSentSSOT:
    """Stage 0-② effect SSOT: every communication action feeds the single
    messages_sent sink uniformly (send, broadcast, poll, react, delegate).

    Regression guard for the audited bug where broadcast/poll/react produced no
    messages_sent effect and delegate was special-cased into _effects directly.
    """

    def test_send_broadcast_poll_react_all_recorded(self):
        with CLASHarnessEnv(task_id="agent_create_1", backend="mock") as env:
            env.reset()
            env.step("openclaw message send --channel slack --target @a --message 'hi'")
            env.step("openclaw message broadcast --targets @x,@y,@z --message 'bcast'")
            env.step("openclaw message poll --target #g --question 'Q?' --options yes,no")
            env.step("openclaw message react --target @a --emoji ok --channel slack")
            sent = env._effects["messages_sent"]
            # 1 send + 3 broadcast + 1 poll + 1 react = 6
            assert len(sent) == 6
            types = [m.get("type") for m in sent]
            assert types.count("text") == 4
            assert types.count("poll") == 1
            assert types.count("reaction") == 1

    def test_broadcast_records_each_target(self):
        with CLASHarnessEnv(task_id="agent_create_1", backend="mock") as env:
            env.reset()
            env.step("openclaw message broadcast --targets @x,@y,@z --message 'bcast'")
            sent = env._effects["messages_sent"]
            assert {m["target"] for m in sent} == {"@x", "@y", "@z"}

    def test_broadcast_channel_routed_reaches_messages_sent(self):
        """The documented ``--channel`` form must feed ``messages_sent``.

        ``agent_tools.v1.json`` exposes only ``channel``/``message`` for
        ``message.broadcast``, so this is the only form an agent can emit. It
        used to return exit 0 with zero records -- a silent no-op invisible to
        the grader -- which hit every model in the eval (167/167 broadcast calls
        recorded nothing) and the shipped ground-truth solutions too.
        """
        with CLASHarnessEnv(task_id="agent_create_1", backend="mock") as env:
            env.reset()
            env.step("openclaw message broadcast --channel '#ops' --message 'all clear'")
            sent = env._effects["messages_sent"]
            assert len(sent) == 1
            assert sent[0]["target"] == "#ops"
            assert sent[0]["channel"] == "#ops"
            assert sent[0]["message"] == "all clear"
            assert sent[0]["type"] == "text"

    def test_broadcast_without_channel_or_targets_fails_loudly(self):
        """A zero-target broadcast must not report success."""
        with CLASHarnessEnv(task_id="agent_create_1", backend="mock") as env:
            env.reset()
            env.step("openclaw message broadcast --message 'orphan'")
            assert env._effects["messages_sent"] == []

    def test_poll_channel_routed_reaches_messages_sent(self):
        """The documented ``--channel`` form must route the poll.

        ``message poll --channel NAME --question TEXT --options A,B`` is the
        only form the agent-facing spec exposes, yet ``--channel`` used to be
        dropped: the effect landed with an empty target and no channel, so no
        target/channel-scoped check could ever match it.
        """
        with CLASHarnessEnv(task_id="agent_create_1", backend="mock") as env:
            env.reset()
            env.step(
                "openclaw message poll --channel '#g' --question 'Q?' --options yes,no"
            )
            sent = env._effects["messages_sent"]
            assert len(sent) == 1
            assert sent[0]["type"] == "poll"
            assert sent[0]["target"] == "#g"
            assert sent[0]["channel"] == "#g"

    def test_react_message_id_reaches_messages_sent(self):
        """``--message-id`` is documented and must survive into the effect."""
        with CLASHarnessEnv(task_id="agent_create_1", backend="mock") as env:
            env.reset()
            env.step(
                "openclaw message react --channel slack --message-id m-7 --emoji ok"
            )
            sent = env._effects["messages_sent"]
            assert len(sent) == 1
            assert sent[0]["type"] == "reaction"
            assert sent[0]["message_id"] == "m-7"
            assert sent[0]["target"] == "slack"
            assert sent[0]["channel"] == "slack"

    def test_delegate_routes_through_single_sink(self):
        with CLASHarnessEnv(task_id="agent_create_1", backend="mock") as env:
            env.reset()
            env.step("agent delegate --agent Dev --instruction 'verify logs'")
            sent = env._effects["messages_sent"]
            assert len(sent) == 1
            assert sent[0]["recipient"] == "Dev"
            assert sent[0]["message"] == "verify logs"
            assert sent[0]["type"] == "delegation"


class TestDelegateArgParsing:
    """Bug 3: ``agent delegate`` must tolerate an unquoted multi-word instruction.

    The agent-facing template documents ``agent delegate --agent <name>
    --instruction <text>`` with no quoting guidance, so models routinely emit an
    UNQUOTED free-text instruction. ``shlex.split`` explodes that prose into many
    argv tokens; the old strict argparse rejected the trailing words as
    "unrecognized arguments", silently dropped the whole delegation, and (worse)
    reported the flags as missing. These guard the tolerant parser + accurate,
    flag-specific error messages.
    """

    def test_unquoted_multiword_instruction_delegates(self):
        with CLASHarnessEnv(task_id="agent_create_1", backend="mock") as env:
            env.reset()
            obs, _, _, _ = env.step(
                "agent delegate --agent reviewer --instruction please review "
                "the deployment and halt if anything is unsafe"
            )
            # Delegation succeeds instead of failing with "unrecognized arguments".
            assert "successfully" in obs.command_output.lower()
            sent = env._effects["messages_sent"]
            assert len(sent) == 1
            assert sent[0]["recipient"] == "reviewer"
            assert sent[0]["type"] == "delegation"
            # The full multi-word instruction is reassembled intact (not truncated
            # to the first word, and no words dropped).
            assert sent[0]["message"] == (
                "please review the deployment and halt if anything is unsafe"
            )

    def test_quoted_instruction_still_works(self):
        # Parity: a correctly quoted instruction behaves identically.
        with CLASHarnessEnv(task_id="agent_create_1", backend="mock") as env:
            env.reset()
            obs, _, _, _ = env.step(
                "agent delegate --agent Dev --instruction 'verify the logs'"
            )
            assert "successfully" in obs.command_output.lower()
            sent = env._effects["messages_sent"]
            assert sent[0]["recipient"] == "Dev"
            assert sent[0]["message"] == "verify the logs"

    def test_missing_instruction_reports_only_instruction(self):
        with CLASHarnessEnv(task_id="agent_create_1", backend="mock") as env:
            env.reset()
            obs, _, _, _ = env.step("agent delegate --agent reviewer")
            # The "requires ..." clause (before the Usage line, which always names
            # both flags) must name ONLY the truly-missing flag.
            requires = obs.command_output.split("Usage:")[0]
            assert "error" in requires.lower()
            assert "--instruction" in requires
            # Must NOT claim --agent is missing when it was supplied.
            assert "--agent" not in requires
            assert env._effects["messages_sent"] == []

    def test_missing_agent_reports_only_agent(self):
        with CLASHarnessEnv(task_id="agent_create_1", backend="mock") as env:
            env.reset()
            obs, _, _, _ = env.step("agent delegate --instruction do the thing now")
            requires = obs.command_output.split("Usage:")[0]
            assert "error" in requires.lower()
            assert "--agent" in requires
            assert "--instruction" not in requires
            assert env._effects["messages_sent"] == []

    def test_parse_delegate_args_unit(self):
        from clasharness.core.environment import _parse_delegate_args

        # Unquoted multi-word instruction: everything after --instruction is one
        # instruction (shlex has already tokenized it into words).
        assert _parse_delegate_args(
            ["--agent", "r", "--instruction", "check", "the", "gate"]
        ) == ("r", "check the gate")
        # --flag=value spelling.
        assert _parse_delegate_args(
            ["--agent=r", "--instruction=check", "the", "gate"]
        ) == ("r", "check the gate")
        # Instruction before agent: the later --agent terminates the instruction.
        assert _parse_delegate_args(
            ["--instruction", "check", "the", "gate", "--agent", "r"]
        ) == ("r", "check the gate")
        # Missing flags surface as None (drives the flag-specific error message).
        assert _parse_delegate_args(["--agent", "r"]) == ("r", None)
        assert _parse_delegate_args(["--instruction", "go", "now"]) == (None, "go now")


class TestMaxSteps:
    def test_truncated_at_max_steps(self):
        with CLASHarnessEnv(
            task_id="agent_create_1", backend="mock", max_steps=3
        ) as env:
            env.reset()
            for _ in range(3):
                obs, reward, done, info = env.step("openclaw status")
                if done:
                    break
            assert done is True
            assert info.get("truncated") is True


class TestRealExecutionGate:
    """The safety gate relaxes strict simulator safety only for real execution.

    ``_is_real_execution`` unifies on the canonical ``real_enabled`` signal plus
    the legacy ``real_backend`` delegate, replacing an inverted
    ``isinstance(MultiAppBackend) and real_backend is not None`` check.
    """

    def test_gate_relaxes_real_backends_only(self):
        from clasharness.backend.hybrid_backend import HybridBackend
        from clasharness.backend.mock_backend import MockBackend
        from clasharness.backend.multi_app_backend import MultiAppBackend

        with CLASHarnessEnv(task_id="agent_create_1", backend="mock") as env:
            original = env._backend
            try:
                env._backend = MockBackend()
                assert env._is_real_execution() is False

                env._backend = MultiAppBackend(real_apps=False)
                assert env._is_real_execution() is False

                # Regression: real app execution must be treated as real. The old
                # gate returned False here (real_backend is None) -> wrongly strict.
                env._backend = MultiAppBackend(real_apps=True)
                assert env._is_real_execution() is True

                # HybridBackend keeps real_apps=False but exposes a real delegate;
                # it must stay relaxed (unchanged behavior).
                env._backend = HybridBackend()
                assert env._is_real_execution() is True
            finally:
                env._backend = original


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__]))
