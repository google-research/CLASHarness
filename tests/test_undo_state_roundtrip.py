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

"""Phase 2: snapshot/undo state-model soundness tests (multi backend)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from clasharness.core.environment import CLASHarnessEnv


def _history_text(env: CLASHarnessEnv) -> str:
    return " | ".join(entry.get("action", "") for entry in env._command_history)


def test_undo_reverts_agent_and_command_history():
    with CLASHarnessEnv(task_id="agent_create_1", backend="multi") as env:
        env.reset()
        env.step("agent add alice")
        env.step("agent add bob")

        agents = env._backend.get_state().get("agents", {})
        assert "alice" in agents and "bob" in agents

        env.step("undo 1")

        agents = env._backend.get_state().get("agents", {})
        assert "alice" in agents, "first action must survive undo"
        assert "bob" not in agents, "second action must be reverted"

        # command_history must also revert (CommandChecker relies on it).
        history = _history_text(env)
        assert "agent add alice" in history
        assert "agent add bob" not in history

        # Effects must revert in lockstep with backend state.
        created = [a.get("name") for a in env._effects.get("agents_created", [])]
        assert "alice" in created
        assert "bob" not in created


def test_undo_reverts_system_skill_memory():
    """cron/plugin state lives in skill memory; undo must restore it."""
    with CLASHarnessEnv(task_id="agent_create_1", backend="multi") as env:
        env.reset()
        env.step("agent add alice")  # snapshot boundary before cron add
        env.step('system cron add --name nightly --cron "0 0 * * *" --message hi')

        crons = env._backend.get_state().get("cron_jobs", [])
        assert any(c.get("name") == "nightly" for c in crons)

        env.step("undo 1")

        crons = env._backend.get_state().get("cron_jobs", [])
        assert not any(c.get("name") == "nightly" for c in crons), (
            "cron added after the snapshot must be reverted by undo"
        )


def test_undo_reverts_messages_not_wiped():
    """Regression: undo must restore messages to the snapshot, not wipe them."""
    with CLASHarnessEnv(task_id="agent_create_1", backend="multi") as env:
        env.reset()
        env.step("message send --channel slack --target alice --message first")
        env.step("message send --channel slack --target bob --message second")

        env.step("undo 1")

        messages = env._backend.get_state().get("messages", [])
        bodies = {m.get("message") for m in messages}
        assert "first" in bodies, "pre-snapshot message must be preserved"
        assert "second" not in bodies, "post-snapshot message must be reverted"


def test_undo_keeps_virtual_time_moving_forward():
    with CLASHarnessEnv(task_id="agent_create_1", backend="multi") as env:
        env.reset()
        env.step("agent add alice")
        env.step("wait 120")
        assert env._virtual_time == 120

        env.step("undo 1")
        # Virtual time must not rewind even though state was restored.
        assert env._virtual_time == 120


def test_multi_backend_uses_in_memory_snapshots_no_copytree():
    with CLASHarnessEnv(task_id="agent_create_1", backend="multi") as env:
        env.reset()
        env.step("agent add alice")
        env.step("agent add bob")
        # In-memory snapshots are populated...
        assert len(env._snapshots) >= 2
        # ...and no on-disk copytree snapshots were created for the multi backend.
        assert len(env._state_manager._snapshots) == 0


def test_snapshot_eviction_bounds_history():
    with CLASHarnessEnv(task_id="agent_create_1", backend="multi", max_steps=200) as env:
        env.reset()
        for i in range(60):
            env.step(f"agent add agent_{i}")
        # History stack is capped at 50 and snapshots are evicted in lockstep.
        assert len(env._history_stack) <= 50
        assert len(env._snapshots) <= 51


def test_mock_backend_undo_still_works_via_disk():
    """Backward-compat: MockBackend keeps the legacy on-disk snapshot path."""
    with CLASHarnessEnv(task_id="agent_create_1", backend="mock") as env:
        env.reset()
        snap = env.save_state()
        # Mock backend has no in-memory snapshot support -> disk fallback used.
        assert "disk_snapshot_id" in env._snapshots[snap]
        env.load_state(snap)


def test_git_skill_snapshot_roundtrip_restores_staged_and_commits():
    """Regression: GitSkill must declare ``_snapshot_attrs`` so undo restores
    its mutable state. Without it, ``to_state``/``from_state`` capture nothing
    and undo silently leaves staged files / commits mutated."""
    from clasharness.skills.impl.git_skill import GitSkill

    skill = GitSkill()
    skill.initialize("/unused-state-dir", {})
    baseline_commits = len(skill._commits)

    # Stage a file, then snapshot at this boundary.
    skill.execute("git add app.py")
    snap = skill.to_state()
    # The fix wires the generic snapshot to the mutable attrs.
    assert set(snap) == {"_commits", "_staged", "_modified"}
    assert "app.py" in skill.get_state()["git_staged"]

    # Mutate past the snapshot: stage another file and commit.
    skill.execute("git add other.py")
    skill.execute("git commit -m work")
    assert len(skill._commits) == baseline_commits + 1
    assert skill.get_state()["git_staged"] == []  # commit cleared staging

    # Restore snapshot (undo): app.py staged again; other.py + new commit gone.
    skill.from_state(snap)
    state = skill.get_state()
    assert "app.py" in state["git_staged"]
    assert "other.py" not in state["git_staged"]
    assert len(skill._commits) == baseline_commits
    assert "git_modified" in state  # get_state now exposes the modified set


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__]))
