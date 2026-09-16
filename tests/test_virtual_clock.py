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

from clasharness import make_env
from clasharness.core.environment import CLASHarnessEnv

def test_wait_and_delegate():
    # We will use mock mode and basic task data.
    env = make_env("calendar_add_event_1")
    
    # Inject some mock private data for testing
    obs = env.reset()
    env._task.data.private["background_events"] = [
        {"trigger_time": 10, "type": "file_create", "path": "/test.txt", "new_content": "done"}
    ]
    # Re-initialize the events
    env.reset()
    
    # 1. Test wait increments time and skips budget
    initial_step = env._step_count
    obs, reward, done, info = env.step("wait 10")
    
    assert env._step_count == initial_step
    assert env._virtual_time == 10
    
    # 2. Test delegate
    obs, reward, done, info = env.step("openclaw agents delegate --agent TestAgent --instruction 'do something'")
    assert "Task delegated to TestAgent successfully." in obs.command_output
    
    # 3. Test status
    obs, reward, done, info = env.step("openclaw agents status")
    assert "TestAgent" in obs.command_output
    
    env.close()

def test_step_path_plumbs_virtual_time_to_skill():
    """The step dispatch seam publishes the virtual clock onto stateful skills."""
    with CLASHarnessEnv(task_id="agent_create_1", backend="multi") as env:
        env.reset()
        lock_skill = env._backend._lock_skill

        # Advance the clock directly (isolates the step seam from the _tick seam).
        env._virtual_time = 42
        env.step("lock acquire --resource db")
        assert lock_skill._now == 42, lock_skill._now


def test_tick_path_plumbs_virtual_time_to_skill():
    """Background-event state_effects (fired in _tick) see the advanced clock.

    They dispatch through execute_cli from inside _apply_background_event,
    bypassing step's publish, so _tick must refresh the skill clock itself.
    """
    with CLASHarnessEnv(task_id="agent_create_1", backend="multi") as env:
        env.reset()
        lock_skill = env._backend._lock_skill
        env._background_events.append({
            "trigger_time": 10,
            "type": "subagent_completion",
            "agent_name": "peer",
            "state_effects": ["lock acquire --resource evt --holder peer"],
        })

        env.step("wait 10")  # fires the event via _tick(10)

        assert env._virtual_time == 10
        # The lock acquired inside the background event observed the advanced
        # clock, not a stale 0.
        assert lock_skill._now == 10, lock_skill._now
        assert any(r.get("resource") == "evt" for r in lock_skill._records)


def test_background_effect_stamped_at_trigger_time_not_overwait():
    """M8/CL2 Deliv 7: a subagent completion is stamped at its ``trigger_time``.

    The environment stamps a background event's effects with the event's own
    ``trigger_time`` (its true completion time), NOT the post-``_tick`` clock. So
    even when the solver over-waits far past the completion, the temporal grader
    still sees the subagent as having finished on time -- an over-waiting solver
    is not penalized, and the reference trajectory is on time by construction.
    """
    with CLASHarnessEnv(task_id="agent_create_1", backend="multi") as env:
        env.reset()
        env._background_events.append({
            "trigger_time": 10,
            "type": "subagent_completion",
            "agent_name": "Iris_SLA_Monitor",
            "mock_response": "SLA compliant",
            "reply_channel": "ops",
        })

        env.step("wait 60")  # OVER-wait far past the completion at t=10

        assert env._virtual_time == 60, env._virtual_time
        msgs = env._effects["messages_sent"]
        mine = [m for m in msgs if m.get("sender") == "Iris_SLA_Monitor"]
        assert mine, msgs
        # Stamped at the completion time (10), not the over-waited clock (60).
        assert mine[0]["virtual_time"] == 10, mine[0]


def test_solver_effect_stamped_at_current_clock():
    """M8/CL2 Deliv 7: the solver's own effects carry the current virtual time.

    ``_track_effects`` stamps with ``at=None`` -> ``self._virtual_time``, i.e.
    *when the solver acted*, distinguishing a solver action from a delayed
    background completion (which is stamped at its ``trigger_time``).
    """
    with CLASHarnessEnv(task_id="agent_create_1", backend="multi") as env:
        env.reset()
        env._virtual_time = 25
        env.step("agent add peer2")
        created = env._effects["agents_created"]
        assert created, env._effects["agents_created"]
        # Stamped with the clock at the moment the solver acted (25), not 0.
        assert all(e.get("virtual_time") == 25 for e in created), created


if __name__ == "__main__":
    test_wait_and_delegate()
    test_step_path_plumbs_virtual_time_to_skill()
    test_tick_path_plumbs_virtual_time_to_skill()
    test_background_effect_stamped_at_trigger_time_not_overwait()
    test_solver_effect_stamped_at_current_clock()
    print("All tests passed!")
