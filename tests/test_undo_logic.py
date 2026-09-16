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

import os
import time

# Ensure we're in the right directory
from clasharness.core.environment import make_env

def test_undo():
    env = make_env("calendar_add_event_1")
    obs = env.reset()
    
    # Inject a mock background event with trigger_time 15
    env._background_events.append({
        "type": "file_create",
        "path": "/background.txt",
        "new_content": "hello",
        "trigger_time": 15
    })
    
    # 1. config set
    obs, reward, done, info = env.step("openclaw config set test.key 1")
    assert env._step_count == 1
    assert len(env._history_stack) == 1
    
    # Verify config changed
    assert env._backend.get_config().get("test", {}).get("key") == "1"
    
    # 2. wait 10
    obs, reward, done, info = env.step("wait 10")
    assert env._step_count == 1  # wait doesn't cost step
    assert len(env._history_stack) == 2
    assert env._virtual_time == 10
    assert len(env._background_events) == 1
    assert env._background_events[0]["trigger_time"] == 15
    
    # 3. config set again
    obs, reward, done, info = env.step("openclaw config set test.key 2")
    assert env._step_count == 2
    assert len(env._history_stack) == 3
    assert env._backend.get_config().get("test", {}).get("key") == "2"
    
    # 4. undo 2
    obs, reward, done, info = env.step("undo 2")
    assert env._step_count == 3  # undo costs 1 step
    assert len(env._history_stack) == 1  # popped 2
    assert env._virtual_time == 10  # time shouldn't revert
    
    # config should revert to 1 (state from before wait 10 and second config set)
    assert env._backend.get_config().get("test", {}).get("key") == "1"
    
    # background events should have shifted!
    assert len(env._background_events) == 1
    assert env._background_events[0]["trigger_time"] == 25
    
    # Now undo 1 more
    obs, reward, done, info = env.step("undo")
    assert env._step_count == 4
    assert len(env._history_stack) == 0
    # config should revert to empty
    assert env._backend.get_config().get("test", {}).get("key") is None
    
    # Check that another undo fails gracefully
    obs, reward, done, info = env.step("undo")
    assert env._step_count == 5
    assert "Error: No history available to undo." in obs.command_output
    
    print("All tests passed!")

if __name__ == "__main__":
    test_undo()
