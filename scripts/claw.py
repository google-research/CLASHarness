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

"""OpenClaw CLI command runner. Runs simulated CLI commands on a task environment."""

import sys
from pathlib import Path

# Add root directory to python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from clasharness.core.environment import make_env

def main():
    if len(sys.argv) < 3:
        print("Usage: claw <task_id> <command...>")
        print("Example: claw calendar_add_event_1 message send --channel slack --target @alice --message hello")
        sys.exit(1)
        
    task_id = sys.argv[1]
    command = " ".join(sys.argv[2:])
    
    # Initialize environment in native Python multi mode
    env = make_env(task_id, mode="multi")
    
    # Execute the command
    res, reward, done, info = env.step(command)
    
    # Print outputs
    if res.stdout:
        print(res.stdout)
    if res.stderr:
        print(res.stderr, file=sys.stderr)
        
    sys.exit(res.exit_code)

if __name__ == "__main__":
    main()
