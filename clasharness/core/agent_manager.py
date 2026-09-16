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

"""Agent manager for background subagents."""

import threading
from typing import Any


class AgentManager:
  """Manages background execution of sub-agents in real and hybrid modes."""

  def __init__(self, mode: str, env: Any):
    self.mode = mode
    self.env = env
    self.active_tasks = {}

  def delegate(self, agent_name: str, instruction: str) -> None:
    """Delegates a task to a sub-agent."""
    if self.mode == "mock":
      self.active_tasks[agent_name] = {
          "status": "pending",
          "instruction": instruction,
      }

      # Dynamically lookup mock behavior and schedule completion event
      # pylint: disable=protected-access
      if (
          self.env
          and hasattr(self.env, "_task")
          and self.env._task
          and hasattr(self.env._task, "data")
      ):
        mock_subagents = self.env._task.data.private.get("mock_subagents", [])
        def _normalize(s: str | None) -> str:
          if not s:
            return ""
          return "".join(c for c in s.lower() if c.isalnum())

        norm_agent_name = _normalize(agent_name)
        behavior = next(
            (
                b
                for b in mock_subagents
                if norm_agent_name in _normalize(b.get("agent_role"))
                or norm_agent_name in _normalize(b.get("agent_name"))
            ),
            None,
        )
        if behavior:
          trigger_time = self.env._virtual_time + behavior.get(
              "virtual_duration_seconds", 0
          )
          completion_event = {
              "type": "subagent_completion",
              "trigger_time": trigger_time,
              "agent_name": agent_name,
              "mock_response": behavior.get(
                  "mock_response", "Task completed."
              ),
              "state_effects": behavior.get("state_effects", []),
              # G1/R1: canonical channel the subagent replies on, so the
              # runtime can route its spoken reply into graded messages_sent.
              "reply_channel": behavior.get("reply_channel", ""),
          }
          if not hasattr(self.env, "_background_events"):
            self.env._background_events = []
          self.env._background_events.append(completion_event)
          # Keep sorted by trigger_time
          self.env._background_events.sort(
              key=lambda x: x.get("trigger_time", 0)
          )
      # pylint: enable=protected-access
    else:
      # In real/hybrid mode, we spawn a background thread.
      cancel_event = threading.Event()
      self.active_tasks[agent_name] = {
          "status": "running",
          "instruction": instruction,
          "cancel_event": cancel_event,
      }
      thread = threading.Thread(
          target=self._run_sub_agent_thread,
          args=(agent_name, instruction, cancel_event),
          daemon=True,
      )
      thread.start()

  def status(self, agent_name: str | None = None) -> dict[str, str]:
    """Returns the status of pending sub-agent tasks."""
    if agent_name:
      task = self.active_tasks.get(agent_name)
      return {agent_name: task["status"] if task else "unknown"}
    return {name: task["status"] for name, task in self.active_tasks.items()}

  def _run_sub_agent_thread(
      self,
      agent_name: str,
      instruction: str,
      cancel_event: threading.Event,
  ) -> None:
    """Runs the sub-agent in a background thread."""
    del instruction  # Unused.
    try:
      # Check if there is a virtual background event for this agent's completion
      has_virtual_event = False
      if hasattr(self.env, "_background_events") and self.env._background_events:
        for event in self.env._background_events:
          if event.get("type") == "subagent_completion" and event.get("agent_name") == agent_name:
            has_virtual_event = True
            break

      if cancel_event.wait(2.0):
        # Subagent was cancelled
        self.active_tasks[agent_name]["status"] = "cancelled"
        return

      if has_virtual_event:
        # Do not mark completed here; let virtual time background event handle it.
        return

      # pylint: disable=protected-access
      if hasattr(self.env, "_backend") and self.env._backend:
        self.env._backend.execute_cli(
            f"file create --path /reports/{agent_name}_done.txt "
            "--content 'done'"
        )
      # pylint: enable=protected-access
      self.active_tasks[agent_name]["status"] = "completed"
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.active_tasks[agent_name]["status"] = f"failed: {e}"

  def cancel_all_tasks(self) -> None:
    """Cancel all running background sub-agent threads."""
    for task in self.active_tasks.values():
      if "cancel_event" in task:
        task["cancel_event"].set()
