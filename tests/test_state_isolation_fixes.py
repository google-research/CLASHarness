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

"""Tests for state isolation, AST sandbox, and checkpointing fixes."""

# pylint: disable=protected-access

import json
import os
import pathlib
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from clasharness import make_env
from clasharness.scenario import conflict_llm_pipeline
from clasharness.scenario import conflict_vlm_materializer
from clasharness.scenario import conflict_vlm_spec
from clasharness.scenario import llm_client
from clasharness.scenario import vlm_asset_generators
from clasharness.tools import agent_tool_spec

class pytest:
  @staticmethod
  def raises(expected_exception, match=None):
    import re
    class RaisesContext:
      def __enter__(self):
        return self
      def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
          raise AssertionError(f"{expected_exception.__name__} not raised")
        if not issubclass(exc_type, expected_exception):
          return False
        if match and not re.search(match, str(exc_val)):
          raise AssertionError(f"Pattern '{match}' not found in '{exc_val}'")
        return True
    return RaisesContext()

from tests import test_conflict_pipeline


# =====================================================================
# 1. AST Sandbox Hardening Tests (UT-AST-1, UT-AST-2, UT-AST-3)
# =====================================================================


def test_ast_blocked_imports():
  """Verify that importing forbidden modules raises ValueError (UT-AST-1)."""
  blocked_code_1 = "import os\nprint('hello')"
  blocked_code_2 = "import subprocess"
  blocked_code_3 = "from shutil import copyfile"

  with pytest.raises(ValueError, match="import is not allowed"):
    vlm_asset_generators._validate_render_code(blocked_code_1)

  with pytest.raises(ValueError, match="import is not allowed"):
    vlm_asset_generators._validate_render_code(blocked_code_2)

  with pytest.raises(ValueError, match="import is not allowed"):
    vlm_asset_generators._validate_render_code(blocked_code_3)


def test_ast_blocked_calls_and_dunders():
  """Verify that calling dynamic lookup or blocked builtins (UT-AST-2)."""
  # Blocked builtins
  with pytest.raises(ValueError, match="call is not allowed: getattr"):
    vlm_asset_generators._validate_render_code(
        "fn = getattr(pathlib.Path('/tmp'), 'write_text')"
    )

  with pytest.raises(ValueError, match="call is not allowed: eval"):
    vlm_asset_generators._validate_render_code("eval('print(1)')")

  with pytest.raises(ValueError, match="call is not allowed: __import__"):
    vlm_asset_generators._validate_render_code(
        "__import__('os').system('ls')"
    )

  # Dunder accesses
  with pytest.raises(
      ValueError, match="double-underscore attributes is blocked"
  ):
    vlm_asset_generators._validate_render_code("a = [].__class__")

  # Blocklisted Path attribute functions
  with pytest.raises(ValueError, match="call is not allowed: .write_text"):
    vlm_asset_generators._validate_render_code("p.write_text('bad')")


def test_ast_safe_execution():
  """Verify standard rendering code completes successfully (UT-AST-3)."""
  safe_code = """
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])
"""
  # Should not raise any exceptions
  vlm_asset_generators._validate_render_code(safe_code)


# =====================================================================
# 2. Integration Tests (IT-LIFECYCLE-1, IT-VFS-1, IT-VFS-2, IT-SNAP-2)
# =====================================================================


def test_undo_thread_cancellation():
  """Verify subagent threads are aborted on undo/load_state (IT-LIFECYCLE-1)."""
  env = make_env("setup_workspace_1", mode="multi")
  env.reset()
  try:
    # Save state before delegation
    snapshot_id = env.save_state()

    # Trigger background delegation (requires 2 seconds simulate latency)
    env.step(
        "agent delegate --agent assistant --instruction 'create done file'"
    )

    # Revert state immediately
    env.load_state(snapshot_id)

    # Reverted workspace should not have the file immediately
    workspace_dir = pathlib.Path(env._state_manager.state_dir) / "workspace"
    done_file = workspace_dir / "reports" / "assistant_done.txt"
    assert not done_file.exists()

    # Wait briefly for thread to context-switch and execute cancellation exit
    time.sleep(0.2)

    # Verify that status is cancelled
    assert env._agent_manager.status("assistant")["assistant"] == "cancelled"
    assert not done_file.exists()
  finally:
    env.close()


def test_vfs_physical_sync():
  """Verify virtual filesystem mutations sync to disk (IT-VFS-1)."""
  env = make_env("setup_workspace_1", mode="multi")
  env.reset()
  try:
    # Create file virtually
    env.step("file create --path /sync_test.txt --content 'synced content'")

    # Verify physically on disk
    workspace_dir = pathlib.Path(env._state_manager.state_dir) / "workspace"
    physical_file = workspace_dir / "sync_test.txt"
    assert physical_file.exists()
    assert physical_file.read_text(encoding="utf-8") == "synced content"

    # Append to file
    env.step("file append --path /sync_test.txt --content ' appended'")
    assert (
        physical_file.read_text(encoding="utf-8") == "synced content appended"
    )

    # Delete file
    env.step("file delete --path /sync_test.txt")
    assert not physical_file.exists()
  finally:
    env.close()


def test_skills_disk_serialization():
  """Verify skills serialize back to disk JSON (IT-VFS-2)."""
  env = make_env("setup_workspace_1", mode="multi")
  env.reset()
  try:
    state_dir = pathlib.Path(env._state_manager.state_dir)
    emails_json = state_dir / "emails.json"
    calendar_json = state_dir / "calendar_events.json"
    tasks_json = state_dir / "tasks.json"

    # 1. Modify emails and verify disk
    env.step(
        "email send --to test@example.com --subject 'snapshot test' "
        "--body 'test'"
    )
    assert emails_json.exists()
    emails_payload = json.loads(emails_json.read_text(encoding="utf-8"))
    assert emails_payload.get("replace_seeded")
    emails = emails_payload.get("emails", [])
    assert len(emails) > 0
    assert emails[-1]["to"] == "test@example.com"

    # 2. Modify calendar events and verify disk
    env.step(
        "calendar add-event --title 'Test meeting' --start '2026-06-12 10:00' "
        "--end '2026-06-12 11:00'"
    )
    assert calendar_json.exists()
    calendar_payload = json.loads(calendar_json.read_text(encoding="utf-8"))
    events = calendar_payload.get("events", [])
    assert len(events) > 0
    assert events[-1]["title"] == "Test meeting"

    # 3. Modify tasks and verify disk
    env.step("tasks add --title 'Refactor pipeline' --priority 'high'")
    assert tasks_json.exists()
    tasks_payload = json.loads(tasks_json.read_text(encoding="utf-8"))
    tasks = tasks_payload.get("tasks", [])
    assert len(tasks) > 0
    assert tasks[-1]["title"] == "Refactor pipeline"
  finally:
    env.close()


def test_direct_snapshot_restore():
  """Verify load_state restores clocks and effects to snapshot (IT-SNAP-2)."""
  env = make_env("setup_workspace_1", mode="multi")
  env.reset()
  try:
    # Verify initial clock is 0
    assert env._virtual_time == 0
    assert len(env._effects.get("emails_sent", [])) == 0

    # 1. Save snapshot
    snapshot_id = env.save_state()

    # 2. Mutate virtual time, effects, and create files
    env.step("wait 60")
    env.step(
        "email send --to test@example.com --subject 'snapshot test' "
        "--body 'test'"
    )
    env.step("file create --path /mutate_snap.txt --content 'modified'")

    assert env._virtual_time == 60
    assert len(env._effects.get("emails_sent", [])) == 1

    # Verify physical file exists
    workspace_dir = pathlib.Path(env._state_manager.state_dir) / "workspace"
    physical_file = workspace_dir / "mutate_snap.txt"
    assert physical_file.exists()

    # 3. Revert to snapshot
    env.load_state(snapshot_id)

    # 4. Check if virtual_time, effects, and files reverted cleanly!
    assert env._virtual_time == 0
    assert len(env._effects.get("emails_sent", [])) == 0
    assert not physical_file.exists(), (
        "Physical file created after snapshot was not deleted on restore"
    )
  finally:
    env.close()


# =====================================================================
# 3. Scenario Generation & Normalization Tests (UT-1, UT-2, UT-3, IT-1, IT-2)
# =====================================================================


def test_family_name_normalization():
  """Verify family name normalization matches spec categories (UT-1, UT-2)."""
  # Singular / Plural conversions
  assert agent_tool_spec.normalize_family_name("agents") == "agent"
  assert agent_tool_spec.normalize_family_name("agent") == "agent"
  assert agent_tool_spec.normalize_family_name("tasks") == "tasks"
  assert agent_tool_spec.normalize_family_name("task") == "tasks"
  assert agent_tool_spec.normalize_family_name("files") == "file"
  assert agent_tool_spec.normalize_family_name("file") == "file"

  # Casing and spaces
  assert agent_tool_spec.normalize_family_name("  emails ") == "email"
  assert agent_tool_spec.normalize_family_name("System") == "system"

  # Fallback for custom families
  assert (
      agent_tool_spec.normalize_family_name("custom_family")
      == "custom_family"
  )

  # Dynamic lookup fallback for custom families (UT-2)
  mock_spec = agent_tool_spec.AgentToolSpec(
      version="agent_tools.v1",
      families=(
          agent_tool_spec.AgentToolFamily(
              family="custom_tool", domains=(), prefixes=(), commands=()
          ),
          agent_tool_spec.AgentToolFamily(
              family="special_indices", domains=(), prefixes=(), commands=()
          ),
      ),
  )
  with mock.patch(
      "clasharness.tools.agent_tool_spec.load_agent_tool_spec",
      return_value=mock_spec,
  ):
    # custom_tools -> custom_tool
    assert (
        agent_tool_spec.normalize_family_name("custom_tools") == "custom_tool"
    )
    # custom_tool -> custom_tool
    assert (
        agent_tool_spec.normalize_family_name("custom_tool") == "custom_tool"
    )
    # special_indices -> special_indices
    assert (
        agent_tool_spec.normalize_family_name("special_indices")
        == "special_indices"
    )
    # special_index -> special_indices
    assert (
        agent_tool_spec.normalize_family_name("special_index")
        == "special_indices"
    )
    # completely unknown stays as is
    assert agent_tool_spec.normalize_family_name("unknowns") == "unknowns"


def test_validation_under_normalized_families():
  """Verify validation of commands under normalized family lists (UT-3)."""
  # Case 1: "agent delegate" is allowed if allowed_families
  # contains "agents" or "agent".
  cmd = "agent delegate --agent assistant --instruction 'do task'"

  # Valid cases
  agent_tool_spec.validate_command_against_spec(
      cmd, allowed_families=["agents"]
  )
  agent_tool_spec.validate_command_against_spec(
      cmd, allowed_families=["agent"]
  )
  agent_tool_spec.validate_command_against_spec(
      cmd, allowed_families=["AGENTS", "file"]
  )

  # Case 2: Rejected if family not in allowed list
  with pytest.raises(
      agent_tool_spec.ToolSpecViolation,
      match="command is not defined in AgentToolSpec",
  ):
    agent_tool_spec.validate_command_against_spec(
        cmd, allowed_families=["file", "email"]
    )


def test_prompt_syntax_filtering():
  """Verify prompt builder filters command syntaxes to allowed families (IT-1)."""

  # Mock LLM client just to instantiate the pipeline
  class DummyClient(llm_client.ScenarioLLMClient):

    def generate_layer(self, name, payload):
      del name, payload  # Unused
      return {}

  pipeline = conflict_llm_pipeline.ConflictLLMPipeline(client=DummyClient())

  # Filtered prompt for ["asset", "agents"]
  prompt_output = pipeline._build_allowed_commands_prompt(["asset", "agents"])

  # Should contain asset commands
  assert "asset list [--kind KIND]" in prompt_output
  assert "asset open --id ID" in prompt_output
  # Should contain agent commands (due to normalization)
  assert "agent delegate --agent NAME --instruction TEXT" in prompt_output
  # Should NOT contain email or calendar commands
  assert "email send" not in prompt_output
  assert "calendar add-event" not in prompt_output


def test_scenario_compilation_with_aliases():
  """Verify materialize compiles with casing & alias allowed families (IT-2)."""
  # Allowed tool families has mixed cases and plural forms
  seed = conflict_vlm_spec.ConflictVLMSeed(
      slug="test_normalize_compilation",
      domain="system",
      title="Test Norm",
      target_rounds=1,
      core_conflict="Mock conflict",
      conflict_taxonomy=["temporal_staleness"],
      participants=["owner", "assistant"],
      required_constraints=[],
      allowed_tool_families=["agents", "Files", "MESSAGES"],
  )

  with tempfile.TemporaryDirectory() as tmp:
    tmp_path = pathlib.Path(tmp)
    client = test_conflict_pipeline.MockConflictLLMClient()
    # Should not raise ValueError because "agents" and "Files" and "MESSAGES"
    # normalize to "agent", "file", "message", matching mock commands
    conflict_spec, _ = conflict_vlm_materializer.materialize_conflict_scenario(
        seed,
        client,
        output_dir=tmp_path,
        image_provider="mock",
        chart_provider="mock",
    )
    assert conflict_spec.scenario_slug == "test_normalize_compilation"


def test_session_updates_reconstruction_in_compilation():
  """Verify that previous rounds' session appends and redactions are backfilled in compilation."""
  from clasharness.scenario.spec import ScenarioSession, SessionMessage, ScenarioWorkspace
  from clasharness.scenario import conflict_vlm_spec
  from clasharness.scenario.conflict_vlm_spec import (
      ConflictVLMScenarioSpec,
      ConflictVLMRoundSpec,
      ConflictScenarioUpdate,
  )

  # 1. Setup seed
  seed = conflict_vlm_spec.ConflictVLMSeed(
      slug="test_session_recon",
      domain="system",
      title="Test Session Recon",
      target_rounds=3,
      core_conflict="Mock conflict",
      conflict_taxonomy=["temporal_staleness"],
      participants=["owner", "assistant"],
      required_constraints=[],
      allowed_tool_families=["agents", "file", "message"],
  )

  # 2. Setup baseline sessions
  baseline_msg = SessionMessage(
      message_id="msg_001",
      timestamp="2026-06-01T10:00:00Z",
      sender="owner",
      text="Hello, please start.",
  )
  session = ScenarioSession(
      session_id="channel_general",
      kind="chat",
      channel="#general",
      path="sessions/channel_general.jsonl",
      messages=[baseline_msg],
  )

  # 3. Setup updates
  # Update 1 (r01): appends message 2
  update_1 = ConflictScenarioUpdate(
      update_id="upd_r01",
      workspace_adds=[],
      session_appends=[{
          "session_id": "channel_general",
          "messages": [{
              "message_id": "msg_002",
              "timestamp": "2026-06-01T10:05:00Z",
              "sender": "assistant",
              "text": "Started.",
          }],
      }],
      session_redactions=[],
  )
  # Update 2 (r02): redacts message 1
  update_2 = ConflictScenarioUpdate(
      update_id="upd_r02",
      workspace_adds=[],
      session_appends=[],
      session_redactions=["msg_001"],
  )

  # 4. Setup rounds
  round_1 = ConflictVLMRoundSpec(
      round_id="r01",
      round_type="action",
      visible_asset_ids=[],
      required_modalities=[],
      expected_anchor_ids=["EVIDENCE:TEST_SESSION_RECON_VISUAL_BLOCKER"],
      required_tool_families=["message"],
      update_ids=["upd_r01"],
  )
  round_2 = ConflictVLMRoundSpec(
      round_id="r02",
      round_type="action",
      visible_asset_ids=[],
      required_modalities=[],
      expected_anchor_ids=["EVIDENCE:TEST_SESSION_RECON_TEXT_OWNER_NOTE"],
      required_tool_families=["message"],
      update_ids=["upd_r02"],
  )
  # Round 3: Compilation target, which should inherit round 1 and round 2 updates
  round_3 = ConflictVLMRoundSpec(
      round_id="r03",
      round_type="action",
      visible_asset_ids=[],
      required_modalities=[],
      expected_anchor_ids=["EVIDENCE:TEST_SESSION_RECON_CHART_CONFLICT"],
      required_tool_families=["message"],
      update_ids=[],
  )

  spec = ConflictVLMScenarioSpec(
      series="conflict_vlm_v1",
      scenario_slug="test_session_recon",
      title="Test Session Recon",
      domain="system",
      core_conflict="Mock conflict",
      conflict_taxonomy=["temporal_staleness"],
      modalities=[],
      input_policy=conflict_vlm_spec.VLMInputPolicy(),
      generation_policy=conflict_vlm_spec.VLMGenerationPolicy(),
      allowed_tool_families=["agents", "file", "message"],
      assets=[],
      updates=[update_1, update_2],
      rounds=[round_1, round_2, round_3],
      evals=[],
      workspace=ScenarioWorkspace(),
      sessions=[session],
  )

  # 5. Compile round 3 task
  task = conflict_vlm_materializer._compile_round_to_task(
      seed=seed,
      round_spec=round_3,
      evals=[],
      round_updates=[],
      round_mocks=[],
      spec=spec,
  )

  # 6. Verify task overrides
  overrides = task.data.private["initial_state_overrides"]
  assert "sessions/channel_general.jsonl" in overrides
  log_content = overrides["sessions/channel_general.jsonl"]
  lines = log_content.strip().split("\n")
  assert len(lines) == 2

  msg1 = json.loads(lines[0])
  msg2 = json.loads(lines[1])

  # Check redaction of msg_001
  assert msg1["message_id"] == "msg_001"
  assert msg1["is_redacted"] is True
  assert msg1["text"] == "[Message redacted by compliance]"

  # Check append of msg_002
  assert msg2["message_id"] == "msg_002"
  assert msg2["text"] == "Started."
  assert msg2["is_redacted"] is False


class TestStateIsolationFixes(unittest.TestCase):
  pass


# Dynamically bind all plain functions starting with test_ to the TestCase class
for name, value in list(globals().items()):
  if name.startswith("test_") and callable(value):
    def make_method(func=value):
      return lambda self: func()
    setattr(TestStateIsolationFixes, name, make_method())


if __name__ == "__main__":
  unittest.main()
