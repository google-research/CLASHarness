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

"""Tests for OpenClaw runtime compatibility rewrite helpers."""

from __future__ import annotations

from clasharness.backend.openclaw_compat import classify_error, rewrite_command


def test_rewrite_message_poll_question_flag():
    cmd = "openclaw message poll --target #general --question 'Vote?' --options yes,no"
    d = rewrite_command(cmd)
    assert d.compat_status == "rewritten"
    assert "--poll-question" in d.executed_action
    assert "--question" not in d.executed_action


def test_rewrite_cron_add_injects_name():
    cmd = "openclaw cron add --schedule '0 9 * * *' --command 'openclaw status'"
    d = rewrite_command(cmd)
    assert d.compat_status == "rewritten"
    assert "--name" in d.executed_action


def test_rewrite_cron_nested_command_quotes():
    cmd = (
        "openclaw cron add --schedule '0 8 * * 1' "
        "--command 'openclaw message send --channel whatsapp --target @alice "
        "--message 'Daily report''"
    )
    d = rewrite_command(cmd)
    assert d.compat_status == "rewritten"
    assert "rewritten_cron_command_quoting" in d.error_tags
    # Ensure there is no trailing token outside --command due broken quoting.
    assert not d.executed_action.endswith(" report")
    assert "--name" in d.executed_action


def test_skip_incomplete_cron_add():
    d = rewrite_command("openclaw cron add", skip_incompatible=True)
    assert d.compat_status == "skipped_incompatible"
    assert "incomplete_cron_add" in d.error_tags


def test_skip_incompatible_message_react_target_only():
    cmd = "openclaw message react --channel slack --target @bob --emoji '✅'"
    d = rewrite_command(cmd, skip_incompatible=True)
    assert d.compat_status == "skipped_incompatible"
    assert "incompatible_message_react_target_only" in d.error_tags


def test_rewrite_agents_set_identity_name_to_agent():
    cmd = "openclaw agents set-identity --name helper --emoji '🔍'"
    d = rewrite_command(cmd)
    assert d.compat_status == "rewritten"
    assert "--agent" in d.executed_action
    assert "--name" not in d.executed_action
    assert "rewritten_agents_set_identity_name" in d.error_tags


def test_rewrite_configure_key_value_to_config_set():
    cmd = "openclaw configure gateway.port=18789"
    d = rewrite_command(cmd)
    assert d.compat_status == "rewritten"
    assert d.executed_action == "system config set gateway.port 18789"
    assert "rewritten_configure_key_value" in d.error_tags


def test_rewrite_run_script_to_python3():
    d = rewrite_command("run_script --path scripts/fix.py --args '--dry-run 5'")
    assert d.compat_status == "rewritten"
    assert "rewritten_mcp_tool_verb" in d.error_tags
    assert d.executed_action == "python3 scripts/fix.py --dry-run 5"


def test_rewrite_run_script_shell_uses_bash():
    d = rewrite_command("run_script --path deploy/rollback.sh")
    assert d.compat_status == "rewritten"
    assert d.executed_action == "bash deploy/rollback.sh"


def test_rewrite_run_script_positional_path():
    d = rewrite_command("run_script scripts/audit.py")
    assert d.compat_status == "rewritten"
    assert d.executed_action == "python3 scripts/audit.py"


def test_rewrite_decompress_archive_zip():
    d = rewrite_command("decompress_archive --path bundle.zip --output-dir out")
    assert d.compat_status == "rewritten"
    assert d.executed_action == "unzip -d out bundle.zip"


def test_rewrite_decompress_archive_tar_gz():
    d = rewrite_command("decompress_archive --path logs.tar.gz")
    assert d.compat_status == "rewritten"
    assert d.executed_action == "tar -xzf logs.tar.gz"


def test_rewrite_execute_db_query_to_sqlite3():
    d = rewrite_command('execute_db_query --db app.db --query "SELECT 1"')
    assert d.compat_status == "rewritten"
    assert d.executed_action == "sqlite3 app.db 'SELECT 1'"


def test_non_mcp_command_untouched():
    # An already-canonical CLI command (not an MCP verb) must pass through as-is.
    d = rewrite_command("python3 scripts/audit.py --check")
    assert d.compat_status == "ok"
    assert d.executed_action == "python3 scripts/audit.py --check"
    assert "rewritten_mcp_tool_verb" not in d.error_tags


def test_classify_error_tags():
    text = "Invalid config at /tmp/x/openclaw.json\\nerror: unknown option '--name'"
    tags = classify_error(text)
    assert "invalid_config" in tags
    assert "unknown_option" in tags


def test_classify_error_channel_and_plugin_tags():
    text = (
        "Error: Unknown channel: discord\n"
        "Channel login failed: Error: Unsupported channel: discord\n"
        "Error: Channel is required (no configured channels detected).\n"
        "Error: Broadcast requires at least one configured channel.\n"
        "Plugin not found: weather-plugin\n"
    )
    tags = classify_error(text)
    assert "unknown_channel" in tags
    assert "unsupported_channel" in tags
    assert "channel_required" in tags
    assert "channel_not_configured" in tags
    assert "plugin_not_found" in tags
