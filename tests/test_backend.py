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

"""Unit tests for the MockBackend."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from clasharness.backend.mock_backend import MockBackend
from clasharness.skills.impl.system_skill import SystemSkill


@pytest.fixture
def backend(tmp_path):
    b = MockBackend()
    config = {
        "agent": {"model": "anthropic/claude-sonnet-4-5-20250929"},
        "gateway": {"port": 18789, "auth": {"mode": "token", "token": "test"}},
        "channels": {},
    }
    (tmp_path / "openclaw.json").write_text(json.dumps(config))
    b.initialize(str(tmp_path), {})
    return b


# ---- Status / Health ----

def test_status_text(backend):
    result = backend.execute_cli("openclaw status")
    assert result.exit_code == 0
    assert "Gateway" in result.stdout


def test_status_json(backend):
    result = backend.execute_cli("openclaw status --json")
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert "gateway" in data


def test_health_no_gateway(backend):
    result = backend.execute_cli("openclaw health")
    assert result.exit_code == 1
    assert "not running" in result.stderr


def test_health_after_start(backend):
    backend.execute_cli("openclaw gateway start")
    result = backend.execute_cli("openclaw health")
    assert result.exit_code == 0
    assert "OK" in result.stdout


# ---- Gateway ----

def test_gateway_start_stop(backend):
    r = backend.execute_cli("openclaw gateway start")
    assert r.exit_code == 0
    assert backend._gateway_running is True

    r = backend.execute_cli("openclaw gateway stop")
    assert r.exit_code == 0
    assert backend._gateway_running is False


def test_gateway_start_idempotent(backend):
    backend.execute_cli("openclaw gateway start")
    r = backend.execute_cli("openclaw gateway start")
    assert r.exit_code == 0
    assert "already running" in r.stdout


# ---- Agents ----

def test_agent_create(backend):
    result = backend.execute_cli("openclaw agents add --name alice --model openai/gpt-4o")
    assert result.exit_code == 0
    assert "alice" in result.stdout
    assert "alice" in backend._agents


def test_agent_create_duplicate(backend):
    backend.execute_cli("openclaw agents add --name bob --model openai/gpt-4o")
    r = backend.execute_cli("openclaw agents add --name bob --model openai/gpt-4o")
    assert r.exit_code == 1
    assert "already exists" in r.stderr


def test_agent_list_empty(backend):
    r = backend.execute_cli("openclaw agents list")
    assert r.exit_code == 0
    assert "No agents" in r.stdout


def test_agent_list(backend):
    backend.execute_cli("openclaw agents add --name researcher --model openai/gpt-4o")
    r = backend.execute_cli("openclaw agents list")
    assert "researcher" in r.stdout


def test_agent_delete(backend):
    backend.execute_cli("openclaw agents add --name temp --model openai/gpt-4o")
    r = backend.execute_cli("openclaw agents delete --name temp")
    assert r.exit_code == 0
    assert "temp" not in backend._agents


def test_agent_set_identity(backend):
    backend.execute_cli("openclaw agents add --name helper --model openai/gpt-4o")
    r = backend.execute_cli("openclaw agents set-identity --name helper --emoji 🤖")
    assert r.exit_code == 0
    assert backend._agents["helper"]["emoji"] == "🤖"


def test_agent_online_weather_pattern(backend):
    r = backend.execute_cli(
        "openclaw agent --local --agent main "
        "--message \"Use weather for location 'London'. "
        "Reply with WEATHER_RESULT:.\""
    )
    assert r.exit_code == 0
    assert "WEATHER_RESULT:" in r.stdout
    assert "London" in r.stdout


def test_agent_online_calendar_pattern(backend):
    r = backend.execute_cli(
        "openclaw agent --local --agent main "
        "--message \"Summarize events for day 'today'. "
        "Reply with CALENDAR_RESULT:.\""
    )
    assert r.exit_code == 0
    assert "CALENDAR_RESULT:" in r.stdout
    assert "today" in r.stdout.lower()


# ---- Messaging ----

def test_message_send(backend):
    r = backend.execute_cli(
        "openclaw message send --channel telegram --target @alice --message 'Hello world'"
    )
    assert r.exit_code == 0
    assert len(backend._messages) == 1
    assert backend._messages[0]["target"] == "@alice"


def test_message_send_with_channel(backend):
    r = backend.execute_cli(
        "openclaw message send --channel telegram --target @bob --message 'Hi'"
    )
    assert r.exit_code == 0
    assert backend._messages[0]["channel"] == "telegram"


def test_message_send_missing_target(backend):
    r = backend.execute_cli("openclaw message send --message 'oops'")
    assert r.exit_code == 1


def test_message_send_missing_channel(backend):
    r = backend.execute_cli("openclaw message send --target @alice --message 'oops'")
    assert r.exit_code == 1


def test_message_broadcast(backend):
    r = backend.execute_cli(
        "openclaw message broadcast --targets @a,@b,@c --message 'Broadcast'"
    )
    assert r.exit_code == 0
    assert len(backend._messages) == 3


def test_message_broadcast_channel_routed(backend):
    """``--channel NAME --message TEXT`` is the documented syntax.

    It is also the ONLY form the agent-facing spec exposes: ``message.broadcast``
    in ``agent_tools.v1.json`` declares just ``channel``/``message``, so
    ``--targets`` is unreachable from the agent path. This form used to fall
    through the ``--targets`` loop with an empty list and return exit 0 having
    recorded nothing, making every agent broadcast a silent no-op.
    """
    r = backend.execute_cli(
        "openclaw message broadcast --channel '#ops' --message 'all clear'"
    )
    assert r.exit_code == 0
    assert len(backend._messages) == 1
    rec = backend._messages[0]
    assert rec["target"] == "#ops"
    assert rec["channel"] == "#ops"
    assert rec["message"] == "all clear"


def test_message_broadcast_channel_and_targets(backend):
    r = backend.execute_cli(
        "openclaw message broadcast --channel ops --targets @a,@b --message 'hi'"
    )
    assert r.exit_code == 0
    assert len(backend._messages) == 2
    assert {m["target"] for m in backend._messages} == {"@a", "@b"}
    assert all(m["channel"] == "ops" for m in backend._messages)


def test_message_broadcast_without_channel_or_targets_fails(backend):
    r = backend.execute_cli("openclaw message broadcast --message 'orphan'")
    assert r.exit_code == 1
    assert backend._messages == []


def test_message_broadcast_without_message_fails(backend):
    r = backend.execute_cli("openclaw message broadcast --channel ops")
    assert r.exit_code == 1
    assert backend._messages == []


def test_message_search_finds_channel_broadcast(backend):
    """A broadcast now carries its channel, so ``search --channel`` matches it."""
    backend.execute_cli(
        "openclaw message broadcast --channel ops --message 'gateway restored'"
    )
    r = backend.execute_cli("openclaw message search --channel ops --query gateway")
    assert r.exit_code == 0
    assert "gateway restored" in r.stdout


def test_message_poll(backend):
    r = backend.execute_cli(
        "openclaw message poll --target #general --question 'Vote?' --options 'yes,no'"
    )
    assert r.exit_code == 0
    assert backend._messages[0]["type"] == "poll"


def test_message_poll_channel_routed(backend):
    """``--channel`` is the documented routing flag and must reach the record.

    It used to be dropped: a poll written exactly as the spec documents it
    (``message poll --channel NAME --question TEXT --options A,B``) landed with
    an empty target and no channel, invisible to every scoped check.
    """
    r = backend.execute_cli(
        "openclaw message poll --channel '#general' --question 'Vote?' --options 'yes,no'"
    )
    assert r.exit_code == 0
    rec = backend._messages[0]
    assert rec["type"] == "poll"
    assert rec["channel"] == "#general"
    assert rec["target"] == "#general"
    assert rec["options"] == ["yes", "no"]


def test_message_poll_explicit_target_wins_over_channel(backend):
    backend.execute_cli(
        "openclaw message poll --channel ops --target @lead --question 'Go?' --options 'y,n'"
    )
    rec = backend._messages[0]
    assert rec["target"] == "@lead"
    assert rec["channel"] == "ops"


def test_message_search(backend):
    backend.execute_cli("openclaw message send --channel slack --target @x --message 'hello team'")
    backend.execute_cli("openclaw message send --channel slack --target @y --message 'random stuff'")
    r = backend.execute_cli("openclaw message search --query hello")
    assert r.exit_code == 0
    assert "Found 1" in r.stdout


def test_message_react(backend):
    r = backend.execute_cli(
        "openclaw message react --target @alice --emoji 👍 --channel telegram"
    )
    assert r.exit_code == 0
    assert "👍" in r.stdout


def test_message_react_message_id_is_recorded(backend):
    """``--message-id`` is the documented flag and must survive into the record.

    It used to be dropped, so a reaction could not be tied back to the message
    it reacted to, and a doc-following call (which passes no ``--target``) was
    recorded with an empty target.
    """
    r = backend.execute_cli(
        "openclaw message react --channel telegram --message-id m-42 --emoji 👍"
    )
    assert r.exit_code == 0
    assert "m-42" in r.stdout
    rec = r.state_changes["messages"][0]
    assert rec["message_id"] == "m-42"
    assert rec["channel"] == "telegram"
    assert rec["target"] == "telegram"
    assert rec["type"] == "reaction"


# ---- Channels ----

def test_channels_login(backend):
    r = backend.execute_cli("openclaw channels login --channel telegram")
    assert r.exit_code == 0
    assert "telegram" in backend._channels
    assert backend._channels["telegram"]["status"] == "connected"


def test_channels_login_requires_flag(backend):
    r = backend.execute_cli("openclaw channels login telegram")
    assert r.exit_code == 1


def test_channels_list(backend):
    r = backend.execute_cli("openclaw channels list")
    assert r.exit_code == 0


def test_channels_config_read(backend):
    backend.execute_cli("openclaw channels login --channel slack")
    r = backend.execute_cli("openclaw channels config --channel slack")
    assert r.exit_code == 0


def test_channels_config_set(backend):
    backend.execute_cli("openclaw channels login --channel discord")
    r = backend.execute_cli(
        "openclaw channels config --channel discord --key notification_level --value all"
    )
    assert r.exit_code == 0
    assert backend._channels["discord"]["config"]["notification_level"] == "all"


def test_channels_config_not_found(backend):
    r = backend.execute_cli("openclaw channels config --channel nonexistent")
    assert r.exit_code == 1


# ---- Plugins ----

def test_plugin_install(backend):
    r = backend.execute_cli("openclaw plugins install slack-integration")
    assert r.exit_code == 0
    assert "installed" in r.stdout.lower()
    assert len(backend._plugins) == 1
    assert backend._plugins[0]["name"] == "slack-integration"
    assert r.state_changes is not None
    assert "plugins" in r.state_changes


def test_plugin_install_duplicate(backend):
    backend.execute_cli("openclaw plugins install weather-plugin")
    r = backend.execute_cli("openclaw plugins install weather-plugin")
    assert r.exit_code == 0
    assert "already installed" in r.stdout.lower()
    assert len(backend._plugins) == 1  # no duplicate


def test_plugin_list(backend):
    r = backend.execute_cli("openclaw plugins list")
    assert r.exit_code == 0
    assert "No plugins" in r.stdout


def test_plugin_remove(backend):
    backend.execute_cli("openclaw plugins install github-notifier")
    r = backend.execute_cli("openclaw plugins remove github-notifier")
    assert r.exit_code == 0
    assert len(backend._plugins) == 0


# ---- Cron ----

def test_cron_add(backend):
    r = backend.execute_cli(
        "openclaw cron add --schedule '0 9 * * *' --command 'openclaw status'"
    )
    assert r.exit_code == 0
    assert len(backend._cron_jobs) == 1
    assert backend._cron_jobs[0]["schedule"] == "0 9 * * *"
    assert r.state_changes is not None
    assert "cron_jobs" in r.state_changes


def test_cron_add_missing_required_fields_fails(backend):
    r = backend.execute_cli("openclaw cron add")
    assert r.exit_code == 1
    assert len(backend._cron_jobs) == 0


def test_cron_list(backend):
    r = backend.execute_cli("openclaw cron list")
    assert r.exit_code == 0
    assert "No cron" in r.stdout


def test_cron_delete(backend):
    backend.execute_cli("openclaw cron add --schedule '* * * * *' --command 'echo hi'")
    job_id = backend._cron_jobs[0]["id"]
    r = backend.execute_cli(f"openclaw cron delete {job_id}")
    assert r.exit_code == 0
    assert len(backend._cron_jobs) == 0


# ---- Security ----

def test_security_status(backend):
    r = backend.execute_cli("openclaw security")
    assert r.exit_code == 0
    assert "Auth" in r.stdout


def test_security_set_token(backend):
    r = backend.execute_cli("openclaw security set-token mysecrettoken")
    assert r.exit_code == 0
    assert backend._config["gateway"]["auth"]["token"] == "mysecrettoken"
    assert backend._config["gateway"]["auth"]["mode"] == "token"


# ---- Models ----

def test_models_list(backend):
    r = backend.execute_cli("openclaw models")
    assert r.exit_code == 0
    assert "claude" in r.stdout.lower()


def test_models_set(backend):
    r = backend.execute_cli("openclaw models set anthropic/claude-opus-4-6")
    assert r.exit_code == 0
    assert backend._config["agent"]["model"] == "anthropic/claude-opus-4-6"


# ---- Configure ----

def test_configure_set_value(backend):
    r = backend.execute_cli("openclaw configure gateway.port=9090")
    assert r.exit_code == 0
    assert backend._config["gateway"]["port"] == "9090"


# ---- Doctor ----

def test_doctor(backend):
    r = backend.execute_cli("openclaw doctor")
    assert r.exit_code == 0
    assert "Doctor" in r.stdout


# ---- Devices ----

def test_devices_pair(backend):
    r = backend.execute_cli("openclaw devices pair")
    assert r.exit_code == 0
    assert "pairing" in r.stdout.lower()


def test_devices_list(backend):
    r = backend.execute_cli("openclaw devices list")
    assert r.exit_code == 0


# ---- Unknown commands ----

def test_unknown_subcommand(backend):
    r = backend.execute_cli("openclaw nonexistent")
    assert r.exit_code == 1


def test_non_openclaw_command(backend):
    r = backend.execute_cli("ls -la")
    assert r.exit_code == 127


# ---- wait / undo (CommandResult must always include stderr) ----
#
# Regression guard: wait/undo build a CommandResult directly. CommandResult
# requires stderr (no default), so omitting it raises TypeError at runtime.
# These paths are normally intercepted by Environment.step(), so the bug was
# latent until the skill/backend are invoked directly. Calling them here would
# raise before the assertions if stderr were dropped again.

@pytest.mark.parametrize("cmd", ["wait 5", "undo 2"])
def test_mock_backend_wait_undo_have_stderr(backend, cmd):
    r = backend.execute_cli(cmd)
    assert r.exit_code == 0
    assert r.stderr == ""


@pytest.mark.parametrize("cmd", ["wait 5", "undo 2"])
def test_system_skill_wait_undo_have_stderr(cmd):
    r = SystemSkill().execute(cmd)
    assert r.exit_code == 0
    assert r.stderr == ""


# ---- MockBackend <-> MultiAppBackend parity for plugins / skills / models ----
#
# Evaluation runs on the `multi` backend, but nearly every test here uses
# `mock`, so SystemSkill silently drifted away from MockBackend:
#   * `plugins install` reported "enabled" and had no "already" branch,
#   * `plugins disable` was missing entirely,
#   * `skills info` (rewritten to `plugin skill info`) returned exit 1 + no
#     stdout,
#   * neither `plugins install` nor `plugins enable` emitted
#     ``state_changes={"plugins": [...]}``, which Environment maps to the
#     `plugins_installed` effect (114 shipped tasks depend on it),
#   * `models` was absent from the SystemSkill handler table, so every
#     `openclaw models set ...` returned "Unknown system subcommand: models"
#     and left `agent.model` at its default (80 shipped tasks depend on it).
#
# These tests pin both backends to the same observable behaviour so the drift
# cannot silently return.

def _parity_backend(cls, tmp_path_factory):
    """Initialize a backend from an identical on-disk config.

    Both model paths are preloaded on purpose: MockBackend ships no default
    `agent`/`agents` config while SystemSkill seeds one, and that pre-existing
    difference is not what these tests are about.
    """
    state_dir = tmp_path_factory.mktemp(cls.__name__)
    config = {
        "agent": {"model": "anthropic/claude-sonnet-4-5-20250929"},
        "agents": {
            "defaults": {"model": {"primary": "anthropic/claude-sonnet-4-5-20250929"}}
        },
        "gateway": {"port": 18789, "auth": {"mode": "token", "token": "test"}},
        "channels": {},
    }
    (state_dir / "openclaw.json").write_text(json.dumps(config))
    b = cls()
    b.initialize(str(state_dir), {})
    return b


_PARITY_SEQUENCES = [
    ["openclaw plugins list"],
    ["openclaw plugins install slack-connector", "openclaw plugins list"],
    [
        "openclaw plugins install slack-connector",
        "openclaw plugins install slack-connector",
    ],
    ["openclaw plugins enable slack-connector", "openclaw plugins list"],
    [
        "openclaw plugins install slack-connector",
        "openclaw plugins disable slack-connector",
        "openclaw plugins list",
    ],
    ["openclaw skills info web-search"],
    ["openclaw models set openai/gpt-4o", "openclaw models"],
]


@pytest.mark.parametrize(
    "sequence", _PARITY_SEQUENCES, ids=[s[0].split()[1] + f"_{i}" for i, s in enumerate(_PARITY_SEQUENCES)]
)
def test_mock_and_multi_backends_agree(sequence, tmp_path_factory):
    from clasharness.backend.multi_app_backend import MultiAppBackend

    mock = _parity_backend(MockBackend, tmp_path_factory)
    multi = _parity_backend(MultiAppBackend, tmp_path_factory)
    for command in sequence:
        a = mock.execute_cli(command)
        b = multi.execute_cli(command)
        assert (a.exit_code, a.stdout, a.stderr, a.state_changes) == (
            b.exit_code,
            b.stdout,
            b.stderr,
            b.state_changes,
        ), f"backend drift on {command!r}"


@pytest.fixture
def multi_backend(tmp_path_factory):
    from clasharness.backend.multi_app_backend import MultiAppBackend

    return _parity_backend(MultiAppBackend, tmp_path_factory)


@pytest.mark.parametrize("action", ["install", "enable"])
def test_multi_plugins_emit_plugins_state_change(multi_backend, action):
    """Environment maps state_changes["plugins"] -> `plugins_installed` effect."""
    r = multi_backend.execute_cli(f"openclaw plugins {action} slack-connector")
    assert r.exit_code == 0
    assert r.state_changes is not None
    assert [p["name"] for p in r.state_changes["plugins"]] == ["slack-connector"]


def test_multi_plugins_install_wording_distinguishes_repeat(multi_backend):
    """`install` must not report "enabled", and a repeat must say "already"."""
    first = multi_backend.execute_cli("openclaw plugins install slack-connector")
    assert "install" in first.stdout.lower()
    assert "enabled" not in first.stdout.lower()

    second = multi_backend.execute_cli("openclaw plugins install slack-connector")
    assert second.exit_code == 0
    assert "already" in second.stdout.lower()


def test_multi_plugins_disable_is_implemented(multi_backend):
    multi_backend.execute_cli("openclaw plugins install slack-connector")
    r = multi_backend.execute_cli("openclaw plugins disable slack-connector")
    assert r.exit_code == 0
    assert "disabled" in r.stdout.lower()
    assert "slack-connector" not in multi_backend.execute_cli(
        "openclaw plugins list"
    ).stdout


def test_multi_skills_info_is_implemented(multi_backend):
    """`openclaw skills info X` is rewritten to `plugin skill info X`."""
    r = multi_backend.execute_cli("openclaw skills info web-search")
    assert r.exit_code == 0
    assert "web-search" in r.stdout


def test_multi_models_set_updates_agent_model(multi_backend):
    r = multi_backend.execute_cli("openclaw models set openai/gpt-4o")
    assert r.exit_code == 0
    assert "Unknown system subcommand" not in r.stderr
    assert multi_backend.get_config()["agent"]["model"] == "openai/gpt-4o"
