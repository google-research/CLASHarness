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

"""Tests for default state-manager OpenClaw config schema."""

from __future__ import annotations

import json

from clasharness.utils.state_manager import StateManager


def test_default_config_uses_new_agents_defaults_schema():
    sm = StateManager(base_config_name="nonexistent_base_config_for_test")
    try:
        sm.initialize()
        config = sm.get_config()
        assert "agents" in config
        assert "defaults" in config["agents"]
        assert "model" in config["agents"]["defaults"]
        assert "primary" in config["agents"]["defaults"]["model"]
        # Legacy root path should no longer exist in generated default config.
        assert not isinstance(config.get("agent"), dict)
    finally:
        sm.cleanup()


def test_default_config_aligns_gateway_token_from_host_config(monkeypatch, tmp_path):
    host_home = tmp_path / "host_home"
    host_cfg_dir = host_home / ".openclaw"
    host_cfg_dir.mkdir(parents=True, exist_ok=True)
    host_cfg_path = host_cfg_dir / "openclaw.json"
    host_cfg_path.write_text(
        json.dumps(
            {
                "gateway": {
                    "bind": "loopback",
                    "port": 18789,
                    "auth": {"mode": "token", "token": "host-token-123"},
                }
            }
        )
    )
    monkeypatch.setenv("HOME", str(host_home))

    sm = StateManager(base_config_name="nonexistent_base_config_for_test")
    try:
        sm.initialize()
        config = sm.get_config()
        gateway = config.get("gateway", {})
        assert gateway.get("auth", {}).get("token") == "host-token-123"
        assert gateway.get("remote", {}).get("token") == "host-token-123"
        assert gateway.get("remote", {}).get("url") == "ws://127.0.0.1:18789"
    finally:
        sm.cleanup()
