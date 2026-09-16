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

"""Phase 4: conflict-mechanic + latent-bug fixes."""

from __future__ import annotations

from clasharness.core.environment import OpenClawEnv, _get_available_commands


# --------------------------------------------------------------------------- #
# D1: temporal / async primitives are discoverable in available_commands       #
# --------------------------------------------------------------------------- #


def test_available_commands_surface_temporal_primitives():
    cmds = _get_available_commands(["messaging"])
    blob = "\n".join(cmds)
    assert any(c.startswith("wait ") for c in cmds), "wait must be surfaced"
    assert any(c.startswith("undo") for c in cmds), "undo must be surfaced"
    assert "agent delegate" in blob, "agent delegate must be surfaced"
    assert "agent status" in blob, "agent status must be surfaced"


# --------------------------------------------------------------------------- #
# D3: background (late) messages reach the searchable message pool             #
# --------------------------------------------------------------------------- #


def test_background_message_is_searchable_multi_backend():
    with OpenClawEnv(task_id="agent_create_1", backend="multi") as env:
        env.reset()
        env._background_events.append({
            "type": "session_append",
            "trigger_time": 5,
            "session_id": "general",
            "message": {
                "message_id": "m1",
                "sender": "ops_bot",
                "text": "regression detected in payment service",
                "target": "general",
            },
        })
        # Advance virtual time to fire the background event.
        env.step("wait 10")
        # The late message must now be findable via message search.
        obs, _, _, _ = env.step("message search --channel general --query regression")
        assert "regression detected in payment service" in obs.command_output


# --------------------------------------------------------------------------- #
# E1: gateway status is reported (multi backend) — json import bug fixed        #
# --------------------------------------------------------------------------- #


def test_gateway_status_reported_multi_backend():
    with OpenClawEnv(task_id="agent_create_1", backend="multi") as env:
        env.reset()
        status = env._backend.get_gateway_status()
        assert status is not None
        assert "running" in status


# --------------------------------------------------------------------------- #
# E2: background .py writes are verbatim (no silent f-string rewriting)         #
# --------------------------------------------------------------------------- #


def test_background_py_write_is_verbatim():
    with OpenClawEnv(task_id="agent_create_1", backend="multi") as env:
        env.reset()
        # Content the old regex would have mangled (it stripped escaped quotes
        # around f-strings). It must now be written byte-for-byte.
        content = 'msg = f\\"{n}\\"\n'
        env._background_events.append({
            "type": "file_create",
            "trigger_time": 1,
            "path": "/gen.py",
            "new_content": content,
        })
        env.step("wait 2")
        written = env._backend.file_backend._files.get("/gen.py")
        assert written == content
