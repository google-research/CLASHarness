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

from __future__ import annotations

from clasharness.core.environment import _as_named_mapping, _merge_named_entities


def test_as_named_mapping_handles_list_by_name_or_id():
    raw = [
        {"id": "main", "model": "vllm/Qwen"},
        {"name": "researcher", "model": "anthropic/claude-opus-4-6"},
    ]
    mapping = _as_named_mapping(raw)
    assert "main" in mapping
    assert "researcher" in mapping
    assert mapping["researcher"]["model"] == "anthropic/claude-opus-4-6"


def test_merge_named_entities_backfills_state():
    state_agents = {}
    effects = [{"name": "researcher", "model": "anthropic/claude-opus-4-6"}]
    _merge_named_entities(state_agents, effects)
    assert state_agents["researcher"]["model"] == "anthropic/claude-opus-4-6"


def test_as_named_mapping_handles_wrapped_dict_with_list_values():
    raw = {
        "channels": [
            {"id": "telegram", "status": "connected"},
            {"name": "discord", "status": "disconnected"},
        ]
    }
    mapping = _as_named_mapping(raw)
    assert mapping["telegram"]["status"] == "connected"
    assert mapping["discord"]["status"] == "disconnected"
