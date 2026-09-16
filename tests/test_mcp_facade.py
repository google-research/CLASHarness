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

"""Parity tests for the declarative MCP facade (``backend/mcp_facade.py``).

The facade is the single source of truth for both the ``tools/list`` schema
surface and the ``tools/call`` dispatch (CLI-command synthesis). These tests
pin its output against two frozen goldens so any accidental drift in the
curated 104-tool contract is caught:

  * ``fixtures/mcp_tools_golden.json``    -- the exact ``tools/list`` payload.
  * ``fixtures/mcp_dispatch_golden.json`` -- ``build_cli`` cases covering
    every tool plus optional-argument / composite / direct-result branches.
"""

import json
import os
import shlex

from clasharness.backend import mcp_facade
from clasharness.backend.mcp_server import MultiAgentArenaMCPServer
from clasharness.core.environment import make_env

_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")

# Number of tools in the curated, stable MCP API surface.
_CURATED_TOOL_COUNT = 104


def _load(name):
    with open(os.path.join(_FIXTURES, name)) as fh:
        return json.load(fh)


def _fallback_channel_url(channel_name):
    """Mirror of ``MultiAgentArenaMCPServer._get_channel_url`` for the case
    where no channel is configured -- the state under which the dispatch
    golden was captured. ``test_server_channel_url_fallback_matches_stub``
    pins this against the real server so the two cannot silently diverge.
    """
    if channel_name.lower() == "telegram":
        return "https://api.telegram.org/botTOKEN/sendMessage"
    return "https://hooks.slack.com/services/T000/B000/XXXX"


def test_tools_list_matches_golden():
    """``build_tools_list()`` reproduces the frozen schema surface exactly."""
    golden = _load("mcp_tools_golden.json")
    live = mcp_facade.build_tools_list()

    assert len(live) == _CURATED_TOOL_COUNT
    assert len(golden) == _CURATED_TOOL_COUNT

    gmap = {t["name"]: t for t in golden}
    lmap = {t["name"]: t for t in live}
    assert set(lmap) == set(gmap), (
        f"tool-name drift: {set(lmap) ^ set(gmap)}"
    )
    for name, want in gmap.items():
        assert lmap[name] == want, f"schema drift for tool {name!r}"


def test_all_tools_covered_by_dispatch_golden():
    """Every curated tool has at least one dispatch case."""
    golden = _load("mcp_dispatch_golden.json")
    covered = {c["name"] for c in golden}
    assert covered == set(mcp_facade.ALL_TOOL_NAMES)
    assert len(mcp_facade.ALL_TOOL_NAMES) == _CURATED_TOOL_COUNT


def test_build_cli_matches_dispatch_golden():
    """``build_cli`` reproduces every frozen dispatch case byte-for-byte.

    This covers the CLI-string tools as well as the two ``DirectResult``
    (input-validation error) branches that return content without stepping
    the environment.
    """
    golden = _load("mcp_dispatch_golden.json")
    for case in golden:
        got = mcp_facade.build_cli(
            case["name"], case["args"], get_channel_url=_fallback_channel_url
        )
        if "direct" in case:
            assert isinstance(got, mcp_facade.DirectResult), (
                f"{case['name']} expected DirectResult, got {got!r}"
            )
            assert got.content == case["direct"], (
                f"direct-result drift for {case['name']} args={case['args']}"
            )
        else:
            assert not isinstance(got, mcp_facade.DirectResult), (
                f"{case['name']} unexpectedly returned DirectResult"
            )
            assert got == case["cli"], (
                f"cli drift for {case['name']} args={case['args']}\n"
                f"  want: {case['cli']!r}\n  got:  {got!r}"
            )


def test_unknown_tool_raises():
    """An unregistered tool name is a hard error, not a silent no-op."""
    try:
        mcp_facade.build_cli("nonexistent_tool", {})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown tool")


def test_server_channel_url_fallback_matches_stub():
    """Pin the real server's no-channel fallback that produced the golden.

    If the server's fallback URLs ever change, the dispatch golden (captured
    with those URLs) would go stale; this guards against that drift.
    """
    server = MultiAgentArenaMCPServer("calendar_add_event_1", mode="multi")
    assert server._get_channel_url("telegram") == _fallback_channel_url("telegram")
    assert server._get_channel_url("slack") == _fallback_channel_url("slack")


def test_projected_commands_route_to_registered_skill():
    """Lenient cross-check: every synthesized command's leading token is a
    prefix that actually routes to a registered skill. Independent of the
    golden capture, this catches a tool wired to a nonexistent command.
    """
    env = make_env("calendar_add_event_1", mode="multi")
    env.reset()
    prefixes = set(env._backend._skill_registry.list_prefixes())

    golden = _load("mcp_dispatch_golden.json")
    for case in golden:
        if "cli" not in case:  # DirectResult cases synthesize no command
            continue
        try:
            tokens = shlex.split(case["cli"])
        except ValueError:
            tokens = case["cli"].split()
        assert tokens, f"empty command for {case['name']}"
        assert tokens[0] in prefixes, (
            f"tool {case['name']} -> command {tokens[0]!r} is not a "
            f"registered skill prefix"
        )


if __name__ == "__main__":
    test_tools_list_matches_golden()
    test_all_tools_covered_by_dispatch_golden()
    test_build_cli_matches_dispatch_golden()
    test_unknown_tool_raises()
    test_server_channel_url_fallback_matches_stub()
    test_projected_commands_route_to_registered_skill()
    print("All MCP facade parity tests passed!")
