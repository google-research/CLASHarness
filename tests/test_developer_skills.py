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

"""Integration tests for Git, jq, and grep tools in Multi-agent MCP Server."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from clasharness.backend.mcp_server import MultiAgentArenaMCPServer


def test_mcp_git_workflow():
    server = MultiAgentArenaMCPServer("calendar_add_event_1", mode="multi")
    
    # 1. Check status
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "git_status",
            "arguments": {}
        }
    }
    res = server.handle_request(req)
    assert "nothing to commit, working tree clean" in res["result"]["content"][0]["text"]
    assert server.env.trajectory.steps[-1].action == "git status"

    # 2. Write a mock file
    req_write = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "write_file",
            "arguments": {
                "path": "app.py",
                "content": "print('hello')"
            }
        }
    }
    server.handle_request(req_write)

    # 3. Git add
    req_add = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "git_add",
            "arguments": {
                "path": "app.py"
            }
        }
    }
    server.handle_request(req_add)
    assert server.env.trajectory.steps[-1].action == "git add app.py"

    # 4. Check status shows staged
    req_status2 = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "git_status",
            "arguments": {}
        }
    }
    res_status2 = server.handle_request(req_status2)
    assert "Changes to be committed:" in res_status2["result"]["content"][0]["text"]
    assert "modified:   app.py" in res_status2["result"]["content"][0]["text"]

    # 5. Git commit
    req_commit = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "git_commit",
            "arguments": {
                "message": "add app script"
            }
        }
    }
    res_commit = server.handle_request(req_commit)
    assert "add app script" in res_commit["result"]["content"][0]["text"]
    assert "1 file changed" in res_commit["result"]["content"][0]["text"]
    assert server.env.trajectory.steps[-1].action == "git commit -m 'add app script'"

    # 6. Check status is clean again
    res_status3 = server.handle_request(req_status2)
    assert "nothing to commit, working tree clean" in res_status3["result"]["content"][0]["text"]


def test_mcp_jq_query():
    server = MultiAgentArenaMCPServer("calendar_add_event_1", mode="multi")

    # 1. Create a json file
    payload = {
        "app": "openclaw",
        "version": "2.0",
        "users": [
            {"name": "alice", "role": "admin"},
            {"name": "bob", "role": "user"}
        ]
    }
    req_write = {
        "jsonrpc": "2.0",
        "id": 10,
        "method": "tools/call",
        "params": {
            "name": "write_file",
            "arguments": {
                "path": "config.json",
                "content": json.dumps(payload)
            }
        }
    }
    server.handle_request(req_write)

    # 2. Run jq query for alice
    req_jq = {
        "jsonrpc": "2.0",
        "id": 11,
        "method": "tools/call",
        "params": {
            "name": "run_jq_query",
            "arguments": {
                "path": "config.json",
                "filter": ".users[0].name"
            }
        }
    }
    res_jq = server.handle_request(req_jq)
    assert res_jq["result"]["content"][0]["text"] == "alice"
    assert server.env.trajectory.steps[-1].action == "jq '.users[0].name' config.json"

    # 3. Run jq query for entire settings object
    req_jq_all = {
        "jsonrpc": "2.0",
        "id": 12,
        "method": "tools/call",
        "params": {
            "name": "run_jq_query",
            "arguments": {
                "path": "config.json",
                "filter": "."
            }
        }
    }
    res_jq_all = server.handle_request(req_jq_all)
    parsed = json.loads(res_jq_all["result"]["content"][0]["text"])
    assert parsed["app"] == "openclaw"


def test_mcp_search_grep():
    server = MultiAgentArenaMCPServer("calendar_add_event_1", mode="multi")

    # 1. Create text file
    content = "INFO: app started\nERROR: timeout connection on line 5\nINFO: check healthy\n"
    req_write = {
        "jsonrpc": "2.0",
        "id": 20,
        "method": "tools/call",
        "params": {
            "name": "write_file",
            "arguments": {
                "path": "logs/app.log",
                "content": content
            }
        }
    }
    server.handle_request(req_write)

    # 2. Run grep search
    req_grep = {
        "jsonrpc": "2.0",
        "id": 21,
        "method": "tools/call",
        "params": {
            "name": "search_grep",
            "arguments": {
                "path": "logs/app.log",
                "pattern": "ERROR"
            }
        }
    }
    res_grep = server.handle_request(req_grep)
    assert "2:ERROR: timeout connection on line 5" in res_grep["result"]["content"][0]["text"]
    assert server.env.trajectory.steps[-1].action == "grep -n ERROR logs/app.log"


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__]))
