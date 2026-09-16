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

import json
import io
import sys
from clasharness.backend.mcp_server import MultiAgentArenaMCPServer

def test_mcp_server():
    server = MultiAgentArenaMCPServer("calendar_add_event_1")
    
    # Test 1: Initialize
    req1 = {"jsonrpc": "2.0", "id": 1, "method": "initialize"}
    res1 = server.handle_request(req1)
    assert res1["result"]["serverInfo"]["name"] == "multi-agent-arena-mcp"
    
    # Test 2: List Tools
    req2 = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    res2 = server.handle_request(req2)
    tools = res2["result"]["tools"]
    assert len(tools) > 10
    
    # Verify undo is present
    undo_tool = next((t for t in tools if t["name"] == "undo"), None)
    assert undo_tool is not None
    
    # Verify create_calendar_event and its schema
    cal_tool = next((t for t in tools if t["name"] == "create_calendar_event"), None)
    assert cal_tool is not None
    assert "title" in cal_tool["inputSchema"]["properties"]
    assert "start_time" in cal_tool["inputSchema"]["properties"]
    assert "location" in cal_tool["inputSchema"]["properties"]
    assert "title" in cal_tool["inputSchema"]["required"]
    
    # Test 3: Call Tool (Wait)
    req3 = {
        "jsonrpc": "2.0", 
        "id": 3, 
        "method": "tools/call", 
        "params": {
            "name": "wait",
            "arguments": {
                "seconds": 5
            }
        }
    }
    res3 = server.handle_request(req3)
    assert res3["result"]["content"][0]["text"] == "Waited for 5 seconds."
    assert server.env._virtual_time == 5
    
    # Test 4: Call Tool (Undo)
    req4 = {
        "jsonrpc": "2.0", 
        "id": 4, 
        "method": "tools/call", 
        "params": {
            "name": "undo",
            "arguments": {
                "steps": 1
            }
        }
    }
    res4 = server.handle_request(req4)
    assert "Successfully undid 1 steps" in res4["result"]["content"][0]["text"]
    assert server.env._virtual_time == 5  # Time doesn't revert
 
    # Test 5: Call Tool (Config Set - clean parameters)
    req5 = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "set_config_value",
            "arguments": {
                "path": "agent.model",
                "value": "gemini-1.5-flash"
            }
        }
    }
    res5 = server.handle_request(req5)
    assert "Updated agent.model." in res5["result"]["content"][0]["text"]
    
    print("All MCP Server tests passed!")


def test_mcp_server_real_mode():
    from clasharness.backend.multi_app_backend import MultiAppBackend

    server = MultiAgentArenaMCPServer("calendar_add_event_1", mode="real")
    assert isinstance(server.env._backend, MultiAppBackend)

    # Call a tool in real mode. It should run natively in Python without launching openclaw subprocess.
    req = {
        "jsonrpc": "2.0",
        "id": 10,
        "method": "tools/call",
        "params": {
            "name": "list_agents",
            "arguments": {}
        }
    }
    res = server.handle_request(req)
    # The native skill should run cleanly and return "No agents configured." with exit code 0.
    content = res["result"]["content"][0]["text"]
    assert "No agents configured" in content
    print("Real mode MCP server test passed natively!")



def test_mcp_standard_command_mapping():
    server = MultiAgentArenaMCPServer("calendar_add_event_1", mode="multi")

    
    # 1. Test write_file -> echo
    req_write = {
        "jsonrpc": "2.0",
        "id": 20,
        "method": "tools/call",
        "params": {
            "name": "write_file",
            "arguments": {
                "path": "test_mcp.txt",
                "content": "hello world from mcp"
            }
        }
    }
    res_write = server.handle_request(req_write)
    # The return content should indicate success
    assert "File created: /test_mcp.txt" in res_write["result"]["content"][0]["text"]

    
    # Check executed action in environment history/trajectory
    last_step = server.env.trajectory.steps[-1]
    assert last_step.action == "echo 'hello world from mcp' > test_mcp.txt"
    
    # 2. Test read_file -> cat
    req_read = {
        "jsonrpc": "2.0",
        "id": 21,
        "method": "tools/call",
        "params": {
            "name": "read_file",
            "arguments": {
                "path": "test_mcp.txt"
            }
        }
    }
    res_read = server.handle_request(req_read)
    assert "hello world from mcp" in res_read["result"]["content"][0]["text"]
    
    last_step = server.env.trajectory.steps[-1]
    assert last_step.action == "cat test_mcp.txt"
    
    # 3. Test send_message -> curl
    req_msg = {
        "jsonrpc": "2.0",
        "id": 22,
        "method": "tools/call",
        "params": {
            "name": "send_message",
            "arguments": {
                "channel": "slack",
                "target": "@bob",
                "message": "hi bob"
            }
        }
    }
    res_msg = server.handle_request(req_msg)
    assert res_msg["result"]["content"][0]["text"] == "ok"
    
    last_step = server.env.trajectory.steps[-1]
    # Check that the command is a curl POST call to the slack webhook
    assert last_step.action.startswith("curl -X POST -H 'Content-Type: application/json'")
    assert "https://hooks.slack.com/services/T000/B000/XXXX" in last_step.action
    assert '{"text": "hi bob"}' in last_step.action


    print("MCP standard command mapping test passed successfully!")


if __name__ == "__main__":
    test_mcp_server()
    test_mcp_server_real_mode()
    test_mcp_standard_command_mapping()

