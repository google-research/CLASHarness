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

import argparse
import json
import sys
from typing import Any

# Ensure we're in the right directory
try:
    from clasharness.core.environment import make_env
    from clasharness.backend import mcp_facade
except ImportError:
    # If run directly without PYTHONPATH
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from clasharness.core.environment import make_env
    from clasharness.backend import mcp_facade


# The curated MCP tool surface (``tools/list``) and the ``tools/call`` dispatch
# are both driven by the single declarative source of truth in ``mcp_facade``.
# ``MCP_TOOLS`` is retained as a module attribute for backwards compatibility.
MCP_TOOLS = mcp_facade.build_tools_list()


class MultiAgentArenaMCPServer:
    """Zero-dependency JSON-RPC MCP server over StdIO."""
    
    def __init__(self, task_id: str, mode: str = "mock"):
        self.env = make_env(task_id, mode=mode)
        self.env.reset()
        self.tools_list = MCP_TOOLS

    def _get_channel_url(self, channel_name: str) -> str:
        backend = getattr(self.env, "_backend", None)
        if backend and hasattr(backend, "_channel_skill"):
            channels = backend._channel_skill.get_state().get("channels", {})
            ch = channels.get(channel_name.lower())
            if ch and ch.get("url"):
                return ch["url"]
        if channel_name.lower() == "telegram":
            return "https://api.telegram.org/botTOKEN/sendMessage"
        return "https://hooks.slack.com/services/T000/B000/XXXX"


    def serve(self):
        """Read JSON-RPC from stdin, write to stdout."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
                
            try:
                request = json.loads(line)
                response = self.handle_request(request)
                if response:
                    print(json.dumps(response), flush=True)
            except Exception as e:
                # If error happens, write an error response if it had an ID
                if isinstance(line, str) and '"id"' in line:
                    try:
                        req_id = json.loads(line).get("id")
                        if req_id:
                            print(json.dumps({
                                "jsonrpc": "2.0",
                                "id": req_id,
                                "error": {"code": -32603, "message": str(e)}
                            }), flush=True)
                    except:
                        pass

    def handle_request(self, req: dict[str, Any]) -> dict[str, Any] | None:
        """Handle a single JSON-RPC request."""
        req_id = req.get("id")
        method = req.get("method")
        
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "multi-agent-arena-mcp", "version": "0.1.0"}
                }
            }
            
        if method == "notifications/initialized":
            return None
            
        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": self.tools_list
                }
            }
            
        if method == "tools/call":
            params = req.get("params", {})
            name = params.get("name")
            arguments = params.get("arguments", {})
            
            try:
                result_content = self.call_tool(name, arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": result_content
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error: {str(e)}"}],
                        "isError": True
                    }
                }
                
        # Unknown method
        if req_id is not None:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"}
            }
        return None

    def call_tool(self, name: str, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        # Build the workspace CLI (or a direct result) from the declarative
        # facade, then execute it in the environment.
        result = mcp_facade.build_cli(
            name, arguments, get_channel_url=self._get_channel_url
        )
        if isinstance(result, mcp_facade.DirectResult):
            return result.content

        # Execute in environment
        obs, reward, done, info = self.env.step(result)

        output_text = obs.command_output
        if obs.error_output:
            output_text += f"\nError: {obs.error_output}"

        return [{"type": "text", "text": output_text}]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True, help="Task ID to initialize the environment with.")
    parser.add_argument("--mode", choices=["mock", "multi", "real", "hybrid"], default="mock", help="Backend mode.")
    args = parser.parse_args()
    
    server = MultiAgentArenaMCPServer(task_id=args.task_id, mode=args.mode)
    server.serve()
