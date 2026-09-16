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

import unittest
import os
import shutil
import sqlite3
import zipfile
import tempfile
from clasharness.backend.mcp_server import MultiAgentArenaMCPServer


class TestMCPCategorizedTools(unittest.TestCase):

    def setUp(self):
        # We use a dummy scenario task ID that has workspace support
        self.server = MultiAgentArenaMCPServer("calendar_add_event_1", mode="real")
        self.workspace_dir = str(self.server.env._state_manager.state_dir / "workspace")
        os.makedirs(self.workspace_dir, exist_ok=True)
        # Enable real execution fallback for testing the real subprocess behavior
        self.server.env._backend._shell_exec_skill._real_mode = True

    def tearDown(self):
        # Clean up files created in the virtual workspace
        if os.path.exists(self.workspace_dir):
            shutil.rmtree(self.workspace_dir)

    def test_run_script_python(self):
        # Create a dummy python script in workspace
        script_path = os.path.join(self.workspace_dir, "test_script.py")
        with open(script_path, "w") as f:
            f.write("import sys\nprint('ARGS:', sys.argv[1:])\n")

        req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "run_script",
                "arguments": {
                    "script_path": "test_script.py",
                    "args": ["hello", "world"]
                }
            }
        }
        res = self.server.handle_request(req)
        self.assertIn("ARGS: ['hello', 'world']", res["result"]["content"][0]["text"])

    def test_run_script_bash(self):
        # Create a dummy bash script in workspace
        script_path = os.path.join(self.workspace_dir, "test_script.sh")
        with open(script_path, "w") as f:
            f.write("#!/bin/bash\necho 'SHELL ARGS:' \"$@\"\n")
        os.chmod(script_path, 0o755)

        req = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "run_script",
                "arguments": {
                    "script_path": "test_script.sh",
                    "args": ["foo", "bar"]
                }
            }
        }
        res = self.server.handle_request(req)
        self.assertIn("SHELL ARGS: foo bar", res["result"]["content"][0]["text"])

    def test_decompress_archive_zip(self):
        # Create a test zip file
        test_file_content = "secret_key_123"
        zip_path = os.path.join(self.workspace_dir, "archive.zip")
        
        # Write zip file
        with zipfile.ZipFile(zip_path, "w") as myzip:
            myzip.writestr("secret.txt", test_file_content)

        req = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "decompress_archive",
                "arguments": {
                    "archive_path": "archive.zip",
                    "output_dir": "extracted_dir"
                }
            }
        }
        res = self.server.handle_request(req)
        self.assertIn("extracted_dir/secret.txt", res["result"]["content"][0]["text"])

        # Check file was extracted for real
        extracted_file = os.path.join(self.workspace_dir, "extracted_dir", "secret.txt")
        self.assertTrue(os.path.exists(extracted_file))
        with open(extracted_file, "r") as f:
            self.assertEqual(f.read().strip(), test_file_content)

    def test_execute_db_query(self):
        # Create a dummy SQLite database
        db_path = os.path.join(self.workspace_dir, "test.sqlite")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, role TEXT)")
        cursor.execute("INSERT INTO users (name, role) VALUES ('Avery', 'Risk_Analyst')")
        cursor.execute("INSERT INTO users (name, role) VALUES ('Casey', 'Database_Admin')")
        conn.commit()
        conn.close()

        req = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "execute_db_query",
                "arguments": {
                    "db_path": "test.sqlite",
                    "query": "SELECT role FROM users WHERE name='Avery'"
                }
            }
        }
        res = self.server.handle_request(req)
        self.assertIn("Risk_Analyst", res["result"]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
