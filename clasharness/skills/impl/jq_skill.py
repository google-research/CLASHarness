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

"""Jq skill implementation with local mock + optional real jq subprocess fallback."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from typing import Any

from clasharness.backend.base import CommandResult, REAL_ENV_FLAG
from clasharness.backend.file_backend import FileSystemBackend
from clasharness.skills.base import Skill
from clasharness.skills.spec import CommandSpec, RealKind, SkillSpec


def _evaluate_jq(data: Any, filter_str: str) -> Any:
    filter_str = filter_str.strip()
    if not filter_str or filter_str == ".":
        return data

    # Parse basic lookups like .key, .[index], .key[index], .key.subkey
    # Matches a dot-prefixed key name or a bracketed index
    pattern = re.compile(r'\.?([a-zA-Z0-9_\-]+)|\[([0-9]+)\]')
    matches = pattern.findall(filter_str)
    
    if not matches:
        return data

    curr = data
    for key, idx in matches:
        if key:
            if isinstance(curr, dict) and key in curr:
                curr = curr[key]
            else:
                return None
        elif idx:
            index = int(idx)
            if isinstance(curr, list) and 0 <= index < len(curr):
                curr = curr[index]
            else:
                return None
    return curr


class JqSkill(Skill):
    """Processes JSON files using jq expressions natively in Python or via system subprocess."""

    SPEC = SkillSpec(
        name="jq",
        category="compute",
        summary="Evaluate a jq filter over a JSON file in the shared virtual filesystem.",
        prefixes=("jq",),
        real_kind=RealKind.PURE_COMPUTE,
        commands=(
            CommandSpec(
                "jq",
                "Apply a jq FILTER to FILE (e.g. `jq .name /data.json`).",
                args={
                    "filter": {"type": "string"},
                    "file": {"type": "string"},
                },
                required=("filter", "file"),
            ),
        ),
        smoke=(
            "file create --path /d.json --content '{\"n\": 1}'",
            "jq .n /d.json",
        ),
    )

    def __init__(self, file_backend: FileSystemBackend) -> None:
        super().__init__(prefixes=("jq",))
        self._file_backend = file_backend
        self._initialized = False
        self._real_mode = False

    def initialize(self, state_dir: str, env_vars: dict[str, str]) -> None:
        del state_dir
        self._initialized = True
        val = str(env_vars.get(REAL_ENV_FLAG, "")).lower()
        self._real_mode = val in {"1", "true", "yes", "on"}

    def execute(self, command: str) -> CommandResult:
        if not self._initialized:
            return CommandResult(stdout="", stderr="jq skill not initialized", exit_code=1)

        parts = shlex.split(command.strip())
        if not parts or parts[0] != "jq":
            return CommandResult(stdout="", stderr="Not a jq command", exit_code=1)

        if self._real_mode:
            return self._execute_real(parts)
        return self._execute_mock(parts)

    def cleanup(self) -> None:
        self._initialized = False

    def get_state(self) -> dict[str, Any]:
        return {}

    # ------------------------------------------------------------------ #
    # Real jq Execution                                                  #
    # ------------------------------------------------------------------ #

    def _execute_real(self, parts: list[str]) -> CommandResult:
        try:
            proc = subprocess.run(parts, capture_output=True, text=True, timeout=5)
            return CommandResult(
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                exit_code=proc.returncode,
            )
        except FileNotFoundError:
            return CommandResult(
                stdout="",
                stderr="jq not found in PATH",
                exit_code=127,
            )
        except subprocess.TimeoutExpired:
            return CommandResult(
                stdout="",
                stderr="jq command timed out",
                exit_code=124,
            )

    # ------------------------------------------------------------------ #
    # Mock jq Emulation                                                  #
    # ------------------------------------------------------------------ #

    def _execute_mock(self, parts: list[str]) -> CommandResult:
        # Expected: jq <filter> <file>
        if len(parts) < 3:
            return CommandResult(
                stdout="",
                stderr="Usage: jq <filter> <file>",
                exit_code=1,
            )

        filter_str = parts[1]
        filepath = parts[2]

        # Read file content from filesystem backend
        from clasharness.backend.file_backend import _normalize
        norm_path = _normalize(filepath)
        files_state = self._file_backend.get_state().get("files", {})
        
        if norm_path not in files_state:
            return CommandResult(
                stdout="",
                stderr=f"jq: error: {filepath}: No such file or directory",
                exit_code=1,
            )

        content = files_state[norm_path]
        try:
            data = json.loads(content)
        except Exception as e:
            return CommandResult(
                stdout="",
                stderr=f"jq: error: Invalid JSON: {e}",
                exit_code=1,
            )

        res = _evaluate_jq(data, filter_str)
        if res is None:
            return CommandResult(stdout="null", stderr="", exit_code=0)

        # Print formatted output
        if isinstance(res, (dict, list)):
            out = json.dumps(res, indent=2, ensure_ascii=False)
        else:
            out = str(res)

        return CommandResult(stdout=out, stderr="", exit_code=0)
