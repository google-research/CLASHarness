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

"""Grep skill implementation with local mock + optional real grep subprocess fallback."""

from __future__ import annotations

import re
import shlex
import subprocess
from typing import Any

from clasharness.backend.base import CommandResult, REAL_ENV_FLAG
from clasharness.backend.file_backend import FileSystemBackend
from clasharness.skills.base import Skill
from clasharness.skills.spec import CommandSpec, RealKind, SkillSpec


class GrepSkill(Skill):
    """Processes pattern search commands (grep) natively in Python or via system subprocess."""

    SPEC = SkillSpec(
        name="grep",
        category="compute",
        summary="Search for a pattern across files in the shared virtual filesystem.",
        prefixes=("grep",),
        real_kind=RealKind.PURE_COMPUTE,
        commands=(
            CommandSpec(
                "grep",
                "Search PATTERN in FILE (flags: -i -r -R -n -l).",
                args={
                    "pattern": {"type": "string"},
                    "file": {"type": "string"},
                },
                required=("pattern",),
            ),
        ),
        smoke=(
            "file create --path /g.txt --content hello",
            "grep hello /g.txt",
        ),
    )

    def __init__(self, file_backend: FileSystemBackend) -> None:
        super().__init__(prefixes=("grep",))
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
            return CommandResult(stdout="", stderr="grep skill not initialized", exit_code=1)

        parts = shlex.split(command.strip())
        if not parts or parts[0] != "grep":
            return CommandResult(stdout="", stderr="Not a grep command", exit_code=1)

        if self._real_mode:
            return self._execute_real(parts)
        return self._execute_mock(parts)

    def cleanup(self) -> None:
        self._initialized = False

    def get_state(self) -> dict[str, Any]:
        return {}

    # ------------------------------------------------------------------ #
    # Real grep Execution                                                #
    # ------------------------------------------------------------------ #

    def _execute_real(self, parts: list[str]) -> CommandResult:
        try:
            proc = subprocess.run(parts, capture_output=True, text=True, timeout=10)
            return CommandResult(
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                exit_code=proc.returncode,
            )
        except FileNotFoundError:
            return CommandResult(
                stdout="",
                stderr="grep not found in PATH",
                exit_code=127,
            )
        except subprocess.TimeoutExpired:
            return CommandResult(
                stdout="",
                stderr="grep command timed out",
                exit_code=124,
            )

    # ------------------------------------------------------------------ #
    # Mock grep Emulation                                                #
    # ------------------------------------------------------------------ #

    def _execute_mock(self, parts: list[str]) -> CommandResult:
        # Simple flag parser
        flags = []
        positional = []
        for p in parts[1:]:
            if p.startswith("-"):
                flags.append(p)
            else:
                positional.append(p)

        if len(positional) < 1:
            return CommandResult(
                stdout="",
                stderr="Usage: grep [options] <pattern> [file/dir]",
                exit_code=1,
            )

        pattern = positional[0]
        target = positional[1] if len(positional) > 1 else "."

        # Check options
        flat_flags = "".join(f.lstrip("-") for f in flags)
        case_insensitive = "i" in flat_flags
        recursive = any(x in flat_flags for x in ("r", "R"))
        show_line_numbers = "n" in flat_flags
        names_only = "l" in flat_flags

        # Compile pattern
        try:
            flags_re = re.IGNORECASE if case_insensitive else 0
            rx = re.compile(pattern, flags_re)
        except Exception as e:
            return CommandResult(
                stdout="",
                stderr=f"grep: error: invalid regex pattern: {e}",
                exit_code=2,
            )

        # Get mock files list
        from clasharness.backend.file_backend import _normalize
        files_state = self._file_backend.get_state().get("files", {})
        
        matches = []
        
        # Determine files to search
        target_norm = _normalize(target).lstrip("/")
        
        for rel_path, content in files_state.items():
            rel_path_clean = rel_path.lstrip("/")
            
            # Match target file or directory
            if recursive or target_norm == "." or not target_norm:
                # Search everything
                pass
            elif rel_path_clean != target_norm and not rel_path_clean.startswith(target_norm + "/"):
                # Doesn't match the file or subdirectory path
                continue

            # Scan lines
            lines = content.splitlines()
            for line_idx, line in enumerate(lines, 1):
                if rx.search(line):
                    if names_only:
                        matches.append(rel_path)
                        break  # Stop checking this file
                    else:
                        prefix = f"{rel_path}:" if (recursive or len(files_state) > 1) else ""
                        line_num = f"{line_idx}:" if show_line_numbers else ""
                        matches.append(f"{prefix}{line_num}{line}")

        if not matches:
            return CommandResult(stdout="", stderr="", exit_code=1) # exit code 1 if no matches found in grep

        return CommandResult(stdout="\n".join(matches), stderr="", exit_code=0)
