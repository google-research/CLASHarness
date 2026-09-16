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

"""File skill adapter translating standard Unix CLI commands to FileSystemBackend."""

from __future__ import annotations

import shlex
from typing import Any

from clasharness.backend.base import CommandResult
from clasharness.backend.file_backend import FileSystemBackend
from clasharness.skills.base import Skill
from clasharness.skills.spec import CommandSpec, RealKind, SkillSpec


class FileSkillAdapter(Skill):
    """Exposes standard Unix commands (cat, echo, ls, rm, mv, mkdir) to the registry."""

    SPEC = SkillSpec(
        name="file",
        category="filesystem",
        summary="Virtual filesystem operations plus Unix aliases (cat/echo/ls/rm/mv/mkdir).",
        prefixes=("file", "cat", "echo", "ls", "rm", "mv", "mkdir"),
        real_kind=RealKind.PURE_COMPUTE,
        commands=(
            CommandSpec(
                "file create",
                "Create a file with content.",
                args={"--path": {"type": "string"}, "--content": {"type": "string"}},
                required=("--path",),
                emits_effects=("files_created",),
            ),
            CommandSpec(
                "file read",
                "Read a file's content.",
                args={"--path": {"type": "string"}},
                required=("--path",),
            ),
            CommandSpec(
                "file delete",
                "Delete a file.",
                args={"--path": {"type": "string"}},
                required=("--path",),
                emits_effects=("files_deleted",),
            ),
            CommandSpec(
                "file list",
                "List files under a path.",
                args={"--path": {"type": "string"}},
            ),
            CommandSpec(
                "file move",
                "Move/rename a file.",
                args={"--src": {"type": "string"}, "--dst": {"type": "string"}},
                required=("--src", "--dst"),
            ),
            CommandSpec(
                "file append",
                "Append content to a file.",
                args={"--path": {"type": "string"}, "--content": {"type": "string"}},
                required=("--path",),
                emits_effects=("files_appended",),
            ),
            CommandSpec("cat", "Unix alias: print a file (`cat FILE`)."),
            CommandSpec("echo", "Unix alias: echo text or write via `>`/`>>`."),
            CommandSpec("ls", "Unix alias: list a directory (`ls PATH`)."),
            CommandSpec("rm", "Unix alias: delete a file (`rm PATH`)."),
            CommandSpec("mv", "Unix alias: move a file (`mv SRC DST`)."),
            CommandSpec("mkdir", "Unix alias: create a directory (no-op)."),
        ),
        smoke=(
            "file create --path /a.txt --content hi",
            "file read --path /a.txt",
            "file list --path /",
        ),
    )

    def __init__(self, backend: FileSystemBackend) -> None:
        super().__init__(prefixes=("file", "cat", "echo", "ls", "rm", "mv", "mkdir"))
        self._backend = backend

    def initialize(self, state_dir: str, env_vars: dict[str, str]) -> None:
        self._backend.initialize(state_dir, env_vars)

    def execute(self, command: str) -> CommandResult:
        try:
            parts = shlex.split(command.strip())
        except ValueError as e:
            return CommandResult(stdout="", stderr=f"Invalid command syntax: {e}", exit_code=1)

        if not parts:
            return CommandResult(stdout="", stderr="Empty command", exit_code=1)

        prefix = parts[0]
        if prefix == "file":
            # Pass directly to legacy file backend parsing
            return self._backend.execute_cli(command)

        # Standard Unix Command Translations
        if prefix == "cat":
            if len(parts) < 2:
                return CommandResult(stdout="", stderr="cat: missing file operand", exit_code=1)
            filepath = parts[1]
            return self._backend._cmd_read(["--path", filepath])

        if prefix == "ls":
            clean_path = "."
            # Filter out flags like -la, -lh, -l
            args = [p for p in parts[1:] if not p.startswith("-")]
            if args:
                clean_path = args[0]
            return self._backend._cmd_list(["--path", clean_path])

        if prefix == "rm":
            # Filter out flags like -rf, -f
            args = [p for p in parts[1:] if not p.startswith("-")]
            if not args:
                return CommandResult(stdout="", stderr="rm: missing operand", exit_code=1)
            return self._backend._cmd_delete(["--path", args[0]])

        if prefix == "mv":
            if len(parts) < 3:
                return CommandResult(stdout="", stderr="mv: missing file operand", exit_code=1)
            return self._backend._cmd_move(["--src", parts[1], "--dst", parts[2]])

        if prefix == "mkdir":
            # Mock mkdir success
            return CommandResult(stdout="", stderr="", exit_code=0)

        if prefix == "echo":
            # echo "content" > filepath -> file create --path filepath --content "content"
            # echo "content" >> filepath -> file append --path filepath --content "content"
            if ">" in parts or ">>" in parts:
                op = ">>" if ">>" in parts else ">"
                idx = parts.index(op)
                content = " ".join(parts[1:idx]).strip("'\"")
                filepath = parts[idx + 1]
                if op == ">":
                    return self._backend._cmd_create(["--path", filepath, "--content", content])
                else:
                    return self._backend._cmd_append(["--path", filepath, "--content", content])
            else:
                content = " ".join(parts[1:]).strip("'\"")
                return CommandResult(stdout=content, stderr="", exit_code=0)

        return CommandResult(stdout="", stderr=f"Unsupported standard command: {prefix}", exit_code=1)

    def get_state(self) -> dict[str, Any]:
        return self._backend.get_state()

    def to_state(self) -> dict[str, Any]:
        return self._backend.to_state()

    def from_state(self, data: dict[str, Any]) -> None:
        self._backend.from_state(data)

    def cleanup(self) -> None:
        self._backend.cleanup()
