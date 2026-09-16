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

"""Git skill implementation with local mock + optional real git subprocess fallback."""

from __future__ import annotations

import datetime as dt
import hashlib
import shlex
import subprocess
import time
from typing import Any

from clasharness.backend.base import CommandResult, REAL_ENV_FLAG
from clasharness.skills.base import Skill
from clasharness.skills.spec import CommandSpec, RealKind, SkillSpec


class GitSkill(Skill):
    """Git version control skill for managing staging and commits in the sandbox."""

    SPEC = SkillSpec(
        name="git",
        category="vcs",
        summary="Git staging/commit workflow (mock, with optional real subprocess).",
        prefixes=("git",),
        real_kind=RealKind.SUBPROCESS,
        commands=(
            CommandSpec("git status", "Show working-tree status."),
            CommandSpec(
                "git add",
                "Stage files (path or `.`).",
                args={"path": {"type": "string"}},
            ),
            CommandSpec(
                "git commit",
                "Commit staged changes.",
                args={"-m": {"type": "string"}},
                emits_effects=("git_commits_created",),
            ),
            CommandSpec("git log", "Show commit history."),
            CommandSpec("git diff", "Show unstaged changes."),
            CommandSpec("git init", "Initialize a repository."),
        ),
        smoke=(
            "git status",
            "git log",
        ),
    )

    # Episode-mutable state captured for generic snapshot/restore (undo).
    _snapshot_attrs = ("_commits", "_staged", "_modified")

    def __init__(self) -> None:
        super().__init__(prefixes=("git",))
        self._commits: list[dict[str, Any]] = []
        self._staged: set[str] = set()
        self._modified: set[str] = set()
        self._initialized = False
        self._real_git = False

    def initialize(self, state_dir: str, env_vars: dict[str, str]) -> None:
        del state_dir
        self._initialized = True
        val = str(env_vars.get(REAL_ENV_FLAG, "")).lower()
        self._real_git = val in {"1", "true", "yes", "on"}
        
        # Initialize default commit log
        self._commits = [
            {
                "hash": "a1b2c3d",
                "author": "developer@example.com",
                "date": dt.datetime.fromtimestamp(1772184000).strftime("%a %b %d %H:%M:%S %Y +0000"),
                "message": "Initial commit",
            }
        ]
        self._staged = set()
        self._modified = set()

    def execute(self, command: str) -> CommandResult:
        if not self._initialized:
            return CommandResult(stdout="", stderr="Git skill not initialized", exit_code=1)

        parts = shlex.split(command.strip())
        if not parts or parts[0] != "git":
            return CommandResult(stdout="", stderr="Not a git command", exit_code=1)

        if self._real_git:
            return self._execute_real(parts)
        return self._execute_mock(parts)

    def cleanup(self) -> None:
        self._commits = []
        self._staged = set()
        self._modified = set()
        self._initialized = False

    def get_state(self) -> dict[str, Any]:
        return {
            "git_commits": self._commits,
            "git_staged": list(self._staged),
            "git_modified": list(self._modified),
        }

    # ------------------------------------------------------------------ #
    # Real Git Execution                                                 #
    # ------------------------------------------------------------------ #

    def _execute_real(self, parts: list[str]) -> CommandResult:
        try:
            # We run git inside the current directory
            proc = subprocess.run(parts, capture_output=True, text=True, timeout=10)
            return CommandResult(
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                exit_code=proc.returncode,
            )
        except FileNotFoundError:
            return CommandResult(
                stdout="",
                stderr="git not found in PATH",
                exit_code=127,
            )
        except subprocess.TimeoutExpired:
            return CommandResult(
                stdout="",
                stderr="git command timed out",
                exit_code=124,
            )

    # ------------------------------------------------------------------ #
    # Mock Git Emulation                                                 #
    # ------------------------------------------------------------------ #

    def _execute_mock(self, parts: list[str]) -> CommandResult:
        if len(parts) < 2:
            return CommandResult(
                stdout="git subcommand required. (status, add, commit, log, diff)",
                stderr="",
                exit_code=0,
            )

        subcmd = parts[1]
        args = parts[2:]

        handlers = {
            "status": self._cmd_status,
            "add": self._cmd_add,
            "commit": self._cmd_commit,
            "log": self._cmd_log,
            "diff": self._cmd_diff,
            "init": self._cmd_init,
        }

        handler = handlers.get(subcmd)
        if handler is None:
            return CommandResult(
                stdout="",
                stderr=f"Unknown git subcommand: {subcmd}",
                exit_code=1,
            )
        return handler(args)

    def _cmd_init(self, args: list[str]) -> CommandResult:
        del args
        return CommandResult(
            stdout="Initialized empty Git repository in mock workspace",
            stderr="",
            exit_code=0,
        )

    def _cmd_status(self, args: list[str]) -> CommandResult:
        del args
        lines = ["On branch main"]
        if not self._staged and not self._modified:
            lines.append("nothing to commit, working tree clean")
            return CommandResult(stdout="\n".join(lines), stderr="", exit_code=0)

        if self._staged:
            lines.append("Changes to be committed:")
            lines.append("  (use \"git restore --staged <file>...\" to unstage)")
            for f in sorted(self._staged):
                lines.append(f"\tmodified:   {f}")
            lines.append("")

        # To populate modified, we check if any files were staged or we just support tracking
        # If files in staging but no others:
        return CommandResult(stdout="\n".join(lines), stderr="", exit_code=0)

    def _cmd_add(self, args: list[str]) -> CommandResult:
        if not args:
            return CommandResult(stdout="", stderr="Nothing specified, nothing added.", exit_code=0)

        for path in args:
            if path == ".":
                # Add a dummy mock file if staging everything
                self._staged.add("main.py")
            else:
                self._staged.add(path)
        return CommandResult(stdout="", stderr="", exit_code=0)

    def _cmd_commit(self, args: list[str]) -> CommandResult:
        # Parse -m or --message flag
        message = ""
        for i, a in enumerate(args):
            if a in ("-m", "--message") and i + 1 < len(args):
                message = args[i + 1]
                break

        if not message:
            return CommandResult(stdout="", stderr="error: switch `m' requires a value", exit_code=1)

        if not self._staged:
            return CommandResult(
                stdout="On branch main\nnothing to commit, working tree clean",
                stderr="",
                exit_code=0,
            )

        # Generate a commit hash
        h = hashlib.sha256(f"{message}{time.time()}".encode("utf-8")).hexdigest()[:7]
        now = dt.datetime.now().strftime("%a %b %d %H:%M:%S %Y +0000")
        
        new_commit = {
            "hash": h,
            "author": "developer@example.com",
            "date": now,
            "message": message,
        }
        self._commits.insert(0, new_commit)
        
        num_files = len(self._staged)
        self._staged.clear()

        return CommandResult(
            stdout=f"[main {h}] {message}\n {num_files} file changed, 1 insertion(+)",
            stderr="",
            exit_code=0,
            state_changes={"git_commits_created": [new_commit]},
        )

    def _cmd_log(self, args: list[str]) -> CommandResult:
        del args
        if not self._commits:
            return CommandResult(stdout="No commits yet.", stderr="", exit_code=0)

        lines = []
        for c in self._commits:
            lines.append(f"commit {c['hash']}")
            lines.append(f"Author: {c['author']}")
            lines.append(f"Date:   {c['date']}")
            lines.append("")
            lines.append(f"    {c['message']}")
            lines.append("")

        return CommandResult(stdout="\n".join(lines), stderr="", exit_code=0)

    def _cmd_diff(self, args: list[str]) -> CommandResult:
        del args
        # Return a simple generic diff output if staged/modified
        if not self._staged:
            return CommandResult(stdout="", stderr="", exit_code=0)

        lines = []
        for f in self._staged:
            lines.append(f"diff --git a/{f} b/{f}")
            lines.append("index 0000000..1111111 100644")
            lines.append(f"--- a/{f}")
            lines.append(f"+++ b/{f}")
            lines.append("@@ -1,1 +1,2 @@")
            lines.append("-old line")
            lines.append("+new line")
        return CommandResult(stdout="\n".join(lines), stderr="", exit_code=0)
