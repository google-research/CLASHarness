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

"""File system app backend — virtual in-memory file system."""

from __future__ import annotations
import os
import re
import shlex
import shutil

import posixpath
from pathlib import Path
from typing import Any

from clasharness.backend.base import BaseBackend, CommandResult


def _get_arg(args: list[str], flag: str, default: str | None = None) -> str | None:
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
    return default


def _get_arg_rest(args: list[str], flag: str, default: str | None = None) -> str | None:
    """Return an argument value, preserving unquoted multi-token tail content.

    LLM agents often emit `file create --path X --content long report text`
    without quoting the content. For content-bearing file commands, treating the
    remaining tokens as the value makes the CLI more robust while preserving the
    normal quoted form.
    """

    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            value_parts: list[str] = []
            for part in args[i + 1 :]:
                if part.startswith("--") and value_parts:
                    break
                value_parts.append(part)
            return " ".join(value_parts)
    return default


_CONTENT_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "'": "'"}


def _unescape_content(text: str) -> str:
    """Render common backslash escapes in agent-provided file content.

    Agents frequently encode newlines/tabs as literal ``\\n``/``\\t`` inside a
    single-line ``--content`` value (the dominant Claude style). Written
    verbatim, the backslash-``n`` pollutes the file body and every content
    check. Convert the common escapes in a single pass so a lone backslash
    before an unrecognized character (e.g. ``\\d``) is preserved unchanged, and
    ``\\\\`` collapses to a single literal backslash exactly once.
    """
    if not text or "\\" not in text:
        return text
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "\\" and i + 1 < n:
            repl = _CONTENT_ESCAPES.get(text[i + 1])
            if repl is not None:
                out.append(repl)
                i += 2
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _peel_content(command: str) -> tuple[str, str | None]:
    """Split a terminal free-text ``--content`` value off a file command.

    Returns ``(head, content)`` where ``head`` is the command with the
    ``--content`` flag and its value removed (safe to ``shlex.split``) and
    ``content`` is the literal value with a single surrounding quote pair
    removed, or ``content=None`` when there is no ``--content`` flag.

    ``--content`` is the terminal free-text argument of ``file create`` /
    ``file append`` (per AgentToolSpec), so its value is taken verbatim through
    end-of-string. This keeps embedded quotes, apostrophes, backticks and real
    or escaped newlines intact instead of letting ``shlex`` choke on them (the
    dominant multi-line-report failure mode). Escape rendering is deferred to
    ``_unescape_content`` so it happens exactly once.
    """
    match = re.search(r"(?:^|\s)--content(?:\s+|=)", command)
    if not match:
        return command, None
    head = command[: match.start()].rstrip()
    value = command[match.end() :].strip()
    if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
        return head, value[1:-1]
    return head, value


class FileSystemBackend(BaseBackend):
    """Mock virtual file system backend."""

    def __init__(self) -> None:
        self._files: dict[str, str] = {}  # path -> content
        self._initialized = False

    # ------------------------------------------------------------------ #
    # BaseBackend interface                                                 #
    # ------------------------------------------------------------------ #

    def initialize(self, state_dir: str, env_vars: dict[str, str]) -> None:
        self._files = {}
        self._workspace_dir = Path(env_vars.get("OPENCLAW_STATE_DIR", state_dir)) / "workspace"
        if self._workspace_dir.exists():
            for path in sorted(self._workspace_dir.rglob("*")):
                if not path.is_file():
                    continue
                rel_path = path.relative_to(self._workspace_dir).as_posix()
                self._files[_normalize(rel_path)] = path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
        self._initialized = True

    def execute_cli(self, command: str) -> CommandResult:
        # Peel a terminal free-text --content value off BEFORE tokenizing so
        # embedded quotes/newlines/backslashes in report bodies never reach
        # shlex (the dominant multi-line-report failure mode). The remaining
        # head is a clean, quote-balanced command.
        head, raw_content = _peel_content((command or "").strip())
        try:
            parts = shlex.split(head)
        except ValueError:
            # Defense-in-depth: recover any residual quote imbalance in the head
            # (e.g. an apostrophe in a --path) rather than hard-failing.
            from clasharness.tools.agent_tool_spec import (  # pylint: disable=g-import-not-at-top
                normalize_command_quoting,
            )
            try:
                parts = shlex.split(normalize_command_quoting(head))
            except ValueError as exc:
                return CommandResult(
                    stdout="", stderr=f"shell parse error: {exc}", exit_code=1
                )
        if not parts or parts[0] != "file":
            return CommandResult(stdout="", stderr="Not a file command", exit_code=1)

        if len(parts) < 2:
            return CommandResult(
                stdout="file <subcommand> [options]\n"
                       "Subcommands: create, read, delete, list, move, append",
                stderr="",
                exit_code=0,
            )

        sub = parts[1]
        args = parts[2:]
        content = _unescape_content(raw_content) if raw_content is not None else None

        # Full-content write synonyms all map to create (overwrite). These are
        # canonicalized to ``file create`` by ``openclaw_compat.rewrite_command``
        # before execution; handled here as defense-in-depth for any direct
        # ``execute_cli`` caller that bypasses the rewrite layer.
        if sub in ("create", "update", "write", "edit", "modify"):
            return self._cmd_create(args, content)
        if sub == "append":
            return self._cmd_append(args, content)

        handlers = {
            "read": self._cmd_read,
            "delete": self._cmd_delete,
            "list": self._cmd_list,
            "move": self._cmd_move,
        }
        handler = handlers.get(sub)
        if handler is None:
            return CommandResult(stdout="", stderr=f"Unknown file subcommand: {sub}", exit_code=1)
        return handler(args)

    def execute_python(self, code: str) -> CommandResult:
        return CommandResult(stdout="", stderr="Python interface not supported for FileSystemBackend", exit_code=1)

    def get_gateway_status(self) -> dict[str, Any] | None:
        return None

    def get_config(self) -> dict[str, Any]:
        return {}

    def cleanup(self) -> None:
        self._files = {}
        self._initialized = False

    def get_state(self) -> dict[str, Any]:
        return {"files": {k: v for k, v in self._files.items()}}

    def to_state(self) -> dict[str, Any]:
        """Snapshot the virtual file system for undo support."""
        return {"files": dict(self._files)}

    def from_state(self, data: dict[str, Any]) -> None:
        """Restore the virtual file system and re-sync the workspace on disk."""
        if not isinstance(data, dict):
            return
        files = data.get("files")
        if not isinstance(files, dict):
            return
        self._files = dict(files)
        self._sync_files_to_disk()

    def _sync_files_to_disk(self) -> None:
        """Make the on-disk workspace match the in-memory file map after a restore."""
        workspace = getattr(self, "_workspace_dir", None)
        if not workspace:
            return
        for path, content in self._files.items():
            disk_path = workspace / path.lstrip("/")
            disk_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                disk_path.write_text(content, encoding="utf-8")
            except OSError:
                pass
        # Remove workspace files that no longer exist in the snapshot.
        try:
            for disk_path in sorted(workspace.rglob("*")):
                if not disk_path.is_file():
                    continue
                rel = _normalize("/" + disk_path.relative_to(workspace).as_posix())
                if rel not in self._files:
                    try:
                        disk_path.unlink()
                    except OSError:
                        pass
        except OSError:
            pass

    # ------------------------------------------------------------------ #
    # Command handlers                                                      #
    # ------------------------------------------------------------------ #

    def _cmd_create(self, args: list[str], content: str | None = None) -> CommandResult:
        path = _get_arg(args, "--path")
        if content is None:
            # Direct caller (not via execute_cli's content peel): recover the
            # value from tokens and render escapes exactly once.
            content = _unescape_content(_get_arg_rest(args, "--content", "") or "")

        if not path:
            return CommandResult(stdout="", stderr="--path is required", exit_code=1)

        path = _normalize(path)
        self._files[path] = content or ""

        if hasattr(self, "_workspace_dir") and self._workspace_dir:
            disk_path = self._workspace_dir / path.lstrip("/")
            disk_path.parent.mkdir(parents=True, exist_ok=True)
            disk_path.write_text(content or "", encoding="utf-8")

        return CommandResult(
            stdout=f"File created: {path}",
            stderr="",
            exit_code=0,
            state_changes={"files_created": [{"path": path, "content": content or ""}]},
        )

    def _cmd_read(self, args: list[str]) -> CommandResult:
        path = _get_arg(args, "--path")

        if not path:
            return CommandResult(stdout="", stderr="--path is required", exit_code=1)

        path = _normalize(path)
        if path not in self._files:
            return CommandResult(
                stdout="", stderr=f"File not found: {path}", exit_code=1
            )

        return CommandResult(stdout=self._files[path], stderr="", exit_code=0)

    def _cmd_delete(self, args: list[str]) -> CommandResult:
        path = _get_arg(args, "--path")

        if not path:
            return CommandResult(stdout="", stderr="--path is required", exit_code=1)

        path = _normalize(path)
        if path not in self._files:
            return CommandResult(
                stdout="", stderr=f"File not found: {path}", exit_code=1
            )

        del self._files[path]
        if hasattr(self, "_workspace_dir") and self._workspace_dir:
            disk_path = self._workspace_dir / path.lstrip("/")
            if disk_path.exists():
                disk_path.unlink()

        return CommandResult(
            stdout=f"File deleted: {path}",
            stderr="",
            exit_code=0,
            state_changes={"files_deleted": [{"path": path}]},
        )

    def _cmd_list(self, args: list[str]) -> CommandResult:
        prefix = _get_arg(args, "--path", "/") or "/"
        prefix = _normalize(prefix)

        matches = sorted(
            p for p in self._files if p.startswith(prefix)
        )

        if not matches:
            return CommandResult(
                stdout=f"No files under {prefix}.", stderr="", exit_code=0
            )

        lines = [f"Files under {prefix}:"]
        for p in matches:
            size = len(self._files[p])
            lines.append(f"  {p}  ({size} bytes)")
        return CommandResult(stdout="\n".join(lines), stderr="", exit_code=0)

    def _cmd_move(self, args: list[str]) -> CommandResult:
        src = _get_arg(args, "--src")
        dst = _get_arg(args, "--dst")

        if not src:
            return CommandResult(stdout="", stderr="--src is required", exit_code=1)
        if not dst:
            return CommandResult(stdout="", stderr="--dst is required", exit_code=1)

        src = _normalize(src)
        dst = _normalize(dst)

        if src not in self._files:
            return CommandResult(stdout="", stderr=f"File not found: {src}", exit_code=1)

        self._files[dst] = self._files.pop(src)
        if hasattr(self, "_workspace_dir") and self._workspace_dir:
            src_disk = self._workspace_dir / src.lstrip("/")
            dst_disk = self._workspace_dir / dst.lstrip("/")
            if src_disk.exists():
                dst_disk.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src_disk), str(dst_disk))

        return CommandResult(
            stdout=f"File moved: {src} → {dst}",
            stderr="",
            exit_code=0,
        )

    def _cmd_append(self, args: list[str], content: str | None = None) -> CommandResult:
        path = _get_arg(args, "--path")
        if content is None:
            # Direct caller (not via execute_cli's content peel): recover the
            # value from tokens and render escapes exactly once.
            raw = _get_arg_rest(args, "--content")
            content = _unescape_content(raw) if raw is not None else None

        if not path:
            return CommandResult(stdout="", stderr="--path is required", exit_code=1)
        if content is None:
            return CommandResult(stdout="", stderr="--content is required", exit_code=1)

        path = _normalize(path)
        if path not in self._files:
            return CommandResult(stdout="", stderr=f"File not found: {path}", exit_code=1)

        self._files[path] += content
        if hasattr(self, "_workspace_dir") and self._workspace_dir:
            disk_path = self._workspace_dir / path.lstrip("/")
            with open(disk_path, "a", encoding="utf-8") as f:
                f.write(content)

        return CommandResult(
            stdout=f"Content appended to {path}.",
            stderr="",
            exit_code=0,
            # Emit a first-class effect so appended content is gradeable. Without
            # this, `file append` produced NO state_changes and its content was
            # invisible to the effect stream (only files_created/files_deleted
            # were tracked), so an agent that appended the decisive answer to a
            # report scored zero on content checks. ``full_content`` is the
            # authoritative post-append file body; the environment folds it into
            # files_created for content-based checkers.
            state_changes={
                "files_appended": [
                    {"path": path, "content": content, "full_content": self._files[path]}
                ]
            },
        )


def _normalize(path: str) -> str:
    """Normalize a virtual path (ensure leading /)."""
    if not path.startswith("/"):
        path = "/" + path
    return posixpath.normpath(path)
