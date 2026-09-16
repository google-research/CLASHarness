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

"""Safety guard to prevent agents from executing dangerous operations."""

from __future__ import annotations

import re
import shlex

from clasharness.tools.agent_tool_spec import allowed_prefixes


class SafetyViolation(Exception):
    """Raised when an agent attempts a blocked action."""


BLOCKED_PATTERNS = [
    re.compile(r"rm\s+-rf\s+/"),
    re.compile(r"rm\s+-rf\s+~"),
    re.compile(r"openclaw\s+reset\s+--hard"),
    re.compile(r"openclaw\s+uninstall"),
    re.compile(r"sudo\s+"),
    re.compile(r"chmod\s+777"),
    re.compile(r">\s*/dev/"),
    re.compile(r"mkfs"),
    re.compile(r"dd\s+if="),
    re.compile(r":\(\)\s*\{"),  # fork bomb
]

ALLOWED_COMMAND_PREFIXES = list(allowed_prefixes())

# Inline interpreter bodies (`python3 -c "..."`) bypass the standard `python`
# interface safety check, so they are scanned separately for escape vectors:
# command execution and network egress. File writes are permitted because the
# subprocess is already confined to the workspace by ShellExecutionSkill.
_INLINE_PYTHON_PATTERNS = [
    re.compile(r"os\.system\("),
    re.compile(r"os\.popen\("),
    re.compile(r"os\.exec[lv]"),
    re.compile(r"subprocess\."),
    re.compile(r"pty\.spawn\("),
    re.compile(r"\b__import__\("),
    re.compile(r"\beval\("),
    re.compile(r"\bexec\("),
    re.compile(r"shutil\.rmtree\("),
    re.compile(r"\bimport\s+(socket|requests|ftplib|smtplib|telnetlib)\b"),
    re.compile(r"\b(urllib|http\.client)\b"),
]


def _extract_inline_code(command: str) -> tuple[str | None, str | None]:
    """Return (interpreter, code) when a command runs inline `-c` code."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None, None
    if not tokens:
        return None, None
    interpreter = tokens[0]
    if interpreter not in ("python3", "python", "bash", "sh"):
        return None, None
    if "-c" in tokens:
        idx = tokens.index("-c")
        if idx + 1 < len(tokens):
            return interpreter, tokens[idx + 1]
    return interpreter, None


def check_inline_python_safety(code: str) -> None:
    """Scan inline python code for command-execution / network-egress vectors."""
    for pattern in _INLINE_PYTHON_PATTERNS:
        if pattern.search(code):
            raise SafetyViolation(
                f"Blocked dangerous inline python pattern: {pattern.pattern}"
            )


def check_command_safety(command: str, strict: bool = True) -> None:
    """Validate that a CLI command is safe to execute.

    Raises SafetyViolation if the command matches a blocked pattern.
    When strict=True (default), also enforces the allowed-prefix allowlist.
    When strict=False, only blocked patterns are checked (for real backends).
    """
    command = command.strip()

    for pattern in BLOCKED_PATTERNS:
        if pattern.search(command):
            raise SafetyViolation(
                f"Blocked dangerous command pattern: {pattern.pattern}"
            )

    # Inline interpreter bodies are scanned regardless of strict mode: the `-c`
    # body never reaches the stricter `python` interface check.
    interpreter, inline_code = _extract_inline_code(command)
    if inline_code and interpreter in ("python3", "python"):
        check_inline_python_safety(inline_code)

    if strict and not any(command.startswith(prefix) for prefix in ALLOWED_COMMAND_PREFIXES):
        raise SafetyViolation(
            f"Command not in allowed list. Must start with one of: "
            f"{', '.join(p for p in ALLOWED_COMMAND_PREFIXES[:10])}..."
        )


def check_python_safety(code: str) -> None:
    """Validate that Python code is safe to execute.

    Raises SafetyViolation for dangerous operations.
    """
    dangerous_patterns = [
        re.compile(r"os\.system\("),
        re.compile(r"subprocess\.(run|call|Popen|check_output)\("),
        re.compile(r"shutil\.rmtree\("),
        re.compile(r"os\.remove\("),
        re.compile(r"os\.unlink\("),
        re.compile(r"__import__\("),
        re.compile(r"exec\("),
        re.compile(r"eval\("),
        re.compile(r"open\(.*(w|a)\)"),
    ]

    for pattern in dangerous_patterns:
        if pattern.search(code):
            raise SafetyViolation(
                f"Blocked dangerous Python pattern: {pattern.pattern}"
            )
