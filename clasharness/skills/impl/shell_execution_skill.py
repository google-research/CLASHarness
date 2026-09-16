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

"""Shell execution skill: runs python/zip/sqlite commands in a hardened sandbox.

Tier-1 hardening (native, no container):
  * Subprocesses receive a SCRUBBED environment (no API keys / tokens / cloud
    credentials), so agent-controlled code cannot exfiltrate harness secrets.
  * Resource limits (CPU / file-size / open-files, optionally address space) are
    applied via ``preexec_fn`` to bound runaway processes.
  * Path-bearing arguments for archive/db/script commands are validated to stay
    inside the task workspace (blocks absolute paths and ``..`` escapes such as
    zip-slip / ``tar -C /etc``).
"""

from __future__ import annotations

import os
import pathlib
import shlex
import shutil
import subprocess

from clasharness.backend.base import CommandResult, REAL_ENV_FLAG
from clasharness.env_knobs import getenv_int
from clasharness.skills.base import Skill
from clasharness.skills.spec import CommandSpec, RealKind, SkillSpec

try:  # POSIX only; preexec_fn / rlimits are unavailable on Windows.
    import resource
except ImportError:  # pragma: no cover - non-POSIX fallback
    resource = None


_DEFAULT_TIMEOUT = 30

# Environment variables that must never reach an agent-controlled subprocess.
_SENSITIVE_ENV_PREFIXES = (
    "ANTHROPIC", "OPENAI", "VERTEX", "GOOGLE", "GCLOUD", "GCP", "GEMINI",
    "AWS", "AZURE", "SLACK", "TELEGRAM", "GITHUB", "GH_", "HF_", "HUGGING",
    "OPENCLAW_API", "CLASHARNESS_API",
)
_SENSITIVE_ENV_SUBSTRINGS = (
    "KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "CRED",
    "APIKEY", "API_KEY", "AUTH", "PRIVATE", "SESSION", "COOKIE", "BEARER",
)
# Variables that are always safe (and often required) for normal tool operation.
_ALLOWED_ENV_KEYS = frozenset({
    "PATH", "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "TZ", "TERM", "PWD",
    "SHELL", "USER", "LOGNAME", "HOSTNAME", "PYTHONHASHSEED",
})


def _is_sensitive_env_key(key: str) -> bool:
    upper = key.upper()
    if upper in _ALLOWED_ENV_KEYS:
        return False
    if any(upper.startswith(prefix) for prefix in _SENSITIVE_ENV_PREFIXES):
        return True
    return any(token in upper for token in _SENSITIVE_ENV_SUBSTRINGS)


class SandboxPathError(ValueError):
    """Raised when a command references a path outside the workspace sandbox."""


class ShellExecutionSkill(Skill):
    """Executes allowlisted terminal commands in a hardened local sandbox."""

    SPEC = SkillSpec(
        name="shell",
        category="compute",
        summary="Run allowlisted binaries (python3/bash/unzip/tar/sqlite3) in a hardened sandbox.",
        prefixes=("python3", "bash", "unzip", "tar", "sqlite3"),
        real_kind=RealKind.SUBPROCESS,
        commands=(
            CommandSpec("python3", "Run a Python interpreter invocation."),
            CommandSpec("bash", "Run a bash invocation."),
            CommandSpec("unzip", "Extract a zip archive inside the workspace."),
            CommandSpec("tar", "Create/extract a tar archive inside the workspace."),
            CommandSpec("sqlite3", "Run a sqlite3 invocation inside the workspace."),
        ),
        smoke=(
            "python3 -c print(1)",
            "bash -c true",
        ),
    )

    def __init__(self) -> None:
        super().__init__(prefixes=("python3", "bash", "unzip", "tar", "sqlite3"))
        self._real_mode = False
        self._workspace_dir: str | None = None

    def initialize(self, state_dir: str, env_vars: dict[str, str]) -> None:
        val = str(env_vars.get(REAL_ENV_FLAG, "")).lower()
        self._real_mode = val in {"1", "true", "yes", "on"}
        self._workspace_dir = str(pathlib.Path(state_dir) / "workspace")

    # ------------------------------------------------------------------ #
    # Sandbox helpers                                                       #
    # ------------------------------------------------------------------ #

    def _workspace_root(self) -> pathlib.Path:
        return pathlib.Path(self._workspace_dir or os.getcwd()).resolve()

    def _build_safe_env(self) -> dict[str, str]:
        """Return a scrubbed copy of the environment for the subprocess."""
        env: dict[str, str] = {}
        for key, value in os.environ.items():
            if _is_sensitive_env_key(key):
                continue
            env[key] = value
        workspace = str(self._workspace_root())
        # Pin HOME/TMPDIR into the workspace so dotfiles / temp writes stay local.
        env["HOME"] = workspace
        env["TMPDIR"] = workspace
        env.setdefault("PATH", os.environ.get("PATH", "/usr/bin:/bin"))
        env["OPENCLAW_SANDBOX"] = "1"
        return env

    def _rlimit_preexec(self):
        """Build a preexec callable that applies resource limits (POSIX only)."""
        res = resource
        if res is None:
            return None

        cpu_seconds = getenv_int("CLASHARNESS_SANDBOX_CPU_SECONDS", _DEFAULT_TIMEOUT)
        fsize_mb = getenv_int("CLASHARNESS_SANDBOX_FSIZE_MB", 256)
        nofile = getenv_int("CLASHARNESS_SANDBOX_NOFILE", 256)
        # Address-space cap is opt-in: too low a value breaks numpy/matplotlib.
        as_mb = getenv_int("CLASHARNESS_SANDBOX_AS_MB", 0)

        def _apply() -> None:  # pragma: no cover - runs in forked child
            if cpu_seconds > 0:
                try:
                    res.setrlimit(res.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
                except (ValueError, OSError):
                    pass
            if fsize_mb > 0:
                fbytes = fsize_mb * 1024 * 1024
                try:
                    res.setrlimit(res.RLIMIT_FSIZE, (fbytes, fbytes))
                except (ValueError, OSError):
                    pass
            if nofile > 0:
                try:
                    res.setrlimit(res.RLIMIT_NOFILE, (nofile, nofile))
                except (ValueError, OSError):
                    pass
            if as_mb > 0:
                abytes = as_mb * 1024 * 1024
                try:
                    res.setrlimit(res.RLIMIT_AS, (abytes, abytes))
                except (ValueError, OSError):
                    pass

        return _apply

    def _candidate_path_tokens(self, parts: list[str]) -> list[str]:
        """Return path-bearing tokens that must be confined to the workspace."""
        cmd = parts[0]
        rest = parts[1:]
        # Inline interpreter code (`python3 -c` / `bash -c`) is not a path; its
        # safety is handled by env-scrub + rlimits + safety_guard inline scan.
        if cmd in ("python3", "bash") and rest and rest[0] in ("-c", "-"):
            return []
        if cmd == "sqlite3":
            # Validate only the DB path (first non-flag token); skip the SQL query.
            for tok in rest:
                if not tok.startswith("-"):
                    return [tok]
            return []
        tokens: list[str] = []
        for tok in rest:
            if tok.startswith("-"):
                continue
            if tok.startswith("/") or "/" in tok or tok == ".." or tok.startswith(".."):
                tokens.append(tok)
        return tokens

    def _validate_paths(self, parts: list[str]) -> None:
        root = self._workspace_root()
        for tok in self._candidate_path_tokens(parts):
            candidate = pathlib.Path(tok)
            resolved = (candidate if candidate.is_absolute() else (root / tok)).resolve()
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise SandboxPathError(
                    f"path argument escapes the workspace sandbox: {tok!r}"
                ) from exc

    # ------------------------------------------------------------------ #
    # Skill interface                                                      #
    # ------------------------------------------------------------------ #

    def execute(self, command: str) -> CommandResult:
        parts = shlex.split(command.strip())
        if not parts or parts[0] not in self.prefixes:
            return CommandResult(
                stdout="", stderr=f"Command prefix not supported: {command}", exit_code=1
            )

        try:
            self._validate_paths(parts)
        except SandboxPathError as exc:
            return CommandResult(stdout="", stderr=f"Sandbox violation: {exc}", exit_code=1)

        # sqlite3 fallback: emulate via python3 when the binary is unavailable.
        if parts[0] == "sqlite3" and shutil.which("sqlite3") is None:
            if len(parts) < 3:
                return CommandResult(
                    stdout="", stderr="sqlite3 requires a db path and a query", exit_code=1
                )
            db_path = parts[1]
            query = parts[2]
            python_code = (
                "import sqlite3; "
                f"conn = sqlite3.connect({db_path!r}); "
                "cursor = conn.cursor(); "
                f"cursor.execute({query!r}); "
                "rows = cursor.fetchall(); "
                "[print('|'.join(str(x) for x in r)) for r in rows]; "
                "conn.close()"
            )
            parts = ["python3", "-c", python_code]

        try:
            res = subprocess.run(
                parts,
                cwd=self._workspace_dir,
                capture_output=True,
                text=True,
                timeout=_DEFAULT_TIMEOUT,
                env=self._build_safe_env(),
                preexec_fn=self._rlimit_preexec(),
                start_new_session=True,
            )
            return CommandResult(
                stdout=res.stdout,
                stderr=res.stderr,
                exit_code=res.returncode,
            )
        except Exception as e:  # noqa: BLE001 - surface as command failure
            # In mock mode, tolerate missing binaries / environments.
            if not self._real_mode:
                return CommandResult(
                    stdout=f"Mock execute (binary execution failed: {e}): {command}",
                    stderr="",
                    exit_code=0,
                )
            return CommandResult(stdout="", stderr=str(e), exit_code=1)

    def cleanup(self) -> None:
        pass
