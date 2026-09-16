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

"""Layer-1 real-path tests for the internal (hermetic) skills.

Phase 3 context
---------------
Each skill has a *mock* path (deterministic simulation, the eval default) and a
*real* path. Only a subset of skills actually **branch** between the two based on
the canonical ``CLASHARNESS_REAL`` flag (``REAL_ENV_FLAG``):

  * ``git`` / ``shell`` (RealKind.SUBPROCESS) and ``grep`` / ``jq``
    (RealKind.PURE_COMPUTE) shell out to a host binary on their real path.
  * The remaining "hermetic" skills (file, asset, agent, channel, deploy, lock,
    system) are *real == mock by construction* — they do not read the flag and
    run the same code either way, so their existing mock/conformance tests already
    cover the real path. We therefore only add dedicated real-path tests where a
    genuine branch exists.

Part A of Phase 3 fixed a bug: ``git``/``shell``/``grep``/``jq`` used to read a
dead flag (``OPENCLAW_REAL_MODE``) that was never set by any backend, so their
real paths were unreachable in ``mode="real"``. They now read the canonical
``CLASHARNESS_REAL`` that the environment actually threads.

Test layers here
----------------
  * Flag wiring (hermetic, no host binaries): ``mode="real"`` flips the toggles
    on; ``mode="multi"`` (full mock) leaves them off. Regression guard for Part A.
  * Cutover guards: the skills key off ``REAL_ENV_FLAG``; the retired
    ``OPENCLAW_REAL_MODE`` no longer activates the real path.
  * Real subprocess behavior (skip when the host binary is absent): with the real
    flag set, git/grep/jq actually shell out and differ from the mock path.

Skips use ``pytest.skip()`` at call time so they behave under a bare-``unittest``
runner as well as pytest (``@pytest.mark.skipif`` is a no-op without pytest, and
``unittest`` skip raises an uncaught error there).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile

import pytest

from clasharness.backend.base import REAL_ENV_FLAG
from clasharness.backend.file_backend import FileSystemBackend
from clasharness.core.environment import make_env
from clasharness.skills.impl.git_skill import GitSkill
from clasharness.skills.impl.grep_skill import GrepSkill
from clasharness.skills.impl.jq_skill import JqSkill
from clasharness.skills.impl.shell_execution_skill import ShellExecutionSkill

# A task with workspace support; also used by tests/test_mcp_categorized_tools.py.
_TASK_ID = "calendar_add_event_1"

_LEGACY_DEAD_FLAG = "OPENCLAW_REAL_MODE"  # retired in Phase 3 Part A


def _backend_for(mode: str):
    """Build and reset an env, returning its (MultiAppBackend) backend."""
    env = make_env(_TASK_ID, mode=mode)
    env.reset()
    return env._backend


def _init_skill(skill, real: bool):
    """Initialize a skill in a fresh temp state dir, in real or mock mode."""
    state_dir = tempfile.mkdtemp(prefix="realpath_" if real else "mockpath_")
    skill.initialize(state_dir, {REAL_ENV_FLAG: "1"} if real else {})
    return skill


# --------------------------------------------------------------------------- #
# Part A regression: flag wiring end-to-end (hermetic, no host binaries)        #
# --------------------------------------------------------------------------- #
def test_real_mode_enables_internal_subprocess_toggles():
    """mode="real" must thread CLASHARNESS_REAL through to every branching skill.

    This is the direct regression guard for the Part A bug: pre-fix these toggles
    stayed False in real mode because the skills read a never-set legacy flag.
    """
    be = _backend_for("real")
    assert be._git_skill._real_git is True
    assert be._shell_exec_skill._real_mode is True
    assert be._grep_skill._real_mode is True
    assert be._jq_skill._real_mode is True


def test_multi_mode_leaves_internal_toggles_off():
    """Full-mock ("multi") mode must leave every real toggle off."""
    be = _backend_for("multi")
    assert be._git_skill._real_git is False
    assert be._shell_exec_skill._real_mode is False
    assert be._grep_skill._real_mode is False
    assert be._jq_skill._real_mode is False


def test_multi_mode_git_uses_mock_emulation():
    """Behavioral corollary: in mock mode git returns the emulated output."""
    be = _backend_for("multi")
    res = be._git_skill.execute("git status")
    assert res.exit_code == 0
    assert "On branch main" in res.stdout  # signature of the mock emulator


# --------------------------------------------------------------------------- #
# Cutover guards (unit level): skills key off the canonical flag only           #
# --------------------------------------------------------------------------- #
def test_skills_read_canonical_flag():
    fb = FileSystemBackend()
    git = _init_skill(GitSkill(), real=True)
    grep = _init_skill(GrepSkill(file_backend=fb), real=True)
    jq = _init_skill(JqSkill(file_backend=FileSystemBackend()), real=True)
    shell = _init_skill(ShellExecutionSkill(), real=True)
    assert git._real_git is True
    assert grep._real_mode is True
    assert jq._real_mode is True
    assert shell._real_mode is True


def test_skills_default_to_mock_without_flag():
    fb = FileSystemBackend()
    git = _init_skill(GitSkill(), real=False)
    grep = _init_skill(GrepSkill(file_backend=fb), real=False)
    jq = _init_skill(JqSkill(file_backend=FileSystemBackend()), real=False)
    shell = _init_skill(ShellExecutionSkill(), real=False)
    assert git._real_git is False
    assert grep._real_mode is False
    assert jq._real_mode is False
    assert shell._real_mode is False


def test_retired_legacy_flag_does_not_enable_real():
    """Option A (clean cutover): OPENCLAW_REAL_MODE must no longer do anything."""
    git = GitSkill()
    git.initialize(tempfile.mkdtemp(prefix="legacy_"), {_LEGACY_DEAD_FLAG: "1"})
    assert git._real_git is False


# --------------------------------------------------------------------------- #
# Real subprocess behavior (skip when the host binary is absent)                #
# --------------------------------------------------------------------------- #
def test_git_real_version_differs_from_mock():
    if shutil.which("git") is None:
        pytest.skip("git binary not available on host")
    real = _init_skill(GitSkill(), real=True)
    res = real.execute("git --version")
    assert res.exit_code == 0
    assert "git version" in res.stdout.lower()

    # The mock emulator has no real git: '--version' is not a known subcommand.
    mock = _init_skill(GitSkill(), real=False)
    mres = mock.execute("git --version")
    assert mres.exit_code != 0
    assert "git version" not in mres.stdout.lower()


def test_grep_real_matches_on_real_file():
    if shutil.which("grep") is None:
        pytest.skip("grep binary not available on host")
    d = tempfile.mkdtemp(prefix="grep_")
    fpath = os.path.join(d, "data.txt")
    with open(fpath, "w", encoding="utf-8") as f:
        f.write("alpha line\nBETA needle here\ngamma line\n")
    grep = _init_skill(GrepSkill(file_backend=FileSystemBackend()), real=True)
    res = grep.execute(f"grep needle {fpath}")
    assert res.exit_code == 0
    assert "needle" in res.stdout


def test_jq_real_extracts_field():
    if shutil.which("jq") is None:
        pytest.skip("jq binary not available on host")
    d = tempfile.mkdtemp(prefix="jq_")
    fpath = os.path.join(d, "data.json")
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump({"name": "Avery", "role": "analyst"}, f)
    jq = _init_skill(JqSkill(file_backend=FileSystemBackend()), real=True)
    res = jq.execute(f"jq -r .role {fpath}")
    assert res.exit_code == 0
    assert "analyst" in res.stdout


if __name__ == "__main__":  # pragma: no cover - convenience for direct runs
    import sys

    sys.exit(0)
