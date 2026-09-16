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

import pathlib
import shutil

import pytest

from clasharness.scenario import vlm_asset_generators
from clasharness.skills.impl.shell_execution_skill import (
    ShellExecutionSkill,
    _is_sensitive_env_key,
)
from clasharness.utils.safety_guard import SafetyViolation, check_command_safety

_HAS_BASH = shutil.which("bash") is not None

def test_sandbox_blocked_sys_import():
    # sys is no longer allowed
    code = "import sys\nprint(sys.argv)"
    with pytest.raises(ValueError, match="import is not allowed: sys"):
        vlm_asset_generators._validate_render_code(code)

def test_sandbox_blocked_sys_modules_bypass():
    # Even if they try to bypass, accessing modules or os as attribute is blocked
    code1 = "import matplotlib\nmatplotlib.os.system('echo BYPASS')"
    with pytest.raises(ValueError, match="Access to attribute 'os' is blocked"):
        vlm_asset_generators._validate_render_code(code1)

    code2 = "import matplotlib\nmatplotlib.sys.modules['os'].system('echo BYPASS')"
    # AST walk visits outer attribute 'modules' before inner 'sys'
    with pytest.raises(ValueError, match="Access to attribute '(sys|modules)' is blocked"):
        vlm_asset_generators._validate_render_code(code2)

def test_sandbox_blocked_pandas_write():
    code = "import pandas as pd\ndf = pd.DataFrame()\ndf.to_csv('escaped.csv')"
    with pytest.raises(ValueError, match="File write method 'to_csv' is blocked"):
        vlm_asset_generators._validate_render_code(code)

def test_sandbox_allowed_output_path():
    code = """
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([1, 2], [3, 4])
plt.savefig(OUTPUT_PATH)
"""
    # Should pass validation
    vlm_asset_generators._validate_render_code(code)


# --------------------------------------------------------------------------- #
# Agent-facing shell execution hardening (Tier-1)                              #
# --------------------------------------------------------------------------- #


def _make_shell_skill(tmp_path) -> ShellExecutionSkill:
    skill = ShellExecutionSkill()
    state_dir = str(tmp_path)
    (pathlib.Path(state_dir) / "workspace").mkdir(parents=True, exist_ok=True)
    skill.initialize(state_dir, {})
    return skill


def test_is_sensitive_env_key_classification():
    assert _is_sensitive_env_key("ANTHROPIC_API_KEY")
    assert _is_sensitive_env_key("MY_SECRET_TOKEN")
    assert _is_sensitive_env_key("GOOGLE_APPLICATION_CREDENTIALS")
    assert _is_sensitive_env_key("SLACK_WEBHOOK_URL")
    # Safe operational vars must be preserved.
    assert not _is_sensitive_env_key("PATH")
    assert not _is_sensitive_env_key("LANG")
    assert not _is_sensitive_env_key("TZ")


@pytest.mark.skipif(not _HAS_BASH, reason="bash not available")
def test_shell_env_scrub_hides_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "supersecret-key-value")
    monkeypatch.setenv("MY_AUTH_TOKEN", "tok-12345")
    skill = _make_shell_skill(tmp_path)
    res = skill.execute('bash -c "echo K=$ANTHROPIC_API_KEY T=$MY_AUTH_TOKEN"')
    assert res.exit_code == 0
    assert "supersecret-key-value" not in res.stdout
    assert "tok-12345" not in res.stdout


@pytest.mark.skipif(not _HAS_BASH, reason="bash not available")
def test_shell_env_scrub_keeps_path(tmp_path):
    skill = _make_shell_skill(tmp_path)
    res = skill.execute('bash -c "echo $PATH"')
    assert res.stdout.strip() != ""


@pytest.mark.skipif(not _HAS_BASH, reason="bash not available")
def test_shell_rlimit_cpu_applied(tmp_path, monkeypatch):
    import resource as _resource  # noqa: F401 - ensure POSIX rlimit support

    monkeypatch.setenv("CLASHARNESS_SANDBOX_CPU_SECONDS", "17")
    skill = _make_shell_skill(tmp_path)
    res = skill.execute('bash -c "ulimit -t"')
    assert res.stdout.strip() == "17"


@pytest.mark.skipif(not _HAS_BASH, reason="bash not available")
def test_shell_workspace_write_allowed(tmp_path):
    skill = _make_shell_skill(tmp_path)
    res = skill.execute('bash -c "echo hi > note.txt && cat note.txt"')
    assert res.exit_code == 0
    assert "hi" in res.stdout
    assert (pathlib.Path(skill._workspace_dir) / "note.txt").exists()


def test_shell_path_escape_absolute_blocked(tmp_path):
    skill = _make_shell_skill(tmp_path)
    res = skill.execute("tar -xf archive.tar -C /etc")
    assert res.exit_code == 1
    assert "Sandbox violation" in res.stderr


def test_shell_path_escape_dotdot_blocked(tmp_path):
    skill = _make_shell_skill(tmp_path)
    res = skill.execute("unzip ../../secret.zip")
    assert res.exit_code == 1
    assert "Sandbox violation" in res.stderr


def test_shell_sqlite_absolute_db_blocked(tmp_path):
    skill = _make_shell_skill(tmp_path)
    res = skill.execute('sqlite3 /etc/passwd "SELECT 1"')
    assert res.exit_code == 1
    assert "Sandbox violation" in res.stderr


def test_inline_python_blocks_os_system():
    with pytest.raises(SafetyViolation):
        check_command_safety("python3 -c \"import os; os.system('id')\"", strict=True)


def test_inline_python_blocks_subprocess():
    with pytest.raises(SafetyViolation):
        check_command_safety(
            "python3 -c \"import subprocess; subprocess.run(['ls'])\"", strict=True
        )


def test_inline_python_blocks_network_import():
    with pytest.raises(SafetyViolation):
        check_command_safety("python3 -c \"import socket; socket.socket()\"", strict=True)


def test_inline_python_allows_file_write():
    # Writing a file inline is permitted; the subprocess is workspace-confined.
    check_command_safety("python3 -c \"open('out.txt','w').write('x')\"", strict=True)
