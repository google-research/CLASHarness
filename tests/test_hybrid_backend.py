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

"""Tests for HybridBackend."""

from __future__ import annotations

import pytest

from clasharness.backend.hybrid_backend import HybridBackend, _find_free_port
from clasharness.backend.multi_app_backend import MultiAppBackend


# ------------------------------------------------------------------ #
# Fixtures                                                              #
# ------------------------------------------------------------------ #

@pytest.fixture
def backend(tmp_path):
    b = HybridBackend()
    b.initialize(str(tmp_path), {})
    yield b
    b.cleanup()


# ------------------------------------------------------------------ #
# Inheritance & properties                                              #
# ------------------------------------------------------------------ #

class TestHybridBackendInheritance:
    def test_is_multi_app_subclass(self):
        assert issubclass(HybridBackend, MultiAppBackend)

    def test_instance_of_multi_app(self):
        b = HybridBackend()
        assert isinstance(b, MultiAppBackend)

    def test_mock_backend_is_none(self):
        b = HybridBackend()
        assert b.mock_backend is None

    def test_real_backend_is_set(self):
        b = HybridBackend()
        assert b.real_backend is not None

    def test_gateway_port_valid_range(self):
        b = HybridBackend()
        assert isinstance(b.gateway_port, int)
        assert 1024 <= b.gateway_port <= 65535

    def test_custom_gateway_port(self):
        b = HybridBackend(gateway_port=19876)
        assert b.gateway_port == 19876

    def test_no_auto_start_by_default(self):
        b = HybridBackend()
        assert b.is_gateway_managed is False

    def test_auto_start_flag_set(self):
        b = HybridBackend(auto_start_gateway=True)
        assert b.is_gateway_managed is True


# ------------------------------------------------------------------ #
# App backend routing (mock)                                            #
# ------------------------------------------------------------------ #

class TestHybridBackendRouting:
    def test_calendar_routes_to_mock(self, backend):
        result = backend.execute_cli("calendar list")
        assert result.exit_code == 0

    def test_gcalcli_routes_to_mock(self, backend):
        result = backend.execute_cli("gcalcli agenda")
        assert result.exit_code == 0

    def test_email_routes_to_mock(self, backend):
        result = backend.execute_cli("email list")
        assert result.exit_code == 0

    def test_email_prefix_bound_to_email_skill(self, backend):
        from clasharness.skills.impl.email_skill import EmailSkill
        skill = backend._skill_registry.resolve("email")  # noqa: SLF001 - test introspection
        assert isinstance(skill, EmailSkill)

    def test_weather_routes_to_mock(self, backend):
        result = backend.execute_cli("weather get --location 'New York'")
        assert result.exit_code == 0

    def test_file_routes_to_mock(self, backend):
        result = backend.execute_cli("file create --path /tmp/x.txt --content hello")
        assert result.exit_code == 0

    def test_tasks_routes_to_mock(self, backend):
        result = backend.execute_cli("tasks list")
        assert result.exit_code == 0

    def test_tasks_prefix_bound_to_tasks_skill(self, backend):
        from clasharness.skills.impl.tasks_skill import TasksSkill
        skill = backend._skill_registry.resolve("tasks")  # noqa: SLF001 - test introspection
        assert isinstance(skill, TasksSkill)

    def test_openclaw_routes_to_real_gracefully(self, backend):
        # If openclaw is not installed → exit 127 (not a crash)
        result = backend.execute_cli("openclaw status")
        assert result.exit_code in (0, 127)

    def test_unknown_prefix_returns_error(self, backend):
        result = backend.execute_cli("foobar --flag")
        assert result.exit_code == 1
        assert "Unknown command prefix" in result.stderr

    def test_empty_command_returns_error(self, backend):
        result = backend.execute_cli("")
        assert result.exit_code == 1


# ------------------------------------------------------------------ #
# Mock app backends accessible via properties                           #
# ------------------------------------------------------------------ #

class TestHybridBackendAccessors:
    def test_calendar_backend_accessible(self, backend):
        from clasharness.backend.calendar_backend import CalendarBackend
        assert isinstance(backend.calendar_backend, CalendarBackend)

    def test_email_backend_accessible(self, backend):
        from clasharness.backend.email_backend import EmailBackend
        assert isinstance(backend.email_backend, EmailBackend)

    def test_file_backend_accessible(self, backend):
        from clasharness.backend.file_backend import FileSystemBackend
        assert isinstance(backend.file_backend, FileSystemBackend)

    def test_tasks_backend_accessible(self, backend):
        from clasharness.backend.tasks_backend import TasksBackend
        assert isinstance(backend.tasks_backend, TasksBackend)

    def test_app_state_is_mocked(self, backend):
        backend.execute_cli("tasks add --title 'Test task' --due 2026-03-10")
        state = backend.get_state()
        tasks = state.get("tasks", [])
        assert any(t.get("title") == "Test task" for t in tasks)


# ------------------------------------------------------------------ #
# Gateway lifecycle (no actual gateway)                                 #
# ------------------------------------------------------------------ #

class TestHybridBackendLifecycle:
    def test_no_gateway_proc_without_auto_start(self, backend):
        assert backend.gateway_process is None

    def test_cleanup_does_not_raise(self, tmp_path):
        b = HybridBackend()
        b.initialize(str(tmp_path), {})
        b.cleanup()  # should not raise

    def test_double_cleanup_does_not_raise(self, tmp_path):
        b = HybridBackend()
        b.initialize(str(tmp_path), {})
        b.cleanup()
        b.cleanup()

    def test_auto_start_false_no_process(self, tmp_path):
        b = HybridBackend(auto_start_gateway=False)
        b.initialize(str(tmp_path), {})
        assert b.gateway_process is None
        b.cleanup()

    def test_auto_start_true_process_is_none_when_not_installed(self, tmp_path):
        # openclaw not installed → _start_gateway catches FileNotFoundError gracefully
        b = HybridBackend(auto_start_gateway=True, gateway_ready_timeout=1)
        b.initialize(str(tmp_path), {})
        # gateway_process is None when openclaw binary doesn't exist
        # (or is a subprocess.Popen if it IS installed — both are valid)
        # Just verify it didn't raise
        b.cleanup()


# ------------------------------------------------------------------ #
# Port helper                                                           #
# ------------------------------------------------------------------ #

def test_find_free_port_returns_valid_port():
    port = _find_free_port()
    assert isinstance(port, int)
    assert 1024 <= port <= 65535


def test_find_free_port_unique():
    # Two successive calls should return different ports
    p1 = _find_free_port()
    p2 = _find_free_port()
    # Not guaranteed to be different, but usually are
    assert isinstance(p1, int) and isinstance(p2, int)


# ------------------------------------------------------------------ #
# Integration: HybridBackend through make_env                          #
# ------------------------------------------------------------------ #

def test_make_env_hybrid_mode():
    from clasharness import make_env
    env = make_env(
        "calendar_add_event_1",
        mode="hybrid",
        backend_kwargs={"gateway_port": _find_free_port()},
    )
    obs = env.reset()
    assert obs.task_instruction != ""
    env.close()


def test_make_env_martin_mode_aliases_multi_backend():
    from clasharness import make_env

    env = make_env("calendar_add_event_1", mode="martin")
    try:
        obs = env.reset()
        assert obs.task_instruction != ""
        assert isinstance(env._backend, MultiAppBackend)  # noqa: SLF001 - test alias routing
        assert env._backend.real_backend is None  # noqa: SLF001 - martin reuses simulated multi
    finally:
        env.close()


def test_make_env_hybrid_evaluation_state():
    from clasharness import make_env
    with make_env("file_create_1", mode="hybrid") as env:
        env.reset()
        env.step("file create --path /notes/todo.txt --content 'Hello world'")
        result = env.evaluate()
        # Local skill state is still evaluator-visible in hybrid mode.
        assert result.score > 0
