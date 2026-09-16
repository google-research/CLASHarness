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

"""Tests for the additive skill extension interface (Phase 2).

Covers ``@register_skill``, :class:`SkillContext`, ``from_context``, and the
auto-load + registration path in ``MultiAppBackend``. Every registration is
wrapped in :func:`extension_registry_sandbox` so no global state leaks between
tests sharing the process.
"""

from __future__ import annotations

from clasharness.backend.base import CommandResult
from clasharness.backend.multi_app_backend import MultiAppBackend
from clasharness.skills import (
    CommandSpec,
    RealKind,
    Skill,
    SkillContext,
    SkillSpec,
    build_extension_skills,
    extension_registry_sandbox,
    register_skill,
    registered_extension_skills,
)


def _make_ping_cls():
    """A minimal, dependency-free, spec-carrying example extension skill."""

    class PingSkill(Skill):
        SPEC = SkillSpec(
            name="ping",
            category="compute",
            summary="Example extension skill.",
            prefixes=("ping",),
            real_kind=RealKind.PURE_COMPUTE,
            commands=(CommandSpec(name="ping", summary="Return pong."),),
            smoke=("ping",),
        )

        def __init__(self) -> None:
            super().__init__(prefixes=("ping",))
            self.initialized = False

        def initialize(self, state_dir: str, env_vars: dict) -> None:
            self.initialized = True

        def execute(self, command: str) -> CommandResult:
            return CommandResult(stdout="pong", stderr="", exit_code=0)

        def cleanup(self) -> None:
            pass

    return PingSkill


# --------------------------------------------------------------------------- #
# Registry / build                                                             #
# --------------------------------------------------------------------------- #
def test_no_extensions_registered_by_default():
    # Outside any sandbox registration, the core arena ships zero extensions.
    assert registered_extension_skills() == []
    be = MultiAppBackend()
    assert be._extension_errors == []


def test_register_and_build_and_sandbox_restores():
    ping_cls = _make_ping_cls()
    with extension_registry_sandbox():
        register_skill(ping_cls)
        assert ping_cls in registered_extension_skills()
        built = build_extension_skills(SkillContext())
        assert len(built) == 1
        assert isinstance(built[0], ping_cls)
    # sandbox restored the registry
    assert ping_cls not in registered_extension_skills()


def test_decorator_form_registers():
    with extension_registry_sandbox():

        @register_skill
        class DecoSkill(_make_ping_cls()):  # subclass reuses ping behavior
            pass

        assert DecoSkill in registered_extension_skills()


def test_register_rejects_non_skill():
    with extension_registry_sandbox():
        try:
            register_skill(object)  # type: ignore[arg-type]
            raised = False
        except TypeError:
            raised = True
        assert raised


# --------------------------------------------------------------------------- #
# from_context / factory                                                       #
# --------------------------------------------------------------------------- #
def test_from_context_default_builds_dependency_free():
    ping_cls = _make_ping_cls()
    skill = ping_cls.from_context(SkillContext())
    assert isinstance(skill, ping_cls)


def test_factory_receives_context():
    seen = {}

    class NeedsDeps(Skill):
        def __init__(self, file_backend=None) -> None:
            super().__init__(prefixes=("needsdeps",))
            self.file_backend = file_backend

        def initialize(self, state_dir, env_vars):
            pass

        def execute(self, command):
            return CommandResult(stdout="", stderr="", exit_code=0)

        def cleanup(self):
            pass

    sentinel = object()
    with extension_registry_sandbox():
        register_skill(
            NeedsDeps,
            factory=lambda ctx: NeedsDeps(file_backend=ctx.file_backend),
        )
        built = build_extension_skills(SkillContext(file_backend=sentinel))
        assert built[0].file_backend is sentinel
        seen["ok"] = True
    assert seen["ok"]


def test_build_error_is_collected_not_raised():
    class Broken(Skill):
        def __init__(self):
            super().__init__(prefixes=("broken",))

        @classmethod
        def from_context(cls, context):
            raise RuntimeError("boom")

        def initialize(self, state_dir, env_vars):
            pass

        def execute(self, command):
            return CommandResult(stdout="", stderr="", exit_code=0)

        def cleanup(self):
            pass

    errors = []
    with extension_registry_sandbox():
        register_skill(Broken)
        built = build_extension_skills(
            SkillContext(), on_error=lambda cls, exc: errors.append((cls, exc))
        )
    assert built == []
    assert errors and errors[0][0] is Broken


# --------------------------------------------------------------------------- #
# MultiAppBackend integration                                                  #
# --------------------------------------------------------------------------- #
def test_backend_autoloads_and_routes_extension(tmp_path):
    ping_cls = _make_ping_cls()
    with extension_registry_sandbox():
        register_skill(ping_cls)
        be = MultiAppBackend()
        be.initialize(str(tmp_path), {})
        assert be._extension_errors == []
        # the extension prefix now resolves and routes
        assert "ping" in be._skill_registry.list_prefixes()
        res = be.execute_cli("ping hello")
        assert res.exit_code == 0
        assert "pong" in res.stdout


def test_extension_spec_passes_conformance(tmp_path):
    ping_cls = _make_ping_cls()
    with extension_registry_sandbox():
        register_skill(ping_cls)
        be = MultiAppBackend()
        # The extension's SPEC participates in registry conformance with no errors.
        assert be._skill_registry.validate_specs() == []


def test_backend_collects_prefix_collision(tmp_path):
    # An extension that reuses a core prefix must not crash the backend; the
    # collision is recorded and the core skill keeps that prefix.
    core = MultiAppBackend()
    core_prefix = core._skill_registry.list_prefixes()[0]

    class Collide(Skill):
        def __init__(self, _p=core_prefix):
            super().__init__(prefixes=(_p,))

        def initialize(self, state_dir, env_vars):
            pass

        def execute(self, command):
            return CommandResult(stdout="collide", stderr="", exit_code=0)

        def cleanup(self):
            pass

    with extension_registry_sandbox():
        register_skill(Collide)
        be = MultiAppBackend()  # must not raise
        assert any("register failed" in e for e in be._extension_errors)
        # core skill (not the extension) still owns the prefix
        owner = be._skill_registry.resolve(core_prefix)
        assert type(owner).__name__ != "Collide"


# --------------------------------------------------------------------------- #
# Doc template guard: the GreetSkill in docs/mcp-extension-guide.md must work  #
# --------------------------------------------------------------------------- #
def _make_greet_cls():
    """Faithful reproduction of the copy-paste template in the extension guide.

    Kept in sync with ``docs/mcp-extension-guide.md`` so the documented
    template can't rot.
    """

    class GreetSkill(Skill):
        SPEC = SkillSpec(
            name="greet",
            category="compute",
            summary="Greet a person by name.",
            prefixes=("greet",),
            real_kind=RealKind.PURE_COMPUTE,
            supports_mock=True,
            supports_real=True,
            commands=(
                CommandSpec(
                    name="greet hello",
                    summary="Print a greeting.",
                    args={"name": {"type": "string"}},
                    required=("name",),
                    emits_effects=(),
                ),
            ),
            smoke=("greet hello world",),
        )

        _snapshot_attrs = ("_greeted",)

        def __init__(self) -> None:
            super().__init__(prefixes=("greet",))
            self._greeted: list[str] = []
            self._real = False

        def initialize(self, state_dir: str, env_vars: dict) -> None:
            from clasharness.backend.base import REAL_ENV_FLAG

            self._real = env_vars.get(REAL_ENV_FLAG) == "1"
            self._greeted = []

        def execute(self, command: str) -> CommandResult:
            parts = command.split()
            if len(parts) >= 3 and parts[1] == "hello":
                name = parts[2]
                self._greeted.append(name)
                return CommandResult(stdout=f"Hello, {name}!", stderr="", exit_code=0)
            return CommandResult(
                stdout="", stderr="usage: greet hello <name>", exit_code=1
            )

        def get_state(self) -> dict:
            return {"greeted": list(self._greeted)}

        def cleanup(self) -> None:
            pass

    return GreetSkill


def test_doc_template_greet_skill_smoke_and_conformance(tmp_path):
    greet_cls = _make_greet_cls()
    with extension_registry_sandbox():
        register_skill(greet_cls)
        be = MultiAppBackend()
        be.initialize(str(tmp_path), {})
        assert be._extension_errors == []
        # conformance clean with the template's SPEC in the registry
        assert be._skill_registry.validate_specs() == []
        # every declared smoke command runs green
        spec = greet_cls.get_spec()
        for cmd in spec.smoke:
            res = be.execute_cli(cmd)
            assert res.exit_code == 0, f"smoke failed: {cmd!r} -> {res.stderr}"
        assert "Hello, world!" in be.execute_cli("greet hello world").stdout
        # state contribution is surfaced
        assert "world" in be.get_state().get("greeted", [])
