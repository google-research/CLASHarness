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

"""Extension interface for user-contributed skills / MCP tools.

Additive plugin layer on top of the core skill registry. The built-in skills
remain wired explicitly in :class:`~clasharness.backend.multi_app_backend.MultiAppBackend`;
*additional* skills can be contributed without editing backend code by:

1. subclassing :class:`~clasharness.skills.base.Skill` (or ``StatefulProtocolSkill``)
   and attaching a ``SPEC`` (see :mod:`clasharness.skills.spec`),
2. decorating the class with :func:`register_skill`,
3. importing the module so the decorator runs — drop it into
   ``clasharness/skills/contrib`` (auto-loaded by the backend) or import it
   yourself before constructing the environment.

At backend construction, :func:`build_extension_skills` instantiates every
registered extension skill from a :class:`SkillContext` (which exposes the
shared, construction-time building blocks — file backend, message skill,
real-mode flags, …) and registers it alongside the core skills.

See ``docs/mcp-extension-guide.md`` for the full guide and a copy-paste
template.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from clasharness.skills.base import Skill

#: A factory turns a :class:`SkillContext` into a ready-to-register skill.
SkillFactory = Callable[["SkillContext"], Skill]


@dataclass
class SkillContext:
    """Construction-time shared dependencies handed to extension skills.

    Only construction-time collaborators live here. Per-episode data
    (``state_dir`` / ``env_vars``) still flows through the normal
    :meth:`~clasharness.skills.base.Skill.initialize` call the runtime makes
    on every skill, so extension skills never need to reach for it here.
    """

    real_apps: bool = False
    real_openclaw: bool = False
    strict_online_data: bool = True
    file_backend: Any = None
    message_skill: Any = None
    asset_backend: Any = None
    http_backend: Any = None
    agent_skill: Any = None
    channel_skill: Any = None
    #: Free-form extras for forward-compatibility / custom wiring.
    extras: dict[str, Any] = field(default_factory=dict)


# (class, optional factory) pairs, in registration order.
_EXTENSION_SKILLS: list[tuple[type[Skill], "SkillFactory | None"]] = []


def register_skill(
    cls: type[Skill] | None = None,
    *,
    factory: "SkillFactory | None" = None,
):
    """Class decorator that registers an extension skill.

    Usage::

        @register_skill
        class MySkill(Skill):
            ...

        # or, when construction needs the shared context:
        @register_skill(factory=lambda ctx: MySkill(file_backend=ctx.file_backend))
        class MySkill(Skill):
            ...

    ``factory`` (when given) takes precedence over
    :meth:`~clasharness.skills.base.Skill.from_context`. Re-registering the
    same class is a no-op, so re-importing a module is safe.
    """

    def deco(klass: type[Skill]) -> type[Skill]:
        if not (isinstance(klass, type) and issubclass(klass, Skill)):
            raise TypeError("register_skill expects a Skill subclass")
        for existing, _ in _EXTENSION_SKILLS:
            if existing is klass:
                return klass  # idempotent: re-import shouldn't double-register
        _EXTENSION_SKILLS.append((klass, factory))
        return klass

    return deco(cls) if cls is not None else deco


def registered_extension_skills() -> list[type[Skill]]:
    """Return the currently-registered extension skill classes (in order)."""
    return [cls for cls, _ in _EXTENSION_SKILLS]


def build_extension_skills(
    context: SkillContext,
    *,
    on_error: "Callable[[type[Skill], Exception], None] | None" = None,
) -> list[Skill]:
    """Instantiate every registered extension skill from ``context``.

    Each skill is built via its registered ``factory`` if one was supplied,
    otherwise via :meth:`~clasharness.skills.base.Skill.from_context`. If a
    skill fails to build and ``on_error`` is provided, it is called with
    ``(cls, exc)`` and that skill is skipped (so one broken extension cannot
    take down the others); otherwise the exception propagates.
    """
    skills: list[Skill] = []
    for cls, factory in _EXTENSION_SKILLS:
        try:
            skill = factory(context) if factory is not None else cls.from_context(context)
        except Exception as exc:  # noqa: BLE001 - surfaced via on_error
            if on_error is None:
                raise
            on_error(cls, exc)
            continue
        skills.append(skill)
    return skills


def clear_extension_skills() -> None:
    """Remove all registered extension skills (test hygiene)."""
    _EXTENSION_SKILLS.clear()


@contextmanager
def extension_registry_sandbox() -> Iterator[None]:
    """Save/restore the extension registry around a block.

    Lets a test register throw-away skills (including via the decorator on a
    class defined inside the block) without leaking global state into other
    tests sharing the process.
    """
    saved = list(_EXTENSION_SKILLS)
    try:
        yield
    finally:
        _EXTENSION_SKILLS[:] = saved
