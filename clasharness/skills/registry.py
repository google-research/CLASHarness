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

"""Skill registry for command-prefix routing."""

from __future__ import annotations

from clasharness.skills.base import Skill


class SkillRegistry:
    """Register and resolve skills by command prefix."""

    def __init__(self) -> None:
        self._prefix_to_skill: dict[str, Skill] = {}
        self._skills: list[Skill] = []

    def register(self, prefix: str, skill: Skill) -> None:
        if not prefix:
            raise ValueError("Prefix cannot be empty")
        existing = self._prefix_to_skill.get(prefix)
        if existing is not None and existing is not skill:
            raise ValueError(
                f"Prefix '{prefix}' already registered by {existing.__class__.__name__}"
            )
        self._prefix_to_skill[prefix] = skill
        if skill not in self._skills:
            self._skills.append(skill)

    def register_skill(self, skill: Skill) -> None:
        for prefix in skill.prefixes:
            self.register(prefix, skill)

    def resolve(self, prefix: str) -> Skill | None:
        return self._prefix_to_skill.get(prefix)

    def list_prefixes(self) -> list[str]:
        return sorted(self._prefix_to_skill)

    def iter_skills(self) -> list[Skill]:
        return list(self._skills)

    # ------------------------------------------------------------------ #
    # Unified-spec introspection / conformance                            #
    # ------------------------------------------------------------------ #
    def iter_specs(self) -> list:
        """Return the :class:`SkillSpec` of every registered skill (may be
        ``None`` for a skill that has not declared one yet)."""
        return [skill.get_spec() for skill in self._skills]

    def validate_specs(self, *, require_spec: bool = True) -> list[str]:
        """Validate that every registered skill has a consistent ``SkillSpec``.

        Checks, per skill: a spec exists, it is internally valid, and its
        declared ``prefixes`` match the skill's actual routing prefixes.
        Cross-skill: generated MCP tool names are globally unique.

        Returns a list of human-readable error strings (empty == all good) so
        callers/tests can assert on it without catching exceptions.
        """
        errors: list[str] = []
        seen_tools: dict[str, str] = {}
        for skill in self._skills:
            cls_name = skill.__class__.__name__
            spec = skill.get_spec()
            if spec is None:
                if require_spec:
                    errors.append(f"{cls_name}: missing SPEC")
                continue
            try:
                spec.validate()
            except ValueError as exc:
                errors.append(f"{cls_name}: {exc}")
                continue
            if set(spec.prefixes) != set(skill.prefixes):
                errors.append(
                    f"{cls_name}: spec.prefixes {spec.prefixes} != "
                    f"skill.prefixes {skill.prefixes}"
                )
            for tool in spec.to_mcp_tools():
                name = tool["name"]
                if name in seen_tools:
                    errors.append(
                        f"duplicate MCP tool name '{name}' "
                        f"({seen_tools[name]} & {cls_name})"
                    )
                seen_tools[name] = cls_name
        return errors
