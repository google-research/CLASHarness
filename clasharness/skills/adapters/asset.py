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

"""Asset skill adapter."""

from __future__ import annotations

from clasharness.backend.asset_backend import AssetBackend
from clasharness.skills.base import BackendSkillAdapter
from clasharness.skills.spec import CommandSpec, RealKind, SkillSpec


class AssetSkillAdapter(BackendSkillAdapter):
    SPEC = SkillSpec(
        name="asset",
        category="filesystem",
        summary="Read-only access to scenario-provided assets.",
        prefixes=("asset",),
        real_kind=RealKind.PURE_COMPUTE,
        commands=(
            CommandSpec("asset list", "List available assets."),
            CommandSpec(
                "asset info",
                "Show metadata for an asset.",
                args={"--id": {"type": "string"}},
            ),
            CommandSpec(
                "asset read-text",
                "Read a text asset's content.",
                args={"--id": {"type": "string"}},
                emits_effects=("assets_accessed",),
            ),
            CommandSpec(
                "asset describe",
                "Describe an asset.",
                args={"--id": {"type": "string"}},
            ),
            CommandSpec(
                "asset open",
                "Open an asset.",
                args={"--id": {"type": "string"}},
                emits_effects=("assets_opened",),
            ),
        ),
        smoke=(
            "asset list",
        ),
    )

    def __init__(self, backend: AssetBackend) -> None:
        super().__init__(backend, prefixes=("asset",))
