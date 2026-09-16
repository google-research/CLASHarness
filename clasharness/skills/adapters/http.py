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

"""HTTP skill adapter."""

from __future__ import annotations

from clasharness.backend.http_backend import HttpBackend
from clasharness.skills.base import BackendSkillAdapter
from clasharness.skills.spec import CommandSpec, RealKind, SkillSpec


class HttpSkillAdapter(BackendSkillAdapter):
    SPEC = SkillSpec(
        name="curl",
        category="http",
        summary="HTTP requests via curl (mock synthesizes known APIs offline; real path hits network).",
        prefixes=("curl",),
        real_kind=RealKind.EXTERNAL_NETWORK,
        commands=(
            CommandSpec(
                "curl",
                "Perform an HTTP request (supports -d / --data-urlencode / -X).",
                args={
                    "url": {"type": "string"},
                    "-X": {"type": "string"},
                    "-d": {"type": "string"},
                    "--data-urlencode": {"type": "string"},
                },
                required=("url",),
            ),
        ),
        smoke=(
            "curl https://timeapi.io/api/Time/current/zone?timeZone=UTC",
        ),
    )

    def __init__(self, backend: HttpBackend) -> None:
        super().__init__(backend, prefixes=("curl",))
