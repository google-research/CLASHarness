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

"""Shared, byte-identical helpers for scenario generation/materialization.

P1.3 consolidation. Only helpers that were provably identical across scenario
modules are hoisted here (zero behavior change). Near-duplicate variants with
semantic differences (e.g. ``_write_text`` safe-path gating, ``_extract_anchor_ids``
ordering, ``_ensure_anchor_text`` wording) are intentionally left in their
original modules.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from clasharness.scenario.llm_client import ScenarioLLMClient


def _round_type(index: int) -> str:
    if index <= 7:
        return "observation"
    if index <= 14:
        return "reconciliation"
    if index <= 20:
        return "update_aware"
    return "action"


def _round_index(round_id: str | None) -> int:
    text = str(round_id or "").lower()
    if len(text) == 3 and text.startswith("q") and text[1:].isdigit():
        return int(text[1:])
    return 0


def _update_anchor_id(update_id: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]", "_", str(update_id).upper()).strip("_")
    return f"UPDATE:{clean or 'UNKNOWN'}_APPLIED"


class ScenarioPipelineBase:
    """Common constructor + progress logging for scenario LLM pipelines."""

    def __init__(
        self,
        client: ScenarioLLMClient,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.client = client
        self.progress = progress

    def _log(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)
