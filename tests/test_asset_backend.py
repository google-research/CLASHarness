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

from __future__ import annotations

import json
from pathlib import Path

from clasharness.backend.asset_backend import AssetBackend


def test_asset_backend_preload_and_commands(tmp_path: Path) -> None:
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "board.svg").write_text("<svg>Board</svg>", encoding="utf-8")
    (tmp_path / "asset_manifest.json").write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "id": "asset_board",
                        "kind": "image",
                        "path": "board.svg",
                        "mime_type": "image/svg+xml",
                        "anchor_ids": ["EVIDENCE:BOARD_VISIBLE"],
                        "description": "Visual board",
                        "caption": "Board caption",
                        "ocr_text": "Board OCR EVIDENCE:BOARD_VISIBLE",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    backend = AssetBackend()
    backend.initialize(str(tmp_path), {"OPENCLAW_STATE_DIR": str(tmp_path)})

    listed = backend.execute_cli("asset list")
    assert listed.exit_code == 0
    assert "asset_board" in listed.stdout

    info = backend.execute_cli("asset info --id asset_board")
    assert info.exit_code == 0
    assert info.state_changes["assets_accessed"][0]["action"] == "info"
    # Modality-neutral: `info` emits only structural fields via an allowlist and
    # never echoes answer-bearing text, even though this fixture's manifest still
    # carries description/caption/ocr_text/anchor_ids.
    info_payload = json.loads(info.stdout)
    assert info_payload["id"] == "asset_board"
    for leaked in ("description", "caption", "ocr_text", "anchor_ids"):
        assert leaked not in info_payload

    text = backend.execute_cli("asset read-text --id asset_board")
    assert text.exit_code == 0
    # Answer-bearing OCR/caption/description text is NOT surfaced; the agent must
    # open the image and read it visually. A neutral stub is returned instead.
    assert "EVIDENCE:BOARD_VISIBLE" not in text.stdout
    assert "Board OCR" not in text.stdout
    assert "asset open --id asset_board" in text.stdout

    described = backend.execute_cli("asset describe --id asset_board")
    assert described.exit_code == 0
    assert "asset_board" in described.stdout
    # describe surfaces identity/structure only -- no description/caption/anchor text.
    for leaked in ("Visual board", "Board caption", "EVIDENCE:BOARD_VISIBLE"):
        assert leaked not in described.stdout

    opened = backend.execute_cli("asset open --id asset_board")
    assert opened.exit_code == 0
    assert opened.state_changes["assets_opened"][0]["id"] == "asset_board"
    assert opened.meta["asset_ref"]["path"] == "board.svg"

    state = backend.get_state()
    assert state["assets"][0]["exists"] is True
