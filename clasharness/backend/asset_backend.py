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

"""Asset backend for multimodal scenario artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
from pathlib import Path
from typing import Any

from clasharness.backend.base import BaseBackend, CommandResult


# --------------------------------------------------------------------------
# Optional OCR side-channel (opt-in, off by default)
# --------------------------------------------------------------------------
# Set OPENCLAW_ASSET_OCR_INDEX to a JSON file mapping sha256(image bytes) ->
# extracted text, and `asset read-text` / `asset open` will serve that text
# instead of the "open it and look" stub. This exists ONLY so that text-only
# models can be probed against a corpus whose evidence is rendered into images;
# it deliberately defeats the visual requirement, so results obtained with it
# are NOT comparable with native-vision runs and must be reported separately.
#
# When the variable is unset every code path below is inert and the backend
# behaves exactly as before -- native-vision arms are unaffected.
#
# Keyed by content hash rather than asset id because asset ids repeat across
# tasks (`release_portal_release_runbook` occurs 20x in 120 sampled tasks) and
# the backend cannot identify its task: StateManager roots the episode at
# tempfile.mkdtemp(prefix="clasharness_"). Materialisation writes the blob
# with write_bytes(b64decode(...)), so the on-disk bytes hash identically to
# the source.
_OCR_INDEX_ENV = "OPENCLAW_ASSET_OCR_INDEX"
_OCR_INDEX_CACHE: dict[str, str] | None = None
_OCR_INDEX_LOADED_FROM: str | None = None


def _ocr_index() -> dict[str, str]:
    """Load (once per process) the sha256 -> text index, or {} if disabled."""
    global _OCR_INDEX_CACHE, _OCR_INDEX_LOADED_FROM
    path = os.environ.get(_OCR_INDEX_ENV, "").strip()
    if not path:
        return {}
    if _OCR_INDEX_CACHE is not None and _OCR_INDEX_LOADED_FROM == path:
        return _OCR_INDEX_CACHE
    try:
        with open(path, encoding="utf-8") as fh:
            loaded = json.load(fh)
        _OCR_INDEX_CACHE = loaded if isinstance(loaded, dict) else {}
    except Exception:
        _OCR_INDEX_CACHE = {}
    _OCR_INDEX_LOADED_FROM = path
    return _OCR_INDEX_CACHE


# Fields the AssetBackend may surface to the solving agent. This is an ALLOWLIST
# (not a denylist) so that any answer-bearing field -- description, caption,
# ocr_text, chart_rows, custom_overrides, region labels, or any field added
# later -- is withheld by default. The rendered image carries the actual
# content; the agent must open it and look. Keep in sync with the runtime/base
# payload in ``VLMAssetSpec.manifest_entry`` (clasharness/scenario/vlm_spec.py).
_AGENT_VISIBLE_FIELDS = frozenset(
    {
        "id",
        "kind",
        "generator",
        "source_type",
        "path",
        "mime_type",
        "available_from_round",
        "dimensions",
        "provider_model",
        "has_direct_image",
        "preset_id",
        "platform_id",
        "exists",
    }
)


def _get_arg(args: list[str], flag: str, default: str | None = None) -> str | None:
    for i, arg in enumerate(args):
        if arg == flag and i + 1 < len(args):
            return args[i + 1]
    return default


# ``--kind`` values that mean "do not filter" (i.e. list every asset). Agents --
# especially native tool-calling models that feel obliged to populate the
# optional ``kind`` argument -- routinely pass ``all``/``any`` expecting it to
# behave like an unfiltered listing rather than an exact-match filter.
_KIND_WILDCARDS = frozenset({"", "all", "any", "*", "everything", "none"})

# Common non-canonical ``--kind`` guesses mapped onto the canonical asset kinds
# (chart / natural_image / rich_text / diagram). Keeps a reasonable guess
# (e.g. ``image``) working instead of silently returning nothing.
_KIND_SYNONYMS = {
    "image": "natural_image",
    "images": "natural_image",
    "img": "natural_image",
    "photo": "natural_image",
    "photograph": "natural_image",
    "picture": "natural_image",
    "pictures": "natural_image",
    "natural": "natural_image",
    "text": "rich_text",
    "richtext": "rich_text",
    "rich-text": "rich_text",
    "document": "rich_text",
    "doc": "rich_text",
    "graph": "chart",
    "plot": "chart",
    "figure": "diagram",
    "schematic": "diagram",
}


class AssetBackend(BaseBackend):
    """Read-only backend exposing scenario-local multimodal assets."""

    def __init__(self) -> None:
        self._state_dir: Path | None = None
        self._assets_dir: Path | None = None
        self._assets: dict[str, dict[str, Any]] = {}
        self._ocr_by_asset_id: dict[str, str | None] = {}

    def _ocr_text(self, asset: dict[str, Any]) -> str | None:
        """Extracted text for this asset, or None when the side-channel is off."""
        index = _ocr_index()
        if not index:
            return None
        asset_id = str(asset.get("id") or "")
        if asset_id in self._ocr_by_asset_id:
            return self._ocr_by_asset_id[asset_id]
        text: str | None = None
        try:
            path = self._asset_path(str(asset.get("path")))
            if path.exists():
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                found = index.get(digest)
                if isinstance(found, str) and found.strip():
                    text = found
        except Exception:
            text = None
        self._ocr_by_asset_id[asset_id] = text
        return text

    def initialize(self, state_dir: str, env_vars: dict[str, str]) -> None:
        root = Path(env_vars.get("OPENCLAW_STATE_DIR", state_dir))
        self._state_dir = root
        self._assets_dir = root / "assets"
        self._assets = {}
        manifest_path = root / "asset_manifest.json"
        if not manifest_path.exists():
            alt = self._assets_dir / "asset_manifest.json"
            manifest_path = alt if alt.exists() else manifest_path
        if not manifest_path.exists():
            return
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        items = raw.get("assets", raw if isinstance(raw, list) else [])
        if not isinstance(items, list):
            return
        for item in items:
            if not isinstance(item, dict):
                continue
            asset_id = str(item.get("id") or "").strip()
            rel_path = str(item.get("path") or "").strip()
            if not asset_id or not rel_path:
                continue
            normalized = dict(item)
            normalized["id"] = asset_id
            normalized["path"] = rel_path
            normalized["exists"] = bool(self._asset_path(rel_path).exists())
            self._assets[asset_id] = normalized

    def execute_cli(self, command: str) -> CommandResult:
        parts = shlex.split(command.strip())
        if not parts or parts[0] != "asset":
            return CommandResult(stdout="", stderr="Not an asset command", exit_code=1)
        if len(parts) < 2:
            return CommandResult(
                stdout="asset <subcommand> [options]\nSubcommands: list, info, read-text, describe, open",
                stderr="",
                exit_code=0,
            )
        sub = parts[1]
        args = parts[2:]
        if sub == "list":
            return self._cmd_list(args)
        if sub == "info":
            return self._cmd_info(args)
        if sub == "read-text":
            return self._cmd_read_text(args)
        if sub == "describe":
            return self._cmd_describe(args)
        if sub == "open":
            return self._cmd_open(args)
        return CommandResult(stdout="", stderr=f"Unknown asset subcommand: {sub}", exit_code=1)

    def execute_python(self, code: str) -> CommandResult:
        return CommandResult(stdout="", stderr="Python interface not supported for AssetBackend", exit_code=1)

    def get_gateway_status(self) -> dict[str, Any] | None:
        return None

    def get_config(self) -> dict[str, Any]:
        return {}

    def cleanup(self) -> None:
        self._assets = {}
        self._state_dir = None
        self._assets_dir = None

    def get_state(self) -> dict[str, Any]:
        return {"assets": [dict(item) for item in self._assets.values()]}

    def _cmd_list(self, args: list[str]) -> CommandResult:
        raw_kind = _get_arg(args, "--kind")
        available = sorted(
            {str(item.get("kind")) for item in self._assets.values() if item.get("kind")}
        )
        kind_filter, error = self._resolve_kind_filter(raw_kind, available)
        if error is not None:
            return error
        items = [
            item
            for item in self._assets.values()
            if kind_filter is None or str(item.get("kind")) == kind_filter
        ]
        if not items:
            return CommandResult(stdout="No assets visible.", stderr="", exit_code=0)
        lines = ["Visible assets:"]
        for item in sorted(items, key=lambda x: str(x.get("id"))):
            lines.append(
                f"  {item.get('id')}  kind={item.get('kind', 'unknown')}  "
                f"path={item.get('path')}  mime={item.get('mime_type', 'unknown')}"
            )
        return CommandResult(stdout="\n".join(lines), stderr="", exit_code=0)

    def _resolve_kind_filter(
        self, raw_kind: str | None, available: list[str]
    ) -> tuple[str | None, CommandResult | None]:
        """Resolve a ``--kind`` value to a canonical filter (or a helpful error).

        Returns ``(kind_filter, None)`` on success, where ``kind_filter is None``
        means "no filtering" (list every asset). Returns ``(None, error_result)``
        when the value cannot be mapped to a kind present in this scenario -- the
        error explicitly lists the valid kinds instead of the previous behaviour
        of returning "No assets visible." (indistinguishable from a genuinely
        empty scenario), which repeatedly derailed agents that guessed a
        ``--kind`` value and concluded there was nothing to inspect.
        """
        if raw_kind is None:
            return None, None
        normalized = raw_kind.strip().lower()
        if normalized in _KIND_WILDCARDS:
            return None, None
        # Case-insensitive match against the kinds actually present.
        by_lower = {kind.lower(): kind for kind in available}
        if normalized in by_lower:
            return by_lower[normalized], None
        # Forgiving synonym mapping (e.g. ``image`` -> ``natural_image``).
        synonym = _KIND_SYNONYMS.get(normalized)
        if synonym is not None and synonym in available:
            return synonym, None
        valid = ", ".join(available) if available else "(none)"
        return None, CommandResult(
            stdout="",
            stderr=(
                f"Unknown --kind {raw_kind!r}. Valid kinds in this scenario: "
                f"{valid}. Omit --kind (or use --kind all) to list every asset."
            ),
            exit_code=1,
        )

    def _cmd_info(self, args: list[str]) -> CommandResult:
        asset, error = self._require_asset(args)
        if error:
            return error
        # Allowlist: emit only modality-neutral structural fields. Never echo
        # answer-bearing text/data (description/caption/ocr_text/chart_rows/
        # custom_overrides/regions), even if a stale manifest still carries them.
        payload = {k: v for k, v in asset.items() if k in _AGENT_VISIBLE_FIELDS}
        return CommandResult(
            stdout=json.dumps(payload, indent=2, ensure_ascii=False),
            stderr="",
            exit_code=0,
            state_changes={"assets_accessed": [_access_record(asset, "info")]},
        )

    def _cmd_read_text(self, args: list[str]) -> CommandResult:
        asset, error = self._require_asset(args)
        if error:
            return error
        extracted = self._ocr_text(asset)
        if extracted is not None:
            return CommandResult(
                stdout=(
                    f"Text extracted from asset '{asset.get('id')}' by OCR "
                    "(layout reconstructed; cells on one line are one row):\n\n"
                    f"{extracted}"
                ),
                stderr="",
                exit_code=0,
                state_changes={"assets_accessed": [_access_record(asset, "read-text")]},
            )
        # For this modality-neutral benchmark, answer-bearing text
        # (ocr_text/caption/description) is deliberately absent from the runtime
        # manifest -- the agent must open the asset and read it visually rather
        # than pull the answer from a text side-channel. Return a clear,
        # intentional stub instead of an empty string that looks like a failure.
        return CommandResult(
            stdout=(
                f"No extractable text is attached to asset '{asset.get('id')}'. "
                f"Use `asset open --id {asset.get('id')}` to view the image and read it directly."
            ),
            stderr="",
            exit_code=0,
            state_changes={"assets_accessed": [_access_record(asset, "read-text")]},
        )

    def _cmd_describe(self, args: list[str]) -> CommandResult:
        asset, error = self._require_asset(args)
        if error:
            return error
        # Neutral, non-leaking description: identity + structure only. The
        # asset's actual contents/values live in the pixels; direct the agent to
        # open it. No description/caption/anchor text is surfaced here.
        lines = [
            f"Asset: {asset.get('id')}",
            f"Kind: {asset.get('kind', 'unknown')}",
            f"Path: {asset.get('path')}",
            f"MIME: {asset.get('mime_type', 'unknown')}",
        ]
        dims = asset.get("dimensions")
        if isinstance(dims, dict) and dims.get("width") and dims.get("height"):
            lines.append(f"Dimensions: {dims.get('width')}x{dims.get('height')}")
        lines.append(
            f"Contents are visual. Use `asset open --id {asset.get('id')}` to view "
            "the image and read any values directly from it."
        )
        return CommandResult(
            stdout="\n".join(lines),
            stderr="",
            exit_code=0,
            state_changes={"assets_accessed": [_access_record(asset, "describe")]},
        )

    def _cmd_open(self, args: list[str]) -> CommandResult:
        asset, error = self._require_asset(args)
        if error:
            return error
        path = self._asset_path(str(asset.get("path")))
        if not path.exists():
            return CommandResult(stdout="", stderr=f"Asset file not found: {asset.get('path')}", exit_code=1)
        record = _access_record(asset, "open")
        header = (
            f"Asset opened: {asset.get('id')}\n"
            f"Path: {asset.get('path')}\n"
            f"MIME: {asset.get('mime_type', 'unknown')}\n"
        )
        # `open` -- not `read-text` -- is the primary scoring hook (it is the only
        # command emitting `assets_opened`), and a text-only model cannot act on
        # the image reference. So the side-channel has to answer here too, or the
        # OCR arm sees nothing after doing the "right" thing.
        extracted = self._ocr_text(asset)
        if extracted is not None:
            body = (
                "This asset is an image; its text has been extracted by OCR and is "
                "reproduced below (layout reconstructed; cells on one line are one row):\n\n"
                f"{extracted}"
            )
        else:
            body = "Use the asset reference from trajectory metadata for direct multimodal providers."
        return CommandResult(
            stdout=header + body,
            stderr="",
            exit_code=0,
            state_changes={"assets_accessed": [record], "assets_opened": [record]},
            meta={"asset_ref": record},
        )

    def _require_asset(self, args: list[str]) -> tuple[dict[str, Any], CommandResult | None]:
        asset_id = _get_arg(args, "--id")
        if not asset_id:
            return {}, CommandResult(stdout="", stderr="--id is required", exit_code=1)
        asset = self._assets.get(asset_id)
        if asset is None:
            return {}, CommandResult(stdout="", stderr=f"Asset not found: {asset_id}", exit_code=1)
        return asset, None

    def _asset_path(self, rel_path: str) -> Path:
        base = self._assets_dir or Path("assets")
        return base / rel_path.lstrip("/")


def _access_record(asset: dict[str, Any], action: str) -> dict[str, Any]:
    return {
        "id": asset.get("id"),
        "kind": asset.get("kind"),
        "path": asset.get("path"),
        "mime_type": asset.get("mime_type"),
        "anchor_ids": list(asset.get("anchor_ids", [])),
        "action": action,
    }
