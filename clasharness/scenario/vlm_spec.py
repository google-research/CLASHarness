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

"""Typed source-of-truth specs for CLASHarness PRO-VLM scenarios."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_SAFE_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]*$")
_DEFAULT_TOOL_FAMILIES = ["file", "email", "tasks", "calendar", "asset"]
_DEFAULT_MODALITIES = ["image", "chart", "table_screenshot", "document_image", "rich_text"]

# Visual asset kinds rendered by an image-generation model (not HTML/Chrome).
# ``natural_image`` -> a real photograph of a physical subject; ``sketch`` -> a
# photo of a HAND-DRAWN diagram on paper or a whiteboard. Both dispatch to the
# configured image provider rather than the chart/text renderer. NOTE: ``diagram``
# is deliberately NOT here -- a clean DIGITAL schematic is rendered as an in-UI
# HTML module (inline SVG) exactly like chart/rich_text.
IMAGE_KINDS = frozenset({"natural_image", "sketch"})

# Generators that call the image model. Kept distinct so the render dispatch and
# the detail-layer prompt can pick a photographic vs. hand-drawn style.
IMAGE_GENERATORS = frozenset({"google_image", "sketch_image"})

# Default generator per image kind. Used to backfill an empty ``generator`` so an
# image asset can never fall through to the mock renderer (the shared root cause
# of bare-twin ``natural_image`` assets rendering as a mock placeholder).
_DEFAULT_IMAGE_GENERATOR = {
    "natural_image": "google_image",
    "sketch": "sketch_image",
}

# Default generator for non-image kinds that still need an explicit generator to
# avoid a mock fallthrough. ``diagram`` is a DIGITAL schematic rendered as an
# in-UI HTML module (inline SVG), so like chart/rich_text it backfills to
# ``template_chart`` and is later resolved by ``chart_provider`` at render time.
_DEFAULT_NONIMAGE_GENERATOR = {
    "diagram": "template_chart",
}


def default_generator_for_kind(kind: str, generator: Any = None) -> str:
    """Return an explicit generator, backfilling kinds with an empty generator.

    A non-empty ``generator`` is always respected. Otherwise, image kinds
    (``natural_image``/``sketch``) get their canonical image generator so they
    route to the image provider; ``diagram`` backfills to ``template_chart`` (it
    renders as an in-UI HTML module); other kinds (chart/rich_text) are left empty
    and are resolved later by ``chart_provider``.
    """
    g = str(generator or "").strip()
    if g:
        return g
    k = str(kind or "").strip()
    if k in _DEFAULT_IMAGE_GENERATOR:
        return _DEFAULT_IMAGE_GENERATOR[k]
    return _DEFAULT_NONIMAGE_GENERATOR.get(k, "")


@dataclass(frozen=True)
class VLMInputPolicy:
    default_mode: str = "path"
    supports_direct: bool = True
    supports_both: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "VLMInputPolicy":
        data = data or {}
        return cls(
            default_mode=str(data.get("default_mode") or "path"),
            supports_direct=bool(data.get("supports_direct", True)),
            supports_both=bool(data.get("supports_both", True)),
        )


@dataclass(frozen=True)
class VLMGenerationPolicy:
    chart_provider: str = "template"
    natural_image_provider: str = "mock"
    google_model: str = "gemini-3.1-flash-image"
    chart_layouts: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "VLMGenerationPolicy":
        data = data or {}
        return cls(
            chart_provider=str(data.get("chart_provider") or "template"),
            natural_image_provider=str(data.get("natural_image_provider") or "mock"),
            google_model=str(data.get("google_model") or "gemini-3.1-flash-image"),
            chart_layouts=dict(data.get("chart_layouts") or {}),
        )


@dataclass(frozen=True)
class VLMVisualRegion:
    id: str
    label: str
    bbox: list[float] = field(default_factory=list)
    anchor_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VLMVisualRegion":
        return cls(
            id=str(data.get("id") or ""),
            label=str(data.get("label") or ""),
            bbox=[float(item) for item in data.get("bbox", [])],
            anchor_ids=[str(item) for item in data.get("anchor_ids", [])],
        )


@dataclass(frozen=True)
class VLMAssetSpec:
    id: str
    kind: str
    generator: str
    path: str
    mime_type: str
    available_from_round: int
    anchor_ids: list[str]
    description: str = ""
    caption: str = ""
    ocr_text: str = ""
    prompt: str = ""
    source_type: str = "generated"
    regions: list[VLMVisualRegion] = field(default_factory=list)
    dimensions: dict[str, int] = field(default_factory=dict)
    provider_model: str = ""
    render_source_path: str = ""
    render_code: str = ""
    chart_spec_path: str = ""
    metadata_path: str = ""
    has_direct_image: bool = True
    content_text: str = ""
    content_b64: str = ""
    provider_metadata: dict[str, Any] = field(default_factory=dict)
    preset_id: str | None = None
    custom_overrides: dict[str, Any] = field(default_factory=dict)
    chart_rows: list[dict[str, Any]] = field(default_factory=list)
    platform_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VLMAssetSpec":
        asset_id = str(data.get("id") or "").strip()
        path = str(data.get("path") or "").strip()
        if not path:
            path = f"charts/{asset_id}.png" if asset_id else "charts/unnamed_asset.png"
        kind = str(data.get("kind") or "").strip()
        return cls(
            id=asset_id,
            kind=kind,
            generator=default_generator_for_kind(kind, data.get("generator")),
            path=path,
            mime_type=str(data.get("mime_type") or ""),
            available_from_round=int(data.get("available_from_round") or 1),
            anchor_ids=[str(item) for item in data.get("anchor_ids", [])],
            description=str(data.get("description") or ""),
            caption=str(data.get("caption") or ""),
            ocr_text=str(data.get("ocr_text") or ""),
            prompt=str(data.get("prompt") or ""),
            source_type=str(data.get("source_type") or "generated"),
            regions=[VLMVisualRegion.from_dict(item) for item in data.get("regions", []) if isinstance(item, dict)],
            dimensions={str(k): int(v) for k, v in dict(data.get("dimensions", {})).items()},
            provider_model=str(data.get("provider_model") or ""),
            render_source_path=str(data.get("render_source_path") or ""),
            render_code=str(data.get("render_code") or ""),
            chart_spec_path=str(data.get("chart_spec_path") or ""),
            metadata_path=str(data.get("metadata_path") or ""),
            has_direct_image=bool(data.get("has_direct_image", True)),
            content_text=str(data.get("content_text") or ""),
            content_b64=str(data.get("content_b64") or ""),
            provider_metadata=dict(data.get("provider_metadata", {})),
            preset_id=data.get("preset_id"),
            custom_overrides=dict(data.get("custom_overrides") or {}),
            chart_rows=[dict(item) for item in data.get("chart_rows", []) if isinstance(item, dict)],
            platform_id=str(data.get("platform_id") or "").strip(),
        )

    def manifest_entry(self, *, include_generation_metadata: bool = True) -> dict[str, Any]:
        # Base payload = modality-NEUTRAL structural fields that are safe to hand
        # to the solving agent at runtime. Any field that can carry the round's
        # resolved answer -- in prose (description/caption/ocr_text/region labels)
        # or as literal data (chart_rows/custom_overrides) -- MUST live in the
        # ``include_generation_metadata`` block below so it is stripped from the
        # runtime manifest (``to_runtime_dict``) and survives only in the
        # build-time package (``to_package_dict``, used by the renderer/checker).
        # The rendered PNG bakes those values into pixels, so the agent loses no
        # information -- it just has to *look* instead of reading leaked text.
        payload = {
            "id": self.id,
            "kind": self.kind,
            "generator": self.generator,
            "source_type": self.source_type,
            "path": self.path,
            "mime_type": self.mime_type,
            "available_from_round": self.available_from_round,
            "dimensions": dict(self.dimensions),
            "provider_model": self.provider_model,
            "has_direct_image": self.has_direct_image,
            "preset_id": self.preset_id,
            "platform_id": self.platform_id,
        }
        if include_generation_metadata:
            payload.update(
                {
                    "anchor_ids": list(self.anchor_ids),
                    "description": self.description,
                    "caption": self.caption,
                    "ocr_text": self.ocr_text,
                    "prompt": self.prompt,
                    "regions": [asdict(region) for region in self.regions],
                    "custom_overrides": dict(self.custom_overrides),
                    # Guard against non-dict rows (e.g. positional arrays emitted
                    # by an LLM); dict() on a list row raises "dictionary update
                    # sequence element ... 2 is required". Mirrors the from_dict
                    # isinstance filter.
                    "chart_rows": [dict(row) for row in self.chart_rows if isinstance(row, dict)],
                    "render_source_path": self.render_source_path,
                    "render_code": self.render_code,
                    "chart_spec_path": self.chart_spec_path,
                    "metadata_path": self.metadata_path,
                    "provider_metadata": dict(self.provider_metadata),
                }
            )
        return payload


@dataclass(frozen=True)
class VLMEvalSpec:
    eval_id: str
    round_id: str
    expected_anchor_ids: list[str]
    checks: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VLMEvalSpec":
        return cls(
            eval_id=str(data.get("eval_id") or ""),
            round_id=str(data.get("round_id") or ""),
            expected_anchor_ids=[str(item) for item in data.get("expected_anchor_ids", [])],
            checks=list(data.get("checks", [])),
        )


@dataclass(frozen=True)
class VLMRoundSpec:
    round_id: str
    round_type: str
    visible_asset_ids: list[str]
    required_modalities: list[str]
    expected_anchor_ids: list[str]
    required_tool_families: list[str]
    question: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VLMRoundSpec":
        return cls(
            round_id=str(data.get("round_id") or ""),
            round_type=str(data.get("round_type") or ""),
            visible_asset_ids=[str(item) for item in data.get("visible_asset_ids", [])],
            required_modalities=[str(item) for item in data.get("required_modalities", [])],
            expected_anchor_ids=[str(item) for item in data.get("expected_anchor_ids", [])],
            required_tool_families=[str(item) for item in data.get("required_tool_families", [])],
            question=str(data.get("question") or ""),
        )


@dataclass(frozen=True)
class VLMSeed:
    slug: str
    domain: str
    title: str
    target_rounds: int
    core_conflict: str
    participants: list[str]
    required_constraints: list[str]
    allowed_tool_families: list[str]
    modalities: list[str] = field(default_factory=lambda: list(_DEFAULT_MODALITIES))
    generation_policy: VLMGenerationPolicy = field(default_factory=VLMGenerationPolicy)
    input_policy: VLMInputPolicy = field(default_factory=VLMInputPolicy)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source: str = "<dict>") -> "VLMSeed":
        slug = str(data.get("slug") or "").strip()
        if not _SAFE_SLUG_RE.match(slug):
            raise ValueError(f"invalid VLM seed slug in {source}: {slug!r}")
        return cls(
            slug=slug,
            domain=str(data.get("domain") or "multimodal operations"),
            title=str(data.get("title") or slug.replace("_", " ").title()),
            target_rounds=int(data.get("target_rounds") or 24),
            core_conflict=str(data.get("core_conflict") or ""),
            participants=[str(item) for item in data.get("participants", [])],
            required_constraints=[str(item) for item in data.get("required_constraints", [])],
            allowed_tool_families=[str(item) for item in data.get("allowed_tool_families", _DEFAULT_TOOL_FAMILIES)],
            modalities=[str(item) for item in data.get("modalities", _DEFAULT_MODALITIES)],
            generation_policy=VLMGenerationPolicy.from_dict(data.get("generation_policy")),
            input_policy=VLMInputPolicy.from_dict(data.get("input_policy")),
        )


@dataclass(frozen=True)
class VLMAssetManifest:
    assets: list[VLMAssetSpec]

    def to_runtime_dict(self, *, visible_asset_ids: set[str] | None = None) -> dict[str, Any]:
        visible = visible_asset_ids or {asset.id for asset in self.assets}
        return {
            "assets": [
                asset.manifest_entry(include_generation_metadata=False)
                for asset in self.assets
                if asset.id in visible
            ]
        }

    def to_package_dict(self) -> dict[str, Any]:
        return {"assets": [asset.manifest_entry(include_generation_metadata=True) for asset in self.assets]}


@dataclass(frozen=True)
class VLMScenarioSpec:
    series: str
    scenario_slug: str
    title: str
    domain: str
    core_conflict: str
    modalities: list[str]
    input_policy: VLMInputPolicy
    generation_policy: VLMGenerationPolicy
    allowed_tool_families: list[str]
    assets: list[VLMAssetSpec]
    rounds: list[VLMRoundSpec]
    evals: list[VLMEvalSpec]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_asset_content: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        if not include_asset_content:
            for asset in payload.get("assets", []):
                asset.pop("content_text", None)
                asset.pop("content_b64", None)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VLMScenarioSpec":
        return cls(
            series=str(data.get("series") or "pro_vlm_v1"),
            scenario_slug=str(data.get("scenario_slug") or ""),
            title=str(data.get("title") or ""),
            domain=str(data.get("domain") or ""),
            core_conflict=str(data.get("core_conflict") or ""),
            modalities=[str(item) for item in data.get("modalities", [])],
            input_policy=VLMInputPolicy.from_dict(data.get("input_policy")),
            generation_policy=VLMGenerationPolicy.from_dict(data.get("generation_policy")),
            allowed_tool_families=[str(item) for item in data.get("allowed_tool_families", [])],
            assets=[VLMAssetSpec.from_dict(item) for item in data.get("assets", []) if isinstance(item, dict)],
            rounds=[VLMRoundSpec.from_dict(item) for item in data.get("rounds", []) if isinstance(item, dict)],
            evals=[VLMEvalSpec.from_dict(item) for item in data.get("evals", []) if isinstance(item, dict)],
            metadata=dict(data.get("metadata", {})),
        )

    def write_json(self, path: Path, *, include_asset_content: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(include_asset_content=include_asset_content), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def read_json(cls, path: Path) -> "VLMScenarioSpec":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_vlm_seed_file(path: Path) -> VLMSeed:
    return VLMSeed.from_dict(json.loads(Path(path).read_text(encoding="utf-8")), source=str(path))


def validate_vlm_scenario_spec(spec: VLMScenarioSpec) -> list[str]:
    errors: list[str] = []
    if spec.series != "pro_vlm_v1":
        errors.append(f"unsupported VLM series: {spec.series}")
    if not _SAFE_SLUG_RE.match(spec.scenario_slug):
        errors.append(f"invalid scenario_slug: {spec.scenario_slug!r}")
    seen_assets: set[str] = set()
    for asset in spec.assets:
        if not asset.id:
            errors.append("asset missing id")
        if asset.id in seen_assets:
            errors.append(f"duplicate asset id: {asset.id}")
        seen_assets.add(asset.id)
        if not asset.kind:
            errors.append(f"asset {asset.id}: kind is required")
        if not asset.path:
            errors.append(f"asset {asset.id}: path is required")
        if not asset.mime_type:
            errors.append(f"asset {asset.id}: mime_type is required")
        if not asset.generator:
            errors.append(f"asset {asset.id}: generator is required")
        if asset.available_from_round < 1:
            errors.append(f"asset {asset.id}: available_from_round must be >= 1")
        if not asset.anchor_ids:
            errors.append(f"asset {asset.id}: anchor_ids are required")
        if asset.generator in IMAGE_GENERATORS and (not asset.prompt or not asset.caption):
            errors.append(f"asset {asset.id}: {asset.generator} requires prompt and caption")
        if asset.generator in {"code_chart", "template_chart"} and not (asset.chart_spec_path or asset.render_source_path):
            errors.append(f"asset {asset.id}: chart asset requires chart_spec_path or render_source_path")
    for round_item in spec.rounds:
        if not round_item.round_id:
            errors.append("round missing round_id")
        for asset_id in round_item.visible_asset_ids:
            if asset_id not in seen_assets:
                errors.append(f"round {round_item.round_id}: unknown visible asset {asset_id}")
        for family in round_item.required_tool_families:
            if family not in spec.allowed_tool_families:
                errors.append(f"round {round_item.round_id}: tool family {family} not allowed")
    return errors
