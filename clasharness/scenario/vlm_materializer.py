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

# pylint: skip-file

"""Materializer for CLASHarness PRO-VLM scenario packages."""

from __future__ import annotations

import base64
import json
import logging
import re
import shlex
import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

from clasharness.core.task import Difficulty, GroundTruth, Task, TaskData, save_task
from clasharness.env_knobs import getenv
from clasharness.scenario import design_archetypes
from clasharness.scenario._common import _round_type
from clasharness.scenario.llm_client import ScenarioLLMClient, ScenarioLLMError
from clasharness.scenario.spec import (
    ScenarioEvalItem,
    ScenarioManifest,
    ScenarioRound,
    ScenarioSession,
    ScenarioSpec,
    ScenarioUpdate,
    ScenarioWorkspace,
)
from clasharness.scenario.vlm_asset_generators import (
    CodeChartGenerator,
    GeneratedAsset,
    GoogleNanoBananaGenerator,
    MockImageGenerator,
    VertexImageGenerator,
)
from clasharness.scenario.vlm_spec import (
    IMAGE_GENERATORS,
    IMAGE_KINDS,
    VLMAssetManifest,
    VLMAssetSpec,
    VLMEvalSpec,
    VLMGenerationPolicy,
    VLMRoundSpec,
    VLMScenarioSpec,
    VLMSeed,
    load_vlm_seed_file,
    validate_vlm_scenario_spec,
)
from clasharness.tools.agent_tool_spec import validate_task_solution_commands

_SAFE_PATH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/\-]*$")


def materialize_vlm_scenario(
    seed: VLMSeed,
    *,
    scenario_series_dir: Path,
    output_data_dir: Path,
    clean: bool = True,
    provider: str = "mock",
    image_provider: str = "mock",
    chart_provider: str = "template",
    google_model: str = "gemini-3.1-flash-image",
    google_api_key_env: str = "GEMINI_API_KEY",
    google_base_url: str = "https://generativelanguage.googleapis.com/v1",
    raw_response_dir: Path | None = None,
    llm_client: ScenarioLLMClient | None = None,
    progress: Any | None = None,
    num_generation_workers: int | None = None,
    num_render_workers: int | None = None,
) -> ScenarioSpec:
    """Write a PRO-VLM package and its compiled task data."""
    scenario_dir = scenario_series_dir / "pro_vlm_workflow" / seed.slug
    if clean and scenario_dir.exists():
        shutil.rmtree(scenario_dir)
    for rel in ("assets", "workspace", "sessions", "updates"):
        (scenario_dir / rel).mkdir(parents=True, exist_ok=True)

    anchors = _canonical_anchors(seed)
    workspace_files = _build_workspace_files(seed, anchors)

    vlm_spec = build_vlm_scenario_spec(
        seed,
        provider=provider,
        image_provider=image_provider,
        chart_provider=chart_provider,
        google_model=google_model,
        llm_client=llm_client,
        progress=progress,
        num_generation_workers=num_generation_workers,
    )
    vlm_spec = generate_vlm_assets(
        vlm_spec,
        scenario_dir=scenario_dir,
        image_provider=image_provider,
        chart_provider=chart_provider,
        google_model=google_model,
        google_api_key_env=google_api_key_env,
        google_base_url=google_base_url,
        raw_response_dir=raw_response_dir,
        llm_client=llm_client,
        progress=progress,
        workspace_files=workspace_files,
        num_render_workers=num_render_workers,
        num_generation_workers=num_generation_workers,
    )
    spec_errors = validate_vlm_scenario_spec(vlm_spec)
    if spec_errors:
        raise ValueError("Invalid VLMScenarioSpec:\n" + "\n".join(spec_errors))

    VLMAssetManifest(vlm_spec.assets).to_package_dict()
    _write_json(scenario_dir / "asset_manifest.json", VLMAssetManifest(vlm_spec.assets).to_package_dict())
    vlm_spec.write_json(scenario_dir / "vlm_spec.json", include_asset_content=False)

    for rel_path, content in workspace_files.items():
        _write_text(scenario_dir / "workspace" / rel_path, content)

    sessions = _build_sessions(seed, anchors)
    for session in sessions:
        _write_text(scenario_dir / str(session["path"]), session["content"])

    updates = _build_updates(seed, anchors)
    for update in updates:
        for rel_path, content in update["files"].items():
            _write_text(scenario_dir / rel_path, content)

    tasks: list[Task] = []
    rounds: list[ScenarioRound] = []
    eval_items: list[ScenarioEvalItem] = []
    for round_spec in vlm_spec.rounds:
        task, update_ids = _compile_round_to_task(
            seed,
            vlm_spec=vlm_spec,
            round_spec=round_spec,
            workspace_files=workspace_files,
            updates=updates,
        )
        tasks.append(task)
        visible_assets = [asset for asset in vlm_spec.assets if asset.id in set(round_spec.visible_asset_ids)]
        rounds.append(
            ScenarioRound(
                round_id=round_spec.round_id,
                task_id=task.task_id,
                instruction=task.instruction,
                task_spec=task.to_dict(),
                update_ids=update_ids,
                session_refs=["vlm_briefing"],
                workspace_refs=["workspace_brief"],
                visible_refs=[asset.id for asset in visible_assets],
                expected_anchor_ids=list(round_spec.expected_anchor_ids),
                expert_trajectory=list(task.ground_truth.solution_commands if task.ground_truth else []),
                metadata={
                    "round_type": round_spec.round_type,
                    "visible_asset_ids": [asset.id for asset in visible_assets],
                    "required_modalities": list(round_spec.required_modalities),
                },
            )
        )
        eval_items.append(
            ScenarioEvalItem(
                eval_id=f"eval_{round_spec.round_id}",
                round_id=round_spec.round_id,
                kind="task_checks",
                checks=list(task.ground_truth.evaluation_checks if task.ground_truth else []),
                metadata={
                    "required_evidence_ids": list(round_spec.expected_anchor_ids),
                    "visible_asset_ids": [asset.id for asset in visible_assets],
                },
            )
        )

    spec = ScenarioSpec(
        scenario_id=f"pro_vlm_v1:{seed.slug}",
        generator_id="pro_vlm_v1",
        scenario_slug=seed.slug,
        title=seed.title,
        description=seed.core_conflict,
        version="1.0",
        narrative={
            "domain": seed.domain,
            "core_conflict": seed.core_conflict,
            "anchors": anchors,
            "allowed_tool_families": seed.allowed_tool_families,
            "vlm_spec": "vlm_spec.json",
        },
        workspace=ScenarioWorkspace(
            files=[
                {
                    "id": "workspace_brief" if rel_path == "brief.md" else rel_path.replace(".", "_"),
                    "path": rel_path,
                    "materialized_path": f"workspace/{rel_path}",
                    "substrate": "file",
                    "available_from_round": 1,
                    "anchor_ids": _extract_anchor_ids(content),
                }
                for rel_path, content in workspace_files.items()
            ],
            notes={"asset_manifest": "asset_manifest.json", "vlm_spec": "vlm_spec.json"},
        ),
        sessions=[
            ScenarioSession(
                session_id=str(session["session_id"]),
                kind="chat",
                channel=str(session["channel"]),
                path=str(session["path"]),
                metadata={"available_from_round": 1},
            )
            for session in sessions
        ],
        updates=[
            ScenarioUpdate(
                update_id=str(update["update_id"]),
                before_round=str(update["before_round"]),
                anchor_id=str(update["anchor_id"]),
                workspace_adds=[
                    {"id": key, "path": key, "materialized_path": rel_path}
                    for key, rel_path in update["workspace_adds"].items()
                ],
                metadata={"asset_ids": list(update.get("asset_ids", []))},
            )
            for update in updates
        ],
        rounds=rounds,
        eval_items=eval_items,
        manifest=ScenarioManifest(
            source_generator="pro_vlm_v1",
            compiled_task_ids=[task.task_id for task in tasks],
            metadata={"target_rounds": seed.target_rounds, "vlm_spec": "vlm_spec.json"},
        ),
        metadata={"seed": seed.__dict__, "asset_manifest": "asset_manifest.json", "vlm_spec": "vlm_spec.json"},
    )
    spec.write_json(scenario_dir / "scenario.json")
    _write_tasks_and_splits(tasks, output_data_dir)
    return spec


def build_vlm_scenario_spec(
    seed: VLMSeed,
    *,
    provider: str = "mock",
    image_provider: str = "mock",
    chart_provider: str = "template",
    google_model: str = "gemini-3.1-flash-image",
    llm_client: ScenarioLLMClient | None = None,
    progress: Any | None = None,
    num_generation_workers: int | None = None,
) -> VLMScenarioSpec:
    """Create a typed VLM source spec from a seed brief."""
    anchors = _canonical_anchors(seed)
    generation_policy = VLMGenerationPolicy(
        chart_provider=chart_provider,
        natural_image_provider=image_provider,
        google_model=google_model,
    )
    assets = _asset_specs(seed, anchors, generation_policy)
    if llm_client is not None:
        assets = llm_refine_vlm_assets(
            seed,
            assets=assets,
            client=llm_client,
            progress=progress,
            num_generation_workers=num_generation_workers,
        )
    def _process_round_task(index: int) -> tuple[VLMRoundSpec, VLMEvalSpec]:
        round_id = f"q{index:02d}"
        round_type = _round_type(index)
        visible_asset_ids = [asset.id for asset in assets if asset.available_from_round <= index]
        expected_anchor_ids = _expected_anchors(round_type, anchors)
        required_modalities = sorted({asset.kind for asset in assets if asset.id in visible_asset_ids})
        families = ["asset"]
        if round_type == "observation":
            families.append("tasks")
        elif round_type == "reconciliation":
            families.append("email")
        else:
            families.append("file")
        question = _round_question(seed, round_id, round_type)
        if llm_client is not None:
            question = _llm_round_question(
                seed,
                client=llm_client,
                round_id=round_id,
                round_type=round_type,
                visible_asset_ids=visible_asset_ids,
                required_modalities=required_modalities,
                progress=progress,
            )
        round_spec = VLMRoundSpec(
            round_id=round_id,
            round_type=round_type,
            visible_asset_ids=visible_asset_ids,
            required_modalities=required_modalities,
            expected_anchor_ids=expected_anchor_ids,
            required_tool_families=families,
            question=question,
        )
        eval_spec = VLMEvalSpec(
            eval_id=f"eval_{round_id}",
            round_id=round_id,
            expected_anchor_ids=expected_anchor_ids,
        )
        return round_spec, eval_spec

    # Decide worker thread count
    workers_input = num_generation_workers
    if workers_input is None:
        env_workers = getenv("CLASHARNESS_GENERATION_WORKERS")
        if env_workers is not None:
            try:
                workers_input = int(env_workers)
            except ValueError:
                workers_input = None

    if workers_input is None:
        workers_input = 4

    rounds: list[VLMRoundSpec] = []
    evals: list[VLMEvalSpec] = []

    if workers_input > 1 and seed.target_rounds > 1:
        _progress(progress, f"  Refining round questions in parallel using {workers_input} threads...")
        with ThreadPoolExecutor(max_workers=workers_input) as executor:
            futures = {
                executor.submit(_process_round_task, index=idx): idx
                for idx in range(1, seed.target_rounds + 1)
            }
            results = {}
            for fut in as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result()
        for idx in sorted(results.keys()):
            r_spec, e_spec = results[idx]
            rounds.append(r_spec)
            evals.append(e_spec)
    else:
        for idx in range(1, seed.target_rounds + 1):
            r_spec, e_spec = _process_round_task(idx)
            rounds.append(r_spec)
            evals.append(e_spec)
    return VLMScenarioSpec(
        series="pro_vlm_v1",
        scenario_slug=seed.slug,
        title=seed.title,
        domain=seed.domain,
        core_conflict=seed.core_conflict,
        modalities=list(seed.modalities),
        input_policy=seed.input_policy,
        generation_policy=generation_policy,
        allowed_tool_families=list(seed.allowed_tool_families),
        assets=assets,
        rounds=rounds,
        evals=evals,
        metadata={"provider": provider},
    )




def _process_caption_and_template(caption_data: Any, anchor_ids: list[str], custom_overrides: dict[str, Any]) -> str:
    from clasharness.scenario.vlm_materializer import _ensure_anchors
    if isinstance(caption_data, dict):
        template_id = custom_overrides.get("caption_template_id") or "operational_scene_update"
        from clasharness.scenario.chart_presets import CAPTION_TEMPLATES
        template = CAPTION_TEMPLATES.get(template_id)
        if not template:
            raise ValueError(f"Unknown caption template ID: {template_id}")
        try:
            formatted = template.template_string.format(**caption_data)
        except KeyError as exc:
            raise ValueError(f"Missing caption template key: {exc}") from exc
        caption = formatted
    else:
        caption = str(caption_data or "")

    caption = _ensure_anchors(caption, anchor_ids)
    return caption


def _fold_for_match(text: str) -> str:
    """Case/whitespace-insensitive fold used for decisive-value survival checks.

    MUST stay identical to ``file_checker._fold`` (what the deterministic grader
    applies to a ``contains`` check) and to
    ``conflict_llm_pipeline._fold_for_grounding`` (what the pipeline's grounding
    gate applies when it accepts a ``resolved_value``). It is defined locally
    rather than imported so this general-purpose materializer keeps no
    dependency on the conflict layer; a drift guard test pins the three folds to
    each other.
    """
    return " ".join(str(text or "").split()).lower()


def _missing_verbatim(fields: dict[str, Any], must_carry: list[str]) -> list[str]:
    """Values from ``must_carry`` that no longer read off caption/ocr_text."""
    blob = _fold_for_match(
        f"{fields.get('caption') or ''}\n{fields.get('ocr_text') or ''}"
    )
    return [v for v in must_carry if _fold_for_match(v) not in blob]


def _verbatim_retry_feedback(missing: list[str]) -> str:
    return (
        "Your previous answer DELETED reading(s) that must survive verbatim: "
        + "; ".join(f"'{v}'" for v in missing)
        + ". Rewrite 'caption' and 'ocr_text' again, reproducing every character of"
        " those strings exactly as given, and place them early enough that a length"
        " cap cannot cut them off. Keep them buried among the other readings on the"
        " surface -- do not highlight, bold, or annotate them."
    )


def llm_refine_vlm_assets(
    seed: VLMSeed,
    *,
    assets: list[VLMAssetSpec],
    client: ScenarioLLMClient,
    progress: Any | None,
    num_generation_workers: int | None = None,
    decisive_values_by_asset: dict[str, list[str]] | None = None,
) -> list[VLMAssetSpec]:
    """Rewrite each asset's authored text with the LLM.

    ``decisive_values_by_asset`` maps ``asset_id -> readings that MUST still be
    legible on that asset afterwards``.

    Why the guard exists: ``ConflictLLMPipeline`` already GUARANTEES that a
    round's ``resolved_value`` appears literally in the asset its
    ``resolved_value_locus`` names -- ``_validate_round_resolution`` rejects a
    resolution that is not grounded there, and the retry-exhausted fallback
    degrades to an empty value rather than an ungrounded one. This refiner then
    rewrites ``caption``/``ocr_text``, and nothing downstream re-checked it, so
    the pipeline's guarantee was being undone underneath it: in a 523-round
    corpus, 84 rounds (16.1%) ended up with a decisive value present in no asset
    and no workspace file.

    Only the LOCUS asset of a round is listed here, never every asset that
    happens to mention the value. Measured difficulty is highest when a reading
    has exactly one carrier (mean score 0.63 at 1 carrier vs 0.71-0.75 at 2-4,
    a gap that survives controlling for value uniqueness and length), so
    protecting extra carriers would make scenarios easier, not harder.
    """
    carriers = {
        str(asset_id): [str(v) for v in (values or []) if str(v).strip()]
        for asset_id, values in (decisive_values_by_asset or {}).items()
    }

    def _refine_single_asset_task(index: int, asset: VLMAssetSpec) -> VLMAssetSpec:
        _progress(progress, f"  VLM asset detail {index}/{len(assets)}: {asset.id}")
        requirements = [
            "Ensure description and caption are highly detailed, realistic operational descriptions (explaining specific trends, metrics, or anomalies) rather than generic text.",
            "For ocr_text/caption include the provided anchor_ids exactly when relevant.",
            "Do NOT include literal anchor IDs (such as 'EVIDENCE:...' or 'external_ingress_1015') inside the 'prompt' field. The prompt must only describe the visual scene, layout, and graphic contents. Avoid meta-phrases like 'this image represents anchor X'."
        ]
        if asset.kind == "natural_image":
            requirements.append(
                "For natural_image, you MUST write a photorealistic prompt for image generation in the 'prompt' field that "
                "depicts the REAL, PHYSICAL SUBJECT itself -- the actual equipment, device, or scene -- NOT a photo of a screen. "
                "First, infer the concrete physical object implied by the asset id/description (e.g. a network router/switch, a "
                "robotic arm and its work cell, a satellite antenna dish, a server rack, an industrial valve/pump, a medical "
                "infusion pump) and describe a close, well-lit photograph of THAT object. Convey the operational state through "
                "PHYSICAL cues only: status LED colors (green/amber/red), lit indicator lamps, an illuminated E-stop button, a "
                "seven-segment/LCD readout on the device itself, physical damage or a jammed part, warning placards, or a printed "
                "label/asset tag. "
                "HARD CONSTRAINTS -- the image must NOT be, and the prompt must NOT mention: a computer monitor, laptop, TV, or any "
                "screen displaying a dashboard/diagram/topology/chart; a whiteboard or printout/poster of a diagram; sticky notes, "
                "marker trays, or an office desk scene; the phrases 'the display shows', 'dashboard', or 'status board'. "
                "If the underlying content is inherently a diagram/topology/floor-plan/flowchart rather than a physical object, do "
                "NOT use natural_image -- it should be authored as a 'diagram' (clean digital schematic) or 'sketch' (hand-drawn) "
                "asset instead."
            )
        elif asset.kind == "sketch":
            requirements.append(
                "For sketch, you MUST write a photorealistic prompt for image generation in the 'prompt' field that depicts a "
                "PHOTOGRAPH of a HAND-DRAWN diagram -- either sketched with marker on a WHITEBOARD or drawn with pen/pencil on "
                "PAPER or in a notebook (choose whichever best fits the scene, e.g. a whiteboard for an incident war-room, a "
                "notebook for a field technician). Convey the schematic by hand: hand-lettered labels, slightly uneven freehand "
                "boxes/nodes, arrows drawn by hand, and natural imperfections (marker sheen, a smudge, a crossed-out correction). "
                "Describe the nodes/blocks, the connections between them, concise hand-written labels (include the anchor_ids where "
                "relevant), and which element is emphasized to convey the state (e.g. a node circled or underlined in red marker). "
                "HARD CONSTRAINTS -- it IS a casual photo of a REAL hand-drawn artifact: the prompt must NOT describe clean flat "
                "vector art, crisp digital typography, a computer monitor/screen, a rendered dashboard, or a printout."
            )
        elif asset.kind == "diagram":
            requirements.append(
                "For diagram, this is a CLEAN DIGITAL schematic that will be rendered as an in-UI module (inline SVG/HTML) inside a "
                "software tool's web UI -- it is NOT an image-generation prompt and NOT a photograph, so do NOT write an image "
                "'prompt'. Instead, make 'description' precisely enumerate the diagram's structure so it can be drawn as SVG: the "
                "nodes/blocks and their labels, the directional connections/edges between them, a short legend, and which single "
                "element is highlighted to convey the state (e.g. a node outlined in red for a fault, a green 'OK' node). Put the "
                "concrete node/edge labels (including the anchor_ids where relevant) into 'ocr_text' so they render as on-diagram "
                "text. Examples: a network topology, system architecture, facility/floor-plan layout, wiring/one-line diagram, or "
                "process flowchart."
            )
        else:
            requirements.append(
                "For chart/table assets, generate a detailed dataset of 5-10 entries in 'chart_rows' (essential for template rendering). "
                "You may also optionally generate valid Python code using matplotlib to save a clean, modern chart to OUTPUT_PATH in the 'render_code' field. IMPORTANT: Make the chart look like a professional, modern operational dashboard (e.g., use plt.style.use('dark_background') or set a dark slate/gray background, add subtle grid lines, use clean sans-serif fonts, select professional color themes like alert-red, warning-orange, and system-blue, and add clear axis labels with metric units)."
            )

        must_carry = list(carriers.get(asset.id) or [])
        if must_carry:
            requirements.append(
                "VERBATIM READINGS (HARD REQUIREMENT): this surface is the designated"
                " carrier for reading(s) the scenario is graded on. Every string listed"
                " under 'must_carry_verbatim' MUST still appear, character for"
                " character, in your 'caption' or your 'ocr_text': "
                + "; ".join(f"'{v}'" for v in must_carry)
                + ". Do not reword, re-order, round, abbreviate, translate, unit-convert"
                " or split them, and do not push them past the end of a long field."
                " Rewrite everything AROUND them as freely as you like. Do NOT call"
                " attention to them: no bold, no 'NOTE:', no 'discrepancy here' -- they"
                " must read as ordinary values among the other readings on this surface,"
                " and the surface should carry at least eight comparable readings of the"
                " same kind so that no single one stands out."
            )

        def _ask(feedback: str | None) -> dict[str, Any]:
            slot: dict[str, Any] = {
                "asset_id": asset.id,
                "kind": asset.kind,
                "generator": asset.generator,
                "available_from_round": asset.available_from_round,
                "anchor_ids": asset.anchor_ids,
                "default_description": asset.description,
                "default_caption": asset.caption,
                "default_ocr_text": asset.ocr_text,
                "default_prompt": asset.prompt,
                "requirements": requirements,
            }
            if must_carry:
                slot["must_carry_verbatim"] = list(must_carry)
            if feedback:
                slot["retry_feedback"] = feedback
            payload = {"seed": _seed_payload(seed), "slot": slot}
            try:
                return client.generate_layer("vlm_asset_detail", payload)
            except ScenarioLLMError as exc:
                raise ScenarioLLMError(f"VLM asset detail failed for {asset.id}: {exc}") from exc

        def _materialize(data: dict[str, Any]) -> dict[str, Any]:
            description = _compact_text(data.get("description"), fallback=asset.description)
            ocr_text = _compact_text(data.get("ocr_text"), fallback=asset.ocr_text)
            prompt = _compact_text(data.get("prompt"), fallback=asset.prompt)
            render_code = str(data.get("render_code") or "").strip()
            if not render_code:
                render_code = asset.render_code

            preset_id = data.get("preset_id") or asset.preset_id
            custom_overrides = dict(data.get("custom_overrides") or asset.custom_overrides or {})
            chart_rows = data.get("chart_rows") or asset.chart_rows or []
            # Normalize to a list of dict rows. LLMs occasionally emit chart rows as
            # positional arrays (e.g. ['03:00', 1210, 0.05, 'Operational']) instead of
            # keyed objects. Such rows are unusable downstream: the template renderer
            # calls row.get(...) and manifest serialization calls dict(row), which
            # raises "dictionary update sequence element #0 has length N; 2 is
            # required". This mirrors VLMAssetSpec.from_dict, which already drops
            # non-dict rows on load, keeping serialize/deserialize symmetric.
            if isinstance(chart_rows, list):
                chart_rows = [row for row in chart_rows if isinstance(row, dict)]
            else:
                chart_rows = []

            caption_raw = data.get("caption")
            caption = _process_caption_and_template(caption_raw, asset.anchor_ids, custom_overrides)
            ocr_text = _ensure_anchors(ocr_text, asset.anchor_ids)

            if asset.generator in IMAGE_GENERATORS or asset.kind in IMAGE_KINDS:
                prompt = prompt or asset.prompt
            provider_metadata = dict(asset.provider_metadata)
            if isinstance(chart_rows, list):
                provider_metadata["chart_rows"] = chart_rows
            provider_metadata["text_provider"] = "llm"
            return {
                "description": description,
                "caption": caption,
                "ocr_text": ocr_text,
                "prompt": prompt,
                "render_code": render_code,
                "provider_metadata": provider_metadata,
                "preset_id": preset_id,
                "custom_overrides": custom_overrides,
                "chart_rows": chart_rows,
            }

        fields = _materialize(_ask(None))

        if must_carry:
            missing = _missing_verbatim(fields, must_carry)
            if missing:
                # Repair by asking the model again with the loss spelled out,
                # rather than by scripting the string back in.
                fields = _materialize(_ask(_verbatim_retry_feedback(missing)))
                missing = _missing_verbatim(fields, must_carry)
            if missing:
                # Transaction abort. Restore the two surfaces the pipeline's
                # grounding gate blessed; everything the rewrite improved that
                # CANNOT destroy a reading (description, prompt, chart_rows,
                # render_code, styling) is kept. We deliberately do not splice
                # the value back into the new text: an appended "value: X" tail
                # is exactly the un-buried giveaway the burial requirement above
                # exists to prevent, and it would trade a grounding bug for a
                # difficulty bug.
                fields["caption"] = _ensure_anchors(asset.caption, asset.anchor_ids)
                fields["ocr_text"] = _ensure_anchors(asset.ocr_text, asset.anchor_ids)
                message = (
                    f"VLM asset {asset.id}: refiner dropped decisive reading(s)"
                    f" {missing} twice; reverted caption/ocr_text to the grounded"
                    " originals."
                )
                _progress(progress, f"  {message}")
                logging.warning(message)

        return replace(asset, **fields)

    # Decide worker thread count
    workers_input = num_generation_workers
    if workers_input is None:
        env_workers = getenv("CLASHARNESS_GENERATION_WORKERS")
        if env_workers is not None:
            try:
                workers_input = int(env_workers)
            except ValueError:
                workers_input = None

    if workers_input is None:
        workers_input = 4

    if workers_input > 1 and len(assets) > 1:
        _progress(progress, f"  Refining visual assets in parallel using {workers_input} threads...")
        with ThreadPoolExecutor(max_workers=workers_input) as executor:
            futures = {
                executor.submit(_refine_single_asset_task, index=idx, asset=asset): idx
                for idx, asset in enumerate(assets, start=1)
            }
            results = {}
            for fut in as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result()
        return [results[idx] for idx in sorted(results.keys())]
    else:
        refined: list[VLMAssetSpec] = []
        for idx, asset in enumerate(assets, start=1):
            refined.append(_refine_single_asset_task(idx, asset))
        return refined


def _llm_round_question(
    seed: VLMSeed,
    *,
    client: ScenarioLLMClient,
    round_id: str,
    round_type: str,
    visible_asset_ids: list[str],
    required_modalities: list[str],
    progress: Any | None,
) -> str:
    _progress(progress, f"  VLM eval round {round_id}: {round_type}")
    payload = {
        "seed": _seed_payload(seed),
        "slot": {
            "round_id": round_id,
            "round_type": round_type,
            "visible_asset_ids": visible_asset_ids,
            "required_modalities": required_modalities,
            "requirements": [
                "Do not include EVIDENCE:, DECISION:, RULE:, or UPDATE: ids.",
                "Mention the concrete side effect expected for the round type.",
                "Mention that only visible assets should be used.",
            ],
        },
    }
    try:
        data = client.generate_layer("vlm_round_question", payload)
    except ScenarioLLMError as exc:
        raise ScenarioLLMError(f"VLM round question failed for {round_id}: {exc}") from exc
    return _sanitize_question(_compact_text(data.get("question"), fallback=_round_question(seed, round_id, round_type)))


def _seed_payload(seed: VLMSeed) -> dict[str, Any]:
    return {
        "slug": seed.slug,
        "domain": seed.domain,
        "title": seed.title,
        "target_rounds": seed.target_rounds,
        "core_conflict": seed.core_conflict,
        "participants": list(seed.participants),
        "required_constraints": list(seed.required_constraints),
        "allowed_tool_families": list(seed.allowed_tool_families),
        "modalities": list(seed.modalities),
    }


def _compact_text(value: Any, *, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    return " ".join(text.split())[:1200]


def _ensure_anchors(text: str, anchors: list[str]) -> str:
    out = text.strip()
    missing = [anchor for anchor in anchors if anchor and anchor not in out]
    if missing:
        out = (out + " Evidence anchors: " + " ".join(missing)).strip()
    return out


def _sanitize_question(text: str) -> str:
    sanitized = re.sub(r"\b(?:EVIDENCE|DECISION|RULE|UPDATE):[A-Z0-9_\-]+", "the relevant visible evidence", text)
    return " ".join(sanitized.split())


def _progress(progress: Any | None, message: str) -> None:
    if progress is not None:
        progress(message)


# Image providers whose output is a real generated photo worth vision-auditing.
# ``mock`` paints the prompt onto a card, so auditing it is pointless (and would
# spend real API calls offline), hence it is excluded.
_FIDELITY_PROVIDERS = frozenset({"google", "vertex"})

# Chart providers whose output is a real headless-Chrome screenshot worth
# vision-auditing. These render to PNG exactly like the image providers, and the
# runtime strips their authored text just the same, so a decisive chart is every
# bit as pixel-exclusive as a decisive photo -- and is the *majority* of decisive
# assets in practice. ``mock`` is excluded for the same reason as above.
_FIDELITY_CHART_PROVIDERS = frozenset({"code", "template"})


def _audit_fidelity_safe(
    llm_client: Any,
    generated: GeneratedAsset,
    token: str,
    asset: VLMAssetSpec,
    progress: Any | None,
) -> dict[str, Any]:
    """Call the client's vision audit, never raising.

    A client that lacks the method, returns a non-dict, or errors is reported as
    ``skipped`` (a benign pass) so a flaky or vision-less audit can never wrongly
    downgrade a good decisive asset.

    Because a skip is indistinguishable from a pass downstream, every
    *operational* skip is logged at WARNING (not merely echoed to ``progress``,
    which is a transient console callback). At full decisive coverage a silent
    swallow here is how audit coverage erodes without anyone noticing; the log
    line is what makes the erosion countable after the fact. A client that
    simply has no vision support is a structural, expected skip and stays quiet.
    """
    audit_fn = getattr(llm_client, "audit_image_fidelity", None)
    if not callable(audit_fn):
        return {"legible": True, "skipped": True, "reason": "client has no vision audit"}
    try:
        result = audit_fn(
            image_bytes=generated.data,
            mime_type=generated.mime_type,
            expected_token=token,
            context=f"{asset.description} {asset.caption}".strip(),
        )
    except Exception as exc:  # noqa: BLE001 - audit must never break generation
        _progress(progress, f"  Warning: vision fidelity audit errored ({exc}); treating as pass")
        logging.warning(
            "Vision fidelity audit errored for decisive asset %s (token %r):"
            " %s -- treating as pass (asset NOT verified).",
            asset.id,
            token,
            exc,
        )
        return {"legible": True, "skipped": True, "reason": f"audit error: {exc}"}
    if not isinstance(result, dict):
        _progress(progress, "  Warning: vision fidelity audit returned non-dict; treating as pass")
        logging.warning(
            "Vision fidelity audit returned %s (not a dict) for decisive asset"
            " %s (token %r) -- treating as pass (asset NOT verified).",
            type(result).__name__,
            asset.id,
            token,
        )
        return {"legible": True, "skipped": True, "reason": "audit returned non-dict"}
    return result


def _record_vision_outcome(
    asset: VLMAssetSpec,
    audit: dict[str, Any],
    *,
    channel: str,
) -> bool:
    """Stamp one audit verdict onto the asset; return True if it may be kept.

    Splits apart the two outcomes the retry loop has to treat identically but
    which mean opposite things for dataset quality:

    * ``vision_verified=True`` -- a vision model actually looked at these pixels
      and confirmed the deciding token is readable. This is the only positive
      evidence that the round is genuinely solvable by looking;
    * ``vision_verified=False`` + ``vision_audit_skipped=<reason>`` -- the audit
      never produced a verdict (no vision-capable client, a backend error, a
      malformed response). Treated as a pass so a flaky audit can never
      wrongly downgrade a good asset, but recorded so the unverified population
      stays countable instead of being silently indistinguishable from a
      confirmed one.

    Without this split every decisive asset looks identical after generation and
    "N decisive assets audited" silently conflates "N confirmed legible" with "N
    never actually checked". Writes to ``provider_metadata``, which is not in
    ``AssetBackend._AGENT_VISIBLE_FIELDS`` and so never reaches the solving
    agent. Returns True when the asset is acceptable as-is (verified or
    unverified-but-benign), False when the audit positively found the token
    illegible and a re-render should be attempted.
    """
    pm = asset.provider_metadata
    pm["vision_audit_channel"] = channel
    if audit.get("skipped"):
        pm["vision_verified"] = False
        pm["vision_audit_skipped"] = str(audit.get("reason") or "unknown")
        return True
    pm.pop("vision_audit_skipped", None)
    if audit.get("legible"):
        pm["vision_verified"] = True
        return True
    pm["vision_verified"] = False
    return False


def _maybe_audit_and_fix_fidelity(
    asset: VLMAssetSpec,
    generated: GeneratedAsset,
    regenerate: Callable[[str], GeneratedAsset] | None,
    *,
    enabled: bool,
    llm_client: Any | None,
    retries: int,
    progress: Any | None,
    vision_audit: bool = True,
    channel: str = "image",
) -> GeneratedAsset:
    """Enforce that a DECISIVE asset legibly shows its deciding token in pixels.

    Runs on assets tagged ``evidence_role="decisive"`` by the conflict
    materializer, for every render channel that produces real pixels -- the
    image model AND the headless-Chrome HTML/chart renderer. This is not an
    image-only concern: the runtime strips ``ocr_text``/``content_text``/
    ``caption``/``description`` for *all* asset kinds (the manifest is emitted
    with ``include_generation_metadata=False`` and ``AssetBackend`` serves an
    allowlist), so a chart whose deciding value the HTML composer silently
    dropped, clipped or overflowed is unsolvable in exactly the same way an
    illegible photo is. Chart-family assets are the overwhelming majority of
    decisive assets (~93-95% measured across both generated sets), so auditing
    only the image channel left almost the entire population unverified.

    On a 'not legible' verdict it re-renders via ``regenerate(note)``, which
    receives a channel-appropriate strengthening instruction and returns a fresh
    ``GeneratedAsset`` (bounded, reusing the ``retries`` budget).
    ``regenerate=None`` means detect-only: a deterministic template render is a
    pure function of the asset's own text, so re-running it would reproduce the
    same pixels byte-for-byte. There the audit is still worth its API call as a
    *detector* -- it converts a silently-unsolvable round into an explicit
    downgrade -- so it skips straight to the downgrade instead of burning
    pointless retries. If still illegible it mutates ``asset.provider_metadata``
    in place --
    ``fidelity_failed=True`` and ``evidence_role`` downgraded to
    ``corroborating`` -- which the conflict materializer later uses to drop/relax
    the bound decisive check. Corroborating assets are never failed here (role is
    orthogonal to kind). Returns the (possibly re-rendered) ``GeneratedAsset``.

    ``enabled`` is the caller's channel-specific "these are real pixels" gate
    (a real image provider, or a real Chrome render rather than a mock stand-in).
    ``vision_audit=False`` (the ``--novision_audit`` escape hatch) skips the
    audit entirely, leaving every decisive asset as-is -- useful to save vision
    API calls or when the audit model is unavailable.

    ``VLMAssetSpec`` is frozen; only the mutable ``provider_metadata`` dict is
    mutated (never reassigned), and ``_write_generated_asset`` copies it into the
    written asset so the downgrade survives.
    """
    if not vision_audit:
        return generated  # audit disabled via --novision_audit
    if not enabled:
        return generated  # mock/offline channel: nothing meaningful to audit
    if asset.provider_metadata.get("evidence_role") != "decisive":
        return generated  # only decisive assets are gated
    token = str(asset.provider_metadata.get("expected_token") or "").strip()
    if not token or llm_client is None:
        return generated

    audit = _audit_fidelity_safe(llm_client, generated, token, asset, progress)
    if _record_vision_outcome(asset, audit, channel=channel):
        return generated

    note = (
        f" CRITICAL: a previous render did NOT clearly show the required reading \"{token}\"."
        f" Make the exact text \"{token}\" prominent, high-contrast, sharp, and unmistakably"
        " legible, while keeping the result realistic and natural."
    )
    # Reuse the retry budget but cap fidelity regenerations so a stubborn token
    # cannot explode the API-call count (decisive assets are a minority).
    # ``regenerate is None`` is the detect-only channel: re-rendering a
    # deterministic template reproduces identical pixels, so go straight to the
    # downgrade rather than burning retries that cannot possibly help.
    attempts = 0 if regenerate is None else max(1, min(int(retries or 1), 3))
    if regenerate is not None:
        for attempt in range(attempts):
            _progress(progress, f"  Fidelity retry {attempt + 1}/{attempts} for decisive asset {asset.id} (token not legible)")
            try:
                generated = regenerate(note)
            except Exception as exc:  # noqa: BLE001
                _progress(progress, f"  Fidelity retry generation failed: {exc}")
                continue
            audit = _audit_fidelity_safe(llm_client, generated, token, asset, progress)
            if _record_vision_outcome(asset, audit, channel=channel):
                return generated

    # Still illegible after retries: downgrade + flag (the conflict materializer
    # drops/relaxes the bound decisive check on this signal).
    asset.provider_metadata["fidelity_failed"] = True
    asset.provider_metadata["evidence_role"] = "corroborating"
    _progress(progress, f"  WARNING: decisive asset {asset.id} token \"{token}\" not legible after {attempts} retries; downgraded to corroborating")
    logging.warning(
        "Decisive %s asset %s token %r not legible after %d fidelity"
        " retries; downgraded to corroborating (bound decisive check relaxed).",
        channel,
        asset.id,
        token,
        attempts,
    )
    return generated


def generate_vlm_assets(
    spec: VLMScenarioSpec,
    *,
    scenario_dir: Path,
    image_provider: str,
    chart_provider: str,
    google_model: str,
    google_api_key_env: str,
    google_base_url: str,
    raw_response_dir: Path | None,
    google_project: str | None = None,
    vertex_project: str | None = None,
    vertex_location: str | None = None,
    request_delay_seconds: float = 0.0,
    llm_client: ScenarioLLMClient | None = None,
    progress: Any | None = None,
    workspace_files: dict[str, str] | None = None,
    style_seed: int = 0,
    num_render_workers: int | None = None,
    num_generation_workers: int | None = None,
    retries: int = 4,
    vision_audit: bool = True,
) -> VLMScenarioSpec:
    chart_gen = CodeChartGenerator()
    mock_gen = MockImageGenerator()

    effective_image_provider = image_provider
    if chart_provider == "google" and image_provider != "vertex":
        effective_image_provider = "google"

    if effective_image_provider == "google":
        image_gen = GoogleNanoBananaGenerator(
            model=google_model,
            api_key_env=google_api_key_env,
            base_url=google_base_url,
            request_delay_seconds=request_delay_seconds,
            raw_response_dir=raw_response_dir,
            retries=retries,
        )
    elif effective_image_provider == "vertex":
        image_gen = VertexImageGenerator(
            model=google_model,
            project=vertex_project,
            location=vertex_location,
            request_delay_seconds=request_delay_seconds,
            raw_response_dir=raw_response_dir,
            retries=retries,
        )
    else:
        image_gen = mock_gen

    # Pre-generate unified UI layout designs for all unique platform_ids
    platform_layout_specs = {}
    platform_noise_specs = {}
    if llm_client is not None:
        platform_ids = {a.platform_id or "default" for a in spec.assets if a.kind in {"chart", "rich_text", "diagram"}}

        def _design_platform_task(pid: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
            # Compose one whole, coherent design language from the SSoT archetype
            # library (clasharness.scenario.design_archetypes). This replaces the
            # historical MD5-index-into-flat-lists selection, which could pair
            # incoherent style/theme/shell combos. Sampling is deterministic for a
            # given (title, platform_id, style_seed) and stays coherent because it
            # picks an archetype fitting the shell and varies tokens only within
            # that archetype's declared ranges.
            seed_str = f"{spec.title}_{pid}_{style_seed}"
            design = design_archetypes.sample_platform_design(seed_str)
            suggested_theme = design["color_theme"]
            suggested_color = design["primary_color"]
            suggested_style = design["visual_style"]
            suggested_shell = design["shell_type"]
            suggested_style_details = design["style_details"]
            ws_files = []
            if workspace_files is not None:
                ws_files = list(workspace_files.keys())
            elif hasattr(spec, "workspace") and spec.workspace and hasattr(spec.workspace, "files"):
                ws_files = [f.get("path") for f in spec.workspace.files if isinstance(f, dict) and f.get("path")]

            layout_payload = {
                "scenario_title": spec.title,
                "core_conflict": spec.core_conflict,
                "suggested_theme": suggested_theme,
                "suggested_primary_color": suggested_color,
                "suggested_visual_style": suggested_style,
                "suggested_shell_type": suggested_shell,
                "suggested_style_details": suggested_style_details,
                "platform_id": pid,
                "workspace_files": ws_files,
            }
            _progress(progress, f"  Designing unified UI layout for platform: {pid}...")
            layout_spec = llm_client.generate_layer("generate_vlm_layout_spec", layout_payload)

            # Generate platform-wide redundant noise spec
            noise_payload = {
                "scenario_title": spec.title,
                "workspace_files": ws_files,
                "ui_shell_type": layout_spec.get("ui_shell_type", "ide"),
                "platform_context": layout_spec.get("platform_context", {}),
            }
            noise_spec = llm_client.generate_layer("generate_vlm_redundant_noise", noise_payload)

            return pid, layout_spec, noise_spec

        # Decide worker thread count
        workers_input = num_generation_workers
        if workers_input is None:
            env_workers = getenv("CLASHARNESS_GENERATION_WORKERS")
            if env_workers is not None:
                try:
                    workers_input = int(env_workers)
                except ValueError:
                    workers_input = None

        if workers_input is None:
            workers_input = 4

        if workers_input > 1 and len(platform_ids) > 1:
            _progress(progress, f"  Designing platform layouts in parallel using {workers_input} threads...")
            with ThreadPoolExecutor(max_workers=workers_input) as executor:
                futures = [executor.submit(_design_platform_task, pid) for pid in sorted(platform_ids)]
                for fut in as_completed(futures):
                    try:
                        pid, layout_spec, noise_spec = fut.result()
                        platform_layout_specs[pid] = layout_spec
                        platform_noise_specs[pid] = noise_spec
                    except Exception as e:
                        _progress(progress, f"  Warning: Failed to pre-design UI layout: {e}")
        else:
            for pid in sorted(platform_ids):
                try:
                    pid, layout_spec, noise_spec = _design_platform_task(pid)
                    platform_layout_specs[pid] = layout_spec
                    platform_noise_specs[pid] = noise_spec
                except Exception as e:
                    _progress(progress, f"  Warning: Failed to pre-design UI layout for platform {pid}: {e}")

    # Standalone helper for rendering a single visual asset (thread-safe)
    def _render_single_asset_task(index: int, asset: VLMAssetSpec) -> VLMAssetSpec:
        rel_path = asset.path
        if rel_path.startswith("assets/"):
            rel_path = rel_path[len("assets/"):]
        elif rel_path.startswith("/assets/"):
            rel_path = rel_path[len("/assets/"):]
        
        target_img_path = scenario_dir / "assets" / rel_path
        target_meta_path = scenario_dir / "assets" / f"{asset.id}.metadata.json"

        if target_img_path.is_file() and target_meta_path.is_file():
            try:
                import json
                with open(target_meta_path, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)
                
                # Determine expected provider for validation
                expected_generator = asset.generator
                if asset.kind in {"chart", "rich_text", "diagram"} and chart_provider:
                    if chart_provider == "code":
                        expected_generator = "code_chart"
                    elif chart_provider == "google":
                        expected_generator = "image_chart"
                    elif chart_provider == "template":
                        expected_generator = "template_chart"

                # Mirror the render dispatch order below (image_chart -> chart
                # renderer -> google_image -> mock). A chart/rich_text asset whose
                # generator is "google_image" is still rendered through the chart
                # path (its kind promotes effective_generator to a *_chart), so it
                # produces a "code_chart" provider on disk. The previous ordering
                # short-circuited on `asset.generator == "google_image"` and expected
                # "vertex", so those assets never cache-hit and were needlessly
                # re-rendered on every resume. Resolve expected_generator first, and
                # only fall back to the image provider for a bare google_image asset.
                if expected_generator == "image_chart":
                    expected_provider = effective_image_provider
                elif expected_generator in {"template_chart", "code_chart"}:
                    expected_provider = "code_chart"
                elif asset.generator in IMAGE_GENERATORS or asset.kind in IMAGE_KINDS:
                    # natural_image (photo) and diagram (schematic) both render via
                    # the configured image provider. Match on kind too so a bare
                    # image asset with an empty/normalized generator still cache-hits
                    # against its real provider instead of expecting "mock".
                    expected_provider = effective_image_provider
                else:
                    expected_provider = "mock"

                cached_provider = meta_data.get("provider_metadata", {}).get("provider")

                if cached_provider == expected_provider:
                    # Re-hydrate the rendered bytes from disk. `VLMScenarioSpec.write_json`
                    # persists spec.json with `include_asset_content=False`, so it pops
                    # `content_b64`/`content_text` to keep the checkpoint small. A resumed
                    # asset therefore arrives with an empty `content_b64`. If we returned it
                    # as-is, the cache-hit asset would flow into `_compile_round_to_task`
                    # with no bytes and the task's `data.private.initial_state_overrides`
                    # would embed an empty asset payload (falling through to the empty
                    # `content_text`). Read the cached pixels back and mirror the
                    # fresh-render `replace()` so a resumed materialize yields byte-identical
                    # tasks without re-rendering. On any read failure, fall through to a
                    # normal re-render rather than emit an empty asset.
                    try:
                        cached_bytes = target_img_path.read_bytes()
                    except Exception as e:
                        _progress(progress, f"  [CACHE MISS] Failed to read cached asset bytes for {asset.id}: {e}. Re-rendering...")
                    else:
                        _progress(progress, f"  [CACHE HIT] Asset {asset.id} is already generated on disk. Skipping rendering...")
                        return replace(
                            asset,
                            path=rel_path,
                            content_b64=base64.b64encode(cached_bytes).decode("ascii"),
                            mime_type=asset.mime_type or _guess_asset_mime(rel_path),
                            dimensions=asset.dimensions or _asset_dimensions(cached_bytes),
                            metadata_path=target_meta_path.name,
                            prompt=meta_data.get("prompt", asset.prompt),
                            caption=meta_data.get("caption", asset.caption),
                            provider_metadata=meta_data.get("provider_metadata", asset.provider_metadata)
                        )
                else:
                    _progress(progress, f"  [CACHE MISS] Provider mismatch for {asset.id} (cached: {cached_provider}, expected: {expected_provider}). Re-rendering...")
            except Exception as e:
                _progress(progress, f"  [CACHE MISS] Failed to read cached metadata for {asset.id}: {e}. Re-rendering...")

        _progress(progress, f"  Rendering visual asset {index}/{len(spec.assets)}: {asset.id} ({asset.generator}) -> assets/{asset.path}")
        effective_generator = asset.generator
        if asset.kind in {"chart", "rich_text", "diagram"} and chart_provider:
            if chart_provider == "code":
                effective_generator = "code_chart"
            elif chart_provider == "google":
                effective_generator = "image_chart"
            elif chart_provider == "template":
                effective_generator = "template_chart"

        if effective_generator == "image_chart":
            raw_prompt = asset.prompt or f"A professional operational dashboard chart. Description: {asset.description}. Caption: {asset.caption}."
            if llm_client is not None:
                try:
                    _progress(progress, f"  Calling LLM to clean prompt for asset: {asset.id}")
                    clean_res = llm_client.generate_layer("clean_vlm_prompt", {"prompt": raw_prompt})
                    chart_prompt = clean_res.get("cleaned_prompt") or raw_prompt
                except Exception as exc:
                    _progress(progress, f"  Warning: LLM prompt cleaning failed: {exc}. Falling back to basic regex...")
                    chart_prompt = _clean_prompt_text(raw_prompt)
            else:
                chart_prompt = _clean_prompt_text(raw_prompt)

            chart_prompt = chart_prompt + " Do NOT paint any metadata labels, prompt tags, or instructions literally on the image (e.g. do not write 'EVIDENCE:...' or 'Caption:...')."
            generated = image_gen.generate(prompt=chart_prompt, caption=asset.caption, anchors=asset.anchor_ids)
            # A model-drawn chart is pixels like any other decisive asset, so it
            # gets the same legibility guarantee as the photo branch below.
            generated = _maybe_audit_and_fix_fidelity(
                asset,
                generated,
                lambda note: image_gen.generate(
                    prompt=chart_prompt + note,
                    caption=asset.caption,
                    anchors=asset.anchor_ids,
                ),
                enabled=effective_image_provider in _FIDELITY_PROVIDERS,
                llm_client=llm_client,
                retries=retries,
                progress=progress,
                vision_audit=vision_audit,
                channel="model-chart",
            )
        elif effective_generator in {"template_chart", "code_chart"}:
            # Set only on the LLM-compose path, which is the one render in this
            # branch that can actually be *repaired* (re-composing the HTML with
            # a strengthening note can move the token somewhere legible). The
            # deterministic template/code renders are pure functions of the
            # asset's own text, so they stay ``None`` = detect-only.
            chart_recompose: Callable[[str], GeneratedAsset] | None = None
            if effective_generator == "code_chart" and asset.render_code:
                generated = chart_gen.render_python_code(asset.render_code)
            else:
                rendered_via_llm = False
                if llm_client is not None:
                    try:
                        _progress(progress, f"  LLM generating custom UI template for asset: {asset.id}")
                        ws_files = []
                        if workspace_files is not None:
                            ws_files = list(workspace_files.keys())
                        elif hasattr(spec, "workspace") and spec.workspace and hasattr(spec.workspace, "files"):
                            ws_files = [f.get("path") for f in spec.workspace.files if isinstance(f, dict) and f.get("path")]

                        # Distribute themes and shell types systematically using
                        # the asset index. Vocabularies come from the SSoT design
                        # library (design_archetypes) rather than a second
                        # hardcoded copy, so the shell/theme enums can never drift
                        # (WD5). This only seeds the layout-spec *fallback* (when a
                        # platform was not pre-designed); the primary path uses the
                        # pre-designed platform_layout_specs below.
                        suggested_themes = list(design_archetypes.COLOR_THEMES)
                        suggested_shells = list(design_archetypes.UI_SHELL_TYPES)
                        suggested_theme = suggested_themes[index % len(suggested_themes)]
                        suggested_shell = suggested_shells[index % len(suggested_shells)]

                        # Step 1: Resolve UI Layout Spec
                        pid = asset.platform_id or "default"
                        layout_spec = platform_layout_specs.get(pid)
                        if not layout_spec:
                            # Fallback if pre-design failed
                            layout_payload = {
                                "scenario_title": spec.title,
                                "core_conflict": spec.core_conflict,
                                "asset_kind": asset.kind,
                                "suggested_theme": suggested_theme,
                                "suggested_shell_type": suggested_shell,
                            }
                            layout_spec = llm_client.generate_layer("generate_vlm_layout_spec", layout_payload)

                        # Dynamically propagate design colors/theme into asset styling overrides
                        new_overrides = dict(asset.custom_overrides)
                        if "styling" not in new_overrides:
                            new_overrides["styling"] = {}
                        else:
                            new_overrides["styling"] = dict(new_overrides["styling"])
                        new_overrides["styling"]["theme"] = layout_spec.get("color_theme", "light")
                        new_overrides["styling"]["primary_color"] = layout_spec.get("primary_color", "blue")
                        new_overrides["styling"]["visual_style"] = layout_spec.get("visual_style", "modern_clean")
                        new_overrides["styling"]["style_details"] = layout_spec.get("style_details", {})

                        new_provider_metadata = dict(asset.provider_metadata) if isinstance(asset.provider_metadata, dict) else {}
                        ui_shell = layout_spec.get("ui_shell_type", "single")
                        
                        if "dashboard_data" not in new_overrides:
                            new_overrides["dashboard_data"] = {}
                        else:
                            new_overrides["dashboard_data"] = dict(new_overrides["dashboard_data"])
                        
                        # Store system name and merge dynamic platform context variables
                        new_overrides["dashboard_data"]["system_name"] = layout_spec.get("system_name", "System Control Console")
                        new_overrides["dashboard_data"].update(layout_spec.get("platform_context", {}))

                        if ui_shell == "monitoring_dashboard":
                            new_provider_metadata["layout"] = "devops_dashboard"
                        else:
                            new_provider_metadata["layout"] = "single"

                        asset = replace(asset, custom_overrides=new_overrides, provider_metadata=new_provider_metadata)

                        # Step 2: Retrieve Pre-designed Redundant Visual Noise Spec or Fallback
                        noise_spec = platform_noise_specs.get(pid)
                        if not noise_spec:
                            noise_payload = {
                                "scenario_title": spec.title,
                                "workspace_files": ws_files,
                                "ui_shell_type": layout_spec.get("ui_shell_type", "ide"),
                                "platform_context": layout_spec.get("platform_context", {}),
                            }
                            noise_spec = llm_client.generate_layer("generate_vlm_redundant_noise", noise_payload)

                        # Step 3: Compose HTML Page.
                        # Inject the SSoT design-archetype guidance for the
                        # platform's selected visual_style so the renderer draws
                        # from the canonical (library) design language rather than
                        # a static per-style prose blob baked into the contract.
                        _arch = design_archetypes.get_archetype(
                            layout_spec.get("visual_style", "modern_clean")
                        )
                        compose_payload = {
                            "layout_spec": layout_spec,
                            "noise_spec": noise_spec,
                            "design_guidance": {
                                "visual_style": _arch.name,
                                "aesthetic_descriptor": _arch.aesthetic_descriptor,
                                "render_guidance": _arch.render_guidance,
                            },
                            "asset_info": {
                                "id": asset.id,
                                "kind": asset.kind,
                                "description": asset.description,
                                "caption": asset.caption,
                                "content_text": asset.content_text or asset.ocr_text,
                                "anchor_ids": asset.anchor_ids,
                            },
                        }
                        res = llm_client.generate_layer("compose_vlm_html", compose_payload)
                        html_content = res["html_content"]
                        generated = chart_gen.render_raw_html(html_content)
                        rendered_via_llm = True

                        def _recompose_html(
                            note: str,
                            _payload: dict[str, Any] = compose_payload,
                            _client: Any = llm_client,
                        ) -> GeneratedAsset:
                            """Re-compose the page with a legibility instruction.

                            The composer is an LLM handed the asset's text: it can
                            drop, truncate or overflow the deciding value while
                            still emitting a plausible-looking page. Re-running it
                            with an explicit note is the repair, mirroring the
                            strengthened-prompt retry on the image branch.
                            Defaults bind the payload/client per render so the
                            closure is safe under the render thread pool.
                            """
                            retry_payload = dict(_payload)
                            retry_payload["fidelity_note"] = note
                            retry_res = _client.generate_layer(
                                "compose_vlm_html", retry_payload
                            )
                            return chart_gen.render_raw_html(
                                retry_res["html_content"]
                            )

                        chart_recompose = _recompose_html
                    except Exception as exc:
                        raise ScenarioLLMError(f"LLM custom template generation failed for asset {asset.id}: {exc}") from exc

                if not rendered_via_llm:
                    if asset.kind == "rich_text":
                        file_type = "TEXT"
                        name_lower = asset.path.lower()
                        if name_lower.endswith(".yaml") or name_lower.endswith(".yml"):
                            file_type = "YAML"
                        elif name_lower.endswith(".json"):
                            file_type = "JSON"
                        elif name_lower.endswith(".py"):
                            file_type = "Python"
                        elif name_lower.endswith(".sh"):
                            file_type = "Bash"
                        
                        title = Path(asset.path).name
                        content = asset.content_text or asset.ocr_text
                        
                        text_spec = {
                            "title": title,
                            "type": file_type,
                            "content": content
                        }
                        generated = chart_gen.render_text_spec(text_spec)
                    else:
                        chart_spec = _chart_spec_for_asset(asset, spec, style_seed)
                        generated = chart_gen.render_chart_spec(chart_spec)

            # Chrome-rendered assets are pixels too, and the runtime strips their
            # authored text exactly as it does an image's -- so a decisive chart
            # whose value the composer dropped or the layout clipped is just as
            # unsolvable as an illegible photo. This branch carries the large
            # majority of decisive assets, so leaving it unaudited left almost
            # the whole population unverified.
            generated = _maybe_audit_and_fix_fidelity(
                asset,
                generated,
                chart_recompose,
                enabled=chart_provider in _FIDELITY_CHART_PROVIDERS,
                llm_client=llm_client,
                retries=retries,
                progress=progress,
                vision_audit=vision_audit,
                channel="chart",
            )
        elif asset.generator in IMAGE_GENERATORS or asset.kind in IMAGE_KINDS:
            # Image-model assets: natural_image -> a real photo of the physical
            # subject; sketch -> a photo of a hand-drawn diagram on paper or a
            # whiteboard. (A DIGITAL diagram is NOT here -- it renders as an in-UI
            # HTML/SVG module via the chart branch above.) Routing on kind (not
            # just generator) guarantees a bare/normalized image asset never falls
            # through to the mock renderer below.
            is_sketch = asset.kind == "sketch" or asset.generator == "sketch_image"
            if asset.prompt:
                raw_prompt = asset.prompt
            elif is_sketch:
                raw_prompt = f"A casual photograph of a hand-drawn technical diagram sketched with marker on a whiteboard or pen on paper. Description: {asset.description}. Caption: {asset.caption}."
            else:
                raw_prompt = f"A realistic close-up photograph of the physical equipment/scene itself. Description: {asset.description}. Caption: {asset.caption}."
            if llm_client is not None:
                try:
                    _progress(progress, f"  Calling LLM to clean prompt for asset: {asset.id}")
                    clean_res = llm_client.generate_layer("clean_vlm_prompt", {"prompt": raw_prompt})
                    image_prompt = clean_res.get("cleaned_prompt") or raw_prompt
                except Exception as exc:
                    _progress(progress, f"  Warning: LLM prompt cleaning failed: {exc}. Falling back to basic regex...")
                    image_prompt = _clean_prompt_text(raw_prompt)
            else:
                image_prompt = _clean_prompt_text(raw_prompt)

            if is_sketch:
                image_prompt = image_prompt + " Render this as a REAL PHOTOGRAPH of a HAND-DRAWN diagram -- marker on a whiteboard or pen on paper/in a notebook -- with hand-lettered labels, slightly uneven freehand boxes/nodes, hand-drawn arrows, and natural imperfections (marker sheen, a smudge, a crossed-out correction). The highlighted element may be circled or underlined in red marker. It must NOT be clean flat vector art, crisp digital typography, a computer monitor/screen, a rendered dashboard, or a printout."
            else:
                image_prompt = image_prompt + " This must be a real photograph of the physical device/scene itself. Do NOT photograph a computer monitor, screen, laptop, whiteboard, or printout displaying a dashboard/diagram; convey state via physical cues (status LEDs, indicator lamps, labels)."
            image_prompt = image_prompt + " Do NOT paint any metadata labels, prompt tags, or instructions literally on the image (e.g. do not write 'EVIDENCE:...' or 'Caption:...')."
            # Decisive image assets (tagged by the conflict materializer) must
            # render their deciding token/state legibly IN the scene: the runtime
            # strips caption/OCR text, so the coordinator can only read it from
            # the pixels. Appended AFTER prompt cleaning so the token can never be
            # stripped. This is genuine in-scene physical text (a nameplate /
            # readout / stamp) -- the intended exception to the watermark ban.
            if asset.provider_metadata.get("evidence_role") == "decisive":
                decisive_token = str(asset.provider_metadata.get("expected_token") or "").strip()
                if decisive_token:
                    image_prompt = image_prompt + (
                        f" Exactly one element of the scene must clearly and legibly show the reading \"{decisive_token}\""
                        " the way it would genuinely appear on the real subject -- e.g. printed on the device's"
                        " nameplate/label, shown on a physical status readout or indicator, or stamped on the part --"
                        " rendered sharp, in focus, and large enough to read unambiguously. This is real in-scene"
                        " physical text, NOT a metadata caption, so it is the intended exception to the rule just"
                        " above. Keep it a natural, realistic scene (the labelled subject among believable"
                        " surroundings), NOT a sterile close-up of a sticker on a blank background."
                    )
            generated = image_gen.generate(prompt=image_prompt, caption=asset.caption, anchors=asset.anchor_ids)
            # Vision fidelity audit: a DECISIVE image must legibly show its
            # deciding token in the pixels (the runtime strips caption/OCR).
            # No-op for corroborating assets and for mock/offline providers;
            # a persistent failure downgrades the asset to corroborating.
            generated = _maybe_audit_and_fix_fidelity(
                asset,
                generated,
                lambda note: image_gen.generate(
                    prompt=image_prompt + note,
                    caption=asset.caption,
                    anchors=asset.anchor_ids,
                ),
                enabled=effective_image_provider in _FIDELITY_PROVIDERS,
                llm_client=llm_client,
                retries=retries,
                progress=progress,
                vision_audit=vision_audit,
                channel="image",
            )
        else:
            # Genuinely-unknown, non-image generator. Image kinds/generators can
            # never reach here (they are caught by the elif above), so a mock
            # placeholder can never silently stand in for a real image.
            raw_prompt = asset.prompt or asset.description
            clean_prompt = _clean_prompt_text(raw_prompt)
            generated = mock_gen.generate(prompt=clean_prompt, caption=asset.caption, anchors=asset.anchor_ids)
        
        return _write_generated_asset(scenario_dir, asset, generated)

    # Decide whether to run in parallel or sequentially based on num_render_workers
    workers_input = num_render_workers
    if workers_input is None:
        env_workers = getenv("CLASHARNESS_RENDER_WORKERS")
        if env_workers is not None:
            try:
                workers_input = int(env_workers)
            except ValueError:
                workers_input = None

    if workers_input is not None and workers_input > 1:
        _progress(progress, f"Starting parallel visual asset rendering with {workers_input} workers...")
        with ThreadPoolExecutor(max_workers=workers_input) as executor:
            futures = {
                executor.submit(_render_single_asset_task, index=idx, asset=asset): idx
                for idx, asset in enumerate(spec.assets, start=1)
            }
            
            results = {}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    # Attempt to cancel any pending tasks on failure
                    for pending_future in futures:
                        pending_future.cancel()
                    raise exc
                    
        generated_assets = [results[idx] for idx in sorted(results.keys())]
    else:
        _progress(progress, "Starting sequential visual asset rendering...")
        generated_assets = []
        for idx, asset in enumerate(spec.assets, start=1):
            generated_assets.append(_render_single_asset_task(idx, asset))

    return replace(spec, assets=generated_assets)


def _asset_specs(seed: VLMSeed, anchors: dict[str, str], policy: VLMGenerationPolicy) -> list[VLMAssetSpec]:
    title = seed.title
    if policy.chart_provider == "code":
        chart_generator = "code_chart"
    elif policy.chart_provider == "google":
        chart_generator = "image_chart"
    else:
        chart_generator = "template_chart"
    return [
        VLMAssetSpec(
            id="asset_board_early",
            kind="chart",
            generator=chart_generator,
            path="board_current.png",
            mime_type="image/png",
            available_from_round=1,
            anchor_ids=[anchors["early"]],
            description=f"Current visual status board for {title}.",
            caption=f"The board marks a blocker that conflicts with stale launch text. Anchor {anchors['early']}.",
            ocr_text=f"Visual board says: blocker active; do not close current review. Evidence anchor {anchors['early']}.",
            chart_spec_path="asset_board_early.chart.json",
            source_type="chart_spec",
            provider_metadata={
                "layout": policy.chart_layouts.get("asset_board_early", "devops_dashboard"),
                "chart_type": "bar_chart",
                "chart_rows": [
                    {"label": "Review Alpha", "value": 100.0, "anchor": anchors["early"]},
                    {"label": "Review Beta", "value": 120.0, "anchor": ""},
                    {"label": "Review Gamma", "value": 85.0, "anchor": ""},
                    {"label": "Review Delta", "value": 90.0, "anchor": ""},
                    {"label": "Review Epsilon", "value": 110.0, "anchor": ""},
                ]
            },
        ),
        VLMAssetSpec(
            id="asset_table_conflict",
            kind="table_screenshot",
            generator=chart_generator,
            path="evidence_table.png",
            mime_type="image/png",
            available_from_round=1,
            anchor_ids=[anchors["chart"]],
            description=f"Screenshot-style table with current counts and mismatch cells for {title}.",
            caption=f"Current visual table shows mismatched totals. Anchor {anchors['chart']}.",
            ocr_text=f"Table OCR: current reviewed total does not match stale summary. Evidence anchor {anchors['chart']}.",
            chart_spec_path="asset_table_conflict.chart.json",
            source_type="chart_spec",
            provider_metadata={
                "layout": policy.chart_layouts.get("asset_table_conflict", "devops_dashboard"),
                "chart_type": "stacked_bar_chart",
                "series_names": ["Reviewed Total", "Target Total"],
                "chart_rows": [
                    {"label": "Ingress-1", "values": [120.0, 120.0], "anchor": ""},
                    {"label": "Ingress-2", "values": [115.0, 115.0], "anchor": ""},
                    {"label": "Ingress-3", "values": [121.0, 121.0], "anchor": ""},
                    {"label": "Ingress-4", "values": [10.0, 120.0], "anchor": anchors["chart"]},
                    {"label": "Ingress-5", "values": [0.0, 110.0], "anchor": ""},
                ]
            },
        ),
        VLMAssetSpec(
            id="asset_owner_note_doc",
            kind="document_image",
            generator=chart_generator,
            path="owner_note_card.png",
            mime_type="image/png",
            available_from_round=8,
            anchor_ids=[anchors["text"], anchors["rule"]],
            description="Owner note rendered as a visual document with a precedence rule.",
            caption=f"Owner note states visual evidence must override stale summary. Anchors {anchors['text']} and {anchors['rule']}.",
            ocr_text=f"Owner note OCR: use current owner-controlled evidence over stale summary. Evidence anchors {anchors['text']} {anchors['rule']}.",
            chart_spec_path="asset_owner_note_doc.chart.json",
            source_type="chart_spec",
            provider_metadata={
                "layout": policy.chart_layouts.get("asset_owner_note_doc", "devops_dashboard"),
                "chart_type": "bar_chart",
                "chart_rows": [
                    {"label": "10:00 Ingress", "value": 1200.0, "anchor": ""},
                    {"label": "10:05 Ingress", "value": 1150.0, "anchor": ""},
                    {"label": "10:10 Ingress", "value": 1210.0, "anchor": ""},
                    {"label": "10:15 Ingress", "value": 0.0, "anchor": anchors["text"]},
                    {"label": "10:20 Ingress", "value": 0.0, "anchor": anchors["rule"]},
                ]
            },
        ),
        VLMAssetSpec(
            id="asset_scene_update",
            kind="natural_image",
            generator="google_image",
            path="scene_update.png",
            mime_type="image/png",
            available_from_round=15,
            anchor_ids=[anchors["update"], anchors["decision"]],
            description="Natural-scene-style operational photo showing the later update context.",
            caption=f"A natural operations scene confirms a later visual update and cross-modal review requirement. Anchors {anchors['update']} and {anchors['decision']}.",
            ocr_text=f"Scene interpretation: visual update applied; cross-modal review is required. Evidence anchors {anchors['update']} {anchors['decision']}.",
            prompt=(
                f"A realistic close-up photograph of the physical equipment involved in '{title}'. Show the actual device/hardware "
                "with its operational state conveyed through physical cues: a lit amber/red status LED or indicator lamp, a physical "
                "warning label or tag, and any visible fault (e.g. an unseated cable or an engaged stop). It must be a photo of the "
                "real equipment itself -- NOT a computer monitor, screen, whiteboard, or printout showing a dashboard or flowchart."
            ),
            provider_model=policy.google_model,
            source_type="google_image" if policy.natural_image_provider == "google" else "mock_image",
        ),
        VLMAssetSpec(
            id="asset_final_handoff",
            kind="rich_text",
            generator=chart_generator,
            path="final_handoff_card.png",
            mime_type="image/png",
            available_from_round=21,
            anchor_ids=[anchors["decision"], anchors["rule"], anchors["update"]],
            description="Final handoff card combining visual and text evidence.",
            caption=f"Final handoff requires cross-modal review. Anchors {anchors['decision']}, {anchors['rule']}, {anchors['update']}.",
            ocr_text=f"Final OCR: cite visual update, owner rule, and decision anchor {anchors['decision']} {anchors['rule']} {anchors['update']}.",
            chart_spec_path="asset_final_handoff.chart.json",
            source_type="chart_spec",
            provider_metadata={
                "layout": policy.chart_layouts.get("asset_final_handoff", "devops_dashboard"),
                "chart_type": "bar_chart",
                "chart_rows": [
                    {"label": "Final Block-A", "value": 10.0, "anchor": anchors["decision"]},
                    {"label": "Final Block-B", "value": 20.0, "anchor": anchors["rule"]},
                    {"label": "Final Block-C", "value": 30.0, "anchor": anchors["update"]},
                    {"label": "Final Block-D", "value": 0.0, "anchor": ""},
                    {"label": "Final Block-E", "value": 0.0, "anchor": ""},
                ]
            },
        ),
    ]


def _write_generated_asset(scenario_dir: Path, asset: VLMAssetSpec, generated: GeneratedAsset) -> VLMAssetSpec:
    rel_path = asset.path
    if rel_path.startswith("assets/"):
        rel_path = rel_path[len("assets/"):]
    elif rel_path.startswith("/assets/"):
        rel_path = rel_path[len("/assets/"):]

    if not rel_path.lower().endswith("." + generated.extension.lower()):
        rel_path = str(Path(rel_path).with_suffix("." + generated.extension))
    # Stripping a literal ``assets/`` prefix is not enough: a model may author any
    # other absolute path (``/lab_portal/charts/x.png``) or a ``..`` segment, and
    # ``Path.__truediv__`` *discards* the left operand when the right is absolute,
    # so the write would land outside the package (or die with PermissionError and
    # take the whole 10-task scenario-version with it). Contain it, as the sibling
    # ``render_source_path`` below already does.
    rel_path = _safe_asset_relpath(rel_path)
    asset_path = scenario_dir / "assets" / rel_path
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    asset_path.write_bytes(generated.data)

    metadata_path = f"{asset.id}.metadata.json"
    _write_json(
        scenario_dir / "assets" / metadata_path,
        {
            "id": asset.id,
            "generator": asset.generator,
            "provider_metadata": {
                **asset.provider_metadata,
                **_safe_provider_metadata(generated.metadata)
            },
            "caption": asset.caption,
            "prompt": asset.prompt,
            "anchor_ids": list(asset.anchor_ids),
        },
    )
    chart_spec_path = asset.chart_spec_path
    if generated.chart_spec:
        chart_spec_path = chart_spec_path or f"{asset.id}.chart.json"
        if chart_spec_path.startswith("assets/"):
            chart_spec_path = chart_spec_path[len("assets/"):]
        elif chart_spec_path.startswith("/assets/"):
            chart_spec_path = chart_spec_path[len("/assets/"):]
        # ``_write_json`` has no ``_SAFE_PATH_RE`` guard (unlike ``_write_text``),
        # so an escape here is silent and leaves ``visualize.html`` pointing at a
        # chart spec that never made it into the package.
        chart_spec_path = _safe_asset_relpath(chart_spec_path)
        _write_json(scenario_dir / "assets" / chart_spec_path, generated.chart_spec)
    render_source_path = asset.render_source_path
    if generated.render_source:
        if not render_source_path:
            ext = "html" if generated.render_source.strip().startswith("<") else "py"
            render_source_path = f"{asset.id}.render.{ext}"
        else:
            if generated.render_source.strip().startswith("<") and render_source_path.endswith(".py"):
                render_source_path = str(Path(render_source_path).with_suffix(".html"))
            elif not generated.render_source.strip().startswith("<") and render_source_path.endswith(".html"):
                render_source_path = str(Path(render_source_path).with_suffix(".py"))
        if render_source_path.startswith("assets/"):
            render_source_path = render_source_path[len("assets/"):]
        elif render_source_path.startswith("/assets/"):
            render_source_path = render_source_path[len("/assets/"):]
        render_source_path = _safe_asset_relpath(render_source_path)
        _write_text(scenario_dir / "assets" / render_source_path, generated.render_source)
    return replace(
        asset,
        path=rel_path,
        mime_type=generated.mime_type,
        content_b64=base64.b64encode(generated.data).decode("ascii"),
        provider_metadata={
            **asset.provider_metadata,
            **_safe_provider_metadata(generated.metadata)
        },
        provider_model=str(generated.metadata.get("model") or asset.provider_model),
        dimensions=_asset_dimensions(generated.data),
        metadata_path=metadata_path,
        chart_spec_path=chart_spec_path,
        render_source_path=render_source_path,
    )


def _template_styling_for_asset(
    asset: VLMAssetSpec, spec: VLMScenarioSpec, style_seed: int = 0
) -> dict[str, Any]:
    """Resolve archetype styling for the no-LLM ``chart_template.html`` path.

    ``chart_template.html`` reads styling from ``spec.custom_overrides.styling``
    in the shape ``{theme, primary_color, visual_style, style_details}``. The
    historical code emitted styling at the *top level* (``chart_spec['styling']``),
    which the template never reads and the matplotlib fallback ignores -- so the
    no-LLM template path silently rendered every chart with the default
    modern_clean / blue / light look regardless of the platform's selected
    archetype (the WD9 bug).

    This resolves the styling from the SSoT design-archetype library instead. Any
    styling already attached to the asset (e.g. by the LLM design path, or set
    explicitly by a spec author) wins; missing fields are filled from a coherent,
    deterministically sampled design keyed on ``(title, platform_id)`` so every
    asset on the same platform shares one look, and the same platform reproduces.

    Only *template-valid* pre-set values are honored: the top-level preset
    styling uses ``theme`` as a matplotlib palette hint ('cool'/'warm'), which
    must not leak into the template's light/dark ``theme``. An explicitly-set,
    valid ``visual_style`` anchors a coherent base (its own token preset), so we
    never pair one archetype's name with another's tokens.

    Sampling is keyed on ``(title, platform_id, style_seed)`` -- the same key the
    LLM design path uses in ``_design_platform_task`` -- so a pinned ``style_seed``
    reproduces the platform's look on this fallback path too (WD5).
    """
    existing: dict[str, Any] = {}
    if isinstance(asset.custom_overrides, dict):
        raw = asset.custom_overrides.get("styling")
        if isinstance(raw, dict):
            existing = raw

    pid = asset.platform_id or "default"

    # Base = an explicitly-set, valid archetype (coherent token preset) if the
    # author named one, else a deterministically sampled coherent design.
    explicit_style = existing.get("visual_style")
    if design_archetypes.has_archetype(explicit_style):
        archetype = design_archetypes.get_archetype(explicit_style)
        base = {
            "color_theme": archetype.default_theme,
            "primary_color": archetype.default_primary_color,
            "visual_style": archetype.name,
            "style_details": archetype.default_style_details(),
        }
    else:
        base = design_archetypes.sample_platform_design(f"{spec.title}_{pid}_{style_seed}")

    # Layer only template-valid explicit overrides on top of the coherent base.
    theme = existing.get("theme")
    if theme not in design_archetypes.COLOR_THEMES:
        theme = base["color_theme"]
    primary = existing.get("primary_color")
    if primary not in design_archetypes.PRIMARY_COLORS:
        primary = base["primary_color"]
    style_details = existing.get("style_details")
    if not isinstance(style_details, dict) or not style_details:
        style_details = base["style_details"]

    return {
        "theme": theme,
        "primary_color": primary,
        "visual_style": base["visual_style"],
        "style_details": style_details,
    }


def _chart_spec_for_asset(
    asset: VLMAssetSpec, spec: VLMScenarioSpec, style_seed: int = 0
) -> dict[str, Any]:
    if asset.preset_id:
        from clasharness.scenario.chart_presets import CHART_PRESETS
        preset = CHART_PRESETS.get(asset.preset_id)
        if not preset:
            raise ValueError(f"Unknown chart preset ID: {asset.preset_id}")

        overrides = asset.custom_overrides or {}
        pattern = overrides.get("pattern")
        if pattern:
            from clasharness.scenario.pattern_generator import generate_pattern_data
            labels = overrides.get("labels")
            if not labels and asset.chart_rows:
                labels = [
                    (
                        r.get("label")
                        or r.get("Timestamp")
                        or r.get("time")
                    )
                    for r in asset.chart_rows
                ]
            if not labels:
                labels = [f"10:{i*5:02d}" for i in range(6)]
            length = len(labels)
            pattern_params = overrides.get("pattern_params", {})
            series_names = overrides.get("series_names", preset.series_names)
            if series_names:
                series_data = []
                for idx, sname in enumerate(series_names):
                    s_params = (
                        pattern_params[idx]
                        if isinstance(pattern_params, list) and idx < len(pattern_params)
                        else pattern_params
                    )
                    s_pattern = (
                        pattern[idx]
                        if isinstance(pattern, list) and idx < len(pattern)
                        else pattern
                    )
                    series_data.append(
                        generate_pattern_data(s_pattern, s_params, length)
                    )
                rows = []
                for i, label in enumerate(labels):
                    row = {"label": label}
                    row["values"] = [
                        series_data[k][i] for k in range(len(series_names))
                    ]
                    for k, sname in enumerate(series_names):
                        row[sname] = series_data[k][i]
                    rows.append(row)
            else:
                data_vals = generate_pattern_data(pattern, pattern_params, length)
                rows = [
                    {"label": labels[i], "value": data_vals[i]}
                    for i in range(length)
                ]
        else:
            rows = asset.chart_rows
            if not rows:
                rows = [
                    {"label": "Current visual status", "value": asset.caption, "anchor": asset.anchor_ids[0] if asset.anchor_ids else ""},
                    {"label": "OCR/evidence", "value": asset.ocr_text[:120], "anchor": " ".join(asset.anchor_ids)},
                ]

        chart_spec = {
            "chart_type": overrides.get("chart_type", preset.chart_type),
            "title": overrides.get("title", preset.title),
            "subtitle": overrides.get("subtitle", preset.subtitle),
            "series_names": overrides.get("series_names", preset.series_names),
            "x_label": overrides.get("x_label", preset.x_label),
            "y_label": overrides.get("y_label", preset.y_label),
            "asset_id": asset.id,
            "kind": asset.kind,
            "rows": rows,
        }
        # Top-level 'styling' carries the preset's matplotlib-fallback hints
        # (e.g. cool/warm palette, show_grid). The HTML template does NOT read
        # this key -- it reads custom_overrides.styling (attached below).
        styling = {**preset.default_styling, **overrides.get("styling", {})}
        if styling:
            chart_spec["styling"] = styling
        layout = asset.provider_metadata.get("layout", "single")
        chart_spec["layout"] = layout
        if layout == "devops_dashboard":
            chart_spec["dashboard_data"] = _generate_telemetry_noise(spec, asset.id, style_seed)
        chart_spec.setdefault("custom_overrides", {})["styling"] = _template_styling_for_asset(asset, spec, style_seed)
        return chart_spec

    rows = asset.provider_metadata.get("chart_rows") if isinstance(asset.provider_metadata, dict) else None
    if not isinstance(rows, list) or not rows:
        rows = [
            {"label": "Current visual status", "value": asset.caption, "anchor": asset.anchor_ids[0] if asset.anchor_ids else ""},
            {"label": "OCR/evidence", "value": asset.ocr_text[:120], "anchor": " ".join(asset.anchor_ids)},
        ]
    chart_spec = {
        "title": spec.title,
        "subtitle": asset.description,
        "asset_id": asset.id,
        "kind": asset.kind,
        "rows": rows,
    }
    if isinstance(asset.provider_metadata, dict):
        for k, v in asset.provider_metadata.items():
            if k != "chart_rows":
                chart_spec[k] = v
    layout = asset.provider_metadata.get("layout", "single")
    chart_spec["layout"] = layout
    if layout == "devops_dashboard":
        chart_spec["dashboard_data"] = _generate_telemetry_noise(spec, asset.id, style_seed)
    chart_spec.setdefault("custom_overrides", {})["styling"] = _template_styling_for_asset(asset, spec, style_seed)
    return chart_spec


def _generate_telemetry_noise(
    spec: VLMScenarioSpec, asset_id: str, style_seed: int = 0
) -> dict[str, Any]:
    import random
    # Seed a *local* RNG so a pinned ``style_seed`` yields byte-identical noise
    # across runs and interpreters. A string seed uses version-2 seeding, which
    # is ``PYTHONHASHSEED``-independent (same guarantee sample_platform_design
    # relies on). A local ``random.Random`` also avoids mutating the global RNG
    # state other callers may depend on. (WD5 reproducibility hardening: the
    # historical code used the unseeded module-global ``random``, so a pinned
    # seed still jittered.)
    rng = random.Random(f"{spec.title}_{asset_id}_{style_seed}_telemetry")
    services_pool = ["auth-service", "gateway-proxy", "payment-api", "inventory-db", "frontend-app"]
    services = []
    for sname in services_pool:
        is_suspect = (
            sname in asset_id.lower()
            or sname.replace("-", "_") in spec.core_conflict.lower()
            or sname.replace("-", " ") in spec.core_conflict.lower()
            or any(part in spec.core_conflict.lower() for part in sname.split("-") if len(part) > 3)
        )
        services.append({
            "name": sname,
            "status": "warning" if is_suspect else "healthy",
            "details": "Latency anomaly" if is_suspect else "Healthy"
        })
    return {
        "system_name": "Production Control Cluster",
        "timestamp": "2023-10-27 10:45:00 UTC",
        "services": services,
        "metrics": [
            {
                "name": "CPU Usage",
                "value": f"{rng.randint(18, 45)}%",
                "status": "healthy",
                "sparkline": [rng.randint(15, 45) for _ in range(7)]
            },
            {
                "name": "Memory Allocation",
                "value": f"{rng.randint(62, 75)}%",
                "status": "healthy",
                "sparkline": [rng.randint(60, 75) for _ in range(7)]
            },
            {
                "name": "Network Volume",
                "value": f"{rng.randint(100, 950)} MB/s",
                "status": "healthy",
                "sparkline": [rng.randint(100, 950) for _ in range(7)]
            }
        ]
    }


def _clean_prompt_text(text: str) -> str:
    import re
    # Remove EVIDENCE:NAME tags (e.g. EVIDENCE:ARCH_DIAGRAM)
    text = re.sub(r'\bEVIDENCE:[A-Z0-9_]+\b', '', text)
    # Remove "Evidence anchors: ..." or "Anchors ..." lists (case-insensitive label, uppercase-only anchor list)
    anchor_pattern = r'\b(?:[eE][vV][iI][dD][eE][nN][cC][eE]\s+)?[aA][nN][cC][hH][oO][rR][sS]?:?\s*[A-Z0-9_]+(?:\s*,\s*[A-Z0-9_]+)*'
    text = re.sub(anchor_pattern, '', text)
    # Clean up double spaces or trailing punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.rstrip(" .,") + "."
    return text


def _compile_round_to_task(
    seed: VLMSeed,
    *,
    vlm_spec: VLMScenarioSpec,
    round_spec: VLMRoundSpec,
    workspace_files: dict[str, str],
    updates: list[dict[str, Any]],
) -> tuple[Task, list[str]]:
    round_index = _round_number(round_spec.round_id)
    task_id = f"pro_vlm_{seed.slug}_{round_spec.round_id}"
    visible_assets = [asset for asset in vlm_spec.assets if asset.id in set(round_spec.visible_asset_ids)]
    active_updates = [update for update in updates if _round_number(str(update["before_round"])) <= round_index]
    update_ids = [str(update["update_id"]) for update in active_updates]
    overrides = _initial_state_overrides(
        workspace_files=workspace_files,
        visible_assets=visible_assets,
        active_updates=active_updates,
    )
    action_command, effect_check = _action_for_round(seed, round_index, round_spec.round_type, round_spec.expected_anchor_ids)
    asset_ids = _asset_ids_for_round(visible_assets, round_spec.round_type)
    solution_commands = ["asset list"]
    for asset_id in asset_ids:
        solution_commands.append(f"asset read-text --id {asset_id}")
        solution_commands.append(f"asset open --id {asset_id}")
    solution_commands.extend([action_command, "DONE"])
    checks = [
        {
            "type": "effect",
            "effect_type": "assets_accessed",
            "condition": "field_contains",
            "expected": {"field": "anchor_ids", "value": round_spec.expected_anchor_ids[0]},
            "name": "asset_anchor_accessed",
            "weight": 0.35,
        },
        effect_check,
    ]
    if round_spec.round_type == "action":
        checks.append(
            {
                "type": "effect",
                "effect_type": "assets_accessed",
                "condition": "field_contains",
                "expected": {"field": "anchor_ids", "value": round_spec.expected_anchor_ids[-1]},
                "name": "cross_modal_asset_anchor_accessed",
                "weight": 0.0,
            }
        )
    instruction = _instruction(seed, round_spec, visible_assets)
    task = Task(
        task_id=task_id,
        instruction=instruction,
        canonical_instruction=instruction,
        domains=["composite", "asset", "multimodal", "file", "email", "tasks", "calendar"],
        difficulty=int(Difficulty.REASONING),
        initial_state="default",
        template_id="pro_vlm_mixed_action",
        generator_id="pro_vlm_v1",
        data=TaskData(
            public={
                "scenario_slug": seed.slug,
                "round_id": round_spec.round_id,
                "round_type": round_spec.round_type,
                "visible_assets": [asset.manifest_entry(include_generation_metadata=False) for asset in visible_assets],
                "allowed_tool_families": list(seed.allowed_tool_families),
                "required_modalities": list(round_spec.required_modalities),
                "multimodal_input_mode": vlm_spec.input_policy.default_mode,
            },
            private={
                "initial_state_overrides": overrides,
                "expected_anchor_ids": list(round_spec.expected_anchor_ids),
                "multimodal_input_supported": True,
                "visible_asset_ids": list(round_spec.visible_asset_ids),
                "vlm_asset_anchors": {asset.id: list(asset.anchor_ids) for asset in visible_assets},
            },
        ),
        ground_truth=GroundTruth(
            solution_commands=solution_commands,
            evaluation_checks=checks,
            metadata={"expected_anchor_ids": list(round_spec.expected_anchor_ids), "round_type": round_spec.round_type},
        ),
        decision_requirements=["Use visible asset evidence and current text state before action."],
        realism_tags=["pro_vlm_v1", round_spec.round_type, "multimodal_evidence"],
    )
    errors = validate_task_solution_commands([task], allowed_families=seed.allowed_tool_families)
    if errors:
        raise ValueError("Invalid PRO-VLM solution commands:\n" + "\n".join(errors))
    return task, update_ids


def _initial_state_overrides(
    *,
    workspace_files: dict[str, str],
    visible_assets: list[VLMAssetSpec],
    active_updates: list[dict[str, Any]],
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for rel_path, content in workspace_files.items():
        overrides[f"workspace/{rel_path}"] = content
    for update in active_updates:
        for rel_path, content in update["files"].items():
            if "/workspace/" in rel_path:
                workspace_path = rel_path.split("/workspace/", 1)[1]
                overrides[f"workspace/{workspace_path}"] = content
    overrides["asset_manifest.json"] = VLMAssetManifest(visible_assets).to_runtime_dict()
    for asset in visible_assets:
        if asset.content_b64:
            overrides[f"assets/{asset.path}"] = {"__bytes_b64__": asset.content_b64}
        else:
            overrides[f"assets/{asset.path}"] = asset.content_text
    overrides["emails.json"] = {
        "replace_seeded": True,
        "emails": [
            {
                "id": "email-vlm-brief",
                "sender": "owner@example.com",
                "to": "agent@example.com",
                "subject": "Use visual evidence before action",
                "body": "The visual assets carry the current status. Reconcile them with workspace text before notifying owners.",
                "folder": "inbox",
                "read": False,
            }
        ],
    }
    overrides["tasks.json"] = {"replace_seeded": True, "tasks": []}
    overrides["calendar_events.json"] = {
        "events": [
            {
                "id": "cal-vlm-review",
                "title": "Cross-modal evidence review window",
                "start": "2026-05-01T10:00:00Z",
                "end": "2026-05-01T11:00:00Z",
            }
        ]
    }
    return overrides


def _canonical_anchors(seed: VLMSeed) -> dict[str, str]:
    base = re.sub(r"[^A-Z0-9]+", "_", seed.slug.upper()).strip("_")
    return {
        "early": f"EVIDENCE:{base}_VISUAL_BLOCKER",
        "text": f"EVIDENCE:{base}_TEXT_OWNER_NOTE",
        "chart": f"EVIDENCE:{base}_CHART_CONFLICT",
        "update": f"UPDATE:{base}_VISUAL_UPDATE_APPLIED",
        "decision": f"DECISION:{base}_CROSS_MODAL_REVIEW_REQUIRED",
        "rule": f"RULE:{base}_VISUAL_EVIDENCE_OVERRIDES_STALE_TEXT",
    }


def _build_workspace_files(seed: VLMSeed, anchors: dict[str, str]) -> dict[str, str]:
    return {
        "brief.md": (
            f"# {seed.title}\n\n"
            f"Domain: {seed.domain}\n\n"
            f"Conflict brief: {seed.core_conflict}\n\n"
            "The text-only summary is incomplete. Use visible asset evidence, email, task, and calendar state before taking action.\n"
            f"Current owner note anchor: {anchors['text']}\n"
        ),
        "stale_summary.md": (
            "# Stale Summary\n\n"
            "This summary says the workflow can proceed immediately. It predates the visual evidence and must be reconciled.\n"
        ),
    }


def _build_sessions(seed: VLMSeed, anchors: dict[str, str]) -> list[dict[str, Any]]:
    lines = [
        {"speaker": "ops", "text": f"Please reconcile visual and text evidence for {seed.title}."},
        {"speaker": "owner", "text": f"The visual board is current. Track {anchors['early']} before proceeding."},
    ]
    return [
        {
            "session_id": "vlm_briefing",
            "channel": "#vlm-review",
            "path": "sessions/vlm_briefing.jsonl",
            "content": "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in lines),
        }
    ]


def _build_updates(seed: VLMSeed, anchors: dict[str, str]) -> list[dict[str, Any]]:
    update_id = "upd_visual_01"
    rel_path = f"updates/{update_id}/workspace/visual_update_note.md"
    content = (
        f"# Visual Update Note\n\n"
        f"A later image update is now visible for {seed.title}. The updated visual status confirms {anchors['update']} "
        f"and keeps the final decision on {anchors['decision']}.\n"
    )
    return [
        {
            "update_id": update_id,
            "before_round": "q15",
            "anchor_id": anchors["update"],
            "workspace_adds": {"visual_update_note": rel_path},
            "files": {rel_path: content},
            "asset_ids": ["asset_scene_update"],
        }
    ]


def _action_for_round(seed: VLMSeed, round_index: int, round_type: str, expected_anchor_ids: list[str]) -> tuple[str, dict[str, Any]]:
    anchor_text = " ".join(expected_anchor_ids)
    if round_type == "observation":
        title = f"{seed.slug} visual blocker triage {round_index:02d} {anchor_text}"
        return (
            f"tasks add --title {_q(title)} --priority high",
            {
                "type": "effect",
                "effect_type": "tasks_created",
                "condition": "field_contains",
                "expected": {"field": "title", "value": expected_anchor_ids[0]},
                "name": "task_created_with_asset_anchor",
                "weight": 0.65,
            },
        )
    if round_type == "reconciliation":
        body = f"Use cross-modal evidence before proceeding: {anchor_text}"
        return (
            f"email send --to owner@example.com --subject {_q('Cross-modal evidence reconciliation')} --body {_q(body)}",
            {
                "type": "effect",
                "effect_type": "emails_sent",
                "condition": "field_contains",
                "expected": {"field": "body", "value": expected_anchor_ids[0]},
                "name": "email_sent_with_cross_modal_anchor",
                "weight": 0.65,
            },
        )
    if round_type == "update_aware":
        content = f"Update-aware action log for {seed.slug}: {anchor_text}"
        return (
            f"file create --path /actions/{seed.slug}_{round_index:02d}.md --content {_q(content)}",
            {
                "type": "effect",
                "effect_type": "files_created",
                "condition": "field_contains",
                "expected": {"field": "content", "value": expected_anchor_ids[0]},
                "name": "action_log_created_with_update_anchor",
                "weight": 0.65,
            },
        )
    content = f"Final VLM handoff for {seed.slug}: {anchor_text}"
    return (
        f"file create --path /reports/{seed.slug}_{round_index:02d}.md --content {_q(content)}",
        {
            "type": "effect",
            "effect_type": "files_created",
            "condition": "field_contains",
            "expected": {"field": "content", "value": expected_anchor_ids[0]},
            "name": "final_report_created_with_vlm_anchor",
            "weight": 0.65,
        },
    )


def _instruction(seed: VLMSeed, round_spec: VLMRoundSpec, visible_assets: list[VLMAssetSpec]) -> str:
    visible_lines = "\n".join(
        f"- {asset.id}: kind={asset.kind}, path=assets/{asset.path}, available via asset commands"
        for asset in visible_assets
    )
    if round_spec.round_type == "observation":
        action = "Create a task-board item capturing the current visual blocker."
    elif round_spec.round_type == "reconciliation":
        action = "Send an email to the owner/team with the correct cross-modal reconciliation path."
    elif round_spec.round_type == "update_aware":
        action = "Create an action log under /actions/ that records how the newly visible update changes the decision state."
    else:
        action = "Write a concise final handoff/report under /reports/."
    return (
        f"Scenario: {seed.title}\n"
        f"Round: {round_spec.round_id} ({round_spec.round_type})\n"
        f"Task: {action}\n\n"
        "Visible multimodal assets:\n"
        f"{visible_lines}\n\n"
        "Use asset list/info/read-text/open plus file/email/tasks/calendar commands as needed. "
        "Do not rely on stale text alone; cite the evidence anchors you find in visible asset evidence."
    )


def _round_question(seed: VLMSeed, round_id: str, round_type: str) -> str:
    if round_type == "observation":
        return f"Record the current visual blocker for {seed.title}."
    if round_type == "reconciliation":
        return f"Notify the owner about the current cross-modal reconciliation path for {seed.title}."
    if round_type == "update_aware":
        return f"Update the action log after new visual evidence appears for {seed.title}."
    return f"Prepare the final multimodal handoff for {seed.title}."


def _expected_anchors(round_type: str, anchors: dict[str, str]) -> list[str]:
    if round_type == "observation":
        return [anchors["early"]]
    if round_type == "reconciliation":
        return [anchors["chart"], anchors["rule"]]
    if round_type == "update_aware":
        return [anchors["update"], anchors["decision"]]
    return [anchors["decision"], anchors["rule"], anchors["update"]]


def _asset_ids_for_round(visible_assets: list[VLMAssetSpec], round_type: str) -> list[str]:
    if round_type == "observation":
        wanted = {"asset_board_early", "asset_table_conflict"}
    elif round_type == "reconciliation":
        wanted = {"asset_table_conflict", "asset_owner_note_doc"}
    elif round_type == "update_aware":
        wanted = {"asset_scene_update"}
    else:
        wanted = {"asset_final_handoff", "asset_scene_update"}
    selected = [asset.id for asset in visible_assets if asset.id in wanted]
    return selected or [visible_assets[-1].id]


def _write_tasks_and_splits(tasks: list[Task], output_data_dir: Path) -> None:
    for task in tasks:
        save_task(task, data_dir=output_data_dir)
    datasets_dir = output_data_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    task_ids = sorted(task.task_id for task in tasks)
    for split in ("train", "dev", "test", "total"):
        (datasets_dir / f"{split}.txt").write_text("\n".join(task_ids) + "\n", encoding="utf-8")


def _extract_anchor_ids(text: str) -> list[str]:
    return sorted(set(re.findall(r"\b(?:EVIDENCE|DECISION|RULE|UPDATE):[A-Z0-9_\-]+", text)))


def _round_number(round_id: str) -> int:
    text = str(round_id or "").lower().strip()
    if text.startswith("q") and text[1:].isdigit():
        return int(text[1:])
    return 0


def _asset_dimensions(data: bytes) -> dict[str, int]:
    try:
        from PIL import Image
        import io

        image = Image.open(io.BytesIO(data))
        return {"width": int(image.width), "height": int(image.height)}
    except Exception:
        return {}


def _guess_asset_mime(rel_path: str) -> str:
    """Best-effort MIME type from an asset file extension.

    Used only as a fallback when a resumed (cache-hit) asset has no ``mime_type``;
    rendered VLM assets are raster images, so default to PNG.
    """
    ext = Path(rel_path).suffix.lower().lstrip(".")
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "gif": "image/gif",
        "bmp": "image/bmp",
        "svg": "image/svg+xml",
    }.get(ext, "image/png")


def _safe_provider_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    blocked = {"raw", "raw_response", "api_key", "authorization", "x-goog-api-key"}
    return {str(k): v for k, v in metadata.items() if str(k).lower() not in blocked}


def _q(text: str) -> str:
    return shlex.quote(text)


def _safe_asset_relpath(rel: str) -> str:
    """Sanitize an asset-relative path so each component satisfies _SAFE_PATH_RE.

    Model-authored asset ids can contain characters that are invalid in file
    names - notably a ``:`` namespace separator (e.g.
    ``workspace:database_connections_chart``). Since asset file names are derived
    from those ids, an unsanitized id makes ``_write_text`` reject the whole
    version. Replace any disallowed character with ``_``, drop path-traversal
    segments, and guarantee each component starts with an alphanumeric so the
    asset materializes instead of hard-failing generation.
    """
    parts = [p for p in rel.split("/") if p not in ("", ".", "..")]
    safe_parts: list[str] = []
    for part in parts:
        safe = re.sub(r"[^A-Za-z0-9._\-]+", "_", part).lstrip("._-")
        if not safe or not safe[0].isalnum():
            safe = "a" + safe
        safe_parts.append(safe)
    return "/".join(safe_parts) if safe_parts else "asset"


def _write_text(path: Path, content: str) -> None:
    if not _SAFE_PATH_RE.match(path.name) and path.name not in {"scenario.json", "vlm_spec.json", "asset_manifest.json"}:
        raise ValueError(f"unsafe file name: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
