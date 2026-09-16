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

import json
import logging
from typing import Any
from concurrent.futures import ThreadPoolExecutor
from clasharness.scenario.spec import ScenarioSpec
from clasharness.scenario.llm_client import ScenarioLLMClient


def audit_scenario_with_llm(
    spec: ScenarioSpec, llm_client: ScenarioLLMClient | None = None, max_iterations: int = 3
) -> ScenarioSpec:
    """Audit and iteratively repair the scenario spec using split LLM agents."""
    if llm_client is None:
        from clasharness.scenario.consistency_layer import audit_and_repair_scenario
        return audit_and_repair_scenario(spec)

    def _audit_single_round(round_item) -> None:
        task_spec = round_item.task_spec
        if not isinstance(task_spec, dict):
            return

        for attempt in range(max_iterations):
            logging.info(
                "Auditing round_id=%s attempt=%d/%d",
                round_item.round_id,
                attempt + 1,
                max_iterations,
            )

            # 1. Gather current spec state
            instruction = str(task_spec.get("instruction", ""))
            ground_truth = task_spec.get("ground_truth") or {}
            checks = ground_truth.get("evaluation_checks") or []
            private_data = task_spec.get("data", {}).get("private", {})
            if not isinstance(private_data, dict):
                break
            overrides = private_data.get("initial_state_overrides", {})
            if not isinstance(overrides, dict):
                break

            public_data = task_spec.get("data", {}).get("public", {})
            allowed_tool_families = public_data.get("allowed_tool_families") or []

            # 2. Run Auditor 1: Instruction vs Checker
            payload_1 = {
                "instruction": instruction,
                "checks": json.dumps(checks, indent=2),
                "allowed_tool_families": allowed_tool_families,
            }
            res_1 = llm_client.generate_layer("audit_instruction_checker", payload_1)
            has_contradiction = res_1.get("has_contradiction", False)

            # 3. Run Auditor 2: Anchor Discoverability
            expected_anchors = (
                ground_truth.get("answer") or {}
            ).get("expected_anchor_ids") or private_data.get("expected_anchor_ids", [])
            missing_anchors = []
            if expected_anchors:
                visible_files = {
                    k: v
                    for k, v in overrides.items()
                    if k.startswith("workspace/") and isinstance(v, str)
                }
                payload_2 = {
                    "expected_anchors": json.dumps(expected_anchors),
                    "visible_files_content": json.dumps(visible_files, indent=2),
                    "visible_visual_assets": json.dumps(round_item.visible_refs or []),
                }
                res_2 = llm_client.generate_layer("audit_anchor_discoverability", payload_2)
                missing_anchors = res_2.get("missing_anchors") or []

            # 4. Run Auditor 3: Visual-Textual Telemetry
            workspace_configs = {
                k: v
                for k, v in overrides.items()
                if k.endswith((".json", ".yaml", ".txt")) and isinstance(v, str)
            }
            chart_data = {
                k: v
                for k, v in overrides.items()
                if "chart" in k.lower() and isinstance(v, dict) and "__bytes_b64__" not in v
            }
            telemetry_mismatches = []
            if workspace_configs and chart_data:
                payload_3 = {
                    "chart_data": json.dumps(chart_data, indent=2),
                    "workspace_files": json.dumps(workspace_configs, indent=2),
                }
                res_3 = llm_client.generate_layer("audit_telemetry_alignment", payload_3)
                if res_3.get("has_mismatch"):
                    telemetry_mismatches = res_3.get("corrections") or []

            # If no issues found across all checks, this round is fully consistent!
            if not has_contradiction and not missing_anchors and not telemetry_mismatches:
                logging.info("Round %s is fully consistent.", round_item.round_id)
                break

            logging.info(
                "Contradictions found in round %s: has_contradiction=%s, missing_anchors_count=%d, telemetry_mismatches_count=%d. Repairing...",
                round_item.round_id,
                has_contradiction,
                len(missing_anchors),
                len(telemetry_mismatches),
            )

            # 5. Apply LLM Repairs
            # Repair 1: Instruction Contradiction
            if has_contradiction:
                repair_payload = {
                    "instruction": instruction,
                    "checks": json.dumps(checks, indent=2),
                    "allowed_tool_families": allowed_tool_families,
                }
                repair_res = llm_client.generate_layer(
                    "repair_instruction_contradiction", repair_payload
                )
                updated_inst = repair_res.get("updated_instruction")
                if updated_inst:
                    round_item.instruction = updated_inst
                    task_spec["instruction"] = updated_inst
                    task_spec["canonical_instruction"] = updated_inst
                subagent_effects = repair_res.get("subagent_effects_to_add") or []
                if subagent_effects:
                    subagents = private_data.setdefault("mock_subagents", [])
                    bg_events = private_data.setdefault("background_events", [])
                    targets = subagents if subagents else bg_events
                    if targets:
                        target = targets[0]
                        effects = target.setdefault("state_effects", [])
                        for eff in subagent_effects:
                            if eff not in effects:
                                effects.append(eff)

            # Repair 2: Missing Citation Anchors (Naturally integrate)
            for item in missing_anchors:
                anchor_id = item.get("anchor_id")
                target_file = item.get("target_file")
                if anchor_id and target_file:
                    clean_target = (
                        target_file.lstrip("/")
                        .replace("workspace/workspace/", "workspace/", 1)
                    )
                    # Skip image and binary assets since they cannot be edited via text rewriting
                    if clean_target.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf")):
                        logging.info("Skipping text-based anchor repair for binary/image asset: %s", target_file)
                        continue
                    matched_key = None
                    for key in overrides:
                        if (
                            key == clean_target
                            or key.endswith(clean_target)
                            or clean_target.endswith(key)
                        ):
                            matched_key = key
                            break

                    if matched_key:
                        original_content = overrides[matched_key]
                        repair_payload = {
                            "anchor_id": anchor_id,
                            "file_path": matched_key,
                            "file_content": original_content,
                        }
                        repair_res = llm_client.generate_layer(
                            "repair_missing_anchors", repair_payload
                        )
                        edited_content = repair_res.get("edited_content")
                        if edited_content:
                            overrides[matched_key] = edited_content

            # Repair 3: Telemetry Mismatches (Natively update numbers)
            for corr in telemetry_mismatches:
                file_path = corr.get("file_path")
                if file_path:
                    clean_path = (
                        file_path.lstrip("/")
                        .replace("workspace/workspace/", "workspace/", 1)
                    )
                    matched_key = None
                    for key in overrides:
                        if (
                            key == clean_path
                            or key.endswith(clean_path)
                            or clean_path.endswith(key)
                        ):
                            matched_key = key
                            break

                    if matched_key:
                        original_content = overrides[matched_key]
                        repair_payload = {
                            "mismatch_details": json.dumps(corr, indent=2),
                            "file_path": matched_key,
                            "file_content": original_content,
                        }
                        repair_res = llm_client.generate_layer(
                            "repair_telemetry_mismatches", repair_payload
                        )
                        edited_content = repair_res.get("edited_content")
                        if edited_content:
                            overrides[matched_key] = edited_content

    def _audit_single_round_safe(round_item) -> None:
        # Best-effort: the auditor is a repair layer, not a gate. If any LLM call
        # (generate_layer) or repair step raises for this round, log and keep the
        # round un-audited rather than aborting materialization before
        # scenario.json is written. F1 (_verify_and_repair) remains the
        # non-discarding verification backstop downstream, so a skipped audit
        # never means a discarded version.
        try:
            _audit_single_round(round_item)
        except Exception as e:  # pylint: disable=broad-except
            logging.warning(
                "LLM auditor failed for round_id=%s; keeping round un-audited "
                "(best-effort). Error: %s",
                getattr(round_item, "round_id", "?"),
                e,
            )

    # Empty rounds would make ThreadPoolExecutor(max_workers=0) raise; nothing to
    # audit anyway.
    if not spec.rounds:
        return spec

    with ThreadPoolExecutor(max_workers=max(1, len(spec.rounds))) as executor:
        list(executor.map(_audit_single_round_safe, spec.rounds))

    return spec
