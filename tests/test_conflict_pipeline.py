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

import collections
import copy
import os
from pathlib import Path
import shlex
import sys
import tempfile
from typing import Any
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from clasharness.scenario.conflict_strict_validator import ConflictValidationError, validate_conflict_scenario_strict, iter_mock_command_effects, _value_leaks_in_text, _is_asset_locus, _session_visible_text, _validate_protocol_completion, _required_completion_effects, _completion_command_prefixes, enforce_async_wait_contract, _validate_async_wait_contract, _gold_delegates_then_waits, _round_has_delayed_subagent
from clasharness.scenario.conflict_vlm_materializer import (
    materialize_conflict_scenario,
    _effect_target_file_path,
    _drop_conflicting_subagent_file_effects,
    _calibrate_checks_against_state,
    _demote_vacuous_checks,
    _is_negative_assertion,
    _replay_zero_action,
    _ZERO_ACTION_JUDGE_ENV,
    _is_semantic_check,
    _stub_semantic_file_checks,
    _ReplaySafeStub,
    _norm_file_path,
    _tag_decisive_assets,
    _reconcile_decisive_fidelity_checks,
    _resolved_value_has_readable_carrier,
    _check_asserts_value,
    _build_async_within_window_check,
    _task_visible_text_surfaces,
    _classify_round_vision_exclusivity,
    _summarize_vision_audit,
    _vision_exclusivity_report,
)
from clasharness.evaluation.checkers.effect_checker import EffectChecker
from clasharness.scenario.vlm_materializer import (
    _maybe_audit_and_fix_fidelity,
    _audit_fidelity_safe,
    _FIDELITY_PROVIDERS,
)
from clasharness.scenario.vlm_asset_generators import GeneratedAsset
from clasharness.scenario.consistency_layer import (
    _is_content_unassertable_path,
    _has_content_assertion,
    audit_and_repair_scenario,
)
from clasharness.scenario.conflict_vlm_spec import ConflictVLMSeed, ConflictEvidenceSpec, ConflictVLMRoundSpec, ConflictVLMScenarioSpec
from clasharness.tools.agent_tool_spec import normalize_command_quoting
from clasharness.tools.agent_tool_spec import normalize_family_name
from clasharness.tools.agent_tool_spec import validate_command_against_spec
from clasharness.scenario.vlm_spec import VLMAssetSpec
from clasharness.scenario import design_archetypes as da
from clasharness.scenario.llm_auditor import audit_scenario_with_llm
from clasharness.scenario.spec import ScenarioSpec, ScenarioRound, ScenarioSession, SessionMessage
from clasharness.scenario.conflict_llm_pipeline import (
    ConflictLLMPipeline,
    _coerce_dependency_step,
    _validate_expanded_phase,
    _validate_expanded_phase_assets,
    _validate_expanded_phase_decisive,
    _is_compound_decisive_value,
    _norm_decisive,
    _format_round_carrier_directive,
    _assets_visible_in_round,
    _asset_kind_by_id,
    _format_decisive_stakes,
    _DECISIVE_POINT_MAX_LEN,
    _validate_generated_round,
    _normalize_and_prefix_assets,
    _sanitize_platform_id,
    _build_platform_vocab,
    _stable_platform_choice,
    _dedupe_preserve,
    _DEFAULT_PLATFORM_VOCAB,
    _MIN_PLATFORM_VOCAB,
    _DEFAULT_PLATFORM_ID,
    _validate_background_payload,
    _validate_generated_round_narrative,
    _validate_workspace_paths,
    _sanitize_leaked_asset_ids,
    _neutral_asset_phrase,
    _extract_flag_value,
    _normalize_ws_path,
    _is_code_file,
    _classify_step_command,
    _classify_trajectory_command,
    _scaffold_round_checkers,
    _parse_resolved_locus,
    _validate_round_resolution,
    _validate_round_decision_outcome,
    _validate_round_confidentiality,
    _fold_for_grounding,
    _asset_grounding_blob,
    _collect_round_workspace_texts,
    _collect_round_session_texts,
    _collect_round_mock_texts,
    _audit_extractor_liveness,
    ExtractorBlindError,
    _build_resolution_grounding_text,
    _format_retry_feedback,
    _round_command_signature,
    _coerce_evidence_anchor,
    _parse_semantic_fields,
    _apply_round_semantic_fields,
    _steps_from_scaffolds,
    _validate_generated_round_commands,
    _validate_generated_round_checkers,
    _validate_generated_round_mocks,
    _normalize_mock_state_effect_quoting,
    _normalize_mock_durations,
    _validate_generated_screenplay_triggers,
    _validate_generated_agent_mock_subagents,
    _ensure_async_delegate_wait,
    _has_delegate_then_wait,
)
from clasharness.scenario.llm_client import ScenarioLLMError
from unittest.mock import MagicMock, patch


class MockConflictLLMClient:

  def audit_image_fidelity(
      self, *, image_bytes: bytes, mime_type: str, expected_token: str,
      context: str = "",
  ) -> dict[str, Any]:
    # Offline mock: no real vision model. Report a benign skip so the fidelity
    # audit never downgrades a decisive asset in tests that do not target it
    # (and to satisfy the ScenarioLLMClient protocol).
    return {"legible": True, "skipped": True, "reason": "mock client"}

  def generate_layer(
      self, layer: str, payload: dict[str, Any]
  ) -> dict[str, Any]:
    target_rounds = payload.get("seed", {}).get("target_rounds", 5)
    # Instruction consistency + self-repair layers: default to "instruction
    # already directs a write" so ``_reconcile_instruction_with_checks`` is a
    # no-op in tests (leaving generated instructions untouched). Tests that
    # exercise the repair path inject their own client.
    if layer == "audit_instruction_write_directive":
      return {"directs_write": True}
    if layer == "repair_instruction_write_directive":
      return {"updated_instruction": ""}
    if layer == "conflict_layer0a_screenplay":
      return {
          "summary": "Mock global phase narrative outline.",
          "screenplay_outline": {
              "conflict_arc": "mock conflict arc",
              "character_cast": [{
                  "role_name": "assistant",
                  "dramatic_role": "protagonist",
                  "stance": "mock stance",
                  "motivation": "mock motivation"
              }],
              "narrative_conflict_triggers": [{
                  "conflict_name": "mock_conflict",
                  "description": "mock conflict description",
                  "workflow_block": "mock workflow block",
                  "resolution_cues": "mock cues",
                  "affected_participants": ["assistant"],
                  "evidence_sources": ["workspace_index.md"]
              }]
          }
      }
    elif layer == "conflict_layer0a_cast":
      return {
          "summary": "Mock global phase narrative outline.",
          "screenplay_outline": {
              "conflict_arc": "mock conflict arc",
              "character_cast": [{
                  "role_name": "assistant",
                  "dramatic_role": "protagonist",
                  "stance": "mock stance",
                  "motivation": "mock motivation"
              }]
          }
      }
    elif layer == "conflict_layer0a_triggers":
      return {
          "narrative_conflict_triggers": [{
              "conflict_name": "mock_conflict",
              "description": "mock conflict description",
              "workflow_block": "mock workflow block",
              "resolution_cues": "mock cues",
              "affected_participants": ["assistant"],
              "evidence_sources": ["workspace_index.md"]
          }]
      }
    elif layer == "conflict_layer0b_timeline":
      phases = []
      num_phases = (target_rounds + 1) // 2
      for p_idx in range(1, num_phases + 1):
        start = (p_idx - 1) * 2 + 1
        end = min(p_idx * 2, target_rounds)
        timeline_screenplay = []
        for r_idx in range(start, end + 1):
          timeline_screenplay.append({
              "round": r_idx,
              "narrative_beat": f"Beat for round {r_idx}",
              "active_conflict_names": ["mock_conflict"],
              "character_claims": [{"role_name": "assistant", "claim": "mock claim"}],
              "vlm_evidence_requirement": "mock VLM requirement",
              "dialogue_conflict": "mock dialogue conflict"
          })
        phases.append({
            "phase_id": f"p{p_idx:02d}",
            "start_round": start,
            "end_round": end,
            "theme": "detection" if p_idx == 1 else "investigation",
            "description": f"Mock Phase {p_idx}",
            "timeline_screenplay": timeline_screenplay
        })
      return {"phases": phases}
    elif layer == "conflict_layer0c_plans":
      return {
          "global_assets_plan": [{
              "asset_id": "ops_dashboard_asset_bg_01",
              "available_from_round": 1,
              "kind": "chart",
              "purpose": "mock chart",
              "platform_id": "ops_dashboard",
          }],
          "global_updates_plan": [{
              "update_id": "u_r01_01",
              "before_round": "r01",
              "purpose": "mock update",
          }],
          "global_interactions_plan": [{
              "interaction_id": "inter_01",
              "active_subagent": "assistant",
              "interaction_type": "mock_interaction",
              "purpose": "mock purpose",
          }],
      }
    # Tier 3.3 (design 6.3): the three global plans split into single-plan calls.
    elif layer == "conflict_layer0c_assets_plan":
      return {
          "platforms": ["ops_dashboard", "physical_datacenter", "status_page"],
          "global_assets_plan": [{
              "asset_id": "ops_dashboard_asset_bg_01",
              "available_from_round": 1,
              "kind": "chart",
              "purpose": "mock chart",
              "platform_id": "ops_dashboard",
          }],
      }
    elif layer == "conflict_layer0c_updates_plan":
      return {
          "global_updates_plan": [{
              "update_id": "u_r01_01",
              "before_round": "r01",
              "purpose": "mock update",
          }],
      }
    elif layer == "conflict_layer0c_interactions_plan":
      return {
          "global_interactions_plan": [{
              "interaction_id": "inter_01",
              "active_subagent": "assistant",
              "interaction_type": "mock_interaction",
              "purpose": "mock purpose",
          }],
      }
    elif layer == "conflict_layer0_expand_phase_timeline":
      phase_id = payload.get("current_phase_id", "p01")
      global_phases = payload.get("global_phases", {}).get("phases", [])
      curr_phase = [p for p in global_phases if p.get("phase_id") == phase_id]
      start = curr_phase[0].get("start_round", 1) if curr_phase else 1
      end = curr_phase[0].get("end_round", 2) if curr_phase else 2

      timeline = []
      for i in range(start, end + 1):
        timeline.append({
            "round_id": f"r{i:02d}",
            "phase": "action",
            "description": f"Mock expanded round outline for r{i:02d}",
            "active_participants": ["Ops_Agent"],
            "introduced_anchors": ["external_ingress_1015"] if i == 1 else [],
        })

      return {"timeline": timeline}
    elif layer == "conflict_layer0_expand_phase_assets":
      return {"assets": []}
    elif layer == "conflict_layer1_agents":
      return {
          "mock_subagents": [{
              "agent_role": "assistant",
              "mock_response": "I have created the file.",
              "virtual_duration_seconds": 15,
              "state_effects": [],
          }],
          "agents": [{
              "agent_role": "assistant",
              "agent_name": "Assistant Subagent",
              "capabilities": ["file_read"],
              "private_knowledge": ["secret knowledge"],
              "virtual_duration_seconds": 15,
          }],
      }
    elif layer == "conflict_layer1_agent_roster":
      return {
          "agents": [{
              "agent_role": "assistant",
              "agent_name": "Assistant Subagent",
              "capabilities": ["file_read"],
              "private_knowledge": ["secret knowledge"],
              "virtual_duration_seconds": 15,
          }],
      }
    elif layer == "conflict_layer1_mock_subagents":
      return {
          "mock_subagents": [{
              "agent_role": "assistant",
              "mock_response": "I have created the file.",
              "virtual_duration_seconds": 15,
              "state_effects": [],
          }],
      }
    elif layer == "conflict_layer1_interactions":
      return {
          "summary": "Mock summary with preference_rule: trust chart",
          "interactions": [{
              "interaction_type": "mock_interaction",
              "active_subagent": "assistant",
              "trigger_evidence": "external_ingress_1015",
              "trigger_source_file": "config/edge_proxy.yaml",
              "resolution_action": "nothing",
              "dependency_steps": ["file read config/edge_proxy.yaml"],
          }],
      }
    elif layer == "conflict_layer1b_files":
      return {
          "workspace": {
              "files": [{
                  "path": "config/edge_proxy.yaml",
                  "content": "redirect_target: default_upstream",
              }],
              "notes": {},
          }
      }
    elif layer == "conflict_layer1b_sessions":
      return {
          "sessions": [{
              "session_id": "slack_ops_dev_bridge",
              "kind": "text",
              "channel": "general",
              "messages": [{
                  "message_id": "m1",
                  "timestamp": "2023-10-27T10:00:00Z",
                  "sender": "Ops_Agent",
                  "text": "Hello",
                  "attachments": [],
              }],
          }]
      }
    elif layer == "conflict_layer1b_assets":
      return {
          "assets": [
              {
                  "id": "ops_dashboard_asset_bg_01",
                  "kind": "chart",
                  "generator": "code_chart",
                  "preset_id": "preset_bar_chart",
                  "path": "charts/ops_dashboard_initial_bg.png",
                  "mime_type": "image/png",
                  "available_from_round": 1,
                  "anchor_ids": ["external_ingress_1015"],
                  "description": "API Ingress Chart",
                  "caption": (
                      "API Ingress throughput shows a sudden drop to zero"
                      " starting from 10:15 UTC. The error rate spiked"
                      " immediately following this drop."
                  ),
                  "ocr_text": "Ingress 0",
                  "provider_metadata": {
                      "chart_type": "bar_chart",
                      "chart_rows": [
                          {
                              "label": "10:00",
                              "value": 1200.0,
                              "anchor": "external_ingress_1015",
                          },
                          {
                              "label": "10:05",
                              "value": 1150.0,
                              "anchor": "external_ingress_1015",
                          },
                          {
                              "label": "10:10",
                              "value": 1210.0,
                              "anchor": "external_ingress_1015",
                          },
                          {
                              "label": "10:15",
                              "value": 0.0,
                              "anchor": "external_ingress_1015",
                          },
                          {
                              "label": "10:20",
                              "value": 0.0,
                              "anchor": "external_ingress_1015",
                          },
                          {
                              "label": "10:25",
                              "value": 0.0,
                              "anchor": "external_ingress_1015",
                          },
                          {
                              "label": "10:30",
                              "value": 0.0,
                              "anchor": "external_ingress_1015",
                          },
                      ],
                  },
                  "source_type": "chart_spec",
                  "platform_id": "ops_dashboard",
              },
              {
                  "id": "physical_datacenter_asset_bg_02",
                  "kind": "image",
                  "generator": "google_image",
                  "path": "images/physical_datacenter_server_rack.png",
                  "mime_type": "image/png",
                  "available_from_round": 1,
                  "anchor_ids": ["EVIDENCE:RACK_C2"],
                  "description": (
                      "Photo of the server rack showing warning light"
                  ),
                  "caption": "Rack C2 status warning lamp is lit",
                  "prompt": (
                      "Close up photo of a server rack C2 in datacenter with a"
                      " flashing orange warning light next to a network router"
                      " module"
                  ),
                  "source_type": "image_generation",
                  "platform_id": "physical_datacenter",
              },
          ],
      }
    elif layer == "conflict_layer3_round_narrative":
      round_idx = payload.get("slot", {}).get("round_index", 1)
      round_id = f"r{round_idx:02d}"
      return {
          "rounds": [{
              "round_id": round_id,
              "round_type": "action",
              "instruction": "Verify config and send message",
              "update_ids": [f"u_{round_id}_01"],
              "mock_subagents": [{
                  "agent_role": "assistant",
                  "mock_response": "Subagent response",
                  "virtual_duration_seconds": 15,
                  "state_effects": [],
              }],
          }],
      }
    elif layer == "conflict_layer3_round_instruction":
      # Tier 1.2 split: instruction prose ONLY (no mock_subagents).
      round_idx = payload.get("slot", {}).get("round_index", 1)
      round_id = f"r{round_idx:02d}"
      return {
          "rounds": [{
              "round_id": round_id,
              "round_type": "action",
              "instruction": "Verify config and send message",
              "update_ids": [f"u_{round_id}_01"],
          }],
      }
    elif layer == "conflict_layer3_round_mocks":
      # Tier 1.2 split: mock subagents ONLY, with an executable state_effect so
      # decision-4 command validation exercises a real (valid) command.
      return {
          "mock_subagents": [{
              "agent_role": "assistant",
              "mock_response": "Subagent response",
              "virtual_duration_seconds": 15,
              "state_effects": ["file create --path workspace/note.txt --content ok"],
          }],
      }
    elif layer == "conflict_layer3_round_updates":
      round_idx = payload.get("slot", {}).get("round_index", 1)
      round_id = f"r{round_idx:02d}"
      return {
          "updates": [{
              "update_id": f"u_{round_id}_01",
              "trigger_virtual_time": 30,
              "workspace_adds": [{
                  "path": "config/edge_proxy.yaml",
                  "content": "redirect_target: default_upstream",
              }],
              "session_appends": [],
              "session_redactions": [],
          }],
      }
    elif layer == "conflict_layer3_round_assets":
      return {
          "assets": [],
      }
    elif layer == "conflict_layer3_round_resolution":
      # Difficulty layer: the decisive evidence + concrete value for this round.
      # resolved_value is a substring of the round's workspace file content
      # (config/edge_proxy.yaml -> "redirect_target: default_upstream") so the
      # deterministic file_check derived from a `file read` stays gold-solvable,
      # and the anchor is one already carried by a visible asset.
      return {
          "decisive_anchor": "EVIDENCE:RACK_C2",
          "resolved_value": "default_upstream",
          # The fixture used to omit this, and the confidentiality gate used to
          # let an undeclared locus through unscanned. Both were wrong: this IS
          # a file-grounded round, and saying so is what makes its value's
          # presence in the workspace file legitimate rather than a leak. With
          # every locus now scanned, an undeclared round is rejected -- see
          # TestValidateRoundConfidentiality.
          "resolved_value_locus": "file:config/edge_proxy.yaml",
          # ③: short free-form verdict label, distinct from resolved_value.
          "decision_outcome": "restore default upstream routing",
      }
    elif layer == "conflict_layer3_round_outline":
      return {
          "outline": [
              {"step_number": 1, "description": "Verify config", "expected_check": "Checked config"},
              {"step_number": 2, "description": "List files", "expected_check": "Listed files"},
              {"step_number": 3, "description": "Verify DB status", "expected_check": "Checked DB status"},
              {"step_number": 4, "description": "Inspect messages", "expected_check": "Reviewed messages"},
              {"step_number": 5, "description": "Apply fix to config", "expected_check": "Config updated"},
              {"step_number": 6, "description": "Re-verify config after fix", "expected_check": "Re-checked config"}
          ]
      }
    elif layer == "conflict_layer3_round_steps":
      round_idx = payload.get("slot", {}).get("round_index", 1)
      round_id = f"r{round_idx:02d}"
      allowed_families = payload.get("seed", {}).get("allowed_tool_families")
      
      steps = []
      if allowed_families is not None:
        families = {normalize_family_name(f) for f in allowed_families}
        if "file" in families:
          steps.append({
              "command": "file read --path config/edge_proxy.yaml",
              "checker": {"type": "file_check", "path": "config/edge_proxy.yaml", "contains": "default_upstream"}
          })
          steps.append({
              "command": "file list",
              "checker": None
          })
        if "message" in families:
          steps.append({
              "command": "message send --channel general --target slack_ops_dev_bridge --message \"loopback\"",
              "checker": {"type": "effect", "effect_type": "messages_sent", "condition": "field_semantic_match", "expected": {"field": "message", "criterion": "send loopback"}}
          })
        if "agent" in families:
          steps.append({
              "command": "agent delegate --agent assistant --instruction 'verify status'",
              "checker": None
          })
        if "asset" in families:
          steps.append({
              "command": "asset list",
              "checker": None
          })
          steps.append({
              "command": "asset open --id ops_dashboard_asset_bg_01",
              "checker": {"type": "effect", "effect_type": "assets_accessed", "condition": "field_contains", "expected": {"field": "id", "value": "ops_dashboard_asset_bg_01"}}
          })
      else:
        steps = [
            {"command": "file list", "checker": None},
            {"command": "file read --path config/edge_proxy.yaml", "checker": {"type": "file_check", "path": "config/edge_proxy.yaml", "contains": "default_upstream"}},
            {"command": "message send --channel general --target slack_ops_dev_bridge --message \"loopback\"", "checker": None}
        ]
      steps.append({"command": "wait 1", "checker": None})
      steps.append({"command": "DONE", "checker": None})
      return {"steps": steps}

    elif layer == "conflict_layer3_round_commands":
      # Tier 1.1 split: emit ONLY command lines (no checkers). Mirrors the
      # command set of the legacy round_steps branch above so downstream
      # deterministic scaffolding produces equivalent checkers.
      allowed_families = payload.get("seed", {}).get("allowed_tool_families")
      commands: list[str] = []
      if allowed_families is not None:
        families = {normalize_family_name(f) for f in allowed_families}
        if "file" in families:
          commands.append("file read --path config/edge_proxy.yaml")
          commands.append("file list")
        if "message" in families:
          commands.append("message send --channel general --target slack_ops_dev_bridge --message \"loopback\"")
        if "agent" in families:
          commands.append("agent delegate --agent assistant --instruction 'verify status'")
        if "asset" in families:
          commands.append("asset list")
          commands.append("asset open --id ops_dashboard_asset_bg_01")
      else:
        commands = [
            "file list",
            "file read --path config/edge_proxy.yaml",
            "message send --channel general --target slack_ops_dev_bridge --message \"loopback\"",
        ]
      # Final verification: end on an inspect (re-read of a changed artifact) after
      # the last action so require_final_verification is satisfied. Pick an inspect
      # command from an allowed family.
      if allowed_families is not None:
        _fv_families = {normalize_family_name(f) for f in allowed_families}
        if "file" in _fv_families:
          commands.append("file read --path config/edge_proxy.yaml")
        elif "asset" in _fv_families:
          commands.append("asset open --id ops_dashboard_asset_bg_01")
      else:
        commands.append("file read --path config/edge_proxy.yaml")
      commands.append("wait 1")
      commands.append("DONE")
      return {
          "commands": [
              {"step_number": i + 1, "command": c}
              for i, c in enumerate(commands)
          ]
      }

    elif layer == "conflict_layer3_round_checkers":
      # Tier 1.1 split: fill ONLY the requested semantic fields. The pipeline
      # tells us exactly which step_numbers/fields it needs via semantic_targets.
      targets = payload.get("slot", {}).get("semantic_targets", []) or []
      fields = []
      for target in targets:
        step_number = target.get("step_number")
        field = target.get("field")
        if field == "criterion":
          fields.append({
              "step_number": step_number,
              "criterion": "confirms the current operational status with the peer",
          })
        elif field == "contains":
          fields.append({
              "step_number": step_number,
              "contains": "default_upstream",
          })
      return {"fields": fields}

    elif layer == "vlm_asset_detail":
      asset_id = payload.get("slot", {}).get("asset_id")
      if asset_id == "ops_dashboard_asset_bg_01":
        return {
            "description": "Enriched API Ingress Chart",
            "caption": "Ingress is zero",
            "ocr_text": "Ingress 0",
            "prompt": "A modern clean bar chart showing API Gateway ingress traffic at 0",
            "render_code": (
                "import matplotlib.pyplot as plt\n"
                "fig, ax = plt.subplots()\n"
                "ax.bar(['Ingress'], [0.0], color='red')\n"
                "ax.set_title('Dynamic Ingress Chart')\n"
                "plt.savefig(OUTPUT_PATH)\n"
                "plt.close(fig)\n"
            ),
        }
      else:
        return {
            "description": "Mock description",
            "caption": "Mock caption",
            "ocr_text": "Mock OCR",
        }
    elif layer == "generate_vlm_layout_spec":
      return {
          "ui_shell_type": payload.get("suggested_shell_type", "monitoring_dashboard"),
          "color_theme": payload.get("suggested_theme", "light"),
          "system_name": "OpenClaw Test System Console",
          "primary_color": payload.get("suggested_primary_color", "blue"),
          "visual_style": payload.get("suggested_visual_style", "modern_clean"),
          "style_details": {
              "font_family": "sans-serif-modern",
              "font_size_scale": "default",
              "border_style": "solid_thin",
              "border_radius": "rounded-lg",
              "shadow_type": "soft-modern",
              "header_style": "solid",
              "display_effect": "none",
              "divider_style": "solid",
              "background_texture": "solid",
              "custom_css_rules": []
          },
          "platform_context": {
              "node_hostname": "test-node-01.openclaw",
              "operator_email": "ops@openclaw.internal"
          },
          "grid_layout": {},
          "common_header_html": "<header>Mock Header</header>",
          "common_sidebar_html": "<aside>Mock Sidebar</aside>",
          "common_footer_html": "<footer>Mock Footer</footer>",
          "common_styles_css": "body { margin: 0; }"
      }
    elif layer == "generate_vlm_redundant_noise":
      return {
          "sidebar_content": "Sidebar dummy content.",
          "terminal_logs": "System initialized successfully.",
          "alerts_and_metrics": "System status: OK"
      }
    elif layer == "compose_vlm_html":
      return {
          "html_content": "<html><body><h1>Mock VLM HTML Page</h1></body></html>"
      }
    return {}


class TestConflictPipeline(unittest.TestCase):

  def test_pipeline_integration(self):
    seed = ConflictVLMSeed(
        slug="test_conflict_01",
        domain="system",
        title="Test Conflict",
        target_rounds=1,
        core_conflict="Time delay breaks assumptions.",
        conflict_taxonomy=["temporal_staleness", "subagent_delay"],
        participants=["owner", "assistant"],
        required_constraints=[],
        allowed_tool_families=["agents", "file", "message"],
        min_round_steps=4,
        require_final_verification=False,
    )
    with tempfile.TemporaryDirectory() as tmp:
      tmp_path = Path(tmp)
      client = MockConflictLLMClient()
      conflict_spec, _ = materialize_conflict_scenario(
          seed,
          client,
          output_dir=tmp_path,
          image_provider="mock",
          chart_provider="mock",
      )

      print("MOCK TEST updates:", conflict_spec.updates)
      print(
          "MOCK TEST rounds mock subagents:",
          [r.mock_subagents for r in conflict_spec.rounds],
      )

      # Validation should pass
      validate_conflict_scenario_strict(conflict_spec)

      self.assertEqual(conflict_spec.scenario_slug, "test_conflict_01")
      self.assertEqual(len(conflict_spec.updates), 1)
      self.assertEqual(conflict_spec.updates[0].trigger_virtual_time, 30)
      self.assertEqual(len(conflict_spec.rounds), 1)
      self.assertEqual(len(conflict_spec.assets), 2)
      self.assertEqual(conflict_spec.assets[0].id, "ops_dashboard_asset_bg_01")
      self.assertIn(
          "import matplotlib.pyplot", conflict_spec.assets[0].render_code
      )
      self.assertTrue(
          conflict_spec.assets[0].render_source_path.endswith(
              "ops_dashboard_asset_bg_01.render.py"
          )
      )
      self.assertEqual(conflict_spec.assets[1].id, "physical_datacenter_asset_bg_02")

      # Check global mock subagent applied to round
      self.assertEqual(len(conflict_spec.rounds[0].mock_subagents), 1)
      self.assertEqual(
          conflict_spec.rounds[0].mock_subagents[0].agent_role, "assistant"
      )
      self.assertEqual(
          conflict_spec.rounds[0].mock_subagents[0].virtual_duration_seconds, 15
      )

  def _materialize(self, style_seed: int = 0):
    seed = ConflictVLMSeed(
        slug="test_conflict_01",
        domain="system",
        title="Test Conflict",
        target_rounds=1,
        core_conflict="Time delay breaks assumptions.",
        conflict_taxonomy=["temporal_staleness", "subagent_delay"],
        participants=["owner", "assistant"],
        required_constraints=[],
        allowed_tool_families=["agents", "file", "message"],
        min_round_steps=4,
        require_final_verification=False,
    )
    with tempfile.TemporaryDirectory() as tmp:
      spec, _ = materialize_conflict_scenario(
          seed,
          MockConflictLLMClient(),
          output_dir=Path(tmp),
          image_provider="mock",
          chart_provider="mock",
          style_seed=style_seed,
      )
    return seed, spec

  def test_wd8_conflict_assets_carry_coherent_ssot_styling(self):
    # UT-WD8-06: the conflict materializer's per-platform styling loop draws from
    # the SSoT sampler and lands where the template reads it
    # (custom_overrides.styling). Every chart asset must carry a coherent,
    # library-valid style bundle -- not the retired 2-field STYLE_PRESETS pair.
    seed, spec = self._materialize(style_seed=0)
    chart_assets = [a for a in spec.assets if a.kind == "chart"]
    self.assertTrue(chart_assets, "expected at least one chart asset")

    for asset in chart_assets:
      styling = asset.custom_overrides.get("styling", {})
      # All four coherent axes are present and library-valid.
      self.assertIn(styling.get("theme"), da.COLOR_THEMES)
      self.assertIn(styling.get("primary_color"), da.PRIMARY_COLORS)
      self.assertTrue(da.has_archetype(styling.get("visual_style", "")))
      self.assertIsInstance(styling.get("style_details"), dict)
      self.assertTrue(styling["style_details"])
      # Tokens stay coherent with the named archetype: signature (non-varying)
      # axes match the preset exactly, so no cross-archetype token bleed.
      arch = da.get_archetype(styling["visual_style"])
      for axis in da.STYLE_DETAIL_AXES:
        if axis not in arch.variation_ranges:
          self.assertEqual(styling["style_details"][axis], arch.token_preset[axis])
      # The bundle is exactly what the SSoT sampler yields for this platform key.
      pid = asset.platform_id or "default"
      want = da.sample_platform_design(f"{seed.slug}_{pid}_0")
      self.assertEqual(styling["theme"], want["color_theme"])
      self.assertEqual(styling["primary_color"], want["primary_color"])
      self.assertEqual(styling["visual_style"], want["visual_style"])
      self.assertEqual(styling["style_details"], want["style_details"])

  def test_wd8_conflict_styling_is_reproducible_for_pinned_seed(self):
    # UT-WD8-07: a pinned style_seed reproduces the whole platform look through
    # the full conflict pipeline (WD5 reproducibility, end-to-end), while a
    # different seed is a real knob that can change it.
    def _looks(style_seed: int):
      _, spec = self._materialize(style_seed=style_seed)
      return {
          a.platform_id: a.custom_overrides.get("styling")
          for a in spec.assets
          if a.kind == "chart"
      }

    self.assertEqual(_looks(0), _looks(0))  # deterministic for a pinned seed
    # The seed is wired through: sampling the SSoT with different keys yields
    # more than one look across seeds (guards against a hardcoded constant).
    across = {
        da.sample_platform_design(f"test_conflict_01_ops_dashboard_{s}")["visual_style"]
        + da.sample_platform_design(f"test_conflict_01_ops_dashboard_{s}")["color_theme"]
        for s in range(12)
    }
    self.assertGreater(len(across), 1)


class TestConflictStrictValidator(unittest.TestCase):

  def setUp(self):
    seed = ConflictVLMSeed(
        slug="test_conflict_01",
        domain="system",
        title="Test Conflict",
        target_rounds=1,
        core_conflict="Time delay breaks assumptions.",
        conflict_taxonomy=["temporal_staleness", "subagent_delay"],
        participants=["owner", "assistant"],
        required_constraints=[],
        allowed_tool_families=["agents", "file", "message"],
        min_round_steps=4,
        require_final_verification=False,
    )
    with tempfile.TemporaryDirectory() as tmp:
      tmp_path = Path(tmp)
      client = MockConflictLLMClient()
      self.valid_spec, _ = materialize_conflict_scenario(
          seed,
          client,
          output_dir=tmp_path,
          image_provider="mock",
          chart_provider="mock",
      )

  def test_valid_spec_passes(self):
    validate_conflict_scenario_strict(self.valid_spec)

  def test_zero_duration_mock_fails_check_three(self):
    # Documents the failure the Option 1 clamp prevents upstream: a single
    # non-positive mock duration in a subagent_delay scenario fails strict
    # check #3 (the LLM emitting 0 for a helper mock is what silently dropped
    # sla_support_response versions post-materialization).
    spec = copy.deepcopy(self.valid_spec)
    mock_round = next(r for r in spec.rounds if r.mock_subagents)
    mock_round.mock_subagents[0].virtual_duration_seconds = 0
    with self.assertRaises(ConflictValidationError) as ctx:
      validate_conflict_scenario_strict(spec)
    self.assertIn("duration <= 0", str(ctx.exception))

  def test_asset_id_leakage_fails(self):
    spec = copy.deepcopy(self.valid_spec)
    spec.rounds[0].visible_asset_ids = ["asset_bg_01"]
    spec.rounds[0].instruction = "Check the asset_bg_01 chart first."

    with self.assertRaises(ConflictValidationError) as ctx:
      validate_conflict_scenario_strict(spec)
    self.assertIn(
        "instruction leaks visible asset ID: 'asset_bg_01'", str(ctx.exception)
    )

  def test_future_round_leakage_fails(self):
    spec = copy.deepcopy(self.valid_spec)
    spec.rounds[0].round_id = "r01"
    spec.rounds[0].instruction = "Wait until round r02 to complete."

    with self.assertRaises(ConflictValidationError) as ctx:
      validate_conflict_scenario_strict(spec)
    self.assertIn(
        "instruction leaks future round references", str(ctx.exception)
    )

  def test_current_round_reference_passes(self):
    spec = copy.deepcopy(self.valid_spec)
    spec.rounds[0].round_id = "r01"
    spec.rounds[0].instruction = "This is round r01 instruction."
    validate_conflict_scenario_strict(spec)

  def test_invalid_state_effect_command_fails(self):
    # Decision 4: a prose / non-spec state_effect must be caught as a final gate.
    spec = copy.deepcopy(self.valid_spec)
    self.assertTrue(spec.rounds[0].mock_subagents)
    # Runtime keeps bare-string CLI effects; declare list[Any] so this
    # intentionally-invalid prose string satisfies the declared field type.
    prose_effects: list[Any] = ["frobnicate the widget"]
    spec.rounds[0].mock_subagents[0].state_effects = prose_effects
    with self.assertRaises(ConflictValidationError) as ctx:
      validate_conflict_scenario_strict(spec)
    self.assertIn("not an executable command", str(ctx.exception))

  def test_file_write_dict_state_effect_passes(self):
    # A structured file-write effect is a valid runtime primitive (not a CLI
    # command) and must NOT be rejected by command validation.
    spec = copy.deepcopy(self.valid_spec)
    self.assertTrue(spec.rounds[0].mock_subagents)
    spec.rounds[0].mock_subagents[0].state_effects = [
        {"type": "file_create", "path": "workspace/x.txt", "content": "hi"}
    ]
    validate_conflict_scenario_strict(spec)

  # --- I1 locus-aware confidentiality backstop (Section 5) -----------------
  # The mock builds a FILE-grounded round: the decisive value lives in a
  # workspace file and no asset locus is declared. The locus-aware net must
  # permit that (text round) yet still catch it when the SAME value is declared
  # asset-grounded (vision round), and must always reject an instruction leak.

  def test_confidentiality_precondition(self):
    # Guards the locus tests below against silent drift in the mock fixture.
    rs = self.valid_spec.rounds[0]
    self.assertEqual(rs.resolved_value, "default_upstream")
    # The mock declares the file it is grounded in. It used to declare nothing
    # and rely on the confidentiality gate exempting unnamed loci; that
    # exemption is deleted, so an honest declaration is now required.
    self.assertEqual(rs.resolved_value_locus, "file:config/edge_proxy.yaml")
    updates_text = " ".join(
        str(add.get("content", ""))
        for upd in self.valid_spec.updates
        if upd.update_id in rs.update_ids
        for add in upd.workspace_adds
    )
    self.assertIn("default_upstream", updates_text)

  def test_nonasset_locus_value_in_workspace_file_passes(self):
    # A DECLARED file locus is legitimately text-grounded: the value living in
    # the file it names is not a leak. The comment here used to add "empty
    # locus (as generated)" as an equally valid case; that is no longer true
    # and was never a good idea -- an undeclared locus is indistinguishable
    # from a leak, so the confidentiality gate now rejects it and asks for the
    # declaration. The strict validator is the LAST net, not the first, so it
    # stays permissive; the gate upstream is where the declaration is forced.
    validate_conflict_scenario_strict(self.valid_spec)
    spec = copy.deepcopy(self.valid_spec)
    spec.rounds[0].resolved_value_locus = "file:config/edge_proxy.yaml"
    validate_conflict_scenario_strict(spec)

  def test_asset_locus_value_in_workspace_file_fails(self):
    # Re-declared as a VISION round: the value must live only in the pixels, so
    # its presence in the agent-visible workspace file IS a difficulty collapse.
    spec = copy.deepcopy(self.valid_spec)
    spec.rounds[0].resolved_value_locus = "asset:img_A"
    with self.assertRaises(ConflictValidationError) as ctx:
      validate_conflict_scenario_strict(spec)
    self.assertIn("leaks decisive resolved_value", str(ctx.exception))

  def test_instruction_leak_fails_for_any_locus(self):
    # The instruction check is UNIVERSAL: even a file-grounded round must not
    # spell out its own answer in the prompt.
    spec = copy.deepcopy(self.valid_spec)
    spec.rounds[0].resolved_value_locus = "file:config/edge_proxy.yaml"
    spec.rounds[0].instruction = "Set the redirect to default_upstream now."
    with self.assertRaises(ConflictValidationError) as ctx:
      validate_conflict_scenario_strict(spec)
    self.assertIn("instruction leaks decisive", str(ctx.exception))

  # --- base chat session (spec.sessions) leak backstop ---------------------
  # Base sessions are scenario-wide and authored before the round's decisive
  # value is chosen, so the round-scoped ``update`` scan cannot see them. They
  # are rendered into the session JSONL the agent reads, so a verbatim decisive
  # value there makes an asset-locus round text-solvable without the pixels.

  def _leaky_session(self, value: str) -> ScenarioSession:
    return ScenarioSession(
        session_id="ops_chat",
        messages=[
            SessionMessage(
                message_id="m1", timestamp="t0", sender="Avery",
                text="pull up the dashboard when you get a sec"),
            SessionMessage(
                message_id="m2", timestamp="t1", sender="Blair",
                text=f"heads up, the rollback target is {value}"),
        ],
    )

  def test_asset_locus_value_in_base_session_fails(self):
    # VISION round: the value must live only in the pixels, so transcribing it
    # into an agent-visible base chat session IS a difficulty collapse.
    spec = copy.deepcopy(self.valid_spec)
    spec.rounds[0].resolved_value_locus = "asset:img_A"
    spec.sessions = list(getattr(spec, "sessions", []) or [])
    spec.sessions.append(self._leaky_session(spec.rounds[0].resolved_value))
    with self.assertRaises(ConflictValidationError) as ctx:
      validate_conflict_scenario_strict(spec)
    self.assertIn("base chat session", str(ctx.exception))

  def test_nonasset_locus_value_in_base_session_passes(self):
    # File-grounded round: base chat may reference the value (the round is
    # legitimately text-solvable), so the base-session scan must NOT fire.
    spec = copy.deepcopy(self.valid_spec)
    spec.rounds[0].resolved_value_locus = "file:config/edge_proxy.yaml"
    spec.sessions = list(getattr(spec, "sessions", []) or [])
    spec.sessions.append(self._leaky_session(spec.rounds[0].resolved_value))
    validate_conflict_scenario_strict(spec)

  def _scrub_update_leak(self, spec) -> None:
    """Redact the decisive value from the round's agent-visible update text.

    The shared fixture deliberately grounds the value in a workspace file (see
    ``test_confidentiality_precondition``), so re-declaring the round as
    asset-locus trips the *update* arm of the scan no matter what the sessions
    look like. Scrubbing that channel first isolates the base-session arm, so a
    test about sessions fails only for session reasons. Mirrors the keys
    ``_update_visible_text`` actually reads.
    """
    rs = spec.rounds[0]
    placeholder = "the target shown on the dashboard"
    for upd in spec.updates:
      if upd.update_id not in rs.update_ids:
        continue
      for add in getattr(upd, "workspace_adds", None) or []:
        if isinstance(add, dict):
          for key in ("content", "text", "body", "path", "filename", "name"):
            if add.get(key):
              add[key] = str(add[key]).replace(rs.resolved_value, placeholder)
      for msg in getattr(upd, "session_appends", None) or []:
        if isinstance(msg, dict):
          for key in ("content", "text", "message", "body"):
            if msg.get(key):
              msg[key] = str(msg[key]).replace(rs.resolved_value, placeholder)

  def test_asset_locus_clean_base_session_passes(self):
    # A base session that only REFERS to the evidence (no verbatim value) is
    # fine even for a vision round.
    spec = copy.deepcopy(self.valid_spec)
    spec.rounds[0].resolved_value_locus = "asset:img_A"
    self._scrub_update_leak(spec)
    spec.sessions = list(getattr(spec, "sessions", []) or [])
    spec.sessions.append(ScenarioSession(
        session_id="ops_chat",
        messages=[SessionMessage(
            message_id="m1", timestamp="t0", sender="Avery",
            text="read the target straight off the dashboard chart")],
    ))
    validate_conflict_scenario_strict(spec)


class TestProtocolCompletionGuard(unittest.TestCase):
  """M8 CL1 coordination-integrity guard (``_validate_protocol_completion``).

  A seed opting in via ``require_protocol_completion`` must drive its core
  protocol mechanic to completion in the graded gold trajectory; a seed that
  narrates the protocol but never completes it (coordination theatre) is
  flagged. The guard is behaviour-gated: off by default so every other seed is
  unaffected (E6).
  """

  _CONFIRMING = [
      "pairing start --device d1 --ttl 300",
      "pairing show-code --device d1",
      "pairing enter-code --device d1",
      "pairing confirm --device d1",
      "DONE",
  ]
  # Same handshake but it never confirms -> emits no ``pairing_completed`` effect.
  _THEATRE = [
      "pairing start --device d1 --ttl 300",
      "pairing show-code --device d1",
      "message send --to peer --body done",
      "DONE",
  ]

  def _pairing_seed(self, *, opt_in: bool) -> ConflictVLMSeed:
    return ConflictVLMSeed.from_dict({
        "slug": "multi_device_pairing",
        "archetype": "multi_device_pairing",
        "conflict_taxonomy": ["multi_device_pairing", "time_window_expiration"],
        "allowed_tool_families": ["pairing"],
        "participants": ["coordinator", "peer"],
        "core_conflict": "ordered pairing handshake before session expiry",
        "require_protocol_completion": opt_in,
    })

  def _spec(self, solution_commands: list[str]) -> ConflictVLMScenarioSpec:
    return ConflictVLMScenarioSpec.from_dict({
        "series": "conflict_vlm_v1",
        "scenario_slug": "multi_device_pairing",
        "conflict_taxonomy": ["multi_device_pairing"],
        "allowed_tool_families": ["pairing"],
        "rounds": [{"round_id": "r01", "solution_commands": solution_commands}],
    })

  def test_seed_parses_opt_in_flag(self):
    self.assertTrue(self._pairing_seed(opt_in=True).require_protocol_completion)
    self.assertFalse(self._pairing_seed(opt_in=False).require_protocol_completion)

  def test_default_flag_is_off(self):
    # E6: a seed that does not declare the key defaults to off.
    seed = ConflictVLMSeed.from_dict({
        "slug": "x", "allowed_tool_families": ["file"],
    })
    self.assertFalse(seed.require_protocol_completion)

  def test_required_effects_resolved_from_archetype_plugin(self):
    self.assertEqual(
        _required_completion_effects(self._pairing_seed(opt_in=True)),
        {"pairing_completed"},
    )

  def test_completion_prefix_reverse_maps_effect_to_command(self):
    self.assertEqual(
        _completion_command_prefixes({"pairing_completed"}),
        {"pairing_completed": ["pairing confirm"]},
    )

  def test_opted_in_with_confirm_passes(self):
    validate_conflict_scenario_strict(
        self._spec(self._CONFIRMING), seed=self._pairing_seed(opt_in=True)
    )

  def test_opted_in_without_confirm_fails(self):
    with self.assertRaises(ConflictValidationError) as ctx:
      validate_conflict_scenario_strict(
          self._spec(self._THEATRE), seed=self._pairing_seed(opt_in=True)
      )
    msg = str(ctx.exception)
    self.assertIn("coordination theatre", msg)
    self.assertIn("pairing_completed", msg)

  def test_non_opted_seed_is_noop_even_without_confirm(self):
    # E6: a seed that does not opt in is unaffected even if it never completes.
    validate_conflict_scenario_strict(
        self._spec(self._THEATRE), seed=self._pairing_seed(opt_in=False)
    )

  def test_missing_seed_is_noop(self):
    # Backward compatible: callers holding only the spec skip the seed-gated check.
    validate_conflict_scenario_strict(self._spec(self._THEATRE))

  def test_completion_may_live_in_any_round(self):
    # The confirm need not be in the first round; any round completing suffices.
    spec = ConflictVLMScenarioSpec.from_dict({
        "series": "conflict_vlm_v1",
        "scenario_slug": "multi_device_pairing",
        "conflict_taxonomy": ["multi_device_pairing"],
        "allowed_tool_families": ["pairing"],
        "rounds": [
            {"round_id": "r01", "solution_commands": self._THEATRE},
            {"round_id": "r02", "solution_commands": self._CONFIRMING},
        ],
    })
    validate_conflict_scenario_strict(spec, seed=self._pairing_seed(opt_in=True))

  def test_opt_in_without_declared_effect_is_flagged(self):
    # Misconfiguration: opts in but the archetype routes to no plugin declaring
    # emits_effects, so there is no protocol completion to enforce.
    seed = ConflictVLMSeed.from_dict({
        "slug": "misconfigured",
        "archetype": "not_a_real_plugin_token",
        "conflict_taxonomy": ["not_a_real_plugin_token"],
        "allowed_tool_families": ["file"],
        "require_protocol_completion": True,
    })
    errors = _validate_protocol_completion(self._spec(self._THEATRE), seed)
    self.assertTrue(errors)
    self.assertIn("no conflict plugin declaring emits_effects", errors[0])


class TestWithinWindowChecker(unittest.TestCase):
  """M8 CL2 temporal grader operator (``EffectChecker`` ``within_window``, Deliv 8).

  Passes iff at least one effect of the type carries a ``virtual_time`` stamp at
  or before the deadline; an optional ``field``/``value`` filter scopes the
  assertion to the decisive (sender-attributed) effect. Fail-closed: an absent,
  unstamped, past-deadline, or wrong-sender effect cannot prove on-time
  completion -- the temporal analogue of the M7 vision-blind ablation (F6).
  """

  SENDER = "Iris_SLA_Monitor"

  def _checker(self, deadline=30, *, scoped=True, lo=None):
    expected: dict[str, Any] = {"max_virtual_time": deadline}
    if lo is not None:
      expected["min_virtual_time"] = lo
    if scoped:
      expected.update({"field": "sender", "value": self.SENDER})
    return EffectChecker(
        effect_type="messages_sent", condition="within_window", expected=expected
    )

  def _state(self, messages):
    return {"effects": {"messages_sent": messages}}

  def _run(self, checker, messages):
    return checker.evaluate(self._state(messages), {})

  def test_on_time_matching_sender_passes(self):
    r = self._run(
        self._checker(30),
        [{"sender": self.SENDER, "message": "compliant", "virtual_time": 30}],
    )
    self.assertTrue(r.passed, r.message)

  def test_late_effect_fails(self):
    # The delegated reply arrived, but after the SLA deadline -> breach.
    r = self._run(
        self._checker(30),
        [{"sender": self.SENDER, "message": "compliant", "virtual_time": 45}],
    )
    self.assertFalse(r.passed, r.message)

  def test_no_matching_effect_fails_closed(self):
    # No-wait ablation: the completion never fired, so no sender effect exists.
    r = self._run(self._checker(30), [])
    self.assertFalse(r.passed, r.message)

  def test_unstamped_effect_fails_closed(self):
    # An effect with no ``virtual_time`` cannot prove on-time completion.
    r = self._run(
        self._checker(30),
        [{"sender": self.SENDER, "message": "compliant"}],
    )
    self.assertFalse(r.passed, r.message)

  def test_sender_scope_excludes_other_effects(self):
    # A different sender's on-time effect (or a solver message with no sender)
    # must NOT satisfy an assertion scoped to the decisive sub-agent's reply.
    r = self._run(
        self._checker(30),
        [
            {"message": "early note from the solver", "virtual_time": 1},
            {"sender": "Someone_Else", "message": "hi", "virtual_time": 2},
        ],
    )
    self.assertFalse(r.passed, r.message)

  def test_unscoped_window_passes_on_any_on_time_effect(self):
    # Without a field filter, any on-time effect of the type satisfies it.
    r = self._run(
        self._checker(30, scoped=False),
        [{"message": "anything", "virtual_time": 5}],
    )
    self.assertTrue(r.passed, r.message)

  def test_boolean_virtual_time_is_not_a_valid_stamp(self):
    # A bool is an int subclass; it must not be accepted as a timestamp.
    r = self._run(
        self._checker(30),
        [{"sender": self.SENDER, "message": "x", "virtual_time": True}],
    )
    self.assertFalse(r.passed, r.message)

  # --- L1 lower bound (min_virtual_time), M8/CL2 no-wait bypass fix --------- #

  def test_before_lower_bound_fails_closed(self):
    # A decisive reply stamped BEFORE the sub-agent could have completed -- e.g.
    # a synchronous Route-2 mirror at virtual_time=0 elicited without waiting --
    # is too early and must not prove on-time completion.
    r = self._run(
        self._checker(30, lo=30),
        [{"sender": self.SENDER, "message": "compliant", "virtual_time": 0}],
    )
    self.assertFalse(r.passed, r.message)

  def test_exactly_at_lower_bound_passes(self):
    # The legitimate delayed completion is stamped at the sub-agent's trigger,
    # which equals the lower bound -- the window is inclusive at both ends.
    r = self._run(
        self._checker(30, lo=30),
        [{"sender": self.SENDER, "message": "compliant", "virtual_time": 30}],
    )
    self.assertTrue(r.passed, r.message)

  def test_inside_bounded_window_passes(self):
    r = self._run(
        self._checker(40, lo=20),
        [{"sender": self.SENDER, "message": "compliant", "virtual_time": 30}],
    )
    self.assertTrue(r.passed, r.message)

  def test_no_lower_bound_still_accepts_early_effect(self):
    # Back-compat: without ``min_virtual_time`` an early on-time effect still
    # passes, so every non-async within_window check is unaffected (E6).
    r = self._run(
        self._checker(30),
        [{"sender": self.SENDER, "message": "compliant", "virtual_time": 0}],
    )
    self.assertTrue(r.passed, r.message)


class TestAsyncWithinWindowInjection(unittest.TestCase):
  """M8 CL2 materializer helper (``_build_async_within_window_check``, Deliv 4).

  Builds a sender-scoped ``within_window`` check bound to the *slowest* mock
  sub-agent (whose latency forces the wait), with the deadline derived from the
  round's max background-event ``trigger_time``. Returns ``None`` when there is
  nothing async to grade.
  """

  def _mock(self, role, dur):
    return {"agent_role": role, "virtual_duration_seconds": dur}

  def _event(self, trigger):
    return {"type": "subagent_completion", "trigger_time": trigger}

  def test_none_without_mock_subagents(self):
    self.assertIsNone(
        _build_async_within_window_check([], [self._event(30)])
    )

  def test_none_without_delayed_event(self):
    self.assertIsNone(
        _build_async_within_window_check([self._mock("Iris", 30)], [])
    )
    # A zero/negative deadline is not a real window either.
    self.assertIsNone(
        _build_async_within_window_check(
            [self._mock("Iris", 30)], [self._event(0)]
        )
    )

  def test_picks_slowest_sender_and_max_trigger_deadline(self):
    check = _build_async_within_window_check(
        [self._mock("Fast_Agent", 10), self._mock("Iris_SLA_Monitor", 30)],
        [self._event(10), self._event(30)],
    )
    self.assertIsNotNone(check)
    self.assertEqual(check["type"], "effect")
    self.assertEqual(check["effect_type"], "messages_sent")
    self.assertEqual(check["condition"], "within_window")
    self.assertEqual(check["expected"]["max_virtual_time"], 30)
    self.assertEqual(check["expected"]["field"], "sender")
    self.assertEqual(check["expected"]["value"], "Iris_SLA_Monitor")
    # L1 lower bound: bound to the decisive sub-agent's own trigger (its
    # virtual_duration_seconds), so a synchronous vt=0 reply cannot satisfy it.
    self.assertEqual(check["expected"]["min_virtual_time"], 30)

  def test_injected_check_grades_reference_and_fails_no_wait(self):
    # End-to-end wiring: the built check passes for the reference trajectory
    # (subagent completion stamped at its trigger_time) and fails closed for a
    # no-wait trajectory (no completion effect).
    check = _build_async_within_window_check(
        [self._mock("Iris_SLA_Monitor", 30)], [self._event(30)]
    )
    checker = EffectChecker.from_spec(check)
    on_time = {
        "effects": {
            "messages_sent": [
                {"sender": "Iris_SLA_Monitor", "message": "compliant",
                 "virtual_time": 30}
            ]
        }
    }
    no_wait = {"effects": {"messages_sent": []}}
    self.assertTrue(checker.evaluate(on_time, {}).passed)
    self.assertFalse(checker.evaluate(no_wait, {}).passed)

  def test_no_wait_synchronous_bypass_fails_closed(self):
    # REGRESSION (P0): before the L1 lower bound, a no-wait solver that DM'd the
    # decisive sub-agent got its reply mirrored synchronously into messages_sent
    # stamped at virtual_time=0. With only an upper bound (0 <= deadline) that
    # bypass PASSED, making `wait` non-load-bearing. The injected check must now
    # reject a decisive-sender effect stamped before the sub-agent's completion.
    check = _build_async_within_window_check(
        [self._mock("Iris_SLA_Monitor", 30)], [self._event(30)]
    )
    checker = EffectChecker.from_spec(check)
    bypass = {
        "effects": {
            "messages_sent": [
                {"sender": "Iris_SLA_Monitor", "message": "compliant",
                 "virtual_time": 0}
            ]
        }
    }
    self.assertFalse(checker.evaluate(bypass, {}).passed)


class TestAsyncWaitContract(unittest.TestCase):
  """M8 CL2 forced-async structural gate (``enforce_async_wait_contract``, Glue #8).

  A seed opting in via ``force_async_wait`` must ship a scenario that can actually
  grade its forced-async mechanic: a delayed mock sub-agent (so the within_window
  check is injected) AND a gold trajectory that delegates-then-waits. Otherwise
  the requirement collapses into "async theatre". Behaviour-gated: off by default
  so every other seed is unaffected (E6).
  """

  _ASYNC_GOLD = [
      "agent delegate --agent Iris_SLA_Monitor --instruction 'verify SLA'",
      "wait 30",
      "message search --channel ops",
      "DONE",
  ]
  # Same commands but the wait precedes the delegate (order violation).
  _WAIT_FIRST = [
      "wait 30",
      "agent delegate --agent Iris_SLA_Monitor --instruction 'verify SLA'",
      "DONE",
  ]
  # Synchronous shortcut: a message send, never a delegate + wait.
  _SYNC_GOLD = [
      "message send --target Iris_SLA_Monitor --message 'verify SLA'",
      "wait 30",
      "DONE",
  ]
  _FAMILIES = ["agent", "message", "file", "asset", "system"]

  def _seed(self, *, opt_in: bool) -> ConflictVLMSeed:
    return ConflictVLMSeed.from_dict({
        "slug": "sla_support_response",
        "archetype": "time_window_expiration",
        "conflict_taxonomy": ["time_window_expiration", "subagent_delay"],
        "allowed_tool_families": self._FAMILIES,
        "participants": ["coordinator", "Iris_SLA_Monitor"],
        "core_conflict": "respond before the SLA window closes",
        "force_async_wait": opt_in,
    })

  def _spec(self, solution_commands, *, delayed=True):
    mocks = (
        [{"agent_role": "Iris_SLA_Monitor", "mock_response": "compliant",
          "virtual_duration_seconds": 30}]
        if delayed
        else []
    )
    return ConflictVLMScenarioSpec.from_dict({
        "series": "conflict_vlm_v1",
        "scenario_slug": "sla_support_response",
        "conflict_taxonomy": ["time_window_expiration", "subagent_delay"],
        "allowed_tool_families": self._FAMILIES,
        "rounds": [{
            "round_id": "r01",
            "solution_commands": solution_commands,
            "mock_subagents": mocks,
        }],
    })

  def test_seed_parses_opt_in_flag(self):
    self.assertTrue(self._seed(opt_in=True).force_async_wait)
    self.assertFalse(self._seed(opt_in=False).force_async_wait)

  def test_default_flag_is_off(self):
    # E6: a seed that does not declare the key defaults to off.
    seed = ConflictVLMSeed.from_dict({
        "slug": "x", "allowed_tool_families": ["file"],
    })
    self.assertFalse(seed.force_async_wait)

  def test_delegate_then_wait_detected(self):
    self.assertTrue(
        _gold_delegates_then_waits(
            self._spec(self._ASYNC_GOLD).rounds[0], self._FAMILIES
        )
    )

  def test_wait_before_delegate_not_detected(self):
    self.assertFalse(
        _gold_delegates_then_waits(
            self._spec(self._WAIT_FIRST).rounds[0], self._FAMILIES
        )
    )

  def test_synchronous_send_not_detected(self):
    self.assertFalse(
        _gold_delegates_then_waits(
            self._spec(self._SYNC_GOLD).rounds[0], self._FAMILIES
        )
    )

  def test_round_has_delayed_subagent(self):
    self.assertTrue(
        _round_has_delayed_subagent(self._spec(self._ASYNC_GOLD).rounds[0])
    )
    self.assertFalse(
        _round_has_delayed_subagent(
            self._spec(self._ASYNC_GOLD, delayed=False).rounds[0]
        )
    )

  def test_opted_in_valid_passes(self):
    enforce_async_wait_contract(
        self._spec(self._ASYNC_GOLD), self._seed(opt_in=True)
    )

  def test_opted_in_without_delayed_subagent_fails(self):
    with self.assertRaises(ConflictValidationError) as ctx:
      enforce_async_wait_contract(
          self._spec(self._ASYNC_GOLD, delayed=False), self._seed(opt_in=True)
      )
    self.assertIn("delayed mock sub-agent", str(ctx.exception))

  def test_opted_in_without_delegate_then_wait_fails(self):
    with self.assertRaises(ConflictValidationError) as ctx:
      enforce_async_wait_contract(
          self._spec(self._SYNC_GOLD), self._seed(opt_in=True)
      )
    self.assertIn("delegates to a sub-agent and THEN waits", str(ctx.exception))

  def test_wait_before_delegate_fails(self):
    with self.assertRaises(ConflictValidationError):
      enforce_async_wait_contract(
          self._spec(self._WAIT_FIRST), self._seed(opt_in=True)
      )

  def test_non_opted_seed_is_noop(self):
    # E6: a seed that does not opt in is unaffected even with no async at all.
    enforce_async_wait_contract(
        self._spec(self._SYNC_GOLD, delayed=False), self._seed(opt_in=False)
    )
    self.assertEqual(
        _validate_async_wait_contract(
            self._spec(self._SYNC_GOLD, delayed=False), self._seed(opt_in=False)
        ),
        [],
    )

  def test_missing_seed_is_noop(self):
    # Backward compatible: callers holding only the spec skip the seed-gated check.
    self.assertEqual(
        _validate_async_wait_contract(self._spec(self._SYNC_GOLD), None), []
    )
    validate_conflict_scenario_strict(self._spec(self._ASYNC_GOLD))

  def test_full_strict_validator_enforces_contract_under_seed(self):
    # The contract is also wired into the post-generation strict validator.
    with self.assertRaises(ConflictValidationError):
      validate_conflict_scenario_strict(
          self._spec(self._SYNC_GOLD), seed=self._seed(opt_in=True)
      )


class TestSlaSeedOptsIntoAsync(unittest.TestCase):
  """Deliv 9: the ``sla_support_response`` seed opts into the CL2 async mechanic."""

  def _load(self):
    # No `.resolve()`: under sandboxed runners `__file__` can be a symlink into
    # a content-addressed store, so resolving it walks out of the staged tree and
    # the seed is not found.
    root = Path(__file__).parents[1]
    path = (
        root
        / "clasharness/data/scenario_seeds/conflict_vlm_v1/sla_support_response.json"
    )
    import json
    return ConflictVLMSeed.from_dict(json.loads(path.read_text()))

  def test_seed_sets_force_async_wait(self):
    self.assertTrue(self._load().force_async_wait)

  def test_seed_declares_subagent_delay_taxonomy(self):
    self.assertIn("subagent_delay", self._load().conflict_taxonomy)


class TestEnsureAsyncDelegateWait(unittest.TestCase):
  """M8/CL2 (F6): the deterministic ``delegate -> wait`` repair used at
  generation time so a ``force_async_wait`` seed does not pure-reject and burn
  every retry when the LLM writes a synchronous ``message send`` instead."""

  ALLOWED = ["asset", "agents", "file", "message", "tasks"]

  def _cmds(self, lines):
    return [{"step_number": i + 1, "command": c} for i, c in enumerate(lines)]

  def _mocks(self):
    # Two sub-agents; the repair must target the SLOWER one (max duration).
    return [
        {"agent_role": "Harper_Support_Lead", "virtual_duration_seconds": 5},
        {"agent_role": "Iris_SLA_Monitor", "virtual_duration_seconds": 40},
    ]

  def test_repairs_synchronous_trajectory(self):
    cmds = self._cmds([
        "asset open --id chart_01",
        "file create --path /reports/sla_triage.md --content 'P1: TCK-9'",
        "message send --channel '#sla' --target Iris_SLA_Monitor --message 'done'",
        "DONE",
    ])
    self.assertFalse(_has_delegate_then_wait(cmds, self.ALLOWED))
    ok = _ensure_async_delegate_wait(cmds, self._mocks(), self.ALLOWED)
    self.assertTrue(ok)
    # Now satisfies the gate.
    self.assertTrue(_has_delegate_then_wait(cmds, self.ALLOWED))
    # Delegates to the slowest sub-agent and waits after it.
    joined = " || ".join(c["command"] for c in cmds)
    self.assertIn("agent delegate --agent Iris_SLA_Monitor", joined)
    # step_numbers are contiguous and DONE stays terminal.
    self.assertEqual(
        [c["step_number"] for c in cmds], list(range(1, len(cmds) + 1))
    )
    self.assertEqual(cmds[-1]["command"], "DONE")

  def test_delegate_precedes_wait_precedes_done(self):
    cmds = self._cmds(["file list", "DONE"])
    _ensure_async_delegate_wait(cmds, self._mocks(), self.ALLOWED)
    kinds = [c["command"].split()[0] for c in cmds]
    d_idx = next(i for i, c in enumerate(cmds) if c["command"].startswith("agent delegate"))
    w_idx = next(i for i, c in enumerate(cmds) if c["command"].lower().startswith("wait "))
    done_idx = next(i for i, c in enumerate(cmds) if c["command"] == "DONE")
    self.assertLess(d_idx, w_idx)
    self.assertLess(w_idx, done_idx)
    # wait covers the slowest sub-agent's completion.
    self.assertEqual(cmds[w_idx]["command"], "wait 40")

  def test_noop_when_already_async(self):
    cmds = self._cmds([
        "agent delegate --agent Iris_SLA_Monitor --instruction 'verify'",
        "wait 40",
        "message send --channel '#sla' --target Iris_SLA_Monitor --message 'x'",
        "DONE",
    ])
    before = [dict(c) for c in cmds]
    ok = _ensure_async_delegate_wait(cmds, self._mocks(), self.ALLOWED)
    self.assertTrue(ok)
    self.assertEqual(cmds, before)  # byte-identical: no second delegate/wait

  def test_returns_false_without_mock_subagents(self):
    cmds = self._cmds(["file list", "DONE"])
    ok = _ensure_async_delegate_wait(cmds, [], self.ALLOWED)
    self.assertFalse(ok)
    self.assertFalse(_has_delegate_then_wait(cmds, self.ALLOWED))


class TestVLMAssetSpec(unittest.TestCase):

  def test_from_dict_assigns_fallback_path_when_empty(self):
    data = {
        "id": "my_test_chart",
        "kind": "chart",
        "path": "",
    }
    spec = VLMAssetSpec.from_dict(data)
    self.assertEqual(spec.path, "charts/my_test_chart.png")

    data_missing = {
        "id": "my_other_chart",
        "kind": "chart",
    }
    spec_missing = VLMAssetSpec.from_dict(data_missing)
    self.assertEqual(spec_missing.path, "charts/my_other_chart.png")

    data_ok = {
        "id": "db_log",
        "kind": "rich_text",
        "path": "logs/db_log.txt",
    }
    spec_ok = VLMAssetSpec.from_dict(data_ok)
    self.assertEqual(spec_ok.path, "logs/db_log.txt")



class TestPipelineValidators(unittest.TestCase):

  def test_validate_expanded_phase_missing_timeline(self):
    with self.assertRaisesRegex(ValueError, "missing 'timeline'"):
      _validate_expanded_phase({}, ["r01"])

  def test_validate_expanded_phase_timeline_empty(self):
    with self.assertRaisesRegex(ValueError, "Timeline must be a non-empty list"):
      _validate_expanded_phase({"timeline": []}, ["r01"])

  def test_validate_expanded_phase_missing_expected_rounds(self):
    payload = {"timeline": [{"round_id": "r01"}]}
    with self.assertRaisesRegex(ValueError, "failed to generate details for expected round 'r02'"):
      _validate_expanded_phase(payload, ["r01", "r02"])

  def test_validate_expanded_phase_missing_asset_path(self):
    payload = {
        "timeline": [{"round_id": "r01"}],
        "assets": [{"id": "ops_asset_a", "path": "", "platform_id": "ops"}]
    }
    with self.assertRaisesRegex(ValueError, "missing 'path'"):
      _validate_expanded_phase(payload, ["r01"])

  def test_validate_expanded_phase_passes(self):
    payload = {
        "timeline": [{"round_id": "r01"}, {"round_id": "r02"}],
        "assets": [{"id": "ops_asset_a", "path": "charts/ops_asset_a.png", "platform_id": "ops"}]
    }
    # Should run without raising any exceptions
    _validate_expanded_phase(payload, ["r01", "r02"])

  def test_validate_generated_round_missing_rounds(self):
    with self.assertRaisesRegex(ValueError, "missing 'rounds'"):
      _validate_generated_round({}, "r01", [], ["file"])

  def test_validate_generated_round_empty_instruction(self):
    payload = {"rounds": [{"instruction": ""}]}
    with self.assertRaisesRegex(ValueError, "Round instruction is empty"):
      _validate_generated_round(payload, "r01", [], ["file"])

  def test_validate_generated_round_asset_id_leak(self):
    # D4: a leaked visible asset ID is now sanitized in place, not rejected.
    payload = {"rounds": [{"instruction": "Check the asset_a visual chart."}]}
    _validate_generated_round_narrative(payload, "r01", ["asset_a"])
    self.assertNotIn("asset_a", payload["rounds"][0]["instruction"])

  def test_validate_generated_round_future_round_leak(self):
    payload = {"rounds": [{"instruction": "Refer to round r02 for results."}]}
    with self.assertRaisesRegex(ValueError, "leaks future round references"):
      _validate_generated_round(payload, "r01", [], ["file"])

  def test_validate_generated_round_delegation_forbidden_word(self):
    payload = {
        "rounds": [{"instruction": "Check log.", "solution_commands": ["file read --path config.yaml", "file list", "wait 1", "DONE"]}],
        "evals": [{
            "checks": [{
                "type": "effect",
                "effect_type": "messages_sent",
                "expected": {"field": "message", "value": "check their logs"}
            }]
        }]
    }
    with self.assertRaisesRegex(ValueError, "contains forbidden word: 'their'"):
      _validate_generated_round(payload, "r01", [], ["file"])

  def test_validate_generated_round_invalid_solution_command(self):
    payload = {
        "rounds": [{"instruction": "Read file.", "solution_commands": ["file read --invalid-flag flag", "file list", "wait 1", "DONE"]}],
    }
    with self.assertRaisesRegex(ValueError, "Invalid command in compiled trajectory: 'file read --invalid-flag flag'"):
      _validate_generated_round(payload, "r01", [], ["file", "asset"])

  def test_validate_generated_round_solution_commands_too_short(self):
    payload = {
        "rounds": [{"instruction": "Read file.", "solution_commands": ["file read --path config.yaml", "DONE"]}],
    }
    with self.assertRaisesRegex(ValueError, "solution_commands too short"):
      _validate_generated_round(payload, "r01", [], ["file"])

  def test_validate_generated_round_passes(self):
    payload = {
        "rounds": [{"instruction": "Check application config.", "solution_commands": ["file read --path config.yaml", "file list", "wait 1", "DONE"]}],
        "evals": [{
            "checks": [{
                "type": "effect",
                "effect_type": "messages_sent",
                "expected": {"field": "message", "value": "verify config"}
            }]
        }]
    }
    _validate_generated_round(payload, "r01", ["asset_bg_01"], ["file", "asset"])

  def test_normalize_and_prefix_assets(self):
    payload = {
        "assets": [
            {
                "id": "ops_latency_chart",
                "path": "charts/ops_latency_chart.png",
                "platform_id": "ops_dashboard"
            },
            {
                "id": "ci_cd_pipeline_build_log",  # Already prefixed
                "path": "logs/ci_cd_pipeline_build_log.png",
                "platform_id": "ci_cd_pipeline"
            }
        ]
    }
    normalized = _normalize_and_prefix_assets(payload)
    assets = normalized["assets"]
    self.assertEqual(assets[0]["id"], "ops_dashboard_ops_latency_chart")
    self.assertEqual(assets[0]["path"], "charts/ops_dashboard_ops_latency_chart.png")
    self.assertEqual(assets[1]["id"], "ci_cd_pipeline_build_log")
    self.assertEqual(assets[1]["path"], "logs/ci_cd_pipeline_build_log.png")

  def test_normalize_and_prefix_assets_sanitizes_path_like_id(self):
    # A path-like id whose leading segment equals the platform basename must not
    # produce a doubled, slash-bearing id (the root cause of phantom
    # ``ops_dashboard_ops_dashboard/...`` assets). It should collapse to a single
    # prefix, slash-free. ``ops_dashboard`` is a valid (non-blocklisted) slug so
    # it survives sanitization and drives the prefix.
    payload = {
        "assets": [
            {
                "id": "ops_dashboard/flight_schedule",
                "path": "charts/flight_schedule.png",
                "platform_id": "ops_dashboard",
            },
            {
                "id": "reports/q3/summary",  # unrelated leading segment
                "path": "charts/summary.png",
                "platform_id": "ops_dashboard",
            },
            {
                # Real gemini-3.5-flash shape: platform basename + nested path +
                # extension (cf. multimodal_temporal_conflict_01_v05).
                "id": "ops_dashboard/assets/devops_dashboard.png",
                "path": "assets/devops_dashboard.png",
                "platform_id": "ops_dashboard",
            },
        ]
    }
    normalized = _normalize_and_prefix_assets(payload)
    assets = normalized["assets"]
    # Leading ``basename(platform_id)/`` segment is dropped, then prefixed once.
    self.assertEqual(assets[0]["id"], "ops_dashboard_flight_schedule")
    # Non-prefix path segments are folded to '_' and then prefixed once.
    self.assertEqual(assets[1]["id"], "ops_dashboard_reports_q3_summary")
    self.assertEqual(assets[2]["id"], "ops_dashboard_assets_devops_dashboard.png")
    for asset in assets:
      self.assertNotIn("/", asset["id"])
      self.assertNotIn("ops_dashboard_ops_dashboard", asset["id"])

  def test_normalize_and_prefix_assets_remaps_blocklisted_platform_word(self):
    # ``workspace`` is a reserved word the model hallucinated as a platform_id in
    # the previous batch, yielding phantom ``workspace_workspace/...`` ids. It is
    # now blocklisted, so it is remapped away (no scenario vocab -> default) and
    # the resulting id is single-prefixed and slash-free -- the doubled pattern
    # can no longer recur on regeneration.
    payload = {
        "assets": [
            {
                "id": "workspace/flight_schedule",
                "path": "charts/flight_schedule.png",
                "platform_id": "workspace",
            },
        ]
    }
    normalized = _normalize_and_prefix_assets(payload)
    asset = normalized["assets"][0]
    self.assertNotEqual(asset["platform_id"], "workspace")
    self.assertNotIn("/", asset["id"])
    self.assertNotIn("workspace_workspace", asset["id"])

  def test_validate_background_payload_missing_prefix(self):
    payload = {
        "workspace": {"files": []},
        "sessions": [],
        "assets": [{
            "id": "ops_latency_chart",  # Missing prefix
            "path": "charts/ops_latency_chart.png",
            "platform_id": "ops_dashboard"
        }]
    }
    with self.assertRaisesRegex(ValueError, "Asset ID 'ops_latency_chart' must start with platform prefix"):
      _validate_background_payload(payload)

  def test_validate_background_payload_passes(self):
    payload = {
        "workspace": {"files": []},
        "sessions": [],
        "assets": [{
            "id": "ops_dashboard_ops_latency_chart",
            "path": "charts/ops_dashboard_ops_latency_chart.png",
            "platform_id": "ops_dashboard"
        }]
    }
    # Should pass without raising exceptions
    _validate_background_payload(payload)

  def test_validate_background_payload_normalizes_workspace_list(self):
    payload = {
        "workspace": [{"path": "config/edge_proxy.yaml", "content": "redirect_target: backend"}],
        "sessions": [],
    }
    # Should normalize in-place and pass without raising exceptions
    _validate_background_payload(payload)
    self.assertIsInstance(payload["workspace"], dict)
    self.assertIn("files", payload["workspace"])
    self.assertEqual(payload["workspace"]["files"][0]["path"], "config/edge_proxy.yaml")

  def test_validate_background_payload_absolute_path_normalizes(self):
    payload = {
        "workspace": {
            "files": [{"path": "/config/edge_proxy.yaml", "content": "redirect_target: backend"}]
        },
        "sessions": [],
    }
    _validate_background_payload(payload)
    self.assertEqual(payload["workspace"]["files"][0]["path"], "config/edge_proxy.yaml")

  def test_validate_background_payload_traversal_fails(self):
    payload = {
        "workspace": {
            "files": [{"path": "config/../../edge_proxy.yaml", "content": "redirect_target: backend"}]
        },
        "sessions": [],
    }
    with self.assertRaisesRegex(ValueError, "contains forbidden directory traversal"):
      _validate_background_payload(payload)

  def test_validate_round_narrative_updates_absolute_path_normalizes(self):
    payload = {
        "rounds": [{"instruction": "Verify status."}],
        "updates": [{
            "update_id": "u_r01_01",
            "workspace_adds": [{"path": "/config/edge_proxy.yaml", "content": "redirect_target: backend"}]
        }]
    }
    _validate_generated_round_narrative(payload, "r01", [])
    self.assertEqual(payload["updates"][0]["workspace_adds"][0]["path"], "config/edge_proxy.yaml")


  def test_validate_round_narrative_assets_missing_prefix(self):
    payload = {
        "rounds": [{"instruction": "Read status."}],
        "assets": [{
            "id": "ops_latency_chart",  # Missing prefix
            "path": "charts/ops_latency_chart.png",
            "platform_id": "ops_dashboard"
        }]
    }
    with self.assertRaisesRegex(ValueError, "Asset ID 'ops_latency_chart' must start with platform prefix"):
      _validate_generated_round_narrative(payload, "r01", [])

  # --- D3: missing workspace file content is coerced, not fatal ---

  def test_validate_workspace_paths_missing_content_defaults_empty(self):
    files = [{"path": ".gitkeep"}]
    _validate_workspace_paths(files, "test")
    self.assertEqual(files[0]["content"], "")

  def test_validate_workspace_paths_none_content_defaults_empty(self):
    files = [{"path": "config/checksum.txt", "content": None}]
    _validate_workspace_paths(files, "test")
    self.assertEqual(files[0]["content"], "")

  def test_validate_workspace_paths_nonstr_content_stringified(self):
    files = [{"path": "data.json", "content": {"k": 1}}]
    _validate_workspace_paths(files, "test")
    self.assertIsInstance(files[0]["content"], str)
    self.assertIn("k", files[0]["content"])

  def test_validate_workspace_paths_preserves_existing_content(self):
    files = [{"path": "a.txt", "content": "hello"}]
    _validate_workspace_paths(files, "test")
    self.assertEqual(files[0]["content"], "hello")

  def test_validate_workspace_paths_still_requires_path(self):
    with self.assertRaisesRegex(ValueError, "missing 'path'"):
      _validate_workspace_paths([{"content": "x"}], "test")

  def test_validate_workspace_paths_still_rejects_traversal(self):
    with self.assertRaisesRegex(ValueError, "directory traversal"):
      _validate_workspace_paths([{"path": "a/../../b.txt", "content": "x"}], "test")

  # --- D4: leaked asset IDs are sanitized in place, not hard-rejected ---

  def test_sanitize_leaked_asset_ids_image(self):
    out = _sanitize_leaked_asset_ids(
        "Compare slack_metrics_chart_20240101_120000.png with the baseline.",
        ["slack_metrics_chart_20240101_120000.png"],
    )
    self.assertNotIn("slack_metrics_chart", out.lower())
    self.assertIn("the attached screenshot", out)

  def test_sanitize_leaked_asset_ids_collapses_doubled_article(self):
    out = _sanitize_leaked_asset_ids(
        "Review the metrics_chart_v1.png and decide.", ["metrics_chart_v1.png"]
    )
    self.assertNotIn("the the", out.lower())
    self.assertEqual(out, "Review the attached screenshot and decide.")

  def test_sanitize_leaked_asset_ids_multiple(self):
    out = _sanitize_leaked_asset_ids(
        "Open a_invoice_2024.pdf and b_ledger.csv now.",
        ["a_invoice_2024.pdf", "b_ledger.csv"],
    )
    self.assertNotIn(".pdf", out)
    self.assertNotIn(".csv", out)

  def test_sanitize_leaked_asset_ids_noop_when_clean(self):
    text = "No leak here, just plain instructions."
    self.assertEqual(_sanitize_leaked_asset_ids(text, ["unused_asset_9.png"]), text)

  def test_neutral_asset_phrase_inference(self):
    self.assertEqual(_neutral_asset_phrase("x_chart.png"), "the attached screenshot")
    self.assertEqual(_neutral_asset_phrase("x_report.pdf"), "the attached document")
    self.assertEqual(_neutral_asset_phrase("x_ledger.csv"), "the attached spreadsheet")
    self.assertEqual(_neutral_asset_phrase("x_server.log"), "the attached log")
    self.assertEqual(_neutral_asset_phrase("opaque_id_123"), "the attached evidence")

  def test_validate_narrative_sanitizes_leaked_asset_in_place(self):
    payload = {
        "rounds": [{
            "instruction": "Inspect ops_dashboard_cpu_chart.png and act.",
            "mock_subagents": [],
        }],
    }
    # Should NOT raise; instruction is rewritten in place with no leak.
    _validate_generated_round_narrative(
        payload, "r01", ["ops_dashboard_cpu_chart.png"]
    )
    new_instr = payload["rounds"][0]["instruction"]
    self.assertNotIn("ops_dashboard_cpu_chart", new_instr.lower())
    self.assertIn("the attached screenshot", new_instr)


class TestPipelineRetries(unittest.TestCase):

  def setUp(self):
    self.seed = ConflictVLMSeed(
        slug="multimodal_temporal_conflict_01",
        domain="composite",
        title="Conflict Test",
        target_rounds=2,
        core_conflict="Mock conflict.",
        conflict_taxonomy=["temporal_staleness"],
        participants=["Ops_Agent", "Dev_Agent"],
        required_constraints=["Mock constraint."],
        allowed_tool_families=["file", "asset"],
        modalities=["chart"],
        input_policy={"default_mode": "path"},
        generation_policy={"chart_provider": "mock"},
        min_round_steps=4,
        require_final_verification=False,
    )
    self.mock_client = MagicMock()
    self.pipeline = ConflictLLMPipeline(client=self.mock_client, progress=lambda msg: None)

    # Valid layer responses to build on
    self.valid_layer0a = {
        "summary": "Mock summary",
        "screenplay_outline": {
            "conflict_arc": "mock conflict arc",
            "character_cast": [],
            "narrative_conflict_triggers": []
        }
    }
    # Tier 2.1 split fixtures: cast (conflict_arc + character_cast, no triggers)
    # then triggers referencing the finalized cast role_names.
    self.valid_layer0a_cast = {
        "summary": "Mock summary",
        "screenplay_outline": {
            "conflict_arc": "mock conflict arc",
            "character_cast": []
        }
    }
    self.valid_layer0a_triggers = {"narrative_conflict_triggers": []}
    self.valid_layer0b = {
        "phases": [{"phase_id": "p01", "start_round": 1, "end_round": 2, "theme": "investigation", "timeline_screenplay": []}]
    }
    self.valid_layer0c = {
        "global_assets_plan": [],
        "global_updates_plan": [],
        "global_interactions_plan": []
    }
    # Tier 3.3 (design 6.3) split fixtures: the three global plans are now three
    # independently-retried calls. The merged shape is identical to valid_layer0c.
    self.valid_layer0c_assets_plan = {"global_assets_plan": self.valid_layer0c["global_assets_plan"]}
    self.valid_layer0c_updates_plan = {"global_updates_plan": self.valid_layer0c["global_updates_plan"]}
    self.valid_layer0c_interactions_plan = {"global_interactions_plan": self.valid_layer0c["global_interactions_plan"]}
    self.valid_layer1_agents = {"mock_subagents": [], "agents": []}
    # Tier 2.2 split fixtures: roster (agents only) then mock subagents. Kept
    # empty so the merged shape is identical to valid_layer1_agents (an empty
    # roster skips the subset check; the subset logic is unit-tested directly).
    self.valid_layer1_agent_roster = {"agents": []}
    self.valid_layer1_mock_subagents = {"mock_subagents": []}
    self.valid_layer1_interactions = {"summary": "Mock agents summary", "interactions": []}
    self.valid_layer1b_files = {"workspace": {"files": []}}
    self.valid_layer1b_sessions = {"sessions": []}
    self.valid_layer1b_assets = {"assets": []}
    self.valid_phase_expand = {
        "timeline": [
            {"round_id": "r01", "phase": "investigation", "description": "Round 1", "active_participants": ["Ops_Agent"]},
            {"round_id": "r02", "phase": "investigation", "description": "Round 2", "active_participants": ["Ops_Agent"]},
        ],
        "assets": [],
    }
    # Tier 3.1 (design 6.1) split fixtures: the phase timeline and phase assets
    # are now two independently-retried calls. The merged shape is identical to
    # valid_phase_expand.
    self.valid_phase_timeline = {"timeline": self.valid_phase_expand["timeline"]}
    self.valid_phase_assets = {"assets": self.valid_phase_expand["assets"]}
    # S1 (decisive-point half): decided from the FINALIZED timeline, before
    # any of this phase's asset text exists, so the asset halves can be told
    # which value they must carry verbatim.
    self.valid_phase_decisive = {
        "decisive_points": [
            {"round_id": "r01", "decisive_value": "182.1 U/L",
             "decisive_label": "ALT on the 06:14 lab panel",
             "decisive_value_citable": True},
            {"round_id": "r02", "decisive_value": "1,420 queries/min",
             "decisive_label": "sustained ingress rate on the ops console",
             "decisive_value_citable": True},
        ]
    }
    self.valid_narrative_rounds = {
        "rounds": [{"round_id": "r01", "round_type": "action", "instruction": "Read config.", "update_ids": [], "mock_subagents": []}],
    }
    # Tier 1.2 split fixtures: instruction (prose only) + mocks (with an
    # executable state_effect so decision-4 validation has a valid command).
    self.valid_round_instruction = {
        "rounds": [{"round_id": "r01", "round_type": "action", "instruction": "Read config.", "update_ids": []}],
    }
    self.valid_round_mocks = {
        "mock_subagents": [{
            "agent_role": "assistant",
            "mock_response": "Done.",
            "virtual_duration_seconds": 15,
            "state_effects": ["file create --path workspace/note.txt --content ok"],
        }],
    }
    self.valid_narrative_updates = {
        "updates": [],
    }
    self.valid_narrative_assets = {
        "assets": [],
    }
    self.valid_outline = {
        "outline": [
            {"step_number": 1, "description": "Verify config", "expected_check": "Checked config"},
            {"step_number": 2, "description": "List files", "expected_check": "Listed files"},
            {"step_number": 3, "description": "Wait for effect", "expected_check": None},
            {"step_number": 4, "description": "Re-verify config after fix", "expected_check": "Re-checked config"},
        ]
    }
    self.valid_steps = {
        "steps": [
            {"command": "file read --path config.yaml", "checker": {"type": "file_check", "path": "config.yaml", "contains": "timeout"}},
            {"command": "file list", "checker": None},
            {"command": "wait 1", "checker": None},
            {"command": "DONE", "checker": None}
        ]
    }
    # Tier 1.1 split fixtures: commands (layer 3a) are all "plain" here so no
    # semantic checker pass (layer 3b) is triggered; checkers fixture is the
    # empty-fields fallback used if it ever is.
    self.valid_round_commands = {
        "commands": [
            {"step_number": 1, "command": "file read --path config.yaml"},
            {"step_number": 2, "command": "file list"},
            {"step_number": 3, "command": "wait 1"},
            {"step_number": 4, "command": "DONE"},
        ]
    }
    self.valid_round_checkers = {"fields": []}

  def test_generate_retries_on_phase_expansion_failure(self):
    # Mock generate_layer to return invalid phase expansion twice, then succeed
    invalid_phase_expand = {"timeline": []} # Empty timeline fails validation

    def mock_generate_layer(layer, payload):
      if layer == "conflict_layer0a_screenplay":
        return self.valid_layer0a
      elif layer == "conflict_layer0a_cast":
        return self.valid_layer0a_cast
      elif layer == "conflict_layer0a_triggers":
        return self.valid_layer0a_triggers
      elif layer == "conflict_layer0b_timeline":
        return self.valid_layer0b
      elif layer == "conflict_layer0c_plans":
        return self.valid_layer0c
      elif layer == "conflict_layer0c_assets_plan":
        return self.valid_layer0c_assets_plan
      elif layer == "conflict_layer0c_updates_plan":
        return self.valid_layer0c_updates_plan
      elif layer == "conflict_layer0c_interactions_plan":
        return self.valid_layer0c_interactions_plan
      elif layer == "conflict_layer1_agents":
        return self.valid_layer1_agents
      elif layer == "conflict_layer1_agent_roster":
        return self.valid_layer1_agent_roster
      elif layer == "conflict_layer1_mock_subagents":
        return self.valid_layer1_mock_subagents
      elif layer == "conflict_layer1_interactions":
        return self.valid_layer1_interactions
      elif layer == "conflict_layer1b_files":
        return self.valid_layer1b_files
      elif layer == "conflict_layer1b_sessions":
        return self.valid_layer1b_sessions
      elif layer == "conflict_layer1b_assets":
        return self.valid_layer1b_assets
      elif layer == "conflict_layer0_expand_phase_timeline":
        mock_generate_layer.phase_calls += 1
        if mock_generate_layer.phase_calls < 3:
          return invalid_phase_expand
        return self.valid_phase_timeline
      elif layer == "conflict_layer0_expand_phase_assets":
        return self.valid_phase_assets
      elif layer == "conflict_layer3_round_narrative":
        return self.valid_narrative_rounds
      elif layer == "conflict_layer3_round_instruction":
        return self.valid_round_instruction
      elif layer == "conflict_layer3_round_mocks":
        return self.valid_round_mocks
      elif layer == "conflict_layer3_round_updates":
        return self.valid_narrative_updates
      elif layer == "conflict_layer3_round_assets":
        return self.valid_narrative_assets
      elif layer == "conflict_layer3_round_outline":
        return self.valid_outline
      elif layer == "conflict_layer3_round_steps":
        return self.valid_steps
      elif layer == "conflict_layer3_round_commands":
        return self.valid_round_commands
      elif layer == "conflict_layer3_round_checkers":
        return self.valid_round_checkers
      return {}

    mock_generate_layer.phase_calls = 0
    self.mock_client.generate_layer.side_effect = mock_generate_layer

    res = self.pipeline.generate(self.seed)
    self.assertEqual(mock_generate_layer.phase_calls, 3)
    self.assertIn("rounds", res)
    self.assertEqual(len(res["rounds"]), 2)

  def test_generate_salvages_after_max_phase_expansion_failures(self):
    # P0-2: a phase whose expansion never validates must NOT discard the whole
    # scenario (it previously raised and lost every round). Instead the pipeline
    # synthesizes a minimal fallback outline for the phase and keeps going, so
    # `generate` still returns the expected rounds. `phase_expand_attempts=3`
    # (P0-1) also proves the retry budget is configurable and honored.
    invalid_phase_expand = {"timeline": []}  # empty timeline always fails validation

    def mock_generate_layer(layer, payload):
      if layer == "conflict_layer0a_screenplay":
        return self.valid_layer0a
      elif layer == "conflict_layer0a_cast":
        return self.valid_layer0a_cast
      elif layer == "conflict_layer0a_triggers":
        return self.valid_layer0a_triggers
      elif layer == "conflict_layer0b_timeline":
        return self.valid_layer0b
      elif layer == "conflict_layer0c_plans":
        return self.valid_layer0c
      elif layer == "conflict_layer0c_assets_plan":
        return self.valid_layer0c_assets_plan
      elif layer == "conflict_layer0c_updates_plan":
        return self.valid_layer0c_updates_plan
      elif layer == "conflict_layer0c_interactions_plan":
        return self.valid_layer0c_interactions_plan
      elif layer == "conflict_layer1_agents":
        return self.valid_layer1_agents
      elif layer == "conflict_layer1_agent_roster":
        return self.valid_layer1_agent_roster
      elif layer == "conflict_layer1_mock_subagents":
        return self.valid_layer1_mock_subagents
      elif layer == "conflict_layer1_interactions":
        return self.valid_layer1_interactions
      elif layer == "conflict_layer1b_files":
        return self.valid_layer1b_files
      elif layer == "conflict_layer1b_sessions":
        return self.valid_layer1b_sessions
      elif layer == "conflict_layer1b_assets":
        return self.valid_layer1b_assets
      elif layer == "conflict_layer0_expand_phase_timeline":
        mock_generate_layer.phase_calls += 1
        return invalid_phase_expand
      elif layer == "conflict_layer0_expand_phase_assets":
        return self.valid_phase_assets
      elif layer == "conflict_layer3_round_narrative":
        return self.valid_narrative_rounds
      elif layer == "conflict_layer3_round_instruction":
        return self.valid_round_instruction
      elif layer == "conflict_layer3_round_mocks":
        return self.valid_round_mocks
      elif layer == "conflict_layer3_round_updates":
        return self.valid_narrative_updates
      elif layer == "conflict_layer3_round_assets":
        return self.valid_narrative_assets
      elif layer == "conflict_layer3_round_outline":
        return self.valid_outline
      elif layer == "conflict_layer3_round_steps":
        return self.valid_steps
      elif layer == "conflict_layer3_round_commands":
        return self.valid_round_commands
      elif layer == "conflict_layer3_round_checkers":
        return self.valid_round_checkers
      return {}

    mock_generate_layer.phase_calls = 0
    self.mock_client.generate_layer.side_effect = mock_generate_layer

    res = self.pipeline.generate(self.seed, phase_expand_attempts=3)

    # The configured retry budget (3) was exhausted on the single phase p01 ...
    self.assertEqual(mock_generate_layer.phase_calls, 3)
    # ... yet the scenario was salvaged from the fallback outline, not discarded.
    self.assertEqual(len(res["rounds"]), 2)

  def test_fallback_round_outline_has_expansion_shape(self):
    # The synthesized outline must carry the keys downstream round generation
    # reads, plus a marker so a salvaged phase is identifiable.
    phase = {
        "phase_id": "p01",
        "theme": "investigation",
        "description": "Detect the ingress anomaly.",
        "active_participants": ["Ops_Agent"],
    }
    outline = self.pipeline._fallback_round_outline("r02", phase, self.seed)
    self.assertEqual(outline["round_id"], "r02")
    self.assertEqual(outline["phase"], "investigation")
    self.assertEqual(outline["description"], "Detect the ingress anomaly.")
    self.assertEqual(outline["active_participants"], ["Ops_Agent"])
    self.assertEqual(outline["introduced_anchors"], [])
    self.assertTrue(outline["_synthesized_fallback"])

  def test_fallback_round_outline_defaults_from_seed(self):
    # With a bare phase, participants fall back to the seed and a generic
    # description is synthesized so the outline is never empty.
    outline = self.pipeline._fallback_round_outline("r01", {}, self.seed)
    self.assertEqual(outline["round_id"], "r01")
    self.assertEqual(outline["phase"], "action")
    self.assertEqual(outline["active_participants"], list(self.seed.participants))
    self.assertIn("r01", outline["description"])

  def test_generate_retries_on_round_generation_failure(self):
    # Mock generate_layer to return invalid narrative twice (empty instruction
    # -> hard validation failure), then succeed. (An asset-ID leak is no longer
    # a hard failure post-D4; it is auto-sanitized, so it cannot drive retries.)
    invalid_narrative_rounds = {
        "rounds": [{"round_id": "r01", "round_type": "action", "instruction": "", "update_ids": [], "mock_subagents": []}],
    }

    # Populate background with ops_dashboard_asset_bg_01 so it triggers leakage check
    background_with_asset = {
        "workspace": {"files": []},
        "sessions": [],
        "assets": [{"id": "ops_dashboard_asset_bg_01", "path": "charts/ops_dashboard_bg.png", "mime_type": "image/png", "available_from_round": 1, "platform_id": "ops_dashboard"}]
    }

    def mock_generate_layer(layer, payload):
      if layer == "conflict_layer0a_screenplay":
        return self.valid_layer0a
      elif layer == "conflict_layer0a_cast":
        return self.valid_layer0a_cast
      elif layer == "conflict_layer0a_triggers":
        return self.valid_layer0a_triggers
      elif layer == "conflict_layer0b_timeline":
        return self.valid_layer0b
      elif layer == "conflict_layer0c_plans":
        return self.valid_layer0c
      elif layer == "conflict_layer0c_assets_plan":
        return self.valid_layer0c_assets_plan
      elif layer == "conflict_layer0c_updates_plan":
        return self.valid_layer0c_updates_plan
      elif layer == "conflict_layer0c_interactions_plan":
        return self.valid_layer0c_interactions_plan
      elif layer == "conflict_layer1_agents":
        return self.valid_layer1_agents
      elif layer == "conflict_layer1_agent_roster":
        return self.valid_layer1_agent_roster
      elif layer == "conflict_layer1_mock_subagents":
        return self.valid_layer1_mock_subagents
      elif layer == "conflict_layer1_interactions":
        return self.valid_layer1_interactions
      elif layer == "conflict_layer1b_files":
        return self.valid_layer1b_files
      elif layer == "conflict_layer1b_sessions":
        return self.valid_layer1b_sessions
      elif layer == "conflict_layer1b_assets":
        return {"assets": [{"id": "ops_dashboard_asset_bg_01", "path": "charts/ops_dashboard_bg.png", "mime_type": "image/png", "available_from_round": 1, "platform_id": "ops_dashboard"}]}
      elif layer == "conflict_layer0_expand_phase_timeline":
        return self.valid_phase_timeline
      elif layer == "conflict_layer0_expand_phase_assets":
        return self.valid_phase_assets
      elif layer == "conflict_layer3_round_instruction":
        # Tier 1.2: the instruction call is the one whose empty-instruction
        # output drives the round retry (mocks are authored separately).
        mock_generate_layer.narrative_calls += 1
        if mock_generate_layer.narrative_calls < 3:
          return invalid_narrative_rounds
        # 3rd call succeeds (instruction has no leakage)
        return self.valid_round_instruction
      elif layer == "conflict_layer3_round_mocks":
        return self.valid_round_mocks
      elif layer == "conflict_layer3_round_updates":
        return self.valid_narrative_updates
      elif layer == "conflict_layer3_round_assets":
        return self.valid_narrative_assets
      elif layer == "conflict_layer3_round_outline":
        return self.valid_outline
      elif layer == "conflict_layer3_round_steps":
        return self.valid_steps
      elif layer == "conflict_layer3_round_commands":
        return self.valid_round_commands
      elif layer == "conflict_layer3_round_checkers":
        return self.valid_round_checkers
      return {}

    mock_generate_layer.narrative_calls = 0
    self.mock_client.generate_layer.side_effect = mock_generate_layer

    res = self.pipeline.generate(self.seed)
    # 3 calls for round 1 narrative, and 1 call for round 2 narrative (succeeds immediately)
    self.assertEqual(mock_generate_layer.narrative_calls, 4)
    self.assertEqual(len(res["rounds"]), 2)

  def test_generate_aborts_after_max_round_generation_failures(self):
    # Mock generate_layer to always return invalid narrative (empty instruction
    # -> hard validation failure on every attempt).
    invalid_narrative_rounds = {
        "rounds": [{"round_id": "r01", "round_type": "action", "instruction": "", "update_ids": [], "mock_subagents": []}],
    }

    background_with_asset = {
        "workspace": {"files": []},
        "sessions": [],
        "assets": [{"id": "ops_dashboard_asset_bg_01", "path": "charts/ops_dashboard_bg.png", "mime_type": "image/png", "available_from_round": 1, "platform_id": "ops_dashboard"}]
    }

    def mock_generate_layer(layer, payload):
      if layer == "conflict_layer0a_screenplay":
        return self.valid_layer0a
      elif layer == "conflict_layer0a_cast":
        return self.valid_layer0a_cast
      elif layer == "conflict_layer0a_triggers":
        return self.valid_layer0a_triggers
      elif layer == "conflict_layer0b_timeline":
        return self.valid_layer0b
      elif layer == "conflict_layer0c_plans":
        return self.valid_layer0c
      elif layer == "conflict_layer0c_assets_plan":
        return self.valid_layer0c_assets_plan
      elif layer == "conflict_layer0c_updates_plan":
        return self.valid_layer0c_updates_plan
      elif layer == "conflict_layer0c_interactions_plan":
        return self.valid_layer0c_interactions_plan
      elif layer == "conflict_layer1_agents":
        return self.valid_layer1_agents
      elif layer == "conflict_layer1_agent_roster":
        return self.valid_layer1_agent_roster
      elif layer == "conflict_layer1_mock_subagents":
        return self.valid_layer1_mock_subagents
      elif layer == "conflict_layer1_interactions":
        return self.valid_layer1_interactions
      elif layer == "conflict_layer1b_files":
        return self.valid_layer1b_files
      elif layer == "conflict_layer1b_sessions":
        return self.valid_layer1b_sessions
      elif layer == "conflict_layer1b_assets":
        return {"assets": [{"id": "ops_dashboard_asset_bg_01", "path": "charts/ops_dashboard_bg.png", "mime_type": "image/png", "available_from_round": 1, "platform_id": "ops_dashboard"}]}
      elif layer == "conflict_layer0_expand_phase_timeline":
        return self.valid_phase_timeline
      elif layer == "conflict_layer0_expand_phase_assets":
        return self.valid_phase_assets
      elif layer == "conflict_layer3_round_instruction":
        # Tier 1.2: empty instruction here fails the merged-narrative safety net
        # every attempt, driving the round abort after max retries.
        return invalid_narrative_rounds
      elif layer == "conflict_layer3_round_mocks":
        return self.valid_round_mocks
      elif layer == "conflict_layer3_round_updates":
        return self.valid_narrative_updates
      elif layer == "conflict_layer3_round_assets":
        return self.valid_narrative_assets
      elif layer == "conflict_layer3_round_outline":
        return self.valid_outline
      elif layer == "conflict_layer3_round_steps":
        return self.valid_steps
      elif layer == "conflict_layer3_round_commands":
        return self.valid_round_commands
      elif layer == "conflict_layer3_round_checkers":
        return self.valid_round_checkers
      return {}

    self.mock_client.generate_layer.side_effect = mock_generate_layer

    with self.assertRaisesRegex(ScenarioLLMError, "Failed to generate round r01 narrative after 3 attempts"):
      self.pipeline.generate(self.seed)

  def test_future_round_leak_in_instruction_retries_at_instruction_step(self):
    # A (root-cause fix): a future-round leak in the generated instruction must
    # be caught in the INSTRUCTION sub-step's own retry loop so the instruction
    # is re-sampled. Before the fix the leak check ran only in the assets loop
    # against the already-frozen instruction, so a leaked-then-clean instruction
    # still aborted the whole version (the it_deployment_gate_v01 failure mode).
    # Here r01 leaks on attempt 1 and is clean on attempt 2, so generation must
    # SUCCEED (not abort).
    leaky_instruction = {
        "rounds": [{"round_id": "r01", "round_type": "action",
                    "instruction": "Escalate this to the team in round r02 for sign-off.",
                    "update_ids": []}],
    }

    def mock_generate_layer(layer, payload):
      if layer == "conflict_layer0a_screenplay":
        return self.valid_layer0a
      elif layer == "conflict_layer0a_cast":
        return self.valid_layer0a_cast
      elif layer == "conflict_layer0a_triggers":
        return self.valid_layer0a_triggers
      elif layer == "conflict_layer0b_timeline":
        return self.valid_layer0b
      elif layer == "conflict_layer0c_plans":
        return self.valid_layer0c
      elif layer == "conflict_layer0c_assets_plan":
        return self.valid_layer0c_assets_plan
      elif layer == "conflict_layer0c_updates_plan":
        return self.valid_layer0c_updates_plan
      elif layer == "conflict_layer0c_interactions_plan":
        return self.valid_layer0c_interactions_plan
      elif layer == "conflict_layer1_agents":
        return self.valid_layer1_agents
      elif layer == "conflict_layer1_agent_roster":
        return self.valid_layer1_agent_roster
      elif layer == "conflict_layer1_mock_subagents":
        return self.valid_layer1_mock_subagents
      elif layer == "conflict_layer1_interactions":
        return self.valid_layer1_interactions
      elif layer == "conflict_layer1b_files":
        return self.valid_layer1b_files
      elif layer == "conflict_layer1b_sessions":
        return self.valid_layer1b_sessions
      elif layer == "conflict_layer1b_assets":
        return self.valid_layer1b_assets
      elif layer == "conflict_layer0_expand_phase_timeline":
        return self.valid_phase_timeline
      elif layer == "conflict_layer0_expand_phase_assets":
        return self.valid_phase_assets
      elif layer == "conflict_layer3_round_instruction":
        mock_generate_layer.instruction_calls += 1
        # Leak on the first attempt only; clean thereafter.
        if mock_generate_layer.instruction_calls == 1:
          return leaky_instruction
        return self.valid_round_instruction
      elif layer == "conflict_layer3_round_mocks":
        return self.valid_round_mocks
      elif layer == "conflict_layer3_round_updates":
        return self.valid_narrative_updates
      elif layer == "conflict_layer3_round_assets":
        return self.valid_narrative_assets
      elif layer == "conflict_layer3_round_outline":
        return self.valid_outline
      elif layer == "conflict_layer3_round_steps":
        return self.valid_steps
      elif layer == "conflict_layer3_round_commands":
        return self.valid_round_commands
      elif layer == "conflict_layer3_round_checkers":
        return self.valid_round_checkers
      return {}

    mock_generate_layer.instruction_calls = 0
    self.mock_client.generate_layer.side_effect = mock_generate_layer

    res = self.pipeline.generate(self.seed)
    # r01: attempt 1 leaks -> retry -> attempt 2 clean (2 calls); r02: 1 clean
    # call. The leak drove a retry at the INSTRUCTION step, not an abort.
    self.assertEqual(mock_generate_layer.instruction_calls, 3)
    self.assertEqual(len(res["rounds"]), 2)

  def test_future_round_leak_aborts_at_instruction_step_not_assets(self):
    # A (regression guard): when EVERY instruction sample leaks, the round must
    # abort inside the instruction sub-step (fail-closed) BEFORE authoring mocks
    # or assets. Pre-fix, the leaky instruction was accepted, mocks were
    # authored, and the abort happened only after 3 asset attempts. Asserting
    # mocks/assets were never called pins the fix in place.
    leaky_instruction = {
        "rounds": [{"round_id": "r01", "round_type": "action",
                    "instruction": "Escalate this to the team in round r02 for sign-off.",
                    "update_ids": []}],
    }

    def mock_generate_layer(layer, payload):
      if layer == "conflict_layer0a_screenplay":
        return self.valid_layer0a
      elif layer == "conflict_layer0a_cast":
        return self.valid_layer0a_cast
      elif layer == "conflict_layer0a_triggers":
        return self.valid_layer0a_triggers
      elif layer == "conflict_layer0b_timeline":
        return self.valid_layer0b
      elif layer == "conflict_layer0c_plans":
        return self.valid_layer0c
      elif layer == "conflict_layer0c_assets_plan":
        return self.valid_layer0c_assets_plan
      elif layer == "conflict_layer0c_updates_plan":
        return self.valid_layer0c_updates_plan
      elif layer == "conflict_layer0c_interactions_plan":
        return self.valid_layer0c_interactions_plan
      elif layer == "conflict_layer1_agents":
        return self.valid_layer1_agents
      elif layer == "conflict_layer1_agent_roster":
        return self.valid_layer1_agent_roster
      elif layer == "conflict_layer1_mock_subagents":
        return self.valid_layer1_mock_subagents
      elif layer == "conflict_layer1_interactions":
        return self.valid_layer1_interactions
      elif layer == "conflict_layer1b_files":
        return self.valid_layer1b_files
      elif layer == "conflict_layer1b_sessions":
        return self.valid_layer1b_sessions
      elif layer == "conflict_layer1b_assets":
        return self.valid_layer1b_assets
      elif layer == "conflict_layer0_expand_phase_timeline":
        return self.valid_phase_timeline
      elif layer == "conflict_layer0_expand_phase_assets":
        return self.valid_phase_assets
      elif layer == "conflict_layer3_round_instruction":
        mock_generate_layer.instruction_calls += 1
        return leaky_instruction  # always leaks
      elif layer == "conflict_layer3_round_mocks":
        mock_generate_layer.mocks_calls += 1
        return self.valid_round_mocks
      elif layer == "conflict_layer3_round_updates":
        return self.valid_narrative_updates
      elif layer == "conflict_layer3_round_assets":
        mock_generate_layer.assets_calls += 1
        return self.valid_narrative_assets
      elif layer == "conflict_layer3_round_outline":
        return self.valid_outline
      elif layer == "conflict_layer3_round_steps":
        return self.valid_steps
      elif layer == "conflict_layer3_round_commands":
        return self.valid_round_commands
      elif layer == "conflict_layer3_round_checkers":
        return self.valid_round_checkers
      return {}

    mock_generate_layer.instruction_calls = 0
    mock_generate_layer.mocks_calls = 0
    mock_generate_layer.assets_calls = 0
    self.mock_client.generate_layer.side_effect = mock_generate_layer

    with self.assertRaisesRegex(ScenarioLLMError, "Failed to generate round r01 narrative after 3 attempts"):
      self.pipeline.generate(self.seed)
    # 3 attempts, all at the instruction step; never reached mocks or assets.
    self.assertEqual(mock_generate_layer.instruction_calls, 3)
    self.assertEqual(mock_generate_layer.mocks_calls, 0)
    self.assertEqual(mock_generate_layer.assets_calls, 0)


class TestConflictResume(unittest.TestCase):

  def test_resume_from_saved_spec(self):
    import json
    seed = ConflictVLMSeed(
        slug="test_resume",
        domain="system",
        title="Test Resume",
        target_rounds=1,
        core_conflict="Test dynamic recovery.",
        conflict_taxonomy=["temporal_staleness"],
        participants=["owner", "assistant"],
        required_constraints=[],
        allowed_tool_families=["agents", "file"],
        min_round_steps=4,
        require_final_verification=False,
    )
    with tempfile.TemporaryDirectory() as tmp:
      tmp_path = Path(tmp)
      client = MockConflictLLMClient()
      
      # 1. Run fresh materialization to generate initial spec
      spec1, _ = materialize_conflict_scenario(
          seed,
          client,
          output_dir=tmp_path,
          image_provider="mock",
          chart_provider="mock",
      )
      spec_json_path = tmp_path / "test_resume_spec.json"
      self.assertTrue(spec_json_path.is_file())
      
      # 2. Modify one asset description in the saved file to simulate external change
      with open(spec_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
      
      self.assertTrue(len(data["assets"]) > 0)
      data["assets"][0]["description"] = "Modified description during resume"
      
      with open(spec_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
      
      # 3. Call materialize_conflict_scenario with resume=True
      # We pass None for llm_client to prove it doesn't call LLM pipeline (Stages 0-3)!
      spec2, _ = materialize_conflict_scenario(
          seed,
          llm_client=None,
          output_dir=tmp_path,
          image_provider="mock",
          chart_provider="mock",
          resume=True,
      )
      
      # Verify it successfully loaded the spec from disk instead of failing or starting fresh!
      self.assertEqual(spec2.assets[0].description, "Modified description during resume")

  def test_resume_provider_mismatch_triggers_re_render(self):
    seed = ConflictVLMSeed(
        slug="test_resume_mismatch",
        domain="system",
        title="Test Resume Mismatch",
        target_rounds=1,
        core_conflict="Test provider validation.",
        conflict_taxonomy=["temporal_staleness"],
        participants=["owner", "assistant"],
        required_constraints=[],
        allowed_tool_families=["agents", "file"],
        min_round_steps=4,
        require_final_verification=False,
    )
    with tempfile.TemporaryDirectory() as tmp:
      tmp_path = Path(tmp)
      client = MockConflictLLMClient()
      
      # 1. Run fresh materialization with image_provider="mock"
      spec1, _ = materialize_conflict_scenario(
          seed,
          client,
          output_dir=tmp_path,
          image_provider="mock",
          chart_provider="mock",
      )
      
      # 2. Run with resume=True but image_provider="google" (mismatch with cached "mock")
      # This should trigger cache miss and attempt to render via Gemini, failing on missing API key.
      with self.assertRaises(Exception) as ctx:
        materialize_conflict_scenario(
            seed,
            llm_client=None,
            output_dir=tmp_path,
            image_provider="google",
            chart_provider="mock",
            resume=True,
        )
      self.assertIn("Google image API key env var is not set", str(ctx.exception))

  def test_resume_from_pipeline_checkpoint(self):
    import json
    import logging
    seed = ConflictVLMSeed(
        slug="test_resume_pipeline",
        domain="system",
        title="Test Resume Pipeline",
        target_rounds=2,
        core_conflict="Test pipeline checkpointing.",
        conflict_taxonomy=["temporal_staleness"],
        participants=["owner", "assistant"],
        required_constraints=[],
        allowed_tool_families=["agents", "file"],
        min_round_steps=4,
        require_final_verification=False,
    )
    
    calls = []
    should_fail = True

    class CustomMockClient(MockConflictLLMClient):
      def generate_layer(self, layer, payload):
        calls.append(layer)
        if should_fail and layer == "conflict_layer3_round_commands" and payload.get("slot", {}).get("round_index") == 2:
          raise ScenarioLLMError("Simulated Gemini 503 Error on Round 2")
        return super().generate_layer(layer, payload)

    with tempfile.TemporaryDirectory() as tmp:
      tmp_path = Path(tmp)
      client = CustomMockClient()

      # 1. Run first time - should crash on round 2
      with self.assertRaises(ScenarioLLMError) as ctx:
        materialize_conflict_scenario(
            seed,
            client,
            output_dir=tmp_path,
            image_provider="mock",
            chart_provider="mock",
            progress=logging.info,
        )
      self.assertIn("Failed to generate round r02 steps after", str(ctx.exception))

      # Check that the pipeline checkpoint was successfully saved to disk!
      ckpt_path = tmp_path / "test_resume_pipeline_pipeline_checkpoint.json"
      self.assertTrue(ckpt_path.is_file())

      # Read checkpoint to verify round 1 details are saved
      with open(ckpt_path, "r", encoding="utf-8") as f:
        ckpt_data = json.load(f)
      self.assertEqual(len(ckpt_data["rounds"]), 1)
      self.assertEqual(ckpt_data["rounds"][0]["round_id"], "r01")

      # 2. Reset fail flag and calls tracker
      should_fail = False
      calls.clear()

      # 3. Resume generation
      spec, _ = materialize_conflict_scenario(
          seed,
          client,
          output_dir=tmp_path,
          image_provider="mock",
          chart_provider="mock",
          progress=logging.info,
          resume=True,
      )

      # Verify it finished successfully
      self.assertEqual(len(spec.rounds), 2)
      self.assertEqual(spec.rounds[1].round_id, "r02")

      # Verify that Stage 1, Stage 2, Stage 3 and Round 1 calls were completely skipped!
      self.assertNotIn("conflict_layer0a_cast", calls)
      self.assertNotIn("conflict_layer0a_triggers", calls)
      self.assertNotIn("conflict_layer0b_timeline", calls)
      self.assertNotIn("conflict_layer0c_assets_plan", calls)
      self.assertNotIn("conflict_layer0c_updates_plan", calls)
      self.assertNotIn("conflict_layer0c_interactions_plan", calls)
      self.assertNotIn("conflict_layer1_agent_roster", calls)
      self.assertNotIn("conflict_layer1_mock_subagents", calls)
      self.assertNotIn("conflict_layer1_interactions", calls)
      self.assertNotIn("conflict_layer1b_files", calls)
      self.assertNotIn("conflict_layer1b_sessions", calls)
      self.assertNotIn("conflict_layer1b_assets", calls)

      # Check that the pipeline checkpoint file was deleted on successful completion
      self.assertFalse(ckpt_path.is_file())

      # Check that the visualizer html was automatically generated!
      self.assertTrue((tmp_path / "visualize.html").is_file())

  def test_platform_style_consistency(self):
    # Create a dummy seed with assets sharing platform_ids
    seed = ConflictVLMSeed(
        slug="test_style_consistency",
        domain="system",
        title="Test Style Consistency",
        target_rounds=1,
        core_conflict="Style check.",
        conflict_taxonomy=["temporal_staleness"],
        participants=["owner", "assistant"],
        required_constraints=[],
        allowed_tool_families=["agents", "file"],
        min_round_steps=4,
        require_final_verification=False,
    )

    # We can mock client response with assets having platform_id values
    raw_data = {
        "workspace": {"files": []},
        "sessions": [],
        "updates": [],
        "rounds": [
            {
                "round_id": "r01",
                "round_type": "action",
                "instruction": "Inspect status.",
                "solution_commands": [],
            }
        ],
        "evals": [],
        "agents": [],
        "assets": [
            {
                "id": "chart_a1",
                "kind": "chart",
                "generator": "template_chart",
                "path": "charts/a1.png",
                "available_from_round": 1,
                "platform_id": "ops_console",
            },
            {
                "id": "chart_a2",
                "kind": "chart",
                "generator": "template_chart",
                "path": "charts/a2.png",
                "available_from_round": 1,
                "platform_id": "ops_console",
            },
            {
                "id": "chart_b1",
                "kind": "chart",
                "generator": "template_chart",
                "path": "charts/b1.png",
                "available_from_round": 1,
                "platform_id": "billing_portal",
            },
        ],
    }

    from unittest.mock import patch
    with patch("clasharness.scenario.conflict_vlm_materializer.ConflictLLMPipeline.generate", return_value=raw_data):
      with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        spec, _ = materialize_conflict_scenario(
            seed,
            llm_client=None,
            output_dir=tmp_path,
            image_provider="mock",
            chart_provider="mock",
        )

        # Verify that charts a1 and a2 have the same theme and primary color
        a1 = next(a for a in spec.assets if a.id == "chart_a1")
        a2 = next(a for a in spec.assets if a.id == "chart_a2")
        b1 = next(a for a in spec.assets if a.id == "chart_b1")

        styling_a1 = a1.custom_overrides.get("styling", {})
        styling_a2 = a2.custom_overrides.get("styling", {})
        styling_b1 = b1.custom_overrides.get("styling", {})

        self.assertIn("theme", styling_a1)
        self.assertIn("primary_color", styling_a1)

        # Consistent style for same platform
        self.assertEqual(styling_a1["theme"], styling_a2["theme"])
        self.assertEqual(styling_a1["primary_color"], styling_a2["primary_color"])

        # Different platforms styled correctly too
        self.assertIn("theme", styling_b1)
        self.assertIn("primary_color", styling_b1)

  def test_platform_style_is_coherent_and_seed_reproducible(self):
    # WD5: the conflict path now assigns one *coherent* SSoT design bundle per
    # platform (theme + primary_color + visual_style + style_details), replacing
    # the old 6-item STYLE_PRESETS (theme+color only). The look is deterministic
    # for a pinned style_seed and actually changes with the seed.
    from clasharness.scenario import design_archetypes as da

    seed = ConflictVLMSeed(
        slug="test_style_coherence",
        domain="system",
        title="Test Style Coherence",
        target_rounds=1,
        core_conflict="Style check.",
        conflict_taxonomy=["temporal_staleness"],
        participants=["owner", "assistant"],
        required_constraints=[],
        allowed_tool_families=["agents", "file"],
        min_round_steps=4,
        require_final_verification=False,
    )
    raw_data = {
        "workspace": {"files": []},
        "sessions": [],
        "updates": [],
        "rounds": [
            {
                "round_id": "r01",
                "round_type": "action",
                "instruction": "Inspect status.",
                "solution_commands": [],
            }
        ],
        "evals": [],
        "agents": [],
        "assets": [
            {
                "id": "chart_a1", "kind": "chart", "generator": "template_chart",
                "path": "charts/a1.png", "available_from_round": 1,
                "platform_id": "ops_console",
            },
            {
                "id": "chart_a2", "kind": "chart", "generator": "template_chart",
                "path": "charts/a2.png", "available_from_round": 1,
                "platform_id": "ops_console",
            },
            {
                "id": "chart_b1", "kind": "chart", "generator": "template_chart",
                "path": "charts/b1.png", "available_from_round": 1,
                "platform_id": "billing_portal",
            },
        ],
    }

    from unittest.mock import patch

    def _styling_for(style_seed: int) -> dict:
      with patch(
          "clasharness.scenario.conflict_vlm_materializer.ConflictLLMPipeline.generate",
          return_value=raw_data,
      ):
        with tempfile.TemporaryDirectory() as tmp:
          spec, _ = materialize_conflict_scenario(
              seed, llm_client=None, output_dir=Path(tmp),
              image_provider="mock", chart_provider="mock", style_seed=style_seed,
          )
      return {a.id: dict(a.custom_overrides.get("styling", {})) for a in spec.assets if a.kind == "chart"}

    s0 = _styling_for(0)

    # Coherence: all four axes are present and template-valid (STYLE_PRESETS only
    # ever emitted theme+primary_color).
    a1 = s0["chart_a1"]
    self.assertIn("visual_style", a1)
    self.assertIn("style_details", a1)
    self.assertIn(a1["theme"], da.COLOR_THEMES)
    self.assertIn(a1["primary_color"], da.PRIMARY_COLORS)
    self.assertTrue(da.has_archetype(a1["visual_style"]))
    self.assertTrue(isinstance(a1["style_details"], dict) and a1["style_details"])

    # Same platform -> identical bundle; distinct platform -> its own bundle.
    self.assertEqual(s0["chart_a1"], s0["chart_a2"])
    self.assertIn("visual_style", s0["chart_b1"])

    # Reproducible for a pinned seed; the seed is a real knob across platforms.
    self.assertEqual(_styling_for(0), s0)
    looks = set()
    for s in range(6):
      a = _styling_for(s)["chart_a1"]
      looks.add((a["visual_style"], a["theme"], a["primary_color"]))
    self.assertGreater(len(looks), 1)


class TestRoundStepsSplit(unittest.TestCase):
  """Tier 1.1: round_steps -> commands (LLM) + deterministic checker scaffold
  (code) + minimal semantic fill (LLM). See layer_split_design.md 4.1."""

  ALLOWED = ["file", "asset", "message", "agent"]

  # --- helpers: flag extraction / path normalization -----------------------

  def test_extract_flag_value_space_form(self):
    self.assertEqual(
        _extract_flag_value("agent delegate --agent A --instruction 'hi there'", "--instruction"),
        "hi there",
    )

  def test_extract_flag_value_equals_form(self):
    self.assertEqual(
        _extract_flag_value("file create --path=notes/x.txt --content c", "--path"),
        "notes/x.txt",
    )

  def test_extract_flag_value_missing_returns_none(self):
    self.assertIsNone(_extract_flag_value("file list", "--path"))

  def test_normalize_ws_path_strips_leading_slash(self):
    self.assertEqual(_normalize_ws_path("/abs/x.txt"), "abs/x.txt")

  def test_normalize_ws_path_preserves_relative(self):
    self.assertEqual(_normalize_ws_path("rel/x.txt"), "rel/x.txt")

  def test_normalize_ws_path_trims_whitespace(self):
    self.assertEqual(_normalize_ws_path("  spaced/x  "), "spaced/x")

  # --- helper: command classification (truth table) ------------------------

  def test_classify_delegation_agent(self):
    cat, spec = _classify_step_command(
        "agent delegate --agent Devon --instruction 'confirm'", self.ALLOWED
    )
    self.assertEqual(cat, "delegation")
    self.assertEqual(spec.name, "agent.delegate")

  def test_classify_delegation_message_send(self):
    cat, _ = _classify_step_command(
        "message send --channel general --target ops --message \"sync\"", self.ALLOWED
    )
    self.assertEqual(cat, "delegation")

  def test_classify_file_write_create(self):
    cat, spec = _classify_step_command(
        "file create --path notes/out.txt --content \"x\"", self.ALLOWED
    )
    self.assertEqual(cat, "file_write")
    self.assertEqual(spec.name, "file.create")

  def test_classify_asset_access_open(self):
    cat, _ = _classify_step_command("asset open --id chart_01", self.ALLOWED)
    self.assertEqual(cat, "asset_access")

  def test_classify_file_read(self):
    # file.read is its own category: it only yields a checker when the round has
    # a decisive value to assert, otherwise the scaffold leaves it uncheck-ed.
    cat, spec = _classify_step_command("file read --path config.yaml", self.ALLOWED)
    self.assertEqual(cat, "file_read")
    self.assertEqual(spec.name, "file.read")

  def test_classify_unresolvable_degrades_to_plain(self):
    cat, spec = _classify_step_command("frobnicate the widget", self.ALLOWED)
    self.assertEqual(cat, "plain")
    self.assertIsNone(spec)

  # --- scaffolder ----------------------------------------------------------

  def _scaffold_sample(self):
    commands = [
        {"step_number": 1, "command": "agent delegate --agent Devon --instruction 'confirm the deadlock'"},
        {"step_number": 2, "command": "file create --path notes/out.txt --content \"x\""},
        {"step_number": 3, "command": "asset open --id chart_visible"},
        {"step_number": 4, "command": "asset open --id chart_unknown"},
        {"step_number": 5, "command": "file list"},
        {"step_number": 6, "command": "DONE"},
    ]
    return _scaffold_round_checkers(commands, self.ALLOWED, asset_ids=["chart_visible"])

  def test_scaffold_delegation_placeholder_and_semantic(self):
    scaffolds, needed = self._scaffold_sample()
    deleg = scaffolds[0]["checker"]
    self.assertEqual(deleg["type"], "effect")
    self.assertEqual(deleg["effect_type"], "messages_sent")
    self.assertEqual(deleg["condition"], "field_semantic_match")
    self.assertEqual(deleg["expected"]["criterion"], "")  # placeholder, filled later
    self.assertEqual(scaffolds[0]["semantic"]["field"], "criterion")
    self.assertEqual(scaffolds[0]["semantic"]["fallback"], "confirm the deadlock")
    self.assertIn({"step_number": 1, "field": "criterion",
                   "command": "agent delegate --agent Devon --instruction 'confirm the deadlock'"}, needed)

  def test_scaffold_file_write_path_and_semantic(self):
    scaffolds, needed = self._scaffold_sample()
    fw = scaffolds[1]["checker"]
    # Path-agnostic deliverable: graded via the files_created effect stream, not
    # a path-bound file_check. No decisive value seeded here, so it starts empty.
    self.assertEqual(fw["type"], "effect")
    self.assertEqual(fw["effect_type"], "files_created")
    self.assertEqual(fw["condition"], "field_contains")
    self.assertEqual(fw["expected"]["value"], "")
    self.assertNotIn("path", fw)
    self.assertEqual(scaffolds[1]["semantic"]["field"], "contains")
    self.assertTrue(any(t["step_number"] == 2 and t["field"] == "contains" for t in needed))

  def test_scaffold_known_asset_is_deterministic(self):
    scaffolds, _ = self._scaffold_sample()
    a = scaffolds[2]["checker"]
    self.assertEqual(a["type"], "effect")
    self.assertEqual(a["effect_type"], "assets_accessed")
    self.assertEqual(a["expected"]["value"], "chart_visible")
    self.assertIsNone(scaffolds[2]["semantic"])

  def test_scaffold_unknown_asset_degrades_to_command_check(self):
    scaffolds, _ = self._scaffold_sample()
    self.assertEqual(scaffolds[3]["checker"]["type"], "command_check")

  def test_scaffold_plain_has_no_checker(self):
    scaffolds, _ = self._scaffold_sample()
    self.assertIsNone(scaffolds[4]["checker"])  # file list
    self.assertIsNone(scaffolds[5]["checker"])  # DONE

  def test_scaffold_semantic_needed_only_delegation_and_file_write(self):
    _, needed = self._scaffold_sample()
    self.assertEqual({t["step_number"] for t in needed}, {1, 2})

  # --- scaffolder: decisive-value threading (difficulty layer) -------------

  def test_scaffold_file_read_without_value_is_uncheck_ed(self):
    # A file read with no decisive value stays plain (no checker, no semantic).
    scaffolds, needed = _scaffold_round_checkers(
        [{"step_number": 1, "command": "file read --path config/edge.yaml"}],
        self.ALLOWED,
    )
    self.assertIsNone(scaffolds[0]["checker"])
    self.assertIsNone(scaffolds[0]["semantic"])
    self.assertEqual(needed, [])

  def test_scaffold_file_read_with_value_is_deterministic_file_check(self):
    # With a decisive value, a file read becomes a graded, deterministic
    # file_check asserting the value -- and needs no semantic pass.
    scaffolds, needed = _scaffold_round_checkers(
        [{"step_number": 1, "command": "file read --path config/edge.yaml"}],
        self.ALLOWED,
        resolved_value="default_upstream",
    )
    chk = scaffolds[0]["checker"]
    self.assertEqual(chk["type"], "file_check")
    self.assertEqual(chk["path"], "config/edge.yaml")
    self.assertEqual(chk["contains"], "default_upstream")
    self.assertIsNone(scaffolds[0]["semantic"])
    self.assertEqual(needed, [])

  def test_scaffold_file_write_prose_value_routes_semantic(self):
    # S6: a value the deciding step did NOT declare citable makes the write a
    # file-scoped SEMANTIC check embedding the value -- not a verbatim contains,
    # which a correct paraphrase of the finding could never satisfy. No semantic
    # placeholder is needed: the criterion is authored deterministically.
    scaffolds, needed = _scaffold_round_checkers(
        [{"step_number": 1,
          "command": "file create --path notes/out.txt --content \"x\""}],
        self.ALLOWED,
        resolved_value="deadlock_confirmed",
        resolved_value_citable=False,
    )
    chk = scaffolds[0]["checker"]
    self.assertEqual(chk["type"], "effect")
    self.assertEqual(chk["effect_type"], "files_created")
    self.assertEqual(chk["condition"], "field_semantic_match")
    self.assertIn("deadlock_confirmed", chk["expected"]["criterion"])
    self.assertIsNone(scaffolds[0]["semantic"])
    self.assertEqual(needed, [])

  def test_scaffold_file_write_token_value_routes_exact_contains(self):
    # S6: a value the deciding step declared citable stays an exact
    # ``contains`` -- deterministic, free, and unambiguous.
    scaffolds, needed = _scaffold_round_checkers(
        [{"step_number": 1,
          "command": "file create --path reports/out.md --content \"x\""}],
        self.ALLOWED,
        resolved_value="42%",
        resolved_value_citable=True,
    )
    chk = scaffolds[0]["checker"]
    self.assertEqual(chk["type"], "effect")
    self.assertEqual(chk["effect_type"], "files_created")
    self.assertEqual(chk["condition"], "field_contains")
    self.assertEqual(chk["expected"]["value"], "42%")
    self.assertIsNone(scaffolds[0]["semantic"])
    self.assertEqual(needed, [])

  def test_scaffold_file_write_no_value_keeps_semantic_fill(self):
    # No decisive value to seed: keep the legacy write-content check and let the
    # semantic pass (layer 3b) author a specific substring.
    scaffolds, needed = _scaffold_round_checkers(
        [{"step_number": 1,
          "command": "file create --path notes/out.txt --content \"x\""}],
        self.ALLOWED,
    )
    chk = scaffolds[0]["checker"]
    self.assertEqual(chk["type"], "effect")
    self.assertEqual(chk["effect_type"], "files_created")
    self.assertEqual(chk["condition"], "field_contains")
    self.assertEqual(chk["expected"]["value"], "")
    self.assertEqual(scaffolds[0]["semantic"]["field"], "contains")
    self.assertTrue(any(t["step_number"] == 1 and t["field"] == "contains"
                        for t in needed))

  # --- P1: code files never carry a decisive value as text -----------------

  def test_is_code_file_true_for_scripts(self):
    for p in ("scripts/remap.py", "reload.sh", "a/b/main.go", "x.js", "T.tsx"):
      self.assertTrue(_is_code_file(p), p)

  def test_is_code_file_false_for_data_and_reports(self):
    for p in ("notes/out.txt", "config.yaml", "report.md", "state.json",
              "run.log", "a.conf", "cohort.enc", "README", "dir.py/file.txt"):
      self.assertFalse(_is_code_file(p), p)

  def test_scaffold_code_file_read_stays_plain(self):
    # Even with a decisive value, a *read* of a script never becomes a
    # file_check asserting the value is embedded in the code.
    scaffolds, needed = _scaffold_round_checkers(
        [{"step_number": 1, "command": "file read --path scripts/remap.py"}],
        self.ALLOWED,
        resolved_value="MAP_NODE_9921",
    )
    self.assertIsNone(scaffolds[0]["checker"])
    self.assertIsNone(scaffolds[0]["semantic"])
    self.assertEqual(needed, [])

  def test_scaffold_code_file_write_degrades_to_command_check(self):
    # A *write* to a script grades that the action happened, not that a
    # decisive value is present inside the code (no semantic pass).
    scaffolds, needed = _scaffold_round_checkers(
        [{"step_number": 1,
          "command": "file create --path scripts/reload.py --content \"x\""}],
        self.ALLOWED,
        resolved_value="RELOAD REQUIRED",
    )
    self.assertEqual(scaffolds[0]["checker"]["type"], "command_check")
    self.assertIsNone(scaffolds[0]["semantic"])
    self.assertEqual(needed, [])

  def test_scaffold_data_file_write_still_seeds_value(self):
    # Regression guard: the code-file rule must not disturb data files -- a data
    # file write still becomes a graded file_check embedding the decisive value
    # (semantic here, since the value is prose), never degraded to command_check.
    scaffolds, _ = _scaffold_round_checkers(
        [{"step_number": 1,
          "command": "file create --path notes/decision.md --content \"x\""}],
        self.ALLOWED,
        resolved_value="deadlock_confirmed",
    )
    chk = scaffolds[0]["checker"]
    self.assertEqual(chk["type"], "effect")
    self.assertEqual(chk["effect_type"], "files_created")
    self.assertEqual(chk["condition"], "field_semantic_match")
    self.assertIn("deadlock_confirmed", chk["expected"]["criterion"])

  # --- P3: file edit/modify count as concrete actions ----------------------

  def test_classify_trajectory_command_write_family_is_action(self):
    for cmd in ("file write --path x --content y", "file create --path x",
                "file append --path x", "file update --path x",
                "file edit --path x", "file modify --path x"):
      self.assertEqual(_classify_trajectory_command(cmd), "action", cmd)

  def test_classify_trajectory_command_read_is_inspect(self):
    self.assertEqual(_classify_trajectory_command("file read --path x"), "inspect")

  def test_classify_trajectory_command_done_and_wait(self):
    self.assertEqual(_classify_trajectory_command("done"), "done")
    self.assertEqual(_classify_trajectory_command("wait 30"), "wait")

  # --- anti-repetition / anchor helpers ------------------------------------

  def test_round_command_signature_normalizes_and_drops_done(self):
    a = _round_command_signature(
        ["File List", "message send --message  x", "DONE"]
    )
    b = _round_command_signature(["file list", "message send --message x"])
    self.assertEqual(a, b)  # case/whitespace/DONE-insensitive

  def test_round_command_signature_distinguishes_trajectories(self):
    a = _round_command_signature(["file read --path a.txt", "DONE"])
    b = _round_command_signature(["file read --path b.txt", "DONE"])
    self.assertNotEqual(a, b)

  def test_coerce_evidence_anchor_forms_and_empty(self):
    self.assertEqual(_coerce_evidence_anchor("rack c2"), "EVIDENCE:RACK_C2")
    self.assertEqual(
        _coerce_evidence_anchor("EVIDENCE:ingress-zero"), "EVIDENCE:INGRESS_ZERO"
    )
    self.assertEqual(_coerce_evidence_anchor(""), "")
    self.assertEqual(_coerce_evidence_anchor("   "), "")

  # --- semantic parse / apply ---------------------------------------------

  def test_parse_semantic_fields_indexes_and_skips_empty(self):
    parsed = _parse_semantic_fields({"fields": [
        {"step_number": 1, "criterion": "asks X"},
        {"step_number": "2", "contains": "OK"},
        {"step_number": 3},                      # neither -> skipped
        {"step_number": 4, "criterion": ""},     # empty -> skipped
    ]})
    self.assertEqual(parsed, {1: {"criterion": "asks X"}, 2: {"contains": "OK"}})

  def test_apply_semantic_uses_provided_criterion(self):
    scaffolds, _ = self._scaffold_sample()
    _apply_round_semantic_fields(scaffolds, {1: {"criterion": "asks the DBA to confirm"}})
    self.assertEqual(scaffolds[0]["checker"]["expected"]["criterion"], "asks the DBA to confirm")

  def test_apply_semantic_criterion_falls_back_nonempty(self):
    # Strict validator requires a non-empty criterion; empty sem_map must still
    # yield the deterministic fallback (never blank).
    scaffolds, _ = self._scaffold_sample()
    _apply_round_semantic_fields(scaffolds, {})
    self.assertEqual(scaffolds[0]["checker"]["expected"]["criterion"], "confirm the deadlock")

  def test_apply_semantic_contains_may_stay_empty(self):
    scaffolds, _ = self._scaffold_sample()
    _apply_round_semantic_fields(scaffolds, {})
    # File deliverable is an effect check now: an unfilled contains stays as an
    # empty expected.value (not a legacy top-level ``contains`` key).
    self.assertEqual(scaffolds[1]["checker"]["expected"]["value"], "")

  def test_steps_from_scaffolds_preserves_order(self):
    scaffolds, _ = self._scaffold_sample()
    steps = _steps_from_scaffolds(scaffolds)["steps"]
    self.assertEqual([s["command"] for s in steps],
                     [c["command"] for c in scaffolds])

  # --- validators ----------------------------------------------------------

  def test_validate_commands_empty_fails(self):
    with self.assertRaises(ValueError):
      _validate_generated_round_commands({"commands": []}, self.ALLOWED)

  def test_validate_commands_missing_command_fails(self):
    with self.assertRaises(ValueError):
      _validate_generated_round_commands({"commands": [{"step_number": 1}]}, self.ALLOWED)

  def test_validate_commands_invalid_command_fails(self):
    with self.assertRaises(ValueError):
      _validate_generated_round_commands(
          {"commands": [{"step_number": 1, "command": "frobnicate x"}]}, self.ALLOWED
      )

  def test_validate_commands_valid_passes(self):
    _validate_generated_round_commands({"commands": [
        {"step_number": 1, "command": "file list"},
        {"step_number": 2, "command": "DONE"},
    ]}, self.ALLOWED)

  def test_validate_checkers_missing_fields_fails(self):
    with self.assertRaises(ValueError):
      _validate_generated_round_checkers({})

  def test_validate_checkers_entry_without_value_fails(self):
    with self.assertRaises(ValueError):
      _validate_generated_round_checkers({"fields": [{"step_number": 1}]})

  def test_validate_checkers_valid_passes(self):
    _validate_generated_round_checkers({"fields": [{"step_number": 1, "criterion": "asks X"}]})

  # --- end-to-end merge mirrors the pipeline's 3a+code+3b flow -------------

  def test_split_merge_end_to_end_passes_safety_net(self):
    norm_commands = [
        {"step_number": 1, "command": "message send --channel general --target ops --message \"sync now\""},
        {"step_number": 2, "command": "file create --path notes/status.txt --content \"ok\""},
        {"step_number": 3, "command": "file list"},
        {"step_number": 4, "command": "DONE"},
    ]
    _validate_generated_round_commands({"commands": norm_commands}, self.ALLOWED)
    scaffolds, needed = _scaffold_round_checkers(norm_commands, self.ALLOWED, asset_ids=[])
    # Both a criterion (delegation) and a contains (file_write) are requested.
    self.assertEqual({t["field"] for t in needed}, {"criterion", "contains"})
    checkers_payload = {"fields": [
        {"step_number": 1, "criterion": "tells ops to synchronize state"},
        {"step_number": 2, "contains": "ok"},
    ]}
    _validate_generated_round_checkers(checkers_payload)
    _apply_round_semantic_fields(scaffolds, _parse_semantic_fields(checkers_payload))
    steps_payload = _steps_from_scaffolds(scaffolds)
    # The legacy safety-net validator must accept the reconstructed steps.
    from clasharness.scenario.conflict_llm_pipeline import _validate_generated_round_steps
    _validate_generated_round_steps(steps_payload, self.ALLOWED)
    self.assertEqual(steps_payload["steps"][0]["checker"]["expected"]["criterion"],
                     "tells ops to synchronize state")
    self.assertEqual(steps_payload["steps"][1]["checker"]["type"], "effect")
    self.assertEqual(steps_payload["steps"][1]["checker"]["effect_type"],
                     "files_created")
    self.assertEqual(steps_payload["steps"][1]["checker"]["expected"]["value"],
                     "ok")


class TestRoundMocksSplit(unittest.TestCase):
  """Tier 1.2: round_narrative -> instruction prose + mock-subagent commands.
  See layer_split_design.md 4.2 and decision 4."""

  ALLOWED = ["file", "asset", "message", "agent"]

  # --- iter_mock_command_effects: what gets command-validated ---------------

  def test_iter_effects_returns_command_strings(self):
    self.assertEqual(
        iter_mock_command_effects(["file list", "  message send --x y  "]),
        ["file list", "message send --x y"],
    )

  def test_iter_effects_skips_file_write_dict(self):
    effects = [{"type": "file_create", "path": "workspace/x.txt", "content": "hi"}]
    self.assertEqual(iter_mock_command_effects(effects), [])

  def test_iter_effects_extracts_command_dict(self):
    effects = [{"effect_type": "command_execution", "command": "file list"}]
    self.assertEqual(iter_mock_command_effects(effects), ["file list"])

  def test_iter_effects_empty_and_none(self):
    self.assertEqual(iter_mock_command_effects(None), [])
    self.assertEqual(iter_mock_command_effects([]), [])

  def test_iter_effects_wraps_scalar_string(self):
    self.assertEqual(iter_mock_command_effects("file list"), ["file list"])

  # --- _validate_generated_round_mocks -------------------------------------

  def test_mocks_missing_list_fails(self):
    with self.assertRaises(ValueError):
      _validate_generated_round_mocks({}, self.ALLOWED)

  def test_mocks_missing_agent_role_fails(self):
    with self.assertRaises(ValueError):
      _validate_generated_round_mocks(
          {"mock_subagents": [{"mock_response": "hi"}]}, self.ALLOWED
      )

  def test_mocks_missing_response_fails(self):
    with self.assertRaises(ValueError):
      _validate_generated_round_mocks(
          {"mock_subagents": [{"agent_role": "Casey_Database_Admin"}]}, self.ALLOWED
      )

  def test_mocks_valid_command_state_effect_passes(self):
    _validate_generated_round_mocks(
        {"mock_subagents": [{
            "agent_role": "Casey_Database_Admin",
            "mock_response": "done",
            "state_effects": ["file create --path workspace/k.txt --content abc"],
        }]},
        self.ALLOWED,
    )

  def test_mocks_invalid_command_state_effect_fails(self):
    with self.assertRaises(ValueError) as ctx:
      _validate_generated_round_mocks(
          {"mock_subagents": [{
              "agent_role": "Casey_Database_Admin",
              "mock_response": "done",
              "state_effects": ["please go and fix the database"],
          }]},
          self.ALLOWED,
      )
    self.assertIn("not an executable command", str(ctx.exception))

  def test_mocks_broadcast_without_channel_fails(self):
    # G2': `message broadcast` is channel-routed by convention in this pipeline
    # (its reply is routed into messages_sent by --channel at runtime). Unlike
    # `message send`, the tool spec does not already mandate --channel for
    # broadcast, so this guard is what enforces it.
    with self.assertRaises(ValueError) as ctx:
      _validate_generated_round_mocks(
          {"mock_subagents": [{
              "agent_role": "Casey_Database_Admin",
              "mock_response": "done",
              "state_effects": ["message broadcast --message 'all clear'"],
          }]},
          self.ALLOWED,
      )
    self.assertIn("--channel", str(ctx.exception))

  def test_mocks_broadcast_with_channel_passes(self):
    _validate_generated_round_mocks(
        {"mock_subagents": [{
            "agent_role": "Casey_Database_Admin",
            "mock_response": "done",
            "state_effects": [
                "message broadcast --channel ops --message 'all clear'"],
        }]},
        self.ALLOWED,
    )

  def test_mocks_send_with_channel_passes(self):
    _validate_generated_round_mocks(
        {"mock_subagents": [{
            "agent_role": "Casey_Database_Admin",
            "mock_response": "done",
            "state_effects": [
                "message send --channel ops --target oncall --message 'fixed'"],
        }]},
        self.ALLOWED,
    )

  def test_mocks_file_write_dict_state_effect_passes(self):
    _validate_generated_round_mocks(
        {"mock_subagents": [{
            "agent_role": "Casey_Database_Admin",
            "mock_response": "done",
            "state_effects": [{"type": "file_write", "path": "workspace/k.txt", "content": "abc"}],
        }]},
        self.ALLOWED,
    )

  def test_mocks_empty_state_effects_passes(self):
    _validate_generated_round_mocks(
        {"mock_subagents": [{
            "agent_role": "Casey_Database_Admin",
            "mock_response": "just answering, no side effects",
            "state_effects": [],
        }]},
        self.ALLOWED,
    )

  # --- merge mirrors the pipeline's instruction+mocks -> l3_narr flow -------

  def test_split_merge_passes_narrative_safety_net(self):
    # Mimic the pipeline: instruction call yields prose-only rounds; mocks call
    # yields mock_subagents; they merge into the single narrative shape that the
    # legacy _validate_generated_round_narrative still guards.
    l3_instr = {"rounds": [{
        "round_id": "r01", "round_type": "action",
        "instruction": "Inspect the dashboard and message ops.", "update_ids": [],
    }]}
    l3_mocks = {"mock_subagents": [{
        "agent_role": "Casey_Database_Admin", "mock_response": "confirmed",
        "state_effects": ["file create --path workspace/note.txt --content ok"],
    }]}
    _validate_generated_round_mocks(l3_mocks, self.ALLOWED)
    merged = {"rounds": l3_instr["rounds"]}
    merged["rounds"][0]["mock_subagents"] = l3_mocks["mock_subagents"]
    narrative_payload = {"rounds": merged["rounds"], "updates": [], "assets": []}
    _validate_generated_round_narrative(narrative_payload, "r01", [])
    self.assertEqual(
        narrative_payload["rounds"][0]["mock_subagents"][0]["agent_role"],
        "Casey_Database_Admin",
    )


class TestNormalizeCommandQuoting(unittest.TestCase):
  """POSIX re-quoting of generated commands (apostrophe/quote repair).

  Regression guard for the generation-time drop where a free-text flag value
  wrapped in single quotes with an inner apostrophe escaped as ``\\'`` made
  ``shlex.split`` raise 'No closing quotation', failing spec validation and
  burning every retry until the scenario was discarded.
  """

  ALLOWED = ["file", "asset", "message", "agent", "system"]

  def _assert_parses(self, cmd):
    try:
      return shlex.split(cmd)
    except ValueError as exc:
      self.fail(f"command not POSIX-parseable: {cmd!r} ({exc})")

  # --- normalize_command_quoting: repair path ------------------------------

  def test_repairs_apostrophe_in_single_quoted_message(self):
    bad = ("message send --channel ops --target oncall --message "
           "'The gateway shows yesterday\\'s stale data, ignore it.'")
    with self.assertRaises(ValueError):  # precondition: original is broken
      shlex.split(bad)
    fixed = normalize_command_quoting(bad)
    toks = self._assert_parses(fixed)
    self.assertEqual(
        toks[toks.index("--message") + 1],
        "The gateway shows yesterday's stale data, ignore it.",
    )

  def test_repairs_apostrophe_in_agent_delegate_instruction(self):
    bad = ("agent delegate --agent worker --instruction "
           "'Check the file\\'s contents and report back.'")
    with self.assertRaises(ValueError):
      shlex.split(bad)
    fixed = normalize_command_quoting(bad)
    toks = self._assert_parses(fixed)
    self.assertEqual(
        toks[toks.index("--instruction") + 1],
        "Check the file's contents and report back.",
    )

  # --- normalize_command_quoting: no-op / passthrough ----------------------

  def test_valid_single_quoted_unchanged(self):
    good = "message send --channel ops --target oncall --message 'all clear'"
    self.assertEqual(normalize_command_quoting(good), good)

  def test_valid_double_quoted_unchanged(self):
    good = 'message send --channel ops --target oncall --message "it is fine"'
    self.assertEqual(normalize_command_quoting(good), good)

  def test_no_quotes_unchanged(self):
    self.assertEqual(
        normalize_command_quoting("system status --json"), "system status --json")

  def test_empty_unchanged(self):
    self.assertEqual(normalize_command_quoting(""), "")

  def test_unrepairable_returns_original(self):
    # Dangling quote on a positional with no known free-text flag to re-quote:
    # nothing to repair, so the original is returned (never made worse).
    bad = "cat 'foo"
    self.assertEqual(normalize_command_quoting(bad), bad)

  def test_repaired_command_passes_spec_validation(self):
    bad = ("message send --channel ops --target oncall --message "
           "'ignore yesterday\\'s test'")
    fixed = normalize_command_quoting(bad)
    spec = validate_command_against_spec(fixed, allowed_families=self.ALLOWED)
    self.assertEqual(spec.name, "message.send")

  # --- round-command validator wiring (mirrors pipeline :1840) --------------

  def test_round_command_validator_accepts_normalized_apostrophe(self):
    bad = ("message send --channel ops --target oncall --message "
           "'clear yesterday\\'s flag'")
    # The raw command is rejected by the round-command validator ...
    with self.assertRaises(ValueError):
      _validate_generated_round_commands(
          {"commands": [{"step_number": 1, "command": bad}]}, self.ALLOWED)
    # ... but the normalized form (what the pipeline stores) validates cleanly.
    _validate_generated_round_commands(
        {"commands": [{"step_number": 1, "command": normalize_command_quoting(bad)}]},
        self.ALLOWED,
    )

  # --- _normalize_mock_state_effect_quoting (mirrors pipeline :1374) --------

  def test_mock_normalizer_repairs_string_effect_in_place(self):
    payload = {"mock_subagents": [{
        "agent_role": "Casey_Database_Admin", "mock_response": "done",
        "state_effects": [
            "message send --channel ops --target oncall --message 'it\\'s fixed'"],
    }]}
    _normalize_mock_state_effect_quoting(payload)
    # Validation (which shlex.splits every effect) now passes.
    _validate_generated_round_mocks(payload, self.ALLOWED)
    eff = payload["mock_subagents"][0]["state_effects"][0]
    toks = shlex.split(eff)
    self.assertEqual(toks[toks.index("--message") + 1], "it's fixed")

  def test_mock_normalizer_repairs_command_dict_in_place(self):
    payload = {"mock_subagents": [{
        "agent_role": "Casey_Database_Admin", "mock_response": "done",
        "state_effects": [{
            "effect_type": "command_execution",
            "command": ("agent delegate --agent worker --instruction "
                        "'handle Bob\\'s escalation'"),
        }],
    }]}
    _normalize_mock_state_effect_quoting(payload)
    _validate_generated_round_mocks(payload, self.ALLOWED)

  def test_mock_normalizer_leaves_file_write_dict_untouched(self):
    fw = {"type": "file_create", "path": "workspace/k.txt", "content": "abc"}
    payload = {"mock_subagents": [{
        "agent_role": "Casey_Database_Admin", "mock_response": "done",
        "state_effects": [dict(fw)],
    }]}
    _normalize_mock_state_effect_quoting(payload)
    self.assertEqual(payload["mock_subagents"][0]["state_effects"][0], fw)


class TestNormalizeMockDurations(unittest.TestCase):
  """Option 1 deterministic clamp for subagent_delay mock durations.

  Regression guard for the silent version drop: strict check #3
  (:func:`validate_conflict_scenario_strict`) requires every mock in a
  ``subagent_delay`` scenario to declare a positive ``virtual_duration_seconds``,
  but the LLM frequently emits ``0`` for the non-decisive helper mocks -- which
  fails validation AFTER materialization and drops the whole version from the
  dataset. The clamp fixes those in place while preserving the decisive slow
  mock's LLM-provided latency, and is a no-op for non-delay scenarios (E6).
  """

  TAXONOMY = ["time_window_expiration", "subagent_delay"]

  def _dur(self, payload, idx=0):
    return payload["mock_subagents"][idx]["virtual_duration_seconds"]

  def test_clamps_zero_duration_to_default(self):
    payload = {"mock_subagents": [
        {"agent_role": "A", "virtual_duration_seconds": 0}]}
    _normalize_mock_durations(payload, self.TAXONOMY)
    self.assertEqual(self._dur(payload), 10)

  def test_clamps_negative_duration_to_default(self):
    payload = {"mock_subagents": [
        {"agent_role": "A", "virtual_duration_seconds": -5}]}
    _normalize_mock_durations(payload, self.TAXONOMY)
    self.assertEqual(self._dur(payload), 10)

  def test_clamps_missing_key_to_default(self):
    payload = {"mock_subagents": [{"agent_role": "A"}]}
    _normalize_mock_durations(payload, self.TAXONOMY)
    self.assertEqual(self._dur(payload), 10)

  def test_clamps_non_numeric_duration_to_default(self):
    payload = {"mock_subagents": [
        {"agent_role": "A", "virtual_duration_seconds": None},
        {"agent_role": "B", "virtual_duration_seconds": "not-a-number"}]}
    _normalize_mock_durations(payload, self.TAXONOMY)
    self.assertEqual(self._dur(payload, 0), 10)
    self.assertEqual(self._dur(payload, 1), 10)

  def test_preserves_positive_duration(self):
    # The decisive slow sub-agent keeps its LLM-provided latency; only the
    # zero-duration helper is clamped.
    payload = {"mock_subagents": [
        {"agent_role": "Iris_SLA_Monitor", "virtual_duration_seconds": 40},
        {"agent_role": "Harper_Support_Lead", "virtual_duration_seconds": 0}]}
    _normalize_mock_durations(payload, self.TAXONOMY)
    self.assertEqual(self._dur(payload, 0), 40)
    self.assertEqual(self._dur(payload, 1), 10)

  def test_noop_when_taxonomy_lacks_subagent_delay(self):
    # E6: a non-delay scenario is byte-for-byte unaffected.
    payload = {"mock_subagents": [
        {"agent_role": "A", "virtual_duration_seconds": 0}]}
    _normalize_mock_durations(payload, ["temporal_staleness"])
    self.assertEqual(self._dur(payload), 0)

  def test_noop_when_taxonomy_none(self):
    payload = {"mock_subagents": [
        {"agent_role": "A", "virtual_duration_seconds": 0}]}
    _normalize_mock_durations(payload, None)
    self.assertEqual(self._dur(payload), 0)

  def test_custom_default_seconds(self):
    payload = {"mock_subagents": [
        {"agent_role": "A", "virtual_duration_seconds": 0}]}
    _normalize_mock_durations(payload, self.TAXONOMY, default_seconds=25)
    self.assertEqual(self._dur(payload), 25)

  def test_missing_or_malformed_payload_is_safe(self):
    # No crash on absent / non-list / non-dict mock entries.
    _normalize_mock_durations({}, self.TAXONOMY)
    _normalize_mock_durations({"mock_subagents": None}, self.TAXONOMY)
    _normalize_mock_durations({"mock_subagents": ["not-a-dict"]}, self.TAXONOMY)


class TestScreenplayCastTriggersSplit(unittest.TestCase):
  """Tier 2.1 (design 5.1): cast-first then triggers, with the new
  participant-in-cast validation on the merged screenplay_outline."""

  def _cast(self, *role_names):
    return {
        "summary": "s",
        "screenplay_outline": {
            "conflict_arc": "arc",
            "character_cast": [
                {"role_name": r, "dramatic_role": "d", "stance": "s",
                 "motivation": "m"}
                for r in role_names
            ],
        },
    }

  def _trigger(self, *participants):
    return {
        "conflict_name": "c1",
        "description": "d",
        "workflow_block": "w",
        "resolution_cues": "r",
        "affected_participants": list(participants),
        "evidence_sources": ["notes.md"],
    }

  def test_missing_triggers_list_fails(self):
    with self.assertRaises(ValueError):
      _validate_generated_screenplay_triggers(self._cast("Avery_Ops"), {})

  def test_participant_in_cast_passes(self):
    cast = self._cast("Avery_Ops", "Blair_Dev")
    triggers = {"narrative_conflict_triggers": [self._trigger("Avery_Ops")]}
    _validate_generated_screenplay_triggers(cast, triggers)

  def test_participant_not_in_cast_fails(self):
    cast = self._cast("Avery_Ops", "Blair_Dev")
    triggers = {"narrative_conflict_triggers": [self._trigger("Casey_DBA")]}
    with self.assertRaises(ValueError) as ctx:
      _validate_generated_screenplay_triggers(cast, triggers)
    self.assertIn("Casey_DBA", str(ctx.exception))
    self.assertIn("not in the character cast", str(ctx.exception))

  def test_empty_cast_skips_check(self):
    # No cast role_names to anchor against -> reference check is skipped.
    cast = self._cast()
    triggers = {"narrative_conflict_triggers": [self._trigger("Anyone")]}
    _validate_generated_screenplay_triggers(cast, triggers)

  def test_scalar_participant_string_in_cast_passes(self):
    cast = self._cast("Avery_Ops")
    trig = self._trigger()
    trig["affected_participants"] = "Avery_Ops"
    _validate_generated_screenplay_triggers(
        cast, {"narrative_conflict_triggers": [trig]})

  def test_scalar_participant_string_not_in_cast_fails(self):
    cast = self._cast("Avery_Ops")
    trig = self._trigger()
    trig["affected_participants"] = "Ghost_Name"
    with self.assertRaises(ValueError):
      _validate_generated_screenplay_triggers(
          cast, {"narrative_conflict_triggers": [trig]})

  def test_non_dict_trigger_fails(self):
    cast = self._cast("Avery_Ops")
    with self.assertRaises(ValueError):
      _validate_generated_screenplay_triggers(
          cast, {"narrative_conflict_triggers": ["not a dict"]})

  def test_empty_triggers_with_cast_passes(self):
    _validate_generated_screenplay_triggers(
        self._cast("Avery_Ops"), {"narrative_conflict_triggers": []})

  def test_split_merge_reconstructs_screenplay_outline(self):
    # The pipeline merge: cast outline + triggers -> single screenplay_outline
    # shape with all three keys, then it passes the participant check.
    cast = self._cast("Avery_Ops", "Blair_Dev")
    triggers = {"narrative_conflict_triggers": [self._trigger("Blair_Dev")]}
    _validate_generated_screenplay_triggers(cast, triggers)
    outline = dict(cast["screenplay_outline"])
    outline["narrative_conflict_triggers"] = triggers[
        "narrative_conflict_triggers"]
    merged = {"summary": cast["summary"], "screenplay_outline": outline}
    self.assertEqual(
        sorted(merged["screenplay_outline"].keys()),
        ["character_cast", "conflict_arc", "narrative_conflict_triggers"],
    )
    self.assertEqual(
        merged["screenplay_outline"]["narrative_conflict_triggers"][0][
            "affected_participants"],
        ["Blair_Dev"],
    )


class TestAgentRosterMocksSplit(unittest.TestCase):
  """Tier 2.2 (design 5.2): roster-first then mock subagents, with the new
  mock-agent_role-in-roster validation before merging into l1_agents."""

  def _roster(self, *roles):
    return {
        "agents": [
            {"agent_role": r, "agent_name": f"{r} Agent",
             "capabilities": ["file_read"], "private_knowledge": [],
             "virtual_duration_seconds": 15}
            for r in roles
        ],
    }

  def _mock(self, role):
    return {
        "agent_role": role,
        "mock_response": "ok",
        "virtual_duration_seconds": 15,
        "state_effects": [],
    }

  def test_missing_mock_subagents_list_fails(self):
    with self.assertRaises(ValueError):
      _validate_generated_agent_mock_subagents(self._roster("assistant"), {})

  def test_mock_subagents_not_a_list_fails(self):
    with self.assertRaises(ValueError):
      _validate_generated_agent_mock_subagents(
          self._roster("assistant"), {"mock_subagents": "assistant"})

  def test_role_in_roster_passes(self):
    roster = self._roster("assistant", "reviewer")
    mocks = {"mock_subagents": [self._mock("assistant")]}
    _validate_generated_agent_mock_subagents(roster, mocks)

  def test_role_not_in_roster_fails(self):
    roster = self._roster("assistant", "reviewer")
    mocks = {"mock_subagents": [self._mock("ghost")]}
    with self.assertRaises(ValueError) as ctx:
      _validate_generated_agent_mock_subagents(roster, mocks)
    self.assertIn("ghost", str(ctx.exception))
    self.assertIn("not in", str(ctx.exception))

  def test_empty_roster_skips_check(self):
    # No roster agent_roles to anchor against -> reference check is skipped.
    roster = self._roster()
    mocks = {"mock_subagents": [self._mock("anyone")]}
    _validate_generated_agent_mock_subagents(roster, mocks)

  def test_non_dict_mock_fails(self):
    roster = self._roster("assistant")
    with self.assertRaises(ValueError):
      _validate_generated_agent_mock_subagents(
          roster, {"mock_subagents": ["not a dict"]})

  def test_empty_mocks_with_roster_passes(self):
    _validate_generated_agent_mock_subagents(
        self._roster("assistant"), {"mock_subagents": []})

  def test_split_merge_reconstructs_agents_shape(self):
    # The pipeline merge: roster agents + mocks -> single l1_agents shape with
    # both keys, which is what the interactions layer and layer1_payload expect.
    roster = self._roster("assistant", "reviewer")
    mocks = {"mock_subagents": [self._mock("reviewer")]}
    _validate_generated_agent_mock_subagents(roster, mocks)
    merged = {
        "agents": roster.get("agents", []),
        "mock_subagents": mocks.get("mock_subagents", []),
    }
    self.assertEqual(sorted(merged.keys()), ["agents", "mock_subagents"])
    self.assertEqual(len(merged["agents"]), 2)
    self.assertEqual(
        merged["mock_subagents"][0]["agent_role"], "reviewer")


class TestExpandPhaseTimelineAssetsSplit(unittest.TestCase):
  """Tier 3.1 (design 6.1): phase expansion is split into an independently
  retried timeline half (salvage fallback) and assets half (degrades to
  assets:[]). The assets validator is reused by _validate_expanded_phase."""

  def _asset(self, **overrides):
    a = {
        "id": "ops_dashboard_chart_01",
        "path": "charts/ops_dashboard_chart_01.png",
        "platform_id": "ops_dashboard",
        "mime_type": "image/png",
        "available_from_round": 1,
    }
    a.update(overrides)
    return a

  # --- assets-half validator ---
  def test_missing_assets_key_fails(self):
    with self.assertRaises(ValueError):
      _validate_expanded_phase_assets({})

  def test_assets_not_a_list_fails(self):
    with self.assertRaises(ValueError):
      _validate_expanded_phase_assets({"assets": "nope"})

  def test_empty_assets_passes(self):
    _validate_expanded_phase_assets({"assets": []})

  def test_valid_asset_passes(self):
    _validate_expanded_phase_assets({"assets": [self._asset()]})

  def test_non_dict_asset_fails(self):
    with self.assertRaises(ValueError):
      _validate_expanded_phase_assets({"assets": ["not a dict"]})

  def test_asset_missing_id_fails(self):
    with self.assertRaises(ValueError):
      _validate_expanded_phase_assets({"assets": [self._asset(id="")]})

  def test_asset_missing_path_fails(self):
    with self.assertRaises(ValueError):
      _validate_expanded_phase_assets({"assets": [self._asset(path="")]})

  def test_asset_missing_platform_id_fails(self):
    with self.assertRaises(ValueError):
      _validate_expanded_phase_assets({"assets": [self._asset(platform_id="")]})

  def test_asset_bad_prefix_fails(self):
    with self.assertRaises(ValueError) as ctx:
      _validate_expanded_phase_assets(
          {"assets": [self._asset(id="wrong_prefix_chart")]})
    self.assertIn("platform prefix", str(ctx.exception))

  # --- timeline half still validates independently (no assets key) ---
  def test_timeline_only_passes_without_assets(self):
    _validate_expanded_phase(
        {"timeline": [{"round_id": "r01"}, {"round_id": "r02"}]},
        ["r01", "r02"])

  def test_timeline_empty_fails(self):
    with self.assertRaises(ValueError):
      _validate_expanded_phase({"timeline": []}, ["r01"])

  def test_timeline_missing_expected_round_fails(self):
    with self.assertRaises(ValueError):
      _validate_expanded_phase({"timeline": [{"round_id": "r01"}]}, ["r02"])

  # --- merge reconstructs the single expanded_payload shape ---
  def test_split_merge_reconstructs_expanded_shape(self):
    timeline_payload = {"timeline": [{"round_id": "r01"}, {"round_id": "r02"}]}
    assets_payload = {"assets": [self._asset()]}
    merged = {
        "timeline": timeline_payload.get("timeline", []),
        "assets": assets_payload.get("assets", []),
    }
    _validate_expanded_phase(merged, ["r01", "r02"])
    self.assertEqual(sorted(merged.keys()), ["assets", "timeline"])
    self.assertEqual(len(merged["timeline"]), 2)
    self.assertEqual(merged["assets"][0]["id"], "ops_dashboard_chart_01")


class TestPlansSplit(unittest.TestCase):
  """Tier 3.3 (design 6.3): the monolithic conflict_layer0c_plans call is split
  into three independently-retried single-plan calls (assets -> updates ->
  interactions), each aware of the prior finalized plans, then merged back into
  the layer0c shape. layer0c has no validator, so the split is verified
  behaviorally: the pipeline calls the three new layers (never the monolith), in
  dependency order, feeding each later plan the prior finalized plan(s)."""

  def _seed(self):
    return ConflictVLMSeed(
        slug="multimodal_temporal_conflict_01",
        domain="composite",
        title="Conflict Test",
        target_rounds=2,
        core_conflict="Mock conflict.",
        conflict_taxonomy=["temporal_staleness"],
        participants=["Ops_Agent", "Dev_Agent"],
        required_constraints=["Mock constraint."],
        allowed_tool_families=["file", "asset"],
        modalities=["chart"],
        input_policy={"default_mode": "path"},
        generation_policy={"chart_provider": "mock"},
        min_round_steps=4,
        require_final_verification=False,
    )

  def test_pipeline_calls_split_plan_layers_not_monolith(self):
    calls: list[str] = []
    payloads: dict[str, Any] = {}

    class _RecordingClient(MockConflictLLMClient):

      def generate_layer(self, layer, payload):
        calls.append(layer)
        payloads[layer] = payload
        return super().generate_layer(layer, payload)

    pipeline = ConflictLLMPipeline(
        client=_RecordingClient(), progress=lambda msg: None)
    res = pipeline.generate(self._seed())

    # The three split plan layers are each called exactly once...
    self.assertEqual(calls.count("conflict_layer0c_assets_plan"), 1)
    self.assertEqual(calls.count("conflict_layer0c_updates_plan"), 1)
    self.assertEqual(calls.count("conflict_layer0c_interactions_plan"), 1)
    # ...and the pre-split monolith is never called.
    self.assertNotIn("conflict_layer0c_plans", calls)

    # They run in dependency order: assets -> updates -> interactions.
    self.assertLess(
        calls.index("conflict_layer0c_assets_plan"),
        calls.index("conflict_layer0c_updates_plan"))
    self.assertLess(
        calls.index("conflict_layer0c_updates_plan"),
        calls.index("conflict_layer0c_interactions_plan"))

    # Each later plan is made aware of the prior finalized plan(s).
    self.assertIn("assets_plan", payloads["conflict_layer0c_updates_plan"])
    self.assertIn(
        "assets_plan", payloads["conflict_layer0c_interactions_plan"])
    self.assertIn(
        "updates_plan", payloads["conflict_layer0c_interactions_plan"])

    # generate() still completes and produces the requested rounds.
    self.assertIn("rounds", res)
    self.assertEqual(len(res["rounds"]), 2)

  def test_split_merge_reconstructs_layer0c_shape(self):
    # The pipeline merge: three single-plan payloads -> one layer0c shape with
    # all three plan keys, which is what layer0_phases and downstream expect.
    assets_plan = {"global_assets_plan": [{"asset_id": "a1"}]}
    updates_plan = {"global_updates_plan": [{"update_id": "u1"}]}
    interactions_plan = {
        "global_interactions_plan": [{"interaction_id": "i1"}]}
    merged = {
        "global_assets_plan": assets_plan["global_assets_plan"],
        "global_updates_plan": updates_plan["global_updates_plan"],
        "global_interactions_plan": interactions_plan[
            "global_interactions_plan"],
    }
    self.assertEqual(
        sorted(merged.keys()),
        ["global_assets_plan", "global_interactions_plan",
         "global_updates_plan"])
    self.assertEqual(merged["global_assets_plan"][0]["asset_id"], "a1")
    self.assertEqual(merged["global_updates_plan"][0]["update_id"], "u1")
    self.assertEqual(
        merged["global_interactions_plan"][0]["interaction_id"], "i1")


class TestConflictPluginsTaxonomy(unittest.TestCase):

  def test_taxonomy_resolves_all_plugins(self):
    from clasharness.scenario.plugins import get_plugins_for_taxonomy
    
    # Test individual lookups
    plugins_staleness = get_plugins_for_taxonomy(["temporal_staleness"])
    self.assertEqual(len(plugins_staleness), 1)
    self.assertEqual(plugins_staleness[0].name, "temporal_staleness")

    plugins_order = get_plugins_for_taxonomy(["dependency_ordering"])
    self.assertEqual(len(plugins_order), 1)
    self.assertEqual(plugins_order[0].name, "dependency_ordering")

    plugins_deadlock = get_plugins_for_taxonomy(["deadlock_dependency"])
    self.assertEqual(len(plugins_deadlock), 1)
    self.assertEqual(plugins_deadlock[0].name, "deadlock_dependency")

    plugins_mismatch = get_plugins_for_taxonomy(["evidence_mismatch"])
    self.assertEqual(len(plugins_mismatch), 1)
    self.assertEqual(plugins_mismatch[0].name, "evidence_mismatch")

    plugins_window = get_plugins_for_taxonomy(["time_window_expiration"])
    self.assertEqual(len(plugins_window), 1)
    self.assertEqual(plugins_window[0].name, "time_window_expiration")

    # M2 Order-class plugin (deploy family: governance authorization ordering)
    plugins_gate = get_plugins_for_taxonomy(["controlled_deployment_gate"])
    self.assertEqual(len(plugins_gate), 1)
    self.assertEqual(plugins_gate[0].name, "controlled_deployment_gate")

    # Test multi-selection
    multi = get_plugins_for_taxonomy(["temporal_staleness", "credential_lock"])
    self.assertEqual(len(multi), 2)
    self.assertEqual({p.name for p in multi}, {"temporal_staleness", "credential_lock"})

    # M2 seed it_deployment_gate resolves both tokens to distinct plugins.
    gate_order = get_plugins_for_taxonomy(
        ["controlled_deployment_gate", "dependency_ordering"])
    self.assertEqual(len(gate_order), 2)
    self.assertEqual(
        {p.name for p in gate_order},
        {"controlled_deployment_gate", "dependency_ordering"})


class CoerceDependencyStepTest(unittest.TestCase):
  """Regression tests for _coerce_dependency_step.

  Guards against the crash where LLM-returned dict dependency steps caused
  ``TypeError: sequence item 0: expected str instance, dict found`` and the
  entire generated scenario was discarded.
  """

  def test_plain_string_passthrough(self):
    self.assertEqual(
        _coerce_dependency_step("file read config/edge_proxy.yaml"),
        "file read config/edge_proxy.yaml",
    )

  def test_dict_preferred_keys(self):
    self.assertEqual(_coerce_dependency_step({"step": "delegate"}), "delegate")
    self.assertEqual(
        _coerce_dependency_step({"action": "a", "command": "cmd"}), "a"
    )
    self.assertEqual(
        _coerce_dependency_step({"command": "message send"}), "message send"
    )

  def test_dict_without_known_keys_falls_back_to_json(self):
    self.assertEqual(
        _coerce_dependency_step({"foo": "bar"}), '{"foo": "bar"}'
    )

  def test_non_string_scalar(self):
    self.assertEqual(_coerce_dependency_step(123), "123")

  def test_join_of_mixed_steps_does_not_raise(self):
    steps = ["extract", {"step": "delegate"}, {"foo": "bar"}, 7]
    joined = " -> ".join(_coerce_dependency_step(s) for s in steps)
    self.assertEqual(joined, 'extract -> delegate -> {"foo": "bar"} -> 7')


class TestPlatformIdSanitization(unittest.TestCase):
  """Stage 1: platform_id sanitization + validator value safety-net.

  platform_id names a web UI surface and must be a clean lowercase slug. The
  model tends to copy garbage from adjacent prompt context (provider flags,
  generator names, kinds, agent roles, asset ids, file paths). _sanitize_platform_id
  strips/rejects it; the asset validators reject any residual non-slug value.
  """

  def test_clean_slug_passes_through(self):
    self.assertEqual(_sanitize_platform_id("ops_dashboard"), "ops_dashboard")
    self.assertEqual(_sanitize_platform_id("billing-portal"), "billing-portal")
    self.assertEqual(_sanitize_platform_id("api2"), "api2")

  def test_uppercase_is_lowercased(self):
    self.assertEqual(_sanitize_platform_id("Ops_Console"), "ops_console")

  def test_strips_file_extension(self):
    # Root cause of the doubled ".png" filenames: platform_id carried an ext.
    self.assertEqual(_sanitize_platform_id("grafana.png"), "grafana")
    self.assertEqual(_sanitize_platform_id("report.jpeg"), "report")

  def test_strips_path_components(self):
    self.assertEqual(
        _sanitize_platform_id("docs/security_playbook.md"), "security_playbook")
    self.assertEqual(
        _sanitize_platform_id("logs/pipeline_build.log"), "pipeline_build")

  def test_rejects_provider_flags(self):
    for bad in ("vertex", "template", "google_model", "mock", "gemini"):
      self.assertEqual(_sanitize_platform_id(bad), "", msg=f"expected reject: {bad}")

  def test_rejects_agent_roles(self):
    for bad in ("Ops_Agent", "Dev_Agent", "DB_Admin_Agent"):
      self.assertEqual(_sanitize_platform_id(bad), "", msg=f"expected reject: {bad}")

  def test_rejects_asset_ids(self):
    self.assertEqual(_sanitize_platform_id("asset_bg_01"), "")

  def test_rejects_generator_and_kind_tokens(self):
    for bad in ("template_chart", "google_image", "chart", "rich_text",
                "natural_image"):
      self.assertEqual(_sanitize_platform_id(bad), "", msg=f"expected reject: {bad}")

  def test_rejects_generic_fallbacks(self):
    for bad in ("workspace", "vlm_evidence", "vlm_service", "default_platform"):
      self.assertEqual(_sanitize_platform_id(bad), "", msg=f"expected reject: {bad}")

  def test_empty_and_none(self):
    self.assertEqual(_sanitize_platform_id(None), "")
    self.assertEqual(_sanitize_platform_id(""), "")
    self.assertEqual(_sanitize_platform_id("   "), "")
    self.assertEqual(_sanitize_platform_id("///"), "")

  def test_slugifies_punctuation_and_spaces(self):
    self.assertEqual(_sanitize_platform_id("Ops Dashboard!!"), "ops_dashboard")
    self.assertEqual(_sanitize_platform_id("api-gateway v2"), "api-gateway_v2")

  # --- normalize applies sanitization end-to-end ---
  def test_normalize_sanitizes_and_fixes_double_extension(self):
    payload = {
        "assets": [{
            "id": "chart_01",
            "path": "charts/chart_01.png",
            "platform_id": "Grafana.png",
        }]
    }
    out = _normalize_and_prefix_assets(payload)
    asset = out["assets"][0]
    self.assertEqual(asset["platform_id"], "grafana")
    self.assertEqual(asset["id"], "grafana_chart_01")
    self.assertEqual(asset["path"], "charts/grafana_chart_01.png")

  def test_normalize_garbage_platform_falls_back_to_default(self):
    payload = {"assets": [{"id": "a", "path": "p/a.png", "platform_id": "vertex"}]}
    out = _normalize_and_prefix_assets(payload)
    self.assertEqual(out["assets"][0]["platform_id"], "default_platform")

  # --- validator safety-net rejects residual non-slug values ---
  def test_validator_rejects_pathy_platform_id(self):
    bad_asset = {
        "id": "ops_dashboard_chart",
        "path": "charts/ops_dashboard_chart.png",
        "platform_id": "docs/x.png",
    }
    with self.assertRaisesRegex(ValueError, "invalid platform_id"):
      _validate_expanded_phase_assets({"assets": [bad_asset]})

  def test_validator_rejects_uppercase_platform_id(self):
    bad_asset = {
        "id": "Ops_chart",
        "path": "charts/Ops_chart.png",
        "platform_id": "Ops",
    }
    with self.assertRaisesRegex(ValueError, "invalid platform_id"):
      _validate_expanded_phase_assets({"assets": [bad_asset]})


class TestPlatformVocabulary(unittest.TestCase):
  """Stage 2: per-scenario platform vocabulary + deterministic soft remap.

  Stage 1 stripped garbage platform_ids but collapsed anything unusable onto a
  single ``default_platform``, so scenarios lost web-UI diversity. Stage 2 has
  the LLM author a small vocabulary of distinct surfaces per scenario
  (``_build_platform_vocab``, backfilled from a fixed fallback) and
  deterministically spreads out-of-vocabulary / missing asset platform_ids
  across it (``_stable_platform_choice``) instead of defaulting. In-vocabulary
  choices are preserved so the model's deliberate grouping survives.
  """

  # --- _dedupe_preserve ---
  def test_dedupe_preserves_order_and_drops_falsy(self):
    self.assertEqual(
        _dedupe_preserve(["b", "a", "b", "", "c", "a"]), ["b", "a", "c"])
    self.assertEqual(_dedupe_preserve([]), [])

  # --- _build_platform_vocab: sources ---
  def test_vocab_from_platforms_field(self):
    vocab = _build_platform_vocab(
        ["ops_console", "billing_portal", "admin_portal"], None)
    self.assertEqual(vocab, ["ops_console", "billing_portal", "admin_portal"])

  def test_vocab_from_asset_platform_ids(self):
    plan = [
        {"platform_id": "ops_console"},
        {"platform_id": "billing_portal"},
        {"platform_id": "log_viewer"},
    ]
    self.assertEqual(
        _build_platform_vocab(None, plan),
        ["ops_console", "billing_portal", "log_viewer"])

  def test_vocab_merges_both_sources_then_backfills(self):
    # platforms field first, then asset platform_ids, then fallback backfill.
    vocab = _build_platform_vocab(["ops_console"], [{"platform_id": "billing_portal"}])
    self.assertEqual(vocab[0], "ops_console")
    self.assertEqual(vocab[1], "billing_portal")
    self.assertGreaterEqual(len(vocab), _MIN_PLATFORM_VOCAB)

  # --- _build_platform_vocab: cleaning / dedupe / backfill ---
  def test_vocab_dedupes_preserving_order(self):
    vocab = _build_platform_vocab(
        ["ops_console", "ops_console", "billing_portal",
         "billing_portal", "admin_portal"], None)
    self.assertEqual(vocab, ["ops_console", "billing_portal", "admin_portal"])

  def test_vocab_drops_garbage_candidates(self):
    vocab = _build_platform_vocab(
        ["vertex", "template", "Ops_Console.png", "Ops_Agent", "billing_portal"],
        [{"platform_id": "asset_bg_01"}])
    # Garbage (provider flags, agent roles, asset ids) never enters the vocab.
    for bad in ("vertex", "template", "Ops_Agent", "asset_bg_01"):
      self.assertNotIn(bad, vocab)
    # Sanitizable candidates survive (extension stripped, lowercased).
    self.assertIn("ops_console", vocab)
    self.assertIn("billing_portal", vocab)
    # Every member is a clean slug.
    for p in vocab:
      self.assertEqual(_sanitize_platform_id(p), p)

  def test_vocab_backfills_empty_to_minimum(self):
    vocab = _build_platform_vocab(None, None)
    self.assertEqual(len(vocab), _MIN_PLATFORM_VOCAB)
    self.assertEqual(vocab, list(_DEFAULT_PLATFORM_VOCAB[:_MIN_PLATFORM_VOCAB]))

  def test_vocab_ignores_non_list_inputs(self):
    vocab = _build_platform_vocab("not-a-list", "also-not-a-list")
    self.assertEqual(vocab, list(_DEFAULT_PLATFORM_VOCAB[:_MIN_PLATFORM_VOCAB]))

  def test_vocab_always_meets_minimum(self):
    for pf, plan in (
        (None, None),
        (["ops_console"], None),
        (["vertex", "template"], [{"platform_id": "mock"}]),
        ([], []),
    ):
      self.assertGreaterEqual(
          len(_build_platform_vocab(pf, plan)), _MIN_PLATFORM_VOCAB,
          msg=f"vocab too small for pf={pf} plan={plan}")

  def test_vocab_backfill_does_not_duplicate_existing(self):
    # "ops_console" is also the first fallback; backfill must skip it.
    vocab = _build_platform_vocab(["ops_console"], None)
    self.assertEqual(len(vocab), len(set(vocab)))
    self.assertEqual(vocab.count("ops_console"), 1)

  # --- _stable_platform_choice ---
  def test_stable_choice_is_deterministic(self):
    vocab = ["ops_console", "billing_portal", "admin_portal"]
    first = _stable_platform_choice("asset_x", vocab)
    for _ in range(5):
      self.assertEqual(_stable_platform_choice("asset_x", vocab), first)

  def test_stable_choice_always_in_vocab(self):
    vocab = ["ops_console", "billing_portal", "admin_portal"]
    for i in range(50):
      self.assertIn(_stable_platform_choice(f"asset_{i}", vocab), vocab)

  def test_stable_choice_empty_vocab_falls_back_to_default(self):
    self.assertEqual(_stable_platform_choice("anything", []), _DEFAULT_PLATFORM_ID)

  def test_stable_choice_distributes_across_vocab(self):
    # Deterministic (sha1-based): distinct keys must not all collapse onto one
    # surface -- that collapse is the exact bug Stage 2 fixes.
    vocab = ["ops_console", "billing_portal", "admin_portal"]
    picks = {_stable_platform_choice(f"asset_{i}", vocab) for i in range(30)}
    self.assertGreater(len(picks), 1)

  # --- _normalize_and_prefix_assets WITH a scenario vocabulary ---
  def _vocab(self):
    return ["ops_console", "billing_portal", "admin_portal"]

  def test_normalize_in_vocab_platform_preserved(self):
    payload = {"assets": [{
        "id": "chart_01", "path": "charts/chart_01.png",
        "platform_id": "billing_portal",
    }]}
    out = _normalize_and_prefix_assets(payload, allowed_platforms=self._vocab())
    asset = out["assets"][0]
    self.assertEqual(asset["platform_id"], "billing_portal")
    self.assertEqual(asset["id"], "billing_portal_chart_01")
    self.assertEqual(asset["path"], "charts/billing_portal_chart_01.png")

  def test_normalize_out_of_vocab_remapped_into_vocab(self):
    # A clean but out-of-vocabulary slug is remapped onto a vocab member
    # (deterministically keyed on the asset id), not left as-is.
    vocab = self._vocab()
    payload = {"assets": [{
        "id": "chart_01", "path": "charts/chart_01.png",
        "platform_id": "some_other_surface",
    }]}
    out = _normalize_and_prefix_assets(payload, allowed_platforms=vocab)
    asset = out["assets"][0]
    self.assertIn(asset["platform_id"], vocab)
    self.assertEqual(asset["platform_id"], _stable_platform_choice("chart_01", vocab))
    self.assertTrue(asset["id"].startswith(f"{asset['platform_id']}_"))

  def test_normalize_garbage_with_vocab_maps_to_vocab_not_default(self):
    # Key Stage 2 behavior change: with a vocabulary, unusable platform_ids are
    # spread across it rather than collapsed onto default_platform.
    vocab = self._vocab()
    payload = {"assets": [{
        "id": "chart_01", "path": "charts/chart_01.png",
        "platform_id": "vertex",  # blocklisted -> sanitizes to ""
    }]}
    out = _normalize_and_prefix_assets(payload, allowed_platforms=vocab)
    self.assertIn(out["assets"][0]["platform_id"], vocab)
    self.assertNotEqual(out["assets"][0]["platform_id"], _DEFAULT_PLATFORM_ID)

  def test_normalize_missing_platform_with_vocab_maps_to_vocab(self):
    vocab = self._vocab()
    payload = {"assets": [{"id": "chart_01", "path": "charts/chart_01.png"}]}
    out = _normalize_and_prefix_assets(payload, allowed_platforms=vocab)
    self.assertIn(out["assets"][0]["platform_id"], vocab)

  def test_normalize_remap_key_falls_back_to_path_then_index(self):
    vocab = self._vocab()
    # No id -> key off the path.
    payload = {"assets": [{"path": "charts/only_path.png", "platform_id": "junk!"}]}
    out = _normalize_and_prefix_assets(payload, allowed_platforms=vocab)
    self.assertEqual(
        out["assets"][0]["platform_id"],
        _stable_platform_choice("charts/only_path.png", vocab))

  def test_normalize_no_vocab_garbage_still_defaults(self):
    # Stage 1 behavior must be unchanged when no vocabulary is supplied.
    payload = {"assets": [{"id": "a", "path": "p/a.png", "platform_id": "vertex"}]}
    out = _normalize_and_prefix_assets(payload, allowed_platforms=None)
    self.assertEqual(out["assets"][0]["platform_id"], _DEFAULT_PLATFORM_ID)

  def test_normalize_no_vocab_valid_slug_preserved(self):
    payload = {"assets": [{"id": "a", "path": "p/a.png", "platform_id": "grafana"}]}
    out = _normalize_and_prefix_assets(payload, allowed_platforms=None)
    self.assertEqual(out["assets"][0]["platform_id"], "grafana")

  def test_normalize_spreads_many_assets_across_vocab(self):
    vocab = self._vocab()
    payload = {"assets": [
        {"id": f"chart_{i}", "path": f"charts/chart_{i}.png",
         "platform_id": "junk"}
        for i in range(24)
    ]}
    out = _normalize_and_prefix_assets(payload, allowed_platforms=vocab)
    picks = {a["platform_id"] for a in out["assets"]}
    self.assertTrue(picks.issubset(set(vocab)))
    self.assertGreater(len(picks), 1)

  # --- spec round-trip carries the platforms vocabulary ---
  def test_spec_roundtrip_preserves_platforms(self):
    from clasharness.scenario.conflict_vlm_spec import ConflictVLMScenarioSpec
    spec = ConflictVLMScenarioSpec.from_dict(
        {"scenario_slug": "s", "platforms": ["ops_console", "billing_portal"]})
    self.assertEqual(spec.platforms, ["ops_console", "billing_portal"])
    self.assertEqual(
        spec.to_dict()["platforms"], ["ops_console", "billing_portal"])

  def test_spec_missing_platforms_defaults_empty(self):
    from clasharness.scenario.conflict_vlm_spec import ConflictVLMScenarioSpec
    spec = ConflictVLMScenarioSpec.from_dict({"scenario_slug": "s"})
    self.assertEqual(spec.platforms, [])

  # --- generate() exposes the vocabulary and de-contaminates the seed ---
  def _redaction_seed(self):
    return ConflictVLMSeed(
        slug="test_conflict_redact",
        domain="system",
        title="Redaction Test",
        target_rounds=1,
        core_conflict="Time delay breaks assumptions.",
        conflict_taxonomy=["temporal_staleness"],
        participants=["owner", "assistant"],
        required_constraints=[],
        allowed_tool_families=["agents", "file", "message"],
        min_round_steps=4,
        require_final_verification=False,
    )

  def test_generate_result_exposes_platform_vocabulary(self):
    pipeline = ConflictLLMPipeline(
        client=MockConflictLLMClient(), progress=lambda msg: None)
    res = pipeline.generate(self._redaction_seed())
    self.assertIn("platforms", res)
    self.assertGreaterEqual(len(res["platforms"]), _MIN_PLATFORM_VOCAB)
    # Vocabulary is clean slugs.
    for p in res["platforms"]:
      self.assertEqual(_sanitize_platform_id(p), p)

  def test_generate_redacts_provider_flags_from_seed_payload(self):
    captured: list[dict[str, Any]] = []

    class _RecordingClient(MockConflictLLMClient):

      def generate_layer(self, layer, payload):
        captured.append(copy.deepcopy(payload))
        return super().generate_layer(layer, payload)

    seed = self._redaction_seed()
    original_chart_provider = seed.generation_policy.chart_provider
    self.assertTrue(original_chart_provider)  # sanity: seed carries the flag

    pipeline = ConflictLLMPipeline(
        client=_RecordingClient(), progress=lambda msg: None)
    pipeline.generate(seed)

    seen_a_policy = False
    for payload in captured:
      gp = payload.get("seed", {}).get("generation_policy")
      if not isinstance(gp, dict):
        continue
      seen_a_policy = True
      for leaked in ("chart_provider", "natural_image_provider", "google_model"):
        self.assertNotIn(
            leaked, gp,
            msg=f"provider flag {leaked!r} leaked into LLM seed payload")
      # Non-provider policy fields are retained.
      self.assertIn("chart_layouts", gp)
    self.assertTrue(seen_a_policy, "no seed payload with generation_policy seen")

    # The real seed object is never mutated (asdict made a fresh copy).
    self.assertEqual(seed.generation_policy.chart_provider, original_chart_provider)


class TestSubagentEffectClobber(unittest.TestCase):
  """A1: subagent file-write effects that clobber expert-authored files are
  dropped for BOTH the dict shape and the raw ``file <verb> --path`` CLI-string
  shape, across every write verb (the string form was the leaked B-class bug)."""

  def test_effect_target_dict_write_shapes(self):
    for t in ("file_create", "file_write", "write_file"):
      eff = {"type": t, "path": "reports/summary.md", "content": "x"}
      self.assertEqual(
          _effect_target_file_path(eff), _norm_file_path("reports/summary.md"), t)

  def test_effect_target_string_verbs_including_append(self):
    # The string form (append/write/update/edit ...) is the shape the old
    # dict-only drop missed, letting the subagent clobber the expert write.
    for verb in ("create", "write", "append", "update", "edit", "modify"):
      eff = f"file {verb} --path reports/summary.md --content 'evil'"
      self.assertEqual(
          _effect_target_file_path(eff),
          _norm_file_path("reports/summary.md"), verb)

  def test_effect_target_normalizes_absolute_path(self):
    self.assertEqual(
        _effect_target_file_path("file append --path /workspace/reports/x.md"),
        _norm_file_path("reports/x.md"))

  def test_effect_target_non_file_effect_is_none(self):
    self.assertIsNone(_effect_target_file_path({"type": "command", "command": "system notify"}))
    self.assertIsNone(_effect_target_file_path("system notify --msg hi"))
    self.assertIsNone(_effect_target_file_path("file list"))  # not a write verb
    self.assertIsNone(_effect_target_file_path(42))

  def test_drop_removes_string_and_dict_conflicts_keeps_others(self):
    expert = {_norm_file_path("reports/summary.md")}
    dict_eff = {"type": "file_create", "path": "reports/summary.md", "content": "expert-conflicting"}
    str_eff = "file append --path reports/summary.md --content 'evil'"
    other_file = {"type": "file_create", "path": "reports/subagent_only.md"}
    cmd_eff = {"type": "command", "command": "system notify"}
    kept = _drop_conflicting_subagent_file_effects(
        [dict_eff, str_eff, other_file, cmd_eff], expert)
    # Both conflicting shapes gone; subagent-only file and command survive.
    self.assertIn(other_file, kept)
    self.assertIn(cmd_eff, kept)
    self.assertNotIn(dict_eff, kept)
    self.assertNotIn(str_eff, kept)
    self.assertEqual(len(kept), 2)

  def test_drop_noop_when_no_expert_files(self):
    effects = ["file append --path reports/x.md --content y"]
    self.assertEqual(_drop_conflicting_subagent_file_effects(effects, set()), effects)


class TestOpaqueContentPruning(unittest.TestCase):
  """A2: content assertions against encrypted/binary blobs (``.enc`` etc.) and
  unrendered templates (``*.template``) can never hold on the raw bytes, so they
  are pruned; a bare existence check on the same path is preserved."""

  def test_unassertable_true_for_opaque_and_template(self):
    for p in ("secrets/master.key.enc", "vault/creds.gpg", "data/blob.bin",
              "archive/logs.tar.gz", "img/chart.png", "book.pdf",
              "config/app.yaml.template", "deploy/values.tmpl", "x.j2",
              "schema.proto.bin", "SECRETS/UPPER.ENC"):
      self.assertTrue(_is_content_unassertable_path(p), p)

  def test_unassertable_false_for_plaintext(self):
    for p in ("notes/summary.md", "config.yaml", "state.json", "run.log",
              "report.txt", "a.conf", ""):
      self.assertFalse(_is_content_unassertable_path(p), p)

  def test_has_content_assertion_detects_content_vs_existence(self):
    self.assertTrue(_has_content_assertion({"type": "file_check", "contains": "42"}))
    self.assertTrue(_has_content_assertion({"type": "file_check", "equals": "9"}))
    self.assertTrue(_has_content_assertion({"condition": "contains", "expected": "x"}))
    # Bare existence / empty content -> not a content assertion (kept intact).
    self.assertFalse(_has_content_assertion({"type": "file_check"}))
    self.assertFalse(_has_content_assertion({"type": "file_check", "contains": ""}))
    self.assertFalse(_has_content_assertion({"type": "file_check", "contains": "   "}))

  def test_opaque_existence_check_is_not_pruned_signal(self):
    # An existence check on an opaque file must survive: unassertable path but no
    # content assertion -> the Rule-6 prune predicate is False.
    check = {"type": "file_check", "path": "secrets/master.key.enc"}
    prune = _is_content_unassertable_path(check["path"]) and _has_content_assertion(check)
    self.assertFalse(prune)


class TestResolvedValueLocus(unittest.TestCase):
  """B1: parsing the resolution ``resolved_value_locus`` provenance token."""

  def test_asset_locus(self):
    self.assertEqual(_parse_resolved_locus("asset:chart_01"), ("asset", "chart_01"))

  def test_file_locus_is_ws_normalized(self):
    # Leading slash is stripped so the target matches scaffold paths, which are
    # ``_normalize_ws_path``-ed too.
    self.assertEqual(
        _parse_resolved_locus("file:/reports/out.md"), ("file", "reports/out.md"))
    self.assertEqual(
        _parse_resolved_locus("file: reports/out.md"), ("file", "reports/out.md"))

  def test_computed_synonyms(self):
    for r in ("computed", "derived", "aggregate", "calculated"):
      self.assertEqual(_parse_resolved_locus(r), ("computed", ""), r)

  def test_unknown_and_empty_degrade_to_blank(self):
    for r in (None, "", "   ", "garbage", "asset", "file"):
      self.assertEqual(_parse_resolved_locus(r), ("", ""), repr(r))


class TestScaffoldLocusGating(unittest.TestCase):
  """B2: the locus provenance drives whether a *read* step becomes a file_check.
  A read only asserts the decisive value lives in THAT file when the value
  actually resides there (unknown locus keeps the legacy check as a backstop)."""

  ALLOWED = ["file", "asset", "message", "agent"]

  def _read_checker(self, locus, path="reports/data.md", resolved="42"):
    scaffolds, _ = _scaffold_round_checkers(
        [{"step_number": 1, "command": f"file read --path {path}"}],
        self.ALLOWED, asset_ids=[], resolved_value=resolved,
        resolved_value_locus=locus)
    return scaffolds[0]["checker"]

  def test_asset_locus_read_is_not_filecheck(self):
    # Value comes from an asset/OCR, not the read file -> no doomed file_check.
    self.assertIsNone(self._read_checker("asset:chart_01"))

  def test_computed_locus_read_is_not_filecheck(self):
    self.assertIsNone(self._read_checker("computed"))

  def test_file_locus_other_path_read_is_not_filecheck(self):
    self.assertIsNone(self._read_checker("file:reports/other.md"))

  def test_file_locus_same_path_read_is_filecheck(self):
    chk = self._read_checker("file:reports/data.md")
    self.assertEqual(chk, {"type": "file_check", "path": "reports/data.md", "contains": "42"})

  def test_unknown_locus_keeps_legacy_filecheck(self):
    # Backstop: unknown provenance keeps the legacy check (calibration validates).
    chk = self._read_checker(None)
    self.assertEqual(chk, {"type": "file_check", "path": "reports/data.md", "contains": "42"})

  def test_no_resolved_value_read_stays_plain(self):
    self.assertIsNone(self._read_checker("file:reports/data.md", resolved=""))


class TestDecisionOutcomeValidation(unittest.TestCase):
  """③: light gate on a round's free-form decision_outcome verdict label. It is
  authored (not grounded), so we only enforce present + concise + distinct from
  the decisive value; a bad label feeds the resolution retry loop."""

  def test_valid_label_passes(self):
    # A short label distinct from the decisive value is accepted (no raise).
    _validate_round_decision_outcome("roll back to v4.2", "v4.2")

  def test_empty_raises(self):
    with self.assertRaises(ValueError):
      _validate_round_decision_outcome("", "default_upstream")
    with self.assertRaises(ValueError):
      _validate_round_decision_outcome("   ", "default_upstream")

  def test_too_long_raises(self):
    with self.assertRaises(ValueError):
      _validate_round_decision_outcome("x" * 121, "v")

  def test_equal_to_value_raises_case_insensitive(self):
    with self.assertRaises(ValueError):
      _validate_round_decision_outcome("Default_Upstream", "default_upstream")


class TestScaffoldDecisionOutcome(unittest.TestCase):
  """③/⑤: the resolution-report write gets a SECOND file_check asserting the
  decision_outcome, alongside the decisive value -- so the round's decision (not
  just its value) is graded by construction against the gold's own write. Both a
  verdict phrase and a prose value are graded SEMANTICALLY (never verbatim: no
  independent solver reproduces a decision/finding word-for-word). Seeded only
  where a decisive value is also written."""

  ALLOWED = ["file", "asset", "message", "agent"]

  def _write_scaffold(
      self,
      path="reports/resolution.md",
      resolved="default_upstream",
      outcome="restore default upstream routing",
  ):
    scaffolds, _ = _scaffold_round_checkers(
        [{"step_number": 1, "command": f"file write --path {path} --content x"}],
        self.ALLOWED, asset_ids=[], resolved_value=resolved,
        resolved_value_locus=None, decision_outcome=outcome)
    return scaffolds[0]

  def test_report_write_emits_decision_outcome_extra_check(self):
    s = self._write_scaffold()
    # Primary check grades the decisive value semantically (prose value, embedded
    # in the criterion); no semantic-fill placeholder is needed.
    self.assertEqual(s["checker"]["type"], "effect")
    self.assertEqual(s["checker"]["effect_type"], "files_created")
    self.assertEqual(s["checker"]["condition"], "field_semantic_match")
    self.assertIn("default_upstream", s["checker"]["expected"]["criterion"])
    self.assertIsNone(s["semantic"])
    # Extra check grades the decision_outcome SEMANTICALLY via files_created.
    self.assertEqual(len(s["extra_checkers"]), 1)
    extra = s["extra_checkers"][0]
    self.assertEqual(extra["type"], "effect")
    self.assertEqual(extra["effect_type"], "files_created")
    self.assertEqual(extra["condition"], "field_semantic_match")
    self.assertIn("restore default upstream routing",
                  extra["expected"]["criterion"])

  def test_no_decision_outcome_means_no_extra_check(self):
    self.assertIsNone(self._write_scaffold(outcome="")["extra_checkers"])

  def test_no_resolved_value_means_no_extra_check(self):
    # Gated on BOTH a decisive value and an outcome (a real report has both).
    self.assertIsNone(
        self._write_scaffold(resolved="")["extra_checkers"])

  def test_code_file_write_has_no_extra_check(self):
    # The decisive value/verdict belong in a data/report file, never in code.
    s = self._write_scaffold(path="src/app.py")
    self.assertEqual(s["checker"]["type"], "command_check")
    self.assertIsNone(s["extra_checkers"])

  def test_steps_from_scaffolds_preserves_extra_checkers(self):
    s = self._write_scaffold()
    steps = _steps_from_scaffolds([s])["steps"]
    self.assertEqual(steps[0]["extra_checkers"], s["extra_checkers"])


class TestDecisionOutcomeMetadataFlow(unittest.TestCase):
  """③: decision_outcome + resolved_value flow end-to-end from the resolution
  layer through the materializer onto the shipped round spec and the compiled
  ScenarioRound metadata."""

  def _seed(self):
    return ConflictVLMSeed(
        slug="test_decision_outcome_flow_01",
        domain="system",
        title="Decision Outcome Flow",
        target_rounds=1,
        core_conflict="Time delay breaks assumptions.",
        conflict_taxonomy=["temporal_staleness", "subagent_delay"],
        participants=["owner", "assistant"],
        required_constraints=[],
        allowed_tool_families=["agents", "file", "message"],
        min_round_steps=4,
        require_final_verification=False,
    )

  def test_decision_outcome_on_round_spec_and_metadata(self):
    with tempfile.TemporaryDirectory() as tmp:
      conflict_spec, openclaw_spec = materialize_conflict_scenario(
          self._seed(), MockConflictLLMClient(), output_dir=Path(tmp),
          image_provider="mock", chart_provider="mock",
      )
      # Round spec carries the graded conclusion.
      self.assertEqual(len(conflict_spec.rounds), 1)
      rs = conflict_spec.rounds[0]
      self.assertEqual(rs.decision_outcome, "restore default upstream routing")
      self.assertEqual(rs.resolved_value, "default_upstream")
      # Compiled ScenarioRound metadata mirrors it for inspection/coverage.
      md = openclaw_spec.rounds[0].metadata
      self.assertEqual(md["decision_outcome"], "restore default upstream routing")
      self.assertEqual(md["resolved_value"], "default_upstream")


class TestDecisionOutcomeCheckerWiring(unittest.TestCase):
  """③/⑤: the decision_outcome extra file_check is flattened -- alongside the
  decisive-value check -- into the round eval's ``checks`` by the pipeline
  assembly, so the report-write scaffold's SECOND deterministic checker actually
  reaches the shipped eval (not just the scaffold). Drives generate() directly so
  the assembly link is proven without any materializer replay. The base mock's
  resolution round only READS, so a subclass steers the gold to WRITE the report,
  exercising the file_write branch that seeds the decision_outcome check."""

  REPORT_PATH = "reports/resolution.md"

  def _seed(self):
    return ConflictVLMSeed(
        slug="test_decision_outcome_wiring_01",
        domain="system",
        title="Decision Outcome Wiring",
        target_rounds=1,
        core_conflict="Time delay breaks assumptions.",
        conflict_taxonomy=["temporal_staleness", "subagent_delay"],
        participants=["owner", "assistant"],
        required_constraints=[],
        allowed_tool_families=["agents", "file", "message"],
        min_round_steps=4,
        require_final_verification=False,
    )

  def test_decision_outcome_check_flattened_into_eval(self):
    report_path = self.REPORT_PATH

    class _ReportWriteMockClient(MockConflictLLMClient):
      """Steer the resolution round's gold to WRITE a resolution report (the base
      mock only reads config), exercising the report-write scaffold branch and its
      decision_outcome extra checker end-to-end through generate()."""

      def generate_layer(self, layer, payload):
        result = super().generate_layer(layer, payload)
        if layer == "conflict_layer3_round_commands":
          cmds = [c["command"] for c in result.get("commands", [])]
          write = (
              f'file write --path {report_path} --content '
              '"Resolved value: default_upstream. '
              'Decision: restore default upstream routing."'
          )
          # Insert the report write just before the trailing DONE.
          insert_at = len(cmds)
          for i, c in enumerate(cmds):
            if c.strip().upper() == "DONE":
              insert_at = i
              break
          cmds.insert(insert_at, write)
          return {
              "commands": [
                  {"step_number": i + 1, "command": c}
                  for i, c in enumerate(cmds)
              ]
          }
        return result

    pipeline = ConflictLLMPipeline(
        client=_ReportWriteMockClient(), progress=lambda msg: None)
    res = pipeline.generate(self._seed())

    all_checks = [c for e in res["evals"] for c in e.get("checks", [])]
    # Both the value and the verdict are prose, so each is a path-agnostic
    # files_created SEMANTIC check whose criterion embeds the value/verdict.
    outcome_checks = [
        c for c in all_checks
        if c.get("type") == "effect"
        and c.get("effect_type") == "files_created"
        and c.get("condition") == "field_semantic_match"
        and "restore default upstream routing"
        in (c.get("expected", {}).get("criterion") or "")
    ]
    value_checks = [
        c for c in all_checks
        if c.get("type") == "effect"
        and c.get("effect_type") == "files_created"
        and c.get("condition") == "field_semantic_match"
        and "default_upstream" in (c.get("expected", {}).get("criterion") or "")
    ]
    # Both the decisive-value check and the decision_outcome check land in the
    # eval as files_created deliverables -- the round's verdict is graded together
    # with its evidence, by construction, without pinning a path.
    self.assertEqual(len(outcome_checks), 1, all_checks)
    self.assertTrue(value_checks, all_checks)


class TestCalibrateChecksAgainstState(unittest.TestCase):
  """C2: replay the gold trajectory and reconcile file_checks it cannot satisfy
  against the resolved final state -- degrade a failing content assertion to
  existence when the file exists, drop it when the file is gone, and leave passing
  or non-file checks untouched. Verified with the real FileChecker semantics."""

  def _round(self, checks):
    from types import SimpleNamespace
    return SimpleNamespace(
        task_id="r01",
        expert_trajectory=["file read --path reports/present.md", "DONE"],
        task_spec={"ground_truth": {"evaluation_checks": checks},
                   "id": "r01", "instruction": "x"},
    )

  def test_degrade_drop_and_keep(self):
    from types import SimpleNamespace
    checks = [
        {"type": "file_check", "path": "reports/present.md", "contains": "XYZ",
         "name": "content_a", "weight": 2},          # fails, file exists -> degrade
        {"type": "file_check", "path": "reports/missing.md", "contains": "ABC"},  # fails, gone -> drop
        {"type": "file_check", "path": "reports/present.md"},                     # passes -> keep
        {"type": "effect", "effect_type": "messages_sent"},                      # non-file -> keep
    ]
    round_spec = self._round(checks)
    spec = SimpleNamespace(rounds=[round_spec])
    env_state = {"files": {"/reports/present.md": "hello world"}}
    details = [SimpleNamespace(passed=False), SimpleNamespace(passed=False),
               SimpleNamespace(passed=True), SimpleNamespace(passed=False)]
    fake_result = SimpleNamespace(details=details)

    logs: list[str] = []
    with patch(
        "clasharness.scenario.conflict_vlm_materializer._replay_trajectory",
        return_value=(fake_result, "", env_state),
    ), patch(
        "clasharness.scenario.conflict_vlm_materializer.save_task"
    ) as save_mock, patch(
        "clasharness.scenario.conflict_vlm_materializer.Task"
    ):
      adjusted = _calibrate_checks_against_state(
          spec, Path("/tmp/does-not-matter"), logs.append)

    self.assertEqual(adjusted, 2)  # one degrade + one drop
    new_checks = round_spec.task_spec["ground_truth"]["evaluation_checks"]
    self.assertEqual(len(new_checks), 3)  # degraded, kept-pass, kept-effect
    degraded = new_checks[0]
    self.assertEqual(degraded["type"], "file_check")
    self.assertEqual(degraded["path"], "reports/present.md")
    self.assertNotIn("contains", degraded)          # content assertion stripped
    self.assertEqual(degraded["name"], "content_a")  # identity preserved
    self.assertEqual(degraded["weight"], 2)
    self.assertEqual(new_checks[1], {"type": "file_check", "path": "reports/present.md"})
    self.assertEqual(new_checks[2], {"type": "effect", "effect_type": "messages_sent"})
    save_mock.assert_called_once()  # re-saved because a round was adjusted

  def test_no_adjustment_leaves_checks_and_skips_save(self):
    from types import SimpleNamespace
    checks = [{"type": "file_check", "path": "reports/present.md", "contains": "ok"}]
    round_spec = self._round(checks)
    spec = SimpleNamespace(rounds=[round_spec])
    env_state = {"files": {"/reports/present.md": "ok"}}
    fake_result = SimpleNamespace(details=[SimpleNamespace(passed=True)])
    with patch(
        "clasharness.scenario.conflict_vlm_materializer._replay_trajectory",
        return_value=(fake_result, "", env_state),
    ), patch(
        "clasharness.scenario.conflict_vlm_materializer.save_task"
    ) as save_mock, patch(
        "clasharness.scenario.conflict_vlm_materializer.Task"
    ):
      adjusted = _calibrate_checks_against_state(spec, Path("/tmp/x"), lambda _m: None)
    self.assertEqual(adjusted, 0)
    self.assertEqual(round_spec.task_spec["ground_truth"]["evaluation_checks"], checks)
    save_mock.assert_not_called()


class TestIsNegativeAssertion(unittest.TestCase):
  """C3 predicate: a check asserting the ABSENCE of something is SUPPOSED to hold
  at zero actions, so it must be exempt from vacuity demotion."""

  def test_negative_conditions_are_exempt(self):
    for cond in ("not_exists", "not_contains", "not_something_future"):
      self.assertTrue(
          _is_negative_assertion({"condition": cond}), cond)

  def test_positive_conditions_are_not_exempt(self):
    for cond in ("exists", "contains", "equals", "field_contains",
                 "field_semantic_match", "within_window", "exit_code_zero"):
      self.assertFalse(
          _is_negative_assertion({"condition": cond}), cond)

  def test_missing_or_malformed_condition_is_not_exempt(self):
    self.assertFalse(_is_negative_assertion({}))
    self.assertFalse(_is_negative_assertion({"condition": None}))
    self.assertFalse(_is_negative_assertion("not_a_dict"))
    self.assertFalse(_is_negative_assertion(None))


class _FakeZeroActionEnv:
  """Records exactly what the zero-action probe does to an env."""

  instances: list["_FakeZeroActionEnv"] = []

  def __init__(self, **kwargs):
    self.kwargs = kwargs
    self.reset_calls = 0
    self.steps: list[str] = []
    self.evaluate_calls = 0
    self.closed = False
    _FakeZeroActionEnv.instances.append(self)

  def __enter__(self):
    return self

  def __exit__(self, *_a):
    self.closed = True
    return False

  def reset(self):
    self.reset_calls += 1

  def step(self, cmd):
    self.steps.append(cmd)
    return ("", 0.0, False, {})

  def evaluate(self):
    self.evaluate_calls += 1
    from types import SimpleNamespace
    return SimpleNamespace(details=[], success=False)


class TestReplayZeroAction(unittest.TestCase):
  """C3 probe: reset + evaluate with NO agent actions, and -- critically -- WITHOUT
  the always-pass semantic stub that ``_replay_trajectory`` installs. Reusing that
  helper would report every ``field_semantic_match`` check as already satisfied at
  t=0, i.e. the exact vacuity signal C3 acts on, neutralising the whole semantic
  population."""

  def setUp(self):
    _FakeZeroActionEnv.instances = []

  def test_resets_evaluates_and_never_steps(self):
    with patch("clasharness.core.environment.CLASHarnessEnv", _FakeZeroActionEnv):
      res = _replay_zero_action("r01", Path("/tmp/x"))
    self.assertEqual(len(_FakeZeroActionEnv.instances), 1)
    env = _FakeZeroActionEnv.instances[0]
    self.assertEqual(env.reset_calls, 1)
    self.assertEqual(env.steps, [])          # ZERO agent actions
    self.assertEqual(env.evaluate_calls, 1)
    self.assertTrue(env.closed)              # context manager exited
    self.assertEqual(res.details, [])

  def test_does_not_install_the_always_pass_semantic_stub(self):
    with patch("clasharness.core.environment.CLASHarnessEnv", _FakeZeroActionEnv), \
        patch(
            "clasharness.scenario.conflict_vlm_materializer._stub_semantic_file_checks"
        ) as stub_mock:
      _replay_zero_action("r01", Path("/tmp/x"))
    stub_mock.assert_not_called()

  def test_passes_task_id_and_data_dir_through(self):
    with patch("clasharness.core.environment.CLASHarnessEnv", _FakeZeroActionEnv):
      _replay_zero_action("round-7", Path("/tmp/data"))
    kw = _FakeZeroActionEnv.instances[0].kwargs
    self.assertEqual(kw["task_id"], "round-7")
    self.assertEqual(kw["task_data_dir"], Path("/tmp/data"))


class TestDemoteVacuousChecks(unittest.TestCase):
  """C3: a check that passes with zero agent actions measures nothing -- every
  model scores it -- so it is demoted to ``diagnostic`` (kept for observability,
  no longer gating success nor scoring). Conservative: only genuine verdicts on
  positive, currently-deciding checks are acted on, and a round never has its
  entire ``required`` set demoted."""

  def _d(self, passed, ok=True):
    from types import SimpleNamespace
    return SimpleNamespace(passed=passed, ok=ok)

  def _round(self, checks, task_id="r01"):
    from types import SimpleNamespace
    return SimpleNamespace(
        task_id=task_id,
        task_spec={"ground_truth": {"evaluation_checks": checks},
                   "id": task_id, "instruction": "x"},
    )

  def _run(self, round_specs, details_by_task, logs=None):
    from types import SimpleNamespace
    spec = SimpleNamespace(rounds=list(round_specs))
    logs = logs if logs is not None else []

    def fake_probe(task_id, _out):
      d = details_by_task[task_id]
      if isinstance(d, Exception):
        raise d
      return SimpleNamespace(details=d)

    with patch(
        "clasharness.scenario.conflict_vlm_materializer._replay_zero_action",
        side_effect=fake_probe,
    ), patch(
        "clasharness.scenario.conflict_vlm_materializer.save_task"
    ) as save_mock, patch(
        "clasharness.scenario.conflict_vlm_materializer.Task"
    ):
      out = _demote_vacuous_checks(spec, Path("/tmp/x"), logs.append)
    return out, save_mock, logs

  def test_vacuous_required_check_is_demoted_not_dropped(self):
    checks = [
        {"type": "effect", "name": "vacuous", "condition": "exists"},
        {"type": "effect", "name": "real", "condition": "field_contains"},
    ]
    rs = self._round(checks)
    (demoted, guarded), save_mock, _ = self._run(
        [rs], {"r01": [self._d(True), self._d(False)]})

    self.assertEqual((demoted, guarded), (1, 0))
    new = rs.task_spec["ground_truth"]["evaluation_checks"]
    self.assertEqual(len(new), 2)                  # demoted, NOT dropped
    self.assertEqual(new[0]["role"], "diagnostic")
    self.assertEqual(new[0]["name"], "vacuous")    # identity preserved
    self.assertEqual(new[0]["condition"], "exists")
    self.assertNotIn("role", new[1])               # untouched check left alone
    save_mock.assert_called_once()

  def test_demotion_log_identifies_the_check_by_path_when_unnamed(self):
    """The 4 real-corpus offenders carry neither ``name`` nor ``condition``;
    logging a bare 'file_check' would make the demotion unauditable, which defeats
    the point of demoting (rather than dropping) for traceability."""
    checks = [{"type": "file_check", "path": "workspace/logs/pipeline_build.log",
               "contains": "f8c22d1"},
              {"type": "effect", "name": "real", "condition": "field_contains"}]
    rs = self._round(checks)
    (demoted, _g), _sm, logs = self._run(
        [rs], {"r01": [self._d(True), self._d(False)]})
    self.assertEqual(demoted, 1)
    self.assertTrue(
        any("workspace/logs/pipeline_build.log" in m for m in logs), logs)

  def test_original_check_dict_is_not_mutated_in_place(self):
    original = {"type": "effect", "name": "vacuous", "condition": "exists"}
    rs = self._round([original])
    # 2 required so the invariant guard does not trip; only index 0 is vacuous.
    rs.task_spec["ground_truth"]["evaluation_checks"].append(
        {"type": "effect", "name": "real", "condition": "field_contains"})
    self._run([rs], {"r01": [self._d(True), self._d(False)]})
    self.assertNotIn("role", original)  # caller's dict untouched

  def test_ungradable_check_is_left_alone(self):
    """``ok=False`` means the grader never reached a verdict: the zero-action
    result is UNKNOWN, not clean. Demoting it would silently disarm a check on
    infrastructure failure."""
    checks = [{"type": "effect", "name": "a", "condition": "field_semantic_match"},
              {"type": "effect", "name": "b", "condition": "field_contains"}]
    rs = self._round(checks)
    (demoted, guarded), save_mock, _ = self._run(
        [rs], {"r01": [self._d(True, ok=False), self._d(False)]})
    self.assertEqual((demoted, guarded), (0, 0))
    self.assertNotIn("role", rs.task_spec["ground_truth"]["evaluation_checks"][0])
    save_mock.assert_not_called()

  def test_failing_check_is_left_alone(self):
    checks = [{"type": "effect", "name": "a", "condition": "field_contains"}]
    rs = self._round(checks)
    (demoted, _g), save_mock, _ = self._run([rs], {"r01": [self._d(False)]})
    self.assertEqual(demoted, 0)
    save_mock.assert_not_called()

  def test_passing_negative_assertion_is_exempt(self):
    """``not_exists`` is SUPPOSED to hold at t=0; its power is whether an acting
    agent breaks it."""
    checks = [{"type": "file_check", "name": "keep", "condition": "not_exists"},
              {"type": "effect", "name": "real", "condition": "field_contains"}]
    rs = self._round(checks)
    (demoted, guarded), save_mock, _ = self._run(
        [rs], {"r01": [self._d(True), self._d(False)]})
    self.assertEqual((demoted, guarded), (0, 0))
    self.assertNotIn("role", rs.task_spec["ground_truth"]["evaluation_checks"][0])
    save_mock.assert_not_called()

  def test_passing_diagnostic_check_is_left_alone(self):
    """A diagnostic check already gates nothing and scores nothing."""
    checks = [{"type": "effect", "name": "d", "role": "diagnostic",
               "condition": "exists"},
              {"type": "effect", "name": "real", "condition": "field_contains"}]
    rs = self._round(checks)
    (demoted, _g), save_mock, _ = self._run(
        [rs], {"r01": [self._d(True), self._d(False)]})
    self.assertEqual(demoted, 0)
    save_mock.assert_not_called()

  def test_passing_scored_check_is_demoted(self):
    """``scored`` does not gate success but DOES inflate the score."""
    checks = [{"type": "effect", "name": "s", "role": "scored",
               "condition": "exists"},
              {"type": "effect", "name": "real", "condition": "field_contains"}]
    rs = self._round(checks)
    (demoted, guarded), save_mock, _ = self._run(
        [rs], {"r01": [self._d(True), self._d(False)]})
    self.assertEqual((demoted, guarded), (1, 0))
    self.assertEqual(
        rs.task_spec["ground_truth"]["evaluation_checks"][0]["role"], "diagnostic")
    save_mock.assert_called_once()

  def test_all_required_vacuous_is_guarded_and_shipped_unchanged(self):
    """``success = all(required_passed)`` and ``all([]) is True``: emptying the
    required set turns the round into a guaranteed win -- strictly worse than the
    vacuity being repaired. Nothing is demoted; the case is logged loudly."""
    checks = [{"type": "effect", "name": "a", "condition": "exists"},
              {"type": "effect", "name": "b", "condition": "exists"}]
    rs = self._round(checks)
    (demoted, guarded), save_mock, logs = self._run(
        [rs], {"r01": [self._d(True), self._d(True)]})
    self.assertEqual((demoted, guarded), (0, 1))
    for c in rs.task_spec["ground_truth"]["evaluation_checks"]:
      self.assertNotIn("role", c)
    save_mock.assert_not_called()
    self.assertTrue(any("ALL 2 required" in m for m in logs), logs)

  def test_partial_required_demotion_is_allowed(self):
    """The guard trips only when the required set would be EMPTIED."""
    checks = [{"type": "effect", "name": "a", "condition": "exists"},
              {"type": "effect", "name": "b", "condition": "exists"}]
    rs = self._round(checks)
    (demoted, guarded), save_mock, _ = self._run(
        [rs], {"r01": [self._d(True), self._d(False)]})
    self.assertEqual((demoted, guarded), (1, 0))
    new = rs.task_spec["ground_truth"]["evaluation_checks"]
    self.assertEqual(new[0]["role"], "diagnostic")
    self.assertNotIn("role", new[1])
    save_mock.assert_called_once()

  def test_guard_ignores_scored_when_counting_required(self):
    """All ``scored`` checks vacuous is fine: they never gated success."""
    checks = [{"type": "effect", "name": "s1", "role": "scored", "condition": "exists"},
              {"type": "effect", "name": "s2", "role": "scored", "condition": "exists"},
              {"type": "effect", "name": "req", "condition": "field_contains"}]
    rs = self._round(checks)
    (demoted, guarded), save_mock, _ = self._run(
        [rs], {"r01": [self._d(True), self._d(True), self._d(False)]})
    self.assertEqual((demoted, guarded), (2, 0))
    save_mock.assert_called_once()

  def test_details_shorter_than_checks_leaves_the_tail_alone(self):
    """Positional alignment is the contract; an index with no verdict has no
    evidence of vacuity."""
    checks = [{"type": "effect", "name": "a", "condition": "exists"},
              {"type": "effect", "name": "b", "condition": "exists"},
              {"type": "effect", "name": "c", "condition": "exists"}]
    rs = self._round(checks)
    (demoted, guarded), _sm, _ = self._run(
        [rs], {"r01": [self._d(True), self._d(False)]})
    self.assertEqual((demoted, guarded), (1, 0))
    new = rs.task_spec["ground_truth"]["evaluation_checks"]
    self.assertEqual(new[0]["role"], "diagnostic")
    self.assertNotIn("role", new[2])

  def test_probe_error_skips_the_round_without_aborting_the_pass(self):
    bad = self._round([{"type": "effect", "name": "a", "condition": "exists"}], "bad")
    good = self._round(
        [{"type": "effect", "name": "a", "condition": "exists"},
         {"type": "effect", "name": "b", "condition": "field_contains"}], "good")
    (demoted, guarded), save_mock, logs = self._run(
        [bad, good],
        {"bad": RuntimeError("boom"), "good": [self._d(True), self._d(False)]})
    self.assertEqual((demoted, guarded), (1, 0))
    self.assertNotIn("role", bad.task_spec["ground_truth"]["evaluation_checks"][0])
    self.assertEqual(
        good.task_spec["ground_truth"]["evaluation_checks"][0]["role"], "diagnostic")
    save_mock.assert_called_once()
    self.assertTrue(any("boom" in m for m in logs), logs)

  def test_malformed_rounds_are_skipped(self):
    from types import SimpleNamespace
    rounds = [
        SimpleNamespace(task_id="a", task_spec=None),
        SimpleNamespace(task_id="b", task_spec={"ground_truth": None}),
        SimpleNamespace(task_id="c", task_spec={"ground_truth": {}}),
        SimpleNamespace(task_id="d",
                        task_spec={"ground_truth": {"evaluation_checks": []}}),
    ]
    (demoted, guarded), save_mock, _ = self._run(rounds, {})
    self.assertEqual((demoted, guarded), (0, 0))
    save_mock.assert_not_called()

  def test_judge_env_is_cleared_during_probe_and_restored_after(self):
    """A semantic check would otherwise bill k model calls per task just to grade
    an empty transcript."""
    from types import SimpleNamespace
    seen: dict[str, object] = {}
    sentinel = "sentinel-value-do-not-lose"
    key = _ZERO_ACTION_JUDGE_ENV[0]
    prior = os.environ.get(key)
    os.environ[key] = sentinel
    try:
      rs = self._round([{"type": "effect", "name": "a", "condition": "exists"}])
      spec = SimpleNamespace(rounds=[rs])

      def probe(_tid, _out):
        seen["inside"] = {k: os.environ.get(k) for k in _ZERO_ACTION_JUDGE_ENV}
        return SimpleNamespace(details=[self._d(False)])

      with patch(
          "clasharness.scenario.conflict_vlm_materializer._replay_zero_action",
          side_effect=probe,
      ), patch(
          "clasharness.scenario.conflict_vlm_materializer.save_task"
      ), patch("clasharness.scenario.conflict_vlm_materializer.Task"):
        _demote_vacuous_checks(spec, Path("/tmp/x"), lambda _m: None)

      self.assertEqual(
          seen["inside"], {k: None for k in _ZERO_ACTION_JUDGE_ENV})
      self.assertEqual(os.environ.get(key), sentinel)  # restored
    finally:
      if prior is None:
        os.environ.pop(key, None)
      else:
        os.environ[key] = prior

  def test_judge_env_restored_even_when_the_pass_raises(self):
    from types import SimpleNamespace
    key = _ZERO_ACTION_JUDGE_ENV[0]
    prior = os.environ.get(key)
    os.environ[key] = "keepme"
    try:
      rs = self._round([{"type": "effect", "name": "a", "condition": "exists"}])
      spec = SimpleNamespace(rounds=[rs])

      def boom(_m):
        raise RuntimeError("logger exploded")

      with patch(
          "clasharness.scenario.conflict_vlm_materializer._replay_zero_action",
          side_effect=RuntimeError("probe down"),
      ):
        with self.assertRaises(RuntimeError):
          _demote_vacuous_checks(spec, Path("/tmp/x"), boom)
      self.assertEqual(os.environ.get(key), "keepme")
    finally:
      if prior is None:
        os.environ.pop(key, None)
      else:
        os.environ[key] = prior

  def test_resave_failure_is_logged_without_aborting(self):
    from types import SimpleNamespace
    checks = [{"type": "effect", "name": "a", "condition": "exists"},
              {"type": "effect", "name": "b", "condition": "field_contains"}]
    rs = self._round(checks)
    spec = SimpleNamespace(rounds=[rs])
    logs: list[str] = []
    with patch(
        "clasharness.scenario.conflict_vlm_materializer._replay_zero_action",
        return_value=SimpleNamespace(details=[self._d(True), self._d(False)]),
    ), patch(
        "clasharness.scenario.conflict_vlm_materializer.save_task",
        side_effect=OSError("disk full"),
    ), patch("clasharness.scenario.conflict_vlm_materializer.Task"):
      demoted, guarded = _demote_vacuous_checks(spec, Path("/tmp/x"), logs.append)
    self.assertEqual((demoted, guarded), (1, 0))
    self.assertTrue(any("disk full" in m for m in logs), logs)



class TestStubSemanticFileChecks(unittest.TestCase):
  """Step 4: compile-time replay must NEVER invoke the run-time LLM judge. Semantic
  (``field_semantic_match``) file checks are swapped for always-pass stubs that
  keep positional alignment, so the C2/F1 passes neither judge them nor act on an
  assertion graded only at run time."""

  def test_is_semantic_check_predicate(self):
    self.assertTrue(_is_semantic_check(
        {"type": "file_check", "condition": "field_semantic_match"}))
    self.assertTrue(_is_semantic_check(
        {"type": "file", "condition": "field_semantic_match"}))
    # Broadened (Edit 4): effect-stream deliverables graded by the same run-time
    # judge (files_created report / messages_sent delegation) are semantic too.
    self.assertTrue(_is_semantic_check(
        {"type": "effect", "condition": "field_semantic_match"}))
    self.assertFalse(_is_semantic_check(
        {"type": "file_check", "contains": "x"}))
    self.assertFalse(_is_semantic_check("nope"))

  def test_replay_safe_stub_always_passes_without_judge(self):
    from clasharness.evaluation import llm_judge
    with patch.object(llm_judge, "judge_majority",
                      side_effect=AssertionError("judge must not run")):
      res = _ReplaySafeStub("nm", 3.0).evaluate({"files": {}}, {})
    self.assertTrue(res.passed)
    self.assertEqual(res.score, 1.0)
    self.assertEqual(res.weight, 3.0)
    self.assertEqual(res.name, "nm")

  def test_stub_swaps_only_semantic_file_checks(self):
    from types import SimpleNamespace
    from clasharness.evaluation.evaluator import build_evaluator
    from clasharness.evaluation.checkers.file_checker import FileChecker
    checks = [
        {"type": "file_check", "path": "/r.md", "contains": "42"},
        {"type": "file_check", "path": "/r.md",
         "condition": "field_semantic_match", "criterion": "states the verdict",
         "name": "verdict", "weight": 2},
        {"type": "command_check", "command": "file read --path /r.md"},
    ]
    evaluator = build_evaluator(checks)
    env = SimpleNamespace(
        _evaluator=evaluator,
        _task=SimpleNamespace(
            ground_truth=SimpleNamespace(evaluation_checks=checks)))
    self.assertIsInstance(evaluator.evaluators[1], FileChecker)
    _stub_semantic_file_checks(env)
    # Only the semantic file check (position 1) is swapped; identity preserved.
    self.assertIsInstance(evaluator.evaluators[0], FileChecker)
    self.assertIsInstance(evaluator.evaluators[1], _ReplaySafeStub)
    self.assertEqual(evaluator.evaluators[1].name, "verdict")
    self.assertEqual(evaluator.evaluators[1].weight, 2)
    self.assertNotIsInstance(evaluator.evaluators[2], _ReplaySafeStub)

  def test_stubbed_evaluator_grades_without_touching_judge(self):
    from types import SimpleNamespace
    from clasharness.evaluation.evaluator import build_evaluator
    from clasharness.evaluation import llm_judge
    checks = [{
        "type": "file_check", "path": "/r.md",
        "condition": "field_semantic_match", "criterion": "states the verdict",
    }]
    evaluator = build_evaluator(checks)
    env = SimpleNamespace(
        _evaluator=evaluator,
        _task=SimpleNamespace(
            ground_truth=SimpleNamespace(evaluation_checks=checks)))
    _stub_semantic_file_checks(env)
    with patch.object(
        llm_judge, "judge_majority",
        side_effect=AssertionError("judge must not run during replay")):
      result = evaluator({"files": {"/r.md": "we rolled back"}}, {})
    self.assertTrue(result.success)


class TestResolutionGroundingHelpers(unittest.TestCase):
  """P1a: helpers that surface real asset/file content to the resolver so it can
  PICK a genuinely present decisive value instead of inventing one."""

  def test_fold_matches_filechecker_semantics(self):
    # Case-insensitive + whitespace-run collapse, mirroring file_checker._fold.
    self.assertEqual(_fold_for_grounding("  Foo   Bar\tBaz\n"), "foo bar baz")
    self.assertEqual(_fold_for_grounding(None), "")

  def test_asset_blob_combines_caption_and_ocr(self):
    blob = _asset_grounding_blob(
        {"id": "c1", "caption": "Rack C2 outage", "ocr_text": "cpu 92%"})
    self.assertIn("Rack C2 outage", blob)
    self.assertIn("cpu 92%", blob)

  def test_asset_blob_json_encodes_dict_caption(self):
    blob = _asset_grounding_blob({"id": "c1", "caption": {"title": "Rack C2"}})
    # dict caption is json-serialized so its values remain searchable.
    self.assertIn("Rack C2", blob)

  def test_collect_workspace_texts_base_and_updates_normalized(self):
    layer1b = {"workspace": {"files": [{"path": "/base/index.md", "content": "base"}]}}
    narrative = {"updates": [
        {"workspace_adds": [{"path": "config/edge_proxy.yaml", "content": "redirect_target: default_upstream"}]},
        {"workspace_adds": [{"path": "/reports/out.md", "content": "total 42"}]},
    ]}
    texts = _collect_round_workspace_texts(layer1b, narrative)
    # Leading slashes stripped so paths match command/check paths.
    self.assertEqual(texts["base/index.md"], "base")
    self.assertEqual(texts["config/edge_proxy.yaml"], "redirect_target: default_upstream")
    self.assertEqual(texts["reports/out.md"], "total 42")

  def test_collect_workspace_texts_accepts_list_workspace(self):
    layer1b = {"workspace": [{"path": "a.md", "content": "x"}]}
    texts = _collect_round_workspace_texts(layer1b, {"updates": []})
    self.assertEqual(texts["a.md"], "x")

  def test_build_grounding_text_includes_content_and_caps(self):
    assets = [{"id": "chart_01", "caption": "Rack C2", "ocr_text": "Z" * 900}]
    texts = {"reports/out.md": "Q" * 900}
    out = _build_resolution_grounding_text(assets, texts)
    self.assertIn("[asset:chart_01]", out)
    self.assertIn("Rack C2", out)
    self.assertIn("[file:reports/out.md]", out)
    # Capped snippets keep the prompt bounded.
    self.assertLess(out.count("Z"), 900)
    self.assertLess(out.count("Q"), 900)

  def test_build_grounding_text_placeholder_when_empty(self):
    out = _build_resolution_grounding_text([], {})
    self.assertIn("no readable asset text or file content", out)

  def test_build_grounding_text_dedupes_assets(self):
    assets = [{"id": "c1", "caption": "one"}, {"id": "c1", "caption": "dup"}]
    out = _build_resolution_grounding_text(assets, {})
    self.assertEqual(out.count("[asset:c1]"), 1)


class TestValidateRoundResolution(unittest.TestCase):
  """P1b: in-loop grounding gate. Rejects (raises, feeding the retry loop) an
  empty value, a bogus/mismatched locus, or an unlabeled ungrounded value;
  ``computed`` is lenient (a genuinely derived value need not be verbatim)."""

  ASSETS = [{"id": "chart_01", "caption": "Rack C2 outage", "ocr_text": "cpu 92% degraded"}]
  FILES = {"config/edge_proxy.yaml": "redirect_target: default_upstream"}

  def test_empty_value_raises(self):
    for v in ("", "   "):
      with self.assertRaises(ValueError):
        _validate_round_resolution(v, "asset:chart_01", self.ASSETS, self.FILES)

  def test_asset_locus_present_ok(self):
    _validate_round_resolution("cpu 92%", "asset:chart_01", self.ASSETS, self.FILES)

  def test_asset_locus_unknown_asset_raises(self):
    with self.assertRaises(ValueError) as cm:
      _validate_round_resolution("cpu 92%", "asset:missing", self.ASSETS, self.FILES)
    self.assertIn("visible asset", str(cm.exception))

  def test_asset_locus_value_absent_raises(self):
    with self.assertRaises(ValueError):
      _validate_round_resolution("nonexistent", "asset:chart_01", self.ASSETS, self.FILES)

  def test_file_locus_present_ok(self):
    _validate_round_resolution("default_upstream", "file:config/edge_proxy.yaml", self.ASSETS, self.FILES)

  def test_file_locus_unknown_path_raises(self):
    with self.assertRaises(ValueError) as cm:
      _validate_round_resolution("default_upstream", "file:nope.md", self.ASSETS, self.FILES)
    self.assertIn("readable workspace", str(cm.exception))

  def test_file_locus_value_absent_raises(self):
    with self.assertRaises(ValueError):
      _validate_round_resolution("999", "file:config/edge_proxy.yaml", self.ASSETS, self.FILES)

  def test_fold_insensitive_match(self):
    # Different case + whitespace still grounds, matching FileChecker semantics.
    files = {"r.md": "Redirect_Target:    Default_Upstream"}
    _validate_round_resolution("default_upstream", "file:r.md", [], files)

  def test_computed_grounded_is_clean_without_warn(self):
    logs: list[str] = []
    _validate_round_resolution("default_upstream", "computed", self.ASSETS, self.FILES, logger=logs.append)
    self.assertEqual(logs, [])  # present verbatim -> no warning

  def test_computed_ungrounded_accepted_with_warn(self):
    logs: list[str] = []
    _validate_round_resolution("aggregated_total_xyz", "computed", self.ASSETS, self.FILES, logger=logs.append)
    self.assertTrue(any("computed" in m for m in logs))

  def test_computed_ungrounded_no_logger_does_not_crash(self):
    _validate_round_resolution("aggregated_total_xyz", "computed", self.ASSETS, self.FILES)

  def test_unknown_locus_grounded_in_file_ok(self):
    _validate_round_resolution("default_upstream", "", self.ASSETS, self.FILES)

  def test_unknown_locus_grounded_in_asset_ok(self):
    _validate_round_resolution("cpu 92%", "garbage", self.ASSETS, self.FILES)

  def test_unknown_locus_ungrounded_raises(self):
    with self.assertRaises(ValueError) as cm:
      _validate_round_resolution("ghost_value", "", self.ASSETS, self.FILES)
    self.assertIn("missing/unknown", str(cm.exception))


class TestValidateRoundConfidentiality(unittest.TestCase):
  """I1 (narrative confidentiality): the DUAL of the grounding gate. The decisive
  value must be absent from every agent-visible TEXT surface -- the round
  instruction, any workspace file that is NOT the declared locus, chat/session
  messages, and mock-subagent text -- so the round cannot be solved from a text
  side-channel (for an ``asset`` locus, without looking at the pixels). It
  deliberately does NOT scan other assets: a corroborating chart showing the
  same value is still pixels. Short values (<3 non-space chars) are skipped.

  EVERY locus is scanned. ``computed`` and unknown loci used to be exempt on
  the theory that an aggregated value legitimately appears in its sources; the
  exemption is deleted and the tests below are inverted. Measured: ``computed``
  appeared 0 times on the pre-binding corpus and 43 times on the next one, all
  43 emitted only after this gate had rejected the round for leaking, none a
  first choice -- it was the single unguarded channel, not a design choice. A
  ``file:`` locus is still exempt for its OWN declared path only."""

  FILES = {"config/edge_proxy.yaml": "redirect_target: default_upstream"}
  INSTR = "Reconcile the release board and record the correct rollback target."

  def test_clean_asset_locus_ok(self):
    # Value lives only in the asset (pixels): absent from instruction and files.
    _validate_round_confidentiality("cpu 92%", "asset:chart_01", self.INSTR, self.FILES)

  def test_value_in_instruction_raises(self):
    with self.assertRaises(ValueError) as cm:
      _validate_round_confidentiality(
          "default_upstream", "asset:chart_01",
          "The answer is default_upstream; just confirm it.", self.FILES)
    self.assertIn("instruction", str(cm.exception))

  def test_value_in_non_locus_file_raises(self):
    with self.assertRaises(ValueError) as cm:
      _validate_round_confidentiality(
          "default_upstream", "asset:chart_01", self.INSTR, self.FILES)
    self.assertIn("workspace file", str(cm.exception))

  def test_value_in_locus_file_is_allowed(self):
    # When the declared locus IS that file, the value's presence there is expected.
    _validate_round_confidentiality(
        "default_upstream", "file:config/edge_proxy.yaml", self.INSTR, self.FILES)

  def test_file_locus_but_leak_in_other_file_raises(self):
    files = dict(self.FILES)
    files["notes/leak.md"] = "the rollback target is default_upstream"
    with self.assertRaises(ValueError):
      _validate_round_confidentiality(
          "default_upstream", "file:config/edge_proxy.yaml", self.INSTR, files)

  def test_computed_is_scanned_like_every_other_locus(self):
    # INVERTED. This used to assert "computed values legitimately aggregate
    # visible sources -> files not scanned", which was a plausible story with
    # no measurement behind it. What the exemption actually did: ``computed``
    # went from 0 occurrences on the pre-binding corpus to 43 on the next one,
    # and 43 of 43 were emitted only AFTER this gate had already rejected the
    # round for leaking -- 0 of 43 were a first choice. It was not a locus the
    # resolver believed in, it was the one unguarded channel, and binding
    # pressure found it. Every one of those 43 rounds shipped text-solvable.
    with self.assertRaises(ValueError) as cm:
      _validate_round_confidentiality(
          "default_upstream", "computed", self.INSTR, self.FILES)
    self.assertIn("workspace file", str(cm.exception))

  def test_a_genuinely_derived_computed_value_still_passes(self):
    # The CONTROL for the inversion, and the reason removing the exemption is
    # not a difficulty tax: the scan only rejects a VERBATIM hit on the
    # computed RESULT. A value actually derived from visible sources is not
    # sitting written out in a file, so it passes untouched -- here the file
    # holds the inputs and the total appears nowhere.
    files = {"ledger/rows.csv": "alpha,120\nbravo,301\ncharlie,88"}
    _validate_round_confidentiality("509 units", "computed", self.INSTR, files)

  def test_computed_still_blocks_instruction_leak(self):
    with self.assertRaises(ValueError):
      _validate_round_confidentiality(
          "default_upstream", "computed", "the total is default_upstream", self.FILES)

  def test_short_value_skipped(self):
    # <3 non-space chars -> skipped to avoid matching ubiquitous substrings.
    _validate_round_confidentiality(
        "v2", "asset:chart_01", "uses v2 everywhere", {"a.md": "v2 v2 v2"})

  def test_fold_insensitive_detects_leak(self):
    files = {"r.md": "Redirect:   DEFAULT_UPSTREAM"}
    with self.assertRaises(ValueError):
      _validate_round_confidentiality("default_upstream", "asset:chart_01", self.INSTR, files)

  def test_clean_logs_confirmation(self):
    logs: list[str] = []
    _validate_round_confidentiality(
        "cpu 92%", "asset:chart_01", self.INSTR, self.FILES, logger=logs.append)
    self.assertTrue(any("confidentiality ok" in m for m in logs))

  # --- chat/session confinement (the observed difficulty-collapse leak) --------
  # ``CHAT_VALUE`` is a decisive value that lives ONLY in the asset pixels: it is
  # absent from ``INSTR`` and ``FILES``, so these tests isolate the chat scan
  # (a value also present in a file would trip the workspace scan first).
  CHAT_VALUE = "AUTH-882-X99"

  def test_value_in_chat_session_raises(self):
    # asset-locus value transcribed verbatim into agent-visible chat -> reject:
    # the round would be solvable from the session log without opening the asset.
    chat = [f"Avery: the passcode is {self.CHAT_VALUE}, go ahead"]
    with self.assertRaises(ValueError) as cm:
      _validate_round_confidentiality(
          self.CHAT_VALUE, "asset:chart_01", self.INSTR, self.FILES,
          session_texts=chat)
    self.assertIn("chat/session", str(cm.exception))

  def test_clean_chat_session_ok(self):
    # Chat may REFER to the evidence without transcribing the verbatim value.
    chat = ["Avery: check the figure on the dashboard before you proceed"]
    _validate_round_confidentiality(
        self.CHAT_VALUE, "asset:chart_01", self.INSTR, self.FILES,
        session_texts=chat)

  def test_file_locus_chat_leak_raises(self):
    # Chat is NEVER a declared locus, so even a file-locus round must not echo
    # the value into chat (it would be trivially text-solvable from the log).
    chat = [f"Blair: fyi the passcode is {self.CHAT_VALUE}"]
    with self.assertRaises(ValueError) as cm:
      _validate_round_confidentiality(
          self.CHAT_VALUE, "file:config/edge_proxy.yaml", self.INSTR,
          self.FILES, session_texts=chat)
    self.assertIn("chat/session", str(cm.exception))

  def test_computed_chat_leak_raises(self):
    # INVERTED alongside the file scan -- a mock or teammate reciting the
    # "computed total" verbatim hands the round over just as completely as
    # reciting a gauge reading does. The locus label does not change what the
    # player can read out of the log.
    chat = [f"Avery: the total came out to {self.CHAT_VALUE}"]
    with self.assertRaises(ValueError) as cm:
      _validate_round_confidentiality(
          self.CHAT_VALUE, "computed", self.INSTR, self.FILES,
          session_texts=chat)
    self.assertIn("chat/session", str(cm.exception))

  def test_an_unknown_locus_label_is_scanned_too(self):
    # The exemption keyed off "kind not in {asset, file}", so an empty or
    # garbage locus string was unguarded by the same branch. Closing it for
    # ``computed`` only would leave the identical escape one typo away.
    for locus in ("", "garbage", "derived", "aggregate"):
      with self.subTest(locus=locus):
        with self.assertRaises(ValueError):
          _validate_round_confidentiality(
              self.CHAT_VALUE, locus, self.INSTR, self.FILES,
              session_texts=[f"Blair: it is {self.CHAT_VALUE}"])

  def test_chat_scan_is_fold_insensitive(self):
    chat = [f"Avery:  {self.CHAT_VALUE.upper()}   is the one"]
    with self.assertRaises(ValueError) as cm:
      _validate_round_confidentiality(
          self.CHAT_VALUE.lower(), "asset:chart_01", self.INSTR, self.FILES,
          session_texts=chat)
    self.assertIn("chat/session", str(cm.exception))

  def test_no_session_texts_is_backward_compatible(self):
    # Omitting session_texts must behave exactly as before (no chat scan).
    _validate_round_confidentiality(
        "cpu 92%", "asset:chart_01", self.INSTR, self.FILES)


class TestCollectRoundSessionTexts(unittest.TestCase):
  """The chat/session collector feeding the confidentiality gate: it must gather
  BOTH the scenario-wide base sessions (layer 1b) and this round's async
  ``session_appends``, tolerating the loose key shapes the generator emits."""

  def test_collects_base_sessions_and_update_appends(self):
    l1b = {"sessions": [
        {"session_id": "s1", "messages": [
            {"sender": "a", "text": "base message one"},
            {"sender": "b", "content": "base message two"},
        ]},
    ]}
    narr = {"updates": [
        {"session_appends": [{"message": "async one"}]},
        {"session_appends": [{"text": "async two"}]},
    ]}
    texts = _collect_round_session_texts(l1b, narr)
    self.assertEqual(
        texts,
        ["base message one", "base message two", "async one", "async two"],
    )

  def test_tolerates_alternate_container_keys(self):
    # 'chat'/'dialogs' are accepted aliases for 'messages'.
    l1b = {"sessions": [{"chat": [{"body": "via chat key"}]},
                        {"dialogs": [{"text": "via dialogs key"}]}]}
    texts = _collect_round_session_texts(l1b, {"updates": []})
    self.assertEqual(texts, ["via chat key", "via dialogs key"])

  def test_empty_and_missing_are_safe(self):
    self.assertEqual(_collect_round_session_texts({}, {}), [])
    self.assertEqual(
        _collect_round_session_texts({"sessions": []}, {"updates": []}), [])


class TestAuditExtractorLiveness(unittest.TestCase):
  """Tripwire for a confidentiality collector that goes blind. An arm scanning
  an empty corpus reports every round clean -- a confident false negative,
  strictly worse than no gate, and exactly how the mock channel stayed unguarded
  through two 60-scenario corpora. Fires only when a surface HAS content and its
  collector returned nothing; a legitimately empty surface is silent."""

  L1B = {
      "workspace": {"files": [{"path": "docs/a.md", "content": "hello"}]},
      "sessions": [{"session_id": "s1", "messages": [{"text": "hi there"}]}],
  }
  NARR = {
      "updates": [{"workspace_adds": [{"path": "b.md", "content": "x"}],
                   "session_appends": [{"text": "async"}]}],
      "rounds": [{"mock_subagents": [
          {"agent_role": "reviewer", "mock_response": "the gate is green",
           "state_effects": ["message send --channel ops --content ok"]},
      ]}],
  }

  def _audit(self, *, l1b=None, narr=None, ws=None, se=None, mo=None,
             prior_updates=None, prior_rounds=None):
    l1b = self.L1B if l1b is None else l1b
    narr = self.NARR if narr is None else narr
    _audit_extractor_liveness(
        layer1b_payload=l1b, narrative_payload=narr,
        prior_updates=prior_updates, prior_rounds=prior_rounds,
        workspace_texts=(
            _collect_round_workspace_texts(l1b, narr, prior_updates)
            if ws is None else ws),
        session_texts=(
            _collect_round_session_texts(l1b, narr) if se is None else se),
        mock_texts=(
            _collect_round_mock_texts(narr, prior_rounds) if mo is None else mo),
        round_id="r03",
    )

  # -- the tripwire stays silent on healthy and on legitimately empty input ---

  def test_healthy_payload_with_real_collectors_passes(self):
    self._audit()

  def test_fully_empty_payload_passes(self):
    self._audit(l1b={}, narr={}, ws={}, se=[], mo=[])

  def test_scenario_without_sessions_or_mocks_passes(self):
    l1b = {"workspace": {"files": [{"path": "a.md", "content": "x"}]},
           "sessions": []}
    narr = {"updates": [], "rounds": [{"mock_subagents": []}]}
    self._audit(l1b=l1b, narr=narr)

  def test_session_declared_with_zero_messages_passes(self):
    # Has the key, holds nothing: nothing to extract, not a blind collector.
    l1b = {"workspace": {"files": []}, "sessions": [{"session_id": "s",
                                                     "messages": []}]}
    self._audit(l1b=l1b, narr={"updates": [], "rounds": []})

  def test_mock_with_only_unread_fields_passes(self):
    # private_knowledge/capabilities are deliberately NOT collected (stripped
    # before the task ships), so extracting nothing from them is correct.
    narr = {"updates": [], "rounds": [{"mock_subagents": [
        {"agent_role": "auditor", "private_knowledge": "the code is 4471",
         "capabilities": ["read"]}]}]}
    self._audit(narr=narr)

  # -- populated-but-empty: an unambiguous collector bug --------------------

  def test_workspace_declared_but_collector_returned_nothing(self):
    with self.assertRaises(ExtractorBlindError) as cm:
      self._audit(ws={})
    self.assertIn("workspace file(s) declared", str(cm.exception))

  def test_sessions_declared_but_collector_returned_nothing(self):
    with self.assertRaises(ExtractorBlindError) as cm:
      self._audit(se=[])
    self.assertIn("chat message(s) declared", str(cm.exception))

  def test_mocks_declared_but_collector_returned_nothing(self):
    with self.assertRaises(ExtractorBlindError) as cm:
      self._audit(mo=[])
    self.assertIn("mock text(s) declared", str(cm.exception))

  def test_prior_round_mocks_alone_are_enough_to_trip(self):
    narr = {"updates": [], "rounds": []}
    with self.assertRaises(ExtractorBlindError):
      self._audit(narr=narr, mo=[],
                  prior_rounds=[{"mock_subagents": [
                      {"mock_response": "leaked from r01"}]}])

  # -- schema drift: the container renamed its keys out from under us -------

  def test_workspace_dict_without_files_key_trips(self):
    l1b = {"workspace": {"documents": [{"path": "a.md", "content": "x"}]},
           "sessions": []}
    with self.assertRaises(ExtractorBlindError) as cm:
      self._audit(l1b=l1b, narr={"updates": [], "rounds": []})
    self.assertIn("no 'files' key", str(cm.exception))

  def test_session_without_any_known_message_key_trips(self):
    l1b = {"workspace": {"files": []},
           "sessions": [{"session_id": "s1", "turns": [{"text": "renamed"}]}]}
    with self.assertRaises(ExtractorBlindError) as cm:
      self._audit(l1b=l1b, narr={"updates": [], "rounds": []})
    self.assertIn("expose none of", str(cm.exception))

  def test_error_names_the_round(self):
    with self.assertRaises(ExtractorBlindError) as cm:
      self._audit(ws={})
    self.assertIn("round r03", str(cm.exception))


class TestValueLeaksInText(unittest.TestCase):
  """Strict-validator backstop scanner: fold-insensitive verbatim detection with
  token-boundary matching for simple tokens and substring matching for multi-word
  values; trivially short values are ignored."""

  def test_token_boundary_hit(self):
    self.assertTrue(_value_leaks_in_text("v4.2", "please roll back to v4.2 now"))

  def test_token_boundary_no_partial(self):
    # 'v4.2' must NOT match inside 'v4.25'.
    self.assertFalse(_value_leaks_in_text("v4.2", "deploying v4.25 tonight"))

  def test_case_and_space_insensitive(self):
    self.assertTrue(_value_leaks_in_text("SALT_2024", "uses   salt_2024 key"))

  def test_multiword_substring(self):
    self.assertTrue(
        _value_leaks_in_text("roll back to v4.2", "we should Roll Back To v4.2 asap"))

  def test_short_value_ignored(self):
    self.assertFalse(_value_leaks_in_text("42", "the answer is 42"))

  def test_absent_value(self):
    self.assertFalse(_value_leaks_in_text("default_upstream", "nothing to see here"))

  def test_empty_text(self):
    self.assertFalse(_value_leaks_in_text("default_upstream", ""))


class TestIsAssetLocus(unittest.TestCase):
  """Strict-validator locus discriminator: only an ``asset:`` locus is a vision
  round subject to the workspace/chat confinement scan; file/computed/unknown
  loci are legitimately text-grounded."""

  def test_asset_prefix_true(self):
    self.assertTrue(_is_asset_locus("asset:chart_01"))

  def test_asset_prefix_case_insensitive(self):
    self.assertTrue(_is_asset_locus("  ASSET:Chart_01 "))

  def test_file_locus_false(self):
    self.assertFalse(_is_asset_locus("file:config/edge_proxy.yaml"))

  def test_computed_false(self):
    self.assertFalse(_is_asset_locus("computed"))

  def test_empty_and_none_false(self):
    self.assertFalse(_is_asset_locus(""))
    self.assertFalse(_is_asset_locus(None))


class TestRoundSpecLocusField(unittest.TestCase):
  """``resolved_value_locus`` is threaded onto the shipped round spec so the
  post-generation confidentiality net can tell a vision round from a text one."""

  def test_from_dict_reads_locus(self):
    rs = ConflictVLMRoundSpec.from_dict(
        {"round_id": "r01", "resolved_value_locus": "asset:img_A"})
    self.assertEqual(rs.resolved_value_locus, "asset:img_A")

  def test_from_dict_defaults_empty(self):
    rs = ConflictVLMRoundSpec.from_dict({"round_id": "r01"})
    self.assertEqual(rs.resolved_value_locus, "")

  def test_from_dict_none_coerced_to_empty(self):
    rs = ConflictVLMRoundSpec.from_dict(
        {"round_id": "r01", "resolved_value_locus": None})
    self.assertEqual(rs.resolved_value_locus, "")


class _RetryResolutionMockClient(MockConflictLLMClient):
  """Stateful mock: the FIRST resolution call returns an UNGROUNDED value (no
  locus) so the in-loop gate must reject it; subsequent calls return the grounded
  value. Proves ungrounded resolutions are self-corrected at the source (retry)
  rather than leaking downstream."""

  def __init__(self):
    self.resolution_calls = 0

  def generate_layer(self, layer, payload):
    if layer == "conflict_layer3_round_resolution":
      self.resolution_calls += 1
      if self.resolution_calls == 1:
        return {"decisive_anchor": "EVIDENCE:RACK_C2",
                "resolved_value": "TOTALLY_UNGROUNDED_VALUE_XYZ",
                "decision_outcome": "restore default upstream routing"}
      return {"decisive_anchor": "EVIDENCE:RACK_C2",
              "resolved_value": "default_upstream",
              "decision_outcome": "restore default upstream routing"}
    return super().generate_layer(layer, payload)


class TestResolutionGroundingIntegration(unittest.TestCase):

  def _seed(self):
    return ConflictVLMSeed(
        slug="test_grounding_retry_01",
        domain="system",
        title="Grounding Retry",
        target_rounds=1,
        core_conflict="Time delay breaks assumptions.",
        conflict_taxonomy=["temporal_staleness", "subagent_delay"],
        participants=["owner", "assistant"],
        required_constraints=[],
        allowed_tool_families=["agents", "file", "message"],
        min_round_steps=4,
        require_final_verification=False,
    )

  def test_ungrounded_resolution_retries_then_succeeds(self):
    with tempfile.TemporaryDirectory() as tmp:
      client = _RetryResolutionMockClient()
      conflict_spec, _ = materialize_conflict_scenario(
          self._seed(), client, output_dir=Path(tmp),
          image_provider="mock", chart_provider="mock",
      )
      # The gate rejected the ungrounded first value and forced a regeneration.
      self.assertGreaterEqual(client.resolution_calls, 2)
      # Scenario still materializes and passes strict validation.
      validate_conflict_scenario_strict(conflict_spec)
      self.assertEqual(len(conflict_spec.rounds), 1)

  def test_grounded_resolution_is_single_pass(self):
    with tempfile.TemporaryDirectory() as tmp:
      client = MockConflictLLMClient()
      calls = {"n": 0}
      orig = client.generate_layer
      def counting(layer, payload):
        if layer == "conflict_layer3_round_resolution":
          calls["n"] += 1
        return orig(layer, payload)
      client.generate_layer = counting  # type: ignore[method-assign]
      conflict_spec, _ = materialize_conflict_scenario(
          self._seed(), client, output_dir=Path(tmp),
          image_provider="mock", chart_provider="mock",
      )
      # Grounded value (default_upstream lives in the update file) => no retry.
      self.assertEqual(calls["n"], 1)
      validate_conflict_scenario_strict(conflict_spec)


class TestConflictEvidenceSpec(unittest.TestCase):
  """Phase 0: optional, backward-compatible structured evidence-modality field.

  Lets a seed name the REAL-WORLD form of each artifact so the asset planner can
  choose an authentic kind (natural_image for photos, etc.) instead of flattening
  everything to charts/diagrams. Absent -> [] (behavior unchanged).
  """

  _BASE = {
      "slug": "evidence_probe",
      "domain": "warehouse robotics",
      "title": "Evidence Probe",
      "target_rounds": 2,
      "core_conflict": "c",
      "conflict_taxonomy": ["spatial"],
      "participants": ["Ops_Agent"],
      "required_constraints": [],
      "allowed_tool_families": ["file", "asset"],
  }

  def test_absent_evidence_defaults_to_empty_list(self):
    seed = ConflictVLMSeed.from_dict(dict(self._BASE))
    self.assertEqual(seed.evidence, [])

  def test_parses_items_and_defaults_missing_keys(self):
    data = dict(self._BASE)
    data["evidence"] = [
        {
            "id": "beacon_photo",
            "summary": "Cell 3 beacon lit red.",
            "real_world_source": "physical_photo",
            "role": "corroborating",
        },
        {"id": "lat_chart", "real_world_source": "dashboard_chart"},
    ]
    seed = ConflictVLMSeed.from_dict(data)
    self.assertEqual(len(seed.evidence), 2)
    first = seed.evidence[0]
    self.assertIsInstance(first, ConflictEvidenceSpec)
    self.assertEqual(first.id, "beacon_photo")
    self.assertEqual(first.summary, "Cell 3 beacon lit red.")
    self.assertEqual(first.real_world_source, "physical_photo")
    self.assertEqual(first.role, "corroborating")
    # Missing keys fall back to "" rather than raising.
    self.assertEqual(seed.evidence[1].summary, "")
    self.assertEqual(seed.evidence[1].role, "")

  def test_non_dict_evidence_items_are_skipped(self):
    data = dict(self._BASE)
    data["evidence"] = ["nope", 5, None, {"id": "ok"}]
    seed = ConflictVLMSeed.from_dict(data)
    self.assertEqual([e.id for e in seed.evidence], ["ok"])

  def test_evidence_flows_into_planner_payload_via_asdict(self):
    import dataclasses
    data = dict(self._BASE)
    data["evidence"] = [
        {"id": "beacon_photo", "real_world_source": "physical_photo"}
    ]
    seed = ConflictVLMSeed.from_dict(data)
    # conflict_llm_pipeline.generate() feeds dataclasses.asdict(seed) to every
    # planner layer, so the new field must survive the round-trip.
    seed_dict = dataclasses.asdict(seed)
    self.assertIn("evidence", seed_dict)
    self.assertEqual(seed_dict["evidence"][0]["id"], "beacon_photo")
    self.assertEqual(
        seed_dict["evidence"][0]["real_world_source"], "physical_photo")

  def test_evidencespec_from_dict_tolerates_empty(self):
    spec = ConflictEvidenceSpec.from_dict({})
    self.assertEqual(spec.id, "")
    self.assertEqual(spec.summary, "")
    self.assertEqual(spec.real_world_source, "")
    self.assertEqual(spec.role, "")


# ---------------------------------------------------------------------------
# Decisive visual evidence + vision fidelity audit (role orthogonal to kind:
# charts/rich_text/diagrams are pixel-exclusive too). Covers the three seams:
#   * _tag_decisive_assets                (materializer: tag legible carriers)
#   * _maybe_audit_and_fix_fidelity       (vlm_materializer: audit/regen/downgrade)
#   * _reconcile_decisive_fidelity_checks (materializer: relax orphaned checks)
# ---------------------------------------------------------------------------


def _img_asset_dict(
    asset_id, *, token="v4.2.1", anchor="E1", kind="natural_image",
    generator="google_image", in_text=True, pm=None,
):
  """A minimal image VLMAssetSpec dict for the decisive-evidence tests."""
  return {
      "id": asset_id,
      "kind": kind,
      "generator": generator,
      "path": f"charts/{asset_id}.png",
      "mime_type": "image/png",
      "available_from_round": 1,
      "anchor_ids": [anchor],
      "caption": f"nameplate reads {token}" if in_text else "a nameplate",
      "ocr_text": token if in_text else "",
      "prompt": "photo",
      "content_b64": "QUJD",
      "provider_metadata": dict(pm or {}),
  }


class TestTagDecisiveAssets(unittest.TestCase):
  """Task 2: tag an image asset ``decisive`` iff it backs a round's deciding
  anchor AND legibly carries that round's ``resolved_value`` (role is orthogonal
  to kind; most photos stay untagged)."""

  def _spec(self, assets, rounds):
    from clasharness.scenario.conflict_vlm_spec import ConflictVLMScenarioSpec
    return ConflictVLMScenarioSpec.from_dict({
        "series": "conflict_vlm_v1", "scenario_slug": "t", "title": "T",
        "domain": "d", "core_conflict": "c", "conflict_taxonomy": [],
        "modalities": [], "allowed_tool_families": ["asset"],
        "assets": assets, "rounds": rounds, "evals": [],
    })

  def _round(self, resolved="v4.2.1", anchor="E1", rid="r1"):
    return {
        "round_id": rid, "round_type": "resolution", "visible_asset_ids": [],
        "required_modalities": [], "expected_anchor_ids": [anchor],
        "required_tool_families": [], "resolved_value": resolved,
    }

  def test_tags_matching_photo(self):
    spec = self._spec([_img_asset_dict("img_A")], [self._round()])
    n = _tag_decisive_assets(spec)
    self.assertEqual(n, 1)
    pm = spec.assets[0].provider_metadata
    self.assertEqual(pm.get("evidence_role"), "decisive")
    self.assertEqual(pm.get("expected_token"), "v4.2.1")

  def test_no_tag_when_anchor_mismatch(self):
    spec = self._spec([_img_asset_dict("img_A", anchor="E1")],
                      [self._round(anchor="E9")])
    self.assertEqual(_tag_decisive_assets(spec), 0)
    self.assertNotIn("evidence_role", spec.assets[0].provider_metadata)

  def test_no_tag_when_value_absent_from_grounded_text(self):
    spec = self._spec([_img_asset_dict("img_A", in_text=False)], [self._round()])
    self.assertEqual(_tag_decisive_assets(spec), 0)

  def test_tags_chart_kind_too(self):
    # Role is orthogonal to kind. A chart renders to a PNG via headless Chrome
    # and the runtime strips its caption/ocr_text exactly as it does a photo's,
    # so it is just as pixel-exclusive and must be audited. Charts are also the
    # overwhelming majority (~93-95%) of decisive-eligible assets, so excluding
    # them here left almost all decisive evidence unverified.
    chart = {
        "id": "chart_A", "kind": "chart", "generator": "template_chart",
        "path": "charts/chart_A.png", "mime_type": "image/png",
        "available_from_round": 1, "anchor_ids": ["E1"],
        "caption": "v4.2.1", "ocr_text": "v4.2.1", "chart_spec_path": "x.json",
    }
    spec = self._spec([chart], [self._round()])
    self.assertEqual(_tag_decisive_assets(spec), 1)
    pm = spec.assets[0].provider_metadata
    self.assertEqual(pm.get("evidence_role"), "decisive")
    self.assertEqual(pm.get("expected_token"), "v4.2.1")

  def test_no_tag_for_chart_without_the_value(self):
    # The precision guard (b) still applies to non-image kinds: backing the
    # deciding anchor is not enough, the grounded text must bear the value.
    chart = {
        "id": "chart_A", "kind": "chart", "generator": "template_chart",
        "path": "charts/chart_A.png", "mime_type": "image/png",
        "available_from_round": 1, "anchor_ids": ["E1"],
        "caption": "quarterly totals", "ocr_text": "Q1 Q2 Q3",
        "chart_spec_path": "x.json",
    }
    spec = self._spec([chart], [self._round()])
    self.assertEqual(_tag_decisive_assets(spec), 0)
    self.assertNotIn("evidence_role", spec.assets[0].provider_metadata)

  def test_empty_resolved_value_round_skipped(self):
    spec = self._spec([_img_asset_dict("img_A")], [self._round(resolved="")])
    self.assertEqual(_tag_decisive_assets(spec), 0)

  def test_first_decisive_round_wins_token(self):
    # Same asset backs two rounds; the first tagged round claims expected_token.
    asset = _img_asset_dict("img_A")
    asset["ocr_text"] = "v4.2.1 and v5.0.0"
    asset["caption"] = "shows v4.2.1 and v5.0.0"
    spec = self._spec(
        [asset],
        [self._round(resolved="v4.2.1", rid="r1"),
         self._round(resolved="v5.0.0", rid="r2")],
    )
    n = _tag_decisive_assets(spec)
    self.assertEqual(n, 1)
    self.assertEqual(spec.assets[0].provider_metadata["expected_token"], "v4.2.1")


class TestConsistencyLayerDoesNotInjectAnchors(unittest.TestCase):
  """The removed Rule 3 appended "[System Note: Evidence anchor for this state
  is <ANCHOR>]" to a visible workspace file whenever an expected anchor was
  absent from all of them -- i.e. exactly when the anchor was carried only by a
  visual asset. That guard made it a targeted eraser of asset exclusivity, and
  it fired on 44%/85.5% of compiled tasks in the two generated sets. Anchor
  discoverability is owned by llm_auditor's Auditor 2, which can see the visual
  assets and so knows an asset-only anchor is already discoverable."""

  BANNED = "[System Note: Evidence anchor for this state is"

  def _spec(self, *, overrides, anchors, checks=None):
    task_spec = {
        "instruction": "Reconcile the two reports and file a decision.",
        "ground_truth": {"evaluation_checks": checks or []},
        "data": {
            "public": {},
            "private": {
                "expected_anchor_ids": anchors,
                "initial_state_overrides": dict(overrides),
            },
        },
    }
    return ScenarioSpec(
        scenario_id="s1", generator_id="g", scenario_slug="s", title="T",
        rounds=[ScenarioRound(
            round_id="r1", task_id="t1",
            instruction=task_spec["instruction"], task_spec=task_spec,
            expected_anchor_ids=list(anchors),
        )],
    )

  def _overrides_of(self, spec):
    return spec.rounds[0].task_spec["data"]["private"]["initial_state_overrides"]

  def test_asset_only_anchor_is_left_out_of_visible_files(self):
    # The exact shape that used to trigger the injection: the anchor appears in
    # no visible workspace file because it lives in a chart/photo.
    files = {"workspace/docs/routing_status.md": "# Routing\n\n1. Page the oncall.\n"}
    spec = self._spec(overrides=files, anchors=["EVIDENCE:ACCESS_SCOPE_MATRIX"])
    audit_and_repair_scenario(spec)
    out = self._overrides_of(spec)["workspace/docs/routing_status.md"]
    self.assertNotIn(self.BANNED, out)
    self.assertNotIn("EVIDENCE:ACCESS_SCOPE_MATRIX", out)
    self.assertEqual(out, files["workspace/docs/routing_status.md"])

  def test_no_system_note_anywhere_in_the_repaired_spec(self):
    files = {
        "workspace/index.md": "overview\n",
        "workspace/logs/status.txt": "ok\n",
        "workspace/config/routing.json": '{"mode":"auto"}\n',
    }
    spec = self._spec(
        overrides=files,
        anchors=["EVIDENCE:BAY_PALLET_PHOTO", "EVIDENCE:GATE_BOARD"],
    )
    audit_and_repair_scenario(spec)
    for path, content in self._overrides_of(spec).items():
      self.assertNotIn(self.BANNED, str(content), msg=f"injected into {path}")
    self.assertEqual(self._overrides_of(spec), files)

  def test_anchor_already_present_is_still_untouched(self):
    # Control: the rule never fired in this case either, so behaviour is
    # unchanged. Guards against a replacement rule that rewrites unconditionally.
    body = "Cite EVIDENCE:GATE_BOARD when escalating.\n"
    spec = self._spec(overrides={"workspace/index.md": body},
                      anchors=["EVIDENCE:GATE_BOARD"])
    audit_and_repair_scenario(spec)
    self.assertEqual(self._overrides_of(spec)["workspace/index.md"], body)


class _FakeImageGen:
  """Counts generate() calls and returns a deterministic GeneratedAsset."""

  def __init__(self):
    self.calls = 0

  def generate(self, *, prompt, caption, anchors, size=(960, 540)):
    self.calls += 1
    return GeneratedAsset(data=b"PNGDATA", mime_type="image/png", extension="png")


class _PassClient:
  def audit_image_fidelity(self, *, image_bytes, mime_type, expected_token, context=""):
    return {"legible": True, "seen_text": expected_token}


class _FailClient:
  def __init__(self):
    self.calls = 0

  def audit_image_fidelity(self, *, image_bytes, mime_type, expected_token, context=""):
    self.calls += 1
    return {"legible": False, "reason": "blurry"}


class _FailThenPassClient:
  def __init__(self, fail_times):
    self.calls = 0
    self.fail_times = fail_times

  def audit_image_fidelity(self, *, image_bytes, mime_type, expected_token, context=""):
    self.calls += 1
    return {"legible": self.calls > self.fail_times}


class _ErrorClient:
  def audit_image_fidelity(self, **kwargs):
    raise RuntimeError("vision backend down")


class _NonDictClient:
  def audit_image_fidelity(self, **kwargs):
    return "not a dict"


class _NoMethodClient:
  pass


class TestMaybeAuditAndFixFidelity(unittest.TestCase):
  """Task 4: decisive images must legibly show their token; audit -> regen ->
  auto-downgrade. The user-mandated failure action is degrade (never fail-loud);
  corroborating images are never audited."""

  def _asset(self, *, role="decisive", token="v4.2.1"):
    pm = {}
    if role:
      pm["evidence_role"] = role
    if token:
      pm["expected_token"] = token
    return VLMAssetSpec.from_dict(_img_asset_dict("img_A", token=token, pm=pm))

  def _gen(self):
    return GeneratedAsset(data=b"ORIG", mime_type="image/png", extension="png")

  def _regen(self, image_gen):
    """Adapt a fake generator to the ``regenerate(note) -> GeneratedAsset``
    callback. Real call sites close over the image prompt or the HTML compose
    payload; here we only need the call count."""
    return lambda note: image_gen.generate(
        prompt=f"p\n{note}", caption="", anchors=[])

  def test_disabled_channel_is_skipped(self):
    # ``enabled`` is the caller's "these are real pixels" answer (mock/none
    # providers, and template charts under a non-HTML chart provider, pass
    # False). The provider -> enabled mapping now lives at the call sites.
    asset, gen, ig = self._asset(), self._gen(), _FakeImageGen()
    out = _maybe_audit_and_fix_fidelity(
        asset, gen, self._regen(ig), enabled=False, llm_client=_FailClient(),
        retries=3, progress=None)
    self.assertIs(out, gen)
    self.assertEqual(ig.calls, 0)
    self.assertNotIn("fidelity_failed", asset.provider_metadata)

  def test_non_decisive_asset_untouched(self):
    asset = self._asset(role="corroborating")
    gen, ig = self._gen(), _FakeImageGen()
    out = _maybe_audit_and_fix_fidelity(
        asset, gen, self._regen(ig), enabled=True, llm_client=_FailClient(),
        retries=3, progress=None)
    self.assertIs(out, gen)
    self.assertEqual(ig.calls, 0)
    self.assertNotIn("fidelity_failed", asset.provider_metadata)

  def test_missing_token_untouched(self):
    asset = self._asset(token="")
    gen, ig = self._gen(), _FakeImageGen()
    out = _maybe_audit_and_fix_fidelity(
        asset, gen, self._regen(ig), enabled=True, llm_client=_FailClient(),
        retries=3, progress=None)
    self.assertIs(out, gen)
    self.assertEqual(ig.calls, 0)

  def test_none_client_untouched(self):
    asset, gen, ig = self._asset(), self._gen(), _FakeImageGen()
    out = _maybe_audit_and_fix_fidelity(
        asset, gen, self._regen(ig), enabled=True, llm_client=None,
        retries=3, progress=None)
    self.assertIs(out, gen)
    self.assertEqual(ig.calls, 0)

  def test_legible_first_pass_no_regen(self):
    asset, gen, ig = self._asset(), self._gen(), _FakeImageGen()
    out = _maybe_audit_and_fix_fidelity(
        asset, gen, self._regen(ig), enabled=True, llm_client=_PassClient(),
        retries=3, progress=None)
    self.assertIs(out, gen)
    self.assertEqual(ig.calls, 0)
    self.assertNotIn("fidelity_failed", asset.provider_metadata)
    self.assertEqual(asset.provider_metadata["evidence_role"], "decisive")

  def test_persistent_failure_downgrades_and_flags(self):
    asset, gen = self._asset(), self._gen()
    ig, client = _FakeImageGen(), _FailClient()
    out = _maybe_audit_and_fix_fidelity(
        asset, gen, self._regen(ig), enabled=True, llm_client=client,
        retries=3, progress=None)
    # Downgraded in place; the bound decisive check is relaxed downstream.
    self.assertTrue(asset.provider_metadata["fidelity_failed"])
    self.assertEqual(asset.provider_metadata["evidence_role"], "corroborating")
    # 3 regen attempts (cap), + 1 initial audit + 3 retry audits.
    self.assertEqual(ig.calls, 3)
    self.assertEqual(client.calls, 4)
    self.assertIsInstance(out, GeneratedAsset)

  def test_detect_only_downgrades_without_regenerating(self):
    # ``regenerate=None`` is the deterministic-render channel (template chart):
    # re-running it is a pure function of the asset's own text and would
    # reproduce identical pixels, so the audit runs exactly once as a detector
    # and goes straight to the downgrade instead of burning the retry budget.
    asset, gen, client = self._asset(), self._gen(), _FailClient()
    out = _maybe_audit_and_fix_fidelity(
        asset, gen, None, enabled=True, llm_client=client,
        retries=3, progress=None, channel="chart")
    self.assertIs(out, gen)
    self.assertEqual(client.calls, 1)
    self.assertTrue(asset.provider_metadata["fidelity_failed"])
    self.assertEqual(asset.provider_metadata["evidence_role"], "corroborating")

  def test_detect_only_legible_leaves_role_intact(self):
    asset, gen = self._asset(), self._gen()
    out = _maybe_audit_and_fix_fidelity(
        asset, gen, None, enabled=True, llm_client=_PassClient(),
        retries=3, progress=None, channel="chart")
    self.assertIs(out, gen)
    self.assertNotIn("fidelity_failed", asset.provider_metadata)
    self.assertEqual(asset.provider_metadata["evidence_role"], "decisive")

  def test_retry_cap_is_three(self):
    asset, gen, ig = self._asset(), self._gen(), _FakeImageGen()
    _maybe_audit_and_fix_fidelity(
        asset, gen, self._regen(ig), enabled=True, llm_client=_FailClient(),
        retries=10, progress=None)
    self.assertEqual(ig.calls, 3)  # min(retries, 3)

  def test_regen_then_legible_recovers(self):
    asset, gen = self._asset(), self._gen()
    ig, client = _FakeImageGen(), _FailThenPassClient(fail_times=2)
    out = _maybe_audit_and_fix_fidelity(
        asset, gen, self._regen(ig), enabled=True, llm_client=client,
        retries=3, progress=None)
    # initial audit fails, retry1 fails, retry2 passes -> recovered, no downgrade.
    self.assertNotIn("fidelity_failed", asset.provider_metadata)
    self.assertEqual(asset.provider_metadata["evidence_role"], "decisive")
    self.assertEqual(ig.calls, 2)
    self.assertIsNot(out, gen)  # returns the regenerated asset

  def test_audit_error_is_benign_pass(self):
    asset, gen, ig = self._asset(), self._gen(), _FakeImageGen()
    out = _maybe_audit_and_fix_fidelity(
        asset, gen, self._regen(ig), enabled=True, llm_client=_ErrorClient(),
        retries=3, progress=None)
    self.assertIs(out, gen)
    self.assertNotIn("fidelity_failed", asset.provider_metadata)

  def test_nondict_audit_is_benign_pass(self):
    asset, gen, ig = self._asset(), self._gen(), _FakeImageGen()
    out = _maybe_audit_and_fix_fidelity(
        asset, gen, self._regen(ig), enabled=True, llm_client=_NonDictClient(),
        retries=3, progress=None)
    self.assertIs(out, gen)
    self.assertNotIn("fidelity_failed", asset.provider_metadata)

  def test_client_without_method_is_benign_pass(self):
    asset, gen, ig = self._asset(), self._gen(), _FakeImageGen()
    out = _maybe_audit_and_fix_fidelity(
        asset, gen, self._regen(ig), enabled=True, llm_client=_NoMethodClient(),
        retries=3, progress=None)
    self.assertIs(out, gen)
    self.assertNotIn("fidelity_failed", asset.provider_metadata)

  def test_audit_safe_reports_skip_for_missing_method(self):
    res = _audit_fidelity_safe(_NoMethodClient(), self._gen(), "v4.2.1",
                               self._asset(), None)
    self.assertTrue(res["skipped"])
    self.assertTrue(res["legible"])

  def test_vision_audit_false_skips_entirely(self):
    # The --novision_audit escape hatch: never audit, never downgrade, even for a
    # real provider with a failing client.
    asset, gen, ig = self._asset(), self._gen(), _FakeImageGen()
    client = _FailClient()
    out = _maybe_audit_and_fix_fidelity(
        asset, gen, self._regen(ig), enabled=True, llm_client=client,
        retries=3, progress=None, vision_audit=False)
    self.assertIs(out, gen)
    self.assertEqual(ig.calls, 0)
    self.assertEqual(client.calls, 0)
    self.assertNotIn("fidelity_failed", asset.provider_metadata)
    self.assertEqual(asset.provider_metadata["evidence_role"], "decisive")

  # -- vision_verified bookkeeping -------------------------------------------
  # A skip and a pass are deliberately identical *control flow* (a flaky audit
  # must never downgrade a good asset), which is exactly why they must not be
  # identical *records*: otherwise "N decisive assets audited" silently
  # conflates "N confirmed legible by a vision model" with "N never actually
  # checked", and the dataset's only positive evidence that a round is solvable
  # by looking becomes unfalsifiable.

  def test_pass_records_verified(self):
    asset, gen, ig = self._asset(), self._gen(), _FakeImageGen()
    _maybe_audit_and_fix_fidelity(
        asset, gen, self._regen(ig), enabled=True, llm_client=_PassClient(),
        retries=3, progress=None)
    self.assertIs(asset.provider_metadata["vision_verified"], True)
    self.assertNotIn("vision_audit_skipped", asset.provider_metadata)

  def test_skips_record_unverified_with_reason(self):
    # Every benign-pass path must be distinguishable from a real pass.
    for name, client in (("no method", _NoMethodClient()),
                         ("error", _ErrorClient()),
                         ("non-dict", _NonDictClient())):
      with self.subTest(client=name):
        asset, gen, ig = self._asset(), self._gen(), _FakeImageGen()
        out = _maybe_audit_and_fix_fidelity(
            asset, gen, self._regen(ig), enabled=True, llm_client=client,
            retries=3, progress=None)
        self.assertIs(out, gen)  # still a benign pass: no regen, no downgrade
        self.assertEqual(ig.calls, 0)
        self.assertNotIn("fidelity_failed", asset.provider_metadata)
        self.assertIs(asset.provider_metadata["vision_verified"], False)
        self.assertTrue(asset.provider_metadata["vision_audit_skipped"])

  def test_recovery_clears_the_unverified_reason(self):
    # First audit says illegible (not a skip), a retry then passes: the asset
    # ends up genuinely verified with no stale skip reason left behind.
    asset, gen = self._asset(), self._gen()
    ig, client = _FakeImageGen(), _FailThenPassClient(fail_times=2)
    _maybe_audit_and_fix_fidelity(
        asset, gen, self._regen(ig), enabled=True, llm_client=client,
        retries=3, progress=None)
    self.assertIs(asset.provider_metadata["vision_verified"], True)
    self.assertNotIn("vision_audit_skipped", asset.provider_metadata)

  def test_persistent_failure_records_unverified(self):
    asset, gen, ig = self._asset(), self._gen(), _FakeImageGen()
    _maybe_audit_and_fix_fidelity(
        asset, gen, self._regen(ig), enabled=True, llm_client=_FailClient(),
        retries=3, progress=None)
    self.assertIs(asset.provider_metadata["vision_verified"], False)
    # A judged-illegible asset is not a skip: the audit did return a verdict.
    self.assertNotIn("vision_audit_skipped", asset.provider_metadata)
    self.assertTrue(asset.provider_metadata["fidelity_failed"])

  def test_channel_is_recorded(self):
    # The channel makes the downstream report attributable: an unverified chart
    # (deterministic renderer) and an unverified photo (image model) have
    # completely different causes and fixes.
    for channel in ("image", "chart", "model-chart"):
      with self.subTest(channel=channel):
        asset, gen, ig = self._asset(), self._gen(), _FakeImageGen()
        _maybe_audit_and_fix_fidelity(
            asset, gen, self._regen(ig), enabled=True,
            llm_client=_PassClient(), retries=3, progress=None,
            channel=channel)
        self.assertEqual(
            asset.provider_metadata["vision_audit_channel"], channel)

  def test_no_vision_keys_when_audit_never_ran(self):
    # Absent keys (not vision_verified=False) for every pre-audit bail-out, so
    # the report can separate "audit ran and told us nothing" from "this asset
    # was never in scope".
    cases = {
        "disabled channel": dict(enabled=False, llm_client=_PassClient()),
        "novision_audit": dict(enabled=True, llm_client=_PassClient(),
                               vision_audit=False),
        "no client": dict(enabled=True, llm_client=None),
    }
    for name, kwargs in cases.items():
      with self.subTest(case=name):
        asset, gen, ig = self._asset(), self._gen(), _FakeImageGen()
        _maybe_audit_and_fix_fidelity(
            asset, gen, self._regen(ig), retries=3, progress=None, **kwargs)
        for key in ("vision_verified", "vision_audit_skipped",
                    "vision_audit_channel"):
          self.assertNotIn(key, asset.provider_metadata)
    # Same for the two asset-shape bail-outs.
    for name, asset in (("corroborating", self._asset(role="corroborating")),
                        ("no token", self._asset(token=""))):
      with self.subTest(case=name):
        _maybe_audit_and_fix_fidelity(
            asset, self._gen(), self._regen(_FakeImageGen()), enabled=True,
            llm_client=_PassClient(), retries=3, progress=None)
        self.assertNotIn("vision_verified", asset.provider_metadata)


class TestReconcileDecisiveFidelityChecks(unittest.TestCase):
  """Task 5: when a decisive photo cannot render its token legibly and no other
  legible channel carries the round's ``resolved_value``, relax the
  value-asserting checks to ``role="diagnostic"`` (recorded, non-gating)."""

  TOKEN = "v4.2.1"

  def _checks(self):
    return [
        {"type": "file_check", "path": "decision.md", "condition": "contains",
         "expected": self.TOKEN, "role": "scored"},
        {"type": "command_check",
         "command": f"file create --path d.md --content {self.TOKEN}",
         "role": "scored"},
        {"type": "effect", "effect_type": "field_semantic_match",
         "expected": {"field": "answer", "value": self.TOKEN},
         "condition": "field_semantic_match"},
        {"type": "file_check", "path": "other.md", "condition": "exists",
         "role": "scored"},  # no token -> never relaxed
        {"type": "effect", "effect_type": "assets_accessed",
         "expected": {"field": "id", "value": "img_A"},
         "role": "diagnostic"},  # already diagnostic -> not re-relaxed
    ]

  def _failed_photo(self, aid="img_A"):
    return _img_asset_dict(aid, token=self.TOKEN, pm={
        "evidence_role": "corroborating", "expected_token": self.TOKEN,
        "fidelity_failed": True})

  def _spec(self, assets, *, workspace_files=None, sessions=None,
            resolved="v4.2.1"):
    from clasharness.scenario.conflict_vlm_spec import ConflictVLMScenarioSpec
    spec = ConflictVLMScenarioSpec.from_dict({
        "series": "conflict_vlm_v1", "scenario_slug": "t", "title": "T",
        "domain": "d", "core_conflict": "c", "conflict_taxonomy": [],
        "modalities": [], "allowed_tool_families": ["file", "asset"],
        "assets": assets,
        "rounds": [{
            "round_id": "r1", "round_type": "resolution",
            "visible_asset_ids": [a["id"] for a in assets],
            "required_modalities": [], "expected_anchor_ids": ["E1"],
            "required_tool_families": [], "resolved_value": resolved,
        }],
        "evals": [], "workspace": {"files": workspace_files or []},
        "sessions": sessions or [],
    })
    return spec, spec.rounds[0]

  def _run(self, spec, rnd):
    return _reconcile_decisive_fidelity_checks(
        self._checks(), rnd, list(spec.assets), spec)

  def test_noop_when_no_resolved_value(self):
    spec, rnd = self._spec([self._failed_photo()], resolved="")
    reconciled, relaxed = self._run(spec, rnd)
    self.assertEqual(relaxed, [])

  def test_noop_when_no_failed_photo(self):
    ok = _img_asset_dict("img_A", token=self.TOKEN, pm={
        "evidence_role": "decisive", "expected_token": self.TOKEN})
    spec, rnd = self._spec([ok])
    _, relaxed = self._run(spec, rnd)
    self.assertEqual(relaxed, [])

  def test_relaxes_value_checks_when_no_carrier(self):
    spec, rnd = self._spec([self._failed_photo()])
    reconciled, relaxed = self._run(spec, rnd)
    self.assertEqual(len(relaxed), 3)  # file_check, command_check, semantic
    # The token-free check and the already-diagnostic check are untouched.
    roles = [c.get("role") for c in reconciled]
    self.assertEqual(roles.count("diagnostic"), 4)  # 3 newly + 1 pre-existing
    # 'other.md exists' stays scored.
    other = [c for c in reconciled if c.get("path") == "other.md"][0]
    self.assertEqual(other["role"], "scored")

  def test_workspace_file_carrier_keeps_checks(self):
    spec, rnd = self._spec(
        [self._failed_photo()],
        workspace_files=[{"path": "notes.md", "content": f"pin {self.TOKEN}"}])
    _, relaxed = self._run(spec, rnd)
    self.assertEqual(relaxed, [])

  def test_session_carrier_keeps_checks(self):
    spec, rnd = self._spec(
        [self._failed_photo()],
        sessions=[{"session_id": "s1", "messages": [
            {"message_id": "m1", "sender": "u", "text": f"use {self.TOKEN}"}]}])
    _, relaxed = self._run(spec, rnd)
    self.assertEqual(relaxed, [])

  def test_chart_carrier_keeps_checks(self):
    chart = {
        "id": "chart_B", "kind": "chart", "generator": "template_chart",
        "path": "charts/chart_B.png", "mime_type": "image/png",
        "available_from_round": 1, "anchor_ids": ["E2"],
        "caption": "release", "ocr_text": self.TOKEN, "chart_spec_path": "x.json",
    }
    spec, rnd = self._spec([self._failed_photo(), chart])
    _, relaxed = self._run(spec, rnd)
    self.assertEqual(relaxed, [])

  def test_audit_passed_decisive_photo_is_carrier(self):
    passed = _img_asset_dict("img_C", token=self.TOKEN, pm={
        "evidence_role": "decisive", "expected_token": self.TOKEN})
    spec, rnd = self._spec([self._failed_photo(), passed])
    _, relaxed = self._run(spec, rnd)
    self.assertEqual(relaxed, [])

  def test_corroborating_photo_is_not_a_carrier(self):
    # A corroborating natural image may itself be illegible -> not reliable.
    corrob = _img_asset_dict("img_D", token=self.TOKEN, anchor="E9", pm={})
    spec, rnd = self._spec([self._failed_photo(), corrob])
    _, relaxed = self._run(spec, rnd)
    self.assertEqual(len(relaxed), 3)

  def test_check_asserts_value_helper(self):
    self.assertTrue(_check_asserts_value(
        {"expected": {"value": "v4.2.1"}}, "v4.2.1"))
    self.assertFalse(_check_asserts_value(
        {"path": "other.md", "condition": "exists"}, "v4.2.1"))
    self.assertFalse(_check_asserts_value("notadict", "v4.2.1"))

  def test_readable_carrier_helper_direct(self):
    spec, _ = self._spec([self._failed_photo()],
                         workspace_files=[{"path": "n.md", "content": "v4.2.1"}])
    self.assertTrue(_resolved_value_has_readable_carrier(
        "v4.2.1", {"img_A"}, list(spec.assets), spec))
    spec2, _ = self._spec([self._failed_photo()])
    self.assertFalse(_resolved_value_has_readable_carrier(
        "v4.2.1", {"img_A"}, list(spec2.assets), spec2))


class _AuditRaisingClient:
  """generate_layer always raises, simulating an intermittent LLM error
  (429/503/timeout/JSON-parse/safety-block) during the auditor pass."""

  def __init__(self):
    self.calls = 0

  def audit_image_fidelity(self, **kwargs):
    return {"legible": True, "skipped": True, "reason": "mock client"}

  def generate_layer(self, layer, payload):
    self.calls += 1
    raise RuntimeError("simulated LLM backend error")


class TestAuditorNeverDiscards(unittest.TestCase):
  """P1: the LLM auditor is a best-effort repair layer, not a gate.

  It runs before scenario.json is written, so an uncaught error there would
  abort materialization and drop the whole version (the root cause of broken
  scenario dirs with no scenario.json). audit_scenario_with_llm must swallow the
  error and return the spec un-audited instead; F1 remains the non-discarding
  backstop downstream.
  """

  def _spec_with_round(self):
    task_spec = {
        "instruction": "Do the thing.",
        "ground_truth": {"evaluation_checks": []},
        "data": {"private": {"initial_state_overrides": {}}, "public": {}},
    }
    rnd = ScenarioRound(
        round_id="r01",
        task_id="t_r01",
        instruction="Do the thing.",
        task_spec=task_spec,
    )
    return ScenarioSpec(
        scenario_id="s", generator_id="g", scenario_slug="slug", title="T",
        rounds=[rnd],
    )

  def test_generate_layer_error_does_not_propagate(self):
    spec = self._spec_with_round()
    client = _AuditRaisingClient()
    # Must NOT raise, and must return the (un-audited) spec with its round intact.
    out = audit_scenario_with_llm(spec, client, max_iterations=3)
    self.assertIs(out, spec)
    self.assertEqual(len(out.rounds), 1)
    self.assertGreaterEqual(client.calls, 1)  # the audit was attempted

  def test_empty_rounds_does_not_crash_executor(self):
    spec = ScenarioSpec(
        scenario_id="s", generator_id="g", scenario_slug="slug", title="T",
        rounds=[],
    )
    # Without the guard, ThreadPoolExecutor(max_workers=0) raises ValueError.
    out = audit_scenario_with_llm(spec, _AuditRaisingClient())
    self.assertIs(out, spec)
    self.assertEqual(out.rounds, [])


class TestTaskVisibleTextSurfaces(unittest.TestCase):
  """L5b: read the agent-readable TEXT surfaces off the COMPILED task.

  This deliberately reads the shipped ``task_spec`` rather than re-deriving the
  state from the authoring spec. Measured on the 3.6-flash corpus a
  reconstruction found 1 workspace leak where the compiled task shows 32,
  because ``initial_state_overrides`` carries app-state documents
  (``emails.json``, ``tasks.json``, ...) that ``spec.workspace.files`` never
  exposes."""

  def _task(self, *, overrides=None, events=None, mocks=None):
    return {"data": {"private": {
        "initial_state_overrides": overrides or {},
        "background_events": events or [],
        "mock_subagents": mocks or [],
    }}}

  def test_state_overrides_split_by_kind(self):
    s = _task_visible_text_surfaces(self._task(overrides={
        "workspace/notes.md": "WS",
        "sessions/chan.jsonl": "SESS",
        "emails.json": "MAIL",       # app-state: readable via the email tools
        "tasks.json": "TASKS",
    }))
    self.assertCountEqual(s["workspace"], ["WS", "MAIL", "TASKS"])
    self.assertEqual(s["session"], ["SESS"])

  def test_asset_blobs_are_excluded(self):
    # The rendered pixels are the channel we are testing FOR; counting their
    # base64 as readable text would make every round look text-solvable.
    s = _task_visible_text_surfaces(self._task(overrides={
        "assets/portal/shot.png": "{'__bytes_b64__': 'AAAA'}",
        "asset_manifest.json": "{...}",
        "workspace/a.md": "WS",
    }))
    self.assertEqual(s["workspace"], ["WS"])
    self.assertEqual(s["session"], [])

  def test_background_events_route_to_their_surface(self):
    s = _task_visible_text_surfaces(self._task(events=[
        {"type": "file_update", "path": "a.md", "new_content": "FILE"},
        {"type": "session_append", "session_id": "c", "message": "MSG"},
        {"type": "subagent_completion", "agent_name": "A",
         "mock_response": "RESP", "state_effects": ["EFFECT"]},
        {"type": "unknown_future_kind", "payload": "IGNORED"},
    ]))
    self.assertEqual(s["workspace"], ["FILE"])
    self.assertEqual(s["session"], ["MSG"])
    self.assertIn("RESP", s["mock"])
    self.assertTrue(any("EFFECT" in m for m in s["mock"]))

  def test_mock_subagents_are_collected(self):
    s = _task_visible_text_surfaces(self._task(mocks=[{
        "agent_role": "r", "mock_response": "RESP",
        "state_effects": ["message send --content 'EFFECT'"],
        "private_knowledge": ["SECRET"], "capabilities": ["CAP"],
    }]))
    joined = " ".join(s["mock"])
    self.assertIn("RESP", joined)
    self.assertIn("EFFECT", joined)
    # Stripped before the task ships, so no solving agent can ever read them.
    self.assertNotIn("SECRET", joined)
    self.assertNotIn("CAP", joined)

  def test_empty_and_malformed_task_is_safe(self):
    for payload in ({}, {"data": {}}, {"data": {"private": None}},
                    self._task(events=["not-a-dict"], mocks=["not-a-dict"])):
      with self.subTest(payload=str(payload)[:40]):
        s = _task_visible_text_surfaces(payload)
        self.assertEqual(set(s), {"workspace", "session", "mock"})


class TestClassifyRoundVisionExclusivity(unittest.TestCase):
  """L5b: classify whether a round genuinely requires opening the asset."""

  TOKEN = "SEC-9042"

  def _spec(self, *, locus="asset:img_A", resolved=None, pm=None):
    resolved = self.TOKEN if resolved is None else resolved
    spec = ConflictVLMScenarioSpec.from_dict({
        "series": "conflict_vlm_v1", "scenario_slug": "t", "title": "T",
        "domain": "d", "core_conflict": "c", "conflict_taxonomy": [],
        "modalities": [], "allowed_tool_families": ["file", "asset"],
        "assets": [_img_asset_dict("img_A", token=self.TOKEN, pm=dict(
            pm if pm is not None else {"expected_token": self.TOKEN,
                                       "evidence_role": "decisive"}))],
        "rounds": [{
            "round_id": "r01", "round_type": "resolution",
            "visible_asset_ids": ["img_A"], "required_modalities": [],
            "expected_anchor_ids": ["E1"], "required_tool_families": [],
            "resolved_value": resolved, "resolved_value_locus": locus,
        }],
        "evals": [], "workspace": {"files": []}, "sessions": [],
    })
    return spec, spec.rounds[0]

  def _task(self, **surfaces):
    return {"data": {"private": {
        "initial_state_overrides": surfaces.get("overrides", {}),
        "background_events": surfaces.get("events", []),
        "mock_subagents": surfaces.get("mocks", []),
    }}}

  def test_no_resolved_value(self):
    spec, rnd = self._spec(resolved="")
    self.assertEqual(
        _classify_round_vision_exclusivity(spec, rnd, self._task())[0],
        "no_resolved_value")

  def test_non_asset_locus(self):
    spec, rnd = self._spec(locus="workspace:notes.md")
    self.assertEqual(
        _classify_round_vision_exclusivity(spec, rnd, self._task())[0],
        "non_asset_locus")

  def test_missing_task_spec_is_reported_not_guessed(self):
    # Silently falling back to the authoring spec would report an exclusivity
    # rate the dataset does not actually have. 3 of 58 scenarios in the
    # 3.5-flash corpus crashed before writing scenario.json; those 29 rounds
    # must surface as unmeasured rather than as clean.
    spec, rnd = self._spec()
    for missing in (None, {}, "not-a-dict"):
      with self.subTest(task_spec=missing):
        self.assertEqual(
            _classify_round_vision_exclusivity(spec, rnd, missing)[0],
            "not_compiled")

  def test_vision_exclusive_when_verified_and_withheld(self):
    spec, rnd = self._spec(pm={"expected_token": self.TOKEN,
                               "evidence_role": "decisive",
                               "vision_verified": True})
    verdict, leaked = _classify_round_vision_exclusivity(
        spec, rnd, self._task(overrides={"workspace/a.md": "nothing here"}))
    self.assertEqual(verdict, "vision_exclusive")
    self.assertEqual(leaked, [])

  def test_unverified_exclusive_when_audit_never_confirmed(self):
    spec, rnd = self._spec()  # no vision_verified key
    self.assertEqual(
        _classify_round_vision_exclusivity(spec, rnd, self._task())[0],
        "unverified_exclusive")

  def test_unsolvable_when_pixels_failed_and_text_withheld(self):
    spec, rnd = self._spec(pm={"expected_token": self.TOKEN,
                               "evidence_role": "corroborating",
                               "fidelity_failed": True})
    self.assertEqual(
        _classify_round_vision_exclusivity(spec, rnd, self._task())[0],
        "unsolvable")

  def test_text_solvable_names_every_leaking_surface(self):
    cases = {
        "workspace": self._task(overrides={"workspace/a.md": f"see {self.TOKEN}"}),
        "session": self._task(overrides={"sessions/c.jsonl": f"{self.TOKEN}"}),
        "mock": self._task(mocks=[{"mock_response": f"it is {self.TOKEN}"}]),
    }
    for surface, task in cases.items():
      with self.subTest(surface=surface):
        spec, rnd = self._spec()
        verdict, leaked = _classify_round_vision_exclusivity(spec, rnd, task)
        self.assertEqual(verdict, "text_solvable")
        self.assertEqual(leaked, [surface])

  def test_leak_detection_is_case_insensitive(self):
    spec, rnd = self._spec()
    verdict, leaked = _classify_round_vision_exclusivity(
        spec, rnd, self._task(overrides={"workspace/a.md": self.TOKEN.lower()}))
    self.assertEqual(verdict, "text_solvable")
    self.assertEqual(leaked, ["workspace"])

  def test_leak_beats_verification(self):
    # A vision-verified asset whose value also sits in a chat message is NOT a
    # vision round: the agent can answer without ever looking.
    spec, rnd = self._spec(pm={"expected_token": self.TOKEN,
                               "evidence_role": "decisive",
                               "vision_verified": True})
    verdict, _ = _classify_round_vision_exclusivity(
        spec, rnd, self._task(mocks=[{"mock_response": self.TOKEN}]))
    self.assertEqual(verdict, "text_solvable")


class TestVisionExclusivityReport(unittest.TestCase):
  """L5b: the two-layer end-of-generation report."""

  TOKEN = "SEC-9042"

  def _spec(self, assets):
    return ConflictVLMScenarioSpec.from_dict({
        "series": "conflict_vlm_v1", "scenario_slug": "t", "title": "T",
        "domain": "d", "core_conflict": "c", "conflict_taxonomy": [],
        "modalities": [], "allowed_tool_families": ["file", "asset"],
        "assets": assets,
        "rounds": [{
            "round_id": "r01", "round_type": "resolution",
            "visible_asset_ids": ["img_A"], "required_modalities": [],
            "expected_anchor_ids": ["E1"], "required_tool_families": [],
            "resolved_value": self.TOKEN,
            "resolved_value_locus": "asset:img_A",
        }],
        "evals": [], "workspace": {"files": []}, "sessions": [],
    })

  def test_summary_counts_downgraded_assets(self):
    # A failed audit rewrites evidence_role to "corroborating"; keying the
    # decisive population off that would make the failures vanish from their
    # own report, so expected_token is the stable marker.
    spec = self._spec([
        _img_asset_dict("img_A", token=self.TOKEN, pm={
            "expected_token": self.TOKEN, "evidence_role": "corroborating",
            "fidelity_failed": True, "vision_audit_channel": "chart"}),
        _img_asset_dict("img_B", token=self.TOKEN, pm={
            "expected_token": self.TOKEN, "evidence_role": "decisive",
            "vision_verified": True, "vision_audit_channel": "image"}),
        _img_asset_dict("img_C", token=self.TOKEN, pm={
            "expected_token": self.TOKEN, "vision_verified": False,
            "vision_audit_skipped": "client has no vision audit",
            "vision_audit_channel": "image"}),
        _img_asset_dict("img_D", token=self.TOKEN, pm={}),  # not decisive
    ])
    a = _summarize_vision_audit(spec)
    self.assertEqual(a["decisive"], 3)
    self.assertEqual(a["verified"], 1)
    self.assertEqual(a["failed"], 1)
    self.assertEqual(a["never_ran"], 0)
    self.assertEqual(a["channels"], {"chart": 1, "image": 2})
    self.assertEqual(a["skipped_reasons"], {"client has no vision audit": 1})

  def test_untagged_assets_are_not_counted_as_decisive(self):
    spec = self._spec([_img_asset_dict("img_A", token=self.TOKEN, pm={})])
    self.assertEqual(_summarize_vision_audit(spec)["decisive"], 0)

  def test_never_audited_is_distinct_from_skipped(self):
    spec = self._spec([_img_asset_dict("img_A", token=self.TOKEN, pm={
        "expected_token": self.TOKEN, "evidence_role": "decisive"})])
    a = _summarize_vision_audit(spec)
    self.assertEqual(a["never_ran"], 1)
    self.assertEqual(a["verified"], 0)
    self.assertEqual(a["channels"], {})

  def _state(self, task_spec):
    from clasharness.scenario.spec import ScenarioRound, ScenarioSpec
    return ScenarioSpec(
        scenario_id="s", generator_id="g", scenario_slug="t", title="T",
        rounds=[ScenarioRound(round_id="r01", task_id="t_r01",
                              instruction="i", task_spec=task_spec)])

  def test_report_reflects_the_leak(self):
    spec = self._spec([_img_asset_dict("img_A", token=self.TOKEN, pm={
        "expected_token": self.TOKEN, "evidence_role": "decisive",
        "vision_verified": True, "vision_audit_channel": "image"})])
    state = self._state({"data": {"private": {
        "initial_state_overrides": {"workspace/a.md": f"value {self.TOKEN}"}}}})
    text = "\n".join(_vision_exclusivity_report(spec, state))
    self.assertIn("1 vision-verified", text)
    self.assertIn("1 text-solvable", text)
    self.assertIn("leaked via: workspace=1", text)

  def test_report_reflects_a_clean_vision_round(self):
    spec = self._spec([_img_asset_dict("img_A", token=self.TOKEN, pm={
        "expected_token": self.TOKEN, "evidence_role": "decisive",
        "vision_verified": True, "vision_audit_channel": "image"})])
    state = self._state({"data": {"private": {
        "initial_state_overrides": {"workspace/a.md": "unrelated"}}}})
    text = "\n".join(_vision_exclusivity_report(spec, state))
    self.assertIn("1 vision-exclusive", text)
    self.assertNotIn("leaked via", text)

  def test_report_warns_when_a_round_was_never_compiled(self):
    spec = self._spec([_img_asset_dict("img_A", token=self.TOKEN, pm={
        "expected_token": self.TOKEN, "evidence_role": "decisive"})])
    text = "\n".join(_vision_exclusivity_report(spec, None))
    self.assertIn("could not be measured", text)

  def test_report_never_raises_on_an_empty_scenario(self):
    spec = ConflictVLMScenarioSpec.from_dict({
        "series": "conflict_vlm_v1", "scenario_slug": "t", "title": "T",
        "domain": "d", "core_conflict": "c", "conflict_taxonomy": [],
        "modalities": [], "allowed_tool_families": [], "assets": [],
        "rounds": [], "evals": [], "workspace": {"files": []},
        "sessions": [],
    })
    text = "\n".join(_vision_exclusivity_report(spec, None))
    self.assertIn("no decisive assets tagged", text)


class TestTheExclusivityReportScansEveryLocus(unittest.TestCase):
  """L5b: the report must not certify a population it never looked at.

  The scan used to be gated on an ``asset:`` locus, so any round whose locus
  said ``computed`` or ``file:`` was filed as a clean ``non_asset_locus``
  without a single text surface being read. That is precisely the population
  that most needed reading: a round is re-labelled ``computed`` when the
  confidentiality gate rejects it for leaking, so "non-asset locus" had become
  a synonym for "was caught leaking and relabelled".

  Re-classifying two shipped corpora with the scan ungated:

      verdict                            v2     v3
      non_asset_locus_text_solvable       2     56
      text_solvable                       3      9
      TEXT-SOLVABLE (all loci)      5 (1.0%)  65 (11.9%)
      as the old report printed it       0.6%   1.8%

  The old report understated the shipped rate by 6.6x. The headline line now
  runs against every round that carries a decisive value, and the asset-locus
  denominator excludes the new verdict so the two numbers stay comparable.
  """

  TOKEN = "SEC-9042"

  def _spec(self, locus):
    return ConflictVLMScenarioSpec.from_dict({
        "series": "conflict_vlm_v1", "scenario_slug": "t", "title": "T",
        "domain": "d", "core_conflict": "c", "conflict_taxonomy": [],
        "modalities": [], "allowed_tool_families": ["file", "asset"],
        "assets": [_img_asset_dict("img_A", token=self.TOKEN, pm={
            "expected_token": self.TOKEN, "evidence_role": "decisive",
            "vision_verified": True, "vision_audit_channel": "image"})],
        "rounds": [{
            "round_id": "r01", "round_type": "resolution",
            "visible_asset_ids": ["img_A"], "required_modalities": [],
            "expected_anchor_ids": ["E1"], "required_tool_families": [],
            "resolved_value": self.TOKEN, "resolved_value_locus": locus,
        }],
        "evals": [], "workspace": {"files": []}, "sessions": [],
    })

  def _task(self, **surfaces):
    return {"data": {"private": {
        "initial_state_overrides": surfaces.get("overrides", {}),
        "background_events": surfaces.get("events", []),
        "mock_subagents": surfaces.get("mocks", []),
    }}}

  def _leaking_task(self):
    return self._task(overrides={"workspace/a.md": f"value {self.TOKEN}"})

  def test_a_leaking_computed_round_gets_its_own_verdict(self):
    # The 43-round population. It used to classify as a clean
    # ``non_asset_locus`` no matter what the workspace said.
    spec = self._spec("computed")
    verdict, leaked = _classify_round_vision_exclusivity(
        spec, spec.rounds[0], self._leaking_task())
    self.assertEqual(verdict, "non_asset_locus_text_solvable")
    self.assertEqual(leaked, ["workspace"])

  def test_a_leaking_file_round_gets_it_too(self):
    # The 13-round population, 92.3% of which leaked.
    spec = self._spec("file:reports/out.md")
    verdict, leaked = _classify_round_vision_exclusivity(
        spec, spec.rounds[0], self._leaking_task())
    self.assertEqual(verdict, "non_asset_locus_text_solvable")
    self.assertEqual(leaked, ["workspace"])

  def test_every_text_surface_is_scanned_for_a_non_asset_locus(self):
    # Not just the workspace: chat and mock text reach the solving agent too,
    # and gating the scan on the locus had hidden all three equally.
    for kw, surface in ((
        {"overrides": {"workspace/a.md": self.TOKEN}}, "workspace"),
        ({"events": [{"type": "session_append", "message": self.TOKEN}]},
         "session"),
        ({"mocks": [{"mock_response": self.TOKEN}]}, "mock")):
      with self.subTest(surface=surface):
        spec = self._spec("computed")
        verdict, leaked = _classify_round_vision_exclusivity(
            spec, spec.rounds[0], self._task(**kw))
        self.assertEqual(verdict, "non_asset_locus_text_solvable")
        self.assertIn(surface, leaked)

  def test_a_clean_non_asset_round_stays_out_of_scope(self):
    # CONTROL. Ungating the scan must not reclassify honest text rounds: a
    # non-asset locus that does not leak is still simply not a vision question.
    spec = self._spec("computed")
    verdict, leaked = _classify_round_vision_exclusivity(
        spec, spec.rounds[0], self._task(overrides={"workspace/a.md": "none"}))
    self.assertEqual(verdict, "non_asset_locus")
    self.assertEqual(leaked, [])

  def test_an_uncompiled_round_is_unmeasured_whatever_its_locus(self):
    # The ordering matters: ``not_compiled`` is now checked BEFORE the locus,
    # so a non-asset round with no task can no longer be reported as a clean
    # ``non_asset_locus`` on the strength of a scan that never ran.
    spec = self._spec("computed")
    self.assertEqual(
        _classify_round_vision_exclusivity(spec, spec.rounds[0], None)[0],
        "not_compiled")

  def _compiled(self, task_spec):
    from clasharness.scenario.spec import ScenarioRound, ScenarioSpec
    return ScenarioSpec(
        scenario_id="s", generator_id="g", scenario_slug="t", title="T",
        rounds=[ScenarioRound(round_id="r01", task_id="t_r01",
                              instruction="i", task_spec=task_spec)])

  def test_the_headline_counts_the_laundered_rounds(self):
    spec = self._spec("computed")
    state = self._compiled(self._leaking_task())
    text = "\n".join(_vision_exclusivity_report(spec, state))
    self.assertIn("TEXT-SOLVABLE (all loci): 1/1 (100.0%)", text)
    self.assertIn("1 of them on a non-asset locus", text)
    # And it must not be quietly absorbed into the asset-locus denominator,
    # which would restore the understatement in a different place.
    self.assertIn("of 0 asset-locus round(s)", text)

  def test_an_unmeasured_round_does_not_dilute_the_headline(self):
    # The round HAS a value; it was simply never compiled, so no surface was
    # scanned. Counting it as "not text-solvable" would understate the rate --
    # the exact failure this line exists to fix -- so it leaves the denominator
    # entirely and is reported on its own line instead.
    text = "\n".join(_vision_exclusivity_report(self._spec("computed"), None))
    self.assertNotIn("TEXT-SOLVABLE (all loci)", text)
    self.assertIn("could not be measured", text)


def _PIPELINE_FIXTURE_LAYERS(tc, layer):
  """Map a layer name to ``TestPipelineRetries``' happy-path fixture.

  Shared by the tests that need a fully successful generation except for the
  one layer they are actually exercising, so each of them only has to spell
  out its own deviation.
  """
  return {
      "conflict_layer0a_screenplay": tc.valid_layer0a,
      "conflict_layer0a_cast": tc.valid_layer0a_cast,
      "conflict_layer0a_triggers": tc.valid_layer0a_triggers,
      "conflict_layer0b_timeline": tc.valid_layer0b,
      "conflict_layer0c_plans": tc.valid_layer0c,
      "conflict_layer0c_assets_plan": tc.valid_layer0c_assets_plan,
      "conflict_layer0c_updates_plan": tc.valid_layer0c_updates_plan,
      "conflict_layer0c_interactions_plan": tc.valid_layer0c_interactions_plan,
      "conflict_layer1_agents": tc.valid_layer1_agents,
      "conflict_layer1_agent_roster": tc.valid_layer1_agent_roster,
      "conflict_layer1_mock_subagents": tc.valid_layer1_mock_subagents,
      "conflict_layer1_interactions": tc.valid_layer1_interactions,
      "conflict_layer1b_files": tc.valid_layer1b_files,
      "conflict_layer1b_sessions": tc.valid_layer1b_sessions,
      "conflict_layer1b_assets": tc.valid_layer1b_assets,
      "conflict_layer0_expand_phase_timeline": tc.valid_phase_timeline,
      "conflict_layer0_expand_phase_decisive": tc.valid_phase_decisive,
      "conflict_layer0_expand_phase_assets": tc.valid_phase_assets,
      "conflict_layer3_round_instruction": tc.valid_round_instruction,
      "conflict_layer3_round_mocks": tc.valid_round_mocks,
      "conflict_layer3_round_updates": tc.valid_narrative_updates,
      "conflict_layer3_round_assets": tc.valid_narrative_assets,
      "conflict_layer3_round_outline": tc.valid_outline,
      "conflict_layer3_round_steps": tc.valid_steps,
      "conflict_layer3_round_commands": tc.valid_round_commands,
      "conflict_layer3_round_checkers": tc.valid_round_checkers,
  }.get(layer, {})


class TestFormatRetryFeedback(unittest.TestCase):
  """Gate 0: the resolution retry loop must feed back EVERY prior failure.

  ``error_feedback = str(e)`` overwrote the previous attempt's error, so the
  model only ever saw the most recent one. Measured consequence: on 67 degraded
  rounds, 56 (83.6%) of third attempts returned the SAME value as the first --
  the model oscillated because nothing told it attempt 1 had already been
  rejected.
  """

  def test_empty_history_renders_nothing(self):
    self.assertEqual(_format_retry_feedback([]), "")

  def test_single_failure_is_reported(self):
    out = _format_retry_feedback(["value not in asset"])
    self.assertIn("value not in asset", out)

  def test_all_prior_failures_are_reported_not_just_the_last(self):
    out = _format_retry_feedback(["first bad", "second bad", "third bad"])
    for expected in ("first bad", "second bad", "third bad"):
      self.assertIn(expected, out)

  def test_attempts_are_numbered_so_the_model_sees_the_count(self):
    out = _format_retry_feedback(["alpha fail", "beta fail"])
    self.assertIn("1", out)
    self.assertIn("2", out)
    # An explicit "do not repeat" directive: the oscillation is the model
    # re-proposing a value it already had rejected.
    self.assertRegex(out.lower(), r"do not repeat|already been rejected")

  def test_duplicate_errors_are_kept_as_loop_signal(self):
    # The same error twice is signal ("you are looping"), not noise to drop.
    out = _format_retry_feedback(["same", "same"])
    self.assertIn("same", out)


class TestResolutionRetryFeedbackAccumulates(unittest.TestCase):
  """Gate 0 wiring: attempt N's prompt carries attempts 1..N-1's errors."""

  def setUp(self):
    TestPipelineRetries.setUp(self)

  def _run_with_failing_resolution(self):
    seen: list[str] = []

    def mock_generate_layer(layer, payload):
      if layer == "conflict_layer0a_screenplay":
        return self.valid_layer0a
      elif layer == "conflict_layer0a_cast":
        return self.valid_layer0a_cast
      elif layer == "conflict_layer0a_triggers":
        return self.valid_layer0a_triggers
      elif layer == "conflict_layer0b_timeline":
        return self.valid_layer0b
      elif layer == "conflict_layer0c_plans":
        return self.valid_layer0c
      elif layer == "conflict_layer0c_assets_plan":
        return self.valid_layer0c_assets_plan
      elif layer == "conflict_layer0c_updates_plan":
        return self.valid_layer0c_updates_plan
      elif layer == "conflict_layer0c_interactions_plan":
        return self.valid_layer0c_interactions_plan
      elif layer == "conflict_layer1_agents":
        return self.valid_layer1_agents
      elif layer == "conflict_layer1_agent_roster":
        return self.valid_layer1_agent_roster
      elif layer == "conflict_layer1_mock_subagents":
        return self.valid_layer1_mock_subagents
      elif layer == "conflict_layer1_interactions":
        return self.valid_layer1_interactions
      elif layer == "conflict_layer1b_files":
        return self.valid_layer1b_files
      elif layer == "conflict_layer1b_sessions":
        return self.valid_layer1b_sessions
      elif layer == "conflict_layer1b_assets":
        return self.valid_layer1b_assets
      elif layer == "conflict_layer0_expand_phase_timeline":
        return self.valid_phase_timeline
      elif layer == "conflict_layer0_expand_phase_assets":
        return self.valid_phase_assets
      elif layer == "conflict_layer3_round_instruction":
        return self.valid_round_instruction
      elif layer == "conflict_layer3_round_mocks":
        return self.valid_round_mocks
      elif layer == "conflict_layer3_round_updates":
        return self.valid_narrative_updates
      elif layer == "conflict_layer3_round_assets":
        return self.valid_narrative_assets
      elif layer == "conflict_layer3_round_resolution":
        if int(payload.get("slot", {}).get("round_index", 0)) == 1:
          seen.append(str(payload.get("slot", {}).get("instruction", "")))
          n = len(seen)
        else:
          n = 1
        # A locus naming an asset that does not exist is a hard, stable error
        # (it is rejected before any value-class or commissioning logic), and
        # each attempt names a DIFFERENT phantom so the accumulated feedback is
        # distinguishable attempt by attempt. The value itself is a clean
        # reading so Gate A never intercepts first.
        return {
            "decisive_anchor": "EVIDENCE:GHOST",
            "resolved_value": f"GHOST-{n}",
            "resolved_value_locus": f"asset:phantom_{n}",
            "decision_outcome": "block the deploy",
        }
      elif layer == "conflict_layer3_round_outline":
        return self.valid_outline
      elif layer == "conflict_layer3_round_steps":
        return self.valid_steps
      elif layer == "conflict_layer3_round_commands":
        return self.valid_round_commands
      elif layer == "conflict_layer3_round_checkers":
        return self.valid_round_checkers
      return {}

    self.mock_client.generate_layer.side_effect = mock_generate_layer
    # Resolution degrades non-fatally, so generation completes.
    self.pipeline.generate(self.seed)
    return seen

  def test_second_attempt_sees_the_first_failure(self):
    seen = self._run_with_failing_resolution()
    self.assertGreaterEqual(len(seen), 2)
    self.assertIn("phantom_1", seen[1])

  def test_third_attempt_sees_BOTH_earlier_failures(self):
    # The regression: previously attempt 3 saw only attempt 2's error, so the
    # model was free to re-propose attempt 1's already-rejected value.
    seen = self._run_with_failing_resolution()
    self.assertGreaterEqual(len(seen), 3)
    self.assertIn("phantom_1", seen[2])
    self.assertIn("phantom_2", seen[2])


# ---------------------------------------------------------------------------
# Gate A is DELETED. The decisive value is not re-shaped after the fact.
#
# Gate A classified every resolver pick as reading / buried / verdict and had
# an LLM rewrite the last two, on the theory that a verdict-shaped value lets
# the round be answered without reading any evidence. Measured on the 523-round
# 3.6-flash corpus -- which was generated BEFORE Gate A existed, so it is a
# clean sample of exactly what Gate A would have rewritten -- that theory fails
# every test available:
#
#   * Reachable from the round's own text without opening an image:
#     reading 1/257 (0.4%), buried 0/97 (0.0%), verdict 2/169 (1.2%). Two cases.
#   * Overlapping ``decision_outcome`` (i.e. restating the ruling): 0.0% in all
#     three classes.
#   * "Answerable without reading" predicts a HIGHER pass rate. Holding the
#     check arm fixed to remove the routing confound, verdict values score
#     LOWER: on ``semantic(D)`` 53.0% against reading's 70.8% (-17.8pp), and on
#     ``semantic(verdict)`` 86.6% against 91.5% (-4.9pp). That second arm is
#     the telling one -- it never quotes the decisive value, only grades
#     whether the agent reached the right ruling -- so it cannot be a
#     string-matching artifact. These rounds are genuinely harder.
#
# Non-reading values are 266 of 523 rounds (50.9%). Rewriting them into
# readings deletes the harder half of the corpus. The concern Gate A was
# reaching for -- the answer being handed to the player in text -- is real and
# is covered at the source by S5's secrecy guard, which forbids the round's
# prose from printing the value at all.
# ---------------------------------------------------------------------------


class TestTheDecisiveValueIsNotReshapedAfterTheFact(unittest.TestCase):
  """Gate A deletion: what the resolver picks is what the round grades on.

  These drive the FREE-PICK path (S1's decisive half degraded), because after
  S5 that is the only path Gate A still ran on -- a bound value was already
  exempt, since rewriting it would swap in a string the asset authors were
  never told to print.
  """

  # Real corpus values Gate A classified as non-reading and would have sent
  # for rewriting. Both are legitimate decisive values.
  VERDICT_SHAPED = "ALL SYSTEMS NOMINAL"
  BURIED_SHAPED = "100% PACKET DROP ON PORT 443"

  def setUp(self):
    TestPipelineRetries.setUp(self)

  def _run(self, resolved_value):
    """Free-pick generation returning ``resolved_value``; records every layer."""
    layers = []

    def mock_generate_layer(layer, payload):
      layers.append(layer)
      if layer == "conflict_layer0_expand_phase_decisive":
        return {"decisive_points": []}
      if layer == "conflict_layer3_round_resolution":
        return {
            "decisive_anchor": "EVIDENCE:METRIC",
            "resolved_value": resolved_value,
            # 'computed' grounds leniently, isolating this from the grounding
            # gate -- which is deliberately KEPT.
            "resolved_value_locus": "computed",
            "decision_outcome": "block the deploy",
        }
      return _PIPELINE_FIXTURE_LAYERS(self, layer)

    self.mock_client.generate_layer.side_effect = mock_generate_layer
    spec = self.pipeline.generate(self.seed)
    values = [str(r.get("resolved_value", "")) for r in spec.get("rounds", [])]
    return layers, values

  def test_a_verdict_shaped_pick_survives_to_the_graded_field(self):
    _, values = self._run(self.VERDICT_SHAPED)
    self.assertIn(self.VERDICT_SHAPED, values)

  def test_a_buried_shaped_pick_survives_to_the_graded_field(self):
    _, values = self._run(self.BURIED_SHAPED)
    self.assertIn(self.BURIED_SHAPED, values)

  def test_no_repair_layer_is_called_for_a_verdict_shaped_pick(self):
    layers, _ = self._run(self.VERDICT_SHAPED)
    self.assertEqual(
        [name for name in layers if name.startswith("repair_decisive_value")],
        [])

  def test_a_reading_shaped_pick_still_survives(self):
    # CONTROL: passed before the deletion too. Pins that removing the gate did
    # not disturb the path that was already clean.
    _, values = self._run("14.2 dB")
    self.assertIn("14.2 dB", values)

  def test_the_classifier_and_its_vocabulary_are_gone(self):
    # Left in place they would be dead code that reads like live policy, and
    # the word lists in particular invite being wired back in.
    import clasharness.scenario.conflict_llm_pipeline as mod
    for name in ("_classify_decisive_value", "_repair_decisive_value_class",
                 "_decisive_value_prose_words", "_DECISIVE_VALUE_UNIT_WORDS",
                 "_DECISIVE_VALUE_VERDICT_WORDS", "_DECISIVE_VALUE_MAX_LEN",
                 "_DECISIVE_VALUE_MAX_PROSE_WORDS"):
      with self.subTest(name=name):
        self.assertFalse(hasattr(mod, name),
                         f"{name} must not survive Gate A's deletion")

  def test_the_repair_layer_contracts_are_gone(self):
    from clasharness.scenario.llm_client import _layer_contract
    for layer in ("repair_decisive_value_tighten",
                  "repair_decisive_value_rederive"):
      with self.subTest(layer=layer):
        self.assertNotIn("instruction", _layer_contract(layer))



# ---------------------------------------------------------------------------
# S1 -- the decisive value is DECIDED before the phase's assets are authored.
# ---------------------------------------------------------------------------


# The diversity guarantee's shared list. Every entry is a VERBATIM decisive
# value from the 523-round corpus, re-verified against the shipped
# ``*_spec.json`` files rather than copied from a note.
#
# It is one list because it has to be asserted at TWO layers, and asserting it
# at one was measured to be worthless. The validator half
# (``test_shapes_the_legacy_regex_killed_are_accepted``) has always passed --
# including on the run where no-digit values collapsed from 33.7% of the corpus
# to 4.9%. Nothing rejected them; the S1 PROMPT had simply been told they were
# wrong, and a validator cannot see an instruction. 2,433 green tests coexisted
# with a 6.9x species collapse for exactly that reason.
#
# So the list is also read by ``TestTheS1PromptBansNoSpeciesOfValue``, which
# asserts the prompt does not name any of these as a wrong choice. Add a value
# here only if it is real and verified; both layers pick it up.
_CORPUS_VALUES_THE_PIPELINE_MUST_ACCEPT = (
    # (a) measurements -- shapes the legacy _is_groundable_token regex rejected
    "182.1 U/L",
    "28.1 U/L",
    "1,420 queries/min",
    "0 KB/s (BASELINE NOMINAL)",
    "3/4 Actioned",
    # (b) codes, identifiers, timestamps, paths
    "SEC-CONTAIN-8842 [VERIFIED]",
    "2024-05-20T14:42:00Z",
    "http://localhost:8080/v1/telemetry",
    "/reports/fault_analysis.md",
    # (c) state and status phrases -- the species the shipped prompt ban
    # deleted. These five are the ones the S1 prompt now cites as valid
    # examples, so if the prompt ever turns on them the contradiction is
    # caught here rather than in the next corpus.
    "BLOCKED (3/4 Ready)",
    "FULLY CONTAINED",
    "PENDING FINAL SIGN-OFF",
    "ALL POST-DEPLOYMENT CHECKS PASSED",
    "transitioned from LOCKED_EXCLUSIVE to UNLOCKED",
    "ALL SYSTEMS NOMINAL",
)


class TestValidateExpandedPhaseDecisive(unittest.TestCase):
  """S1 validator: structural checks on the per-round decisive data point.

  Why S1 exists. The pipeline authored a phase's visual assets first
  (``conflict_layer0_expand_phase_assets``) and only afterwards asked a
  resolver to point at the round's decisive value
  (``conflict_layer3_round_resolution``, the "Step 1.5" call). The dependency
  ran backwards: the value could only ever be whatever the already-frozen
  asset text happened to say. Measured over 590 generated rounds -- 67 (11.4%)
  ended with no value at all and 84 (14.2%) named a value appearing nowhere in
  the scenario. S1 chooses the value from the FINALIZED round timeline, before
  any of the phase's asset text exists, so the asset steps can be told to
  carry it verbatim.

  This docstring used to carry a third defect: "169 (28.6%) named a verdict
  rather than a reading". That was never a defect and the claim is DELETED.
  It is the same 169 rounds the Gate A post-mortem above clears -- reachable
  from the round's own text 1.2% of the time, overlapping ``decision_outcome``
  0.0% of the time, and scoring LOWER (harder) than readings on both check
  arms including the one that never quotes the value. Re-measured directly on
  the 17,804 joined eval observations: no-digit values mean 0.6628 against
  0.6782 for values carrying a digit, negative in all three settings and in 8
  of 10 model configurations. Counting them as a defect is what licensed an
  "EVIDENCE, not a VERDICT" clause into the S1 prompt, which collapsed
  no-digit values from 33.7% of the corpus to 4.9% and took the harder half
  with it. The claim is recorded here as retracted so it cannot be cited again.

  This validator asserts STRUCTURAL facts only -- presence, round coverage,
  single line, length. It deliberately does NOT constrain the SHAPE of the
  value. A shape regex was tried (the legacy ``_is_groundable_token``) and
  measured to accept only 127 of 523 corpus values (24.3%), rejecting every
  reading carrying a unit ("182.1 U/L"), a thousands separator
  ("1,420 queries/min") or a bracketed state ("SEC-CONTAIN-8842 [VERIFIED]").
  Shape is the layer contract's job, not a regex's -- and, as the collapse
  above shows, not the prompt's either.
  """

  def _point(self, round_id="r01", value="182.1 U/L",
             label="ALT on the 06:14 lab panel"):
    return {
        "round_id": round_id,
        "decisive_value": value,
        "decisive_label": label,
        # S6: required of every point. Its own behaviour is covered by
        # TestPhaseDecisiveDeclaresItsOwnCitability; here it is just a
        # well-formed point so the structural checks are what is measured.
        "decisive_value_citable": True,
    }

  def test_missing_key_is_rejected(self):
    with self.assertRaises(ValueError):
      _validate_expanded_phase_decisive({}, ["r01"])

  def test_non_list_is_rejected(self):
    with self.assertRaises(ValueError):
      _validate_expanded_phase_decisive({"decisive_points": {}}, ["r01"])

  def test_empty_list_is_rejected(self):
    with self.assertRaises(ValueError):
      _validate_expanded_phase_decisive({"decisive_points": []}, ["r01"])

  def test_a_round_without_a_point_is_rejected(self):
    # A phase covering r01+r02 that only decides r01 would leave r02 back in
    # the old "resolver searches the finished asset text" regime.
    with self.assertRaises(ValueError):
      _validate_expanded_phase_decisive(
          {"decisive_points": [self._point("r01")]}, ["r01", "r02"])

  def test_blank_value_is_rejected(self):
    for blank in ("", "   ", None):
      with self.subTest(value=blank):
        with self.assertRaises(ValueError):
          _validate_expanded_phase_decisive(
              {"decisive_points": [self._point(value=blank)]}, ["r01"])

  def test_multiline_value_is_rejected(self):
    # A multi-line value cannot be stamped onto a caption verbatim, and a
    # field_contains checker built from it would never match.
    with self.assertRaises(ValueError):
      _validate_expanded_phase_decisive(
          {"decisive_points": [self._point(value="182.1 U/L\nALT")]}, ["r01"])

  def test_overlong_value_is_rejected(self):
    with self.assertRaises(ValueError):
      _validate_expanded_phase_decisive(
          {"decisive_points": [
              self._point(value="A" * (_DECISIVE_POINT_MAX_LEN + 1))]},
          ["r01"])

  def test_a_value_exactly_at_the_cap_is_accepted(self):
    _validate_expanded_phase_decisive(
        {"decisive_points": [self._point(value="A" * _DECISIVE_POINT_MAX_LEN)]},
        ["r01"])

  def test_the_cap_sits_above_the_measured_healthy_maximum(self):
    # Derived, not guessed. Of the 523 corpus values, the 127 that the legacy
    # groundable-token regex accepted top out at 34 characters, and every
    # legitimate "core" reading in the 30-70 band ("410 KB/s", "45.0 A",
    # "SALT_KEY_9942A") is shorter still. The long tail is panel titles and
    # verdict prose ("AUDIT STATUS: COMPLETE - READY FOR CHANGE TICKET
    # CHG-88120 CLOSURE", 66 chars). The cap must stay a runaway backstop, not
    # become a shape filter.
    self.assertGreater(_DECISIVE_POINT_MAX_LEN, 34)

  def test_missing_label_is_rejected(self):
    point = self._point()
    del point["decisive_label"]
    with self.assertRaises(ValueError):
      _validate_expanded_phase_decisive({"decisive_points": [point]}, ["r01"])

  def test_valid_payload_is_accepted(self):
    _validate_expanded_phase_decisive(
        {"decisive_points": [
            self._point("r01", value="182.1 U/L"),
            self._point("r02", value="1,420 queries/min")]},
        ["r01", "r02"])

  def test_two_rounds_of_one_phase_may_not_share_a_value(self):
    # Measured: 213 of 523 corpus rounds (40.7%) reuse a value another round in
    # the same scenario already turned on, and 8 scenarios ran all 10 rounds on
    # a single value. Controlling for whether the value was findable at all,
    # those rounds scored 0.6950 against 0.6601 -- reuse hands the player the
    # answer for free after the first round.
    with self.assertRaises(ValueError):
      _validate_expanded_phase_decisive(
          {"decisive_points": [self._point("r01", value="SEC892"),
                               self._point("r02", value="SEC892")]},
          ["r01", "r02"])

  def test_reuse_is_caught_regardless_of_casing_or_padding(self):
    # Comparison-only normalization. The value is still carried onto the asset
    # exactly as written; this just stops "SEC892" / "sec892 " counting as two
    # different answers.
    with self.assertRaises(ValueError):
      _validate_expanded_phase_decisive(
          {"decisive_points": [self._point("r01", value="SEC892"),
                               self._point("r02", value=" sec892 ")]},
          ["r01", "r02"])

  def test_a_value_an_earlier_phase_already_used_is_rejected(self):
    # Phases are expanded one at a time, so within-phase distinctness alone
    # would still let every phase land on the same value.
    with self.assertRaises(ValueError):
      _validate_expanded_phase_decisive(
          {"decisive_points": [self._point("r03", value="SEC892")]},
          ["r03"],
          already_decided={"sec892"})

  def test_an_unused_value_passes_the_already_decided_check(self):
    _validate_expanded_phase_decisive(
        {"decisive_points": [self._point("r03", value="410 KB/s")]},
        ["r03"],
        already_decided={"sec892"})

  def test_already_decided_defaults_to_empty(self):
    _validate_expanded_phase_decisive(
        {"decisive_points": [self._point("r01")]}, ["r01"])

  def test_shapes_the_legacy_regex_killed_are_accepted(self):
    # The diversity guarantee, VALIDATOR half. The PROMPT half lives in
    # TestTheS1PromptBansNoSpeciesOfValue and reads the same list -- see the
    # note there for why one half alone did not hold.
    for value in _CORPUS_VALUES_THE_PIPELINE_MUST_ACCEPT:
      with self.subTest(value=value):
        _validate_expanded_phase_decisive(
            {"decisive_points": [self._point(value=value)]}, ["r01"])


class TestTheLengthCapAdmitsLongLegitimateValues(unittest.TestCase):
  """S1's cap is a runaway backstop, not a shape filter -- now measured.

  43 of the 523 corpus values exceed the old 48-character cap, and they are two
  different things wearing the same length:

    * long ``buried`` values, mean score 0.5784 against 0.6883 for short ones.
      These are a real defect -- the "value" is the panel's own title, e.g.
      'TREASURY DASHBOARD - EX-8902 RECONCILIATION BREAKDOWN (20231024_083300)'.
    * long ``verdict`` values, mean score 0.7084 against 0.6577 for short ones
      -- HIGHER. These are legitimate, and the old cap threw them away.

  Length does not separate the two: the longest legitimate value measured is 76
  characters and panel titles run from 64 to 93, so any cap that admits the
  first admits most of the second. The cap therefore stops pretending to catch
  the defect and goes back to being a pure runaway backstop. The panel-title
  defect is handled where the value is CHOSEN instead -- the S1 prompt already
  demands "the reading itself, not the panel title and not the sentence around
  it" (see
  ``TestDecisiveValueIsSpelledTheWayAReportPrintsIt.test_prompt_still_forbids_the_panel_title``).
  """

  # Measured over-48 values the old cap rejected and the new one must admit.
  LONG_LEGITIMATE = [
      "Warning: Mutual pending condition detected. Auto-polling cannot clear"
      " cycle.",
      "Next Step: Chat Permit ID to Security Reviewer for Runbook Release.",
      "Platform_Approver -> Security_Approver -> Platform_Approver",
  ]

  def _point(self, value):
    return {
        "round_id": "r01",
        "decisive_value": value,
        "decisive_label": "ALT on the 06:14 lab panel",
        "decisive_value_citable": True,
    }

  def test_the_measured_long_values_are_admitted(self):
    for value in self.LONG_LEGITIMATE:
      with self.subTest(value=value):
        _validate_expanded_phase_decisive(
            {"decisive_points": [self._point(value)]}, ["r01"])

  def test_the_cap_clears_the_longest_measured_legitimate_value(self):
    self.assertGreaterEqual(
        _DECISIVE_POINT_MAX_LEN,
        max(len(v) for v in self.LONG_LEGITIMATE),
        "the cap must not re-reject the values it was raised for")

  def test_a_runaway_is_still_rejected(self):
    # CONTROL: passed before the raise too. The cap is still a backstop -- the
    # longest thing in the whole corpus is 93 characters.
    with self.assertRaises(ValueError):
      _validate_expanded_phase_decisive(
          {"decisive_points": [self._point("A" * 400)]}, ["r01"])


class TestS1RejectsAValueTheBackgroundAlreadyPrinted(unittest.TestCase):
  """S1 sees the text Stage 3 already shipped, and will not pick out of it.

  The binding pipeline told the asset authors to print S1's value verbatim,
  which made the value's collisions matter in a way they had not before. The
  base workspace files and chat sessions are authored in Stage 3, BEFORE this
  stage runs, so by the time S1 picks, that text is frozen -- and a value that
  already appears in it is text-solvable the moment it is chosen: the player
  greps the workspace and never opens the evidence.

  Why the check belongs HERE and not only downstream. The confidentiality gate
  catches the same collision after resolution, but by then the value is bound
  into the assets and the resolver's cheapest escape is to re-label the locus
  rather than change the value. Measured on the resulting corpus: ``computed``
  went 0 -> 43 occurrences, 43 of 43 emitted only AFTER a leak rejection, 0 of
  43 a first choice; with 13 similarly re-labelled ``file`` rounds that
  laundered 56 rounds (11.9%) into a text-solvable state. At S1 the value is
  still soft, so the retry can actually change it. The evidence that the base
  layer is the dominant source: binding-release rate FELL with round index
  (r01 50% -> r06 20%), the signature of a fixed base corpus rather than
  accumulating per-round history.

  The repair the message asks for is deliberately "make it more specific", NOT
  "pick a different kind of value" -- inventing an ID is the cheapest way to
  dodge a clash, and a phase that does that every time has traded its
  status-phrase species away to satisfy a uniqueness check. That is the same
  species collapse this file's diversity guarantee exists to prevent.
  """

  def _point(self, value):
    return {
        "round_id": "r01",
        "decisive_value": value,
        "decisive_label": "ALT on the 06:14 lab panel",
        "decisive_value_citable": True,
    }

  def _run(self, value, **kw):
    _validate_expanded_phase_decisive(
        {"decisive_points": [self._point(value)]}, ["r01"], **kw)

  def test_a_value_already_in_a_base_workspace_file_is_rejected(self):
    with self.assertRaises(ValueError) as cm:
      self._run("BARRIER HOLD ENFORCED",
                workspace_texts={"notes/handoff.md":
                                 "status: BARRIER HOLD ENFORCED since 06:00"})
    msg = str(cm.exception)
    self.assertIn("notes/handoff.md", msg)
    self.assertIn("ALREADY WRITTEN", msg)

  def test_a_value_already_in_a_base_chat_session_is_rejected(self):
    with self.assertRaises(ValueError) as cm:
      self._run("SEC-CONTAIN-8842 [VERIFIED]",
                session_texts=["Avery: ticket SEC-CONTAIN-8842 [VERIFIED] is in"])
    self.assertIn("chat/session", str(cm.exception))

  def test_the_collision_scan_is_fold_insensitive(self):
    # Must use the SAME folding as the downstream grounding/confidentiality
    # gates. If S1 were stricter or looser than they are, a value could pass
    # here and be rejected there -- which is exactly the state that pushed the
    # resolver into re-labelling the locus.
    with self.assertRaises(ValueError):
      self._run("fully contained",
                workspace_texts={"a.md": "Status:   FULLY    CONTAINED"})

  def test_the_repair_asks_for_specificity_not_a_different_species(self):
    # The wording is load-bearing, not decoration: this feedback is what the
    # model reads on retry, and "pick something else" is what produces an
    # invented identifier and a phase with no status phrases left in it.
    for kw in ({"workspace_texts": {"a.md": "state: FULLY CONTAINED"}},
               {"session_texts": ["Avery: it is FULLY CONTAINED now"]}):
      with self.subTest(surface=list(kw)[0]):
        with self.assertRaises(ValueError) as cm:
          self._run("FULLY CONTAINED", **kw)
        msg = str(cm.exception).lower()
        self.assertIn("same kind", msg)
        self.assertIn("more specific", msg)
        self.assertIn("invented identifier", msg)

  def test_a_value_absent_from_the_background_passes(self):
    # CONTROL. A status phrase is not rejected for being a status phrase --
    # only for already being written down.
    self._run("PENDING FINAL SIGN-OFF",
              workspace_texts={"a.md": "awaiting the reviewer"},
              session_texts=["Avery: check the panel when you get a moment"])

  def test_a_merely_overlapping_phrase_passes(self):
    # The scan is a containment test on the WHOLE value, so a background that
    # shares words with it does not trip: 'FULLY CONTAINED' is not written.
    self._run("FULLY CONTAINED",
              workspace_texts={"a.md": "the incident is not yet contained"})

  def test_omitting_the_surfaces_is_backward_compatible(self):
    # Both arguments are optional; every existing caller passing neither must
    # behave exactly as before.
    self._run("FULLY CONTAINED")

  def test_a_trivially_short_value_is_not_scanned(self):
    # <3 non-space characters would match ubiquitous substrings. Same skip the
    # downstream confidentiality gate applies, for the same reason.
    self._run("v2", workspace_texts={"a.md": "we run v2 everywhere"})

  def test_the_shared_corpus_values_survive_a_clean_background(self):
    # The diversity guarantee again, now against the collision check: adding a
    # gate is the classic way a species quietly stops being reachable.
    for value in _CORPUS_VALUES_THE_PIPELINE_MUST_ACCEPT:
      with self.subTest(value=value):
        self._run(value,
                  workspace_texts={"a.md": "nothing relevant here"},
                  session_texts=["Avery: ping me when you have looked"])


class TestPhaseDecisiveDeclaresItsOwnCitability(unittest.TestCase):
  """S6: the step that DECIDES the value also declares how it must be graded.

  Grading used to be routed by a shape regex (the deleted
  ``_is_groundable_token``): a value matching number / version / clock time /
  hex kept an exact ``field_contains``; everything else went to a 3-vote LLM
  judge. Measured over 523 corpus values the regex accepted 127 (24.3%), and
  among its rejections were 42 (8.0%) plain number+unit readings -- '182.1
  U/L', '185 U/L' -- which are perfectly safe to match exactly and were being
  sent to the judge for nothing.

  Shape cannot answer the question anyway. What decides it is whether a
  correct report would print those very characters, and that is a fact about
  the value's meaning which the step choosing the value already holds. So it
  declares it, and nothing re-derives it afterwards.
  """

  def _point(self, round_id="r01", value="182.1 U/L",
             label="ALT on the 06:14 lab panel", citable=True):
    point = {
        "round_id": round_id,
        "decisive_value": value,
        "decisive_label": label,
    }
    if citable is not None:
      point["decisive_value_citable"] = citable
    return point

  def test_missing_declaration_is_rejected(self):
    # Silently defaulting would put the router back, just with a constant.
    with self.assertRaises(ValueError) as ctx:
      _validate_expanded_phase_decisive(
          {"decisive_points": [self._point(citable=None)]}, ["r01"])
    self.assertIn("decisive_value_citable", str(ctx.exception))

  def test_both_declared_values_are_accepted(self):
    for citable in (True, False):
      with self.subTest(citable=citable):
        _validate_expanded_phase_decisive(
            {"decisive_points": [self._point(citable=citable)]}, ["r01"])

  def test_json_string_booleans_are_accepted(self):
    # Tolerating "true"/"false" is parsing, not judgement: a model that spells
    # its boolean as a string has still answered the question, and failing the
    # whole phase over the quoting would degrade S1 for a typo.
    for raw, _expected in (("true", True), ("TRUE", True), ("false", False),
                           (" False ", False)):
      with self.subTest(raw=raw):
        _validate_expanded_phase_decisive(
            {"decisive_points": [self._point(citable=raw)]}, ["r01"])

  def test_a_non_boolean_declaration_is_rejected(self):
    for raw in ("maybe", "", 1, 0, [], {}):
      with self.subTest(raw=raw):
        with self.assertRaises(ValueError):
          _validate_expanded_phase_decisive(
              {"decisive_points": [self._point(citable=raw)]}, ["r01"])

  def test_a_string_boolean_is_normalized_to_a_real_boolean(self):
    # Downstream reads the flag with ``is False`` / ``bool()``, so a surviving
    # "false" string would silently route every such round to the exact check.
    payload = {"decisive_points": [self._point(citable="false")]}
    _validate_expanded_phase_decisive(payload, ["r01"])
    self.assertIs(payload["decisive_points"][0]["decisive_value_citable"],
                  False)


# The complete exact-arm vocabulary of the v2 corpus: the 47 distinct decisive
# values behind its 127 exact-graded rounds. That arm was routed by the deleted
# shape regex, so every entry is a value the pipeline has already agreed is
# safe to match character-for-character -- which makes the set a clean
# FALSE-POSITIVE control for the compound guard, and nothing else. Its limits
# are real and worth stating: the shape regex only ever admitted atomic tokens,
# so the set is 1-2 words wide and tops out at 34 characters, and it therefore
# cannot show that the guard leaves LONG atomic readings alone. That is what
# ``test_a_long_atomic_reading_keeps_its_exact_check`` is for.
_V2_EXACT_ARM_VALUES = (
    'http://localhost:8080/v1/telemetry', 'PASSCODE-TAB3-9842-SECURE',
    'RELOAD-PASSCODE-8821-X9', '2024-05-20T14:42:00Z', '2026-03-30 09:22:00',
    'AUTH_PASS_7749_SYNC', 'REL-2023-ARM04-991', 'SEC-PASSCODE-7739',
    '195.154.122.0/24', '20231024_151230', '20231024_153500', '20240520_143130',
    '20241024_141800', '20260328_023100', '20260328_023200', 'JOB-AI-EXP-8842',
    'PASS-8921-SYNC', 'AUTH-9021-SEC', 'PASS-SEC-8821', 'PRM-2026-0391',
    'SEC-7709-SYNC', 'SEC-PASS-8842', 'SEC-PASS-9942', '/dev/ttyUSB0',
    'CTL-101..150', '849-204-SEC', 'PASS-8842-X', 'PASS-9842-X', 'REC-991204',
    'TIX-990218', '09:27:00', 'SEC-8842', 'LH-2040', '3850ms', '4455ms',
    '4800ms', 'RB-702', 'SEC892', '0.02%', '03:30', '12.0A', '470ms', '12ms',
    '135s', '14ms', '180s', '42ms',
)


class TestAnOverDeclaredCitabilityIsDowngraded(unittest.TestCase):
  """The ``decisive_value_citable`` declaration is guarded, not trusted.

  S6 deleted the shape router in favour of the model declaring how its own
  value must be graded, and nothing re-derived the declaration afterwards. The
  shipped corpus says the declaration alone does not hold: 178 of the 400
  exact-graded rounds (44.5%) turned on a COMPOUND phrase --
  'LOCKED_EXCLUSIVE (Holder: Avery_Fulfillment_Coordinator, TTL: 120 s)',
  '3/3 RESPONDED (SCOPE UNRECONCILED)' -- while their round instruction asked
  only that the agent "record the state shown on the panel".

  That combination is unfair, not merely imprecise. ``effect_checker``
  tokenizes ``messages_sent`` but grades ``files_created.content`` as a raw
  case-insensitive substring, so a report that writes "held by Avery, TTL 120s"
  scores zero on a round it answered correctly. Measured over run observations
  the exact arm passes 66% at 0-15 characters and 0.0% at 45+; across 437
  observations of compound needles it passed 0 times. 170 of 600 rounds
  (28.3%) shipped with a check no correct paraphrase could satisfy.

  C2 calibration cannot see it: it replays the GOLD, and the gold is
  prompt-steered to print the value verbatim, so the check always looks
  satisfiable (1 re-derive and 0 degrades over the whole corpus).

  The guard is a backstop, deliberately narrow. The primary repair is the S1
  prompt, which now defines citability by ATOMICITY and carries the negative
  examples; this only catches the residue.
  """

  def _run(self, value, citable=True, round_id="r01"):
    """Validate one point; return (downgrades, mutated point)."""
    point = {
        "round_id": round_id,
        "decisive_value": value,
        "decisive_label": "the reading on the panel",
        "decisive_value_citable": citable,
    }
    payload = {"decisive_points": [point]}
    downgrades = _validate_expanded_phase_decisive(payload, [round_id])
    return downgrades, payload["decisive_points"][0]

  # --- 1. the over-declaration is caught -----------------------------------

  def test_a_compound_value_declared_citable_is_downgraded(self):
    # One real corpus value per branch of the guard, so no branch can rot
    # untested: a bracketed qualifier (159 of the 178), a six-plus-word phrase
    # (16) and a clause comma (3).
    for value in (
        'LOCKED_EXCLUSIVE (Holder: Avery_Fulfillment_Coordinator, TTL: 120 s)',
        'READ (Scope Gap: WRITE Required)',
        'NODE_GATEWAY_C4 (STATUS: DEAD) -> ROUTE_NODE_E3 (STATUS: HEALTHY)',
        'Temporal anomaly shift: +48.0 hours post-dosage',
        'IRP-882-REQ-04: ALL 4 CONTROL POINTS ACTIONED',
        '115 on-hand, 35 reserved, 80 available',
        'Terminal 2E, Gate K32',
    ):
      with self.subTest(value=value):
        downgrades, point = self._run(value)
        self.assertIs(point["decisive_value_citable"], False)
        self.assertEqual(downgrades, [("r01", value)])

  def test_the_value_itself_is_never_touched(self):
    # The whole point of downgrading instead of rejecting: the compound value
    # is a GOOD value and the phase keeps it. A guard that rewrote or
    # truncated it would be buying fairness with diversity, which is the
    # trade this change exists to avoid.
    value = 'LOCKED_EXCLUSIVE (Holder: Avery_Fulfillment_Coordinator, TTL: 120 s)'
    _, point = self._run(value)
    self.assertEqual(point["decisive_value"], value)

  def test_a_downgrade_does_not_raise(self):
    # A raise here re-runs the whole ``generate_layer`` call, so every OTHER
    # round in the phase re-rolls its value too -- one wrong boolean would
    # churn a phase's worth of species for nothing.
    payload = {"decisive_points": [
        {"round_id": "r01", "decisive_value": 'READ (Scope Gap: WRITE Required)',
         "decisive_label": "scope", "decisive_value_citable": True},
        {"round_id": "r02", "decisive_value": "182.1 U/L",
         "decisive_label": "ALT", "decisive_value_citable": True},
    ]}
    downgrades = _validate_expanded_phase_decisive(payload, ["r01", "r02"])
    self.assertEqual([r for r, _ in downgrades], ["r01"])
    # r02 is untouched: the downgrade is per-point, not per-phase.
    self.assertIs(payload["decisive_points"][1]["decisive_value_citable"], True)

  def test_downgrades_are_returned_so_the_rate_stays_countable(self):
    # Returned rather than swallowed, so the caller can log each one and the
    # over-declaration rate is readable off a run log -- the same way the
    # degrade path is. A silent guard is an unmeasurable one.
    downgrades, _ = self._run('READ (Scope Gap: WRITE Required)', round_id="r07")
    self.assertEqual(downgrades, [("r07", 'READ (Scope Gap: WRITE Required)')])

  def test_a_quoted_boolean_is_coerced_before_it_is_guarded(self):
    # Order matters: the guard reads the normalized bool, so a model that
    # spelled its over-declaration as the string "true" must still be caught.
    # Guarding before coercion would let the whole compound class through on a
    # quoting difference.
    downgrades, point = self._run('READ (Scope Gap: WRITE Required)',
                                  citable="true")
    self.assertIs(point["decisive_value_citable"], False)
    self.assertEqual(len(downgrades), 1)

  def test_an_already_false_declaration_is_not_reported_as_a_downgrade(self):
    # Nothing changed, so nothing is logged: the count must mean
    # "over-declarations caught", not "compound values seen".
    downgrades, point = self._run('READ (Scope Gap: WRITE Required)',
                                  citable=False)
    self.assertEqual(downgrades, [])
    self.assertIs(point["decisive_value_citable"], False)

  # --- 2. difficulty guard: long atomic readings survive --------------------

  def test_a_long_atomic_reading_keeps_its_exact_check(self):
    # THE regression this guard must not become. The deleted shape regex
    # rejected 42 corpus readings (8.0%) that were perfectly safe to match
    # exactly, and length is what would re-create that mistake: every value
    # here is long AND atomic. A length rule would lose all of them; the
    # guard asks whether the value carries anything ALONGSIDE the reading,
    # which none of these do.
    #
    # The 'SHA256: ' entry stays even though the S1 prompt now steers away from
    # emitting a caption inside the value. Prompt and guard point different
    # ways ON PURPOSE: the prompt discourages the shape at the source, the
    # guard stays lenient about it, because a caption is not evidence that the
    # reading underneath is compound. Deleting this entry to "align" them would
    # re-open the false-negative class the shape regex was removed for.
    for value in (
        'https://telemetry-alpha.cluster.internal/v1/sync',   # 48
        'SHA256: f2c9a18d04b3e7125a89601d2e4f5139',           # 40
        'db-prod-backfill-02.internal:5432',                  # 33
        '1,420 queries/min',   # a thousands separator is not a clause comma
        '12,500',
        '182.1 U/L',
        '2026-03-14T08:22:11Z',
        '(REDACTED)',          # wholly bracketed is still one token
    ):
      with self.subTest(value=value):
        downgrades, point = self._run(value)
        self.assertIs(point["decisive_value_citable"], True,
                      f"{value!r} is atomic and must keep its exact check")
        self.assertEqual(downgrades, [])

  # --- 3. false-positive control on a known-good corpus arm -----------------

  def test_no_known_good_exact_value_is_downgraded(self):
    # The previous corpus's entire exact-arm vocabulary. Zero may be flagged:
    # those 127 rounds graded exactly and were not the fairness problem.
    flagged = [v for v in _V2_EXACT_ARM_VALUES
               if _is_compound_decisive_value(v)]
    self.assertEqual(flagged, [], f"false positives on known-good values: {flagged}")

  def test_the_guard_reads_the_declaration_never_the_value(self):
    # Stated as an executable fact so the guard cannot drift into a species
    # filter: no value is rejected, whatever its shape. Every compound value
    # in the suite above still validates and still ships.
    for value in ('LOCKED_EXCLUSIVE (Holder: Avery, TTL: 120 s)',
                  '115 on-hand, 35 reserved, 80 available'):
      with self.subTest(value=value):
        _, point = self._run(value)  # no raise
        self.assertEqual(point["decisive_value"], value)

  # --- 4. the downgrade actually changes how the round is graded ------------

  def test_a_downgraded_round_is_emitted_as_a_semantic_check(self):
    # Composition, not plumbing: the plumbing from declaration to scaffolder is
    # covered by TestDeclarationReachesTheScaffolder. What is asserted here is
    # that the downgraded boolean lands on the arm that can actually pass --
    # the round still grades the same reading, just by meaning rather than by
    # punctuation. Separating the arms over 18,194 run observations, atomic
    # values matched exactly passed 66.1% and compound values graded
    # semantically passed 63.1%, so this move costs no difficulty; it converts
    # a check that passed 0 times in 437 observations into one that can.
    value = 'LOCKED_EXCLUSIVE (Holder: Avery_Fulfillment_Coordinator, TTL: 120 s)'
    _, point = self._run(value)
    scaffolds, _ = _scaffold_round_checkers(
        [{"step_number": 1,
          "command": "file create --path reports/out.md --content \"x\""}],
        ["file", "asset", "message"],
        resolved_value=point["decisive_value"],
        resolved_value_citable=point["decisive_value_citable"],
    )
    checker = scaffolds[0]["checker"]
    self.assertEqual(checker["condition"], "field_semantic_match")
    # The reading is still what is graded -- downgrading the arm must not
    # downgrade what the round is about.
    self.assertIn(value, checker["expected"]["criterion"])

  def test_an_atomic_round_still_reaches_the_exact_check(self):
    # The other half of the same claim: the guard must not quietly empty the
    # exact arm. 'field_contains' is the sharpest discriminator in the suite
    # and 222 of the 400 exact-graded rounds keep it.
    _, point = self._run("182.1 U/L")
    scaffolds, _ = _scaffold_round_checkers(
        [{"step_number": 1,
          "command": "file create --path reports/out.md --content \"x\""}],
        ["file", "asset", "message"],
        resolved_value=point["decisive_value"],
        resolved_value_citable=point["decisive_value_citable"],
    )
    checker = scaffolds[0]["checker"]
    self.assertEqual(checker["condition"], "field_contains")
    self.assertEqual(checker["expected"]["value"], "182.1 U/L")


class TestDecisiveValueIsSpelledTheWayAReportPrintsIt(unittest.TestCase):
  """S6: the decided value must be spelled as a written report would print it.

  Measured on 11,295 aligned round-runs, exact ``field_contains`` on the
  decisive value passed 61.4% while the verdict judge passed 88.0% -- the
  exact check is the sharpest discriminator in the suite and must stay
  required. But 178 of its 1,019 failures were spelling, not substance, and
  they collapse onto 8 distinct values of one shape: the generator wrote
  '3850ms' where every report writes '3,850 ms'. 107 failures came from that
  single value.

  The asset is told to print the decided value verbatim, so an unnatural
  spelling propagates: the panel shows '3850ms', the agent transcribes it as
  '3,850 ms', and a correct answer is scored wrong. Fixing it where the value
  is chosen costs no difficulty -- the player must still find the same
  reading.
  """

  def setUp(self):
    TestPipelineRetries.setUp(self)

  def _decisive_instruction(self):
    seen = []

    def mock_generate_layer(layer, payload):
      if layer == "conflict_layer0_expand_phase_decisive":
        seen.append(str((payload.get("slot") or {}).get("instruction") or ""))
        return self.valid_phase_decisive
      return _PIPELINE_FIXTURE_LAYERS(self, layer)

    self.mock_client.generate_layer.side_effect = mock_generate_layer
    self.pipeline.generate(self.seed)
    self.assertTrue(seen, "the decisive step must have run")
    return seen[0]

  def test_prompt_demands_the_spelling_a_report_would_print(self):
    low = self._decisive_instruction().lower()
    self.assertIn("thousands separator", low)
    self.assertIn("space", low,
                  "the unit spacing is the measured failure mode")
    self.assertTrue(
        "report" in low and "print" in low,
        "the prompt must anchor the spelling to what a report prints")

  def test_prompt_asks_for_the_citability_declaration(self):
    text = self._decisive_instruction()
    # Named in the field list AND in the return template, so it is part of the
    # schema rather than an aside the model can drop.
    self.assertGreaterEqual(text.count("decisive_value_citable"), 2)
    # It must state what the declaration DOES, not just name the field.
    self.assertIn("exact match", text.lower())

  def test_layer_contract_names_the_declaration(self):
    from clasharness.scenario.llm_client import _layer_contract
    contract = _layer_contract("conflict_layer0_expand_phase_decisive")
    self.assertIn("decisive_value_citable", contract["instruction"])

  def test_prompt_defines_citability_by_atomicity_not_length(self):
    # The measured misreading. The definition used to end "a reading stays
    # citable however long it is", which the model generalised from "length
    # does not disqualify" to "nothing disqualifies": 178 of 400 exact-graded
    # rounds shipped a compound phrase declared citable, and compound needles
    # passed 0 of 437 times. The question the declaration answers has to be
    # about SHAPE -- is the whole string one reading -- not about size.
    text = self._decisive_instruction()
    low = text.lower()
    self.assertIn("atomic", low,
                  "citability must be defined by atomicity")
    self.assertIn("alongside", low,
                  "the false case must name what makes a value compound:"
                  " something carried ALONGSIDE the reading")
    # Length must be explicitly neutralised rather than silently dropped, or
    # the model re-derives a length rule and loses the long atomic readings
    # (a 48-character URL, a 40-character digest) the shape router used to lose.
    self.assertIn("length never disqualifies", low)

  def test_prompt_carries_worked_negative_examples(self):
    # An abstract definition did not hold. These are real values off the
    # shipped corpus that were declared citable and should not have been, so
    # the model is shown the exact confusion it made rather than a paraphrase.
    text = self._decisive_instruction()
    self.assertIn("LOCKED_EXCLUSIVE (Holder:", text)
    self.assertGreaterEqual(
        text.count("("), 8,
        "the false case is carried by worked examples, not by the rule alone")

  def test_prompt_forbids_reshaping_a_value_to_win_the_declaration(self):
    # THE diversity guard on this change. The cheapest way for a model to
    # answer "is it citable?" with a confident true is to stop choosing
    # compound values -- which would delete species (c), the status phrases,
    # exactly the way the removed "EVIDENCE, not a VERDICT" clause did (no-digit
    # values fell 33.7% -> 4.9%). The prompt must therefore say, in the same
    # breath, that the declaration is about the value already chosen.
    low = self._decisive_instruction().lower()
    self.assertIn("never constrains which value", low)
    self.assertIn("never simplify", low)
    self.assertTrue(
        "compound value is not a lesser value" in low,
        "the prompt must state that a compound value is not a worse value")

  def test_prompt_states_that_the_two_arms_cost_the_same(self):
    # The model has to believe the honest answer is free, or it will trade the
    # value away for the check. Both measured rates are quoted: 66.1% exact on
    # atomic values against 63.1% semantic on compound ones.
    text = self._decisive_instruction()
    self.assertIn("66.1%", text)
    self.assertIn("63.1%", text)
    self.assertIn("437", text,
                  "the 0-of-437 result is the reason the declaration matters")

  def test_layer_contract_agrees_with_the_prompt(self):
    # The contract is the other half of the same instruction; if it still
    # describes citability the old way the two disagree and the model may
    # follow either.
    from clasharness.scenario.llm_client import _layer_contract
    low = _layer_contract("conflict_layer0_expand_phase_decisive")["instruction"].lower()
    self.assertIn("atomic", low)
    self.assertNotIn("however long it is", low,
                     "the phrasing the model read as 'nothing disqualifies'")

  def test_prompt_still_forbids_the_panel_title(self):
    # CONTROL on the wording, but load-bearing since the length cap was raised
    # from 48 to 80: the cap no longer backstops the measured "the value is the
    # panel's own title" defect (mean 0.5784 against 0.6883), so this
    # instruction is now the only thing preventing it.
    self.assertIn("not the panel title", self._decisive_instruction().lower())

  # --- shape A: the captioned reading ---------------------------------------

  def _true_case_sentence(self, text):
    """The 'true --' clause of the citability definition, on its own."""
    start = text.index("true  -- the WHOLE string")
    end = text.index("false -- the string carries", start)
    return text[start:end]

  def test_prompt_does_not_demonstrate_a_caption_as_an_atomic_reading(self):
    # ROOT CAUSE of shape A, measured. The true-case example list shipped
    # 'SHA256: f2c9a18d04b3e7125a89601d2e4f5139' -- a caption glued to a
    # digest, declared citable BY THE PROMPT ITSELF -- and the corpus came
    # back with 'SHA256: 7f8a91b2...' and 'SALT-VERIFY-KEY: c1cd_sal7_8f92a3'
    # in the exact arm. The model was copying the demonstration.
    #
    # The caption is pure loss: the same asset prints three 'SHA256: ' lines,
    # so the prefix distinguishes nothing, while the bare digest is a
    # substring of BOTH the captioned panel line and a report that writes
    # only the hash. Dropping it is strictly more permissive, never less.
    true_case = self._true_case_sentence(self._decisive_instruction())
    self.assertNotIn(
        "SHA256:", true_case,
        "the atomic examples must not demonstrate a caption glued to a"
        " reading -- that is the shape the corpus copied back")
    self.assertIn(
        "'f2c9a18d04b3e7125a89601d2e4f5139'", true_case,
        "the digest example itself must survive: it is the long-but-atomic"
        " case that stops the model re-deriving a length rule")

  def test_prompt_routes_a_caption_to_the_label_field(self):
    # The positive half. Removing the bad example only stops the model
    # copying it; the rule has to say where a caption goes instead. The slot
    # already exists -- decisive_label is defined as the phrase naming what
    # the value is -- so this is a routing instruction, not a new concept.
    text = self._decisive_instruction()
    low = text.lower()
    self.assertIn("caption is not part of the reading", low)
    caption_rule = text[text.index("A CAPTION IS NOT PART OF THE READING"):]
    caption_rule = caption_rule[:caption_rule.index("HARD REQUIREMENT")]
    self.assertIn(
        "decisive_label", caption_rule,
        "the caption must be given a destination, or the model drops it")
    self.assertIn(
        "decisive_value", caption_rule,
        "the rule must say what is left in the value")
    # Both measured shape-A values are shown, not paraphrased.
    self.assertIn("SHA256:", caption_rule)
    self.assertIn("SALT-VERIFY-KEY:", caption_rule)

  def test_caption_rule_does_not_authorise_shortening_a_value(self):
    # THE diversity guard on the caption rule, and the reason it is worded as
    # a spelling rule. "Cut the front off the value" is one bad generalisation
    # away from "prefer shorter values", which is how the removed "EVIDENCE,
    # not a VERDICT" clause deleted species (c) (no-digit values 33.7% ->
    # 4.9%). A status phrase has no caption to lift -- the whole string is the
    # reading -- and neither does an endpoint that merely contains a colon.
    text = self._decisive_instruction()
    caption_rule = text[text.index("A CAPTION IS NOT PART OF THE READING"):]
    caption_rule = caption_rule[:caption_rule.index("HARD REQUIREMENT")]
    self.assertIn(
        "FULL CONTAINMENT VERIFIED", caption_rule,
        "species (c) must be named as having no caption to strip")
    self.assertIn(
        "db-prod-02.internal:5432", caption_rule,
        "a colon inside an atomic endpoint is not a caption boundary")
    self.assertIn(
        "not the simplification", caption_rule.lower(),
        "the rule must reconcile itself with the NEVER-simplify clause above"
        " it, or it reads as a licence to shorten values")

  # --- shape B: the reading with a trailing qualifier -----------------------

  def test_prompt_ties_an_allowed_compound_value_to_a_false_declaration(self):
    # ROOT CAUSE of shape B. This paragraph is a DIVERSITY guard -- it exists
    # so the model does not fall back on bare numbers that identify nothing --
    # and it must not be deleted. But it taught "carry the qualifier" without
    # ever saying what that does to the declaration, and the corpus came back
    # with '18.2 % connection pool utilization' and '68.5% HTTP 502 Bad
    # Gateway' declared citable. The value was the right choice; the
    # declaration was wrong. So the allowance stays and gains its consequence.
    text = self._decisive_instruction()
    para = text[text.index("COMPOUND VALUES ARE ALLOWED"):]
    para = para[:para.index("\n")]
    self.assertIn(
        "100% packet drop on port 443", para,
        "the allowance and its example are the diversity guard and stay")
    self.assertIn(
        "false", para.lower(),
        "an allowed compound value must be told what to declare, in the same"
        " breath -- the missing link is what produced shape B")

  def test_the_compound_link_does_not_swallow_a_unit(self):
    # FALSE-POSITIVE CONTROL on the fix above. "A trailing qualifier makes it
    # false" is one careless reading away from "a unit makes it false", which
    # would push '1,420 queries/min' and '182.1 U/L' out of the exact arm and
    # shrink it for no fairness gain. The paragraph must draw that line
    # itself, next to the rule it qualifies.
    text = self._decisive_instruction()
    para = text[text.index("COMPOUND VALUES ARE ALLOWED"):]
    para = para[:para.index("\n")]
    self.assertIn(
        "1,420 queries/min", para,
        "the paragraph must protect the unit case where it makes the"
        " trailing-qualifier rule, or the rule over-reaches")
    self.assertIn("atomic", para.lower())


class TestTheS1PromptBansNoSpeciesOfValue(unittest.TestCase):
  """The diversity guarantee, PROMPT half -- the half that was missing.

  What happened. An "It must be EVIDENCE, not a VERDICT" clause was added to
  the S1 instruction, naming 'APPROVED', 'BLOCKED (3/4 Ready)' and 'AUDIT
  VERDICT: COMPLETE' as wrong choices. The next corpus came back with no-digit
  values down from 33.7% (176/523) to 4.9% (27/548) and the specific
  'APPROVED'-shaped family down from 21.6% to 1.6% -- a 6.9x collapse of one
  of the three species. It also pushed the decisive value through the exact
  ``field_contains`` route 24.3% -> 84.5% of the time, because a clause that
  demands readings and codes is a clause that demands citable strings.

  Why nothing caught it. The whole suite was green, including a test literally
  commented "The diversity guarantee" that asserts 'BLOCKED (3/4 Ready)' is
  accepted -- the exact string the clause called wrong. Both were true at once:
  the VALIDATOR accepted the value, and the PROMPT ensured the model never
  proposed one. The guarantee was asserted at the layer that does not decide.
  No test can execute an instruction, so the only available check is on the
  instruction TEXT, and there was none.

  The ban was also substantively wrong, not merely overreaching. It restated
  Gate A's theory, which was deleted after measurement: verdict-shaped values
  overlap ``decision_outcome`` 0.0% of the time and score LOWER -- 0.6628
  against 0.6782 across 17,804 graded observations, negative in all three
  settings and 8 of 10 model configurations. The clause deleted the harder
  species. The failure it claimed to prevent is blocked structurally
  downstream by ``_validate_round_decision_outcome``.
  """

  # Verdicts a sentence can pass on an example value. Deliberately NOT bare
  # "never"/"not": the prompt legitimately says "'3,850 ms', never '3850ms'",
  # which corrects the SPELLING of a value inside a species and is a measured
  # fix (107 otherwise-correct answers failed on that one string). What must
  # never appear is a sentence that rules a value out as a KIND.
  WRONGNESS = (
      "are wrong", "is wrong", "are the wrong", "is the wrong",
      "are conclusions", "never use", "do not use", "must not be used",
      "not a verdict", "is unacceptable", "are unacceptable",
  )

  def setUp(self):
    TestPipelineRetries.setUp(self)

  def _decisive_instruction(self):
    return TestDecisiveValueIsSpelledTheWayAReportPrintsIt._decisive_instruction(
        self)

  def test_no_corpus_value_is_named_as_a_wrong_choice(self):
    # The general assertion, and the one that would have failed on the shipped
    # ban: 'BLOCKED (3/4 Ready)' sat on a line ending "are wrong here".
    text = self._decisive_instruction()
    for line in text.splitlines():
      low = line.lower()
      hit = next((m for m in self.WRONGNESS if m in low), None)
      if hit is None:
        continue
      for value in _CORPUS_VALUES_THE_PIPELINE_MUST_ACCEPT:
        with self.subTest(value=value, marker=hit):
          self.assertNotIn(
              value.lower(), low,
              f"the S1 prompt rules out the real corpus value {value!r} "
              f"({hit!r} on the same line). A prompt that names a value wrong "
              "deletes its whole species from the next corpus -- measured "
              "33.7% -> 4.9% for no-digit values.")

  def test_the_evidence_not_verdict_clause_stays_deleted(self):
    # Regression pin on the exact shipped wording, so a revert is unambiguous
    # rather than something the general test above has to infer.
    low = self._decisive_instruction().lower()
    for banned in ("not a verdict", "evidence, not", "are conclusions"):
      with self.subTest(phrase=banned):
        self.assertNotIn(banned, low)

  def test_all_three_species_are_offered_as_equally_valid(self):
    # Positive requirement, not just absence of a ban. "Vary the kind" alone
    # was already in the prompt while the ban shipped, and did not hold.
    low = self._decisive_instruction().lower()
    self.assertIn("182.1 u/l", low, "species (a) measurements")
    self.assertIn("sec-contain-8842", low, "species (b) codes/identifiers")
    self.assertIn("fully contained", low, "species (c) state/status phrases")

  def test_the_status_phrase_species_is_marked_not_lesser(self):
    # The species that collapsed is the one a model will drop first if it is
    # merely listed, so the prompt has to say it is not a fallback AND say why
    # (it measured harder, not easier).
    low = self._decisive_instruction().lower()
    self.assertIn("not a lesser choice", low)
    self.assertIn("harder", low)

  def test_the_prompt_states_a_digit_only_phase_is_worse(self):
    # The collapse was gradual per-phase, not a single refusal: every round
    # quietly picking a number is what 4.9% looks like from the inside.
    low = self._decisive_instruction().lower()
    self.assertTrue(
        "every value carries a digit" in low,
        "the prompt must name the all-numeric phase as the failure mode")


class TestCheckerRoutingFollowsTheDeclaration(unittest.TestCase):
  """S6: the emitted checker reads the declaration; nothing re-derives it."""

  ALLOWED = ["file", "asset", "message"]

  def _write(self, resolved, **kw):
    scaffolds, _ = _scaffold_round_checkers(
        [{"step_number": 1,
          "command": "file create --path reports/out.md --content \"x\""}],
        self.ALLOWED,
        resolved_value=resolved,
        **kw,
    )
    return scaffolds[0]["checker"]

  def test_declared_citable_unit_reading_gets_exact_contains(self):
    # The exact case the deleted regex got wrong: anchored, it rejected any
    # unit suffix, so 42 corpus readings like this one lost their exact check.
    chk = self._write("182.1 U/L", resolved_value_citable=True)
    self.assertEqual(chk["condition"], "field_contains")
    self.assertEqual(chk["expected"]["value"], "182.1 U/L")

  def test_declared_non_citable_number_gets_semantic(self):
    # Shape no longer decides: a bare percentage the regex would have accepted
    # is graded semantically when the deciding step says it is paraphrasable.
    chk = self._write("42%", resolved_value_citable=False)
    self.assertEqual(chk["condition"], "field_semantic_match")
    self.assertIn("42%", chk["expected"]["criterion"])

  def test_an_absent_declaration_defaults_to_semantic(self):
    # The degraded path (S1 produced nothing, or the resolver picked freely)
    # carries no declaration. Semantic is the safe default and costs no
    # measured difficulty: across the corpus the semantic arm passed 59.7%
    # against the exact arm's 61.4%. Uses a bare percentage -- a shape the
    # deleted regex accepted -- so the absence of the declaration, not the
    # value's shape, is what is being measured.
    chk = self._write("42%")
    self.assertEqual(chk["condition"], "field_semantic_match")

  def test_the_shape_regex_is_gone(self):
    import clasharness.scenario.conflict_llm_pipeline as mod
    for name in ("_is_groundable_token", "_GROUNDABLE_TOKEN_RE"):
      with self.subTest(name=name):
        self.assertFalse(
            hasattr(mod, name),
            f"{name} must not survive: a regex judging the value is exactly"
            " the heuristic S6 removes")


class TestDeclarationReachesTheScaffolder(unittest.TestCase):
  """S6 wiring: the declaration follows the value, and only that value."""

  def setUp(self):
    TestPipelineRetries.setUp(self)

  def _citable_kwargs(self, *, decide=True, bound_locus="computed"):
    """Return the ``resolved_value_citable`` seen per scaffolder call."""
    import clasharness.scenario.conflict_llm_pipeline as mod
    seen = []
    real = mod._scaffold_round_checkers

    def spy(*a, **kw):
      seen.append(kw.get("resolved_value_citable"))
      return real(*a, **kw)

    def mock_generate_layer(layer, payload):
      if layer == "conflict_layer0_expand_phase_decisive":
        return self.valid_phase_decisive if decide else {"decisive_points": []}
      if layer == "conflict_layer3_round_resolution":
        slot = payload.get("slot", {})
        bound = bool(str(slot.get("decided_value") or "").strip())
        return {
            "decisive_anchor": "EVIDENCE:RACK_C2",
            "resolved_value": "default_upstream",
            "resolved_value_locus": bound_locus if bound else "computed",
            "decision_outcome": "block the deploy",
        }
      return _PIPELINE_FIXTURE_LAYERS(self, layer)

    self.mock_client.generate_layer.side_effect = mock_generate_layer
    with patch.object(mod, "_scaffold_round_checkers", spy):
      self.pipeline.generate(self.seed)
    self.assertTrue(seen, "the scaffolder must have run")
    return seen

  def test_a_bound_value_carries_its_declaration(self):
    self.assertTrue(any(self._citable_kwargs()))

  def test_a_degraded_phase_declares_nothing(self):
    # S1 produced no points, so there is no declaration to inherit. Asserted
    # as an explicit False rather than merely falsy, so the keyword going
    # missing entirely cannot pass this vacuously.
    seen = self._citable_kwargs(decide=False)
    self.assertTrue(seen and all(c is False for c in seen), seen)

  def test_a_released_binding_does_not_inherit_the_declaration(self):
    # The resolver broke free of the decided value and picked its own, so the
    # declaration describes a different string and must not be applied to it.
    seen = self._citable_kwargs(bound_locus="asset:phantom_panel")
    self.assertTrue(seen and all(c is False for c in seen), seen)


class TestPhaseDecisivePointsPrecedeAssets(unittest.TestCase):
  """S1 wiring: the decision lands between the two existing phase halves.

  Order must be timeline -> decisive -> assets. A decisive call placed after
  the assets call would be the old backwards dependency wearing a new name.
  """

  def setUp(self):
    TestPipelineRetries.setUp(self)

  def _run(self, *, fail_decisive=False):
    order: list[str] = []
    payloads: dict[str, Any] = {}

    def mock_generate_layer(layer, payload):
      order.append(layer)
      payloads.setdefault(layer, copy.deepcopy(payload))
      if layer == "conflict_layer0_expand_phase_decisive" and fail_decisive:
        return {}
      return _PIPELINE_FIXTURE_LAYERS(self, layer)

    self.mock_client.generate_layer.side_effect = mock_generate_layer
    spec = self.pipeline.generate(self.seed)
    return order, payloads, spec

  def test_decisive_runs_after_the_timeline_and_before_the_assets(self):
    order, _, _ = self._run()
    self.assertIn("conflict_layer0_expand_phase_decisive", order)
    timeline_at = order.index("conflict_layer0_expand_phase_timeline")
    decisive_at = order.index("conflict_layer0_expand_phase_decisive")
    assets_at = order.index("conflict_layer0_expand_phase_assets")
    self.assertLess(timeline_at, decisive_at,
                    "the decision must see the FINALIZED round timeline")
    self.assertLess(
        decisive_at, assets_at,
        "assets must be authored already knowing the value they carry")

  def test_the_decisive_call_receives_the_finalized_timeline(self):
    _, payloads, _ = self._run()
    payload = payloads["conflict_layer0_expand_phase_decisive"]
    round_ids = {r.get("round_id") for r in payload.get("phase_timeline", [])}
    self.assertEqual(round_ids, {"r01", "r02"})

  def test_the_assets_call_is_told_which_value_to_carry(self):
    _, payloads, _ = self._run()
    blob = str(payloads["conflict_layer0_expand_phase_assets"])
    self.assertIn("182.1 U/L", blob)
    self.assertIn("1,420 queries/min", blob)

  def test_the_decided_value_rides_along_on_the_round_outline(self):
    _, payloads, _ = self._run()
    # The per-round steps read their outline out of the expanded timeline, so
    # the value has to travel on it rather than live in a side table.
    blob = str(payloads.get("conflict_layer3_round_instruction", {}))
    self.assertIn("182.1 U/L", blob)

  def _run_two_phases(self):
    """Drive a run whose rounds are split across two separately-expanded phases."""
    self.valid_layer0b = {
        "phases": [
            {"phase_id": "p01", "start_round": 1, "end_round": 1,
             "theme": "investigation", "timeline_screenplay": []},
            {"phase_id": "p02", "start_round": 2, "end_round": 2,
             "theme": "decision", "timeline_screenplay": []},
        ]
    }
    per_phase_timeline = {
        "p01": {"timeline": [self.valid_phase_expand["timeline"][0]]},
        "p02": {"timeline": [self.valid_phase_expand["timeline"][1]]},
    }
    per_phase_decisive = {
        "p01": {"decisive_points": [self.valid_phase_decisive["decisive_points"][0]]},
        "p02": {"decisive_points": [self.valid_phase_decisive["decisive_points"][1]]},
    }
    payloads: dict[str, list[Any]] = collections.defaultdict(list)

    def mock_generate_layer(layer, payload):
      payloads[layer].append(copy.deepcopy(payload))
      phase = payload.get("current_phase_id")
      if layer == "conflict_layer0_expand_phase_timeline":
        return copy.deepcopy(per_phase_timeline[phase])
      if layer == "conflict_layer0_expand_phase_decisive":
        return copy.deepcopy(per_phase_decisive[phase])
      return _PIPELINE_FIXTURE_LAYERS(self, layer)

    self.mock_client.generate_layer.side_effect = mock_generate_layer
    self.pipeline.generate(self.seed)
    return payloads

  def test_both_phases_are_expanded_separately(self):
    payloads = self._run_two_phases()
    phases = [p.get("current_phase_id")
              for p in payloads["conflict_layer0_expand_phase_decisive"]]
    self.assertEqual(phases, ["p01", "p02"])

  def test_a_later_phase_is_told_which_values_are_already_taken(self):
    # Without this, phase 2 has no way to know phase 1 already used the value,
    # which is how 8 corpus scenarios ended up running all 10 rounds on one.
    payloads = self._run_two_phases()
    second = payloads["conflict_layer0_expand_phase_decisive"][1]
    self.assertIn("182.1 U/L", str(second.get("previously_decided_values")))

  def test_the_first_phase_has_nothing_taken_yet(self):
    payloads = self._run_two_phases()
    first = payloads["conflict_layer0_expand_phase_decisive"][0]
    self.assertEqual(list(first.get("previously_decided_values") or []), [])

  def test_a_failed_decisive_half_does_not_lose_the_timeline(self):
    # Same resilience contract the assets half already carries: one failed
    # half degrades that half only, it does not discard the scenario.
    order, _, spec = self._run(fail_decisive=True)
    self.assertIn("conflict_layer0_expand_phase_assets", order)
    self.assertTrue(spec.get("rounds"))



class TestPhaseAssetsAreToldWhatToCarry(unittest.TestCase):
  """S2: the phase asset step authors evidence AROUND an already-decided value.

  S1 decides the value; this step is the first one that can actually print it.
  The contract has to do two jobs at once, and the second is easy to lose:

  1. Carry the value VERBATIM, so the round has an exact anchor at all. 82 of
     523 corpus rounds named a value that appeared in no asset's caption,
     ocr_text or description, and those rounds passed their decisive checks
     2.8% of the time against 53.1% when the value was findable.

  2. Carry it on EXACTLY ONE surface, buried in a realistically dense readout.
     This is the difficulty guard. Measured over 9,267 graded findable rounds,
     score rises monotonically with how many assets repeat the value --
     1 carrier 0.6336, 2-4 carriers 0.7137, 5-9 carriers 0.7424. Stamping the
     value everywhere is the easy failure mode of "make sure it appears", and
     it would hand back ~0.08 of difficulty.
  """

  def setUp(self):
    TestPipelineRetries.setUp(self)

  def _assets_instruction(self, *, fail_decisive=False):
    payloads: dict[str, Any] = {}

    def mock_generate_layer(layer, payload):
      payloads.setdefault(layer, copy.deepcopy(payload))
      if layer == "conflict_layer0_expand_phase_decisive" and fail_decisive:
        return {}
      return _PIPELINE_FIXTURE_LAYERS(self, layer)

    self.mock_client.generate_layer.side_effect = mock_generate_layer
    self.pipeline.generate(self.seed)
    return str(payloads["conflict_layer0_expand_phase_assets"]
               .get("slot", {}).get("instruction", ""))

  def test_the_instruction_names_each_round_value_and_where_it_is_read(self):
    text = self._assets_instruction()
    self.assertIn("182.1 U/L", text)
    self.assertIn("1,420 queries/min", text)
    self.assertIn("ALT on the 06:14 lab panel", text)

  def test_the_instruction_demands_the_value_verbatim(self):
    text = self._assets_instruction()
    self.assertIn("VERBATIM", text.upper())

  def test_the_instruction_discourages_gratuitous_extra_carriers(self):
    # The difficulty guard. "Make sure the value appears" is most safely
    # satisfied by printing it on every asset, and that is measurably the
    # easiest configuration: within every stratum that can be controlled for,
    # moving from one carrier to two-to-four costs 0.05-0.12 of score
    # (unique-value rounds 0.6314 -> 0.7331; unique and 15+ characters
    # 0.6310 -> 0.7545). It is discouraged rather than capped, because a hard
    # "exactly one" would forbid two sources agreeing against a third -- a
    # reconciliation shape the decisive half explicitly asks for.
    text = self._assets_instruction()
    low = text.lower()
    # Must be specific to the carrier. The step already carries an unrelated
    # de-duplication rule worded "EXACTLY ONE id per snapshot", so matching a
    # bare "exactly one" would pass without any carrier guidance existing.
    self.assertIn("one primary carrier", low)
    self.assertIn("never as insurance", low)

  def test_the_instruction_states_the_measured_density_floor(self):
    # Carrier surfaces in the corpus run a median 857 characters of
    # caption+ocr_text carrying a median 17 numeric readings, and 97.4% carry
    # at least 5. The decisive value occupies a median 2.1% of that text. The
    # floor keeps the value a needle rather than the whole haystack.
    text = self._assets_instruction()
    self.assertRegex(text, r"at least 8 other")

  def test_the_instruction_forbids_spotlighting_the_value(self):
    text = self._assets_instruction()
    low = text.lower()
    self.assertTrue(
        "do not highlight" in low or "must not be highlighted" in low,
        "the carrier must not annotate the value as the answer")

  def test_no_decisive_block_is_emitted_when_the_decisive_half_failed(self):
    # Degradation must not leave a dangling header with nothing under it,
    # which would read as an instruction to carry a value that was never
    # chosen.
    text = self._assets_instruction(fail_decisive=True)
    self.assertNotIn("DECISIVE VALUES DECIDED FOR THIS PHASE", text)

  def test_the_decisive_block_is_present_on_the_happy_path(self):
    text = self._assets_instruction()
    self.assertIn("DECISIVE VALUES DECIDED FOR THIS PHASE", text)


# ---------------------------------------------------------------------------
# S5 -- the resolver BINDS to the value S1 decided, it does not re-choose one.
# ---------------------------------------------------------------------------


class TestResolverBindsToTheDecidedValue(unittest.TestCase):
  """S5: the graded ``resolved_value`` is the one the phase layer designed.

  S1 decides a value per round before any asset text exists and S2 orders the
  asset authors to print it verbatim. Without S5 none of that reaches the
  graded artifact: the resolver was still free to pick ANY string it could find
  in the finished asset text, so a round could be designed around
  '1,420 queries/min' and end up graded on a decorative '12'. Every S1
  guarantee -- reconciliation-required, evidence-not-verdict, and above all
  ONE DISTINCT VALUE PER ROUND -- only binds if the resolver is held to it.
  The last of those is worth 213 of 523 corpus rounds (40.7%) that shared their
  decisive value with another round of the same scenario, 8 scenarios of which
  ran all 10 rounds on a single value.

  The grounding gate (``_validate_round_resolution``) is deliberately KEPT: it
  is what guarantees the value is really readable off the named artifact.
  Binding changes who chooses the value, not whether it has to be findable.
  """

  def setUp(self):
    TestPipelineRetries.setUp(self)

  def _run(self, *, resolver_value="default_upstream", decide=True,
           bound_locus="computed", free_locus="computed"):
    """Drive one generation, recording every resolution slot.

    ``bound_locus`` is what the mock resolver answers while it is being held to
    a decided value; ``free_locus`` is what it answers once binding is
    released. A locus of ``computed`` grounds leniently, which isolates these
    tests from the grounding gate; naming a phantom asset fails it hard.
    """
    slots = []
    layers = []

    def mock_generate_layer(layer, payload):
      layers.append(layer)
      if layer == "conflict_layer0_expand_phase_decisive":
        # ``decide=False`` degrades the S1 half, which is how a real scenario
        # falls back to the pre-inversion behaviour.
        return self.valid_phase_decisive if decide else {"decisive_points": []}
      if layer == "conflict_layer3_round_resolution":
        slot = payload.get("slot", {})
        slots.append(slot)
        bound = bool(str(slot.get("decided_value") or "").strip())
        return {
            "decisive_anchor": "EVIDENCE:RACK_C2",
            "resolved_value": resolver_value,
            "resolved_value_locus": bound_locus if bound else free_locus,
            "decision_outcome": "block the deploy",
        }
      return _PIPELINE_FIXTURE_LAYERS(self, layer)

    self.mock_client.generate_layer.side_effect = mock_generate_layer
    spec = self.pipeline.generate(self.seed)
    return slots, layers, spec

  def _values(self, spec):
    return {
        str(r.get("round_id")): str(r.get("resolved_value", ""))
        for r in spec.get("rounds", [])
    }

  def test_resolver_is_told_the_value_is_already_decided(self):
    slots, _, _ = self._run()
    r01 = [s for s in slots if int(s.get("round_index", 0)) == 1]
    self.assertTrue(r01, "round 1 must have reached the resolver")
    slot = r01[0]
    # Structured, not only prose: downstream code reads the field.
    self.assertEqual(slot.get("decided_value"), "182.1 U/L")
    self.assertEqual(slot.get("decided_label"),
                     "ALT on the 06:14 lab panel")
    text = str(slot.get("instruction", ""))
    self.assertIn("182.1 U/L", text)
    low = text.lower()
    self.assertIn("already", low)
    self.assertIn("verbatim", low)

  def test_the_decided_value_wins_over_the_resolvers_own_pick(self):
    # The resolver answers with something else entirely; the decided value is
    # what the round is graded on.
    _, _, spec = self._run(resolver_value="some other reading")
    values = self._values(spec)
    self.assertEqual(values.get("r01"), "182.1 U/L")
    self.assertNotIn("some other reading", values.values())

  def test_each_round_grades_on_its_own_decided_value(self):
    # The point of S1's cross-round de-duplication: it has to survive all the
    # way to the graded field, not just to the asset brief.
    _, _, spec = self._run(resolver_value="default_upstream")
    values = self._values(spec)
    self.assertEqual(values.get("r01"), "182.1 U/L")
    self.assertEqual(values.get("r02"), "1,420 queries/min")

  def test_a_verdict_shaped_pick_cannot_displace_the_decided_value(self):
    # CONTROL: passed before Gate A was deleted too, because Gate A already
    # exempted bound values. It is kept -- and renamed off the dead gate --
    # because the BINDING is now the only thing holding this line. A rewrite
    # here would swap in a string the asset authors were never told to print,
    # guaranteeing the value is readable nowhere: the very failure the
    # inversion exists to remove.
    _, layers, spec = self._run(resolver_value="ALL SYSTEMS NOMINAL")
    self.assertEqual(
        [n for n in layers if n.startswith("repair_decisive_value")], [])
    self.assertEqual(self._values(spec).get("r01"), "182.1 U/L")

  def test_resolver_picks_freely_when_no_value_was_decided(self):
    slots, _, spec = self._run(decide=False, resolver_value="default_upstream")
    self.assertTrue(slots)
    for slot in slots:
      self.assertFalse(str(slot.get("decided_value") or "").strip())
      self.assertNotIn("already been decided",
                       str(slot.get("instruction", "")).lower())
    self.assertEqual(self._values(spec).get("r01"), "default_upstream")

  def test_binding_is_released_after_the_bound_attempts_all_fail(self):
    # A decided value the asset author never printed cannot be rescued by
    # retrying the resolver -- the value is fixed and the evidence is fixed.
    # Rather than degrade the round to no decisive check at all, spend one
    # last pass letting the resolver pick a grounded value, which is exactly
    # the pre-inversion behaviour.
    slots, _, spec = self._run(
        resolver_value="default_upstream",
        bound_locus="asset:phantom_never_authored",
        free_locus="computed",
    )
    r01 = [s for s in slots if int(s.get("round_index", 0)) == 1]
    bound = [s for s in r01 if str(s.get("decided_value") or "").strip()]
    free = [s for s in r01 if not str(s.get("decided_value") or "").strip()]
    self.assertGreater(len(bound), 1, "binding must be retried before release")
    self.assertEqual(len(free), 1, "exactly one free pass after release")
    self.assertEqual(self._values(spec).get("r01"), "default_upstream")


class TestNormDecisive(unittest.TestCase):
  """The comparison used to decide whether two rounds grade on the same value.

  Deliberately weak. It has to catch the giveaway -- the same reading written
  with different case or spacing -- without merging readings that genuinely
  differ, because a false merge rejects a legitimately distinct round and the
  final-attempt concession then silently keeps it anyway.
  """

  def test_case_and_spacing_are_not_part_of_the_reading(self):
    same = [
        "telemetry-pass-8899",
        "TELEMETRY-PASS-8899",
        "  Telemetry-Pass-8899  ",
        "telemetry-pass-8899\n",
    ]
    self.assertEqual(len({_norm_decisive(v) for v in same}), 1)

  def test_internal_whitespace_is_collapsed_not_deleted(self):
    self.assertEqual(_norm_decisive("1,420  queries/min"), "1,420 queries/min")
    # Collapsed, NOT stripped: '28.6A' is a different string from '28.6 A' and
    # nothing here should decide they are the same reading.
    self.assertNotEqual(_norm_decisive("28.6 A"), _norm_decisive("28.6A"))

  def test_values_that_merely_look_alike_stay_distinct(self):
    # The failure mode a looser comparison would introduce: same number,
    # different unit, genuinely different reading.
    for a, b in [
        ("28.6 A", "28.6 s"),
        ("182.1 U/L", "182.1 mg/L"),
        ("5,420 ms", "5,420 s"),
        ("0 of 4 control points contained", "1 of 4 control points contained"),
    ]:
      self.assertNotEqual(_norm_decisive(a), _norm_decisive(b), f"{a} vs {b}")

  def test_empty_and_none_normalise_to_empty(self):
    for v in ("", "   ", None):
      self.assertEqual(_norm_decisive(v), "")


class TestTheFreePickStaysDistinctFromOtherRounds(unittest.TestCase):
  """The cross-round distinctness guarantee must survive the binding release.

  S1 holds its DECIDED values distinct, but that is not where the corpus loses
  the property. When a decided value cannot be grounded the binding is released
  and the resolver picks freely -- and until this gate the free pick was
  unconstrained, so it could land on a value another round of the same scenario
  was already graded on. A player who solved that round has the answer for free.

  MEASURED on v6 (60 scenarios, 600 rounds): 43 rounds shared their graded value
  with another round of the same scenario, against 18/600 on v5. Free picks are
  25.5% of all rounds (153/600) but 55.8% of the sharing ones (24/43) -- a 2.2x
  enrichment that identifies the release as the leak. Worst case,
  expense_reimbursement_conflict_v01 ran 5 of its 10 rounds on
  'telemetry-pass-8899', 4 of them free picks.
  """

  def setUp(self):
    TestPipelineRetries.setUp(self)

  def _run(self, *, free_values, bound_locus="asset:phantom_never_authored"):
    """Drive one generation, forcing r01 onto the free path.

    ``bound_locus`` names an asset that does not exist, so every BOUND attempt
    fails the grounding gate and the binding is released -- which is the only
    way to reach the free pick this class is about. ``free_values`` is consumed
    one entry per free attempt, so a test can make the first free pick collide
    and the second one not.
    """
    slots = []
    pending = list(free_values)

    def mock_generate_layer(layer, payload):
      if layer == "conflict_layer0_expand_phase_decisive":
        return self.valid_phase_decisive
      if layer == "conflict_layer3_round_resolution":
        slot = payload.get("slot", {})
        slots.append(slot)
        bound = bool(str(slot.get("decided_value") or "").strip())
        if bound:
          return {
              "decisive_anchor": "EVIDENCE:RACK_C2",
              "resolved_value": "182.1 U/L",
              "resolved_value_locus": bound_locus,
              "decision_outcome": "block the deploy",
          }
        value = pending.pop(0) if pending else "default_upstream"
        return {
            "decisive_anchor": "EVIDENCE:RACK_C2",
            "resolved_value": value,
            "resolved_value_locus": "computed",
            "decision_outcome": "block the deploy",
        }
      return _PIPELINE_FIXTURE_LAYERS(self, layer)

    self.mock_client.generate_layer.side_effect = mock_generate_layer
    spec = self.pipeline.generate(self.seed)
    return slots, spec

  def _free_slots(self, slots, round_index=1):
    return [
        s for s in slots
        if int(s.get("round_index", 0)) == round_index
        and not str(s.get("decided_value") or "").strip()
    ]

  def _values(self, spec):
    return {
        str(r.get("round_id")): str(r.get("resolved_value", ""))
        for r in spec.get("rounds", [])
    }

  def test_the_free_pass_is_told_which_values_are_already_taken(self):
    # Named BEFORE the pick, not only rejected after it: a rejection costs a
    # whole extra LLM call on a path a quarter of the corpus takes.
    slots, _ = self._run(free_values=["default_upstream"])
    free = self._free_slots(slots)
    self.assertTrue(free, "round 1 must have reached a free pass")
    text = str(free[0].get("instruction", ""))
    self.assertIn("1,420 queries/min", text,
                  "the other round's decided value must be listed as taken")
    self.assertIn("ALREADY TAKEN", text)

  def test_a_free_pick_repeating_another_rounds_value_is_rejected(self):
    # First free answer steals r02's decided value; second is clean.
    slots, spec = self._run(
        free_values=["1,420 queries/min", "default_upstream"])
    free = self._free_slots(slots)
    self.assertEqual(len(free), 2, "the colliding pick must be retried")
    self.assertEqual(self._values(spec).get("r01"), "default_upstream")
    # And the retry is told WHY, so it can actually move off the value.
    self.assertIn("cross-round distinctness",
                  str(free[1].get("instruction", "")).lower())

  def test_a_clean_free_pick_costs_no_extra_attempt(self):
    slots, spec = self._run(free_values=["default_upstream"])
    self.assertEqual(len(self._free_slots(slots)), 1)
    self.assertEqual(self._values(spec).get("r01"), "default_upstream")

  def test_the_round_keeps_its_check_when_every_free_attempt_collides(self):
    # The concession path. Budgeted like every other gate in this loop: a round
    # whose evidence genuinely carries only one decisive reading keeps a graded
    # check rather than degrading to no check at all.
    slots, spec = self._run(
        free_values=["1,420 queries/min", "1,420 queries/min"])
    # The budget must actually have been spent before conceding -- asserting
    # only the final value would pass just as well with the gate deleted.
    self.assertEqual(len(self._free_slots(slots)), 2)
    self.assertEqual(self._values(spec).get("r01"), "1,420 queries/min")

  def test_a_bound_pick_is_not_second_guessed(self):
    # The phase layer owns distinctness over the values it DECIDES. Re-checking
    # here would only burn the bound attempts and force the very free pass this
    # gate exists to constrain, so r01 must still ship its own decided value
    # even though r02's sits in the same scenario.
    slots = []

    def mock_generate_layer(layer, payload):
      if layer == "conflict_layer0_expand_phase_decisive":
        return self.valid_phase_decisive
      if layer == "conflict_layer3_round_resolution":
        slots.append(payload.get("slot", {}))
        return {
            "decisive_anchor": "EVIDENCE:RACK_C2",
            "resolved_value": "182.1 U/L",
            "resolved_value_locus": "computed",
            "decision_outcome": "block the deploy",
        }
      return _PIPELINE_FIXTURE_LAYERS(self, layer)

    self.mock_client.generate_layer.side_effect = mock_generate_layer
    spec = self.pipeline.generate(self.seed)
    self.assertEqual(self._values(spec).get("r01"), "182.1 U/L")
    self.assertEqual(len(self._free_slots(slots)), 0,
                     "a bound pick must not be pushed onto the free path")


class TestDecidedValueStaysOutOfTheRoundText(unittest.TestCase):
  """S5 (companion): deciding the value early must not LEAK it early.

  Riding the decision on ``round_outline`` is what lets the asset authors carry
  it -- but that same outline is handed to the sub-steps that write the player
  instruction, the teammate chatter and the async updates. Those are exactly
  the surfaces the confidentiality gate forbids the value to appear on: a round
  whose instruction quotes the reading is solvable without opening the
  evidence.

  It is also self-defeating. A leak makes ``_validate_round_confidentiality``
  reject every bound resolution attempt, which releases the binding and drops
  the round back to a free pick -- S1, S2 and S5 all silently bypassed. So the
  three text sub-steps are told the exact string and ordered not to write it,
  instead of being left to infer it from a generic "don't reveal readings"
  line and repaired afterwards.
  """

  def setUp(self):
    TestPipelineRetries.setUp(self)

  def _instructions(self, *, decide=True):
    """Return ``layer -> [instruction text, ...]`` for one generation."""
    seen = {}

    def mock_generate_layer(layer, payload):
      text = str((payload.get("slot") or {}).get("instruction") or "")
      if text:
        seen.setdefault(layer, []).append(text)
      if layer == "conflict_layer0_expand_phase_decisive":
        return self.valid_phase_decisive if decide else {"decisive_points": []}
      if layer == "conflict_layer3_round_resolution":
        return {
            "decisive_anchor": "EVIDENCE:RACK_C2",
            "resolved_value": "default_upstream",
            "resolved_value_locus": "computed",
            "decision_outcome": "block the deploy",
        }
      return _PIPELINE_FIXTURE_LAYERS(self, layer)

    self.mock_client.generate_layer.side_effect = mock_generate_layer
    self.pipeline.generate(self.seed)
    return seen

  _TEXT_LAYERS = (
      "conflict_layer3_round_instruction",
      "conflict_layer3_round_mocks",
      "conflict_layer3_round_updates",
  )

  def test_each_text_substep_is_told_the_exact_string_to_avoid(self):
    seen = self._instructions()
    for layer in self._TEXT_LAYERS:
      with self.subTest(layer=layer):
        texts = seen.get(layer) or []
        self.assertTrue(texts, f"{layer} must have run")
        first = texts[0]
        self.assertIn("182.1 U/L", first,
                      "the sub-step must be shown the string it must avoid")
        low = first.lower()
        self.assertIn("must not", low)

  def test_no_secrecy_block_when_nothing_was_decided(self):
    # No dangling order to avoid a value that was never chosen.
    seen = self._instructions(decide=False)
    for layer in self._TEXT_LAYERS:
      with self.subTest(layer=layer):
        for text in seen.get(layer) or []:
          self.assertNotIn("182.1 U/L", text)

  def test_the_asset_substeps_are_still_told_to_carry_it(self):
    # The dual of the guard: evidence surfaces must NOT be told to hide the
    # value -- carrying it verbatim is the whole point of the inversion.
    seen = self._instructions()
    assets_texts = seen.get("conflict_layer0_expand_phase_assets") or []
    self.assertTrue(assets_texts, "the phase assets step must have run")
    joined = assets_texts[0]
    self.assertIn("182.1 U/L", joined)
    self.assertIn("DECISIVE VALUES DECIDED FOR THIS PHASE", joined)
    self.assertNotIn("must not appear", joined.lower())


class TestRoundAssetsLearnWhetherTheyMustCarryTheValue(unittest.TestCase):
  """S3: the round-level asset step is the usual carrier and knew nothing.

  Of the 523 corpus rounds that named a value, 427 had it printed on an asset
  the round could actually see. Of those 427, 241 (56.4%) had EVERY visible
  carrier introduced at that very round and another 131 (30.7%) had at least
  one -- 87.1% of carried rounds lean on this step. Yet its prompt says
  nothing whatsoever about the decisive value, and its contract
  (``conflict_layer3_round_assets``) places no requirement on ``ocr_text`` at
  all.

  It must not simply inherit S2's brief. The phase step has already been told
  to carry the value; telling this step the same thing manufactures a second
  carrier for free, and a second carrier is measurably easier -- one visible
  carrier scores 0.6310 against 0.7545 for two-to-four, controlling for value
  reuse and value length. The corpus shows that leak has already happened:
  only 209 of 523 rounds (40.0%) have exactly one visible carrier, while 56
  (10.7%) have six or more.

  So the two jobs are opposite depending on a fact the pipeline already holds
  -- whether an asset visible in this round literally prints the value -- and
  the prompt is routed on it. That is the same folded-substring predicate
  ``_validate_round_resolution`` already enforces and ``FileChecker`` already
  grades with, not a new heuristic: it routes a prompt, it does not judge or
  patch a result.
  """

  _OUTLINE = {
      "round_id": "r01",
      "decisive_value": "182.1 U/L",
      "decisive_label": "ALT on the 06:14 lab panel",
  }

  def test_it_is_told_to_carry_the_value_when_no_visible_asset_prints_it(self):
    text = _format_round_carrier_directive(self._OUTLINE, [
        {"id": "vlm_cpu_trend_01", "caption": "CPU", "ocr_text": "44% 51% 63%"},
    ])
    self.assertIn("182.1 U/L", text)
    self.assertIn("ALT on the 06:14 lab panel", text)
    self.assertIn("VERBATIM", text.upper())

  def test_the_carry_branch_still_states_the_density_floor(self):
    # Same floor as the phase brief: carriers in the corpus run a median 17
    # numeric readings. Without it "carry the value" is most easily satisfied
    # by a panel built around the answer.
    text = _format_round_carrier_directive(self._OUTLINE, [])
    self.assertRegex(text, r"at least 8 other")
    self.assertIn("do not highlight", text.lower())

  def test_it_is_told_not_to_repeat_the_value_when_a_visible_asset_prints_it(self):
    text = _format_round_carrier_directive(self._OUTLINE, [
        {"id": "vlm_lab_panel_01", "caption": "Lab panel",
         "ocr_text": "AST 41 U/L\nALT 182.1 U/L\nGGT 63 U/L"},
    ])
    self.assertIn("vlm_lab_panel_01", text)
    self.assertIn("do not print it again", text.lower())
    self.assertNotRegex(text, r"at least 8 other")

  def test_the_already_carried_branch_asks_for_disagreeing_readings(self):
    # The new assets still have a job -- they are the sources that conflict
    # with the carrier. Saying only "do not repeat it" would invite assets
    # that have nothing to do with the decision.
    text = _format_round_carrier_directive(self._OUTLINE, [
        {"id": "vlm_lab_panel_01", "ocr_text": "ALT 182.1 U/L"},
    ])
    self.assertIn("disagree", text.lower())

  def test_presence_is_folded_exactly_like_the_grader_folds_it(self):
    # Case and whitespace-run insensitive, matching ``file_checker._fold`` and
    # ``_fold_for_grounding``. A carrier that differs only by casing is still
    # a carrier, and treating it as absent would mint a redundant second one.
    text = _format_round_carrier_directive(self._OUTLINE, [
        {"id": "vlm_lab_panel_01", "ocr_text": "alt   182.1  u/l   ref 7-56"},
    ])
    self.assertIn("do not print it again", text.lower())

  def test_a_caption_only_carrier_counts(self):
    text = _format_round_carrier_directive(self._OUTLINE, [
        {"id": "vlm_lab_panel_01", "caption": "Panel showing ALT 182.1 U/L"},
    ])
    self.assertIn("vlm_lab_panel_01", text)

  def test_nothing_is_emitted_when_no_value_was_decided(self):
    # S1 degrading must not leave the asset step under orders about a value
    # that was never chosen.
    self.assertEqual(_format_round_carrier_directive({}, []), "")
    self.assertEqual(_format_round_carrier_directive(None, []), "")
    self.assertEqual(
        _format_round_carrier_directive({"decisive_value": "   "}, []), "")

  def test_the_directive_reaches_the_round_asset_prompt(self):
    # Wiring: the helper is worthless if the sub-step never sees it.
    payloads = {}

    def mock_generate_layer(layer, payload):
      if layer not in payloads:
        payloads[layer] = copy.deepcopy(payload)
      return _PIPELINE_FIXTURE_LAYERS(self, layer)

    TestPipelineRetries.setUp(self)
    self.mock_client.generate_layer.side_effect = mock_generate_layer
    self.pipeline.generate(self.seed)
    text = str(payloads["conflict_layer3_round_assets"]
               .get("slot", {}).get("instruction", ""))
    self.assertIn("182.1 U/L", text)

  def test_the_contract_requires_the_reading_to_land_in_the_text_fields(self):
    # The per-call instruction can be overridden or truncated; the contract is
    # the standing obligation, and it currently says nothing about ocr_text.
    from clasharness.scenario.llm_client import _layer_contract
    instruction = _layer_contract("conflict_layer3_round_assets")["instruction"]
    low = instruction.lower()
    self.assertIn("ocr_text", low)
    self.assertIn("caption", low)
    self.assertIn("verbatim", low)


class TestAssetsVisibleInRound(unittest.TestCase):
  """Gate F: name the "can this round actually open it?" predicate once.

  Three sites in the round loop each spelled out
  ``int(a.get("available_from_round", 1)) <= round_idx`` inline; the fourth --
  the list handed to the resolver -- forgot it entirely. Naming the predicate
  is what makes that omission visible. ``vlm_spec.VLMAssetSpec.from_dict``
  parses the same field as ``int(x or 1)``, so this reads it the same way.
  """

  def test_an_asset_declared_for_a_later_round_is_not_visible_yet(self):
    assets = [
        {"id": "now", "available_from_round": 1},
        {"id": "later", "available_from_round": 2},
    ]
    self.assertEqual(
        [a["id"] for a in _assets_visible_in_round(assets, 1)], ["now"])
    self.assertEqual(
        [a["id"] for a in _assets_visible_in_round(assets, 2)],
        ["now", "later"])

  def test_availability_that_is_missing_or_empty_means_visible_now(self):
    # ``vlm_spec`` parses this field as ``int(x or 1)``: an asset that never
    # declared a round has always been present from round 1, and hiding it
    # here would silently delete real evidence.
    for value in ({}, {"available_from_round": None},
                  {"available_from_round": ""}, {"available_from_round": 0}):
      with self.subTest(value=value):
        self.assertEqual(len(_assets_visible_in_round([value], 1)), 1)

  def test_a_string_round_number_is_read_as_a_number(self):
    # Layer output is JSON authored by a model; "2" is a routine shape and
    # must not be waved through as visible-from-round-1.
    assets = [{"id": "later", "available_from_round": "2"}]
    self.assertEqual(_assets_visible_in_round(assets, 1), [])
    self.assertEqual(len(_assets_visible_in_round(assets, 2)), 1)

  def test_junk_is_dropped_and_survivors_are_the_original_objects(self):
    # Callers read ``caption``/``ocr_text`` straight off the returned dicts,
    # so these must be the same objects in the same order, not copies.
    a = {"id": "a", "available_from_round": 1}
    b = {"id": "b", "available_from_round": 1}
    out = _assets_visible_in_round(["junk", None, a, 7, b], 1)
    self.assertEqual(len(out), 2)
    self.assertIs(out[0], a)
    self.assertIs(out[1], b)
    self.assertEqual(_assets_visible_in_round(None, 1), [])


class TestAssetKindById(unittest.TestCase):
  """Decisive-carrier gate: resolve 'asset:<id>' to the surface's kind.

  WHERE the decisive value sits decides whether a round measures visual
  grounding or just reading. Measured on the shipped corpus (4,496 graded
  round-runs): a 'rich_text' decisive carrier passed 54.9% against 38.2% for
  every other carrier combined, and it was the EASIER carrier inside every
  round type (+17.3/+5.0/+11.8/+12.5 points), while carrying 36.5% of all
  decisive values -- the largest single difficulty leak in the corpus.

  The gate rejects a 'rich_text' locus into the resolver's retry loop so a
  better carrier is picked AT THE SOURCE, and it is BUDGETED: the final
  attempt always accepts, so a round whose evidence genuinely is a log or a
  document keeps its decisive check instead of degrading to none.

  This lookup is the gate's only judgement, so it must fire on a POSITIVE
  match and abstain on everything else -- an id it cannot resolve is already
  owned by the visibility check in ``_validate_round_resolution``, which
  reports it with the right message.
  """

  def test_it_returns_the_kind_of_the_named_asset(self):
    assets = [
        {"id": "chart_panel", "kind": "chart"},
        {"id": "deploy_log", "kind": "rich_text"},
    ]
    self.assertEqual(_asset_kind_by_id(assets, "deploy_log"), "rich_text")
    self.assertEqual(_asset_kind_by_id(assets, "chart_panel"), "chart")

  def test_an_id_it_cannot_resolve_makes_the_gate_abstain(self):
    # Abstaining (not guessing) is the point: a locus naming an asset this
    # round cannot open is a DIFFERENT failure, already reported by the
    # visibility gate. Firing here would mask it behind a carrier message.
    assets = [{"id": "chart_panel", "kind": "chart"}]
    self.assertEqual(_asset_kind_by_id(assets, "not_here"), "")
    self.assertEqual(_asset_kind_by_id([], "chart_panel"), "")
    self.assertEqual(_asset_kind_by_id(None, "chart_panel"), "")

  def test_a_blank_id_never_matches(self):
    # ``cand_locus[len("asset:"):]`` on a bare 'asset:' yields ''. An empty
    # id must not match an asset that merely forgot to declare one.
    assets = [{"kind": "rich_text"}, {"id": "", "kind": "rich_text"}]
    for bad in ("", "   ", None):
      with self.subTest(bad=bad):
        self.assertEqual(_asset_kind_by_id(assets, bad), "")

  def test_ids_and_kinds_are_matched_after_stripping(self):
    # Layer output is model-authored JSON; stray whitespace is routine and
    # must not silently turn the gate off.
    assets = [{"id": " deploy_log ", "kind": " rich_text "}]
    self.assertEqual(_asset_kind_by_id(assets, "deploy_log"), "rich_text")
    self.assertEqual(_asset_kind_by_id(assets, " deploy_log "), "rich_text")

  def test_junk_entries_and_a_missing_kind_are_survivable(self):
    assets = ["junk", None, 7, {"id": "no_kind"}, {"id": "log",
                                                   "kind": "rich_text"}]
    self.assertEqual(_asset_kind_by_id(assets, "no_kind"), "")
    self.assertEqual(_asset_kind_by_id(assets, "log"), "rich_text")

  def test_the_first_declaration_of_an_id_wins(self):
    # The resolver is shown one list; a duplicated id is a generator bug, but
    # the gate must still be deterministic rather than order-dependent noise.
    assets = [{"id": "dup", "kind": "chart"}, {"id": "dup",
                                               "kind": "rich_text"}]
    self.assertEqual(_asset_kind_by_id(assets, "dup"), "chart")


class TestTheDecisiveEvidenceMustBeReadableInItsOwnRound(unittest.TestCase):
  """Gate F: the resolver could ground the answer in a panel nobody can open.

  Measured over 59 specs / 590 rounds / 16,206 graded observations: 8 rounds
  ship a ``resolved_value_locus`` naming an asset whose
  ``available_from_round`` falls AFTER that round. Their 216 observations
  average 0.3731 against 0.6590 for clean rounds -- a 0.286 drop, the largest
  of any defect class measured, with success-like runs falling from 0.257 to
  0.116. The player is told to cite a reading off evidence the round never
  hands them.

  Two things produced it. (a) The round asset step was told to carry the value
  verbatim (S3) but never that the carrier has to be openable NOW -- the phase
  brief says exactly that, the round-level directive did not -- and the
  histogram of ``available_from_round`` minus first-mention round shows 1,065
  assets correct against 23 deferred by one to three rounds. (b) The
  resolver's grounding list appended EVERY asset this round's narrative
  invented, unfiltered, so the check already inside
  ``_validate_round_resolution`` ("does not name a visible asset this round")
  was comparing against a list that had already been polluted with the
  invisible ones and could never fire.

  So the fix is to stop lying to the gate, not to add another one: show the
  resolver only what the player can open. A resolver that names something else
  now trips the existing gate and is regenerated with that feedback.

  Foreshadowing in the round PROSE is deliberately untouched: the 17 rounds
  that merely MENTION a not-yet-visible asset average 0.7251, above the clean
  baseline. That is legitimate setup, not a defect.
  """

  _OUTLINE = {
      "round_id": "r01",
      "decisive_value": "182.1 U/L",
      "decisive_label": "ALT on the 06:14 lab panel",
  }

  _ROUND_ASSETS = {
      "assets": [
          {
              "id": "lab_panel_0614",
              "kind": "chart",
              "generator": "template_chart",
              "path": "workspace/lab_panel_0614.png",
              "mime_type": "image/png",
              "available_from_round": 1,
              "anchor_ids": ["EVIDENCE:LAB0614"],
              "description": "Morning chemistry panel",
              "caption": "Chemistry panel 06:14",
              "ocr_text": "AST 41 U/L\nALT 182.1 U/L\nGGT 63 U/L",
              "platform_id": "workspace",
          },
          {
              "id": "lab_panel_tomorrow",
              "kind": "chart",
              "generator": "template_chart",
              "path": "workspace/lab_panel_tomorrow.png",
              "mime_type": "image/png",
              "available_from_round": 9,
              "anchor_ids": ["EVIDENCE:TOMORROW"],
              "description": "Repeat panel, drawn later",
              "caption": "Chemistry panel, repeat draw",
              "ocr_text": "AST 44 U/L\nALT 903.4 U/L\nGGT 61 U/L",
              "platform_id": "workspace",
          },
      ]
  }

  def setUp(self):
    TestPipelineRetries.setUp(self)

  def _run(self):
    """Drive a full generation whose round 1 invents one deferred asset."""
    prompts = {}

    def mock_generate_layer(layer, payload):
      slot = payload.get("slot", {}) if isinstance(payload, dict) else {}
      round_index = int(slot.get("round_index", 0) or 0)
      if layer == "conflict_layer3_round_assets":
        # Only round 1 introduces them; later rounds inherit round 1's assets
        # through ``phase_expanded_assets``, and re-emitting the same ids
        # would be a duplicate, not a visibility question.
        if round_index == 1:
          return copy.deepcopy(self._ROUND_ASSETS)
        return {"assets": []}
      if layer == "conflict_layer3_round_resolution":
        if round_index == 1:
          prompts.setdefault("resolution", str(slot.get("instruction", "")))
        return {
            "decisive_anchor": "EVIDENCE:LAB0614",
            "resolved_value": "182.1 U/L",
            "resolved_value_locus": "asset:phantom",
            "decision_outcome": "hold the next dose",
        }
      return _PIPELINE_FIXTURE_LAYERS(self, layer)

    self.mock_client.generate_layer.side_effect = mock_generate_layer
    # A phantom locus degrades resolution non-fatally, so generation finishes.
    self.pipeline.generate(self.seed)
    return prompts

  # --- the order that stops it being generated wrong in the first place ---

  def test_the_carry_order_says_the_carrier_must_open_in_this_round(self):
    text = _format_round_carrier_directive(self._OUTLINE, [])
    low = text.lower()
    self.assertIn("available_from_round", low)
    self.assertIn("this round", low)
    self.assertRegex(low, r"later round|defer")

  def test_the_already_carried_branch_does_not_repeat_the_timing_order(self):
    # There the new assets are the foils. When a foil becomes readable is not
    # what makes the decisive reading citable, and pasting the same order into
    # both branches is how the branch distinction rots.
    text = _format_round_carrier_directive(self._OUTLINE, [
        {"id": "vlm_lab_panel_01", "ocr_text": "ALT 182.1 U/L"},
    ])
    self.assertNotIn("available_from_round", text.lower())

  def test_the_contract_requires_the_carrier_to_be_openable_this_round(self):
    # The per-call instruction can be overridden or truncated; the contract is
    # the standing obligation.
    from clasharness.scenario.llm_client import _layer_contract
    low = _layer_contract("conflict_layer3_round_assets")["instruction"].lower()
    self.assertIn("available_from_round", low)

  # --- and the gate that catches it when it is generated wrong anyway ---

  def test_a_deferred_asset_is_not_offered_to_the_grounding_gate(self):
    import clasharness.scenario.conflict_llm_pipeline as mod
    seen = []

    def spy(value, locus, assets, workspace_texts, logger=None):
      seen.append([str(a.get("id", "")) for a in assets])

    with patch.object(mod, "_validate_round_resolution", spy):
      self._run()
    self.assertTrue(seen, "the grounding gate never ran")
    ids = seen[0]
    self.assertTrue(any("lab_panel_0614" in i for i in ids), ids)
    self.assertFalse(any("lab_panel_tomorrow" in i for i in ids), ids)

  def test_the_resolver_is_never_shown_text_it_cannot_open(self):
    prompt = self._run()["resolution"]
    self.assertIn("182.1 U/L", prompt)
    self.assertNotIn("903.4 U/L", prompt)

  def test_a_deferred_asset_contributes_no_anchor_to_the_resolution_prompt(self):
    # ``round_visible_anchor_ids`` is offered to the resolver as "anchors
    # already present on visible assets (prefer reusing one)". An anchor on a
    # panel this round cannot open is not one of those.
    prompt = self._run()["resolution"]
    self.assertIn("EVIDENCE:LAB0614", prompt)
    self.assertNotIn("EVIDENCE:TOMORROW", prompt)

  def test_the_existing_gate_already_rejects_an_invisible_locus(self):
    # CONTROL -- passes before the fix. The gate was never missing; it was
    # handed a list that had already been polluted. This pins the behaviour
    # the filter exists to switch back on.
    with self.assertRaises(ValueError) as ctx:
      _validate_round_resolution(
          "903.4 U/L",
          "asset:lab_panel_tomorrow",
          [{"id": "lab_panel_0614", "ocr_text": "ALT 182.1 U/L"}],
          {},
      )
    self.assertIn("visible asset", str(ctx.exception))

  def test_the_round_prose_may_still_foreshadow_an_asset_it_cannot_open(self):
    # CONTROL -- passes before the fix, and must keep passing after it. The 17
    # rounds that only MENTION a not-yet-visible asset average 0.7251, ABOVE
    # the 0.6590 clean baseline, so the filter must NOT reach the list the
    # prose leak-check uses.
    import clasharness.scenario.conflict_llm_pipeline as mod
    real = mod._validate_generated_round_narrative
    seen = []

    def spy(payload, round_id, visible_asset_ids):
      seen.append(list(visible_asset_ids))
      return real(payload, round_id, visible_asset_ids)

    with patch.object(mod, "_validate_generated_round_narrative", spy):
      self._run()
    self.assertTrue(seen, "the narrative validator never ran")
    self.assertTrue(
        any(any("lab_panel_tomorrow" in str(i) for i in ids) for ids in seen),
        seen)


class TestDecisiveCarrierMapReachesTheAssetRefiner(unittest.TestCase):
  """S7: the asset refiner must be told which readings it may not delete.

  ``_validate_round_resolution`` already guarantees a round's ``resolved_value``
  is literally present in the asset its ``resolved_value_locus`` names. But
  ``_generate_fresh_spec`` then hands every asset to ``llm_refine_vlm_assets``,
  which rewrites ``caption``/``ocr_text``, and nothing re-checked the grounding
  afterwards -- 84 of 523 rounds (16.1%) lost their decisive value that way.
  These tests pin the hand-off that closes the gap.
  """

  def _rounds(self):
    return [
        {"round_id": "r01", "resolved_value": "TICKET-4471",
         "resolved_value_locus": "asset:chart_a1"},
        {"round_id": "r02", "resolved_value": "SEC-PASS-9942",
         "resolved_value_locus": "asset:chart_a1"},
        {"round_id": "r03", "resolved_value": "runbook line 12",
         "resolved_value_locus": "file:/workspace/runbook.md"},
        {"round_id": "r04", "resolved_value": "42",
         "resolved_value_locus": "computed"},
        {"round_id": "r05", "resolved_value": "",
         "resolved_value_locus": "asset:chart_a2"},
        {"round_id": "r06", "resolved_value": "TICKET-4471",
         "resolved_value_locus": "asset:chart_a1"},
    ]

  def test_only_asset_loci_become_carrier_obligations(self):
    from clasharness.scenario.conflict_vlm_materializer import _decisive_values_by_asset

    got = _decisive_values_by_asset(self._rounds())
    # file: points at a workspace file the refiner never rewrites; computed was
    # never grounded in the first place; an empty value has nothing to protect.
    self.assertEqual(sorted(got), ["chart_a1"])
    # Two distinct readings on the same carrier, and the r06 repeat deduped.
    self.assertEqual(got["chart_a1"], ["TICKET-4471", "SEC-PASS-9942"])

  def test_malformed_rounds_are_skipped_not_fatal(self):
    from clasharness.scenario.conflict_vlm_materializer import _decisive_values_by_asset

    self.assertEqual(_decisive_values_by_asset([]), {})
    self.assertEqual(_decisive_values_by_asset(None), {})
    self.assertEqual(_decisive_values_by_asset(["not-a-dict", {}]), {})

  def test_fresh_spec_hands_the_map_to_the_refiner(self):
    from unittest.mock import patch

    seed = ConflictVLMSeed(
        slug="s7_carrier_handoff",
        domain="system",
        title="S7 carrier hand-off",
        target_rounds=1,
        core_conflict="Carrier hand-off check.",
        conflict_taxonomy=["temporal_staleness"],
        participants=["owner", "assistant"],
        required_constraints=[],
        allowed_tool_families=["agents", "file"],
        min_round_steps=4,
        require_final_verification=False,
    )
    raw_data = {
        "workspace": {"files": []},
        "sessions": [],
        "updates": [],
        "evals": [],
        "agents": [],
        "rounds": self._rounds(),
        "assets": [
            {"id": "chart_a1", "kind": "chart", "generator": "template_chart",
             "path": "charts/a1.png", "available_from_round": 1},
            {"id": "chart_a2", "kind": "chart", "generator": "template_chart",
             "path": "charts/a2.png", "available_from_round": 1},
        ],
    }

    class _Probe(Exception):
      pass

    captured = {}

    def _recorder(_seed, **kwargs):
      captured.update(kwargs)
      raise _Probe()

    class _DummyClient:
      def generate_layer(self, layer, payload):
        raise AssertionError(f"unexpected layer {layer}")

    from clasharness.scenario import conflict_vlm_materializer as cvm

    with patch.object(cvm.ConflictLLMPipeline, "generate", return_value=raw_data), \
         patch.object(cvm, "llm_refine_vlm_assets", _recorder):
      with self.assertRaises(_Probe):
        cvm._generate_fresh_spec(
            seed,
            llm_client=_DummyClient(),
            image_provider="mock",
            chart_provider="mock",
        )

    self.assertEqual(
        captured.get("decisive_values_by_asset"),
        {"chart_a1": ["TICKET-4471", "SEC-PASS-9942"]},
    )
    # The dead ``anchors`` knob must not be passed alongside the live one.
    self.assertNotIn("anchors", captured)


class TestTheRoundTextMustNotSettleTheDecision(unittest.TestCase):
  """S4: the narrative was answering the round on its own.

  Measured over the last corpus run, splitting on what a run could actually
  reach before it wrote anything:

  * 548 runs opened no asset and had no decisive value anywhere in the text
    they had seen -- prose only. Their verdict check passed 88.9% of the time.
  * 8,341 runs did open evidence. 88.4%.

  Opening the evidence was worth -0.5 percentage points. Several scenarios
  (``access_request_approval_v03``, ``employee_onboarding_workflow_v02``,
  ``multi_tab_slot_monitoring_v02``) were solved from prose 100% of the time.
  It is not that the verdict is a coin flip either: only 16 of 58 scenarios
  reuse two or fewer distinct ``decision_outcome`` values across their rounds.

  The obvious culprit -- the value being written into the text -- is not the
  mechanism. The decisive string appears in only 0.6% of round instructions
  (78/13,801) and is readable in pre-write stdout in 3.3% of runs
  (458/13,816); S5's secrecy guard already targets that path. What is left is
  that the prose settles the conflict by how it is written, not by quoting the
  reading.

  So the three text sub-steps get the positive dual of the secrecy guard. The
  guard says "do not print this string"; the stakes block says "you know which
  side this string favours -- do not let that show". The specific handle is
  credibility symmetry: that is a hypothesis about the mechanism, not a
  measured cause, but it is the actionable form of "leave the decision open".

  (Caveat kept honest: the prose-only sample is self-selected -- agents that
  skip the evidence may be a different population. The direction of 88.9% vs
  88.4% is unambiguous regardless.)
  """

  _OUTLINE = {
      "round_id": "r01",
      "decisive_value": "182.1 U/L",
      "decisive_label": "ALT on the 06:14 lab panel",
  }

  def test_it_names_the_quantity_the_conflict_turns_on(self):
    # Disagreement has to be ABOUT something. Naming the label (never the
    # value -- that is the guard's job) is what points the chatter at it.
    text = _format_decisive_stakes(self._OUTLINE)
    self.assertIn("ALT on the 06:14 lab panel", text)

  def test_it_orders_the_text_to_leave_the_decision_unsettled(self):
    text = _format_decisive_stakes(self._OUTLINE)
    low = text.lower()
    self.assertIn("must", low)
    self.assertRegex(low, r"unsettled|undecided|stay open")
    self.assertRegex(low, r"obviously right|obviously wrong")

  def test_it_asks_for_the_losing_side_to_be_written_just_as_credibly(self):
    # The operational handle. "Keep it open" alone is too abstract to act on;
    # the concrete failure is one side sounding precise and the other vague.
    text = _format_decisive_stakes(self._OUTLINE)
    low = text.lower()
    self.assertRegex(low, r"equal|just as credib|same.*credib")
    self.assertRegex(low, r"specific")
    self.assertRegex(low, r"confiden")

  def test_it_cites_the_measurement_that_motivates_it(self):
    # The prompt states why the constraint exists; a bare prohibition is
    # easier to rationalise around than one with a number attached.
    text = _format_decisive_stakes(self._OUTLINE)
    self.assertIn("88.9%", text)
    self.assertIn("88.4%", text)

  def test_it_does_not_relax_the_secrecy_guard(self):
    # The stakes block is the reason the value is in the prompt at all, which
    # makes it the most likely place to accidentally licence writing it.
    text = _format_decisive_stakes(self._OUTLINE)
    self.assertNotIn("182.1 U/L", text)
    low = text.lower()
    self.assertRegex(low, r"without writing the reading|still (binds|applies)")

  def test_it_works_without_a_label(self):
    text = _format_decisive_stakes({"decisive_value": "182.1 U/L"})
    self.assertTrue(text.strip())
    self.assertNotIn("182.1 U/L", text)

  def test_nothing_is_emitted_when_no_value_was_decided(self):
    # S1 degrading must not leave the text steps under orders about a
    # conflict whose decisive reading was never chosen.
    self.assertEqual(_format_decisive_stakes({}), "")
    self.assertEqual(_format_decisive_stakes(None), "")
    self.assertEqual(_format_decisive_stakes({"decisive_value": "   "}), "")

  # ---------------------------------------------------------------- wiring --

  _TEXT_LAYERS = (
      "conflict_layer3_round_instruction",
      "conflict_layer3_round_mocks",
      "conflict_layer3_round_updates",
  )

  def _instructions(self, *, decide=True):
    """Return ``layer -> [instruction text, ...]`` for one generation."""
    seen = {}

    def mock_generate_layer(layer, payload):
      text = str((payload.get("slot") or {}).get("instruction") or "")
      if text:
        seen.setdefault(layer, []).append(text)
      if layer == "conflict_layer0_expand_phase_decisive":
        return self.valid_phase_decisive if decide else {"decisive_points": []}
      return _PIPELINE_FIXTURE_LAYERS(self, layer)

    TestPipelineRetries.setUp(self)
    self.mock_client.generate_layer.side_effect = mock_generate_layer
    self.pipeline.generate(self.seed)
    return seen

  def test_every_text_substep_gets_the_stakes_and_the_asset_steps_do_not(self):
    # Scoping is the point. The evidence steps have the OPPOSITE job: S2 and
    # S3 order them to make one surface decisive. Telling them to keep things
    # balanced would contradict "carry it verbatim".
    seen = self._instructions()
    for layer in self._TEXT_LAYERS:
      with self.subTest(layer=layer):
        texts = seen.get(layer) or []
        self.assertTrue(texts, f"{layer} must have run")
        self.assertIn("the decision must stay open", texts[0].lower())
    for layer in ("conflict_layer0_expand_phase_assets",
                  "conflict_layer3_round_assets"):
      with self.subTest(layer=layer):
        for text in seen.get(layer) or []:
          self.assertNotIn("the decision must stay open", text.lower())

  def test_the_guard_is_stated_before_the_stakes(self):
    # The stakes block refers back to the reading the guard names. If the
    # order flips, "the reading above" points at nothing.
    seen = self._instructions()
    for layer in self._TEXT_LAYERS:
      with self.subTest(layer=layer):
        text = (seen.get(layer) or [""])[0].lower()
        guard = text.find("confidentiality -- the exact string")
        stakes = text.find("the decision must stay open")
        self.assertGreaterEqual(guard, 0, "the S5 guard must still be there")
        self.assertGreaterEqual(stakes, 0)
        self.assertLess(guard, stakes)

  def test_no_stakes_block_when_nothing_was_decided(self):
    seen = self._instructions(decide=False)
    for layer in self._TEXT_LAYERS:
      with self.subTest(layer=layer):
        for text in seen.get(layer) or []:
          self.assertNotIn("the decision must stay open", text.lower())

  def test_the_contract_also_forbids_settling_the_round(self):
    # The per-call instruction can be overridden or truncated; the contract is
    # the standing obligation for the dominant narrative surface.
    from clasharness.scenario.llm_client import _layer_contract
    instruction = _layer_contract("conflict_layer3_round_instruction")[
        "instruction"].lower()
    self.assertRegex(instruction, r"obviously right|settl")
    self.assertRegex(instruction, r"resolv|decision")


class TestTheWorkspaceContractDeclaresTheShapeItIsCheckedAgainst(
    unittest.TestCase
):
  """The workspace layer must state the nested shape its validator enforces.

  ``conflict_layer1b_files`` was the only 1b sub-layer that paired a narrow
  downstream validator with an undeclared nested shape, and it was the largest
  non-resolution failure source in the v3 run (24 stage retries; 18 of 60
  scenarios hit it; 2 scenarios died on it and were lost outright).

  The cause was not weak instruction-following. Its sibling
  ``conflict_layer1b_sessions`` has no shape declaration either and produced
  771/771 well-formed messages, because "a chat message" has one obvious JSON
  encoding. "A workspace" has two equally natural ones -- a path->content
  mapping and a list of {path, content} objects -- and the validator accepts
  only the second. Worse, the first failure's error text pushed the model from
  the mapping form into a list of bare path strings, which fails with a
  *different* error, so extra retries alone never recovered it.
  """

  def _files_instruction(self) -> str:
    from clasharness.scenario.llm_client import _layer_contract
    return _layer_contract("conflict_layer1b_files")["instruction"]

  def test_the_contract_spells_out_the_nested_shape(self):
    instruction = self._files_instruction()
    self.assertIn('"workspace"', instruction)
    self.assertIn('"files"', instruction)
    self.assertIn('"path"', instruction)
    self.assertIn('"content"', instruction)

  def test_the_contract_names_both_observed_wrong_encodings(self):
    """Naming them is the whole point: the model converged on exactly these two."""
    low = self._files_instruction().lower()
    self.assertIn("mapping of path", low)
    self.assertIn("bare path strings", low)

  def test_the_declared_shape_is_the_shape_the_validator_accepts(self):
    """Ties the prompt text to the validator so the two cannot drift apart."""
    from clasharness.scenario.conflict_llm_pipeline import (
        _validate_background_payload,
    )
    declared = {
        "workspace": {"files": [{"path": "src/app.py", "content": "x = 1\n"}]},
        "sessions": [],
    }
    _validate_background_payload(declared)  # must not raise

  def test_the_encodings_the_contract_forbids_really_are_rejected(self):
    from clasharness.scenario.conflict_llm_pipeline import (
        _validate_background_payload,
    )
    # The a1 failure shape: a path -> content mapping.
    with self.assertRaises(ValueError):
      _validate_background_payload(
          {"workspace": {"src/app.py": "x = 1\n"}, "sessions": []}
      )
    # The a2/a3 failure shape the a1 error feedback pushed the model into.
    with self.assertRaises(ValueError):
      _validate_background_payload(
          {"workspace": {"files": ["src/app.py"]}, "sessions": []}
      )

  def test_the_declaration_survives_into_the_prompt(self):
    """The contract is JSON-serialized into the user prompt; braces must survive."""
    import json
    from clasharness.scenario.llm_client import _layer_contract
    prompt = json.dumps(
        {"layer": "conflict_layer1b_files",
         "contract": _layer_contract("conflict_layer1b_files")},
        ensure_ascii=False,
    )
    self.assertIn('{\\"workspace\\": {\\"files\\"', prompt)


class TestNoContractShipsAnUnreplacedPlaceholder(unittest.TestCase):
  """``_layer_contract`` returns its instruction verbatim -- it never formats.

  So a ``{min_files}``-style placeholder left in a contract reaches the model
  as literal braces instead of a number. Three had accumulated across two
  contracts, which is also evidence those contracts were never read end to end.
  This is a standing guard over all of them, not a one-off fix.
  """

  def test_every_contract_instruction_is_fully_resolved(self):
    import inspect
    import re
    from clasharness.scenario import llm_client

    src = inspect.getsource(llm_client._layer_contract)
    layers = re.findall(r'^\s{8}"([a-z0-9_]+)":\s*\{', src, re.M)
    self.assertGreater(len(layers), 50, "contract scrape found too few layers")

    offenders = {}
    for layer in layers:
      instruction = llm_client._layer_contract(layer).get("instruction", "")
      # Deliberately narrow: only bare `{snake_case}` tokens, so a declared
      # JSON shape like {"workspace": ...} is not mistaken for a placeholder.
      found = sorted(set(re.findall(r"\{([a-z_][a-z0-9_]*)\}", instruction)))
      if found:
        offenders[layer] = found
    self.assertEqual(offenders, {})


if __name__ == "__main__":
  unittest.main()

