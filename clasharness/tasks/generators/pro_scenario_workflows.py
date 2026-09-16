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

"""Scenario-forge-style PRO workflow generator.

This generator creates new PRO tasks from scenario-level specs instead of
wrapping the existing hard/complex task families.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import shlex
from typing import Any, Iterator

from clasharness.core.task import TaskData
from clasharness.tasks.generation_options import get_generation_options
from clasharness.tasks.registry import BaseTaskGenerator, Pass, SetupResult


@dataclass(frozen=True)
class ProSession:
    name: str
    channel: str
    type: str
    role: str
    participants: tuple[str, ...]


@dataclass(frozen=True)
class ProUpdate:
    update_id: str
    before_round: int
    reveals: str
    session_appends: tuple[str, ...]
    workspace_adds: tuple[str, ...]


@dataclass(frozen=True)
class ProSeed:
    slug: str
    domain: str
    title: str
    protagonist: str
    role: str
    organization: str
    core_narrative: str
    contradictions: tuple[dict[str, str], ...]
    biases: tuple[dict[str, str], ...]
    preferences: dict[str, str]
    sessions: tuple[ProSession, ...]
    workspace_files: tuple[dict[str, str], ...]
    updates: tuple[ProUpdate, ...]


_SEEDS: tuple[ProSeed, ...] = (
    ProSeed(
        slug="api_security_breach_response",
        domain="tech_security",
        title="API Security Breach Response",
        protagonist="Alex Rivera",
        role="Product Manager",
        organization="NexaFlow",
        core_narrative=(
            "A startup PM investigates an API vulnerability with conflicting record-count, "
            "exposure-window, and disclosure-strategy evidence."
        ),
        contradictions=(
            {"id": "C1", "type": "factual", "topic": "records exposed", "truth": "2,340 confirmed records"},
            {"id": "C2", "type": "authority", "topic": "exposure window", "truth": "43 days from Oct 14 to Nov 26"},
            {"id": "C3", "type": "non_conflict", "topic": "vulnerability timeline", "truth": "Oct 14 deploy and Nov 5 exploitation are consistent"},
            {"id": "C4", "type": "temporal_process", "topic": "disclosure strategy", "truth": "CEO moved from minimal to full transparency after update 3"},
        ),
        biases=(
            {"id": "B1", "type": "anchoring", "phrase": "enumeration is practically impossible", "correction_update": "upd1"},
            {"id": "B2", "type": "authority", "phrase": "the incident is contained", "correction_update": "upd2"},
        ),
        preferences={
            "P1": "Use ISO timestamps and thousands separators.",
            "P2": "Name files YYYY-MM-DD_{topic}_v{N}.md.",
            "P3": "Use Summary, Details, Action Items sections.",
            "P4": "Use source citations for every claim.",
            "P5": "Tag unverified claims with [UNVERIFIED].",
        },
        sessions=(
            ProSession("security_jake", "discord", "dm", "Security Consultant", ("Jake", "Alex")),
            ProSession("cto_sana", "discord", "dm", "CTO", ("Sana", "Alex")),
            ProSession("devops_diego", "telegram", "dm", "DevOps Engineer", ("Diego", "Alex")),
            ProSession("ceo_jordan", "slack", "dm", "CEO", ("Jordan", "Alex")),
            ProSession("security_response", "discord", "group", "Security response team", ("Sana", "Jake", "Diego", "Alex")),
            ProSession("customer_notif", "slack", "group", "Customer notification team", ("Jordan", "Raj", "Mia", "Alex")),
        ),
        workspace_files=(
            {"path": "api_endpoint_register.md", "type": "system_export", "facts": "Oct 14 deploy and endpoint versions"},
            {"path": "customer_data_inventory.md", "type": "system_export", "facts": "PII table inventory"},
            {"path": "developer_docs_screenshot.md", "type": "system_export", "facts": "list=true parameter snapshot"},
            {"path": "disclosure_report_initial.md", "type": "human_document", "facts": "minimal disclosure draft"},
            {"path": "incident_response_checklist.md", "type": "human_document", "facts": "response status"},
            {"path": "notification_draft_v1.md", "type": "human_document", "facts": "customer notification v1"},
            {"path": "vulnerability_technical_brief.md", "type": "analysis", "facts": "technical exploit notes"},
        ),
        updates=(
            ProUpdate("upd1", 8, "access logs confirm 2,340 records", ("security_jake", "devops_diego"), ("access_log_analysis.md",)),
            ProUpdate("upd2", 15, "deployment timeline confirms 43-day window", ("security_response", "cto_sana"), ("deployment_timeline.md",)),
            ProUpdate("upd3", 22, "CEO shifts to full transparency", ("ceo_jordan", "customer_notif"), ("notification_final.md",)),
        ),
    ),
    ProSeed(
        slug="grant_launch_evidence_reconciliation",
        domain="research_operations",
        title="Grant Launch Evidence Reconciliation",
        protagonist="Maya Chen",
        role="Research Operations Lead",
        organization="Aster Lab",
        core_narrative=(
            "A research-ops lead reconciles contradictory launch, budget, IRB, and partner-readiness evidence "
            "across lab docs and multi-channel history."
        ),
        contradictions=(
            {"id": "C1", "type": "factual", "topic": "available launch budget", "truth": "$184,500 usable after restricted funds"},
            {"id": "C2", "type": "authority", "topic": "IRB readiness", "truth": "conditional approval requires revised consent appendix"},
            {"id": "C3", "type": "non_conflict", "topic": "partner onboarding", "truth": "two partners ready now, one delayed by procurement"},
            {"id": "C4", "type": "temporal_process", "topic": "launch recommendation", "truth": "recommendation changes from launch-now to staged launch after update 3"},
        ),
        biases=(
            {"id": "B1", "type": "anchoring", "phrase": "the full budget is already cleared", "correction_update": "upd1"},
            {"id": "B2", "type": "authority", "phrase": "IRB is done, do not slow launch", "correction_update": "upd2"},
        ),
        preferences={
            "P1": "Use a risk register table with owner and due date.",
            "P2": "Separate confirmed, conditional, and blocked items.",
            "P3": "Cite the evidence source next to each recommendation.",
            "P4": "Use YYYY-MM-DD dates.",
            "P5": "End with a go/no-go decision and fallback path.",
        },
        sessions=(
            ProSession("pi_elena", "slack", "dm", "Principal Investigator", ("Elena", "Maya")),
            ProSession("finance_noah", "email", "dm", "Finance Manager", ("Noah", "Maya")),
            ProSession("irb_samira", "teams", "dm", "IRB Coordinator", ("Samira", "Maya")),
            ProSession("partner_ops", "slack", "group", "Partner operations", ("Maya", "Lena", "Omar", "Priya")),
            ProSession("launch_room", "discord", "group", "Launch planning", ("Maya", "Elena", "Noah", "Samira")),
            ProSession("field_team", "telegram", "group", "Field team", ("Maya", "Iris", "Ben", "Kai")),
        ),
        workspace_files=(
            {"path": "grant_budget_initial.md", "type": "finance_export", "facts": "$220,000 headline budget"},
            {"path": "restricted_funds_memo.md", "type": "finance_export", "facts": "$35,500 restricted"},
            {"path": "irb_approval_email.md", "type": "human_document", "facts": "conditional approval language"},
            {"path": "partner_readiness_grid.md", "type": "system_export", "facts": "partner launch readiness"},
            {"path": "launch_plan_v1.md", "type": "human_document", "facts": "launch-now recommendation"},
            {"path": "consent_appendix_draft.md", "type": "human_document", "facts": "outdated consent appendix"},
            {"path": "field_staffing_notes.md", "type": "analysis", "facts": "field coverage constraints"},
        ),
        updates=(
            ProUpdate("upd1", 8, "restricted funds reduce usable budget to $184,500", ("finance_noah", "launch_room"), ("budget_reconciliation.md",)),
            ProUpdate("upd2", 15, "IRB approval is conditional on revised consent appendix", ("irb_samira", "pi_elena"), ("irb_conditions.md",)),
            ProUpdate("upd3", 22, "one partner delayed by procurement, staged launch is safest", ("partner_ops", "field_team"), ("staged_launch_recommendation.md",)),
        ),
    ),
)

_SEED_BY_SLUG = {seed.slug: seed for seed in _SEEDS}


_MC_ROUNDS = {1, 4, 7, 10, 14, 18, 22}


def _quote(value: str) -> str:
    return shlex.quote(value)


def _scenario_metadata(seed: ProSeed) -> dict[str, Any]:
    return {
        "layer0_narrative": {
            "domain": seed.domain,
            "protagonist": seed.protagonist,
            "role": seed.role,
            "organization": seed.organization,
            "core_narrative": seed.core_narrative,
            "contradictions": list(seed.contradictions),
            "biases": list(seed.biases),
            "preferences": seed.preferences,
        },
        "layer1_workspace": {
            "files": list(seed.workspace_files),
            "initial_file_count": len(seed.workspace_files),
        },
        "layer2_sessions": [
            {
                "session_id": session.name,
                "channel": session.channel,
                "type": session.type,
                "role": session.role,
                "participants": list(session.participants),
                "phase_count": 3,
            }
            for session in seed.sessions
        ],
        "layer4_dynamic": [
            {
                "update_id": update.update_id,
                "before_round": update.before_round,
                "reveals": update.reveals,
                "session_appends": list(update.session_appends),
                "workspace_adds": list(update.workspace_adds),
            }
            for update in seed.updates
        ],
    }


def _active_updates(seed: ProSeed, round_number: int) -> list[str]:
    return [
        update.update_id
        for update in seed.updates
        if round_number >= update.before_round
    ]


def _round_target(seed: ProSeed, round_number: int) -> dict[str, str]:
    contradiction = seed.contradictions[(round_number - 1) % len(seed.contradictions)]
    preference_id = f"P{((round_number - 1) % 5) + 1}"
    if round_number in _MC_ROUNDS:
        round_type = "multi_choice"
        action = "select all evidence-grounded options"
    else:
        round_type = "exec_check"
        action = "write an evidence-backed decision artifact"
    return {
        "round_id": f"q{round_number:02d}",
        "round_type": round_type,
        "contradiction_id": contradiction["id"],
        "topic": contradiction["topic"],
        "truth": contradiction["truth"],
        "preference_id": preference_id,
        "preference_rule": seed.preferences[preference_id],
        "action": action,
    }


@BaseTaskGenerator.register("pro_scenario_workflow")
class ProScenarioWorkflowGenerator(BaseTaskGenerator):
    """Generate new scenario-forge-style PRO tasks."""

    required_domains = ("composite", "file", "messaging", "monitoring")
    difficulty = 3
    parameters = {"seed_slug": [seed.slug for seed in _SEEDS]}

    def setup(
        self, params: dict[str, Any], initial_config: dict[str, Any]
    ) -> Iterator[tuple[SetupResult, TaskData]]:
        del initial_config
        options = get_generation_options()
        if not options.include_pro_scenarios:
            return
        seed = _SEED_BY_SLUG[str(params["seed_slug"])]
        target_rounds = options.pro_rounds_per_scenario
        scenario_meta = _scenario_metadata(seed)
        for round_number in range(1, target_rounds + 1):
            target = _round_target(seed, round_number)
            active_updates = _active_updates(seed, round_number)
            yield Pass(), TaskData(
                public={
                    "pro_series": "pro_strict_v1",
                    "pro_scenario_slug": seed.slug,
                    "pro_scenario_title": seed.title,
                    "pro_round_id": target["round_id"],
                    "pro_round_type": target["round_type"],
                    "pro_active_updates": active_updates,
                    "pro_target": target,
                    "pro_scenario": scenario_meta,
                    "prompt_style": "scenario_forge",
                },
                private={
                    "expected_truth": target["truth"],
                    "expected_preference": target["preference_rule"],
                },
            )

    def build_task_id(
        self, params: dict[str, Any], data: TaskData, task_counter: int
    ) -> str:
        del params, task_counter
        return f"pro_scenario_workflow_{data.public['pro_scenario_slug']}_{data.public['pro_round_id']}"

    def get_instruction(self, params: dict[str, Any], data: TaskData) -> str:
        seed = _SEED_BY_SLUG[str(params["seed_slug"])]
        target = data.public["pro_target"]
        updates = data.public["pro_active_updates"]
        update_text = ", ".join(updates) if updates else "none yet"
        return (
            f"You are handling PRO scenario '{seed.title}' for {seed.protagonist}, "
            f"{seed.role} at {seed.organization}. Round {target['round_id']} asks you to "
            f"{target['action']} for topic '{target['topic']}'. Active updates: {update_text}. "
            f"Use the scenario workspace, session evidence, and preference rule {target['preference_id']}: "
            f"{target['preference_rule']} Resolve the conflict using objective truth, not the most recent or loudest claim. "
            f"Create the final artifact at /reports/{seed.slug}_{target['round_id']}.md."
        )

    def get_solution(self, params: dict[str, Any], data: TaskData) -> list[str]:
        seed = _SEED_BY_SLUG[str(params["seed_slug"])]
        target = data.public["pro_target"]
        path = f"/reports/{seed.slug}_{target['round_id']}.md"
        content = (
            f"Summary: {target['truth']}. "
            f"Details: resolved {target['contradiction_id']} on {target['topic']} using workspace and session evidence. "
            f"Action Items: follow {target['preference_id']} - {target['preference_rule']} "
            f"Active updates: {', '.join(data.public['pro_active_updates']) or 'none'}."
        )
        return [
            "file list --path /",
            f"file create --path {_quote(path)} --content {_quote(content)}",
            f"file read --path {_quote(path)}",
            "done",
        ]

    def get_evaluation_checks(
        self, params: dict[str, Any], data: TaskData
    ) -> list[dict[str, Any]]:
        seed = _SEED_BY_SLUG[str(params["seed_slug"])]
        target = data.public["pro_target"]
        path = f"/reports/{seed.slug}_{target['round_id']}.md"
        truth_pattern = re.escape(target["truth"])
        pref_pattern = re.escape(target["preference_id"])
        return [
            {
                "type": "effect",
                "effect_type": "files_created",
                "condition": "field_equals",
                "expected": {"field": "path", "value": path},
                "name": "creates required PRO artifact",
            },
            {
                "type": "output",
                "condition": "regex",
                "expected": truth_pattern,
                "output_field": "command_history",
                "name": "uses objective truth rather than bias claim",
                "ignore_case": False,
            },
            {
                "type": "output",
                "condition": "regex",
                "expected": pref_pattern,
                "output_field": "command_history",
                "name": "applies scoped preference rule",
                "ignore_case": False,
            },
        ]

    def get_task_schema_metadata(
        self,
        params: dict[str, Any],
        data: TaskData,
        instruction: str,
        solution: list[str],
        checks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        del params, solution, checks
        return {
            "canonical_instruction": instruction,
            "decision_requirements": [
                "resolve_cross_source_conflict",
                "apply_dynamic_update",
                "respect_personalization_rule",
                "cite_workspace_and_session_evidence",
            ],
            "realism_tags": [
                "pro_scenario",
                "scenario_forge_style",
                "multi_round",
                "dynamic_updates",
                "workspace_sessions",
                str(data.public["pro_round_type"]),
            ],
            "online_requirement": "none",
            "availability_tier": "stable",
        }
