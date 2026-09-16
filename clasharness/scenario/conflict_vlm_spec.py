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
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from clasharness.scenario.vlm_spec import (
    VLMInputPolicy, VLMGenerationPolicy, VLMAssetSpec, VLMEvalSpec
)
from clasharness.scenario.spec import (
    MockSubagent,
    ScenarioWorkspace,
    ScenarioSession,
)

_SAFE_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]*$")
_DEFAULT_TOOL_FAMILIES = ["file", "email", "tasks", "calendar", "asset", "agents"]
_DEFAULT_MODALITIES = ["image", "chart", "table_screenshot", "document_image", "rich_text"]

@dataclass
class ConflictAgentSpec:
    agent_role: str
    agent_name: str
    capabilities: list[str] = field(default_factory=list)
    private_knowledge: list[str] = field(default_factory=list)
    virtual_duration_seconds: int = 10

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConflictAgentSpec":
        return cls(
            agent_role=str(data.get("agent_role", "")),
            agent_name=str(data.get("agent_name", "")),
            capabilities=list(data.get("capabilities", [])),
            private_knowledge=list(data.get("private_knowledge", [])),
            virtual_duration_seconds=int(data.get("virtual_duration_seconds", 10)),
        )


@dataclass
class ConflictInteractionSpec:
    interaction_type: str
    active_subagent: str
    trigger_evidence: str
    trigger_source_file: str
    resolution_action: str
    dependency_steps: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConflictInteractionSpec":
        return cls(
            interaction_type=str(data.get("interaction_type", "")),
            active_subagent=str(data.get("active_subagent", "")),
            trigger_evidence=str(data.get("trigger_evidence", "")),
            trigger_source_file=str(data.get("trigger_source_file", "")),
            resolution_action=str(data.get("resolution_action", "")),
            dependency_steps=list(data.get("dependency_steps", [])),
        )


@dataclass
class ConflictScenarioUpdate:
    update_id: str
    trigger_virtual_time: int | None = None  # Asynchronous injection trigger
    anchor_id: str | None = None
    workspace_adds: list[dict[str, Any]] = field(default_factory=list)
    session_appends: list[dict[str, Any]] = field(default_factory=list)
    session_redactions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConflictScenarioUpdate":
        return cls(
            update_id=str(data.get("update_id", "")),
            trigger_virtual_time=data.get("trigger_virtual_time"),
            anchor_id=data.get("anchor_id"),
            workspace_adds=list(data.get("workspace_adds", [])),
            session_appends=list(data.get("session_appends", [])),
            session_redactions=list(data.get("session_redactions", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ConflictVLMRoundSpec:
    round_id: str
    round_type: str
    visible_asset_ids: list[str]
    required_modalities: list[str]
    expected_anchor_ids: list[str]
    required_tool_families: list[str]
    instruction: str = ""
    update_ids: list[str] = field(default_factory=list)
    solution_commands: list[str] = field(default_factory=list)
    mock_subagents: list[MockSubagent] = field(default_factory=list)
    # First-class classification of the round's CORRECT resolution/verdict (item
    # ③). A short, free-form label the generator authors (grounded in the round's
    # conflict) naming which way the coordinator must resolve the conflict, e.g.
    # "roll back to v4.2", "block the deploy", "escalate to on-call". Paired with
    # the decisive ``resolved_value`` (the concrete evidence/answer the label rests
    # on) it forms the round's graded final conclusion. Empty only when the
    # resolution layer could not ground one.
    decision_outcome: str = ""
    # The concrete decisive value the ``decision_outcome`` rests on, mirrored here
    # (also seeded into the deterministic checker) so the shipped round records its
    # own graded conclusion for inspection/coverage.
    resolved_value: str = ""
    # The declared LOCUS of ``resolved_value`` -- where the decisive value is meant
    # to be read: ``asset:<id>`` (vision-gated: only in the rendered pixels),
    # ``file:<path>`` (a workspace file), ``computed`` (aggregated from visible
    # sources), or ``""`` (unknown / not provided). Threaded from the resolver so
    # the post-generation confidentiality backstop can distinguish a vision round
    # (value must NOT appear in any agent-visible text) from a legitimately
    # text-grounded one. Empty when the resolution layer did not declare one.
    resolved_value_locus: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConflictVLMRoundSpec":
        return cls(
            round_id=str(data.get("round_id", "")),
            round_type=str(data.get("round_type", "")),
            visible_asset_ids=[str(item) for item in data.get("visible_asset_ids", [])],
            required_modalities=[str(item) for item in data.get("required_modalities", [])],
            expected_anchor_ids=[str(item) for item in data.get("expected_anchor_ids", [])],
            required_tool_families=[str(item) for item in data.get("required_tool_families", [])],
            instruction=str(data.get("instruction") or ""),
            update_ids=[str(item) for item in data.get("update_ids", [])],
            solution_commands=[str(item) for item in data.get("solution_commands", [])],
            mock_subagents=[MockSubagent.from_dict(m) for m in data.get("mock_subagents", []) if isinstance(m, dict)],
            decision_outcome=str(data.get("decision_outcome") or ""),
            resolved_value=str(data.get("resolved_value") or ""),
            resolved_value_locus=str(data.get("resolved_value_locus") or ""),
        )


# Recognized real-world evidence forms and grading roles for ConflictEvidenceSpec.
# Advisory only: parsing never rejects unknown values, so seeds stay forward
# compatible. The asset planner maps each source to an authentic asset kind
# (see docs/seed-design-guide.md, Principle D).
_EVIDENCE_SOURCES = frozenset({
    "dashboard_chart",
    "in_tool_diagram",
    "document",
    "log",
    "physical_photo",
    "hand_sketch",
})
_EVIDENCE_ROLES = frozenset({"decisive", "corroborating"})


@dataclass
class ConflictEvidenceSpec:
    """One artifact the scenario hinges on, described by its REAL-WORLD form.

    This lets the asset planner choose an authentic asset kind instead of
    flattening every artifact into a chart/diagram (see the seed design guide,
    Principle D). It is entirely optional and backward compatible: seeds without
    an ``evidence`` list behave exactly as before.

    Fields:
      id: Stable identifier for the evidence (referenced by summaries/anchors).
      summary: Short human description of what the artifact shows.
      real_world_source: How the evidence exists in reality; one of
        ``_EVIDENCE_SOURCES`` (advisory). The planner maps this to an asset kind
        (physical_photo->natural_image, dashboard_chart->chart, log/document->
        rich_text, in_tool_diagram->diagram, hand_sketch->sketch).
      role: Independent of modality -- any ``real_world_source`` can be either.
        ``corroborating`` (supports the decision; redundant context) or
        ``decisive`` (carries the deciding token/state the verdict rests on).
        Most photos are corroborating: a photo is not automatically the key
        evidence, just as a chart often carries redundant data. A ``decisive``
        photo is fully supported -- the deciding token/state must be legibly
        visible in the pixels (runtime strips caption/OCR text) and is checked by
        the vision fidelity audit; if it cannot be rendered legibly the asset
        auto-downgrades to ``corroborating``. Advisory; one of ``_EVIDENCE_ROLES``.
    """

    id: str
    summary: str = ""
    real_world_source: str = ""
    role: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConflictEvidenceSpec":
        return cls(
            id=str(data.get("id", "")),
            summary=str(data.get("summary", "")),
            real_world_source=str(data.get("real_world_source", "")),
            role=str(data.get("role", "")),
        )


@dataclass
class ConflictVLMSeed:
    slug: str
    domain: str
    title: str
    target_rounds: int
    core_conflict: str
    conflict_taxonomy: list[str]
    participants: list[str]
    required_constraints: list[str]
    allowed_tool_families: list[str]
    modalities: list[str] = field(default_factory=lambda: list(_DEFAULT_MODALITIES))
    generation_policy: VLMGenerationPolicy = field(default_factory=VLMGenerationPolicy)
    input_policy: VLMInputPolicy = field(default_factory=VLMInputPolicy)
    min_workspace_files: int = 5
    min_chat_sessions: int = 2
    min_chat_messages: int = 15
    difficulty_challenges: list[str] = field(default_factory=list)
    # Coarse scenario family this seed belongs to (e.g. a conflict archetype),
    # used for coverage reporting and stratified train/dev/test splits. Optional
    # and backward compatible: absent -> "" (uncategorized).
    archetype: str = ""
    # Expert-trajectory richness controls. The generated per-round trajectory must
    # realize a full Discovery -> Inspection -> Wait -> Action -> Final Verification
    # workflow with at least `min_round_steps` steps, targeting the
    # [target_round_steps_min, target_round_steps_max] range. When
    # `require_final_verification` is set, the step before DONE must re-read the
    # asset/file changed by the action step.
    min_round_steps: int = 6
    target_round_steps_min: int = 6
    target_round_steps_max: int = 10
    require_final_verification: bool = True
    # M7 difficulty knob. Defaults to a no-op so existing seeds stay
    # byte-identical (E6). ``force_vision_locus`` makes the round resolver confine
    # the decisive value to a visual asset (locus ``asset:<id>``); combined with the
    # confidentiality gate this lets the materializer promote that asset's access
    # check to ``role="required"`` (graded), so a confined round cannot be solved
    # without looking at the image. Per-round conflict density is steered by the
    # ``vision_decoy`` conflict plugin (routed via a seed's ``difficulty_challenges``
    # token), not by a global knob, so difficulty stays selective and seeds keep
    # their native conflict archetypes.
    force_vision_locus: bool = False
    # M8/CL2 async-temporal knob (L1). Defaults to a no-op so existing seeds stay
    # byte-identical (E6), modelled exactly on ``force_vision_locus``. When set,
    # the round resolver must route the round's DECISIVE value only through a
    # sub-agent that completes after a delay (``agent delegate`` ->
    # ``subagent_completion`` at ``trigger_time``), so the reference trajectory
    # must ``agent delegate ... `` -> ``wait N`` -> read the result before it can
    # answer -- a synchronous ``message send`` shortcut that returns a pre-written
    # reply is rejected. Paired with an L3 temporal check (``within_window``) this
    # makes "waited for the delegated result and answered on time" load-bearing
    # instead of cosmetic. Like ``require_protocol_completion`` it is a
    # generation/grading-side control and is stripped from the LLM-facing seed
    # payload, so enabling it never changes generated prose/asset bytes.
    force_async_wait: bool = False
    # M8 coordination-integrity knob. Defaults to a no-op so existing seeds stay
    # byte-identical (E6). When set, the seed asserts its CORE protocol mechanic
    # (its ``archetype`` plugin, e.g. ``multi_device_pairing`` ->
    # ``pairing_completed``) is actually driven to completion in the graded gold
    # trajectory: some round's ``solution_commands`` must perform the completion
    # command whose success emits that effect (``pairing confirm``), which the
    # checker scaffolder deterministically turns into a required ``exists`` effect
    # check. The strict validator (``_validate_protocol_completion``) enforces
    # this, catching the "coordination theatre" failure where the scenario
    # narrates a protocol it never completes or grades. Grading-side only: this
    # flag is stripped from the LLM-facing seed payload (steering is the plugin's
    # job), so turning it on never changes generated prose/asset bytes.
    require_protocol_completion: bool = False
    # --- Difficulty knobs M1/M2/M3 (see the audit in the round-type block of
    # ``conflict_llm_pipeline``). All three default to no-ops so existing seeds
    # stay byte-identical (E6).
    #
    # M1 (``force_round_type_plan``). The corpus came back 62.8% 'action' /
    # 3.7% 'decision' even after the round-instruction prompt was rewritten to
    # argue against that skew, because ``round_type`` is chosen by the LLM.
    # When this is set the type is assigned DETERMINISTICALLY from a fixed
    # 40-slot pattern (inv .45 / dec .25 / act .25 / ver .05) indexed by
    # ``stable_hash(slug) + round_idx``, handed to the narrative prompt as a
    # constraint, and re-asserted on the way out so it cannot drift.
    #
    # CAVEAT, measured, do not lose this: ``round_type`` is a PURE LABEL -- no
    # checker, environment gate, or plugin reads it. Forcing the label only
    # buys difficulty insofar as the generator actually writes the beat in that
    # style. The -15.0pp investigation-vs-action gap this knob is aiming at was
    # measured WITHIN seed at fixed visible-asset count (152 seed x model
    # cells, 63% same-sign); the raw marginal gap (-19.9pp) is inflated because
    # it also carries round position. Treat -15.0pp as an upper bound: the
    # observational gap is contaminated by selection (the LLM picks
    # 'investigation' for beats that are genuinely investigative), so forcing
    # the label onto a shallow beat cannot deliver the full effect.
    force_round_type_plan: bool = False
    # M2 (``force_decoy``). Materialization-side: inject a deterministic decoy
    # reading that competes with the round's decisive value. Grading is NOT
    # changed -- the existing ``field_contains``/``field_semantic_match`` checks
    # against the TRUE value do the work, so an agent that believes the decoy
    # reports the wrong value and fails them. Deliberately NOT implemented as a
    # negative check: ``not_exists`` is whole-effect-type only (it cannot filter
    # by field value) and ``field_contains`` matches messages by token
    # SUBSEQUENCE, so a "must not say X" check would misfire whenever X's tokens
    # happen to appear in order. Zero checker changes, zero brittleness.
    force_decoy: bool = False
    # M3 (``force_no_signpost``). Materialization-side: strip the sentences in
    # mock sub-agent replies that name the decisive asset outright ("please
    # verify the Lock Manager Console"). 80.8% of asset-locus rounds shipped
    # with a sub-agent pointing straight at the answer; only 2.5% were
    # assertive. This is the one knob that can render a round UNSOLVABLE, so
    # the strip is all-or-nothing per round: it is applied only if the result
    # still passes both a decisive-asset-reachability check and a
    # gold-solvability check, otherwise the round keeps its original replies.
    force_no_signpost: bool = False
    # Optional structured description of the evidence the scenario hinges on,
    # each item named by its REAL-WORLD form so the asset planner can choose an
    # authentic asset kind (Principle D). Backward compatible: absent -> [].
    evidence: list[ConflictEvidenceSpec] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source: str = "<dict>") -> "ConflictVLMSeed":
        slug = str(data.get("slug") or "").strip()
        if not _SAFE_SLUG_RE.match(slug):
            raise ValueError(f"invalid ConflictVLM seed slug in {source}: {slug!r}")
        return cls(
            slug=slug,
            domain=str(data.get("domain") or "multimodal operations"),
            title=str(data.get("title") or slug.replace("_", " ").title()),
            target_rounds=int(data.get("target_rounds") or 24),
            core_conflict=str(data.get("core_conflict") or ""),
            conflict_taxonomy=[str(i) for i in data.get("conflict_taxonomy", [])],
            participants=[str(item) for item in data.get("participants", [])],
            required_constraints=[str(item) for item in data.get("required_constraints", [])],
            allowed_tool_families=[str(item) for item in data.get("allowed_tool_families", _DEFAULT_TOOL_FAMILIES)],
            modalities=[str(item) for item in data.get("modalities", _DEFAULT_MODALITIES)],
            generation_policy=VLMGenerationPolicy.from_dict(data.get("generation_policy")),
            input_policy=VLMInputPolicy.from_dict(data.get("input_policy")),
            min_workspace_files=int(data.get("min_workspace_files") or 5),
            min_chat_sessions=int(data.get("min_chat_sessions") or 2),
            min_chat_messages=int(data.get("min_chat_messages") or 15),
            difficulty_challenges=[str(item) for item in data.get("difficulty_challenges", [])],
            archetype=str(data.get("archetype") or ""),
            min_round_steps=int(data.get("min_round_steps") or 6),
            target_round_steps_min=int(data.get("target_round_steps_min") or 6),
            target_round_steps_max=int(data.get("target_round_steps_max") or 10),
            require_final_verification=bool(
                data.get("require_final_verification", True)
            ),
            force_vision_locus=bool(data.get("force_vision_locus", False)),
            force_async_wait=bool(data.get("force_async_wait", False)),
            require_protocol_completion=bool(
                data.get("require_protocol_completion", False)
            ),
            force_round_type_plan=bool(data.get("force_round_type_plan", False)),
            force_decoy=bool(data.get("force_decoy", False)),
            force_no_signpost=bool(data.get("force_no_signpost", False)),
            evidence=[
                ConflictEvidenceSpec.from_dict(e)
                for e in data.get("evidence", [])
                if isinstance(e, dict)
            ],
        )


@dataclass
class ConflictVLMScenarioSpec:
    series: str
    scenario_slug: str
    title: str
    domain: str
    core_conflict: str
    conflict_taxonomy: list[str]
    modalities: list[str]
    input_policy: VLMInputPolicy
    generation_policy: VLMGenerationPolicy
    allowed_tool_families: list[str]
    assets: list[VLMAssetSpec]
    updates: list[ConflictScenarioUpdate]
    rounds: list[ConflictVLMRoundSpec]
    evals: list[VLMEvalSpec]
    agents: list[ConflictAgentSpec] = field(default_factory=list)
    interactions: list[ConflictInteractionSpec] = field(default_factory=list)
    workspace: ScenarioWorkspace = field(default_factory=ScenarioWorkspace)
    sessions: list[ScenarioSession] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Scenario platform vocabulary (distinct web-UI surfaces assets are rendered
    # on). Authored/sanitized upstream; surfaced here for downstream diversity
    # checks and inspection. Serialized automatically via asdict in to_dict.
    platforms: list[str] = field(default_factory=list)

    def to_dict(self, *, include_asset_content: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        if not include_asset_content:
            for asset in payload.get("assets", []):
                asset.pop("content_text", None)
                asset.pop("content_b64", None)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConflictVLMScenarioSpec":
        return cls(
            series=str(data.get("series") or "conflict_vlm_v1"),
            scenario_slug=str(data.get("scenario_slug") or ""),
            title=str(data.get("title") or ""),
            domain=str(data.get("domain") or ""),
            core_conflict=str(data.get("core_conflict") or ""),
            conflict_taxonomy=[str(i) for i in data.get("conflict_taxonomy", [])],
            modalities=[str(item) for item in data.get("modalities", [])],
            input_policy=VLMInputPolicy.from_dict(data.get("input_policy")),
            generation_policy=VLMGenerationPolicy.from_dict(data.get("generation_policy")),
            allowed_tool_families=[str(item) for item in data.get("allowed_tool_families", [])],
            workspace=ScenarioWorkspace.from_dict(data.get("workspace", {})),
            sessions=[ScenarioSession.from_dict(s) for s in data.get("sessions", [])],
            assets=[VLMAssetSpec.from_dict(item) for item in data.get("assets", []) if isinstance(item, dict)],
            updates=[ConflictScenarioUpdate.from_dict(item) for item in data.get("updates", []) if isinstance(item, dict)],
            rounds=[ConflictVLMRoundSpec.from_dict(item) for item in data.get("rounds", []) if isinstance(item, dict)],
            evals=[VLMEvalSpec.from_dict(item) for item in data.get("evals", []) if isinstance(item, dict)],
            agents=[ConflictAgentSpec.from_dict(item) for item in data.get("agents", []) if isinstance(item, dict)],
            interactions=[ConflictInteractionSpec.from_dict(item) for item in data.get("interactions", []) if isinstance(item, dict)],
            metadata=dict(data.get("metadata", {})),
            platforms=[str(p) for p in data.get("platforms", [])],
        )

    def write_json(self, path: Path, *, include_asset_content: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(include_asset_content=include_asset_content), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def read_json(cls, path: Path) -> "ConflictVLMScenarioSpec":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_conflict_vlm_seed_file(path: Path) -> ConflictVLMSeed:
    return ConflictVLMSeed.from_dict(json.loads(Path(path).read_text(encoding="utf-8")), source=str(path))


_UNCATEGORIZED_ARCHETYPE = "(uncategorized)"


def archetype_coverage(seeds: Iterable[ConflictVLMSeed]) -> dict[str, int]:
    """Tally how many seeds fall under each archetype.

    Seeds with an empty ``archetype`` are counted under ``"(uncategorized)"``.
    The result is ordered by archetype name for stable reporting, so it can back
    a coverage report and, later, archetype-stratified train/dev/test splits.
    """
    counts: dict[str, int] = {}
    for seed in seeds:
        key = (getattr(seed, "archetype", "") or "").strip() or _UNCATEGORIZED_ARCHETYPE
        counts[key] = counts.get(key, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


# Ordered list of the five top-level conflict categories from docs/task-list.md.
CONFLICT_CATEGORIES: tuple[str, ...] = (
    "main_supervised",
    "subagent_collaboration",
    "timeout",
    "order",
    "inconsistency",
)

# Seeds whose archetype token is unknown/empty are bucketed here (e.g. the
# original pre-expansion seeds that predate the 15-archetype taxonomy).
_OTHER_CATEGORY = "(other)"

# Authoritative mapping of archetype token -> top-level category, mirroring the
# five-category grouping in docs/task-list.md. This is intentionally keyed on the
# ``archetype`` token (not the seed slug or the per-plugin ``category`` field,
# whose finer 7-value vocabulary does not line up 1:1 with these five) so that
# category coverage is unambiguous.
ARCHETYPE_CATEGORY: dict[str, str] = {
    # 1. Main Agent Supervised
    "evidence_mismatch": "main_supervised",
    "scope_reconciliation": "main_supervised",
    # 2. SubAgent Collaboration
    "multi_party_synchronization": "subagent_collaboration",
    "coordinated_containment": "subagent_collaboration",
    "non_overlapping_assignment": "subagent_collaboration",
    "duplicate_hold_stale": "subagent_collaboration",
    # 3. Timeout
    "time_window_expiration": "timeout",
    "reversible_action_hold": "timeout",
    "multi_device_pairing": "timeout",
    # 4. Order
    "dependency_ordering": "order",
    "controlled_deployment_gate": "order",
    # 5. Inconsistency
    "inconsistency_expense": "inconsistency",
    "distributed_lock": "inconsistency",
    "deadlock_dependency": "inconsistency",
    "barrier_synchronization": "inconsistency",
}


def category_coverage(seeds: Iterable[ConflictVLMSeed]) -> dict[str, int]:
    """Tally how many seeds fall under each top-level conflict category.

    Each seed's ``archetype`` token is resolved to one of the five categories in
    ``CONFLICT_CATEGORIES`` via ``ARCHETYPE_CATEGORY``; seeds with an empty or
    unrecognized archetype are counted under ``"(other)"``. All five canonical
    categories are always present in the result (with a count of 0 if none map to
    them), so the report can assert full category coverage; ``"(other)"`` is
    appended only when non-empty.
    """
    counts: dict[str, int] = {cat: 0 for cat in CONFLICT_CATEGORIES}
    other = 0
    for seed in seeds:
        token = (getattr(seed, "archetype", "") or "").strip()
        category = ARCHETYPE_CATEGORY.get(token)
        if category is None:
            other += 1
        else:
            counts[category] += 1
    if other:
        counts[_OTHER_CATEGORY] = other
    return counts


def validate_conflict_vlm_scenario_spec(spec: ConflictVLMScenarioSpec) -> list[str]:
    errors: list[str] = []
    if spec.series != "conflict_vlm_v1":
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
