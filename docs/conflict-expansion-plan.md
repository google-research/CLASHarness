# Conflict-Scenario Expansion Plan

Covering all 15 conflict archetypes in [`docs/task-list.md`](./task-list.md) inside the
`conflict_vlm_v1` line of CLASHarness.

> Status: **In progress — M0–M5 are built; M6 is pending.** The M0
> foundations (`StatefulProtocolSkill` in `clasharness/skills/stateful.py`, generic
> effect tracking, a shared `ALLOWED_EFFECT_TYPES`, the scaffolder registry, and the
> `archetype` seed tag), the M1 vertical slice (the `lock` and `deploy` skills, the
> `distributed_lock` and `barrier_synchronization` plugins, and the three M1 seeds —
> `distributed_lock_shared_resource`, `barrier_sync_deployment`,
> `expense_reimbursement_conflict`), the M2 Order slice (the
> `controlled_deployment_gate` plugin and the `employee_onboarding_workflow` +
> `it_deployment_gate` seeds, both reusing the M1 `deploy` family), and the M3 Timeout
> slice (the `booking` and `pairing` skills, real virtual-time plumbing into stateful
> skills, the `reversible_action_hold` and `multi_device_pairing` plugins, and the
> three M3 seeds — `expiring_resource_hold`, `multi_device_pairing`,
> `sla_support_response`), and the M4 SubAgent-Collaboration slice (the `approval`,
> `containment`, and `review` skills, the `scope_reconciliation`,
> `multi_party_synchronization`, `coordinated_containment`,
> `non_overlapping_assignment`, and `duplicate_hold_stale` plugins, and the five M4
> seeds — `multi_party_access_permit`, `deadlock_multi_approval`,
> `coordinated_containment`, `collaborative_contract_review`,
> `multi_tab_slot_monitoring`), and the M5 Main-Supervised slice (the
> `access_request_approval` seed for #2, reusing the M4 `approval` family and the
> `scope_reconciliation` plugin; backfilled `archetype` tags on the reused
> `incident_commander_selection` (#1, `evidence_mismatch`) and `sla_support_response`
> (#7, `time_window_expiration`) seeds; and an authoritative `ARCHETYPE_CATEGORY`
> map + `category_coverage` helper + `test_archetype_coverage` enforcing coverage)
> have landed, bringing `conflict_vlm_v1` to **20 seeds** with `7/7` new skill
> families and `10/10` new plugins in place, and **all 15 archetypes present with
> 5/5 top-level categories covered** (main_supervised 2, subagent_collaboration 4,
> timeout 3, order 2, inconsistency 4; plus 5 original pre-taxonomy seeds).
> Milestones **M0–M8** are defined in §7. **M6** (scale & baseline) remains; **M7**
> (difficulty hardening) shipped as vision-graded (`force_vision_locus`, 14 seeds) +
> a `vision_decoy` information-conflict plugin (2 seeds) — see the note at the head of
> §10.7. **M8** (conflict realignment: multi-agent + temporal made load-bearing) is
> planned — see §10.8.

---

## 0. Decisions (agreed)

| Decision | Choice |
|---|---|
| **Fidelity** | **Hybrid, first-class mechanics.** The 12 coordination-protocol archetypes get real, rule-enforcing skills (deterministic + verifiable). The 2 data-conflict archetypes (#1 Incident Commander, #12 Expense) reuse the existing "stale alert vs. fresh evidence" spine. |
| **Coverage** | **15 seeds — one per archetype** (~13 net-new; reuse `incident_commander_selection` for #1, adapt `payment_chargeback_dispute` for #12). Dataset volume scales via `--num_versions`, not more seeds. |
| **Rollout** | **Vertical slice first.** M0 (foundations) → M1 (`lock`+`deploy` slice, end-to-end) → M2–M5 (remaining categories) → M6 (scale + baseline) → **M7 (difficulty hardening)**. |
| **Difficulty** | **M7 hardens the built seeds along two axes: A = grade vision (decisive value made vision-only), C = densify each round.** Regeneration of the 6 live seeds is accepted and **eval comparability with prior baselines is waived** (user decision). Genuine cross-round dependency (Axis B) is recorded as a future non-goal (§10.7 E). |

### Headline counts

| Lever | Count |
|---|---|
| New scenario seeds | **15** (≈13 net-new) |
| New command families (skills) | **7** (`approval`, `lock`, `booking`, `pairing`, `deploy`, `containment`, `review`) |
| New commands = new MCP tools | **~38** (one skill handler + one `agent_tools.v1.json` entry + one MCP tool each) |
| New conflict-type plugins (script interfaces) | **10** (+ 5 reused) |
| New effect types | **~12** |
| Generator allow-lists to widen | **2** |
| Scaffolder mappings to extend | **1** |
| Difficulty knobs (M7 shipped) | **`force_vision_locus`** (Axis A, 14 seeds) + **`vision_decoy`** plugin (replaced the planned Axis C densify, 2 evidence-mismatch seeds). The planned `min_conflicting_subagents`/`min_distractor_assets` knobs were dropped. |
| Realignment knob (M8 planned) | **`force_async_wait`** + `within_window`/`ordered_before` `EffectChecker` operators |

---

## 1. Background: how the machine works today

Understanding these four facts is required to see why the counts are what they are.

1. **A seed is a JSON recipe, not a scenario.** The generator
   ([`scripts/generate_conflict_vlm_scenarios.py`](../scripts/generate_conflict_vlm_scenarios.py))
   auto-discovers every `*.json` in
   `clasharness/data/scenario_seeds/conflict_vlm_v1/` and expands each into
   `num_versions × target_rounds` tasks via an 8-stage LLM pipeline. **Adding a
   scenario needs no code** — only new *mechanics* do. Seed dataclass:
   `clasharness/scenario/conflict_vlm_spec.py` (the `ConflictVLMSeed` dataclass).

2. **A "conflict-type script interface" is a `BaseConflictPlugin`**
   (`clasharness/scenario/plugins/base.py`). It is **not** a simulator — it
   injects per-type instructions into 5 generation layers
   (`get_layer0_instruction`, `get_layer1_instruction`, `get_layer1b_instruction`,
   `get_trajectory_instruction`, `get_evals_instruction`). Each plugin now also
   declares a `ConflictPluginSpec` (`plugins/spec.py`) and self-registers via
   `@register_conflict_plugin` (`plugins/registry.py`), mirroring the skill-side
   `SkillSpec` + `@register_skill` model; `_ALL_PLUGINS` is built from that registry.
   A plugin is matched to a seed **by exact string** in `conflict_taxonomy` /
   `difficulty_challenges` against its spec's `effective_tokens`
   (`get_plugins_for_taxonomy` in `plugins/__init__.py`). Typos silently match
   nothing — this bit us once (`db_audit` vs the plugin's real `name`
   `credential_lock`); **already fixed** (all seeds resolve every token), and
   `validate_plugins()` now fails a test on token collisions / invalid specs. See
   [`docs/conflict-plugin-guide.md`](./conflict-plugin-guide.md).

3. **Runtime mechanics are data-driven except where a first-class skill exists.**
   Conflicts are realized via `background_events` (`file_update`, `session_append`,
   `session_redaction`, `subagent_completion`) with `trigger_time`, plus
   `mock_subagents` (`virtual_duration_seconds` = delay, `state_effects` = commands).
   When this plan started there was **no** lock / barrier / pairing state machine and
   **no IAM / approval system**; M1 has since added the `lock` and `deploy` (barrier)
   families and M3 the `booking` (reversible hold) and `pairing` (handshake) families,
   all as real, rule-enforcing skills. The remaining families (`approval`,
   `containment`, `review`) are what the rest of this plan adds.

4. **The verification bottleneck.** The generator only permits rounds to emit
   checks of type `effect{messages_sent, assets_accessed, files_created}`,
   `file_check`, `command_check` — hard-coded in **two** validators
   (`conflict_llm_pipeline.py`: `_validate_generated_round_steps` and
   `_validate_generated_round_trajectory_and_evals_compiled`). The checker
   *scaffolder* (`_scaffold_round_checkers` → `_classify_step_command`) only knows
   the message / file / asset verb frozensets (`_FILE_WRITE_COMMAND_NAMES` etc.);
   every other verb is classified `plain` and gets **no checker**. Meanwhile
   `EffectChecker`'s generic path already handles **any** `effect_type`
   (`clasharness/evaluation/checkers/effect_checker.py`). So new primitives are
   verifiable *only after* we (a) record their effects and (b) teach the scaffolder
   + widen the two allow-lists (both done structurally in M0).

### Data flow (unchanged by this plan)

```
seed JSON ──load_conflict_vlm_seed_file──▶ ConflictVLMSeed
   │  (conflict_taxonomy + difficulty_challenges → plugins → prompt hooks)
   ▼
generate script → materialize_conflict_scenario → ConflictLLMPipeline.generate
   │      (8 stages: screenplay → agents → background → rounds → audit → assets)
   ▼
ConflictVLMScenarioSpec → _compile_round_to_task (one Task per round)
   ▼
ScenarioSpec + per-round tasks/<id>/specs.json  →  train_and_eval.py (--task-data-dir)
```

---

## 2. The 5×15 taxonomy → build mapping

| # | Category / Archetype | Existing? | Plugin | New command | Verified by |
|---|---|---|---|---|---|
| 1 | **Main-Supervised** / Incident Commander Selection | ✅ seed | evidence_mismatch + dependency_ordering *(reuse)* | — (agent/message) | messages_sent + command_check |
| 2 | Access Request Approval | ✗ | scope_reconciliation *(new)* | `approval` | `approval_granted` effect |
| 3 | **SubAgent-Collab** / Multi-Party Access Permit | ✗ | multi_party_synchronization *(new)* | `approval` (time-window) | simultaneity effect + file_check |
| 4 | Coordinated Multi-Point Containment | ✗ | coordinated_containment *(new)* | `containment` | `containment_action` (all 4 points) |
| 5 | Collaborative Contract Review | ✗ | non_overlapping_assignment *(new)* | `review` (+ `lock`) | `document_signed` + command_check |
| 6 | Multi-Tab Slot / Stock Monitoring | ✗ | duplicate_hold_stale *(new)* + temporal_staleness *(reuse)* | `booking` | single-hold effect + command_check |
| 7 | **Timeout** / SLA Support Response | ✅ seed | time_window_expiration *(reuse)* | — (tasks/message + wait) | holding msg before deadline |
| 8 | Expiring Resource Hold | ✅ seed | reversible_action_hold *(new)* + time_window *(reuse)* | `booking` | hold made, NOT prematurely confirmed |
| 9 | Multi-Device Setup / Pairing | ✅ seed | multi_device_pairing *(new)* | `pairing` | `pairing_completed` before code expiry |
| 10 | **Order** / Employee Onboarding | ✅ seed | dependency_ordering *(reuse)* | `deploy` (gate) | command-order + gate effect |
| 11 | IT Change / Deployment Gate | ✅ seed | controlled_deployment_gate *(new)* | `deploy` | cutover only after approvals |
| 12 | **Inconsistency** / Expense Reimbursement | ~payment | evidence_mismatch *(reuse)* | — (file/asset) | file_check(amount) + messages_sent |
| 13 | Distributed Lock for Shared Resource | ~warehouse | distributed_lock *(new)* | `lock` | acquire/release order, no overwrite |
| 14 | Deadlock in Multi-Approval | ~warehouse | deadlock_dependency *(reuse)* | `approval` (provisional/escalate) | `approval_escalated` + messages_sent |
| 15 | Barrier Synchronization for Deployment | ~warehouse | barrier_synchronization *(new)* | `deploy` (barrier cutover) | cutover iff all gates ready |

**New plugins (10):** scope_reconciliation, multi_party_synchronization,
coordinated_containment, non_overlapping_assignment, duplicate_hold_stale,
reversible_action_hold, multi_device_pairing, controlled_deployment_gate,
distributed_lock, barrier_synchronization.
**Reused plugins (5):** evidence_mismatch, dependency_ordering,
time_window_expiration, temporal_staleness, deadlock_dependency.

---

## 3. Deliverable A — 7 new skills (commands / MCP tools)

All 7 inherit a new **`StatefulProtocolSkill`** base (see §6) providing: seed-load
from `state_dir/<family>.json` (following the `tasks_skill.py` / `email_skill.py`
preload precedent), a virtual-time expiry helper, and effect emission via
`CommandResult.state_changes` (`clasharness/backend/base.py`). Snapshot/undo is
**already** inherited from the base `Skill` (`_snapshot_attrs`, `to_state`/`from_state`).

Convention: a command that violates its protocol returns `exit_code=1` with a
clear `stderr` (the environment *enforces* the rule); a successful outcome emits an
effect the evaluator can assert on.

**Add-a-Family vertical stack (reused by M1–M5).** Each new command family is the
same 15-step stack; milestones in §7 just name the family/plugin/seed:

1. `clasharness/skills/impl/<family>_skill.py`: subclass `StatefulProtocolSkill`
   **and attach a `SPEC = SkillSpec(...)`** (category, `prefixes`, one `CommandSpec` per
   verb, `real_kind`, `smoke`) — see `lock_skill.py` / `deploy_skill.py`. The `SPEC` is
   **mandatory**: `SkillRegistry.validate_specs(require_spec=True)` fails the build
   (`<Skill>: missing SPEC`) without it, and it drives conformance + the per-skill
   `to_mcp_tools()` surface.
2. Re-export in `clasharness/skills/impl/__init__.py`.
3. Register in `clasharness/backend/multi_app_backend.py` (import + instantiate + register block). *(Core skills register here; the additive `@register_skill` / `contrib/` extension layer — see §10.1 — is for out-of-tree skills, not the 15 archetype families.)*
4. Add family + `prefixes` to `clasharness/specs/agent_tools.v1.json` (auto-allowlisted by `safety_guard.py`).
5. Add a `normalize_family_name` alias in `clasharness/tools/agent_tool_spec.py` if the family name needs mapping.
6. **MCP surface = curated facade (not a hand-edited server).** Add the family's tools to
   `clasharness/backend/mcp_facade.py` (`TOOL_SCHEMAS` + `build_cli` dispatch) — or
   regenerate with `scratch/gen_mcp_facade.py` — then refresh the two frozen goldens
   `tests/fixtures/mcp_tools_golden.json` + `tests/fixtures/mcp_dispatch_golden.json`.
   `mcp_server.py` consumes the facade via `mcp_facade.build_tools_list()` / `build_cli()`
   (`MCP_TOOLS` is a back-compat alias), so it needs **no** per-tool edit. Pinned by
   `tests/test_mcp_facade.py`.
7. `clasharness/backend/openclaw_compat.py::_rewrite_mcp_tool_verb` if using agent-verb style.
8. Add the family's effect types to `ALLOWED_EFFECT_TYPES` (the M0 constant).
9. `environment.py::_track_effects` generic path already records them (M0) — confirm buckets.
10. Scaffolder verb-map in `conflict_llm_pipeline.py` (`_classify_step_command` + frozensets + `_scaffold_round_checkers`).
11. Seed initial state via `conflict_vlm_materializer.py` (`initial_state_overrides` → `<family>.json`).
12. Tests: `tests/test_<family>_preload.py` + an undo-roundtrip case. When the family adds
    MCP tools, also re-run the surface guards after refreshing the goldens (step 6):
    `tests/test_mcp_facade.py`, `tests/test_mcp_server.py`, `tests/test_skill_registry.py`,
    `tests/test_skill_runtime_dispatch.py`, `tests/test_stateful_skill.py`.
13. Plugin(s): `clasharness/scenario/plugins/<name>.py` + register in `plugins/__init__.py`. **Pre-flight:** the seed's `conflict_taxonomy` / `difficulty_challenges` tokens must equal the plugin's canonical `.name` exactly (see §10.2).
14. Seed(s): `clasharness/data/scenario_seeds/conflict_vlm_v1/<slug>.json` with the `archetype` tag.
15. Generate (mock → vertex) → `validate_conflict_vlm_data.py` → expert/flash/pro eval.

### A.1 `approval` — access grants & multi-party approvals (#2, #3, #14)
State: approvals `{approver_role, scope, status(pending|approved|rejected|provisional), granted_at, ttl_seconds, expires_at}`.
```
approval request  --resource <r> --scope <s> --approvers <roleA,roleB,...>
approval approve  --resource <r> --approver <role> [--scope <s>] [--ttl <sec>]
approval grant    --resource <r>          # exit 1 unless ALL required approvals valid & unexpired & scopes reconcile
approval revoke   --resource <r> [--approver <role>]
approval provisional --resource <r> --approver <role>     # deadlock-breaking (#14)
approval escalate --resource <r> --to <role>              # deadlock-breaking (#14)
approval status   [--resource <r>]
```
Effects: `approval_granted` (`{resource, scope}`), `approval_escalated`.

### A.2 `lock` — distributed / section locks (#13, #5)
State: `resource[/section] -> holder` (exclusive).
```
lock acquire --resource <r> [--section <s>]   # exit 1 if held by another
lock release --resource <r> [--section <s>]
lock list
lock status  --resource <r>
```
Effects: `lock_acquired`, `lock_released`.

### A.3 `booking` — slots/stock + reversible expiring holds (#6, #8)
State: inventory availability (mutated by background events → stale-state), holds with expiry, confirmed bookings.
```
booking search  [--resource <r>]           # availability may be stale
booking hold    --resource <r> [--ttl <sec>]  # reversible; exit 1 if taken (duplicate avoidance)
booking confirm --resource <r>             # irreversible; requires a live (unexpired) hold
booking release --resource <r>
booking extend  --resource <r> --ttl <sec>
booking status  [--resource <r>]
```
Effects: `booking_held`, `booking_confirmed`, `booking_released`.

### A.4 `pairing` — multi-device handshake (#9)
State: two endpoints; a code with lifetime; FSM `idle→code_shown→entered→confirmed`, plus `expired`.
```
pairing start      --device <A>            # generates code + expiry
pairing show-code  --device <A>
pairing enter-code --device <B> --code <c> # must match & be unexpired
pairing confirm    --device <B>
pairing status     [--device <d>]
```
Effect: `pairing_completed`.

### A.5 `deploy` — deployment gate + barrier cutover (#11, #15, #10)
State: readiness gates `{name -> pending|ready|failed|stale|expired}` (usually driven by background events / mock agents), deployment status, approvals, monitoring.
```
deploy request  --change <c>
deploy approve  --change <c> --approver <role>
deploy gate-set --gate <g> --state <ready|...>
deploy execute  --change <c>               # ordered; blocks premature deploy
deploy cutover  --change <c>               # BARRIER: exit 1 unless ALL gates ready simultaneously
deploy rollback --change <c>
deploy monitor  --change <c>               # post-deploy monitoring gate
deploy status   [--change <c>]
```
Effects: `deployment_executed`, `deployment_cutover`, `deployment_rolledback`.

### A.6 `containment` — coordinated multi-point security (#4)
State: 4 control points, each `active|contained`. Success = all four actioned.
```
containment revoke-sessions    --account <a>
containment disable-forwarding --account <a>
containment isolate-device     --device <d>
containment block-token        --token <t>
containment status
```
Effect: `containment_action` (verify `count_gte` across all points).

### A.7 `review` — collaborative review + sign gate (#5)
State: review aspects `{legal, privacy, security, business, layout}` each `{assignee, status, blocking_issues}`; document sign state.
```
review assign  --aspect <a> --reviewer <role>       # non-overlapping
review submit  --aspect <a> --status <ok|blocked> [--issue <text>]
review sign    --document <d>                       # exit 1 unless all aspects ok & no blockers
review status
```
Effects: `review_submitted`, `document_signed`.

**Files touched per family:** see the 15-step vertical stack above.

---

## 4. Deliverable B — 10 new conflict plugins

Each is a `BaseConflictPlugin` subclass in `clasharness/scenario/plugins/<name>.py`
that **declares a `ConflictPluginSpec` (`SPEC`) and self-registers via
`@register_conflict_plugin`** (imported in `plugins/__init__.py`; no
hand-maintained list). It implements the relevant prompt hooks so the LLM builds a
scenario around the mechanic and emits a trajectory + checks that exercise the
matching skill. The `SPEC` is the plugin analogue of a skill's `SkillSpec`: it
declares the taxonomy tokens the plugin binds to, the skill families its trajectory
exercises (`skill_families`), the effect keys its evals assert (`emits_effects`),
and which hooks it implements — driving routing, conformance (`validate_plugins`),
and docs. Authoring steps + a copy-paste template are in
[`docs/conflict-plugin-guide.md`](./conflict-plugin-guide.md).

| Plugin `name` | Drives | Key hooks |
|---|---|---|
| `scope_reconciliation` | #2 | layer0 (conflicting approval scopes) + trajectory + evals (`approval grant`) |
| `multi_party_synchronization` | #3 | layer0 (simultaneous validity window) + evals (grant only while all valid) |
| `coordinated_containment` | #4 | layer0 (4 control points) + trajectory + evals (all points actioned) |
| `non_overlapping_assignment` | #5 | layer0 (non-overlapping roles/sections) + evals (`review sign` gate) |
| `duplicate_hold_stale` | #6 | layer0 (parallel monitoring, stale availability) + evals (single `booking confirm`) |
| `reversible_action_hold` | #8 | layer0 (reversible hold vs irreversible confirm) + evals (no premature confirm) |
| `multi_device_pairing` | #9 | layer0 (code lifetime, mismatch, reset) + evals (`pairing_completed`) |
| `controlled_deployment_gate` | #11 | layer0 (change workflow order) + evals (execute after approvals) |
| `distributed_lock` | #13 | layer0 (shared resource, acquire-before-edit) + evals (acquire/release + no overwrite) |
| `barrier_synchronization` | #15 | layer0 (readiness gates) + evals (`deploy cutover` iff all ready) |

Note: the earlier `db_audit` → `credential_lock` taxonomy-token mismatch in
`multimodal_temporal_conflict_01.json` is **already fixed**; no action here.

---

## 5. Deliverable C — 15 seeds

One `ConflictVLMSeed` JSON per archetype in
`clasharness/data/scenario_seeds/conflict_vlm_v1/`, authored per
[`docs/seed-design-guide.md`](./seed-design-guide.md) (scale-agnostic phase
constraints, active-coordinator design, non-anthropomorphic framing). Each seed
sets `conflict_taxonomy` / `difficulty_challenges` to the exact plugin `name`s
above, `allowed_tool_families` including the new family, and a new **`archetype`**
tag (see §6) for coverage reporting.

| Seed slug | Archetype | taxonomy tokens | new tool family |
|---|---|---|---|
| `incident_commander_selection` *(exists)* | #1 | evidence_mismatch, dependency_ordering | — |
| `access_request_approval` | #2 | scope_reconciliation | approval |
| `multi_party_access_permit` | #3 | multi_party_synchronization, time_window_expiration | approval |
| `coordinated_containment` | #4 | coordinated_containment | containment |
| `collaborative_contract_review` | #5 | non_overlapping_assignment | review, lock |
| `multi_tab_slot_monitoring` | #6 | duplicate_hold_stale, temporal_staleness | booking |
| `sla_support_response` | #7 | time_window_expiration | — |
| `expiring_resource_hold` | #8 | reversible_action_hold, time_window_expiration | booking |
| `multi_device_pairing` | #9 | multi_device_pairing, time_window_expiration | pairing |
| `employee_onboarding_workflow` | #10 | dependency_ordering | deploy |
| `it_deployment_gate` | #11 | controlled_deployment_gate | deploy |
| `expense_reimbursement_conflict` *(adapt payment)* | #12 | evidence_mismatch | — |
| `distributed_lock_shared_resource` | #13 | distributed_lock | lock |
| `deadlock_multi_approval` | #14 | deadlock_dependency | approval |
| `barrier_sync_deployment` | #15 | barrier_synchronization | deploy |

**Volume math:** `#tasks = #seeds × num_versions × target_rounds`. At 15 seeds ×
~7 versions × ~10 rounds ≈ **~1,000 conflict tasks**.

---

## 6. Deliverable D — supporting "glue" (the essential non-obvious work)

1. **`StatefulProtocolSkill` base** (new `clasharness/skills/stateful.py`):
   state-file load from `state_dir/<family>.json`, virtual-time expiry helper,
   effect emission. Snapshot/undo is **already** provided by the base `Skill`
   (`_snapshot_attrs`, `to_state`/`from_state`) — do not reimplement.
2. **Effect recording** (`clasharness/core/environment.py`): make `_track_effects`
   generic — map any `state_changes["effects"][<type>]` into `self._effects[<type>]`
   (lazily create the bucket) instead of the current hard-coded messages/agents
   handling; `reset()` already clears `_effects` generically.
3. **Generator wiring** (`clasharness/scenario/conflict_llm_pipeline.py`):
   - extend the verb frozensets + `_classify_step_command` / `_scaffold_round_checkers`
     so new verbs map to their effect / `command_check` checkers;
   - widen **both** allow-lists (`_validate_generated_round_steps`,
     `_validate_generated_round_trajectory_and_evals_compiled`) — replace the inline
     literals with a shared `ALLOWED_EFFECT_TYPES` constant.
4. **Seeding** (`clasharness/scenario/conflict_vlm_materializer.py`): emit each
   skill's initial state file into `initial_state_overrides` (materialized to
   `state_dir` by `environment.reset()`), mirroring `emails.json` / `tasks.json`.
5. **Seed schema** (`clasharness/scenario/conflict_vlm_spec.py`): add an optional
   `archetype` string tag; add a category/archetype **coverage report** (the
   conflict line currently has none).
6. **MCP** (`clasharness/backend/mcp_facade.py`): add `TOOL_SCHEMAS` + `build_cli`
   dispatch for the ~38 new commands (or regenerate via `scratch/gen_mcp_facade.py`),
   then refresh the `tests/fixtures/mcp_{tools,dispatch}_golden.json` goldens.
   `mcp_server.py` (`MCP_TOOLS = mcp_facade.build_tools_list()`) picks them up with no
   per-tool edit; pinned by `tests/test_mcp_facade.py`. *(The facade is a curated,
   ergonomically-renamed surface — `create_calendar_event` vs the CLI's
   `calendar add-event` — mapped to commands by `build_cli` and cross-checked against
   registered skill prefixes by `test_projected_commands_route_to_registered_skill`.)*
7. **Strict-validator rules** (`clasharness/scenario/conflict_strict_validator.py`)
   per new type, mirroring the `temporal_staleness` pattern — **batched in M6**.

---

## 7. Phased implementation — milestones M0–M8

Each milestone M1–M5 applies the **Add-a-Family vertical stack** (§3) to its
families, adds its plugin(s) + seed(s), then generates → validates → evaluates.
References are given as **function/symbol anchors** (the pipeline file moves often).

**Cross-milestone decisions (locked in review):**
- **M1 is a minimal slice** — `lock` + `deploy` only. Seed #14
  (`deadlock_multi_approval`) needs `approval` and is built in **M4**, not M1.
- **Strict-validator rules** are **deferred to M6**; M1–M5 rely on
  effect / `file_check` / `command_check`.
- **Splits**: the identical-split + cross-run accumulation behavior is fixed in
  **M6**, which also adds archetype-stratified train/dev/test.

### M0 — Foundations (no new scenarios) — ✅ **Built** — ~2-3 days
*Goal:* build reusable machinery so later families are pure content; lock a
regression gate. **M0 changes nothing observable.**
1. **`clasharness/skills/stateful.py` (new) — `StatefulProtocolSkill`:** state-file
   load from `state_dir/<family>.json` (per `tasks_skill.py` / `email_skill.py`
   preload precedent), virtual-time expiry helper, effect-emit helper packing
   `CommandResult.state_changes["effects"][<type>]`. (Snapshot/undo inherited.)
2. **`core/environment.py` — generic `_track_effects`:** map any
   `state_changes["effects"][<type>]` → `self._effects[<type>]` (lazy bucket);
   confirm `reset()` clears generically.
3. **`conflict_llm_pipeline.py` — shared `ALLOWED_EFFECT_TYPES` constant** near the
   verb frozensets; replace the two inline literals in `_validate_generated_round_steps`
   and `_validate_generated_round_trajectory_and_evals_compiled`. Initial value =
   current set (no new types yet).
4. **Scaffolder registry:** refactor `_classify_step_command` + the verb frozensets
   (`_DELEGATION_COMMAND_NAMES`, `_FILE_WRITE_COMMAND_NAMES`,
   `_FILE_READ_COMMAND_NAMES`, `_ASSET_ACCESS_COMMAND_NAMES`) and
   `_scaffold_round_checkers` into a table new verbs plug into; unknown verbs keep
   today's `plain` (no-checker) fall-through.
5. **`conflict_vlm_spec.py` — `archetype` tag:** add optional `archetype: str = ""`
   to `ConflictVLMSeed` (field + `from_dict`); add a category/archetype coverage
   report helper.

*Exit (regression gate):*
- `python3 -m py_compile` clean on all touched modules.
- `pytest tests/test_conflict_pipeline.py tests/test_conflict_vlm_reconcile.py
  tests/test_state_isolation_fixes.py tests/test_llm_client.py tests/test_llm_auditor.py tests/test_evaluators.py
  tests/test_pro_vlm_generation.py tests/test_train_and_eval_cli.py` all green.
- **MCP/skill surface guards** (required whenever a milestone adds a family or MCP
  tool — see §10.1): `pytest tests/test_mcp_facade.py tests/test_mcp_server.py
  tests/test_mcp_categorized_tools.py tests/test_skill_registry.py tests/test_skill_runtime_dispatch.py
  tests/test_stateful_skill.py tests/test_skill_extensions.py tests/test_agent_tool_spec.py` all green.
- Mock-generate one existing seed + one **vertex** seed; specs equivalent to pre-M0
  (no new effects, same checks).

### M1 — Vertical slice: `lock` + `deploy` (Inconsistency #12,#13,#15) — ✅ **Built**
*Skills (vertical stack ×2):* `lock` (`acquire|release|list|status`; effects
`lock_acquired`/`lock_released`), `deploy` (`request|approve|gate-set|execute|
cutover|rollback|monitor|status`; effects `deployment_executed`/`deployment_cutover`/
`deployment_rolledback`).
*Plugins:* `distributed_lock` (#13), `barrier_synchronization` (#15); reuse
`evidence_mismatch` (#12).
*Seeds:* `expense_reimbursement_conflict` (adapt `payment_chargeback_dispute`),
`distributed_lock_shared_resource`, `barrier_sync_deployment` — each with `archetype`
+ `allowed_tool_families` incl. the new family.
*M0 constant/materializer:* add the new effect types to `ALLOWED_EFFECT_TYPES`;
`lock.json` / `deploy.json` seeding; scaffolder verb-maps for `lock`/`deploy`.
*Tests:* `test_lock_preload.py`, `test_deploy_preload.py`, undo-roundtrip.
*Exit:* 3 seeds generate mock → vertex (`--resume`/`--retries`); 0 allow-list
rejections; `validate_conflict_vlm_data.py` clean (no FLAGGED/NON-DISC/UNDER-K);
expert(gold) ≈ 1.0; flash/pro sane spread; full test suite green.

### M2 — Order (#10, #11) — ✅ **Built**
*No new skill* (reuse `deploy`). *Plugin:* `controlled_deployment_gate` (#11); reuse
`dependency_ordering` (#10). *Seeds:* `employee_onboarding_workflow` (#10, taxonomy
`dependency_ordering`), `it_deployment_gate` (#11, taxonomy **both**
`controlled_deployment_gate` + `dependency_ordering`). *Default:* `deploy` covers the
onboarding gate; revisit only if #10 reads unnaturally. *Exit:* same eval bar as M1.
**Full detailed execution plan (files, 15-step mapping, exact commands): §10.4.**

### M3 — Timeout (#7, #8, #9) — ✅ **Built**
*Skills:* `booking` (`search|hold|confirm|release|extend|status`), `pairing`
(`start|show-code|enter-code|confirm|status`). *Plugins:* `reversible_action_hold`
(#8), `multi_device_pairing` (#9); reuse `time_window_expiration` (#7). *Seeds:*
`sla_support_response` (#7, no new family), `expiring_resource_hold` (#8),
`multi_device_pairing` (#9). *Exit:* same eval bar; virtual-time expiry verified
(holds/codes expire on the clock). **Landed:** both families + the D1 virtual-time
plumbing, both plugins, all three seeds, the full 15-step glue stack (MCP facade
`_CURATED_TOOL_COUNT=88`), and four unit-test files; `426` targeted unit/spec/facade
tests green and the mock smoke materializes all three seeds (3/3, F1 gold-check
passes). The remaining exit-gate steps are the credential-bound Vertex generation +
replay audit + eval (§10.5.C.4–C.6).
**Full detailed execution plan (D1 virtual-time plumbing, files, decisions,
phases, exact commands): §10.5.**

### M4 — SubAgent-Collaboration (#3,#4,#5,#6) + #14 — ✅ **Built**
*Skills:* `approval` (`request|approve|grant|revoke|provisional|escalate|status`;
effects `approval_granted`/`approval_escalated`), `containment`
(`revoke-sessions|disable-forwarding|isolate-device|block-token|status`; effect
`containment_action`), `review` (`assign|submit|sign|status`; effects
`review_submitted`/`document_signed`). *Plugins:* `scope_reconciliation`,
`multi_party_synchronization`, `coordinated_containment`, `non_overlapping_assignment`,
`duplicate_hold_stale`. *Seeds:* `multi_party_access_permit` (#3),
`coordinated_containment` (#4), `collaborative_contract_review` (#5, uses
`review`+`lock`), `multi_tab_slot_monitoring` (#6, uses `booking`),
**`deadlock_multi_approval` (#14, moved from M1)** — reuse `deadlock_dependency`.
*Default:* `approval` serves #14's provisional/escalate; kept as one family. *Exit:*
same eval bar; `containment` verified by `count_gte` across all 4 points.
**Landed:** all three families + their `SPEC`s, the five plugins (all registered;
token pre-flight clean), all five seeds, the full 15-step glue stack (MCP facade
`_CURATED_TOOL_COUNT=88→104`, both goldens refreshed), and three preload test files;
the targeted MCP/skill/plugin cluster (168 tests) and the MCP facade parity suite
(6 tests) are green, and the mock smoke materializes all five seeds (5/5, 50/50 F1
gold-check rounds clean). The remaining exit-gate steps are the credential-bound
Vertex generation + replay audit + eval (§10.6.C.5).
**Full detailed execution plan (decisions, files, 15-step mapping, phased rollout,
exact commands): §10.6.**

### M5 — Main-Supervised (#1, #2) — ✅ Built
Mostly reuse. *Seeds:* `access_request_approval` (#2, uses the M4 `approval`
family); reuse `incident_commander_selection` (#1) + add its `archetype` tag.
*Exit:* all 15 archetypes present; coverage report shows 5/5 categories.

**Landed:** the `access_request_approval` seed (#2) built on the M4 `approval`
family and the `scope_reconciliation` plugin (no new skill/plugin/MCP tool); the
reused `incident_commander_selection` (#1 → `evidence_mismatch`) and
`sla_support_response` (#7 → `time_window_expiration`) seeds each backfilled with
their `archetype` tag; and a new coverage layer in `conflict_vlm_spec.py` — the
authoritative `ARCHETYPE_CATEGORY` token→category map (15 tokens over the five
`CONFLICT_CATEGORIES`), a `category_coverage(seeds)` helper, and the
`tests/test_archetype_coverage.py` suite that enforces 15/15
archetype coverage and 5/5 category coverage. `conflict_vlm_v1` now holds **20
seeds**; all 15 archetypes are present and every top-level category is covered
(main_supervised 2, subagent_collaboration 4, timeout 3, order 2, inconsistency 4,
plus 5 original pre-taxonomy seeds under `(other)`). Offline gates green: token
pre-flight clean, mock smoke materializes the new seed (10/10 F1 gold rounds
clean), the coverage suite passes (7/7), and the full unit suite is green apart
from 4 pre-existing unrelated failures (1 `test_environment` system-skill
migration, 3 the degenerate-PNG Chrome render test). Credential-bound Vertex
generation + eval remain the user's to run.

### M6 — Scale & baseline
1. **Fix the split writer** (`generate_conflict_vlm_scenarios.py`, split-write
   block): stop writing all 4 splits identically and stop accumulating across runs;
   add **archetype-stratified** train/dev/test.
2. **Strict-validator rules** — add per-type structural enforcement in
   `conflict_strict_validator.py` for every new type (batched here), mirroring the
   `temporal_staleness` pattern.
3. **Scale generation** — bump `--num_versions`; full run with `--resume`,
   `--retries`, larger `--auditor_max_iterations`.
4. **Baselines & report** — emit archetype coverage report; run flash/pro over the
   full set; compare to the then-current **26.7% / 32.4%** (pre-M7 snapshot) and flagged
   counts.
*Exit:* stratified splits reproducible; coverage 15/15; baseline numbers recorded in
`CHANGELOG.md`.

### M7 — Difficulty hardening (vision-graded + densified conflict)
Cross-model eval shows the current seeds are easier than intended (**pre-M7 snapshot:**
opus-class 43.9%, sonnet-class 46.7%, ~8.7 steps). Root causes: (a) vision/asset access is graded
`diagnostic`-only (never gates success), and (b) each round is a self-contained,
low-density conflict (protocol state resets per round; ~3 conjunctive checks).
Two agreed axes: **A = make vision actually graded**, **C = densify each round.**
Regeneration of the 6 current seeds is accepted; **comparability with prior eval
baselines is explicitly waived** (user decision). Detailed plan in §10.7.
*Exit:* opus/sonnet drop from ~44–47% toward the intended <40% band; a
vision-blind ablation (strip images) collapses pass rate, proving vision is
load-bearing; gold still self-solves (≈1.0).
*Outcome:* **met, with margin.** On the current corpus the strongest cell in the
nine-model panel is `opus-5` at **25.3%** full-pass — well inside the intended `<40%`
band. Current numbers live in
[evaluation.md](evaluation.md#reference-results); the figures above are retained as the
pre-M7 motivation, not as current results.

> **What actually shipped for M7:** Axis A (`force_vision_locus`) landed on the 14
> seeds that own a decisive image. Axis C ("densify") was **superseded** — instead of
> the `min_conflicting_subagents`/`min_distractor_assets` knobs (dropped), an
> information-conflict `vision_decoy` plugin was added to the 2 `evidence_mismatch`
> seeds (`expense_reimbursement_conflict`, `distributed_lock_shared_resource`). See
> the note at the head of §10.7.

### M8 — Conflict realignment: multi-agent + temporal made load-bearing (planned)
A three-way generation audit found the built seeds under-exercise the very mechanics
the proposal is about. Across the 18 generated packages there are **0** `wait`
commands; sub-agents are static `mock_subagents` answered **synchronously** by
`message send` (ignoring `virtual_duration_seconds`); temporal "conflict" is a baked
timestamp read once (no runtime state machine for deadlock/time-window); and grading
is dominated by asset/file/message **counts** while domain/coordination actions are
barely asserted. Yet the runtime already has the right primitives — `agent delegate`
+ `wait` + a virtual clock (`core/agent_manager.py:48-65`,
`core/environment.py:774-790`) and 7 fail-closed protocol skills. M8 makes those
primitives **load-bearing and gradeable** along three levers, keeping the vision axis
(M7) as a frozen secondary axis:
- **L1 — force async trajectories:** a new `force_async_wait` seed knob routes the
  decisive value through a slow sub-agent so the reference path MUST `agent delegate`
  → `wait N` → read the completed result, instead of a synchronous `message send`.
- **L2 — grade coordination:** assert the protocol/domain effects themselves (the
  fail-closed skills already gate them at runtime), not just asset/file/message counts.
- **L3 — temporal grader:** stamp effects with `virtual_time` and add `within_window`
  / `ordered_before` operators to `EffectChecker` so "responded before the SLA window"
  and "handshake steps in order" become checkable.
First vertical slice = **Cap 2 (temporal/async)** on `sla_support_response` +
`multi_device_pairing`. Scope is **L1+L2+L3 only**; real concurrent sub-agents (L4)
and cross-round compounding (Axis B / L5) stay explicit non-goals. Detailed plan in §10.8.
*Exit:* the gold trajectory delegates-then-waits and passes; a **no-wait** ablation
(act on the synchronous stale value) and an **out-of-order** ablation collapse pass
rate ⇒ async + ordering are load-bearing; non-opted seeds stay byte-identical.

---

## 8. Effort & risk

| | Estimate |
|---|---|
| M0 | 2-3 days |
| 7 skills (skill + tests + MCP + spec + scaffolder + effect) | 5-7 days |
| 10 plugins | ~5 days |
| 15 seeds (author + generate/validate iteration) | 7-8 days |
| Integration + eval + guardrails | ~1 week |
| **Total (hybrid)** | **~4-6 weeks** |

**Risks & mitigations**
- *Generation reliability* — ~27% raw scenario loss observed
  ([`docs/analysis/generator-failure-modes.md`](./analysis/generator-failure-modes.md));
  use `--phase_expand_attempts`, `--retries`, `--resume`, larger `--auditor_max_iterations`.
- *Allow-list / scaffolder regressions* — mitigated by the M0 regression gate.
- *LLM-judge nondeterminism* in `field_semantic_match` — prefer deterministic
  effect/`file_check`/`command_check` checks for the new primitives; reserve the
  judge for messaging criteria.
- *Vertex cost* — generation is `~8 + ceil(R/P) + K*R + A` calls per version × 15 seeds
  × versions, where `7 ≤ K ≤ 8` (7 guaranteed round sub-calls — instruction, mocks,
  updates, assets, resolution, outline, commands — plus 1 conditional semantic-checker
  call); budget accordingly and generate incrementally per milestone. The decisive-asset
  vision audit adds up to 1 vision call per decisive asset (plus bounded re-renders);
  `--novision_audit` disables it.

---

## 9. Design decisions (resolved during review)

1. **M1 scope / #14 dependency** — **Resolved:** M1 is a minimal `lock`+`deploy`
   slice (#12,#13,#15). #14 `deadlock_multi_approval` needs `approval` → moved to **M4**.
2. **Strict-validator rules** — **Resolved:** defer per-type structural enforcement
   to **M6**; M1–M5 rely on effect / `file_check` / `command_check`.
3. **Split strategy** — **Resolved:** the generator writes all 4 splits identically
   and accumulates across runs unless `--clean` (split-write block in
   `generate_conflict_vlm_scenarios.py`). **M6** fixes this and adds
   archetype-stratified train/dev/test.
4. **Family consolidation (default; revisit if needed)** — `deploy` also covers the
   onboarding gate (#10); `approval` also serves #14's provisional/escalate. Kept as
   designed (no separate `workflow` family, no unified "gate" primitive) unless a
   milestone shows it reads unnaturally.

---

## 10. Addenda — architecture drift & out-of-scope directions (post-M1)

Captured after an M0/M1 code audit: things the plan did not originally describe.
§10.1 items are now reflected in §3/§6; §10.3 records directions that were considered
and deliberately excluded, so future readers know they were weighed (not forgotten).

### 10.1 Architecture updates since the plan was written (now folded into §3/§6)

1. **MCP surface is a curated facade + frozen goldens, not a hand-edited server.** The
   served `tools/list` / `tools/call` are built by `clasharness/backend/mcp_facade.py`
   (`TOOL_SCHEMAS`, `build_tools_list()`, `build_cli()`), generated by
   `scratch/gen_mcp_facade.py` and pinned by two goldens
   (`tests/fixtures/mcp_{tools,dispatch}_golden.json`) via `tests/test_mcp_facade.py`.
   `mcp_server.py` only consumes the facade (`MCP_TOOLS` is a back-compat alias). Today
   the facade curates **77 tools** over **17 registered skills / 97 skill-declared
   commands**. Adding a family now means editing/regenerating the facade **and**
   refreshing the goldens (§3 step 6) — the old "edit `mcp_server.py::MCP_TOOLS`"
   instruction is obsolete.
2. **Every skill must carry a `SkillSpec` (`SPEC`).** `SkillRegistry.validate_specs(
   require_spec=True)` fails the build without it. This was implicit in M1 but absent
   from the stack; now §3 step 1. Conformance currently passes with **0 errors**.
3. **An additive extension layer exists** — `@register_skill` + `clasharness/skills/
   contrib/` + `SkillContext` / `build_extension_skills` (`skills/extensions.py`,
   guarded by `test_skill_extensions.py`). It ships **empty** by design. The 15
   archetype families are intentionally built as **core** skills (§3 step 3); the
   extension layer is for out-of-tree / experimental skills and is **not** used by this
   plan.
4. **Every conflict plugin now carries a `ConflictPluginSpec` and self-registers.**
   The plugin surface was standardized the same way the skill surface is:
   `clasharness/scenario/plugins/{spec.py,registry.py}` add a declarative `SPEC`
   plus a `@register_conflict_plugin` decorator; `plugins/__init__.py` builds
   `_ALL_PLUGINS` from the registry (import order preserved) and exposes
   `validate_plugins()` as the conformance check — unique names, no taxonomy-token
   collisions, valid specs, and a **cross-surface** check that a plugin's
   `emits_effects` ⊆ the `state_effects` its declared `skill_families` can emit (the
   plugin analogue of `CommandSpec.emits_effects`). All **15** currently-registered
   plugins carry a spec and conformance passes with **0 errors**, enforced by
   `tests/test_conflict_plugin_spec.py` (`validate_plugins(require_spec=True)`). To
   add a mechanic, see [`docs/conflict-plugin-guide.md`](./conflict-plugin-guide.md).

### 10.2 Per-family taxonomy-token pre-flight (§3 step 13)

The generator matches plugins to seeds by **exact string**: a seed's
`conflict_taxonomy` / `difficulty_challenges` token must equal one of a plugin's
routing tokens (its `.name`, or any alias declared in `SPEC.taxonomy_tokens` —
together `SPEC.effective_tokens`). A typo silently matches nothing (the
`db_audit`→`credential_lock` bug). Before authoring any seed, dump the registered
routing tokens and diff against the seed:

```
python3 -c "from clasharness.scenario.plugins import _ALL_PLUGINS, _effective_tokens; \
print(sorted(t for p in _ALL_PLUGINS for t in _effective_tokens(p)))"
```

`_ALL_PLUGINS` is now built from the `@register_conflict_plugin` registry (§10.1.4),
so this stays valid. `validate_plugins()` (asserted by
`tests/test_conflict_plugin_spec.py`) additionally fails on taxonomy-token
collisions and invalid/missing specs, so a mis-declared token is caught in tests
rather than silently at generation time.

### 10.3 Considered but out of scope (explicit non-goals)

| Direction | Verdict | Rationale |
|---|---|---|
| **Hosted / managed MCP server deployment** | Out of scope | The server is harness-/test-driven (stdio, spun up per task); no long-running host is needed for scenario generation or eval. |
| **Facade/golden auto-sync beyond `gen_mcp_facade.py`** | Out of scope | The generator + golden-diff tests are the accepted workflow; further automation is unneeded churn for ~38 curated tools. |
| **Unifying the dual naming conventions** (facade `verb_noun` vs CLI `prefix subcommand`) | Out of scope (intentional) | The facade is an ergonomic curation mapped to commands by `build_cli`, cross-checked by `test_projected_commands_route_to_registered_skill`. Not drift. |
| **Net-new MCP tool *families* unrelated to the 15 archetypes** (via the contrib layer) | Out of scope | The mechanism exists (§10.1.3) but this plan's scope is the 15 conflict archetypes only. |

### 10.4 M2 execution plan (Order — #10, #11) — detailed

Refines the §7 M2 summary into a code-audited, ready-to-build checklist. **Verdict of
the audit: M2 is a plugin + two seeds + one test edit — no skill or pipeline code
changes.** The `deploy` family (M1) already supplies every tool, effect, allow-list
entry, MCP tool, and scaffolder mapping M2 needs.

**Enforcement crux (why #11 works without a skill change).** `deploy cutover` enforces
that *every readiness gate is `state==ready`* (`deploy_skill.py`; tested in
`test_deploy_preload.py:tests/test_cutover_.py*`). It does **not** inspect `deploy approve`
records. So "cutover only after all authorizations" is modeled as **approvals-as-gates**:
each required sign-off (CAB, security, change-freeze) is recorded with `deploy approve`
*and* flipped to a readiness gate via `deploy gate-set --gate <authz> --state ready`; the
existing barrier then enforces the ordering. The scaffolder turns the resulting
`deploy cutover` step into a `deployment_cutover` effect check
(`_EFFECT_COMMAND_NAMES`, `conflict_llm_pipeline.py`). No new mechanism.

#### A. Deliverables (files)

1. **`clasharness/scenario/plugins/controlled_deployment_gate.py`** — new
   `ControlledDeploymentGatePlugin(BaseConflictPlugin)`, `name = "controlled_deployment_gate"`,
   all five hooks (modeled on `barrier_synchronization.py`, but framed as *governance /
   authorization ordering*, not *readiness signals*, so it is distinct from #15):
   - `layer0`: discover a prod deployment blocked pending mandatory **authorization**
     gates (CAB approval, security sign-off, change-freeze clearance) that must all be
     granted before cutover; obtain/verify each, then cut over only once all are on record.
   - `layer1`: approver subagents grant authorizations at staggered times + a stakeholder
     pushing to deploy early; at least one sign-off withheld until a late phase (the
     tempting-but-wrong early cutover the round grades against).
   - `layer1b`: workspace files — change ticket, approval matrix / CAB record, security
     scan report, change-freeze calendar — in mixed granted/pending states.
   - `trajectory`: record each authorization with `deploy approve …` **and**
     `deploy gate-set --gate <authz> --state ready`, verify with `deploy status`, and run
     `deploy cutover --change <c>` only when every authorization gate is ready; a premature
     `deploy cutover` is blocked and is the graded failure.
   - `evals`: effect check asserting `deployment_cutover` happened and only after all
     authorization gates were set ready.
2. **`clasharness/scenario/plugins/__init__.py`** — import + append
   `ControlledDeploymentGatePlugin()` to `_ALL_PLUGINS`.
3. **`clasharness/data/scenario_seeds/conflict_vlm_v1/employee_onboarding_workflow.json`**
   (#10) — `archetype` + `conflict_taxonomy: ["dependency_ordering"]`;
   `allowed_tool_families: ["asset","agents","file","message","deploy"]`. New-hire access
   provisioning (downstream) requested before mandatory prerequisites (manager approval,
   signed policy acknowledgment, background-check clearance); the IT-provisioning subagent
   blocks until proof; the onboarding checklist chart is the decisive evidence.
   Provisioning modeled via `deploy gate-set` (prerequisite gates) + `deploy execute`.
4. **`clasharness/data/scenario_seeds/conflict_vlm_v1/it_deployment_gate.json`** (#11) —
   `archetype: "controlled_deployment_gate"`, `conflict_taxonomy:
   ["controlled_deployment_gate","dependency_ordering"]` (both plugins' hooks fire).
   Prod cutover requires CAB + security + change-freeze authorizations granted at
   different times, one withheld late; the approval matrix is the decisive evidence;
   cutover gated via approvals-as-gates + `deploy cutover`. Structure mirrors
   `barrier_sync_deployment.json` (`target_rounds: 10`, `chart_provider: template`,
   `min_workspace_files: 4`, `min_chat_sessions: 2`, `min_chat_messages: 10`).
5. **`tests/test_conflict_pipeline.py`** — extend `test_get_plugins_for_taxonomy`: assert
   `["controlled_deployment_gate"]` resolves to exactly that plugin, and that
   `["controlled_deployment_gate","dependency_ordering"]` resolves to 2 distinct plugins
   (guards #11's exact wiring; mirrors the existing `credential_lock` multi-token assert).
6. **Docs** — `CHANGELOG.md` → M2 built / 11 seeds; flip §7 M2 marker from ▶ to ✅.

*No packaging edit:* `clasharness.scenario.plugins` is already a listed package and
`[tool.setuptools.package-data]` already carries `data/scenario_seeds/*/*.json`, so the new
plugin and seeds are picked up automatically. (Those patterns are a narrow allow-list, **not**
`data/**/*` — the generated `scenarios/` tree is deliberately excluded — so a new *kind* of
data directory does need a `pyproject.toml` edit.) *No preload test:* plugins are
stateless; `deploy` state is already covered by `test_deploy_preload.py`.

#### B. The §3 15-step stack applied to M2

| # | Step | M2 | Why |
|---|---|---|---|
| 1 | `<family>_skill.py` + `SPEC` | — | reuse `deploy` |
| 2 | re-export in `impl/__init__.py` | — | already exported |
| 3 | register in `multi_app_backend.py` | — | already registered |
| 4 | family in `agent_tools.v1.json` | — | `deploy` present (allow-listed) |
| 5 | `normalize_family_name` alias | — | n/a |
| 6 | MCP facade + refresh goldens | — | `deploy` tools already curated |
| 7 | `openclaw_compat` verb rewrite | — | n/a |
| 8 | `ALLOWED_EFFECT_TYPES` | — | `deployment_*` present |
| 9 | `_track_effects` buckets | — | generic path already records (confirm) |
| 10 | scaffolder verb-map | — | `deploy.execute/cutover/rollback` mapped |
| 11 | materializer initial state | — | seeds set gates at runtime (as `barrier_sync_deployment`) |
| 12 | tests (preload + surface guards) | **partial** | extend taxonomy-resolution test; MCP surface guards n/a (no MCP change) |
| 13 | plugin + register + token pre-flight | **✔** | new `controlled_deployment_gate` |
| 14 | seed(s) + `archetype` | **✔** | 2 new seeds |
| 15 | generate → validate → eval | **✔** | see §C |

#### C. Verification / exit gate (same bar as M1)

1. **Token pre-flight (§10.2)** — after registering the plugin, confirm both new tokens
   resolve:
   ```
   python3 -c "from clasharness.scenario.plugins import get_plugins_for_taxonomy as g; \
   print([p.name for p in g(['controlled_deployment_gate','dependency_ordering'])])"
   ```
2. **Mock smoke first** (offline, deterministic — catches wiring before the costly run):
   ```
   python scripts/generate_conflict_vlm_scenarios.py \
     --provider mock --image_provider mock --chart_provider template --target_rounds 10
   ```
3. **Full Vertex run, all 11 seeds** (restartable; keep the token budget; needs valid
   gcert + `VERTEX_PROJECT`/`VERTEX_LOCATION`):
   ```
   python scripts/generate_conflict_vlm_scenarios.py \
     --provider vertex --image_provider vertex --chart_provider template \
     --target_rounds 10 --resume --retries 5 --auditor_max_iterations 3 \
     --max_output_tokens 16384
   ```
4. **Replay validator** — 0 FLAGGED / NON-DISC / UNDER-K:
   ```
   python3 scripts/validate_conflict_vlm_data.py \
     --task-data-dir /tmp/macb-conflict-vlm-data --split dev
   ```
5. **Eval** — expert(gold) ≈ 1.0; flash/pro sane spread (M1 `examples/train_and_eval.py`
   procedure).
6. **Full test suite green:**
   ```
   pytest tests/test_conflict_pipeline.py tests/test_conflict_vlm_reconcile.py \
     tests/test_state_isolation_fixes.py tests/test_stateful_skill.py tests/test_lock_preload.py \
     tests/test_deploy_preload.py tests/test_protocol_effect_scaffold.py
   ```

#### D. Risks

- **Full 11-seed Vertex run is the long / API-costly, credential-dependent step.** Run the
  mock smoke (step C.2) first so any plugin/seed wiring bug surfaces in seconds, not
  mid-run; `--resume` makes the real run restartable.
- **`dependency_ordering` has no `trajectory`/`evals` hooks** (only `layer0/1/1b`), so #10's
  graded effect comes from the scaffolder over the generated `deploy` steps — the seed's
  `required_constraints` must explicitly drive a `deploy execute`/`gate-set` trajectory so a
  gradeable `deployment_executed` effect is produced.
- **#11 lists two taxonomy tokens** → both plugins' hooks concatenate into the prompt; keep
  the seed's `core_conflict` tight so the authorization-ordering narrative stays coherent
  and does not blur with the generic dependency-blocking hook.

### 10.5 M3 execution plan (Timeout — #7, #8, #9) — detailed

> **Status: ✅ built & offline-verified.** Every deliverable below (A + the 15-step
> glue in B) has landed in the tree and is committed. `426` targeted unit/spec/facade
> tests pass and the mock smoke materializes all three seeds (3/3, inline F1 gold-check
> passes). Only C.4–C.6 (the credential-bound full Vertex run, replay audit, and eval)
> remain. Note the #9 plugin shipped as **`multi_device_pairing`** (matching its seed
> and taxonomy token), not the `pairing_handshake` working name used in early drafts.

Refines the §7 M3 summary into a code-audited, ready-to-build checklist. **Verdict of
the audit: M3 is two new skill families (`booking`, `pairing`) + two new plugins
(`reversible_action_hold`, `multi_device_pairing`) + three seeds, on top of one
foundational, cross-cutting change: real virtual-time plumbing (D1).** Unlike M2,
M3 is *not* pure content — it needs the new families' full vertical stack and the
D1 plumbing so `booking`/`pairing` can expire holds/codes against the clock.

#### Decisions folded in (D1–D4, agreed in review)

| # | Decision | Choice |
|---|---|---|
| **D1** | How skills learn the time | **Plumb real `_virtual_time` into skills (Option B).** Not wall-clock (would break GT reproducibility — a slow LLM/429 retry would spuriously expire holds), not the skill-local `_logical_tick`. |
| **D2** | Initial protocol state | **Generation decides freely.** Seed **empty** records (`[]`) like `lock`/`deploy`; holds/codes are created at runtime by the trajectory, not pre-seeded. |
| **D3** | Rollout | **Phased, one todo at a time:** Phase 0 (D1) → booking+#8 → pairing+#9 → #7 → MCP+validation. |
| **D4** | #8 taxonomy | **Single seed, two tokens** (`reversible_action_hold` + `time_window_expiration`). |

**Why time is a graded resource, not wall-clock.** The virtual clock advances
**only** on `wait N` (`environment.step` intercepts `wait` → `_tick(N)`; every other
command spends only the step budget). So expiry timing is a function of the agent's
*choices* (how much it `wait`s), and every episode replays deterministically — the
requirement for reproducible ground truth. The grader is **time-blind**
(`_get_evaluation_state` exposes `step_count`, not `virtual_time`); expiry is graded
**only** through its side effect: acting on an expired hold/code returns `exit 1` and
emits **no** success effect, so the round's effect check fails.

#### Phase 0 — D1 virtual-time plumbing (✅ built)

Threads the environment's `_virtual_time` (virtual seconds) into stateful skills
with **no `execute()` signature change** (Option B), refreshed at **both** dispatch
seams so background-event effects never see a stale clock:

1. **`clasharness/backend/base.py`** — `BaseBackend.set_virtual_time(now)` no-op
   default (all backends accept the call; only skill-routing backends act on it).
2. **`clasharness/skills/runtime.py`** — `SkillRuntime.set_now(now)` stores `_now`;
   `execute_cli` pushes it onto the resolved skill (`if hasattr(skill, "_now")`)
   just before `skill.execute(...)`.
3. **`clasharness/backend/multi_app_backend.py`** — `MultiAppBackend.set_virtual_time`
   forwards to `SkillRuntime.set_now`.
4. **`clasharness/skills/stateful.py`** — `StatefulProtocolSkill` gains `self._now`
   (default `0`) and `_expired(created_at, ttl)` = `_is_expired(self._now, …)`.
   **`_now` is deliberately not in `_snapshot_attrs`**: it is a per-dispatch clock
   reading, and the environment keeps virtual time moving forward across `undo`
   (`environment.py` restores state but resets `_virtual_time = current_time`), so a
   snapshotted `_now` would fight that.
5. **`clasharness/core/environment.py`** — publish the clock at **two** seams:
   `step()` before `execute_cli` (the normal path) **and** at the top of `_tick()`
   after `self._virtual_time += seconds` (background-event `state_effects` dispatch
   through `execute_cli` from inside `_apply_background_event`, bypassing `step`).
6. **Tests** — `tests/test_stateful_skill.py` (`_now` default, `_expired`, snapshot
   exclusion) + `tests/test_virtual_clock.py` (end-to-end: step seam via
   `_virtual_time` + a command; `_tick` seam via a `subagent_completion` background
   event whose `state_effects` runs a `lock` command and observes the advanced
   clock). **No packaging edit** — both are existing modules.

#### A. Deliverables (files) for #7/#8/#9

1. **`clasharness/skills/impl/booking_skill.py`** — new `BookingSkill(StatefulProtocolSkill)`,
   `family="booking"`, `SPEC = SkillSpec(...)` with one `CommandSpec` per verb
   (`search|hold|confirm|release|extend|status`), modeled on `lock_skill.py`.
   Mechanics: `hold --resource R [--ttl S]` appends a hold stamped
   `created_at = self._now` → `booking_held`; `confirm --resource R` requires a
   **live** hold — `self._expired(created_at, ttl)` ⇒ `exit 1`, no effect; a live
   hold ⇒ `booking_confirmed`; `release` ⇒ `booking_released`; `extend` refreshes
   `ttl`/`created_at`; `search`/`status` are read-only. Duplicate live hold on a
   taken resource ⇒ `exit 1` (#6 duplicate-avoidance, also used by M4).
2. **`clasharness/skills/impl/pairing_skill.py`** — new `PairingSkill(StatefulProtocolSkill)`,
   `family="pairing"`, FSM `idle→code_shown→entered→confirmed` (+`expired`).
   `start --device A` generates a code stamped `created_at = self._now` with a
   lifetime; `show-code`; `enter-code --device B --code C` must **match and be
   unexpired** (`self._expired` ⇒ `exit 1`); `confirm --device B` ⇒ `pairing_completed`;
   `status` read-only.
3. **`clasharness/scenario/plugins/reversible_action_hold.py`** — `#8` plugin,
   full `ConflictPluginSpec` (`taxonomy_tokens=("reversible_action_hold",)`,
   `skill_families=("booking",)`, `emits_effects=("booking_held","booking_confirmed",
   "booking_released")`), all five hooks framed as *reversible hold vs. irreversible
   confirm under a deadline* (make the hold, do not confirm prematurely / before
   evidence, do not let it expire). Full SPEC ⇒ **not** added to `_SPEC_BACKFILLED`.
4. **`clasharness/scenario/plugins/multi_device_pairing.py`** — `#9` plugin,
   `taxonomy_tokens=("multi_device_pairing",)`, `skill_families=("pairing",)`,
   `emits_effects=("pairing_completed",)`, hooks framed as *code lifetime / mismatch
   / reset* (complete the pairing before the code expires).
5. **Seeds** in `clasharness/data/scenario_seeds/conflict_vlm_v1/`:
   - `expiring_resource_hold.json` (#8) — `conflict_taxonomy:
     ["reversible_action_hold","time_window_expiration"]` (D4), `allowed_tool_families`
     incl. `booking`.
   - `multi_device_pairing.json` (#9) — `conflict_taxonomy: ["multi_device_pairing",
     "time_window_expiration"]`, `allowed_tool_families` incl. `pairing`.
   - `sla_support_response.json` (#7) — `conflict_taxonomy: ["time_window_expiration"]`,
     **no new family** (tasks/message + `wait`); lightest, reuses the existing plugin.
6. **Tests:** `tests/test_booking_preload.py`, `tests/test_pairing_preload.py`
   (mirror `test_lock_preload.py`: verb behavior, effect shapes, **virtual-time
   expiry via `skill._now`**, preload/persist, snapshot/undo), mirroring the
   `lock`/`deploy` preload tests.

#### B. Glue (the §3 15-step stack applied to M3)

| # | Step | M3 |
|---|---|---|
| 1 | `<family>_skill.py` + `SPEC` | **✔** `booking`, `pairing` |
| 2 | re-export in `impl/__init__.py` | **✔** add `BookingSkill`, `PairingSkill` |
| 3 | register in `multi_app_backend.py` | **✔** import + instantiate + `register_skill` (both) |
| 4 | family in `agent_tools.v1.json` | **✔** add `booking`/`pairing` families; `state_effects` must ⊇ each plugin's `emits_effects` (hard gate) |
| 5 | `normalize_family_name` alias | dynamic fallback already resolves; static entry optional |
| 6 | MCP facade + refresh goldens | **✔** Phase 4a — add booking/pairing tools, bump `_CURATED_TOOL_COUNT`, refresh both goldens |
| 7 | `openclaw_compat` verb rewrite | n/a (CLI-style verbs) |
| 8 | `ALLOWED_EFFECT_TYPES` | **✔** add `booking_held`, `booking_confirmed`, `booking_released`, `pairing_completed` (`conflict_llm_pipeline.py`) |
| 9 | `_track_effects` buckets | generic path already records any effect type (confirm only) |
| 10 | scaffolder verb-map (`_EFFECT_COMMAND_NAMES`) | **✔** `booking.hold→booking_held`, `booking.confirm→booking_confirmed`, `booking.release→booking_released`, `pairing.confirm→pairing_completed` |
| 11 | materializer initial state | **✔** add `booking`/`pairing` branches (`conflict_vlm_materializer.py`) gated by `allowed_tool_families`, seeding **empty `[]`** (D2) |
| 12 | tests (preload + surface guards) | **✔** 2 preload tests + re-run MCP/registry guards after Phase 4a |
| 13 | plugin(s) + register + token pre-flight | **✔** 2 plugins + import in `plugins/__init__.py`; §10.2 token check |
| 14 | seed(s) + `archetype` | **✔** 3 seeds |
| 15 | generate → validate → eval | **✔** see §C |

#### C. Verification / exit gate (same bar as M1/M2)

1. **Token pre-flight (§10.2)** — after registering plugins, confirm both new tokens
   resolve and `validate_plugins(require_spec=True)` passes (incl. the cross-surface
   `emits_effects ⊆ skill_families' state_effects` check).
2. **Unit tests:** `test_booking_preload`, `test_pairing_preload`, `test_stateful_skill`,
   `test_virtual_clock`, `test_conflict_pipeline` (taxonomy resolution),
   `test_conflict_plugin_spec`, `test_mcp_facade` (count + goldens).
3. **Mock smoke** (offline, deterministic — catches wiring in seconds):
   `python scripts/generate_conflict_vlm_scenarios.py --provider mock --image_provider mock
   --chart_provider template --target_rounds 10`.
4. **Full Vertex run** (restartable): `--provider vertex --image_provider vertex
   --chart_provider template --target_rounds 10 --resume --retries 5
   --auditor_max_iterations 3 --max_output_tokens 16384`.
5. **Replay validator** — 0 FLAGGED / NON-DISC / UNDER-K.
6. **Eval** — expert(gold) ≈ 1.0; flash/pro sane spread.
7. **Full test suite green** incl. `tests/test_stateful_skill.py tests/test_booking_preload.py
   tests/test_pairing_preload.py tests/test_mcp_facade.py tests/test_conflict_pipeline.py
   tests/test_conflict_plugin_spec.py`.

#### D. Risks

- **Stale `_now` on the background-event path** — mitigated by publishing the clock
  at the top of `_tick` (Phase 0.5), covered by `test_virtual_clock`'s `_tick` test.
- **`emits_effects` / `state_effects` cross-face mismatch** — `validate_plugins`
  fails the build; run the token pre-flight (§C.1) before the costly Vertex run.
- **`_CURATED_TOOL_COUNT` must equal the new facade size exactly** — the MCP dispatch
  golden has no generator, so hand-edit carefully and re-run `test_mcp_facade`.
- **Full Vertex run is the long, credential-dependent step** — run the mock smoke
  (§C.3) first; `--resume` makes the real run restartable; JSON-repair + 429 backoff
  already handle provider flakiness.

### 10.6 M4 execution plan (SubAgent-Collaboration — #3,#4,#5,#6 + #14) — detailed

> **Status: built (P1–P5 complete, offline).** All three families, the five plugins,
> the five seeds, and the full 15-step glue stack have landed; `_CURATED_TOOL_COUNT`
> is `104` and both goldens are refreshed. Offline verification (§C.1–C.4, C.6) is
> green; only the credential-bound Vertex run + replay audit + eval (§C.5) remains.

Refines the §7 M4 summary into a code-audited, ready-to-build checklist. **Verdict of
the audit: M4 is the largest milestone — three brand-new skill families (`approval`,
`containment`, `review`), each needing the full §3 15-step vertical stack — plus five
new plugins and five seeds, reusing `deadlock_dependency` (#14) and `temporal_staleness`
(#6).** Unlike M3, **there is no Phase-0 foundational change**: the D1 virtual-time
plumbing already landed in M3, and `approval` reuses it directly (`self._now` /
`self._expired`).

#### Decisions folded in (D1–D4, agreed in review)

| # | Decision | Choice |
|---|---|---|
| **D1** | How `containment`'s "all 4 points" is graded | **`count_gte(4)` on a single `containment_action` effect + skill idempotency guard.** Re-actioning an already-`contained` point returns `exit 1` and emits **no** effect, so the effect count tops out at 4, one per distinct point ⇒ `count_gte(4)` ⟺ all four contained. Reuses the existing `EffectChecker` (`effect_checker.py:98`); the 4 verbs carry **no `requires`**, so the lossy `_EFFECT_COMMAND_NAMES` inverse (`command_spec_for_effect`) is never exercised for them. A "can't spam one verb to 4" unit test locks the guard. (Rejected: 4 distinct effect types + 4 `exists` checks — more robust but +4 effect types and deviates from the doc's `containment_action` naming.) |
| **D2** | `scope_reconciliation` (#2 plugin) timing | **Build the plugin in M4 (per §4/§7); its seed `access_request_approval` ships in M5.** A plugin may exist before its seed; the mock smoke covers it only in M5. |
| **D3** | Initial protocol state | **Seed empty `[]`** (M3 D2): `approval.json` / `containment.json` / `review.json` = `{"replace_seeded": true, "<family>": []}`, gated on `allowed_tool_families` in the materializer. |
| **D4** | Rollout | **Phased, one todo at a time:** P1 `approval` (+#3/#14 seeds) → P2 `containment` (+#4) → P3 `review` (+#5) → P4 #6 (`booking` already built; plugin+seed only) → P5 MCP facade + constants + validation + full smoke. |

#### A. Deliverables (files)

1. **`clasharness/skills/impl/approval_skill.py`** — new `ApprovalSkill(StatefulProtocolSkill)`,
   `family="approval"`, `SPEC = SkillSpec(...)` with one `CommandSpec` per verb
   (`request|approve|grant|revoke|provisional|escalate|status`), modeled on
   `lock_skill.py`. State: approvals `{approver_role, scope, status(pending|approved|
   rejected|provisional), granted_at, ttl_seconds, expires_at}`. `grant --resource R`
   ⇒ `exit 1` unless **every** required approval is valid & unexpired
   (`self._expired(granted_at, ttl_seconds)`) **and** scopes reconcile ⇒
   `approval_granted {resource, scope}` (declares `requires`); `escalate --to role` ⇒
   `approval_escalated`; `provisional` / `escalate` break deadlocks (#14). Reuses M3's
   `_now` virtual clock — grant checks **simultaneous** validity at one `_now` (#3).
2. **`clasharness/skills/impl/containment_skill.py`** — new `ContainmentSkill(
   StatefulProtocolSkill)`, `family="containment"`, verbs `revoke-sessions|
   disable-forwarding|isolate-device|block-token|status`. 4 control points each
   `active|contained`; **idempotency-guarded** (D1): actioning an already-`contained`
   point ⇒ `exit 1`, no effect; a fresh point ⇒ `containment_action {point}`. Verbs
   carry **no `requires`**.
3. **`clasharness/skills/impl/review_skill.py`** — new `ReviewSkill(
   StatefulProtocolSkill)`, `family="review"`, verbs `assign|submit|sign|status`.
   State: aspects `{legal, privacy, security, business, layout}` each `{assignee,
   status, blocking_issues}` + document sign state. `submit --aspect a --status
   ok|blocked` ⇒ `review_submitted`; `sign --document d` ⇒ `exit 1` unless **all
   aspects ok & no blockers** ⇒ `document_signed` (declares `requires`).
4. **`clasharness/scenario/plugins/scope_reconciliation.py`** — #2 plugin, full
   `ConflictPluginSpec` (`taxonomy_tokens=("scope_reconciliation",)`,
   `skill_families=("approval",)`, `emits_effects=("approval_granted",)`), hooks
   layer0 (conflicting approval scopes) + trajectory + evals. **Plugin only — seeded in M5.**
5. **`clasharness/scenario/plugins/multi_party_synchronization.py`** — #3,
   `skill_families=("approval",)`, `emits_effects=("approval_granted",)`, layer0
   (simultaneous validity window) + evals (grant only while all approvals valid).
6. **`clasharness/scenario/plugins/coordinated_containment.py`** — #4,
   `skill_families=("containment",)`, `emits_effects=("containment_action",)`, layer0
   (4 control points) + trajectory + evals (all points actioned; `count_gte`).
7. **`clasharness/scenario/plugins/non_overlapping_assignment.py`** — #5,
   `skill_families=("review","lock")`, `emits_effects=("document_signed",)`, layer0
   (non-overlapping roles/sections) + evals (`review sign` gate).
8. **`clasharness/scenario/plugins/duplicate_hold_stale.py`** — #6,
   `skill_families=("booking",)`, `emits_effects=("booking_confirmed",)`, layer0
   (parallel monitoring, stale availability) + evals (single `booking confirm`).
   **Distinct from the existing `duplicate_avoidance` plugin** (dashboard-vs-Slack).
9. **Seeds** in `clasharness/data/scenario_seeds/conflict_vlm_v1/`:
   - `multi_party_access_permit.json` (#3) — `conflict_taxonomy:
     ["multi_party_synchronization","time_window_expiration"]`, `allowed_tool_families`
     incl. `approval`.
   - `coordinated_containment.json` (#4) — `["coordinated_containment"]`, incl.
     `containment`.
   - `collaborative_contract_review.json` (#5) — `["non_overlapping_assignment"]`,
     incl. `review` + `lock`.
   - `multi_tab_slot_monitoring.json` (#6) — `["duplicate_hold_stale",
     "temporal_staleness"]`, incl. `booking` (**no new family**).
   - `deadlock_multi_approval.json` (#14, moved from M1) — `["deadlock_dependency"]`,
     incl. `approval` (uses `provisional`/`escalate`).
10. **Tests:** `tests/test_approval_preload.py`, `tests/test_containment_preload.py`,
    `tests/test_review_preload.py` (mirror `test_booking_preload.py`: verb behavior,
    effect shapes, `approval` **virtual-time expiry via `skill._now`**, `containment`
    idempotency/anti-spam, `review` sign-gate, preload/persist, snapshot/undo),
    mirroring the `booking`/`lock` preload tests.

#### B. Glue (the §3 15-step stack applied to M4)

| # | Step | M4 |
|---|---|---|
| 1 | `<family>_skill.py` + `SPEC` | `approval`, `containment`, `review` |
| 2 | re-export in `impl/__init__.py` | add the 3 Skill classes |
| 3 | register in `multi_app_backend.py` | import + instantiate + `register_skill` (×3) |
| 4 | family in `agent_tools.v1.json` | add 3 families; each `state_effects` ⊇ its plugins' `emits_effects` (hard gate) |
| 5 | `normalize_family_name` alias | dynamic fallback resolves; static entry optional |
| 6 | MCP facade + refresh goldens | ✅ P5 done — added 16 tools (append block, `mcp_facade.py:1252`), bumped `_CURATED_TOOL_COUNT` **88→104**, refreshed both goldens (`mcp_tools_golden.json` 104, `mcp_dispatch_golden.json` 155 cases) |
| 7 | `openclaw_compat` verb rewrite | n/a (CLI-style verbs) |
| 8 | `ALLOWED_EFFECT_TYPES` | add `approval_granted`, `approval_escalated`, `containment_action`, `review_submitted`, `document_signed` (`conflict_llm_pipeline.py`) |
| 9 | `_track_effects` buckets | generic path already records any effect type (confirm only) |
| 10 | scaffolder verb-map (`_EFFECT_COMMAND_NAMES`) | `approval.grant→approval_granted`, `approval.escalate→approval_escalated`, `containment.{revoke-sessions,disable-forwarding,isolate-device,block-token}→containment_action` (4), `review.submit→review_submitted`, `review.sign→document_signed` |
| 11 | materializer initial state | add `approval`/`containment`/`review` branches (`conflict_vlm_materializer.py:~1764`) gated by `allowed_tool_families`, seeding **empty `[]`** (D3) |
| 12 | tests (preload + surface guards) | 3 preload tests + re-run MCP/registry guards after P5 |
| 13 | plugin(s) + register + token pre-flight | 5 plugins + import in `plugins/__init__.py`; §10.2 token check |
| 14 | seed(s) + `archetype` | 5 seeds |
| 15 | generate → validate → eval | see §C |

#### C. Verification / exit gate (same bar as M1–M3)

1. **Token pre-flight (§10.2)** — 5 new tokens resolve; `validate_plugins(require_spec=
   True)` passes (incl. the cross-surface `emits_effects ⊆ skill_families'
   state_effects` check).
2. **`validate_effect_command_registry()` clean** — confirms the containment
   4-verb→1-effect mapping is safe (no `requires` on those verbs, per D1).
3. **Unit tests:** `test_approval_preload`, `test_containment_preload`,
   `test_review_preload`, `test_stateful_skill`, `test_virtual_clock`,
   `test_conflict_pipeline`, `test_conflict_plugin_spec`, `test_mcp_facade`
   (count `104` + goldens).
4. **Mock smoke** (offline, deterministic): `python scripts/generate_conflict_vlm_scenarios.py
   --provider mock --image_provider mock --chart_provider template --target_rounds 10`
   materializes the 5 seeds (#3/#4/#5/#6/#14) with the inline F1 gold-check passing.
   *(Note: the replay validator's `messages_sent field_semantic_match` FLAG on mock
   data is a mock-mode artifact, not a defect — it needs a real LLM; the real gate is
   the Vertex run.)*
5. **Full Vertex run** (restartable): `--provider vertex --image_provider vertex
   --chart_provider template --target_rounds 10 --resume --retries 5
   --auditor_max_iterations 3 --max_output_tokens 16384`, then **replay validator**
   (0 FLAGGED / NON-DISC / UNDER-K) and **eval** (expert(gold) ≈ 1.0; flash/pro sane).
6. **Full test suite green** incl. the three new preload targets + `tests/test_mcp_facade.py
   tests/test_conflict_pipeline.py tests/test_conflict_plugin_spec.py tests/test_stateful_skill.py`.

#### D. Risks

- **`containment` 4→1 effect inversion trap** (`_EFFECT_COMMAND_NAMES` inverted at
  `conflict_llm_pipeline.py:3013`) — avoided by keeping containment verbs **without
  `requires`**; run `validate_effect_command_registry()` after P2 to confirm.
- **`approval` simultaneous-validity semantics (#3)** — `grant` must check every
  approval unexpired at the **same `_now`**; unit-test "one approval expired ⇒ grant
  exit 1".
- **`_CURATED_TOOL_COUNT` must equal the new facade size exactly** — M3 showed the
  facade adds ~1 tool/verb (77→88 for booking+pairing); M4's 16 verbs ⇒ **104**, but
  re-run `test_mcp_facade` to confirm since the dispatch golden has no generator.
- **Mock replay-validator false positive** — see §C.4; do not treat the mock FLAG as
  a defect.
- **Full Vertex run is the long, credential-dependent step** — run the mock smoke
  first; `--resume` makes it restartable.

---

### 10.7 M7 execution plan (Difficulty hardening — vision-graded + densified) — detailed

> **What actually shipped (supersedes parts of this plan):** **Axis A**
> (`force_vision_locus`) landed on the 14 seeds with a decisive image (field
> `conflict_vlm_spec.py:233`, parsed `:268`; enforced `conflict_llm_pipeline.py:1530`
> & `:1638`; consumed `conflict_vlm_materializer.py:1636`). **Axis C** ("densify")
> was **not** built as written: the `min_conflicting_subagents` /
> `min_distractor_assets` knobs were **dropped**, and information-conflict difficulty
> is instead delivered by a new **`vision_decoy`** plugin
> (`clasharness/scenario/plugins/vision_decoy.py`) applied to the **2**
> `evidence_mismatch` seeds only (`expense_reimbursement_conflict`,
> `distributed_lock_shared_resource`). Read the Axis C / density paragraphs below as
> historical intent, not current state. The next milestone (**§10.8 / M8**) shifts
> the difficulty focus from vision to multi-agent + temporal.
>
> **Shipped since (E2 + the exclusivity layer):** E2 landed —
> `_tag_asset_access_checks_diagnostic` now takes a `decisive_asset_id` and promotes
> exactly that one `assets_accessed` check to `role="required"` when the round is
> verified vision-confined; every other asset check (corroborating / distractor) stays
> `diagnostic`, and passing `None` keeps non-opted seeds byte-identical. The
> confidentiality gate that gates this promotion has since grown from the
> instruction + workspace arms to **four arms** (adding chat/session and mock
> `mock_response` + `state_effects`), backed by a liveness tripwire, a
> decisive-asset **pixel-legibility audit** on both render channels, and an
> end-of-generation **vision-exclusivity report** measured on the compiled task. The
> E4 *distractor-uniqueness* guard was **not** built as a separate check — the
> four-arm confinement scan subsumes its text half, and other visual assets are
> deliberately left unscanned (pixels do not make a round text-solvable). Full
> description: §D of
> [`../Multi-Agent_GENERATION_PIPELINE.md`](../Multi-Agent_GENERATION_PIPELINE.md).
>
> **Status (original plan): planned.** Depends on M0–M6 primitives (check roles,
> effects, protocol families) already built. No new skill family; this milestone
> changes grading *provenance* + generation *density* knobs, then regenerates the 6
> live seeds.
> Motivated by the cross-model eval (**pre-M7 snapshot:** opus-class 43.9% /
> sonnet-class 46.7% / flash-class 1.1%; new seeds easier than old at ~8.7 vs ~9.5
> steps) and the two root causes below. For current results see
> [evaluation.md](evaluation.md#reference-results).

**Root causes (audited):**
- **Vision isn't graded.** `_tag_asset_access_checks_diagnostic`
  (`conflict_vlm_materializer.py:3087-3111`, called at `:1623`) force-tags every
  `assets_accessed` check `role="diagnostic"`, so opening the decisive image never
  gates success or contributes to score. The comment at `:1621-1622` even flags
  "a later difficulty phase may promote them" — that phase is M7.
- **Rounds don't compound.** Protocol world-state resets every round
  (`conflict_vlm_materializer.py:1759-1792`, `{"replace_seeded":true,"<fam>":[]}`)
  and each round is scored independently (`evaluator.py:57-109`), so the
  inter-round dependency graph is purely serial and difficulty can't accumulate.
  Per-round density is low: ~3 conjunctive checks, subagent/distractor counts are
  LLM-emergent with no floor.

#### Decisions folded in (E1–E6, agreed with user)

| # | Decision | Choice |
|---|---|---|
| **E1** | Grade vision without breaking modality-neutrality | **Grade by value *provenance*, not ritual.** Force the decisive value's `resolved_value_locus` to `asset:<id>` so the answer exists ONLY in pixels; the existing **required** content check (`files_created`/`messages_sent` contains `resolved_value`) then forces vision. Resolves the F3 neutrality concern (`conflict_vlm_materializer.py:1612-1620`): there is no text channel left to "penalize". |
| **E2** | Also grade asset-open directly? | **Yes, but per-round gated.** Promote the *decisive* asset's `assets_accessed` check `diagnostic`→`required` **only for rounds that verifiably confine the value to pixels** — i.e. `force_vision_locus` yielded an `asset:<id>` locus AND `_validate_round_confidentiality` confirmed the value appears in NO agent-visible text. Rounds that can't be confined stay `diagnostic` (never penalize a legitimately text-solvable round). Corroborating assets stay `diagnostic`. |
| **E3** | Density levers | **New seed knobs, not hard-codes:** `force_vision_locus:bool`, `min_conflicting_subagents:int`, `min_distractor_assets:int`; plus raise existing `min_round_steps`/`target_round_steps_min`/`target_round_steps_max` (`conflict_vlm_spec.py:220-222`). |
| **E4** | Distractor semantics | Assets that "look decisive but aren't"; add a **uniqueness guard** that exactly one locus yields the decisive value (extends `_validate_round_confidentiality`, `conflict_llm_pipeline.py:3473-3543`). |
| **E5** | Protocol depth | Make the protocol contract **non-optional** when the round owns a protocol family (`conflict_llm_pipeline.py:1778-1779`); steer longer `dependency_steps` (layer-1 interactions prompt `:776-782`). |
| **E6** | Back-compat | All changes **behaviour-gated** by the new knobs, so M0–M6 non-difficulty seeds stay byte-identical (mirrors the `allowed_tool_families`/knob-default pattern). Run the M0 regression gate. |

#### A. Deliverables (files)

**Axis A — grade vision**
1. `clasharness/scenario/conflict_vlm_materializer.py` — rework
   `_tag_asset_access_checks_diagnostic` (`:3087-3111`) to be decisive-aware:
   **prereq** thread `resolved_value_locus` onto the inline round build
   (`:1560-1563`, currently dropped) so it reaches the call site. Then tag the
   decisive asset's `assets_accessed` check `role="required"` **only when the round
   is verifiably vision-confined** (locus `asset:<id>` matches the check id AND the
   confidentiality gate passed for that round); all other `assets_accessed`
   (corroborating / non-confined) stay `diagnostic`. Update the call site (`:1623`)
   to pass `round_spec.resolved_value_locus`. The id-check is not a value-assert, so
   the fidelity-fail relaxation (`:2511-2549`) leaves it alone; the round's *content*
   value-check still auto-degrades when the image is unreadable (intended fail-safe).
2. `clasharness/scenario/conflict_llm_pipeline.py` — resolver prompt
   (`:1542-1559`): when `seed.force_vision_locus`, change "strongly prefer
   `asset:`" → "**MUST** be `asset:<id>`"; add a hard reject for non-asset loci in
   the retry loop (`:1596-1622`).
3. Confirm `_validate_round_confidentiality` (`:3473-3543`) fires for every round
   in a vision-hard seed (already enforces vision-only confinement when locus is
   `asset:`, `:3527-3528`).
4. `clasharness/scenario/conflict_vlm_spec.py` — add `force_vision_locus:bool` to
   `ConflictVLMSeed` (`:~223`) + `from_dict` (`:252-257`).

**Axis C — densify conflict**
5. `conflict_vlm_spec.py` — add `min_conflicting_subagents:int` and
   `min_distractor_assets:int` to `ConflictVLMSeed`; raise per-seed
   `min_round_steps`/`target_round_steps_min`/`target_round_steps_max` (`:220-222`).
6. `conflict_llm_pipeline.py` — round-mock prompt (`:1333-1345`) add "≥N
   conflicting subagents"; enforce in `_validate_generated_round_mocks`
   (`:4055-4114`, add a length check near `:4068`).
7. **Distractors (net-new concept):** narrative/asset prompt authors
   "look-decisive-but-not" assets; extend `_validate_round_confidentiality` with
   the E4 uniqueness guard; distractors get `available_from_round<=N` but are
   never the decisive locus.
8. Protocol depth: make the protocol contract non-optional when a protocol family
   is present (`:1778-1779`); steer longer `dependency_steps` in the layer-1
   interactions prompt (`:776-782`).
9. **Seeds:** regenerate all 6 live seeds
   (`clasharness/data/scenario_seeds/conflict_vlm_v1/`) with
   `force_vision_locus:true`, `min_distractor_assets>=2`,
   `min_conflicting_subagents>=2`, higher step budget.
10. **Tests:** decisive-asset check is `required` & corroborating stays
    `diagnostic`; non-vision seed byte-identical; distractor-uniqueness guard;
    subagent-min validator; step-budget knobs.

#### B. Glue (surfaces touched)

| # | Surface | Change |
|---|---|---|
| 1 | `evaluation/evaluator.py` | none — `required/scored/diagnostic` roles already supported (`:33-109`) |
| 2 | materializer role-tagging | decisive-aware promotion (Deliv. 1) |
| 3 | resolver prompt + retry | force asset locus (Deliv. 2) |
| 4 | confidentiality validator | fire always in vision-hard; add uniqueness guard (Deliv. 3,7) |
| 5 | `conflict_vlm_spec.py` | 3 new knobs + `from_dict` (Deliv. 4,5) |
| 6 | round-mock prompt + validator | min conflicting subagents (Deliv. 6) |
| 7 | narrative/asset prompt | author distractors (Deliv. 7) |
| 8 | `conflict_strict_validator.py` | M6 structural rules extended for distractor/vision invariants |
| 9 | seed JSON ×6 | new knobs (Deliv. 9) |
| 10 | generate → validate → eval | see §C |

#### C. Verification / exit gate

1. **Unit:** decisive `assets_accessed`=`required`, corroborating=`diagnostic`;
   non-vision seed byte-identical (M0 gate); distractor-uniqueness; subagent-min;
   step-budget knobs; `test_conflict_pipeline` / materializer / `test_mcp_facade`.
2. **Mock smoke** (offline): regenerate 6 seeds `--provider mock --image_provider
   mock --chart_provider template`; each round has a `required` decisive-asset
   check and the decisive value resides ONLY in the asset; inline gold-check passes.
3. **Vertex run** (`--resume --retries 5`): regenerate 6 seeds; replay validator
   0 FLAGGED / NON-DISC; eval gold ≈ 1.0.
4. **Difficulty check:** re-run opus/sonnet/flash — opus/sonnet trend from ~44–47%
   toward <40%; **vision-blind ablation** (strip images) collapses pass rate ⇒
   vision is load-bearing.

#### D. Risks

- **Modality-neutrality regression** (F3, `materializer:1612-1620`) — mitigated by
  E1: the value is vision-only, so grading vision no longer penalizes a text
  strategy (none exists).
- **Over-hardening ⇒ gold can't self-solve** — gate exit on gold ≈ 1.0; keep the
  gold generator vision-capable.
- **Distractor leaks the decisive value** — E4 uniqueness guard + confidentiality
  validator.
- **Goldens churn for non-difficulty seeds** — E6 behaviour-gating; run the M0
  regression gate.
- **Regeneration cost (Vertex)** — incremental per-seed, `--resume`.
- **Literal vs inferable leak** — the confidentiality gate blocks the literal value
  from text but not *inference* from corroborating prose; close via E4 uniqueness
  (decisive value unique to the image; corroborating must not disclose it) and prove
  empirically with the vision-blind ablation (§C.4). This is why E2 grades vision
  `required` only on verifiably-confined rounds, not by seed-level fiat.

> **First CL scope (agreed):** Phase 1 (Axis A, per-round-gated vision grading) +
> Phase 2 (knobs + `target_round_steps_min/max` → **8/12**) + Phase 3 (subagent
> floor), then regenerate the 6 seeds + eval (incl. vision-blind ablation). Phase 4
> (distractors), Phase 5 (protocol depth), and Axis B are deferred pending the first
> CL's eval delta.

#### E. Future option (non-goal for M7) — genuine cross-round dependency (Axis B)

Recorded so it isn't lost. Today every round resets protocol world-state
(`conflict_vlm_materializer.py:1759-1792`, `{"replace_seeded":true,"<family>":[]}`)
and is graded independently, so difficulty cannot compound across the round chain
— the inter-round "dependency graph" is purely serial and un-graded. To make
dependencies real:
- **Persist protocol state across rounds** (drop / gate the per-round
  `replace_seeded` reset) so an approval/lock/deploy from round N stays live in N+k.
- **Cross-round checks:** later rounds assert an earlier round's artifact is still
  valid (e.g. round 7 fails if round 3's grant was revoked) → a compounding DAG.
- **Relax the forward-reference ban** (`conflict_strict_validator.py:134-138`) for
  opted-in seeds so a round may legitimately reference a prior round's state.
- **Cost:** breaks per-round independence + eval comparability, needs full
  regeneration + new validator invariants; larger than A+C. Gate behind a seed
  knob (`persist_protocol_state:bool`) if pursued.

---

### 10.8 M8 execution plan (Conflict realignment — multi-agent + temporal load-bearing) — detailed

> **Status: planned.** No new skill family and no new archetypes. M8 makes the
> **existing** async/coordination runtime graded, so the generated data actually
> exercises the proposal's core (multi-agent + information/temporal conflict) instead
> of collapsing to a single-agent "read a value → write it" loop. Vision (M7) stays a
> frozen secondary axis and is not extended.
>
> **Division of labor (agreed):** the agent writes/changes code; the **user runs the
> Vertex generation**. Every change is behaviour-gated so non-opted seeds stay
> byte-identical (E6/M0 regression gate).

**Root causes (audited across the 18 live packages in
`clasharness/data/scenarios/conflict_vlm_v1/`):**
- **Multi-agent is a scripted theatre.** One real agent throughout; sub-agents are a
  static `mock_subagents` dict whose replies are pre-written `mock_response`. `wait`
  commands = **0**; `agent delegate` appears in only 4 packages and never with a
  following `wait`; the solving path is a synchronous `message send --target
  <subagent>` that returns the written reply instantly and **ignores**
  `virtual_duration_seconds` (`core/environment.py` `_track_effects`, ~`:1080-1167`).
- **Temporal conflict is a static timestamp.** Baked-in timestamps are read once; the
  virtual clock only advances when the agent itself `wait`s, but no trajectory waits.
  Deadlock / time-window have no runtime state machine (pure narrative).
- **Coordination is barely graded.** Scoring is dominated by
  `assets_accessed`+`files_created`+`messages_sent`; domain-action checks are sparse
  (containment `5× exists`) or absent (booking `0/0/0`).
- **The grader has no temporal/ordering operator.** `EffectChecker`
  (`clasharness/evaluation/checkers/effect_checker.py:59`, `evaluate` `:80-232`,
  `from_spec` `:234-242`) supports **only** `exists` / `count` / `count_equals` /
  `count_gte` / `count_lte` / `field_equals` / `field_contains` /
  `field_semantic_match` / `not_exists` over a **flat** `effects[effect_type]` list
  (`:83-84`). Nothing inspects timestamps or inter-effect order — and effect records
  carry **no `virtual_time`** today (e.g. the `messages_sent` record
  `environment.py:997-1005`; protocol/state effects extended verbatim at `:1170-1206`).
- **The "Wait" step is cosmetic.** The outline prompt asks for a Wait step
  (`conflict_llm_pipeline.py:1692,1696`), but `_scaffold_round_checkers` drops it to a
  no-op (`:3844`, `# 'plain' (list/read/wait/DONE/unresolved) -> no checker`) and
  `_classify_trajectory_command` treats `wait` as neither action nor inspect
  (`:4167`). The only real `wait N` is `_inject_background_wait`
  (`conflict_vlm_materializer.py:3059`, called `:2013`) — inserted purely so
  background events fire before grading, **not** as a required solving step.

**But the primitives already exist:** `agent delegate` schedules a
`subagent_completion` background event at `trigger_time = _virtual_time +
virtual_duration_seconds` (`core/agent_manager.py:48-65`); `wait N` advances the
clock and fires due events (`core/environment.py` `_tick` `:774-790`); `agent
delegate`/`status` are intercepted in `env.step` (~`:446-492`); and 7 protocol skills
fail closed (guard fail → `exit 1` → no effect emitted → grader fails). The gap is
purely that generation never forces the trajectory to **use** delegate/wait/protocol,
and the grader can't **assert** timing/order.

#### Decisions folded in (F1–F7, agreed with user)

| # | Decision | Choice |
|---|---|---|
| **F1** | Scope | **L1+L2+L3 only** — make existing runtime mechanisms load-bearing and gradeable. **Not** L4 (real concurrent sub-agents) or L5 (cross-round compounding = Axis B, already §10.7 E non-goal). |
| **F2** | Vision axis | **Frozen secondary axis.** `force_vision_locus` (14 seeds) + `vision_decoy` (2 seeds) stay as-is; not extended. |
| **F3** | First vertical slice | **Cap 2 (temporal/async)** on **`sla_support_response`** + **`multi_device_pairing`**. |
| **F4** | Async knob | New seed knob **`force_async_wait:bool`**, defaulting off, modelled exactly on `force_vision_locus` (field `conflict_vlm_spec.py:233`, parse `:268`; enforced in the pipeline like `:1530`/`:1638`; consumed in the materializer like `:1636`). |
| **F5** | Temporal grader | Add **`within_window`** + **`ordered_before`** conditions to `EffectChecker`; prerequisite is stamping effects with `virtual_time`. Keep the operator set minimal (only what the 2 slice seeds need). |
| **F6** | Consumption/ablation gate | Difficulty is proven by **negative** trajectories: a **no-wait** run (act on the synchronous stale value) and an **out-of-order** run must FAIL — the async/ordering analogue of the M7 vision-blind ablation. |
| **F7** | Back-compat | Everything **behaviour-gated** on `force_async_wait` (and on the presence of a temporal check). Non-opted seeds/effects stay byte-identical; run the M0 regression gate. |

#### Refinements folded in after auditing the two slice seeds (F8–F11)

Reading `multi_device_pairing.json`, `sla_support_response.json`, `pairing_skill.py`,
and the eval-state path sharpened the plan and split it into two CLs:

| # | Finding | Consequence |
|---|---|---|
| **F8** | **`multi_device_pairing` is a pure L2 seed.** The `pairing` skill already enforces order **and** the 300 s window at runtime (`pairing_skill.py`: every step re-checks `_rec_expired` + FSM state; `confirm` emits `pairing_completed` **only** on an in-order, in-window chain, `:257-281`). So *if `pairing_completed` exists, order + timeliness held.* No sub-agent/async dimension (`agents` family is narrative sign-off only). | Pairing needs **no** `force_async_wait`, **no** `environment.py`/`effect_checker.py` change — just grade the existing `pairing_completed` effect as `required` with the existing `exists` operator. This becomes **CL1** (low risk). |
| **F9** | **`sla_support_response` has no protocol skill.** The response is a generic `message`; SLA compliance is narrative. Nothing at runtime enforces "waited for the delegated result" or "responded before the window." | This is where L1 (`force_async_wait`) + L3 (`within_window`) do the real work. Becomes **CL2**. |
| **F10** | **`within_window` is the sole P0 operator; `ordered_before` is deferred.** If `force_async_wait` routes the decisive value **only** through the delayed `subagent_completion`, then "the value appears in the response" (existing `field_contains`) already proves the agent waited; `within_window` adds the on-time assertion. | CL2 implements **`within_window` only**. `ordered_before` moves to the deferred list (revisit after CL2 eval). Supersedes the two-operator claim in F5. |
| **F11** | **Byte-safety is narrower than F7 implied.** Graded state is built by `_get_evaluation_state` (`environment.py:1031-1071`, `"effects": dict(self._effects)`) and in-memory snapshots (`:639`); neither is written into any generated scenario JSON. | Stamping `virtual_time` on runtime effects does **not** affect the generated-data byte gate (E6/M0). The only churn is unit tests that snapshot effect dicts — update those. Chosen approach: **stamp `virtual_time` on the effect dicts** (simplest), not a side-channel. |

**CL split:**
- **CL1 — `multi_device_pairing` / L2:** grade `pairing_completed` as `required`; strict-validator rule "a coordination seed must ship a required protocol-effect check"; out-of-order/expired ablation fails. No runtime/grader code. Deliverables 5 (partial) + tests below. **✅ shipped — but the deliverable changed during implementation: "grade as required" was already automatic, so CL1 landed as an opt-in gold-trajectory *completion guard*. See §10.8.E for the as-built record.**
 - **CL2 — `sla_support_response` / L1+L3:** `virtual_time` stamping (Deliv. 7), `within_window` (Deliv. 8, P0 only), `force_async_wait` knob + pipeline/materializer wiring (Deliv. 1–4), sla authoring (Deliv. 9). Sequenced after CL1's eval. **✅ shipped (code + tests, offline-verified) — with 4 as-built deviations from the deliverable framing (sender-scoped `within_window`; temporal check injected in the *materializer* not the scaffolder; effect stamping always-on and made byte-safe at the serialization boundary rather than by gating; background effects stamped at `trigger_time`). See §10.8.F for the as-built record.**

#### A. Deliverables (files)

**L1 — force the reference trajectory to actually go async (`agent delegate` → `wait` → read)**
1. `clasharness/scenario/conflict_vlm_spec.py` — add `force_async_wait:bool=False`
   to `ConflictVLMSeed` (next to `force_vision_locus`, `:233`) + parse in `from_dict`
   (`:268`). Pure template copy of the vision knob.
2. `clasharness/scenario/conflict_llm_pipeline.py` — when `seed.force_async_wait`,
   the round generator must route the **decisive value** so it is only obtainable
   from a sub-agent that completes **after a delay** (delegate → `subagent_completion`
   at `trigger_time`), **not** from a synchronous `message send` that returns a
   pre-written `mock_response`. Concretely: (a) strengthen the outline prompt
   (`:1692,1696`) from a cosmetic "Wait" to a REQUIRED `agent delegate ...` + `wait N`
   + read-result triad; (b) add a hard reject in the round-validation path (mirroring
   the `force_vision_locus` gate at `:1638`) if the emitted reference commands lack a
   `delegate` followed by a `wait` that covers the sub-agent's
   `virtual_duration_seconds`.
3. `conflict_llm_pipeline.py` `_scaffold_round_checkers` (`:3648`, called `:1922`) —
   today a `wait` step yields **no checker** (`:3844`). Under `force_async_wait`, emit
   a checker for the delegated result (an `effect` on the delegated-completion effect
   type) so "the slow sub-agent's result was actually consumed" is graded — combined
   with the L3 ordering operator to require it happened **after** the wait.
4. `clasharness/scenario/conflict_vlm_materializer.py` — ensure `_inject_background_wait`
   (`:3059`, called `:2013`) still covers the delegated event's trigger time, and that
   the mock sub-agent for the decisive value is delivered via the **delayed**
   completion path, not the synchronous `message send` mirror.

**L2 — make coordination/domain actions actually graded**
5. Target seeds only (first slice): ensure the reference trajectory exercises the full
   protocol contract and that the grader asserts the protocol **effects** (not just
   asset/file/message counts). The fail-closed skills already make "effect exists" a
   real signal:
   - `sla_support_response` (`time_window_expiration`): assert the response/booking
     effect exists **and** (L3) occurred **before** the SLA window closed.
   - `multi_device_pairing`: assert all four pairing effects
     (start→show-code→enter-code→confirm) exist **and** (L3) occur **in order** and
     **within** the 300 s window. The `pairing` FSM guard
     (`clasharness/skills/impl/pairing_skill.py`) already rejects out-of-order steps
     at runtime, so the ordering check mainly hardens the *grader* against a trajectory
     that games effect emission.
6. `conflict_llm_pipeline.py` — where `protocol_effect` commands already scaffold an
   `effect/<type>/exists` checker (`:3839-3843`), extend to attach the L3 temporal
   `expected` fields for async/temporal seeds.

**L3 — give the grader temporal/ordering operators**
7. **Prerequisite — stamp effects with `virtual_time`.** In
   `clasharness/core/environment.py`, record `virtual_time = self._virtual_time` at
   effect append/merge sites (`:886`, `:997-1005`, protocol/state merges `:1170-1206`,
   agent-status `:458/:467`, and delegated `subagent_completion` via
   `_apply_background_event`). **Byte-safety:** existing operators read
   `e.get(field)` and ignore extra keys, but serialized `env_state` snapshots may diff
   — gate the extra field on "a temporal check consumes this effect type" (or carry
   the timestamp in a side map keyed by effect identity) so non-temporal grading stays
   byte-identical. Verify against the M0 regression gate.
8. `clasharness/evaluation/checkers/effect_checker.py` — extend the `evaluate`
   if/elif ladder (`:86-224`) and `from_spec` (`:234-242`) with:
   - **`within_window`** — `expected={"max_virtual_time": T}` (or a reference effect):
     passes if the relevant effect's `virtual_time` ≤ deadline (e.g. respond before
     SLA close; confirm within 300 s).
   - **`ordered_before`** — `expected={"other_effect_type": ..., "criterion"?: ...}`:
     passes if this effect's `virtual_time` precedes the referenced effect's (e.g.
     delegate < wait < read; pairing steps strictly increasing).
   - (optional, only if a slice seed needs it) **`count_within_window`** — N effects
     inside one window.
   Keep the set minimal; do not add speculative operators.
9. **Seeds (first slice):** author/regenerate `sla_support_response.json` +
   `multi_device_pairing.json` with `force_async_wait:true` and the L3 temporal checks.
   (The seeds' `core_conflict`/`constraints` text already says "advance virtual time
   only with 'wait'" / "respond before the SLA window closes" — L1–L3 make that text
   enforceable.)
10. **Tests:** `within_window`/`ordered_before` pass & fail cases; effects carry
    `virtual_time` only when a temporal check consumes them; non-opted seed
    byte-identical (M0 gate); `_scaffold_round_checkers` emits the async checkers only
    under `force_async_wait`; a synthetic no-wait / out-of-order trajectory fails.

#### B. Glue (surfaces touched)

| # | Surface | Change |
|---|---|---|
| 1 | `conflict_vlm_spec.py` | `force_async_wait` knob + `from_dict` (Deliv. 1) |
| 2 | `conflict_llm_pipeline.py` outline + round-validation | require `delegate → wait → read`; reject synchronous shortcut (Deliv. 2) |
| 3 | `_scaffold_round_checkers` (`:3648`) | emit delegated-result + temporal checkers under the knob (Deliv. 3,6) |
| 4 | `conflict_vlm_materializer.py` | deliver decisive value via delayed completion; wait covers trigger time (Deliv. 4) |
| 5 | `core/environment.py` effect sinks | stamp `virtual_time` (gated) (Deliv. 7) |
| 6 | `evaluation/checkers/effect_checker.py` | `within_window` + `ordered_before` (+`from_spec`) (Deliv. 8) |
| 7 | seed JSON ×2 | `force_async_wait:true` + temporal checks (Deliv. 9) |
| 8 | `conflict_strict_validator.py` | (M6-style) structural rule: async seed must ship a delegate+wait reference + a temporal check |
| 9 | generate → validate → eval | user runs; see §C |

#### C. Verification / exit gate

1. **Unit (agent):** new operators pass/fail correctly; `virtual_time` stamped only
   when consumed; non-opted seed byte-identical (M0/E6 gate); scaffolder emits async
   checkers only under `force_async_wait`.
2. **Mock smoke (agent, offline):** regenerate the 2 seeds `--provider mock
   --image_provider mock --chart_provider template`; the gold trajectory contains
   `agent delegate` → `wait N` → read-result and passes; a synthetic **no-wait** and
   an **out-of-order** trajectory FAIL.
3. **Vertex run (user):** regenerate the 2 seeds `--resume --retries 5`; strict
   validator 0 FLAGGED; eval gold ≈ 1.0.
4. **Consumption/difficulty ablation (user):** a **no-wait** ablation (act on the
   synchronous stale value) and an **out-of-order confirm** ablation collapse pass
   rate ⇒ async + ordering are load-bearing (the temporal analogue of the M7
   vision-blind ablation).

#### D. Risks

- **Byte-drift from timestamping effects** — the biggest back-compat risk; mitigate by
  gating the `virtual_time` field on temporal consumption (or a side map) + M0 gate.
- **Gold can't self-solve after hardening** — gate exit on gold ≈ 1.0; keep the gold
  generator delegate/wait-capable.
- **`force_async_wait` produces an ungradeable "wait theatre"** — require the delegated
  result to be *consumed* (Deliv. 3) and ordered after the wait (Deliv. 8), not merely
  that a `wait` token appears.
- **Mock sub-agent still answers synchronously** — Deliv. 4 must route the decisive
  value through the delayed `subagent_completion` path, not the `message send` mirror
  (`environment.py:~1080-1167`).
- **Scope creep into L4/L5** — L4 (concurrent sub-agents) and Axis B/L5 stay non-goals
  (F1); revisit only after the Cap 2 slice shows an eval delta.

> **CL sequencing (agreed):**
> - **CL1 = `multi_device_pairing` / L2** — ✅ **shipped as an opt-in gold-trajectory
>   *completion guard*** (`enforce_protocol_completion`, gated on the new
>   `require_protocol_completion` seed flag). Grading `pairing_completed` as `required`
>   turned out to be already automatic (the `protocol_effect` scaffolder + fail-closed
>   `pairing_skill`), so CL1's real work is preventing gold *coordination theatre*
>   (a gold that never runs `pairing confirm`). No `environment.py`/`effect_checker.py`
>   change. Full as-built record in §10.8.E.
> - **CL2 = `sla_support_response` / L1+L3** — ✅ **shipped**: stamp `virtual_time`
>   (Deliv. 7) + sender-scoped `within_window` **only** (Deliv. 8; `ordered_before`
>   deferred) + `force_async_wait` knob & pipeline/materializer wiring (Deliv. 1–4) +
>   the async-wait structural contract in the strict validator (Glue #8) + sla
>   authoring (Deliv. 9). No new grader op beyond `within_window`; consumption/ordering
>   is enforced *structurally* (the delayed reply only lands in `messages_sent` after a
>   `wait`) rather than via a separate `ordered_before` check. Full as-built record in
>   §10.8.F. Awaits (user) Vertex regenerate + eval with the no-wait / late-response
>   ablations.
> - **Deferred (post-CL2 eval):** `ordered_before`, `count_within_window`, and L2 breadth
>   to other protocol seeds.

#### E. CL1 as-built (implemented — `multi_device_pairing` / L2)

> **Status: ✅ shipped (code + tests, offline-verified); awaits user Vertex
> regeneration to confirm on real data.** Implementing CL1 showed the original F8
> framing ("grade `pairing_completed` as `required`") was **already satisfied by
> existing machinery**; the real, un-covered gap was gold-trajectory *coordination
> theatre*. CL1 therefore shipped as a deterministic **completion guard**, not a new
> "required-effect" grader rule.

**Refinements found during implementation (F12–F15):**

| # | Finding | Consequence |
|---|---|---|
| **F12** | **"Grade `pairing_completed` as required" is already automatic.** `pairing confirm` is classified as a `protocol_effect` by `_classify_step_command`, so `_scaffold_round_checkers` (`conflict_llm_pipeline.py:3833-3843`) already emits `{effect_type: pairing_completed, condition: exists}`; with no `role` it defaults to **required** and the materializer does not downgrade it. Runtime `pairing_skill.py` is fail-closed (out-of-order / expired → no effect). So *if the gold trajectory runs `confirm`, order + timeliness are already enforced and graded.* No new grader rule needed. |
| **F13** | **The real gap is gold *coordination theatre*.** Nothing forced the graded gold trajectory to ever execute `pairing confirm`; a degraded gold that only reads status emits no effect, so the required check is never exercised and the scenario silently grades nothing. This is CL1's actual target. |
| **F14** | **The guard must be opt-in, not `emits_effects`-driven.** `coordinated_containment` declares `emits_effects=("containment_action",)` but its shipped gold only *reads* `containment status` — already theatre. A guard keyed on plugin `emits_effects` would retroactively fail the 3 shipped containment packages, violating E6. So the guard is gated on a new **opt-in seed flag** `require_protocol_completion` (default `False`); only `multi_device_pairing` opts in. |
| **F15** | **Wire the *narrow* guard, not the full strict validator, into `materialize()`.** Calling the full `validate_conflict_scenario_strict` from generation newly enforced its *other* checks (e.g. temporal-staleness needs `trigger_virtual_time`) and broke two existing `TestConflictResume` fixtures. Shipped a narrow public entry `enforce_protocol_completion(spec, seed)` that runs **only** the completion check; the other strict checks stay post-generation test backstops. |

**Shipped (files):**
- `conflict_vlm_spec.py` — `ConflictVLMSeed.require_protocol_completion: bool = False` (field + `from_dict`).
- `conflict_llm_pipeline.py` — strip `require_protocol_completion` from the `seed_dict` prompt copy (E6: enabling the flag changes no LLM prompt bytes).
- `conflict_strict_validator.py` — completion guard: `enforce_protocol_completion` (narrow public entry) + `_validate_protocol_completion` + `_required_completion_effects` (archetype → plugin `emits_effects`) + `_completion_command_prefixes` (reverse `_EFFECT_COMMAND_NAMES`); also added as check #6 of `validate_conflict_scenario_strict(spec, seed=None)`.
- `conflict_vlm_materializer.py` — call `enforce_protocol_completion(spec, seed)` after the update-id repair loop (no-op unless the seed opts in).
- `data/scenario_seeds/conflict_vlm_v1/multi_device_pairing.json` — `require_protocol_completion: true`.
- `tests/test_conflict_pipeline.py` — `TestProtocolCompletionGuard` (10 tests).

**Verification (done, offline):**
- `TestProtocolCompletionGuard` (10) + the 2 previously-failing `TestConflictResume` platform-style tests: green.
- Full `tests/test_conflict_pipeline.py`: **346 passed**.
- Full `tests/`: **1598 passed**; the only failures are 2 pre-existing/environmental issues unrelated to CL1 (missing `incident_commander_selection_v01` data dir → `FileNotFoundError` before any changed code runs; headless chart-render quality floor `degenerate 8x8`).
- **E6/M0 byte gate:** 20 seeds — only `multi_device_pairing` opts in; the flag is stripped from every prompt payload and appears in **none** of the 18 existing generated `*_spec.json` (the `ConflictVLMScenarioSpec` dataclass has no `seed` field, so `to_dict()` cannot leak it).

**Deferred to user:** regenerate `multi_device_pairing` on Vertex; confirm the guard fires on a degraded (never-`confirm`) gold and that eval gold ≈ 1.0 on a proper trajectory.

#### F. CL2 as-built (implemented — `sla_support_response` / L1+L3)

> **Status: ✅ shipped (code + tests, offline-verified); awaits user Vertex
> regeneration to confirm on real data.** CL2 makes "delegate to a slow sub-agent,
> `wait` for it, and act on its reply *before the SLA window closes*" a graded,
> load-bearing requirement (the temporal analogue of M7's vision-blind ablation).
> Implementing it kept the deliverable *shape* (L1 forced-async trajectory, L3
> temporal operator, `force_async_wait` knob) but landed **4 deviations** from the
> Deliv. 3/7/8 framing — each because the value that must be graded (a *specific* slow
> sub-agent's reply, timed at its true completion) is only knowable at materialize
> time and never crosses the scenario-JSON serialization boundary.

**Refinements found during implementation (F16–F19):**

| # | Finding | Consequence |
|---|---------|-------------|
| **F16** | **`within_window` shipped *sender-scoped*, not value-async-only.** F10/Deliv. 8 framed the temporal op as generic. But the decisive signal is "*the slow sub-agent's* reply landed on time", not "*some* effect did" — a throwaway early `messages_sent` would otherwise satisfy an on-time assertion. | `within_window` carries an optional `expected.field`/`expected.value` content filter (default `sender=<slowest role>`), matched by `_effect_field_contains` (same tokenized/substring logic as `field_contains`). Fail-closed on absent/unstamped/past-deadline effects; a `bool` `virtual_time` is explicitly **not** a valid stamp (`effect_checker.py:264-268`). |
| **F17** | **The temporal check is injected in the *materializer*, not `_scaffold_round_checkers` (Deliv. 3).** The `within_window` deadline and sender are derived from the **materialized** `mock_subagents` (roles, `virtual_duration_seconds`) and `background_events` (`trigger_time`), which don't exist at scaffold time. | New `_build_async_within_window_check(mock_subagents, background_events)` (`conflict_vlm_materializer.py:3094`) picks the **slowest** sub-agent as `sender` and `max_virtual_time = max(background trigger_time)` (the "derive = max background trigger" decision), and the materializer appends it **gated on `seed.force_async_wait`** (`:2046-2047`). The scaffolder path is untouched (E6). Returns `None` (no check) when a round has no mock sub-agent or no delayed event. |
| **F18** | **Effect `virtual_time` stamping is *always-on*, made byte-safe at the serialization boundary — not gated on temporal consumption (Deliv. 7 / Risk D).** Deliv. 7 proposed gating the stamp on "a temporal check consumes this effect type" (or a side map) to avoid byte-drift. | Unnecessary: the stamp lives only on in-memory `_effects`/`env_state` dicts, which are **never serialized into the generated scenario JSON** (`ConflictVLMScenarioSpec` has no effects field). So `_stamp_new_effects` (`environment.py:1259`) runs unconditionally (one code path, no side map), and E6 is preserved by the *serialization boundary*. **Verified offline:** a materialized non-opted seed's task JSON contains **0** `within_window` checks and no leaked `virtual_time`. |
| **F19** | **Background effects are stamped at `trigger_time`, not the post-tick "current clock".** Deliv. 7 listed `_apply_background_event` as a stamp site but not *which* time. A sub-agent completing at `trigger_time=10` must read `virtual_time=10` even if the solver over-waits to vt=60 — otherwise a correct-but-over-waiting solver is scored late. | `_apply_background_event` stamps at `event["trigger_time"]` (`environment.py:941`, `_stamp_new_effects(at=...)`); solver actions stamp at the current clock (`at=None`). The reference trajectory (one injected `wait` reaching `max_trigger`) is on time **by construction**, and over-waiting is not penalized. |
| **F20** | **The forced-async gate (F6/`:1951`) was a *pure reject* and killed generation.** During the first Vertex regen, all 3 `sla_support_response` versions failed spec-gen: Vertex writes a synchronous `message send` to the SLA monitor instead of `agent delegate`, so `_has_delegate_then_wait` rejected every attempt and — because `max_attempts` is hardcoded to 3 and the gate offered **no deterministic repair** (unlike final-verification/DONE/quoting, which all self-repair) — the round exhausted retries and the scenario was dropped. | New `_ensure_async_delegate_wait(commands, mock_subagents, allowed_families)` (`conflict_llm_pipeline.py:243`) mirrors `_ensure_final_verification_command`: when a `force_async_wait` round lacks delegate→wait, it synthesizes `agent delegate --agent <slowest sub-agent> …` + `wait <duration>` and splices them before the terminal DONE. It targets the **slowest** mock sub-agent so it matches the role `_build_async_within_window_check` binds the temporal check to (F17). Called in the command loop **before** `_ensure_final_verification_command` (so the injected inspect lands after the delegate action) and before the gate (`:2019`, gated on `force_async_wait`); the gate at `:2034` is now a **backstop** for the no-mock-sub-agent case (returns `False` → gate raises as before, no regression). |

**Design decisions carried through (user-approved):** (1) `within_window` is scoped to
the slow sub-agent's `sender`; (2) the derived SLA deadline is the round's **max**
background `trigger_time`. **Consumption/ordering is enforced structurally, not by a new
op:** the delayed reply only appears in `messages_sent` *after* the solver `wait`s past
`trigger_time`, so `ordered_before` was **deferred** (F1/scope) — a no-wait trajectory
produces no on-time sender effect and fails closed.

**Shipped (files):**
- `conflict_vlm_spec.py` — `ConflictVLMSeed.force_async_wait: bool = False` (field + `from_dict`), next to `force_vision_locus`.
- `conflict_llm_pipeline.py` — **L1 forced-async trajectory:** outline/round prompts require an `agent delegate … → wait N → consume-reply` triad under the knob (`:1739`, `:1842`); a **deterministic repair** (`_ensure_async_delegate_wait`, `:243`; **F20**) synthesizes the missing `agent delegate <slowest sub-agent> → wait` before the gate; the gate (`_has_delegate_then_wait`, `:3304+`; delegate detected by tool-spec classification so a synchronous `message send` does **not** count; `wait` matched by prefix; order enforced) is now a **backstop** that raises `ValueError` (→ standard retry-with-feedback) only when no repair could be built (no mock sub-agent). Also strips `force_async_wait` from the `seed_dict` prompt copy (E6: enabling the flag changes no LLM prompt bytes).
- `core/environment.py` — **Deliv. 7:** `_stamp_new_effects(at=None)` (`:1259`) stamps every new effect lacking `virtual_time`; solver sinks stamp at the current clock, `_apply_background_event` stamps at `trigger_time` (`:941`); the delayed sub-agent reply lands via `_record_subagent_message`.
- `evaluation/checkers/effect_checker.py` — **Deliv. 8:** `within_window` operator (`:237`) + `_effect_field_contains` (`:38`) + `from_spec` wiring; `ordered_before`/`count_within_window` **not** shipped (deferred).
- `conflict_vlm_materializer.py` — `_build_async_within_window_check` (`:3094`) + injection gated on `force_async_wait` (`:2046`); calls `enforce_async_wait_contract(spec, seed)` (`:244`, import `:20`).
- `conflict_strict_validator.py` — **Glue #8, async-wait structural contract:** `enforce_async_wait_contract` (narrow public entry, mirrors CL1's `enforce_protocol_completion`) + `_validate_async_wait_contract` (seed-gated; asserts ① some round declares a delayed mock sub-agent and ② some such round's gold `solution_commands` delegates-then-waits) + `_round_has_delayed_subagent` + `_gold_delegates_then_waits` (lazy-imports the pipeline's `_has_delegate_then_wait` — single source of truth, no drift). Added as check #7 of `validate_conflict_scenario_strict` (`:204`).
- `data/scenario_seeds/conflict_vlm_v1/sla_support_response.json` — `force_async_wait: true` + `subagent_delay` in `conflict_taxonomy`; `core_conflict`/`required_constraints` reframed so the decisive on-time confirmation comes from a **delayed** delegated sub-agent (delegate → wait → consume required).
- `tests/test_conflict_pipeline.py` — `TestWithinWindowChecker` (7), `TestAsyncWithinWindowInjection` (4), `TestAsyncWaitContract` (13), `TestSlaSeedOptsIntoAsync` (2), `TestEnsureAsyncDelegateWait` (4, **F20**) = **30**.
- `tests/test_virtual_clock.py` — `test_background_effect_stamped_at_trigger_time_not_overwait`, `test_solver_effect_stamped_at_current_clock` (2).

**Verification (done, offline):**
- CL2 subset: **30** pipeline (incl. `TestEnsureAsyncDelegateWait`, F20) + **2** clock tests green.
- Full `tests/test_conflict_pipeline.py` + `tests/test_virtual_clock.py`: **381 passed**.
- **F20 integration check:** a synthetic synchronous SLA trajectory (asset→file→`message send`→DONE) run through the exact loop order `_ensure_async_delegate_wait → _ensure_final_verification_command → _validate_generated_round_commands → gate` yields `… → agent delegate Iris_SLA_Monitor → wait 40 → file read → DONE`, passing the validator and gate.
- Effect/state/clock suites: green apart from the same 2 pre-existing/environmental failures noted under CL1 (missing `incident_commander_selection_v01` data dir; headless chart-render `degenerate 8x8`) — both fail before any CL2 code runs.
- **E6/M0 byte gate:** only `sla_support_response` opts in; `force_async_wait` is stripped from every prompt payload, the `within_window` check materializes for **no** non-opted seed, and the `virtual_time` stamp never reaches serialized scenario JSON (in-memory `_effects` only).

**Deferred to user:** regenerate `sla_support_response` on Vertex (strict 0 FLAGGED; gold ≈ 1.0); confirm the **no-wait** ablation (act on the synchronous stale value) and the **late-response** ablation collapse the pass rate ⇒ async + temporal are load-bearing.

**Deferred (post-CL2 eval):** `ordered_before`, `count_within_window`, and L2 breadth to other protocol/async seeds (e.g. `multi_device_pairing`'s temporal window).
