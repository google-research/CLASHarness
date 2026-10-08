# Changelog

> Maintainer changelog of recent benchmark-surface changes. For the pipeline
> design see the root
> [Multi-Agent_GENERATION_PIPELINE.md](../Multi-Agent_GENERATION_PIPELINE.md); for
> the coordination-protocol roadmap and current build status see
> [conflict-expansion-plan.md](conflict-expansion-plan.md). This file is **not** the
> canonical source for current release defaults.

## Current benchmark surface

This is a multi-agent computer-use benchmark. A coordinator agent inspects state,
acts through CLI commands, and is scored by resulting state and side effects. The
repository is centered on a single line:

- `conflict_vlm_v1` — the multi-agent + temporal conflict VLM line. A coordinator
  must reconcile subagents whose diagnoses disagree, cross-reference chart / image /
  rich-text assets, and drive the workflow to a correct terminal action.
  **Multi-agent conflict and temporal (timeout / staleness) conflict are the core
  of this line.** It is under active development.

Current counts:

- `20` `conflict_vlm_v1` seed fixtures (pipeline fixtures that evolve with the
  generator, not yet a frozen `train`/`dev`/`test` release profile).

## Conflict-VLM line (`conflict_vlm_v1`)

### What it is

Multi-agent coordination under conflicting, time-sensitive, multimodal evidence.
A coordinator must reconcile subagents whose diagnoses disagree, cross-reference
chart / image / rich-text assets, and drive the workflow to a correct terminal
action.

### Current status: M0 + M1 + M2 + M3 + M4 + M5 built

`clasharness/data/scenario_seeds/conflict_vlm_v1/` currently holds `20` seeds:

- `clinical_trial_dosage_conflict`
- `incident_commander_selection`
- `multimodal_temporal_conflict_01`
- `payment_chargeback_dispute`
- `satellite_ground_station_handover`
- `warehouse_robotics_safety_halt`
- `distributed_lock_shared_resource` *(M1)*
- `barrier_sync_deployment` *(M1)*
- `expense_reimbursement_conflict` *(M1)*
- `employee_onboarding_workflow` *(M2)*
- `it_deployment_gate` *(M2)*
- `sla_support_response` *(M3)*
- `expiring_resource_hold` *(M3)*
- `multi_device_pairing` *(M3)*
- `multi_party_access_permit` *(M4)*
- `deadlock_multi_approval` *(M4)*
- `coordinated_containment` *(M4)*
- `collaborative_contract_review` *(M4)*
- `multi_tab_slot_monitoring` *(M4)*
- `access_request_approval` *(M5)*

Seeds are still pipeline fixtures that evolve with the generator, not a frozen
release profile. All 15 conflict archetypes are now represented across these
seeds (the six seeds above the M1 marker are the original pre-taxonomy fixtures);
`category_coverage` reports all 5 top-level categories covered.

Expansion progress against [conflict-expansion-plan.md](conflict-expansion-plan.md):

- **M0 (foundations) — built.** `StatefulProtocolSkill`
  (`clasharness/skills/stateful.py`), generic `_track_effects`, a shared
  `ALLOWED_EFFECT_TYPES`, the scaffolder registry, and the `archetype` seed tag.
- **M1 (vertical slice) — built.** The `lock` and `deploy` skills
  (`clasharness/skills/impl/lock_skill.py`, `deploy_skill.py`), the
  `distributed_lock` and `barrier_synchronization` plugins, and the three M1 seeds.
- **M2 (Order class) — built.** The `controlled_deployment_gate` plugin
  (approvals-as-gates governance authorization ordering) plus two Order-class
  seeds — `employee_onboarding_workflow` (reuses the `dependency_ordering` plugin,
  drives `deploy execute` → `deployment_executed`) and `it_deployment_gate`
  (drives `deploy cutover` → `deployment_cutover`). Both ride the existing M1
  `deploy` skill family, so no new skill family was required.
- **M3 (Timeout class) — built.** The `booking` (reversible resource hold) and
  `pairing` (ordered handshake) skills, real virtual-time plumbing into
  `StatefulProtocolSkill` (`_now`/`_expired`, published by the environment before
  each dispatch and after every `wait`), the `reversible_action_hold` and
  `multi_device_pairing` plugins, and three seeds — `sla_support_response` (#7,
  reuses `time_window_expiration`, no new family), `expiring_resource_hold` (#8,
  `booking`) and `multi_device_pairing` (#9, `pairing`). Holds/codes expire against
  the virtual clock, so confirming after the window closes fails and emits no
  effect.
- **M4 (SubAgent-Collaboration class) — built.** The `approval`
  (`request|approve|grant|revoke|provisional|escalate|status`), `containment`
  (`revoke-sessions|disable-forwarding|isolate-device|block-token|status`), and
  `review` (`assign|submit|sign|status`) skills, the `scope_reconciliation`,
  `multi_party_synchronization`, `coordinated_containment`,
  `non_overlapping_assignment`, and `duplicate_hold_stale` plugins, and five seeds —
  `multi_party_access_permit` (#3), `deadlock_multi_approval` (#14),
  `coordinated_containment` (#4), `collaborative_contract_review` (#5, `review`+`lock`),
  and `multi_tab_slot_monitoring` (#6, reuses `booking`). This completes all `7/7`
  new skill families and `10/10` new plugins.
- **M5 (Main-Supervised class) — built.** The `access_request_approval` seed (#2)
  built on the M4 `approval` family and the `scope_reconciliation` plugin (no new
  skill/plugin/MCP tool), plus `archetype` tags backfilled on the reused
  `incident_commander_selection` (#1 → `evidence_mismatch`) and
  `sla_support_response` (#7 → `time_window_expiration`) seeds. Added a coverage
  layer in `conflict_vlm_spec.py`: the authoritative `ARCHETYPE_CATEGORY`
  token→category map, a `category_coverage()` helper, and
  `tests/test_archetype_coverage.py` enforcing 15/15 archetype and
  5/5 category coverage. Brings `conflict_vlm_v1` to `20` seeds with all 15
  archetypes present.
- **Remaining.** Milestone **M6** (scale + baseline).
  New conflict types come from extending the MCP / tool surface with
  coordination-protocol families, not from hand-writing more one-off seeds. Adding
  one family is the vertical stack in
  [conflict-expansion-plan.md §3](conflict-expansion-plan.md).

### Recent changes

- **v8 is the published corpus.** The dataset release is the corpus the paper was computed
  on: 600 rounds from the 20 seeds, generated with `gemini-3.6-flash`, together with the
  14-model panel and effort sweep, the `claude-sonnet-5` generator ablation (arm B, released
  as it was run), the 30- and 40-step budget arms and the text-only arm. That is eleven components, 26,853 files and 46.3 GB,
  re-measured on the published tree; `eval-budget30` is 440 MB because the incomplete
  `gemini-3.7-flash` cell, which is in neither the paper's budget ablation nor the other two
  ablation arms, is left out. `CHECKSUMS.txt` holds one digest per component. No v8 file was
  changed and the paper's numbers stand. An audit of the v8 checks after submission found
  checks a fair grader would not count; the dataset card's
  [Known issues](dataset-card.md#known-issues) section gives the counts and how much they
  move the results, `analysis/v8_check_fairness_audit.json` lists the checks, and
  `analysis/sonnet5_gold_replay_audit.json` lists the arm-B checks the reference solution
  fails. The generator in this repository has changed since v8 was generated, so a corpus
  generated with it is not v8 and its difficulty has not been calibrated. Corpus version
  numbers other than v8 that appear in this file or in the docs name unpublished development
  runs.
- **Claude and GPT judges get room to reach their verdict.** Their output cap was 256
  tokens. On a multi-part criterion `claude-sonnet-5-5` checks each element in prose
  before the JSON verdict (348 to 545 tokens in a probe), and the Responses API counts a
  gpt-5-class model's reasoning tokens against the same cap, so the reply was cut off
  before the verdict and the check came back unparseable (`ok=False`), most often on the
  borderline checks. A generation run with a Claude judge therefore skipped its
  gold-satisfiability gate on exactly those checks; the generation logs of the arm-B
  corpus (generator `claude-sonnet-5`) record 83 such skips. The cap is now 4096; unused
  headroom is not billed. Gemini judges are held to a JSON-only reply, have no cap and are
  unchanged, so grades from a Gemini judge, as used for v8, are unaffected. A Claude judge
  that rejects sampling parameters (Sonnet 5.x, Opus 4.7 and later) also gets no
  `temperature`, so `judge_majority` votes are sampled at the model default rather than
  greedily; the run report now says so under its `Judge :` line. Verified: 8 tests in
  `tests/test_llm_judge.py` and `tests/test_judge_health.py`.
- **The generator's judge stays on Vertex AI when generation does.**
  `generate_conflict_vlm_scenarios.py --provider vertex` points its judge at the generation
  provider through `sync_judge_env`, which wrote `LLM_JUDGE_PROVIDER=gemini`; `_judge_gemini`
  then picked the Gemini Developer API whenever `GEMINI_API_KEY` was set, so with a key
  exported the generation gates' judge left Vertex AI while generation stayed on it.
  `LLM_JUDGE_PROVIDER=vertex` is now kept and ignores the key; `gemini` still prefers it.
  The Claude-on-Vertex judge also falls back to
  `GOOGLE_CLOUD_PROJECT` / `GEMINI_PROJECT` and `GOOGLE_CLOUD_LOCATION`, as the generation
  client does, instead of requiring `VERTEX_PROJECT`. Verified: 3 tests in
  `tests/test_llm_judge.py`.
- **Release tools: the eval visualizer works on a checkout without scenarios, and
  `download_data.py` does what was named.** `generate_eval_visualizer.py` and
  `eval_report.py` fell back to an undefined `py_file_path` whenever a scenario was not
  under `clasharness/data/scenarios`, which is always the case in a release checkout, so
  the `NameError` aborted `visualize.html` (15 times in one set of evaluation logs). Both
  now look under the checkout root and then under the source tree the script belongs to.
  In `tools/download_data.py`, `--component`, `--all` and `--default` are mutually
  exclusive, where `--all` or `--default` used to win silently over a named component;
  `--list` and the download plan print totals summed from the manifest instead of a
  hand-written figure; a component that arrives missing or empty ends the run with an
  error instead of `Done.`; and the closing message gives the `--task-data-dir` path for
  each task component. Verified: 5 tests in `tests/test_eval_visualizer_assets.py` and 6 in
  `tests/test_release_tools.py`.
- **Vision-exclusivity instrumentation + honest `asset` tool descriptions.** Closed the
  gap between "this round has a decisive value" and "this round actually requires
  *looking* at the asset". Three layers, documented as §D in
  [Multi-Agent_GENERATION_PIPELINE.md](../Multi-Agent_GENERATION_PIPELINE.md):
  (1) the generation-time confidentiality gate now scans all four agent-visible text
  surfaces (instruction, non-locus workspace files, chat/session, and mock
  `mock_response` + `state_effects`), with a deterministic mock rewrite ahead of it so
  a same-round collision is repaired rather than burning ~25% of the retry budget;
  replaying the current gate over the two shipped corpora rejects **170/178 leaking
  rounds on 3.5-flash (95.5%)** and **143/144 on 3.6-flash (99.3%)**, the 9 residuals
  all being whitespace variants (`45 ms` vs `45ms`).
  (2) `_record_vision_outcome` splits the render-time fidelity audit's single boolean
  into *confirmed legible* (`vision_verified=True`) vs *never actually checked*
  (`vision_audit_skipped=<reason>`, still treated as a pass so a flaky audit cannot
  downgrade a good asset) vs *positively illegible* (`fidelity_failed` + downgrade to
  `corroborating`); the audit also covers the headless-Chrome chart/HTML channel, not
  images only — chart-family assets are **~93–95% of all decisive assets**, so the
  image-only version left almost the whole population unverified.
  (3) `_vision_exclusivity_report` measures the finished dataset off the **compiled
  `task_spec`** rather than the authoring spec (repair layers rewrite it; 299/675
  updates on 3.6-flash ship to no round; `initial_state_overrides` app-state docs are
  invisible to `spec.workspace.files`), classifying every round into seven verdicts and
  warning explicitly when a round had no compiled task to read. It reports only and
  repairs nothing. Separately, `read_asset_text` and `describe_asset` advertised OCR /
  alt-text / "evidence anchors" that `AssetBackend` has never returned — both MCP tool
  descriptions and the `mcp_tools_golden` fixture now state the real behaviour and
  point at `open_asset`. The same two claims also lived on a second agent-visible
  surface, `clasharness/specs/agent_tools.v1.json` (`asset.read_text`,
  `asset.describe`), which reaches the solver as the *native function-calling*
  description via `_tool_def_from_spec`; both now state the real behaviour, in the
  spec file's own house style (single semicolon-joined sentence, no cross-reference
  to a sibling command, since that command is spelled `asset open` in CLI mode but
  `asset_open` in native mode). `asset.info` was audited and left alone — its payload
  is filtered through the `_AGENT_VISIBLE_FIELDS` allowlist, which carries no
  `caption`/`ocr_text`/`description`, so its description is already truthful.
  Verified: targeted suite 21 tests + 10 subtests, full pipeline
  regression **602 passed / 21 subtests**, backend + facade suite **90 passed**,
  mutation-tested (each of 2 seeded mutations killed exactly its one target test), and
  the report rendered over all **118** generated scenarios with zero false warnings
  (the 29 `not_compiled` warnings match the 29 uncompiled rounds exactly).
- **M5 Main-Supervised expansion (1 seed, 2 archetype backfills, coverage layer).**
  Added the `access_request_approval` seed (#2, "reconcile approval scope before
  granting"), which rides the existing M4 `approval` family and `scope_reconciliation`
  plugin — no new skill, plugin, MCP tool, effect type, or facade golden change.
  Backfilled the `archetype` tag on the two reused Main-Supervised/Timeout seeds
  that were missing it: `incident_commander_selection` (#1 → `evidence_mismatch`)
  and `sla_support_response` (#7 → `time_window_expiration`), leaving their
  `conflict_taxonomy` untouched. Added a coverage layer to `conflict_vlm_spec.py` —
  the authoritative `ARCHETYPE_CATEGORY` map (15 archetype tokens over the five
  `CONFLICT_CATEGORIES` from `task-list.md`), a `category_coverage(seeds)` helper,
  and `tests/test_archetype_coverage.py` that enforces 15/15
  archetype coverage and 5/5 category coverage and catches orphaned/typo'd
  archetype tokens. `conflict_vlm_v1` now holds `20` seeds with **all 15 archetypes
  present and every top-level category covered** (main_supervised 2,
  subagent_collaboration 4, timeout 3, order 2, inconsistency 4; plus 5 original
  pre-taxonomy seeds). Offline gates green: token pre-flight clean, mock smoke
  materializes the new seed (10/10 F1 gold rounds clean), the coverage suite passes
  (7/7), and the full unit suite is green apart from 4 pre-existing unrelated
  failures. Credential-bound Vertex generation + eval remain to be run.
- **M4 SubAgent-Collaboration expansion (`approval` + `containment` + `review`
  families, 5 plugins, 5 seeds).** Added three rule-enforcing stateful skills —
  `approval` (`request|approve|grant|revoke|provisional|escalate|status`, effects
  `approval_granted`/`approval_escalated`; `grant` requires every approval valid and
  unexpired at the *same* virtual `_now`, and `provisional`/`escalate` break the #14
  deadlock), `containment`
  (`revoke-sessions|disable-forwarding|isolate-device|block-token|status`, effect
  `containment_action` from all four control points), and `review`
  (`assign|submit|sign|status`, effects `review_submitted`/`document_signed`; `sign`
  is gated until every aspect is `ok` with no outstanding blocker) — plus the
  `scope_reconciliation`, `multi_party_synchronization`, `coordinated_containment`,
  `non_overlapping_assignment`, and `duplicate_hold_stale` plugins and five seeds
  (`multi_party_access_permit`, `deadlock_multi_approval`, `coordinated_containment`,
  `collaborative_contract_review`, `multi_tab_slot_monitoring`). The full 15-step
  glue landed (re-exports, backend registration, `agent_tools.v1` families, MCP
  facade `_CURATED_TOOL_COUNT=88→104` + both goldens refreshed — `mcp_tools_golden`
  104 tools, `mcp_dispatch_golden` 155 cases, `ALLOWED_EFFECT_TYPES`, scaffolder
  verb-map incl. the containment 4-verb→1-effect mapping, materializer branches
  seeding empty `[]`). This completes all `7/7` new skill families and `10/10` new
  plugins, bringing `conflict_vlm_v1` to `19` seeds. The targeted MCP/skill/plugin
  cluster (168 tests) and MCP facade parity suite (6 tests) pass, and the mock smoke
  materializes all five seeds (5/5, 50/50 F1 gold-check rounds clean). Remaining
  exit-gate steps are the credential-bound Vertex generation + replay audit + eval.
- **M3 Timeout-class expansion (`booking` + `pairing` families, virtual-time
  plumbing, 2 plugins, 3 seeds).** Added two rule-enforcing stateful skills —
  `booking` (`search|hold|confirm|release|extend|status`, effects
  `booking_held`/`booking_confirmed`/`booking_released`) and `pairing`
  (`start|show-code|enter-code|confirm|status`, effect `pairing_completed`) — plus
  the `reversible_action_hold` and `multi_device_pairing` plugins and three seeds
  (`sla_support_response`, `expiring_resource_hold`, `multi_device_pairing`). The
  cross-cutting change is real virtual-time plumbing: the environment publishes its
  `_virtual_time` onto stateful skills (`BaseBackend.set_virtual_time` →
  `SkillRuntime.set_now` → `skill._now`) at both dispatch seams, so holds/codes
  expire on the clock (only `wait` advances it, keeping ground truth reproducible).
  The full 15-step glue landed (re-exports, backend registration, `agent_tools.v1`
  families, MCP facade `_CURATED_TOOL_COUNT=88` + refreshed goldens,
  `ALLOWED_EFFECT_TYPES`, scaffolder verb-map, materializer branches seeding empty
  `[]`). All `14` seeds resolve every `conflict_taxonomy` token to a registered
  plugin; `426` targeted unit/spec/facade tests pass and the mock smoke materializes
  all three seeds (3/3, inline F1 gold-check passes). Remaining exit-gate steps are
  the credential-bound Vertex generation + replay audit + eval.
- **Conflict-plugin standardization (`ConflictPluginSpec` + decorator registry).**
  The conflict-plugin surface is now standardized the same way the skill/MCP surface
  is: every plugin declares a declarative `ConflictPluginSpec`
  (`clasharness/scenario/plugins/spec.py`) and self-registers via
  `@register_conflict_plugin` (`plugins/registry.py`); `plugins/__init__.py` builds
  `_ALL_PLUGINS` from that registry and exposes `validate_plugins()` — unique names,
  no taxonomy-token collisions, valid specs, and a cross-surface check that a
  plugin's `emits_effects` ⊆ the `state_effects` its declared `skill_families` can
  emit (the plugin analogue of `CommandSpec.emits_effects`). All `15` registered
  plugins carry a spec; conformance is enforced by
  `tests/test_conflict_plugin_spec.py`. This is metadata *around* the five prompt
  hooks, not a replacement for them. Authoring guide:
  [conflict-plugin-guide.md](conflict-plugin-guide.md).
- **M2 Order-class expansion (`controlled_deployment_gate` + 2 seeds).** Added the
  `controlled_deployment_gate` plugin (approvals-as-gates governance ordering) and
  two Order-class seeds — `employee_onboarding_workflow` (reuses
  `dependency_ordering`, drives `deploy execute` → `deployment_executed`) and
  `it_deployment_gate` (drives `deploy cutover` → `deployment_cutover`). Both ride
  the existing M1 `deploy` family, so no skill or pipeline changes were
  needed. All `11` seeds resolve every `conflict_taxonomy` token to a registered
  plugin.
- **natural_image decisive-evidence upgrade.** Two `DOMAIN-AWARENESS` passes in
  `conflict_llm_pipeline.py` steer seeds toward using photographs as the decisive
  visual evidence where the domain warrants it. The `warehouse` and `satellite`
  seeds now carry `natural_image` / `photograph`, and `seed-design-guide.md`
  documents the "Automatic selection" rule.
- **Taxonomy token fix (`db_audit` → `credential_lock`).**
  `multimodal_temporal_conflict_01.json` listed `db_audit` in `conflict_taxonomy`,
  but `get_plugins_for_taxonomy` matches on each plugin's `name`, and
  `DBAuditPlugin.name` is `credential_lock`. The mismatch silently dropped the
  plugin. All `9` seeds now resolve every `conflict_taxonomy` token to a registered
  plugin.

### Key files

- `clasharness/data/scenario_seeds/conflict_vlm_v1/` — seeds (fixtures)
- `clasharness/skills/stateful.py` — `StatefulProtocolSkill` base (M0)
- `clasharness/skills/impl/lock_skill.py`, `deploy_skill.py` — M1 skills
- `clasharness/skills/impl/booking_skill.py`, `pairing_skill.py` — M3 skills (virtual-time expiry)
- `clasharness/scenario/conflict_vlm_spec.py` — conflict seed / scenario dataclasses; `ARCHETYPE_CATEGORY` map + `archetype_coverage`/`category_coverage` helpers
- `tests/test_archetype_coverage.py` — enforces 15/15 archetype + 5/5 category coverage (M5)
- `clasharness/scenario/conflict_llm_pipeline.py` — generation pipeline
- `clasharness/scenario/conflict_vlm_materializer.py` — scenario → runtime state
- `clasharness/scenario/conflict_strict_validator.py` — strict validation
- `clasharness/scenario/plugins/` — conflict taxonomy plugins
- `clasharness/scenario/plugins/spec.py` — `ConflictPluginSpec` (declarative plugin spec)
- `clasharness/scenario/plugins/registry.py` — `@register_conflict_plugin` decorator registry
- [conflict-plugin-guide.md](conflict-plugin-guide.md) — conflict-plugin authoring guide
- `scripts/generate_conflict_vlm_scenarios.py` — conflict generation entrypoint
- [../Multi-Agent_GENERATION_PIPELINE.md](../Multi-Agent_GENERATION_PIPELINE.md) — pipeline reference
- [conflict-expansion-plan.md](conflict-expansion-plan.md) — forward roadmap
- [seed-design-guide.md](seed-design-guide.md) — seed authoring standard

## Verify

Tests run through pytest:

```bash
# Conflict-VLM line
pytest tests/test_conflict_pipeline.py tests/test_conflict_vlm_reconcile.py \
  tests/test_state_isolation_fixes.py tests/test_stateful_skill.py \
  tests/test_lock_preload.py tests/test_deploy_preload.py \
  tests/test_protocol_effect_scaffold.py tests/test_booking_preload.py \
  tests/test_pairing_preload.py tests/test_virtual_clock.py tests/test_mcp_facade.py

# Generate the conflict-VLM dataset (offline, deterministic)
python scripts/generate_conflict_vlm_scenarios.py \
  --provider mock --image-provider mock --chart-provider template --target-rounds 12
```

Then confirm all `20` `conflict_vlm_v1` seeds resolve every `conflict_taxonomy`
token to a registered plugin.
