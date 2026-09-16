# Agent ToolSpec

`AgentToolSpec v1` is the internal source of truth for the command/MCP-like tool surface used by CLASHarness agents, generated ground truth, safety checks, and Conflict-VLM (`conflict_vlm_v1`) scenario generation.

Canonical spec file:

```text
clasharness/specs/agent_tools.v1.json
```

Loader and validator:

```text
clasharness/tools/agent_tool_spec.py
```

## What The Spec Defines

Each tool family defines:

- `family`: stable family id, such as `file`, `email`, or `lock`
- `domains`: task domains that should expose the family in the LLM command catalog
- `prefixes`: first command token routed by `SkillRuntime`
- `commands`: concrete commands agents may issue

Each command defines:

- `name`: stable command id, such as `file.create`
- `tokens`: command prefix tokens used for parsing, such as `["file", "create"]`
- `syntax`: human-facing catalog line shown to LLM agents
- `required_flags`: flags that must appear in concrete commands
- `required_positionals`: minimum positional args after `tokens`
- `state_effects`: evaluator-visible effects the command can produce
- `modes`: supported execution modes
- `safe_for_gt`: whether generated ground-truth solutions may use it

## Current Tool Families

The v1 spec covers 23 families. Eighteen are `safe_for_gt` (generated
ground-truth solutions may use them); the remaining five are host/shell/HTTP
surfaces that agents may issue but that generated GT must not depend on.

GT-eligible families (`safe_for_gt: true`):

- `file`: virtual workspace files; default text evidence substrate
- `asset`: multimodal assets for the Conflict-VLM line — listing, identity/structure metadata, and direct-open references. Deliberately exposes **no** OCR/alt-text side-channel: values are readable only by opening the asset (see `docs/mcp-tooling-overview.md`)
- `tasks`: task-board workflows
- `email`: local or provider-backed email interactions
- `calendar`: local or provider-backed calendar interactions (prefixes `calendar`, `gcalcli`)
- `weather`: local or provider-backed weather reads
- `system`: monitoring, setup/config, cron/webhook, security, and device-node ops (prefixes `system`, `wait`, `undo`)
- `agent`: multi-agent lifecycle management (add/remove/list agents)
- `message`: cross-channel message send/read
- `plugin`: plugin/skill enable/disable/config
- `channel`: channel login and management
- `lock`: distributed-lock coordination protocol (M1 stateful family)
- `deploy`: deployment request/approve/barrier protocol (M1 stateful family)
- `approval`: multi-party approval with scope reconciliation and deadlock breakers (M2 stateful family)
- `booking`: reversible, time-limited resource holds (M2 stateful family)
- `pairing`: ordered, time-bounded two-device pairing handshake (M3 stateful family)
- `containment`: multi-control-point incident containment (M4 stateful family)
- `review`: multi-aspect collaborative review and sign-off (M4 stateful family)

Host/shell/HTTP families, **not** allowed for generated GT (`safe_for_gt: false`):

- `curl`: HTTP backend surface
- `shell_inspection`: safety-allowed local inspection commands (`cat`, `ls`, `grep`, `jq`, `git`, …)
- `shell_scripting`: `python3` / `bash` execution
- `shell_archiving`: `unzip` / `tar`
- `shell_database`: `sqlite3`

## How To Add A New Tool Family

1. Add the family and commands to `clasharness/specs/agent_tools.v1.json`.
2. Implement or register the runtime surface through `SkillRuntime` / `SkillRegistry` / backend adapters.
3. Add command validation tests in `tests/test_agent_tool_spec.py`.
4. Add runtime dispatch tests if the family has a new prefix.
5. If generated tasks may use the command, set `safe_for_gt=true` and add generator tests.
6. If conflict-line scenarios may use the family, add it to the seed's `allowed_tool_families` and verify generated GT commands pass ToolSpec validation.

## Conflict-VLM Tool Surface

The Conflict-VLM line (`conflict_vlm_v1`, generated via
`scripts/generate_conflict_vlm_scenarios.py`) exercises two parts of this spec:
a multimodal evidence surface and the coordination-protocol skills the
coordinator uses to reconcile disagreeing subagents.

The `asset` family is the multimodal evidence surface. Agents inspect visible
assets without leaving the command catalog:

```text
asset list [--kind KIND]
asset info --id ID
asset read-text --id ID
asset describe --id ID
asset open --id ID
```

Coordination-protocol skills are implemented through `StatefulProtocolSkill`
(`clasharness/skills/stateful.py`). All seven stateful families are built today:

- `lock`: distributed-lock coordination (`distributed_lock` taxonomy plugin)
- `deploy`: request/approve/barrier coordination (`barrier_synchronization` /
  `controlled_deployment_gate` taxonomy plugins)
- `approval`: multi-party approval, scope reconciliation, and deadlock breakers
  (`multi_party_synchronization` / `scope_reconciliation` / `duplicate_hold_stale`)
- `booking`: reversible, time-limited holds (`reversible_action_hold` /
  `non_overlapping_assignment`)
- `pairing`: ordered two-device pairing handshake (`multi_device_pairing`)
- `containment`: multi-control-point incident containment (`coordinated_containment`)
- `review`: multi-aspect collaborative sign-off

A Conflict-VLM seed declares the families a scenario is allowed to use. A seed
that mixes multimodal evidence with coordination might declare:

```json
"allowed_tool_families": ["file", "email", "tasks", "calendar", "asset", "lock", "deploy"]
```

Seeds may omit this field and receive the default family set. A seed can narrow
or extend the list when a scenario requires a different evidence or coordination
surface.

The materializer validates compiled `solution_commands` before writing task
specs, so a generated scenario cannot silently use an undefined or disallowed
tool.

## Public Repo Hygiene

Do not commit local credentials, OAuth client secrets, provider tokens, local generated scratch data, or model checkpoints. The `.gitignore` includes common patterns, but reviewers should still inspect `git status --short` before publishing.
