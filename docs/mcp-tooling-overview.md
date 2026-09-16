# MCP / Tooling Overview for Collaborators

This repo exposes agent tools two ways, both backed by the same environment:

1. An **internal MCP-like command layer** — agents emit plain-text commands, the
   runtime routes them to registered skills/backends, and evaluators check the
   resulting state. This is what the train/eval pipeline uses.
2. A **runnable, zero-dependency JSON-RPC MCP server**
   (`clasharness/backend/mcp_server.py`) that speaks the Model Context Protocol
   over stdio, backed by a curated tool facade
   (`clasharness/backend/mcp_facade.py`). See [The MCP server](#the-mcp-server).

There is no single "source of truth" for the whole tool surface — there are
**three coordinated surfaces**, each authoritative for a different consumer:

| Surface | Where | Authoritative for |
| --- | --- | --- |
| **AgentToolSpec v1** | `clasharness/specs/agent_tools.v1.json` + `clasharness/tools/agent_tool_spec.py` | agent prompt catalog, safety allowlist, generated-GT / scenario command validation |
| **SkillSpec + registry** | `clasharness/skills/spec.py` + each skill's `SPEC` | runtime routing registration, conformance, the extension layer (see [mcp-extension-guide.md](mcp-extension-guide.md)) |
| **MCP facade + server** | `clasharness/backend/mcp_facade.py` + `clasharness/backend/mcp_server.py` | the live MCP `tools/list` + `tools/call` surface |

## Current Design In One Sentence

```text
LLM path:  agent catalog (AgentToolSpec)   -> text command ----\
MCP path:  tools/call (mcp_facade)         -> workspace CLI ---/-> safety guard
           -> SkillRuntime -> SkillRegistry -> skill/backend -> environment state -> evaluator checks
```

`AgentToolSpec` defines and validates the command catalog the LLM agent sees; the
MCP server exposes a curated subset as MCP tools; both funnel into the same
`SkillRuntime` / `SkillRegistry` stack.

## What Exists Today

The AgentToolSpec definition lives at:

```text
clasharness/specs/agent_tools.v1.json
```

The loader and validators live at:

```text
clasharness/tools/agent_tool_spec.py
```

Current command families (23):

| Family | Purpose | Runtime surface | GT allowed |
| --- | --- | --- | --- |
| `file` | Virtual workspace evidence and report writing | `FileSkillAdapter` over `FileSystemBackend` | Yes |
| `asset` | Multimodal assets: listing, identity/structure metadata, and direct-open references | `AssetSkillAdapter` over `AssetBackend` | Yes |
| `tasks` | Task-board style workflows | `TasksSkill` | Yes |
| `email` | Email read/write/search workflows | `EmailSkill` | Yes |
| `calendar` | Calendar reads and event edits | `CalendarSkill` | Yes |
| `weather` | Weather reads and alerts | `WeatherSkill` | Yes |
| `system` | Status, gateway health, config, crons, devices, time/wait, undo | `SystemSkill` | Yes |
| `agent` | Multi-agent delegation, status, identity | `AgentSkill` | Yes |
| `message` | Chat messaging, broadcast, reacts, polls | `MessageSkill` | Yes |
| `plugin` | Install, list, enable/disable plugins and skills | `SystemSkill` (`plugin` prefix) | Yes |
| `channel` | Channel logins, lists, status checks | `ChannelSkill` | Yes |
| `lock` | Advisory locks over named resources (M1 coordination) | `LockSkill` | Yes |
| `deploy` | Deployment protocol with a readiness-gate barrier (M1) | `DeploySkill` | Yes |
| `approval` | Multi-party approval with scope reconciliation and deadlock breakers (M2) | `ApprovalSkill` | Yes |
| `booking` | Reversible, time-limited resource holds (M2) | `BookingSkill` | Yes |
| `pairing` | Ordered, time-bounded two-device pairing handshake (M3) | `PairingSkill` | Yes |
| `containment` | Multi-control-point incident containment (M4) | `ContainmentSkill` | Yes |
| `review` | Multi-aspect collaborative review and sign-off (M4) | `ReviewSkill` | Yes |
| `curl` | HTTP fetch backend | `HttpSkillAdapter` over `HttpBackend` | No |
| `shell_inspection` | Safety-guarded host inspection commands | Safety guard; `git`/`jq`/`grep` route to `GitSkill`/`JqSkill`/`GrepSkill` | No |
| `shell_scripting` | Run a python or bash script | `ShellExecutionSkill` (`run_script`) | No |
| `shell_archiving` | Extract an archive | `ShellExecutionSkill` (`decompress_archive`) | No |
| `shell_database` | Query a database file | `ShellExecutionSkill` (`execute_db_query`) | No |

`GT allowed` means generated `ground_truth.solution_commands` may use the command when `safe_for_gt=true`. The `curl`, `shell_inspection`, `shell_scripting`, `shell_archiving`, and `shell_database` families are runtime/safety surfaces flagged `safe_for_gt=false`, so generators do not emit them as benchmark ground truth.

---

## Detailed Command Breakdown

Below is the list of all command families, their specific subcommands, syntax, and effects:

### 1. `calendar` Family (Domain: calendar)
* **`calendar list`**
  * **Syntax**: `calendar list [--from DATE] [--to DATE]`
  * **Description**: List calendar events.
* **`calendar add-event`**
  * **Syntax**: `calendar add-event --title TITLE --start DATETIME [--location LOC] [--attendees A,B]`
  * **Description**: Create a calendar event.
  * **State Effects**: `calendar_events_created`
* **`calendar update-event`**
  * **Syntax**: `calendar update-event --id ID [--title TITLE] [--start DATETIME] [--location LOC]`
  * **Description**: Update a calendar event.
  * **State Effects**: `calendar_events_updated`
* **`calendar delete-event`**
  * **Syntax**: `calendar delete-event --id ID`
  * **Description**: Delete a calendar event.
  * **State Effects**: `calendar_events_deleted`
* **`calendar today`**
  * **Syntax**: `calendar today --timezone TIMEZONE`
  * **Description**: Return benchmark-local date for a timezone.

### 2. `email` Family (Domain: email)
* **`email list`**
  * **Syntax**: `email list [--folder FOLDER]`
  * **Description**: List emails.
* **`email read`**
  * **Syntax**: `email read --id ID`
  * **Description**: Read an email.
* **`email send`**
  * **Syntax**: `email send --to EMAIL --subject SUBJECT --body BODY`
  * **Description**: Send an email.
  * **State Effects**: `emails_sent`
* **`email reply`**
  * **Syntax**: `email reply --id ID --body BODY`
  * **Description**: Reply to an email.
  * **State Effects**: `emails_sent`
* **`email move`**
  * **Syntax**: `email move --id ID --folder FOLDER`
  * **Description**: Move an email to a folder.
  * **State Effects**: `emails_moved`
* **`email mark`**
  * **Syntax**: `email mark --id ID --flag read|unread`
  * **Description**: Mark an email read or unread.
* **`email search`**
  * **Syntax**: `email search --query QUERY`
  * **Description**: Search emails.

### 3. `weather` Family (Domain: weather)
* **`weather get`**
  * **Syntax**: `weather get --location LOCATION`
  * **Description**: Get current weather.
* **`weather forecast`**
  * **Syntax**: `weather forecast --location LOCATION --days N`
  * **Description**: Get a weather forecast.
* **`weather alerts`**
  * **Syntax**: `weather alerts --location LOCATION`
  * **Description**: Get weather alerts.

### 4. `file` Family (Domain: file)
* **`file list`**
  * **Syntax**: `file list [--path PATH]`
  * **Description**: List virtual workspace files.
* **`file create`**
  * **Syntax**: `file create --path PATH --content CONTENT`
  * **Description**: Create or overwrite a virtual file.
  * **State Effects**: `files_created`
* **`file read`**
  * **Syntax**: `file read --path PATH`
  * **Description**: Read a virtual file.
* **`file move`**
  * **Syntax**: `file move --src SRC --dst DST`
  * **Description**: Move a virtual file.
* **`file append`**
  * **Syntax**: `file append --path PATH --content CONTENT`
  * **Description**: Append to a virtual file.
* **`file delete`**
  * **Syntax**: `file delete --path PATH`
  * **Description**: Delete a virtual file.
  * **State Effects**: `files_deleted`

### 5. `asset` Family (Domain: asset, multimodal)
* **`asset list`**
  * **Syntax**: `asset list [--kind KIND]`
  * **Description**: List visible multimodal assets.
* **`asset info`**
  * **Syntax**: `asset info --id ID`
  * **Description**: Show metadata for a visible asset.
  * **State Effects**: `assets_accessed`
* **`asset read-text`**
  * **Syntax**: `asset read-text --id ID`
  * **Description**: Check for text attached to a visible asset. Assets here carry no OCR or alt-text, so this always reports none and points at `asset open`.
  * **State Effects**: `assets_accessed`
* **`asset describe`**
  * **Syntax**: `asset describe --id ID`
  * **Description**: Show a visible asset's identity and structure (kind, path, MIME, dimensions). Contents are visual and are **not** described here.
  * **State Effects**: `assets_accessed`
* **`asset open`**
  * **Syntax**: `asset open --id ID`
  * **Description**: Open an asset reference for multimodal providers without printing binary content.
  * **State Effects**: `assets_opened`

> **No text side-channel on assets.** `read-text` and `describe` deliberately surface
> **no** answer-bearing text. `ocr_text` / `caption` / `description` / `content_text`
> are authoring-side fields: the manifest is emitted with
> `include_generation_metadata=False` and `AssetBackend` serves only an allowlist
> (`_AGENT_VISIBLE_FIELDS`), so the only way to read a value off an asset is
> `asset open` and *looking* at it. `read-text` returns an explicit stub rather than an
> empty string so the agent can tell "nothing attached" from "the call failed".
> The generator relies on this: see the vision-exclusivity safeguards (§D) in
> [`Multi-Agent_GENERATION_PIPELINE.md`](../Multi-Agent_GENERATION_PIPELINE.md).

### 6. `tasks` Family (Domain: tasks)
* **`tasks list`**
  * **Syntax**: `tasks list [--status STATUS] [--priority PRIORITY]`
  * **Description**: List task-board entries.
* **`tasks add`**
  * **Syntax**: `tasks add --title TITLE [--due DATE] [--priority PRIORITY]`
  * **Description**: Add a task-board entry.
  * **State Effects**: `tasks_created`
* **`tasks complete`**
  * **Syntax**: `tasks complete --id ID | --title TITLE`
  * **Description**: Complete a task-board entry by id or title.
  * **State Effects**: `tasks_completed`
* **`tasks search`**
  * **Syntax**: `tasks search --query QUERY`
  * **Description**: Search task-board entries.

### 7. `system` Family (Domain: monitoring, setup_config, cron_webhook, security, device_node)
* **`system status`**
  * **Syntax**: `system status [--json]`
  * **Description**: Show system status.
* **`system health`**
  * **Syntax**: `system health`
  * **Description**: Check gateway health.
* **`system doctor`**
  * **Syntax**: `system doctor`
  * **Description**: Run diagnostics.
* **`system gateway start`**
  * **Syntax**: `system gateway start`
  * **Description**: Start the gateway.
* **`system models set`**
  * **Syntax**: `system models set MODEL`
  * **Description**: Set default model.
* **`system config get`**
  * **Syntax**: `system config get PATH`
  * **Description**: Read config value (e.g. `agent.model`).
* **`system config set`**
  * **Syntax**: `system config set PATH VALUE`
  * **Description**: Set config value.
  * **State Effects**: `config_changes`
* **`system cron add`**
  * **Syntax**: `system cron add --name NAME --cron SCHEDULE --message TEXT`
  * **Description**: Add a cron job.
  * **State Effects**: `cron_jobs_created`
* **`system cron list`**
  * **Syntax**: `system cron list`
  * **Description**: List cron jobs.
* **`system security audit`**
  * **Syntax**: `system security audit`
  * **Description**: Run security audit.
* **`system setup`**
  * **Syntax**: `system setup`
  * **Description**: Run system setup.
* **`system devices pair`**
  * **Syntax**: `system devices pair`
  * **Description**: Pair a device.
* **`system devices list`**
  * **Syntax**: `system devices list`
  * **Description**: List devices.
* **`wait`**
  * **Syntax**: `wait SECONDS`
  * **Description**: Wait for a specified number of virtual seconds to allow asynchronous tasks to complete.
* **`undo`**
  * **Syntax**: `undo [N]`
  * **Description**: Undo the last N steps (reverts file system and memory).

### 8. `agent` Family (Domain: agent_mgmt)
* **`agent add`**
  * **Syntax**: `agent add NAME [--model MODEL]`
  * **Description**: Create an agent.
  * **State Effects**: `agents_created`
* **`agent list`**
  * **Syntax**: `agent list`
  * **Description**: List agents.
* **`agent set-identity`**
  * **Syntax**: `agent set-identity NAME --role ROLE`
  * **Description**: Set agent identity.
  * **State Effects**: `agents_created`
* **`agent delegate`**
  * **Syntax**: `agent delegate --agent NAME --instruction TEXT`
  * **Description**: Delegate an asynchronous task to a sub-agent.
  * **State Effects**: `agent_tasks_delegated`
* **`agent status`**
  * **Syntax**: `agent status [--agent NAME]`
  * **Description**: Check the status of pending asynchronous agent tasks.

### 9. `message` Family (Domain: messaging)
* **`message send`**
  * **Syntax**: `message send --channel NAME --target TARGET --message TEXT`
  * **Description**: Send a message.
  * **State Effects**: `messages_sent`
* **`message broadcast`**
  * **Syntax**: `message broadcast --channel NAME --message TEXT`
  * **Description**: Broadcast a message.
  * **State Effects**: `messages_sent`
* **`message search`**
  * **Syntax**: `message search --channel NAME --query QUERY`
  * **Description**: Search messages.
* **`message react`**
  * **Syntax**: `message react --channel NAME --message-id ID --emoji EMOJI`
  * **Description**: React to a message.
* **`message poll`**
  * **Syntax**: `message poll --channel NAME --question TEXT --options A,B`
  * **Description**: Create a message poll.
  * **State Effects**: `messages_sent`

### 10. `plugin` Family (Domain: plugin_skill)
* **`plugin enable`**
  * **Syntax**: `plugin enable NAME`
  * **Description**: Enable plugin.
  * **State Effects**: `plugins_installed`
* **`plugin disable`**
  * **Syntax**: `plugin disable NAME`
  * **Description**: Disable plugin.
  * **State Effects**: `plugins_installed`
* **`plugin list`**
  * **Syntax**: `plugin list`
  * **Description**: List plugins.
* **`plugin install`**
  * **Syntax**: `plugin install NAME`
  * **Description**: Install plugin.
  * **State Effects**: `plugins_installed`
* **`plugin skill info`**
  * **Syntax**: `plugin skill info NAME`
  * **Description**: Show skill info.

### 11. `channel` Family (Domain: channel_mgmt)
* **`channel login`**
  * **Syntax**: `channel login --channel NAME`
  * **Description**: Log in to a channel.
  * **State Effects**: `channels_configured`
* **`channel list`**
  * **Syntax**: `channel list [--json]`
  * **Description**: List channels.
* **`channel status`**
  * **Syntax**: `channel status`
  * **Description**: Show channel status.

### 12. `curl` Family (Domain: http)
* **`curl`**
  * **Syntax**: `curl URL`
  * **Description**: Fetch a URL through the HTTP backend.

### 13. `shell_inspection` Family (Domain: inspection)
Allows host inspection commands under the safety guard, including: `cat`, `ls`, `echo`, `grep`, `head`, `tail`, `wc`, `jq`, `pwd`, `whoami`, `env`, `printenv`, `git`. Not allowed for generated GT.

### 14. `shell_scripting` Family (Domain: scripting)
Agent verb `run_script` (underlying prefixes `python3`, `bash`). Not allowed for generated GT.
* **`run_script`**
  * **Syntax**: `run_script --path PATH [--args ARGS]`
  * **Description**: Execute a python or bash script.

### 15. `shell_archiving` Family (Domain: archiving)
Agent verb `decompress_archive` (underlying prefixes `unzip`, `tar`). Not allowed for generated GT.
* **`decompress_archive`**
  * **Syntax**: `decompress_archive --path PATH [--password PASSWORD] [--output-dir DIR]`
  * **Description**: Extract an archive file.

### 16. `shell_database` Family (Domain: database)
Agent verb `execute_db_query` (underlying prefix `sqlite3`). Not allowed for generated GT.
* **`execute_db_query`**
  * **Syntax**: `execute_db_query --db DB --query QUERY`
  * **Description**: Query a database file.

### 17. `lock` Family (Domain: coordination)
M1 coordination-protocol family, backed by `LockSkill` (a `StatefulProtocolSkill`).
* **`lock acquire`**
  * **Syntax**: `lock acquire --resource RESOURCE [--section SECTION] [--holder HOLDER]`
  * **Description**: Acquire an advisory lock over a named resource; fails if held by a different holder.
  * **State Effects**: `lock_acquired`
* **`lock release`**
  * **Syntax**: `lock release --resource RESOURCE [--section SECTION]`
  * **Description**: Release an advisory lock over a named resource.
  * **State Effects**: `lock_released`
* **`lock list`**
  * **Syntax**: `lock list`
  * **Description**: List currently held locks.
* **`lock status`**
  * **Syntax**: `lock status --resource RESOURCE [--section SECTION]`
  * **Description**: Report whether a resource is free or held, and by whom.

### 18. `deploy` Family (Domain: deployment)
M1 barrier-synchronization family, backed by `DeploySkill`. `deploy cutover` is the barrier: it is blocked unless every readiness gate is `ready`.
* **`deploy request`**
  * **Syntax**: `deploy request --change CHANGE`
  * **Description**: Register a deployment request for a change.
* **`deploy approve`**
  * **Syntax**: `deploy approve --change CHANGE --approver APPROVER`
  * **Description**: Record an approval for a deployment change.
* **`deploy gate-set`**
  * **Syntax**: `deploy gate-set --gate GATE --state ready|pending|failed|stale|expired`
  * **Description**: Create or update a readiness gate state.
* **`deploy execute`**
  * **Syntax**: `deploy execute --change CHANGE`
  * **Description**: Execute a previously requested deployment.
  * **State Effects**: `deployment_executed`
* **`deploy cutover`**
  * **Syntax**: `deploy cutover --change CHANGE`
  * **Description**: Cut over a deployment; blocked unless every readiness gate is ready.
  * **State Effects**: `deployment_cutover`
* **`deploy rollback`**
  * **Syntax**: `deploy rollback --change CHANGE`
  * **Description**: Roll back a deployment change.
  * **State Effects**: `deployment_rolledback`
* **`deploy monitor`**
  * **Syntax**: `deploy monitor --change CHANGE`
  * **Description**: Report the current status of a deployment change.
* **`deploy status`**
  * **Syntax**: `deploy status [--change CHANGE]`
  * **Description**: List readiness gates and deployment statuses.

### 19. `approval` Family (Domain: coordination)
M2 multi-party approval family, backed by `ApprovalSkill`. `approval grant` is the barrier: it succeeds only when every required approval is valid, unexpired (per the virtual clock), and scope-reconciled; `provisional` / `escalate` are the deadlock breakers.
* **`approval request`**
  * **Syntax**: `approval request --resource RESOURCE [--scope SCOPE] [--approvers ROLES]`
  * **Description**: Open an access request naming the required approvers and requested scope.
* **`approval approve`**
  * **Syntax**: `approval approve --resource RESOURCE --approver ROLE [--scope SCOPE] [--ttl SECONDS]`
  * **Description**: Record one approver's approval, optionally scoped and time-limited.
* **`approval grant`**
  * **Syntax**: `approval grant --resource RESOURCE`
  * **Description**: Grant access once every required approval is valid, unexpired, and scope-reconciled; fails otherwise.
  * **State Effects**: `approval_granted`
* **`approval revoke`**
  * **Syntax**: `approval revoke --resource RESOURCE [--approver ROLE]`
  * **Description**: Revoke a resource's request, or one approver's approval.
* **`approval provisional`**
  * **Syntax**: `approval provisional --resource RESOURCE --approver ROLE [--scope SCOPE] [--ttl SECONDS]`
  * **Description**: Grant a provisional (deadlock-breaking) approval for an approver.
* **`approval escalate`**
  * **Syntax**: `approval escalate --resource RESOURCE --to ROLE`
  * **Description**: Escalate a stuck approval to a higher authority to break a circular dependency.
  * **State Effects**: `approval_escalated`
* **`approval status`**
  * **Syntax**: `approval status [--resource RESOURCE]`
  * **Description**: Show requests and approvals (read-only).

### 20. `booking` Family (Domain: coordination)
M2 reversible-hold family, backed by `BookingSkill`. A `hold` is time-limited and expires against the virtual clock; `confirm` is the barrier that must land before the window closes.
* **`booking search`**
  * **Syntax**: `booking search [--resource RESOURCE] [--query QUERY]`
  * **Description**: List resource holds and their availability (read-only).
* **`booking hold`**
  * **Syntax**: `booking hold --resource RESOURCE [--holder HOLDER] [--ttl SECONDS]`
  * **Description**: Place a reversible, time-limited hold on a resource.
  * **State Effects**: `booking_held`
* **`booking confirm`**
  * **Syntax**: `booking confirm --resource RESOURCE [--holder HOLDER]`
  * **Description**: Confirm a hold before its time window closes; fails if the hold has expired.
  * **State Effects**: `booking_confirmed`
* **`booking release`**
  * **Syntax**: `booking release --resource RESOURCE`
  * **Description**: Release (reverse) a hold on a resource.
  * **State Effects**: `booking_released`
* **`booking extend`**
  * **Syntax**: `booking extend --resource RESOURCE [--ttl SECONDS]`
  * **Description**: Refresh a hold's time window before it expires.
* **`booking status`**
  * **Syntax**: `booking status [--resource RESOURCE]`
  * **Description**: Show the status of resource holds (read-only).

### 21. `pairing` Family (Domain: coordination)
M3 two-device pairing family, backed by `PairingSkill`. The handshake is order-sensitive and time-bounded: the code from `show-code` must be entered via `enter-code` and then `confirm`-ed before the session expires.
* **`pairing start`**
  * **Syntax**: `pairing start --device DEVICE [--ttl SECONDS]`
  * **Description**: Open a time-bounded pairing session for a device.
* **`pairing show-code`**
  * **Syntax**: `pairing show-code --device DEVICE [--code CODE]`
  * **Description**: Reveal the pairing code for an open session; fails if the session has expired.
* **`pairing enter-code`**
  * **Syntax**: `pairing enter-code --device DEVICE [--code CODE]`
  * **Description**: Enter the pairing code from the peer device; fails if out of order or expired.
* **`pairing confirm`**
  * **Syntax**: `pairing confirm --device DEVICE`
  * **Description**: Complete the pairing once the code is entered; fails if out of order or expired.
  * **State Effects**: `pairing_completed`
* **`pairing status`**
  * **Syntax**: `pairing status [--device DEVICE]`
  * **Description**: Show the status of pairing sessions (read-only).

### 22. `containment` Family (Domain: coordination)
M4 incident-containment family, backed by `ContainmentSkill`. Containment requires acting on all four control points; each action emits the same `containment_action` effect.
* **`containment revoke-sessions`**
  * **Syntax**: `containment revoke-sessions --account ACCOUNT`
  * **Description**: Revoke all active sessions for the compromised account (one of four control points).
  * **State Effects**: `containment_action`
* **`containment disable-forwarding`**
  * **Syntax**: `containment disable-forwarding --account ACCOUNT`
  * **Description**: Disable mail/message forwarding rules on the account (one of four control points).
  * **State Effects**: `containment_action`
* **`containment isolate-device`**
  * **Syntax**: `containment isolate-device --device DEVICE`
  * **Description**: Isolate the compromised device from the network (one of four control points).
  * **State Effects**: `containment_action`
* **`containment block-token`**
  * **Syntax**: `containment block-token --token TOKEN`
  * **Description**: Revoke and block the leaked access token (one of four control points).
  * **State Effects**: `containment_action`
* **`containment status`**
  * **Syntax**: `containment status`
  * **Description**: Show which control points are contained (read-only).

### 23. `review` Family (Domain: coordination)
M4 collaborative sign-off family, backed by `ReviewSkill`. `review sign` is the barrier: it is blocked unless every assigned aspect has been submitted `ok` with no outstanding blockers.
* **`review assign`**
  * **Syntax**: `review assign --aspect ASPECT --reviewer ROLE`
  * **Description**: Assign a review aspect (legal, privacy, security, ...) to a reviewer.
* **`review submit`**
  * **Syntax**: `review submit --aspect ASPECT --status ok|blocked [--issue TEXT]`
  * **Description**: Submit one aspect's verdict; a block carries its blocking issue.
  * **State Effects**: `review_submitted`
* **`review sign`**
  * **Syntax**: `review sign --document DOCUMENT`
  * **Description**: Sign the document; blocked unless every aspect is ok with no outstanding blockers.
  * **State Effects**: `document_signed`
* **`review status`**
  * **Syntax**: `review status [--document DOCUMENT]`
  * **Description**: Show aspects, verdicts, and sign state (read-only).

---

## The MCP server

Besides the in-process command layer, the repo ships a **runnable MCP server** that
speaks the Model Context Protocol (JSON-RPC 2.0) over stdio, with **no third-party
MCP SDK dependency** — the protocol is implemented by hand.

- **Server:** `clasharness/backend/mcp_server.py` — class `MultiAgentArenaMCPServer`
  ("Zero-dependency JSON-RPC MCP server over StdIO."). It advertises
  `protocolVersion "2024-11-05"` and
  `serverInfo {"name": "multi-agent-arena-mcp", "version": "0.1.0"}`, and handles the
  JSON-RPC methods `initialize`, `notifications/initialized`, `tools/list`, and
  `tools/call`.
- **Tool surface:** `clasharness/backend/mcp_facade.py` — a declarative facade whose
  `TOOL_SCHEMAS` list is the curated, agent-facing `tools/list` payload (each entry has
  `name`, `description`, `inputSchema`). `tools/call` maps a tool name + arguments to a
  workspace CLI string via `mcp_facade.build_cli(...)`, which the server runs through
  `env.step(...)`. `mcp_facade.py` is now **hand-edited** and is the source of truth,
  guarded by the goldens in `tests/test_mcp_facade.py`.
  `scratch/gen_mcp_facade.py` was the one-time migration script that produced it from
  the old hand-written `mcp_server.MCP_TOOLS` blob; since `MCP_TOOLS` is now just
  `mcp_facade.build_tools_list()`, re-running it is an idempotent round-trip and not a
  way to regenerate the surface from anything upstream.

Launch it against a task:

```bash
python clasharness/backend/mcp_server.py --task-id <TASK_ID> --mode mock
# --mode is one of: mock, multi, real, hybrid
```

Two things worth knowing:

- The curated MCP `tools/list` surface (`mcp_facade.TOOL_SCHEMAS`) is intentionally
  **distinct from** the raw per-command `SkillSpec` surface and from the full
  `AgentToolSpec` catalog. Adding a command to a skill or to `agent_tools.v1.json`
  does **not** automatically add it to the MCP server — that is a separate, manual edit
  in `mcp_facade.py`, plus a matching refresh of the `mcp_tools_golden` /
  `mcp_dispatch_golden` fixtures.
- `SkillSpec.to_mcp_tools()` exists but is **not** what the server serves; it is used
  only by `SkillRegistry.validate_specs()` to enforce unique MCP tool names. The live
  server surface comes from `mcp_facade`.

---

## What AgentToolSpec Controls

`AgentToolSpec` is used by multiple pieces of the repo:

- LLM agent command catalog in `examples/train_and_eval.py`
- strict-mode command allowlist in `clasharness/utils/safety_guard.py`
- conflict scenario generation lint in `clasharness/scenario/conflict_llm_pipeline.py`
- conflict scenario materialization lint in `clasharness/scenario/conflict_vlm_materializer.py`
- collaborator tests in `tests/test_agent_tool_spec.py`

This avoids the old failure mode where a command was added to the prompt but not to the safety guard, or added to a generator but not implemented by the backend.

## Runtime Flow

1. The agent sees a command catalog derived from `AgentToolSpec`.
2. The agent emits a plain text command, for example:

```bash
file read --path /Evidence/release_log.md
```

3. `SafetyGuard` blocks dangerous patterns and checks the allowed prefixes in strict simulated modes.
4. `SkillRuntime` reads the first command token, such as `file`.
5. `SkillRegistry` resolves that prefix to a registered skill.
6. The skill delegates to the concrete backend.
7. The backend mutates or reads environment state.
8. The evaluator checks final state and intermediate evidence.

The MCP server (`tools/call`) enters this same flow at step 3: `mcp_facade.build_cli`
turns the MCP tool call into a workspace command string, which is then run through
`env.step(...)` and the same safety-guard / `SkillRuntime` / `SkillRegistry` path.

The command syntax remains text-first for compatibility with current tasks and paper experiments.

## Execution Modes

The same ToolSpec surface is exposed across modes, but the backend behind a family can differ.

| Mode | Meaning |
| --- | --- |
| `mock` | Simulated commands and fixed local state, no real external provider required |
| `multi` | Multi-app simulated backend across file/email/calendar/tasks/weather/openclaw/http surfaces |
| `martin` | Alias for `multi`; agent-in-sandbox CLI operation with the simulated/local skill stack |
| `real` | Real OpenClaw or provider-backed execution where configured |
| `hybrid` | Mixed real and simulated execution |

ToolSpec does not replace these modes. It only defines and validates the command surface.

## Conflict Scenario Logic

The `conflict_vlm_v1` line uses ToolSpec to keep generated scenarios controllable.
Each seed declares an `allowed_tool_families` list; the generator materializes
evidence across the enabled substrates and compiles GT commands that stay within
that list, so the coordinator's terminal action is expressed only in tools the
scenario intends.

Base conflict seeds use the multimodal reconciliation tool set — the `asset`
family for time-sensitive evidence, plus `agents`, `file`, and `message` for
subagent coordination and reporting:

```json
"allowed_tool_families": ["asset", "agents", "file", "message"]
```

Coordination-protocol seeds extend that base with the one protocol family the
archetype exercises, for example the distributed-lock seed:

```json
"allowed_tool_families": ["asset", "agents", "file", "message", "lock"]
```

All seven coordination families (`lock`, `deploy`, `approval`, `booking`,
`pairing`, `containment`, `review`) are `StatefulProtocolSkill`-backed protocols,
built with their respective taxonomy plugins (`distributed_lock`,
`barrier_synchronization`/`controlled_deployment_gate`, and the M2–M4 approval,
booking, pairing, containment, and review plugins). Taxonomy plugins in
`clasharness/scenario/plugins/` bind a seed's
`conflict_taxonomy` / `difficulty_challenges` tokens to the mechanic the LLM
builds around.

Like skills, conflict plugins are standardized declaratively: each plugin declares
a `ConflictPluginSpec` (`clasharness/scenario/plugins/spec.py`) and self-registers
via `@register_conflict_plugin` (`plugins/registry.py`), and `validate_plugins()`
enforces conformance — unique names, no taxonomy-token collisions, valid specs, and
a **cross-surface** check that a plugin's declared `emits_effects` are a subset of
the `state_effects` its declared `skill_families` can emit (the plugin analogue of
the skill-side `CommandSpec.emits_effects` check, linking a generation-time plugin
to the runtime skill family its trajectory exercises). To add a new mechanic, see
[conflict-plugin-guide.md](conflict-plugin-guide.md).

When a scenario is materialized, compiled `solution_commands` are validated
before task specs are written. If the LLM or generator emits an undefined or
disallowed command, generation fails early.

## How To Add A New Tool Family

Use this workflow for collaboration:

1. Add the family and commands to `clasharness/specs/agent_tools.v1.json`.
2. Decide whether each command should set `safe_for_gt=true`.
3. Implement the skill and attach a `SkillSpec` (`SPEC = SkillSpec(...)`). For the
   additive path that needs no backend wiring, see
   [mcp-extension-guide.md](mcp-extension-guide.md).
4. Register the skill in `MultiAppBackend` / `SkillRegistry` (or via the
   `@register_skill` extension layer).
5. If the command should be callable over the MCP server, add it to
   `clasharness/backend/mcp_facade.py` (regenerate via `scratch/gen_mcp_facade.py`).
6. Add command validation tests in `tests/test_agent_tool_spec.py`.
7. Add runtime dispatch tests if a new prefix is introduced.
8. If generators may use it, add generator tests that prove generated GT commands validate.
9. If conflict seeds may use it, add the family to `allowed_tool_families` and run mock conflict generation.
10. Update docs with a short example command and expected state effect.

Do not add commands only to the LLM prompt or only to a generator. The spec should be changed first.

## Command Spec Fields

Each command entry should include:

- `name`: stable id, for example `file.create`
- `tokens`: parse prefix, for example `["file", "create"]`
- `syntax`: human-readable command line shown to agents
- `required_flags`: flags that must appear in concrete commands
- `required_positionals`: minimum positional arguments after the prefix tokens
- `state_effects`: state mutations evaluators may observe
- `modes`: supported execution modes
- `safe_for_gt`: whether generated GT may use the command
- `description`: one-line explanation for collaborators

Keep command syntax stable after tasks are released. If a command must change, add a new command rather than silently changing old semantics.

## Current Limitations

- The MCP server is a hand-rolled JSON-RPC implementation over stdio, not built on an
  official MCP SDK, and it is exercised mainly by tests rather than a hosted deployment.
- The three tool surfaces (AgentToolSpec catalog, SkillSpec registry, and the
  `mcp_facade` MCP surface) are coordinated by convention and tests, not generated from
  one another, so a new tool can require edits in more than one place.
- Runtime routing is prefix-based, not capability-negotiated.
- Some real provider behavior depends on local credentials and setup.
- `curl`, `shell_inspection`, `shell_scripting`, `shell_archiving`, and `shell_database`
  are intentionally not allowed for generated GT (`safe_for_gt=false`).

## Recommended Checks Before Merging Tool Changes

```bash
pytest -q tests/test_agent_tool_spec.py
pytest -q tests/test_skill_runtime_dispatch.py
pytest -q tests/test_train_and_eval_cli.py
pytest -q tests/test_file_backend_workspace.py
pytest -q tests/test_conflict_pipeline.py
pytest -q tests/test_mcp_facade.py
pytest -q tests/test_mcp_server.py
```

For conflict-related tool changes, also run a mock generation of the
`conflict_vlm_v1` line and a short expert replay over the compiled tasks:

```bash
python scripts/generate_conflict_vlm_scenarios.py \
  --provider mock \
  --image-provider mock \
  --seed-dir clasharness/data/scenario_seeds/conflict_vlm_v1 \
  --output-dir /tmp/conflict_vlm_scenarios \
  --output-data-dir /tmp/conflict_vlm_tasks \
  --target-rounds 24

python examples/train_and_eval.py \
  --agent expert \
  --task-data-dir /tmp/conflict_vlm_tasks \
  --split total \
  --limit 3 \
  --mode multi
```

## Mental Model For Collaborators

`agent_tools.v1.json` is the contract for the **command catalog** (what the agent
sees, what the safety guard allows, what generated GT may use). A new MCP-like
capability is not considered integrated until it has:

- an AgentToolSpec entry
- a runtime route (a skill `SPEC` + registry registration)
- safety behavior
- GT validation behavior
- an `mcp_facade` entry, if it should be reachable over the MCP server
- tests
- docs

This keeps future MCP/tool expansion modular instead of spreading one feature across prompt text, backend code, generator logic, and evaluator assumptions.
