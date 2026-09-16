# CLASHarness Integration Guide

Back to [README](../README.md) · See also [docs/README.md](README.md), [Multi-Agent generation pipeline](../Multi-Agent_GENERATION_PIPELINE.md), [evaluation.md](evaluation.md), and [project-structure.md](project-structure.md)

## Purpose

This document is a detailed introduction to **CLASHarness** for developers who did not build it originally but now need to:

- understand what CLASHarness is actually claiming
- place it correctly inside a larger repository or documentation set
- know which modules are core to the Conflict-VLM line
- know which parts are retained infrastructure and which parts are conflict-benchmark-specific
- write accurate umbrella-level documentation without diluting or mislabeling the benchmark

This is not a user quickstart and not a paper abstract. It is an integration-facing technical overview.

## Executive summary

CLASHarness is an **interactive execution benchmark for computer-use agents** operating over persistent software state. Its product focus is a single line: **Conflict-VLM (`conflict_vlm_v1`)**.

Its central evaluation question is:

> Can a coordinator agent reconcile disagreeing subagents and time-sensitive, multimodal evidence, then drive a workflow to the correct terminal action through observable state changes and side effects?

That is different from a static text benchmark, where a model only needs to emit one final answer. In CLASHarness:

- the environment is stateful
- the coordinator acts command by command
- subagents report information that may be stale, partial, or wrong
- backends and rendered assets mutate after each step
- evaluation is result-first and state-aware
- multiple trajectories can succeed if they produce the same evaluated outcome

The core of the Conflict-VLM line is **multi-agent conflict** (subagents whose diagnoses disagree) combined with **temporal conflict** (timeout and staleness). Scenarios are produced by an 8-stage hierarchical generation pipeline that compiles a single JSON seed into a multi-round, multimodal conflict scenario; the full design is in [Multi-Agent_GENERATION_PIPELINE.md](../Multi-Agent_GENERATION_PIPELINE.md).

> **Base codebase.** This repository grew out of an earlier single-agent CLI-benchmark codebase. That older machinery — the environment, task schema, execution backends, evaluator, and tool specs — is retained as the **infrastructure** the Conflict-VLM line runs on. It is no longer the product focus and is not the flagship.

## Installation and first-run path

Another development team usually needs more than a conceptual explanation. They also need the minimal operational path that proves CLASHarness is concrete and runnable.

### Editable installation

From the repository root:

```bash
pip install -e .
pip install -e ".[dev]"
```

This installs the environment plus development dependencies used by generation scripts and tests.

### Regenerating conflict scenarios

CLASHarness benchmark assets are largely generated rather than manually edited. The standard offline, deterministic regeneration command is:

```bash
python scripts/generate_conflict_vlm_scenarios.py \
  --provider mock \
  --image-provider mock \
  --chart-provider template \
  --target-rounds 12
```

This writes:

- scenario packages under `clasharness/data/scenarios/conflict_vlm_v1`
- compiled task data under `/tmp/clasharness-conflict-vlm-data` (override with `--output-data-dir`)

If another team is integrating CLASHarness into a broader repo, this command is the canonical way to rebuild scenario artifacts before validating packages, task counts, or coverage. Live generation uses a Gemini model and supports parallel generation/render workers and checkpoint/resume; see the "Running Generation & Verification" section of [the pipeline doc](../Multi-Agent_GENERATION_PIPELINE.md).

### First evaluation run

The main evaluation entrypoint is:

```bash
python examples/train_and_eval.py
```

A representative expert sanity check over the generated conflict tasks is:

```bash
python examples/train_and_eval.py \
  --agent expert \
  --task-data-dir /tmp/clasharness-conflict-vlm-data \
  --split total \
  --mode martin \
  --max-steps 20
```

The CLI default step budget is `15`, but benchmark-facing runs should usually set `--max-steps` explicitly. Multi-round conflict scenarios need enough budget to inspect evidence, act, wait, and verify, so an explicit larger budget is required. **The conflict evaluation protocol is `--max-steps 20`**, applied uniformly to every model and every run — the resolver deliberately ignores per-split and per-task overrides, so the budget cannot vary across a comparison.

### Hosted OpenAI-compatible example

One common hosted LLM setup is:

```bash
python examples/train_and_eval.py \
  --agent llm \
  --llm-provider openai \
  --llm-base-url https://api.example.com/v1 \
  --model Kimi-K2.5 \
  --task-data-dir /tmp/clasharness-conflict-vlm-data \
  --split total \
  --mode multi \
  --max-steps 20 \
  --llm-max-tokens 192 \
  -v
```

### Claude via LiteLLM proxy

The repo also supports a local LiteLLM proxy path for Claude-style models. In practice:

- run a local LiteLLM proxy
- point `--llm-provider openai` at that proxy
- use a `claude-*` model name

When `--llm-provider openai` is used with a `claude-*` model name, the client defaults to the local LiteLLM proxy URL and reads `LITELLM_PROXY_KEY` unless `--llm-base-url` or `--llm-api-key-env` is overridden. The detailed operational steps belong in [quickstart.md](quickstart.md) and [evaluation.md](evaluation.md), but another team documenting CLASHarness should know this path exists because it affects how conflict-line comparisons are actually run.

## OpenClaw and runtime prerequisites

CLASHarness can run under several execution modes, and setup expectations depend on the mode.

### Modes that do not require a full OpenClaw deployment

- `mock`
- `multi`
- `martin`

These are the easiest modes for benchmark development and reproducible evaluation. `multi` is the benchmark-default mode, and `martin` is a semantic alias for it.

### Modes that require real OpenClaw-backed setup

- `real`
- `hybrid`

These require a working OpenClaw installation and provider bootstrap. In practice that means:

- installing the `openclaw` CLI
- starting the gateway
- setting up Google-backed providers where Calendar, Gmail, or Tasks are involved
- exporting the required provider environment variables

Another team integrating CLASHarness should understand that OpenClaw setup is shared runtime infrastructure, while the conflict-line semantics remain harness-specific.

## Operational details that matter during integration

### History modes

The LLM rollout client currently supports three history policies:

- `full`
- `summary`
- `auto`

The current default is `full`.

This changes how much interaction history is sent back to the model during rollout. It does not change the task definition or evaluator contract, but it materially affects rollout behavior and should therefore be documented in evaluation-facing material.

### Useful output and reporting flags

When CLASHarness is run through `examples/train_and_eval.py`, the most useful additional outputs are:

- `--verbose-log PATH`
- `--save-report PATH`
- `--collect-data --out-dir PATH`
- `--record-step-eval`
- `--llm-request-retries N`
- `--llm-retry-backoff-s S`
- `--inter-task-sleep S`

These matter in integration contexts because another team will often want:

- raw rollout traces
- JSON reports for downstream aggregation
- JSONL trajectories for behavior analysis
- provider-aware accounting for unstable interactive endpoints

### Development checks

The most useful targeted checks for CLASHarness are:

```bash
pytest -q tests/test_conflict_pipeline.py tests/test_conflict_vlm_reconcile.py
pytest -q tests/test_conflict_mechanics.py
pytest -q tests/test_train_and_eval_cli.py tests/test_agent_tool_spec.py
```

If another team is validating an integration branch, these are the first checks they should run before attempting a larger benchmark comparison.

## One-paragraph description for external use

If another team needs a short, accurate description for an umbrella README or repo introduction, use wording close to this:

> CLASHarness is an interactive benchmark for multi-agent, computer-use coordination. Its core line, Conflict-VLM (`conflict_vlm_v1`), evaluates whether a coordinator agent can reconcile disagreeing subagents and time-sensitive, multimodal evidence, then drive a workflow to the correct terminal action through real state changes and observable side effects. The benchmark is generator-backed, result-first, and evaluated through normalized execution state rather than exact action-sequence matching.

That wording matches the actual CLASHarness contract and avoids importing claims that belong to other systems.

## What CLASHarness is and is not

### What CLASHarness is

CLASHarness is:

- an interactive benchmark centered on the Conflict-VLM line
- a stateful execution environment for evaluation
- a generator-backed conflict scenario suite
- a result-first evaluator over normalized state
- a benchmark for multi-agent + temporal conflict resolution

### What CLASHarness is not

CLASHarness is not:

- a static QA benchmark
- a one-shot text generation task
- a generic tool-use demo
- a training or continual-learning method
- a pure prompting framework
- an exact-trajectory imitation benchmark

The repository contains retained runtime infrastructure (environment, task schema, backends, evaluator, tool specs) that is reusable elsewhere, but the CLASHarness contribution itself is about **multi-agent conflict resolution under evolving, multimodal state**.

## The core CLASHarness design

CLASHarness is organized around three high-level design choices.

### 1. Interactive execution instead of final-answer scoring

In a CLASHarness rollout, the coordinator does not see one prompt and respond once. Instead:

1. the coordinator receives a task instruction
2. it inspects the current observation, subagent reports, and visible environment state (including rendered charts/images/rich-text assets)
3. it emits one command
4. the environment executes that command
5. the environment returns updated observation and state
6. the loop repeats until success or the step budget is exhausted

This means the benchmark can expose failures that static answer matching hides:

- trusting a stale subagent report over the latest on-disk or visual state
- acting on a text claim that a rendered chart or image contradicts
- declaring a task finished before a configured delay has elapsed
- partial completion without closure
- wrong-state replacement under conflicting evidence

### 2. Automated generation instead of one fixed authored list

The benchmark is not defined as one immutable table of handcrafted tasks. Conflict scenarios are produced by an 8-stage hierarchical generation pipeline that compiles a single JSON seed into a multi-round, multimodal scenario.

At a high level, generation combines:

- a scenario seed (personas, core conflict, target rounds)
- an outline-first phase plan with rolling-window expansion
- subagent personas, private knowledge, and scripted mock responses
- a background workspace, chat sessions, and visual evidence assets
- per-round instructions, environment updates, resolution anchors, and reference trajectories
- code-derived, result-first graders

The generated packages should be read as **pipeline output for the conflict line**, not as a single frozen CSV. The full stage-by-stage design, cost model, and failure-mode map are documented in [Multi-Agent_GENERATION_PIPELINE.md](../Multi-Agent_GENERATION_PIPELINE.md).

### 3. Functional evaluation over normalized state

CLASHarness does not require one exact privileged action sequence.

Instead, the environment builds a normalized evaluator-facing state and runs typed result-first checks over it. This is what allows:

- different valid command sequences to pass
- evaluation to focus on actual environment outcomes
- partial-credit scoring to coexist with exact pass/fail checks

The key idea is that the evaluator reasons over what the coordinator *caused*, not merely over what text it *said*.

## The CLASHarness runtime contract

If another team is integrating CLASHarness, they should understand the runtime contract, because this is where CLASHarness differs most sharply from text-only benchmarks.

### Inputs

At task start, the coordinator is given:

- one task instruction
- an initial environment state (workspace files, chat sessions, visible assets)
- subagents that respond with scripted, possibly conflicting information
- a set of command families / skills / backends available in the current mode

### Per-step behavior

At each step, the coordinator:

- observes the latest environment output, subagent messages, and any newly available assets
- emits one command
- triggers execution through the routed backend layer

The environment then:

- executes the command
- updates persistent state as needed
- records side effects
- returns a new observation

### Outputs recorded during the rollout

The rollout can produce several kinds of evidence:

- raw command outputs such as stdout/stderr
- backend state changes such as file/config/session updates
- explicit effect traces such as messages sent or delegations issued
- command history and step usage

These are later merged into the evaluator-facing normalized state.

## The CLASHarness evaluation contract

Another team integrating CLASHarness should not describe the evaluator loosely as "grading commands" or "checking answers." The evaluator is stricter and more structured than that.

### Normalized evaluator state

The environment materializes a normalized state, commonly denoted `S-hat`, that includes:

- task/instruction context
- bounded command history
- recent command outputs
- merged backend state
- explicit effect traces
- configuration state where relevant

This normalized state is what the evaluator actually checks.

### Checker types

Tasks can combine several checker types:

- `state`: validate final backend state values
- `effect`: validate observable side effects
- `config`: validate relevant configuration values
- `output`: validate command outputs when needed
- optional `llm`: validate open-ended semantic criteria

In practice, conflict rounds are primarily driven by `state`, `effect`, and `config` checks, with file-scoped semantic assertions used only where a decisive value is paraphrasable prose.

### Result-first semantics

The evaluator is result-first:

- it checks whether the right effects and state transitions happened
- it does not require one exact command trace
- multiple trajectories can pass if they satisfy the same evaluated outcome

This point should be preserved in any unified documentation. If another team rewrites CLASHarness as an exact-match benchmark, that rewrite is incorrect.

### Metrics

CLASHarness commonly reports both:

- **full-pass accuracy**
- **partial-credit average score**

These are not redundant metrics.

Full-pass accuracy requires all required checks to pass.

Average score is the dataset mean of the per-task weighted aggregate score computed from the same result-first checks. That means a task can receive substantial partial credit even when it fails to satisfy the full contract.

### Provider-aware accounting

When external LLM endpoints are used, interactive runs can be affected by retries, filtered outputs, compact fallbacks, or upstream failures. CLASHarness therefore keeps provider-aware accounting fields such as:

- `provider_failures`
- `provider_impacted_tasks`
- `provider_adjusted_accuracy`

These fields do not redefine the evaluator; they make interactive rollout results more interpretable.

## The Conflict-VLM benchmark line

The product focus of CLASHarness is a single line:

- `conflict_vlm_v1`

This line tests whether a coordinator agent can navigate situations where subagent reports disagree, where visual evidence contradicts a text claim, and where the correct action depends on waiting for time-sensitive state to settle.

### Key conflict dimensions

The conflict line probes three dimensions:

- **Temporal staleness** — a subagent reports success at `T=0`, but a late-arriving log or updated chart at a later round shows a regression. The coordinator is tested on whether it detects the report is *stale* and instead trusts the latest state of files or charts on disk.
- **Cross-modal discrepancy** — a subagent's text claim contradicts a rendered chart or image asset. The coordinator must cross-check text declarations against the multimodal visual evidence before taking a critical action.
- **Actionable delay** — a subagent reports that it is working on a change and will finish shortly. The coordinator must wait for the configured delay to elapse and verify the real file mutation on disk before declaring the task finished.

### Coordination-protocol families

The conflict mechanics are built on the `StatefulProtocolSkill` base (`clasharness/skills/stateful.py`). All seven coordination-protocol families are now implemented (`clasharness/skills/impl/`):

- `lock` (`lock_skill.py`) with its `distributed_lock` taxonomy plugin
- `deploy` (`deploy_skill.py`) with its `barrier_synchronization` / `controlled_deployment_gate` plugins
- `approval` (`approval_skill.py`) with `multi_party_synchronization` / `scope_reconciliation` / `duplicate_hold_stale`
- `booking` (`booking_skill.py`) with `reversible_action_hold` / `non_overlapping_assignment`
- `pairing` (`pairing_skill.py`) with `multi_device_pairing`
- `containment` (`containment_skill.py`) with `coordinated_containment`
- `review` (`review_skill.py`) with the collaborative sign-off plugins

See [conflict-expansion-plan.md](conflict-expansion-plan.md) for the coordination-protocol roadmap (milestones M0–M6): M0–M5 are built and only M6 (scale & baseline) remains.

The full mapping from conflict archetypes to seeds is catalogued in [task-list.md](task-list.md), and seed authoring standards are in [seed-design-guide.md](seed-design-guide.md).

## Current status and seed fixtures

The conflict line is under active development. The current on-disk state is:

- `20` conflict seed fixtures in `clasharness/data/scenario_seeds/conflict_vlm_v1/`, spanning all 15 archetypes across the 5 top-level categories

These are **pipeline fixtures**: they exercise and harden the generator and are **not** wired into a frozen `train`/`dev`/`test` release split. They should be introduced with wording like:

> The current conflict seeds are pipeline fixtures under active development ...

and not with wording that implies a permanently fixed benchmark split. Task packages are regenerated from these seeds with `scripts/generate_conflict_vlm_scenarios.py`; the number of rounds per scenario is controlled by `--target-rounds`.

## Execution modes

CLASHarness exposes five public mode names over four backend implementations:

- `mock`
- `multi`
- `martin`
- `real`
- `hybrid`

The exact semantics are documented in [execution-modes.md](execution-modes.md), but for integration purposes the most important summary is:

- `multi` is the benchmark-default mode
- `martin` is a semantic alias for `multi`; the model directly operates the CLI while the simulator/local skill stack responds
- execution modes change backend realization, not benchmark semantics
- the same conflict scenario can be evaluated under different backend realizations

If another team writes umbrella docs, execution modes usually belong in shared runtime documentation rather than in a conflict-line-only overview.

## History modes in LLM rollouts

The LLM rollout client currently supports three history policies:

- `full`
- `summary`
- `auto`

The current default is `full`.

This matters because it changes how much interaction history is provided to the LLM during rollout. It does **not** change the task definition or evaluator contract, but it can affect rollout behavior and should therefore be documented in evaluation-facing docs rather than in one-line benchmark summaries.

The canonical place for that is [evaluation.md](evaluation.md).

## Main code areas another team should understand

For integration or unified documentation work, these are the important code regions.

### 1. Repo-facing docs

- `README.md`
- `docs/README.md`
- `../Multi-Agent_GENERATION_PIPELINE.md` (main design doc)
- `docs/quickstart.md`
- `docs/evaluation.md`
- `docs/execution-modes.md`
- `docs/cli-runtime-setup.md`
- `docs/task-generation.md`
- `docs/conflict-expansion-plan.md`
- `docs/project-structure.md`

These are the primary materials another team should use when writing umbrella documentation.

### 2. Runtime and environment

- `examples/`
- `clasharness/backend/`
- `clasharness/core/`
- `clasharness/skills/`

This layer contains the shared runtime substrate: execution routing, backend implementations, skill families (including the stateful coordination-protocol skills), and environment abstractions. It is retained infrastructure that the conflict line runs on.

### 3. Conflict scenario generation

- `clasharness/scenario/conflict_vlm_spec.py`
- `clasharness/scenario/conflict_llm_pipeline.py`
- `clasharness/scenario/conflict_vlm_materializer.py`
- `clasharness/scenario/conflict_strict_validator.py`
- `clasharness/scenario/plugins/`
- `scripts/generate_conflict_vlm_scenarios.py`

This layer defines the generator-backed conflict line: the seed/scenario dataclasses, the 8-stage LLM pipeline, scenario-to-runtime materialization, strict validation, and the taxonomy plugins. If another team wants to understand how conflict scenarios are defined and compiled, this is the layer to inspect. The retained task-schema and generation infrastructure lives under `clasharness/tasks/`.

### 4. Scenario and task artifacts

- `clasharness/data/scenario_seeds/conflict_vlm_v1/` (20 seed fixtures)
- `clasharness/data/scenarios/conflict_vlm_v1/` (materialized scenario packages)
- `/tmp/clasharness-conflict-vlm-data` (compiled runnable task data, override with `--output-data-dir`)

This layer contains the seed inputs, the materialized scenario packages (spec, workspace, sessions, updates, and rendered assets), and the compiled task specs that evaluation consumes.

### 5. Evaluation

- `clasharness/evaluation/`
- `clasharness/utils/episode_memory.py`
- `examples/train_and_eval.py`

This layer defines result-first checks, rollout behavior, reporting, and LLM-facing history policies.

## How to document CLASHarness inside a larger repo

If another team is preparing one shared repository homepage, the cleanest approach is a three-layer documentation structure.

### Layer 1: umbrella README

The root README should explain:

- what the shared environment/repo is
- that the Conflict-VLM line is the product focus, running on retained infrastructure
- where to find conflict-line-specific documentation

It should not inline the full conflict taxonomy or all generation details.

### Layer 2: shared infrastructure docs

The following topics are usually better written once at the repo level:

- installation
- OpenClaw setup
- provider bootstrap
- execution modes
- common evaluation entrypoints
- project structure

These are shared runtime concerns, not unique conflict-line claims.

### Layer 3: conflict-line-specific docs

The following topics should remain clearly labeled as conflict-line-specific:

- the conflict-benchmark overview and conflict dimensions
- coordination-protocol families and the seed catalog
- current seed-fixture status
- the 8-stage generation pipeline and generator semantics
- normalized evaluator state and result-first checks

This is what keeps a unified repo readable and scientifically accurate.

## Common mistakes to avoid

These are the mistakes another development team is most likely to make when summarizing CLASHarness quickly.

### Mistake 1: describing CLASHarness as a static benchmark

Wrong idea:

- "CLASHarness is a set of fixed benchmark questions."

Correct idea:

- CLASHarness is a generator-backed conflict benchmark; scenarios are compiled from seeds by an 8-stage pipeline.

### Mistake 2: flattening CLASHarness into "tool use"

Wrong idea:

- "CLASHarness measures whether an agent can call tools."

Correct idea:

- CLASHarness measures whether a coordinator can reconcile disagreeing subagents and time-sensitive, multimodal evidence and drive the workflow to the correct terminal action.

### Mistake 3: describing evaluation as exact-match

Wrong idea:

- "The agent must match the reference trajectory."

Correct idea:

- CLASHarness is result-first and allows multiple valid trajectories to pass if they yield the same evaluated outcome.

### Mistake 4: presenting the seed fixtures as a frozen split

Wrong idea:

- "CLASHarness is a fixed set of 20 conflict tasks."

Correct idea:

- The 20 conflict seeds are pipeline fixtures under active development; scenarios and their round counts are regenerated from them, not a frozen `train`/`dev`/`test` split.

### Mistake 5: mixing shared setup with conflict-specific claims

Wrong idea:

- mixing provider setup, conflict taxonomy, and generation internals on one page

Correct idea:

- keep setup docs shared; keep conflict-line benchmark semantics and claims under conflict-line-specific docs

### Mistake 6: omitting the operational path

Wrong idea:

- assuming another team only needs high-level benchmark prose

Correct idea:

- include the editable install path, the conflict-scenario generation command, the evaluation entrypoint, execution-mode requirements, and the location of canonical CLASHarness docs

## Suggested wording blocks for external teams

### Very short form

- **CLASHarness**: interactive benchmark for multi-agent, computer-use coordination
- **Core contract**: command-by-command execution under persistent, multimodal state
- **Evaluation**: normalized state plus result-first checks
- **Generation**: 8-stage pipeline compiling a single seed into a multi-round conflict scenario
- **Core line**: `conflict_vlm_v1`

### Medium form

> CLASHarness is a benchmark for evaluating computer-use agents in stateful, multimodal execution environments. Its core line, Conflict-VLM (`conflict_vlm_v1`), asks a coordinator to reconcile subagents whose reports disagree and to cross-check time-sensitive chart/image/rich-text evidence before driving a workflow to the correct terminal action. Agents interact step by step, environment state evolves after each action, and success is determined by result-first checks over normalized state and observable side effects. Scenarios are generator-backed, compiled from seeds by an 8-stage pipeline rather than authored as one permanently fixed task list.

## Practical checklist for external developers

Use this checklist before integrating CLASHarness into a broader repo or rewriting docs around it.

1. Read [README](../README.md), [Multi-Agent generation pipeline](../Multi-Agent_GENERATION_PIPELINE.md), and [Evaluation Guide](evaluation.md).
2. Keep conflict-line docs under an explicitly labeled conflict-line path.
3. Treat the current seeds as pipeline fixtures, not an immutable release split.
4. Keep setup and execution-mode docs shared if the larger repo uses the same runtime.
5. Keep conflict taxonomy, seed status, and evaluator semantics conflict-line-specific.
6. Point design questions to [Multi-Agent_GENERATION_PIPELINE.md](../Multi-Agent_GENERATION_PIPELINE.md) and roadmap questions to [conflict-expansion-plan.md](conflict-expansion-plan.md).
7. If summarizing the benchmark briefly, emphasize multi-agent + temporal conflict, generator-backed scenarios, and normalized result-first evaluation.
8. If another team will actually run the code, also point them to:
   - [quickstart.md](quickstart.md)
   - [evaluation.md](evaluation.md)
   - [cli-runtime-setup.md](cli-runtime-setup.md)
   - [task-generation.md](task-generation.md)

## Recommended next documents for a larger repo

If another team is building a unified documentation tree around this codebase, the most useful next documents are:

- one umbrella README
- one Conflict-VLM overview page
- one shared setup guide
- one shared execution-modes guide
- one shared project-structure guide
- one shared evaluation entrypoint guide

This is enough to preserve CLASHarness semantics without overloading the root README.

## Final recommendation

If you need one sentence that reliably explains CLASHarness to another development team, use this:

**CLASHarness is an interactive execution benchmark whose core line, Conflict-VLM (`conflict_vlm_v1`), evaluates whether a coordinator can reconcile disagreeing subagents and time-sensitive multimodal evidence to reach the correct terminal action, built on a shared OpenClaw-style environment with generator-backed scenario creation and normalized result-first evaluation over persistent state.**

That sentence is short enough for an umbrella repo and still accurate enough to avoid the most common integration mistakes.
