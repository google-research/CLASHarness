# Running Evaluations

Back to [README](../README.md) · See also [quickstart.md](quickstart.md), [execution-modes.md](execution-modes.md), and [task-generation.md](task-generation.md)

## Primary entrypoint

The main evaluation command is:

```bash
python examples/train_and_eval.py
```

Conflict-VLM evaluation is run against generated task data. A typical run is:

```bash
python examples/train_and_eval.py \
  --agent llm \
  --task-data-dir /tmp/clasharness-conflict-vlm-data \
  --split total \
  --mode martin \
  --max-steps 20
```

Generate the task data first with `scripts/generate_conflict_vlm_scenarios.py` (see [task-generation.md](task-generation.md) and [../Multi-Agent_GENERATION_PIPELINE.md](../Multi-Agent_GENERATION_PIPELINE.md)); the generator writes compiled task data to `/tmp/clasharness-conflict-vlm-data` by default (override with `--output-data-dir`).

### Step budget

The CLI default step budget is `15` (`examples/train_and_eval.py`, `--max-steps`), but
benchmark-facing runs must set it explicitly. **The conflict evaluation protocol is
`--max-steps 20`**, and every reference number in this document was measured under it.

The budget is deliberately uniform: `_resolve_max_steps` discards `split`, `task_prefix`,
and `task_id` overrides, so all models in a comparison get exactly the same allowance and
step budget cannot become a confound.

`20` is calibrated against the reference solutions rather than picked by feel. Over the
600 compiled conflict tasks, the ground-truth `solution_commands` length is:

| statistic | gold steps |
| --- | --- |
| min | 8 |
| median | 11 |
| mean | 11.3 |
| max | 19 |
| tasks needing more than 20 | **0** |

So the cap is about `1.8x` the gold median and **no reference solution exceeds it**. Note
the shape of that headroom: it is generous on the typical task and thin on the hardest —
the longest gold path is `19` steps, leaving one step of slack. The budget is therefore
sufficient rather than lavish, which is the intent.

The consequence for interpretation: a model that exhausts 20 steps was not cut off before
a solution became reachable, so exhausting the budget is a convergence failure, not a
measurement artifact. That matters when reading the results below, where the weakest model
hits the cap on most tasks while its step-count distribution sits far above gold.

## Common commands

Conflict-VLM run with a hosted OpenAI-compatible endpoint:

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

Expert sanity check (upper bound):

```bash
python examples/train_and_eval.py \
  --agent expert \
  --task-data-dir /tmp/clasharness-conflict-vlm-data \
  --split total \
  --mode martin \
  --max-steps 20
```

**Pass `--max-steps` explicitly here.** It defaults to `15` and is not adapted to the
data — `_resolve_max_steps` returns the flag value unchanged. The expert agent replays
the authored solution, so any task whose solution is longer than the ceiling is cut off
with `stop_reason = "reached max_steps=15"` and the "upper bound" silently lands below
`1.0` — which looks like a broken benchmark rather than a truncated run. Use the same
ceiling you use for the model runs (`20` above), since results are only comparable
across cells that share a step budget.

Small sanity check on a few tasks:

```bash
python examples/train_and_eval.py \
  --agent llm \
  --task-data-dir /tmp/clasharness-conflict-vlm-data \
  --split dev \
  --limit 10
```

## Useful output flags

- `--verbose-log PATH`: save the full command trace
- `--save-report PATH`: save structured JSON output (evaluation mode only — see below)
- `--collect-data --out-dir PATH`: save JSONL trajectories for later analysis

> **`--collect-data` and `--save-report` are mutually exclusive in practice.**
> `--collect-data` takes a separate branch that returns before the evaluation path runs
> (`examples/train_and_eval.py:3041-3070`), so the report block at `:3099` is never
> reached and `--save-report` is **silently ignored** — no warning, no file. Run the two
> as separate invocations. For the same reason `--parallel-workers`,
> `--resume-trajectories`, and `--retry-provider-failures-only` only do anything when
> `--collect-data` is given; without it they are no-ops.
- `--record-step-eval`: add per-step evaluator progress to collected trajectories
- `--llm-request-retries N`: retry provider failures
- `--llm-retry-backoff-s S`: backoff base for retries
- `--inter-task-sleep S`: throttle between tasks

Structured reports include provider-aware fields such as:

- `provider_failures`
- `provider_impacted_tasks`
- `provider_adjusted_accuracy`

## History modes

The rollout client supports three history policies:

- `full`: send the full message history
- `summary`: force the benchmark's compressed-history path
- `auto`: use the benchmark's automatic trimming logic

The current default rollout policy is `full`. Example:

```bash
python examples/train_and_eval.py \
  --agent llm \
  --llm-provider openai \
  --model claude-sonnet-4.6 \
  --task-data-dir /tmp/clasharness-conflict-vlm-data \
  --split total \
  --mode multi \
  --llm-history-mode full \
  -v
```

## Metrics and reports

The evaluator reports:

- full-pass accuracy
- partial-credit average score
- scenario-level breakdowns
- primary-ability and overlapping-tag summaries
- provider-aware accounting such as `provider_failures` and `provider_impacted_tasks`

Provider-aware fields are useful when comparing interactive runs across external endpoints, especially when retries, filtered responses, or compact fallbacks affect the rollout without fully invalidating the task.

`avg_score` is the dataset mean of the per-task weighted aggregate score computed by the same result-first evaluator used for full-pass accuracy. A task can therefore receive substantial partial credit even when it does not satisfy every required check.

## Reference results

Protocol: `600` tasks per cell, `--mode multi`, `--max-steps 20`, `n=1`, semantic checks
graded by `gemini-3.6-flash`. Each model is run at three reasoning-effort settings.
Across all 27 collected cells `provider_failures` never exceeded `5` (`5/600` = `0.8%`), so
no cell is provider-impacted enough to need `provider_adjusted_accuracy`.

Full-pass accuracy (%):

| model | high | medium | low |
| --- | --- | --- | --- |
| `opus-5` | 25.2 | **25.3** | 23.5 |
| `sol` | 24.5 | 23.0 | 20.8 |
| `opus-4.8` | 22.8 | 23.8 | 21.3 |
| `sonnet-5` | 20.8 | 17.0 | 13.8 |
| `terra` | 20.0 | 20.0 | 17.7 |
| `luna` | 19.8 | 17.7 | 19.5 |
| `3.6-flash` | 18.5 | 20.2 | 18.5 |
| `3.1-pro` | 12.0 | 9.8 | 8.3 |
| `3.5-flash` | 3.0 | 3.2 | 13.5 |
| `gpt-5.5` | — | — | — |
| **mean** (9 models) | **18.5** | **17.8** | **17.4** |

`gpt-5.5` is intentionally blank: those cells could not be collected. The endpoint returned
a stable `input-tpm ... (InputTokens) exceeded` 429 on every attempt. This is contention on
a **shared** token-per-minute quota, not a property of this benchmark's requests: the
prompts are small (a low single-digit thousands of tokens), the reported overage was
byte-identical across attempts rather than scaling with request size, and reducing client
concurrency did not help — the bottleneck is not on the client side. Treat the panel as
nine models until those cells are re-collected off-peak.

### Reading the panel

- **Headroom.** The strongest cell in the panel is `25.3%`. The benchmark is far from
  saturated, and the ceiling is not an artifact of the step budget (see
  [Step budget](#step-budget) — no task needs more than 20 steps, and the gold path
  median is 11).
- **Separation.** Using `mean(top 3) − mean(bottom 3)`, the spread is `13.0pp` at high,
  **`14.0pp` at medium**, and `10.0pp` at low. Medium separates the field best; a
  single-setting protocol should use it.
- **Reasoning effort is a weak lever.** The panel mean moves only `1.1pp` from high to
  low. Only `sonnet-5` is monotone in effort (`+7.0pp` high over low). Effort is not a
  reliable difficulty knob on this benchmark.
- **The budget does not bind the strong models.** At high effort `opus-5` has a median
  run length of `10` steps against a gold median of `11`, and repeats only `2.1%` of its
  commands. It is capped on `3/600` tasks. Whatever is stopping it at `25.2%`, it is not
  the step budget.
- **`3.5-flash` inverts.** It scores `3.0%` at high but `13.5%` at low. At high effort it
  exhausts the budget on `89.2%` of tasks (`535/600`), median run length `19` steps
  against a gold median of `11`. It is not looping: only `3.5%` of its attempts repeat a
  command already issued in the same episode — *lower* than `3.1-pro`'s `7.6%`. It is
  over-exploring, issuing mostly novel commands and still failing to converge, which is a
  real capability gap rather than a budget artifact.

  (Duplicate rate here is computed from the collected interaction logs — the share of
  `attempt` records whose normalized command already appeared earlier in the same
  episode — not from `parallel_run.summary`.)

Partial credit tracks accuracy closely. At high effort, `avg_score` runs from `.6419`
(`opus-5`) down to `.2594` (`3.5-flash`), panel mean `.5588` — models that miss full-pass
still accumulate meaningful partial credit, so accuracy alone understates the gap between
the middle of the field and the bottom.

## Trajectory analysis and export

Collected trajectories are the behavior-observation surface for interactive LLM runs. They can be summarized without re-running evaluation:

```bash
python scripts/analyze_trajectories.py \
  /tmp/clasharness-conflict-vlm-data/trajectories/llm_total.jsonl
```

The analyzer reports overall success, provider-impacted records, average steps, per-stage performance, per-scenario performance, and failure modes.

To export reusable behavior data:

```bash
python scripts/export_trajectory_dataset.py \
  /tmp/clasharness-conflict-vlm-data/trajectories/llm_total.jsonl \
  --output-dir /tmp/clasharness-conflict-behavior-data
```

This writes:

- `episodes.jsonl`: one normalized record per task episode
- `steps.jsonl`: one transition per executed command
- `sft_commands.jsonl`: instruction/context to next-command examples from successful or expert trajectories

Filters such as `--success-only`, `--scenario-slug`, `--round-type`, `--agent`, and `--min-score` are available for both analysis and export scripts.

## Baseline agents

| Agent | Class | Description |
| --- | --- | --- |
| `expert` | `ExpertAgent` | replays ground-truth commands |
| `random` | `RandomAgent` | samples commands randomly |
| `rule` | `RuleAgent` | keyword-matching heuristic |
| `llm` | `LLMAgent` | multi-turn LLM-driven command agent |
