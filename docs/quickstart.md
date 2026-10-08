# Quick Start

Back to [README](../README.md) · See also [evaluation.md](evaluation.md), [execution-modes.md](execution-modes.md), [task-generation.md](task-generation.md), [cli-runtime-setup.md](cli-runtime-setup.md), [conflict-expansion-plan.md](conflict-expansion-plan.md), and [../Multi-Agent_GENERATION_PIPELINE.md](../Multi-Agent_GENERATION_PIPELINE.md)

A first-run guide for the Conflict-VLM line (`conflict_vlm_v1`). It walks three
separate things, in the order you should do them:

1. **Set up and verify the checkout** — no credentials needed.
2. **Smoke-test the pipeline offline** with the `mock` provider — no credentials
   needed. This proves the plumbing runs; it does **not** produce benchmark data.
   Read [Offline smoke test](#2-offline-smoke-test-no-credentials) for why.
3. **Generate real benchmark data and evaluate a model** — needs a live model
   provider.

The generator's design is documented in
[../Multi-Agent_GENERATION_PIPELINE.md](../Multi-Agent_GENERATION_PIPELINE.md).

---

## 1. Set up

Requires **Python 3.11+** (developed and tested on 3.13).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

That installs the runtime dependencies (`anthropic`, `google-genai`,
`google-api-python-client`, `google-auth`, `google-auth-oauthlib`, `Pillow`,
`matplotlib`, `numpy`) plus `pytest` and `pytest-asyncio`.

Both entrypoints (`examples/train_and_eval.py` and
`scripts/generate_conflict_vlm_scenarios.py`) put the repository root on
`sys.path` themselves, so **installing this package is not required to run
them** — only the third-party dependencies are. If your environment blocks
`pip install -e .` (for example a restricted internal index with no
`setuptools`), install just the dependencies above and the commands below still
work unchanged.

Verify the checkout:

```bash
python -m pytest tests -q
```

Expected: **2966 passed, 2 skipped, 497 subtests passed** (2,968 collected, ~8 min).
`pyproject.toml` pins `testpaths = ["tests"]`, so a bare `pytest` from the
repository root does the same thing.

---

## 2. Offline smoke test (no credentials)

The `mock` providers are fully offline and deterministic:

```bash
python scripts/generate_conflict_vlm_scenarios.py \
  --provider mock \
  --image-provider mock \
  --chart-provider template \
  --target-rounds 12 \
  --output-dir /tmp/clasharness-smoke-scenarios \
  --output-data-dir /tmp/clasharness-smoke-data
```

**Read the output carefully — this command reports 14 failures and still exits
zero.** Measured on the checked-in seeds:

| What you will see | Value |
| --- | --- |
| Seeds attempted | 20 |
| Successfully materialized | **6** |
| Failed (retries exhausted) | **14** |
| Tasks written from the 6 | 72 (`train` 50 / `dev` 11 / `test` 11) |
| Wall clock | ~20 s |
| Exit code | **0** |

Two distinct things are happening, and neither is a broken checkout:

- **14 seeds do not materialize.** The mock narrative provider emits a fixed
  6-step outline, and those seeds require at least 8 steps
  (`Discovery -> Inspection -> Wait -> Action -> Final Verification`). The exit
  code does not reflect this, so *always read the "Successfully materialized"
  line* rather than trusting the exit code.
- **The 72 tasks that do materialize start out non-discriminating, and are
  repaired in place.** Each scenario's zero-action resolver replays the round
  with an empty trajectory and demotes the checks that still pass — 12 per
  scenario, here — from required to diagnostic. By the time the run-level
  `--audit-discrimination` gate replays all 72 tasks it therefore finds
  `0 vacuous required check(s), 0 non-discriminating task(s)` and passes. The
  demotion count is the number worth reading: on mock data essentially every
  check is vacuous before repair.

So: **`--provider mock` is a pipeline smoke test, not a way to produce benchmark
data.** Use it to confirm generation, materialization, and evaluation run
end-to-end on your machine. The narrative, the conflicts and the visual evidence
are all fixed templates, so nothing generated this way measures a model's ability
to resolve conflict.

To exercise the evaluation half too, generate one seed's worth of smoke data:

```bash
python scripts/generate_conflict_vlm_scenarios.py \
  --provider mock --image-provider mock --chart-provider template \
  --seed-name incident_commander_selection \
  --num-versions 1 --target-rounds 12 \
  --output-dir /tmp/clasharness-smoke-scenarios \
  --output-data-dir /tmp/clasharness-smoke-data
```

Then replay the reference solution with the scripted `expert` agent:

```bash
python examples/train_and_eval.py \
  --agent expert \
  --task-data-dir /tmp/clasharness-smoke-data \
  --split total \
  --mode multi \
  --max-steps 30
```

**With no judge key set this reports `0/12 passed` and still exits 0, and that is
the expected offline result.** The reason is worth understanding, because it is
the same mechanism as the zero-action resolver above: on this mock scenario the
resolver demotes the two deterministic checks to `role: diagnostic`, which leaves
exactly one gating check per task —

```json
{"type": "effect", "effect_type": "messages_sent", "condition": "field_semantic_match",
 "expected": {"field": "message", "criterion": "confirms the current operational status with the peer"}}
```

— and `field_semantic_match` is graded by the LLM judge. With no judge configured
the harness fails those checks closed rather than silently passing them, so the
run prints `no LLM judge configured`, then `infra_failures=12 (100.0%)` on the
judge-health line, and scores zero. Set `--llm-judge-provider` (or any of
`GEMINI_API_KEY` / `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`) to grade it for real.

Read the `Judge health` line before reading the score on any run. On real data an
`expert` failure with a healthy judge means the generated data is broken, not that
the model is weak.

---

## 3. Generate real benchmark data

Discriminating tasks require a live text provider. `--provider` accepts
`mock`, `gemini`, `anthropic`, `openai`, `vertex`, `vertex-anthropic`, and
`bedrock-gpt`. The first four need nothing but an API key
(`GEMINI_API_KEY` / `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`). The last three
reach the same model families through Google Cloud and AWS — how the released
corpus was generated — and need cloud credentials.

```bash
python scripts/generate_conflict_vlm_scenarios.py \
  --provider gemini --model <text-model-id> \
  --image-provider google --chart-provider code \
  --target-rounds 12 \
  --num-workers 20 --num-generation-workers 20 --num-render-workers 20 \
  --resume true \
  --output-dir /tmp/clasharness-scenarios \
  --output-data-dir /tmp/clasharness-data
```

Two quality gates are **on by default** and both exit non-zero without deleting
the generated data:

- `--audit-discrimination` — fails if any task passes with an empty trajectory.
- `--fail-on-unwinnable` — fails if an exact-graded round demands a value that
  is readable from nothing the round shows the solver.

Treat a non-zero exit here as a real finding about the generated corpus. See
[task-generation.md](task-generation.md) for the full flag surface, and
[../Multi-Agent_GENERATION_PIPELINE.md](../Multi-Agent_GENERATION_PIPELINE.md)
for the pipeline design.

Every flag with more than one word accepts both `--snake_case` and
`--kebab-case` on this generator. `examples/train_and_eval.py` accepts the
kebab form only.

---

## 4. Evaluate a model

```bash
python examples/train_and_eval.py \
  --agent llm \
  --llm-provider gemini --model <model-id> \
  --llm-judge-provider gemini --llm-judge-model <judge-model-id> \
  --task-data-dir /tmp/clasharness-data \
  --split total \
  --mode multi \
  --max-steps 20 \
  --parallel-workers 20 \
  --llm-tool-mode native \
  --collect-data --capture-log \
  --out-dir /tmp/clasharness-eval
```

`--parallel-workers`, `--resume-trajectories`, and
`--retry-provider-failures-only` only take effect together with
`--collect-data`. `--save-report` is ignored when `--collect-data` is set — the
collection path returns before the report is written
(`examples/train_and_eval.py:3070`).

The judge grades the semantic checks, so it is part of the protocol. To compare against the
published v8 numbers, use the judge they used, `gemini-3.6-flash`. The exception is
`gemini-3.7-flash`, whose three effort cells as `analysis/` reads them were graded by
`gemini-3.7-flash` (see `RUN_OVERRIDE` in `analysis/corpus.py`).

Each semantic check gets three votes at temperature 0. Claude judges that reject a
temperature (Sonnet 5.x, Opus 4.7 and later) vote at their default temperature instead, so
their grades are sampled rather than greedy and less reproducible; the run's `Judge :` line
notes this.

### Choosing the step budget

`--max-steps` defaults to `15`, which is below the longest reference solution
(`19` steps, see [evaluation.md](evaluation.md)). Always set it explicitly for
benchmark-facing runs.

**There is no single correct value, and it is not a neutral knob.** A panel is
comparable only if every cell shares one budget:

| Corpus | Budget | Why |
| --- | --- | --- |
| v8 | `20` | The shipped v8 panel ran at 20. |
| Sonnet-5 | `30` | At 20, five of its six arms hit the ceiling on 38–93% of rounds, and a truncated arm measures the budget rather than the model. |

The authoritative value for a panel lives in `scripts/run_effort_panel.sh`
(`MAX_STEPS`, default `20`). Never compare cells run under different budgets.

---

## 5. Execution modes

`--mode` accepts `mock`, `multi`, `martin`, `real`, and `hybrid`. `multi` (and
its alias `martin`) is the default benchmark mode: it preserves interactive
state across command families without requiring an online runtime. See
[execution-modes.md](execution-modes.md) for exact backend behavior, and
[cli-runtime-setup.md](cli-runtime-setup.md) for the extra bootstrap `real` and
`hybrid` need.

## Common next steps

- Evaluation commands, output flags, and metric interpretation: [evaluation.md](evaluation.md)
- Generator flags and regenerating snapshots: [task-generation.md](task-generation.md)
- Mode backends and when to use each: [execution-modes.md](execution-modes.md)
- CLI runtime, gateway, and provider bootstrap: [cli-runtime-setup.md](cli-runtime-setup.md)
- Coordination-protocol expansion roadmap: [conflict-expansion-plan.md](conflict-expansion-plan.md)
- The full 8-stage generation pipeline: [../Multi-Agent_GENERATION_PIPELINE.md](../Multi-Agent_GENERATION_PIPELINE.md)
