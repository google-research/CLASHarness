# Task Generation

Back to [README](../README.md) · [Generation Pipeline](../Multi-Agent_GENERATION_PIPELINE.md) · [Seed Design](seed-design-guide.md) · [Conflict Expansion Plan](conflict-expansion-plan.md) · [Evaluation](evaluation.md)

This is the practical, run-the-generator guide for the Conflict-VLM line
(`conflict_vlm_v1`). It documents the `scripts/generate_conflict_vlm_scenarios.py`
CLI: its flags, the canonical commands, and where outputs land. For the deep
design of the 8-stage hierarchical generation pipeline, see
[Multi-Agent_GENERATION_PIPELINE.md](../Multi-Agent_GENERATION_PIPELINE.md).

The generator compiles each JSON seed into a multi-round, multimodal conflict
scenario package and then compiles those rounds into runnable `Task` specs.
Generation is an offline data-construction step: it may call an LLM (and, for
live runs, image/chart backends) while building data, but evaluation runs against
the frozen tasks on disk.

## Canonical commands

Offline, deterministic (no network; mock providers). Use this for tests, CI, and
local development:

```bash
python scripts/generate_conflict_vlm_scenarios.py \
  --provider mock \
  --image-provider mock \
  --chart-provider template \
  --target-rounds 12
```

Live generation with Gemini and parallel workers:

```bash
python scripts/generate_conflict_vlm_scenarios.py \
  --model gemini-3.5-flash \
  --max-output-tokens 8192 \
  --target-rounds 12 \
  --num-generation-workers 4 \
  --num-render-workers 8
```

Resume an interrupted run from the cached spec and rendered assets by adding
`--resume` to any of the above:

```bash
python scripts/generate_conflict_vlm_scenarios.py \
  --model gemini-3.5-flash \
  --target-rounds 12 \
  --resume
```

## Output locations

By default the generator writes:

- **Scenario packages** to `--output-dir`
  (`clasharness/data/scenarios/conflict_vlm_v1`): the materialized
  `scenario.json`, workspace files, sessions, updates, and rendered assets.
- **Compiled task data** to `--output-data-dir`
  (`/tmp/clasharness-conflict-vlm-data`): runnable `Task` specs under `tasks/`,
  plus split index files `datasets/{train,dev,test,total}.txt`.

Point `--output-data-dir` at a scratch root (as the default already does) when you
want runner-facing outputs kept out of the checked-in data tree.

## Flags

All flags accept the hyphenated form shown below (matching the root README). Only
the flags listed here exist on the generator; anything else is not supported.

### Provider and model

| Flag | Purpose |
| --- | --- |
| `--provider {mock,gemini,anthropic,openai,vertex,vertex-anthropic,bedrock-gpt}` | LLM provider for narrative text (default `vertex`). **API-key providers**, which need no cloud account: `mock` is fully offline; `gemini` is the official Gemini API; `anthropic` is Claude's official API; `openai` is GPT's official Responses API. **Cloud providers**, which the released corpus was generated through and which need cloud credentials: `vertex` is Gemini on Vertex AI; `vertex-anthropic` is Claude on Vertex AI (ADC; the model must be an `@`-form publisher id such as `claude-sonnet-4@20250514`, served in a Claude region such as `us-east5`); `bedrock-gpt` is GPT on Bedrock via the OpenAI Responses API (the model carries an `openai.` prefix, e.g. `openai.gpt-5.5`). |
| `--model <name>` | Text model for the chosen `--provider` (default `gemini-1.5-pro`): a Gemini id for `gemini`/`vertex`, a public Claude id (`claude-sonnet-5`) for `anthropic`, a bare GPT id (`gpt-5.5`) for `openai`, an `@`-form Claude publisher id for `vertex-anthropic`, an `openai.`-prefixed id for `bedrock-gpt`. |
| `--vertex-project <id>` | GCP project ID for Vertex AI generator calls. Ignored by the API-key providers. |
| `--vertex-location <loc>` | GCP location for Vertex AI generator calls. Ignored by the API-key providers. |
| `--base-url <url>` | API endpoint base URL for the HTTP providers (`gemini`, `openai`, `bedrock-gpt`). Left at the Gemini default it is auto-swapped to the chosen provider's default — `https://api.openai.com/v1` for `openai`, the Bedrock endpoint for `bedrock-gpt`. Ignored by `anthropic` (the SDK owns the host) and by the ADC providers. |
| `--api-key-env <name>` | Env var holding the text API key (default `GEMINI_API_KEY`). Left at that default it is auto-swapped to the chosen provider's default: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `BEDROCK_OPENAI_API_KEY`. Whatever it names is also mirrored into the canonical variable so the semantic-match judge shares the same credential. |

### Seeds and outputs

| Flag | Purpose |
| --- | --- |
| `--seed-dir <path>` | Directory of VLM seed files (default `clasharness/data/scenario_seeds/conflict_vlm_v1`). |
| `--seed-name <stem>` | Run only the seed whose filename stem matches (e.g. `distributed_lock_shared_resource`). |
| `--output-dir <path>` | Scenario package output directory (default `clasharness/data/scenarios/conflict_vlm_v1`). |
| `--output-data-dir <path>` | Compiled task-data root (default `/tmp/clasharness-conflict-vlm-data`); writes `datasets/{train,dev,test,total}.txt`. |
| `--num-versions <int>` | Number of scenario variations to generate from each seed (default `1`). |

### Visual assets

| Flag | Purpose |
| --- | --- |
| `--image-provider {mock,google,vertex}` | Image generator for visual assets (default `google`). |
| `--chart-provider {mock,template,code,google}` | Chart rendering strategy (default `template`). |
| `--google-model <name>` | VLM model used to refine visual definitions (default `gemini-3.1-flash-image`). |
| `--google-api-key-env <name>` | API-key env var for the VLM image model (default `GEMINI_API_KEY`). |
| `--google-project <id>` | GCP project ID for Vertex AI image calls. |
| `--image-vertex-location <loc>` | Location for the Vertex AI image provider; falls back to `--vertex-location`. Set this when the image model is only served in a different region. |
| `--vision-audit` / `--novision-audit` | Audit each decisive image with the vision model so its deciding token renders legibly (on by default). `--novision-audit` skips it; it is already a no-op for `--image-provider mock`. |

### Generation control

| Flag | Purpose |
| --- | --- |
| `--target-rounds <int>` | Override the per-seed target round count. |
| `--max-output-tokens <int>` | Output cap per LLM call (default `16384`). |
| `--temperature <float>` | Model generation temperature (default `0.2`). |
| `--retries <int>` | Retries per call after retryable provider failures (default `5`). |
| `--version-retries <int>` | Extra whole-version attempts after a version fails (default `1`). Each retry resumes from the pipeline checkpoint, which is kept outside the deleted scenario directory, so it re-runs only the stage that failed rather than the whole scenario. Retries change nothing about validation — every gate still applies. `0` restores the historical fail-once behaviour. |
| `--auditor-max-iterations <int>` | Max iterations for the inconsistency-auditor repair loop (default `3`). |
| `--phase-expand-attempts <int>` | Retry budget for the abort-prone phase-expansion step; a single phase failure degrades to a salvage outline instead of discarding the scenario. |
| `--style-seed <int>` | Pin visual styles/themes; defaults to a random seed each run. |
| `--request-delay <float>` | Delay in seconds between API calls. |

### Parallelism

| Flag | Purpose |
| --- | --- |
| `--num-workers <int>` | Worker threads across seeds/versions. |
| `--num-generation-workers <int>` | Worker threads for parallel round generation within a scenario. |
| `--num-render-workers <int>` | Worker threads for parallel headless-Chrome asset rendering. |

Nested concurrency is allowed (version workers plus generation/render workers).
Reduce `--num-generation-workers` or `--num-render-workers` if you hit rate-limit
warnings or high memory use.

### Train/dev/test splits

Splits are written to `--output-data-dir` as `datasets/{train,dev,test,total}.txt`.

| Flag | Purpose |
| --- | --- |
| `--split-ratios <t,d,e>` | `train,dev,test` proportions, normalised to sum to 1 (default `0.7,0.15,0.15`). Applied to **groups**, not to individual tasks. |
| `--split-group-by {seed,version,task}` | Unit kept wholly inside one split (default `seed`). `seed` holds out entire scenario families; `version` holds out scenario instances; `task` does not group at all and **leaks rounds of the same episode across splits**. |
| `--split-salt <str>` | Salt for the deterministic group-to-split hash (default `clasharness-conflict-vlm`). Change it to reshuffle the split; keep it fixed to keep results comparable. |

### Post-generation gates

Both gates run after generation and **exit non-zero on a finding without deleting
the generated data**, so you can inspect what tripped them.

| Flag | Purpose |
| --- | --- |
| `--audit-discrimination` / `--noaudit-discrimination` | On by default. Replays every new task with an **empty** trajectory and reports REQUIRED checks that still pass (free points) and tasks that succeed outright (non-discriminating). |
| `--fail-on-unwinnable` / `--nofail-on-unwinnable` | On by default. Fails if any exact-graded round demands a value readable from nothing the round shows the solver. Such a round caps its task below 1.0 for every model forever, and gold still passes it (gold is authored from `resolved_value`), so no other gate catches it. `--nofail-on-unwinnable` ships anyway. |

### Run lifecycle

| Flag | Purpose |
| --- | --- |
| `--resume` | Resume from the existing spec and cached assets on disk instead of regenerating. |
| `--clean` | Delete compiled tasks, datasets, trajectories, and scenario specs in the output directories before starting. |
| `--log-file <path>` | Mirror all log output to this file (append mode) so per-attempt failure reasons survive terminal scrollback and parallel runs. |

## Seeds

`conflict_vlm_v1` ships `20` seed fixtures under
`clasharness/data/scenario_seeds/conflict_vlm_v1/`, covering all 15 conflict
archetypes across the 5 top-level categories. These are pipeline fixtures
that exercise and harden the generator; they are not a frozen `train`/`dev`/`test`
release. For how to author or extend seeds, see
[seed-design-guide.md](seed-design-guide.md); for the coordination-protocol
expansion roadmap, see [conflict-expansion-plan.md](conflict-expansion-plan.md).

## Verify

Confirm the pipeline end to end with the offline regression test:

```bash
python -m pytest tests/test_conflict_pipeline.py
```

This runs the generator against the `mock` provider so it stays deterministic and
network-free. To confirm archetype/category coverage across the seed set — that
every one of the 15 archetypes is realized by a seed and all 5 categories are
covered (via `category_coverage()`) — also run:

```bash
python -m pytest tests/test_archetype_coverage.py
```

## See also

- [Multi-Agent_GENERATION_PIPELINE.md](../Multi-Agent_GENERATION_PIPELINE.md) — full 8-stage pipeline design.
- [evaluation.md](evaluation.md) — run and interpret evaluations over the generated tasks.
- [release-snapshot.md](release-snapshot.md) — how a snapshot is captured and described.
- [agent-tool-spec.md](agent-tool-spec.md) — the tool/skill surface the tasks exercise.
