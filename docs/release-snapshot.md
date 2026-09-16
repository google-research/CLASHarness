# Release Snapshot

Back to [README](../README.md) · See also [task-generation.md](task-generation.md), [evaluation.md](evaluation.md), and [conflict-expansion-plan.md](conflict-expansion-plan.md)

## What this page covers

This page explains what is treated as checked in for the Conflict-VLM (`conflict_vlm_v1`) line of CLASHarness and what is regenerated locally.

The `conflict_vlm_v1` seeds are pipeline fixtures under active development, not a frozen release. They exercise and harden the 8-stage generation pipeline (see [../Multi-Agent_GENERATION_PIPELINE.md](../Multi-Agent_GENERATION_PIPELINE.md)) and are not yet wired into a frozen `train`/`dev`/`test` release profile.

## Checked-in artifacts

The repository checks in the source-of-truth conflict artifacts under `clasharness/data/`:

- `20` conflict seed fixtures under `clasharness/data/scenario_seeds/conflict_vlm_v1/` (the source of truth, covering all 15 archetypes)
- a regenerable **subset** of materialized conflict scenario packages under `clasharness/data/scenarios/conflict_vlm_v1/` — only some seeds are materialized at any given time, and any of them can be rebuilt from the seeds (see below), so this directory is a convenience cache rather than a frozen, complete artifact
- base configuration fixtures under `clasharness/data/base_configs/`

The repository also checks in documentation-facing figure assets under `docs/assets/`.

## Locally regenerable artifacts

Compiled task data is not a frozen release artifact; it is regenerated locally from the checked-in seeds. The deterministic offline command is:

```bash
python scripts/generate_conflict_vlm_scenarios.py \
  --provider mock \
  --image-provider mock \
  --chart-provider template \
  --target-rounds 12
```

That command rebuilds the scenario packages under `clasharness/data/scenarios/conflict_vlm_v1` and writes compiled task data to `/tmp/clasharness-conflict-vlm-data`, including the split files `datasets/{train,dev,test,total}.txt`. Point the runner-facing outputs at a different root with `--output-data-dir` if you do not want to use the default `/tmp` location.

Because the seeds are still under active development, those split files are working outputs for evaluation runs rather than a frozen train/dev/test release. The mock providers and the `template` chart renderer keep the command offline and reproducible without any API calls.

## Evaluating the regenerated data

Evaluation runs against the regenerated task data through `examples/train_and_eval.py`, which supports the `mock`, `multi`, `martin`, `real`, and `hybrid` modes. See [evaluation.md](evaluation.md) for the full set of options.

For version history see [CHANGELOG.md](CHANGELOG.md).
