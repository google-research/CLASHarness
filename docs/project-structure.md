# Project Structure

Back to [README](../README.md) · See also [task-generation.md](task-generation.md) and the [Multi-Agent generation pipeline](../Multi-Agent_GENERATION_PIPELINE.md)

## Repository map

```text
cli-agent-env/
├── docs/
├── examples/
├── clasharness/
│   ├── backend/
│   ├── core/
│   ├── data/
│   ├── evaluation/
│   ├── scenario/
│   ├── skills/
│   ├── tasks/
│   └── utils/
├── scripts/
├── tests/
└── README.md
```

## Where things live

- `examples/`: runnable entrypoints such as `train_and_eval.py`
- `clasharness/backend/`: backend implementations and execution routing
- `clasharness/core/`: environment and task abstractions
- `clasharness/data/`: generated tasks, splits, and coverage reports
- `clasharness/data/scenario_seeds/`: brief Conflict-VLM scenario seeds (`conflict_vlm_v1/`, 20 fixtures spanning all 15 archetypes) used only at data-generation time
- `clasharness/data/scenarios/`: materialized Conflict-VLM scenario packages under `conflict_vlm_v1`
- `clasharness/evaluation/`: checkers, metrics, and evaluator logic
- `clasharness/scenario/`: Conflict-VLM scenario spec, LLM pipeline, materializer, and strict validation (`conflict_vlm_spec.py`, `conflict_llm_pipeline.py`, `conflict_vlm_materializer.py`, `conflict_strict_validator.py`), with taxonomy plugins in `plugins/`. `conflict_vlm_spec.py` also carries the archetype taxonomy: the `archetype` seed tag, the authoritative `ARCHETYPE_CATEGORY` token→category map, and the `archetype_coverage()` / `category_coverage()` helpers.
- `clasharness/skills/`: app-family skills and adapters, including the seven stateful coordination-protocol families (`stateful.py` base plus `impl/approval_skill.py`, `impl/lock_skill.py`, `impl/booking_skill.py`, `impl/pairing_skill.py`, `impl/deploy_skill.py`, `impl/containment_skill.py`, `impl/review_skill.py`)
- `clasharness/specs/agent_tools.v1.json`: canonical internal command/tool surface
- `clasharness/tools/`: ToolSpec loader and command validation helpers
- `clasharness/tasks/`: task generators, registry, and generation options
- `scripts/`: benchmark generation and utility scripts
- `scripts/generate_conflict_vlm_scenarios.py`: generation-time LLM pipeline for `conflict_vlm_v1` scenario packages and compiled tasks
- `tests/`: generator, runtime, and CLI regression tests

## Development checks

Targeted checks:

```bash
pytest -q tests/test_conflict_pipeline.py tests/test_conflict_vlm_reconcile.py
pytest -q tests/test_conflict_mechanics.py
pytest -q tests/test_stateful_skill.py tests/test_lock_preload.py tests/test_deploy_preload.py
pytest -q tests/test_state_isolation_fixes.py tests/test_protocol_effect_scaffold.py
pytest -q tests/test_conflict_plugin_spec.py tests/test_archetype_coverage.py
pytest -q tests/test_train_and_eval_cli.py tests/test_agent_tool_spec.py
```

Broader suite:

```bash
python -m pytest tests/ -v
```

Regenerate the Conflict-VLM scenario packages and compiled tasks offline (no network access):

```bash
python scripts/generate_conflict_vlm_scenarios.py \
  --provider mock \
  --image-provider mock \
  --chart-provider template \
  --target-rounds 12
```

Task data is written to `/tmp/clasharness-conflict-vlm-data`. See the [Multi-Agent generation pipeline](../Multi-Agent_GENERATION_PIPELINE.md) for the full 8-stage flow.

## Safety summary

The environment applies two layers of protection:

- an allowlist derived from [Agent ToolSpec](agent-tool-spec.md) for strict simulated backends
- a blocked-pattern filter for commands such as `rm -rf`, `sudo`, unrestricted `curl`/`wget`, and shell redirection

`real` and `hybrid` disable the allowlist for real `openclaw` execution, but blocked patterns still remain in force.
