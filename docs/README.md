# Documentation

This directory is the public docs hub for CLASHarness, a multi-agent computer-use benchmark centered on the **Conflict-VLM** line (`conflict_vlm_v1`). Start from the goal that matches what you need to do next.

## Understand the conflict benchmark

- [../Multi-Agent_GENERATION_PIPELINE.md](../Multi-Agent_GENERATION_PIPELINE.md): the main design doc — the 8-stage multi-agent + temporal conflict scenario generation pipeline, its failure-mode map, and the generation commands
- [conflict-expansion-plan.md](conflict-expansion-plan.md): coordination-protocol expansion roadmap (milestones M0–M6) and current build status
- [visual-web-render-expansion-plan.md](visual-web-render-expansion-plan.md): visual-surface roadmap — rendering diversity, same-platform data relatedness, and future web operations (phases D → A → C)
- [task-list.md](task-list.md): the multi-agent conflict archetype catalog that the conflict seeds are built from
- [seed-design-guide.md](seed-design-guide.md): seed authoring guidelines, scale-agnostic outage rules, and active coordinator design patterns
- [analysis/generator-failure-modes.md](analysis/generator-failure-modes.md): generator model-capability analysis and failure modes (drift, overreach, formatting gaps) with the safeguards that defeat them

## Run the benchmark

- [quickstart.md](quickstart.md): install the repo, generate the conflict dataset, and run a first evaluation
- [evaluation.md](evaluation.md): common evaluation commands, history modes, trajectory collection, per-step scoring, and behavior-data export
- [execution-modes.md](execution-modes.md): exact behavior of `mock`, `multi`, `martin`, `real`, and `hybrid`

## Extend the tool / skill surface

- [agent-tool-spec.md](agent-tool-spec.md): internal ToolSpec schema and collaborator workflow for adding command families
- [mcp-tooling-overview.md](mcp-tooling-overview.md): current MCP-like runtime logic and tool integration model
- [mcp-extension-guide.md](mcp-extension-guide.md): how to add your own command family / skill without editing backend wiring
- [conflict-plugin-guide.md](conflict-plugin-guide.md): how to add a conflict plugin — the decorator, registration, the `SPEC` contract, and a copy-paste template

## Set up live CLI-backed modes

- [cli-runtime-setup.md](cli-runtime-setup.md): CLI, gateway, and provider bootstrap for `real` and `hybrid`
- [execution-modes.md](execution-modes.md): routing and backend behavior by mode

## Regenerate artifacts

- [task-generation.md](task-generation.md): generator flags, output locations, and coverage reports
- [release-snapshot.md](release-snapshot.md): what is checked in versus what can be regenerated locally

## Develop locally

- [project-structure.md](project-structure.md): repository map, subsystem ownership, and targeted test commands
- [integration-guide.md](integration-guide.md): deeper maintainer-facing integration notes
- [../CONTRIBUTING.md](../CONTRIBUTING.md): local setup, test workflow, and contribution expectations

## Changelog

- [CHANGELOG.md](CHANGELOG.md): maintainer changelog of recent benchmark-surface changes; not the canonical source for current release defaults

Back to the repository landing page: [../README.md](../README.md)
