# CLASHarness Scenario Seed Design Guide

This guide details how to design and write high-quality **Scenario Seeds** for the CLASHarness multi-agent conflict benchmark. A seed is a declarative JSON file that guides the LLM pipeline to expand a high-level troubleshooting situation into a complete, multi-round, multi-modal incident scenario.

---

## 1. Anatomy of a Scenario Seed

Scenario seeds are stored under `clasharness/data/scenario_seeds/`. A typical seed file contains the following fields:

```json
{
  "slug": "multimodal_temporal_conflict_01",
  "domain": "multimodal DevOps operations",
  "title": "Cascading Outage and Out-of-Sync Context Reconciliation",
  "target_rounds": 12,
  "participants": [
    "Ops_Agent",
    "Dev_Agent",
    "DB_Admin_Agent"
  ],
  "required_constraints": [
    "Enforce a cascading outage: Latency spike (initial phase) -> Outage (middle phases) -> Database pool connection exhaustion (final phases before recovery).",
    "The edge proxy configuration file edge_proxy.yaml is encrypted. The coordinator cannot audit it directly without the key file proxy_key.txt.",
    "The Dev Agent has mutated the configuration but relies on stale local cache. The coordinator must extract the mutation commit SHA 'f8c22d1' from the pipeline execution log, and align the Dev Agent context using instructions containing this exact SHA.",
    "The Dev Agent will remain out-of-sync until they are aligned with the exact commit SHA, after which they will write the decryption key file 'proxy_key.txt' to the workspace.",
    "The coordinator must read 'proxy_key.txt', decrypt and audit edge_proxy.yaml directly, identify the misconfigured routing target causing the outage, and edit the configuration file to restore healthy routing."
  ],
  "allowed_tool_families": [
    "asset",
    "agents",
    "file",
    "message"
  ],
  "modalities": [
    "chart",
    "rich_text"
  ],
  "input_policy": {
    "default_mode": "path",
    "supports_direct": true,
    "supports_both": true
  },
  "generation_policy": {
    "chart_provider": "template",
    "natural_image_provider": "mock",
    "google_model": "gemini-3.1-flash-image",
    "chart_layouts": {
      "asset_bg_01": "devops_dashboard"
    }
  },
  "min_workspace_files": 5,
  "min_chat_sessions": 2,
  "min_chat_messages": 15
}
```

### Core Schema Definition:
*   `slug`: A unique, snake_case string identifying the scenario.
*   `domain`: The operational context (e.g. `multimodal database management`, `release engineering`).
*   `target_rounds`: The default number of rounds for evaluation.
*   `participants`: The list of active mock subagents that the main agent can delegate tasks to or chat with.
*   `required_constraints`: A list of rules and chronological events that the generation pipeline MUST enforce in the generated outlines, dialogue updates, and evaluators.
*   `allowed_tool_families`: The CLI command families available to the candidate agent during evaluation (e.g. `file` for read/edit, `agents` for delegation, `asset` for opening metrics, `message` for chat).

---

## 2. Core Design Principles

To ensure scenarios are robust, scale-flexible, and challenging, always follow these four design principles:

### A. Scale-Agnostic Constraint Design (Dynamic Round Scaling)
CLASHarness allows users to override the round count during task generation (e.g., generating a short 10-round test vs. a long 24-round full benchmark). 
**Never hardcode specific round numbers** in seed constraints. Instead, express constraints using relative or proportional timeline phases.

*   ❌ **Bad (Hardcoded Rounds)**:
    `"Enforce a cascading outage: Latency spike (Rounds 1-3) -> Outage (Rounds 4-8) -> Database pool connection exhaustion (Rounds 9-12)."`
*   ✔ **Good (Scale-Agnostic Phases)**:
    `"Enforce a cascading outage: Latency spike (initial phase) -> Outage (middle phases) -> Database pool connection exhaustion (final phases before recovery)."`

#### Why it matters:
The LLM pipeline dynamically scales stages (Phases) to fit the `target_rounds` count. When relative terms are used, the generator will correctly stretch or shrink the outage stages to match any round length without creating empty rounds or parsing conflicts.

---

### B. Active Coordinator Mitigation (Evaluation Depth)
The evaluated agent acts as the **Coordinator (main agent)**. In multi-agent conflict scenarios, do not let the Coordinator behave passively by delegating all heavy-lifting tasks to subagents.

*   ❌ **Bad (Passive Coordinator)**:
    `"The coordinator must read 'proxy_key.txt' and pass it to DB_Admin_Agent to decrypt and audit edge_proxy.yaml, resolving the outage."`
    *(Impact: Coordinator becomes a passive message-passer that only runs `agent delegate` commands.)*
*   ✔ **Good (Active Coordinator)**:
    `"The coordinator must read 'proxy_key.txt', decrypt and audit edge_proxy.yaml directly, identify the misconfigured routing target, and edit the configuration file to restore routing."`
    *(Impact: Coordinator must directly execute `file read`, run decryption scripts, perform file comparisons, and use `file edit` to modify configs, testing its direct engineering skills.)*

#### Guidelines for active steps:
1.  Use subagents for **information retrieval**, **domain-specific telemetry queries**, and **unlocking credentials** (the "cache alignment" part).
2.  Use the Coordinator for **critical configuration audits**, **running decryption tools**, **deciding on mitigation edits**, and **loading updated settings**.

---

### C. Multi-Agent Conflict Semantics (Cache & Context Alignment)
CLASHarness DevOps scenarios model realistic system and communication challenges, not human malice.
**Do not use anthropomorphic terms** like "deception", "lying", "withholding secrets", or "confessing". Instead, frame the friction around **stale caches** and **evidence-based context alignment**.

*   ❌ **Bad (Adversarial Framing)**:
    `"Dev Agent mutated the routing but lies about it. The coordinator must confront the Dev Agent with the commit SHA to force them to confess and hand over the password."`
*   ✔ **Good (Technical Cache Alignment)**:
    `"Dev Agent mutated the configuration but relies on stale local cache. The coordinator must align the Dev Agent context by presenting the latest commit SHA, prompting the agent to refresh its context and write the decryption key file."`

#### Key mappings:
*   *Lying/Withholding* ➔ **Out-of-sync local cache**, **stale telemetry view**, **hallucinated metrics**, or **attention slip**.
*   *Confrontation/Confession* ➔ **Evidence-based context alignment**, **context synchronization**, or **metadata reconciliation**.

---

### D. Authentic Evidence Modality (Visual Diversity)
The asset planner chooses each visual's kind (`chart`, `rich_text`, `diagram`, `natural_image`, `sketch`) from **how the seed describes its evidence**. Describe every artifact by its **real-world form** and let the mix of kinds follow the domain honestly — never add a photograph just to look varied, and never flatten a physical object into a chart/diagram.

*   ❌ **Bad (Form flattening)**:
    Describing every artifact as a "dashboard chart" or "topology diagram" even when the ground truth is a physical object. The planner faithfully renders whatever the seed names, so the benchmark collapses to charts/diagrams with **no real photographs**.
*   ✔ **Good (Authentic forms)**:
    A warehouse safety-halt scenario cites a **photograph** of the robot cell's status beacon; a purely digital outage cites **charts** and **logs** and legitimately has **zero** photographs.

#### Automatic selection (no fields required):
The asset planner is **domain-aware**: when a seed's `domain` centers on physical equipment or environments (industrial robots / robot cells, antennas & dishes, hardware / server racks, vehicles, lab instruments or samples, field sites), it **automatically** includes at least one `natural_image` photograph of the physical subject — you do **not** need an `evidence` list or any special field for photos to appear. Purely digital domains (dashboards, logs, code, web tooling) automatically stay photo-free.

What the planner cannot infer on its own is which artifact is **decisive**. To make a photo decisive (so it fires the vision fidelity audit), write the `core_conflict` and `required_constraints` so the deciding value is *physically observed* — e.g. "a photo of the motor's IR thermometer reads 58 °C, within range" instead of "trend charts prove the temperature is within range". Wording the deciding evidence physically is enough; the per-round resolver then grounds that round's decisive value in the photo. If a seed's text instead pins the deciding value to a chart/log, any photo stays `corroborating`.

#### Optional `evidence` field:
For **explicit, deterministic** control (rarely needed given the automatic behavior above), add an `evidence` list. Each item names one artifact the scenario hinges on and its real-world form:

```json
"evidence": [
  {
    "id": "beacon_photo",
    "summary": "Wide shot of cell 3 with the safety beacon lit red.",
    "real_world_source": "physical_photo",
    "role": "corroborating"
  }
]
```

*   `real_world_source`: one of `dashboard_chart` | `in_tool_diagram` | `document` | `log` | `physical_photo` | `hand_sketch`. The planner maps this to an asset kind (`physical_photo`→`natural_image`, `dashboard_chart`→`chart`, `log`/`document`→`rich_text`, `in_tool_diagram`→`diagram`, `hand_sketch`→`sketch`).
*   `role` is **independent of modality** — any kind (`chart`, `rich_text`, `natural_image`, …) can be `decisive` (carries the deciding token/state the verdict rests on) or `corroborating` (supports it; redundant context). **Most natural images are corroborating** — a photo is not automatically the key evidence, just as a chart often carries redundant/context data. A `decisive` photo is fully supported: the deciding token/state must be **legibly visible in the pixels** (runtime strips caption/OCR text, so the agent must read it visually) and is verified by a vision fidelity audit; if it cannot be rendered legibly the asset auto-downgrades to `corroborating`.

#### Why it matters:
Seeds that pre-label all evidence as charts/diagrams yield benchmarks with no real photographs, under-testing genuine visual grounding. Describing authentic forms (or setting `evidence`) lets the generator produce the natural mix a real incident would have — including, correctly, none when the domain is purely digital.

---

## 3. Best Practices Checklist

When writing a new seed, check that:
- [ ] Relative terminology (e.g. `initial phase`, `middle phases`) is used for cascading milestones instead of specific round indexes.
- [ ] At least one `required_constraints` rule requires the Coordinator to read a file, run a tool, or perform a `file edit` directly.
- [ ] The `allowed_tool_families` includes `file` if the Coordinator is expected to decrypt or edit config files directly.
- [ ] All friction is explained technically (e.g. outdated configuration run logs, cache misalignment).
- [ ] `modalities` are correctly aligned with the `generation_policy` (e.g., if you have metric charts, `chart_provider` should be set to `template`).
- [ ] Each piece of evidence is described by its **real-world form** (Principle D): physical evidence is not flattened into a chart/diagram, and any optional `evidence` entries set a sensible `real_world_source`.
- [ ] The seed JSON file is validated against the schema by running mock scenario generation.
