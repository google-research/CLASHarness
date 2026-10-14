---
pretty_name: "CLASHarness: Benchmarking Multi-Agent Conflict in Computer Use"
license: cc-by-4.0
language:
  - en
size_categories:
  - 10K<n<100K
task_categories:
  - image-text-to-text
  - text-generation
  - other
tags:
  - agents
  - multi-agent
  - computer-use
  - vlm
  - multimodal
  - conflict-resolution
  - tool-use
  - evaluation
  - benchmark
---

<div align="center">

# CLASHarness Corpus
### Benchmarking Multi-Agent Conflict in Computer Use via a Controllable and Verifiable Generation Pipeline

[![GitHub Code](https://img.shields.io/badge/GitHub-google--research%2FCLASHarness-181717.svg?style=flat-square&logo=github&logoColor=white)](https://github.com/google-research/CLASHarness)
[![Dataset License: CC BY 4.0](https://img.shields.io/badge/Dataset_License-CC_BY_4.0-EF9421.svg?style=flat-square)](https://creativecommons.org/licenses/by/4.0/)
[![Code License: Apache 2.0](https://img.shields.io/badge/Code_License-Apache_2.0-2BA02D.svg?style=flat-square)](https://github.com/google-research/CLASHarness/blob/main/LICENSE)
[![Preprint: arXiv](https://img.shields.io/badge/arXiv-Preprint-B31B1B.svg?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org)
[![Google Research](https://img.shields.io/badge/Google-Research-4285F4.svg?style=flat-square&logo=google&logoColor=white)](https://research.google)

<p align="center">
  <b>The official 600-episode multi-agent conflict evaluation benchmark and recorded 15-model trajectory corpus for <a href="https://github.com/google-research/CLASHarness">CLASHarness</a>.</b>
</p>

<p align="center">
  <a href="https://github.com/google-research/CLASHarness">💻 GitHub Repository</a> &nbsp;•&nbsp;
  <a href="#overview">🚀 Overview</a> &nbsp;•&nbsp;
  <a href="#quick-start">⚡ Quick Start & Evaluation</a> &nbsp;•&nbsp;
  <a href="#leaderboard">🏆 Leaderboard</a> &nbsp;•&nbsp;
  <a href="#dataset-structure">📦 Dataset Structure & Schema</a> &nbsp;•&nbsp;
  <a href="#conflict-dimensions">⚖️ Conflict Dimensions</a> &nbsp;•&nbsp;
  <a href="#citation">📜 Citation</a>
</p>

---

</div>

## <a id="overview"></a>🚀 Dataset Overview & Highlights

This repository hosts the compiled task specifications (`tasks-v8`), full-benchmark episode index (`splits-v8`), and recorded frontier model evaluation trajectories (`eval-v8`) for [**CLASHarness**](https://github.com/google-research/CLASHarness) — a controllable generation and evaluation harness for multi-agent coordination under conflicting, time-sensitive, multimodal evidence.

> 🔗 **Code, Generation Pipeline & Evaluation Harness**: Visit the official GitHub repository at [**`github.com/google-research/CLASHarness`**](https://github.com/google-research/CLASHarness) for the 8-stage scenario generator (`S1`–`S8`), interactive multi-agent CLI runtime, 20 conflict seed recipes, and analysis suite.

- **Verifiable by Construction ($v^\star$ + Live Sandbox Execution)** ◈ Every benchmark episode pre-commits an atomic ground-truth decision value ($v^\star$), executes a reference solver trajectory inside a stateful multi-agent CLI sandbox, and compiles dual-track grading rubrics ($\Phi = \Phi_{\text{det}} \cup \Phi_{\text{sem}}$) directly from recorded side effects.
- **Full 600-Episode Evaluation (`--split total`)** ◈ Built from 20 conflict seeds across 15 conflict archetypes (3 multi-round scenario versions per seed × 10 rounds per version = **600 episodes**). All 600 episodes are evaluated directly without a train/test split, stratified by reference trajectory length into **Easy (236)**, **Mid (224)**, and **Hard (140)**.
- **Multimodal & Temporal Conflict Grounding** ◈ Tests whether agents cross-examine conflicting peer claims against rendered UI screenshots, dashboards, invoices, and real on-disk state rather than blindly trusting verbal messages.
- **15-Model Frontier Trajectory Corpus** ◈ Includes full step-by-step execution transcripts, tool calls, environment mutations, and check-level grades across frontier proprietary APIs (GPT-6 Astra, Claude Opus 5, GPT-5.6 Sol, Claude Opus 4.8, Gemini 3.7 Flash) and open-weight models (Kimi K3, DeepSeek V4 Pro, GLM-5.2 FP8).

---

## <a id="quick-start"></a>⚡ Quick Start: Downloading & Running the Benchmark

Use the downloader in the [CLASHarness GitHub repository](https://github.com/google-research/CLASHarness) to fetch the components directly into the `data/` directory structure expected by the evaluation harness and analysis scripts.

### 1. Clone the Code Repository & Download the Dataset

```bash
git clone https://github.com/google-research/CLASHarness.git
cd CLASHarness

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]" huggingface_hub

# Inspect available dataset components
python tools/download_data.py --list

# Download the default release set (tasks-v8 + splits-v8 + eval-v8, 8.5 GB) into ./data/
python tools/download_data.py --repo yuxianglai117/CLASHarness

# Verify SHA-256 integrity against CHECKSUMS.txt
python tools/checksum_corpus.py --root data --verify CHECKSUMS.txt
```

### 2. Evaluate an Agent on All 600 Benchmark Episodes

If you only need to evaluate your own model on the 600 benchmark episodes, you only need `tasks-v8` and `splits-v8` (`5.7 GB`):

```bash
# Fetch the 600 compiled task specs and episode index
python tools/download_data.py --repo yuxianglai117/CLASHarness --component tasks-v8 --component splits-v8

# Run evaluation across all 600 episodes (--split total)
export GEMINI_API_KEY="your-api-key"

python examples/train_and_eval.py \
  --agent llm \
  --llm-provider gemini --model gemini-3.6-flash \
  --llm-judge-provider gemini --llm-judge-model gemini-3.6-flash \
  --task-data-dir data/multiagent_conflict_data_v8_Gemini_3_6_flash \
  --split total \
  --mode multi \
  --max-steps 20 \
  --parallel-workers 10 \
  --collect-data \
  --out-dir /tmp/clasharness-eval
```

### 3. Reproduce Paper Tables & Figures from Recorded Trajectories (`eval-v8`)

```bash
python analysis/table1_v8.py
python analysis/difficulty_bands.py
python analysis/mcnemar_v8.py
python analysis/cost.py
```

---

## <a id="leaderboard"></a>🏆 Benchmark Leaderboard (600 Episodes, High Effort)

<p align="center">
  <img src="../fig/fig-standings.svg" alt="Model Standings Across Difficulty Bands" width="92%">
</p>

<table>
<thead>
  <tr>
    <th rowspan="2" align="left">Model</th>
    <th rowspan="2" align="center">All Pass<br>(%)</th>
    <th colspan="3" align="center">Pass Rate by Difficulty (%)</th>
    <th rowspan="2" align="center">Mean Score<br>(%)</th>
    <th rowspan="2" align="center">Avg. Cost<br>($ / Tokens)</th>
  </tr>
  <tr>
    <th align="center">Easy</th>
    <th align="center">Mid</th>
    <th align="center">Hard</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="left"><b>GPT-6&nbsp;Astra</b></td>
    <td align="center"><b>28.7</b><sub>±1.9</sub></td>
    <td align="center"><b>35.2</b><sub>±3.1</sub></td>
    <td align="center"><b>28.1</b><sub>±3.1</sub></td>
    <td align="center"><b>18.6</b><sub>±3.3</sub></td>
    <td align="center"><b>66.8</b><sub>±1.2</sub></td>
    <td align="center">$3,003&nbsp;(663M)</td>
  </tr>
  <tr>
    <td align="left"><b>Claude&nbsp;Opus&nbsp;5</b></td>
    <td align="center"><b>25.2</b><sub>±1.8</sub></td>
    <td align="center">32.2<sub>±3.0</sub></td>
    <td align="center">25.0<sub>±2.9</sub></td>
    <td align="center">13.6<sub>±2.9</sub></td>
    <td align="center">64.2<sub>±1.2</sub></td>
    <td align="center">$2,631&nbsp;(511M)</td>
  </tr>
  <tr>
    <td align="left"><b>GPT-5.6&nbsp;Sol</b></td>
    <td align="center"><b>24.5</b><sub>±1.7</sub></td>
    <td align="center">31.4<sub>±3.0</sub></td>
    <td align="center">24.1<sub>±2.9</sub></td>
    <td align="center">13.6<sub>±2.9</sub></td>
    <td align="center">63.3<sub>±1.2</sub></td>
    <td align="center">$2,165&nbsp;(454M)</td>
  </tr>
  <tr>
    <td align="left"><b>Claude&nbsp;Opus&nbsp;4.8</b></td>
    <td align="center"><b>22.8</b><sub>±1.7</sub></td>
    <td align="center">30.1<sub>±3.0</sub></td>
    <td align="center">23.7<sub>±2.8</sub></td>
    <td align="center">9.3<sub>±2.4</sub></td>
    <td align="center">62.1<sub>±1.2</sub></td>
    <td align="center">$2,933&nbsp;(571M)</td>
  </tr>
  <tr>
    <td align="left"><b>Kimi&nbsp;K3</b></td>
    <td align="center"><b>21.8</b><sub>±1.7</sub></td>
    <td align="center">26.3<sub>±2.8</sub></td>
    <td align="center">23.2<sub>±2.9</sub></td>
    <td align="center">12.1<sub>±2.7</sub></td>
    <td align="center">55.9<sub>±1.4</sub></td>
    <td align="center">$2,173&nbsp;(706M)</td>
  </tr>
  <tr>
    <td align="left"><b>Claude&nbsp;Sonnet&nbsp;5</b></td>
    <td align="center"><b>20.8</b><sub>±1.7</sub></td>
    <td align="center">28.0<sub>±2.9</sub></td>
    <td align="center">21.4<sub>±2.7</sub></td>
    <td align="center">7.9<sub>±2.3</sub></td>
    <td align="center">59.6<sub>±1.3</sub></td>
    <td align="center">$1,031&nbsp;(497M)</td>
  </tr>
  <tr>
    <td align="left"><b>Gemini&nbsp;3.7&nbsp;Flash</b></td>
    <td align="center"><b>20.3</b><sub>±1.6</sub></td>
    <td align="center">26.7<sub>±2.9</sub></td>
    <td align="center">21.9<sub>±2.8</sub></td>
    <td align="center">7.1<sub>±2.2</sub></td>
    <td align="center">58.1<sub>±1.3</sub></td>
    <td align="center">$307&nbsp;(395M)</td>
  </tr>
  <tr>
    <td align="left"><b>GPT-5.6&nbsp;Terra</b></td>
    <td align="center"><b>20.0</b><sub>±1.6</sub></td>
    <td align="center">27.1<sub>±2.9</sub></td>
    <td align="center">20.1<sub>±2.7</sub></td>
    <td align="center">7.9<sub>±2.3</sub></td>
    <td align="center">60.2<sub>±1.2</sub></td>
    <td align="center">$731&nbsp;(382M)</td>
  </tr>
  <tr>
    <td align="left"><b>GPT-5.6&nbsp;Luna</b></td>
    <td align="center"><b>19.8</b><sub>±1.6</sub></td>
    <td align="center">23.7<sub>±2.7</sub></td>
    <td align="center">22.3<sub>±2.8</sub></td>
    <td align="center">9.3<sub>±2.5</sub></td>
    <td align="center">60.4<sub>±1.2</sub></td>
    <td align="center">$86&nbsp;(450M)</td>
  </tr>
  <tr>
    <td align="left"><b>DeepSeek&nbsp;V4&nbsp;Pro</b><sup>†</sup></td>
    <td align="center"><b>19.8</b><sub>±1.6</sub></td>
    <td align="center">24.2<sub>±2.8</sub></td>
    <td align="center">20.5<sub>±2.7</sub></td>
    <td align="center">11.4<sub>±2.7</sub></td>
    <td align="center">58.1<sub>±1.3</sub></td>
    <td align="center">$32&nbsp;(70M)</td>
  </tr>
  <tr>
    <td align="left"><b>Gemini&nbsp;3.6&nbsp;Flash</b></td>
    <td align="center"><b>18.5</b><sub>±1.6</sub></td>
    <td align="center">24.6<sub>±2.8</sub></td>
    <td align="center">21.4<sub>±2.7</sub></td>
    <td align="center">3.6<sub>±1.6</sub></td>
    <td align="center">58.6<sub>±1.3</sub></td>
    <td align="center">$565&nbsp;(363M)</td>
  </tr>
  <tr>
    <td align="left"><b>GLM-5.2&nbsp;FP8</b><sup>†</sup></td>
    <td align="center"><b>14.7</b><sub>±1.4</sub></td>
    <td align="center">18.2<sub>±2.5</sub></td>
    <td align="center">16.5<sub>±2.5</sub></td>
    <td align="center">5.7<sub>±2.0</sub></td>
    <td align="center">48.7<sub>±1.3</sub></td>
    <td align="center">$120&nbsp;(82M)</td>
  </tr>
  <tr>
    <td align="left"><b>Gemini&nbsp;3.1&nbsp;Pro</b></td>
    <td align="center"><b>12.0</b><sub>±1.3</sub></td>
    <td align="center">15.7<sub>±2.4</sub></td>
    <td align="center">12.1<sub>±2.2</sub></td>
    <td align="center">5.7<sub>±2.0</sub></td>
    <td align="center">48.7<sub>±1.3</sub></td>
    <td align="center">$578&nbsp;(280M)</td>
  </tr>
  <tr>
    <td align="left"><b>DeepSeek&nbsp;V4&nbsp;Flash</b><sup>†</sup></td>
    <td align="center"><b>8.5</b><sub>±1.1</sub></td>
    <td align="center">8.5<sub>±1.8</sub></td>
    <td align="center">11.2<sub>±2.1</sub></td>
    <td align="center">4.3<sub>±1.7</sub></td>
    <td align="center">37.5<sub>±1.3</sub></td>
    <td align="center">N/A&nbsp;(95M)</td>
  </tr>
  <tr>
    <td align="left"><b>Gemini&nbsp;3.5&nbsp;Flash</b></td>
    <td align="center"><b>3.0</b><sub>±0.7</sub></td>
    <td align="center">3.8<sub>±1.2</sub></td>
    <td align="center">2.7<sub>±1.1</sub></td>
    <td align="center">2.1<sub>±1.2</sub></td>
    <td align="center">25.9<sub>±1.1</sub></td>
    <td align="center">$712&nbsp;(445M)</td>
  </tr>
</tbody>
</table>

<sub><sup>†</sup>Text-only models evaluated using OCR transcripts of visual assets. Task difficulty is stratified by ground-truth reference trajectory length across all 600 episodes (<i>Easy</i>: 236 rounds, ≤ 9 steps; <i>Mid</i>: 224 rounds, 10–11 steps; <i>Hard</i>: 140 rounds, ≥ 12 steps).</sub>

---

## <a id="dataset-structure"></a>📦 Dataset Components, Layout & Record Schemas

### 1. Component Manifest

| Component | Size | Description |
| :--- | :---: | :--- |
| **`tasks-v8`** ★ | **5.7 GB** | Compiled task specifications (`specs.json`) for all 600 episodes of the primary benchmark (generated with `gemini-3.6-flash`). Required to run evaluation via `examples/train_and_eval.py`. |
| **`splits-v8`** ★ | **58 KB** | Episode index files (`datasets/total.txt` listing all 600 evaluated episodes). Required by `examples/train_and_eval.py --split total`. |
| **`eval-v8`** ★ | **2.8 GB** | Primary 20-step evaluation panel: 14 models (10 API models swept across `low`/`medium`/`high` reasoning effort + 4 open-weight models at `high`). |
| `eval-sonnet5` | 1.9 GB | Cross-generator ablation trajectories (Arm B, generated by `claude-sonnet-5`). |
| `tasks-sonnet5` | 7.1 GB | Compiled task specs for Arm B (`claude-sonnet-5`). |
| `splits-sonnet5` | 58 KB | Episode index files (`datasets/total.txt`) for Arm B. |
| `eval-budget30` | 440 MB | Step-budget ablation trajectories at 30 steps. |
| `eval-budget40` | 577 MB | Step-budget ablation trajectories at 40 steps. |
| `eval-blind` | 219 MB | Modality ablation trajectories (text-only arm with visual assets withheld). |
| `scenarios-v8` | 12.3 GB | Raw materialized scenario packages for the primary corpus. |
| `scenarios-sonnet5` | 15.3 GB | Raw materialized scenario packages for Arm B. |

> **★ = Default Release Set (`8.5 GB`)**: Fetched by default via `python tools/download_data.py --repo yuxianglai117/CLASHarness`. Sufficient to run the full 600-episode benchmark and reproduce all primary paper tables.

### 2. Directory Layout

```text
multiagent_conflict_data_v8_Gemini_3_6_flash/
  tasks/<task_id>/specs.json            # tasks-v8 (600 compiled task specifications + embedded base64 images)
  datasets/total.txt                    # splits-v8 (600-episode index read by --split total)
eval_results/
  data_3_6_flash_v8/                    # eval-v8 (recorded model evaluation trajectories)
    aligned_setting_{low,medium,high}/  # 10 API models across reasoning efforts
    open_source_high/                   # 4 open-weight models at high effort
      <Vendor>/<model>/<run>/<scenario>.jsonl
```

Each model directory holds run directories (`r01`, `r02`, ...). `analysis/` reads `r01` throughout, with one recorded exception — `aligned_setting_high/Gemini/3_7_flash` reads `r02` — declared as `RUN_OVERRIDE` in [`analysis/corpus.py`](https://github.com/google-research/CLASHarness/blob/main/analysis/corpus.py).

### 3. Record Schemas

#### Task Specifications (`multiagent_conflict_data_v8_Gemini_3_6_flash/tasks/<task_id>/specs.json`)

| Field | Description |
| :--- | :--- |
| `instruction`, `canonical_instruction` | Natural-language coordinator prompt and canonical task specification |
| `initial_state` | Initial workspace files, communication channel history, active locks/approvals, and base64-encoded visual assets (`__bytes_b64__`) |
| `visible_constraints`, `hidden_constraints` | Explicit and latent operational constraints governing conflict resolution |
| `decision_requirements` | Pre-committed ground-truth decision requirements ($v^\star$) and authoritative source provenance |
| `ground_truth` | Reference solver trajectory (`reference_solution`) and compiled dual-track evaluation checks (`evaluation_checks`: deterministic state checks $\Phi_{\text{det}}$ + semantic LLM-judge checks $\Phi_{\text{sem}}$) |
| `domains`, `difficulty`, `template_id` | Domain tags, difficulty band (`easy` / `medium` / `hard`), and seed archetype metadata |

#### Evaluation Trajectories (`eval_results/**/*.jsonl`, one JSON object per episode)

| Field | Description |
| :--- | :--- |
| `task_id`, `agent` | Episode identifier (`<scenario>_v0X_r0Y`) and evaluated model identifier |
| `success`, `score` | All-or-nothing episode pass verdict (`bool`) and fraction of satisfied checks in `[0, 1]` |
| `steps`, `duration_s` | Total CLI actions executed and wall-clock duration in seconds |
| `trajectory[]` | Step-by-step trace: `action`, `executed_action`, `stdout`, `stderr`, `exit_code`, `reward`, `current_score`, `passed_checks`, `failed_checks` |
| `final_response` | The agent's closing summary message |
| `metadata` | Includes `difficulty`, `scenario_slug`, `round_id`, `instruction`, and per-check grading breakdown (`check_results`) |

### 4. Benchmark Specifications

| Dimension | Specification | Description |
| :--- | :---: | :--- |
| **Conflict Seeds** | **20** | Diverse domains across system admin, finance, cloud deployment, healthcare, and IAM |
| **Scenario Packages** | **60** | 3 multi-round scenario packages generated per seed |
| **Evaluation Episodes** | **600** | 10 evaluation rounds per package (evaluated in full via `--split total`) |
| **Difficulty Bands** | **236 / 224 / 140** | Easy (≤ 9 steps) / Mid (10–11 steps) / Hard (≥ 12 steps) across all 600 episodes |
| **Conflict Plugins** | **23** | Covering challenges, coordination protocols, and subagent interactions |
| **State Protocols** | **7** | `approval`, `booking`, `containment`, `deploy`, `lock`, `pairing`, `review` |
| **Execution Modes** | **5** | `multi` (default), `mock`, `martin`, `real`, `hybrid` |
| **Evaluated Models** | **15** | Frontier proprietary APIs + open-weight vision-language models |

---

## <a id="conflict-dimensions"></a>⚖️ The Three Conflict Dimensions & Generation Pipeline

<p align="center">
  <img src="../fig/fig-conflict-dimensions.svg" alt="The Three Conflict Dimensions in CLASHarness" width="100%">
</p>

| Dimension | Operational Failure Mode | Coordinator Requirement |
| :--- | :--- | :--- |
| **◈ Temporal Staleness** | A subagent claims task completion at `T = 0`, but subsequent telemetry or logs show a regression. | Cross-check real on-disk state and prefer recent authoritative ground truth over stale verbal claims. |
| **◈ Cross-Modal Discrepancy** | A subagent's verbal text report directly contradicts a rendered chart, status dashboard, or UI screenshot. | Visually inspect and cross-examine text assertions against rendered visual evidence before acting. |
| **◈ Actionable Delay** | Operations require waiting for asynchronous delays to elapse and verifying physical file mutations. | Handle asynchronous latency and verify physical state mutations rather than prematurely declaring completion. |

<p align="center">
  <img src="../fig/figure1-structured-v11.svg" alt="CLASHarness Hierarchical Generation Pipeline" width="95%">
</p>

For full architectural details on the 3-tier, 8-stage (`S1`–`S8`) verifiable generation pipeline, see [`Multi-Agent_GENERATION_PIPELINE.md`](https://github.com/google-research/CLASHarness/blob/main/Multi-Agent_GENERATION_PIPELINE.md) in the GitHub repository.

---

## <a id="known-issues"></a>🔍 Provenance, Audit Notes & Known Issues

Every scenario, document, chart, and conflict in this corpus is **100% synthetic** — generated by `gemini-3.6-flash` (primary v8 corpus) or `claude-sonnet-5` (Arm B) and materialized by the pipeline in the [CLASHarness GitHub repository](https://github.com/google-research/CLASHarness). No real user data, company data, or scraped content is included.

<details>
<summary><b>Post-Submission Audit Notes, Known Issues & Diagnostic Redaction Details (Click to Expand)</b></summary>
<br>

The paper's numbers are computed on the corpus exactly as published here. The items below were audited after submission; none of them altered any file in this dataset, and supplementary audit manifests are shipped under [`analysis/`](https://github.com/google-research/CLASHarness/tree/main/analysis) in the GitHub repository so any view can be recomputed.

### 1. Eighteen Unwinnable Pre-Repair Rounds
Eighteen of the 600 rounds were unwinnable when these runs were scored due to a materializer visibility defect that graded them on a value absent from solver-visible materials (across 18 rounds × 10 API models × 3 efforts = 540 round-runs, there is 1 pass). The paper reports the full 600 with those rows retained because dropping them rescales every model by the same `582/600` factor and changes no ranking or statistical test. The defect is fixed in the generator code, and [`analysis/dead_rounds_exclusion.json`](https://github.com/google-research/CLASHarness/blob/main/analysis/dead_rounds_exclusion.json) lists the 18 rounds.

### 2. Post-Submission Check Fairness Audit
A check is fair if the instruction asks for it, or if it can reasonably be inferred from the materials the player is shown. A post-submission audit labelled all 1,194 required message criteria (Claude Opus 5.5, three votes each, with quoted evidence) and all 515 required protocol checks (Claude Opus 5.5 and, independently, Gemini 3.1 Pro). Re-grading the recorded high-effort episodes of 13 of the 14 panel models with flagged checks demoted to diagnostic yields:

| Checks demoted | Checks | Claude Opus 5 (%) | Kendall τ-b vs. published | Opus 5 vs. Opus 4.8, p |
| --- | --- | --- | --- | --- |
| None (as published) | 0 | 25.2 | 1.00 | 0.120 |
| Message criteria whose required value conflicts with the authoritative material | 200 | 27.2 | 0.99 | 0.057 |
| Protocol checks both judges call unasked or contradicted | 106 | 29.5 | 0.91 | 0.005 |
| Both of the above | 306 | 31.7 | 0.94 | 0.002 |
| Protocol checks Opus 5.5 calls unasked, contradicted or ambiguous, plus message criteria it calls unsupported or unasked | 485 | 36.3 | 0.88 | 0.004 |
| *Upper bound, not an estimate:* the previous row plus message criteria it calls over-specified or ambiguous | 775 | 42.7 | 0.68 | 0.031 |

In every row the top three (`Claude Opus 5`, `GPT-5.6 Sol`, `Claude Opus 4.8`) and bottom three (`GLM-5.2 FP8`, `Gemini 3.1 Pro`, `Gemini 3.5 Flash`) of the 13 keep their places. [`analysis/v8_check_fairness_audit.json`](https://github.com/google-research/CLASHarness/blob/main/analysis/v8_check_fairness_audit.json) lists all 775 flagged checks.

### 3. Reference Replay & Seeding Notes
- **Checks the reference solution fails (20 of 3,760)**: Replaying all 600 reference solutions twice with the v8 judge (`gemini-3.6-flash`) passed 100% of deterministic checks and flagged 20 semantic checks across 18 rounds (removing them changes one Claude Opus 5 round: `25.2% -> 25.3%`).
- **Stale cross-round file seeding (50 of 600 rounds)**: Each round seeds from scenario snapshots rather than earlier rounds' reference mutations; reference replay found no grading failures caused by this.
- **Arm B (`tasks-sonnet5`, `eval-sonnet5`)**: [`analysis/sonnet5_gold_replay_audit.json`](https://github.com/google-research/CLASHarness/blob/main/analysis/sonnet5_gold_replay_audit.json) lists 74 required checks across 59 rounds flagged by reference replay.
- **Empty replies on Claude Opus 5 (8 of 1,800 episodes)**: Recorded with `"stop_reason": "generation_stall_empty_reply"` and `"error_type": "provider_failure"` in `payment_chargeback_dispute` (at most 0.5 points at high effort).

### 4. Diagnostic Error Redaction
Failed episodes that encountered infrastructure transport errors had internal cluster addresses, workspace paths, or numeric quota IDs in their `.error` strings replaced with placeholders (`<cluster-address-redacted>`, `<internal-diagnostic-trimmed>`, `input-tpm:<quota-id-redacted>`). No successful episode, trajectory, input, or score field was touched, and no reported metric depends on `.error` text.

</details>

---

## <a id="citation"></a>📜 License & Citation

- **Dataset License**: Released under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
- **Code & Harness License**: Released under [Apache License 2.0](https://github.com/google-research/CLASHarness/blob/main/LICENSE) at [`github.com/google-research/CLASHarness`](https://github.com/google-research/CLASHarness).

If you use the CLASHarness dataset or codebase in your research, please cite:

```bibtex
@article{lai2026clasharness,
  title   = {CLASHarness: Benchmarking Multi-Agent Conflict in Computer Use via a Controllable and Verifiable Generation Pipeline},
  author  = {Lai, Yuxiang and Zhong, Jike and Liu, Chengzhi and Li, Ming and
             Wang, Haonan and Yang, Xiaofeng and Zhao, Liang and Tang, Youbao and
             Fan, Ruixi},
  journal = {arXiv preprint},
  year    = {2026},
  url     = {https://github.com/google-research/CLASHarness}
}
```
