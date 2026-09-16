# Reproducing Paper Results and Analysis

This directory contains the analysis and reproduction scripts that generate every table, figure, and empirical metric reported in the paper:
**"CLASHarness: Generating Controllable and Verifiable Multi-Agent Conflict Scenarios in Computer Use"**.

---

## Overview of Analysis Scripts

| Script | Purpose & Output |
| :--- | :--- |
| `difficulty_bands.py` | Computes the main results breakdown across Easy / Mid / Hard difficulty bands |
| `table1_v8.py` | Full-corpus pass rate, mean score, and step-cap hit rate across reasoning effort levels |
| `ablation.py` | Step-budget sweep (30 vs. 40 steps) and multimodal vs. text-only (blind) ablations |
| `generator_ablation.py` | Cross-generator comparison (Gemini 3.6 Flash vs. Claude Sonnet 5 generator arms) |
| `mcnemar_v8.py` | Paired statistical significance tests across models and conditions |
| `cost.py` | Token expenditure and dollar cost accounting across providers |
| `capability_axes_v8.py` | Performance breakdown across 7 capability dimensions |
| `conflict_gate_v8.py` | Gating efficacy and discrimination audit across conflict plugin types |
| `coverage_v8.py` | Coverage analysis across all registered plugins and environment skills |
| `clustering_v8.py` | Behavioral clustering across model error modes |
| `round_example_facts.py` | Re-derives the worked round-example exhibits (requires `scenarios-v8`) |
| `rejudge/` | Four-phase pipeline to re-judge a corpus under alternative judge models |

---

## Running the Reproduction

### 1. Download the Evaluation Corpus
The analysis scripts read from `data/` at the repository root (or the directory specified by `CLASHARNESS_DATA_ROOT`). To download the official evaluation trees:

```bash
pip install huggingface_hub
python tools/download_data.py --component eval-v8 --component tasks-v8 --component splits-v8
python tools/checksum_corpus.py --root data --verify CHECKSUMS.txt
```

See [docs/dataset-card.md](../docs/dataset-card.md) for full details on each component.

### 2. Execute Analysis Scripts

Run any analysis script directly:

```bash
python analysis/table1_v8.py
python analysis/difficulty_bands.py
python analysis/ablation.py
```

If your downloaded data lives in a custom directory:

```bash
CLASHARNESS_DATA_ROOT=/path/to/data python analysis/table1_v8.py
```

---

## Paper Fact-Checkers

Three scripts verify that every number and figure drawn in the paper LaTeX source matches the computed corpus data exactly:

| Script | Verification Target |
| :--- | :--- |
| `results_facts.py` | Checks every quantitative claim in `main.tex` and `appendix_report.tex` |
| `figure_facts.py` | Checks every data point in the body ablation figures |
| `judge_ablation.py` | Verifies judge-model sensitivity against the tables in `main.tex` |

These scripts additionally require `CLASHARNESS_PAPER_SRC` pointing to the paper repository root:

```bash
CLASHARNESS_DATA_ROOT=/path/to/data \
CLASHARNESS_PAPER_SRC=/path/to/paper/repo \
python analysis/results_facts.py
```

---

## Notes on Reproducibility Variations

When re-evaluating models live on the benchmark, three factors mean rerun numbers may slightly vary from published tables:

1. **API Routing & Model Deployment**: Published runs executed via Vertex AI and Bedrock enterprise endpoints. Direct commercial API endpoints serve identical model families but may route to slightly different deployment revisions.
2. **Snapshot Pinning**: Not all API providers provide static date-pinned model checkpoints; living endpoints undergo periodic behavioral updates.
3. **Reasoning Effort Budgets**: `--max-steps` binds more tightly on high-effort reasoning models. Always compare models under identical step ceilings.
