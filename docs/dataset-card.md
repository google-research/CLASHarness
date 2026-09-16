---
pretty_name: CLASHarness Corpus
license: cc-by-4.0
language:
  - en
size_categories:
  - 10K<n<100K
task_categories:
  - other
tags:
  - agents
  - multi-agent
  - tool-use
  - multimodal
  - evaluation
  - benchmark
---

# CLASHarness Corpus

The evaluation trees and compiled task specifications behind
[CLASHarness](https://github.com/google-research/CLASHarness), a generator and evaluation
harness for multi-agent coordination under conflicting, time-sensitive, multimodal evidence.

The code repository ships the generator, the harness, the 20 scenario seeds and the frozen
report artifacts. It does not ship the corpus those artifacts were computed from, which runs
to 46.4 GB. That corpus is here.

You do not need any of this to install the package, run its test suite, or generate a fresh
corpus of your own. You need it to recompute the published numbers, or to use the 600-round
benchmark as a benchmark.

## Getting it

Use the fetcher in the code repository rather than cloning this dataset wholesale — it places
each component at the one path `analysis/corpus.py` resolves against, and it lets you take the
8.5 GB you need instead of the 46.4 GB you don't.

```bash
git clone https://github.com/google-research/CLASHarness && cd clash-harness
pip install -e . huggingface_hub

python tools/download_data.py --list                     # the manifest below
python tools/download_data.py --repo <hf-user>/<dataset> # the default set
python tools/download_data.py --repo <hf-user>/<dataset> --all

python tools/checksum_corpus.py --root data --verify CHECKSUMS.txt
```

Everything lands under `data/` at the repository root; override with `CLASHARNESS_DATA_ROOT`.

Run the last command. A truncated component does not announce itself — `analysis/` will happily
compute a number over whatever files arrived. `CHECKSUMS.txt` in the code repository carries one
SHA-256 rollup per component; verification only checks the components you actually downloaded.

## Components

| Component | Size | What it is |
| --- | --- | --- |
| `eval-v8` ★ | 2.8 GB | Primary panel: 14 models, the effort sweep, every main-paper figure. |
| `tasks-v8` ★ | 5.7 GB | Compiled task specs for the primary corpus (generator `gemini-3.6-flash`). |
| `splits-v8` ★ | 58 KB | Official train/dev/test split, 420/90/90 of 600 rounds, held out by seed. |
| `eval-sonnet5` | 1.9 GB | Generator ablation, arm B. |
| `tasks-sonnet5` | 7.1 GB | Task specs for arm B (generator `claude-sonnet-5`). |
| `splits-sonnet5` | 58 KB | Split for arm B, same 420/90/90 shape. |
| `eval-budget30` | 463 MB | Step-budget ablation at 30 steps. |
| `eval-budget40` | 577 MB | Step-budget ablation at 40 steps. |
| `eval-blind` | 219 MB | Text-only arm: the same rounds with the images withheld. |
| `scenarios-v8` | 12.3 GB | Materialized scenario packages incl. rendered visual assets. |
| `scenarios-sonnet5` | 15.3 GB | Materialized scenario packages for arm B. |

★ = fetched by `--default`. 46.4 GB for all eleven, 26,897 files. Sizes are measured on this
tree, so they are what you download.

## Layout

```
eval_results/
  data_3_6_flash_v8/                    # eval-v8
    aligned_setting_{low,medium,high}/  # the swept efforts
    aligned_setting_{default,nothink,...}/
    open_source_high/                   # open-weight panel, high only
      <Vendor>/<model>/<run>/<scenario>.jsonl
  data_sonnet_5/  data_3_6_flash_v8_{blind,budget30,budget40}/
multiagent_conflict_data_v8_Gemini_3_6_flash/
  tasks/<task_id>/specs.json            # tasks-v8
  datasets/{train,dev,test,total}.txt   # splits-v8
multiagent_conflict_data_Gemini_Sonnet_5/
scenarios/conflict_vlm_v8_3.6_flash/    # scenarios-v8
scenarios/conflict_vlm_sonnet_5/
```

Each model directory holds one or more run directories (`r01`, `r02`, ...). `analysis/` reads
`r01` throughout, with one recorded exception — `aligned_setting_high/Gemini/3_7_flash` reads
`r02` — declared as `RUN_OVERRIDE` in `analysis/corpus.py`. A consumer walking the tree by hand
should honor that or it will read a superseded run for one cell.

## Records

**Episodes** (`eval_results/**/*.jsonl`, one JSON object per line):

| Field | Meaning |
| --- | --- |
| `task_id`, `agent` | Round identifier and the model that ran it. |
| `success`, `score` | Terminal verdict and graded score in `[0, 1]`. |
| `steps`, `duration_s` | Step count and wall-clock time. |
| `trajectory[]` | Per-step `action`, `executed_action`, `stdout`/`stderr`, `exit_code`, `reward`, `current_score`, `score_delta`, `passed_checks`, `failed_checks`, `step_evaluation`. |
| `final_response` | The agent's closing message. |
| `error`, `error_type`, `stop_reason` | Populated on failed episodes; see *Redaction*. |
| `provider_impacted`, `compact_fallback`, `date_anchor_mismatch` | Run-health flags. |
| `metadata` | `domains`, `difficulty`, `scenario_slug`, `round_id`, `round_type`, `generator_id`, `template_id`, `instruction`, `canonical_instruction`, `check_results`. |

**Task specs** (`tasks/<task_id>/specs.json`): `instruction`, `canonical_instruction`,
`instruction_variants`, `visible_constraints`, `hidden_constraints`, `decision_requirements`,
`realism_tags`, `online_requirement`, `provider_dependencies`, `availability_tier`, `domains`,
`difficulty`, `initial_state`, `template_id`, `generator_id`, `weight`, `data`, `ground_truth`.

Agents are scored on the end state they leave behind in the simulated product, not on their
transcripts; `ground_truth` and `check_results` are where that grading lives.

## Design

600 rounds: 20 scenario seeds covering 15 conflict archetypes across 5 top-level categories,
3 scenario versions per seed, 10 rounds per version. Splits hold out whole seeds — 420 train,
90 dev, 90 test — under a fixed hash salt, so no round of an episode leaks across a boundary.

The reported panel is 14 models. Ten of them are API endpoints exposing a reasoning-effort
knob and are the only ones swept across low/medium/high; the four open-weight models are run
at high only. The paper's primary effort is high, so an effort statement is over 10 models and
a high-effort statement is over 14.

**Eighteen of the 600 rounds were unwinnable when these runs were scored.** A materializer
visibility defect graded them on a value that appeared in nothing the solver was shown; across
18 rounds × 10 API models × 3 efforts = 540 round-runs there is exactly one pass. The paper
reports the full 600 with those rows retained, because dropping them rescales every model by
the same 582/600 and changes no ordering and no test. The defect is fixed and the corpus
repaired in place — only these pre-repair eval rows carry it. `analysis/dead_rounds_exclusion.json`
identifies them so either view can be computed.

## Redaction

These runs executed on internal infrastructure, and some episodes failed with runner or
transport errors whose messages embedded cluster addresses, workspace paths, service targets and
an operator account name. Those messages were captured verbatim into the record and have been
replaced with placeholders — you will see `<cluster-address-redacted>` and similar.

An audit of all 893,951 records in every published component found each such identifier in
exactly one place: the `error` string of episodes already marked `"success": false, "score": 0.0`
with an empty `final_response`. 524 episodes have a non-empty `error`, in six distinct shapes. No
successful episode carried an identifier; neither did `trajectory`, `input`, `metadata` or any
score field.

Two of those six shapes — 31 episodes — are transport errors whose body is tens of kilobytes of
serialised internal stack detail. Patching that in place would leave kilobytes of placeholders,
so the body is dropped and the leading sentence kept:

```
"error": "504 DEADLINE_EXCEEDED. <internal-diagnostic-trimmed>"
"error": "499 CANCELLED. <internal-diagnostic-trimmed>"
```

The other four shapes keep their text. Rate limits (485 episodes), context-length errors (6)
and one timeout carry nothing internal at all; the rate-limit message has its numeric quota id
replaced (`input-tpm:<quota-id-redacted>`), and one connection abort has its workspace path
replaced (`<workspace-path-redacted>`). In each case what the error says about the failure
survives.

**No reported number depends on any of this.** `analysis/` reads `bool(row["error"])` and the
separate `error_type` field, never the message text, and every affected episode was already
recorded as a failure. `tools/scrub_corpus.py --check` re-establishes that on any tree,
including this one.

**Not published.** Superseded eval trees, run provenance, staging scratch, the per-run `logs/`
directories, and the `minimax_m3` arm (not part of the reported panel). Those carry far more
operator detail than the published trees, and redacting and shipping them would have been
strictly worse than not shipping them.

Also dropped: `visualize.html`, one per scenario package and per eval run directory — 200 files
and 9.0 GB, a fifth of what the tree would otherwise weigh. Each is a self-contained browser
view of the JSON published beside it, nothing in the code repository reads it, and the scenario
version loads Tailwind, Mermaid and svg-pan-zoom from public CDNs, so it does not render
offline. The generator for it is not part of the release, so this one is not reconstructible
from what is here.


## Provenance and limitations

Every scenario, document, chart and conflict here is **synthetic** — generated by
`gemini-3.6-flash` (primary) or `claude-sonnet-5` (arm B) and materialized by the pipeline in
the code repository. No real user data, no real company data, no scraped content. Names,
hostnames, email addresses, incident numbers and dashboards are invented. Any resemblance to a
real system is coincidental.

That is also the main limitation. The corpus measures coordination under conflicting evidence in
a simulated CLI product; it is not a sample of production agent traffic, and its difficulty
distribution reflects the generator's priors. The generator-ablation arm exists precisely so the
size of that effect can be measured rather than assumed.

## License and citation

Released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The CLASHarness code
is separately licensed under Apache-2.0. Content was produced by third-party language models;
your use of it remains subject to those providers' terms.

```bibtex
@software{clasharness,
  title  = {CLASHarness: multi-agent coordination under conflicting multimodal evidence},
  author = {Lai, Yuxiang and Xia, Peng and Ji, Haonian and Xiong, Kaiwen and
            Zeng, Kaide and Liu, Jiaqi and Wu, Fang and Zhong, Jike and
            Zheng, Zeyu and Xie, Cihang and Yao, Huaxiu},
  url    = {https://github.com/google-research/CLASHarness},
  license = {Apache-2.0}
}
```
