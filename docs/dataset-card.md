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
to 46.3 GB. That corpus is here.

You do not need any of this to install the package, run its test suite, or generate a fresh
corpus of your own. You need it to recompute the published numbers, or to use the 600-round
benchmark as a benchmark.

## Getting it

Use the fetcher in the code repository rather than cloning this dataset wholesale — it places
each component at the one path `analysis/corpus.py` resolves against, and it lets you take the
8.5 GB you need instead of the 46.3 GB you don't.

```bash
git clone https://github.com/google-research/CLASHarness.git && cd CLASHarness
pip install -e . huggingface_hub

python tools/download_data.py --list                                   # the manifest below
python tools/download_data.py --repo yuxianglai117/CLASHarness         # the default set
python tools/download_data.py --repo yuxianglai117/CLASHarness --all

python tools/checksum_corpus.py --root data --verify CHECKSUMS.txt
```

Everything lands under `data/` at the repository root; override with `CLASHARNESS_DATA_ROOT`.

Run the last command. A truncated component does not announce itself — `analysis/` will happily
compute a number over whatever files arrived. `CHECKSUMS.txt` in the code repository carries one
SHA-256 rollup per component; verification only checks the components you actually downloaded.

## Components

| Component | Size | What it is |
| --- | --- | --- |
| `eval-v8` ★ | 2.8 GB | Primary 20-step panel: 14 models (10 API effort-swept + 4 open-weight at high), plus the main-paper figures (the paper additionally reports `GPT-6 Astra` at 40 steps as the 15th model). |
| `tasks-v8` ★ | 5.7 GB | Compiled task specs for all 600 episodes of the primary corpus (generator `gemini-3.6-flash`). |
| `splits-v8` ★ | 58 KB | Episode index files (`datasets/total.txt` listing all 600 evaluated episodes, plus optional subset files). Required by `examples/train_and_eval.py --split total`. |
| `eval-sonnet5` | 1.9 GB | Generator ablation, arm B. |
| `tasks-sonnet5` | 7.1 GB | Task specs for arm B (generator `claude-sonnet-5`). |
| `splits-sonnet5` | 58 KB | Episode index files (`datasets/total.txt`) for arm B. |
| `eval-budget30` | 440 MB | Step-budget ablation at 30 steps. |
| `eval-budget40` | 577 MB | Step-budget ablation at 40 steps. |
| `eval-blind` | 219 MB | Text-only arm: the same rounds with the images withheld. |
| `scenarios-v8` | 12.3 GB | Materialized scenario packages incl. rendered visual assets. |
| `scenarios-sonnet5` | 15.3 GB | Materialized scenario packages for arm B. |

★ = fetched by `--default`. 46.3 GB for all eleven, 26,853 files. Sizes are measured on this
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
  datasets/total.txt                    # splits-v8 (600-episode index for --split total)
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
3 scenario versions per seed, 10 rounds per version. All 600 rounds are evaluated directly
(`--split total`), stratified by reference trajectory length into Easy (236 rounds, ≤ 9 steps),
Mid (224 rounds, 10–11 steps), and Hard (140 rounds, ≥ 12 steps).

The 20-step panel in `eval-v8` covers 14 models: ten API endpoints exposing a reasoning-effort
knob (swept across `low`/`medium`/`high`) and four open-weight models run at `high` only. The
paper additionally reports `GPT-6 Astra` (evaluated at `high` effort under a 40-step action
budget) as the 15th frontier model. The paper's primary effort is `high`, so an effort-sweep
statement is over the 10 API models and a full high-effort frontier statement is over 14 (or
15 including `GPT-6 Astra`).

**Eighteen of the 600 rounds were unwinnable when these runs were scored.** A materializer
visibility defect graded them on a value that appeared in nothing the solver was shown; across
18 rounds × 10 API models × 3 efforts = 540 round-runs there is exactly one pass. The paper
reports the full 600 with those rows retained, because dropping them rescales every model by
the same 582/600 and changes no ordering and no test. The defect is fixed and the corpus
repaired in place — only these pre-repair eval rows carry it. `analysis/dead_rounds_exclusion.json`
identifies them so either view can be computed.

## Known issues

The paper's numbers are computed on the corpus exactly as published here. The issues below were
found after submission; none of them changed a file in this dataset.

**Checks a fair grader would not count.** A check is fair if the instruction asks for it, or if it
can reasonably be inferred from the materials the player is shown. It is not fair if nothing asks
for it, if the materials say someone else does it, or if its criterion asks for a value that
conflicts with the authoritative material. A post-submission audit labelled all 1,194 required
message criteria (Claude Opus 5.5, three votes each, with quoted evidence) and all 515 required
protocol checks (Claude Opus 5.5 and, independently, Gemini 3.1 Pro) against that standard. The
table re-grades the recorded high-effort episodes of 13 of the 14 panel models (DeepSeek V4 Flash
was not re-scored) with the flagged checks demoted to diagnostic:

| Checks demoted | Checks | Claude Opus 5 (%) | Kendall τ-b vs. published | Opus 5 vs. Opus 4.8, p |
| --- | --- | --- | --- | --- |
| None (as published) | 0 | 25.2 | 1.00 | 0.120 |
| Message criteria whose required value conflicts with the authoritative material | 200 | 27.2 | 0.99 | 0.057 |
| Protocol checks both judges call unasked or contradicted | 106 | 29.5 | 0.91 | 0.005 |
| Both of the above | 306 | 31.7 | 0.94 | 0.002 |
| Protocol checks Opus 5.5 calls unasked, contradicted or ambiguous, plus message criteria it calls unsupported or unasked | 485 | 36.3 | 0.88 | 0.004 |
| *Upper bound, not an estimate:* the previous row plus message criteria it calls over-specified or ambiguous | 775 | 42.7 | 0.68 | 0.031 |

p is an exact McNemar test on the rounds where the two models disagree. An over-specified
criterion grades a message the round does ask for, but requires more elements than the request
implies, which is why the last row is only a bound.

- In every row the top three (Claude Opus 5, GPT-5.6 Sol, Claude Opus 4.8) and the bottom three
  (GLM-5.2 FP8, Gemini 3.1 Pro, Gemini 3.5 Flash) of the 13 keep their places, and Claude Opus 5
  against GPT-5.6 Sol is never significant (p ≥ 0.14).
- The seven models between them are within 3.3 points of one another as published, and change
  order among themselves once protocol checks are demoted.
- Claude Opus 5's lead over Claude Opus 4.8 is not significant as published and is significant in
  every row that demotes protocol checks.

`analysis/v8_check_fairness_audit.json` in the code repository lists the 775 checks by task id
and index into `ground_truth.evaluation_checks`, with their labels, which rows count them, and
every model's pass rate in each row.

**Checks the reference solution fails (20 of 3,760).** The reference solutions of all 600
rounds were replayed twice with the v8 judge (`gemini-3.6-flash`). No deterministic check failed.
20 semantic checks failed in both replays: 13 `A written report states the decisive finding: <V>`
criteria whose slot holds an identifier, timestamp or passcode rather than a finding, so a report
that labels the value correctly fails, and 7 message criteria that require more than the
reference message says. They touch 18 rounds, and removing them changes the result of one
Claude Opus 5 round (25.2% → 25.3%). The replay outputs were not kept, so these checks are not
listed individually.

**Files that later rounds seed with stale content (50 of 600 rounds).** Each round is seeded from
its scenario's files, earlier updates and files earlier subagents created, not from what earlier
rounds' reference solutions wrote. In 50 rounds the reference solution reads a file that an
earlier round's reference solution had changed, and the round seeds the earlier content. Grading
uses the round as seeded, and the reference replay above found no failure caused by this.

**Arm B** (`tasks-sonnet5`, `eval-sonnet5`) is published as it was run. The same replay found
59 of its 600 rounds with at least one required check the reference solution fails: 65 semantic
checks (failed in both replays) and 9 deterministic ones. In 8 of those rounds none of the 24
arm-B runs passes the failing check: 7 `sla_support_response` rounds bind a timed message check
to a human stakeholder as sender while the reference delegates to a subagent, and in
`expiring_resource_hold_v03` r06 the graded booking extension is never emitted. Counting every
round that fails only on these checks as a pass raises each model's high-effort arm-B pass rate
by 0.2–0.8 points and changes no ordering. `analysis/sonnet5_gold_replay_audit.json` lists the
checks.

**Claude Opus 5 episodes that end in an empty reply (8 of 1,800).** In 3 high-, 3 medium- and 2
low-effort episodes, all in `payment_chargeback_dispute`, Claude Opus 5 returned no text and the
harness recorded `"stop_reason": "generation_stall_empty_reply"`, `"error_type":
"provider_failure"` and an empty `error`. Most are probably safety refusals of the dispute
scenario: the harness that ran v8 did not keep the provider's stop reason, and a later version
that does saw refusals on these seeds. The records here cannot confirm it. They are scored as
failures like any other episode, so the published rates include them; at high effort they are 3
of 600 rounds, at most 0.5 points. The ablation trees have 12 more (`eval-blind`,
`eval-budget30`, `eval-budget40`) and each arm-B Claude Opus 5 run 9–13, some of them in
`multimodal_temporal_conflict_01`. No other Claude model has any.

**The generator has changed since v8.** The generator in the code repository includes fixes made
after v8 was generated, such as the one behind the eighteen unwinnable rounds above. A corpus you
generate with it is not v8, and its difficulty has not been calibrated against v8.

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
