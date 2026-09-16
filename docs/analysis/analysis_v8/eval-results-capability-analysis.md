# Multi-Agent VLM Conflict Eval (v8 corpus): Model Capability & Failure-Mode Analysis

This document analyzes the **evaluation results** of the `multiagent_vlm_conflict`
benchmark on the v8 corpus — how well each model performs **as the agent under
test**, and *how* it fails. It is a **results** analysis, not a generator analysis.

> [!NOTE]
> **Data provenance.** The benchmark scenarios — task specs, ground-truth solution
> commands, and VLM assets — were generated with **gemini-3.6-flash**
> (`generator_id = conflict_vlm_v1`). **Nine** models (three Gemini, three Claude,
> three OpenAI) are then scored as solvers against this shared scenario set at
> **three reasoning-effort levels each**. All numbers below are recomputed from
> `eval_results/data_3_6_flash_v8/aligned_setting_{low,medium,high}/<Vendor>/<model>/r01/`
> — **9 × 3 × 600 = 16,200 episodes**. Unless a table says otherwise, figures are
> at `reasoning_effort=medium`. Regenerate the underlying digest with:
>
> ```
> python tools/failure_taxonomy.py                 # all 27 cells -> failure_taxonomy.json
> python tools/failure_taxonomy.py --effort medium # one cell, with a text summary
> ```

> [!NOTE]
> **Scope boundary.** Every model here is the *solver* being scored. This document
> covers **measurement validity**, the **failure taxonomy**, and **scenario-level
> difficulty**. Behavioral comparison — action mixes, command surfaces, clustering,
> the effort sweep, curated round walk-throughs — lives in the cross-model report
> and is **cross-referenced, not repeated**.

> [!NOTE]
> **`gpt-5.5` is not in this corpus.** The `OpenAI/5_5` directories exist under
> `aligned_setting_{medium,high}` but contain no interaction logs. It is reported
> nowhere in this document, and there are **nine** per-model deep-dives, not ten.

> [!TIP]
> **Companion documents.** This overview is the index for ten deep-dives:
> - **Per-model solver failure-mode taxonomies** — one per scored model, each
>   expanding its §4 profile below:
>   - *Gemini*: [3.5-flash](./gemini-3-5-flash-solver-failures.md) ·
>     [3.6-flash](./gemini-3-6-flash-solver-failures.md) ·
>     [3.1-pro](./gemini-3-1-pro-solver-failures.md)
>   - *Claude*: [opus-4.8](./claude-opus-4-8-solver-failures.md) ·
>     [opus-5](./claude-opus-5-solver-failures.md) ·
>     [sonnet-5](./claude-sonnet-5-solver-failures.md)
>   - *OpenAI*: [sol](./gpt-5-6-sol-solver-failures.md) ·
>     [luna](./gpt-5-6-luna-solver-failures.md) ·
>     [terra](./gpt-5-6-terra-solver-failures.md)
> - **Cross-model behavioral comparison** (action mix, command surface, grounding ×
>   action map, failure clustering, difficulty axes, the three-level effort sweep,
>   eight curated rounds):
>   [`cross-model-behavioral-comparison.md`](./cross-model-behavioral-comparison.md)
>   + interactive report
>   [`cross-model-behavioral-comparison.html`](./cross-model-behavioral-comparison.html).

> [!TIP]
> **Section map.** §1 ladder · §2 measurement validity (**the rubric is 28 checks
> wide; seven episodes have no rubric at all**) · §3 the **F1–F6 failure taxonomy**
> and which buckets actually discriminate · §4 per-model profiles (index) ·
> §5 difficulty as a property of the scenario · §6 interpretation · §7 caveats ·
> **Appendix A** full 60 × 9 per-scenario table.

---

## 1. TL;DR

- **Capability ladder** (600 tasks, nine models, `reasoning_effort=medium`):
  **claude-opus-5 leads on both metrics — 25.3% pass, avg 0.633** — but its lead
  is narrow. **gpt-5.6-sol (23.0%, 0.628)** and **claude-opus-4.8 (23.8%, 0.618)**
  are within a point of it on score, and head-to-head over 60 scenarios
  **opus-5 and sol are dead even at 30–30** (§5.2). **gemini-3.6-flash (20.2%,
  0.602)** leads a middle band with **gpt-5.6-terra (20.0%, 0.589)**,
  **gpt-5.6-luna (17.7%, 0.591)** and **claude-sonnet-5 (17.0%, 0.570)**.
  **gemini-3.1-pro (9.8%, 0.479)** and **gemini-3.5-flash (3.2%, 0.299)** are far
  below the field.
- **The corpus is hard by construction and the difficulty target is met.** The
  best model passes one task in four. This is the design goal, not a defect —
  see the cross-model report for the difficulty axes that produce it (§8 there).
- **A task fails in one of six ways, and only four of them are about the model.**
  The taxonomy (§3) splits every required check into F1–F6. **F5 (required
  terminal action) spreads just 9.3 points across all nine models** — the worst
  model in the field is only 8.7 points behind the best on it — and **F6 (acting
  inside a time window) is 80–100% failure for everyone**. These two buckets
  measure the corpus. F1/F2/F4 are where models separate.
- **Reasoning effort buys exactly one thing: looking at the image.** Across the
  field, required grounding failures (F3) fall **−48.9%** from `low` to `high`
  (693 → 354), monotonically for **8 of 9** models and net-negative for **9 of 9**.
  Every other bucket is flat or worse: F1 **−0.7%**, F5 **−5.2%**, F6 **−2.0%**,
  F2 **+18.0%**, F4 **+15.4%** (§3.4).
- **gemini-3.5-flash does not fail by reasoning badly — it fails by not producing
  anything to grade.** It leaves **390 of 600 tasks with zero writes**, step-caps
  on **510**, of which **483 are single-verb loops**, and **50.2% of its semantic
  checks never reached the judge** because there was no transcript to evaluate
  (§2.4). It is also the only model that gets *worse* with more effort.
- **The ladder is not a total order.** `it_deployment_gate_v02` is passed by
  **3.5-flash 4 times out of 10 and by opus-5 zero times** — those four passes are
  **21% of 3.5-flash's entire 19-task haul** (§5.2).
- **Three scenarios were solved by nobody**, in 90 attempts each:
  `warehouse_robotics_safety_halt_v03`, `distributed_lock_shared_resource_v02`,
  `multi_tab_slot_monitoring_v02` (§5.1).
- **opus-5 at medium is the only model in the field with zero step-caps and zero
  loops** — an empty loop histogram across all 600 tasks.

### Capability ladder

| Model | Provider | pass@budget | avg_score | per-check | near_miss | zero_score | capped | zero-write |
|---|---|---|---|---|---|---|---|---|
| **claude-opus-5** | `vertex-anthropic` | **25.3% (152/600)** | **0.6328** | 61.7% | 264 | **21** | **0** | 3 |
| claude-opus-4.8 | `vertex-anthropic` | 23.8% (143/600) | 0.6180 | 60.8% | 262 | 28 | 17 | 6 |
| gpt-5.6-sol | `bedrock-gpt` | 23.0% (138/600) | 0.6276 | **61.8%** | 278 | **20** | 12 | **0** |
| gemini-3.6-flash | `gemini` | 20.2% (121/600) | 0.6023 | 59.3% | 273 | 29 | 87 | 36 |
| gpt-5.6-terra | `bedrock-gpt` | 20.0% (120/600) | 0.5891 | 57.9% | 267 | 35 | 8 | 1 |
| gpt-5.6-luna | `bedrock-gpt` | 17.7% (106/600) | 0.5909 | 58.3% | **293** | 34 | 16 | 2 |
| claude-sonnet-5 | `vertex-anthropic` | 17.0% (102/600) | 0.5701 | 55.6% | 269 | 41 | 6 | 2 |
| gemini-3.1-pro | `gemini` | 9.8% (59/600) | 0.4793 | 48.2% | 250 | 62 | 61 | 31 |
| gemini-3.5-flash | `gemini` | 3.2% (19/600) | 0.2990 | 32.0% | 128 | 145 | 510 | 390 |

> All nine rows are at `reasoning_effort=medium`, and — unlike the earlier `think`
> corpus — **every** row carries a real effort knob that was requested *and*
> confirmed effective (§2.1).
>
> **The top three are not separated by this run.** opus-5 leads sol by 0.0052
> avg-score and 2.3 pass points, and leads opus-4.8 by 0.0148 avg-score and 1.5
> pass points. But **the two metrics disagree about second place**: sol is ahead
> of opus-4.8 on avg-score (+0.0096) and behind it on pass rate (−0.8 points). On
> per-check accuracy **sol is ahead of opus-5** (61.8% vs 61.7%); opus-5 converts
> a per-check tie into a pass-rate lead purely through failure clustering
> (cross-model report, §7.1). Treat the top three as one band.
>
> **`near_miss` rank is a counting artifact for gemini-3.5-flash.** Its 128
> near-misses look like the best figure in the table only because a near-miss is
> drawn from the *failed* pool and 3.5-flash fails so completely that most of its
> tasks never get close enough to count. Read it against `zero_score`, where
> 3.5-flash's 145 is worse than the rest of the field combined would suggest.
>
> **The Gemini spread is the cleanest capability signal in the run.** 3.6-flash
> (0.6023) to 3.5-flash (0.2990) is a 0.3033 avg-score range inside one vendor on
> one serving stack — wider than the entire spread of the other six models
> (0.5701–0.6328). Unlike every cross-vendor comparison here, it carries no
> serving-stack confound (§7).

---

## 2. Measurement validity

### 2.1 What "aligned" means in v8 — and why it is stronger than the reference run

Three facts, each verified from the run headers rather than assumed:

1. **All nine models expose `reasoning_effort`.** Every one of the **1,620** run
   headers (9 models × 3 efforts × 60 scenarios) records
   `reasoning_effort == reasoning_effort_effective`, split exactly 60/60/60. In
   the earlier `think` corpus the three Gemini models exposed no such knob, and
   "aligned" meant only "aligned harness". Here it means aligned *inference
   budget request*, confirmed effective.
2. **Config drift is zero.** `history_mode=full`, `tool_mode=native`,
   `tool_schema_mode=native`, `native_tool_choice=auto`,
   `multimodal_input_mode=both`, `max_image_edge=2576`, and a 20-step budget hold
   across all 1,620 headers. The three Gemini models additionally run at
   `gemini_media_resolution=ultra_high`.
3. **The scenario set is literally shared.** All nine models see the same 60
   scenarios × 10 rounds, so every comparison in §5 is a within-scenario paired
   comparison, never a pooled one.

What alignment does **not** cover: the three vendors are served through three
different stacks (`gemini`, `vertex-anthropic`, `bedrock-gpt`). A requested
`reasoning_effort=medium` is not guaranteed to mean the same number of thinking
tokens on each — and measurably does not (cross-model report §9.2, where per-task
reasoning tokens at medium range from 184 for opus-5 to 8,800 for 3.5-flash).
**Effort is aligned as a request, not as a quantity.**

### 2.2 The rubric: 28 check names, one role split, no third category

Enumerated directly from the check records, the v8 rubric uses exactly **28
distinct check names** built from **four** condition types:

| condition | distinct names | checks | role |
|---|---:|---:|---|
| `any_field_equals` | 23 | 515 | all `required` |
| `field_semantic_match` | 2 | 2,137 | all `required` |
| `field_contains` | 2 | 1,078 | 602 `required`, 476 `diagnostic` |
| `within_window` | 1 | 30 | all `required` |
| **total** | **28** | **3,760** | **3,284 required / 476 diagnostic** |

Two consequences worth stating plainly:

- **The scored units are exactly the `required` checks.** All 476 non-required
  checks in the corpus are `assets_accessed` diagnostics; there is no third role
  and no unlabelled check. `score = satisfied required units / 3,284`.
- **The 23 `any_field_equals` names are long-tailed.** The largest is
  `effect:deployment_gate_set` at 82 checks; `effect:containment_action` has 58,
  `effect:deployment_requested` 39, and ten names have fewer than 15. Any
  per-name claim about a small side-effect check is a claim about a dozen
  observations.

### 2.3 The judge and self-consistency

The 2,137 `field_semantic_match` checks are decided by an LLM judge, and the
message string records the verdict *and* the self-consistency vote when one was
taken — e.g.
`LLM semantic match: The report references transaction ID TXN-APP-WRITE-9912 as
the decisive finding instead of TXN-APP-WRITE-9908. [self-consistency 0/3 passed]`.
The vote suffix is not present on every record. Because semantic checks are
**65.1% of all required units** (2,137 / 3,284), the judge is the single largest
source of grading variance in this corpus, and every F1/F2 number below inherits
it. This is a limitation of the measurement, not a finding about the models.

Separately, **`ok=False` never occurs**: across all 27 cells there is not one
check record where the evaluator itself errored. Grading either ran or was
fail-closed (§2.4).

### 2.4 Fail-closed empty transcripts — and how they rewrite gemini-3.5-flash

When a model produces no content to grade, the evaluator does **not** call the
judge; it fails the check closed and records
`LLM semantic match: no 'message' content in messages_sent to evaluate (empty
transcript)`. The precise indicator is `judge_invoked == False`.

| model | semantic checks | fail-closed | % |
|---|---:|---:|---:|
| gemini-3.5-flash | 2,132 | **1,070** | **50.2** |
| gemini-3.6-flash | 2,137 | 98 | 4.6 |
| gemini-3.1-pro | 2,135 | 65 | 3.0 |
| claude-opus-4.8 | 2,137 | 24 | 1.1 |
| claude-opus-5 | 2,137 | 20 | 0.9 |
| claude-sonnet-5 | 2,137 | 18 | 0.8 |
| gpt-5.6-sol | 2,137 | **14** | **0.7** |
| gpt-5.6-luna | 2,137 | 22 | 1.0 |
| gpt-5.6-terra | 2,137 | 16 | 0.7 |

**Half of gemini-3.5-flash's semantic checks were never judged at all: 1,070 of
2,132, or 50.2%.** Its reported semantic pass rate is therefore not a statement
about its reasoning; it is dominated by absence of output. Any comparison that
treats 3.5-flash's semantic score as a reasoning measurement is wrong. The right
reading is the participation one: 390 zero-write tasks, 510 step-caps, 483 of
them loops.

### 2.5 Seven episodes have no rubric at all

Two distinct integrity populations exist in the grid, and they do **not** overlap:

| population | field | count | effect |
|---|---|---:|---|
| provider-impacted but graded | `provider_impacted` | **14** | episode graded normally |
| hard-failed before step 1 | `error` (504 `DEADLINE_EXCEEDED`) | **7** | **no `check_results` at all** |

The 14 are opus-5 8, terra 3, 3.5-flash 2, sol 1 (cross-model report §11.4). The
7 are **gemini-3.1-pro 6 and gemini-3.5-flash 1**, every one of them
`steps=0`, `score=0.0`, and every one of them missing its check records entirely.
The two sets have **zero intersection**, and `error` is exactly equivalent to
"no `check_results`" (7 = 7, matching per model).

This has a small, *directional* consequence that must be stated rather than
buried. The affected cells lose required units from the per-check denominator:

| cell | required units | vs. 3,284 |
|---|---:|---:|
| medium / gemini-3.5-flash | 3,278 | −6 |
| medium / gemini-3.1-pro | 3,278 | −6 |
| high / gemini-3.1-pro | 3,257 | −27 |

Because these episodes took **zero steps**, the natural counterfactual is that
essentially all of their checks would have failed. Excluding them therefore
**flatters** the per-check rates of the two weakest models. The magnitude is
bounded and tiny: re-adding all six medium units as failures moves gemini-3.5-flash
from 32.03% to 31.97% and gemini-3.1-pro from 48.17% to 48.08% — **under 0.1
points in both cases**. Pass rates are unaffected: the episode is still present
and still counted as a failure out of 600.

---

## 3. How a task fails: the F1–F6 taxonomy

### 3.1 The six buckets

Every one of the 3,284 required units maps to exactly one bucket. The mapping is
mechanical (`tools/failure_taxonomy.py`), and the tool carries a guard bucket
`other` that must stay empty — it does, in all 27 cells.

| bucket | what it asserts | check name(s) | required | diagnostic |
|---|---|---|---:|---:|
| **F1** content_msg | the *message you sent* says the right thing | `effect:messages_sent field_semantic_match` | 1,194 | 0 |
| **F2** content_file | the *file you wrote* says the right thing | `effect:files_created field_semantic_match` | 943 | 0 |
| **F3** grounding | you opened the asset you needed | `effect:assets_accessed field_contains` | 346 | 476 |
| **F4** anchor | a literal identifier appears in the file | `effect:files_created field_contains` | 256 | 0 |
| **F5** side_effect | the required terminal action happened | 23 × `effect:<x> any_field_equals` | 515 | 0 |
| **F6** window | you acted inside the required time window | `effect:messages_sent within_window` | 30 | 0 |

This is a strict refinement of the check-type split in the cross-model report
(§7.2): semantic 2,137 = F1 1,194 + F2 943; `field_contains` 1,078 = F3 822 +
F4 256; the remaining 545 = F5 515 + F6 30.

> [!WARNING]
> **A porting hazard, recorded because it silently produces a wrong taxonomy.**
> The reference-corpus version of this tool identified side-effect checks with
> `name.endswith(" exists")`. **The v8 corpus contains no check named `... exists`.**
> Applying the old predicate here drops all 23 side-effect names into the `other`
> bucket and reports F5 as empty. The v8 tool matches on `any_field_equals` and
> treats a non-empty `other` bucket as a hard error (`exit 1`).

### 3.2 Per-bucket required failure rates

Failure counts (required / diagnostic) at medium:

| model | F1 | F2 | F3 | F4 | F5 | F6 | total |
|---|---|---|---|---|---|---|---:|
| gemini-3.5-flash | 894/0 | 714/0 | 46/93 | 193/0 | 353/0 | 28/0 | 2,321 |
| gemini-3.6-flash | 656/0 | 239/0 | 35/83 | 64/0 | 316/0 | 27/0 | 1,420 |
| gemini-3.1-pro | 845/0 | 369/0 | 54/121 | 87/0 | 319/0 | 25/0 | 1,820 |
| claude-opus-4.8 | 648/0 | 202/0 | 48/100 | 55/0 | 308/0 | 27/0 | 1,388 |
| claude-opus-5 | 597/0 | 224/0 | 39/85 | 48/0 | 325/0 | 24/0 | 1,342 |
| claude-sonnet-5 | 641/0 | 247/0 | **122/219** | 63/0 | 356/0 | 30/0 | 1,678 |
| gpt-5.6-sol | 630/0 | **191/0** | **31/71** | 66/0 | 311/0 | 27/0 | **1,327** |
| gpt-5.6-luna | 669/0 | 217/0 | 50/121 | 68/0 | 337/0 | 27/0 | 1,489 |
| gpt-5.6-terra | 658/0 | 222/0 | 61/141 | 63/0 | 350/0 | 28/0 | 1,523 |

As required-failure *rates* (failed / required units in that bucket) — the two
Gemini cells with missing episodes use their own slightly smaller denominators
per §2.5:

| model | F1 | F2 | F3 | F4 | F5 | F6 |
|---|---:|---:|---:|---:|---:|---:|
| gemini-3.5-flash | 75.1 | 75.9 | 13.3 | 75.4 | 68.5 | 93.3 |
| gemini-3.6-flash | 54.9 | 25.3 | 10.1 | 25.0 | 61.4 | 90.0 |
| gemini-3.1-pro | 70.8 | 39.2 | 15.7 | 34.1 | 62.2 | 83.3 |
| claude-opus-4.8 | 54.3 | 21.4 | 13.9 | 21.5 | **59.8** | 90.0 |
| claude-opus-5 | **50.0** | 23.8 | 11.3 | **18.8** | 63.1 | **80.0** |
| claude-sonnet-5 | 53.7 | 26.2 | **35.3** | 24.6 | 69.1 | **100.0** |
| gpt-5.6-sol | 52.8 | **20.3** | **9.0** | 25.8 | 60.4 | 90.0 |
| gpt-5.6-luna | 56.0 | 23.0 | 14.5 | 26.6 | 65.4 | 90.0 |
| gpt-5.6-terra | 55.1 | 23.5 | 17.6 | 24.6 | 68.0 | 93.3 |

Note **F1 is at or above 50% for every model in the field** — the floor is
claude-opus-5's exactly 50.0%. Writing the right thing in a message is the single
hardest thing this corpus asks for, and it is also its largest bucket (1,194 of
3,284 required units, 36.4%).

### 3.3 Two buckets measure the corpus, not the model

Rank-correlating each bucket's failure rate against pass rate across the nine
models (Spearman ρ; more negative = the bucket tracks overall capability better)
gives:

| bucket | ρ (n=9) | ρ (n=8, excl. 3.5-flash) | spread, n=9 | spread, n=8 |
|---|---:|---:|---:|---:|
| F1 content_msg | −0.80 | −0.71 | 25.1 | 20.8 |
| F4 anchor | −0.79 | −0.69 | 56.6 | 15.4 |
| F2 content_file | −0.72 | −0.60 | 55.6 | 18.9 |
| F3 grounding | −0.50 | −0.69 | 26.3 | 26.3 |
| **F5 side_effect** | **−0.63** | **−0.52** | **9.3** | **9.3** |
| **F6 window** | **−0.48** | **−0.34** | 20.0 | 20.0 |

**F5 is the flattest bucket in the taxonomy, and its flatness survives removing
the outlier.** Its 9.3-point spread is the narrowest of the six and does not
shrink at all when gemini-3.5-flash is dropped — because 3.5-flash is not an
outlier on F5. F3 and F6 are spread-invariant too, but for unrelated reasons:
3.5-flash sits in the interior of F3 (13.3%, between gpt-5.6-sol's 9.0% and
claude-sonnet-5's 35.3%), so dropping it cannot move the range; and F6 is
saturated. The three buckets whose spreads shrink on removal — F1, F2, F4 — are
exactly the ones where 3.5-flash holds the extreme. Sorted best-to-worst on F5,
the ordering barely resembles the capability ladder:

| F5 rank | model | F5 fail % | overall rank |
|---:|---|---:|---:|
| 1 | claude-opus-4.8 | 59.8 | 2 |
| 2 | gpt-5.6-sol | 60.4 | 3 |
| 3 | gemini-3.6-flash | 61.4 | 4 |
| 4 | **gemini-3.1-pro** | 62.2 | **8** |
| 5 | **claude-opus-5** | 63.1 | **1** |
| 6 | gpt-5.6-luna | 65.4 | 6 |
| 7 | gpt-5.6-terra | 68.0 | 5 |
| 8 | **gemini-3.5-flash** | 68.5 | **9** |
| 9 | claude-sonnet-5 | 69.1 | 7 |

The overall winner is **fifth** on F5, behind the model ranked eighth overall;
and the model ranked last overall is **eighth, not ninth**. Emitting the required
terminal action is something every model does about 60–69% of the time
regardless of how good it otherwise is.

**F6 is near-total failure for everyone.** The best result in the field is
claude-opus-5's 24 of 30 failed — six passes. claude-sonnet-5 fails all 30. On
n=30 the 80–100% range is a difference of six checks, so the ordering is noise.
What is *not* noise is the level: **acting inside a required time window is
essentially unsolved on this corpus.**

> [!IMPORTANT]
> **F6 is hard, not unwinnable.** Six of the 30 window checks were passed
> (claude-opus-5, medium). A bucket that no model ever passed would be a rubric
> bug; this one is a difficulty finding. It is still worth flagging as a candidate
> for review, because a check that 8 of 9 models fail 90–100% of the time
> contributes almost no discrimination for the 30 units of scoring weight it
> consumes.

### 3.4 What reasoning effort moves, and what it does not

Field-wide required-failure totals across the three effort levels (sum over all
nine models):

| bucket | low | medium | high | low→high | excl. 3.5-flash | strictly monotone ↓ |
|---|---:|---:|---:|---:|---:|---:|
| **F3 grounding** | 693 | 486 | **354** | **−48.9%** | **−49.1%** | **8/9** |
| F5 side_effect | 3,069 | 2,975 | 2,910 | −5.2% | −7.2% | 5/9 |
| F6 window | 245 | 243 | 240 | −2.0% | −2.3% | 0/9 |
| F1 content_msg | 6,240 | 6,238 | 6,196 | −0.7% | −4.8% | 7/9 |
| F4 anchor | 629 | 707 | 726 | +15.4% | +1.8% | 2/9 |
| F2 content_file | 2,331 | 2,625 | 2,751 | +18.0% | +4.0% | 3/9 |

**Effort buys grounding and nothing else.** F3 nearly halves, and the effect is
not an artifact of one model: it is net-negative for **all nine**
(claude-sonnet-5 −112, claude-opus-5 −42, gemini-3.6-flash −32, gpt-5.6-terra −30,
gemini-3.5-flash −29, claude-opus-4.8 −28, gpt-5.6-sol −27, gpt-5.6-luna −26,
gemini-3.1-pro −13) and strictly monotone for eight, with gemini-3.1-pro the only
non-monotone case (51 → 54 → 38).

The apparent F2/F4 *regressions* are almost entirely gemini-3.5-flash, which gets
worse with more effort in every content bucket (F2 422 → 765, F4 123 → 211).
Removing it collapses both to near-flat (+4.0% and +1.8%). The honest summary is
that for eight of nine models more effort changes content quality **not at all**,
while for gemini-3.5-flash it actively hurts — consistent with the paired McNemar
result of **−10.5 points low→high** for that model (cross-model report §9).

---

## 4. Per-model profiles

Each row indexes a full deep-dive. All figures at medium.

| model | pass | score | signature in one line | deep-dive |
|---|---:|---:|---|---|
| **claude-opus-5** | 25.3% | 0.6328 | Only model in the field with **zero caps and zero loops**; schedules rather than polls (median wait 60s, never under 30s); best F1 (50.0) and best F4 (18.8). | [→](./claude-opus-5-solver-failures.md) |
| **claude-opus-4.8** | 23.8% | 0.6180 | The other non-poller (zero sub-30s waits); **zero zero-open tasks** — it always looks at something; best F5 in the field (59.8). | [→](./claude-opus-4-8-solver-failures.md) |
| **gpt-5.6-sol** | 23.0% | 0.6276 | Highest per-check accuracy (61.8%) and the fewest total failures (1,327); best F2 (20.3) and best F3 (9.0); ties opus-5 head-to-head 30–30. | [→](./gpt-5-6-sol-solver-failures.md) |
| **gemini-3.6-flash** | 20.2% | 0.6023 | Best Gemini by a wide margin and genuinely in the middle band; waits least of anyone (47 waits total); still caps on 87 tasks. | [→](./gemini-3-6-flash-solver-failures.md) |
| **gpt-5.6-terra** | 20.0% | 0.5891 | Fast and shallow — median 9 steps, 2.13 opens/task, 223 of 230 waits under 30s; F3 17.6 is the second-worst grounding in the field. | [→](./gpt-5-6-terra-solver-failures.md) |
| **gpt-5.6-luna** | 17.7% | 0.5909 | Field-high near-miss count (293) with a middling pass rate — it gets close constantly; median wait **3s**, the twitchiest poller here. | [→](./gpt-5-6-luna-solver-failures.md) |
| **claude-sonnet-5** | 17.0% | 0.5701 | **Looks least and pays for it**: 185 zero-open tasks, F3 35.3% (2.0× the next worst), fails all 30 window checks; breaks the Claude wait pattern with sub-30s polls. | [→](./claude-sonnet-5-solver-failures.md) |
| **gemini-3.1-pro** | 9.8% | 0.4793 | Over-investigates and under-delivers: worst F1 (70.8) and F2 (39.2) outside 3.5-flash; escalates **600 → 1200 → 3600 → 7200s** of waiting inside one task; 6 of the corpus's 7 hard 504 failures. | [→](./gemini-3-1-pro-solver-failures.md) |
| **gemini-3.5-flash** | 3.2% | 0.2990 | Does not produce a gradeable deliverable: 390 zero-write tasks, 510 caps (483 loops), **50.2% of semantic checks never judged**; the only model that degrades with effort. | [→](./gemini-3-5-flash-solver-failures.md) |

---

## 5. Difficulty is a property of the scenario

### 5.1 Three scenarios nobody solved

Over 90 attempts each (9 models × 10 rounds), three scenarios produced **zero**
passes:

| scenario | field mean score | passes |
|---|---:|---:|
| `warehouse_robotics_safety_halt_v03` | **0.246** | 0 / 90 |
| `distributed_lock_shared_resource_v02` | 0.312 | 0 / 90 |
| `multi_tab_slot_monitoring_v02` | 0.438 | 0 / 90 |

`warehouse_robotics_safety_halt_v03` is the hardest scenario in the corpus on
both counts, and appears in the zero-pass list of **all nine** models. Six more
scenarios were solved once or twice in 90: `sla_support_response_v02` (1, by
claude-opus-5 alone), `multi_party_access_permit_v01` (2, both claude-opus-5),
`sla_support_response_v01` (2), `coordinated_containment_v01` (2),
`multi_tab_slot_monitoring_v01` (3), `distributed_lock_shared_resource_v03` (3).

At the other end, `payment_chargeback_dispute_v01` is the easiest scenario at
field mean **0.779** and 55/90 passes.

> [!NOTE]
> **The two deliberately-unhardened control arms do not behave alike.**
> `payment_chargeback_dispute` and `clinical_trial_dosage_conflict` were both left
> untreated as controls. The former yields the corpus's easiest scenario
> (`_v01`, 0.779) but also two mid-difficulty ones (`_v02` 0.423, `_v03` 0.533);
> the latter has **all three variants in the hardest five** (0.343 / 0.373 /
> 0.378). "Unhardened" is not a synonym for "easy" — variant-level difficulty
> dominates the control/treatment distinction on this corpus.

Zero-pass scenario counts per model make the same point from the model side:
gemini-3.5-flash scores nothing at all in **47 of 60** scenarios, gemini-3.1-pro
in 23, claude-sonnet-5 in 17, gemini-3.6-flash and gpt-5.6-sol in 13,
gpt-5.6-terra 12, gpt-5.6-luna 11, and **claude-opus-4.8 and claude-opus-5 in
just 5 each**. Full table in Appendix A.

### 5.2 Head-to-head: the ladder is not a total order

Comparing mean score scenario-by-scenario (60 paired comparisons per pair), the
net record against the other eight models:

| model | wins | losses | ties | net |
|---|---:|---:|---:|---:|
| claude-opus-5 | 340 | 132 | 8 | **+208** |
| gpt-5.6-sol | 336 | 135 | 9 | +201 |
| claude-opus-4.8 | 336 | 137 | 7 | +199 |
| gemini-3.6-flash | 284 | 184 | 12 | +100 |
| gpt-5.6-luna | 262 | 207 | 11 | +55 |
| gpt-5.6-terra | 243 | 223 | 14 | +20 |
| claude-sonnet-5 | 223 | 251 | 6 | −28 |
| gemini-3.1-pro | 83 | 396 | 1 | −313 |
| gemini-3.5-flash | 19 | 461 | 0 | −442 |

Two results here matter more than the ordering:

- **claude-opus-5 vs gpt-5.6-sol is 30–30 with zero ties.** The two models split
  the corpus exactly down the middle. Whatever separates them in the aggregate
  table, it is not a consistent per-scenario advantage. (opus-5 vs opus-4.8 is
  35–23 with 2 ties — a real but modest edge.)
- **Even the bottom model wins somewhere.** gemini-3.5-flash takes 19 scenario
  comparisons, including 7 against gemini-3.1-pro.

The sharpest single counterexample to a total order is `it_deployment_gate_v02`
(field mean 0.588, 23/90 passes): **claude-opus-5 passes it 0 times out of 10,
while gemini-3.5-flash passes it 4 times.** Those four passes are **21% of
gemini-3.5-flash's entire 19-task haul**, and this is opus-5's largest negative
gap against the field (−0.14). It is also the only mid-difficulty scenario in
opus-5's five-scenario zero-pass list — the other four are
`warehouse_robotics_safety_halt_v03`, `distributed_lock_shared_resource_v02` and
`multi_tab_slot_monitoring_v02` (the three nobody solved) plus
`clinical_trial_dosage_conflict_v02` (field 0.378). A benchmark summary that
reports only the ladder hides this entirely.

Per-model largest positive gaps against the field mean show the same specificity:
claude-opus-5 `warehouse_robotics_safety_halt_v02` **+0.26**, gpt-5.6-sol
`coordinated_containment_v02` +0.21, gpt-5.6-terra
`payment_chargeback_dispute_v03` +0.20, claude-opus-4.8
`expense_reimbursement_conflict_v02` +0.19, gemini-3.6-flash
`clinical_trial_dosage_conflict_v02` +0.19.

### 5.3 By designed difficulty axis

Everything above treats a scenario as an opaque unit. But the corpus was not
generated blind: each of the 20 seeds in
`clasharness/data/scenario_seeds/conflict_vlm_v1/` declares the difficulty it is
supposed to create — a list of `conflict_taxonomy` tokens, a list of
`difficulty_challenges` tokens, and two boolean knobs (`force_vision_locus`,
`force_async_wait`). The generator concatenates the two token lists and resolves
them against the conflict-plugin registry (`conflict_llm_pipeline.py:1582`); the
plugins that come back are what inject the constraints into the generated
scenario. The 60 scenarios are those 20 seeds x 3 variants, so every task joins
to exactly one seed and the *designed* difficulty of every task is known. This
section asks the question the rest of the document cannot: **did the levers we
built actually produce difficulty?**

**All 23 registered conflict plugins are active in this corpus** — 7 reached
through `difficulty_challenges` and the other 16 through `conflict_taxonomy`,
with no token appearing in both fields. Each seed activates 4 to 7 of them (6
seeds activate 4, 6 activate 5, 7 activate 6, one activates 7). Note what does
*not* participate: `archetype` is declared on 15 seeds but never reaches the
router, so it groups nothing and is not an axis here — see the footnote at the
end of this section for what it used to group and why that was misleading.

Coverage is extremely uneven by construction, and the `seeds` column is the part
of every table below that matters most. `workflow_completion` is carried by 18 of
the 20 seeds, `information_transfer` by 15, `multi_source_reasoning` by 12 —
while **10 of the 23 plugins are carried by a single seed**, i.e. 30 tasks per
model. In a one-seed bucket the plugin's effect cannot be separated from that one
scenario's difficulty, so a large delta there describes a scenario, not a
designed lever. `force_vision_locus` is set on 14 seeds and **`force_async_wait`
on exactly one**, `sla_support_response`.

Buckets on the plugin, category and challenge axes overlap, so those rows do not
sum to 600. Raw bucket rates are not comparable across a field spanning 3.2% to
25.3% pass@budget, so the last table restates every lever as a delta against
*that model's own* 600-task baseline. All tables are generated by
`tools/design_axis_slice.py` from the raw trajectories and re-verified cell by
cell by `tools/audit_report.py` (pass D13); bold marks the row maximum and is
generated, not hand-applied. The 7 `challenge` rows of the plugin table restate
the challenge table exactly — the token and the plugin are the same bucket over
the same tasks — and are kept so the 23-plugin view is complete.
#### by conflict plugin (overlapping; buckets do not sum)

| bucket | category | seeds | tasks | 3.5 flash | 3.6 flash | 3.1 pro | opus 4.8 | opus 5 | sonnet 5 | 5.6-sol | 5.6-luna | 5.6-terra |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| workflow_completion | challenge | 18 | 540 | 3.5% (0.31) | 20.6% (0.62) | 10.4% (0.50) | 24.8% (0.64) | **26.5% (0.65)** | 17.2% (0.58) | 23.9% (0.64) | 18.5% (0.61) | 21.3% (0.61) |
| information_transfer | challenge | 15 | 450 | 3.6% (0.32) | 18.0% (0.60) | 9.6% (0.50) | 22.4% (0.62) | **22.9% (0.62)** | 15.3% (0.57) | 20.0% (0.62) | 16.7% (0.60) | 17.6% (0.59) |
| multi_source_reasoning | challenge | 12 | 360 | 2.8% (0.30) | 19.7% (0.60) | 8.9% (0.47) | 23.3% (0.62) | **26.1% (0.65)** | 17.5% (0.57) | 21.9% (0.63) | 17.2% (0.59) | 18.9% (0.58) |
| state_repair | challenge | 10 | 300 | 2.7% (0.27) | 23.7% (0.60) | 11.3% (0.47) | **25.0% (0.61)** | 24.0% (0.61) | 18.0% (0.55) | 24.3% (0.61) | 17.0% (0.57) | 19.0% (0.56) |
| evidence_mismatch | inconsistency | 8 | 240 | 1.2% (0.20) | 23.8% (0.55) | 9.2% (0.39) | 25.0% (0.58) | **28.3% (0.61)** | 21.2% (0.54) | 27.1% (0.60) | 17.9% (0.54) | 22.5% (0.54) |
| temporal_staleness | inconsistency | 7 | 210 | 1.0% (0.20) | 21.4% (0.56) | 9.0% (0.41) | 24.3% (0.60) | **26.7% (0.61)** | 21.0% (0.56) | **26.7% (0.61)** | 18.6% (0.57) | 22.9% (0.58) |
| dependency_ordering | order | 6 | 180 | 5.0% (0.32) | 22.8% (0.62) | 13.9% (0.52) | 27.8% (0.64) | 26.1% (0.63) | 18.9% (0.59) | **28.3% (0.66)** | 20.6% (0.63) | 26.1% (0.63) |
| time_window_expiration | timeout | 6 | 180 | 1.1% (0.29) | 21.7% (0.62) | 9.4% (0.49) | 24.4% (0.62) | **30.0% (0.68)** | 16.7% (0.58) | 22.8% (0.63) | 16.1% (0.59) | 20.0% (0.60) |
| gap_completion | challenge | 3 | 90 | 2.2% (0.26) | 13.3% (0.50) | 5.6% (0.33) | 17.8% (0.52) | **21.1% (0.56)** | 8.9% (0.50) | 17.8% (0.56) | 15.6% (0.50) | 17.8% (0.53) |
| credential_lock | coordination | 2 | 60 | 0.0% (0.18) | **16.7% (0.44)** | 5.0% (0.27) | 15.0% (0.46) | 15.0% (0.47) | 15.0% (0.46) | 15.0% (0.48) | 10.0% (0.41) | 8.3% (0.42) |
| deadlock_dependency | coordination | 2 | 60 | 0.0% (0.27) | **33.3% (0.62)** | 15.0% (0.46) | 30.0% (0.63) | 31.7% (0.69) | 18.3% (0.58) | 28.3% (0.65) | 21.7% (0.59) | 23.3% (0.58) |
| duplicate_avoidance | challenge | 2 | 60 | 3.3% (0.20) | 28.3% (0.63) | 11.7% (0.43) | 31.7% (0.64) | 41.7% (0.70) | 31.7% (0.64) | **45.0% (0.71)** | 26.7% (0.62) | 41.7% (0.69) |
| vision_decoy | challenge | 2 | 60 | 1.7% (0.27) | 16.7% (0.60) | 3.3% (0.45) | **23.3% (0.64)** | 21.7% (0.61) | 10.0% (0.53) | 18.3% (0.59) | 11.7% (0.57) | 11.7% (0.53) |
| barrier_synchronization | coordination | 1 | 30 | 6.7% (0.39) | 26.7% (0.69) | 20.0% (0.59) | 30.0% (0.67) | 16.7% (0.62) | 26.7% (0.63) | 26.7% (0.69) | 23.3% (0.68) | **33.3% (0.68)** |
| controlled_deployment_gate | order | 1 | 30 | 16.7% (0.47) | 26.7% (0.66) | 20.0% (0.61) | 30.0% (0.63) | 20.0% (0.56) | 16.7% (0.59) | **36.7% (0.68)** | 10.0% (0.60) | 23.3% (0.64) |
| coordinated_containment | subagent_collaboration | 1 | 30 | 0.0% (0.24) | 10.0% (0.57) | 3.3% (0.46) | 16.7% (0.65) | 16.7% (0.60) | 3.3% (0.54) | **20.0% (0.66)** | **20.0% (0.62)** | 10.0% (0.54) |
| distributed_lock | coordination | 1 | 30 | 3.3% (0.26) | 10.0% (0.50) | 0.0% (0.36) | **16.7% (0.52)** | **16.7% (0.54)** | 10.0% (0.46) | 13.3% (0.50) | 6.7% (0.45) | 6.7% (0.39) |
| duplicate_hold_stale | subagent_collaboration | 1 | 30 | 0.0% (0.26) | 0.0% (0.45) | 0.0% (0.40) | **10.0% (0.56)** | 6.7% (0.57) | 3.3% (0.51) | 3.3% (0.52) | 3.3% (0.56) | 3.3% (0.51) |
| multi_device_pairing | timeout | 1 | 30 | 0.0% (0.38) | 16.7% (0.65) | 13.3% (0.54) | 23.3% (0.61) | **26.7% (0.67)** | 16.7% (0.62) | 20.0% (0.60) | 16.7% (0.59) | 16.7% (0.61) |
| multi_party_synchronization | subagent_collaboration | 1 | 30 | 3.3% (0.37) | 20.0% (0.64) | 6.7% (0.57) | 30.0% (0.70) | **33.3% (0.76)** | 13.3% (0.59) | 20.0% (0.68) | 16.7% (0.65) | 16.7% (0.65) |
| non_overlapping_assignment | subagent_collaboration | 1 | 30 | 10.0% (0.40) | 33.3% (0.75) | 16.7% (0.61) | 36.7% (0.68) | **50.0% (0.74)** | 46.7% (0.73) | 33.3% (0.72) | 33.3% (0.71) | 36.7% (0.70) |
| reversible_action_hold | timeout | 1 | 30 | 0.0% (0.30) | 13.3% (0.67) | 3.3% (0.59) | 16.7% (0.68) | **26.7% (0.69)** | 10.0% (0.59) | 16.7% (0.63) | 13.3% (0.62) | 6.7% (0.60) |
| scope_reconciliation | main_supervised | 1 | 30 | 13.3% (0.41) | 13.3% (0.69) | 6.7% (0.56) | 16.7% (0.66) | 16.7% (0.67) | 13.3% (0.59) | 16.7% (0.68) | 16.7% (0.63) | **20.0% (0.68)** |

#### by conflict category (overlapping; buckets do not sum)

| bucket | seeds | tasks | 3.5 flash | 3.6 flash | 3.1 pro | opus 4.8 | opus 5 | sonnet 5 | 5.6-sol | 5.6-luna | 5.6-terra |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| inconsistency | 9 | 270 | 1.1% (0.20) | 21.1% (0.54) | 8.1% (0.39) | 23.3% (0.58) | **25.9% (0.60)** | 19.3% (0.54) | 24.4% (0.59) | 16.3% (0.54) | 20.4% (0.54) |
| coordination | 6 | 180 | 1.7% (0.26) | **22.8% (0.55)** | 10.0% (0.40) | **22.8% (0.56)** | 21.1% (0.58) | 17.2% (0.53) | 21.1% (0.57) | 15.6% (0.52) | 17.2% (0.51) |
| order | 6 | 180 | 5.0% (0.32) | 22.8% (0.62) | 13.9% (0.52) | 27.8% (0.64) | 26.1% (0.63) | 18.9% (0.59) | **28.3% (0.66)** | 20.6% (0.63) | 26.1% (0.63) |
| timeout | 6 | 180 | 1.1% (0.29) | 21.7% (0.62) | 9.4% (0.49) | 24.4% (0.62) | **30.0% (0.68)** | 16.7% (0.58) | 22.8% (0.63) | 16.1% (0.59) | 20.0% (0.60) |
| subagent_collaboration | 4 | 120 | 3.3% (0.32) | 15.8% (0.60) | 6.7% (0.51) | 23.3% (0.65) | **26.7% (0.67)** | 16.7% (0.59) | 19.2% (0.65) | 18.3% (0.63) | 16.7% (0.60) |
| main_supervised | 1 | 30 | 13.3% (0.41) | 13.3% (0.69) | 6.7% (0.56) | 16.7% (0.66) | 16.7% (0.67) | 13.3% (0.59) | 16.7% (0.68) | 16.7% (0.63) | **20.0% (0.68)** |

#### by difficulty knob: force_vision_locus (vision graded)

| bucket | seeds | tasks | 3.5 flash | 3.6 flash | 3.1 pro | opus 4.8 | opus 5 | sonnet 5 | 5.6-sol | 5.6-luna | 5.6-terra |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vision_graded | 14 | 420 | 4.0% (0.35) | 17.6% (0.63) | 9.3% (0.52) | **23.1% (0.64)** | **23.1% (0.64)** | 13.6% (0.58) | 20.0% (0.64) | 16.7% (0.62) | 17.4% (0.61) |
| not_vision_graded | 6 | 180 | 1.1% (0.17) | 26.1% (0.54) | 11.1% (0.38) | 25.6% (0.56) | **30.6% (0.61)** | 25.0% (0.55) | 30.0% (0.60) | 20.0% (0.53) | 26.1% (0.55) |

#### by difficulty knob: force_async_wait (temporal async)

| bucket | seeds | tasks | 3.5 flash | 3.6 flash | 3.1 pro | opus 4.8 | opus 5 | sonnet 5 | 5.6-sol | 5.6-luna | 5.6-terra |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| async_required | 1 | 30 | 0.0% (0.44) | 3.3% (0.55) | 3.3% (0.46) | 10.0% (0.58) | **13.3% (0.61)** | 0.0% (0.51) | 0.0% (0.55) | 6.7% (0.56) | 6.7% (0.54) |
| no_async | 19 | 570 | 3.3% (0.29) | 21.1% (0.60) | 10.2% (0.48) | 24.6% (0.62) | **26.0% (0.63)** | 17.9% (0.57) | 24.2% (0.63) | 18.2% (0.59) | 20.7% (0.59) |

#### by difficulty_challenge (overlapping tokens; buckets do not sum)

| bucket | seeds | tasks | 3.5 flash | 3.6 flash | 3.1 pro | opus 4.8 | opus 5 | sonnet 5 | 5.6-sol | 5.6-luna | 5.6-terra |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| workflow_completion | 18 | 540 | 3.5% (0.31) | 20.6% (0.62) | 10.4% (0.50) | 24.8% (0.64) | **26.5% (0.65)** | 17.2% (0.58) | 23.9% (0.64) | 18.5% (0.61) | 21.3% (0.61) |
| information_transfer | 15 | 450 | 3.6% (0.32) | 18.0% (0.60) | 9.6% (0.50) | 22.4% (0.62) | **22.9% (0.62)** | 15.3% (0.57) | 20.0% (0.62) | 16.7% (0.60) | 17.6% (0.59) |
| multi_source_reasoning | 12 | 360 | 2.8% (0.30) | 19.7% (0.60) | 8.9% (0.47) | 23.3% (0.62) | **26.1% (0.65)** | 17.5% (0.57) | 21.9% (0.63) | 17.2% (0.59) | 18.9% (0.58) |
| state_repair | 10 | 300 | 2.7% (0.27) | 23.7% (0.60) | 11.3% (0.47) | **25.0% (0.61)** | 24.0% (0.61) | 18.0% (0.55) | 24.3% (0.61) | 17.0% (0.57) | 19.0% (0.56) |
| gap_completion | 3 | 90 | 2.2% (0.26) | 13.3% (0.50) | 5.6% (0.33) | 17.8% (0.52) | **21.1% (0.56)** | 8.9% (0.50) | 17.8% (0.56) | 15.6% (0.50) | 17.8% (0.53) |
| duplicate_avoidance | 2 | 60 | 3.3% (0.20) | 28.3% (0.63) | 11.7% (0.43) | 31.7% (0.64) | 41.7% (0.70) | 31.7% (0.64) | **45.0% (0.71)** | 26.7% (0.62) | 41.7% (0.69) |
| vision_decoy | 2 | 60 | 1.7% (0.27) | 16.7% (0.60) | 3.3% (0.45) | **23.3% (0.64)** | 21.7% (0.61) | 10.0% (0.53) | 18.3% (0.59) | 11.7% (0.57) | 11.7% (0.53) |

#### relative to each model's own baseline (pass-point delta)

| lever | seeds | tasks | 3.5 flash | 3.6 flash | 3.1 pro | opus 4.8 | opus 5 | sonnet 5 | 5.6-sol | 5.6-luna | 5.6-terra | below baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| async_required | 1 | 30 | -3.2 | -16.8 | -6.5 | -13.8 | -12.0 | -17.0 | -23.0 | -11.0 | -13.3 | 9/9 |
| no_async | 19 | 570 | +0.2 | +0.9 | +0.3 | +0.7 | +0.6 | +0.9 | +1.2 | +0.6 | +0.7 | 0/9 |
| vision_graded | 14 | 420 | +0.9 | -2.5 | -0.5 | -0.7 | -2.2 | -3.4 | -3.0 | -1.0 | -2.6 | 8/9 |
| not_vision_graded | 6 | 180 | -2.1 | +5.9 | +1.3 | +1.7 | +5.2 | +8.0 | +7.0 | +2.3 | +6.1 | 1/9 |
| workflow_completion | 18 | 540 | +0.4 | +0.4 | +0.5 | +1.0 | +1.1 | +0.2 | +0.9 | +0.9 | +1.3 | 0/9 |
| information_transfer | 15 | 450 | +0.4 | -2.2 | -0.3 | -1.4 | -2.4 | -1.7 | -3.0 | -1.0 | -2.4 | 8/9 |
| multi_source_reasoning | 12 | 360 | -0.4 | -0.4 | -0.9 | -0.5 | +0.8 | +0.5 | -1.1 | -0.4 | -1.1 | 7/9 |
| state_repair | 10 | 300 | -0.5 | +3.5 | +1.5 | +1.2 | -1.3 | +1.0 | +1.3 | -0.7 | -1.0 | 4/9 |
| gap_completion | 3 | 90 | -0.9 | -6.8 | -4.3 | -6.1 | -4.2 | -8.1 | -5.2 | -2.1 | -2.2 | 9/9 |
| duplicate_avoidance | 2 | 60 | +0.2 | +8.2 | +1.8 | +7.8 | +16.3 | +14.7 | +22.0 | +9.0 | +21.7 | 0/9 |
| vision_decoy | 2 | 60 | -1.5 | -3.5 | -6.5 | -0.5 | -3.7 | -7.0 | -4.7 | -6.0 | -8.3 | 9/9 |
| evidence_mismatch | 8 | 240 | -1.9 | +3.6 | -0.7 | +1.2 | +3.0 | +4.2 | +4.1 | +0.2 | +2.5 | 2/9 |
| temporal_staleness | 7 | 210 | -2.2 | +1.3 | -0.8 | +0.5 | +1.3 | +4.0 | +3.7 | +0.9 | +2.9 | 2/9 |
| dependency_ordering | 6 | 180 | +1.8 | +2.6 | +4.1 | +3.9 | +0.8 | +1.9 | +5.3 | +2.9 | +6.1 | 0/9 |
| time_window_expiration | 6 | 180 | -2.1 | +1.5 | -0.4 | +0.6 | +4.7 | -0.3 | -0.2 | -1.6 | +0.0 | 5/9 |
| credential_lock | 2 | 60 | -3.2 | -3.5 | -4.8 | -8.8 | -10.3 | -2.0 | -8.0 | -7.7 | -11.7 | 9/9 |
| deadlock_dependency | 2 | 60 | -3.2 | +13.2 | +5.2 | +6.2 | +6.3 | +1.3 | +5.3 | +4.0 | +3.3 | 1/9 |
| barrier_synchronization | 1 | 30 | +3.5 | +6.5 | +10.2 | +6.2 | -8.7 | +9.7 | +3.7 | +5.7 | +13.3 | 1/9 |
| controlled_deployment_gate | 1 | 30 | +13.5 | +6.5 | +10.2 | +6.2 | -5.3 | -0.3 | +13.7 | -7.7 | +3.3 | 3/9 |
| coordinated_containment | 1 | 30 | -3.2 | -10.2 | -6.5 | -7.2 | -8.7 | -13.7 | -3.0 | +2.3 | -10.0 | 8/9 |
| distributed_lock | 1 | 30 | +0.2 | -10.2 | -9.8 | -7.2 | -8.7 | -7.0 | -9.7 | -11.0 | -13.3 | 8/9 |
| duplicate_hold_stale | 1 | 30 | -3.2 | -20.2 | -9.8 | -13.8 | -18.7 | -13.7 | -19.7 | -14.3 | -16.7 | 9/9 |
| multi_device_pairing | 1 | 30 | -3.2 | -3.5 | +3.5 | -0.5 | +1.3 | -0.3 | -3.0 | -1.0 | -3.3 | 7/9 |
| multi_party_synchronization | 1 | 30 | +0.2 | -0.2 | -3.2 | +6.2 | +8.0 | -3.7 | -3.0 | -1.0 | -3.3 | 6/9 |
| non_overlapping_assignment | 1 | 30 | +6.8 | +13.2 | +6.8 | +12.8 | +24.7 | +29.7 | +10.3 | +15.7 | +16.7 | 0/9 |
| reversible_action_hold | 1 | 30 | -3.2 | -6.8 | -6.5 | -7.2 | +1.3 | -7.0 | -6.3 | -4.3 | -13.3 | 8/9 |
| scope_reconciliation | 1 | 30 | +10.2 | -6.8 | -3.2 | -7.2 | -8.7 | -3.7 | -6.3 | -1.0 | +0.0 | 7/9 |

Baseline (all 600 tasks, `reasoning_effort=medium`): 3.2% (0.30) | 20.2% (0.60) | 9.8% (0.48) | 23.8% (0.62) | 25.3% (0.63) | 17.0% (0.57) | 23.0% (0.63) | 17.7% (0.59) | 20.0% (0.59).

Four levers behave as designed, three do not, and the plugin view adds a fourth
category: levers that look decisive but rest on one scenario.

> [!IMPORTANT]
> **`force_async_wait` is the hardest designed lever in the corpus and it is
> close to unsolved.** It puts **all nine models below their own baseline**, and
> by the widest margin of anything here: −3.2 to −23.0 pass points. Three models
> score **0.0%** on its 30 tasks (gemini-3.5-flash, claude-sonnet-5,
> gpt-5.6-sol), and claude-opus-5's 13.3% is the field maximum. This is the same
> capability as the `within_window` bucket in section 3.3 — the bucket whose
> required checks effort barely moves — measured on a completely independent
> axis, and the two agree.

The async result has a shape worth naming, because it is not "this scenario is
hard". `sla_support_response` is the **7th hardest of the 20 seeds by field mean
score (0.533) but the 2nd lowest by pass rate (13 passes out of 270)**. Models do
most of the work and then miss on the timing: the failure is *point-like*, at the
time window, not diffuse across the task. That is exactly the profile a
temporal-reasoning probe should produce, and it is measured on 30 tasks in the
whole corpus.

One caveat on attributing that result to the plugin layer. The seed's
`conflict_taxonomy` carries a `subagent_delay` token that **is not a registered
plugin** — `get_plugins_for_taxonomy` silently drops it — and the behaviour it
names is implemented by hardcoded branches in `conflict_strict_validator.py:91`
and `conflict_llm_pipeline.py:2942`, which force every mock subagent to declare a
positive `virtual_duration_seconds`. So the corpus's one async scenario gets its
core mechanism from outside the plugin system. `force_async_wait` is still the
lever that selects it, but "the async plugin is hard" would be the wrong sentence.

#### What the plugin view adds

Five buckets put **all nine models below their own baseline**: `async_required`,
`duplicate_hold_stale`, `credential_lock`, `gap_completion` and `vision_decoy`.
Every one of them is carried by 3 seeds or fewer. Five buckets are unanimously
*easy* — 0 of 9 below baseline: `no_async`, `workflow_completion`,
`dependency_ordering`, `duplicate_avoidance` and `non_overlapping_assignment`.

- **`duplicate_hold_stale` is the hardest bucket in the benchmark**: 3.3% pooled,
  9 passes in 270 attempts, with three models at 0.0% and no model above 10.0%.
  It is one seed (`multi_tab_slot_monitoring`), so read it as a scenario result.
- **`credential_lock` is the hardest bucket with more than one seed** (11.1%
  pooled, −2.0 to −11.7). It is also prompt-only: it declares no `emits_effects`,
  so nothing in the runtime enforces it. A constraint that exists only as
  generated instructions is still costing every model pass rate.
- **Exactly two plugins carried by 6 or more seeds return a unanimous verdict,
  and both point the wrong way** — `workflow_completion` and
  `dependency_ordering`, 0 of 9 below baseline each. The first is a near-constant
  (18 of 20 seeds) and cannot discriminate by construction; the second is not.
  `dependency_ordering` covers 6 seeds and 180 tasks per model at +0.8 to +6.1,
  so ordering constraints as built make tasks *easier*, most likely because an
  explicit required order is a plan the model can follow.
- **`non_overlapping_assignment` is the easiest bucket anywhere** (33.0% pooled,
  up to +29.7 for claude-sonnet-5) and again is a single seed.

The category table adds less than it appears to. Three of its six buckets are
numerically identical to a single plugin — `order` to `dependency_ordering`,
`timeout` to `time_window_expiration`, `main_supervised` to
`scope_reconciliation` — because in each case one plugin covers every seed in
the category. The other three — `inconsistency` (9 seeds), `coordination` (6)
and `subagent_collaboration` (4) — pool more seeds than any single plugin they
contain, and none of them separates from baseline (4, 6 and 6 of 9 below
baseline, deltas inside −4.3 to +2.6). **Conflict category is not a difficulty
axis on this corpus.**

#### Depth of enforcement does not predict difficulty

The 23 plugins split into 13 that gate real command execution — they declare
`emits_effects`, which the generator intersects with the scenario's command
vocabulary — and 10 that only inject prompt text. All 7 `challenge` plugins are
in the prompt-only group. Grouping the seeds by how many hard-enforced mechanics
they route to gives 2 seeds with none, 12 with one, and 6 with two, at 11.1%,
18.7% and 18.2% pooled pass rate respectively.

The 1-versus-2 contrast is the informative one and it is **null**: 18.7% against
18.2%, with 3 of 9 models below their own baseline in each group. Adding a second
hard-enforced mechanic to a scenario does not make it measurably harder. The
0-mechanic group looks dramatic (11.1%, 9 of 9 below baseline) but cannot be
attributed: its two seeds — `clinical_trial_dosage_conflict` and
`multimodal_temporal_conflict_01` — are *exactly* the two seeds carrying
`credential_lock`, so "no hard mechanic" and "has `credential_lock`" are the same
bucket under two names, and nothing in this corpus can tell them apart.

#### The three levers that do not produce difficulty

These are the ones with the widest coverage:

- **`workflow_completion` (540 tasks) is null and slightly positive** — 0 of 9
  models below baseline, every delta between +0.2 and +1.3. It tags 18 of 20
  seeds, so it is close to a constant and cannot discriminate by construction.
- **`multi_source_reasoning` (360 tasks) is null.** Seven of nine models sit
  below baseline but the entire spread is −1.1 to +0.8 points, inside single-run
  noise. The plugin intended to force cross-source reconciliation is not, on this
  corpus, making tasks harder than average.
- **`duplicate_avoidance` (60 tasks) is the easiest challenge token**, 0 of 9
  below baseline and up to +22.0 points (gpt-5.6-sol 45.0%). Whatever it is
  testing, models already do it.

`gap_completion` (90 tasks) and `vision_decoy` (60 tasks) do work, putting 9 of 9
models below baseline at −0.9 to −8.1 and −0.5 to −8.3 points. The `vision_decoy`
result **inverts** the reading a smaller decoy bucket would have supported:
resisting a misleading chart is not free here. `information_transfer` is mildly
negative (8 of 9, −3.0 to +0.4) and `vision_graded` likewise (8 of 9, never worse
than −3.4).

#### Statistical power is allocated opposite to the signal

Every lever with a unanimous negative verdict is carried by 3 seeds or fewer,
and no lever carried by 10 or more seeds moves any model more than 3.5 pass
points in either direction — the six widest-coverage buckets
(`workflow_completion`, `no_async`, `information_transfer`, `vision_graded`,
`multi_source_reasoning`, `state_repair`) together span only −3.4 to +3.5
across all nine models. `force_async_wait`
covers 30 tasks, `duplicate_hold_stale` 30, `credential_lock` and `vision_decoy`
60 each, `gap_completion` 90 — while the two null plugins cover 540 and 360.
Ten of the 23 plugins cannot be distinguished from their single scenario at all.
The corpus spends most of its measurement on constraints that do not
discriminate, and measures the ones that do with the fewest tasks it has.

Three difficulty knobs make this concrete. `force_round_type_plan`,
`force_decoy` and `force_no_signpost` are declared on `ConflictVLMSeed`
(`conflict_vlm_spec.py:282`, `:292`, `:301`), fully implemented in the pipeline
and materializer, and covered by a 569-line test
(`tests/test_difficulty_knobs_m1_m2_m3.py`) — and **set on 0 of the 20 seeds**.
They are working levers that the corpus never pulls. Rebalancing coverage toward
the levers that discriminate, and turning on the three that are already built,
is the cheapest available increase in this benchmark's resolution.

> [!NOTE]
> **Footnote on `archetype`.** Earlier revisions of this section grouped tasks by
> `ARCHETYPE_CATEGORY` (`conflict_vlm_spec.py:459`), a 15-entry hand-written map
> from `archetype` to a category name. That grouping is retired here for two
> reasons. First, `archetype` is not read by the router, so it never influenced
> what any scenario contained. Second, the map disagrees with the plugins' own
> `ConflictPluginSpec.category` on 5 of its 15 entries: it has no `coordination`
> category at all and files `barrier_synchronization`, `deadlock_dependency` and
> `distributed_lock` under `inconsistency`, files `evidence_mismatch` under
> `main_supervised` where the plugin says `inconsistency`, and contains one entry
> (`inconsistency_expense`) that is not a plugin. The old grouping also produced
> an `other` bucket of "the 5 seeds with no archetype", described there as having
> no declared mechanism; those 5 seeds each activate 6 or more conflict plugins,
> so that description was wrong. The category table above is computed from
> `ConflictPluginSpec.category` over the plugins each seed actually routes to.

---


## 6. Interpretation

- **The corpus is doing its job on difficulty.** A 25.3% ceiling with 60
  scenarios and 3,284 scored units per model leaves ample headroom, and the three
  zero-pass scenarios show the hard tail is genuinely hard rather than merely
  noisy.
- **Discrimination is concentrated in three buckets.** F1, F2 and F4 — all
  content — carry the capability signal. F5 and F6 consume **545 of 3,284
  required units (16.6%)** while separating models by 9.3 and ~6 checks
  respectively. If the goal is discrimination per unit of scoring weight, that is
  where the slack is.
- **Grounding is the one thing effort fixes, and it is also the thing the rubric
  mostly does not score.** Only 346 of 822 `assets_accessed` checks are required,
  and 254 of 600 tasks score nothing for looking at an image (cross-model report
  §11.1). So the benchmark rewards more thinking with better grounding, then
  declines to grade most of it. These two facts should be reconciled before the
  next corpus revision.
- **Clustering, not per-check accuracy, is what separates the leader.**
  gpt-5.6-sol is *more* accurate per check than claude-opus-5 (61.8% vs 61.7%)
  and passes fewer tasks. The full derivation is in the cross-model report §7.1;
  the implication for this document is that **per-check summaries and pass-rate
  summaries are answering different questions**, and the ladder in §1 should
  always be read with the per-check column beside it.
- **One model needs a different diagnosis from the rest.** For eight models the
  question is "what content did it get wrong". For gemini-3.5-flash the question
  is "why did it not finish" — half its semantic checks were never judged. Its
  failure taxonomy is dominated by an availability problem upstream of reasoning.
- **Statistical power is allocated opposite to the signal.** §5.3 measures the
  designed levers directly: the three that separate models (`force_async_wait`,
  `gap_completion`, `vision_decoy`) cover 30, 90 and 60 tasks, while the two that
  do nothing (`workflow_completion`, `multi_source_reasoning`) cover 540 and 360.
  The temporal lever in particular is one seed of twenty. Rebalancing seed
  coverage toward the levers that discriminate would buy more resolution than any
  change to the rubric, and it is the cheapest change available — it is a
  property of the seed corpus, not of the scoring code.

---

## 7. Caveats & reproducibility

- **Single run per cell.** One round (`r01`) per model per effort. Small gaps are
  noise. Where a claim depends on a gap being real it is backed by a paired test
  (McNemar over the 600 shared tasks) or a bootstrap interval, both in the
  cross-model report — not by the point estimate here.
- **Serving-stack confound.** Gemini, Claude and OpenAI models run on three
  different stacks. Cross-vendor gaps carry that confound; the Gemini-internal
  spread (§1) does not.
- **Effort is a request, not a quantity.** See §2.1.
- **The judge decides 65.1% of required units.** See §2.3.
- **Seven episodes have no rubric**, biasing two Gemini cells favorably by under
  0.1 points. See §2.5.
- **`reconciliation` round type has n=1** and is excluded from all round-type
  analysis (cross-model report §8.2).
- **The eight curated rounds are biased on grounding**: 7 of 8 score no grounding
  at all, so they are not a representative sample for that axis (cross-model
  report §11.2).
- **`waits.hist` in `failure_taxonomy.json` is truncated to the eight most common
  durations** (`hist.most_common(8)`), so its counts do not sum to `waits.n`. Use
  `n` / `mean` / `median` / `max` / `total` for totals, and re-scan the raw
  trajectories for any claim about a rare duration. The
  600 → 1200 → 3600 → 7200s escalation cited in §4 is one such re-scan: all four
  of gemini-3.1-pro's waits above 300s fall outside the truncated histogram and
  belong to the single task `sla_support_response_v01_r08`.
- **§5.3 is the only section that joins the seed corpus.** Every other section
  reads task metadata and the rubric, both of which live in the eval records;
  `difficulty_challenges`, `archetype` and the two knobs exist only in the 20 seed
  files and appear in no spec, which is why `difficulty_axes.json` reports an
  empty `conflict_taxonomy` axis and why earlier revisions of this document had no
  designed-lever analysis at all. The join is by `scenario_slug` with the trailing
  `_v<NN>` stripped; all 60 scenarios map, and `design_axis_slice.py` warns if any
  does not.
- **The async lever is one seed.** Every `force_async_wait` number in §5.3 comes
  from `sla_support_response` × 3 variants × 10 rounds. The effect is large and
  agrees with the independent `within_window` measurement in §3.3, but it is a
  single-scenario result and should not be read as a general temporal-reasoning
  score.
- **Reproduce everything** from the repository root of the eval checkout:

  ```
  python tools/failure_taxonomy.py                                      # F1-F6, ranks, head-to-head
  python tools/build_behavior_report.py --root <eval-root> --effort medium
  python tools/scan_difficulty_axes.py  --root <eval-root> --effort medium
  python tools/scan_curated_rounds.py   --root <eval-root> --effort medium
  python tools/scan_reasoning_tokens.py
  python tools/design_axis_slice.py --root <eval-root> --markdown        # section 5.3 tables
  python tools/design_axis_slice.py --root <eval-root> --json            # design_axes.json
  python tools/audit_report.py                                          # recomputing audit of both documents
  ```

  `design_axis_slice.py` needs two roots: `--root` is the data checkout holding
  `eval_results/`, and `--seeds` is the source tree holding
  `clasharness/data/scenario_seeds/conflict_vlm_v1/`. They are not the same
  directory, and `--seeds` defaults correctly off the tool's own location.

---

## Appendix A: full per-scenario mean scores (60 × 9)

Mean score over 10 rounds, with pass count beneath. **Bold** = at least one pass.
Sorted hardest-first by field mean. Generated by `tools/failure_taxonomy.py`
(`cells.medium.models.<key>.scenarios`).

<!-- APPENDIX-A-BEGIN -->

| scenario | field | 3.5-flash | 3.6-flash | 3.1-pro | opus-4.8 | opus-5 | sonnet-5 | sol | luna | terra |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `warehouse_robotics_safety_halt_v03` | 0.246 | 0.19<br>0 | 0.20<br>0 | 0.09<br>0 | 0.29<br>0 | 0.36<br>0 | 0.24<br>0 | 0.28<br>0 | 0.29<br>0 | 0.27<br>0 |
| `distributed_lock_shared_resource_v02` | 0.312 | 0.17<br>0 | 0.41<br>0 | 0.27<br>0 | 0.36<br>0 | 0.40<br>0 | 0.27<br>0 | 0.40<br>0 | 0.30<br>0 | 0.24<br>0 |
| `clinical_trial_dosage_conflict_v01` | 0.343 | 0.17<br>0 | **0.38**<br>1 | **0.23**<br>1 | **0.49**<br>2 | **0.53**<br>2 | **0.41**<br>1 | 0.28<br>0 | **0.29**<br>1 | **0.32**<br>1 |
| `clinical_trial_dosage_conflict_v03` | 0.373 | 0.20<br>0 | 0.34<br>0 | 0.13<br>0 | **0.46**<br>1 | **0.39**<br>1 | **0.45**<br>1 | **0.48**<br>1 | **0.45**<br>1 | **0.45**<br>1 |
| `clinical_trial_dosage_conflict_v02` | 0.378 | 0.08<br>0 | **0.57**<br>4 | 0.26<br>0 | **0.31**<br>1 | 0.40<br>0 | 0.39<br>0 | **0.54**<br>2 | **0.39**<br>1 | **0.48**<br>1 |
| `multimodal_temporal_conflict_01_v01` | 0.383 | 0.27<br>0 | **0.40**<br>2 | **0.30**<br>1 | **0.40**<br>1 | **0.40**<br>1 | **0.44**<br>2 | **0.46**<br>2 | **0.38**<br>1 | **0.38**<br>1 |
| `distributed_lock_shared_resource_v03` | 0.387 | **0.29**<br>1 | 0.37<br>0 | 0.26<br>0 | **0.49**<br>1 | **0.42**<br>1 | 0.43<br>0 | 0.41<br>0 | 0.43<br>0 | 0.38<br>0 |
| `coordinated_containment_v01` | 0.392 | 0.20<br>0 | 0.42<br>0 | 0.27<br>0 | **0.53**<br>1 | **0.54**<br>1 | 0.34<br>0 | 0.49<br>0 | 0.42<br>0 | 0.32<br>0 |
| `payment_chargeback_dispute_v02` | 0.423 | 0.21<br>0 | **0.46**<br>1 | **0.38**<br>1 | **0.47**<br>1 | **0.50**<br>1 | **0.50**<br>2 | **0.40**<br>1 | **0.46**<br>1 | **0.42**<br>1 |
| `multi_tab_slot_monitoring_v02` | 0.438 | 0.35<br>0 | 0.42<br>0 | 0.39<br>0 | 0.50<br>0 | 0.47<br>0 | 0.40<br>0 | 0.45<br>0 | 0.51<br>0 | 0.45<br>0 |
| `sla_support_response_v03` | 0.441 | 0.39<br>0 | **0.45**<br>1 | **0.43**<br>1 | **0.48**<br>2 | **0.46**<br>2 | 0.39<br>0 | 0.44<br>0 | **0.45**<br>2 | **0.48**<br>2 |
| `multimodal_temporal_conflict_01_v02` | 0.442 | 0.14<br>0 | **0.46**<br>1 | **0.36**<br>1 | **0.53**<br>1 | **0.58**<br>2 | **0.49**<br>2 | **0.56**<br>1 | 0.44<br>0 | 0.41<br>0 |
| `incident_commander_selection_v01` | 0.443 | 0.11<br>0 | **0.48**<br>1 | 0.24<br>0 | **0.51**<br>1 | **0.63**<br>3 | **0.56**<br>2 | **0.51**<br>1 | **0.48**<br>1 | **0.46**<br>1 |
| `multimodal_temporal_conflict_01_v03` | 0.471 | 0.18<br>0 | **0.51**<br>2 | 0.34<br>0 | **0.59**<br>3 | **0.53**<br>3 | **0.57**<br>3 | **0.57**<br>3 | **0.51**<br>2 | **0.45**<br>1 |
| `multi_tab_slot_monitoring_v01` | 0.495 | 0.20<br>0 | 0.43<br>0 | 0.37<br>0 | **0.60**<br>2 | **0.65**<br>1 | 0.55<br>0 | 0.56<br>0 | 0.56<br>0 | 0.54<br>0 |
| `employee_onboarding_workflow_v03` | 0.496 | **0.44**<br>1 | **0.57**<br>2 | 0.36<br>0 | **0.49**<br>1 | **0.51**<br>1 | 0.45<br>0 | 0.53<br>0 | **0.57**<br>1 | **0.55**<br>1 |
| `multi_device_pairing_v01` | 0.502 | 0.39<br>0 | 0.62<br>0 | 0.42<br>0 | **0.47**<br>1 | **0.62**<br>1 | **0.46**<br>1 | 0.51<br>0 | **0.49**<br>1 | **0.53**<br>1 |
| `multi_tab_slot_monitoring_v03` | 0.510 | 0.23<br>0 | 0.50<br>0 | 0.43<br>0 | **0.57**<br>1 | **0.59**<br>1 | **0.57**<br>1 | **0.53**<br>1 | **0.60**<br>1 | **0.55**<br>1 |
| `expiring_resource_hold_v02` | 0.516 | 0.33<br>0 | **0.67**<br>2 | 0.50<br>0 | **0.63**<br>2 | **0.58**<br>2 | **0.57**<br>1 | **0.48**<br>1 | **0.47**<br>1 | 0.41<br>0 |
| `employee_onboarding_workflow_v02` | 0.521 | 0.34<br>0 | **0.49**<br>1 | **0.43**<br>1 | **0.53**<br>2 | **0.59**<br>2 | 0.55<br>0 | **0.66**<br>2 | **0.57**<br>1 | **0.53**<br>1 |
| `employee_onboarding_workflow_v01` | 0.524 | 0.31<br>0 | **0.55**<br>1 | **0.53**<br>2 | **0.59**<br>3 | **0.58**<br>3 | 0.46<br>0 | **0.58**<br>2 | **0.57**<br>3 | **0.55**<br>3 |
| `warehouse_robotics_safety_halt_v01` | 0.527 | 0.17<br>0 | **0.57**<br>4 | **0.36**<br>1 | **0.61**<br>4 | **0.72**<br>4 | **0.62**<br>2 | **0.67**<br>3 | **0.55**<br>1 | **0.46**<br>2 |
| `multi_party_access_permit_v01` | 0.527 | 0.35<br>0 | 0.53<br>0 | 0.45<br>0 | 0.59<br>0 | **0.69**<br>2 | 0.52<br>0 | 0.53<br>0 | 0.53<br>0 | 0.56<br>0 |
| `payment_chargeback_dispute_v03` | 0.533 | 0.07<br>0 | **0.50**<br>2 | **0.47**<br>1 | **0.63**<br>4 | **0.60**<br>3 | **0.50**<br>3 | **0.70**<br>3 | **0.60**<br>3 | **0.73**<br>5 |
| `warehouse_robotics_safety_halt_v02` | 0.540 | 0.07<br>0 | **0.65**<br>5 | **0.42**<br>2 | **0.58**<br>3 | **0.80**<br>5 | **0.58**<br>3 | **0.62**<br>3 | **0.52**<br>2 | **0.62**<br>3 |
| `access_request_approval_v03` | 0.554 | **0.56**<br>2 | **0.66**<br>1 | **0.58**<br>1 | **0.46**<br>1 | **0.53**<br>1 | **0.41**<br>1 | **0.60**<br>2 | **0.54**<br>2 | **0.63**<br>2 |
| `sla_support_response_v01` | 0.555 | 0.44<br>0 | 0.58<br>0 | 0.42<br>0 | **0.60**<br>1 | **0.66**<br>1 | 0.58<br>0 | 0.57<br>0 | 0.59<br>0 | 0.54<br>0 |
| `collaborative_contract_review_v01` | 0.557 | 0.35<br>0 | **0.70**<br>2 | **0.47**<br>1 | **0.48**<br>1 | **0.58**<br>4 | **0.61**<br>4 | **0.66**<br>2 | **0.57**<br>1 | **0.58**<br>2 |
| `satellite_ground_station_handover_v01` | 0.561 | 0.05<br>0 | **0.61**<br>4 | **0.45**<br>3 | **0.59**<br>4 | **0.76**<br>5 | **0.61**<br>4 | **0.69**<br>5 | **0.68**<br>5 | **0.61**<br>4 |
| `incident_commander_selection_v03` | 0.564 | 0.18<br>0 | **0.60**<br>1 | **0.40**<br>1 | **0.62**<br>2 | **0.69**<br>3 | **0.62**<br>2 | **0.70**<br>3 | **0.58**<br>2 | **0.70**<br>3 |
| `coordinated_containment_v02` | 0.576 | 0.15<br>0 | **0.59**<br>1 | 0.49<br>0 | **0.69**<br>3 | **0.62**<br>3 | 0.58<br>0 | **0.79**<br>4 | **0.66**<br>3 | **0.63**<br>2 |
| `barrier_sync_deployment_v03` | 0.581 | 0.38<br>0 | **0.71**<br>3 | **0.61**<br>2 | **0.53**<br>1 | **0.52**<br>1 | **0.51**<br>1 | **0.65**<br>2 | **0.67**<br>2 | **0.67**<br>3 |
| `it_deployment_gate_v02` | 0.588 | **0.64**<br>4 | **0.64**<br>3 | **0.61**<br>3 | **0.62**<br>4 | 0.45<br>0 | **0.52**<br>2 | **0.64**<br>4 | **0.57**<br>1 | **0.59**<br>2 |
| `satellite_ground_station_handover_v03` | 0.591 | 0.10<br>0 | **0.67**<br>3 | **0.47**<br>2 | **0.62**<br>4 | **0.71**<br>6 | **0.65**<br>4 | **0.75**<br>5 | **0.60**<br>2 | **0.75**<br>5 |
| `expense_reimbursement_conflict_v03` | 0.602 | 0.28<br>0 | **0.68**<br>1 | 0.48<br>0 | **0.68**<br>1 | **0.75**<br>4 | 0.52<br>0 | **0.66**<br>1 | **0.71**<br>1 | **0.66**<br>1 |
| `expense_reimbursement_conflict_v02` | 0.603 | 0.29<br>0 | **0.70**<br>1 | **0.60**<br>1 | **0.79**<br>3 | **0.59**<br>1 | **0.63**<br>2 | **0.60**<br>1 | **0.62**<br>1 | 0.59<br>0 |
| `sla_support_response_v02` | 0.604 | 0.48<br>0 | 0.62<br>0 | 0.52<br>0 | 0.66<br>0 | **0.70**<br>1 | 0.56<br>0 | 0.64<br>0 | 0.64<br>0 | 0.61<br>0 |
| `it_deployment_gate_v03` | 0.605 | 0.33<br>0 | **0.67**<br>2 | **0.67**<br>2 | **0.58**<br>2 | **0.60**<br>2 | **0.60**<br>1 | **0.74**<br>3 | 0.57<br>0 | **0.69**<br>2 |
| `expiring_resource_hold_v01` | 0.610 | 0.30<br>0 | **0.64**<br>2 | **0.62**<br>1 | **0.64**<br>2 | **0.72**<br>4 | **0.66**<br>2 | **0.64**<br>2 | **0.65**<br>2 | **0.61**<br>1 |
| `incident_commander_selection_v02` | 0.616 | **0.52**<br>1 | **0.50**<br>1 | 0.44<br>0 | **0.67**<br>3 | **0.71**<br>4 | **0.65**<br>2 | **0.72**<br>5 | **0.64**<br>3 | **0.70**<br>4 |
| `multi_party_access_permit_v03` | 0.617 | 0.26<br>0 | **0.67**<br>2 | **0.60**<br>1 | **0.64**<br>2 | **0.72**<br>2 | **0.55**<br>2 | **0.75**<br>2 | **0.67**<br>2 | **0.69**<br>2 |
| `it_deployment_gate_v01` | 0.620 | **0.42**<br>1 | **0.66**<br>3 | **0.55**<br>1 | **0.70**<br>3 | **0.62**<br>4 | **0.65**<br>2 | **0.67**<br>4 | **0.66**<br>2 | **0.66**<br>3 |
| `barrier_sync_deployment_v01` | 0.622 | **0.40**<br>1 | **0.67**<br>3 | **0.59**<br>2 | **0.69**<br>3 | **0.66**<br>2 | **0.65**<br>3 | **0.65**<br>2 | **0.67**<br>3 | **0.62**<br>2 |
| `multi_device_pairing_v02` | 0.622 | 0.30<br>0 | **0.68**<br>2 | **0.57**<br>1 | **0.67**<br>3 | **0.67**<br>3 | **0.75**<br>3 | **0.68**<br>3 | **0.63**<br>1 | **0.65**<br>1 |
| `distributed_lock_shared_resource_v01` | 0.625 | 0.31<br>0 | **0.71**<br>3 | 0.54<br>0 | **0.70**<br>4 | **0.80**<br>4 | **0.69**<br>3 | **0.70**<br>4 | **0.61**<br>2 | **0.55**<br>2 |
| `access_request_approval_v01` | 0.630 | **0.37**<br>2 | **0.69**<br>2 | **0.57**<br>1 | **0.76**<br>2 | **0.64**<br>1 | **0.60**<br>1 | **0.71**<br>2 | **0.63**<br>2 | **0.70**<br>2 |
| `multi_device_pairing_v03` | 0.632 | 0.45<br>0 | **0.66**<br>3 | **0.62**<br>3 | **0.68**<br>3 | **0.72**<br>4 | **0.64**<br>1 | **0.62**<br>3 | **0.64**<br>3 | **0.66**<br>3 |
| `deadlock_multi_approval_v01` | 0.649 | 0.40<br>0 | **0.69**<br>3 | **0.63**<br>2 | **0.72**<br>3 | **0.72**<br>3 | **0.69**<br>2 | **0.67**<br>3 | **0.68**<br>3 | **0.65**<br>3 |
| `coordinated_containment_v03` | 0.659 | 0.37<br>0 | **0.72**<br>2 | **0.61**<br>1 | **0.73**<br>1 | **0.64**<br>1 | **0.69**<br>1 | **0.71**<br>2 | **0.78**<br>3 | **0.68**<br>1 |
| `expiring_resource_hold_v03` | 0.663 | 0.28<br>0 | 0.70<br>0 | 0.65<br>0 | **0.76**<br>1 | **0.77**<br>2 | 0.53<br>0 | **0.78**<br>2 | **0.74**<br>1 | **0.78**<br>1 |
| `expense_reimbursement_conflict_v01` | 0.666 | 0.27<br>0 | **0.71**<br>5 | **0.53**<br>1 | **0.82**<br>5 | **0.72**<br>3 | **0.62**<br>1 | **0.79**<br>5 | **0.77**<br>3 | **0.77**<br>4 |
| `access_request_approval_v02` | 0.674 | 0.32<br>0 | **0.71**<br>1 | 0.52<br>0 | **0.76**<br>2 | **0.84**<br>3 | **0.77**<br>2 | **0.73**<br>1 | **0.71**<br>1 | **0.71**<br>2 |
| `barrier_sync_deployment_v02` | 0.675 | **0.39**<br>1 | **0.68**<br>2 | **0.56**<br>2 | **0.79**<br>5 | **0.69**<br>2 | **0.74**<br>4 | **0.76**<br>4 | **0.71**<br>2 | **0.75**<br>5 |
| `deadlock_multi_approval_v03` | 0.677 | 0.44<br>0 | **0.76**<br>4 | **0.53**<br>1 | **0.74**<br>4 | **0.68**<br>2 | **0.69**<br>3 | **0.78**<br>3 | **0.74**<br>4 | **0.73**<br>3 |
| `collaborative_contract_review_v02` | 0.719 | **0.51**<br>2 | **0.81**<br>5 | **0.62**<br>3 | **0.77**<br>5 | **0.83**<br>6 | **0.76**<br>4 | **0.73**<br>5 | **0.71**<br>3 | **0.73**<br>6 |
| `multi_party_access_permit_v02` | 0.724 | **0.50**<br>1 | **0.71**<br>4 | **0.65**<br>1 | **0.88**<br>7 | **0.86**<br>6 | **0.69**<br>2 | **0.77**<br>4 | **0.75**<br>3 | **0.71**<br>3 |
| `satellite_ground_station_handover_v02` | 0.733 | **0.23**<br>1 | **0.90**<br>7 | **0.60**<br>1 | **0.83**<br>5 | **0.73**<br>4 | **0.77**<br>5 | **0.90**<br>8 | **0.73**<br>3 | **0.90**<br>8 |
| `deadlock_multi_approval_v02` | 0.737 | 0.35<br>0 | **0.84**<br>4 | **0.73**<br>3 | **0.82**<br>4 | **0.85**<br>5 | **0.66**<br>1 | **0.87**<br>5 | **0.77**<br>3 | **0.75**<br>3 |
| `collaborative_contract_review_v03` | 0.743 | **0.36**<br>1 | **0.73**<br>3 | **0.74**<br>1 | **0.80**<br>5 | **0.81**<br>5 | **0.84**<br>6 | **0.77**<br>3 | **0.85**<br>6 | **0.79**<br>3 |
| `payment_chargeback_dispute_v01` | 0.779 | 0.12<br>0 | **0.90**<br>8 | **0.84**<br>5 | **0.88**<br>6 | **0.87**<br>8 | **0.83**<br>7 | **0.93**<br>8 | **0.87**<br>7 | **0.77**<br>6 |

<!-- APPENDIX-A-END -->

---

## Cross-references

- [`cross-model-behavioral-comparison.md`](./cross-model-behavioral-comparison.md)
  — behavioral signatures, action mix, command surface, grounding × action map,
  stop-reason mix, loops, failure clustering, the difficulty axes, the
  three-level effort sweep, eight curated rounds, limitations.
- Per-model deep-dives: see the companion-document list at the top of this file.
- `failure_taxonomy.json` — the machine-readable source for every number in §3,
  §4 and §5, covering all 27 cells.
- [`design_axes.json`](./design_axes.json) — the machine-readable source for §5.3:
  every designed lever, per model, with the delta against that model's own
  baseline. Produced by [`tools/design_axis_slice.py`](./tools/design_axis_slice.py),
  which is the only tool in this directory that joins the eval records to the 20
  scenario seeds. The same buckets appear per model in §4.3 of each deep-dive.
