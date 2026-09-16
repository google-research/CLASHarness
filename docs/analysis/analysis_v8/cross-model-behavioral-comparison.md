# Cross-Model Behavioral Comparison: multiagent_vlm_conflict (v8 corpus)

Side-by-side behavioral comparison of **fourteen solver models** on
`multiagent_vlm_conflict` — what each one *does* with its step budget, how those
habits turn into scores, and how all of it moves when `reasoning_effort` is
swept across `low` / `medium` / `high` for the ten models that were swept.

An interactive, self-contained rendering of the same data (with charts) is
generated alongside this file as
[`cross-model-behavioral-comparison.html`](./cross-model-behavioral-comparison.html).

<!-- block:provenance.para -->
**Provenance.** Every figure is recomputed from `eval_results/data_3_6_flash_v8/`, over **34 analysis cells**: `aligned_setting_{low,medium,high}/<Vendor>/<model>/` for the 10 swept models (18,000 episodes) plus `aligned_setting_high_ocr/<Vendor>/<model>/` (v4-flash) and `open_source_high/<Vendor>/<model>/` (v4-pro, glm-5.2 and kimi-k3) for the 4 open-source models, which were run once at `high` rather than swept (2,400 episodes) — **20,400 episodes in total**, each cell being 60 scenarios × 10 rounds = 600 tasks. `aligned_setting_high_ocr` is a staging directory, not an OCR condition — the runs under it were not preprocessed with OCR, and the name is describing that they were served asset file paths as text (`multimodal_input_mode=path`), which is stated as a run condition below in its own right. The run subdirectory is `r01/` throughout, with one exception — 3.7-flash is read from `r02`, for the reason given under run conditions below.

The corpus is **not** a clean cross-product, so do not multiply anything by 14 here: only 10 of the 14 models have an effort sweep, which is why §9 covers 10 models and every other section covers 14.

Unless a table says otherwise, figures are at `reasoning_effort=high`, the primary setting: 14 models × 600 tasks = **8,400 episodes** and **120,575 recorded actions**. Regenerate with:

```
python tools/build_behavior_report.py --root $CLASHARNESS_DATA_ROOT --effort high
```

**Not every row here is like-for-like.** v4-pro, v4-flash and glm-5.2 were handed asset file paths as text and never received an image (`multimodal_input_mode=path`); kimi-k3, the open-source model that did receive images, is the control; also, 3.7-flash is reported on `r02`, the higher of its two complete runs, where every other row is a single run (the two differ by 1.7pp, McNemar p=0.27). Full statement: [run conditions](#run-conditions-that-are-not-comparable).
<!-- /block:provenance.para -->

The prose blocks in §3 and §10 are versioned in
[`behavior_notes.json`](./behavior_notes.json) and are embedded byte-for-byte in
both this file and the HTML report.

**`gpt-5.5` is not in this corpus.** The `OpenAI/5_5` directories exist under
`aligned_setting_{medium,high}` but contain no interaction logs. It is reported
nowhere in this document.

---

## How to read this report

> [!IMPORTANT]
> **Five things to hold onto before reading any table.**
>
> 1. **The effort sweep is genuinely like-for-like.** All ten models expose
>    `reasoning_effort`, and every one of the **1,800** run headers
>    (10 models × 3 efforts × 60 scenarios) records
>    `reasoning_effort == reasoning_effort_effective`, split exactly 60/60/60.
>    This is the main respect in which v8 improves on the earlier `think`
>    corpus, where the Gemini models exposed no such knob and "aligned" meant
>    only "aligned harness".
> 2. **Scoring is fractional, and `success` is all-or-nothing.**
>    `score = satisfied units / total units`, where total units = every
>    `evaluation_check` **except** `assets_accessed` checks not marked
>    `role: required`. That is **3,284** scored units per model out of 3,760 raw
>    checks. Equivalently and more simply: **the scored units are exactly the
>    checks with `role: required`**, because all **476** non-required checks in
>    the corpus are `assets_accessed` diagnostics — there is no third role and no
>    unlabelled check. `success` is exactly `score == 1.0` — verified with zero
>    counter-examples on all **20,393** scored episodes in the corpus (the
>    whole 20,400 less the seven provider errors, which carry no score).
> 3. **Grounding is mostly unscored.** Only **346** of the corpus's **822**
>    `assets_accessed` checks count. **254 of 600 tasks (42.3%) score nothing at
>    all for looking at an image.** Treat every "passed without opening
>    anything" observation as a statement about the rubric, not about whether
>    vision was needed. See §11.
> 4. **Notification checks are decided by body content, not routing.** In three
>    §10 rounds, ten to fourteen of fourteen models message the agent the
>    criterion names and at most three are credited. Routing is near-universal
>    and therefore not a discriminator on this corpus; message body is. See
>    §11.3.
> 5. **This is a single run per model per effort** — `r01` throughout, except
>    3.7-flash, which is reported on `r02`, and the four open-weight models,
>    which were run once at `high` rather than swept. Small gaps are noise.
>    Where a claim depends on a gap being real, it is backed by a paired test
>    (McNemar, §9) or a bootstrap interval (§7), not by the point estimate.

**The roster.** Row labels in every generated table are the short display names
in the first column below.

<!-- block:roster.table -->
| Model | model id | provider | `reasoning_effort` | run |
|---|---|---|---|---|
| 3.5-flash | `gemini-3.5-flash` | Gemini | `low` / `medium` / `high` | `r01` |
| 3.6-flash | `gemini-3.6-flash` | Gemini | `low` / `medium` / `high` | `r01` |
| 3.7-flash | `gemini-3.7-flash` | Gemini | `low` / `medium` / `high` | **`r02`** (see below) |
| 3.1-pro | `gemini-3.1-pro-preview` | Gemini | `low` / `medium` / `high` | `r01` |
| opus-4.8 | `claude-opus-4-8` | Claude | `low` / `medium` / `high` | `r01` |
| opus-5 | `claude-opus-5` | Claude | `low` / `medium` / `high` | `r01` |
| sonnet-5 | `claude-sonnet-5` | Claude | `low` / `medium` / `high` | `r01` |
| sol | `openai.gpt-5.6-sol` | OpenAI | `low` / `medium` / `high` | `r01` |
| luna | `openai.gpt-5.6-luna` | OpenAI | `low` / `medium` / `high` | `r01` |
| terra | `openai.gpt-5.6-terra` | OpenAI | `low` / `medium` / `high` | `r01` |
| v4-pro | `deepseek-ai/DeepSeek-V4-Pro` | Open-source | `high` only (not swept) | `r01` |
| v4-flash | `deepseek-ai/DeepSeek-V4-Flash-0731` | Open-source | `high` only (not swept) | `r01` |
| kimi-k3 | `moonshotai/Kimi-K3` | Open-source | `high` only (not swept) | `r01` |
| glm-5.2 | `zai-org/GLM-5.2-FP8` | Open-source | `high` only (not swept) | `r01` |

Rows are grouped by provider, in the order every table and chart in this report uses. Every field here is read off the 840 episode headers of the 14 primary cells, not transcribed.

Identical across all 14: `tool_mode=native`, `tool_schema_mode=native`, `native_tool_choice=auto`, `history_mode=full`, and a **20-step budget**. `gemini_media_resolution=ultra_high` is a Gemini-only knob, set on those 4 models and absent elsewhere. The fields that are *not* uniform are `multimodal_input_mode`, `multimodal_input_mode_effective`, `reasoning_effort`, `reasoning_effort_effective`, `max_image_edge` — every one of them is stated per-model below.

**Not every row here is like-for-like.** v4-pro, v4-flash and glm-5.2 were handed asset file paths as text and never received an image (`multimodal_input_mode=path`); kimi-k3, the open-source model that did receive images, is the control; also, 3.7-flash is reported on `r02`, the higher of its two complete runs, where every other row is a single run (the two differ by 1.7pp, McNemar p=0.27). Full statement: [run conditions](#run-conditions-that-are-not-comparable).
<!-- /block:roster.table -->

<!-- block:roster.conditions -->
<a name="run-conditions-that-are-not-comparable"></a>

**Run conditions that are not comparable across the field.** All 14 models ran the same harness build, the same 600 tasks and the same nominal effort; what follows is the residue that did not align. Each line is read off that model's own episode headers and checked against this list on every build.

- **3.7-flash** — run selection: reported on r02, the higher of its two complete runs, while every other model is reported on its single r01 — r01 scored 18.67% and r02 20.33% (paired: 38 gained, 28 lost, McNemar p=0.27), so the gap is within run-to-run noise and this row is a best-of-two where the others are single draws.
- **v4-pro** — multimodal_input_mode=path (never received an image); reasoning_effort=high, set via chat_template_kwargs (with thinking=true) rather than the API field; harness max_image_edge=1920 rather than the 2576 the other thirty-one cells used, which is inert for a model sent no images.
- **v4-flash** — multimodal_input_mode=path (never received an image); reasoning_effort=high, set via chat_template_kwargs (with thinking=true) rather than the API field; harness max_image_edge=1920 rather than the 2576 the other thirty-one cells used, which is inert for a model sent no images.
- **kimi-k3** — multimodal_input_mode=both; reasoning_effort medium requested, high served.
- **glm-5.2** — multimodal_input_mode=path (never received an image); reasoning_effort=high, set via chat_template_kwargs rather than the API field; served FP8-quantized (zai-org/GLM-5.2-FP8); harness max_image_edge=1920 rather than the 2576 the other thirty-one cells used, which is inert for a model sent no images.

These are conditions, not excuses, and this report does not assert that any of them caused the scores they sit beside. The one that looks most damning is measurably not. The corpus’ required grounding check scores whether the needed asset was *opened* — which a file path supports — and on the 343 tasks where all fourteen models have it scored, the three path-mode models rank **1st, 3rd and 6th of fourteen** (glm-5.2 95.3%, v4-flash 93.6%, v4-pro 91.3%). So the ranking is **led** by one of them, and its floor — sonnet-5 at 77.8%, 17.5 points below — is a model that did receive images. The two input modes also do not separate into two score populations: the path-mode band — **8.5% to 19.8%** pass — lies inside the **3.0%–25.2%** the image-receiving models span, and the two rows at its ends (v4-flash and v4-pro, DeepSeek’s own two entries) ran under identical conditions yet finish **11.3 points** apart. So the path rows are best read as measurements of those models; this corpus does not isolate an effect of the mode itself.

Separately, the 4 open-source models were run **once**, at `high`, so they appear in every main table but not in the effort chapter (§9).
<!-- /block:roster.conditions -->

---

## 1. Behavioral signatures at a glance

<!-- block:signatures.table -->
Every model over all 600 tasks at `reasoning_effort=high`. **pass@budget** is the all-or-nothing rate: every scored unit of the task satisfied within the step budget. **capped** counts episodes stopped at `max_steps` rather than by the model. Each column's lone high and lone low are bold.

| Model | pass@budget | passed | avg score | step p50 | mean steps | capped | opens/task | writes/task | zero-open | $/task | $/solved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3.5-flash | **3.0%** | **18** | **0.259** | 20.0 | **19.70** | **535** | 3.56 | **0.31** | 18 | 1.1861 | 39.5374 |
| 3.6-flash | 18.5% | 111 | 0.586 | 16.5 | 16.30 | 148 | 2.90 | 0.99 | 5 | 0.9414 | 5.0889 |
| 3.7-flash | 20.3% | 122 | 0.581 | 19.0 | 17.80 | 244 | 2.25 | 0.91 | 60 | 0.5110 | 2.5130 |
| 3.1-pro | 12.0% | 72 | 0.487 | 12.0 | 13.20 | 152 | 3.15 | 1.00 | 14 | 0.9630 | 8.0251 |
| opus-4.8 | 22.8% | 137 | 0.621 | 12.0 | 12.20 | 37 | 2.71 | 1.04 | 1 | 4.8879 | 21.4070 |
| opus-5 | **25.2%** | **151** | **0.642** | 11.0 | 11.10 | **3** | 2.98 | 1.14 | 1 | 4.3843 | 17.4209 |
| sonnet-5 | 20.8% | 125 | 0.596 | **10.0** | **10.60** | 20 | **1.83** | 1.05 | **92** | 1.7188 | 8.2502 |
| sol | 24.5% | 147 | 0.633 | 14.0 | 13.70 | 7 | 4.08 | 1.24 | 0 | 3.6082 | 14.7273 |
| luna | 19.8% | 119 | 0.604 | 13.0 | 13.50 | 55 | 3.90 | **1.43** | 0 | 0.1429 | 0.7207 |
| terra | 20.0% | 120 | 0.602 | 11.0 | 11.40 | 8 | 2.65 | 1.24 | 1 | 1.2180 | 6.0898 |
| v4-pro | 19.8% | 119 | 0.581 | 13.0 | 13.40 | 81 | 4.36 | 1.00 | 0 | 0.0534 | 0.2690 |
| v4-flash | 8.5% | 51 | 0.375 | 20.0 | 17.80 | 358 | 6.00 | 0.58 | 0 | n/a | n/a |
| kimi-k3 | 21.8% | 131 | 0.559 | 16.0 | 15.40 | 170 | 6.72 | 0.88 | 3 | 3.6210 | 16.5846 |
| glm-5.2 | 14.7% | 88 | 0.487 | 15.0 | 14.90 | 207 | **7.26** | 0.78 | 0 | 0.2006 | 1.3677 |

The money columns carry no bold and read `n/a` for v4-flash: token usage was recorded, but this model has no verified first-party rate card, and a blank is the honest cell. Costs are as-run list prices, not like-for-like — see the cost chapter for the two confounds that make the cheapest rows look cheaper than they are.

**Not every row here is like-for-like.** v4-pro, v4-flash and glm-5.2 were handed asset file paths as text and never received an image (`multimodal_input_mode=path`); kimi-k3, the open-source model that did receive images, is the control; also, 3.7-flash is reported on `r02`, the higher of its two complete runs, where every other row is a single run (the two differ by 1.7pp, McNemar p=0.27). Full statement: [run conditions](#run-conditions-that-are-not-comparable).
<!-- /block:signatures.table -->

Four signatures are visible before any deeper cut:

- **opus-5 wins by finishing, not by working harder.** It has the top pass rate
  and top score at a *median of 11 steps*, and it hits the step cap just **3
  times in 600** — the fewest in the field by better than a factor of two. Its
  nearest rival on score, sol, needs 13.70 mean steps to get 0.633.
- **sonnet-5 is the minimalist.** Fewest opens per task (1.83), lowest mean
  steps (10.60), and **92 tasks completed without opening a single asset** —
  1.5× the next-highest zero-open count (3.7-flash, 60), so skipping assets is
  a two-model habit now, not a sonnet-5 quirk. It still beats 3.1-pro and is
  nearly seven times 3.5-flash. Grounding volume is not the axis it appears to
  be (§4).
- **3.5-flash is not a weak model so much as a non-terminating one.** 535 of 600
  tasks hit the 20-step cap, writes/task is 0.31 against a field of 0.58–1.43,
  and `$/solved` is **39.54** — 54.9× luna's 0.72 and 147× v4-pro's 0.27. Its
  failure is submission, not comprehension (§6).
- **Cost and capability are close to orthogonal.** v4-pro is the cheapest per
  solved task (0.2690) at 19.8% pass, with luna second at 0.7207; opus-4.8 is
  the most expensive per task (4.8879) and does not lead. The Pearson-free
  statement: the two ends of the cost range bracket the middle of the pass
  range. Across the thirteen priced rows the per-task spread is **91.5×**
  (v4-pro 0.0534 to opus-4.8 4.8879), and neither end holds a rank near the top
  of the pass column.

Cache behaviour differs by vendor and is worth knowing before reading cost:
the three OpenAI models each report **50k–63k cached tokens per task**, while
**every Claude, Gemini and open-weight row reports zero cache reads
throughout**. Image cost is comparable across the eleven models that received
images (`imgs_step0` = 13.3; tokens-per-image: Gemini 2,207 / OpenAI 3,721 /
Claude 4,740 / kimi-k3 4,786); the three path-mode rows sit at 0.0 because they
were served file paths and never received an image at all.

---

## 2. Action-category mix

<!-- block:catmix.table -->
Total actions and their distribution across the 12 action categories defined in the reading guide, all 600 tasks at `reasoning_effort=high`. Percentages are share of that model's own total and sum to 100% by construction; each column's lone high and lone low are bold.

| Model | total | `asset_open` | `asset_inspect` | `file_read` | `file_list` | `file_write` | `file_delete` | `message_send` | `message_read` | `agent_delegate` | `agent_status` | `wait` | `other` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3.5-flash | **11,820** | 18.1% | 8.3% | **28.3%** | 5.2% | **1.6%** | 0.0% | **6.5%** | **22.9%** | 0.1% | 1.7% | 2.3% | 5.1% |
| 3.6-flash | 9,786 | 17.8% | 7.6% | 25.9% | **6.1%** | 6.1% | 0.0% | 16.4% | 12.3% | 0.1% | 0.6% | **0.6%** | 6.4% |
| 3.7-flash | 10,674 | **12.7%** | 8.0% | 26.4% | 6.0% | 5.1% | 0.0% | 19.1% | 11.1% | 0.2% | **3.3%** | 0.7% | 7.4% |
| 3.1-pro | 7,898 | 24.0% | 12.7% | 12.2% | 2.0% | 7.6% | 0.0% | 16.9% | 10.9% | 0.4% | 1.4% | **5.1%** | 7.0% |
| opus-4.8 | 7,303 | 22.3% | 9.6% | 19.4% | 3.0% | 8.6% | 0.0% | 21.6% | 5.8% | 0.4% | 0.5% | 1.7% | 7.2% |
| opus-5 | 6,672 | 26.8% | 9.1% | 16.7% | 1.4% | 10.3% | 0.0% | **23.5%** | **2.0%** | 0.2% | 0.3% | 1.3% | **8.3%** |
| sonnet-5 | **6,385** | 17.2% | **15.6%** | 17.6% | 1.1% | 9.9% | 0.0% | 20.9% | 6.2% | 0.2% | 0.5% | 3.4% | 7.3% |
| sol | 8,191 | 29.9% | 7.8% | 15.2% | 0.9% | 9.1% | 0.0% | 16.3% | 10.7% | 0.4% | 0.8% | 2.9% | 6.0% |
| luna | 8,125 | 28.8% | 8.3% | 17.7% | 1.2% | 10.5% | 0.0% | 18.3% | 6.4% | 0.2% | 0.6% | 2.5% | 5.4% |
| terra | 6,812 | 23.4% | 9.6% | 17.7% | 1.0% | **10.9%** | 0.0% | 19.1% | 7.6% | 0.3% | 0.9% | 3.9% | 5.7% |
| v4-pro | 8,051 | 32.5% | 7.6% | 13.9% | 1.6% | 7.5% | 0.0% | 19.1% | 9.5% | 0.3% | 0.6% | 2.0% | 5.6% |
| v4-flash | 10,674 | 33.7% | **5.9%** | 15.7% | 5.7% | 3.3% | 0.0% | 9.0% | 20.1% | 0.2% | 0.6% | 0.8% | **4.9%** |
| kimi-k3 | 9,251 | 43.6% | 7.4% | 16.0% | 1.7% | 5.7% | 0.0% | 10.7% | 6.7% | 0.2% | 0.4% | 1.4% | 6.1% |
| glm-5.2 | 8,933 | **48.8%** | 6.9% | **11.3%** | **0.8%** | 5.3% | 0.0% | 14.2% | 6.6% | 0.1% | **0.2%** | 1.0% | 5.0% |

One column is carried because the roll-up is MECE rather than because the field uses it: `file_delete` accounts for **2** actions in the entire corpus (luna; zero for the other 13 models). No action is dropped from the totals — the build fails if the category counts and the raw command counts disagree.

**Not every row here is like-for-like.** v4-pro, v4-flash and glm-5.2 were handed asset file paths as text and never received an image (`multimodal_input_mode=path`); kimi-k3, the open-source model that did receive images, is the control; also, 3.7-flash is reported on `r02`, the higher of its two complete runs, where every other row is a single run (the two differ by 1.7pp, McNemar p=0.27). Full statement: [run conditions](#run-conditions-that-are-not-comparable).
<!-- /block:catmix.table -->

The two ends of the field invert each other. 3.5-flash spends **51.2%** of its actions reading (`file_read` plus `message_read`) against **8.1%** producing (`file_write` plus `message_send`); opus-5 nearly reverses that, **18.8%** reading against **33.8%** producing. The field does not, however, lie on the line between them. Across the 14 models those two shares correlate at only Spearman **−0.34**, and the lightest reader is not opus-5 but **glm-5.2** at **17.9%**, which produces just **19.5%**. Reading little does not imply producing much. <!-- notes:catmix.contrast -->

What the mix does track is the vendor. The **four** heaviest users of `asset_open` are **glm-5.2, kimi-k3, v4-flash and v4-pro** — the whole **Open-source** block, at **48.8%, 43.6%, 33.7% and 32.5%** against **29.9%** for the next model down — and opening trades off against reading across the field (Spearman **−0.56**), which is what puts them at the bottom of the reading column without putting them at the top of the producing one. One column over the same thing happens to a different vendor: the top **three** `message_send` shares are the entire **Claude** block. <!-- notes:catmix.blocks -->

`other` barely moves: **4.9–8.3%** across the field, a **1.69×** spread, against **11.40×** for `message_read` and **21.22×** for `agent_status`. It collects the scenario-specific verb families — `approval`, `booking`, `containment`, `deploy`, `lock`, `pairing`, `review`, `system`, `tasks` and `undo`. <!-- notes:catmix.other -->

---

## 3. Command-surface deep-dive

> The `multiagent_vlm_conflict` action DSL is structured, so the **specific sub-command** a model reaches for is as diagnostic as the coarse category. One verb is near-universal: `asset list` is the step-0 action in **89.0–100.0%** of tasks for every model (opus-4.8 lowest at **89.0%**; 3.5-flash, 3.6-flash, 3.7-flash, sol, luna and terra open with it every single time), and the field issues it **601–662** times each across 600 tasks — a **1.10×** spread, the flattest verb in the DSL. After that shared opening the field splits into recognisably different working styles. Counts are total invocations across all 600 tasks at `reasoning_effort=high`. <!-- notes:command_surface.intro prefix="> " -->

<!-- block:command_surface.table -->
| Model | total | `asset open` | `asset list` | `file read` | `message search` | `file create` | `message send` |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3.5-flash | **11,820** | 2,137 | 604 | **3,342** | **2,707** | **187** | **747** |
| 3.6-flash | 9,786 | 1,740 | 606 | 2,538 | 1,206 | 593 | 1,593 |
| 3.7-flash | 10,674 | 1,351 | 603 | 2,814 | 1,188 | 547 | **1,704** |
| 3.1-pro | 7,898 | 1,892 | **662** | **966** | 861 | 581 | 1,283 |
| opus-4.8 | 7,303 | 1,625 | 604 | 1,416 | 427 | 617 | 1,534 |
| opus-5 | 6,672 | 1,786 | 604 | 1,117 | **134** | 669 | 1,502 |
| sonnet-5 | **6,385** | **1,100** | 616 | 1,125 | 397 | 612 | 1,334 |
| sol | 8,191 | 2,448 | 619 | 1,248 | 877 | 708 | 1,337 |
| luna | 8,125 | 2,341 | 629 | 1,437 | 522 | **773** | 1,480 |
| terra | 6,812 | 1,592 | 610 | 1,206 | 516 | 727 | 1,286 |
| v4-pro | 8,051 | 2,618 | 612 | 1,117 | 760 | 595 | 1,533 |
| v4-flash | 10,674 | 3,601 | 607 | 1,679 | 2,137 | 335 | 961 |
| kimi-k3 | 9,251 | 4,030 | **601** | 1,476 | 621 | 514 | 992 |
| glm-5.2 | 8,933 | **4,355** | 606 | 1,013 | 587 | 468 | 1,265 |

Counts are total invocations across all 600 tasks at `reasoning_effort=high`; each column's high and low are bold. The complete per-command inventory is in the command appendix of the HTML report.

**Not every row here is like-for-like.** v4-pro, v4-flash and glm-5.2 were handed asset file paths as text and never received an image (`multimodal_input_mode=path`); kimi-k3, the open-source model that did receive images, is the control; also, 3.7-flash is reported on `r02`, the higher of its two complete runs, where every other row is a single run (the two differ by 1.7pp, McNemar p=0.27). Full statement: [run conditions](#run-conditions-that-are-not-comparable).
<!-- /block:command_surface.table -->

- **3.5-flash interrogates the workspace instead of committing to it — but it no longer leads every inspection verb.** It still leads `file read` (**3342**, **5.57**/task, 3.46× the field floor at 3.1-pro's **966**) and `message search` (**2707**, **4.51**/task, **20.9%** of every search the field runs). It also posts the field's lowest `file create` (**187**, **0.31**/task) and lowest `message send` (**747**). What has changed since the nine-model cut is the rest of the inspection surface: `asset info` now belongs to sonnet-5 (**330** against 3.5-flash's **306**), and `file list` and `agent list` both belong to 3.7-flash (**640** and **325**). The reading is still a substitute for acting rather than grounding for it: **535** of 600 tasks hit the 20-step cap and it finishes at **3.0%**, the field floor. <!-- notes:command_surface.bullets.0 prefix="- " -->
- **The submission verbs are far flatter than the investigation verbs.** `asset list` spans just **601–662** across all 14 models (**1.10×**), `file create` and `message send` look wider at **4.13×** and **2.28×**, but that is 3.5-flash alone dragging both floors down: hold it out and they tighten to **335–773** (**2.31×**) and **961–1704** (**1.77×**). Every model knows what a finished task looks like. The spread is upstream of the deliverable: `message search` runs **134–2707** (**20.20×**), `asset open` **1100–4355** (**3.96×**), `file read` **966–3342** and `asset info` **1–330**. Total actions differ by only **1.85×** (3.5-flash **11,820** against sonnet-5's **6,385**), so the models are not doing wildly different amounts of work — they are doing differently shaped work. <!-- notes:command_surface.bullets.1 prefix="- " -->
- **Search falls as capability rises, but not monotonically — and the exception is the second-best model in the field.** `message search` runs from 3.5-flash's **2707** (pass **3.0%**) down to opus-5's **134** (pass **25.2%**), a **20.20×** range, and across the 14 models search count and pass rate correlate at Spearman **−0.63**. That is a real association and it is not a rule: sol passes **24.5%** — second in the field — while running **877** searches — more than glm-5.2 (**587**, **14.7%**) and 3.1-pro (**861**, **12.0%**), both of which score far below it. The clean version of the claim is narrower: heavy search reliably predicts a **low** score, light search predicts nothing. opus-5 sits at the light extreme on every inspection verb — **5** `asset info`, **0** `asset describe`, **0** `asset read-text` — and still tops the field. <!-- notes:command_surface.bullets.2 prefix="- " -->
- **Grounding volume is not the axis it looks like.** sonnet-5 opens the fewest assets in the field (**1100×**, **1.83**/task) and leaves **92** of 600 tasks having opened nothing at all, yet passes **20.8%**, well above both 3.1-pro (**3.15** opens/task, **12.0%**) and 3.5-flash (**3.56** opens/task, **3.0%**). At the other end glm-5.2 opens **7.26**/task — the heaviest in the field — and passes **14.7%**, below every Claude and every OpenAI model. Across the resulting **3.97×** range in opens per task the pass rate moves in no consistent direction (Spearman **−0.29**). Part of this is the rubric rather than the models: only **346** of the corpus's **822** `assets_accessed` checks carry `role: required` and are scored at all. <!-- notes:command_surface.bullets.3 prefix="- " -->
- **Waiting is a family trait, and the Claude floor is the real divide.** Neither opus model ever issues a wait shorter than **20 s** (opus-4.8: **121** calls, mean **55.5 s**; opus-5: floor **30 s**, **90** calls, mean **83.2 s** — the highest mean in the field), while all three OpenAI models bottom out at **1 s** and average **10.0–14.6 s**. sonnet-5 does not follow its family to the floor — it goes as low as **5 s** — but it is nonetheless the field's heaviest waiter in aggregate, **9,110 s** over **215** calls, and the model behind the longest single wait anywhere in the corpus (**600 s**, reached by sonnet-5 alone). 3.1-pro issues the most wait calls (**402**) but keeps them short, mean **17.5 s**. <!-- notes:command_surface.bullets.4 prefix="- " -->

> One verb is shared and the rest diverge. `asset list` opens **89.0–100.0%** of every model's tasks; from there 3.5-flash reads (**3342** `file read`s, **2707** `message search`es) instead of writing (**187** `file create`s) and passes **3.0%**, while opus-5 runs **134** `message search`es and **5** `asset info`, writes **669** reports, and passes **25.2%**. The submission surface is nearly identical across the field — `asset list` spans just **1.10×** — while the investigation surface runs to **20.20×** on `message search` and **330.00×** on `asset info`. On this corpus the command surface is not measuring diligence — it is measuring whether a model knows when it has seen enough. The corpus permits that reading only so far: just **346** of its **822** `assets_accessed` checks are scored at all, so a model can take full marks on a visual-conflict task without opening a single image — and **28** episodes do, spread across **17** scenarios and confined to 3.7-flash, 3.1-pro and sonnet-5. None of them is one of the eight walkthroughs below. <!-- notes:command_surface.kicker prefix="> " -->

---

## 4. Grounding × Action map

<!-- block:grounding.table -->
Grounding volume against outcome over all 600 tasks at `reasoning_effort=high`, **ordered by opens/task** rather than by provider: this is the one table here whose row order carries an argument, so it is sorted along the axis being tested. `zero-open tasks` counts episodes in which the model never issued a single `asset open`.

| Model | opens/task | zero-open tasks | pass |
|---|---:|---:|---:|
| sonnet-5 | 1.83 | **92** | 20.8% |
| 3.7-flash | 2.25 | 60 | 20.3% |
| terra | 2.65 | 1 | 20.0% |
| opus-4.8 | 2.71 | 1 | 22.8% |
| 3.6-flash | 2.90 | 5 | 18.5% |
| opus-5 | 2.98 | 1 | **25.2%** |
| 3.1-pro | 3.15 | 14 | 12.0% |
| 3.5-flash | 3.56 | 18 | **3.0%** |
| luna | 3.90 | 0 | 19.8% |
| sol | 4.08 | 0 | 24.5% |
| v4-pro | 4.36 | 0 | 19.8% |
| v4-flash | 6.00 | 0 | 8.5% |
| kimi-k3 | 6.72 | 3 | 21.8% |
| glm-5.2 | 7.26 | 0 | 14.7% |

**Not every row here is like-for-like.** v4-pro, v4-flash and glm-5.2 were handed asset file paths as text and never received an image (`multimodal_input_mode=path`); kimi-k3, the open-source model that did receive images, is the control; also, 3.7-flash is reported on `r02`, the higher of its two complete runs, where every other row is a single run (the two differ by 1.7pp, McNemar p=0.27). Full statement: [run conditions](#run-conditions-that-are-not-comparable).
<!-- /block:grounding.table -->

Two models treat opening an asset as optional. sonnet-5 finishes **92** of 600 tasks without a single `asset open` and 3.7-flash **60**; **12** of the 14 never exceed **18**, and **5** of them — sol, luna, v4-pro, v4-flash and glm-5.2 — open an asset on every task. §3 covers why the pass column does not follow the opens column (Spearman **−0.29** over a **3.97×** range) and why most of the grounding here is unscored. What the ordering adds is that the field's three strongest models, opus-5, sol and opus-4.8, are scattered along the axis at **2.98, 4.08 and 2.71** opens per task. No particular amount of grounding characterises a good model. <!-- notes:grounding.spread -->

The other half of the answer is that an open is not evidence of grounding that gets used. 3.5-flash's **3.56** opens per task arrive alongside **5.57** `file read`s and **4.51** `message search`es per task and a write rate of **0.31** — the opens are part of a search that never terminates, not grounding that feeds a deliverable. <!-- notes:grounding.mechanism -->

---

## 5. Outcome and stop-reason mix

<!-- block:stops.table -->
Episode outcomes over all 600 tasks at `reasoning_effort=high`. `none` means the episode ended cleanly with the model declaring itself finished; `provider_failure` means the provider killed the call mid-episode and is the one column here that is not a decision the model made. The columns are exhaustive: every episode lands in exactly one, and the build fails if the row does not sum to 600.

| Model | none | missing_required_side_effect | incomplete_required_submission | reached max_steps | 3-identical loop | empty reply | provider_failure |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3.5-flash | **18** | **26** | **19** | **535** | 1 | 1 | 0 |
| 3.6-flash | 111 | 171 | 170 | 148 | 0 | 0 | 0 |
| 3.7-flash | 122 | 128 | 106 | 244 | 0 | 0 | 0 |
| 3.1-pro | 72 | 203 | 168 | 152 | 0 | 0 | **5** |
| opus-4.8 | 137 | 233 | 193 | 37 | 0 | 0 | 0 |
| opus-5 | **151** | 239 | 204 | **3** | 0 | 3 | 0 |
| sonnet-5 | 125 | 255 | 200 | 20 | 0 | 0 | 0 |
| sol | 147 | 238 | 208 | 7 | 0 | 0 | 0 |
| luna | 119 | 226 | 194 | 55 | 6 | 0 | 0 |
| terra | 120 | **261** | **210** | 8 | 1 | 0 | 0 |
| v4-pro | 119 | 218 | 182 | 81 | 0 | 0 | 0 |
| v4-flash | 51 | 103 | 85 | 358 | 2 | 1 | 0 |
| kimi-k3 | 131 | 165 | 132 | 170 | 0 | 2 | 0 |
| glm-5.2 | 88 | 156 | 139 | 207 | 6 | **4** | 0 |

**Not every row here is like-for-like.** v4-pro, v4-flash and glm-5.2 were handed asset file paths as text and never received an image (`multimodal_input_mode=path`); kimi-k3, the open-source model that did receive images, is the control; also, 3.7-flash is reported on `r02`, the higher of its two complete runs, where every other row is a single run (the two differ by 1.7pp, McNemar p=0.27). Full statement: [run conditions](#run-conditions-that-are-not-comparable).
<!-- /block:stops.table -->

There is no single dominant failure mode. `missing_required_side_effect` is the largest outcome corpus-wide — **2,622** episodes, against **2,210** for `incomplete_required_submission` and **2,025** for the step cap — and it is the largest non-clean outcome for **9** of the 14 models: 3.6-flash, 3.1-pro, opus-4.8, opus-5, sonnet-5, sol, luna, terra and v4-pro. For the other **5** — 3.5-flash, 3.7-flash, v4-flash, kimi-k3 and glm-5.2 — the step cap dominates instead. Those are different diseases: one is a model that finishes and leaves the world unchanged, the other is a model that never finishes at all (§6). Where the side-effect failure dominates it has a consistent shape — a correct written analysis and no state change. §10's `coordinated_containment_v02` and `distributed_lock_shared_resource_v01` are it in miniature: in the lock round **9** of the 14 models stop this way, and the **4** that do not are 3.5-flash, 3.6-flash, 3.7-flash and v4-flash. <!-- notes:stops.regimes -->

The shortest episodes in the field belong to **sonnet-5** — **10.6** steps on average — and it pairs them with **255** side-effect misses, behind only **terra**'s **261**. It stops early and confidently. That is the opposite pathology to 3.5-flash, which stops **535** times only because the harness makes it. <!-- notes:stops.sonnet -->

The `none` column is not a measure of tidiness. An episode is recorded as `none` exactly when it passed every required check, so `none` and the pass count are the same quantity: they agree exactly for **14** of the 14 models. Holding that identity took one repair. **3.1-pro** lost **5** episodes to an HTTP 504 provider timeout, and the harness files those with no stop reason recorded and a step count of zero even though their trajectories still hold the 57 actions the model had already taken — so they used to land in this column and inflate it by exactly 5. They now sit in their own `provider_failure` bucket with their step counts recovered from the trajectory. They are still scored as zeros and counted against 3.1-pro in every pass rate in this report, so that every model is reported over the same 600 tasks; dropping them instead would read **12.1%** over 595 episodes. <!-- notes:stops.none -->

---

## 6. Loops and termination

<!-- block:steps.table -->
Step behaviour over all 600 tasks at `reasoning_effort=high`. **Step errors** is the share of steps that returned a non-zero exit code or carried an error tag. **Tail waste** is the number of steps taken after the last step that actually gained score, so it is defined only for episodes that gained score at all: the column averages over 412–582 of the 600 episodes depending on the model (3.5-flash lowest, at 412), and says nothing about the rest.

| Model | mean steps | p50 | p50 on success | p50 on failure | tail waste (mean) | step errors |
|---|---:|---:|---:|---:|---:|---:|
| 3.5-flash | **19.7** | 20.0 | **18.0** | 20.0 | **6.6** | **0.3%** |
| 3.6-flash | 16.3 | 16.5 | 15.0 | 17.0 | 1.6 | 0.7% |
| 3.7-flash | 17.8 | 19.0 | 17.0 | 20.0 | 1.9 | 0.6% |
| 3.1-pro | 13.2 | 12.0 | 10.0 | 13.0 | 3.0 | 1.1% |
| opus-4.8 | 12.2 | 12.0 | 11.0 | 12.0 | 1.6 | 0.7% |
| opus-5 | 11.1 | 11.0 | 10.0 | 11.0 | 1.5 | 0.9% |
| sonnet-5 | **10.6** | **10.0** | 10.0 | **10.0** | **1.3** | 0.9% |
| sol | 13.7 | 14.0 | 12.0 | 15.0 | 1.6 | 1.0% |
| luna | 13.5 | 13.0 | 11.0 | 14.0 | 2.5 | 1.2% |
| terra | 11.4 | 11.0 | 10.0 | 11.0 | 1.8 | 1.0% |
| v4-pro | 13.4 | 13.0 | 12.0 | 13.0 | 1.6 | 1.2% |
| v4-flash | 17.8 | 20.0 | 15.0 | 20.0 | 4.9 | 0.5% |
| kimi-k3 | 15.4 | 16.0 | 14.0 | 17.0 | 2.8 | 0.4% |
| glm-5.2 | 14.9 | 15.0 | 13.0 | 16.0 | 3.3 | 0.7% |

**Not every row here is like-for-like.** v4-pro, v4-flash and glm-5.2 were handed asset file paths as text and never received an image (`multimodal_input_mode=path`); kimi-k3, the open-source model that did receive images, is the control; also, 3.7-flash is reported on `r02`, the higher of its two complete runs, where every other row is a single run (the two differ by 1.7pp, McNemar p=0.27). Full statement: [run conditions](#run-conditions-that-are-not-comparable).
<!-- /block:steps.table -->

**The cap is fatal.** Across the 14 models at `reasoning_effort=high`, **2,025** episodes hit the 20-step cap and **0** of them passed, against **1,511** passes in **6,375** uncapped episodes (**23.70%**). Being capped is not “scored lower”; it is a guaranteed non-pass. It is not the same as scoring nothing, though — capped episodes average **0.29** and the best of them reached **0.91**. They run out of steps while still collecting credit. <!-- notes:steps.cap -->

That reframes 3.5-flash. Its **3.0%** pass rate is not first of all a comprehension result: **89.2%** of its episodes were structurally ineligible to pass before scoring began. §9 shows the effect is an effort artefact — at `low` the same model caps on **38.7%** of episodes and passes **13.5%**. <!-- notes:steps.flash -->

Tail waste is small and tight for everyone except 3.5-flash: **6.6** steps after the last step that actually gained score, against a field floor of **1.3**, and a median of **4.0** against **1.0**. The waste is not repetition, though. The harness's three-identical-transitions detector fires **16** times in the entire 14-model corpus, and exactly **1** of those is 3.5-flash — the model that hit the step cap **535** times. Whatever it spends those steps on, it is not an action the detector can recognise as a loop: it never converges, but it never visibly stalls either. <!-- notes:steps.tail -->

---

## 7. Failure clustering: the shape of a model's losses

### 7.1 Concentration of losses

If a model's failed checks were independent coin flips, its pass rate would be
the product of its per-check rates. It is not: failures cluster on tasks.
`lift` = observed pass rate ÷ pass rate predicted from independent per-check
rates. Higher lift means losses are concentrated on fewer tasks — the model is
either right about a task or wrong about it, rather than dropping units at
random.

<!-- block:concentration.table -->
Mean scored units per task is 5.47. Computed over 600 tasks / 3,284 checks — except 3.1-pro on 595 / 3,257, which lost an episode to a provider failure and is scored on what it was actually given. **Ordered by lift**, which is the column the section is about: observed pass rate over the rate an independent-failure model predicts from the same per-check accuracy. Lift above 1 means a model is either right about a task or wrong about it rather than shedding units at random.

| Model | per-check pass | observed pass | if independent | excess | lift | 95% CI | overdispersion |
|---|---:|---:|---:|---:|---:|---:|---:|
| opus-5 | **63.0%** | **25.2%** | **10.8%** | +14.4pp | **1.49** | [1.39, 1.59] | **2.23** |
| sol | 62.2% | 24.5% | 10.2% | +14.3pp | 1.42 | [1.33, 1.51] | 2.05 |
| opus-4.8 | 60.9% | 22.8% | 9.3% | +13.5pp | 1.41 | [1.33, 1.50] | 2.08 |
| kimi-k3 | 55.8% | 21.8% | 6.5% | **+15.3pp** | 1.40 | [1.33, 1.47] | 2.11 |
| sonnet-5 | 58.2% | 20.8% | 7.7% | +13.1pp | 1.39 | [1.32, 1.47] | 2.13 |
| terra | 59.3% | 20.0% | 8.3% | +11.7pp | 1.37 | [1.30, 1.45] | 2.01 |
| luna | 59.8% | 19.8% | 8.6% | +11.2pp | 1.36 | [1.28, 1.44] | 1.93 |
| v4-pro | 57.7% | 19.8% | 7.4% | +12.4pp | 1.35 | [1.28, 1.41] | 1.95 |
| 3.7-flash | 57.5% | 20.3% | 7.3% | +13.0pp | 1.30 | [1.24, 1.36] | 1.83 |
| 3.6-flash | 58.0% | 18.5% | 7.6% | +10.9pp | 1.28 | [1.23, 1.34] | 1.79 |
| glm-5.2 | 48.2% | 14.7% | 3.6% | +11.0pp | 1.27 | [1.22, 1.32] | 1.97 |
| 3.1-pro | 48.8% | 12.1% | 3.8% | +8.3pp | 1.19 | [1.15, 1.24] | 1.66 |
| v4-flash | 38.8% | 8.5% | 1.6% | +6.9pp | 1.16 | [1.12, 1.19] | 1.56 |
| 3.5-flash | **28.1%** | **3.0%** | **0.5%** | **+2.5pp** | **1.05** | [1.03, 1.08] | **1.13** |

**Not every row here is like-for-like.** v4-pro, v4-flash and glm-5.2 were handed asset file paths as text and never received an image (`multimodal_input_mode=path`); kimi-k3, the open-source model that did receive images, is the control; also, 3.7-flash is reported on `r02`, the higher of its two complete runs, where every other row is a single run (the two differ by 1.7pp, McNemar p=0.27). Full statement: [run conditions](#run-conditions-that-are-not-comparable).
<!-- /block:concentration.table -->

**opus-5 leads both columns (63.0% per check, 25.2% pass), and the two columns
still disagree about the rest of the field.** kimi-k3 is 2.2pp *worse* per check
than 3.6-flash (55.8% against 58.0%) and 3.3pp *better* on pass rate (21.8%
against 18.5%), because its lift is 1.40 against 3.6-flash's 1.28; glm-5.2 is
0.6pp worse per check than 3.1-pro and 2.6pp better on pass rate, at lift 1.27
against 1.19. Fifteen such inversions exist in the panel, though none survives a
Bonferroni-corrected McNemar test on its own (widest: kimi-k3 over 3.6-flash,
p = 0.07), so the ordering disagreement is a property of the column, not a
verdict on any one pair. The column that separates models *significantly* is
lift, not per-check accuracy — see the peer test below.

**3.5-flash's lift of 1.05 is the most diagnostic number in this section.** Its
losses are almost perfectly independent — it is not failing *tasks*, it is
dropping *units*, more or less at random, across nearly every task. Every other
model's failures cluster (1.16–1.49), which is what competence looks like: you
either understood the conflict or you didn't.

**Paired peer test.** Resampling the 595 common rounds once per draw (4,000
draws), recomputing every model on that same resample, and comparing each
model's lift against the mean of its thirteen peers — corpus difficulty cancels
because all fourteen models are scored on the same rounds:

- **3.5-flash: diff −0.288, 95% CI [−0.338, −0.242], p = 0.0 → verdict `down`.**
  The most independent failure pattern in the panel, and the extreme of a group:
  v4-flash (−0.177), 3.1-pro (−0.136) and glm-5.2 (−0.053) also sit significantly
  below their peers. Those four are exactly the four weakest models on pass rate
  (3.0%, 8.5%, 12.1%, 14.7%), which is the point — shedding units at random and
  losing tasks are the same thing seen from two ends.
- **3.6-flash: diff −0.036, p = 0.09 → flat.** One of only three models (with
  v4-pro and 3.7-flash) that the test cannot distinguish from the peer pool. The
  other eleven separate — seven above, led by opus-5 (+0.188) and sol (+0.116),
  and four below. Concentration is not a shared property of the field; it is
  where the field spreads out.

**Gap decomposition, opus-5 vs sol** (n = 595, 12 peers, 4,000 draws, terciles
of 198 / 198 / 199 tasks) — the two models are 0.50pp apart on pass rate.
Decomposing that gap:

| decomposition | per-check component | clustering component | total |
|---|---:|---:|---:|
| pooled | +0.58 | −0.07 | 0.50 |
| stratified | −2.33 | +2.83 | 0.50 |

Pooled and stratified disagree in sign on the ability term (+0.58 against
−2.33). Pooling mixes easy and hard rounds, where the two models' relative
strengths differ; the stratified split is the one to read.

**This pair is a demonstration of the instrument, not a finding about the two
models.** opus-5 and sol are 0.50pp apart on pass rate over the 595 common
rounds, and that gap is indistinguishable from zero (exact McNemar b = 48,
c = 45, p = 0.84, against a Bonferroni α of 0.05/91 = 0.00055), so the partition
above describes what the decomposition does rather than what separates these
two. Read that way it is still informative: stratified, opus-5 is 2.33pp *worse*
per check and recovers 2.83pp by concentrating its failures, and the
concentration difference is real where the pass difference is not — paired lift
diff +0.96 [0.35, 1.56] on the easy tercile and +0.37 [0.15, 0.55] on the
middle, against +0.02 [−0.02, 0.06] on the hard. Overall lift 1.50 vs 1.43,
paired diff 0.07 [−0.02, 0.15]. The tier profile has also inverted since the
medium build: opus-5 is now 0.5pp *behind* on the easy tercile (56.6% vs 57.1%)
and 1.0pp ahead on each of the other two (16.7% vs 15.7%; 2.5% vs 1.5%).

### 7.2 Check-type split

<!-- block:check_split.table -->
`graded_pass_pct` over all 600 tasks at `reasoning_effort=high`, **ordered by the judged column**. Rates are over units the checker actually ran on: `field_semantic_match` units go ungraded when the artefact they would grade was never produced, which happens 13–1,222 times per model, and counting those as wrong answers would charge the same episode twice.

| Model | `field_semantic_match` | `field_contains` | `other` |
|---|---:|---:|---:|
| kimi-k3 | **65.7** | 84.9 | **42.6** |
| opus-5 | 62.7 | **87.9** | 38.2 |
| 3.7-flash | 62.2 | 80.7 | 40.4 |
| sol | 62.1 | 87.6 | 38.0 |
| 3.6-flash | 61.6 | 83.4 | 38.5 |
| opus-4.8 | 61.1 | 83.2 | 39.3 |
| luna | 60.9 | 83.8 | 34.7 |
| terra | 60.4 | 79.6 | 33.8 |
| v4-pro | 60.0 | 84.1 | 33.2 |
| sonnet-5 | 59.1 | 73.6 | 36.1 |
| glm-5.2 | 58.9 | 83.3 | 28.3 |
| v4-flash | 53.9 | 79.1 | 27.7 |
| 3.1-pro | 46.4 | 79.9 | 36.2 |
| 3.5-flash | **45.6** | **70.8** | **26.8** |

**Not every row here is like-for-like.** v4-pro, v4-flash and glm-5.2 were handed asset file paths as text and never received an image (`multimodal_input_mode=path`); kimi-k3, the open-source model that did receive images, is the control; also, 3.7-flash is reported on `r02`, the higher of its two complete runs, where every other row is a single run (the two differ by 1.7pp, McNemar p=0.27). Full statement: [run conditions](#run-conditions-that-are-not-comparable).
<!-- /block:check_split.table -->

Two anomalies:

- **sonnet-5's `field_contains` (73.6) sits only 14.5 points above its
  `semantic_match` (59.1)** — the narrowest gap in the field, where every other
  model runs 18.5–33.5 points above. That is the numeric shadow of its 92
  zero-open tasks (see the caveat below).
- **3.5-flash's `semantic_match` graded rate (45.6) is respectable** while its
  raw rate is 19.5, because 1,222 of its 2,137 semantic checks were
  `not_invoked` — it never produced the artefact to grade. Its problem is
  emitting deliverables, not reasoning about them.

> [!WARNING]
> **The pooled `field_contains` column is not what it looks like.** Of its 1,078
> checks, **822 (76.3%) are `assets_accessed` gates** — "did you open the right
> asset" — and only **256 are literal-content checks**. The two are disjoint and
> sum exactly to the column, so the pooled number is a 76/24 weighted average
> that mostly reports grounding, not transcription. **476 of the 822 asset gates
> carry `role: diagnostic` and never count toward score**; the other 346 and all
> 256 content checks are `required`. Read the split below, not the pooled value.

### 7.3 `field_contains`, decomposed

<!-- block:fc_decomp.table -->
The deterministic column decomposed: **asset gate** (814–822 units) is *did you open the right file*, **literal content** (254–256 units) is *does what you wrote contain this exact string*. `graded == total` for both on every model — these are the units no judge grades — so nothing here is exposed to the attrition that distorts the judged column. **Ordered by the pooled rate.**

| Model | asset gate | literal content | pooled | content − asset |
|---|---:|---:|---:|---:|
| opus-5 | 89.7 | **82.4** | **87.9** | −7.3 |
| sol | 91.2 | 75.8 | 87.6 | −15.4 |
| kimi-k3 | 92.5 | 60.5 | 84.9 | −32.0 |
| v4-pro | 88.9 | 68.8 | 84.1 | −20.1 |
| luna | 86.3 | 75.8 | 83.8 | −10.5 |
| 3.6-flash | 87.8 | 69.1 | 83.4 | −18.7 |
| glm-5.2 | **93.2** | 51.6 | 83.3 | −41.6 |
| opus-4.8 | 84.8 | 78.1 | 83.2 | −6.7 |
| 3.7-flash | 84.9 | 67.2 | 80.7 | −17.7 |
| 3.1-pro | 84.4 | 65.4 | 79.9 | −19.0 |
| terra | 80.9 | 75.4 | 79.6 | −5.5 |
| v4-flash | 91.8 | 38.3 | 79.1 | −53.5 |
| sonnet-5 | **72.6** | 76.6 | 73.6 | **+4.0** |
| 3.5-flash | 87.3 | **17.6** | **70.8** | **−69.7** |

**Not every row here is like-for-like.** v4-pro, v4-flash and glm-5.2 were handed asset file paths as text and never received an image (`multimodal_input_mode=path`); kimi-k3, the open-source model that did receive images, is the control; also, 3.7-flash is reported on `r02`, the higher of its two complete runs, where every other row is a single run (the two differ by 1.7pp, McNemar p=0.27). Full statement: [run conditions](#run-conditions-that-are-not-comparable).
<!-- /block:fc_decomp.table -->

**The two sub-columns pull against each other.** Spearman ρ between the two
rankings is **−0.40** (n=14), and dropping 3.5-flash strengthens it to
**−0.47** rather than explaining it away. Finding the evidence and writing it
down correctly are separate abilities on this corpus, and on this panel they are
mildly *opposed*: the gate's three best models — glm-5.2, kimi-k3, v4-flash —
rank 12th, 11th and 13th on content. The pooled column averages the two into a
number that describes neither.

- **3.5-flash inverts completely.** It scores **87.3** on the asset gate — 8th of
  fourteen, ahead of opus-4.8 — and **17.6** on literal content, last by **20.7
  points**. The pooled 70.8 splits the difference and is wrong in both
  directions. This is the same model that spends **51.2%** of its actions reading
  and writes **0.31** files per task: it locates the evidence and then does not
  write it down. Its ceiling is an output problem, not a perception one.
- **sonnet-5 is the exact mirror.** **72.6** on the asset gate, last by **8.3
  points**, against **76.6** on literal content — 3rd, behind only the two opus
  models. Its pooled **73.6** (13th of fourteen, ahead only of 3.5-flash) is
  largely an artefact of not opening assets: **1.83** opens/task and **92**
  zero-open tasks. Nothing here says it transcribes badly.
- **No model is in the top three of both.** The gate's top three (glm-5.2 93.2,
  kimi-k3 92.5, v4-flash 91.8) and content's top three (opus-5 82.4, opus-4.8
  78.1, sonnet-5 76.6) are disjoint sets. opus-5 tops the pooled column not by
  leading both but by pairing 1st on content with 5th on the gate — the best
  combined position in the panel, and the per-check shape behind its overall
  lead. sol is the only model in the top four of both.
- **sol is 4th on the asset gate (91.2) and tied 4th on content (75.8);** its
  87.6 is the *pooled* figure, second only to opus-5. It opens **4.08** assets
  per task — more than every model except v4-pro, v4-flash, kimi-k3 and
  glm-5.2 — and converts less of what it opens, a **−15.4** gap. That is only
  the 9th widest of fourteen: seven models other than 3.5-flash spread wider, so
  the shape is ordinary for this panel rather than diagnostic of sol.
- terra is **80.9 / 75.4**, a −5.5 gap. The flattest model is sonnet-5, and it
  is the only one whose content rate *exceeds* its gate rate (**+4.0**).

---

## 8. Difficulty landscape

**The v8 specs carry no usable intrinsic difficulty label.** This is worth
stating plainly because it defeats the obvious analysis. Across all 600 specs:

| field | cardinality | usable? |
|---|---|---|
| `difficulty` | 1 (every task is `3`) | no |
| `domains` | 1 (`composite`, `conflict`) | no |
| `template_id` | 1 | no |
| `weight` | 1 (`1.0`) | no |
| `hidden_constraints` / `visible_constraints` | 0 on every task | no |
| `decision_requirements` | 1 on every task | no |
| `availability_tier` / `online_requirement` | null on every task | no |
| `realism_tags` | 5, but = `conflict_vlm_v1` + `round_type` | redundant |
| `conflict_taxonomy` | **absent from v8 entirely** | no |
| `round_type` | 5 (one with n=1) | **yes** |
| `required_modalities` | 2 | yes, with caveats |
| `allowed_tool_families` | 10 | yes, confounded with scenario |
| **scored-unit count** | 2–14 per task | **yes — the best axis** |

### 8.1 Scored-unit count is the one clean difficulty axis

<!-- block:unit_bands.table -->
Banding the 600 tasks by number of **scored units** (2–4 / 5–6 / 7+; n = 192 / 258 / 150; total 3,284 units) — except 3.1-pro on 190 / 257 / 148, five episodes short, lost to a provider failure and carrying no rubric to band, so those tasks are dropped rather than scored as misses. Pass rate falls monotonically across the three bands for thirteen of fourteen models (3.5-flash is the exception), pooled 26.2% → 9.3%, a 16.9pp spread. Mean score does the same for one of fourteen models (sol), so the axis is clean on all-or-nothing passing and not on partial credit.

| Model | 2–4 pass | 5–6 pass | 7+ pass | 2–4 score | 5–6 score | 7+ score |
|---|---:|---:|---:|---:|---:|---:|
| 3.5-flash | **3.1** | **3.5** | **2.0** | **0.160** | **0.303** | **0.311** |
| 3.6-flash | 29.7 | 17.1 | 6.7 | 0.585 | 0.614 | 0.538 |
| 3.7-flash | 31.8 | 19.4 | 7.3 | 0.572 | 0.619 | 0.528 |
| 3.1-pro | 20.0 | 10.1 | 5.4 | 0.469 | 0.523 | 0.464 |
| opus-4.8 | 33.3 | 21.7 | 11.3 | 0.638 | 0.652 | 0.543 |
| opus-5 | 34.4 | **24.0** | 15.3 | **0.655** | **0.670** | 0.576 |
| sonnet-5 | 32.8 | 18.2 | 10.0 | 0.621 | 0.623 | 0.518 |
| sol | **35.9** | 21.3 | 15.3 | 0.654 | 0.645 | **0.584** |
| luna | 28.6 | 19.8 | 8.7 | 0.606 | 0.627 | 0.561 |
| terra | 28.6 | 19.8 | 9.3 | 0.618 | 0.619 | 0.553 |
| v4-pro | 27.6 | 18.2 | 12.7 | 0.571 | 0.608 | 0.548 |
| v4-flash | 9.9 | 9.3 | 5.3 | 0.279 | 0.444 | 0.380 |
| kimi-k3 | 29.2 | 21.7 | 12.7 | 0.536 | 0.594 | 0.529 |
| glm-5.2 | 22.4 | 12.8 | 8.0 | 0.468 | 0.518 | 0.457 |
| **pooled** | 26.2 | 16.9 | 9.3 | 0.531 | 0.576 | 0.506 |

**Not every row here is like-for-like.** v4-pro, v4-flash and glm-5.2 were handed asset file paths as text and never received an image (`multimodal_input_mode=path`); kimi-k3, the open-source model that did receive images, is the control; also, 3.7-flash is reported on `r02`, the higher of its two complete runs, where every other row is a single run (the two differ by 1.7pp, McNemar p=0.27). Full statement: [run conditions](#run-conditions-that-are-not-comparable).
<!-- /block:unit_bands.table -->

**Pass rate falls monotonically with unit count for 13 of 14 models** (3.5-flash
the exception), pooled 26.2% → 9.3%, a 16.9pp spread. Part of this is
mechanical — `success` requires *all* units — but the effect survives on the
fractional score too, just not monotonically:

**Mean score is monotone for only 1 of 14 models** (sol, decreasing). Twelve of
the other thirteen *peak in the middle band*; 3.5-flash alone rises
(0.160 → 0.303 → 0.311). In the 2–4 band a single missed
unit costs ~29% of the score, so partial credit is coarse there even though
all-or-nothing passing is easiest. Reporting only pass rate would have hidden
this.

**3.5-flash inverts on both measures**, and the mechanism is measurable. Its
absolute satisfied-unit count grows **0.60 → 1.67 → 2.51 (4.2×)** across bands
while the rubric grows **3.51 → 5.44 → 8.05 (2.3×)** — super-proportional. Every
other model is sub-proportional; opus-5 goes 2.28 → 3.65 → 4.60 (2.0×) and its
score therefore falls. 3.5-flash's score is a function of how many units were on
the table, not of how many it understood. This is the same fact as its 1.05
clustering lift (§7), seen from a different angle.

### 8.2 Round type carries signal *against* its rubric size

| round_type | n | mean units | 2–4 band | 5–6 band | 7+ band | marginal |
|---|---:|---:|---:|---:|---:|---:|
| investigation | 274 | 4.91 | 27.5 [119] | 13.4 [112] | **2.3** [43] | 17.8 |
| verification | 174 | 5.83 | **28.6** [39] | 18.1 [80] | 11.2 [55] | 18.3 |
| action | 86 | 6.55 | 21.0 [17] | **26.7** [31] | 13.7 [38] | **19.9** |
| decision | 65 | 5.49 | 17.4 [16] | 16.7 [35] | 11.2 [14] | 15.7 |

*(A fifth value, `reconciliation`, has n = 1 and is excluded as noise.)*

**`action` rounds are the easiest in both hard bands despite carrying the most
units** (6.55 mean), and **`investigation` rounds are the hardest in both hard
bands despite carrying the fewest** (4.91). The two axes push in opposite
directions, which is why the marginal column understates the effect.

**The benchmark's floor is `investigation` × 7+ units: 2.3% pooled pass across
43 tasks × 14 models — 14 successes out of 602 attempts.**

Do **not** read the marginal column as "decision is hardest". Decision is worst
only in the 2–4 band; in both harder bands investigation is worse. The marginal
15.7% is a composition artefact.

### 8.3 Natural images do not make tasks harder — and the raw number lies

Marginally, tasks requiring a natural image pass at **22.3%** versus **16.9%**
for chart-only tasks — apparently *easier*. Stratified by scored-unit count:

| band | n (no-img) | n (img) | no-img pass | img pass | diff |
|---|---:|---:|---:|---:|---:|
| 2–4 | 120 | 72 | 24.8 | 28.6 | +3.8 |
| 5–6 | 223 | 35 | 17.1 | 16.0 | −1.1 |
| 7+ | 137 | 13 | 9.8 | 4.4 | **−5.4** |

The marginal advantage is pure composition: natural-image tasks average **4.42**
scored units against **5.74** for chart-only tasks. Once you hold rubric size
fixed the sign flips in the two harder bands. This is a textbook Simpson
reversal and it means **`required_modalities` is not an independent difficulty
axis on this corpus.**

### 8.4 Tool family (confounded, reported for completeness)

Pooled pass by the extra tool family beyond the shared `asset`/`agents`/`file`/
`message` base:

| family | n | pooled pass |
|---|---:|---:|
| lock + review | 30 | **32.2%** |
| shell_scripting | 150 | 22.4% |
| approval | 90 | 20.0% |
| deploy | 90 | 19.5% |
| *(none)* | 60 | 19.0% |
| pairing | 30 | 16.2% |
| lock | 30 | 10.5% |
| containment | 30 | 10.0% |
| booking | 60 | 9.3% |
| tasks | 30 | **4.1%** |

28.1pp spread, but each family maps onto a small fixed set of scenarios, so this
is largely a scenario-identity axis wearing a tool-shaped hat. It is useful for
spotting which capability surfaces are weakest (`tasks`, `booking`,
`containment`, `lock` all under 12%) and not for causal claims.

**Ten combinations, not nine — and the tenth matters.** Thirty tasks grant
`lock` *and* `review`, thirty grant `lock` alone, and they come from disjoint
scenarios: `collaborative_contract_review_*` for the pair,
`distributed_lock_shared_resource_*` for the singleton. An earlier draft of this
report merged them into one `lock` row of n = 60 at 21.4%, which is arithmetically
correct for that pooling and substantively misleading — it averages the corpus's
**best** group (32.2%) with one of its **bottom four** (10.5%), hiding a 21.7pp gap
and understating the axis's spread as 18.3pp. Distributed locking is genuinely hard
(§10.6: nine of fourteen models stop on `missing_required_side_effect` in that
round); contract review is not. Merging them cancelled the finding out.

### 8.5 Gold-trajectory length

<!-- block:gold_length.table -->
Banding the 600 covered tasks by the length of their reference solution into short / mid / long tertiles (n = 113 / 347 / 140; reference length runs 7–18 steps). Pass rate falls monotonically across the three bands for twelve of fourteen models (luna and v4-flash are the exceptions).

| Model | short | mid | long | all |
|---|---:|---:|---:|---:|
| 3.5-flash | **3.5** | **3.2** | **2.1** | **3.0** |
| 3.6-flash | 26.5 | 21.9 | 3.6 | 18.5 |
| 3.7-flash | 30.1 | 22.5 | 7.1 | 20.3 |
| 3.1-pro | 18.6 | 12.4 | 5.7 | 12.0 |
| opus-4.8 | 32.7 | 25.1 | 9.3 | 22.8 |
| opus-5 | 36.3 | **26.2** | 13.6 | **25.2** |
| sonnet-5 | 32.7 | 22.2 | 7.9 | 20.8 |
| sol | **37.2** | 24.8 | 13.6 | 24.5 |
| luna | 23.0 | 23.1 | 9.3 | 19.8 |
| terra | 28.3 | 22.2 | 7.9 | 20.0 |
| v4-pro | 24.8 | 21.6 | 11.4 | 19.8 |
| v4-flash | 9.7 | 9.8 | 4.3 | 8.5 |
| kimi-k3 | 30.1 | 23.1 | 12.1 | 21.8 |
| glm-5.2 | 19.5 | 16.7 | 5.7 | 14.7 |

**Not every row here is like-for-like.** v4-pro, v4-flash and glm-5.2 were handed asset file paths as text and never received an image (`multimodal_input_mode=path`); kimi-k3, the open-source model that did receive images, is the control; also, 3.7-flash is reported on `r02`, the higher of its two complete runs, where every other row is a single run (the two differ by 1.7pp, McNemar p=0.27). Full statement: [run conditions](#run-conditions-that-are-not-comparable).
<!-- /block:gold_length.table -->

Monotone decreasing for **twelve of fourteen**, and 3.5-flash is *not* the
exception here that it is in §8.1 — at high it falls cleanly, 3.5% → 3.2% → 2.1%,
on 4 / 11 / 3 passes. The two rows that tick *up* into the mid band are luna
(23.0% → 23.1%) and v4-flash (9.7% → 9.8%), each by a tenth of a point, which is
noise at these band sizes. The long band is where the field collapses: sonnet-5
falls from 32.7% to 7.9%, 3.6-flash from 26.5% to 3.6%. sol and opus-5 hold up
best (13.6% each), then kimi-k3 at 12.1% and v4-pro at 11.4%; opus-4.8 manages
only 9.3%, level with luna.

**Step efficiency on success** (median of steps taken ÷ gold length, over the
passing episodes that finished inside the 20-step budget):
3.7-flash 1.73 · 3.5-flash 1.70 · 3.6-flash 1.60 · v4-flash 1.55 · kimi-k3 1.40 ·
sol 1.25 · glm-5.2 1.25 · v4-pro 1.24 · opus-4.8 1.11 · luna 1.10 · 3.1-pro 1.00 ·
terra 1.00 · sonnet-5 1.00 · opus-5 1.00. **No model beats the reference
solution.** Four sit exactly on it, and 1.00 is a floor rather than a
distinction — it is what you get when the median passing run retraces the gold
trajectory step for step. 3.5-flash's 1.70 rests on 14 uncapped passes and
should not be read against rows built on ten times as many.

---

## 9. The reasoning-effort sweep

This is the section the v8 corpus exists for, and the one the earlier `think`
corpus could not produce.

### 9.1 The sweep is real and verified

Two independent confirmations that the knob was actually applied:

1. **Configuration.** All 1,800 run headers record
   `reasoning_effort == reasoning_effort_effective`, exactly 60 per
   (model, effort) cell, no exceptions.
2. **Consumption.** Reasoning tokens were extracted from all **230,849** model
   calls. The three families report it under different keys —
   Gemini `usage.thoughts`, OpenAI `usage.output_tokens_details.reasoning_tokens`,
   Claude `usage.output_tokens_details.thinking_tokens` — and Gemini reports
   `thoughts` as *null*, rather than as `0`, when a call did no thinking (6,985
   such calls; the field is never absent). 96.6% of them are 3.6-flash and
   3.7-flash at `low` — 3,707 and 3,039 calls respectively. Null is
   counted as zero.

| Model | reasoning tok/call (l/m/h) | reasoning tok/task (l/m/h) | low→high | zero-reasoning calls (l/m/h) |
|---|---:|---:|---:|---:|
| 3.5-flash | 236.3 / 449.3 / **445.8** | 3,945 / 8,800 / 8,826 | 1.89× | 0.6% / 0.4% / 0.4% |
| 3.6-flash | 65.3 / 231.3 / 252.7 | 704 / 3,568 / 4,266 | 3.87× | **57.3%** / 0.0% / 0.0% |
| 3.7-flash | 83.4 / 165.6 / 233.0 | 1,079 / 2,578 / 4,237 | 2.80× | 39.1% / 0.0% / 0.0% |
| 3.1-pro | 93.6 / 118.0 / 171.2 | 1,002 / 1,394 / 2,359 | 1.83× | 1.0% / 0.1% / 0.2% |
| opus-4.8 | 68.6 / 126.1 / 177.8 | 717 / 1,493 / 2,291 | 2.59× | 64.1% / 57.2% / 51.4% |
| opus-5 | **5.1 / 17.6 / 46.5** | **45 / 184 / 553** | **9.07×** | 95.3% / 88.7% / 76.3% |
| sonnet-5 | 49.6 / 114.4 / 249.8 | 406 / 1,095 / 2,847 | 5.03× | 72.7% / 64.5% / 53.0% |
| sol | 27.7 / 57.8 / 94.0 | 301 / 760 / 1,354 | 3.39× | 10.6% / 9.7% / 10.2% |
| luna | 29.1 / 54.0 / 140.0 | 289 / 632 / 1,994 | 4.82× | 23.4% / 26.6% / 29.0% |
| terra | 25.3 / 57.2 / 142.5 | 243 / 625 / 1,760 | 5.63× | 0.4% / 0.1% / 0.1% |

**Reasoning spend and capability are mildly anti-correlated** (Spearman ρ =
−0.35 over the ten models at medium). The two endpoints make the point:

- **opus-5 is the thriftiest of the ten swept models and the best.** 184 reasoning
  tokens per task at medium — **1/48th** of 3.5-flash's 8,800 and **1/8th** of
  opus-4.8's 1,493 — while beating both. 88.7% of its medium calls emit *zero*
  reasoning tokens.
- **3.5-flash spends the most and scores the worst**, by an order of magnitude
  in both directions.
- **3.5-flash is also the only model whose reasoning spend does not rise with
  effort**: 449.3 → 445.8 tokens/call from medium to high. The knob stops doing
  anything, but the step budget keeps being consumed.

### 9.2 What effort actually buys

<!-- block:effort_grid.table -->
The sweep, ten models × three settings, 600 tasks per cell, **ordered by pass rate at `high`**. Each cell reads low / medium / high. In the two outcome columns a lone row-best is bold, so the mark shows which setting a model peaks at rather than how it ranks against the field; mean steps and capped carry no mark because their maximum is the worst cell, not the best. The remaining models in this report were run at a single setting and have no cells here.

| Model | pass low/med/high | score low/med/high | mean steps l/m/h | capped% l/m/h |
|---|---:|---:|---:|---:|
| opus-5 | 23.5 / **25.3** / 25.2 | 0.609 / 0.633 / **0.642** | 7.99 / 9.65 / 11.12 | 0.0 / 0.0 / 0.5 |
| sol | 20.8 / 23.0 / **24.5** | 0.610 / 0.628 / **0.633** | 10.04 / 12.41 / 13.65 | 0.2 / 2.0 / 1.2 |
| opus-4.8 | 21.3 / **23.8** / 22.8 | 0.608 / 0.618 / **0.621** | 9.67 / 11.11 / 12.17 | 0.8 / 2.8 / 6.2 |
| sonnet-5 | 13.8 / 17.0 / **20.8** | 0.537 / 0.570 / **0.596** | 7.31 / 8.76 / 10.64 | 0.3 / 1.0 / 3.3 |
| 3.7-flash | 20.5 / **21.5** / 20.3 | 0.600 / **0.618** / 0.581 | 12.18 / 14.91 / 17.79 | 3.0 / 12.7 / 40.7 |
| terra | 17.7 / 20.0 / 20.0 | 0.572 / 0.589 / **0.602** | 8.65 / 9.98 / 11.35 | 0.3 / 1.3 / 1.3 |
| luna | 19.5 / 17.7 / **19.8** | 0.590 / 0.591 / **0.604** | 9.12 / 10.91 / 13.54 | 0.5 / 2.7 / 9.2 |
| 3.6-flash | 18.5 / **20.2** / 18.5 | 0.595 / **0.602** / 0.586 | 9.99 / 14.77 / 16.31 | 1.8 / 14.5 / 24.7 |
| 3.1-pro | 8.3 / 9.8 / **12.0** | 0.463 / 0.479 / **0.487** | 9.84 / 11.02 / 13.16 | 4.7 / 10.2 / 25.3 |
| 3.5-flash | **13.5** / 3.2 / 3.0 | **0.473** / 0.299 / 0.259 | 16.22 / 19.46 / 19.70 | 38.7 / 85.0 / 89.2 |

**Not every row here is like-for-like.** 3.7-flash is reported on `r02`, the higher of its two complete runs, where every other row is a single run (the two differ by 1.7pp, McNemar p=0.27). Full statement: [run conditions](#run-conditions-that-are-not-comparable).
<!-- /block:effort_grid.table -->

**Mean steps rise with effort for all 10 of 10 models.** Mean score `low` →
`high` rises for only **7 of 10**. 3.5-flash collapses (0.473 → 0.259), and two
more slip back: 3.6-flash (0.595 → 0.586) and 3.7-flash (0.600 → 0.581). The
inversion is not a 3.5-flash quirk — it is every Gemini flash model.

<!-- block:effort_mcnemar.table -->
Paired McNemar, `low` → `high`, on the 600 tasks each model attempted at both settings. **lost** and **gained** are tasks that flipped, so a model can move a great many tasks and still end level. Five of ten models shift significantly at p < 0.05, and one of those shifts is downward.

| Model | Δpp | lost | gained | p | verdict |
|---|---:|---:|---:|---:|---|
| sonnet-5 | **+7.0** | 11 | **53** | 1.0e-07 | **real** |
| 3.1-pro | +3.7 | **9** | 31 | 6.8e-04 | **real** |
| sol | +3.7 | 12 | 34 | 1.6e-03 | **real** |
| terra | +2.3 | 15 | 29 | 0.049 | **real** |
| opus-5 | +1.7 | 23 | 33 | 0.23 | n.s. |
| opus-4.8 | +1.5 | 18 | 27 | 0.23 | n.s. |
| luna | +0.3 | 24 | 26 | 0.89 | n.s. |
| 3.6-flash | 0.0 | 34 | 34 | 1.00 | n.s. |
| 3.7-flash | −0.2 | 34 | 33 | 1.00 | n.s. |
| 3.5-flash | **−10.5** | **63** | **0** | 2.2e-19 | **real (negative)** |

**Not every row here is like-for-like.** 3.7-flash is reported on `r02`, the higher of its two complete runs, where every other row is a single run (the two differ by 1.7pp, McNemar p=0.27). Full statement: [run conditions](#run-conditions-that-are-not-comparable).
<!-- /block:effort_mcnemar.table -->

**Four models genuinely improve with effort, five are flat, and one is
destroyed by it.** That 4 / 5 / 1 split is at the uncorrected `p < 0.05` the
table above uses. The paper corrects over all 30 effort contrasts at
`0.05/30 = 0.00167` and therefore reports 3 / 6 / 1: terra (p = 0.049) is the
only row that moves between the two thresholds. The 3.5-flash result is
unusually clean: going from `low` to
`high` it lost 63 tasks and gained **zero**. Not a single task in 600 was
rescued by more thinking.

The mechanism is the interaction between effort and the step budget:

- Capped episodes **never** pass (0 / 2,304 across the sweep).
- Effort raises mean steps for every model (10/10).
- For models with headroom — opus-5 caps 0.0% at medium — that extra thinking is
  free and sometimes helps.
- For 3.5-flash, which already caps 38.7% of episodes at `low`, raising effort
  raises capping to 85.0% then 89.2%. The thinking has to be paid for in steps
  it does not have.

**The practical reading: `reasoning_effort` is a step-budget multiplier as much
as a quality knob.** Whether it helps depends almost entirely on how much cap
headroom the model has. Sweeping effort without also sweeping `--max-steps`
measures the interaction, not the knob.

### 9.3 Effort and cross-model spread

Max−min pass rate across the ten models: **low 15.2pp · medium 22.1pp · high
22.2pp**. opus-5 leads at all three levels (23.5 / 25.3 / 25.2). The growth in
spread from low to medium is almost entirely 3.5-flash falling out of the field
(13.5% → 3.2%); the top of the range moves by 1.8pp across the whole sweep.

`high` is the primary setting throughout this report (§1), chosen because it is
the only setting the four open-weight models were run at and therefore the only
one on which all fourteen rows are comparable. It costs little in separation:
**8 of the 10 swept models sit within 1.5pp of their own best at `high`**. The
two exceptions are 3.6-flash, 1.7pp below its `medium` peak, and 3.5-flash,
10.5pp below its `low` peak — and that model is treated as its own case
throughout.

---

## 10. Eight rounds, fourteen models each

Eight rounds selected on v8 for **cross-model score spread** — these are the
rounds where the field disagrees most, which makes them diagnostic and
unrepresentative in equal measure. See §11 before generalising from any of them.

Two of the eight (`clinical_trial_dosage_conflict`, `payment_chargeback_dispute`)
are deliberately unremediated control arms; `sla_support_response` is the
benchmark's calibration standard and is not in this set.

### 10.1 `payment_chargeback_dispute_v01_r04`

**Every model addresses the right agent; three of them are credited for it and one converts that into a pass.** This round is one of the two deliberately unremediated control arms, scored on three units: the report must contain the literal string `PASSPORT-VERIFY-9982`, the report must state the verdict, and a message must tell `Casey_Support_Specialist` the code. Every model that writes uses the same path, `reports/support_escrow_key_verification.txt`, and **fourteen of fourteen name `Casey_Support_Specialist` in a message** — so routing explains nothing here. The message unit is nevertheless credited to only **three** models: sol at step 5, v4-pro at step 4 and luna at step 10. sol is the round's only pass (11 steps) because it is the only model that pairs that credited message with both report units; luna and v4-pro each take the message and then only one report unit, landing on **0.67** from the opposite direction to everyone else on that score. The **0.67** tier is eight models deep (sonnet-5, 3.1-pro, terra, opus-5, luna, v4-pro, 3.7-flash and kimi-k3) and is reached from two directions: luna and v4-pro arrive with the message plus one report unit, and the rest arrive the opposite way, earning both report units in a single `file create` and never being credited for the message at all. Four models reproduce the code but miss a report unit for **0.33** (glm-5.2, 3.6-flash, opus-4.8 and v4-flash), and one cap out at 20 steps having written nothing (3.5-flash). sonnet-5 is the fastest writer at 7 steps with **0** `asset open` calls — it never looks at the escrow panel the rubric names, which costs it nothing, because that asset check is `role: diagnostic` and is not scored. kimi-k3 reaches the same **0.67** but needs all 20 steps and 5 opens, landing its report at step 19 — the last action it takes before the cap. As a control arm this round is hard for a reason; do not read it as representative of the remediated set. <!-- notes:scenarios.payment_chargeback_dispute_v01 -->

### 10.2 `warehouse_robotics_safety_halt_v02_r01`

**Four models burn the entire budget without writing a line, and only one model passes.** Three units: the report must contain the literal `88.5 °C`, the report must state the verdict, and a message must reach `Avery_Safety_Officer` in `#ops-safety-emergency-b7`. The split is nearly, but not quite, along vendor lines — the four models at **0.00** (3.5-flash, 3.6-flash, 3.1-pro and v4-flash) are three of the four Gemini models and one DeepSeek model, while the fourth Gemini, 3.7-flash, sits in the **0.67** group; all four of them are capped at 20 steps with **0** writes between them. kimi-k3 is the only pass, at 17 steps, taking the notification at step 11 and both report units at step 16. Six models write reports that earn both report units and no notification credit (**0.67**): opus-5, sonnet-5, v4-pro, 3.7-flash, opus-4.8 and luna. Three more get one report unit only (terra, sol and glm-5.2). Two failure modes are worth separating from the rest. 3.1-pro is the single model whose chain never names `Avery_Safety_Officer` at all — its one `message send` goes to `Blair_Robotics_Engineer`, and it spends its last actions on `wait`, `message search`, `asset list` and `file list` before the cap. luna is the inverse: it issues no `message send` in the entire episode, using `message broadcast` to the right channels instead, and still earns both report units for **0.67** across 3 writes. Routing, content and delivery verb are three different ways to lose the same unit, and this round shows all three. <!-- notes:scenarios.warehouse_robotics_safety_halt_v02 -->

### 10.3 `satellite_ground_station_handover_v02_r01`

**The easiest round in the set, and one of three where every model looks at exactly the same amount of evidence.** An identically shaped three-unit rubric to the warehouse round — the alert code `ERR-LOCK-049-DRV` in the report, the verdict, and a notification to `Devon_Ground_Ops_Lead` in `#ops-handover` — but the outcome inverts: nine of fourteen pass (opus-4.8, opus-5, sonnet-5, v4-pro, kimi-k3, terra, luna, 3.6-flash and 3.7-flash). Every single model opens exactly **2** assets; the field is unanimous on grounding volume in three of the eight rounds (warehouse, satellite and lock), and those are the rounds where the evidence budget can be ruled out as an explanation by inspection. Whatever separates the field here, it is not how much of it anyone looked at. What separates it is the verdict check. Three models carry the code and the notification but not the conclusion and finish at **0.67** (3.1-pro, glm-5.2 and sol), and two models cap with no report for **0.00** (3.5-flash and v4-flash). Speed spans nearly the whole budget among passes: opus-4.8 and opus-5 both close at 8 steps against 3.6-flash's 16. And speed is not what buys the verdict: 3.1-pro stops at 8 steps, exactly as fast as the two fastest passers, and still finishes at **0.67**. <!-- notes:scenarios.satellite_ground_station_handover_v02 -->

### 10.4 `clinical_trial_dosage_conflict_v02_r03`

**Transcription is universal, interpretation is not — and the model that reads the least gets it right anyway.** The second unremediated control arm, three units, and the round with the most passes after satellite: eight of fourteen (sonnet-5, opus-5, v4-pro, glm-5.2, 3.1-pro, v4-flash, kimi-k3 and 3.7-flash). Every model that writes at all puts the literal `142 U/L` in `reports/cohort_audit.md` and notifies `Avery_Trial_Safety_Monitor` in `#pv-safety-governance`; the discriminator is the verdict check, which asks the report to state that the elevated readings come from control-cohort mislabeling rather than true dosage toxicity. Four models miss exactly that and tie at **0.67** (opus-4.8, terra, sol and luna). Grounding volume does not predict the answer. The modal grounding load is **6** opens, shared by nine of the fourteen, but 3.7-flash passes on **2** and opus-4.8 fails on **3**, while 3.5-flash opens **0**, writes nothing across all 20 steps and scores **0.00**. 3.6-flash is the round's other cap: **0.33** from a notification it never follows with a report. The inference about which cohort the number belongs to is what separates a pass from a tie, and it is not bought with more looking. <!-- notes:scenarios.clinical_trial_dosage_conflict_v02 -->

### 10.5 `coordinated_containment_v02_r04`

**The cleanest discriminator in the set: everyone can write the report, four models can act, and one gets the argument wrong.** Five units — two `containment_action` effects (blocking `standard_user_449` and revoking `tok_live_sec_99321`), the literal `142 msgs/hr` in the report, the report verdict, and a notification. The written units are close to universal; the containment actions are the entire story. **Four models issue both containment verbs with the arguments the rubric names** — sonnet-5, terra, opus-5 and opus-4.8 — and those are exactly the four passes. Eight models issue **neither** verb (luna, sol, glm-5.2, v4-pro, 3.1-pro, 3.6-flash, v4-flash and 3.7-flash), stopping on `missing_required_side_effect` with a correct written analysis of an incident they never contained. Not one model performs exactly one of the two. The single intermediate score in the round belongs to kimi-k3 at **0.80**, and it is an argument error rather than a missing action: it runs both `containment disable-forwarding` and `containment block-token`, revokes the right token, and disables forwarding for `j.doe@company.com` instead of `standard_user_449`. Speed does not track capability here — sonnet-5 and terra both pass in 8 steps while opus-4.8 needs 18, and luna produces its report and notification in 5 steps and still scores **0.60**. 3.5-flash runs `containment status`, never executes either verb, issues **0** `file create`s and caps at 20 for **0.00**. <!-- notes:scenarios.coordinated_containment_v02 -->

### 10.6 `distributed_lock_shared_resource_v01_r02`

**Acquiring the lock is necessary and nowhere near sufficient.** Four units: two report checks (the decisive finding `UNLOCKED (Un-serialized Write Blocked)` and the verdict), a `lock_acquired` effect on `inventory_ledger`, and a notification. Only terra passes, taking all four in 13 steps. The field is unanimous on evidence here, as it is in three of the eight rounds (warehouse, satellite and lock) — every model opens exactly **3** assets — and as in the containment round the side effect is the wall: the two models on **0.00** (3.5-flash and 3.6-flash) cap at 20 steps with **0** writes between them and the great majority of the rest stop on `missing_required_side_effect`. The middle of the table is where the round earns its place. Three models reach **0.75** (opus-5, glm-5.2 and sol) and four sit at **0.50** (sonnet-5, kimi-k3, opus-4.8 and luna). sonnet-5 is the instructive case: it does acquire the lock, at step 9, and still finishes at **0.50** because it takes only one of the two report units and is never credited for a notification — the failure is the write-up and the message, not the action. v4-pro inverts that shape, writing a report and reaching only **0.25**. <!-- notes:scenarios.distributed_lock_shared_resource_v01 -->

### 10.7 `expense_reimbursement_conflict_v02_r05`

**Both Claude opus models pass, and the model that skips the passcode file lands at the bottom of the table — tied with one that wrote it.** Four units: two report checks and two separate messages, to `Blair_Audit_Investigator` with the passcode `AUDIT-8842-REV` and to `Casey_Approver` confirming the update. opus-5 passes in 9 steps and opus-4.8 in 11; both follow the same shape — read `config/telemetry_settings.json`, rewrite it with the passcode, then report and notify. A broad **0.75** plateau of seven models sits one unit behind (3.6-flash, 3.7-flash, sol, luna, kimi-k3, glm-5.2 and 3.5-flash), all of them having written the config file and both messages. sonnet-5 is the outlier worth reading: it is the only model in the round that acts and never writes `config/telemetry_settings.json` at all, going straight from `file read` to the report at step 4, and it finishes at **0.25** in the fewest steps of any model that acted (8). It does not sit at the bottom alone, though: sonnet-5 and terra share that score, and terra got there having written the config file, so the two reach the same number from opposite directions. Grounding volume is again uninformative: kimi-k3 and glm-5.2 open **10** assets each, the most in the round, and both finish on the **0.75** plateau, while opus-5 passes on **3**. 3.1-pro is not really a result in this round: its episode ended in a provider-side 504 after the harness had logged **4** asset opens and no scored action, so the **0.00** on its row is a recorded timeout rather than a failed attempt. <!-- notes:scenarios.expense_reimbursement_conflict_v02 -->

### 10.8 `expiring_resource_hold_v03_r04`

**The only round of the eight where grounding is scored, and it is the round where looking hardest pays least.** Here the `assets_accessed` check on `cluster_dashboard_reservation_board_active_hold_r04` is `role: required`, so reaching the board is worth a full unit of five — the exception that proves how unusual the other seven rounds are. Two models take all five: opus-5 in 10 steps and sol in 15. The notification is the wall for everyone else — it must state that hold token `HOLD-80G-9412-PROVISIONAL` was acquired for slot `A100-80G-02` with a 300 s TTL — and six models miss nothing but that, tying at **0.80** (kimi-k3, 3.7-flash, opus-4.8, luna, v4-pro and 3.6-flash). The two ends of the grounding axis both fail, in opposite ways. sonnet-5 never opens the board at all (**0** opens), forfeits the grounding unit outright, and still places the hold and writes the report for **0.60**. At the other end, the heaviest grounding load reached by any model in any of the eight rounds belongs to glm-5.2: **13** opens, converted into **0.20**, capping at 20 steps with **0** writes. Three models hit the 20-step cap in total (3.5-flash, 3.1-pro and glm-5.2), and 3.5-flash manages **0** opens and **0** writes for **0.00**. <!-- notes:scenarios.expiring_resource_hold_v03 -->

---

## 11. Limitations

Three of these are severe enough that they should be quoted alongside any figure
taken from this report.

### 11.1 Grounding is largely unscored (severe)

Across the 600 v8 specs there are **822 `assets_accessed` checks but only 346
are `role: required`**, and **no task carries more than one scored asset check**.
Therefore:

- **346 of 600 tasks** score anything at all for grounding.
- **254 of 600 tasks (42.3%)** score *nothing* for looking at an image — 233 of
  them carry an `assets_accessed` check that simply does not count, and 21 carry
  none at all.
- **18 of the 60 scenarios never score grounding in any round.**

On a benchmark whose premise is visual conflict resolution, a model can take
full marks without opening an image, and at `high` **28** episodes do — spread
over 25 distinct rounds, none of them among the eight in §10 (sonnet-5 18,
3.7-flash 9, 3.1-pro 1). Every "passed without opening anything" observation in this document is a statement
about the rubric, not about whether vision was necessary. Any claim of the form
"model X grounds better than model Y" is under-determined by this corpus.

### 11.2 The curated eight are biased on exactly that axis (severe)

The eight rounds in §10 were selected for **cross-model score spread**, and a
check that nearly everyone passes suppresses spread. The consequence is
systematic: **7 of the 8 rounds do not score grounding at all** (the exception is
§10.8). The curated set is therefore *worse than the corpus baseline* on the
precise dimension §11.1 already warns about. Do not estimate corpus-level
grounding behaviour from §10.

### 11.3 Notification checks are decided by body content, not routing (severe)

Across the eight rounds of §10, routing to the agent the criterion names is
close to universal while the share credited is not:

| round | named recipient | route to them | credited |
|---|---|---:|---:|
| `payment_chargeback_dispute_v01_r04` | `Casey_Support_Specialist` | 14/14 | 3/14 |
| `warehouse_robotics_safety_halt_v02_r01` | `Avery_Safety_Officer` | 12/14 | **1/14** |
| `satellite_ground_station_handover_v02_r01` | `Devon_Ground_Ops_Lead` | 12/14 | 12/14 |
| `clinical_trial_dosage_conflict_v02_r03` | `Avery_Trial_Safety_Monitor` | 13/14 | 13/14 |
| `coordinated_containment_v02_r04` | `Owen_Platform_On_Call` | 11/14 | 13/14 |
| `distributed_lock_shared_resource_v01_r02` | `Casey_Inventory_Controller` | 12/14 | 4/14 |
| `expense_reimbursement_conflict_v02_r05` | `Blair_Audit_Investigator` | 12/14 | 11/14 |
| `expense_reimbursement_conflict_v02_r05` | `Casey_Approver` | 12/14 | 11/14 |
| `expiring_resource_hold_v03_r04` | `Rowan_Research_Lead` | 10/14 | 2/14 |

Routing sits at 10–14 of fourteen in every round; crediting swings from 1/14
to 13/14. Routing therefore carries almost no signal, and the three rounds
credited at 3 of fourteen or fewer are the proof: the models reach the right
agent in the right channel and are still refused, because the body does not
carry the specific finding the grader wants.

The converse claim mostly holds, but not without exception. **70** notification
units are credited across these eight rounds, and in eight of the nine rows
crediting is consistent with routing being a precondition. It is not in
`coordinated_containment_v02_r04`, which credits **13** models while only **11**
reach `Owen_Platform_On_Call` by the routing measure used here — so at least two
credits were earned by a path this table does not count as routing. Correct
routing looks close to necessary, and is nowhere near sufficient. Read a failed notification check as "the body was wrong", not as
"the model messaged the wrong person".

### 11.4 Eighteen exact-match checks are unwinnable (contained)

Of the 256 checks that grade a round by exact substring match on written file
content, 18 (7.0%, spread over 14 of the 60 scenarios) demand a value that
appears in **no asset that round makes visible**. The reading exists in the
scenario package, but only in an asset outside the round's `visible_refs`, so no
amount of looking recovers it. All 18 are `required`, so each permanently caps
its task below a passing score for every model.

This is not the same defect as an unsatisfiable gold trajectory, and the
generation-time gate for that cannot see it: gold is authored *from* the round's
`resolved_value`, so the reference solution writes the value whether or not
anything the round displays contains it. **None** of the 18 has gold
`solution_commands` that open an asset holding the value — nine issue no `asset`
command at all.

The decisive evidence is a within-round comparison, which is what separates
"unwinnable" from "merely hard" — in the very same rounds, the *other* checks
keep passing at a normal rate:

| exact-match checks | n | this check passes | sibling checks, same rounds | never passed |
|---|---:|---:|---:|---:|
| value readable from `visible_refs` | 238 | 75.8% | 67.6% | 17 |
| value **not** readable | 18 | **0.7%** | 37.7% | **15** |

Fisher exact, one-sided, on the never-passed split: p = 5.614e-13. The four
scattered passes are all values a model can *derive* rather than read (a gate
percentage, an elapsed-seconds figure).

**The effect on every number in this report is negligible, and that is measured,
not assumed.** `score` was verified to equal the required-pass fraction on all
540 affected episodes and `success` to equal all-required-passed on all 17,993
scored episodes, so the counterfactual "these checks were never shipped" is
computed exactly rather than estimated. Under it, one of the 30 cells moves by
one task (`claude-sonnet-5` at medium, 102 to 103), average score rises by about
0.002 everywhere, and the model ordering is unchanged at all three efforts. The
affected rounds were already failing other required checks, so the unwinnable
one was almost never the binding constraint. The §10 finding on
`it_deployment_gate_v02` is likewise unaffected: no model passed its `r02` under
any effort, so dropping that round changes no pass count.

The generator now audits this unconditionally and names each offending round and
the asset that holds the value, so the defect cannot ship again unnoticed.

### 11.5 Other limitations

- **One run per cell, on a grid that is not rectangular.** The corpus is 34
  (model, effort) cells: ten API models × three efforts, plus the four
  open-weight models at `high` only. Every cell is a single run, on `r01`
  except 3.7-flash at `high`, which is reported on `r02`. Point differences
  under ~3pp are not resolvable; §7 and §9 use paired/bootstrap tests where a
  claim depends on a gap being real.
- **`success` is all-or-nothing**, so pass rate is partly a function of rubric
  size (§8.1). Score and pass tell different stories and both are reported.
- **The `field_contains` check column is 76.3% asset-access checks**, 476 of
  which are unscored (§7). It is not a transcription metric.
- **Effort and step budget interact** (§9.2). All effort results are conditional
  on `--max-steps 20`.
- **No `conflict_taxonomy` and no varying `difficulty` field** (§8). All
  difficulty analysis rests on structural proxies, chiefly scored-unit count.
- **Tool-family results are confounded with scenario identity** (§8.4).
- **21 empty-reply generation stalls** across all 20,400 episodes (opus-5 8,
  glm-5.2 4, terra 3, 3.5-flash 2, kimi-k3 2, sol 1, v4-flash 1) are counted as
  ordinary episodes. Separately, **7 episodes** carry a provider error
  (3.1-pro 6, 3.5-flash 1) and are dropped from scored counts, leaving
  **20,393**.
- **Do not difference this corpus against `analysis_think_history/` or
  `analysis_nothink_history/`.** Different
  generation run, different scenarios, different model list. Task-by-task
  differencing across corpora is invalid.

---

## 12. What each model would need

Ordered by pass rate at high. Each entry names the single change with the
largest expected effect, given the evidence above.

| Model | binding constraint | evidence |
|---|---|---|
| **opus-5** | Side-effect execution, not deliberation. It has the field's best per-check accuracy (**63.0%**) and the most concentrated losses (lift **1.49**), yet **239** rounds end with a required side effect undone and only **3** ever reach the step cap. **140** rounds finish exactly one required check short, 47 of them a side-effect unit. Effort buys nothing (+1.7pp, n.s.). | §5, §7.1, §9.2 |
| **sol** | Retrieval that does not convert. **877** `message search` calls and **4.08** opens/task — 6.5× and 1.4× opus-5's 134 and 2.98 — buy a **91.2%** asset-gate rate, the highest of any Claude or OpenAI model, and then a literal-content rate of **75.8%** against opus-5's 82.4 on a fifth of the searching. It reads more and records less faithfully. Still monotone in effort (+3.7pp, p = 0.0016), so not yet saturated. | §3, §7.3, §9.2 |
| **opus-4.8** | Cost, not capability. **Highest $/task in the field (4.8879)** for 2.3pp less pass than opus-5, at **4.1×** its reasoning spend (2,291 vs 553 tok/task) and a $/solved of 21.41 against 17.42. Effort adds nothing (+1.5pp, n.s.). It is a strictly dominated point on the frontier. | §1, §9.1, §9.2 |
| **kimi-k3** | Finishing the artifact. On the checks it does invoke it is the field's **best** reasoner — semantic **65.7%** and `other` **42.6%**, both field highs — but **437** semantic units are never invoked, **128** rounds produce no write at all, **170** hit the cap, and literal content is **60.5%** against a **92.5%** gate. It sees and reasons; it does not transcribe, and it does not stop. | §7.2, §7.3, §5 |
| **sonnet-5** | Grounding, uniquely in the field. Fewest opens (**1.83**/task), **92** zero-open tasks, the **lowest asset-gate rate (72.6%)** by 8.3pp, and **76** failed grounding units — the field high. It is also the **only** model whose literal-content rate *exceeds* its gate rate (76.6 vs 72.6): what it looks at, it transcribes. And it gains the **most** from effort of any model (**+7.0pp**, p = 1.0e-07). Under-grounded, not incapable. | §7.2, §7.3, §9.2 |
| **3.7-flash** | The 20-step budget. **244** capped rounds (**40.7%**) — its single largest outcome, more than its 122 passes — at 17.79 mean steps and a p50 of 19. It searches on all but 17 tasks and reads 4.69 files/task, then runs out. Effort is *negative* (**−0.2pp**, 34 lost / 33 gained) while cap% climbs 3.0 → 12.7 → **40.7** across low/medium/high. Give it steps, not thinking. | §5, §3, §9.2 |
| **terra** | Premature termination. **261** `missing_required_side_effect` and **210** incomplete submissions — both **field highs** — against only **8** capped rounds and 11.35 mean steps of a 20-step budget. **138** rounds end exactly one required check short. It has headroom it does not use, and effort only converts a little of it (+2.3pp, p = 0.049). | §5, §6 |
| **luna** | Latency and consistency, not price or skill. At **$0.1429/task** and **$0.7207/solved** it is the second-cheapest priced model in the field — only v4-pro is cheaper — and matches terra to within one round (119 vs 120 passes) for **1/8.5** the money. But it takes **620.5 s/task**, second only to kimi-k3's 963.8 and 1.7× the next row down (sol, 368.1); it is the only swept model whose pass rate *dips* in the middle of the sweep (19.5 → 17.7 → 19.8, all n.s.; four other rows are non-monotone the other way, peaking at medium); and it logs **6** no-progress stalls, tied for the field high. | §1, §9.2, §5 |
| **v4-pro** | *The least clear row in the table.* It is the price floor — **$0.0534/task** and **$0.2690/solved**, 2.7× under the next model and **82×** under opus-5 for 5.3pp less pass — and no failure mode dominates: 218 side-effect / 182 submission / 81 capped / 119 clean, zero zero-open rounds, mid-field search. Its one field-extreme weakness is verdict-level reasoning: `other` at **33.2%**, lowest of the ten models above 18% pass. No effort sweep exists to test whether deliberation would move it. | §1, §7.2 |
| **3.6-flash** | Superseded by its own successor. 3.7-flash beats it on pass (20.3 vs **18.5**), on $/task (0.5110 vs **0.9414**) and on $/solved (2.5130 vs **5.0889**). Effort buys it *exactly* nothing — **+0.0pp, 34 lost / 34 gained, p = 1** — while cap% goes 1.8 → 14.5 → **24.7**. The change with the largest expected effect is to stop running it. | §1, §9.2 |
| **glm-5.2** | Recording, not looking. It opens more than any model in the field (**7.26**/task, 4,355 opens) and clears asset gates better than any model in the field (**93.2%**) — then writes **0.78**/task with **157** zero-write rounds and a literal-content rate of **51.6%**, a **41.6pp** gate-to-content collapse. **207** capped, **494** semantic units never invoked. The pixels reach it; the file never gets written. | §3, §7.3 |
| **3.1-pro** | Message content itself. Semantic **46.4%** on 1,992 invoked units — below every model in the field except 3.5-flash, and 3.5-flash only reaches 45.6 by never invoking 1,222 of its semantic checks — and **831** failed `content_msg` units. Clustering lift **1.19** says the losses are scattered rather than task-concentrated: broadly wrong, not selectively. Also **402** `wait` calls, the field maximum. Effort helps (+3.7pp, p = 0.00068), but from 8.3%. | §7.1, §7.2, §3 |
| **v4-flash** | Termination. **358 of 600 rounds (59.7%)** end at the step cap, against 51 passes. It produces no file at all on **284** rounds (0.58 writes/task) while running 3.56 searches/task, and **837** semantic checks are never invoked. Asset gate **91.8%** → literal content **38.3%**. It has no verified rate card either, so it cannot be defended on price. | §5, §3, §7.3 |
| **3.5-flash** | Everything downstream of one thing: **it must stop searching and start writing.** **2,707** searches (4.51/task, 20× opus-5) and 5.57 file reads/task against **0.31** writes/task and **440** zero-write rounds; **535** capped (89.2%); clustering lift **1.05** [1.03, 1.08], statistically indistinguishable from independent per-check failure; **1,222** of 2,137 semantic checks never invoked, and literal content at **17.6%**. Effort actively harms it (**−10.5pp**, 63 lost / **0** gained, p = 2.2e-19). Running it at `low` recovers **13.5%** — 4.5× its high-effort result — for free. | §3, §6, §7.1, §9.2 |

The single highest-leverage harness change, for the corpus rather than any
model, is **scoring grounding**. 42.3% of tasks currently award nothing for
asset access on a visual-conflict benchmark, which caps what any grounding claim
from this data can mean.

---

## Cross-references

**Companion analyses.** These do not restate this document; each takes a cut
this one deliberately leaves out.

- [`eval-results-capability-analysis.md`](./eval-results-capability-analysis.md)
  — measurement validity, the F1–F6 failure taxonomy, the capability ladder,
  per-scenario difficulty and head-to-head records, and reproducibility.
- Per-model solver-failure deep-dives — **thirteen of the fourteen models**
  (v4-flash has none), every table generated from `failure_taxonomy.json` and
  re-verified cell by cell:
  - *Gemini*: [3.5-flash](./gemini-3-5-flash-solver-failures.md) ·
    [3.6-flash](./gemini-3-6-flash-solver-failures.md) ·
    [3.7-flash](./gemini-3-7-flash-solver-failures.md) ·
    [3.1-pro](./gemini-3-1-pro-solver-failures.md)
  - *Claude*: [opus-4.8](./claude-opus-4-8-solver-failures.md) ·
    [opus-5](./claude-opus-5-solver-failures.md) ·
    [sonnet-5](./claude-sonnet-5-solver-failures.md)
  - *OpenAI*: [sol](./gpt-5-6-sol-solver-failures.md) ·
    [luna](./gpt-5-6-luna-solver-failures.md) ·
    [terra](./gpt-5-6-terra-solver-failures.md)
  - *Open-weight*: [v4-pro](./deepseek-v4-pro-solver-failures.md) ·
    [kimi-k3](./kimi-k3-solver-failures.md) ·
    [glm-5.2](./glm-5-2-solver-failures.md)

**Artifacts.**

- [`cross-model-behavioral-comparison.html`](./cross-model-behavioral-comparison.html)
  — interactive version with per-scenario command timelines and charts.
- [`cross-model-behavioral-comparison.stats.json`](./cross-model-behavioral-comparison.stats.json)
  — every figure in this document in machine-readable form.
- [`behavior_notes.json`](./behavior_notes.json) — source of truth for the prose
  in §3 and §10.
- [`tools/build_behavior_report.py`](./tools/build_behavior_report.py) —
  regenerates the HTML and stats from the raw eval directory.
