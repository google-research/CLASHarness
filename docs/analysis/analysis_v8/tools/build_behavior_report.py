#!/usr/bin/env python3

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate the cross-model behavioral comparison report for multiagent_vlm_conflict.

v8 corpus: fourteen models over the same 600 tasks. TEN of them are swept across
`data_3_6_flash_v8/aligned_setting_{low,medium,high}` at three reasoning-effort
levels; the other FOUR exist at high only, under `open_source_high/` and
`aligned_setting_high_ocr/` (see FIXED_TREE). The shape is NOT a clean
cross-product -- do not multiply 14 x 3 x 600 anywhere.

The swept cells share one aligned harness config (tool_mode=native,
history_mode=full, native_tool_choice=auto, max_image_edge=2576,
multimodal_input_mode=both, max_steps=20); three of the four fixed-tree models
ran at multimodal_input_mode=path instead, which OPEN_SOURCE_CAVEAT quantifies.

EFFORT IS ALIGNED HERE -- this is the re-run the THINK corpus needed. The older
`data_3_5_flash` tree carried a standing caveat that Gemini ignored the
reasoning-effort pin (the request path did not read the flag), which made
Gemini-vs-Claude/OpenAI an unequal-effort comparison. That plumbing fix
(ThinkingConfig.thinking_level) has landed in this tree, and the logs confirm it
behaviorally: mean reasoning tokens per call rise monotonically low->high for ALL
NINE models of the field AS IT STOOD ON 2026-08-03, Gemini included -- 3.6-flash
65.3 -> 231.3 -> 252.7 (3.87x), 3.5-flash 236.3 -> 449.3 -> 445.8 (1.89x),
3.1-pro 93.6 -> 118.0 -> 171.2 (1.83x), against 5.1 -> 17.6 -> 46.5 for opus-5
and 27.7 -> 57.8 -> 94.0 for 5.6-sol (measured over every logged attempt).
Cross-family numbers in this tree are therefore like-for-like at a fixed effort
level. The run summaries do NOT record the flag, so this telemetry IS the
evidence -- do not restate the old caveat without re-checking it.

SCOPE OF THAT SWEEP, EXACTLY: nine models, not the current fourteen. It predates
gemini-3.7-flash (added 2026-08-14, swept, NOT re-measured for this monotonicity)
and it cannot cover the four fixed-tree models at all, since they exist at one
effort level and a low->high slope is undefined for them. Do not upgrade the
"NINE" to "TEN" or "FOURTEEN" without re-running the measurement.

Prior corpora (`docs/analysis/analysis_think_history`,
`docs/analysis/analysis_nothink_history`) differ from this tree in several harness
settings and in model roster, so they are NOT a clean A/B against v8 and must not
be differenced task-by-task.

Reads the per-task eval trajectories for the fourteen scored models and emits:

  * cross-model-behavioral-comparison.html  -- self-contained (inline CSS) report:
      action-category mix bars, a command-surface (sub-command) heatmap, a
      grounding x action map, stop-reason mix, an all-60-scenario score heatmap,
      a designed-difficulty-axis table (pass rate per *designed* lever, joined to
      the scenario seeds -- this is the one section that needs `--seeds`),
      and per-scenario *paired* tool-chains. Each chain is shown twice: a compact
      dot overview (one colored dot per step; red ring = the tool errored) and an
      expandable full command list (verbatim action text, +score-delta markers,
      error flags, stdout on hover). Curated scenarios carry an inline prose
      analysis pulled from behavior_notes.json.
  * cross-model-behavioral-comparison.stats.json -- the aggregate numbers
      (including the per-model sub-command mix) for reproducibility and for the
      companion markdown doc.

Prose is authored once in docs/analysis/analysis_v8/behavior_notes.json (the single
source of truth) and embedded here; the companion markdown uses the same strings verbatim.

It is deliberately dependency-free (stdlib only) so it runs anywhere the eval
artifacts are available.

MAINTENANCE RULE: this file was forked from the THINK-corpus copy, and that copy's
prose carried hard-coded measurements of ITS run. Every such number has been
re-measured against v8 or deleted (2026-08-03). If you fork this file again, do the
same -- a stale constant in a caption is indistinguishable from a real result.

Usage:
    python3 docs/analysis/analysis_v8/tools/build_behavior_report.py            # defaults
    python3 docs/analysis/analysis_v8/tools/build_behavior_report.py --root REPO --out DIR
    python3 docs/analysis/analysis_v8/tools/build_behavior_report.py --effort high
    python3 docs/analysis/analysis_v8/tools/build_behavior_report.py --seeds DIR

NOTE two roots. `--root` is the DATA checkout (it must contain eval_results/);
`--seeds` is the SOURCE tree holding the scenario seeds. They are separate
checkouts in this project, so a single --root will not satisfy both.
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import os
import random
import re
from collections import Counter, defaultdict
from math import ceil, comb
from pathlib import Path

# --- v8 corpus: fourteen models, ten of them swept over three effort levels ---
# The v8 eval tree is a 3-D grid, not the flat 2-D tree the THINK corpus had:
#   eval_results/data_3_6_flash_v8/aligned_setting_{low,medium,high}/<Vendor>/<slug>/r01
# so the model table stores the VENDOR PATH only and the effort level is composed
# in. Everything downstream of `MODELS` still sees a flat (name, subdir, key) triple
# pointing at ONE effort level -- PRIMARY_EFFORT -- so the sixteen behavioral
# sections are unchanged and describe a single, internally consistent run. The
# effort dimension is analysed separately in render_effort().
#
# PRIMARY_EFFORT = high. It was medium up to 2026-08-14, chosen then on measured
# separation (top-3 minus bottom-3 spread was 14.0 pp at medium vs 13.0 at high and
# 10.0 at low over the nine-model field). It is now high because the three
# open-source models joined the field and exist at ONE setting only, the one this
# corpus files under `open_source_high`; putting the primary level anywhere else
# would mean the main tables mixed a high-effort open-source row into a medium-effort
# comparison without saying so. High costs a little separation and buys a main table
# whose rows are all drawn from the same nominal setting. The current separation
# figure is recomputed and printed in the effort chapter rather than asserted here.
#
# HONEST LIMIT ON THAT ALIGNMENT: "same nominal setting" is as far as it goes, but
# all fourteen models really were run at high. Only the ten sweep models record it
# the plain way, reasoning_effort == reasoning_effort_effective == "high". The other
# FOUR declare it through a different channel: kimi-k3 records medium/high (the
# request asked for medium, the provider served high), while deepseek-v4-pro,
# deepseek-v4-flash and glm-5.2 leave `reasoning_effort` empty at both ends and
# carry `chat_template_kwargs = {"reasoning_effort": "high", ...}` instead.
# Re-censused over all 600 episode headers of each of the four fixed cells
# (2026-08-14): the two DeepSeek models carry the identical
# `{"reasoning_effort": "high", "thinking": true}`, glm-5.2 carries
# `{"reasoning_effort": "high"}` with no `thinking` key, and every one of those
# values is uniform across all 600 headers -- none of these is a per-episode mix.
# DO NOT SAY THESE THREE "EXPOSE NO EFFORT KNOB" -- an earlier version of this
# comment did, from reading only `run_config.reasoning_effort`, and it is false; see
# OPEN_SOURCE_CAVEAT for the verified per-model strings. The distinction matters in
# the reader's favour: the alignment behind PRIMARY_EFFORT is stronger than "we put
# the unlabelled runs in the high bucket because they had to go somewhere".
#
# gpt-5.5 is ABSENT by necessity, not by choice. Its three cells are unusable: the
# `low` directory does not exist at all; `medium` holds 480 records that are all
# `success=false, steps=0, trajectory=[]` behind a 429 quota wall; `high` holds five
# records (one file empty) and no run summary. Five non-empty records is enough to
# slip past an `if not recs` guard and would have silently manufactured a full
# gpt-5.5 row out of provider failures, so the model is removed from the table
# outright. It shares a provider quota pool with the three gpt-5.6 models and lost
# the race for it; this is an infrastructure outcome and says nothing about gpt-5.5.
# Ordered by PROVIDER, and every consumer of this list renders it in this order, so
# the provider blocks stay contiguous in every table and chart. Open-source is one
# block by editorial choice even though the three models ship from three different
# organisations -- see PROVIDER.
MODELS = [
    ("gemini-3.5-flash", "Gemini/3_5_flash", "flash"),
    ("gemini-3.6-flash", "Gemini/3_6_flash", "flash36"),
    ("gemini-3.7-flash", "Gemini/3_7_flash", "flash37"),
    ("gemini-3.1-pro", "Gemini/3_1_pro", "pro"),
    ("claude-opus-4.8", "Claude/opus_4_8", "opus"),
    ("claude-opus-5", "Claude/opus_5", "opus5"),
    ("claude-sonnet-5", "Claude/sonnet_5", "sonnet"),
    ("gpt-5.6-sol", "OpenAI/5_6_sol", "sol"),
    ("gpt-5.6-luna", "OpenAI/5_6_luna", "luna"),
    ("gpt-5.6-terra", "OpenAI/5_6_terra", "terra"),
    ("deepseek-v4-pro", "DeepSeek/v4_pro", "dsv4"),
    ("deepseek-v4-flash", "DeepSeek/dsv4_flash_v8_high_ocr", "dsv4f"),
    ("kimi-k3", "Moonshot/kimi_k3", "kimi"),
    ("glm-5.2", "ZhipuAI/glm_5_2", "glm"),
]

# Provider grouping, keyed by model key and stated EXPLICITLY rather than sliced out
# of the path. Two reasons, one editorial and one a bug fix:
#   * editorial -- the three open-source models come from three different orgs
#     (DeepSeek, Moonshot, ZhipuAI) but are reported as ONE provider block.
#   * bug fix -- the family-level comparison in disagreement_stats() derived the
#     vendor as `path.split("/")[2]`, which is correct for the THINK corpus
#     (`data_3_5_flash/aligned_setting/Gemini/3_5_flash`, four segments) but not for
#     v8, where MODELS holds two segments. Every model therefore fell into the "?"
#     bucket, `outs` came out empty for it, and the whole group comparison was
#     skipped: `clustering_strata.groups` has been shipping as `{}`.
PROVIDER = {
    "flash": "Gemini", "flash36": "Gemini", "flash37": "Gemini", "pro": "Gemini",
    "opus": "Claude", "opus5": "Claude", "sonnet": "Claude",
    "sol": "OpenAI", "luna": "OpenAI", "terra": "OpenAI",
    "dsv4": "Open-source", "dsv4f": "Open-source",
    "kimi": "Open-source", "glm": "Open-source",
}
# Render order for the provider blocks; also the order MODELS itself is written in.
PROVIDER_ORDER = ("Gemini", "Claude", "OpenAI", "Open-source")

# Models pinned to ONE subtree, bypassing the aligned_setting_{eff} composition
# because they were not run as an effort sweep. Keyed by model key; the value is a
# complete eval_results-relative path. `open_source_high/<Vendor>/<slug>` is
# byte-identical to `aligned_setting_high_ocr/<Vendor>/<slug>` for deepseek and glm
# and to `aligned_setting_high{,_vlm}/Moonshot/kimi_k3` for kimi -- verified by md5
# over all 60 scenario files -- so this picks the tree that names the cohort.
#
# `aligned_setting_high_ocr` IS NOT A DISTINCT RUN CONDITION, whatever its name
# suggests. Measured: its `ZhipuAI/glm_5_2` and `DeepSeek/v4_pro` are byte-identical
# (md5 over all 60 scenario files) to the `open_source_high` copies, and every model
# under it records multimodal_input_mode=path -- which is what the "ocr" in the name
# is describing, not an OCR preprocessing step applied to some models and not others.
# It is an alias/staging tree for the same path-mode open-source runs.
#
# It does hold ONE cell that exists nowhere else, and that cell is a SEPARATE MODEL
# rather than a variant of an existing one: `DeepSeek/dsv4_flash_v8_high_ocr` reports
# run_config.model = `deepseek-ai/DeepSeek-V4-Flash-0731`, against `v4_pro`'s
# `deepseek-ai/DeepSeek-V4-Pro`. Diffed field by field over the full run_config, the
# MODEL ID IS THE ONLY DIFFERENCE: both are path/1920/native/full with
# chat_template_kwargs={"reasoning_effort":"high","thinking":true}. So it enters as a
# fourteenth peer model, not as an ablation of v4-pro, and its 600 episodes are
# directly comparable to v4-pro's. DO NOT READ THE `_high_ocr` SUFFIX ON THIS SLUG AS
# A CONDITION -- an earlier reading of this tree took it for an OCR variant of
# v4-pro, which would have merged two different models into one row.
FIXED_TREE = {
    "dsv4": "data_3_6_flash_v8/open_source_high/DeepSeek/v4_pro",
    "dsv4f": ("data_3_6_flash_v8/aligned_setting_high_ocr/DeepSeek/"
              "dsv4_flash_v8_high_ocr"),
    "kimi": "data_3_6_flash_v8/open_source_high/Moonshot/kimi_k3",
    "glm": "data_3_6_flash_v8/open_source_high/ZhipuAI/glm_5_2",
}

# What to tell a reader who meets one of those directory names in the provenance
# paragraph. KEYED BY TREE NAME, not by position in a sorted list: the first
# attempt at this sentence said "the second of those directory names describes
# how the runs were staged", which sorted alphabetically and so pointed at
# `open_source_high` -- the innocent one -- while the name that actually needs
# defusing sorted first. Every tree in FIXED_TREE must appear here; the build
# fails otherwise, so a new staging tree cannot arrive unexplained.
TREE_NOTE = {
    "open_source_high": None,       # self-describing; nothing to correct
    "aligned_setting_high_ocr": (
        "`aligned_setting_high_ocr` is a staging directory, not an OCR "
        "condition \u2014 the runs under it were not preprocessed with OCR, and "
        "the name is describing that they were served asset file paths as text "
        "(`multimodal_input_mode=path`), which is stated as a run condition "
        "below in its own right"),
}
_missing_tree_note = sorted({p.split("/")[1] for p in FIXED_TREE.values()}
                            - set(TREE_NOTE))
if _missing_tree_note:
    raise SystemExit(
        "FIXED_TREE uses staging tree(s) " + ", ".join(_missing_tree_note)
        + " with no TREE_NOTE entry. Add one (None if the name needs no "
          "clarification) so the provenance paragraph stays explicit.")

# What the open-source rows are NOT comparable on. Rendered next to every table that
# includes them; do not drop it when editing the prose.
#
# EVERY FIGURE BELOW WAS RE-MEASURED AT FOURTEEN MODELS. The previous version of this
# comment was written for a thirteen-model field and was not revisited when the
# fourteenth arrived, so it went on describing 33 cells, two path-mode models and a
# thirteen-row ranking. Re-derive rather than patch: tools/../census34.py in the
# scratch directory walks all 34 cells and prints these in the shape used here.
#   * multimodal_input_mode: deepseek-v4-pro, deepseek-v4-flash and glm-5.2 ran at
#     `path`, i.e. they were handed asset FILE PATHS as text and never received an
#     image. The other eleven models ran at `both`. Declared == effective in all 34
#     cells, so nothing was silently downgraded by a provider.
#   * reasoning_effort: ALL FOUR OPEN-SOURCE MODELS RAN AT HIGH, declared through two
#     different channels. kimi-k3 requested `medium` on the API field and was served
#     `high` (reasoning_effort_effective; it is the only cell in the corpus where
#     declared and effective differ). The other three leave the API field empty and
#     carry it in `chat_template_kwargs` instead --
#       dsv4, dsv4f: {"reasoning_effort": "high", "thinking": true}
#       glm        : {"reasoning_effort": "high"}
#     -- which is how the OpenAI-compatible endpoint they were served through takes
#     it. DO NOT WRITE "no reasoning_effort recorded" FOR THESE THREE: an earlier
#     version of this comment did, by reading only `run_config.reasoning_effort`,
#     and it is false. It also matters for the effort chapter: it is what makes
#     placing all four in the `high` cell correct rather than merely convenient.
#   * quantization: glm-5.2 alone was served quantized -- its model id is
#     `zai-org/GLM-5.2-FP8`. The other three are `deepseek-ai/DeepSeek-V4-Pro`,
#     `deepseek-ai/DeepSeek-V4-Flash-0731` and `moonshotai/Kimi-K3`, none of which
#     declares a quantization. Report it as a run condition; this corpus cannot
#     separate an FP8 penalty from the model.
#   * max_image_edge: 31 of the 34 analysis cells ran at 2576; the three path-mode
#     cells ran at 1920. DO NOT WRITE "max_image_edge=2576 for every model" -- an
#     earlier version of the report header did. The cap is however INERT for exactly
#     those three, because they are the ones sent no images at all, so it cannot be
#     reached for a difference in what any model saw. Measured over all
#     34 cells x 600 = 20,400 headers:
#       2576 / ''           19 cells, 11,400 headers  (Claude, OpenAI, kimi-k3)
#       2576 / 'ultra_high' 12 cells,  7,200 headers  (the 4 Gemini models x 3
#                                                      efforts -- gemini_media_
#                                                      resolution is a Gemini-only
#                                                      knob, not a misalignment)
#       1920 / ''            3 cells,  1,800 headers  (dsv4, dsv4f, glm)
#     tool_mode=native, history_mode=full, native_tool_choice=auto and
#     tool_schema_mode=native are identical across all 34.
# COUNT UNITS CAREFULLY HERE. The previous text read "33 cells x 60 headers = 1980
# headers" and its per-group counts (1140/720/120) were FILE counts wearing the word
# "headers": each cell holds 60 log files of 10 rounds, i.e. 600 headers, not 60.
# The shape of the finding was right and every quantity in it was off by 10x.
# All statements verified over 600/600 episode headers per cell (record_type=
# episode_header in log/*.interactions.jsonl -- NOT the top-level *.jsonl, which
# carries scores and no run_config at all), one distinct value per field per cell.
#
# WHAT THIS DOES *NOT* LICENCE YOU TO SAY -- an earlier draft of this comment got it
# wrong and the error is easy to repeat. It claimed that because 346 of the 600 tasks
# carry a required grounding check, the path-mode models must be penalised on it,
# so their scores are a harness artifact rather than a measurement. MEASURED, THAT IS
# FALSE. The required grounding check is `effect:assets_accessed field_contains`: it
# scores whether the model OPENED the asset it needed, which a model handed a file
# path can do perfectly well.
#
# DENOMINATOR, EXACTLY: 346 tasks carry that check as `role=required`, for thirteen of
# the fourteen models. gemini-3.1-pro has 343, because three of its records came back
# with an EMPTY metadata.check_results -- collaborative_contract_review_v03_r05,
# distributed_lock_shared_resource_v02_r02, sla_support_response_v03_r03 (re-verified
# at fourteen models: still exactly those three, and pro's 343 IS the shared set).
# Those are the same missing-check records that make its high cell 3257+471 rather
# than 3284+476 in failure_taxonomy. Quote per-model rates out of 346, or rank over
# the 343 all fourteen share -- but say which. Do not print "X/346" for gemini-3.1-pro.
#
# Ranked over the 343 shared tasks (own-set X/346 in brackets), path-mode marked *:
#      1 glm-5.2*           95.3%  [330/346]     8 gemini-3.5-flash   90.7%  [314/346]
#      2 kimi-k3            95.0%  [329/346]     9 gemini-3.7-flash   89.8%  [311/346]
#      3 deepseek-v4-flash* 93.6%  [324/346]    10 gpt-5.6-luna       89.5%  [310/346]
#      4 gpt-5.6-sol        93.3%  [323/346]    11 gemini-3.1-pro     88.9%  [305/343]
#      5 claude-opus-5      92.4%  [320/346]    12 claude-opus-4.8    88.9%  [305/346]
#      6 deepseek-v4-pro*   91.3%  [316/346]    13 gpt-5.6-terra      84.3%  [292/346]
#      7 gemini-3.6-flash   91.0%  [315/346]    14 claude-sonnet-5    77.8%  [270/346]
# Ranks 11 and 12 are a GENUINE TIE on the shared set -- gemini-3.1-pro and
# claude-opus-4.8 both clear 305 of 343; they differ only in own-set denominator.
# gemini-3.7-flash is 89.8% and not the 88.9% an older copy of this table shows,
# because that row is now read off r02 (see RUN_SUBDIR).
#
# The three path-mode models rank 1st, 3rd and 6th of fourteen -- the ranking is now
# LED by one of them, which it was not at thirteen. The floor, claude-sonnet-5 at
# 77.8%, is 17.5 points below glm-5.2 and DID receive images. DO NOT WRITE "glm-5.2
# is highest of the five": an earlier draft compared only five models and understated
# its own finding. And do not restore the sentence that followed, either --
#
#   "Nor is the pair floored overall: deepseek-v4-pro ranks 8th of 13 at 19.8% pass,
#    exactly the field median"
#
# -- which was fair at thirteen models and is not at fourteen. On pass rate the group
# now places 9th, 11th and 13th of 14, its median (14.7%) sits BELOW the field's
# (19.9%), and 19.8% is no longer the median. What survives measurement is narrower:
# the two modes do not separate into two score populations. The path band, 8.5-19.8%,
# lies inside the 3.0-25.2% the image-receiving models span, and the band's own two
# ends are DeepSeek's two entries -- same lab, same condition, 11.3 points apart.
# _grounding_claim() renders exactly that, derives every number in it, and raises if
# the containment or the floor ever stops holding.
# So: report the input-mode difference as a difference in run condition, and let the
# reader weigh it. Do not assert it depressed their scores, and do not assert it did
# not: this corpus does not isolate an effect of the mode itself.
OPEN_SOURCE_CAVEAT = {
    "dsv4": ("multimodal_input_mode=path (never received an image); "
             "reasoning_effort=high, set via chat_template_kwargs "
             "(with thinking=true) rather than the API field; "
             "harness max_image_edge=1920 rather than the 2576 the other "
             "thirty-one cells used, which is inert for a model sent no images"),
    "dsv4f": ("multimodal_input_mode=path (never received an image); "
              "reasoning_effort=high, set via chat_template_kwargs "
              "(with thinking=true) rather than the API field; "
              "harness max_image_edge=1920 rather than the 2576 the other "
              "thirty-one cells used, which is inert for a model sent no images"),
    "glm": ("multimodal_input_mode=path (never received an image); "
            "reasoning_effort=high, set via chat_template_kwargs rather than "
            "the API field; served FP8-quantized (zai-org/GLM-5.2-FP8); "
            "harness max_image_edge=1920 rather than the 2576 the other "
            "thirty-one cells used, which is inert for a model sent no images"),
    "kimi": ("multimodal_input_mode=both; reasoning_effort medium requested, "
             "high served"),
}

# Confounds that apply specifically to the DOLLAR/TOKEN chapter. Kept separate from
# OPEN_SOURCE_CAVEAT because they hit a different set of models and a different
# claim: these are about cost comparability, not about grounding. Both were measured
# on this corpus, and both make a model look BETTER than a like-for-like comparison
# would. Render them next to any cost ranking, value-frontier or "cheapest model"
# statement; without them the report draws a conclusion the data does not support.
#
#   flash37 -- its cost advantage is a PROMOTIONAL ARTIFACT, not an efficiency win.
#     3.7-flash ran inside its introductory window ($0.75/$3.75) while 3.6-flash ran
#     at standard rates ($1.50/$7.50) ten days before that window opened. Measured on
#     this corpus that makes 3.7-flash look 45% cheaper per task than 3.6-flash
#     ($0.513 vs $0.941). Priced on the SAME card -- the $1.50/$7.50 that 3.7-flash
#     itself moves to on 2027-01-01 -- it is instead 9% MORE expensive ($1.026 vs
#     $0.941), because it burns more input tokens (654,875 vs 598,689 per task).
#     The sign of the comparison flips. Report the as-run number, but never as
#     evidence that the newer model is intrinsically cheaper.
#
#   dsv4 / glm -- their cost advantage is largely IMAGE-BLINDNESS. Input tokens
#     dominate spend here, and these two were sent asset file paths as text instead
#     of images (multimodal_input_mode=path). Measured: 112,166 and 132,361 input
#     tokens per task against a 727,325 mean for the six image-receiving models --
#     a 5.9x gap. deepseek-v4-pro is the cheapest model in the field at $0.053/task,
#     but a large part of that is a bill it was never sent. kimi-k3 is the control
#     that proves the point: same open-source cohort, same shim, but it DID receive
#     images and lands at 1,168,663 input tokens per task and $3.621 -- costlier per
#     task than every model here except the two Claude Opuses.
COST_CAVEAT = {
    "flash37": ("as-run introductory pricing ($0.75/$3.75, through 2026-12-31); "
                "at its own post-introductory rate this run would cost $1.026/task, "
                "i.e. more than gemini-3.6-flash, not less"),
    "dsv4": ("never sent images (multimodal_input_mode=path), so input tokens -- "
             "which dominate cost -- are ~5.9x below the image-receiving models"),
    # NO dsv4f ENTRY, deliberately. It is path-mode like the two above and its
    # token counts show the same image-blindness, but it has no PRICING entry, so
    # it renders N/A across the cost table (see COST_NA_NOPRICE) and carries no
    # cost figure for a caveat to qualify. A caveat attached to a row that does
    # not exist trains the reader to discount the ones that do.
    "glm": ("never sent images (multimodal_input_mode=path), so input tokens -- "
            "which dominate cost -- are ~5.9x below the image-receiving models"),
}

# Reasoning-effort levels, weakest first. This is the x-axis of every effort panel;
# keeping it ordered here means the charts never have to re-sort it.
EFFORTS = ("low", "medium", "high")
PRIMARY_EFFORT = "high"
EFFORT_TREE = "data_3_6_flash_v8/aligned_setting_{eff}"

# The effort chapter needs all three levels to exist, so it runs over this subset
# rather than MODELS. Ten of fourteen: the four FIXED_TREE models have one setting
# each. Every effort panel, the McNemar table and the low->high aggregates are
# computed over SWEEP_MODELS and say "ten" -- they must never silently fall back to
# MODELS, or the open-source rows would be read out of a directory that is not there.
SWEEP_MODELS = [m for m in MODELS if m[2] not in FIXED_TREE]


def tree(subdir, eff=PRIMARY_EFFORT, key=None):
    """Compose a (vendor path, effort) pair into an eval_results subdirectory.

    `key` pins FIXED_TREE models to their single subtree and ignores `eff`."""
    if key is not None and key in FIXED_TREE:
        return FIXED_TREE[key]
    return f"{EFFORT_TREE.format(eff=eff)}/{subdir}"

# Per-model run-subdirectory override, applied at PRIMARY_EFFORT only (see
# run_for). Absent -> DEFAULT_RUN.
#
# THE PREVIOUS VERSION OF THIS COMMENT DESCRIBED A DIFFERENT CORPUS. It justified
# an empty override by pointing at OpenAI reruns -- "the four OpenAI models also
# have an r02 ... sol 32.3% -> 30.9%". Measured on v8, no OpenAI cell has an r02
# -- there are nine of them (three models x three efforts) and every one is
# single-run: those directories live under eval_results/data_3_6_flash/, the
# earlier tree this build was forked from, and the figures are that corpus's, not
# this one's. In v8 exactly ONE cell of the 34 has a second run:
# aligned_setting_high/Gemini/3_7_flash. Do not re-derive policy from the old
# text. (The count was "33" while the roster was thirteen models; it is the
# 10 swept x 3 efforts + 4 fixed-effort cells that tree()/run_for() enumerate.)
#
# WHAT THE OVERRIDE BELOW COSTS, STATED PLAINLY BECAUSE IT IS A REAL COST.
# gemini-3.7-flash is reported on r02, chosen by the author over r01. Both runs
# are complete and clean -- 60 scenario files, 600 deduped tasks and 600 episode
# headers each; one distinct run_config within each run and the same one in both;
# every task scored; zero episodes carrying an `error` field, the marker that
# flags a provider failure here (gemini-3.1-pro carries 5 of them, so the check
# is a real discriminator and not a vacuous one). This is therefore not a repair
# of a damaged run -- it is a selection between two clean draws, made on the
# outcome:
#     r01  18.67% pass (112)  avg score 0.5650  246 capped
#     r02  20.33% pass (122)  avg score 0.5814  244 capped
#     paired over all 600: r02 gains 38, loses 28, McNemar exact p = 0.268
# Those figures are no longer typed into any string the reader sees:
# run_override_facts() recomputes them from both runs on every build and the
# footnote is formatted from the result, so the disclosure cannot drift from the
# corpus the way the paragraph above it once did.
#
# The gap is not distinguishable from run-to-run noise, and taking the higher of
# two draws for one model while the other thirteen get a single draw makes that
# model's figure an order statistic rather than a sample. For two exchangeable
# draws E[max] - E[draw] is half the expected gap between them; the single
# observed pair puts that at 1.67/2 = +0.8pp. (An earlier draft of this comment
# said +0.9pp, which is the same estimator rounded the wrong way.)
#
# The override also moves the model up the pass ranking, past three others, and
# desynchronises the effort chapter: its low and medium cells exist only as r01,
# so its measured effort gain now carries a run change inside it. The exact
# movement is NOT restated here in prose -- it is pinned in
# RUN_OVERRIDE_EXPECTED below and re-proved against the corpus on every build,
# because a hand-typed rank is precisely the kind of number that silently went
# stale when the roster grew.
#
# None of this is hidden from the reader: the run-condition footnote puts it on
# every table that renders the row, in the same channel as the open-source
# conditions.
DEFAULT_RUN = "r01"
RUN_SUBDIR = {"flash37": "r02"}

# What each override is known to buy, in pass-ranking terms, pinned so that a
# corpus or roster change cannot quietly rewrite the cost described above.
# check_run_overrides() recomputes both ranks (competition ranking over all
# MODELS at PRIMARY_EFFORT) and the set of models overtaken, and fails the build
# on any mismatch -- at which point the paragraph above must be re-read, not the
# constant nudged. Verified at the fourteen-model roster: 112 passes on r01 puts
# 3.7-flash 9th, behind terra (120) and the luna/v4-pro tie (119 each); 122 on
# r02 puts it 6th, ahead of all three.
RUN_OVERRIDE_EXPECTED = {
    "flash37": {"rank_default": 9, "rank_override": 6,
                "overtakes": ("deepseek-v4-pro", "gpt-5.6-luna",
                              "gpt-5.6-terra")},
}

def run_for(key, eff=PRIMARY_EFFORT):
    """Run subdirectory for one (model, effort) cell.

    Overrides apply at PRIMARY_EFFORT only, because that is the only effort at
    which a second run exists: r02 is present for 3_7_flash under
    aligned_setting_high and nowhere else in the tree. Silently returning "r02"
    for a low/medium cell would resolve to a directory that does not exist and
    fail far from the cause, so the rule is stated here once.
    """
    if eff == PRIMARY_EFFORT:
        return RUN_SUBDIR.get(key, DEFAULT_RUN)
    return DEFAULT_RUN


def _run_pair_stats(recs):
    """Pass/score/cap/failure totals for one run, on the report's definitions."""
    passed = sum(1 for r in recs if r.get("success"))
    capped = sum(1 for r in recs
                 if str(r.get("stop_reason") or "").startswith("reached max_steps"))
    # An `error` field marks a provider failure (gemini-3.1-pro carries five).
    # Counted rather than assumed absent, so "both runs are clean" is a
    # measurement and the RUN_SUBDIR comment can cite it.
    errs = sum(1 for r in recs
               if r.get("error") or r.get("error_tags")
               or (r.get("metadata") or {}).get("error"))
    return {"n": len(recs), "passed": passed,
            "pct": 100.0 * passed / max(len(recs), 1),
            "avg": sum(float(r.get("score") or 0) for r in recs) / max(len(recs), 1),
            "capped": capped, "errors": errs,
            "unscored": sum(1 for r in recs if r.get("score") is None)}


def run_override_facts(root):
    """Measure what each RUN_SUBDIR override bought, by loading BOTH runs.

    Every number the report tells the reader about a promoted run comes from
    here. They used to be typed into RUN_CAVEAT and into the per-table footnote
    as literals -- correct when written, unguarded thereafter, and in the same
    bug class as every other hand-counted figure this build has had to retract.
    """
    out = {}
    for key, run in sorted(RUN_SUBDIR.items()):
        subdir = next(s for _, s, k in MODELS if k == key)
        sub = tree(subdir, PRIMARY_EFFORT, key=key)
        old = load_model(root, sub, DEFAULT_RUN)
        new = load_model(root, sub, run)
        a = {r.get("task_id"): bool(r.get("success")) for r in old}
        b = {r.get("task_id"): bool(r.get("success")) for r in new}
        shared = set(a) & set(b)
        gain = sum(1 for t in shared if b[t] and not a[t])
        lose = sum(1 for t in shared if a[t] and not b[t])
        out[key] = {
            "run": run, "base_run": DEFAULT_RUN,
            "old": _run_pair_stats(old), "new": _run_pair_stats(new),
            "shared": len(shared), "gain": gain, "lose": lose,
            "p": mcnemar_p(lose, gain),
        }
        o, n = out[key]["old"], out[key]["new"]
        out[key]["delta_pp"] = n["pct"] - o["pct"]
        # E[max]-E[draw] for two exchangeable draws is half their expected gap;
        # the one observed pair is the only estimate of that gap available.
        out[key]["order_bias_pp"] = abs(n["pct"] - o["pct"]) / 2
    return out


def _pass_ranks(stats, override=None):
    """Competition ranking of every model by pass count at PRIMARY_EFFORT.

    `override` swaps one model's pass count, which is how the counterfactual
    "where would this row sit on its other run" is obtained without reloading.
    """
    tot = {k: stats[k]["passed"] for _, _, k in MODELS}
    if override:
        tot.update(override)
    return {k: 1 + sum(1 for o in tot if tot[o] > v) for k, v in tot.items()}, tot


def check_run_overrides(root, stats=None, facts=None):
    """Prove every RUN_SUBDIR override points at a real, complete, clean run.

    With `stats` and `facts` supplied it additionally re-proves the ranking
    movement pinned in RUN_OVERRIDE_EXPECTED, which is the one cost of the
    override that is described in prose and would otherwise rot silently.
    """
    bad = []
    known = {k for _, _, k in MODELS}
    for key, run in sorted(RUN_SUBDIR.items()):
        if key not in known:
            bad.append(f"{key}: not a model in MODELS")
            continue
        subdir = next(s for _, s, k in MODELS if k == key)
        d = os.path.join(root, "eval_results", tree(subdir, key=key), run)
        if not os.path.isdir(d):
            bad.append(f"{key}: override run {run!r} does not exist at {d}")
            continue
        base = os.path.join(root, "eval_results", tree(subdir, key=key),
                            DEFAULT_RUN)
        n_new = len(glob.glob(os.path.join(d, "*.jsonl")))
        n_old = len(glob.glob(os.path.join(base, "*.jsonl")))
        if n_new != n_old:
            bad.append(f"{key}: override run {run} has {n_new} scenario files "
                       f"but {DEFAULT_RUN} has {n_old} -- a truncated rerun is "
                       f"not a comparable draw and must not be promoted")
    if bad:
        raise SystemExit("RUN_SUBDIR overrides are wrong:\n  - " + "\n  - ".join(bad))
    if stats is None or facts is None:
        return
    for key, f in sorted(facts.items()):
        # "Both runs are clean" is asserted in the RUN_SUBDIR comment and in the
        # reader-facing footnote. Promoting a run that turns out to carry
        # provider failures would make both statements false.
        for tag in ("old", "new"):
            s = f[tag]
            if s["errors"] or s["unscored"]:
                bad.append(f"{key}: {f['base_run'] if tag == 'old' else f['run']} "
                           f"has {s['errors']} errored and {s['unscored']} "
                           f"unscored episodes -- it is not a clean draw, so it "
                           f"cannot be promoted or compared as one")
        exp = RUN_OVERRIDE_EXPECTED.get(key)
        if not exp:
            bad.append(f"{key}: overridden to {f['run']} with no entry in "
                       "RUN_OVERRIDE_EXPECTED, so its cost is undocumented")
            continue
        r_new, _ = _pass_ranks(stats)
        r_old, tot = _pass_ranks(stats, {key: f["old"]["passed"]})
        got = {"rank_default": r_old[key], "rank_override": r_new[key]}
        over = tuple(sorted(nm for nm, _, k in MODELS
                            if k != key
                            and f["old"]["passed"] < stats[k]["passed"]
                            <= f["new"]["passed"]))
        if got["rank_default"] != exp["rank_default"] or \
                got["rank_override"] != exp["rank_override"] or \
                over != tuple(sorted(exp["overtakes"])):
            bad.append(
                f"{key}: RUN_OVERRIDE_EXPECTED says the override moves it "
                f"{exp['rank_default']} -> {exp['rank_override']} past "
                f"{', '.join(sorted(exp['overtakes']))}, but the corpus now says "
                f"{got['rank_default']} -> {got['rank_override']} past "
                f"{', '.join(over) or 'nobody'}. Re-read the RUN_SUBDIR comment "
                "and update it deliberately; do not just edit the constant")
    if bad:
        raise SystemExit("RUN_SUBDIR overrides are wrong:\n  - " + "\n  - ".join(bad))


# Run-selection conditions, rendered in the same footnote channel as
# OPEN_SOURCE_CAVEAT. Kept separate because it is a different KIND of
# non-comparability: the open-source rows differ in how they were run, this row
# differs in which of its runs was chosen.
#
# Deliberately EMPTY at import: every figure in the sentence is a measurement,
# and a measurement typed into a source file is a measurement nobody re-checks.
# build_run_caveats() fills it from run_override_facts() before anything
# renders, and _run_caveats() -- the only legal reader -- fails the build if
# that never happened, so an unfilled dict cannot silently drop the disclosure.
RUN_CAVEAT = {}
# The measurements behind it, kept so the short per-table footnote and the long
# caveat-box sentence are formatted from one set of numbers rather than two.
RUN_FACTS = {}


def build_run_caveats(facts):
    """Compose the reader-facing run-selection footnote from measured facts."""
    RUN_CAVEAT.clear()
    RUN_FACTS.clear()
    RUN_FACTS.update(facts)
    for key, f in facts.items():
        o, n = f["old"], f["new"]
        RUN_CAVEAT[key] = (
            f"reported on {f['run']}, the higher of its two complete runs, "
            f"while every other model is reported on its single "
            f"{f['base_run']} \u2014 {f['base_run']} scored {o['pct']:.2f}% and "
            f"{f['run']} {n['pct']:.2f}% (paired: {f['gain']} gained, "
            f"{f['lose']} lost, McNemar p={f['p']:.2f}), so the gap is within "
            f"run-to-run noise and this row is a best-of-two where the others "
            f"are single draws")
    return RUN_CAVEAT


def _run_caveats():
    """RUN_CAVEAT, proven to have been built from the corpus first."""
    if set(RUN_CAVEAT) != set(RUN_SUBDIR):
        raise SystemExit(
            "run-condition footnotes were read before build_run_caveats() "
            f"populated them (have {sorted(RUN_CAVEAT)}, expected "
            f"{sorted(RUN_SUBDIR)}). Every promoted run must carry its "
            "disclosure on every table that renders the row.")
    return RUN_CAVEAT


def _run_facts():
    """RUN_FACTS, under the same fill-before-read guard as RUN_CAVEAT."""
    _run_caveats()
    return RUN_FACTS


# Per-model override for the run the TOKEN/COST is read from. Deliberately EMPTY:
# unlike the nothink corpus (where the archived vertex-anthropic run predated the
# usage-capture wiring and Claude cost had to be read off a separate re-run), every
# model in this tree records response.usage on 100% of attempts, so cost is measured
# on the same run as behavior for all fourteen models. Verified 2026-08-14 off the
# built stats: cost.has_tokens is True for all 14. What ONE model lacks is a rate
# card, not usage -- deepseek-v4-flash has has_price=False, so its three dollar
# columns render N/A while its token columns are fully measured. Do not "fix" that
# by routing it to a different run; there is nothing missing in the run.
COST_RUN_SUBDIR = {}

# Models whose scores define the cross-model score SPREAD of a round. The eight
# showcased rounds are pinned in CURATED_ROUNDS, so this tuple no longer *selects*
# them; it defines the population the reported spread is measured over and the
# population check_curated_pins() re-proves maximality against.
#
# ALL FOURTEEN vote. An earlier draft held out deepseek-v4-pro and glm-5.2 on the
# grounds that, having run at multimodal_input_mode=path, they were blindfolded
# models "pinned at the floor" that would drag a spread-maximising selector onto
# rounds where the spread is a harness artifact. Both halves of that turned out to
# be false when measured -- they pass the required grounding check at parity with
# the image-receiving models, and deepseek-v4-pro sits at the exact field median
# rather than the floor (the numbers are in the OPEN_SOURCE_CAVEAT comment). With
# the stated reason refuted there is nothing left to justify giving eleven models a
# vote and two models none, so the report measures the spread over everything it
# renders.
#
# Verified this costs nothing: all eight pinned rounds attain their scenario's
# maximum spread (all eight at spread exactly 1.000) under the eleven-, thirteen-
# AND fourteen-voter populations alike. So widening the electorate changes no
# showcased evidence -- it only removes an unsupported distinction from the
# methodology.
#
# The TIE COUNTS are not invariant, and an earlier version of this comment claimed
# they were. Rounds tied at the maximum, per curated scenario in CURATED_ROUNDS
# order: (7, 5, 9, 5, 1, 2, 1, 1) at eleven and thirteen voters, but
# (8, 5, 9, 5, 1, 2, 1, 1) at fourteen -- payment_chargeback_dispute_v01 goes from
# seven tied rounds to eight, because the added voter widens one more of its rounds
# to 1.000 as well. The reason maximality survives while the tie count does not:
# spread = max - min over the voters, so adding a voter is monotonically
# non-decreasing for EVERY round, and all eight pins already sit at 1.000, the
# largest spread a [0,1] score can produce. A pin at the ceiling can gain company
# but cannot be overtaken. Five of the eight have company, which is what
# check_curated_pins()'s docstring says.
# Re-measured 2026-08-14 over all fourteen; do not restate without re-measuring.
#
# All fourteen models carry a complete 600/600 task grid with no empty trajectories
# (re-verified 2026-08-14: stats[k]["n"] == 600 for every one of the 14), so this
# cannot silently narrow the candidate rounds through the completeness guard in
# _scenario_rounds().
PROSE_KEYS = ("flash", "flash36", "flash37", "pro", "opus", "opus5", "sonnet",
              "sol", "luna", "terra", "dsv4", "dsv4f", "kimi", "glm")

# Compact-but-versioned labels for tight table headers. The full name is used on
# the aggregate rows / legends; never abbreviate a model to a bare family word.
MODEL_LABEL = {
    "flash": "3.5 flash",
    "flash36": "3.6 flash",
    "flash37": "3.7 flash",
    "pro": "3.1 pro",
    "opus": "opus 4.8",
    "opus5": "opus 5",
    "sonnet": "sonnet 5",
    "sol": "5.6-sol",
    "luna": "5.6-luna",
    "terra": "5.6-terra",
    "dsv4": "v4-pro",
    "dsv4f": "v4-flash",
    "kimi": "kimi-k3",
    "glm": "glm-5.2",
}

# Per-model API list pricing, USD per 1M tokens. The ORIGINAL NINE were re-verified
# 2026-08-05 against first-party pricing pages: OpenAI
# developers.openai.com/api/docs/pricing (Standard tier), Gemini
# cloud.google.com/vertex-ai/generative-ai/pricing ("Google models / Gemini 3 /
# Standard"), Anthropic anthropic.com/pricing. Seven of the nine were unchanged from
# the earlier 2026-07-23 / 2026-07-25 verification. The FOUR models added 2026-08-14
# -- gemini-3.7-flash and the three open-source models -- were verified separately
# against their own first-party pages; see the ADDED-2026-08-14 block below.
# CORRECTION 2026-08-05 -- gpt-5.6-terra and gpt-5.6-luna were repriced downward and
# this table had been carrying the pre-cut rates. Per the AWS announcement "Amazon
# Bedrock announces up to 80% lower prices for OpenAI GPT-5.6 models" (posted
# 2026-07-30): "Effective July 30, 2026, on-demand inference prices on Amazon Bedrock
# for GPT-5.6 Luna are reduced by 80%, while prices for GPT-5.6 Terra are reduced by
# 20%. These reductions are in-line with OpenAI's first-party pricing changes for
# these models. ... Pricing for GPT-5.6 Sol remains unchanged." The cut is a uniform
# scale on every tier of each model -- terra x0.8 and luna x0.2 across all eight
# fields -- confirmed field by field against OpenAI's Standard table.
# THIS RUN WAS BILLED AT THE REDUCED RATES. The three OpenAI models executed
# 2026-08-02 22:41 through 2026-08-03 08:49 (all three effort levels; per-scenario
# log mtimes under eval_results/data_3_6_flash_v8/aligned_setting_*/OpenAI/*/r01),
# i.e. three to four days AFTER the 2026-07-30 effective date. So the pre-cut rates
# were not a stale-but-defensible "as-run" choice: they overstated terra and luna
# cost outright. Correcting them cuts luna's run total 341.37 -> 68.27 and terra's
# 795.58 -> 636.46.
# Gemini 3.5-flash is priced at its GLOBAL endpoint rate ($1.50/$9.00); the Vertex
# page lists a non-global rate of $1.65/$9.90 in effect since 2026-07-01, which would
# apply if this run had used a regional endpoint (not recorded in run_config).
# claude-opus-5 verified 2026-07-25 against Anthropic's launch post
# (anthropic.com/news/claude-opus-5): "priced at $5 per million input tokens and $25
# per million output tokens (the same as Opus 4.8)"; still $5/$25 on 2026-08-05.
#   in/out            standard (short-context) input / output rate
#   cached            cache-read input rate (a subset of input; omit -> billed as
#                     normal input, e.g. gpt-5.5 lists no separate cache-write rate)
#   cache_write       cache-write input rate (OpenAI 1.25x input; omit -> normal input)
#   long_threshold    input-token boundary above which the long-context column applies
#   long_*            long-context rates (input/output/cached/cache_write)
# Long-context tiers DO trigger here, unlike in the earlier corpora -- the inherited
# "never trigger" note was false for v8 and was removed. Measured over every logged
# attempt at medium effort (2026-08-03): the three Bedrock GPT models each push a
# handful of calls past their 128K threshold (largest single call 128,767 / 128,595 /
# 128,488 for sol/luna/terra; 3 calls apiece at medium, 5/5/4 at high, 3/2/0 at low).
# Their maxima sit within ~0.6% of the boundary, which looks like the harness context
# clamp landing on the same number, not a genuine long-context workload -- terra's
# largest low-effort call is 127,998, two tokens under. The dollar effect is real but
# negligible because _rate_tier already prices it: +0.096% for sol, +0.109% luna,
# +0.116% terra on the medium run total (sol $3.2318/task tier-on vs $3.2287 flat).
# Gemini 3.1-pro never approaches its 200K threshold (largest call 85,272 at medium,
# 84,561 at high). Gemini 3.5-flash reaches 389,940 on its largest call and Claude
# 170,949 / 181,642 / 172,171 (opus-4.8 / opus-5 / sonnet-5), but neither family has a
# long tier defined here, so both bill flat throughout.
# Claude token usage is read from the SAME run as behavior -- all fourteen models
# record response.usage on 100% of attempts (cost.has_tokens is True for all 14,
# re-verified 2026-08-14). The "at all three effort levels" half of this, verified
# 2026-08-03, can only be about the ten SWEPT models; the four fixed-tree models
# exist at high alone.
# Claude records ZERO cache-read and ZERO cache-creation tokens across all three
# models, so only input/output bill and no cached/cache_write rate is listed. OpenAI
# caches heavily by contrast (medium: 29.6M/24.9M/22.4M cache-read tokens for
# sol/luna/terra), and Gemini exposes no cache-read field at all, so no discount is
# applied there.
# The Claude thinking split IS recorded in v8. The inherited note claimed this run
# predated the harness fix for usage.output_tokens_details.thinking_tokens; that is
# false here -- all three Claude models populate it, so the report shows a MEASURED
# reasoning share for every one of the TEN swept models and nothing is reconstructed
# or estimated. It does NOT extend to the four open-source models, whose shim
# reports no split at all (reason_measured=False; they render "split not reported",
# never "0% reasoning"). Measured reasoning share of billed output at medium:
# opus-4.8 27.3%,
# opus-5 4.6%, sonnet-5 20.7%, against 68-90% for Gemini (which bills `thoughts` at
# the output rate) and 33-37% for the GPT models. Extended thinking bills INSIDE
# output_tokens for Claude and OpenAI, so dollar cost is exact either way.
# CAVEAT (unchanged): the OpenAI models ran on Amazon Bedrock (provider=bedrock-gpt);
# OpenAI bills Bedrock via AWS and rates "may differ from direct OpenAI pricing", so
# the direct list rates below are used as the best available proxy.
# --- ADDED 2026-08-14: the four new models, each priced AS RUN ---------------
# Same principle the OpenAI correction above established: the rate that applies is
# the one in force on the model's own execution date, not the rate on the page today.
# Run windows from per-scenario log mtimes (local PDT = UTC-7):
#   gemini-3.7-flash 2026-08-13 13:42-15:26 PDT  (= 20:42-22:26 UTC)
#   deepseek-v4-pro  2026-08-12 16:34-18:43 PDT
#   kimi-k3          2026-08-10 20:25 - 08-11 05:28 PDT
#   glm-5.2          2026-08-11 17:09-19:54 PDT
#
# gemini-3.7-flash -- $0.75 / $3.75, cached $0.075, verified on
# ai.google.dev/gemini-api/docs/pricing: "Output price (including thinking tokens)
# ... $3.75 through December 31, 2026." Flat: Vertex lists <=200K and >200K at the
# same $0.75/$3.75, so NO long tier is defined. Only a Global region rate exists for
# this model (unlike 3.5-flash, which also has a non-global rate), so the global/
# regional ambiguity noted above does not arise here. The model launched
# 2026-08-13 17:00 UTC and the run began 20:42 UTC, so the introductory rate -- the
# only rate it has ever had -- is the as-run rate. From 2027-01-01 it becomes
# $1.50/$7.50/$0.15; that is a FUTURE rate and must not be applied to this run.
#
# DO NOT "FIX" THE flash36 ROW TO $0.75/$3.75. Today's pricing page bundles 3.6-flash
# into the same introductory banner as 3.7-flash, which makes the $1.50/$7.50 above
# look stale. It is not: that discount began with the 3.7-flash launch on 2026-08-13,
# ten days AFTER the 3.6-flash run of 2026-08-02/03. Verified from raw Wayback `id_`
# bytes of the first-party page -- 3.6-flash Standard/Paid reads $1.50/$7.50 on the
# 07-23, 08-02, 08-03, 08-06 and 08-12 captures and only flips to "$0.75 through
# December 31, 2026" on the 08-13 capture -- and confirmed by Google's own launch
# post, which prices 3.7-flash at "half the original 3.6 Flash cost". THE TRAP: the
# strings "$0.75" and "$3.75" DO occur on the 08-02/03 captures, as the Batch and
# Flex tier rates (a standing 50% off Standard, which numerically equals the later
# introductory Standard rate). A grep for the number "confirms" the wrong answer.
# This run is interactive multi-step and synchronous -- 16.9 sequential tool-calling
# turns per task -- so Standard, not Batch or Flex, is the applicable tier.
#
# The three OPEN-SOURCE models. All three are served through an OpenAI-COMPATIBLE
# SHIM (provider=openai) rather than by these vendors' own SDKs, so the usage schema
# is Chat-Completions; see the dispatch note in load_token_usage.
#   deepseek-v4-pro -- $0.435 in / $0.87 out, cache-hit input $0.003625, from
#     api-docs.deepseek.com/quick_start/pricing. Flat, 1M context, no long tier.
#     Two dating hazards, both checked: (a) DeepSeek moves to a peak/off-peak
#     two-tier scheme effective 2026-08-16 16:00 UTC -- AFTER this 08-12 run, so the
#     flat rate applies and the two-tier rates must not be used; (b) the April 2026
#     launch graphic still served from that domain shows SUPERSEDED rates of
#     $0.145/$1.74/$3.48. Archived captures of the pricing page read $0.435/$0.87
#     continuously from 2026-06-02 through 08-14, including 08-12 itself.
#   kimi-k3 -- $3.00 in / $15.00 out, cache-hit input $0.30, from
#     platform.moonshot.ai/docs/pricing/chat-k3 (301s to platform.kimi.ai, Moonshot's
#     own successor docs domain). Flat: the K3 FAQ states "there is no tiering by
#     context length". The legacy 8k/32k/128k tiers belong to moonshot-v1-* and do
#     NOT apply. Price row is byte-identical on the 2026-07-28 and 2026-08-11
#     captures, so it is bracketed across the run rather than assumed.
#   glm-5.2 -- $1.4 in / $4.4 out, cache-hit input $0.26, from
#     docs.z.ai/guides/overview/pricing ("All prices are in USD."). ZhipuAI runs TWO
#     INDEPENDENT price lists: the international z.ai list used here, and a domestic
#     open.bigmodel.cn list quoting 8元/28元/2元. Those are NOT a currency conversion
#     of each other (8元 is about $1.10, not $1.4), so the CNY list is recorded here
#     for provenance only and is not used. Flat: the same table gives GLM-5.1/5/4.7
#     explicit [0,32)/[32+) input-length tiers but GLM-5.2 a single row.
#
# CACHE-WRITE IS DELIBERATELY OMITTED FOR ALL FOUR, and that is not the same as zero.
# None of the four vendors publishes a cache-write rate, and none affirmatively says
# cache writes are free, so no rate is asserted. It is moot in practice: the shim
# reports prompt_tokens_details as null on every one of the 27,331 open-source calls,
# so zero cached and zero cache-write tokens are ever counted and the `cached` rates
# below never fire. They are listed because they are the verified published tier.
# (ZhipuAI does publish a cache STORAGE price, currently "Limited-time Free"; that is
# a per-hour storage charge, not a per-token write rate, and this table has no field
# for it.)
#
# REASONING SPLIT: all four bill reasoning INSIDE the output token count, so every
# dollar figure here is exact. But the shim reports no completion_tokens_details at
# all, so for the three open-source models the visible/reasoning SPLIT is UNRECORDED
# -- distinct from a measured zero. summarize_cost carries `reason_measured: False`
# for them; do not render their 0 reasoning tokens as a measured 0% share.
#   The 27,331 above counts deepseek-v4-pro + kimi-k3 + glm-5.2 only (8,453 + 9,641
#   + 9,237), because it was measured before deepseek-v4-flash joined. v4-flash runs
#   on the SAME shim and behaves the same way -- 10,867 further calls, all with null
#   prompt_tokens_details and reason_measured=False, for 38,198 open-source calls in
#   total. It is absent from this block only because it has no rate card and so no
#   PRICING row; the reasoning-split caveat applies to all FOUR open-source models,
#   not to the three priced ones. Verified 2026-08-14 off the built stats.
PRICING = {
    "flash":   {"in": 1.50, "out": 9.00,  "cached": 0.15},
    "flash36": {"in": 1.50, "out": 7.50,  "cached": 0.15},  # as-run 08-02/03; see trap note
    "flash37": {"in": 0.75, "out": 3.75,  "cached": 0.075},  # intro rate, as-run 08-13
    "dsv4":    {"in": 0.435, "out": 0.87, "cached": 0.003625},
    "kimi":    {"in": 3.00, "out": 15.00, "cached": 0.30},
    "glm":     {"in": 1.40, "out": 4.40,  "cached": 0.26},
    "pro":     {"in": 2.00, "out": 12.00,
                "long_threshold": 200_000, "long_in": 4.00, "long_out": 18.00},
    "gpt55":   {"in": 5.00, "out": 30.00, "cached": 0.50,
                "long_threshold": 128_000, "long_in": 10.00, "long_out": 45.00,
                "long_cached": 1.00},
    "sol":     {"in": 5.00, "out": 30.00, "cached": 0.50, "cache_write": 6.25,
                "long_threshold": 128_000, "long_in": 10.00, "long_out": 45.00,
                "long_cached": 1.00, "long_cache_write": 12.50},
    "terra":   {"in": 2.00, "out": 12.00, "cached": 0.20, "cache_write": 2.50,
                "long_threshold": 128_000, "long_in": 4.00, "long_out": 18.00,
                "long_cached": 0.40, "long_cache_write": 5.00},
    "luna":    {"in": 0.20, "out": 1.20,  "cached": 0.02, "cache_write": 0.25,
                "long_threshold": 128_000, "long_in": 0.40, "long_out": 1.80,
                "long_cached": 0.04, "long_cache_write": 0.50},
    "opus":    {"in": 5.00, "out": 25.00},   # Claude opus 4.8 (vertex-anthropic)
    "opus5":   {"in": 5.00, "out": 25.00},   # Claude opus 5: same rate as opus 4.8 per Anthropic's launch post
    "sonnet":  {"in": 2.00, "out": 10.00},   # Claude sonnet 5: intro price thru 2026-08-31 (std $3/$15); run predates cutover
}

# Generic fallback note for any model whose interaction logs recorded no per-call
# usage. Kept for robustness only: it does not trigger anywhere in this corpus --
# all fourteen models record response.usage on 100% of attempts (cost.has_tokens is
# True for all 14, re-verified 2026-08-14; the "at all three effort levels" form of
# this, checked 2026-08-03, can only be about the ten swept models). Do not confuse
# this with the deepseek-v4-flash case below: that model HAS usage and lacks a rate
# card, so it takes the unpriced branch, not this one. The `gpt55` rate entry above
# is likewise retained
# but unused: gpt-5.5 is not in MODELS (see the note there), so nothing reads it.
COST_NA_NOTE = "no per-call token usage was recorded in this run's interaction logs"
# A SECOND, different reason a row can be N/A in the cost table, kept separate
# because collapsing the two states a falsehood about the corpus. deepseek-v4-flash
# records usage on 100% of its attempts; what it has no entry for is PRICING. The
# single note claimed its logs were empty, which they are not -- and the fix is not
# to invent a rate card for it. An unpriced model is dropped from the cost chapter
# and says why, rather than being priced off a guess.
COST_NA_NOPRICE = ("token usage was recorded, but this model has no verified "
                   "first-party rate card in PRICING, so it is not priced here")

# Action categories -> display label + color (kept in one place for legend + bars).
CATS = [
    ("asset_open", "open asset", "#8b5cf6"),
    ("asset_inspect", "inspect asset (list/info)", "#c4b5fd"),
    ("file_read", "file read", "#0ea5e9"),
    ("file_list", "file list", "#7dd3fc"),
    ("file_write", "file write/create", "#ec4899"),
    ("file_delete", "file delete", "#ef4444"),
    ("message_send", "message send", "#10b981"),
    ("message_read", "message search/read", "#14b8a6"),
    ("agent_delegate", "agent delegate", "#6366f1"),
    ("agent_status", "agent status", "#94a3b8"),
    ("wait", "wait", "#f59e0b"),
    ("other", "other", "#cbd5e1"),
]
CAT_COLOR = {c: col for c, _, col in CATS}
CAT_LABEL = {c: lab for c, lab, _ in CATS}
CAT_ORDER = [c for c, _, _ in CATS]

# Scenarios featured in the paired-timeline section, each PINNED to one round.
# RE-SELECTED ON V8, not inherited: tools/_probe_curated.py ranks all 60 scenarios by
# mean cross-model score spread over their ten rounds (every scenario has all fourteen
# models in all ten rounds, so the ranking is apples-to-apples) and then takes the top
# eight with at most one variant per scenario family, so eight slots buy eight
# different stories instead of one told twice.
#
# Re-running this mattered: the THINK-corpus list this file was forked from overlaps
# the v8 top eight in only THREE entries, and its #3 pick,
# multimodal_temporal_conflict_01_v01 (spread 0.779 there), ranks DEAD LAST in v8 at
# 0.270 -- inheriting the list would have spent a walkthrough slot on the single least
# revealing scenario in the corpus.
#
# WHY THE ROUND IS PINNED HERE INSTEAD OF BEING RE-DERIVED EVERY BUILD.
# The selection rule is "the round with the widest score spread (max-min) across the
# PROSE_KEYS voters". On this corpus that rule SATURATES: with eleven voters at high
# effort, 32 of the 60 scenarios have at least one round where some model scores 1.0
# and some model scores 0.0, so their max spread is exactly 1.000 and the metric stops
# discriminating. All eight scenarios below are in that saturated set, and five of the
# eight have MULTIPLE rounds tied at 1.000 (satellite_..._v02 ties on nine of its ten
# rounds). A max()-scan therefore does not select "the most revealing round" -- it
# selects the LOWEST-NUMBERED round among the ties, which is an artifact of loop order.
#
# Rather than let an arbitrary tie-break silently choose which evidence the report
# showcases -- and silently re-choose it whenever the roster changes -- the round is
# fixed here and check_curated_pins() re-verifies on every build that each pinned
# round is still a maximum-spread round for its scenario. So the pin is auditable:
# it can never quietly degrade into showcasing a non-maximal round.
#
# The pinned rounds are the ones the section-10 prose was written against under the
# nine-voter/medium configuration. Re-verified under the current eleven-voter/high
# configuration: all eight still attain their scenario's maximum spread, so pinning
# costs nothing in fidelity to the stated rule. Trailing comment = (spread, how many
# of the ten rounds tie at that spread).
#
# Two of these (payment_chargeback_dispute, clinical_trial_dosage_conflict) are the
# deliberately unremediated control arms of the generator work -- they are high-spread
# partly because they were left hard on purpose. That is a reason to read them, not to
# drop them, but do not treat them as representative of the remediated set.
CURATED_ROUNDS = [
    ("payment_chargeback_dispute_v01", "r04"),         # 1.000, 7 of 10 tied
    ("warehouse_robotics_safety_halt_v02", "r01"),     # 1.000, 5 of 10 tied
    ("satellite_ground_station_handover_v02", "r01"),  # 1.000, 9 of 10 tied
    ("clinical_trial_dosage_conflict_v02", "r03"),     # 1.000, 5 of 10 tied
    ("coordinated_containment_v02", "r04"),            # 1.000, sole maximum
    ("distributed_lock_shared_resource_v01", "r02"),   # 1.000, 2 of 10 tied
    ("expense_reimbursement_conflict_v02", "r05"),     # 1.000, sole maximum
    ("expiring_resource_hold_v03", "r04"),             # 1.000, sole maximum
]
CURATED_SCENARIOS = [s for s, _ in CURATED_ROUNDS]
CURATED_ROUND_OF = dict(CURATED_ROUNDS)

# Sub-commands surfaced in the command-surface heatmap, grouped by family. These
# are the diagnostic verbs; low-frequency variants are folded into "other" family
# rows are omitted for readability. (family label, [sub-command keys]).
CMD_FAMILIES = [
    ("asset", ["asset open", "asset list", "asset info", "asset describe", "asset read-text"]),
    ("file", ["file read", "file create", "file list", "file append", "file delete"]),
    ("message", ["message send", "message search", "message broadcast"]),
    ("agent", ["agent status", "agent delegate", "agent list"]),
    ("wait", ["wait 10", "wait 30", "wait 60"]),
]


# Shared prefix every LLM-judge-originated CheckResult carries (effect/file
# ``field_semantic_match`` and LLMChecker). Mirrors
# train_eval.eval_report._JUDGE_MSG_PREFIX, which uses it to scope judge-health
# aggregation to judge-authored records.
_JUDGE_MSG_PREFIX = "LLM semantic match:"

# Prose the checkers emitted for a fail-closed semantic check in records written
# before ``judge_invoked`` existed. Canonical implementation lives in
# train_eval.eval_report._judge_was_invoked; this tool is deliberately
# stdlib-only, so the logic is mirrored here and pinned to the original by
# tests/test_build_behavior_report.py rather than imported.
_LEGACY_FAIL_CLOSED_MARKERS = (
    "(empty transcript)",
    "no file ",
    "empty criterion",
)


def _judge_reached(check: dict) -> bool:
    """Did a judge actually read this check's content and return a verdict?

    Only meaningful for a judge-graded check. A deterministic checker never asks
    a judge anything, so "the judge was not reached" is not a defect of it and
    must not be reported as judge attrition -- it answers True.

    That distinction is what the ``_JUDGE_MSG_PREFIX`` guard enforces on the
    legacy path. The markers are ordinary English (``"no file "``), and
    ``FileChecker``'s deterministic branches emit prose about files too, so an
    unscoped substring match misfiles a perfectly graded ``field_contains``
    check as never-invoked and depresses its ``graded``/``pass_pct``. The
    canonical ``eval_report._compute_judge_health`` avoids this by filtering on
    the prefix at the call site before it ever consults the markers; this tool
    buckets every check rather than skipping non-judge ones, so the same guard
    lives inside the helper instead.

    The structured flag is checked first and unconditionally: it is an explicit
    signal from the checker, whereas the prefix is a heuristic over prose. Only
    the legacy fallback needs scoping.
    """
    invoked = check.get("judge_invoked")
    if invoked is not None:
        return bool(invoked)
    message = str(check.get("message", ""))
    if not message.startswith(_JUDGE_MSG_PREFIX):
        return True
    return not any(m in message for m in _LEGACY_FAIL_CLOSED_MARKERS)


def categorize(action):
    a = action if isinstance(action, str) else json.dumps(action)
    t = a.strip().split()
    if not t:
        return "other", "(empty)"
    head = t[0]
    sub = t[1] if len(t) > 1 else ""
    key = (head + " " + sub).strip()
    if head == "asset":
        return ("asset_open" if sub == "open" else "asset_inspect"), key
    if head == "file":
        if sub in ("create", "write", "edit", "append"):
            return "file_write", key
        if sub == "delete":
            return "file_delete", key
        if sub == "read":
            return "file_read", key
        if sub == "list":
            return "file_list", key
        return "other", key
    if head == "message":
        return ("message_send" if sub in ("send", "broadcast", "reply", "notify")
                else "message_read"), key
    if head == "agent":
        return ("agent_delegate" if sub in ("delegate", "spawn", "fork", "assign")
                else "agent_status"), key
    if head == "wait":
        return "wait", key
    return "other", key


# Domain command families that fall into the "other" bucket. This map exists ONLY
# to give the catch-all a readable, MECE breakdown in the report -- it changes no
# categorization. Any "other" sub-command whose head is not listed here still
# shows up (under its own head), so nothing is silently dropped.
OTHER_FAMILY_LABEL = {
    "approval": "approval (grant/deny/request)",
    "deploy": "deploy (gate-set/request/cutover)",
    "review": "review (submit/approve)",
    "booking": "booking",
    "pairing": "pairing",
    "lock": "lock (acquire/release)",
    "containment": "containment",
    "tasks": "tasks",
    "system": "system",
    "undo": "undo",
}


def other_family(key):
    """Roll an 'other'-category sub-command key up to a readable domain family.

    Keyed by the leading token so the breakdown stays MECE and self-updating: an
    unrecognised domain verb surfaces under its own head rather than vanishing."""
    if not key or key == "(empty)":
        return "(empty)"
    head = key.split()[0]
    if head == "file":  # e.g. "file move" -- categorized other, not a file_* verb
        return "file (move/other)"
    return OTHER_FAMILY_LABEL.get(head, head)


def action_text(s, limit=160):
    a = s.get("action", "")
    a = a if isinstance(a, str) else json.dumps(a)
    a = " ".join(a.split())
    return a[:limit] + ("\u2026" if len(a) > limit else "")


def stdout_snippet(s, limit=220):
    t = s.get("clean_stdout")
    if not t:
        t = s.get("stdout") or ""
    if not isinstance(t, str):
        t = json.dumps(t)
    t = " ".join(t.split())
    return t[:limit] + ("\u2026" if len(t) > limit else "")


def load_model(root, subdir, run=DEFAULT_RUN):
    d = os.path.join(root, "eval_results", subdir, run)
    # Deduplicate by task_id, last-wins: resumed/retried runs (notably the OpenAI
    # _02 reruns) append extra records for re-attempted tasks, so a raw line count
    # over-counts. parallel_run.summary reports one result per task_id keeping the
    # final attempt; matching that (last occurrence in file+line order) makes n=600
    # and reconciles pass@budget / avg_score with every run summary.
    by_id = {}
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        for line in open(f):
            line = line.strip()
            if line:
                r = json.loads(line)
                by_id[r.get("task_id")] = r
    return list(by_id.values())


def load_gold(root):
    """Map task_id -> gold-trajectory length (expert solution steps, excluding the
    terminal DONE) by reading each scenario round's specs.json. Missing files are
    simply absent from the map, so tasks without an expert solution are excluded
    downstream rather than guessed.

    v8 path fix: the THINK-corpus pattern pointed at `clasharness/data/scenarios/`,
    a directory that does not exist in this repo -- it matched zero files and
    silently emptied the whole steps-vs-gold section instead of failing. The v8
    scenarios live at `data/scenarios/conflict_vlm_v8_3.6_flash/`; the pattern below
    matches 600 files and covers 600/600 task_ids."""
    gold = {}
    pat = os.path.join(root, "data", "scenarios",
                       "conflict_vlm_v8_3.6_flash", "*", "tasks", "*", "specs.json")
    for p in glob.glob(pat):
        try:
            with open(p) as f:
                d = json.load(f)
            sc = (d.get("ground_truth") or {}).get("solution_commands") or []
            tid = d.get("task_id") or os.path.basename(os.path.dirname(p))
        except (OSError, ValueError):
            continue
        gold[tid] = len([c for c in sc if str(c).strip().upper() != "DONE"])
    return gold


def step_is_error(s):
    return (s.get("exit_code") not in (0, None)) or bool(s.get("error_tags"))


def co_failure_lift(pairs):
    """P(second required check fails | one already did) / marginal failure rate.

    `pairs` is [(k_i, f_i)] -- required checks in task i, and how many it failed.

    Counting argument: within a task, the number of ORDERED pairs of distinct
    required checks where both members failed is f(f-1); the number where the
    first failed and the second is any other check in the task is f(k-1). Pooling
    numerator and denominator over tasks gives the conditional failure rate, and
    dividing by the marginal rate q = sum(f)/sum(k) gives the lift.

    Under independence the ratio is exactly q, so the lift is exactly 1.0 -- an
    identity, with nothing estimated. Returns nan when there is nothing to
    condition on (no failures, or every task has a single check).
    """
    den = sum(f * (k - 1) for k, f in pairs)
    tot_f = sum(f for _, f in pairs)
    tot_k = sum(k for k, _ in pairs)
    if not den or not tot_f or not tot_k:
        return float("nan")
    return (sum(f * (f - 1) for _, f in pairs) / den) / (tot_f / tot_k)


def _boot_ci(pairs, fn, seed, draws=4000, alpha=0.025):
    """Percentile bootstrap CI over TASKS (the unit of resampling).

    Deterministic by construction: a local random.Random seeded from a caller-
    supplied integer derived from the data, never the global RNG and never
    time/hash based. Two runs of this script therefore produce byte-identical
    intervals, which matters because the markdown companion document quotes them.
    """
    rnd = random.Random(seed)
    n = len(pairs)
    if n < 2:
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(draws):
        sample = [pairs[rnd.randrange(n)] for _ in range(n)]
        v = fn(sample)
        if v == v:  # drop nan draws
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    vals.sort()
    return (vals[int(alpha * len(vals))], vals[int((1 - alpha) * len(vals))])


def failure_clustering(vectors):
    """Third headline metric: how CLUSTERED a model's required-check failures are.

    A task passes only if every `role=="required"` check passes, so pass@budget is an
    all-or-nothing aggregate over k_i checks. Two models with the SAME per-check pass
    rate can land at very different pass rates depending on whether their failures
    concentrate on a few tasks (few tasks lose everything, the rest are clean -> more
    tasks pass) or spread thin (many tasks lose exactly one check -> fewer pass).
    That difference is a real behavioral property, and it is invisible to both
    pass@budget and average score, so it is reported as its own axis.

    `vectors` is one list of per-required-check booleans per task.

    Returned:
      per_check_pct  pooled per-check pass rate p -- the "raw ability" level.
      obs_pass_pct   observed all-required-pass rate (== pass@budget on graded tasks).
      indep_pass_pct what the pass rate WOULD be if the k_i outcomes within a task
                     were independent draws at rate p: mean_i(p ** k_i). This is
                     analytic and deterministic -- no RNG, no seed. It was validated
                     against a 2000-draw permutation baseline (shuffle the pooled
                     outcomes, re-deal into the same per-task sizes) and agrees to
                     within 0.04pp on every model in this corpus, so the cheap
                     closed form is used instead of a sampled one.
                     NOTE mean_i(p**k_i) != p**mean(k): p**k is convex in k, so
                     using the mean k would understate the baseline (26.0% vs 27.8%
                     for claude-opus-5) and overstate excess by ~1.8pp.
      excess_pp      obs - indep, in percentage points. Positive = failures cluster
                     (the model converts a given per-check ability into MORE passes
                     than independence predicts). Same unit as pass@budget, so it
                     slots directly into an additive decomposition of a model gap.
      lift           CO-FAILURE LIFT, the headline correlation measure: P(a second
                     required check in a task fails | one already has) divided by
                     the model's own marginal failure rate q. Pooled as
                       (sum_i f_i(f_i-1) / sum_i f_i(k_i-1)) / q
                     Its independence baseline is EXACTLY 1.0 as an algebraic
                     identity, not as the output of a fitted variance model: under
                     independence E[f(f-1)] = k(k-1)q^2 and E[f(k-1)] = k(k-1)q, so
                     the ratio is q and the lift is 1. Nothing is estimated for the
                     null. That is why it is preferred here over overdispersion,
                     whose null depends on a variance formula that is easy to get
                     wrong (see below).
                     CAVEAT: lift is bounded above by 1/q, and q differs by model
                     (ceiling 3.82 for claude-opus-5 vs 1.63 for gemini-3.5-flash),
                     so rank order is comparable across models but magnitudes are
                     not. `lift_ceiling` is reported so the reader can see it.
      lift_ci        95% bootstrap CI over tasks, 4000 resamples. Seeded off the
                     model's own check totals so it is deterministic and stable
                     across runs and independent of evaluation order.
      overdispersion Var(failures per task) observed / predicted under independence.
                     1.0 = independent, >1 = clustered. Retained only as a
                     CROSS-CHECK on lift: it agrees with lift on the significance
                     verdict in every cell of the stratified table, and on the top
                     three models, differing only in point magnitude.
                     The independence variance MUST include the k-heterogeneity term:
                       Var = mean(k)*p*(1-p) + (1-p)**2 * Var(k)
                     via the law of total variance (E[F|k] = k(1-p) varies with k).
                     NB the second coefficient is (1-p) = the FAILURE rate, because
                     f ~ Bin(k, 1-p); writing p**2 there silently deflates every
                     ratio. Dropping the term entirely inflates the ratio unevenly
                     across models -- it flips gemini-3.5-flash from 1.30 to 1.98
                     and manufactures a crossing that does not exist.

    IMPORTANT SCOPE NOTE. Everything here is measured against an INDEPENDENCE null
    on the full corpus, and on this corpus that null is the wrong one to read as a
    model property: the task set spans a wide difficulty range, so failures pile
    onto the hard tasks even for a model that fails independently. Stratifying by
    difficulty removes most of the effect (see clustering_strata). Treat these
    numbers as descriptive shape, and use the stratified / paired figures for any
    claim about a model.
    """
    vectors = [v for v in vectors if v]
    if not vectors:
        return None
    n = len(vectors)
    ks = [len(v) for v in vectors]
    n_checks = sum(ks)
    n_passed_checks = sum(sum(1 for x in v if x) for v in vectors)
    p = n_passed_checks / n_checks
    obs = sum(1 for v in vectors if all(v)) / n
    indep = sum(p ** k for k in ks) / n
    fails = [len(v) - sum(1 for x in v if x) for v in vectors]
    mean_k = n_checks / n
    var_k = sum((k - mean_k) ** 2 for k in ks) / n
    mean_f = sum(fails) / n
    var_obs = sum((f - mean_f) ** 2 for f in fails) / n
    var_indep = mean_k * p * (1 - p) + (1 - p) ** 2 * var_k
    pairs = list(zip(ks, fails))
    lo, hi = _boot_ci(pairs, co_failure_lift, seed=n_passed_checks)
    q = 1 - p
    # Independence-predicted failure-count histogram, in the same five buckets as
    # fail_hist_pct. Under the SAME null that produces indep_pass_pct, the failures
    # inside task i are k_i independent draws at rate q, so f_i ~ Binomial(k_i, q)
    # and bucket b is mean_i P(Bin(k_i, q) = b), the last bucket pooling f >= 4.
    # Built from the ACTUAL per-task k_i rather than mean(k), for exactly the reason
    # indep_pass_pct is: the binomial pmf is not linear in k, so collapsing to the
    # mean k would bias every bucket.
    #
    # Bucket 0 is an identity that ties this vector to the summary columns:
    #   fail_hist_pct[0]  == obs_pass_pct    (a task passes iff it loses 0 checks)
    #   indep_hist_pct[0] == indep_pass_pct  (mean_i(C(k,0) q^0 p^k) == mean_i(p^k))
    # so fail_hist_pct[0] - indep_hist_pct[0] IS excess_pp. The difference vector is
    # the ONLY frame in which the clustering shape is visible: the raw histogram is
    # dominated by the model's ability level (a strong model piles at 0, a weak one
    # at 3-4), which is just the per_check_pct column drawn again. Against its own
    # baseline every model instead shows the same signature -- mass pulled OUT of
    # the middle buckets and pushed into BOTH ends. Tests assert both.
    indep_counts = [0.0] * 5
    for k in ks:
        for f in range(k + 1):
            indep_counts[min(f, 4)] += comb(k, f) * (q ** f) * (p ** (k - f))
    indep_raw = [100 * x / n for x in indep_counts]
    obs_raw = [100 * sum(1 for f in fails if f == b) / n for b in range(4)] \
        + [100 * sum(1 for f in fails if f >= 4) / n]
    indep_hist = [round(x, 1) for x in indep_raw]
    obs_hist = [round(x, 1) for x in obs_raw]
    # Differences come from the UNROUNDED vectors, not from the rounded ones above:
    # differencing two independently rounded numbers can be off by 0.1, which would
    # break the delta[0] == excess_pp identity the reader is told to rely on.
    # `or 0.0` collapses negative zero, which would format as "-0.0".
    delta_hist = [round(o - i, 1) or 0.0 for o, i in zip(obs_raw, indep_raw)]
    return {
        "n_tasks": n,
        "n_checks": n_checks,
        "mean_k": round(mean_k, 2),
        "per_check_pct": round(100 * p, 1),
        "obs_pass_pct": round(100 * obs, 1),
        "indep_pass_pct": round(100 * indep, 1),
        "excess_pp": round(100 * (obs - indep), 1),
        "lift": round(co_failure_lift(pairs), 2),
        "lift_ci": [round(lo, 2), round(hi, 2)],
        "lift_ceiling": round(1 / q, 2) if q else None,
        "overdispersion": round(var_obs / var_indep, 2) if var_indep else None,
        # Observed failure-count histogram (% of tasks losing 0,1,2,3,4+ required
        # checks) and the independence prediction for the same buckets. Read the
        # DIFFERENCE, not either vector alone -- see the comment above.
        "fail_hist_pct": obs_hist,
        "indep_hist_pct": indep_hist,
        "delta_hist_pp": delta_hist,
    }


def _ci_tail(vals, null=0.0):
    """95% percentile interval plus the one-sided bootstrap tail probability.

    The tail is the share of draws landing on the far side of `null` from the
    bulk of the distribution -- i.e. how often the resample contradicts the sign
    of the point estimate. Reported so that "not significant" claims carry a
    number instead of resting on whether an interval happens to straddle zero.
    """
    vals = sorted(vals)
    if not vals:
        return float("nan"), float("nan"), float("nan")
    lo = vals[int(0.025 * len(vals))]
    hi = vals[int(0.975 * len(vals))]
    below = sum(1 for v in vals if v <= null)
    return lo, hi, min(below, len(vals) - below) / len(vals)


_STRATA_CACHE: dict = {}


def clustering_strata(stats, draws=4000):
    """Separate MODEL-driven failure correlation from CORPUS-driven correlation.

    The clustering numbers in failure_clustering() are measured against an
    independence null over the whole corpus, and on a task set with a wide
    difficulty range that null is misleading: failures concentrate on the hard
    tasks even for a model whose checks fail independently, so every model looks
    clustered. Two corrections are computed here, both restricted to the tasks
    that EVERY model was graded on so the comparison is like-for-like.

    1. STRATIFIED lift. Each task is scored for difficulty by the pooled required-
       check failure rate of the OTHER nine models (leave-one-out, so a model
       never contributes to the difficulty axis it is judged against), split into
       terciles, and lift is recomputed inside each tercile where tasks are of
       comparable difficulty. Most of the whole-corpus effect disappears here --
       that part was the corpus, not the models.

    2. PAIRED peer comparison. "Is this model's lift above 1.0?" is largely a
       question about the corpus; "is it above the other nine ON THE SAME TASKS?"
       is a question about the model. Tasks are resampled once per draw and all
       ten models recomputed together on that resample, so the difference is
       properly paired and the corpus difficulty cancels.

    Deterministic: fixed seeds, local RNG (see _boot_ci). Memoized per stats object
    because the bootstrap is the slowest thing in this script and both the renderer
    and main() need the result.
    """
    cached = _STRATA_CACHE.get(id(stats))
    if cached is not None:
        return cached
    keys = [key for _, _, key in MODELS if (stats.get(key) or {}).get("by_task")]
    if len(keys) < 3:
        return None
    common_set: set = set()
    for i, k in enumerate(keys):
        graded = {t for t, v in stats[k]["by_task"].items() if v.get("req_k")}
        common_set = graded if i == 0 else (common_set & graded)
    common = sorted(common_set)
    if len(common) < 30:
        return None
    kf = {k: {t: (stats[k]["by_task"][t]["req_k"], stats[k]["by_task"][t]["req_f"])
              for t in common} for k in keys}

    def verdict(lo, hi, null=1.0):
        if lo != lo:
            return "flat"
        return "up" if lo > null else ("down" if hi < null else "flat")

    per_model = {}
    for i, mk in enumerate(keys):
        peers = [m for m in keys if m != mk]
        # leave-one-out difficulty: how the rest of the field did on this task
        diff = {}
        for t in common:
            pk = sum(kf[m][t][0] for m in peers)
            pf = sum(kf[m][t][1] for m in peers)
            diff[t] = pf / pk if pk else 0.0
        order = sorted(common, key=lambda t: diff[t])
        n = len(order)
        tiers = []
        for j, group in enumerate((order[:n // 3], order[n // 3:2 * n // 3],
                                   order[2 * n // 3:])):
            pairs = [kf[mk][t] for t in group]
            lo, hi = _boot_ci(pairs, co_failure_lift,
                              seed=90000 + i * 10 + j, draws=draws)
            tiers.append({
                "n": len(group),
                "lift": round(co_failure_lift(pairs), 2),
                "ci": [round(lo, 2), round(hi, 2)],
                "verdict": verdict(lo, hi),
            })
        per_model[mk] = {"tiers": tiers}

    # paired peer comparison -- one shared resample per draw, all models together
    #
    # NOTE: the RNG is consumed ONLY by the index draw below. Everything else in
    # this loop reads the already-drawn resample, so adding statistics here does
    # not perturb the per-model intervals.
    # Vendor label comes from the explicit PROVIDER map, NOT from slicing the path.
    # BUG FIX: this used to be `_path.split("/")[2] if len(_parts) > 2 else "?"`,
    # which assumed the four-segment THINK layout
    # (data_3_5_flash/aligned_setting/Gemini/3_5_flash). v8's MODELS holds two
    # segments (Gemini/3_5_flash), so the guard fell through for EVERY model, every
    # vendor came out "?", `groups` collapsed to a single bucket, `outs` was empty
    # for it, and the whole family-vs-rest comparison was silently skipped --
    # `clustering_strata.groups` has been shipping as `{}`. PROVIDER also gives the
    # editorial grouping the path cannot: DeepSeek, Moonshot and ZhipuAI share one
    # "Open-source" block.
    vendor = {_key: PROVIDER[_key] for _name, _path, _key in MODELS if _key in kf}
    missing = [k for k in keys if k not in vendor]
    if missing:
        raise SystemExit(f"clustering_strata: no PROVIDER entry for {missing}")
    # Render in PROVIDER_ORDER so the blocks stay contiguous and match every other
    # table, instead of alphabetically.
    groups = [g for g in PROVIDER_ORDER if g in {vendor[k] for k in keys}]
    rnd = random.Random(77001)
    n = len(common)
    collected = {k: [] for k in keys}
    gcollected = {g: [] for g in groups}
    for _ in range(draws):
        idx = [rnd.randrange(n) for _ in range(n)]
        tasks = [common[i] for i in idx]
        lifts = {}
        ok = True
        for k in keys:
            v = co_failure_lift([kf[k][t] for t in tasks])
            if v != v:
                ok = False
                break
            lifts[k] = v
        if not ok:
            continue
        for k in keys:
            others = [lifts[m] for m in keys if m != k]
            collected[k].append(lifts[k] - sum(others) / len(others))
        for g in groups:
            ins = [lifts[m] for m in keys if vendor[m] == g]
            outs = [lifts[m] for m in keys if vendor[m] != g]
            if ins and outs:
                gcollected[g].append(sum(ins) / len(ins) - sum(outs) / len(outs))
    for mk in keys:
        own = co_failure_lift([kf[mk][t] for t in common])
        peer_lifts = [co_failure_lift([kf[m][t] for t in common])
                      for m in keys if m != mk]
        peer_mean = sum(peer_lifts) / len(peer_lifts)
        lo, hi, pv = _ci_tail(collected[mk])
        per_model[mk]["peer"] = {
            "lift": round(own, 2),
            "peer_mean": round(peer_mean, 2),
            "diff": round(own - peer_mean, 3),
            "ci": [round(lo, 3), round(hi, 3)],
            "p": round(pv, 4),
            "verdict": verdict(lo, hi, null=0.0),
        }

    # vendor-group comparison: does a whole family sit above the rest? This is
    # the statistic that supports a family-level claim. Comparing the three
    # single-model intervals by eye does NOT: those intervals overlap the
    # non-family ones even when each clears zero on its own.
    full = {k: co_failure_lift([kf[k][t] for t in common]) for k in keys}
    group_out = {}
    for g in groups:
        ins = [k for k in keys if vendor[k] == g]
        outs = [k for k in keys if vendor[k] != g]
        if not ins or not outs:
            continue
        gm = sum(full[k] for k in ins) / len(ins)
        om = sum(full[k] for k in outs) / len(outs)
        lo, hi, pv = _ci_tail(gcollected[g])
        group_out[g] = {
            "models": [name for name, _p, key in MODELS if key in ins],
            "mean_lift": round(gm, 3),
            "other_mean_lift": round(om, 3),
            "diff": round(gm - om, 3),
            "ci": [round(lo, 3), round(hi, 3)],
            "p": round(pv, 4),
            "verdict": verdict(lo, hi, null=0.0),
        }
    out = {"n_tasks": len(common), "draws": draws, "per_model": per_model,
           "groups": group_out}
    _STRATA_CACHE[id(stats)] = out
    return out


# Two models whose per-check accuracy is this close count as "same raw ability"
# for the exemplar. 3.0pp keeps opus-5 (73.8) paired with gpt-5.5 (71.9), the
# largest lift gap available under that constraint.
EXEMPLAR_PC_TOL = 3.0
_ROUND_RE = re.compile(r"_r\d+$")


def clustering_exemplar(stats):
    """One named scenario where two models lose the SAME checks to different effect.

    Everything else in this section is a difference against a per-model baseline.
    That is the correct way to measure clustering but it is abstract, so this picks
    a concrete case a reader can look at directly.

    WHY THE COMPARISON IS CONSTRAINED. A naive cross-model contrast of "how
    concentrated are the misses" does not work on this corpus, and the measurement
    says so: lost-checks-per-spoiled-task ranks the ten models in almost exactly
    their ability order (gemini-3.5-flash 2.96 down to gpt-5.5 1.79, monotone in
    per-check accuracy with two inversions). It is a restatement of ability, which
    is the same trap the raw failure histogram fell into. Two constraints fix it:

      * The MODEL PAIR must have near-identical per-check accuracy (within
        EXEMPLAR_PC_TOL), so ability is held roughly fixed and only shape is left.
        Among such pairs, take the largest co-failure-lift gap.
      * The SCENARIO must be one where both models lost the SAME total number of
        required checks. Equal totals is what makes the resulting pass-count gap
        attributable to arrangement rather than to one model simply erring less.

    Ties break toward the family with the most rounds where exactly one of the two
    passed (those rounds carry the contrast), then toward more checks lost, then by
    name, so the choice is reproducible rather than curated.

    This is an ILLUSTRATION. The evidence for a per-model clustering claim is
    clustering_strata(); nothing here is corrected for corpus difficulty.
    """
    keys = [k for _, _, k in MODELS if (stats.get(k) or {}).get("by_task")
            and (stats.get(k) or {}).get("clustering")]
    if len(keys) < 2:
        return None
    graded = [{t for t, v in stats[k]["by_task"].items() if v.get("req_k")}
              for k in keys]
    common = sorted(set.intersection(*graded))
    if not common:
        return None

    def pc(k):
        return stats[k]["clustering"]["per_check_pct"]

    def lift(k):
        return stats[k]["clustering"]["lift"]

    pairs = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]
             if abs(pc(a) - pc(b)) <= EXEMPLAR_PC_TOL]
    if not pairs:
        return None
    a, b = max(pairs, key=lambda p: (abs(lift(p[0]) - lift(p[1])), p))
    if lift(a) < lift(b):
        a, b = b, a                      # a is always the clustered one

    fam = {}
    for t in common:
        fam.setdefault(_ROUND_RE.sub("", t), []).append(t)

    def score(g):
        ts = fam[g]
        la = sum(stats[a]["by_task"][t]["req_f"] for t in ts)
        lb = sum(stats[b]["by_task"][t]["req_f"] for t in ts)
        pa = sum(1 for t in ts if stats[a]["by_task"][t]["req_f"] == 0)
        pb = sum(1 for t in ts if stats[b]["by_task"][t]["req_f"] == 0)
        solo = sum(1 for t in ts
                   if (stats[a]["by_task"][t]["req_f"] == 0)
                   != (stats[b]["by_task"][t]["req_f"] == 0))
        return la, lb, pa, pb, solo

    cands = [(g, *score(g)) for g in fam]
    cands = [c for c in cands if c[1] == c[2] and c[3] != c[4]]
    if not cands:
        return None
    g, la, _lb, pa, pb, solo = max(
        cands, key=lambda c: (abs(c[3] - c[4]), c[5], c[1], c[0]))
    if pa < pb:                          # orient so `a` is the one that passes more
        a, b = b, a
        pa, pb = pb, pa

    rounds = []
    for t in sorted(fam[g]):
        rounds.append({
            "task": t,
            "round": t[len(g):].lstrip("_"),
            "k": stats[a]["by_task"][t]["req_k"],
            "a": stats[a]["by_task"][t]["req_f"],
            "b": stats[b]["by_task"][t]["req_f"],
        })
    # The round that makes the mechanism visible: both models failed it, so the
    # extra misses one of them spent there changed no outcome at all.
    free = max((r for r in rounds if r["a"] and r["b"]),
               key=lambda r: (abs(r["a"] - r["b"]), r["round"]), default=None)
    # Corpus-wide asymmetry the exemplar stands for, so it is not a lone anecdote.
    a_only = sum(1 for t in common if stats[a]["by_task"][t]["req_f"] == 0
                 and stats[b]["by_task"][t]["req_f"] > 0)
    b_only = sum(1 for t in common if stats[b]["by_task"][t]["req_f"] == 0
                 and stats[a]["by_task"][t]["req_f"] > 0)
    names = {k: n for n, _s, k in MODELS}
    return {
        "a": a, "b": b, "a_name": names[a], "b_name": names[b],
        "a_pc": pc(a), "b_pc": pc(b), "a_lift": lift(a), "b_lift": lift(b),
        "scenario": g, "rounds": rounds, "lost": la,
        "a_passed": pa, "b_passed": pb, "solo": solo, "free": free,
        "n_common": len(common), "a_only": a_only, "b_only": b_only,
    }

def gap_decomposition(stats, a="opus5", b="sol", draws=4000):
    """Split the pass-rate gap between two models into per-check quality vs clustering.

    `observed = independent + excess` holds by construction for every model, so
    the gap between two models splits EXACTLY into two additive terms:

        per-check quality  = indep(a) - indep(b)      (moves the independence baseline)
        failure clustering = excess(a) - excess(b)    (difference in how failures pack)

    and the two sum to the observed pass-rate gap. Nothing is estimated.

    The split is reported twice, because it is not stable under stratification --
    this corpus produces a Simpson's paradox and the sign of the per-check term
    flips:

    * POOLED uses each model's whole-corpus per-check rate `p` to build the
      independence baseline in every tier.
    * STRATIFIED recomputes `p` WITHIN each difficulty tercile and weights the
      tiers by size. Both totals are identical by construction; the components
      are not.

    Difficulty partition: the pooled required-check failure rate of the models
    OTHER THAN these two -- eight models here, deliberately NOT the nine-model
    leave-one-out axis used by clustering_strata(). Using a partition that
    excludes BOTH models keeps the axis neutral between them, which the
    leave-one-out axis is not (it would be built from a different eight-plus-one
    set for each of the two). The two partitions give genuinely different tier
    lifts -- 4.90 vs 4.48 for the easy tercile -- so the numbers are not
    interchangeable and each section must quote its own.

    Restricted, like clustering_strata(), to the tasks every model was graded on.

    Deterministic: the decomposition itself consumes no RNG at all. The paired
    tier CIs use _boot_ci with local, fixed-seed generators (see _boot_ci), on a
    seed block of their own, so adding this function cannot perturb any interval
    produced elsewhere in this script.
    """
    keys = [key for _, _, key in MODELS if (stats.get(key) or {}).get("by_task")]
    if a not in keys or b not in keys or len(keys) < 4:
        return None
    common_set: set = set()
    for i, k in enumerate(keys):
        graded = {t for t, v in stats[k]["by_task"].items() if v.get("req_k")}
        common_set = graded if i == 0 else (common_set & graded)
    common = sorted(common_set)
    if len(common) < 30:
        return None
    kf = {k: {t: (stats[k]["by_task"][t]["req_k"], stats[k]["by_task"][t]["req_f"])
              for t in common} for k in keys}

    peers = [k for k in keys if k not in (a, b)]
    diff = {}
    for t in common:
        pk = sum(kf[m][t][0] for m in peers)
        pf = sum(kf[m][t][1] for m in peers)
        diff[t] = pf / pk if pk else 0.0
    order = sorted(common, key=lambda t: diff[t])
    n = len(order)
    groups = [("easy", order[:n // 3]),
              ("middle", order[n // 3:2 * n // 3]),
              ("hard", order[2 * n // 3:])]

    def rate(mk, ids):
        tk = sum(kf[mk][t][0] for t in ids)
        tf = sum(kf[mk][t][1] for t in ids)
        return (1 - tf / tk) if tk else float("nan")

    def obs(mk, ids):
        return sum(1 for t in ids if kf[mk][t][1] == 0) / len(ids)

    def indep(mk, ids, p):
        return sum(p ** kf[mk][t][0] for t in ids) / len(ids)

    gp = {a: rate(a, common), b: rate(b, common)}

    def split(ids, within):
        """(per_check_pp, clustering_pp, total_pp) for one task group."""
        cell = {}
        for mk in (a, b):
            p = rate(mk, ids) if within else gp[mk]
            o, i = obs(mk, ids) * 100, indep(mk, ids, p) * 100
            cell[mk] = (o, i, o - i)
        return (cell[a][1] - cell[b][1],
                cell[a][2] - cell[b][2],
                cell[a][0] - cell[b][0]), cell

    tiers = []
    acc = {True: [0.0, 0.0, 0.0], False: [0.0, 0.0, 0.0]}
    for j, (name, ids) in enumerate(groups):
        w = len(ids) / n
        (s_pc, s_cl, s_tot), cell = split(ids, True)
        (p_pc, p_cl, p_tot), _ = split(ids, False)
        for within, trio in ((True, (s_pc, s_cl, s_tot)), (False, (p_pc, p_cl, p_tot))):
            for x in range(3):
                acc[within][x] += w * trio[x]
        la = co_failure_lift([kf[a][t] for t in ids])
        lb = co_failure_lift([kf[b][t] for t in ids])
        lo, hi = _boot_ci(list(range(len(ids))),
                          lambda idx, _ids=ids: (
                              co_failure_lift([kf[a][_ids[i]] for i in idx])
                              - co_failure_lift([kf[b][_ids[i]] for i in idx])),
                          seed=96000 + j, draws=draws)
        tiers.append({
            "name": name,
            "n": len(ids),
            "a_pass": round(cell[a][0], 1),
            "b_pass": round(cell[b][0], 1),
            "gap": round(cell[a][0] - cell[b][0], 1),
            "a_indep": round(cell[a][1], 1),
            "b_indep": round(cell[b][1], 1),
            "per_check": round(s_pc, 2),
            "clustering": round(s_cl, 2),
            "weighted_clustering": round((len(ids) / n) * s_cl, 2),
            "a_lift": round(la, 2),
            "b_lift": round(lb, 2),
            "paired_diff": round(la - lb, 2),
            "paired_ci": [round(lo, 2), round(hi, 2)],
        })

    la = co_failure_lift([kf[a][t] for t in common])
    lb = co_failure_lift([kf[b][t] for t in common])
    lo, hi = _boot_ci(list(range(n)),
                      lambda idx: (co_failure_lift([kf[a][order[i]] for i in idx])
                                   - co_failure_lift([kf[b][order[i]] for i in idx])),
                      seed=96009, draws=draws)
    return {
        "models": [name for name, _p, key in MODELS if key in (a, b)],
        "n_tasks": n,
        "partition_peers": len(peers),
        "draws": draws,
        "pooled": {"per_check": round(acc[False][0], 2),
                   "clustering": round(acc[False][1], 2),
                   "total": round(acc[False][2], 2)},
        "stratified": {"per_check": round(acc[True][0], 2),
                       "clustering": round(acc[True][1], 2),
                       "total": round(acc[True][2], 2)},
        "overall": {"a_lift": round(la, 2), "b_lift": round(lb, 2),
                    "paired_diff": round(la - lb, 2),
                    "paired_ci": [round(lo, 2), round(hi, 2)]},
        "tiers": tiers,
    }


def analyze(recs):
    catmix = Counter()
    cmdmix = Counter()  # sub-command (head+sub) counts -> command-surface heatmap
    othermix = Counter()  # breakdown of the "other" bucket by domain family
    stops = Counter()
    n = len(recs)
    passed = 0
    open_total = write_total = 0
    zero_open = 0
    # Step/effort accumulators (feed the markdown doc's step-effort section) and a
    # reasoning-vs-format split of the evaluator checks (semantic LLM-judged match
    # vs literal field_contains presence).
    step_counts, succ_steps, fail_steps, tail_waste = [], [], [], []
    durations = []  # wall-clock seconds per task (duration_s) -- the one efficiency
    # axis available for all ten models (Claude included, unlike token usage).
    err_steps = tot_steps = 0
    # Per bucket: total, passed, and -- for judge-backed checks -- how many
    # never yielded a verdict at all. "The judge said no", "the judge broke",
    # and "the judge was never called because the solver produced nothing" all
    # arrive as passed=False, and pooling them misreads a model that did not act
    # as one that reasoned badly. Measured on r01 the gap is enormous: 39.4% of
    # 3.5-flash's semantic checks never reached the judge versus 1.4% for terra,
    # so the pooled rate ranks the two on how often they acted, not on semantics.
    # ``field_contains`` additionally carries a disjoint sub-split. Three
    # quarters of those checks are ``assets_accessed`` gates -- did the solver
    # open the right image -- not transcription of a literal value into a
    # report. Pooling the two makes the column read as a transcription score it
    # is not, so the parts are counted separately. The union key is kept so
    # previously emitted stats.json files stay comparable.
    # ``required`` is tracked alongside ``total`` because the single most
    # load-bearing caveat in this report -- that most assets_accessed checks are
    # not scored at all, so "passed without opening anything" is a statement
    # about the rubric rather than about vision -- rested on a pair of numbers
    # that lived only in prose and could not be recomputed. Counting the
    # role=="required" subset here puts both halves of that ratio in stats.json
    # where the notes can cite them and the audit can check them.
    chk = {k: {"passed": 0, "total": 0, "required": 0, "infra": 0,
               "not_invoked": 0}
           for k in ("semantic_match", "field_contains", "other",
                     "field_contains_asset", "field_contains_content")}
    req_vectors = []  # per task: the role=="required" check outcomes -> clustering
    by_task = {}
    # Episodes the PROVIDER killed, not the model. These carry a non-empty
    # ``error`` ("504 DEADLINE_EXCEEDED. ..."), ``error_type ==
    # "provider_failure"``, an EMPTY ``stop_reason`` -- and, the trap,
    # ``steps == 0`` even though their ``trajectory`` still holds the 6-19
    # actions the model really took before the gateway timed out. Those actions
    # were already being counted into catmix, n_open and n_write, so leaving
    # ``steps`` at zero shipped a self-contradicting caption in the curated
    # timelines ("0 steps - 4 opens - 0 writes") next to a dot row that visibly
    # drew six dots, and it pulled the mean chain length down with steps that
    # demonstrably happened. Repair ``steps`` from the trajectory, give the
    # episode its own stop bucket instead of letting it hide inside "none"
    # alongside models that finished normally, and carry the counterfactual so
    # the report can state what the affected model's headline numbers would be
    # if these episodes were dropped instead of scored.
    # DO NOT reach for ``provider_impacted`` here. It is a DIFFERENT and, in
    # this corpus, perfectly DISJOINT flag -- a call that hit a hiccup and
    # recovered -- and it is False on every single one of these dead records.
    # Measured across all 24,485 episodes on disk: 29 carry provider_impacted
    # and 493 carry ``error``, with zero overlap. A counter keyed on
    # ``provider_impacted`` therefore reports "0 provider failures" for the one
    # model that actually lost episodes to the provider.
    pf_n = 0
    pf_tasks = []
    pf_codes = Counter()
    pf_step_sum = pf_score_sum = 0
    for r in recs:
        md = r.get("metadata") or {}
        succ = bool(r.get("success"))
        passed += succ
        traj = r.get("trajectory") or []
        err_msg = str(r.get("error") or "")
        if err_msg:
            pf_n += 1
            pf_tasks.append(r.get("task_id"))
            m = re.match(r"\s*(\d{3})\b", err_msg)
            pf_codes[m.group(1) if m else "unknown"] += 1
            # The counterfactual below reports `passed` against a shrunken
            # denominator, which is only the right arithmetic if none of the
            # dropped episodes was a pass.
            assert not succ, (
                f"{r.get('task_id')} is recorded as both a provider failure and "
                "a success; the provider-failure counterfactual assumes these "
                "episodes never passed")
            stop_disp = (r.get("error_type") or "provider_failure").split(":")[0]
            sr = stop_disp
        else:
            stop_disp = (r.get("stop_reason") or "none").split(":")[0]
            sr = ("reached max_steps" if stop_disp.startswith("reached max_steps")
                  else stop_disp)
        stops[sr] += 1
        steps = []
        n_open = n_write = 0
        last_gain = -1  # trajectory index of the last score-gaining step
        for i, s in enumerate(traj):
            cat, key = categorize(s.get("action", ""))
            catmix[cat] += 1
            cmdmix[key] += 1
            if cat == "other":
                othermix[other_family(key)] += 1
            if cat == "asset_open":
                n_open += 1
            if cat == "file_write":
                n_write += 1
            err = step_is_error(s)
            tot_steps += 1
            if err:
                err_steps += 1
            if float(s.get("score_delta") or 0) > 0:
                last_gain = i
            steps.append({
                "cat": cat,
                "key": key,
                "err": err,
                "act": action_text(s),
                "sd": round(float(s.get("score_delta") or 0), 2),
                "out": stdout_snippet(s),
            })
        open_total += n_open
        write_total += n_write
        if n_open == 0:
            zero_open += 1
        # step-count distribution, split by outcome, and how many trajectory
        # steps were spent after the last score gain. The harness `steps` field
        # and len(trajectory) agree on every healthy episode in the corpus, so a
        # disagreement is the provider-failure artifact described above and the
        # trajectory is the honest count. Assert the identity on everything
        # else: if the harness ever starts reporting a step count that is not
        # the chain length for a reason we have not diagnosed, this report
        # should stop rather than quietly average the two definitions together.
        st_ct = r.get("steps")
        if not err_msg:
            assert (st_ct or 0) == len(traj), (
                f"{r.get('task_id')}: harness reports steps={st_ct} but the "
                f"trajectory holds {len(traj)} actions, and the episode carries "
                "no provider error to explain the gap")
        elif (st_ct or 0) != len(traj):
            st_ct = len(traj)
        if isinstance(st_ct, (int, float)):
            step_counts.append(st_ct)
            (succ_steps if succ else fail_steps).append(st_ct)
        if err_msg:
            pf_step_sum += st_ct or 0
            pf_score_sum += float(r.get("score") or 0)
        dur = r.get("duration_s")
        if isinstance(dur, (int, float)):
            durations.append(dur)
        if steps and last_gain >= 0:
            tail_waste.append(len(steps) - 1 - last_gain)
        req_vec = []
        # Did this task's REQUIRED asset gate pass? None means the task carries
        # no such gate (or its checks came back empty, as they did for three
        # gemini-3.1-pro records), which is different from failing one and must
        # not be collapsed into False -- the caveat box ranks models over the
        # tasks where every model has this scored, and a None read as False
        # would silently penalise whichever model lost its check_results.
        asset_ok = None
        for c in (md.get("check_results") or []):
            nm = c.get("name", "")
            t = ("semantic_match" if "semantic" in nm
                 else "field_contains" if "field_contains" in nm else "other")
            # ``name`` is "effect:<effect_type> <condition>", so the asset gate
            # is identifiable by prefix. Every check lands in exactly one of the
            # three top-level buckets; a field_contains check additionally lands
            # in exactly one of the two sub-buckets.
            targets = [t]
            if t == "field_contains":
                targets.append("field_contains_asset"
                               if nm.startswith("effect:assets_accessed")
                               else "field_contains_content")
            for tname in targets:
                bucket = chk[tname]
                bucket["total"] += 1
                if c.get("role") == "required":
                    bucket["required"] += 1
                # Precedence mirrors train_eval.eval_report._compute_judge_health
                # so the two reports can never disagree about the same record.
                if not c.get("ok", True):
                    bucket["infra"] += 1
                elif not _judge_reached(c):
                    bucket["not_invoked"] += 1
                elif c.get("passed"):
                    bucket["passed"] += 1
            # Only role=="required" gates success. NB the field is `role`, not
            # `required`: a `c.get("required") is False` filter silently matches
            # nothing because the key does not exist.
            if c.get("role") == "required":
                req_vec.append(bool(c.get("passed")))
                if nm.startswith("effect:assets_accessed"):
                    asset_ok = bool(c.get("passed")) and (asset_ok is not False)
        if req_vec:
            req_vectors.append(req_vec)
        by_task[r["task_id"]] = {
            "asset_ok": asset_ok,
            "success": succ,
            "score": round(float(r.get("score") or 0), 2),
            # Repaired step count and, for a provider-killed episode, a stop
            # label that names the cause instead of the "none" it would
            # otherwise share with models that simply finished.
            "steps": st_ct,
            "stop": stop_disp,
            "n_open": n_open,
            "n_write": n_write,
            "chain": steps,
            # Per-task required-check counts, kept so clustering can be recomputed
            # ACROSS models on a common task set (see clustering_strata). k is a
            # property of the task and is identical for every model that was
            # graded on it; f is the model's own failure count.
            "req_k": len(req_vec),
            "req_f": len(req_vec) - sum(1 for x in req_vec if x),
        }
    # MECE guarantee: every action lands in exactly one category, and the "other"
    # bucket is fully accounted for by its family breakdown. If either identity
    # breaks the report would silently mislead, so fail loudly instead.
    catmix_total = sum(catmix.values())
    assert sum(cmdmix.values()) == catmix_total, (
        "cmdmix/catmix disagree -- categorization dropped actions")
    assert sum(othermix.values()) == catmix.get("other", 0), (
        "othermix does not reconcile with the 'other' bucket")

    def _mean(xs):
        return round(sum(xs) / len(xs), 1) if xs else None

    effort = {
        "step_p50": _median(step_counts),
        "step_mean": _mean(step_counts),
        "step_min": min(step_counts) if step_counts else None,
        "step_max": max(step_counts) if step_counts else None,
        "succ_step_p50": _median(succ_steps),
        "fail_step_p50": _median(fail_steps),
        "tail_waste_mean": _mean(tail_waste),
        "tail_waste_p50": _median(tail_waste),
        # Tail waste is only defined for an episode that gained score at least
        # once (there is no "last gain" to measure from otherwise), so the two
        # figures above are means over a SUBSET of the episodes. On a model
        # that rarely scores, that subset is small and unrepresentative --
        # carry its size so any reader of the number can see the denominator.
        "tail_waste_n": len(tail_waste),
        "step_err_pct": round(100 * err_steps / tot_steps, 1) if tot_steps else None,
    }
    def _split(v):
        graded = v["total"] - v["infra"] - v["not_invoked"]
        return {
            # ``passed``/``total``/``pass_pct`` keep their original meaning so
            # previously emitted stats.json files stay comparable.
            "passed": v["passed"],
            "total": v["total"],
            # How many of those checks actually gate success. For the asset
            # bucket this is the numerator of the grounding caveat: most
            # assets_accessed checks are advisory, so a model can "pass without
            # opening anything" without that saying anything about vision.
            "required": v["required"],
            "pass_pct": round(100 * v["passed"] / v["total"], 1) if v["total"] else None,
            "infra_failures": v["infra"],
            "not_invoked": v["not_invoked"],
            "graded": graded,
            # The honest semantic number: of the checks a judge actually read,
            # how many passed. Unlike pass_pct this is not depressed by a model
            # simply declining to act.
            "graded_pass_pct": round(100 * v["passed"] / graded, 1) if graded else None,
        }

    check_split = {k: _split(v) for k, v in chk.items()}
    score_sum = sum(float(r.get("score") or 0) for r in recs)
    # What this model's headline numbers would be if the provider-killed
    # episodes were DROPPED rather than scored as zeros. The report keeps them
    # (every model is then reported over the same 600 tasks) and prints this
    # alongside, so a reader can see the size of the choice instead of taking
    # it on trust. Derived, never typed: if the failure count changes, so do
    # these. ex_n is the honest denominator; `passed` needs no adjustment
    # because the assert above guarantees no dropped episode was a pass.
    ex_n = n - pf_n
    provider_failures = {
        "n": pf_n,
        "tasks": pf_tasks,
        "codes": dict(pf_codes),
        # Trajectory steps that the harness had reported as zero.
        "steps_recovered": pf_step_sum,
        "ex_n": ex_n,
        "ex_pass_pct": round(100 * passed / ex_n, 1) if ex_n else None,
        # Two decimals, unlike effort.step_mean's one: at one decimal the
        # inclusive and exclusive means round to the same number, and a
        # sentence that says "13.2 rather than 13.2" tells the reader nothing
        # about the size of the choice it is trying to expose. The inclusive
        # figure is carried at matching precision so the pair is comparable.
        "step_mean": (round(sum(step_counts) / n, 2)
                      if n and step_counts else None),
        "ex_step_mean": (round((sum(step_counts) - pf_step_sum) / ex_n, 2)
                         if ex_n and step_counts else None),
        "ex_avg_score": round((score_sum - pf_score_sum) / ex_n, 3) if ex_n else None,
    }
    return {
        "n": n,
        "passed": passed,
        "pass_pct": round(100 * passed / max(n, 1), 1),
        "avg_score": round(score_sum / max(n, 1), 3),
        "provider_failures": provider_failures,
        "catmix": dict(catmix),
        "catmix_total": catmix_total,
        "cmdmix": dict(cmdmix),
        "othermix": dict(othermix),
        "stops": dict(stops),
        "open_per_task": round(open_total / max(n, 1), 2),
        "write_per_task": round(write_total / max(n, 1), 2),
        "zero_open": zero_open,
        "duration_per_task": _mean(durations),
        "duration_total": round(sum(durations)) if durations else None,
        "effort": effort,
        "check_split": check_split,
        "clustering": failure_clustering(req_vectors),
        "by_task": by_task,
    }


# ---------------------------------------------------------------------------
# API cost accounting. Token usage lives in the interaction sidecar logs
# (log/*.interactions.jsonl), one "attempt" record per model call, NOT in the main
# trajectory jsonl. Cost sums over ALL attempts (retries are billed too). Token
# identities (verified against the logs): OpenAI/Bedrock total = input + output,
# with reasoning_tokens a subset of output and cached_tokens / cache_write_tokens
# subsets of input; Gemini total = prompt + candidates + thoughts, with thoughts
# (reasoning) billed at the output rate and no cache-read field recorded. To make
# the columns comparable, "visible output" is output minus reasoning for OpenAI and
# candidates for Gemini, so total = input + visible_output + reasoning for both.
# ---------------------------------------------------------------------------
def _rate_tier(pr, in_tokens):
    """Pick (input, output, cached, cache_write) rates for a call by its input size.
    Long-context rates apply above the model's threshold. The threshold IS crossed on
    this run -- the three Bedrock GPT models push a handful of calls past 128K -- so
    this branch is live; see the PRICING note above for the measured dollar effect."""
    if pr.get("long_threshold") and in_tokens > pr["long_threshold"]:
        return (pr["long_in"], pr["long_out"],
                pr.get("long_cached"), pr.get("long_cache_write"))
    return pr["in"], pr["out"], pr.get("cached"), pr.get("cache_write")


def _openai_call_cost(it, ot, cached, cwrite, pr):
    """USD for one OpenAI/Bedrock call. cached & cwrite are subsets of `it` billed
    at their own rates; the remainder bills at the full input rate; `ot` is the full
    output (reasoning already included) at the output rate. A missing cached /
    cache_write rate means those tokens bill as normal input (e.g. gpt-5.5)."""
    in_r, out_r, cached_r, cw_r = _rate_tier(pr, it)
    if cached_r is None:
        cached = 0
    if cw_r is None:
        cwrite = 0
    full = it - cached - cwrite
    cost = full * in_r + cached * (cached_r or 0) + cwrite * (cw_r or 0) + ot * out_r
    return cost / 1_000_000.0


def _gemini_call_cost(prompt, cand, thoughts, pr):
    """USD for one Gemini call: prompt at input rate; candidates + thoughts
    (reasoning) at the output rate. No cache-read field is recorded, so no discount."""
    in_r, out_r, _, _ = _rate_tier(pr, prompt)
    return (prompt * in_r + (cand + thoughts) * out_r) / 1_000_000.0


def _anthropic_call_cost(it, ot, cread, cwrite, pr):
    """USD for one Anthropic call. Unlike OpenAI, ``input_tokens`` (``it``) EXCLUDES
    the cache read/creation counts -- they are separate ADDITIVE buckets -- so total
    input is ``it + cread + cwrite`` and nothing is subtracted. ``ot`` is the full
    output and is billed whole at the output rate: extended-thinking tokens are
    billed INSIDE ``output_tokens`` (there is no separate reasoning rate), so costing
    ``ot`` is correct whether or not the thinking split was recorded. A missing
    cached / cache_write rate bills those tokens at the normal input rate (Claude
    here records zero cache tokens, so this is moot)."""
    in_r, out_r, cached_r, cw_r = _rate_tier(pr, it + cread + cwrite)
    cost = (it * in_r
            + cread * (cached_r if cached_r is not None else in_r)
            + cwrite * (cw_r if cw_r is not None else in_r)
            + ot * out_r)
    return cost / 1_000_000.0


def load_token_usage(root, subdir, run, key):
    """Per-task token usage + API $ from the interaction sidecar logs, summed over
    ALL attempts (retries billed too). Returns None when the model recorded no
    usage at all (Claude opus/sonnet: response.usage is empty on every attempt).

    A MISSING RATE CARD IS NOT MISSING DATA. This used to open with
    `if PRICING.get(key) is None: return None`, so a model with no price was
    never read at all and arrived at the cost table indistinguishable from one
    whose logs were empty -- nine N/A cells, of which only three (the dollar
    columns) were actually unknowable. deepseek-v4-flash is the case in point:
    it records response.usage on 10,867 of 10,867 attempts, 600/600 tasks, and
    the table still told the reader nothing was measured. Tokens are counted
    for every model now; `pr` only gates the dollar accumulation.
    """
    pr = PRICING.get(key)
    d = os.path.join(root, "eval_results", subdir, run, "log")

    def need(u, field, schema):
        """Read a REQUIRED token field, refusing to invent a zero when absent.

        `or 0` on an absent usage key is how both shipped bugs in this file were
        born: the record still has a non-empty usage object, so `any_usage` flips
        True and the model earns a 0-token, $0.00 row that renders exactly like a
        measured one -- summed into the grand total and, on the cost x pass-rate
        panel, plotted at x=0 where the frontier rule then names it the best buy.
        The dispatch below was taught the three spellings; this is the half it was
        never taught. A field that is genuinely reported as zero passes through
        (0 is not None); only a field the provider never sent raises.
        """
        v = u.get(field)
        if v is None:
            raise SystemExit(
                f"load_token_usage: {subdir}/{run} ({key}): a record dispatched "
                f"to the {schema} schema has no {field!r} (usage keys: "
                f"{sorted(u)}). Coercing this to 0 would fabricate a free, "
                "zero-token row that no downstream guard catches. Teach the "
                "dispatch the new spelling instead.")
        return v

    by_task = {}
    any_usage = False
    for fp in sorted(glob.glob(os.path.join(d, "*.interactions.jsonl"))):
        for line in open(fp):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("record_type") != "attempt":
                continue
            u = (r.get("response") or {}).get("usage") or {}
            if not u:
                continue
            any_usage = True
            # `reason_seen` counts calls that actually CARRIED a reasoning-split
            # field, so a 0 in `reason` can be told apart from an unreported split.
            agg = by_task.setdefault(r.get("task_id"), {
                "in": 0, "vis_out": 0, "reason": 0, "reason_seen": 0,
                "cached": 0, "cache_write": 0, "calls": 0, "cost": 0.0})
            if (r.get("provider") or "") == "gemini":
                p = need(u, "prompt", "gemini")
                c = need(u, "candidates", "gemini")
                # `thoughts` stays optional -- its absence is a real, meaningful
                # state (no reasoning split reported) and is already distinguished
                # from a measured zero by the `"thoughts" in u` probe below.
                th = u.get("thoughts") or 0
                agg["in"] += p
                agg["vis_out"] += c
                agg["reason"] += th
                if "thoughts" in u:
                    agg["reason_seen"] += 1
                if pr is not None:
                    agg["cost"] += _gemini_call_cost(p, c, th, pr)
            elif "anthropic" in (r.get("provider") or ""):
                # Anthropic usage schema: input_tokens EXCLUDES cache tokens (cache
                # read/creation are separate ADDITIVE buckets, unlike OpenAI where
                # cached is a subset of input_tokens). Extended thinking is billed
                # INSIDE output_tokens; the split lives in
                # output_tokens_details.thinking_tokens. This archived run predates
                # the harness fix that captures that field, so `thinking_tokens` is
                # absent here and `reason` lands at 0 (output_tokens still bills in
                # full, so dollar cost is exact). Nothing downstream reconstructs it:
                # the report sums vis_out + reason into one measured output column, so
                # a missing split changes no rendered number. A re-run with the harness
                # fix populates thinking_tokens and the split becomes available.
                it = need(u, "input_tokens", "anthropic")
                ot = need(u, "output_tokens", "anthropic")
                # Cache stays optional: both native fields are present and exactly
                # 0 on all 21,718 Anthropic attempts in this corpus (verified
                # 2026-08-14, including the `cache_creation.ephemeral_{5m,1h}`
                # sub-objects), which is the sole evidence for the "Claude records
                # zero cache tokens" statement in the cost chapter. Absence here
                # would mean no cache was offered, not that a read went unbilled.
                cr = u.get("cache_read_input_tokens") or 0
                cw = u.get("cache_creation_input_tokens") or 0
                odet = u.get("output_tokens_details") or {}
                tt = odet.get("thinking_tokens") or 0
                agg["in"] += it + cr + cw
                agg["vis_out"] += (ot - tt)
                agg["reason"] += tt
                if "thinking_tokens" in odet:
                    agg["reason_seen"] += 1
                agg["cached"] += cr
                agg["cache_write"] += cw
                if pr is not None:
                    agg["cost"] += _anthropic_call_cost(it, ot, cr, cw, pr)
            else:  # bedrock-gpt / openai / vertex-xai: the OpenAI usage FAMILY
                # TWO spellings live in this corpus and they are disjoint by
                # provider, so the inherited "they all share the OpenAI schema"
                # assumption was false:
                #   * Responses API  (provider=bedrock-gpt, the three gpt-5.6 models)
                #       input_tokens / output_tokens
                #       input_tokens_details.cached_tokens / .cache_write_tokens
                #       output_tokens_details.reasoning_tokens
                #   * Chat Completions (provider=openai, the OpenAI-COMPATIBLE SHIM
                #     the FOUR open-source models are served through)
                #       prompt_tokens / completion_tokens
                #       prompt_tokens_details / completion_tokens_details
                # Reading only the Responses spelling yields 0 in / 0 out / $0.00 for
                # every shim call while `any_usage` still flips True on the non-empty
                # dict -- i.e. a fabricated free, zero-token row that no guard catches.
                # That sentence went unheeded twice: image_token_stats read only the
                # Responses spelling and shipped "0 tokens per image" for kimi-k3, and
                # the `or 0` fallbacks here meant this branch would have done the same
                # for a fourth spelling. `need()` now closes it.
                #
                # Re-verified over EVERY attempt record in the corpus (2026-08-14,
                # 125,691 records carrying usage, zero violations):
                #   gemini            41,183   prompt / candidates
                #   vertex-anthropic  21,718   input_tokens / output_tokens
                #   bedrock-gpt       24,592   Responses spelling, 0% prompt_tokens
                #   openai (shim)     38,198   Chat-Completions spelling, 0% input_tokens
                # The shim figure is the FOUR open-source models -- deepseek-v4-pro
                # 8,453, kimi-k3 9,641, glm-5.2 9,237 and deepseek-v4-flash 10,867.
                # The previous count named only the first three (27,331) and predates
                # deepseek-v4-flash joining MODELS.
                # Dispatch on which key is present, not on provider.
                if u.get("input_tokens") is None:
                    it = need(u, "prompt_tokens", "chat-completions")
                    ot = need(u, "completion_tokens", "chat-completions")
                    idet = u.get("prompt_tokens_details") or {}
                    odet = u.get("completion_tokens_details") or {}
                else:
                    # `it` is guaranteed non-None by the dispatch condition; need()
                    # is a no-op here and kept for symmetry so a future refactor of
                    # that condition cannot quietly reopen the hole. `ot` is NOT
                    # guaranteed: the dispatch branches on the INPUT spelling and
                    # then assumes the matching output one, so a record pairing
                    # `input_tokens` with `completion_tokens` would have billed
                    # half its output at zero. Not present in this corpus -- the
                    # two spellings are disjoint per the census above -- but the
                    # premise "they are disjoint" is exactly the assumption that
                    # was already falsified once.
                    it = need(u, "input_tokens", "responses")
                    ot = need(u, "output_tokens", "responses")
                    idet = u.get("input_tokens_details") or {}
                    odet = u.get("output_tokens_details") or {}
                # Both detail objects are explicitly `null` on every shim call, so
                # normalise before .get -- `or {}` already turns null into {}.
                idet = idet if isinstance(idet, dict) else {}
                odet = odet if isinstance(odet, dict) else {}
                cr = idet.get("cached_tokens") or 0
                cw = idet.get("cache_write_tokens") or 0
                rt = odet.get("reasoning_tokens") or 0
                # Distinguish "measured zero reasoning" from "the provider never
                # reported a split". The shim emits no completion_tokens_details at
                # all, so rt would read 0 and the report would claim a MEASURED 0%
                # reasoning share for models that may well have reasoned. Count the
                # calls where the field is actually present.
                if "reasoning_tokens" in odet:
                    agg["reason_seen"] += 1
                agg["in"] += it
                agg["vis_out"] += (ot - rt)
                agg["reason"] += rt
                agg["cached"] += cr
                agg["cache_write"] += cw
                if pr is not None:
                    agg["cost"] += _openai_call_cost(it, ot, cr, cw, pr)
            agg["calls"] += 1
    return by_task if any_usage else None


def _unpriced_note(stats):
    """Name the models this chapter cannot price, and say why, in the chapter.

    Without this the reader meets N/A cells under a report that claims to cover
    the whole field and has only a hover tooltip to explain it. Derived from
    PRICING so a model added without a rate card announces itself instead of
    quietly becoming a blank row.

    The claim that the run DID record token usage is now read off the loaded
    corpus rather than asserted. The previous version said it unconditionally,
    from a code path that had never opened the logs -- it would have said the
    same thing about a model with no usage at all, and the three token columns
    it was excusing were blank for the wrong reason.
    """
    missing = [(n, k) for n, _, k in MODELS if k not in PRICING]
    if not missing:
        return ""
    one = len(missing) == 1
    names = _join([n for n, _ in missing])
    with_tok = [k for _, k in missing if stats[k]["cost"].get("has_tokens")]
    if len(with_tok) != len(missing):
        # Mixed case: some unpriced rows also have no tokens. The sentence below
        # would then overclaim for them, so refuse rather than generalise.
        raise SystemExit(
            "cost chapter: unpriced models "
            f"{sorted(k for _, k in missing)} do not all have measured token "
            f"usage (only {sorted(with_tok)} do), so the note cannot say they "
            "all recorded usage. Split the sentence deliberately.")
    tok = sum(stats[k]["cost"]["tok_total"] for _, k in missing)
    return (f'<b>{esc(names)}</b> {"is" if one else "are"} unpriced here: the '
            f'run {"does" if one else "do"} record per-call token usage '
            f'\u2014 {tok:,} tokens, shown in the token columns above like '
            f'every other row \u2014 but no first-party rate card for '
            f'{"it" if one else "them"} has been verified, and this report does '
            f'not price a model off an estimate. Only the three dollar cells '
            f'are therefore blank, and '
            f'{"it is" if one else "they are"} absent from the grand total, '
            f'from the cost/benefit scatter and from every $-ranked statement '
            f'below. ')


def summarize_cost(by_task, model_stats, key=None):
    """Roll per-task token/cost aggregates up to per-model means + run totals. Cost
    is reported per task, as a run total, and per solved task (cost of a success).

    TWO INDEPENDENT AVAILABILITIES, deliberately not one flag. `has_tokens` is
    a property of the run's logs; `has_price` is a property of PRICING, i.e. of
    what could be verified off a vendor page. The single `available` flag they
    replace conflated them, and every consumer inherited the conflation: an
    unpriced model was dropped from the token columns it could perfectly well
    fill. Renderers must now say which one they mean -- token columns gate on
    `has_tokens`, dollar columns and every $-ranked statement on `has_price`.

    When there is no rate card the cost_* fields are None rather than 0.0, so a
    renderer that forgets to check formats a visible None instead of quietly
    printing a free model into a grand total.
    """
    has_price = key is None or key in PRICING
    if not by_task:
        note = COST_NA_NOPRICE if not has_price else COST_NA_NOTE
        return {"has_tokens": False, "has_price": has_price, "note": note}
    tasks = list(by_task.values())
    n = len(tasks)

    def _sum(f):
        return sum(t[f] for t in tasks)

    total_in, total_vo, total_re = _sum("in"), _sum("vis_out"), _sum("reason")
    total_cost = _sum("cost")
    passed = model_stats.get("passed", 0)
    return {
        "has_tokens": True,
        "has_price": has_price,
        "n_tasks": n,
        "in_per_task": round(total_in / n),
        "vis_out_per_task": round(total_vo / n),
        "reason_per_task": round(total_re / n),
        # Sum of the ROUNDED components, not the rounded sum, so the rendered row
        # actually adds up: the table shows input + output = total, and
        # round(a) + round(b) != round(a + b) drifts by up to 1 token otherwise.
        # The unrounded run total lives in `tok_total` and is unaffected.
        "total_per_task": (round(total_in / n) + round(total_vo / n)
                           + round(total_re / n)),
        "cached_per_task": round(_sum("cached") / n),
        "cache_write_per_task": round(_sum("cache_write") / n),
        "calls_per_task": round(_sum("calls") / n, 1),
        # None, not 0.0, when there is no rate card: a zero here would sum into
        # the grand total as a free model and rank first on $ / solved task.
        "cost_per_task": round(total_cost / n, 4) if has_price else None,
        # run totals across all n tasks (every attempt billed): the actual money /
        # tokens the whole eval consumed for this model, not a per-task average.
        "in_total": total_in,
        "vis_out_total": total_vo,
        "reason_total": total_re,
        "tok_total": total_in + total_vo + total_re,
        "calls_total": _sum("calls"),
        "cost_total": round(total_cost, 2) if has_price else None,
        "cost_per_solved": (round(total_cost / passed, 4)
                            if has_price and passed else None),
        # Why the dollar columns are blank for this row, in the same channel the
        # empty-log case uses, so the tooltip is never silently absent.
        "note": None if has_price else COST_NA_NOPRICE,
        # True only if the provider actually REPORTED a reasoning/thinking split on
        # at least one call. False means `reason_total` is 0 because nothing was
        # reported, NOT because the model demonstrably did no reasoning -- the two
        # must never be rendered the same way. The OpenAI-compatible shim serving
        # deepseek-v4-pro, kimi-k3 and glm-5.2 emits no completion_tokens_details,
        # so all three land here at False while every other model is True.
        # Dollar cost is unaffected either way: reasoning bills inside the output
        # token count for all of these providers, so the total is exact and only
        # the visible/reasoning SPLIT is unavailable.
        "reason_measured": _sum("reason_seen") > 0,
    }


def image_token_stats(root, subdir, run):
    """Measure how many input tokens each screenshot costs, at step 0.

    These are VLM tasks: step 0 hands every model the same initial evidence bundle
    (screenshots) plus the system prompt + task text. For each task we take its first
    step-0 attempt and record (n_images, input_tokens), then fit input_tokens on
    n_images by ordinary least squares. The SLOPE is the marginal input tokens per
    screenshot -- the regression's intercept absorbs the text-only baseline (system
    prompt + task text), so the slope is that baseline removed -- and the mean
    n_images is the size of the initial bundle. Because the identical images are sent
    to every model and each provider's tokenizer is deterministic, the slope is a
    per-provider-family constant: it comes out the same across models and independent
    runs within a family, which doubles as a consistency check.

    Returns (imgs_step0_mean, tok_per_image); either is None when the run recorded no
    usage or step-0 image counts do not vary enough to fit a slope.
    """
    d = os.path.join(root, "eval_results", subdir, run, "log")
    xs, ys = [], []  # n_images, input_tokens (one point per task)
    for fp in sorted(glob.glob(os.path.join(d, "*.interactions.jsonl"))):
        seen = set()
        for line in open(fp):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("record_type") != "attempt" or r.get("step") != 0:
                continue
            tid = r.get("task_id")
            if tid in seen:
                continue
            u = (r.get("response") or {}).get("usage") or {}
            if not u:
                continue
            # THREE input-token spellings live in this corpus, not two.
            #   * gemini          -> "prompt"
            #   * Responses API   -> "input_tokens"   (bedrock-gpt, vertex-anthropic)
            #   * Chat Completions-> "prompt_tokens"  (the OpenAI-COMPATIBLE SHIM the
            #                        four open-source models are served through)
            # This function knew only the first two and fell back to `or 0`, so every
            # shim call contributed the point (n_images, 0). load_token_usage had
            # already been taught the third spelling -- and its comment predicts this
            # exact failure, "a fabricated free, zero-token row that no guard catches"
            # -- but the lesson was never carried here. The consequence shipped: with
            # y identically zero the OLS slope is exactly 0.0, and kimi-k3, the one
            # open-source model that is sent screenshots, rendered "0 tokens per
            # image" for a 3840x2160 PNG. Its real slope is ~4.8K, the HIGHEST in the
            # field -- the printed value was not merely imprecise, it inverted the
            # ranking. Dispatch on which key is present, exactly as load_token_usage
            # does; the other three open-source models send no images at step 0, so
            # their cells (already a dash, via denom == 0) do not move.
            prov = r.get("provider") or ""
            if prov == "gemini":
                it = u.get("prompt")
            elif u.get("input_tokens") is not None:
                it = u.get("input_tokens")
            else:
                it = u.get("prompt_tokens")
            if it is None:
                raise SystemExit(
                    f"image_token_stats: {subdir}/{run}: provider {prov!r} reports "
                    f"none of prompt / input_tokens / prompt_tokens in its usage "
                    f"object (keys: {sorted(u)}). Coercing this to 0 is what "
                    "produced the '0 tokens per image' cell; add the spelling "
                    "instead.")
            xs.append(len(r.get("input", {}).get("image_refs") or []))
            ys.append(it)
            seen.add(tid)
    n = len(xs)
    if n < 2:
        return None, None
    imgs_mean = round(sum(xs) / n, 1)
    sx, sy = sum(xs), sum(ys)
    denom = n * sum(x * x for x in xs) - sx * sx
    if denom == 0:  # no variation in image count -> slope is undefined
        return imgs_mean, None
    # A run that sends images but reports no input tokens at all cannot be fitted;
    # the OLS below would happily return a slope of 0.0 from it. That is the bug
    # above, and it is indistinguishable in the rendered table from a genuine
    # measurement, so refuse rather than print it.
    if sy == 0 and sx > 0:
        raise SystemExit(
            f"image_token_stats: {subdir}/{run}: {n} step-0 points carry images "
            f"(mean {imgs_mean}) but zero input tokens in total, so the fitted "
            "slope would be a fabricated 0 tokens per image. The usage schema is "
            "not being read correctly.")
    slope = (n * sum(x * y for x, y in zip(xs, ys)) - sx * sy) / denom
    return imgs_mean, round(slope)


# Scored-unit bands: tasks grouped by how many units the rubric scores them on.
# This is section 8.1's difficulty axis, and the boundaries are the ones the
# section has always used; they are kept because the distribution still fills
# all three (192 / 258 / 150 tasks), and unit_band_summary() fails the build if
# a band ever empties rather than printing a table with a hollow column.
#
# A SCORED UNIT IS A role=="required" CHECK, not an entry in check_results. The
# two differ by a quarter: this corpus carries 3,760 checks and 3,284 required
# ones, because most assets_accessed gates are advisory and score nothing (see
# the grounding caveat). 3,284 is the number the rest of the report calls
# "scored units" -- it is what failure_clustering() counts, and what puts mean
# units per task at 5.47 -- so banding on len(check_results) would shift most
# tasks one band right and quietly disagree with sections 7.1 and 8.0.
UNIT_BANDS = [
    ("2\u20134", lambda k: 2 <= k <= 4),
    ("5\u20136", lambda k: 5 <= k <= 6),
    ("7+", lambda k: k >= 7),
]

# Gold-trajectory length bands: tasks grouped by how long the expert solution is,
# so model effort/pass-rate can be read against intrinsic task length. Gold lengths
# cluster tightly around the mode (9), so the boundaries are chosen to pull the
# per-band medians as far apart as the distribution allows (8 / 9 / 13) while
# keeping every band non-trivial; the band labels also carry the raw min-max range
# so the "different-length tasks" story survives the median compression.
GOLD_BANDS = [
    ("short (\u22648)", lambda n: n <= 8),
    ("mid (9\u201311)", lambda n: 9 <= n <= 11),
    ("long (\u226512)", lambda n: n >= 12),
]
# Aggregate row pooling every gold-covered task (not a length band).
OVERALL_BAND = "all tasks (overall)"

# Hard per-run command limit enforced by the harness. A run that reaches it was
# stopped, not finished, so its step count is a censoring boundary rather than a
# measurement and is kept out of every effort figure. Asserted against the data
# in gold_summary() so a harness change cannot silently invalidate the analysis.
STEP_CAP = 20

# Below this many uncapped passes a median is too noisy to print as if precise;
# such cells are greyed and carry their count. Every cell of the long band is in
# that state (6-13 passes), which is itself the finding.
THIN_PASSES = 20

# Overshoot buckets, in steps past THAT TASK's own gold length. Inclusive bounds.
# The cap forms an implicit sixth, terminal bucket handled separately.
OVER_BUCKETS = [
    ("\u2264 gold", -10 ** 9, 0),
    ("+1..2", 1, 2),
    ("+3..5", 3, 5),
    ("+6..9", 6, 9),
    ("+10 or more", 10, 10 ** 9),
]


def _median(xs):
    s = sorted(x for x in xs if x is not None)
    m = len(s)
    if m == 0:
        return None
    h = m // 2
    return s[h] if m % 2 else (s[h - 1] + s[h]) / 2


def gold_summary(stats, gold):
    """Cross-model effort-vs-gold table data, bucketed by gold-trajectory length.

    For each length band (and overall) and each model, compute pass rate, the
    effort spent WHEN THE MODEL PASSES (median steps and median overshoot against
    each task's own gold), and the share of runs stopped by the step cap. Only
    tasks with a gold trajectory on disk are counted, so 'covered' records
    coverage. Also returns over_hist: the per-model distribution of steps minus
    gold, which is the only view in this section that shows a model with two
    modes as having two modes.
    """
    bands = GOLD_BANDS + [(OVERALL_BAND, lambda n: True)]
    out = {"covered": len(gold), "bands": [], "by_model": {}}
    for lab, pred in bands:
        gv = [n for n in gold.values() if pred(n)]
        out["bands"].append(
            {"band": lab, "n_gold": len(gv), "gold_median": _median(gv),
             "gold_min": min(gv) if gv else None,
             "gold_max": max(gv) if gv else None})
    over = [t.get("steps") for st in stats.values() for t in st["by_task"].values()
            if isinstance(t.get("steps"), (int, float)) and t["steps"] > STEP_CAP]
    assert not over, f"steps above STEP_CAP={STEP_CAP}: {sorted(set(over))[:5]}"
    # ------------------------------------------------------------------
    # Effort lives on two populations that must not be pooled, and one of them
    # is censored. Both facts broke the column this replaces.
    #
    # 1. THE RUN CAP. No run may exceed STEP_CAP commands and 1338 of the 5700
    #    gold-covered runs here stop exactly there. A capped run's step count is
    #    not a measurement of how long the model would have taken; it is the cap.
    #    Once a model's capped share passes 50% its median IS the cap
    #    (gemini-3.5-flash is 83% capped, so its median was 20.0 in every band),
    #    and the old "median actual steps / band gold median" column collapsed to
    #    cap / gold_median -- a statement about the band, not the model. It also
    #    made such a model appear to IMPROVE on longer tasks (2.50x -> 1.54x)
    #    purely because the denominator grew while the numerator stayed pinned.
    #    Capped runs are therefore reported as their own rate and excluded from
    #    every effort figure.
    #
    # 2. PASSES VS FAILURES. A failed run's step count mostly measures how long
    #    the model flailed before giving up. Pooling the two lands the summary
    #    between the modes: gemini-3.1-pro's pooled median was 17 steps, but it
    #    takes 9 when it succeeds and 20 when it fails, and only 7.5% of its runs
    #    land in the 15-17 range at all. Effort is reported over passes only, and
    #    the count travels with it so a thin cell can be marked instead of being
    #    read as precise.
    #
    # ratio_pass divides by each task's OWN gold length, never the band median.
    # The short band spans gold 6-8, so dividing every run in it by 8 understates
    # the overshoot of a run whose task only needed 6 steps.
    # ------------------------------------------------------------------
    for _, _, key in MODELS:
        bt = stats[key]["by_task"]
        rows = []
        for lab, pred in bands:
            ntot = npass_all = ncap = 0
            psteps, pratios = [], []
            for tid, n in gold.items():
                if not pred(n):
                    continue
                t = bt.get(tid)
                if not t:
                    continue
                ntot += 1
                ok = bool(t["success"])
                npass_all += 1 if ok else 0
                s = t.get("steps")
                if not isinstance(s, (int, float)):
                    continue
                if s >= STEP_CAP:
                    ncap += 1
                    continue          # censored: not a measured step count
                if ok:
                    psteps.append(s)
                    pratios.append(s / n)
            mratio = _median(pratios)
            rows.append({
                "band": lab,
                "n": ntot,
                "pass_pct": round(100 * npass_all / ntot, 1) if ntot else 0.0,
                "capped_pct": round(100 * ncap / ntot, 1) if ntot else None,
                "n_pass_uncapped": len(psteps),
                "med_steps_pass": _median(psteps),
                "ratio_pass": None if mratio is None else round(mratio, 2),
                "thin": len(psteps) < THIN_PASSES,
            })
        out["by_model"][key] = rows

    # Overshoot distribution, one row per model, over every gold-covered run.
    # The x axis is already a difference -- steps minus THAT TASK's gold length --
    # so no statistical null is needed; the expert trajectory is the reference.
    # Capped runs cannot be placed on that axis (their true length is unknown), so
    # they form a separate terminal category rather than being binned as if the
    # cap were their step count. The finite buckets plus the cap bucket partition
    # the runs exactly, so each row sums to 100 up to rounding.
    out["over_hist"] = {}
    for _, _, key in MODELS:
        bt = stats[key]["by_task"]
        counts = [0] * (len(OVER_BUCKETS) + 1)
        tot = 0
        for tid, n in gold.items():
            t = bt.get(tid)
            if not t:
                continue
            s = t.get("steps")
            if not isinstance(s, (int, float)):
                continue
            tot += 1
            if s >= STEP_CAP:
                counts[-1] += 1
                continue
            d = s - n
            for j, (_lab, lo, hi) in enumerate(OVER_BUCKETS):
                if lo <= d <= hi:
                    counts[j] += 1
                    break
        out["over_hist"][key] = {
            "n": tot,
            "pct": [round(100 * c / tot, 1) if tot else 0.0 for c in counts],
        }
    return out


def unit_band_summary(stats):
    """Section 8.1's data: pass rate and mean score banded by scored-unit count.

    The band a task sits in is a property of the TASK, not of the model that
    attempted it -- ``req_k`` is the rubric's own count and is identical for
    every model graded on that task. So the partition is cut ONCE, from a
    canonical per-task count, and every row is read off it. Deriving each row's
    bands from that model's own counts would silently compare a short-corpus
    model against a different banding than the field, which is the one error
    this table can make that a reader cannot see.

    The canonical count is the maximum over models, with the disagreement guard
    below doing the actual work: a model either reports the rubric's count or
    reports nothing at all for that task.

    Nothing-at-all happens. gemini-3.1-pro lost five episodes to provider 504s
    and those records carry no ``check_results``, so its own ``req_k`` for them
    is 0. They are DROPPED from its row rather than banded as zero-unit tasks
    or scored as failures against the full 192 / 258 / 150 -- the same choice
    failure_clustering() makes (which is why 7.1 reports it on 595 tasks /
    3,257 checks), and for the same reason: the model was never given them.
    """
    keys = [k for _, _, k in MODELS]
    kmax = {}
    for k in keys:
        for tid, t in stats[k]["by_task"].items():
            kmax[tid] = max(kmax.get(tid, 0), t["req_k"])
    # A non-zero count that differs from the canonical one would mean the rubric
    # itself is not fixed across models, and every band in this table would be
    # comparing different task sets. Refuse rather than average over it.
    clash = sorted({(tid, t["req_k"], kmax[tid]) for k in keys
                    for tid, t in stats[k]["by_task"].items()
                    if t["req_k"] and t["req_k"] != kmax[tid]})
    if clash:
        raise ValueError(
            f"scored-unit counts disagree across models on {len(clash)} "
            f"task(s), e.g. {clash[:3]} \u2014 the band partition assumes the "
            "rubric is a property of the task")

    out = {"n_tasks": len(kmax), "n_units": sum(kmax.values()),
           "bands": [], "by_model": {}, "pooled": {}, "short": {}}
    members = {}
    for lab, pred in UNIT_BANDS:
        members[lab] = [tid for tid, kk in kmax.items() if pred(kk)]
        # An empty band is not a finding, it is a broken axis: the column would
        # render as three dashes and the lead-in would still promise three bands.
        if not members[lab]:
            raise ValueError(
                f"scored-unit band {lab!r} is empty on this corpus \u2014 "
                "UNIT_BANDS no longer fits the rubric's unit distribution")
        units = sum(kmax[t] for t in members[lab])
        out["bands"].append({"band": lab, "n_tasks": len(members[lab]),
                             "n_units": units,
                             "mean_units": round(units / len(members[lab]), 2)})
    covered = sum(len(v) for v in members.values())
    if covered != len(kmax):
        stray = sorted({kmax[t] for t in kmax
                        if not any(p(kmax[t]) for _, p in UNIT_BANDS)})
        raise ValueError(f"{len(kmax) - covered} task(s) fall in no scored-unit "
                         f"band (unit counts {stray}) \u2014 UNIT_BANDS must "
                         "partition the corpus")

    for lab, _ in UNIT_BANDS:
        npass = 0
        ssum = 0.0
        ntot = 0
        for k in keys:
            bt = stats[k]["by_task"]
            sel = [bt[t] for t in members[lab] if t in bt and bt[t]["req_k"]]
            p = sum(1 for t in sel if t["success"])
            s = sum(t["score"] for t in sel)
            out["by_model"].setdefault(k, {})[lab] = {
                "n": len(sel),
                "pass_pct": round(100 * p / len(sel), 1),
                "avg_score": round(s / len(sel), 3),
            }
            if len(sel) != len(members[lab]):
                out["short"].setdefault(k, {})[lab] = len(sel)
            npass += p
            ssum += s
            ntot += len(sel)
        # Pooled over model-task attempts, so a short corpus contributes the
        # attempts it actually has rather than being weighted as a full row.
        out["pooled"][lab] = {"n": ntot,
                              "pass_pct": round(100 * npass / ntot, 1),
                              "avg_score": round(ssum / ntot, 3)}
    return out


# ----------------------------- HTML rendering ------------------------------

def esc(s):
    return html.escape(str(s), quote=True)


def _fmtnum(x):
    if x is None:
        return "\u2014"
    return f"{x:g}"


def md_inline(s):
    """Minimal inline-markdown -> HTML: **bold** and `code`. Escapes first.

    behavior_notes.json is the source of truth; the markdown doc uses the same
    strings verbatim, so the exit check can assert prose parity across both."""
    s = html.escape(str(s), quote=False)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`([^`]+?)`", r"<code>\1</code>", s)
    return s


_NUMWORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
             7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
             12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
             16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
             20: "twenty"}


def _numword(n):
    """Spell a small count. Prose that says "all nine models" goes stale the moment
    the field changes -- this corpus grew from nine to thirteen to fourteen, and
    every hardcoded "nine" became a lie at the first step, every "thirteen" at the
    second. Derive the word from len(MODELS) instead of typing it."""
    return _NUMWORDS.get(n, str(n))


def load_notes(out_dir):
    """Load the prose sidecar. Returns {} if absent so the report still builds."""
    path = os.path.join(out_dir, "behavior_notes.json")
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


# ---------------------------------------------------------------------------
# PROSE NUMBERS ARE DERIVED, NOT TYPED.
#
# behavior_notes.json used to hold literal figures -- "opus-5 passes in 8
# steps", "all nine models route the notification correctly". Every one of them
# was computed at reasoning_effort=medium over a nine-model field. When the
# field grew to thirteen and the primary effort moved to high, all of them
# silently became false. A spot check of twelve such claims found twelve wrong,
# several of them exact inversions: the model named as a round's ONLY PASS had
# become its only zero, and the model named as its only zero had become a pass.
# Nothing in the build noticed, because prose is just a string to a renderer.
#
# So the notes carry {{tokens}} where digits used to be. Each token is resolved
# against the same in-memory `stats` the surrounding chain rows are rendered
# from, which makes a stale figure unrepresentable -- there is no digit in the
# file to go stale, and a token that cannot be resolved raises and stops the
# build rather than shipping a literal "{{steps.opus5}}" into the document.
#
# The rule is narrower than it looks. Every NUMBER and every MODEL-COUNT is a
# token; the argument around them stays hand-written, because "the failure is
# interpretation, not transcription" is a claim about the corpus that no token
# can check. Tokens keep the arithmetic honest; they cannot keep the reasoning
# honest, and pretending otherwise would be its own kind of lie.
#
# Indexing note: step numbers here are 0-based chain indices, matching the
# <span class="cnum"> the reader sees in the expanded command chain directly
# below the note. curated_rounds.json numbers the same actions differently;
# never copy a step number across from that file.
# ---------------------------------------------------------------------------

# Model names as they appear in prose. Deliberately NOT MODEL_LABEL (which is
# space-separated for tight table headers) -- running prose has always
# hyphenated these, and the markdown mirror is compared byte-for-byte.
PROSE_NAME = {
    "flash": "3.5-flash", "flash36": "3.6-flash", "flash37": "3.7-flash",
    "pro": "3.1-pro", "opus": "opus-4.8", "opus5": "opus-5",
    "sonnet": "sonnet-5", "sol": "sol", "luna": "luna", "terra": "terra",
    "dsv4": "v4-pro", "dsv4f": "v4-flash", "kimi": "kimi-k3", "glm": "glm-5.2",
}

# A model name that is inert in BOTH output formats: no HTML metacharacter and
# no markdown emphasis/link punctuation. Prose shared between the HTML report
# and the markdown mirror (see _grounding_claim) can then interpolate a name
# without knowing which format it is rendering into. Every name in PROSE_NAME
# and in MODELS satisfies it today; the check exists so that the day one does
# not -- an underscore in a name would silently italicise half a markdown
# sentence -- the build stops instead of shipping mangled text.
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9.\-]*")

# Short aliases so a note can reference another curated round without spelling
# out the full scenario slug: {{payment:npass}}.
SCEN_ALIAS = {
    "payment": "payment_chargeback_dispute_v01",
    "warehouse": "warehouse_robotics_safety_halt_v02",
    "satellite": "satellite_ground_station_handover_v02",
    "clinical": "clinical_trial_dosage_conflict_v02",
    "contain": "coordinated_containment_v02",
    "lock": "distributed_lock_shared_resource_v01",
    "expense": "expense_reimbursement_conflict_v02",
    "hold": "expiring_resource_hold_v03",
}

_TOKEN_RE = re.compile(r"\{\{([^{}]+)\}\}")

# Tokens that take no argument, enforced centrally in _resolve_token_body.
# Registration is one-directional in the safe sense: forgetting to add a new
# zero-arg token here only restores the old lenient behaviour, whereas listing
# a token that does take an argument breaks every use of it loudly and at once.
_ZERO_ARG = frozenset({
    "n_models", "n_models_word", "step_cap", "n_tasks", "primary_effort",
    "opensratio", "n_asset_checks", "n_scored_asset_checks",
    "blindpass", "blindpassmodels", "blindpassscen", "spearmanopens",
    "noneident", "nonegapmodels", "otherfams",
})


class NoteTokenError(Exception):
    """A token in behavior_notes.json could not be resolved."""


def _round_rows(stats):
    """{scenario: {model_key: task}} for the eight pinned curated rounds."""
    out = {}
    for scen in CURATED_SCENARIOS:
        picked = pick_round(stats, scen)
        if picked:
            out[scen] = picked[2]
    return out


def _order(rows, keys):
    """Deterministic prose order: fastest first, then roster order. Prose lists
    models the way the reader scans them (who got there first), and an unstable
    order would churn the markdown mirror on every rebuild for no reason."""
    idx = {k: i for i, k in enumerate(PROSE_KEYS)}
    return sorted(keys, key=lambda k: (rows[k]["steps"], idx.get(k, 99)))


def _join(names):
    if not names:
        return "none"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


_FIELD_COL = {"steps": "steps", "opens": "n_open", "writes": "n_write"}


# Per-model scalars addressable by name, for {{nwhere}}. Deliberately the same
# quantities the {{zeroopen}}/{{opensper}}/... tokens expose, so a threshold
# count and the individual values it is counting can never disagree.
_SCALAR = {
    "zeroopen": lambda s: s["zero_open"],
    "opensper": lambda s: s["open_per_task"],
    "writesper": lambda s: s["write_per_task"],
    "pass": lambda s: s["pass_pct"],
    "passed": lambda s: s["passed"],
    "avgscore": lambda s: s["avg_score"],
    "acts": lambda s: sum(s["cmdmix"].values()),
    "maxsteps": lambda s: s["stops"].get("reached max_steps", 0),
    "meansteps": lambda s: s["effort"]["step_mean"],
}

# How each of those renders in prose. Split out from the token resolver, which
# used to carry the same knowledge as an if-chain, so that a token emitting a
# LIST of these values (see {{passtopvals}}) is formatted by the same code as
# the token emitting one of them. Two decimals in one place and three in the
# other, inside a single sentence, is the kind of thing no reviewer catches.
_SCALAR_FMT = {
    "zeroopen": str,
    "opensper": lambda v: f"{v:.2f}",
    "writesper": lambda v: f"{v:.2f}",
    "pass": lambda v: f"{v:.1f}",
    "passed": str,
    "avgscore": lambda v: f"{v:.3f}",
    "acts": lambda v: f"{v:,}",
    "maxsteps": str,
    "meansteps": lambda v: f"{v:.1f}",
}
# The subset addressable as a bare per-model token, {{opensper.opus5}}. Kept
# narrower than _SCALAR on purpose: _SCALAR also backs {{nwhere}} thresholds,
# and widening the prose surface is a decision, not a side effect.
_SCALAR_TOKENS = frozenset({"acts", "opensper", "writesper", "zeroopen",
                            "pass", "avgscore", "maxsteps", "passed"})
if not _SCALAR_TOKENS <= set(_SCALAR) or set(_SCALAR) != set(_SCALAR_FMT):
    raise SystemExit("_SCALAR / _SCALAR_FMT / _SCALAR_TOKENS are out of step")


def _spearman_text(xs, ys):
    """Spearman rho, rendered for prose (unicode minus, two decimals).

    Extracted from the {{spearman}} branch so {{catcorr}} cannot drift into a
    second, subtly different correlation. Ties are rank-averaged; without that
    the flat columns (thirteen of the fourteen models sit at a RAW zero on
    file_delete -- only luna is nonzero, with 2 events) would bias rho."""
    def _rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):              # average ties, else rho is biased
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for t in range(i, j + 1):
                r[order[t]] = (i + j) / 2.0
            i = j + 1
        return r
    rx, ry = _rank(xs), _rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx)
           * sum((b - my) ** 2 for b in ry)) ** 0.5
    if not den:
        raise NoteTokenError("degenerate ranks; correlation undefined")
    return f"{num / den:+.2f}".replace("+", "").replace("-", "\u2212")


def _corpus_token(fn, rest, tok, stats):
    """Resolve the corpus-wide half of the note vocabulary, or return None.

    The scenario tokens above answer questions about one curated round. The
    command-surface prose asks a different kind of question -- how often does
    this model reach for this sub-command across all 600 tasks -- so it needs
    its own tokens over stats[key]["cmdmix"] and the per-model scalars.

    It needs them badly. Every one of the five command-surface bullets was
    written against nine models at reasoning_effort=medium, and moving to
    thirteen at high did not merely shift the numbers: 3.5-flash stopped being
    the field's heaviest `asset info` and `agent list` user, and the claim that
    search collapses monotonically as capability rises stopped being true the
    moment sol -- second in the field by score -- turned up with 877 searches.
    Prose that states a rank or a superlative has to recompute it."""
    keys = [k for _, _, k in MODELS]

    def counts(verb):
        return {k: stats[k]["cmdmix"].get(verb, 0) for k in keys}

    def one_model(name):
        # Check against the roster, not against stats: the dict handed to the
        # resolver also carries the effort grid under a non-model key, and a
        # membership test would accept {{pass.effort}} and then die on a
        # KeyError far from the token that caused it.
        if name not in keys:
            raise NoteTokenError(f"unknown model key {name!r} in {tok!r}")
        return stats[name]

    def verb_and_model():
        verb, _, key = rest.rpartition(".")
        if not verb or not key:
            raise NoteTokenError(f"{fn} takes VERB.KEY: {tok!r}")
        return verb, key

    def names(ks):
        order = {k: i for i, k in enumerate(PROSE_KEYS)}
        return _join([PROSE_NAME[k] for k in sorted(ks, key=lambda k: order.get(k, 99))])

    if fn == "n_tasks":
        ns = {stats[k]["n"] for k in keys}
        if len(ns) != 1:
            raise NoteTokenError(f"models disagree on task count: {sorted(ns)}")
        return str(ns.pop())

    if fn == "primary_effort":
        return PRIMARY_EFFORT

    if fn == "opensratio":
        per = [stats[k]["open_per_task"] for k in keys]
        return f"{max(per) / min(per):.2f}"

    # The grounding caveat's two halves. Taken as the max across models because
    # a model whose episode died before evaluation contributes no check records
    # at all -- 3.1-pro's five provider timeouts cost it eight asset checks --
    # and the corpus figure is the one the caveat is about.
    if fn in ("n_asset_checks", "n_scored_asset_checks"):
        col = "total" if fn == "n_asset_checks" else "required"
        vals = [stats[k]["check_split"]["field_contains_asset"].get(col)
                for k in keys]
        if any(v is None for v in vals):
            raise NoteTokenError(
                f"check_split lacks {col!r}; rebuild stats from the corpus")
        return str(max(vals))

    # Full marks with the images never opened. This is the sharp end of the
    # grounding caveat: not "the check is unscored" in the abstract, but the
    # episodes that actually took every point without looking at an asset.
    #
    # It replaces a claim that was false. The kicker used to say a model can
    # score full marks without opening an image "and in the walkthroughs below
    # one does" -- on this corpus none of the eight curated rounds contains such
    # an episode, so the sentence pointed the reader at evidence that is not
    # there. The count, the models and the scenario spread are all derived here
    # instead, and the prose says plainly that the walkthroughs do not show it.
    if fn in ("blindpass", "blindpassmodels", "blindpassscen"):
        hits = []
        for k in keys:
            bt = stats[k].get("by_task")
            if bt is None:
                raise NoteTokenError(
                    "by_task is absent; resolve notes against the in-memory "
                    "stats rather than the slim sidecar")
            hits += [(k, tid) for tid, t in bt.items()
                     if t["score"] >= 1.0 and t["n_open"] == 0]
        if not hits:
            raise NoteTokenError(
                "no episode takes full marks with zero asset opens; the "
                "grounding kicker no longer has a referent")
        if fn == "blindpass":
            return str(len(hits))
        if fn == "blindpassmodels":
            return names({k for k, _ in hits})
        return str(len({scenario_of(tid) for _, tid in hits}))

    # ---- per-model verb counts -------------------------------------------
    if fn in ("cmd", "cmdrate", "cmdshare"):
        verb, key = verb_and_model()
        c = counts(verb)
        if key not in c:
            raise NoteTokenError(f"unknown model key {key!r} in {tok!r}")
        if sum(c.values()) == 0:
            raise NoteTokenError(f"no model ever ran {verb!r}")
        if fn == "cmd":
            return f"{c[key]:,}" if c[key] >= 10000 else str(c[key])
        if fn == "cmdrate":
            return f"{c[key] / stats[key]['n']:.2f}"
        return f"{100.0 * c[key] / sum(c.values()):.1f}"

    # ---- field-wide shape of one verb ------------------------------------
    if fn in ("cmdmax", "cmdmin", "cmdargmax", "cmdargmin", "cmdspan",
              "cmdratio", "cmd2nd", "cmdtotal"):
        c = counts(rest)
        if sum(c.values()) == 0:
            raise NoteTokenError(f"no model ever ran {rest!r}")
        hi, lo = max(c.values()), min(c.values())
        if fn == "cmdmax":
            return str(hi)
        if fn == "cmdmin":
            return str(lo)
        if fn == "cmdtotal":
            return f"{sum(c.values()):,}"
        if fn == "cmdargmax":
            return names([k for k in c if c[k] == hi])
        if fn == "cmdargmin":
            return names([k for k in c if c[k] == lo])
        if fn == "cmdspan":
            return f"{lo}\u2013{hi}"
        if fn == "cmd2nd":
            return str(sorted(c.values(), reverse=True)[1])
        # cmdratio: guard the zero denominator rather than emitting "inf".
        if lo == 0:
            raise NoteTokenError(
                f"{rest!r} has a zero floor, so a max/min ratio is undefined")
        return f"{hi / lo:.2f}"

    # {{cmdargmaxeq.v1+v2}} / {{cmdargmineq.v1+v2}} -- the model that leads (or
    # trails) EVERY one of these verbs, and a hard error if no single model
    # does.
    #
    # The corpus-level twin of {{eq}}. The sentence it was written for said
    # "`file list` and `agent list` both belong to {{cmdargmax.agent list}}":
    # an assertion about two verbs, resolved from one of them. If the two
    # leaders ever diverge the sentence keeps rendering -- naming the leader of
    # one verb, and quoting {{cmdmax.file list}}, a figure belonging to the
    # other. Asking for the common argmax makes the sentence testable.
    if fn in ("cmdargmaxeq", "cmdargmineq"):
        verbs = [v.strip() for v in rest.split("+") if v.strip()]
        if len(verbs) < 2:
            raise NoteTokenError(
                f"{fn} takes v1+v2[+v3]; got {rest!r} in {tok!r}")
        common, per = None, {}
        for v in verbs:
            c = counts(v)
            if sum(c.values()) == 0:
                raise NoteTokenError(f"no model ever ran {v!r} in {tok!r}")
            edge = (max if fn == "cmdargmaxeq" else min)(c.values())
            per[v] = {k for k in c if c[k] == edge}
            common = per[v] if common is None else (common & per[v])
        if not common:
            raise NoteTokenError(
                f"{fn} asserts one model leads all of {_join(verbs)}, but the "
                "leaders differ: "
                + "; ".join(f"{v} \u2192 {names(sorted(ks))}"
                            for v, ks in per.items())
                + f" ({tok!r}). Rewrite the sentence; do not re-point the token.")
        return names(sorted(common))

    # ---- one verb with a model held out -----------------------------------
    # 3.5-flash is the floor of both submission verbs and it barely submits at
    # all, so a raw max/min ratio describes 3.5-flash rather than the field. The
    # held-out form lets the prose say "outside 3.5-flash" and mean it.
    if fn in ("cmdspanex", "cmdratioex"):
        verb, key = verb_and_model()
        one_model(key)
        c = {k: v for k, v in counts(verb).items() if k != key}
        hi, lo = max(c.values()), min(c.values())
        if fn == "cmdspanex":
            return f"{lo}\u2013{hi}"
        if lo == 0:
            raise NoteTokenError(f"{verb!r} floors at zero without {key}")
        return f"{hi / lo:.2f}"

    # ---- rank correlation against pass rate -------------------------------
    # Spearman rather than Pearson: the question the prose asks is whether the
    # ORDER holds, and one extreme model (3.5-flash searches 2,707 times) would
    # otherwise carry a Pearson coefficient on its own.
    if fn in ("spearman", "spearmanopens"):
        ys = [stats[k]["pass_pct"] for k in keys]
        if fn == "spearmanopens":
            xs = [stats[k]["open_per_task"] for k in keys]
        else:
            c = counts(rest)
            if sum(c.values()) == 0:
                raise NoteTokenError(f"no model ever ran {rest!r}")
            xs = [c[k] for k in keys]
        return _spearman_text(xs, ys)

    # ---- per-model scalars ------------------------------------------------
    if fn in _SCALAR_TOKENS:
        return _SCALAR_FMT[fn](_SCALAR[fn](one_model(rest)))

    # {{nwhere.zeroopen.<=18}} / {{nwheremodels.zeroopen.>=50}} -- how many
    # models, and which, satisfy a threshold on a per-model scalar.
    #
    # Sentences of the form "eleven of the thirteen sit below X" are numbers
    # too, and they were the ones the nine-model text got wrong most often:
    # they read as structural observations, so nobody rechecks them when the
    # field grows from nine models to thirteen. Deriving the count means the
    # sentence cannot survive the fact changing underneath it.
    if fn in ("nwhere", "nwheremodels"):
        metric, _, cond = rest.partition(".")
        metric, cond = metric.strip(), cond.strip()
        if metric not in _SCALAR:
            raise NoteTokenError(
                f"unknown metric {metric!r} in {tok!r}; "
                f"expected one of {sorted(_SCALAR)}")
        for op in ("<=", ">=", "==", "<", ">"):
            if cond.startswith(op):
                break
        else:
            raise NoteTokenError(
                f"{fn} takes METRIC.<op><value>, e.g. "
                f"{{{{nwhere.zeroopen.<=18}}}}; got {cond!r} in {tok!r}")
        # The threshold may be a literal, or `@<model key>` meaning "that
        # model's own value of this metric".
        #
        # WHY THE SECOND FORM EXISTS. The note this was written for reads
        # "**{{nwhere.zeroopen.<=18}}** of the {{n_models}} never exceed
        # **{{zeroopen.flash}}**" -- a count computed against a TYPED 18, and a
        # bound printed from 3.5-flash's MEASURED value. They agree today by
        # coincidence of maintenance. Nothing connects them, nothing checks
        # them, and if 3.5-flash's figure moves the sentence quietly becomes
        # "12 of the 14 never exceed 22" with the 12 still counted at 18.
        # `<=@flash` makes the bound and the count one fact.
        raw = cond[len(op):].strip()
        if raw.startswith("@"):
            ref = raw[1:]
            if ref not in stats:
                raise NoteTokenError(
                    f"{fn} threshold refers to unknown model {ref!r} in {tok!r}")
            thr = _SCALAR[metric](stats[ref])
        else:
            try:
                thr = float(raw)
            except ValueError:
                raise NoteTokenError(
                    f"{fn} threshold is not a number or @model: {raw!r} in "
                    f"{tok!r}") from None
        test = {"<=": lambda v: v <= thr, ">=": lambda v: v >= thr,
                "==": lambda v: v == thr, "<": lambda v: v < thr,
                ">": lambda v: v > thr}[op]
        hit = [k for k in keys if test(_SCALAR[metric](stats[k]))]
        if fn == "nwhere":
            return str(len(hit))
        if not hit:
            raise NoteTokenError(
                f"no model satisfies {metric} {cond} in {tok!r}")
        return names(hit)

    if fn in ("actsratio", "actsargmax", "actsargmin"):
        tot = {k: sum(stats[k]["cmdmix"].values()) for k in keys}
        hi, lo = max(tot.values()), min(tot.values())
        if fn == "actsratio":
            return f"{hi / lo:.2f}"
        want = hi if fn == "actsargmax" else lo
        return names([k for k in tot if tot[k] == want])

    # ---- the wait family --------------------------------------------------
    # `wait N` is the one verb whose ARGUMENT carries the behaviour, so counting
    # invocations tells you nothing: the interesting quantities are the floor a
    # model will stoop to and the simulated seconds it burns. This group exists
    # because the previous draft reported 3.1-pro accumulating 18,460 s over 281
    # calls with single waits of 3600 s and 7200 s. At high none of that
    # survives -- the longest single wait in the corpus is 600 s, and the
    # heaviest waiter is sonnet-5.
    if fn.startswith("wait"):
        def secs(key):
            out = []
            for verb, cnt in stats[key]["cmdmix"].items():
                m = re.match(r"wait (\d+)$", verb)
                if m:
                    out += [int(m.group(1))] * cnt
            return out
        if fn in ("waitcalls", "waitmin", "waitmax", "waitmean", "waittotal"):
            one_model(rest)
            v = secs(rest)
            if not v:
                raise NoteTokenError(f"{rest} never waits")
            if fn == "waitcalls":
                return str(len(v))
            if fn == "waitmin":
                return str(min(v))
            if fn == "waitmax":
                return str(max(v))
            if fn == "waitmean":
                return f"{sum(v) / len(v):.1f}"
            return f"{sum(v):,}"
        if fn in ("waitargmaxtotal", "waitargmaxcalls", "waitargmaxsingle",
                  "waitmaxsingle"):
            per = {k: secs(k) for k in keys}
            per = {k: v for k, v in per.items() if v}
            if fn == "waitmaxsingle":
                return str(max(max(v) for v in per.values()))
            if fn == "waitargmaxsingle":
                hi = max(max(v) for v in per.values())
                return names([k for k in per if max(per[k]) == hi])
            agg = {k: (sum(v) if fn == "waitargmaxtotal" else len(v))
                   for k, v in per.items()}
            hi = max(agg.values())
            return names([k for k in agg if agg[k] == hi])
        raise NoteTokenError(f"unknown wait token {tok!r}")

    # ---- opening move -----------------------------------------------------
    # {{step0.VERB.KEY}} and its field-wide forms. "asset list is the step-0
    # action in 89.0-100% of tasks for every model" is a FOURTEEN-model claim about
    # 8,400 chains; it cannot be spot-checked by hand, which is the whole reason it
    # is a token. Re-measured 2026-08-14: floor 89.0% (claude-opus-4.8), ceiling
    # 100.0% (both flash generations, all three GPT models). An earlier revision of
    # this comment said "85.8-100% ... 7,800 chains"; the 7,800 was simply the
    # thirteen-model count, but the 85.8 floor does not reproduce at thirteen either
    # -- adding a model can only lower a floor, never raise it to 89.0 -- so it is
    # retired as unverifiable rather than rescaled. The shipped number comes from
    # the token and is correct either way.
    if fn in ("step0", "step0min", "step0max", "step0argmin", "step0argmax"):
        def share(verb, key):
            chains = [t["chain"] for t in stats[key]["by_task"].values()
                      if t["chain"]]
            if not chains:
                raise NoteTokenError(f"{key} has no chains")
            return 100.0 * sum(ch[0].get("key") == verb
                               for ch in chains) / len(chains)
        if fn == "step0":
            verb, key = verb_and_model()
            one_model(key)
            return f"{share(verb, key):.1f}"
        # The field-wide forms take a bare verb, and an absent or misspelt one
        # used to sail through: share() would count the chains whose step-0 key
        # equals "" -- none of them -- and the token would resolve to a serene
        # "0.0". {{step0min}} without its argument rendered "0.0-100.0%" into
        # the kicker and nothing objected. A verb no model ever opens with is
        # not a measurement of zero; it is a broken token.
        if not rest:
            raise NoteTokenError(f"{fn} takes a verb, e.g. "
                                 f"{{{{{fn}.asset list}}}}: {tok!r}")
        sh = {k: share(rest, k) for k in keys}
        if not any(sh.values()):
            raise NoteTokenError(f"{rest!r} is never a step-0 action for any "
                                 f"model in {tok!r}")
        if fn == "step0min":
            return f"{min(sh.values()):.1f}"
        if fn == "step0max":
            return f"{max(sh.values()):.1f}"
        want = min(sh.values()) if fn == "step0argmin" else max(sh.values())
        return names([k for k in sh if abs(sh[k] - want) < 5e-2])

    # ---- action-category share -------------------------------------------
    # {{cat.file_read+message_read.flash}} -- one or more categories joined by
    # "+", as a percentage of that model's own total actions. The multi-category
    # form exists because the section's central claim is about READING versus
    # PRODUCING, and each side of that contrast is two categories added together;
    # written out by hand it was two subtractions the prose could get wrong, and
    # in the nine-model text it did.
    #
    # Every name is checked against CAT_ORDER. An unknown category summing to a
    # tranquil 0.0% is the {{step0min}} failure wearing a different hat.
    if fn in ("cat", "catn", "catargmax", "catargmin", "catmax", "catmin",
              "catspan", "catratio"):
        needs_model = fn in ("cat", "catn")
        if needs_model:
            spec, _, key = rest.rpartition(".")
            if not spec or not key:
                raise NoteTokenError(f"{fn} takes CATS.KEY: {tok!r}")
            one_model(key)
        else:
            spec, key = rest, ""
        cats = [c.strip() for c in spec.split("+") if c.strip()]
        if not cats:
            raise NoteTokenError(f"{fn} takes at least one category: {tok!r}")
        unknown = [c for c in cats if c not in CAT_ORDER]
        if unknown:
            raise NoteTokenError(
                f"unknown action category {unknown!r} in {tok!r}; "
                f"expected one of {CAT_ORDER}")
        raw = {k: sum(stats[k]["catmix"].get(c, 0) for c in cats) for k in keys}
        if sum(raw.values()) == 0:
            raise NoteTokenError(f"no model ever ran {spec!r}")
        sh = {k: 100.0 * raw[k] / stats[k]["catmix_total"] for k in keys}
        if fn == "catn":
            return f"{raw[key]:,}"
        if fn == "cat":
            return f"{sh[key]:.1f}"
        hi, lo = max(sh.values()), min(sh.values())
        if fn == "catmax":
            return f"{hi:.1f}"
        if fn == "catmin":
            return f"{lo:.1f}"
        if fn == "catspan":
            return f"{lo:.1f}\u2013{hi:.1f}"
        if fn == "catargmax":
            return names([k for k in sh if abs(sh[k] - hi) < 5e-2])
        if fn == "catargmin":
            return names([k for k in sh if abs(sh[k] - lo) < 5e-2])
        if lo <= 0:
            raise NoteTokenError(
                f"{spec!r} floors at zero, so a max/min ratio is undefined")
        return f"{hi / lo:.2f}"

    # {{catcorr.file_read+message_read.file_write+message_send}} -- Spearman
    # between two category shares across the field.
    #
    # This guards a claim about geometry rather than about any one number. The
    # nine-model text said the field lay on a single read-versus-produce line,
    # which is a statement that the two shares trade off against each other;
    # they do not, and no individual cell in the table shows that. Only the
    # correlation does.
    if fn == "catcorr":
        a, _, b = rest.partition(".")
        vecs = []
        for spec in (a, b):
            cats = [c.strip() for c in spec.split("+") if c.strip()]
            if not cats:
                raise NoteTokenError(
                    f"catcorr takes SPEC.SPEC, got {rest!r} in {tok!r}")
            unknown = [c for c in cats if c not in CAT_ORDER]
            if unknown:
                raise NoteTokenError(
                    f"unknown action category {unknown!r} in {tok!r}; "
                    f"expected one of {CAT_ORDER}")
            v = [sum(stats[k]["catmix"].get(c, 0) for c in cats)
                 / stats[k]["catmix_total"] for k in keys]
            if not any(v):
                raise NoteTokenError(f"no model ever ran {spec!r}")
            vecs.append(v)
        return _spearman_text(*vecs)

    # {{cattop.3.asset_open}} -- the N heaviest users of a category, and
    # {{catblock.asset_open}} -- the provider whose models hold the top slots
    # outright, which raises if no provider does.
    #
    # catblock exists because the claim it carries is structural, not numeric:
    # "the open-source block are the three heaviest asset_open users in the
    # field" is the kind of sentence that survives in a document long after the
    # ordering that justified it has gone, because every individual number in
    # the neighbourhood still renders fine. Making the grouping itself the
    # token means the build fails the day it stops being true.
    #
    # THREE MORE, ADDED AFTER A NEAR-MISS. The note that uses catblock also
    # quoted the three leaders' shares as {{cat.asset_open.glm}} /
    # {{cat.asset_open.kimi}} / {{cat.asset_open.dsv4}} -- pinned to MODEL KEYS
    # while the names beside them came from cattop, pinned to RANK. Adding
    # deepseek-v4-flash put a fourth model in the open-source block and a new
    # model in third place, and the sentence rendered "glm-5.2, kimi-k3 and
    # v4-flash ... at 48.8%, 43.6% and 32.5%" -- the third number belonging to
    # v4-pro, who was no longer the third name. Every token resolved; the
    # sentence was false. So:
    #   {{catnth.I.CAT}}     -- the I-th ranked share, so a value can never
    #                           drift away from the name it sits beside
    #   {{cattop.block.CAT}} -- names sized by the block itself, not by a
    #                           literal N that has to be edited to match
    #   {{catblockn.CAT}}    -- that block's size as a word
    if fn == "catnth":
        num, _, cat = rest.partition(".")
        try:
            i = int(num)
        except ValueError:
            raise NoteTokenError(f"catnth takes I.CATEGORY: {tok!r}") from None
        if cat not in CAT_ORDER:
            raise NoteTokenError(f"unknown action category {cat!r} in {tok!r}")
        if not 1 <= i <= len(keys):
            raise NoteTokenError(f"catnth out of range 1..{len(keys)}: {tok!r}")
        sh = {k: 100.0 * stats[k]["catmix"].get(cat, 0) / stats[k]["catmix_total"]
              for k in keys}
        ranked = sorted(keys, key=lambda k: -sh[k])
        return f"{sh[ranked[i - 1]]:.1f}"

    if fn in ("cattop", "catblock", "catblockn", "catvals", "catnext"):
        if fn == "cattop":
            num, _, cat = rest.partition(".")
            if num == "block":
                n = None          # sized below, once the block is known
            else:
                try:
                    n = int(num)
                except ValueError:
                    raise NoteTokenError(
                        f"cattop takes N.CATEGORY or block.CATEGORY: "
                        f"{tok!r}") from None
            if n is not None and not 1 <= n <= len(keys):
                raise NoteTokenError(f"cattop out of range 1..{len(keys)}: {tok!r}")
        else:
            cat, n = rest, None
        if cat not in CAT_ORDER:
            raise NoteTokenError(
                f"unknown action category {cat!r} in {tok!r}; "
                f"expected one of {CAT_ORDER}")
        sh = {k: 100.0 * stats[k]["catmix"].get(cat, 0) / stats[k]["catmix_total"]
              for k in keys}
        if not any(sh.values()):
            raise NoteTokenError(f"no model ever ran {cat!r}")
        ranked = sorted(keys, key=lambda k: -sh[k])

        if fn == "cattop" and n is not None:
            pick = ranked[:n]
            edge = sh[pick[-1]]
            ties = [k for k in keys if abs(sh[k] - edge) < 5e-2]
            if len(ties) > 1 and not set(ties) <= set(pick):
                raise NoteTokenError(
                    f"cattop cuts through a tie at {edge:.1f}% ({names(ties)}); "
                    f"the rank {tok!r} asserts is not well defined")
            return _join([PROSE_NAME[k] for k in pick])

        # Block-sized forms: catblock, catblockn and cattop.block all assert the
        # SAME structural fact, so they share one validation. Whichever of them a
        # note uses, the build fails together the day the block stops leading.
        prov = PROVIDER[ranked[0]]
        block = [k for k in keys if PROVIDER[k] == prov]
        head = ranked[:len(block)]
        if set(head) != set(block):
            raise NoteTokenError(
                f"no provider holds the top {cat!r} slots outright: the leaders "
                f"are {names(head)} but {prov}'s models are {names(block)}")
        edge = sh[head[-1]]
        nxt = ranked[len(block):]
        if nxt and abs(sh[nxt[0]] - edge) < 5e-2:
            raise NoteTokenError(
                f"{prov}'s hold on the top {cat!r} slots is a tie with "
                f"{PROSE_NAME[nxt[0]]}, not a lead, in {tok!r}")
        if fn == "cattop":
            return _join([PROSE_NAME[k] for k in head])
        if fn == "catblockn":
            return _numword(len(block))
        if fn == "catvals":
            # The whole block's shares, in the same rank order cattop.block
            # names them. Emitted as ONE string precisely so the note cannot
            # carry a per-slot index that has to be kept equal to the block
            # size by hand -- which is the failure this whole group exists to
            # prevent.
            return _join([f"{sh[k]:.1f}%" for k in head])
        if fn == "catnext":
            if not nxt:
                raise NoteTokenError(
                    f"no model below the {prov} block on {cat!r}, so there is "
                    f"no 'next model down' for {tok!r}")
            return f"{sh[nxt[0]]:.1f}%"
        return prov

    # ---- stop reasons ------------------------------------------------------
    # {{stop.missing_required_side_effect.sonnet}} and its field-wide forms.
    # Reason strings are validated against what the corpus actually contains,
    # so a renamed or misspelled reason fails the build instead of resolving to
    # a plausible-looking zero.
    if fn in ("stop", "stopargmax", "stopargmin", "stopmax", "stopmin",
              "stopspan", "stopshare"):
        if fn in ("stop", "stopshare"):
            reason, _, key = rest.rpartition(".")
            if not reason or not key:
                raise NoteTokenError(f"{fn} takes REASON.KEY: {tok!r}")
            one_model(key)
        else:
            reason, key = rest, ""
        seen = set()
        for k in keys:
            seen.update(stats[k]["stops"])
        if reason not in seen:
            raise NoteTokenError(
                f"unknown stop reason {reason!r} in {tok!r}; "
                f"observed reasons are {sorted(seen)}")
        c = {k: stats[k]["stops"].get(reason, 0) for k in keys}
        hi, lo = max(c.values()), min(c.values())
        if fn == "stop":
            return f"{c[key]:,}"
        if fn == "stopshare":
            return f"{100.0 * c[key] / stats[key]['n']:.1f}"
        if fn == "stopmax":
            return f"{hi:,}"
        if fn == "stopmin":
            return f"{lo:,}"
        if fn == "stopspan":
            return f"{lo:,}\u2013{hi:,}"
        want = hi if fn == "stopargmax" else lo
        return names([k for k in c if c[k] == want])

    # {{stopsum.REASON}} -- the reason's corpus-wide total, and {{stoptop.REASON}}
    # / {{stoptopmodels.REASON}} -- how many models, and which, have that reason
    # as their largest NON-CLEAN outcome. The last two exist because the claim
    # they replace ("the field's dominant failure, for every model but one") is
    # a statement about per-model argmax that the old prose asserted by eye and
    # got wrong: at effort=high the field splits into two failure regimes. Nine
    # models take missing_required_side_effect as their largest non-clean outcome
    # and FIVE take reached max_steps (re-counted 2026-08-14 over all fourteen; the
    # split was 9/4 at thirteen, and deepseek-v4-flash joined the max_steps side).
    if fn in ("stopsum", "stoptop", "stoptopmodels"):
        seen = set()
        for k in keys:
            seen.update(stats[k]["stops"])
        if rest not in seen:
            raise NoteTokenError(
                f"unknown stop reason {rest!r} in {tok!r}; "
                f"observed reasons are {sorted(seen)}")
        if fn == "stopsum":
            return f"{sum(stats[k]['stops'].get(rest, 0) for k in keys):,}"
        top = []
        for k in keys:
            bad = {r: v for r, v in stats[k]["stops"].items() if r != "none"}
            if bad and max(bad, key=lambda r: bad[r]) == rest:
                top.append(k)
        if fn == "stoptop":
            return str(len(top))
        if not top:
            raise NoteTokenError(
                f"{rest!r} is no model's largest non-clean outcome in {tok!r}")
        return names(top)

    # {{noneident}} / {{nonegap.KEY}} / {{nonegapmodels}}. `stop == "none"` is
    # not "ended tidily", it is "passed every required check" -- the two counts
    # are identical episode for episode. They used NOT to be: 3.1-pro's five
    # provider timeouts arrived with an empty stop_reason, fell into the "none"
    # bucket, and inflated that column by exactly five. analyze() now files a
    # provider-killed episode under its own `provider_failure` stop, which
    # CLOSES the gap at the source -- so {{nonegapmodels}} raises by design and
    # the prose that named an exception has been retired in favour of the
    # {{pf...}} family below. The tokens stay because the identity is worth
    # re-asserting on every build: if a future corpus reopens a gap,
    # {{noneident}} stops reading the full roster and the note has to say why.
    if fn in ("noneident", "nonegap", "nonegapmodels"):
        gap = {k: stats[k]["stops"].get("none", 0) - stats[k]["passed"]
               for k in keys}
        if fn == "nonegap":
            one_model(rest)
            return f"{gap[rest]:,}"
        off = [k for k in keys if gap[k]]
        if fn == "noneident":
            return str(len(keys) - len(off))
        if not off:
            raise NoteTokenError(
                "every model's clean-stop count equals its pass count; the "
                "exception the prose describes no longer exists")
        return names(off)

    # {{passunder.KEY}} -- the provider blocks that OUTRANK this model outright,
    # phrased for the sentence "…passes 14.7%, below {{passunder.glm}} model."
    # The literal it replaces ("below every Claude and every OpenAI model") was
    # a hand-checked cross-provider comparative sitting one roster change away
    # from being false: it is a statement about seven other models, none of
    # which the sentence names, and nothing in the build was checking it.
    if fn == "passunder":
        one_model(rest)
        mine = stats[rest]["pass_pct"]
        blocks = [p for p in PROVIDER_ORDER
                  if p != PROVIDER[rest]
                  and all(stats[k]["pass_pct"] > mine
                          for k in keys if PROVIDER[k] == p)]
        if not blocks:
            raise NoteTokenError(
                f"{tok!r}: no provider block outranks "
                f"{PROSE_NAME.get(rest, rest)} ({mine}%) on every member, so "
                "there is no 'below every X' left to claim")
        return _join([f"every {p}" for p in blocks])

    # {{pfmodels}} / {{pfn.KEY}} / {{pftotal}} / {{pfcode}} / {{pfex.FIELD.KEY}}
    # -- episodes the PROVIDER killed, not the model. These are the only records
    # in the corpus whose reported step count disagreed with their own
    # trajectory, and the report makes an editorial choice about them: keep them,
    # scored as the zeros they are, so that every model is reported over the same
    # 600 tasks. That choice is defensible but it is a choice, so the prose is
    # required to show its size. {{pfex.*}} is the counterfactual -- what the
    # affected model would read if the episodes were dropped instead -- computed
    # in analyze() from the same records, never typed by hand.
    if fn in ("pfmodels", "pfn", "pftotal", "pfcode", "pfex", "pfsteps"):
        pf = {k: (stats[k].get("provider_failures") or {}) for k in keys}
        hit = [k for k in keys if pf[k].get("n")]
        if fn == "pftotal":
            return f"{sum(pf[k].get('n', 0) for k in keys):,}"
        if fn == "pfn":
            one_model(rest)
            return f"{pf[rest].get('n', 0):,}"
        if fn == "pfsteps":
            # Trajectory actions the harness had reported as zero steps.
            one_model(rest)
            return f"{pf[rest].get('steps_recovered', 0):,}"
        if not hit:
            raise NoteTokenError(
                "no model in this cut lost an episode to a provider failure, so "
                f"the prose around {tok!r} describes a corpus that no longer "
                "exists")
        if fn == "pfmodels":
            return names(hit)
        if fn == "pfcode":
            codes = sorted({c for k in hit for c in pf[k].get("codes", {})})
            if len(codes) != 1:
                raise NoteTokenError(
                    f"provider failures span HTTP codes {codes}, but {tok!r} "
                    "states a single one")
            return codes[0]
        field, _, key = rest.rpartition(".")
        one_model(key)
        col = {"pass": "ex_pass_pct", "steps": "ex_step_mean",
               "score": "ex_avg_score", "n": "ex_n"}.get(field)
        if col is None:
            raise NoteTokenError(
                f"{tok!r}: unknown counterfactual field {field!r}; expected one "
                "of pass, steps, score, n")
        # Quoting "what it would read without the failures" for a model that had
        # none prints the headline number twice and implies a correction that
        # was never made.
        if not pf[key].get("n"):
            raise NoteTokenError(
                f"{tok!r} quotes a drop-the-failures counterfactual for "
                f"{PROSE_NAME.get(key, key)}, which lost no episodes to the "
                "provider")
        v = pf[key][col]
        return f"{v:,}" if col == "ex_n" else f"{v}"

    # {{passtop.N}} / {{passbottom.N}} -- the N best or worst scorers by name.
    # Naming a rank in prose is the single most drift-prone thing the document
    # does; the nine-model text named a "top three" that the thirteen-model cut
    # reorders.
    def _pass_pick(which, count_txt):
        """The `count_txt` best or worst models by pass rate, in rank order."""
        try:
            n = int(count_txt)
        except ValueError:
            raise NoteTokenError(f"{fn} takes a count: {tok!r}") from None
        if not 1 <= n <= len(keys):
            raise NoteTokenError(f"{fn} out of range 1..{len(keys)}: {tok!r}")
        ranked = sorted(keys, key=lambda k: -stats[k]["pass_pct"])
        pick = ranked[:n] if which == "top" else list(reversed(ranked))[:n]
        edge = stats[pick[-1]]["pass_pct"]
        ties = [k for k in keys if stats[k]["pass_pct"] == edge]
        if len(ties) > 1 and not set(ties) <= set(pick):
            raise NoteTokenError(
                f"{fn} cuts through a tie at {edge}% ({names(ties)}); the "
                f"rank {tok!r} asserts is not well defined")
        return pick

    if fn in ("passtop", "passbottom"):
        return _join([PROSE_NAME[k]
                      for k in _pass_pick(fn[4:], rest)])

    # {{passtopvals.STAT.N}} -- STAT for those same N models, IN THE SAME ORDER.
    #
    # This exists because of a bug shipped in the sibling note: a sentence named
    # its models with a rank-indexed token and then gave their numbers with
    # key-pinned ones ({{cattop.3.asset_open}} ... {{cat.asset_open.dsv4}}).
    # Every token resolved, so nothing failed; the names simply stopped
    # belonging to the numbers when the ranking moved. Any sentence pairing a
    # ranked list of names with a list of values must draw both from one
    # ordering, and this is how.
    if fn in ("passtopvals", "passbottomvals"):
        stat, _, cnt = rest.partition(".")
        if stat not in _SCALAR or not cnt:
            raise NoteTokenError(
                f"{fn} takes <stat>.<count> where stat is one of "
                f"{sorted(_SCALAR)}: {tok!r}")
        which = "top" if fn == "passtopvals" else "bottom"
        return _join([_SCALAR_FMT[stat](_SCALAR[stat](stats[k]))
                      for k in _pass_pick(which, cnt)])

    # {{otherfams}} -- the scenario-specific verb families behind the `other`
    # column, read off the data. The hand-typed list had drifted: it named nine
    # families and the corpus has more.
    if fn == "otherfams":
        fams = set()
        for k in keys:
            fams.update(stats[k]["othermix"])
        if not fams:
            raise NoteTokenError("no `other` families observed")
        heads = sorted({f.split(" (")[0].strip() for f in fams})
        return _join([f"`{h}`" for h in heads])

    # ---- the step cap, crossed with the outcome ----------------------------
    # {{cap.n}}, {{cap.passed}}, {{cap.uncapped_pct}} and friends. The per-model
    # `stops` counter can say how often the cap was hit but not what became of
    # those episodes, so this walks by_task and builds the cross-tab.
    #
    # by_task spells the reason "reached max_steps=20", carrying the cap value;
    # the `stops` counter normalises that away. Matching on the prefix is what
    # keeps the two in agreement -- an equality test against "reached max_steps"
    # silently matches nothing and reports a serene zero capped episodes.
    #
    # The prose quotes the pass count rather than asserting "none of them", so a
    # future corpus in which a capped episode does pass rewrites the sentence
    # instead of contradicting it.
    if fn == "cap":
        capped, uncapped = [], []
        for k in keys:
            bt = stats[k].get("by_task")
            if bt is None:
                raise NoteTokenError(
                    "by_task is absent; resolve notes against the in-memory "
                    "stats rather than the slim sidecar")
            for t in bt.values():
                (capped if str(t["stop"]).startswith("reached max_steps")
                 else uncapped).append(t)
        if not capped:
            raise NoteTokenError("no episode hit the step cap")
        cp = sum(1 for t in capped if t["success"])
        up = sum(1 for t in uncapped if t["success"])
        table = {
            "n": f"{len(capped):,}",
            "passed": f"{cp:,}",
            "pct": f"{100.0 * len(capped) / (len(capped) + len(uncapped)):.1f}",
            "uncapped_n": f"{len(uncapped):,}",
            "uncapped_passed": f"{up:,}",
            "uncapped_pct": (f"{100.0 * up / len(uncapped):.2f}"
                             if uncapped else "0.00"),
            "mean_score": f"{sum(t['score'] for t in capped) / len(capped):.2f}",
            "max_score": f"{max(t['score'] for t in capped):.2f}",
        }
        if rest not in table:
            raise NoteTokenError(
                f"unknown cap field {rest!r} in {tok!r}; "
                f"available: {sorted(table)}")
        return table[rest]

    # ---- one cell of the effort sweep --------------------------------------
    # {{cell.capped_pct.flash@low}}. Lets prose in the primary-setting sections
    # reach across to a different effort level without hard-coding the figure,
    # which is how "at low it caps on 38.7% of episodes" survived into a
    # document whose every other number had moved.
    if fn == "cell":
        field, _, name = rest.rpartition(".")
        if not field or not name:
            raise NoteTokenError(f"{fn} takes FIELD.KEY@LEVEL: {tok!r}")
        cells = (stats.get("effort") or {}).get("cells") or {}
        if name not in cells:
            raise NoteTokenError(
                f"unknown sweep cell {name!r} in {tok!r}; "
                f"{len(cells)} cells available")
        if field not in cells[name]:
            raise NoteTokenError(
                f"unknown cell field {field!r} in {tok!r}; "
                f"available: {sorted(cells[name])}")
        v = cells[name][field]
        return f"{v:,}" if isinstance(v, int) else f"{v}"

    # ---- step / termination statistics -------------------------------------
    # {{stepstat.step_mean.sonnet}}. The field name is a key of the per-model
    # "effort" block, checked against that block rather than assumed, because
    # these are the numbers most likely to be renamed out from under the prose.
    if fn in ("stepstat", "stepstatmax", "stepstatmin", "stepstatargmax",
              "stepstatargmin", "stepstatspan", "stepstatratio"):
        if fn == "stepstat":
            field, _, key = rest.rpartition(".")
            if not field or not key:
                raise NoteTokenError(f"{fn} takes FIELD.KEY: {tok!r}")
            one_model(key)
        else:
            field, key = rest, ""
        avail = stats[keys[0]]["effort"]
        if field not in avail:
            raise NoteTokenError(
                f"unknown step field {field!r} in {tok!r}; "
                f"available: {sorted(avail)}")
        v = {k: stats[k]["effort"][field] for k in keys}
        if any(x is None for x in v.values()):
            raise NoteTokenError(f"{field!r} is undefined for some model")
        show = ((lambda x: f"{x:,}") if field.endswith("_n")
                else (lambda x: f"{x:.1f}"))
        hi, lo = max(v.values()), min(v.values())
        if fn == "stepstat":
            return show(v[key])
        if fn == "stepstatmax":
            return show(hi)
        if fn == "stepstatmin":
            return show(lo)
        if fn == "stepstatspan":
            return f"{show(lo)}\u2013{show(hi)}"
        if fn == "stepstatargmax":
            return names([k for k in v if v[k] == hi])
        if fn == "stepstatargmin":
            return names([k for k in v if v[k] == lo])
        if lo <= 0:
            raise NoteTokenError(
                f"{field!r} floors at zero, so a max/min ratio is undefined")
        return f"{hi / lo:.2f}"

    return None


def _field_arg(fn, tok, name):
    if name not in _FIELD_COL:
        raise NoteTokenError(
            f"{fn} takes steps|opens|writes, got {name!r} in {tok!r}")
    return _FIELD_COL[name]


def _eq_members(who, tok, row, scen):
    """Who {{eq}} is asserting an equality over: an explicit list, or a group.

    `k1+k2` names the models outright. `@SELECTOR` derives them, and that form
    is the one that matters: the sentences this token was written for say
    things like "{{argmaxof.opens}} open the most assets each, and BOTH finish
    on the {{score.kimi}} plateau". The left half is derived from the corpus and
    the right half was pinned to one model, so the plateau claim covered
    whichever models the argmax happened to return -- checked for exactly one of
    them. `@argmaxof.opens` makes the group that is named and the group that is
    checked the same group by construction.
    """
    who = who.strip()
    if who.startswith("@"):
        sel, _, arg = who[1:].partition(".")
        if sel == "passers":
            ks = [k for k in row if row[k]["success"]]
        elif sel == "at":
            try:
                want = float(arg)
            except ValueError:
                raise NoteTokenError(
                    f"eq: @at takes a score, got {arg!r} in {tok!r}") from None
            ks = [k for k in row if abs(row[k]["score"] - want) < 1e-9]
        elif sel in ("argmaxof", "argminof"):
            col = _field_arg("eq", tok, arg)
            edge = (max if sel == "argmaxof" else min)(
                row[k][col] for k in row)
            ks = [k for k in row if row[k][col] == edge]
        else:
            raise NoteTokenError(
                f"eq: unknown group selector @{sel} in {tok!r}; expected "
                "@passers, @at.<score>, @argmaxof.<field> or @argminof.<field>")
        if len(ks) < 2:
            raise NoteTokenError(
                f"eq: group {who} selects {len(ks)} model(s) in {scen} "
                f"({tok!r}); an equality claim needs at least two")
        return _order(row, ks)

    picks = [w.strip() for w in who.split("+") if w.strip()]
    if len(picks) < 2:
        raise NoteTokenError(
            f"eq takes FIELD.k1+k2[+k3] or FIELD.@GROUP; got {who!r} in {tok!r}")
    missing = [w for w in picks if w not in row]
    if missing:
        raise NoteTokenError(
            f"eq: {', '.join(missing)} absent from {scen} in {tok!r}")
    return picks


def _resolve_token(tok, rows, scen, stats):
    """Resolve one {{token}}, treating a leading capital as "capitalise me".

    Sentence-initial counts are why: {{nat_word.0.33}} renders "three", which is
    correct mid-sentence and wrong at the start of one. Freezing a capitalised
    twin of every counting function would double the vocabulary, so instead an
    initial capital on the function name means "resolve lowercase, then upcase
    the first character" -- {{Nat_word.0.33}} -> "Three"."""
    body = tok.strip()
    if ":" in body:
        alias, tail = body.split(":", 1)
        prefix = alias.strip() + ":"
    else:
        prefix, tail = "", body
    fn0, dot, arg = tail.strip().partition(".")
    fn0 = fn0.strip()
    if fn0[:1].isupper():
        low = prefix + fn0[0].lower() + fn0[1:] + dot + arg
        out = _resolve_token_body(low, rows, scen, stats)
        return out[:1].upper() + out[1:]
    return _resolve_token_body(body, rows, scen, stats)


def _resolve_token_body(tok, rows, scen, stats):
    """Resolve one {{token}}. Raises NoteTokenError on anything unrecognised."""
    body = tok.strip()
    if ":" in body:                      # cross-round reference
        alias, body = body.split(":", 1)
        alias = alias.strip()
        if alias not in SCEN_ALIAS:
            raise NoteTokenError(f"unknown scenario alias {alias!r}")
        scen = SCEN_ALIAS[alias]
    # Split ONCE. Arguments legitimately contain dots -- a score threshold is
    # "0.67" and a chain substring can be "telemetry_settings.json" -- so a
    # naive split(".") shreds them into pieces that no function accepts.
    fn, _, rest = body.partition(".")
    fn, rest = fn.strip(), rest.strip()

    # A token that takes no argument must not be *given* one. Without this the
    # zero-arg handlers just ignore `rest`, so {{n_models.flash}} -- a plausible
    # thing to write while converting a per-model claim into a corpus-wide one,
    # or to leave behind half-edited -- renders "13" and reads as if it had been
    # checked against flash. Every other guard in the chain is happy: the token
    # resolved, so nothing is left unresolved, and the output is a bare number,
    # so nothing looks like a template. Refuse centrally, here, because this is
    # the one place fn and rest are split.
    if fn in _ZERO_ARG and rest:
        raise NoteTokenError(
            f"{fn} takes no argument, got {rest!r} in {tok!r}")

    if fn == "n_models":
        return str(len(MODELS))
    if fn == "n_models_word":
        return _numword(len(MODELS))
    if fn == "step_cap":
        return str(STEP_CAP)

    corpus = _corpus_token(fn, rest, tok, stats)
    if corpus is not None:
        return corpus

    if scen is None or scen not in rows:
        raise NoteTokenError(
            f"token {tok!r} needs a curated round; this note is not attached to "
            f"one (use the alias form, e.g. {{{{payment:npass}}}})")
    row = rows[scen]

    def need(want_arg):
        if want_arg and not rest:
            raise NoteTokenError(f"{fn} needs an argument: {tok!r}")
        if not want_arg and rest:
            raise NoteTokenError(f"{fn} takes no argument: {tok!r}")

    def model(key):
        if key not in row:
            raise NoteTokenError(f"no model {key!r} in round for {scen}")
        return row[key]

    def score_arg():
        try:
            return float(rest)
        except ValueError:
            raise NoteTokenError(f"{fn} needs a numeric score: {tok!r}")

    if fn in ("score", "steps", "opens", "writes"):
        need(True)
        t = model(rest)
        if fn == "score":
            return f"{t['score']:.2f}"
        return str(t["steps"] if fn == "steps" else
                   t["n_open"] if fn == "opens" else t["n_write"])

    if fn in ("npass", "npass_word", "passers"):
        need(False)
        ks = _order(row, [k for k in row if row[k]["success"]])
        if fn == "npass":
            return str(len(ks))
        if fn == "npass_word":
            return _numword(len(ks))
        return _join([PROSE_NAME[k] for k in ks])

    if fn in ("nat", "nat_word", "at"):
        need(True)
        want = score_arg()
        ks = _order(row, [k for k in row
                          if abs(row[k]["score"] - want) < 1e-9])
        if fn == "nat":
            return str(len(ks))
        if fn == "nat_word":
            return _numword(len(ks))
        return _join([PROSE_NAME[k] for k in ks])

    if fn in ("ncap", "ncap_word", "capped"):
        need(False)
        ks = _order(row, [k for k in row if row[k]["steps"] >= STEP_CAP])
        if fn == "ncap":
            return str(len(ks))
        if fn == "ncap_word":
            return _numword(len(ks))
        return _join([PROSE_NAME[k] for k in ks])

    # {{nwith.SUBSTR}} / {{with.SUBSTR}} / {{nwithout.SUBSTR}} -- models whose
    # command chain contains an action matching SUBSTR. Lets a note say "every
    # model sends to Casey" without the writer counting thirteen chains by eye.
    if fn in ("nwith", "nwith_word", "with",
              "nwithout", "nwithout_word", "without"):
        need(True)
        hit = [k for k in row
               if any(rest in (st.get("act") or "") for st in row[k]["chain"])]
        if fn in ("nwith", "nwith_word", "with"):
            ks = _order(row, hit)
        else:
            ks = _order(row, [k for k in row if k not in hit])
        if fn.endswith("_word"):
            return _numword(len(ks))
        if fn.startswith("n"):
            return str(len(ks))
        return _join([PROSE_NAME[k] for k in ks])

    # {{nstopped.REASON}} / {{nstopped_word.REASON}} / {{stopped.REASON}} -- the
    # models that ended THIS round with that outcome. {{nwith}} cannot answer
    # it: it matches the action chain, and "never changed the world" is defined
    # by the absence of an action, not the presence of one.
    #
    # by_task spells the cap "reached max_steps=20" while the corpus-wide
    # counter normalises it to "reached max_steps", so the two vocabularies
    # differ by exactly the value of the cap. Match on the prefix; an equality
    # test here returns a calm zero rather than an error.
    if fn in ("nstopped", "nstopped_word", "stopped"):
        need(True)
        seen = {str(row[k]["stop"]) for k in row}
        hit = [k for k in row
               if str(row[k]["stop"]) == rest
               or str(row[k]["stop"]).startswith(rest + "=")]
        if not hit and not any(s == rest or s.startswith(rest + "=")
                               for s in seen):
            raise NoteTokenError(
                f"no model ends {scen!r} with stop {rest!r} in {tok!r}; "
                f"observed outcomes are {sorted(seen)}")
        ks = _order(row, hit)
        if fn.endswith("_word"):
            return _numword(len(ks))
        if fn.startswith("n"):
            return str(len(ks))
        return _join([PROSE_NAME[k] for k in ks])

    # {{ncredit.VERB}} / {{ncredit_word.VERB}} / {{credited.VERB}} -- the models
    # that were actually awarded score on a step of that verb. "Three models are
    # credited for the message" is a claim about the rubric's crediting, not
    # about who sent one, and those two sets differ sharply in the payment
    # round: ALL FOURTEEN models message the right agent, three get a unit for it
    # (gpt-5.6-sol, gpt-5.6-luna, deepseek-v4-pro; re-counted 2026-08-14). The
    # shipped sentence says "every model", so it did not go stale when the field
    # grew -- this comment did.
    if fn in ("ncredit", "ncredit_word", "credited"):
        need(True)
        ks = _order(row, [k for k in row
                          if any((st.get("sd") or 0) > 0
                                 and st.get("key") == rest
                                 for st in row[k]["chain"])])
        if fn == "ncredit":
            return str(len(ks))
        if fn == "ncredit_word":
            return _numword(len(ks))
        return _join([PROSE_NAME[k] for k in ks])

    # {{sumat.FIELD.SCORE}} -- FIELD totalled over every model on that score.
    # Needed for collective claims: "the four Gemini models wrote nothing
    # between them" must be tested against all four, not against one of them
    # with the other three taken on faith.
    if fn == "sumat":
        field, _, want = rest.partition(".")
        if field not in ("steps", "opens", "writes"):
            raise NoteTokenError(
                f"sumat takes steps|opens|writes, got {field!r} in {tok!r}")
        try:
            want = float(want)
        except ValueError:
            raise NoteTokenError(f"sumat needs a numeric score: {tok!r}")
        col = {"steps": "steps", "opens": "n_open", "writes": "n_write"}[field]
        ks = [k for k in row if abs(row[k]["score"] - want) < 1e-9]
        if not ks:
            raise NoteTokenError(f"no model scored {want} in {scen}")
        return str(sum(row[k][col] for k in ks))

    # {{credit.KEY}} -- 0-based index of the first step that earned score.
    if fn == "credit":
        need(True)
        t = model(rest)
        for i, st in enumerate(t["chain"]):
            if (st.get("sd") or 0) > 0:
                return str(i)
        raise NoteTokenError(f"{rest} earned no score in {scen}")

    # {{alleq.FIELD}} -- the value of FIELD when every model in the round shares
    # it, and a hard error otherwise. "Every model opens exactly two assets" is
    # a thirteen-model claim; spelling it {{opens.opus5}} tests one model and
    # takes the other twelve on faith, which is exactly how the previous pass of
    # this prose ended up asserting things that had stopped being true.
    if fn == "alleq":
        col = _field_arg(fn, tok, rest)
        vals = {row[k][col] for k in row}
        if len(vals) != 1:
            raise NoteTokenError(
                f"{rest} is not uniform across models in {scen}: "
                f"{sorted(vals)}")
        return str(vals.pop())

    # {{eq.FIELD.k1+k2}} -- the value of FIELD shared by exactly these models,
    # and a hard error if they do not in fact share it.
    #
    # {{alleq}} covers "the whole field agrees"; this covers the far more common
    # sentence "A and B both close at N steps", which the notes were writing as
    # `{{steps.A}}` and leaving B unchecked. That is not a typo-class risk, it
    # is a claim: the sentence asserts an equality and then measures one side of
    # it. Two of them were live in this file, and both happened to still be true
    # -- which is the only reason nobody noticed that nothing was testing them.
    if fn == "eq":
        field, _, who = rest.partition(".")
        field = field.strip()
        # `score` is not in _FIELD_COL (that table exists for the count-like
        # fields) but is the field these sentences most often assert equality
        # on -- "the two reach the same number", "both finish on the X plateau".
        if field == "score":
            col, show = "score", lambda v: f"{v:.2f}"
            # ^ same two decimals the {{score.KEY}} branch renders, so a tie
            #   value and the individual scores beside it cannot print
            #   differently.
        else:
            col, show = _field_arg(fn, tok, field), str
        picks = _eq_members(who, tok, row, scen)
        vals = {w: row[w][col] for w in picks}
        if len({show(v) for v in vals.values()}) != 1:
            raise NoteTokenError(
                f"eq asserts {field} is equal across "
                f"{_join([PROSE_NAME[w] for w in picks])} in {scen}, "
                "but they differ: "
                + ", ".join(f"{PROSE_NAME[w]}={show(v)}" for w, v in vals.items())
                + f" ({tok!r}). Rewrite the sentence; do not re-point the token.")
        return show(next(iter(vals.values())))

    # {{modal.FIELD}} / {{nmodal.FIELD}} / {{nmodal_word.FIELD}} -- the most
    # common value of FIELD and how many models share it. Lets "most of the
    # field opens six assets" carry its own evidence.
    if fn in ("modal", "nmodal", "nmodal_word"):
        col = _field_arg(fn, tok, rest)
        counts = Counter(row[k][col] for k in row)
        # Ties break toward the larger value so the answer never depends on
        # dict ordering.
        best = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
        if fn == "modal":
            return str(best[0])
        return str(best[1]) if fn == "nmodal" else _numword(best[1])

    # {{maxof.FIELD}} / {{argmaxof.FIELD}} / {{minof.FIELD}} / {{argminof.FIELD}}
    # -- extremes within this round, with the arg* forms naming every model tied
    # at the extreme rather than picking one arbitrarily.
    if fn in ("maxof", "argmaxof", "minof", "argminof"):
        col = _field_arg(fn, tok, rest)
        pick = max if fn in ("maxof", "argmaxof") else min
        want = pick(row[k][col] for k in row)
        if fn in ("maxof", "minof"):
            return str(want)
        return _join([PROSE_NAME[k]
                      for k in _order(row, [k for k in row
                                            if row[k][col] == want])])

    # {{alleqrounds.FIELD}} / {{nalleqrounds.FIELD}} -- which of the eight
    # rounds have every model agreeing on FIELD. This exists because the first
    # draft called the satellite round's unanimous open count "the tightest
    # agreement anywhere in these eight rounds", and it is not: warehouse is
    # unanimous at the same value and lock is unanimous at another. A
    # cross-round superlative has to be computed across the rounds.
    if fn in ("alleqrounds", "nalleqrounds"):
        col = _field_arg(fn, tok, rest)
        hit = [a for a in SCEN_ALIAS
               if SCEN_ALIAS[a] in rows
               and len({rows[SCEN_ALIAS[a]][k][col]
                        for k in rows[SCEN_ALIAS[a]]}) == 1]
        hit.sort(key=lambda a: CURATED_SCENARIOS.index(SCEN_ALIAS[a]))
        if fn == "nalleqrounds":
            return _numword(len(hit))
        return _join(hit)

    # {{maxall.FIELD}} / {{argmaxall.FIELD}} -- the same extreme taken across
    # all eight curated rounds at once, so a superlative like "more than any
    # model in any of these rounds" is checked against all eight instead of
    # being asserted from memory.
    if fn in ("maxall", "argmaxall"):
        col = _field_arg(fn, tok, rest)
        want = max(r[k][col] for r in rows.values() for k in r)
        if fn == "maxall":
            return str(want)
        hit = []
        for sc in CURATED_SCENARIOS:
            r = rows.get(sc) or {}
            for k in _order(r, [k for k in r if r[k][col] == want]):
                if PROSE_NAME[k] not in hit:
                    hit.append(PROSE_NAME[k])
        return _join(hit)

    # {{creditof.KEY.VERB}} -- index of the first SCORING step whose action verb
    # is exactly VERB. {{credit.KEY}} alone goes ambiguous the moment a model
    # earns units from two different kinds of action: in the payment round
    # luna's first credit is a report and its message credit lands six steps
    # later, while v4-pro's arrive in the opposite order. Naming the verb pins
    # which one the sentence is actually talking about.
    if fn == "creditof":
        key, _, verb = rest.partition(".")
        if not key or not verb:
            raise NoteTokenError(f"creditof takes KEY.VERB: {tok!r}")
        t = model(key)
        for i, st in enumerate(t["chain"]):
            if (st.get("sd") or 0) > 0 and st.get("key") == verb:
                return str(i)
        raise NoteTokenError(
            f"{key} never scored on a {verb!r} step in {scen}")

    # {{stepof.KEY.SUBSTR}} -- 0-based index of that model's first matching act.
    if fn == "stepof":
        key, _, sub = rest.partition(".")
        if not key or not sub:
            raise NoteTokenError(f"stepof takes KEY.SUBSTR: {tok!r}")
        t = model(key)
        for i, st in enumerate(t["chain"]):
            if sub in (st.get("act") or ""):
                return str(i)
        raise NoteTokenError(f"{key} never ran {sub!r} in {scen}")

    raise NoteTokenError(f"unknown token function {fn!r} in {tok!r}")


def resolve_notes(notes, stats):
    """Expand every {{token}} in the prose sidecar against the loaded corpus.

    Returns a deep copy with tokens replaced. Raises SystemExit listing every
    bad token at once -- a build that dies on the first one makes fixing a
    rewritten note a slow game of whack-a-mole."""
    if not notes:
        return notes
    rows = _round_rows(stats)
    errors = []

    def walk(obj, scen):
        if isinstance(obj, dict):
            return {k: walk(v, SCEN_ALIAS.get(k, k) if k in rows or k in SCEN_ALIAS
                            else scen)
                    for k, v in obj.items()}
        if isinstance(obj, list):
            return [walk(v, scen) for v in obj]
        if isinstance(obj, str):
            def sub(m):
                try:
                    return _resolve_token(m.group(1), rows, scen, stats)
                except NoteTokenError as e:
                    errors.append(str(e))
                    return m.group(0)
            return _TOKEN_RE.sub(sub, obj)
        return obj

    out = walk(notes, None)
    if errors:
        raise SystemExit("behavior_notes.json has unresolvable tokens:\n  "
                         + "\n  ".join(sorted(set(errors))))
    return out


def check_notes_resolved(notes):
    """Belt and braces: no note may reach the renderer still holding a token.

    resolve_notes() already raises on anything it cannot resolve, so this only
    fires if a token slipped through by a route that bypassed the resolver --
    which is exactly the sort of thing that happens when someone adds a new
    notes consumer later."""
    bad = []

    def walk(obj, path):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]")
        elif isinstance(obj, str):
            m = _TOKEN_RE.search(obj)
            if m:
                bad.append(f"{path}: {m.group(0)}")

    walk(notes or {}, "")
    if bad:
        raise SystemExit("unresolved note tokens reached the renderer:\n  "
                         + "\n  ".join(bad))


_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.S)


def check_notes_derived(notes):
    """No passage may bold a figure it did not derive from the corpus.

    Run on the TEMPLATE, before resolution. The prose style bolds every number
    that came from the data, so a passage carrying a bolded digit and not one
    {{token}} is a hand-typed figure: a claim nothing recomputes, and nothing
    invalidates when the corpus moves under it.

    The command-surface kicker was exactly that. It survived the thirteen-model
    rewrite unedited, still quoting the nine-model cut -- `asset list` at
    85.8-100%, 3,191 file reads, a 31.5x search span -- every one of which the
    new corpus contradicts. Worse, the markdown mirror certified it as in sync,
    because the mirror compares the two copies to each other and both were
    stale in the same way. Agreement between two stale copies is not a check;
    this is the check.

    The rule is deliberately narrow -- zero tokens AND a bolded digit -- so it
    cannot fire on bolded model names like **3.5-flash**, which appear only in
    passages that derive something else. On the current template it selects the
    kicker and nothing else."""
    bad = []

    def walk(obj, path):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]")
        elif isinstance(obj, str):
            if _TOKEN_RE.search(obj):
                return
            for m in _BOLD_RE.finditer(obj):
                if any(c.isdigit() for c in m.group(1)):
                    bad.append(f"{path}: **{m.group(1)[:48]}**")
                    return

    walk(notes or {}, "")
    if bad:
        raise SystemExit(
            "these notes bold a figure but derive nothing; replace the "
            "literal with a {{token}}:\n  " + "\n  ".join(bad))


def scenario_of(tid):
    s = tid[len("conflict_vlm_"):] if tid.startswith("conflict_vlm_") else tid
    return re.sub(r"_r\d+$", "", s)


def bar(segments, total):
    """segments: list of (label, count, color). Returns stacked-bar HTML."""
    out = ['<div class="catbar">']
    for label, cnt, color in segments:
        if not cnt:
            continue
        pct = 100 * cnt / max(total, 1)
        out.append(
            f'<span class="seg" style="width:{pct:.2f}%;background:{color}" '
            f'title="{esc(label)}: {cnt} ({pct:.1f}%)"></span>'
        )
    out.append("</div>")
    return "".join(out)


# -- open-source / cost caveats, rendered ---------------------------------------
# OPEN_SOURCE_CAVEAT and COST_CAVEAT used to be defined and then never emitted: the
# only reference to their VALUES was an arithmetic one (n_mm_both), so the report
# promised "see the open-source caveat on each table that includes them" and shipped
# no such caveat anywhere. This block keeps that promise: os_note() is the one-line
# footnote, render_caveat_box() is the single full statement it links to, and
# annotate_os_tables() attaches the footnote to every table that needs one.
#
# WHY A DOCUMENT PASS RATHER THAN A CALL AT EACH TABLE. Hand-attaching the note at
# each render_* site was the obvious approach and is the wrong one, for two reasons
# found while doing it:
#   * it goes stale silently. A table that gains an open-source row later -- or a new
#     table added by someone who never read this comment -- ships uncaveated, and
#     nothing catches it. The promise is exactly the kind that rots.
#   * it cannot see conditional content. render_clustering_exemplar() picks TWO
#     models by a data-driven rule and returns "" when no exemplar qualifies, so
#     whether it needs the note is not knowable at the call site. Testing the emitted
#     markup answers it for free, and correctly, every build.
# So the test is on what was actually rendered: any <table> whose body names an
# open-source model gets the note, and check_os_notes() re-derives the same set from
# the finished document and fails the build on any mismatch in either direction.
CAVEAT_ANCHOR = "oscaveat"

# Stand-in for the count of behavioral sections, substituted once the document
# exists and its <h2>s can be counted. A sentinel rather than a number because
# the sentence sits in the header, ABOVE every section it is counting.
_NSEC = "\u0000NSEC\u0000"

# Tables are matched non-greedily and never nest in this document; the pass asserts
# the open/close counts agree before it touches anything.
_TABLE_RE = re.compile(r"<table\b.*?</table>", re.S)


def _model_aliases(k):
    """Every string this report may use to name model `k` in a rendered row.

    The footnote machinery decides what a table is about by looking for model
    names in it, and this report calls the same model three different things:
    MODELS' full name (deepseek-v4-pro), PROSE_NAME (v4-pro) and MODEL_LABEL
    (v4-pro). Matching on only one of them silently under-reports. It did:
    the markdown tables label rows with PROSE_NAME, so a probe for the full
    name "deepseek-v4-pro" found nothing and the path-mode footnote named only
    glm-5.2 -- dropping half of the very condition it exists to disclose --
    while gemini-3.7-flash matched purely by accident, through the model-id
    column of the one table that has one.
    """
    full = {n: key for n, _, key in MODELS}
    out = {n for n, key in full.items() if key == k}
    for src in (PROSE_NAME, MODEL_LABEL):
        if k in src:
            out.add(src[k])
    return out


def _caveat_keys():
    """Every model carrying a documented, non-comparable run condition.

    Keyed off the caveat dictionaries rather than off the provider, so adding a
    model to either dictionary extends footnote coverage automatically. That
    matters: the footnote used to be triggered by "is this row open-source",
    which silently gave no coverage at all when a NON-open-source model
    (gemini-3.7-flash, promoted to its r02) acquired a condition of its own.
    """
    return set(OPEN_SOURCE_CAVEAT) | set(_run_caveats())


def _os_model_names():
    """Every alias of every model that needs the run-conditions footnote."""
    return sorted(a for k in _caveat_keys() for a in _model_aliases(k))


def _caveat_fragments(body, code):
    """Sentence fragments for the conditions this table's rows actually carry.

    Built once and shared by the HTML and markdown footnotes so the two cannot
    say different things. `code` wraps an inline literal for the target format.

    Only conditions PRESENT in the table are named. A footnote that recites
    every caveat in the report under a table that carries one of them trains
    the reader to skip it.
    """
    nm = PROSE_NAME
    here = lambda k: any(a in body for a in _model_aliases(k))    # noqa: E731
    out = []
    path_k = [k for k in PROSE_KEYS
              if k in OPEN_SOURCE_CAVEAT
              and "multimodal_input_mode=path" in OPEN_SOURCE_CAVEAT[k]
              and here(k)]
    if path_k:
        # _join, not " and ".join: at two path-mode models the latter read
        # correctly, at three it produced "v4-pro and v4-flash and glm-5.2".
        who = _join([nm[k] for k in path_k])
        frag = (f"{who} {'were' if len(path_k) > 1 else 'was'} handed asset "
                f"file paths as text and never received an image "
                f"({code('multimodal_input_mode=path')})")
        # Name the control in the same breath, or the reader generalises the
        # condition to the whole open-source block. kimi-k3 did get images.
        # Deliberately NOT "the third open-source model" -- that was written
        # when the block held three and silently became false at four.
        if here("kimi"):
            frag += (f"; {nm['kimi']}, the open-source model that did receive "
                     f"images, is the control")
        out.append(frag)
    # Every promoted run in the table, not just flash37 by name: the previous
    # version tested the literal key, so a second override would have rendered
    # the caveat box entry (which iterates PROSE_KEYS) while this per-table
    # footnote silently omitted the row. The gap and p-value are read off the
    # same measurement the footnote text uses, so the short form here and the
    # long form in the caveat box cannot disagree.
    for k in PROSE_KEYS:
        if k not in _run_caveats() or not here(k):
            continue
        f = _run_facts()[k]
        out.append(f"{nm[k]} is reported on {code(f['run'])}, the higher of "
                   f"its two complete runs, where every other row is a single "
                   f"run (the two differ by {abs(f['delta_pp']):.1f}pp, "
                   f"McNemar p={f['p']:.2f})")
    return out


def os_note(body, cost=False, cls="tblnote"):
    """One-line footnote for a view carrying a non-comparable run condition.

    `cls` exists so a FIGURE can carry the same derived sentence as a table
    without being counted as one. check_os_notes() proves its table census by
    counting `<div class="tblnote">` against the number of qualifying tables,
    so emitting one under a chart made that count disagree with itself and
    failed the build -- correctly. Charts pass cls="fignote" and are checked
    separately; the wording stays shared so the two can never diverge.
    """
    nm = {k: n for n, _, k in MODELS}
    code = lambda s: f"<code>{esc(s)}</code>"                     # noqa: E731
    frags = _caveat_fragments(body, code)
    noun = "row" if cls == "tblnote" else "model"
    txt = f"<b>Not every {noun} here is like-for-like.</b> " + esc_join(frags)
    if cost:
        txt += (f". Input tokens dominate spend, so that is most of why the "
                f"path-mode rows are cheap here; separately, "
                f"{esc(nm['flash37'])} ran on introductory pricing")
    txt += f'. Full statement: <a href="#{CAVEAT_ANCHOR}">run conditions</a>.'
    return f'<div class="{cls}">{txt}</div>'


def esc_join(frags):
    """Join pre-built fragments into one sentence body."""
    return "; also, ".join(frags) if len(frags) > 1 else (frags[0] if frags else "")


def asset_gate_ranking(stats):
    """Rank every model on the REQUIRED asset gate, over the tasks all share.

    WHY DERIVED. The caveat box's load-bearing paragraph -- the one that stops
    a reader concluding the path-mode rows are floored by the harness rather
    than measured -- used to recite a hand-maintained table: "1st and 5th of
    thirteen (glm-5.2 95.3%, deepseek-v4-pro 91.3%) ... the floor is
    claude-sonnet-5 at 77.8%, 17.5 points below". Every one of those figures is
    a function of who is in the field, and the field just grew by a model. A
    hand-written ranking of a changing population is a stale claim waiting to
    ship, so it is computed here from the same by_task the tables render from.

    Two denominators, deliberately kept apart because they answer different
    questions and mixing them is the easy error:
      * SHARED -- tasks where EVERY model has the gate scored. The only honest
        basis for a rank, and what the ordering below uses.
      * OWN -- each model's own scored count. Larger for thirteen of the models
        than for gemini-3.1-pro, whose check_results came back empty on three
        records. Reported alongside, never ranked on.
    """
    per = {}
    for k in PROSE_KEYS:
        bt = (stats.get(k) or {}).get("by_task") or {}
        per[k] = {t: v.get("asset_ok") for t, v in bt.items()
                  if v.get("asset_ok") is not None}
    if not per or not all(per.values()):
        return None
    shared = set.intersection(*(set(v) for v in per.values()))
    rows = []
    for k in PROSE_KEYS:
        own = per[k]
        n_sh = sum(1 for t in shared if own[t])
        rows.append({
            "key": k,
            "pct": 100.0 * n_sh / len(shared),
            "own_pass": sum(1 for v in own.values() if v),
            "own_n": len(own),
        })
    rows.sort(key=lambda r: (-r["pct"], PROSE_KEYS.index(r["key"])))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return {
        "shared": len(shared),
        "rows": rows,
        "by_key": {r["key"]: r for r in rows},
        "own_n_max": max(r["own_n"] for r in rows),
    }


def _ordinal(n):
    """1 -> 1st. Used for ranks inside a sentence, where "rank 1" reads wrong."""
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }".replace(" ", "")


def render_caveat_box(stats=None, rank=None):
    """The full, per-model statement of every non-comparable run condition."""
    nm = {k: n for n, _, k in MODELS}
    rows = []
    for k in PROSE_KEYS:
        bits = []
        if k in OPEN_SOURCE_CAVEAT:
            bits.append(OPEN_SOURCE_CAVEAT[k])
        if k in _run_caveats():
            bits.append("run selection: " + _run_caveats()[k])
        if k in COST_CAVEAT:
            bits.append("cost: " + COST_CAVEAT[k])
        if not bits:
            continue
        rows.append(f"<li><b>{esc(nm[k])}</b> \u2014 "
                    + esc("; ".join(bits)) + "</li>")
    nmod = _numword(len(MODELS))
    n_os = _numword(len(FIXED_TREE))
    # Task count from the corpus, and only if every model agrees on it -- the
    # sentence claims a COMMON grid, so it may not be rendered off one model.
    ns = {(stats.get(k) or {}).get("n") for k in PROSE_KEYS} if stats else set()
    ns.discard(None)
    ntask = next(iter(ns)) if len(ns) == 1 else None
    grid = (f"the same {ntask} tasks and the same nominal effort" if ntask
            else "the same task grid and the same nominal effort")
    hdrs = (f"read off {ntask}/{ntask} episode headers for that model"
            if ntask else "read off every episode header for that model")
    return (
        f'<div class="callout" id="{CAVEAT_ANCHOR}">'
        "<b>Run conditions that are not comparable across the field.</b> "
        f"All {nmod} models ran the same harness build, {grid}; "
        "what follows is the residue that did not align. Each "
        f"line was {hdrs}, "
        "and is stated here rather than left for the reader to discover."
        f"<ul>{''.join(rows)}</ul>"
        + _caveat_grounding_para(stats, rank) +
        f"Separately, the {n_os} open-source models were run <b>once</b>, at "
        f"{PRIMARY_EFFORT}, rather than swept over "
        + "/".join(EFFORTS) +
        ", so they appear in the main tables "
        "and the heatmap but not in the effort chapter.</div>")


def _caveat_grounding_para(stats, rank):
    """HTML wrapper around the shared grounding claim."""
    nm = {k: n for n, _, k in MODELS}
    txt = _grounding_claim(stats, rank, nm,
                           lambda s: f"<b>{s}</b>", lambda s: f"<i>{s}</i>")
    return (txt + " ") if txt else ""


def _grounding_claim(stats, rank, nm, b, i):
    """The "this condition did not depress their scores" paragraph, measured.

    Shared by the HTML caveat box and the markdown roster block so the two can
    never state different rankings of the same corpus -- they did: the markdown
    copy went on saying "the 3 path-mode models rank 1st and 5th of 14" (three
    models, two ranks, and the third one unnamed) for a build after the HTML
    copy had been derived. `b`/`i` wrap bold/italic for the target format and
    `nm` picks the naming convention, which is the ONLY thing that may differ.

    Every quantity here is read off the corpus, and every COMPARATIVE claim is
    re-checked before it is written: if the path-mode models ever do sink to
    the floor of the grounding ranking, this raises instead of printing a
    sentence that says they did not.
    """
    if not rank or not stats:
        return ""
    by = rank["by_key"]
    n_all = len(MODELS)
    # Escaping is the one thing that CANNOT be threaded through b()/i(): HTML
    # wants esc(), markdown wants the raw string, and applying either to the
    # other's output corrupts it. Rather than take a third callback for it,
    # assert the only property that makes the question moot -- these are model
    # names, and every one of them is a plain identifier that is inert in both
    # formats. If a name ever arrives with a character that is not, this stops
    # rather than silently emitting `<` into HTML or `_` into markdown.
    bad = sorted(n for n in nm.values() if not _SAFE_NAME.fullmatch(n))
    if bad:
        raise SystemExit(
            "grounding claim: model name(s) " + ", ".join(repr(x) for x in bad)
            + " contain characters that are not inert in both HTML and "
              "markdown. Add per-format escaping before rendering them here.")
    path_k = [k for k in PROSE_KEYS
              if k in OPEN_SOURCE_CAVEAT
              and "multimodal_input_mode=path" in OPEN_SOURCE_CAVEAT[k]]
    path_k.sort(key=lambda k: by[k]["rank"])
    ranks = _join([_ordinal(by[k]["rank"]) for k in path_k])
    detail = ", ".join(f"{nm[k]} {by[k]['pct']:.1f}%" for k in path_k)

    floor = rank["rows"][-1]
    top = rank["rows"][0]
    best_path = path_k[0]
    # The paragraph exists to say the harness condition is not what floors
    # these rows. If it ever IS, the honest move is to stop the build.
    if floor["key"] in path_k:
        raise SystemExit(
            "caveat box: a path-mode model is now the FLOOR of the grounding "
            f"ranking ({nm[floor['key']]} at {floor['pct']:.1f}%). The "
            "paragraph that says otherwise must be rewritten by hand, not "
            "re-rendered.")
    gap = by[best_path]["pct"] - floor["pct"]
    # The shape of this sentence depends on WHO holds each end, so it is chosen
    # rather than assumed. The earlier version hardcoded "both ends are held by
    # image-receiving models", which was true at two path-mode models and became
    # false -- and ungrammatical when patched -- the moment one of them took the
    # top of the ranking.
    if top["key"] in path_k:
        led = (f"the ranking is {b('led')} by one of them, and its floor \u2014 "
               f"{nm[floor['key']]} at {floor['pct']:.1f}%, {gap:.1f} points "
               f"below \u2014 is a model that did receive images")
    else:
        led = (f"both ends of that ranking are held by image-receiving models: "
               f"the floor is {nm[floor['key']]} at {floor['pct']:.1f}%, "
               f"{gap:.1f} points below {nm[best_path]}")

    # LEG TWO, and the reason it does not read "nor are they floored overall"
    # any more. That sentence was written when the path-mode group held ranks
    # 8 and 11 of thirteen, where it was fair. A third path-mode model then
    # entered at 13th of fourteen, and the sentence went on rendering -- "nor
    # does the group sit at the bottom ... it places 9th, 11th and 13th of
    # fourteen", a denial refuted by its own evidence two clauses later. The
    # group's median pass rate is BELOW the field's; pretending otherwise is
    # not a caveat, it is a sales pitch.
    #
    # What the corpus does still support is narrower and, for the purpose of
    # this paragraph, stronger: input mode does not SORT the field. The
    # path-mode band lies inside the image-receiving band, and the two models
    # holding the ends of that band ran under identical conditions. Both
    # halves are measured, and the containment is CHECKED rather than assumed
    # -- if the band ever escapes, this stops the build instead of printing a
    # sentence asserting that it did not.
    head = (
        "These are conditions, not excuses, and this report does not assert that "
        "any of them caused the scores they sit beside. The one that looks most "
        "damning is measurably not. The corpus\u2019 required grounding check scores "
        f"whether the needed asset was {i('opened')} \u2014 which a file path "
        f"supports \u2014 and on the {rank['shared']} tasks where all "
        f"{_numword(n_all)} models have it scored, the "
        f"{_numword(len(path_k))} path-mode models rank "
        f"{b(f'{ranks} of {_numword(n_all)}')} ({detail}). So {led}. ")

    pv = {k: stats[k]["pass_pct"] for k in PROSE_KEYS}
    both_k = [k for k in PROSE_KEYS if k not in path_k]
    # Nothing to contrast against if every model ran the same way; the leg is
    # dropped rather than degraded into a comparison with an empty set.
    if not both_k or len(path_k) < 2:
        return head
    lo_p, hi_p = (min(path_k, key=lambda k: pv[k]),
                  max(path_k, key=lambda k: pv[k]))
    lo_b, hi_b = (min(both_k, key=lambda k: pv[k]),
                  max(both_k, key=lambda k: pv[k]))
    if not (pv[lo_b] <= pv[lo_p] and pv[hi_p] <= pv[hi_b]):
        raise SystemExit(
            "caveat box: the path-mode pass band "
            f"({pv[lo_p]:.1f}-{pv[hi_p]:.1f}%) no longer sits inside the "
            f"image-receiving band ({pv[lo_b]:.1f}-{pv[hi_b]:.1f}%), so the "
            "sentence saying the two modes do not separate into two score "
            "populations is no longer supported. Rewrite it by hand rather "
            "than re-rendering.")
    span = pv[hi_p] - pv[lo_p]
    # Both ends being one vendor's two models is the cleanest form of the
    # argument -- same lab, same condition, same effort, and still far apart --
    # but it is a fact about today's field, not a guarantee, so the clause
    # appears only when it is true.
    vend = {k: v.split("/")[0] for _, v, k in MODELS}
    lab = (f", {vend[lo_p]}\u2019s own two entries"
           if vend[lo_p] == vend[hi_p] else "")
    return head + (
        "The two input modes also do not separate into two score populations: "
        f"the path-mode band \u2014 {b(f'{pv[lo_p]:.1f}% to {pv[hi_p]:.1f}%')} "
        f"pass \u2014 lies inside the "
        f"{b(f'{pv[lo_b]:.1f}%\u2013{pv[hi_b]:.1f}%')} the image-receiving "
        f"models span, and the two rows at its ends ({nm[lo_p]} and "
        f"{nm[hi_p]}{lab}) ran under identical conditions yet finish "
        f"{b(f'{span:.1f} points')} apart. So the path rows are best read as "
        "measurements of those models; this corpus does not isolate an effect "
        "of the mode itself. ")


def _table_needs_note(body):
    """True if this rendered <table> names an open-source model in its own rows."""
    return any(n in body for n in _os_model_names())


def annotate_os_tables(doc):
    """Insert the run-conditions footnote after every table carrying an OS row."""
    tables = _TABLE_RE.findall(doc)
    n_open = len(re.findall(r"<table\b", doc))
    if len(tables) != n_open or "</table>" in "".join(
            t[len("<table"):-len("</table>")] for t in tables):
        raise SystemExit(f"annotate_os_tables: {n_open} <table> opens but "
                         f"{len(tables)} matched -- nested or unbalanced markup")

    def sub(m):
        body = m.group(0)
        if not _table_needs_note(body):
            return body
        return body + os_note(body, cost="costtbl" in body[:200])

    return _TABLE_RE.sub(sub, doc)


def check_os_notes(doc):
    """Re-derive which tables need the footnote and prove the document matches.

    Deliberately independent of annotate_os_tables(): it re-reads the finished
    document, so it catches a table emitted after the pass ran, a note attached to
    a table that has no open-source row, and a note whose anchor points nowhere.
    """
    problems = []
    if f'id="{CAVEAT_ANCHOR}"' not in doc:
        problems.append(f"caveat box (id={CAVEAT_ANCHOR}) is never rendered, but "
                        "every footnote links to it")
    for i, m in enumerate(_TABLE_RE.finditer(doc), 1):
        body = m.group(0)
        tail = doc[m.end():m.end() + 400]
        has = tail.lstrip().startswith('<div class="tblnote">')
        want = _table_needs_note(body)
        if want and not has:
            problems.append(f"table #{i} has an open-source row and no run-conditions "
                            "footnote")
        if has and not want:
            problems.append(f"table #{i} carries a run-conditions footnote but has no "
                            "open-source row")
    n_notes = doc.count('<div class="tblnote">')
    n_want = sum(1 for m in _TABLE_RE.finditer(doc) if _table_needs_note(m.group(0)))
    if n_notes != n_want:
        problems.append(f"{n_notes} footnotes rendered for {n_want} qualifying tables")
    # Figure footnotes are counted apart from table ones so neither census can
    # absorb the other's error. The standings chart plots every model on one
    # axis, so it is the one figure that MUST carry the conditions with it.
    for fig in ('id="standings"',):
        if fig in doc and '<div class="fignote">' not in doc:
            problems.append(f"{fig} is rendered but no figure-level "
                            "run-conditions footnote accompanies it")
    for m in re.finditer(r'<div class="fignote">(.*?)</div>', doc, re.S):
        if f"#{CAVEAT_ANCHOR}" not in m.group(1):
            problems.append("a figure footnote does not link to the caveat box")
    if problems:
        raise SystemExit("run-conditions footnotes are wrong:\n  - "
                         + "\n  - ".join(problems))
    return n_want


def render_legend():
    items = "".join(
        f'<span class="lg"><span class="lgd" style="background:{col}"></span>{esc(lab)}</span>'
        for _, lab, col in CATS
    )
    items += '<span class="lg"><span class="ring"></span>tool returned an error</span>'
    return f'<div class="legend">{items}</div>'


def render_mix(stats, meta):
    rows = []
    for name, _, key in MODELS:
        st = stats[key]
        segs = [(CAT_LABEL[c], st["catmix"].get(c, 0), CAT_COLOR[c]) for c in CAT_ORDER]
        tot = st["catmix_total"]
        oth = st["catmix"].get("other", 0)
        opct = 100 * oth / max(tot, 1)
        rows.append(
            f'<div class="aggrow"><span class="nm">{esc(name)}</span>'
            f'{bar(segs, tot)}'
            f'<span class="cnt">{tot} actions \u00b7 other {oth} ({opct:.1f}%)</span></div>'
        )
    return (f"<h2>Action-category mix (all {meta['n_tasks']} tasks per model)</h2>"
            '<p class="h2sub">Every action is assigned to exactly one of the 12 '
            "categories below (the last, <code>other</code>, is the catch-all), so "
            "each bar sums to 100% of that model\u2019s actions. Category definitions "
            'are in the <a href="#glossary">reading guide</a>; the <code>other</code> '
            "bucket is itemised right below so nothing is hidden in it.</p>"
            + "".join(rows)
            + render_other_breakdown(stats))


def render_other_breakdown(stats):
    """Itemise the 'other' catch-all by domain family, so the bucket is never a
    silent black box. Reconciles with the 'other' segment in the bars above."""
    fams = set()
    for _, _, key in MODELS:
        fams.update(stats[key].get("othermix", {}).keys())

    def total(f):
        return sum(stats[key].get("othermix", {}).get(f, 0) for _, _, key in MODELS)

    order = sorted(fams, key=lambda f: (-total(f), f))
    head = ("<tr><th>domain family (inside <code>other</code>)</th>"
            + "".join(f"<th>{esc(MODEL_LABEL[k])}</th>" for _, _, k in MODELS) + "</tr>")
    body = []
    for f in order:
        cells = "".join(
            f'<td class="sc">{stats[key].get("othermix", {}).get(f, 0)}</td>'
            for _, _, key in MODELS)
        body.append(f'<tr><td class="ck">{esc(f)}</td>{cells}</tr>')
    tot_cells = "".join(
        f'<td class="sc"><b>{stats[key]["catmix"].get("other", 0)}</b></td>'
        for _, _, key in MODELS)
    body.append(f'<tr class="famrow"><td>other total</td>{tot_cells}</tr>')
    return ('<h3 class="oh">What\u2019s inside <code>other</code> '
            "(domain operations, by model)</h3>"
            f'<table class="cmdtable otherbrk"><thead>{head}</thead>'
            f'<tbody>{"".join(body)}</tbody></table>')


def heat_cell(val, rowmax, is_max):
    a = 0.0 if rowmax <= 0 else 0.10 + 0.60 * (val / rowmax)
    inner = f"<b>{val}</b>" if is_max else str(val)
    return (f'<td class="hc" style="background:rgba(96,165,250,{a:.2f})">{inner}</td>')


def render_command_surface(stats, notes, meta):
    ns = (notes or {}).get("command_surface", {})
    out = ["<h2>Command-surface deep-dive (which sub-command each model reaches for)</h2>"]
    if ns.get("intro"):
        out.append(f'<p class="h2sub">{md_inline(ns["intro"])}</p>')
    out.append('<table class="cmdtable"><thead><tr><th>sub-command</th>'
               + "".join(f"<th>{esc(MODEL_LABEL[k])}</th>" for _, _, k in MODELS)
               + "</tr></thead><tbody>")
    def emit_rows(keys):
        for k in keys:
            vals = [stats[key]["cmdmix"].get(k, 0) for _, _, key in MODELS]
            rowmax = max(vals) if vals else 0
            cells = "".join(heat_cell(v, rowmax, v == rowmax and v > 0) for v in vals)
            out.append(f'<tr><td class="ck"><code>{esc(k)}</code></td>{cells}</tr>')

    for fam, keys in CMD_FAMILIES:
        out.append(f'<tr class="famrow"><td colspan="{len(MODELS) + 1}">{esc(fam)}</td></tr>')
        emit_rows(keys)
    # Domain-operation verbs that fall into the "other" category are usually
    # omitted from a "diagnostic verbs" heatmap; surface them here so the deep-dive
    # is exhaustive rather than core-only. Data-driven so new verbs appear on rerun.
    shown = {k for _, ks in CMD_FAMILIES for k in ks}
    other_keys = set()
    for _, _, key in MODELS:
        for k in stats[key]["cmdmix"]:
            if k not in shown and k != "(empty)" and categorize(k)[0] == "other":
                other_keys.add(k)
    if other_keys:
        def kt(k):
            return sum(stats[key]["cmdmix"].get(k, 0) for _, _, key in MODELS)
        out.append(f'<tr class="famrow"><td colspan="{len(MODELS) + 1}">'
                   "domain operations (folded into <code>other</code>)</td></tr>")
        emit_rows(sorted(other_keys, key=lambda k: (-kt(k), k)))
    out.append("</tbody></table>")
    out.append(f'<p class="tblnote">Counts are total invocations across all {meta["n_tasks"]} '
               "tasks; each row is shaded relative to its own max, and the row-leading model is "
               "bold. Core diagnostic verbs are grouped by family; the domain-operation verbs "
               "that make up the <code>other</code> category follow. The complete per-command "
               'inventory is in the <a href="#cmd-appendix">command appendix</a>.</p>')
    for b in ns.get("bullets", []):
        out.append(f'<div class="csbul">{md_inline(b)}</div>')
    if ns.get("kicker"):
        out.append(f'<div class="callout">{md_inline(ns["kicker"])}</div>')
    return "".join(out)


def render_md_blocks(stats, meta, gold_sum, grid, unit_sum):
    """Generated markdown for the parts of the mirror that are not prose.

    The mirror carried a hand-typed NINE-model command table sitting directly
    above the thirteen-model bullets and disagreeing with them cell for cell:
    3,191 `file read`s for 3.5-flash in the table, 3,342 four lines below it,
    and a caption asserting a 592-632 `asset list` span "across all nine
    models" against the bullets' 601-662 across thirteen. Beneath it sat the
    same 31.5x search figure the bullets had already been corrected away from.

    A table is pure transcription. Typing one buys nothing and costs exactly
    this, so it is emitted here and placed by tools/sync_notes_md.py, on the
    same footing as the prose. The analytic caption is deliberately NOT
    reproduced: every claim it made is already in the bullets, tokenised, and
    two copies of a claim is the condition that produced the drift.

    Model order is MODELS order, which runs Gemini, Claude, OpenAI then the
    three open-source vendors, so a provider's models stay adjacent.
    """
    out = {}
    out.update(_mdb_provenance(stats, meta))
    out.update(_mdb_roster(stats, meta))
    out.update(_mdb_signatures(stats, meta))
    out.update(_mdb_command_surface(stats, meta))
    out.update(_mdb_catmix(stats, meta))
    out.update(_mdb_grounding(stats, meta))
    out.update(_mdb_stops(stats, meta))
    out.update(_mdb_steps(stats, meta))
    out.update(_mdb_concentration(stats, meta))
    out.update(_mdb_check_split(stats, meta))
    out.update(_mdb_fc_decomp(stats, meta))
    out.update(_mdb_unit_bands(stats, meta, unit_sum))
    out.update(_mdb_gold_length(stats, meta, gold_sum))
    out.update(_mdb_effort_grid(stats, meta, grid))
    out.update(_mdb_effort_mcnemar(stats, meta, grid))
    out = annotate_os_blocks(out)
    check_os_blocks(out)
    _assert_house_dashes(out)
    return out


# -- run-conditions footnote, markdown side -------------------------------------
# The HTML side solves this with a document pass (annotate_os_tables) precisely
# because attaching the note by hand at each render site "goes stale silently ...
# a new table added by someone who never read this comment ships uncaveated, and
# nothing catches it". That comment predicted this bug and then this file shipped
# it: the five generated MARKDOWN tables all carry open-source rows and went out
# with no footnote at all, while their HTML twins carried ten. The mirror is not
# a lesser artifact -- it is the copy that gets pasted into review threads.
#
# So the markdown side gets the same treatment, not a copy of the text: one pass
# over the emitted blocks, and an independent re-derivation that fails the build.
MD_CAVEAT_ANCHOR = "run-conditions-that-are-not-comparable"


def _os_note_md(body):
    """Markdown twin of os_note(): the one-line run-conditions footnote.

    Shares _caveat_fragments() with the HTML note, so the mirror cannot end up
    warning the reader about a different set of conditions than the report.
    """
    frags = _caveat_fragments(body, lambda s: f"`{s}`")
    return ("**Not every row here is like-for-like.** " + esc_join(frags)
            + f". Full statement: [run conditions](#{MD_CAVEAT_ANCHOR}).")


def _block_needs_note(body):
    """True if this block's rendered rows name a model with a run condition."""
    return any(n in body for n in _os_model_names())


def annotate_os_blocks(blocks):
    """Append the footnote to every generated block naming an open-source model.

    The run-conditions box itself is exempt: it IS the full statement, so a
    footnote pointing the reader at the paragraph they are already reading is
    noise, and it would make the anchor self-referential.
    """
    out = {}
    for name, body in blocks.items():
        if name != "roster.conditions" and _block_needs_note(body):
            body = body + "\n\n" + _os_note_md(body)
        out[name] = body
    return out


def check_os_blocks(blocks):
    """Re-derive which blocks need the footnote; fail on a mismatch either way.

    Deliberately independent of annotate_os_blocks(): it reads the finished
    text, so it also catches a block appended after the pass, and a footnote
    whose anchor no longer matches the heading the box renders.
    """
    problems = []
    box = blocks.get("roster.conditions", "")
    if f'name="{MD_CAVEAT_ANCHOR}"' not in box and f"#{MD_CAVEAT_ANCHOR}" not in box:
        problems.append(
            f"the run-conditions box does not define anchor "
            f"'{MD_CAVEAT_ANCHOR}', but every footnote links to it")
    # Presence is detected by the link to the box, which appears nowhere else;
    # the note's TEXT varies with the conditions the table carries, so equality
    # against a freshly built note would be checking the builder against itself.
    marker = f"](#{MD_CAVEAT_ANCHOR})"
    for name, body in sorted(blocks.items()):
        has = marker in body
        want = _block_needs_note(body) and name != "roster.conditions"
        if want and not has:
            problems.append(f"block '{name}' has an open-source row and no "
                            "run-conditions footnote")
        if has and not want:
            problems.append(f"block '{name}' carries a run-conditions footnote "
                            "but has no open-source row")
    if problems:
        raise SystemExit("markdown run-conditions footnotes are wrong:\n  - "
                         + "\n  - ".join(problems))


def _assert_house_dashes(blocks):
    """Refuse to emit a caption that punctuates differently from the prose.

    The mirror sets every parenthetical break as a spaced em dash; a generated
    caption written in ASCII `--` renders as a stray double hyphen in the
    middle of a document that has 100+ em dashes and nothing else, which is
    the kind of thing a reader notices and an author does not. The catmix
    caption shipped exactly that. It is cheaper to fail the build than to
    re-read every caption on every edit.
    """
    bad = sorted(name for name, body in blocks.items() if " -- " in body)
    if bad:
        raise ValueError(
            "ASCII '--' in generated block(s) %s \u2014 the report sets breaks "
            "as a spaced em dash; use '\\u2014'" % bad)


def _lone_extremes(texts):
    """Indices of the uniquely attained max and min, judged on RENDERED text.

    Two traps, both of which this table hit on its first render:

    Bolding a shared extreme is worse than not bolding at all. `file_delete` is
    zero for thirteen of the fourteen models, so the obvious `v == min(vals)`
    would set thirteen zeros in bold and present them as a finding.

    Subtler, and the reason the comparison is on the formatted string rather
    than the raw value: an extreme that is invisible at display precision is
    not an extreme the reader can see. luna issues the corpus's only two
    `file_delete` calls, which is a genuine unique maximum and renders as
    **0.0%** -- a bolded zero sitting in a column of unbolded zeros, which
    reads as a typesetting bug. glm-5.2's `agent_delegate` low renders as
    **0.1%** against two other models' unbolded 0.1%. Ranking the rendered
    text collapses both cases to a tie, and a tie goes unmarked.
    """
    out = set()
    if not texts:
        return out
    for target in (max(texts, key=_num_key), min(texts, key=_num_key)):
        hits = [i for i, t in enumerate(texts) if t == target]
        if len(hits) == 1:
            out.add(hits[0])
    return out


def _num_key(text):
    """Numeric sort key for an already-formatted cell (strips , % and x).

    U+2212 MINUS is folded to ASCII before the strip. The report sets negative
    numbers with a real minus sign, the character class below does not contain
    one, and so a cell reading `\u221210.5` stripped to `10.5` and ranked as the
    LARGEST value in its column -- which would have bolded the sharpest effort
    REGRESSION in the sweep as though it were the biggest gain. Latent rather
    than shipped: until the effort and decomposition tables were generated,
    every emitted block held non-negative values only.
    """
    return float(re.sub(r"[^0-9.\-]", "", text.replace("\u2212", "-")) or 0)


def _md_table(keys, cols, fmt=None, nobold=(), left=()):
    """Assemble a markdown table, one row per model key.

    `cols` is an ordered list of (header, {model_key: value}); `fmt` maps a
    header to a formatter for its values (default: thousands-separated int);
    `nobold` lists headers whose extremes should go unmarked; `left` lists
    headers to set flush left rather than the numeric default of flush right.
    Row labels are PROSE_NAME, so the table can never disagree with the prose
    about what a model is called.

    Columns are right-aligned by default because nearly every one of them is a
    number, and a column of decimals only lines up on the point when it is. A
    verdict or any other label is not a magnitude and reads as a ragged left
    edge when forced into the numeric column, hence `left`.
    """
    fmt = fmt or {}
    default = lambda v: f"{v:,}"                                  # noqa: E731
    text = {h: [fmt.get(h, default)(s[k]) for k in keys] for h, s in cols}
    marks = {h: (set() if h in nobold else _lone_extremes(text[h]))
             for h, _ in cols}
    rows = ["| Model | " + " | ".join(h for h, _ in cols) + " |",
            "|---|" + "".join("---|" if h in left else "---:|"
                              for h, _ in cols)]
    for i, k in enumerate(keys):
        cells = [f"**{text[h][i]}**" if i in marks[h] else text[h][i]
                 for h, _ in cols]
        rows.append(f"| {PROSE_NAME[k]} | " + " | ".join(cells) + " |")
    return rows


def _mdb_provenance(stats, meta):
    """Front matter: what the corpus is, in numbers the build just counted.

    The hand-typed predecessor described a corpus that no longer exists -- "9
    models x 3 effort levels x 600 tasks = 16,200 episodes", and "figures are at
    reasoning_effort=medium, the primary setting", with a regenerate command
    passing `--effort medium`. By then the tables below it were being emitted at
    thirteen models and `high`, so the document's own front matter contradicted
    every table in it.

    The shape is also no longer a clean cross-product and must not be written as
    one: ten models were swept over three efforts, four open-source models were
    run once at high. Multiplying anything by fourteen here gives a wrong answer.

    The open-source path template is DERIVED from FIXED_TREE, not typed. It used
    to read `open_source_high/<Vendor>/<model>/` for all of them, which was true
    while that was the only fixed tree and became false the moment v4-flash came
    in from `aligned_setting_high_ocr/`. A provenance paragraph is the one place
    in the report where a wrong path is not a cosmetic error -- it is the whole
    claim -- so it is now read off the same constant the loader reads.
    """
    n_sweep, n_fixed = len(SWEEP_MODELS), len(FIXED_TREE)
    n_cells = n_sweep * len(EFFORTS) + n_fixed
    eps_sweep = n_sweep * len(EFFORTS) * meta["n_tasks"]
    eps_fixed = n_fixed * meta["n_tasks"]
    actions = sum(stats[k]["catmix_total"] for _, _, k in MODELS)
    # Naming the default run in the path template while one cell is served from
    # a different one would be a false provenance statement, so the exception is
    # named here rather than left to the reader to trip over.
    if RUN_SUBDIR:
        who = ", ".join(f"{PROSE_NAME[k]} is read from `{r}`"
                        for k, r in sorted(RUN_SUBDIR.items()))
        run_txt = (f"`{DEFAULT_RUN}/` throughout, with one exception \u2014 "
                   f"{who}, for the reason given under run conditions below")
    else:
        run_txt = f"`{DEFAULT_RUN}/` throughout"

    # Which top-level tree each fixed-effort model actually lives under. Grouped
    # so the common case stays one short clause and the exception is named with
    # the model that causes it, rather than the reader being handed a path
    # template that is wrong for a quarter of the rows.
    trees = {}
    for k, p in FIXED_TREE.items():
        parts = p.split("/")
        if len(parts) != 4:
            raise SystemExit(
                f"provenance: FIXED_TREE[{k}!r] = {p!r} is not the expected "
                "<corpus>/<tree>/<Vendor>/<model> shape, so the path template "
                "cannot be derived. Update _mdb_provenance deliberately.")
        trees.setdefault(parts[1], []).append(k)
    tmpl = _join([
        f"`{t}/<Vendor>/<model>/`"
        + ("" if len(trees) == 1
           else " (" + _join([PROSE_NAME[k] for k in sorted(ks)]) + ")")
        for t, ks in sorted(trees.items())])
    # A directory name that reads like a condition and is not one gets corrected
    # in the same breath as the path, not three sections away. Looked up by tree
    # name (see TREE_NOTE) so the clarification can never end up attached to the
    # wrong directory.
    notes_ = [TREE_NOTE[t] for t in sorted(trees) if TREE_NOTE.get(t)]
    misread = (" " + _join(notes_) + "." if notes_ else "")
    body = (
        f"**Provenance.** Every figure is recomputed from "
        f"`eval_results/data_3_6_flash_v8/`, over **{n_cells} analysis cells**: "
        f"`aligned_setting_{{{','.join(EFFORTS)}}}/<Vendor>/<model>/` for the "
        f"{n_sweep} swept models ({eps_sweep:,} episodes) plus "
        f"{tmpl} for the {n_fixed} open-source "
        f"models, which were run once at `{PRIMARY_EFFORT}` rather than swept "
        f"({eps_fixed:,} episodes) \u2014 **{eps_sweep + eps_fixed:,} episodes "
        f"in total**, each cell being {meta['n_scenarios']} scenarios "
        f"\u00d7 {meta['n_rounds']} rounds = {meta['n_tasks']} tasks."
        f"{misread} "
        f"The run subdirectory is {run_txt}.\n\n"
        f"The corpus is **not** a clean cross-product, so do not multiply "
        f"anything by {len(MODELS)} here: only {n_sweep} of the {len(MODELS)} "
        f"models have an effort sweep, which is why \u00a79 covers {n_sweep} "
        f"models and every other section covers {len(MODELS)}.\n\n"
        f"Unless a table says otherwise, figures are at "
        f"`reasoning_effort={PRIMARY_EFFORT}`, the primary setting: "
        f"{len(MODELS)} models \u00d7 {meta['n_tasks']} tasks = "
        f"**{len(MODELS) * meta['n_tasks']:,} episodes** and "
        f"**{actions:,} recorded actions**. Regenerate with:\n\n"
        f"```\n"
        f"python tools/build_behavior_report.py --root $CLASHARNESS_DATA_ROOT"
        f" --effort {PRIMARY_EFFORT}\n"
        f"```")
    return {"provenance.para": body}


def _html_provenance(primary):
    """The one-line corpus provenance under the HTML title.

    The markdown front matter above has been derived from the loader's own
    constants since v4-flash arrived. This line was not, and by the fourteen-
    model roster it had drifted into two statements the report itself
    contradicts elsewhere:

      "open_source_high/ for 4"  -- three models live there; v4-flash is read
                                    from aligned_setting_high_ocr/.
      "r01 for all 14"           -- gemini-3.7-flash is read from r02, which the
                                    RUN_SUBDIR comment, the caveat box and every
                                    affected table footnote all say plainly.

    A provenance line that disagrees with the footnotes below it is worse than
    no provenance line, because it is the first thing a reader checks the
    corpus against. Both halves are now read off FIXED_TREE and RUN_SUBDIR.
    """
    swept = EFFORT_TREE.format(eff=primary)
    corpus = swept.split("/")[0]
    trees = {swept.split("/", 1)[1]: [k for _, _, k in SWEEP_MODELS]}
    for k, p in FIXED_TREE.items():
        parts = p.split("/")
        if parts[0] != corpus or len(parts) != 4:
            raise SystemExit(
                f"provenance: FIXED_TREE[{k!r}] = {p!r} is not "
                f"{corpus}/<tree>/<Vendor>/<model>, so the corpus prefix cannot "
                "be factored out of the header line. Update _html_provenance "
                "deliberately rather than letting it print a wrong path.")
        trees.setdefault(parts[1], []).append(k)
    # Swept tree first, then the fixed ones largest-first; naming each tree with
    # the number of models actually read from it is the whole point.
    order = [swept.split("/", 1)[1]] + sorted(
        (t for t in trees if t != swept.split("/", 1)[1]),
        key=lambda t: (-len(trees[t]), t))
    where = " + ".join(f"{t}/ for {len(trees[t])}" for t in order)
    # Full MODELS name, not the PROSE_NAME short form the body uses: this line
    # sits above the prose that establishes the short-name convention, so
    # "3.7-flash" would be the first model reference a reader meets.
    full = {k: n for n, _, k in MODELS}
    if RUN_SUBDIR:
        runs = (f"{DEFAULT_RUN} for {len(MODELS) - len(RUN_SUBDIR)} of "
                f"{len(MODELS)}, "
                + ", ".join(f"{r} for {full[k]}"
                            for k, r in sorted(RUN_SUBDIR.items())))
    else:
        runs = f"{DEFAULT_RUN} for all {len(MODELS)}"
    return (f"v8 corpus \u00b7 eval_results/{corpus}/{{{where}}} \u00b7 {runs} "
            f"\u00b7 effort grid read from all {len(EFFORTS)} levels for the "
            f"{len(SWEEP_MODELS)} swept models \u00b7 max_steps=20")


def _mdb_roster(stats, meta):
    """Front matter: the model roster and the run conditions that differ.

    Both halves are generated because both were wrong by hand. The roster
    listed nine rows against a thirteen-row report and gave every one of them
    "low / medium / high", which is false for the three open-source models. The
    run-conditions statement did not exist in the mirror at all: it was rendered
    into the HTML and simply never mirrored, so the markdown copy showed
    open-source rows with nothing to warn the reader that two of them were never
    sent an image.

    Model ids are read off the corpus headers by scan_run_conditions() rather
    than typed, which is how `gemini-3.7-flash` got into the roster without
    anyone guessing whether it carried a `-preview` suffix the way 3.1-pro does.
    """
    cond = meta["conditions"]
    swept = " / ".join(f"`{e}`" for e in EFFORTS)
    rows = ["| Model | model id | provider | `reasoning_effort` | run |",
            "|---|---|---|---|---|"]
    for _, _, k in MODELS:
        if k in FIXED_TREE:
            eff = f"`{PRIMARY_EFFORT}` only (not swept)"
        else:
            eff = swept
        # The run column exists because one row is not on the default run, and
        # the roster is where a reader looks to find that out.
        run = run_for(k)
        run = f"`{run}`" if run == DEFAULT_RUN else f"**`{run}`** (see below)"
        rows.append(f"| {PROSE_NAME[k]} | `{cond[k]['model']}` | "
                    f"{PROVIDER[k]} | {eff} | {run} |")
    # Which run_config settings are genuinely uniform, measured rather than
    # asserted. The hand-typed predecessor listed six settings as "identical
    # across all nine" and added "Measured config drift is zero" -- a claim
    # about a field (multimodal_input_mode) that is NOT uniform, in a document
    # that elsewhere devotes a box to saying so.
    same, differ, scoped = [], [], []
    for f in _COND_FIELDS:
        if f in _COND_IDENTITY:
            continue
        vals = {c[f] for c in cond.values()}
        if len(vals) == 1:
            v = next(iter(vals))
            if v not in (None, ""):
                same.append(f"`{f}={v}`")
            continue
        # A field can be non-uniform for two quite different reasons, and
        # collapsing them misleads. gemini_media_resolution is set on the four
        # Gemini models and absent everywhere else: that is a vendor-specific
        # knob, not a misalignment, and promising the reader it is "stated
        # per-model below" writes a cheque the run-conditions box does not
        # cash. The test is structural -- does the set of models carrying a
        # value coincide exactly with one provider block? -- so it cannot be
        # satisfied by a field that merely happens to be blank for two models.
        users = {k for k, c in cond.items() if c[f] not in (None, "")}
        held = {v for v in (cond[k][f] for k in users)}
        block = {k for k, p in PROVIDER.items() if p == PROVIDER[next(iter(users))]}
        if users and len(held) == 1 and users == block:
            scoped.append((f, next(iter(held)), PROVIDER[next(iter(users))],
                           len(users)))
        else:
            differ.append(f)
    rows += ["",
             f"Rows are grouped by provider, in the order every table and chart "
             f"in this report uses. Every field here is read off the "
             f"{sum(c['n_headers'] for c in cond.values()):,} episode headers "
             f"of the {len(MODELS)} primary cells, not transcribed.",
             "",
             f"Identical across all {len(MODELS)}: "
             + ", ".join(same) + f", and a **{STEP_CAP}-step budget**. "
             + "".join(f"`{f}={v}` is a {prov}-only knob, set on those {n} "
                       f"models and absent elsewhere. " for f, v, prov, n in scoped)
             + (f"The fields that are *not* uniform are "
                + ", ".join(f"`{f}`" for f in differ)
                + " \u2014 every one of them is stated per-model below."
                if differ else "No other field varies.")]
    table = "\n".join(rows)

    # The full statement the per-table footnotes point at. Markdown has no
    # id-carrying div, so the anchor is an explicit <a name>: GitHub strips
    # heading anchors of punctuation in ways that are easy to get subtly wrong,
    # and check_os_blocks() verifies this exact string is present.
    nm = PROSE_NAME
    bits = []
    for k in PROSE_KEYS:
        parts = []
        if k in OPEN_SOURCE_CAVEAT:
            parts.append(OPEN_SOURCE_CAVEAT[k])
        if k in _run_caveats():
            parts.append("run selection: " + _run_caveats()[k])
        if parts:
            bits.append(f"- **{nm[k]}** \u2014 {'; '.join(parts)}.")
    box = "\n".join([
        f'<a name="{MD_CAVEAT_ANCHOR}"></a>',
        "",
        "**Run conditions that are not comparable across the field.** All "
        f"{len(MODELS)} models ran the same harness build, the same "
        f"{meta['n_tasks']} tasks and the same nominal effort; what follows is "
        "the residue that did not align. Each line is read off that model's own "
        "episode headers and checked against this list on every build.",
        "",
        *bits,
        "",
        # Shared with the HTML caveat box, not restated. What stood here was a
        # hand-typed copy that had drifted into nonsense: it said the "3
        # path-mode models rank 1st and 5th of 14" -- three models given two
        # ranks, the third never named -- while quoting a floor gap computed
        # over an earlier, smaller field. Rendering both copies from one
        # function makes that particular divergence unrepresentable.
        _grounding_claim(stats, asset_gate_ranking(stats), nm,
                         lambda s: f"**{s}**", lambda s: f"*{s}*").rstrip(),
        "",
        f"Separately, the {len(FIXED_TREE)} open-source models were run "
        f"**once**, at `{PRIMARY_EFFORT}`, so they appear in every main table "
        "but not in the effort chapter (\u00a79).",
    ])
    return {"roster.table": table, "roster.conditions": box}


def _mdb_command_surface(stats, meta):
    """Section 3: raw invocation counts for the six load-bearing verbs."""
    keys = [k for _, _, k in MODELS]
    verbs = ["asset open", "asset list", "file read", "message search",
             "file create", "message send"]
    cols = [("total", {k: sum(stats[k]["cmdmix"].values()) for k in keys})]
    cols += [(f"`{v}`", {k: stats[k]["cmdmix"].get(v, 0) for k in keys})
             for v in verbs]
    rows = _md_table(keys, cols)
    rows += ["",
             f"Counts are total invocations across all {meta['n_tasks']} tasks "
             f"at `reasoning_effort={PRIMARY_EFFORT}`; each column's high and "
             "low are bold. The complete per-command inventory is in the "
             "command appendix of the HTML report."]
    return {"command_surface.table": "\n".join(rows)}


def _mdb_catmix(stats, meta):
    """Section 2: every action rolled up into the glossary's categories.

    The hand-typed predecessor showed TEN columns -- it merged the two
    `agent_*` categories and dropped `file_delete` outright -- under a caption
    claiming "eleven". Neither number was the truth: the categorisation the
    rest of the report is built on has len(CAT_ORDER) of them, the HTML bars
    draw all of them, and the build asserts the roll-up loses no action. Emit
    the same set the glossary defines so the two cannot disagree.
    """
    keys = [k for _, _, k in MODELS]
    pct = lambda v: f"{v:.1f}%"                                   # noqa: E731
    cols = [("total", {k: stats[k]["catmix_total"] for k in keys})]
    cols += [(f"`{c}`",
              {k: 100.0 * stats[k]["catmix"].get(c, 0) / stats[k]["catmix_total"]
               for k in keys})
             for c in CAT_ORDER]
    fmt = {f"`{c}`": pct for c in CAT_ORDER}
    rows = [f"Total actions and their distribution across the "
            f"{len(CAT_ORDER)} action categories defined in the reading guide, "
            f"all {meta['n_tasks']} tasks at "
            f"`reasoning_effort={PRIMARY_EFFORT}`. Percentages are share of "
            "that model's own total and sum to 100% by construction; each "
            "column's lone high and lone low are bold.",
            ""]
    rows += _md_table(keys, cols, fmt)

    # A column of zeros invites the reader to assume it was padded in. Say why
    # it is there, with the count derived rather than asserted.
    rare = []
    for c in CAT_ORDER:
        users = [k for k in keys if stats[k]["catmix"].get(c, 0)]
        if len(users) * 2 <= len(keys):
            total = sum(stats[k]["catmix"].get(c, 0) for k in keys)
            rare.append((c, total, users))
    caption = []
    if rare:
        bits = []
        for c, total, users in rare:
            who = ", ".join(PROSE_NAME[k] for k in users) or "no model at all"
            bits.append(f"`{c}` accounts for **{total:,}** actions in the "
                        f"entire corpus ({who}; zero for the other "
                        f"{len(keys) - len(users)} models)")
        head = ("One column is carried" if len(rare) == 1
                else f"{len(rare)} columns are carried")
        caption.append(f"{head} because the roll-up is MECE rather than "
                       f"because the field uses it: " + "; ".join(bits) + ". "
                       "No action is dropped from the totals \u2014 the build "
                       "fails if the category counts and the raw command "
                       "counts disagree.")
    if caption:
        rows += ["", " ".join(caption)]
    return {"catmix.table": "\n".join(rows)}


def _mdb_grounding(stats, meta):
    """Section 4: opens per task against pass rate.

    This is the one table in the report that is NOT in provider order. The
    sort IS the finding -- the point is that walking the grounding axis from
    end to end moves the pass column in no direction -- so the row order is
    load-bearing here in a way it is nowhere else, and the lead-in says so.
    """
    keys = sorted([k for _, _, k in MODELS],
                  key=lambda k: (stats[k]["open_per_task"], PROSE_NAME[k]))
    cols = [("opens/task", {k: stats[k]["open_per_task"] for k in keys}),
            ("zero-open tasks", {k: stats[k]["zero_open"] for k in keys}),
            ("pass", {k: stats[k]["pass_pct"] for k in keys})]
    fmt = {"opens/task": lambda v: f"{v:.2f}", "pass": lambda v: f"{v:.1f}%"}
    rows = [f"Grounding volume against outcome over all {meta['n_tasks']} "
            f"tasks at `reasoning_effort={PRIMARY_EFFORT}`, **ordered by "
            f"opens/task** rather than by provider: this is the one table "
            f"here whose row order carries an argument, so it is sorted along "
            f"the axis being tested. `zero-open tasks` counts episodes in "
            f"which the model never issued a single `asset open`.",
            ""]
    # No bold on the sort column: its extremes are the first and last row by
    # construction, so marking them adds ink and no information.
    rows += _md_table(keys, cols, fmt, nobold=("opens/task",))
    return {"grounding.table": "\n".join(rows)}


# Stop reasons, in the order the section presents them, with the display label
# for the two whose raw names are unreadable in a column header. The build
# refuses to render an unrecognised reason rather than silently omitting it,
# which is the only way a column can go missing without anyone noticing.
STOP_ORDER = [
    ("none", "none"),
    ("missing_required_side_effect", "missing_required_side_effect"),
    ("incomplete_required_submission", "incomplete_required_submission"),
    ("reached max_steps", "reached max_steps"),
    ("stopped after 3 identical no-progress transitions", "3-identical loop"),
    ("generation_stall_empty_reply", "empty reply"),
    # Last, and deliberately not adjacent to the model-authored outcomes: this
    # column is the provider giving up, not the model. It exists as a column at
    # all because these episodes used to be filed under `none` -- an empty
    # stop_reason reads as a clean finish -- which put infrastructure failures
    # in the same bucket as passes. See analyze() and the {{pf...}} tokens.
    ("provider_failure", "provider_failure"),
]


def _mdb_stops(stats, meta):
    """Section 5: how episodes ended."""
    keys = [k for _, _, k in MODELS]
    known = {raw for raw, _ in STOP_ORDER}
    seen = set()
    for k in keys:
        seen.update(stats[k]["stops"])
    unknown = seen - known
    if unknown:
        raise ValueError(
            "unmapped stop reason(s) %s -- add them to STOP_ORDER; a stop "
            "reason that is not a column is an episode the table loses"
            % sorted(unknown))
    for k in keys:
        total = sum(stats[k]["stops"].values())
        if total != stats[k]["n"]:
            raise ValueError(
                f"{k}: stop reasons sum to {total}, not {stats[k]['n']}")
    cols = [(label, {k: stats[k]["stops"].get(raw, 0) for k in keys})
            for raw, label in STOP_ORDER]
    rows = [f"Episode outcomes over all {meta['n_tasks']} tasks at "
            f"`reasoning_effort={PRIMARY_EFFORT}`. `none` means the episode "
            f"ended cleanly with the model declaring itself finished; "
            f"`provider_failure` means the provider killed the call mid-episode "
            f"and is the one column here that is not a decision the model made. "
            f"The columns are exhaustive: every episode lands in exactly one, "
            f"and the build fails if the row does not sum to "
            f"{meta['n_tasks']}.",
            ""]
    rows += _md_table(keys, cols)
    return {"stops.table": "\n".join(rows)}


def _mdb_steps(stats, meta):
    """Section 6: step counts, tail waste and step-level errors."""
    keys = [k for _, _, k in MODELS]
    e = {k: stats[k]["effort"] for k in keys}
    one = lambda v: f"{v:.1f}"                                    # noqa: E731
    cols = [("mean steps", {k: e[k]["step_mean"] for k in keys}),
            ("p50", {k: e[k]["step_p50"] for k in keys}),
            ("p50 on success", {k: e[k]["succ_step_p50"] for k in keys}),
            ("p50 on failure", {k: e[k]["fail_step_p50"] for k in keys}),
            ("tail waste (mean)", {k: e[k]["tail_waste_mean"] for k in keys}),
            ("step errors", {k: e[k]["step_err_pct"] for k in keys})]
    fmt = {h: one for h, _ in cols}
    fmt["step errors"] = lambda v: f"{v:.1f}%"
    # Tail waste is measured from the last step that ACTUALLY gained score, so
    # an episode that never gained any has no tail to measure and is left out
    # of the mean. Quoting the mean without its denominator would be the worst
    # kind of wrong on exactly the model the section is about: 3.5-flash scores
    # on a small minority of episodes, so its tail-waste figure describes that
    # minority and not its behaviour at large.
    tw_n = {k: e[k]["tail_waste_n"] for k in keys}
    lo_k = min(keys, key=lambda k: tw_n[k])
    rows = [f"Step behaviour over all {meta['n_tasks']} tasks at "
            f"`reasoning_effort={PRIMARY_EFFORT}`. **Step errors** is the "
            f"share of steps that returned a non-zero exit code or carried an "
            f"error tag. **Tail waste** is the number of steps taken after "
            f"the last step that actually gained score, so it is defined only "
            f"for episodes that gained score at all: the column averages over "
            f"{min(tw_n.values()):,}\u2013{max(tw_n.values()):,} of the "
            f"{meta['n_tasks']} episodes depending on the model "
            f"({PROSE_NAME[lo_k]} lowest, at {tw_n[lo_k]:,}), and says "
            f"nothing about the rest.",
            ""]
    rows += _md_table(keys, cols, fmt)
    return {"steps.table": "\n".join(rows)}


# -- the tables that were transcribed by hand, and went stale -------------------
# Seven more tables in the mirror were typed out by hand and then left behind
# when the study moved from a nine-model `medium` cut to the current
# fourteen-model `high` one. They did not merely lose five rows each. EVERY CELL
# IN THEM WAS A MEASUREMENT OF A DIFFERENT CELL OF THE GRID: checked against
# this build's own stats, the section-1 pass column reproduced the `medium`
# column of the effort sweep to the decimal on all nine rows, and sonnet-5's
# `field_contains` read 62.5 where the primary level says 73.6 -- an 11.1-point
# error presented as a measurement. The mirror went on calling medium "the
# primary setting throughout this report" while its HTML twin had moved to high
# and said so ten times.
#
# Nothing caught it, and nothing could have: sync_notes_md.py compares only what
# is GENERATED, these were not generated, and so the sync tool reported "all
# passages and blocks match" over a document whose every remaining table was
# wrong. A guard that covers a fifth of a file certifies the other four fifths
# by omission.
#
# So they are emitted here, on the same footing as the five blocks that already
# were. The rule this file has been applying since the command table is that a
# table is pure transcription and typing one buys nothing -- the only reason
# these were exempt is that nobody had got to them yet.
#
# What is deliberately NOT converted:
#   * the reasoning-token sweep (section 9.1) is nine models because the
#     measurement is nine models -- see the header of this file.
#   * the notification-routing table (section 11.3) is nine ROUNDS, not nine
#     models, and is unaffected by the roster.
#   * section 12's binding-constraint table is an editorial judgement per model,
#     not a transcription, and there is nothing here to generate it from.


def _mdb_signatures(stats, meta):
    """Section 1: one headline row per model \u2014 outcome, shape and price.

    The two money columns are left unbolded on purpose. `_lone_extremes` ranks
    the RENDERED cell, an unpriced model renders as `n/a`, and `n/a` strips to
    zero -- so the mark for "cheapest" would land on the one model whose cost
    is not known rather than on the one that is actually cheapest. Teaching the
    shared helper about a per-column sentinel would push that special case onto
    every other table; naming the gap in the caption costs one sentence.
    """
    keys = [k for _, _, k in MODELS]
    c = {k: stats[k]["cost"] for k in keys}
    e = {k: stats[k]["effort"] for k in keys}
    money = lambda v: "n/a" if v is None else f"{v:,.4f}"         # noqa: E731
    cols = [
        ("pass@budget", {k: stats[k]["pass_pct"] for k in keys}),
        ("passed", {k: stats[k]["passed"] for k in keys}),
        ("avg score", {k: stats[k]["avg_score"] for k in keys}),
        ("step p50", {k: e[k]["step_p50"] for k in keys}),
        ("mean steps", {k: e[k]["step_mean"] for k in keys}),
        ("capped", {k: stats[k]["stops"].get("reached max_steps", 0)
                    for k in keys}),
        ("opens/task", {k: stats[k]["open_per_task"] for k in keys}),
        ("writes/task", {k: stats[k]["write_per_task"] for k in keys}),
        ("zero-open", {k: stats[k]["zero_open"] for k in keys}),
        ("$/task", {k: c[k]["cost_per_task"] for k in keys}),
        ("$/solved", {k: c[k]["cost_per_solved"] for k in keys}),
    ]
    fmt = {"pass@budget": lambda v: f"{v:.1f}%",
           "avg score": lambda v: f"{v:.3f}",
           "step p50": lambda v: f"{v:.1f}",
           "mean steps": lambda v: f"{v:.2f}",
           "opens/task": lambda v: f"{v:.2f}",
           "writes/task": lambda v: f"{v:.2f}",
           "$/task": money, "$/solved": money}
    unpriced = [k for k in keys if c[k]["cost_per_task"] is None]
    rows = [f"Every model over all {meta['n_tasks']} tasks at "
            f"`reasoning_effort={PRIMARY_EFFORT}`. **pass@budget** is the "
            f"all-or-nothing rate: every scored unit of the task satisfied "
            f"within the step budget. **capped** counts episodes stopped at "
            f"`max_steps` rather than by the model. Each column's lone high "
            f"and lone low are bold.",
            ""]
    rows += _md_table(keys, cols, fmt, nobold=("$/task", "$/solved"))
    if unpriced:
        # _join, not esc_join: these are bare model names with no internal
        # punctuation, and esc_join's "; also," separator is for fragments
        # that carry their own commas.
        who = _join([PROSE_NAME[k] for k in unpriced])
        rows += ["",
                 f"The money columns carry no bold and read `n/a` for "
                 f"{who}: token usage was recorded, but "
                 f"{'these models have' if len(unpriced) > 1 else 'this model has'}"
                 f" no verified first-party rate card, and a blank is the "
                 f"honest cell. Costs are as-run list prices, not "
                 f"like-for-like \u2014 see the cost chapter for the two "
                 f"confounds that make the cheapest rows look cheaper than "
                 f"they are."]
    return {"signatures.table": "\n".join(rows)}


def _mdb_concentration(stats, meta):
    """Section 7.1: are failures independent across the units of a task?

    `lift` is the observed all-or-nothing pass rate over the rate predicted
    from that model's own per-check rate raised to its own mean unit count.
    Above 1 means losses pile onto fewer tasks than independence predicts.
    """
    keys = sorted([k for _, _, k in MODELS],
                  key=lambda k: (-stats[k]["clustering"]["lift"], PROSE_NAME[k]))
    cl = {k: stats[k]["clustering"] for k in keys}
    pct = lambda v: f"{v:.1f}%"                                   # noqa: E731
    two = lambda v: f"{v:.2f}"                                    # noqa: E731
    cols = [
        ("per-check pass", {k: cl[k]["per_check_pct"] for k in keys}),
        ("observed pass", {k: cl[k]["obs_pass_pct"] for k in keys}),
        ("if independent", {k: cl[k]["indep_pass_pct"] for k in keys}),
        ("excess", {k: cl[k]["excess_pp"] for k in keys}),
        ("lift", {k: cl[k]["lift"] for k in keys}),
        ("95% CI", {k: cl[k]["lift_ci"] for k in keys}),
        ("overdispersion", {k: cl[k]["overdispersion"] for k in keys}),
    ]
    fmt = {"per-check pass": pct, "observed pass": pct, "if independent": pct,
           "excess": lambda v: f"{_signed(v)}pp",
           "lift": two, "overdispersion": two,
           "95% CI": lambda v: f"[{v[0]:.2f}, {v[1]:.2f}]"}
    # Denominators are not uniform: an episode lost to a provider failure takes
    # its task's units out of that model's corpus, so quoting one n_tasks for
    # the table would be wrong for exactly the models that lost episodes.
    full = max(cl[k]["n_tasks"] for k in keys)
    short = [k for k in keys if cl[k]["n_tasks"] != full]
    span = (f"{full} tasks / {max(cl[k]['n_checks'] for k in keys):,} checks")
    note = ""
    if short:
        bits = esc_join([f"{PROSE_NAME[k]} on {cl[k]['n_tasks']} / "
                         f"{cl[k]['n_checks']:,}" for k in short])
        note = (f" \u2014 except {bits}, which lost an episode to a provider "
                f"failure and is scored on what it was actually given")
    rows = [f"Mean scored units per task is "
            f"{max(cl[k]['mean_k'] for k in keys):.2f}. Computed over {span}"
            f"{note}. **Ordered by lift**, which is the column the section is "
            f"about: observed pass rate over the rate an independent-failure "
            f"model predicts from the same per-check accuracy. Lift above 1 "
            f"means a model is either right about a task or wrong about it "
            f"rather than shedding units at random.",
            ""]
    # The CI is a pair, not a magnitude; ranking it would compare its rendered
    # digits as one number and mark a bracket at random.
    rows += _md_table(keys, cols, fmt, nobold=("95% CI",))
    return {"concentration.table": "\n".join(rows)}


def _mdb_check_split(stats, meta):
    """Section 7.2: pass rate by the kind of checker that graded the unit.

    `graded_pass_pct`, not `pass_pct`: a `semantic_match` unit whose check was
    never invoked (the model never produced the artefact to grade) is attrition
    rather than a wrong answer, and pooling the two makes the judged column
    look worse for the models that fail earliest.
    """
    keys = sorted([k for _, _, k in MODELS],
                  key=lambda k: (-stats[k]["check_split"]["semantic_match"]
                                 ["graded_pass_pct"], PROSE_NAME[k]))
    cs = {k: stats[k]["check_split"] for k in keys}
    one = lambda v: f"{v:.1f}"                                    # noqa: E731
    buckets = ["semantic_match", "field_contains", "other"]
    cols = [(f"`field_{b}`" if b == "semantic_match" else f"`{b}`",
             {k: cs[k][b]["graded_pass_pct"] for k in keys})
            for b in buckets]
    fmt = {h: one for h, _ in cols}
    ni = {k: cs[k]["semantic_match"]["not_invoked"] for k in keys}
    rows = [f"`graded_pass_pct` over all {meta['n_tasks']} tasks at "
            f"`reasoning_effort={PRIMARY_EFFORT}`, **ordered by the judged "
            f"column**. Rates are over units the checker actually ran on: "
            f"`field_semantic_match` units go ungraded when the artefact they "
            f"would grade was never produced, which happens "
            f"{min(ni.values()):,}\u2013{max(ni.values()):,} times per model, "
            f"and counting those as wrong answers would charge the same "
            f"episode twice.",
            ""]
    rows += _md_table(keys, cols, fmt)
    return {"check_split.table": "\n".join(rows)}


def _mdb_fc_decomp(stats, meta):
    """Section 7.3: the deterministic column split into its two sub-buckets.

    `field_contains` pools two different abilities. The asset gate asks
    whether the model opened the right file at all; the literal-content bucket
    asks whether what it then wrote contains an exact string. Both are graded
    by deterministic checkers -- no judge touches either -- so this is the part
    of the corpus where a per-model number is free of judge attrition.
    """
    keys = sorted([k for _, _, k in MODELS],
                  key=lambda k: (-stats[k]["check_split"]["field_contains"]
                                 ["graded_pass_pct"], PROSE_NAME[k]))
    cs = {k: stats[k]["check_split"] for k in keys}
    asset = {k: cs[k]["field_contains_asset"]["graded_pass_pct"] for k in keys}
    content = {k: cs[k]["field_contains_content"]["graded_pass_pct"]
               for k in keys}
    one = lambda v: f"{v:.1f}"                                    # noqa: E731
    cols = [("asset gate", asset),
            ("literal content", content),
            ("pooled", {k: cs[k]["field_contains"]["graded_pass_pct"]
                        for k in keys}),
            ("content \u2212 asset",
             {k: round(content[k] - asset[k], 1) for k in keys})]
    fmt = {"asset gate": one, "literal content": one, "pooled": one,
           "content \u2212 asset": _signed}
    # Sub-bucket denominators, derived. They are NOT a constant: the hand-typed
    # predecessor put "(822)" and "(256)" in the column headers, which was
    # wrong for the one model that lost an episode and so lost its units too.
    na = {cs[k]["field_contains_asset"]["total"] for k in keys}
    nc = {cs[k]["field_contains_content"]["total"] for k in keys}
    def _span(s):
        return f"{min(s)}" if len(s) == 1 else f"{min(s)}\u2013{max(s)}"
    rows = [f"The deterministic column decomposed: **asset gate** "
            f"({_span(na)} units) is *did you open the right file*, "
            f"**literal content** ({_span(nc)} units) is *does what you wrote "
            f"contain this exact string*. `graded == total` for both on every "
            f"model \u2014 these are the units no judge grades \u2014 so "
            f"nothing here is exposed to the attrition that distorts the "
            f"judged column. **Ordered by the pooled rate.**",
            ""]
    rows += _md_table(keys, cols, fmt)
    return {"fc_decomp.table": "\n".join(rows)}


def _mono_phrase(keys, mono):
    """"N of M models (...)", naming whichever side of the split is shorter.

    _mdb_gold_length names the EXCEPTIONS, which reads well while almost every
    row is monotone. The same sentence about mean score would have to name
    thirteen exceptions to report one monotone row, so the side that gets
    named is chosen by length rather than fixed. Either way the count itself is
    counted, never typed: the prose this replaces claimed "8 of 9" and "3 of 9"
    over a nine-model table, and both numbers were already wrong for the field
    the table now carries.
    """
    n = len(keys)
    if len(mono) == n:
        return f"all {_numword(n)} models"
    if not mono:
        return f"none of the {_numword(n)} models"
    off = [PROSE_NAME[k] for k in keys if k not in mono]
    hit = [PROSE_NAME[k] for k in keys if k in mono]
    head = f"{_numword(len(mono))} of {_numword(n)} models"
    if len(off) <= len(hit):
        tail = "is the exception" if len(off) == 1 else "are the exceptions"
        return f"{head} ({_join(off)} {tail})"
    return f"{head} ({_join(hit)})"


def _mdb_unit_bands(stats, meta, unit_sum):
    """Section 8.1: pass rate and mean score banded by scored-unit count.

    MODELS order, not a ranking, for the reason _mdb_gold_length gives: the
    claim is about the SHAPE of each row across the bands, and sorting on any
    one band would put a thumb on it.

    The pooled row is appended AFTER _md_table rather than passed in as a
    fourteenth key, so that the column extremes rank models against models. An
    aggregate cannot be the field's best or worst row, and letting it into the
    ranking would eventually bold it as though it were -- which is the reading
    the hand-typed predecessor invited by setting the whole pooled row in bold.
    Here only the label carries emphasis, marking it as the summary line.
    """
    keys = [k for _, _, k in MODELS]
    labels = [b for b, _ in UNIT_BANDS]
    by, pooled = unit_sum["by_model"], unit_sum["pooled"]
    one = lambda v: f"{v:.1f}"                                    # noqa: E731
    three = lambda v: f"{v:.3f}"                                  # noqa: E731
    cols = ([(f"{b} pass", {k: by[k][b]["pass_pct"] for k in keys})
             for b in labels]
            + [(f"{b} score", {k: by[k][b]["avg_score"] for k in keys})
               for b in labels])
    fmt = {h: (one if h.endswith("pass") else three) for h, _ in cols}

    ns = " / ".join(str(r["n_tasks"]) for r in unit_sum["bands"])
    # Denominators are not uniform, for the reason unit_band_summary explains,
    # and the models they are short on are named rather than averaged over.
    note = ""
    if unit_sum["short"]:
        bits = []
        for k in sorted(unit_sum["short"], key=lambda k: PROSE_NAME[k]):
            got = " / ".join(str(by[k][b]["n"]) for b in labels)
            lost = unit_sum["n_tasks"] - sum(by[k][b]["n"] for b in labels)
            bits.append(f"{PROSE_NAME[k]} on {got}, "
                        f"{_numword(lost)} episode{'' if lost == 1 else 's'} "
                        f"short")
        note = (f" \u2014 except {esc_join(bits)}, lost to a provider failure "
                f"and carrying no rubric to band, so those tasks are dropped "
                f"rather than scored as misses")
    mono_p = [k for k in keys
              if by[k][labels[0]]["pass_pct"] >= by[k][labels[1]]["pass_pct"]
              >= by[k][labels[2]]["pass_pct"]]
    mono_s = [k for k in keys
              if by[k][labels[0]]["avg_score"] >= by[k][labels[1]]["avg_score"]
              >= by[k][labels[2]]["avg_score"]]
    lo, hi = pooled[labels[0]]["pass_pct"], pooled[labels[2]]["pass_pct"]
    rows = [f"Banding the {unit_sum['n_tasks']} tasks by number of **scored "
            f"units** ({' / '.join(labels)}; n = {ns}; total "
            f"{unit_sum['n_units']:,} units){note}. Pass rate falls "
            f"monotonically across the three bands for "
            f"{_mono_phrase(keys, mono_p)}, pooled {lo:.1f}% \u2192 {hi:.1f}%, "
            f"a {lo - hi:.1f}pp spread. Mean score does the same for "
            f"{_mono_phrase(keys, mono_s)}, so the axis is clean on "
            f"all-or-nothing passing and not on partial credit.",
            ""]
    rows += _md_table(keys, cols, fmt)
    rows.append("| **pooled** | " + " | ".join(
        fmt[h](pooled[h.split()[0]]["pass_pct" if h.endswith("pass")
                                    else "avg_score"]) for h, _ in cols) + " |")
    return {"unit_bands.table": "\n".join(rows)}


def _mdb_gold_length(stats, meta, gold_sum):
    """Section 8.5: pass rate banded by the length of the reference solution.

    Model order is MODELS order rather than a ranking: the claim the section
    makes is about the SHAPE of each row (does it fall as the reference gets
    longer), and sorting by any one band would put a thumb on that scale.
    """
    keys = [k for _, _, k in MODELS]
    bands = [b for b, _ in GOLD_BANDS] + [OVERALL_BAND]
    short_hdr = ["short", "mid", "long", "all"]
    by = {k: {r["band"]: r for r in gold_sum["by_model"][k]} for k in keys}
    missing = sorted({b for k in keys for b in bands if b not in by[k]})
    if missing:
        raise ValueError(f"gold summary is missing band(s) {missing} \u2014 "
                         "GOLD_BANDS and the per-model rows have diverged")
    one = lambda v: f"{v:.1f}"                                    # noqa: E731
    cols = [(h, {k: by[k][b]["pass_pct"] for k in keys})
            for h, b in zip(short_hdr, bands)]
    fmt = {h: one for h, _ in cols}
    meta_b = {r["band"]: r for r in gold_sum["bands"]}
    ns = " / ".join(str(meta_b[b]["n_gold"]) for b, _ in
                    zip(bands[:-1], GOLD_BANDS))
    # Count the monotone rows rather than asserting a number: the previous copy
    # of this paragraph claimed "all nine" against its own table, was corrected
    # to "eight of nine", and would have gone wrong again on a wider roster.
    mono = [k for k in keys
            if by[k][bands[0]]["pass_pct"] >= by[k][bands[1]]["pass_pct"]
            >= by[k][bands[2]]["pass_pct"]]
    if len(mono) == len(keys):
        mono_txt = f"all {_numword(len(keys))} models"
    else:
        # _join, not esc_join: bare model names, no internal punctuation.
        off = [PROSE_NAME[k] for k in keys if k not in mono]
        mono_txt = (f"{_numword(len(mono))} of {_numword(len(keys))} models "
                    f"({_join(off)} "
                    f"{'are the exceptions' if len(off) > 1 else 'is the exception'})")
    rows = [f"Banding the {gold_sum['covered']} covered tasks by the length of "
            f"their reference solution into short / mid / long tertiles "
            f"(n = {ns}; reference length runs "
            f"{meta_b[OVERALL_BAND]['gold_min']}\u2013"
            f"{meta_b[OVERALL_BAND]['gold_max']} steps). Pass rate falls "
            f"monotonically across the three bands for {mono_txt}.",
            ""]
    rows += _md_table(keys, cols, fmt)
    return {"gold_length.table": "\n".join(rows)}


def _effort_triple(vals, fmt, mark=True):
    """One cell holding low / medium / high, with a lone row-best in bold.

    Bolding here is ROW-local and cannot come from `_md_table`, which ranks
    down a column. The comparison is on rendered text for the same reason
    `_lone_extremes` uses it: an advantage invisible at display precision is
    not one the reader can check.

    `mark=False` for columns where the maximum is not the best outcome. Mean
    steps and capped share the row with pass rate and score but point the
    other way, and marking their maximum would put the emphasis on a model's
    WORST setting under a caption promising its best -- while telling the
    reader nothing, since both rise with effort for almost every row anyway.
    """
    text = [fmt(v) for v in vals]
    if not mark:
        return " / ".join(text)
    best = max(text, key=_num_key)
    lone = sum(1 for t in text if t == best) == 1
    return " / ".join(f"**{t}**" if lone and t == best else t for t in text)


def _mdb_effort_grid(stats, meta, grid):
    """Section 9.2: the low/medium/high grid, one row per swept model.

    SWEEP_MODELS, not MODELS. Four models were run at a single setting and have
    no low or medium cell at all, so they are absent from this chapter by
    construction rather than by omission -- the chapter says so in its own
    scope note, and naming them here would attach a run-conditions footnote to
    a table none of their rows appear in.
    """
    C = grid["cells"]
    keys = [k for _n, _s, k in SWEEP_MODELS]
    order = sorted(keys, key=lambda k: (-C[(k, PRIMARY_EFFORT)]["pass_pct"],
                                        PROSE_NAME[k]))
    one = lambda v: f"{v:.1f}"                                    # noqa: E731
    # (header, stats field, formatter, is the maximum the best outcome)
    specs = [("pass low/med/high", "pass_pct", lambda v: f"{v:.1f}", True),
             ("score low/med/high", "avg_score", lambda v: f"{v:.3f}", True),
             ("mean steps l/m/h", "mean_steps", lambda v: f"{v:.2f}", False),
             ("capped% l/m/h", "capped_pct", one, False)]
    cols = [(h, {k: _effort_triple([C[(k, e)][f] for e in EFFORTS], g, m)
                 for k in order})
            for h, f, g, m in specs]
    rows = [f"The sweep, {_numword(len(order))} models \u00d7 "
            f"{_numword(len(EFFORTS))} settings, {meta['n_tasks']} tasks per "
            f"cell, **ordered by pass rate at "
            f"`{PRIMARY_EFFORT}`**. Each cell reads low / medium / high. In "
            f"the two outcome columns a lone row-best is bold, so the mark "
            f"shows which setting a model peaks at rather than how it ranks "
            f"against the field; mean steps and capped carry no mark because "
            f"their maximum is the worst cell, not the best. The remaining "
            f"models in this report were run at a single setting and have no "
            f"cells here.",
            ""]
    # Every column is a composite; ranking one down the column would compare
    # three concatenated numbers as if they were one.
    rows += _md_table(order, cols, {h: (lambda s: s) for h, _ in cols},
                      nobold=tuple(h for h, _ in cols))
    return {"effort_grid.table": "\n".join(rows)}


def _mdb_effort_mcnemar(stats, meta, grid):
    """Section 9.2: paired low -> high, same tasks, per swept model."""
    C, M = grid["cells"], grid["mcnemar"]
    order = sorted((k for _n, _s, k in SWEEP_MODELS),
                   key=lambda k: (-M[k]["delta_pp"], PROSE_NAME[k]))
    def verdict(k):
        if not M[k]["significant"]:
            return "n.s."
        return "**real**" if M[k]["delta_pp"] > 0 else "**real (negative)**"
    cols = [("\u0394pp", {k: M[k]["delta_pp"] for k in order}),
            ("lost", {k: M[k]["lost"] for k in order}),
            ("gained", {k: M[k]["gained"] for k in order}),
            ("p", {k: M[k]["p"] for k in order}),
            ("verdict", {k: verdict(k) for k in order})]
    fmt = {"\u0394pp": _signed, "p": _pval, "verdict": lambda s: s}
    paired = {M[k]["paired_n"] for k in order}
    n_sig = sum(1 for k in order if M[k]["significant"])
    n_neg = sum(1 for k in order if M[k]["significant"]
                and M[k]["delta_pp"] < 0)
    rows = [f"Paired McNemar, `low` \u2192 `high`, on the "
            f"{'/'.join(str(p) for p in sorted(paired))} tasks each model "
            f"attempted at both settings. **lost** and **gained** are tasks "
            f"that flipped, so a model can move a great many tasks and still "
            f"end level. {_numword(n_sig).capitalize()} of "
            f"{_numword(len(order))} models shift significantly at "
            f"p < 0.05"
            + (f", and {_numword(n_neg)} of those "
               f"{'shifts are' if n_neg > 1 else 'shifts is'} downward"
               if n_neg else "") + ".",
            ""]
    # `verdict` is a label, not a magnitude. `p` is a magnitude but goes
    # unmarked for two reasons: `_num_key` has no exponent in its character
    # class, so a cell reading `1.0e-07` would be ranked as the string `1.0-07`
    # and land nowhere near where it belongs -- and a bolded smallest p would
    # in any case only restate the verdict column sitting next to it.
    rows += _md_table(order, cols, fmt, nobold=("verdict", "p"),
                      left=("verdict",))
    return {"effort_mcnemar.table": "\n".join(rows)}


def _signed(v):
    """A signed number set with the report's minus sign, not a hyphen.

    A value that rounds to zero at display precision is emitted unsigned. The
    sign is a claim about direction, and `+0.0` asserts one the printed digits
    cannot support -- 3.6-flash gained 34 tasks and lost 34 going low to high,
    which is not a small improvement, it is none.
    """
    s = f"{v:+.1f}"
    return s[1:] if s in ("+0.0", "-0.0") else s.replace("-", "\u2212")


def _pval(p):
    """p to the precision it carries, scientific once it stops being readable."""
    if p < 0.01:
        return f"{p:.1e}"
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def _all_cmd_keys(stats):
    keys = set()
    for _, _, key in MODELS:
        keys.update(stats[key]["cmdmix"].keys())
    return keys


def render_glossary(stats, meta):
    """Reading guide: how the raw action DSL maps into the 12 categories. Built
    from the data (every observed sub-command routed through categorize) so it can
    never drift from what the bars actually count."""
    bycat = defaultdict(list)
    other_fams = set()
    for k in _all_cmd_keys(stats):
        c = categorize(k)[0]
        bycat[c].append(k)
        if c == "other":
            other_fams.add(other_family(k))
    rows = []
    for c in CAT_ORDER:
        if c == "other":
            # 47 raw verbs would be unreadable in one cell -- list the domain
            # families instead and defer the full list to the appendix.
            fams = ", ".join(f"<code>{esc(f)}</code>" for f in sorted(other_fams))
            sub_html = (f"domain operations: {fams} "
                        '\u2014 every verb enumerated in the '
                        '<a href="#cmd-appendix">command appendix</a>')
        else:
            subs = sorted(bycat.get(c, []))
            sub_html = ", ".join(f"<code>{esc(s)}</code>" for s in subs) or "\u2014"
        rows.append(
            f'<tr><td class="gl-cat"><span class="lgd" style="background:{CAT_COLOR[c]}">'
            f'</span>{esc(CAT_LABEL[c])}</td><td>{sub_html}</td></tr>')
    return (
        '<h2 id="glossary">How to read this report</h2>'
        '<p class="h2sub">The action DSL has '
        f"{len(_all_cmd_keys(stats))} distinct sub-commands. To keep the "
        "charts legible each one is rolled up into a single <b>action category</b>; "
        '<b>rates are calls \u00f7 ' f"{meta['n_tasks']}" ' tasks</b>. The mapping is '
        "<b>MECE</b> \u2014 every action lands in exactly one category, and everything "
        "that is not a core asset / file / message / agent / wait verb falls into "
        "<code>other</code> (itemised under \u00a7 Action-category mix and listed in "
        'full in the <a href="#cmd-appendix">command appendix</a>). No command is '
        "dropped from the totals.</p>"
        '<table class="cmdtable glossary"><thead><tr><th>action category</th>'
        "<th>sub-commands mapped into it (observed in this dataset)</th></tr></thead>"
        f'<tbody>{"".join(rows)}</tbody></table>')


def render_appendix(stats, meta):
    """Exhaustive per-command inventory: every sub-command observed, with its
    category and per-model counts. This is the 'nothing is hidden' backstop."""
    keys = _all_cmd_keys(stats)

    def kt(k):
        return sum(stats[key]["cmdmix"].get(k, 0) for _, _, key in MODELS)

    ordered = sorted(keys, key=lambda k: (CAT_ORDER.index(categorize(k)[0]), -kt(k), k))
    head = ("<tr><th>sub-command</th><th>category</th>"
            + "".join(f"<th>{esc(MODEL_LABEL[k])}</th>" for _, _, k in MODELS)
            + "<th>total</th></tr>")
    body = []
    for k in ordered:
        cat = categorize(k)[0]
        vals = [stats[key]["cmdmix"].get(k, 0) for _, _, key in MODELS]
        cells = "".join(f'<td class="sc">{v}</td>' for v in vals)
        body.append(
            f'<tr><td class="ck"><code>{esc(k)}</code></td>'
            f'<td class="catc"><span class="lgd" style="background:{CAT_COLOR[cat]}">'
            f'</span>{esc(CAT_LABEL[cat])}</td>{cells}'
            f'<td class="sc"><b>{kt(k)}</b></td></tr>')
    tot_cells = "".join(
        f'<td class="sc"><b>{stats[key]["catmix_total"]}</b></td>' for _, _, key in MODELS)
    grand = sum(stats[key]["catmix_total"] for _, _, key in MODELS)
    body.append(f'<tr class="famrow"><td>all commands</td><td>{len(keys)} distinct</td>'
                f'{tot_cells}<td class="sc"><b>{grand}</b></td></tr>')
    return (f'<h2 id="cmd-appendix">Appendix: complete command inventory '
            f'({len(keys)} sub-commands)</h2>'
            '<p class="h2sub">Every sub-command observed across all '
            f"{meta['n_tasks']} tasks per model, with the category it rolls up into. "
            "Rows sum (bottom) to the same action totals as the category bars \u2014 the "
            "full audit trail behind every aggregate in this report.</p>"
            f'<table class="cmdtable cmdinv"><thead>{head}</thead>'
            f'<tbody>{"".join(body)}</tbody></table>')


# Nine hues, retuned for the v8 slope panels. In the ten-model THINK corpus four of
# these sat in the same green/teal family (flash36 #16a34a, opus5 #065f46, luna
# #0f766e, opus #0891b2), which was survivable on a scatter -- dots there are far
# apart and individually labelled -- but not on a slope chart, where nine polylines
# cross in the middle of the frame and hue is the only thing telling them apart.
# Two changes: flash36 takes the magenta freed by gpt-5.5's removal, and sol moves
# from indigo #4338ca to a plain blue so it stops reading as sonnet's violet. Every
# label in every panel now also carries a swatch in its line color, so identification
# never depends on discriminating hue from memory.
# EXTENDED to thirteen. The original nine occupy roughly hue 0 (red), 28 (dark
# orange), 38 (amber), 166 (dark green), 177 (teal), 191 (cyan), 221 (blue), 262
# (violet) and 333 (pink), which leaves two usable arcs -- yellow-green through green
# (~60-140) and magenta-purple (~290-320) -- plus the desaturated axis. The four new
# entries are placed there rather than squeezed between existing hues:
#   flash37 lime ~82, kimi green ~142, glm fuchsia ~294, dsv4 neutral slate.
# Only flash37 joins the SLOPE panels, which draw SWEEP_MODELS and therefore now
# carry ten polylines rather than nine; the three open-source models appear only on
# scatters and tables, where points are labelled and separated.
DOTCOLORS = {"flash": "#ef4444", "flash36": "#db2777", "pro": "#f59e0b",
             "opus": "#0891b2", "opus5": "#065f46", "sonnet": "#7c3aed",
             "sol": "#2563eb", "luna": "#0f766e", "terra": "#b45309",
             "flash37": "#84cc16", "kimi": "#16a34a", "glm": "#c026d3",
             "dsv4": "#64748b", "dsv4f": "#334155"}

# A missing colour used to surface as a KeyError from inside SVG path
# generation, thousands of lines from the dictionary that caused it. Adding a
# model is the moment to notice, so check it here.
_missing_dot = [k for _, _, k in MODELS if k not in DOTCOLORS]
if _missing_dot:
    raise SystemExit(f"DOTCOLORS has no entry for {_missing_dot}")
del _missing_dot


def _scatter_svg(points, xtitle, ytitle, corners, xtick, ytick, aria,
                 xfloor=None, yfloor=None):
    """Shared 2-axis scatter for the nine models, drawn as inline SVG.

    points: list of {name, key, x, y, coord} (coord = the value string shown under
    the model name). Because several models cluster tightly, labels are de-cluttered
    into vertical ladders in the left/right margins -- a dot in the left half of the
    panel gets a left-margin label, one in the right half a right-margin label --
    each joined to its dot by a thin connector so no two labels ever overlap.
    corners: dict(tl, tr, bl, br) quadrant captions. xtick/ytick: value->str.

    xfloor/yfloor: hard lower bound for a quantity that cannot physically go below
    it (dollars, rates). Opt-in and default None, so a panel that does not pass one
    renders byte-for-byte as before. See bounds() for why the floor is needed."""
    # Checked here, before anything is drawn, because the dot loop below indexes
    # DOTCOLORS[p["key"]] directly: an unknown key otherwise surfaces as a bare
    # `KeyError: 'foo'` from inside an f-string with no clue which of the four
    # panels raised it. Same failure, named.
    plotted = {p["key"] for p in points}
    unknown = plotted - {key for _, _, key in MODELS}
    if unknown:
        raise SystemExit(f"scatter '{aria}': plotted keys {sorted(unknown)} are "
                         "not in MODELS, so they have no colour and no legend "
                         "entry. Add them to MODELS or drop them from the panel.")
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]

    def bounds(vals, floor=None):
        # The pad is scaled off the SPREAD (hi-lo) but subtracted from the MIN, so
        # a lineup whose smallest member is small relative to the spread lands below
        # zero: at $68 (luna) against $2,646 (opus-4.8) the pad is $412, six times
        # luna's entire run cost, and the axis opened at -$344. Nothing here knows a
        # quantity has a physical floor, so callers declare it. Only the low side is
        # clamped -- the high side has no such bound.
        lo, hi = min(vals), max(vals)
        pad = max((hi - lo) * 0.16, abs(hi) * 0.05 or 1.0)
        lo -= pad
        if floor is not None and lo < floor:
            lo = floor
        return lo, hi + pad

    xlo, xhi = bounds(xs, xfloor)
    ylo, yhi = bounds(ys, yfloor)
    W, H = 960, 480
    ml, mr, mt, mb = 188, 200, 30, 50
    pw, ph = W - ml - mr, H - mt - mb

    def sx(x):
        return ml + (x - xlo) / (xhi - xlo) * pw

    def sy(y):
        return mt + (1 - (y - ylo) / (yhi - ylo)) * ph

    for p in points:
        p["cx"], p["cy"] = sx(p["x"]), sy(p["y"])
    mid = ml + pw / 2
    left = sorted([p for p in points if p["cx"] < mid], key=lambda p: p["cy"])
    right = sorted([p for p in points if p["cx"] >= mid], key=lambda p: p["cy"])

    def ladder(group, anchor_x, anchor):
        gap = 34
        prev = -1e9
        for p in group:
            p["ly"] = max(p["cy"], prev + gap)
            p["lx"] = anchor_x
            p["anchor"] = anchor
            prev = p["ly"]
        if group:  # keep the ladder inside the panel
            over = group[-1]["ly"] - (mt + ph)
            if over > 0:
                for p in group:
                    p["ly"] -= over
            under = (mt + 14) - group[0]["ly"]
            if under > 0:
                for p in group:
                    p["ly"] += under

    ladder(left, ml - 12, "end")
    ladder(right, ml + pw + 12, "start")

    P = []
    P.append(f'<svg viewBox="0 0 {W} {H}" class="scz" '
             f'preserveAspectRatio="xMidYMid meet" role="img" aria-label="{aria}">')
    P.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" '
             f'fill="#fbfcfe" stroke="#e5e7eb"/>')
    midx, midy = ml + pw / 2, mt + ph / 2
    P.append(f'<line x1="{midx:.1f}" y1="{mt}" x2="{midx:.1f}" y2="{mt+ph}" '
             f'stroke="#eceff3" stroke-dasharray="4 4"/>')
    P.append(f'<line x1="{ml}" y1="{midy:.1f}" x2="{ml+pw}" y2="{midy:.1f}" '
             f'stroke="#eceff3" stroke-dasharray="4 4"/>')
    # ticks: y inside the top/bottom-left of the plot, x below the plot
    tk = 'fill="#5b6472" font-size="11" font-family="system-ui,sans-serif"'
    P.append(f'<text x="{ml+5}" y="{mt+13}" {tk}>{ytick(yhi)}</text>')
    P.append(f'<text x="{ml+5}" y="{mt+ph-5}" {tk}>{ytick(ylo)}</text>')
    P.append(f'<text x="{ml}" y="{mt+ph+17}" {tk}>{xtick(xlo)}</text>')
    P.append(f'<text x="{ml+pw}" y="{mt+ph+17}" text-anchor="end" {tk}>{xtick(xhi)}</text>')
    # axis titles
    at = 'fill="#5b6472" font-size="12" font-family="system-ui,sans-serif"'
    P.append(f'<text x="{ml+pw}" y="{H-6}" text-anchor="end" {at}>{esc(xtitle)} \u2192</text>')
    cyax = mt + ph / 2
    P.append(f'<text x="14" y="{cyax:.1f}" text-anchor="middle" {at} '
             f'transform="rotate(-90 14 {cyax:.1f})">{esc(ytitle)} \u2192</text>')
    # quadrant captions, centered in each half so they clear ticks and ladders
    # Quadrant captions name what each corner of the scatter means, so they are
    # content, not chrome, and have to clear AA (#b9bfc9 was 1.8:1 on the plot
    # fill -- effectively invisible). Placement at the quadrant centres, away
    # from the dots and ticks, is what keeps them subordinate now.
    cc = 'fill="#5b6472" font-size="10" font-family="system-ui,sans-serif"'
    P.append(f'<text x="{ml+pw*0.72:.0f}" y="{mt+14}" text-anchor="middle" {cc}>{esc(corners["tr"])}</text>')
    P.append(f'<text x="{ml+pw*0.28:.0f}" y="{mt+14}" text-anchor="middle" {cc}>{esc(corners["tl"])}</text>')
    P.append(f'<text x="{ml+pw*0.72:.0f}" y="{mt+ph-6}" text-anchor="middle" {cc}>{esc(corners["br"])}</text>')
    P.append(f'<text x="{ml+pw*0.28:.0f}" y="{mt+ph-6}" text-anchor="middle" {cc}>{esc(corners["bl"])}</text>')
    # connectors (under dots)
    for p in points:
        tx = p["lx"] + (4 if p["anchor"] == "end" else -4)
        P.append(f'<line x1="{p["cx"]:.1f}" y1="{p["cy"]:.1f}" x2="{tx:.1f}" '
                 f'y2="{p["ly"]:.1f}" stroke="#d1d5db" stroke-width="1"/>')
    # dots
    for p in points:
        P.append(f'<circle cx="{p["cx"]:.1f}" cy="{p["cy"]:.1f}" r="6" '
                 f'fill="{DOTCOLORS[p["key"]]}" stroke="#fff" stroke-width="2">'
                 f'<title>{esc(p["name"])}: {esc(p["coord"])}</title></circle>')
    # labels
    nm = 'font-size="12" font-weight="600" fill="#374151" font-family="system-ui,sans-serif"'
    co = 'font-size="10.5" fill="#5b6472" font-family="system-ui,sans-serif"'
    for p in points:
        P.append(f'<text x="{p["lx"]:.1f}" y="{p["ly"]-3:.1f}" text-anchor="{p["anchor"]}" '
                 f'{nm}>{esc(p["name"])}</text>'
                 f'<text x="{p["lx"]:.1f}" y="{p["ly"]+10:.1f}" text-anchor="{p["anchor"]}" '
                 f'{co}>{esc(p["coord"])}</text>')
    P.append("</svg>")
    # THE LEGEND KEYS THE DOTS THAT EXIST, NOT THE ROSTER.
    # This used to iterate MODELS unconditionally, so every scatter advertised a
    # colour swatch for all fourteen models no matter how many were actually
    # plotted. Three of the four panels happen to pass every model, which is why
    # it went unnoticed; the cost x pass-rate panel plots only the PRICED models,
    # and so shipped a swatch for deepseek-v4-flash pointing at a dot that is not
    # in the chart -- under a caption that reads "one dot per priced model
    # (colors keyed below)". A reader hunting for that dot cannot find it.
    # Deriving from `points` fixes the live panel and makes the other three
    # self-correcting if they ever start filtering (the clustering panel already
    # filters on stats[key]["clustering"]; it is only whole today by luck).
    # MODELS order is preserved so the legend stays grouped by provider.
    leg = "".join(
        f'<span class="lg"><span class="lgd" style="background:{DOTCOLORS[key]}"></span>'
        f'{esc(name)}</span>' for name, _, key in MODELS if key in plotted)
    return f'<div class="maplegend">{leg}</div><div class="sczbox">{"".join(P)}</div>'


def render_map(stats):
    """Grounding (asset opens/task, x) x Action (file writes/task, y) map."""
    points = []
    for name, _, key in MODELS:
        st = stats[key]
        x, y = st["open_per_task"], st["write_per_task"]
        points.append({"name": name, "key": key, "x": x, "y": y,
                       "coord": f"({x:g}, {y:g})"})
    corners = {"tr": "grounds & acts", "tl": "acts, low grounding",
               "br": "grounds, low action", "bl": "looks, neither"}
    svg = _scatter_svg(points, "grounding (opens/task)", "writes/task", corners,
                       lambda v: f"{v:.2f}", lambda v: f"{v:.2f}",
                       "grounding vs action scatter")
    return (
        "<h2>Grounding \u00d7 Action map</h2>"
        '<p class="h2sub">x = asset opens per task (grounding) \u00b7 '
        "y = file writes per task (create / write / edit). One dot per model "
        "(colors keyed below), with its (x, y) beside it \u2014 labels are laddered "
        "into the margins with connector lines so the tight clusters stay legible. "
        "Axes are zoomed to the data range, so position is <b>relative</b> across "
        f"these {len(MODELS)} models; dashed lines mark each axis mid-point. "
        "Top-right = grounds and acts the most; bottom-left = does least of both.</p>"
        + svg)


def _competition_ranks(keys, value):
    """Rank keys by `value` descending, sharing a rank on an exact tie.

    Competition ranking (1, 2, 2, 4) rather than a dense or arbitrary order.
    The distinction is not academic here: gpt-5.6-luna and deepseek-v4-pro both
    pass exactly 119 of 600, and printing one of them 8th and the other 9th
    invents a separation the corpus does not contain -- the reader would take
    it for a result. Ties are broken for LAYOUT only (by MODELS order), never
    for the rank number.
    """
    order = sorted(keys, key=lambda k: (-value(k), [m[2] for m in MODELS].index(k)))
    ranks, seen = {}, None
    for i, k in enumerate(order, 1):
        if seen is None or value(k) != value(seen):
            ranks[k], seen = i, k
        else:
            ranks[k] = ranks[seen]
    return order, ranks


def render_leaderboard(stats):
    """The standings, as a picture: who is ahead, by how much, on both metrics.

    WHY THIS EXISTS. Every performance figure in the report was either a table
    (exact but unordered on the page -- the main table is written in provider
    order, so the reader has to sort fourteen rows in their head to answer "who
    won") or the accuracy x score scatter (ordered, but by POSITION, which
    shows separation and hides rank -- a cluster of eight dots in the top-right
    is precisely where rank is least readable). Neither answers "where does
    this model place" at a glance. This does, and it is the only chart here
    that states a rank as a number.

    Two tracks, deliberately on SEPARATE scales and labelled as such:
      * pass@budget -- fully-solved tasks. The headline, and the sort key.
      * mean score  -- partial credit banked along the way, 0-1 by definition,
        so its axis runs to a fixed 1.0 rather than to the data.
    Rank is computed per track, and the DIFFERENCE between the two ranks is the
    interesting column: a model well above its process-score rank converts
    credit into closed tasks, one below it banks credit it never cashes. That
    is the near-miss signature the scatter can only gesture at.
    """
    keys = list(PROSE_KEYS)
    nm = {k: n for n, _, k in MODELS}
    pv = {k: stats[k]["pass_pct"] for k in keys}
    sv = {k: stats[k]["avg_score"] for k in keys}
    # Rank on the exact count, not the rounded percentage: at 600 tasks two
    # models 0.04 pp apart would print the same "19.8%" and take different
    # ranks, which looks like a typo rather than a measurement.
    exact = {k: stats[k]["passed"] / max(stats[k]["n"], 1) for k in keys}
    order, prank = _competition_ranks(keys, lambda k: exact[k])
    _, srank = _competition_ranks(keys, lambda k: sv[k])
    # ... and if the display ever DOES collapse two distinct rates onto one
    # printed value, say so here rather than shipping the ambiguity.
    for a, b_ in zip(order, order[1:]):
        if pv[a] == pv[b_] and exact[a] != exact[b_]:
            raise SystemExit(
                f"leaderboard: {nm[a]} and {nm[b_]} both print {pv[a]}% but "
                "hold different ranks. Raise the displayed precision.")

    W, rh, top, bot = 980, 31, 92, 52
    H = top + rh * len(order) + bot
    x_rank, x_name = 34, 208               # right edges of the two text columns
    x0, x1 = 220, 606                      # pass track
    x_pv = x1 + 12                         # pass value, left-aligned
    s0, s1 = 726, 856                      # score track
    x_sv = s1 + 10
    x_d = W - 8                            # delta column, right-aligned
    pcap = max(5.0, ceil(max(pv.values()) / 5.0) * 5.0)
    field = sum(pv.values()) / len(pv)
    fnt = 'font-family="system-ui,sans-serif"'
    P = [f'<svg viewBox="0 0 {W} {H}" class="scz" preserveAspectRatio="xMidYMid '
         f'meet" role="img" aria-label="{esc(_leaderboard_aria(order, nm, pv))}">']

    # --- headers -------------------------------------------------------
    P.append(f'<text x="{x1}" y="26" text-anchor="end" {fnt} font-size="12.5" '
             f'font-weight="700" fill="#1f2937">pass@budget '
             f'(% of {stats[order[0]]["n"]} tasks)</text>')
    P.append(f'<text x="{s1}" y="26" text-anchor="end" {fnt} font-size="12.5" '
             f'font-weight="700" fill="#1f2937">mean score (0\u20131)</text>')
    P.append(f'<text x="{x_d}" y="26" text-anchor="end" {fnt} font-size="12.5" '
             f'font-weight="700" fill="#1f2937">rank \u0394</text>')
    # pass axis ticks
    t = 0.0
    while t <= pcap + 1e-9:
        gx = x0 + (t / pcap) * (x1 - x0)
        P.append(f'<line x1="{gx:.1f}" y1="{top-16}" x2="{gx:.1f}" '
                 f'y2="{top + rh*len(order) - 8}" stroke="#eef2f7" '
                 f'stroke-width="1"/>')
        P.append(f'<text x="{gx:.1f}" y="{top-22}" text-anchor="middle" {fnt} '
                 f'font-size="10" fill="#94a3b8">{t:.0f}%</text>')
        t += 5.0
    # score axis: 0 and 1.0 only -- the track is 130px wide and more ticks in
    # it read as hatching rather than as a scale.
    for sv_t in (0.0, 0.5, 1.0):
        gx = s0 + sv_t * (s1 - s0)
        P.append(f'<line x1="{gx:.1f}" y1="{top-16}" x2="{gx:.1f}" '
                 f'y2="{top + rh*len(order) - 8}" stroke="#eef2f7" '
                 f'stroke-width="1"/>')
        P.append(f'<text x="{gx:.1f}" y="{top-22}" text-anchor="middle" {fnt} '
                 f'font-size="10" fill="#94a3b8">{sv_t:g}</text>')
    # --- rows ----------------------------------------------------------
    for i, k in enumerate(order):
        cy = top + rh * i
        mid = cy + rh / 2 - 1
        col = DOTCOLORS[k]
        if i % 2 == 0:
            P.append(f'<rect x="8" y="{cy - 2}" width="{W-16}" height="{rh-2}" '
                     f'fill="#fafbfc"/>')
        P.append(f'<text x="{x_rank}" y="{mid+4}" text-anchor="end" {fnt} '
                 f'font-size="13" font-weight="700" fill="#475569">'
                 f'{prank[k]}</text>')
        P.append(f'<text x="{x_name}" y="{mid+4}" text-anchor="end" {fnt} '
                 f'font-size="12.5" fill="#1f2937">{esc(nm[k])}</text>')
        # pass bar
        bw = (pv[k] / pcap) * (x1 - x0)
        P.append(f'<rect x="{x0}" y="{cy+4}" width="{max(bw,1.2):.1f}" '
                 f'height="{rh-12}" rx="3" fill="{col}"/>')
        P.append(f'<text x="{x_pv}" y="{mid+4}" {fnt} font-size="12" '
                 f'font-weight="700" fill="#1f2937">{pv[k]:.1f}%</text>')
        # score bar, same hue at low opacity so the eye reads it as secondary
        P.append(f'<rect x="{s0}" y="{cy+7}" width="{s1-s0}" '
                 f'height="{rh-18}" rx="2.5" fill="#eef2f7"/>')
        P.append(f'<rect x="{s0}" y="{cy+7}" width="{sv[k]*(s1-s0):.1f}" '
                 f'height="{rh-18}" rx="2.5" fill="{col}" opacity="0.45"/>')
        P.append(f'<text x="{x_sv}" y="{mid+4}" {fnt} font-size="11.5" '
                 f'fill="#475569">{sv[k]:.3f}</text>')
        # rank delta
        d = srank[k] - prank[k]
        if d:
            P.append(f'<text x="{x_d}" y="{mid+4}" text-anchor="end" {fnt} '
                     f'font-size="11.5" font-weight="700" '
                     f'fill="{"#15803d" if d > 0 else "#b91c1c"}">'
                     f'{"\u25b2" if d > 0 else "\u25bc"}{abs(d)}</text>')
        else:
            P.append(f'<text x="{x_d}" y="{mid+4}" text-anchor="end" {fnt} '
                     f'font-size="11.5" fill="#cbd5e1">\u2014</text>')

    # Field mean LAST, so it lies over the bars. Drawn first it was hidden
    # behind every bar that clears it -- i.e. behind exactly the models the
    # reader wants to compare against it, which is the whole point of the line.
    fx = x0 + (field / pcap) * (x1 - x0)
    P.append(f'<line x1="{fx:.1f}" y1="{top-16}" x2="{fx:.1f}" '
             f'y2="{top + rh*len(order) - 4}" stroke="#0f172a" opacity="0.55" '
             f'stroke-width="1.4" stroke-dasharray="5 3"/>')
    P.append(f'<text x="{fx:.1f}" y="{top + rh*len(order) + 13}" '
             f'text-anchor="middle" {fnt} font-size="10.5" fill="#475569">'
             f'field mean {field:.1f}%</text>')
    P.append("</svg>")
    return "".join(P), order, prank, srank


def _leaderboard_aria(order, nm, pv):
    """Screen-reader summary: the standings as a sentence."""
    return "model standings by pass rate: " + ", ".join(
        f"{nm[k]} {pv[k]:.1f}%" for k in order)


def render_standings(stats):
    """Leaderboard section: the chart plus a caption that reads it for you.

    Every example named in the caption is picked by the data, not typed. The
    caption's whole job is to teach the rank-delta column, so it works the
    largest delta in each direction as an example -- and if the field ever
    produces no divergence at all, it says that instead of inventing one.
    """
    svg, order, prank, srank = render_leaderboard(stats)
    nm = {k: n for n, _, k in MODELS}
    pv = {k: stats[k]["pass_pct"] for k in PROSE_KEYS}
    d = {k: srank[k] - prank[k] for k in PROSE_KEYS}
    top, bottom = order[0], order[-1]

    # Worked examples for the delta column, chosen by magnitude.
    up = max(order, key=lambda k: d[k])
    dn = min(order, key=lambda k: d[k])
    if d[up] > 0 and d[dn] < 0:
        gloss = (
            f"{esc(nm[up])} is the clearest converter \u2014 "
            f"{_ordinal(prank[up])} on pass but only {_ordinal(srank[up])} on "
            f"mean score, so it closes tasks that its running partial credit "
            f"does not predict. {esc(nm[dn])} runs the other way "
            f"({_ordinal(prank[dn])} on pass, {_ordinal(srank[dn])} on score): "
            f"credit banked and not cashed, which is the near-miss signature.")
    else:
        gloss = ("On this field the two rankings agree everywhere, so the "
                 "column is all dashes \u2014 no model closes tasks out of "
                 "line with the partial credit it accumulates.")

    # Ties are a property of the data, so the sentence about them is too.
    tied = sorted({r for r in prank.values()
                   if sum(1 for k in PROSE_KEYS if prank[k] == r) > 1})
    tie_txt = ""
    for r in tied:
        who = [k for k in order if prank[k] == r]
        tie_txt += (
            f" {esc(_join([nm[k] for k in who]))} share {_ordinal(r)}: they "
            f"pass the identical {stats[who[0]]['passed']} of "
            f"{stats[who[0]]['n']} tasks, so the chart gives them one rank "
            f"rather than inventing an order between them.")

    return (
        '<h2 id="standings">Standings</h2>'
        f'<p class="h2sub">One row per model, sorted by <b>pass@budget</b> at '
        f'<code>reasoning_effort={PRIMARY_EFFORT}</code> \u2014 the rank number '
        f'on the left is the answer to \u201cwhere does this model place\u201d, '
        f'which every other view in this report makes you derive. The wide bar '
        f'is pass rate (share of {stats[top]["n"]} tasks fully solved, scale '
        f'along the top); the narrow pale bar is <b>mean score</b>, the partial '
        f'credit banked along the way, on its own fixed <b>0\u20131</b> scale. '
        f'<b>The two scales are different</b> \u2014 bar lengths are comparable '
        f'down a column, never across one. The <b>rank \u0394</b> column is the '
        f'model\u2019s mean-score rank minus its pass rank: '
        f'<span style="color:#15803d;font-weight:700">\u25b2</span> means it '
        f'places better on closing than on process score, '
        f'<span style="color:#b91c1c;font-weight:700">\u25bc</span> the '
        f'reverse. {gloss} The spread is the point: {esc(nm[top])} at '
        f'<b>{pv[top]:.1f}%</b> to {esc(nm[bottom])} at '
        f'<b>{pv[bottom]:.1f}%</b>, against a field mean of '
        f'<b>{sum(pv.values())/len(pv):.1f}%</b>.{tie_txt}</p>'
        + svg + os_note(svg, cls="fignote"))


def render_acc_score_scatter(stats):
    """Accuracy (pass@budget, y) x process score (mean partial-credit score, x).

    Pass rate counts only fully-solved tasks; the mean score is the partial-credit
    the harness awards along the way (per-step score deltas summed per task, then
    averaged over all tasks). Plotting the two together separates *closing* from
    *progress*: a model far to the right of its height is banking partial credit it
    never converts into a pass -- the near-miss signature. Labels are laddered into
    the margins with connector lines because pass-rate and mean-score are highly
    correlated, so most of the ten models pile into a tight top-right cluster."""
    points = []
    for name, _, key in MODELS:
        st = stats[key]
        x, y = st["avg_score"], st["pass_pct"]
        points.append({"name": name, "key": key, "x": x, "y": y,
                       "coord": f"({x:.3f}, {y:.1f}%)"})
    corners = {"tr": "high score & pass rate",
               "tl": "passes despite lower score",
               "br": "banks partial credit, rarely closes",
               "bl": "low on both"}
    # y floored: a pass rate cannot be negative. x needs no floor -- the padded
    # low bound sits at 0.25, well clear of zero.
    svg = _scatter_svg(points, "mean score (partial credit)", "pass rate %",
                       corners, lambda v: f"{v:.2f}", lambda v: f"{v:.0f}%",
                       "pass rate vs mean score scatter", yfloor=0.0)
    return (
        "<h2>Accuracy \u00d7 process-score map</h2>"
        '<p class="h2sub">x = mean score per task (partial credit accumulated over '
        "the run \u2014 the intermediate-step score) \u00b7 y = pass rate "
        "(pass@budget, fully-solved tasks only). One dot per model (colors keyed "
        "below), with its (mean score, pass%) beside it \u2014 labels are laddered "
        "into the margins with connector lines so the tight clusters stay legible. "
        "Axes are zoomed to the data range, so position is <b>relative</b> across "
        f"these {len(MODELS)} models; dashed lines mark each axis mid-point. The two "
        "metrics "
        "are strongly correlated \u2014 the useful signal is the <b>gap</b>: a model "
        "to the <b>right of and below</b> the pack banks partial credit it never "
        "converts into a pass (the near-miss signature), while pass rate rewards "
        "<i>closing</i> the task.</p>"
        + svg)


def _clustering_walkthrough(stats, have):
    """Plain-language read-through of ONE model's row, generated from that row.

    The section previously opened with five stacked definitions and no worked
    example, so a reader had to hold p, mean_i(p**k), obs, excess and lift in their
    head simultaneously before any of them meant anything. This walks the model with
    the largest excess through all five in order, in sentences, with every number
    pulled from `stats` so it can never drift from the table underneath it.
    """
    name, key = max(have, key=lambda nk: stats[nk[1]]["clustering"]["excess_pp"])
    c = stats[key]["clustering"]
    miss = round(100 - c["per_check_pct"], 1)
    return (
        '<div class="callout"><b>How to read one row.</b> Take '
        f"<b>{esc(name)}</b>. It passes <b>{c['per_check_pct']}%</b> of the "
        "individual required checks it is graded on \u2014 that is its raw ability, "
        "and it is the only ability number in this section. Its tasks carry "
        f"{c['mean_k']:.2f} required checks on average and a task scores nothing "
        "unless <i>all</i> of them pass, so if those checks succeeded or failed "
        "independently of each other it would clear "
        f"<b>{c['indep_pass_pct']}%</b> of tasks. It actually clears "
        f"<b>{c['obs_pass_pct']}%</b> \u2014 <b>{c['excess_pp']:+.1f} points</b> more "
        "than its own accuracy can explain. Those extra passes are not extra "
        "skill; the per-check rate has already priced the skill in. They come "
        f"from <i>where</i> the misses land. The same {miss}% of checks that this "
        "model gets wrong would, if dealt out one per task, spoil "
        f"{c['excess_pp']:+.1f} points&rsquo; worth of otherwise-clean tasks; "
        "dealt in clumps, they mostly land on tasks that were already lost, and "
        "cost nothing extra. <b>Co-failure lift</b> states the same fact from the "
        "other side: once one required check in a task has failed, the next one "
        f"is <b>{c['lift']:.2f}\u00d7</b> likelier to fail than this "
        "model&rsquo;s overall miss rate would predict.</div>")


def _delta_hist_block(stats, have):
    """The failure-shape chart: observed minus each model's OWN independence baseline.

    This replaces a raw five-bucket histogram of "how many required checks did the
    task lose". That chart could not show what its own caption claimed. The raw
    histogram is dominated by ABILITY, not by clustering -- a strong model piles up
    at 0 lost checks, a weak one at 3-4 -- so all ten rows were near-monotone and
    near-identical in shape, and the caption's promised "bimodal, mass at both ends"
    signature appears in exactly zero of the ten models. It is not a rendering
    accident: no model's observed histogram is bimodal, because the corpus-wide
    difficulty gradient sets the location of the mode and clustering only perturbs
    it.

    Against each model's own independence baseline the signature is real: mass is
    pulled OUT of the middle buckets (lost exactly 1, 2 and 3) and pushed into BOTH
    ends (lost nothing / lost 4+). That is what "clustered" means mechanically --
    the same failed checks, repacked onto fewer tasks -- and it is only visible as a
    difference. It is NOT unanimous, and the caption no longer says it is; the count
    and the exceptions are computed below. Note which buckets are the "ends": lost 3
    sits with the middle, not with lost 4+. An earlier caption grouped it with the
    ends and thereby claimed a pattern only 6 of 13 rows have.

    The leftmost bucket is the anchor: delta at "lost 0" IS the excess column of the
    table above, by the identity noted in failure_clustering(). So the chart and the
    summary number are the same measurement at two resolutions, and a reader can
    check one against the other.
    """
    mx = max((abs(d) for _, k in have
              for d in stats[k]["clustering"]["delta_hist_pp"]), default=1.0) or 1.0
    labels = ["lost 0", "lost 1", "lost 2", "lost 3", "lost 4+"]
    sub = ["task passed", "", "", "", ""]
    head = ("<tr><th>model</th>"
            + "".join(f'<th>{l}{f"<div class=bsub>{s}</div>" if s else ""}</th>'
                      for l, s in zip(labels, sub))
            + "</tr>")
    rows = []
    for name, key in have:
        c = stats[key]["clustering"]
        cells = []
        for b in range(5):
            d = c["delta_hist_pp"][b]
            o, i = c["fail_hist_pct"][b], c["indep_hist_pct"][b]
            cls = "pos" if d > 0 else ("neg" if d < 0 else "zero")
            w = abs(d) / mx * 30.0
            cells.append(
                f'<td class="dv"><span class="dvn {cls}">{d:+.1f}</span>'
                f'<span class="dvb"><i class="{cls}" style="width:{w:.1f}px"></i>'
                f"</span>"
                f'<span class="dvo">{o:.1f} vs {i:.1f}</span></td>')
        rows.append(f"<tr><td>{esc(name)}</td>" + "".join(cells) + "</tr>")
    # The signature claim is COUNTED, not asserted. It used to read "All nine models
    # show the same signature ... every row is positive at both ends and negative in
    # the middle", written when the field was nine. On the current field that
    # sentence is false BOTH as stated and as originally reasoned. Counts below
    # re-measured 2026-08-14 over all FOURTEEN rows:
    #   * as originally worded, "both ends" meant lost-0 and "lost 3 or more", i.e.
    #     buckets 3 AND 4. ZERO of the 14 rows qualify on that reading, because
    #     bucket 3 ("lost exactly 3") is NEGATIVE for thirteen of the fourteen (the
    #     lone hold-out, gemini-3.5-flash, is +0.0, not positive) -- bucket 3 belongs
    #     to the middle, not to an end. The old wording mis-split the histogram.
    #     (An earlier revision of this comment put that figure at "6 of 13"; that
    #     number is not reproducible from delta_hist_pp under any reading of "both
    #     ends" tried here, so it is retired rather than rescaled.)
    #   * read correctly -- out of lost 1/2/3, into lost 0 and lost 4+ -- 11 of 14
    #     rows show it exactly, and the three exceptions (gemini-3.5-flash flat at
    #     +0.0 on lost 3, deepseek-v4-flash +1.8 and glm-5.2 +0.7 on lost 1) are
    #     near-zero, not reversed. The shipped sentence reads "11 of the 14 rows".
    # So count the rows and name the exceptions rather than claim unanimity, and let
    # the unanimity that IS there (every row up at both true ends) be counted too.
    dh = {k: stats[k]["clustering"]["delta_hist_pp"] for _n, k in have}

    def _sig(d):
        return d[0] > 0 and d[4] > 0 and d[1] < 0 and d[2] < 0 and d[3] < 0

    n_sig = sum(1 for d in dh.values() if _sig(d))
    BUCKET = {0: "lost 0", 1: "lost 1", 2: "lost 2", 3: "lost 3", 4: "lost 4+"}

    def _why(d):
        bad = [f"{BUCKET[b]} {d[b]:+.1f}" for b in (1, 2, 3) if d[b] >= 0]
        bad += [f"{BUCKET[b]} {d[b]:+.1f}" for b in (0, 4) if d[b] <= 0]
        return ", ".join(bad)

    exc = [(n, k) for n, k in have if not _sig(dh[k])]
    exc_txt = ""
    if exc:
        exc_txt = (f"The {_numword(len(exc))} exception"
                   + ("s are" if len(exc) > 1 else " is")
                   + " a near-zero bucket rather than a reversal \u2014 "
                   + "; ".join(f"{esc(n)} ({_why(dh[k])})" for n, k in exc) + ". ")
    _una = [(lambda d: d[0] > 0, "positive at <i>lost nothing</i>"),
            (lambda d: d[4] > 0, "positive at <i>lost 4 or more</i>"),
            (lambda d: d[2] < 0, "negative at <i>lost exactly 2</i>")]
    hold = [t for f, t in _una if all(f(d) for d in dh.values())]
    una_txt = (f"All {len(have)} rows are " + ", ".join(hold[:-1])
               + (" and " if len(hold) > 1 else "") + hold[-1] + ". ") if hold else ""

    legend = (
        '<div class="legend">'
        '<span class="lg"><span class="lgd" style="background:#4f46e5"></span>'
        "more tasks here than independence predicts</span>"
        '<span class="lg"><span class="lgd" style="background:#d97706"></span>'
        "fewer than independence predicts</span>"
        '<span class="lg">each cell: <b>signed gap in points</b> \u00b7 bar length '
        "\u00b7 <span style=\"color:#5b6472\">observed vs independent</span></span>"
        "</div>")
    return (
        "<h3>Where the failures actually go, against each model&rsquo;s own baseline</h3>"
        '<p class="h2sub">Each row is one model&rsquo;s distribution of <i>how many '
        "required checks a task lost</i>, minus what that same model&rsquo;s own "
        "per-check rate predicts if the checks failed independently. The raw "
        "distribution is not shown because it measures the wrong thing: it is "
        "dominated by how good the model is (a strong model piles up on the left, a "
        "weak one on the right), so it just redraws the per-check column. The "
        "<i>difference</i> isolates the shape. <b>The leftmost column is the "
        "excess column of the table above</b> \u2014 a task "
        "passes exactly when it loses zero required checks \u2014 so the chart and that "
        "number are one measurement at two resolutions.</p>"
        + legend
        + f'<table class="cmdtable dvt"><thead>{head}</thead><tbody>'
        + "".join(rows) + "</tbody></table>"
        + f'<p class="h2sub"><b>{n_sig} of the {len(have)} rows show the same '
        "signature, and it is not the one the middle of this section is about.</b> "
        "Mass is pulled out of the middle buckets \u2014 <i>lost exactly 1</i>, "
        "<i>2</i> and <i>3</i> \u2014 and pushed into both ends, <i>lost nothing</i> "
        "and <i>lost 4 or more</i>. That is clustering in its mechanical form: the "
        "same wrong checks, repacked onto fewer tasks. "
        + exc_txt + una_txt
        + "The models differ in "
        "<i>magnitude</i>, not in direction, and the magnitude ordering is the "
        "excess column. Note that the raw (undifferenced) histograms are all "
        "<i>unimodal</i>; the two-humped shape exists only relative to the baseline, "
        "which is why it has to be drawn as a difference.</p>")


def render_clustering_exemplar(stats):
    """The abstract difference above, shown once on named tasks.

    Every number in the prose is computed from the chosen scenario, so the text
    cannot drift away from the grid if the corpus or the selection rule changes.
    """
    ex = clustering_exemplar(stats)
    if not ex:
        return ""
    rounds = ex["rounds"]
    hdr = "".join(f'<th>{esc(r["round"])}</th>' for r in rounds)

    def line(mk, label, tot, passed, cls):
        cells = []
        for r in rounds:
            v = r[mk]
            k = "ex0" if v == 0 else ("exall" if v >= r["k"] else "exn")
            cells.append(f'<td class="{k}">{v}</td>')
        return (f'<tr><td class="{cls}">{esc(label)}</td>' + "".join(cells)
                + f'<td class="extot">{tot}</td>'
                + f'<td class="extot">{passed}/{len(rounds)}</td></tr>')

    free = ex["free"]
    free_txt = ""
    if free:
        hi, lo = (ex["a_name"], ex["b_name"]) if free["a"] > free["b"] \
            else (ex["b_name"], ex["a_name"])
        hiv, lov = max(free["a"], free["b"]), min(free["a"], free["b"])
        free_txt = (
            f' Look at <b>{esc(free["round"])}</b>: {esc(hi)} loses {hiv} of the '
            f'{free["k"]} required checks there and {esc(lo)} loses {lov} \u2014 and '
            "the round is lost either way, so the "
            f"{hiv - lov} extra misses cost nothing. That is where a clustered "
            "model puts them.")
    return (
        "<h3>The same wrong checks, arranged two ways</h3>"
        '<p class="h2sub">One scenario family, ten rounds, two models chosen so that '
        "raw ability is nearly held fixed: "
        f"<b>{esc(ex['a_name'])}</b> ({ex['a_pc']:.1f}% of required checks correct, "
        f"lift {ex['a_lift']:.2f}\u00d7) and <b>{esc(ex['b_name'])}</b> "
        f"({ex['b_pc']:.1f}%, lift {ex['b_lift']:.2f}\u00d7). Each cell is how many "
        "of that round\u2019s required checks the model lost; a round is passed only "
        "at <b>0</b>. This family was picked because both models lost the "
        f"<b>same total</b> \u2014 {ex['lost']} checks each \u2014 so anything left in "
        "the outcome is arrangement, not accuracy.</p>"
        f'<table class="cmdtable extbl"><thead><tr><th>{esc(ex["scenario"])}</th>'
        f'{hdr}<th>lost</th><th>passed</th></tr></thead><tbody>'
        + f'<tr><td class="exk">required checks</td>'
        + "".join(f'<td class="exk">{r["k"]}</td>' for r in rounds)
        + '<td class="exk"></td><td class="exk"></td></tr>'
        + line("a", ex["a_name"], ex["lost"], ex["a_passed"], "exm")
        + line("b", ex["b_name"], ex["lost"], ex["b_passed"], "exm")
        + "</tbody></table>"
        + '<p class="h2sub">Identical error budget, '
        f"{ex['a_passed']} rounds passed versus {ex['b_passed']}."
        + free_txt
        + f" {esc(ex['b_name'])} spends the same misses one at a time on rounds "
        "that were otherwise clean, and each one costs a whole task. Across all "
        f"{ex['n_common']} shared tasks the asymmetry holds in the same direction: "
        f"{ex['a_only']} tasks are clean for {esc(ex['a_name'])} while "
        f"{esc(ex['b_name'])} loses at least one check, against {ex['b_only']} the "
        "other way round. This is an illustration of the mechanism, not evidence "
        "for it \u2014 nothing here is corrected for task difficulty; that correction "
        "is the last table in this section.</p>")

def render_clustering(stats):
    """Third headline axis: the SHAPE of a model's required-check failures.

    pass@budget says how often a model wins; mean score says how much partial credit
    it banks; neither says whether its losses are concentrated or spread. Two models
    at the same per-check ability land at different pass rates purely because of that
    shape, so it is reported as its own axis rather than left implicit inside
    pass@budget. See failure_clustering() for the definitions.

    Layout, in the order a reader needs it: one worked row in sentences, then the
    summary table, then the shape chart that shows the mechanism, then the
    difficulty correction that takes most of it back.
    """
    have = [(name, key) for name, _, key in MODELS if stats[key].get("clustering")]
    if not have:
        return ""
    rows = []
    best_ex = max(stats[k]["clustering"]["excess_pp"] for _, k in have)
    for name, key in have:
        c = stats[key]["clustering"]
        strong = ' style="font-weight:700"' if c["excess_pp"] == best_ex else ""
        lc = c.get("lift_ci") or [None, None]
        rows.append(
            f'<tr><td>{esc(name)}</td>'
            f'<td class="num">{c["per_check_pct"]}%</td>'
            f'<td class="num">{c["indep_pass_pct"]}%</td>'
            f'<td class="num">{c["obs_pass_pct"]}%</td>'
            f'<td class="num"{strong}>+{c["excess_pp"]}</td>'
            f'<td class="num">{c.get("lift"):.2f}\u00d7'
            f'<span class="ci">[{lc[0]:.2f}, {lc[1]:.2f}]</span></td></tr>')
    head = ("<tr><th>model</th><th>per-check</th><th>if independent</th>"
            "<th>observed</th><th>excess (pp)</th>"
            "<th>co-failure lift <span class=\"ci\">95% CI</span></th></tr>")
    points = [{"name": name, "key": key,
               "x": stats[key]["clustering"]["per_check_pct"],
               "y": stats[key]["clustering"]["excess_pp"],
               "coord": f'({stats[key]["clustering"]["per_check_pct"]}%, '
                        f'+{stats[key]["clustering"]["excess_pp"]}pp)'}
              for name, key in have]
    corners = {"tr": "able & clustered (converts ability into passes)",
               "tl": "clustered despite low per-check ability",
               "br": "able but spreads its failures thin",
               "bl": "low on both"}
    svg = _scatter_svg(points, "per-check pass rate %", "excess pass rate (pp)",
                       corners, lambda v: f"{v:.0f}%", lambda v: f"+{v:.0f}",
                       "per-check ability vs failure clustering scatter")
    mean_k = stats[have[0][1]]["clustering"]["mean_k"]
    return (
        "<h2>Failure clustering (third axis)</h2>"
        '<p class="h2sub">Two models with the <i>same</i> per-check accuracy can '
        "land at very different task pass rates. A task passes only if "
        f"<i>every</i> required check passes (mean {mean_k:.2f} required checks per "
        "task \u2014 and that count is a property of the <b>task</b>, so the same "
        "distribution applies to every model). So it matters enormously whether a "
        "model&rsquo;s wrong answers <i>pile onto the same few tasks</i>, leaving "
        "the rest clean, or <i>spread one-per-task</i> and spoil many tasks that "
        "were otherwise finished. Nothing in pass@budget or mean score separates "
        "those two, which is why the shape gets its own axis here.</p>"
        + _clustering_walkthrough(stats, have)
        + '<p class="h2sub"><b>if independent</b> = the pass rate this '
        "model&rsquo;s own per-check rate would produce if the checks within a task "
        "failed independently: mean<sub>i</sub>(p<sup>k\u1d62</sup>), a closed form "
        "validated against a 2000-draw permutation baseline to within 0.05pp. "
        "<b>excess</b> = observed \u2212 independent, in points of pass rate. "
        "<b>co-failure lift</b> = the chance a <i>second</i> required check in a "
        "task fails once one already has, over the model&rsquo;s own marginal "
        "failure rate. Its independence baseline is exactly 1.0\u00d7 as an algebraic "
        "identity \u2014 nothing is estimated for the null \u2014 and it is capped at "
        "1/q, which differs by model, so read the rank order rather than the "
        "magnitudes.</p>"
        f'<table class="cmdtable"><thead>{head}</thead><tbody>'
        + "".join(rows) + "</tbody></table>"
        + _delta_hist_block(stats, have)
        + render_clustering_exemplar(stats)
        + '<p class="h2sub warn"><b>Read everything above as description, not as a '
        "model property.</b> Every lift sits over 1.0, but so would every lift from "
        "a model that failed <i>independently</i>: this corpus spans a wide "
        "difficulty range, so failures pile onto the hard tasks by construction, for "
        "everyone. The correction is below, and it takes most of the effect back.</p>"
        + '<p class="h2sub">Plotted against raw ability below: vertical position is '
        "the clustering premium, horizontal position the per-check accuracy that "
        "premium is applied to. A model sitting high is converting the same "
        "underlying accuracy into more pass@budget.</p>"
        + svg
        + render_clustering_strata(stats))


def render_clustering_strata(stats):
    """The correction to the table above: how much of the clustering is the corpus.

    Merged into ONE table. This used to be two, presenting the same quantity --
    co-failure lift -- under three different nulls (whole corpus, within difficulty
    tercile, against the other nine models) in two separate places, leaving the
    reader to carry three framings and work out unaided which one settles the
    question. They are now four columns of one row per model, so the retreat from
    "1.60x, clearly clustered" to "+0.34 above its peers, and only that" is legible
    left-to-right, and the column that actually isolates the model is marked.

    Merging also fixes a quiet inconsistency: the headline lift was computed on each
    model's own graded set (596-600 tasks) while the terciles were computed on the
    595 tasks common to all nine. Every lift in this table is now on the common 595,
    so the four numbers in a row are comparable to each other.
    """
    st = clustering_strata(stats)
    if not st:
        return ""
    name_of = {key: name for name, _, key in MODELS}
    pm = st["per_model"]
    mark = {"up": '<span class="up">\u25b2</span>',
            "down": '<span class="down">\u25bc</span>', "flat": ""}

    def cell(d, cls="num"):
        return (f'<td class="{cls}">{d["lift"]:.2f}{mark[d["verdict"]]}'
                f'<span class="ci">[{d["ci"][0]:.2f}, {d["ci"][1]:.2f}]</span></td>')

    order = sorted(pm, key=lambda k: -pm[k]["peer"]["diff"])
    trows = "".join(
        f'<tr><td>{esc(name_of.get(k, k))}</td>'
        f'<td class="num dim">{pm[k]["peer"]["lift"]:.2f}</td>'
        + "".join(cell(t) for t in pm[k]["tiers"])
        + f'<td class="num key">{pm[k]["peer"]["diff"]:+.3f}'
        f'{mark[pm[k]["peer"]["verdict"]]}'
        f'<span class="ci">[{pm[k]["peer"]["ci"][0]:+.3f}, '
        f'{pm[k]["peer"]["ci"][1]:+.3f}]</span></td></tr>' for k in order)
    n_shift = sum(1 for k in pm if abs(pm[k]["peer"]["lift"]
                  - stats[k]["clustering"]["lift"]) >= 0.005)
    n_up = sum(1 for k in pm if pm[k]["tiers"][1]["verdict"] == "up")
    n_dn = sum(1 for k in pm if pm[k]["tiers"][1]["verdict"] == "down")
    easy_up = [name_of.get(k, k) for _n, _p, k in MODELS
               if k in pm and pm[k]["tiers"][0]["verdict"] == "up"]
    easy_txt = ", ".join(esc(x) for x in easy_up[:-1]) + " and " + esc(easy_up[-1]) \
        if len(easy_up) > 1 else (esc(easy_up[0]) if easy_up else "no model")

    def _ptxt(d):
        return ("p &lt; 0.001" if d["p"] < 0.001 else f"p = {d['p']:.3f}")

    def _gtxt(g, d):
        word = {"up": "above", "down": "below",
                "flat": "indistinguishable from"}[d["verdict"]]
        return (f"<b>{esc(g)}</b> {d['mean_lift']:.2f} vs {d['other_mean_lift']:.2f}, "
                f"{d['diff']:+.3f} "
                f'<span class="ci">[{d["ci"][0]:+.3f}, {d["ci"][1]:+.3f}]</span>'
                f" \u2014 {word} the rest ({_ptxt(d)})")

    gr = st.get("groups") or {}
    if gr:
        ordered = sorted(gr.items(), key=lambda kv: -kv[1]["diff"])
        grp = (
            '<p class="h2sub"><b>A family-level claim has to be tested at the '
            "family level.</b> The intervals in the last column that clear zero "
            "still <i>overlap</i> intervals that do not, so reading that column by "
            "eye is not a test. Averaging each family against the rest of the field "
            "on the same resample: "
            + "; ".join(_gtxt(g, d) for g, d in ordered)
            + ". The vendor split therefore survives the correction, but as a "
            "statement about families, not about any single pair of models.</p>")
    else:
        grp = ""
    return (
        "<h3>Is the clustering the model, or the corpus?</h3>"
        '<p class="h2sub"><b>Mostly the corpus.</b> The same quantity \u2014 '
        "co-failure lift \u2014 is shown below under three progressively stricter "
        "nulls, left to right. Column 1 is the headline number and it is "
        "<i>not</i> a model property. Columns 2\u20134 recompute it inside bands of "
        "comparable task difficulty, where most of it evaporates. The last column "
        "is the only one that isolates the model, and it is the one to quote.</p>"
        f'<p class="h2sub">Task difficulty is scored by how the <i>other '
        f"{len(pm) - 1}</i> "
        "models did on each task (leave-one-out, so no model contributes to the "
        "axis it is judged against), split into terciles. The final column asks the "
        "comparative question instead: not &ldquo;is this lift above 1.0?&rdquo; but "
        f"&ldquo;is it above the other {len(pm) - 1} <i>on the same tasks</i>?&rdquo; "
        f"\u2014 tasks are resampled once per draw and all {len(pm)} models "
        "recomputed together, "
        f"so corpus difficulty cancels. All columns use the {st['n_tasks']} tasks "
        f"every model was graded on, with {st['draws']}-draw bootstrap CIs over "
        "tasks; the headline lift in the first table of this section is computed "
        "on each model\u2019s own graded set instead, which is why "
        f"{n_shift} of the {len(pm)} models print 0.01 apart between the two "
        "tables. "
        '<span class="up">\u25b2</span> = CI entirely above the null '
        '(1.0\u00d7 for columns 1\u20134, 0 for the last), <span class="down">'
        "\u25bc</span> = entirely below it (failures spread <i>more evenly</i> "
        "than independence).</p>"
        '<table class="cmdtable"><thead><tr><th>model</th>'
        f'<th class="dim">all {st["n_tasks"]}<div class="bsub">descriptive</div></th>'
        "<th>easy tercile</th><th>middle tercile</th><th>hard tercile</th>"
        f'<th class="key">vs other {len(pm) - 1}'
        '<div class="bsub">the one to quote</div></th>'
        "</tr></thead><tbody>" + trows + "</tbody></table>"
        + f'<p class="h2sub">In the middle tercile {n_up} models have a lift '
        f"significantly above 1.0 and {n_dn} are significantly <i>below</i> it, so "
        f"the whole-corpus excesses are mostly an artifact of grading {len(pm)} "
        "models on "
        "a task set with a wide difficulty spread. Two effects survive. On "
        f"<i>easy</i> tasks {easy_txt} still cluster \u2014 for the Claude models "
        "that is all-or-nothing comprehension (rarely wrong, but wrong about the "
        "whole task); for the two Gemini models it is a far smaller effect sitting "
        "on the field&rsquo;s weakest per-check accuracy, which is not the same "
        "phenomenon. In the middle tercile the OpenAI models spread <i>more evenly "
        "than independence</i> (stable partial coverage, the worst shape under an "
        "all-or-nothing rubric).</p>"
        + grp)


def _pf_prose(stats):
    """The reader-facing account of the episodes the provider killed.

    This paragraph exists because the report makes a defensible but non-obvious
    editorial choice -- keep the dead episodes, scored as zeros, so every model
    is reported over the same task set -- and a choice like that has to be shown
    rather than assumed. Every number in it is derived from
    stats[key]["provider_failures"], so the paragraph shrinks to nothing the day
    a corpus arrives without provider failures, and cannot survive as a stale
    caveat describing a problem that has been fixed upstream.
    """
    pf = {k: (stats[k].get("provider_failures") or {}) for _, _, k in MODELS}
    hit = [(n, k) for n, _, k in MODELS if pf[k].get("n")]
    if not hit:
        return ""
    codes = sorted({c for _, k in hit for c in pf[k].get("codes", {})})
    code_txt = (f"all HTTP {codes[0]}" if len(codes) == 1
                else "HTTP " + _join(codes))
    who = _join([f"<b>{esc(n)}</b> ({pf[k]['n']} of {pf[k]['n'] + pf[k]['ex_n']})"
                 for n, k in hit])
    steps = sum(pf[k]["steps_recovered"] for _, k in hit)
    # The counterfactual is only quotable model by model, so state it for the
    # affected models rather than as a field-level aggregate.
    cf = _join([
        f"{esc(n)} would read {pf[k]['ex_pass_pct']}% rather than "
        f"{stats[k]['pass_pct']}% on pass rate, and {pf[k]['ex_step_mean']} "
        f"rather than {pf[k]['step_mean']} mean steps"
        for n, k in hit])
    return (
        '<p class="h2sub"><b>provider_failure</b> is the one outcome above that '
        "is not a decision the model made \u2014 the provider killed the call "
        f"mid-episode. It costs {who}, {code_txt}, and no other model a single "
        "episode. The harness files these with an empty stop reason and a step "
        "count of zero, which would otherwise sort them alongside the clean "
        "finishes in <code>none</code>; this report gives them their own bucket "
        f"and recovers their step counts from the {steps:,} trajectory actions "
        "the model had already taken before the call died. They are still "
        "scored as the zeros they are and still counted in every denominator, "
        f"so all {_numword(len(MODELS))} models are compared over the same task "
        f"set. Dropped instead, {cf}.</p>")


def render_stops(stats):
    palette = {
        "none": "#10b981",
        "incomplete_required_submission": "#f59e0b",
        "missing_required_side_effect": "#ef4444",
        "reached max_steps": "#8b5cf6",
        "invalid_command_exhausted": "#e11d48",
        "generation_stall_empty_reply": "#0891b2",
        # Slate, and last in the order: this segment is infrastructure, not
        # behaviour. It needs its own entry rather than the unnamed grey
        # fallback, which it would otherwise share with the 3-identical-loop
        # stop and be unreadable in the bar.
        "provider_failure": "#475569",
    }
    order = ["none", "incomplete_required_submission", "missing_required_side_effect",
             "reached max_steps", "invalid_command_exhausted",
             "generation_stall_empty_reply", "provider_failure"]
    rows = []
    for name, _, key in MODELS:
        st = stats[key]
        segs = []
        for s in order:
            segs.append((s, st["stops"].get(s, 0), palette.get(s, "#94a3b8")))
        for s, c in st["stops"].items():
            if s not in palette:
                segs.append((s, c, "#cbd5e1"))
        rows.append(
            f'<div class="aggrow"><span class="nm">{esc(name)}</span>'
            f'{bar(segs, st["n"])}'
            f'<span class="cnt">pass {st["passed"]}/{st["n"]}</span></div>'
        )
    leg = "".join(
        f'<span class="lg"><span class="lgd" style="background:{palette[s]}"></span>{esc(s)}</span>'
        for s in order
    )
    return ("<h2>Outcome / stop-reason mix</h2>"
            f'<div class="legend">{leg}</div>' + "".join(rows)
            + _pf_prose(stats))


def _tok_per_image_groups(stats):
    """Group the fitted tok/image slopes for the prose, checking the claim the
    prose makes about them instead of restating a literal.

    The sentence this feeds used to hard-code "~2.2K on Gemini, ~3.7K on OpenAI
    and ~4.7K on Claude". That enumeration was written when those were the only
    three families with images; kimi-k3 has since joined at ~4.8K -- the LARGEST
    value in the column -- and the literal silently stopped covering its own
    table. Worse, the number it omitted had been rendering as 0 (see
    image_token_stats), so the sentence agreed with a broken cell.

    Returned as (label, value) ascending. The label is the FAMILY name only when
    more than one model in that family carries a slope; with a single model it is
    that model's name. "Open-source: 4,786" would otherwise imply a shared
    tokenizer across deepseek-v4-pro/-flash, kimi-k3 and glm-5.2, three of which
    send no images at all and have no fitted slope.
    """
    fam = {}
    for name, _, key in MODELS:
        v = stats[key]["cost"].get("tok_per_image")
        if v is None:
            continue
        fam.setdefault(PROVIDER[key], []).append((name, v))
    out = []
    for prov in PROVIDER_ORDER:
        got = fam.get(prov)
        if not got:
            continue
        vals = {v for _, v in got}
        if len(vals) != 1:
            # The docstring of image_token_stats claims the slope is a
            # per-family constant and uses that as a consistency check. If it
            # ever is not, the prose must not average it away.
            raise SystemExit(
                f"cost chapter: {prov} models disagree on tokens per image "
                f"({sorted(got, key=lambda p: p[1])}), so they cannot be quoted "
                "as one per-family figure. The per-family-constant assumption in "
                "image_token_stats no longer holds.")
        label = prov if len(got) > 1 else got[0][0]
        out.append((label, vals.pop()))
    return sorted(out, key=lambda p: p[1])


def _tok_per_image_prose(stats):
    """Build the two derived fragments of the tokenizer sentence in the cost note:
    the ascending enumeration, and the sentence that names the top of the column.

    Returns (phrase, top_sentence). Both come from _tok_per_image_groups, so a
    model joining or leaving the field rewrites them instead of leaving a literal
    behind. The enumeration keeps the ~K rounding the sentence has always used
    because it is read as prose; the top sentence quotes EXACT figures, because a
    ranking claim rounded to 0.1K would be unfalsifiable exactly where the margin
    is smallest (4,786 vs 4,740 both display as ~4.7-4.8K).
    """
    groups = _tok_per_image_groups(stats)
    if len(groups) < 2:
        # The sentence is a cross-family contrast; with one slope there is nothing
        # to contrast and the "not comparable across families" claim it sets up
        # would be asserting a spread that no longer exists in the table.
        raise SystemExit(
            "cost chapter: the tokens-per-image sentence compares families, but "
            f"only {len(groups)} family has a fitted slope ({groups}). Rewrite the "
            "sentence rather than shipping a comparison of one.")
    parts = []
    for i, (label, v) in enumerate(groups):
        # "input tokens" rides on the first item only, as the hand-written
        # sentence did; repeating it on all four reads as a stutter.
        unit = " input tokens" if i == 0 else ""
        parts.append(f"<b>~{v / 1000:.1f}K</b>{unit} on {esc(label)}")
    phrase = ", ".join(parts[:-1]) + " and " + parts[-1]
    (lo_label, lo_v) = groups[0]
    (second_label, second_v) = groups[-2]
    (top_label, top_v) = groups[-1]
    top_sentence = (
        f"<b>{esc(top_label)}</b> tokenizes an image most expensively of all "
        f"\u2014 <b>{top_v:,}</b> tok / image against {esc(second_label)}'s "
        f"<b>{second_v:,}</b>, a <b>{top_v / lo_v:.1f}\u00d7</b> spread from "
        f"{esc(lo_label)} at the other end. ")
    return phrase, top_sentence


def render_cost(stats, meta):
    """API cost & token efficiency per task. Tokens are summed from the interaction
    sidecar logs over ALL attempts (retries billed too). All fourteen models carry
    TOKENS from the SAME run as the behavior above -- every model records
    response.usage on 100% of attempts. THIRTEEN of the fourteen also carry DOLLARS;
    deepseek-v4-flash has no verified first-party rate card, so its three dollar
    columns render N/A while its token columns are as measured as anyone's. Keep
    those two words apart in any prose here: "carries token/$" was written when the
    two coincided for all nine models of the original field, and it is now false.
    The output column carries ONE billed figure (everything in it
    bills at the output rate) with the MEASURED reasoning share beneath it: unlike the
    THINK corpus, all three families report the split in v8, so no estimate enters
    this table. Cross-family "visible output" is still not comparable in absolute
    terms -- Anthropic bills ~1.86x OpenAI for an identical tool call (see note) --
    so the share is there to be read down a family, not across one row."""
    # `avail` used to mean both "has tokens" and "has a price", which is how an
    # unpriced model lost its token columns as well. The grand total and the
    # $-ranked bolding are money statements and take the priced set; the token
    # columns below take the wider one.
    avail = [key for _, _, key in MODELS
             if stats[key]["cost"].get("has_price")
             and stats[key]["cost"].get("has_tokens")]
    solved = [stats[k]["cost"]["cost_per_solved"] for k in avail
              if stats[k]["cost"].get("cost_per_solved") is not None]
    best_solved = min(solved) if solved else None
    durs = [stats[k]["duration_per_task"] for _, _, k in MODELS
            if stats[k]["duration_per_task"] is not None]
    best_dur = min(durs) if durs else None

    # Grand total across the priced models -- the whole eval's token/$ footprint.
    grand_tok = sum(stats[k]["cost"]["tok_total"] for k in avail)
    grand_cost = sum(stats[k]["cost"]["cost_total"] for k in avail)
    head = ("<tr><th>model</th><th>input<br>tok/task</th>"
            "<th>imgs @<br>step 0</th><th>tok /<br>image (s0)</th>"
            '<th title="All billed output tokens, including internal reasoning: '
            'every provider bills reasoning at the output rate. Gemini reports '
            '`candidates` exclusive of `thoughts`, so the two are added; OpenAI and '
            'Anthropic already include reasoning in `output_tokens`. The sub-line is '
            'the measured reasoning share of that total, where the provider reports '
            'one: the three open-source models are served through an '
            'OpenAI-compatible endpoint that returns no reasoning split, so they '
            'read "split not reported" rather than 0%. Nothing in this column is '
            'estimated.">output<br>tok/task</th>'
            "<th>total<br>tok/task</th><th>$ / task</th>"
            "<th>total tok<br>(run, 600)</th><th>$ total<br>(run, 600)</th>"
            "<th>$ / solved<br>task</th><th>time per<br>task (s)</th></tr>")
    rows = []
    for name, _, key in MODELS:
        c = stats[key]["cost"]
        dur = stats[key]["duration_per_task"]
        durcell = "\u2014" if dur is None else f"{dur:,.1f}"
        if best_dur is not None and dur == best_dur:
            durcell = f"<b>{durcell}</b>"
        na = f'<td class="sc" title="{esc(c.get("note") or "")}">N/A</td>'
        if not c.get("has_tokens"):
            # Nothing was measured for this row: all nine data cells are N/A and
            # the tooltip says the logs are empty.
            rows.append(f'<tr><td class="ck">{esc(name)}</td>' + na * 9
                        + f'<td class="sc">{durcell}</td></tr>')
            continue
        # Tokens measured, no rate card: only the three dollar cells are N/A.
        # Blanking the other six as well -- which is what a single `available`
        # flag did -- tells the reader the run recorded nothing, which is the
        # opposite of true.
        unpriced = not c.get("has_price")
        cps = c.get("cost_per_solved")
        cpscell = "\u2014" if cps is None else f"${cps:,.3f}"
        if best_solved is not None and cps == best_solved:
            cpscell = f"<b>{cpscell}</b>"
        imgs0 = c.get("imgs_step0")
        tpi = c.get("tok_per_image")
        imgs0cell = "\u2014" if imgs0 is None else f"{imgs0:,.1f}"
        tpicell = "\u2014" if tpi is None else f"{tpi:,}"
        # One billed output figure -- everything in it bills at the output rate, so
        # this is the number that costs money -- with the measured reasoning share
        # underneath. The three proprietary families populate the split in v8 (Gemini
        # `thoughts`, OpenAI `reasoning_tokens`, Anthropic
        # `output_tokens_details.thinking_tokens`).
        #
        # THE OPEN-SOURCE MODELS DO NOT, AND MUST NOT BE PRINTED AS "0% reasoning".
        # They are served through an OpenAI-compatible endpoint whose usage object is
        # the older Chat-Completions shape: no `reasoning_tokens`, no
        # `completion_tokens_details`, nothing. summarize_cost already distinguishes
        # this -- reason_measured=False -- and the comment at PRICING says in as many
        # words "do not render their 0 reasoning tokens as a measured 0% share". This
        # renderer did exactly that anyway, shipping "0% reasoning" for
        # deepseek-v4-pro, kimi-k3 and glm-5.2. It reads as "these models did not
        # think", which is false: all of them ran at reasoning_effort=high (see
        # OPEN_SOURCE_CAVEAT), and the split is simply unreported. Print a dash.
        # This now covers FOUR models, not the three named above: deepseek-v4-flash
        # is on the same shim and also carries reason_measured=False. The branch is
        # driven by the flag rather than by a model list, so it picked v4-flash up
        # with no edit -- keep it that way.
        _out_tot = c["vis_out_per_task"] + c["reason_per_task"]
        _measured = c.get("reason_measured", True)
        _share = (100.0 * c["reason_per_task"] / _out_tot) if _out_tot else None
        outcell = f'{_out_tot:,}'
        if not _measured:
            outcell += ('<div class="bsub" title="This provider returns no '
                        'reasoning/thinking split in its usage object, so the share '
                        'is UNKNOWN, not zero. The model still reasoned \u2014 it ran '
                        'at reasoning_effort=high \u2014 and that reasoning is billed '
                        'inside the output tokens shown above, so the dollar figures '
                        'are unaffected.">split not reported</div>')
        elif _share is not None:
            outcell += (f'<div class="bsub" title="Measured internal reasoning as a '
                        f'share of billed output tokens; billed at the output rate '
                        f'like the rest.">{_share:.0f}% reasoning</div>')
        # The three money cells, blank exactly when there is no verified rate
        # card. Formatted from values that are None in that case, so a renderer
        # change that drops this branch raises instead of printing $0.000.
        cpt = na if unpriced else f'<td class="sc">${c["cost_per_task"]:,.3f}</td>'
        ctot = na if unpriced else (f'<td class="sc hc"><b>'
                                    f'${c["cost_total"]:,.2f}</b></td>')
        csol = na if unpriced else f'<td class="sc">{cpscell}</td>'
        rows.append(
            f'<tr><td class="ck">{esc(name)}</td>'
            f'<td class="sc">{c["in_per_task"]:,}</td>'
            f'<td class="sc">{imgs0cell}</td>'
            f'<td class="sc">{tpicell}</td>'
            f'<td class="sc">{outcell}</td>'
            f'<td class="sc hc"><b>{c["total_per_task"]:,}</b></td>'
            f'{cpt}'
            f'<td class="sc">{c["tok_total"]:,}</td>'
            f'{ctot}'
            f'{csol}'
            f'<td class="sc">{durcell}</td></tr>')
    _tpi_phrase, _tpi_top = _tok_per_image_prose(stats)
    note = (
        f'<p class="tblnote"><b>Run grand total</b> across all {len(avail)} priced models: '
        f'{grand_tok:,} tokens \u2248 <b>${grand_cost:,.2f}</b> for all '
        f'{len(avail) * meta["n_tasks"]:,} model-tasks. '
        'API list prices verified against the first-party pricing pages (per 1M '
        'tokens): 2026-08-05 for the nine models of the original field (OpenAI, '
        'Google Vertex, Anthropic), and 2026-08-14 for the four added since '
        '\u2014 gemini-3.7-flash, deepseek-v4-pro, kimi-k3 and glm-5.2 \u2014 each '
        'against its own vendor page, and each priced at the rate in force on its '
        'own run date. '
        + _unpriced_note(stats) +
        '<b>gpt-5.6-terra and gpt-5.6-luna were repriced on 2026-07-30</b> \u2014 Luna '
        '\u221280%, Terra \u221220%, Sol unchanged \u2014 and this run, executed 2026-08-02 to '
        '08-03, was billed at the reduced rates; the figures here use them. '
        'Tokens are summed over <b>every attempt</b> (retries are billed too). '
        '<b>Input dominates</b> total tokens because each task replays a growing '
        'multi-turn context across dozens of calls, and the asset screenshots weigh '
        'heavily: tasks send <b>13.3 screenshots</b> on average at step 0 and the full '
        'history replays them on every call. Each family tokenizes the same '
        f'3840\u00d72160 PNG differently \u2014 {_tpi_phrase} (the <i>tok / image</i> '
        'column, fitted per family on this run) \u2014 so raw input-token counts are '
        f'<b>not comparable across families</b> (dollar cost still is). {_tpi_top}'
        'Image tokens are '
        'billed at the <b>standard input rate</b>, already inside the input-token counts '
        'here, not a separate line item. <b>Cache:</b> OpenAI cache-read and cache-write '
        'tokens are billed at their own rates from the real per-call counts \u2014 '
        '5.6-sol/luna/terra cache heavily (29.6M / 24.9M / 22.4M cache-read tokens over '
        'this run) while Claude records <b>zero</b> cache tokens and Gemini logs expose no '
        'cache-read field, so no discount is applied to either. <b>Long-context tiers do '
        'trigger here</b>, barely: the three GPT models each push <b>3 calls</b> past their '
        '128K threshold (largest single call 128,767 / 128,595 / 128,488), which is within '
        '0.6% of the boundary and looks like the harness context clamp landing on the same '
        'number rather than a real long-context workload. Those calls are priced at the '
        'long rate, worth <b>+0.10%</b> on the GPT run totals. Gemini 3.1-pro never '
        'approaches its 200K threshold (largest call 85,272). Gemini 3.5-flash reaches '
        '389,940 and Claude 171\u2013182K on their largest calls, but neither family has a '
        'long tier defined, so both bill flat throughout. '
        '<b>The reasoning share is measured, not estimated:</b> token/$ come from the '
        '<b>same run</b> as the behavior above (no re-run needed \u2014 every model records '
        'usage on 100% of attempts). Every provider bills reasoning at the output rate, and '
        'in this corpus <b>all three families report the split</b>: Gemini as '
        '<code>thoughts</code> (added here, since its <code>candidates</code> count excludes '
        'them), OpenAI as <code>reasoning_tokens</code> and Anthropic as '
        '<code>output_tokens_details.thinking_tokens</code> \u2014 the last of these was '
        'unrecorded in the older THINK tree and is populated here, so nothing in this table '
        'is reconstructed. Read the share <b>down a family, not across a row</b>: it says '
        'how much of a model\u2019s billed output went to private reasoning rather than to '
        'the tool call. <b>opus-5 is the outlier</b> \u2014 4.6% against 27.3% for opus-4.8 '
        'and 20.7% for sonnet-5 \u2014 the most accurate model here is also the one that '
        'thinks least per token emitted, which is the same efficiency its low step counts '
        'show. Gemini sits at the other end (68\u201390%), spending most of its output '
        'budget on <code>thoughts</code>. '
        '<b>Why Claude\u2019s output is so much larger \u2014 two measured causes, and neither is '
        'mostly thinking:</b> Claude emits <b>2.50\u00d7</b> the billed output per task that '
        'the GPT models do (4,921 vs 1,970, family means). Two independently measured '
        'effects push in that direction. (1) <b>Anthropic bills ~1.86\u00d7 what OpenAI does '
        'for an identical tool call</b>: restricting to the 4 payload strings both families '
        'emitted at least 20 times each, the minimum billed output is a median of '
        '<b>27 tokens for Claude vs 14.5 for GPT</b> (<code>asset_list({})</code>: 27 vs 14 '
        'over 1,771/1,809 calls; <code>done({})</code>: 24 vs 13 over 1,377/1,398). (2) '
        '<b>Claude emits ~1.6\u00d7 longer calls</b> \u2014 684 / 474 / 730 chars per call for '
        'opus-5 / opus-4.8 / sonnet-5 against 356\u2013420 for GPT and 140\u2013311 for Gemini. '
        'Those two do not compose cleanly (their product, 2.95\u00d7, overshoots the measured '
        '2.50\u00d7, since the per-call floor is amortised over longer calls), so treat them '
        'as two contributing causes rather than a factorisation. The practical consequence '
        'holds either way: <b>Claude is expensive not only because its per-token rate is '
        'high, but because the same action costs it more tokens.</b> '
        'Sonnet is billed here at Anthropic\u2019s '
        'introductory $2/$10 per-1M rate (valid through 2026-08-31); at the standard '
        '$3/$15 its run cost rises to ~$1,254 (~$12.30 per solved task). '
        'OpenAI models ran on Amazon Bedrock (billed via AWS, which "may differ from '
        'direct OpenAI pricing"); direct list rates are used as the best proxy. AWS '
        'describes the 2026-07-30 Bedrock cut as "in-line with OpenAI\u2019s first-party '
        'pricing changes for these models", so the proxy tracks the change on both sides. '
        '<b>$ / solved task</b> = run cost \u00f7 tasks solved (the cost of one success); '
        'lowest value in each of the last two columns is bolded.</p>')
    return (
        "<h2>API cost &amp; token efficiency</h2>"
        f'<div class="h2sub">Per-task token spend and dollar cost for all '
        f'{_numword(len(MODELS))} models, plus time per task. This is the '
        'price of the behavior shown above \u2014 read it against pass rate, not alone.</div>'
        f'<table class="cmdtable costtbl"><thead>{head}</thead><tbody>'
        + "".join(rows) + "</tbody></table>" + note)


def render_cost_vs_pass(stats):
    """Cost/benefit scatter: total run $ (x) x pass@budget (y). Answers the host's
    question directly -- what did each model's accuracy actually cost? The upper-left
    is the value frontier (high pass rate for little money). Every PRICED model is
    plotted, and its cost comes from the same run as its behavior, so no re-run
    substitution is needed. A model with measured tokens but no verified rate
    card has no x coordinate and is correctly absent -- gating on `has_price`
    rather than on the old combined flag keeps it out of this chart while
    letting it keep its token columns in the table above."""
    priced = [(name, key) for name, _, key in MODELS
              if stats[key]["cost"].get("has_price")
              and stats[key]["cost"].get("has_tokens")]
    points = []
    for name, key in priced:
        x = stats[key]["cost"]["cost_total"]
        y = stats[key]["pass_pct"]
        points.append({"name": name, "key": key, "x": x, "y": y,
                       "coord": f"(${x:,.0f}, {y:.1f}%)"})
    # frontier = models not dominated by any other (nobody is both cheaper AND higher
    # pass). Named in the caption so the "best buys" are called out explicitly.
    front = [p for p in points
             if not any(q["x"] < p["x"] and q["y"] >= p["y"] and q is not p
                        for q in points)]
    front_names = ", ".join(p["name"] for p in sorted(front, key=lambda p: p["x"]))
    corners = {"tl": "high pass, low spend (best value)",
               "tr": "high pass, high spend",
               "bl": "low pass, low spend",
               "br": "low pass, high spend (worst value)"}
    # Both axes are floored: spend cannot be negative and neither can a pass rate.
    # Without the floors this panel opened at -$344 on x and "-0%" on y.
    svg = _scatter_svg(points, "total run cost (USD, 600 tasks)", "pass rate %",
                       corners, lambda v: f"${v:,.0f}", lambda v: f"{v:.0f}%",
                       "pass rate vs total run cost scatter",
                       xfloor=0.0, yfloor=0.0)
    return (
        "<h2>Cost \u00d7 pass-rate map</h2>"
        '<p class="h2sub">x = total API cost for the whole 600-task run (list price, '
        "every attempt billed) \u00b7 y = pass@budget. One dot per <b>priced</b> model "
        "(colors keyed below), with its (cost, pass%) beside it. Axes are zoomed to "
        "the data range, so position is <b>relative</b>; dashed lines mark each "
        "axis mid-point. <b>Up and to the left is better</b> \u2014 more accuracy per "
        "dollar. The value frontier (no other model is both cheaper and at least as "
        f"accurate) is <b>{esc(front_names)}</b>.</p>"
        + svg)


def _gold_cell(row):
    """One (band, model) cell: pass rate, effort when passing, capped rate.

    The three numbers sit on three different populations and the cell says so:
    the rate is over every run in the band, the effort over that band's UNCAPPED
    PASSES only, and the capped rate over every run again. They are not three
    views of one population and must not be read as if they were.
    """
    if not row or not row["n"]:
        return '<td class="sc">\u2014</td>'
    bg = gold_shade(row["pass_pct"])
    ms, ratio = row["med_steps_pass"], row["ratio_pass"]
    npass = row["n_pass_uncapped"]
    if ms is None or ratio is None:
        eff = '<span class="ge none">no uncapped pass</span>'
    else:
        cls = "ge thin" if row["thin"] else "ge"
        tail = f' <span class="gn">n{npass}</span>' if row["thin"] else ""
        eff = (f'<span class="{cls}">{_fmtnum(ms)} st \u00b7 '
               f'{ratio:.2f}\u00d7{tail}</span>')
    cap = row["capped_pct"]
    ccls = "gc hi" if cap is not None and cap >= 33 else "gc"
    caps = (f'<span class="{ccls}">{cap:.0f}% capped</span>'
            if cap is not None else "")
    return (f'<td class="sc" style="background:{bg}">'
            f'<b>{row["pass_pct"]:.0f}%</b>{eff}{caps}</td>')


def render_gold_distribution(summary):
    """Overshoot distribution: steps minus that task's own gold, per model.

    This is the figure the band table cannot show. A single median per cell
    describes a model with one mode; several models here have two. The axis is
    already a difference (actual minus expert), so no statistical null is needed
    -- the gold trajectory IS the reference. Capped runs get their own terminal
    column because their true length is unknown and binning them at 20 would
    assert a measurement that was never taken.

    Rows are sorted by capped share so the two regimes separate visually:
    models that finish near the expert's budget at the top, models that mostly
    run out the clock at the bottom.
    """
    oh = (summary or {}).get("over_hist")
    if not oh:
        return ""
    labels = [lab for lab, _lo, _hi in OVER_BUCKETS] + ["hit the cap"]
    subs = ["at or under<br>the expert", "", "", "",
            "far past it<br>but finished", f"stopped at<br>{STEP_CAP} steps"]
    names = {k: n for n, _s, k in MODELS}
    order = sorted((k for _, _, k in MODELS),
                   key=lambda k: oh[k]["pct"][-1])
    head = ('<tr><th>model</th>'
            + "".join(f'<th>{lab}<div class="bsub">{sub}</div></th>'
                      for lab, sub in zip(labels, subs))
            + "</tr>")
    body = []
    for k in order:
        cells = []
        for j, v in enumerate(oh[k]["pct"]):
            cap = j == len(labels) - 1
            cls = "ohn cap" if cap else "ohn"
            bcl = "cap" if cap else ""
            cells.append(f'<td class="oh"><span class="{cls}">{v:.1f}</span>'
                         f'<span class="ohb"><i class="{bcl}" '
                         f'style="width:{v * 0.72:.1f}px"></i></span></td>')
        body.append(f'<tr><td class="ck">{esc(names[k])}</td>'
                    + "".join(cells) + "</tr>")
    return (
        '<p class="h2sub"><b>How far past the expert each run actually went.</b> '
        "One row per model, over every gold-covered run. A run is placed by "
        "<b>its own task\u2019s</b> gold length \u2014 steps minus that gold \u2014 so a "
        "run that took 11 steps on a 9-step task and one that took 8 on a 6-step "
        "task both land in <b>+1..2</b>. Runs stopped by the "
        f"{STEP_CAP}-step cap are held out in the last column instead of being "
        "binned at 20: their true length was never observed, and pretending it was "
        "20 is what made the old overshoot column readable as a measurement. Rows "
        "are ordered by that last column. Cells are percentages of the model\u2019s "
        "own runs and each row sums to 100; bars share one scale.</p>"
        '<div class="legend">'
        '<span class="lg"><span class="lgd" style="background:#4f46e5"></span>'
        "finished on its own</span>"
        '<span class="lg"><span class="lgd" style="background:#dc2626"></span>'
        "stopped by the cap \u2014 true length never observed</span>"
        '<span class="lg">each cell: <b>% of that model\u2019s runs</b> '
        "\u00b7 bar length</span></div>"
        f'<table class="cmdtable oht"><thead>{head}</thead>'
        f'<tbody>{"".join(body)}</tbody></table>')


def render_steps_vs_gold(summary, meta):
    """Effort-vs-gold: pass rate, effort when passing, and capped share."""
    if not summary or not summary.get("covered"):
        return ""
    bands_meta = {b["band"]: b for b in summary["bands"]}
    band_order = [lab for lab, _ in GOLD_BANDS] + [OVERALL_BAND]
    head = ('<tr><th>gold band'
            '<div class="bsub">tasks \u00b7 gold median \u00b7 range</div></th>'
            + "".join(f"<th>{esc(MODEL_LABEL[k])}</th>" for _, _, k in MODELS)
            + "</tr>")
    body = []
    for lab in band_order:
        bm = bands_meta.get(lab, {})
        lo, hi = bm.get("gold_min"), bm.get("gold_max")
        rng = (f' ({_fmtnum(lo)}\u2013{_fmtnum(hi)})'
               if lo is not None and hi is not None and lo != hi else "")
        first = (f'<td class="ck"><b>{esc(lab)}</b><br>'
                 f'<span style="color:#5b6472">{bm.get("n_gold", 0)} tasks \u00b7 '
                 f'gold {_fmtnum(bm.get("gold_median"))}{rng}</span></td>')
        cells = "".join(
            _gold_cell(next((r for r in summary["by_model"][key]
                             if r["band"] == lab), None))
            for _, _, key in MODELS)
        body.append(f'<tr>{first}{cells}</tr>')
    covered = summary["covered"]
    missing = meta["n_tasks"] - covered
    # Pass rate is not monotone in gold length, so the band axis must not be
    # sold as a difficulty axis. Counted from the rendered numbers rather than
    # asserted, so the sentence cannot drift away from the table.
    bm_ = summary["by_model"]

    def _p(key, lab):
        r = next((x for x in bm_[key] if x["band"] == lab), None)
        return r["pass_pct"] if r else None

    s_lab, m_lab = GOLD_BANDS[0][0], GOLD_BANDS[1][0]
    n_mid_higher = sum(1 for _, _, k in MODELS
                       if (_p(k, m_lab) or 0) > (_p(k, s_lab) or 0))
    return (
        "<h2>Steps vs. gold trajectory \u2014 steps / effort vs. expert "
        "solution length</h2>"
        '<p class="h2sub">Every task ships an expert <i>gold trajectory</i> in its '
        "<code>specs.json</code>; its length (commands, excluding the terminal "
        "<code>DONE</code>) is the reference step budget. Tasks are grouped into "
        "three length bands, then an <b>all tasks (overall)</b> row pools every "
        "gold-covered task. Each cell carries three numbers on three different "
        "populations, and they are not interchangeable: <b>pass rate</b> "
        f"(the indigo tint, deepening from 0 to {GOLD_SHADE_MAX:.0f}% \u2014 a "
        "narrower scale than the 0\u2013100% heatmap earlier in this report, "
        "because nothing here clears 40%) over every run in the band; "
        "<b>median steps and "
        "overshoot when it passes</b>, over that band\u2019s uncapped passes only, "
        "dividing by each task\u2019s own gold rather than the band median; and "
        f"<b>% capped</b>, the share of runs stopped by the {STEP_CAP}-step "
        "limit.</p>"
        '<p class="h2sub warn">Why effort is not reported over all runs. A capped '
        "run\u2019s step count is the cap, not a measurement, and a failed run\u2019s "
        "mostly measures how long the model flailed. Pooling them puts the summary "
        "between the modes: gemini-3.1-pro averages out to 17 steps, but it takes "
        "9 when it succeeds and 20 when it fails, and almost nothing in between. "
        "The distribution below shows that directly. Effort numbers resting on "
        f"fewer than {THIN_PASSES} passes are italicised and carry their count "
        "(<i>n</i>) \u2014 "
        "every cell in the <b>long</b> row is in that state, so read that row for "
        "its pass rate and its capped share, not for its medians.</p>"
        f'<table class="cmdtable gold"><thead>{head}</thead>'
        f'<tbody>{"".join(body)}</tbody></table>'
        f'<p class="tblnote">Gold trajectories cover {covered} of the '
        f'{meta["n_tasks"]} tasks per model. The {missing} excluded here are three '
        "whole <code>it_deployment_gate</code> scenarios (v01/v02/v03 \u00d7 10 "
        "rounds) whose <code>specs.json</code> is absent from this tree, so there "
        "is no reference budget to divide by. Those runs were still scored by the "
        "environment (pass/fail comes from recorded check results, never from the "
        "gold trajectory) and <b>remain in every other pass-rate figure in this "
        f"report</b>; only this effort-vs-gold view drops them. The bands order "
        f"tasks by intrinsic solution length, not by difficulty: {n_mid_higher} of "
        f"the {len(MODELS)} models pass <b>more</b> of the mid band than the short "
        "band, and the two bands\u2019 gold medians differ by a single step (8 vs 9), "
        "so treat the top two rows as similar-length tasks rather than as an easy "
        "and a harder tier.</p>"
        + render_gold_distribution(summary))

GOLD_SHADE_MAX = 40.0  # pass-rate (%) that saturates the gold-table tint


def gold_shade(pct):
    """Light sequential tint for one gold-band cell, keyed to pass rate.

    Deliberately NOT score_color, for two reasons that were both visible in the
    rendered page before this was split out:

    1. Domain. score_color spans 0-100 because it shades a 0..1 mean score. Pass
       rates in this table span 1-37%, so every cell landed in the red->amber
       third of that ramp and the shading separated almost nothing. This ramp is
       clamped to 0-GOLD_SHADE_MAX so the observed spread uses its full range.
    2. Hue. Rescaling score_color instead would have painted the best cell (37%)
       green, i.e. asserted "good" about a 37% pass rate. A single-hue sequential
       ramp says "more is more" and makes no such claim. It is the same indigo as
       the overshoot-distribution bars directly below, which ties the two views
       together.

    The tint tops out light on purpose: the cell carries three lines of small
    text (effort, capped share) whose greys and reds were unreadable on the old
    saturated fill. Keep the alpha low if you touch this.
    """
    t = max(0.0, min(1.0, (pct or 0.0) / GOLD_SHADE_MAX))
    a = 0.06 + 0.36 * t  # blend fraction of #4f46e5 over white
    r = round(255 + (79 - 255) * a)
    g = round(255 + (70 - 255) * a)
    b = round(255 + (229 - 255) * a)
    return f"rgb({r},{g},{b})"


def score_color(v):
    """Red (0) -> amber (0.5) -> green (1.0)."""
    if v <= 0.5:
        r, g = 239, int(68 + (185 - 68) * (v / 0.5))
        b = 68
    else:
        r = int(245 - (245 - 16) * ((v - 0.5) / 0.5))
        g = int(158 + (185 - 158) * ((v - 0.5) / 0.5))
        b = int(11 + (129 - 11) * ((v - 0.5) / 0.5))
    return f"rgb({r},{g},{b})"


def render_all27(stats, meta):
    # Average score per scenario per model across its rounds.
    agg = defaultdict(lambda: defaultdict(list))
    for _, _, key in MODELS:
        for tid, t in stats[key]["by_task"].items():
            agg[scenario_of(tid)][key].append(t["score"])
    rows = []
    for scen, md in agg.items():
        allv = [v for _, _, key in MODELS for v in md.get(key, [])]
        mean_all = sum(allv) / max(len(allv), 1)
        per = {key: (sum(md[key]) / len(md[key]) if md.get(key) else 0.0)
               for _, _, key in MODELS}
        rows.append((mean_all, scen, per))
    rows.sort(key=lambda x: x[0])  # hardest -> easiest
    body = []
    for mean_all, scen, per in rows:
        curated = " \u2605" if scen in CURATED_SCENARIOS else ""
        cells = [f'<td class="sc" style="background:{score_color(mean_all)}">{mean_all:.2f}</td>']
        for _, _, key in MODELS:
            v = per[key]
            cells.append(f'<td class="sc" style="background:{score_color(v)}">{v:.2f}</td>')
        body.append(f'<tr><td class="scn">{esc(scen)}{curated}</td>{"".join(cells)}</tr>')
    head = ("<tr><th>scenario (hardest \u2192 easiest)</th><th>ALL</th>"
            + "".join(f"<th>{esc(MODEL_LABEL[k])}</th>" for _, _, k in MODELS) + "</tr>")
    return (f"<h2>All-{meta['n_scenarios']}-scenario score heatmap</h2>"
            f'<p class="h2sub">Mean score per scenario ({meta["n_rounds"]} rounds each), '
            'sorted hardest '
            "to easiest by cross-model mean. \u2605 = walked through step-by-step below. "
            "Reconciles with eval-results-capability-analysis.md Appendix A.</p>"
            f'<table class="h27"><thead>{head}</thead><tbody>{"".join(body)}</tbody></table>')


# --- designed difficulty axes -------------------------------------------------
# The heatmap above answers "which scenarios are hard". This section answers the
# question the benchmark was written to ask: "which DESIGNED lever is hard". The
# per-task eval metadata varies only on round and scenario slug, so the design
# intent is not in the eval tree at all -- it lives in the 20 scenario seeds under
# clasharness/data/scenario_seeds/conflict_vlm_v1/. That is why this section
# needs a SECOND root (--seeds) pointed at the source checkout: `data_3_6_flash_v8`
# and the seeds do not share a parent directory.
#
# WHAT A LEVER IS. A seed names its conflict in two fields, and the generator
# concatenates them into ONE token list before routing (conflict_llm_pipeline.py
# :1582): `conflict_taxonomy` + `difficulty_challenges`. Both halves resolve
# through the same plugin registry, so a "difficulty challenge" is not a separate
# kind of thing from a conflict mechanic -- it is a plugin like any other, and all
# 23 registered plugins are live somewhere in this corpus. The groups below are
# therefore split for READABILITY, not because the router treats them
# differently: the 7 challenge tokens all only inject prompt text, whereas 13 of
# the other 16 additionally declare `emits_effects` and gate real command
# execution. The split is NOT prompt-versus-mechanic -- `credential_lock`,
# `evidence_mismatch` and `temporal_staleness` sit on the mechanics side and are
# prompt-only too. Capability-doc 5.3 reports the mechanic-depth contrast, which
# is null, as prose; it is not an axis here.
# `archetype` is NOT a lever: it never reaches the
# router. The archetype map that used to bucket this section has been removed;
# the reason it disagrees with the plugin specs is a footnote in capability-doc
# 5.3, which is the only place it still appears.
#
# THREE INDEPENDENT DERIVATIONS. tools/design_axis_slice.py re-derives these same
# numbers straight from the raw jsonl, and the SF11 block of the per-model
# deep-dives re-derives them from failure_taxonomy.json. Audit pass D14 asserts
# all three agree on every integer. The constants below are therefore DUPLICATED
# on purpose -- design_axis_slice.py *imports* the live plugin registry, this file
# hand-copies it, and audit pass D13 re-derives it a third way by parsing each
# plugin module's `SPEC = ConflictPluginSpec(...)` block. Sharing one import would
# collapse two of the three chains and the cross-check would stop proving
# anything, so the duplication is checked rather than trusted.
#
# PASS RATES ONLY, DELIBERATELY. `by_task[tid]["score"]` is rounded to 2 dp when
# it is loaded, so a mean score derived from it would disagree in the third
# decimal with the other two chains and a reader would have no way to tell a
# rounding artifact from a real difference. `success` is a bool and survives
# exactly. The mean-score view of these same buckets lives in
# eval-results-capability-analysis.md 5.3.
SEED_SUBDIR = "clasharness/data/scenario_seeds/conflict_vlm_v1"
N_SEEDS = 20

# plugin -> ConflictPluginSpec.category, one entry per registered conflict
# plugin. Hand-copied from the 23 modules under clasharness/scenario/plugins/;
# see the THREE INDEPENDENT DERIVATIONS note above for why this is a literal and
# not an import. Token names are 1:1 with plugin names across all 23 (no plugin
# declares taxonomy_tokens), so routing is just "keep the known names" and is
# independent of the order the seed happens to list them in.
PLUGIN_CATEGORY = {
    # inconsistency
    "evidence_mismatch": "inconsistency",
    "temporal_staleness": "inconsistency",
    # coordination
    "barrier_synchronization": "coordination",
    "credential_lock": "coordination",
    "deadlock_dependency": "coordination",
    "distributed_lock": "coordination",
    # order
    "controlled_deployment_gate": "order",
    "dependency_ordering": "order",
    # timeout
    "multi_device_pairing": "timeout",
    "reversible_action_hold": "timeout",
    "time_window_expiration": "timeout",
    # subagent_collaboration
    "coordinated_containment": "subagent_collaboration",
    "duplicate_hold_stale": "subagent_collaboration",
    "multi_party_synchronization": "subagent_collaboration",
    "non_overlapping_assignment": "subagent_collaboration",
    # main_supervised
    "scope_reconciliation": "main_supervised",
    # challenge -- the difficulty_challenges half of the token list
    "duplicate_avoidance": "challenge",
    "gap_completion": "challenge",
    "information_transfer": "challenge",
    "multi_source_reasoning": "challenge",
    "state_repair": "challenge",
    "vision_decoy": "challenge",
    "workflow_completion": "challenge",
}

# Render order for the two plugin groups, widest coverage first, which is the
# order capability-doc 5.3 prints the same buckets in. Order is cosmetic -- the
# audits key on bucket NAME -- but keeping the two documents in step matters more
# than the few bytes. Together these must partition PLUGIN_CATEGORY; _check_plugin_lists
# enforces that, because a name dropped from one list would silently vanish from
# the table rather than error.
CHALLENGE_TOKENS = [
    "workflow_completion",
    "information_transfer",
    "multi_source_reasoning",
    "state_repair",
    "gap_completion",
    "duplicate_avoidance",
    "vision_decoy",
]

MECHANIC_ORDER = [
    "evidence_mismatch",
    "temporal_staleness",
    "dependency_ordering",
    "time_window_expiration",
    "credential_lock",
    "deadlock_dependency",
    "barrier_synchronization",
    "controlled_deployment_gate",
    "coordinated_containment",
    "distributed_lock",
    "duplicate_hold_stale",
    "multi_device_pairing",
    "multi_party_synchronization",
    "non_overlapping_assignment",
    "reversible_action_hold",
    "scope_reconciliation",
]


def _check_plugin_lists():
    """The two render lists must cover PLUGIN_CATEGORY exactly, once each."""
    listed = MECHANIC_ORDER + CHALLENGE_TOKENS
    dupes = sorted({n for n in listed if listed.count(n) > 1})
    missing = sorted(set(PLUGIN_CATEGORY) - set(listed))
    unknown = sorted(set(listed) - set(PLUGIN_CATEGORY))
    miscat = sorted(n for n in CHALLENGE_TOKENS
                    if PLUGIN_CATEGORY.get(n) != "challenge")
    miscat += sorted(n for n in MECHANIC_ORDER
                     if PLUGIN_CATEGORY.get(n) == "challenge")
    if dupes or missing or unknown or miscat:
        raise SystemExit(
            "designed-difficulty plugin lists are inconsistent: "
            f"duplicated={dupes} missing={missing} unknown={unknown} "
            f"wrong_group={miscat}")


_check_plugin_lists()

# Render order of the conflict-category axis. NOT to be confused with CAT_ORDER
# above, which orders the ACTION categories of the behavior mix -- these are the
# category families the plugin specs declare. `challenge` is not among them: all
# 20 seeds carry challenge plugins, so that bucket is the whole corpus and
# discriminates nothing. There is no "other" bucket either -- every seed routes
# to at least one categorised plugin.
CONFLICT_CAT_ORDER = ["inconsistency", "coordination", "order", "timeout",
                      "subagent_collaboration", "main_supervised"]

# (group heading, axis, [(bucket, is_lever)]). Headings are SHORT plain words on
# purpose: .famrow uppercases its text, which mangles a seed field name into
# FORCE_ASYNC_WAIT, and .cmdtable right-aligns it, which reads badly past a few
# words. The field names each axis comes from are named in the table footnote
# instead, where they can stay lowercase and monospaced.
#
# is_lever=False marks a CONTROL arm -- the complement of a knob rather than the
# knob itself. Both arms are rendered, because the control is what makes the
# lever's delta readable, but only levers are counted in the generated summary:
# `no_async` sits above baseline exactly when `async_required` sits below it, so
# counting both would report one fact twice.
#
# The conflict-category axis is NOT rendered here. A category is a grouping of
# levers, not a lever, so a category delta has no single design decision behind
# it -- and on this corpus three of the six categories are carried by exactly one
# plugin each, so those rows would restate a plugin row under a grander name. It
# is still carried in the stats sidecar (and in capability-doc 5.3, where that is
# spelled out with the numbers). Note the buckets OVERLAP: a seed routing to a
# coordination plugin and an order plugin is counted in both, so unlike the old
# archetype partition they do not sum to the corpus.
#
# Group order: the two binary knobs first, each with its control arm, then the
# two overlapping token groups. Mechanics precede challenges because
# `conflict_taxonomy` is the primary routing field and `difficulty_challenges`
# modifies it.
DESIGN_GROUPS = [
    ("temporal async", "async",
     [("async_required", True), ("no_async", False)]),
    ("vision locus", "vision",
     [("vision_graded", True), ("not_vision_graded", False)]),
    ("conflict mechanics (overlapping)", "plugin",
     [(t, True) for t in MECHANIC_ORDER]),
    ("difficulty challenges (overlapping)", "challenge",
     [(t, True) for t in CHALLENGE_TOKENS]),
]

# Which seed field each axis is read off. Rendered in the footnote rather than in
# the group heading; see the DESIGN_GROUPS comment. The plugin and category axes
# both come from the concatenated token list, so they name both halves.
AXIS_FIELD = {
    "async": "force_async_wait",
    "vision": "force_vision_locus",
    "plugin": "conflict_taxonomy",
    "challenge": "difficulty_challenges",
    "category": "conflict_taxonomy",
}


def load_seeds(seed_dir):
    """base seed slug -> seed dict.

    Hard-fails on an empty directory instead of rendering an empty axis: the
    seeds live in a different checkout from --root, and a silently missing
    section is far worse than a stopped build.
    """
    seeds = {}
    for f in sorted(glob.glob(os.path.join(seed_dir, "*.json"))):
        with open(f, encoding="utf-8") as fh:
            seeds[os.path.basename(f)[:-len(".json")]] = json.load(fh)
    if not seeds:
        raise SystemExit(
            f"no scenario seeds under {seed_dir}\n"
            "The designed-difficulty section reads the SOURCE tree, which is not "
            "the same checkout as --root. Pass --seeds explicitly.")
    if len(seeds) != N_SEEDS:
        print(f"WARNING: expected {N_SEEDS} scenario seeds, found {len(seeds)} "
              f"in {seed_dir}")
    return seeds


def base_of(scen):
    """scenario slug -> base seed name (strip the trailing _v<NN> variant)."""
    return re.sub(r"_v\d+$", "", scen or "")


def seed_plugins(seed):
    """The conflict plugins one seed activates, as the generator routes them.

    Mirrors conflict_llm_pipeline.py:1582: `conflict_taxonomy` and
    `difficulty_challenges` are concatenated into one token list and resolved
    through the registry, with unknown tokens dropped -- `sla_support_response`
    carries a `subagent_delay` token that is not a registered plugin, and the
    generator ignores it, so this must too. `archetype` is deliberately absent.

    The registry's own edge case is that an EMPTY token list resolves to every
    plugin. No seed in this corpus hits it (all 20 carry 4-7 tokens), but a seed
    that ever did would silently join all 23 buckets, so guard it here the same
    way design_axis_slice.py does.
    """
    tokens = ((seed.get("conflict_taxonomy") or [])
              + (seed.get("difficulty_challenges") or []))
    if not tokens:
        return []
    seen = []
    for t in tokens:
        if t in PLUGIN_CATEGORY and t not in seen:
            seen.append(t)
    return seen


def seed_buckets(seed):
    """One seed -> every (axis, bucket) pair it belongs to."""
    plugins = seed_plugins(seed)
    # Challenge plugins get their own axis and are excluded from `category`, so
    # the category buckets describe the conflict mechanic rather than being
    # swamped by a bucket that every seed is in.
    pairs = [("plugin", p) for p in plugins]
    pairs += [("category", c) for c in sorted(
        {PLUGIN_CATEGORY[p] for p in plugins
         if PLUGIN_CATEGORY[p] != "challenge"})]
    pairs += [
        ("vision", "vision_graded" if seed.get("force_vision_locus")
         else "not_vision_graded"),
        ("async", "async_required" if seed.get("force_async_wait") else "no_async"),
    ]
    pairs += [("challenge", t) for t in (seed.get("difficulty_challenges") or [])]
    return pairs


def bucket_seed_counts(seeds):
    """(axis, bucket) -> how many SCENARIO SEEDS are in it. Model-independent.

    Reported beside every rate because the task count alone hides how thin the
    design is: 10 of the 23 plugin buckets rest on a single seed, so a 30-task
    column is 30 rounds of one scenario, not 30 independent trials.
    """
    counts = defaultdict(set)
    for slug, sd in seeds.items():
        for pair in seed_buckets(sd):
            counts[pair].add(slug)
    return {k: len(v) for k, v in counts.items()}


def design_axes(stats, seeds):
    """Bucket every task onto the designed axes, per model.

    Integer counts only -- see the section comment above for why the mean score
    is deliberately absent. A task whose scenario has no seed is counted in
    `unmapped` and excluded from the baseline, so a broken join shows up as a
    shrunken denominator rather than as a model that failed.
    """
    nseeds = bucket_seed_counts(seeds)
    out = {}
    for _, _, key in MODELS:
        agg = defaultdict(lambda: [0, 0])  # (axis, bucket) -> [n, passed]
        unmapped = set()
        n = passed = 0
        for tid, t in stats[key]["by_task"].items():
            base = base_of(scenario_of(tid))
            seed = seeds.get(base)
            if seed is None:
                unmapped.add(base)
                continue
            ok = 1 if t["success"] else 0
            n += 1
            passed += ok
            for pair in seed_buckets(seed):
                agg[pair][0] += 1
                agg[pair][1] += ok
        base_pct = 100 * passed / n if n else 0.0
        axes = defaultdict(dict)
        for (axis, bucket), (bn, bp) in agg.items():
            pct = 100 * bp / bn if bn else 0.0
            axes[axis][bucket] = {"n": bn, "seeds": nseeds.get((axis, bucket), 0),
                                  "passed": bp, "pass_pct": pct,
                                  "d_pass_pct": pct - base_pct}
        out[key] = {
            "baseline": {"n": n, "passed": passed, "pass_pct": base_pct},
            "axes": {a: dict(sorted(b.items())) for a, b in sorted(axes.items())},
            "unmapped": sorted(unmapped),
        }
    return out


def _codelist(names):
    """Oxford-comma list of <code> spans."""
    xs = [f"<code>{esc(n)}</code>" for n in names]
    if len(xs) <= 2:
        return " and ".join(xs)
    return ", ".join(xs[:-1]) + " and " + xs[-1]


def render_design_axes(da, meta):
    keys = [k for _, _, k in MODELS]
    nk = len(keys)

    # Pass 1: measure. The prose under the heading is generated from the same
    # dict the table prints, so a claim there cannot drift from the numbers
    # beside it -- the failure mode this file's MAINTENANCE RULE exists for.
    measured = {}  # bucket -> {"n", "below", "mean_d", "recs", "lever"}
    for _heading, axis, buckets in DESIGN_GROUPS:
        for bucket, is_lever in buckets:
            recs = [da[k]["axes"].get(axis, {}).get(bucket) for k in keys]
            got = [r for r in recs if r]
            if not got:
                continue
            measured[bucket] = {
                "n": got[0]["n"],
                "seeds": got[0].get("seeds", 0),
                "below": sum(1 for r in got if r["d_pass_pct"] < 0),
                "mean_d": sum(r["d_pass_pct"] for r in got) / len(got),
                "recs": recs,
                "lever": is_lever,
            }
    levers = {b: m for b, m in measured.items() if m["lever"]}
    universal = [b for b, m in levers.items() if m["below"] == nk]
    inert = [b for b, m in levers.items() if m["below"] == 0]
    # Name breaks every tie, because these pick which bucket gets named in the
    # generated prose and dict order must not decide that.
    sharp = min(universal, key=lambda b: (levers[b]["mean_d"], b)) if universal else None
    # The thinnest slice is a SET, not a single bucket: 11 levers sit on one seed
    # each, so `min()` over them would name whichever the dict happened to hold
    # first and the "is the sharpest also the thinnest" test would be a coin flip.
    thin_n = min((m["n"] for m in levers.values()), default=0)
    thinnest = {b for b, m in levers.items() if m["n"] == thin_n}
    sharp_is_thin = bool(sharp) and sharp in thinnest

    lead = (f"<b>{len(universal)} of the {len(levers)} designed levers push "
            f"every one of the {nk} models below its own baseline</b>"
            f"{' &mdash; ' + _codelist(sorted(universal)) if universal else ''}.")
    if sharp:
        lead += (f" The sharpest is <code>{esc(sharp)}</code> at "
                 f"<b>{levers[sharp]['mean_d']:+.1f} pp</b> averaged over the field")
        if sharp_is_thin:
            tied = len(thinnest) - 1
            lead += (", and it is "
                     + (f"tied with {tied} other lever{'s' if tied != 1 else ''} for "
                        if tied else "")
                     + f"the thinnest slice on the board: {levers[sharp]['n']} of "
                     f"{da[keys[0]]['baseline']['n']} tasks, from "
                     + (f"{levers[sharp]['seeds']} scenario seeds"
                        if levers[sharp]["seeds"] != 1 else "a single scenario seed"))
        lead += "."
    if inert:
        lead += (f" {len(inert)} lever{'s' if len(inert) != 1 else ''} "
                 f"push{'' if len(inert) != 1 else 'es'} none of them below "
                 f"baseline: {_codelist(sorted(inert))}.")

    head = ("<tr><th>designed lever</th><th>tasks</th><th>seeds</th>"
            + "".join(f"<th>{esc(MODEL_LABEL[k])}</th>" for k in keys)
            + "<th>below<br>baseline</th></tr>")

    # Baseline row first, so every delta below it has a visible anchor.
    base = [da[k]["baseline"] for k in keys]
    rows = ['<tr><td class="ck dim">all tasks (baseline)</td>'
            f'<td class="dim">{base[0]["n"]}</td>'
            f'<td class="dim">{N_SEEDS}</td>'
            + "".join(f'<td class="dim">{x["pass_pct"]:.1f}%</td>' for x in base)
            + '<td class="dim">\u2014</td></tr>']
    for heading, _axis, buckets in DESIGN_GROUPS:
        rows.append(f'<tr class="famrow"><td colspan="{nk + 4}">{esc(heading)}'
                    '</td></tr>')
        for bucket, _is_lever in buckets:
            m = measured.get(bucket)
            if not m:
                continue
            cells = []
            for r in m["recs"]:
                if not r:
                    cells.append('<td class="dim">\u2014</td>')
                    continue
                d = r["d_pass_pct"]
                tone = "pos" if d > 0 else ("neg" if d < 0 else "zero")
                cells.append(f'<td>{r["pass_pct"]:.1f}%'
                             f'<div class="dxd {tone}">{d:+.1f} pp</div></td>')
            hit = ' class="key"' if m["below"] == nk else ""
            thin = ' class="thin"' if m["lever"] and m["seeds"] <= 1 else ' class="dim"'
            rows.append(f'<tr><td class="ck"><code>{esc(bucket)}</code></td>'
                        f'<td class="dim">{m["n"]}</td>'
                        f'<td{thin}>{m["seeds"]}</td>' + "".join(cells)
                        + f'<td{hit}>{m["below"]}/{nk}</td></tr>')

    warn = ""
    if sharp_is_thin:
        one_seed = sorted(b for b, m in levers.items() if m["seeds"] == 1)
        widest = max(levers, key=lambda b: (levers[b]["n"], b))
        warn = ('<p class="h2sub warn"><b>Statistical power is allocated opposite '
                f'to the signal.</b> The lever with the largest effect, '
                f'<code>{esc(sharp)}</code>, carries {levers[sharp]["n"]} of the '
                f'{da[keys[0]]["baseline"]["n"]} tasks and comes from '
                + (f'{levers[sharp]["seeds"]} scenario seeds'
                   if levers[sharp]["seeds"] != 1 else 'a single scenario seed')
                + ', so its column is that seed\u2019s difficulty as much as it is '
                f'the lever\u2019s. {len(one_seed)} of the {len(levers)} levers rest '
                'on one seed each, while the widest, '
                f'<code>{esc(widest)}</code>, spans {levers[widest]["seeds"]} seeds '
                f'and moves the field {levers[widest]["mean_d"]:+.1f} pp. Treat the '
                'ranking as a hypothesis to re-run with a balanced design, not as a '
                'measured effect size.</p>')

    return (f'<h2 id="design">Designed difficulty axes</h2>'
            f'<p class="h2sub">{lead}</p>'
            + warn
            + f'<table class="cmdtable"><thead>{head}</thead>'
            f'<tbody>{"".join(rows)}</tbody></table>'
            '<p class="tblnote">Cells are pass@budget within the bucket; the small '
            'line under each is that model\u2019s distance from its <i>own</i> '
            'all-task baseline (top row), which is the only way these are '
            'comparable \u2014 the field spans '
            f'{min(x["pass_pct"] for x in base):.1f}% to '
            f'{max(x["pass_pct"] for x in base):.1f}% overall. Green is above that '
            'model\u2019s baseline, amber below; the rightmost column counts how '
            f'many of the {_numword(nk)} landed below it. Buckets come from the '
            f'{N_SEEDS} scenario seeds in <code>{esc(SEED_SUBDIR)}</code>, joined '
            'to each task by scenario slug with the trailing <code>_v&lt;NN&gt;</code> '
            'variant suffix stripped. The first two groups are the seed booleans '
            f'<code>{esc(AXIS_FIELD["async"])}</code> and '
            f'<code>{esc(AXIS_FIELD["vision"])}</code>, each shown with its control '
            'arm; the last two are the conflict plugins the generator actually '
            f'routes to, named in <code>{esc(AXIS_FIELD["plugin"])}</code> and '
            f'<code>{esc(AXIS_FIELD["challenge"])}</code>. Those two fields are '
            'concatenated into a single token list before routing, so a challenge '
            'is a plugin like any other and the split here is for readability '
            'rather than mechanism: all seven challenge plugins only inject prompt '
            'text, but so do three of the sixteen mechanics, and the remaining '
            'thirteen are the ones that gate command execution. <b>seeds</b> is how '
            f'many of the {N_SEEDS} scenarios carry that lever \u2014 the plugin rows '
            'overlap, so they do not sum to the corpus, and a one-seed row (amber) '
            'is ten rounds of one scenario in three variants rather than 30 '
            'independent trials. Pass '
            'rates are the only statistic here on purpose: the per-task score is '
            'rounded on load, so a mean taken from it would disagree in the third '
            'decimal with the two other derivations of this table. The mean-score '
            'view, the conflict-category buckets, the prompt-only-versus-mechanic '
            'contrast and the per-seed detail are in '
            '<code>eval-results-capability-analysis.md</code> \u00a75.3.</p>')


def _scenario_rounds(stats, scenario):
    """All ten rounds of one scenario as [(spread, tid, row)], weakest spread first.

    A round is skipped only if a voter is missing from it, so the spread is always
    measured over a constant field and cannot widen merely because a model dropped
    out. All fourteen models are complete at 600/600 on this corpus, so the guard
    never fires today; it stays as a tripwire for future partial runs."""
    rounds = []
    for rr in range(1, 11):
        tid = f"conflict_vlm_{scenario}_r{rr:02d}"
        row = {}
        for _, _, key in MODELS:
            t = stats[key]["by_task"].get(tid)
            if t:
                row[key] = t
        if any(k not in row for k in PROSE_KEYS):
            continue
        pv = [row[k]["score"] for k in PROSE_KEYS]
        rounds.append((max(pv) - min(pv), tid, row))
    return rounds


def pick_round(stats, scenario):
    """The round to showcase for `scenario`, as (spread, tid, row).

    For the eight curated scenarios this returns the PINNED round from
    CURATED_ROUNDS. The max-spread rule that originally chose them saturates on
    this corpus (see the CURATED_ROUNDS comment), so re-deriving the round every
    build would let loop order, not the data, pick the evidence. Any other
    scenario still falls back to max spread, lowest round number breaking ties."""
    rounds = _scenario_rounds(stats, scenario)
    if not rounds:
        return None
    pin = CURATED_ROUND_OF.get(scenario)
    if pin:
        want = f"conflict_vlm_{scenario}_{pin}"
        for cand in rounds:
            if cand[1] == want:
                return cand
        raise SystemExit(
            f"curated pin {want} is not present in the corpus. Either the round "
            f"was dropped or a PROSE_KEYS voter is missing from it; fix the corpus "
            f"or re-pin CURATED_ROUNDS deliberately.")
    best = None
    for cand in rounds:
        if best is None or cand[0] > best[0]:
            best = cand
    return best


def check_curated_pins(stats):
    """Fail the build if a pinned curated round is no longer a maximum-spread round.

    This is what makes the pin honest rather than merely convenient: the report
    tells the reader each showcased round attains its scenario's widest spread,
    and this re-proves that claim against the freshly loaded corpus on every
    build. It does NOT require the pin to be the unique maximum -- on this corpus
    five of the eight are tied at 1.000 with other rounds, which is exactly why
    the choice among them is pinned instead of left to loop order."""
    bad = []
    for scen, rnd in CURATED_ROUNDS:
        rounds = _scenario_rounds(stats, scen)
        if not rounds:
            bad.append(f"{scen}: no complete round in the corpus")
            continue
        top = max(r[0] for r in rounds)
        want = f"conflict_vlm_{scen}_{rnd}"
        got = next((r[0] for r in rounds if r[1] == want), None)
        if got is None:
            bad.append(f"{scen}: pinned round {rnd} absent")
        elif got < top - 1e-9:
            ties = sorted(r[1][-3:] for r in rounds if r[0] >= top - 1e-9)
            bad.append(f"{scen}: pinned {rnd} spread {got:.3f} < max {top:.3f} "
                       f"(maximal rounds: {', '.join(ties)})")
    if bad:
        raise SystemExit("curated round pins are stale:\n  " + "\n  ".join(bad))


def curated_pin_facts(stats):
    """The tie structure behind the curated pins, for the prose that describes it.

    render_timelines used to state this by hand -- "all eight scenarios reach a
    spread of 1.000, and five of the eight reach it on several rounds at once
    (satellite... on nine of its ten)". Every one of those is a function of the
    voter population, which is exactly the thing that changes when a model is
    added, so they are computed rather than typed. check_curated_pins() proves
    the pins are still maximal; this describes the maxima it proved.
    """
    facts = []
    for scen, rnd in CURATED_ROUNDS:
        rounds = _scenario_rounds(stats, scen)
        if not rounds:
            continue
        top = max(r[0] for r in rounds)
        facts.append({
            "scen": scen,
            "top": top,
            "ties": sum(1 for r in rounds if r[0] >= top - 1e-9),
            "n_rounds": len(rounds),
        })
    if not facts:
        return None
    tops = {round(f["top"], 3) for f in facts}
    tied = [f for f in facts if f["ties"] > 1]
    most = max(facts, key=lambda f: f["ties"])
    return {
        "n": len(facts),
        "uniform_top": next(iter(tops)) if len(tops) == 1 else None,
        "n_tied": len(tied),
        "most": most,
        "tie_counts": [f["ties"] for f in facts],
    }


def render_chain(chain):
    dots = []
    for st in chain:
        cls = "dot err" if st["err"] else "dot"
        dots.append(
            f'<span class="{cls}" style="background:{CAT_COLOR[st["cat"]]}" '
            f'title="{esc(st["key"])}"></span>'
        )
    return '<div class="chain">' + "".join(dots) + "</div>"


def render_cmdlist(name, t):
    steps = []
    for i, st in enumerate(t["chain"]):
        chip = f'<span class="cchip" style="background:{CAT_COLOR[st["cat"]]}"></span>'
        delta = f'<span class="cd">+{st["sd"]:.2f}</span>' if st.get("sd", 0) > 0 else ""
        errf = '<span class="cerr">err</span>' if st["err"] else ""
        title = esc(st.get("out") or "(no stdout captured)")
        act = esc(st.get("act") or st["key"])
        steps.append(
            f'<div class="cstep"><span class="cnum">{i}</span>{chip}'
            f'<code class="cact" title="{title}">{act}</code>{delta}{errf}</div>'
        )
    vcls = "pass" if t["success"] else "failv"
    verdict = "PASS" if t["success"] else "fail"
    return (f'<div class="cmlist"><div class="cmnm">{esc(name)}'
            f'<span class="verd {vcls}">{verdict}</span>'
            f'<span class="cmsc">score {t["score"]:.2f} \u00b7 {esc(t["stop"])}</span></div>'
            + "".join(steps) + "</div>")


def render_timelines(stats, notes):
    pf = curated_pin_facts(stats)
    nsc = _numword(pf["n"]) if pf else "eight"
    if pf and pf["uniform_top"] is not None:
        tie_txt = (f"all {nsc} scenarios reach a spread of "
                   f"{pf['uniform_top']:.3f}, and {_numword(pf['n_tied'])} of the "
                   f"{nsc} reach it on several rounds at once "
                   f"({esc(pf['most']['scen'])} on "
                   f"{_numword(pf['most']['ties'])} of its "
                   f"{_numword(pf['most']['n_rounds'])})")
    else:
        tie_txt = ("several scenarios reach their maximum spread on more than "
                   "one round")
    out = [f"<h2>Per-scenario paired tool-chains \u2014 {pf['n'] if pf else 8} "
           "curated scenarios (1 representative round each)</h2>",
           f'<p class="h2sub">A qualitative walkthrough: {pf["n"] if pf else 8} '
           "scenarios hand-picked from the "
           f"full 60, shown as the same task across all {_numword(len(MODELS))} "
           "models. This is not the "
           "whole set \u2014 every aggregate elsewhere in this report is still computed "
           "over all 600 tasks (60 scenarios \u00d7 10 rounds). The dot row is the shape "
           "at a glance (one dot per step, colored by action category; red ring = the "
           "tool errored). Expand <b>full command chains</b> for the verbatim actions "
           "with +score-delta markers, error flags, and stdout on hover. "
           "<b>How the round was chosen:</b> each round shown attains the widest score "
           f"spread its scenario reaches across all {_numword(len(MODELS))} models. "
           f"That criterion does not pick a unique round here: {tie_txt}. "
           "Where rounds "
           "tie, the one shown is fixed in advance rather than left to scan order, and "
           "the build re-verifies on every run that it is still a maximum-spread "
           "round.</p>"]
    scen_notes = (notes or {}).get("scenarios", {})
    for scen in CURATED_SCENARIOS:
        picked = pick_round(stats, scen)
        if not picked:
            continue
        _, tid, row = picked
        out.append('<div class="pair">')
        out.append(f'<div class="ptitle">{esc(tid)}</div>')
        if scen_notes.get(scen):
            out.append(f'<p class="scennote">{md_inline(scen_notes[scen])}</p>')
        for name, _, key in MODELS:
            t = row.get(key)
            if not t:
                continue
            verdict = "PASS" if t["success"] else "fail"
            vcls = "pass" if t["success"] else "failv"
            cap = (f'{t["steps"]} steps \u00b7 {t["n_open"]} opens \u00b7 '
                   f'{t["n_write"]} writes \u00b7 score {t["score"]:.2f} \u00b7 {esc(t["stop"])}')
            out.append(
                f'<div class="trow"><span class="tnm">{esc(name)}</span>'
                f'{render_chain(t["chain"])}'
                f'<span class="verd {vcls}">{verdict}</span>'
                f'<span class="tcap">{cap}</span></div>'
            )
        out.append('<details class="cmd"><summary>full command chains \u00b7 '
                   "verbatim actions, +score-delta, err flags, stdout on hover</summary>")
        for name, _, key in MODELS:
            if key in row:
                out.append(render_cmdlist(name, row[key]))
        out.append("</details>")
        out.append("</div>")
    return "".join(out)


def _chain_count(t, cats):
    return sum(1 for s in t["chain"] if s["cat"] in cats)


def _open_acts(t):
    """The asset-open steps of one episode, as the acted-on asset identifiers."""
    return [s.get("act") or s["key"] for s in t["chain"] if s["cat"] == "asset_open"]


def _sig_metrics(stats):
    """Per-model aggregates that the signature LABELS assert things about.

    Kept separate from the selectors on purpose: a selector only has to find a
    representative episode, and a wrong one is merely a dull example. A label is
    a claim printed next to the model's name, and a wrong one is a false
    statement in the report. Everything a label quotes is computed here so it can
    be re-checked at build time by _sig_labels.
    """
    out = {}
    for _, _, k in MODELS:
        ts = list(stats[k]["by_task"].values())
        n = len(ts) or 1
        op = [len(_open_acts(t)) for t in ts]
        dis = [len(set(_open_acts(t))) for t in ts]
        steps = sum(len(t["chain"]) for t in ts) or 1
        out[k] = {
            "n": n,
            "pass_pct": 100.0 * sum(1 for t in ts if t["success"]) / n,
            "cap_pct": 100.0 * sum(
                1 for t in ts if str(t["stop"]).startswith("reached max_steps")) / n,
            "opens": sum(op) / n,
            "distinct": sum(dis) / n,
            "zero_open": sum(1 for t in ts if t["n_open"] == 0),
            "zero_write": sum(1 for t in ts if t["n_write"] == 0),
            "open_share": 100.0 * stats[k]["catmix"].get("asset_open", 0)
                          / (stats[k]["catmix_total"] or 1),
            "err_step_pct": 100.0 * sum(
                1 for t in ts for s in t["chain"] if s["err"]) / steps,
            "ground_abandon": 100.0 * sum(
                1 for t in ts if t["n_open"] >= 3 and t["n_write"] == 0) / n,
        }
    return out


def _sig_only(m, pred, who, claim):
    """Assert `who` is the ONLY model satisfying `pred`, else stop the build.

    Every signature label is a uniqueness claim ("the only model that ..."), and
    uniqueness is exactly the property that rots when the field grows. Two labels
    in this report had already rotted before this check existed: gpt-5.6-sol was
    still described as sweeping "the widest set of distinct assets in one
    episode" when glm-5.2 (7.21/task), kimi-k3 (6.71) and deepseek-v4-flash
    (5.98) had all overtaken it (4.04, fifth); and claude-opus-5's "misses nearly
    every required check at once" outlived the numbers too -- it has the field's
    most correlated checks (highest lift and overdispersion) but the LOWEST
    absolute all-fail rate of the fourteen, 19.7%.
    """
    # `m` may be a SUBSET of the field when the claim is scoped to one block
    # (dsv4's is "alone among the four open-source entrants"), so iterate what
    # was handed in rather than all of MODELS -- which would KeyError.
    hits = sorted(k for _, _, k in MODELS if k in m and pred(m[k]))
    if hits != [who]:
        raise SystemExit(
            f"signature label for {who} claims: {claim}\n"
            f"  but the models satisfying it are {hits or '[none]'}. The label is "
            "no longer a uniqueness claim -- rewrite it against the current field "
            "rather than shipping a superlative the data does not support.")


def _sig_max(m, field, who, claim):
    """Assert `who` holds the strict field maximum of `field`."""
    ranked = sorted(((m[k][field], k) for _, _, k in MODELS), reverse=True)
    if ranked[0][1] != who or ranked[0][0] == ranked[1][0]:
        raise SystemExit(
            f"signature label for {who} claims: {claim}\n"
            f"  but the ranking on {field!r} is {[(k, round(v, 2)) for v, k in ranked[:4]]}"
            " -- not a strict maximum for this model.")


def _sig_labels(stats):
    """The reader-facing signature line for each model, derived and self-checked.

    These used to be nine hand-written strings. They are built here instead so
    that (a) the magnitudes in them come from the run being rendered, and (b) the
    uniqueness each one asserts is verified against the field at build time --
    see _sig_only for the two that had already gone stale.
    """
    m = _sig_metrics(stats)
    OS = [k for _, _, k in MODELS if PROVIDER[k] == "Open-source"]
    # One threshold for "grounds heavily", shared by the kimi label AND the guard
    # behind it. Spelled out because an earlier draft of that label said "6.7+
    # opens" in the prose while the guard tested 5, and then cited
    # deepseek-v4-flash at 6.00 as one of the models clearing the stated bar --
    # which it does not. Prose and guard must read the same number.
    HEAVY = 5.0
    _astat = {k: 100.0 * stats[k]["catmix"].get("agent_status", 0)
                 / (stats[k]["catmix_total"] or 1) for _, _, k in MODELS}
    lab = {}

    lab["flash"] = (
        "step-capped loop \u2014 keeps acting past the point of completion and hits "
        f"the 20-step cap on <b>{m['flash']['cap_pct']:.0f}%</b> of tasks, the "
        "highest in the field")
    _sig_max(m, "cap_pct", "flash", "the highest step-cap rate in the field")

    lab["flash36"] = (
        "near-miss submission \u2014 grounds and commits fast (no loop), but the "
        "graded content just misses on a task its peers solve")

    lab["flash37"] = (
        "loops without grounding \u2014 the only model that both grounds this lightly "
        f"(<b>{m['flash37']['opens']:.2f}</b> asset opens per task) and still burns "
        f"out the 20-step cap on <b>{m['flash37']['cap_pct']:.0f}%</b> of tasks. It "
        "also spends a larger share of its steps on <code>agent status</code> than "
        f"any other model (<b>{_astat['flash37']:.1f}%</b>, against "
        f"<b>{sorted(_astat.values(), reverse=True)[1]:.1f}%</b> for the runner-up) "
        "\u2014 the steps go to polling rather than to evidence")
    _sig_only(m, lambda d: d["opens"] < 2.5 and d["cap_pct"] > 40, "flash37",
              "under 2.5 opens per task AND over 40% of tasks step-capped")
    _ar = sorted(((v, k) for k, v in _astat.items()), reverse=True)
    if _ar[0][1] != "flash37" or _ar[0][0] == _ar[1][0]:
        raise SystemExit(
            "gemini-3.7-flash's signature claims the largest agent_status share, "
            f"but the ranking is {[(k, round(v, 2)) for v, k in _ar[:4]]}.")

    # Narrowed from "wait / agent status" to `wait` alone: flash37's label now
    # claims the largest agent_status share (3.3%), which is true, so leaving both
    # tools in pro's line had the two exemplars claiming the same behavior. pro's
    # own field high is `wait` -- 5.1% of its steps against 3.9% for the runner-up
    # -- and the combined wait+agent_status rate per task, 0.85, is also its.
    _wait = {k: 100.0 * stats[k]["catmix"].get("wait", 0)
                / (stats[k]["catmix_total"] or 1) for _, _, k in MODELS}
    lab["pro"] = (
        "over-investigation \u2014 delegates, then stalls on <code>wait</code>: "
        f"<b>{_wait['pro']:.1f}%</b> of its steps are spent waiting, more than any "
        f"other model (runner-up <b>{sorted(_wait.values(), reverse=True)[1]:.1f}%</b>)")
    _wr = sorted(((v, k) for k, v in _wait.items()), reverse=True)
    if _wr[0][1] != "pro" or _wr[0][0] == _wr[1][0]:
        raise SystemExit(
            "gemini-3.1-pro's signature claims the largest `wait` share, but the "
            f"ranking is {[(k, round(v, 2)) for v, k in _wr[:4]]}.")

    lab["sonnet"] = (
        "grounding gap \u2014 writes deliverables from listed / text context without "
        f"opening the asset evidence at all on <b>{m['sonnet']['zero_open']}</b> of "
        f"{m['sonnet']['n']} tasks, more than any other model")
    _sig_max(m, "zero_open", "sonnet", "the most tasks completed with zero asset opens")

    lab["opus"] = (
        "content-precision miss on thin evidence \u2014 commits a deliverable off very "
        "few asset opens, and the graded content just misses")

    # Rewritten. The old line said "when a task goes wrong it misses nearly every
    # required check at once", which the numbers contradict: opus-5 has the LOWEST
    # all-fail rate of the fourteen. What it does hold is the strongest coupling
    # between checks, in both directions -- and mostly in the winning direction.
    _c = stats["opus5"]["clustering"]
    lab["opus5"] = (
        "clustered outcomes \u2014 the field&rsquo;s most correlated checks "
        f"(lift <b>{_c['lift']}</b>, overdispersion <b>{_c['overdispersion']}</b>, "
        "both field highs): it takes tasks whole rather than banking partial "
        f"credit, clearing every check on <b>{_c['fail_hist_pct'][0]:.1f}%</b> of "
        f"tasks against the <b>{_c['indep_hist_pct'][0]:.1f}%</b> independence "
        "predicts. The exemplar is the same coupling running the other way")
    for _f in ("lift", "overdispersion"):
        _r = sorted(((stats[k]["clustering"][_f], k) for _, _, k in MODELS), reverse=True)
        if _r[0][1] != "opus5" or _r[0][0] == _r[1][0]:
            raise SystemExit(
                f"opus-5's signature claims the field-high {_f}, but the ranking is "
                f"{[(k, v) for v, k in _r[:4]]}.")

    # Rewritten. "Sweeps the widest set of distinct assets in one episode" was
    # true of the nine-model field and is false of this one (sol is fifth on
    # distinct opens). What is still sol's alone is that it never skips either
    # half of the job.
    lab["sol"] = (
        "no skipped steps \u2014 the only model that grounds on <b>every</b> task "
        f"({m['sol']['n']}/{m['sol']['n']} open at least one asset) and commits a "
        f"deliverable on all but <b>{m['sol']['zero_write']}</b> of them; when it "
        "fails it fails on content, not for want of looking")
    _sig_only(m, lambda d: d["zero_open"] == 0 and d["zero_write"] <= 1, "sol",
              "zero tasks without an asset open AND at most one without a write")

    lab["luna"] = (
        "redundant re-grounding \u2014 re-opens assets it has already opened this "
        "episode instead of committing, and caps most often of the OpenAI three")
    _sig_only(m, lambda d: d["opens"] - d["distinct"] > 0.2, "luna",
              "more than 0.2 repeat asset opens per task")
    _oai = sorted(((m[k]["cap_pct"], k) for _, _, k in MODELS
                   if PROVIDER[k] == "OpenAI"), reverse=True)
    if _oai[0][1] != "luna" or _oai[0][0] == _oai[1][0]:
        raise SystemExit(
            "luna's signature claims it caps most often of the OpenAI three, but "
            f"the cap rates are {[(k, round(v, 1)) for v, k in _oai]}.")

    lab["terra"] = (
        "early commit \u2014 among the shortest chains in the field; it stops on its "
        "own rather than looping, and stops short of the graded anchors")
    _steps = sorted((sum(len(t["chain"]) for t in stats[k]["by_task"].values())
                     / (len(stats[k]["by_task"]) or 1), k) for _, _, k in MODELS)
    if "terra" not in [k for _, k in _steps[:3]]:
        raise SystemExit(
            "gpt-5.6-terra's signature says it is among the shortest chains in the "
            f"field, but the three shortest are {[k for _, k in _steps[:3]]} "
            f"({[(k, round(v, 2)) for v, k in _steps[:4]]}).")

    lab["dsv4"] = (
        "the open-source model that converges \u2014 alone among the four open-source "
        f"entrants it stops on its own (<b>{m['dsv4']['cap_pct']:.1f}%</b> "
        "step-capped, against "
        + " / ".join(f"{m[k]['cap_pct']:.1f}%"
                     for k in sorted((x for x in OS if x != "dsv4"),
                                     key=lambda x: -m[x]["cap_pct"]))
        + f") and commits a deliverable on <b>"
        f"{100 - 100 * m['dsv4']['zero_write'] / m['dsv4']['n']:.1f}%</b> of tasks")
    _sig_only({k: m[k] for k in OS},
              lambda d: d["cap_pct"] < 20 and d["zero_write"] / d["n"] < 0.15,
              "dsv4", "the only open-source model under a 20% cap rate that also "
              "writes a deliverable on 85%+ of tasks")

    lab["dsv4f"] = (
        "grounds hardest, delivers least \u2014 opens "
        f"<b>{m['dsv4f']['opens']:.2f}</b> assets per task yet ends "
        f"<b>{100 * m['dsv4f']['zero_write'] / m['dsv4f']['n']:.1f}%</b> of them "
        "with nothing written at all; the only model that pairs heavy grounding "
        "with a near coin-flip chance of producing no deliverable")
    _sig_only(m, lambda d: d["opens"] >= 5 and d["zero_write"] / d["n"] >= 0.40,
              "dsv4f", "5+ opens per task AND 40%+ of tasks with no write")

    _heavy = sorted((k for _, _, k in MODELS if m[k]["opens"] >= HEAVY),
                    key=lambda k: -m[k]["opens"])
    lab["kimi"] = (
        "heavy grounding that pays \u2014 opens "
        f"<b>{m['kimi']['opens']:.2f}</b> assets per task and still converts, at "
        f"<b>{m['kimi']['pass_pct']:.1f}%</b> pass: the only model in the field that "
        f"grounds this heavily ({HEAVY:.0f}+ opens per task) and clears 20%. The "
        f"other {_numword(len(_heavy) - 1)} that do land at "
        + " and ".join(f"<b>{m[k]['pass_pct']:.1f}%</b>"
                       for k in _heavy if k != "kimi"))
    _sig_only(m, lambda d: d["opens"] >= HEAVY and d["pass_pct"] >= 20, "kimi",
              f"{HEAVY:.0f}+ opens per task AND a 20%+ pass rate")

    # The "ends with nothing written" clause was cut from this label: at
    # 26.2% it is real, but deepseek-v4-flash's label is built on exactly that
    # failure at 47.3%, and with both making the claim the two exemplars selected
    # the same shape of episode -- 19 opens, 0 writes, capped -- and told one
    # story twice. glm keeps the breadth, dsv4f keeps the abandonment.
    # "converts least of what it sees" was in an earlier draft of this line and is
    # false: glm passes 14.7%, above deepseek-v4-flash's 8.5%, and it is not last
    # on pass-per-open either. Replaced with its actual rank, computed here.
    _glm_rank = 1 + sum(1 for _, _, k in MODELS
                        if m[k]["pass_pct"] > m["glm"]["pass_pct"])
    lab["glm"] = (
        "grounding as the whole strategy \u2014 "
        f"<b>{m['glm']['open_share']:.1f}%</b> of everything it does is opening an "
        f"asset and it sweeps <b>{m['glm']['distinct']:.1f}</b> distinct assets per "
        "task, both field highs \u2014 and the widest sweep in the field is not the "
        f"best-converting one: it places <b>{_ordinal(_glm_rank)}</b> "
        f"of {len(MODELS)} on pass rate, at <b>{m['glm']['pass_pct']:.1f}%</b>")
    _sig_max(m, "open_share", "glm", "the largest asset-open share of all steps")
    _sig_max(m, "distinct", "glm", "the most distinct assets opened per task")

    missing = [k for _, _, k in MODELS if k not in lab]
    if missing:
        raise SystemExit(
            f"signature labels missing for {missing}. The section is titled 'one "
            "exemplar per model'; add a label (and a selector) or retitle it.")
    return lab


def pick_signatures(stats):
    """Pick one round per model that most clearly exhibits its signature behavior.

    Each model is given a trait it holds uniquely in the FOURTEEN-model field, so no
    two exemplars tell the same story: flash -> the longest step-capped run; flash36
    -> the sharpest near-miss submission (grounds and commits fast without looping) on
    a task its peers solve; flash37 -> looping without grounding; pro -> the most
    wait/agent-status polling; sonnet -> the most writing done with zero asset opens;
    opus -> a content-precision miss reached on thin evidence; opus5 -> coupled check
    outcomes; sol -> never skipping either half of the job; luna -> redundant
    re-grounding (re-opens an asset it already opened this episode); terra -> early
    commit (the shortest chains); dsv4 -> the one open-source model that converges;
    dsv4f -> heaviest grounding with the least delivered; kimi -> heavy grounding that
    converts; glm -> grounding as the entire strategy. Only tasks where all fourteen
    models ran are eligible so the paired view is complete; picks are de-duplicated
    across models.

    The reader-facing text of each trait lives in _sig_labels, which derives its
    magnitudes from this run and CHECKS the uniqueness each label asserts. That check
    exists because two labels had already gone stale when the field grew from nine to
    fourteen -- see _sig_only.

    CAVEAT (v8 port): every SELECTOR below is computed from the loaded run, so the
    exemplar it returns is always real. The per-selector code comments on the nine
    ORIGINAL models, however, were written against the THINK corpus and quote ITS
    magnitudes; those figures have not been re-derived for v8 and must not be cited.
    Treat those comments as a record of why each selector is shaped the way it is, not
    as v8 measurements. The numbers that ARE v8 are the ones _sig_labels and the
    renderers compute."""
    def _peers_passed(tid, me):
        # "Peers" means every model EXCEPT the one being given an exemplar. This
        # used to hard-code `k != "flash36"`, which is right for the flash36
        # selector and wrong for every other caller: opus5_key was counting opus-5
        # itself among its own peers while excluding flash36 from the tally.
        return sum(1 for _, _, k in MODELS
                   if k != me and stats[k]["by_task"][tid]["success"])

    def flash_key(t, tid):
        return (1 if str(t["stop"]).startswith("reached max_steps") else 0, t["steps"] or 0)

    def flash36_key(t, tid):
        # Near-miss submission: failed despite grounding (>=1 asset open), committing
        # a written deliverable, and stopping short of the step cap, with a high
        # score. Tie-break toward tasks its peers solve -- that isolates content
        # precision on the graded anchors as the ceiling, not task difficulty.
        near = (not t["success"] and t["score"] >= 0.5 and t["n_write"] > 0
                and t["n_open"] > 0 and not str(t["stop"]).startswith("reached max_steps"))
        return (1 if near else 0, t["score"], _peers_passed(tid, "flash36"), -(t["steps"] or 999))

    def pro_key(t, tid):
        return (_chain_count(t, {"wait", "agent_status"}), t["steps"] or 0)

    def sonnet_key(t, tid):
        return (1 if (t["n_open"] == 0 and t["n_write"] > 0) else 0, t["n_write"])

    def opus_key(t, tid):
        # Content-precision miss reached on THIN evidence: opus 4.8 grounds far less
        # than the no-think corpus suggested -- 1.89 opens/task here, 8th of 10, and
        # only 140 of 600 tasks reach three opens -- yet it still commits a
        # deliverable and near-misses. So the tie-break runs toward the FEWEST opens
        # (was: the most), isolating "committed decisively on little evidence" rather
        # than "ground everything and still missed", which is no longer true of it.
        miss = (not t["success"] and t["n_open"] > 0 and t["n_write"] > 0
                and not str(t["stop"]).startswith("reached max_steps"))
        return (1 if miss else 0, t["score"], -t["n_open"], -(t["steps"] or 999))

    def opus5_key(t, tid):
        # Clustered collapse: opus 5 carries the field's highest excess clustering --
        # when it fails it tends to miss EVERY required check at once rather than
        # dropping one. Select the sharpest instance: a low-scoring run on a task most
        # peers solve, where the model still grounded and acted (so it is a
        # coherent-but-wrong run, not a no-op). Tie-break toward the most peers passed.
        collapse = (not t["success"] and t["score"] <= 0.25
                    and (t["n_open"] + t["n_write"]) > 0)
        return (1 if collapse else 0, _peers_passed(tid, "opus5"), -t["score"], t["n_open"])

    def _key_count(t, key):
        return sum(1 for s in t["chain"] if s["key"] == key)

    def sol_key(t, tid):
        # No skipped steps. The previous selector chased "the widest DISTINCT sweep",
        # on the claim that breadth was sol's field high. That was true of the
        # nine-model field and is false of this one: sol is FIFTH on distinct opens
        # (4.04/task) behind glm-5.2, kimi-k3, deepseek-v4-flash and deepseek-v4-pro,
        # and the breadth exemplar has moved to glm where the number now lives.
        # What is still sol's alone is completeness of process -- it is the only
        # model that opens an asset on all 600 tasks and writes on 599. So select the
        # run that shows process complete and content still wrong: failed, but it
        # both grounded and committed, widest sweep as the tie-break.
        complete = not t["success"] and t["n_open"] > 0 and t["n_write"] > 0
        return (1 if complete else 0, len(set(_open_acts(t))), t["score"])

    def luna_key(t, tid):
        # Redundant re-grounding. "Write-heavy" does not separate luna here (1.26
        # writes/task vs gpt-5.5's 1.29), but one number does, by a wide margin: 9.7%
        # of luna's asset opens re-open an asset it had already opened this episode,
        # against <=1.2% for every other model -- roughly 8x the runner-up. It also
        # caps most often of the OpenAI four (83 of 600). Select the run with the most
        # repeat opens, then the most steps: re-reading in place of committing.
        acts = _open_acts(t)
        return (0 if t["success"] else 1, len(acts) - len(set(acts)), t["steps"] or 0)

    def terra_key(t, tid):
        # Early commit. The inherited "grounds the most of any model" claim belongs to
        # luna (3.95 opens/task) not terra (2.57), so it is gone. terra's own extreme
        # is the opposite of a loop: 11.9 steps/task, the shortest chains in the
        # ten-model field, and it stops on its own (44 capped). Select the shortest
        # chain that committed a deliverable and still near-missed -- stopped early,
        # stopped short.
        near = (not t["success"] and t["score"] >= 0.5 and t["n_write"] > 0
                and not str(t["stop"]).startswith("reached max_steps"))
        return (1 if near else 0, -(t["steps"] or 999), t["score"])

    # --- the five models added after the nine-model field ------------------
    # These selectors are written against v8 directly; unlike the nine above,
    # their comments quote THIS run and may be cited.

    def _capped(t):
        return str(t["stop"]).startswith("reached max_steps")

    def flash37_key(t, tid):
        # Looping without grounding. flash37 opens 2.25 assets/task -- the second
        # lightest in the field -- yet caps on 40.7% of tasks, a combination no
        # other model shows (the lighter grounder, sonnet-5 at 1.83, caps on 3.3%).
        # Select the capped run that grounded the LEAST: twenty steps spent, almost
        # none of them on evidence.
        return (1 if _capped(t) else 0, -len(set(_open_acts(t))), t["steps"] or 0)

    def dsv4_key(t, tid):
        # The open-source model that converges. dsv4 caps on 13.5% of tasks where
        # its three open-source peers cap on 28.3 / 34.5 / 59.7, and it writes a
        # deliverable on 92.8%. Select a run where it terminated on its own with
        # both halves of the job done and still missed, tie-broken toward tasks
        # where the OTHER open-source models ran out of steps -- the contrast is
        # the point.
        os_capped = sum(1 for _, _, k in MODELS
                        if PROVIDER[k] == "Open-source" and k != "dsv4"
                        and _capped(stats[k]["by_task"][tid]))
        done = (not t["success"] and not _capped(t)
                and t["n_open"] > 0 and t["n_write"] > 0)
        return (1 if done else 0, os_capped, t["score"])

    def dsv4f_key(t, tid):
        # Grounds hardest, delivers least. 6.00 opens/task and 47.3% of tasks end
        # with nothing written -- the only model pairing 5+ opens with a 40%+
        # no-deliverable rate. Select the most extreme instance of exactly that:
        # the largest pile of asset opens that produced no write at all.
        abandoned = t["n_write"] == 0 and t["n_open"] > 0
        return (1 if abandoned else 0, len(_open_acts(t)), t["steps"] or 0)

    def kimi_key(t, tid):
        # Heavy grounding that pays. kimi is the only model that opens 6+ assets a
        # task and still converts (21.8% pass; the other two models grounding this
        # widely land at 14.7% and 8.5%). Every other exemplar in this section is a
        # failure; this one is deliberately a SUCCESS, because the claim is that
        # the breadth is productive rather than a symptom. Widest sweep that
        # actually solved the task.
        return (1 if t["success"] else 0, len(set(_open_acts(t))), -(t["steps"] or 999))

    def glm_key(t, tid):
        # Grounding as the whole strategy. 48.8% of every step glm takes is an
        # asset open and it sweeps 7.21 distinct assets a task, both field highs --
        # this is where the "widest sweep" exemplar moved to from sol.
        #
        # Requiring a WRITE is what keeps this exemplar distinct from
        # deepseek-v4-flash's. Without it both selectors converge on the same
        # shape (19 opens, no write, capped) and the section shows one behavior
        # twice. dsv4f owns "looked at everything and submitted nothing"; glm owns
        # "looked at everything, submitted, and still got it wrong".
        opens = _open_acts(t)
        share = len(opens) / len(t["chain"]) if t["chain"] else 0
        looked = not t["success"] and t["n_write"] > 0
        return (1 if looked else 0, len(set(opens)), share)

    labels = _sig_labels(stats)
    # Selector order == MODELS order, so the de-dup below hands each model the best
    # remaining task in the order the report lists them. Labels come from
    # _sig_labels; nothing reader-facing is spelled out here any more.
    specs = [(k, n, labels[k], f) for k, n, f in [
        ("flash", "gemini-3.5-flash", flash_key),
        ("flash36", "gemini-3.6-flash", flash36_key),
        ("flash37", "gemini-3.7-flash", flash37_key),
        ("pro", "gemini-3.1-pro", pro_key),
        ("opus", "claude-opus-4.8", opus_key),
        ("opus5", "claude-opus-5", opus5_key),
        ("sonnet", "claude-sonnet-5", sonnet_key),
        ("sol", "gpt-5.6-sol", sol_key),
        ("luna", "gpt-5.6-luna", luna_key),
        ("terra", "gpt-5.6-terra", terra_key),
        ("dsv4", "deepseek-v4-pro", dsv4_key),
        ("dsv4f", "deepseek-v4-flash", dsv4f_key),
        ("kimi", "kimi-k3", kimi_key),
        ("glm", "glm-5.2", glm_key),
    ]]
    # The section promises "one exemplar per model"; keep the promise checkable.
    if [k for k, _, _, _ in specs] != [k for _, _, k in MODELS]:
        raise SystemExit(
            "signature specs and MODELS disagree: "
            f"{[k for k, _, _, _ in specs]} vs {[k for _, _, k in MODELS]}")
    # A `generation_stall_empty_reply` is the provider returning nothing, not a
    # behavior -- it happens 11 times in 8,400 model-tasks (opus-5 3, glm-5.2 4,
    # kimi-k3 2, flash 1, deepseek-v4-flash 1) and always looks like a total
    # collapse (0 score, chain truncated). Without this guard it hijacks the
    # opus5 slot with a 4-step infra event on a task its peers solve, which would
    # illustrate the harness rather than the model.
    STALL = "generation_stall_empty_reply"

    picks = []
    used = set()
    for key, name, label, keyfn in specs:
        best = None
        for tid, t in stats[key]["by_task"].items():
            if tid in used:
                continue
            if str(t["stop"]).startswith(STALL):
                continue
            if any(stats[k]["by_task"].get(tid) is None for _, _, k in MODELS):
                continue
            sc = keyfn(t, tid)
            if best is None or sc > best[0]:
                best = (sc, tid)
        if best:
            used.add(best[1])
            picks.append((key, name, label, best[1]))
    return picks


def render_signatures(stats):
    picks = pick_signatures(stats)
    if not picks:
        return ""
    out = ["<h2>Signature failure modes (one exemplar per model)</h2>",
           '<p class="h2sub">One round chosen to most sharply illustrate each model\u2019s '
           f"characteristic behavior. Same paired format as above \u2014 all "
           f"{_numword(len(MODELS))} models on the "
           "same task, dot overview plus expandable full command chains \u2014 with the model "
           "whose signature it demonstrates highlighted.</p>"]
    for _key, name, label, tid in picks:
        row = {k: stats[k]["by_task"][tid] for _, _, k in MODELS}
        out.append('<div class="pair sig">')
        out.append(f'<div class="sigcap"><b>{esc(name)}</b> \u2014 {label}</div>')
        out.append(f'<div class="ptitle">{esc(tid)}</div>')
        for nm, _, k in MODELS:
            t = row[k]
            verdict = "PASS" if t["success"] else "fail"
            vcls = "pass" if t["success"] else "failv"
            cap = (f'{t["steps"]} steps \u00b7 {t["n_open"]} opens \u00b7 '
                   f'{t["n_write"]} writes \u00b7 score {t["score"]:.2f} \u00b7 {esc(t["stop"])}')
            out.append(
                f'<div class="trow"><span class="tnm">{esc(nm)}</span>'
                f'{render_chain(t["chain"])}'
                f'<span class="verd {vcls}">{verdict}</span>'
                f'<span class="tcap">{cap}</span></div>')
        out.append('<details class="cmd"><summary>full command chains \u00b7 '
                   "verbatim actions, +score-delta, err flags, stdout on hover</summary>")
        for nm, _, k in MODELS:
            out.append(render_cmdlist(nm, row[k]))
        out.append("</details>")
        out.append("</div>")
    return "".join(out)


# ---------------------------------------------------------------------------
# Reasoning-effort dimension. The v8 tree runs every model at low / medium / high
# over the SAME 600 tasks, which makes the effort comparison paired: each task is
# its own control, so the right test is McNemar's on the discordant pairs, not a
# two-proportion z-test on the marginals. The panels below exist because the
# 9x3 table of pass rates -- the obvious way to show this -- hides the two things
# that actually matter: the SHAPE of each model's response (monotone? saturating?
# inverted?) and the MECHANISM behind it.
# ---------------------------------------------------------------------------

EFFORT_X = {"low": "low", "medium": "medium", "high": "high"}


def _ladder(group, anchor_x, anchor, top, bot, gap=31):
    """De-clutter a column of labels into a monotone vertical ladder.

    Same trick as _scatter_svg's inner ladder(), lifted to module scope so the slope
    panels can share it: each label starts at its data y, is pushed down until it
    clears the previous one by `gap`, then the whole stack is shifted back inside
    [top, bot] if it overflowed. Connectors are drawn separately."""
    group = sorted(group, key=lambda p: p["_y"])
    prev = -1e9
    for p in group:
        p["ly"] = max(p["_y"], prev + gap)
        p["lx"] = anchor_x
        p["anchor"] = anchor
        prev = p["ly"]
    if group:
        over = group[-1]["ly"] - bot
        if over > 0:
            for p in group:
                p["ly"] -= over
        under = top - group[0]["ly"]
        if under > 0:
            for p in group:
                p["ly"] += under
    return group


def _slope_svg(series, ytitle, yfmt, aria, zero_line=False, left_labels=True,
               floor=None):
    """Three-column slope chart: one polyline per model across low/medium/high.

    A slope chart is used rather than a grouped bar chart or a colored table because
    the question here is about a TREND, and a trend is a slope: the eye reads angle
    far faster than it reads three numbers in a row. Level is vertical position,
    effort response is the slope, and a rank change is a visible crossing -- three
    facts in one picture that a 9x3 table makes you compute yourself.

    series: [{name, key, ys: [lo, med, hi], solid: bool, note: str}]
      solid=False draws the line dashed, reserved for models whose low->high change
      does NOT survive the paired McNemar test -- so a flat-looking line and a
      statistically-flat line are visually distinguishable.
    left_labels=False drops the left ladder, used by the delta panel where every
      line starts at exactly 0 and a left ladder would be nine copies of "0.0"."""
    allv = [v for s in series for v in s["ys"]]
    lo, hi = min(allv), max(allv)
    pad = max((hi - lo) * 0.14, 0.5)
    ylo, yhi = lo - pad, hi + pad
    if zero_line:  # keep 0 comfortably inside the frame
        ylo, yhi = min(ylo, -pad), max(yhi, pad)
    if floor is not None:  # a percentage axis must not pad itself below its floor
        ylo = max(ylo, floor)

    W, H = 980, 520
    ml, mr, mt, mb = 196, 214, 40, 58
    pw, ph = W - ml - mr, H - mt - mb
    cols = [ml + pw * f for f in (0.04, 0.5, 0.96)]

    def sy(v):
        return mt + (1 - (v - ylo) / (yhi - ylo)) * ph

    P = [f'<svg viewBox="0 0 {W} {H}" class="scz" preserveAspectRatio="xMidYMid meet" '
         f'role="img" aria-label="{esc(aria)}">',
         f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="#fbfcfe" '
         f'stroke="#e5e7eb"/>']
    # effort columns
    for cx, eff in zip(cols, EFFORTS):
        P.append(f'<line x1="{cx:.1f}" y1="{mt}" x2="{cx:.1f}" y2="{mt+ph}" '
                 f'stroke="#e8ecf1" stroke-width="1"/>')
        P.append(f'<text x="{cx:.1f}" y="{mt+ph+20}" text-anchor="middle" '
                 f'fill="#4b5563" font-size="12.5" font-weight="600" '
                 f'font-family="system-ui,sans-serif">{esc(EFFORT_X[eff])}</text>')
    if zero_line:
        z = sy(0.0)
        P.append(f'<line x1="{ml}" y1="{z:.1f}" x2="{ml+pw}" y2="{z:.1f}" '
                 f'stroke="#9ca3af" stroke-width="1.2" stroke-dasharray="5 4"/>')
        # Caption parked in the empty top-left corner. Neither end of the zero line
        # is free: the left is where all nine series originate and the right is where
        # the label ladder's connectors fan in.
        P.append(f'<text x="{ml+10}" y="{mt+15}" fill="#5b6472" font-size="10.5" '
                 f'font-family="system-ui,sans-serif">'
                 f'dashed baseline = each model\u2019s own low-effort score</text>')
    # y ticks at the frame corners
    tk = 'fill="#5b6472" font-size="11" font-family="system-ui,sans-serif"'
    P.append(f'<text x="{ml-8}" y="{mt+11}" text-anchor="end" {tk}>{yfmt(yhi)}</text>')
    P.append(f'<text x="{ml-8}" y="{mt+ph}" text-anchor="end" {tk}>{yfmt(ylo)}</text>')
    at = 'fill="#5b6472" font-size="12" font-family="system-ui,sans-serif"'
    cy = mt + ph / 2
    P.append(f'<text x="15" y="{cy:.1f}" text-anchor="middle" {at} '
             f'transform="rotate(-90 15 {cy:.1f})">{esc(ytitle)} \u2192</text>')
    P.append(f'<text x="{ml+pw:.0f}" y="{H-8}" text-anchor="end" {at}>'
             f'reasoning effort \u2192</text>')

    for s in series:
        s["pts"] = [(cols[i], sy(s["ys"][i])) for i in range(3)]
    # lines first, then dots, then labels: dots must sit on top of every stroke
    for s in series:
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in s["pts"])
        dash = "" if s["solid"] else ' stroke-dasharray="6 5"'
        P.append(f'<polyline points="{d}" fill="none" stroke="{DOTCOLORS[s["key"]]}" '
                 f'stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"'
                 f'{dash} opacity="0.9"/>')
    for s in series:
        for (x, y), eff, v in zip(s["pts"], EFFORTS, s["ys"]):
            P.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.6" '
                     f'fill="{DOTCOLORS[s["key"]]}" stroke="#fff" stroke-width="1.8">'
                     f'<title>{esc(s["name"])} @ {eff}: {yfmt(v)}</title></circle>')

    nm = 'font-size="12" font-weight="600" fill="#374151" font-family="system-ui,sans-serif"'
    co = 'font-size="10.5" fill="#5b6472" font-family="system-ui,sans-serif"'

    def label(p, side, head, sub):
        """One ladder label: connector, color swatch, bold head, grey sub-line.

        The swatch is what makes nine crossing lines readable -- it ties the label
        to its polyline by color without the reader having to trace the line back
        across the frame or match a hue against a legend at the top of the panel."""
        s = p["s"]
        i = 0 if side == "end" else 2
        px, py = s["pts"][i]
        tip = p["lx"] + (10 if side == "end" else -10)  # stop at the swatch edge
        out = [f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{tip:.1f}" y2="{p["ly"]:.1f}" '
               f'stroke="#d1d5db" stroke-width="1"/>']
        # swatch sits just outside the text, on the side facing the plot
        swx = p["lx"] + (4 if side == "end" else -10)
        out.append(f'<rect x="{swx:.1f}" y="{p["ly"]-11:.1f}" width="6" height="6" '
                   f'rx="1.5" fill="{DOTCOLORS[s["key"]]}"/>')
        out.append(f'<text x="{p["lx"]:.1f}" y="{p["ly"]-5:.1f}" text-anchor="{side}" '
                   f'{nm}>{esc(head)}</text>'
                   f'<text x="{p["lx"]:.1f}" y="{p["ly"]+8:.1f}" text-anchor="{side}" '
                   f'{co}>{esc(sub)}</text>')
        return "".join(out)

    if left_labels:
        L = [{"_y": s["pts"][0][1], "s": s} for s in series]
        _ladder(L, ml - 16, "end", mt + 14, mt + ph - 4)
        for p in L:
            P.append(label(p, "end", p["s"]["name"], yfmt(p["s"]["ys"][0])))
    R = [{"_y": s["pts"][2][1], "s": s} for s in series]
    _ladder(R, ml + pw + 16, "start", mt + 14, mt + ph - 4)
    for p in R:
        s = p["s"]
        # With a left ladder the name is already on screen, so the right head leads
        # with the value; without one it has to carry the name.
        head = f'{yfmt(s["ys"][2])}  {s["name"]}' if left_labels else s["name"]
        P.append(label(p, "start", head, s["note"]))
    P.append("</svg>")
    return f'<div class="sczbox">{"".join(P)}</div>'


def _budget_svg(rows, cap, aria):
    """Step-budget headroom chart: where each model sits against the hard step wall.

    This replaced a Delta-cap-rate x Delta-pass-rate scatter, which was the obvious
    way to plot the mechanism and the wrong one: one model moves its cap rate by
    +50 pp and the other eight by under +23, so the outlier set the x-scale, packed
    the rest into the left edge, and stacked eight labels into one unreadable ladder.
    It also implied a tidy negative correlation the data does not support (a model
    can pay a 20 pp cap tax and still net a gain).

    The quantity that actually predicts whether effort helps is HEADROOM -- how much
    of the 20-step budget the model has left before the wall -- and that is an
    absolute position on a fixed 0..cap axis, so it plots without distortion. Each
    row: the upper bar spans mean steps at low effort to mean steps at high effort
    (what effort costs this model), and the lower red bar is the share of its
    high-effort episodes that hit the wall (what that cost it).

    rows: [{name, key, lo, med, hi, capped_lo, capped_hi, delta_pp, solid}]"""
    W = 980
    rh, top, bot = 46, 62, 46
    H = top + rh * len(rows) + bot
    nx, x0 = 150, 172           # name column right edge, track left edge
    x1 = W - 196                # track right edge (room for the delta column)
    tw = x1 - x0

    def sx(v):
        return x0 + (v / cap) * tw

    P = [f'<svg viewBox="0 0 {W} {H}" class="scz" preserveAspectRatio="xMidYMid meet" '
         f'role="img" aria-label="{esc(aria)}">']
    tk = 'fill="#5b6472" font-size="10.5" font-family="system-ui,sans-serif"'
    for g in range(0, cap + 1, 5):
        gx = sx(g)
        P.append(f'<line x1="{gx:.1f}" y1="{top-14}" x2="{gx:.1f}" '
                 f'y2="{top + rh*len(rows) - 6}" stroke="#eef2f7" stroke-width="1"/>')
        if g != cap:  # the cap tick is labelled in red below, not twice
            P.append(f'<text x="{gx:.1f}" y="{top-20}" text-anchor="middle" {tk}>'
                     f'{g}</text>')
    P.append(f'<text x="{x0}" y="{top-38}" {tk}>steps per task</text>')
    # the wall
    P.append(f'<line x1="{sx(cap):.1f}" y1="{top-14}" x2="{sx(cap):.1f}" '
             f'y2="{top + rh*len(rows) - 6}" stroke="#dc2626" stroke-width="1.6"/>')
    P.append(f'<text x="{sx(cap):.1f}" y="{top-20}" text-anchor="middle" fill="#dc2626" '
             f'font-size="10.5" font-weight="700" '
             f'font-family="system-ui,sans-serif">{cap} = cap</text>')

    nm = 'font-size="12" font-weight="600" fill="#374151" font-family="system-ui,sans-serif"'
    co = 'font-size="10" fill="#5b6472" font-family="system-ui,sans-serif"'
    for i, r in enumerate(rows):
        y = top + i * rh
        col = DOTCOLORS[r["key"]]
        P.append(f'<rect x="{x0}" y="{y-1}" width="{tw}" height="14" fill="#fbfcfe" '
                 f'stroke="#f1f5f9"/>')
        P.append(f'<rect x="{nx + 6:.1f}" y="{y+3}" width="6" height="6" rx="1.5" '
                 f'fill="{col}"/>')
        P.append(f'<text x="{nx}" y="{y+12}" text-anchor="end" {nm}>'
                 f'{esc(r["name"])}</text>')
        # span low -> high
        a, b = sx(r["lo"]), sx(r["hi"])
        P.append(f'<rect x="{a:.1f}" y="{y+2}" width="{max(b-a, 1.5):.1f}" height="8" '
                 f'rx="4" fill="{col}" opacity="0.28"/>')
        for v, rr, fill in ((r["lo"], 4.2, "#ffffff"), (r["med"], 3.2, col),
                            (r["hi"], 4.8, col)):
            P.append(f'<circle cx="{sx(v):.1f}" cy="{y+6}" r="{rr}" fill="{fill}" '
                     f'stroke="{col}" stroke-width="1.6">'
                     f'<title>{esc(r["name"])}: {v:.2f} steps</title></circle>')
        P.append(f'<text x="{sx(r["hi"])+9:.1f}" y="{y+10}" {co}>{r["hi"]:.1f}</text>')
        # Capped share at high effort. The bar is 10px rather than a hairline so that
        # the white in-bar caption below fits INSIDE it: a 10px glyph on a 6px bar
        # spills its ascenders onto the near-white track and goes invisible.
        cw = max(tw * r["capped_hi"] / 100.0, 0.0)
        P.append(f'<rect x="{x0}" y="{y+16}" width="{tw}" height="10" rx="5" '
                 f'fill="#f8fafc" stroke="#f1f5f9"/>')
        if cw > 0:
            # Full opacity, not 0.72: the softened bar blends to ~#e46162 against
            # the track, and the white in-bar caption below only reaches 3.40:1
            # on that. Solid #dc2626 puts it at 4.83:1, just over AA.
            P.append(f'<rect x="{x0}" y="{y+16}" width="{cw:.1f}" height="10" rx="5" '
                     f'fill="#dc2626"><title>{esc(r["name"])}: '
                     f'{r["capped_hi"]:.1f}% of high-effort episodes hit the cap'
                     f'</title></rect>')
        # A long red bar leaves no room to its right, so its caption moves inside the
        # bar in white rather than running over the cap wall.
        cap_txt = f'{r["capped_lo"]:.1f}% \u2192 {r["capped_hi"]:.1f}% capped'
        if cw > tw * 0.55:
            P.append(f'<text x="{x0+9:.1f}" y="{y+24}" fill="#ffffff" font-size="9.5" '
                     f'font-weight="700" font-family="system-ui,sans-serif">'
                     f'{esc(cap_txt)}</text>')
        else:
            P.append(f'<text x="{x0 + cw + 7:.1f}" y="{y+24}" {co}>{esc(cap_txt)}</text>')
        # net outcome
        d = r["delta_pp"]
        dc = "#047857" if d > 0 else ("#b91c1c" if d < 0 else "#5b6472")
        mark = "\u25cf" if r["solid"] else "\u25cb"
        P.append(f'<text x="{W-14}" y="{y+12}" text-anchor="end" fill="{dc}" '
                 f'font-size="12.5" font-weight="700" '
                 f'font-family="system-ui,sans-serif">{d:+.1f} pp</text>')
        P.append(f'<text x="{W-14}" y="{y+24}" text-anchor="end" {co}>{mark} pass '
                 f'lo\u2192hi</text>')
    P.append(f'<text x="{x0}" y="{H-16}" {co}>\u25cb low \u00b7 \u2022 medium \u00b7 '
             f'\u25cf high mean steps \u00b7 red bar = share of high-effort episodes '
             f'stopped by the cap</text>')
    P.append("</svg>")
    return f'<div class="sczbox">{"".join(P)}</div>'


def mcnemar_p(b, c):
    """Exact two-sided binomial test on the discordant pairs (b vs c).

    b = tasks the model passed at the LOW level and lost at HIGH; c = the reverse.
    Concordant tasks (passed at both, failed at both) carry no information about a
    change and are correctly ignored. Exact rather than chi-square because several
    models here have single-digit discordant counts, where the chi-square
    approximation is not trustworthy."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


# Fields lifted off every episode_header, and what the report says about each.
_COND_FIELDS = ("model", "provider", "multimodal_input_mode",
                "multimodal_input_mode_effective", "reasoning_effort",
                "reasoning_effort_effective", "max_image_edge",
                "tool_mode", "tool_schema_mode", "native_tool_choice",
                "history_mode", "gemini_media_resolution")

# Fields that identify the row rather than describe how it was run: excluded
# from the "identical across the field" sentence, where saying that `model`
# differs between models would be a tautology.
_COND_IDENTITY = ("model", "provider")


def scan_run_conditions(root):
    """Read the run_config of every analysis cell straight off its log headers.

    WHY THIS EXISTS. Every run-condition the report states -- which models were
    sent images, which reached `high` through a side channel, which ran at a
    smaller image cap, what each model's API id even is -- was a hand-maintained
    constant in this file, verified once by eye. The comments above
    OPEN_SOURCE_CAVEAT record TWO occasions on which those constants were wrong
    and shipped ("DO NOT SAY THESE TWO EXPOSE NO EFFORT KNOB -- an earlier
    version of this comment did"; "DO NOT WRITE max_image_edge=2576 for every
    model -- an earlier version of the report header did"). A claim class with a
    two-for-two error rate under hand maintenance should not be hand-maintained.

    It is affordable because it stops at the first `episode_header` in each
    file, which sits at the top: ~1,980 partial reads rather than a second pass
    over ~36 MB per cell.

    Returns {model_key: {field: value}} for the PRIMARY cell of each model, and
    raises if a field is not constant across that cell's 60 files -- a cell that
    changed configuration mid-run is not a cell any single row can describe.
    """
    out = {}
    for _, subdir, key in MODELS:
        d = os.path.join(root, "eval_results", tree(subdir, key=key),
                         run_for(key), "log")
        seen = {f: set() for f in _COND_FIELDS}
        n_files = 0
        for fp in sorted(glob.glob(os.path.join(d, "*.interactions.jsonl"))):
            for line in open(fp):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("record_type") != "episode_header":
                    continue
                rc = r.get("run_config") or {}
                for f in _COND_FIELDS:
                    seen[f].add(rc.get(f))
                n_files += 1
                break
        if not n_files:
            raise SystemExit(f"scan_run_conditions: no episode_header for {key} "
                             f"under {d}")
        split = {f: v for f, v in seen.items() if len(v) > 1}
        if split:
            raise SystemExit(
                f"scan_run_conditions: {key} is not uniform over its "
                f"{n_files} files: "
                + "; ".join(f"{f}={sorted(map(str, v))}"
                            for f, v in sorted(split.items())))
        out[key] = {f: next(iter(v)) for f, v in seen.items()}
        out[key]["n_headers"] = n_files
    _check_conditions_match_constants(out)
    return out


def _check_conditions_match_constants(cond):
    """Fail the build if the measured conditions contradict OPEN_SOURCE_CAVEAT.

    The caveat strings stay hand-written -- they are prose, and prose is the
    thing worth writing by hand -- but the FACTS inside them are now checked
    against the corpus on every build. This is the guard the two historical
    errors would each have tripped.
    """
    bad = []
    path_measured = {k for k, c in cond.items()
                     if c["multimodal_input_mode"] == "path"}
    path_claimed = {k for k, v in OPEN_SOURCE_CAVEAT.items()
                    if "multimodal_input_mode=path" in v}
    if path_measured != path_claimed:
        bad.append(f"path-mode models: measured {sorted(path_measured)}, "
                   f"OPEN_SOURCE_CAVEAT claims {sorted(path_claimed)}")

    # Every model the report places in the primary cell must actually be there,
    # by one channel or the other. The two empty-field models carry it in
    # chat_template_kwargs, which is why an empty reasoning_effort is allowed
    # ONLY for models the caveat already documents.
    for k, c in sorted(cond.items()):
        eff = c["reasoning_effort_effective"] or c["reasoning_effort"]
        if eff == PRIMARY_EFFORT:
            continue
        if not eff and k in OPEN_SOURCE_CAVEAT:
            continue
        bad.append(f"{k}: effective reasoning_effort {eff!r} is not "
                   f"{PRIMARY_EFFORT!r} and it carries no documented caveat")

    edges = sorted({c["max_image_edge"] for c in cond.values()})
    off_edge = {k for k, c in cond.items()
                if c["max_image_edge"] != max(edges, key=lambda e: int(e or 0))}
    if off_edge and not off_edge <= set(OPEN_SOURCE_CAVEAT):
        bad.append(f"models at a non-majority max_image_edge {sorted(off_edge)} "
                   f"are not all covered by OPEN_SOURCE_CAVEAT")
    if bad:
        raise SystemExit("measured run conditions contradict the constants:\n  - "
                         + "\n  - ".join(bad))


def load_effort_grid(root):
    """Load the 10 x 3 (model x effort) grid and derive the paired effort statistics.

    Only the fields the effort panels need are kept -- pass/fail, cap hit, steps,
    score -- so this stays cheap enough to run alongside the full analysis of the
    primary level (each cell is ~36 MB of trajectory jsonl).

    SWEEP_MODELS, not MODELS: the three FIXED_TREE (open-source) models were run at a
    single setting and have no aligned_setting_{low,medium,high} directories at all,
    so including them here would hit the `no records` guard below and abort the whole
    report. Note this deliberately calls tree() WITHOUT `key`, because composing the
    effort level in is exactly the point of this grid."""
    cells = {}
    for _name, subdir, key in SWEEP_MODELS:
        for eff in EFFORTS:
            # run_for, not DEFAULT_RUN: the primary column of this grid is the
            # same cell the main tables report, so if a model is promoted to a
            # rerun there it must be promoted here too. Letting the two diverge
            # would print two different pass rates for one (model, effort) cell
            # in one document.
            d = os.path.join(root, "eval_results", tree(subdir, eff),
                             run_for(key, eff))
            succ, capped = {}, {}
            steps, scores, pf = [], [], 0
            for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
                for line in open(f):
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    tid = r.get("task_id")
                    succ[tid] = bool(r.get("success"))
                    capped[tid] = "max_steps" in str(r.get("stop_reason") or "")
                    # Same repair as analyze(): a provider-killed episode reports
                    # steps=0 while its trajectory still holds the actions the
                    # model took. Averaging the raw field here while analyze()
                    # averages the repaired one would print two different mean
                    # chain lengths for the same cell in the same document.
                    st = r.get("steps") or 0
                    if r.get("error") and st != len(r.get("trajectory") or []):
                        st = len(r.get("trajectory") or [])
                    steps.append(st)
                    scores.append(float(r.get("score") or 0))
                    # ``error``, NOT ``provider_impacted``: the latter is a
                    # disjoint flag that is False on every episode the provider
                    # actually killed, so keying this counter on it reported
                    # zero for the only cells that lost episodes. See analyze().
                    pf += 1 if r.get("error") else 0
            n = len(succ)
            if not n:
                raise SystemExit(f"no records for {tree(subdir, eff)}/{DEFAULT_RUN}")
            cells[(key, eff)] = {
                "n": n,
                "passed": sum(succ.values()),
                "pass_pct": round(100 * sum(succ.values()) / n, 1),
                "capped": sum(capped.values()),
                "capped_pct": round(100 * sum(capped.values()) / n, 1),
                "mean_steps": round(sum(steps) / n, 2),
                "avg_score": round(sum(scores) / n, 4),
                "provider_failures": pf,
                "_succ": succ,
                "_capped": capped,
            }

    # Whole-grid check: does hitting the step cap ever coexist with a pass?
    cap_n = cap_p = unc_n = unc_p = 0
    for cell in cells.values():
        for tid, ok in cell["_succ"].items():
            if cell["_capped"][tid]:
                cap_n += 1
                cap_p += ok
            else:
                unc_n += 1
                unc_p += ok

    mcn = {}
    for _name, _subdir, key in SWEEP_MODELS:  # only the swept models have low/high
        lo, hi = cells[(key, "low")]["_succ"], cells[(key, "high")]["_succ"]
        ts = set(lo) & set(hi)
        b = sum(1 for t in ts if lo[t] and not hi[t])
        c = sum(1 for t in ts if hi[t] and not lo[t])
        p = mcnemar_p(b, c)
        mcn[key] = {"paired_n": len(ts), "lost": b, "gained": c, "p": p,
                    "significant": p < 0.05,
                    "delta_pp": round(cells[(key, "high")]["pass_pct"]
                                      - cells[(key, "low")]["pass_pct"], 1)}
    return {
        "cells": cells,
        "mcnemar": mcn,
        "cap_pass": {"capped_n": cap_n, "capped_passed": cap_p,
                     "uncapped_n": unc_n, "uncapped_passed": unc_p},
    }


def render_effort(grid):
    """The reasoning-effort chapter: three panels plus the audit table."""
    C, M = grid["cells"], grid["mcnemar"]
    cp = grid["cap_pass"]
    # SWEEP_MODELS: this whole chapter is a low/medium/high comparison, and the three
    # open-source models exist at one setting only. They are absent from every panel,
    # every mean and the McNemar table by construction -- see EFFORT_SCOPE_NOTE, which
    # is rendered into the chapter so the omission is stated rather than silent.
    order = sorted(SWEEP_MODELS, key=lambda m: -C[(m[2], PRIMARY_EFFORT)]["pass_pct"])

    def series(valfn, notefn):
        return [{"name": name, "key": key,
                 "ys": [valfn(C[(key, e)]) for e in EFFORTS],
                 "solid": M[key]["significant"], "note": notefn(key)}
                for name, _s, key in order]

    n_sig = sum(1 for _n, _s, k in order if M[k]["significant"])
    mean_by_eff = {e: sum(C[(k, e)]["pass_pct"] for _n, _s, k in order) / len(order)
                   for e in EFFORTS}
    unc_rate = 100 * cp["uncapped_passed"] / max(cp["uncapped_n"], 1)
    # "raising effort makes every model take more steps" was asserted as "all nine,
    # monotonically" -- a hardcoded count on a field that is now ten swept models.
    # Count it instead, and name the exceptions if the unanimity ever breaks, so the
    # sentence cannot outlive the fact.
    _mono = [(n, k) for n, _s, k in order
             if C[(k, "low")]["mean_steps"] <= C[(k, "medium")]["mean_steps"]
             <= C[(k, "high")]["mean_steps"]]
    n_mono = len(_mono)
    if n_mono == len(order):
        mono_txt = f"all {_numword(len(order))}, monotonically"
    else:
        _bad = [n for n, _s, k in order if (n, k) not in _mono]
        mono_txt = (f"{n_mono} of {len(order)} monotonically, the exception"
                    + ("s " if len(_bad) > 1 else " ")
                    + esc(", ".join(_bad)))

    def sigmark(k):
        return "\u25cf" if M[k]["significant"] else "\u25cb"

    # --- Panel A: absolute pass rate ---------------------------------------
    sA = series(lambda c: c["pass_pct"],
                lambda k: f'{sigmark(k)} {M[k]["delta_pp"]:+.1f} pp')
    panelA = _slope_svg(sA, "pass@budget (%)", lambda v: f"{v:.1f}%",
                        f"pass rate by reasoning effort, {len(order)} models",
                        floor=0.0)
    # --- Panel B: response shape, each model against its own low ------------
    sB = [{"name": s["name"], "key": s["key"],
           "ys": [round(v - s["ys"][0], 1) for v in s["ys"]],
           "solid": s["solid"], "note": s["note"]} for s in sA]
    sB.sort(key=lambda s: -s["ys"][2])
    panelB = _slope_svg(sB, "\u0394 pass@budget vs own low (pp)",
                        lambda v: f"{v:+.1f}",
                        "change in pass rate relative to each model's own low-effort run",
                        zero_line=True, left_labels=False)
    # --- Panel C: the mechanism --------------------------------------------
    brows = sorted(
        [{"name": name, "key": key, "solid": M[key]["significant"],
          "lo": C[(key, "low")]["mean_steps"], "med": C[(key, "medium")]["mean_steps"],
          "hi": C[(key, "high")]["mean_steps"],
          "capped_lo": C[(key, "low")]["capped_pct"],
          "capped_hi": C[(key, "high")]["capped_pct"],
          "delta_pp": M[key]["delta_pp"]} for name, _s, key in order],
        key=lambda r: r["hi"])
    panelC = _budget_svg(brows, STEP_CAP,
                         "mean steps per task at each effort level against the "
                         "20-step budget")
    monotone = sum(1 for r in brows if r["lo"] < r["med"] < r["hi"])

    # --- audit table --------------------------------------------------------
    def tri(key, field, fmt):
        return "".join(f'<td>{fmt(C[(key, e)][field])}</td>' for e in EFFORTS)

    head = ('<tr><th rowspan="2">model</th>'
            '<th colspan="3">pass@budget %</th>'
            '<th rowspan="2">\u0394<br>lo\u2192hi</th>'
            '<th colspan="3">mean steps / task</th>'
            '<th colspan="3">step-capped %</th>'
            '<th colspan="2">discordant</th>'
            '<th rowspan="2">McNemar<br>p</th>'
            '<th rowspan="2">verdict</th></tr>'
            '<tr>' + "".join(f'<th class="bsub2">{e}</th>' for e in EFFORTS)
            + "".join(f'<th class="bsub2">{e}</th>' for e in EFFORTS)
            + "".join(f'<th class="bsub2">{e}</th>' for e in EFFORTS)
            + '<th class="bsub2" title="passed at low, failed at high">lost</th>'
              '<th class="bsub2" title="failed at low, passed at high">gained</th></tr>')
    rows = []
    for name, _s, key in order:
        m = M[key]
        d = m["delta_pp"]
        dcls = "up" if d > 0 else ("down" if d < 0 else "")
        pv = m["p"]
        pstr = "&lt; 1e-6" if pv < 1e-6 else f"{pv:.3g}"
        vcls, vtxt = (("ex0", "real") if m["significant"] else ("exn", "n.s."))
        rows.append(
            f'<tr><td class="exm">{esc(name)}</td>'
            + tri(key, "pass_pct", lambda v: f"{v:.1f}")
            + f'<td class="{dcls}">{d:+.1f}</td>'
            + tri(key, "mean_steps", lambda v: f"{v:.1f}")
            + tri(key, "capped_pct", lambda v: f"{v:.1f}")
            + f'<td>{m["lost"]}</td><td>{m["gained"]}</td>'
            + f'<td>{pstr}</td><td class="{vcls}">{vtxt}</td></tr>')
    mean_row = ('<tr class="famrow"><td>field mean</td>'
                + "".join(f'<td>{mean_by_eff[e]:.1f}</td>' for e in EFFORTS)
                + f'<td>{mean_by_eff["high"] - mean_by_eff["low"]:+.1f}</td>'
                + '<td colspan="8"></td>'
                + f'<td>{n_sig} of {len(order)} real</td></tr>')

    return (
        '<h2 id="effort">Reasoning effort: a weak lever with a sharp edge</h2>'
        '<p class="h2sub">Every model ran the same '
        f'{C[(order[0][2], "low")]["n"]} tasks three times, at '
        '<code>reasoning_effort</code> low, medium and high. Because the task set is '
        'identical across levels the comparison is <b>paired</b> \u2014 each task is its '
        'own control \u2014 so significance is judged with an exact McNemar test on the '
        'discordant pairs (tasks whose outcome actually changed), not with a '
        'two-proportion test on the marginals.</p>'

        f'<div class="callout"><b>Averaged over the field, effort buys almost nothing '
        f'\u2014 {mean_by_eff["low"]:.1f}% \u2192 {mean_by_eff["medium"]:.1f}% \u2192 '
        f'{mean_by_eff["high"]:.1f}%, a '
        f'{mean_by_eff["high"] - mean_by_eff["low"]:+.1f} pp swing across the whole '
        'range.</b> That average is misleading in both directions. Per model the '
        f'low\u2192high change runs from <b>{min(M[k]["delta_pp"] for _n, _s, k in order):+.1f} pp</b> '
        f'to <b>{max(M[k]["delta_pp"] for _n, _s, k in order):+.1f} pp</b>, and only '
        f'<b>{n_sig} of {len(order)}</b> models move at all once the paired test is '
        'applied. The mechanism is visible in the third panel: raising effort makes '
        f'every model take more steps \u2014 {mono_txt} \u2014 and steps are '
        'a finite resource here. An episode that exhausts the 20-step budget is '
        f'graded as it stands, and across all {len(order) * len(EFFORTS)} cells '
        f'<b>{cp["capped_passed"]} of {cp["capped_n"]}</b> step-capped episodes '
        f'passed, against {unc_rate:.1f}% of the {cp["uncapped_n"]:,} that '
        'terminated on their own. Effort therefore buys reasoning and spends '
        'termination budget at the same time; the net slope is which effect won.</div>'

        '<h3>Panel A \u00b7 level and trend together</h3>'
        '<p class="h2sub">Vertical position is the pass rate, the slope is the effort '
        'response, and a crossing is a rank change. Lines are <b>solid</b> where the '
        'low\u2192high change survives the paired test at \u03b1 = 0.05 and <b>dashed</b> '
        'where it does not, so a line that merely looks flat can be told apart from '
        'one that is statistically flat. Right-hand labels carry the signed change and '
        'the same marker (\u25cf real \u00b7 \u25cb not significant).</p>'
        + panelA +

        '<h3>Panel B \u00b7 response shape, each model against itself</h3>'
        '<p class="h2sub">The same data with every model\u2019s own low-effort score '
        f'subtracted, so all {_numword(len(order))} lines start at zero and only the '
        '<i>shape</i> of the '
        'response is left. This is the panel that separates a model which keeps '
        'improving from one that saturates at medium, from one that peaks and falls '
        'back, from one that degrades outright \u2014 four distinct shapes that Panel A '
        'compresses into a band and a table hides completely.</p>'
        + panelB +

        '<h3>Panel C \u00b7 why: effort is paid for in steps</h3>'
        '<p class="h2sub">The upper bar spans each model\u2019s mean steps per task at '
        f'low effort through high effort \u2014 what raising effort <i>costs</i> it \u2014 '
        f'plotted against the hard {STEP_CAP}-step budget. All '
        f'{monotone} of {len(brows)} models move rightward monotonically: more effort '
        'is more steps, without exception. The red bar underneath is the share of that '
        'model\u2019s high-effort episodes that ran into the wall, and since a capped '
        f'episode has never once passed here ({cp["capped_passed"]} of '
        f'{cp["capped_n"]} across the whole grid), that share is a straight deduction '
        'from the pass rate. Read the rows top to bottom: models with headroom bank '
        'the reasoning and pay almost nothing, and models already parked against the '
        'cap have nothing left to spend.</p>'
        + panelC +

        '<h3>Every number behind the three panels</h3>'
        '<p class="h2sub"><b>lost</b> = tasks passed at low and failed at high; '
        '<b>gained</b> = the reverse. These two counts are the entire evidence for the '
        'change: tasks with the same outcome at both levels are uninformative about it '
        'and are excluded by construction. p is the exact two-sided binomial '
        'probability of a split at least this lopsided under the null that effort does '
        'nothing.</p>'
        f'<table class="cmdtable extbl efftbl"><thead>{head}</thead>'
        f'<tbody>{"".join(rows)}{mean_row}</tbody></table>')


CSS = """
:root{color-scheme:light}
*{box-sizing:border-box}
body{font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif;
 margin:0;background:#ffffff;color:#1f2937}
.wrap{max-width:1040px;margin:0 auto;padding:28px 22px 60px}
h1{font-size:24px;margin:0 0 4px;color:#111827}
h2{font-size:17px;margin:34px 0 12px;border-bottom:1px solid #e5e7eb;padding-bottom:6px;color:#111827}
.sub{color:#5b6472;font-size:12.5px;margin:2px 0}
.sub a{color:#2563eb}
.callout{background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:12px 14px;margin:16px 0;font-size:13px;color:#1f2937}
.h2sub{color:#5b6472;font-size:12.5px;margin:-4px 0 12px}
.h2sub a{color:#2563eb}
.legend{display:flex;flex-wrap:wrap;gap:10px 16px;margin:10px 0 14px;font-size:12px;color:#4b5563}
.lg{display:inline-flex;align-items:center;gap:6px}
.lgd{width:12px;height:12px;border-radius:3px;display:inline-block}
.ring{width:11px;height:11px;border-radius:50%;display:inline-block;box-shadow:0 0 0 2px #ef4444;background:#e5e7eb}
.aggrow{display:flex;align-items:center;gap:10px;margin:7px 0}
.nm{width:130px;font-size:12.5px;color:#374151;text-align:right;flex:none}
.cnt{font-size:11.5px;color:#5b6472;flex:none}
.catbar{flex:1;height:20px;border-radius:5px;overflow:hidden;display:flex;background:#eef2f7}
.seg{height:100%}
.sczbox{margin:14px 0 26px;width:100%}
.scz{display:block;width:100%;height:auto;max-width:960px;margin:0 auto}
.maplegend{display:flex;flex-wrap:wrap;gap:8px 16px;margin:10px 0 4px;font-size:12px;color:#4b5563}
.pair{margin:16px 0;padding:10px 12px;background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px}
.ptitle{font:12.5px ui-monospace,Menlo,Consolas,monospace;color:#1d4ed8;margin-bottom:6px}
.sig{border:1px solid #bfdbfe;background:#f5f9ff}
.sigcap{font-size:12px;color:#1f2937;margin:2px 0 8px;padding:6px 9px;background:#eff6ff;
 border-left:3px solid #2563eb;border-radius:4px}
.sigcap b{color:#1d4ed8}
.trow{display:flex;align-items:center;gap:9px;margin:4px 0;flex-wrap:wrap}
.tnm{width:120px;font-size:12px;color:#374151;text-align:right;flex:none}
.chain{display:flex;gap:4px;align-items:center;flex:none}
.dot{width:13px;height:13px;border-radius:50%;display:inline-block}
.dot.err{box-shadow:0 0 0 2px #ef4444}
.verd{font-size:10.5px;font-weight:700;padding:1px 6px;border-radius:4px;flex:none}
.verd.pass{background:#d1fae5;color:#065f46}
.verd.failv{background:#fee2e2;color:#991b1b}
.tcap{font-size:11px;color:#5b6472}
.foot{margin-top:40px;color:#5b6472;font-size:11.5px;border-top:1px solid #e5e7eb;padding-top:12px}
/* command-surface heatmap */
.cmdtable{border-collapse:collapse;width:100%;margin:8px 0 6px;font-size:12.5px}
.cmdtable th{text-align:right;color:#5b6472;font-weight:600;padding:5px 9px;border-bottom:1px solid #e5e7eb}
.cmdtable th:first-child{text-align:left}
.cmdtable td{padding:4px 9px;text-align:right;border-bottom:1px solid #f1f5f9}
.cmdtable td.ck{text-align:left}
/* Failure-shape chart (observed minus the model's own independence baseline).
   Every number is real text, not a title= tooltip: the previous version encoded
   the whole histogram in bar widths with the percentages reachable only on hover,
   so that column was blank in print, in PDF and on copy-paste. */
.dvt td.dv{text-align:center;padding:5px 6px;vertical-align:top}
.dvt th{text-align:center}
.dvt th:first-child,.dvt td:first-child{text-align:left}
.dvn{display:block;font-size:12.5px;font-weight:700;font-variant-numeric:tabular-nums}
.dvn.pos{color:#4338ca}
.dvn.neg{color:#b45309}
.dvn.zero{color:#5b6472}
.dvb{display:block;position:relative;width:64px;height:9px;margin:3px auto 2px;
 background:#f8fafc;border-radius:2px}
.dvb::before{content:"";position:absolute;left:50%;top:-1px;bottom:-1px;width:1px;
 background:#cbd5e1}
.dvb i{position:absolute;top:1px;bottom:1px;border-radius:1px}
.dvb i.pos{left:50%;background:#4f46e5}
.dvb i.neg{right:50%;background:#d97706}
.dvo{display:block;font-size:10.5px;color:#5b6472;font-variant-numeric:tabular-nums}
/* Subordinate text token. The former #9ca3af is only ~2.5:1 on white and
   #6b7280 is ~4.3-4.4:1 on the tinted table fills (#eef2f7, #f5f3ff, #f1f5f9),
   so both fall below WCAG AA for the 10-12px sizes used here. #5b6472 clears
   4.5:1 on every background the report paints (worst case 5.32:1 on #eef2f7)
   while staying clearly lighter than the #111827/#0b1020 primary text.
   tools/audit_contrast.py enforces this. */
.bsub{font-size:10.5px;font-weight:400;color:#5b6472;margin-top:1px}
.extbl td,.extbl th{text-align:center;font-variant-numeric:tabular-nums}
.extbl td:first-child,.extbl th:first-child{text-align:left;white-space:nowrap}
.extbl td.exm{font-weight:600;color:#111827}
.extbl td.exk{color:#5b6472;font-size:11px}
.extbl td.ex0{background:#dcfce7;color:#166534;font-weight:700}
.extbl td.exn{color:#5b6472}
.extbl td.exall{background:#fee2e2;color:#991b1b;font-weight:700}
.extbl td.extot{font-weight:700;color:#1d4ed8;border-left:1px solid #e5e7eb}
.cmdtable td.dim,.cmdtable th.dim{color:#5b6472}
.cmdtable td.key,.cmdtable th.key{background:#f5f3ff}
/* one-seed lever: the row is ten rounds of a single scenario in three variants,
   not 30 independent trials. Flagged in the cell so the caveat travels with the
   number instead of living only in the footnote. */
.cmdtable td.thin{color:#b45309;font-weight:700}
.ci{color:#5b6472;font-size:11px;margin-left:5px;white-space:nowrap}
.up{color:#047857;font-weight:700;margin-left:3px}
.down{color:#b45309;font-weight:700;margin-left:3px}
.h2sub.warn{color:#7c2d12;background:#fff7ed;border-left:3px solid #fb923c;padding:7px 10px;margin:8px 0 12px}
.cmdtable td.ck code{background:none;color:#334155;font-size:12px}
.cmdtable .hc{color:#0b1020;font-variant-numeric:tabular-nums}
.cmdtable .hc b{color:#0b1020}
.famrow td{color:#2563eb;font-size:11px;text-transform:uppercase;letter-spacing:.06em;
 background:#f1f5f9;padding-top:8px!important}
.tblnote,.fignote{color:#5b6472;font-size:11.5px;line-height:1.6;margin:2px 0 12px}
.tblnote a,.fignote a{color:#2563eb}
.fignote{max-width:960px;margin-left:auto;margin-right:auto}
.csbul{font-size:13px;color:#334155;margin:8px 0;padding-left:14px;border-left:2px solid #bfdbfe}
/* Designed-difficulty axes: the signed pass-point delta under each rate.
   Its own class rather than .bsub, and those cells are deliberately NOT tinted.
   The obvious design -- a gold_shade fill plus a .bsub sub-line, exactly as the
   gold table does it -- fails audit_contrast.py outright: #5b6472 at 10.5px
   falls to 3.0:1 on the deep end of that indigo ramp, and every colour dark
   enough to clear 4.5:1 there is too dark to still read as subordinate. The
   absolute level is already a heatmap directly above, and this table exists for
   the DIRECTION of the delta, so the delta takes the colour and the fill goes. */
.dxd{font-size:10.5px;font-weight:700;margin-top:1px;font-variant-numeric:tabular-nums}
.dxd.pos{color:#047857}
.dxd.neg{color:#b45309}
.dxd.zero{color:#5b6472}
/* all-60 score heatmap */
.h27{border-collapse:collapse;width:100%;margin:8px 0;font-size:12px}
.h27 th{color:#5b6472;font-weight:600;padding:5px 8px;border-bottom:1px solid #e5e7eb;text-align:right}
.h27 th:first-child{text-align:left}
.h27 .scn{font:11.5px ui-monospace,Menlo,Consolas,monospace;color:#334155;white-space:nowrap;padding:3px 8px}
.h27 .sc{text-align:center;color:#0b1020;font-weight:600;padding:3px 8px;font-variant-numeric:tabular-nums}
/* per-scenario prose + full command chains */
.scennote{font-size:13px;line-height:1.55;color:#1f2937;margin:2px 0 12px;
 background:#f8fafc;border:1px solid #e5e7eb;border-radius:7px;padding:10px 12px}
.scennote code{background:#eef2f7;padding:0 4px;border-radius:3px;font-size:11.5px}
.scennote b{color:#111827}
details.cmd{margin:8px 0 2px;border:1px solid #e5e7eb;border-radius:7px;background:#f8fafc}
details.cmd>summary{cursor:pointer;padding:7px 11px;font-size:12px;color:#1d4ed8;user-select:none}
details.cmd[open]>summary{border-bottom:1px solid #e5e7eb}
.cmlist{padding:6px 12px 10px}
.cmnm{display:flex;align-items:center;gap:8px;margin:8px 0 3px;font-size:12.5px;color:#1f2937;font-weight:600}
.cmsc{font-size:11px;color:#5b6472;font-weight:400}
.cstep{display:flex;align-items:center;gap:8px;margin:2px 0;font-size:12px}
.cnum{width:20px;text-align:right;color:#5b6472;font-variant-numeric:tabular-nums;flex:none}
.cchip{width:10px;height:10px;border-radius:3px;flex:none}
.cact{font:12px/1.45 ui-monospace,Menlo,Consolas,monospace;color:#334155;background:none;
 white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:1;min-width:0}
.cd{font-size:10.5px;font-weight:700;color:#065f46;background:#d1fae5;padding:1px 5px;border-radius:4px;flex:none}
.cerr{font-size:10px;font-weight:700;color:#991b1b;background:#fee2e2;padding:1px 5px;border-radius:4px;flex:none}
/* reading guide (glossary), other-breakdown, command inventory, steps-vs-gold */
h3.oh{font-size:13.5px;margin:16px 0 6px;color:#374151;font-weight:600}
h3.oh code{background:#eef2f7;padding:0 4px;border-radius:3px}
.cmdtable td.sc{text-align:right;font-variant-numeric:tabular-nums}
.glossary td{vertical-align:top;padding:6px 9px;line-height:1.7}
.glossary td.gl-cat{white-space:nowrap;color:#1f2937;font-weight:600}
.glossary td.gl-cat .lgd{vertical-align:middle;margin-right:8px}
.cmdinv td.catc{text-align:left;white-space:nowrap}
.cmdinv td.catc .lgd{vertical-align:middle;margin-right:8px}
.glossary code,.cmdinv code,.otherbrk code{background:#eef2f7;padding:0 4px;border-radius:3px;font-size:11.5px;margin:0 1px;display:inline-block}
.cmdinv td:nth-child(2){white-space:nowrap;color:#4b5563}
/* Gold-band cells sit on a light indigo tint (see gold_shade), so the muted
   greys below are readable again. They were specified against a white
   background but used to render on a saturated red fill, which erased them. */
/* The data cells are nowrap (see .ge/.gc), so they no longer compress; without
   a reserved width the band column is what gets squeezed instead. */
.gold th:first-child,.gold td.ck{width:148px;min-width:148px;white-space:nowrap}
.gold td.ck{color:#334155}
.gold td.sc{padding:7px 7px;line-height:1.45}
.gold td.sc b{display:block;color:#0b1020;font-size:15px;letter-spacing:-.01em}
.ge{display:block;font-size:11px;color:#1f2937;font-variant-numeric:tabular-nums;
 white-space:nowrap}
/* "Low confidence" used to be encoded as a light grey, which on any tinted cell
   fell to ~3.6:1 -- under AA, and unreadable in the all-thin `long` row. The
   signal is carried by the italic and by the explicit nN count instead, so the
   colour can stay dark enough to read (~5.7:1 on the tints thin cells get). */
.ge.thin{color:#475569;font-style:italic}
.ge.none{color:#475569;font-style:italic}
.gn{font-size:10px;color:#475569;font-style:normal}
.gc{display:block;font-size:11px;color:#334155;font-variant-numeric:tabular-nums;
 white-space:nowrap}
.gc.hi{color:#b91c1c;font-weight:700}
.oht td.oh{text-align:center;padding:5px 6px;vertical-align:top}
.oht th{text-align:center}
.oht th:first-child,.oht td:first-child{text-align:left;white-space:nowrap}
.ohn{display:block;font-size:12.5px;font-weight:700;color:#4338ca;
 font-variant-numeric:tabular-nums}
.ohn.cap{color:#b91c1c}
.ohb{display:block;position:relative;width:72px;height:9px;margin:3px 0 0;
 background:#f1f5f9;border-radius:2px}
.ohb i{position:absolute;left:0;top:1px;bottom:1px;background:#4f46e5;border-radius:1px}
.ohb i.cap{background:#dc2626}
/* reasoning-effort chapter */
h3{font-size:14px;margin:26px 0 6px;color:#111827}
.bsub2{font-size:10.5px;font-weight:500;color:#5b6472;text-transform:lowercase;
 letter-spacing:.02em}
.efftbl{font-size:12px}
.efftbl th{padding:4px 6px;text-align:center;vertical-align:bottom}
.efftbl th:first-child{text-align:left}
.efftbl td{padding:4px 6px}
.efftbl thead tr:first-child th{border-bottom:none}
.efftbl thead tr:first-child th[colspan]{border-bottom:1px solid #e5e7eb;
 color:#4b5563;font-size:11px;text-transform:uppercase;letter-spacing:.05em}
.efftbl td.up,.efftbl td.down{font-weight:700}
.efftbl .famrow td{color:#1d4ed8;text-transform:none;letter-spacing:0;font-size:12px;
 font-weight:700}
"""


def render_html(stats, provenance, notes, meta, gold_sum, grid, design):
    _code = lambda s: f"<code>{esc(s)}</code>"                    # noqa: E731
    _sweep_glob, _fixed_globs, _run_note = reproduce_globs(_code)
    _fixed_html = " and ".join(_code(g) for g in _fixed_globs)
    ladder = " \u00b7 ".join(
        f'{name} {stats[key]["pass_pct"]}%' for name, _, key in MODELS
    )
    C, M = grid["cells"], grid["mcnemar"]
    cp = grid["cap_pass"]
    # Effort aggregates run over SWEEP_MODELS ONLY: `grid` has no cells and no
    # McNemar entry for the four FIXED_TREE models, so indexing C or M with their
    # keys is a KeyError, and averaging over len(MODELS) would divide a ten-model sum
    # by fourteen. Anything that says "effort" below counts ten.
    n_sweep = len(SWEEP_MODELS)
    n_sig = sum(1 for _n, _s, k in SWEEP_MODELS if M[k]["significant"])
    eff_mean = {e: sum(C[(k, e)]["pass_pct"] for _n, _s, k in SWEEP_MODELS) / n_sweep
                for e in EFFORTS}
    best = max(SWEEP_MODELS, key=lambda m: M[m[2]]["delta_pp"])
    worst = min(SWEEP_MODELS, key=lambda m: M[m[2]]["delta_pp"])
    # "Best in the field" spans all FOURTEEN, so it reads the primary-level stats
    # rather than the effort grid.
    top = max(MODELS, key=lambda m: stats[m[2]]["pass_pct"])
    field_mean = sum(stats[k]["pass_pct"] for _n, _s, k in MODELS) / len(MODELS)
    # Separation per effort level, COMPUTED rather than asserted. This used to be a
    # hardcoded "14.0 pp at medium, vs 13.0 at high and 10.0 at low", which was
    # measured on the nine-model field and silently went stale the moment the field
    # changed. Top-3 minus bottom-3 mean pass rate over the swept models.
    def _sep(e):
        v = sorted((C[(k, e)]["pass_pct"] for _n, _s, k in SWEEP_MODELS), reverse=True)
        return sum(v[:3]) / 3 - sum(v[-3:]) / 3
    sep = {e: _sep(e) for e in EFFORTS}
    # Is "higher effort => more steps" actually still unanimous? Counted, not claimed.
    n_mono = sum(1 for _n, _s, k in SWEEP_MODELS
                 if C[(k, "low")]["mean_steps"] <= C[(k, "medium")]["mean_steps"]
                 <= C[(k, "high")]["mean_steps"])
    # Cells: ten swept models x three levels, plus one cell each for the three
    # single-setting open-source models.
    n_cells = n_sweep * len(EFFORTS) + len(FIXED_TREE)
    # The path-mode models. Each is a FIXED_TREE model, i.e. exactly one cell, so
    # this is both a model count and a cell count -- which is what lets the header
    # state the max_image_edge split without a second hardcoded number.
    n_path = sum(1 for k in OPEN_SOURCE_CAVEAT if "path" in OPEN_SOURCE_CAVEAT[k])
    n_mm_both = len(MODELS) - n_path
    # The header used to name the path-mode models in a hand-typed list --
    # "deepseek-v4-pro and glm-5.2 ran at path ... the {n_path} cells belonging
    # to those same two models". n_path is computed, the names and the "two"
    # were not, so when a third path-mode model arrived the sentence shipped as
    # "1920 in the 3 belonging to those same two models": a header that
    # contradicts itself inside one clause. Derive both from the same set.
    _path_keys = [k for k in PROSE_KEYS
                  if k in OPEN_SOURCE_CAVEAT and "path" in OPEN_SOURCE_CAVEAT[k]]
    path_names = _join([PROSE_NAME[k] for k in _path_keys])
    path_those = f"those same {_numword(n_path)} models"
    if n_path != len(_path_keys):
        raise SystemExit(
            f"header: n_path={n_path} but {len(_path_keys)} path-mode models "
            "were named. These must come from the same set.")
    doc = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Multi-Agent VLM Conflict \u2014 Cross-Model Behavioral Comparison (v8)</title>
<style>{CSS}</style></head><body><div class="wrap">
<h1>Cross-Model Behavioral Comparison</h1>
<div class="sub">multiagent_vlm_conflict \u00b7 v8 corpus \u00b7 solver behavior of {len(MODELS)} models over the same {meta['n_tasks']} tasks ({meta['n_scenarios']} scenarios \u00d7 {meta['n_rounds']} rounds). {n_sweep} of them were additionally swept across three reasoning-effort levels; the {len(FIXED_TREE)} open-source models were run at a single setting \u2014 {n_cells} cells, {n_cells * meta['n_tasks']:,} episodes</div>
<div class="sub">Aligned harness: tool_mode=native, history_mode=full, native_tool_choice=auto, <code>--max-steps 20</code> \u2014 identical across all {n_cells} cells. Two settings are <b>not</b> uniform. <b>multimodal_input_mode=both for {n_mm_both} of the {len(MODELS)} models</b>: {esc(path_names)} ran at <code>multimodal_input_mode=path</code> and never received an image, so their grounding and cost numbers are not comparable to the rest. And max_image_edge is 2576 in {n_cells - n_path} cells but 1920 in the {n_path} belonging to {path_those} \u2014 where the cap is inert, since a model sent no images cannot reach it. Both are stated per-model under <a href="#{CAVEAT_ANCHOR}">run conditions</a>, and every table below that includes an open-source row carries a footnote linking there.</div>
<div class="sub"><b>The {_NSEC} behavioral sections below all describe <code>reasoning_effort={PRIMARY_EFFORT}</code></b> \u2014 the only nominal setting the entire field shares, because the {_numword(len(FIXED_TREE))} open-source models exist at {PRIMARY_EFFORT} alone. That choice costs some discrimination: measured on the {n_sweep} swept models, the top-3 minus bottom-3 mean pass rate is {sep['low']:.1f} pp at low, {sep['medium']:.1f} pp at medium and {sep['high']:.1f} pp at high. It buys a main table whose every row is drawn from one setting. The effort dimension itself is analysed in its own chapter: <a href="#effort">Reasoning effort</a>.</div>
<div class="sub">Pass@budget ({PRIMARY_EFFORT}): {esc(ladder)}</div>
<div class="sub">Generated by <code>docs/analysis/analysis_v8/tools/build_behavior_report.py</code> \u00b7 {esc(provenance)}</div>
<div class="sub">This report is self-contained (no external assets or links required). Companion prose lives alongside it in the repo: <code>docs/analysis/analysis_v8/cross-model-behavioral-comparison.md</code></div>

<div class="callout"><b>Two findings frame everything below.</b>
<b>(1) The corpus is hard and it discriminates.</b> The best model in the field,
{esc(top[0])}, passes <b>{C[(top[2], PRIMARY_EFFORT)]["pass_pct"]}%</b> of
{meta['n_tasks']} tasks at {PRIMARY_EFFORT} effort \u2014 far below the &lt;&nbsp;35%
target that motivated this generation \u2014 while the {len(MODELS)}-model field mean
sits at {field_mean:.1f}% and the spread between the strongest and weakest
model is wide enough to rank them without ambiguity.
<b>(2) Reasoning effort is a weak lever with a sharp edge.</b> Averaged over the
{n_sweep} swept models, going from low to high effort moves the pass rate by
{eff_mean['high'] - eff_mean['low']:+.1f} pp
({eff_mean['low']:.1f}% \u2192 {eff_mean['medium']:.1f}% \u2192 {eff_mean['high']:.1f}%),
and under a paired McNemar test only <b>{n_sig} of {n_sweep}</b> models move at
all. But the per-model range is enormous \u2014 from
<b>{M[best[2]]["delta_pp"]:+.1f} pp</b> ({esc(best[0])}) to
<b>{M[worst[2]]["delta_pp"]:+.1f} pp</b> ({esc(worst[0])}) \u2014 because effort is paid
for in steps. Higher effort makes {n_mono} of those {n_sweep} models take more steps
monotonically,
and an episode that exhausts the 20-step budget is graded where it stands:
<b>{cp["capped_passed"]} of {cp["capped_n"]}</b> step-capped episodes across the
whole grid passed, against
{100 * cp["uncapped_passed"] / max(cp["uncapped_n"], 1):.1f}% of the
{cp["uncapped_n"]:,} that terminated on their own. Whether effort helps a model is
therefore a race between the reasoning it buys and the termination budget it
spends \u2014 see <a href="#effort">Panel C</a>.</div>

{render_caveat_box(stats, asset_gate_ranking(stats))}

{render_standings(stats)}

{render_effort(grid)}

{render_glossary(stats, meta)}
{render_legend()}
{render_mix(stats, meta)}
{render_command_surface(stats, notes, meta)}
{render_map(stats)}
{render_acc_score_scatter(stats)}
{render_clustering(stats)}
{render_stops(stats)}
{render_steps_vs_gold(gold_sum, meta)}
{render_cost(stats, meta)}
{render_cost_vs_pass(stats)}
{render_all27(stats, meta)}
{render_design_axes(design, meta)}
{render_timelines(stats, notes)}
{render_signatures(stats)}
{render_appendix(stats, meta)}

<div class="foot">Regenerate from the trajectory <code>*.jsonl</code> under
{_code(_sweep_glob)}
and, for the {len(FIXED_TREE)} single-setting open-source models,
{_fixed_html}.{_run_note}
gpt-5.5 is absent from this report: its three cells are unusable (no <code>low</code>
directory; <code>medium</code> is 480 records of <code>steps=0</code> behind a 429 quota
wall; <code>high</code> has five records and no run summary). It shares a provider quota
pool with the three gpt-5.6 models and lost the race for it \u2014 an infrastructure
outcome, not a result about gpt-5.5.</div>
</div></body></html>"""
    # Attach the run-conditions footnote to every table that turned out to carry an
    # open-source row. Done here, on the finished document, rather than at each
    # render_* site -- see the comment above annotate_os_tables() for why.
    doc = annotate_os_tables(doc)
    # "The N behavioral sections below" was the word "sixteen", typed. Adding a
    # section does not make a writer revisit the header, so it silently became
    # a lie the first time one was added -- which is exactly what happened when
    # the standings chart went in. Count the headings the sentence is about.
    n_sec = len(re.findall(r"<h2\b", doc))
    if _NSEC not in doc:
        raise SystemExit("the section-count placeholder never reached the "
                         "document; the header sentence is unguarded again")
    return doc.replace(_NSEC, _numword(n_sec))


def reproduce_globs(code):
    """The exact directory globs this build read, for the "regenerate" footer.

    `code` wraps an inline code span for the target format. It is a parameter
    rather than a hardcoded backtick because the first version returned
    markdown backticks into an HTML footer, where they render as literal
    backticks next to real <code> spans.

    Typed out, this footer was wrong in two independent ways and neither was
    visible from the footer itself. It named `open_source_high/{DeepSeek,
    Moonshot,ZhipuAI}/*/r01/` for the fixed-effort cohort, which silently
    excludes v4-flash (it lives under `aligned_setting_high_ocr/`); and it
    pinned `*/r01/` for the swept cohort, which is wrong for the one model
    promoted to a rerun. A reader following either glob would rebuild a
    different corpus than the one the tables report and have no way to tell.

    Returns (sweep_glob, [fixed_glob, ...], run_note). All three are derived from the
    same constants the loader uses, so they cannot describe a corpus the build
    did not read. A brace glob over the run directories would be a slight
    over-claim on its own -- it matches (cell, run) pairs that do not exist --
    so the run component is left at the default and the exceptions are named in
    run_note instead of being smuggled into the pattern.
    """
    def braces(xs):
        xs = sorted(set(xs))
        return xs[0] if len(xs) == 1 else "{" + ",".join(xs) + "}"

    # EFFORTS keeps its own order -- low/medium/high is an ordinal scale, and
    # braces() would sort it to {high,low,medium}, which reads as a typo.
    sweep = (f"eval_results/data_3_6_flash_v8/"
             f"aligned_setting_{{{','.join(EFFORTS)}}}/"
             f"{braces(s.split('/')[0] for _, s, _ in SWEEP_MODELS)}/*/"
             f"{DEFAULT_RUN}/")

    # One glob per tree, because the fixed cohort no longer lives in one.
    # Returned as a list so the caller can wrap each in its own <code>: joining
    # them here put the connecting " and " inside the code span.
    by_tree = {}
    for k, p in FIXED_TREE.items():
        _, tree, vendor, _slug = p.split("/")
        by_tree.setdefault(tree, []).append(vendor)
    fixed = [f"eval_results/data_3_6_flash_v8/{tree}/{braces(vs)}/*/{DEFAULT_RUN}/"
             for tree, vs in sorted(by_tree.items())]

    note = ""
    if RUN_SUBDIR:
        note = (" Substitute "
                + _join([f"{code(r + '/')} for {PROSE_NAME[k]}"
                         for k, r in sorted(RUN_SUBDIR.items())])
                + f"; every other cell is read from {code(DEFAULT_RUN + '/')}.")
    return sweep, fixed, note


def compute_meta(stats):
    """Derive dataset shape (tasks / scenarios / rounds) from the loaded data so
    labels stay correct if the eval set grows (e.g. 270/27 -> 600/60)."""
    n_tasks = max((st["n"] for st in stats.values()), default=0)
    scenarios = set()
    rounds = set()
    for st in stats.values():
        for tid in st["by_task"]:
            scenarios.add(scenario_of(tid))
            m = re.search(r"_r(\d+)$", tid)
            if m:
                rounds.add(int(m.group(1)))
    return {
        "n_tasks": n_tasks,
        "n_scenarios": len(scenarios),
        "n_rounds": max(rounds) if rounds else 0,
    }


def main():
    here = Path(__file__).resolve()
    # tools -> analysis_v8 -> analysis -> docs -> repo root. The v8 copy sits one
    # level deeper than the corpus this file was forked from, so parents[3] (which
    # the fork inherited) lands on docs/ and finds no eval_results.
    src_root = here.parents[4]
    # Mirrors analysis/corpus.py: CLASHARNESS_DATA_ROOT, else <repo>/data.
    data_root = os.environ.get("CLASHARNESS_DATA_ROOT", str(src_root / "data"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=data_root,
                    help="corpus root containing eval_results/")
    ap.add_argument("--out", default=str(here.parents[1]),
                    help="output dir (default docs/analysis/analysis_v8/)")
    ap.add_argument("--effort", default=PRIMARY_EFFORT, choices=list(EFFORTS),
                    help="reasoning-effort level the behavioral sections describe")
    # Second root, and it is NOT --root. The eval corpus and the scenario seeds
    # live in different checkouts: eval_results/data_3_6_flash_v8 exists only in
    # the data tree, the seeds only in the source tree. Defaulted off this file's
    # own location, which is in the source tree.
    ap.add_argument("--seeds", default=str(src_root / SEED_SUBDIR),
                    help="scenario-seed directory for the designed-difficulty "
                         "section (source tree, not --root)")
    args = ap.parse_args()

    primary = args.effort
    stats = {}
    for _, vendor, key in MODELS:
        # key=key so the three FIXED_TREE models resolve to their single subtree.
        # Without it this composes aligned_setting_{primary}/DeepSeek/v4_pro, which
        # does not exist (that cohort lives under open_source_high/), and the run
        # dies on the `no records found` guard below.
        subdir = tree(vendor, primary, key=key)
        run = run_for(key)
        recs = load_model(args.root, subdir, run)
        if not recs:
            raise SystemExit(
                f"no records found for {subdir}/{run} under {args.root}/eval_results")
        stats[key] = analyze(recs)
        # API cost / token usage from the interaction sidecar logs (None -> N/A).
        # COST_RUN_SUBDIR is empty in this corpus: every model records usage on 100%
        # of attempts, so cost is read from the same run as behavior.
        cost_run = COST_RUN_SUBDIR.get(key, run)
        by_task = load_token_usage(args.root, subdir, cost_run, key)
        stats[key]["cost"] = summarize_cost(by_task, stats[key], key)
        # Step-0 per-image input-token cost, from the same run the token/$ came from.
        imgs0, tok_per_img = image_token_stats(args.root, subdir, cost_run)
        stats[key]["cost"]["imgs_step0"] = imgs0
        stats[key]["cost"]["tok_per_image"] = tok_per_img

    # Re-prove the curated-round pins against the corpus just loaded, BEFORE any
    # rendering. The showcase section tells the reader each round attains its
    # scenario's widest spread; if a roster or corpus change ever breaks that, the
    # build must stop rather than quietly print a claim it can no longer support.
    check_curated_pins(stats)

    meta = compute_meta(stats)
    # Run conditions come off the corpus headers, not out of this file. The
    # check inside also fails the build if OPEN_SOURCE_CAVEAT has drifted from
    # what the logs say -- see scan_run_conditions() for why that guard exists.
    # Load the run each override was chosen OVER, so the disclosure attached to
    # the promoted row is measured rather than remembered, and so the ranking
    # cost pinned in RUN_OVERRIDE_EXPECTED can be re-proved. One extra cell.
    run_facts = run_override_facts(args.root)
    build_run_caveats(run_facts)
    check_run_overrides(args.root, stats, run_facts)
    meta["conditions"] = scan_run_conditions(args.root)
    provenance = _html_provenance(primary)
    gold = load_gold(args.root)
    gold_sum = gold_summary(stats, gold)
    # Section 8.1's bands. Computed here, alongside gold_sum, rather than inside
    # the block that renders it: it reads by_task across every model to cut one
    # shared partition, which is a corpus-level derivation and not a rendering
    # decision. Deliberately NOT added to the slim sidecar -- the figures it
    # holds are a re-banding of stats already in there, and the sidecar is a
    # published artifact whose shape other tools read.
    unit_sum = unit_band_summary(stats)
    grid = load_effort_grid(args.root)
    seeds = load_seeds(args.seeds)
    # Prose figures are tokens in the sidecar; bind them to the corpus that was
    # just loaded. Must run after check_curated_pins() -- the tokens resolve
    # against the pinned rounds, so a stale pin would silently bind the prose to
    # a round the reader is not being shown -- and after load_effort_grid(), so
    # that {{cell.*}} can reach the sweep.
    #
    # The grid rides in on a SHALLOW COPY rather than on stats itself. Every
    # other consumer of stats -- check_curated_pins, compute_meta, render_html,
    # the slim sidecar -- iterates it as {model_key: per-model dict}, and a
    # stray non-model key would be silently treated as a fourteenth model.
    tstats = dict(stats)
    tstats["effort"] = {
        "cells": {f"{k}@{e}": {kk: vv for kk, vv in c.items()
                               if not kk.startswith("_")}
                  for (k, e), c in grid["cells"].items()},
    }
    template = load_notes(args.out)
    check_notes_derived(template)
    notes = resolve_notes(template, tstats)
    check_notes_resolved(notes)
    design = design_axes(stats, seeds)
    stray = {k: v["unmapped"] for k, v in design.items() if v["unmapped"]}
    if stray:
        print(f"WARNING scenarios with no seed: {stray}")

    out_html = os.path.join(args.out, "cross-model-behavioral-comparison.html")
    doc = render_html(stats, provenance, notes, meta, gold_sum, grid, design)
    # Prove, against the finished document, that every table carrying an
    # open-source row also carries the run-conditions footnote -- and that none
    # carries one spuriously. Raises rather than warns: the header promises the
    # reader that footnote exists, and shipping the promise without the footnote
    # is exactly the failure this replaced.
    n_noted = check_os_notes(doc)
    with open(out_html, "w") as f:
        f.write(doc)

    # trimmed stats (drop per-step chains) for a compact reproducibility artifact
    slim = {}
    for key, st in stats.items():
        slim[key] = {k: v for k, v in st.items() if k != "by_task"}
    slim["steps_vs_gold"] = gold_sum
    slim["primary_effort"] = primary
    # Effort grid, minus the per-task success/cap maps that drive McNemar (they are
    # 27 x 600 booleans and would triple the file for no reader benefit).
    slim["effort"] = {
        "cells": {f"{k}@{e}": {kk: vv for kk, vv in c.items()
                               if not kk.startswith("_")}
                  for (k, e), c in grid["cells"].items()},
        "mcnemar": grid["mcnemar"],
        "cap_pass": grid["cap_pass"],
    }
    # The difficulty-stratified and paired clustering corrections. These are the
    # figures the markdown companion quotes for any per-model claim; the
    # whole-corpus clustering block is descriptive only (see clustering_strata).
    slim["clustering_strata"] = clustering_strata(stats)
    slim["gap_decomposition"] = gap_decomposition(stats)
    # Designed-difficulty buckets. Emitted so the section above is reproducible
    # from the sidecar alone, and so audit pass D14 has a second chain to check
    # design_axes.json and the per-model SF11 blocks against. Integer counts only
    # -- see the section comment on render_design_axes for why.
    slim["design_axes"] = {
        "seed_dir": SEED_SUBDIR,
        "n_seeds": len(seeds),
        "category_order": CONFLICT_CAT_ORDER,
        # The full 23-plugin registry copy, so a reader of the sidecar alone can
        # see which buckets are challenge plugins and reconstruct the category
        # axis (which is carried in by_model but deliberately not rendered).
        "plugin_category": PLUGIN_CATEGORY,
        "groups": [{"axis": ax, "buckets": [b for b, _ in bs],
                    "levers": [b for b, lv in bs if lv]}
                   for _h, ax, bs in DESIGN_GROUPS],
        "by_model": design,
    }
    out_json = os.path.join(args.out, "cross-model-behavioral-comparison.stats.json")
    with open(out_json, "w") as f:
        json.dump(slim, f, indent=2)

    # The resolved prose, emitted as its own artefact. behavior_notes.json is now
    # a template full of {{tokens}}, so the markdown mirror can no longer be
    # byte-compared against it -- and the markdown must still say exactly what
    # the HTML says. Writing the expanded text here gives the mirror and the
    # audit one shared, regenerated source instead of two hand-kept copies that
    # drift apart silently, which is precisely how the markdown ended up
    # asserting the opposite of the data.
    out_notes = os.path.join(args.out,
                             "cross-model-behavioral-comparison.notes.json")
    with open(out_notes, "w") as f:
        json.dump(notes, f, indent=2, ensure_ascii=False)

    # Generated markdown for the mirror's non-prose blocks (see
    # render_md_blocks). Same contract as the notes sidecar: the build owns the
    # text, sync_notes_md.py places it, and nothing is transcribed by hand.
    out_blocks = os.path.join(args.out,
                              "cross-model-behavioral-comparison.blocks.json")
    with open(out_blocks, "w") as f:
        json.dump(render_md_blocks(stats, meta, gold_sum, grid, unit_sum), f,
                  indent=2, ensure_ascii=False)

    print("wrote", out_html, f"({n_noted} tables carry the run-conditions footnote)")
    print("wrote", out_json)
    print("wrote", out_notes)
    print("wrote", out_blocks)


if __name__ == "__main__":
    main()
