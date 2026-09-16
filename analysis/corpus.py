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

"""Single authoritative loader for every table in the paper.

WHY THIS FILE EXISTS
--------------------
Each analysis script used to re-implement dead-round exclusion, error-row
handling and the effort/population choice on its own. That is how the
`round_id` bug shipped: one script parsed the round id out of `task_id` by
splitting on `__`, a separator that does not exist in these ids, so its
exclusion set silently matched nothing while the other scripts were fine.
The fix is structural, not another downstream audit -- every table imports
this module and nobody re-derives the population.

INVARIANTS ENFORCED HERE (assertions, not comments)
  * dead-round list resolves to exactly N_DEAD_V8 = 18 (slug, round_id) pairs
  * every dead pair actually matches at least one eval row (the bug's tell)
  * the best-covered cell drops exactly 18 rows
  * round ids come from metadata["round_id"]; the task_id regex is a fallback
    that must still produce a non-empty id or we abort
  * every cell has exactly 600 rounds in the FULL view and 582 in the EXCLUDED
    view, and error rows are counted either way, never silently absorbed
  * seed -> conflict-category mapping is the generator's own registry output,
    cached to seed_axis_map.json and re-checked against the shipped
    design_axes.json on every load

POPULATION
----------
The paper reports the FULL 600-round view with error rows retained. That is
also exactly the population the behavioural report was built on, so the
paper's tables and the report's figures and appendix tables all agree by
construction rather than by conversion.

Eighteen of those 600 rounds were provably unwinnable when these runs were
scored: a materializer visibility defect graded them on a value that appeared
in nothing the solver was shown. Across 18 rounds x 10 API models x 3 efforts =
540 round-runs there is exactly ONE pass. Keeping them therefore rescales every
model by the same 582/600 = 0.9700 and changes no ordering and no test; the
EXCLUDED view is kept alongside so that can be shown rather than asserted.

The defect is fixed and the corpus repaired in place; only these pre-repair
eval rows carry it. The paper must say so where it states n.

PRIMARY EFFORT is "high" for the paper, and since the 14-model rebuild it is
"high" for the behavioural report too (build_behavior_report.PRIMARY_EFFORT; it
was "medium" up to 2026-08-14). The two no longer need converting between.

ROSTER is the axis that still splits. N_PANEL = 14 models are reported at high;
N_MODELS = 10 of them are API endpoints that expose a reasoning-effort knob and
are the only ones swept, so any effort statement is over 10 and any high-effort
statement is over 14. That is the distinction the 486-versus-540 arithmetic
above turns on.
"""
import glob
import hashlib
import json
import math
import os
import random
import re
import sys
import zlib
from collections import defaultdict

PAPER = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PAPER)

# The corpus these figures are computed over is far too large to keep in a git
# repository -- 9.5 GB of eval results across the five arms and 12 GB of task
# specs across the two generators -- so it is
# hosted separately and fetched by `tools/download_data.py`, which unpacks it
# under `data/` at the repository root. Point CLASHARNESS_DATA_ROOT somewhere
# else to keep it off this disk, or override an individual path below to read a
# corpus you generated yourself.
DATA_ROOT = os.environ.get(
    "CLASHARNESS_DATA_ROOT", os.path.join(REPO_ROOT, "data"))

ROOT_V8 = os.environ.get(
    "CLASHARNESS_EVAL_ROOT",
    os.path.join(DATA_ROOT, "eval_results", "data_3_6_flash_v8"))
TASKS_V8 = os.environ.get(
    "CLASHARNESS_TASKS_V8",
    os.path.join(DATA_ROOT, "multiagent_conflict_data_v8_Gemini_3_6_flash",
                 "tasks"))
# Small enough to travel with the code, unlike the corpus it excludes rounds
# from: this is the hand-audited list of rounds whose visible references were
# unrecoverable, and every figure depends on excluding exactly these.
DEAD_JSON = os.environ.get(
    "CLASHARNESS_DEAD_ROUNDS",
    os.path.join(PAPER, "dead_rounds_exclusion.json"))
AXIS_MAP = f"{PAPER}/seed_axis_map.json"
GATE_MAP = f"{PAPER}/mechanic_gate.json"
# The report artefacts this module cross-checks itself against. Both are
# committed under src/report, so the self-check runs from a clean clone with no
# source checkout present; CLASHARNESS_REPORT_DIR repoints it at a live analysis
# directory when regenerating.
REPORT_DIR = os.environ.get(
    "CLASHARNESS_REPORT_DIR", os.path.join(PAPER, os.pardir, "src", "report"))
DESIGN_AXES = os.path.join(REPORT_DIR, "design_axes.json")
STATS_JSON = os.path.join(
    REPORT_DIR, "cross-model-behavioral-comparison.stats.json")


def frozen_effort():
    """The reasoning effort the frozen factory artifact was built at.

    EVERY REPLICATION CHECK MUST READ THIS RATHER THAN NAME AN EFFORT. Three
    modules -- cost.py, clustering_v8.py and check_split_v8.py -- each verify
    that their reimplementation of a factory statistic reproduces stats.json
    exactly before they report anything of their own. All three named "medium"
    inline, which was correct only while the report happened to be a
    medium-effort build. Rebuilding it at high turned each replication into a
    silent cross-effort comparison: the loader fetched the medium tree and the
    reference values were high, so the checks failed on every field that moved
    (gemini-3.1-pro's 599 graded rounds against the artifact's 595, and so on)
    and read as arithmetic bugs in code that was correct.

    A hardcoded effort here is a landmine with a long fuse -- it is fine until
    someone rebuilds the report, which is exactly when these checks are most
    needed. There is no default and no fallback: if the artifact does not say,
    the caller should fail rather than guess.
    """
    eff = json.load(open(STATS_JSON))["primary_effort"]
    assert eff in LEVELS, f"frozen artifact reports unknown effort {eff!r}"
    return eff


# The factory's own re-bucketing of the same episodes, committed beside the
# report for the same reason: a figure that reads them must be reproducible
# from a clean clone. Callers are expected to verify them against load_eval
# (n, pass rate, cap share, mean score) before using them -- they are inputs,
# not authorities.
FAILURE_TAXONOMY = os.path.join(REPORT_DIR, "failure_taxonomy.json")
REASONING_TOKENS = os.path.join(REPORT_DIR, "reasoning_tokens.json")
# Per-task spec metadata for all 600 rounds (gold length, scored units, round
# type, tool families, natural-image flag) plus the factory's own cardinality
# census of every declared spec field. Section 4's coverage table is built from
# it; see coverage_v8.py for why most of those declared fields are constant.
DIFFICULTY_AXES = os.path.join(REPORT_DIR, "difficulty_axes.json")

RUN_V8 = "r01"

# PER-MODEL RUN SELECTION, keyed by on-disk directory name.
#
# Every cell in the corpus has exactly one run, r01, with a single exception:
# aligned_setting_high/Gemini/3_7_flash carries both r01 and r02. Both are
# complete and clean (60 scenario files, 600 deduped tasks, zero rows carrying
# an `error` field, zero empty trajectories). This is therefore a SELECTION
# between two usable draws, not the repair of a damaged one.
#
# The two runs are not exchangeable draws, and anything that reasons about
# them as if they were will be wrong:
#   * r02 is a fresh policy rollout, not a re-grade of r01. All 600
#     trajectories and all 600 durations differ.
#   * the grader differs. r01 was scored by gemini-3.6-flash, the judge used
#     by 40 of the 43 runs in this corpus; r02 was scored by gemini-3.7-flash,
#     i.e. by the model under test. The only other two runs graded by
#     3.7-flash are this model's own low and medium cells.
#   * so the r01 -> r02 delta (18.67% -> 20.33% pass, 0.5650 -> 0.5814 score)
#     confounds rollout noise with a judge change and cannot be attributed to
#     either. A paired McNemar over the 600 gives p = 0.27, which bounds how
#     large the combined effect is; it does NOT license calling them one
#     population.
#
# Selecting r02 makes this model's three effort cells internally consistent
# (all three then graded by 3.7-flash) at the cost of making it the one row in
# the table not graded by the corpus-standard judge. Selecting r01 would do the
# reverse. r02 is the author's call and matches the behavioural report.
#
# NOTE FOR ANYONE ADDING A CHECK HERE: the judge is NOT recorded in the episode
# headers -- run_config carries only the policy provider/model/effort -- so no
# invariant built on run_config can see a judge change. The only place it is
# written down is parallel_run.summary.
#
# Keyed by the FULL `<tree>/<vendor>/<model>` path, not by the bare directory
# name, because only the high-effort cell has a second run: this model's low
# and medium cells are r01-only, and a name-keyed override would send them
# looking for an r02 that does not exist and drop them from the roster.
RUN_OVERRIDE = {"aligned_setting_high/Gemini/3_7_flash": "r02"}


def run_for(rel_dir, run=None):
    """The run subdirectory to read for one `<tree>/<vendor>/<model>` cell."""
    return RUN_OVERRIDE.get(rel_dir, RUN_V8 if run is None else run)


BUDGET_V8 = 20
# Every gold trajectory ends in a bare "DONE" sentinel (600/600 in both
# corpora). It marks the end of the recording, not an action the solver has to
# take: scored agents never emit it -- 0 occurrences across 600 evaluation
# episodes -- and the step budget counts only real tool calls. Counting it
# would overstate gold length by exactly one step and, worse, put the slack
# denominator on a different footing from the budget numerator. So gold length
# is real commands only, which is also the definition the frozen
# difficulty_axes.json artifact uses (its gold_len is this count exactly, on
# all 600 tasks).
GOLD_SENTINEL = "DONE"
DEAD_LABEL_V8 = "conflict_vlm_v8_3.6_flash"
N_DEAD_V8 = 18
N_RAW = 600                        # FULL view: every round, error rows kept
N_EXCLUDED = N_RAW - N_DEAD_V8     # 582, robustness view

# The two populations, and which one the paper reports.
#   "full"      600 rounds, dead rounds and error rows retained. Matches the
#               behavioural report, so its 23 tables and 7 charts drop into the
#               appendix without recomputation.
#   "excluded"  582 rounds, the 18 provably-unwinnable rounds removed, error
#               rows dropped. Every rate here is the full rate / 0.9700.
POPULATIONS = ("full", "excluded")
PRIMARY_POPULATION = "full"

LEVELS = ("high", "medium", "low")
PRIMARY_EFFORT = "high"

# On-disk directory name -> the paper's display name. Never abbreviate a model
# to a bare family word; these strings are what goes into the LaTeX tables. An
# unmapped directory aborts rather than passing through, so a renamed tree
# cannot quietly appear as a new model row.
#
# Names are fully qualified: the family prefix is part of the name, so a model
# can be quoted in running prose without the surrounding table to disambiguate
# it. `sol`, `luna` and `terra` in particular mean nothing on their own.
#
# Two pieces of the raw provider id are dropped here because they describe how
# the call was routed rather than which model answered: the `openai.` prefix on
# the GPT ids is a Bedrock routing namespace, and `-preview` on the Gemini pro
# id is a release channel. Both survive verbatim in the appendix roster table,
# whose entire purpose is to document this mapping -- see the `is_roster`
# branch in src/extract_tables.py.
MODEL_LABEL = {
    "opus_5": "claude-opus-5",
    "opus_4_8": "claude-opus-4.8",
    "sonnet_5": "claude-sonnet-5",
    "3_7_flash": "gemini-3.7-flash",
    "3_6_flash": "gemini-3.6-flash",
    "3_5_flash": "gemini-3.5-flash",
    "3_1_pro": "gemini-3.1-pro",
    "5_6_sol": "gpt-5.6-sol",
    "5_6_luna": "gpt-5.6-luna",
    "5_6_terra": "gpt-5.6-terra",
}
N_MODELS = len(MODEL_LABEL)

# Display name -> provider group. Result tables are blocked by provider so a
# within-family comparison (opus-5 against opus-4.8) sits on adjacent rows.
# The group label is the family as the paper names it, not the serving
# platform: the Claude models are reached through Vertex and the GPT models
# through Bedrock, and neither fact belongs in a results table.
MODEL_PROVIDER = {
    "claude-opus-5": "Claude",
    "claude-opus-4.8": "Claude",
    "claude-sonnet-5": "Claude",
    "gpt-5.6-sol": "GPT-5.6",
    "gpt-5.6-luna": "GPT-5.6",
    "gpt-5.6-terra": "GPT-5.6",
    "gemini-3.7-flash": "Gemini",
    "gemini-3.6-flash": "Gemini",
    "gemini-3.5-flash": "Gemini",
    "gemini-3.1-pro": "Gemini",
}
assert set(MODEL_PROVIDER) == set(MODEL_LABEL.values()), (
    "MODEL_PROVIDER must cover exactly the API display names, or a model "
    "would silently sort into a provider block of its own")

# --------------------------------------------------------------------------
# The open-weight panel.
#
# Three open-weight checkpoints served from a self-hosted OpenAI-compatible
# endpoint, evaluated at high effort ONLY, and living in their own tree
# (OPEN_TREE) rather than under aligned_setting_high.
#
# WHY THEY ARE NOT IN MODEL_LABEL. That map is held in bijection with the
# behavioural report's own vocabulary by four separate asserts below
# (REPORT_RAW_ID, REPORT_SPACE_FORM, REPORT_SHORT, STATS_KEY). The report is a
# frozen artefact that predates these models, so an entry in MODEL_LABEL would
# demand report keys that cannot be made to exist, and the nine-model
# replication checks in clustering_v8.py and check_split_v8.py -- which index
# stats.json by STATS_KEY -- would fail on a missing key. Keeping the two
# rosters apart is precisely what lets those checks keep passing untouched.
#
# Display names drop the HuggingFace org prefix (`deepseek-ai/`, `moonshotai/`,
# `zai-org/`) for the same reason MODEL_LABEL drops Bedrock's `openai.`: it
# records where the weights were distributed from, not which model answered.
#
# `-FP8` is deliberately NOT dropped. A quantisation is part of what answered,
# and the recorded ids show the other two checkpoints carry no such suffix, so
# eliding it would present an asymmetric comparison as a uniform one. If the
# other two turn out to be quantised as well, the fix is to spell that out
# here too, not to drop it from this one.
#
# Recorded ids, verbatim from response records in the interaction logs:
#   v4_pro   deepseek-ai/DeepSeek-V4-Pro
#   kimi_k3  moonshotai/Kimi-K3
#   glm_5_2  zai-org/GLM-5.2-FP8
OPEN_MODEL_LABEL = {
    "v4_pro": "deepseek-v4-pro",
    "kimi_k3": "kimi-k3",
    "glm_5_2": "glm-5.2-fp8",
    "dsv4_flash_v8_high_ocr": "deepseek-v4-flash",
}
# Provider group labels match the on-disk vendor directories rather than
# inventing new ones, so a row in a table can be traced back to a path.
OPEN_MODEL_PROVIDER = {
    "deepseek-v4-pro": "DeepSeek",
    "kimi-k3": "Moonshot",
    "glm-5.2-fp8": "ZhipuAI",
    "deepseek-v4-flash": "DeepSeek",
}
# The tree these three live in, and the only effort they were run at. Named
# rather than spelled inline so that a caller cannot quietly ask for
# `open_source_medium`, which does not exist.
OPEN_TREE = "open_source_high"
OPEN_EFFORT = "high"

# OPEN-WEIGHT CELLS THAT DO NOT LIVE UNDER OPEN_TREE.
#
# OPEN_TREE is the pooled tree the harness filed the first three checkpoints
# under. deepseek-v4-flash was added later and was only ever written to the
# modality-specific tree, so there is no pooled copy of it to read -- it is a
# real run at a second location, which is exactly what ALIAS_DIRS must NOT be
# used for (that list is for byte-identical duplicates and re-proves them).
#
# The tree it does live in also holds v4_pro and glm_5_2, but both of those are
# registered in ALIAS_DIRS as duplicates of their OPEN_TREE copies and are
# skipped on load, so reading this tree against a one-model panel picks up
# exactly the one cell that is only here.
OPEN_EXTRA_TREES = {
    "aligned_setting_high_ocr": {
        "dsv4_flash_v8_high_ocr": OPEN_MODEL_LABEL["dsv4_flash_v8_high_ocr"],
    },
}
assert all(d in OPEN_MODEL_LABEL and OPEN_MODEL_LABEL[d] == lbl
           for panel in OPEN_EXTRA_TREES.values()
           for d, lbl in panel.items()), (
    "an OPEN_EXTRA_TREES entry disagrees with OPEN_MODEL_LABEL about a "
    "directory's display name")
assert OPEN_EFFORT in LEVELS and OPEN_EFFORT == PRIMARY_EFFORT, (
    "the open-weight group is only comparable against the effort the paper "
    "leads with; if that changes, this group needs re-running, not remapping")

assert not (OPEN_MODEL_LABEL.keys() & MODEL_LABEL.keys()), (
    "an open-weight directory name collides with an API model directory; the "
    "two panels are merged by directory key and one would shadow the other")
assert not (set(OPEN_MODEL_LABEL.values()) & set(MODEL_LABEL.values())), (
    "an open-weight display name collides with an API model display name")
assert set(OPEN_MODEL_PROVIDER) == set(OPEN_MODEL_LABEL.values()), (
    "OPEN_MODEL_PROVIDER must cover exactly the open-weight display names")

# The combined roster. `provider_blocks` and `group_by_provider` resolve
# providers through PANEL_PROVIDER so that they work for either panel; since
# it is a superset of MODEL_PROVIDER, nine-model callers are unaffected.
PANEL_LABEL = {**MODEL_LABEL, **OPEN_MODEL_LABEL}
PANEL_PROVIDER = {**MODEL_PROVIDER, **OPEN_MODEL_PROVIDER}
N_PANEL = len(PANEL_LABEL)

# HOW A RESULT TABLE IS BLOCKED.
#
# The nine API models are blocked by provider, so a within-family comparison
# sits on adjacent rows. The three open-weight models are deliberately NOT
# blocked that way: they form one block, pinned to the bottom of every result
# table, no matter how well they score.
#
# The reason is that vendor is not the axis that matters for them. All three
# carry the same set of caveats -- we served the officially released
# checkpoints on our own hardware, so their dollar figure is a list-rate
# reference rather than an invoice, their endpoints do not break out a
# reasoning-token count, and two of the three are text-only and read the
# rendered assets through OCR. Blocking them by vendor would scatter three
# rows that share every one of those caveats into three separate places in
# the table, next to metered API rows they are not directly comparable with.
# One pinned block lets a single rule and a single caption sentence cover
# them, and lets a reader see the API field as an uninterrupted run.
#
# This is presentation only. It never changes which rounds a number is
# computed over, and it never changes `rank_order`, which is what any
# "who beat whom" claim in the paper is required to use.
OPEN_BLOCK = "Open-weight (self-hosted)"
assert OPEN_BLOCK not in set(MODEL_PROVIDER.values()), (
    "the open-weight block label collides with an API provider block, so the "
    "two would merge into one run of rows")


def block_key(m):
    """The table block a model's row belongs to.

    Provider for the API models, one shared block for the open-weight group.
    Nine-model callers are unaffected: for any model in MODEL_PROVIDER this
    returns exactly what PANEL_PROVIDER would have.
    """
    return OPEN_BLOCK if m in OPEN_MODEL_PROVIDER else PANEL_PROVIDER[m]

# Width of the name column in this package's console tables. Derived over the
# COMBINED roster, so a longer model name widens the column instead of running
# into the next one, and so a nine-model table and a twelve-model one align on
# the same left edge. Derived after the open-weight names are known: taking it
# over MODEL_LABEL alone would leave a longer open-weight name to overflow.
NAME_W = max(len(v) for v in PANEL_LABEL.values())

# --------------------------------------------------------------------------
# The other names these nine models go by.
#
# The behavioural report is the upstream artefact the whole appendix is lifted
# from, and it does not use one vocabulary throughout. Three spellings occur,
# and every one of them reaches the page by some route, so all three have to
# be translatable or a single document ends up calling one model three things.
# They live here, beside MODEL_LABEL, so that there is exactly one file to
# read to learn every name a model answers to -- the maps had been sitting in
# the script that happened to need them first, which stopped working as soon
# as a second script needed one.
#
# A fourth spelling needs no map: the SVG charts label their series with
# strings identical to the display names, because the paper's names were
# chosen to be the provider ids minus precisely the two pieces of routing
# metadata the charts had already dropped. src/extract_figures.py states that
# as a checked fact rather than assuming it.
# --------------------------------------------------------------------------

# WHICH MODELS THE FROZEN REPORT ACTUALLY CONTAINS.
#
# The four alias maps below, and every replication check keyed by them
# (clustering_v8, check_split_v8, cost.py's medium self-check), address the
# behavioural report -- a FROZEN artefact under src/report. They can only reach
# models that artefact has rows for.
#
# That set used to be spelled `MODEL_LABEL`, which was correct only for as long
# as the API roster and the report's roster happened to be the same nine
# models. They are not the same thing, and conflating them meant that adding
# any API model to the panel silently widened the alias asserts and the
# replication loops onto keys the frozen JSON will never have -- surfacing as a
# KeyError inside a self-check rather than as a roster error. The open-weight
# group only avoided this by being kept out of MODEL_LABEL entirely (see the
# block above), which is a workaround for this same conflation.
#
# So the report's roster is now named separately. Adding a model to
# MODEL_LABEL no longer touches anything report-facing; a model joins this set
# only when a regenerated report actually contains it.
REPORT_DIRS = ("opus_5", "opus_4_8", "sonnet_5", "3_7_flash", "3_6_flash",
               "3_5_flash", "3_1_pro", "5_6_sol", "5_6_luna", "5_6_terra")
REPORT_MODELS = frozenset(MODEL_LABEL[k] for k in REPORT_DIRS)
N_REPORT_MODELS = len(REPORT_MODELS)
assert REPORT_MODELS <= set(MODEL_LABEL.values()), (
    "REPORT_DIRS names a directory that is not in MODEL_LABEL")

# The provider id verbatim, as the report's roster and endpoint columns print
# it. This is the only key space that still carries `openai.` and `-preview`.
REPORT_RAW_ID = {
    "gemini-3.5-flash": MODEL_LABEL["3_5_flash"],
    "gemini-3.6-flash": MODEL_LABEL["3_6_flash"],
    "gemini-3.7-flash": MODEL_LABEL["3_7_flash"],
    "gemini-3.1-pro-preview": MODEL_LABEL["3_1_pro"],
    "claude-opus-4-8": MODEL_LABEL["opus_4_8"],
    "claude-opus-5": MODEL_LABEL["opus_5"],
    "claude-sonnet-5": MODEL_LABEL["sonnet_5"],
    "openai.gpt-5.6-sol": MODEL_LABEL["5_6_sol"],
    "openai.gpt-5.6-luna": MODEL_LABEL["5_6_luna"],
    "openai.gpt-5.6-terra": MODEL_LABEL["5_6_terra"],
}

# A space-separated short form the report uses in a handful of its generated
# table headers ("3.5 flash").
REPORT_SPACE_FORM = {
    "3.5 flash": MODEL_LABEL["3_5_flash"],
    "3.6 flash": MODEL_LABEL["3_6_flash"],
    # Inert as of the 14-model report -- that build spells the Gemini and
    # Claude rows only in the hyphenated short form, so none of the space-form
    # keys above still match anything either. Carried anyway because the alias
    # asserts below require every map to reach the whole report roster, and a
    # roster-shaped hole here would read as "3.7-flash is not in the report".
    "3.7 flash": MODEL_LABEL["3_7_flash"],
    "3.1 pro": MODEL_LABEL["3_1_pro"],
    "opus 4.8": MODEL_LABEL["opus_4_8"],
    "opus 5": MODEL_LABEL["opus_5"],
    "sonnet 5": MODEL_LABEL["sonnet_5"],
    "5.6-sol": MODEL_LABEL["5_6_sol"],
    "5.6-luna": MODEL_LABEL["5_6_luna"],
    "5.6-terra": MODEL_LABEL["5_6_terra"],
}

# The unqualified short form -- `sol`, `opus-5`. This one is load-bearing in
# two places: it keys reasoning_tokens.json, and it is what the report's own
# prose and table cells say, which the appendix carries across verbatim.
# Matching it in text needs the guard in src/extract_tables.py: `sol` on its
# own also occurs inside `solver`, `solved` and `console`.
REPORT_SHORT = {
    "opus-5": MODEL_LABEL["opus_5"],
    "opus-4.8": MODEL_LABEL["opus_4_8"],
    "sonnet-5": MODEL_LABEL["sonnet_5"],
    "3.7-flash": MODEL_LABEL["3_7_flash"],
    "3.6-flash": MODEL_LABEL["3_6_flash"],
    "3.5-flash": MODEL_LABEL["3_5_flash"],
    "3.1-pro": MODEL_LABEL["3_1_pro"],
    "sol": MODEL_LABEL["5_6_sol"],
    "luna": MODEL_LABEL["5_6_luna"],
    "terra": MODEL_LABEL["5_6_terra"],
}

# Display name -> the key that model has in the report's stats.json, which is
# neither of the above: it is an identifier-safe squashing of the short form
# (`opus5`, `flash36`) invented by the report generator for its JSON dump.
# Every factory-replication check has to go through it.
STATS_KEY = {
    MODEL_LABEL["opus_5"]: "opus5",
    MODEL_LABEL["opus_4_8"]: "opus",
    MODEL_LABEL["sonnet_5"]: "sonnet",
    MODEL_LABEL["3_7_flash"]: "flash37",
    MODEL_LABEL["3_6_flash"]: "flash36",
    MODEL_LABEL["3_5_flash"]: "flash",
    MODEL_LABEL["3_1_pro"]: "pro",
    MODEL_LABEL["5_6_sol"]: "sol",
    MODEL_LABEL["5_6_luna"]: "luna",
    MODEL_LABEL["5_6_terra"]: "terra",
}
assert set(STATS_KEY) == REPORT_MODELS and \
    len(set(STATS_KEY.values())) == N_REPORT_MODELS, \
    "STATS_KEY must be a bijection with the display names the report contains"

# Every id-shaped spelling a model answers to, mapped to the display name.
#
# The report is not consistent about this: its roster and endpoint columns
# print the routed id (`openai.gpt-5.6-sol`, `gemini-3.1-pro-preview`), but the
# CSS panels caption the same models with the bare id (`gemini-3.1-pro`). A
# consumer that reads panel text has to accept either, so it should not be
# reconstructing the union itself -- src/extract_sections.py used to, via a map
# borrowed from another script, and broke the moment that map changed shape.
#
# Note `claude-opus-4-8` versus the display `claude-opus-4.8`: the dash form is
# a real third spelling, not an identity, which is why this is a union of two
# maps rather than "the raw ids, plus dropping a prefix".
REPORT_ANY_ID = {**REPORT_RAW_ID,
                 **{v: v for v in REPORT_MODELS}}

for _name, _alias in (("REPORT_RAW_ID", REPORT_RAW_ID),
                      ("REPORT_SPACE_FORM", REPORT_SPACE_FORM),
                      ("REPORT_SHORT", REPORT_SHORT)):
    assert set(_alias.values()) == REPORT_MODELS, (
        f"{_name} must reach all {N_REPORT_MODELS} models the report "
        f"contains; an unmapped spelling would survive into the paper as a "
        f"bare report string")
    assert len(_alias) == N_REPORT_MODELS, f"{_name} has two keys for one model"
del _name, _alias

assert set(REPORT_ANY_ID.values()) == REPORT_MODELS, \
    "REPORT_ANY_ID must reach every model the report contains"
assert all(REPORT_ANY_ID[k] == v for k, v in REPORT_RAW_ID.items()), \
    "REPORT_ANY_ID disagrees with REPORT_RAW_ID on a routed id"
assert REPORT_ANY_ID.keys() & REPORT_SPACE_FORM.keys() == set(), \
    "an id spelling collides with a space form; callers merge the two maps"

# THE REPORT DOES NOT HAVE ONE ROSTER.
#
# The maps above are ten models, and that ten is structural, not stale. Two of
# the report's artefacts can only ever hold the metered API rows:
#   * reasoning_tokens.json is keyed by REPORT_SHORT and the open-weight
#     endpoints do not break out a reasoning-token count at all, so there is
#     no row to key (src/make_body_figures.py asserts set equality on it);
#   * cost.py's FROZEN_KEYS is keyed by REPORT_DIRS and asserts every entry is
#     priced, while deepseek-v4-flash is deliberately unpriced -- DeepSeek
#     publishes no rate card for that checkpoint and we will not invent one.
# Widening those two would either break the assert or demand a fabricated
# number, so they stay at ten and say so.
#
# The report's *panels* are a different matter. Its stats.json, heatmap,
# category bars, per-scenario walkthroughs and signature exhibits all carry
# the full fourteen. Panel text therefore has to be renamed through a
# fourteen-model union, which is what these two maps are. They are a superset
# of the report maps, so a ten-model caller that keeps using those is
# unaffected.
#
# Spellings taken from the emitted HTML, not guessed: the heatmap header says
# `v4-pro`, `v4-flash`, `kimi-k3`, `glm-5.2`, and the panel captions say the
# display name (`deepseek-v4-pro`), which the identity half already covers.
OPEN_SPACE_FORM = {
    "v4-pro": OPEN_MODEL_LABEL["v4_pro"],
    "v4-flash": OPEN_MODEL_LABEL["dsv4_flash_v8_high_ocr"],
    "kimi-k3": OPEN_MODEL_LABEL["kimi_k3"],
    # The report never writes the display spelling for this one: the panels
    # say `glm-5.2` and only the provenance line carries the `-FP8` that the
    # paper prints. The quantisation is kept in the display name on purpose
    # (see OPEN_MODEL_LABEL), so this map is where the two meet.
    "glm-5.2": OPEN_MODEL_LABEL["glm_5_2"],
}
PANEL_ANY_ID = {**REPORT_ANY_ID,
                **{v: v for v in OPEN_MODEL_LABEL.values()}}
PANEL_SPACE_FORM = {**REPORT_SPACE_FORM, **OPEN_SPACE_FORM}

assert set(PANEL_ANY_ID.values()) == set(PANEL_LABEL.values()), \
    "PANEL_ANY_ID must reach every model the report's panels contain"
assert set(PANEL_SPACE_FORM.values()) == set(PANEL_LABEL.values()), \
    "PANEL_SPACE_FORM must reach every model the report's panels contain"
assert len(PANEL_SPACE_FORM) == N_PANEL, "PANEL_SPACE_FORM has two keys for one model"
# Unlike the ten-model pair, these two key spaces are allowed to touch:
# kimi-k3's short form and its display name are the same string, because it
# is the one model whose report spelling needed no vendor prefix dropped.
# What is not allowed is the two maps disagreeing about what that key means.
assert all(PANEL_SPACE_FORM[k] == PANEL_ANY_ID[k]
           for k in PANEL_ANY_ID.keys() & PANEL_SPACE_FORM.keys()), \
    "an id spelling and a space form name different models; callers merge them"

# The short form, fourteen-wide. For the API models the space form and the
# short form are two different strings ("opus 4.8" against "opus-4.8"), which
# is why REPORT_SPACE_FORM and REPORT_SHORT are separate maps. For the
# open-weight four they are the same string -- none of those names contains a
# space to begin with -- so the same four entries serve both, and reusing them
# is what keeps the two from drifting into different spellings of one model.
PANEL_SHORT = {**REPORT_SHORT, **OPEN_SPACE_FORM}
assert set(PANEL_SHORT.values()) == set(PANEL_LABEL.values()), \
    "PANEL_SHORT must reach every model the report's panels contain"
assert len(PANEL_SHORT) == N_PANEL, "PANEL_SHORT has two keys for one model"

# Directories that exist on disk but are NOT part of the fourteen-model panel.
# Every one is named explicitly, with its reason, rather than filtered by some
# property such as "has no usable rows": a silent filter would drop a model
# the day its data changed shape, whereas an unlisted directory trips the
# assertion in `load_eval` and forces a decision.
#
# The two entries are excluded for opposite reasons, and the difference
# matters when reading any table:
#
#   `5_5` has no data to report. Every row it produced is a provider failure
#   (high 5/5, medium 480/480, low absent), so it contributes zero evaluable
#   episodes. If a rerun ever lands real rows, this entry should go.
#
#   `minimax_m3` has real data and is excluded anyway. Its high-effort run
#   landed on 2026-08-20, after the reported corpus was frozen, and it is
#   structurally complete (open_source_high/MiniMaxAI/minimax_m3/r01, sixty
#   scenario files, same shape as the other three open-weight models). It is
#   held out because admitting it would make the panel fifteen wide and
#   invalidate every count, figure, judge-ablation statistic and guard that
#   the paper states over fourteen models -- not because the run is suspect.
#   Adding it is a corpus decision, not a bug fix.
EXCLUDED_MODELS = {
    "5_5": "aborted run, 100% provider_failure rows",
    "minimax_m3": "complete run, landed after the fourteen-model corpus froze",
}

# SECOND PATHS TO A RUN THAT ALREADY LIVES UNDER OPEN_TREE, mapped to it.
#
# The harness filed each open-weight run under two names: the pooled name
# (OPEN_TREE) and a name describing the input modality it actually ran under.
# THE SECOND NAME IS THE ACCURATE ONE, and reading it is how the modality split
# below was found:
#     aligned_setting_high_ocr/  DeepSeek/v4_pro, ZhipuAI/glm_5_2
#     aligned_setting_high_vlm/  Moonshot/kimi_k3
#     aligned_setting_high/      Moonshot/kimi_k3   (the main sighted tree)
# That is not an arbitrary scatter. v4_pro and glm_5_2 are text-only
# checkpoints and received every asset as an OCR transcription; kimi_k3 is a
# VLM and received the same 7,989 image attachments as the nine API models,
# which is why it is filed in the sighted tree alongside them and they are not.
# See ASSET_MODALITY for the counts and the check that enforces them.
#
# So these paths are DE-DUPLICATED, NOT DISCARDED AS MISTAKES. OPEN_TREE is
# canonical here only because it is the one tree that holds all three; the
# pooling is what makes it the less informative name.
#
# `aligned_setting_high/Moonshot/kimi_k3` is the entry that had to exist: it
# sits in the main high-effort tree, so load_eval globbed it, found no
# MODEL_LABEL entry and aborted -- every script in this package was dead until
# this list existed.
#
# WHY THIS IS NOT A NAME-BASED SKIP. Skipping a path because of what it is
# called would discard a real run the day one lands there. Every entry is
# instead re-proved byte-for-byte on load by `assert_alias_is_duplicate`, so a
# genuine second run at one of these paths diverges and aborts the load rather
# than disappearing as "a known duplicate".
ALIAS_DIRS = {
    "aligned_setting_high/Moonshot/kimi_k3": f"{OPEN_TREE}/Moonshot/kimi_k3",
    "aligned_setting_high_ocr/DeepSeek/v4_pro": f"{OPEN_TREE}/DeepSeek/v4_pro",
    "aligned_setting_high_ocr/ZhipuAI/glm_5_2": f"{OPEN_TREE}/ZhipuAI/glm_5_2",
    "aligned_setting_high_vlm/Moonshot/kimi_k3": f"{OPEN_TREE}/Moonshot/kimi_k3",
}
assert not any(k.startswith(OPEN_TREE + "/") for k in ALIAS_DIRS), (
    "a canonical open-weight tree is listed as an alias of itself; that would "
    "skip the only real copy of the data")

# HOW EACH MODEL RECEIVED THE RENDERED ASSETS. Ten models were sent the image;
# two are text-only checkpoints with no vision path at all and were sent an OCR
# transcription of the same image instead. This is a property of the model, not
# a harness misconfiguration, and it is not the blind ablation: the OCR text
# reproduces the asset's content (with real OCR noise -- "APPRoVED", "wRITE"),
# so the task-critical evidence is present in a different modality rather than
# withheld. The blind arm in ablation.py withholds it outright and is a
# different condition.
#
# Measured over every attempt in the high-effort trees, by counting image
# attachments against asset opens carrying the harness's OCR preamble:
#     ten models     7,989 image attachments each, 0 OCR assets
#     v4_pro         0 images, 2,602 asset opens, 100% OCR
#     glm_5_2        0 images, 4,283 asset opens, 100% OCR
# kimi_k3's image count is not merely nonzero, it is IDENTICAL to all nine API
# models, which is what makes its row comparable to theirs.
#
# Declared here because the tables need it, VERIFIED in cost.py, which is the
# only place that already walks all 182,241 interaction records and can check
# it for free. A table that marks a row as OCR must not be the only thing that
# believes it.
ASSET_MODALITY = {
    "deepseek-v4-pro": "ocr",
    "glm-5.2-fp8": "ocr",
    # Same path-mode condition as the two above, measured the same way: it
    # received asset file paths as text and zero image attachments.
    "deepseek-v4-flash": "ocr",
}
OCR_PREAMBLE = "its text has been extracted by OCR"

_alias_checked: dict = {}


def _tree_digest(root, rel, run=None):
    """md5 over the eval jsonl of one `<tree>/<vendor>/<model>` directory."""
    run = RUN_V8 if run is None else run
    h = hashlib.md5()
    paths = sorted(glob.glob(os.path.join(root, rel, run, "*.jsonl")))
    assert paths, f"no eval jsonl under {rel}/{run}"
    for p in paths:
        h.update(os.path.basename(p).encode())
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def assert_alias_is_duplicate(root, rel, run=None):
    """Abort unless `rel` still byte-matches the copy it is registered against.

    Cached per (root, rel, run): the check reads the whole tree, and a single
    load_eval sweep would otherwise re-read it once per effort level.
    """
    canon = ALIAS_DIRS[rel]
    key = (root, rel, run)
    if key in _alias_checked:
        return
    got = _tree_digest(root, rel, run_for(rel, run))
    want = _tree_digest(root, canon, run_for(canon, run))
    assert got == want, (
        f"{rel} is registered in ALIAS_DIRS as a copy of {canon}, but "
        f"their eval rows no longer match ({got} vs {want}). If a real run "
        f"landed here, give it a panel entry; do not widen this list.")
    _alias_checked[key] = got

# Conflict-category axis, exactly as the generator defines it. `challenge` is
# the 7th declared category but is NOT an axis bucket: all 20 seeds carry
# challenge plugins, so that bucket is the whole corpus and discriminates
# nothing (design_axis_slice.py makes the same exclusion, for the same stated
# reason). Order is the shipped axis_order.category.
CAT_ORDER = ("inconsistency", "coordination", "order", "timeout",
             "subagent_collaboration", "main_supervised")

# ---- check taxonomy: mirrored verbatim from the factory report generator ----
# docs/analysis/analysis_v8/tools/build_behavior_report.py:1196-1219 and :340.
# Do not "improve" these predicates; the whole point is that the paper and the
# behavioural report can never disagree about the same record.
#
# A check's `name` is "effect:<effect_type> <condition>". Buckets are decided by
# SUBSTRING on the name, over checks of EVERY role (the factory does not filter
# on role here -- role filtering happens only for the clustering vectors).
JUDGE_MSG_PREFIX = "LLM semantic match:"
_LEGACY_FAIL_CLOSED_MARKERS = ("(empty transcript)", "no file ", "empty criterion")
ASSET_GATE_PREFIX = "effect:assets_accessed"


# --------------------------------------------------------------------------
# population
# --------------------------------------------------------------------------
def load_dead(label=DEAD_LABEL_V8, expect=N_DEAD_V8):
    """{(slug, round_id)} for one corpus.

    Slugs carry the variant suffix (`access_request_approval_v03`), which is
    what the eval jsonl basenames use.
    """
    out = {(s, r) for c, s, r in json.load(open(DEAD_JSON))["exclude"] if c == label}
    assert len(out) == expect, f"dead list for {label!r}: expected {expect}, got {len(out)}"
    return out


def round_id_of(row):
    md = row.get("metadata")
    if isinstance(md, dict) and md.get("round_id"):
        return str(md["round_id"])
    m = re.search(r"_(r\d+)$", row.get("task_id") or "")   # fallback only
    return m.group(1) if m else ""


def _cap_hit(row):
    return str(row.get("stop_reason") or "").startswith("reached max_steps")


def load_eval(effort=PRIMARY_EFFORT, root=ROOT_V8, run=RUN_V8,
              population=PRIMARY_POPULATION, keep_checks=True, dead=None,
              roster=None, panel=None, tree=None):
    """model label -> {task_id: episode dict}.

    Episode keys: task_id slug round_id success score steps stop_reason
    cap_hit dead is_error checks metadata. `checks` is the raw check_results
    list (or None when keep_checks is False, to keep whole-sweep loads small).

    Also returns per-model error counts. In the "full" view error rows are
    RETAINED (they are part of the 600 and the report counted them), but the
    tally is still returned so a table can report it.

    The dead-round set is resolved and verified in BOTH views. In "full" the
    rounds are kept and merely flagged on the episode, but the guard that
    every dead (slug, round_id) matched a real eval row still runs -- that
    guard is what catches a round_id parsing regression, and it must not go
    away just because the paper stopped excluding them.

    `roster` is the exact set of model LABELS the caller expects the tree to
    contain; None means every model in `panel`. It exists for the ablation
    trees (analysis/ablation.py), which re-run a subset of the panel under a
    changed step budget or a stripped modality and so legitimately hold three
    or four models rather than nine. It is checked by SET EQUALITY, not by
    count: a tree that is missing an expected model and a tree that carries an
    unexpected extra one both abort, which is strictly stronger than the
    `len(out) == N_MODELS` test this replaced. Do NOT pass a roster merely to
    silence that assertion on the main tree -- the whole point is that a cell
    which failed to produce output cannot pass unnoticed as a smaller panel.

    `panel` is the directory -> display-name map the tree is read against, and
    `tree` is the subdirectory of `root` to glob. Both default to the nine
    API models under `aligned_setting_<effort>`, so every pre-existing caller
    keeps its exact behaviour; `load_open` passes the open-weight pair. They
    are separate arguments on purpose: the panel decides which directories are
    legal and what they are called, the tree decides where to look, and the
    modality-ablation trees are an existing case of the same panel in a
    different tree.
    """
    assert population in POPULATIONS, population
    panel = MODEL_LABEL if panel is None else panel
    tree = f"aligned_setting_{effort}" if tree is None else tree
    drop_dead = population == "excluded"
    dead = load_dead() if dead is None else dead
    out = defaultdict(dict)
    errors = defaultdict(int)
    hit = set()
    per_cell_dead = defaultdict(int)

    # Globbed across every run subdirectory and then filtered, rather than
    # globbed at a fixed `{run}`, so that RUN_OVERRIDE can select a different
    # run for a single cell. `run` remains the default for every cell that has
    # no override, so callers that pass one keep their exact behaviour.
    for path in glob.glob(f"{root}/{tree}/*/*/*/*.jsonl"):
        raw_model = path.split("/")[-3]
        rel_dir = "/".join(path.split("/")[-5:-2])
        if path.split("/")[-2] != run_for(rel_dir, run):
            continue
        if rel_dir in ALIAS_DIRS:
            # A second path to a run already read from OPEN_TREE, not a second
            # run. The bytes are re-proved rather than trusted; see ALIAS_DIRS.
            assert_alias_is_duplicate(root, rel_dir, run)
            continue
        if raw_model in EXCLUDED_MODELS:
            continue
        assert raw_model in panel, (
            f"unmapped model directory {raw_model!r} under {root}/{tree}; add "
            f"it to the panel rather than letting the raw name into a table")
        model = panel[raw_model]
        slug = os.path.basename(path)[:-6]
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tid = row.get("task_id") or ""
            rid = round_id_of(row)
            assert rid, f"could not resolve round_id for {tid!r} in {path}"
            is_dead = (slug, rid) in dead
            if is_dead:
                hit.add((slug, rid))
                per_cell_dead[model] += 1
                if drop_dead:
                    continue
            is_error = bool(row.get("error"))
            if is_error:
                errors[model] += 1
                if drop_dead:
                    continue
            md = row.get("metadata") or {}
            out[model][tid] = {
                "task_id": tid, "slug": slug, "round_id": rid,
                "success": bool(row.get("success")),
                "score": float(row.get("score") or 0.0),
                "steps": row.get("steps"),
                "stop_reason": row.get("stop_reason"),
                "cap_hit": _cap_hit(row),
                "dead": is_dead,
                "is_error": is_error,
                "checks": (md.get("check_results") or []) if keep_checks else None,
                "metadata": md,
            }

    want_models = set(panel.values()) if roster is None else set(roster)
    assert set(out) == want_models, (
        f"{effort} @ {root}/{tree}: roster mismatch. "
        f"missing={sorted(want_models - set(out))} "
        f"unexpected={sorted(set(out) - want_models)}")
    # These guards run in BOTH views. The paper no longer excludes dead rounds,
    # but resolving them is still the canary for a round_id parsing regression.
    missed = dead - hit
    assert not missed, (
        f"{len(missed)} dead rounds never matched any eval row -- round_id "
        f"resolution is broken. examples: {sorted(missed)[:5]}")
    best = max(per_cell_dead.values())
    assert best == N_DEAD_V8, (
        f"best-covered cell saw {best} dead rounds, expected {N_DEAD_V8}")
    want = N_RAW if population == "full" else N_EXCLUDED
    for m, eps in out.items():
        got = len(eps) if population == "full" else len(eps) + errors[m]
        assert got == want, (
            f"{m} @ {effort}/{population}: {got} rounds != {want}")
    return dict(out), dict(errors)


def load_sweep(root=ROOT_V8, run=RUN_V8, population=PRIMARY_POPULATION,
               keep_checks=False):
    """effort -> (episodes, errors), all three levels.

    Nine API models only. The open-weight group has no medium or low run, so
    it cannot appear in a sweep; a table that wants it has to be a high-effort
    table and should call `load_panel`.
    """
    dead = load_dead()
    return {lvl: load_eval(lvl, root, run, population, keep_checks, dead)
            for lvl in LEVELS}


def load_open(root=ROOT_V8, run=RUN_V8, population=PRIMARY_POPULATION,
              keep_checks=True, dead=None):
    """The open-weight models, from their own trees. High effort only.

    Most of them live under OPEN_TREE; OPEN_EXTRA_TREES names the ones that do
    not. Each tree is loaded against its OWN roster, so a cell that failed to
    produce output aborts here rather than being absorbed as a smaller
    combined panel -- the same property `load_eval`'s roster assert gives the
    API side.
    """
    dead = load_dead() if dead is None else dead
    extra = {d for panel in OPEN_EXTRA_TREES.values() for d in panel}
    pooled = {lbl for d, lbl in OPEN_MODEL_LABEL.items() if d not in extra}
    eps, errs = load_eval(OPEN_EFFORT, root, run, population, keep_checks,
                          dead, roster=pooled, panel=OPEN_MODEL_LABEL,
                          tree=OPEN_TREE)
    for tree, panel in OPEN_EXTRA_TREES.items():
        e_eps, e_errs = load_eval(OPEN_EFFORT, root, run, population,
                                  keep_checks, dead,
                                  roster=set(panel.values()), panel=panel,
                                  tree=tree)
        assert not (eps.keys() & e_eps.keys()), (
            f"{tree} re-reads {sorted(eps.keys() & e_eps.keys())}, which the "
            f"pooled tree already supplied; one copy would be dropped")
        eps.update(e_eps)
        errs.update(e_errs)
    return eps, errs


def load_panel(effort=PRIMARY_EFFORT, root=ROOT_V8, run=RUN_V8,
               population=PRIMARY_POPULATION, keep_checks=True, dead=None):
    """The nine API models plus, at high effort, the three open-weight ones.

    The two groups are loaded from separate trees against separate panels and
    merged here, so each tree is still checked against its OWN roster and its
    own n: a missing open-weight cell cannot be absorbed as a smaller combined
    panel, which is the failure `load_eval`'s roster assert exists to prevent.

    At medium and low this returns exactly what `load_eval` returns, because
    the open-weight group was not run there. Callers that must have all twelve
    should assert on the roster they get back rather than assuming twelve.
    """
    dead = load_dead() if dead is None else dead
    eps, errs = load_eval(effort, root, run, population, keep_checks, dead)
    if effort != OPEN_EFFORT:
        return eps, errs
    o_eps, o_errs = load_open(root, run, population, keep_checks, dead)
    assert not (eps.keys() & o_eps.keys()), (
        f"panels overlap on {sorted(eps.keys() & o_eps.keys())}; the merge "
        f"would drop one group's episodes")
    return {**eps, **o_eps}, {**errs, **o_errs}


# --------------------------------------------------------------------------
# spec side
# --------------------------------------------------------------------------
def base_slug(slug):
    """scenario_slug -> base seed name (strip the trailing _v<NN> variant)."""
    return re.sub(r"_v\d+$", "", slug or "")


def load_axis_map():
    """Seed -> conflict categories, as routed by the generator's own registry.

    The mapping is cached (seed_axis_map.json) so the paper's analysis tree
    does not need the source checkout importable, but it is verified against
    the shipped design_axes.json on every load: identical plugin->category
    table and identical seeds-per-category counts. If the registry ever moves,
    this fails loudly instead of drifting.
    """
    m = json.load(open(AXIS_MAP))
    if os.path.exists(DESIGN_AXES):
        d = json.load(open(DESIGN_AXES))
        assert m["plugin_category"] == d["plugin_category"], (
            "cached plugin->category table disagrees with design_axes.json")
        mine = defaultdict(int)
        for v in m["seeds"].values():
            for c in v["categories"]:
                mine[c] += 1
        any_model = next(iter(d["by_model"].values()))
        fac = {k: v["seeds"] for k, v in any_model["axes"]["category"].items()}
        assert dict(mine) == fac, f"seeds/category drift: {dict(mine)} vs {fac}"
        assert tuple(d["axis_order"]["category"]) == CAT_ORDER, "category order drift"
    return m


def load_gate_map():
    """The generator's round-level conflict-protocol gate, as a two-arm design.

    The harness does not ask every round to RESOLVE its conflict. Six of the
    ten positions in a package (r02, r03, r05, r06, r08, r09) are selected by a
    deterministic rule -- ``floor(i*2/3) > floor((i-1)*2/3)`` -- and on those
    the gold trajectory must carry out a real state-changing protocol step
    rather than merely inspect the evidence; r09 must COMPLETE the protocol.
    The other four rounds carry the same conflicting evidence and ask only that
    the agent investigate it.

    The rule fires only for seeds that route to a plugin declaring
    ``emits_effects``. For the eight that do not, every injection and gate is a
    no-op, so the identical round positions carry no extra demand. That is what
    makes this a design and not just a contrast: the same positions, the same
    packages, the same models, with the treatment switched off by a property of
    the seed that has nothing to do with round index. The placebo arm measures
    whatever ROUND POSITION alone is worth, and the difference of the two
    differences is the part attributable to being asked to resolve.

    Cached (mechanic_gate.json) so the analysis tree runs even where the harness
    is absent, and re-derived from the live registry whenever CLASHARNESS_SRC
    points at a checkout -- field for field, so a plugin that changes its
    declared effects fails here instead of quietly moving a published number.
    The default is this repository, so the re-derivation runs by default.
    """
    m = json.load(open(GATE_MAP))
    assert m["ratio"] == [2, 3], f"unexpected gate ratio {m['ratio']}"
    src = os.environ.get("CLASHARNESS_SRC", REPO_ROOT)
    if src and os.path.isdir(src):
        if src not in sys.path:
            sys.path.insert(0, src)
        try:
            from clasharness.scenario.conflict_llm_pipeline import (  # noqa: E501
                MECHANIC_ROUND_RATIO, _final_mechanic_round,
                _mechanic_command_sets, _round_requires_mechanic)
            from clasharness.scenario.conflict_vlm_spec import (
                load_conflict_vlm_seed_file)
            from clasharness.scenario.plugins import get_plugins_for_taxonomy
        except ImportError:                       # not a harness checkout
            return m
        r = m["rounds_per_package"]
        assert abs(MECHANIC_ROUND_RATIO - 2.0 / 3.0) < 1e-12, (
            f"harness ratio is now {MECHANIC_ROUND_RATIO}, cache says 2/3")
        assert m["gated_positions"] == [
            i for i in range(1, r + 1) if _round_requires_mechanic(i)], (
            "gated round positions drifted from the generator's rule")
        assert m["terminal_position"] == _final_mechanic_round(r), (
            "terminal round drifted from the generator's rule")
        seeds = os.path.join(
            src, "clasharness", "data", "scenario_seeds", "conflict_vlm_v1")
        live = {}
        for f in sorted(glob.glob(os.path.join(seeds, "*.json"))):
            s = load_conflict_vlm_seed_file(f)
            cmds = _mechanic_command_sets(
                get_plugins_for_taxonomy(
                    (s.conflict_taxonomy or []) + (s.difficulty_challenges or [])),
                s.allowed_tool_families)
            live[s.slug] = {"mechanic": bool(cmds),
                            "effects": list(cmds.effects),
                            "protocol": list(cmds.protocol),
                            "terminals": list(cmds.terminals)}
        assert live == m["seeds"], (
            "cached mechanic_gate.json disagrees with the live registry on "
            f"{sorted(k for k in set(live) | set(m['seeds']) if live.get(k) != m['seeds'].get(k))}")
    return m


def gate_arm(seed, round_id, gate=None):
    """(arm, gated) for one episode: which side of the ablation it sits on.

    arm    "mechanic" if the gate can fire for this seed, else "placebo"
    gated  whether this round position is one the gate selects

    Both arms are keyed on the SAME positions, so the placebo cells are the
    control for round position and the mechanic cells are the control for seed
    identity. Callers that only want the treated contrast should still compute
    the placebo one -- reporting the mechanic drop alone is the version of this
    result that a reviewer can dismiss as a round-order artefact.
    """
    g = load_gate_map() if gate is None else gate
    s = g["seeds"].get(seed)
    assert s is not None, f"seed {seed!r} is not in the gate map"
    pos = int(str(round_id).lstrip("r") or 0)
    assert 1 <= pos <= g["rounds_per_package"], (
        f"round {round_id!r} is outside the {g['rounds_per_package']}-round package")
    return ("mechanic" if s["mechanic"] else "placebo"), pos in g["gated_positions"]


def gold_length(ground_truth):
    """Real commands in a gold trajectory, excluding the terminal sentinel.

    The single definition of gold length for the whole paper -- prose, tables
    and the generator ablation all route through here so that the slack
    denominator can never drift away from the budget numerator again. The
    assertion is the point: if a regenerated corpus stops terminating its gold
    trajectories with GOLD_SENTINEL, dropping the last command is no longer
    correct and this should fail loudly rather than silently shorten every
    trajectory by one.
    """
    cmds = ground_truth.get("solution_commands") or []
    if not cmds:
        return 0
    last = str(cmds[-1]).strip()
    assert last == GOLD_SENTINEL, (
        f"gold trajectory ends in {last[:60]!r}, not the {GOLD_SENTINEL!r} "
        "sentinel; re-check whether excluding the final command is still right")
    return len(cmds) - 1


def load_specs(tasks=TASKS_V8):
    """task_id -> spec-side facts.

    gold      real commands in ground_truth.solution_commands (see gold_length:
              the trailing DONE sentinel is not a step)
    slack     step budget / gold length
    units     evaluation_checks with role == "required"
    diag      evaluation_checks that are not required
    rtype     round type
    cats      conflict categories of the seed (OVERLAPPING -- a task is in
              1..n category buckets, so category rows never sum to the corpus)
    """
    axis = load_axis_map()["seeds"]
    out = {}
    for p in glob.glob(f"{tasks}/*/specs.json"):
        d = json.load(open(p))
        gt = d.get("ground_truth") or {}
        checks = gt.get("evaluation_checks") or []
        gold = gold_length(gt)
        pub = ((d.get("data") or {}).get("public") or {})
        slug = pub.get("scenario_slug") or ""
        seed = base_slug(slug)
        assert seed in axis, f"task {d['task_id']}: seed {seed!r} not in axis map"
        out[d["task_id"]] = {
            "gold": gold,
            "slack": (BUDGET_V8 / gold) if gold else None,
            "units": sum(1 for c in checks if c.get("role") == "required"),
            "diag": sum(1 for c in checks if c.get("role") != "required"),
            "rtype": pub.get("round_type") or (gt.get("metadata") or {}).get("round_type"),
            "slug": slug,
            "seed": seed,
            "cats": tuple(axis[seed]["categories"]),
            "vision": axis[seed]["vision"],
            "async": axis[seed]["async"],
        }
    assert len(out) == N_RAW, f"expected {N_RAW} specs, got {len(out)}"
    per_seed = defaultdict(int)
    for v in out.values():
        per_seed[v["seed"]] += 1
    assert set(per_seed.values()) == {30}, f"seeds are not 30 rounds each: {dict(per_seed)}"
    return out


# --------------------------------------------------------------------------
# check-level helpers
# --------------------------------------------------------------------------
def scored_checks(ep):
    """The checks that carry score. Diagnostics are excluded."""
    return [c for c in (ep["checks"] or []) if c.get("role") == "required"]


def check_type(check):
    """"effect:messages_sent field_semantic_match" -> "field_semantic_match"."""
    toks = str(check.get("name") or "").split()
    return toks[-1] if toks else ""


def check_bucket(check):
    """Top-level bucket. Mirrors build_behavior_report.py:1198.

    semantic_match  LLM-judged
    field_contains  deterministic substring match, splits further
    other           every remaining deterministic condition
    """
    nm = str(check.get("name") or "")
    if "semantic" in nm:
        return "semantic_match"
    if "field_contains" in nm:
        return "field_contains"
    return "other"


def check_subbucket(check):
    """The ability split inside field_contains. Mirrors :1205-1208.

    asset_gate  did the agent reach the right artefact at all
    content     did the file it wrote carry the exact string
    """
    if check_bucket(check) != "field_contains":
        return None
    return ("asset_gate" if str(check.get("name") or "").startswith(ASSET_GATE_PREFIX)
            else "content")


def judge_reached(check):
    """Did a judge actually return a verdict? Mirrors :340-367 exactly.

    A deterministic checker never asks a judge, so it answers True -- "not
    reached" must not be reported as judge attrition for it. The structured
    flag wins unconditionally; the prose markers are a legacy fallback and are
    scoped to judge-authored messages.
    """
    invoked = check.get("judge_invoked")
    if invoked is not None:
        return bool(invoked)
    message = str(check.get("message") or "")
    if not message.startswith(JUDGE_MSG_PREFIX):
        return True
    return not any(m in message for m in _LEGACY_FAIL_CLOSED_MARKERS)


def check_status(check):
    """graded_pass / graded_fail / infra / not_invoked. Precedence per :1214.

    `graded` in the factory report is total - infra - not_invoked, so a check
    that the judge never reached is neither a pass nor a fail: it is missing
    measurement, and pooling it as a failure would flatter or punish a model
    for an attrition artefact.
    """
    if not check.get("ok", True):
        return "infra"
    if not judge_reached(check):
        return "not_invoked"
    return "graded_pass" if check.get("passed") else "graded_fail"


_VOTE_RE = re.compile(r"\[self-consistency (\d+)/(\d+) passed\]")


def judge_votes(check):
    """(passed_votes, total_votes) for a semantic check, or None.

    The judge is not a single sample. Every semantic verdict is decided by a
    self-consistency ensemble whose tally the harness writes into the message
    verbatim -- "[self-consistency 2/3 passed]". Parsing it back out is the
    only way the paper can say anything about judge RELIABILITY rather than
    about judge AGREEMENT WITH SOMETHING ELSE, and it costs no new data.

    Returns None for the two cases that legitimately carry no tally: any
    deterministic check (no judge ran) and any semantic check the judge never
    reached (`judge_reached` false -- the run died before producing anything
    to grade). Callers must treat None as "no measurement", never as a tie.

    A verdict is UNANIMOUS when k is 0 or n and SPLIT otherwise. Split verdicts
    are the judge disagreeing with itself under identical input, which is the
    tightest available bound on how much of the grade is judge noise: it is
    measured per check over ~19k checks, not inferred from a rank correlation
    over ten models.
    """
    if check_bucket(check) != "semantic_match":
        return None
    m = _VOTE_RE.search(str(check.get("message") or ""))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def judge_split(check):
    """True when the judge ensemble did not agree with itself on this check."""
    v = judge_votes(check)
    return v is not None and v[0] not in (0, v[1])


# --------------------------------------------------------------------------
# failure clustering -- mirrored verbatim from build_behavior_report.py:496-681
# --------------------------------------------------------------------------
# Copied rather than re-derived, for the same reason as the check predicates:
# the paper and the shipped report must not be able to disagree about the same
# record. check_split_v8 / clustering_v8 re-verify these against stats.json at
# medium before reporting anything at high.
#
# READ THE SCOPE NOTE AT :605 BEFORE USING WHOLE-CORPUS LIFT FOR ANYTHING.
# The independence null is taken over a corpus with a wide difficulty range, so
# failures pile onto hard tasks even for a model whose checks fail
# independently, and EVERY model therefore looks clustered. The whole-corpus
# lift is descriptive shape, not a model property. Any claim about a model must
# come from `stratified_lift` (difficulty terciles) or `paired_peer_lift`
# (same tasks, all models resampled together).
def co_failure_lift(pairs):
    """P(a 2nd required check fails | one already did) / marginal failure rate.

    `pairs` is [(k_i, f_i)]. Under independence this is exactly 1.0 as an
    algebraic identity -- nothing is estimated for the null, which is why the
    factory prefers it to overdispersion.

    Bounded above by 1/q where q is the model's own failure rate, and q differs
    by model, so RANKS are comparable across models but MAGNITUDES are not.
    Always report lift_ceiling alongside.
    """
    den = sum(f * (k - 1) for k, f in pairs)
    tot_f = sum(f for _, f in pairs)
    tot_k = sum(k for k, _ in pairs)
    if not den or not tot_f or not tot_k:
        return float("nan")
    return (sum(f * (f - 1) for _, f in pairs) / den) / (tot_f / tot_k)


def stable_seed(*parts):
    """A seed that is identical in every process, for naming a bootstrap stream.

    Python randomizes `hash()` on str/bytes per interpreter run (PEP 456), so
    `seed=hash(name)` silently produces a DIFFERENT bootstrap on every
    invocation -- which is exactly the reproducibility that boot_ci promises.
    That bug shipped here once: capability_axes_v8 seeded its per-category
    half-widths with `hash(category)` and its `boot +-` row moved by up to
    0.4 points between two consecutive runs of the same script on the same
    data. crc32 is stable across processes, versions and platforms.

    Use this whenever a bootstrap stream is named after something (a category,
    a model, a stratum) rather than seeded from a count in the data.
    """
    key = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return zlib.crc32(key)


DRAWS = 10000
"""Bootstrap resamples behind the intervals this repo COMPUTES.

Scope, because there are two budgets and conflating them corrupts the paper.
This one governs numbers we derive ourselves: Table 1's +- standard errors and
percentile intervals (via difficulty_bands, which now reads this constant) and
the capability-axis half-widths (via boot_ci's default). `FACTORY_DRAWS` below
governs the replication path and is a different number for a reason.

Why 10,000 rather than the 4,000 that shipped previously. The number is not
chosen to make the estimates better; it is chosen to make the PRINTED digit
stable. A bootstrap SE is itself estimated with Monte Carlo error ~ s/sqrt(2B),
which at B=4,000 is ~0.02 points on a table that prints one decimal, i.e.
comparable to the 0.05 half-width of a rounding cell: 17.7% of the 240 pass and
score cells printed a last digit that differed from the B->infinity answer, and
claude-opus-5's headline SE was observed flipping between 1.7 and 1.8 across
seeds. At 10,000 that rate is 11.0% and the same cell is stable.

Read that as presentation, not precision. The SE is itself uncertain to
~s/sqrt(2n) from having only n rounds -- 0.037 points at n=600, 0.146 at
n=140 -- and no value of B touches that. Monte Carlo was already the minority
term at 4,000 (19-39% of the statistical uncertainty, and under 15% of its
variance); at 10,000 the TOTAL uncertainty of an SE falls by about 4%. Driving
the misprint rate below 2% would need B=160,000. So this buys a conventional
number and a stable last digit, and the honest accounting is that it buys
little else.

The cost is real and recurring: 2.5x the bootstrap work, which is most of the
verification budget (verify_table_bodies.py plus results_facts.py go from about
11 minutes to about 27). Do not raise it further without a reason that survives
the arithmetic above.

NOT the permutation count in capability_axes_v8.specialization(), which draws
random seed SETS to build a null rather than resampling rounds, and whose
resolution question is 1/(nperm+1) on a p-value. That one is pinned separately
as cap_nperm and is deliberately left where it is.
"""

FACTORY_DRAWS = 4000
"""Bootstrap resamples the FROZEN artifacts were built with. Do not raise.

This is not a stale copy of DRAWS that someone forgot to bump. It is the draw
count baked into two things this repo cannot recompute:

  * `cross-model-behavioral-comparison.stats.json`, which clustering_v8's
    `verify_against_factory` reproduces and asserts on -- its tolerance on
    lift_ci is one rounding step (0.0101), and Monte Carlo noise on a
    percentile endpoint is roughly four times larger than on an SE, so
    resampling a different number of times is exactly the way to make that
    assertion fire for no scientific reason.
  * `report/cross-model-behavioral-comparison.md`, whose tables extract_tables
    parses VERBATIM into the appendix. Those cells were computed at 4,000 and
    their captions say 4,000. Raising this constant would not recompute a
    single one of them; it would only make the surrounding prose describe an
    experiment that was never run.

So the functions on the replication path -- failure_clustering,
stratified_lift, paired_peer_lift -- default to this, and clustering_v8.DRAWS
is now an alias of it rather than a hand-kept 4000 with a comment. The comment
was right; it just was not enforced.

The consequence to accept: the appendix's clustering intervals are drawn 4,000
times and the main table's are drawn 10,000, and that asymmetry is real. It is
the correct asymmetry. One side is a reproduction of a fixed document and the
other is ours to choose.
"""


def boot_vals(pairs, fns, seed, draws=DRAWS):
    """Bootstrap replicates of SEVERAL functionals over ONE resample stream.

    `fns` is a sequence of callables. Each is evaluated on the SAME resampled
    list and gets its own list of replicate values back, NaNs dropped per
    functional so that one degenerate statistic cannot silently shorten
    another's sample.

    ONE STREAM, NOT ONE PER FUNCTIONAL. A cell's pass rate, that rate's
    interval and that rate's spread are three readings of a single bootstrap;
    computing them in separate loops would give the paper two or three
    defensible numbers that disagree in the last digit, which is the failure
    mode table1_v8.py's docstring exists to prevent. It also halves the work
    when a caller wants a point estimate and a standard error together.

    RNG CONSUMPTION IS FIXED AT n DRAWS PER REPLICATE no matter how many
    functionals are passed, so boot_vals(pairs, (fn,), seed) walks exactly the
    stream boot_ci(pairs, fn, seed) walked before it was refactored onto this.
    That invariant is what makes every interval already in the paper survive
    the refactor unchanged, and it is why the resample is built ONCE per draw
    and then handed to each fn rather than rebuilt inside the loop.
    """
    rnd = random.Random(seed)
    n = len(pairs)
    out = [[] for _ in fns]
    if n < 2:
        return out
    for _ in range(draws):
        samp = [pairs[rnd.randrange(n)] for _ in range(n)]
        for k, fn in enumerate(fns):
            v = fn(samp)
            if v == v:
                out[k].append(v)
    return out


def pct_ci(vals, alpha=0.025):
    """Percentile interval of a replicate list. Split out of boot_ci so that a
    caller holding boot_vals output can take an interval off it WITHOUT
    resampling a second time."""
    if not vals:
        return (float("nan"), float("nan"))
    vals = sorted(vals)
    return (vals[int(alpha * len(vals))], vals[int((1 - alpha) * len(vals))])


def boot_sd(vals):
    """Sample SD of a replicate list -- i.e. the bootstrap standard error.

    Takes the REPLICATES, not (pairs, fn, seed), on purpose. A signature that
    accepted the data would let a caller draw a fresh stream for the spread
    after already drawing one for the interval, and the table would then carry
    a point estimate whose +- and whose bracket came from different draws.
    Forcing the caller to hand over boot_vals output makes that impossible to
    write by accident.
    """
    if len(vals) < 2:
        return float("nan")
    mu = sum(vals) / len(vals)
    return (sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def boot_ci(pairs, fn, seed, draws=DRAWS, alpha=0.025):
    """Percentile bootstrap over TASKS. Local RNG explicitly seeded, so two
    runs give byte-identical intervals.

    `seed` must be reproducible across processes: either a count taken from
    the data, or stable_seed(...) for a named stream. Never `hash(str)`.

    A thin wrapper over boot_vals since the +- columns landed; see the RNG
    note there for why this still returns exactly what it always did.
    """
    (vals,) = boot_vals(pairs, (fn,), seed, draws)
    return pct_ci(vals, alpha)


def ci_tail(vals, null=0.0):
    """95% interval plus the share of draws contradicting the point estimate."""
    vals = sorted(vals)
    if not vals:
        return float("nan"), float("nan"), float("nan")
    lo = vals[int(0.025 * len(vals))]
    hi = vals[int(0.975 * len(vals))]
    below = sum(1 for v in vals if v <= null)
    return lo, hi, min(below, len(vals) - below) / len(vals)


def required_vectors(episodes):
    """task_id -> [passed?] over that task's required checks, SORTED BY task_id.

    Tasks with no required checks are dropped, exactly as the factory does
    (`vectors = [v for v in vectors if v]`). Provider-error rows carry no
    checks and so fall out here, which is why a model with errors shows a
    slightly smaller n_tasks than its peers.

    The sort is not cosmetic. Every point estimate downstream is a sum and so
    is order-invariant, but the bootstrap draws an INDEX into this sequence, so
    a different ordering resamples different tasks and shifts a CI endpoint by
    about 0.01. Insertion order here would be glob order, which is filesystem-
    dependent -- the paper's intervals would then not be reproducible on
    another machine. Sorting by task_id pins them.
    """
    out = {}
    for tid in sorted(episodes):
        v = [bool(c.get("passed")) for c in scored_checks(episodes[tid])]
        if v:
            out[tid] = v
    return out


def failure_clustering(vectors, draws=FACTORY_DRAWS):
    """Whole-corpus clustering summary. See the scope note above."""
    vectors = [v for v in vectors if v]
    if not vectors:
        return None
    n = len(vectors)
    ks = [len(v) for v in vectors]
    n_checks = sum(ks)
    n_passed = sum(sum(1 for x in v if x) for v in vectors)
    p = n_passed / n_checks
    obs = sum(1 for v in vectors if all(v)) / n
    indep = sum(p ** k for k in ks) / n      # mean_i(p**k_i), NOT p**mean(k)
    fails = [len(v) - sum(1 for x in v if x) for v in vectors]
    mean_k = n_checks / n
    var_k = sum((k - mean_k) ** 2 for k in ks) / n
    mean_f = sum(fails) / n
    var_obs = sum((f - mean_f) ** 2 for f in fails) / n
    # The k-heterogeneity term is mandatory; the coefficient is (1-p), the
    # FAILURE rate, because f ~ Bin(k, 1-p).
    var_indep = mean_k * p * (1 - p) + (1 - p) ** 2 * var_k
    pairs = list(zip(ks, fails))
    lo, hi = boot_ci(pairs, co_failure_lift, seed=n_passed, draws=draws)
    q = 1 - p
    return {
        "n_tasks": n, "n_checks": n_checks, "mean_k": round(mean_k, 2),
        "per_check_pct": round(100 * p, 1),
        "obs_pass_pct": round(100 * obs, 1),
        "indep_pass_pct": round(100 * indep, 1),
        "excess_pp": round(100 * (obs - indep), 1),
        "lift": round(co_failure_lift(pairs), 2),
        "lift_ci": [round(lo, 2), round(hi, 2)],
        "lift_ceiling": round(1 / q, 2) if q else None,
        "overdispersion": round(var_obs / var_indep, 2) if var_indep else None,
    }


def kf_table(per_model_vectors):
    """model -> {task: (k, f)} restricted to tasks EVERY model was graded on.

    Returns (kf, common). Like-for-like is not optional here: the stratified
    and paired statistics compare models to each other, so a task one model
    lost to a provider error must leave the comparison for all of them.
    """
    common = None
    for v in per_model_vectors.values():
        s = set(v)
        common = s if common is None else (common & s)
    common = sorted(common or [])
    kf = {m: {t: (len(v[t]), len(v[t]) - sum(v[t])) for t in common}
          for m, v in per_model_vectors.items()}
    return kf, common


def stratified_lift(kf, common, model, draws=FACTORY_DRAWS, seed=0):
    """Lift inside leave-one-out difficulty terciles. Mirrors :752-775.

    Difficulty is the pooled required-check FAILURE rate of the OTHER models,
    so a model never helps define the axis it is judged on. Terciles are
    ordered easiest first.
    """
    peers = [m for m in kf if m != model]
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
        pairs = [kf[model][t] for t in group]
        lo, hi = boot_ci(pairs, co_failure_lift, seed=seed + j, draws=draws)
        tiers.append({"n": len(group), "lift": co_failure_lift(pairs),
                      "ci": (lo, hi),
                      "verdict": ("up" if lo > 1.0 else
                                  "down" if hi < 1.0 else "flat")})
    return tiers


def paired_peer_lift(kf, common, draws=FACTORY_DRAWS, seed=77001):
    """Each model's lift minus the mean of its peers, ON THE SAME RESAMPLE.

    One shared task resample per draw with all models recomputed together, so
    corpus difficulty cancels. This -- not whole-corpus lift -- is the
    statistic that supports a claim about a model.
    """
    keys = sorted(kf)
    rnd = random.Random(seed)
    n = len(common)
    collected = {k: [] for k in keys}
    for _ in range(draws):
        tasks = [common[rnd.randrange(n)] for _ in range(n)]
        lifts, ok = {}, True
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
    full = {k: co_failure_lift([kf[k][t] for t in common]) for k in keys}
    out = {}
    for k in keys:
        peer_mean = sum(full[m] for m in keys if m != k) / (len(keys) - 1)
        lo, hi, pv = ci_tail(collected[k])
        out[k] = {"lift": full[k], "peer_mean": peer_mean,
                  "diff": full[k] - peer_mean, "ci": (lo, hi), "p": pv,
                  "verdict": ("up" if lo > 0 else "down" if hi < 0 else "flat")}
    return out


# --------------------------------------------------------------------------
# stats (no scipy on this machine)
# --------------------------------------------------------------------------
def exact_mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def bonferroni(n_tests, alpha=0.05):
    return alpha / n_tests


def rank_order(eps):
    """Models sorted by pass rate, descending. Ignores provider."""
    return sorted(eps, key=lambda m: -100.0 * sum(
        e["success"] for e in eps[m].values()) / len(eps[m]))


def leading_group(eps):
    """The models that are not separated from the best one. THE frontier cut.

    Returns them in rank order, best first; the best model is always in.

    WHY THIS EXISTS AT ALL. Three sections used to say "the top six" and mean
    three different sets of models. sec:taxonomy and clustering_v8 sliced
    ``model_order(eps)[:6]`` -- the PROVIDER-BLOCKED display order -- while
    mcnemar_v8 and capability_axes_v8 sliced ``rank_order(eps)[:6]``. On the
    nine-API-model roster those happened to be the same six models, because the
    Claude and GPT blocks led and were also the six best, so the distinction
    never showed and nobody noticed the name was doing two jobs. Adding
    gemini-3.7-flash broke the coincidence: it passes 20.3 against
    gpt-5.6-luna's 19.8, so it enters the ranked cut and luna leaves it, while
    the display-order cut is unchanged. That is a 0.5-point difference deciding
    which models the paper calls "the frontier".

    It was not cosmetic. The claim "within the top six, content-level
    differences largely disappear" held only under the display-order cut
    (content spreads 5.0/4.6/7.0); under the ranked cut the same three spreads
    are 4.9/12.9/15.2, as large as the structural spread the claim contrasts
    them against. A headline finding turned on an alphabetical-ish accident.

    THE DEFINITION. A model is in the leading group when an exact McNemar test
    against the top-ranked model does not separate it at the ALL-PAIRS
    Bonferroni threshold, alpha = 0.05 / C(n, 2) over the panel. Three
    properties make this the cut to standardize on:

    1. It is not selected on the outcome. mcnemar_v8's own module docstring
       already says the top-six family "is itself chosen after seeing the pass
       rates, so the corrected threshold is optimistic", and that the paper
       "should only lean on a conclusion that holds under the stricter one".
       The all-pairs family is the stricter one, so the frontier is now drawn
       with the same yardstick the paper already promised to lean on.
    2. It has no free parameter. "Six" was a round number; this is whatever
       the data supports.
    3. It is not a knife edge. On the API panel the boundary p-values jump by a
       factor of ~25 (gpt-5.6-luna in at p=0.00179, gemini-3.6-flash out at
       p=0.00007) across a real gap in the pass-rate distribution (19.8 -> 18.5).
       The cut lands at the same place in the ranking on the 10-model API panel
       (7 in) and the 14-model full panel (9 in) even though their thresholds
       differ by 2x, which is the robustness check worth having.

    NOT A TRANSITIVE CLIQUE. Membership is "indistinguishable from the best",
    the usual statistical-tie-for-first reading, not "mutually
    indistinguishable". The latter is not well defined as a single set -- it
    depends on the elimination order -- and would let a model outside the group
    tie a model inside it, which reads worse than it sounds.
    """
    import itertools as _it
    ranked = rank_order(eps)
    alpha = bonferroni(len(list(_it.combinations(ranked, 2))))
    best = ranked[0]
    out = [best]
    for m in ranked[1:]:
        common = sorted(set(eps[best]) & set(eps[m]))
        b = sum(1 for k in common
                if eps[best][k]["success"] and not eps[m][k]["success"])
        c = sum(1 for k in common
                if not eps[best][k]["success"] and eps[m][k]["success"])
        if exact_mcnemar(b, c) >= alpha:
            out.append(m)
    return out


def provider_blocks(eps):
    """The paper's table order, as ``[(provider, [model, ...]), ...]``.

    API models are blocked by provider so a within-family comparison is on
    adjacent rows. Within a block, and between blocks, the ordering key is
    pass rate: models descend within their block, and the blocks themselves
    are ordered by their own best model. That keeps the first row of the table
    the best API model overall, so blocking does not cost the table its
    headline.

    The open-weight group is the one exception: it is a single block pinned
    last, ahead of no one, whatever it scored. See OPEN_BLOCK for why. On a
    nine-model panel no such block exists and this is pure provider blocking.

    Blocking is presentation only -- it never changes which rounds a number is
    computed over. Note the consequence for the reader: pass rate is no longer
    monotone down the whole column, so any caption that wants the global
    ranking has to say it in words.
    """
    ranked = rank_order(eps)
    rank = {m: i for i, m in enumerate(ranked)}
    groups: dict[str, list] = {}
    for m in ranked:
        groups.setdefault(block_key(m), []).append(m)
    # The `kv[0] == OPEN_BLOCK` term is the pin: False sorts before True, so
    # every provider block keeps its pass-rate position and the open-weight
    # block drops to the bottom even when it outranks one.
    return sorted(groups.items(),
                  key=lambda kv: (kv[0] == OPEN_BLOCK,
                                  min(rank[m] for m in kv[1])))


def group_by_provider(order):
    """Re-split a `model_order` list back into its ``[(provider, [m, ...])]``.

    For callers that were handed the flat order and need the block structure
    again -- a figure legend that wants one row per provider, say -- without
    holding the episodes the order was computed from. The contiguity assert is
    the point: if the list was not produced by `model_order`, regrouping it
    would silently invent blocks that no table shows.

    Keys off `block_key`, exactly as `provider_blocks` does, so the open-weight
    group regroups as the one pinned block it was laid out as. Keying off
    PANEL_PROVIDER here instead would split it back into three vendor blocks
    and hand the caller a structure no table draws.
    """
    blocks: list = []
    for m in order:
        p = block_key(m)
        if blocks and blocks[-1][0] == p:
            blocks[-1][1].append(m)
        else:
            assert all(q != p for q, _ in blocks), (
                f"{p} appears in two runs of {order}; this is not a "
                f"provider-blocked order")
            blocks.append((p, [m]))
    return blocks


def model_order(eps):
    """The paper's table order, flattened -- provider blocks, best block first.

    Was pure pass-rate descending until the provider grouping landed. Callers
    that specifically need the ranking (a "who beat whom" statement, a legend
    that reads best to worst) must ask for `rank_order` explicitly rather than
    relying on this.
    """
    return [m for _, ms in provider_blocks(eps) for m in ms]


def and_list(items):
    """`a`, `a and b`, `a, b and c` -- a caption clause, not a debug repr.

    LIVES HERE BECAUSE TWO TABLES BUILD THE SAME SENTENCE. tab:main and
    tab:capability both footnote the text-only rows, and both derive that set
    from ASSET_MODALITY, so both grow silently when a model is added. The
    naive " and ".join reads correctly for the two marked rows the clause was
    written against and ungrammatically for the third -- which is exactly
    what shipped in one of the two after deepseek-v4-flash arrived, because
    the fix was applied where it was noticed instead of where it was shared.
    """
    items = list(items)
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


_ORDINAL = ("first", "second", "third", "fourth", "fifth", "sixth",
            "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth",
            "thirteenth", "fourteenth", "fifteenth", "sixteenth")
assert len(_ORDINAL) >= N_PANEL, (
    "blocking_note indexes _ORDINAL by table position; it must cover every "
    "row the combined panel can produce")


def blocking_note(eps):
    """The caption sentence that provider blocking obliges a table to carry.

    Generated, not typed into the caption that needs it. Blocking costs the
    reader the one guarantee an unblocked table gives for free -- that the
    pass column descends -- and the honest repair is to name the model the eye
    would otherwise misplace. That is a claim about a rank and a rate, which
    is exactly the sort of sentence that goes stale in a hand-written caption
    once a number moves.

    Only Table 1 carries it. Table 3 leads with the same pass column but says
    "Rows are in the order of Table~\\ref{tab:main}" instead: one explanation
    of the ordering is enough, and the second table would pay a caption line
    to repeat it. `blocking_ref` emits that cross-reference.

    The wording is on a diet. Measured in the real caption at \\linewidth by
    `src/measure_caption.py`, the sentence sits on one caption line (13.6pt)
    up to roughly 150 characters and spills to two beyond that -- naming the
    block the model opens costs 177 characters and so costs a whole line, for
    a fact the table already shows. The nine-model wording is 149 and fits.
    The twelve-model wording does not: it has to explain the pinned block as
    well as the provider blocking, and that is a cause the reader cannot infer
    from the table, unlike a rank the table already prints. It buys the second
    line deliberately. Any re-wording should be run past that script, which
    reports the live cost rather than guessing from a character count.
    """
    ranked = rank_order(eps)
    flat = model_order(eps)
    if flat == ranked:
        return ("Rows are blocked by provider; on these numbers that blocking "
                "happens to leave the pass column descending.")
    i = next(k for k, m in enumerate(flat) if m != ranked[k])
    m = ranked[i]
    # On the twelve-model panel the pin, not the provider blocking, is what
    # breaks monotonicity first, so saying only "blocked by provider" would
    # point the reader at the wrong cause.
    lead = ("The API rows are blocked by provider, blocks and rows both by "
            "pass rate, and the open-weight models are one block pinned last"
            if any(x in OPEN_MODEL_PROVIDER for x in flat) else
            "Rows are blocked by provider, blocks and rows both by pass rate")
    return (f"{lead}, so the pass column is not monotone: \\texttt{{{m}}} "
            f"({pass_pct(eps[m].values()):.1f}\\%) is {_ORDINAL[i]} overall.")


def blocking_ref(label="tab:main"):
    """What a second blocked table says instead of repeating `blocking_note`.

    Kept next to the note so the two can never drift into disagreeing about
    which table owns the explanation.
    """
    return f"Rows are in the order of Table~\\ref{{{label}}}."


def pass_pct(episodes):
    return 100.0 * sum(e["success"] for e in episodes) / len(episodes)


_MAX_STEPS_RE = re.compile(r"reached max_steps=(\d+)")


def step_budget(episodes):
    """The per-episode step budget, read off the harness's own stop reason.

    The budget is a run parameter, not a property of the corpus, so it is
    recovered from the runs rather than declared here: an episode stopped by
    the budget says so as `reached max_steps=<N>`. Every such row must name the
    same N, and no episode may report more steps than that -- either failure
    means the sweep mixed two configurations and nothing downstream that talks
    about "the budget" is meaningful.

    `episodes` is a {model: {task_id: episode}} mapping as returned by
    load_eval, or any iterable of episode dicts.
    """
    if isinstance(episodes, dict):
        rows = [e for v in episodes.values()
                for e in (v.values() if isinstance(v, dict) else v)]
    else:
        rows = list(episodes)
    seen, hardest = set(), 0
    for e in rows:
        m = _MAX_STEPS_RE.search(e.get("stop_reason") or "")
        if m:
            seen.add(int(m.group(1)))
        if isinstance(e.get("steps"), (int, float)):
            hardest = max(hardest, e["steps"])
    assert len(seen) == 1, (
        f"episodes report {sorted(seen) or 'no'} step budgets; a sweep that "
        f"mixes budgets cannot be summarised by one number")
    budget = seen.pop()
    assert hardest <= budget, (
        f"an episode ran {hardest} steps under a {budget}-step budget")
    return budget


# --------------------------------------------------------------------------
# self-test: reproduce already-verified numbers before any new table trusts it
# --------------------------------------------------------------------------
if __name__ == "__main__":
    specs = load_specs()
    print(f"specs: {len(specs)} tasks, 20 seeds x 30 rounds")
    cats = defaultdict(int)
    for v in specs.values():
        for c in v["cats"]:
            cats[c] += 1
    print("tasks per category (overlapping, 600 raw):",
          {c: cats[c] for c in CAT_ORDER})

    print(f"\nPRIMARY population '{PRIMARY_POPULATION}': {N_RAW} rounds, "
          f"error rows retained (the behavioural report's population)")
    hdr = (f"{'model':<{NAME_W}} {'n(h/m/l)':<14} {'pass h':>7} {'m':>6} {'l':>6} | "
           f"{'cap h':>6} {'m':>6} {'l':>6} | err | 582 high  ratio")
    print(hdr)
    print("-" * len(hdr))
    sweep = load_sweep()
    excl = load_sweep(population="excluded")
    eps_h = sweep["high"][0]
    ratios = []
    for m in model_order(eps_h):
        ns, ps, cs, es = [], [], [], []
        for lvl in LEVELS:
            e, err = sweep[lvl]
            v = list(e[m].values())
            ns.append(len(v))
            ps.append(pass_pct(v))
            cs.append(100.0 * sum(x["cap_hit"] for x in v) / len(v))
            es.append(err.get(m, 0))
        px = pass_pct(list(excl["high"][0][m].values()))
        ratios.append(ps[0] / px)
        print(f"{m:<{NAME_W}} {'/'.join(map(str, ns)):<14} {ps[0]:>7.2f} {ps[1]:>6.2f} {ps[2]:>6.2f} | "
              f"{cs[0]:>6.1f} {cs[1]:>6.1f} {cs[2]:>6.1f} | {'/'.join(map(str, es))} | "
              f"{px:>8.2f} {ps[0]/px:>6.4f}")
    print(f"  -> full/excluded ratio is {N_EXCLUDED}/{N_RAW} = "
          f"{N_EXCLUDED/N_RAW:.4f} for every model "
          f"(observed {min(ratios):.4f}-{max(ratios):.4f}; 3.1-pro differs "
          f"only because 5 provider-error rows are retained in the full view)")

    dp = dn = 0
    for lvl in LEVELS:
        for eps in sweep[lvl][0].values():
            for e in eps.values():
                if e["dead"]:
                    dn += 1
                    dp += e["success"]
    print(f"  -> the {N_DEAD_V8} provably-unwinnable rounds pass {dp}/{dn} "
          f"round-runs across all models and efforts, which is why keeping "
          f"them rescales without reordering")

    # ---- replicate the factory check_split before trusting our own view ----
    # The shipped stats.json was built on 600 rounds with no dead-round
    # exclusion. Reproducing it bit-for-bit proves our predicates match
    # build_behavior_report.py; only then is the 582/high recomputation
    # believable. The effort is the artifact's own -- see frozen_effort.
    feff = frozen_effort()
    print(f"\nfactory check_split replication (600 rounds, {feff}, "
          f"no exclusion)")
    stats = json.load(open(STATS_JSON))
    raw_m, _ = load_eval(feff, population="full", keep_checks=True)
    bad = 0
    for label, skey in STATS_KEY.items():
        want = stats[skey]["check_split"]
        got = defaultdict(lambda: defaultdict(int))
        for e in raw_m[label].values():
            for c in (e["checks"] or []):
                st = check_status(c)
                for b in (check_bucket(c), check_subbucket(c)):
                    if b is None:
                        continue
                    b = f"field_contains_{b}" if b in ("asset_gate", "content") else b
                    got[b]["total"] += 1
                    got[b][{"infra": "infra_failures",
                            "not_invoked": "not_invoked",
                            "graded_pass": "passed"}.get(st, "_fail")] += 1
        for b, w in want.items():
            key = b.replace("field_contains_asset", "field_contains_asset_gate")
            g = got[key if key in got else b]
            for f in ("total", "passed", "not_invoked", "infra_failures"):
                if g[f] != w[f]:
                    print(f"  MISMATCH {label} {b}.{f}: ours {g[f]} vs factory {w[f]}")
                    bad += 1
    print(f"  -> {f'EXACT MATCH on all {len(STATS_KEY)} models x 5 buckets' if not bad else str(bad) + ' MISMATCHES'}")

    _ex = MODEL_LABEL["opus_5"]
    print(f"\ncheck mix, {_ex} @ high, n={N_RAW} (paper view)")
    full_h, _ = load_eval("high", keep_checks=True)
    tot = defaultdict(int)
    role = defaultdict(int)
    for e in full_h[_ex].values():
        for c in (e["checks"] or []):
            tot[check_type(c)] += 1
            role["required" if c.get("role") == "required" else "diagnostic"] += 1
    allc = sum(tot.values())
    for k, v in sorted(tot.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<24} {v:>5}  {100.0*v/allc:5.1f}%")
    print(f"  {'all checks':<24} {allc:>5}   "
          f"required {role['required']}  diagnostic {role['diagnostic']}")

    # leading_group is the frontier cut every downstream section now shares, so
    # it is exercised on BOTH panels here rather than trusted from one caller.
    # The assertions are the three properties its docstring claims, restated as
    # code: the cut is a prefix of the ranking (not a scattered subset), the
    # boundary is a real gap and not a threshold graze, and the two panels --
    # whose thresholds differ by about 2x -- agree on where the line falls.
    print("\nleading_group (frontier cut), both panels")
    _api_eps = load_eval("high", keep_checks=False)[0]
    _all_eps = load_panel("high", keep_checks=False)[0]
    _grp = {}
    for _name, _e in (("API  (load_eval) ", _api_eps),
                      ("full (load_panel)", _all_eps)):
        _ranked, _g = rank_order(_e), leading_group(_e)
        _pp = {m: 100.0 * sum(x["success"] for x in _e[m].values()) / len(_e[m])
               for m in _e}
        assert _g == _ranked[:len(_g)], (
            f"{_name}: leading group is not a prefix of the ranking; "
            f"a lower-ranked model is in while a higher-ranked one is out")
        _grp[_name] = _g
        print(f"  {_name}  n={len(_e):>2}  leading={len(_g)}  "
              f"last in: {_g[-1]} ({_pp[_g[-1]]:.1f})  |  "
              f"first out: {_ranked[len(_g)]} ({_pp[_ranked[len(_g)]]:.1f})")
    # The two panels cannot have the SAME last-in model -- the full panel
    # interleaves open-weight models the API panel has never heard of, and
    # deepseek-v4-pro ties gpt-5.6-luna at 19.8. The invariant that actually
    # matters is stronger and roster-proof: widening the panel from 10 to 14
    # must not change the verdict on any model both panels contain, even though
    # doing so halves the threshold (0.05/45 -> 0.05/91).
    _restricted = [m for m in _grp["full (load_panel)"] if m in _api_eps]
    assert _restricted == _grp["API  (load_eval) "], (
        f"widening the panel changed the frontier verdict for an API model:\n"
        f"  API panel      -> {_grp['API  (load_eval) ']}\n"
        f"  full, API-only -> {_restricted}")
    print(f"  -> the full panel's cut, restricted to the {len(_api_eps)} API "
          f"models, reproduces the API panel's cut exactly "
          f"({len(_restricted)} models)")
