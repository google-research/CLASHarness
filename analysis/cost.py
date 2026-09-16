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

"""T2 -- what a round costs: API dollars and tokens, at the paper's high effort.

WHY THIS SCRIPT EXISTS RATHER THAN A COPY OF THE REPORT'S NUMBERS.
The behavioural report priced the corpus at MEDIUM effort, and its output is
frozen in src/report/cross-model-behavioral-comparison.stats.json, which the
appendix renders as Table~\\ref{tab:rep-02}. The paper reports HIGH effort
(corpus.PRIMARY_EFFORT), and cost is the one axis where the two differ most --
reasoning tokens are the thing effort turns up. Quoting the medium table beside
a high-effort results table would put two efforts in one narrative.

So the pricing engine is ported here and re-run at high. The licence to do that
is the self-check: main() ALWAYS recomputes medium first and asserts every
field of every model's cost block matches the frozen artifact exactly, integer
for integer and cent for cent, before it will print anything about high. If the
port drifted from the factory the script dies rather than shipping a number.
That contract is the same one check_split_v8.py operates under (see its header)
and it is the reason the frozen json MUST NOT be regenerated or overwritten:
it is the reference, and a reference you rewrite when it disagrees with you is
not a reference. High-effort output goes to its own file.

WHAT IS BEING BILLED.
Token usage lives in the interaction sidecar logs (log/*.interactions.jsonl),
one "attempt" record per model call -- NOT in the trajectory jsonl. Cost sums
over ALL attempts, so retries are billed, which is what an operator pays. All
twelve models record response.usage on 100% of attempts at every effort they
were run at (re-verified here: the script asserts it), so nothing is estimated.

FOUR provider schemas, and the differences are not cosmetic:
  gemini            total = prompt + candidates + thoughts. `thoughts`
                    (reasoning) bills at the OUTPUT rate. No cache-read field
                    is exposed, so no discount is applied.
  bedrock-gpt       total = input + output. reasoning_tokens is a SUBSET of
                    output; cached_tokens and cache_write_tokens are SUBSETS of
                    input, so the remainder bills at the full input rate.
  vertex-anthropic  input_tokens EXCLUDES the cache buckets -- they are
                    ADDITIVE, not subsets -- so total input is input +
                    cache_read + cache_creation and nothing is subtracted.
                    Extended thinking bills INSIDE output_tokens, so costing
                    output whole is correct.
  openai            the self-hosted open-weight endpoint, and the ONLY schema
                    here that is a Chat Completions body: the counts are named
                    prompt_tokens / completion_tokens, NOT input_tokens /
                    output_tokens. Reading the Bedrock names off it yields 0
                    and 0 and prices the whole group at zero -- see the guards
                    in load_usage, which exist because that is not hypothetical.
To make the columns comparable, "visible output" is output minus reasoning for
Bedrock/Anthropic and `candidates` for Gemini, so for those nine
total = input + visible_output + reasoning.

REASONING IS NOT SEPARABLE FOR THE OPEN-WEIGHT GROUP, AND 0 IS THE WRONG WAY TO
SAY SO. The Chat Completions body carries no reasoning count: thinking is real
(response.thinking_text is populated on these calls) but it is folded inside
completion_tokens with no field that splits it out. So those rows carry
reason_known=False and their reasoning fields are None, which renders as `--`
rather than as a zero a reader would take for "this model did not think". The
consequence for the neighbouring cell is that `visible output` for that group
is the WHOLE output, thinking included, so it is not the same quantity as the
other nine models' visible output and the caption has to say so. Totals are
unaffected: input + output is the same number either way.

CROSS-CHECKED AGAINST response.usage_normalized. The harness writes a second,
provider-independent view of every usage block (input / output_visible /
output_reasoning / output_billed) alongside the raw one. It is present on
100.00% of all 182,241 attempts and this script asserts, per call, that the
schema it just parsed agrees with it. That is what turns "I read the right
keys" from a claim into a check: with the wrong key names the derived input is
0 while usage_normalized.input is not, and the run dies instead of billing a
model at zero. It is NOT used as the source of truth, because it collapses the
cache buckets that Anthropic and Bedrock pricing need.

Prints a LaTeX body as well as the console view; both come from one dict so
they cannot drift.
"""
import glob
import json
import os
import sys
from decimal import ROUND_HALF_UP, Decimal

import corpus
# One rank correlation in the repository, not two. check_split_v8.spearman
# already handles ties with average ranks and is what Sec 5 quotes; a second
# implementation here would be a second defensible answer. (Consequence: editing
# check_split_v8.py changes this script's output without changing its mtime,
# which is what results_facts.py keys its cache on -- use --fresh.)
from check_split_v8 import spearman

# Per-model API list pricing, USD per 1M tokens. Ported from the harness's
# build_behavior_report.py, and the provenance travels with it because a rate
# with no date is a rate nobody can re-check:
#   All nine re-verified 2026-08-05 against first-party pricing pages -- OpenAI
#   developers.openai.com/api/docs/pricing (Standard tier), Gemini
#   cloud.google.com/vertex-ai/generative-ai/pricing (Google models / Gemini 3 /
#   Standard), Anthropic anthropic.com/pricing. claude-opus-5 verified
#   2026-07-25 against anthropic.com/news/claude-opus-5 ("$5 per million input
#   tokens and $25 per million output tokens, the same as Opus 4.8") and still
#   $5/$25 on 2026-08-05.
#
# These rates are the POST-CUT ones. An earlier draft of this table carried
# gpt-5.6-terra at $2.50/$15.00 and gpt-5.6-luna at $1.00/$6.00 -- the PRE-CUT
# rates. AWS cut Bedrock prices for the
# GPT-5.6 family effective 2026-07-30 (announcement: "Amazon Bedrock announces up
# to 80% lower prices for OpenAI GPT-5.6 models" -- luna -80%, terra -20%, sol
# unchanged), a uniform scale on every tier of each model. These runs executed
# 2026-08-02 22:41 to 2026-08-03 08:49, three days after the effective date, so
# they were BILLED at the reduced rates; the pre-cut table overstated luna's run
# total by 5x ($341.37 against $68.27) and terra's by 1.25x ($795.58 against
# $636.47). The frozen artifact carries the corrected figures, so the medium
# self-check below catches exactly this mistake -- it is how it was caught.
#
# Gemini 3.5-flash is priced at its GLOBAL endpoint rate ($1.50/$9.00). The Vertex
# page lists $1.65/$9.90 for non-global endpoints since 2026-07-01, which would
# apply had the run used a regional endpoint; the endpoint is not recorded in
# run_config, so this is an assumption and not a reading.
#   in/out          standard (short-context) input / output rate
#   cached          cache-read input rate (omit -> billed as normal input)
#   cache_write     cache-write input rate (OpenAI 1.25x input)
#   long_threshold  input-token boundary above which the long column applies
#   long_*          long-context rates
#
# LONG-CONTEXT TIERS DO TRIGGER on this corpus, so _rate_tier is live code and
# not decoration. The three Bedrock GPT models each push a handful of calls past
# their 128K threshold (5/5/4 calls at high for sol/luna/terra). Their maxima sit
# within ~0.6% of the boundary, which looks like the harness context clamp
# landing on the same number rather than a genuine long-context workload, and the
# dollar effect is real but negligible (~+0.1% of the run total). Gemini 3.1-pro
# never approaches its 200K threshold. gemini-3.5-flash and the Claude models
# reach far larger single calls but have no long tier defined, so they bill flat.
#
# CAVEAT: the OpenAI models ran on Amazon Bedrock (provider=bedrock-gpt). OpenAI
# bills Bedrock via AWS and those rates "may differ from direct OpenAI pricing",
# so the direct list rates below are the best available proxy, not an invoice.
# claude-sonnet-5 is at its introductory price (standard is $3/$15); the run
# predates the 2026-08-31 cutover, so the intro rate is the one that was charged.
PRICING = {
    "flash":   {"in": 1.50, "out": 9.00,  "cached": 0.15},
    "flash36": {"in": 1.50, "out": 7.50,  "cached": 0.15},
    # gemini-3.7-flash ran inside its INTRODUCTORY pricing window (through
    # 2026-12-31); this is the rate that was charged for this run. It is not
    # the rate the model settles at, and the difference reverses a comparison:
    # priced on the standard $1.50/$7.50 card that 3.7-flash itself moves to,
    # this run costs MORE per task than gemini-3.6-flash, not less, because it
    # burns more input tokens. Any "newer model is cheaper" reading of this row
    # is an artefact of the promotion, not an efficiency result.
    "flash37": {"in": 0.75, "out": 3.75,  "cached": 0.075},
    "pro":     {"in": 2.00, "out": 12.00,
                "long_threshold": 200_000, "long_in": 4.00, "long_out": 18.00},
    "sol":     {"in": 5.00, "out": 30.00, "cached": 0.50, "cache_write": 6.25,
                "long_threshold": 128_000, "long_in": 10.00, "long_out": 45.00,
                "long_cached": 1.00, "long_cache_write": 12.50},
    "terra":   {"in": 2.00, "out": 12.00, "cached": 0.20, "cache_write": 2.50,
                "long_threshold": 128_000, "long_in": 4.00, "long_out": 18.00,
                "long_cached": 0.40, "long_cache_write": 5.00},
    "luna":    {"in": 0.20, "out": 1.20,  "cached": 0.02, "cache_write": 0.25,
                "long_threshold": 128_000, "long_in": 0.40, "long_out": 1.80,
                "long_cached": 0.04, "long_cache_write": 0.50},
    "opus":    {"in": 5.00, "out": 25.00},   # claude-opus-4.8
    "opus5":   {"in": 5.00, "out": 25.00},   # claude-opus-5, same rate per the launch post
    "sonnet":  {"in": 2.00, "out": 10.00},   # claude-sonnet-5, intro price
}

# THE OPEN-WEIGHT GROUP WAS NOT BILLED BY ANYBODY. Those three checkpoints ran
# on a self-hosted OpenAI-compatible endpoint, so unlike every rate above --
# which prices a call some provider actually invoiced, even where the invoice
# and the list rate may differ -- these rates price a call that was never sold.
# They are the vendors' own first-party API list rates, used as a STATED
# COMPARABILITY PROXY: "what this run would have cost at the vendor's counter".
# The table and its caption have to say that in those terms. Do not let the
# column be read as spend.
#
# Verified against first-party pricing pages, 2026-08-12:
#   deepseek-v4-pro  api-docs.deepseek.com/quick_start/pricing
#                    $0.435 in (cache miss) / $0.003625 in (cache hit) / $0.87 out
#   kimi-k3          platform.kimi.ai/docs/pricing/chat-k3
#                    $3.00 in (cache miss) / $0.30 in (cache hit) / $15.00 out
#   GLM-5.2          docs.z.ai/guides/overview/pricing
#                    $1.40 in / $0.26 cached in / $4.40 out
#
# The `cached` rates are recorded for completeness but never fire: the endpoint
# reports `prompt_tokens_details: null` on every call, so no call has a cached
# subset and all input bills at the cache-miss rate. `_openai_call_cost` zeroes
# a cached count whose rate is absent, and here the count is what is absent, so
# the guard in load_usage that a priced call consumed nonzero input is what
# actually holds this together.
#
# Z.AI PRICES `GLM-5.2`; THE CHECKPOINT RUN WAS `zai-org/GLM-5.2-FP8`. The
# vendor does not list a separate rate for the FP8 weights, so the unquantised
# list rate is the only published number and is used as-is. That is one more
# reason the column is a proxy rather than a cost.
PRICING_OPEN = {
    "v4pro":  {"in": 0.435, "out": 0.87,  "cached": 0.003625},
    "kimik3": {"in": 3.00,  "out": 15.00, "cached": 0.30},
    "glm52":  {"in": 1.40,  "out": 4.40,  "cached": 0.26},
}
PRICING.update(PRICING_OPEN)

# On-disk model directory -> the pricing key the frozen artifact is keyed by.
# corpus.MODEL_LABEL maps the same directories to display names; both are needed
# because the artifact and the paper disagree about what a model is called.
#
# A value of None means THE MODEL IS KNOWN AND DELIBERATELY UNPRICED, which is
# not the same as missing from this map. Missing is an error -- it is how a new
# directory would bill at zero. None is a recorded fact: the vendor has not
# published a rate for that checkpoint, so there is no number to charge and the
# dollar columns for that row are absent rather than cheap. The distinction has
# to live here, because `raw_model in PRICE_KEY` is the guard in load_usage and
# a None that were spelled as an omission would trip it.
PRICE_KEY = {
    "3_5_flash": "flash",
    "3_6_flash": "flash36",
    "3_7_flash": "flash37",
    "3_1_pro":   "pro",
    "opus_4_8":  "opus",
    "opus_5":    "opus5",
    "sonnet_5":  "sonnet",
    "5_6_sol":   "sol",
    "5_6_luna":  "luna",
    "5_6_terra": "terra",
    # Open-weight group. These three have NO entry in the frozen artifact --
    # they did not exist when it was built and were never run at medium -- so
    # they are deliberately absent from FROZEN_KEYS below.
    "v4_pro":    "v4pro",
    "kimi_k3":   "kimik3",
    "glm_5_2":   "glm52",
    # DeepSeek publishes no rate card for the -flash checkpoint. The other three
    # open-weight rows carry a stated list-rate proxy (PRICING_OPEN); this one
    # has nothing to proxy, so its dollar columns render as absent. Inventing a
    # rate -- the v4-pro card, a scaled-down guess, anything -- would put a
    # fabricated number in the one column readers quote, so None it is. Its
    # TOKEN columns are unaffected: those are measured, not priced.
    "dsv4_flash_v8_high_ocr": None,
}
assert set(PRICE_KEY) == set(corpus.PANEL_LABEL), (
    "PRICE_KEY and corpus.PANEL_LABEL disagree about the panel: "
    f"{sorted(set(PRICE_KEY) ^ set(corpus.PANEL_LABEL))}")
assert set(PRICE_KEY.values()) - {None} <= set(PRICING), (
    f"unpriced keys {sorted(set(PRICE_KEY.values()) - {None} - set(PRICING))}")

# The subset the medium self-check runs over: exactly the models the frozen
# artifact contains. Keyed off corpus.REPORT_DIRS -- the report's own roster --
# and NOT off corpus.MODEL_LABEL, which is the API panel and is free to grow.
# Those two were the same nine directories once; binding to the wrong one meant
# that adding an API model turned this dict into a KeyError against an artifact
# that will never hold that key, which is the exact failure the comment here
# used to claim it prevented.
FROZEN_KEYS = {raw: PRICE_KEY[raw] for raw in corpus.REPORT_DIRS}
assert None not in FROZEN_KEYS.values(), (
    "a model in the frozen artifact has no price key; the medium self-check "
    "compares dollar fields and cannot run against an unpriced row")

HIGH_JSON = os.path.join(corpus.PAPER, "cost_high.json")

# What a dollar cell renders as when the vendor publishes no rate. One constant
# so the table body, the console dump and the caption facts cannot drift apart,
# and so `grep NA_CELL` finds every place the reader could meet it.
NA_CELL = r"N/A"


def _rate_tier(pr, in_tokens):
    """(input, output, cached, cache_write) rates for a call, by its input size."""
    if pr.get("long_threshold") and in_tokens > pr["long_threshold"]:
        return (pr["long_in"], pr["long_out"],
                pr.get("long_cached"), pr.get("long_cache_write"))
    return pr["in"], pr["out"], pr.get("cached"), pr.get("cache_write")


def _openai_call_cost(it, ot, cached, cwrite, pr):
    """USD for one OpenAI/Bedrock call. cached & cwrite are SUBSETS of `it` billed
    at their own rates; the remainder bills at the full input rate. `ot` is the
    full output, reasoning already inside it, at the output rate."""
    in_r, out_r, cached_r, cw_r = _rate_tier(pr, it)
    if cached_r is None:
        cached = 0
    if cw_r is None:
        cwrite = 0
    full = it - cached - cwrite
    cost = full * in_r + cached * (cached_r or 0) + cwrite * (cw_r or 0) + ot * out_r
    return cost / 1_000_000.0


def _gemini_call_cost(prompt, cand, thoughts, pr):
    """USD for one Gemini call: prompt at the input rate; candidates + thoughts
    at the OUTPUT rate. No cache-read field is recorded, so no discount."""
    in_r, out_r, _, _ = _rate_tier(pr, prompt)
    return (prompt * in_r + (cand + thoughts) * out_r) / 1_000_000.0


def _anthropic_call_cost(it, ot, cread, cwrite, pr):
    """USD for one Anthropic call. `it` EXCLUDES the cache counts -- they are
    separate ADDITIVE buckets -- so total input is it + cread + cwrite and
    nothing is subtracted. `ot` is billed whole: extended thinking bills inside
    output_tokens and there is no separate reasoning rate."""
    in_r, out_r, cached_r, cw_r = _rate_tier(pr, it + cread + cwrite)
    cost = (it * in_r
            + cread * (cached_r if cached_r is not None else in_r)
            + cwrite * (cw_r if cw_r is not None else in_r)
            + ot * out_r)
    return cost / 1_000_000.0


def load_usage(effort, root=corpus.ROOT_V8, run=corpus.RUN_V8,
               panel=None, tree=None):
    """model directory -> {task_id: token/cost aggregate}, summed over ALL attempts.

    Asserts that every attempt record carries response.usage. The factory kept a
    "no usage recorded" fallback for robustness; here its absence is the claim
    being made, so a missing usage block is a hard failure rather than a silent
    None that would quietly drop a model out of the table.

    `panel` and `tree` mean exactly what they mean in corpus.load_eval and
    default the same way, to the nine API models under aligned_setting_<effort>,
    so every pre-existing caller keeps its behaviour. They are what let the
    open-weight group be read from its own tree without the two rosters mixing.
    """
    panel = corpus.MODEL_LABEL if panel is None else panel
    tree = f"aligned_setting_{effort}" if tree is None else tree
    out = {}
    # Globbed across every run subdirectory and then filtered, rather than
    # globbed at a fixed `{run}` -- the same shape corpus.load_eval uses, and
    # for the same reason: RUN_OVERRIDE selects a different run for a single
    # cell, and a fixed glob cannot see it.
    #
    # THIS WAS A BUG, NOT A STYLE CHOICE. Reading at a fixed `{run}` billed
    # gemini-3.7-flash's high cell on r01 while corpus.load_eval scored it on
    # r02, so tab:cost reported one run's tokens and dollars against the other
    # run's outcome, in the same row. The failure is invisible from either half
    # alone -- both are complete, well-formed runs of 600 tasks -- which is why
    # the run choice has to live in one function that every reader calls rather
    # than being re-expressed as a glob argument per module. Two other scanners
    # had the identical defect (tools/failure_taxonomy.py,
    # tools/scan_reasoning_tokens.py); if you add a third reader of this tree,
    # route it through corpus.run_for.
    for logdir in sorted(glob.glob(f"{root}/{tree}/*/*/*/log")):
        raw_model = logdir.split("/")[-3]
        rel_dir = "/".join(logdir.split("/")[-5:-2])
        if logdir.split("/")[-2] != corpus.run_for(rel_dir, run):
            continue
        if rel_dir in corpus.ALIAS_DIRS:
            # A second path to a run already billed from OPEN_TREE, not a second
            # run. corpus.load_eval skips the same directory for the same reason
            # and re-proves the bytes rather than trusting the registration;
            # billing it here would double-count a model the outcome side has
            # already de-duplicated, which is the one way the two halves of
            # build() could disagree silently.
            corpus.assert_alias_is_duplicate(root, rel_dir, run)
            continue
        if raw_model in corpus.EXCLUDED_MODELS:
            continue
        assert raw_model in panel, (
            f"unmapped model directory {raw_model!r} under {root}/{tree}; add "
            f"it to the panel rather than letting the raw name into a table")
        assert raw_model in PRICE_KEY, (
            f"unpriced model directory {raw_model!r} under {root}; add it to "
            f"PRICING and PRICE_KEY rather than letting it bill at zero")
        # pr is None for a model PRICE_KEY deliberately maps to None. Every cost
        # expression below is then skipped and `priced` rides along in the
        # per-task aggregate so summarize can emit None instead of the 0.0 the
        # accumulator would otherwise hold. 0.0 is a price; this is the absence
        # of one, and the two must not render the same.
        price_key = PRICE_KEY[raw_model]
        pr = PRICING[price_key] if price_key is not None else None
        by_task, seen, missing = {}, 0, 0
        n_img = n_ocr = n_open = 0
        for fp in sorted(glob.glob(os.path.join(logdir, "*.interactions.jsonl"))):
            for line in open(fp):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("record_type") != "attempt":
                    continue
                seen += 1
                inp = r.get("input") or {}
                refs = inp.get("image_refs")
                n_img += len(refs) if isinstance(refs, list) else 0
                text = inp.get("last_user_text") or ""
                n_ocr += text.count(corpus.OCR_PREAMBLE)
                n_open += text.count("Asset opened:")
                resp = r.get("response") or {}
                u = resp.get("usage") or {}
                if not u:
                    missing += 1
                    continue
                agg = by_task.setdefault(r.get("task_id"), {
                    "in": 0, "vis_out": 0, "reason": 0, "cached": 0,
                    "cache_write": 0, "calls": 0, "cost": 0.0,
                    "reason_known": True, "priced": pr is not None})
                provider = r.get("provider") or ""
                # Each branch sets (in_tok, vis, reas, cost) and those four are
                # the only things that leave it, so the cross-check below can be
                # written once instead of four times.
                if provider == "gemini":
                    p = u.get("prompt") or 0
                    c = u.get("candidates") or 0
                    th = u.get("thoughts") or 0
                    in_tok, vis, reas, cr, cw = p, c, th, 0, 0
                    cost = 0.0 if pr is None else _gemini_call_cost(p, c, th, pr)
                    reas_known = True
                elif "anthropic" in provider:
                    it = u.get("input_tokens") or 0
                    cr = u.get("cache_read_input_tokens") or 0
                    cw = u.get("cache_creation_input_tokens") or 0
                    ot = u.get("output_tokens") or 0
                    tt = (u.get("output_tokens_details") or {}).get("thinking_tokens") or 0
                    in_tok, vis, reas = it + cr + cw, ot - tt, tt
                    cost = (0.0 if pr is None
                            else _anthropic_call_cost(it, ot, cr, cw, pr))
                    reas_known = True
                elif provider == "bedrock-gpt":
                    it = u.get("input_tokens") or 0
                    ot = u.get("output_tokens") or 0
                    idet = u.get("input_tokens_details") or {}
                    odet = u.get("output_tokens_details") or {}
                    cr = idet.get("cached_tokens") or 0
                    cw = idet.get("cache_write_tokens") or 0
                    rt = odet.get("reasoning_tokens") or 0
                    in_tok, vis, reas = it, ot - rt, rt
                    cost = (0.0 if pr is None
                            else _openai_call_cost(it, ot, cr, cw, pr))
                    reas_known = True
                elif provider == "openai":
                    # Self-hosted open-weight endpoint: a Chat Completions body.
                    # DIFFERENT KEY NAMES from bedrock-gpt above, and no
                    # reasoning split -- thinking is inside completion_tokens.
                    # prompt_tokens_details is null on every one of these calls,
                    # so there is no cached subset and all input bills at the
                    # cache-miss rate; _openai_call_cost is still the right
                    # arithmetic with the cache arguments zeroed.
                    it = u.get("prompt_tokens") or 0
                    ot = u.get("completion_tokens") or 0
                    idet = u.get("prompt_tokens_details") or {}
                    cr = idet.get("cached_tokens") or 0
                    in_tok, vis, reas, cw = it, ot, 0, 0
                    cost = (0.0 if pr is None
                            else _openai_call_cost(it, ot, cr, 0, pr))
                    reas_known = False
                else:
                    raise AssertionError(
                        f"{raw_model} at {effort}: unknown provider "
                        f"{provider!r}. Add a branch with that provider's usage "
                        f"schema; do NOT fall through to another provider's key "
                        f"names, which is how a model gets billed at zero.")
                # THE GUARD THAT MAKES THE ABOVE A READING RATHER THAN A GUESS.
                # response.usage_normalized is the harness's own provider-
                # independent view of the same call. It is present on 100% of
                # attempts, so its absence is a failure, not a skip. Parsing the
                # wrong key names yields in_tok == 0 against a nonzero
                # normalized input and dies here.
                un = resp.get("usage_normalized")
                assert un, (
                    f"{raw_model} at {effort}: attempt carries response.usage "
                    f"but no response.usage_normalized, so the parse below "
                    f"cannot be cross-checked")
                assert un.get("input") == in_tok, (
                    f"{raw_model} at {effort}: parsed {in_tok} input tokens from "
                    f"the {provider!r} schema but usage_normalized says "
                    f"{un.get('input')!r} -- the key names are wrong")
                assert un.get("output_billed") == vis + reas, (
                    f"{raw_model} at {effort}: parsed {vis}+{reas} billed output "
                    f"from the {provider!r} schema but usage_normalized says "
                    f"{un.get('output_billed')!r}")
                assert in_tok > 0, (
                    f"{raw_model} at {effort}: a call consumed 0 input tokens, "
                    f"which no real call does; it would bill at zero and would "
                    f"understate the token columns even where nothing is billed")
                agg["in"] += in_tok
                agg["vis_out"] += vis
                agg["reason"] += reas
                agg["cached"] += cr
                agg["cache_write"] += cw
                agg["cost"] += cost
                agg["reason_known"] &= reas_known
                agg["calls"] += 1
        assert missing == 0, (
            f"{raw_model} at {effort}: {missing} of {seen} attempts carry no "
            f"response.usage -- the paper's claim of 100% usage coverage is "
            f"stale, and those calls would bill at zero")
        # THE MODALITY CLAIM, CHECKED WHERE THE DATA ALREADY IS. The tables mark
        # two rows as OCR-fed and the caption makes a claim about it; this is the
        # only place in the package that reads all 182,241 interaction records,
        # so it is the cheapest honest place to prove the mark. Both directions
        # are checked, because a model silently LOSING its images is the failure
        # that would flatter every other row in the table.
        mode = corpus.ASSET_MODALITY.get(panel[raw_model], "image")
        if mode == "ocr":
            assert n_img == 0 and n_open > 0 and n_ocr == n_open, (
                f"{panel[raw_model]} is declared OCR-fed in "
                f"corpus.ASSET_MODALITY but at {effort} it saw {n_img} image "
                f"attachments and {n_ocr} of {n_open} asset opens came through "
                f"OCR -- the modality footnote would be wrong")
        else:
            assert n_ocr == 0 and n_img > 0, (
                f"{panel[raw_model]} is treated as image-fed but at {effort} it "
                f"saw {n_img} image attachments and {n_ocr} OCR asset opens; if "
                f"its assets stopped being delivered as images, its row is no "
                f"longer comparable to the rest of the table")
        out[raw_model] = by_task
    return out


def summarize(by_task, passed):
    """Per-task aggregates -> per-model means and run totals.

    Field-for-field the factory's summarize_cost, including the deliberate
    oddity in `total_per_task`: it is the sum of the ROUNDED components, not the
    rounded sum, so a rendered row of input + output + reasoning actually adds
    up to the total printed beside it.
    """
    tasks = list(by_task.values())
    n = len(tasks)

    def _sum(f):
        return sum(t[f] for t in tasks)

    total_in, total_vo, total_re = _sum("in"), _sum("vis_out"), _sum("reason")
    total_cost = _sum("cost")
    # Whether this model's provider separates reasoning from visible output at
    # all. False only for the self-hosted open-weight endpoint. The reasoning
    # fields then go to None, NOT to the 0 the accumulator holds: 0 is a
    # measurement and this is the absence of one. Everything that does not
    # depend on the split -- input, totals, cost -- is unaffected, because the
    # thinking tokens are inside vis_out and are billed there.
    known = all(t["reason_known"] for t in tasks)
    # Whether a published rate exists for this model at all (PRICE_KEY -> None).
    # Same discipline as `reason_known` one line up, applied to money instead of
    # to the reasoning split: where there is no rate, every dollar field goes to
    # None and not to the 0.0 the accumulator carries. It is per-model, so it is
    # constant across tasks; the all() is a consistency check as much as a read.
    priced = all(t["priced"] for t in tasks)
    assert priced == any(t["priced"] for t in tasks), (
        "some tasks of one model are priced and others are not, which cannot "
        "happen -- PRICE_KEY is per model")
    return {
        "available": True,
        "n_tasks": n,
        "reason_known": known,
        "priced": priced,
        "in_per_task": round(total_in / n),
        "vis_out_per_task": round(total_vo / n),
        "reason_per_task": round(total_re / n) if known else None,
        "total_per_task": (round(total_in / n) + round(total_vo / n)
                           + round(total_re / n)),
        "cached_per_task": round(_sum("cached") / n),
        "cache_write_per_task": round(_sum("cache_write") / n),
        "calls_per_task": round(_sum("calls") / n, 1),
        "cost_per_task": round(total_cost / n, 4) if priced else None,
        "in_total": total_in,
        "vis_out_total": total_vo,
        "reason_total": total_re if known else None,
        "tok_total": total_in + total_vo + total_re,
        "calls_total": _sum("calls"),
        "cost_total": round(total_cost, 2) if priced else None,
        "cost_per_solved": (round(total_cost / passed, 4)
                            if priced and passed else None),
    }


# Fields of the frozen artifact this port is responsible for. imgs_step0 and
# tok_per_image come from the factory's step-0 image regression, which is a
# different measurement and is not re-derived here.
CHECKED = ("n_tasks", "in_per_task", "vis_out_per_task", "reason_per_task",
           "total_per_task", "cached_per_task", "cache_write_per_task",
           "calls_per_task", "cost_per_task", "in_total", "vis_out_total",
           "reason_total", "tok_total", "calls_total", "cost_total",
           "cost_per_solved")


def verify_frozen():
    """Reproduce the frozen artifact exactly, or die.

    This is the licence to report the table below: it demonstrates that the
    port is the same engine that produced the appendix's cost table, so a
    difference between the two tables is a difference in inputs and not a
    difference in accounting.

    THE EFFORT IS READ OFF THE ARTIFACT, NOT ASSUMED. This was `verify_medium`
    and hard-asserted `primary_effort == "medium"`, which was true only while
    the frozen report was built at medium. Rebuilding the report at high effort
    turned the licence into an abort. The check does not care which effort the
    artifact holds -- it cares that the same effort is fed to load_usage -- so
    the effort is now a variable and the assertion is gone. Note the check got
    STRONGER, not weaker: when the artifact and the reported table sit at the
    same effort, this reproduces the very numbers the paper prints.
    """
    ref = json.load(open(corpus.STATS_JSON))
    frozen_eff = ref["primary_effort"]
    assert frozen_eff in corpus.LEVELS, (
        f"frozen artifact reports an unknown effort {frozen_eff!r}")
    usage = load_usage(frozen_eff)
    # SUBSET, NOT EQUALITY. The artifact's roster is frozen; the medium tree is
    # not, and a model added to the panel after the artifact was built has no
    # key to compare against. What must hold is that every model the artifact
    # DOES cover is present to be checked -- equality would turn each new panel
    # member into a spurious failure here, and the usual reaction to that is to
    # weaken the check rather than to notice it was asking the wrong question.
    missing = set(FROZEN_KEYS) - set(usage)
    assert not missing, (
        f"the frozen artifact covers {sorted(missing)} but the {frozen_eff} "
        f"tree does not hold {'it' if len(missing) == 1 else 'them'}; the "
        f"self-check would silently skip a model it is supposed to reproduce")
    bad = []
    for raw_model, key in sorted(FROZEN_KEYS.items()):
        want = ref[key]["cost"]
        got = summarize(usage[raw_model], ref[key]["passed"])
        for f in CHECKED:
            if got[f] != want[f]:
                bad.append(f"{key}.{f}: recomputed {got[f]!r} != frozen {want[f]!r}")
    assert not bad, (
        "the ported pricing engine no longer reproduces "
        f"{os.path.basename(corpus.STATS_JSON)}:\n  " + "\n  ".join(bad) +
        "\nFIX THE PORT. Do NOT regenerate the artifact: it is the reference.")
    return frozen_eff, len(FROZEN_KEYS) * len(CHECKED)


def load_usage_open(root=corpus.ROOT_V8, run=corpus.RUN_V8):
    """Token usage for the open-weight models, from their own trees.

    Mirrors corpus.load_open, including its tree layout: most of the group sits
    under OPEN_TREE and OPEN_EXTRA_TREES names the ones that do not. The mirror
    has to be exact, because build() joins this roster to the one load_open
    returns and asserts they match -- a tree read there and not here shows up as
    a model with outcomes and no tokens, i.e. as a blank row rather than as the
    missing read it is.
    """
    usage = load_usage(corpus.OPEN_EFFORT, root, run,
                       panel=corpus.OPEN_MODEL_LABEL, tree=corpus.OPEN_TREE)
    for tree, panel in corpus.OPEN_EXTRA_TREES.items():
        extra = load_usage(corpus.OPEN_EFFORT, root, run,
                           panel=panel, tree=tree)
        assert not (usage.keys() & extra.keys()), (
            f"{tree} re-bills {sorted(usage.keys() & extra.keys())}, which the "
            f"pooled tree already supplied; one copy would be dropped")
        usage.update(extra)
    return usage


def load_usage_panel(effort=corpus.PRIMARY_EFFORT, root=corpus.ROOT_V8,
                     run=corpus.RUN_V8):
    """The API models plus, at high effort, the open-weight ones.

    Mirrors corpus.load_panel exactly, including the rule that at medium and low
    it returns only the API group, because the open-weight models were not run
    there.
    Each tree is still read against its OWN panel, so a missing cell cannot be
    absorbed as a smaller combined roster.
    """
    usage = load_usage(effort, root, run)
    if effort != corpus.OPEN_EFFORT:
        return usage
    o_usage = load_usage_open(root, run)
    assert not (usage.keys() & o_usage.keys()), (
        f"panels overlap on {sorted(usage.keys() & o_usage.keys())}; the merge "
        f"would drop one group's usage")
    return {**usage, **o_usage}


def build(effort=corpus.PRIMARY_EFFORT, panel=False):
    """display model name -> cost block joined to that model's high-effort outcome.

    The join is on the model, not on the task, and the two sides come from
    different files (trajectory jsonl for outcomes, interaction sidecar for
    tokens), so the task sets are cross-checked rather than assumed equal.

    `panel=True` adds the open-weight group. It is opt-in rather than the
    default because the medium self-check and any caller that wants the paper's
    original nine-model roster must keep getting exactly that.
    """
    if panel:
        eps, errs = corpus.load_panel(effort, keep_checks=False)
        usage = load_usage_panel(effort)
        label = corpus.PANEL_LABEL if effort == corpus.OPEN_EFFORT else corpus.MODEL_LABEL
    else:
        eps, errs = corpus.load_eval(effort, keep_checks=False)
        usage = load_usage(effort)
        label = corpus.MODEL_LABEL
    # The outcome side and the cost side are globbed independently, from
    # different files under different trees. Equal rosters is therefore a fact
    # to check, not a consequence of the code above.
    assert set(usage) == set(label), (
        f"cost covers {sorted(usage)} but the panel is {sorted(label)}")
    assert set(eps) == set(label.values()), (
        f"outcomes cover {sorted(eps)} but the panel is {sorted(label.values())}")
    out = {}
    for raw_model in label:
        model = label[raw_model]
        ep = eps[model]
        by_task = usage[raw_model]
        assert set(by_task) == set(ep), (
            f"{model}: {len(by_task)} tasks have token usage but {len(ep)} have "
            f"an eval row -- the cost and outcome columns would be over "
            f"different corpora ({len(set(by_task) ^ set(ep))} disagree)")
        passed = sum(1 for e in ep.values() if e["success"])
        c = summarize(by_task, passed)
        c["pass_pct"] = round(100 * passed / len(ep), 1)
        c["cap_pct"] = round(100 * sum(1 for e in ep.values() if e["cap_hit"])
                             / len(ep), 1)
        # Reasoning as a share of billed output. For Gemini `thoughts` bills at
        # the output rate; for Bedrock and Anthropic thinking bills inside
        # output_tokens. Either way the denominator is what output cost. None
        # where the provider does not separate the two: the share is not 0 and
        # not 100, it is unmeasured, and every consumer below must skip it
        # rather than substitute a number.
        if c["reason_known"]:
            billed_out = c["vis_out_total"] + c["reason_total"]
            c["reason_share"] = round(100 * c["reason_total"] / billed_out, 1)
        else:
            c["reason_share"] = None
        out[model] = c
    return out, eps, errs


def _grp(x):
    """Round half-up to an integer and group it the way LaTeX wants: 2{,}933.

    ROUND_HALF_UP, NOT python's round(). round() is banker's rounding and
    settles a .5 tie toward even, so round(730.5) is 730 while round(731.5) is
    732 -- two neighbouring rows would round in opposite directions off the
    same rule. No value in this table sits on a tie today, but the run columns
    are the two the reader is most likely to add up by hand, and a tie is one
    corpus rebuild away.
    """
    if x is None:
        # No published rate for this model, so no total to group. Rendering
        # "N/A" rather than an em-dash keeps it distinct from the `--` the
        # reasoning columns use for "the provider did not measure it": one cell
        # is a missing measurement, this one is a missing price, and the caption
        # has to explain only the second.
        return NA_CELL
    return f"{int(Decimal(str(x)).quantize(Decimal('1'), ROUND_HALF_UP)):,}" \
        .replace(",", "{,}")


def effort_response(draws=corpus.DRAWS):
    """Within-model pass-rate response to the reasoning-effort knob.

    WHY THIS EXISTS. The two rho's above are BETWEEN models: they say that
    models which ruminate more tend to score less. They cannot say what
    happens when you turn the knob up on ONE model, because every point is a
    different system and model identity is confounded with everything. The
    effort sweep already ran that experiment -- the same 600 tasks at low,
    medium and high on all ten API models -- and nothing was reading it as an
    experiment. This does.

    THE CONTRAST IS PAIRED AND CLUSTERED. Paired: a task appears at both
    efforts and enters as one unit, so task difficulty cancels instead of
    being averaged over. Clustered on seed, matching difficulty_bands'
    easy-hard delta: the 600 rounds are 20 scenarios deep and rounds inside a
    scenario are not independent draws. Pairing already removes most of the
    scenario effect, so the clustered interval here comes out close to the
    task-level one rather than far wider -- that is a property of this
    contrast, not a sign the clustering is inert, and it is the conservative
    of the two in any case.

    Returns model -> {"delta", "ci", "pass", "tokens"} with delta in points of
    pass rate, high minus low.
    """
    import collections
    import random

    specs = corpus.load_specs()
    eps = {L: corpus.load_eval(L, keep_checks=False)[0] for L in corpus.LEVELS}
    out = {}
    for m in sorted(set(eps["high"]) & set(eps["low"])):
        A, B = eps["high"][m], eps["low"][m]
        tids = sorted(set(A) & set(B))
        per = collections.defaultdict(list)
        for t in tids:
            per[specs[t]["seed"]].append((bool(A[t]["success"]),
                                          bool(B[t]["success"])))

        def stat(groups):
            hi = [x for g in groups for x, _ in g]
            lo = [y for g in groups for _, y in g]
            return 100.0 * sum(hi) / len(hi) - 100.0 * sum(lo) / len(lo)

        seeds = sorted(per)
        n = len(seeds)
        # Seeded on the model and its two pass counts, for the reason
        # difficulty_bands.band_sd gives: two models with identical counts
        # would otherwise share a stream and move together for no reason a
        # reader could infer.
        rnd = random.Random(corpus.stable_seed(
            "effort-response", m,
            sum(A[t]["success"] for t in tids),
            sum(B[t]["success"] for t in tids)))
        vals = sorted(stat([per[seeds[rnd.randrange(n)]] for _ in range(n)])
                      for _ in range(draws))
        out[m] = {
            "n": len(tids), "seeds": n,
            "delta": stat(list(per.values())),
            "ci": (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]),
            "pass": {L: 100.0 * sum(e["success"] for e in eps[L][m].values())
                     / len(eps[L][m]) for L in corpus.LEVELS},
        }
    return out


def print_effort_response():
    """Console block plus the counts the prose is allowed to quote."""
    resp = effort_response()
    # Tokens per round at each end of the knob, so the table shows the DOSE
    # next to the response. Without it a reader cannot tell whether "high"
    # actually bought more reasoning on a given model, and on one of them
    # (gemini-3.5-flash) the answer is that it saturates before high.
    tok = {L: build(L, panel=False)[0] for L in ("low", "high")}
    order = sorted(resp, key=lambda m: resp[m]["delta"])
    print(f"\nwithin-model effort response  (paired on task, "
          f"cluster bootstrap over {resp[order[0]]['seeds']} seeds, "
          f"{corpus.DRAWS} draws)")
    print("  the rho above is BETWEEN models; this is the same knob turned "
          "WITHIN one")
    w = max(len(m) for m in order)
    print(f"  {'model':<{w}} {'tok@low':>8} {'tok@high':>9} {'x':>5} | "
          f"{'low':>6} {'med':>6} {'high':>6} | {'D=Hi-Lo':>8}  "
          f"{'95% CI':>16}")
    helped = hurt = flat = 0
    for m in order:
        r = resp[m]
        lo, hi = r["ci"]
        if lo > 0:
            mark, _ = "  helps", (helped := helped + 1)
        elif hi < 0:
            mark, _ = "  HURTS", (hurt := hurt + 1)
        else:
            mark, _ = "      .", (flat := flat + 1)
        tl, th = (tok["low"][m]["reason_per_task"],
                  tok["high"][m]["reason_per_task"])
        print(f"  {m:<{w}} {tl:>8.0f} {th:>9.0f} {th / tl:>5.1f} | "
              f"{r['pass']['low']:>6.1f} {r['pass']['medium']:>6.1f} "
              f"{r['pass']['high']:>6.1f} | {r['delta']:>+8.1f}  "
              f"[{lo:>+6.1f},{hi:>+6.1f}]{mark}")
    print(f"  counts: {helped} helped, {hurt} hurt, {flat} indistinguishable "
          f"(n={len(order)})")
    # The one model the causal reading of rho actually fits, and the reason
    # the prose is allowed to name it: it is the only NEGATIVE interval, and
    # its collapse tracks the dose.
    worst = order[0]
    r = resp[worst]
    print(f"  sole inversion: {worst} {r['pass']['low']:.1f}% at low -> "
          f"{r['pass']['high']:.1f}% at high, {r['delta']:+.1f} "
          f"[{r['ci'][0]:+.1f}, {r['ci'][1]:+.1f}]")
    return resp


def main():
    frozen_eff, n = verify_frozen()
    print(f"{frozen_eff} self-check: {n} fields reproduce "
          f"{os.path.basename(corpus.STATS_JSON)} exactly "
          f"({len(FROZEN_KEYS)} models x {len(CHECKED)} fields)\n")

    eff = corpus.PRIMARY_EFFORT
    tab, eps, errs = build(eff, panel=True)
    order = corpus.model_order(eps)

    # THE TWO GROUPS ARE NOT ADDED TOGETHER, AND THIS IS THE REASON.
    # `billed` priced calls that a provider actually invoiced; `proxy` priced
    # calls that nobody sold, at the vendor's list rate (see PRICING_OPEN). A
    # pooled total, a pooled cheapest-solve or a pooled spread would be one
    # number made of two different kinds of thing, and no caption can repair
    # that. Every economics line below is therefore computed over `billed`
    # only -- which is also why they are unchanged from the nine-model run --
    # and the proxy group gets its own line that says what it is.
    open_labels = set(corpus.OPEN_MODEL_LABEL.values())
    billed = [m for m in order if m not in open_labels]
    proxy = [m for m in order if m in open_labels]
    assert len(billed) == len(corpus.MODEL_LABEL) and len(proxy) == len(open_labels), (
        f"roster split went wrong: {len(billed)} billed, {len(proxy)} proxy")
    # THE OPEN GROUP NO LONGER PRICES UNIFORMLY. One checkpoint has no published
    # rate (PRICE_KEY -> None), so its dollar cells are N/A and it cannot enter a
    # dollar subtotal; its TOKENS are measured like everyone else's and do enter
    # the token subtotal. The two subtotals therefore run over different rosters,
    # and the only honest handling is to say so wherever either is printed --
    # not to drop the row, and not to impute a rate so the arithmetic closes.
    proxy_usd = [m for m in proxy if tab[m]["cost_total"] is not None]
    unpriced = [m for m in order if tab[m]["cost_total"] is None]
    assert not (set(unpriced) & set(billed)), (
        f"billed API models {sorted(set(unpriced) & set(billed))} have no "
        f"price; the API subtotal would cover fewer rows than the table shows")

    def _c(v, w, prec, scale=1.0):
        """A numeric cell, or `--` where the provider did not measure it."""
        return f"{'--':>{w}}" if v is None else f"{v / scale:>{w}.{prec}f}"

    def _usd(v, w, prec):
        """A dollar cell, or the marker for a model with no published rate.

        Deliberately NOT `_c`. Both render an absence, but they are different
        absences: `--` means the provider never reported the quantity, N/A means
        it was never given a price. One row in this table carries both, and a
        reader who sees the same glyph in the token half and the money half will
        take them for the same fact.
        """
        return f"{NA_CELL:>{w}}" if v is None else f"{v:>{w}.{prec}f}"

    print(f"T2  cost    effort={eff}   population={corpus.PRIMARY_POPULATION} "
          f"(n={corpus.N_RAW})   all attempts billed, retries included")
    hdr = (f"{'model':<{corpus.NAME_W}} | {'$/round':>8} {'$/solved':>9} "
           f"{'pass%':>6} | {'calls':>6} {'in kt':>7} {'out':>6} {'reas kt':>8} "
           f"{'reas%':>6} | {'cap%':>6} {'$ total':>9}")
    print(hdr)
    print("-" * len(hdr))
    for m in order:
        c = tab[m]
        print(f"{m:<{corpus.NAME_W}} | {_usd(c['cost_per_task'], 8, 4)} "
              f"{_usd(c['cost_per_solved'], 9, 2)} {c['pass_pct']:>6.1f} | "
              f"{c['calls_per_task']:>6.1f} {c['in_per_task']/1000:>7.1f} "
              f"{c['vis_out_per_task']:>6d} "
              f"{_c(c['reason_per_task'], 8, 1, 1000)} "
              f"{_c(c['reason_share'], 6, 1)} | {c['cap_pct']:>6.1f} "
              f"{_usd(c['cost_total'], 9, 2)}")
    print(f"{'':<{corpus.NAME_W}}   ({len(proxy)} rows are list-rate proxies, "
          f"not spend: {', '.join(proxy)}; their `out` column includes "
          f"thinking, which is why `reas` is -- and not 0)")

    cheap = min(billed, key=lambda m: tab[m]["cost_per_solved"])
    dear = max(billed, key=lambda m: tab[m]["cost_per_solved"])
    best = max(billed, key=lambda m: tab[m]["pass_pct"])
    worst = min(billed, key=lambda m: tab[m]["pass_pct"])
    # Ratios are formed from the UNROUNDED aggregates. Dividing the two-decimal
    # dollar figures printed in the table would be rounding a rounded value, and
    # on the luna row (0.1429 -> 0.14) that is a 2% error before it starts.
    npass = {m: round(tab[m]["pass_pct"] * corpus.N_RAW / 100) for m in order}
    print(f"\nrun total, {len(billed)} API models, one pass over the corpus: "
          f"${sum(tab[m]['cost_total'] for m in billed):,.2f} over "
          f"{sum(tab[m]['calls_total'] for m in billed):,} calls")
    print(f"open-weight group, same corpus, AT LIST RATES AND NOT BILLED: "
          f"${sum(tab[m]['cost_total'] for m in proxy_usd):,.2f} over "
          f"{sum(tab[m]['calls_total'] for m in proxy_usd):,} calls "
          f"({len(proxy_usd)} of {len(proxy)} open rows; no published rate for "
          f"{', '.join(m for m in proxy if m not in proxy_usd) or 'none'})")
    print(f"pass spread     : {npass[best]}/{npass[worst]} rounds = "
          f"{npass[best]/npass[worst]:.1f}x")
    print(f"cheapest solve  : {cheap} at ${tab[cheap]['cost_per_solved']:.2f} "
          f"({tab[cheap]['pass_pct']:.1f}% pass, "
          f"{100*(npass[best]-npass[cheap])/corpus.N_RAW:.1f} points behind {best})")
    print(f"dearest solve   : {dear} at ${tab[dear]['cost_per_solved']:.2f}, "
          f"cost-per-solve spread "
          f"{tab[dear]['cost_per_solved']/tab[cheap]['cost_per_solved']:.1f}x")
    print(f"most accurate   : {best} at ${tab[best]['cost_per_solved']:.2f} "
          f"per solved round = "
          f"{tab[best]['cost_per_solved']/tab[cheap]['cost_per_solved']:.1f}x "
          f"the cheapest, and "
          f"{tab[best]['cost_per_task']/tab[cheap]['cost_per_task']:.1f}x per round")

    # Reasoning SHARE of billed output ranks capability more tightly than the
    # raw token count does. Both are printed because the paper quotes both and
    # the comparison is the point; with n = 9 neither is a test.
    #
    # THE OPEN-WEIGHT ROWS ARE ABSENT FROM BOTH CORRELATIONS, not zero in them.
    # Their endpoint does not report a reasoning count at all, so they have no
    # x-value; dropping them keeps n at 9 and keeps these two numbers the same
    # statistic the paper already quotes. Imputing 0 would invent the strongest
    # possible negative evidence out of a missing field.
    known = [m for m in order if tab[m]["reason_known"]]
    assert known == billed, (
        f"reasoning is measured for {known} but the billed roster is {billed}; "
        f"the correlations below would no longer be the nine-model statistic")
    ys = [tab[m]["pass_pct"] for m in known]
    r_share = spearman([tab[m]["reason_share"] for m in known], ys)
    r_tok = spearman([tab[m]["reason_per_task"] for m in known], ys)
    lo_s = min(known, key=lambda m: tab[m]["reason_share"])
    hi_s = max(known, key=lambda m: tab[m]["reason_share"])
    print(f"reasoning share : {tab[lo_s]['reason_share']:.1f}% ({lo_s}) to "
          f"{tab[hi_s]['reason_share']:.1f}% ({hi_s})")
    print(f"  rho vs pass, reasoning SHARE of output : {r_share:+.3f}")
    print(f"  rho vs pass, reasoning TOKENS per round: {r_tok:+.3f}")
    print(f"  reasoning tokens per round span "
          f"{max(tab[m]['reason_per_task'] for m in known)
             / min(tab[m]['reason_per_task'] for m in known):.1f}x")
    # Leave-one-out, because the first thing anyone asks of a rank correlation
    # over ten points is whether one point is carrying it. gemini-3.5-flash is
    # the obvious suspect -- it is the extreme on both axes -- and the answer
    # is no: dropping it moves the token rho from -0.697 to -0.583, which is
    # the weakest the statistic gets under any single deletion. The
    # correlation is real. What it cannot do is establish direction, which is
    # what the within-model block below is for.
    loo_t = sorted((spearman([tab[x]["reason_per_task"] for x in known if x != m],
                             [tab[x]["pass_pct"] for x in known if x != m]), m)
                   for m in known)
    print(f"  leave-one-out rho(tokens): {loo_t[0][0]:+.3f} (drop "
          f"{loo_t[0][1]}) to {loo_t[-1][0]:+.3f} (drop {loo_t[-1][1]})")
    print(f"  (n={len(known)}; {len(proxy)} open-weight models report no "
          f"reasoning count and are excluded, not imputed)")

    # A rank correlation over ten models is not a test and the comment above
    # says so. The block below is the test it is not: same knob, one model at
    # a time, on tasks that are held fixed. Printed immediately after the
    # rho's because the two answer different questions and quoting the first
    # without the second is what let the causal reading stand unchallenged.
    print_effort_response()

    # The integers the prose quotes, rounded ONCE from the unrounded ratios.
    # Recomputing them from the two-decimal dollars in the table would round a
    # rounded value and drift; this line is the single place they are formed.
    print(f"prose ratios    : solve-spread "
          f"{tab[dear]['cost_per_solved']/tab[cheap]['cost_per_solved']:.0f}x, "
          f"leader-over-cheapest per solve "
          f"{tab[best]['cost_per_solved']/tab[cheap]['cost_per_solved']:.0f}x, "
          f"per round "
          f"{tab[best]['cost_per_task']/tab[cheap]['cost_per_task']:.0f}x")
    if errs:
        print(f"provider errors retained as failures: {errs}")

    with open(HIGH_JSON, "w") as fh:
        json.dump({"effort": eff, "population": corpus.PRIMARY_POPULATION,
                   "models": tab}, fh, indent=2, sort_keys=True)
    print(f"\nwrote {os.path.basename(HIGH_JSON)} "
          f"(the frozen medium artifact is NOT touched)")

    # THE TWO RUN COLUMNS ARE RESCALINGS, NOT NEW MEASUREMENTS. `$ total` is
    # cost_per_task x 600 and `tok (M)` is (in + out + reas) x 600, to within
    # the rounding. They are here because the per-round figures hide the scale a
    # reader actually budgets against, which is what appendix Figure
    # \ref{fig:rep-cost} shows and this table did not -- but the caption has to
    # say they are derived, or a reviewer will read two extra columns as two
    # extra pieces of evidence.
    #
    # TOKENS ARE PRINTED IN MILLIONS, NOT AS THE 9-DIGIT INTEGERS THE APPENDIX
    # FIGURE USES. At 11 columns inside an adjustbox, one 570,635,183-wide cell
    # shrinks the whole table; nothing at this scale turns on the last six
    # digits. The exact integers stay in cost_high.json.
    # THE TWO RUN COLUMNS SIT LAST, NOT IN THE MIDDLE. They are the only cells
    # in the table that are per-RUN rather than per-round, so putting them at
    # the right edge stops the eye from reading `$ total` as another per-round
    # rate on its way across the per-round group.
    #
    # THREE EFFORT ROWS PER API MODEL, ONE FOR THE OPEN GROUP. Table 1 gained a
    # Think Level column and this table mirrors it, so the two read the same way
    # -- but the open-weight group was only ever run at high, so it keeps a
    # single row here exactly as it does there. medium and low are built with
    # panel=False for that reason: asking for the twelve-model panel at an
    # effort three of them were never run at would be a roster error, not a
    # smaller table.
    tab_med, _eps_med, _ = build("medium", panel=False)
    tab_low, _eps_low, _ = build("low", panel=False)
    blocks = corpus.provider_blocks(eps)

    def _cells(c):
        """The per-round and per-run cells of one row, run columns last."""
        return [_usd(c["cost_per_task"], 0, 2), _usd(c["cost_per_solved"], 0, 2),
                f"{c['calls_per_task']:.1f}",
                f"{c['in_per_task']/1000:.0f}",
                f"{c['vis_out_per_task']/1000:.1f}",
                _c(c["reason_per_task"], 0, 1, 1000).strip(),
                _c(c["reason_share"], 0, 1).strip(),
                f"{c['cap_pct']:.1f}",
                _grp(c["cost_total"]), _grp(c["tok_total"] / 1e6)]

    print("\n% ---- LaTeX body for T2 across all efforts (paste into the tabular) ----")
    print("%      column spec: l c | cc | cccc | cc | cc")
    for i, (_prov, ms) in enumerate(blocks):
        if i:
            print(r"    \midrule")
        for j, m in enumerate(ms):
            multi = m in tab_med and m in tab_low
            if j > 0 and multi:
                print(r"    \addlinespace[2.5pt]")
            if multi:
                rows = [("High", tab[m]), ("Med", tab_med[m]), ("Low", tab_low[m])]
                for k, (eff_label, c) in enumerate(rows):
                    if k == 0:
                        prefix = (f"    \\multirow{{3}}{{*}}{{\\texttt{{{m}}}}}"
                                  + " " * (18 - len(m)))
                    else:
                        prefix = " " * (36 - len(m)) + " " * len(m)
                    print(f"{prefix} & {eff_label:<4} & "
                          + " & ".join(_cells(c)) + r" \\")
            else:
                print(f"    \\texttt{{{m}}}" + " " * (18 - len(m))
                      + " & High & " + " & ".join(_cells(tab[m])) + r" \\")
    # THE TOTAL ROW FILLS ONLY THE CELLS THAT ADD UP. $/round and $/solved would
    # be pooled means over a mixed roster and calls and the token columns are
    # per-round rates; a "total" under any of them is a different statistic
    # wearing the same word. Blank is the honest cell.
    #
    # Both totals are formed from the UNROUNDED aggregates, so they do not equal
    # the sum of the rounded cells printed above them -- add the column by hand
    # and you get 11,432 and 3,953. That is a rounding artifact, not a
    # disagreement, and the caption says so rather than leaving the reader to
    # discover it. Rounding the column to the cent instead would make it add up
    # and cost the width this table does not have.
    #
    # THERE ARE TWO TOTAL ROWS AND THERE IS NO GRAND TOTAL. One sums money that
    # was invoiced, the other sums money that was not (PRICING_OPEN). Adding
    # them would produce a figure that is neither, and it is precisely the
    # figure a reader would quote. The API subtotal is also, deliberately, the
    # same number this table carried when it had nine rows.
    def _lbl(s):
        # Aligns with the \texttt{model} rows above, which occupy 31 columns.
        return "    " + s + " " * (27 - len(s))

    # ONE TOTAL ROW PER EFFORT, NOT ONE FOR THE TABLE. With three effort rows
    # per API model a single "Total" would be a sum over a column the reader
    # cannot identify -- it would look like the whole table added up, which is
    # three runs of the same nine models and a quantity nobody spent. Each row
    # closes exactly the effort it names. The open group still has only the one,
    # because it was only run at high.
    #
    # THE RENDERED LABEL DOES NOT REPEAT THE EFFORT; the Think column already
    # carries it. Spelling it in both places produced "Total (API, billed)" for
    # high next to "Total (API, billed, Med)" for medium, which reads as though
    # the first row were the effort-independent one. The caption facts below
    # still need a unique key per row, so they append "@ <effort>" -- that lives
    # in a comment the reader never sees and the regex in results_facts does.
    print(r"    \midrule")
    groups = [("Total (API, billed)", billed, tab, "High"),
              ("Total (API, billed)", billed, tab_med, "Med"),
              ("Total (API, billed)", billed, tab_low, "Low"),
              ("Total (open, list rate)", proxy, tab, "High")]
    facts = []
    for name, ms, src, eff_label in groups:
        # THE DOLLAR SUBTOTAL AND THE TOKEN SUBTOTAL CAN COVER DIFFERENT ROWS.
        # A row with no published rate contributes tokens but no money, so it
        # belongs in one sum and not the other. Summing it as zero would quietly
        # shrink the group's cost; omitting it from the token sum would quietly
        # shrink its size. Each sum takes the rows it actually has.
        ms_usd = [m for m in ms if src[m]["cost_total"] is not None]
        cells_usd = [_grp(src[m]["cost_total"]) for m in ms_usd]
        cells_tok = [_grp(src[m]["tok_total"] / 1e6) for m in ms]
        tot_usd = _grp(sum(src[m]["cost_total"] for m in ms_usd))
        tot_tok = _grp(sum(src[m]["tok_total"] for m in ms) / 1e6)
        # think + eight blanks (the per-round group) + the two run columns.
        print(_lbl(name) + f" & {eff_label:<4} " + "& " * 8
              + f"& {tot_usd} & {tot_tok} " + r"\\")
        if len(ms_usd) != len(ms):
            print(f"% caption fact [{name} @ {eff_label}] ROSTER SPLIT: the $ "
                  f"total covers {len(ms_usd)} of {len(ms)} rows in this group "
                  f"and the tok (M) total covers all {len(ms)}; "
                  f"{', '.join(m for m in ms if m not in ms_usd)} has no "
                  f"published rate and shows {NA_CELL} in every dollar cell. "
                  f"The caption must state this or the $ column reads as a "
                  f"total over the rows printed above it.")
        facts.append((f"{name} @ {eff_label}", cells_usd, cells_tok,
                      tot_usd, tot_tok))

    # The caption tells the reader each Total row sits BELOW the column it
    # closes, and by how much. That gap is an arithmetic claim about the
    # rendered cells, so it is derived from the rendered cells here rather than
    # typed into the caption and trusted. Add the printed column by hand and you
    # get these sums; the Total row is the unrounded aggregate.
    def _plain(cs):
        return sum(int(c.replace("{,}", "")) for c in cs)

    # THE GAP IS SIGNED, and on some rows it is negative. Rounding each cell to
    # the nearest integer moves it either way, so the rounded column can land
    # just under the unrounded total as easily as over it -- the medium API row
    # does exactly that. Printing it signed keeps one sentence for both cases;
    # a negative value reads as "the total sits above the printed column". The
    # captions only quote the high-effort rows, where the gap is positive, so
    # the awkward reading never reaches the paper.
    for name, cells_usd, cells_tok, tot_usd, tot_tok in facts:
        print(f"% caption fact [{name}]: the printed $ cells sum to "
              f"{_plain(cells_usd):,} and the printed tok (M) cells to "
              f"{_plain(cells_tok):,}, so the row sits "
              f"{_plain(cells_usd) - int(tot_usd.replace('{,}', '')):+,} dollars "
              f"and {_plain(cells_tok) - int(tot_tok.replace('{,}', '')):+,} "
              f"million tokens below them")
    return 0


if __name__ == "__main__":
    sys.exit(main())
