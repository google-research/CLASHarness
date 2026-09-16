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

"""Full 30-cell reasoning-token scan for v8: ten swept models x three efforts.

Schema differs per family (verified empirically on medium/r01):
  Gemini  : usage.thoughts               ; usage.candidates EXCLUDES thoughts
  OpenAI  : usage.output_tokens_details.reasoning_tokens ; output_tokens INCLUDES it
  Claude  : usage.output_tokens_details.thinking_tokens  ; output_tokens INCLUDES it

Bytes-regex extraction (multi_match asserted == 0 on sample); only lines
carrying record_type=attempt are considered.

THE THREE OPEN-SOURCE MODELS ARE ABSENT FROM THIS SCAN, FOR TWO INDEPENDENT
REASONS, AND THAT ABSENCE IS DELIBERATE:

  1. They were not run as an effort sweep. All three ran at exactly one setting,
     high: kimi-k3 requested medium on the API field and was served high, while
     deepseek-v4-pro and glm-5.2 carry `reasoning_effort: high` in
     `chat_template_kwargs` and leave the API field empty. (They DO record an
     effort -- do not repeat the earlier claim that they record none; it came from
     reading only `run_config.reasoning_effort`.) One setting means no
     low/medium/high triple to put in a cell, so `mult_low_high` -- the column
     this table exists for -- is undefined for them.

  2. More importantly, their responses carry NO reasoning-token field of any kind.
     Scanning their logs for `thoughts`, `reasoning_tokens`, `thinking_tokens`,
     `output_tokens_details` and `completion_tokens_details` returns zero matches
     on every attempt: their usage block is the older Chat-Completions shape
     (`prompt_tokens` / `completion_tokens` / `total_tokens` /
     `prompt_tokens_details`), which has nowhere to put a reasoning split.

Reason 2 is the one that matters editorially. If they were added here they would
score reason_sum = 0 and zero_pct = 100.0%, which reads as the measurement "these
models did no thinking" when the truth is "this provider does not report thinking".
Those are opposite claims. build_behavior_report.py carries a `reason_measured`
flag for exactly this distinction; keep the two consistent and never render an
unreported field as a measured zero.
"""

import json
import os
import re
import glob
from concurrent.futures import ProcessPoolExecutor

# Corpus root: the data checkout holding eval_results/. Mirrors
# analysis/corpus.py -- CLASHARNESS_DATA_ROOT, else <repo>/data. Five dirnames
# walk tools -> analysis_v8 -> analysis -> docs -> repo root.
DEFAULT_ROOT = os.environ.get(
    "CLASHARNESS_DATA_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))), "data"))
ROOT = os.path.join(DEFAULT_ROOT, "eval_results", "data_3_6_flash_v8")

# Order mirrors MODELS in build_behavior_report.py (provider blocks, Gemini first)
# minus the three open-source models, which are excluded for the two reasons in the
# module docstring. Keep the two lists in the same order: DISPLAY below is iterated
# in insertion order and fixes the key order of reasoning_tokens.json.
MODELS = [
    ("flash", "Gemini/3_5_flash", "gemini"),
    ("flash36", "Gemini/3_6_flash", "gemini"),
    ("flash37", "Gemini/3_7_flash", "gemini"),
    ("pro", "Gemini/3_1_pro", "gemini"),
    ("opus", "Claude/opus_4_8", "claude"),
    ("opus5", "Claude/opus_5", "claude"),
    ("sonnet", "Claude/sonnet_5", "claude"),
    ("sol", "OpenAI/5_6_sol", "openai"),
    ("luna", "OpenAI/5_6_luna", "openai"),
    ("terra", "OpenAI/5_6_terra", "openai"),
]
EFFORTS = ["low", "medium", "high"]

# Run-subdirectory selection. THIS MUST MIRROR build_behavior_report.py's
# DEFAULT_RUN / RUN_SUBDIR / run_for, and the reason is not cosmetic: the
# pass_pct column of the table this script writes is not measured here at all --
# build_table() reads it out of cross-model-behavioral-comparison.stats.json,
# which build_behavior_report.py produced under ITS run policy. If the two
# policies disagree, one row of reasoning_tokens.json ends up splicing token
# counts from one run onto the pass rate of another, and nothing downstream can
# detect it because both halves are individually well-formed.
#
# That is exactly what happened. This scan hardcoded "r01" while
# build_behavior_report.py reported gemini-3.7-flash on r02, so the high cell
# carried r01 tokens (230.5/attempt over 10,979 attempts) beside a pass rate that
# was, once stats.json was rebuilt, r02's 20.3%. Do not reintroduce a literal
# run directory below.
#
# The override applies at PRIMARY_EFFORT only: r02 exists for 3_7_flash under
# aligned_setting_high and nowhere else, so returning it for a low/medium cell
# would resolve to a missing directory and fail far from the cause.
PRIMARY_EFFORT = "high"
DEFAULT_RUN = "r01"
RUN_SUBDIR = {"flash37": "r02"}


def run_for(key, eff):
    if eff == PRIMARY_EFFORT:
        return RUN_SUBDIR.get(key, DEFAULT_RUN)
    return DEFAULT_RUN


RE_REASON = re.compile(rb'"(?:thoughts|reasoning_tokens|thinking_tokens)":\s*(\d+)')
RE_OUT_G = re.compile(rb'"candidates":\s*(\d+)')
RE_OUT_OC = re.compile(rb'"output_tokens":\s*(\d+)')
RE_ATTEMPT = re.compile(rb'"record_type":\s*"attempt"')


def scan_cell(args):
    key, rel, fam, eff = args
    run = run_for(key, eff)
    d = f"{ROOT}/aligned_setting_{eff}/{rel}/{run}/log"
    files = sorted(glob.glob(f"{d}/*.interactions.jsonl"))
    if not files:
        raise SystemExit(f"ABORT: no interaction logs under {d}")
    re_out = RE_OUT_G if fam == "gemini" else RE_OUT_OC

    reason = []
    out_sum = 0
    out_n = 0
    multi_reason = 0
    multi_out = 0
    attempts = 0
    no_reason_field = 0

    # header check: requested vs effective effort
    hdr_req = set()
    hdr_eff = set()
    hdr_n = 0

    for f in files:
        with open(f, "rb") as fh:
            first = True
            for line in fh:
                if first:
                    first = False
                    try:
                        h = json.loads(line)
                    except Exception:
                        h = {}
                    rc = h.get("run_config") or {}
                    if "reasoning_effort" in rc:
                        hdr_n += 1
                        hdr_req.add(rc.get("reasoning_effort"))
                        hdr_eff.add(rc.get("reasoning_effort_effective"))
                    continue
                if not RE_ATTEMPT.search(line):
                    continue
                attempts += 1
                mr = RE_REASON.findall(line)
                if len(mr) > 1:
                    multi_reason += 1
                if mr:
                    reason.append(int(mr[0]))
                else:
                    no_reason_field += 1
                mo = re_out.findall(line)
                if len(mo) > 1:
                    multi_out += 1
                if mo:
                    out_sum += int(mo[0])
                    out_n += 1

    reason.sort()
    n = len(reason)

    def pct(p):
        if not n:
            return None
        return reason[min(n - 1, int(round(p * (n - 1))))]

    return {
        "key": key,
        "effort": eff,
        "family": fam,
        # Recorded so reasoning_tokens.raw.json shows which draw each cell came
        # from; it is the only place the choice is visible after the scan.
        "run": run,
        "files": len(files),
        "attempts": attempts,
        "reason_n": n,
        "reason_sum": sum(reason),
        "reason_mean": (sum(reason) / n) if n else None,
        "reason_median": pct(0.5),
        "reason_p90": pct(0.9),
        "reason_max": reason[-1] if n else None,
        "reason_zero": sum(1 for x in reason if x == 0),
        "no_reason_field": no_reason_field,
        "out_sum": out_sum,
        "out_n": out_n,
        "out_mean": (out_sum / out_n) if out_n else None,
        "multi_reason": multi_reason,
        "multi_out": multi_out,
        "hdr_n": hdr_n,
        "hdr_req": sorted(x for x in hdr_req if x is not None),
        "hdr_eff": sorted(x for x in hdr_eff if x is not None),
    }


HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_OUT = os.path.join(HERE, "tools", "reasoning_tokens.raw.json")
TABLE_OUT = os.path.join(HERE, "reasoning_tokens.json")
STATS = os.path.join(HERE, "cross-model-behavioral-comparison.stats.json")

# NOTE: reasoning_tokens.json is consumed as a PLAIN MODEL->CELLS mapping --
# audit_report.py iterates `for disp, e in RT.items()` and assumes every value is a
# model entry with low/medium/high keys. Do NOT add metadata (run date, effort,
# corpus id) as a top-level key here; that loop would crash on it. Metadata belongs
# in reasoning_tokens.raw.json, which is not iterated that way.
DISPLAY = {"flash": "3.5-flash", "flash36": "3.6-flash", "flash37": "3.7-flash",
           "pro": "3.1-pro",
           "opus": "opus-4.8", "opus5": "opus-5", "sonnet": "sonnet-5",
           "sol": "sol", "luna": "luna", "terra": "terra"}


def build_table(cells):
    """Collapse the 30 raw cells into the per-model table the report cites.

    Three details matter and are easy to get wrong:

    * ``per_attempt`` divides by *attempts*, not by ``reason_n``. Gemini omits
      the ``thoughts`` field entirely when a call did no thinking -- 6,985 calls
      corpus-wide, 96.6% of them a Flash model at low effort (3,707 on 3.6-flash,
      3,039 on 3.7-flash; the 239 remainder are 3.5-flash and 3.1-pro) -- and
      those calls are genuine zeros. Dividing by ``reason_n`` would silently drop
      them and inflate the low-effort mean by 2.3x on 3.6-flash and 1.6x on
      3.7-flash. Adding 3.7-flash to this scan moved the corpus-wide total from
      3,946; if you re-add or drop a model, recompute both figures from
      reasoning_tokens.raw.json rather than editing the prose.
    * ``zero_pct`` therefore counts ``reason_zero + no_reason_field``.
    * ``mult_low_high`` divides the *unrounded* per-attempt means. Taking the
      ratio of the rounded values moves the last digit on several models.
    """
    by_key = {(c["key"], c["effort"]): c for c in cells}
    eff_cells = json.load(open(STATS))["effort"]["cells"]
    table = {}
    for key, disp in DISPLAY.items():
        entry = {}
        for eff in EFFORTS:
            c = by_key[(key, eff)]
            n_tasks = eff_cells[f"{key}@{eff}"]["n"]
            entry[eff] = {
                "per_attempt": round(c["reason_sum"] / c["attempts"], 1),
                "per_task": round(c["reason_sum"] / n_tasks),
                "zero_pct": round(
                    100 * (c["reason_zero"] + c["no_reason_field"]) / c["attempts"], 1),
                "attempts": c["attempts"],
                "pass_pct": eff_cells[f"{key}@{eff}"]["pass_pct"],
            }
        lo, hi = by_key[(key, "low")], by_key[(key, "high")]
        entry["mult_low_high"] = round(
            (hi["reason_sum"] / hi["attempts"]) / (lo["reason_sum"] / lo["attempts"]), 2)
        table[disp] = entry
    return table


def main():
    jobs = [(k, rel, fam, eff) for (k, rel, fam) in MODELS for eff in EFFORTS]
    out = []
    with ProcessPoolExecutor(max_workers=10) as ex:
        for r in ex.map(scan_cell, jobs):
            out.append(r)
            print(
                f"{r['key']:8s} {r['effort']:6s} files={r['files']:3d} att={r['attempts']:6d} "
                f"rn={r['reason_n']:6d} mean={r['reason_mean']}",
                flush=True,
            )
    multi = sum(r["multi_reason"] for r in out)
    if multi:
        raise SystemExit(
            f"ABORT: {multi} log lines matched the reasoning-token regex more than "
            f"once, so the per-call attribution is ambiguous. Expected 0.")
    with open(RAW_OUT, "w") as f:
        json.dump(out, f, indent=1)
    with open(TABLE_OUT, "w") as f:
        json.dump(build_table(out), f, indent=1)
    print(f"WROTE {RAW_OUT}")
    print(f"WROTE {TABLE_OUT}")


if __name__ == "__main__":
    main()
