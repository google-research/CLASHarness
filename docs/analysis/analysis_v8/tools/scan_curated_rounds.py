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

"""Extract the eight curated rounds of section 10 from the raw v8 corpus.

Section 10 of ``cross-model-behavioral-comparison.md`` makes roughly a hundred
very specific per-round, per-model claims: scores, step counts, which rubric
unit each model took, who a message was addressed to and at which step, how
many assets a model opened, why a run stopped. Those were originally verified
against throwaway dumps in ``/tmp``, which means the claims stop being
checkable the moment ``/tmp`` is cleared.

This script promotes that derivation into the repository. It reads the raw
eval jsonl, distils exactly the fields section 10 talks about, and writes
``curated_rounds.json`` next to the report. ``tools/audit_report.py`` then
re-checks the prose against that file, so a stale number in section 10 fails
the gate instead of surviving unnoticed.

Message bodies and file contents are deliberately NOT stored: they are large,
they are the model's own text rather than a measurement, and section 10 only
ever makes claims about their recipients, channels and paths.

Usage::

    python tools/scan_curated_rounds.py --root $CLASHARNESS_DATA_ROOT
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

# Corpus root: the data checkout holding eval_results/. Mirrors
# analysis/corpus.py -- CLASHARNESS_DATA_ROOT, else <repo>/data. Five dirnames
# walk tools -> analysis_v8 -> analysis -> docs -> repo root.
DEFAULT_ROOT = os.environ.get(
    "CLASHARNESS_DATA_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))), "data"))

# Same table and layout the report builder uses; kept local so this script can
# run standalone. THIRTEEN models, ordered by provider so the emitted JSON keeps
# provider blocks contiguous. Mirrors build_behavior_report.MODELS -- if you add a
# model there, add it here, or audit_report's S10 block silently checks a smaller
# field than the report renders.
MODELS = [
    ("3.5-flash", "Gemini/3_5_flash"),
    ("3.6-flash", "Gemini/3_6_flash"),
    ("3.7-flash", "Gemini/3_7_flash"),
    ("3.1-pro", "Gemini/3_1_pro"),
    ("opus-4.8", "Claude/opus_4_8"),
    ("opus-5", "Claude/opus_5"),
    ("sonnet-5", "Claude/sonnet_5"),
    ("sol", "OpenAI/5_6_sol"),
    ("luna", "OpenAI/5_6_luna"),
    ("terra", "OpenAI/5_6_terra"),
    ("v4-pro", "DeepSeek/v4_pro"),
    ("kimi-k3", "Moonshot/kimi_k3"),
    ("glm-5.2", "ZhipuAI/glm_5_2"),
]

TREE = "eval_results/data_3_6_flash_v8/aligned_setting_{eff}"

# The three open-source models were not run as an effort sweep, so they are pinned
# to one subtree and IGNORE --effort. Value is a complete eval_results-relative
# path. Mirrors build_behavior_report.FIXED_TREE.
FIXED_TREE = {
    "v4-pro": "eval_results/data_3_6_flash_v8/open_source_high/DeepSeek/v4_pro",
    "kimi-k3": "eval_results/data_3_6_flash_v8/open_source_high/Moonshot/kimi_k3",
    "glm-5.2": "eval_results/data_3_6_flash_v8/open_source_high/ZhipuAI/glm_5_2",
}

# How much of the grader's rationale to keep. Long enough that the recipient
# and channel it credits always survive; short enough to keep the file legible.
JUDGE_CHARS = 300

# The eight rounds section 10 walks through, in the order the prose does.
# Each entry is (scenario file stem, round suffix).
#
# MUST MIRROR build_behavior_report.CURATED_ROUNDS. Those rounds are PINNED there
# rather than re-derived, because the max-spread selection rule saturates on this
# corpus (all eight scenarios reach spread 1.000 and five reach it on several
# rounds at once), so re-deriving would let loop order pick the evidence. If this
# list and CURATED_ROUNDS ever disagree, the prose in section 10 describes one
# round while the report renders another.
ROUNDS = [
    ("payment_chargeback_dispute_v01", "r04"),
    ("warehouse_robotics_safety_halt_v02", "r01"),
    ("satellite_ground_station_handover_v02", "r01"),
    ("clinical_trial_dosage_conflict_v02", "r03"),
    ("coordinated_containment_v02", "r04"),
    ("distributed_lock_shared_resource_v01", "r02"),
    ("expense_reimbursement_conflict_v02", "r05"),
    ("expiring_resource_hold_v03", "r04"),
]

# Action strings are CLI-shaped, e.g.
#   message send --channel '#ops' --target Avery_X --message '...'
# Only the leading verb pair and a few flags are ever cited by the prose.
_VERB = re.compile(r"^\s*([a-z_]+(?:\s+[a-z_]+)?)")
_FLAG = re.compile(r"--(\w+)\s+(?:'([^']*)'|\"([^\"]*)\"|(\S+))")


def parse_action(raw):
    """Return (verb, {flag: value}) for one trajectory action string."""
    if not isinstance(raw, str):
        return None, {}
    m = _VERB.match(raw)
    verb = m.group(1).strip() if m else ""
    # A single-word verb followed by a flag (e.g. "wait 60") keeps just the word.
    if " " in verb and verb.split()[1].startswith("-"):
        verb = verb.split()[0]
    flags = {}
    for fm in _FLAG.finditer(raw):
        val = next(g for g in fm.groups()[1:] if g is not None)
        # Only the first occurrence matters; later ones are content we discard.
        flags.setdefault(fm.group(1), val)
    return verb, flags


def distil(rec):
    """Reduce one task record to the facts section 10 can cite."""
    verbs = Counter()
    messages, files, opens = [], [], []
    for t in rec.get("trajectory") or []:
        raw = t.get("action") or t.get("executed_action")
        verb, flags = parse_action(raw)
        if not verb:
            continue
        verbs[verb] += 1
        step = t.get("step")
        if verb == "message send":
            messages.append({"step": step,
                             "target": flags.get("target"),
                             "channel": flags.get("channel")})
        elif verb in ("file create", "file write"):
            files.append({"step": step, "path": flags.get("path")})
        elif verb == "asset open":
            opens.append({"step": step, "id": flags.get("id")})

    checks = []
    for c in (rec.get("metadata") or {}).get("check_results") or []:
        # The grader's own rationale. For semantic_match checks this names the
        # recipient and channel it actually credited, which is the only
        # evidence that attributes a passing notification to one message rather
        # than another -- an attribution the prose gets wrong if it guesses.
        judge = str(c.get("message") or "")
        checks.append({"name": c.get("name"),
                       "role": c.get("role"),
                       "passed": bool(c.get("passed")),
                       "judge": judge[:JUDGE_CHARS]})
    req = [c for c in checks if c["role"] == "required"]
    return {
        "score": rec.get("score"),
        "steps": rec.get("steps"),
        "success": bool(rec.get("success")),
        "stop_reason": rec.get("stop_reason"),
        "checks": checks,
        "n_required": len(req),
        "n_required_passed": sum(1 for c in req if c["passed"]),
        "verbs": dict(sorted(verbs.items())),
        "messages": messages,
        "files": files,
        "opens": opens,
        # Verb-only trace: enough to support claims like "eight consecutive
        # message search calls from step 12 to 19" without storing model prose.
        "verb_trace": [parse_action((t.get("action") or t.get("executed_action")))[0]
                       for t in (rec.get("trajectory") or [])],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help="corpus root holding eval_results/")
    # Default mirrors build_behavior_report.PRIMARY_EFFORT. The FIXED_TREE models
    # ignore it -- they have exactly one setting each.
    ap.add_argument("--effort", default="high", choices=("low", "medium", "high"))
    ap.add_argument("--out", default=None,
                    help="output path (default: curated_rounds.json beside the report)")
    args = ap.parse_args(argv)

    root = os.path.expanduser(args.root)
    base = os.path.join(root, TREE.format(eff=args.effort))
    if not os.path.isdir(base):
        sys.exit(f"corpus not found: {base}")
    for nm, fixed in FIXED_TREE.items():
        if not os.path.isdir(os.path.join(root, fixed)):
            sys.exit(f"corpus not found for {nm}: {os.path.join(root, fixed)}")

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = args.out or os.path.join(here, "curated_rounds.json")

    result = {}
    missing = []
    for scenario, rnd in ROUNDS:
        key = f"{scenario}_{rnd}"
        per_model = {}
        for name, slug in MODELS:
            mbase = (os.path.join(root, FIXED_TREE[name]) if name in FIXED_TREE
                     else os.path.join(base, slug))
            path = os.path.join(mbase, "r01", f"{scenario}.jsonl")
            if not os.path.exists(path):
                missing.append(f"{name}: {path}")
                continue
            hit = None
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if str(rec.get("task_id", "")).endswith("_" + rnd):
                        hit = rec
                        break
            if hit is None:
                missing.append(f"{name}: {key} not in {path}")
                continue
            per_model[name] = distil(hit)
        result[key] = per_model

    if missing:
        # A silently missing model would turn a real disagreement into a
        # vacuous pass downstream, so refuse to emit a partial file.
        sys.exit("ABORT, incomplete corpus:\n  " + "\n  ".join(missing))

    # `effort` describes the ten swept models only; the three in FIXED_TREE are
    # recorded separately so a reader cannot mistake them for having been swept.
    payload = {"effort": args.effort, "fixed_tree": FIXED_TREE, "rounds": result}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
        fh.write("\n")

    n = sum(len(v) for v in result.values())
    print(f"wrote {out_path}")
    print(f"  {len(result)} rounds x {len(MODELS)} models = {n} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
