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

"""Build the per-task difficulty table behind section 8 of the report.

Section 8 argues that scored-unit count is the only candidate difficulty axis
in the v8 corpus that has not degenerated, and then reads round type, natural
images and tool family through it. Every number in it is an aggregate over 600
tasks joined against 9 model runs. This script performs that join once and
writes ``difficulty_axes.json``; ``tools/audit_report.py`` recomputes the
section's tables from that file and compares them to the prose.

Two joins matter and are easy to get wrong:

* **Scored units come from the run, not the spec.** A task's rubric lists more
  checks than it scores; only ``role: required`` entries are scored units. The
  count is taken from a model whose run covers all 600 tasks.
* **Task identity is the spec ``task_id``**, not the scenario, because a
  scenario contributes ten rounds with different rubrics.

Usage::

    python tools/scan_difficulty_axes.py --root $CLASHARNESS_DATA_ROOT
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Corpus root: the data checkout holding eval_results/. Mirrors
# analysis/corpus.py -- CLASHARNESS_DATA_ROOT, else <repo>/data. Five dirnames
# walk tools -> analysis_v8 -> analysis -> docs -> repo root.
DEFAULT_ROOT = os.environ.get(
    "CLASHARNESS_DATA_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))), "data"))

# THIRTEEN models, provider-ordered, mirroring build_behavior_report.MODELS.
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

EVAL_TREE = "eval_results/data_3_6_flash_v8/aligned_setting_{eff}"
SPEC_TREE = "data/scenarios/conflict_vlm_v8_3.6_flash"

# Models pinned to one subtree, ignoring --effort: they were not run as an effort
# sweep. Complete eval_results-relative paths, mirroring
# build_behavior_report.FIXED_TREE.
FIXED_TREE = {
    "v4-pro": "eval_results/data_3_6_flash_v8/open_source_high/DeepSeek/v4_pro",
    "kimi-k3": "eval_results/data_3_6_flash_v8/open_source_high/Moonshot/kimi_k3",
    "glm-5.2": "eval_results/data_3_6_flash_v8/open_source_high/ZhipuAI/glm_5_2",
}

# Models that ran at multimodal_input_mode=path: they were handed asset file paths
# as text and never received an image (verified over 600/600 episode headers each).
# Section 8.3 slices difficulty by the `natural_image` axis, and these two met that
# axis under a different input condition from the other eleven, so the flag travels
# with the data and the slice can be read with that in mind.
#
# State it as a difference in condition, NOT as an explanation of their scores. On
# this corpus the difference did not measurably cost them: both pass the required
# grounding check (`effect:assets_accessed field_contains`, 346 tasks) at parity
# with the image-receiving models -- v4-pro 91.3%, glm-5.2 95.4%, against 91.0-95.1%
# for models that did receive images -- because that check scores whether the needed
# asset was OPENED, which a path is enough to do.
BLIND_TO_IMAGES = ("v4-pro", "glm-5.2")

# The model used to read each task's scored-unit count. It must be one whose
# run covers all 600 tasks; 3.5-flash and 3.1-pro each drop one task's
# check_results and would silently shorten the table.
UNIT_SOURCE = "opus-5"

# Tool families every task in the corpus is granted. Anything beyond these is
# the task's "extra" family, which section 8.4 reports (confounded with
# scenario identity, which the prose states).
BASE_TOOLS = frozenset({"asset", "agents", "file", "message"})


def gold_len(spec):
    """Length of the expert solution, excluding the terminal ``DONE`` sentinel.

    Every one of the 600 specs ends its ``solution_commands`` with exactly one
    ``DONE``, which is a submission marker rather than a step of work. Dropping
    it is the convention ``build_behavior_report.py`` already uses for the
    steps-vs-gold bands, and keeping the two in sync matters: counting the
    sentinel shifts every gold length up by one, which silently renames the
    bands (short becomes <=9 instead of <=8) while leaving their sizes -- 113,
    347, 140 -- identical, so the error would not show up as a count mismatch.
    """
    sc = (spec.get("ground_truth") or {}).get("solution_commands") or []
    return len([c for c in sc if str(c).strip().upper() != "DONE"])


# Candidate difficulty axes the report's opening table rules on, with where to
# read each one. Most of them turn out to be constant across all 600 specs,
# which is the table's point; recording their cardinality here lets the audit
# recompute that verdict instead of trusting the prose.
CANDIDATE_AXES = {
    "difficulty": ("top", "difficulty"),
    "domains": ("top", "domains"),
    "template_id": ("top", "template_id"),
    "weight": ("top", "weight"),
    "visible_constraints": ("top", "visible_constraints"),
    "hidden_constraints": ("top", "hidden_constraints"),
    "decision_requirements": ("top", "decision_requirements"),
    "availability_tier": ("top", "availability_tier"),
    "online_requirement": ("top", "online_requirement"),
    "realism_tags": ("top", "realism_tags"),
    "conflict_taxonomy": ("top", "conflict_taxonomy"),
    "round_type": ("public", "round_type"),
    "required_modalities": ("public", "required_modalities"),
    "allowed_tool_families": ("public", "allowed_tool_families"),
}

_MISSING = object()


def axis_value(spec, where, key):
    """Raw value of a candidate axis, or ``_MISSING`` if the key is absent.

    Absent and null are deliberately distinguished: ``availability_tier`` is
    present-but-null on every task (a live field nobody populated) whereas
    ``conflict_taxonomy`` does not exist in v8 at all, and the report states
    those as different verdicts.
    """
    src = spec if where == "top" else (spec.get("data") or {}).get("public") or {}
    return src.get(key, _MISSING)


def summarize_axes(specs_raw):
    """Cardinality of each candidate axis over the whole corpus.

    Lists are canonicalised by sorting, so ``[a, b]`` and ``[b, a]`` count as
    one value: the report is asking whether the axis *partitions* the corpus,
    and element order is not part of that.
    """
    out = {}
    for name, (where, key) in CANDIDATE_AXES.items():
        distinct, n_present, n_null, lengths = set(), 0, 0, set()
        for spec in specs_raw:
            v = axis_value(spec, where, key)
            if v is _MISSING:
                continue
            n_present += 1
            if v is None:
                n_null += 1
            if isinstance(v, list):
                lengths.add(len(v))
                v = sorted(str(x) for x in v)
            distinct.add(json.dumps(v, sort_keys=True))
        out[name] = {
            "n_present": n_present,
            "n_null": n_null,
            "n_distinct": len(distinct),
            "values": sorted(distinct)[:8],
            "list_lengths": sorted(lengths),
        }
    return out


def load_specs(root):
    """Return ({task_id: {round_type, natural_image, tool_extra, gold_len}}, raw)."""
    base = os.path.join(root, SPEC_TREE)
    if not os.path.isdir(base):
        sys.exit(f"spec tree not found: {base}")
    out, raw = {}, []
    for scen in sorted(os.listdir(base)):
        tdir = os.path.join(base, scen, "tasks")
        if not os.path.isdir(tdir):
            continue
        for tid in sorted(os.listdir(tdir)):
            p = os.path.join(tdir, tid, "specs.json")
            if not os.path.exists(p):
                continue
            with open(p, encoding="utf-8") as fh:
                spec = json.load(fh)
            raw.append(spec)
            pub = (spec.get("data") or {}).get("public") or {}
            fams = set(pub.get("allowed_tool_families") or [])
            extra = sorted(fams - BASE_TOOLS)
            mods = set(pub.get("required_modalities") or [])
            out[spec.get("task_id", tid)] = {
                "scen": pub.get("scenario_slug", scen),
                "round_type": pub.get("round_type"),
                "natural_image": "natural_image" in mods,
                # Reported as a single label; more than one extra family would
                # make the 8.4 grouping ambiguous, so surface it rather than
                # silently picking the first.
                "tool_extra": extra[0] if len(extra) == 1 else
                              ("" if not extra else "+".join(extra)),
                "gold_len": gold_len(spec),
            }
    return out, raw


def load_runs(root, eff):
    """Return {model: {task_id: {score, success, units}}}."""
    base = os.path.join(root, EVAL_TREE.format(eff=eff))
    if not os.path.isdir(base):
        sys.exit(f"eval tree not found: {base}")
    runs = {}
    for name, slug in MODELS:
        # FIXED_TREE models ignore `eff` -- they have one setting each.
        if name in FIXED_TREE:
            d = os.path.join(root, FIXED_TREE[name], "r01")
        else:
            d = os.path.join(base, slug, "r01")
        if not os.path.isdir(d):
            sys.exit(f"missing run dir: {d}")
        per = {}
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".jsonl"):
                continue
            with open(os.path.join(d, fn), encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    checks = (rec.get("metadata") or {}).get("check_results")
                    units = (sum(1 for c in checks if c.get("role") == "required")
                             if checks else None)
                    per[rec["task_id"]] = {
                        "score": rec.get("score"),
                        "success": bool(rec.get("success")),
                        "units": units,
                    }
        runs[name] = per
    return runs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help="corpus root holding eval_results/")
    # Mirrors build_behavior_report.PRIMARY_EFFORT.
    ap.add_argument("--effort", default="high", choices=("low", "medium", "high"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    root = os.path.expanduser(args.root)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = args.out or os.path.join(here, "difficulty_axes.json")

    specs, specs_raw = load_specs(root)
    runs = load_runs(root, args.effort)

    src = runs[UNIT_SOURCE]
    missing_units = [t for t, v in src.items() if v["units"] is None]
    if missing_units:
        sys.exit(f"ABORT: {UNIT_SOURCE} has {len(missing_units)} tasks without "
                 f"check_results; it cannot be the unit source")

    tasks, unmatched = [], []
    for tid, v in sorted(src.items()):
        s = specs.get(tid)
        if s is None:
            unmatched.append(tid)
            continue
        row = {"task_id": tid, "units": v["units"], **s}
        for name, _ in MODELS:
            r = runs[name].get(tid)
            row[name] = None if r is None else [r["score"], r["success"]]
        tasks.append(row)

    if unmatched:
        sys.exit("ABORT: run tasks with no spec:\n  " + "\n  ".join(unmatched[:10]))

    # `effort` describes the ten swept models only. fixed_tree / blind_to_images
    # travel with the data so a consumer cannot read the open-source columns as
    # swept, or read their natural_image slice as a vision measurement.
    payload = {"effort": args.effort, "unit_source": UNIT_SOURCE,
               "fixed_tree": FIXED_TREE, "blind_to_images": list(BLIND_TO_IMAGES),
               "axis_cardinality": summarize_axes(specs_raw), "tasks": tasks}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"), sort_keys=True)
        fh.write("\n")

    total = sum(t["units"] for t in tasks)
    print(f"wrote {out_path}")
    print(f"  {len(tasks)} tasks, {total} scored units")
    for name, _ in MODELS:
        n = sum(1 for t in tasks if t[name] is not None)
        if n != len(tasks):
            print(f"  note: {name} covers {n}/{len(tasks)} tasks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
