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

"""Offline trajectory replay + judge-prompt capture.

Phase 0 of the judge-provenance ablation. Re-executes an eval round's RECORDED
actions inside a fresh ``OpenClawEnv`` so the terminal ``env_state`` -- and
therefore the exact ``(content, criterion)`` pair every semantic check would
hand the LLM judge -- can be reconstructed offline. No judge is contacted: the
module-level ``llm_judge.judge`` entry point is swapped for a capture stub, so
this phase costs zero provider calls.

Two things make the replay sound:

  * ``OpenClawEnv`` performs no LLM calls of its own. The multi-app backend and
    the ``agent delegate`` skill are simulations (``skills/impl/agent_skill.py``
    is explicit that a real system "could" invoke a subagent but this one does
    not), so stepping recorded actions is deterministic and free.
  * ``record_step_evaluation`` defaults to False, so ``evaluate()`` runs exactly
    once, at the end. The captured pairs are therefore the FINAL-state pairs --
    the same ones behind the recorded ``metadata.check_results`` -- and not the
    per-step dense-evaluation prefixes.

The fidelity gate compares only DETERMINISTIC checks (everything whose message
is not judge-originated). Semantic verdicts are stubbed here by construction,
so they carry no information about whether the replay reproduced the episode.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

CLASHARNESS_SRC = os.environ.get("CLASHARNESS_SRC", REPO_ROOT)
if CLASHARNESS_SRC not in sys.path:
    sys.path.insert(0, CLASHARNESS_SRC)

# The corpus, fetched by tools/download_data.py. See analysis/corpus.py for the
# same knobs; they are read independently here so a rejudge can point at one
# corpus while the analysis reads another.
DATA_ROOT = os.environ.get(
    "CLASHARNESS_DATA_ROOT", os.path.join(REPO_ROOT, "data"))
EVAL_ROOT = os.environ.get(
    "CLASHARNESS_EVAL_ROOT",
    os.path.join(DATA_ROOT, "eval_results", "data_3_6_flash_v8"))
# ``OpenClawEnv`` takes the DATA ROOT and appends ``tasks/<task_id>`` itself,
# so this is the corpus directory, not the tasks/ subdirectory that
# analysis/corpus.py's TASKS_V8 points at.
DATA_V8 = os.path.join(
    DATA_ROOT, "multiagent_conflict_data_v8_Gemini_3_6_flash")
TASKS_V8 = os.path.join(DATA_V8, "tasks")

# Where the four phases write. Each one appends "rejudge" itself, so this is the
# PARENT of that directory rather than the directory. It defaults under
# DATA_ROOT because a rejudge is derived from a corpus and is the same order of
# size as one -- tens of GB of replayed rounds and verdicts -- and because
# `.gitignore` already excludes `data/`, so a run cannot accidentally stage its
# output. Override with CLASHARNESS_WORK_ROOT to keep it off the corpus volume.
WORK = os.environ.get("CLASHARNESS_WORK_ROOT", DATA_ROOT)

# Judge-originated checks all carry this prefix -- it is the single hook the
# harness itself uses to identify them (see llm_checker._call_llm, which calls
# it "the ONLY way judge-originated checks are identified").
JUDGE_PREFIX = "LLM semantic match:"

# Trajectory terminators recognised by the materializer's replay loop.
STOP_TOKENS = {"done", "exit", "quit"}


# --------------------------------------------------------------------------- #
# The 14-model high-effort panel                                              #
# --------------------------------------------------------------------------- #
# (slug, setting-dir, vendor-dir, model-dir, run). Slugs match analysis/cost.py.
#
# The run column is r01 everywhere except gemini-3.7-flash's high cell, which is
# the corpus's only cell with two complete runs. analysis/corpus.py:155 selects
# r02 for the published table, so this panel must select it too or the recorded
# baseline will not reproduce tab:main. That choice matters here more than
# anywhere else in the paper: corpus.py:131-134 records that r01 was graded by
# gemini-3.6-flash while **r02 was graded by gemini-3.7-flash, the model under
# test**. The published gemini-3.7-flash row is therefore self-graded, and it is
# the one row the corpus-standard judge did not score.
PANEL: list[tuple[str, str, str, str, str]] = [
    ("opus_5",     "aligned_setting_high",     "Claude",   "opus_5",    "r01"),
    ("5_6_sol",    "aligned_setting_high",     "OpenAI",   "5_6_sol",   "r01"),
    ("opus_4_8",   "aligned_setting_high",     "Claude",   "opus_4_8",  "r01"),
    ("kimi_k3",    "aligned_setting_high",     "Moonshot", "kimi_k3",   "r01"),
    ("sonnet_5",   "aligned_setting_high",     "Claude",   "sonnet_5",  "r01"),
    ("3_7_flash",  "aligned_setting_high",     "Gemini",   "3_7_flash", "r02"),
    ("5_6_terra",  "aligned_setting_high",     "OpenAI",   "5_6_terra", "r01"),
    ("5_6_luna",   "aligned_setting_high",     "OpenAI",   "5_6_luna",  "r01"),
    ("v4_pro",     "aligned_setting_high_ocr", "DeepSeek", "v4_pro",    "r01"),
    ("3_6_flash",  "aligned_setting_high",     "Gemini",   "3_6_flash", "r01"),
    ("glm_5_2",    "aligned_setting_high_ocr", "ZhipuAI",  "glm_5_2",   "r01"),
    ("3_1_pro",    "aligned_setting_high",     "Gemini",   "3_1_pro",   "r01"),
    ("dsv4_flash", "aligned_setting_high_ocr", "DeepSeek",
     "dsv4_flash_v8_high_ocr", "r01"),
    ("3_5_flash",  "aligned_setting_high",     "Gemini",   "3_5_flash", "r01"),
]

# Auxiliary cells: replayed and re-judged, but NOT part of the 14-model panel.
#
# gemini-3.7-flash r01 is the discarded twin of the published r02. corpus.py
# states the r01 -> r02 delta (18.67% -> 20.33%) "confounds rollout noise with a
# judge change and cannot be attributed to either", because the two runs differ
# in BOTH the policy rollout and the grader. Re-judging both runs under one
# common judge removes the grader term, which is what makes the decomposition
# possible for the first time.
AUX: list[tuple[str, str, str, str, str]] = [
    ("3_7_flash_r01", "aligned_setting_high", "Gemini", "3_7_flash", "r01"),
]

ALL_CELLS = PANEL + AUX

# The ten models billed through a hosted API. Used to reproduce the paper's
# 19,419-verdict denominator as a cross-check on the replay's own accounting.
API_TEN = {
    "opus_5", "opus_4_8", "sonnet_5", "3_1_pro", "3_5_flash", "3_6_flash",
    "3_7_flash", "5_6_luna", "5_6_sol", "5_6_terra",
}


def model_dir(slug: str) -> str:
    for s, setting, vendor, mdir, run in ALL_CELLS:
        if s == slug:
            return os.path.join(EVAL_ROOT, setting, vendor, mdir, run)
    raise KeyError(slug)


def iter_records(slug: str):
    """Yield every recorded round for ``slug`` as (package, index, record)."""
    root = model_dir(slug)
    for name in sorted(os.listdir(root)):
        if not name.endswith(".jsonl"):
            continue
        path = os.path.join(root, name)
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                yield name[:-6], i, json.loads(line)


# --------------------------------------------------------------------------- #
# Judge capture                                                               #
# --------------------------------------------------------------------------- #

@dataclass
class Capture:
    """Collects every (content, criterion) the judge would have been asked."""

    pairs: list[dict[str, Any]] = field(default_factory=list)
    # Index of the check currently being evaluated, maintained by
    # ``bind_check_indices``. Stamped on every captured pair so the pair is tied
    # to its check by construction, instead of assuming the evaluator visits
    # checks in the same order it emits results.
    current: int = -1

    def clear(self) -> None:
        self.pairs.clear()
        self.current = -1

    def bind_check_indices(self, evaluator) -> None:
        """Wrap each sub-evaluator so ``current`` tracks the check being run."""
        for index, sub in enumerate(evaluator.evaluators):
            original = sub.evaluate

            def wrapped(env_state, task_data, _i=index, _f=original):
                self.current = _i
                try:
                    return _f(env_state, task_data)
                finally:
                    self.current = -1

            sub.evaluate = wrapped  # type: ignore[method-assign]

    def install(self) -> None:
        """Swap ``llm_judge.judge`` for a recording stub.

        ``judge_majority`` resolves ``judge`` through the module global, and the
        three checkers import their entry point at call time, so patching this
        one attribute intercepts every path. The stub returns a well-formed
        ``ok=True`` verdict: an ``ok=False`` would be re-tried by
        ``judge_majority`` and would also change how the harness classifies the
        check, whereas a clean deterministic ``passed=False`` leaves the
        surrounding control flow on its normal path.
        """
        from clasharness.evaluation import llm_judge

        sink = self.pairs

        def _stub(content, criterion, *, model=None, use_cache=True,
                  temperature=0.0, cache_salt=""):
            # Only the first vote of a k=3 majority is recorded: the other two
            # are the same pair with a different cache salt, and Phase 1 re-runs
            # the vote itself. Recording all three would triple the dump for no
            # information.
            if cache_salt in ("", "vote:0"):
                sink.append({"check_index": self.current, "criterion": criterion,
                             "content": content, "model": model})
            return llm_judge.JudgeResult(
                passed=False, score=0.0, reason="[replay stub]", ok=True,
                error="",
            )

        llm_judge.judge = _stub  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Replay                                                                      #
# --------------------------------------------------------------------------- #

def replay_round(record: dict, capture: Capture) -> dict[str, Any]:
    """Replay one recorded round and return its reconstructed evaluation.

    Returns a dict with ``details`` (one entry per check, in evaluator order,
    aligned 1:1 with the recorded ``metadata.check_results``) and ``pairs``
    (the judge prompts captured during the FINAL evaluate()).
    """
    from clasharness.core.environment import OpenClawEnv

    task_id = record["task_id"]
    steps = record.get("trajectory") or []
    actions = [s.get("executed_action") or s.get("action") or "" for s in steps]

    out: dict[str, Any] = {
        "task_id": task_id, "status": "ok", "error": None,
        "details": [], "pairs": [], "n_actions": 0,
    }
    try:
        max_steps = max(50, len(actions) + 10)
        with OpenClawEnv(
            task_id=task_id,
            task_data_dir=DATA_V8,
            backend="multi",
            max_steps=max_steps,
        ) as env:
            env.reset()
            n = 0
            for cmd in actions:
                if not cmd or cmd.strip().lower() in STOP_TOKENS:
                    continue
                env.step(cmd)
                n += 1
            out["n_actions"] = n
            # Only the terminal evaluation is of interest; anything the steps
            # above captured is per-step noise (record_step_evaluation is off by
            # default, but clear defensively so the dump is unambiguous).
            capture.clear()
            if env._evaluator is not None:
                capture.bind_check_indices(env._evaluator)
            result = env.evaluate()
        out["details"] = [
            {"name": c.name, "passed": bool(c.passed), "score": float(c.score),
             "weight": float(c.weight), "role": c.role,
             "message": (c.message or "")[:400], "ok": bool(c.ok)}
            for c in result.details
        ]
        out["pairs"] = list(capture.pairs)
    except Exception as exc:  # noqa: BLE001 - one bad round must not stop the sweep
        out["status"] = "error"
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        capture.clear()
    return out


def is_semantic(message: str) -> bool:
    return (message or "").startswith(JUDGE_PREFIX)
