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

"""Evaluation result/summary dataclasses + visualizer extracted from examples/train_and_eval.py."""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clasharness.evaluation.llm_retry import anthropic_rejects_sampling_params
from clasharness.utils.episode_memory import is_read_only_action


def _repo_root() -> Path:
  """Directory holding ``clasharness/`` -- the checkout root.

  Walks up from this file looking for ``pyproject.toml``. ``CLASHARNESS_REPO_ROOT``
  overrides it for callers that run the package from outside the checkout.
  """
  override = os.environ.get("CLASHARNESS_REPO_ROOT")
  if override:
    return Path(override)
  for parent in Path(__file__).resolve().parents:
    if (parent / "pyproject.toml").is_file():
      return parent
  return Path(".")


# Terminal stop_reason emitted when the reply-retry budget is exhausted on a
# persistently empty generation (set by SolverAgent.act via the runner). Kept in
# sync with train_eval/agents.py.
GENERATION_STALL_EMPTY_REPLY = "generation_stall_empty_reply"

# Shared prefix every LLM-judge-originated CheckResult carries (effect/file
# ``field_semantic_match`` and LLMChecker). Judge-health aggregation keys off it.
_JUDGE_MSG_PREFIX = "LLM semantic match:"
# Self-consistency vote annotation appended by ``llm_judge.judge_majority`` --
# e.g. "... [self-consistency 2/3 passed]". Scored runs vote (k=3) at
# temperature 0, so a SPLIT vote can no longer be explained away as deliberate
# sampling: it means the backend returned different verdicts for byte-identical
# greedy requests. ``split_vote_rate`` is therefore a provider-nondeterminism
# gauge, and a rise in it invalidates cross-run comparisons.
_VOTE_RE = re.compile(r"\[self-consistency (\d+)/(\d+) passed\]")
# Fraction of judge checks that may fail to run (ok=False) before the run's
# semantic grades are called into question and a loud warning is emitted.
_JUDGE_FAILURE_WARN_RATE = 0.01

# ``llm_judge.resolve_provider`` names for the two Claude judge backends.
_ANTHROPIC_JUDGE_PROVIDERS = frozenset({"anthropic", "vertex-anthropic"})


@dataclass
class JudgeHealth:
  """Aggregate reliability signal for the LLM-judge layer across a whole run.

  Computed from the recorded ``check_results`` rather than re-invoking the judge,
  so it is a free by-product of a completed evaluation. Every counted check is a
  judge-originated semantic check (identified by ``_JUDGE_MSG_PREFIX``); each
  falls into exactly one primary status (infra_failures / fail_closed_empty /
  voted / single_shot), and ``split_votes`` further partitions ``voted``.
  """

  judge_checks: int = 0
  infra_failures: int = 0  # ok=False: the judge could not run at all
  fail_closed_empty: int = 0  # empty transcript/file/criterion -> no judge call
  voted: int = 0  # went through k-way self-consistency voting (the default)
  split_votes: int = 0  # of ``voted``: verdict was NOT unanimous
  single_shot: int = 0  # ran the judge once (explicit k=1, or no annotation)

  @property
  def unanimous_votes(self) -> int:
    return self.voted - self.split_votes

  @property
  def infra_failure_rate(self) -> float:
    return self.infra_failures / self.judge_checks if self.judge_checks else 0.0

  @property
  def split_vote_rate(self) -> float:
    return self.split_votes / self.voted if self.voted else 0.0

  @property
  def degraded(self) -> bool:
    """True when judge-infra failures are frequent enough to distrust grades."""
    return (
        self.judge_checks > 0
        and self.infra_failure_rate > _JUDGE_FAILURE_WARN_RATE
    )


# Prose the checkers emitted when a semantic check failed closed before ever
# reaching the judge. Only consulted for records written before ``judge_invoked``
# existed; new records carry the structured flag and never touch this.
_LEGACY_FAIL_CLOSED_MARKERS = (
    "(empty transcript)",
    "no file ",
    "empty criterion",
)


def _judge_was_invoked(check: dict, message: str) -> bool:
  """Did the grader actually reach the judge for this check?

  Prefers the structured ``judge_invoked`` flag the checkers now emit. Matching
  the message text instead is a latent bug: the strings are human-facing prose,
  so a reworded message silently reclassifies "never asked" as a real semantic
  failure and inflates the apparent judge workload. The text match survives only
  as a fallback for result files written before the field existed, where it is
  the sole available signal.
  """
  invoked = check.get("judge_invoked")
  if invoked is None:
    return not any(m in message for m in _LEGACY_FAIL_CLOSED_MARKERS)
  return bool(invoked)


def _compute_judge_health(results: list[EpisodeResult]) -> JudgeHealth:
  """Aggregate per-check judge telemetry across every episode in a run."""
  health = JudgeHealth()
  for result in results:
    check_results = (result.metadata or {}).get("check_results", [])
    if not isinstance(check_results, list):
      continue
    for check in check_results:
      if not isinstance(check, dict):
        continue
      message = str(check.get("message", ""))
      if not message.startswith(_JUDGE_MSG_PREFIX):
        continue
      health.judge_checks += 1
      # ``ok`` defaults True for records predating the field / non-judge grades.
      if not check.get("ok", True):
        health.infra_failures += 1
        continue
      if not _judge_was_invoked(check, message):
        health.fail_closed_empty += 1
        continue
      vote = _VOTE_RE.search(message)
      if vote:
        health.voted += 1
        passed_votes, total_votes = int(vote.group(1)), int(vote.group(2))
        if passed_votes not in (0, total_votes):
          health.split_votes += 1
      else:
        health.single_shot += 1
  return health


# A run needs more than one task before "do failures cluster across tasks?" is
# even a question, so the report block stays silent below this.
_CLUSTERING_MIN_TASKS = 2


@dataclass
class FailureClustering:
  """Whether a model's check failures concentrate in few tasks or spread thin.

  The headline pass rate is an all-or-nothing gate: a task counts only if EVERY
  one of its ``required`` checks passed. That rate therefore sits far below the
  per-check pass rate, and how far below depends on how failures are
  distributed. Two models failing the same 30% of checks land very differently
  -- spread one failure into each task and almost nothing passes, concentrate
  them in a few hopeless tasks and the rest still clear. This separates the two
  regimes from the recorded ``check_results``, so it is a free by-product of a
  completed evaluation (no re-grading, no re-reading of trajectories).

  Only ``required`` checks are counted: those are exactly the ones
  ``EvaluatorComb`` ands together into ``success``, so the baseline is directly
  comparable to the headline pass rate. ``scored`` (partial credit) and
  ``diagnostic`` (observability) checks never gate success and are excluded.
  Tasks carrying no required check are skipped entirely -- they cannot pass, and
  ``p ** 0 == 1`` would otherwise credit them as a guaranteed pass.

  Stored as sufficient statistics. ``k_histogram`` (per-task gating-check count
  -> number of tasks) is kept whole rather than collapsed to a mean because the
  independence baseline is convex in ``k``; see ``independent_rate``.
  """

  # k -> how many tasks had exactly k required checks.
  k_histogram: dict[int, int] = field(default_factory=dict)
  failed_checks: int = 0  # sum of f_i, failed required checks per task
  observed_all_pass: int = 0  # tasks with f_i == 0
  sum_f_sq: int = 0  # sum of f_i**2, for the observed variance of f

  @property
  def tasks(self) -> int:
    """Tasks carrying at least one required check."""
    return sum(self.k_histogram.values())

  @property
  def gating_checks(self) -> int:
    """Total required checks across those tasks (sum of k_i)."""
    return sum(k * n for k, n in self.k_histogram.items())

  @property
  def per_check_pass_rate(self) -> float:
    """Fraction of individual required checks that passed (``p``)."""
    if not self.gating_checks:
      return 0.0
    return (self.gating_checks - self.failed_checks) / self.gating_checks

  @property
  def mean_checks_per_task(self) -> float:
    return self.gating_checks / self.tasks if self.tasks else 0.0

  @property
  def var_checks_per_task(self) -> float:
    """Population variance of k across tasks."""
    n = self.tasks
    if not n:
      return 0.0
    mean_k = self.mean_checks_per_task
    return sum(c * (k - mean_k) ** 2 for k, c in self.k_histogram.items()) / n

  @property
  def observed_rate(self) -> float:
    """Observed fraction of counted tasks whose every required check passed.

    Close to, but not identical with, the headline ``tgc``: the denominator here
    is only the tasks that recorded at least one required check, so episodes
    that died before grading (provider failures) or carried no gating check are
    absent. Read it against ``independent_rate``, not against the headline.
    """
    return self.observed_all_pass / self.tasks if self.tasks else 0.0

  @property
  def independent_rate(self) -> float:
    """All-pass rate expected if every check failed independently at ``1 - p``.

    Analytic (``mean_i(p ** k_i)``), not sampled -- it agrees with a 2000-draw
    permutation baseline to within 0.04pp while staying deterministic and free.

    It must be averaged over the per-task check counts, NOT evaluated at the
    mean count: ``x -> p ** x`` is convex, so by Jensen ``p ** mean(k)`` is the
    smaller quantity and understates the baseline, inflating ``excess`` by
    roughly 2pp on real runs.
    """
    n = self.tasks
    if not n:
      return 0.0
    p = self.per_check_pass_rate
    return sum(c * (p ** k) for k, c in self.k_histogram.items()) / n

  @property
  def excess(self) -> float:
    """Observed all-pass rate minus the independence baseline.

    Positive means failures CLUSTER: the model concentrates its misses in fewer
    tasks than chance would, so more tasks come out clean than independence
    predicts. Near zero means failures are scattered independently.
    """
    return self.observed_rate - self.independent_rate

  @property
  def overdispersion(self) -> float:
    """Observed variance of per-task failure counts over the independent one.

    ``> 1`` means failures bunch up (some tasks fail nearly everything, others
    nothing); ``~1`` means they are scattered as independent draws predict.

    The denominator must carry the ``k``-heterogeneity term. Under independence
    ``f_i | k_i ~ Binomial(k_i, 1 - p)``, so by the law of total variance
    ``Var(f) = E[k] p (1 - p) + (1 - p)**2 Var(k)``. Dropping the second term
    (i.e. pretending every task has the same number of checks) shrinks the
    denominator and manufactures overdispersion that is really just variation in
    task size.

    Returns 0.0 when the baseline variance vanishes (no failures at all, or a
    single uniform-size task), where the ratio is undefined.
    """
    n = self.tasks
    if not n:
      return 0.0
    p = self.per_check_pass_rate
    var_independent = (
        self.mean_checks_per_task * p * (1.0 - p)
        + (1.0 - p) ** 2 * self.var_checks_per_task
    )
    if var_independent <= 0.0:
      return 0.0
    mean_f = self.failed_checks / n
    var_observed = self.sum_f_sq / n - mean_f**2
    return var_observed / var_independent

  @property
  def available(self) -> bool:
    """Whether the run has enough gating structure for the metric to mean much."""
    return self.tasks >= _CLUSTERING_MIN_TASKS and self.gating_checks > 0


def _compute_failure_clustering(
    results: list[EpisodeResult],
) -> FailureClustering:
  """Aggregate per-task required-check outcomes across every episode in a run."""
  clustering = FailureClustering()
  for result in results:
    check_results = (result.metadata or {}).get("check_results", [])
    if not isinstance(check_results, list):
      continue
    gating = 0
    failed = 0
    for check in check_results:
      if not isinstance(check, dict):
        continue
      # ``role`` defaults to "required" to match CheckResult's own default, so
      # records predating the field keep their historical gating semantics
      # (success = all(passed)) instead of silently dropping out of the metric.
      if str(check.get("role", "required")) != "required":
        continue
      gating += 1
      if not check.get("passed", False):
        failed += 1
    if not gating:
      continue
    clustering.k_histogram[gating] = clustering.k_histogram.get(gating, 0) + 1
    clustering.failed_checks += failed
    clustering.sum_f_sq += failed * failed
    if not failed:
      clustering.observed_all_pass += 1
  return clustering


def _is_done_early(stop_reason: str) -> bool:
  """Whether a stop_reason denotes a premature DONE / missing-required-submission.

  Note this deliberately excludes ``generation_stall_empty_reply`` (a generation
  failure, not an early stop) and ``reached max_steps=`` (step-capped).
  """
  stop_reason = stop_reason or ""
  return (
      "DONE before completing required action" in stop_reason
      or "DONE before executing another command" in stop_reason
      or stop_reason.startswith("missing_required_report:")
      or stop_reason.startswith("missing_required_action_log:")
      or stop_reason.startswith("missing_required_side_effect:")
  )


def _trajectory_issued_state_changing(trajectory: list[dict[str, Any]]) -> bool:
  """True if any recorded step executed a state-changing (non read-only) command.

  Trajectory steps only ever hold *executed* env commands (a DONE reply breaks
  before a step is recorded), so a step counts as state-changing when its action
  is neither read-only nor DONE. Uses the authoritative ``is_read_only_action``
  classifier rather than a private prefix list to avoid drift.
  """
  for step in trajectory or []:
    if not isinstance(step, dict):
      continue
    action = str(step.get("executed_action") or step.get("action") or "").strip()
    if not action or action.upper() == "DONE":
      continue
    if not is_read_only_action(action):
      return True
  return False


def _count_recovered_empty_replies(trajectory: list[dict[str, Any]]) -> int:
  """Count empty API replies that were later recovered within the same episode.

  An empty Gemini/OpenAI reply is logged in ``model_debug`` as ``api: content=''``
  (optionally followed by ``finish_reason=``/``usage=``). Because the retry loop
  re-prompts on empties, any such event in a NON-stalled episode was recovered
  by a subsequent valid command -- i.e. an intermediate issue, not a failure.
  """
  count = 0
  for step in trajectory or []:
    if not isinstance(step, dict):
      continue
    for entry in step.get("model_debug") or []:
      if isinstance(entry, str) and entry.startswith("api: content=''"):
        count += 1
  return count


@dataclass
class EpisodeResult:
  task_id: str
  agent_name: str
  success: bool
  score: float
  steps: int
  duration_s: float
  trajectory: list[dict[str, Any]] = field(default_factory=list)
  final_response: str = ""
  error: str = ""
  error_type: str = ""
  provider_impacted: bool = False
  # True when at least one step used the compact-prompt fallback (a content
  # filter / invalid_prompt error made us drop images + history and resend a
  # terse text-only prompt). This is a degraded outcome: the model acted on
  # far less context, so it must not be read as a clean result.
  compact_fallback: bool = False
  stop_reason: str = ""
  date_anchor_mismatch: bool = False
  mode: str = ""
  metadata: dict[str, Any] = field(default_factory=dict)

  def to_dict(self) -> dict[str, Any]:
    return {
        "task_id": self.task_id,
        "agent": self.agent_name,
        "success": self.success,
        "score": self.score,
        "steps": self.steps,
        "duration_s": round(self.duration_s, 3),
        "trajectory": self.trajectory,
        "final_response": self.final_response,
        "error": self.error,
        "error_type": self.error_type,
        "provider_impacted": self.provider_impacted,
        "compact_fallback": self.compact_fallback,
        "stop_reason": self.stop_reason,
        "date_anchor_mismatch": self.date_anchor_mismatch,
        "mode": self.mode,
        "metadata": self.metadata,
    }

@dataclass
class DomainMetrics:
  passed: int = 0
  total: int = 0
  total_score: float = 0.0

  @property
  def tgc(self) -> float:
    return self.passed / self.total if self.total else 0.0

  @property
  def avg_score(self) -> float:
    return self.total_score / self.total if self.total else 0.0

@dataclass
class EvalSummary:
  agent_name: str
  split: str
  total: int
  passed: int
  total_score: float
  exec_mode: str | None = None
  llm_provider: str | None = None
  llm_model: str | None = None
  llm_history_mode: str | None = None
  # Which backend graded the semantic checks (LLM judge). Recorded separately
  # from the solver's llm_provider/llm_model: the two are frequently different
  # (a small judge scoring a large solver), and reproducibility needs both.
  judge_provider: str | None = None
  judge_model: str | None = None
  provider_failures: int = 0
  provider_impacted_tasks: int = 0
  compact_fallback_tasks: int = 0
  by_domain: dict[str, DomainMetrics] = field(default_factory=dict)
  by_difficulty: dict[int, DomainMetrics] = field(default_factory=dict)
  by_hard_scenario: dict[str, DomainMetrics] = field(default_factory=dict)
  by_hard_ability: dict[str, DomainMetrics] = field(default_factory=dict)
  by_hard_ability_tag: dict[str, DomainMetrics] = field(default_factory=dict)
  results: list[EpisodeResult] = field(default_factory=list)

  @property
  def tgc(self) -> float:
    return self.passed / self.total if self.total else 0.0

  @property
  def avg_score(self) -> float:
    return self.total_score / self.total if self.total else 0.0

  @property
  def completed_tasks(self) -> int:
    return self.total - self.provider_failures

  @property
  def clean_completed_tasks(self) -> int:
    return max(0, self.completed_tasks - self.provider_impacted_tasks)

  @property
  def clean_passed(self) -> int:
    """Passes among clean tasks (excludes provider-impacted / degraded runs).

    ``provider_impacted`` covers the compact-prompt fallback (dropped
    images/history) plus other provider noise, so a task that only "passed"
    in a degraded mode is not counted here.
    """
    return sum(1 for r in self.results if r.success and not r.provider_impacted)

  @property
  def clean_tgc(self) -> float:
    """Accuracy over clean tasks only, so degraded passes don't inflate it."""
    return (
        self.clean_passed / self.clean_completed_tasks
        if self.clean_completed_tasks
        else 0.0
    )

  @property
  def provider_adjusted_tgc(self) -> float:
    return self.passed / self.completed_tasks if self.completed_tasks else 0.0

  @property
  def pass_at_budget(self) -> float:
    return self.tgc

  @property
  def near_miss_tasks(self) -> int:
    return sum(
        1
        for result in self.results
        if not result.success and result.score >= 0.75
    )

  @property
  def near_miss_rate(self) -> float:
    return self.near_miss_tasks / self.total if self.total else 0.0

  @property
  def done_early_tasks(self) -> int:
    return sum(1 for result in self.results if _is_done_early(result.stop_reason))

  @property
  def gave_up_no_action_tasks(self) -> int:
    """Done-early episodes that never executed a state-changing command.

    The "pure inspector" sub-population: looked around, then quit without ever
    attempting a side effect. Split out from ``acted_but_wrong_tasks`` because
    ``done_early_tasks`` alone conflates the two very different failure modes.
    """
    return sum(
        1
        for result in self.results
        if _is_done_early(result.stop_reason)
        and not _trajectory_issued_state_changing(result.trajectory)
    )

  @property
  def acted_but_wrong_tasks(self) -> int:
    """Done-early episodes that DID execute a state-changing command.

    These issued at least one side-effecting command but still failed the
    required check -- the dominant done-early mode. "Acted" uses the broad
    definition (issued a state-changing command), not "produced a tracked
    effect".
    """
    return sum(
        1
        for result in self.results
        if _is_done_early(result.stop_reason)
        and _trajectory_issued_state_changing(result.trajectory)
    )

  @property
  def generation_stall_tasks(self) -> int:
    """Episodes coerced to DONE by exhausting the reply-retry budget on empty
    generation. A generation failure, NOT an agent-chosen early stop -- kept out
    of ``done_early_tasks`` so the two are never conflated."""
    return sum(
        1
        for result in self.results
        if result.stop_reason == GENERATION_STALL_EMPTY_REPLY
    )

  @property
  def recovered_empty_reply_events(self) -> int:
    """Total empty API replies that the retry loop recovered from (telemetry).

    Intermediate-issue signal only: an already-recovered empty is NOT a failure
    cause. Counted only in non-stalled episodes (a terminal stall's empties are
    the failure and are accounted for by ``generation_stall_tasks``)."""
    return sum(
        _count_recovered_empty_replies(result.trajectory)
        for result in self.results
        if result.stop_reason != GENERATION_STALL_EMPTY_REPLY
    )

  @property
  def recovered_empty_reply_tasks(self) -> int:
    """Distinct non-stalled episodes with >=1 retry-recovered empty reply."""
    return sum(
        1
        for result in self.results
        if result.stop_reason != GENERATION_STALL_EMPTY_REPLY
        and _count_recovered_empty_replies(result.trajectory) > 0
    )

  @property
  def step_capped_tasks(self) -> int:
    return sum(
        1
        for result in self.results
        if result.stop_reason.startswith("reached max_steps=")
    )

  @property
  def date_anchor_mismatch_tasks(self) -> int:
    return sum(1 for result in self.results if result.date_anchor_mismatch)

  @property
  def judge_health(self) -> JudgeHealth:
    return _compute_judge_health(self.results)

  @property
  def failure_clustering(self) -> FailureClustering:
    return _compute_failure_clustering(self.results)

  @property
  def score_rank_note(self) -> str:
    return (
        "Compare avg_score first and raw_accuracy second; rank inversions can"
        " happen when one run has more near-miss failures."
    )

  def print_report(self) -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    header = f"Agent : {self.agent_name}   Split : {self.split}"
    if self.exec_mode:
      header += f"   Mode : {self.exec_mode}"
    print(header)
    if self.llm_provider or self.llm_model or self.llm_history_mode:
      llm_line = "LLM   : "
      provider_model = " / ".join(
          part for part in (self.llm_provider, self.llm_model) if part
      )
      if provider_model:
        llm_line += provider_model
      if self.llm_history_mode:
        if provider_model:
          llm_line += f"   History : {self.llm_history_mode}"
        else:
          llm_line += f"History : {self.llm_history_mode}"
      print(llm_line)
    if self.judge_provider or self.judge_model:
      judge_model = " / ".join(
          part for part in (self.judge_provider, self.judge_model) if part
      )
      print(f"Judge : {judge_model}")
      if self.judge_provider in _ANTHROPIC_JUDGE_PROVIDERS and (
          anthropic_rejects_sampling_params(self.judge_model or "")
      ):
        print(
            "  note: this judge takes no temperature, so its votes are sampled"
            " at the model default rather than greedy; grades are less"
            " reproducible and split votes are not only provider noise."
        )
    print(
        f"Tasks : {self.passed}/{self.total} passed  "
        f"pass@budget={self.pass_at_budget*100:.1f}%  "
        f"raw_accuracy={self.tgc*100:.1f}%  avg_score={self.avg_score:.4f}"
    )
    print(
        f"Completed : {self.passed}/{self.completed_tasks}  "
        f"provider_failures={self.provider_failures}  "
        f"provider_impacted_tasks={self.provider_impacted_tasks}  "
        f"clean_completed_tasks={self.clean_completed_tasks}  "
        f"provider_adjusted_accuracy={self.provider_adjusted_tgc*100:.1f}%  "
        f"clean_accuracy={self.clean_tgc*100:.1f}%"
    )
    print(
        f"Diagnostics : near_miss_rate={self.near_miss_rate*100:.1f}%  "
        f"near_miss_tasks={self.near_miss_tasks}  "
        f"done_early_tasks={self.done_early_tasks}  "
        f"step_capped_tasks={self.step_capped_tasks}  "
        f"date_anchor_mismatch_tasks={self.date_anchor_mismatch_tasks}  "
        f"provider_impacted_tasks={self.provider_impacted_tasks}  "
        f"compact_fallback_tasks={self.compact_fallback_tasks}"
    )
    print(
        f"  done_early split: gave_up_no_action={self.gave_up_no_action_tasks}  "
        f"acted_but_wrong={self.acted_but_wrong_tasks}  "
        f"generation_stall={self.generation_stall_tasks}"
    )
    # Intermediate-issue telemetry (NOT a failure cause): empty replies the
    # retry loop already recovered from.
    print(
        "  recovered_empty_replies (telemetry, not failures): "
        f"{self.recovered_empty_reply_events} events across "
        f"{self.recovered_empty_reply_tasks} tasks"
    )
    print(f"Ranking : {self.score_rank_note}")

    jh = self.judge_health
    if jh.judge_checks:
      print(
          f"Judge health : {jh.judge_checks} judge checks  "
          f"infra_failures={jh.infra_failures} "
          f"({jh.infra_failure_rate*100:.1f}%)  "
          f"voted={jh.voted} (split={jh.split_votes}, "
          f"unanimous={jh.unanimous_votes})  "
          f"single_shot={jh.single_shot}  "
          f"fail_closed_empty={jh.fail_closed_empty}"
      )
      if jh.degraded:
        warning = (
            f"WARNING: LLM judge failed to run on {jh.infra_failures} of "
            f"{jh.judge_checks} semantic checks "
            f"({jh.infra_failure_rate*100:.1f}% > "
            f"{_JUDGE_FAILURE_WARN_RATE*100:.0f}%) -- semantic grades for this "
            "run are unreliable; check judge provider/model configuration."
        )
        print(warning)
        print(warning, file=sys.stderr)

    fc = self.failure_clustering
    if fc.available:
      print(
          "Failure clustering : "
          f"per_check_pass={fc.per_check_pass_rate*100:.1f}%  "
          f"all_pass_observed={fc.observed_rate*100:.1f}%  "
          f"all_pass_if_independent={fc.independent_rate*100:.1f}%  "
          f"excess={fc.excess*100:+.1f}pp  "
          f"overdispersion={fc.overdispersion:.2f}  "
          f"(gating_checks={fc.gating_checks} over {fc.tasks} tasks)"
      )

    if self.by_domain:
      print("\nBy domain:")
      for domain, m in sorted(self.by_domain.items()):
        print(
            f"  {domain:<25} {m.passed:>3}/{m.total:<3}  "
            f"TGC={m.tgc*100:.0f}%  avg={m.avg_score:.3f}"
        )

    if self.by_difficulty:
      print("\nBy difficulty:")
      for diff, m in sorted(self.by_difficulty.items()):
        label = {
            1: "L1 single-step",
            2: "L2 multi-step",
            3: "L3 reasoning",
        }.get(diff, f"L{diff}")
        print(
            f"  {label:<20} {m.passed:>3}/{m.total:<3}  "
            f"TGC={m.tgc*100:.0f}%  avg={m.avg_score:.3f}"
        )

    if self.by_hard_scenario:
      print("\nBy hard scenario:")
      for scenario, m in sorted(self.by_hard_scenario.items()):
        print(
            f"  {scenario:<32} {m.passed:>3}/{m.total:<3}  "
            f"TGC={m.tgc*100:.0f}%  avg={m.avg_score:.3f}"
        )

    if self.by_hard_ability:
      print("\nBy hard primary ability:")
      for ability, m in sorted(self.by_hard_ability.items()):
        print(
            f"  {ability:<32} {m.passed:>3}/{m.total:<3}  "
            f"TGC={m.tgc*100:.0f}%  avg={m.avg_score:.3f}"
        )

    if self.by_hard_ability_tag:
      print("\nBy hard ability tag (overlapping):")
      for ability, m in sorted(self.by_hard_ability_tag.items()):
        print(
            f"  {ability:<32} {m.passed:>3}/{m.total:<3}  "
            f"TGC={m.tgc*100:.0f}%  avg={m.avg_score:.3f}"
        )
    print(bar)

def _load_asset_refs(runs: list[dict], out_dir: Path) -> dict:
  """Builds a map of image asset references (relative file paths) for the
  scenarios in the runs.

  Historically this base64-encoded every image into a data: URI and inlined it
  into visualize.html. When many scenarios/seeds are aggregated into a single
  dashboard the payload blew past the filesystem's per-file size cap, raising
  EFBIG ([Errno 27] File too large) and aborting the whole visualizer. Instead
  we now emit a path relative to the dashboard's own directory (out_dir) so the
  browser loads each image lazily from the in-place scenario assets when the
  report is opened. The de-duplicated {blobs, alias} layout is understood by
  resolveAssetBlob() in the HTML template.
  """
  assets_map = {}

  # Find unique scenarios
  scenarios = set()
  for run in runs:
    metadata = run.get("metadata", {})
    gen_id = metadata.get("generator_id") or "conflict_vlm_v1"
    slug = metadata.get("scenario_slug") or run.get("scenario_slug")
    if slug:
      scenarios.add((gen_id, slug))

  # The checkout root first, then the directory above this file in case
  # pyproject.toml is missing and _repo_root() fell back to the cwd.
  scenario_roots = []
  for root in (_repo_root(), Path(__file__).resolve().parents[1]):
    if root not in scenario_roots:
      scenario_roots.append(root)

  try:
    out_dir_abs = Path(out_dir).resolve()
  except Exception:
    out_dir_abs = Path(out_dir)

  def _ref_for(asset_file: Path) -> str:
    """Path to embed in <img src>, relative to out_dir when possible."""
    resolved = asset_file.resolve()
    try:
      return Path(os.path.relpath(resolved, out_dir_abs)).as_posix()
    except Exception:
      return resolved.as_posix()

  for gen_id, slug in scenarios:
    candidates = (root / "clasharness" / "data" / "scenarios" / gen_id / slug
                  for root in scenario_roots)
    scenario_path = next((p for p in candidates if p.is_dir()), None)
    if scenario_path is None:
      continue

    manifest_path = scenario_path / "asset_manifest.json"
    if not manifest_path.is_file():
      continue

    try:
      with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

      assets_list = manifest.get("assets", manifest if isinstance(manifest, list) else [])
      # De-duplicated layout: each unique image is referenced once under "blobs"
      # (keyed by asset_id); "alias" maps the alternative lookup keys (relative
      # path, file name) to the owning asset_id.
      blobs = {}
      alias = {}
      for asset in assets_list:
        asset_id = asset.get("id")
        rel_path = asset.get("path")
        if not asset_id or not rel_path:
          continue

        asset_file = scenario_path / "assets" / rel_path
        if asset_file.is_file():
          blobs[asset_id] = _ref_for(asset_file)
          alias[rel_path] = asset_id
          alias[Path(rel_path).name] = asset_id

      assets_map[slug] = {"blobs": blobs, "alias": alias}
    except Exception as e:
      print(f"Warning: Failed to load assets for {slug}: {e}")

  return assets_map

def _generate_evaluation_visualizer(out_path: Path):
  """Compiles a standalone visualizer HTML dashboard for evaluation trajectories."""
  try:
    print("Auto-generating visualizer HTML dashboard...")
    jsonl_files = sorted(out_path.glob("*.jsonl"))
    if not jsonl_files:
      print("No JSONL files found for visualizer.")
      return

    runs = []
    for file_path in jsonl_files:
      with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
          line = line.strip()
          if not line:
            continue
          try:
            runs.append(json.loads(line))
          except Exception:
            pass

    summary_stats = {}
    summary_file = out_path / "parallel_run.summary"
    if not summary_file.is_file():
      # If parallel summary doesn't exist, check for single-thread summary format
      # e.g., the summary file name matches the out_file name with .summary suffix
      # We look for any file with .summary suffix
      summary_files = list(out_path.glob("*.summary"))
      if summary_files:
        summary_file = summary_files[0]

    if summary_file.is_file():
      try:
        content = summary_file.read_text(encoding="utf-8")
        m_config = re.search(r"Agent\s*:\s*([^\s]+)\s+Split\s*:\s*([^\s]+)\s+Mode\s*:\s*([^\s]+)", content)
        if m_config:
          summary_stats["agent"] = m_config.group(1)
          summary_stats["split"] = m_config.group(2)
          summary_stats["mode"] = m_config.group(3)
        m_llm = re.search(r"LLM\s*:\s*([^\n]+)", content)
        if m_llm:
          summary_stats["model"] = m_llm.group(1).strip()
        m_tasks = re.search(r"Tasks\s*:\s*(\d+)/(\d+)\s+passed\s+pass@budget=([^\s]+)\s+raw_accuracy=([^\s]+)\s+avg_score=([^\s]+)", content)
        if m_tasks:
          summary_stats["passed_tasks"] = int(m_tasks.group(1))
          summary_stats["total_tasks"] = int(m_tasks.group(2))
          summary_stats["pass_rate"] = m_tasks.group(4)
          summary_stats["avg_score"] = float(m_tasks.group(5))
        m_completed = re.search(r"Completed\s*:\s*(\d+)/(\d+)\s+provider_failures=(\d+)\s+provider_impacted_tasks=(\d+)\s+clean_completed_tasks=(\d+)\s+provider_adjusted_accuracy=([^\s]+)", content)
        if m_completed:
          summary_stats["completed_tasks"] = int(m_completed.group(5))
          summary_stats["provider_failures"] = int(m_completed.group(3))
          summary_stats["adjusted_accuracy"] = m_completed.group(6)
        m_diag = re.search(r"Diagnostics\s*:\s*near_miss_rate=([^\s]+)\s+near_miss_tasks=(\d+)", content)
        if m_diag:
          summary_stats["near_miss_rate"] = m_diag.group(1)
          summary_stats["near_miss_tasks"] = int(m_diag.group(2))
      except Exception:
        pass

    if "total_tasks" not in summary_stats:
      summary_stats["total_tasks"] = len(runs)
      passed = sum(1 for r in runs if r.get("success"))
      summary_stats["passed_tasks"] = passed
      summary_stats["pass_rate"] = f"{(passed / len(runs) * 100):.1f}%" if runs else "0.0%"
      summary_stats["avg_score"] = sum(r.get("score", 0.0) for r in runs) / len(runs) if runs else 0.0
      summary_stats["completed_tasks"] = len(runs)
      prov_fail = sum(1 for r in runs if r.get("provider_impacted") or r.get("error_type") == "provider_failure")
      summary_stats["provider_failures"] = prov_fail
      clean_runs = len(runs) - prov_fail
      summary_stats["adjusted_accuracy"] = f"{(passed / clean_runs * 100):.1f}%" if clean_runs > 0 else "0.0%"
      near_miss = sum(1 for r in runs if not r.get("success") and r.get("score", 0.0) > 0.0)
      summary_stats["near_miss_tasks"] = near_miss
      summary_stats["near_miss_rate"] = f"{(near_miss / len(runs) * 100):.1f}%" if runs else "0.0%"

    # Locate template file
    py_file_path = Path(__file__).resolve()
    template_path = py_file_path.parent / "scripts" / "templates" / "visualize_eval_template.html"
    if not template_path.exists():
      template_path = py_file_path.parent.parent / "scripts" / "templates" / "visualize_eval_template.html"
    if not template_path.exists():
      template_path = _repo_root() / "scripts" / "templates" / "visualize_eval_template.html"

    if not template_path.exists():
      print(f"Warning: Visualizer template not found at {template_path}")
      return

    template_content = template_path.read_text(encoding="utf-8")
    # Reference images by path (relative to this dashboard) instead of inlining
    # base64 data URIs, which pushed aggregated dashboards past the filesystem's
    # max file size (EFBIG). Images load lazily from the scenario assets.
    assets_map = _load_asset_refs(runs, out_path)
    payload = {
        "summary": summary_stats,
        "runs": runs,
        "assets_map": assets_map
    }
    payload_str = json.dumps(payload, indent=2, ensure_ascii=False)
    output_content = template_content.replace("/* EVAL_DATA_PLACEHOLDER */", payload_str)

    output_file = out_path / "visualize.html"
    output_file.write_text(output_content, encoding="utf-8")
    print("======================================================================")
    print("SUCCESS: Standalone Evaluation Results Visualizer Compiled!")
    print(f"Open visualizer in browser: file://{output_file}")
    print("======================================================================\n")
  except Exception as e:
    print(f"Warning: Failed to auto-generate visualizer: {e}")

