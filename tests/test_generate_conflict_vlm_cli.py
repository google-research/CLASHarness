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

"""Regression tests for ``scripts/generate_conflict_vlm_scenarios.py``.

These cover the CLI-level safety and dataset-layout guarantees of the
conflict-VLM generator. None of this code is exercised by the scenario/task
tests, and none of the data it writes is in version control, so a regression
here is unrecoverable rather than merely annoying.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.generate_conflict_vlm_scenarios as gen
from clasharness.evaluation.results import CheckResult, EvaluationResult
from clasharness.scenario.conflict_vlm_materializer import (
    lint_assets_accessed_satisfiability as lint_assets,
)


# ---- _resolve_under: tilde / relative path routing ----
#
# A shell does NOT expand ``~`` after ``=``, so ``--output_data_dir=~/Data/x``
# arrives as the literal string ``~/Data/x``, which is a RELATIVE path. Writes
# then landed in ``<cwd>/~/Data/x`` -- a real directory named ``~`` -- while the
# evaluator, given the same path as a separate argv word, read ``$HOME/Data/x``.
# The two halves of the pipeline silently addressed different trees.


class TestResolveUnder:
    def test_expands_leading_tilde(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        resolved = gen._resolve_under(Path("/some/base"), "~/Data/foo")
        assert resolved == (tmp_path / "Data/foo").resolve()
        assert "~" not in str(resolved)

    def test_relative_path_resolves_under_base(self, tmp_path):
        base = tmp_path / "base"
        base.mkdir()
        assert gen._resolve_under(base, "out/data") == (base / "out/data").resolve()

    def test_absolute_path_is_preserved(self, tmp_path):
        target = tmp_path / "abs"
        assert gen._resolve_under(Path("/other/base"), str(target)) == target.resolve()

    def test_result_is_always_absolute(self, tmp_path):
        for raw in ("~/x", "x", str(tmp_path / "x")):
            assert gen._resolve_under(tmp_path, raw).is_absolute()

    def test_accepts_path_objects(self, tmp_path):
        assert gen._resolve_under(tmp_path, Path("rel")) == (tmp_path / "rel").resolve()


# ---- _guarded_rmtree: --clean deletes, but never silently ----
#
# ``--clean`` IS the request to delete, so requiring a second confirmation flag
# only trains people to always pass it. What must not regress is the accounting:
# an unconditional silent rmtree of a multi-GB tree left no record of what a
# mistyped ``--output_data_dir`` had just destroyed, and none of this data is in
# version control. The count and size must therefore be logged BEFORE the
# delete, so the run log answers "what did I just lose?".


class TestGuardedRmtree:
    def _populate(self, root: Path, n: int = 3) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        (root / "nested").mkdir(exist_ok=True)
        for i in range(n):
            (root / "nested" / f"f{i}.bin").write_bytes(b"x" * 1024)
        return root

    def test_deletes_non_empty_tree(self, tmp_path):
        """--clean is itself the explicit request; no second flag is required."""
        target = self._populate(tmp_path / "data")
        gen._guarded_rmtree(target, "output data dir")
        assert not target.exists()

    def test_takes_no_force_argument(self):
        """The confirmation flag is gone -- callers must not resurrect it."""
        import inspect

        params = inspect.signature(gen._guarded_rmtree).parameters
        assert list(params) == ["path", "label"]

    def test_reports_path_count_and_size_before_deleting(self, tmp_path, capsys):
        """A mistyped path must be reconstructable from the run log afterwards."""
        target = self._populate(tmp_path / "data")
        gen._guarded_rmtree(target, "output data dir")

        out = capsys.readouterr().out
        assert str(target) in out  # which tree
        assert "3 files" in out  # how much was in it
        assert "KB" in out  # and how big
        assert "output data dir" in out  # which of the two trees

    def test_size_is_logged_even_though_delete_follows(self, tmp_path, capsys):
        """The accounting line must precede the deletion line, not replace it."""
        target = self._populate(tmp_path / "data")
        gen._guarded_rmtree(target, "output data dir")

        out = capsys.readouterr().out
        assert out.index("holds 3 files") < out.index("deleting")

    def test_empty_tree_is_removed(self, tmp_path):
        target = tmp_path / "empty"
        target.mkdir()
        gen._guarded_rmtree(target, "output data dir")
        assert not target.exists()

    def test_missing_tree_is_a_noop(self, tmp_path):
        gen._guarded_rmtree(tmp_path / "nope", "output data dir")


# ---- --clean + --resume: the guard that actually prevents the expensive loss ----


class TestCleanResumeMutualExclusion:
    """``--clean`` deletes precisely the tree ``--resume`` exists to continue from.

    With the confirmation flag removed, this is the ONLY thing standing between a
    stale copy-pasted command line and a full-price regeneration of a spec/asset
    tree that had already been generated. It must fire before anything is
    deleted.
    """

    def _run_main(self, monkeypatch, argv):
        import sys

        monkeypatch.setattr(sys, "argv", ["generate_conflict_vlm_scenarios.py"] + argv)
        # Keep the judge-env sync (and its network-facing model defaults) out of
        # a pure argument-validation test.
        import clasharness.evaluation.llm_judge as judge

        monkeypatch.setattr(judge, "sync_judge_env", lambda *a, **k: None)
        return gen.main()

    def test_clean_with_resume_is_rejected(self, monkeypatch, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "keep.bin").write_bytes(b"x" * 512)

        with pytest.raises(SystemExit) as exc:
            self._run_main(
                monkeypatch,
                ["--clean", "--resume", f"--output_data_dir={data_dir}"],
            )

        msg = str(exc.value)
        assert "--clean" in msg and "--resume" in msg
        assert "mutually exclusive" in msg
        # Must fail BEFORE deleting anything.
        assert (data_dir / "keep.bin").exists()

    def test_clean_alone_does_not_trip_the_guard(self, monkeypatch):
        """The guard keys on the PAIR -- --clean by itself is the normal path."""
        import sys

        monkeypatch.setattr(
            sys, "argv", ["generate_conflict_vlm_scenarios.py", "--clean"]
        )
        parsed = gen.parse_args()
        assert parsed.clean is True
        assert parsed.resume is False

    def test_force_flag_no_longer_exists(self, monkeypatch):
        """A leftover --force in a copy-pasted command must fail loudly, not be ignored."""
        import sys

        monkeypatch.setattr(
            sys, "argv", ["generate_conflict_vlm_scenarios.py", "--clean", "--force"]
        )
        with pytest.raises(SystemExit):
            gen.parse_args()


class TestTreeStatsAndHumanBytes:
    def test_tree_stats_counts_files_and_bytes(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "x.bin").write_bytes(b"y" * 100)
        (tmp_path / "b.bin").write_bytes(b"z" * 50)
        assert gen._tree_stats(tmp_path) == (2, 150)

    def test_tree_stats_on_empty_dir(self, tmp_path):
        assert gen._tree_stats(tmp_path) == (0, 0)

    @pytest.mark.parametrize(
        "n,expected",
        [(0, "0 B"), (512, "512 B"), (2048, "2.0 KB"), (5 * 1024**2, "5.0 MB")],
    )
    def test_human_bytes(self, n, expected):
        assert gen._human_bytes(n) == expected


# ---- dataset splits ----
#
# train/dev/test used to be four byte-identical copies of the full id list, so
# every "held-out" evaluation was scored on data the split called training data.
#
# Task ids are ``<seed>_v<NN>_r<NN>``. ``_rNN`` is a ROUND of one cumulative
# episode, not a replicate: on the shipped 600-task set an episode's
# visible_asset_ids grow 4 -> 13 -> 21 across r01/r05/r10, so r10 contains r01's
# world. Rounds of one episode must never straddle a split.

_IDS = [
    f"conflict_vlm_seed{s:02d}_v{v:02d}_r{r:02d}"
    for s in range(1, 11)
    for v in (1, 2, 3)
    for r in range(1, 11)
]
_RATIOS = [0.7, 0.15, 0.15]
_SALT = "test-salt"


def _assign(ids=None, ratios=None, group_by="seed", salt=_SALT):
    # Note: an explicit ``ids=[]`` must stay empty, so no ``or`` defaulting here.
    return gen._assign_splits(
        _IDS if ids is None else ids,
        _RATIOS if ratios is None else ratios,
        group_by,
        salt,
    )


class TestSplitGroupKey:
    def test_seed_groups_all_versions_and_rounds(self):
        key = gen._split_group_key("conflict_vlm_pay_dispute_v03_r08", "seed")
        assert key == "conflict_vlm_pay_dispute"

    def test_version_groups_only_rounds(self):
        key = gen._split_group_key("conflict_vlm_pay_dispute_v03_r08", "version")
        assert key == "conflict_vlm_pay_dispute_v03"

    def test_task_grouping_is_the_id_itself(self):
        tid = "conflict_vlm_pay_dispute_v03_r08"
        assert gen._split_group_key(tid, "task") == tid

    def test_unparseable_id_stays_atomic(self):
        # Must not collapse every malformed id into one shared bucket.
        a = gen._split_group_key("weird-id-a", "seed")
        b = gen._split_group_key("weird-id-b", "seed")
        assert a == "weird-id-a" and b == "weird-id-b"

    def test_seed_name_containing_v_and_r_is_not_truncated(self):
        assert (
            gen._split_group_key("conflict_vlm_v2_rollout_v01_r02", "seed")
            == "conflict_vlm_v2_rollout"
        )


class TestAssignSplits:
    def test_splits_are_disjoint_and_cover_everything(self):
        splits, _, _ = _assign()
        covered = [t for s in splits.values() for t in s]
        assert sorted(covered) == sorted(_IDS)
        assert len(covered) == len(set(covered))  # no id in two splits

    def test_no_seed_spans_two_splits(self):
        splits, _, _ = _assign(group_by="seed")
        owner: dict[str, str] = {}
        for name, ids in splits.items():
            for tid in ids:
                key = gen._split_group_key(tid, "seed")
                assert owner.setdefault(key, name) == name

    def test_no_episode_spans_two_splits_under_version_grouping(self):
        splits, _, _ = _assign(group_by="version")
        owner: dict[str, str] = {}
        for name, ids in splits.items():
            for tid in ids:
                key = gen._split_group_key(tid, "version")
                assert owner.setdefault(key, name) == name

    def test_splits_are_not_identical_copies(self):
        # The exact bug being fixed: all four files held the same 600 ids.
        splits, _, _ = _assign()
        assert splits["train"] != splits["dev"] != splits["test"]
        assert not set(splits["train"]) & set(splits["test"])
        assert not set(splits["train"]) & set(splits["dev"])
        assert not set(splits["dev"]) & set(splits["test"])

    def test_group_counts_honour_ratios(self):
        _, counts, n_groups = _assign(group_by="seed")
        assert n_groups == 10
        assert counts == {"train": 7, "dev": 2, "test": 1}
        assert sum(counts.values()) == n_groups

    def test_counts_sum_exactly_under_awkward_ratios(self):
        # Largest-remainder must not drop or duplicate a group.
        for ratios in ([1, 1, 1], [0.8, 0.1, 0.1], [0.34, 0.33, 0.33]):
            splits, counts, n = _assign(ratios=gen._parse_ratios(
                ",".join(str(x) for x in ratios)))
            assert sum(counts.values()) == n
            assert sum(len(v) for v in splits.values()) == len(_IDS)

    def test_is_deterministic(self):
        assert _assign()[0] == _assign()[0]

    def test_is_independent_of_input_order(self):
        assert _assign()[0] == _assign(ids=list(reversed(_IDS)))[0]

    def test_salt_reshuffles_the_split(self):
        assert _assign(salt="a")[0] != _assign(salt="b")[0]

    def test_zero_ratio_split_is_empty(self):
        splits, counts, _ = _assign(ratios=gen._parse_ratios("1,0,0"))
        assert counts["dev"] == 0 and counts["test"] == 0
        assert splits["dev"] == [] and splits["test"] == []
        assert len(splits["train"]) == len(_IDS)

    def test_empty_input(self):
        splits, counts, n = _assign(ids=[])
        assert n == 0
        assert all(v == [] for v in splits.values())
        assert sum(counts.values()) == 0


class TestParseRatios:
    def test_normalises_to_one(self):
        assert gen._parse_ratios("70,15,15") == pytest.approx([0.7, 0.15, 0.15])

    def test_accepts_fractions(self):
        assert gen._parse_ratios("0.7,0.15,0.15") == pytest.approx([0.7, 0.15, 0.15])

    @pytest.mark.parametrize("bad", ["0.8,0.2", "1,1,1,1", "1"])
    def test_rejects_wrong_arity(self, bad):
        with pytest.raises(Exception, match="exactly 3"):
            gen._parse_ratios(bad)

    def test_rejects_negative(self):
        with pytest.raises(Exception, match="non-negative"):
            gen._parse_ratios("-1,1,1")

    def test_rejects_all_zero(self):
        with pytest.raises(Exception, match="more than 0"):
            gen._parse_ratios("0,0,0")


class TestSplitDefaults:
    def test_cli_defaults_are_grouped_and_non_trivial(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["generate_conflict_vlm_scenarios.py"])
        args = gen.parse_args()
        # Default must isolate whole scenario families, and must not be the
        # leaky "task" grouping.
        assert args.split_group_by == "seed"
        assert args.split_ratios == pytest.approx([0.7, 0.15, 0.15])
        assert sum(args.split_ratios) == pytest.approx(1.0)

    def test_cli_rejects_unknown_group_by(self, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["generate_conflict_vlm_scenarios.py", "--split-group-by", "bogus"],
        )
        with pytest.raises(SystemExit):
            gen.parse_args()

    def test_task_grouping_is_available_but_leaks_rounds(self):
        # Documented escape hatch: grouping by task puts rounds of one episode
        # in different splits. Asserted so the hazard stays visible.
        splits, _, _ = _assign(group_by="task")
        episodes = {}
        for name, ids in splits.items():
            for tid in ids:
                episodes.setdefault(
                    gen._split_group_key(tid, "version"), set()).add(name)
        assert any(len(v) > 1 for v in episodes.values())


# ---- lint_assets_accessed_satisfiability: the generation-time hard gate ----
#
# This lint is a HARD gate: ``generate_conflict_vlm_scenarios.py`` raises and
# discards the whole scenario version when it reports a problem. It had zero
# test coverage, which cuts both ways -- a false positive silently burns
# generation budget on versions that were fine, and a false negative ships a
# required check no agent can ever satisfy, capping the achievable score for
# every model with no visible symptom. It also has to stay bug-for-bug aligned
# with ``effect_checker``'s matching rules; a drift in either direction is
# exactly what these tests pin.


def _asset_task(checks, visible=(), manifest_ids=()):
    return {
        "task_id": "t1",
        "data": {
            "private": {
                "visible_asset_ids": list(visible),
                "initial_state_overrides": {
                    "asset_manifest.json": {
                        "assets": [{"id": i} for i in manifest_ids]
                    }
                },
            }
        },
        "ground_truth": {"evaluation_checks": checks},
    }


def _aa_check(value, role="required"):
    return {
        "type": "effect",
        "effect_type": "assets_accessed",
        "condition": "field_contains",
        "expected": {"field": "id", "value": value},
        "role": role,
    }


class TestLintAssetsAccessedSatisfiability:
    def test_exact_visible_asset_is_clean(self):
        task = _asset_task([_aa_check("chart_a_20241024_081523")],
                           visible=["chart_a_20241024_081523"])
        assert lint_assets(task) == []

    def test_manifest_only_asset_is_also_visible(self):
        """The manifest and visible_asset_ids are BOTH sources of reachability.

        Dropping either one turns a satisfiable check into a spurious abort.
        """
        task = _asset_task([_aa_check("chart_a")], manifest_ids=["chart_a"])
        assert lint_assets(task) == []

    def test_unreachable_asset_is_flagged(self):
        task = _asset_task([_aa_check("chart_ghost")], visible=["chart_a"])
        problems = lint_assets(task)
        assert len(problems) == 1
        assert "chart_ghost" in problems[0]
        assert "never pass" in problems[0]

    def test_unambiguous_normalized_base_is_clean(self):
        """Exactly one visible asset under the base -> the fallback can reach it."""
        task = _asset_task([_aa_check("chart_a")],
                           visible=["chart_a_20241024_081523"])
        assert lint_assets(task) == []

    def test_ambiguous_normalized_base_is_flagged(self):
        """Two timestamped twins: the grader refuses to guess, so nothing passes.

        This is the temporal_staleness shape -- a stale snapshot kept visible
        beside the decisive one.
        """
        task = _asset_task(
            [_aa_check("chart_a")],
            visible=["chart_a_20241024_081523", "chart_a_20241024_083510"],
        )
        problems = lint_assets(task)
        assert len(problems) == 1
        assert "ambiguous" in problems[0]

    def test_diagnostic_checks_are_not_gated(self):
        """A diagnostic check cannot cap the score, so it must not abort a run."""
        task = _asset_task([_aa_check("chart_ghost", role="diagnostic")],
                           visible=["chart_a"])
        assert lint_assets(task) == []

    def test_other_effect_types_are_ignored(self):
        task = _asset_task([{
            "type": "effect",
            "effect_type": "messages_sent",
            "condition": "field_contains",
            "expected": {"field": "message", "value": "anything"},
            "role": "required",
        }])
        assert lint_assets(task) == []

    def test_empty_needle_is_ignored(self):
        assert lint_assets(_asset_task([_aa_check("")], visible=["chart_a"])) == []

    def test_reports_every_offending_check_not_just_the_first(self):
        task = _asset_task(
            [_aa_check("ghost_one"), _aa_check("ghost_two")], visible=["chart_a"]
        )
        problems = lint_assets(task)
        assert len(problems) == 2
        assert "ghost_one" in problems[0] and "ghost_two" in problems[1]

    def test_matching_is_case_insensitive(self):
        """Ids arrive lowercased from the grader; a case-only diff is not a defect."""
        task = _asset_task([_aa_check("Chart_A")], visible=["chart_a"])
        assert lint_assets(task) == []

    def test_task_with_no_checks_is_clean(self):
        assert lint_assets(_asset_task([])) == []


# ---- audit_zero_action_discrimination: the free-points gate ----
#
# A REQUIRED check that already passes before the agent does anything is a free
# point: it lifts every model's score by the same constant and discriminates
# nothing. The satisfiability lint cannot catch these -- it asks "can this check
# EVER pass?", which is the opposite failure mode. On the shipped 600-task set
# this audit found 4 vacuous checks, all ``file_check``s asserting content that
# ``initial_state_overrides`` had already seeded into the file.


def _check(name, passed, *, role="required", ok=True, message=""):
    return CheckResult(name=name, passed=passed, role=role, ok=ok, message=message)


def _eval_result(success, checks):
    return EvaluationResult(success=success, score=0.0, details=list(checks))


class _FakeEnv:
    """Stands in for CLASHarnessEnv and snapshots the env visible to the grader."""

    def __init__(self, result, seen_env, specs=None):
        self._result = result
        self._seen_env = seen_env
        self.reset_called = False
        self.exited = False
        # Mirrors CLASHarnessEnv._task: the audit reads the authoring specs off it
        # to tell a negative assertion from a genuinely vacuous check.
        self._task = SimpleNamespace(
            ground_truth=SimpleNamespace(evaluation_checks=list(specs or []))
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.exited = True
        return False

    def reset(self):
        self.reset_called = True

    def evaluate(self):
        self._seen_env.append(
            {k: os.environ.get(k) for k in gen._ZERO_ACTION_JUDGE_ENV}
        )
        return self._result


def _install_fake_env(monkeypatch, per_task, specs_per_task=None):
    """Route CLASHarnessEnv to a fake. ``per_task`` maps task_id -> result | exc.

    ``specs_per_task`` optionally maps task_id -> the authoring
    ``evaluation_checks`` list, positionally aligned with the result's details.
    Omitting it leaves the list empty, which is what a caller that does not care
    about the negative-assertion exemption wants: no check is exempt.
    """
    seen_env: list[dict] = []
    built: list[dict] = []

    def factory(**kwargs):
        tid = kwargs["task_id"]
        outcome = per_task[tid]
        if isinstance(outcome, BaseException):
            raise outcome
        env = _FakeEnv(outcome, seen_env, (specs_per_task or {}).get(tid))
        built.append({**kwargs, "env": env})
        return env

    monkeypatch.setattr("clasharness.core.environment.CLASHarnessEnv", factory)
    return seen_env, built


class TestAuditZeroActionDiscrimination:
    def test_required_check_passing_at_zero_actions_is_vacuous(self, monkeypatch):
        _install_fake_env(
            monkeypatch, {"t1": _eval_result(False, [_check("free_point", True)])}
        )
        vacuous, non_disc, unaudited, n_ok = gen.audit_zero_action_discrimination(
            ["t1"], "/data"
        )
        assert [v[:2] for v in vacuous] == [("t1", "free_point")]
        assert non_disc == [] and unaudited == 0 and n_ok == 1

    def test_failing_required_check_is_clean(self, monkeypatch):
        _install_fake_env(
            monkeypatch, {"t1": _eval_result(False, [_check("earns_it", False)])}
        )
        vacuous, non_disc, unaudited, n_ok = gen.audit_zero_action_discrimination(
            ["t1"], "/data"
        )
        assert vacuous == [] and non_disc == [] and unaudited == 0 and n_ok == 1

    def test_diagnostic_check_passing_is_not_vacuous(self, monkeypatch):
        """Diagnostics gate nothing, so a free diagnostic costs no discrimination."""
        _install_fake_env(
            monkeypatch,
            {"t1": _eval_result(False, [_check("obs", True, role="diagnostic")])},
        )
        vacuous, _, _, _ = gen.audit_zero_action_discrimination(["t1"], "/data")
        assert vacuous == []

    def test_scored_check_passing_is_not_vacuous(self, monkeypatch):
        _install_fake_env(
            monkeypatch,
            {"t1": _eval_result(False, [_check("partial", True, role="scored")])},
        )
        vacuous, _, _, _ = gen.audit_zero_action_discrimination(["t1"], "/data")
        assert vacuous == []

    def test_whole_task_succeeding_at_zero_actions_is_non_discriminating(
        self, monkeypatch
    ):
        _install_fake_env(
            monkeypatch, {"t1": _eval_result(True, [_check("free_point", True)])}
        )
        _, non_disc, _, _ = gen.audit_zero_action_discrimination(["t1"], "/data")
        assert non_disc == ["t1"]

    def test_ungradable_required_check_is_unaudited_not_clean(self, monkeypatch):
        """``ok=False`` means the verdict is UNKNOWN, not that the check is fine."""
        _install_fake_env(
            monkeypatch,
            {"t1": _eval_result(False, [_check("sem", False, ok=False)])},
        )
        vacuous, _, unaudited, _ = gen.audit_zero_action_discrimination(
            ["t1"], "/data"
        )
        assert unaudited == 1 and vacuous == []

    def test_ungradable_check_reporting_passed_is_still_unaudited(self, monkeypatch):
        """``ok`` is tested before ``passed``: a dead grader cannot prove a pass."""
        _install_fake_env(
            monkeypatch,
            {"t1": _eval_result(False, [_check("sem", True, ok=False)])},
        )
        vacuous, _, unaudited, _ = gen.audit_zero_action_discrimination(
            ["t1"], "/data"
        )
        assert unaudited == 1 and vacuous == []

    # ---- negative assertions ----
    #
    # ``not_exists`` / ``not_contains`` say "do not break this". They are MEANT
    # to hold before the agent acts; their discriminating power is whether an
    # acting agent violates them. The compile-time demotion pass
    # (conflict_vlm_materializer._demote_vacuous_checks) exempts them, so this
    # gate must too -- otherwise the materializer ships a check the gate then
    # fails the whole run over.

    def test_negative_assertion_passing_at_zero_actions_is_not_vacuous(
        self, monkeypatch
    ):
        for condition in ("not_exists", "not_contains"):
            _install_fake_env(
                monkeypatch,
                {"t1": _eval_result(False, [_check("keeps_it", True)])},
                {"t1": [{"condition": condition}]},
            )
            vacuous, _, _, n_ok = gen.audit_zero_action_discrimination(
                ["t1"], "/data"
            )
            assert vacuous == [], condition
            assert n_ok == 1, condition

    def test_positive_assertion_passing_at_zero_actions_is_still_vacuous(
        self, monkeypatch
    ):
        """The exemption keys off ``not_``; it must not swallow ordinary checks."""
        for condition in ("exists", "field_contains", "field_semantic_match"):
            _install_fake_env(
                monkeypatch,
                {"t1": _eval_result(False, [_check("free_point", True)])},
                {"t1": [{"condition": condition}]},
            )
            vacuous, _, _, _ = gen.audit_zero_action_discrimination(["t1"], "/data")
            assert [v[:2] for v in vacuous] == [("t1", "free_point")], condition

    def test_the_exemption_is_matched_positionally_not_by_any_spec(
        self, monkeypatch
    ):
        """``specs[i]`` describes ``details[i]`` -- a per-index decision."""
        _install_fake_env(
            monkeypatch,
            {
                "t1": _eval_result(
                    False,
                    [
                        _check("positive_first", True),
                        _check("negative_second", True),
                        _check("positive_third", True),
                    ],
                )
            },
            {
                "t1": [
                    {"condition": "exists"},
                    {"condition": "not_exists"},
                    {"condition": "field_contains"},
                ]
            },
        )
        vacuous, _, _, _ = gen.audit_zero_action_discrimination(["t1"], "/data")
        assert [v[1] for v in vacuous] == ["positive_first", "positive_third"]

    def test_a_missing_or_short_spec_list_grants_no_exemption(self, monkeypatch):
        """A desync must fail LOUD: no spec means the check is reported, not hidden."""
        for specs in ([], [{"condition": "not_exists"}], None):
            _install_fake_env(
                monkeypatch,
                {
                    "t1": _eval_result(
                        False,
                        [_check("a", True), _check("unspecced", True)],
                    )
                },
                {"t1": specs},
            )
            vacuous, _, _, _ = gen.audit_zero_action_discrimination(["t1"], "/data")
            assert "unspecced" in [v[1] for v in vacuous], specs

    def test_a_non_dict_spec_entry_grants_no_exemption(self, monkeypatch):
        _install_fake_env(
            monkeypatch,
            {"t1": _eval_result(False, [_check("weird", True)])},
            {"t1": ["not_exists"]},  # a bare string, not a check dict
        )
        vacuous, _, _, _ = gen.audit_zero_action_discrimination(["t1"], "/data")
        assert [v[1] for v in vacuous] == ["weird"]

    def test_negative_assertions_do_not_exempt_the_task_level_verdict(
        self, monkeypatch
    ):
        """Different question, opposite answer -- and the gate must keep asking it.

        One negative assertion holding at t=0 is expected. A task whose ENTIRE
        required set is negative assertions is genuinely free: an agent that does
        nothing wins it. That stays reportable even though no single check is
        flagged vacuous.
        """
        _install_fake_env(
            monkeypatch,
            {"t1": _eval_result(True, [_check("keeps_it", True)])},
            {"t1": [{"condition": "not_exists"}]},
        )
        vacuous, non_disc, _, _ = gen.audit_zero_action_discrimination(
            ["t1"], "/data"
        )
        assert vacuous == []
        assert non_disc == ["t1"]

    def test_exemption_shares_the_materializer_predicate(self):
        """Imported, not re-implemented: a mirrored copy is how ② drifted."""
        from clasharness.scenario import conflict_vlm_materializer as mat

        assert gen._is_negative_assertion is mat._is_negative_assertion

    def test_judge_env_is_cleared_while_the_grader_runs(self, monkeypatch):
        for key in gen._ZERO_ACTION_JUDGE_ENV:
            monkeypatch.setenv(key, "set-before")
        seen_env, _ = _install_fake_env(
            monkeypatch, {"t1": _eval_result(False, [])}
        )
        gen.audit_zero_action_discrimination(["t1"], "/data")
        assert seen_env == [{k: None for k in gen._ZERO_ACTION_JUDGE_ENV}]

    def test_judge_env_is_restored_after_the_audit(self, monkeypatch):
        for key in gen._ZERO_ACTION_JUDGE_ENV:
            monkeypatch.setenv(key, f"val-{key}")
        _install_fake_env(monkeypatch, {"t1": _eval_result(False, [])})
        gen.audit_zero_action_discrimination(["t1"], "/data")
        for key in gen._ZERO_ACTION_JUDGE_ENV:
            assert os.environ[key] == f"val-{key}"

    def test_unset_judge_env_stays_unset_after_the_audit(self, monkeypatch):
        for key in gen._ZERO_ACTION_JUDGE_ENV:
            monkeypatch.delenv(key, raising=False)
        _install_fake_env(monkeypatch, {"t1": _eval_result(False, [])})
        gen.audit_zero_action_discrimination(["t1"], "/data")
        for key in gen._ZERO_ACTION_JUDGE_ENV:
            assert key not in os.environ

    def test_judge_env_is_restored_when_the_audit_is_interrupted(self, monkeypatch):
        """A Ctrl-C mid-audit must not leave the process without credentials."""
        monkeypatch.setenv("GEMINI_API_KEY", "keep-me")
        _install_fake_env(monkeypatch, {})

        def exploding_ids():
            raise KeyboardInterrupt
            yield  # pragma: no cover

        with pytest.raises(KeyboardInterrupt):
            gen.audit_zero_action_discrimination(exploding_ids(), "/data")
        assert os.environ["GEMINI_API_KEY"] == "keep-me"

    def test_replay_failure_does_not_abort_the_remaining_tasks(self, monkeypatch):
        _install_fake_env(
            monkeypatch,
            {
                "bad": RuntimeError("no such task dir"),
                "good": _eval_result(False, [_check("free_point", True)]),
            },
        )
        vacuous, _, _, n_ok = gen.audit_zero_action_discrimination(
            ["bad", "good"], "/data"
        )
        assert n_ok == 1
        assert [v[:2] for v in vacuous] == [("good", "free_point")]

    def test_env_is_built_for_each_task_under_the_given_data_dir(self, monkeypatch):
        _, built = _install_fake_env(
            monkeypatch,
            {"t1": _eval_result(False, []), "t2": _eval_result(False, [])},
        )
        gen.audit_zero_action_discrimination(["t1", "t2"], "/somewhere/data")
        assert [b["task_id"] for b in built] == ["t1", "t2"]
        assert {b["task_data_dir"] for b in built} == {"/somewhere/data"}
        assert all(b["env"].reset_called and b["env"].exited for b in built)

    def test_vacuous_message_is_truncated(self, monkeypatch):
        _install_fake_env(
            monkeypatch,
            {"t1": _eval_result(False, [_check("free", True, message="x" * 500)])},
        )
        vacuous, _, _, _ = gen.audit_zero_action_discrimination(["t1"], "/data")
        assert len(vacuous[0][2]) == 160

    def test_findings_accumulate_across_tasks(self, monkeypatch):
        _install_fake_env(
            monkeypatch,
            {
                "t1": _eval_result(True, [_check("a", True)]),
                "t2": _eval_result(False, [_check("b", True), _check("c", False)]),
                "t3": _eval_result(False, [_check("d", False, ok=False)]),
            },
        )
        vacuous, non_disc, unaudited, n_ok = gen.audit_zero_action_discrimination(
            ["t1", "t2", "t3"], "/data"
        )
        assert [v[1] for v in vacuous] == ["a", "b"]
        assert non_disc == ["t1"] and unaudited == 1 and n_ok == 3


# ---------------------------------------------------------------------------
# Version-level durability: the checkpoint and the retry that consumes it.
# ---------------------------------------------------------------------------


def _seed_dir_with(tmp_path, name="clinical_trial_dosage_conflict"):
    """A one-seed seed dir, copied from the real corpus seeds.

    Copied rather than pointed at in-tree so the run's cleanup logic can never
    touch the repo, and so the test does not depend on how many seeds ship.
    """
    import shutil

    src = (
        Path(gen.ROOT)
        / "clasharness/data/scenario_seeds/conflict_vlm_v1"
        / f"{name}.json"
    )
    assert src.is_file(), f"seed fixture missing: {src}"
    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    shutil.copy(src, seed_dir / f"{name}.json")
    return seed_dir


def _run_generation(monkeypatch, tmp_path, materialize, *, extra_argv=()):
    """Drive ``main()`` over a single seed with materialization stubbed out.

    Everything upstream of the version loop is real (arg parsing, path
    resolution, seed loading, the loop itself); only the pipeline call is
    replaced, so the test observes exactly what the loop hands the pipeline and
    what it does when the pipeline raises.
    """
    import sys

    import clasharness.evaluation.llm_judge as judge

    monkeypatch.setattr(judge, "sync_judge_env", lambda *a, **k: None)
    monkeypatch.setattr(gen, "validate_conflict_scenario_strict", lambda spec: None)
    monkeypatch.setattr(gen, "materialize_conflict_scenario", materialize)

    seed_dir = _seed_dir_with(tmp_path)
    out_dir = tmp_path / "specs"
    data_dir = tmp_path / "data"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_conflict_vlm_scenarios.py",
            "--provider=mock",
            f"--seed_dir={seed_dir}",
            f"--output_dir={out_dir}",
            f"--output_data_dir={data_dir}",
            "--num_versions=1",
            "--num_workers=1",
            "--noaudit_discrimination",
            *extra_argv,
        ],
    )
    gen.main()
    return out_dir, data_dir


def _ok_spec():
    """The (conflict_spec, openclaw_spec) pair a successful materialization returns.

    ``rounds=[]`` keeps the task-export and dataset-split tail of ``main()``
    inert; this suite is about the version loop, not about exporting.
    """
    return SimpleNamespace(), SimpleNamespace(rounds=[])


class TestTheCheckpointOutlivesTheFailureThatNeedsIt:
    """The resume checkpoint must not live inside the dir a failure deletes.

    This pairing is what turned recoverable late-stage failures into total
    losses: the pipeline writes its checkpoint after Stage 1c and Stage 2, a
    Stage 3 failure then propagates up here, and the failure handler rmtree's
    the scenario output dir -- which used to be the very directory holding that
    checkpoint. The state a retry needed was destroyed microseconds after being
    written, so ``--resume`` had nothing to resume from.
    """

    def test_checkpoint_dir_is_outside_the_scenario_output_dir(
        self, monkeypatch, tmp_path
    ):
        seen = {}

        def _materialize(**kwargs):
            seen.update(kwargs)
            return _ok_spec()

        out_dir, _ = _run_generation(monkeypatch, tmp_path, _materialize)

        checkpoint_dir = Path(seen["checkpoint_dir"])
        scenario_dir = Path(seen["output_dir"])
        assert checkpoint_dir == out_dir / ".checkpoints"
        # The actual invariant: deleting the scenario dir cannot reach it.
        assert scenario_dir not in checkpoint_dir.parents
        assert checkpoint_dir != scenario_dir

    def test_checkpoint_survives_the_failed_version_cleanup(
        self, monkeypatch, tmp_path
    ):
        """End-to-end: write a checkpoint, fail, and it is still on disk."""
        seen = {}

        def _materialize(**kwargs):
            seen.update(kwargs)
            # Stand in for the pipeline's Stage 1c/Stage 2 checkpoint write,
            # which happens before the stage that fails.
            ckpt_dir = Path(kwargs["checkpoint_dir"])
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            (ckpt_dir / "ckpt.json").write_text("{}", encoding="utf-8")
            # And for the half-written scenario tree the cleanup targets.
            Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)
            raise RuntimeError("Failed to generate background payload after 3 attempts.")

        _run_generation(
            monkeypatch, tmp_path, _materialize, extra_argv=["--version_retries=0"]
        )

        assert (Path(seen["checkpoint_dir"]) / "ckpt.json").is_file()
        # The debris cleanup still fires -- the point is that it misses the
        # checkpoint, not that it stopped running.
        assert not Path(seen["output_dir"]).exists()

    def test_versions_share_the_dir_but_carry_distinct_slugs(
        self, monkeypatch, tmp_path
    ):
        """A shared checkpoint DIR is only safe because the file is slug-keyed.

        This is the concrete hazard of moving the checkpoint to a corpus-level
        directory: every version now writes into one directory, so if two
        versions could produce the same checkpoint filename they would
        overwrite each other's resume state and a retry could resume from a
        different scenario. The script's half of that contract is handing the
        pipeline a distinct ``seed.slug`` per version.
        """
        calls = []

        def _materialize(**kwargs):
            calls.append(kwargs)
            return _ok_spec()

        _run_generation(
            monkeypatch, tmp_path, _materialize, extra_argv=["--num_versions=2"]
        )

        assert len(calls) == 2
        assert calls[0]["checkpoint_dir"] == calls[1]["checkpoint_dir"]
        slugs = [c["seed"].slug for c in calls]
        assert len(set(slugs)) == 2, f"versions must not collide: {slugs}"

    def test_the_checkpoint_filename_is_keyed_by_slug(self):
        """The pipeline's half of that same contract."""
        from clasharness.scenario.conflict_llm_pipeline import ConflictLLMPipeline
        import inspect

        src = inspect.getsource(ConflictLLMPipeline.generate)
        assert 'f"{seed.slug}_pipeline_checkpoint.json"' in src, (
            "checkpoint filename is no longer slug-keyed; a corpus-level "
            "checkpoint dir is unsafe without it"
        )


class TestAFailedVersionIsRetriedRatherThanLost:
    """A version that dies to a stochastic schema miss must be re-drawn.

    Measured on the 60-scenario v3 run: 30% of versions emitted a malformed
    background workspace on the first try and 2 exhausted the stage's retry
    budget and were lost -- yet both of those seeds had generated cleanly in
    the previous corpus. The loss was luck, not content, and nothing above the
    stage retried.
    """

    def test_a_version_that_fails_once_is_retried_and_kept(
        self, monkeypatch, tmp_path
    ):
        calls = []

        def _materialize(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError("Failed to generate background payload after 3 attempts.")
            return _ok_spec()

        _run_generation(monkeypatch, tmp_path, _materialize)

        assert len(calls) == 2, "a failed version was not retried"

    def test_the_retry_resumes_from_the_checkpoint(self, monkeypatch, tmp_path):
        """Without resume=True the retry would redo the ~85s of valid Stage 1-2 work."""
        calls = []

        def _materialize(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError("boom")
            return _ok_spec()

        _run_generation(monkeypatch, tmp_path, _materialize)

        assert calls[0]["resume"] is False, "first attempt must honour the user's flag"
        assert calls[1]["resume"] is True, "retry must resume from the checkpoint"
        assert calls[0]["checkpoint_dir"] == calls[1]["checkpoint_dir"]

    def test_retries_are_bounded_and_the_failure_is_still_reported(
        self, monkeypatch, tmp_path
    ):
        calls = []

        def _materialize(**kwargs):
            calls.append(kwargs)
            raise RuntimeError("always fails")

        _run_generation(
            monkeypatch, tmp_path, _materialize, extra_argv=["--version_retries=2"]
        )

        assert len(calls) == 3, "expected 1 initial attempt + 2 retries"

    def test_zero_retries_restores_the_historical_fail_once_behaviour(
        self, monkeypatch, tmp_path
    ):
        calls = []

        def _materialize(**kwargs):
            calls.append(kwargs)
            raise RuntimeError("always fails")

        _run_generation(
            monkeypatch, tmp_path, _materialize, extra_argv=["--version_retries=0"]
        )

        assert len(calls) == 1

    def test_a_clean_first_attempt_is_not_retried(self, monkeypatch, tmp_path):
        calls = []

        def _materialize(**kwargs):
            calls.append(kwargs)
            return _ok_spec()

        _run_generation(monkeypatch, tmp_path, _materialize)

        assert len(calls) == 1
        assert calls[0]["resume"] is False


class _YieldingStdout:
    """A stdout that hands the GIL over in the middle of a ``print``.

    ``print(x, flush=True)`` is three calls -- ``write(x)``, ``write("\\n")``,
    ``flush()`` -- so there are scheduling points INSIDE one print. Under real
    contention (runs use 20 workers) another thread wins those points every so
    often; sleeping in ``write`` makes the same hazard happen every time
    instead of sometimes, which is the difference between a test and a coin
    flip. It manufactures the SCHEDULING, never the interleave: if the writes
    were atomic, or held a lock, no amount of yielding could splice them.
    """

    def __init__(self):
        self.chunks = []

    def write(self, s):
        self.chunks.append(s)
        if s != "\n":
            time.sleep(0.001)
        return len(s)

    def flush(self):
        pass

    def getvalue(self):
        return "".join(self.chunks)


def _log_from_threads(monkeypatch, messages):
    """Run ``gen._log`` from one thread per message; return what stdout got."""
    sink = _YieldingStdout()
    monkeypatch.setattr(sys, "stdout", sink)
    threads = [threading.Thread(target=gen._log, args=(m,)) for m in messages]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return sink


class TestLogIsAtomicAcrossWorkers:
    """One ``_log`` call must reach stdout as one whole line.

    Only the FILE mirror was ever locked. The ``print`` sat outside the lock, so
    at ``--num_workers 20`` the run log stayed trustworthy while the terminal
    silently welded records together -- and the terminal is what anyone watching
    a six-hour generation actually reads. Measured on the unlocked version with
    twenty threads: twenty lines came out as eleven, five of them two or three
    records spliced end to end.
    """

    MESSAGES = [f"worker-{i:02d} finished phase 3" for i in range(20)]

    def test_concurrent_logging_does_not_splice_lines_on_stdout(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(gen, "_LOG_FILE", None)
        sink = _log_from_threads(monkeypatch, self.MESSAGES)

        lines = [l for l in sink.getvalue().split("\n") if l]
        assert len(lines) == len(self.MESSAGES), (
            f"{len(self.MESSAGES)} log calls produced {len(lines)} lines;"
            " records were spliced together mid-write"
        )
        # Every line is exactly one record: one timestamp, one message.
        for line in lines:
            assert line.count("worker-") == 1, f"spliced line: {line!r}"
        assert {l.split("] ", 1)[1] for l in lines} == set(self.MESSAGES)

    def test_stdout_and_the_run_log_receive_the_same_lines_in_the_same_order(
        self, monkeypatch, tmp_path
    ):
        # The two sinks are written under one lock, so they cannot disagree on
        # ORDER either. That is what lets a scrollback be compared against a run
        # log line for line -- the exact operation used to attribute the 208
        # gold-satisfiability aborts to particular scenarios.
        log = tmp_path / "run.log"
        monkeypatch.setattr(gen, "_LOG_FILE", str(log))
        monkeypatch.setattr(gen, "_LOG_FH", None)
        try:
            sink = _log_from_threads(monkeypatch, self.MESSAGES)
        finally:
            if gen._LOG_FH is not None:
                gen._LOG_FH.close()

        on_stdout = [l for l in sink.getvalue().split("\n") if l]
        in_file = [l for l in log.read_text().split("\n") if l]
        assert on_stdout == in_file, (
            "the two sinks disagree, so a scrollback cannot be lined up against"
            " the run log"
        )

    def test_both_sinks_are_written_under_one_single_lock_acquisition(self):
        # STRUCTURAL, because the property it guards is not reliably observable
        # from the outside. Splitting the body into two consecutive
        # ``with _LOG_LOCK`` blocks keeps every line whole -- the behavioural
        # test above stays green -- while letting the two sinks disagree on
        # ORDER: thread B can win the second lock after losing the first, so
        # stdout says A,B and the run log says B,A. That divergence needs a
        # precise interleaving to show up, so a timing test for it would be a
        # coin flip; the shape is what is actually being promised, so the shape
        # is what is pinned.
        #
        # Being able to line the two transcripts up matters concretely: it is
        # how the 208 gold-satisfiability aborts were attributed to individual
        # scenarios.
        import ast

        tree = ast.parse(Path(gen.__file__).read_text())
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_log")
        locks = [n for n in ast.walk(fn) if isinstance(n, ast.With) and any(
            isinstance(i.context_expr, ast.Name)
            and i.context_expr.id == "_LOG_LOCK" for i in n.items)]
        assert len(locks) == 1, (
            f"_log takes _LOG_LOCK {len(locks)} times; the sinks must share one"
            " acquisition or they can be ordered differently"
        )
        prints = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Name) and n.func.id == "print"]
        assert len(prints) == 1
        assert prints[0] in set(ast.walk(locks[0])), (
            "the print sits outside the lock -- this is the original race:"
            " print() is write(text) then write(newline), so a concurrent"
            " worker splices its line into the gap"
        )

    def test_a_run_with_no_log_file_still_prints(self, monkeypatch, tmp_path):
        # The file mirror is optional and the early-return that used to skip it
        # sat between the print and the lock. Folding both sinks into one locked
        # block must not make stdout conditional on a log file existing.
        monkeypatch.setattr(gen, "_LOG_FILE", None)
        monkeypatch.setattr(gen, "_LOG_FH", None)
        sink = _log_from_threads(monkeypatch, ["solo message"])

        assert "solo message" in sink.getvalue()
        assert gen._LOG_FH is None, "no log file was configured; none was opened"


# ---- exact-arm fairness audit ----
#
# The defect this audit watches for shipped a whole corpus: 178 of 400
# exact-graded rounds (44.5%) turned on a compound phrase put through a raw
# substring match, which no correct-but-differently-punctuated report can pass.
# It was found by hand, weeks later, because no part of the run reported it.


def _spec_with(rounds_and_checks):
    """Minimal shipped-spec shape the audit reads: rounds + evals."""
    rounds, evals = [], []
    for i, (value, checks) in enumerate(rounds_and_checks, start=1):
        rid = f"r{i:02d}"
        rounds.append({"round_id": rid, "resolved_value": value})
        evals.append({"round_id": rid, "checks": checks})
    return {"rounds": rounds, "evals": evals}


def _exact_check(value):
    return {
        "type": "effect",
        "effect_type": "files_created",
        "condition": "field_contains",
        "expected": {"field": "content", "value": value},
    }


def _write_scenario(tmp_path, name, spec):
    import json

    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}_spec.json").write_text(json.dumps(spec))
    return d


class TestExactArmFairnessAudit:
    def test_only_a_raw_substring_check_counts_as_the_exact_arm(self, tmp_path):
        # THE measurement trap, one round per conjunct. Each decoy below is a
        # check that grades the same value MORE forgivingly than a raw
        # substring match, so counting it would inflate the arm and dilute
        # every rate computed against it -- including the residual rate that
        # is the whole point of the audit.
        #
        #  * assets_accessed  -- also field_contains, just on field 'id'. This
        #    is the big one: 1,914 such checks against 554 real exact-arm
        #    checks on the last two corpora, so admitting the shape would
        #    roughly quadruple the arm.
        #  * messages_sent    -- also field_contains, but effect_checker
        #    matches it with a TOKENIZED, order-tolerant, stop-word-stripped
        #    subsequence (``_effect_field_contains``), not a substring. A
        #    compound value survives that check, so it is not the unfair arm.
        #  * field_semantic_match -- the semantic arm by definition.
        #  * field 'path' -- an assertion about WHERE a file was written, not
        #    about what the report says.
        #  * a content assertion carrying some OTHER string -- a protocol
        #    phrase the report must also contain ('release database write lock'
        #    is a real one). Graded by substring, but not on the decisive
        #    value, so it says nothing about whether that value is citable.
        #
        # Those last two are DEFENSIVE and are labelled so nobody reads more
        # into them than is there. Measured over the last two corpora: every
        # files_created + field_contains check is on 'content' (none on
        # 'path'), and every one of them on a round that HAS a decisive value
        # carries exactly that value (zero mismatches -- the 73 that differ are
        # all on rounds with no decisive value at all, which the audit drops
        # earlier). They are precision against a future check shape.
        #
        # This test is the ONLY place the four conjuncts are pinned, because
        # the corpora cannot pin them: every excluded shape out there is
        # blocked by two or more conjuncts at once, so deleting any single one
        # leaves the measured arm unchanged at 412 rounds. Real data cannot
        # tell you a conjunct matters; these decoys can.
        spec = _spec_with([
            ("42.0 ms", [_exact_check("42.0 ms")]),
            ("3.3 ms", [{
                "type": "effect",
                "effect_type": "files_created",
                "condition": "field_contains",
                "expected": {"field": "path", "value": "3.3 ms"},
            }]),
            ("8.8 ms", [{
                "type": "effect",
                "effect_type": "files_created",
                "condition": "field_contains",
                "expected": {"field": "content",
                             "value": "release database write lock"},
            }]),
            ("99.9 ms", [{
                "type": "effect",
                "effect_type": "assets_accessed",
                "condition": "field_contains",
                "expected": {"field": "id", "value": "99.9 ms"},
            }]),
            ("55.5 ms", [{
                "type": "effect",
                "effect_type": "messages_sent",
                "condition": "field_contains",
                "expected": {"field": "content", "value": "55.5 ms"},
            }]),
            ("7.5 ms", [{
                "type": "effect",
                "effect_type": "files_created",
                "condition": "field_semantic_match",
                "expected": {"field": "content", "value": "7.5 ms"},
            }]),
        ])
        d = _write_scenario(tmp_path, "s_v01", spec)
        report = gen.audit_exact_arm_fairness([d])

        assert report["rounds_with_value"] == 6
        assert report["exact_arm"] == 1, (
            "only files_created + field_contains + content is graded by a raw"
            " substring match"
        )

    def test_the_residual_instrument_sees_what_the_pipeline_rule_misses(
        self, tmp_path
    ):
        # THE anti-tautology test, and the reason the audit carries a second
        # INSTRUMENT (never a second decision rule). Everything
        # ``_is_compound_decisive_value`` flags was already downgraded out of
        # the exact arm, so measuring the arm with that rule returns 0 by
        # construction -- a guard cannot observe its own blind spot.
        #
        # Both values below are real, off the last shipped corpus, and both are
        # invisible to the rule: 'SHA256: ...' is two words with no bracket and
        # no clause comma, '68.5% HTTP 502 Bad Gateway' is five words against a
        # six-word threshold. The audit must still surface them.
        from clasharness.scenario.conflict_llm_pipeline import (
            _is_compound_decisive_value,
        )

        shape_a = "SHA256: 7f8a91b2c3d4e5f60718293a4b5c6d7e"
        shape_b = "68.5% HTTP 502 Bad Gateway"
        # The premise of the test: the decision rule really does miss these.
        assert not _is_compound_decisive_value(shape_a)
        assert not _is_compound_decisive_value(shape_b)

        d = _write_scenario(tmp_path, "s_v01", _spec_with([
            (shape_a, [_exact_check(shape_a)]),
            (shape_b, [_exact_check(shape_b)]),
        ]))
        report = gen.audit_exact_arm_fairness([d])

        found = {value for _s, _rid, value in report["residual"]}
        assert found == {shape_a, shape_b}
        assert report["residual_pct"] == 100.0

    def test_a_thousands_separator_is_not_a_clause_boundary(self, tmp_path):
        # FALSE-POSITIVE CONTROL, measured. An earlier version of this probe
        # treated every ',' as a clause separator and reported 12 unfair values
        # on a corpus that had 4: eight of them were readings like '12,450 ms'.
        # An audit that cries wolf about thousands separators gets ignored,
        # which is the same outcome as not having one.
        values = ["12,450 ms", "2,150 queries/min", "1,250 queued requests"]
        d = _write_scenario(tmp_path, "s_v01", _spec_with(
            [(v, [_exact_check(v)]) for v in values]
        ))
        report = gen.audit_exact_arm_fairness([d])

        assert report["exact_arm"] == 3
        assert report["residual"] == []

    def test_an_atomic_arm_reports_a_clean_residual(self, tmp_path):
        # The negative control the whole report hangs on: if the audit flagged
        # ordinary atomic readings it would be unreadable. These are the long
        # atomic shapes the guard is also required to leave alone.
        values = [
            "https://telemetry-alpha.cluster.internal/v1/sync",
            "db-prod-02.internal:5432",
            "2024-05-20T14:42:00Z",
            "1,420 queries/min",
            "SEC-CONTAIN-8842",
        ]
        d = _write_scenario(tmp_path, "s_v01", _spec_with(
            [(v, [_exact_check(v)]) for v in values]
        ))
        report = gen.audit_exact_arm_fairness([d])

        assert report["exact_arm"] == 5
        assert report["residual"] == []
        assert report["exact_arm_pct"] == 100.0

    def test_the_downgrade_marker_matches_the_line_the_pipeline_emits(self):
        # The downgrade count is recovered by grepping the run log, because the
        # pipeline deliberately keeps no run-level counter. That coupling fails
        # SILENTLY -- reword the log line and the audit reports 0 downgrades
        # forever, which reads as "the guard never fires" rather than as a
        # broken probe. So the marker is pinned against the emitting source.
        #
        # Matched against parsed STRING LITERALS, not against the raw file. The
        # log line is written as two adjacent f-strings broken over a line
        # boundary, so 'citable DOWNGRADED:' never appears contiguously in the
        # source text even though it always appears in the output. The parser
        # folds adjacent literals, so this compares what actually gets printed.
        import ast

        from clasharness.scenario import conflict_llm_pipeline

        tree = ast.parse(Path(conflict_llm_pipeline.__file__).read_text())
        literals = [
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
        assert any(gen._CITABLE_DOWNGRADE_MARKER in text for text in literals), (
            "the audit greps for a log line the pipeline no longer emits;"
            " the downgrade rate is silently 0"
        )

    def test_the_audit_reports_and_never_repairs(self, tmp_path):
        # Report-only is a design constraint, not an implementation detail: the
        # fix for an unfair value belongs in the S1 prompt, and rewriting the
        # shipped corpus here would hide a prompt that had stopped working.
        # A malformed spec must also degrade to a skip, never to an exception
        # that takes down a finished run's summary.
        import json

        bad = tmp_path / "broken_v01"
        bad.mkdir()
        (bad / "broken_v01_spec.json").write_text("{not json")
        good = _write_scenario(tmp_path, "s_v01", _spec_with([
            ("SHA256: 7f8a91b2c3d4e5f60718293a4b5c6d7e",
             [_exact_check("SHA256: 7f8a91b2c3d4e5f60718293a4b5c6d7e")]),
        ]))
        before = (good / "s_v01_spec.json").read_bytes()

        report = gen.audit_exact_arm_fairness([bad, good])

        assert report["specs"] == 1, "the unreadable spec is skipped, not fatal"
        assert len(report["residual"]) == 1
        assert (good / "s_v01_spec.json").read_bytes() == before, (
            "the audit must not rewrite the corpus it is measuring"
        )
        # Round-trips as the same JSON it started as.
        assert json.loads(before)["rounds"][0]["round_id"] == "r01"

    def test_the_downgrade_count_is_read_from_the_run_log(self, tmp_path):
        # And is reported as unavailable rather than as 0 when there is no log:
        # a missing measurement and a measurement of zero are different facts,
        # and collapsing them would make a blind run look like a clean one.
        d = _write_scenario(tmp_path, "s_v01", _spec_with([
            ("42.0 ms", [_exact_check("42.0 ms")]),
        ]))
        assert gen.audit_exact_arm_fairness([d])["downgrades"] is None

        log = tmp_path / "run.log"
        log.write_text(
            "[10:00:00] [s_v01]   [Phase Decisive phase_1 r01] citable"
            " DOWNGRADED: 'BLOCKED (3/4 gates ready)' carries more than the"
            " reading, so an exact substring match would fail.\n"
            "[10:00:01] [s_v01] something else entirely\n"
            "[10:00:02] [s_v01]   [Phase Decisive phase_2 r04] citable"
            " DOWNGRADED: '16 open tickets (1 P1 critical)' carries more.\n"
        )
        assert gen.audit_exact_arm_fairness([d], log_path=log)["downgrades"] == 2

    def test_both_gold_satisfiability_abort_wordings_are_counted(self, tmp_path):
        # The gold-satisfiability gate FAILS OPEN: when the judge dies the
        # round keeps its checks and generation carries on. That is the right
        # call -- dropping checks on an infrastructure blip would silently thin
        # the corpus -- but it makes a dead judge invisible, and on the sonnet-5
        # run the gate aborted 208 times without one word in the run summary.
        #
        # The gate reports the outage in TWO wordings and both are real: it logs
        # 'judge unavailable' when the call RAISED and 'judge error' when the
        # call returned a verdict with ok=False. Counting only one of them would
        # under-report the outage by whichever half happened to be quiet, which
        # is worse than not counting at all -- a small number reads as "a couple
        # of blips" and gets waved through.
        d = _write_scenario(tmp_path, "s_v01", _spec_with([
            ("42.0 ms", [_exact_check("42.0 ms")]),
        ]))
        log = tmp_path / "run.log"
        log.write_text(
            "[10:00:00] [s_v01]     [gold-satisfiability] judge unavailable"
            " ('ThinkingBlock' object has no attribute 'text'); skipping\n"
            "[10:00:01] [s_v01] a perfectly ordinary line\n"
            "[10:00:02] [s_v01]     [gold-satisfiability] judge error (Error"
            " code: 401 - Unauthorized); skipping\n"
            "[10:00:03] [s_v01]     [gold-satisfiability] judge unavailable"
            " ('ThinkingBlock' object has no attribute 'text'); skipping\n"
        )
        report = gen.audit_exact_arm_fairness([d], log_path=log)
        assert report["gold_sat_aborts"] == 3

    def test_a_run_with_no_aborts_reports_zero_and_a_blind_run_reports_neither(
        self, tmp_path
    ):
        # THE silent-zero test. 'the gate never broke' and 'nobody looked' are
        # different facts and the audit must not launder the second into the
        # first: an unreadable log has to stay distinguishable from a clean run
        # all the way to the banner, which renders None as 'unavailable'.
        d = _write_scenario(tmp_path, "s_v01", _spec_with([
            ("42.0 ms", [_exact_check("42.0 ms")]),
        ]))
        # No log at all -> not measured.
        assert gen.audit_exact_arm_fairness([d])["gold_sat_aborts"] is None
        # A log that cannot be read -> also not measured, never 0.
        assert gen.audit_exact_arm_fairness(
            [d], log_path=tmp_path / "does_not_exist.log"
        )["gold_sat_aborts"] is None
        # A log with no aborts -> measured, and the measurement is 0.
        clean = tmp_path / "clean.log"
        clean.write_text("[10:00:00] [s_v01] nothing went wrong here\n")
        assert gen.audit_exact_arm_fairness(
            [d], log_path=clean
        )["gold_sat_aborts"] == 0

    def test_a_gate_that_dropped_a_check_did_its_job_and_is_not_an_abort(
        self, tmp_path
    ):
        # FALSE-POSITIVE CONTROL, and the reason the markers carry a trailing
        # ' ('. The gate logs a THIRD line under the same '[gold-satisfiability]'
        # prefix -- 'dropping unsatisfiable check after final attempt' -- and
        # that one means the gate WORKED: it caught a check the gold trajectory
        # could not satisfy and removed it. Counting it as an outage would
        # invert the metric's meaning, reporting the gate as broken precisely
        # when it is doing the job it exists for.
        d = _write_scenario(tmp_path, "s_v01", _spec_with([
            ("42.0 ms", [_exact_check("42.0 ms")]),
        ]))
        log = tmp_path / "run.log"
        log.write_text(
            "[10:00:00] [s_v01]     [gold-satisfiability] dropping unsatisfiable"
            " check after final attempt: 'the report names every on-call' --"
            " the gold trajectory messages only one recipient\n"
            "[10:00:01] [s_v01]   - criterion 'x' is NOT satisfied by the gold"
            " trajectory. The judge said: no\n"
        )
        assert gen.audit_exact_arm_fairness(
            [d], log_path=log
        )["gold_sat_aborts"] == 0

    def test_the_abort_markers_match_the_lines_the_pipeline_emits(self):
        # Same silent-coupling trap as the downgrade marker above, same fix.
        # The pipeline keeps no run-level abort counter, so the only way to know
        # the gate went offline is to grep its log lines -- reword either one and
        # this audit reports 0 aborts forever, which reads as 'the judge was
        # healthy' rather than as 'the probe is broken'. That is the exact
        # failure mode this whole counter exists to prevent, so it must not be
        # reintroduced one level up.
        #
        # Matched against parsed STRING LITERALS. Both lines are f-strings, and
        # the parser exposes their literal segments as Constant nodes, so this
        # compares against text that really reaches the log.
        import ast

        from clasharness.scenario import conflict_llm_pipeline

        tree = ast.parse(Path(conflict_llm_pipeline.__file__).read_text())
        literals = [
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
        for marker in gen._GOLD_SAT_ABORT_MARKERS:
            assert any(marker in text for text in literals), (
                f"the audit greps for {marker!r}, which the pipeline no longer"
                " emits; the gold-satisfiability abort count is silently 0"
            )
        assert len(gen._GOLD_SAT_ABORT_MARKERS) == 2, (
            "the gate has two abort wordings (raised vs ok=False); dropping one"
            " halves the reported outage"
        )

    def test_the_audit_actually_runs_at_the_end_of_a_generation(
        self, monkeypatch, tmp_path, capsys
    ):
        # THE WIRING TEST, and it is not optional. The call site sits inside a
        # broad ``except`` so a bad corpus can never take down a finished run's
        # summary -- which means a bug in the call site ITSELF (a wrong
        # variable, a renamed field) degrades to one WARNING line and the audit
        # silently stops running. Every test above calls the function directly
        # and would not notice, so this one drives real ``main()``.
        import json

        def _materialize(**kwargs):
            # Stand in for a finished scenario: the spec plus scenario.json,
            # which is the completion marker the audit filters on.
            d = Path(kwargs["output_dir"])
            d.mkdir(parents=True, exist_ok=True)
            (d / "scenario.json").write_text("{}", encoding="utf-8")
            # A real materialization writes the manifest BEFORE scenario.json,
            # so a finished package always has both. Omitting it here would make
            # the scenario untestable for sourceability, which the shipping gate
            # (rightly) treats as a defect -- and this test is about the
            # residual banner, not about that.
            (d / "asset_manifest.json").write_text('{"assets": []}',
                                                   encoding="utf-8")
            (d / f"{d.name}_spec.json").write_text(json.dumps(_spec_with([
                ("SHA256: 7f8a91b2c3d4e5f60718293a4b5c6d7e",
                 [_exact_check("SHA256: 7f8a91b2c3d4e5f60718293a4b5c6d7e")]),
                ("2024-05-20T14:42:00Z",
                 [_exact_check("2024-05-20T14:42:00Z")]),
            ])), encoding="utf-8")
            return _ok_spec()

        _run_generation(monkeypatch, tmp_path, _materialize)
        out = capsys.readouterr().out

        assert "Exact-arm fairness audit" in out, (
            "the audit never ran, or its exception was swallowed into a warning"
        )
        assert "WARNING: exact-arm fairness audit failed" not in out
        # It reported the finding, not just the header.
        assert "UNGRADABLE RESIDUAL          : 1" in out
        assert "SHA256: 7f8a91b2c3d4e5f60718293a4b5c6d7e" in out
        # And the atomic timestamp sharing the corpus was not dragged in.
        assert "2024-05-20T14:42:00Z" not in out

    def _run_with_seeded_log(self, monkeypatch, tmp_path, log_text):
        """Drive ``main()`` against a run log pre-seeded with ``log_text``.

        ``_log`` opens the run log in append mode and caches the handle in a
        module global, so a previous test's handle would otherwise still be open
        and this run's lines would land in the WRONG file -- leaving the audit
        reading a log that does not exist and reporting 'unavailable' by
        accident. Clearing the cached handle makes the assertion below depend on
        the banner and not on test ordering.
        """
        import json

        monkeypatch.setattr(gen, "_LOG_FH", None)
        log = tmp_path / "seeded.log"
        log.write_text(log_text, encoding="utf-8")

        def _materialize(**kwargs):
            d = Path(kwargs["output_dir"])
            d.mkdir(parents=True, exist_ok=True)
            (d / "scenario.json").write_text("{}", encoding="utf-8")
            # See the note above: a finished package always carries the
            # manifest, and leaving it out would trip the shipping gate on
            # untestability rather than on anything these banner tests assert.
            (d / "asset_manifest.json").write_text('{"assets": []}',
                                                   encoding="utf-8")
            (d / f"{d.name}_spec.json").write_text(json.dumps(_spec_with([
                ("2024-05-20T14:42:00Z", [_exact_check("2024-05-20T14:42:00Z")]),
            ])), encoding="utf-8")
            return _ok_spec()

        _run_generation(monkeypatch, tmp_path, _materialize,
                        extra_argv=(f"--log_file={log}",))
        return log

    def test_the_banner_shouts_when_a_fairness_gate_was_offline(
        self, monkeypatch, tmp_path, capsys
    ):
        # THE point of the counter. The gate fails open, so an outage leaves no
        # trace in the corpus and none in the summary -- the sonnet-5 run shipped
        # 208 aborts and read as clean. The count alone is not enough either: a
        # bare number in a twenty-line banner is easy to skim past, so a non-zero
        # count has to say what it MEANS, which is that rounds shipped with a
        # fairness gate switched off.
        self._run_with_seeded_log(monkeypatch, tmp_path, (
            "[09:59:58] [seed_v01]     [gold-satisfiability] judge unavailable"
            " ('ThinkingBlock' object has no attribute 'text'); skipping\n"
            "[09:59:59] [seed_v01]     [gold-satisfiability] judge error"
            " (Error code: 401); skipping\n"
        ))
        out = capsys.readouterr().out

        assert "gold-satisfiability aborts   : 2" in out
        assert "a fairness gate was OFFLINE" in out, (
            "a bare count gets skimmed past; a non-zero count must say what it"
            " means"
        )

    def test_the_banner_prints_a_clean_run_as_zero_rather_than_saying_nothing(
        self, monkeypatch, tmp_path, capsys
    ):
        # The reassuring reading has to be PRINTED. If the line only appeared
        # when non-zero, its absence would be ambiguous between 'no aborts' and
        # 'the counter broke / was never wired up' -- and since the answer is
        # almost always zero, a wiring bug would sit undetected behind the
        # common case forever. Printing 0 is what makes the counter falsifiable.
        self._run_with_seeded_log(monkeypatch, tmp_path,
                                  "[09:59:58] [seed_v01] a quiet run\n")
        out = capsys.readouterr().out

        assert "gold-satisfiability aborts   : 0" in out
        assert "a fairness gate was OFFLINE" not in out, (
            "a clean run must not carry the alarm text, or the alarm stops"
            " meaning anything"
        )

    def test_the_banner_says_unavailable_rather_than_zero_when_it_could_not_look(
        self, monkeypatch, tmp_path, capsys
    ):
        # The silent-zero test, at the BANNER rather than at the function. The
        # audit already returns None for 'could not read the log', but that only
        # helps if the rendering keeps the distinction -- write the line as
        # ``{_gsa or 0}`` and an unmeasured run prints a confident 0, which is
        # the single most misleading thing this banner could say.
        real = gen.audit_exact_arm_fairness
        seen = {}

        def _blind(dirs, log_path=None):
            # Force the unmeasured path by looking at no log at all.
            report = real(dirs, log_path=None)
            seen["aborts"] = report["gold_sat_aborts"]
            return report

        monkeypatch.setattr(gen, "audit_exact_arm_fairness", _blind)
        self._run_with_seeded_log(monkeypatch, tmp_path, "")
        out = capsys.readouterr().out

        assert seen["aborts"] is None, "the premise of the test did not hold"
        assert "gold-satisfiability aborts   : unavailable (no run log)" in out
        assert "gold-satisfiability aborts   : 0" not in out
        assert "a fairness gate was OFFLINE" not in out, (
            "not looking is not evidence of an outage either"
        )


def _write_scenario_pkg(tmp_path, name, spec, rounds, assets, workspace=None,
                        sessions=None, updates=None):
    """Spec plus the two companion files the sourceability audit reads."""
    import json

    d = _write_scenario(tmp_path, name, spec)
    (d / "scenario.json").write_text(json.dumps({
        "workspace": workspace or {},
        "sessions": sessions or [],
        "updates": updates or [],
        "rounds": rounds,
    }))
    (d / "asset_manifest.json").write_text(json.dumps({"assets": assets}))
    return d


class TestExactArmSourceability:
    """Can a solver READ the value the exact-arm check demands?

    A separate invariant from the residual audit above (which asks whether a
    correct report can fail on WORDING) and from the gold-satisfiability gate in
    test_gold_message_satisfiability.py (which asks whether the round's own
    reference trajectory passes). Neither can see this defect: gold is authored
    from ``resolved_value``, so it writes the value whether or not the round
    shows the solver anything containing it.

    On the last shipped corpus this found 18 of 255 exact-arm checks (7.1%)
    demanding a value that appears in NO asset the round exposes. All 18 are
    ``required``, so each capped its task below a passing score for every model.
    Empirically they passed 4 times in 486 model-round attempts (0.8%) while
    the other checks in those very same rounds passed at 37.4% -- the sibling
    comparison is what rules out "those rounds were just hard".
    """

    QUEUE = {"id": "queue", "ocr_text": "oldest ticket waiting 3 min 42 s"}
    BOARD = {"id": "board", "ocr_text": "elapsed 14 min 18 s"}

    def _spec(self):
        return _spec_with([("3 min 42 s", [_exact_check("3 min 42 s")])])

    def test_a_value_only_in_an_asset_the_round_hides_is_unwinnable(self, tmp_path):
        # THE defect, reduced to its smallest form. The round shows a board
        # reading 14 min 18 s and grades on 3 min 42 s, which lives only in an
        # asset it does not show. This is not a hard round -- it is a round with
        # no winning move, and the audit has to say so by name.
        d = _write_scenario_pkg(
            tmp_path, "s_v01", self._spec(),
            rounds=[{"round_id": "r01", "visible_refs": ["board"]}],
            assets=[self.BOARD, self.QUEUE])
        report = gen.audit_exact_arm_fairness([d])

        assert report["exact_arm"] == 1
        assert len(report["unwinnable"]) == 1
        scen, rid, value, tier, holders = report["unwinnable"][0]
        assert (scen, rid, value) == ("s_v01", "r01", "3 min 42 s")
        assert tier == "invisible-asset"
        assert holders == ["queue"], "naming the holder is what makes it fixable"

    def test_the_same_value_is_clean_once_its_asset_is_visible(self, tmp_path):
        # THE FALSE-POSITIVE CONTROL, and it is the paired half of the test
        # above: identical spec, identical assets, one entry added to
        # visible_refs. If this ever reports a finding then the audit is
        # flagging on something other than visibility and every count it
        # produces is noise.
        d = _write_scenario_pkg(
            tmp_path, "s_v01", self._spec(),
            rounds=[{"round_id": "r01", "visible_refs": ["board", "queue"]}],
            assets=[self.BOARD, self.QUEUE])
        report = gen.audit_exact_arm_fairness([d])

        assert report["exact_arm"] == 1, "same arm, so the two runs are comparable"
        assert report["unwinnable"] == []
        assert report["sourceability_specs"] == 1, "and the test really ran"

    def test_available_from_round_is_not_the_visibility_test(self, tmp_path):
        # THE ROOT CONFUSION, pinned. The holder is available from round 1 and
        # the failing round IS round 1, so any audit keyed on
        # ``available_from_round`` calls this clean. It is not: entering the
        # scenario is not the same as being put in front of the solver, and
        # reading the former as the latter is exactly how 18 unwinnable checks
        # shipped. ``visible_refs`` is the authority.
        queue = dict(self.QUEUE, available_from_round=1)
        d = _write_scenario_pkg(
            tmp_path, "s_v01", self._spec(),
            rounds=[{"round_id": "r01", "visible_refs": ["board"]}],
            assets=[self.BOARD, queue])
        report = gen.audit_exact_arm_fairness([d])

        assert len(report["unwinnable"]) == 1, (
            "available_from_round was read as visibility; that is the original"
            " bug, not a fix for it"
        )

    @pytest.mark.parametrize("where", ["workspace", "sessions", "updates"])
    def test_a_value_reachable_off_the_assets_is_not_flagged(self, tmp_path, where):
        # The reachable side is a deliberate OVER-estimate: a solver can read
        # files, scroll back through sessions and receive updates, so a value
        # sitting in any of those is obtainable even with every asset hidden.
        # Erring this way means the audit under-reports, which is the safe
        # direction -- a false accusation sends an author chasing a defect that
        # is not there.
        payload = {"workspace": {"notes.txt": "measured 3 min 42 s"},
                   "sessions": [{"body": "measured 3 min 42 s"}],
                   "updates": [{"body": "measured 3 min 42 s"}]}[where]
        d = _write_scenario_pkg(
            tmp_path, "s_v01", self._spec(),
            rounds=[{"round_id": "r01", "visible_refs": ["board"]}],
            assets=[self.BOARD], **{where: payload})
        report = gen.audit_exact_arm_fairness([d])

        assert report["unwinnable"] == []

    def test_tiers_separate_an_invented_value_from_a_hidden_one(self, tmp_path):
        # The two findings need different repairs and must not be pooled. A
        # hidden value is fixed by widening visible_refs; an invented one --
        # present in no asset anywhere -- means the generator produced a reading
        # nothing renders, and widening visible_refs would not help at all. The
        # last corpus was 18 hidden and 0 invented, and that split is only
        # meaningful if the audit can actually express both.
        d = _write_scenario_pkg(
            tmp_path, "s_v01",
            _spec_with([("3 min 42 s", [_exact_check("3 min 42 s")]),
                        ("XY-NOWHERE-9", [_exact_check("XY-NOWHERE-9")])]),
            rounds=[{"round_id": "r01", "visible_refs": ["board"]},
                    {"round_id": "r02", "visible_refs": ["board"]}],
            assets=[self.BOARD, self.QUEUE])
        report = gen.audit_exact_arm_fairness([d])

        assert {rid: tier for _s, rid, _v, tier, _h in report["unwinnable"]} == {
            "r01": "invisible-asset", "r02": "unsourced"}

    def test_a_scenario_missing_its_companion_files_counts_as_untested(
        self, tmp_path
    ):
        # THE SILENT-ZERO GUARD, the same failure mode the downgrade counter is
        # protected against above. A spec-only directory cannot be tested for
        # sourceability at all, and reporting that as "no findings" would turn
        # a blind audit into a clean bill of health. The count of scenarios the
        # test could actually run on is therefore reported alongside it.
        spec_only = _write_scenario(tmp_path, "s_v01", self._spec())
        report = gen.audit_exact_arm_fairness([spec_only])

        assert report["specs"] == 1
        assert report["exact_arm"] == 1, "the residual half still measures it"
        assert report["unwinnable"] == []
        assert report["sourceability_specs"] == 0, (
            "an untested scenario must not be indistinguishable from a clean one"
        )

    def test_the_finding_reaches_the_run_log(self, monkeypatch, tmp_path, capsys):
        # WIRING. The call site sits inside a broad ``except``, so a renamed
        # field there degrades to a single WARNING and the audit stops
        # reporting. Every test above calls the function directly and would not
        # notice, so this one drives real ``main()``.
        import json

        def _materialize(**kwargs):
            d = Path(kwargs["output_dir"])
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{d.name}_spec.json").write_text(
                json.dumps(_spec_with([("3 min 42 s",
                                        [_exact_check("3 min 42 s")])])),
                encoding="utf-8")
            (d / "scenario.json").write_text(json.dumps({
                "workspace": {}, "sessions": [], "updates": [],
                "rounds": [{"round_id": "r01", "visible_refs": ["board"]}],
            }), encoding="utf-8")
            (d / "asset_manifest.json").write_text(
                json.dumps({"assets": [self.BOARD, self.QUEUE]}),
                encoding="utf-8")
            return _ok_spec()

        # --nofail_on_unwinnable because this fixture ships a genuinely
        # unwinnable round on purpose and the gate would otherwise abort the
        # run before ``capsys`` could be read. What the gate does with the same
        # fixture is asserted in TestUnwinnableShippingGate below.
        _run_generation(monkeypatch, tmp_path, _materialize,
                        extra_argv=("--nofail_on_unwinnable",))
        out = capsys.readouterr().out

        assert "WARNING: exact-arm fairness audit failed" not in out
        assert "UNWINNABLE (unreadable value)  : 1" in out
        # Slug-agnostic: at --num_versions=1 the directory is the bare seed
        # slug, so pinning a name here would test the fixture, not the audit.
        assert "UNWINNABLE [invisible-asset]" in out
        assert "r01: '3 min 42 s'" in out
        assert "readable in: ['queue']" in out


class TestUnwinnableShippingGate:
    """Finding the defect and REFUSING TO SHIP it are two different things.

    Everything above proves the audit can see an unwinnable round. None of it
    proves a run stops because of one -- and for the whole life of the previous
    corpus the audit was report-only, so 27 unwinnable rounds were detected by
    nobody and blocked by nothing. The exit code is the part that a wrapper
    script, a cron job or a tired operator actually reads.

    The gate is exercised from both ends: the pure decision function (every
    combination, cheaply) and real ``main()`` (which proves the decision is
    wired to a non-zero exit and is not evaluated inside the broad ``except``
    that swallows the audit's own exceptions).
    """

    # One finding in the shape audit_exact_arm_fairness emits:
    # (scenario, round_id, value, tier, holders).
    ROW = ("s_v01", "r02", "Deploy blocked", "invisible-asset", ["board"])

    # -- the decision function ------------------------------------------------

    def _gate(self, *, vacuous=(), non_discriminating=(), unwinnable=(),
              audit_status=gen.UNWINNABLE_AUDIT_OK, untested_specs=0,
              fail_on_unwinnable=True):
        return gen.shipping_gate_failure(
            vacuous=list(vacuous),
            non_discriminating=list(non_discriminating),
            unwinnable=list(unwinnable),
            audit_status=audit_status,
            untested_specs=untested_specs,
            fail_on_unwinnable=fail_on_unwinnable,
        )

    def _blocked(self, **kw) -> str:
        """The gate's verdict, asserted to be a refusal, as text."""
        msg = self._gate(**kw)
        assert msg is not None, "expected the gate to block this run"
        return msg

    def test_a_clean_corpus_ships(self):
        # THE FALSE-POSITIVE CONTROL, and the one that matters most: a gate
        # that blocks a good corpus costs a full regeneration run, so it would
        # be switched off within a day and then it protects nothing.
        assert self._gate() is None

    def test_one_unwinnable_round_blocks_the_run(self):
        msg = self._blocked(unwinnable=[self.ROW])
        # Named, not counted. "1 unwinnable round" sends nobody anywhere.
        assert "s_v01" in msg and "r02" in msg and "Deploy blocked" in msg
        assert "invisible-asset" in msg

    def test_the_escape_hatch_ships_anyway(self):
        # A blocking gate with no override is a gate that gets deleted the
        # first time it is wrong. The override is deliberate, named in the
        # message, and has to actually work.
        assert self._gate(unwinnable=[self.ROW],
                          fail_on_unwinnable=False) is None

    def test_a_crashed_audit_blocks_rather_than_passing(self):
        # THE SILENT-ZERO GUARD at gate level. `unwinnable` is empty here for
        # exactly the wrong reason: the audit never produced a verdict. Reading
        # that as "clean" would rebuild the failure mode this whole gate exists
        # to close, and it would do it invisibly.
        msg = self._blocked(audit_status=gen.UNWINNABLE_AUDIT_FAILED)
        assert "UNVERIFIED" in msg

    def test_an_empty_corpus_is_not_treated_as_a_crash(self):
        # The paired control for the test above. A run where every version
        # failed ships nothing, so there is nothing to grade unfairly; blocking
        # here would just bury the real failure report under a bogus one.
        assert self._gate(
            audit_status=gen.UNWINNABLE_AUDIT_NOTHING_SHIPPED) is None

    def test_an_untestable_scenario_blocks_as_unchecked(self):
        # Between "clean" and "broken" sits "not looked at", and only the third
        # one is silent. asset_manifest.json is written before scenario.json on
        # every successful materialization, so a finished package missing it is
        # itself anomalous AND leaves that scenario ungraded by this audit.
        msg = self._blocked(untested_specs=2)
        assert "could not test 2 shipped scenario(s)" in msg

    def test_untested_is_not_reported_when_the_audit_never_ran(self):
        # Don't stack a coverage complaint on top of a crash: the count comes
        # from the report the crash prevented, so it is meaningless here.
        msg = self._blocked(audit_status=gen.UNWINNABLE_AUDIT_FAILED,
                            untested_specs=99)
        assert "could not test" not in msg

    def test_both_audits_report_together_rather_than_first_match(self):
        # Raising on the first failure means the second one costs another full
        # generation run to discover. Both defects are independent, so both are
        # reported at once.
        msg = self._blocked(vacuous=[("t", "c", "m")],
                            non_discriminating=["t2"],
                            unwinnable=[self.ROW])
        assert "Zero-action audit failed" in msg
        assert "Unwinnable-round audit failed" in msg

    def test_the_zero_action_gate_is_unaffected_by_the_unwinnable_override(self):
        # --nofail_on_unwinnable is scoped to ONE audit. If it silenced the
        # other one too, the flag would be a blanket "ship anything" switch
        # that nobody would read as such.
        msg = self._blocked(vacuous=[("t", "c", "m")],
                            unwinnable=[self.ROW],
                            fail_on_unwinnable=False)
        assert "Zero-action audit failed" in msg
        assert "Unwinnable-round audit failed" not in msg

    def test_a_large_finding_list_is_truncated_but_counted(self):
        # 27 findings pasted into a traceback is not a report. The full list is
        # already in the log; the message carries a preview plus the true total
        # so the preview can never be mistaken for the whole of it.
        rows = [(f"s{i}", "r01", f"v{i}", "invisible-asset", []) for i in range(25)]
        msg = self._blocked(unwinnable=rows)
        assert "25 exact-graded" in msg
        assert f"and {25 - gen._GATE_PREVIEW} more" in msg
        assert "v24" not in msg, "the cap is not applied"

    # -- the flag -------------------------------------------------------------

    def test_the_gate_is_on_by_default(self, monkeypatch):
        # The whole point. An opt-in fairness gate is one nobody opts into.
        import sys

        monkeypatch.setattr(
            sys, "argv", ["generate_conflict_vlm_scenarios.py"])
        assert gen.parse_args().fail_on_unwinnable is True

        monkeypatch.setattr(
            sys, "argv",
            ["generate_conflict_vlm_scenarios.py", "--nofail_on_unwinnable"])
        assert gen.parse_args().fail_on_unwinnable is False

    # -- real main() ----------------------------------------------------------

    QUEUE = TestExactArmSourceability.QUEUE
    BOARD = TestExactArmSourceability.BOARD

    def _materializer(self, visible_refs):
        import json

        def _materialize(**kwargs):
            d = Path(kwargs["output_dir"])
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{d.name}_spec.json").write_text(
                json.dumps(_spec_with([("3 min 42 s",
                                        [_exact_check("3 min 42 s")])])),
                encoding="utf-8")
            (d / "scenario.json").write_text(json.dumps({
                "workspace": {}, "sessions": [], "updates": [],
                "rounds": [{"round_id": "r01", "visible_refs": visible_refs}],
            }), encoding="utf-8")
            (d / "asset_manifest.json").write_text(
                json.dumps({"assets": [self.BOARD, self.QUEUE]}),
                encoding="utf-8")
            return _ok_spec()

        return _materialize

    def test_main_exits_non_zero_on_an_unwinnable_round(
        self, monkeypatch, tmp_path, capsys
    ):
        # WIRING, and the one assertion the pure-function tests cannot make:
        # the verdict is read OUTSIDE the broad ``except`` around the audit and
        # is turned into a real SystemExit.
        with pytest.raises(SystemExit) as exc:
            _run_generation(monkeypatch, tmp_path,
                            self._materializer(["board"]))

        msg = str(exc.value)
        assert "Unwinnable-round audit failed" in msg
        assert "3 min 42 s" in msg
        # The statistics report is what makes a failed run diagnosable, so the
        # raise has to come after it rather than instead of it.
        assert "CLASHarness GENERATION STATISTICS REPORT" in capsys.readouterr().out

    def test_the_generated_data_is_left_on_disk(self, monkeypatch, tmp_path):
        # Blocking the SHIP, not destroying the run. Eight hours of generation
        # has to survive the gate so the flagged rounds can be inspected -- and
        # so the corpus can be repaired in place instead of regenerated.
        with pytest.raises(SystemExit):
            _run_generation(monkeypatch, tmp_path,
                            self._materializer(["board"]))

        pkgs = list((tmp_path / "specs").glob("*/scenario.json"))
        assert pkgs, "the gate deleted the corpus it was only supposed to flag"

    def test_main_ships_the_same_corpus_once_the_asset_is_visible(
        self, monkeypatch, tmp_path
    ):
        # THE END-TO-END FALSE-POSITIVE CONTROL: byte-identical fixture with one
        # extra entry in visible_refs. If this raises, the gate is keying on
        # something other than readability and every run it blocks is noise.
        _run_generation(monkeypatch, tmp_path,
                        self._materializer(["board", "queue"]))

    def test_main_survives_a_crashing_audit_but_refuses_to_ship(
        self, monkeypatch, tmp_path, capsys
    ):
        # The two halves of the contract, together. The audit BODY still must
        # not abort the run (it degrades to a WARNING, as before), but its
        # missing VERDICT must not read as approval.
        def _boom(dirs, log_path=None):
            raise RuntimeError("audit exploded")

        monkeypatch.setattr(gen, "audit_exact_arm_fairness", _boom)
        with pytest.raises(SystemExit) as exc:
            _run_generation(monkeypatch, tmp_path,
                            self._materializer(["board", "queue"]))

        out = capsys.readouterr().out
        assert "WARNING: exact-arm fairness audit failed" in out, (
            "the audit body aborted the run instead of degrading to a warning"
        )
        assert "audit exploded" in out
        assert "UNVERIFIED" in str(exc.value)

    def test_the_override_ships_the_defective_corpus(self, monkeypatch, tmp_path):
        # The escape hatch, end to end. Same defective fixture as the blocking
        # test above; only the flag differs.
        _run_generation(monkeypatch, tmp_path, self._materializer(["board"]),
                        extra_argv=("--nofail_on_unwinnable",))


# ---- provider wiring: text-client construction + judge-env propagation ----


def _client_args(**over):
    """Minimal args namespace for _build_text_client / _bedrock_endpoint_from_args."""
    base = dict(
        provider="vertex",
        model="x",
        vertex_project="proj",
        vertex_location=None,
        base_url=gen._GEMINI_DEFAULT_BASE_URL,
        api_key_env=gen._GEMINI_DEFAULT_API_KEY_ENV,
        temperature=0.2,
        retries=5,
        request_delay=0.0,
        max_output_tokens=16384,
    )
    base.update(over)
    return SimpleNamespace(**base)


class _RecordingClient:
    """Stub client class that records its construction kwargs instead of dialing out."""

    def __init__(self, name, store):
        self._name = name
        self._store = store

    def __call__(self, **kwargs):
        self._store["name"] = self._name
        self._store["kwargs"] = kwargs
        return self._store


class TestBuildTextClient:
    def _patch_clients(self, monkeypatch, store):
        for name in (
            "VertexScenarioLLMClient",
            "AnthropicScenarioLLMClient",
            "OpenAICompatibleScenarioLLMClient",
            "GeminiScenarioLLMClient",
        ):
            monkeypatch.setattr(gen, name, _RecordingClient(name, store))

    def test_vertex_anthropic_uses_claude_client(self, monkeypatch):
        store: dict = {}
        self._patch_clients(monkeypatch, store)
        gen._build_text_client(
            _client_args(provider="vertex-anthropic", model="claude-sonnet-4@20250514"),
            lambda s: None,
        )
        assert store["name"] == "AnthropicScenarioLLMClient"
        assert store["kwargs"]["model"] == "claude-sonnet-4@20250514"
        assert store["kwargs"]["project"] == "proj"
        # Claude keeps a real temperature (unlike the gpt-5 opt-out below).
        assert store["kwargs"]["temperature"] == 0.2

    def test_bedrock_gpt_defaults_are_swapped_in(self, monkeypatch):
        store: dict = {}
        self._patch_clients(monkeypatch, store)
        gen._build_text_client(
            _client_args(provider="bedrock-gpt", model="openai.gpt-5.5"),
            lambda s: None,
        )
        assert store["name"] == "OpenAICompatibleScenarioLLMClient"
        kw = store["kwargs"]
        # Gemini defaults auto-swapped to Bedrock defaults; Responses API forced;
        # temperature omitted (gpt-5-class 400s on an explicit value).
        assert kw["base_url"] == gen._BEDROCK_GPT_DEFAULT_BASE_URL
        assert kw["api_key_env"] == gen._BEDROCK_GPT_DEFAULT_API_KEY_ENV
        assert kw["api"] == "responses"
        assert kw["temperature"] is None

    def test_bedrock_gpt_explicit_overrides_are_honoured(self, monkeypatch):
        store: dict = {}
        self._patch_clients(monkeypatch, store)
        gen._build_text_client(
            _client_args(
                provider="bedrock-gpt",
                model="openai.gpt-5.5",
                base_url="https://custom/v1",
                api_key_env="MY_KEY",
            ),
            lambda s: None,
        )
        kw = store["kwargs"]
        assert kw["base_url"] == "https://custom/v1"
        assert kw["api_key_env"] == "MY_KEY"

    def test_gemini_fallback_keeps_flags(self, monkeypatch):
        store: dict = {}
        self._patch_clients(monkeypatch, store)
        gen._build_text_client(
            _client_args(provider="gemini", model="gemini-1.5-pro"),
            lambda s: None,
        )
        assert store["name"] == "GeminiScenarioLLMClient"
        assert store["kwargs"]["base_url"] == gen._GEMINI_DEFAULT_BASE_URL


    def test_anthropic_uses_official_api_not_adc(self, monkeypatch):
        # --provider=anthropic reuses the Claude client class, but passing
        # api_key_env is what flips it off the Vertex/ADC path; project must NOT
        # be forwarded or the client would look like a misconfigured Vertex run.
        store: dict = {}
        self._patch_clients(monkeypatch, store)
        gen._build_text_client(
            _client_args(provider="anthropic", model="claude-sonnet-5"),
            lambda s: None,
        )
        assert store["name"] == "AnthropicScenarioLLMClient"
        kw = store["kwargs"]
        assert kw["model"] == "claude-sonnet-5"
        assert kw["api_key_env"] == gen._ANTHROPIC_DEFAULT_API_KEY_ENV
        assert "project" not in kw
        assert "location" not in kw
        assert kw["temperature"] == 0.2

    def test_anthropic_honours_custom_key_env(self, monkeypatch):
        store: dict = {}
        self._patch_clients(monkeypatch, store)
        gen._build_text_client(
            _client_args(
                provider="anthropic",
                model="claude-sonnet-5",
                api_key_env="MY_CLAUDE_KEY",
            ),
            lambda s: None,
        )
        assert store["kwargs"]["api_key_env"] == "MY_CLAUDE_KEY"

    def test_openai_defaults_are_swapped_in(self, monkeypatch):
        store: dict = {}
        self._patch_clients(monkeypatch, store)
        gen._build_text_client(
            _client_args(provider="openai", model="gpt-5.5"),
            lambda s: None,
        )
        assert store["name"] == "OpenAICompatibleScenarioLLMClient"
        kw = store["kwargs"]
        # Gemini defaults auto-swapped to OpenAI's; Responses API forced;
        # temperature omitted (gpt-5-class 400s on an explicit value).
        assert kw["base_url"] == gen._OPENAI_DEFAULT_BASE_URL
        assert kw["api_key_env"] == gen._OPENAI_DEFAULT_API_KEY_ENV
        assert kw["api"] == "responses"
        assert kw["temperature"] is None

    def test_openai_explicit_overrides_are_honoured(self, monkeypatch):
        store: dict = {}
        self._patch_clients(monkeypatch, store)
        gen._build_text_client(
            _client_args(
                provider="openai",
                model="gpt-5.5",
                base_url="https://proxy/v1",
                api_key_env="MY_OPENAI_KEY",
            ),
            lambda s: None,
        )
        kw = store["kwargs"]
        assert kw["base_url"] == "https://proxy/v1"
        assert kw["api_key_env"] == "MY_OPENAI_KEY"

    def test_official_and_cloud_gpt_paths_differ_only_in_endpoint(
        self, monkeypatch
    ):
        # The whole point of the openai provider: same client, same Responses
        # API, only the host+key change. If these ever diverge, one of the two
        # GPT paths has silently grown behaviour the other lacks.
        logger = lambda s: None  # one object, so it compares equal across calls
        official: dict = {}
        self._patch_clients(monkeypatch, official)
        gen._build_text_client(
            _client_args(provider="openai", model="gpt-5.5"), logger
        )
        cloud: dict = {}
        self._patch_clients(monkeypatch, cloud)
        gen._build_text_client(
            _client_args(provider="bedrock-gpt", model="gpt-5.5"), logger
        )
        assert official["name"] == cloud["name"]
        differing = {
            k
            for k in official["kwargs"]
            if official["kwargs"][k] != cloud["kwargs"][k]
        }
        assert differing == {"base_url", "api_key_env"}


class TestBedrockEndpointFromArgs:
    def test_defaults_swap_to_bedrock(self):
        base_url, api_key_env = gen._bedrock_endpoint_from_args(
            _client_args(provider="bedrock-gpt")
        )
        assert base_url == gen._BEDROCK_GPT_DEFAULT_BASE_URL
        assert api_key_env == gen._BEDROCK_GPT_DEFAULT_API_KEY_ENV

    def test_explicit_values_pass_through(self):
        base_url, api_key_env = gen._bedrock_endpoint_from_args(
            _client_args(
                provider="bedrock-gpt", base_url="https://x/v1", api_key_env="K"
            )
        )
        assert base_url == "https://x/v1"
        assert api_key_env == "K"


class TestOpenAIEndpointFromArgs:
    def test_defaults_swap_to_openai(self):
        base_url, api_key_env = gen._openai_endpoint_from_args(
            _client_args(provider="openai")
        )
        assert base_url == gen._OPENAI_DEFAULT_BASE_URL
        assert api_key_env == gen._OPENAI_DEFAULT_API_KEY_ENV

    def test_explicit_values_pass_through(self):
        base_url, api_key_env = gen._openai_endpoint_from_args(
            _client_args(
                provider="openai", base_url="https://y/v1", api_key_env="K2"
            )
        )
        assert base_url == "https://y/v1"
        assert api_key_env == "K2"

    def test_anthropic_key_env_swaps_only_the_key(self):
        assert (
            gen._anthropic_key_env_from_args(_client_args(provider="anthropic"))
            == gen._ANTHROPIC_DEFAULT_API_KEY_ENV
        )
        assert (
            gen._anthropic_key_env_from_args(
                _client_args(provider="anthropic", api_key_env="K3")
            )
            == "K3"
        )


class _StopBeforePipeline(Exception):
    """Sentinel: abort main() right after the judge-env sync so only env setup runs."""


class TestJudgeEnvPropagation:
    """The run's Vertex/Bedrock flags must reach the env the judge reads.

    Generation resolves --vertex_project/--vertex_location from the flag first,
    so without exporting them a judge invoked only via flags would have no
    VERTEX_PROJECT and fail; bedrock-gpt likewise needs the resolved endpoint in
    BEDROCK_OPENAI_BASE_URL so field_semantic_match hits the same deployment.
    """

    def _run_until_sync(self, monkeypatch, argv):
        import sys

        monkeypatch.setattr(
            sys, "argv", ["generate_conflict_vlm_scenarios.py"] + argv
        )
        import clasharness.evaluation.llm_judge as judge

        def _stop(*a, **k):
            raise _StopBeforePipeline()

        monkeypatch.setattr(judge, "sync_judge_env", _stop)
        with pytest.raises(_StopBeforePipeline):
            gen.main()

    def test_vertex_project_and_location_are_exported(self, monkeypatch):
        for var in ("VERTEX_PROJECT", "VERTEX_LOCATION"):
            monkeypatch.delenv(var, raising=False)
        self._run_until_sync(
            monkeypatch,
            [
                "--provider=vertex-anthropic",
                "--model=claude-sonnet-4@20250514",
                "--vertex_project=proj-x",
                "--vertex_location=us-east5",
            ],
        )
        assert os.environ["VERTEX_PROJECT"] == "proj-x"
        assert os.environ["VERTEX_LOCATION"] == "us-east5"

    def test_bedrock_base_url_is_exported_by_default(self, monkeypatch):
        monkeypatch.delenv("BEDROCK_OPENAI_BASE_URL", raising=False)
        self._run_until_sync(
            monkeypatch,
            ["--provider=bedrock-gpt", "--model=openai.gpt-5.5"],
        )
        assert (
            os.environ["BEDROCK_OPENAI_BASE_URL"]
            == gen._BEDROCK_GPT_DEFAULT_BASE_URL
        )

    def test_bedrock_custom_key_env_is_mirrored(self, monkeypatch):
        monkeypatch.delenv("BEDROCK_OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("MY_BEDROCK_KEY", "sk-secret")
        self._run_until_sync(
            monkeypatch,
            [
                "--provider=bedrock-gpt",
                "--model=openai.gpt-5.5",
                "--api_key_env=MY_BEDROCK_KEY",
            ],
        )
        # The judge reads BEDROCK_OPENAI_API_KEY; a custom --api_key_env's value
        # is mirrored there so generation and judge share one credential.
        assert os.environ["BEDROCK_OPENAI_API_KEY"] == "sk-secret"

    def test_non_bedrock_provider_does_not_set_bedrock_env(self, monkeypatch):
        monkeypatch.delenv("BEDROCK_OPENAI_BASE_URL", raising=False)
        self._run_until_sync(
            monkeypatch,
            ["--provider=gemini", "--model=gemini-1.5-pro"],
        )
        assert "BEDROCK_OPENAI_BASE_URL" not in os.environ

    def test_openai_base_url_is_exported_by_default(self, monkeypatch):
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        self._run_until_sync(
            monkeypatch, ["--provider=openai", "--model=gpt-5.5"]
        )
        assert os.environ["OPENAI_BASE_URL"] == gen._OPENAI_DEFAULT_BASE_URL

    def test_openai_custom_key_env_is_mirrored(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("MY_OPENAI_KEY", "sk-openai")
        self._run_until_sync(
            monkeypatch,
            [
                "--provider=openai",
                "--model=gpt-5.5",
                "--api_key_env=MY_OPENAI_KEY",
            ],
        )
        assert os.environ["OPENAI_API_KEY"] == "sk-openai"

    def test_anthropic_custom_key_env_is_mirrored(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("MY_CLAUDE_KEY", "sk-claude")
        self._run_until_sync(
            monkeypatch,
            [
                "--provider=anthropic",
                "--model=claude-sonnet-5",
                "--api_key_env=MY_CLAUDE_KEY",
            ],
        )
        assert os.environ["ANTHROPIC_API_KEY"] == "sk-claude"

    def test_official_providers_do_not_set_bedrock_env(self, monkeypatch):
        for provider, model in (
            ("openai", "gpt-5.5"),
            ("anthropic", "claude-sonnet-5"),
        ):
            monkeypatch.delenv("BEDROCK_OPENAI_BASE_URL", raising=False)
            self._run_until_sync(
                monkeypatch, [f"--provider={provider}", f"--model={model}"]
            )
            assert "BEDROCK_OPENAI_BASE_URL" not in os.environ
